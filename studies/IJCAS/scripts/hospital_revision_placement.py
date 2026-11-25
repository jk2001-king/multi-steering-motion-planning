"""Wrapped-yaw placement check and continuation of an unstarted preparation.

No physical threshold, model, controller, source asset or recorded outcome is
changed. The legacy validator is otherwise retained statement for statement.
"""
import ast
import copy
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace

import hospital_revision_campaign as campaign
import hospital_revision_live as live

runtime = campaign.runtime
CELL = 'H2_P_dynamics_full_envelope_s03'
CONTINUATION = campaign.OUTPUT/'execution_queue_v3_wrapped_placement'
REVIEW_VERSION = 'wrapped_pose_retired_stop_fields_v2'


def pose_error(actual, expected):
    if len(actual) != 3 or len(expected) != 3 or not all(
            math.isfinite(float(v)) for v in (*actual, *expected)):
        raise ValueError('Finite three-component placement poses required')
    delta = actual[2]-expected[2]
    return max(abs(actual[0]-expected[0]), abs(actual[1]-expected[1]),
               abs(math.atan2(math.sin(delta), math.cos(delta))))


def wrapped_validator(original):
    source = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree = copy.deepcopy(source)
    needle = ast.parse('max(abs(a-b) for a, b in zip(actual, start))', mode='eval').body
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and ast.dump(node.left) == ast.dump(needle):
            if len(node.comparators) != 1 or not isinstance(node.comparators[0], ast.Constant) or node.comparators[0].value != 2e-5:
                raise ValueError('Original placement tolerance must remain 2e-5')
            found.append(node)
    if len(found) != 1:
        raise ValueError('One original pose-difference expression required')
    found[0].left = ast.Call(func=ast.Name(id='_wrapped_pose_error', ctx=ast.Load()),
                            args=[ast.Name(id='actual',ctx=ast.Load()), ast.Name(id='start',ctx=ast.Load())], keywords=[])
    undo = copy.deepcopy(tree)
    next(n for n in ast.walk(undo) if isinstance(n, ast.Compare)
         and isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Name)
         and n.left.func.id == '_wrapped_pose_error').left = needle
    if ast.dump(undo) != ast.dump(source):
        raise ValueError('Every other source/geometry/mass/drive check must remain identical')
    scope = dict(original.__globals__, _wrapped_pose_error=pose_error)
    exec(compile(ast.fix_missing_locations(tree), '<same_placement_tolerance_wrapped_yaw>', 'exec'), scope)
    return scope[original.__name__]


def source_context(context):
    """Private namespace, including when an initial-joint overlay is in use."""
    p = context.preparation
    old = SimpleNamespace(**vars(p.old))
    source = SimpleNamespace(**vars(old.constructor))
    source.validate_new_robot = wrapped_validator(source.validate_new_robot)
    source.author_new_robot = runtime.bind(source.author_new_robot, _wrapped_pose_error=pose_error)
    old.constructor = source
    old.constructor_for = runtime.bind(p.old.constructor_for, constructor=source)
    p.old = old
    helpers = SimpleNamespace(**dict(vars(runtime.shared), preparation=p))
    helpers.fresh_source_reference = runtime.bind(runtime.shared.fresh_source_reference, preparation=p)
    helpers.fresh_source_for_live_actor = runtime.bind(runtime.shared.fresh_source_for_live_actor,
        fresh_source_reference=helpers.fresh_source_reference)
    context.placement_source_helpers = helpers
    return context


def context_for(folder, actor, geometry):
    return source_context(runtime.context_for(folder, actor, geometry))


async def prepare_source(context, sdk, geometry, *, prior_result=None):
    return await runtime.bind(runtime.prepare_source, shared=context.placement_source_helpers)(
        context, sdk, geometry, prior_result=prior_result)


class PlacementStaticHost(runtime.ControlledStaticHost):
    async def preflight(self):
        proxy = SimpleNamespace(**dict(vars(runtime.retained), shared=self.context.placement_source_helpers))
        return await runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.preflight, retained=proxy)(self)


def private_runtime():
    return SimpleNamespace(**dict(vars(runtime), context_for=context_for,
        prepare_source=prepare_source, ControlledStaticHost=PlacementStaticHost))


def bound_run_cell():
    return runtime.bind(live.run_cell, runtime=private_runtime(), __file__=__file__)


def stopped_root_cleanup(before, after, old_actor):
    """Allow only removal of runtime poses/velocities on the retired actor."""
    fields = runtime.shared.preparation.old.fields
    a, b = fields.layer_snapshot(before), fields.layer_snapshot(after)
    allowed = {'physics:velocity', 'physics:angularVelocity', 'xformOp:orient', 'xformOp:translate'}
    changed = []
    for path in sorted(set(a)|set(b)):
        if a.get(path) == b.get(path):
            continue
        joint_states = {
            old_actor+'/joints/'+joint+'.state:angular:physics:'+quantity
            for joint in ('base_link_to_front_active_steering','base_link_to_rear_active_steering')
            for quantity in ('position','velocity')}
        if path in joint_states and path in a and path in b:
            expected=copy.deepcopy(a[path])
            expected['info']['default']=0.
            if b[path]!=expected:
                raise ValueError('Retired steering Stop cleanup must be default-to-zero only: '+path)
            changed.append(path)
            continue
        if (not path.startswith(old_actor+'/') or path.rsplit('.',1)[-1] not in allowed
                or path not in a or path in b):
            raise ValueError('Unreviewed Root change beyond retired runtime defaults: '+path)
        changed.append(path)
    return changed


async def finish_unstarted_preparation(context, sdk, geometry):
    """Same already-created actor; no constructor, Stop, reset or re-placement."""
    from pxr import Usd, UsdPhysics
    import carb.settings
    from omni.kit.viewport.utility import get_active_viewport
    import hospital_h4_visual_instance_refresh as display
    output = context.output
    if (output/'preparation.json').exists() or list(output.glob('static/*')) or list(output.glob('route/*')):
        raise ValueError('Only the still-unstarted staging boundary may continue')
    if runtime.read(output/'infrastructure_error.json')['error'] != 'ValueError: Fresh H4 placement differs from fixed geometry-derived start':
        raise ValueError('Exact diagnosed placement error required')
    prior = runtime.read(output/'prior_boundary.json')
    stage = sdk.omni.usd.get_context().get_stage()
    timeline = sdk.omni.timeline.get_timeline_interface()
    state = lambda: runtime.evidence.timeline_state(timeline)
    stopped = dict(playing=False, stopped=True, current_time_s=0.)
    def stable():
        if sdk.omni.usd.get_context().get_stage() != stage or state() != stopped:
            raise ValueError('Same untouched STOPPED source required')
    stable()
    if stage.GetPrimAtPath(context.cohort.MARKER):
        raise ValueError('First-display preparation has already been consumed')
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 13 or any(not p.startswith(context.cohort.ROBOT+'/') for p in bodies):
        raise ValueError('Only the same new thirteen-body actor may be active')
    if stage.GetPrimAtPath(prior['actor']).IsActive():
        raise ValueError('Prior archived actor must remain inactive')
    cleanup = stopped_root_cleanup(prior['Root_text'], stage.GetRootLayer().ExportToString(), prior['actor'])
    expected = runtime.read(runtime.retained.H6/'actual_common_environment_rows.json')
    await runtime.bootstrap.verify_rows(stage, expected['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async, unchanged=stable)
    helpers = context.placement_source_helpers
    reference = helpers.fresh_source_reference(context)
    constructor = context.preparation.old.constructor_for(context.cohort.ROBOT).__globals__['validate_new_robot']
    source = Usd.Stage.Open(str(context.preparation.ASSET))
    made = constructor(stage, source, geometry['start_xyyaw'])
    if made != reference['source_constructor_readback'] or context.stationary.actor_configuration(stage) != reference['fixed_configuration_rows']:
        raise ValueError('Complete same-SDK constructor, 145 settings and seed equality required')
    runtime.evidence.write_exclusive(output/'placement_continuation_preflight.json', dict(
        preserved_error=str(output/'infrastructure_error.json'),
        actual_pose=made['start_xyyaw'], expected_pose=geometry['start_xyyaw'],
        wrapped_pose_error=pose_error(made['start_xyyaw'],geometry['start_xyyaw']),
        unchanged_pose_tolerance=2e-5, stopped_root_runtime_cleanup_paths=cleanup,
        same_new_actor=True, constructor_or_replacement_calls=0, additional_Stop_calls=0,
        no_static_or_route_started=True, source=str(Path(__file__).resolve()),source_sha256=runtime.sha(__file__)))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        marker = stage.DefinePrim(context.cohort.MARKER, 'Xform')
        marker.SetCustomData(dict(status='fresh_controlled_static_pending_NO_PLAY',
            robot_root=context.cohort.ROBOT, cohort=output.name))
    def physical():
        fixed = context.stationary.actor_configuration(stage)
        return dict(nonvisual_fixed_configuration_rows={p:v for p,v in fixed.items()
            if 'visuals' not in p[len(context.cohort.ROBOT):].split('/')},
            constructor=constructor(stage, source, geometry['start_xyyaw']))
    registered = await display.initialize_stopped_actor(stage, context.cohort.ROBOT,
        state=state, next_frame=sdk.omni.kit.app.get_app().next_update_async,
        physical_readback=physical, viewport=get_active_viewport(), settings=carb.settings.get_settings())
    runtime.evidence.write_exclusive(output/'display_initialization/renderer_initialization.json', registered)
    prep = dict(new_robot_root=context.cohort.ROBOT,start_xyyaw=geometry['start_xyyaw'],
        simulation_mass_kg=265.,live_new_robot_readback=made,source_constructor_readback=made,
        protected_environment_sha256=expected['sha256'],source_reference=reference,
        expected_nominal_motor_baseline=runtime.read(runtime.retained.H6/'preparation.json')['expected_nominal_motor_baseline'],
        timeline=stopped,Play_or_static_or_drive_executed=False,Stop_calls=1,
        prior_boundary=str(output/'prior_boundary.json'),empty_inactive_legacy_names_are_NOT_restored_actors=True,
        completion_of_same_first_staging_NOT_physical_retry=True)
    prep['critical_environment'] = context.stationary.critical_environment(stage,prep)
    layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    prep.update({k+'_sha256':hashlib.sha256(v.encode()).hexdigest() for k,v in layers.items()})
    runtime.evidence.write_exclusive(context.preparation.LIVE,prep)
    runtime.evidence.write_exclusive(output/'prepared_layer_texts.json',layers)


def remaining_original_cell():
    """Extract unchanged post-preparation statements, not a new controller."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(live.run_cell)))
    blocks=[n for n in tree.body[0].body if isinstance(n,ast.Try)]
    if len(blocks)!=1 or ast.unparse(blocks[0].body[0])!='await runtime.prepare_source(context, sdk, geometry, prior_result=prior_result)':
        raise ValueError('One original preparation boundary required')
    names='context sdk geometry geometry_path grid_path output cell_id label cell protocol original_plan model_policy_path'.split()
    function=ast.AsyncFunctionDef(name='_remaining_original_cell',
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in names],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=copy.deepcopy(blocks[0].body[1:]),decorator_list=[])
    scope=dict(live.run_cell.__globals__,runtime=private_runtime(),__file__=__file__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 '<unchanged_original_static_and_route_statements>','exec'),scope)
    return scope[function.name]


async def resume_and_run(*,authorized=False):
    if authorized is not True:
        raise ValueError('Current reviewer-campaign authority required')
    import hospital_revision_execution as execution
    protocol=runtime.read(campaign.PROTOCOL)
    cell=next(c for c in protocol['main_cells'] if c['cell_id']==CELL)
    output=campaign.OUTPUT/'live'/CELL
    audit=output/REVIEW_VERSION
    attempt=audit/'reservation.json'
    runtime.evidence.write_exclusive(attempt,dict(one_unstarted_phase_continuation=True,
        physical_retry=False,source_sha256=runtime.sha(__file__),authority=runtime.APPROVAL))
    plan_folder=live.planning.OUTPUT/CELL
    if execution.eligibility(cell,False)[0]!='ELIGIBLE_FOR_OWN_FRESH_STATIC':
        raise ValueError('Existing frozen plan/audit/model evidence must remain valid')
    runtime.lc.freeze_workflow_sources(audit/'sources.zip',
        {str(Path(p).resolve()):runtime.sha(p) for p in (__file__,live.__file__,execution.__file__)})
    geometry_path=plan_folder/'task_geometry.json';grid_path=plan_folder/'planning_occupancy.npz'
    geometry=runtime.read(geometry_path);original_plan=Path(runtime.read(plan_folder/'summary.json')['plan'])
    actor='/World/patient_transport_revision_'+CELL.replace('_dynamics_full_envelope','').lower()
    context=context_for(output,actor,geometry);sdk=runtime.retained.runtime.live_sdk()
    try:
        await finish_unstarted_preparation(context,sdk,geometry)
        outcome=await remaining_original_cell()(context,sdk,geometry,geometry_path,grid_path,output,
            CELL,CELL,cell,protocol,original_plan,execution.POLICY)
        report=Path(outcome['report'])
        if not execution.restored_boundary(runtime.read(report)):
            raise ValueError('Actual Pause and complete restoration required before next cell')
        if outcome.get('actual_route_started'):
            runtime.evidence.write_exclusive(report.parent/'controlled_actual_metrics.json',execution.metrics.metrics(report))
        print('[REVIEW CONTINUATION RESULT]',outcome,flush=True)
        proxy=SimpleNamespace(**dict(vars(live),run_cell=bound_run_cell()))
        run=runtime.bind(execution.run,live=proxy,QUEUE=CONTINUATION)
        result=await run(prior_result=report,authorized=True)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(audit/'error.json',dict(
            error=type(error).__name__+': '+str(error),no_automatic_retry=True))
        raise


def dispatch():
    from hospital_revision_runner import remote
    entry=Path(__file__).resolve()
    command=("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\n"
        "assert old is None or old.done(),'Another experiment is active'\n"
        f"p=Path({str(entry)!r})\n"
        "m=types.ModuleType('hospital_revision_placement');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.resume_and_run(authorized=True))\n"
        "builtins.ijcas_hospital_latest_task_experiment=job\nbuiltins.ijcas_revision_primary_task=job\n"
        "print({'same_unstarted_preparation_scheduled':True,'additional_Stop_for_this_actor':False,'no_physical_retry':True})\n")
    runtime.evidence.write_exclusive(campaign.OUTPUT/(REVIEW_VERSION+'_dispatch.json'),dict(
        command=command,source_sha256=runtime.sha(entry)))
    return remote(command)


if __name__=='__main__':
    print(dispatch())
