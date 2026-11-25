"""Typed SDK proof for the six predeclared initialized swivel joints.

The original first-physics allowlist covers two active steering joints. The
initial-angle supplement additionally authors JointStateAPI on four passive
swivels. Their native physics writeback must be checked, not mistaken for an
external configuration edit. No controller, actuator or acceptance limit is
changed here; all additional fields must agree with the measured SDK state.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
import ast
import copy
import inspect
import math
from pathlib import Path
import textwrap
from types import FunctionType, SimpleNamespace

import numpy as np
import hospital_revision_serialization as serialization

runtime=serialization.runtime
initial=serialization.initial
campaign=serialization.campaign
JOINTS=(*initial.ACTIVE,*initial.CASTERS)
CELL='H2_B2_multimode_transition_cost_ramp_s01_joint_initial'
ROOT=initial.OUTPUT/'joint_writeback_requalification'
POLICY=initial.OUTPUT/'joint_writeback_resolution_policy.json'
QUEUE=_ijcas_path(ROOT, 'execution')


def exact_joint_radians(degrees,sdk_radians,type_name):
    """Prove native float radians->degrees encoding; do not loosen residuals."""
    degrees=float(degrees);sdk_radians=float(sdk_radians)
    if not math.isfinite(degrees) or not math.isfinite(sdk_radians):
        raise ValueError('Finite scalar joint state and SDK value required')
    rounded=float(np.float32(math.degrees(sdk_radians)))
    native=np.float32(sdk_radians)
    native_degrees=float(native*np.float32(180./math.pi))
    if str(type_name)=='float' and (degrees==rounded or
            float(native)==sdk_radians and degrees==native_degrees):
        return sdk_radians
    return math.radians(degrees)


def source_state_types(stage,actor,link_names,joint_names):
    result=runtime.retained.prepared_source_state_types(stage,actor,link_names,joint_names)
    if len(result)!=56 or not all(name in joint_names for name in JOINTS):
        raise ValueError('Original 56 state fields and all six initialized joints required')
    for name in initial.CASTERS:
        for field in ('position','velocity'):
            path=actor+'/joints/'+name+'.state:angular:physics:'+field
            attr=stage.GetAttributeAtPath(path)
            if not attr or str(attr.GetTypeName())!='float' or attr.GetNumTimeSamples():
                raise ValueError('Exactly typed, untimed initialized swivel state required: '+path)
            result[path]='float'
    if len(result)!=64:raise ValueError('Exactly eight additional passive-swivel state fields required')
    return result


def first_writeback_binding():
    original=runtime.retained.prepared_source_writeback_review
    before=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=copy.deepcopy(before)
    expected=ast.dump(ast.parse(repr(tuple(initial.ACTIVE)),mode='eval').body)
    loops=[node for node in ast.walk(tree) if isinstance(node,ast.For)
        and isinstance(node.target,ast.Name) and node.target.id=='name'
        and ast.dump(node.iter)==expected]
    angular=[node for node in ast.walk(tree) if isinstance(node,ast.If)
        and ast.unparse(node.test)=="field == 'physics:angularVelocity'"
        and len(node.body)==1 and ast.unparse(node.body[0])=='value = np.radians(value)']
    scalar=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n)=='math.radians(actual)']
    if len(loops)!=1 or len(angular)!=1 or len(scalar)!=1:
        raise ValueError('Exactly the original two-joint loop and angular unit conversion required')
    loops[0].iter=ast.parse(repr(JOINTS),mode='eval').body
    angular[0].body=ast.parse(
        'value = _exact_degree_value(value, velocities[i,part], prim.GetAttribute(field).GetTypeName())').body
    scalar[0].func=ast.Name(id='_exact_joint_radians',ctx=ast.Load())
    scalar[0].args=ast.parse("(actual,value,prim.GetAttribute('state:angular:physics:'+field).GetTypeName())",mode='eval').body.elts
    undo=copy.deepcopy(tree)
    changed_loop=next(n for n in ast.walk(undo) if isinstance(n,ast.For)
        and isinstance(n.target,ast.Name) and n.target.id=='name'
        and ast.dump(n.iter)==ast.dump(ast.parse(repr(JOINTS),mode='eval').body))
    changed_loop.iter=ast.parse(repr(tuple(initial.ACTIVE)),mode='eval').body
    changed_angular=next(n for n in ast.walk(undo) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="field == 'physics:angularVelocity'")
    changed_angular.body=ast.parse('value = np.radians(value)').body
    changed_scalar=next(n for n in ast.walk(undo) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='_exact_joint_radians')
    changed_scalar.func=ast.parse('math.radians',mode='eval').body
    changed_scalar.args=[ast.Name(id='actual',ctx=ast.Load())]
    if ast.dump(undo)!=ast.dump(before):
        raise ValueError('All other metadata, canonical-layer, SDK and numerical checks must remain identical')
    namespace=dict(original.__globals__,prepared_source_state_types=source_state_types,
        _exact_degree_value=serialization.exact_degree_value,_exact_joint_radians=exact_joint_radians)
    exec(compile(ast.fix_missing_locations(tree),'<six_initialized_joint_typed_writeback>','exec'),namespace)
    checked=namespace[original.__name__]
    def review(*args,**kwargs):
        result=checked(*args,**kwargs)
        result.update(initialized_joint_names=list(JOINTS),exact_state_field_count=64,
            additional_state_fields_compared_to_actual_SDK=8,
            nonstate_actor_subtree_metadata_or_physical_limit_waived=False,
            all_original_numerical_residual_limits_unchanged=True,
            all_other_original_review_AST_statements_identical=True)
        return result
    review.checked_original=checked
    return review


FIRST_REVIEW=first_writeback_binding()


def recorded_writeback_binding():
    original=runtime.retained.recorded_physics_writeback_review
    before=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=copy.deepcopy(before)
    expected=ast.dump(ast.parse(repr(tuple(initial.ACTIVE)),mode='eval').body)
    found=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id=='steering' and ast.dump(n.value)==expected]
    if len(found)!=1:raise ValueError('One original recorded-state two-joint tuple required')
    found[0].value=ast.parse(repr(JOINTS),mode='eval').body
    scalar=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and ast.unparse(n)=='math.radians(float(spec.default))']
    if len(scalar)!=1:raise ValueError('One original scalar joint degree conversion required')
    scalar[0].func=ast.Name(id='_exact_joint_radians',ctx=ast.Load())
    scalar[0].args=ast.parse('(float(spec.default),value,spec.typeName)',mode='eval').body.elts
    undo=copy.deepcopy(tree)
    assignment=next(n for n in ast.walk(undo) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='steering')
    assignment.value=ast.parse(repr(tuple(initial.ACTIVE)),mode='eval').body
    changed_scalar=next(n for n in ast.walk(undo) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='_exact_joint_radians')
    changed_scalar.func=ast.parse('math.radians',mode='eval').body
    changed_scalar.args=ast.parse('(float(spec.default),)',mode='eval').body.elts
    if ast.dump(undo)!=ast.dump(before):
        raise ValueError('Recorded state proof changed outside the six measured joint names')
    namespace=dict(original.__globals__,_exact_joint_radians=exact_joint_radians)
    exec(compile(ast.fix_missing_locations(tree),'<six_joint_recorded_state_proof>','exec'),namespace)
    checked=namespace[original.__name__]
    def review(*args,**kwargs):
        result=checked(*args,**kwargs)
        result.update(all_six_initialized_joint_positions_and_velocities_matched_to_SDK=True,
            initialized_joint_names=list(JOINTS),all_other_original_review_AST_statements_identical=True,
            physical_and_numerical_limits_unchanged=True)
        return result
    review.checked_original=checked
    return review


RECORDED_REVIEW=recorded_writeback_binding()


def route_subscriber():
    original=runtime.lc.RetainedMissionSDKHost.subscribe
    before=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=copy.deepcopy(before)
    expected=ast.dump(ast.parse('motion.RetainedTensorDriveAdapter.STEERING',mode='eval').body)
    found=[n for n in ast.walk(tree) if isinstance(n,ast.comprehension)
        and isinstance(n.target,ast.Name) and n.target.id=='name' and ast.dump(n.iter)==expected]
    if len(found)!=1:raise ValueError('One exact original route-state joint allowlist required')
    found[0].iter=ast.Name(id='_INITIALIZED_JOINTS',ctx=ast.Load())
    undo=copy.deepcopy(tree)
    changed=next(n for n in ast.walk(undo) if isinstance(n,ast.comprehension)
        and isinstance(n.iter,ast.Name) and n.iter.id=='_INITIALIZED_JOINTS')
    changed.iter=ast.parse('motion.RetainedTensorDriveAdapter.STEERING',mode='eval').body
    if ast.dump(undo)!=ast.dump(before):
        raise ValueError('Route callback/metadata/resync checks must remain identical')
    wrapper=ast.ClassDef(name='_JointNoticeBinding',bases=[],keywords=[],body=tree.body,decorator_list=[])
    namespace=dict(original.__globals__,_INITIALIZED_JOINTS=JOINTS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
        '<initialized_joint_route_notices>','exec'),namespace)
    candidate=namespace['_JointNoticeBinding'].__dict__['subscribe']
    if candidate.__code__.co_freevars!=original.__code__.co_freevars:
        raise ValueError('Original route super-class closure must be retained')
    return FunctionType(candidate.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)


class InitializedJointRouteHost(serialization.placement.live.PrimaryRouteHost):
    subscribe=route_subscriber()

    def check_post_observation_layers(self):
        return runtime.bind(runtime.retained.RetainedZeroSettleSDKPort.check_post_observation_layers,
            recorded_physics_writeback_review=RECORDED_REVIEW)(self)


def retained_proxy(context):
    return SimpleNamespace(**dict(vars(runtime.retained),
        shared=context.initial_condition_source_helpers,
        prepared_source_state_types=source_state_types,
        prepared_source_writeback_review=FIRST_REVIEW))


class InitializedJointStaticHost(initial.InitialConditionStaticHost):
    async def preflight(self):
        return await runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.preflight,
            retained=retained_proxy(self.context))(self)

    def guard(self,row,info):
        return runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.guard,
            retained=retained_proxy(self.context))(self,row,info)

    def check_post_observation_layers(self):
        return runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.check_post_observation_layers,
            retained=retained_proxy(self.context))(self)


def cell_bindings(cell):
    bound=runtime.bind(initial.cell_bindings,
        private_source_context=serialization.follow.initial_context,
        InitialConditionStaticHost=InitializedJointStaticHost,__file__=__file__)(cell)
    return runtime.bind(bound,PrimaryRouteHost=InitializedJointRouteHost,__file__=__file__)


def qualified_first_cell_binding(cell):
    bound=cell_bindings(cell)
    if cell['cell_id']!=CELL:return bound
    original_runtime=bound.__globals__['runtime']
    def context_for(folder,actor,geometry):
        return original_runtime.context_for(folder,actor+'_joint_setup02',geometry)
    scoped=SimpleNamespace(**dict(vars(original_runtime),context_for=context_for))
    output=SimpleNamespace(**dict(vars(bound.__globals__['campaign']),OUTPUT=ROOT))
    return runtime.bind(bound,runtime=scoped,campaign=output,__file__=__file__)


async def prove_current_boundary():
    """GET only: no fabricated pre-Play tensor, no static or drive admission."""
    old=initial.OUTPUT/'live'/CELL
    raw=runtime.read(old/'static/held_static_readback.json')
    if (raw.get('sample_count')!=0 or raw.get('new_route_trials')!=0
            or not raw['checks'].get('no_motor_write_restoration_required')
            or not raw['checks'].get('actual_Pause_confirmed')
            or not raw.get('error','').startswith('RuntimeError: First physics initialization monitor error:')):
        raise ValueError('Only the diagnosed zero-command, pre-binding setup abort may be reviewed')
    outcome=runtime.read(old/'trial_outcome.json')
    if outcome.get('actual_route_started') or outcome['status']!='STATIC_NOT_PASS_NO_ROUTE':
        raise ValueError('A started method route cannot be replaced')
    sdk=runtime.retained.runtime.live_sdk()
    stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    state=lambda:runtime.evidence.timeline_state(timeline)
    saved=raw['actual_final_timeline']
    if state()!=saved or saved['playing'] or saved['stopped']:
        raise ValueError('Exact retained actual PAUSED boundary required; no timeline restoration')
    texts=lambda:dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    idle=runtime.retained.split_viewport_renderer_defaults(stage,
        raw['actual_final_readback']['layer_texts'],texts())
    if idle['normalized_after']!=raw['actual_final_readback']['layer_texts']:
        raise ValueError('Exact saved post-abort layers required for this GET proof')
    packet=runtime.read(raw['protocol']);prep=runtime.read(raw['preparation_record'])
    runtime.retained.runtime.task.verify_pins(packet['input_sha256'])
    cell=next(c for c in runtime.read(initial.PROTOCOL)['main_cells'] if c['cell_id']==CELL)
    source=initial.OUTPUT/'source_conditions'/(cell['paired_condition_id']+'.usda')
    receipt=runtime.read(source.with_suffix('.json'))
    if runtime.sha(source)!=receipt['sha256'] or receipt['initial_angles_deg']!=cell['initial_joint_angles_deg']:
        raise ValueError('Immutable predeclared source-angle overlay required')
    # These added attributes existed in the weak source BEFORE the first Play.
    source_stage=sdk.Usd.Stage.Open(str(source))
    types=dict(raw['preflight']['prePlay_source_state_types'])
    for name in initial.CASTERS:
        for field in ('position','velocity'):
            attr=source_stage.GetAttributeAtPath(initial.runtime.shared.preparation.old.constructor.OLD_ROBOT+
                '/joints/'+name+'.state:angular:physics:'+field)
            if not attr or str(attr.GetTypeName())!='float' or attr.GetNumTimeSamples():
                raise ValueError('Passive state must be authored in the immutable pre-Play source')
            types[packet['actor']+'/joints/'+name+'.state:angular:physics:'+field]='float'
    view=sdk.SimulationManager.get_physics_sim_view().create_articulation_view(packet['actor']+'/base_link')
    snapshot=runtime.shared.preparation.old.tensor_snapshot(view)
    baseline=prep['expected_nominal_motor_baseline']
    if (snapshot['names']!=baseline['joint_names'] or len(snapshot['link_names'])!=13
            or any(snapshot['gains'][k]!=baseline[k] for k in ('kp','kd','max_efforts'))
            or snapshot['joint_position_targets_rad']!=[0.]*12
            or snapshot['joint_velocity_targets_rad_s']!=[0.]*12):
        raise ValueError('All original gains, effort limits and zero targets required')
    proof=FIRST_REVIEW(stage,runtime.read(old/'prepared_layer_texts.json'),texts(),snapshot,packet['actor'],types)
    before=texts()
    def stable():
        if sdk.omni.usd.get_context().get_stage()!=stage or state()!=saved or texts()!=before:
            raise ValueError('GET-only stage/layers/Pause boundary changed')
    expected=runtime.read(runtime.retained.H6/'actual_common_environment_rows.json')
    await runtime.bootstrap.verify_rows(stage,expected['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async,unchanged=stable)
    stable()
    if runtime.shared.preparation.old.tensor_snapshot(view)!=snapshot:
        raise ValueError('Actual SDK state changed during GET-only proof')
    observed={n:math.degrees(snapshot['positions'][snapshot['names'].index(n)]) for n in JOINTS}
    delta={n:math.degrees(math.atan2(math.sin(math.radians(observed[n]-v)),
        math.cos(math.radians(observed[n]-v)))) for n,v in cell['initial_joint_angles_deg'].items()}
    value=dict(status='GET_ONLY_INITIALIZED_JOINT_WRITEBACK_BOUNDARY_NOT_STATIC_OR_ROUTE_ADMISSION',
        actual_final_timeline=saved,actual_final_readback=dict(tensor=snapshot),
        checks=dict(original_gains_targets_restored=True),
        original_static_HOLD_NOT_changed=True,original_static=str(old/'static/held_static_readback.json'),
        actual_SDK_snapshot_measured_NOW_not_fabricated_prePlay=True,
        original_unbound_static_report_NOT_promoted_to_a_bound_observation=True,
        initialized_joint_writeback_proof=proof,all4997_protected_rows_verified=True,
        prescribed_angles_deg=cell['initial_joint_angles_deg'],GET_angles_deg=observed,
        wrapped_angle_difference_deg=delta,post_abort_renderer_only_differences=idle['exact_renderer_default_differences'],
        no_stage_timeline_gain_or_target_write=True)
    path=_ijcas_path(ROOT, 'current_boundary_GET.json')
    runtime.evidence.write_exclusive(path,value)
    return path


def export_policy():
    old=initial.OUTPUT/'live'/CELL
    original=runtime.read(old/'trial_outcome.json')
    if original.get('actual_route_started') or original['status']!='STATIC_NOT_PASS_NO_ROUTE':
        raise ValueError('Only the diagnosed setup incident, never a started route, may be requalified')
    value=dict(version='SIX_DECLARED_INITIALIZED_JOINT_STATES_TYPED_SDK_PROOF_v1',
        cell_id=CELL,source_static=str(old/'static/held_static_readback.json'),
        source_static_sha256=runtime.sha(old/'static/held_static_readback.json'),
        original_static_HOLD_preserved=True,original_route_attempts=0,
        original_motor_writes=0,maximum_additional_setup_statics=1,maximum_FIRST_method_routes=1,
        qualified_output=str(_ijcas_path(ROOT, 'live', CELL)),queue=str(QUEUE),
        initialized_joint_names=list(JOINTS),extra_measured_state_fields=8,
        original_controller_safety_success_limits_speed_clocks_and_geometry_unchanged=True,
        no_retry_or_replacement_of_any_started_method_route=True,
        all_remaining_fifty_nine_predeclared_cells_unchanged=True,
        original_protocol=str(initial.PROTOCOL),original_protocol_sha256=runtime.sha(initial.PROTOCOL),
        source_sha256=runtime.sha(__file__),authority=runtime.APPROVAL)
    if POLICY.exists():
        if runtime.read(POLICY)!=value:raise ValueError('Preserve the prospective setup-only resolution policy')
    else:runtime.evidence.write_exclusive(POLICY,value)
    return value


async def run_remaining_once(*,authorized=False):
    if authorized is not True:raise ValueError('Current reviewer-experiment authority required')
    policy=export_policy()
    boundary=_ijcas_path(ROOT, 'current_boundary_GET.json');get=runtime.read(boundary)
    if not serialization.execution.restored_boundary(get) or not get['all4997_protected_rows_verified']:
        raise ValueError('Own exact GET-proven retained boundary required')
    if runtime.sha(policy['source_static'])!=policy['source_static_sha256']:
        raise ValueError('Original failed setup bytes changed')
    runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'resume_reservation.json'),dict(
        original_execution_stop_retained=str(initial.OUTPUT/'execution/stopped.json'),
        policy=str(POLICY),policy_sha256=runtime.sha(POLICY),prior_GET=str(boundary),
        first_cell_additional_setup_only=True,remaining_cells=59,no_automatic_method_route_retry=True))
    files=(__file__,initial.__file__,serialization.__file__,serialization.placement.__file__,
        serialization.follow.__file__,serialization.placement.live.__file__,serialization.execution.__file__)
    runtime.lc.freeze_workflow_sources(_ijcas_path(ROOT, 'continuation_sources.zip'),
        {str(Path(p).resolve()):runtime.sha(p) for p in files})
    try:
        # Reuse the complete original finite supplement loop. Only the queue
        # output and typed initialization adapters are privately substituted.
        run=runtime.bind(initial.run,OUTPUT=ROOT,cell_bindings=qualified_first_cell_binding,__file__=__file__)
        result=await run(prior_result=boundary,authorized=True)
        runtime.evidence.write_exclusive(initial.OUTPUT/'after_primary_result.json',result)
        runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'all_remaining_completed.json'),dict(
            status='FINITE_INITIAL_CONDITION_SUPPLEMENT_FINISHED',result=result,
            old_setup_HOLD_NOT_promoted=True,all_main_and_ablation_results_unchanged=True))
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'continuation_stopped.json'),dict(
            error=type(error).__name__+': '+str(error),no_automatic_retry=True))
        raise


def dispatch_get():
    from hospital_revision_runner import remote
    path=Path(__file__).resolve()
    command=("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\n"
        "assert old is None or old.done(), 'Another operation is active'\n"
        f"p=Path({str(path)!r})\n"
        "m=types.ModuleType('hospital_revision_joint_writeback');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.prove_current_boundary())\n"
        "builtins.ijcas_joint_writeback_GET=job\n"
        "builtins.ijcas_hospital_latest_task_experiment=job\n"
        "print({'scheduled_GET_only':True,'Play_Stop_or_motor_write':False})\n")
    runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'GET_dispatch.json'),dict(command=command,
        entry_source=str(path),entry_source_sha256=runtime.sha(path),no_physical_trial=True))
    return remote(command)


def dispatch_remaining():
    from hospital_revision_runner import remote
    export_policy()
    path=Path(__file__).resolve()
    command=("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\n"
        "assert old is None or old.done(), 'Another operation is active'\n"
        f"p=Path({str(path)!r})\n"
        "m=types.ModuleType('hospital_revision_joint_writeback');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.run_remaining_once(authorized=True))\n"
        "builtins.ijcas_hospital_latest_task_experiment=job\n"
        "builtins.ijcas_revision_primary_task=job\n"
        "print({'scheduled_initial_condition_continuation':True,'maximum_remaining_conditions':60,"
        "'original_setup_preserved':True,'method_route_retry':False})\n")
    runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'continuation_dispatch.json'),dict(command=command,
        entry_source=str(path),entry_source_sha256=runtime.sha(path),policy=str(POLICY)))
    return remote(command)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--get',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.get and args.resume:raise ValueError('Separate GET and new physical continuation commands required')
    print(dispatch_get() if args.get else dispatch_remaining() if args.resume else 'No action requested')
