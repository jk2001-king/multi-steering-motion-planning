"""Fresh-app controlled comparison, using the existing guarded SDK hosts.

No SimulationApp, implicit import-time dispatch, old approval reuse, or blind
retry. Fresh construction is not historical physical-state restoration.
The first H2 pair is engineering only and excluded from final repetitions.
"""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time
from types import FunctionType, SimpleNamespace

import numpy as np
import hospital_controlled_comparison as comparison
import hospital_mission_sequence as sequence
import hospital_session_bootstrap as bootstrap
import hospital_runtime_evidence_io as evidence

lc = sequence.lifecycle
retained = sequence.retained
shared = retained.shared
ROOT, GATE = comparison.ROOT, comparison.GATE
BASE = GATE/'controlled_comparison_20261120/H2/live_engineering_pair_r01'
APPROVAL = dict(approval_source='USER_MESSAGE',
    user_message='1위의 피드ㅐㄱ 모 만족하기위한 실험 다 진행해서 결과 뽑아야해 진행하',
    scope='ADDITIONAL_REVIEW_EXPERIMENTS_FRESH_STATIC_THEN_CONDITIONAL_CONTROLLED_ROUTE_NO_BLIND_RETRY')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(function, **bindings):
    result = FunctionType(function.__code__, dict(function.__globals__, **bindings),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def checked_approval(value):
    if value != APPROVAL:
        raise ValueError('Exact current additional-experiment user authority required')
    return copy.deepcopy(value)


def controlled_contract(packet):
    return bind(sequence.checked_route_lifecycle_contract,
                REMAINING_NO_NAV2_APPROVAL=APPROVAL)(packet)


def route_constructor():
    """Extend only two locally imported authority identities; undo proves AST."""
    original = lc.RetainedMissionSDKHost.__init__
    source = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree = copy.deepcopy(source)
    body = tree.body[0].body
    indexes = [i for i,n in enumerate(body) if isinstance(n, ast.ImportFrom)
        and n.module == 'hospital_mission_sequence'
        and 'checked_route_lifecycle_contract' in [a.name for a in n.names]]
    if len(indexes) != 1:
        raise ValueError('One original local route-authority import required')
    index = indexes[0]+1
    inserted = ast.parse('checked_route_lifecycle_contract = _controlled_contract\n'
                         'REMAINING_NO_NAV2_APPROVAL = _controlled_approval').body
    body[index:index] = inserted
    undone = copy.deepcopy(tree)
    del undone.body[0].body[index:index+2]
    if ast.dump(undone, include_attributes=False) != ast.dump(source, include_attributes=False):
        raise ValueError('Original complete host initializer must be recoverable')
    namespace = dict(original.__globals__, _controlled_contract=controlled_contract,
                     _controlled_approval=copy.deepcopy(APPROVAL))
    exec(compile(ast.fix_missing_locations(tree), '<controlled_authority_only_host>', 'exec'), namespace)
    return namespace['__init__']


class ControlledStaticHost(lc.PreparedHeldStaticSDKHost):
    __init__ = bind(lc.PreparedHeldStaticSDKHost.__init__, checked_fresh_held_approval=checked_approval)


class ControlledRouteHost(lc.RetainedMissionSDKHost):
    __init__ = route_constructor()


def context_for(folder, actor, geometry):
    p = SimpleNamespace(**vars(shared.preparation))
    p.ROBOT = actor
    p.LIVE = folder/'preparation.json'
    expected = copy.deepcopy(read(retained.H6/'actual_common_environment_rows.json')['compact_rows'])
    for path in p.SELECTED:
        expected[path]['collision_enabled'] = True
    p.load_manifest = lambda **kwargs: dict(compact_environment_rows=copy.deepcopy(expected))
    s = SimpleNamespace(**vars(shared.stationary))
    s.ROBOT, s.preparation = actor, p
    s.actor_configuration = bind(shared.stationary.actor_configuration, ROBOT=actor)
    s.critical_environment = bind(shared.stationary.critical_environment, ROBOT=actor, preparation=p)
    s.classify_contact = bind(shared.stationary.classify_contact, ROBOT=actor)
    s.RESULT = folder/'static/held_static_readback.json'
    s.TRACE = folder/'static/held_static_trace.csv'
    c = SimpleNamespace(**vars(shared.cohort))
    c.ROBOT = actor
    c.MARKER = actor.replace('patient_transport_', 'IJCAS_')
    context = SimpleNamespace(output=folder, preparation=p, stationary=s, cohort=c,
        protocol=dict(start_xyyaw=geometry['start_xyyaw']),
        protocol_path=folder/'workflow_protocol.json', source_archive=folder/'workflow_sources.zip')
    context.operations = SimpleNamespace(replay_static=lambda ignored:
        sequence.held_static_adapter(s.RESULT))
    return context


async def prepare_source(context, sdk, geometry, *, prior_result=None):
    from pxr import Usd, UsdGeom, UsdPhysics
    import carb.settings
    from omni.kit.viewport.utility import get_active_viewport
    import hospital_h4_visual_instance_refresh as display
    stage = sdk.omni.usd.get_context().get_stage()
    timeline = sdk.omni.timeline.get_timeline_interface()
    state = lambda: evidence.timeline_state(timeline)
    stopped = dict(playing=False, stopped=True, current_time_s=0.)
    if state()['playing']:
        raise ValueError('Never replace a playing experiment')
    original_root = stage.GetRootLayer().ExportToString()
    expected_receipt = read(retained.H6/'actual_common_environment_rows.json')
    before = state()
    def stable():
        if sdk.omni.usd.get_context().get_stage() != stage or state() != before:
            raise RuntimeError('Nonplaying stage/timeline changed during source preparation')
    await bootstrap.verify_rows(stage, expected_receipt['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async, unchanged=stable)
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 13:
        raise ValueError('Exactly one current thirteen-body actor required')
    old_actor = bodies[0].rsplit('/', 1)[0]
    if any(not p.startswith(old_actor+'/') for p in bodies):
        raise ValueError('Do not replace concurrent actors')
    archive = dict(timeline=before, Root_text=original_root,
        Session_text=stage.GetSessionLayer().ExportToString(), actor=old_actor,
        old_dynamic_state_restored=False)
    if before != stopped:
        if prior_result is None:
            raise ValueError('Own completed prior result required before retiring a PAUSED actor')
        previous = read(prior_result)
        view = sdk.SimulationManager.get_physics_sim_view().create_articulation_view(old_actor+'/base_link')
        snapshot = shared.preparation.old.tensor_snapshot(view)
        if (previous['actual_final_timeline'] != before
                or previous['actual_final_readback']['tensor'] != snapshot
                or not previous['checks'].get('original_gains_targets_restored')):
            raise ValueError('Exact restored complete prior Pause boundary required')
        archive.update(prior_result=str(prior_result), prior_result_sha256=sha(prior_result),
                       actual_tensor_snapshot=snapshot)
    else:
        if old_actor != '/World/patient_transport':
            raise ValueError('Only the first original untouched source can omit a prior physical result')
        archive['nominal_USD_only_no_historical_tensor_or_physical_replay'] = True
    evidence.write_exclusive(context.output/'prior_boundary.json', archive)
    stop_calls = 0
    if before != stopped:
        timeline.stop()
        stop_calls = 1
        begun = time.monotonic()
        while state() != stopped:
            if time.monotonic()-begun > 15.:
                raise TimeoutError('ONE Stop did not finish; no second Stop')
            await sdk.omni.kit.app.get_app().next_update_async()
    before = stopped
    await bootstrap.verify_rows(stage, expected_receipt['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async, unchanged=stable)
    if stage.GetPrimAtPath(context.cohort.ROBOT):
        raise ValueError('Fresh actor already reserved; no overwrite')
    constructor = shared.preparation.old.constructor_for(context.cohort.ROBOT)
    # These empty inactive names satisfy historical no-old-actor guards; they
    # are explicitly NOT restored H3/H4 actors or dynamic-state snapshots.
    retired = {old_actor, context.preparation.OLD_ROBOT,
               context.preparation.old.original.OLD_ROBOT, '/World/patient_transport'}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in retired:
            prim = stage.GetPrimAtPath(path) or stage.DefinePrim(path, 'Xform')
            prim.SetActive(False)
        made = constructor(stage, geometry['start_xyyaw'])
        marker = stage.DefinePrim(context.cohort.MARKER, 'Xform')
        marker.SetCustomData(dict(status='fresh_controlled_static_pending_NO_PLAY',
            robot_root=context.cohort.ROBOT, cohort=context.output.name))
    reference = shared.fresh_source_reference(context)
    if (reference['source_constructor_readback'] != made
            or context.stationary.actor_configuration(stage) != reference['fixed_configuration_rows']):
        raise ValueError('Fresh source/complete145 settings must match the anonymous constructor')
    validate = constructor.__globals__['validate_new_robot']
    source_stage = Usd.Stage.Open(str(context.preparation.ASSET))
    def physical():
        fixed = context.stationary.actor_configuration(stage)
        return dict(nonvisual_fixed_configuration_rows={p:v for p,v in fixed.items()
            if 'visuals' not in p[len(context.cohort.ROBOT):].split('/')},
            constructor=validate(stage, source_stage, geometry['start_xyyaw']))
    registered = await display.initialize_stopped_actor(stage, context.cohort.ROBOT,
        state=state, next_frame=sdk.omni.kit.app.get_app().next_update_async,
        physical_readback=physical, viewport=get_active_viewport(), settings=carb.settings.get_settings())
    evidence.write_exclusive(context.output/'display_initialization/renderer_initialization.json', registered)
    prep = dict(new_robot_root=context.cohort.ROBOT, start_xyyaw=geometry['start_xyyaw'],
        simulation_mass_kg=265., live_new_robot_readback=made, source_constructor_readback=made,
        protected_environment_sha256=expected_receipt['sha256'], source_reference=reference,
        expected_nominal_motor_baseline=read(retained.H6/'preparation.json')['expected_nominal_motor_baseline'],
        timeline=stopped, Play_or_static_or_drive_executed=False,
        Stop_calls=stop_calls, prior_boundary=str(context.output/'prior_boundary.json'),
        empty_inactive_legacy_names_are_NOT_restored_actors=True)
    prep['critical_environment'] = context.stationary.critical_environment(stage, prep)
    layers = dict(Root=stage.GetRootLayer().ExportToString(), Session=stage.GetSessionLayer().ExportToString())
    prep.update({k+'_sha256':hashlib.sha256(v.encode()).hexdigest() for k,v in layers.items()})
    evidence.write_exclusive(context.preparation.LIVE, prep)
    evidence.write_exclusive(context.output/'prepared_layer_texts.json', layers)
    print('[CONTROLLED SOURCE] NEW265kg/12joints/13links; first display verified; NO Play yet.', flush=True)
    return prep


def static_packet(context, geometry_path, grid_path):
    geometry = read(geometry_path)
    prep = read(context.preparation.LIVE)
    query = copy.deepcopy(read(retained.H6/'static_protocol.json'))
    query.update(new_robot_root=context.cohort.ROBOT, start_xyyaw=geometry['start_xyyaw'],
        goal_xyyaw=geometry['goal_xyyaw'], sampled_cooked_support_queries=retained.static_query_specs(geometry,query),
        source_actor_node_paths=list(prep['source_reference']['fixed_configuration_rows']))
    query_file = context.output/'static_query_contract.json'
    evidence.write_exclusive(query_file, query)
    files = [context.preparation.LIVE, context.output/'prepared_layer_texts.json',
        context.output/'display_initialization/renderer_initialization.json',
        retained.H6/'actual_common_environment_rows.json', geometry_path, grid_path,
        context.preparation.ASSET, retained.H6/'static_protocol.json', query_file,
        Path(__file__), Path(bootstrap.__file__), Path(sequence.__file__), *lc.host_source_files()]
    pins = {str(Path(p).resolve()):sha(p) for p in files}
    packet = dict(version=lc.PreparedHeldStaticSDKHost.VERSION, actor=context.cohort.ROBOT,
        cohort=context.output.name, scenario=geometry['scenario'], marker=context.cohort.MARKER,
        new_robot_root=context.cohort.ROBOT, run_id='hospital_h2_controlled_'+context.output.name+'_static',
        maximum_Play_calls=1, maximum_Stop_reset_restaging_calls=0, maximum_route_trials=0,
        automatic_retry=False, actual_held_static_authorized=True, use_for_final_statistics=False,
        explicit_static_approval=checked_approval(APPROVAL), held_static_contract=lc.held_static_contract(),
        required_STOPPED_timeline=dict(retained.boundary.STOPPED),
        preparation_layers_file=str(context.output/'prepared_layer_texts.json'),
        display_receipt=str(context.output/'display_initialization/renderer_initialization.json'),
        environment_rows_file=str(retained.H6/'actual_common_environment_rows.json'),
        geometry_file=str(geometry_path), grid_file=str(grid_path),
        static_query_contract_file=str(query_file), query_contract=query,
        workflow_protocol=str(context.protocol_path), input_sha256=pins)
    path = context.output/'static/static_protocol.json'
    evidence.write_exclusive(path, packet)
    lc.freeze_workflow_sources(context.source_archive, pins)
    evidence.write_exclusive(context.protocol_path, dict(user_approval=APPROVAL, input_sha256=pins,
        fresh_app_lineage_NOT_prior_physical_restoration=True, maximum_static_trials=1,
        maximum_route_trials=1, automatic_retry=False))
    return path, packet


def route_packet(context, original_plan, geometry_path, grid_path):
    prep, adapter, replay = sequence.held_static_adapter(context.stationary.RESULT)
    output = context.output/'route'
    plan = dict(read(original_plan), physical_execution_verified=False,
        integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True)
    layers = adapter['actual_final_readback']['layer_texts']
    frozen = output/'frozen_plan.json'
    static = output/'same_static_schema_binding.json'
    evidence.write_exclusive(frozen, plan)
    evidence.write_exclusive(static, adapter)
    profile = plan['geometry_profile']
    packet = dict(version=lc.RetainedMissionSDKHost.VERSION, actor=context.cohort.ROBOT,
        scenario=plan['scenario'], method=plan['method'], cohort=context.output.name,
        original_static_record=str(static), original_static_trace=str(context.stationary.TRACE),
        original_held_static_result=str(context.stationary.RESULT), plan_filename=frozen.name,
        geometry_file=str(geometry_path), grid_file=str(grid_path),
        environment_rows_file=str(retained.H6/'actual_common_environment_rows.json'),
        required_PAUSED_timeline=adapter['actual_final_timeline'],
        required_tensor_snapshot=adapter['actual_post_static_tensor_snapshot'],
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],
            unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS), policy=retained.TOP_XY_DISPLAY_POLICY),
        simulation_budget_s=180., maximum_wall_time_s=900., maximum_Play_calls=1,
        maximum_Stop_reset_restaging_calls=0, maximum_route_trials=1, automatic_retry=False,
        actual_route_authorized=True, use_for_final_statistics=False, explicit_route_approval=APPROVAL,
        common_tracking_policy=sequence.FINAL_TARGET_HOLD_POLICY,
        common_forward_tracking_floor_m_s=.010, common_reverse_tracking_floor_m_s=.020,
        geometry_profile=profile, common_lifecycle_contract=sequence.unified_geometry_lifecycle_contract(
            profile, simulation_budget_s=180.), raw_static_replay=replay)
    files = [frozen, static, context.stationary.RESULT, context.stationary.TRACE,
        context.preparation.LIVE, geometry_path, grid_path, Path(__file__),
        Path(sequence.__file__), *lc.host_source_files()]
    packet['input_sha256'] = {str(Path(p).resolve()):sha(p) for p in files}
    if packet['common_lifecycle_contract'] != controlled_contract(packet):
        raise ValueError('Unchanged common runtime contract required')
    path = output/'sdk_binding.json'
    evidence.write_exclusive(path, packet)
    lc.freeze_workflow_sources(output/'sources.zip', packet['input_sha256'])
    # The actual raw static record is preserved separately. The schema view
    # has aliases only; it is neither a new released-static test nor trial.
    context.stationary.RESULT, context.stationary.TRACE = static, Path(packet['original_static_trace'])
    context.operations.replay_static = lambda ignored: sequence.held_static_adapter(
        Path(packet['original_held_static_result']))
    return path, packet, plan


async def run_pilot_cell(method, *, prior_result=None, authorized=False):
    if authorized is not True:
        raise ValueError('Current user additional-experiment authority required')
    if method not in (comparison.B2, comparison.P):
        raise ValueError('First engineering pair is H2 B2/P only')
    folder = BASE/('B2_CS' if method == comparison.B2 else 'P_CS')
    if folder.exists():
        raise FileExistsError('Consumed live engineering cell; no automatic retry')
    original_plan = comparison.OUTPUT/(method+'.json')
    model_path = comparison.OUTPUT/'executor_model_screen'/method/'model_screen.json'
    model = read(model_path)
    if not model['both_nominal_resolutions_pass'] or read(model['plan']) != read(original_plan):
        raise ValueError('Own exact two-resolution model PASS required before scene changes')
    independent = read(comparison.OUTPUT/'offline_audits'/(method+'.json'))
    if independent['verdict'] != 'PASS' or independent['plan_sha256'] != sha(original_plan):
        raise ValueError('Own independent frozen-plan replay must pass before scene changes')
    retained.runtime.task.verify_pins(model['input_sha256'])
    retained.runtime.task.verify_pins(model['output_sha256'])
    geometry_path = Path(read(original_plan)['geometry_manifest'])
    grid_path = comparison.OUTPUT/'planning_occupancy.npz'
    geometry = read(geometry_path)
    folder.mkdir(parents=True, exist_ok=False)
    evidence.write_exclusive(folder/'cell_definition.json', dict(method=method, scenario=geometry['scenario'],
        authorization=APPROVAL, plan=str(original_plan), model_screen=str(model_path),
        common_speed_policy=read(original_plan)['common_speed_policy'],
        use_for_final_statistics=False, independent_final_repetition=False,
        no_blind_retry=True, maximum_static_trials=1, maximum_route_trials=1))
    actor = '/World/patient_transport_controlled_h2_'+folder.name.lower()+'_pilot_r01'
    context = context_for(folder, actor, geometry)
    sdk = retained.runtime.live_sdk()
    try:
        await prepare_source(context, sdk, geometry, prior_result=prior_result)
        path, packet = static_packet(context, geometry_path, grid_path)
        with np.load(grid_path) as grid:
            args = dict(occupied=grid['occupied'].copy(), origin=grid['origin_xy_m'].copy(),
                        resolution=float(grid['resolution_m']))
        host = ControlledStaticHost(sdk, packet, path, context=context, geometry=geometry,
                                    **args, authorized=True)
        result = await lc.run_held_static_once(host, lc.HeldStaticController(geometry['start_xyyaw']), authorized=True)
        if result['verdict'] != 'PASS':
            return dict(static_verdict=result['verdict'], route_verdict='NOT_RUN_STATIC_NOT_PASS',
                        result=str(context.stationary.RESULT))
        path, packet, plan = route_packet(context, original_plan, geometry_path, grid_path)
        port = ControlledRouteHost(sdk, packet, path, context=context, plan=plan, geometry=geometry,
                                   **args, authorized=True)
        controller = sequence.unified_geometry_lifecycle_controller(plan, geometry,
            profile=plan['geometry_profile'], simulation_budget_s=180.)
        result = await lc.run_lifecycle_once(port, controller, authorized=True,
                                             sdk_bridge_class=sequence.RetainFinalTargetsBridge)
        return dict(static_verdict='PASS', route_verdict=result['verdict'], result=str(port.result))
    except BaseException as failure:
        evidence.write_exclusive(folder/'workflow_error.json', dict(error=type(failure).__name__+': '+str(failure),
            no_automatic_retry=True, original_records_preserved=True))
        raise
