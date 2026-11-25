"""Mission static binding, retained diagnostics and approved zero settling.

Import/export are OFFLINE operations. Static entries require separate explicit
consent, an existing STOPPED actor and exact prepared layers. The SDK diagnostic
requires an already PAUSED retained actor and permits GETs only. Neither entry
constructs an actor, touches the camera, calls Stop/reset, admits a route or retries.
Historical preparation/runtime sources and their evidence remain immutable.
Separately authorized settling resumes the SAME PAUSED actor for fixed3+2s,
without steering/gain changes, actor construction, Stop/reset or route admission.
"""

import asyncio
import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np

import hospital_mission_geometry as geometry
import hospital_task_boundary as boundary
import hospital_task_runtime as runtime


shared = runtime.task.shared
evidence = shared.evidence
ROOT, GATE = shared.ROOT, shared.GATE
VERSION = 'MISSION_EXISTING_SOURCE_SINGLE_ZERO_COMMAND_STATIC_v1'
H6 = GATE/'hospital_h6_experiments/cohort_01_common_preparation_r01'
GEOMETRY = GATE/'hospital_h6_task_geometry/0e5f25f64d2256c2/common_geometry.json'
APPROVAL = dict(question_item_id='call_d1d9f39ac5a54fb6951b914a50a87186:0',
    answer_ko='코드 검증 후 정적 시험 1회 승인',
    includes=['ONE static: Play / 1s settling / 2s recording / confirmed Pause'],
    excludes=['forward', 'Stop', 'reset', 'restaging', 'display_refresh', 'route', 'automatic_retry'])


def read(path):
    return shared.preparation.read_small(path)


def pins(*maps):
    return runtime.task.merge_pins(*maps)


def verify_prepared_records(folder):
    """Validate actual saved preparation, not infer a new physical outcome."""
    folder = Path(folder).resolve()
    if folder != H6:
        raise ValueError('Only the separately approved actual H6 preparation is supported')
    context = geometry.load_common_preparation_protocol(folder/'preparation_protocol.json')
    prepared = read(folder/'preparation.json')
    closeout = read(folder/'prepare_only_closeout.json')
    rows = read(folder/'actual_common_environment_rows.json')
    display = read(folder/'display_initialization/renderer_initialization.json')
    stopped = dict(boundary.STOPPED)
    if (prepared['new_robot_root'] != context.cohort.ROBOT or prepared['timeline'] != stopped
            or prepared['start_xyyaw'] != context.protocol['start_xyyaw']
            or prepared['Play_or_static_or_drive_executed'] is not False
            or prepared['source_constructor_readback'] != prepared['live_new_robot_readback']
            or prepared['expected_nominal_motor_baseline'] != context.protocol['expected_nominal_motor_baseline']
            or prepared['source_constructor_readback']['simulation_mass_kg'] != 265.
            or len(prepared['source_constructor_readback']['rigid_bodies']) != 13
            or len(prepared['source_constructor_readback']['revolute_joint_names']) != 12
            or closeout['new_robot_root'] != context.cohort.ROBOT or closeout['timeline'] != stopped
            or (closeout['Stop_calls'], closeout['static_trials'], closeout['route_trials']) != (1, 0, 0)
            or closeout['automatic_retry'] is not False
            or closeout['actual_environment_sha256'] != prepared['protected_environment_sha256']
            or rows['codec'] != shared.preparation.compact.FORMAT or len(rows['compact_rows']) != 4997
            or rows['sha256'] != prepared['protected_environment_sha256']
            or shared.digest(dict(format=rows['codec'], rows=rows['compact_rows'])) != rows['sha256']
            or display['actor'] != context.cohort.ROBOT or display['registered_visual_root_count'] != 13
            or display['visible_USD_shape_count'] != 22
            or display['original_Root_and_Session_text_exactly_restored'] is not True
            or display['timeline_before_and_after'] != stopped):
        raise ValueError('Own actual complete STOPPED preparation/display/environment evidence required')
    for field, name in (('archive', 'actual_prior_state_before_Stop.json'), ('Stop_boundary', 'actual_Stop_boundary.json')):
        if prepared[field] != str(folder/name) or prepared[field+'_sha256'] != runtime.task.sha(folder/name):
            raise ValueError('Actual prior archive / consumed ONE Stop evidence changed')
    return context, prepared, rows, display


def static_query_specs(task_geometry, baseline):
    """Prospectively fixed H6 endpoint/initial-footprint samples plus a control.

    The selected control is cart7, NOT moved H6 cart14. Its exact old collider
    row is checked unchanged against actual H6 rows. Nothing is floor-exempt.
    These samples are deliberately NOT complete route/cooked-map validation.
    """
    length, width = task_geometry['full_body_rectangle_m']
    margin = task_geometry['full_body_clearance_m']
    if [length, width, margin] != [1.63, .965, .010]:
        raise ValueError('Original full body and 10mm outline must remain unchanged')
    a, b = length/2 + margin, width/2 + margin
    result = []
    def add(name, kind, pose, u, v):
        x, y, yaw = pose
        result.append(dict(id=name, kind=kind,
            xy_m=[x+math.cos(yaw)*u-math.sin(yaw)*v, y+math.sin(yaw)*u+math.cos(yaw)*v],
            origin_z_m=.6, length_m=.8, direction=[0., 0., -1.]))
    for role in ('start', 'goal'):
        pose = task_geometry[role+'_xyyaw']
        for label, u, v in (('center', 0., 0.), ('front_left', a, b), ('front_right', a, -b),
                            ('rear_left', -a, b), ('rear_right', -a, -b)):
            add(role+'/'+label, 'endpoint', pose, u, v)
    for i, u in enumerate(np.linspace(-a*.75, a*.75, 4)):
        for j, v in enumerate(np.linspace(-b*.75, b*.75, 4)):
            add('initial_interior/%d/%d' % (i, j), 'initial_footprint',
                task_geometry['start_xyyaw'], float(u), float(v))
    result.append(copy.deepcopy(baseline['sampled_cooked_support_queries'][-1]))
    if result[-1]['kind'] != 'cart_control' or len({r['id'] for r in result}) != 27:
        raise ValueError('26 H6 samples plus one unchanged cart7 control required')
    rectangles = task_geometry['same_three_authored_support_rectangles']
    for row in result:
        if not any(min(p[0] for p in poly) <= row['xy_m'][0] <= max(p[0] for p in poly)
                   and min(p[1] for p in poly) <= row['xy_m'][1] <= max(p[1] for p in poly) for poly in rectangles):
            raise ValueError('Prescribed sample is outside authored support: '+row['id'])
    return result


def static_definition(folder=H6):
    """OFFLINE prospective definition. No physical/task reservation is made."""
    context, prepared, rows, display = verify_prepared_records(folder)
    original = read(shared.STATIC_DESIGN)
    previous_path = GATE/'hospital_h5_experiments/cohort_04_B2_development_r01/runtime_protocol.json'
    baseline = read(previous_path)
    task_geometry = read(GEOMETRY)
    if (task_geometry['start_xyyaw'] != prepared['start_xyyaw']
            or task_geometry['goal_xyyaw'] != context.protocol['goal_xyyaw']
            or not all(r['original_full_body10mm_clear'] and r['authored_support_missing_area_m2'] == 0.
                       for r in task_geometry['endpoint_footprint_and_authored_support_review'])):
        raise ValueError('Actual adopted H6 task/endpoint geometry must match')
    original_rows = read(context.preparation.MANIFEST)['compact_environment_rows']
    unchanged = {name for name, row in rows['compact_rows'].items() if original_rows.get(name) == row}
    inventory = {name: value for name, value in baseline['query_known_enabled_colliders'].items() if name in unchanged}
    cart = baseline['selected_original_cart']
    cart_paths = [name for name in inventory if name.startswith(cart+'/')]
    if cart != '/World/Hospital/SM_SupplyCart_02a7' or len(cart_paths) != 1:
        raise ValueError('Exact unchanged original cart7 control identity required; not moved cart14')
    source = prepared['source_reference']['fixed_configuration_rows']
    if len(source) != 145 or len(original['additional_disabled_subfloor_probes']) != 4:
        raise ValueError('Complete source actor and original four absence probes required')
    files = [context.protocol_path, GEOMETRY, previous_path, shared.STATIC_DESIGN,
        context.preparation.MANIFEST, Path(__file__), ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',
        *[context.output/n for n in ('preparation.json', 'prepare_only_closeout.json',
            'actual_common_environment_rows.json', 'actual_Stop_boundary.json', 'actual_prior_state_before_Stop.json',
            'environment_application.json', 'environment_contract.json', 'display_initialization/renderer_initialization.json')]]
    result = dict(version=VERSION, scenario='H6_integrated_transfer_mission', cohort=context.protocol['cohort'],
        output_directory=str(context.output), preparation_protocol=str(context.protocol_path),
        new_robot_root=context.cohort.ROBOT, marker=context.cohort.MARKER,
        retired_robot_root=context.protocol['retired_robot_root'], start_xyyaw=prepared['start_xyyaw'],
        goal_xyyaw=context.protocol['goal_xyyaw'], expected_nominal_motor_baseline=prepared['expected_nominal_motor_baseline'],
        protected_environment_sha256=prepared['protected_environment_sha256'],
        trial_role='H6_OWN_PREPARED_SOURCE_ZERO_COMMAND_STATIC_DEVELOPMENT_ONLY', run_id='hospital_h6_static_r01',
        supports=original['supports'], support_top_m=original['support_top_m'],
        additional_sampled_query_limits=original['additional_sampled_query_limits'],
        sampled_cooked_support_queries=static_query_specs(task_geometry, baseline), selected_original_cart=cart,
        query_known_enabled_colliders=inventory, source_actor_node_paths=list(source),
        approved_disabled_subfloor_paths=original['approved_disabled_subfloor_paths'],
        original_disabled_collider_paths=baseline['original_disabled_collider_paths'],
        additional_disabled_subfloor_probes=original['additional_disabled_subfloor_probes'],
        auxiliary_probe_known_nonself_colliders=[n for n in original['auxiliary_probe_known_nonself_colliders']
                                                if n in unchanged or n in original['supports']],
        compact_readback_wall_budget_s=180., unchanged_original_static_limits=dict(shared.stationary.LIMITS),
        unchanged_original_static_guards=dict(shared.stationary.GUARDS), settle_s=1., record_s=2., physics_hz=120,
        same_prepared_actor_only=True, maximum_static_trials=1, maximum_route_trials=0, maximum_Stop_calls=0,
        automatic_retry=False, actual_motion_authorized=False, use_for_final_statistics=False,
        static_permission_is_NOT_P_path_or_drive_permission=True, user_approval=APPROVAL,
        exact_original_static_function_code_reused=True,
        query_scope='26_H6_INITIAL_AND_ENDPOINT_SAMPLES_PLUS_UNCHANGED_CART7_CONTROL_NOT_WHOLE_ROUTE',
        required_display_settings=display['settings_before_and_after'],
        input_sha256=pins(context.protocol['input_sha256'], {str(p):runtime.task.sha(p) for p in files}))
    runtime.task.verify_pins(result['input_sha256'])
    return evidence.json_value(result)


def support_evaluation(protocol, rows):
    result = runtime.support_evaluation(protocol, rows)
    return dict(result, version='H6_PROSPECTIVE_TWO_CHANNEL_ALL_HIT_STATIC_QUERIES_v1',
                original_evaluator_code_unchanged=True)


def subfloor_evaluation(protocol, support, rows):
    result = shared.stationary.evaluate_subfloor_queries(protocol, support, rows)
    return dict(result, scope='FOUR_UNCHANGED_SUBFLOOR_DIAGNOSTICS_WITH_27_H6_SAMPLE_RAYS',
                inherited_scope_text=result['scope'])


def static_log(*args, **kwargs):
    args = tuple(a.replace('[H4 STATIC]', '[H6 STATIC]').replace(
        '=== H4 FRESH FIXED-START STATIC ===', '=== H6 EXISTING PREPARED SOURCE STATIC ===')
        if isinstance(a, str) else a for a in args)
    builtins.print(*args, **kwargs)


def static_context(protocol, path):
    """Private configuration only; no live SDK import or actor construction."""
    path = Path(path).resolve()
    if (protocol['version'] != VERSION or path != H6/'static_protocol.json'
            or protocol['user_approval'] != APPROVAL or protocol['maximum_route_trials'] != 0
            or protocol['maximum_Stop_calls'] != 0 or protocol['maximum_static_trials'] != 1
            or protocol['automatic_retry'] is not False):
        raise ValueError('Exact separately approved static-only mission definition required')
    return static_binding(protocol, path, H6/'static', H6/'preparation.json')


def static_binding(protocol, path, static_folder, preparation_receipt):
    """One shared binding; never clone or change the physical observation loop."""
    base = geometry.common_preparation_context(protocol, path)
    p, s = base.preparation, base.stationary
    p.LIVE = Path(preparation_receipt)
    # Saved rows normalize the four flag values to True ONLY in the expected
    # archive representation. Original compare_rows still requires live False
    # for exactly those four; every property digest/other identity stays exact.
    def manifest(verify_inputs=True):
        value = read(H6/'actual_common_environment_rows.json')
        expected = copy.deepcopy(value['compact_rows'])
        for name in p.SELECTED:
            if expected[name]['collision_enabled'] is not False:
                raise ValueError('All four already-disabled actual identities must remain disabled')
            expected[name]['collision_enabled'] = True
        return dict(compact_environment_rows=expected)
    p.load_manifest = manifest
    s.ROBOT, s.MARKER, s.preparation = p.ROBOT, p.MARKER, p
    wheels = SimpleNamespace(WHEELS=tuple(s.WHEELS.values()))
    s.classify_contact = shared._bound(shared.stationary.classify_contact, ROBOT=p.ROBOT, cohort=wheels)
    s.evaluate_subfloor_queries = subfloor_evaluation
    s.RESULT = Path(static_folder)/(protocol['run_id']+'.json')
    s.TRACE, s.PROTOCOL = s.RESULT.with_suffix('.csv'), path
    base.static_run_id = protocol['run_id']
    base.static_folder = s.RESULT.parent
    base.support_evaluator = support_evaluation
    reference = shared._bound(shared.fresh_source_reference, preparation=p)
    source = shared._bound(shared.fresh_source_for_live_actor, preparation=p, fresh_source_reference=reference)
    completed = SimpleNamespace(**vars(shared.completed_static))
    completed.frozen_query_functions = runtime.query_functions
    static = shared._bound(shared.run_retained_static, preparation=p, cohort=base.cohort,
        completed_static=completed, STATIC_OUTPUT=s.RESULT.parent, STATIC_RUN_ID=base.static_run_id,
        STATIC_DESIGN=path, fresh_source_for_live_actor=source,
        evaluate_support_queries_for_actor=support_evaluation, print=static_log)
    replay = shared._bound(shared.load_fresh_static_inputs, preparation=p, completed_static=completed)
    base.operations = SimpleNamespace(static=static, source_for_live_actor=source, replay_static=replay)
    return base


def load_static(path):
    path = Path(path).resolve()
    protocol = read(path)
    context = static_context(protocol, path)
    if protocol != static_definition(H6):
        raise ValueError('Prospective static definition changed; no Play or repair')
    return context


def validate_live_boundary(context, sdk):
    """GET only. No changing hashes to accept arbitrary current layer edits."""
    stage = sdk.omni.usd.get_context().get_stage()
    timeline = sdk.omni.timeline.get_timeline_interface()
    prep = read(context.preparation.LIVE)
    if stage is None or evidence.timeline_state(timeline) != boundary.STOPPED:
        raise ValueError('Same already prepared actor must remain STOPPED at zero')
    for key, layer in (('Root', stage.GetRootLayer()), ('Session', stage.GetSessionLayer())):
        if hashlib.sha256(layer.ExportToString().encode()).hexdigest() != prep[key+'_sha256']:
            raise ValueError('Prepared '+key+' boundary changed; no automatic restoration/Play')
    marker = stage.GetPrimAtPath(context.cohort.MARKER)
    if not marker or marker.GetCustomDataByKey('robot_root') != context.cohort.ROBOT:
        raise ValueError('Exact own prepared actor marker required')
    return stage, timeline


def no_other_task():
    current = asyncio.current_task()
    for name in boundary.ACTIVE_TASKS:
        previous = getattr(builtins, name, None)
        if previous is not None and previous is not current and not previous.done():
            raise ValueError('Another research operation is active; no competing task')


async def run_static_once(path, *, authorized=False):
    if authorized is not True:
        raise ValueError('Explicit ONE static consent required before SDK access')
    context = load_static(path)
    target = context.output/'static'
    dispatch = target/'dispatch_reserved.json'
    if dispatch.exists():
        print('[H6 STATIC] Prior dispatch retained; no new action or retry.', flush=True)
        return dict(status='PRIOR_STATIC_DISPATCH_RETAINED_NO_NEW_ACTION')
    no_other_task()
    evidence.write_exclusive(dispatch, dict(protocol=str(context.protocol_path),
        protocol_sha256=runtime.task.sha(context.protocol_path), user_approval=APPROVAL,
        maximum_static_trials=1, maximum_Stop_calls=0, maximum_route_trials=0, automatic_retry=False))
    phase = 'existing_stopped_preflight'
    try:
        sdk = runtime.live_sdk()
        stage, timeline = validate_live_boundary(context, sdk)
        # Verify renderer preferences without toggling anything or refreshing.
        import carb.settings
        settings = carb.settings.get_settings()
        if any(settings.get(k) != v for k, v in context.protocol['required_display_settings'].items()):
            raise ValueError('Prepared renderer/physics preferences changed; no setting edit or Play')
        phase = 'original_zero_command_static'
        report = await context.operations.static(context)
        replay = None
        if report['verdict'] == 'PASS':
            phase = 'same_static_raw_evidence_replay'
            _, _, raw = context.operations.replay_static(context)
            replay = raw['verdict']
        result = dict(status='ONE_H6_STATIC_COMPLETED_NO_DRIVE_ADMISSION',
            static_verdict=report['verdict'], own_raw_replay_verdict=replay,
            actual_timeline=evidence.timeline_state(timeline), maximum_route_trials=0,
            automatic_retry=False, new_route_trials=0, result=str(context.stationary.RESULT))
        evidence.write_exclusive(target/'static_only_closeout.json', shared.normalize_runtime_evidence(result))
        print('[H6 STATIC] Finished:', result, flush=True)
        return result
    except BaseException as error:
        evidence.write_exclusive(target/'static_operation_error.json', dict(phase=phase,
            error=type(error).__name__+': '+str(error), no_automatic_retry=True,
            no_Stop_reset_restaging_display_refresh_or_route=True))
        raise


def schedule_static(path, *, authorized=False):
    if authorized is not True:
        raise ValueError('Explicit single-static consent required before scheduling')
    # Scheduling does not reserve a second physical attempt or rerun old work.
    if (H6/'static/dispatch_reserved.json').exists():
        print('[H6 STATIC] Existing dispatch retained; do not rerun.', flush=True)
        return None
    no_other_task()
    def done(future):
        if future.cancelled():
            print('[H6 STATIC] Cancelled; preserve state. No retry.', flush=True)
        elif future.exception() is not None:
            import traceback
            error = future.exception()
            traceback.print_exception(type(error), error, error.__traceback__)
            print('[H6 STATIC] Aborted; do not rerun or manipulate timeline.', flush=True)
    future = asyncio.ensure_future(run_static_once(path, authorized=True))
    future.add_done_callback(done)
    builtins.ijcas_hospital_latest_task_experiment = future
    print('[H6 STATIC] ONE approved static preflight scheduled. Same actor; NO Stop/restaging/display refresh/route/retry.', flush=True)
    return future


def export_static(folder=H6):
    """Export one command + immutable inputs OFFLINE, never schedule/Play."""
    protocol = static_definition(folder)
    target = Path(folder)/'static_protocol.json'
    if target.exists():
        if read(target) != protocol:
            raise ValueError('Existing static definition preserved; no overwrite')
        return target
    evidence.write_exclusive(target, protocol)
    archive = Path(folder)/'static_runtime_sources.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(protocol['input_sha256']):
            source = Path(name)
            if source.suffix == '.py' and source.is_relative_to(ROOT):
                stream.write(source, str(source.relative_to(ROOT)))
    return target


def recorded_unstarted_static():
    """Authenticate the actual pre-Play abort against its executed source ZIP.

    The original protocol/command/result are never edited. Only this shared
    module and its test may evolve; their EXACT previously pinned bytes must
    exist in the original archive. This is historical source resolution for a
    GET diagnosis, not a pin exemption for launching the old static command.
    """
    protocol_path = H6/'static_protocol.json'
    protocol = read(protocol_path)
    static = H6/'static'
    reserve, error = read(static/'dispatch_reserved.json'), read(static/'static_operation_error.json')
    if (reserve['protocol'] != str(protocol_path)
            or reserve['protocol_sha256'] != runtime.task.sha(protocol_path)
            or reserve['user_approval'] != APPROVAL
            or error['phase'] != 'existing_stopped_preflight'
            or error['error'] != 'ValueError: Prepared Session boundary changed; no automatic restoration/Play'
            or error['no_automatic_retry'] is not True
            or error['no_Stop_reset_restaging_display_refresh_or_route'] is not True
            or any((static/n).exists() for n in ('single_attempt_reserved.json',
                protocol['run_id']+'.json', protocol['run_id']+'.csv', 'static_only_closeout.json'))):
        raise ValueError('Exact recorded pre-Play abort and absence of a physical attempt required')
    evolving = {Path(__file__).resolve(), ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py'}
    resolved = []
    with zipfile.ZipFile(H6/'static_runtime_sources.zip') as archive:
        for name, digest in protocol['input_sha256'].items():
            path = Path(name)
            if runtime.task.sha(path) == digest:
                continue
            member = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
            if path not in evolving or hashlib.sha256(archive.read(member)).hexdigest() != digest:
                raise ValueError('Unreviewed historical static dependency change: '+name)
            resolved.append(dict(path=name, original_sha256=digest, archived_member=member,
                                 current_sha256=runtime.task.sha(path)))
    return protocol, dict(original_protocol_sha256=runtime.task.sha(protocol_path),
        original_runtime_source_archive_sha256=runtime.task.sha(H6/'static_runtime_sources.zip'),
        exact_executed_source_resolution=resolved, phase_before_static_operation_verified=True,
        no_physical_static_reservation_or_result=True, historical_callback_or_timeline_outcome_NOT_invented=True)


def snapshot_deltas(before, after):
    """Exact native field comparison, including spec metadata and root fields."""
    missing = dict(field_is_not_authored=True)
    result = []
    for path in sorted(set(before)|set(after)):
        a, b = before.get(path, {}), after.get(path, {})
        first, last = a.get('info', {}), b.get('info', {})
        fields = sorted(set(first)|set(last))
        if a.get('spec_type') != b.get('spec_type'):
            fields = ['__spec_type__', *fields]
        for field in fields:
            x = a.get('spec_type', missing) if field == '__spec_type__' else first.get(field, missing)
            y = b.get('spec_type', missing) if field == '__spec_type__' else last.get(field, missing)
            if x != y: result.append(dict(path=path, field=field, before=x, after=y))
    return result


def apply_recorded_field_changes(snapshot, changes):
    """Apply literal saved transaction fields to a COPY, never a USD stage."""
    result = copy.deepcopy(snapshot)
    missing = dict(field_is_not_authored=True)
    for row in changes:
        path, field = row['path'], row['field']
        value = result.get(path, {})
        actual = value.get('spec_type', missing) if field == '__spec_type__' else value.get('info', {}).get(field, missing)
        if actual != row['before']:
            raise ValueError('Saved transaction does not apply exactly: '+path+':'+field)
        if row['after'] == missing:
            raise ValueError('This additive environment transaction must not remove a field')
        entry = result.setdefault(path, dict(info={}))
        if field == '__spec_type__': entry['spec_type'] = row['after']
        else: entry['info'][field] = copy.deepcopy(row['after'])
    return result


def expected_staged_native_fields():
    """OFFLINE reconstruction of declared noncamera fields, NOT lost live text.

    Start with the ACTUAL completed-Stop Session; apply every recorded common
    environment field transaction; reproduce the original constructor/marker
    in an isolated anonymous source fixture; retire exactly the old actor.
    Camera values are the saved Stop values, NOT asserted to be the later
    preparation camera values. The lost full prepared Session text/hash is
    not fabricated. Every noncamera field is compared without subtree waivers.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    context, prepared, _, _ = verify_prepared_records(H6)
    fields = context.preparation.old.fields
    stop = read(H6/'actual_Stop_boundary.json')
    if hashlib.sha256(stop['Root_text'].encode()).hexdigest() != prepared['Root_sha256']:
        raise ValueError('Actual completed-Stop Root must equal the prepared Root')
    original = fields.layer_snapshot(stop['Session_text'])
    applied = read(H6/'environment_application.json')['exact_Session_changes']
    expected = apply_recorded_field_changes(original, applied)
    fixture = Usd.Stage.CreateInMemory('H6_NATIVE_PREPARATION_FIELD_FIXTURE_NOT_LIVE.usda')
    UsdShade.Material.Define(fixture, context.preparation.old.constructor.MATERIAL)
    made = context.preparation.old.constructor_for(context.cohort.ROBOT)(fixture, context.protocol['start_xyyaw'])
    attr = fixture.GetPrimAtPath(context.cohort.ROBOT).CreateAttribute(
        'isaac:namespace', Sdf.ValueTypeNames.String, custom=False, variability=Sdf.VariabilityVarying)
    attr.SetDisplayName('Namespace'); attr.SetDocumentation('Namespace of the prim in Isaac Sim')
    marker = UsdGeom.Xform.Define(fixture, context.cohort.MARKER).GetPrim()
    for key, value in dict(robot_root=context.cohort.ROBOT, status='fresh_static_pending_NO_PLAY',
                           cohort=context.protocol['cohort']).items():
        marker.SetCustomDataByKey(key, value)
    offline_source = read(H6/'anonymous_source_reference.json')
    if (made != prepared['source_constructor_readback']
            or context.stationary.actor_configuration(fixture) != offline_source['fixed_configuration_rows']):
        raise ValueError('Original offline anonymous constructor/registered-schema settings must reproduce exactly')
    source_fields = fields.layer_snapshot(fixture.GetRootLayer().ExportToString())
    roots = [Sdf.Path(context.cohort.ROBOT), Sdf.Path(context.cohort.MARKER)]
    for path, value in source_fields.items():
        if any(Sdf.Path(path).GetPrimPath().HasPrefix(root) for root in roots):
            if path in expected: raise ValueError('Fresh actor/marker unexpectedly exists in the saved Stop')
            expected[path] = value
    expected[context.protocol['retired_robot_root']]['info']['active'] = False
    return dict(Root=fields.layer_snapshot(stop['Root_text']), Session=expected), dict(
        actual_completed_Stop_Root_text=stop['Root_text'],
        recorded_environment_field_transactions=len(applied),
        original_constructor_and_all145_OFFLINE_source_settings_reproduced=True,
        live_GET_must_independently_compare_all145_settings_to_own_saved_SAME_SDK_live_reference=True,
        offline_and_live_registry_fallback_fingerprints_are_NOT_silently_equated=True,
        prepared_Session_text_was_NOT_saved_or_recovered=True,
        camera_reference_is_actual_completed_Stop_NOT_assumed_final_preparation=True,
        reconstruction_does_NOT_prove_the_missing_historical_failed_transaction=True)


def export_boundary_get():
    """A GET-only contract, independent of the consumed static dispatch."""
    protocol, history = recorded_unstarted_static()
    snapshots, reconstruction = expected_staged_native_fields()
    folder = H6/'static_boundary_GET'
    contract_path = folder/'GET_contract.json'
    if contract_path.exists(): raise ValueError('Existing GET contract retained; do not replace or repeat')
    evidence.write_exclusive(folder/'declared_preparation_native_fields.json', dict(
        snapshots=snapshots, reconstruction=reconstruction))
    files = [H6/'static_protocol.json', H6/'static/dispatch_reserved.json',
        H6/'static/static_operation_error.json', H6/'static_runtime_sources.zip',
        folder/'declared_preparation_native_fields.json', Path(__file__).resolve(),
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',
        H6/'preparation_protocol.json']
    preparation = read(H6/'preparation_protocol.json')
    contract = dict(version='H6_RECORDED_PREPLAY_SESSION_FIELD_DIAGNOSIS_GET_ONLY_v1',
        new_robot_root=protocol['new_robot_root'], required_timeline=dict(boundary.STOPPED),
        historical_static=history, exactly_one_GET=True, actual_motion_authorized=False,
        maximum_static_trials=0, maximum_Stop_calls=0, maximum_route_trials=0,
        no_stage_or_camera_or_renderer_or_timeline_commands=True,
        input_sha256=pins(preparation['input_sha256'], {str(p):runtime.task.sha(p) for p in files}))
    evidence.write_exclusive(contract_path, contract)
    return contract_path


async def run_boundary_get(path):
    """Existing scene reads only; NEVER calls a static/control/display operation."""
    path = Path(path).resolve()
    if path != H6/'static_boundary_GET/GET_contract.json': raise ValueError('Own GET-only contract required')
    contract = read(path); runtime.task.verify_pins(contract['input_sha256'])
    if contract['maximum_static_trials'] != 0 or contract['maximum_Stop_calls'] != 0 or contract['maximum_route_trials'] != 0:
        raise ValueError('GET contract must not grant a physical operation')
    target = path.parent/'live_boundary.json'
    if target.exists():
        print('[H6 BOUNDARY GET] Existing readback retained; no duplicate operation.', flush=True)
        return read(target)
    no_other_task()
    import omni.usd
    import omni.timeline
    import omni.kit.app
    from pxr import Usd
    context, prepared, _, _ = verify_prepared_records(H6)
    sdk_usd = omni.usd.get_context(); stage = sdk_usd.get_stage()
    timeline = omni.timeline.get_timeline_interface()
    state = evidence.timeline_state(timeline)
    if stage is None or state != contract['required_timeline']:
        raise ValueError('Existing same STOPPED-zero scene required; no timeline command issued')
    def texts():
        value = dict(Root=stage.GetRootLayer().ExportToString(), Session=stage.GetSessionLayer().ExportToString())
        if sum(len(v.encode()) for v in value.values()) > 2*1024*1024: raise ValueError('Layer diagnostic text budget exceeded')
        return value
    initial = texts(); rows = {}; transactions = []; began = heartbeat = time.monotonic()
    captured = path.parent/'actual_layers_before_environment_GET.json'
    if captured.exists():
        print('[H6 BOUNDARY GET] Prior captured boundary retained; no duplicate GET.', flush=True)
        return read(captured)
    # Persist the primary diagnostic BEFORE yielding, scanning or validating.
    # A later error can no longer lose the actual changed Session fields.
    evidence.write_exclusive(captured, dict(actual_layer_texts=initial, actual_timeline=state,
        recorded_preparation_layer_sha256={k:prepared[k+'_sha256'] for k in ('Root', 'Session')},
        no_stage_or_timeline_command=True, new_static_or_route_trials=0))
    def unchanged():
        same = sdk_usd.get_stage() == stage and evidence.timeline_state(timeline) == state
        current = texts()
        if not same or current != initial:
            transactions.append(dict(same_stage_and_timeline=same, layer_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in current.items()}))
            evidence.write_exclusive(path.parent/'changed_during_GET.json', dict(
                actual_current_layer_texts=current, actual_timeline=evidence.timeline_state(timeline),
                actual_before_layer_capture=str(captured), rows_read=len(rows),
                layer_changes={k:shared.layer_field_differences(initial[k],current[k]) for k in ('Root','Session')},
                no_coherent_environment_or_continuation_claim=True, no_stage_or_timeline_command=True))
            raise ValueError('State changed during GET; no coherent readback or continuation claim')
    for name, row in context.preparation.compact.stage_rows(stage, context.preparation.SELECTED, context.preparation.old.original):
        if name in rows: raise ValueError('Duplicate protected environment identity')
        rows[name] = row
        if len(rows)%16 == 0:
            unchanged(); await omni.kit.app.get_app().next_update_async(); unchanged()
        if time.monotonic()-heartbeat >= 15.:
            print('[H6 BOUNDARY GET] Protected rows=%d/4997; no motion.' % len(rows), flush=True); heartbeat=time.monotonic()
        if time.monotonic()-began > 180.: raise TimeoutError('Bounded GET readback exceeded')
    unchanged()
    expected_rows = read(H6/'actual_common_environment_rows.json')['compact_rows']
    row_changes = sorted(name for name in set(rows)|set(expected_rows) if rows.get(name) != expected_rows.get(name))
    declared = read(path.parent/'declared_preparation_native_fields.json')
    fields = context.preparation.old.fields
    actual = {key:fields.layer_snapshot(value) for key,value in initial.items()}
    changes = {key:snapshot_deltas(declared['snapshots'][key], actual[key]) for key in ('Root', 'Session')}
    display_only = not changes['Root']; classifications = []
    for row in changes['Session']:
        try:
            boundary._validate_display_row(row)
            classifications.append(dict(**row, classification='EXACT_REVIEWED_TOP_XY_OR_APERTURE_DEFAULT'))
        except ValueError as error:
            display_only = False
            classifications.append(dict(**row, classification='UNREVIEWED_FIELD_NOT_WAIVED', reason=str(error)))
    validator = shared._bound(context.preparation.old.constructor.validate_new_robot, ROBOT=context.cohort.ROBOT)
    source = fixed = critical = None
    read_errors = {}
    for name, operation in (
        ('source', lambda:validator(stage, Usd.Stage.Open(str(context.preparation.ASSET)), context.protocol['start_xyyaw'])),
        ('fixed', lambda:context.stationary.actor_configuration(stage)),
        ('critical', lambda:context.stationary.critical_environment(stage, prepared))):
        try:
            value = operation()
            if name == 'source': source = value
            elif name == 'fixed': fixed = value
            else: critical = value
        except Exception as error: read_errors[name] = type(error).__name__+': '+str(error)
    unchanged()
    result = dict(status='H6_SAME_STOPPED_ACTOR_SESSION_NATIVE_FIELD_GET_ONLY_NO_STATIC_ADMISSION',
        actual_timeline=state, actor=context.cohort.ROBOT, actual_layer_texts=initial,
        actual_layer_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in initial.items()},
        recorded_preparation_layer_sha256={k:prepared[k+'_sha256'] for k in ('Root','Session')},
        declared_reconstruction=declared['reconstruction'], exact_native_field_differences=changes,
        Session_field_classifications=classifications, only_narrow_Top_display_differences_vs_declared_fields=display_only,
        source_constructor_unchanged=source == prepared['source_constructor_readback'],
        all145_source_fixed_settings_unchanged=fixed == prepared['source_reference']['fixed_configuration_rows'],
        critical_physics_and_material_and_support_unchanged=critical == prepared['critical_environment'],
        source_and_physics_readback_errors=read_errors,
        protected_environment_row_count=len(rows), protected_environment_changed_paths=row_changes,
        actual_compact_environment_sha256=shared.digest(dict(format=context.preparation.compact.FORMAT, rows=rows)),
        same_stage_layers_timeline_throughout_GET=True, GET_time_transactions=transactions,
        new_static_or_route_trials=0, timeline_or_stage_commands=0, actual_motion_authorized=False,
        missing_historical_failed_field_transaction_cause_NOT_claimed=True,
        input_sha256=pins(contract['input_sha256'], {str(path):runtime.task.sha(path)}))
    evidence.write_exclusive(target, shared.normalize_runtime_evidence(result))
    print('=== H6 PREPLAY SESSION / SAME ACTOR GET-ONLY DIAGNOSIS ===', flush=True)
    print('timeline:', state, '; Root / Session native changed fields vs DECLARED preparation:',
          len(changes['Root']), len(changes['Session']), flush=True)
    for row in classifications:
        print('FIELD:', row['path'], row['field'], 'before:', row['before'], 'now:', row['after'], row['classification'], flush=True)
    print('source / all145 settings / physics unchanged:', result['source_constructor_unchanged'],
          result['all145_source_fixed_settings_unchanged'], result['critical_physics_and_material_and_support_unchanged'], flush=True)
    print('protected rows / changed paths:', len(rows), row_changes, flush=True)
    print('saved:', target, flush=True)
    print('GET only. No static retry, Play/Stop/Pause/reset, source/camera/display edit or drive admission.', flush=True)
    return result


def schedule_boundary_get(path):
    no_other_task()
    def done(future):
        if future.cancelled(): print('[H6 BOUNDARY GET] Cancelled; no automatic retry.', flush=True)
        elif future.exception() is not None:
            import traceback
            error = future.exception(); traceback.print_exception(type(error), error, error.__traceback__)
    future = asyncio.ensure_future(run_boundary_get(path)); future.add_done_callback(done)
    builtins.ijcas_hospital_latest_task_experiment = future
    print('[H6 BOUNDARY GET] Same actor / exact fields / all4997 rows ONLY. NO timeline or stage command.', flush=True)
    return future


def reconstruct_preparation_session_text():
    """OFFLINE anonymous COPY only, retaining opaque/unregistered metadata.

    Reproduce the exact recorded232-field environment transaction and original
    source constructor. Do not open or edit the current hospital stage. This
    reconstructs DECLARED staging; camera values are from the saved Stop, not
    asserted to be the missing final-preparation camera snapshot.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    context, prepared, _, _ = verify_prepared_records(H6)
    stop = read(H6/'actual_Stop_boundary.json')
    patch, attrs, roots = geometry.checked_common_Session_patch(Path(context.protocol['common_environment_patch']).read_text())
    session = Sdf.Layer.CreateAnonymous('H6_DECLARED_PREPARATION_COPY.sdf')
    if not session.ImportFromString(stop['Session_text']): raise ValueError('Cannot parse recorded Stop COPY')
    for path in roots: Sdf.CreatePrimInLayer(session, path).instanceable = False
    for path, spec in attrs.items():
        Sdf.CreatePrimInLayer(session, spec.path.GetPrimPath())
        if not Sdf.CopySpec(patch, spec.path, session, spec.path): raise ValueError('Anonymous property copy failed')
    actual_deltas = shared.layer_field_differences(stop['Session_text'], session.ExportToString())
    if actual_deltas != read(H6/'environment_application.json')['exact_Session_changes']:
        raise ValueError('Every original232 environment field transaction must reproduce exactly')
    stage = Usd.Stage.Open(Sdf.Layer.CreateAnonymous('H6_MINIMAL_FIXTURE_ROOT.usda'), session)
    UsdShade.Material.Define(stage, context.preparation.old.constructor.MATERIAL)
    with Usd.EditContext(stage, session):
        stage.GetPrimAtPath(context.protocol['retired_robot_root']).SetActive(False)
        made = context.preparation.old.constructor_for(context.cohort.ROBOT)(stage, context.protocol['start_xyyaw'])
        attr = stage.GetPrimAtPath(context.cohort.ROBOT).CreateAttribute(
            'isaac:namespace', Sdf.ValueTypeNames.String, custom=False, variability=Sdf.VariabilityVarying)
        attr.SetDisplayName('Namespace'); attr.SetDocumentation('Namespace of the prim in Isaac Sim')
        marker = UsdGeom.Xform.Define(stage, context.cohort.MARKER).GetPrim()
        for key, value in dict(robot_root=context.cohort.ROBOT, status='fresh_static_pending_NO_PLAY',
                               cohort=context.protocol['cohort']).items(): marker.SetCustomDataByKey(key, value)
    if made != prepared['source_constructor_readback']: raise ValueError('Original source constructor does not reproduce')
    return session.ExportToString()


def canonical_display_only_review(expected_text, current_text):
    """Same-parser typed diff AND lossless canonical text check on COPIES.

    ListInfoKeys can omit Kit-registered fields in standalone USD. Never call
    their apparent absence a real scene edit or waive their whole subtree.
    After checking only the three existing Top defaults, replace those values
    in the expected ANONYMOUS copy with type-correct USD setters. The ENTIRE
    canonical layer text must then match, including opaque metadata that the
    standalone field iterator did not expose. No live layer is passed/edited.
    """
    from pxr import Sdf, Usd
    first, last = [Sdf.Layer.CreateAnonymous('H6_TEXT_AUDIT_COPY'+('.usda' if text.lstrip().startswith('#usda') else '.sdf'))
        for text in (expected_text,current_text)]
    if not first.ImportFromString(expected_text) or not last.ImportFromString(current_text):
        raise ValueError('Both saved Session layer COPIES must parse')
    changes = shared.layer_field_differences(first.ExportToString(), last.ExportToString())
    for row in changes: boundary._validate_display_row(row)
    stage = Usd.Stage.Open(first)
    for row in changes:
        # Sdf.SetInfo(default, Python float) can store a double in a float
        # attribute and change textual round-tripping. USD Set performs the
        # ORIGINAL attribute's type conversion; metadata are not copied over.
        attr = stage.GetAttributeAtPath(row['path'])
        if not attr or not attr.Set(last.GetPropertyAtPath(row['path']).default):
            raise ValueError('Cannot normalize the isolated typed display default')
    if first.ExportToString() != last.ExportToString():
        raise ValueError('Non-display canonical layer content differs, including opaque metadata')
    return dict(exact_Top_default_differences=changes,
        same_parser_field_comparison=True,full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults=True,
        no_camera_Z_orientation_clipping_or_other_subtree_exemption=True,
        original_current_saved_layer_texts_unchanged=True,actual_motion_authorized=False)


def review_recorded_boundary_get():
    """Verify saved GET evidence OFFLINE; no new scene read or physical action."""
    folder = H6/'static_boundary_GET'
    contract, live = read(folder/'GET_contract.json'), read(folder/'live_boundary.json')
    evolving = {Path(__file__).resolve(), ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py'}
    history = []
    with zipfile.ZipFile(folder/'GET_sources.zip') as archive:
        for name, digest in contract['input_sha256'].items():
            path = Path(name)
            if runtime.task.sha(path) == digest: continue
            member = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
            if path not in evolving or hashlib.sha256(archive.read(member)).hexdigest() != digest:
                raise ValueError('Unreviewed GET evidence/dependency change: '+name)
            history.append(dict(path=name, archived_member=member, original_sha256=digest))
    prepared = read(H6/'preparation.json'); stop = read(H6/'actual_Stop_boundary.json')
    expected_pins = pins(contract['input_sha256'], {str(folder/'GET_contract.json'):runtime.task.sha(folder/'GET_contract.json')})
    if (live['input_sha256'] != expected_pins or live['actual_timeline'] != boundary.STOPPED
            or live['actor'] != prepared['new_robot_root']
            or live['actual_layer_texts']['Root'] != stop['Root_text']
            or live['actual_layer_sha256']['Root'] != prepared['Root_sha256']
            or any(hashlib.sha256(live['actual_layer_texts'][k].encode()).hexdigest() != live['actual_layer_sha256'][k]
                   for k in ('Root','Session'))
            or not all(live[k] is True for k in ('source_constructor_unchanged', 'all145_source_fixed_settings_unchanged',
                'critical_physics_and_material_and_support_unchanged','same_stage_layers_timeline_throughout_GET'))
            or live['source_and_physics_readback_errors'] or live['GET_time_transactions']
            or live['protected_environment_row_count'] != 4997 or live['protected_environment_changed_paths']
            or live['actual_compact_environment_sha256'] != prepared['protected_environment_sha256']
            or live['new_static_or_route_trials'] != 0 or live['timeline_or_stage_commands'] != 0):
        raise ValueError('Own stable STOPPED GET, complete actor/environment and exact Root bytes required')
    review = canonical_display_only_review(reconstruct_preparation_session_text(),live['actual_layer_texts']['Session'])
    stable_pins = {name:digest for name,digest in contract['input_sha256'].items() if Path(name) not in evolving}
    current_pins = {str(path):runtime.task.sha(path) for path in (*evolving,
        folder/'live_boundary.json',folder/'GET_contract.json',folder/'GET_sources.zip')}
    review_pins = pins(stable_pins,current_pins)
    runtime.task.verify_pins(review_pins)
    return dict(status='SAVED_H6_GET_REVIEW_ROOT_BYTE_IDENTICAL_ONLY_3_TOP_DEFAULTS_DIFFER_NO_PHYSICAL_ADMISSION',
        actual_GET_sha256=runtime.task.sha(folder/'live_boundary.json'),
        Root_actual_bytes_identical_to_actual_completed_Stop_and_preparation=True,
        prior_Root9_and_Session10_UI_metadata_missing_fields_were_cross_registry_comparison_artifacts=True,
        prior_live_layer_hash_and_raw_fields_preserved_not_relabelled=True,
        Session_review=review, same_stopped_actor_all145_settings_13bodies_12joints_and4997_rows_verified=True,
        original_static_dispatch_failed_before_physical_operation=True,
        failed_historical_Session_transaction_cause_NOT_claimed_from_later_GET=True,
        no_old_preparation_record_or_current_layer_restored_or_overwritten=True,
        historical_GET_source_resolution=history, actual_motion_authorized=False,
        input_sha256=review_pins,
        static_continuation_requires_separate_user_launch_approval=True)


REVIEWED_STATIC_VERSION = 'MISSION_SAME_ACTOR_REVIEWED_DISPLAY_FIRST_STATIC_v1'
REVIEWED_STATIC_FOLDER = H6/'reviewed_first_static'


def reviewed_static_definition():
    """OFFLINE prospective first physical static, not a launch authorization.

    Failed dispatches and original preparation remain unchanged. Old changing
    source bytes are retained by their exact source archives. A NEW receipt
    will link the same physical preparation to reviewed display defaults;
    never overwrite the old Session hash or alter the actual display/actor.
    """
    original, history = recorded_unstarted_static()
    fresh_review = review_recorded_boundary_get()
    saved_path = H6/'static_boundary_GET/same_registry_lossless_review.json'
    saved_review = read(saved_path)
    for key in set(fresh_review)|set(saved_review):
        if key != 'input_sha256' and fresh_review.get(key) != saved_review.get(key):
            raise ValueError('Saved lossless review does not reproduce: '+key)
    evolving = {Path(__file__).resolve(), ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py'}
    with zipfile.ZipFile(H6/'static_boundary_GET/lossless_review_sources.zip') as archive:
        for name, digest in saved_review['input_sha256'].items():
            source = Path(name)
            if runtime.task.sha(source) == digest: continue
            member = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else None
            if source not in evolving or hashlib.sha256(archive.read(member)).hexdigest() != digest:
                raise ValueError('Lossless review evidence/dependency changed: '+name)
    protocol = copy.deepcopy(original)
    files = [saved_path, H6/'static_boundary_GET/lossless_review_sources.zip',
        H6/'static_runtime_sources.zip', H6/'static/dispatch_reserved.json',
        H6/'static/static_operation_error.json', H6/'static_protocol.json']
    protocol.update(version=REVIEWED_STATIC_VERSION, run_id='hospital_h6_reviewed_static_r01',
        trial_role='H6_SAME_SOURCE_FIRST_PHYSICAL_STATIC_AFTER_REVIEWED_DISPLAY_BOUNDARY',
        original_unstarted_static_history=history,
        original_static_protocol=str(H6/'static_protocol.json'),
        static_output_directory=str(REVIEWED_STATIC_FOLDER),
        reviewed_preparation_receipt=str(REVIEWED_STATIC_FOLDER/'same_actor_handoff.json'),
        saved_verified_GET=str(H6/'static_boundary_GET/live_boundary.json'),
        saved_lossless_review=str(saved_path),
        launch_requires_NEW_explicit_single_static_consent=True,
        original_approval_does_NOT_authorize_automatic_retry=True,
        status='OFFLINE_VERIFIED_SAME_ACTOR_FIRST_STATIC_DESIGN_PENDING_SEPARATE_LAUNCH_CONSENT',
        maximum_static_trials=1,maximum_Stop_calls=0,maximum_route_trials=0,
        automatic_retry=False,actual_motion_authorized=False,
        current_display_differences_recorded_NOT_reverted=True,
        original_preparation_record_and_failed_dispatch_NOT_overwritten=True,
        input_sha256=pins(fresh_review['input_sha256'],{str(p):runtime.task.sha(p) for p in files}))
    protocol.pop('definition_sha256', None)
    protocol['definition_sha256'] = shared.digest(protocol)
    return evidence.json_value(protocol)


def reviewed_static_context(protocol, path):
    path = Path(path).resolve()
    payload = {k:v for k,v in protocol.items() if k != 'definition_sha256'}
    if (path != REVIEWED_STATIC_FOLDER/'static_protocol.json' or protocol.get('version') != REVIEWED_STATIC_VERSION
            or protocol.get('definition_sha256') != shared.digest(payload)
            or (protocol.get('maximum_static_trials'),protocol.get('maximum_Stop_calls'),protocol.get('maximum_route_trials')) != (1,0,0)
            or protocol.get('automatic_retry') is not False or protocol.get('actual_motion_authorized') is not False
            or protocol.get('launch_requires_NEW_explicit_single_static_consent') is not True
            or protocol.get('original_preparation_record_and_failed_dispatch_NOT_overwritten') is not True
            or protocol.get('new_robot_root') != '/World/patient_transport_h6_cohort_01_common_preparation_r01'
            or protocol.get('static_output_directory') != str(REVIEWED_STATIC_FOLDER)
            or protocol.get('reviewed_preparation_receipt') != str(REVIEWED_STATIC_FOLDER/'same_actor_handoff.json')):
        raise ValueError('Exact prospective same-actor first-static scope required')
    # A pinned command is required at handoff. Also require the original
    # fixed queries/physical parameters, independently of descriptive fields.
    original = read(H6/'static_protocol.json')
    for key in ('new_robot_root','marker','start_xyyaw','goal_xyyaw','expected_nominal_motor_baseline',
        'protected_environment_sha256','supports','support_top_m','additional_sampled_query_limits',
        'sampled_cooked_support_queries','source_actor_node_paths','approved_disabled_subfloor_paths',
        'original_disabled_collider_paths','additional_disabled_subfloor_probes',
        'query_known_enabled_colliders','auxiliary_probe_known_nonself_colliders',
        'unchanged_original_static_limits','unchanged_original_static_guards','required_display_settings'):
        if protocol.get(key) != original[key]: raise ValueError('Original physical/query definition changed: '+key)
    return static_binding(protocol,path,REVIEWED_STATIC_FOLDER,Path(protocol['reviewed_preparation_receipt']))


def load_reviewed_static(path):
    protocol = read(path)
    context = reviewed_static_context(protocol,path)
    runtime.task.verify_pins(protocol['input_sha256'])
    return context


def review_live_display_texts(recorded, current):
    """Exact Root bytes + lossless Session, not a blanket camera exemption."""
    if current['Root'] != recorded['Root']:
        raise ValueError('Original Root bytes changed; no physical field or UI-field waiver')
    return canonical_display_only_review(recorded['Session'],current['Session'])


def same_actor_handoff(prepared, initial_GET, current_texts, display_review):
    """Build a NEW receipt on plain copies; no live stage or old record write."""
    if (initial_GET['actual_timeline'] != boundary.STOPPED
            or initial_GET['actor'] != prepared['new_robot_root']
            or current_texts['Root'] != initial_GET['actual_layer_texts']['Root']
            or display_review['full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults'] is not True):
        raise ValueError('Same own stopped source and exact reviewed display-only handoff required')
    result = copy.deepcopy(prepared)
    result['Root_sha256'] = hashlib.sha256(current_texts['Root'].encode()).hexdigest()
    result['Session_sha256'] = hashlib.sha256(current_texts['Session'].encode()).hexdigest()
    result['reviewed_display_handoff_NOT_new_staging'] = dict(
        original_preparation=str(H6/'preparation.json'),original_preparation_sha256=runtime.task.sha(H6/'preparation.json'),
        actual_GET=str(H6/'static_boundary_GET/live_boundary.json'),
        actual_GET_sha256=runtime.task.sha(H6/'static_boundary_GET/live_boundary.json'),
        original_prepared_Session_sha256=prepared['Session_sha256'],
        original_preparation_fields_NOT_overwritten=True,display_defaults_NOT_restored=True,
        exact_existing_Top_default_differences_since_GET=display_review['exact_Top_default_differences'],
        no_Stop_restaging_display_refresh_reset_or_trial=True)
    return result


def require_reviewed_static_consent(authorized, approval):
    if (authorized is not True or not isinstance(approval,dict)
            or approval.get('answer_ko') != '검증 후 같은 로봇 정적 최초 1회 진행 승인'
            or not isinstance(approval.get('question_item_id'),str) or not approval['question_item_id'].startswith('call_')):
        raise ValueError('NEW explicit same-actor ONE static launch consent required before SDK access')


async def run_reviewed_static_once(path, *, authorized=False, approval=None):
    require_reviewed_static_consent(authorized,approval)
    context = load_reviewed_static(path)
    output = context.static_folder;dispatch=output/'dispatch_reserved.json'
    if dispatch.exists():
        print('[H6 REVIEWED STATIC] Existing dispatch retained; no new action or retry.',flush=True)
        return dict(status='PRIOR_REVIEWED_STATIC_DISPATCH_RETAINED_NO_NEW_ACTION')
    no_other_task()
    recorded_unstarted_static()
    evidence.write_exclusive(dispatch,dict(protocol=str(context.protocol_path),protocol_sha256=runtime.task.sha(path),
        explicit_NEW_single_static_consent=approval, maximum_static_trials=1,maximum_Stop_calls=0,maximum_route_trials=0,
        no_actor_display_or_physics_action_yet=True,automatic_retry=False))
    phase='same_stopped_source_display_boundary';sdk=None;initial=None
    try:
        sdk=runtime.live_sdk();usd=sdk.omni.usd.get_context();stage=usd.get_stage()
        timeline=sdk.omni.timeline.get_timeline_interface();app=sdk.omni.kit.app.get_app()
        if stage is None or evidence.timeline_state(timeline)!=boundary.STOPPED:
            raise ValueError('Same existing STOPPED-zero scene required; no Stop/reset to create it')
        def texts():
            value=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
            if sum(len(v.encode()) for v in value.values())>2*1024*1024:raise ValueError('Bounded current layer snapshot required')
            return value
        initial=texts();saved_GET=read(context.protocol['saved_verified_GET']);prepared=read(H6/'preparation.json')
        evidence.write_exclusive(output/'actual_layers_before_preflight.json',dict(actual_layer_texts=initial,
            actual_timeline=evidence.timeline_state(timeline),no_timeline_or_stage_command=True))
        def same_source_boundary():
            if usd.get_stage()!=stage or evidence.timeline_state(timeline)!=boundary.STOPPED:
                raise ValueError('Actual stage/timeline changed during reviewed preflight')
        # Drain ONLY existing UI updates, never frame/select/refresh/restore.
        # Every newly seen text pair is fully audited, including opaque fields.
        previous=None;stable=0;began=time.monotonic();review=None
        for _ in range(60):
            same_source_boundary();current=texts()
            if current!=previous:review=review_live_display_texts(saved_GET['actual_layer_texts'],current)
            stable=stable+1 if current==previous else 0;previous=current
            if stable>=3:break
            if time.monotonic()-began>15.:raise TimeoutError('Reviewed display boundary did not settle; no repair or Play')
            await app.next_update_async()
        else:raise TimeoutError('Reviewed display boundary did not settle within60 UI updates')
        same_source_boundary()
        import carb.settings
        if any(carb.settings.get_settings().get(k)!=v for k,v in context.protocol['required_display_settings'].items()):
            raise ValueError('Original renderer/physics preferences changed; no setting edit or Play')
        handoff=same_actor_handoff(prepared,saved_GET,current,review)
        evidence.write_exclusive(context.preparation.LIVE,handoff)
        evidence.write_exclusive(output/'reviewed_display_boundary.json',dict(actual_layer_texts=current,
            actual_timeline=evidence.timeline_state(timeline),display_review=review,
            no_live_camera_or_layer_or_actor_edit=True))
        # The ORIGINAL static loop independently validates the source seed,
        # same-SDK145 settings, all4997 rows, material/physics and new receipt
        # boundary before it reserves its ONE physical attempt and calls Play.
        phase='original_zero_command_first_physical_static'
        report=await context.operations.static(context)
        replay=None
        if report['verdict']=='PASS':
            phase='same_trial_raw_evidence_replay'
            _,_,raw=context.operations.replay_static(context);replay=raw['verdict']
        result=dict(status='ONE_SAME_H6_SOURCE_FIRST_STATIC_FINISHED_NO_DRIVE_ADMISSION',
            static_verdict=report['verdict'],own_raw_replay_verdict=replay,
            actual_timeline=evidence.timeline_state(timeline),new_route_trials=0,additional_Stop_calls=0,
            no_reset_restaging_display_refresh_or_automatic_retry=True,result=str(context.stationary.RESULT))
        evidence.write_exclusive(output/'static_only_closeout.json',shared.normalize_runtime_evidence(result))
        print('[H6 REVIEWED STATIC] Finished:',result,flush=True)
        return result
    except BaseException as error:
        diagnostic=dict(phase=phase,error=type(error).__name__+': '+str(error),automatic_retry=False,
            no_Stop_reset_restaging_display_refresh_or_route=True)
        if sdk is not None:
            try:
                observed=sdk.omni.usd.get_context().get_stage()
                diagnostic['actual_timeline']=evidence.timeline_state(sdk.omni.timeline.get_timeline_interface())
                if observed is not None:
                    current=dict(Root=observed.GetRootLayer().ExportToString(),Session=observed.GetSessionLayer().ExportToString())
                    if sum(len(v.encode()) for v in current.values())<=2*1024*1024:
                        diagnostic['actual_current_layer_texts']=current
                        if initial is not None:diagnostic['exact_changes_since_entry']={k:shared.layer_field_differences(initial[k],current[k]) for k in ('Root','Session')}
            except Exception as secondary:diagnostic['readback_error']=type(secondary).__name__+': '+str(secondary)
        evidence.write_exclusive(output/'static_operation_error.json',shared.normalize_runtime_evidence(diagnostic))
        raise


def schedule_reviewed_static(path, *, authorized=False, approval=None):
    require_reviewed_static_consent(authorized,approval)
    if (REVIEWED_STATIC_FOLDER/'dispatch_reserved.json').exists():
        print('[H6 REVIEWED STATIC] Existing dispatch retained; do not rerun.',flush=True);return None
    no_other_task()
    def done(future):
        if future.cancelled():print('[H6 REVIEWED STATIC] Cancelled; preserve state, no retry.',flush=True)
        elif future.exception() is not None:
            import traceback
            error=future.exception();traceback.print_exception(type(error),error,error.__traceback__)
            print('[H6 REVIEWED STATIC] Aborted; exact boundary/error retained. No retry or timeline manipulation.',flush=True)
    future=asyncio.ensure_future(run_reviewed_static_once(path,authorized=True,approval=approval))
    future.add_done_callback(done);builtins.ijcas_hospital_latest_task_experiment=future
    print('[H6 REVIEWED STATIC] SAME actor -> checked display-only handoff -> ONE original zero-command static. NO Stop/restaging/refresh/route/retry.',flush=True)
    return future


def export_reviewed_static_definition():
    protocol=reviewed_static_definition();target=REVIEWED_STATIC_FOLDER/'static_protocol.json'
    if target.exists():raise ValueError('Existing prospective revised definition retained; no overwrite')
    evidence.write_exclusive(target,protocol)
    with zipfile.ZipFile(REVIEWED_STATIC_FOLDER/'runtime_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(protocol['input_sha256']):
            path=Path(name)
            if path.suffix=='.py' and path.is_relative_to(ROOT):archive.write(path,str(path.relative_to(ROOT)))
    return target


def verify_reviewed_executed_pins(mapping):
    """Historical source resolution ONLY; never authority to rerun a command.

    The consumed command archived the exact runtime/test bytes. All physical
    kernels, inputs and actual records must still match their recorded bytes.
    """
    evolving={Path(__file__).resolve(),ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py'}
    resolved=[]
    with zipfile.ZipFile(REVIEWED_STATIC_FOLDER/'runtime_sources.zip') as archive:
        for name,digest in mapping.items():
            path=Path(name)
            if runtime.task.sha(path)==digest:continue
            member=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else None
            if path not in evolving or hashlib.sha256(archive.read(member)).hexdigest()!=digest:
                raise ValueError('Executed static evidence or physical dependency changed: '+name)
            resolved.append(dict(path=name,executed_sha256=digest,exact_archived_member=member))
    return dict(verified_pins=len(mapping),historical_source_resolution=resolved,
        archived_source_is_NOT_launch_authority=True)


def saved_post_static_release_review(raw):
    """Exact saved-snapshot gate, NOT a fresh live-state or motion check.

    Static force/support acceptance and the original stricter stopped release
    gate are different requirements. Never zero a recorded residual velocity
    or relax the original5mm/s threshold to manufacture a successful replay.
    """
    actual=raw['actual_post_static_tensor_snapshot']
    shapes=dict(transform=(7,),velocity=(6,),positions=(12,),joint_velocities_rad_s=(12,),
        joint_position_targets_rad=(12,),joint_velocity_targets_rad_s=(12,),
        all_thirteen_link_world_transforms_xyzw=(13,7),all_thirteen_link_velocities_at_CoM=(13,6))
    for name,shape in shapes.items():
        values=np.asarray(actual[name],float)
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError('Finite complete recorded post-static tensor required: '+name)
    names=actual['names'];baseline=raw['runtime_motor_baseline']
    if (len(names)!=12 or len(set(names))!=12 or set(names)!=set(baseline['joint_names'])
            or len(actual['link_names'])!=13 or len(set(actual['link_names']))!=13):
        raise ValueError('Complete original twelve-joint/thirteen-link correspondence required')
    qx,qy,qz,qw=actual['transform'][3:]
    yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    normalized=dict(raw,final_actual_world_xyyaw=[*actual['transform'][:2],yaw],
        final_actual_joint_names=names,final_actual_twelve_joint_positions_rad=actual['positions'],
        final_actual_twelve_joint_velocities_rad_s=actual['joint_velocities_rad_s'],
        passive_post_static_link_readback=actual)
    limits=shared.preparation.old.cached.continuity.LIVE_TOLERANCES
    gains_match=True
    for key in ('kp','kd','max_efforts'):
        target=dict(zip(baseline['joint_names'],baseline[key]))
        gains_match &= actual['gains'][key]==[target[name] for name in names]
    speed=math.hypot(*actual['velocity'][:2])
    checks=dict(root_quaternion_normalized=abs(math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)-1.)<=1e-4,
        original_all_twelve_motor_settings_match=bool(gains_match),
        horizontal_speed_within_original_stop_limit=speed<=limits['horizontal_stop_m_s'])
    links=np.asarray(actual['all_thirteen_link_world_transforms_xyzw'],float)
    checks['all_thirteen_recorded_link_quaternions_normalized']=bool(
        np.allclose(np.linalg.norm(links[:,3:],axis=1),1.,rtol=0.,atol=1e-4))
    error=None;original_result=None
    try:original_result=shared.preparation.old.validate_actual_state(normalized,actual)
    except ValueError as failure:error=type(failure).__name__+': '+str(failure)
    if (error is None)!=all(checks.values()):
        raise ValueError('Recorded metric diagnosis disagrees with original retained-state gate')
    return normalized,dict(status='SAVED_POST_STATIC_RELEASE_PASS' if error is None else 'SAVED_POST_STATIC_RELEASE_HOLD',
        checks=checks,original_gate_error=error,original_gate_result=original_result,
        horizontal_speed_m_s=speed,original_horizontal_stop_limit_m_s=limits['horizontal_stop_m_s'],
        excess_speed_m_s=max(0.,speed-limits['horizontal_stop_m_s']),
        saved_snapshot_compared_to_ITSELF_not_fresh_live_state=True,
        pose_joint_correspondence_not_an_observed_current_live_continuity_proof=True,
        recorded_tensor_and_original_tolerances_NOT_changed=True,actual_motion_authorized=False)


def review_completed_reviewed_static():
    """Replay the executed static OFFLINE and retain a distinct release HOLD.

    No SDK import, timeline command, live readback, new trial or record repair.
    Earlier dispatch/error/report/CSV and the command's source ZIP stay exact.
    """
    folder=REVIEWED_STATIC_FOLDER;path=folder/'static_protocol.json';protocol=read(path)
    context=reviewed_static_context(protocol,path)
    # The private preparation namespace resolves exact EXECUTED source bytes,
    # not arbitrary current physical code or an exemption of original hashes.
    context.preparation.verify=verify_reviewed_executed_pins
    raw=read(context.stationary.RESULT);dispatch=read(folder/'dispatch_reserved.json')
    approval=dispatch['explicit_NEW_single_static_consent'];require_reviewed_static_consent(True,approval)
    if (dispatch['protocol']!=str(path) or dispatch['protocol_sha256']!=runtime.task.sha(path)
            or raw['static_protocol_sha256']!=runtime.task.sha(path)
            or raw['verdict']!='PASS' or not raw['checks'] or not all(raw['checks'].values())
            or raw['trace_sha256']!=runtime.task.sha(context.stationary.TRACE)):
        raise ValueError('Exact completed own static PASS and unchanged actual dispatch/CSV required')
    pins_review=verify_reviewed_executed_pins(raw['input_sha256'])
    normalized,release=saved_post_static_release_review(raw)
    replay_error=None;replay_result=None
    try:_,_,replay_result=context.operations.replay_static(context)
    except ValueError as failure:replay_error=type(failure).__name__+': '+str(failure)
    if replay_error!=release['original_gate_error']:
        raise ValueError('Original raw CSV/query/contact/clock replay failed at another requirement: '+str(replay_error))
    import csv
    with context.stationary.TRACE.open(newline='',encoding='utf-8') as stream:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    physical=shared.completed_static.physical.evaluate_static(rows,protocol['start_xyyaw'],265.,
        raw['contact_report_counts_by_category'],raw['contact_callback_errors'])
    support=context.support_evaluator(protocol,raw['actual_sampled_cooked_support_queries'])
    probes=context.stationary.evaluate_subfloor_queries(protocol,raw['actual_sampled_cooked_support_queries'],
        raw['actual_disabled_subfloor_probes'])
    if (physical['verdict']!='PASS' or not all(support['checks'].values()) or not all(probes['checks'].values())
            or shared.normalize_runtime_evidence(physical)!=raw['physical_stationary_evaluation']):
        raise ValueError('Saved original static evaluation does not reproduce exactly')
    files=[path,context.stationary.RESULT,context.stationary.TRACE,folder/'same_actor_handoff.json',
        folder/'dispatch_reserved.json',folder/'single_attempt_reserved.json',folder/'static_operation_error.json',
        folder/'runtime_sources.zip',Path(__file__),ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py']
    return shared.normalize_runtime_evidence(dict(
        status='ACTUAL_H6_STATIC_PASS_RAW_PHYSICAL_REPLAY_PASS_RETAINED_RELEASE_HOLD' if replay_error else
            'ACTUAL_H6_STATIC_PASS_AND_SAVED_RELEASE_REPLAY_PASS_NOT_DRIVE_ADMISSION',
        actual_static_verdict=raw['verdict'],sample_count=len(rows),physical_evaluation_reproduced_exactly=True,
        original_support_and_four_probe_replay_PASS=True,original_full_replay_error_reproduced=replay_error,
        release_gate=release,physical_stationary_evaluation=physical,
        last_recorded_PAUSED_timeline=raw['actual_final_timeline'],
        source145_environment4997_contact_counts_clocks_gains_and_complete_tensor_pass_before_original_release_gate=True,
        executed_input_resolution=pins_review,explicit_executed_approval=approval,
        all_original_actual_evidence_unchanged=True,no_fresh_live_readback_or_trial=True,
        additional_Stop_static_or_route_trials=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={str(p):runtime.task.sha(p) for p in files}))


def retained_sdk_snapshot_metrics(saved,current):
    """GET-only differences, not a waiver or a new static/release verdict."""
    if current['names']!=saved['names'] or current['link_names']!=saved['link_names']:
        raise ValueError('Exact saved twelve-joint/thirteen-link correspondence required')
    def array(snapshot,key,shape):
        result=np.asarray(snapshot[key],float)
        if result.shape!=shape or not np.isfinite(result).all():raise ValueError('Complete finite snapshot required: '+key)
        return result
    root0=array(saved,'transform',(7,));root1=array(current,'transform',(7,))
    links0=array(saved,'all_thirteen_link_world_transforms_xyzw',(13,7))
    links1=array(current,'all_thirteen_link_world_transforms_xyzw',(13,7))
    def angles(a,b):
        n0=np.linalg.norm(a,axis=-1);n1=np.linalg.norm(b,axis=-1)
        if not np.allclose(n0,1.,rtol=0.,atol=1e-4) or not np.allclose(n1,1.,rtol=0.,atol=1e-4):
            raise ValueError('Normalized observed quaternions required')
        return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)/(n0*n1)),0.,1.)))
    velocity=array(current,'velocity',(6,));old_velocity=array(saved,'velocity',(6,))
    speed=math.hypot(*velocity[:2]);limit=shared.preparation.old.cached.continuity.LIVE_TOLERANCES['horizontal_stop_m_s']
    return dict(root_position_difference_m=float(np.linalg.norm(root1[:3]-root0[:3])),
        root_orientation_difference_deg=float(angles(root1[3:],root0[3:])),
        maximum_joint_position_difference_rad=float(np.max(abs(array(current,'positions',(12,))-array(saved,'positions',(12,))))),
        maximum_joint_velocity_difference_rad_s=float(np.max(abs(array(current,'joint_velocities_rad_s',(12,))-array(saved,'joint_velocities_rad_s',(12,))))),
        maximum_link_position_difference_m=float(np.max(np.linalg.norm(links1[:,:3]-links0[:,:3],axis=1))),
        maximum_link_orientation_difference_deg=float(np.max(angles(links1[:,3:],links0[:,3:]))),
        maximum_root_velocity_difference=float(np.max(abs(velocity-old_velocity))),
        current_horizontal_speed_m_s=speed,original_horizontal_stop_limit_m_s=limit,
        current_original_stopped_speed_check=speed<=limit,
        actual_static_outcome_NOT_changed=True,metrics_are_NOT_full_environment_or_drive_admission=True)


def retained_sdk_get_observation(sdk,raw):
    """Existing-view GETs only. No constructor/init, control or timeline call."""
    import hospital_mission_execution as control
    stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    before=evidence.timeline_state(timeline)
    if stage is None:raise ValueError('No current stage; do not open/reload or Play to obtain one')
    if before['playing'] or before['stopped']:raise ValueError('Existing retained PAUSED actor required; no timeline recovery')
    actor='/World/patient_transport_h6_cohort_01_common_preparation_r01'
    prim=stage.GetPrimAtPath(actor)
    if not prim.IsValid() or not prim.IsActive() or not prim.IsLoaded():raise ValueError('Exact existing H6 actor unavailable; no restaging')
    layers_before=(stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString())
    def same_context():
        if sdk.omni.usd.get_context().get_stage()!=stage or evidence.timeline_state(timeline)!=before:
            raise RuntimeError('Stage or PAUSED clock changed during GET; no restoration')
    simulation=sdk.SimulationManager.get_physics_sim_view()
    if simulation is None:raise ValueError('No retained physics view; do not Play or initialize one')
    view=simulation.create_articulation_view(actor+'/base_link')
    if not bool(view.check()):raise ValueError('Retained physics handle invalid; no recovery')
    snapshot=shared.preparation.old.tensor_snapshot(view)
    def blocked_encoder(*args):raise RuntimeError('GET-only encoder may never be invoked')
    try:
        adapter=control.RetainedTensorDriveAdapter(view,raw['runtime_motor_baseline'],encode=blocked_encoder,
            check_context=same_context,paused_check=lambda:not timeline.is_playing() and not timeline.is_stopped(),
            physics_stamp=before['current_time_s'],authorized=False)
        confirmation=adapter.confirmation(before['current_time_s'])
    except Exception as failure:
        failure.retained_sdk_snapshot_before_rejection=snapshot
        raise
    after_snapshot=shared.preparation.old.tensor_snapshot(view);same_context()
    layers_after=(stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString())
    return shared.normalize_runtime_evidence(dict(status='CURRENT_RETAINED_H6_SDK_GET_COMPLETE_NOT_DRIVE_ADMISSION',
        actor=actor,timeline=before,saved_static_timeline=raw['actual_final_timeline'],
        same_saved_PAUSED_timeline=before==raw['actual_final_timeline'],
        actual_tensor_snapshot=after_snapshot,tensor_stable_during_GET=snapshot==after_snapshot,
        metrics=retained_sdk_snapshot_metrics(raw['actual_post_static_tensor_snapshot'],after_snapshot),
        original_motor_baseline_backend_readback_verified=True,observed_hold_confirmation=confirmation,
        readback_is_NOT_an_actual_hold_capture_release_or_force_test=True,
        Root_Session_unchanged_during_GET=layers_before==layers_after,
        layer_sha256_before=[hashlib.sha256(s.encode()).hexdigest() for s in layers_before],
        layer_sha256_after=[hashlib.sha256(s.encode()).hexdigest() for s in layers_after],
        complete4997_environment_and145_source_fields_NOT_rechecked=True,
        any_Play_Stop_Pause_reset_control_gain_or_scene_command=False,new_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False))


SDK_GET_FOLDER=H6/'retained_sdk_preflight_GET_r01'


def export_retained_sdk_get():
    """Append-only OFFLINE packet for ONE genuinely required SDK GET."""
    import hospital_mission_execution as control
    raw_path=REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json'
    raw=read(raw_path)
    if raw['verdict']!='PASS' or runtime.task.sha(raw_path)!='7d0371665b84dc5ac4ea81132be75e7939eb6e4010a55842ac88cebb1c39123b':
        raise ValueError('Exact preserved first H6 static record required')
    target=SDK_GET_FOLDER/'GET_contract.json'
    if target.exists():raise ValueError('Existing GET packet retained; do not replace or repeat')
    sources=[Path(__file__),Path(control.__file__),ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',
        ROOT/'src/patient_initial_stability/test/test_hospital_timed_path_execution.py']
    contract=dict(version='H6_RETAINED_SDK_TARGET_GAIN_AND_TENSOR_GET_ONLY_v1',
        result=str(SDK_GET_FOLDER/'live_SDK_readback.json'),static_record=str(raw_path),
        actor='/World/patient_transport_h6_cohort_01_common_preparation_r01',
        maximum_Play_Stop_Pause_reset_static_route_or_gain_commands=0,
        scope='existing PAUSED actor/SDK getter availability/motor targets/gains/12joints/13links/metric differences ONLY',
        no_full_environment_or_support_or_release_or_drive_admission=True,
        actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={str(p):runtime.task.sha(p) for p in [raw_path,*sources]})
    SDK_GET_FOLDER.mkdir(parents=True)
    evidence.write_exclusive(target,contract)
    with zipfile.ZipFile(SDK_GET_FOLDER/'executed_GET_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for p in sources:archive.write(p,str(p.relative_to(ROOT)))
    return target


def run_retained_sdk_get(path,*,read_only_authorized=False,_sdk=None):
    """Run once in the user's ALREADY OPEN app. Never starts a trial."""
    if read_only_authorized is not True:raise ValueError('Explicit GET-only entry required; no SDK import')
    path=Path(path).resolve()
    if path!=SDK_GET_FOLDER/'GET_contract.json':raise ValueError('Exact reviewed GET packet required')
    contract=read(path)
    if (contract['version']!='H6_RETAINED_SDK_TARGET_GAIN_AND_TENSOR_GET_ONLY_v1'
            or contract['maximum_Play_Stop_Pause_reset_static_route_or_gain_commands']!=0
            or contract['actual_motion_authorized'] is not False):raise ValueError('GET-only contract required')
    runtime.task.verify_pins(contract['input_sha256'])
    result=Path(contract['result']);dispatch=path.parent/'GET_reserved.json'
    if dispatch.exists() or result.exists():
        print('[H6 SDK GET] Prior current-state readback/dispatch retained; no duplicate.',flush=True)
        return read(result) if result.exists() else None
    for name in boundary.ACTIVE_TASKS:
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():raise ValueError('Another research task is active; no competing GET')
    evidence.write_exclusive(dispatch,dict(contract=str(path),contract_sha256=runtime.task.sha(path),GET_only=True,no_trial_or_control=True))
    began=time.monotonic();report=dict(status='READ_ONLY_GET_ERROR_NO_RECOVERY',new_trial_count=0,
        any_Play_Stop_Pause_reset_control_gain_or_scene_command=False,actual_motion_authorized=False,use_for_final_statistics=False)
    sdk=None;raw=None
    try:
        sdk=runtime.live_sdk() if _sdk is None else _sdk
        raw=read(contract['static_record']);report.update(retained_sdk_get_observation(sdk,raw))
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)
        if hasattr(error,'retained_sdk_snapshot_before_rejection'):
            report['actual_tensor_snapshot_before_rejection']=error.retained_sdk_snapshot_before_rejection
        if sdk is not None:
            try:report['actual_timeline_on_error']=evidence.timeline_state(sdk.omni.timeline.get_timeline_interface())
            except Exception as secondary:report['timeline_readback_error']=type(secondary).__name__+': '+str(secondary)
    report.update(wall_time_s=time.monotonic()-began,contract=str(path),contract_sha256=runtime.task.sha(path))
    evidence.write_exclusive(result,shared.normalize_runtime_evidence(report))
    print('=== H6 RETAINED SDK / TARGET / GAIN GET ONLY ===',flush=True)
    print('status:',report['status'],'error:',report.get('error'),flush=True)
    if 'metrics' in report:print('saved/current differences + separate stopped-speed check:',report['metrics'],flush=True)
    print('saved:',result,'; NO Play/Stop/Pause/reset/gain/target/scene command or trial.',flush=True)
    return report


ZERO_SETTLE_FOLDER=H6/'retained_zero_command_settling_r01'
ZERO_SETTLE_APPROVAL=dict(question_item_id='call_9143c780a44c46daa3294c41fe4d4b7f:0',
    answer_ko='코드 검증 후 0명령 안정화·정지 확인 1회 승인')
REVIEWED_ZERO_FOLDER=ZERO_SETTLE_FOLDER/'first_observation_after_display_review_r01'
REVIEWED_ZERO_APPROVAL=dict(question_item_id='call_1d9ccb85510b4bc7b57b9ebfb274c0b2:0',
    answer_ko='검증 후 미시작 0명령 확인 1회 진행')
ZERO_DISPLAY_PATHS=('/OmniverseKit_Top.horizontalAperture','/OmniverseKit_Top.verticalAperture')
TOP_XY_DISPLAY_PATHS=(*ZERO_DISPLAY_PATHS,'/OmniverseKit_Top.xformOp:transform')
TOP_XY_DISPLAY_POLICY='EXACT_TYPED_TOP_XY_APERTURES_ONLY_NO_Z_ROTATION_CLIP_METADATA_v1'


def verified_retained_sdk_get():
    """Resolve executed GET sources in their archive, never waive old pins."""
    path=SDK_GET_FOLDER/'live_SDK_readback.json';result=read(path)
    contract_path=SDK_GET_FOLDER/'GET_contract.json';contract=read(contract_path)
    if (result.get('status')!='CURRENT_RETAINED_H6_SDK_GET_COMPLETE_NOT_DRIVE_ADMISSION'
            or result.get('error') is not None or result['contract_sha256']!=runtime.task.sha(contract_path)
            or not all(result[k] is True for k in ('same_saved_PAUSED_timeline','tensor_stable_during_GET',
                'Root_Session_unchanged_during_GET','original_motor_baseline_backend_readback_verified'))
            or result['observed_hold_confirmation'] is not False
            or result['new_trial_count']!=0 or result['actual_motion_authorized'] is not False
            or result['any_Play_Stop_Pause_reset_control_gain_or_scene_command'] is not False):
        raise ValueError('Own completed unchanged PAUSED GET required, not a new physical verdict')
    with zipfile.ZipFile(SDK_GET_FOLDER/'executed_GET_sources.zip') as archive:
        for name,expected in contract['input_sha256'].items():
            file=Path(name)
            if runtime.task.sha(file)==expected:continue
            member=str(file.relative_to(ROOT))
            if member not in archive.namelist() or hashlib.sha256(archive.read(member)).hexdigest()!=expected:
                raise ValueError('Executed GET source/input unavailable: '+name)
    return path,result


def zero_settle_definition():
    """OFFLINE definition of the separately approved fixed3+2s observation."""
    path,current=verified_retained_sdk_get()
    static=REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json'
    if runtime.task.sha(static)!='7d0371665b84dc5ac4ea81132be75e7939eb6e4010a55842ac88cebb1c39123b':
        raise ValueError('Original physical static must remain immutable')
    source_files=[Path(__file__),ROOT/'studies/IJCAS/scripts/hospital_mission_execution.py',
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',
        ROOT/'studies/IJCAS/scripts/hospital_private_loader.py']
    inputs=[path,SDK_GET_FOLDER/'GET_contract.json',SDK_GET_FOLDER/'executed_GET_sources.zip',static,
        H6/'actual_common_environment_rows.json',H6/'reviewed_first_static/static_protocol.json',
        H6/'reviewed_first_static/same_actor_handoff.json',*source_files]
    return dict(version='H6_RETAINED_FIXED_3_PLUS_2_SECOND_ZERO_COMMAND_SETTLING_v1',
        approval=ZERO_SETTLE_APPROVAL,actor=current['actor'],required_PAUSED_timeline=current['timeline'],
        required_layer_sha256=current['layer_sha256_after'],required_tensor_snapshot=current['actual_tensor_snapshot'],
        original_static_record=str(static),prior_GET_record=str(path),settling_s=3.,recording_s=2.,
        maximum_control_gap_s=.25,maximum_physics_duration_s=5.25,maximum_wall_time_s=120.,
        stop_speed_limit_m_s=.005,original_static_limits=dict(shared.stationary.LIMITS),
        maximum_Play_calls=1,maximum_Stop_reset_restaging_route_or_gain_changes=0,
        automatic_retry=False,actual_drive_admission=False,use_for_final_statistics=False,
        output=str(ZERO_SETTLE_FOLDER),input_sha256={str(p):runtime.task.sha(p) for p in inputs})


def export_zero_settle():
    protocol=zero_settle_definition();path=ZERO_SETTLE_FOLDER/'protocol.json'
    if path.exists():raise ValueError('Existing single observation packet preserved; no overwrite')
    evidence.write_exclusive(path,protocol)
    with zipfile.ZipFile(path.parent/'executed_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for file in protocol['input_sha256']:
            if file.endswith('.py'):archive.write(file,str(Path(file).relative_to(ROOT)))
    return path


def reviewed_zero_settle_definition():
    """Continue an UNSTARTED observation after exact two-default review."""
    prior_path=ZERO_SETTLE_FOLDER/'zero_settling_readback.json';prior=read(prior_path)
    old=read(ZERO_SETTLE_FOLDER/'protocol.json')
    if (prior['new_zero_command_observations']!=0 or prior['sample_count']!=0
            or prior.get('automatic_Play_requested',False) is not False
            or prior['actual_final_timeline']!=old['required_PAUSED_timeline']
            or prior['pause_confirmation']['pause_requested_by_this_call'] is not False
            or prior['error']!='ValueError: Current GET layer boundary changed; no waiver or repair'):
        raise ValueError('Only the proven pre-Play layer-boundary abort may continue; no physical retry')
    with zipfile.ZipFile(ZERO_SETTLE_FOLDER/'executed_sources.zip') as archive:
        for file,h in old['input_sha256'].items():
            if runtime.task.sha(file)==h:continue
            name=str(Path(file).relative_to(ROOT))
            if name not in archive.namelist() or hashlib.sha256(archive.read(name)).hexdigest()!=h:
                raise ValueError('Original unstarted dispatch executing source/input changed: '+file)
    path=ZERO_SETTLE_FOLDER/'preplay_layer_fields_GET.json';actual=read(path)
    if (actual['timeline']!=old['required_PAUSED_timeline'] or actual['no_timeline_or_control_command'] is not True
            or actual['actual_hashes'][0]!=old['required_layer_sha256'][0]
            or actual['expected_hashes']!=old['required_layer_sha256']):
        raise ValueError('Exact same PAUSED Root/current field GET required; no inferred restoration')
    for name,h in zip(('Root','Session'),actual['actual_hashes']):
        if hashlib.sha256(actual['actual_layer_texts'][name].encode()).hexdigest()!=h:raise ValueError('Actual field GET bytes changed')
    base=read(REVIEWED_STATIC_FOLDER/'reviewed_display_boundary.json')['actual_layer_texts']['Session']
    if hashlib.sha256(base.encode()).hexdigest()!=old['required_layer_sha256'][1]:raise ValueError('Exact former Session text required')
    review=canonical_display_only_review(base,actual['actual_layer_texts']['Session'])
    if {r['path'] for r in review['exact_Top_default_differences']}!=set(ZERO_DISPLAY_PATHS):
        raise ValueError('Exactly the two actually reviewed aperture defaults required')
    result=zero_settle_definition()
    result.update(version='H6_UNSTARTED_ZERO_OBSERVATION_EXACT_APERTURE_HANDOFF_v1',approval=REVIEWED_ZERO_APPROVAL,
        output=str(REVIEWED_ZERO_FOLDER),required_layer_sha256=actual['actual_hashes'],
        reviewed_display_boundary=dict(reference_Session_text=actual['actual_layer_texts']['Session'],
            unchanged_Root_sha256=actual['actual_hashes'][0],reviewed_default_paths=list(ZERO_DISPLAY_PATHS),
            only_typed_aperture_default_differences_NOT_subtree_or_physics_exemption=True),
        original_preplay_abort_preserved=str(prior_path),original_physical_observation_NOT_started=True,
        exact_actual_display_review=review)
    for file in (prior_path,ZERO_SETTLE_FOLDER/'protocol.json',ZERO_SETTLE_FOLDER/'single_observation_reserved.json',
            ZERO_SETTLE_FOLDER/'executed_sources.zip',path):result['input_sha256'][str(file)]=runtime.task.sha(file)
    return result


def export_reviewed_zero_settle():
    protocol=reviewed_zero_settle_definition();path=REVIEWED_ZERO_FOLDER/'protocol.json'
    if path.exists():raise ValueError('Existing reviewed observation packet retained; no replacement')
    evidence.write_exclusive(path,protocol)
    with zipfile.ZipFile(path.parent/'executed_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for file in protocol['input_sha256']:
            if file.endswith('.py'):archive.write(file,str(Path(file).relative_to(ROOT)))
    return path


def reviewed_display_paths(protocol):
    """An explicit prospective policy, never a blanket camera subtree waiver.

    Existing zero/static packets retain their original TWO-default policy.
    A newly frozen route may declare XY navigation as well; the shared typed
    matrix validator still requires unchanged orientation/Z/affine values.
    """
    binding=protocol['reviewed_display_boundary']
    paths=binding['reviewed_default_paths']
    if paths==list(ZERO_DISPLAY_PATHS) and binding.get('policy') is None:return ZERO_DISPLAY_PATHS
    if paths==list(TOP_XY_DISPLAY_PATHS) and binding.get('policy')==TOP_XY_DISPLAY_POLICY:return TOP_XY_DISPLAY_PATHS
    raise ValueError('Exact prospective typed Top display policy required')


def review_zero_display_layers(protocol,texts):
    """Exact Root and lossless Session except the explicitly declared defaults."""
    binding=protocol['reviewed_display_boundary'];allowed=reviewed_display_paths(protocol)
    if hashlib.sha256(texts['Root'].encode()).hexdigest()!=binding['unchanged_Root_sha256']:
        raise ValueError('Root or reviewed display scope changed; no Play')
    review=canonical_display_only_review(binding['reference_Session_text'],texts['Session'])
    if any(row['path'] not in allowed or row['field']!='default'
            for row in review['exact_Top_default_differences']):
        raise ValueError('Only reviewed aperture defaults may differ; no XY/rotation/Z/clip/physics waiver')
    return review


def evaluate_fixed_zero_settling(rows,protocol,counts,errors):
    """One declared2s tail, no best window, speed waiver or drive admission."""
    from hospital_path_execution import wrap
    if not rows or any(not math.isfinite(float(r['time_s'])) for r in rows):
        raise ValueError('Finite zero-command observations required')
    if any(b['time_s']<=a['time_s'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Strictly increasing actual physics observations required')
    if any(not math.isfinite(float(r['command_speed_m_s'])) or abs(float(r['command_speed_m_s']))>1e-9 for r in rows):
        raise ValueError('Only zero desired drive velocity is authorized')
    end=protocol['settling_s']+protocol['recording_s']
    if rows[-1]['time_s']<end or rows[-1]['time_s']>protocol['maximum_physics_duration_s']:
        raise ValueError('Fixed3+2s observation incomplete or exceeded; no extended window')
    tail=[r for r in rows if r['time_s']>=protocol['settling_s']]
    raw=read(protocol['original_static_record'])
    physical=shared.completed_static.physical.evaluate_static(tail,raw['start_xyyaw'],265.,counts,errors)
    checks=dict(physical['checks'])
    checks.update(fixed_tail_all_physx_speeds_under_original5mm_s=all(
        math.hypot(r['vx_physx_m_s'],r['vy_physx_m_s'])<=.005 for r in tail),
        fixed_tail_all_filtered_speeds_under_original5mm_s=all(r['pose_filtered_speed_m_s']<=.005 for r in tail),
        final_actual_physx_speed_under_original5mm_s=math.hypot(rows[-1]['vx_physx_m_s'],rows[-1]['vy_physx_m_s'])<=.005,
        final_original_start_heading_under_half_deg=abs(wrap(
            math.radians(rows[-1]['yaw_deg'])-raw['start_xyyaw'][2]))<=math.radians(.5))
    return shared.normalize_runtime_evidence(dict(checks=checks,physical_stationary_evaluation=physical,
        verdict='PASS' if all(checks.values()) else 'HOLD',fixed_tail_first_s=tail[0]['time_s'],
        fixed_tail_last_s=tail[-1]['time_s'],fixed_tail_sample_count=len(tail),original_static_result_NOT_replaced=True,
        actual_drive_admission=False,use_for_final_statistics=False))


def zero_sample_safety_metrics(row,start):
    """Measured limits, not the requested ZERO target or a relaxed threshold."""
    values=(*start,*[row[k] for k in ('x_m','y_m','roll_deg','pitch_deg','vx_physx_m_s','vy_physx_m_s')])
    if len(start)!=3 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Finite fixed-start zero observation required')
    metrics=dict(position_drift_m=math.hypot(row['x_m']-start[0],row['y_m']-start[1]),
        maximum_tilt_deg=max(abs(row['roll_deg']),abs(row['pitch_deg'])),
        horizontal_speed_m_s=math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s']))
    limits=dict(position_drift_m=.030,maximum_tilt_deg=2.,horizontal_speed_m_s=.080)
    return dict(measured=metrics,unchanged_limits=limits,
        exceeded=[k for k,value in metrics.items() if value>limits[k]])


async def observe_zero_settle_once(port,protocol,*,authorized=False):
    """Shared control lifecycle. Tests supply software ports, NEVER a plant.

    The live port owns the exact same-state/all4997 preflight and readbacks.
    Every independent cleanup is attempted; Pause is observed, not assumed.
    No captured-wheel hold, steering command or gain setter is used here.
    """
    if authorized is not True or protocol.get('approval') not in (ZERO_SETTLE_APPROVAL,REVIEWED_ZERO_APPROVAL):
        raise ValueError('NEW explicit ONE zero-command settling consent required before SDK access')
    if (protocol['settling_s'],protocol['recording_s'],protocol['maximum_control_gap_s'],
            protocol['maximum_physics_duration_s'],protocol['stop_speed_limit_m_s'])!=(3.,2.,.25,5.25,.005):
        raise ValueError('Exact fixed3+2s limits required; no threshold/window change')
    report=dict(version=protocol['version'],verdict='HOLD',checks={},new_zero_command_observations=0,
        new_route_trials=0,additional_Stop_calls=0,automatic_retry=False,actual_drive_admission=False,
        original_static_PASS_and_prior_release_HOLD_NOT_overwritten=True,use_for_final_statistics=False)
    rows=[];playing=False;preflight_complete=False;candidate=None
    try:
        report['preflight']=await port.preflight()
        preflight_complete=True
        port.subscribe();port.zero()
        playing=True;report['automatic_Play_requested']=True;report['new_zero_command_observations']=1
        port.play()
        end=protocol['settling_s']+protocol['recording_s'];previous=0.
        while previous<end:
            await port.next_sample();row=port.measure();candidate=row;now=row['time_s']
            if (not math.isfinite(now) or now<=previous or now-previous>protocol['maximum_control_gap_s']
                    or now>protocol['maximum_physics_duration_s']):raise RuntimeError('Actual physics/control-gap guard')
            port.guard(row);rows.append(row);previous=now
            port.zero()
        report.update(evaluate_fixed_zero_settling(rows,protocol,port.counts,port.errors))
        report['checks']['fixed_observation_completed']=True
    except (Exception,asyncio.CancelledError) as failure:
        report['error']=type(failure).__name__+': '+str(failure)
        report['checks']['fixed_observation_completed']=False
        capture=getattr(port,'preflight_capture',None)
        if capture is not None:report['actual_preflight_boundary_capture']=capture
        if candidate is not None and (not rows or candidate is not rows[-1]):
            # Keep the rejected measurement separately; never silently put it
            # into the accepted trace or discard the exact stopping evidence.
            report['rejected_observation_before_Pause']=candidate
        if hasattr(failure,'zero_safety_metrics'):
            report['exact_zero_safety_guard_failure']=failure.zero_safety_metrics
    finally:
        # Independent cleanup: a zero-target error MUST NOT skip Pause.
        if playing:
            try:port.zero();report['checks']['final_zero_command_sent']=True
            except Exception as failure:
                report['checks']['final_zero_command_sent']=False;report['zero_command_error']=str(failure)
        try:
            report['pause_confirmation']=await port.confirm_pause()
            report['checks']['actual_Pause_confirmed']=True
        except (Exception,asyncio.CancelledError) as failure:
            report['pause_error']=type(failure).__name__+': '+str(failure)
            report['checks']['actual_Pause_confirmed']=False
        # Keep contact/physics monitors until the Pause transition is observed.
        try:port.unsubscribe()
        except Exception as failure:
            report['monitor_closeout_error']=str(failure);report['checks']['monitors_released']=False
        if report['checks']['actual_Pause_confirmed'] and preflight_complete:
            try:
                report['postflight']=await port.postflight()
                report['checks'].update(report['postflight']['checks'])
            except Exception as failure:
                report['postflight_error']=type(failure).__name__+': '+str(failure)
                report['checks']['complete_postflight_verified']=False
        elif not preflight_complete:
            report['postflight_not_attempted']='Preflight incomplete: no tensor/actuator/physical observation to finalize'
        report.update(contact_report_counts_by_category=dict(port.counts),contact_callback_errors=list(port.errors),sample_count=len(rows))
        try:report['actual_final_timeline']=port.timeline_state()
        except Exception as failure:
            report['final_timeline_error']=str(failure);report['checks']['final_timeline_readback']=False
        report['checks'].update(final_contact_monitor_healthy=not port.errors,
            no_unexpected_contacts_through_actual_Pause=not any(port.counts.get(k,0) for k in shared.stationary.BAD_CONTACTS))
        if report.get('error') or not report['checks'] or not all(report['checks'].values()):report['verdict']='HOLD'
    return shared.normalize_runtime_evidence(report),rows


def zero_preflight_boundary_capture(protocol,texts,timeline):
    """Exact current fields on failure; no layer edit or hash substitution."""
    hashes=[hashlib.sha256(texts[k].encode()).hexdigest() for k in ('Root','Session')]
    capture=dict(actual_layer_texts=texts,actual_layer_sha256=hashes,
        expected_GET_layer_sha256=protocol['required_layer_sha256'],actual_timeline=timeline,
        exact_GET_hash_matches=[a==b for a,b in zip(hashes,protocol['required_layer_sha256'])],
        no_live_layer_or_control_edit=True)
    baseline=(protocol['reviewed_display_boundary']['reference_Session_text'] if 'reviewed_display_boundary' in protocol
        else read(REVIEWED_STATIC_FOLDER/'reviewed_display_boundary.json')['actual_layer_texts']['Session'])
    if hashlib.sha256(baseline.encode()).hexdigest()==protocol['required_layer_sha256'][1]:
        try:capture['Session_fields_vs_exact_GET_baseline']=shared.layer_field_differences(baseline,texts['Session'])
        except Exception as failure:capture['Session_field_parse_error']=type(failure).__name__+': '+str(failure)
    else:capture['exact_Session_text_baseline_unavailable_NOT_invented']=True
    return shared.normalize_runtime_evidence(capture)


class RetainedZeroSettleSDKPort:
    """Existing tensor/contact views only; no constructor/initialize/state edit."""
    def __init__(self,sdk,protocol,*,context=None):
        context=(reviewed_static_context(read(REVIEWED_STATIC_FOLDER/'static_protocol.json'),REVIEWED_STATIC_FOLDER/'static_protocol.json')
            if context is None else context)
        self._bind_sdk(sdk,protocol,context)
        self.raw=read(protocol['original_static_record']);self.prep=read(self.context.preparation.LIVE)
        self.source=self.raw['runtime_anonymous_source_reference']['fixed_configuration_rows']

    def _bind_sdk(self,sdk,protocol,context):
        """Common already-open-app bindings; no fabricated static report."""
        from collections import Counter,deque
        self.sdk=sdk;self.protocol=protocol;self.counts=Counter();self.errors=[];self.history=deque()
        self.physics_sub=self.contact_sub=self.adapter=None;self.clock=0.;self.last_dt=None;self.views=[]
        self.display_audits=[];self._last_reviewed_layer_texts=None
        self.usd=sdk.omni.usd.get_context();self.stage=self.usd.get_stage()
        self.timeline=sdk.omni.timeline.get_timeline_interface();self.app=sdk.omni.kit.app.get_app()
        self.began=time.monotonic();self.last_watch=self.began
        self.context=context

    def timeline_state(self):return evidence.timeline_state(self.timeline)

    def check_display_layers(self):
        texts=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
        if 'reviewed_display_boundary' not in self.protocol:
            hashes=[hashlib.sha256(texts[k].encode()).hexdigest() for k in ('Root','Session')]
            if hashes!=self.protocol['required_layer_sha256']:raise ValueError('Current GET layer boundary changed; no waiver or repair')
        elif texts!=self._last_reviewed_layer_texts:
            review=review_zero_display_layers(self.protocol,texts)
            self.display_audits.append(dict(Session_sha256=hashlib.sha256(texts['Session'].encode()).hexdigest(),review=review))
        self._last_reviewed_layer_texts=texts
        return texts

    def same_context(self):
        if self.usd.get_stage()!=self.stage or self.timeline.is_stopped():
            raise RuntimeError('Same stage/retained physics required; no recovery/reset')
        prim=self.stage.GetPrimAtPath(self.protocol['actor'])
        if not prim.IsValid() or not prim.IsActive() or not prim.IsLoaded():
            raise RuntimeError('Same active loaded actor required; no recovery')

    def check_post_observation_layers(self):
        """Match expected physics writeback, never exempt an actor subtree."""
        texts=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
        before=self.preflight_capture['actual_layer_texts']
        snapshot=shared.preparation.old.tensor_snapshot(self.tensor)
        if snapshot!=self.post_observation_snapshot:raise ValueError('Actual PAUSED tensor changed during postflight')
        if texts==getattr(self,'_last_post_observation_layer_texts',None):return
        renderer=split_viewport_renderer_defaults(self.stage,before,texts)
        normalized=renderer['normalized_after']
        if 'reviewed_display_boundary' in self.protocol:
            display=review_zero_display_layers(self.protocol,dict(Root=before['Root'],Session=normalized['Session']))
        else:
            if before['Session']!=normalized['Session']:raise ValueError('Exact original Session required during postflight')
            display=None
        # Session has its OWN exact/two-default proof above; the Root review
        # below cannot accidentally waive camera or opaque Session changes.
        root=recorded_physics_writeback_review(dict(Root=before['Root'],Session=normalized['Session']),normalized,
            self.protocol['required_tensor_snapshot'],snapshot,self.protocol['actor'])
        audit=dict(Root_sha256=hashlib.sha256(texts['Root'].encode()).hexdigest(),
            Session_sha256=hashlib.sha256(texts['Session'].encode()).hexdigest(),state_writeback=root,display_review=display,
            exact_renderer_default_differences=renderer['exact_renderer_default_differences'])
        if not self.post_observation_boundary_audits or audit!=self.post_observation_boundary_audits[-1]:
            self.post_observation_boundary_audits.append(audit)
        self._last_post_observation_layer_texts=texts

    async def environment(self,*,post_observation=False):
        self.same_context();before=self.timeline_state()
        if before['playing'] or before['stopped']:raise ValueError('Full environment GET requires PAUSED state')
        root=self.stage.GetRootLayer().ExportToString();session=self.stage.GetSessionLayer().ExportToString()
        p=self.context.preparation;rows={};began=watch=time.monotonic()
        def stable():
            self.same_context()
            if self.timeline_state()!=before:raise RuntimeError('PAUSED layer/clock changed during environment GET')
            if post_observation:self.check_post_observation_layers()
            elif 'reviewed_display_boundary' in self.protocol:self.check_display_layers()
            elif (self.stage.GetRootLayer().ExportToString()!=root or self.stage.GetSessionLayer().ExportToString()!=session):
                raise RuntimeError('PAUSED layer/clock changed during environment GET')
            if time.monotonic()-began>90.:raise TimeoutError('Bounded environment GET exceeded90s')
        for path,row in p.compact.stage_rows(self.stage,p.SELECTED,p.old.original):
            if path in rows:raise ValueError('Duplicate protected environment row')
            rows[path]=row
            if len(rows)%16==0:stable();await self.app.next_update_async();stable()
            if time.monotonic()-watch>=15.:
                print('[H6 ZERO] Protected environment rows %d/4997; PAUSED, NO Play'%len(rows),flush=True);watch=time.monotonic()
        stable();expected=p.load_manifest(verify_inputs=False)['compact_environment_rows']
        digest=p.compact.compare_rows(expected,rows,p.SELECTED,after_flags=True)
        if len(rows)!=4997 or digest!=self.prep['protected_environment_sha256']:raise ValueError('Protected H6 environment changed')
        if self.context.stationary.actor_configuration(self.stage)!=self.source:raise ValueError('Own145 source actor settings changed')
        if self.context.stationary.critical_environment(self.stage,self.prep)!=self.raw['environment_after_static']:
            raise ValueError('Original nominal material/mass/physics/support changed')
        return dict(row_count=4997,sha256=digest,all145_source_settings_and_critical_environment_verified=True)

    async def preflight(self):
        import importlib
        import hospital_mission_execution as control
        if self.stage is None or self.timeline_state()!=self.protocol['required_PAUSED_timeline']:
            raise ValueError('Exact current retained PAUSED boundary required; no Play to obtain it')
        texts=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
        self.preflight_capture=zero_preflight_boundary_capture(self.protocol,texts,self.timeline_state())
        hashes=self.preflight_capture['actual_layer_sha256']
        self.check_display_layers()
        self.simulation=self.sdk.SimulationManager.get_physics_sim_view()
        if self.simulation is None:raise ValueError('Existing physics view required; no initialize/Play recovery')
        self.tensor=self.simulation.create_articulation_view(self.protocol['actor']+'/base_link')
        snapshot=shared.preparation.old.tensor_snapshot(self.tensor)
        self.preflight_capture['actual_tensor_snapshot_before_rejection']=snapshot
        if snapshot!=self.protocol['required_tensor_snapshot']:raise ValueError('Same complete GET body/12joints/13links/targets/gains required')
        checked=await self.environment()
        if shared.preparation.old.tensor_snapshot(self.tensor)!=snapshot:raise ValueError('Physical state changed during preflight')
        backend=self.sdk.SimulationManager.get_backend()
        if backend not in ('numpy','torch','warp'):raise ValueError('Verified SDK tensor converter backend required')
        utils=importlib.import_module('isaacsim.core.utils.'+backend)
        self.adapter=control.RetainedTensorDriveAdapter(self.tensor,self.raw['runtime_motor_baseline'],
            encode=control.retained_tensor_encoder(utils,self.simulation.device),check_context=self.same_context,
            paused_check=lambda:not self.timeline.is_playing() and not self.timeline.is_stopped(),
            physics_stamp=0.,authorized=True)
        for name in self.context.stationary.WHEELS.values():
            path=self.protocol['actor']+'/'+name;view=self.simulation.create_rigid_contact_view(path)
            if view.sensor_count!=1 or list(view.sensor_paths)!=[path]:raise ValueError('Exact single original wheel contact view required')
            self.views.append(view)
        self.sim=self.sdk.omni.physx.get_physx_simulation_interface()
        self.stage_id=self.sdk.UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        self.encoded=self.sdk.PhysicsSchemaTools.sdfPathToInt(self.protocol['actor']+'/base_link')
        # Retained views must not author stage/actor/renderer properties.
        if self.timeline_state()!=self.protocol['required_PAUSED_timeline']:
            raise ValueError('View binding changed preflight boundary; no Play')
        self.check_display_layers()
        if shared.preparation.old.tensor_snapshot(self.tensor)!=snapshot:
            raise ValueError('Existing view binding changed actual body/joints; no Play')
        return dict(environment=checked,original_complete_tensor_retained=True,read_only_view_binding=True,
            backend=backend,device=str(self.simulation.device),exact_display_only_audits=self.display_audits)

    def subscribe(self):
        def step(dt):
            if not math.isfinite(float(dt)) or dt<=0. or dt>.25:self.errors.append('Invalid actual physics step')
            else:self.clock+=float(dt);self.last_dt=float(dt)
        def contacts(headers,data):
            try:
                for header in headers:
                    pair=tuple(sorted(str(self.sdk.PhysicsSchemaTools.intToSdfPath(v)) for v in (header.collider0,header.collider1)))
                    s=self.context.stationary;category=s.classify_contact(pair,self.prep['critical_environment']['disabled_legacy_colliders'])
                    if category not in s.KNOWN_CATEGORIES:raise ValueError('Unknown contact category')
                    if category!='unrelated':self.counts[category]+=1
            except Exception as failure:self.errors.append(type(failure).__name__+': '+str(failure))
        self.physics_sub=self.sdk.omni.physx.get_physx_interface().subscribe_physics_step_events(step)
        self.contact_sub=self.sim.subscribe_contact_report_events(contacts)

    def zero(self):
        if self.adapter is None:raise RuntimeError('No admitted retained actuator; no control')
        if self.adapter.confirmation(self.clock) is not False:raise RuntimeError('No gain/hold transition is authorized')
        self.adapter.emergency_zero()
        if self.adapter.confirmation(self.clock) is not False:raise RuntimeError('No gain/hold transition is authorized')

    def play(self):
        if self.timeline_state()!=self.protocol['required_PAUSED_timeline']:raise ValueError('Original PAUSED boundary lost before ONE Play')
        self.motion_wall=time.monotonic();self.timeline.play()

    async def next_sample(self):
        previous=self.clock
        while self.clock<=previous:
            self.same_context();self.sim.wake_up(self.stage_id,self.encoded)
            await self.app.next_update_async()
            if not self.timeline.is_playing():raise RuntimeError('Timeline controls changed during observation')
            if self.errors:raise RuntimeError('Physics/contact callback error: '+str(self.errors))
            if time.monotonic()-self.motion_wall>self.protocol['maximum_wall_time_s']:raise TimeoutError('Observation wall budget exceeded')
            if time.monotonic()-self.last_watch>=15.:
                print('[H6 ZERO] sim=%.3fs; desired drive velocity ZERO; contacts=%s'%(self.clock,dict(self.counts)),flush=True)
                self.last_watch=time.monotonic()

    def measure(self):
        from hospital_path_execution import wrap
        self.same_context();snapshot=shared.preparation.old.tensor_snapshot(self.tensor)
        xyz=snapshot['transform'][:3];x,y,z,w=snapshot['transform'][3:];v=snapshot['velocity']
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));pose=(xyz[0],xyz[1],yaw)
        self.history.append((self.clock,pose))
        while len(self.history)>2 and self.history[1][0]<=self.clock-.2:self.history.popleft()
        dt=self.clock-self.history[0][0]
        filtered=math.hypot(pose[0]-self.history[0][1][0],pose[1]-self.history[0][1][1])/dt if dt>0 else math.hypot(v[0],v[1])
        row=dict(time_s=self.clock,x_m=xyz[0],y_m=xyz[1],z_m=xyz[2],yaw_deg=math.degrees(yaw),
            roll_deg=math.degrees(roll),pitch_deg=math.degrees(pitch),vx_physx_m_s=v[0],vy_physx_m_s=v[1],vz_physx_m_s=v[2],
            pose_filtered_speed_m_s=filtered,command_speed_m_s=0.)
        for label,view in zip(self.context.stationary.WHEELS,self.views):
            force=view.get_net_contact_forces(self.last_dt)
            if hasattr(force,'detach'):force=force.detach().cpu().numpy()
            elif hasattr(force,'numpy'):force=force.numpy()
            values=np.asarray(force,float)
            if values.shape!=(1,3) or not np.isfinite(values).all():raise ValueError('Finite actual wheel force readback required')
            row[label+'_fz_n']=float(values[0,2])
        self.last_snapshot=snapshot;return shared.normalize_runtime_evidence(row)

    def guard(self,row):
        self.same_context()
        if self.errors or any(self.counts[k] for k in self.context.stationary.BAD_CONTACTS):raise RuntimeError('Unexpected contact or monitor error')
        metrics=zero_sample_safety_metrics(row,self.raw['start_xyyaw'])
        if metrics['exceeded']:
            failure=RuntimeError('Original static drift/tilt/speed safety guard: '+', '.join(metrics['exceeded']))
            failure.zero_safety_metrics=metrics;raise failure
        if self.adapter.confirmation(self.clock) is not False:raise RuntimeError('Original no-hold motor baseline required')

    async def confirm_pause(self):
        return await evidence.confirm_pause(self.timeline,self.app.next_update_async,check_context=self.same_context)

    def unsubscribe(self):self.physics_sub=self.contact_sub=None

    async def postflight(self):
        from hospital_path_execution import wrap
        self.post_observation_snapshot=shared.preparation.old.tensor_snapshot(self.tensor)
        self.post_observation_boundary_audits=[]
        result=await self.environment(post_observation=True);snapshot=shared.preparation.old.tensor_snapshot(self.tensor)
        if snapshot!=self.post_observation_snapshot:raise ValueError('Retained postflight body/joints changed while PAUSED')
        self.adapter.confirmation(self.clock)
        current=self.adapter._motor()
        if not all(np.array_equal(current[k],v) for k,v in self.adapter.original.items()):
            raise RuntimeError('Original complete gains/targets/limits changed; no automatic repair')
        runtime.task.verify_pins(self.protocol['input_sha256'])
        xyz=snapshot['transform'][:3];x,y,z,w=snapshot['transform'][3:]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        return dict(environment=result,actual_post_observation_tensor_snapshot=snapshot,exact_display_only_audits=self.display_audits,
            post_observation_boundary_audits=self.post_observation_boundary_audits,
            checks=dict(all4997_environment_rows_and145_source_settings_unchanged=True,
                original_gains_targets_and_force_limits_retained=True,all_frozen_inputs_unchanged=True,
                final_actual_speed_under_original5mm_s=math.hypot(*snapshot['velocity'][:2])<=.005,
                final_actual_start_position_under10mm=math.hypot(xyz[0]-self.raw['start_xyyaw'][0],xyz[1]-self.raw['start_xyyaw'][1])<=.010,
                final_actual_start_heading_under_half_deg=abs(wrap(yaw-self.raw['start_xyyaw'][2]))<=math.radians(.5),
                final_actual_nominal_steering_under1deg=max(abs(snapshot['positions'][i]) for i in self.adapter.steering)<=math.radians(1.),
                Pause_observed_before_total_physics_duration5_25s=self.clock<=self.protocol['maximum_physics_duration_s']))


async def run_zero_settle_once(path,*,authorized=False):
    if authorized is not True:raise ValueError('Separate approved ONE zero-command observation required before SDK access')
    path=Path(path).resolve();protocol=read(path)
    if path==ZERO_SETTLE_FOLDER/'protocol.json':expected=zero_settle_definition()
    elif path==REVIEWED_ZERO_FOLDER/'protocol.json':expected=reviewed_zero_settle_definition()
    else:raise ValueError('Exact approved observation packet path required')
    if protocol!=expected:
        raise ValueError('Exact approved fixed3+2s definition required')
    runtime.task.verify_pins(protocol['input_sha256']);no_other_task()
    reserve=path.parent/'single_observation_reserved.json';result=path.parent/'zero_settling_readback.json'
    if reserve.exists() or result.exists() or (path.parent/'zero_settling_trace.csv').exists():
        print('[H6 ZERO] Existing observation/dispatch retained; no retry or new action.',flush=True);return
    evidence.write_exclusive(reserve,dict(protocol=str(path),protocol_sha256=runtime.task.sha(path),approval=protocol['approval'],no_Play_yet=True))
    try:port=RetainedZeroSettleSDKPort(runtime.live_sdk(),protocol)
    except Exception as failure:
        report=dict(verdict='HOLD',error=type(failure).__name__+': '+str(failure),
            phase='read_only_SDK_binding_before_any_Play_or_control',protocol=str(path),
            new_zero_command_observations=0,new_route_trials=0,automatic_retry=False,actual_drive_admission=False)
        evidence.write_exclusive(result,report)
        print('[H6 ZERO] SDK binding refused before Play/control; saved:',result,'error:',report['error'],flush=True)
        return report
    report,rows=await observe_zero_settle_once(port,protocol,authorized=True)
    report.update(protocol=str(path),protocol_sha256=runtime.task.sha(path))
    if rows:
        import csv,os
        trace=path.parent/'zero_settling_trace.csv'
        try:
            with trace.open('x',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                stream.flush();os.fsync(stream.fileno())
            report.update(trace=str(trace),trace_sha256=runtime.task.sha(trace))
        except Exception as failure:
            report['trace_write_error']=type(failure).__name__+': '+str(failure)
            report['checks']['durable_complete_trace']=False;report['verdict']='HOLD'
    evidence.write_exclusive(result,shared.normalize_runtime_evidence(report))
    print('=== H6 SAME ROBOT / ONE FIXED ZERO-COMMAND SETTLING OBSERVATION ===',flush=True)
    print('VERDICT:',report['verdict'],'samples:',len(rows),'actual timeline:',report['actual_final_timeline'],flush=True)
    print('failed checks:',[k for k,v in report['checks'].items() if not v],'error:',report.get('error'),flush=True)
    print('saved:',result,'; original static and HOLD retained. NO route/retry/Stop/reset.',flush=True)
    return report


def schedule_zero_settle(path,*,authorized=False):
    if authorized is not True:raise ValueError('Separate single-observation consent required')
    name=('_ijcas_h6_reviewed_first_zero_settle_task' if Path(path).resolve()==REVIEWED_ZERO_FOLDER/'protocol.json'
        else '_ijcas_h6_zero_settle_task');prior=getattr(builtins,name,None)
    if prior is not None:
        print('[H6 ZERO] Original dispatch retained; no duplicate/retry.',flush=True);return prior
    task=asyncio.ensure_future(run_zero_settle_once(path,authorized=True));setattr(builtins,name,task)
    def done(job):
        try:job.result()
        except (Exception,asyncio.CancelledError) as failure:
            print('[H6 ZERO] Aborted; no rerun or timeline manipulation:',type(failure).__name__,str(failure),flush=True)
    task.add_done_callback(done)
    print('[H6 ZERO] ONE approved preflight ->3s zero settling ->fixed2s recording ->confirmed Pause. NO gain change/Stop/reset/route/retry.',flush=True)
    return task


def verify_executed_zero_inputs(protocol,path):
    """Resolve consumed runtime/test bytes from their exact archive only."""
    evolving={Path(__file__).resolve(),ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py'}
    resolved=[]
    with zipfile.ZipFile(Path(path).parent/'executed_sources.zip') as archive:
        for name,expected in protocol['input_sha256'].items():
            file=Path(name)
            if runtime.task.sha(file)==expected:continue
            if file not in evolving:
                import hospital_mission_lifecycle as lifecycle
                if file!=Path(lifecycle.motion.__file__).resolve():
                    raise ValueError('Executed zero physical/input dependency changed: '+name)
                archived=archive.read(file.relative_to(ROOT).as_posix())
                if hashlib.sha256(archived).hexdigest()!=expected:
                    raise ValueError('Executed zero physical/input dependency has no exact archived bytes: '+name)
                lifecycle.reviewed_mission_schema_extension(archived,file.read_bytes())
            member=str(file.relative_to(ROOT))
            if hashlib.sha256(archive.read(member)).hexdigest()!=expected:
                raise ValueError('Original executed zero source not retained: '+name)
            resolved.append(dict(path=name,executed_sha256=expected,archived_member=member))
    return dict(executed_inputs_verified=len(protocol['input_sha256']),historical_source_resolution=resolved,
        archive_resolution_is_NOT_trial_or_rerun_authority=True)


def review_completed_zero_observation(path):
    """Read saved evidence ONLY. Partial tails cannot become completed trials."""
    import csv
    path=Path(path).resolve()
    if path!=REVIEWED_ZERO_FOLDER/'protocol.json':raise ValueError('Own consumed reviewed observation required')
    protocol=read(path);report_path=path.parent/'zero_settling_readback.json';report=read(report_path)
    trace=path.parent/'zero_settling_trace.csv'
    if (report['protocol_sha256']!=runtime.task.sha(path) or report['trace_sha256']!=runtime.task.sha(trace)
            or report['new_zero_command_observations']!=1 or report['new_route_trials']!=0
            or report['pause_confirmation']['after']!=report['actual_final_timeline']
            or report['actual_final_timeline']['playing'] or report['actual_final_timeline']['stopped']):
        raise ValueError('Own immutable consumed observation and confirmed PAUSED report required')
    verified=verify_executed_zero_inputs(protocol,path)
    with trace.open(newline='',encoding='utf-8') as stream:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if len(rows)!=report['sample_count'] or not rows:raise ValueError('Complete saved CSV/report correspondence required')
    for a,b in zip(rows,rows[1:]):
        if not 0.<b['time_s']-a['time_s']<=protocol['maximum_control_gap_s']:
            raise ValueError('Saved actual physics clock is not monotonic/bounded')
    if any(not math.isfinite(v) for row in rows for v in row.values()) or any(row['command_speed_m_s']!=0. for row in rows):
        raise ValueError('Finite zero-command saved samples required')
    raw=read(protocol['original_static_record']);start=raw['start_xyyaw'];tail=[r for r in rows if r['time_s']>=3.]
    snapshot=protocol['required_tensor_snapshot'];force_keys=tuple(k for k in rows[0] if k.endswith('_fz_n'))
    if len(force_keys)!=6:raise ValueError('All six original measured wheel force channels required')
    speed=lambda row:math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])
    safety=[zero_sample_safety_metrics(row,start) for row in rows]
    if any(item['exceeded'] for item in safety):raise ValueError('Accepted trace contains a rejected safety sample')
    return shared.normalize_runtime_evidence(dict(
        status='SAVED_ZERO_COMMAND_DRIFT_AND_PARTIAL_TAIL_REVIEW_NOT_NEW_PHYSICAL_VERDICT',
        original_verdict=report['verdict'],original_error=report.get('error'),sample_count=len(rows),
        last_accepted_elapsed_s=rows[-1]['time_s'],actual_PAUSED_timeline=report['actual_final_timeline'],
        actual_Pause_confirmed=report['checks']['actual_Pause_confirmed'],
        exact_trigger_sample_available='rejected_observation_before_Pause' in report,
        exact_trigger_cannot_be_reconstructed_from_last_accepted_sample=True,
        initial_offset_from_fixed_start_m=math.dist(snapshot['transform'][:2],start[:2]),
        additional_displacement_during_saved_observation_m=math.dist(
            (rows[-1]['x_m'],rows[-1]['y_m']),snapshot['transform'][:2]),
        last_accepted_safety_metrics=safety[-1],
        maximum_recorded_tilt_deg=max(item['measured']['maximum_tilt_deg'] for item in safety),
        recorded_height_range_m=[min(r['z_m'] for r in rows),max(r['z_m'] for r in rows)],
        recorded_mean_vertical_force_weight_ratio=sum(sum(r[k] for k in force_keys) for r in rows)/len(rows)/(265.*9.81),
        partial_fixed_tail=dict(sample_count=len(tail),full_fixed2s_window_completed=False if not tail else
            rows[-1]['time_s']>=protocol['settling_s']+protocol['recording_s'],
            physx_speed_over_original5mm_s=sum(speed(r)>.005 for r in tail),
            filtered_speed_over_original5mm_s=sum(r['pose_filtered_speed_m_s']>.005 for r in tail),
            maximum_physx_horizontal_speed_m_s=max((speed(r) for r in tail),default=None),
            maximum_filtered_speed_m_s=max((r['pose_filtered_speed_m_s'] for r in tail),default=None)),
        contacts=report['contact_report_counts_by_category'],contact_errors=report['contact_callback_errors'],
        postflight_environment_NOT_verified=not report['checks'].get('complete_postflight_verified',True),
        postflight_error=report.get('postflight_error'),source_resolution=verified,
        original_report_trace_protocol_and_archive_NOT_modified=True,
        new_PhysX_trials=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={str(p):runtime.task.sha(p) for p in (path,report_path,trace,path.parent/'executed_sources.zip',
            Path(__file__),ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py')}))


def zero_observation_paused_closeout_get(path,*,read_only_authorized=False,_sdk=None):
    """ONE current GET after consumed observation; no control/timeline writes.

    Record the final tensor/gains and exact native Root/Session deltas even
    when the prior motion's postflight rejected its whole-layer boundary.
    Unknown fields are evidence, NOT exempted, repaired or drive admission.
    """
    if read_only_authorized is not True:raise ValueError('Explicit GET-only entry required; no SDK import')
    path=Path(path).resolve()
    if path!=REVIEWED_ZERO_FOLDER/'protocol.json':raise ValueError('Own consumed zero observation required')
    target=path.parent/'paused_closeout_GET.json'
    if target.exists():
        print('[H6 ZERO CLOSEOUT GET] Existing readback retained; NO duplicate or action.',flush=True);return read(target)
    protocol=read(path);saved=read(path.parent/'zero_settling_readback.json')
    if (saved['protocol_sha256']!=runtime.task.sha(path) or saved['new_zero_command_observations']!=1
            or not saved['checks']['actual_Pause_confirmed']):raise ValueError('Consumed observation/actual Pause required')
    verified=verify_executed_zero_inputs(protocol,path)
    sdk=runtime.live_sdk() if _sdk is None else _sdk
    stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    if stage is None or evidence.timeline_state(timeline)!=saved['actual_final_timeline']:
        raise ValueError('Exact current observed PAUSED clock required; no timeline recovery')
    before={k:layer.ExportToString() for k,layer in (('Root',stage.GetRootLayer()),('Session',stage.GetSessionLayer()))}
    baseline=saved['actual_preflight_boundary_capture']['actual_layer_texts']
    result=dict(status='ACTUAL_PAUSED_ZERO_CLOSEOUT_GET_ONLY_NOT_RELEASE_OR_DRIVE_ADMISSION',
        exact_saved_PAUSED_timeline=saved['actual_final_timeline'],actual_layer_texts=before,
        native_layer_differences={k:shared.layer_field_differences(baseline[k],before[k]) for k in before},
        every_recorded_field_difference_is_UNREVIEWED_NOT_waived=True,source_resolution=verified,
        new_PhysX_trials=0,any_Play_Stop_Pause_reset_control_gain_or_scene_command=False,
        complete4997_environment_rows_NOT_rechecked=True,actual_motion_authorized=False,use_for_final_statistics=False)
    try:result['SDK_GET']=retained_sdk_get_observation(sdk,read(protocol['original_static_record']))
    except Exception as failure:
        result['SDK_error']=type(failure).__name__+': '+str(failure)
        if hasattr(failure,'retained_sdk_snapshot_before_rejection'):
            result['actual_tensor_before_SDK_rejection']=failure.retained_sdk_snapshot_before_rejection
    after={k:layer.ExportToString() for k,layer in (('Root',stage.GetRootLayer()),('Session',stage.GetSessionLayer()))}
    result.update(Root_Session_unchanged_during_GET=before==after,
        actual_PAUSED_timeline_unchanged=evidence.timeline_state(timeline)==saved['actual_final_timeline'],
        same_stage=sdk.omni.usd.get_context().get_stage()==stage,
        input_sha256={str(p):runtime.task.sha(p) for p in (path,path.parent/'zero_settling_readback.json',
            path.parent/'executed_sources.zip',Path(__file__))})
    evidence.write_exclusive(target,shared.normalize_runtime_evidence(result))
    print('=== H6 SAME PAUSED ROBOT / ZERO OBSERVATION CLOSEOUT GET ONLY ===',flush=True)
    print('Root/Session native changed fields:',*[len(result['native_layer_differences'][k]) for k in ('Root','Session')],flush=True)
    for layer,changes in result['native_layer_differences'].items():
        for row in changes[:4]:print('FIELD:',layer,row['path'],row['field'],flush=True)
        if len(changes)>4:print(layer,':',len(changes)-4,'additional exact field deltas retained in saved JSON; console preview ONLY.',flush=True)
    print('SDK error:',result.get('SDK_error'),'; exact PAUSED clock retained:',result['actual_PAUSED_timeline_unchanged'],flush=True)
    if 'SDK_GET' in result:print('actual snapshot metrics:',result['SDK_GET']['metrics'],flush=True)
    print('saved:',target,'; NO timeline/control/gain/scene command or trial.',flush=True)
    return result


def prepared_source_state_types(stage,actor,link_names,joint_names):
    """GET the fresh source's exact native state types BEFORE its first Play."""
    if (len(link_names)!=13 or len(set(link_names))!=13 or 'base_link' not in link_names
            or len(joint_names)!=12 or len(set(joint_names))!=12):
        raise ValueError('Exact source13link/12joint names required')
    paths={actor+'/'+name+'.'+field for name in link_names for field in
        ('xformOp:translate','xformOp:orient','physics:velocity','physics:angularVelocity')}
    steering=('base_link_to_front_active_steering','base_link_to_rear_active_steering')
    if not all(name in joint_names for name in steering):raise ValueError('Both source steering joints required')
    paths.update(actor+'/joints/'+name+'.state:angular:physics:'+field for name in steering for field in ('position','velocity'))
    result={}
    for path in sorted(paths):
        attribute=stage.GetAttributeAtPath(path)
        if not attribute:raise ValueError('Original composed source state attribute missing: '+path)
        result[path]=str(attribute.GetTypeName())
    return result


def split_viewport_renderer_defaults(stage,before,after):
    """Remove ONLY existing typed viewport ECO Boolean deltas on layer COPIES.

    Root is a valid renderer-authoring location in Kit. Its entire physical
    content remains protected: callers still compare every remaining field
    and lossless canonical text. No prim/spec creation, metadata or subtree
    exclusion is accepted, and no live layer/renderer setting is written.
    """
    import re
    from pxr import Sdf
    if set(before)!={'Root','Session'} or set(after)!={'Root','Session'}:
        raise ValueError('Both exact Root/Session texts required')
    normalized=dict(after);changes=[]
    for label in ('Root','Session'):
        if before[label]==after[label]:continue
        copies=[]
        for text in (before[label],after[label]):
            layer=Sdf.Layer.CreateAnonymous('VIEWPORT_DEFAULT_COPY'+('.usda' if text.lstrip().startswith('#usda') else '.sdf'))
            if not layer.ImportFromString(text):raise ValueError('Viewport default COPY cannot parse')
            copies.append(layer)
        first,last=copies;changed=False
        for row in shared.layer_field_differences(first.ExportToString(),last.ExportToString()):
            path=row['path']
            if not re.fullmatch(r'/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_[0-9]+\.omni:rtx:rt:ecoMode:enabled',path):continue
            from pxr import UsdRender
            a,b=first.GetAttributeAtPath(path),last.GetAttributeAtPath(path)
            live=stage.GetAttributeAtPath(path)
            if (row['field']!='default' or not a or not b
                    or a.typeName!=Sdf.ValueTypeNames.Bool or b.typeName!=a.typeName
                    or (a.HasInfo('default') and type(a.default) is not bool)
                    or (b.HasInfo('default') and type(b.default) is not bool)
                    or first.GetNumTimeSamplesForPath(path) or last.GetNumTimeSamplesForPath(path)
                    or not live or live.GetTypeName()!=Sdf.ValueTypeNames.Bool
                    or not live.GetPrim().IsA(UsdRender.Product) or type(live.Get()) is not bool or live.GetNumTimeSamples()):
                raise ValueError('Only existing viewport RenderProduct ECO Boolean DEFAULT opinions may be separated')
            # Kit clears its idle ECO opinion on Pause, exposing the typed
            # schema/weak fallback. None here means NO authored default, not
            # a Boolean value. Restore authoredness on the anonymous COPY;
            # never substitute True or author a value in the live Root.
            if a.HasInfo('default'):b.default=a.default
            else:b.ClearInfo('default')
            changed=True
            changes.append(dict(layer=label,**row))
        if changed:normalized[label]=last.ExportToString()
    return dict(normalized_after=normalized,exact_renderer_default_differences=changes,
        no_live_setting_or_layer_edit=True,all_remaining_fields_and_canonical_text_STILL_require_validation=True)


def prepared_source_writeback_review(stage,before,after,snapshot,actor,source_types):
    """Prove FIRST source state writeback, including newly authored overrides.

    Root may have had NO new-actor opinions before Play. Permit only typed
    state AttributeSpecs plus their minimal Over ancestors; compare every
    resulting world pose/velocity/steering state against the actual SDK.
    Only ANONYMOUS layer copies are modified. Never infer an SDK pre-state
    from a later snapshot or waive an actor subtree, metadata or source edit.
    """
    from pxr import Sdf,UsdGeom
    texts=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    if texts!=after:raise ValueError('Exact current post-Pause layer text required for first writeback proof')
    renderer=split_viewport_renderer_defaults(stage,before,after)
    after=renderer['normalized_after']
    if set(source_types)!=set(prepared_source_state_types(stage,actor,snapshot['link_names'],snapshot['names'])):
        raise ValueError('Exact pre-Play56 source state paths required')
    def parse(text):
        suffix='.usda' if text.lstrip().startswith('#usda') else '.sdf'
        layer=Sdf.Layer.CreateAnonymous('FRESH_STATE_WRITEBACK_COPY'+suffix)
        if not layer.ImportFromString(text):raise ValueError('Fresh state proof layer COPY cannot parse')
        return layer
    changes={};display=[]
    for label in ('Root','Session'):
        first,last=map(parse,(before[label],after[label]));deltas=shared.layer_field_differences(first.ExportToString(),last.ExportToString())
        paths={row['path'] for row in deltas if row['path'] in source_types}
        ancestors=set()
        for path in paths:
            parent=Sdf.Path(path).GetPrimPath()
            while parent!=Sdf.Path.absoluteRootPath:
                ancestors.add(str(parent));parent=parent.GetParentPath()
        def prepare_ancestors(prim):
            # Retain native sibling order too, not merely equivalent fields.
            # Only independently classified state ancestors may be created.
            if str(prim.path) in ancestors and not first.GetPrimAtPath(prim.path):
                if prim.specifier!=Sdf.SpecifierOver or set(prim.ListInfoKeys())-{'specifier'}:
                    raise ValueError('New state ancestor is not a minimal Over: '+str(prim.path))
                Sdf.CreatePrimInLayer(first,prim.path)
            for child in prim.nameChildren:prepare_ancestors(child)
        for prim in last.rootPrims:prepare_ancestors(prim)
        for path in sorted(paths):
            target=last.GetAttributeAtPath(path);original=first.GetAttributeAtPath(path)
            if not target or str(target.typeName)!=source_types[path]:raise ValueError('Source state type changed: '+path)
            if original:
                if any(row['field']!='default' for row in deltas if row['path']==path):
                    raise ValueError('Existing state metadata changed: '+path)
                original.default=target.default
            else:
                if (target.custom or target.variability!=Sdf.VariabilityVarying
                        or set(target.ListInfoKeys())-{'custom','typeName','variability','default'}):
                    raise ValueError('New state override contains non-source metadata: '+path)
                # Construct ONLY the reviewed native type/default, instead
                # of CopySpec (which could copy opaque/unregistered metadata
                # omitted by standalone ListInfoKeys). Whole canonical text
                # below still has to match the actual layer losslessly.
                fresh=Sdf.AttributeSpec(first.GetPrimAtPath(target.path.GetPrimPath()),target.name,
                    target.typeName,Sdf.VariabilityVarying,False)
                fresh.default=target.default
        if label=='Session':
            for row in deltas:
                if row['path'] in ZERO_DISPLAY_PATHS:
                    boundary._validate_display_row(row);display.append(row)
                    first.GetAttributeAtPath(row['path']).default=last.GetAttributeAtPath(row['path']).default
        if first.ExportToString()!=last.ExportToString():
            raise ValueError('Non-state canonical '+label+' changed during first physics writeback; no waiver')
        changes[label]=deltas
    def finite(value,shape):
        value=np.asarray(value,float)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Complete finite actual first-static SDK snapshot required')
        return value
    names=snapshot['link_names'];world=finite(snapshot['all_thirteen_link_world_transforms_xyzw'],(13,7))
    velocities=finite(snapshot['all_thirteen_link_velocities_at_CoM'],(13,6))
    q=finite(snapshot['positions'],(12,));qd=finite(snapshot['joint_velocities_rad_s'],(12,))
    maxima=dict(world_position_m=0.,world_orientation_deg=0.,linear_velocity_m_s=0.,angular_velocity_rad_s=0.,steering_state_rad=0.)
    cache=UsdGeom.XformCache()
    for i,name in enumerate(names):
        prim=stage.GetPrimAtPath(actor+'/'+name);matrix=cache.GetLocalToWorldTransform(prim)
        position=np.asarray(matrix.ExtractTranslation(),float);rotation=matrix.ExtractRotationQuat()
        usd_q=np.asarray((*rotation.GetImaginary(),rotation.GetReal()),float);sdk_q=world[i,3:]
        if abs(np.linalg.norm(usd_q)-1.)>1e-4 or abs(np.linalg.norm(sdk_q)-1.)>1e-4:
            raise ValueError('Normalized actual/source rotation required')
        cosine=abs(float(usd_q@sdk_q/(np.linalg.norm(usd_q)*np.linalg.norm(sdk_q))))
        maxima['world_position_m']=max(maxima['world_position_m'],float(np.linalg.norm(position-world[i,:3])))
        maxima['world_orientation_deg']=max(maxima['world_orientation_deg'],math.degrees(2*math.acos(float(np.clip(cosine,-1.,1.)))))
        for field,part,key in (('physics:velocity',slice(0,3),'linear_velocity_m_s'),('physics:angularVelocity',slice(3,6),'angular_velocity_rad_s')):
            value=finite(prim.GetAttribute(field).Get(),(3,))
            if field=='physics:angularVelocity':value=np.radians(value)
            maxima[key]=max(maxima[key],float(np.max(abs(value-velocities[i,part]))))
    for name in ('base_link_to_front_active_steering','base_link_to_rear_active_steering'):
        index=snapshot['names'].index(name);prim=stage.GetPrimAtPath(actor+'/joints/'+name)
        for field,value in (('position',q[index]),('velocity',qd[index])):
            actual=float(prim.GetAttribute('state:angular:physics:'+field).Get())
            if not math.isfinite(actual):raise ValueError('Finite actual steering state required')
            maxima['steering_state_rad']=max(maxima['steering_state_rad'],abs(math.radians(actual)-value))
    limits=dict(world_position_m=2e-6,world_orientation_deg=1e-4,linear_velocity_m_s=1e-7,angular_velocity_rad_s=1e-7,steering_state_rad=1e-7)
    if any(value>limits[key] for key,value in maxima.items()):raise ValueError('First writeback disagrees with actual SDK snapshot: '+str(maxima))
    if dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())!=texts:
        raise ValueError('Live layers changed during read-only first-writeback proof')
    return shared.normalize_runtime_evidence(dict(status='FIRST_PREPARED_SOURCE_WRITEBACK_TYPED_AND_MATCHED_TO_ACTUAL_SDK',
        exact_native_changes=changes,reviewed_aperture_defaults=display,
        exact_renderer_default_differences=renderer['exact_renderer_default_differences'],maximum_cross_representation_errors=maxima,
        SDK_representation_tolerances_NOT_physical_acceptance_relaxation=limits,
        prePlay_source_types=dict(source_types),canonical_nonstate_Root_and_Session_retained=True,
        prePlay_source_NOT_fabricated_as_a_measured_SDK_snapshot=True,
        no_live_layer_edit=True,actual_motion_authorized=False))


def recorded_physics_writeback_review(before,after,tensor_before,tensor_after,actor):
    """Retrospective, typed tensor-to-USD comparison on ANONYMOUS COPIES.

    A simulation can author state into Root. Its whole-text hash is therefore
    not an invariant across Play. Admit NO future run here: match every actual
    changed field to retained SDK data and prove all remaining canonical text,
    including opaque metadata, unchanged. Never touch a live layer or waive
    actuator/configuration, material, geometry, camera or other actor fields.
    """
    from pxr import Sdf
    if before['Session']!=after['Session']:raise ValueError('This recorded postflight requires EXACT retained Session')
    def parse(text):
        extension='.usda' if text.lstrip().startswith('#usda') else '.sdf'
        layer=Sdf.Layer.CreateAnonymous('H6_STATE_REVIEW_COPY'+extension)
        if not layer.ImportFromString(text):raise ValueError('Recorded Root COPY must parse')
        return layer
    first,last=map(parse,(before['Root'],after['Root']))
    changes=shared.layer_field_differences(first.ExportToString(),last.ExportToString())
    names=tensor_before['link_names'];joints=tensor_before['names']
    if (len(names)!=13 or len(set(names))!=13 or len(joints)!=12 or len(set(joints))!=12
            or names!=tensor_after['link_names'] or joints!=tensor_after['names']):
        raise ValueError('Complete same13link/12joint correspondence required')
    def finite(value,shape):
        value=np.asarray(value,float)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Complete finite retained tensor required')
        return value
    def rotation(q):
        q=finite(q,(4,));norm=np.linalg.norm(q)
        if abs(norm-1.)>1e-4:raise ValueError('Normalized measured orientation required')
        x,y,z,w=q/norm
        return np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
            (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
            (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))))
    def local(layer,name):
        stem=actor+'/'+name
        p=layer.GetAttributeAtPath(stem+'.xformOp:translate');q=layer.GetAttributeAtPath(stem+'.xformOp:orient')
        if not p or not q:raise ValueError('Original direct-body local state attributes required')
        quat=q.default
        return finite(p.default,(3,)),rotation((*quat.GetImaginary(),quat.GetReal()))
    world0=finite(tensor_before['all_thirteen_link_world_transforms_xyzw'],(13,7))
    local0,rot0=local(first,'base_link');base=names.index('base_link')
    parent_rotation=rotation(world0[base,3:])@rot0.T
    parent_translation=world0[base,:3]-parent_rotation@local0
    allowed={actor+'/'+name+'.'+field for name in names for field in
        ('xformOp:translate','xformOp:orient','physics:velocity','physics:angularVelocity')}
    steering=('base_link_to_front_active_steering','base_link_to_rear_active_steering')
    allowed.update(actor+'/joints/'+name+'.state:angular:physics:'+field for name in steering for field in ('position','velocity'))
    if any(row['path'] not in allowed or row['field']!='default' for row in changes):
        raise ValueError('Unreviewed NON-state Root field changed; no waiver')
    maxima=dict(world_position_m=0.,world_orientation_deg=0.,linear_velocity_m_s=0.,angular_velocity_rad_s=0.,steering_state_rad=0.)
    exact_degree_roundtrip=dict(matched_vector3f_components=0,maximum_native_roundtrip_error_rad_s=0.,
        exact_float32_multiply_vectors=0)
    for layer,tensor in ((first,tensor_before),(last,tensor_after)):
        world=finite(tensor['all_thirteen_link_world_transforms_xyzw'],(13,7))
        velocities=finite(tensor['all_thirteen_link_velocities_at_CoM'],(13,6))
        q=finite(tensor['positions'],(12,));qd=finite(tensor['joint_velocities_rad_s'],(12,))
        for i,name in enumerate(names):
            position,orientation=local(layer,name)
            maxima['world_position_m']=max(maxima['world_position_m'],float(np.linalg.norm(parent_translation+parent_rotation@position-world[i,:3])))
            delta=(parent_rotation@orientation).T@rotation(world[i,3:])
            angle=math.degrees(math.acos(float(np.clip((np.trace(delta)-1.)/2.,-1.,1.))))
            maxima['world_orientation_deg']=max(maxima['world_orientation_deg'],angle)
            for field,part,key in (('physics:velocity',slice(0,3),'linear_velocity_m_s'),
                    ('physics:angularVelocity',slice(3,6),'angular_velocity_rad_s')):
                spec=layer.GetAttributeAtPath(actor+'/'+name+'.'+field)
                if not spec:raise ValueError('Original authored body velocity required')
                value=finite(spec.default,(3,))
                if field=='physics:angularVelocity':
                    converted=np.radians(value)
                    # PhysX radians -> authored USD vector3f degrees loses
                    # precision even for the EXACT correctly rounded value.
                    # Prove that exact typed serialization, never increase a
                    # physical threshold or waive a changed velocity field.
                    rounded=np.degrees(velocities[i,part]).astype(np.float32).astype(float)
                    native_rad=velocities[i,part].astype(np.float32)
                    native_multiply=(native_rad*np.float32(180./math.pi)).astype(float)
                    exact_native=(np.array_equal(native_rad.astype(float),velocities[i,part])
                        and np.array_equal(value,native_multiply))
                    if spec.typeName==Sdf.ValueTypeNames.Vector3f and (np.array_equal(value,rounded) or exact_native):
                        if exact_native and not np.array_equal(value,rounded):
                            exact_degree_roundtrip['exact_float32_multiply_vectors']+=1
                        exact_degree_roundtrip['matched_vector3f_components']+=3
                        exact_degree_roundtrip['maximum_native_roundtrip_error_rad_s']=max(
                            exact_degree_roundtrip['maximum_native_roundtrip_error_rad_s'],float(np.max(abs(converted-velocities[i,part]))))
                        value=velocities[i,part]
                    else:value=converted
                maxima[key]=max(maxima[key],float(np.max(abs(value-velocities[i,part]))))
        for name in steering:
            i=joints.index(name)
            for field,value in (('position',q[i]),('velocity',qd[i])):
                spec=layer.GetAttributeAtPath(actor+'/joints/'+name+'.state:angular:physics:'+field)
                if not spec:raise ValueError('Original authored steering state required')
                maxima['steering_state_rad']=max(maxima['steering_state_rad'],abs(math.radians(float(spec.default))-value))
    limits=dict(world_position_m=2e-6,world_orientation_deg=1e-4,linear_velocity_m_s=1e-7,
        angular_velocity_rad_s=1e-7,steering_state_rad=1e-7)
    if any(value>limits[key] for key,value in maxima.items()):
        raise ValueError('Recorded state writeback disagrees with actual SDK tensors: '+str(maxima))
    for row in changes:
        first.GetAttributeAtPath(row['path']).default=last.GetAttributeAtPath(row['path']).default
    if first.ExportToString()!=last.ExportToString():
        raise ValueError('Other canonical Root content changed, including opaque metadata; no waiver')
    return shared.normalize_runtime_evidence(dict(
        status='RECORDED_STATE_WRITEBACK_MATCHES_TENSORS_OTHER_ROOT_AND_EXACT_SESSION_RETAINED',
        changed_native_state_defaults=len(changes),all13links_pose_and_velocity_and_two_steering_states_matched=True,
        maximum_cross_representation_errors=maxima,SDK_float_readback_tolerances_NOT_physical_acceptance_relaxation=limits,
        exact_typed_float32_degree_serialization_proof=exact_degree_roundtrip,
        entire_canonical_Root_except_tensor_matched_state_defaults_identical=True,Session_text_exactly_identical=True,
        before_after_SDK_frames_unchanged_NOT_rebased=True,no_live_layer_or_control_edit=True,
        this_is_retrospective_NOT_future_boundary_waiver_or_drive_admission=True,actual_motion_authorized=False))


def review_zero_paused_closeout():
    """OFFLINE full-file analysis of the user's completed GET, never rerun it."""
    folder=REVIEWED_ZERO_FOLDER;path=folder/'paused_closeout_GET.json';record=read(path)
    zero_path=folder/'zero_settling_readback.json';zero=read(zero_path);protocol=read(folder/'protocol.json')
    sdk=record.get('SDK_GET',{})
    if (record.get('SDK_error') is not None or not all(record[k] is True for k in
            ('Root_Session_unchanged_during_GET','actual_PAUSED_timeline_unchanged','same_stage'))
            or not sdk.get('tensor_stable_during_GET') or not sdk.get('original_motor_baseline_backend_readback_verified')
            or sdk.get('observed_hold_confirmation') is not False
            or sdk.get('timeline')!=zero['actual_final_timeline']
            or record['any_Play_Stop_Pause_reset_control_gain_or_scene_command'] is not False):
        raise ValueError('Own unchanged PAUSED getter-only readback required')
    with zipfile.ZipFile(folder/'paused_closeout_GET_sources.zip') as archive:
        for name,expected in record['input_sha256'].items():
            file=Path(name)
            if runtime.task.sha(file)==expected:continue
            if file!=Path(__file__).resolve() or hashlib.sha256(archive.read(str(file.relative_to(ROOT)))).hexdigest()!=expected:
                raise ValueError('Actual closeout GET input/source changed: '+name)
    before=zero['actual_preflight_boundary_capture']['actual_layer_texts'];after=record['actual_layer_texts']
    tensor_before=protocol['required_tensor_snapshot'];tensor_after=sdk['actual_tensor_snapshot']
    writeback=recorded_physics_writeback_review(before,after,tensor_before,tensor_after,protocol['actor'])
    raw=read(protocol['original_static_record']);transform=tensor_after['transform'];velocity=tensor_after['velocity']
    x,y,z,w=transform[3:]
    row=dict(x_m=transform[0],y_m=transform[1],vx_physx_m_s=velocity[0],vy_physx_m_s=velocity[1],
        roll_deg=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),
        pitch_deg=math.degrees(math.asin(float(np.clip(2*(w*y-z*x),-1.,1.)))))
    metrics=zero_sample_safety_metrics(row,raw['start_xyyaw'])
    return shared.normalize_runtime_evidence(dict(status='ACTUAL_ZERO_HOLD_AND_CURRENT_PAUSED_STATE_INDEPENDENTLY_REVIEWED',
        original_zero_verdict=zero['verdict'],actual_PAUSED_timeline=sdk['timeline'],actual_actor=protocol['actor'],
        current_paused_safety_metrics=metrics,current_horizontal_stop_limit_m_s=.005,
        current_horizontal_stopped_check=math.hypot(*velocity[:2])<=.005,
        current_snapshot_is_same_PAUSED_clock_NOT_a_reconstruction_of_discarded_trigger_sample=True,
        original_motor_gains_and_targets_verified=True,drive_hold_active=False,recorded_state_writeback_review=writeback,
        full4997_environment_postflight_STILL_not_rechecked=True,original_HOLD_and_all_inputs_NOT_modified=True,
        new_PhysX_trials=0,new_GETs=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={str(p):runtime.task.sha(p) for p in (path,zero_path,folder/'protocol.json',
            folder/'paused_closeout_GET_sources.zip',Path(__file__),ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py')}))
