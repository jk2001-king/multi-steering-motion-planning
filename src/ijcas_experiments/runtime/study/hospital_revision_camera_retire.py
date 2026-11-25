"""Preserve a consumed camera-interrupted trial; continue only unconsumed cells.

No original source, monitoring policy, camera, motor setting, verdict, trace,
or time budget is edited. A separate PAUSED GET proves the retained physical
boundary. The interrupted run is never resumed, retried, or promoted.
"""
import argparse
import asyncio
import math
from pathlib import Path

import hospital_revision_timeout_resume as previous

experiment = previous.experiment
runtime = previous.runtime
OUTPUT = experiment.OUTPUT / 'camera_boundary_resume_r01'
KEY = 'ijcas_revision_n3_camera_boundary_resume_r01'
INITIAL = 'H6_B1_multimode_kinematic_ramp_f02'
CAMERA = '/OmniverseKit_Top'
ERROR = 'RuntimeError: Unreviewed non-state USD edit during route: ' + CAMERA + '.xformOp:translate'
AUTHORITY = '사용자: 방금 시점 이동 확인; 주행 문제와 구분하고 그대로 남은 검증 계속 진행'


def check_camera_report(raw, packet):
    checks = raw.get('checks', {})
    if (raw.get('verdict') != 'HOLD' or raw.get('new_route_trials') != 1
            or raw.get('callback_errors') != [ERROR] * 3
            or raw.get('error') != 'RuntimeError: Physics/contact/USD monitor error: ' + repr([ERROR])
            or raw.get('postflight_error') != 'ValueError: Non-display or unreviewed camera field changed: ' + CAMERA + '.xformOp:translate'
            or any(checks.get(k) is not True for k in (
                'final_zero_target_sent', 'actual_Pause_confirmed', 'original_gains_targets_restored'))
            or checks.get('whole_route_and_fixed_hold_completed') is not False
            or checks.get('complete_postflight_verified') is not False
            or checks.get('contacts_healthy_through_actual_Pause') is not False
            or raw.get('notice_counts', {}).get('unreviewed_paths') != 3
            or raw.get('runtime_guard', {}).get('failed') is not False
            or raw.get('runtime_guard', {}).get('rejected') is not None
            or not raw.get('actual_final_readback', {}).get('stable_PAUSED_tensor')):
        raise ValueError('Only the exact camera-interrupted, zeroed/paused/restored trial is eligible')
    if set(raw.get('contact_counts', {})) != {'wheel_support'} or raw['contact_counts']['wheel_support'] <= 0:
        raise ValueError('No unexpected recorded contact category is permitted')
    state = raw['actual_final_timeline']
    final = raw['actual_final_readback']
    tensor = final['tensor']
    original = packet['required_tensor_snapshot']
    if state['playing'] or state['stopped'] or final['timeline'] != state:
        raise ValueError('Exact actual PAUSED boundary required')
    if (len(tensor['names']) != 12 or len(tensor['link_names']) != 13
            or any(tensor[k] != original[k] for k in (
                'names', 'link_names', 'gains', 'joint_position_targets_rad', 'joint_velocity_targets_rad_s'))):
        raise ValueError('All original source identities, gains, limits and targets required')
    return True


def camera_translation(row):
    if row['path'] != CAMERA + '.xformOp:translate' or row['field'] != 'default':
        raise ValueError('Exact camera translation default required')
    vectors = []
    for side in ('before', 'after'):
        value = row[side]
        if (not isinstance(value, dict) or value.get('type') != 'Vec3d'
                or len(value.get('values', [])) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value['values'])):
            raise ValueError('Finite native Vec3d camera default required')
        vectors.append(value['values'])
    if vectors[0][2] != vectors[1][2]:
        raise ValueError('Camera Z must remain unchanged; only XY was reviewed')


def camera_default(row):
    if row['path'] == CAMERA + '.xformOp:translate':
        return camera_translation(row)
    if row['path'] in (CAMERA + '.horizontalAperture', CAMERA + '.verticalAperture'):
        from hospital_task_boundary import _validate_display_row
        return _validate_display_row(row)
    raise ValueError('Only exact reviewed Top XY and aperture defaults permitted')


async def prove_camera_boundary(report_path, cell, *, preview=False):
    from pxr import UsdGeom, UsdPhysics
    report_path = Path(report_path).resolve()
    raw = runtime.read(report_path)
    packet = runtime.read(raw['protocol'])
    check_camera_report(raw, packet)
    if packet['primary_cell']['cell_id'] != cell['cell_id'] or cell['cell_id'] != INITIAL:
        raise ValueError('Only the exact consumed camera interruption may be retired')
    runtime.retained.runtime.task.verify_pins(packet['input_sha256'])
    if runtime.sha(raw['protocol']) != raw['protocol_sha256'] or runtime.sha(raw['trace']) != raw['trace_sha256']:
        raise ValueError('Immutable original protocol and trace required')
    sdk = runtime.retained.runtime.live_sdk()
    stage = sdk.omni.usd.get_context().get_stage()
    timeline = sdk.omni.timeline.get_timeline_interface()
    state = lambda: runtime.evidence.timeline_state(timeline)
    texts = lambda: dict(Root=stage.GetRootLayer().ExportToString(), Session=stage.GetSessionLayer().ExportToString())
    saved = raw['actual_final_timeline']
    current = texts()
    if state() != saved:
        raise ValueError('Exact current PAUSED state required; no repair')
    narrow = runtime.retained.split_viewport_renderer_defaults(
        stage, raw['actual_final_readback']['layer_texts'], current)
    postabort_display = []
    for label in current:
        for row in runtime.shared.layer_field_differences(
                raw['actual_final_readback']['layer_texts'][label], narrow['normalized_after'][label]):
            if label != 'Session':
                raise ValueError('Post-abort Root changed; no repair')
            camera_default(row)
            postabort_display.append(row)
    camera = stage.GetPrimAtPath(CAMERA)
    if (not camera.IsA(UsdGeom.Camera) or camera.HasAPI(UsdPhysics.RigidBodyAPI)
            or camera.HasAPI(UsdPhysics.CollisionAPI)
            or any(any(k in schema for k in ('Physics', 'Collision', 'Scripting')) for schema in camera.GetAppliedSchemas())):
        raise ValueError('Actual nonphysical typed viewport camera required')
    old = runtime.read(packet['original_static_record'])['actual_final_readback']['layer_texts']
    display = runtime.retained.split_viewport_renderer_defaults(stage, old, current)
    normalized = display['normalized_after']
    actor = packet['actor']
    allowed = {actor + '/' + name + '.' + field
        for name in packet['required_tensor_snapshot']['link_names']
        for field in ('xformOp:translate', 'xformOp:orient', 'physics:velocity', 'physics:angularVelocity')}
    steering = [name for name in packet['required_tensor_snapshot']['names']
        if name.startswith('base_link_to_') and name.endswith('_active_steering')]
    if len(steering) != 2:
        raise ValueError('Exact two active steering joints required')
    # Historical USD state readbacks also contain passive-joint evolution.
    # This is an offline state comparison, NOT a live-monitor exemption.
    allowed.update(actor + '/joints/' + name + '.state:angular:physics:' + field
        for name in packet['required_tensor_snapshot']['names']
        for field in ('position', 'velocity'))
    differences = {label: runtime.shared.layer_field_differences(old[label], normalized[label]) for label in old}
    camera_rows = []
    state_counts = {}
    for label, rows in differences.items():
        state_counts[label] = 0
        for row in rows:
            if row['path'] in allowed and row['field'] == 'default':
                state_counts[label] += 1
            elif label == 'Session' and row['path'] == CAMERA + '.xformOp:translate':
                camera_translation(row)
                camera_rows.append(row)
            elif label == 'Session' and row['path'] in (
                    CAMERA + '.horizontalAperture', CAMERA + '.verticalAperture'):
                camera_default(row)
                camera_rows.append(row)
            else:
                raise ValueError('Unexpected non-state/physical delta: ' + repr(row))
    if not any(row['path'] == CAMERA + '.xformOp:translate' for row in camera_rows):
        raise ValueError('The reported camera XY difference must actually be present')
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 13 or any(not p.startswith(actor + '/') for p in bodies):
        raise ValueError('Exactly the retained thirteen-body actor required')
    view = sdk.SimulationManager.get_physics_sim_view().create_articulation_view(actor + '/base_link')
    snapshot = runtime.shared.preparation.old.tensor_snapshot(view)
    if snapshot != raw['actual_final_readback']['tensor']:
        raise ValueError('The retained full tensor changed')
    prep = runtime.read(raw['preparation_record'])
    context = runtime.context_for(report_path.parent.parent, actor, runtime.read(packet['geometry_file']))
    static = runtime.read(packet['original_static_record'])

    def stable():
        if sdk.omni.usd.get_context().get_stage() != stage or state() != saved or texts() != current:
            raise ValueError('GET-only stage/layers/timeline changed')

    expected = runtime.read(runtime.retained.H6 / 'actual_common_environment_rows.json')
    digest = await runtime.bootstrap.verify_rows(stage, expected['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async, unchanged=stable)
    stable()
    if (digest != prep['protected_environment_sha256']
            or context.stationary.actor_configuration(stage) != prep['source_reference']['fixed_configuration_rows']
            or context.stationary.critical_environment(stage, prep) != static['environment_after_static']
            or runtime.shared.preparation.old.tensor_snapshot(view) != snapshot):
        raise ValueError('Complete physical/environment/source boundary proof failed')
    target = OUTPUT / 'paused_boundary_GET' / (cell['cell_id'] + '.json')
    receipt = dict(
        status='CAMERA_TECHNICAL_INTERRUPTION_RETAINED_NOT_SUCCESS_OR_METHOD_FAILURE',
        original_report=str(report_path), original_report_sha256=runtime.sha(report_path),
        original_trace_sha256=raw['trace_sha256'], cell_id=cell['cell_id'],
        original_HOLD_failed_postflight_and_contact_health_flag_NOT_changed=True,
        recorded_contacts_only_wheel_support=True, callback_errors_only_exact_camera_translation=True,
        NOT_a_retrospective_continuous_contact_health_certification=True,
        camera_differences=camera_rows, postabort_camera_differences=postabort_display,
        allowed_physics_state_field_counts=state_counts,
        renderer_default_review=display['exact_renderer_default_differences'],
        all4997_rows_all145_source_settings_critical_physics_and_full_tensor_verified_NOW=True,
        protected_environment_sha256=digest, actual_final_timeline=saved,
        actual_final_readback=dict(tensor=snapshot), no_scene_camera_timeline_motor_command=True,
        no_retry_no_extension_no_promotion_no_monitoring_policy_change=True)
    if preview:
        return receipt
    runtime.evidence.write_exclusive(target, receipt)
    return target


async def _run(parent, *, authorized=False):
    import builtins
    if (authorized is not True or not parent.done() or parent.exception() is None
            or getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not parent):
        raise ValueError('Exact stopped prior owner and current continuation authority required')
    block = experiment.campaign()
    block.verify_sources()
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT / 'definition.json')['input_sha256'])
    cells = runtime.read(block.EXECUTION_PROTOCOL)['main_cells']
    initial = next(c for c in cells if c['cell_id'] == INITIAL)
    stopped = runtime.read(previous.OUTPUT / 'stopped.json')
    last = Path(stopped['last_actual_report'])
    if last != block.OUTPUT / 'live' / INITIAL / 'route/route_readback.json':
        raise ValueError('Only the diagnosed consumed camera interruption may initiate continuation')
    if len(list((block.OUTPUT / 'outcomes').glob('*.json'))) != 73:
        raise ValueError('73 outcomes plus exactly one consumed camera interruption required')
    proof = await prove_camera_boundary(last, initial)
    task = asyncio.current_task()
    builtins.ijcas_hospital_latest_task_experiment = task
    builtins.ijcas_revision_primary_task = task
    results = []
    runtime.evidence.write_exclusive(OUTPUT / 'execution_reservation.json', dict(
        authority=AUTHORITY, no_consumed_route_retried=True, no_frozen_monitoring_policy_change=True,
        unchanged_controller_search_speed_physics_and_acceptance=True))
    try:
        for cell in cells:
            name = cell['cell_id']
            target = block.OUTPUT / 'outcomes' / (name + '.json')
            if target.exists():
                results.append(runtime.read(target))
                continue
            block.verify_sources()
            if name == INITIAL:
                value = dict(runtime.read(last.parent.parent / 'trial_outcome.json'),
                    status='TECHNICAL_CAMERA_INTERRUPTION_NOT_METHOD_PERFORMANCE_FAILURE',
                    technical_interruption=True, completion_evaluable=False,
                    separate_camera_boundary_GET=str(proof), metrics=block.metrics.metrics(last))
            else:
                print('[N3 REMAINING ACTUAL START]', name, flush=True)
                value = await block.bound_cell(cell)(name, prior_result=last,
                    authorized=True, model_policy_path=block.policy_file())
                last = Path(value['report'])
                if not block.safe_restored_boundary(runtime.read(last)):
                    receipt = await previous.prove_paused_boundary(last, cell)
                    value = dict(value, separate_timeout_boundary_GET=str(receipt))
                value = dict(value, metrics=block.metrics.metrics(last) if value['actual_route_started'] else None)
            value.update(experiment_role=cell['experiment_role'], controller_version=block.control.VERSION,
                predictive_history_policy=cell['predictive_history_policy'],
                final_fixed_version_trial=True, attempt_replaced=False)
            runtime.evidence.write_exclusive(target, value)
            results.append(value)
            print('[N3 REMAINING RESULT]', name, value['actual_verdict'], flush=True)
            await asyncio.sleep(0)
        runtime.evidence.write_exclusive(block.OUTPUT / 'completed.json', dict(
            status='FINAL_N3_ALL_DECLARED_CELLS_FINISHED', results=results, last_actual_report=str(last),
            all_negative_results_preserved=True, automatic_retry=False,
            technical_camera_interruption_retained_separately=True))
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT / 'stopped.json', dict(
            error=type(error).__name__ + ': ' + str(error), last_actual_report=str(last),
            accounted_cells=len(results), automatic_retry=False))
        raise


async def run(parent, *, authorized=False):
    try:
        return await _run(parent, authorized=authorized)
    except BaseException as error:
        if not (OUTPUT / 'stopped.json').exists():
            runtime.evidence.write_exclusive(OUTPUT / 'stopped.json', dict(
                phase='GET_boundary_before_any_new_motion',
                error=type(error).__name__ + ': ' + str(error),
                last_actual_report=str(experiment.OUTPUT / 'live' / INITIAL / 'route/route_readback.json'),
                automatic_retry=False))
        raise


def dispatch():
    block = experiment.campaign()
    block.verify_sources()
    preview = OUTPUT / 'preview_GET_r03.json'
    if runtime.read(preview).get('status') != 'PASS_GET_ONLY':
        raise ValueError('Actual read-only full-boundary preview required before scheduling')
    files = [Path(__file__).resolve(), previous.OUTPUT / 'stopped.json', previous.OUTPUT / 'definition.json', preview]
    runtime.evidence.write_exclusive(OUTPUT / 'definition.json', dict(
        authority=AUTHORITY, technical_camera_interruption_retained=True,
        no_frozen_experiment_source_or_monitor_change=True, maximum_routes_per_unconsumed_cell=1,
        input_sha256={str(p): runtime.sha(p) for p in files}))
    definition = runtime.read(OUTPUT / 'definition.json')
    runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip', definition['input_sha256'])
    source = ("import importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        "import hospital_revision_camera_retire as _camera\n"
        "_camera=importlib.reload(_camera)\n"
        f"assert not hasattr(builtins,{KEY!r}),'Already reserved'\n"
        f"builtins.{KEY}=asyncio.ensure_future(_camera.run(builtins.{previous.KEY},authorized=True))\n"
        "print({'unconsumed_queue_scheduled':True,'consumed_camera_trial_NOT_retried':True})\n")
    runtime.evidence.write_exclusive(OUTPUT / 'dispatch.json', dict(source=source, authority=AUTHORITY))
    return experiment.original.predecessor.control.base.shared.runner.remote(source)


def monitor():
    return runtime.bind(previous.monitor, OUTPUT=OUTPUT, KEY=KEY)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('dispatch', 'monitor'))
    args = parser.parse_args()
    print(dispatch(), flush=True) if args.action == 'dispatch' else monitor()
