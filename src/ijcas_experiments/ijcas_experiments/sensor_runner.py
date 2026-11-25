"""One fresh replay of a retained sensor case; no ground-truth fallback."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import asyncio
import gzip
import json
import os
import subprocess
import time

from ijcas_paths import ROOT, load_archived_json

BASE = ROOT / 'studies/IJCAS/results/hospital_paper_gate/hospital_sensor_navigation'


def sensor_selection(task):
    cohort, bridge = {'H5': ('04', 'whole_H5_P_strict15_20261119_r01'),
                      'H6': ('03', 'whole_H6_P_closed_cells_20261119_r01')}[task]
    path = BASE / f'whole_{task}_P_sensor_nav2/cohort_{cohort}_r01/route/protocol.json'
    packet = load_archived_json(path)
    cfg = load_archived_json(BASE / 'ros_localization' / bridge / 'configuration.json')
    plan_path = Path(cfg['mission_platform_definition']['frozen_plan'])
    plan = load_archived_json(plan_path)
    return dict(route_template=str(path), frozen_plan=str(plan_path),
        bridge_configuration=str(BASE / 'ros_localization' / bridge / 'configuration.json'),
        native_frames=str(path.parent / 'original_native_frames.jsonl.gz'),
        controller_definition=cfg['mission_platform_definition']['controller_definition'],
        map_report=cfg['map_report'], geometry=packet['geometry_file'], grid=packet['grid_file'],
        start_xyyaw=plan['start_world_xyyaw'], goal_xyyaw=plan['goal_world_xyyaw'],
        simulation_budget_s=packet['simulation_budget_s'], wall_budget_s=packet['maximum_wall_time_s'],
        archived_case_id=f'{task}_sensor_cohort_{cohort}_r01',
        speed_policy=dict(version='RETAINED_SENSOR_PATH_NOT_MAIN_COMMON_SPEED_PROTOCOL',
            planned_target_speeds_m_s=sorted({a['target_speed_m_s'] for a in plan['selected_actions']}),
            common_max_command_speed_m_s=plan['common_max_command_speed_m_s'],
            controller_policy=packet['common_tracking_policy']),
        condition_meaning='ONE_SEPARATE_RETAINED_SENSOR_CASE_NOT_MAIN_F01_DRAW')


def audit_frozen(spec, output):
    from .workflow import write, sha, source_pins
    import hospital_mission_execution as motion
    import hospital_revision_planning as planning
    import numpy as np
    import shutil
    data = spec['sensor_case']
    plan = load_archived_json(data['frozen_plan'])
    geometry = load_archived_json(data['geometry'])
    motion.validate_mission_plan(plan, geometry)
    write(output / 'task_geometry.json', geometry)
    shutil.copyfile(data['grid'], output / 'planning_occupancy.npz')
    with np.load(data['grid']) as grid:
        _, auditor, _, _, _ = planning.planner_bindings(plan['geometry_profile'], geometry,
            grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m']))
        # Repropagate and independently check the complete retained primitives.
        write(output / 'frozen_plan.json', plan)
        audit = auditor(output / 'frozen_plan.json')
    write(output / 'independent_plan_audit.json', audit)
    prepare_controller_template(spec, output)
    write(output / 'protocol.json', dict(version='FRESH_SENSOR_FUNCTIONAL_REPLAY_v1',
        current_motion_authorized=spec['stage'] == 'run' and spec['allow_motion'],
        main_cells=[spec['cell']], sensor_case=data,
        original_sensor_case_not_main_repetition=True, automatic_retry=False))
    pins = source_pins()
    for path in (data['route_template'], data['bridge_configuration'], data['native_frames'], data['controller_definition'], data['frozen_plan'], data['map_report'], data['geometry'],
                 data['grid'], output / 'selection.json', output / 'frozen_plan.json',
                 output / 'controller_template.json', output / 'route_sources.zip'):
        pins[str(path)] = sha(path)
    write(output / 'sources.json', pins)
    result = dict(independently_valid_reference=audit['verdict'] == 'PASS',
        kind='AUDIT_RETAINED_SENSOR_PATH_NOT_NEW_PLANNING', planning_time_s=None,
        plan=str(output / 'frozen_plan.json'), new_simulator_trials=0)
    write(output / 'planning_result.json', result)
    return result


def prepare_controller_template(spec, output):
    """Complete source compatibility review before any SDK/Play/static trial."""
    import ast
    import shutil
    import zipfile
    import hospital_mission_sequence as sequence
    import hospital_sensor_platform as platform
    from .workflow import write, sha
    original_route = load_archived_json(spec['sensor_case']['controller_definition'])
    archive_path = Path(spec['sensor_case']['controller_definition']).parent / 'route_sources.zip'
    sequence.lifecycle.verify_workflow_sources(archive_path, original_route['input_sha256'])
    with zipfile.ZipFile(archive_path) as archive:
        saved = archive.read('studies/IJCAS/scripts/hospital_mission_sequence.py')
    active = Path(sequence.__file__).read_bytes()
    if ast.dump(ast.parse(saved), include_attributes=False) != ast.dump(ast.parse(active), include_attributes=False):
        original_route, review = platform.reviewed_controller_packet(
            spec['sensor_case']['controller_definition'], original_route,
            json.loads((output / 'frozen_plan.json').read_text()))
    else:
        review = dict(complete_original_sequence_AST_identical=True)
    shutil.copyfile(archive_path, output / 'route_sources.zip')
    original_route['common_lifecycle_contract'] = sequence.checked_route_lifecycle_contract(original_route)
    original_route['input_sha256'] = {str(Path(sequence.__file__)): sha(sequence.__file__)}
    write(output / 'controller_template.json', original_route)
    write(output / 'controller_source_review.json', dict(
        archived_route_template=spec['sensor_case']['controller_definition'],
        frozen_plan_sha256=sha(spec['sensor_case']['frozen_plan']), source_compatibility_review=review,
        new_controller_contract=original_route['common_lifecycle_contract'],
        explicit_old_route_policy_is_algorithm_identity_not_new_motion_authority=True))


def verify(output):
    from .workflow import sha
    for path, expected in json.loads((output / 'sources.json').read_text()).items():
        if sha(path) != expected:
            raise ValueError('Frozen sensor replay input changed: ' + path)


def bridge_command(configuration):
    # Reset the SDK binary environment before loading system ROS Python.
    script = ('source /opt/ros/humble/setup.bash; source "$1/install/setup.bash"; '
              'export PYTHONPATH="$1/tools:$1/studies/IJCAS/scripts:$1/isaac/scripts:$PYTHONPATH"; '
              'exec /usr/bin/python3 "$1/studies/IJCAS/scripts/hospital_sensor_ros_bridge.py" "$2"')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'PYTHONHOME', 'CONDA_PREFIX',
                'AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH'):
        env.pop(key, None)
    return ['/bin/bash', '--noprofile', '--norc', '-c', script, 'ijcas_sensor_bridge',
            str(ROOT), str(configuration)], env


def restore_paper_bridge_settings(generated, archived):
    """Keep ALL scientific settings from the selected paper case explicitly.

    Only current file pins/paths, stream identities and measured native
    joint baselines may differ. This does not restore a past physical state
    or use an old approval as current motion permission.
    """
    dynamic = {'input_sha256', 'actor_root', 'capture_id', 'topic_prefix',
        'initial_native_stamp_ns', 'map_report', 'encoder_geometry',
        'encoder_wrap_contract', 'nav2_plugin_prefix', 'mission_platform_definition'}
    for key in set(generated) - dynamic:
        if key not in archived or generated[key] != archived[key]:
            raise ValueError('Unreviewed sensor setting difference: ' + key)
    identity = {'actor_root', 'native_joint_names', 'original_native_position_targets_rad',
        'input_sha256', 'controller_definition', 'frozen_plan', 'geometry',
        'encoder_geometry', 'encoder_wrap_contract', 'original_controller_source_archive'}
    old, new = archived['mission_platform_definition'], generated['mission_platform_definition']
    if {k: v for k, v in old.items() if k not in identity} != {
            k: v for k, v in new.items() if k not in identity}:
        raise ValueError('Original paper controller limits/policy must match exactly')
    if archived.get('known_fixed_start_local_prior') != dict(position_std_m=.003, yaw_std_deg=.1):
        raise ValueError('The retained whole-mission paper prior must be explicit, not a bridge default')
    result = deepcopy(archived)
    for key in dynamic:
        if key in generated:
            result[key] = deepcopy(generated[key])
        else:
            result.pop(key, None)
    protected = {k: v for k, v in result.items() if k not in dynamic}
    if protected != {k: v for k, v in archived.items() if k not in dynamic}:
        raise ValueError('Complete protected sensor configuration parity required')
    return result, dict(all_nonidentity_top_level_settings_match_paper=True,
        all_controller_nonidentity_settings_match_paper=True,
        explicitly_restored_fields=sorted(set(archived) - set(generated)),
        known_fixed_start_local_prior=deepcopy(result['known_fixed_start_local_prior']),
        no_current_GT_reanchoring_no_goal_shift_no_guard_relaxation=True)


def paper_scan_settings(native_frames, own_collision_paths):
    """Use the complete scan definition actually stored by the paper run.

    The older paused scan template predates the first-return model and is
    not an execution configuration. Only the new actor's exact self-shape
    identities change; range, noise, extrinsics and return processing do not.
    """
    from hospital_sensor_messages import FIRST_RETURN_MODEL
    with gzip.open(native_frames, 'rt') as stream:
        first = json.loads(next(stream))
    archived = first['scan_source']['configuration']
    if archived.get('return_model') != FIRST_RETURN_MODEL:
        raise ValueError('Explicit original first-external-return sensor policy required')
    old_actor = first['scan_source']['acquisition']['actor_root']
    old = sorted(p.removeprefix(old_actor) for p in archived['own_collision_paths'])
    roots = {p.split('/')[2] for p in own_collision_paths}
    if len(roots) != 1 or any(not p.startswith('/World/') for p in own_collision_paths):
        raise ValueError('Exact single current actor collider inventory required')
    new_actor = '/World/' + roots.pop()
    if old != sorted(p.removeprefix(new_actor) for p in own_collision_paths):
        raise ValueError('Current sensor collider suffixes differ from original paper actor')
    result = deepcopy(archived)
    result['own_collision_paths'] = list(own_collision_paths)
    return result


async def run(spec, output):
    from .sdk_runner import current_authority, bind_relocated_support, prepare_environment
    from .workflow import write, sha
    import hospital_controlled_runtime as runtime
    import hospital_sensor_workflow as workflow
    import hospital_sensor_runtime as native
    import hospital_sensor_ros_bridge as ros
    import hospital_sensor_platform as platform
    import numpy as np
    bind = runtime.bind
    authority = current_authority(spec, output)
    def check(value):
        if value != authority:
            raise ValueError('Exact CURRENT sensor selection authority required')
        return deepcopy(authority)
    verify(output)
    await prepare_environment(spec, output)
    geometry = json.loads((output / 'task_geometry.json').read_text())
    context = bind_relocated_support(runtime.context_for(output / 'native',
        '/World/patient_transport_selected_' + spec['scenario'].lower(), geometry))
    sdk = runtime.retained.runtime.live_sdk()
    await runtime.prepare_source(context, sdk, geometry, prior_result=None)
    packet_fn = bind(runtime.static_packet, APPROVAL=authority, checked_approval=check)
    static_path, packet = packet_fn(context, output / 'task_geometry.json', Path(spec['grid']))
    class StaticHost(runtime.ControlledStaticHost):
        __init__ = bind(runtime.lc.PreparedHeldStaticSDKHost.__init__, checked_fresh_held_approval=check)
    with np.load(spec['grid']) as grid:
        static_host = StaticHost(sdk, packet, static_path, context=context, geometry=geometry,
            occupied=grid['occupied'].copy(), origin=grid['origin_xy_m'].copy(),
            resolution=float(grid['resolution_m']), authorized=True)
    static = await runtime.lc.run_held_static_once(static_host,
        runtime.lc.HeldStaticController(geometry['start_xyyaw']), authorized=True)
    if static['verdict'] != 'PASS':
        return dict(status='STATIC_NOT_PASS_NO_SENSOR_ROUTE', actual_verdict=static['verdict'],
            actual_route_started=False, report=str(context.stationary.RESULT))
    # The original context already owns this new static report and geometry.
    # Do not reinterpret historical static evidence as a live boundary.
    retained = SimpleNamespace(**dict(vars(workflow.sequence.retained),
        read=load_archived_json, static_binding=lambda *args: context))
    sequence = SimpleNamespace(**dict(vars(workflow.sequence), retained=retained))
    task = spec['scenario']
    scope = dict(BASE=output, sequence=sequence, checked_approval=check,
        STRICT_CAPTURE_APPROVAL=authority if task == 'H5' else None,
        H6_CELL_GUARD_APPROVAL=authority if task == 'H6' else None)
    host = bind(workflow.bind_whole_sensor_host, **scope)(context.stationary.RESULT,
        output / 'controller_template.json', output / f'whole_{task}_P_sensor_nav2/route',
        approved=True, approval=authority)
    write(host.output / 'frozen_plan.json', host.plan)
    read, raycast = native.existing_isaac_paused_bindings(host.protocol['actor'],
        first_external_return_only=True, endpoint_surface_consistency=True)
    boundary = read()
    scan = paper_scan_settings(spec['sensor_case']['native_frames'], boundary['own_collision_paths'])
    write(output / 'native/scan_configuration_parity.json', dict(
        original_native_frames=spec['sensor_case']['native_frames'],
        all_nonidentity_scan_settings_match_paper=True,
        original_return_model_explicit_not_paused_probe_default=True,
        current_configuration=scan, current_actor_colliders_suffix_identical=True))
    encoder_folder = BASE / 'encoder_diagnostic'
    enc_geometry = load_archived_json(encoder_folder / 'retrospective_h5_r01/source_geometry_GET.json')['geometry']
    wrap = load_archived_json(encoder_folder / 'wrapped_positions_h5_r01/source_wrap_and_cap_contract.json')
    estimator = native.EncoderOdometrySequence(actor_root=host.protocol['actor'], geometry=enc_geometry,
        **{k: wrap[k] for k in ('wrap_period_rad', 'maximum_abs_rate_rad_s', 'position_roundoff_rad')},
        maximum_sample_gap_s=.25)
    capture = output.name + '_native'
    sensor = native.TimestampedNativeSensorSequence(read, raycast,
        native.existing_isaac_encoder_reader(host.protocol['actor']), scan,
        check_environment=host.same_context, encoder_estimator=estimator, capture_id=capture, authorized=True)
    # New map descriptor, original binary raster and provenance untouched.
    map_data = load_archived_json(spec['sensor_case']['map_report'])
    archive = map_data.get('data_archive', {})
    map_data['map_data_file'] = archive.get('path', str(Path(spec['sensor_case']['map_report']).with_suffix('.npz')))
    map_path = output / 'bridge/map.json'
    write(map_path, map_data)
    bridge_def = bind(workflow.bridge_definition, sequence=sequence)
    config = bridge_def(output / 'controller_template.json', actor=host.protocol['actor'],
        snapshot=host.protocol['required_tensor_snapshot'], map_report=map_path,
        plugin_prefix=ROOT / 'install/ijcas_nav2_sensor_adapter', capture_id=capture,
        initial_stamp_ns=round(host.timeline_state()['current_time_s'] * 1e9),
        initial_prior_xyyaw=geometry['start_xyyaw'], topic_name='case_' + output.name.lower(),
        sensor_capture_policy=platform.SENSOR_CAPTURE_POLICY)
    archived_path = Path(spec['sensor_case']['bridge_configuration'])
    config, parity = restore_paper_bridge_settings(config, load_archived_json(archived_path))
    config['input_sha256'][str(archived_path)] = sha(archived_path)
    config = ros.checked_configuration(config)
    write(output / 'bridge/configuration_parity.json', dict(parity,
        original_configuration=str(archived_path), original_configuration_sha256=sha(archived_path)))
    config_path = output / 'bridge/configuration.json'
    write(config_path, config)
    command, env = bridge_command(config_path)
    child = client = None
    try:
        with (output / 'bridge/process.log').open('x') as log:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 45.
            ready = output / 'bridge/bridge_ready.json'
            while not ready.exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Owned ROS bridge did not become ready before Play; inspect bridge/process.log')
                await host.app.next_update_async()
            client = ros.BridgeClient(ready, expected_configuration=config_path)
            report = await bind(workflow.run_whole_sensor_once, checked_approval=check)(
                host, sensor, client, authorized=True)
        verify(output)
        return dict(status='COMPLETED' if report['verdict'] == 'PASS' else 'EXECUTION_NOT_COMPLETED',
            actual_verdict=report['verdict'], actual_route_started=report['new_route_trials'] == 1,
            report=str(host.output / 'route_readback.json'),
            fresh_sensor_feedback=True, ground_truth_control_fallback=False,
            original_paper_results_replaced=False)
    finally:
        if client is not None and not client.closed:
            try:
                client.close()
            except Exception:
                pass
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5.)
