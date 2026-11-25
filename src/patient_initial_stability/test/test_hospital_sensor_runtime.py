"""Injected GET-only acquisition tests; never an Isaac/ROS trial."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import copy
import asyncio
import math

import pytest
import numpy as np

import hospital_sensor_runtime as runtime


@pytest.mark.parametrize('model', ['beam', '', None, 'unreviewed'])
def test_unreviewed_amcl_model_blocked_before_any_ROS_initialization(model):
    with pytest.raises(ValueError, match='reviewed AMCL'):
        runtime.verify_static_amcl({}, {}, laser_model_type=model, authorized=True)


@pytest.mark.parametrize('particles', [True, 1999, 10001, 2000., None])
def test_amcl_particle_budget_is_bounded_before_any_ROS_initialization(particles):
    with pytest.raises(ValueError, match='particle computation budget'):
        runtime.verify_static_amcl({}, {}, maximum_particles=particles, authorized=True)


@pytest.mark.parametrize('width', [0., .019, .201, float('nan'), True, None])
def test_amcl_likelihood_width_is_explicit_and_bounded_before_ROS(width):
    with pytest.raises(ValueError, match='likelihood width'):
        runtime.verify_static_amcl({}, {}, laser_sigma_hit_m=width, authorized=True)


def inputs():
    actor = '/World/FIXTURE_ROBOT'
    boundary = dict(actor_root=actor, base_xyz=[1., 2., 0.], base_wxyz=[1., 0., 0., 0.],
        timeline=dict(playing=False, stopped=False, current_time_s=2.),
        own_collision_paths=[actor + '/shape'], body_native_transforms={'fixture': 'not SDK'},
        layer_identifiers=['fixture_root', 'fixture_session'], layer_sha256=['fixture_A', 'fixture_B'])
    config = dict(base_frame_id='base_link', sensor_frame_id='laser', sensor_xyz_in_base=[0., 0., .35],
        sensor_wxyz_in_base=[1., 0., 0., 0.], ray_count=4, angle_min_rad=-math.pi,
        angle_increment_rad=math.pi / 2, range_min_m=.1, range_max_m=12., rate_hz=5.,
        noise_model='ideal_native_raycast_no_noise', own_collision_paths=[actor + '/shape'])
    return boundary, config


def encoder_inputs():
    names = list(runtime.EncoderOdometrySequence.CHANNELS) + ['passive_' + str(i) for i in range(8)]
    sample = dict(actor_root='/World/FIXTURE_ROBOT', stamp_ns=1000000000,
        joint_names=names, positions_rad=[0.] * 12, velocities_rad_s=[0.] * 12,
        native_dof_types=[0] * 12, native_maximum_rates_rad_s=[25.] * 12,
        native_joint_types_by_name={name: 'JointType.Revolute' for name in names},
        source='EXISTING_NATIVE_ARTICULATION_ENCODER_GET',
        commands_modes_or_ground_truth_used_as_encoder_inputs=False)
    geometry = dict(front_axis_xy_m=[.665, 0.], rear_axis_xy_m=[-.665, 0.],
        front_rolling_offset_xy_m=[0., 0.], rear_rolling_offset_xy_m=[0., 0.],
        front_radius_m=.074, rear_radius_m=.074, front_drive_sign=1, rear_drive_sign=1,
        front_steering_zero_rad=0., rear_steering_zero_rad=0.,
        front_steering_sign=1, rear_steering_sign=1)
    settings = dict(actor_root=sample['actor_root'], geometry=geometry,
        wrap_period_rad=4 * math.pi, maximum_abs_rate_rad_s=[25., 25.],
        position_roundoff_rad=1e-6, maximum_sample_gap_s=.25)
    return sample, settings


def test_causal_encoder_sequence_uses_measured_angles_not_velocity_targets_or_truth():
    sample, settings = encoder_inputs(); stream = runtime.EncoderOdometrySequence(**settings)
    assert stream.accept(sample)['odom_xyyaw'] == [0., 0., 0.]
    sample['stamp_ns'] += 200000000
    sample['positions_rad'][:2] = [.1, .1]
    sample['velocities_rad_s'][:2] = [-2., -2.]  # Not an estimator substitute.
    result = stream.accept(sample)
    assert result['odom_xyyaw'] == pytest.approx([.0074, 0., 0.])
    assert result['measured_interval_twist']['body_twist_u_v_r'] == pytest.approx([.037, 0., 0.])
    assert result['covariance_NOT_invented'] and not result['drive_admission']


def test_causal_encoder_sequence_accepts_only_rate_bounded_actual_wrap():
    sample, settings = encoder_inputs(); stream = runtime.EncoderOdometrySequence(**settings)
    sample['positions_rad'][:2] = [2 * math.pi - .05] * 2; stream.accept(sample)
    sample['stamp_ns'] += 200000000; sample['positions_rad'][:2] = [-2 * math.pi + .05] * 2
    result = stream.accept(sample)
    assert result['odom_xyyaw'] == pytest.approx([.0074, 0., 0.])
    assert result['wrapped_increment_diagnostic']['wrap_correction_counts'] == [1, 1]


@pytest.mark.parametrize('change', ['duplicate', 'regressed', 'gap', 'actor', 'source',
    'command_input', 'missing', 'order', 'nonfinite', 'type', 'subtype', 'cap', 'rate', 'range'])
def test_encoder_failure_is_terminal_without_clipping_reset_or_target_substitution(change):
    sample, settings = encoder_inputs(); stream = runtime.EncoderOdometrySequence(**settings)
    stream.accept(sample); sample['stamp_ns'] += 200000000
    if change == 'duplicate':sample['stamp_ns'] -= 200000000
    if change == 'regressed':sample['stamp_ns'] = 0
    if change == 'gap':sample['stamp_ns'] += 100000000
    if change == 'actor':sample['actor_root'] += '_other'
    if change == 'source':sample['source'] = 'GROUND_TRUTH'
    if change == 'command_input':sample['commands_modes_or_ground_truth_used_as_encoder_inputs'] = True
    if change == 'missing':sample['positions_rad'].pop()
    if change == 'order':sample['joint_names'] = list(reversed(sample['joint_names']))
    if change == 'nonfinite':sample['positions_rad'][0] = float('nan')
    if change == 'type':sample['native_dof_types'][0] = 1
    if change == 'subtype':sample['native_joint_types_by_name'][sample['joint_names'][0]] = 'JointType.RevoluteUnwrapped'
    if change == 'cap':sample['native_maximum_rates_rad_s'][0] = 26.
    if change == 'rate':sample['velocities_rad_s'][0] = 26.
    if change == 'range':sample['positions_rad'][0] = 7.
    with pytest.raises((ValueError, TypeError)):
        stream.accept(sample)
    assert stream.failed and stream.pose == [0., 0., 0.]
    with pytest.raises(ValueError, match='terminal'):stream.accept(encoder_inputs()[0])


def test_causal_encoder_geometry_cannot_change_after_initialization():
    sample, settings = encoder_inputs(); stream = runtime.EncoderOdometrySequence(**settings)
    stream.accept(sample); stream.geometry['front_radius_m'] = .08
    sample['stamp_ns'] += 200000000
    with pytest.raises(ValueError, match='contract changed'):stream.accept(sample)


@pytest.mark.parametrize('gap', [0., -.1, .251, float('nan'), True])
def test_encoder_gap_contract_rejected_before_accepting_native_data(gap):
    _, settings = encoder_inputs(); settings['maximum_sample_gap_s'] = gap
    with pytest.raises(ValueError, match='acquisition gap'):runtime.EncoderOdometrySequence(**settings)


def sensor_sequence_fixture():
    boundary, config = inputs(); sample, settings = encoder_inputs()
    boundary['timeline']['current_time_s'] = 1.
    checks = []
    stream = runtime.TimestampedNativeSensorSequence(lambda: boundary, lambda ray: [], lambda: sample,
        config, check_environment=lambda: checks.append(True),
        encoder_estimator=runtime.EncoderOdometrySequence(**settings), capture_id='SOFTWARE_ONLY', authorized=True)
    return stream, boundary, sample, checks


def test_paired_sensor_observer_never_starts_physics_and_uses_exact_encoder_anchor():
    stream, boundary, sample, checks = sensor_sequence_fixture()
    first = stream.capture()
    assert first['scan_packet']['acquisition']['stamp_ns'] == first['native_encoder']['stamp_ns']
    assert first['causal_encoder_estimate']['odom_xyyaw'] == [0., 0., 0.]
    assert first['timeline_or_actuator_commands'] == 0 and len(checks) == 2
    boundary['timeline'].update(playing=True, current_time_s=1.2)
    boundary['base_xyz'] = [20., 30., 0.]  # Deliberately NOT encoder truth.
    sample['stamp_ns'] += 200000000; sample['positions_rad'][:2] = [.1, .1]
    second = stream.capture()
    assert second['causal_encoder_estimate']['odom_xyyaw'] == pytest.approx([.0074, 0., 0.])
    assert not second['continuous_sensor_or_localization_or_Nav2_success']


@pytest.mark.parametrize('change', ['duplicate', 'stamp', 'actor', 'layer', 'collider', 'stop', 'configuration'])
def test_paired_sensor_observer_rejects_stale_or_changed_capture_without_repair(change):
    stream, boundary, sample, _ = sensor_sequence_fixture(); stream.capture()
    if change != 'duplicate':
        boundary['timeline']['current_time_s'] += .2; sample['stamp_ns'] += 200000000
    if change == 'stamp':sample['stamp_ns'] += 1
    if change == 'actor':boundary['actor_root'] += '_other'
    if change == 'layer':boundary['layer_identifiers'][0] = 'other'
    if change == 'collider':boundary['own_collision_paths'].append('/World/ADDED')
    if change == 'stop':boundary['timeline']['stopped'] = True
    if change == 'configuration':stream.configuration['rate_hz'] = 10.
    with pytest.raises((ValueError, RuntimeError)):stream.capture()
    assert stream.failed
    with pytest.raises(ValueError, match='terminal'):stream.capture()


def test_paired_sensor_observer_cannot_ignore_host_environment_failure():
    stream, _, _, _ = sensor_sequence_fixture()
    def rejected():raise RuntimeError('ORIGINAL_ENVIRONMENT_GUARD')
    stream.check = rejected
    with pytest.raises(RuntimeError, match='ORIGINAL_ENVIRONMENT_GUARD'):stream.capture()


def test_paired_sensor_observer_rejects_encoder_change_during_native_ray_queries():
    stream, _, sample, _ = sensor_sequence_fixture()
    def query(ray):
        sample['positions_rad'][0] += .001
        return []
    stream.raycast = query
    with pytest.raises(RuntimeError, match='atomic acquisition'):stream.capture()


def test_native_joint_message_keeps_actual_encoder_units_stamp_and_all_names():
    pytest.importorskip('sensor_msgs.msg')
    stream, _, _, _ = sensor_sequence_fixture(); frame = stream.capture()
    message = runtime.native_joint_state_message(frame['scan_packet'], frame['native_encoder'])
    assert message.header.stamp.sec == 1 and message.header.stamp.nanosec == 0
    assert message.name == frame['native_encoder']['joint_names']
    assert list(message.position) == frame['native_encoder']['positions_rad']
    assert list(message.velocity) == frame['native_encoder']['velocities_rad_s']
    assert not message.effort


@pytest.mark.parametrize('field', ['stamp', 'actor', 'source', 'command', 'names', 'missing'])
def test_native_joint_message_rejects_substitution_before_ROS_context_creation(field):
    stream, _, _, _ = sensor_sequence_fixture(); frame = stream.capture(); sample = frame['native_encoder']
    if field == 'stamp':sample['stamp_ns'] += 1
    if field == 'actor':sample['actor_root'] += '_other'
    if field == 'source':sample['source'] = 'GT_ODOMETRY'
    if field == 'command':sample['commands_modes_or_ground_truth_used_as_encoder_inputs'] = True
    if field == 'names':sample['joint_names'][0] = sample['joint_names'][1]
    if field == 'missing':sample['velocities_rad_s'].pop()
    with pytest.raises(ValueError):runtime.native_joint_state_message(frame['scan_packet'], sample)


class SoftwareZeroStreamPort:
    """Injected lifecycle only; no native actor, dynamics or sensors."""
    def __init__(self, failure=None):
        from types import SimpleNamespace
        self.clock = 0.; self.playing = False; self.play_calls = 0; self.zero_calls = 0
        self.counts = {}; self.errors = []; self.failure = failure; self.saved = None
        self.monitors = False
        self.actuator = SimpleNamespace(dirty=False, emergency_zero=lambda: None)
    async def preflight(self):
        if self.failure == 'preflight':raise RuntimeError('SOFTWARE_PREFLIGHT')
        return {'SOFTWARE_ONLY': True}
    def subscribe(self):self.monitors = True
    def zero(self):self.zero_calls += 1
    def play(self):self.playing = True; self.play_calls += 1
    async def next_sample(self):self.clock += .3 if self.failure == 'gap' else .05
    def measure(self):
        speed = .006 if self.failure == 'tail_speed' else 0.
        return dict(time_s=self.clock, actual_timeline_s=1. + self.clock,
            vx_physx_m_s=speed, vy_physx_m_s=0., pose_filtered_speed_m_s=speed)
    def guard(self, row, info):
        if self.failure == 'guard':raise RuntimeError('SOFTWARE_ORIGINAL_GUARD')
    async def confirm_pause(self):
        assert self.monitors or self.play_calls == 0
        self.playing = False
        return {'SOFTWARE_ONLY_Pause': True}
    def unsubscribe(self):
        assert not self.playing
        self.monitors = False
    async def postflight(self):
        if self.failure == 'postflight':raise RuntimeError('SOFTWARE_POSTFLIGHT')
        return {'checks': {'SOFTWARE_postflight': True}}
    def save(self, report, rows):self.saved = (copy.deepcopy(report), copy.deepcopy(rows))


class SoftwareZeroFrameSensor:
    def __init__(self, port, failure=None):self.port = port; self.count = 0; self.failure = failure
    def capture(self):
        if self.failure == 'sensor' and self.count:raise RuntimeError('SOFTWARE_SENSOR_FAILURE')
        stamp = round((1. + self.port.clock) * 10**9)
        value = dict(scan_packet=dict(sequence_number=self.count, acquisition=dict(stamp_ns=stamp,
            base_xyz=[0., 0., 0.], base_wxyz=[1., 0., 0., 0.], source_role='SOFTWARE_FIXTURE')),
            native_encoder=dict(stamp_ns=stamp + (1 if self.failure == 'stamp' else 0)),
            causal_encoder_estimate=dict(odom_xyyaw=[0., 0., 0.]))
        self.count += 1
        return value


def test_zero_stream_contract_and_cleanup_without_native_experiment():
    port = SoftwareZeroStreamPort(); sensor = SoftwareZeroFrameSensor(port); retained = []
    result = asyncio.run(runtime.observe_zero_sensor_stream_once(port, sensor,
        retain_frame=retained.append, authorized=True))
    assert result['verdict'] == 'PASS' and len(retained) >= 21
    assert port.play_calls == 1 and not port.playing and not port.monitors
    assert not result['Nav2_success'] and result['actual_forward_or_route_trials'] == 0
    assert port.saved is not None


@pytest.mark.parametrize('failure', ['preflight', 'gap', 'guard', 'postflight', 'tail_speed', 'sensor', 'stamp', 'sink'])
def test_zero_sensor_failure_always_pauses_preserves_error_and_never_retries(failure):
    port = SoftwareZeroStreamPort(failure); sensor = SoftwareZeroFrameSensor(port, failure)
    def sink(frame):
        if failure == 'sink':raise RuntimeError('SOFTWARE_EVIDENCE_SINK')
    result = asyncio.run(runtime.observe_zero_sensor_stream_once(port, sensor, retain_frame=sink, authorized=True))
    assert result['verdict'] == 'HOLD' and not port.playing and not port.monitors
    assert port.play_calls <= 1 and port.saved is not None and not result['automatic_retry']
    if failure in ('preflight', 'stamp', 'sink'):assert port.play_calls == 0


def test_unapproved_zero_sensor_test_never_calls_host_or_sink():
    class Forbidden:
        def __getattr__(self, key):raise AssertionError('NO_NATIVE_ACCESS')
    with pytest.raises(ValueError, match='Explicit ONE'):
        asyncio.run(runtime.observe_zero_sensor_stream_once(Forbidden(), Forbidden(), retain_frame=lambda frame: None))


def test_original_sensor_archive_is_exclusive_complete_and_hash_pinned(tmp_path):
    stream, boundary, sample, _ = sensor_sequence_fixture()
    path = tmp_path / 'original_frames.jsonl.gz'; archive = runtime.SensorFrameArchive(path)
    archive(stream.capture())
    boundary['timeline']['current_time_s'] += .2; sample['stamp_ns'] += 200000000
    archive(stream.capture()); receipt = archive.finish()
    assert receipt['original_raw_frames'] == 2 and receipt['exact_scan_encoder_stamps_and_raw_packets_verified']
    with pytest.raises(FileExistsError):runtime.SensorFrameArchive(path)
    with pytest.raises(ValueError, match='already finalized'):archive.finish()


def test_duplicate_sensor_clock_in_original_archive_is_not_repaired(tmp_path):
    stream, _, _, _ = sensor_sequence_fixture(); frame = stream.capture()
    archive = runtime.SensorFrameArchive(tmp_path / 'frames.gz'); archive(frame); archive(frame)
    with pytest.raises(ValueError, match='identity/clock'):archive.finish()


def triangle_range_fixture():
    triangles = np.array([[[2., -1., -1.], [2., -1., 1.], [2., 1., -1.]]])
    ray = dict(origin_world_m=[0., 0., 0.], direction_world=[1., 0., 0.], length_m=12.)
    hit = dict(collision='/World/EXTERNAL_STATIC_MESH', distance_m=2.,
        position_m=[2., 0., .00072], normal=[-1., 0., 0.])
    import hashlib
    return ray, hit, triangles, hashlib.sha256(triangles.tobytes()).hexdigest()


def test_independent_triangle_certification_preserves_raw_native_position_and_range():
    ray, hit, triangles, digest = triangle_range_fixture(); original = copy.deepcopy(hit)
    value = runtime.certify_native_range_position(ray, hit, triangles, source_sha256=digest, double_sided=False)
    assert hit == original and value['distance_m'] == hit['distance_m']
    assert value['native_position_m'] == hit['position_m']
    assert value['position_m'] == [2., 0., 0.]
    assert value['range_endpoint_certification']['independent_intersection']['range_m'] == 2.
    assert value['range_endpoint_certification']['unchanged_range_agreement_tolerance_m'] == 1e-4


def test_explicit_finite_surface_metric_does_not_amplify_grazing_angle_or_change_original_range():
    import hashlib,json
    from pathlib import Path
    path=_ijcas_path(_ijcas_workspace(),'studies/IJCAS/results/hospital_paper_gate/hospital_sensor_navigation/whole_H5_P_sensor_nav2/cohort_01_r01/first_route_with_declared_fixed_start_prior/range_source_disagreement_GET.json')
    issue=json.loads(path.read_text())['issues'][0]
    ray=issue['ray'];hit=issue['original_native_hit'];original=copy.deepcopy(hit)
    triangles=np.array([issue['independent_source_intersection']['source_triangle_vertices_m']],float)
    digest=hashlib.sha256(triangles.tobytes()).hexdigest()
    with pytest.raises(ValueError):runtime.certify_native_range_position(ray,hit,triangles,source_sha256=digest,double_sided=True)
    result=runtime.certify_native_range_position(ray,hit,triangles,source_sha256=digest,double_sided=True,endpoint_surface_consistency=True)
    proof=result['range_endpoint_certification']
    assert proof['independent_along_ray_range_difference_m']>1e-4
    assert 0.<proof['independent_endpoint_to_finite_triangle_distance_m']<1e-4
    assert proof['perpendicular_projection_inside_original_triangle']
    assert hit==original and result['distance_m']==hit['distance_m'] and result['native_position_m']==hit['position_m']
    outside=copy.deepcopy(hit);outside['distance_m']+=.01
    with pytest.raises(ValueError):runtime.certify_native_range_position(ray,outside,triangles,source_sha256=digest,double_sided=True,endpoint_surface_consistency=True)


@pytest.mark.parametrize('change', ['range', 'no_intersection', 'backface', 'digest', 'nonfinite'])
def test_uncertified_native_range_is_never_repaired_or_replaced_by_free_space(change):
    ray, hit, triangles, digest = triangle_range_fixture()
    if change == 'range':hit['distance_m'] += .0002
    if change == 'no_intersection':ray['origin_world_m'][1] = 5.
    if change == 'backface':triangles = triangles[:, ::-1].copy()
    if change == 'digest':digest = 'invalid'
    if change == 'nonfinite':hit['distance_m'] = float('nan')
    if change == 'backface':
        import hashlib
        digest = hashlib.sha256(triangles.tobytes()).hexdigest()
    with pytest.raises(ValueError):runtime.certify_native_range_position(ray, hit, triangles, source_sha256=digest, double_sided=False)


def test_common_held_sensor_observation_waits_actual_stop_and_backend_ack():
    port = SoftwareZeroStreamPort(); sensor = SoftwareZeroFrameSensor(port); events = []
    port.requested = None
    def confirmation(now):return port.requested is not None and now > port.requested
    def request(enabled, now, *, command_speed_m_s, measured_speed_m_s):
        assert enabled and command_speed_m_s == 0. and measured_speed_m_s <= .005
        events.append(now); port.requested = now
    port.actuator.confirmation = confirmation; port.actuator.request_hold = request
    result = asyncio.run(runtime.observe_zero_sensor_stream_once(port, sensor,
        retain_frame=lambda frame: None, use_common_hold=True, authorized=True))
    assert result['verdict'] == 'PASS' and len(events) == 1
    assert result['actual_held_since_physics_s'] > events[0]
    assert result['checks']['one_actual_captured_wheel_hold_confirmed']


def test_common_hold_cannot_capture_while_speed_exceeds_original5mm_s():
    port = SoftwareZeroStreamPort(failure='tail_speed'); sensor = SoftwareZeroFrameSensor(port)
    port.actuator.confirmation = lambda now: False
    def forbidden(*args, **kwargs):raise AssertionError('No hold above original5mm_s')
    port.actuator.request_hold = forbidden
    result = asyncio.run(runtime.observe_zero_sensor_stream_once(port, sensor,
        retain_frame=lambda frame: None, use_common_hold=True, authorized=True))
    assert result['verdict'] == 'HOLD' and 'fixed8s' in result['error']
    assert port.play_calls == 1 and not port.playing


def bridge_configuration_fixture(tmp_path):
    import json, hashlib
    _, settings = encoder_inputs()
    values = dict(map_report={'SOFTWARE_ONLY': True}, encoder_geometry=dict(geometry=settings['geometry']),
        encoder_wrap_contract={k:settings[k] for k in ('wrap_period_rad','maximum_abs_rate_rad_s','position_roundoff_rad')})
    paths = {}; pins = {}
    for key, value in values.items():
        path = tmp_path / (key + '.json'); path.write_text(json.dumps(value))
        paths[key] = str(path); pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(version='ISOLATED_NATIVE_SENSOR_AMCL_BRIDGE_v1', authorized_data_only=True,
        actuator_connection=False, capture_kind='REPLAY_ACTUAL_ZERO_OBSERVATIONS', domain_id=79,
        topic_prefix='/ijcas_sensor_dev/software_only', map_frame_id='map', odom_frame_id='odom',
        initial_prior_xyyaw=[0., 0., 0.], force_stationary_updates=True, maximum_frames=26,
        actor_root='/World/FIXTURE_ROBOT', capture_id='SOFTWARE_ONLY', initial_native_stamp_ns=1000000000,
        input_sha256=pins, **paths)


class SoftwareMemoryWire:
    """Wire-format fixture, not a socket or actual transport validation."""
    def __init__(self):
        import io
        self.buffer = io.BytesIO()
    def sendall(self, value):
        position = self.buffer.tell(); self.buffer.seek(0, 2); self.buffer.write(value); self.buffer.seek(position)
    def recv(self, size):return self.buffer.read(size)


def test_bridge_wire_exact_fields_without_ROS_or_physics():
    import hospital_sensor_ros_bridge as bridge
    wire = SoftwareMemoryWire()
    bridge.send_object(wire, dict(stamp_ns=123, values=[1., 'actual_fields', False]))
    assert bridge.receive_object(wire) == dict(stamp_ns=123, values=[1., 'actual_fields', False])


@pytest.mark.parametrize('payload', [b'{"x":1,"x":2}', b'{"x":NaN}', b''])
def test_bridge_wire_rejects_duplicate_nonfinite_and_empty_payload(payload):
    import struct
    import hospital_sensor_ros_bridge as bridge
    wire = SoftwareMemoryWire(); wire.sendall(struct.pack('!I', len(payload)) + payload)
    with pytest.raises(ValueError):bridge.receive_object(wire)


@pytest.mark.parametrize('mutation', ['default_domain', 'global_topic', 'actuator', 'unapproved', 'unbounded', 'capture_kind', 'digest'])
def test_bridge_scope_rejected_before_any_ROS_or_TCP_initialization(tmp_path, mutation):
    import hospital_sensor_ros_bridge as bridge
    config = bridge_configuration_fixture(tmp_path)
    if mutation == 'default_domain':config['domain_id'] = 0
    if mutation == 'global_topic':config['topic_prefix'] = '/scan'
    if mutation == 'actuator':config['actuator_connection'] = True
    if mutation == 'unapproved':config['authorized_data_only'] = False
    if mutation == 'unbounded':config['maximum_frames'] = 100000
    if mutation == 'capture_kind':config['capture_kind'] = 'LIVE_PASS'
    if mutation == 'digest':config['input_sha256'][config['encoder_geometry']] = 'changed'
    with pytest.raises(ValueError):bridge.checked_configuration(config)


def test_bridge_recomputes_encoder_pose_without_using_simulator_world_pose(tmp_path):
    import hospital_sensor_ros_bridge as bridge
    config = bridge_configuration_fixture(tmp_path); consumer = bridge.MeasuredFrameConsumer(config)
    stream, boundary, sample, _ = sensor_sequence_fixture()
    frame = stream.capture(); frame['scan_packet']['acquisition']['source_role'] = 'UNVALIDATED_NATIVE_QUERY_CAPTURE'
    _, _, first = consumer.accept(frame); assert first['odom_xyyaw'] == [0., 0., 0.]
    boundary['base_xyz'] = [500., 900., 0.]; boundary['timeline']['current_time_s'] += .2
    sample['stamp_ns'] += 200000000; sample['positions_rad'][:2] = [.1, .1]
    frame = stream.capture(); frame['scan_packet']['acquisition']['source_role'] = 'UNVALIDATED_NATIVE_QUERY_CAPTURE'
    _, _, second = consumer.accept(frame)
    assert second['odom_xyyaw'] == pytest.approx([.0074, 0., 0.])


def test_bridge_cannot_accept_changed_or_target_derived_odometry_and_then_restart(tmp_path):
    import hospital_sensor_ros_bridge as bridge
    config = bridge_configuration_fixture(tmp_path); consumer = bridge.MeasuredFrameConsumer(config)
    stream, _, _, _ = sensor_sequence_fixture(); frame = stream.capture()
    frame['scan_packet']['acquisition']['source_role'] = 'UNVALIDATED_NATIVE_QUERY_CAPTURE'
    frame['causal_encoder_estimate']['odom_xyyaw'] = [20., 30., 0.]
    with pytest.raises(ValueError, match='recomputation'):consumer.accept(frame)
    with pytest.raises(ValueError, match='Terminal'):consumer.accept(frame)


def short_definition_fixture():
    return dict(scope='SHORT_STRAIGHT_NAV2_INTERFACE_NOT_HOSPITAL_MISSION',mode='coordinated_counterphase',
        direction=-1,length_m=.35,maximum_speed_m_s=.03,maximum_acceleration_m_s2=.02,
        maximum_steering_rate_deg_s=8.,maximum_curvature_1_m=.70,maximum_physics_duration_s=27.25,
        maximum_wall_time_s=120.,terminal_hold_s=5.,fixed_hold_evaluation_s=3.,
        goal_position_tolerance_m=.03,goal_heading_tolerance_deg=2.,stop_speed_limit_m_s=.005,
        states_xyyaw=[[1.-.35*i/70,2.,0.] for i in range(71)])


@pytest.mark.parametrize('field,value', [('maximum_speed_m_s',.05),('stop_speed_limit_m_s',.006),
    ('goal_position_tolerance_m',.04),('direction',0),('terminal_hold_s',3.)])
def test_short_nav2_scope_cannot_silently_change_speed_acceptance_or_fixed_hold(field,value):
    definition=short_definition_fixture();definition[field]=value
    with pytest.raises(ValueError):runtime.checked_short_nav2_definition(definition)


def test_short_nav2_goal_cannot_be_moved_to_first_localization_estimate():
    definition=short_definition_fixture();definition['states_xyyaw'][-1][0]+=.01
    with pytest.raises(ValueError,match='goal differs'):runtime.checked_short_nav2_definition(definition)


def software_short_row(time):
    return dict(time_s=time,actual_timeline_s=1.+time,x_m=1.,y_m=2.,yaw_rad=0.,vx_physx_m_s=0.,vy_physx_m_s=0.)


def software_short_feedback(time):
    return dict(stamp_ns=round((1.+time)*1e9),fix_stamp_ns=round((1.+time)*1e9),
        estimated_map_xyyaw=[1.,2.,0.],no_ground_truth_pose_input=True)


def software_short_tracking_controller():
    _,settings=encoder_inputs();controller=runtime.ShortNav2MotionController(short_definition_fixture(),settings['geometry'],[0.,0.])
    first=controller.step(0.,software_short_row(0.),hold_ack=False,feedback=None,command=None,nav2_status=None)
    assert first['hold_request'] is True
    controller.step(.01,software_short_row(.01),hold_ack=True,feedback=None,command=None,nav2_status=None)
    for i in range(1,22):
        t=.01+i*.1
        info=controller.step(t,software_short_row(t),hold_ack=True,feedback=software_short_feedback(t),command=None,nav2_status=None)
    assert info['hold_request'] is False
    t+=.1;controller.step(t,software_short_row(t),hold_ack=False,feedback=software_short_feedback(t),command=None,nav2_status=None)
    return controller,t


def test_short_controller_only_ramps_actual_nav2_command_after_measured_hold_and_release():
    controller,t=software_short_tracking_controller();t+=.1
    command=dict(linear_m_s=[-.03,0.,0.],angular_rad_s=[0.,0.,0.],received_native_clock_ns=round((1.+t)*1e9))
    result=controller.step(t,software_short_row(t),hold_ack=False,feedback=software_short_feedback(t),command=command,nav2_status=None)
    assert result['speed_m_s']==pytest.approx(-.002)
    assert result['steering_targets_rad']==[0.,0.]


@pytest.mark.parametrize('speed,requested',[(.0048,True),(.005,True),(.005001,None),(.006,None)])
def test_short_goal_brake_reuses_original_measured5mm_s_capture_without_unbraked_wait(speed,requested):
    _,settings=encoder_inputs()
    controller=runtime.ShortNav2MotionController(short_definition_fixture(),settings['geometry'],[0.,0.])
    controller.stage='entry_braking';controller.last=1.;controller.speed=0.
    row=software_short_row(1.1);row['vx_physx_m_s']=speed
    result=controller.step(1.1,row,hold_ack=False,feedback=None,command=None,nav2_status=4)
    assert result['hold_request'] is requested
    assert controller.capture_count==(1 if requested else 0)
    assert controller.definition['stop_speed_limit_m_s']==.005
    assert controller.definition['terminal_hold_s']==5.


@pytest.mark.parametrize('error',['wrong_direction','stale_command','stale_feedback','GT_feedback','in_place_rotation','aborted_action'])
def test_short_controller_fails_closed_for_unsupported_or_stale_nav2_feedback(error):
    controller,t=software_short_tracking_controller();t+=.1
    command=dict(linear_m_s=[-.03,0.,0.],angular_rad_s=[0.,0.,0.],received_native_clock_ns=round((1.+t)*1e9))
    feedback=software_short_feedback(t);status=None
    if error=='wrong_direction':command['linear_m_s'][0]=.03
    if error=='stale_command':command['received_native_clock_ns']-=300000000
    if error=='stale_feedback':feedback['stamp_ns']-=300000000
    if error=='GT_feedback':feedback['no_ground_truth_pose_input']=False
    if error=='in_place_rotation':command['linear_m_s'][0]=0.;command['angular_rad_s'][2]=.02
    if error=='aborted_action':status=6
    with pytest.raises((RuntimeError,ValueError)):
        controller.step(t,software_short_row(t),hold_ack=False,feedback=feedback,command=command,nav2_status=status)
    assert controller.failed


def test_matched_amcl_pose_uses_original_encoder_anchor_and_current_odometry_only():
    import hospital_sensor_ros_bridge as bridge
    localization=dict(stamp=dict(sec=1,nanosec=0),xyyaw=[5.,2.,math.pi/2])
    result=bridge.matched_current_pose(localization,{1000000000:[1.,0.,0.]},1200000000,[1.1,0.,0.])
    assert result['estimated_map_xyyaw']==pytest.approx([5.,2.1,math.pi/2])
    assert result['no_ground_truth_pose_input']


@pytest.mark.parametrize('change',['missing_anchor','stale','future'])
def test_matched_amcl_pose_does_not_guess_an_anchor_or_postdate_old_fix(change):
    import hospital_sensor_ros_bridge as bridge
    localization=dict(stamp=dict(sec=1,nanosec=0),xyyaw=[5.,2.,0.]);history={1000000000:[0.,0.,0.]};now=1200000000
    if change=='missing_anchor':history={}
    if change=='stale':now=2100000001
    if change=='future':now=900000000
    with pytest.raises(ValueError):bridge.matched_current_pose(localization,history,now,[.1,0.,0.])


def test_full_query_uses_exact_sensor_origin_and_retains_own_and_external_hits():
    before, config = inputs()
    original = copy.deepcopy((before, config))
    calls = []
    def query(ray):
        calls.append(copy.deepcopy(ray))
        def hit(path, distance):
            return dict(collision=path, distance_m=distance, normal=[0., 0., 1.],
                position_m=[o + distance * d for o, d in zip(ray['origin_world_m'], ray['direction_world'])])
        return [hit(config['own_collision_paths'][0], .2), hit('/World/WALL', 2.)]
    report = runtime.capture_paused_scan(lambda: before, query, config,
        capture_id='SOFTWARE_ONLY_NOT_AN_EXPERIMENT', authorized=True)
    assert len(calls) == report['actual_existing_native_queries'] == 4
    assert all(ray['origin_world_m'] == [1., 2., .35] for ray in calls)
    assert report['exact_self_hit_count'] == report['external_hit_count'] == 4
    assert report['return_status_counts'] == {'VALID_RANGE': 4}
    assert report['current_native_body_layers_and_PAUSED_clock_unchanged']
    assert not report['Nav2_success'] and not report['drive_admission']
    assert not report['actual_sensor_stream_verified'] and report['new_motion_trials'] == 0
    assert (before, config) == original


@pytest.mark.parametrize('field,value', [('playing', True), ('stopped', True),
    ('current_time_s', float('nan')), ('current_time_s', -1.)])
def test_invalid_timeline_never_queries_or_initializes(field, value):
    boundary, config = inputs(); boundary['timeline'][field] = value
    def forbidden(ray):raise AssertionError('Query must not run')
    with pytest.raises(ValueError):
        runtime.capture_paused_scan(lambda: boundary, forbidden, config, capture_id='fixture', authorized=True)


@pytest.mark.parametrize('change', ['clock', 'layer', 'body', 'actor', 'inventory'])
def test_changed_native_boundary_never_yields_a_valid_capture(change):
    boundary, config = inputs(); after = copy.deepcopy(boundary)
    if change == 'clock':after['timeline']['current_time_s'] += .01
    if change == 'layer':after['layer_sha256'][1] = 'changed'
    if change == 'body':after['body_native_transforms'] = {'changed': True}
    if change == 'actor':after['actor_root'] += '_changed'
    if change == 'inventory':after['own_collision_paths'].append('/World/ADDITIONAL')
    reads = iter([boundary, after])
    with pytest.raises(RuntimeError, match='changed during acquisition'):
        runtime.capture_paused_scan(lambda: next(reads), lambda ray: [], config,
            capture_id='fixture', authorized=True)


def test_unauthorized_calls_do_not_even_read_live_state():
    def forbidden():raise AssertionError('No live access')
    with pytest.raises(ValueError, match='authorization'):
        runtime.capture_paused_scan(forbidden, lambda ray: [], {}, capture_id='fixture')


def test_query_error_is_not_replaced_by_a_no_return():
    boundary, config = inputs()
    def query(ray):raise RuntimeError('NATIVE_QUERY_FAILURE')
    with pytest.raises(RuntimeError, match='NATIVE_QUERY_FAILURE'):
        runtime.capture_paused_scan(lambda: boundary, query, config, capture_id='fixture', authorized=True)


def test_missing_raw_hit_list_is_not_replaced_by_empty_free_space():
    boundary, config = inputs()
    with pytest.raises(ValueError, match='complete hit list'):
        runtime.capture_paused_scan(lambda: boundary, lambda ray: None, config,
            capture_id='fixture', authorized=True)


def test_configuration_must_use_exact_current_self_collision_inventory():
    boundary, config = inputs(); config['own_collision_paths'] = ['/World/FIXTURE_ROBOT/unknown']
    with pytest.raises(ValueError, match='inventory'):
        runtime.capture_paused_scan(lambda: boundary, lambda ray: [], config,
            capture_id='fixture', authorized=True)


def test_below_minimum_external_return_is_retained_and_not_publishable():
    boundary, config = inputs()
    def query(ray):
        return [dict(collision='/World/CLOSE_OBSTACLE', distance_m=.05, normal=[0., 0., 1.],
            position_m=[o + .05 * d for o, d in zip(ray['origin_world_m'], ray['direction_world'])])]
    report = runtime.capture_paused_scan(lambda: boundary, query, config, capture_id='fixture', authorized=True)
    assert report['return_status_counts'] == {'BELOW_MIN_NONFREE': 4}
    assert not report['complete_raw_packet']['publishable_without_silent_near_obstacle_loss']


def test_unknown_robot_descendant_is_not_prefix_filtered():
    boundary, config = inputs()
    def query(ray):
        return [dict(collision=boundary['actor_root'] + '/unreviewed_shape', distance_m=1., normal=[0., 0., 1.],
            position_m=[o + d for o, d in zip(ray['origin_world_m'], ray['direction_world'])])]
    report = runtime.capture_paused_scan(lambda: boundary, query, config, capture_id='fixture', authorized=True)
    assert report['exact_self_hit_count'] == 0 and report['external_hit_count'] == 4


@pytest.mark.parametrize('kwargs', [dict(), dict(authorized=True, domain_id=0),
    dict(authorized=True, domain_id=True), dict(authorized=True, topic_prefix='/scan'),
    dict(authorized=True, wall_budget_s=float('nan')), dict(authorized=True, wall_budget_s=21.)])
def test_delivery_rejects_unsafe_scope_before_importing_or_starting_ROS(kwargs):
    arguments = dict(domain_id=79, topic_prefix='/ijcas_sensor_dev/test')
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        runtime.verify_scan_ros_delivery({}, **arguments)


def test_delivery_rejects_incomplete_acquisition_before_ROS_message_factory():
    with pytest.raises(ValueError, match='complete PAUSED native'):
        runtime.verify_scan_ros_delivery({}, domain_id=79,
            topic_prefix='/ijcas_sensor_dev/test', authorized=True)


@pytest.mark.parametrize('field', ['same', 'stamp', 'parent_frame', 'child_frame', 'translation', 'quaternion'])
def test_ROS_field_identity_checks_every_TF_field_not_CDR_padding(field):
    geometry = pytest.importorskip('geometry_msgs.msg')
    serialization = pytest.importorskip('rclpy.serialization')
    expected = geometry.TransformStamped()
    expected.header.stamp.sec = 3; expected.header.frame_id = 'base_link'
    expected.child_frame_id = 'laser'; expected.transform.translation.z = .35
    expected.transform.rotation.w = 1.
    actual = serialization.deserialize_message(serialization.serialize_message(expected), type(expected))
    if field == 'stamp':actual.header.stamp.nanosec = 1
    if field == 'parent_frame':actual.header.frame_id = 'different_base'
    if field == 'child_frame':actual.child_frame_id = 'different_laser'
    if field == 'translation':actual.transform.translation.z += 1e-12
    if field == 'quaternion':actual.transform.rotation.z = 1e-12
    assert runtime.same_ros_message_fields(expected, actual) is (field == 'same')


def plane_map_configuration():
    return dict(origin_xy_m=[1., 2.], shape_rows_columns=[2, 3], resolution_m=.02,
        plane_z_m=.35, half_thickness_m=.005)


def test_H6_full12m_sensor_disc_extent_fits_bounded20mm_reference_not_a_new_safety_map():
    import json
    from pathlib import Path
    root=_ijcas_workspace()
    plan=json.loads((_ijcas_path(root, 'studies/IJCAS/results/hospital_paper_gate/hospital_h6_experiments/cohort_20_held_static_r01/route/P_dynamics_full_envelope/frozen_plan.json')).read_text())
    lower=[1.5,-1.];shape=[1500,1775];resolution=.02
    upper=[lower[0]+shape[1]*resolution,lower[1]+shape[0]*resolution]
    assert shape[0]*shape[1]==2662500<=runtime.MAX_REFERENCE_MAP_CELLS==3000000
    for pose in [plan['start_world_xyyaw'],plan['goal_world_xyyaw'],*[s['world_xyyaw'] for s in plan['selected_states']]]:
        assert all(lower[i]<=pose[i]-12. and pose[i]+12.<=upper[i] for i in (0,1))


def test_registered_map_queries_all_cells_in_bottom_left_row_major_and_yields():
    boundary, _ = inputs(); calls = []; yields = []
    async def update():yields.append(True)
    def overlap(centre, half):
        calls.append((centre, half))
        return len(calls) == 2
    result = asyncio.run(runtime.capture_paused_plane_map(lambda: boundary, overlap,
        plane_map_configuration(), next_update=update, cells_per_update=2, authorized=True))
    assert result['data'] == [0, 100, 0, 0, 0, 0] and len(yields) == 3
    assert calls[0][0] == pytest.approx([1.01, 2.01, .35])
    assert calls[3][0] == pytest.approx([1.01, 2.03, .35])
    assert all(h == [.01, .01, .005] for _, h in calls)
    assert not result['localization_success'] and result['new_motion_trials'] == 0


def test_map_body_or_clock_change_during_app_yield_prevents_a_complete_map():
    boundary, _ = inputs()
    async def update():boundary['timeline']['current_time_s'] += .01
    with pytest.raises(RuntimeError, match='changed'):
        asyncio.run(runtime.capture_paused_plane_map(lambda: boundary, lambda centre, half: False,
            plane_map_configuration(), next_update=update, cells_per_update=2, authorized=True))


@pytest.mark.parametrize('value', [None, 0, 1, 'free'])
def test_incomplete_or_nonboolean_map_query_never_becomes_free_space(value):
    boundary, _ = inputs()
    async def update():pass
    with pytest.raises(ValueError, match='complete native overlap'):
        asyncio.run(runtime.capture_paused_plane_map(lambda: boundary, lambda centre, half: value,
            plane_map_configuration(), next_update=update, authorized=True))


def test_map_query_failure_is_not_a_free_cell():
    boundary, _ = inputs()
    async def update():pass
    def fail(*args):raise RuntimeError('NATIVE_OVERLAP_ERROR')
    with pytest.raises(RuntimeError, match='NATIVE_OVERLAP_ERROR'):
        asyncio.run(runtime.capture_paused_plane_map(lambda: boundary, fail,
            plane_map_configuration(), next_update=update, authorized=True))


def test_unauthorized_map_does_not_access_sdk_or_scene():
    async def update():raise AssertionError('No app updates')
    def fail(*args):raise AssertionError('No SDK access')
    with pytest.raises(ValueError, match='authorization'):
        asyncio.run(runtime.capture_paused_plane_map(fail, fail,
            plane_map_configuration(), next_update=update))


def complete_map_fixture():
    boundary, _ = inputs()
    async def update():pass
    return asyncio.run(runtime.capture_paused_plane_map(lambda: boundary, lambda *args: False,
        plane_map_configuration(), next_update=update, authorized=True))


def test_map_archive_roundtrip_is_exact_and_exclusive(tmp_path):
    record = complete_map_fixture(); path = tmp_path / 'map.json'
    archived = runtime.archive_reference_map(path, record)
    assert 'data' not in archived and archived['data_archive']['dtype'] == 'uint8'
    assert runtime.reference_map_data(archived) == record['data']
    with pytest.raises(FileExistsError):runtime.archive_reference_map(path, record)


def test_changed_map_archive_never_becomes_unknown_or_free_cells(tmp_path):
    record = complete_map_fixture(); archived = runtime.archive_reference_map(tmp_path / 'map.json', record)
    from pathlib import Path
    Path(archived['data_archive']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='digest'):runtime.reference_map_data(archived)


@pytest.mark.parametrize('mutation', ['dtype', 'shape', 'two_sources'])
def test_archive_metadata_cannot_relabel_original_map(tmp_path, mutation):
    record = complete_map_fixture(); archived = runtime.archive_reference_map(tmp_path / 'map.json', record)
    if mutation == 'dtype':archived['data_archive']['dtype'] = 'float64'
    if mutation == 'shape':archived['data_archive']['shape'] = [3, 2]
    if mutation == 'two_sources':archived['data'] = record['data']
    with pytest.raises(ValueError):runtime.reference_map_data(archived)


def map_record():
    boundary, _ = inputs()
    async def update():pass
    return asyncio.run(runtime.capture_paused_plane_map(lambda: boundary, lambda centre, half: False,
        plane_map_configuration(), next_update=update, authorized=True))


def test_registered_geometry_map_ROS_encoding_preserves_fixed_frame_origin_and_all_cells():
    pytest.importorskip('nav_msgs.msg')
    record = map_record(); record['data'][1] = 100
    record['occupied_cell_count'] = 1; record['free_cell_count'] = 5
    before = copy.deepcopy(record)
    message = runtime.plane_map_message(record, map_frame_id='fixture_map')
    assert list(message.data) == [0, 100, 0, 0, 0, 0]
    assert message.info.width == 3 and message.info.height == 2
    assert message.info.origin.position.x == 1. and message.info.origin.position.y == 2.
    assert message.header.frame_id == 'fixture_map' and record == before


@pytest.mark.parametrize('fault', ['missing_cell', 'unknown_cell', 'bad_count', 'wrong_role'])
def test_invalid_map_never_reaches_ROS_or_localization(fault):
    record = map_record()
    if fault == 'missing_cell':record['data'].pop()
    if fault == 'unknown_cell':record['data'][0] = -1
    if fault == 'bad_count':record['occupied_cell_count'] = 1
    if fault == 'wrong_role':record['version'] = 'planning_mask'
    with pytest.raises(ValueError):runtime.plane_map_message(record, map_frame_id='fixture_map')


@pytest.mark.parametrize('kwargs', [dict(), dict(authorized=True, domain_id=0),
    dict(authorized=True, domain_id=True), dict(authorized=True, wall_budget_s=float('nan')),
    dict(authorized=True, wall_budget_s=46.), dict(authorized=True)])
def test_invalid_static_localization_scope_never_starts_ROS_or_AMCL(kwargs):
    with pytest.raises(ValueError):runtime.verify_static_amcl({}, {}, **kwargs)


def test_real_ROS_numpy_covariance_fields_are_normalized_before_json_output():
    geometry = pytest.importorskip('geometry_msgs.msg')
    message = geometry.PoseWithCovarianceStamped()
    message.header.stamp.sec = 188; message.header.stamp.nanosec = 933333333
    message.pose.pose.position.x = 18.1; message.pose.pose.position.y = 22.7
    message.pose.pose.orientation.w = 1.
    message.pose.covariance = np.zeros(36, dtype=np.float64)
    message.pose.covariance[0] = np.float64(.001)
    fields = runtime.static_pose_message_fields(message)
    assert fields['xyyaw'] == [18.1, 22.7, 0.]
    assert all(type(value) is float for value in fields['covariance'])
    assert fields['stamp'] == dict(sec=188, nanosec=933333333)
    assert fields['covariance'][0] == .001


@pytest.mark.parametrize('coordinate', [0., .1, 20.75, 20.77, -20.77, 50., 1000.])
def test_outward_float32_cells_cover_rounding_cracks_and_strict_wall_contact(coordinate):
    centre = [coordinate, 17.91, .3469854458845547]; half = [.01, .01, .005]
    actual, extents = runtime.outward_native_box(centre, half)
    for c, h, a, e in zip(centre, half, actual, extents):
        # Include the final native arithmetic rounding as well as the input
        # conversion; strict inequality prevents zero-thickness face misses.
        lower = float(np.float32(np.float32(a) - np.float32(e)))
        upper = float(np.float32(np.float32(a) + np.float32(e)))
        assert lower <= c - h and upper >= c + h
    if abs(coordinate) < 50.:
        assert max(extents) < .01001  # micron-level, not a clearance change


def test_native_wall_on_hospital_grid_boundary_is_inside_outward_query_box():
    wall = float(np.float32(20.76))
    for centre_x in (20.75, 20.77):
        centre, half = runtime.outward_native_box([centre_x, 17.91, .347], [.01, .01, .005])
        assert float(np.float32(centre[0] - half[0])) < wall < float(np.float32(centre[0] + half[0]))


def test_short_runner_configures_before_Play_but_arms_ONLY_after_measured_prehold(monkeypatch, tmp_path):
    """Software order fixture, NOT a physics or Nav2 acceptance result."""
    from types import SimpleNamespace
    import hospital_mission_lifecycle as lifecycle
    events = []; clock = [0.]
    definition = short_definition_fixture()
    _, settings = encoder_inputs(); geometry = settings['geometry']

    class Controller:
        def __init__(self, *args):
            self.stage='pre_hold'; self.held_since=0.; self.final_since=None
            self.capture_count=1; self.done=False
        def step(self, t, row, **kwargs):
            self.done=t>2.
            return dict(stage='pre_hold',hold_request=None,speed_m_s=0.,steering_targets_rad=[0.,0.])

    class Port:
        output=tmp_path
        clock=property(lambda self:clock[0])
        adapter=SimpleNamespace(_motor=lambda:dict(positions=[0.,0.]),steering=[0,1],
            confirmation=lambda t:True,measured_steering=lambda:[0.,0.],
            command=lambda *args:events.append(('command',clock[0])))
        async def preflight(self):events.append(('preflight',clock[0]));return {}
        def subscribe(self):pass
        def frame_archive(self, frame):pass
        def measure(self):
            return dict(time_s=clock[0],actual_timeline_s=clock[0]+10.,x_m=0.,y_m=0.,yaw_rad=0.,
                vx_physx_m_s=0.,vy_physx_m_s=0.,pose_filtered_speed_m_s=0.)
        def guard(self,*args):pass
        def play(self):events.append(('Play',clock[0]))
        async def next_sample(self):clock[0]=2.1

    class Bridge:
        def frame(self,frame):return dict(stamp_ns=frame['native_encoder']['stamp_ns'],
            current_matched_feedback=None,actual_nav2_command=None,actual_nav2_action_status=None)
        def prepare_controller(self):events.append(('prepare',clock[0]));return {}
        def start_follow_path(self):
            events.append(('goal',clock[0]));return dict(actual_nav2_command=None)
        def close(self):events.append(('close',clock[0]));return {}

    async def finish(port,report,rows,**kwargs):
        events.append(('cleanup',clock[0]));return report
    sensor=SimpleNamespace(capture=lambda:dict(native_encoder=dict(stamp_ns=round((clock[0]+10)*1e9))))
    monkeypatch.setattr(runtime,'ShortNav2MotionController',Controller)
    monkeypatch.setattr(lifecycle,'_finish_observation',finish)
    result=asyncio.run(runtime.run_short_sensor_nav2_once(Port(),sensor,Bridge(),definition,geometry,authorized=True))
    names=[name for name,t in events]
    assert names.index('prepare')<names.index('Play')<names.index('goal')<names.index('cleanup')<names.index('close')
    assert [t for name,t in events if name=='goal']==[2.1]
    assert result['checks']['actual_Nav2_parameters_readback_before_Play'] is True
    assert result['checks']['one_fixed_Nav2_path_accepted_after_measured_prehold'] is True
    assert result['Nav2_success'] is False  # Pure order fixture cannot admit motion.
    assert 'error' not in result
