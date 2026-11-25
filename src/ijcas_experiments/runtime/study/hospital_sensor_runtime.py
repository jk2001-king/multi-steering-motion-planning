"""Shared read-only native scan acquisition; no scene or timeline commands.

Importing this module does not import Isaac/ROS or initialize a publisher.
The native binding reads an ALREADY initialized PAUSED physics scene. It does
not cook geometry, construct actors, change GPU/CPU settings, or press Play.
One acquisition verifies the same boundary before and after every raw scan.
This is a sensor acquisition component, not localization or drive admission.
"""

import copy
import hashlib
import math

import hospital_runtime_evidence_io as evidence
import hospital_sensor_messages as messages


VERSION = 'HOSPITAL_EXISTING_PAUSED_NATIVE_SCAN_ACQUISITION_v1'
MAX_REFERENCE_MAP_CELLS = 3000000  # Bounded GET-only H6 path + complete12m sensor-disc reference extent.


def closest_triangle_range(origin, direction, triangles, maximum_range, *, double_sided):
    """Independent float64 static source-mesh intersection, not a new range.

    Used ONLY to certify a native range when the SDK's barycentric position
    is inconsistent. The native distance must agree within the unchanged
    100-micrometre consistency tolerance. No missing/failed ray becomes free.
    """
    import numpy as np
    origin = messages.frames.vector(origin, 3, 'source ray origin')
    direction = messages.frames.vector(direction, 3, 'source ray direction')
    values = np.asarray(triangles, dtype=float)
    if (values.ndim != 3 or values.shape[1:] != (3, 3) or not 0 < len(values) <= 200000
            or not np.isfinite(values).all() or abs(float(np.linalg.norm(direction)) - 1.) > 1e-8
            or type(double_sided) is not bool or not math.isfinite(maximum_range) or maximum_range <= 0.):
        raise ValueError('Bounded finite independent source triangles and unit ray required')
    e1 = values[:, 1] - values[:, 0]; e2 = values[:, 2] - values[:, 0]
    p = np.cross(direction, e2); determinant = np.einsum('ij,ij->i', e1, p)
    selected = np.abs(determinant) > 1e-12 if double_sided else determinant > 1e-12
    indices = np.flatnonzero(selected)
    if not len(indices):raise ValueError('No independent source-mesh intersection')
    inverse = 1. / determinant[selected]; offset = origin - values[selected, 0]
    u = np.einsum('ij,ij->i', offset, p[selected]) * inverse
    q = np.cross(offset, e1[selected]); v = q @ direction * inverse
    ranges = np.einsum('ij,ij->i', e2[selected], q) * inverse
    valid = np.flatnonzero((u >= 0.) & (v >= 0.) & (u + v <= 1.) & (ranges >= 0.) & (ranges <= maximum_range))
    if not len(valid):raise ValueError('No independent source-mesh intersection')
    chosen = valid[np.argmin(ranges[valid])]
    return dict(range_m=float(ranges[chosen]), source_triangle_index=int(indices[chosen]),
        source_triangle_vertices_m=values[indices[chosen]].tolist(), barycentric_uv=[float(u[chosen]), float(v[chosen])])


def certify_native_range_position(ray, hit, source_triangles, *, source_sha256, double_sided,
        endpoint_surface_consistency=False):
    """Preserve raw distance AND raw SDK point; certify a range-based endpoint.

    PhysX returns distance along the ray and a barycentrically reconstructed
    position. These are different output fields. Only an independently
    confirmed distance can define the LaserScan endpoint; there is no global
    relaxation of point tolerances, discarded return or safety-map change.

    Default: historical100um ALONG-RAY agreement. Explicit surface mode:
    require the native endpoint's perpendicular projection INSIDE the finite
    source triangle and Euclidean distance<=100um. This avoids grazing-angle
    amplification, but does NOT claim along-ray range error<=100um. Neither
    mode changes the measured range or drops the original SDK position.
    """
    import numpy as np
    original = copy.deepcopy(hit)
    if hashlib.sha256(np.asarray(source_triangles, dtype=float).tobytes()).hexdigest() != source_sha256:
        raise ValueError('Exact independent source triangle digest required')
    distance = hit['distance_m']
    if type(distance) not in (int, float) or not math.isfinite(distance) or not 0. <= distance <= ray['length_m']:
        raise ValueError('Finite bounded original native range required')
    check = closest_triangle_range(ray['origin_world_m'], ray['direction_world'], source_triangles,
        ray['length_m'], double_sided=double_sided)
    if type(endpoint_surface_consistency) is not bool:raise ValueError('Explicit range/surface consistency policy required')
    endpoint = np.asarray(ray['origin_world_m']) + distance * np.asarray(ray['direction_world'])
    triangle=np.asarray(source_triangles,float)[check['source_triangle_index']]
    e0,e1=triangle[1]-triangle[0],triangle[2]-triangle[0]
    normal=np.cross(e0,e1);normal/=np.linalg.norm(normal)
    signed=float(np.dot(endpoint-triangle[0],normal));projection=endpoint-signed*normal
    gram=np.array([[np.dot(e0,e0),np.dot(e0,e1)],[np.dot(e0,e1),np.dot(e1,e1)]])
    uv=np.linalg.solve(gram,np.array([np.dot(projection-triangle[0],e0),np.dot(projection-triangle[0],e1)]))
    projected_inside=bool(uv[0]>=0. and uv[1]>=0. and uv.sum()<=1.)
    along_error=abs(check['range_m']-distance)
    surface_agreed=projected_inside and abs(signed)<=1e-4
    if along_error > 1e-4 and not (endpoint_surface_consistency and surface_agreed):
        raise ValueError('Native range disagrees with independent source triangles; no endpoint repair')
    if endpoint_surface_consistency and not surface_agreed:
        raise ValueError('Native range endpoint lies outside the original finite triangle or100um surface tube')
    result = dict(hit, native_position_m=copy.deepcopy(hit['position_m']), position_m=endpoint.tolist(),
        range_endpoint_certification=dict(method='INDEPENDENT_FLOAT64_STATIC_SOURCE_TRIANGLES',
            unchanged_range_agreement_tolerance_m=1e-4,
            original_native_position_off_ray_m=float(np.max(np.abs(np.asarray(hit['position_m']) - endpoint))),
            source_mesh_sha256=source_sha256, independent_intersection=check,
            original_native_distance_NOT_changed=True, original_native_position_retained=True))
    if endpoint_surface_consistency:
        proof=result['range_endpoint_certification']
        proof.pop('unchanged_range_agreement_tolerance_m')
        proof.update(method='INDEPENDENT_FLOAT64_FINITE_SOURCE_TRIANGLE_SURFACE',
            endpoint_to_source_spatial_tolerance_m=1e-4,
            independent_along_ray_range_difference_m=float(along_error),
            independent_endpoint_to_finite_triangle_distance_m=float(abs(signed)),
            perpendicular_projection_inside_original_triangle=projected_inside,
            perpendicular_projection_barycentric_uv=uv.tolist(),
            grazing_ray_range_error_NOT_claimed_under100um=True)
    if hit != original:raise RuntimeError('Original raw hit unexpectedly mutated')
    return result


def zero_stream_contract(use_common_hold=False):
    """Prospective fixed observation; not a navigation/route success rule."""
    if type(use_common_hold) is not bool:raise ValueError('Explicit common held or released observation required')
    result = dict(settling_s=3., recording_s=2., maximum_physics_duration_s=5.25,
        maximum_control_gap_s=.25, minimum_scan_period_ns=200000000,
        maximum_scan_gap_ns=250000000, final_stop_speed_m_s=.005,
        maximum_wall_time_s=120., maximum_Play_calls=1, maximum_Stop_reset_calls=0,
        nonzero_drive_commands=0, automatic_retry=False)
    if use_common_hold:
        result.update(common_original_captured_wheel_hold=True, hold_capture_stop_limit_m_s=.005,
            hold_initial_stop_timeout_s=8., hold_backend_ACK_timeout_s=12.,
            maximum_physics_duration_s=27.25, original_gains_and_targets_restored_after_actual_Pause=True)
    return result


def checked_short_nav2_definition(value):
    """Fixed, separate interface probe; not an H1-H6 result or a P replan."""
    value = evidence.json_value(value)
    if (value.get('scope') != 'SHORT_STRAIGHT_NAV2_INTERFACE_NOT_HOSPITAL_MISSION'
            or value.get('mode') != 'coordinated_counterphase'
            or type(value.get('direction')) is not int or value['direction'] not in (-1, 1)
            or value.get('length_m') != .35 or value.get('maximum_speed_m_s') != .03
            or value.get('maximum_acceleration_m_s2') != .02
            or value.get('maximum_steering_rate_deg_s') != 8.
            or value.get('maximum_curvature_1_m') != .70
            or value.get('maximum_physics_duration_s') not in (27.25,45.)
            or value.get('maximum_wall_time_s') not in (120.,900.)
            or value.get('terminal_hold_s') != 5. or value.get('fixed_hold_evaluation_s') != 3.
            or value.get('goal_position_tolerance_m') != .03 or value.get('goal_heading_tolerance_deg') != 2.
            or value.get('stop_speed_limit_m_s') != .005):
        raise ValueError('Exact separately declared35cm sensor/Nav2 probe and unchanged safety limits required')
    states = [messages.frames.vector(row, 3, 'fixed short path pose') for row in value['states_xyyaw']]
    if not 71 <= len(states) <= 100:raise ValueError('Complete fixed5mm short-path sampling required')
    x, y, heading = states[0]
    for i, row in enumerate(states):
        distance = value['direction'] * .35 * i / (len(states) - 1)
        expected = [x + distance * math.cos(heading), y + distance * math.sin(heading), heading]
        if max(abs(float(a) - float(b)) for a, b in zip(row, expected)) > 1e-10:
            raise ValueError('Fixed short path/goal differs; no AMCL estimate anchoring or unchecked endpoint')
    return value


class ShortNav2MotionController:
    """Bounded native adapter for ONE declared straight Nav2 interface probe.

    Nav2 commands and AMCL+encoder feedback are the control inputs. Simulator
    body speed is ONLY the independent safe brake-capture interlock, never a
    replacement feedback pose or a commanded wheel-rate measurement.
    """
    def __init__(self, definition, geometry, initial_targets):
        self.definition = checked_short_nav2_definition(definition)
        self.geometry = copy.deepcopy(geometry)
        self.targets = messages.frames.vector(initial_targets, 2, 'original native steering targets').tolist()
        self.stage = 'initial_capture'; self.last = None; self.speed = 0.; self.curvature = 0.
        self.held_since = None; self.quiet_since = None; self.final_since = None
        self.capture_count = 0; self.failed = False; self.done = False

    def step(self, time_s, row, *, hold_ack, feedback, command, nav2_status):
        if self.failed or self.done:raise ValueError('Terminal short Nav2 adapter; no reset/retry')
        try:return self._step(time_s, row, hold_ack, feedback, command, nav2_status)
        except BaseException:
            self.failed = True
            raise

    def _step(self, time_s, row, hold_ack, feedback, command, nav2_status):
        if (not math.isfinite(time_s) or not 0 <= time_s <= self.definition['maximum_physics_duration_s']
                or self.last is not None and not 0 < time_s - self.last <= .25
                or hold_ack is not None and type(hold_ack) is not bool):
            raise ValueError('Original bounded actual physics/control clock and measured backend hold ACK required')
        dt = 0. if self.last is None else time_s - self.last
        self.last = time_s; request = None; desired = 0.
        body_speed = math.hypot(row['vx_physx_m_s'], row['vy_physx_m_s'])
        now_ns = round(row['actual_timeline_s'] * 10**9)
        if nav2_status not in (None, 4):raise RuntimeError('Actual Nav2 action failed/canceled before completion')
        if self.stage == 'initial_capture':
            if time_s > 8.:raise RuntimeError('Original5mm/s initial capture timeout')
            if hold_ack is False and body_speed <= .005:
                request = True; self.capture_count += 1; self.stage = 'pre_hold_pending'
        elif self.stage == 'pre_hold_pending':
            if hold_ack is True:self.held_since = time_s; self.stage = 'pre_hold'
        elif self.stage == 'pre_hold':
            if time_s - self.held_since >= 2.:
                self._feedback(feedback, now_ns)
                estimate = feedback['estimated_map_xyyaw']; start = self.definition['states_xyyaw'][0]
                if math.hypot(estimate[0]-row['x_m'], estimate[1]-row['y_m']) > .03 or abs(
                        messages.frames.wrap(estimate[2]-row['yaw_rad'])) > math.radians(2.):
                    raise RuntimeError('Independent actual localization30mm/2deg admission not met; NO forward')
                if body_speed <= .005 and hold_ack is True:
                    request = False; self.stage = 'release_pending'
        elif self.stage == 'release_pending':
            if hold_ack is False:self.stage = 'path_tracking'
        elif self.stage == 'path_tracking':
            self._feedback(feedback, now_ns)
            if nav2_status == 4:self.stage = 'entry_braking'
            else:
                if (command is None or not 0 <= now_ns-command['received_native_clock_ns'] <= 250000000):
                    raise RuntimeError('Fresh actual Nav2 command required; no stale command fallback')
                linear = messages.frames.vector(command['linear_m_s'], 3, 'actual Nav2 linear command')
                angular = messages.frames.vector(command['angular_rad_s'], 3, 'actual Nav2 angular command')
                if (max(abs(linear[1]), abs(linear[2]), abs(angular[0]), abs(angular[1])) > 1e-8
                        or self.definition['direction']*linear[0] < -1e-8 or abs(linear[0]) > .0300001
                        or abs(angular[2]) > math.radians(8.)):
                    raise RuntimeError('Nav2 command outside declared steering/velocity interface')
                desired = float(linear[0])
                if abs(desired) < 1e-8:
                    if abs(angular[2]) > 1e-7:raise RuntimeError('In-place yaw is not this dual-steer interface')
                else:self.curvature = max(-.70, min(.70, float(angular[2]/desired)))
        elif self.stage == 'entry_braking':
            if abs(self.speed) < 1e-10 and body_speed <= .005:
                # Use the ALREADY physically verified common captured-wheel
                # brake gate: measured <=5mm/s and zero requested velocity.
                # Waiting for an additional, undeclared3mm/s/250ms release
                # window lets the unbraked body creep past the goal. Actual
                # final5s/last3s <=5mm/s acceptance is NOT changed.
                if hold_ack is False:
                    request = True; self.capture_count += 1; self.stage = 'final_hold_pending'
            else:self.quiet_since = None
        elif self.stage == 'final_hold_pending':
            if hold_ack is True:self.final_since = time_s; self.stage = 'final_hold'
        elif self.stage == 'final_hold':
            if time_s-self.final_since >= 5.:self.done = True
        else:raise RuntimeError('Unknown short Nav2 adapter state')
        change = .02*dt
        self.speed += max(-change, min(change, desired-self.speed))
        if abs(self.speed) < 1e-12:self.speed = 0.
        if hold_ack is not False and abs(self.speed) > 1e-8:
            raise RuntimeError('No requested wheel motion while captured/pending')
        if self.stage == 'path_tracking':
            target = [math.atan(self.geometry[key][0]*self.curvature) for key in ('front_axis_xy_m','rear_axis_xy_m')]
            maximum = math.radians(8.)*dt
            self.targets = [old+max(-maximum,min(maximum,new-old)) for old,new in zip(self.targets,target)]
        phase = ('pre_hold' if self.stage in ('initial_capture','pre_hold_pending','pre_hold','release_pending') else
            'path_tracking' if self.stage == 'path_tracking' else 'final_hold')
        return dict(phase=phase, stage=self.stage, speed_m_s=self.speed, curvature_1_m=self.curvature,
            steering_targets_rad=self.targets[:], hold_request=request, fixed_hold_attempts=int(self.final_since is not None),
            actual_Nav2_action_succeeded=nav2_status == 4, done=self.done)

    @staticmethod
    def _feedback(feedback, now_ns):
        if (feedback is None or feedback.get('no_ground_truth_pose_input') is not True
                or not 0 <= now_ns-feedback['stamp_ns'] <= 250000000
                or not 0 <= now_ns-feedback['fix_stamp_ns'] <= 1000000000):
            raise RuntimeError('Actual AMCL + exact measured encoder feedback required, no GT/stale fallback')


async def run_short_sensor_nav2_once(port, sensor, bridge, definition, geometry, *, authorized=False):
    """ONE short native closed loop after all actual SDK and Nav2 readbacks."""
    if authorized is not True:raise ValueError('Explicit ONE guarded short sensor/Nav2 trial required')
    definition = checked_short_nav2_definition(definition)
    report = dict(version='ONE_GUARDED_SHORT_SENSOR_NAV2_INTERFACE_v1', verdict='HOLD', checks={},
        actual_forward_or_route_trials=0, Nav2_success=False, original_hospital_mission_success=False,
        actual_Play_requested=False, automatic_retry=False, definition=definition)
    rows = []; summaries = []; preflight = actuation = False; receipt = None; controller = None
    try:
        report['preflight'] = await port.preflight(); preflight = True
        port.subscribe()
        def capture():
            frame = sensor.capture(); port.frame_archive(frame)
            answer = bridge.frame(frame); stamp = frame['native_encoder']['stamp_ns']
            if answer['stamp_ns'] != stamp:raise RuntimeError('Bridge ACK acquisition timestamp differs')
            summaries.append(dict(stamp_ns=stamp, bridge_receipt=answer))
            return answer
        receipt = capture()
        report['actual_Nav2_preparation'] = bridge.prepare_controller()
        report['checks']['actual_Nav2_parameters_readback_before_Play'] = True
        motor = port.adapter._motor()
        controller = ShortNav2MotionController(definition, geometry,
            [float(motor['positions'][index]) for index in port.adapter.steering])
        def apply(row):
            now = row['time_s']; ack = port.adapter.confirmation(now)
            if (controller.stage == 'pre_hold' and controller.held_since is not None
                    and now-controller.held_since >= 2. and 'actual_Nav2_path_acceptance' not in report):
                accepted = bridge.start_follow_path()
                report['actual_Nav2_path_acceptance'] = accepted
                report['checks']['one_fixed_Nav2_path_accepted_after_measured_prehold'] = True
                receipt['actual_nav2_command'] = accepted['actual_nav2_command']
            feedback = None if receipt is None else receipt['current_matched_feedback']
            command = None if receipt is None else receipt['actual_nav2_command']
            status = None if receipt is None else receipt['actual_nav2_action_status']
            info = controller.step(now, row, hold_ack=ack, feedback=feedback, command=command, nav2_status=status)
            port.guard(row, info)
            if info['hold_request'] is not None:
                port.adapter.request_hold(info['hold_request'], now, command_speed_m_s=0.,
                    measured_speed_m_s=math.hypot(row['vx_physx_m_s'], row['vy_physx_m_s']))
            actual = port.adapter.measured_steering()
            rates = [info['speed_m_s']/(geometry[name]*math.cos(angle)) for name,angle in zip(
                ('front_radius_m','rear_radius_m'),actual)]
            port.adapter.command(info['steering_targets_rad'], rates, now)
            row.update(sensor_nav2_phase=info['stage'], actual_applied_target_speed_m_s=info['speed_m_s'],
                actual_Nav2_action_status=status, final_hold_elapsed_s=0. if controller.final_since is None else now-controller.final_since)
            if feedback is not None:
                estimated = feedback['estimated_map_xyyaw']
                row.update(sensor_feedback_x_m=estimated[0],sensor_feedback_y_m=estimated[1],sensor_feedback_yaw_rad=estimated[2],
                    independent_localization_position_error_m=math.hypot(estimated[0]-row['x_m'],estimated[1]-row['y_m']),
                    independent_localization_heading_error_deg=math.degrees(messages.frames.wrap(estimated[2]-row['yaw_rad'])))
            else:
                row.update(sensor_feedback_x_m=None,sensor_feedback_y_m=None,sensor_feedback_yaw_rad=None,
                    independent_localization_position_error_m=None,independent_localization_heading_error_deg=None)
            return info
        initial = port.measure(); actuation = True; apply(initial); rows.append(initial)
        report['actual_Play_requested'] = True; report['actual_forward_or_route_trials'] = 1; port.play()
        while not controller.done:
            await port.next_sample(); row = port.measure()
            stamp = round(row['actual_timeline_s']*10**9)
            if stamp-summaries[-1]['stamp_ns'] >= 200000000:receipt = capture()
            apply(row); rows.append(row)
        tail = [row for row in rows if row['final_hold_elapsed_s'] >= 2.]
        goal = definition['states_xyyaw'][-1]
        report['checks'].update(actual_Nav2_FollowPath_succeeded=receipt['actual_nav2_action_status'] == 4,
            one_actual_fixed5s_final_hold=controller.final_since is not None and controller.done,
            fixed_final3s_evaluated=bool(tail) and tail[-1]['time_s']-tail[0]['time_s'] >= 2.9,
            final3s_all_position_under30mm=bool(tail) and all(math.hypot(row['x_m']-goal[0],row['y_m']-goal[1]) <= .03 for row in tail),
            final3s_all_heading_under2deg=bool(tail) and all(abs(messages.frames.wrap(row['yaw_rad']-goal[2])) <= math.radians(2.) for row in tail),
            final3s_all_stopped_under5mm_s=bool(tail) and all(math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s']) <= .005 and row['pose_filtered_speed_m_s'] <= .005 for row in tail),
            actual_sensor_feedback_used_not_GT_corrections=True)
        report['actual_sensor_bridge_frames'] = summaries
        report['checks']['actual_observation_within_declared_short_physics_cap']=port.clock<=definition['maximum_physics_duration_s']
    except BaseException as failure:
        report['error'] = type(failure).__name__ + ': ' + str(failure)
        report['checks']['short_motion_and_fixed_hold_completed'] = False
        report['actual_sensor_bridge_frames'] = summaries
    finally:
        import hospital_mission_lifecycle as lifecycle
        report = await lifecycle._finish_observation(port, report, rows, preflight=preflight, actuation=actuation,
            capture_count=0 if controller is None else controller.capture_count)
        try:
            closed = bridge.close()
            evidence.write_exclusive(port.output/'owned_ROS_bridge_closeout.json',closed)
        except Exception as failure:
            evidence.write_exclusive(port.output/'owned_ROS_bridge_closeout_error.json',dict(error=type(failure).__name__+': '+str(failure)))
    return report


class SensorFrameArchive:
    """Exclusive bounded gzip JSONL of original raw observations, no trimming."""
    def __init__(self, path):
        from pathlib import Path
        import gzip
        self.path = Path(path).resolve()
        self.raw = self.path.open('xb')
        self.stream = gzip.GzipFile(fileobj=self.raw, mode='wb', mtime=0, compresslevel=1)
        self.count = 0; self.finished = False

    def __call__(self, frame):
        import json
        if self.finished or self.count >= 228:raise ValueError('Closed/bounded sensor archive; no omitted frame')
        value = evidence.json_value(frame)
        payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(payload) > evidence.MAX_BYTES:raise ValueError('Sensor frame exceeds explicit raw evidence bound')
        self.stream.write(payload); self.count += 1

    def finish(self):
        import os, gzip, json
        if self.finished:raise ValueError('Sensor archive already finalized; no second finalization')
        self.finished = True
        try:
            self.stream.close(); self.raw.flush(); os.fsync(self.raw.fileno())
        finally:self.raw.close()
        #228 full all-return360-beam frames can need ~60MiB at low-latency
        # lossless compression in the actual native
        # capture. This is a storage cap, NOT a scan/motion acceptance limit.
        if self.path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError('Compressed sensor archive exceeds bounded evidence size')
        count = 0; previous = None; capture = None
        with gzip.open(self.path, 'rb') as stream:
            while True:
                line = stream.readline(evidence.MAX_BYTES + 1)
                if not line:break
                if len(line) > evidence.MAX_BYTES or count >= 228:
                    raise ValueError('Sensor archive readback exceeded frame/byte budget')
                frame = json.loads(line); packet = frame['scan_packet']; messages.validate_packet(packet)
                stamp = packet['acquisition']['stamp_ns']; current_capture = packet['acquisition']['capture_id']
                if (frame['native_encoder']['stamp_ns'] != stamp or previous is not None and stamp <= previous
                        or capture is not None and current_capture != capture):
                    raise ValueError('Original scan/encoder archive identity/clock differs')
                previous = stamp; capture = current_capture; count += 1
        if count != self.count:raise ValueError('Raw sensor archive incomplete; no frame-count repair')
        return dict(path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            original_raw_frames=count, exact_scan_encoder_stamps_and_raw_packets_verified=True,
            compressed_bytes=self.path.stat().st_size)


async def observe_zero_sensor_stream_once(port, sensor, *, retain_frame, use_common_hold=False, authorized=False):
    """One actual 3+2s observation through an already checked common SDK host.

    No fresh actor, Stop or forward command. Optional COMMON captured-wheel
    hold changes temporary runtime gains/targets only below measured5mm/s.
    Independent cleanup
    always requests zero and observes Pause, even after a sensor exception.
    The shared original host restores its original targets and checks all
    protected geometry. This is NOT moving localization or a Nav2 route.
    """
    if authorized is not True or not callable(retain_frame):
        raise ValueError('Explicit ONE zero sensor observation and evidence sink required')
    contract = zero_stream_contract(use_common_hold)
    report = dict(version='ONE_RETAINED_ZERO_NATIVE_SENSOR_STREAM_v1', verdict='HOLD',
        checks={}, contract=contract, actual_Play_requested=False,
        actual_forward_or_route_trials=0, Nav2_success=False, dynamic_localization_success=False,
        automatic_retry=False)
    rows = []; frame_summaries = []; preflight = actuation = False
    hold_requested = None; held_since = None; captures = 0
    try:
        report['preflight'] = await port.preflight(); preflight = True
        port.subscribe()
        def capture():
            frame = sensor.capture(); packet = frame['scan_packet']; stamp = packet['acquisition']['stamp_ns']
            if frame_summaries:
                delta = stamp - frame_summaries[-1]['stamp_ns']
                if not contract['minimum_scan_period_ns'] <= delta <= contract['maximum_scan_gap_ns']:
                    raise RuntimeError('Actual scan/encoder stream gap outside original 0.20..0.25s contract')
            if frame['native_encoder']['stamp_ns'] != stamp:
                raise RuntimeError('Actual scan and encoder acquisition timestamps differ')
            retain_frame(frame)
            frame_summaries.append(dict(stamp_ns=stamp, sequence_number=packet['sequence_number'],
                encoder_odom_xyyaw=frame['causal_encoder_estimate']['odom_xyyaw'],
                simulator_scoring_xyz=packet['acquisition']['base_xyz'],
                simulator_scoring_wxyz=packet['acquisition']['base_wxyz'],
                source_role=packet['acquisition']['source_role'],
                actual_scans_and_encoders_same_stamp=True))
        capture()  # Exact initial PAUSED anchor, not a new physics step.
        port.zero(); actuation = True
        report['actual_Play_requested'] = True; port.play()
        previous = 0.; observed = 0.
        while observed < contract['settling_s'] + contract['recording_s']:
            await port.next_sample(); row = port.measure(); now = row['time_s']
            if (not math.isfinite(now) or not previous < now <= contract['maximum_physics_duration_s']
                    or now - previous > contract['maximum_control_gap_s']):
                raise RuntimeError('Original actual physics/control gap exceeded during zero sensor observation')
            port.guard(row, None); previous = now; port.zero()
            if use_common_hold:
                speed = math.hypot(row['vx_physx_m_s'], row['vy_physx_m_s'])
                ack = port.actuator.confirmation(now)
                if hold_requested is None:
                    if now > contract['hold_initial_stop_timeout_s']:
                        raise RuntimeError('Original5mm/s hold capture not reached within fixed8s; no forced hold')
                    if speed <= contract['hold_capture_stop_limit_m_s']:
                        port.actuator.request_hold(True, now, command_speed_m_s=0., measured_speed_m_s=speed)
                        hold_requested = now; captures += 1
                elif held_since is None:
                    if now - hold_requested > contract['hold_backend_ACK_timeout_s']:
                        raise RuntimeError('Captured-wheel backend hold ACK not confirmed')
                    if ack is True:held_since = now
                observed = 0. if held_since is None else now - held_since
            else:observed = now
            row['sensor_observation_phase_s'] = observed
            row['captured_wheel_hold_confirmed'] = held_since is not None
            rows.append(row)
            current_stamp = round(row['actual_timeline_s'] * 10**9)
            if current_stamp - frame_summaries[-1]['stamp_ns'] >= contract['minimum_scan_period_ns']:
                capture()
        tail = [row for row in rows if row['sensor_observation_phase_s'] >= contract['settling_s']]
        report['checks'].update(fixed_actual_3_plus_2s_observation_completed=True,
            complete_fixed_recording_tail=bool(tail) and tail[-1]['time_s'] - tail[0]['time_s'] >= 1.9,
            all_fixed_tail_measured_speeds_under_original5mm_s=bool(tail) and all(
                math.hypot(row['vx_physx_m_s'], row['vy_physx_m_s']) <= contract['final_stop_speed_m_s']
                and row['pose_filtered_speed_m_s'] <= contract['final_stop_speed_m_s'] for row in tail),
            actual_advancing_scan_encoder_frames=len(frame_summaries) >= 21,
            scan_encoder_clock_span_over_fixed_duration=len(frame_summaries) > 1 and
                frame_summaries[-1]['stamp_ns'] - frame_summaries[0]['stamp_ns'] >= 4900000000)
        if use_common_hold:
            report['checks']['one_actual_captured_wheel_hold_confirmed'] = captures == 1 and held_since is not None
    except BaseException as failure:
        report['error'] = type(failure).__name__ + ': ' + str(failure)
        report['checks']['fixed_actual_3_plus_2s_observation_completed'] = False
    finally:
        # Existing common cleanup attempts zero, Pause and restoration
        # independently, then keeps monitors until actual Pause is observed.
        report['sensor_frame_summaries'] = frame_summaries
        report['actual_sensor_frames_including_PAUSED_anchor'] = len(frame_summaries)
        report['actual_held_since_physics_s'] = held_since
        import hospital_mission_lifecycle as lifecycle
        report = await lifecycle._finish_observation(port, report, rows,
            preflight=preflight, actuation=actuation, capture_count=captures)
    return report


def bind_retained_zero_sensor_host(route_path, reference_folder, output, *, authorization,
        prior_observation=None, use_common_hold=False, short_motion=None, authorized=False):
    """Reuse frozen source/context and original SDK guards, not an actor clone.

    This binds only a stationary sensor observation at the CURRENT retained
    pose. It never redefines a hospital task's fixed start/goal. Construction
    reads cached views and writes a NEW exclusive observation protocol only.
    The original route reservation and all static/route outcomes stay intact.
    """
    if authorized is not True or not isinstance(authorization, dict) or not authorization.get('user_message'):
        raise ValueError('Current explicit sensor observation authority required')
    from pathlib import Path
    import time
    import numpy as np
    import hospital_mission_sequence as sequence
    lifecycle = sequence.lifecycle; retained = sequence.retained
    retained.no_other_task()
    route_path = Path(route_path).resolve(); reference_folder = Path(reference_folder).resolve()
    output = Path(output).resolve()
    allowed = retained.GATE / 'hospital_sensor_navigation/live_observation'
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError('New exact exclusive sensor observation directory required')
    context, original, _, plan, geometry = sequence.load_route(route_path)
    short_motion = None if short_motion is None else checked_short_nav2_definition(short_motion)
    reference = retained.read(reference_folder / 'native_scan.json')
    actor = original['actor']
    read, ray = existing_isaac_paused_bindings(actor,first_external_return_only=short_motion is not None); before = read()
    # Camera preferences are checked separately by the original typed monitor.
    # Every native body, collider, layer ID and paused clock must remain exact.
    previous = None
    if prior_observation is not None:
        prior_observation = Path(prior_observation).resolve()
        if not prior_observation.is_relative_to(allowed):raise ValueError('Own sensor observation lineage required')
        previous = retained.read(prior_observation)
        previous_protocol_path = Path(previous['protocol']); previous_protocol = retained.read(previous_protocol_path)
        if (previous.get('version') not in ('ONE_RETAINED_ZERO_NATIVE_SENSOR_STREAM_v1', 'ONE_GUARDED_SHORT_SENSOR_NAV2_INTERFACE_v1')
                or previous['source_protocol_sha256'] != retained.runtime.task.sha(previous_protocol_path)
                or previous['actual_final_timeline'] != before['timeline']
                or previous_protocol['actor'] != actor or previous_protocol['original_route_definition'] != str(route_path)
                or previous['actual_forward_or_route_trials'] not in (0, 1)
                or previous.get('actual_original_H1_to_H6_mission_PASS') is not False):
            raise ValueError('Exact own completed zero-observation boundary required; no retry of prior dispatch')
        for key in ('actual_Pause_confirmed', 'all4997_environment_rows_and145_source_settings_and_physics_unchanged',
                'original_all12joint_gains_targets_and_limits_restored', 'no_unreviewed_USD_edits_or_callback_errors',
                'contacts_healthy_through_actual_Pause', 'original_sensor_frame_archive_durably_verified'):
            if previous['checks'].get(key) is not True:raise ValueError('Prior physical boundary not verified: ' + key)
        lifecycle.verify_workflow_sources(previous_protocol_path.parent / 'executing_sources.zip', previous_protocol['input_sha256'])
    initial_keys = ('actor_root', 'base_xyz', 'base_wxyz', 'body_native_transforms',
        'own_collision_paths', 'layer_identifiers', 'timeline')
    for key in (initial_keys if previous is None else ('actor_root', 'own_collision_paths', 'layer_identifiers')):
        if before[key] != reference['actual_boundary'][key]:
            raise ValueError('Original retained native boundary changed: ' + key)
    sdk = retained.runtime.live_sdk(); simulation = sdk.SimulationManager.get_physics_sim_view()
    if simulation is None:raise ValueError('Existing physics view required; no initialize')
    view = simulation.create_articulation_view(actor + '/base_link')
    snapshot = retained.shared.preparation.old.tensor_snapshot(view)
    if previous is not None and snapshot != previous['postflight']['actual_post_observation_tensor_snapshot']:
        raise ValueError('Exact previous all12joint/13body/gain/target PAUSED snapshot required')
    stage = sdk.omni.usd.get_context().get_stage()
    texts = dict(Root=stage.GetRootLayer().ExportToString(), Session=stage.GetSessionLayer().ExportToString())
    start = lifecycle.RetainedMissionSDKHost._pose(snapshot)
    if short_motion is not None:
        expected_start = short_motion['states_xyyaw'][0]
        if (math.hypot(start[0] - expected_start[0], start[1] - expected_start[1]) > 1e-8
                or abs(messages.frames.wrap(start[2] - expected_start[2])) > 1e-8):
            raise ValueError('Exact independently assigned short start required BEFORE any AMCL result')
        with np.load(original['grid_file']) as grid:
            for pose in short_motion['states_xyyaw']:
                if lifecycle.pose_collision(grid['occupied'], pose, grid['origin_xy_m'], float(grid['resolution_m']), clearance=.010):
                    raise ValueError('Short Nav2 probe intersects original full-body10mm planning raster')
                if lifecycle.support_area_missing(pose, geometry['same_three_authored_support_rectangles']) > 1e-9:
                    raise ValueError('Short Nav2 probe leaves original full-body support union')
    if math.hypot(*snapshot['velocity'][:2]) > (.080 if use_common_hold else .005):
        raise ValueError('Actual retained speed exceeds original5mm/s; no sensor Play')
    contract = zero_stream_contract(use_common_hold)
    protocol = copy.deepcopy(original)
    encoder_folder = retained.GATE / 'hospital_sensor_navigation/encoder_diagnostic'
    encoder_geometry_file = encoder_folder / 'retrospective_h5_r01/source_geometry_GET.json'
    encoder_wrap_file = encoder_folder / 'wrapped_positions_h5_r01/source_wrap_and_cap_contract.json'
    protocol.update(version='RETAINED_EXISTING_SOURCE_ZERO_SENSOR_OBSERVATION_v1',
        output_directory=str(output), original_route_definition=str(route_path),
        original_route_outcome_NOT_changed=True, authorization=copy.deepcopy(authorization),
        required_PAUSED_timeline=before['timeline'], required_tensor_snapshot=snapshot,
        required_layer_sha256=before['layer_sha256'],
        reviewed_display_boundary=dict(reference_Session_text=texts['Session'],
            unchanged_Root_sha256=before['layer_sha256'][0],
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS), policy=retained.TOP_XY_DISPLAY_POLICY),
        stationary_observation_anchor_xyyaw=list(start), hospital_task_start_goal_NOT_changed=True,
        maximum_Play_calls=1, maximum_Stop_reset_restaging_calls=0,
        simulation_budget_s=lifecycle.held_static_contract()['maximum_physics_duration_s'],
        maximum_wall_time_s=contract['maximum_wall_time_s'], automatic_retry=False,
        actual_route_authorized=False, sensor_zero_observation_authorized=True,
        sensor_observation_contract=contract, use_for_final_statistics=False)
    protocol.update(encoder_geometry_file=str(encoder_geometry_file), encoder_wrap_contract_file=str(encoder_wrap_file))
    if short_motion is not None:
        protocol.update(version='RETAINED_SOURCE_SHORT_SENSOR_NAV2_INTERFACE_v1', short_motion=short_motion,
            actual_route_authorized=True, actual_hospital_mission_authorized=False, maximum_route_trials=1,
            maximum_wall_time_s=short_motion['maximum_wall_time_s'],simulation_budget_s=original['simulation_budget_s'],
            maximum_short_observation_physics_s=short_motion['maximum_physics_duration_s'])
    for name in ('hospital_sensor_runtime.py', 'hospital_sensor_messages.py', 'hospital_sensor_navigation.py', 'hospital_sensor_ros_bridge.py'):
        file = Path(__file__).parent / name
        protocol['input_sha256'][str(file.resolve())] = retained.runtime.task.sha(file)
    protocol['input_sha256'][str(route_path)] = retained.runtime.task.sha(route_path)
    for file in (encoder_geometry_file, encoder_wrap_file):
        protocol['input_sha256'][str(file)] = retained.runtime.task.sha(file)
    if previous is not None:
        protocol.update(prior_zero_observation=str(prior_observation), prior_outcome_preserved=previous['verdict'])
        for file in (prior_observation, previous_protocol_path, previous_protocol_path.parent / 'executing_sources.zip'):
            protocol['input_sha256'][str(file)] = retained.runtime.task.sha(file)
    path = output / 'protocol.json'; evidence.write_exclusive(path, protocol)
    lifecycle.freeze_workflow_sources(output / 'executing_sources.zip', protocol['input_sha256'])

    class SensorHost(lifecycle.RetainedMissionSDKHost):
        async def preflight(self):
            result = await retained.RetainedZeroSettleSDKPort.preflight(self)
            self.last_dt = 1. / 120.
            self.runtime_guard(self.measure(), None, self.counts, self.errors)
            return result

        def zero(self):
            if use_common_hold:
                self.adapter.emergency_zero(); self.adapter.confirmation(self.clock)
            else:retained.RetainedZeroSettleSDKPort.zero(self)

        def guard(self, row, info):
            super().guard(row, info)
            if short_motion is not None:
                first = short_motion['states_xyyaw'][0]; last = short_motion['states_xyyaw'][-1]
                dx, dy = row['x_m'] - first[0], row['y_m'] - first[1]; heading = first[2]
                along = short_motion['direction'] * (dx * math.cos(heading) + dy * math.sin(heading))
                lateral = -dx * math.sin(heading) + dy * math.cos(heading)
                distance = math.hypot(lateral, max(0., -along, along - .35))
                if distance > .080 or abs(messages.frames.wrap(row['yaw_rad'] - heading)) > math.radians(10.):
                    raise RuntimeError('Original80mm/10deg independent path-deviation guard')
                return
            metrics = retained.zero_sample_safety_metrics(row, start)
            if metrics['exceeded']:raise RuntimeError('Original retained zero safety guard: ' + str(metrics))

        async def postflight(self):
            value = await super().postflight()
            # Independent lossless readback is pure I/O/numerical work.
            # Do not freeze Kit's main thread while checking hundreds of
            # multi-MiB raw frames AFTER actual Pause and motor restoration.
            import asyncio
            paused=self.timeline_state();snapshot=value['actual_post_mission_tensor_snapshot']
            self._sensor_archive_receipt=await asyncio.to_thread(self.frame_archive.finish)
            if (self.timeline_state()!=paused
                    or retained.shared.preparation.old.tensor_snapshot(self.tensor)!=snapshot):
                raise RuntimeError('Actual PAUSED tensor changed during raw archive verification')
            # SAME numerical checks; only remove misleading hospital-goal
            # labels from this stationary, current-pose observation.
            checks = value['checks']
            for old, new in (('final_actual_goal_position_under30mm', 'final_short_sensor_goal_under30mm' if short_motion else 'final_observation_anchor_drift_under30mm'),
                    ('final_actual_goal_heading_under2deg', 'final_short_sensor_heading_under2deg' if short_motion else 'final_observation_anchor_heading_under2deg')):
                checks[new] = checks.pop(old)
            value['actual_post_observation_tensor_snapshot'] = value.pop('actual_post_mission_tensor_snapshot')
            value['NOT_an_original_hospital_task_goal_or_mission_result'] = True
            return value

        def save(self, report, rows):
            import csv, os
            try:
                report['original_sensor_frame_archive'] = (self._sensor_archive_receipt
                    if hasattr(self,'_sensor_archive_receipt') else self.frame_archive.finish())
                report['checks']['original_sensor_frame_archive_durably_verified'] = True
            except Exception as failure:
                report['sensor_archive_error'] = type(failure).__name__ + ': ' + str(failure)
                report['checks']['original_sensor_frame_archive_durably_verified'] = False
                report['verdict'] = 'HOLD'
            if rows:
                with (output / 'physics_trace.csv').open('x', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                    stream.flush(); os.fsync(stream.fileno())
            report.update(actual_final_timeline=self.timeline_state(),
                protocol=str(path), source_protocol_sha256=retained.runtime.task.sha(path),
                actual_forward_or_route_trials=int(report.get('actual_Play_requested') is True) if short_motion else 0,
                frozen_original_task_start_goal=plan['start_world_xyyaw'],
                zero_observation_anchor_NOT_a_redefined_hospital_start=list(start),
                continuous_zero_sensor_acquisition_verified=not short_motion and report.get('verdict') == 'PASS',
                short_sensor_Nav2_interface_PASS=bool(short_motion) and report.get('verdict') == 'PASS',
                Nav2_success=bool(short_motion) and report.get('verdict') == 'PASS',
                actual_original_H1_to_H6_mission_PASS=False)
            evidence.write_exclusive(output / 'observation_readback.json', report)

    port = object.__new__(SensorHost)
    retained.RetainedZeroSettleSDKPort.__init__(port, sdk, protocol, context=context)
    port.path = path; port.output = output; port.plan = dict(goal_world_xyyaw=list(start))
    guard_geometry = copy.deepcopy(geometry)
    if short_motion is not None:
        port.plan['goal_world_xyyaw'] = short_motion['states_xyyaw'][-1]
        guard_geometry['goal_xyyaw'] = short_motion['states_xyyaw'][-1]
    port.geometry = geometry; port.notice_sub = None; port._notice_callback = None
    port.notice_counts = {'allowed_state_paths': 0, 'reviewed_aperture_paths': 0, 'unreviewed_paths': 0}
    with np.load(original['grid_file']) as grid:
        port.runtime_guard = lifecycle.MissionRuntimeGuard(guard_geometry, grid['occupied'], grid['origin_xy_m'],
            float(grid['resolution_m']), simulation_budget_s=protocol['simulation_budget_s'], static_observation=short_motion is None)
    port.frame_archive = SensorFrameArchive(output / 'original_native_frames.jsonl.gz')
    sensor_config=copy.deepcopy(reference['complete_raw_packet']['configuration'])
    if short_motion is not None:sensor_config['return_model']=messages.FIRST_RETURN_MODEL
    return port, read, ray, sensor_config


def _digest(value):
    import json
    return hashlib.sha256(json.dumps(evidence.json_value(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def same_ros_message_fields(expected, received):
    """Exact generated-message field equality, not noncanonical CDR padding.

    ROS generated equality recursively compares every declared field and
    array. CDR may contain alignment padding, which is not message data and
    need not serialize to identical bytes twice. No numerical tolerance,
    timestamp/frame substitution, or omitted-field comparison is applied.
    """
    return type(received) is type(expected) and bool(received == expected)


def static_pose_message_fields(message):
    """Normalize typed ROS float arrays BEFORE any evidence serialization."""
    pose = message.pose.pose
    position = messages.frames.vector([pose.position.x, pose.position.y, pose.position.z], 3, 'AMCL position')
    quaternion = messages.frames.vector([pose.orientation.w, pose.orientation.x,
        pose.orientation.y, pose.orientation.z], 4, 'AMCL WXYZ quaternion')
    rotation = messages.frames.rotation_wxyz(quaternion)
    covariance = messages.frames.vector(message.pose.covariance, 36, 'AMCL row-major covariance')
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return evidence.json_value(dict(xyyaw=[float(position[0]), float(position[1]), yaw],
        covariance=[float(value) for value in covariance],
        stamp=dict(sec=int(message.header.stamp.sec), nanosec=int(message.header.stamp.nanosec))))


def outward_native_box(centre, half_extent):
    """Enclose exact cells despite PhysX Float3 rounding and boundary contact.

    Independently rounded centres/extents can leave micron-wide cracks on
    exact grid boundaries and entirely miss zero-thickness wall triangles.
    Cover centre rounding plus one native coordinate ULP, then round each
    half extent outward. This is numerical enclosure, not safety inflation.
    """
    import numpy as np
    centre = messages.frames.vector(centre, 3, 'native box centre')
    half = messages.frames.vector(half_extent, 3, 'native box half extent')
    if np.any(half <= 0.):raise ValueError('Positive native half extents required')
    native_centre = np.asarray(centre, dtype=np.float32)
    if not np.isfinite(native_centre).all():raise ValueError('Native Float3 centre overflow')
    extents = []
    for c, h, rounded in zip(centre, half, native_centre):
        lower = np.nextafter(rounded, np.float32(-math.inf))
        upper = np.nextafter(rounded, np.float32(math.inf))
        ulp = max(float(rounded) - float(lower), float(upper) - float(rounded))
        needed = float(h) + abs(float(rounded) - float(c)) + ulp
        bound = float(np.nextafter(np.float32(needed), np.float32(math.inf)))
        if not math.isfinite(bound):raise ValueError('Native Float3 extent overflow')
        extents.append(bound)
    return [float(v) for v in native_centre], extents


def capture_paused_scan(read_boundary, raycast_all, configuration, *, capture_id,
        sequence_number=0, authorized=False):
    """Acquire one complete scan through explicit injected GET-only bindings.

    ``raycast_all(ray)`` must return ALL native hits, including exact own
    shapes and near external returns. A failed query is never an empty ray.
    The injected boundary contains actual native body transforms, immutable
    layer identities/digests, the exact collision inventory and clock.
    No implicit sensor mount, actor inference, synthetic return or retry.
    """
    if authorized is not True:
        raise ValueError('Explicit read-only native acquisition authorization required')
    if not callable(read_boundary) or not callable(raycast_all):
        raise ValueError('Explicit boundary and ALL-hit native query bindings required')
    before = evidence.json_value(read_boundary())
    clock = before['timeline']
    if clock['playing'] is not False or clock['stopped'] is not False:
        raise ValueError('Existing initialized PAUSED scene required; no Play/Stop/init')
    time_s = clock['current_time_s']
    if type(time_s) not in (float, int) or not math.isfinite(time_s) or time_s < 0.:
        raise ValueError('Finite nonnegative actual simulation clock required')
    if set(configuration['own_collision_paths']) != set(before['own_collision_paths']):
        raise ValueError('Exact current collision inventory required, not a prefix mask')
    configuration = copy.deepcopy(configuration)
    acquisition = dict(stamp_ns=round(time_s * 10**9), actor_root=before['actor_root'],
        base_xyz=before['base_xyz'], base_wxyz=before['base_wxyz'],
        source_role='UNVALIDATED_NATIVE_QUERY_CAPTURE', capture_id=capture_id)
    rays = messages.scan_geometry(acquisition, configuration)['rays']
    records = []
    for ray in rays:
        hits = raycast_all(copy.deepcopy(ray))
        if type(hits) not in (list, tuple):
            raise ValueError('Every native query must return a complete hit list')
        records.append(dict(index=ray['index'], hits=hits))
    after = evidence.json_value(read_boundary())
    if after != before:
        raise RuntimeError('Scene/physics body/clock changed during acquisition; no repair')
    packet = messages.scan_packet(acquisition, configuration, records, sequence_number)
    # Exercise wire-size and representation validation BEFORE returning a
    # purportedly encodable acquisition or creating any evidence output.
    encoded = messages.encode_packet(packet)
    status_counts = {}
    for ray in packet['decoded_rays']:
        status_counts[ray['status']] = status_counts.get(ray['status'], 0) + 1
    summary = dict(version=VERSION, status='PAUSED_NATIVE_SCAN_CAPTURED_NOT_NAVIGATION',
        boundary_sha256=_digest(before), actual_boundary=before,
        actual_existing_native_queries=len(rays), complete_raw_ray_count=len(records),
        raw_hit_count=sum(len(r['hits']) for r in records),
        exact_self_hit_count=sum(len(r['exact_self_hits']) for r in packet['decoded_rays']),
        external_hit_count=sum(len(r['all_external_hits']) for r in packet['decoded_rays']),
        return_status_counts=status_counts, encoded_packet_bytes=len(encoded),
        current_native_body_layers_and_PAUSED_clock_unchanged=True,
        one_actual_capture_NOT_a_fresh_stream_or_independent_repetition=True,
        complete_raw_packet=packet, actual_sensor_stream_verified=False,
        localization_success=False, Nav2_success=False, drive_admission=False,
        Play_Stop_reset_scene_physics_or_controller_commands=0, new_motion_trials=0)
    return evidence.json_value(summary)


def existing_isaac_encoder_reader(actor_root):
    """Measured joints from an EXISTING physics view; never initialize/command.

    Read the joint subtype as well as DofType.Rotation: rotation alone does
    not distinguish wrapped from unwrapped revolute. Native Revolute plus
    source/documented wrap bounds still do not measure between-sample slip.
    """
    from isaacsim.core.simulation_manager import SimulationManager
    import omni.timeline
    import omni.usd
    if type(actor_root) is not str or not actor_root.startswith('/World/') or actor_root.endswith('/'):
        raise ValueError('Exact existing encoder actor required')
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    simulation = SimulationManager.get_physics_sim_view()
    if stage is None or simulation is None or timeline.is_playing() or timeline.is_stopped():
        raise ValueError('Existing initialized PAUSED physics view required; no initialization')
    view = simulation.create_articulation_view(actor_root + '/base_link')
    if not bool(view.check()) or view.count != 1:
        raise ValueError('Exact one existing articulation required')
    names = list(view.shared_metatype.dof_names)
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError('All twelve unique measured source joint names required')
    types = [int(value) for value in view.shared_metatype.dof_types]
    joint_types = dict(zip(view.shared_metatype.joint_names,
        [str(value) for value in view.shared_metatype.joint_types]))
    def read():
        if omni.usd.get_context().get_stage() != stage or timeline.is_stopped() or not bool(view.check()):
            raise RuntimeError('Existing encoder stage/handle lost; no automatic recreation')
        before = evidence.timeline_state(timeline)
        positions = [float(value) for value in view.get_dof_positions().tolist()[0]]
        velocities = [float(value) for value in view.get_dof_velocities().tolist()[0]]
        limits = [float(value) for value in view.get_dof_max_velocities().tolist()[0]]
        if evidence.timeline_state(timeline) != before:
            raise RuntimeError('Encoder clock changed within acquisition')
        if any(not math.isfinite(value) for value in (*positions, *velocities, *limits)):
            raise ValueError('Finite complete native encoder channels required')
        return evidence.json_value(dict(actor_root=actor_root,
            stamp_ns=round(before['current_time_s'] * 10**9),
            joint_names=names, positions_rad=positions, velocities_rad_s=velocities,
            native_dof_types=types, native_maximum_rates_rad_s=limits,
            native_joint_types_by_name=joint_types,
            source='EXISTING_NATIVE_ARTICULATION_ENCODER_GET',
            commands_modes_or_ground_truth_used_as_encoder_inputs=False,
            Rotation_alone_NOT_used_to_infer_Revolute_subtype=True))
    return read


class EncoderOdometrySequence:
    """Causal, mode-blind position-increment odometry, never GT correction.

    Explicit source geometry/wrap/rate contracts are not calibrated slip or
    uncertainty guarantees. No covariance is invented. This is an estimator
    component, not localization or motion admission. A missing/aliased frame
    is terminal; no stamp reset, velocity-target substitute or jump clipping.
    """
    CHANNELS = ('front_active_steering_to_drive_wheel', 'rear_active_steering_to_drive_wheel',
        'base_link_to_front_active_steering', 'base_link_to_rear_active_steering')

    def __init__(self, *, actor_root, geometry, wrap_period_rad,
            maximum_abs_rate_rad_s, position_roundoff_rad, maximum_sample_gap_s):
        if type(actor_root) is not str or not actor_root.startswith('/World/') or actor_root.endswith('/'):
            raise ValueError('Explicit estimator actor required')
        # Validate the complete geometry without fitting any observed motion.
        messages.frames.centerline_encoder_twist(0., 0., 0., 0., geometry=geometry)
        if (type(maximum_sample_gap_s) not in (int, float) or not math.isfinite(maximum_sample_gap_s)
                or not 0. < maximum_sample_gap_s <= .25):
            raise ValueError('Explicit bounded encoder acquisition gap required')
        self.actor = actor_root; self.geometry = copy.deepcopy(geometry)
        self.contract = dict(wrap_period_rad=wrap_period_rad,
            maximum_abs_rate_rad_s=copy.deepcopy(maximum_abs_rate_rad_s),
            position_roundoff_rad=position_roundoff_rad)
        # Exercise all wrap/rate/roundoff validation BEFORE accepting a frame.
        messages.frames.bounded_wrapped_joint_increments([0., maximum_sample_gap_s],
            [[0., 0.], [0., 0.]], **self.contract)
        self.maximum_gap = maximum_sample_gap_s
        self.previous = None; self.pose = [0., 0., 0.]; self.failed = False
        self.identity = _digest([self.actor, self.geometry, self.contract, self.maximum_gap])

    def accept(self, sample):
        if self.failed:raise ValueError('Encoder stream terminal; no automatic restart')
        try:
            if self.identity != _digest([self.actor, self.geometry, self.contract, self.maximum_gap]):
                raise ValueError('Frozen encoder geometry/contract changed')
            sample = evidence.json_value(sample)
            if (sample['actor_root'] != self.actor or sample['source'] != 'EXISTING_NATIVE_ARTICULATION_ENCODER_GET'
                    or sample['commands_modes_or_ground_truth_used_as_encoder_inputs'] is not False):
                raise ValueError('Exact native encoder actor/source required')
            messages.stamp_parts(sample['stamp_ns'])
            names = sample['joint_names']
            if len(names) != 12 or len(set(names)) != 12:
                raise ValueError('All unique twelve joint channels required')
            for key in ('positions_rad', 'velocities_rad_s', 'native_dof_types', 'native_maximum_rates_rad_s'):
                if len(sample[key]) != len(names):raise ValueError('Complete native joint arrays required')
            q = messages.frames.vector(sample['positions_rad'], 12, 'native positions')
            qd = messages.frames.vector(sample['velocities_rad_s'], 12, 'native velocities')
            caps = messages.frames.vector(sample['native_maximum_rates_rad_s'], 12, 'native rate limits')
            indices = [names.index(name) for name in self.CHANNELS]
            if any(sample['native_dof_types'][index] != 0 for index in indices):
                raise ValueError('Measured rotational active joints required')
            if any(sample['native_joint_types_by_name'].get(name) != 'JointType.Revolute'
                    for name in self.CHANNELS):
                raise ValueError('Actual native Revolute subtype required, not inferred from Rotation')
            for offset, index in enumerate(indices[:2]):
                cap = self.contract['maximum_abs_rate_rad_s'][offset]
                if not 0. < caps[index] <= cap or abs(qd[index]) > cap:
                    raise ValueError('Actual native drive rate/limit violates source contract')
                if abs(q[index]) > self.contract['wrap_period_rad'] / 2. + self.contract['position_roundoff_rad']:
                    raise ValueError('Native drive position outside declared wrapped representation')
            pose = self.pose[:]; twist = None; increment = None
            if self.previous is not None:
                previous = self.previous
                if names != previous['joint_names']:
                    raise ValueError('Joint channel ordering changed')
                dt = (sample['stamp_ns'] - previous['stamp_ns']) / 10**9
                if not 0. < dt <= self.maximum_gap:
                    raise ValueError('Nonadvancing or missing native encoder acquisition')
                old = previous['positions_rad']
                increment = messages.frames.bounded_wrapped_joint_increments([0., dt],
                    [[old[i] for i in indices[:2]], [float(q[i]) for i in indices[:2]]], **self.contract)
                rates = increment['interval_mean_rates_rad_s'][0]
                steer = [(old[i] + float(q[i])) / 2. for i in indices[2:]]
                twist = messages.frames.centerline_encoder_twist(*rates, *steer, geometry=self.geometry)
                value = twist['body_twist_u_v_r']
                pose = messages.frames.integrate_body_twist_samples([0., dt], [value, value],
                    initial_xyyaw=pose, maximum_sample_gap_s=self.maximum_gap)[-1]
            result = evidence.json_value(dict(stamp_ns=sample['stamp_ns'], odom_xyyaw=pose,
                measured_interval_twist=twist, wrapped_increment_diagnostic=increment,
                initial_odom_is_identity_NOT_world_or_ground_truth=True,
                encoder_uncertainty_or_common_slip_NOT_validated=True,
                covariance_NOT_invented=True, drive_admission=False))
            self.pose = pose; self.previous = copy.deepcopy(sample)
            return result
        except Exception:
            self.failed = True
            raise


class TimestampedNativeSensorSequence:
    """Passive acquisition sidecar: scan and encoders share an actual stamp.

    It NEVER starts/stops physics or issues an actuator command. A separately
    admitted host must supply its original environment/safety monitor. Only
    that host can admit a live observation or route. Missing observations are
    terminal, not replacement scans, interpolated anchors or a clock reset.
    World body pose is used ONLY to locate native virtual ray queries. The
    encoder estimate starts at identity and receives no world pose fields.
    """
    def __init__(self, read_boundary, raycast_all, read_encoders, configuration, *,
            check_environment, encoder_estimator, capture_id, authorized=False):
        if authorized is not True:raise ValueError('Explicit passive native sensor authority required')
        if not all(callable(f) for f in (read_boundary, raycast_all, read_encoders, check_environment)):
            raise ValueError('Explicit complete sensor bindings and admitted host monitor required')
        if not isinstance(encoder_estimator, EncoderOdometrySequence) or encoder_estimator.previous is not None:
            raise ValueError('Unused exact causal encoder estimator required')
        before = evidence.json_value(read_boundary())
        if before['timeline']['playing'] or before['timeline']['stopped']:
            raise ValueError('Existing PAUSED sensor-sidecar binding required; no Play/init')
        if (before['actor_root'] != encoder_estimator.actor
                or set(configuration['own_collision_paths']) != set(before['own_collision_paths'])):
            raise ValueError('Same explicit actor/collider sensor and encoder binding required')
        if type(capture_id) is not str or not capture_id:raise ValueError('Explicit sensor capture identity required')
        self.read = read_boundary; self.raycast = raycast_all; self.encoders = read_encoders
        self.check = check_environment; self.estimator = encoder_estimator
        self.configuration = copy.deepcopy(configuration); self.capture_id = capture_id
        self.identity = (before['actor_root'], before['layer_identifiers'], before['own_collision_paths'])
        self.sequence = messages.PacketSequence(); self.count = 0; self.failed = False
        self.configuration_digest = _digest(configuration)

    def capture(self):
        if self.failed:raise ValueError('Native sensor stream terminal; no automatic restart')
        try:
            self.check(); before = evidence.json_value(self.read())
            identity = (before['actor_root'], before['layer_identifiers'], before['own_collision_paths'])
            if identity != self.identity or before['timeline']['stopped']:
                raise RuntimeError('Same live actor/layers/collider inventory required; no Stop/reset')
            if _digest(self.configuration) != self.configuration_digest:
                raise ValueError('Frozen sensor configuration changed')
            stamp = round(before['timeline']['current_time_s'] * 10**9)
            encoder = evidence.json_value(self.encoders())
            if encoder['stamp_ns'] != stamp:raise ValueError('Exact scan/encoder acquisition stamp required')
            acquisition = dict(stamp_ns=stamp, actor_root=before['actor_root'],
                base_xyz=before['base_xyz'], base_wxyz=before['base_wxyz'],
                source_role='UNVALIDATED_NATIVE_QUERY_CAPTURE', capture_id=self.capture_id)
            records = [dict(index=ray['index'], hits=self.raycast(copy.deepcopy(ray)))
                for ray in messages.scan_geometry(acquisition, self.configuration)['rays']]
            if evidence.json_value(self.encoders()) != encoder or evidence.json_value(self.read()) != before:
                raise RuntimeError('Native body/joints/clock changed within atomic acquisition')
            self.check()
            if evidence.json_value(self.read()) != before or evidence.json_value(self.encoders()) != encoder:
                raise RuntimeError('Native boundary changed during host monitor check')
            packet = messages.scan_packet(acquisition, self.configuration, records, self.count)
            # Constructed immediately above from EVERY native raw return;
            # sequence acceptance must not reconstruct the same large packet
            # a second time. The process/storage boundary still reconstructs
            # and compares the entire original source independently.
            self.sequence._accept_immediately_validated(packet)
            odom = self.estimator.accept(encoder)
            result = dict(scan_packet=packet, native_encoder=encoder, causal_encoder_estimate=odom,
                exact_same_scan_encoder_acquisition_stamp=True,
                world_pose_NOT_forwarded_to_encoder_estimator=True,
                separately_admitted_host_environment_and_motion_gate_required=True,
                continuous_sensor_or_localization_or_Nav2_success=False, timeline_or_actuator_commands=0)
            self.count += 1
            return result
        except Exception:
            self.failed = True
            raise


def existing_isaac_paused_bindings(actor_root,*,first_external_return_only=False,endpoint_surface_consistency=False):
    """Bind cached native GETs only, inside the ALREADY running Isaac app.

    Native PhysX poses, not potentially stale USD/Fabric transforms, locate
    the scan. The complete current active body/collider inventory is kept.
    Schema API wrappers READ fields; no Apply/Create/Set or app update occurs.
    """
    import carb
    import omni.physx
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdPhysics, UsdGeom
    import numpy as np

    if (type(first_external_return_only) is not bool or type(endpoint_surface_consistency) is not bool
            or type(actor_root) is not str or not actor_root.startswith('/World/')
            or actor_root.endswith('/')):
        raise ValueError('Explicit exact existing actor root required')
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise ValueError('Existing initialized PAUSED hospital stage required')
    root = stage.GetPrimAtPath(actor_root)
    if not root.IsValid() or not root.IsActive():
        raise ValueError('Exact admitted existing actor must be active')
    physics = omni.physx.get_physx_interface()
    queries = omni.physx.get_physx_scene_query_interface()
    layers = (stage.GetRootLayer(), stage.GetSessionLayer())
    triangle_cache = {}

    def certified_hit(ray, hit):
        expected = np.asarray(ray['origin_world_m']) + hit['distance_m'] * np.asarray(ray['direction_world'])
        if np.max(np.abs(np.asarray(hit['position_m']) - expected)) <= 1e-4:
            return hit
        path = hit['collision']
        if path in collisions_if_known:
            raise ValueError('Own native shape position inconsistent; no mesh-source substitution')
        if path not in triangle_cache:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsA(UsdGeom.Mesh) or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'none':
                raise ValueError('Inconsistent native position has no exact original triangle-mesh source')
            ancestor = prim
            while ancestor.IsValid() and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError('Source certification is restricted to STATIC external meshes')
                ancestor = ancestor.GetParent()
            mesh = UsdGeom.Mesh(prim); counts = list(mesh.GetFaceVertexCountsAttr().Get())
            if not counts or len(counts) > 200000 or any(count != 3 for count in counts):
                raise ValueError('Bounded original triangulated source required for range certification')
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            points = np.asarray([list(matrix.Transform(point)) for point in mesh.GetPointsAttr().Get()], dtype=float)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
            triangles = points[indices]
            triangle_cache[path] = (triangles, hashlib.sha256(triangles.tobytes()).hexdigest(),
                bool(mesh.GetDoubleSidedAttr().Get()))
        triangles, digest, double_sided = triangle_cache[path]
        return certify_native_range_position(ray, hit, triangles, source_sha256=digest,double_sided=double_sided,
            endpoint_surface_consistency=endpoint_surface_consistency)

    # Exact own identities, never an actor-path prefix mask.
    collisions_if_known = frozenset(str(child.GetPath()) for child in
        Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if child.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False)

    def read_boundary():
        if omni.usd.get_context().get_stage() != stage:
            raise RuntimeError('Original stage changed; no new binding or recovery')
        if (stage.GetRootLayer().identifier, stage.GetSessionLayer().identifier) != tuple(
                layer.identifier for layer in layers):
            raise RuntimeError('Original Root/Session identity changed; no cached-layer substitution')
        prim = stage.GetPrimAtPath(actor_root)
        if not prim.IsValid() or not prim.IsActive():
            raise RuntimeError('Existing actor no longer active')
        bodies, collisions = {}, []
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not child.IsActive():
                continue
            path = str(child.GetPath())
            if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(
                    child).GetCollisionEnabledAttr().Get() is not False:
                collisions.append(path)
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                pose = physics.get_rigidbody_transformation(path)
                if pose.get('ret_val') is not True:
                    raise RuntimeError('Cached native rigid-body pose unavailable: ' + path)
                xyz = list(map(float, pose['position']))
                xyzw = list(map(float, pose['rotation']))
                bodies[path] = dict(xyz=xyz, wxyz=[xyzw[3], *xyzw[:3]])
        if len(bodies) != 13 or not collisions or actor_root + '/base_link' not in bodies:
            raise ValueError('Original existing 13-body actor and exact colliders required')
        base = bodies[actor_root + '/base_link']
        return evidence.json_value(dict(actor_root=actor_root, base_xyz=base['xyz'],
            base_wxyz=base['wxyz'], body_native_transforms=bodies,
            own_collision_paths=sorted(collisions),
            layer_identifiers=[layer.identifier for layer in layers],
            layer_sha256=[hashlib.sha256(layer.ExportToString().encode()).hexdigest() for layer in layers],
            timeline=evidence.timeline_state(timeline)))

    def raycast_all(ray):
        hits, errors = [], []
        def receive(hit):
            try:
                hits.append(dict(collision=str(hit.collision), distance_m=float(hit.distance),
                    position_m=list(map(float, hit.position)), normal=list(map(float, hit.normal))))
            except Exception as error:
                errors.append(type(error).__name__ + ': ' + str(error))
            return True
        queries.raycast_all(carb.Float3(*ray['origin_world_m']),
            carb.Float3(*ray['direction_world']), ray['length_m'], receive)
        if errors:
            raise RuntimeError('Native ray callback failed; not free space: ' + str(errors))
        if first_external_return_only:
            # FIRST-return lidar cannot observe a farther occluded surface.
            # Keep its original native point verbatim, never certify/repair
            # it. The message decoder still audits every scalar and range.
            external=[hit['distance_m'] for hit in hits if hit['collision'] not in collisions_if_known]
            first=min(external,default=None)
            return [certified_hit(ray,hit) if hit['collision'] in collisions_if_known or hit['distance_m']==first else hit
                for hit in hits]
        return [certified_hit(ray, hit) for hit in hits]

    return read_boundary, raycast_all


async def capture_paused_plane_map(read_boundary, overlap_external, configuration, *,
        next_update, authorized=False, maximum_wall_s=120., cells_per_update=2048):
    """Registered, sensor-plane GEOMETRY reference map, not sensor-only SLAM.

    An external collider overlapping a thin cell-aligned box marks that cell
    occupied. Empty results mean no external collision geometry in that slab,
    NOT robot-footprint clearance or floor support. Never use this in place of
    the existing full-body planning/safety map. Native query poses are virtual
    map-authoring probes; no robot or simulated sensor is moved to these cells.
    """
    if authorized is not True:
        raise ValueError('Explicit read-only registered reference-map authorization required')
    required = {'origin_xy_m', 'shape_rows_columns', 'resolution_m', 'plane_z_m', 'half_thickness_m'}
    if type(configuration) is not dict or set(configuration) != required:
        raise ValueError('Complete explicit map geometry required')
    origin = configuration['origin_xy_m']; shape = configuration['shape_rows_columns']
    if (len(origin) != 2 or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
            or math.prod(shape) > MAX_REFERENCE_MAP_CELLS):
        raise ValueError('Bounded explicit two-dimensional registered map required')
    resolution = configuration['resolution_m']; height = configuration['plane_z_m']
    thickness = configuration['half_thickness_m']
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (*origin, resolution, height, thickness))
            or resolution <= 0. or thickness <= 0.
            or type(cells_per_update) is not int or not 1 <= cells_per_update <= 4096
            or type(maximum_wall_s) not in (int, float) or not math.isfinite(maximum_wall_s)
            or not 0. < maximum_wall_s <= 180.):
        raise ValueError('Finite bounded map dimensions/query budget required')
    if not all(callable(f) for f in (read_boundary, overlap_external, next_update)):
        raise ValueError('Explicit GET-only native queries and app-yield binding required')
    before = evidence.json_value(read_boundary())
    if before['timeline']['playing'] is not False or before['timeline']['stopped'] is not False:
        raise ValueError('Existing initialized PAUSED map-authoring boundary required')
    import time
    started = time.monotonic(); data = []; rows, columns = shape
    for row in range(rows):
        for column in range(columns):
            if time.monotonic() - started > maximum_wall_s:
                raise TimeoutError('Bounded reference-map acquisition incomplete; no partial free map')
            centre = [origin[0] + (column + .5) * resolution,
                origin[1] + (row + .5) * resolution, height]
            occupied = overlap_external(centre, [resolution / 2., resolution / 2., thickness])
            if type(occupied) is not bool:
                raise ValueError('Explicit complete native overlap result required, not missing/free')
            data.append(100 if occupied else 0)
            if len(data) % cells_per_update == 0:
                if evidence.json_value(read_boundary()) != before:
                    raise RuntimeError('Map environment/body/clock changed; no map repair')
                await next_update()
    if evidence.json_value(read_boundary()) != before:
        raise RuntimeError('Map environment/body/clock changed before completion')
    return evidence.json_value(dict(version='PAUSED_NATIVE_SENSOR_PLANE_REFERENCE_MAP_v1',
        status='REGISTERED_REFERENCE_MAP_CAPTURED_NOT_LOCALIZATION_OR_DRIVE',
        configuration=copy.deepcopy(configuration), boundary_sha256=_digest(before),
        actual_boundary=before, data=data, row_major_bottom_left_origin=True,
        occupied_cell_count=data.count(100), free_cell_count=data.count(0),
        actual_native_cell_queries=len(data), elapsed_wall_s=time.monotonic() - started,
        native_body_layers_and_PAUSED_clock_unchanged=True,
        exact_own_colliders_excluded=before['own_collision_paths'],
        map_authoring_uses_registered_virtual_queries_NOT_sensor_only_SLAM=True,
        source_is_collision_geometry_NOT_visual_or_real_sensor_response=True,
        native_query_metadata=getattr(overlap_external, 'query_metadata', {}),
        existing_planning_full_body_safety_or_support_map_NOT_replaced=True,
        localization_success=False, Nav2_success=False, drive_admission=False,
        Play_Stop_reset_scene_physics_or_controller_commands=0, new_motion_trials=0))


def existing_isaac_plane_overlap(own_collision_paths):
    """Cached overlap_box only: no mesh/shape cooking or scene mutation."""
    import carb
    import omni.physx
    own = frozenset(own_collision_paths)
    if not own or any(type(path) is not str or not path.startswith('/World/') for path in own):
        raise ValueError('Exact nonempty existing collision inventory required')
    queries = omni.physx.get_physx_scene_query_interface()
    def overlap_external(centre, half_extent):
        native_centre, native_half = outward_native_box(centre, half_extent)
        occupied = False; errors = []
        def receive(hit):
            nonlocal occupied
            try:
                path = str(hit.collision)
                if not path.startswith('/World/'):
                    raise ValueError('Unexpected native collision identity: ' + path)
                if path not in own:
                    occupied = True
                    return False
            except Exception as error:
                errors.append(type(error).__name__ + ': ' + str(error))
                return False
            return True
        queries.overlap_box(carb.Float3(*native_half), carb.Float3(*native_centre),
            carb.Float4(0., 0., 0., 1.), receive, False)
        if errors:
            raise RuntimeError('Native map query failed; not free space: ' + str(errors))
        return occupied
    overlap_external.query_metadata = dict(outward_Float3_cell_enclosure=True,
        compensates_centre_rounding_and_one_native_coordinate_ULP=True,
        NOT_a_robot_clearance_or_geometry_change=True)
    return overlap_external


def reference_map_data(record):
    """Exact bounded raster, optionally in a hash-pinned compressed archive."""
    from pathlib import Path
    import numpy as np
    if (record.get('version') != 'PAUSED_NATIVE_SENSOR_PLANE_REFERENCE_MAP_v1'
            or record.get('native_body_layers_and_PAUSED_clock_unchanged') is not True
            or record.get('existing_planning_full_body_safety_or_support_map_NOT_replaced') is not True):
        raise ValueError('Original complete registered plane-map record required')
    config = record['configuration']; rows, columns = config['shape_rows_columns']
    if type(rows) is not int or type(columns) is not int or not 0 < rows * columns <= MAX_REFERENCE_MAP_CELLS:
        raise ValueError('Explicit bounded complete reference-map dimensions required')
    if ('data' in record) == ('data_archive' in record):
        raise ValueError('Exactly one original map data representation required')
    if 'data' in record:
        data = record['data']
    else:
        archive = record['data_archive']; path = Path(archive['path'])
        # Check compressed AND uncompressed budgets before allocating a raster.
        import zipfile
        if (archive.get('dtype') != 'uint8' or archive.get('shape') != [rows, columns]
                or not path.is_absolute() or path.stat().st_size > 2 * 1024 * 1024
                or hashlib.sha256(path.read_bytes()).hexdigest() != archive['sha256']):
            raise ValueError('Exact bounded reference-map archive digest required')
        with zipfile.ZipFile(path) as zipped:
            if (zipped.namelist() != ['data.npy']
                    or zipped.getinfo('data.npy').file_size > MAX_REFERENCE_MAP_CELLS + 1024):
                raise ValueError('Bounded single uint8 raster archive required')
        with np.load(path, allow_pickle=False) as saved:
            values = saved['data']
            if values.dtype != np.uint8 or values.shape != (rows, columns):
                raise ValueError('Exact uint8 reference-map raster shape required')
            data = values.ravel().tolist()
    if (
            type(data) is not list or len(data) != rows * columns
            or any(type(value) is not int or value not in (0, 100) for value in data)
            or record['actual_native_cell_queries'] != len(data)
            or record['occupied_cell_count'] != data.count(100)
            or record['free_cell_count'] != data.count(0)):
        raise ValueError('Exact complete registered row-major map required; no missing/free cells')
    if config['resolution_m'] <= 0. or not all(math.isfinite(v) for v in
            (*config['origin_xy_m'], config['resolution_m'])):
        raise ValueError('Finite positive map geometry required')
    return data


def archive_reference_map(path, record):
    """Exclusive compressed numerical evidence; never a source USD rewrite.

    The small JSON retains full native boundary and a SHA-pinned uint8 array.
    This avoids weakening the common 4-MiB JSON evidence budget for large maps.
    Existing files are never replaced, even if a partial prior write exists.
    """
    import io
    import os
    from pathlib import Path
    import numpy as np
    path = Path(path).resolve(); binary = path.with_suffix('.npz')
    if path.exists() or binary.exists():raise FileExistsError('Existing reference-map evidence preserved')
    data = reference_map_data(record)
    output = io.BytesIO()
    np.savez_compressed(output, data=np.asarray(data, dtype=np.uint8).reshape(record['configuration']['shape_rows_columns']))
    payload = output.getvalue()
    if len(payload) > 2 * 1024 * 1024:raise ValueError('Bounded compressed map evidence required')
    metadata = {key: value for key, value in record.items() if key != 'data'}
    metadata['data_archive'] = dict(path=str(binary), sha256=hashlib.sha256(payload).hexdigest(),
        dtype='uint8', shape=record['configuration']['shape_rows_columns'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with binary.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if reference_map_data(metadata) != data:
        raise RuntimeError('Reference-map archive verification failed; original bytes retained')
    return evidence.write_exclusive(path, metadata)


def scan_reference_coverage(scan_report, map_report):
    """Analytical coverage check, not a claim of map/SLAM or driving safety."""
    if scan_report['actual_boundary'] != map_report['actual_boundary']:
        raise ValueError('Exact same registered scan/map boundary required')
    packet = scan_report['complete_raw_packet']; config = map_report['configuration']
    lower = config['origin_xy_m']; rows, columns = config['shape_rows_columns']
    upper = [lower[0] + columns * config['resolution_m'], lower[1] + rows * config['resolution_m']]
    points = [ray['nearest_external_hit']['position_m'] for ray in packet['decoded_rays']
        if ray['nearest_external_hit'] is not None]
    inside = sum(all(lower[i] <= point[i] < upper[i] for i in (0, 1)) for point in points)
    origin = packet['sensor_origin_world_m']; radius = packet['configuration']['range_max_m']
    circle = all(lower[i] <= origin[i] - radius and upper[i] >= origin[i] + radius for i in (0, 1))
    dz = max((abs(point[2] - config['plane_z_m']) for point in points), default=0.)
    return evidence.json_value(dict(returned_endpoints=len(points), returned_endpoints_inside_map=inside,
        full_maximum_range_XY_circle_covered=circle,
        maximum_return_height_difference_from_reference_plane_m=dz,
        returned_points_within_declared_plane_slab=dz <= config['half_thickness_m'],
        unobserved_map_or_footprint_collision_guarantee=False, drive_admission=False))


def plane_map_message(record, *, map_frame_id):
    """Encode the explicit registered plane map, never a planning-map swap."""
    messages.frame_id(map_frame_id)
    data = reference_map_data(record)
    config = record['configuration']; rows, columns = config['shape_rows_columns']
    from nav_msgs.msg import OccupancyGrid
    result = OccupancyGrid()
    result.header.frame_id = map_frame_id
    stamp = messages.stamp_parts(round(record['actual_boundary']['timeline']['current_time_s'] * 10**9))
    result.header.stamp.sec = stamp['sec']; result.header.stamp.nanosec = stamp['nanosec']
    result.info.map_load_time = result.header.stamp
    result.info.resolution = config['resolution_m']
    result.info.width = columns; result.info.height = rows
    result.info.origin.position.x = float(config['origin_xy_m'][0])
    result.info.origin.position.y = float(config['origin_xy_m'][1])
    result.info.origin.orientation.w = 1.
    result.data = data
    return result


def verify_static_amcl(scan_report, map_report, *, domain_id=79, wall_budget_s=30.,
        laser_model_type='likelihood_field', maximum_particles=2000,
        laser_sigma_hit_m=.1, authorized=False):
    """Run ISOLATED AMCL on one stationary real query, NOT a motion trial.

    Map registration and the initial prior use simulator-known coordinates;
    odom->base is a disclosed stationary identity, not wheel odometry. The
    same scan is re-published for delivery only, never counted as fresh data.
    The result records errors, not an unearned hospital navigation PASS.
    This creates ONLY an isolated AMCL process and data/service publishers.
    """
    if authorized is not True:
        raise ValueError('Explicit isolated static-localization diagnostic authorization required')
    if type(domain_id) is not int or not 1 <= domain_id <= 100:
        raise ValueError('Explicit nondefault isolated ROS domain required')
    if type(wall_budget_s) not in (int, float) or not math.isfinite(wall_budget_s) or not 0. < wall_budget_s <= 45.:
        raise ValueError('Bounded static-localization wall clock required')
    if laser_model_type not in ('likelihood_field', 'likelihood_field_prob'):
        raise ValueError('Explicit reviewed AMCL likelihood-field model required')
    if type(maximum_particles) is not int or not 2000 <= maximum_particles <= 10000:
        raise ValueError('Explicit bounded AMCL particle computation budget required')
    if (type(laser_sigma_hit_m) not in (int, float) or not math.isfinite(laser_sigma_hit_m)
            or not .02 <= laser_sigma_hit_m <= .2):
        raise ValueError('Explicit bounded AMCL likelihood width required')
    if (scan_report.get('version') != VERSION
            or scan_report.get('current_native_body_layers_and_PAUSED_clock_unchanged') is not True
            or map_report.get('version') != 'PAUSED_NATIVE_SENSOR_PLANE_REFERENCE_MAP_v1'):
        raise ValueError('Original native scan and registered map reports required')
    if scan_report.get('actual_boundary') != map_report.get('actual_boundary'):
        raise ValueError('Same actual PAUSED scan/map environment required')
    packet = scan_report['complete_raw_packet']
    if packet['acquisition']['source_role'] != 'UNVALIDATED_NATIVE_QUERY_CAPTURE':
        raise ValueError('Explicit native observation required; not a software sensor fixture')
    objects = messages.ros_message_objects(packet)
    if objects['base_to_sensor_tf'] is None:
        raise ValueError('Explicit child-frame laser mount required for this isolated static probe')
    map_frame = 'ijcas_sensor_dev_map'; odom_frame = 'ijcas_sensor_dev_odom'
    grid = plane_map_message(map_report, map_frame_id=map_frame)
    base_frame = packet['configuration']['base_frame_id']
    xyz = packet['acquisition']['base_xyz']; q = packet['acquisition']['base_wxyz']
    rotation = messages.frames.rotation_wxyz(q)
    truth = [xyz[0], xyz[1], math.atan2(rotation[1, 0], rotation[0, 0])]
    prior = [truth[0] + .1, truth[1] - .1, truth[2] + math.radians(5.)]
    import os
    import subprocess
    import time
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
    from tf2_msgs.msg import TFMessage
    from sensor_msgs.msg import LaserScan
    from rosgraph_msgs.msg import Clock
    from nav_msgs.msg import OccupancyGrid
    from lifecycle_msgs.srv import ChangeState, GetState
    from rcl_interfaces.srv import GetParameters
    from rclpy.context import Context
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import SingleThreadedExecutor

    prefix = '/ijcas_sensor_dev/static'
    parameters = dict(use_sim_time=True, set_initial_pose=False, global_frame_id=map_frame,
        odom_frame_id=odom_frame, base_frame_id=base_frame, tf_broadcast=True,
        scan_topic=prefix + '/scan', robot_model_type='nav2_amcl::OmniMotionModel',
        min_particles=500, max_particles=maximum_particles, max_beams=90,
        laser_min_range=packet['scan']['range_min'], laser_max_range=packet['scan']['range_max'],
        laser_model_type=laser_model_type, do_beamskip=False,
        sigma_hit=laser_sigma_hit_m, z_hit=.8, z_rand=.2,
        update_min_d=.02, update_min_a=.02, transform_tolerance=.5,
        resample_interval=1)  # Same explicit interval as the existing project AMCL YAML.
    command = ['/opt/ros/humble/lib/nav2_amcl/amcl', '--ros-args', '-r', '__node:=ijcas_sensor_static_amcl']
    for source, destination in (('/tf', prefix + '/tf'), ('/tf_static', prefix + '/tf_static'),
            ('map', prefix + '/map'), ('amcl_pose', prefix + '/amcl_pose'),
            ('initialpose', prefix + '/initialpose'), ('/clock', prefix + '/clock')):
        command.extend(['-r', source + ':=' + destination])
    for key, value in parameters.items():
        text = str(value).lower() if type(value) is bool else str(value)
        command.extend(['-p', key + ':=' + text])
    environment = dict(os.environ, ROS_DOMAIN_ID=str(domain_id))
    context = Context(); node = executor = process = None
    estimates = []; log = ''; started = time.monotonic()
    try:
        rclpy.init(args=[], context=context, domain_id=domain_id)
        node = rclpy.create_node('ijcas_static_localization_probe', context=context)
        executor = SingleThreadedExecutor(context=context); executor.add_node(node)
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        map_pub = node.create_publisher(OccupancyGrid, prefix + '/map', reliable)
        tf_pub = node.create_publisher(TFMessage, prefix + '/tf_static', reliable)
        clock_pub = node.create_publisher(Clock, prefix + '/clock', 10)
        scan_pub = node.create_publisher(LaserScan, prefix + '/scan', 10)
        initial_pub = node.create_publisher(PoseWithCovarianceStamped, prefix + '/initialpose', 10)
        subscription = node.create_subscription(PoseWithCovarianceStamped,
            prefix + '/amcl_pose', estimates.append, 10)
        odom = TransformStamped(); odom.header.stamp = objects['scan'].header.stamp
        odom.header.frame_id = odom_frame; odom.child_frame_id = base_frame
        odom.transform.rotation.w = 1.
        transforms = TFMessage(transforms=[odom, *objects['base_to_sensor_tf'].transforms])
        initial = PoseWithCovarianceStamped(); initial.header = grid.header
        initial.pose.pose.position.x = prior[0]; initial.pose.pose.position.y = prior[1]
        initial.pose.pose.orientation.z = math.sin(prior[2] / 2.)
        initial.pose.pose.orientation.w = math.cos(prior[2] / 2.)
        initial.pose.covariance[0] = initial.pose.covariance[7] = .15**2
        initial.pose.covariance[35] = math.radians(10.)**2
        # Static map/TF are latched ONCE. Re-publishing the same map during
        # lifecycle/scan waits can reset AMCL's particle filter repeatedly.
        map_pub.publish(grid); tf_pub.publish(transforms)
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        change = node.create_client(ChangeState, '/ijcas_sensor_static_amcl/change_state')
        state = node.create_client(GetState, '/ijcas_sensor_static_amcl/get_state')
        def pump():
            if time.monotonic() - started > wall_budget_s:
                raise TimeoutError('Static AMCL observation incomplete; no localization/navigation PASS')
            if process.poll() is not None:
                raise RuntimeError('Isolated AMCL process exited before observation')
            clock_pub.publish(objects['clock'])
            executor.spin_once(timeout_sec=.05)
        while not change.service_is_ready():pump()
        for transition in (1, 3):  # configure, activate; never a robot action
            request = ChangeState.Request(); request.transition.id = transition
            future = change.call_async(request)
            while not future.done():pump()
            if not future.result().success:raise RuntimeError('AMCL lifecycle transition failed')
        request = GetState.Request(); future = state.call_async(request)
        while not future.done():pump()
        if future.result().current_state.id != 3:raise RuntimeError('Actual AMCL ACTIVE state required')
        parameter_client = node.create_client(GetParameters, '/ijcas_sensor_static_amcl/get_parameters')
        while not parameter_client.service_is_ready():pump()
        request = GetParameters.Request(); request.names = list(parameters)
        future = parameter_client.call_async(request)
        while not future.done():pump()
        actual_parameters = {}
        fields = {1: 'bool_value', 2: 'integer_value', 3: 'double_value', 4: 'string_value'}
        for name, value in zip(request.names, future.result().values):
            if value.type not in fields:raise ValueError('Unexpected actual AMCL parameter type: ' + name)
            actual_parameters[name] = getattr(value, fields[value.type])
        if actual_parameters != parameters:
            raise ValueError('Actual AMCL parameter readback differs from explicit diagnostic configuration')
        initial_sent = False; initial_sent_wall = None; scan_publications = 0
        while not estimates:
            pump()
            if initial_pub.get_subscription_count() and not initial_sent:
                initial_pub.publish(initial); initial_sent = True; initial_sent_wall = time.monotonic()
            # This is a bounded callback-processing interval, not new sensor
            # time. The saved process log also records prior/laser ordering.
            if (initial_sent and time.monotonic() - initial_sent_wall >= .25
                    and scan_pub.get_subscription_count() and scan_publications == 0):
                scan_pub.publish(objects['scan']); scan_publications += 1
        value = estimates[-1]; fields = static_pose_message_fields(value)
        estimated = fields['xyyaw']
        result = dict(version='ISOLATED_ONE_STATIC_AMCL_NATIVE_SCAN_DIAGNOSTIC_v1',
            status='STATIC_AMCL_ESTIMATE_RECEIVED_NOT_DRIVE_OR_DYNAMIC_LOCALIZATION_PASS',
            domain_id=domain_id, topic_prefix=prefix, actual_AMCL_active=True,
            actual_localization_message_received=True, requested_AMCL_parameters=parameters,
            actual_AMCL_parameters=actual_parameters, all_requested_parameters_read_back=True,
            initial_prior_xyyaw=prior, initial_prior_std_m_rad=[.15, .15, math.radians(10.)],
            simulator_reference_xyyaw=truth, estimated_xyyaw=estimated,
            position_error_m=math.hypot(estimated[0] - truth[0], estimated[1] - truth[1]),
            heading_error_deg=math.degrees(messages.frames.wrap(estimated[2] - truth[2])),
            reported_covariance=fields['covariance'],
            original_scan_stamp=packet['scan']['header']['stamp'],
            actual_reference_map_configuration=map_report['configuration'],
            actual_reference_map_sha256=_digest(map_report),
            reference_map_scan_coverage=scan_reference_coverage(scan_report, map_report),
            actual_estimate_stamp=fields['stamp'],
            known_simulator_pose_used_for_initial_prior_and_scoring=True,
            identity_static_odom_fixture_NOT_encoder_odometry_validation=True,
            repeated_delivery_of_one_scan_NOT_fresh_stream_or_repetitions=True,
            static_map_publications=1, static_tf_publications=1, scan_publications=scan_publications,
            probability_model_is_development_configuration_NOT_frozen_main_comparison=True,
            beam_skipping_disabled_no_success_condition_or_prior_changed=True,
            likelihood_width_is_development_parameter_NOT_a_calibrated_noise_bound=True,
            no_nav2_controller_or_robot_command_publishers_started=True,
            elapsed_wall_s=time.monotonic() - started,
            dynamic_localization_success=False, Nav2_success=False, new_motion_trials=0)
    finally:
        if process is not None:
            if process.poll() is None:process.terminate()
            try:log, _ = process.communicate(timeout=3.)
            except subprocess.TimeoutExpired:
                process.kill(); log, _ = process.communicate(timeout=3.)
        if executor is not None:executor.shutdown()
        if node is not None:node.destroy_node()
        if context.ok():context.shutdown()
    result['AMCL_process_output'] = log
    prior_index = log.find('Setting pose ('); laser_index = log.find('createLaserObject')
    result['prior_processed_before_first_laser_object_log'] = (
        prior_index >= 0 and laser_index >= 0 and prior_index < laser_index)
    return evidence.json_value(result)


def native_joint_state_message(packet, sample):
    """Exact measured encoder message; no command, GT odometry or covariance."""
    messages.validate_packet(packet)
    return _native_joint_state_from_validated(packet,sample)


def _native_joint_state_from_validated(packet,sample):
    """Private immediate conversion after complete wire-packet validation."""
    sample = evidence.json_value(sample)
    acquisition = packet['acquisition']
    if (sample['source'] != 'EXISTING_NATIVE_ARTICULATION_ENCODER_GET'
            or sample['commands_modes_or_ground_truth_used_as_encoder_inputs'] is not False
            or sample['actor_root'] != acquisition['actor_root']
            or sample['stamp_ns'] != acquisition['stamp_ns']):
        raise ValueError('Exact native encoder actor/source/scan stamp required')
    names = sample['joint_names']
    if (len(names) != 12 or len(set(names)) != 12
            or any(type(name) is not str or not name for name in names)):
        raise ValueError('All twelve unique native joint names required')
    positions = messages.frames.vector(sample['positions_rad'], 12, 'native joint message positions')
    velocities = messages.frames.vector(sample['velocities_rad_s'], 12, 'native joint message velocities')
    from sensor_msgs.msg import JointState
    result = JointState(); stamp = messages.stamp_parts(sample['stamp_ns'])
    result.header.stamp.sec = stamp['sec']; result.header.stamp.nanosec = stamp['nanosec']
    result.header.frame_id = packet['configuration']['base_frame_id']
    result.name = list(names); result.position = positions.tolist(); result.velocity = velocities.tolist()
    result.effort = []  # Not measured here: do not invent zero effort.
    return result


def verify_scan_ros_delivery(report, *, domain_id, topic_prefix, wall_budget_s=10.,
        encoder_sample=None, authorized=False):
    """Deliver ONE saved native scan through isolated ROS publishers/subscribers.

    This function starts no Isaac, AMCL, Nav2, controller, or command publisher.
    Re-publication is ONLY DDS delivery of the SAME acquisition/stamp, never a
    fresh sensor stream or a repetition of a physical experiment. Explicitly
    namespaced TF/clock topics must not replace the live robot's authorities.
    Use the matching ROS Python interpreter, not Isaac's Python ABI.
    """
    if authorized is not True:
        raise ValueError('Explicit isolated ROS data-only delivery authorization required')
    if type(domain_id) is not int or not 1 <= domain_id <= 100:
        raise ValueError('Explicit nondefault bounded isolated ROS domain required')
    if (type(topic_prefix) is not str or not topic_prefix.startswith('/ijcas_sensor_dev/')
            or topic_prefix.endswith('/') or any(c.isspace() for c in topic_prefix)):
        raise ValueError('Explicit isolated development topic prefix required')
    if type(wall_budget_s) not in (int, float) or not math.isfinite(wall_budget_s) or not 0. < wall_budget_s <= 20.:
        raise ValueError('Finite bounded DDS-only verification duration required')
    if (report.get('version') != VERSION
            or report.get('current_native_body_layers_and_PAUSED_clock_unchanged') is not True):
        raise ValueError('Original complete PAUSED native acquisition report required')
    packet = report['complete_raw_packet']
    messages.validate_packet(packet)
    if packet['acquisition']['source_role'] != 'UNVALIDATED_NATIVE_QUERY_CAPTURE':
        raise ValueError('Explicit native observation required; not a software sensor fixture')
    objects = messages.ros_message_objects(packet)
    if encoder_sample is not None:
        objects['native_joint_state'] = native_joint_state_message(packet, encoder_sample)
    # All input/schema checks precede creating a context or any publisher.
    import time
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.serialization import serialize_message

    context = Context()
    sender = receiver = executor = None
    received, errors, publishers, subscriptions = {}, [], {}, []
    expected = {key: serialize_message(value) for key, value in objects.items() if value is not None}
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    started = time.monotonic()
    try:
        rclpy.init(args=[], context=context, domain_id=domain_id)
        sender = rclpy.create_node('ijcas_native_scan_delivery_sender', context=context)
        receiver = rclpy.create_node('ijcas_native_scan_delivery_receiver', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(sender); executor.add_node(receiver)
        for key, value in objects.items():
            if value is None:
                continue
            topic = topic_prefix + '/' + key
            publishers[key] = sender.create_publisher(type(value), topic, qos)
            def accept(message, name=key):
                if not same_ros_message_fields(objects[name], message):
                    errors.append('Received ROS fields differ from the same original acquisition: ' + name)
                else:
                    received[name] = True
            subscriptions.append(receiver.create_subscription(type(value), topic, accept, qos))
        while set(received) != set(expected) and not errors:
            if time.monotonic() - started > wall_budget_s:
                raise TimeoutError('Isolated ROS delivery did not complete; not sensor/navigation PASS')
            for key, publisher in publishers.items():
                publisher.publish(objects[key])
            executor.spin_once(timeout_sec=.05)
        if errors:
            raise RuntimeError('; '.join(errors))
        return evidence.json_value(dict(version='ISOLATED_NATIVE_SCAN_ROS_DELIVERY_v1',
            status='SAME_NATIVE_SCAN_FIELDS_RECEIVED_NOT_FRESH_STREAM_OR_NAVIGATION',
            domain_id=domain_id, topic_prefix=topic_prefix,
            acquisition_id=packet['acquisition']['capture_id'],
            acquisition_stamp=packet['scan']['header']['stamp'],
            received_message_types=sorted(received),
            serialized_bytes={key: len(value) for key, value in expected.items()},
            all_received_message_fields_match_original=True,
            exact_native_encoders_delivered=encoder_sample is not None,
            no_ground_truth_odometry_or_invented_covariance_published=True,
            CDR_padding_bytes_NOT_used_as_field_identity=True, elapsed_wall_s=time.monotonic() - started,
            data_only_publishers_no_command_or_global_tf_clock_topics=True,
            re_publications_NOT_new_acquisitions=True,
            actual_sensor_stream_verified=False, localization_success=False,
            Nav2_success=False, new_motion_trials=0))
    finally:
        if executor is not None:
            executor.shutdown()
        for node in (sender, receiver):
            if node is not None:
                node.destroy_node()
        if context.ok():
            context.shutdown()
