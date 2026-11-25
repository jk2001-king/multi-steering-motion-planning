"""Read-only post-cleanup reverse-stall snapshot checks; NOT a time history."""

import math

import numpy as np


WHEELS = ('front_active_drive_wheel_link', 'rear_active_drive_wheel_link',
          'front_left_caster_wheel_link', 'front_right_caster_wheel_link',
          'rear_left_caster_wheel_link', 'rear_right_caster_wheel_link')


def validate_state(record, transform, names, positions):
    transform = np.asarray(transform, float).reshape(-1)
    positions = np.asarray(positions, float).reshape(-1)
    if (len(names) != 12 or len(set(names)) != 12 or positions.size != 12 or transform.size != 7
            or not np.isfinite(transform).all() or not np.isfinite(positions).all()):
        raise ValueError('Finite retained body and twelve-joint readback required')
    last = record['last_recorded_state']
    if any('joint_'+n+'_rad' not in last for n in names):
        raise ValueError('Runtime twelve-joint set differs from recorded rear stall')
    x, y, z, w = transform[3:]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    position = math.hypot(transform[0]-last['x_m'], transform[1]-last['y_m'])
    heading = math.degrees(math.atan2(math.sin(yaw-last['yaw_rad']), math.cos(yaw-last['yaw_rad'])))
    joints = max(abs(q-last['joint_'+n+'_rad']) for n, q in zip(names, positions))
    if position > .001 or abs(heading) > .1 or joints > .001:
        raise ValueError('Live state differs from recorded rear stall; no reset/recovery permitted')
    return dict(position_difference_m=position, heading_difference_deg=heading,
                maximum_joint_difference_rad=joints, live_yaw_rad=yaw)


def summarize_cached_friction(row, sensor_paths, yaw):
    if (len(sensor_paths) != 6 or len(set(sensor_paths)) != 6
            or {p.split('/')[-1] for p in sensor_paths} != set(WHEELS)):
        raise ValueError('Six unique expected native wheel sensors required')
    projected = {}; counts = []
    direction = (-math.cos(yaw), -math.sin(yaw))
    for i, path in enumerate(sensor_paths):
        n = row[f'wheel_{i}__friction_anchor_count']; counts.append(n)
        fx = row[f'wheel_{i}__sdk_tangent_sum_fx_n']; fy = row[f'wheel_{i}__sdk_tangent_sum_fy_n']
        fz = row[f'wheel_{i}__sdk_reported_contact_fz_n']
        if not all(math.isfinite(v) for v in (n, fx, fy, fz)) or n < 0 or n != int(n):
            raise ValueError('Invalid cached native friction values')
        projected[path] = dict(anchor_count=int(n), sdk_vertical_contact_n=fz,
            sdk_vector_projection_along_current_reverse_n=fx*direction[0]+fy*direction[1])
    available = all(n > 0 for n in counts)
    return dict(status='CACHED_SDK_VALUES_RECORDED_NOT_HISTORICAL_STALL_FORCES' if available else
                        'CACHED_ANCHOR_COVERAGE_UNAVAILABLE_DO_NOT_PRESS_PLAY',
                all_six_wheels_have_cached_anchors=available, per_wheel=projected,
                SDK_sign_convention_retained_NOT_an_isolated_traction_or_motor_torque_measurement=True,
                SDK_contact_buffer_has_NO_verified_historical_timestamp=True,
                unique_cause_isolated=False, motor_saturation_proven=False,
                complete_stall_window_force_history_recorded=False)
