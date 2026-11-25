"""NEW method-blind per-leg timing/execution prototype; no Isaac/ROS adapter.

Only time allocation and the braking-to-neutral sequence change. A frozen
spatial reference, steering modes, actuator mapping and physical limits remain
unchanged. No caster assignment, yaw-feedback transplant, model fitting or
automatic multi-leg execution is provided. All methods must use the SAME
version in a future comparison; old controller outcomes cannot be pooled.
"""

import math

import numpy as np

from hospital_path_execution import MODES, ramp, wrap
from hospital_h3b_reverse_closedloop import arrival_checks, terminal_errors


VERSION = 'hospital_shared_spatial_timing_executor_prototype_v1'
SPEED_CAP_M_S = .050
ACCELERATION_M_S2 = .020
STEERING_RATE_RAD_S = math.radians(8.)
LAUNCH_LOOKAHEAD_M = .0025


def execution_contract():
    return dict(version=VERSION, role='NEW_COMMON_PER_LEG_OFFLINE_EXECUTOR_PROTOTYPE',
        existing_modes=list(MODES), directions=[-1, 1], method_identifier_used=False,
        speed_cap_m_s=SPEED_CAP_M_S, acceleration_m_s2=ACCELERATION_M_S2,
        steering_slew_deg_s=8., curvature_cap_1_m=.7,
        timing_uses_body_longitudinal_distance_NOT_center_arc=True,
        speed_reduced_by_spatial_steering_slope=True,
        speed_profile_has_forward_backward_acceleration_passes=True,
        nominal_braking_distance_formula='0.005 + 0.5*v_measured*max(v_measured,abs(v_command))/0.020',
        frozen_curvature_retained_during_measured_braking=True,
        neutral_requested_only_after_zero_command_AND_measured_speed_under_5mm_s=True,
        stop_and_neutral_required_before_hold_or_direction_change=True,
        terminal_position_limit_m=.030, terminal_heading_limit_deg=2.,
        actual_neutral_limit_deg=2.5, stopped_speed_m_s=.005,
        maximum_terminal_reapproaches=1, only_same_direction_frozen_reference_reapproach=True,
        maximum_control_gap_s=.25, braking_timeout_s=8., maximum_simulation_s=90., maximum_wall_s=300.,
        tracking_limits_m_deg=[.080, 10.], final_hold_s=5., fixed_tail_s=3.,
        strict_clearance_m=.010, reverse_episode_limit_m=.5,
        original_global_goal_map_gains_friction_mass_and_caster_history_unchanged=True,
        spatial_feedback_or_yaw_controller_added=False, CICS_input_used=False,
        old_results_preserved_do_NOT_pool_different_executor_versions=True,
        same_executor_and_limits_required_for_all_future_baselines=True,
        lower_speed_does_NOT_certify_caster_traction_or_stall_free_motion=True,
        nominal_timing_NOT_a_robust_stopping_or_tracking_certificate=True,
        launch_lookahead_m=LAUNCH_LOOKAHEAD_M,
        runtime_adapter_prepared=False, automatic_motion_authorized=False)


def steering_angles(curvature, mode):
    k = np.asarray(curvature, float)
    if mode not in MODES or not np.isfinite(k).all() or np.max(abs(k)) > .7 + 1e-10:
        raise ValueError('Finite bounded curvature and existing steering mode required')
    if mode == 'front_only':
        return np.stack((np.arctan(1.29 * k), np.zeros_like(k)), axis=-1)
    if mode == 'rear_only':
        return np.stack((np.zeros_like(k), -np.arctan(1.29 * k)), axis=-1)
    return np.stack((np.arctan(.645 * k), -np.arctan(.645 * k)), axis=-1)


class SpatialLeg:
    def __init__(self, profile):
        self.mode, self.direction = profile.get('mode'), profile.get('direction')
        if self.mode not in MODES or isinstance(self.direction, bool) or self.direction not in (-1, 1):
            raise ValueError('One existing steering mode and fixed travel direction required')
        self.poses = np.asarray(profile['poses_local_xyyaw'], float).copy()
        self.arc = np.asarray(profile['center_arc_m'], float).copy()
        self.k = np.asarray(profile['curvature_1_m'], float).copy()
        self.goal = np.asarray(profile['next_cusp_local_xyyaw'], float).copy()
        if (self.arc.ndim != 1 or len(self.arc) < 3 or self.poses.shape != (len(self.arc), 3)
                or self.k.shape != self.arc.shape or self.goal.shape != (3,)
                or not all(np.isfinite(v).all() for v in (self.arc, self.poses, self.k, self.goal))
                or self.arc[0] != 0. or np.any(np.diff(self.arc) <= 0.)
                or self.arc[-1] <= .05 or (self.direction < 0 and self.arc[-1] > .5)
                or max(abs(self.k)) > .7 + 1e-10 or abs(self.k[0]) + abs(self.k[-1]) > 1e-10
                or not np.allclose(self.poses[-1], self.goal, rtol=0., atol=1e-12)
                or np.any(np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1) < 1e-10)):
            raise ValueError('Complete finite single-leg reference with fixed original endpoint required')
        center_ds = np.linalg.norm(np.diff(self.poses[:, :2], axis=0), axis=1)
        if not np.allclose(center_ds, np.diff(self.arc), rtol=1e-5, atol=1e-8):
            raise ValueError('Center arc must reproduce the frozen sampled center path')
        # The old reference supplies center arc only. Recover longitudinal
        # distance by midpoint quadrature of |v_center/u|. This explicit
        # sampled approximation is NOT an additional fitted response model.
        arm = 0. if self.mode == 'coordinated_counterphase' else .645
        midpoint = .5 * (self.k[:-1] + self.k[1:])
        self.longitudinal = np.r_[0., np.cumsum(np.diff(self.arc) / np.sqrt(1. + (arm * midpoint) ** 2))]
        self.progress = 0.

    def at(self, s):
        s = float(s)
        if not math.isfinite(s) or not 0. <= s <= self.arc[-1] + 1e-9:
            raise ValueError('Finite reference progress inside the frozen leg required')
        pose = np.array([np.interp(s, self.arc, self.poses[:, i]) for i in range(3)])
        return pose, float(np.interp(s, self.arc, self.k))

    def project(self, pose):
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Finite measured pose required; never substitute nominal start')
        a, delta = self.poses[:-1, :2], np.diff(self.poses[:, :2], axis=0)
        fraction = np.clip(np.sum((pose[:2] - a) * delta, axis=1) / np.sum(delta * delta, axis=1), 0., 1.)
        candidates = self.arc[:-1] + fraction * np.diff(self.arc)
        distance = np.linalg.norm(pose[:2] - (a + fraction[:, None] * delta), axis=1)
        distance[candidates < self.progress - .005] = np.inf
        index = int(np.argmin(distance))
        if not math.isfinite(distance[index]) or candidates[index] - self.progress > .1:
            raise RuntimeError('Existing monotone projection lost or jumped over 0.10m')
        self.progress = max(self.progress, float(candidates[index]))
        ref, k = self.at(self.progress)
        return ref, float(np.linalg.norm(pose[:2] - ref[:2])), k


class LegTiming:
    """Method-independent nominal |u|(s), 8deg/s and 20mm/s² limited."""
    def __init__(self, leg):
        self.leg = leg
        dx = np.diff(leg.longitudinal)
        arm = 1.29 if leg.mode in ('front_only', 'rear_only') else .645
        minimum = np.minimum(abs(leg.k[:-1]), abs(leg.k[1:]))
        minimum[leg.k[:-1] * leg.k[1:] <= 0.] = 0.
        self.slope = arm * abs(np.diff(leg.k)) / dx / (1. + (arm * minimum) ** 2)
        self.caps = np.full(len(dx), SPEED_CAP_M_S)
        moving = self.slope > 1e-15
        self.caps[moving] = np.minimum(self.caps[moving], STEERING_RATE_RAD_S / self.slope[moving])
        v = np.r_[self.caps[0], np.minimum(self.caps[:-1], self.caps[1:]), self.caps[-1]]
        v[0] = v[-1] = 0.
        for i, distance in enumerate(dx):
            v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2. * ACCELERATION_M_S2 * distance))
        for i in range(len(dx) - 1, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2. * ACCELERATION_M_S2 * dx[i]))
        if np.any(v[:-1] + v[1:] <= 0.):
            raise ValueError('Spatial samples insufficient for finite stop-to-stop timing')
        self.speed = v
        self.time = np.r_[0., np.cumsum(2. * dx / (v[:-1] + v[1:]))]
        self.acceleration = np.diff(v * v) / (2. * dx)
        self.rate = self.slope * np.maximum(v[:-1], v[1:])
        self.checks = dict(
            same_50mm_s_cap=bool(max(v) <= SPEED_CAP_M_S + 1e-12),
            same_20mm_s2_acceleration=bool(max(abs(self.acceleration)) <= ACCELERATION_M_S2 + 1e-12),
            same_8deg_s_nominal_steering_slew=bool(max(self.rate) <= STEERING_RATE_RAD_S + 1e-12),
            nominal_zero_speed_and_curvature_at_stops=bool(v[0] == v[-1] == 0. and abs(leg.k[0]) + abs(leg.k[-1]) < 1e-10))
        if not all(self.checks.values()):
            raise ValueError('Nominal timing violates the unchanged limits')

    def speed_at(self, s):
        if not math.isfinite(s) or not 0. <= s <= self.leg.arc[-1] + 1e-9:
            raise ValueError('Timing query must stay inside the frozen reference')
        longitudinal = np.interp(s, self.leg.arc, self.leg.longitudinal)
        return math.sqrt(max(0., float(np.interp(longitudinal, self.leg.longitudinal, self.speed ** 2))))

    def report(self):
        return dict(version=VERSION, checks=self.checks,
            body_longitudinal_distance_is_midpoint_reconstruction_from_sampled_center_arc=True,
            mode=self.leg.mode, direction=self.leg.direction,
            body_longitudinal_s_m=self.leg.longitudinal.tolist(), center_arc_m=self.leg.arc.tolist(),
            sample_unsigned_body_u_m_s=self.speed.tolist(), time_s=self.time.tolist(),
            nominal_duration_s=float(self.time[-1]),
            minimum_moving_sample_speed_m_s=float(min(self.speed[1:-1])),
            maximum_nominal_steering_rate_deg_s=math.degrees(max(self.rate)),
            spatial_reference_changed=False, controller_gain_or_physical_setting_changed=False,
            nominal_timing_only_NOT_caster_or_contact_validation=True)


class TimedLegController:
    """Measured progress + planned speed cap, unchanged fixed-path curvature.

    Direction/captured-hold readiness remains the external measured gate.
    This pure per-leg class never issues a reset, handoff or next-leg command.
    """
    def __init__(self, profile, started):
        if not math.isfinite(started):
            raise ValueError('Finite command clock required')
        self.reference = SpatialLeg(profile)
        self.timing = LegTiming(self.reference)
        self.previous_time = self.started = started
        self.speed = 0.
        self.phase = 'tracking'
        self.braking_started = None
        self.reapproaches = 0
        self.finished = False

    def step(self, timestamp, pose, measured_speed, actual_steering):
        if self.finished:
            raise RuntimeError('Completed controller cannot restart or select another hold')
        if (not math.isfinite(timestamp) or not math.isfinite(measured_speed) or measured_speed < 0.
                or len(actual_steering) != 2 or not all(math.isfinite(v) for v in actual_steering)):
            raise ValueError('Finite actual speed and steering required')
        dt = timestamp - self.previous_time
        if not 0. <= dt <= .25:
            raise RuntimeError('Existing 0.25s control-gap guard')
        if timestamp - self.started > 90.:
            raise TimeoutError('Existing 90s simulation budget')
        self.previous_time = timestamp
        ref, error, k = self.reference.project(pose)
        heading = math.degrees(wrap(pose[2] - ref[2]))
        if error > .080 or abs(heading) > 10.:
            raise RuntimeError('Existing 80mm/10deg curved-reference guard')
        remaining = float(self.reference.arc[-1] - self.reference.progress)
        threshold = .005 + .5 * measured_speed * max(measured_speed, abs(self.speed)) / ACCELERATION_M_S2
        transition = None
        if self.phase == 'tracking' and remaining <= threshold:
            self.phase = 'terminal_braking'
            self.braking_started = timestamp
            transition = 'measured_braking_KEEP_frozen_curvature'
        if self.braking_started is not None and timestamp - self.braking_started > 8.:
            raise TimeoutError('Existing 8s braking/neutral guard')
        # A bounded spatial lookahead avoids a zero-speed-at-s=0 launch
        # deadlock. It changes only timing lookup, NOT the curvature/goal or
        # measured progress. Its runtime effects require the model screen.
        query = min(self.reference.arc[-1], self.reference.progress + LAUNCH_LOOKAHEAD_M)
        scheduled = self.timing.speed_at(query)
        target = self.reference.direction * scheduled if self.phase == 'tracking' else 0.
        self.speed = ramp(self.speed, target, dt, ACCELERATION_M_S2)
        if self.phase == 'terminal_braking' and abs(self.speed) < 1e-7 and measured_speed <= .005:
            self.phase = 'neutral_settle'
            transition = 'measured_stop_then_neutral'
        if self.phase == 'neutral_settle':
            k = 0.
        checks = arrival_checks(pose, self.reference.goal, measured_speed, actual_steering)
        stopped_neutral = bool(self.phase == 'neutral_settle' and abs(self.speed) < 1e-7
            and measured_speed <= .005 and max(abs(v) for v in actual_steering) <= math.radians(2.5))
        ready = bool(stopped_neutral and all(checks.values()))
        if ready:
            self.finished = True
            transition = 'measured_terminal_arrival_before_ONE_hold'
        elif stopped_neutral:
            if remaining <= .005:
                raise RuntimeError('Stopped outside acceptance with no admitted same-direction remainder')
            if self.reapproaches >= 1:
                raise RuntimeError('One bounded same-direction terminal reapproach exhausted')
            self.reapproaches += 1
            self.phase = 'tracking'
            self.braking_started = None
            transition = 'one_bounded_same_direction_terminal_reapproach'
        position, yaw = terminal_errors(pose, self.reference.goal)
        return dict(speed_m_s=self.speed, curvature_1_m=k, ready=ready, phase=self.phase,
            mode=self.reference.mode, direction=self.reference.direction,
            scheduled_body_u_m_s=scheduled, timing_query_center_arc_m=query,
            progress_m=self.reference.progress, remaining_m=remaining,
            path_error_m=error, path_heading_error_deg=heading, reference_local_xyyaw=ref.tolist(),
            measured_braking_speed_m_s=measured_speed, measured_braking_threshold_m=threshold,
            terminal_position_error_m=position, terminal_heading_error_deg=yaw,
            terminal_reapproaches=self.reapproaches, transition=transition)
