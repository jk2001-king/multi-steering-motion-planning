"""Separate measured-braking/terminal-arrival executor, NOT a planner change.

The fixed rear-only reference, 50 mm/s cap, actuator gains and safety bounds
are retained. At most ONE additional reverse approach is allowed if the first
stop falls short. There is no forward recovery, pose recapture, repeated hold
selection or change to the original 30 mm / 2 degree acceptance.
"""

import math

import numpy as np

from hospital_h3b_measured_reverse import FirstReverseReference, ramp, wrap


VERSION = 'H3B_measured_reverse_braking_terminal_arrival_v1'
MAX_TERMINAL_REAPPROACHES = 1


def execution_contract():
    return dict(version=VERSION, reverse_speed_limit_m_s=.050, acceleration_m_s2=.020,
        braking_distance_uses_measured_body_speed=True, braking_margin_m=.005,
        braking_time_includes_existing_command_ramp_to_zero=True,
        nominal_braking_distance_formula='0.005 + 0.5*v_measured*max(v_measured,abs(v_command))/0.020',
        braking_distance_is_nominal_NOT_a_proven_contact_stopping_bound=True,
        terminal_position_limit_m=.030, terminal_heading_limit_deg=2.,
        stopped_speed_m_s=.005, actual_neutral_limit_deg=2.5,
        maximum_terminal_reapproaches=MAX_TERMINAL_REAPPROACHES,
        reapproach_keeps_same_reverse_direction_and_frozen_spatial_reference=True,
        reapproach_zero_command_sample_retains_neutral_steering=True,
        maximum_control_gap_s=.25, maximum_braking_time_s=8.,
        maximum_simulation_duration_s=90., maximum_wall_time_s=300.,
        maximum_reverse_episode_m=.5, steering_slew_deg_s=8., strict_clearance_m=.010,
        curved_reference_limits_m_deg=[.080, 10.], final_hold_duration_s=5., fixed_tail_duration_s=3.,
        progress_check_includes_tracking_AND_braking=True,
        actuator_gain_friction_geometry_caster_or_goal_change=False,
        original_10_and_50mm_s_outcomes_preserved=True,
        NEW_executor_NOT_unchanged_speed_only_ablation=True)


def terminal_errors(pose, goal):
    pose, goal = np.asarray(pose, float), np.asarray(goal, float)
    if pose.shape != (3,) or goal.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(goal).all():
        raise ValueError('Finite measured pose and UNCHANGED fixed cusp required')
    return float(np.linalg.norm(pose[:2]-goal[:2])), math.degrees(wrap(pose[2]-goal[2]))


def arrival_checks(pose, goal, measured_speed, actual_steering):
    if (len(actual_steering) != 2 or not math.isfinite(measured_speed) or measured_speed < 0.
            or not all(math.isfinite(v) for v in actual_steering)):
        raise ValueError('Finite actual speed and steering required for arrival')
    position, heading = terminal_errors(pose, goal)
    return dict(terminal_position_under_30mm=position <= .030,
        terminal_heading_under_2deg=abs(heading) <= 2., measured_speed_under_5mm_s=measured_speed <= .005,
        actual_steering_neutral_under_2p5deg=max(abs(v) for v in actual_steering) <= math.radians(2.5))


class FirstReverseProgress:
    def __init__(self, info, started):
        if not math.isfinite(started): raise ValueError('Finite command clock required')
        self.reference = FirstReverseReference(info)
        self.goal = np.array(info['next_cusp_local_xyyaw'], float)
        if not np.allclose(self.reference.poses[-1], self.goal, rtol=0., atol=1e-12):
            raise ValueError('Final cusp must equal the unchanged frozen reference endpoint')
        self.previous_time = self.started = started
        self.speed = 0.; self.phase = 'tracking'; self.braking_started = None
        self.reapproaches = 0; self.finished = False

    def step(self, timestamp, pose, measured_speed, actual_steering):
        if self.finished: raise RuntimeError('Terminal-ready controller cannot be replayed/restarted')
        if (not math.isfinite(timestamp) or not math.isfinite(measured_speed) or measured_speed < 0.
                or len(actual_steering) != 2 or not all(math.isfinite(v) for v in actual_steering)):
            raise ValueError('Finite actual speed/steering required')
        dt = timestamp-self.previous_time
        if not 0. <= dt <= .25: raise RuntimeError('Existing 0.25s control-gap guard')
        if timestamp-self.started > 90.: raise TimeoutError('Existing 90s simulation budget')
        self.previous_time = timestamp
        ref, error, k = self.reference.project(pose)
        heading = math.degrees(wrap(pose[2]-ref[2]))
        remaining = self.reference.arc[-1]-self.reference.progress
        if error > .080 or abs(heading) > 10.: raise RuntimeError('Existing curved-reference 80mm/10deg guard')
        # Retain the existing command ramp's time to zero. Using measured v^2
        # alone would assume a BODY deceleration of 0.020 m/s^2 even when the
        # measured body undershoots the command. This remains a nominal model,
        # not a guaranteed contact-stopping bound; all live guards still apply.
        ramp_time = max(measured_speed, abs(self.speed))/.020
        threshold = .005 + .5*measured_speed*ramp_time
        transition = None
        if self.phase == 'tracking' and remaining <= threshold:
            self.phase = 'braking'; self.braking_started = timestamp; transition = 'measured_braking_entry'
        if self.phase == 'braking' and timestamp-self.braking_started > 8.:
            raise TimeoutError('Existing 8s measured braking/neutral guard')
        self.speed = ramp(self.speed, -.050 if self.phase == 'tracking' else 0., dt, .020)
        stopped = (self.phase == 'braking' and abs(self.speed) < 1e-7 and measured_speed <= .005
                   and max(abs(v) for v in actual_steering) <= math.radians(2.5))
        checks = arrival_checks(pose, self.goal, measured_speed, actual_steering)
        ready = stopped and all(checks.values())
        if ready:
            self.finished = True; transition = 'measured_terminal_arrival_before_ONE_hold'
        elif stopped:
            # Continue only along the SAME reverse reference, never from a new
            # recaptured goal. A reverse-only overshoot/heading failure is not
            # silently recovered by adding forward/yaw commands.
            if remaining <= .005:
                raise RuntimeError('Stopped outside terminal acceptance with no admitted reverse remainder')
            if self.reapproaches >= MAX_TERMINAL_REAPPROACHES:
                raise RuntimeError('One bounded terminal reapproach exhausted')
            self.reapproaches += 1; self.phase = 'tracking'; self.braking_started = None
            transition = 'one_bounded_same_direction_terminal_reapproach'
        if self.phase == 'braking' or transition == 'one_bounded_same_direction_terminal_reapproach': k = 0.
        position, terminal_heading = terminal_errors(pose, self.goal)
        return dict(speed_m_s=self.speed, curvature_1_m=k, ready=ready, phase=self.phase,
            progress_m=self.reference.progress, remaining_m=remaining, path_error_m=error,
            path_heading_error_deg=heading, reference_local_xyyaw=ref.tolist(),
            measured_braking_speed_m_s=measured_speed, measured_braking_threshold_m=threshold,
            terminal_position_error_m=position, terminal_heading_error_deg=terminal_heading,
            terminal_reapproaches=self.reapproaches, transition=transition)


def motion_stalled(rows):
    """Same 5s / <1mm / >=6mm/s guard, including commanded braking motion.

    Use the CURRENT contiguous physical interval, never bridge a zero-command
    stop or a former approach to manufacture a five-second stall window.
    """
    if not rows: return False
    end = rows[-1]['time_s']; tail = [r for r in rows if r['time_s'] >= end-5.]
    return (len(tail) > 1 and tail[-1]['time_s']-tail[0]['time_s'] >= 4.95
        and all(r['phase'] in ('reverse_motion', 'reverse_braking') and not r['position_hold_enabled']
                and r['command_speed_m_s'] <= -.006 for r in tail)
        and max(r['reference_progress_m'] for r in tail)-min(r['reference_progress_m'] for r in tail) < .001)


def reverse_episode_progress(rows):
    moving = [r for r in rows if r['phase'] in ('reverse_motion', 'reverse_braking')
              and not r['position_hold_enabled'] and r['command_speed_m_s'] < -.003]
    return moving[-1]['reference_progress_m']-moving[0]['reference_progress_m'] if moving else 0.
