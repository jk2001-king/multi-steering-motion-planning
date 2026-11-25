"""Versioned OFFLINE steering-response link; never changes the physical drive.

Replace only the legacy constant-bias/rate-limited simulated joint response
with the previously frozen P1 response. Keep command slew, H1 feedback, body
parameters, original task/map and acceptance unchanged. Small-angle validation
does NOT validate this extrapolation, the body model or a physical H3 route.
"""

import json
import math

import numpy as np

from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources, model_bounds
from hospital_h3b_response_validation import MODEL, PRIMARY, load_frozen_fit
from hospital_h3b_first_leg_feedback_review import (
    FirstLegFeedback, INITIAL_CASES, TRACKING_GAINS)
from hospital_path_execution import direct_joint_targets, wrap


VERSION = 'H3B_frozen_P1_response_link_offline_v1'
VALIDATION = GATE / 'hospital_h3b_steering_response_validation/analyses/analysis_r01/response_validation.json'
# Diagnostic cases retained, NOT an identified/certified uncertainty set.
RESIDUAL_CASES = {
    'frozen_nominal': (0., 0.),
    'additional_minus1_deg_synthetic': (-1., -1.),
    'common_plus1p5_deg_synthetic': (1.5, 1.5),
    'front_plus_rear_minus_1p5_deg_synthetic': (1.5, -1.5),
    'front_minus_rear_plus_1p5_deg_synthetic': (-1.5, 1.5)}


def load_validated_response():
    """Require the actual frozen prospective acceptance, not a motion-only PASS."""
    frozen = load_frozen_fit()
    report = json.loads(VALIDATION.read_bytes())
    verify_sources(report)
    SUPPORT['verify_files'](report['input_sha256'])
    SUPPORT['verify_files']({report['prediction_trace']: report['prediction_trace_sha256']})
    if (report.get('verdict') != 'PASS' or not report.get('checks')
            or not all(report['checks'].values())
            or report.get('status') != 'FROZEN_RESPONSE_VALIDATED_FOR_THIS_MIRRORED_SMALL_ANGLE_COMPONENT_ONLY'
            or report.get('validation_data_used_to_refit') is not False
            or report.get('model_applied_to_planner_or_drive_commands') is not False
            or report['frozen_response_sha256'] != SUPPORT['sha'](MODEL)):
        raise ValueError('Actual frozen response validation PASS required; no refitting or command changes')
    return frozen, report


class FrozenSteeringResponse:
    """Exact ZOH P1; initial observed/model state supplied ONCE by the caller.

    The validated model has no actual-angle 8 deg/s clamp. The *command* still
    has its unchanged 8 deg/s slew. Never silently add a different actual-rate
    clamp and call it the validated model.
    """
    def __init__(self, frozen, initial, residual_deg=(0., 0.)):
        if frozen['primary_family'] != PRIMARY:
            raise ValueError('Only frozen primary response supported')
        parameters = [frozen['fit']['axles'][a]['models'][PRIMARY]['parameters'] for a in ('front', 'rear')]
        self.tau = np.array([p['tau_s'] for p in parameters], float)
        self.offset = np.array([p['neutral_offset_rad'] for p in parameters], float) + np.radians(residual_deg)
        self.actual = np.asarray(initial, float).copy()
        if (self.actual.shape != (2,) or self.offset.shape != (2,)
                or not all(np.isfinite(v).all() for v in (self.tau, self.offset, self.actual))
                or np.any(self.tau <= 0)):
            raise ValueError('Finite paired steering states/parameters required')

    def step(self, command, dt):
        command = np.asarray(command, float)
        if command.shape != (2,) or not np.isfinite(command).all() or not math.isfinite(dt) or not 0 < dt <= .25:
            raise ValueError('Finite paired commands and existing <=0.25 s gap required')
        target = command + self.offset
        midpoint = target + (self.actual - target) * np.exp(-dt / (2 * self.tau))
        self.actual = target + (self.actual - target) * np.exp(-dt / self.tau)
        return midpoint, self.actual.copy()


def first_leg_probe(info, frozen, dt=.01, residual_case='frozen_nominal', initial_case='model_origin'):
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import MODE_CURVATURE_RESPONSE_SCALE, MODE_LATERAL_RESPONSE_SCALE

    if not math.isfinite(dt) or not 0 < dt <= .02 or residual_case not in RESIDUAL_CASES or initial_case not in INITIAL_CASES:
        raise ValueError('Only declared offline integration/residual/initial cases permitted')
    bounds = model_bounds()
    body = MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
    pose, velocity = [np.array(v, float) for v in INITIAL_CASES[initial_case]]
    neutral = [frozen['fit']['axles'][a]['models'][PRIMARY]['parameters']['neutral_offset_rad'] for a in ('front', 'rear')]
    response = FrozenSteeringResponse(frozen, neutral, RESIDUAL_CASES[residual_case])
    controller = FirstLegFeedback(info, 0.)
    commanded = (0., 0.)
    now = 0.; rows = []; error = None; tail_start = None
    feasible = True
    maximum_friction = maximum_ltr = maximum_actuator = maximum_slew = maximum_actual_rate = 0.
    try:
        for _ in range(math.ceil(90. / dt)):
            if tail_start is None:
                result = controller.step(now, pose, float(np.linalg.norm(velocity[:2])), response.actual)
                if result['ready']: tail_start = now
            speed, curvature = (0., 0.) if tail_start is not None else (result['speed_m_s'], result['curvature_1_m'])
            previous, old_actual = commanded, response.actual.copy()
            commanded, _ = direct_joint_targets(speed, 'coordinated_counterphase', curvature, commanded, dt, response.actual)
            maximum_slew = max(maximum_slew, *(abs(a-b)/dt for a, b in zip(commanded, previous)))
            midpoint, actual = response.step(commanded, dt)
            maximum_actual_rate = max(maximum_actual_rate, float(np.max(abs(actual-old_actual)/dt)))
            tf, tr = np.tan(midpoint)
            k = (tf-tr)/1.29 * MODE_CURVATURE_RESPONSE_SCALE['coordinated_counterphase']
            ratio = (tf+tr)/2 * MODE_LATERAL_RESPONSE_SCALE['coordinated_counterphase']
            state = body.step(*velocity, speed, speed*ratio, speed*k, dt)
            updated = np.array([state.longitudinal_velocity_m_s, state.lateral_velocity_m_s, state.yaw_rate_rad_s])
            u, v, yaw_rate = (velocity+updated)/2
            mid_yaw = pose[2]+yaw_rate*dt/2
            pose += dt*np.array([u*math.cos(mid_yaw)-v*math.sin(mid_yaw), u*math.sin(mid_yaw)+v*math.cos(mid_yaw), yaw_rate])
            velocity = updated; now += dt
            feasible &= bool(state.feasible)
            maximum_friction = max(maximum_friction, state.friction_utilization)
            maximum_ltr = max(maximum_ltr, state.abs_ltr)
            maximum_actuator = max(maximum_actuator, state.actuator_utilization)
            rows.append(dict(time_s=now, phase='model_zero_command_tail_NOT_PhysX_hold' if tail_start is not None else controller.phase,
                leg_index=0, command_mode='coordinated_counterphase', command_speed_m_s=speed,
                requested_curvature_1_m=curvature, x_m=float(pose[0]), y_m=float(pose[1]), yaw_rad=float(pose[2]),
                u_m_s=float(velocity[0]), v_m_s=float(velocity[1]), r_rad_s=float(velocity[2]),
                front_rad=float(actual[0]), rear_rad=float(actual[1]), command_front_rad=commanded[0], command_rear_rad=commanded[1],
                reference_position_error_m=math.hypot(pose[0]-result['reference_local_xyyaw'][0], pose[1]-result['reference_local_xyyaw'][1]),
                path_error_m=result['path_error_m'], path_heading_error_deg=result['path_heading_error_deg']))
            if tail_start is not None and now-tail_start >= 5.-1e-9: break
        else: raise TimeoutError('Unchanged 90 simulated-second first-component model guard')
    except (RuntimeError, ValueError, TimeoutError) as exception:
        error = f'{type(exception).__name__}: {exception}'
    position = float(np.linalg.norm(pose[:2]-np.array(info['first_stop_model_local_xyyaw'][:2])))
    heading = math.degrees(wrap(pose[2]-info['first_stop_model_local_xyyaw'][2]))
    checks = dict(model_sequence_completed=error is None and tail_start is not None,
        model_terminal_under_30mm=position <= .030, model_terminal_under_2deg=abs(heading) <= 2.,
        model_stopped_under_5mm_s=np.linalg.norm(velocity[:2]) <= .005,
        inherited_force_limits=feasible, command_slew_under_8deg_s=maximum_slew <= math.radians(8.)+1e-9)
    return dict(version=VERSION, dt_s=dt, residual_case=residual_case, initial_case=initial_case,
        synthetic_additive_residual_deg=RESIDUAL_CASES[residual_case], model_parameters=bounds, tracking_gains=TRACKING_GAINS,
        model_checks={k: bool(v) for k, v in checks.items()}, terminal_position_error_m=position,
        terminal_heading_error_deg=heading, final_model_local_xyyaw=pose.tolist(), model_duration_s=now, error=error,
        maximum_friction_utilization=maximum_friction, maximum_abs_ltr=maximum_ltr, maximum_actuator_utilization=maximum_actuator,
        maximum_command_slew_deg_s=math.degrees(maximum_slew), maximum_model_actual_rate_deg_s=math.degrees(maximum_actual_rate),
        no_actual_rate_clamp_added_to_validated_P1=True, no_pose_velocity_or_steering_reset=True,
        command_feedback_is_existing_H1_equation=True, synthetic_cases_NOT_identified_uncertainty_or_robustness_proof=True,
        body_caster_stiction_large_angle_or_reverse_response_NOT_validated=True,
        PhysX_executed=False, full_H3B_turning_path_verified=False, use_for_final_statistics=False), rows
