"""Explicit predictive-state interventions; no simulator or physical state writes.

The old per-primitive standstill reset is preserved as a historical variant.
The new factorial comparison projects omitted history onto the current input's
steady-state manifold, rather than inventing a fresh departure every 50 mm.
All response constants, rate limits, force/load checks and physical control
remain unchanged. These are model interventions, not measured robot states.
"""
from dataclasses import replace
import math

from patient_initial_stability import steering_lattice as lattice


HISTORY_POLICIES = ('carry', 'steady_steering', 'steady_velocity', 'steady_both')


def projected_entry(state, mode, curvature, direction, *, kernel, profile,
                    target_speed_m_s, history_policy):
    """Remove only the declared predictive history; retain every other field.

    Nominal steering/velocity are derived from the CURRENT proposed input,
    with the same bias and mode coefficients as P. In the single-factor
    variants the other factor is still propagated continuously. ``started``
    is deliberately not changed: all campaign dynamic entries are True.
    """
    if history_policy not in HISTORY_POLICIES:
        raise ValueError('One explicit factorial history policy required')
    if history_policy == 'carry':
        return state
    if state.started is not True:
        raise ValueError('History intervention requires an initialized dynamic state')
    if (type(direction) is not int or direction not in (-1, 1)
            or not math.isfinite(target_speed_m_s) or not 0. < target_speed_m_s <= .05):
        raise ValueError('Original finite signed input and speed bound required')
    front, rear = kernel.steering_targets(mode, curvature)
    front += lattice.STEERING_TRACKING_BIAS_RAD['front']
    rear += lattice.STEERING_TRACKING_BIAS_RAD['rear']
    changes = {}
    if history_policy in ('steady_steering', 'steady_both'):
        changes.update(front=front, rear=rear)
    if history_policy in ('steady_velocity', 'steady_both'):
        geometry = profile['geometry']
        u = direction * target_speed_m_s
        changes.update(longitudinal_velocity=u,
            lateral_velocity=u * profile['mode_lateral_response_scale'][mode] * (
                geometry['rear_length_m'] * math.tan(front)
                + geometry['front_length_m'] * math.tan(rear)) / geometry['wheelbase_m'],
            yaw_rate=u * profile['mode_curvature_response_scale'][mode] * (
                math.tan(front)-math.tan(rear)) / geometry['wheelbase_m'])
    return replace(state, **changes)


def bind_propagation(original, *, kernel, profile, history_policy='carry', activity=None):
    """Works around either existing Python or compiled propagation kernel."""
    if history_policy not in HISTORY_POLICIES:
        raise ValueError('Unknown state-history intervention')
    def propagate(state, mode, curvature, direction, *args, **kwargs):
        full = kwargs.get('rigid_body_model') is not None
        if full and history_policy != 'carry':
            # The actual shared planner always supplies these by keyword.
            # Fail closed rather than silently use an incorrect positional dt.
            if args:
                raise ValueError('Named dynamic primitive parameters required')
            speed = kwargs.get('target_speed_m_s')
            if speed is None:
                speed = kwargs.get('speed_m_s', lattice.SPEED_M_S)
            state = projected_entry(state, mode, curvature, direction,
                kernel=kernel, profile=profile, target_speed_m_s=speed,
                history_policy=history_policy)
        result = original(state, mode, curvature, direction, *args, **kwargs)
        if full and activity is not None:
            final = result[0]
            activity['model_primitive_calls'] += 1
            activity['infeasible_result_count'] += int(not final.rigid_body_feasible)
            activity['maximum_friction_utilization'] = max(
                activity['maximum_friction_utilization'], final.friction_utilization)
            activity['maximum_abs_ltr'] = max(activity['maximum_abs_ltr'], final.abs_ltr)
            activity['friction_limit_exceedance_count'] += int(final.friction_utilization > .8)
            activity['LTR_limit_exceedance_count'] += int(final.abs_ltr > .8)
        return result
    propagate.__dict__.update(original.__dict__)
    propagate.predictive_history_policy = history_policy
    return propagate


def ablation_definition():
    return dict(version='P_CURRENT_INPUT_STEADY_ENTRY_FACTORIAL_v1',
        variants=list(HISTORY_POLICIES),
        steady_steering='only delta_f/delta_r replaced by current biased target angles',
        steady_velocity='only u/v/r replaced by current-input steady-state targets',
        steady_both='both of the above projections; not a repeated standstill reset',
        carried_pose_mode_direction_resource_history_and_started_unchanged=True,
        coefficients_time_constants_rate_limits_constraints_speed_and_controller_unchanged=True,
        physical_robot_states_are_NOT_reset=True,
        initial_predictive_transient_also_omitted_by_memoryless_model=True,
        legacy_zero_reset_variant_is_NOT_renamed_or_pooled=True,
        factorial_main_effects_require_matched_outcomes_and_interaction_review=True,
        fixed_action_rollout_is_prediction_sensitivity_NOT_replanned_execution_ablation=True)
