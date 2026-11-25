import math

import pytest

from patient_initial_stability.rigid_body_edge import (
    MassNormalizedRigidBodyEdgeModel,
    RigidBodyParameterBounds,
)
from patient_initial_stability.steering_lattice import FULL_DYNAMICS, State, plan, propagate


def bounds(**overrides):
    values = {
        "yaw_radius_min_m": 0.45,
        "yaw_radius_max_m": 0.65,
        "com_height_min_m": 0.35,
        "com_height_max_m": 0.85,
        "com_longitudinal_min_m": -0.10,
        "com_longitudinal_max_m": 0.10,
        "friction_min": 0.50,
        "track_width_m": 0.90,
        "support_radius_m": 0.70,
        "front_support_x_m": 0.665,
        "rear_support_x_m": -0.665,
        "longitudinal_response_time_s": 0.50,
        "lateral_response_time_s": 0.50,
        "yaw_response_time_s": 0.50,
        "max_longitudinal_accel_m_s2": 1.0,
        "max_lateral_accel_m_s2": 1.0,
        "max_yaw_rate_rad_s": 1.0,
        "max_yaw_accel_rad_s2": 1.0,
        "max_friction_utilization": 0.90,
        "max_abs_ltr": 0.80,
        "friction_cost_weight_s": 0.1,
        "ltr_cost_weight_s": 0.1,
    }
    values.update(overrides)
    return RigidBodyParameterBounds(**values)


def test_parameter_bounds_reject_invalid_uncertainty_interval():
    with pytest.raises(ValueError):
        bounds(yaw_radius_min_m=0.8, yaw_radius_max_m=0.6)


def test_mass_normalized_step_reports_feasibility_metrics():
    result = MassNormalizedRigidBodyEdgeModel(bounds()).step(
        0.0, 0.0, 0.0,
        0.03, 0.0, 0.02,
        0.1,
    )
    assert result.feasible
    assert result.longitudinal_velocity_m_s > 0.0
    assert result.friction_utilization > 0.0
    assert result.front_friction_utilization > 0.0
    assert result.rear_friction_utilization > 0.0
    assert result.minimum_normal_load_fraction > 0.0
    assert result.abs_ltr >= 0.0
    assert 0.0 <= result.actuator_utilization <= 1.0


def test_lateral_com_interval_increases_robust_ltr():
    centered = MassNormalizedRigidBodyEdgeModel(bounds()).step(
        0.03, 0.0, 0.0, 0.03, 0.01, 0.02, 0.1
    )
    offset = MassNormalizedRigidBodyEdgeModel(
        bounds(com_lateral_min_m=-0.05, com_lateral_max_m=0.05)
    ).step(0.03, 0.0, 0.0, 0.03, 0.01, 0.02, 0.1)
    assert offset.abs_ltr > centered.abs_ltr
    assert offset.minimum_normal_load_fraction < centered.minimum_normal_load_fraction


def test_caster_disturbance_interval_increases_friction_demand():
    nominal = MassNormalizedRigidBodyEdgeModel(bounds()).step(
        0.03, 0.0, 0.0, 0.03, 0.01, 0.02, 0.1
    )
    robust = MassNormalizedRigidBodyEdgeModel(
        bounds(
            caster_specific_force_x_abs_max_m_s2=0.028,
            caster_specific_force_y_abs_max_m_s2=0.014,
            caster_specific_yaw_moment_abs_max_m2_s2=0.006,
        )
    ).step(0.03, 0.0, 0.0, 0.03, 0.01, 0.02, 0.1)
    assert robust.friction_utilization > nominal.friction_utilization


def test_low_friction_rejects_aggressive_edge():
    configured_bounds = bounds(
            friction_min=0.02,
            longitudinal_response_time_s=0.05,
            lateral_response_time_s=0.05,
            yaw_response_time_s=0.05,
        )
    model = MassNormalizedRigidBodyEdgeModel(configured_bounds)
    result = model.step(0.0, 0.0, 0.0, 0.5, 0.2, 0.8, 0.1)
    assert not result.feasible
    assert result.friction_utilization > configured_bounds.max_friction_utilization


def test_optional_rigid_body_rollout_changes_edge_and_preserves_report():
    state = State(0.0, 0.0, 0.0, "coordinated_counterphase", 0.0, 0.0, True, 1)
    legacy, _, _ = propagate(
        state, "front_only", 0.70, 1, dynamics=FULL_DYNAMICS
    )
    dynamic, _, _ = propagate(
        state,
        "front_only",
        0.70,
        1,
        dynamics=FULL_DYNAMICS,
        rigid_body_model=MassNormalizedRigidBodyEdgeModel(bounds()),
    )
    assert legacy.x != pytest.approx(dynamic.x)
    assert dynamic.longitudinal_velocity > 0.0
    assert dynamic.yaw_rate > 0.0
    assert dynamic.rigid_body_feasible
    assert math.isfinite(dynamic.rigid_body_penalty_s)


def test_planner_accepts_feasible_rigid_body_edges():
    result = plan(
        (0.15, 0.0, 0.0),
        -1.0,
        1.0,
        node_limit=10000,
        rigid_body_model=MassNormalizedRigidBodyEdgeModel(bounds()),
    )
    assert result is not None
    assert result.actions
    assert all(state.rigid_body_feasible for state in result.states)


def test_speed_target_is_a_planning_state_and_can_decelerate():
    model = MassNormalizedRigidBodyEdgeModel(bounds())
    moving = State(
        0.0, 0.0, 0.0,
        "coordinated_counterphase", 0.0, 0.0, True, 1,
        longitudinal_velocity=0.08,
    )
    slowed, _, transition_time = propagate(
        moving,
        "coordinated_counterphase",
        0.0,
        1,
        speed_m_s=0.08,
        target_speed_m_s=0.01,
        dynamics=FULL_DYNAMICS,
        rigid_body_model=model,
    )
    assert slowed.longitudinal_velocity < moving.longitudinal_velocity
    assert slowed.x == pytest.approx(0.05, abs=2.0e-3)
    assert transition_time > 0.05 / 0.08


def test_planner_enforces_low_terminal_speed_when_requested():
    result = plan(
        (0.10, 0.0, 0.0),
        -1.0,
        1.0,
        speed_m_s=0.03,
        speed_levels_m_s=(0.01, 0.03),
        terminal_speed_tolerance_m_s=0.012,
        node_limit=30000,
        allowed_modes=("coordinated_counterphase",),
        rigid_body_model=MassNormalizedRigidBodyEdgeModel(bounds()),
    )
    assert result is not None
    assert len(result.actions[-1]) == 4
    assert result.actions[-1][3] == pytest.approx(0.01)
    assert abs(result.states[-1].longitudinal_velocity) <= 0.012


def test_stopping_reachability_prunes_edges_before_goal():
    diagnostics = {}
    result = plan(
        (0.30, 0.0, 0.0),
        -1.0,
        1.0,
        speed_m_s=0.08,
        speed_levels_m_s=(0.08,),
        terminal_speed_tolerance_m_s=0.012,
        terminal_deceleration_m_s2=0.02,
        node_limit=10000,
        allowed_modes=("coordinated_counterphase",),
        directions=(1,),
        minimum_x=-0.1,
        maximum_x=0.5,
        rigid_body_model=MassNormalizedRigidBodyEdgeModel(
            bounds(max_longitudinal_accel_m_s2=0.02)
        ),
        diagnostics=diagnostics,
    )
    assert result is None
    assert diagnostics["stopping_reachability"] > 0


def test_terminal_deceleration_requires_terminal_speed_constraint():
    with pytest.raises(ValueError):
        plan(
            (0.10, 0.0, 0.0),
            -1.0,
            1.0,
            terminal_deceleration_m_s2=0.02,
        )
