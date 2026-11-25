#!/usr/bin/env python3
"""Replan the 90-degree corridor with the robust rigid-body edge envelope."""

import json
import math
import time
from pathlib import Path

from patient_initial_stability.general_navigation_scenarios import (
    GENERAL_NAVIGATION_SCENARIOS,
)
from patient_initial_stability.rigid_body_edge import (
    GRAVITY_M_S2,
    MassNormalizedRigidBodyEdgeModel,
    RigidBodyParameterBounds,
)
from patient_initial_stability.steering_lattice import (
    ALL_MODES,
    FULL_DYNAMICS,
    STEP_M,
    plan,
    wrap,
)

# Reuse the same full-envelope collision authority and obstacle-aware heuristic
# as the frozen general-navigation comparison.
from screen_general_navigation_dynamics_ablation import (
    make_collision_checker,
    make_obstacle_distance_heuristic,
)


WORKSPACE = Path(__file__).resolve().parents[3]
RESULT_DIRECTORY = WORKSPACE / "studies/IJCAS/results/rigid_body_edge_gate"
CASTER_DISTURBANCE_PATH = (
    RESULT_DIRECTORY / "caster_model_discrepancy_bounds.json"
)
CONDITION_TAGS = (
    "nominal_265_center",
    "friction_low_0p60_0p45",
    "mass290_frontx0p10",
    "mass240_rearx0p10",
)
SPEED_M_S = 0.03
CLEARANCE_M = 0.010
NODE_LIMIT = 300000


def _load_baselines():
    baselines = []
    for tag in CONDITION_TAGS:
        path = RESULT_DIRECTORY / f"corridor_90_rigid_body_screen_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        baselines.append(json.loads(path.read_text())["simulation_baseline"])
    return baselines


def _model(baselines):
    if not CASTER_DISTURBANCE_PATH.exists():
        raise FileNotFoundError(CASTER_DISTURBANCE_PATH)
    disturbance = json.loads(
        CASTER_DISTURBANCE_PATH.read_text(encoding="utf-8")
    )["selected_conservative_bounds"]
    return MassNormalizedRigidBodyEdgeModel(
        RigidBodyParameterBounds(
            yaw_radius_min_m=min(item["yaw_radius_m"] for item in baselines),
            yaw_radius_max_m=max(item["yaw_radius_m"] for item in baselines),
            com_height_min_m=min(item["com_height_m"] for item in baselines),
            com_height_max_m=max(item["com_height_m"] for item in baselines),
            com_longitudinal_min_m=min(
                item["com_longitudinal_m"] for item in baselines
            ),
            com_longitudinal_max_m=max(
                item["com_longitudinal_m"] for item in baselines
            ),
            friction_min=min(item["dynamic_friction"] for item in baselines),
            track_width_m=0.710,
            support_radius_m=math.hypot(0.705, 0.355),
            front_support_x_m=0.665,
            rear_support_x_m=-0.665,
            longitudinal_response_time_s=0.50,
            lateral_response_time_s=0.50,
            yaw_response_time_s=0.50,
            max_longitudinal_accel_m_s2=0.02,
            max_lateral_accel_m_s2=(
                min(item["dynamic_friction"] for item in baselines) * GRAVITY_M_S2
            ),
            max_yaw_rate_rad_s=1.0,
            max_yaw_accel_rad_s2=0.50,
            max_friction_utilization=0.80,
            max_abs_ltr=0.80,
            # Existing CICS simulation sensitivity conditions; these are not
            # claimed as measured physical CoM bounds.
            com_lateral_min_m=-0.050,
            com_lateral_max_m=0.050,
            caster_specific_force_x_abs_max_m_s2=(
                disturbance["specific_force_x_abs_max_m_s2"]
            ),
            caster_specific_force_y_abs_max_m_s2=(
                disturbance["specific_force_y_abs_max_m_s2"]
            ),
            caster_specific_yaw_moment_abs_max_m2_s2=(
                disturbance["specific_yaw_moment_abs_max_m2_s2"]
            ),
            friction_cost_weight_s=0.10,
            ltr_cost_weight_s=0.10,
            actuator_cost_weight_s=0.05,
        )
    )


def _run(policy, allowed_modes, model):
    scenario = GENERAL_NAVIGATION_SCENARIOS["corridor_90_turn"]
    checker = make_collision_checker(scenario["obstacles"], CLEARANCE_M)
    heuristic = make_obstacle_distance_heuristic(
        scenario, CLEARANCE_M, speed_m_s=SPEED_M_S
    )
    started = time.perf_counter()
    result = plan(
        scenario["goal"],
        -3.0,
        3.0,
        robust_clearance=CLEARANCE_M,
        allowed_modes=allowed_modes,
        speed_m_s=SPEED_M_S,
        node_limit=NODE_LIMIT,
        collision_checker=checker,
        propagate_initial_steering=False,
        directions=(1,),
        minimum_x=scenario["bounds_x"][0],
        maximum_x=scenario["bounds_x"][1],
        dynamics=FULL_DYNAMICS,
        goal_heading_tolerance_rad=math.radians(0.5),
        heuristic_function=heuristic,
        rigid_body_model=model,
    )
    elapsed = time.perf_counter() - started
    if result is None:
        return {
            "policy": policy,
            "plan_success": False,
            "expanded_nodes": NODE_LIMIT,
            "planning_time_s": elapsed,
        }, []
    goal = scenario["goal"]
    final = result.states[-1]
    row = {
        "policy": policy,
        "plan_success": True,
        "expanded_nodes": result.expanded_nodes,
        "planning_time_s": elapsed,
        "action_count": len(result.actions),
        "path_length_m": len(result.actions) * STEP_M,
        "mode_switches": sum(
            first[0] != second[0]
            for first, second in zip(result.actions, result.actions[1:])
        ),
        "position_error_m": math.hypot(final.x - goal[0], final.y - goal[1]),
        "heading_error_deg": math.degrees(wrap(final.yaw - goal[2])),
        "maximum_friction_utilization": max(
            state.friction_utilization for state in result.states
        ),
        "maximum_front_friction_utilization": max(
            state.front_friction_utilization for state in result.states
        ),
        "maximum_rear_friction_utilization": max(
            state.rear_friction_utilization for state in result.states
        ),
        "minimum_normal_load_fraction": min(
            state.minimum_normal_load_fraction for state in result.states
        ),
        "maximum_abs_ltr": max(state.abs_ltr for state in result.states),
        "selected_modes": sorted({action[0] for action in result.actions}),
        "cost": result.cost,
    }
    actions = [
        {"mode": mode, "curvature_1_m": curvature, "direction": direction}
        for mode, curvature, direction in result.actions
    ]
    return row, actions


def main():
    baselines = _load_baselines()
    model = _model(baselines)
    rows = []
    actions = {}
    for policy, allowed_modes in (
        ("fixed_counterphase", ("coordinated_counterphase",)),
        ("proposed_full_dynamics", ALL_MODES),
    ):
        row, policy_actions = _run(policy, allowed_modes, model)
        rows.append(row)
        actions[policy] = policy_actions
        print(
            policy,
            "PASS" if row["plan_success"] else "NO_PATH_WITHIN_BUDGET",
            f"nodes={row['expanded_nodes']}",
            f"time={row['planning_time_s']:.2f}s",
            f"path={row.get('path_length_m')}",
            f"gamma={row.get('maximum_friction_utilization')}",
            flush=True,
        )

    bounds = model.bounds
    report = {
        "schema_version": 1,
        "scenario": "corridor_90_turn",
        "speed_m_s": SPEED_M_S,
        "node_limit": NODE_LIMIT,
        "condition_tags": CONDITION_TAGS,
        "robust_bounds": {
            "yaw_radius_m": [bounds.yaw_radius_min_m, bounds.yaw_radius_max_m],
            "com_height_m": [bounds.com_height_min_m, bounds.com_height_max_m],
            "com_longitudinal_m": [
                bounds.com_longitudinal_min_m,
                bounds.com_longitudinal_max_m,
            ],
            "com_lateral_m": [
                bounds.com_lateral_min_m,
                bounds.com_lateral_max_m,
            ],
            "friction_min": bounds.friction_min,
            "caster_model_discrepancy": {
                "specific_force_x_abs_max_m_s2": (
                    bounds.caster_specific_force_x_abs_max_m_s2
                ),
                "specific_force_y_abs_max_m_s2": (
                    bounds.caster_specific_force_y_abs_max_m_s2
                ),
                "specific_yaw_moment_abs_max_m2_s2": (
                    bounds.caster_specific_yaw_moment_abs_max_m2_s2
                ),
            },
        },
        "results": rows,
        "actions": actions,
    }
    output = RESULT_DIRECTORY / "corridor_90_rigid_body_robust_replanning.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
