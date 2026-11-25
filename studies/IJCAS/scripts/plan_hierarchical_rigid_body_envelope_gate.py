#!/usr/bin/env python3
"""Integrated route/speed selection using swept volume and rigid-body edges."""

import argparse
import json
import math
import time
from pathlib import Path

from patient_initial_stability.general_navigation_scenarios import (
    GENERAL_NAVIGATION_SCENARIOS,
)
from patient_initial_stability.hierarchical_dynamics_planner import (
    refine_speed_consistent_lattice_route,
)
from patient_initial_stability.steering_lattice import STEP_M, wrap
from plan_corridor_90_rigid_body_robust import _load_baselines, _model
from screen_general_navigation_dynamics_ablation import make_collision_checker


WORKSPACE = Path(__file__).resolve().parents[3]
SOURCE = (
    WORKSPACE / "studies/IJCAS/results/rigid_body_edge_gate/"
    "corridor_90_rigid_body_robust_replanning.json"
)
OUTPUT = (
    WORKSPACE / "studies/IJCAS/results/rigid_body_edge_gate/"
    "corridor_90_hierarchical_rigid_body_envelope.json"
)
SPEEDS = (0.01, 0.03, 0.08)
TERMINAL_SPEED_M_S = 0.012
MAX_DECELERATION_M_S2 = 0.02
CLEARANCE_M = 0.010


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="corridor_90_turn",
        choices=tuple(sorted(GENERAL_NAVIGATION_SCENARIOS)),
    )
    parser.add_argument(
        "--source",
        default="",
        help="Spatial-candidate JSON. Required for non-legacy scenarios.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output planner JSON. Defaults to a scenario-named result file.",
    )
    args = parser.parse_args()
    scenario_name = args.scenario
    if args.source:
        source_path = Path(args.source).expanduser().resolve()
    elif scenario_name == "corridor_90_turn":
        source_path = SOURCE
    else:
        parser.error("--source is required outside the legacy corridor scenario")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else OUTPUT.parent / f"{scenario_name}_hierarchical_rigid_body_envelope.json"
    )
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_scenario = source.get("scenario")
    if source_scenario not in (None, scenario_name):
        raise ValueError(
            f"source scenario mismatch: expected {scenario_name}, got {source_scenario}"
        )
    scenario = GENERAL_NAVIGATION_SCENARIOS[scenario_name]
    model = _model(_load_baselines())
    checker = make_collision_checker(scenario["obstacles"], CLEARANCE_M)
    route_candidates = {
        key: value for key, value in source["actions"].items() if value
    }
    candidate_reports = []
    feasible = []
    started = time.perf_counter()
    for route_id, actions in route_candidates.items():
        candidate_started = time.perf_counter()
        rejection_samples = []
        result, rejected = refine_speed_consistent_lattice_route(
            route_id=route_id,
            actions=actions,
            goal=scenario["goal"],
            speed_levels_m_s=SPEEDS,
            rigid_body_model=model,
            collision_checker=checker,
            terminal_speed_tolerance_m_s=TERMINAL_SPEED_M_S,
            terminal_deceleration_m_s2=MAX_DECELERATION_M_S2,
            step_m=STEP_M,
            terminal_extension_edges=1,
            rejection_samples=rejection_samples,
        )
        row = {
            "route_id": route_id,
            "spatial_action_count": len(actions),
            "optimized_action_count": (
                len(result.actions) if result is not None else None
            ),
            "feasible": result is not None,
            "planning_time_s": time.perf_counter() - candidate_started,
            "rejection_counts": rejected,
            "rejection_samples": rejection_samples,
        }
        if result is not None:
            final = result.states[-1]
            row.update(
                {
                    "cost_s": result.cost_s,
                    "task_time_s": sum(result.edge_times_s),
                    "position_error_m": math.hypot(
                        final.x - scenario["goal"][0],
                        final.y - scenario["goal"][1],
                    ),
                    "heading_error_deg": math.degrees(
                        wrap(final.yaw - scenario["goal"][2])
                    ),
                    "terminal_speed_m_s": math.hypot(
                        final.longitudinal_velocity,
                        final.lateral_velocity,
                    ),
                    "maximum_friction_utilization": max(
                        state.friction_utilization for state in result.states
                    ),
                    "minimum_corner_normal_load_fraction": min(
                        state.minimum_normal_load_fraction for state in result.states
                    ),
                    "maximum_abs_ltr": max(
                        state.abs_ltr for state in result.states
                    ),
                    "maximum_actuator_utilization": max(
                        state.actuator_utilization for state in result.states
                    ),
                    "speed_profile_m_s": result.speeds_m_s,
                }
            )
            feasible.append((result.cost_s, result, row))
        candidate_reports.append(row)
    selected = min(feasible, key=lambda item: item[0]) if feasible else None
    report = {
        "schema_version": 1,
        "scenario": scenario_name,
        "method": (
            "speed-consistent full-envelope lattice refinement with robust "
            "Newton-Euler edge constraints and stopping-reachability states"
        ),
        "source_spatial_candidates": str(source_path),
        "speed_levels_m_s": SPEEDS,
        "terminal_speed_tolerance_m_s": TERMINAL_SPEED_M_S,
        "max_deceleration_m_s2": MAX_DECELERATION_M_S2,
        "clearance_m": CLEARANCE_M,
        "planning_time_s": time.perf_counter() - started,
        "candidates": candidate_reports,
        "plan_success": selected is not None,
        "selected_route_id": selected[1].route_id if selected else None,
    }
    if selected is not None:
        result = selected[1]
        report["selected_actions"] = [
            {
                **action,
                "target_speed_m_s": speed,
            }
            for action, speed in zip(result.actions, result.speeds_m_s)
        ]
        report["selected_states"] = [
            {
                "x_m": state.x,
                "y_m": state.y,
                "yaw_deg": math.degrees(state.yaw),
                "u_m_s": state.longitudinal_velocity,
                "v_m_s": state.lateral_velocity,
                "yaw_rate_deg_s": math.degrees(state.yaw_rate),
                "friction_utilization": state.friction_utilization,
                "minimum_corner_normal_load_fraction": (
                    state.minimum_normal_load_fraction
                ),
                "abs_ltr": state.abs_ltr,
                "actuator_utilization": state.actuator_utilization,
            }
            for state in result.states
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("=== IJCAS INTEGRATED ROUTE/SPEED RIGID-BODY GATE ===")
    for row in candidate_reports:
        print(
            f"{row['route_id']}: "
            f"{'PASS' if row['feasible'] else 'NO DYNAMICALLY FEASIBLE PROFILE'} "
            f"actions={row['spatial_action_count']} "
            f"time={row['planning_time_s']:.2f}s "
            f"reject={row['rejection_counts']}"
        )
    print(f"selected={report['selected_route_id']}")
    print(f"saved: {output_path}")
    raise SystemExit(0 if selected is not None else 2)


if __name__ == "__main__":
    main()
