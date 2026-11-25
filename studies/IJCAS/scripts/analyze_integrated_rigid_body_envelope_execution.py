#!/usr/bin/env python3
"""Close the integrated rigid-body + swept-envelope Isaac execution gate."""

import argparse
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
PLANNER_RESULT = (
    WORKSPACE / "studies/IJCAS/results/rigid_body_edge_gate/"
    "corridor_90_hierarchical_rigid_body_envelope.json"
)
DYNAMIC_DIRECTORY = (
    WORKSPACE / "studies/IJCAS/results/general_navigation_dynamics_gate/"
    "dynamic_replay"
)
CONTACT_DIRECTORY = (
    WORKSPACE / "studies/IJCAS/results/general_navigation_dynamics_gate/"
    "isaac_replay"
)
OUTPUT_DIRECTORY = (
    WORKSPACE / "studies/IJCAS/results/rigid_body_edge_gate/execution"
)


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--planner-result",
        default=str(PLANNER_RESULT),
        help="Integrated planner JSON; scenario and action count are inferred from it.",
    )
    parser.add_argument("--expected-condition", default="")
    parser.add_argument("--require-path-feedback", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id
    if not run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--run-id may contain only letters, numbers, '_' and '-'")

    planner_path = Path(args.planner_result).expanduser().resolve()
    planner = load(planner_path)
    scenario = planner.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise ValueError(f"planner result has no valid scenario: {planner_path}")
    if not scenario.replace("_", "").replace("-", "").isalnum():
        raise ValueError("planner scenario may contain only letters, numbers, '_' and '-'")
    dynamic_path = DYNAMIC_DIRECTORY / f"{scenario}_{run_id}.json"
    contact_path = CONTACT_DIRECTORY / f"{scenario}_{run_id}_contact.json"
    dynamic = load(dynamic_path)
    contact = load(contact_path)

    selected = next(
        row for row in planner["candidates"]
        if row["route_id"] == planner["selected_route_id"]
    )
    expected_actions = len(planner["selected_actions"])
    expected_speed_profile = [
        float(action["target_speed_m_s"])
        for action in planner["selected_actions"]
    ]
    executed_speed_profile = [
        float(value) for value in dynamic.get("planned_speed_profile_m_s", [])
    ]
    stopping_rejections = sum(
        int(row["rejection_counts"].get("stopping_reachability", 0))
        for row in planner["candidates"]
    )
    checks = {
        "result_identity_consistent": (
            dynamic.get("scenario") == scenario
            and contact.get("scenario") == scenario
            and dynamic.get("run_id") == run_id
            and contact.get("run_id") == run_id
            and dynamic.get("policy") == "external_integrated_plan"
            and (
                not args.expected_condition
                or dynamic.get("experiment_condition_id")
                == args.expected_condition
            )
            and (
                not args.expected_condition
                or contact.get("experiment_condition_id")
                == args.expected_condition
            )
        ),
        "planner_succeeded": bool(planner["plan_success"]),
        "rigid_stopping_constraint_activated": stopping_rejections > 0,
        "planner_terminal_state_passed": (
            selected["position_error_m"] <= 0.030
            and abs(selected["heading_error_deg"]) <= 2.0
            and selected["terminal_speed_m_s"] <= 0.012
        ),
        "execution_completed_all_edges": (
            dynamic["completed_actions"] == expected_actions
            and dynamic["total_actions"] == expected_actions
        ),
        "execution_used_planned_speed_states": (
            len(executed_speed_profile) == len(expected_speed_profile)
            and all(
                abs(observed - expected) <= 1.0e-12
                for observed, expected in zip(
                    executed_speed_profile, expected_speed_profile
                )
            )
        ),
        "execution_terminal_speed_passed": (
            float(dynamic.get("final_planar_speed_m_s", float("inf")))
            <= float(dynamic.get("stopped_speed_threshold_m_s", 0.005))
        ),
        "execution_terminal_stationary_window_passed": (
            float(
                dynamic.get(
                    "terminal_stationary_window_duration_s", 0.0
                )
            )
            >= float(
                dynamic.get(
                    "terminal_stationary_required_duration_s", float("inf")
                )
            )
            and float(
                dynamic.get(
                    "terminal_stationary_average_speed_m_s", float("inf")
                )
            )
            <= float(dynamic.get("stopped_speed_threshold_m_s", 0.005))
        ),
        "execution_terminal_stop_stable": (
            not bool(dynamic.get("terminal_stop_timeout_reached", False))
        ),
        "execution_terminal_pose_control_active": bool(
            dynamic.get("terminal_pose_control_enabled", False)
        ),
        "execution_path_tracking_requirement_passed": (
            not args.require_path_feedback
            or bool(dynamic.get("path_tracking_feedback_enabled", False))
        ),
        "execution_terminal_pose_passed": (
            dynamic["goal_position_error_m"] <= 0.030
            and abs(dynamic["goal_heading_error_deg"]) <= 2.0
        ),
        "full_envelope_contact_passed": (
            contact["obstacle_contact_event_count"] == 0
            and bool(contact["collision_free"])
        ),
        "finalizer_passed": bool(contact["pass"]),
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "scenario": scenario,
        "run_id": run_id,
        "experiment_condition_id": dynamic.get(
            "experiment_condition_id"
        ),
        "expected_condition_id": args.expected_condition or None,
        "path_tracking_feedback_required": args.require_path_feedback,
        "planner_result": str(planner_path),
        "dynamic_result": str(dynamic_path),
        "contact_result": str(contact_path),
        "selected_route_id": planner["selected_route_id"],
        "selected_action_count": expected_actions,
        "stopping_reachability_rejections": stopping_rejections,
        "planned_terminal": {
            "position_error_m": selected["position_error_m"],
            "heading_error_deg": selected["heading_error_deg"],
            "speed_m_s": selected["terminal_speed_m_s"],
        },
        "executed_terminal": {
            "position_error_m": dynamic["goal_position_error_m"],
            "heading_error_deg": dynamic["goal_heading_error_deg"],
            "speed_m_s": dynamic.get("final_planar_speed_m_s"),
            "stopped_speed_threshold_m_s": dynamic.get(
                "stopped_speed_threshold_m_s", 0.005
            ),
            "terminal_settle_elapsed_s": dynamic.get(
                "terminal_settle_elapsed_s"
            ),
            "terminal_stop_timeout_reached": dynamic.get(
                "terminal_stop_timeout_reached", False
            ),
            "stationary_window_duration_s": dynamic.get(
                "terminal_stationary_window_duration_s"
            ),
            "stationary_window_distance_m": dynamic.get(
                "terminal_stationary_window_distance_m"
            ),
            "stationary_average_speed_m_s": dynamic.get(
                "terminal_stationary_average_speed_m_s"
            ),
        },
        "contact_events": contact["obstacle_contact_event_count"],
        "checks": checks,
        "verdict": verdict,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIRECTORY / f"{scenario}_integrated_{run_id}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("=== IJCAS INTEGRATED RIGID-BODY + ENVELOPE EXECUTION GATE ===")
    print(f"route/actions: {planner['selected_route_id']} / {expected_actions}")
    print(f"stopping-reachability rejected edges: {stopping_rejections}")
    print(
        "executed terminal: "
        f"{1000.0 * dynamic['goal_position_error_m']:.2f} mm / "
        f"{dynamic['goal_heading_error_deg']:.3f} deg / "
        f"{dynamic.get('final_planar_speed_m_s', float('nan')):.4f} m/s"
    )
    print(f"contact events: {contact['obstacle_contact_event_count']}")
    print(f"VERDICT: {verdict}")
    print(f"saved: {output}")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
