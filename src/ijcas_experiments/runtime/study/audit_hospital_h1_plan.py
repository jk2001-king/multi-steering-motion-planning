#!/usr/bin/env python3
"""Independently replay stored H1 actions through the planning model, not Isaac."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
from patient_initial_stability.steering_lattice import (
    FULL_DYNAMICS, KINEMATIC_DYNAMICS, TRANSITION_COST_DYNAMICS, State, propagate, wrap,
)
from hospital_geometry_tools import pose_collision
from hospital_planning_tools import local_to_world
from hospital_path_methods import method_settings


def audit(path):
    report = json.loads(path.read_bytes())
    settings = method_settings(report["method"])
    if (not report.get("plan_success") or not report.get("selected_actions")
            or len(report["selected_actions"]) != len(report["selected_states"])):
        raise ValueError("Need a complete successful H1 plan")
    if report["planner_dynamics"] != settings["planner_dynamics"]:
        raise ValueError("Method/dynamics mismatch")
    is_proposed = report["method"] == "P_dynamics_full_envelope"
    dynamics = {"full": FULL_DYNAMICS, "kinematic": KINEMATIC_DYNAMICS, "transition_cost": TRANSITION_COST_DYNAMICS}[report["planner_dynamics"]]
    with np.load(path.parent / "planning_occupancy.npz") as grid:
        occupied, origin, resolution = grid["occupied"], grid["origin_xy_m"], float(grid["resolution_m"])
        if hashlib.sha256(occupied.tobytes()).hexdigest() != report["planning_occupancy_digest"]:
            raise ValueError("The planning raster changed")
        model = MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**report["rigid_body_parameter_bounds"])) if is_proposed else None
        state = State(0., 0., 0., "coordinated_counterphase", report["initial_front_rad"], report["initial_rear_rad"],
            report["initial_steering_propagated"])
        collisions, infeasible_edges, samples, largest_state_error = 0, 0, 0, 0.
        for action, saved in zip(report["selected_actions"], report["selected_states"]):
            if action["mode"] not in report["allowed_modes"]:
                raise ValueError("Action outside this method's permitted planning modes")
            state, poses, _ = propagate(state, action["mode"], action["curvature_1_m"], action["direction"],
                speed_m_s=report["common_max_command_speed_m_s"], dynamics=dynamics,
                rigid_body_model=model, target_speed_m_s=action["target_speed_m_s"])
            infeasible_edges += not state.rigid_body_feasible
            for pose in poses:
                world = local_to_world(pose, report["start_world_xyyaw"])
                # Deliberately use the original polygon/cell SAT, not the
                # accelerated planning checker, for independent agreement.
                collisions += pose_collision(occupied, world, origin, resolution, report["full_envelope_clearance_m"])
                samples += 1
            error = max(abs(state.x-saved["x_m"]), abs(state.y-saved["y_m"]), abs(wrap(state.yaw-math.radians(saved["yaw_deg"]))),
                abs(state.front-saved["front_rad"]), abs(state.rear-saved["rear_rad"]))
            world = local_to_world((state.x, state.y, state.yaw), report["start_world_xyyaw"])
            error = max(error, *(abs(a-b) for a, b in zip(world, saved["world_xyyaw"])))
            if is_proposed:
                error = max(error, abs(state.longitudinal_velocity-saved["u_m_s"]),
                    abs(state.lateral_velocity-saved["v_m_s"]), abs(state.yaw_rate-saved["yaw_rate_rad_s"]))
            largest_state_error = max(largest_state_error, error)
    goal = report["goal_local_xyyaw"]
    position_error, heading_error = math.hypot(state.x-goal[0], state.y-goal[1]), abs(wrap(state.yaw-goal[2]))
    speed = math.hypot(state.longitudinal_velocity, state.lateral_velocity)
    checks = {"action_state_lengths_match": len(report["selected_actions"]) == len(report["selected_states"]) > 0,
        "all_substep_envelopes_clear_by_original_SAT": collisions == 0,
        "stored_and_repropagated_states_match": largest_state_error <= 1e-10,
        "planning_position_tolerance_25mm": position_error <= .025,
        "planning_heading_tolerance_2deg": heading_error <= math.radians(2.)}
    if is_proposed:
        checks.update({"all_edges_rigid_body_feasible": infeasible_edges == 0,
            "planning_terminal_speed_under_5mm_s": speed <= .005})
    return {"schema_version": 2, "trial_kind": "independent_OFFLINE_plan_repropagation_not_PhysX_execution",
        "method": report["method"], "planner_dynamics": report["planner_dynamics"],
        "rigid_body_feasibility_evaluated": is_proposed,
        "scope": "kinematic geometry/state consistency only; actual dynamic feasibility is NOT certified" if not is_proposed else "full planning model repropagation",
        "plan_path": str(path), "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "verdict": "PASS" if all(checks.values()) else "FAIL", "action_count": len(report["selected_actions"]),
        "substep_footprint_sample_count": samples, "colliding_samples": int(collisions),
        "maximum_stored_state_difference": largest_state_error,
        "terminal_position_error_m": position_error, "terminal_heading_error_deg": math.degrees(heading_error),
        "terminal_speed_m_s": speed if is_proposed else None, "Isaac_execution_verified": False, "Nav2_execution_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    result = audit(args.plan.resolve())
    directory = args.plan.parent / "offline_audits"
    directory.mkdir(exist_ok=True)
    number = 1
    while (directory / f"audit_r{number:02d}.json").exists():
        number += 1
    target = directory / f"audit_r{number:02d}.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("=== HOSPITAL H1 INDEPENDENT OFFLINE PLAN AUDIT ===")
    print("actions/substep footprints:", result["action_count"], result["substep_footprint_sample_count"])
    print("colliding samples:", result["colliding_samples"])
    print("failed checks:", [key for key, value in result["checks"].items() if not value])
    print("VERDICT:", result["verdict"])
    print("saved:", target)
    print("This is NOT an Isaac or Nav2 execution result.")
    raise SystemExit(0 if result["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
