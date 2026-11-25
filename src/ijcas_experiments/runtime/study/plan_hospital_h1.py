#!/usr/bin/env python3
"""Develop matched H1 plans on the actual audited hospital raster, not boxes.

Outputs are offline planning evidence only. They do not claim PhysX execution,
sensor autonomy, frozen paper geometry or physical-platform validity.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
from patient_initial_stability.steering_lattice import (
    ALL_MODES, FULL_DYNAMICS, KINEMATIC_DYNAMICS, TRANSITION_COST_DYNAMICS, STEP_M, plan, wrap,
    State, PlanResult, propagate,
)
from hospital_geometry_tools import polygon_cells
from hospital_planning_tools import HospitalEnvelopeChecker, local_to_world
from plan_corridor_90_rigid_body_robust import _load_baselines, _model


WORKSPACE = _ijcas_workspace()
ROOT = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
METHODS = {
    "B0_fixed_counterphase_ramp": (("coordinated_counterphase",), KINEMATIC_DYNAMICS),
    "B1_multimode_kinematic_ramp": (ALL_MODES, KINEMATIC_DYNAMICS),
    "B2_multimode_transition_cost_ramp": (ALL_MODES, TRANSITION_COST_DYNAMICS),
    "P_dynamics_full_envelope": (ALL_MODES, FULL_DYNAMICS),
}
SPEED = .05
SPEED_LEVELS = (.003, .010, .030, .050)
DECELERATION = .02
TERMINAL_SPEED = .005


def pose_guidance(state, goal):
    """Common non-admissible pose guidance for bounded weighted A*.

    Encourages building lateral displacement early rather than reaching the
    goal's X coordinate straight and then attempting an impossible late turn.
    It is an explicit search heuristic, not an optimality or dynamics claim.
    """
    dx, dy = goal[0]-state.x, goal[1]-state.y
    distance = math.hypot(dx, dy)
    if distance <= .025:
        return abs(wrap(goal[2]-state.yaw))/(.70*SPEED)
    bearing = math.atan2(dy, dx)
    turn = abs(wrap(bearing-state.yaw))+abs(wrap(goal[2]-bearing))
    return distance/SPEED + .4*turn/(.70*SPEED)


def state_record(state, start, dynamic):
    world = local_to_world((state.x, state.y, state.yaw), start)
    return {"x_m": state.x, "y_m": state.y, "yaw_deg": math.degrees(state.yaw),
        "world_xyyaw": world, "mode": state.mode, "front_rad": state.front, "rear_rad": state.rear,
        "u_m_s": state.longitudinal_velocity if dynamic else None,
        "v_m_s": state.lateral_velocity if dynamic else None,
        "yaw_rate_rad_s": state.yaw_rate if dynamic else None,
        "friction_utilization": state.friction_utilization if dynamic else None,
        "minimum_normal_load_fraction": state.minimum_normal_load_fraction if dynamic else None,
        "abs_ltr": state.abs_ltr if dynamic else None,
        "actuator_utilization": state.actuator_utilization if dynamic else None}


def counterphase_connector(goal, checker):
    """Enumerate discrete turn/straight/opposite-turn fixed-mode connections.

    A specialised strong baseline for the H1 same-heading offset task. Every
    state and footprint comes from the existing lattice propagator; no drawn
    or manually fitted route is accepted. This family is not a completeness
    claim; if absent, generic lattice search remains available.
    """
    if abs(wrap(goal[2])) > math.radians(2.):
        return None
    first_count = max(1, int(math.hypot(*goal[:2])/STEP_M)-1)
    tested = 0
    for count in range(first_count, first_count+9):
        feasible = []
        for magnitude in (.25, .50, .70):
            curvature = math.copysign(magnitude, goal[1])
            for turn_count in range(1, count//2+1):
                checker.check_deadline()
                tested += 1
                curvatures = [curvature]*turn_count+[0.]*(count-2*turn_count)+[-curvature]*turn_count
                state = State(0., 0., 0., "coordinated_counterphase", 0., 0., False)
                states, edge_poses = [], []
                for value in curvatures:
                    state, poses, _ = propagate(state, "coordinated_counterphase", value, 1,
                        speed_m_s=SPEED, dynamics=KINEMATIC_DYNAMICS)
                    states.append(state)
                    edge_poses.extend(poses)
                error = math.hypot(state.x-goal[0], state.y-goal[1])
                if error > .025 or abs(wrap(state.yaw-goal[2])) > math.radians(2.):
                    continue
                if checker(edge_poses):
                    continue
                actions = tuple(("coordinated_counterphase", value, 1) for value in curvatures)
                feasible.append((error, PlanResult(tuple(states), actions, tested,
                    count*STEP_M/SPEED+.02*sum(abs(value) for value in curvatures), ())))
        if feasible:
            return min(feasible, key=lambda item: (item[1].cost, item[0]))[1]
    return None


def plan_method(method, manifest, occupied, map_origin, resolution, model, budget_s, node_limit, spatial_candidates=None):
    start, goal = manifest["start_xyyaw"], tuple(manifest["local_goal_xyyaw"])
    modes, dynamics = METHODS[method]
    is_proposed = method == "P_dynamics_full_envelope"
    begun = time.perf_counter()
    checker = HospitalEnvelopeChecker(occupied, map_origin, resolution, start, deadline=begun+budget_s)
    row = {"method": method, "plan_success": False, "status": "NOT_RUN", "use_for_final_statistics": False,
        "planner_dynamics": dynamics.name, "allowed_modes": modes, "node_limit": node_limit,
        "wall_clock_budget_s": budget_s, "heuristic_weight": 1.5,
        "heuristic": "shared_pose_guidance_non_admissible_no_global_optimality_claim",
        "initial_steering_propagated": is_proposed,
        "initial_front_rad": 0., "initial_rear_rad": 0., "initial_u_v_r": [0., 0., 0.],
        "initial_state_scope": "nominal standstill plan; actual release-state tolerance must be checked in execution",
        "selected_actions": [], "selected_states": [], "refinement_attempts": []}
    diagnostics = {}
    try:
        if checker([(0., 0., 0.), goal]):
            raise ValueError("Start/goal full envelope is blocked")
        if is_proposed:
            row["planner_form"] = "hierarchical_spatial_candidates_then_speed_consistent_Newton_Euler_selection"
            row["newton_euler_applied_during_candidate_generation"] = False
            row["newton_euler_applied_to_every_refined_edge"] = True
            row["global_joint_state_search_claimed"] = False
            row["configuration_cost_enabled"] = True
            pool = dict(spatial_candidates or {})
            if not pool:
                connector = counterphase_connector(goal, checker)
                if connector:
                    pool["B0_fixed_counterphase_ramp"] = [{"mode": a[0], "curvature_1_m": a[1], "direction": a[2]} for a in connector.actions]
            row["candidate_generator"] = "shared_collision_checked_spatial_candidates"
            row["spatial_candidates"] = pool
            row["candidate_generation_time_s"] = time.perf_counter()-begun
            feasible = []
            budget_exhausted = False
            for route_id in sorted(pool):
                # At H1 the baseline primitive counts already reach the task
                # endpoint. Refine their controls at executable speeds rather
                # than silently appending a forward edge past the goal.
                try:
                    checker.check_deadline()
                    rejections = []
                    refined, audit = refine_speed_consistent_lattice_route(route_id, pool[route_id], goal, SPEED_LEVELS,
                        model, checker, TERMINAL_SPEED, DECELERATION, step_m=STEP_M,
                        terminal_extension_edges=0, initial_front_rad=0., initial_rear_rad=0.,
                        propagate_initial_steering=True, include_configuration_cost=True, rejection_samples=rejections)
                    row["refinement_attempts"].append({"source_route_id": route_id, "terminal_extension_edges": 0,
                        "success": refined is not None, "diagnostics": audit, "rejection_samples": rejections})
                    if refined is not None:
                        feasible.append((refined, len(row["refinement_attempts"])-1))
                except TimeoutError as error:
                    budget_exhausted = True
                    row["refinement_attempts"].append({"source_route_id": route_id, "success": False,
                        "status": "TIME_BUDGET_EXHAUSTED_NOT_PROVEN_INFEASIBLE", "error": str(error)})
                    break
            row["all_spatial_candidates_evaluated"] = not budget_exhausted
            if not feasible:
                row["status"] = ("PLANNING_TIMEOUT_NOT_PROVEN_INFEASIBLE" if budget_exhausted else
                    "NO_FEASIBLE_REFINED_PROFILE_WITHIN_LOCAL_SEARCH_NOT_GLOBAL_INFEASIBILITY")
                return row
            selected, selected_index = min(feasible, key=lambda item: item[0].cost_s)
            row["selected_refinement_attempt_index"] = selected_index
            row["selected_spatial_source_route_id"] = selected.route_id
            states, final_actions = selected.states, selected.actions
            speeds = selected.speeds_m_s
            row["predicted_edge_times_s"] = selected.edge_times_s
            row["predicted_motion_time_s"] = sum(selected.edge_times_s)
            row["search_cost_s"] = selected.cost_s
        else:
            result = counterphase_connector(goal, checker) if method == "B0_fixed_counterphase_ramp" else None
            row["candidate_generator"] = "enumerated_fixed_mode_lattice_connector" if result else "weighted_steering_lattice"
            if result is None:
                result = plan(goal, -2., 2., allowed_modes=modes, speed_m_s=SPEED,
                    node_limit=node_limit, collision_checker=checker, minimum_x=-.3, maximum_x=goal[0]+.4,
                    dynamics=dynamics, directions=(1,), initial_front_rad=0., initial_rear_rad=0., heuristic_weight=1.5,
                    heuristic_function=lambda state: (checker.check_deadline() or pose_guidance(state, goal)),
                    rigid_body_model=None, diagnostics=diagnostics)
            row["candidate_generation_time_s"] = time.perf_counter()-begun
            row["candidate_generation_diagnostics"] = diagnostics
            if result is None:
                row["status"] = "SEARCH_BUDGET_OR_DISCRETE_FRONTIER_EXHAUSTED_NOT_PROVEN_INFEASIBLE"
                return row
            actions = [{"mode": a[0], "curvature_1_m": a[1], "direction": a[2]} for a in result.actions]
            row["spatial_candidate_actions"] = actions
            row["spatial_candidate_action_count"] = len(actions)
            states, final_actions, speeds = result.states, actions, [SPEED]*len(actions)
            row["search_cost_s"] = result.cost
            row["nominal_constant_speed_travel_time_s"] = len(actions)*STEP_M/SPEED
            row["baseline_execution_schedule"] = "common acceleration/braking/terminal controller; kinematic states do not predict body velocities"
        final = states[-1]
        row.update({"plan_success": True, "status": "OFFLINE_PLAN_FOUND_EXECUTION_PENDING",
            "selected_actions": [{**action, "target_speed_m_s": speed} for action, speed in zip(final_actions, speeds)],
            "selected_states": [state_record(state, start, is_proposed) for state in states],
            "action_count": len(final_actions), "path_length_m": len(final_actions)*STEP_M,
            "mode_switch_count": sum(a["mode"] != b["mode"] for a, b in zip(final_actions, final_actions[1:])),
            "predicted_terminal_position_error_m": math.hypot(final.x-goal[0], final.y-goal[1]),
            "predicted_terminal_heading_error_deg": math.degrees(wrap(final.yaw-goal[2])),
            "predicted_terminal_speed_m_s": math.hypot(final.longitudinal_velocity, final.lateral_velocity) if is_proposed else None,
            "costs_not_directly_cross_comparable": True,
            "physical_execution_verified": False, "Nav2_execution_verified": False})
    except TimeoutError as error:
        row.update(status="PLANNING_TIMEOUT_NOT_PROVEN_INFEASIBLE", error=str(error),
            candidate_generation_diagnostics=diagnostics)
    finally:
        row["planning_time_s"] = time.perf_counter()-begun
        row["collision_checker_statistics"] = checker.statistics
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--budget-s", type=float, default=45.)
    parser.add_argument("--node-limit", type=int, default=20000)
    args = parser.parse_args()
    if not math.isfinite(args.budget_s) or args.budget_s <= 0 or args.node_limit <= 0:
        parser.error("Planning budgets must be positive")
    manifest_path = _ijcas_path(ROOT, 'hospital_h1_geometry.json')
    manifest = json.loads(manifest_path.read_bytes())
    if not all(manifest["checks"].values()):
        raise ValueError("H1 endpoint geometry checks did not pass")
    if hashlib.sha256(Path(manifest["source_audit"]).read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Hospital source audit changed; rebuild the map")
    grid_path = Path(manifest["occupancy_grid"])
    with np.load(grid_path) as grid:
        occupied, origin, resolution = grid["occupied"].copy(), grid["origin_xy_m"].copy(), float(grid["resolution_m"])
    if (occupied.shape != tuple(manifest["raster"]["shape_rows_columns"]) or
            not np.allclose(origin, manifest["raster"]["origin_xy_m"]) or not math.isclose(resolution, manifest["raster"]["resolution_m"])):
        raise ValueError("Hospital raster/manifest mismatch")
    # Unknown wheel/threshold traversal must not be silently treated as free.
    terrain_paths = []
    for terrain in manifest["low_terrain_candidates"]:
        lo, hi = terrain["bound"]["minimum_m"], terrain["bound"]["maximum_m"]
        rr, cc = polygon_cells([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])], origin, resolution, occupied.shape)
        occupied[rr, cc] = True
        terrain_paths.append(terrain["path"])
    parent = _ijcas_path(ROOT, 'planning')
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while (parent / f"hospital_h1_planning_r{index:02d}").exists():
        index += 1
    output = parent / f"hospital_h1_planning_r{index:02d}"
    output.mkdir()
    model = _model(_load_baselines())
    metadata = {"schema_version": 1, "scenario": "H1_bedside_departure", "trial_kind": "offline_hospital_planning_development",
        "geometry_manifest": str(manifest_path), "geometry_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_occupancy_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "planning_occupancy_digest": hashlib.sha256(occupied.tobytes()).hexdigest(),
        "conservatively_blocked_low_terrain_paths": terrain_paths, "source_stage_modified": False,
        "start_world_xyyaw": manifest["start_xyyaw"], "goal_world_xyyaw": manifest["goal_xyyaw"],
        "goal_local_xyyaw": manifest["local_goal_xyyaw"], "common_max_command_speed_m_s": SPEED,
        "common_max_acceleration_m_s2": DECELERATION, "planning_terminal_speed_tolerance_m_s": TERMINAL_SPEED,
        "full_envelope_clearance_m": .010,
        "collision_method": "full_envelope_cell_supercover_at_all_propagation_substeps",
        "rigid_body_parameter_bounds": asdict(model.bounds),
        "rigid_body_parameter_scope": "inherited provisional simulation sensitivity/disturbance bounds; not hospital or physical identification",
        "use_for_final_statistics": False, "Isaac_execution_verified": False, "Nav2_execution_verified": False}
    np.savez_compressed(output / "planning_occupancy.npz", occupied=occupied, origin_xy_m=origin, resolution_m=resolution)
    selected_methods = METHODS if args.method == "all" else (args.method,)
    print("=== IJCAS HOSPITAL H1 OFFLINE PLANNING ===", flush=True)
    print("output:", output, flush=True)
    reports = []
    spatial_candidates = {}
    for method in selected_methods:
        print("planning:", method, flush=True)
        row = plan_method(method, manifest, occupied, origin, resolution, model, args.budget_s, args.node_limit, spatial_candidates)
        if row["plan_success"] and method != "P_dynamics_full_envelope":
            spatial_candidates[method] = row["selected_actions"]
        report = {**metadata, **row}
        with (output / (method+".json")).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        reports.append({key: row.get(key) for key in ("method", "status", "plan_success", "planning_time_s", "action_count", "path_length_m", "mode_switch_count")})
        print(method, row["status"], f"{row['planning_time_s']:.2f}s", "actions=", row.get("action_count"), flush=True)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump({**metadata, "methods": reports}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Offline planning only. Do not count these as hospital execution PASS results.", flush=True)


if __name__ == "__main__":
    main()
