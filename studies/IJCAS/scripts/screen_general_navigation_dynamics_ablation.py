#!/usr/bin/env python3
"""Screen general navigation and isolate the path-planning dynamics benefit."""

import argparse
import csv
import heapq
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from patient_initial_stability.general_navigation_scenarios import (
    GENERAL_NAVIGATION_SCENARIOS,
    SCENARIO_GEOMETRY_STATUS,
)
from patient_initial_stability.steering_lattice import (
    ALL_MODES,
    BODY_LENGTH_M,
    BODY_WIDTH_M,
    FULL_DYNAMICS,
    KINEMATIC_DYNAMICS,
    STEP_M,
    TRANSITION_COST_DYNAMICS,
    STEERING_TRACKING_BIAS_RAD,
    State,
    plan,
    propagate,
    wrap,
)


POLICIES = {
    "fixed_counterphase": (("coordinated_counterphase",), FULL_DYNAMICS),
    "multimode_kinematic": (ALL_MODES, KINEMATIC_DYNAMICS),
    "multimode_transition_cost": (ALL_MODES, TRANSITION_COST_DYNAMICS),
    "proposed_full_dynamics": (ALL_MODES, FULL_DYNAMICS),
}


def make_collision_checker(obstacles, clearance_m):
    half_length = BODY_LENGTH_M / 2.0 + clearance_m
    half_width = BODY_WIDTH_M / 2.0 + clearance_m

    def collides(poses):
        for x, y, yaw in poses:
            cosine, sine = math.cos(yaw), math.sin(yaw)
            abs_cosine, abs_sine = abs(cosine), abs(sine)
            for minimum_x, maximum_x, minimum_y, maximum_y in obstacles:
                obstacle_x = 0.5 * (minimum_x + maximum_x)
                obstacle_y = 0.5 * (minimum_y + maximum_y)
                obstacle_half_x = 0.5 * (maximum_x - minimum_x)
                obstacle_half_y = 0.5 * (maximum_y - minimum_y)
                dx, dy = obstacle_x - x, obstacle_y - y
                if not (
                    abs(dx) > obstacle_half_x + half_length * abs_cosine + half_width * abs_sine
                    or abs(dy) > obstacle_half_y + half_length * abs_sine + half_width * abs_cosine
                    or abs(cosine * dx + sine * dy)
                    > half_length + obstacle_half_x * abs_cosine + obstacle_half_y * abs_sine
                    or abs(-sine * dx + cosine * dy)
                    > half_width + obstacle_half_x * abs_sine + obstacle_half_y * abs_cosine
                ):
                    return True
        return False

    return collides


def make_obstacle_distance_heuristic(scenario, clearance_m, speed_m_s=0.03):
    """Build a coarse obstacle-aware lower-bound distance-to-go heuristic."""
    resolution = 0.10
    minimum_x, maximum_x = scenario["bounds_x"]
    minimum_y, maximum_y = -3.0, 3.0
    x_count = int(math.floor((maximum_x - minimum_x) / resolution)) + 1
    y_count = int(math.floor((maximum_y - minimum_y) / resolution)) + 1
    # Inflating by half the body width guides the search around obstacles while
    # the exact oriented full-envelope checker remains the collision authority.
    inflation = BODY_WIDTH_M / 2.0 + clearance_m

    def point(index):
        ix, iy = index
        return minimum_x + ix * resolution, minimum_y + iy * resolution

    def blocked(index):
        x, y = point(index)
        return any(
            obstacle[0] - inflation <= x <= obstacle[1] + inflation
            and obstacle[2] - inflation <= y <= obstacle[3] + inflation
            for obstacle in scenario["obstacles"]
        )

    def index_for(x, y):
        return (
            min(x_count - 1, max(0, round((x - minimum_x) / resolution))),
            min(y_count - 1, max(0, round((y - minimum_y) / resolution))),
        )

    goal_index = index_for(scenario["goal"][0], scenario["goal"][1])
    distances = {goal_index: 0.0}
    queue = [(0.0, goal_index)]
    neighbours = (
        (-1, 0, resolution),
        (1, 0, resolution),
        (0, -1, resolution),
        (0, 1, resolution),
        (-1, -1, resolution * math.sqrt(2.0)),
        (-1, 1, resolution * math.sqrt(2.0)),
        (1, -1, resolution * math.sqrt(2.0)),
        (1, 1, resolution * math.sqrt(2.0)),
    )
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances.get(current, math.inf):
            continue
        for dx, dy, step in neighbours:
            candidate = (current[0] + dx, current[1] + dy)
            if not (0 <= candidate[0] < x_count and 0 <= candidate[1] < y_count):
                continue
            if blocked(candidate):
                continue
            new_distance = distance + step
            if new_distance >= distances.get(candidate, math.inf):
                continue
            distances[candidate] = new_distance
            heapq.heappush(queue, (new_distance, candidate))

    goal = scenario["goal"]

    def heuristic(state):
        euclidean = math.hypot(state.x - goal[0], state.y - goal[1])
        distance = distances.get(index_for(state.x, state.y), euclidean)
        heading_time = abs(wrap(state.yaw - goal[2])) / (0.70 * speed_m_s)
        return max(distance / speed_m_s, heading_time)

    return heuristic


def transition_counts(actions):
    reversals = sum(a[2] != b[2] for a, b in zip(actions, actions[1:]))
    switches = sum(a[0] != b[0] for a, b in zip(actions, actions[1:]))
    return reversals, switches


def replay_with_full_dynamics(actions, checker):
    # Isaac replay aligns the first steering target before rebasing the task
    # origin, so the first propagated primitive starts from its achieved target.
    state = State(0.0, 0.0, 0.0, "coordinated_counterphase", 0.0, 0.0, False, 0)
    collision = False
    replay_states = []
    for mode, curvature, direction in actions:
        state, poses, _ = propagate(
            state,
            mode,
            curvature,
            direction,
            dynamics=FULL_DYNAMICS,
        )
        collision = collision or checker(poses)
        replay_states.append(state)
    return state, collision, replay_states


def evaluate(task):
    (
        scenario_name,
        policy_name,
        node_limit,
        clearance_m,
        directions,
        heuristic_weight,
        obstacle_aware_heuristic,
    ) = task
    scenario = GENERAL_NAVIGATION_SCENARIOS[scenario_name]
    allowed_modes, dynamics = POLICIES[policy_name]
    checker = make_collision_checker(scenario["obstacles"], clearance_m)
    heuristic_function = (
        make_obstacle_distance_heuristic(scenario, clearance_m)
        if obstacle_aware_heuristic
        else None
    )
    started = time.perf_counter()
    result = plan(
        scenario["goal"],
        -3.0,
        3.0,
        robust_clearance=clearance_m,
        allowed_modes=allowed_modes,
        directions=directions,
        collision_checker=checker,
        node_limit=node_limit,
        minimum_x=scenario["bounds_x"][0],
        maximum_x=scenario["bounds_x"][1],
        # Execution performs an untimed first-target preview and rebases the
        # task origin afterward; mirror that initial condition in planning.
        propagate_initial_steering=False,
        dynamics=dynamics,
        # Leave margin for the measured post-stop yaw drift seen in calibration.
        goal_heading_tolerance_rad=math.radians(0.5),
        heuristic_weight=heuristic_weight,
        heuristic_function=heuristic_function,
    )
    elapsed = time.perf_counter() - started
    row = {
        "scenario": scenario_name,
        "policy": policy_name,
        "planner_model": dynamics.name,
        "plan_success": result is not None,
        "expanded_nodes": node_limit if result is None else result.expanded_nodes,
        "planning_time_s": elapsed,
        "action_count": "",
        "path_length_m": "",
        "direction_reversals": "",
        "mode_switches": "",
        "planned_position_error_m": "",
        "planned_heading_error_deg": "",
        "full_replay_collision": "",
        "full_replay_position_error_m": "",
        "full_replay_heading_error_deg": "",
        "full_replay_pass": False,
        "selected_modes": "",
        "path_cost": "",
        "motion_directions": ";".join(str(value) for value in directions),
        "heuristic_weight": heuristic_weight,
        "obstacle_aware_heuristic": obstacle_aware_heuristic,
        "front_tracking_bias_deg": math.degrees(STEERING_TRACKING_BIAS_RAD["front"])
        if dynamics.apply_identified_response
        else 0.0,
        "rear_tracking_bias_deg": math.degrees(STEERING_TRACKING_BIAS_RAD["rear"])
        if dynamics.apply_identified_response
        else 0.0,
    }
    path = []
    if result is None:
        return row, path

    goal_x, goal_y, goal_yaw = scenario["goal"]
    final = result.states[-1]
    replay_final, replay_collision, replay_states = replay_with_full_dynamics(
        result.actions, checker
    )
    planned_position_error = math.hypot(final.x - goal_x, final.y - goal_y)
    planned_heading_error = math.degrees(wrap(final.yaw - goal_yaw))
    replay_position_error = math.hypot(replay_final.x - goal_x, replay_final.y - goal_y)
    replay_heading_error = math.degrees(wrap(replay_final.yaw - goal_yaw))
    reversals, switches = transition_counts(result.actions)
    replay_pass = (
        not replay_collision
        and replay_position_error <= 0.030
        and abs(replay_heading_error) <= 2.0
    )
    row.update(
        {
            "action_count": len(result.actions),
            "path_length_m": len(result.actions) * STEP_M,
            "direction_reversals": reversals,
            "mode_switches": switches,
            "planned_position_error_m": planned_position_error,
            "planned_heading_error_deg": planned_heading_error,
            "full_replay_collision": replay_collision,
            "full_replay_position_error_m": replay_position_error,
            "full_replay_heading_error_deg": replay_heading_error,
            "full_replay_pass": replay_pass,
            "selected_modes": ";".join(sorted({a[0] for a in result.actions})),
            "path_cost": result.cost,
        }
    )
    path = [
        {
            "planned": {
                "x_m": planned.x,
                "y_m": planned.y,
                "yaw_deg": math.degrees(planned.yaw),
            },
            "full_replay": {
                "x_m": replay.x,
                "y_m": replay.y,
                "yaw_deg": math.degrees(replay.yaw),
            },
            "mode": action[0],
            "curvature_1_m": action[1],
            "direction": action[2],
        }
        for planned, replay, action in zip(result.states, replay_states, result.actions)
    ]
    return row, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-limit", type=int, default=30000)
    parser.add_argument("--clearance-m", type=float, default=0.010)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument("--obstacle-aware-heuristic", action="store_true")
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Restrict the common comparison to forward motion primitives.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(GENERAL_NAVIGATION_SCENARIOS),
        default=tuple(GENERAL_NAVIGATION_SCENARIOS),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(POLICIES),
        default=tuple(POLICIES),
    )
    parser.add_argument(
        "--output-dir",
        default="studies/IJCAS/results/general_navigation_dynamics_gate",
    )
    args = parser.parse_args()
    directions = (1,) if args.forward_only else (1, -1)
    tasks = [
        (
            scenario,
            policy,
            args.node_limit,
            args.clearance_m,
            directions,
            args.heuristic_weight,
            args.obstacle_aware_heuristic,
        )
        for scenario in args.scenarios
        for policy in args.policies
    ]
    rows, paths = [], {}

    def record(result):
        row, path = result
        rows.append(row)
        paths[f"{row['scenario']}:{row['policy']}"] = path
        print(
            row["scenario"],
            row["policy"],
            "PLAN" if row["plan_success"] else "NO_PATH_WITHIN_BUDGET",
            "REPLAY_PASS" if row["full_replay_pass"] else "REPLAY_FAIL",
            f"nodes={row['expanded_nodes']}",
            f"time={row['planning_time_s']:.3f}s",
            flush=True,
        )

    if args.workers == 1:
        for task in tasks:
            record(evaluate(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())

    policy_order = {name: index for index, name in enumerate(args.policies)}
    rows.sort(key=lambda row: (row["scenario"], policy_order[row["policy"]]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "general_navigation_dynamics_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path_path = output / "general_navigation_dynamics_paths.json"
    path_path.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")

    by_key = {(row["scenario"], row["policy"]): row for row in rows}
    lines = [
        "# General-navigation dynamics ablation gate",
        "",
        f"- geometry status: `{SCENARIO_GEOMETRY_STATUS}`",
        f"- common node budget: {args.node_limit}",
        f"- footprint: {BODY_LENGTH_M:.3f} x {BODY_WIDTH_M:.3f} m",
        f"- clearance: {args.clearance_m:.3f} m",
        f"- motion directions: {directions}",
        f"- heuristic weight: {args.heuristic_weight}",
        f"- obstacle-aware heuristic: {args.obstacle_aware_heuristic}",
        "- replay truth model: steering-rate-limited, mode-response-scaled full dynamics",
        "- proposed full-dynamics steering tracking/zero offset: front -1.0 deg, rear -1.0 deg at all targets, including inactive-wheel zero commands",
        "",
        "| scenario | policy | plan | nodes | time | path | reversals | switches | full-model replay | replay terminal error | modes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scenario in args.scenarios:
        for policy in args.policies:
            row = by_key[(scenario, policy)]
            if row["plan_success"]:
                replay_error = (
                    f"{1000*row['full_replay_position_error_m']:.1f} mm / "
                    f"{row['full_replay_heading_error_deg']:.2f} deg"
                )
                path = f"{row['path_length_m']:.2f} m"
            else:
                replay_error = path = "-"
            lines.append(
                f"| {scenario} | {policy} | "
                f"{'PASS' if row['plan_success'] else 'NO PATH'} | "
                f"{row['expanded_nodes']} | {row['planning_time_s']:.2f} s | {path} | "
                f"{row['direction_reversals']} | {row['mode_switches']} | "
                f"{'PASS' if row['full_replay_pass'] else 'FAIL'} | {replay_error} | "
                f"{row['selected_modes'] or '-'} |"
            )
    report_path = output / "general_navigation_dynamics_ablation.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved:", csv_path)
    print("saved:", path_path)
    print("saved:", report_path)


if __name__ == "__main__":
    main()

