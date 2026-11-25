"""Joint route-candidate and speed-state planning with rigid-body edges.

The spatial lattice can generate several collision-free route candidates.  This
module performs dynamic programming over every candidate's motion primitives
and speed states.  Each transition is re-propagated with the same steering-rate
and Newton--Euler model used by the lattice, and its complete swept footprint
is collision checked.  The final route is therefore selected *after* dynamic
feasibility and cost have been applied, rather than by a kinematic post-check.
"""

import math
from dataclasses import dataclass

from .steering_lattice import (
    CURVATURES, FULL_DYNAMICS, MODE_SWITCH_PENALTY_S,
    STEERING_RATE_RAD_S, STEERING_TRANSIENT_PENALTY_WEIGHT,
    State, propagate, steering_targets, wrap,
)


@dataclass(frozen=True)
class DynamicRouteResult:
    route_id: str
    states: tuple
    actions: tuple
    speeds_m_s: tuple
    edge_times_s: tuple
    cost_s: float
    rejection_counts: tuple


def _state_key(state):
    """Quantize only for dominance; exact states are retained in the result."""
    return (
        round(state.x / 0.010),
        round(state.y / 0.010),
        round(state.yaw / math.radians(0.5)),
        round(state.front / math.radians(2.0)),
        round(state.rear / math.radians(2.0)),
        round(state.longitudinal_velocity / 0.005),
        round(state.lateral_velocity / 0.005),
        round(state.yaw_rate / math.radians(0.5)),
    )


def refine_speed_consistent_lattice_route(
    route_id,
    actions,
    goal,
    speed_levels_m_s,
    rigid_body_model,
    collision_checker,
    terminal_speed_tolerance_m_s,
    terminal_deceleration_m_s2,
    step_m=0.05,
    terminal_extension_edges=1,
    initial_mode="coordinated_counterphase",
    goal_position_tolerance_m=0.025,
    goal_heading_tolerance_rad=math.radians(2.0),
    maximum_iterations=12,
    mode_switch_window=0,
    terminal_window=4,
    heading_objective_weight=60.0,
    dynamics=FULL_DYNAMICS,
    rejection_samples=None,
    maximum_rejection_samples=512,
    initial_front_rad=0.0,
    initial_rear_rad=0.0,
    propagate_initial_steering=False,
    include_configuration_cost=False,
):
    """Locally refine a spatial route using its executable speed schedule.

    The search remains on the configured curvature lattice.  It only changes
    the initial steering build-up edge, edges around a steering-mode switch,
    and terminal braking edges.  Those are precisely the locations where a
    route generated at one speed ceases to be geometrically invariant after
    steering-rate-limited speed propagation.
    """
    if not actions:
        raise ValueError("actions must not be empty")
    if terminal_extension_edges < 0:
        raise ValueError("terminal_extension_edges must be nonnegative")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if not all(math.isfinite(value) for value in (initial_front_rad, initial_rear_rad)):
        raise ValueError("Initial steering angles must be finite")
    initial_state = State(0.0, 0.0, 0.0, initial_mode,
        initial_front_rad, initial_rear_rad, propagate_initial_steering, 0)
    speeds = tuple(sorted(set(float(value) for value in speed_levels_m_s)))
    if not speeds or speeds[0] <= 0.0:
        raise ValueError("speed_levels_m_s must contain positive values")

    refined_actions = [
        {
            "mode": action["mode"] if isinstance(action, dict) else action[0],
            "curvature_1_m": float(
                action["curvature_1_m"] if isinstance(action, dict) else action[1]
            ),
            "direction": int(
                action["direction"] if isinstance(action, dict) else action[2]
            ),
        }
        for action in actions
    ]
    refined_actions.extend(
        dict(refined_actions[-1]) for _ in range(terminal_extension_edges)
    )
    speed_profile = []
    for action_index in range(len(refined_actions)):
        remaining_arc_m = (
            len(refined_actions) - action_index - 1
        ) * step_m
        maximum_stoppable_speed = math.sqrt(
            terminal_speed_tolerance_m_s**2
            + 2.0 * terminal_deceleration_m_s2 * remaining_arc_m
        )
        stoppable = tuple(
            speed for speed in speeds
            if speed <= maximum_stoppable_speed + 1.0e-12
        )
        speed_profile.append(max(stoppable) if stoppable else min(speeds))

    candidate_indices = {0}
    for index in range(1, len(refined_actions)):
        if (
            refined_actions[index]["mode"]
            != refined_actions[index - 1]["mode"]
            or refined_actions[index]["curvature_1_m"]
            != refined_actions[index - 1]["curvature_1_m"]
        ):
            candidate_indices.update(
                range(
                    max(0, index - mode_switch_window),
                    min(len(refined_actions), index + mode_switch_window + 1),
                )
            )
    candidate_indices.update(
        range(
            max(0, len(refined_actions) - terminal_window),
            len(refined_actions),
        )
    )
    candidate_indices = tuple(sorted(candidate_indices))
    diagnostics = {
        "generated": 0,
        "rigid_body": 0,
        "stopping_reachability": 0,
        "collision": 0,
        "refinement_changes": [],
        "candidate_action_indices": list(candidate_indices),
        "reference_speed_profile_m_s": list(speed_profile),
    }

    def evaluate(candidate_actions, changed_index=0, base_result=None):
        if changed_index > 0 and base_result is not None:
            state = base_result["states"][changed_index - 1]
            state_chain = list(base_result["states"][:changed_index])
            edge_times = list(base_result["edge_times"][:changed_index])
            total_cost = sum(edge_times) + sum(
                item.rigid_body_penalty_s for item in state_chain
            )
            configuration_costs = list(base_result["configuration_costs"][:changed_index])
            total_cost += sum(configuration_costs)
        else:
            changed_index = 0
            state = initial_state
            state_chain = []
            edge_times = []
            total_cost = 0.0
            configuration_costs = []
        valid = True
        for action_index in range(changed_index, len(candidate_actions)):
            action = candidate_actions[action_index]
            target_speed = speed_profile[action_index]
            previous_state = state
            state, poses, edge_time = propagate(
                state,
                action["mode"],
                action["curvature_1_m"],
                action["direction"],
                step_m=step_m,
                speed_m_s=max(speeds),
                dynamics=dynamics,
                rigid_body_model=rigid_body_model,
                target_speed_m_s=target_speed,
            )
            state_chain.append(state)
            edge_times.append(edge_time)
            configuration_cost = 0.0
            if include_configuration_cost and previous_state.started:
                if dynamics.include_mode_switch_cost and action["mode"] != previous_state.mode:
                    configuration_cost += MODE_SWITCH_PENALTY_S
                if dynamics.include_steering_transient_cost:
                    front_target, rear_target = steering_targets(action["mode"], action["curvature_1_m"])
                    configuration_cost += STEERING_TRANSIENT_PENALTY_WEIGHT*max(
                        abs(front_target-previous_state.front), abs(rear_target-previous_state.rear))/STEERING_RATE_RAD_S
            configuration_costs.append(configuration_cost)
            total_cost += edge_time + state.rigid_body_penalty_s + configuration_cost
            if not state.rigid_body_feasible:
                diagnostics["rigid_body"] += 1
                valid = False
                break
            remaining_arc_m = (
                len(candidate_actions) - action_index - 1
            ) * step_m
            maximum_stoppable_speed = math.sqrt(
                terminal_speed_tolerance_m_s**2
                + 2.0 * terminal_deceleration_m_s2 * remaining_arc_m
            )
            if math.hypot(
                state.longitudinal_velocity,
                state.lateral_velocity,
            ) > maximum_stoppable_speed + 1.0e-12:
                diagnostics["stopping_reachability"] += 1
                valid = False
                break
            if collision_checker(poses):
                diagnostics["collision"] += 1
                valid = False
                break
        if not valid:
            return math.inf, None
        position_error = math.hypot(state.x - goal[0], state.y - goal[1])
        heading_error = abs(wrap(state.yaw - goal[2]))
        terminal_speed = math.hypot(
            state.longitudinal_velocity,
            state.lateral_velocity,
        )
        score = 100.0 * position_error + heading_objective_weight * heading_error
        return score, {
            "states": tuple(state_chain),
            "edge_times": tuple(edge_times),
            "configuration_costs": tuple(configuration_costs),
            "cost": total_cost,
            "position_error": position_error,
            "heading_error": heading_error,
            "terminal_speed": terminal_speed,
        }

    current_score, current = evaluate(refined_actions)
    for _ in range(maximum_iterations):
        best_score = current_score
        best_actions = None
        best_result = None
        best_change = None
        for action_index in candidate_indices:
            current_curvature = refined_actions[action_index]["curvature_1_m"]
            for curvature in CURVATURES:
                if curvature == current_curvature:
                    continue
                diagnostics["generated"] += 1
                candidate_actions = [dict(action) for action in refined_actions]
                candidate_actions[action_index]["curvature_1_m"] = curvature
                score, result = evaluate(
                    candidate_actions,
                    changed_index=action_index,
                    base_result=current,
                )
                if score + 1.0e-12 < best_score:
                    best_score = score
                    best_actions = candidate_actions
                    best_result = result
                    best_change = (action_index, curvature)
        if best_actions is None:
            break
        refined_actions = best_actions
        current_score = best_score
        current = best_result
        diagnostics["refinement_changes"].append(best_change)
        if (
            current["position_error"] <= goal_position_tolerance_m
            and current["heading_error"] <= goal_heading_tolerance_rad
            and current["terminal_speed"] <= terminal_speed_tolerance_m_s
        ):
            break

    if current is None:
        return None, diagnostics
    diagnostics.update(
        {
            "terminal_position_error_m": current["position_error"],
            "terminal_heading_error_deg": math.degrees(current["heading_error"]),
            "terminal_speed_m_s": current["terminal_speed"],
        }
    )
    if not (
        current["position_error"] <= goal_position_tolerance_m
        and current["heading_error"] <= goal_heading_tolerance_rad
        and current["terminal_speed"] <= terminal_speed_tolerance_m_s
    ):
        return None, diagnostics

    # Explicitly audit faster alternatives rejected by the stopping-distance
    # inequality. These are planner edge rejections, not fabricated display
    # samples, and are retained for the validation overlay and causal report.
    previous_state = initial_state
    for action_index, (action, selected_speed) in enumerate(
        zip(refined_actions, speed_profile)
    ):
        remaining_arc_m = (
            len(refined_actions) - action_index - 1
        ) * step_m
        maximum_stoppable_speed = math.sqrt(
            terminal_speed_tolerance_m_s**2
            + 2.0 * terminal_deceleration_m_s2 * remaining_arc_m
        )
        for alternative_speed in speeds:
            if alternative_speed <= selected_speed:
                continue
            alternative_state, _, _ = propagate(
                previous_state,
                action["mode"],
                action["curvature_1_m"],
                action["direction"],
                step_m=step_m,
                speed_m_s=max(speeds),
                dynamics=dynamics,
                rigid_body_model=rigid_body_model,
                target_speed_m_s=alternative_speed,
            )
            if math.hypot(
                alternative_state.longitudinal_velocity,
                alternative_state.lateral_velocity,
            ) <= maximum_stoppable_speed + 1.0e-12:
                continue
            diagnostics["stopping_reachability"] += 1
            if (
                rejection_samples is not None
                and len(rejection_samples) < maximum_rejection_samples
            ):
                rejection_samples.append(
                    {
                        "reason": "stopping_reachability",
                        "action_index": action_index,
                        "start_x_m": previous_state.x,
                        "start_y_m": previous_state.y,
                        "start_yaw_rad": previous_state.yaw,
                        "end_x_m": alternative_state.x,
                        "end_y_m": alternative_state.y,
                        "end_yaw_rad": alternative_state.yaw,
                        "target_speed_m_s": alternative_speed,
                    }
                )
        previous_state = current["states"][action_index]
    return DynamicRouteResult(
        route_id,
        current["states"],
        tuple(refined_actions),
        tuple(speed_profile),
        current["edge_times"],
        current["cost"],
        tuple(sorted(diagnostics.items())),
    ), diagnostics


def optimize_route_speed_profile(
    route_id,
    actions,
    goal,
    speed_levels_m_s,
    rigid_body_model,
    collision_checker,
    terminal_speed_tolerance_m_s,
    terminal_deceleration_m_s2,
    step_m=0.05,
    beam_width=512,
    initial_mode="coordinated_counterphase",
    goal_position_tolerance_m=0.030,
    goal_heading_tolerance_rad=math.radians(2.0),
    dynamics=FULL_DYNAMICS,
    terminal_curvature_refinement_edges=12,
    terminal_mode_candidates=None,
    full_curvature_refinement=False,
    rejection_samples=None,
    maximum_rejection_samples=512,
):
    """Optimize speed states along one spatial route candidate.

    A route is rejected if no speed sequence satisfies all edge constraints.
    The admissible stopping inequality is applied to every successor using the
    remaining route arc length, so an edge that enters the terminal region too
    fast never enters the dynamic-programming frontier.
    """
    speeds = tuple(sorted(set(float(value) for value in speed_levels_m_s)))
    if not speeds or speeds[0] <= 0.0:
        raise ValueError("speed_levels_m_s must contain positive values")
    if not actions:
        raise ValueError("actions must not be empty")
    if terminal_speed_tolerance_m_s < 0.0:
        raise ValueError("terminal speed tolerance must be nonnegative")
    if terminal_deceleration_m_s2 <= 0.0:
        raise ValueError("terminal deceleration must be positive")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    start = State(0.0, 0.0, 0.0, initial_mode, 0.0, 0.0, False, 0)
    # Build a speed-consistent spatial reference used only to rank beam states.
    # The previous implementation propagated this reference at a hard-coded
    # 0.03 m/s while successors could use 0.08 m/s. Steering-rate-limited
    # geometry is speed dependent, so that mismatch pruned the very states the
    # speed optimizer was intended to evaluate.
    reference_states = []
    reference_speed_profile = []
    reference_state = start
    for action_index, action in enumerate(actions):
        remaining_arc_m = (len(actions) - action_index - 1) * step_m
        maximum_stoppable_speed = math.sqrt(
            terminal_speed_tolerance_m_s**2
            + 2.0 * terminal_deceleration_m_s2 * remaining_arc_m
        )
        stoppable_levels = tuple(
            speed for speed in speeds
            if speed <= maximum_stoppable_speed + 1.0e-12
        )
        reference_speed = (
            max(stoppable_levels) if stoppable_levels else min(speeds)
        )
        reference_state, _, _ = propagate(
            reference_state,
            action["mode"] if isinstance(action, dict) else action[0],
            (
                float(action["curvature_1_m"])
                if isinstance(action, dict) else float(action[1])
            ),
            int(action["direction"] if isinstance(action, dict) else action[2]),
            step_m=step_m,
            speed_m_s=max(speeds),
            dynamics=dynamics,
            rigid_body_model=rigid_body_model,
            target_speed_m_s=reference_speed,
        )
        reference_states.append(reference_state)
        reference_speed_profile.append(reference_speed)

    # node = (cost, state, states, selected_actions, speeds, edge_times)
    frontier = [(0.0, start, (), (), (), ())]
    rejected = {
        "generated": 0,
        "rigid_body": 0,
        "friction": 0,
        "support_load": 0,
        "ltr": 0,
        "stopping_reachability": 0,
        "collision": 0,
        "dominated": 0,
        "beam_pruned": 0,
        "curvature_refinement_generated": 0,
        "reference_speed_profile_m_s": reference_speed_profile,
    }

    def record_rejection(reason, action_index, state, new_state, target_speed):
        if rejection_samples is None or len(rejection_samples) >= maximum_rejection_samples:
            return
        rejection_samples.append(
            {
                "reason": reason,
                "action_index": action_index,
                "start_x_m": state.x,
                "start_y_m": state.y,
                "start_yaw_rad": state.yaw,
                "end_x_m": new_state.x,
                "end_y_m": new_state.y,
                "end_yaw_rad": new_state.yaw,
                "target_speed_m_s": target_speed,
            }
        )

    for action_index, action in enumerate(actions):
        mode = action["mode"] if isinstance(action, dict) else action[0]
        curvature = (
            float(action["curvature_1_m"])
            if isinstance(action, dict)
            else float(action[1])
        )
        direction = (
            int(action["direction"])
            if isinstance(action, dict)
            else int(action[2])
        )
        remaining_arc_m = (len(actions) - action_index - 1) * step_m
        successors = {}
        terminal_refinement = (
            action_index >= len(actions) - terminal_curvature_refinement_edges
        )
        reference_curvature_index = min(
            range(len(CURVATURES)),
            key=lambda index: abs(CURVATURES[index] - curvature),
        )
        if terminal_refinement and full_curvature_refinement:
            curvature_candidates = CURVATURES
        elif terminal_refinement:
            curvature_candidates = tuple(
                CURVATURES[index]
                for index in range(
                    max(0, reference_curvature_index - 1),
                    min(len(CURVATURES), reference_curvature_index + 2),
                )
            )
        else:
            curvature_candidates = (curvature,)
        mode_candidates = (
            tuple(terminal_mode_candidates)
            if terminal_refinement and terminal_mode_candidates is not None
            else (mode,)
        )
        for (
            cost,
            state,
            state_chain,
            action_chain,
            speed_chain,
            time_chain,
        ) in frontier:
            for selected_mode in mode_candidates:
                for selected_curvature in curvature_candidates:
                    if selected_curvature != curvature:
                        rejected["curvature_refinement_generated"] += len(speeds)
                    for target_speed in speeds:
                        rejected["generated"] += 1
                        new_state, poses, edge_time = propagate(
                            state,
                            selected_mode,
                            selected_curvature,
                            direction,
                            step_m=step_m,
                            speed_m_s=max(speeds),
                            dynamics=dynamics,
                            rigid_body_model=rigid_body_model,
                            target_speed_m_s=target_speed,
                        )
                        if not new_state.rigid_body_feasible:
                            rejected["rigid_body"] += 1
                            reason = "rigid_body"
                            bounds = rigid_body_model.bounds
                            if new_state.friction_utilization > bounds.max_friction_utilization:
                                rejected["friction"] += 1
                                reason = "friction"
                            if new_state.minimum_normal_load_fraction <= 0.0:
                                rejected["support_load"] += 1
                                reason = "support_load"
                            if new_state.abs_ltr > bounds.max_abs_ltr:
                                rejected["ltr"] += 1
                                reason = "ltr"
                            record_rejection(
                                reason,
                                action_index,
                                state,
                                new_state,
                                target_speed,
                            )
                            continue
                        maximum_stoppable_speed = math.sqrt(
                            terminal_speed_tolerance_m_s**2
                            + 2.0
                            * terminal_deceleration_m_s2
                            * remaining_arc_m
                        )
                        body_speed = math.hypot(
                            new_state.longitudinal_velocity,
                            new_state.lateral_velocity,
                        )
                        if body_speed > maximum_stoppable_speed + 1.0e-12:
                            rejected["stopping_reachability"] += 1
                            record_rejection(
                                "stopping_reachability",
                                action_index,
                                state,
                                new_state,
                                target_speed,
                            )
                            continue
                        if collision_checker(poses):
                            rejected["collision"] += 1
                            record_rejection(
                                "collision",
                                action_index,
                                state,
                                new_state,
                                target_speed,
                            )
                            continue
                        # Time is the principal objective.  The rigid model
                        # adds normalized friction/LTR/actuator penalties.
                        previous_speed = (
                            abs(state.longitudinal_velocity)
                            if state.started
                            else 0.0
                        )
                        new_cost = (
                            cost
                            + edge_time
                            + new_state.rigid_body_penalty_s
                            + 0.05 * abs(target_speed - previous_speed)
                        )
                        key = _state_key(new_state)
                        incumbent = successors.get(key)
                        selected_action = {
                            "mode": selected_mode,
                            "curvature_1_m": selected_curvature,
                            "direction": direction,
                        }
                        candidate = (
                            new_cost,
                            new_state,
                            state_chain + (new_state,),
                            action_chain + (selected_action,),
                            speed_chain + (target_speed,),
                            time_chain + (edge_time,),
                        )
                        if incumbent is not None and incumbent[0] <= new_cost:
                            rejected["dominated"] += 1
                            continue
                        if incumbent is not None:
                            rejected["dominated"] += 1
                        successors[key] = candidate
        if not successors:
            return None, rejected
        reference = reference_states[action_index]
        remaining_edges = len(actions) - action_index - 1
        remaining_route_m = remaining_edges * step_m
        progress = (action_index + 1) / len(actions)

        def beam_score(item):
            state = item[1]
            goal_distance = math.hypot(
                state.x - goal[0], state.y - goal[1]
            )
            terminal_heading_error = abs(wrap(state.yaw - goal[2]))
            return (
                item[0]
                + 5.0 * math.hypot(
                    state.x - reference.x,
                    state.y - reference.y,
                )
                + 0.5 * abs(wrap(state.yaw - reference.yaw))
                + 20.0 * max(0.0, goal_distance - remaining_route_m)
                + 500.0 * progress**6 * goal_distance
                + 50.0 * progress**6 * terminal_heading_error
            )

        ranked = sorted(successors.values(), key=beam_score)
        if len(ranked) > beam_width:
            rejected["beam_pruned"] += len(ranked) - beam_width
            ranked = ranked[:beam_width]
        frontier = ranked

    feasible = []
    terminal_rows = []
    for node in frontier:
        _, state, _, _, _, _ = node
        position_error = math.hypot(state.x - goal[0], state.y - goal[1])
        heading_error = abs(wrap(state.yaw - goal[2]))
        terminal_speed = math.hypot(
            state.longitudinal_velocity,
            state.lateral_velocity,
        )
        terminal_rows.append((position_error, heading_error, terminal_speed))
        if (
            position_error <= goal_position_tolerance_m
            and heading_error <= goal_heading_tolerance_rad
            and terminal_speed <= terminal_speed_tolerance_m_s
        ):
            feasible.append(node)
    rejected["terminal_frontier_states"] = len(frontier)
    rejected["terminal_position_pass"] = sum(
        row[0] <= goal_position_tolerance_m for row in terminal_rows
    )
    rejected["terminal_heading_pass"] = sum(
        row[1] <= goal_heading_tolerance_rad for row in terminal_rows
    )
    rejected["terminal_speed_pass"] = sum(
        row[2] <= terminal_speed_tolerance_m_s for row in terminal_rows
    )
    rejected["best_terminal_position_error_m"] = min(
        row[0] for row in terminal_rows
    )
    rejected["best_terminal_heading_error_deg"] = math.degrees(
        min(row[1] for row in terminal_rows)
    )
    rejected["best_terminal_speed_m_s"] = min(
        row[2] for row in terminal_rows
    )
    if not feasible:
        return None, rejected
    cost, _, states, selected_actions, selected_speeds, edge_times = min(
        feasible, key=lambda item: item[0]
    )
    return DynamicRouteResult(
        route_id,
        states,
        selected_actions,
        selected_speeds,
        edge_times,
        cost,
        tuple(sorted(rejected.items())),
    ), rejected


def select_dynamic_route(route_candidates, **kwargs):
    """Evaluate all swept-volume routes and select the least-cost feasible one."""
    results = []
    diagnostics = {}
    for route_id, actions in route_candidates.items():
        result, rejected = optimize_route_speed_profile(
            route_id=route_id,
            actions=actions,
            **kwargs,
        )
        diagnostics[route_id] = rejected
        if result is not None:
            results.append(result)
    if not results:
        return None, diagnostics
    return min(results, key=lambda item: item.cost_s), diagnostics
