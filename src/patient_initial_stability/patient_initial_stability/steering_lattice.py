"""Pure steering-rate-aware lattice planner used by the IJCAS ROS nodes."""

import heapq
import math
from dataclasses import dataclass


BODY_LENGTH_M = 1.63
# Conservative measured plan-view envelope, including rotated passive casters.
# The 0.760 m value is the crossbeam/frame width and is not collision-safe for
# constrained-space planning.
BODY_WIDTH_M = 0.965
WHEELBASE_M = 1.29
FRONT_LENGTH_M = 0.645
REAR_LENGTH_M = 0.645
STEP_M = 0.05
SPEED_M_S = 0.03
# Conservative effective rolling-transition rate.  The earlier 13.732910
# deg/s value came from small-angle screening; the first obstacle-bypass Isaac
# replay exposed an 8.3 deg/s large-angle rear-steering transition under load.
# Use 8 deg/s until the physical HY platform is identified across angle/load.
STEERING_RATE_RAD_S = math.radians(8.0)
# Provisional tracking/zero offsets identified from Isaac ROS joint-state logs
# (commanded angle -> measured angle). Stable samples with a near-zero command
# still show about -1 deg on both axles, so apply this offset to inactive wheels
# as well as to nonzero targets. Re-identify on the HY hardware.
STEERING_TRACKING_BIAS_RAD = {
    "front": math.radians(-1.0),
    "rear": math.radians(-1.0),
}
# Mode-specific yaw-curvature response. The initial values below are calibrated
# from the full-envelope turn-mode gate. General-navigation replay r01 showed
# larger responses under the current stage/control setup; its stable-segment
# medians are now used provisionally and must be checked in fresh repeats.
# The commanded steering geometry is retained for lateral velocity; only the
# achieved yaw curvature is scaled. These remain simulation-identification
# constants and must be reidentified for the physical HY platform.
MODE_CURVATURE_RESPONSE_SCALE = {
    "coordinated_counterphase": 0.9618004068327801,
    # Updated from the stable large-angle front-only segment in
    # general90_bias_prop_r01 (median yaw-rate / measured-angle geometry).
    "front_only": 0.96545,
    # Stable rear-only samples from general90_rolling_prop_r01.  This run is
    # treated as identification data, not as an independent validation run.
    "rear_only": 0.96430,
}
# Mode-specific body-lateral-velocity response, calibrated from stable portions
# of the general-navigation Isaac replays. The geometry term is still derived
# from the steering angles; this factor accounts for the residual difference
# between that ideal model and the simulated platform.
MODE_LATERAL_RESPONSE_SCALE = {
    # Body-frame lateral/longitudinal velocity divided by the measured-angle
    # geometric prediction in stable general90_rolling_prop_r01 samples.
    # Keep the identification run separate from the next validation replay.
    "coordinated_counterphase": 0.92434,
    "front_only": 1.00542,
    "rear_only": 0.99375,
}
CURVATURES = (-0.70, -0.50, -0.25, 0.0, 0.25, 0.50, 0.70)
ALL_MODES = ("coordinated_counterphase", "front_only", "rear_only")
SUBSTEPS = 10
XY_RESOLUTION_M = 0.025
YAW_RESOLUTION_RAD = math.radians(1.0)
STEERING_RESOLUTION_RAD = math.radians(5.0)
GOAL_POSITION_TOLERANCE_M = 0.025
GOAL_HEADING_TOLERANCE_RAD = math.radians(2.0)
MODE_SWITCH_PENALTY_S = 0.50
DIRECTION_SWITCH_PENALTY_S = 1.00
STEERING_TRANSIENT_PENALTY_WEIGHT = 0.50


@dataclass(frozen=True)
class State:
    x: float
    y: float
    yaw: float
    mode: str
    front: float
    rear: float
    started: bool
    direction: int = 0
    longitudinal_velocity: float = 0.0
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    rigid_body_feasible: bool = True
    friction_utilization: float = 0.0
    front_friction_utilization: float = 0.0
    rear_friction_utilization: float = 0.0
    minimum_normal_load_fraction: float = 1.0
    abs_ltr: float = 0.0
    actuator_utilization: float = 0.0
    rigid_body_penalty_s: float = 0.0


@dataclass(frozen=True)
class PlanResult:
    states: tuple
    actions: tuple
    expanded_nodes: int
    cost: float
    rejection_counts: tuple = ()


@dataclass(frozen=True)
class PlannerDynamics:
    """Dynamics fidelity used by a planning ablation.

    ``kinematic`` deliberately removes steering-rate and identified response
    effects.  ``transition_cost`` keeps ideal geometric propagation but
    penalizes time needed to reconfigure the measured steering joints.
    ``full`` propagates steering state at the identified rate and applies the
    mode-specific curvature response measured in Isaac Sim.
    """

    name: str
    rate_limited_steering: bool
    apply_identified_response: bool
    include_mode_switch_cost: bool
    include_steering_transient_cost: bool


KINEMATIC_DYNAMICS = PlannerDynamics(
    "kinematic",
    rate_limited_steering=False,
    apply_identified_response=False,
    include_mode_switch_cost=False,
    include_steering_transient_cost=False,
)
TRANSITION_COST_DYNAMICS = PlannerDynamics(
    "transition_cost",
    rate_limited_steering=False,
    apply_identified_response=False,
    include_mode_switch_cost=True,
    include_steering_transient_cost=True,
)
FULL_DYNAMICS = PlannerDynamics(
    "full",
    rate_limited_steering=True,
    apply_identified_response=True,
    include_mode_switch_cost=True,
    include_steering_transient_cost=True,
)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def move_toward(current, target, maximum_step):
    return current + max(-maximum_step, min(maximum_step, target - current))


def steering_targets(mode, curvature):
    if mode == "front_only":
        return math.atan(WHEELBASE_M * curvature), 0.0
    if mode == "rear_only":
        return 0.0, -math.atan(WHEELBASE_M * curvature)
    if mode == "coordinated_counterphase":
        return (
            math.atan(FRONT_LENGTH_M * curvature),
            -math.atan(REAR_LENGTH_M * curvature),
        )
    raise ValueError(mode)


def _tracked_steering_target(commanded, axle):
    """Approximate measured joint angle, including the near-zero offset."""
    return commanded + STEERING_TRACKING_BIAS_RAD[axle]


def footprint_y_bounds(y, yaw):
    half_extent = (
        abs(math.sin(yaw)) * BODY_LENGTH_M / 2.0
        + abs(math.cos(yaw)) * BODY_WIDTH_M / 2.0
    )
    return y - half_extent, y + half_extent


def propagate(
    state,
    mode,
    curvature,
    direction,
    step_m=STEP_M,
    speed_m_s=SPEED_M_S,
    dynamics=FULL_DYNAMICS,
    response_scale_by_mode=None,
    rigid_body_model=None,
    target_speed_m_s=None,
):
    commanded_speed_m_s = (
        speed_m_s if target_speed_m_s is None else target_speed_m_s
    )
    target_front, target_rear = steering_targets(mode, curvature)
    if dynamics.apply_identified_response:
        target_front = _tracked_steering_target(target_front, "front")
        target_rear = _tracked_steering_target(target_rear, "rear")
    if not state.started:
        front, rear = target_front, target_rear
    else:
        front, rear = state.front, state.rear
    x, y, yaw = state.x, state.y, state.yaw
    body_u = state.longitudinal_velocity
    body_v = state.lateral_velocity
    body_r = state.yaw_rate
    rigid_feasible = True
    maximum_friction_utilization = 0.0
    maximum_front_friction_utilization = 0.0
    maximum_rear_friction_utilization = 0.0
    minimum_normal_load_fraction = 1.0
    maximum_abs_ltr = 0.0
    maximum_actuator_utilization = 0.0
    rigid_penalty_s = 0.0
    if step_m <= 0.0 or speed_m_s <= 0.0 or commanded_speed_m_s <= 0.0:
        raise ValueError("step_m, speed_m_s and target speed must be positive")
    signed_ds = direction * step_m / SUBSTEPS
    nominal_edge_time = step_m / speed_m_s
    # Keep rigid-body integration fine enough for acceleration saturation and
    # steering-rate transients.  Kinematic propagation retains the historical
    # ten equal-distance substeps exactly.
    rigid_dt = min(nominal_edge_time / SUBSTEPS, 0.05)
    response_scales = (
        MODE_CURVATURE_RESPONSE_SCALE
        if response_scale_by_mode is None
        else response_scale_by_mode
    )
    poses = []

    def steering_midpoint(current_front, current_rear, integration_dt):
        if not dynamics.rate_limited_steering:
            return target_front, target_rear
        half_step = STEERING_RATE_RAD_S * integration_dt / 2.0
        return (
            move_toward(current_front, target_front, half_step),
            move_toward(current_rear, target_rear, half_step),
        )

    def steering_finish(middle_front, middle_rear, integration_dt):
        if not dynamics.rate_limited_steering:
            return target_front, target_rear
        half_step = STEERING_RATE_RAD_S * integration_dt / 2.0
        return (
            move_toward(middle_front, target_front, half_step),
            move_toward(middle_rear, target_rear, half_step),
        )

    def geometry(middle_front, middle_rear):
        tan_front = math.tan(middle_front)
        tan_rear = math.tan(middle_rear)
        curvature_actual = (
            (tan_front - tan_rear)
            / WHEELBASE_M
            * (response_scales[mode] if dynamics.apply_identified_response else 1.0)
        )
        lateral_ratio = (
            REAR_LENGTH_M * tan_front + FRONT_LENGTH_M * tan_rear
        ) / WHEELBASE_M * MODE_LATERAL_RESPONSE_SCALE[mode]
        return curvature_actual, lateral_ratio

    if rigid_body_model is None:
        integration_dt = nominal_edge_time / SUBSTEPS
        for _ in range(SUBSTEPS):
            middle_front, middle_rear = steering_midpoint(
                front, rear, integration_dt
            )
            curvature_actual, lateral_ratio = geometry(middle_front, middle_rear)
            x += (math.cos(yaw) - lateral_ratio * math.sin(yaw)) * signed_ds
            y += (math.sin(yaw) + lateral_ratio * math.cos(yaw)) * signed_ds
            yaw += curvature_actual * signed_ds
            poses.append((x, y, yaw))
            front, rear = steering_finish(
                middle_front, middle_rear, integration_dt
            )
        elapsed_time = nominal_edge_time
    else:
        # A motion primitive is defined by travelled longitudinal distance,
        # not a fixed time.  This matters at startup: with acceleration limits,
        # integrating only step/speed seconds silently produces a short edge.
        travelled_m = 0.0
        elapsed_time = 0.0
        maximum_edge_time = max(
            10.0 * nominal_edge_time,
            2.0 * step_m / commanded_speed_m_s,
            2.0 * math.sqrt(2.0 * step_m / rigid_body_model.bounds.max_longitudinal_accel_m_s2),
        )
        while travelled_m < step_m - 1.0e-12:
            if elapsed_time >= maximum_edge_time:
                rigid_feasible = False
                break
            integration_dt = min(rigid_dt, maximum_edge_time - elapsed_time)
            middle_front, middle_rear = steering_midpoint(
                front, rear, integration_dt
            )
            curvature_actual, lateral_ratio = geometry(middle_front, middle_rear)
            target_u = direction * commanded_speed_m_s
            target_v = target_u * lateral_ratio
            target_r = target_u * curvature_actual
            old_u, old_v, old_r = body_u, body_v, body_r
            rigid_step = rigid_body_model.step(
                old_u,
                old_v,
                old_r,
                target_u,
                target_v,
                target_r,
                integration_dt,
            )
            average_u = 0.5 * (
                old_u + rigid_step.longitudinal_velocity_m_s
            )
            distance_increment = abs(average_u) * integration_dt
            remaining_m = step_m - travelled_m
            if distance_increment > remaining_m and distance_increment > 0.0:
                integration_dt *= remaining_m / distance_increment
                middle_front, middle_rear = steering_midpoint(
                    front, rear, integration_dt
                )
                curvature_actual, lateral_ratio = geometry(
                    middle_front, middle_rear
                )
                target_v = target_u * lateral_ratio
                target_r = target_u * curvature_actual
                rigid_step = rigid_body_model.step(
                    old_u,
                    old_v,
                    old_r,
                    target_u,
                    target_v,
                    target_r,
                    integration_dt,
                )
                average_u = 0.5 * (
                    old_u + rigid_step.longitudinal_velocity_m_s
                )
                distance_increment = remaining_m
            body_u = rigid_step.longitudinal_velocity_m_s
            body_v = rigid_step.lateral_velocity_m_s
            body_r = rigid_step.yaw_rate_rad_s
            rigid_feasible = rigid_feasible and rigid_step.feasible
            maximum_friction_utilization = max(
                maximum_friction_utilization,
                rigid_step.friction_utilization,
            )
            maximum_front_friction_utilization = max(
                maximum_front_friction_utilization,
                rigid_step.front_friction_utilization,
            )
            maximum_rear_friction_utilization = max(
                maximum_rear_friction_utilization,
                rigid_step.rear_friction_utilization,
            )
            minimum_normal_load_fraction = min(
                minimum_normal_load_fraction,
                rigid_step.minimum_normal_load_fraction,
            )
            maximum_abs_ltr = max(maximum_abs_ltr, rigid_step.abs_ltr)
            maximum_actuator_utilization = max(
                maximum_actuator_utilization,
                rigid_step.actuator_utilization,
            )
            rigid_penalty_s += rigid_step.penalty_s
            average_v = 0.5 * (old_v + body_v)
            average_r = 0.5 * (old_r + body_r)
            x += (
                average_u * math.cos(yaw) - average_v * math.sin(yaw)
            ) * integration_dt
            y += (
                average_u * math.sin(yaw) + average_v * math.cos(yaw)
            ) * integration_dt
            yaw += average_r * integration_dt
            travelled_m += distance_increment
            elapsed_time += integration_dt
            poses.append((x, y, yaw))
            front, rear = steering_finish(
                middle_front, middle_rear, integration_dt
            )
    return State(
        x,
        y,
        yaw,
        mode,
        front,
        rear,
        True,
        direction,
        body_u,
        body_v,
        body_r,
        rigid_feasible,
        maximum_friction_utilization,
        maximum_front_friction_utilization,
        maximum_rear_friction_utilization,
        minimum_normal_load_fraction,
        maximum_abs_ltr,
        maximum_actuator_utilization,
        rigid_penalty_s,
    ), poses, elapsed_time


def _collides(poses, minimum_y, maximum_y, robust_clearance):
    safe_minimum = minimum_y + robust_clearance
    safe_maximum = maximum_y - robust_clearance
    return any(
        footprint_y_bounds(y, yaw)[0] < safe_minimum
        or footprint_y_bounds(y, yaw)[1] > safe_maximum
        for _, y, yaw in poses
    )


def _state_key(state, include_rigid_body=False):
    key = (
        round(state.x / XY_RESOLUTION_M),
        round(state.y / XY_RESOLUTION_M),
        round(state.yaw / YAW_RESOLUTION_RAD),
        state.mode,
        round(state.front / STEERING_RESOLUTION_RAD),
        round(state.rear / STEERING_RESOLUTION_RAD),
        state.started,
        state.direction,
    )
    if include_rigid_body:
        key += (
            round(state.longitudinal_velocity / 0.01),
            round(state.lateral_velocity / 0.01),
            round(state.yaw_rate / math.radians(1.0)),
        )
    return key


def _at_goal(state, goal):
    return (
        math.hypot(state.x - goal[0], state.y - goal[1])
        <= GOAL_POSITION_TOLERANCE_M
        and abs(wrap(state.yaw - goal[2])) <= GOAL_HEADING_TOLERANCE_RAD
    )


def _heuristic(state, goal, speed_m_s):
    position_time = math.hypot(state.x - goal[0], state.y - goal[1]) / speed_m_s
    heading_time = abs(wrap(state.yaw - goal[2])) / (0.70 * speed_m_s)
    return max(position_time, heading_time)


def plan(
    goal,
    minimum_y,
    maximum_y,
    robust_clearance=0.010,
    allowed_modes=ALL_MODES,
    allowed_curvatures=CURVATURES,
    speed_m_s=SPEED_M_S,
    node_limit=800000,
    collision_checker=None,
    initial_mode="coordinated_counterphase",
    initial_front_rad=0.0,
    initial_rear_rad=0.0,
    propagate_initial_steering=False,
    directions=(1,),
    minimum_x=-0.4,
    maximum_x=1.5,
    dynamics=FULL_DYNAMICS,
    response_scale_by_mode=None,
    goal_heading_tolerance_rad=GOAL_HEADING_TOLERANCE_RAD,
    heuristic_weight=1.0,
    heuristic_function=None,
    rigid_body_model=None,
    speed_levels_m_s=None,
    terminal_speed_tolerance_m_s=None,
    terminal_deceleration_m_s2=None,
    speed_level_selector=None,
    mode_selector=None,
    curvature_selector=None,
    diagnostics=None,
):
    """Plan from a local zero pose while propagating physical steering state."""
    if initial_mode not in ALL_MODES:
        raise ValueError(f"unsupported initial_mode: {initial_mode}")
    allowed_curvatures = tuple(allowed_curvatures)
    if (
        not allowed_curvatures
        or any(curvature not in CURVATURES for curvature in allowed_curvatures)
    ):
        raise ValueError("allowed_curvatures must be a nonempty CURVATURES subset")
    if speed_levels_m_s is None:
        speed_levels = (speed_m_s,)
    else:
        speed_levels = tuple(sorted(set(speed_levels_m_s)))
        if not speed_levels or any(value <= 0.0 for value in speed_levels):
            raise ValueError("speed_levels_m_s must contain positive speeds")
        if rigid_body_model is None:
            raise ValueError("speed-level planning requires rigid_body_model")
    if terminal_speed_tolerance_m_s is not None:
        if terminal_speed_tolerance_m_s < 0.0:
            raise ValueError("terminal speed tolerance must be nonnegative")
        if rigid_body_model is None:
            raise ValueError("terminal speed constraint requires rigid_body_model")
    if terminal_deceleration_m_s2 is not None:
        if terminal_deceleration_m_s2 <= 0.0:
            raise ValueError("terminal deceleration must be positive")
        if terminal_speed_tolerance_m_s is None:
            raise ValueError(
                "terminal deceleration requires terminal_speed_tolerance_m_s"
            )
    directions = tuple(directions)
    if not directions or any(direction not in (-1, 1) for direction in directions):
        raise ValueError("directions must contain only -1 and/or +1")
    start = State(
        0.0,
        0.0,
        0.0,
        initial_mode,
        initial_front_rad,
        initial_rear_rad,
        propagate_initial_steering,
        0,
    )
    nodes = [{"state": start, "parent": None, "action": None, "cost": 0.0}]
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be at least 1.0")
    def estimate(state):
        if heuristic_function is not None:
            return heuristic_function(state)
        return _heuristic(state, goal, speed_m_s)

    queue = [(heuristic_weight * estimate(start), 0, 0)]
    use_rigid_body = rigid_body_model is not None
    best = {_state_key(start, use_rigid_body): 0.0}
    rejection_counts = {
        "generated": 0,
        "rigid_body": 0,
        "friction": 0,
        "support_load": 0,
        "ltr": 0,
        "bounds": 0,
        "stopping_reachability": 0,
        "collision": 0,
        "dominated": 0,
    }

    best_goal_state = start
    best_goal_position_error = math.hypot(start.x - goal[0], start.y - goal[1])

    def publish_diagnostics():
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(rejection_counts)
            diagnostics["expanded_nodes"] = expanded
            diagnostics["best_goal_position_error_m"] = (
                best_goal_position_error
            )
            diagnostics["best_goal_heading_error_deg"] = math.degrees(
                wrap(best_goal_state.yaw - goal[2])
            )
            diagnostics["best_goal_speed_m_s"] = math.hypot(
                best_goal_state.longitudinal_velocity,
                best_goal_state.lateral_velocity,
            )
    expanded = counter = 0
    while queue and expanded < node_limit:
        _, _, node_id = heapq.heappop(queue)
        node = nodes[node_id]
        state, cost = node["state"], node["cost"]
        if cost > best.get(_state_key(state, use_rigid_body), math.inf) + 1.0e-12:
            continue
        expanded += 1
        goal_position_error = math.hypot(
            state.x - goal[0], state.y - goal[1]
        )
        if goal_position_error < best_goal_position_error:
            best_goal_position_error = goal_position_error
            best_goal_state = state
        terminal_speed_ok = (
            terminal_speed_tolerance_m_s is None
            or math.hypot(
                state.longitudinal_velocity,
                state.lateral_velocity,
            ) <= terminal_speed_tolerance_m_s
        )
        if state.started and terminal_speed_ok and (
            math.hypot(state.x - goal[0], state.y - goal[1])
            <= GOAL_POSITION_TOLERANCE_M
            and abs(wrap(state.yaw - goal[2])) <= goal_heading_tolerance_rad
        ):
            chain = []
            while nodes[node_id]["parent"] is not None:
                current = nodes[node_id]
                chain.append((current["state"], current["action"]))
                node_id = current["parent"]
            chain.reverse()
            publish_diagnostics()
            return PlanResult(
                tuple(state for state, _ in chain),
                tuple(action for _, action in chain),
                expanded,
                cost,
                tuple(sorted(rejection_counts.items())),
            )
        candidate_modes = (
            tuple(mode_selector(state, allowed_modes))
            if mode_selector is not None
            else tuple(allowed_modes)
        )
        candidate_curvatures = (
            tuple(curvature_selector(state, allowed_curvatures))
            if curvature_selector is not None
            else allowed_curvatures
        )
        if not candidate_modes or any(
            mode not in ALL_MODES for mode in candidate_modes
        ):
            raise ValueError(
                "mode_selector must return supported steering modes"
            )
        if not candidate_curvatures or any(
            curvature not in CURVATURES
            for curvature in candidate_curvatures
        ):
            raise ValueError(
                "curvature_selector must return configured curvatures"
            )
        for mode in candidate_modes:
            for curvature in candidate_curvatures:
                for direction in directions:
                    candidate_speeds = (
                        tuple(speed_level_selector(state, speed_levels))
                        if speed_level_selector is not None
                        else speed_levels
                    )
                    if not candidate_speeds or any(
                        value not in speed_levels for value in candidate_speeds
                    ):
                        raise ValueError(
                            "speed_level_selector must return configured speeds"
                        )
                    for target_speed in candidate_speeds:
                        rejection_counts["generated"] += 1
                        new_state, poses, transition_time = propagate(
                            state,
                            mode,
                            curvature,
                            direction,
                            speed_m_s=speed_m_s,
                            dynamics=dynamics,
                            response_scale_by_mode=response_scale_by_mode,
                            rigid_body_model=rigid_body_model,
                            target_speed_m_s=target_speed,
                        )
                        if not new_state.rigid_body_feasible:
                            rejection_counts["rigid_body"] += 1
                            if (
                                rigid_body_model is not None
                                and new_state.friction_utilization
                                > rigid_body_model.bounds.max_friction_utilization
                            ):
                                rejection_counts["friction"] += 1
                            if new_state.minimum_normal_load_fraction <= 0.0:
                                rejection_counts["support_load"] += 1
                            if (
                                rigid_body_model is not None
                                and new_state.abs_ltr
                                > rigid_body_model.bounds.max_abs_ltr
                            ):
                                rejection_counts["ltr"] += 1
                            continue
                        if new_state.x < minimum_x or new_state.x > maximum_x:
                            rejection_counts["bounds"] += 1
                            continue
                        if terminal_deceleration_m_s2 is not None:
                            remaining_distance = max(
                                0.0,
                                math.hypot(
                                    new_state.x - goal[0],
                                    new_state.y - goal[1],
                                ) - GOAL_POSITION_TOLERANCE_M,
                            )
                            maximum_stoppable_speed = math.sqrt(
                                terminal_speed_tolerance_m_s**2
                                + 2.0
                                * terminal_deceleration_m_s2
                                * remaining_distance
                            )
                            body_speed = math.hypot(
                                new_state.longitudinal_velocity,
                                new_state.lateral_velocity,
                            )
                            if body_speed > maximum_stoppable_speed + 1.0e-12:
                                rejection_counts["stopping_reachability"] += 1
                                continue
                        collision = (
                            collision_checker(poses)
                            if collision_checker is not None
                            else _collides(poses, minimum_y, maximum_y, robust_clearance)
                        )
                        if collision:
                            rejection_counts["collision"] += 1
                            continue
                        new_key = _state_key(new_state, use_rigid_body)
                        switch_penalty = (
                            MODE_SWITCH_PENALTY_S
                            if state.started and mode != state.mode
                            and dynamics.include_mode_switch_cost
                            else 0.0
                        )
                        reversal_penalty = (
                            DIRECTION_SWITCH_PENALTY_S
                            if state.started
                            and state.direction != 0
                            and direction != state.direction
                            else 0.0
                        )
                        target_front, target_rear = steering_targets(mode, curvature)
                        transient_penalty = (
                            STEERING_TRANSIENT_PENALTY_WEIGHT
                            * max(abs(target_front - state.front), abs(target_rear - state.rear))
                            / STEERING_RATE_RAD_S
                            if state.started and dynamics.include_steering_transient_cost
                            else 0.0
                        )
                        new_cost = (
                            cost + transition_time + switch_penalty + reversal_penalty
                            + transient_penalty + 0.02 * abs(curvature)
                            + new_state.rigid_body_penalty_s
                        )
                        if new_cost >= best.get(new_key, math.inf):
                            rejection_counts["dominated"] += 1
                            continue
                        best[new_key] = new_cost
                        action = (
                            (mode, curvature, direction)
                            if speed_levels_m_s is None
                            else (mode, curvature, direction, target_speed)
                        )
                        nodes.append(
                            {
                                "state": new_state,
                                "parent": node_id,
                                "action": action,
                                "cost": new_cost,
                            }
                        )
                        counter += 1
                        new_id = len(nodes) - 1
                        heapq.heappush(
                            queue,
                            (
                                new_cost
                                + heuristic_weight * estimate(new_state),
                                counter,
                                new_id,
                            ),
                        )
    publish_diagnostics()
    return None
