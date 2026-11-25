import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .steering_lattice import (
    MODE_CURVATURE_RESPONSE_SCALE,
    MODE_LATERAL_RESPONSE_SCALE,
)
from .steering_lattice import plan as lattice_plan


BODY_LENGTH_M = 1.63
# Conservative measured plan-view envelope, including rotated passive casters.
BODY_WIDTH_M = 0.965
WHEELBASE_M = 1.29
FRONT_LENGTH_M = 0.645
REAR_LENGTH_M = 0.645
STEERING_RATE_DEG_S = 8.0
STEP_M = 0.05
GOAL_POSITION_TOLERANCE_M = 0.025
GOAL_HEADING_TOLERANCE_DEG = 2.0


def apply_terminal_minimum_drive_command(
    command_speed,
    longitudinal_error,
    position_error,
    position_tolerance,
    longitudinal_velocity,
    stopped_speed_threshold,
    minimum_speed,
):
    """Apply breakaway speed only for a genuine terminal position error.

    Heading error alone must not force a signed longitudinal pulse after the
    robot has entered the terminal position band. That pulse changes sign as
    the pose crosses the goal and can create a terminal limit cycle.
    """
    if (
        position_error <= position_tolerance
        or abs(longitudinal_velocity) >= 0.5 * stopped_speed_threshold
        or abs(command_speed) >= minimum_speed
    ):
        return command_speed
    return math.copysign(
        minimum_speed,
        command_speed if abs(command_speed) > 1.0e-9 else longitudinal_error,
    )


def terminal_pose_errors(dx, dy, yaw, goal_yaw):
    """Resolve the goal residual in the current body frame."""
    return (
        math.cos(yaw) * dx + math.sin(yaw) * dy,
        -math.sin(yaw) * dx + math.cos(yaw) * dy,
        wrap(goal_yaw - yaw),
    )


def terminal_correction_direction(
    longitudinal_error,
    previous_direction,
    switch_threshold,
):
    """Select forward/reverse motion without chattering at zero error."""
    if switch_threshold < 0.0:
        raise ValueError("switch_threshold must be nonnegative")
    if previous_direction not in (-1, 0, 1):
        raise ValueError("previous_direction must be -1, 0, or 1")
    if previous_direction == 0:
        return 1 if longitudinal_error >= 0.0 else -1
    if previous_direction * longitudinal_error < -switch_threshold:
        return -previous_direction
    return previous_direction


def terminal_lateral_correction(
    lateral_error,
    heading_error,
    direction,
    previous_mode,
    heading_hysteresis,
    curvature_limit=0.70,
):
    """Choose a bounded one-axle arc that reduces body-frame lateral error.

    Front-only and rear-only steering have opposite body-center sideslip for
    the same yaw curvature.  Selecting between them lets the controller move
    laterally while choosing the yaw-change sign that also reduces heading
    error.  The previous mode is retained through a small heading hysteresis
    so steering-rate lag cannot cause rapid front/rear mode chatter.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if heading_hysteresis < 0.0 or curvature_limit <= 0.0:
        raise ValueError("invalid terminal lateral-correction bounds")
    lateral_sign = 1 if lateral_error >= 0.0 else -1
    signed_heading_need = heading_error * lateral_sign
    if previous_mode == "front_only" and signed_heading_need >= -heading_hysteresis:
        mode = "front_only"
    elif previous_mode == "rear_only" and signed_heading_need <= heading_hysteresis:
        mode = "rear_only"
    else:
        mode = "front_only" if signed_heading_need >= 0.0 else "rear_only"

    # With signed longitudinal travel ds, the requested body-lateral motion
    # has sign(ds * lateral_ratio). Front-only has lateral_ratio ~ +curvature,
    # whereas rear-only has lateral_ratio ~ -curvature.
    lateral_ratio_sign = direction * lateral_sign
    curvature = curvature_limit * lateral_ratio_sign
    if mode == "rear_only":
        curvature = -curvature
    return mode, curvature

# Dynamically replay the immutable paths produced by the steering-rate-aware
# lattice in studies/IJCAS/scripts/screen_steering_rate_lattice_spatial_utility.py.
SCENARIOS = {
    "rear_end_constrained": {
        "minimum_y": -0.40,
        "maximum_y": 1.05,
        "goal": (0.6451846534, 0.3411371489, math.radians(20.0)),
        "terminal_command_distance_m": 0.05,
        "actions": [
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
            ("front_only", 0.50, 1),
        ],
    },
    "front_end_constrained": {
        "minimum_y": -0.80,
        "maximum_y": 0.55,
        "goal": (0.7229260505, -0.1000785532, math.radians(20.0)),
        "terminal_command_distance_m": 0.05,
        "actions": [
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
            ("rear_only", 0.50, 1),
        ],
    },
}

# Full-envelope local-obstacle tasks.  In topic-planning mode the planner
# replaces these seed actions and the goal from the shared scenario map.  The
# seed actions retain deterministic replay/debug behavior.
SCENARIOS.update(
    {
        "rear_end_local_wall": {
            "minimum_y": -2.0,
            "maximum_y": 2.0,
            "goal": (0.6451846534, 0.3411371489, math.radians(20.0)),
            "terminal_command_distance_m": 0.05,
            "actions": [("front_only", 0.50, 1)] * 14,
        },
        "front_end_local_wall": {
            "minimum_y": -2.0,
            "maximum_y": 2.0,
            "goal": (0.7229260505, -0.1000785532, math.radians(20.0)),
            "terminal_command_distance_m": 0.05,
            "actions": [("rear_only", 0.50, 1)] * 14,
        },
    }
)

# Policy-independent operational development task.  Its goal and obstacle were
# fixed from the bedside lateral-alignment requirement before the policies
# were evaluated.  The action sequence is the 100,000-node mixed-mode result:
# each fixed configuration exhausted the same budget without finding a path.
SCENARIOS["bedside_lateral_alignment"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (1.20, 0.38, 0.0),
    "terminal_command_distance_m": 0.05,
    "actions": (
        [("front_only", 0.50, 1)] * 6
        + [("front_only", 0.25, 1)] * 3
        + [("rear_only", -0.25, 1)] * 16
    ),
}

# General-navigation dynamics-ablation tasks. Unlike the bedside development
# case, these represent recurring navigation patterns rather than docking.
# Actions are the calibrated 30,000-node full-dynamics solutions frozen by
# screen_general_navigation_dynamics_ablation.py. The 90-degree corner path
# includes both curvature and body-lateral-response calibration from Isaac r01/r02.
SCENARIOS["corridor_90_turn"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (1.50, 2.20, math.radians(90.0)),
    "terminal_command_distance_m": 0.05,
    "actions": (
        [("front_only", 0.25, 1)]
        + [("front_only", 0.50, 1)]
        + [("front_only", 0.70, 1)] * 33
        + [("front_only", 0.50, 1)]
        + [("rear_only", 0.50, 1)] * 12
        + [("rear_only", 0.70, 1)] * 4
    ),
}
SCENARIOS["narrow_t_intersection"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (1.10, 2.15, math.radians(90.0)),
    "terminal_command_distance_m": 0.05,
    "actions": (
        [("front_only", 0.25, 1)]
        + [("front_only", 0.50, 1)]
        + [("front_only", 0.70, 1)] * 38
        + [("coordinated_counterphase", 0.70, 1)] * 8
        + [("coordinated_counterphase", 0.50, 1)]
    ),
}
SCENARIOS["doorway_immediate_turn"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (2.00, 1.30, math.radians(65.0)),
    "terminal_command_distance_m": 0.05,
    "actions": (
        [("coordinated_counterphase", 0.25, 1)]
        + [("coordinated_counterphase", 0.50, 1)]
        + [("coordinated_counterphase", 0.70, 1)] * 11
        + [("coordinated_counterphase", 0.50, 1)] * 19
        + [("rear_only", 0.50, 1)] * 6
        + [("rear_only", 0.25, 1)] * 11
    ),
}
SCENARIOS["s_corridor"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (5.80, 0.0, 0.0),
    "terminal_command_distance_m": 0.05,
    "actions": (
        [("coordinated_counterphase", 0.0, 1)]
        + [("coordinated_counterphase", -0.25, 1)] * 4
        + [("coordinated_counterphase", -0.50, 1)] * 6
        + [("coordinated_counterphase", -0.25, 1)] * 10
        + [("coordinated_counterphase", 0.0, 1)] * 4
        + [("coordinated_counterphase", 0.25, 1)]
        + [("coordinated_counterphase", 0.50, 1)] * 23
        + [("coordinated_counterphase", 0.25, 1)] * 13
        + [("coordinated_counterphase", 0.0, 1)]
        + [("coordinated_counterphase", -0.25, 1)] * 17
        + [("coordinated_counterphase", -0.50, 1)] * 13
        + [("coordinated_counterphase", -0.25, 1)] * 4
        + [("coordinated_counterphase", 0.0, 1)] * 14
        + [("coordinated_counterphase", 0.25, 1)]
        + [("rear_only", 0.25, 1)] * 3
        + [("rear_only", 0.50, 1)] * 2
        + [("rear_only", 0.70, 1)] * 2
    ),
}
SCENARIOS["obstacle_bypass_return"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (4.50, 0.0, 0.0),
    "terminal_command_distance_m": 0.05,
    "actions": (
        [("coordinated_counterphase", 0.0, 1)]
        + [("coordinated_counterphase", -0.25, 1)] * 4
        + [("coordinated_counterphase", -0.50, 1)] * 14
        + [("coordinated_counterphase", -0.25, 1)] * 10
        + [("coordinated_counterphase", 0.0, 1)] * 10
        + [("coordinated_counterphase", 0.25, 1)]
        + [("coordinated_counterphase", 0.50, 1)] * 2
        + [("coordinated_counterphase", 0.70, 1)] * 25
        + [("coordinated_counterphase", 0.50, 1)] * 2
        + [("coordinated_counterphase", 0.25, 1)] * 3
        + [("coordinated_counterphase", 0.0, 1)] * 5
        + [("coordinated_counterphase", -0.25, 1)] * 3
        + [("coordinated_counterphase", -0.50, 1)] * 5
        + [("rear_only", -0.50, 1)] * 3
        + [("rear_only", -0.70, 1)] * 7
        + [("rear_only", -0.50, 1)]
    ),
}

# Longitudinal rigid-body activation gate.  The matching planner experiment
# proves that a fixed 0.08 m/s state cannot satisfy the conservative 0.02 m/s²
# stopping envelope over 0.30 m, while joint path/speed planning selects the
# final three 0.01 m/s states and reaches a stop-feasible terminal state.
SCENARIOS["rigid_body_stopping_gate"] = {
    "minimum_y": -3.0,
    "maximum_y": 3.0,
    "goal": (0.30, 0.0, 0.0),
    "terminal_command_distance_m": 0.05,
    "actions": [("coordinated_counterphase", 0.0, 1)] * 6,
}

# Frozen 8 deg/s planning-model paths used for paired robustness comparisons.
# The default scenario paths above are retained so earlier run IDs remain
# exactly reproducible.  Selecting one of these policies is explicit.
REPLAY_POLICY_ACTIONS = {
    ("corridor_90_turn", "fixed_counterphase"): (
        # First steering target is aligned before the timed task begins.
        [("coordinated_counterphase", 0.70, 1)] * 46
        + [("coordinated_counterphase", 0.50, 1)]
        + [("coordinated_counterphase", 0.25, 1)] * 3
        + [("coordinated_counterphase", 0.0, 1)] * 9
        + [("coordinated_counterphase", -0.25, 1)]
        + [("coordinated_counterphase", -0.50, 1)] * 2
    ),
    ("corridor_90_turn", "proposed_full_dynamics"): (
        # Initial steering is pre-aligned and the two mode changes are executed
        # as rate-limited rolling transitions, matching lattice propagation.
        # Lateral-response factors are identified from rolling_prop_r01; this
        # frozen path is first validated independently in rolling_prop_r02.
        [("front_only", 0.70, 1)] * 28
        + [("front_only", 0.50, 1)] * 5
        + [("coordinated_counterphase", 0.50, 1)] * 7
        + [("rear_only", 0.50, 1)]
        + [("rear_only", 0.70, 1)] * 9
        + [("rear_only", 0.50, 1)]
    ),
    ("corridor_90_turn", "proposed_rigid_body_robust"): (
        # Replanned with the joint worst-case envelope from nominal 265 kg,
        # low friction, front-CoM 290 kg and rear-CoM 240 kg simulation cells.
        # The final primitive differs from the earlier frozen solution because
        # u/v/r propagation and axle-wise friction cost are inside the search.
        [("front_only", 0.70, 1)] * 28
        + [("front_only", 0.50, 1)] * 5
        + [("coordinated_counterphase", 0.50, 1)] * 7
        + [("rear_only", 0.50, 1)]
        + [("rear_only", 0.70, 1)] * 10
    ),
    ("corridor_90_turn", "proposed_rigid_body_speed_state"): tuple(
        # Corrected fixed-distance rigid-body replan (51 primitives).  The
        # final six primitives carry the operational speed-state decision:
        # 0.08 -> 0.03 -> 0.01 m/s before the terminal stop.
        (*action, target_speed)
        for action, target_speed in zip(
            (
                [("front_only", 0.70, 1)] * 32
                + [("front_only", 0.50, 1)]
                + [("front_only", 0.25, 1)] * 2
                + [("rear_only", 0.25, 1)] * 2
                + [("rear_only", 0.50, 1)] * 3
                + [("rear_only", 0.70, 1)] * 9
                + [("rear_only", 0.50, 1)] * 2
            ),
            [0.08] * 45 + [0.03] * 2 + [0.01] * 4,
        )
    ),
    ("rigid_body_stopping_gate", "joint_path_speed_state"): (
        ("coordinated_counterphase", 0.0, 1, 0.08),
        ("coordinated_counterphase", 0.0, 1, 0.08),
        ("coordinated_counterphase", 0.0, 1, 0.08),
        ("coordinated_counterphase", 0.0, 1, 0.01),
        ("coordinated_counterphase", 0.0, 1, 0.01),
        ("coordinated_counterphase", 0.0, 1, 0.01),
    ),
    ("s_corridor", "fixed_counterphase"): (
        [("coordinated_counterphase", 0.0, 1)]
        + [("coordinated_counterphase", -0.25, 1)] * 4
        + [("coordinated_counterphase", -0.50, 1)] * 6
        + [("coordinated_counterphase", -0.25, 1)] * 10
        + [("coordinated_counterphase", 0.0, 1)] * 4
        + [("coordinated_counterphase", 0.25, 1)]
        + [("coordinated_counterphase", 0.50, 1)] * 23
        + [("coordinated_counterphase", 0.25, 1)] * 13
        + [("coordinated_counterphase", 0.0, 1)]
        + [("coordinated_counterphase", -0.25, 1)] * 17
        + [("coordinated_counterphase", -0.50, 1)] * 13
        + [("coordinated_counterphase", -0.25, 1)] * 13
        + [("coordinated_counterphase", 0.0, 1)] * 3
        + [("coordinated_counterphase", 0.25, 1)] * 2
        + [("coordinated_counterphase", 0.50, 1)]
        + [("coordinated_counterphase", 0.70, 1)] * 6
        + [("coordinated_counterphase", 0.50, 1)]
    ),
    ("s_corridor", "proposed_full_dynamics"): tuple(
        SCENARIOS["s_corridor"]["actions"]
    ),
    ("doorway_immediate_turn", "fixed_counterphase"): (
        [("coordinated_counterphase", 0.0, 1)]
        + [("coordinated_counterphase", 0.25, 1)] * 4
        + [("coordinated_counterphase", 0.50, 1)] * 18
        + [("coordinated_counterphase", 0.70, 1)] * 13
        + [("coordinated_counterphase", 0.50, 1)] * 4
        + [("coordinated_counterphase", 0.25, 1)] * 11
    ),
    ("doorway_immediate_turn", "proposed_full_dynamics"): (
        # Replanned with provisional measured steering tracking/zero offsets
        # (front -1.0 deg, rear -1.0 deg at every target).
        # This candidate stays in counterphase for this particular geometry;
        # the planner is allowed to switch modes but does not require it here.
        [("coordinated_counterphase", 0.0, 1)] * 2
        + [("coordinated_counterphase", 0.25, 1)]
        + [("coordinated_counterphase", 0.50, 1)] * 8
        + [("coordinated_counterphase", 0.70, 1)] * 15
        + [("coordinated_counterphase", 0.50, 1)] * 8
        + [("coordinated_counterphase", 0.25, 1)] * 15
        + [("coordinated_counterphase", 0.50, 1)]
        + [("coordinated_counterphase", 0.70, 1)]
    ),
}
REPLAY_POLICY_ACTIONS = {
    key: tuple(actions) for key, actions in REPLAY_POLICY_ACTIONS.items()
}


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def path_tracking_curvature(
    nominal_curvature,
    current_x,
    current_y,
    current_yaw,
    reference_x,
    reference_y,
    reference_yaw,
    lateral_gain,
    heading_gain,
    maximum_correction,
    curvature_limit=0.70,
):
    """Apply bounded cross-track and heading feedback to one lattice edge."""
    dx = current_x - reference_x
    dy = current_y - reference_y
    lateral_error = -math.sin(reference_yaw) * dx + math.cos(
        reference_yaw
    ) * dy
    heading_error = wrap(reference_yaw - current_yaw)
    correction = (
        heading_gain * heading_error - lateral_gain * lateral_error
    )
    correction = max(
        -maximum_correction, min(maximum_correction, correction)
    )
    command_curvature = max(
        -curvature_limit,
        min(curvature_limit, nominal_curvature + correction),
    )
    return command_curvature, lateral_error, heading_error


def yaw_from_odometry(message):
    q = message.pose.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def stamp_seconds(message):
    return float(message.header.stamp.sec) + 1.0e-9 * float(message.header.stamp.nanosec)


def footprint_y_bounds(y, yaw):
    half_extent = (
        abs(math.sin(yaw)) * BODY_LENGTH_M / 2.0
        + abs(math.cos(yaw)) * BODY_WIDTH_M / 2.0
    )
    return y - half_extent, y + half_extent


def stopped_boundary_trigger_distance(
    target_distance, requested_lead_distance, completion_tolerance
):
    """Return a stop trigger that cannot deadlock below completion tolerance."""
    if min(target_distance, requested_lead_distance, completion_tolerance) < 0.0:
        raise ValueError("stopped-boundary distances must be nonnegative")
    effective_lead = min(requested_lead_distance, completion_tolerance)
    return max(0.0, target_distance - effective_lead)


def split_replay_action(action, default_speed_m_s):
    """Return mode, curvature, direction and per-edge target speed."""
    if len(action) == 3:
        mode, curvature, direction = action
        speed = default_speed_m_s
    elif len(action) == 4:
        mode, curvature, direction, speed = action
    else:
        raise ValueError(f"Replay action must have 3 or 4 fields: {action}")
    speed = float(speed)
    if speed <= 0.0:
        raise ValueError(f"Replay action speed must be positive: {action}")
    return str(mode), float(curvature), int(direction), speed


class SpatialUtilityReplay(Node):
    def __init__(self):
        super().__init__("spatial_utility_replay")
        self.declare_parameter("scenario", "rear_end_constrained")
        self.declare_parameter("run_id", "r01")
        self.declare_parameter("replay_policy", "scenario_default")
        self.declare_parameter("experiment_condition_id", "nominal_265_center")
        self.declare_parameter("planning_mode", "replay")
        self.declare_parameter("external_plan_path", "")
        self.declare_parameter("planner_robust_clearance_m", 0.010)
        self.declare_parameter("speed_m_s", 0.03)
        self.declare_parameter("settle_s", 1.0)
        self.declare_parameter("terminal_stop_timeout_s", 15.0)
        self.declare_parameter("terminal_neutral_alignment_enabled", False)
        self.declare_parameter("terminal_pose_control_enabled", False)
        self.declare_parameter("terminal_control_position_tolerance_m", 0.015)
        self.declare_parameter("terminal_control_target_tolerance_m", 0.005)
        self.declare_parameter("terminal_control_heading_tolerance_deg", 1.5)
        self.declare_parameter("terminal_settle_abort_position_m", 0.030)
        self.declare_parameter("terminal_settle_abort_heading_deg", 2.0)
        self.declare_parameter("terminal_control_max_speed_m_s", 0.010)
        self.declare_parameter("terminal_control_min_speed_m_s", 0.003)
        self.declare_parameter("terminal_control_position_gain_s_inv", 0.5)
        self.declare_parameter("terminal_control_velocity_gain", 0.8)
        self.declare_parameter("path_tracking_feedback_enabled", False)
        self.declare_parameter("path_tracking_lateral_gain", 4.0)
        self.declare_parameter("path_tracking_heading_gain", 1.0)
        self.declare_parameter("path_tracking_max_curvature_correction", 0.25)
        self.declare_parameter("mode_transition_settle_s", 2.0)
        self.declare_parameter("mode_transition_policy", "stopped_alignment")
        self.declare_parameter("braking_distance_m", 0.03)
        self.declare_parameter("minimum_braking_speed_m_s", 0.015)
        self.declare_parameter("primitive_completion_tolerance_m", 0.015)
        self.declare_parameter("stop_lead_distance_m", 0.015)
        self.declare_parameter("stopped_speed_threshold_m_s", 0.005)
        self.declare_parameter("stopped_hold_s", 0.25)
        self.declare_parameter("alignment_preview_speed_m_s", 0.0001)
        self.declare_parameter("output_dir", "")
        self.scenario_name = str(self.get_parameter("scenario").value)
        if self.scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario_name}")
        self.scenario = dict(SCENARIOS[self.scenario_name])
        self.scenario["actions"] = list(self.scenario["actions"])
        self.run_id = str(self.get_parameter("run_id").value)
        self.replay_policy = str(self.get_parameter("replay_policy").value)
        self.experiment_condition_id = str(
            self.get_parameter("experiment_condition_id").value
        )
        self.planning_mode = str(self.get_parameter("planning_mode").value)
        if self.planning_mode not in ("replay", "online", "topic"):
            raise ValueError("planning_mode must be replay, online, or topic")
        self.planner_robust_clearance = float(
            self.get_parameter("planner_robust_clearance_m").value
        )
        self.speed = float(self.get_parameter("speed_m_s").value)
        if self.speed <= 0.0:
            raise ValueError("speed_m_s must be positive")
        self.settle_s = float(self.get_parameter("settle_s").value)
        self.terminal_stop_timeout_s = float(
            self.get_parameter("terminal_stop_timeout_s").value
        )
        self.terminal_neutral_alignment_enabled = bool(
            self.get_parameter("terminal_neutral_alignment_enabled").value
        )
        self.terminal_pose_control_enabled = bool(
            self.get_parameter("terminal_pose_control_enabled").value
        )
        self.terminal_control_position_tolerance = float(
            self.get_parameter("terminal_control_position_tolerance_m").value
        )
        self.terminal_control_target_tolerance = float(
            self.get_parameter("terminal_control_target_tolerance_m").value
        )
        self.terminal_control_heading_tolerance = math.radians(
            float(
                self.get_parameter(
                    "terminal_control_heading_tolerance_deg"
                ).value
            )
        )
        self.terminal_settle_abort_position = float(
            self.get_parameter("terminal_settle_abort_position_m").value
        )
        self.terminal_settle_abort_heading = math.radians(
            float(
                self.get_parameter(
                    "terminal_settle_abort_heading_deg"
                ).value
            )
        )
        self.terminal_control_max_speed = float(
            self.get_parameter("terminal_control_max_speed_m_s").value
        )
        self.terminal_control_min_speed = float(
            self.get_parameter("terminal_control_min_speed_m_s").value
        )
        self.terminal_control_position_gain = float(
            self.get_parameter("terminal_control_position_gain_s_inv").value
        )
        self.terminal_control_velocity_gain = float(
            self.get_parameter("terminal_control_velocity_gain").value
        )
        self.path_tracking_feedback_enabled = bool(
            self.get_parameter("path_tracking_feedback_enabled").value
        )
        self.path_tracking_lateral_gain = float(
            self.get_parameter("path_tracking_lateral_gain").value
        )
        self.path_tracking_heading_gain = float(
            self.get_parameter("path_tracking_heading_gain").value
        )
        self.path_tracking_max_curvature_correction = float(
            self.get_parameter(
                "path_tracking_max_curvature_correction"
            ).value
        )
        if self.settle_s < 0.0 or self.terminal_stop_timeout_s <= 0.0:
            raise ValueError(
                "settle_s must be nonnegative and terminal_stop_timeout_s "
                "must be positive"
            )
        if not (
            0.0 < self.terminal_control_position_tolerance < STEP_M
            and 0.0 < self.terminal_control_target_tolerance
            < self.terminal_control_position_tolerance
            and 0.0 < self.terminal_control_heading_tolerance < math.pi
            and self.terminal_control_position_tolerance
            < self.terminal_settle_abort_position < STEP_M
            and self.terminal_control_heading_tolerance
            < self.terminal_settle_abort_heading < math.pi
            and 0.0 < self.terminal_control_min_speed
            <= self.terminal_control_max_speed
            and self.terminal_control_position_gain > 0.0
            and self.terminal_control_velocity_gain >= 0.0
            and self.path_tracking_lateral_gain >= 0.0
            and self.path_tracking_heading_gain >= 0.0
            and 0.0 <= self.path_tracking_max_curvature_correction <= 0.70
        ):
            raise ValueError("invalid terminal pose-control parameters")
        self.mode_transition_settle_s = float(
            self.get_parameter("mode_transition_settle_s").value
        )
        self.mode_transition_policy = str(
            self.get_parameter("mode_transition_policy").value
        )
        if self.mode_transition_policy not in (
            "stopped_alignment",
            "rolling_rate_limited",
        ):
            raise ValueError(
                "mode_transition_policy must be stopped_alignment or "
                "rolling_rate_limited"
            )
        self.braking_distance = float(
            self.get_parameter("braking_distance_m").value
        )
        self.minimum_braking_speed = float(
            self.get_parameter("minimum_braking_speed_m_s").value
        )
        self.primitive_completion_tolerance = float(
            self.get_parameter("primitive_completion_tolerance_m").value
        )
        if not 0.0 <= self.primitive_completion_tolerance < STEP_M:
            raise ValueError(
                "primitive_completion_tolerance_m must be in [0, STEP_M)"
            )
        self.stop_lead_distance = float(
            self.get_parameter("stop_lead_distance_m").value
        )
        self.stopped_speed_threshold = float(
            self.get_parameter("stopped_speed_threshold_m_s").value
        )
        self.stopped_hold_s = float(self.get_parameter("stopped_hold_s").value)
        self.alignment_preview_speed = float(
            self.get_parameter("alignment_preview_speed_m_s").value
        )
        configured_output_dir = str(self.get_parameter("output_dir").value).strip()
        if configured_output_dir:
            self.output_dir = Path(configured_output_dir)
        elif self.scenario_name in (
            "corridor_90_turn",
            "narrow_t_intersection",
            "s_corridor",
            "doorway_immediate_turn",
            "obstacle_bypass_return",
            "rigid_body_stopping_gate",
        ):
            self.output_dir = Path(
                "studies/IJCAS/results/general_navigation_dynamics_gate/dynamic_replay"
            )
        else:
            self.output_dir = Path(
                "studies/IJCAS/results/steering_mode_utility/dynamic_replay"
            )

        self.planning_time_s = 0.0
        self.planner_expanded_nodes = 0
        self.planned_states = []
        if self.planning_mode == "online":
            planning_start = time.perf_counter()
            result = lattice_plan(
                self.scenario["goal"],
                self.scenario["minimum_y"],
                self.scenario["maximum_y"],
                robust_clearance=self.planner_robust_clearance,
                speed_m_s=self.speed,
            )
            self.planning_time_s = time.perf_counter() - planning_start
            if result is None:
                raise RuntimeError(
                    f"Online lattice found no path for {self.scenario_name}"
                )
            self.scenario["actions"] = list(result.actions)
            self.planned_states = list(result.states)
            self.planner_expanded_nodes = result.expanded_nodes
        elif self.planning_mode == "topic":
            self.scenario["actions"] = []
        elif self.replay_policy == "external_integrated_plan":
            configured_path = str(
                self.get_parameter("external_plan_path").value
            ).strip()
            if not configured_path:
                raise ValueError(
                    "external_plan_path is required for external_integrated_plan"
                )
            plan_path = Path(configured_path)
            if not plan_path.exists():
                raise FileNotFoundError(plan_path)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            if payload.get("scenario") != self.scenario_name:
                raise ValueError(
                    "external plan scenario mismatch: "
                    f"{payload.get('scenario')} != {self.scenario_name}"
                )
            if not payload.get("plan_success"):
                raise ValueError("external integrated plan did not pass planning")
            selected_actions = payload.get("selected_actions") or []
            if not selected_actions:
                raise ValueError("external integrated plan has no selected_actions")
            self.scenario["actions"] = [
                (
                    str(action["mode"]),
                    float(action["curvature_1_m"]),
                    int(action["direction"]),
                    float(action["target_speed_m_s"]),
                )
                for action in selected_actions
            ]
            selected_states = payload.get("selected_states") or []
            if selected_states:
                if len(selected_states) != len(selected_actions):
                    raise ValueError(
                        "external plan state/action length mismatch"
                    )
                self.planned_states = [
                    SimpleNamespace(
                        x=float(state["x_m"]),
                        y=float(state["y_m"]),
                        yaw=math.radians(float(state["yaw_deg"])),
                    )
                    for state in selected_states
                ]
            if self.path_tracking_feedback_enabled and not self.planned_states:
                raise ValueError(
                    "path tracking feedback requires selected_states in the "
                    "external plan"
                )
            self.planning_time_s = float(payload.get("planning_time_s", 0.0))
        elif self.replay_policy != "scenario_default":
            key = (self.scenario_name, self.replay_policy)
            if key not in REPLAY_POLICY_ACTIONS:
                supported = ", ".join(
                    f"{scenario}:{policy}"
                    for scenario, policy in sorted(REPLAY_POLICY_ACTIONS)
                )
                raise ValueError(
                    f"Unsupported replay policy {key}; supported pairs: {supported}"
                )
            self.scenario["actions"] = list(REPLAY_POLICY_ACTIONS[key])

        self.cmd_pub = self.create_publisher(Twist, "/sim/cmd_vel", 10)
        self.mode_pub = self.create_publisher(String, "/sim/steering_configuration", 10)
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(NavPath, "/sim/ijcas/planned_path", latched_qos)
        self.plan_pub = self.create_publisher(String, "/sim/ijcas/steering_plan", latched_qos)
        if self.planning_mode == "topic":
            self.create_subscription(
                String,
                "/sim/ijcas/plan_command",
                self._plan_command_callback,
                latched_qos,
            )
        self.create_subscription(
            Odometry, "/sim/ground_truth/odom", self._odom_callback, 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.create_timer(0.05, self._tick)

        self.odom: Optional[Odometry] = None
        self.actual_front_steering = math.nan
        self.actual_rear_steering = math.nan
        self.origin = None
        self.start_s = None
        self.previous_local = None
        self.segment_distance = 0.0
        self.action_index = 0
        self.settle_start_s = None
        self.active_mode = (
            self.scenario["actions"][0][0]
            if self.scenario["actions"] else "coordinated_counterphase"
        )
        self.active_curvature = (
            self.scenario["actions"][0][1]
            if self.scenario["actions"] else 0.0
        )
        self.transition_start_s = None
        self.transition_duration_s = None
        self.segment_progress_reference_m = 0.0
        self.segment_progress_reference_s = None
        self.stop_phase = False
        self.stop_phase_start_s = None
        self.stopped_within_since_s = None
        self.task_complete_s = None
        self.terminal_stop_timeout_reached = False
        self.terminal_neutral_alignment_start_s = None
        self.terminal_neutral_alignment_duration_s = 0.0
        self.terminal_neutral_alignment_complete = (
            not self.terminal_neutral_alignment_enabled
        )
        self.terminal_pose_correction_active = False
        self.terminal_pose_correction_count = 0
        self.terminal_correction_direction = 0
        self.terminal_lateral_mode = ""
        self.terminal_lateral_mode_switch_count = 0
        self.maximum_abs_terminal_lateral_error_m = 0.0
        self.terminal_settle_latched = False
        self.terminal_stationary_samples = []
        self.terminal_stationary_window_duration_s = 0.0
        self.terminal_stationary_window_distance_m = 0.0
        self.terminal_stationary_average_speed_m_s = float("inf")
        self.maximum_abs_path_lateral_error_m = 0.0
        self.maximum_abs_path_heading_error_rad = 0.0
        self.maximum_abs_path_curvature_correction = 0.0
        self.stop_event_count = 0
        self.mode_transition_count = 0
        self.rolling_curvature_transition_count = 0
        self.total_stopped_boundary_time_s = 0.0
        self.total_mode_alignment_time_s = 0.0
        self.motion_start_logged = False
        self.last_progress_log_s = None
        self.rows = []
        self.clearance_samples = []
        self.done = False
        self.get_logger().info(
            f"Waiting for odometry: scenario={self.scenario_name}, run_id={self.run_id}, "
            f"planning_mode={self.planning_mode}"
        )
        if self.planning_mode == "online":
            self.get_logger().info(
                "Online lattice complete: "
                f"actions={len(self.scenario['actions'])}, "
                f"expanded={self.planner_expanded_nodes}, "
                f"planning_time={self.planning_time_s:.4f} s"
            )

    def _odom_callback(self, message):
        self.odom = message
        if self.origin is None and self.scenario["actions"]:
            self._initialize_execution(message)

    def _initialize_execution(self, message, planned_origin=None):
        if self.origin is not None or not self.scenario["actions"]:
            return
        if planned_origin is None:
            p = message.pose.pose.position
            self.origin = (p.x, p.y, yaw_from_odometry(message))
        else:
            self.origin = tuple(planned_origin)
        first_mode, first_curvature, _, _ = split_replay_action(
            self.scenario["actions"][0], self.speed
        )
        self.active_mode, self.active_curvature = first_mode, first_curvature
        first_targets = self._steering_targets(first_mode, first_curvature)
        initial_duration = max(
            self.mode_transition_settle_s,
            max(abs(value) for value in first_targets)
            / math.radians(STEERING_RATE_DEG_S)
            + 0.5,
        )
        self.start_s = stamp_seconds(message) + initial_duration
        self.get_logger().info(
            "Odometry acquired; initial steering preview: "
            f"mode={first_mode}, curvature={first_curvature:+.2f} 1/m, "
            f"duration={initial_duration:.2f} s sim-time"
        )
        self._publish_plan(message.header.stamp)

    def _plan_command_callback(self, message):
        if self.motion_start_logged or self.scenario["actions"]:
            return
        payload = json.loads(message.data)
        actions = [
            (
                str(action["mode"]),
                float(action["curvature_1_m"]),
                int(action["direction"]),
                float(action.get("target_speed_m_s", self.speed)),
            )
            for action in payload["actions"]
        ]
        if not actions:
            self.get_logger().error("Received an empty plan command")
            return
        self.scenario["actions"] = actions
        self.scenario["goal"] = tuple(float(value) for value in payload["goal_local"])
        self.run_id = str(payload.get("run_id", self.run_id))
        self.planning_time_s = float(payload.get("planning_time_s", 0.0))
        self.planner_expanded_nodes = int(payload.get("expanded_nodes", 0))
        self.get_logger().info(
            f"Accepted /map+/goal_pose plan: actions={len(actions)}, "
            f"planning_time={self.planning_time_s:.4f} s"
        )
        if self.odom is not None:
            self._initialize_execution(self.odom, payload.get("start"))

    def _publish_plan(self, stamp):
        states = self.planned_states
        if not states:
            # Preserve observability for replay mode as well.
            states = []
        path = NavPath()
        path.header.stamp = stamp
        path.header.frame_id = "odom"
        if self.origin is not None:
            x0, y0, yaw0 = self.origin
            for state in states:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = x0 + math.cos(yaw0) * state.x - math.sin(yaw0) * state.y
                pose.pose.position.y = y0 + math.sin(yaw0) * state.x + math.cos(yaw0) * state.y
                world_yaw = yaw0 + state.yaw
                pose.pose.orientation.z = math.sin(world_yaw / 2.0)
                pose.pose.orientation.w = math.cos(world_yaw / 2.0)
                path.poses.append(pose)
        if states:
            self.path_pub.publish(path)
        payload = {
            "scenario": self.scenario_name,
            "run_id": self.run_id,
            "planning_mode": self.planning_mode,
            "planning_time_s": self.planning_time_s,
            "expanded_nodes": self.planner_expanded_nodes,
            "actions": [
                {
                    "mode": split_replay_action(action, self.speed)[0],
                    "curvature_1_m": split_replay_action(action, self.speed)[1],
                    "direction": split_replay_action(action, self.speed)[2],
                    "target_speed_m_s": split_replay_action(action, self.speed)[3],
                }
                for action in self.scenario["actions"]
            ],
        }
        self.plan_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _joint_callback(self, message):
        index = {name: position for position, name in enumerate(message.name)}
        front = index.get("base_link_to_front_active_steering")
        rear = index.get("base_link_to_rear_active_steering")
        if front is not None and front < len(message.position):
            self.actual_front_steering = message.position[front]
        if rear is not None and rear < len(message.position):
            self.actual_rear_steering = message.position[rear]

    @staticmethod
    def _steering_targets(mode, curvature):
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

    def _required_transition_duration(self, mode, curvature):
        old_targets = self._steering_targets(
            self.active_mode, self.active_curvature
        )
        new_targets = self._steering_targets(mode, curvature)
        maximum_change = max(
            abs(new - old) for old, new in zip(old_targets, new_targets)
        )
        return max(
            self.mode_transition_settle_s,
            maximum_change / math.radians(STEERING_RATE_DEG_S) + 0.5,
        )

    def _local_pose(self):
        p = self.odom.pose.pose.position
        yaw = yaw_from_odometry(self.odom)
        x0, y0, yaw0 = self.origin
        dx, dy = p.x - x0, p.y - y0
        return (
            math.cos(yaw0) * dx + math.sin(yaw0) * dy,
            -math.sin(yaw0) * dx + math.cos(yaw0) * dy,
            wrap(yaw - yaw0),
        )

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _complete_action(self, now, x, y, yaw, carry_distance=0.0):
        self.action_index += 1
        self.segment_distance = max(0.0, carry_distance)
        self.segment_progress_reference_m = self.segment_distance
        self.segment_progress_reference_s = now
        self.stop_phase = False
        if self.stop_phase_start_s is not None:
            self.total_stopped_boundary_time_s += now - self.stop_phase_start_s
        self.stop_phase_start_s = None
        self.stopped_within_since_s = None
        self.previous_local = (x, y, yaw)
        if self.action_index >= len(self.scenario["actions"]):
            self.task_complete_s = now
        self.get_logger().info(
            f"Primitive complete: {self.action_index}/{len(self.scenario['actions'])}"
        )

    def _publish_alignment_preview(self, curvature, direction):
        """Turn steering joints while producing negligible translation.

        The configuration controller only advances its steering target for a
        nonzero longitudinal command.  A 0.1 mm/s preview preserves that
        interface while limiting nominal displacement to 0.2 mm over 2 s.
        """
        command = Twist()
        command.linear.x = direction * self.alignment_preview_speed
        command.angular.z = command.linear.x * curvature
        self.cmd_pub.publish(command)

    def _tick(self):
        if self.done or self.odom is None or self.start_s is None:
            return
        now = stamp_seconds(self.odom)
        if now < self.start_s:
            first_mode, first_curvature, first_direction, _ = split_replay_action(
                self.scenario["actions"][0], self.speed
            )
            self.mode_pub.publish(String(data=first_mode))
            self._publish_alignment_preview(first_curvature, first_direction)
            return

        if not self.motion_start_logged:
            self.motion_start_logged = True
            self.last_progress_log_s = now
            # Initial steering preparation is outside the timed planning task.
            # Rebase here so its small physical displacement is not charged to
            # the planned path or final pose error.
            p = self.odom.pose.pose.position
            self.origin = (p.x, p.y, yaw_from_odometry(self.odom))
            self.previous_local = None
            # Re-anchor the published path after the untimed steering preview,
            # matching the execution origin used for all tracking metrics.
            self._publish_plan(self.odom.header.stamp)
            self.get_logger().info(
                "Initial steering preview complete; starting motion primitive 1/"
                f"{len(self.scenario['actions'])}"
            )

        x, y, yaw = self._local_pose()
        minimum_body_y, maximum_body_y = footprint_y_bounds(y, yaw)
        clearance_lower = minimum_body_y - self.scenario["minimum_y"]
        clearance_upper = self.scenario["maximum_y"] - maximum_body_y
        self.clearance_samples.append((clearance_lower, clearance_upper))

        if self.action_index >= len(self.scenario["actions"]):
            if not self.terminal_neutral_alignment_complete:
                self.mode_pub.publish(
                    String(data="coordinated_counterphase")
                )
                self._publish_alignment_preview(0.0, 1)
                if self.terminal_neutral_alignment_start_s is None:
                    measured = (
                        self.actual_front_steering,
                        self.actual_rear_steering,
                    )
                    finite_angles = [
                        abs(value) for value in measured if math.isfinite(value)
                    ]
                    maximum_angle = max(finite_angles, default=0.0)
                    self.terminal_neutral_alignment_duration_s = max(
                        self.mode_transition_settle_s,
                        maximum_angle / math.radians(STEERING_RATE_DEG_S)
                        + 0.5,
                    )
                    self.terminal_neutral_alignment_start_s = now
                    self.get_logger().info(
                        "Terminal neutral steering alignment started: "
                        f"duration={self.terminal_neutral_alignment_duration_s:.2f} s"
                    )
                if (
                    now - self.terminal_neutral_alignment_start_s
                    < self.terminal_neutral_alignment_duration_s
                ):
                    return
                self.terminal_neutral_alignment_complete = True
                # Start the terminal stability timeout only after the common
                # zero-curvature alignment has completed.
                self.task_complete_s = now
                self._stop()
                self.get_logger().info(
                    "Terminal neutral steering alignment complete"
                )
                return
            goal_x, goal_y, goal_yaw = self.scenario["goal"]
            dx, dy = goal_x - x, goal_y - y
            position_error = math.hypot(dx, dy)
            (
                longitudinal_error,
                lateral_error,
                heading_error,
            ) = terminal_pose_errors(dx, dy, yaw, goal_yaw)
            self.maximum_abs_terminal_lateral_error_m = max(
                self.maximum_abs_terminal_lateral_error_m,
                abs(lateral_error),
            )
            pose_inside = (
                position_error <= self.terminal_control_position_tolerance
                and abs(heading_error)
                <= self.terminal_control_heading_tolerance
            )
            target_inside = (
                position_error <= self.terminal_control_target_tolerance
                and abs(heading_error)
                <= self.terminal_control_heading_tolerance
            )
            settle_envelope_inside = (
                position_error <= self.terminal_settle_abort_position
                and abs(heading_error)
                <= self.terminal_settle_abort_heading
            )
            velocity = self.odom.twist.twist.linear
            planar_speed = math.hypot(velocity.x, velocity.y)
            acceptance_hold = (
                pose_inside
                and planar_speed <= self.stopped_speed_threshold
            )
            if self.terminal_pose_control_enabled:
                if not self.terminal_settle_latched and acceptance_hold:
                    self.terminal_settle_latched = True
                    self.get_logger().info(
                        "Terminal settle latched inside acceptance band: "
                        f"position={position_error:.4f} m, "
                        f"speed={planar_speed:.4f} m/s"
                    )
                elif (
                    self.terminal_settle_latched
                    and not settle_envelope_inside
                ):
                    self.terminal_settle_latched = False
                    self.settle_start_s = None
                    self.terminal_stationary_samples.clear()
                    self.get_logger().warning(
                        "Terminal settle envelope exited; resuming pose "
                        f"correction: position={position_error:.4f} m, "
                        f"heading={math.degrees(heading_error):.3f} deg"
                    )
            terminal_command_speed = 0.0
            terminal_curvature = 0.0
            terminal_mode = "coordinated_counterphase"
            correction_active = (
                self.terminal_pose_control_enabled
                and not target_inside
                and not self.terminal_settle_latched
            )
            if correction_active:
                # This remains a bounded execution stabilizer, not a replanner.
                # A one-axle arc is used when lateral error dominates because a
                # counterphase-only command cannot remove body-frame lateral
                # residual.  All commands remain inside the same three steering
                # modes and 0.70 1/m curvature limit used by the lattice.
                self.terminal_correction_direction = (
                    terminal_correction_direction(
                        longitudinal_error,
                        self.terminal_correction_direction,
                        self.terminal_control_position_tolerance,
                    )
                )
                direction = self.terminal_correction_direction
                longitudinal_velocity = velocity.x
                directed_velocity = direction * longitudinal_velocity
                raw_speed_magnitude = (
                    self.terminal_control_position_gain * position_error
                    - self.terminal_control_velocity_gain
                    * directed_velocity
                )
                speed_magnitude = max(
                    0.0,
                    min(
                        self.terminal_control_max_speed,
                        raw_speed_magnitude,
                    ),
                )
                # Apply breakaway speed only outside the acceptance band and
                # while nearly stationary. Keep the latched travel direction
                # through a small longitudinal zero crossing.
                if (
                    position_error
                    > self.terminal_control_position_tolerance
                    and abs(longitudinal_velocity)
                    < 0.5 * self.stopped_speed_threshold
                    and speed_magnitude < self.terminal_control_min_speed
                ):
                    speed_magnitude = self.terminal_control_min_speed
                command_speed = direction * speed_magnitude
                lateral_deadband = max(
                    0.001,
                    0.5 * self.terminal_control_target_tolerance,
                )
                if abs(lateral_error) > lateral_deadband:
                    terminal_mode, curvature = terminal_lateral_correction(
                        lateral_error,
                        heading_error,
                        direction,
                        self.terminal_lateral_mode,
                        math.radians(0.15),
                    )
                    if terminal_mode != self.terminal_lateral_mode:
                        self.terminal_lateral_mode_switch_count += 1
                    self.terminal_lateral_mode = terminal_mode
                else:
                    terminal_mode = "coordinated_counterphase"
                    self.terminal_lateral_mode = ""
                    curvature = max(
                        -0.70,
                        min(
                            0.70,
                            heading_error
                            / math.copysign(
                                max(abs(longitudinal_error), 0.05),
                                longitudinal_error,
                            ),
                        ),
                    )
                command = Twist()
                command.linear.x = command_speed
                command.angular.z = command_speed * curvature
                terminal_command_speed = command_speed
                terminal_curvature = curvature
                self.mode_pub.publish(String(data=terminal_mode))
                self.cmd_pub.publish(command)
                if not self.terminal_pose_correction_active:
                    self.terminal_pose_correction_count += 1
                    self.get_logger().info(
                        "Terminal pose correction started: "
                        f"position={position_error:.4f} m, "
                        f"heading={math.degrees(heading_error):.3f} deg"
                    )
                self.terminal_pose_correction_active = True
                self.settle_start_s = None
            else:
                self.terminal_pose_correction_active = False
                self.terminal_correction_direction = 0
                self.terminal_lateral_mode = ""
                self.mode_pub.publish(
                    String(data="coordinated_counterphase")
                )
                self._stop()
            target_front, target_rear = self._steering_targets(
                terminal_mode, terminal_curvature
            )
            self.rows.append(
                {
                    "execution_phase": "terminal_correction",
                    "sim_time_s": now,
                    "elapsed_s": now - self.start_s,
                    "action_index": self.action_index,
                    "mode": terminal_mode,
                    "direction": (
                        1 if terminal_command_speed >= 0.0 else -1
                    ),
                    "curvature_1_m": terminal_curvature,
                    "planned_speed_m_s": 0.0,
                    "command_speed_m_s": terminal_command_speed,
                    "target_front_steering_deg": math.degrees(target_front),
                    "target_rear_steering_deg": math.degrees(target_rear),
                    "actual_front_steering_deg": math.degrees(
                        self.actual_front_steering
                    ),
                    "actual_rear_steering_deg": math.degrees(
                        self.actual_rear_steering
                    ),
                    "segment_distance_m": 0.0,
                    "target_distance_m": 0.0,
                    "x_m": x,
                    "y_m": y,
                    "yaw_deg": math.degrees(yaw),
                    "footprint_min_y_m": minimum_body_y,
                    "footprint_max_y_m": maximum_body_y,
                    "lower_clearance_m": clearance_lower,
                    "upper_clearance_m": clearance_upper,
                }
            )
            # Do not accept a transient zero crossing as the terminal stop.
            # The vehicle must remain below the stopped-speed threshold for
            # the complete settle interval; any rebound/creep restarts it.
            settle_pose_accepted = (
                self.terminal_settle_latched and settle_envelope_inside
                if self.terminal_pose_control_enabled
                else pose_inside
            )
            settled = False
            if self.terminal_pose_control_enabled:
                if settle_pose_accepted and not correction_active:
                    self.terminal_stationary_samples.append((now, x, y))
                    cutoff_s = now - self.settle_s
                    reference_index = None
                    for index, sample in enumerate(
                        self.terminal_stationary_samples
                    ):
                        if sample[0] <= cutoff_s:
                            reference_index = index
                        else:
                            break
                    if reference_index is not None:
                        window = self.terminal_stationary_samples[
                            reference_index:
                        ]
                        duration = window[-1][0] - window[0][0]
                        distance = sum(
                            math.hypot(
                                current[1] - previous[1],
                                current[2] - previous[2],
                            )
                            for previous, current in zip(
                                window, window[1:]
                            )
                        )
                        average_speed = (
                            distance / duration
                            if duration > 0.0 else float("inf")
                        )
                        self.terminal_stationary_window_duration_s = duration
                        self.terminal_stationary_window_distance_m = distance
                        self.terminal_stationary_average_speed_m_s = (
                            average_speed
                        )
                        settled = (
                            duration >= self.settle_s
                            and average_speed
                            <= self.stopped_speed_threshold
                            and planar_speed
                            <= self.stopped_speed_threshold
                        )
                        # Retain one sample before the next window boundary so
                        # distance remains continuous without unbounded growth.
                        if reference_index > 0:
                            self.terminal_stationary_samples = (
                                self.terminal_stationary_samples[
                                    reference_index:
                                ]
                            )
                else:
                    self.terminal_stationary_samples.clear()
            else:
                if (
                    settle_pose_accepted
                    and not correction_active
                    and planar_speed <= self.stopped_speed_threshold
                ):
                    if self.settle_start_s is None:
                        self.settle_start_s = now
                else:
                    if self.settle_start_s is not None:
                        self.get_logger().warning(
                            "Terminal stop rebound; restarting settle hold: "
                            f"speed={planar_speed:.4f} m/s"
                        )
                    self.settle_start_s = None
                settled = (
                    self.settle_start_s is not None
                    and now - self.settle_start_s >= self.settle_s
                )
            terminal_wait = (
                now - self.task_complete_s
                if self.task_complete_s is not None
                else 0.0
            )
            if settled:
                self._finish(x, y, yaw)
            elif terminal_wait >= self.terminal_stop_timeout_s:
                self.terminal_stop_timeout_reached = True
                self.get_logger().error(
                    "Terminal stop did not remain stable before timeout: "
                    f"speed={planar_speed:.4f} m/s, "
                    f"timeout={self.terminal_stop_timeout_s:.1f} s"
                )
                self._finish(x, y, yaw)
            return

        mode, curvature, direction, action_speed = split_replay_action(
            self.scenario["actions"][self.action_index], self.speed
        )
        self.mode_pub.publish(String(data=mode))

        # A physical configuration change (for example, coordinated steering
        # to rear-only steering) receives an explicit stopped alignment.  A
        # curvature change within the same configuration remains a rolling
        # transition: ConfigurationController already rate-limits the steering
        # joints while continuing to drive.  Treating every curvature sample as
        # a mode transition made a 0.69 m path spend more time stopped than
        # moving and is not representative of the proposed planner.
        mode_changed = mode != self.active_mode
        if mode_changed and self.mode_transition_policy == "stopped_alignment":
            self._publish_alignment_preview(curvature, direction)
            if self.transition_start_s is None:
                self.transition_start_s = now
                self.mode_transition_count += 1
                self.transition_duration_s = self._required_transition_duration(
                    mode, curvature
                )
                self.get_logger().info(
                    "Aligning steering state: "
                    f"({self.active_mode}, {self.active_curvature:+.2f}) -> "
                    f"({mode}, {curvature:+.2f}), "
                    f"duration={self.transition_duration_s:.2f} s"
                )
            self.previous_local = (x, y, yaw)
            if now - self.transition_start_s < self.transition_duration_s:
                return
            self.active_mode = mode
            self.active_curvature = curvature
            self.total_mode_alignment_time_s += now - self.transition_start_s
            self.transition_start_s = None
            self.transition_duration_s = None
            self.segment_distance = 0.0
            self.segment_progress_reference_m = 0.0
            self.segment_progress_reference_s = now
            self.previous_local = (x, y, yaw)
            self.get_logger().info(
                f"Steering state aligned: mode={mode}, curvature={curvature:+.2f}"
            )
            return
        if mode_changed:
            previous_mode = self.active_mode
            self.active_mode = mode
            self.mode_transition_count += 1
            self.get_logger().info(
                "Rolling rate-limited mode transition: "
                f"{previous_mode} -> {mode}, curvature={curvature:+.2f}"
            )

        if abs(curvature - self.active_curvature) > 1.0e-9:
            self.get_logger().info(
                "Rolling curvature transition: "
                f"mode={mode}, {self.active_curvature:+.2f} -> {curvature:+.2f} 1/m"
            )
            self.rolling_curvature_transition_count += 1
            self.active_curvature = curvature

        if self.previous_local is not None:
            px, py, pyaw = self.previous_local
            dx, dy = x - px, y - py
            forward_increment = math.cos(pyaw) * dx + math.sin(pyaw) * dy
            self.segment_distance += direction * forward_increment
        self.previous_local = (x, y, yaw)
        if self.segment_progress_reference_s is None:
            self.segment_progress_reference_s = now
        if self.segment_distance >= self.segment_progress_reference_m + 0.0005:
            self.segment_progress_reference_m = self.segment_distance
            self.segment_progress_reference_s = now

        next_action = (
            self.scenario["actions"][self.action_index + 1]
            if self.action_index + 1 < len(self.scenario["actions"])
            else None
        )
        target_distance = (
            self.scenario.get("terminal_command_distance_m", STEP_M)
            if self.action_index + 1 == len(self.scenario["actions"])
            else STEP_M
        )
        must_stop_after_primitive = (
            next_action is None
            or (
                self.mode_transition_policy == "stopped_alignment"
                and next_action[0] != mode
            )
            or next_action[2] != direction
        )
        if self.stop_phase:
            self._stop()
            velocity = self.odom.twist.twist.linear
            planar_speed = math.hypot(velocity.x, velocity.y)
            if planar_speed <= self.stopped_speed_threshold:
                if self.stopped_within_since_s is None:
                    self.stopped_within_since_s = now
                elif now - self.stopped_within_since_s >= self.stopped_hold_s:
                    minimum_acceptable = max(
                        0.0,
                        target_distance - self.primitive_completion_tolerance,
                    )
                    if self.segment_distance >= minimum_acceptable:
                        self.get_logger().info(
                            "Stopped at primitive boundary: "
                            f"distance={self.segment_distance:.4f} m, "
                            f"speed={planar_speed:.4f} m/s"
                        )
                        self._complete_action(
                            now,
                            x,
                            y,
                            yaw,
                            carry_distance=max(
                                0.0, self.segment_distance - target_distance
                            ),
                        )
                    else:
                        self.get_logger().warning(
                            "Stopped before primitive tolerance; resuming motion: "
                            f"{self.segment_distance:.4f}/{target_distance:.4f} m"
                        )
                        if self.stop_phase_start_s is not None:
                            self.total_stopped_boundary_time_s += (
                                now - self.stop_phase_start_s
                            )
                        self.stop_phase = False
                        self.stop_phase_start_s = None
                        self.stopped_within_since_s = None
            else:
                self.stopped_within_since_s = None
            return

        # Never initiate the stopped-boundary gate before the primitive can be
        # accepted.  Otherwise a short stopping distance can leave the vehicle
        # below target-tolerance, and the next tick immediately re-enters the
        # same stop gate without ever publishing a drive command.
        stop_trigger_distance = stopped_boundary_trigger_distance(
            target_distance,
            self.stop_lead_distance,
            self.primitive_completion_tolerance,
        )
        if must_stop_after_primitive and self.segment_distance >= stop_trigger_distance:
            self.stop_phase = True
            self.stop_phase_start_s = now
            self.stop_event_count += 1
            self.stopped_within_since_s = None
            self._stop()
            self.get_logger().info(
                "Braking for mode/terminal boundary: "
                f"distance={self.segment_distance:.4f}/{target_distance:.4f} m"
            )
            return
        command_speed = action_speed
        if must_stop_after_primitive and self.braking_distance > 0.0:
            remaining = max(0.0, target_distance - self.segment_distance)
            ramped_speed = action_speed * min(
                1.0, remaining / self.braking_distance
            )
            command_speed = min(
                action_speed,
                max(self.minimum_braking_speed, ramped_speed),
            )
        command_curvature = curvature
        path_lateral_error = 0.0
        path_heading_error = 0.0
        if (
            self.path_tracking_feedback_enabled
            and self.action_index < len(self.planned_states)
        ):
            reference = self.planned_states[self.action_index]
            (
                command_curvature,
                path_lateral_error,
                path_heading_error,
            ) = path_tracking_curvature(
                curvature,
                x,
                y,
                yaw,
                reference.x,
                reference.y,
                reference.yaw,
                self.path_tracking_lateral_gain,
                self.path_tracking_heading_gain,
                self.path_tracking_max_curvature_correction,
            )
            self.maximum_abs_path_lateral_error_m = max(
                self.maximum_abs_path_lateral_error_m,
                abs(path_lateral_error),
            )
            self.maximum_abs_path_heading_error_rad = max(
                self.maximum_abs_path_heading_error_rad,
                abs(path_heading_error),
            )
            self.maximum_abs_path_curvature_correction = max(
                self.maximum_abs_path_curvature_correction,
                abs(command_curvature - curvature),
            )
        command = Twist()
        command.linear.x = direction * command_speed
        command.angular.z = command.linear.x * command_curvature
        self.cmd_pub.publish(command)
        self.rows.append(
            {
                "execution_phase": "path",
                "sim_time_s": now,
                "elapsed_s": now - self.start_s,
                "action_index": self.action_index,
                "mode": mode,
                "direction": direction,
                "curvature_1_m": command_curvature,
                "planned_curvature_1_m": curvature,
                "path_lateral_error_m": path_lateral_error,
                "path_heading_error_deg": math.degrees(path_heading_error),
                "planned_speed_m_s": action_speed,
                "command_speed_m_s": command_speed,
                "target_front_steering_deg": math.degrees(
                    self._steering_targets(mode, command_curvature)[0]
                ),
                "target_rear_steering_deg": math.degrees(
                    self._steering_targets(mode, command_curvature)[1]
                ),
                "actual_front_steering_deg": math.degrees(
                    self.actual_front_steering
                ),
                "actual_rear_steering_deg": math.degrees(
                    self.actual_rear_steering
                ),
                "segment_distance_m": self.segment_distance,
                "target_distance_m": target_distance,
                "x_m": x,
                "y_m": y,
                "yaw_deg": math.degrees(yaw),
                "footprint_min_y_m": minimum_body_y,
                "footprint_max_y_m": maximum_body_y,
                "lower_clearance_m": clearance_lower,
                "upper_clearance_m": clearance_upper,
            }
        )
        if self.last_progress_log_s is None or now - self.last_progress_log_s >= 2.0:
            self.last_progress_log_s = now
            self.get_logger().info(
                f"Progress: primitive={self.action_index + 1}/"
                f"{len(self.scenario['actions'])}, mode={mode}, "
                f"segment={self.segment_distance:.3f}/{target_distance:.3f} m, "
                f"pose=({x:.3f}, {y:.3f}, {math.degrees(yaw):.2f} deg)"
            )
        if not must_stop_after_primitive and self.segment_distance >= target_distance:
            self._complete_action(
                now,
                x,
                y,
                yaw,
                carry_distance=self.segment_distance - target_distance,
            )

    def _finish(self, x, y, yaw):
        self._stop()
        goal_x, goal_y, goal_yaw = self.scenario["goal"]
        (
            final_terminal_longitudinal_error,
            final_terminal_lateral_error,
            _,
        ) = terminal_pose_errors(
            goal_x - x,
            goal_y - y,
            yaw,
            goal_yaw,
        )
        final_linear_velocity = self.odom.twist.twist.linear
        final_planar_speed = math.hypot(
            final_linear_velocity.x, final_linear_velocity.y
        )
        task_complete_s = self.task_complete_s if self.task_complete_s is not None else stamp_seconds(self.odom)
        task_time_s = task_complete_s - self.start_s
        terminal_settle_elapsed_s = max(
            0.0, stamp_seconds(self.odom) - task_complete_s
        )
        commanded_path_length_m = sum(
            self.scenario.get("terminal_command_distance_m", STEP_M)
            if index + 1 == len(self.scenario["actions"])
            else STEP_M
            for index in range(len(self.scenario["actions"]))
        )
        nominal_translation_time_s = sum(
            (
                self.scenario.get("terminal_command_distance_m", STEP_M)
                if index + 1 == len(self.scenario["actions"])
                else STEP_M
            )
            / split_replay_action(action, self.speed)[3]
            for index, action in enumerate(self.scenario["actions"])
        )
        time_ratio_to_nominal = task_time_s / nominal_translation_time_s
        lower_min = min(sample[0] for sample in self.clearance_samples)
        upper_min = min(sample[1] for sample in self.clearance_samples)
        front_errors = [
            row["target_front_steering_deg"] - row["actual_front_steering_deg"]
            for row in self.rows
            if math.isfinite(row["actual_front_steering_deg"])
        ]
        rear_errors = [
            row["target_rear_steering_deg"] - row["actual_rear_steering_deg"]
            for row in self.rows
            if math.isfinite(row["actual_rear_steering_deg"])
        ]
        summary = {
            "scenario": self.scenario_name,
            "run_id": self.run_id,
            "experiment_condition_id": self.experiment_condition_id,
            "policy": {
                "online": "online_steering_rate_lattice",
                "topic": "map_goal_online_steering_rate_lattice",
                "replay": self.replay_policy,
            }[self.planning_mode],
            "planning_mode": self.planning_mode,
            "planning_time_s": self.planning_time_s,
            "planner_expanded_nodes": self.planner_expanded_nodes,
            "planner_robust_clearance_m": self.planner_robust_clearance,
            "speed_m_s": self.speed,
            "planned_speed_profile_m_s": [
                split_replay_action(action, self.speed)[3]
                for action in self.scenario["actions"]
            ],
            "mode_transition_settle_s": self.mode_transition_settle_s,
            "mode_transition_policy": self.mode_transition_policy,
            "braking_distance_m": self.braking_distance,
            "minimum_braking_speed_m_s": self.minimum_braking_speed,
            "primitive_completion_tolerance_m": self.primitive_completion_tolerance,
            "stop_lead_distance_m": self.stop_lead_distance,
            "effective_stop_lead_distance_m": min(
                self.stop_lead_distance,
                self.primitive_completion_tolerance,
            ),
            "stopped_speed_threshold_m_s": self.stopped_speed_threshold,
            "stopped_hold_s": self.stopped_hold_s,
            "terminal_stop_timeout_s": self.terminal_stop_timeout_s,
            "terminal_neutral_alignment_enabled": (
                self.terminal_neutral_alignment_enabled
            ),
            "terminal_neutral_alignment_duration_s": (
                self.terminal_neutral_alignment_duration_s
            ),
            "terminal_neutral_alignment_complete": (
                self.terminal_neutral_alignment_complete
            ),
            "terminal_stop_timeout_reached": self.terminal_stop_timeout_reached,
            "terminal_settle_elapsed_s": terminal_settle_elapsed_s,
            "terminal_pose_control_enabled": self.terminal_pose_control_enabled,
            "terminal_pose_controller": "bounded_2d_mode_switching",
            "terminal_pose_correction_count": self.terminal_pose_correction_count,
            "terminal_lateral_mode_switch_count": (
                self.terminal_lateral_mode_switch_count
            ),
            "maximum_abs_terminal_lateral_error_m": (
                self.maximum_abs_terminal_lateral_error_m
            ),
            "final_terminal_longitudinal_error_m": (
                final_terminal_longitudinal_error
            ),
            "final_terminal_lateral_error_m": final_terminal_lateral_error,
            "terminal_settle_latched": self.terminal_settle_latched,
            "terminal_stationary_window_duration_s": (
                self.terminal_stationary_window_duration_s
            ),
            "terminal_stationary_required_duration_s": self.settle_s,
            "terminal_stationary_window_distance_m": (
                self.terminal_stationary_window_distance_m
            ),
            "terminal_stationary_average_speed_m_s": (
                self.terminal_stationary_average_speed_m_s
            ),
            "terminal_control_position_tolerance_m": (
                self.terminal_control_position_tolerance
            ),
            "terminal_control_target_tolerance_m": (
                self.terminal_control_target_tolerance
            ),
            "terminal_control_heading_tolerance_deg": math.degrees(
                self.terminal_control_heading_tolerance
            ),
            "terminal_settle_abort_position_m": (
                self.terminal_settle_abort_position
            ),
            "terminal_settle_abort_heading_deg": math.degrees(
                self.terminal_settle_abort_heading
            ),
            "terminal_control_position_gain_s_inv": (
                self.terminal_control_position_gain
            ),
            "terminal_control_velocity_gain": (
                self.terminal_control_velocity_gain
            ),
            "path_tracking_feedback_enabled": (
                self.path_tracking_feedback_enabled
            ),
            "path_tracking_lateral_gain": self.path_tracking_lateral_gain,
            "path_tracking_heading_gain": self.path_tracking_heading_gain,
            "path_tracking_max_curvature_correction": (
                self.path_tracking_max_curvature_correction
            ),
            "maximum_abs_path_lateral_error_m": (
                self.maximum_abs_path_lateral_error_m
            ),
            "maximum_abs_path_heading_error_deg": math.degrees(
                self.maximum_abs_path_heading_error_rad
            ),
            "maximum_abs_path_curvature_correction": (
                self.maximum_abs_path_curvature_correction
            ),
            "steering_rate_limit_deg_s": STEERING_RATE_DEG_S,
            "planner_mode_curvature_response_scale": MODE_CURVATURE_RESPONSE_SCALE,
            "planner_mode_lateral_response_scale": MODE_LATERAL_RESPONSE_SCALE,
            "steering_state_alignment": (
                "initial_preview_then_rolling_rate_limited_modes"
                if self.mode_transition_policy == "rolling_rate_limited"
                else "mode_stop_preview_rolling_curvature"
            ),
            "terminal_command_distance_m": self.scenario.get(
                "terminal_command_distance_m", STEP_M
            ),
            "alignment_preview_speed_m_s": self.alignment_preview_speed,
            "mode_transition_count": self.mode_transition_count,
            "rolling_curvature_transition_count": self.rolling_curvature_transition_count,
            "stop_event_count": self.stop_event_count,
            "stopped_boundary_time_s": self.total_stopped_boundary_time_s,
            "mode_alignment_time_s": self.total_mode_alignment_time_s,
            "commanded_path_length_m": commanded_path_length_m,
            "nominal_translation_time_s": nominal_translation_time_s,
            "task_time_s": task_time_s,
            "time_ratio_to_nominal": time_ratio_to_nominal,
            "front_steering_tracking_rmse_deg": math.sqrt(
                sum(error * error for error in front_errors) / len(front_errors)
            ) if front_errors else math.nan,
            "rear_steering_tracking_rmse_deg": math.sqrt(
                sum(error * error for error in rear_errors) / len(rear_errors)
            ) if rear_errors else math.nan,
            "front_steering_tracking_max_abs_deg": max(
                (abs(error) for error in front_errors), default=math.nan
            ),
            "rear_steering_tracking_max_abs_deg": max(
                (abs(error) for error in rear_errors), default=math.nan
            ),
            "completed_actions": self.action_index,
            "total_actions": len(self.scenario["actions"]),
            "collision_free_virtual_walls": lower_min >= 0.0 and upper_min >= 0.0,
            "minimum_lower_clearance_m": lower_min,
            "minimum_upper_clearance_m": upper_min,
            "final_x_m": x,
            "final_y_m": y,
            "final_yaw_deg": math.degrees(yaw),
            "final_planar_speed_m_s": final_planar_speed,
            "goal_position_error_m": math.hypot(x - goal_x, y - goal_y),
            "goal_heading_error_deg": math.degrees(wrap(yaw - goal_yaw)),
            "analytic_lattice_position_tolerance_m": GOAL_POSITION_TOLERANCE_M,
            "analytic_lattice_heading_tolerance_deg": GOAL_HEADING_TOLERANCE_DEG,
            "position_tolerance_pass": math.hypot(x - goal_x, y - goal_y)
            <= GOAL_POSITION_TOLERANCE_M,
            "heading_tolerance_pass": abs(
                math.degrees(wrap(yaw - goal_yaw))
            )
            <= GOAL_HEADING_TOLERANCE_DEG,
        }
        output_dir = self.output_dir
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.scenario_name}_{self.run_id}"
        with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        (output_dir / f"{stem}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self.get_logger().info(
            "SPATIAL UTILITY REPLAY RESULT: "
            f"collision_free={summary['collision_free_virtual_walls']}, "
            f"position_error={summary['goal_position_error_m']:.4f} m, "
            f"heading_error={summary['goal_heading_error_deg']:.3f} deg, "
            f"clearance=({lower_min:.4f}, {upper_min:.4f}) m, "
            f"task_time={task_time_s:.2f} s ({time_ratio_to_nominal:.2f}x nominal)"
        )
        self.get_logger().info(f"Saved: {output_dir / (stem + '.json')}")
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = SpatialUtilityReplay()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
