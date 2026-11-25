#!/usr/bin/env python3
"""Track the frozen robust full-dynamics plan through Nav2 FollowPath.

AMCL anchors the local plan in map coordinates. Nav2's controller tracks it,
while this adapter applies the plan's steering-mode and speed schedule. The
Nav2 local costmap consumes the live CPU-PhysX LaserScan for collision checks.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import json
import math
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import FollowPath
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, Float64, String
from tf2_ros import Buffer, TransformException, TransformListener

from .steering_lattice import (
    FRONT_LENGTH_M,
    REAR_LENGTH_M,
    STEERING_TRACKING_BIAS_RAD,
    WHEELBASE_M,
    steering_targets,
    wrap,
)
from .spatial_utility_replay import (
    terminal_correction_direction,
    terminal_lateral_correction,
    terminal_pose_errors,
)


WORKSPACE = _ijcas_workspace()
DEFAULT_PLAN = (
    _ijcas_path(WORKSPACE, 'studies/IJCAS/results/rigid_body_edge_gate/corridor_90_hierarchical_rigid_body_envelope.json')
)
GOAL_LOCAL = (1.50, 2.20, math.radians(90.0))
VALID_MODES = {"coordinated_counterphase", "front_only", "rear_only"}


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def transform_local_pose(origin, local_x, local_y, local_yaw):
    x0, y0, yaw0 = origin
    return (
        x0 + math.cos(yaw0) * local_x - math.sin(yaw0) * local_y,
        y0 + math.sin(yaw0) * local_x + math.cos(yaw0) * local_y,
        wrap(yaw0 + local_yaw),
    )


def pose_distance(pose, target):
    return math.hypot(pose[0] - target[0], pose[1] - target[1])


def path_tracking_deviation(pose, reference):
    """Return position and body-yaw error to a reference state."""
    return pose_distance(pose, reference), abs(wrap(pose[2] - reference[2]))


def planner_curvature_with_yaw_feedback(
    feedforward_curvature,
    current_yaw,
    reference_yaw,
    yaw_gain,
    maximum_abs_curvature,
):
    """Apply bounded body-yaw feedback around the planner curvature."""
    corrected = feedforward_curvature + yaw_gain * wrap(
        reference_yaw - current_yaw
    )
    return max(-maximum_abs_curvature, min(maximum_abs_curvature, corrected))


class ProposedNav2PathRunner(Node):
    def __init__(self):
        super().__init__("proposed_nav2_path_runner")
        self.declare_parameter("run_id", "nav2_proposed_90turn_r01")
        self.declare_parameter("plan_path", str(DEFAULT_PLAN))
        self.declare_parameter("maximum_run_time_s", 150.0)
        self.declare_parameter("initial_steering_tolerance_deg", 3.0)
        self.declare_parameter("initial_alignment_hold_s", 0.5)
        self.declare_parameter("steering_raw_zero_rad", 0.0)
        self.declare_parameter("tracking_abort_position_error_m", 0.12)
        self.declare_parameter("tracking_abort_yaw_error_deg", 12.0)
        self.declare_parameter("scan_stale_timeout_s", 1.0)
        self.declare_parameter("body_yaw_curvature_gain", 0.5)
        self.declare_parameter("maximum_body_curvature_1_m", 0.775)
        self.declare_parameter("terminal_control_max_speed_m_s", 0.010)
        self.declare_parameter("terminal_control_min_speed_m_s", 0.003)
        self.declare_parameter("terminal_control_position_gain_s_inv", 0.5)
        self.declare_parameter("terminal_control_velocity_gain", 0.8)

        self.run_id = str(self.get_parameter("run_id").value)
        self.plan_path = Path(str(self.get_parameter("plan_path").value))
        self.maximum_run_time_s = float(
            self.get_parameter("maximum_run_time_s").value
        )
        self.initial_steering_tolerance = math.radians(
            float(self.get_parameter("initial_steering_tolerance_deg").value)
        )
        self.initial_alignment_hold_s = float(
            self.get_parameter("initial_alignment_hold_s").value
        )
        self.steering_raw_zero_rad = float(
            self.get_parameter("steering_raw_zero_rad").value
        )
        self.tracking_abort_position_error_m = float(
            self.get_parameter("tracking_abort_position_error_m").value
        )
        self.tracking_abort_yaw_error_rad = math.radians(
            float(self.get_parameter("tracking_abort_yaw_error_deg").value)
        )
        self.scan_stale_timeout_s = float(
            self.get_parameter("scan_stale_timeout_s").value
        )
        self.body_yaw_curvature_gain = float(
            self.get_parameter("body_yaw_curvature_gain").value
        )
        self.maximum_body_curvature = float(
            self.get_parameter("maximum_body_curvature_1_m").value
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
        if min(
            self.tracking_abort_position_error_m,
            self.tracking_abort_yaw_error_rad,
            self.scan_stale_timeout_s,
            self.maximum_body_curvature,
        ) <= 0.0:
            raise ValueError("tracking abort and scan freshness limits must be positive")
        if self.body_yaw_curvature_gain < 0.0:
            raise ValueError("body_yaw_curvature_gain must be non-negative")
        if not (
            0.0 < self.terminal_control_min_speed
            <= self.terminal_control_max_speed
            and self.terminal_control_position_gain > 0.0
            and self.terminal_control_velocity_gain >= 0.0
        ):
            raise ValueError("invalid bounded terminal-controller parameters")

        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        if (
            payload.get("scenario") != "corridor_90_turn"
            or not payload.get("plan_success")
            or payload.get("selected_route_id") != "proposed_full_dynamics"
        ):
            raise ValueError(
                "plan_path must contain the successful corridor_90_turn "
                "proposed_full_dynamics route"
            )
        self.actions = payload["selected_actions"]
        self.states = payload["selected_states"]
        if not self.actions or len(self.actions) != len(self.states):
            raise ValueError("selected action/state arrays must be nonempty and equal length")
        for action in self.actions:
            if action.get("mode") not in VALID_MODES:
                raise ValueError(f"Unsupported steering mode in plan: {action}")
            if float(action.get("target_speed_m_s", 0.0)) <= 0.0:
                raise ValueError(f"Invalid speed state in plan: {action}")

        self.action_client = ActionClient(self, FollowPath, "/follow_path")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.mode_pub = self.create_publisher(String, "/sim/steering_configuration", 10)
        self.speed_limit_pub = self.create_publisher(SpeedLimit, "/speed_limit", 10)
        self.emergency_stop_pub = self.create_publisher(Bool, "/sim/emergency_stop", 10)
        self.curvature_pub = self.create_publisher(
            Float64, "/sim/planned_curvature_reference", 10
        )
        self.curvature_enable_pub = self.create_publisher(
            Bool, "/sim/planned_curvature_reference_enabled", 10
        )
        self.terminal_cmd_pub = self.create_publisher(
            Twist, "/sim/terminal_cmd_vel", 10
        )
        self.terminal_cmd_enable_pub = self.create_publisher(
            Bool, "/sim/terminal_cmd_vel_enabled", 10
        )
        # A previous aborted trial may have left the controller stop latched.
        # Clear it before the runner can send a path; the platform is expected
        # to be stationary at this preflight stage.
        clear_stop = Bool()
        clear_stop.data = False
        self.emergency_stop_pub.publish(clear_stop)
        curvature_disabled = Bool()
        curvature_disabled.data = False
        self.curvature_enable_pub.publish(curvature_disabled)
        terminal_disabled = Bool()
        terminal_disabled.data = False
        self.terminal_cmd_enable_pub.publish(terminal_disabled)
        # Humble AMCL publishes PoseWithCovarianceStamped.
        from geometry_msgs.msg import PoseWithCovarianceStamped

        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_callback, 10
        )
        self.create_subscription(Odometry, "/sim/ground_truth/odom", self._odom_callback, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.create_subscription(Twist, "/cmd_vel_nav", self._cmd_callback, 10)
        self.create_subscription(
            String, "/sim/steering_configuration/status", self._status_callback, 10
        )

        self.origin = None
        self.current_pose = None
        self.odom = None
        self.longitudinal_velocity_m_s = 0.0
        self.scan_count = 0
        self.last_scan_time_s = None
        self.joint_angles = None
        self.latest_cmd = Twist()
        self.latest_status = ""
        self.world_states = None
        self.world_goal = None
        self.last_path_index = 0
        self.last_action_index = -1
        self.commanded_planner_curvature = 0.0
        self.terminal_closure_active = False
        self.terminal_correction_direction = 0
        self.terminal_lateral_mode = ""
        self.terminal_lateral_mode_switch_count = 0
        self.terminal_correction_count = 0
        self.goal_handle = None
        self.goal_sent = False
        self.finished = False
        self.result_status = None
        self.prealignment_started_s = None
        self.aligned_since_s = None
        self.last_readiness_log_s = None
        self.start_time_s = None
        self.feedback_distance_m = math.inf
        self.abort_reason = None
        self.abort_started_s = None
        self.rows = []
        self.create_timer(0.05, self._tick)

        self.get_logger().info(
            f"Ready to FollowPath frozen full-dynamics plan: "
            f"run_id={self.run_id}, actions={len(self.actions)}, "
            f"route={payload['selected_route_id']}, plan={self.plan_path}"
        )

    def _amcl_callback(self, message):
        pose = message.pose.pose
        self._set_current_pose((
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        ))

    def _set_current_pose(self, pose):
        self.current_pose = pose
        if self.origin is None:
            self.origin = pose
            self.world_states = [
                transform_local_pose(
                    self.origin,
                    float(state["x_m"]),
                    float(state["y_m"]),
                    math.radians(float(state["yaw_deg"])),
                )
                for state in self.states
            ]
            self.world_goal = transform_local_pose(self.origin, *GOAL_LOCAL)

    def _refresh_pose_from_tf(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", Time()
            ).transform
        except TransformException:
            return False
        translation = transform.translation
        rotation = transform.rotation
        self._set_current_pose(
            (
                float(translation.x),
                float(translation.y),
                yaw_from_quaternion(rotation),
            )
        )
        return True

    def _odom_callback(self, message):
        pose = message.pose.pose
        self.odom = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )
        self.longitudinal_velocity_m_s = float(message.twist.twist.linear.x)
        if self.goal_sent:
            now = self.get_clock().now().nanoseconds / 1.0e9
            current = self.current_pose or self.odom
            path_index = self._nearest_path_index(current)
            action_index = min(path_index, len(self.actions) - 1)
            action = self.actions[action_index]
            self.rows.append(
                {
                    "elapsed_s": now - (self.start_time_s or now),
                    "amcl_x_m": current[0],
                    "amcl_y_m": current[1],
                    "amcl_yaw_deg": math.degrees(current[2]),
                    "odom_x_m": self.odom[0],
                    "odom_y_m": self.odom[1],
                    "odom_yaw_deg": math.degrees(self.odom[2]),
                    "nearest_plan_state": max(-1, path_index - 1),
                    "active_action": action_index,
                    "planned_mode": action["mode"],
                    "planned_speed_m_s": float(action["target_speed_m_s"]),
                    "cmd_linear_m_s": self.latest_cmd.linear.x,
                    "cmd_angular_rad_s": self.latest_cmd.angular.z,
                    "planner_curvature_command_1_m": self.commanded_planner_curvature,
                    "scan_count": self.scan_count,
                }
            )

    def _scan_callback(self, message):
        self.scan_count += 1
        # Use local receipt time: some Isaac sensor graphs stamp scans with a
        # different clock epoch than the ROS node, so comparing header stamps
        # to node time can falsely classify a live scan as stale.
        self.last_scan_time_s = self.get_clock().now().nanoseconds / 1.0e9

    def _joint_callback(self, message):
        index = {name: i for i, name in enumerate(message.name)}
        front_name = "base_link_to_front_active_steering"
        rear_name = "base_link_to_rear_active_steering"
        if front_name not in index or rear_name not in index:
            return
        if len(message.position) <= max(index[front_name], index[rear_name]):
            return
        self.joint_angles = (
            message.position[index[front_name]] - self.steering_raw_zero_rad,
            message.position[index[rear_name]] - self.steering_raw_zero_rad,
        )

    def _cmd_callback(self, message):
        self.latest_cmd = message

    def _status_callback(self, message):
        self.latest_status = message.data

    def _nearest_path_index(self, pose):
        if not self.world_states or self.origin is None:
            return 0
        path_points = [self.origin, *self.world_states]
        first = max(0, self.last_path_index - 1)
        distances = [
            (pose_distance(pose, point), index)
            for index, point in enumerate(path_points)
            if index >= first
        ]
        selected = min(distances)[1]
        self.last_path_index = max(self.last_path_index, selected)
        return self.last_path_index

    def _publish_action_constraints(self, action_index):
        action_index = max(0, min(action_index, len(self.actions) - 1))
        if action_index == self.last_action_index:
            return
        action = self.actions[action_index]
        mode = String()
        mode.data = str(action["mode"])
        self.mode_pub.publish(mode)
        speed = SpeedLimit()
        speed.header.stamp = self.get_clock().now().to_msg()
        speed.percentage = False
        speed.speed_limit = float(action["target_speed_m_s"])
        self.speed_limit_pub.publish(speed)
        self.last_action_index = action_index
        self.get_logger().info(
            f"Plan schedule action {action_index + 1}/{len(self.actions)}: "
            f"mode={mode.data}, speed_limit={speed.speed_limit:.3f} m/s"
        )

    def _publish_planner_curvature(self, action_index, reference):
        action_index = max(0, min(action_index, len(self.actions) - 1))
        feedforward = float(self.actions[action_index]["curvature_1_m"])
        current_yaw = self.current_pose[2] if self.current_pose is not None else reference[2]
        command = planner_curvature_with_yaw_feedback(
            feedforward_curvature=feedforward,
            current_yaw=current_yaw,
            reference_yaw=reference[2],
            yaw_gain=self.body_yaw_curvature_gain,
            maximum_abs_curvature=self.maximum_body_curvature,
        )
        message = Float64()
        message.data = command
        self.curvature_pub.publish(message)
        enabled = Bool()
        enabled.data = True
        self.curvature_enable_pub.publish(enabled)
        self.commanded_planner_curvature = command

    def _start_terminal_closure(self):
        if self.terminal_closure_active:
            return
        self.terminal_closure_active = True
        disabled = Bool()
        disabled.data = False
        self.curvature_enable_pub.publish(disabled)
        enabled = Bool()
        enabled.data = True
        self.terminal_cmd_enable_pub.publish(enabled)
        self.terminal_correction_count += 1
        self.get_logger().info(
            "Planner endpoint reached; switching to bounded 2-D mode-switching "
            "terminal controller while Nav2 FollowPath and scan safety remain active"
        )

    def _publish_terminal_command(self):
        dx = self.world_goal[0] - self.current_pose[0]
        dy = self.world_goal[1] - self.current_pose[1]
        position_error = math.hypot(dx, dy)
        longitudinal_error, lateral_error, heading_error = terminal_pose_errors(
            dx, dy, self.current_pose[2], self.world_goal[2]
        )
        self.terminal_correction_direction = terminal_correction_direction(
            longitudinal_error,
            self.terminal_correction_direction,
            0.015,
        )
        direction = self.terminal_correction_direction
        directed_velocity = direction * self.longitudinal_velocity_m_s
        speed_magnitude = max(
            0.0,
            min(
                self.terminal_control_max_speed,
                self.terminal_control_position_gain * position_error
                - self.terminal_control_velocity_gain * directed_velocity,
            ),
        )
        if (
            position_error > 0.015
            and abs(self.longitudinal_velocity_m_s) < 0.0025
            and speed_magnitude < self.terminal_control_min_speed
        ):
            speed_magnitude = self.terminal_control_min_speed
        command_speed = direction * speed_magnitude
        if abs(lateral_error) > 0.0025:
            mode, curvature = terminal_lateral_correction(
                lateral_error,
                heading_error,
                direction,
                self.terminal_lateral_mode,
                math.radians(0.15),
            )
            if mode != self.terminal_lateral_mode:
                self.terminal_lateral_mode_switch_count += 1
            self.terminal_lateral_mode = mode
        else:
            mode = "coordinated_counterphase"
            self.terminal_lateral_mode = ""
            denominator = math.copysign(
                max(abs(longitudinal_error), 0.05),
                longitudinal_error,
            )
            curvature = max(-0.70, min(0.70, heading_error / denominator))
        mode_message = String()
        mode_message.data = mode
        self.mode_pub.publish(mode_message)
        command = Twist()
        command.linear.x = command_speed
        command.angular.z = command_speed * curvature
        self.terminal_cmd_pub.publish(command)
        self.commanded_planner_curvature = curvature

    def _initial_steering_target(self):
        action = self.actions[0]
        front, rear = steering_targets(
            action["mode"], float(action["curvature_1_m"])
        )
        return (
            front + STEERING_TRACKING_BIAS_RAD["front"],
            rear + STEERING_TRACKING_BIAS_RAD["rear"],
        )

    def _initial_steering_aligned(self):
        if self.joint_angles is None:
            return False
        target_front, target_rear = self._initial_steering_target()
        return (
            abs(self.joint_angles[0] - target_front) <= self.initial_steering_tolerance
            and abs(self.joint_angles[1] - target_rear) <= self.initial_steering_tolerance
        )

    def _build_path(self):
        path = NavPath()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        # The lattice endpoint is accepted within its own 25 mm planning
        # tolerance, whereas the execution gate is measured against the exact
        # task goal. Append that exact goal as a short terminal-closure segment
        # so Nav2 cannot report success at the approximate lattice endpoint.
        poses = [self.origin, *self.world_states, self.world_goal]
        for x, y, yaw in poses:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path

    def _tick(self):
        if self.finished:
            return
        now_s = self.get_clock().now().nanoseconds / 1.0e9
        if self.abort_started_s is not None:
            # Keep this node alive briefly so the hard-stop and action-cancel
            # messages have time to reach their subscribers before shutdown.
            if now_s - self.abort_started_s >= 0.5:
                self._finish()
            return
        self._refresh_pose_from_tf()
        if not self.goal_sent:
            ready = (
                self.origin is not None
                and self.current_pose is not None
                and self.odom is not None
                and self.joint_angles is not None
                and self.scan_count >= 3
                and self.action_client.server_is_ready()
            )
            if not ready:
                if (
                    self.last_readiness_log_s is None
                    or now_s - self.last_readiness_log_s >= 3.0
                ):
                    self.last_readiness_log_s = now_s
                    self.get_logger().info(
                        "Waiting for inputs: "
                        f"localized_pose(map<-base_link)={self.current_pose is not None}, "
                        f"odom={self.odom is not None}, "
                        f"steering_joints={self.joint_angles is not None}, "
                        f"scan_samples={self.scan_count}/3, "
                        f"follow_path_server={self.action_client.server_is_ready()}"
                    )
                return
            if self.prealignment_started_s is None:
                self.prealignment_started_s = now_s
                self._publish_action_constraints(0)
                self._publish_planner_curvature(0, self.world_states[0])
                self.get_logger().info(
                    "Initial planned mode/speed selected. The platform controller's "
                    "configuration-adaptive alignment will handle the starting "
                    "steering transient."
                )
                return
            if now_s - self.prealignment_started_s < self.initial_alignment_hold_s:
                return
            self._send_path()
            return

        if self.start_time_s is not None and now_s - self.start_time_s > self.maximum_run_time_s:
            self.get_logger().error("Proposed Nav2 path timed out; canceling FollowPath")
            self._abort_run("maximum_run_time")
            return

        path_index = self._nearest_path_index(self.current_pose)
        if path_index >= len(self.actions):
            self._start_terminal_closure()
        if self.start_time_s is not None:
            scan_age = (
                math.inf
                if self.last_scan_time_s is None
                else now_s - self.last_scan_time_s
            )
            if scan_age > self.scan_stale_timeout_s:
                self._abort_run(f"stale_scan:{scan_age:.3f}s")
                return

            if self.terminal_closure_active:
                reference = self.world_goal
            else:
                reference_index = max(
                    0, min(path_index - 1, len(self.world_states) - 1)
                )
                reference = self.world_states[reference_index]
            position_error, yaw_error = path_tracking_deviation(
                self.current_pose, reference
            )
            if (
                position_error > self.tracking_abort_position_error_m
                or yaw_error > self.tracking_abort_yaw_error_rad
            ):
                self._abort_run(
                    "path_tracking_deviation:"
                    f"{position_error:.3f}m/{math.degrees(yaw_error):.1f}deg"
                )
                return
        # Path point 0 is the start. Point k>0 is the endpoint of action k-1;
        # the next edge therefore has action index k.
        self._publish_action_constraints(min(path_index, len(self.actions) - 1))
        if not self.terminal_closure_active:
            self._publish_planner_curvature(
                min(path_index, len(self.actions) - 1), reference
            )
        else:
            self._publish_terminal_command()

    def _abort_run(self, reason):
        if self.finished:
            return
        self.abort_reason = str(reason)
        self.get_logger().error(
            f"Closed-loop safety abort ({self.abort_reason}); imposing zero speed "
            "limit and canceling FollowPath"
        )
        stop = SpeedLimit()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.percentage = False
        stop.speed_limit = 0.0
        self.speed_limit_pub.publish(stop)
        hard_stop = Bool()
        hard_stop.data = True
        self.emergency_stop_pub.publish(hard_stop)
        terminal_stop = Twist()
        self.terminal_cmd_pub.publish(terminal_stop)
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.result_status = GoalStatus.STATUS_CANCELED
        self.abort_started_s = self.get_clock().now().nanoseconds / 1.0e9

    def _send_path(self):
        goal = FollowPath.Goal()
        goal.path = self._build_path()
        goal.controller_id = "FollowPath"
        goal.goal_checker_id = "goal_checker"
        self.start_time_s = self.get_clock().now().nanoseconds / 1.0e9
        self.goal_sent = True
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(
            f"Sending proposed full-dynamics path to Nav2 FollowPath: "
            f"{len(goal.path.poses)} poses, map-frame goal=({self.world_goal[0]:.3f}, "
            f"{self.world_goal[1]:.3f}, {math.degrees(self.world_goal[2]):.2f} deg)"
        )

    def _feedback_callback(self, feedback_message):
        self.feedback_distance_m = float(
            feedback_message.feedback.distance_to_goal
        )

    def _goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().error("Nav2 rejected proposed FollowPath path")
            self.result_status = GoalStatus.STATUS_ABORTED
            self._finish()
            return
        self.get_logger().info("Nav2 accepted proposed FollowPath path")
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self.result_status = future.result().status
        self._finish()

    def _finish(self):
        if self.finished:
            return
        terminal_stop = Twist()
        self.terminal_cmd_pub.publish(terminal_stop)
        terminal_disabled = Bool()
        terminal_disabled.data = False
        self.terminal_cmd_enable_pub.publish(terminal_disabled)
        self.finished = True
        final_pose = self.current_pose
        position_error = (
            pose_distance(final_pose, self.world_goal)
            if final_pose is not None and self.world_goal is not None
            else None
        )
        heading_error = (
            math.degrees(wrap(final_pose[2] - self.world_goal[2]))
            if final_pose is not None and self.world_goal is not None
            else None
        )
        status_name = {
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
        }.get(self.result_status, "NOT_STARTED")
        summary = {
            "run_id": self.run_id,
            "scenario": "corridor_90_turn",
            "status": status_name,
            "success": self.result_status == GoalStatus.STATUS_SUCCEEDED,
            "planner_is_proposed_full_dynamics": True,
            "planner_path": str(self.plan_path),
            "route_id": "proposed_full_dynamics",
            "actions": len(self.actions),
            "follow_path_poses": len(self.actions) + 2,
            "exact_terminal_goal_appended": True,
            "steering_modes": sorted({str(a["mode"]) for a in self.actions}),
            "steering_mode_switches": sum(
                a["mode"] != b["mode"]
                for a, b in zip(self.actions, self.actions[1:])
            ),
            "speed_profile_m_s": [
                float(action["target_speed_m_s"]) for action in self.actions
            ],
            "sensor_model": "CPU PhysX 360-degree raycast LaserScan",
            "localized_pose_source": "AMCL pose with map-to-base_link TF fallback",
            "scan_samples": self.scan_count,
            "initial_steering_prealigned_before_start": self._initial_steering_aligned(),
            "local_costmap_live_scan_collision_check": True,
            "closed_loop_abort_reason": self.abort_reason,
            "stop_latched": self.abort_reason is not None,
            "emergency_stop_topic": "/sim/emergency_stop",
            "planned_curvature_topic": "/sim/planned_curvature_reference",
            "planned_curvature_enable_topic": (
                "/sim/planned_curvature_reference_enabled"
            ),
            "terminal_closure_active": self.terminal_closure_active,
            "terminal_pose_controller": "bounded_2d_mode_switching",
            "terminal_command_topic": "/sim/terminal_cmd_vel",
            "terminal_correction_count": self.terminal_correction_count,
            "terminal_lateral_mode_switch_count": (
                self.terminal_lateral_mode_switch_count
            ),
            "body_yaw_curvature_gain": self.body_yaw_curvature_gain,
            "maximum_body_curvature_1_m": self.maximum_body_curvature,
            "tracking_abort_position_error_m": self.tracking_abort_position_error_m,
            "tracking_abort_yaw_error_deg": math.degrees(
                self.tracking_abort_yaw_error_rad
            ),
            "scan_stale_timeout_s": self.scan_stale_timeout_s,
            "nav2_controller": (
                "RegulatedPurePursuitController / FollowPath with proposed "
                "body-curvature actuation and bounded_2d_mode_switching "
                "terminal override"
            ),
            "duration_s": (
                self.rows[-1]["elapsed_s"] if self.rows else 0.0
            ),
            "recorded_samples": len(self.rows),
            "final_goal_error_m": position_error,
            "final_goal_yaw_error_deg": heading_error,
            "final_follow_path_feedback_distance_m": (
                self.feedback_distance_m
                if math.isfinite(self.feedback_distance_m)
                else None
            ),
            "terminal_tolerance_m": 0.030,
            "terminal_heading_tolerance_deg": 2.0,
        }
        summary["pass"] = (
            summary["success"]
            and position_error is not None
            and position_error <= summary["terminal_tolerance_m"]
            and heading_error is not None
            and abs(heading_error) <= summary["terminal_heading_tolerance_deg"]
        )
        output_dir = _ijcas_path(WORKSPACE, 'results/nav2_initial_motion')
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self.run_id}.json"
        json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if self.rows:
            csv_path = output_dir / f"{self.run_id}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        self.get_logger().info(
            "PROPOSED NAV2 PATH RESULT: "
            f"status={status_name}, contacts are finalized separately, "
            f"goal={position_error if position_error is not None else float('nan'):.4f} m, "
            f"yaw={heading_error if heading_error is not None else float('nan'):.3f} deg, "
            f"VERDICT={'PASS' if summary['pass'] else 'FAIL'}"
        )
        self.get_logger().info(f"Saved: {json_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ProposedNav2PathRunner()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
