"""ROS 2 adapter from OccupancyGrid/PoseStamped to a steering lattice plan."""

import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray, String

from .spatial_utility_replay import yaw_from_odometry
from .steering_lattice import BODY_LENGTH_M, BODY_WIDTH_M, plan, wrap


def yaw_from_pose(pose):
    q = pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class GridCollisionChecker:
    def __init__(self, grid, start, clearance, occupied_threshold=50):
        self.grid = grid
        self.start = start
        self.clearance = clearance
        self.occupied_threshold = occupied_threshold
        self.bucket_size = max(0.20, 10.0 * grid.info.resolution)
        self.occupied_buckets = {}
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        for row in range(grid.info.height):
            for column in range(grid.info.width):
                value = grid.data[row * grid.info.width + column]
                if value < self.occupied_threshold and value >= 0:
                    continue
                x = origin_x + (column + 0.5) * grid.info.resolution
                y = origin_y + (row + 0.5) * grid.info.resolution
                key = (math.floor(x / self.bucket_size), math.floor(y / self.bucket_size))
                self.occupied_buckets.setdefault(key, []).append((x, y))

    def local_to_world(self, x, y):
        x0, y0, yaw0 = self.start
        return (
            x0 + math.cos(yaw0) * x - math.sin(yaw0) * y,
            y0 + math.sin(yaw0) * x + math.cos(yaw0) * y,
        )

    def __call__(self, poses):
        half_length = BODY_LENGTH_M / 2.0 + self.clearance
        half_width = BODY_WIDTH_M / 2.0 + self.clearance
        radius = math.hypot(half_length, half_width)
        for local_x, local_y, local_yaw in poses:
            world_x, world_y = self.local_to_world(local_x, local_y)
            world_yaw = self.start[2] + local_yaw
            cosine, sine = math.cos(world_yaw), math.sin(world_yaw)
            min_bucket_x = math.floor((world_x - radius) / self.bucket_size)
            max_bucket_x = math.floor((world_x + radius) / self.bucket_size)
            min_bucket_y = math.floor((world_y - radius) / self.bucket_size)
            max_bucket_y = math.floor((world_y + radius) / self.bucket_size)
            for bucket_x in range(min_bucket_x, max_bucket_x + 1):
                for bucket_y in range(min_bucket_y, max_bucket_y + 1):
                    for cell_x, cell_y in self.occupied_buckets.get(
                        (bucket_x, bucket_y), ()
                    ):
                        dx, dy = cell_x - world_x, cell_y - world_y
                        body_x = cosine * dx + sine * dy
                        body_y = -sine * dx + cosine * dy
                        if abs(body_x) <= half_length and abs(body_y) <= half_width:
                            return True
        return False


class OccupancyLatticePlanner(Node):
    def __init__(self):
        super().__init__("occupancy_lattice_planner")
        self.declare_parameter("scenario", "rear_end_constrained")
        self.declare_parameter("run_id", "map_goal_r01")
        self.declare_parameter("speed_m_s", 0.03)
        self.declare_parameter("robust_clearance_m", 0.010)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("odometry_topic", "/sim/ground_truth/odom")
        self.declare_parameter("plan_topic", "/sim/ijcas/plan_command")
        self.declare_parameter("path_topic", "/sim/ijcas/planned_path")
        self.declare_parameter("planning_mode", "map_goal_online")
        self.declare_parameter("replan_period_s", 0.0)
        self.declare_parameter("occupied_threshold", 50)
        self.declare_parameter("steering_state_topic", "")
        self.declare_parameter("require_steering_state", False)
        self.declare_parameter("initial_mode", "coordinated_counterphase")
        self.scenario = str(self.get_parameter("scenario").value)
        self.run_id = str(self.get_parameter("run_id").value)
        self.speed = float(self.get_parameter("speed_m_s").value)
        self.clearance = float(self.get_parameter("robust_clearance_m").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.planning_mode = str(self.get_parameter("planning_mode").value)
        self.replan_period = float(self.get_parameter("replan_period_s").value)
        self.occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self.steering_state_topic = str(
            self.get_parameter("steering_state_topic").value
        )
        self.require_steering_state = bool(
            self.get_parameter("require_steering_state").value
        )
        self.initial_mode = str(self.get_parameter("initial_mode").value)
        if not 1 <= self.occupied_threshold <= 100:
            raise ValueError("occupied_threshold must be in [1, 100]")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.command_pub = self.create_publisher(String, self.plan_topic, qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos)
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, qos)
        self.create_subscription(Odometry, self.odometry_topic, self._on_odom, 10)
        if self.steering_state_topic:
            self.create_subscription(
                Float32MultiArray,
                self.steering_state_topic,
                self._on_steering_state,
                10,
            )
        self.grid = None
        self.goal = None
        self.odom = None
        self.last_request = None
        self.last_plan_time = None
        self.steering_deg = None
        self.create_timer(0.10, self._try_plan)
        self.get_logger().info(
            "Planner interface: "
            f"map={self.map_topic}, goal={self.goal_topic}, pose={self.odometry_topic}, "
            f"plan={self.plan_topic}, path={self.path_topic}, "
            f"replan_period={self.replan_period:.3f}s, "
            f"occupied_threshold={self.occupied_threshold}, "
            f"steering_state={self.steering_state_topic or 'prealigned-default'}"
        )

    def _on_map(self, message):
        self.grid = message

    def _on_goal(self, message):
        self.goal = message
        self.last_request = None
        self.last_plan_time = None

    def _on_odom(self, message):
        self.odom = message

    def _on_steering_state(self, message):
        if len(message.data) < 2:
            self.get_logger().error(
                "Steering state requires [front_actual_deg, rear_actual_deg, ...]",
                throttle_duration_sec=2.0,
            )
            return
        self.steering_deg = (float(message.data[0]), float(message.data[1]))

    def _try_plan(self):
        if self.grid is None or self.goal is None or self.odom is None:
            return
        if self.require_steering_state and self.steering_deg is None:
            return
        grid_frame = self.grid.header.frame_id or "map"
        goal_frame = self.goal.header.frame_id or grid_frame
        pose_frame = self.odom.header.frame_id or grid_frame
        if goal_frame != grid_frame or pose_frame != grid_frame:
            self.get_logger().error(
                "Frame mismatch; this planner does not silently transform inputs: "
                f"map={grid_frame!r}, goal={goal_frame!r}, pose={pose_frame!r}",
                throttle_duration_sec=2.0,
            )
            return
        request = (
            self.goal.header.stamp.sec,
            self.goal.header.stamp.nanosec,
            self.goal.pose.position.x,
            self.goal.pose.position.y,
        )
        now = time.monotonic()
        periodic_replan_due = (
            self.replan_period > 0.0
            and self.last_plan_time is not None
            and now - self.last_plan_time >= self.replan_period
        )
        if request == self.last_request and not periodic_replan_due:
            return
        start = (
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            yaw_from_odometry(self.odom),
        )
        dx = self.goal.pose.position.x - start[0]
        dy = self.goal.pose.position.y - start[1]
        goal = (
            math.cos(start[2]) * dx + math.sin(start[2]) * dy,
            -math.sin(start[2]) * dx + math.cos(start[2]) * dy,
            wrap(yaw_from_pose(self.goal.pose) - start[2]),
        )
        checker = GridCollisionChecker(
            self.grid, start, self.clearance, self.occupied_threshold
        )
        started = time.perf_counter()
        measured_steering = self.steering_deg or (0.0, 0.0)
        result = plan(
            goal,
            -10.0,
            10.0,
            robust_clearance=self.clearance,
            speed_m_s=self.speed,
            collision_checker=checker,
            initial_mode=self.initial_mode,
            initial_front_rad=math.radians(measured_steering[0]),
            initial_rear_rad=math.radians(measured_steering[1]),
            propagate_initial_steering=self.steering_deg is not None,
        )
        planning_time = time.perf_counter() - started
        self.last_request = request
        self.last_plan_time = now
        if result is None:
            self.get_logger().error("Occupancy lattice found no feasible path")
            return
        path = Path()
        path.header.stamp = self.odom.header.stamp
        path.header.frame_id = grid_frame
        for state in result.states:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = checker.local_to_world(state.x, state.y)
            yaw = start[2] + state.yaw
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        payload = {
            "scenario": self.scenario,
            "run_id": self.run_id,
            "planning_mode": self.planning_mode,
            "planning_time_s": planning_time,
            "expanded_nodes": result.expanded_nodes,
            "start": start,
            "goal_local": goal,
            "initial_mode": self.initial_mode,
            "measured_initial_steering_deg": measured_steering,
            "actions": [
                {"mode": mode, "curvature_1_m": curvature, "direction": direction}
                for mode, curvature, direction in result.actions
            ],
        }
        self.path_pub.publish(path)
        self.command_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
        self.get_logger().info(
            f"PLAN READY ({self.planning_mode}): actions={len(result.actions)}, "
            f"expanded={result.expanded_nodes}, time={planning_time:.4f} s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyLatticePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
