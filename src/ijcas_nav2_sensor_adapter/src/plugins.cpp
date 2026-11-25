// Simulation integration only. No ROS publisher here connects to an actuator.
// The external controller is the frozen common multi-steering implementation;
// Nav2 actually executes this controller plugin, checks its local footprint,
// publishes the accepted command, and owns the FollowPath action lifecycle.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include "ijcas_nav2_sensor_adapter/contracts.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_core/goal_checker.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace ijcas_nav2_sensor_adapter {
template<class T> T parameter(const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  const std::string & name, const T & value)
{
  if (!node->has_parameter(name)) {node->declare_parameter<T>(name, value);}
  return node->get_parameter(name).get_value<T>();
}
void owned_topic(const std::string & value)
{
  if (value.rfind("/ijcas_sensor_dev/", 0) != 0 || value.find(' ') != std::string::npos) {
    throw std::invalid_argument("Explicit isolated simulation topic required");
  }
}
double yaw_of(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

class MeasuredPlatformController : public nav2_core::Controller {
public:
  void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer>, std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap) override
  {
    node_ = parent.lock(); costmap_ = costmap; name_ = name;
    if (!node_ || !costmap_) {throw std::invalid_argument("Actual controller node/costmap required");}
    auto input = parameter<std::string>(node_, name + ".proposal_topic", "");
    auto ack = parameter<std::string>(node_, name + ".acknowledgement_topic", "");
    auto pose = parameter<std::string>(node_, name + ".reported_pose_topic", "");
    owned_topic(input); owned_topic(ack); owned_topic(pose);
    max_age_ = parameter<double>(node_, name + ".maximum_proposal_age_s", .25);
    horizon_ = parameter<double>(node_, name + ".collision_projection_s", .25);
    if (parameter<std::string>(node_, name + ".footprint_guard_version", "") !=
      "CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1") {
      throw std::invalid_argument("Explicit original-cell/full-polygon guard contract required");
    }
    if (max_age_ != .25 || horizon_ != .25) {throw std::invalid_argument("Fixed bounded proposal/collision horizon required");}
    sub_ = node_->create_subscription<geometry_msgs::msg::TwistStamped>(input, 10,
      [this](geometry_msgs::msg::TwistStamped::SharedPtr value) {
        std::lock_guard<std::mutex> lock(mutex_); proposal_ = *value; received_ = true;
      });
    ack_ = node_->create_publisher<geometry_msgs::msg::TwistStamped>(ack, 10);
    pose_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(pose, 10);
  }
  void cleanup() override {sub_.reset(); ack_.reset(); pose_.reset(); node_.reset(); costmap_.reset();}
  void activate() override {ack_->on_activate(); pose_->on_activate();}
  void deactivate() override {ack_->on_deactivate(); pose_->on_deactivate();}
  void setPlan(const nav_msgs::msg::Path & path) override
  {
    if (path.poses.size() < 2 || path.poses.size() > 20000 || path.header.frame_id.empty()) {
      throw std::invalid_argument("Explicit bounded frozen P path required");
    }
    if (plan_received_) {throw nav2_core::PlannerException("ONE original path only, no replacement or retry");}
    plan_received_ = true;
  }
  geometry_msgs::msg::TwistStamped computeVelocityCommands(const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist &, nav2_core::GoalChecker *) override
  {
    geometry_msgs::msg::TwistStamped command;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!received_) {command.header.stamp = node_->now(); command.header.frame_id = "base_link"; return command;}
      command = proposal_;
    }
    auto & t = command.twist;
    std::array<double, 6> values{t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z};
    std::array<double, 6> bounded;
    double age = (node_->now() - rclcpp::Time(command.header.stamp, RCL_ROS_TIME)).seconds();
    if (!plan_received_ || command.header.frame_id != "base_link") {
      throw nav2_core::PlannerException("Original path/base_link proposal required");
    }
    if (!std::isfinite(age) || age < 0.0 || age > max_age_) {
      throw nav2_core::PlannerException("Stale platform proposal age_s=" + std::to_string(age));
    }
    if (!bounded_proposal(values, bounded)) {
      throw nav2_core::PlannerException("Unbounded platform proposal u/v/r=" +
        std::to_string(values[0]) + "/" + std::to_string(values[1]) + "/" + std::to_string(values[5]));
    }
    t.linear.x = bounded[0]; t.linear.y = bounded[1]; t.angular.z = bounded[5];
    const double magnitude = std::max(std::abs(t.linear.x), std::abs(t.linear.y));
    double scale = magnitude > speed_limit_ ? speed_limit_ / magnitude : 1.0;
    command.twist.linear.x *= scale; command.twist.linear.y *= scale; command.twist.angular.z *= scale;
    // Sampled short-horizon KINEMATIC footprint guard, not a continuous-time
    // 3D collision or force guarantee. The original native full-body10mm and
    // actual contact/tilt/resource guards remain independently mandatory.
    with_locked_costmap(*costmap_->getCostmap(), [&](auto & locked_map) {
    double x = pose.pose.position.x, y = pose.pose.position.y, heading = yaw_of(pose.pose.orientation);
    auto footprint = costmap_->getRobotFootprint();
    if (footprint.size() < 3) {throw nav2_core::PlannerException("Actual full-body footprint required");}
    std::vector<Point2> original_polygon;
    for (const auto & point : footprint) {original_polygon.push_back({point.x, point.y});}
    for (int i = 0; i <= 5; ++i) {
      const auto decision = full_polygon_cell_guard(locked_map, x, y, heading, original_polygon);
      const double cost = decision.cost;
      if (!decision.clear) {
        // One bounded ORIGINAL raster from the SAME locked decision, not a
        // post-hoc rebuilt map. The owned ROS log is durably archived.
        std::ostringstream diagnostic; diagnostic << std::setprecision(17);
        const auto width = locked_map.getSizeInCellsX(), height = locked_map.getSizeInCellsY();
        const auto cells = static_cast<std::size_t>(width) * height;
        diagnostic << "{\"version\":\"ACTUAL_LOCKED_NAV2_COSTMAP_REJECTION_v1\","
          << "\"proposal_stamp_ns\":" << rclcpp::Time(command.header.stamp).nanoseconds()
          << ",\"pose_stamp_ns\":" << rclcpp::Time(pose.header.stamp).nanoseconds()
          << ",\"frame_id\":\"" << pose.header.frame_id << "\",\"projection_step\":" << i
          << ",\"geometry_guard_version\":\"CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1\""
          << ",\"rejected_cell_column_row\":[" << decision.column << ',' << decision.row << ']'
          << ",\"cost\":";
        if (std::isfinite(cost)) {diagnostic << cost;} else {diagnostic << "null";}
        diagnostic << ",\"projected_xyyaw\":[" << x << ',' << y << ',' << heading << ']'
          << ",\"origin_xy_m\":[" << locked_map.getOriginX() << ',' << locked_map.getOriginY() << ']'
          << ",\"resolution_m\":" << locked_map.getResolution()
          << ",\"shape_rows_columns\":[" << height << ',' << width << ']'
          << ",\"padded_footprint_xy_m\":[";
        for (std::size_t j = 0; j < footprint.size(); ++j) {
          if (j) {diagnostic << ',';}
          diagnostic << '[' << footprint[j].x << ',' << footprint[j].y << ']';
        }
        diagnostic << "],\"complete_grid_snapshot\":" << (cells <= 65536 ? "true" : "false")
          << ",\"cost_data\":[";
        if (cells <= 65536) {
          const auto data = locked_map.getCharMap();
          for (std::size_t j = 0; j < cells; ++j) {
            if (j) {diagnostic << ',';}
            diagnostic << static_cast<unsigned int>(data[j]);
          }
        }
        diagnostic << "]}";
        RCLCPP_ERROR(node_->get_logger(), "IJCAS_COSTMAP_GUARD_DIAGNOSTIC %s", diagnostic.str().c_str());
        throw nav2_core::PlannerException("Nav2 actual footprint/projection rejected: step=" +
          std::to_string(i) + " cost=" + std::to_string(cost) + " frame=" + pose.header.frame_id);
      }
      x += .05 * (std::cos(heading) * command.twist.linear.x - std::sin(heading) * command.twist.linear.y);
      y += .05 * (std::sin(heading) * command.twist.linear.x + std::cos(heading) * command.twist.linear.y);
      heading += .05 * command.twist.angular.z;
    }
    });
    ack_->publish(command);
    // Keep the ACTUAL TF query stamp; do not relabel an older pose with the
    // newer command's stamp just to make a timestamp check appear matched.
    pose_->publish(pose);
    return command;
  }
  void setSpeedLimit(const double & limit, const bool & percentage) override
  {
    if (!std::isfinite(limit) || limit < 0.0 || (percentage && limit > 100.0)) {
      throw std::invalid_argument("Finite bounded speed limit required");
    }
    speed_limit_ = limit == 0.0 ? .05 : std::min(.05, percentage ? .05 * limit / 100.0 : limit);
  }
private:
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::TwistStamped>::SharedPtr ack_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_;
  std::mutex mutex_; geometry_msgs::msg::TwistStamped proposal_;
  std::string name_; bool received_{false}, plan_received_{false};
  double max_age_{.25}, horizon_{.25}, speed_limit_{.05};
};

class DeclaredTaskGoalChecker : public nav2_core::GoalChecker {
public:
  void initialize(const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, const std::string & name,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap) override
  {
    node_ = parent.lock(); costmap_ = costmap;
    if (!node_ || !costmap_) {throw std::invalid_argument("Actual goal-checker node/costmap required");}
    auto goal = parameter<std::vector<double>>(node_, name + ".declared_task_goal_xyyaw", {});
    map_frame_ = parameter<std::string>(node_, name + ".declared_map_frame", "");
    auto done = parameter<std::string>(node_, name + ".fixed_observation_complete_topic", "");
    owned_topic(done);
    if (goal.size() != 3 || map_frame_.empty() || map_frame_[0] == '/') {throw std::invalid_argument("Exact separately assigned task/map goal required");}
    std::copy(goal.begin(), goal.end(), goal_.begin());
    for (double value : goal_) {if (!std::isfinite(value)) {throw std::invalid_argument("Finite task goal required");}}
    complete_sub_ = node_->create_subscription<std_msgs::msg::Bool>(done, 10,
      [this](std_msgs::msg::Bool::SharedPtr value) {complete_ = value->data;});
  }
  void reset() override {complete_ = false;}
  bool isGoalReached(const geometry_msgs::msg::Pose & query, const geometry_msgs::msg::Pose &,
    const geometry_msgs::msg::Twist & velocity) override
  {
    if (!complete_) {return false;}
    geometry_msgs::msg::PoseStamped original, mapped;
    original.header.frame_id = costmap_->getGlobalFrameID(); original.header.stamp = rclcpp::Time(0, 0, RCL_ROS_TIME);
    original.pose = query;
    try {mapped = costmap_->getTfBuffer()->transform(original, map_frame_, tf2::durationFromSec(.05));}
    catch (const tf2::TransformException &) {return false;}
    return declared_goal_reached({mapped.pose.position.x, mapped.pose.position.y, yaw_of(mapped.pose.orientation)},
      goal_, velocity.linear.x, velocity.linear.y, velocity.angular.z, complete_);
  }
  bool getTolerances(geometry_msgs::msg::Pose & pose, geometry_msgs::msg::Twist & velocity) override
  {
    pose.position.x = pose.position.y = .03; pose.position.z = std::numeric_limits<double>::lowest();
    pose.orientation.z = std::sin(pi / 180.0); pose.orientation.w = std::cos(pi / 180.0);
    velocity.linear.x = velocity.linear.y = .005; velocity.angular.z = .02;
    velocity.linear.z = velocity.angular.x = velocity.angular.y = std::numeric_limits<double>::lowest();
    return true;
  }
private:
  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr complete_sub_;
  std::array<double, 3> goal_{}; std::string map_frame_; std::atomic_bool complete_{false};
};
}  // namespace ijcas_nav2_sensor_adapter
PLUGINLIB_EXPORT_CLASS(ijcas_nav2_sensor_adapter::MeasuredPlatformController, nav2_core::Controller)
PLUGINLIB_EXPORT_CLASS(ijcas_nav2_sensor_adapter::DeclaredTaskGoalChecker, nav2_core::GoalChecker)
