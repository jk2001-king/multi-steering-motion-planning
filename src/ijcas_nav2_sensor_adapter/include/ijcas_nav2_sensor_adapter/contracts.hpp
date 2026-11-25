#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace ijcas_nav2_sensor_adapter {
// Costmap updates/rolling-origin changes must not interleave a multi-pose
// collision decision. Same recursive mutex used by upstream Nav2 RPP.
template<class Map, class Function> decltype(auto) with_locked_costmap(Map & map, Function && function)
{
  std::unique_lock<typename Map::mutex_t> lock(*map.getMutex());
  return std::forward<Function>(function)(map);
}
constexpr double pi = 3.14159265358979323846;
using Point2 = std::array<double, 2>;
struct CellGuardDecision {
  bool clear{false};
  double cost{-1.0};
  int column{-1}, row{-1};
};

// CLOSED convex polygon versus CLOSED original cell squares. No cell centre
// approximation, perimeter rasterization, erosion, or obstacle relabelling.
// Interior occupied cells are checked too. Touching is a collision. Numerical
// allowance only EXPANDS the overlap test (never shrinks a footprint/cell).
class ClosedConvexPolygon {
public:
  explicit ClosedConvexPolygon(const std::vector<Point2> & vertices) : vertices_(vertices)
  {
    if (vertices.size() < 3 || vertices.size() > 32) {return;}
    double scale = 1.0;
    for (const auto & p : vertices) {
      for (double value : p) {
        if (!std::isfinite(value)) {return;}
        scale = std::max(scale, std::abs(value));
      }
    }
    tolerance_ = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    // Every vertex must lie on the same side of EVERY edge, excluding
    // concave, crossing, repeated-edge and degenerate polygon inputs.
    int winding = 0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      const auto & a = vertices[i]; const auto & b = vertices[(i + 1) % vertices.size()];
      double dx = b[0] - a[0], dy = b[1] - a[1], length = std::hypot(dx, dy);
      if (!std::isfinite(length) || length <= tolerance_) {return;}
      int side = 0;
      for (const auto & p : vertices) {
        const double cross = (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / length;
        const int sign = cross > tolerance_ ? 1 : (cross < -tolerance_ ? -1 : 0);
        if (sign && side && sign != side) {return;}
        if (sign) {side = sign;}
      }
      if (!side || (winding && side != winding)) {return;}
      winding = side;
      add_axis(-dy / length, dx / length);
    }
    add_axis(1.0, 0.0); add_axis(0.0, 1.0);
    bounds_ = {vertices[0][0], vertices[0][1], vertices[0][0], vertices[0][1]};
    for (const auto & p : vertices) {
      bounds_[0] = std::min(bounds_[0], p[0]); bounds_[1] = std::min(bounds_[1], p[1]);
      bounds_[2] = std::max(bounds_[2], p[0]); bounds_[3] = std::max(bounds_[3], p[1]);
    }
    valid_ = true;
  }
  bool valid() const {return valid_;}
  const std::array<double, 4> & bounds() const {return bounds_;}
  bool intersects_cell(double x, double y, double half_side) const
  {
    if (!valid_ || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(half_side)
      || half_side <= 0.0) {return true;}  // Invalid geometry is never free.
    const double allowance = tolerance_ + 64.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(x), std::abs(y), half_side});
    for (const auto & axis : axes_) {
      const double centre = axis[0] * x + axis[1] * y;
      const double radius = half_side * (std::abs(axis[0]) + std::abs(axis[1]));
      if (centre - radius > axis[3] + allowance || centre + radius < axis[2] - allowance) {
        return false;
      }
    }
    return true;
  }
private:
  void add_axis(double x, double y)
  {
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (const auto & p : vertices_) {
      const double projection = x * p[0] + y * p[1];
      low = std::min(low, projection); high = std::max(high, projection);
    }
    axes_.push_back({x, y, low, high});
  }
  std::vector<Point2> vertices_;
  std::vector<std::array<double, 4>> axes_;
  std::array<double, 4> bounds_{};
  double tolerance_{0.0}; bool valid_{false};
};

// Caller holds the SAME costmap mutex throughout all six projection poses.
template<class Map> CellGuardDecision full_polygon_cell_guard(Map & map,
  double x, double y, double yaw, const std::vector<Point2> & footprint)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw)) {return {};}
  const double resolution = map.getResolution(), ox = map.getOriginX(), oy = map.getOriginY();
  const auto width = map.getSizeInCellsX(), height = map.getSizeInCellsY();
  if (!std::isfinite(resolution) || resolution <= 0.0 || !std::isfinite(ox) || !std::isfinite(oy)
    || !width || !height || width > 100000 || height > 100000) {return {};}
  std::vector<Point2> vertices; vertices.reserve(footprint.size());
  const double c = std::cos(yaw), s = std::sin(yaw);
  for (const auto & p : footprint) {
    vertices.push_back({x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]});
  }
  const ClosedConvexPolygon polygon(vertices);
  if (!polygon.valid()) {return {};}
  const auto bounds = polygon.bounds();
  // Outside/upper map boundary remains fail-closed, never clamped to freedom.
  if (bounds[0] < ox || bounds[1] < oy || bounds[2] >= ox + width * resolution
    || bounds[3] >= oy + height * resolution) {return {};}
  // Extra boundary neighbours include every CLOSED cell touching the polygon.
  const int first_x = std::max(0, static_cast<int>(std::floor((bounds[0] - ox) / resolution)) - 1);
  const int first_y = std::max(0, static_cast<int>(std::floor((bounds[1] - oy) / resolution)) - 1);
  const int last_x = std::min(static_cast<int>(width) - 1,
    static_cast<int>(std::floor((bounds[2] - ox) / resolution)) + 1);
  const int last_y = std::min(static_cast<int>(height) - 1,
    static_cast<int>(std::floor((bounds[3] - oy) / resolution)) + 1);
  for (int row = first_y; row <= last_y; ++row) {
    for (int column = first_x; column <= last_x; ++column) {
      const auto cost = map.getCost(column, row);
      // 253 is the additional inscribed-radius inflation band, NOT a raw
      // occupied cell. The actual padded full body is already checked here.
      if (cost >= 254 && polygon.intersects_cell(ox + (column + .5) * resolution,
          oy + (row + .5) * resolution, resolution / 2.0)) {
        return {false, static_cast<double>(cost), column, row};
      }
    }
  }
  return {true, 0.0, -1, -1};
}

inline bool declared_goal_reached(const std::array<double, 3> & pose,
  const std::array<double, 3> & goal, double u, double v, double yaw_rate, bool complete)
{
  for (double value : pose) {if (!std::isfinite(value)) {return false;}}
  for (double value : goal) {if (!std::isfinite(value)) {return false;}}
  if (!complete || !std::isfinite(u) || !std::isfinite(v) || !std::isfinite(yaw_rate)) {return false;}
  const double heading = std::atan2(std::sin(pose[2] - goal[2]), std::cos(pose[2] - goal[2]));
  return std::hypot(pose[0] - goal[0], pose[1] - goal[1]) <= 0.030 &&
    std::abs(heading) <= 2.0 * pi / 180.0 && std::hypot(u, v) <= 0.005 &&
    std::abs(yaw_rate) <= 0.02;
}
inline bool command_in_contract(const std::array<double, 6> & value)
{
  for (double number : value) {if (!std::isfinite(number)) {return false;}}
  return std::abs(value[0]) <= 0.0500000001 && std::abs(value[1]) <= 0.0500000001 &&
    std::abs(value[2]) <= 1e-10 && std::abs(value[3]) <= 1e-10 &&
    std::abs(value[4]) <= 1e-10 && std::abs(value[5]) <= 8.0 * pi / 180.0;
}
// A native Float32 wheel radius can turn a nominal .05 m/s proposal into
// .050000000644 m/s. Admit only a tiny REPRESENTATIONAL input allowance,
// then uniformly reduce the WHOLE planar twist to the unchanged output
// limits. Never send the over-limit value or change its curvature.
inline bool bounded_proposal(const std::array<double, 6> & input,
  std::array<double, 6> & output)
{
  for (double number : input) {if (!std::isfinite(number)) {return false;}}
  constexpr double input_roundoff = 1e-8;
  constexpr double angular_limit = 8.0 * pi / 180.0;
  if (std::abs(input[0]) > .05 + input_roundoff ||
    std::abs(input[1]) > .05 + input_roundoff ||
    std::abs(input[5]) > angular_limit + input_roundoff ||
    std::abs(input[2]) > 1e-10 || std::abs(input[3]) > 1e-10 ||
    std::abs(input[4]) > 1e-10) {return false;}
  double scale = 1.0;
  for (int index : {0, 1, 5}) {
    double limit = index == 5 ? angular_limit : .05;
    if (std::abs(input[index]) > limit) {
      scale = std::min(scale, limit / std::abs(input[index]));
    }
  }
  output = input;
  for (int index : {0, 1, 5}) {output[index] *= scale;}
  return command_in_contract(output);
}
}  // namespace ijcas_nav2_sensor_adapter
