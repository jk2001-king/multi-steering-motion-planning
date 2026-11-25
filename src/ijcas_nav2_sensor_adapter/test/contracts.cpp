// Software boundary tests only, not ROS/physics/clinical validation.
#include "ijcas_nav2_sensor_adapter/contracts.hpp"
#include <cassert>
#include <limits>
#include <future>
#include <thread>
#include <iostream>
#include <random>
#include <string>
using namespace ijcas_nav2_sensor_adapter;
struct Raster {
  unsigned int width{100}, height{100}; double resolution{.1}, ox{-5.0}, oy{-5.0};
  std::vector<unsigned char> data = std::vector<unsigned char>(10000, 0);
  unsigned int getSizeInCellsX() const {return width;}
  unsigned int getSizeInCellsY() const {return height;}
  double getResolution() const {return resolution;}
  double getOriginX() const {return ox;}
  double getOriginY() const {return oy;}
  unsigned char getCost(unsigned int x, unsigned int y) const {return data.at(y * width + x);}
};

// Independent oracle: Sutherland-Hodgman clipping, NOT separating axes.
bool clipped_intersection(std::vector<Point2> polygon, double x, double y, double h)
{
  for (int axis = 0; axis < 2; ++axis) {
    for (int sign : {-1, 1}) {
      double bound = (axis == 0 ? x : y) + sign * h;
      auto inside = [&](const Point2 & p) {return sign * (p[axis] - bound) <= 0.0;};
      std::vector<Point2> clipped;
      for (std::size_t i = 0; i < polygon.size(); ++i) {
        const auto & a = polygon[i]; const auto & b = polygon[(i + 1) % polygon.size()];
        bool in_a = inside(a), in_b = inside(b);
        if (in_a != in_b) {
          const double t = (bound - a[axis]) / (b[axis] - a[axis]);
          clipped.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])});
        }
        if (in_b) {clipped.push_back(b);}
      }
      polygon = clipped;
    }
  }
  return !polygon.empty();
}

void polygon_guard_tests()
{
  const std::vector<Point2> square{{1., 1.}, {1., -1.}, {-1., -1.}, {-1., 1.}};
  ClosedConvexPolygon p(square);
  assert(p.valid() && p.intersects_cell(0., 0., .01)); // Interior, not just outline.
  assert(p.intersects_cell(1.01, 0., .01)); // Closed boundary touch.
  assert(p.intersects_cell(1.01, 1.01, .01)); // Closed corner touch.
  assert(!p.intersects_cell(1.011, 0., .01));
  assert(!ClosedConvexPolygon({{0., 0.}, {1., 0.}, {.3, .3}, {1., 1.}, {0., 1.}}).valid());
  assert(!ClosedConvexPolygon({{0., 0.}, {1., 1.}, {0., 1.}, {1., 0.}}).valid());
  assert(!ClosedConvexPolygon({{0., 0.}, {1., 0.}, {2., 0.}}).valid());
  assert(!ClosedConvexPolygon({{0., 0.}, {1., 0.}, {1., 0.}, {0., 1.}}).valid());
  assert(!ClosedConvexPolygon({{0., 0.}, {1., 0.}, {0., std::numeric_limits<double>::infinity()}}).valid());
  auto reversed = square; std::reverse(reversed.begin(), reversed.end());
  assert(ClosedConvexPolygon(reversed).valid());
  Raster map;
  assert(full_polygon_cell_guard(map, 0., 0., 0., square).clear);
  map.data[50 * 100 + 50] = 254; // Entirely inside: legacy outline can miss it.
  assert(!full_polygon_cell_guard(map, 0., 0., 0., square).clear);
  map.data[50 * 100 + 50] = 255;
  assert(full_polygon_cell_guard(map, 0., 0., 0., square).cost == 255.);
  map.data[50 * 100 + 50] = 253;
  assert(full_polygon_cell_guard(map, 0., 0., 0., square).clear);
  map.data[50 * 100 + 60] = 254; // Cell starts exactly at x=1.
  assert(!full_polygon_cell_guard(map, 0., 0., 0., square).clear);
  map.data[50 * 100 + 60] = 0;
  map.data[50 * 100 + 61] = 254; // Full cell separated from original polygon.
  assert(full_polygon_cell_guard(map, 0., 0., 0., square).clear);
  assert(!full_polygon_cell_guard(map, 4.1, 0., 0., square).clear);
  assert(!full_polygon_cell_guard(map, -4.1, 0., 0., square).clear);
  assert(!full_polygon_cell_guard(map, 0., 0., std::numeric_limits<double>::quiet_NaN(), square).clear);
  map.resolution = 0.;
  assert(!full_polygon_cell_guard(map, 0., 0., 0., square).clear);
  std::mt19937 generator(1092026);
  std::uniform_real_distribution<double> angle(-pi, pi), position(-40., 40.), local(-2., 2.);
  for (int i = 0; i < 20000; ++i) {
    double a = angle(generator), tx = position(generator), ty = position(generator);
    std::vector<Point2> rotated;
    for (const auto & v : square) {
      rotated.push_back({tx + std::cos(a) * v[0] - std::sin(a) * v[1],
          ty + std::sin(a) * v[0] + std::cos(a) * v[1]});
    }
    const double cx = tx + local(generator), cy = ty + local(generator), h = .01;
    ClosedConvexPolygon polygon(rotated);
    assert(polygon.valid());
    assert(polygon.intersects_cell(cx, cy, h) == clipped_intersection(rotated, cx, cy, h));
  }
}

int main(int argc, char ** argv)
{
  if (argc == 2 && std::string(argv[1]) == "--saved-raster") {
    Raster map; double x, y, yaw;
    if (!(std::cin >> map.height >> map.width >> map.resolution >> map.ox >> map.oy >> x >> y >> yaw)
      || !map.width || !map.height || map.width * map.height > 65536) {return 2;}
    std::vector<Point2> footprint(4);
    for (auto & point : footprint) {if (!(std::cin >> point[0] >> point[1])) {return 3;}}
    map.data.resize(map.width * map.height);
    for (auto & cell : map.data) {
      unsigned int value; if (!(std::cin >> value) || value > 255) {return 4;}
      cell = static_cast<unsigned char>(value);
    }
    const auto result = full_polygon_cell_guard(map, x, y, yaw, footprint);
    std::cout << "full_closed_polygon_guard_clear=" << result.clear << " cost=" << result.cost
              << " column=" << result.column << " row=" << result.row << '\n';
    return 0;
  }
  if (argc != 1) {return 5;}
  polygon_guard_tests();
  const std::array<double, 3> goal{18.0, 23.0, -pi};
  assert(declared_goal_reached({18.0, 23.0, pi}, goal, 0.0, 0.0, 0.0, true));
  assert(!declared_goal_reached(goal, goal, 0.0, 0.0, 0.0, false));
  assert(!declared_goal_reached({18.031, 23.0, -pi}, goal, 0.0, 0.0, 0.0, true));
  assert(!declared_goal_reached({18.0, 23.0, -pi + 0.036}, goal, 0.0, 0.0, 0.0, true));
  assert(!declared_goal_reached(goal, goal, 0.005001, 0.0, 0.0, true));
  assert(!declared_goal_reached(goal, goal, 0.004, 0.004, 0.0, true));
  assert(!declared_goal_reached(goal, goal, 0.0, 0.0, 0.021, true));
  assert(!declared_goal_reached(goal, goal, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, true));
  assert(command_in_contract({-.05, .02, 0.0, 0.0, 0.0, .1}));
  assert(!command_in_contract({.0501, 0.0, 0.0, 0.0, 0.0, 0.0}));
  assert(!command_in_contract({0.0, 0.0, 0.0, 0.0, 0.0, .14}));
  assert(!command_in_contract({0.0, 0.0, 0.01, 0.0, 0.0, 0.0}));
  std::array<double, 6> bounded;
  const std::array<double, 6> native_roundoff{.0500000006443745, .0004, 0., 0., 0., .022};
  assert(!command_in_contract(native_roundoff));
  assert(bounded_proposal(native_roundoff, bounded));
  assert(bounded[0] <= .05 && command_in_contract(bounded));
  assert(std::abs(bounded[5] / bounded[0] - native_roundoff[5] / native_roundoff[0]) < 1e-12);
  assert(!bounded_proposal({.05000002, 0., 0., 0., 0., 0.}, bounded));
  assert(!bounded_proposal({.05, 0., 0., 0., 0., .14}, bounded));
  assert(!bounded_proposal({0., 0., .001, 0., 0., 0.}, bounded));
  struct Map {
    using mutex_t = std::recursive_mutex;
    mutex_t mutex;
    mutex_t * getMutex() {return &mutex;}
  } map;
  std::promise<void> entered, release;
  auto unlocked = release.get_future();
  std::thread reader([&]() {
    with_locked_costmap(map, [&](auto &) {entered.set_value(); unlocked.wait();});
  });
  entered.get_future().wait();
  // A different thread cannot mutate raster/origin DURING the guard.
  assert(!map.mutex.try_lock());
  release.set_value();reader.join();
  assert(map.mutex.try_lock());map.mutex.unlock();
  try {with_locked_costmap(map, [](auto &) {throw 42;});} catch (int value) {assert(value == 42);}
  assert(map.mutex.try_lock());map.mutex.unlock();
}
