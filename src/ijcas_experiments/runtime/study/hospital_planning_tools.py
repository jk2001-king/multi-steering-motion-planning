"""Full-envelope collision adapter for the audited hospital occupancy raster."""

import math
import time

import numpy as np

from hospital_geometry_tools import footprint_polygon, pose_collision


def local_to_world(pose, reference):
    x, y, yaw = map(float, pose)
    c, s = math.cos(reference[2]), math.sin(reference[2])
    return reference[0]+c*x-s*y, reference[1]+s*x+c*y, reference[2]+yaw


class HospitalEnvelopeChecker:
    """Cell-supercover rectangle SAT with a safe empty-AABB shortcut.

    No occupied cell is removed or approximated by a point: when the bounding
    box contains occupied cells, test the rectangle against ONLY those cells.
    Both world axes and both body axes are tested, with each cell's finite
    square support radius included. This is the same SAT as pose_collision,
    without allocating projection arrays for thousands of empty cells.
    Deadline is per planning attempt, not a scientific acceptance criterion.
    """

    def __init__(self, occupancy, origin, resolution, start, clearance=.010, deadline=None):
        self.occupancy = np.asarray(occupancy, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)
        self.start = tuple(map(float, start))
        self.resolution, self.clearance, self.deadline = resolution, clearance, deadline
        if (self.occupancy.ndim != 2 or not all(self.occupancy.shape) or self.origin.shape != (2,) or
                len(self.start) != 3 or not np.isfinite((*self.origin, *self.start, resolution, clearance)).all() or
                resolution <= 0 or clearance < 0):
            raise ValueError("Invalid hospital raster geometry")
        self.maximum = self.origin+np.asarray(self.occupancy.shape[::-1])*resolution
        self.integral = np.pad(self.occupancy.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        self.cache = {}
        self.statistics = {"pose_queries": 0, "empty_aabb": 0, "exact_sat": 0, "cache_hits": 0}

    def check_deadline(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError("Hospital planning attempt exceeded its wall-clock budget")

    def world_collision(self, pose):
        self.check_deadline()
        pose = tuple(map(float, pose))
        if len(pose) != 3 or not np.isfinite(pose).all():
            raise ValueError("Non-finite hospital pose")
        self.statistics["pose_queries"] += 1
        if pose in self.cache:
            self.statistics["cache_hits"] += 1
            return self.cache[pose]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        a, b = 1.630/2+self.clearance, .965/2+self.clearance
        extent = np.array([abs(c)*a+abs(s)*b, abs(s)*a+abs(c)*b])
        lower, upper = np.asarray(pose[:2])-extent, np.asarray(pose[:2])+extent
        if np.any(lower < self.origin) or np.any(upper > self.maximum):
            collision = True
        else:
            lo = np.maximum(np.floor((lower-self.origin)/self.resolution-1e-10).astype(int), 0)
            hi = np.minimum(np.floor((upper-self.origin)/self.resolution+1e-10).astype(int),
                            np.asarray(self.occupancy.shape[::-1])-1)
            x0, y0 = lo
            x1, y1 = hi+1
            count = self.integral[y1, x1]-self.integral[y0, x1]-self.integral[y1, x0]+self.integral[y0, x0]
            if count == 0:
                self.statistics["empty_aabb"] += 1
                collision = False
            else:
                self.statistics["exact_sat"] += 1
                rows, cols = np.nonzero(self.occupancy[y0:y1, x0:x1])
                dx = self.origin[0]+(cols+x0+.5)*self.resolution-pose[0]
                dy = self.origin[1]+(rows+y0+.5)*self.resolution-pose[1]
                half_cell = self.resolution/2
                body_support = half_cell*(abs(c)+abs(s))
                collision = bool(np.any(
                    (np.abs(dx) <= extent[0]+half_cell+1e-12) &
                    (np.abs(dy) <= extent[1]+half_cell+1e-12) &
                    (np.abs(c*dx+s*dy) <= a+body_support+1e-12) &
                    (np.abs(-s*dx+c*dy) <= b+body_support+1e-12)))
        if len(self.cache) >= 50000:
            self.cache.clear()
        self.cache[pose] = bool(collision)
        return bool(collision)

    def __call__(self, local_poses):
        self.check_deadline()
        return any(self.world_collision(local_to_world(pose, self.start)) for pose in local_poses)
