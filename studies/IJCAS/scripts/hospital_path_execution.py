"""Pure helpers for a development-only, ground-truth H1 path execution pilot.

No ROS, Isaac imports, CICS input, online planning or hardware claims. The
common tracking/terminal controller is to be held fixed for all four methods.
The existing simulation's two direct-drive joint signs are both positive;
do not substitute the ROS URDF's negative rear-wheel convention here.
"""

import bisect
import math


MODES = ("coordinated_counterphase", "front_only", "rear_only")


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def relative_pose(pose, reference):
    dx, dy = pose[0]-reference[0], pose[1]-reference[1]
    c, s = math.cos(reference[2]), math.sin(reference[2])
    return c*dx+s*dy, -s*dx+c*dy, wrap(pose[2]-reference[2])


def ramp(current, target, dt, rate=.02):
    if not all(math.isfinite(v) for v in (current, target, dt, rate)) or dt < 0 or rate <= 0:
        raise ValueError("Invalid finite rate-limiter inputs")
    # Reach zero before changing travel direction; never step through zero.
    if current*target < 0:
        target = 0.
    return current+max(-rate*dt, min(rate*dt, target-current))


def direct_joint_targets(speed, mode, curvature, previous_steer, dt, measured_steer):
    values = (speed, curvature, *previous_steer, dt, *measured_steer)
    if (mode not in MODES or not all(math.isfinite(v) for v in values)
            or not 0 <= dt <= .25 or abs(speed) > .0500001 or abs(curvature) > .7000001):
        raise ValueError("Command exceeds the H1 pilot bounds")
    if mode == "front_only":
        target = math.atan(1.29*curvature), 0.
    elif mode == "rear_only":
        target = 0., -math.atan(1.29*curvature)
    else:
        target = math.atan(.645*curvature), -math.atan(.645*curvature)
    steer = tuple(ramp(old, new, dt, math.radians(8.)) for old, new in zip(previous_steer, target))
    if len(steer) != 2 or len(measured_steer) != 2 or any(abs(v) > math.radians(60.) for v in measured_steer):
        raise ValueError("Invalid measured steering angles")
    wheels = tuple(speed/(.074*math.cos(angle)) for angle in measured_steer)
    return steer, wheels


def validate_plan(plan):
    actions, states = plan.get("selected_actions", []), plan.get("selected_states", [])
    if (plan.get("scenario") != "H1_bedside_departure" or not plan.get("plan_success")
            or len(actions) != len(states) or not actions):
        raise ValueError("Need a successful, complete H1 plan")
    for key in ("start_world_xyyaw", "goal_world_xyyaw"):
        if len(plan[key]) != 3 or not all(math.isfinite(v) for v in plan[key]):
            raise ValueError("Invalid world pose")
    for action, state in zip(actions, states):
        if (action["mode"] not in MODES or action["direction"] != 1
                or not math.isfinite(action["curvature_1_m"]) or abs(action["curvature_1_m"]) > .70+1e-9
                or not math.isfinite(action["target_speed_m_s"]) or not 0 < action["target_speed_m_s"] <= .05):
            raise ValueError("This development adapter supports bounded forward H1 actions only")
        pose = state["world_xyyaw"]
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError("Invalid saved world pose")


class PathReference:
    """Project onto a small monotone neighbourhood, never jump across a loop."""

    def __init__(self, plan):
        validate_plan(plan)
        self.actions = plan["selected_actions"]
        self.poses = [tuple(plan["start_world_xyyaw"])]+[tuple(s["world_xyyaw"]) for s in plan["selected_states"]]
        self.arc = [0.]
        for first, last in zip(self.poses, self.poses[1:]):
            distance = math.hypot(last[0]-first[0], last[1]-first[1])
            if not .001 <= distance <= .10:
                raise ValueError("Unexpected zero/oversized H1 plan edge")
            self.arc.append(self.arc[-1]+distance)
        self.progress = 0.

    def at(self, distance):
        index = min(len(self.actions)-1, max(0, bisect.bisect_right(self.arc, distance)-1))
        alpha = max(0., min(1., (distance-self.arc[index])/(self.arc[index+1]-self.arc[index])))
        first, last = self.poses[index:index+2]
        return index, (first[0]+alpha*(last[0]-first[0]), first[1]+alpha*(last[1]-first[1]),
            first[2]+alpha*wrap(last[2]-first[2]))

    def project(self, pose):
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError("Non-finite tracking pose")
        current, _ = self.at(self.progress)
        candidates = []
        for index in range(max(0, current-1), min(len(self.actions), current+4)):
            first, last = self.poses[index:index+2]
            dx, dy = last[0]-first[0], last[1]-first[1]
            fraction = max(0., min(1., ((pose[0]-first[0])*dx+(pose[1]-first[1])*dy)/(dx*dx+dy*dy)))
            x, y = first[0]+fraction*dx, first[1]+fraction*dy
            candidates.append((math.hypot(pose[0]-x, pose[1]-y), self.arc[index]+fraction*(self.arc[index+1]-self.arc[index])))
        error, projected = min(candidates)
        # Reject a large single-update advance rather than silently skipping
        # actions following a stale pose, teleport or nonlocal projection.
        if projected-self.progress > .10:
            raise ValueError("Path projection jumped over 0.10 m")
        self.progress = max(self.progress, projected)
        index, reference = self.at(self.progress)
        return index, reference, error


class H1TrackingController:
    """Frozen-path feedback plus separately reported bounded terminal closure.

    Tracking is ground-truth feedback, NOT sensor-based navigation. No new
    spatial route or mode is invented during tracking. Terminal mode changes
    are execution stabilization and must not be counted as planner decisions.
    """

    def __init__(self, plan):
        self.reference = PathReference(plan)
        self.goal = tuple(plan["goal_world_xyyaw"])
        self.phase = "path_tracking"
        self.speed = 0.
        self.previous_time = None
        self.mode = plan["selected_actions"][0]["mode"]
        self.curvature = plan["selected_actions"][0]["curvature_1_m"]
        self.direction = 0
        self.terminal_mode = ""
        self.terminal_switches = 0
        self.handoff = None

    def step(self, timestamp, pose, measured_speed):
        if not all(math.isfinite(v) for v in (timestamp, *pose, measured_speed)) or measured_speed < 0:
            raise ValueError("Non-finite/invalid measured execution state")
        dt = 0. if self.previous_time is None else timestamp-self.previous_time
        if not 0 <= dt <= .25:
            raise ValueError("Execution control gap exceeds 0.25 simulated seconds")
        self.previous_time = timestamp
        dx, dy = self.goal[0]-pose[0], self.goal[1]-pose[1]
        position_error = math.hypot(dx, dy)
        lon, lat, yaw_residual = relative_pose(self.goal, pose)
        index, reference, path_error = self.reference.project(pose)
        yaw_error = wrap(reference[2]-pose[2])
        if self.phase == "path_tracking":
            if path_error > .080 or abs(yaw_error) > math.radians(10.):
                raise RuntimeError("H1 path tracking deviation exceeded 80 mm / 10 degrees")
            remaining = self.reference.arc[-1]-self.reference.progress
            if remaining <= .025 and math.hypot(pose[0]-self.reference.poses[-1][0], pose[1]-self.reference.poses[-1][1]) <= .05:
                self.phase = "terminal_correction"
                self.handoff = {"path_progress_m": self.reference.progress, "remaining_path_m": remaining,
                    "goal_position_error_m": position_error, "goal_heading_error_deg": math.degrees(yaw_residual)}
            else:
                action = self.reference.actions[index]
                self.mode = action["mode"]
                _, cross_error, _ = relative_pose(reference, pose)
                self.curvature = max(-.70, min(.70, action["curvature_1_m"]+1.5*yaw_error+2.*cross_error))
                # Bound endpoint speed even if the final action was generated
                # by a purely kinematic baseline. Same limiter for all methods.
                target = min(action["target_speed_m_s"], math.sqrt(2*.02*max(.000225, remaining-.010)))
        if self.phase == "terminal_correction":
            # Interior capture band, not a relaxed acceptance threshold: the
            # complete five-second hold must still satisfy 30 mm / 2 degrees.
            if position_error <= .015 and abs(yaw_residual) <= math.radians(1.5):
                self.phase = "terminal_braking"
            else:
                if self.direction == 0:
                    self.direction = 1 if lon >= 0 else -1
                elif self.direction*lon < -.015:
                    self.direction *= -1
                if abs(lat) > .0025:
                    lateral_sign = 1 if lat >= 0 else -1
                    need = yaw_residual*lateral_sign
                    hysteresis = math.radians(.15)
                    if self.terminal_mode == "front_only" and need >= -hysteresis:
                        mode = "front_only"
                    elif self.terminal_mode == "rear_only" and need <= hysteresis:
                        mode = "rear_only"
                    else:
                        mode = "front_only" if need >= 0 else "rear_only"
                    self.terminal_switches += bool(self.terminal_mode and mode != self.terminal_mode)
                    self.terminal_mode, self.mode = mode, mode
                    self.curvature = .70*self.direction*lateral_sign*(-1 if mode == "rear_only" else 1)
                else:
                    self.mode = "coordinated_counterphase"
                    self.curvature = max(-.70, min(.70, yaw_residual/math.copysign(max(abs(lon), .05), lon)))
                target = self.direction*min(.010, max(.003, .5*position_error-.8*measured_speed))
        if self.phase == "terminal_braking":
            target = 0.
        self.speed = ramp(self.speed, target, dt)
        ready = self.phase == "terminal_braking" and abs(self.speed) < 1e-7 and measured_speed <= .010
        return {"phase": self.phase, "speed_m_s": self.speed, "mode": self.mode, "curvature_1_m": self.curvature,
            "action_index": index, "progress_m": self.reference.progress, "path_error_m": path_error,
            "path_heading_error_deg": math.degrees(yaw_error), "goal_position_error_m": position_error,
            "goal_heading_error_deg": math.degrees(yaw_residual), "hold_ready": ready}
