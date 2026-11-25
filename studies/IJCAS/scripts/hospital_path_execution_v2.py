"""Development stop/align/pulse execution stabilizer, never an IJCAS planner.

Preserves legacy path tracking and command bounds. Terminal reversals are
made only after braking, captured active-drive hold and measured steering
alignment. Passive casters, material and motor limits are never changed.
Needs the v2 Isaac adapter: ``drive_hold_requested`` is a physical handshake,
not a simulated zero-speed substitute. This module alone does not prove PhysX
convergence. All four methods need a new cohort after readiness validation.
"""

import math
from pathlib import Path
import runpy


_legacy = runpy.run_path(str(Path(__file__).with_name("hospital_path_execution.py")))
direct_joint_targets = _legacy["direct_joint_targets"]
relative_pose = _legacy["relative_pose"]
validate_plan = _legacy["validate_plan"]
wrap = _legacy["wrap"]
ramp = _legacy["ramp"]

CONTROLLER_CONFIG = {
    "version": "stop_align_pulse_v2",
    "terminal_speed_limit_m_s": .010,
    "acceleration_m_s2": .020,
    "stopped_speed_m_s": .005,
    "stationary_settle_s": .50,
    "alignment_tolerance_deg": 2.5,
    "alignment_dwell_s": .25,
    "alignment_timeout_s": 16.,
    "pulse_distance_min_m": .010,
    "pulse_distance_max_m": .020,
    "pulse_braking_reserve_m": .002,
    "pulse_timeout_s": 8.,
    "pulse_stall_window_s": 5.,
    "pulse_stall_displacement_m": .001,
    "maximum_pulses": 12,
    "terminal_timeout_s": 90.,
    "capture_position_m": .015,
    "capture_heading_deg": 1.5,
}


def steering_geometry(mode, curvature):
    if mode == "front_only":
        return math.atan(1.29*curvature), 0.
    if mode == "rear_only":
        return 0., -math.atan(1.29*curvature)
    if mode == "coordinated_counterphase":
        return math.atan(.645*curvature), -math.atan(.645*curvature)
    raise ValueError("Unknown terminal steering mode")


class H1TrackingController(_legacy["H1TrackingController"]):
    """Shared v2 execution policy; no method-specific tuning or route changes."""

    def __init__(self, plan):
        super().__init__(plan)
        self.config = dict(CONTROLLER_CONFIG)
        self.terminal_stage = "tracking"
        self.stage_started = None
        self.terminal_started = None
        self.aligned_since = None
        self.pulse_count = 0
        self.pulse_origin = None
        self.pulse_distance = 0.
        self.drive_hold_requested = False
        self.preferred_direction = 1

    def _stage(self, stage, timestamp):
        self.terminal_stage, self.stage_started = stage, timestamp
        self.aligned_since = None

    def _inside_capture(self, pose):
        error = relative_pose(self.goal, pose)
        return math.hypot(error[0], error[1]) <= self.config["capture_position_m"] and abs(error[2]) <= math.radians(self.config["capture_heading_deg"])

    def _choose_pulse(self, pose):
        lon, lat, heading = relative_pose(self.goal, pose)
        # Decide once per fully stopped pulse. The decision is frozen through
        # steering alignment and the following acceleration/braking segment.
        direction = (1 if lon >= 0 else -1) if abs(lon) > .003 else self.preferred_direction
        if abs(lat) > .0025:
            lateral_sign = 1 if lat >= 0 else -1
            mode = "front_only" if heading*lateral_sign >= 0 else "rear_only"
            curvature = .70*direction*lateral_sign*(-1 if mode == "rear_only" else 1)
        else:
            mode = "coordinated_counterphase"
            curvature = max(-.70, min(.70, heading/math.copysign(max(abs(lon), .03), direction)))
        self.terminal_switches += bool(self.terminal_mode and self.terminal_mode != mode)
        self.terminal_mode = self.mode = mode
        self.direction = self.preferred_direction = direction
        self.curvature = curvature
        self.pulse_distance = min(self.config["pulse_distance_max_m"], max(self.config["pulse_distance_min_m"], abs(lon)))

    def step(self, timestamp, pose, measured_speed, measured_steer):
        if (len(measured_steer) != 2 or not all(math.isfinite(v) for v in (timestamp, *pose, measured_speed, *measured_steer))
                or measured_speed < 0):
            raise ValueError("Invalid v2 measured state")
        dt = 0. if self.previous_time is None else timestamp-self.previous_time
        if not 0 <= dt <= .25:
            raise ValueError("Execution control gap exceeds 0.25 simulated seconds")
        if self.terminal_stage == "tracking":
            previous = self.mode, self.curvature, self.speed
            result = super().step(timestamp, pose, measured_speed)
            if self.phase == "path_tracking":
                return {**result, "terminal_stage": "tracking", "drive_hold_requested": False, "pulse_count": 0}
            # Discard the parent's instantaneous terminal command: no command
            # from it is applied. Brake using the last TRACKING geometry first.
            self.mode, self.curvature, self.speed = previous
            self.direction, self.terminal_mode, self.terminal_switches = 0, "", 0
            self.terminal_started = timestamp
            self._stage("entry_braking", timestamp)
        self.previous_time = timestamp
        cfg = self.config
        if timestamp-self.terminal_started > cfg["terminal_timeout_s"]:
            raise TimeoutError("v2 terminal readiness timeout (90 simulated seconds); no acceptance threshold was changed")
        stage = self.terminal_stage
        self.drive_hold_requested = stage in ("stationary_settle", "steering_alignment", "final_hold_ready")
        target_speed = 0.
        ready = False
        if stage in ("entry_braking", "pulse_braking"):
            self.phase = "terminal_braking"
            self.speed = ramp(self.speed, 0., dt, cfg["acceleration_m_s2"])
            if abs(self.speed) < 1e-7 and measured_speed <= cfg["stopped_speed_m_s"]:
                self.speed, self.drive_hold_requested = 0., True
                self._stage("stationary_settle", timestamp)
        elif stage == "stationary_settle":
            self.phase, self.speed = "terminal_correction", 0.
            if measured_speed > cfg["stopped_speed_m_s"]:
                self.stage_started = timestamp
            elif timestamp-self.stage_started >= cfg["stationary_settle_s"]:
                if self._inside_capture(pose):
                    self._stage("final_hold_ready", timestamp)
                    ready = True
                else:
                    if self.pulse_count >= cfg["maximum_pulses"]:
                        raise TimeoutError("v2 terminal maximum pulse count reached")
                    self._choose_pulse(pose)
                    self._stage("steering_alignment", timestamp)
        elif stage == "steering_alignment":
            self.phase, self.speed = "terminal_correction", 0.
            target_steer = steering_geometry(self.mode, self.curvature)
            error = max(abs(actual-target) for actual, target in zip(measured_steer, target_steer))
            if timestamp-self.stage_started > cfg["alignment_timeout_s"]:
                raise TimeoutError(f"v2 steering alignment timeout: remaining error {math.degrees(error):.2f} deg")
            if error <= math.radians(cfg["alignment_tolerance_deg"]) and measured_speed <= cfg["stopped_speed_m_s"]:
                if self.aligned_since is None:
                    self.aligned_since = timestamp
                if timestamp-self.aligned_since >= cfg["alignment_dwell_s"]:
                    self.pulse_origin = tuple(pose)
                    self.pulse_count += 1
                    self._stage("pulse_motion", timestamp)
                    self.drive_hold_requested = False
            else:
                self.aligned_since = None
        elif stage == "pulse_motion":
            self.phase = "terminal_correction"
            distance = math.hypot(pose[0]-self.pulse_origin[0], pose[1]-self.pulse_origin[1])
            elapsed = timestamp-self.stage_started
            if elapsed >= cfg["pulse_stall_window_s"] and distance < cfg["pulse_stall_displacement_m"]:
                raise TimeoutError("v2 terminal pulse stalled: under 1 mm movement in 5 simulated seconds")
            if elapsed > cfg["pulse_timeout_s"]:
                raise TimeoutError("v2 terminal pulse exceeded 8 simulated seconds")
            braking_distance = self.speed*self.speed/(2*cfg["acceleration_m_s2"])+cfg["pulse_braking_reserve_m"]
            if distance >= max(0., self.pulse_distance-braking_distance) or self._inside_capture(pose):
                self._stage("pulse_braking", timestamp)
            else:
                target_speed = self.direction*cfg["terminal_speed_limit_m_s"]
            self.speed = ramp(self.speed, target_speed, dt, cfg["acceleration_m_s2"])
        elif stage == "final_hold_ready":
            self.phase, self.speed, ready = "terminal_braking", 0., True
        else:
            raise RuntimeError("Unknown v2 terminal stage: "+stage)
        if ready and (abs(self.speed) > 1e-7 or measured_speed > cfg["stopped_speed_m_s"]):
            raise RuntimeError("v2 final hold requested before measured stop")
        index, reference = self.reference.at(self.reference.progress)
        error = relative_pose(self.goal, pose)
        return {"phase": self.phase, "terminal_stage": self.terminal_stage, "speed_m_s": self.speed,
            "mode": self.mode, "curvature_1_m": self.curvature, "action_index": index,
            "progress_m": self.reference.progress, "path_error_m": math.hypot(pose[0]-reference[0], pose[1]-reference[1]),
            "path_heading_error_deg": math.degrees(wrap(reference[2]-pose[2])),
            "goal_position_error_m": math.hypot(error[0], error[1]), "goal_heading_error_deg": math.degrees(error[2]),
            "hold_ready": ready, "drive_hold_requested": self.drive_hold_requested, "pulse_count": self.pulse_count}
