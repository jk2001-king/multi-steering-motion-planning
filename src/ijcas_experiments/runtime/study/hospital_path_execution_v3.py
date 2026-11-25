"""Development fixed-hold entry policy; not a planner or a measured PASS.

Keep v2 tracking, braking, steering alignment and pulse safety unchanged.
After a measured stop and its existing settling interval, a pose inside the
pre-existing 30 mm / 2 deg acceptance region may start ONE fixed hold test.
The Isaac adapter still evaluates the full five-second hold / fixed last
three seconds. An entry candidate is not a successful trial, and a failed
hold is not retried or replaced by a more favourable time window.

The stricter 15 mm / 1.5 deg moving-pulse braking target is retained. Neither
motor gains/limits nor material/passive-caster settings are changed here.
All methods require the same new policy in a new comparison cohort.
"""

import math
from pathlib import Path
import runpy


_v2 = runpy.run_path(str(Path(__file__).with_name("hospital_path_execution_v2.py")))
direct_joint_targets = _v2["direct_joint_targets"]
relative_pose = _v2["relative_pose"]
validate_plan = _v2["validate_plan"]
wrap = _v2["wrap"]
steering_geometry = _v2["steering_geometry"]

CONTROLLER_CONFIG = {
    **_v2["CONTROLLER_CONFIG"],
    "version": "stop_align_fixed_hold_v3",
    "stationary_hold_entry_position_m": .030,
    "stationary_hold_entry_heading_deg": 2.,
    "hold_entry_policy": "existing_acceptance_only_after_measured_stop_and_settle",
    "hold_failure_policy": "fail_once_no_retry_or_best_window_selection",
}


class H1TrackingController(_v2["H1TrackingController"]):
    """A stopped acceptance-region candidate must still pass the fixed hold."""

    def __init__(self, plan):
        super().__init__(plan)
        self.config = dict(CONTROLLER_CONFIG)
        self.hold_entry_candidate = None

    def _inside_capture(self, pose):
        if self.terminal_stage != "stationary_settle":
            # Do not start moving-pulse braking earlier merely because a
            # moving pose momentarily crosses the final acceptance region.
            return super()._inside_capture(pose)
        error = relative_pose(self.goal, pose)
        position = math.hypot(error[0], error[1])
        heading = math.degrees(error[2])
        inside = (position <= self.config["stationary_hold_entry_position_m"]
            and abs(heading) <= self.config["stationary_hold_entry_heading_deg"])
        if inside:
            self.hold_entry_candidate = {
                "reason": "stopped_inside_existing_acceptance_start_fixed_hold_NOT_PASS",
                "position_error_m": position,
                "heading_error_deg": heading,
                "settled_duration_s": self.previous_time-self.stage_started,
                "pulse_count": self.pulse_count,
                "fixed_hold_still_required": True,
            }
        return inside
