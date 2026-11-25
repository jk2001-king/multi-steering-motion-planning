"""H2 frame/schema binding to the UNCHANGED v3 tracking/terminal methods.

Only construction and validation are task-specific. Do not relabel an H2
plan as H1, change its world coordinates or modify frozen H1 source files.
The original reference projection, tracking, braking, stop/align/pulse and
single fixed-hold entry implementations are inherited without retuning.
"""

import math
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
_legacy = runpy.run_path(str(HERE / "hospital_path_execution.py"))
_v3 = runpy.run_path(str(HERE / "hospital_path_execution_v3.py"))
_methods = runpy.run_path(str(HERE / "hospital_path_methods.py"))
CONTROLLER_CONFIG = dict(_v3["CONTROLLER_CONFIG"])
EXECUTION_VERSION = "stop_align_v3_H2_adapter_v1"
direct_joint_targets = _v3["direct_joint_targets"]
relative_pose = _v3["relative_pose"]
wrap = _v3["wrap"]
GUARDS = {"path_error_m": .080, "path_heading_error_deg": 10., "tilt_deg": 2., "body_speed_m_s": .080,
    "maximum_control_gap_s": .25, "maximum_simulation_duration_s": 180., "maximum_wall_time_s": 900.}
ACCEPTANCE = {"position_error_m": .030, "heading_error_deg": 2., "terminal_speed_m_s": .005,
    "hold_regression_speed_m_s": .001, "hold_drift_m": .003, "hold_observation_s": 5., "fixed_hold_tail_s": 3.}
COMMAND_LIMITS = {"speed_m_s": .05, "acceleration_m_s2": .02, "steering_rate_deg_s": 8., "curvature_1_m": .70}
COMMON_SOURCE_FILES = ("hospital_h2_path_execution.py", "hospital_path_execution.py", "hospital_path_execution_v2.py",
    "hospital_path_execution_v3.py", "hospital_drive_hold.py", "hospital_stop_analysis.py", "hospital_contact_rules.py",
    "hospital_geometry_tools.py", "hospital_path_methods.py", "hospital_static_gate.py",
    "isaac_prepare_ijcas_hospital_h2_physics.py", "isaac_run_ijcas_hospital_h2_static_gate.py",
    "isaac_run_ijcas_hospital_h2_path_gate.py")


def method_settings(method):
    result = _methods["method_settings"](method)
    return {**result, "run_prefix": result["run_prefix"].replace("hospital_h1_", "hospital_h2_stopalign_v3_", 1)}


def motor_settings_record(kp, kd, efforts):
    """JSON-safe observations: unactuated joints may have unlimited effort.

    This does not change a limit. The actual arrays still undergo unchanged
    runtime equality checks; positive infinity is explicitly recorded, not
    emitted as invalid JSON or replaced by a finite control limit.
    """
    kp, kd, efforts = [float(v) for v in kp], [float(v) for v in kd], [float(v) for v in efforts]
    if (not kp or not len(kp) == len(kd) == len(efforts)
            or not all(math.isfinite(v) for v in (*kp, *kd))
            or any(math.isnan(v) or v == -math.inf for v in efforts)):
        raise ValueError("Invalid observed motor gain/effort vectors")
    return {"kp": kp, "kd": kd, "max_efforts": [v if math.isfinite(v) else "positive_infinity" for v in efforts]}


def validate_plan(plan):
    actions, states = plan.get("selected_actions", []), plan.get("selected_states", [])
    if (plan.get("scenario") != "H2_bedside_lateral_alignment" or not plan.get("plan_success")
            or not actions or len(actions) != len(states)):
        raise ValueError("Need a successful complete H2 plan, not an H1 relabel")
    settings = method_settings(plan["method"])
    if plan["planner_dynamics"] != settings["planner_dynamics"]:
        raise ValueError("H2 method/dynamics mismatch")
    for key in ("start_world_xyyaw", "goal_world_xyyaw", "goal_local_xyyaw"):
        if len(plan[key]) != 3 or not all(math.isfinite(v) for v in plan[key]):
            raise ValueError("Invalid H2 world/local pose")
    expected = relative_pose(plan["goal_world_xyyaw"], plan["start_world_xyyaw"])
    if any(abs(a-b) > 1e-10 for a, b in zip(expected, plan["goal_local_xyyaw"])):
        raise ValueError("H2 local/world task frame mismatch")
    if (plan["common_max_command_speed_m_s"] != .05 or plan["common_max_acceleration_m_s2"] != .02
            or plan["full_envelope_clearance_m"] != .010):
        raise ValueError("The H2 common limits/envelope must stay equal to H1")
    for action, state in zip(actions, states):
        if (action["mode"] not in _legacy["MODES"] or action["mode"] not in plan["allowed_modes"]
                or action["direction"] != 1 or not math.isfinite(action["curvature_1_m"])
                or abs(action["curvature_1_m"]) > .70+1e-9 or not math.isfinite(action["target_speed_m_s"])
                or not 0 < action["target_speed_m_s"] <= .05):
            raise ValueError("Only bounded forward planned H2 actions are supported")
        pose = state["world_xyyaw"]
        if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
            raise ValueError("Invalid saved H2 world pose")


class PathReference(_legacy["PathReference"]):
    """Use the same projection methods with genuinely H2-validated poses."""

    def __init__(self, plan):
        validate_plan(plan)
        self.actions = plan["selected_actions"]
        self.poses = [tuple(plan["start_world_xyyaw"])]+[tuple(s["world_xyyaw"]) for s in plan["selected_states"]]
        self.arc = [0.]
        for first, last in zip(self.poses, self.poses[1:]):
            distance = math.hypot(last[0]-first[0], last[1]-first[1])
            if not .001 <= distance <= .10:
                raise ValueError("Unexpected zero/oversized H2 edge")
            self.arc.append(self.arc[-1]+distance)
        self.progress = 0.


class H2TrackingController(_v3["H1TrackingController"]):
    """Task-specific construction only; every control method is frozen v3."""

    def __init__(self, plan):
        self.reference = PathReference(plan)
        self.goal = tuple(plan["goal_world_xyyaw"])
        self.phase, self.speed, self.previous_time = "path_tracking", 0., None
        self.mode = plan["selected_actions"][0]["mode"]
        self.curvature = plan["selected_actions"][0]["curvature_1_m"]
        self.direction, self.terminal_mode, self.terminal_switches = 0, "", 0
        self.handoff = None
        self.config = dict(CONTROLLER_CONFIG)
        self.terminal_stage = "tracking"
        self.stage_started = self.terminal_started = self.aligned_since = None
        self.pulse_count, self.pulse_origin, self.pulse_distance = 0, None, 0.
        self.drive_hold_requested, self.preferred_direction = False, 1
        self.hold_entry_candidate = None


def comparison_mismatches(reference, current):
    differences = []
    for key in ("scenario", "execution_version", "execution_role", "execution_cohort", "geometry_manifest_sha256",
            "start_world_xyyaw", "goal_world_xyyaw", "planning_occupancy_digest", "command_limits", "guards",
            "acceptance", "terminal_controller_config", "feedback_source"):
        if key not in reference or key not in current or reference[key] != current[key]:
            differences.append(key)
    for name in COMMON_SOURCE_FILES:
        old, new = reference.get("source_sha256", {}).get(name), current.get("source_sha256", {}).get(name)
        if not old or old != new:
            differences.append("source_sha256."+name)
    for key in _methods["PHYSICS_COMPARISON_KEYS"]:
        old, new = reference.get("physics_preparation_snapshot", {}), current.get("physics_preparation_snapshot", {})
        if key not in old or key not in new or old[key] != new[key]:
            differences.append("physics."+key)
    key = "joint_settings_USD_angular_units"
    old, new = reference.get("physics_preparation_snapshot", {}), current.get("physics_preparation_snapshot", {})
    if key not in old or key not in new or old[key] != new[key]:
        differences.append("physics."+key)
    return differences
