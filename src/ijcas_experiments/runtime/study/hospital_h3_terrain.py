"""H3 threshold-component contact/profile rules, NOT a planner/Nav2 policy."""

import math
from pathlib import Path
import runpy


_contact=runpy.run_path(str(Path(__file__).resolve().with_name("hospital_contact_rules.py")))
ROBOT_ROOT=_contact["ROBOT_ROOT"]
WHEELS=_contact["WHEEL_NAMES"]
BAD_CONTACTS=("hospital_obstacle","nonwheel_threshold","unexpected_legacy_support","nonwheel_support","unclassified_external")
COMMAND_LIMITS={"speed_m_s":.030,"acceleration_m_s2":.020,"steering_target_rad":0.}
READINESS={"paused_static_pose_difference_m":.001,"prehold_displacement_m":.003,
    "prehold_position_from_original_start_m":.013,"heading_error_deg":.5,
    "speed_m_s":.005}
GUARDS={"maximum_simulation_duration_s":180.,"maximum_wall_time_s":900.,"maximum_control_gap_s":.25,
    "body_speed_m_s":.080,"roll_pitch_deg":2.,"heading_error_deg":2.,"lateral_error_m":.030,
    "longitudinal_overshoot_m":.030,"longitudinal_backward_m":.020,
    "stall_window_s":5.,"stall_min_command_m_s":.015,"stall_max_displacement_m":.005}
ACCEPTANCE={"position_error_m":.030,"heading_error_deg":2.,"terminal_speed_m_s":.005,
    "hold_observation_s":5.,"fixed_tail_s":3.,"tail_min_duration_s":2.9,
    "hold_regression_speed_m_s":.001,"hold_drift_m":.003,"wheel_clearance_m":.002,
    "final_base_z_error_m":.005}


def wheel_name(path):
    for name in WHEELS:
        prefix=ROBOT_ROOT+"/"+name
        if path==prefix or path.startswith(prefix+"/"):return name
    return None


def classify_contact(pair,support,legacy,thresholds):
    category=_contact["classify_contact"](pair,{support},set(legacy))
    if category!="hospital_obstacle":return category
    if not thresholds or any(not p.startswith("/World/Hospital/") for p in thresholds):
        raise ValueError("Authorize exact exported hospital threshold collider paths")
    robot=next(p for p in pair if p.startswith(ROBOT_ROOT+"/"))
    other=next(p for p in pair if not p.startswith(ROBOT_ROOT+"/"))
    if other not in thresholds:return category
    return "wheel_threshold" if wheel_name(robot) is not None else "nonwheel_threshold"


def relative_pose(pose,origin):
    dx,dy=pose[0]-origin[0],pose[1]-origin[1];c,s=math.cos(origin[2]),math.sin(origin[2])
    return c*dx+s*dy,-s*dx+c*dy,math.atan2(math.sin(pose[2]-origin[2]),math.cos(pose[2]-origin[2]))


class TerrainStopProfile:
    """Ordinary bounded ramp/brake toward a FIXED geometric endpoint.

    This deliberately does not extend the frozen 0.30 m H1 profile's scope or
    alter its source. Hold only after command zero AND measured <=5 mm/s.
    """
    def __init__(self,distance_m):
        if not math.isfinite(distance_m) or not .5<=distance_m<=3.:
            raise ValueError("Component distance must be preflight-checked and within 0.5–3 m")
        self.distance_m=distance_m;self.previous_time_s=0.;self.command_speed_m_s=0.;self.braking=False

    def step(self,elapsed_s,progress_m,measured_speed_m_s):
        if (not all(math.isfinite(v) for v in (elapsed_s,progress_m,measured_speed_m_s))
                or elapsed_s<self.previous_time_s or measured_speed_m_s<0):
            raise ValueError("Invalid/nonmonotonic terrain profile observation")
        dt=elapsed_s-self.previous_time_s;self.previous_time_s=elapsed_s
        remaining=self.distance_m-progress_m
        if remaining<=.015:self.braking=True
        target=0. if self.braking else min(COMMAND_LIMITS["speed_m_s"],
            math.sqrt(2*COMMAND_LIMITS["acceleration_m_s2"]*max(0.,remaining-.010)))
        change=COMMAND_LIMITS["acceleration_m_s2"]*dt
        self.command_speed_m_s+=max(-change,min(change,target-self.command_speed_m_s))
        ready=self.braking and self.command_speed_m_s<=1e-7 and measured_speed_m_s<=ACCEPTANCE["terminal_speed_m_s"]
        return self.command_speed_m_s,ready


def motion_stalled(rows):
    if not rows:return False
    now=float(rows[-1]["time_s"])
    tail=[r for r in rows if r["phase"]=="ramp_motion" and float(r["time_s"])>=now-GUARDS["stall_window_s"]]
    if len(tail)<2 or float(tail[-1]["time_s"])-float(tail[0]["time_s"])<GUARDS["stall_window_s"]-.05:return False
    return (all(float(r["command_speed_m_s"])>=GUARDS["stall_min_command_m_s"] for r in tail)
        and max(float(r["longitudinal_m"]) for r in tail)-min(float(r["longitudinal_m"]) for r in tail)<GUARDS["stall_max_displacement_m"])


def wheel_crossing_checks(rows,threshold_bound,radii):
    """Conservative per-wheel sphere clearance from actual rigid-body poses."""
    if set(radii)!=set(WHEELS) or any(not math.isfinite(r) or r<=0 for r in radii.values()):
        raise ValueError("Need all six conservative wheel-body radii")
    if not rows:raise ValueError("Need actual wheel pose samples")
    xmin,xmax=threshold_bound["minimum_m"][0],threshold_bound["maximum_m"][0]
    if not math.isfinite(xmin) or not math.isfinite(xmax) or xmin>=xmax:raise ValueError("Invalid threshold bound")
    result={}
    for name in WHEELS:
        key="wheel_"+name+"_x_m"
        xs=[float(row[key]) for row in rows]
        if not all(math.isfinite(v) for v in xs):raise ValueError("Non-finite measured wheel position")
        result[name]={"started_fully_east":xs[0]-radii[name]>xmax+ACCEPTANCE["wheel_clearance_m"],
            "finished_fully_west":xs[-1]+radii[name]<xmin-ACCEPTANCE["wheel_clearance_m"],
            "sphere_radius_m":radii[name],"first_x_m":xs[0],"last_x_m":xs[-1]}
    return result
