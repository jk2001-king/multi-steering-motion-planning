"""H3_flush straight passage verdicts, not H3 turning-planner/Nav2 results.

Reuse the frozen ordinary ramp/brake profile and common actuator stop/hold
criteria. There is NO authorized threshold/obstacle contact in this variant.
The old threshold X span is only a doorway reference for six-wheel clearance.
"""

import math
from pathlib import Path
import runpy

import numpy as np


HERE = Path(__file__).resolve().parent
_original = runpy.run_path(str(HERE/"hospital_h3_terrain.py"))
_contacts = runpy.run_path(str(HERE/"hospital_contact_rules.py"))
_stop = runpy.run_path(str(HERE/"hospital_stop_analysis.py"))["analyze_stop_trace"]
ROBOT_ROOT = _original["ROBOT_ROOT"]
WHEELS = _original["WHEELS"]
COMMAND_LIMITS = dict(_original["COMMAND_LIMITS"])
READINESS = dict(_original["READINESS"])
GUARDS = dict(_original["GUARDS"])
ACCEPTANCE = dict(_original["ACCEPTANCE"])
BAD_CONTACTS = ("hospital_obstacle","unexpected_legacy_support","nonwheel_support","unclassified_external")
TerrainStopProfile = _original["TerrainStopProfile"]
relative_pose = _original["relative_pose"]
motion_stalled = _original["motion_stalled"]
wheel_crossing_checks = _original["wheel_crossing_checks"]


def classify_contact(pair, support, legacy):
    # Even contact with the formerly selected sill is an obstacle failure now.
    return _contacts["classify_contact"](pair,{support},set(legacy))


def evaluate_passage(rows, goal, static_rows, mass_kg, doorway_reference_bound,
        radii, contact_counts, callback_errors, hold_attempt_count):
    """Require a completed ONE fixed hold and actual six-wheel doorway passage.

    No best-window selection, endpoint relaxation or original-failure relabelling.
    The caller must still record run guards, cleanup and gain-restoration checks.
    """
    if (not rows or not static_rows or not math.isfinite(mass_kg) or mass_kg <= 0
            or len(goal) != 3 or not all(math.isfinite(v) for v in goal)):
        raise ValueError("Need an actual complete passage/static reference and finite target/mass")
    numeric = ("time_s","phase_time_s","x_m","y_m","z_m","yaw_rad","roll_deg","pitch_deg",
        "command_speed_m_s","pose_filtered_speed_m_s","vx_physx_m_s","vy_physx_m_s","vz_physx_m_s")
    if not all(math.isfinite(float(row[key])) for row in rows for key in numeric):
        raise ValueError("Non-finite passage observation")
    if any(float(b["time_s"]) <= float(a["time_s"]) for a,b in zip(rows,rows[1:])):
        raise ValueError("Passage timestamps must strictly increase")
    hold = [r for r in rows if r["phase"] == "terminal_hold"]
    tail = [r for r in hold if float(r["phase_time_s"]) >= 2.]
    if not tail or rows[-1]["phase"] != "terminal_hold":
        raise ValueError("A failed/incomplete trace cannot certify a terminal hold")
    if not all(math.isfinite(float(r[f"wheel_{i}_fz_n"])) for r in tail for i in range(6)):
        raise ValueError("Non-finite terminal wheel load")
    analysis = _stop(tail)
    crossing = wheel_crossing_checks(rows,doorway_reference_bound,radii)
    last = rows[-1]
    position_error = math.hypot(float(last["x_m"])-goal[0],float(last["y_m"])-goal[1])
    heading_error = math.degrees(relative_pose((last["x_m"],last["y_m"],last["yaw_rad"]),goal)[2])
    balance = float(np.mean([sum(float(r[f"wheel_{i}_fz_n"]) for i in range(6)) for r in tail])/(mass_kg*9.81))
    static_z = float(np.mean([float(r["z_m"]) for r in static_rows]))
    if not math.isfinite(static_z):
        raise ValueError("Non-finite static base-height reference")
    checks = {
        "passage_and_hold_completed":True,
        "single_fixed_hold_attempt":hold_attempt_count == 1,
        "five_second_hold_observed":float(hold[-1]["phase_time_s"]) >= 5.-1e-6,
        "all_six_wheels_started_east_and_finished_west":all(v["started_fully_east"] and v["finished_fully_west"] for v in crossing.values()),
        "terminal_position_under_30mm":position_error <= .030,
        "terminal_heading_under_2deg":abs(heading_error) <= 2.,
        "fixed_tail_complete_3s":analysis["duration_s"] >= 2.9,
        "fixed_tail_pose_standstill":analysis["development_standstill_check"] == "PASS",
        "fixed_tail_speed_under_5mm_s":all(max(float(r["pose_filtered_speed_m_s"]),math.hypot(float(r["vx_physx_m_s"]),float(r["vy_physx_m_s"]))) <= .005 for r in tail),
        "fixed_tail_position_under_30mm":all(math.hypot(float(r["x_m"])-goal[0],float(r["y_m"])-goal[1]) <= .030 for r in tail),
        "fixed_tail_heading_under_2deg":all(abs(math.degrees(relative_pose((r["x_m"],r["y_m"],r["yaw_rad"]),goal)[2])) <= 2. for r in tail),
        "all_six_final_wheels_supported":all(np.mean([float(r[f"wheel_{i}_fz_n"]) for r in tail]) > 1. for i in range(6)),
        "all_six_final_wheels_supported_95percent":all(np.mean([float(r[f"wheel_{i}_fz_n"]) > 1. for r in tail]) >= .95 for i in range(6)),
        "final_vertical_force_weight_balance":.8 <= balance <= 1.2,
        "final_base_height_at_common_floor":all(abs(float(r["z_m"])-static_z) <= .005 for r in tail),
        "final_flat_floor_roll_pitch_under_1deg":all(max(abs(float(r["roll_deg"])),abs(float(r["pitch_deg"]))) <= 1. for r in tail),
        "contact_monitor_healthy":not callback_errors,
        "no_obstacle_nonwheel_or_legacy_contacts":not any(contact_counts.get(key,0) for key in BAD_CONTACTS),
        "expected_support_callbacks_observed":contact_counts.get("wheel_support",0) > 0,
        "ordinary_positive_ramp_within_30mm_s":all(0 <= float(r["command_speed_m_s"]) <= .030+1e-9 for r in rows),
    }
    return {"checks":{key:bool(value) for key,value in checks.items()},"verdict":"PASS" if all(checks.values()) else "FAIL",
        "terminal_position_error_m":position_error,"terminal_heading_error_deg":heading_error,
        "final_world_xyyaw":[float(last["x_m"]),float(last["y_m"]),float(last["yaw_rad"])],
        "fixed_terminal_hold_analysis":analysis,"wheel_doorway_crossing_checks":crossing,
        "final_force_weight_ratio":balance}
