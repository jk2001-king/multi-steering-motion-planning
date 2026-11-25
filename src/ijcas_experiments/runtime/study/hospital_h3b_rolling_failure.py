"""Reproduce a recorded low-speed rolling stall, without changing the trial.

Angle-difference wheel motion is kept separate from instantaneous solver qd.
No torque, contact-friction cause, full-route infeasibility or PASS is inferred.
"""

import json
import math
from pathlib import Path
from statistics import mean, median

from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources
from hospital_h3b_evidence import read_trace
from hospital_h3b_rolling_readiness import ROLE, curvature_at, motion_stalled
from hospital_static_gate import evaluate_static


DEFAULT = GATE/"hospital_h3b_rolling_readiness/hospital_h3b_rolling_readiness_r01.json"


def slope(rows, key):
    t0 = mean(r["time_s"] for r in rows)
    v0 = mean(r[key] for r in rows)
    denominator = sum((r["time_s"]-t0)**2 for r in rows)
    if denominator <= 0.:
        raise ValueError("A nonzero observation interval is required")
    return sum((r["time_s"]-t0)*(r[key]-v0) for r in rows)/denominator


def trace_metrics(rows):
    if (len(rows) < 10 or any(b["time_s"] <= a["time_s"] for a,b in zip(rows, rows[1:]))
            or any(not math.isfinite(value) for row in rows for value in row.values() if isinstance(value, (int,float)))):
        raise ValueError("Finite, strictly monotone actual trace required")
    if rows[-1]["phase"] != "leg_0_motion" or not motion_stalled(rows):
        raise ValueError("The saved trace does not reproduce the original first-leg stall guard")
    if any(motion_stalled(rows[:i]) for i in range(1,len(rows))):
        raise ValueError("An earlier sample should have triggered the frozen stall guard")
    command_error, curvature_error = 0., 0.
    for previous, row in zip(rows, rows[1:]):
        curvature_error = max(curvature_error, abs(row["requested_curvature_1_m"]-curvature_at(previous["longitudinal_m"],row["phase"])))
        for axle in ("front","rear"):
            expected = row["command_speed_m_s"]/(.074*math.cos(previous[axle+"_rad"]))
            command_error = max(command_error,abs(row["drive_"+axle+"_actual_velocity_target_rad_s"]-expected))
    if command_error > 1e-6 or curvature_error > 1e-10:
        raise ValueError("Applied drive or fixed-progress curvature commands do not reproduce")
    last = rows[-1]
    window = [r for r in rows if r["time_s"] >= last["time_s"]-5.]
    tail = [r for r in rows if r["time_s"] >= last["time_s"]-3.]
    if tail[-1]["time_s"]-tail[0]["time_s"] < 2.9:
        raise ValueError("Incomplete fixed stall-tail observation")
    drives = {}
    for axle in ("front","rear"):
        fd = slope(tail,"drive_"+axle+"_rad")
        drives[axle] = {"positive_velocity_target_min_rad_s":min(r["drive_"+axle+"_actual_velocity_target_rad_s"] for r in tail),
            "positive_velocity_target_max_rad_s":max(r["drive_"+axle+"_actual_velocity_target_rad_s"] for r in tail),
            "angle_regression_speed_rad_s":fd,"wheel_periphery_angle_regression_m_s":.074*fd,
            "instantaneous_solver_qd_median_rad_s_NOT_angle_derivative":median(r["drive_"+axle+"_velocity_rad_s"] for r in tail),
            "angle_range_rad":max(r["drive_"+axle+"_rad"] for r in tail)-min(r["drive_"+axle+"_rad"] for r in tail)}
    return {"sample_count":len(rows),"observed_progress_m":last["longitudinal_m"],"first_stall_simulation_time_s":last["time_s"],
        "stall_window_duration_s":window[-1]["time_s"]-window[0]["time_s"],
        "stall_window_progress_range_m":max(r["longitudinal_m"] for r in window)-min(r["longitudinal_m"] for r in window),
        "stall_tail_duration_s":tail[-1]["time_s"]-tail[0]["time_s"],
        "stall_tail_longitudinal_regression_m_s":slope(tail,"longitudinal_m"),
        "stall_tail_body_raw_speed_max_m_s":max(math.hypot(r["vx_physx_m_s"],r["vy_physx_m_s"]) for r in tail),
        "stall_tail_filtered_speed_max_m_s":max(r["pose_filtered_speed_m_s"] for r in tail),
        "stall_tail_commands_m_s":sorted(set(r["command_speed_m_s"] for r in tail)),
        "stall_tail_drive_hold_enabled":any(r["position_hold_enabled"] for r in tail),
        "last_requested_curvature_1_m":last["requested_curvature_1_m"],
        "last_command_steering_deg":[math.degrees(last["command_"+axle+"_rad"]) for axle in ("front","rear")],
        "last_actual_steering_deg":[math.degrees(last[axle+"_rad"]) for axle in ("front","rear")],
        "last_caster_joint_angles_deg_NOT_world_wheel_heading":{key:math.degrees(value) for key,value in last.items()
            if key.startswith("joint_") and key.endswith("_caster_rad")},
        "drive_observations":drives,"maximum_applied_drive_target_reproduction_error_rad_s":command_error,
        "maximum_fixed_progress_curvature_reproduction_error_1_m":curvature_error,
        "stall_tail_wheel_mean_vertical_forces_n":[mean(r[f"wheel_{i}_fz_n"] for r in tail) for i in range(6)],
        "stall_tail_every_wheel_vertical_force_positive":all(r[f"wheel_{i}_fz_n"] > 1. for r in tail for i in range(6)),
        "reverse_or_cusp_or_final_hold_entered":any(r["phase"].startswith(("leg_1_","cusp_","terminal_hold")) for r in rows)}


def analyze(path=DEFAULT):
    path=Path(path).resolve()
    record=json.loads(path.read_bytes())
    if (record.get("trial_kind")!=ROLE or record.get("verdict")!="FAIL" or record.get("PhysX_executed") is not True
            or record.get("error")!="RuntimeError: Five-second measured forward-progress stall"
            or record.get("original_motor_gains_restored") is not True or record.get("fixed_hold_attempt_count")!=0
            or record.get("H3B_rolling_readiness_verified") is not False):
        raise ValueError("Need the original completed-cleanup rolling stall FAIL")
    verify_sources(record)
    inputs={str(path):SUPPORT["sha"](path),str(path.with_suffix(".csv")):SUPPORT["sha"](path.with_suffix(".csv"))}
    for key,digest_key in (("geometry_manifest","geometry_manifest_sha256"),("static_reference","static_reference_sha256"),
            ("rolling_preparation_reference","rolling_preparation_sha256"),("H3B_preparation_reference","H3B_preparation_sha256"),
            ("reviewed_geometry_only_warm_start_reference","reviewed_geometry_only_warm_start_sha256")):
        if SUPPORT["sha"](record[key])!=record[digest_key]:
            raise ValueError("Measured rolling input changed: "+key)
        inputs[record[key]]=record[digest_key]
    static_path=Path(record["static_reference"])
    static=json.loads(static_path.read_bytes())
    verify_sources(static)
    if SUPPORT["sha"](static_path.with_suffix(".csv"))!=record["static_trace_sha256"]:
        raise ValueError("Raw static reference changed")
    inputs[str(static_path.with_suffix(".csv"))]=record["static_trace_sha256"]
    if evaluate_static(read_trace(static_path.with_suffix(".csv")),static["start_xyyaw"],static["simulation_mass_kg"],
            static["contact_report_counts_by_category"],static["contact_callback_errors"])["verdict"]!="PASS":
        raise ValueError("The raw static reference no longer reproduces PASS")
    rows=read_trace(path.with_suffix(".csv"))
    metrics=trace_metrics(rows)
    if rows[-1]!=record["last_recorded_state"] or len(rows)!=record["sample_count"]:
        raise ValueError("The final recorded state or sample count differs from the actual CSV")
    categories=record["contact_report_counts_by_category"]
    only_support=bool(categories.get("wheel_support")) and not any(value for key,value in categories.items() if key!="wheel_support")
    if not only_support or record["contact_callback_errors"]:
        raise ValueError("This actual stall was expected to have healthy support-only contact observations")
    if metrics["stall_tail_drive_hold_enabled"] or metrics["stall_tail_commands_m_s"]!=[.01]:
        raise ValueError("Not the recorded low-speed stall with active positive commands and hold released")
    prep=record["physics_preparation_snapshot"]
    SUPPORT["verify_files"]({prep["source_stage"]:prep["source_stage_sha256"],prep["robot_reference"]:prep["robot_reference_sha256"]})
    return {"status":"RECORDED_ROLLING_STALL_REPRODUCED_POSITIVE_COMMAND_WITH_NEAR_ZERO_BODY_AND_WHEEL_ANGLE_PROGRESS_CAUSE_UNISOLATED",
        "original_verdict":"FAIL","metrics":metrics,"contact_categories":categories,
        "input_sha256":inputs,"source_sha256":{Path(__file__).name:SUPPORT["sha"](__file__)},
        "supported_conclusion":"At this 10 mm/s representative steering probe, forward motion and wheel-angle progress stop while positive drive targets remain applied and the position hold is released.",
        "cause_isolated":False,"not_established":["isolated motor torque saturation","friction or caster stiction as a unique cause",
            "global H3 infeasibility","failure of the saved 3.986 m warm-start route","direction-change or final-hold behavior"],
        "same_condition_retry_or_speed_gain_material_guard_change_performed":False,
        "new_PhysX_trial_performed":False,"full_dynamics_plan_certified":False,"full_H3B_turning_path_verified":False,
        "use_for_final_statistics":False,
        "next_step":"Current PAUSED post-cleanup joint/motor/collider configuration readback only; no motion or settings change."}
