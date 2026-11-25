"""A short representative rolling-steering/straight-cusp probe, NOT H3 replay.

Uses the same counterphase geometry, direct actuator adapter and stop/hold
handshake. This deliberately separate component is NOT the saved 3.986 m
route, a planner improvement, or a stationary seven-target alignment PASS.
"""

import math

import numpy as np

from hospital_h3b_readiness import (POLICY, StopAlignGate, SignedStopProfile,
    READINESS, ACCEPTANCE, BAD_CONTACTS, ROBOT_ROOT, WHEELS, GUARDS,
    classify_contact, relative_pose, wheel_crossing_checks, motion_stalled)
from hospital_h3_flush_crossing import evaluate_passage
from hospital_path_execution_v3 import direct_joint_targets


COMMAND_LIMITS = {"forward_speed_m_s": .010, "reverse_speed_m_s": .010,
    "acceleration_m_s2": .020, "steering_command_slew_deg_s": 8., "curvature_limit_1_m": .7}
KNOTS = ((0., 0.), (.020, 0.), (.060, .7), (.100, 0.), (.140, -.7), (.180, 0.), (.220, 0.))
LEGS = ((1, .220), (-1, .170))
ROLE = "H3B_REPRESENTATIVE_ROLLING_STEERING_STRAIGHT_CUSP_COMPONENT_NOT_SAVED_ROUTE_REPLAY"


def curvature_at(progress, phase):
    if not math.isfinite(progress):
        raise ValueError("Finite fixed-start progress required")
    if phase != "leg_0_motion":
        return 0.
    return float(np.interp(progress, *zip(*KNOTS)))


def nominal_component(step=.001):
    """Reference geometry only, fixed to the ORIGINAL start, never recaptured."""
    if not math.isfinite(step) or not 0 < step <= .005:
        raise ValueError("Require dense component reference samples")
    pose = np.zeros(3)
    result = [pose.tolist()]
    for (left, k0), (right, k1) in zip(KNOTS[:-1], KNOTS[1:]):
        n = math.ceil((right - left) / step)
        ds = (right - left) / n
        for j in range(n):
            a, b = k0 + (k1-k0)*j/n, k0 + (k1-k0)*(j+1)/n
            # Knots use fixed longitudinal x, as the actual controller does,
            # not recaptured progress or an unmeasured travelled arc.
            # d(sin(yaw))/dx = curvature and dy/dx = tan(yaw).
            middle = math.asin(math.sin(pose[2]) + ds*(.375*a + .125*b))
            end = math.asin(math.sin(pose[2]) + .5*ds*(a+b))
            pose[0] += ds
            pose[1] += ds/6*(math.tan(pose[2])+4*math.tan(middle)+math.tan(end))
            pose[2] = end
            result.append(pose.tolist())
    forward_end = pose.copy()
    for distance in np.linspace(0., .050, math.ceil(.050/step)+1)[1:]:
        pose = forward_end - np.array([distance*math.cos(forward_end[2]), distance*math.sin(forward_end[2]), 0.])
        result.append(pose.tolist())
    return result


def component_goal(start):
    x, y, yaw = nominal_component()[-1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return start[0]+c*x-s*y, start[1]+s*x+c*y, start[2]+yaw


def evaluate_readiness(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events):
    if any(not math.isfinite(float(row[key])) for row in rows for key in ("command_time_s", "front_rad", "rear_rad",
            "command_front_rad", "command_rear_rad", "requested_curvature_1_m", "longitudinal_m")):
        raise ValueError("Finite actual steering/command/longitudinal observations required")
    common = evaluate_passage(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count)
    excluded = {"all_six_wheels_started_east_and_finished_west", "ordinary_positive_ramp_within_30mm_s", "passage_and_hold_completed"}
    checks = {key: value for key, value in common["checks"].items() if key not in excluded}
    moving = [row for row in rows if row["phase"] == "leg_0_motion" and row["command_speed_m_s"] > .003]
    reverse = [row for row in rows if row["phase"] == "leg_1_motion"]
    peaks = {axle: {"min_deg": math.degrees(min((r[axle+"_rad"] for r in moving), default=0.)),
        "max_deg": math.degrees(max((r[axle+"_rad"] for r in moving), default=0.))} for axle in ("front", "rear")}
    slew = []
    for a, b in zip(rows, rows[1:]):
        dt = b["command_time_s"] - a["command_time_s"]
        if dt < 0.:
            raise ValueError("Command timestamps must not go backwards")
        for axle in ("front", "rear"):
            delta = abs(b["command_"+axle+"_rad"] - a["command_"+axle+"_rad"])
            if dt == 0. and delta > 1e-12:
                raise ValueError("Steering command changed at zero elapsed time")
            slew.append(delta/dt if dt else 0.)
    neutral = [row for row in rows if row["position_hold_enabled"] or row["phase"].startswith("cusp_") or row["phase"].startswith("leg_1_")]
    aligned = [row for row in rows if row["phase"].startswith("cusp_") and row["phase"].endswith(("_alignment", "_ready"))]
    checks.update({"representative_probe_and_one_hold_completed": True,
        "both_actual_steering_signs_observed_while_rolling": all(p["min_deg"] <= -5. and p["max_deg"] >= 5. for p in peaks.values()),
        "both_requested_curvature_signs_exercised": bool(moving) and min(r["requested_curvature_1_m"] for r in moving) <= -.65
            and max(r["requested_curvature_1_m"] for r in moving) >= .65,
        "one_measured_straight_cusp_handshake": len(gate_events) == 1 and all(g["direction"] == -1
            and g["captured_hold_acknowledged"] and g["hold_release_acknowledged"]
            and abs(g["command_speed_m_s"]) < 1e-7 and g["measured_speed_m_s"] <= .005
            and g["steering_error_deg"] <= 2.5 for g in gate_events),
        "straight_actual_stop_alignment_observations": bool(aligned) and all(max(r["pose_filtered_speed_m_s"],
            math.hypot(r["vx_physx_m_s"], r["vy_physx_m_s"])) <= .005 for r in aligned),
        "actual_short_reverse_response_observed": len(reverse) > 1 and reverse[0]["longitudinal_m"] - reverse[-1]["longitudinal_m"] >= .030,
        "all_signed_commands_within_10mm_s": all(abs(r["command_speed_m_s"]) <= .010+1e-9 for r in rows),
        "steering_commands_within_8deg_s": max(slew, default=0.) <= math.radians(8.)+1e-5,
        "all_hold_commands_zero_speed": all(abs(r["command_speed_m_s"]) < 1e-7 for r in rows if r["position_hold_enabled"]),
        "no_nonzero_steering_commands_during_hold_cusp_or_reverse": bool(neutral) and all(max(abs(r["command_front_rad"]),
            abs(r["command_rear_rad"])) < 1e-7 for r in neutral),
        "actual_neutral_steering_before_drive_hold": all(max(abs(r["front_rad"]), abs(r["rear_rad"])) <= math.radians(2.5)
            for r in rows if r["position_hold_enabled"]),
        "actual_neutral_tail_before_first_stop": any(r["phase"] == "leg_0_motion" and r["longitudinal_m"] >= .190
            and max(abs(r["command_front_rad"]), abs(r["command_rear_rad"]), abs(r["front_rad"]), abs(r["rear_rad"])) <= math.radians(2.5) for r in rows)})
    return {"checks": {key: bool(value) for key, value in checks.items()}, "verdict": "PASS" if all(checks.values()) else "FAIL",
        **{key: common[key] for key in ("terminal_position_error_m", "terminal_heading_error_deg", "final_world_xyyaw",
            "fixed_terminal_hold_analysis", "final_force_weight_ratio")},
        "measured_moving_steering_peaks_deg": peaks, "maximum_observed_command_slew_deg_s": math.degrees(max(slew, default=0.)),
        "representative_component_only_not_saved_warm_start_replay": True,
        "nonzero_stationary_alignment_physically_verified": False,
        "full_H3B_turning_path_verified": False, "full_dynamics_plan_certified": False}
