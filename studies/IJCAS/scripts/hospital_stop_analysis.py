"""Read-only diagnostics for zero-command wheel rolling versus chassis creep.

This does not identify a unique PhysX/material fault and cannot approve a path
execution. Thresholds are development standstill checks, not paper tolerances.
"""

import math


def _slope(times, values):
    center = sum(times)/len(times)
    mean = sum(values)/len(values)
    denominator = sum((t-center)**2 for t in times)
    if denominator <= 0:
        raise ValueError("Need a nonzero measurement interval")
    return sum((t-center)*(v-mean) for t, v in zip(times, values))/denominator


def _joint_travel(values, radius):
    # Handles both continuous positions and +/-pi wrapped wheel angles.
    return radius*sum(math.atan2(math.sin(b-a), math.cos(b-a)) for a, b in zip(values, values[1:]))


def analyze_stop_trace(rows, wheel_radius_m=.074):
    if len(rows) < 10 or not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0:
        raise ValueError("Need enough samples and a positive wheel radius")
    keys = ("time_s", "x_m", "y_m", "yaw_rad", "command_speed_m_s",
            "drive_front_rad", "drive_rear_rad", "drive_front_velocity_rad_s", "drive_rear_velocity_rad_s",
            "vx_physx_m_s", "vy_physx_m_s", "vz_physx_m_s")
    if not all(math.isfinite(float(row[key])) for row in rows for key in keys):
        raise ValueError("Non-finite zero-command trace")
    times = [float(row["time_s"]) for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])) or times[-1]-times[0] < .5:
        raise ValueError("Need increasing timestamps and at least 0.5 seconds")
    if any(abs(float(row["command_speed_m_s"])) > 1e-9 for row in rows):
        raise ValueError("This analyzer accepts only zero-command intervals")
    heading = float(rows[0]["yaw_rad"])
    cosine, sine = math.cos(heading), math.sin(heading)
    x0, y0 = float(rows[0]["x_m"]), float(rows[0]["y_m"])
    forward = [cosine*(float(r["x_m"])-x0)+sine*(float(r["y_m"])-y0) for r in rows]
    lateral = [-sine*(float(r["x_m"])-x0)+cosine*(float(r["y_m"])-y0) for r in rows]
    vx = _slope(times, [float(r["x_m"]) for r in rows])
    vy = _slope(times, [float(r["y_m"]) for r in rows])
    duration = times[-1]-times[0]
    drive_travel = {axle: _joint_travel([float(r["drive_"+axle+"_rad"]) for r in rows], wheel_radius_m)
                    for axle in ("front", "rear")}
    front_travel, rear_travel = drive_travel["front"], drive_travel["rear"]
    chord_speed = math.hypot(vx, vy)
    drift = max(math.hypot(float(r["x_m"])-x0, float(r["y_m"])-y0) for r in rows)
    heading_span = max(abs(math.degrees(math.atan2(math.sin(float(r["yaw_rad"])-heading),
                                                math.cos(float(r["yaw_rad"])-heading)))) for r in rows)
    near_straight = heading_span <= .1 and all(abs(float(r.get(key, 0.))) <= math.radians(.5)
        for r in rows for key in ("front_rad", "rear_rad"))
    stable = chord_speed <= .001 and drift <= .003
    if stable:
        classification = "no_sustained_creep_in_this_observation_window"
    elif not near_straight:
        classification = "motion_observed_but_straight_rolling_comparison_not_applicable"
    elif abs(lateral[-1]) >= .003 and abs(lateral[-1]) > abs(forward[-1]):
        classification = "lateral_chassis_drift_not_explained_by_straight_active_wheel_rolling"
    elif abs(forward[-1]) >= .003 and max(abs(front_travel), abs(rear_travel)) < .2*abs(forward[-1]):
        classification = "chassis_motion_with_little_active_wheel_rotation"
    elif abs(forward[-1]) >= .003 and min(abs(front_travel), abs(rear_travel)) >= .5*abs(forward[-1]):
        classification = "active_wheel_rolling_despite_zero_speed_command"
    else:
        classification = "mixed_rolling_and_chassis_motion_needs_contact_and_drive_inspection"
    return {
        "duration_s": duration, "sample_count": len(rows),
        "pose_regression_speed_m_s": chord_speed,
        "forward_regression_speed_m_s": cosine*vx+sine*vy,
        "maximum_xy_displacement_m": drift,
        "forward_displacement_m": forward[-1], "lateral_displacement_m": lateral[-1],
        "maximum_heading_change_deg": heading_span,
        "active_wheel_roll_travel_m": drive_travel,
        "active_wheel_mean_absolute_rolling_speed_m_s": {
            axle: wheel_radius_m*sum(abs(float(r["drive_"+axle+"_velocity_rad_s"])) for r in rows)/len(rows)
            for axle in ("front", "rear")},
        "mean_physx_speed_m_s": sum(math.hypot(float(r["vx_physx_m_s"]), float(r["vy_physx_m_s"])) for r in rows)/len(rows),
        "mean_physx_vertical_velocity_m_s": sum(float(r["vz_physx_m_s"]) for r in rows)/len(rows),
        "straight_rolling_comparison_applicable": near_straight,
        "observation_classification": classification,
        "development_standstill_check": "PASS" if stable else "HOLD",
        "development_thresholds": {"pose_regression_speed_m_s_max": .001, "xy_displacement_m_max": .003},
        "causal_limit": "Wheel/body mismatch suggests sliding or contact/solver motion; it does not identify a unique cause.",
        "does_not_validate": ["moving_stop_response", "hospital_path_execution", "CICS_initial_motion", "physical_platform"],
    }
