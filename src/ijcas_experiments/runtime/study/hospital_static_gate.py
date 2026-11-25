"""Pure H2 stationary-support verdict rules. Never a path-execution verdict.

Keep the H1 load/contact/drift criteria unchanged. Explicitly require a
complete observation and a healthy contact callback before accepting H2.
"""

import math

import numpy as np


WHEEL_LABELS = ("active_front", "active_rear", "caster_fl", "caster_fr", "caster_rl", "caster_rr")
BAD_CONTACTS = ("hospital_obstacle", "unexpected_legacy_support", "nonwheel_support", "unclassified_external")
LIMITS = {"mean_wheel_load_n_min": 1., "loaded_duty_min": .95,
    "force_weight_ratio_min": .8, "force_weight_ratio_max": 1.2, "total_force_cv_max": .15,
    "z_span_m_max": .005, "roll_pitch_deg_max": 1., "xy_drift_m_max": .010,
    "minimum_samples": 30, "minimum_observation_s": 1.9}


def evaluate_static(rows, start_xyyaw, mass_kg, contact_counts, callback_errors=()):
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Expected positive finite simulation mass")
    if len(rows) < LIMITS["minimum_samples"]:
        raise ValueError("Too few H2 stationary samples")
    keys = ("time_s", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg")
    keys += tuple(name+"_fz_n" for name in WHEEL_LABELS)
    if not all(math.isfinite(float(row[key])) for row in rows for key in keys):
        raise ValueError("Non-finite H2 stationary observation")
    times = np.array([row["time_s"] for row in rows], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Stationary timestamps must be increasing")
    if any(not math.isfinite(v) for v in start_xyyaw) or len(start_xyyaw) != 3:
        raise ValueError("Invalid original H2 start")
    vertical = np.array([[row[label+"_fz_n"] for label in WHEEL_LABELS] for row in rows])
    means = vertical.mean(axis=0)
    duty = (vertical > LIMITS["mean_wheel_load_n_min"]).mean(axis=0)
    totals = vertical.sum(axis=1)
    ratio = float(totals.mean()/(mass_kg*9.81))
    cv = float(totals.std()/max(abs(totals.mean()), 1e-9))
    z_span = float(np.ptp([row["z_m"] for row in rows]))
    tilt = max(abs(row[key]) for row in rows for key in ("roll_deg", "pitch_deg"))
    drift = max(math.hypot(row["x_m"]-start_xyyaw[0], row["y_m"]-start_xyyaw[1]) for row in rows)
    checks = {
        "complete_stationary_observation": times[-1]-times[0] >= LIMITS["minimum_observation_s"],
        "all_six_wheels_mean_load_over_1N": bool(np.all(means > 1.)),
        "all_six_wheels_load_bearing_duty_over_95_percent": bool(np.all(duty >= .95)),
        "vertical_force_weight_balance": .8 <= ratio <= 1.2,
        "total_force_cv_under_15_percent": cv <= .15,
        "base_z_span_under_5mm": z_span <= .005,
        "base_roll_pitch_under_1deg": tilt <= 1.,
        "base_xy_drift_under_10mm": drift <= .010,
        "no_hospital_obstacle_contacts": contact_counts.get("hospital_obstacle", 0) == 0,
        "no_duplicate_legacy_floor_contacts": contact_counts.get("unexpected_legacy_support", 0) == 0,
        "only_wheels_touch_support_floor": contact_counts.get("nonwheel_support", 0) == 0,
        "no_unclassified_external_contacts": contact_counts.get("unclassified_external", 0) == 0,
        "expected_support_floor_contact_observed": contact_counts.get("wheel_support", 0) > 0,
        "contact_callback_monitor_healthy": not callback_errors,
    }
    return {"checks": {key: bool(value) for key, value in checks.items()},
        "verdict": "PASS" if all(checks.values()) else "FAIL", "sample_count": len(rows),
        "mean_wheel_vertical_loads_n": dict(zip(WHEEL_LABELS, map(float, means))),
        "wheel_load_bearing_duty": dict(zip(WHEEL_LABELS, map(float, duty))),
        "mean_vertical_force_to_weight_ratio": ratio, "total_force_cv": cv, "base_z_span_m": z_span,
        "maximum_abs_roll_pitch_deg": tilt, "maximum_xy_drift_m": drift,
        "observation_duration_s": float(times[-1]-times[0]), "acceptance": dict(LIMITS)}
