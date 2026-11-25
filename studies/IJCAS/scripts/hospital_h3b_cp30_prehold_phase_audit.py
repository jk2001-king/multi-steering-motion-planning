"""Read-only phase comparison; never equate fresh static with a post-Play row.

The original CP30 CSV starts AFTER hold engagement and simulated progress.
Its static CSV has body/support data but no twelve-joint position snapshot.
Report every raw discrepancy, not a selected or reset successful state. The
old equality tolerances are reproduced descriptively, never waived here.
"""

import math

import numpy as np


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def compare(names, positions, transform, first, observed_prehold_pose, static_last, paused_xy_tolerance):
    names = list(names)
    positions = np.asarray(positions, dtype=float).reshape(-1)
    transform = np.asarray(transform, dtype=float).reshape(-1)
    if (len(names) != 12 or len(set(names)) != 12 or positions.size != 12 or transform.size != 7
            or not np.isfinite(positions).all() or not np.isfinite(transform).all()
            or not math.isfinite(paused_xy_tolerance) or paused_xy_tolerance <= 0):
        raise ValueError("Finite measured root and all twelve named joints required")
    if first.get("phase") != "pre_hold" or first.get("position_hold_enabled") is not True or first["time_s"] <= 0:
        raise ValueError("Require the actual historical POST-engagement/POST-Play first sample")
    if any("joint_" + name + "_rad" not in first for name in names):
        raise ValueError("Historical joint names do not cover the actual twelve-joint set")
    qx, qy, qz, qw = transform[3:]
    if abs(float(np.linalg.norm(transform[3:])) - 1.) > 1e-3:
        raise ValueError("Invalid measured root quaternion")
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    reference_values = [first[k] for k in ("x_m", "y_m", "yaw_rad")]
    static_pose = [static_last["x_m"], static_last["y_m"], math.radians(static_last["yaw_deg"])]
    if not all(math.isfinite(float(v)) for v in (*reference_values, *static_pose, *observed_prehold_pose)):
        raise ValueError("Finite saved reference and fresh static body data required")
    live = [float(transform[0]), float(transform[1]), yaw]
    def delta(pose):
        return dict(position_difference_m=math.hypot(live[0] - pose[0], live[1] - pose[1]),
                    heading_difference_deg=math.degrees(wrap(yaw - pose[2])))
    body_old, body_static, body_pre = [delta(pose) for pose in (reference_values, static_pose, observed_prehold_pose)]
    joints = []
    for name, value in zip(names, positions):
        old = float(first["joint_" + name + "_rad"])
        if not math.isfinite(old):
            raise ValueError("Nonfinite historical joint state")
        difference = float(value) - old
        group = ("caster_swivel" if name.endswith("_caster") else
                 "wheel_rotation" if "_to_wheel" in name or "_to_drive_wheel" in name else "active_steering")
        joints.append(dict(joint=name, group=group, live_rad=float(value),
                           historical_post_engagement_rad=old, direct_difference_rad=difference,
                           direct_difference_deg=math.degrees(difference),
                           exceeds_old_equality_0p001rad=abs(difference) > .001))
    maximum = max(abs(row["direct_difference_rad"]) for row in joints)
    return dict(status="CP30_STATIC_VS_POST_ENGAGEMENT_PHASE_DIFFERENCES_RECORDED_NO_DRIVE",
        live_world_xyyaw=live, live_all_twelve_joint_positions_rad=dict(zip(names, map(float, positions))),
        historical_first_phase=first["phase"], historical_first_simulated_time_s=first["time_s"],
        historical_first_sample_is_AFTER_hold_engagement_and_Play=True,
        comparison_phases_are_NOT_identical=True,
        live_vs_historical_post_engagement_body=body_old,
        live_vs_historical_observed_PRE_engagement_body=body_pre,
        live_vs_new_actual_static_body=body_static,
        new_static_BODY_matches_existing_paused_xy_tolerance=body_static["position_difference_m"] <= paused_xy_tolerance,
        existing_paused_xy_tolerance_m=paused_xy_tolerance,
        per_joint_differences=joints, maximum_absolute_direct_joint_difference_rad=maximum,
        old_wrong_phase_equality_would_pass=bool(body_old["position_difference_m"] <= .001
            and abs(body_old["heading_difference_deg"]) <= .1 and maximum <= .001),
        new_static_CSV_contains_twelve_joint_reference=False,
        no_missing_historical_static_joint_values_invented=True,
        state_or_acceptance_threshold_modified=False, new_PhysX_executed=False,
        whole_H3_drive_admission=False, component_drive_admission=False)
