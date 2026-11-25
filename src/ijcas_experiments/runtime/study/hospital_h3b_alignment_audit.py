"""Read recorded stationary-steering failures; never relax a measured gate.

Position finite differences and solver velocity samples are different signals.
Their discrepancy is evidence to investigate, not proof of sensor noise or of
one particular motor/contact fault. No controller or original result is edited.
"""

import math
from pathlib import Path
import json
from statistics import median

from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources
from hospital_h3b_evidence import read_trace
from hospital_static_gate import evaluate_static


DEFAULT = GATE / "hospital_h3b_alignment/hospital_h3b_alignment_r01.json"
REASON = "Measured motion exceeded stationary steering 5 mm/s stop threshold"


def summarize_trace(rows, capture, release, hold_targets):
    if len(rows) < 2 or any(b["time_s"] <= a["time_s"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Need an ordered, nonempty measured alignment trace")
    numeric = [value for row in rows for value in row.values() if isinstance(value, (int, float))]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Nonfinite recorded alignment state")
    raw = lambda row: math.hypot(row["vx_physx_m_s"], row["vy_physx_m_s"])
    violations = [i for i, row in enumerate(rows) if max(raw(row), row["pose_filtered_speed_m_s"]) > .005]
    if not violations:
        raise ValueError("Recorded stop-speed failure does not reproduce from raw data")
    phases = []
    for phase in dict.fromkeys(row["phase"] for row in rows):
        sample = [row for row in rows if row["phase"] == phase]
        phases.append({"phase": phase, "sample_count": len(sample),
            "time_range_s": [sample[0]["time_s"], sample[-1]["time_s"]],
            "raw_horizontal_speed_median_m_s": median(raw(row) for row in sample),
            "raw_horizontal_speed_max_m_s": max(raw(row) for row in sample),
            "pose_filtered_speed_max_m_s": max(row["pose_filtered_speed_m_s"] for row in sample),
            "capture_xy_displacement_max_m": max(math.dist((row["x_m"], row["y_m"]), capture[:2]) for row in sample),
            "reported_vertical_speed_median_m_s": median(row["vz_physx_m_s"] for row in sample),
            "base_z_span_m": max(row["z_m"] for row in sample) - min(row["z_m"] for row in sample)})
    differences = []
    for previous, current in zip(rows, rows[1:]):
        dt = current["time_s"] - previous["time_s"]
        differences.append({"time_s": current["time_s"], "dt_s": dt,
            "position_difference_horizontal_speed_m_s": math.dist((current["x_m"], current["y_m"]),
                (previous["x_m"], previous["y_m"])) / dt,
            "position_difference_vertical_speed_m_s": (current["z_m"] - previous["z_m"]) / dt,
            **{axle + "_angle_difference_speed_rad_s": (current["drive_" + axle + "_rad"] - previous["drive_" + axle + "_rad"]) / dt
               for axle in ("front", "rear")}})
    drives = {}
    for index, axle in enumerate(("front", "rear")):
        drives[axle] = {
            "reported_joint_velocity_abs_max_rad_s": max(abs(row["drive_" + axle + "_velocity_rad_s"]) for row in rows),
            "angle_difference_velocity_abs_max_rad_s": max(abs(row[axle + "_angle_difference_speed_rad_s"]) for row in differences),
            "recorded_hold_angle_error_abs_max_rad": max(abs(row["drive_" + axle + "_rad"] - hold_targets[index]) for row in rows),
            "recorded_velocity_targets_all_zero": all(abs(row["drive_" + axle + "_actual_velocity_target_rad_s"]) < 1e-9 for row in rows)}
    index = violations[0]
    first = rows[index]
    return {"sample_count": len(rows), "first_stop_speed_violation_index": index,
        "first_stop_speed_violation_s": first["time_s"], "first_stop_speed_violation_phase": first["phase"],
        "first_stop_speed_violation_phase_time_s": first["phase_time_s"], "stop_speed_limit_m_s": .005,
        "violation_raw_horizontal_speed_m_s": raw(first), "violation_filtered_speed_m_s": first["pose_filtered_speed_m_s"],
        "violation_xy_displacement_from_capture_m": math.dist((first["x_m"], first["y_m"]), capture[:2]),
        "violation_xy_displacement_from_steering_start_m": math.dist((first["x_m"], first["y_m"]), release[:2]),
        "violation_command_front_rear_deg": [math.degrees(first[axis + "_command_rad"]) for axis in ("front", "rear")],
        "violation_actual_front_rear_deg": [math.degrees(first[axis + "_rad"]) for axis in ("front", "rear")],
        "violation_position_difference_speed_m_s": differences[index - 1]["position_difference_horizontal_speed_m_s"] if index else None,
        "zero_translation_commands_entire_trace": all(abs(row["command_speed_m_s"]) < 1e-9 for row in rows),
        "recorded_hold_flag_all_true": all(row["position_hold_enabled"] is True for row in rows),
        "hold_position_targets_not_sampled_in_original_CSV": True,
        "phases": phases, "drive_observations": drives,
        "sampling_interval_median_s": median(row["dt_s"] for row in differences),
        "full_target_observation_count": sum(phase["phase"].endswith("_observation") for phase in phases),
        "finite_differences_are_not_substep_synchronized_velocity_ground_truth": True}


def analyze(path=DEFAULT):
    path = Path(path).resolve()
    report = json.loads(path.read_bytes())
    if (report.get("scenario") != "H3B_stationary_steering_alignment_readiness" or report.get("verdict") != "FAIL"
            or report.get("safety_abort", {}).get("reason") != REASON
            or not report.get("original_motor_gains_restored") or report.get("nonzero_steering_alignment_physically_verified") is not False):
        raise ValueError("Need the original measured stop-speed alignment FAIL with safe cleanup")
    verify_sources(report)
    inputs = {str(path): SUPPORT["sha"](path), str(path.with_suffix(".csv")): SUPPORT["sha"](path.with_suffix(".csv"))}
    for key, digest_key in (("static_reference", "static_reference_sha256"), ("H3B_preparation_reference", "H3B_preparation_sha256"),
            ("alignment_preparation_reference", "alignment_preparation_sha256"), ("geometry_manifest", "geometry_manifest_sha256"),
            ("shared_hold_component_reference", "shared_hold_component_sha256")):
        source = Path(report[key])
        if SUPPORT["sha"](source) != report[digest_key]:
            raise ValueError("Changed measured input: " + str(source))
        inputs[str(source)] = report[digest_key]
    prep = report["physics_preparation_snapshot"]
    for key in ("source_stage", "robot_reference"):
        if SUPPORT["sha"](prep[key]) != prep[key + "_sha256"]:
            raise ValueError("Changed original source USD: " + prep[key])
        inputs[prep[key]] = prep[key + "_sha256"]
    static_path = Path(report["static_reference"])
    static = json.loads(static_path.read_bytes())
    verify_sources(static)
    if SUPPORT["sha"](static_path.with_suffix(".csv")) != report["static_trace_sha256"]:
        raise ValueError("Changed measured static trace")
    inputs[str(static_path.with_suffix(".csv"))] = report["static_trace_sha256"]
    if evaluate_static(read_trace(static_path.with_suffix(".csv")), static["start_xyyaw"], static["simulation_mass_kg"],
            static["contact_report_counts_by_category"], static["contact_callback_errors"])["verdict"] != "PASS":
        raise ValueError("Static PASS does not reproduce")
    rows = read_trace(path.with_suffix(".csv"))
    if len(rows) != report["sample_count"] or rows[-1] != report["safety_abort"]["actual_state"]:
        raise ValueError("Raw sample count or recorded offending state mismatch")
    metrics = summarize_trace(rows, report["observed_prehold_world_xyyaw"], report["release_world_xyyaw"], report["captured_drive_hold_targets_rad"])
    if metrics["first_stop_speed_violation_index"] != len(rows) - 1:
        raise ValueError("The reported failure was not the first measured speed violation")
    if not metrics["zero_translation_commands_entire_trace"] or not metrics["recorded_hold_flag_all_true"]:
        raise ValueError("Recorded stationary command/hold sequence differs")
    return {"status": "RAW_STATIONARY_ALIGNMENT_FAIL_REPRODUCED_CAUSE_NOT_ISOLATED", "original_verdict": "FAIL",
        "run_id": report["run_id"], "metrics": metrics, "contact_categories": report["contact_report_counts_by_category"],
        "original_motor_gains_restored": True, "input_sha256": inputs, "original_source_hashes_match": True,
        "supported_conclusion": "A raw horizontal-speed sample crossed 5 mm/s during the first steering ramp; nonzero stationary alignment was not demonstrated.",
        "not_established": ["noise-only false positive", "unique motor/contact/solver fault", "isolated motor torque saturation",
            "all stationary steering or full H3-B motion is impossible", "permission to relax the original stop gate"],
        "next_gate": "ONE_same_condition_passive_high_rate_joint_body_contact_observation_with_unchanged_controller_and_guards",
        "original_result_modified": False, "new_PhysX_trial_performed": False,
        "nonzero_steering_alignment_physically_verified": False, "full_H3B_turning_path_verified": False,
        "use_for_final_statistics": False}


def telemetry_snapshot(view, time_s):
    """Extend the frozen serializer with raw root velocities; observation only."""
    from hospital_h3_instrumentation import joint_snapshot, vector
    import numpy as np
    row = joint_snapshot(view, time_s)
    velocity = vector(view.get_root_velocities(), 6)
    for key, value in zip(("base_vx_m_s", "base_vy_m_s", "base_vz_m_s", "base_wx_rad_s", "base_wy_rad_s", "base_wz_rad_s"),
            velocity):
        row[key] = value
    # SDK root linear velocity is at the root LINK COM, whereas its transform
    # is at the link origin. Record both point velocities, not a changed gate.
    names = list(view.shared_metatype.link_names)
    index = names.index("base_link")
    com = np.asarray(vector(view.get_coms(), len(names) * 7)).reshape(-1, 7)[index, :3]
    quaternion = np.asarray([row[key] for key in ("base_qx", "base_qy", "base_qz", "base_qw")])
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.) > 1e-3:
        raise ValueError("Invalid measured base orientation")
    quaternion /= norm
    xyz, w = quaternion[:3], quaternion[3]
    world_offset = com + 2. * np.cross(xyz, np.cross(xyz, com) + w * com)
    origin_velocity = np.asarray(velocity[:3]) - np.cross(velocity[3:], world_offset)
    for axis, local, origin in zip("xyz", com, origin_velocity):
        row["base_com_local_" + axis + "_m"] = float(local)
        row["base_origin_v" + axis + "_m_s"] = float(origin)
    return row


def telemetry_coverage(joint_rows, contact_rows, primary, errors):
    end = primary.get("last_recorded_state", {}).get("time_s")
    okay_times = len(joint_rows) >= 2 and all(b["time_s"] > a["time_s"] for a, b in zip(joint_rows, joint_rows[1:]))
    max_gap = max((b["time_s"] - a["time_s"] for a, b in zip(joint_rows, joint_rows[1:])), default=None)
    end_covered = bool(end is not None and joint_rows and joint_rows[-1]["time_s"] >= end - 2. / 120.)
    start_covered = bool(joint_rows and joint_rows[0]["time_s"] <= 2. / 120.)
    contacts_complete = bool(contact_rows) and all(row["complete_point_data"] for row in contact_rows)
    complete = bool(not errors and okay_times and max_gap <= 2. / 120. + 1e-6 and start_covered and end_covered
        and contacts_complete and primary.get("original_motor_gains_restored"))
    return {"status": "RECORDED_FOR_CAUSAL_REVIEW_NOT_AN_ALIGNMENT_PASS" if complete else "INCOMPLETE_INSTRUMENTATION_DO_NOT_INFER_CAUSE",
        "joint_sample_count": len(joint_rows), "contact_point_count": len(contact_rows), "maximum_joint_sample_gap_s": max_gap,
        "telemetry_covers_controller_start": start_covered, "telemetry_covers_controller_end": end_covered,
        "raw_contact_points_complete": contacts_complete}
