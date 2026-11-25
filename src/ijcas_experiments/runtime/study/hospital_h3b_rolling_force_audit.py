"""Reconcile the saved same-condition rolling friction observation.

This is an observational contact-force balance, not a causal intervention or
an isolated motor-torque measurement. Aggregate Ft/Fz is NOT an anchor-level
Coulomb-limit test. No recorded controller/outcome/initial state is modified.
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources
from hospital_h3b_rolling_failure import analyze as analyze_failure, slope
from hospital_h3b_rolling_observation import coverage, review_readback


DEFAULT=GATE/"hospital_h3b_rolling_readiness/instrumented_repeats/hospital_h3b_rolling_readiness_r02/instrumentation.json"


def numeric_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:
        rows=[{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    if not rows or any(not math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError("Finite actual observer rows required")
    if any(b["time_s"]<=a["time_s"] for a,b in zip(rows,rows[1:])):
        raise ValueError("Observer timestamps must increase")
    return rows


def anchor_sums_match(friction_rows,anchor_rows,sensor_paths):
    groups={}
    for raw in anchor_rows:
        key=(float(raw["time_s"]),raw["sensor_path"])
        groups.setdefault(key,[]).append(raw)
    expected=set()
    maximum_error=0.
    for row in friction_rows:
        for i,path in enumerate(sensor_paths):
            key=(row["time_s"],path)
            expected.add(key)
            data=groups.get(key,[])
            count=row[f"wheel_{i}__friction_anchor_count"]
            if count!=len(data) or count<0 or count!=int(count):
                raise ValueError("Raw friction anchors do not match the reported sensor counts")
            if sorted(int(r["anchor_index_within_pair"]) for r in data)!=list(range(int(count))):
                raise ValueError("Duplicate or missing anchor index")
            for axis in "xyz":
                values=[float(r["sdk_tangent_f"+axis+"_n"]) for r in data]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Nonfinite raw anchor force")
                # The native SDK buffer and the saved observer's sum are float32.
                # Reproduce that operation, not Python's different float64 sum.
                if any(float(np.float32(v))!=v for v in values):
                    raise ValueError("Raw anchor is not an exact native float32 value")
                total=float(np.asarray(values,dtype=np.float32).sum())
                maximum_error=max(maximum_error,abs(total-row[f"wheel_{i}__sdk_tangent_sum_f{axis}_n"]))
    if set(groups)-expected or maximum_error>1e-8:
        raise ValueError("Friction sum or timestamp/sensor lineage mismatch")
    return maximum_error


def force_summary(friction_tail,sensor_paths,forward_xy):
    if len(friction_tail)<2 or friction_tail[-1]["time_s"]-friction_tail[0]["time_s"]<2.9:
        raise ValueError("Need the fixed final-three-second friction interval")
    if len(sensor_paths)!=6 or len(set(sensor_paths))!=6 or not math.isclose(math.hypot(*forward_xy),1.,abs_tol=1e-12):
        raise ValueError("Six unique sensors and a unit fixed-start projection direction required")
    wheels=[]
    for i,path in enumerate(sensor_paths):
        projected=[]
        ratios=[]
        for row in friction_tail:
            fx,fy=(row[f"wheel_{i}__sdk_tangent_sum_f{axis}_n"] for axis in "xy")
            fz=row[f"wheel_{i}__sdk_reported_contact_fz_n"]
            if not all(math.isfinite(v) for v in (fx,fy,fz)) or fz<=1.:
                raise ValueError("Nonfinite or unsupported wheel in the analyzed stall tail")
            projected.append(fx*forward_xy[0]+fy*forward_xy[1])
            ratios.append(math.hypot(fx,fy)/fz)
        name=Path(path).name
        if name not in ("front_active_drive_wheel_link","rear_active_drive_wheel_link",
                "front_left_caster_wheel_link","front_right_caster_wheel_link","rear_left_caster_wheel_link","rear_right_caster_wheel_link"):
            raise ValueError("Unexpected saved native wheel path")
        wheels.append({"sensor_path":path,"group":"active_drive" if "active_drive" in name else "caster",
            "mean_sdk_tangent_projection_along_fixed_start_forward_n":mean(projected),
            "mean_sdk_normal_vertical_n":mean(row[f"wheel_{i}__sdk_reported_contact_fz_n"] for row in friction_tail),
            "mean_net_horizontal_tangent_norm_over_vertical_normal":mean(ratios),
            "maximum_net_horizontal_tangent_norm_over_vertical_normal":max(ratios),
            "net_ratio_does_NOT_rule_out_anchor_level_saturation":True})
    active=sum(r["mean_sdk_tangent_projection_along_fixed_start_forward_n"] for r in wheels if r["group"]=="active_drive")
    caster=sum(r["mean_sdk_tangent_projection_along_fixed_start_forward_n"] for r in wheels if r["group"]=="caster")
    if sum(r["group"]=="active_drive" for r in wheels)!=2:
        raise ValueError("Need exactly two active wheels and four casters")
    return {"fixed_tail_duration_s":friction_tail[-1]["time_s"]-friction_tail[0]["time_s"],"friction_tail_sample_count":len(friction_tail),
        "per_wheel":wheels,"active_drive_sdk_forward_projection_sum_n":active,"caster_sdk_forward_projection_sum_n":caster,
        "all_wheels_sdk_forward_projection_sum_n":active+caster,
        "opposing_group_projection_observed":active>0. and caster<0.,
        "projection_is_relative_to_original_task_start_forward_not_instantaneous_yaw":True,
        "SDK_world_vector_sign_convention_retained":True,"not_an_isolated_motor_torque_or_unique_cause":True}


def analyze(path=DEFAULT):
    path=Path(path).resolve()
    observer=json.loads(path.read_bytes())
    if (observer.get("status")!="RECORDED_FOR_CAUSAL_REVIEW_NOT_A_ROLLING_OR_TRAVERSAL_PASS"
            or observer.get("telemetry_errors") or observer.get("control_or_physics_parameters_modified_by_observer") is not False
            or observer.get("original_guards_and_verdict_policy_unchanged") is not True):
        raise ValueError("Need the completed unchanged-controller same-condition observation")
    verify_sources(observer)
    SUPPORT["verify_files"](observer["reference_input_sha256"])
    reference_path=Path(observer["reference_trial"])
    primary_path=Path(observer["original_controller_result"])
    for source,expected in ((reference_path,observer["reference_trial_sha256"]),(primary_path,observer["original_controller_result_sha256"])):
        if SUPPORT["sha"](source)!=expected:
            raise ValueError("Measured reference/result changed")
    reference,primary=(json.loads(p.read_bytes()) for p in (reference_path,primary_path))
    original_audit,repeated_audit=analyze_failure(reference_path),analyze_failure(primary_path)
    if primary["source_sha256"]!=reference["source_sha256"] or primary["source_sha256"]!=observer["original_controller_source_sha256"]:
        raise ValueError("The measured controller changed between trials")
    for key in ("environment_condition","source_stage_sha256","robot_reference_sha256","start_xyyaw","component_end_xyyaw",
            "simulation_mass_kg","live_support_readback","support_floor","disabled_legacy_floor_colliders"):
        if primary["physics_preparation_snapshot"][key]!=reference["physics_preparation_snapshot"][key]:
            raise ValueError("Physical baseline changed between rolling trials: "+key)
    inputs={**original_audit["input_sha256"],**repeated_audit["input_sha256"],str(path):SUPPORT["sha"](path)}
    for name,digest in observer["saved_file_sha256"].items():
        source=path.parent/name
        if SUPPORT["sha"](source)!=digest:
            raise ValueError("Recorded observer data changed: "+name)
        inputs[str(source)]=digest
    joint=numeric_rows(path.parent/"joint_body_solver_trace.csv")
    friction=numeric_rows(path.parent/"wheel_friction_summary.csv")
    with (path.parent/"friction_anchors.csv").open(newline="",encoding="utf-8") as stream:
        anchors=list(csv.DictReader(stream))
    with (path.parent/"raw_contact_points.csv").open(newline="",encoding="utf-8") as stream:
        contacts=[{**row,"complete_point_data":row["complete_point_data"]=="True"} for row in csv.DictReader(stream)]
    covered=coverage(joint,friction,anchors,contacts,primary,[])
    for key,value in covered.items():
        if observer.get(key)!=value:
            raise ValueError("Raw telemetry coverage differs from its recorded summary: "+key)
    sensors=observer["native_sensor_paths_in_recorded_wheel_index_order"]
    error=anchor_sums_match(friction,anchors,sensors)
    end=primary["last_recorded_state"]["time_s"]
    tail=[r for r in friction if end-3.<=r["time_s"]<=end+1e-6]
    joint_tail=[r for r in joint if end-3.<=r["time_s"]<=end+1e-6]
    yaw=primary["start_world_xyyaw"][2]
    forces=force_summary(tail,sensors,(math.cos(yaw),math.sin(yaw)))
    drives={}
    for axle in ("front","rear"):
        name=axle+"_active_steering_to_drive_wheel"
        drives[axle]={"all_motion_samples_position_gain_zero":all(r[name+"__kp_tensor_units"]==0. for r in joint if r["time_s"]>2.1),
            "stall_tail_velocity_target_min_rad_s":min(r[name+"__qd_target_rad_s"] for r in joint_tail),
            "stall_tail_angle_regression_rad_s":slope(joint_tail,name+"__q_rad"),
            "stall_tail_mean_projected_solver_effort_nm_NOT_isolated_actuator_torque":mean(r[name+"__projected_solver_joint_effort_nm"] for r in joint_tail),
            "stall_tail_drive_force_limit_nm":sorted(set(r[name+"__drive_force_limit_nm"] for r in joint_tail))}
    reviewed=review_readback()
    return {"status":"SAME_CONDITION_ROLLING_STALL_REPRODUCED_OPPOSING_WHEEL_CONTACT_FORCES_RECORDED_CAUSAL_INTERVENTION_PENDING",
        "original_verdicts":[reference["verdict"],primary["verdict"]],"same_controller_and_physical_baseline":True,
        "primary_traces_byte_identical":reference_path.with_suffix(".csv").read_bytes()==primary_path.with_suffix(".csv").read_bytes(),
        "coverage":covered,"maximum_raw_anchor_sum_error_n":error,"fixed_stall_tail_contact_observations":forces,
        "active_drive_observations":drives,"measured_stall_progress_m":repeated_audit["metrics"]["observed_progress_m"],
        "neutral_wheel_joint_offsets_from_readback_NOT_measured_contact_trail":reviewed["neutral_wheel_joint_offsets"],
        "supported_conclusion":"With hold released and positive velocity targets maintained, active/caster SDK tangential-force projections oppose each other and nearly balance during the repeated stall.",
        "unique_initial_caster_friction_solver_or_motor_cause_isolated":False,
        "aggregate_force_balance_does_NOT_prove_global_H3_infeasibility":True,
        "net_Ft_over_Fz_is_NOT_a_local_anchor_friction_limit_test":True,
        "projected_solver_joint_effort_is_NOT_an_isolated_motor_torque_or_saturation_measurement":True,
        "proposed_next_intervention_requires_explicit_separate_condition": "Change only pre-motion caster initial orientation, preserve original FAIL and do NOT count as success in the original condition.",
        "input_sha256":inputs,"source_sha256":{Path(__file__).name:SUPPORT["sha"](__file__)},
        "new_PhysX_trial_performed":False,"controller_or_initial_state_or_material_changed":False,
        "full_dynamics_plan_certified":False,"full_H3B_turning_path_verified":False,"use_for_final_statistics":False}
