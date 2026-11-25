"""Passive rolling-stall observation and SDK friction-buffer validation.

No controller, gain, material, gate or pose modifications. Reported contact
vectors retain the SDK world-frame/sign convention; projected generalized
joint forces are never labelled isolated actuator torque.
"""

import json
import math
from pathlib import Path

import numpy as np

from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources
from hospital_h3b_rolling_failure import DEFAULT, analyze
from hospital_h3b_alignment_audit import telemetry_snapshot
from hospital_h3_instrumentation import vector


READBACK = GATE/"hospital_h3b_rolling_readiness/paused_readbacks/hospital_h3b_rolling_readiness_r01_paused_readback_r01.json"
FRICTION_PERIOD_S = 1./30.


def review_readback(path=READBACK):
    path=Path(path).resolve()
    record=json.loads(path.read_bytes())
    failure=analyze()
    verify_sources(record)
    runtime=record["runtime_drive_configuration"]
    if (record.get("status")!="PAUSED_CONFIGURATION_RECORDED_CAUSE_NOT_YET_ISOLATED"
            or record.get("stage_root_and_session_unchanged") is not True
            or record.get("physics_or_control_settings_modified") is not False
            or record.get("new_PhysX_trial_performed") is not False
            or record.get("historical_motor_torque_measured") is not False
            or record.get("selected_flush_threshold_stays_inactive") is not True
            or runtime.get("available") is not True
            or runtime.get("matches_recorded_stall_pose_and_joint_angles") is not True
            or Path(record["source_trial_result"]).resolve()!=DEFAULT.resolve()
            or record["source_trial_sha256"]!=SUPPORT["sha"](DEFAULT)):
        raise ValueError("Need the actual matched, read-only, post-cleanup rolling configuration report")
    joints=record["joint_USD_physics_attributes"]
    conversion={}
    offsets={}
    passive={}
    for path_,attributes in joints.items():
        name=Path(path_).name
        checks=[]
        for field,tensor in (("stiffness","stiffness_tensor_units"),("damping","damping_tensor_units")):
            authored=float(attributes["drive:angular:physics:"+field])
            actual=runtime[tensor][name]
            checks.append(math.isclose(actual,authored*180./math.pi,rel_tol=2e-6,abs_tol=2e-6))
        conversion[name]=all(checks)
        if "caster" in name:
            passive[name]={"stiffness_tensor_units":runtime["stiffness_tensor_units"][name],
                "damping_tensor_units":runtime["damping_tensor_units"][name],
                "joint_friction_coefficient":runtime["joint_friction_coefficients"][name]}
        if name.endswith("_caster_to_wheel"):
            offsets[name]={"wheel_joint_center_offset_in_caster_frame_m":attributes["physics:localPos0"],
                "offset_is_NOT_a_measured_contact_trail_or_unique_cause":True}
    if not all(conversion.values()):
        raise ValueError("Authored angular drive gains and runtime gains do not match the expected unit conversion")
    return {"status":"ACTUAL_PAUSED_CONFIGURATION_REVIEWED_UNIT_CONVERSION_MATCHES_UNIQUE_STALL_CAUSE_UNISOLATED",
        "input_sha256":{**failure["input_sha256"],str(path):SUPPORT["sha"](path)},
        "runtime_to_authored_angular_gain_factor":180./math.pi,
        "angular_gain_conversion_checks":conversion,"passive_joint_settings":passive,
        "neutral_wheel_joint_offsets":offsets,
        "effective_material_physics_attributes":record["effective_material_physics_attributes"],
        "post_cleanup_configuration_is_NOT_historical_stall_motor_torque":True,
        "zero_joint_friction_does_NOT_mean_zero_floor_contact_friction":True,
        "forward_joint_center_offset_is_a_hypothesis_NOT_a_proven_unique_cause":True,
        "full_H3B_turning_path_verified":False,"new_PhysX_trial_performed":False,"use_for_final_statistics":False}


def _array(data):
    if hasattr(data,"detach"):data=data.detach().cpu().numpy()
    elif hasattr(data,"numpy"):data=data.numpy()
    return np.asarray(data)


def friction_snapshot(view,time_s,physics_dt,sensor_paths):
    """Read native SDK impulses/physics_dt, not observer-period-scaled forces.

Only initialized/used buffer entries are inspected: uninitialized capacity
slots must not be treated as zero forces or as nonfinite telemetry failures.
"""
    if (not math.isfinite(time_s) or not math.isfinite(physics_dt) or not 0.<physics_dt<=.025
            or view.sensor_count!=len(sensor_paths) or view.filter_count!=1):
        raise ValueError("Invalid contact observer time, sensor count or filter count")
    n=len(sensor_paths)
    normal=np.asarray(vector(view.get_contact_force_matrix(physics_dt),n*3)).reshape(n,3)
    force,points,counts,starts=view.get_friction_data(physics_dt)
    force,points=_array(force),_array(points)
    counts,starts=_array(counts),_array(starts)
    capacity=int(view.max_contact_data_count)
    if (capacity<=0 or force.shape!=(capacity,3) or points.shape!=(capacity,3)
            or counts.shape!=(n,1) or starts.shape!=(n,1)
            or not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts!=np.floor(counts))
            or int(sum(counts[:,0]))>=capacity):
        raise ValueError("Invalid or possibly truncated native friction buffers")
    row={"time_s":float(time_s),"physics_impulse_conversion_dt_s":float(physics_dt)}
    anchors=[]
    occupied=set()
    for i,path in enumerate(sensor_paths):
        count=int(counts[i,0])
        row[f"wheel_{i}__friction_anchor_count"]=count
        total=np.zeros(3)
        if count:
            raw_start=starts[i,0]
            if not math.isfinite(float(raw_start)) or raw_start!=math.floor(raw_start):
                raise ValueError("Noninteger native friction buffer start")
            begin=int(raw_start)
            if begin<0 or begin+count>capacity or occupied.intersection(range(begin,begin+count)):
                raise ValueError("Invalid or overlapping native friction ranges")
            occupied.update(range(begin,begin+count))
            f,p=force[begin:begin+count],points[begin:begin+count]
            if not np.isfinite(f).all() or not np.isfinite(p).all():
                raise ValueError("Nonfinite used native friction anchor data")
            total=f.sum(axis=0)
            for index,(fv,pv) in enumerate(zip(f,p)):
                anchors.append({"time_s":float(time_s),"sensor_path":str(path),"filter_index":0,"anchor_index_within_pair":index,
                    **{f"sdk_tangent_f{axis}_n":float(value) for axis,value in zip("xyz",fv)},
                    **{f"point_{axis}_m":float(value) for axis,value in zip("xyz",pv)}})
        for axis,value in zip("xyz",normal[i]):row[f"wheel_{i}__sdk_reported_contact_f{axis}_n"]=float(value)
        for axis,value in zip("xyz",total):row[f"wheel_{i}__sdk_tangent_sum_f{axis}_n"]=float(value)
    return row,anchors


def coverage(joint_rows,friction_rows,anchor_rows,raw_contacts,primary,errors):
    end=primary.get("last_recorded_state",{}).get("time_s")
    joint_gap=max((b["time_s"]-a["time_s"] for a,b in zip(joint_rows,joint_rows[1:])),default=None)
    friction_gap=max((b["time_s"]-a["time_s"] for a,b in zip(friction_rows,friction_rows[1:])),default=None)
    ordered_joint=len(joint_rows)>1 and all(b["time_s"]>a["time_s"] for a,b in zip(joint_rows,joint_rows[1:]))
    ordered_friction=len(friction_rows)>1 and all(b["time_s"]>a["time_s"] for a,b in zip(friction_rows,friction_rows[1:]))
    start=bool(joint_rows and friction_rows and joint_rows[0]["time_s"]<=2./120.+1e-7 and friction_rows[0]["time_s"]<=2./120.+1e-7)
    finish=bool(end is not None and joint_rows and friction_rows and joint_rows[-1]["time_s"]>=end-2./120.
        and friction_rows[-1]["time_s"]>=end-.075)
    anchor_coverage=[any(row.get(f"wheel_{i}__friction_anchor_count",0)>0 for row in friction_rows) for i in range(6)]
    contact_complete=bool(raw_contacts) and all(row["complete_point_data"] for row in raw_contacts)
    complete=bool(not errors and ordered_joint and ordered_friction and joint_gap<=2./120.+1e-6
        and friction_gap<=.075 and start and finish and all(anchor_coverage) and anchor_rows and contact_complete
        and primary.get("original_motor_gains_restored") and primary.get("verdict") in ("PASS","FAIL"))
    return {"status":"RECORDED_FOR_CAUSAL_REVIEW_NOT_A_ROLLING_OR_TRAVERSAL_PASS" if complete else "INCOMPLETE_INSTRUMENTATION_DO_NOT_INFER_CAUSE",
        "joint_sample_count":len(joint_rows),"friction_sample_count":len(friction_rows),"friction_anchor_count":len(anchor_rows),
        "raw_contact_point_count":len(raw_contacts),"maximum_joint_sample_gap_s":joint_gap,"maximum_friction_sample_gap_s":friction_gap,
        "telemetry_covers_controller_start":start,"telemetry_covers_controller_end":finish,
        "each_wheel_has_friction_anchor_observations":anchor_coverage,"raw_contact_points_complete":contact_complete}
