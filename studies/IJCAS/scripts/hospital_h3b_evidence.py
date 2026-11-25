"""Read measured H3-B reversal evidence; never retune a model from validation."""

import csv
import json
import math
from pathlib import Path
from statistics import median

from hospital_h3_turn_planning import GATE,SUPPORT,verify_sources
from hospital_h3b_readiness import evaluate_readiness
from hospital_static_gate import evaluate_static


DEFAULT_REVERSE=GATE/"hospital_h3b_reverse_readiness/hospital_h3b_reverse_readiness_r01.json"


def read_trace(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:
        result=[]
        for row in csv.DictReader(stream):
            item={}
            for key,value in row.items():
                if key=="phase":item[key]=value
                elif key=="position_hold_enabled":
                    if value not in ("True","False"):raise ValueError("Invalid recorded hold state")
                    item[key]=value=="True"
                else:item[key]=float(value)
            result.append(item)
        return result


def verify_reverse(path=DEFAULT_REVERSE):
    path=Path(path).resolve();report=json.loads(path.read_bytes());verify_sources(report)
    if (report.get("scenario")!="H3B_common_straight_reversal_readiness" or report.get("verdict")!="PASS"
            or not report.get("H3B_reverse_readiness_verified") or not report.get("original_motor_gains_restored")
            or report.get("nonzero_steering_alignment_physically_verified") is not False
            or not report.get("checks") or not all(report["checks"].values())):
        raise ValueError("Need actual successful short STRAIGHT H3-B reversal evidence")
    inputs={str(path):SUPPORT["sha"](path),str(path.with_suffix(".csv")):SUPPORT["sha"](path.with_suffix(".csv"))}
    for key,digest_key in (("static_reference","static_reference_sha256"),("H3B_preparation_reference","H3B_preparation_sha256"),
            ("shared_hold_component_reference","shared_hold_component_sha256")):
        source=Path(report[key]);digest=SUPPORT["sha"](source)
        if digest!=report[digest_key]:raise ValueError("Changed readiness input: "+str(source))
        inputs[str(source)]=digest
    static_path=Path(report["static_reference"]);static=json.loads(static_path.read_bytes());verify_sources(static)
    if SUPPORT["sha"](static_path.with_suffix(".csv"))!=report["static_trace_sha256"]:raise ValueError("Changed static CSV")
    inputs[str(static_path.with_suffix(".csv"))]=report["static_trace_sha256"]
    static_rows=read_trace(static_path.with_suffix(".csv"));rows=read_trace(path.with_suffix(".csv"))
    if evaluate_static(static_rows,static["start_xyyaw"],static["simulation_mass_kg"],static["contact_report_counts_by_category"],static["contact_callback_errors"])["verdict"]!="PASS":
        raise ValueError("Raw static data no longer reproduces PASS")
    result=evaluate_readiness(rows,report["target_world_xyyaw"],static_rows,report["physics_preparation_snapshot"]["simulation_mass_kg"],
        report["doorway_reference_bound_m"],report["wheel_conservative_sphere_radii_m"],report["contact_report_counts_by_category"],
        report["contact_callback_errors"],report["fixed_hold_attempt_count"],report["direction_gate_events"])
    if result["verdict"]!="PASS":raise ValueError("Raw reversal data no longer reproduces PASS")
    gate_checks=[]
    for i,event in enumerate(report["direction_gate_events"],1):
        group=[r for r in rows if r["phase"].startswith(f"cusp_{i}_")]
        align=[r for r in group if r["phase"].endswith("_alignment")]
        release=[r for r in group if r["phase"].endswith("_release_wait")]
        ready=[r for r in group if r["phase"].endswith("_ready")]
        okay=bool(align and release and ready) and all(abs(r["command_speed_m_s"])<1e-7 for r in group)
        okay=okay and all(r["position_hold_enabled"] for r in align) and all(not r["position_hold_enabled"] for r in release+ready)
        if align and release and ready:
            okay=okay and release[0]["time_s"]-align[0]["time_s"]>=.25-1e-6
            okay=okay and abs(ready[-1]["time_s"]-event["time_s"])<1e-6
            okay=okay and all(max(r["pose_filtered_speed_m_s"],math.hypot(r["vx_physx_m_s"],r["vy_physx_m_s"]))<=.005 for r in align+ready)
        gate_checks.append(bool(okay))
    if len(gate_checks)!=2 or not all(gate_checks):raise ValueError("Raw direction-gate acknowledgement/dwell does not reproduce")
    observed=[];c,s=math.cos(report["start_world_xyyaw"][2]),math.sin(report["start_world_xyyaw"][2])
    for i in range(3):
        stable=[r for r in rows if r["phase"]==f"leg_{i}_motion" and abs(abs(r["command_speed_m_s"])-.01)<1e-8]
        if stable:
            command=median(r["command_speed_m_s"] for r in stable)
            body=median(c*r["vx_physx_m_s"]+s*r["vy_physx_m_s"] for r in stable)
            observed.append({"leg_index":i,"sample_count":len(stable),"median_signed_command_m_s":command,
                "median_signed_body_projection_m_s":body,"body_to_command_ratio":body/command})
    return {"status":"MEASURED_STRAIGHT_REVERSAL_PASS_REPRODUCED_NONZERO_ALIGNMENT_PENDING","verdict":"PASS",
        "input_sha256":inputs,"raw_source_hashes_match":True,"raw_direction_gates_reproduced":gate_checks,
        "sample_count":len(rows),"terminal_position_error_m":result["terminal_position_error_m"],
        "terminal_heading_error_deg":result["terminal_heading_error_deg"],"hold_regression_speed_m_s":result["fixed_terminal_hold_analysis"]["pose_regression_speed_m_s"],
        "signed_speed_observations":observed,"speed_tracking_accuracy_established":False,"model_parameters_changed":False,
        "nonzero_steering_alignment_physically_verified":False,"full_H3B_turning_path_verified":False,
        "full_dynamics_plan_certified":False,"Nav2_executed":False,"use_for_final_statistics":False}
