#!/usr/bin/env python3
"""Repropagate actual H3 task actions/footprints; NOT a PhysX/Nav2 gate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_hospital_h1_plan import audit as audit_actions
from hospital_h3_turn_planning import load_setup,model_bounds,SUPPORT


def audit(path):
    path=Path(path).resolve();report=json.loads(path.read_bytes())
    if report.get("scenario")!="H3_doorway_immediate_turn" or report.get("environment_condition")!=SUPPORT["CONDITION"]:
        raise ValueError("An actual H3_flush turning plan is required, not relabelled H1/H2 actions")
    manifest,occupied,origin,resolution,manifest_sha=load_setup(report["geometry_manifest"],report["preview_receipt"])
    if report["geometry_manifest_sha256"]!=manifest_sha:
        raise ValueError("H3 geometry changed after planning")
    if report["preview_receipt_sha256"]!=SUPPORT["sha"](report["preview_receipt"]):
        raise ValueError("H3 preview receipt changed after planning")
    if (report["start_world_xyyaw"]!=manifest["start_xyyaw"] or report["goal_world_xyyaw"]!=manifest["goal_xyyaw"]
            or report["goal_local_xyyaw"]!=manifest["local_goal_xyyaw"]):
        raise ValueError("H3 plan frame/start/goal mismatch")
    if (report["full_envelope_clearance_m"]!=.01 or report["common_max_command_speed_m_s"]!=.05
            or report["common_max_acceleration_m_s2"]!=.02 or report["planning_terminal_speed_tolerance_m_s"]!=.005
            or report["rigid_body_parameter_bounds"]!=model_bounds()):
        raise ValueError("H3 planning conditions differ from the common H1/H2 model/limits")
    with np.load(path.parent/"planning_occupancy.npz") as grid:
        if (not np.array_equal(grid["occupied"],occupied) or not np.array_equal(grid["origin_xy_m"],origin)
                or float(grid["resolution_m"])!=resolution
                or hashlib.sha256(grid["occupied"].tobytes()).hexdigest()!=report["planning_occupancy_digest"]):
            raise ValueError("H3 planning grid differs from its task raster")
    result=audit_actions(path)
    result.update(scenario=manifest["scenario"],environment_condition=manifest["environment_condition"],
        geometry_manifest_sha256=manifest_sha,trial_kind="independent_H3_TURNING_TASK_OFFLINE_repropagation_NOT_PhysX_execution",
        same_ward_as_H1_H2=True,CICS_input_used=False,use_for_final_statistics=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("plan",type=Path);args=parser.parse_args()
    result=audit(args.plan)
    folder=args.plan.resolve().parent/"offline_audits";folder.mkdir(exist_ok=True);number=1
    while (folder/f"audit_r{number:02d}.json").exists():number+=1
    target=folder/f"audit_r{number:02d}.json"
    with target.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print("=== H3 TURNING TASK INDEPENDENT OFFLINE AUDIT ===")
    print("method:",result["method"]);print("colliding samples:",result["colliding_samples"])
    print("failed checks:",[k for k,v in result["checks"].items() if not v])
    print("VERDICT:",result["verdict"]);print("saved:",target)
    print("No H3 PhysX, sensor/Nav2 or final comparison success implied.")
    raise SystemExit(0 if result["verdict"]=="PASS" else 2)


if __name__=="__main__":main()
