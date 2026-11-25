#!/usr/bin/env python3
"""Four-method H3 task OFFLINE planning with unchanged H1/H2 algorithms.

Use the actual new turning-task map and its matched live preview receipt.
Keep every attempt; bounded search failure is NOT physical infeasibility.
No robot, stage, common terminal controller, method modes or model is changed.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
from audit_hospital_h3_plan import audit
from hospital_h3_turn_planning import load_setup,model_bounds,DEFAULT_SETUP,DEFAULT_PREVIEW,MODEL_PLAN,MODEL_SHA,GATE,SUPPORT
from plan_hospital_h1 import METHODS,SPEED,SPEED_LEVELS,DECELERATION,TERMINAL_SPEED,plan_method


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT_SETUP)
    parser.add_argument("--preview-receipt",type=Path,default=DEFAULT_PREVIEW)
    parser.add_argument("--budget-s",type=float,default=45.)
    parser.add_argument("--node-limit",type=int,default=20000)
    args=parser.parse_args()
    if not math.isfinite(args.budget_s) or args.budget_s<=0 or args.node_limit<=0:
        parser.error("Positive finite planning budgets are required")
    manifest,occupied,origin,resolution,manifest_sha=load_setup(args.manifest,args.preview_receipt)
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    folder=GATE/"planning";folder.mkdir(parents=True,exist_ok=True);number=1
    while (folder/f"hospital_h3_planning_r{number:02d}").exists():number+=1
    output=folder/f"hospital_h3_planning_r{number:02d}";output.mkdir();(output/"offline_audits").mkdir()
    np.savez_compressed(output/"planning_occupancy.npz",occupied=occupied,origin_xy_m=origin,resolution_m=resolution)
    here=Path(__file__).resolve().parent
    sources=("plan_hospital_h3.py","hospital_h3_turn_planning.py","audit_hospital_h3_plan.py",
        "audit_hospital_h1_plan.py","plan_hospital_h1.py","hospital_planning_tools.py","hospital_geometry_tools.py")
    metadata={"schema_version":1,"scenario":manifest["scenario"],"environment_condition":manifest["environment_condition"],
        "trial_kind":"offline_H3_doorway_turn_planning_development_NOT_PhysX_execution",
        "geometry_manifest":str(args.manifest.resolve()),"geometry_manifest_sha256":manifest_sha,
        "preview_receipt":str(args.preview_receipt.resolve()),"preview_receipt_sha256":SUPPORT["sha"](args.preview_receipt),
        "paper_geometry_frozen":False,"explicit_user_viewport_confirmation_recorded":False,
        "source_occupancy_sha256":SUPPORT["sha"](manifest["occupancy_grid"]),
        "planning_occupancy_digest":manifest["planning_occupancy_digest"],
        "start_world_xyyaw":manifest["start_xyyaw"],"goal_world_xyyaw":manifest["goal_xyyaw"],
        "goal_local_xyyaw":manifest["local_goal_xyyaw"],"common_max_command_speed_m_s":SPEED,
        "common_max_acceleration_m_s2":DECELERATION,"planning_terminal_speed_tolerance_m_s":TERMINAL_SPEED,
        "proposed_refinement_speed_levels_m_s":list(SPEED_LEVELS),"full_envelope_clearance_m":.010,
        "collision_method":"full_envelope_cell_supercover_at_all_propagation_substeps",
        "rigid_body_parameter_bounds":asdict(model.bounds),"model_reference_plan":str(MODEL_PLAN),"model_reference_plan_sha256":MODEL_SHA,
        "rigid_body_parameter_scope":"unchanged_H1_H2_provisional_simulation_bounds_NOT_new_hospital_or_physical_identification",
        "candidate_goal_and_environment_not_changed_to_favour_a_method":True,
        "same_ward_as_H1_H2":True,"source_stage_modified":False,"CICS_input_used":False,
        "source_sha256":{name:SUPPORT["sha"](here/name) for name in sources},
        "use_for_final_statistics":False,"Isaac_execution_verified":False,"Nav2_execution_verified":False}
    print("=== H3 FOUR-METHOD OFFLINE TURNING-TASK PLANNING ===",flush=True)
    print("output:",output,flush=True)
    print("unchanged H1/H2 algorithm/model/modes; per-method budget:",args.budget_s,"s",flush=True)
    rows=[];candidates={}
    for method in METHODS:
        print("planning:",method,flush=True)
        row=plan_method(method,manifest,occupied,origin,resolution,model,args.budget_s,args.node_limit,candidates)
        if row["plan_success"] and method!="P_dynamics_full_envelope":candidates[method]=row["selected_actions"]
        target=output/(method+".json")
        with target.open("x",encoding="utf-8") as stream:json.dump({**metadata,**row},stream,indent=2,allow_nan=False);stream.write("\n")
        result=None
        if row["plan_success"]:
            result=audit(target)
            with (output/"offline_audits"/(method+".json")).open("x",encoding="utf-8") as stream:
                json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
        short={key:row.get(key) for key in ("method","status","plan_success","planning_time_s","action_count",
            "path_length_m","mode_switch_count","selected_spatial_source_route_id")}
        short["independent_offline_audit"]=result["verdict"] if result else "NOT_RUN_NO_SUCCESSFUL_PLAN"
        rows.append(short)
        print(method,row["status"],f"{row['planning_time_s']:.2f}s","actions=",row.get("action_count"),
            "offline audit=",short["independent_offline_audit"],flush=True)
    complete=all(row["plan_success"] and row["independent_offline_audit"]=="PASS" for row in rows)
    summary={**metadata,"methods":rows,"status":"FOUR_OFFLINE_PLANS_AUDITED_EXECUTION_PENDING" if complete else
        "HOLD_OFFLINE_PLANNING_OR_AUDIT_NOT_COMPLETE","all_attempts_retained":True,
        "planning_timeout_is_not_physical_infeasibility":True,"proposed_is_hierarchical_not_global_joint_state_search":True}
    with (output/"summary.json").open("x",encoding="utf-8") as stream:json.dump(summary,stream,indent=2,allow_nan=False);stream.write("\n")
    print("status:",summary["status"],flush=True)
    print("No H3 turning PhysX, sensor/Nav2 or final statistical comparison has run.",flush=True)


if __name__=="__main__":main()
