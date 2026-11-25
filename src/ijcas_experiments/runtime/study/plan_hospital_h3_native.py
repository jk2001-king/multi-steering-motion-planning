#!/usr/bin/env python3
"""Four-method H3 task planning with native batching of the SAME exact SAT.

Use the actual new turning-task map and its matched live preview receipt.
Keep every attempt; bounded search failure is NOT physical infeasibility.
No robot, stage, common terminal controller, method modes or model is changed.
The original function bytecode, heuristic and search bounds are unchanged.
Only its private checker-class binding uses native exact finite-cell SAT.
Independent audits always repropagate with the original Python polygon SAT.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
from audit_hospital_h3_plan import audit
from hospital_h3_turn_planning import load_setup,model_bounds,DEFAULT_SETUP,DEFAULT_PREVIEW,MODEL_PLAN,MODEL_SHA,GATE,SUPPORT
from plan_hospital_h1 import METHODS,SPEED,SPEED_LEVELS,DECELERATION,TERMINAL_SPEED
from hospital_native_envelope import initialize_native,cloned_plan_method


def continuous_task_planner(manifest, occupied, origin, resolution,*,additional_execution_reserve_m=0.,initial_state=None):
    """Private shared mission bindings; original H3 task/model stay fixed.

    The old H1-derived invocation allowed forward travel only and used H1
    search bounds. Those are not the declared five-leg H3 task. Reuse the
    existing bidirectional/resource-aware mission search, without borrowing
    measured component endpoints or changing obstacles/terminal tolerances.
    """
    import types
    import hospital_mission_planning as binding
    import hospital_native_envelope as native
    import hospital_h3_navigation_guidance as navigation
    import plan_hospital_h1 as original
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability import hierarchical_dynamics_planner as hierarchy

    maximum=np.asarray(origin)+np.asarray(occupied.shape[::-1])*resolution
    roi=[float(origin[0]),float(maximum[0]),float(origin[1]),float(maximum[1])]
    bounds=binding.mission_search_bounds(roi,manifest['start_xyyaw'],manifest['goal_xyyaw'])
    limits=binding.MotionResourceLimits()
    key=binding.goal_predicate_separating_state_key(lattice._state_key,
        manifest['local_goal_xyyaw'],terminal_speed_m_s=.005)
    propagate=binding.compiled_original_NE_propagation()
    search,refine=binding.bind_resource_kernels(lattice.plan,
        hierarchy.refine_speed_consistent_lattice_route,lattice.State,propagate,key,limits)
    if initial_state is not None:
        complete=search.__globals__['State'](**initial_state)
        if complete.started is not True or complete.rigid_body_feasible is not True:
            raise ValueError('Complete feasible started measured continuation required')
        search=binding.continue_original_lattice_from_state(search,complete)
    search=binding.pareto_resource_lattice_function(search,generated_goal_exit=True)
    if (type(additional_execution_reserve_m) not in (int,float)
            or not math.isfinite(additional_execution_reserve_m) or not 0.<=additional_execution_reserve_m<=.100):
        raise ValueError('Explicit finite common additional planning reserve up to100mm required')
    base_checker=binding.sparse_cell_native_checker(native)
    if additional_execution_reserve_m:
        base_checker=binding.additional_planning_reserve_checker(base_checker,reserve_m=additional_execution_reserve_m)
    checker=binding.resource_checker_class(base_checker)
    field=navigation.NavigationGuidance(occupied,origin,resolution,
        manifest['start_xyyaw'],manifest['goal_xyyaw'])
    guidance=binding.mission_local_heading_guidance(field,original.pose_guidance,
        manifest['local_goal_xyyaw'],directions=(1,-1),speed_m_s=.05,
        maximum_curvature_1_m=.7,heading_weight=.4,terminal_zone_m=1.63,
        terminal_position_tolerance_m=.025)
    method=binding.bind_plan_method(original.plan_method,search,bounds,
        checker_class=checker,guidance=guidance,directions=(1,-1))
    namespace=dict(method.__globals__,refine_speed_consistent_lattice_route=refine)
    method=types.FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__)
    return method,dict(search_bounds=bounds,directions=[1,-1],
        maximum_reverse_episode_m=.5,maximum_total_reverse_m=1.,maximum_direction_changes=4,
        original_H3_start_goal_map_and_NE_model_unchanged=True,
        no_measured_component_endpoint_or_state_injection=True,
        shared_mission_search_and_compute_bindings=True,
        common_additional_planning_reserve_m=additional_execution_reserve_m,
        unchanged_physical_full_body_margin_m=.010,
        original_H1_forward_only_call_not_used=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT_SETUP)
    parser.add_argument("--preview-receipt",type=Path,default=DEFAULT_PREVIEW)
    parser.add_argument("--budget-s",type=float,default=45.)
    parser.add_argument("--node-limit",type=int,default=20000)
    parser.add_argument("--continuous-task",action="store_true",
        help="Use shared bidirectional/resource-aware search for the unchanged whole H3 task")
    parser.add_argument("--execution-reserve-m",type=float,default=0.,
        help="Additional common planning clearance; actual physical10mm acceptance stays unchanged")
    args=parser.parse_args()
    if not math.isfinite(args.budget_s) or args.budget_s<=0 or args.node_limit<=0:
        parser.error("Positive finite planning budgets are required")
    if args.execution_reserve_m and not args.continuous_task:parser.error('Planning reserve requires the explicit continuous-task binding')
    warmup_start=time.perf_counter()
    native_build=initialize_native()
    native_warmup_s=time.perf_counter()-warmup_start
    plan_method=cloned_plan_method()
    manifest,occupied,origin,resolution,manifest_sha=load_setup(args.manifest,args.preview_receipt)
    continuous_binding=None
    if args.continuous_task:
        plan_method,continuous_binding=continuous_task_planner(manifest,occupied,origin,resolution,
            additional_execution_reserve_m=args.execution_reserve_m)
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    folder=GATE/"planning";folder.mkdir(parents=True,exist_ok=True);number=1
    while (folder/f"hospital_h3_native_planning_r{number:02d}").exists():number+=1
    output=folder/f"hospital_h3_native_planning_r{number:02d}";output.mkdir();(output/"offline_audits").mkdir()
    np.savez_compressed(output/"planning_occupancy.npz",occupied=occupied,origin_xy_m=origin,resolution_m=resolution)
    here=Path(__file__).resolve().parent
    sources=("plan_hospital_h3_native.py","plan_hospital_h3.py","hospital_native_envelope.py","hospital_exact_sat.c","hospital_h3_turn_planning.py","audit_hospital_h3_plan.py",
        "audit_hospital_h1_plan.py","plan_hospital_h1.py","hospital_planning_tools.py","hospital_geometry_tools.py")
    if args.continuous_task:sources+=('hospital_mission_planning.py','hospital_h3_navigation_guidance.py')
    metadata={"schema_version":1,"scenario":manifest["scenario"],"environment_condition":manifest["environment_condition"],
        "trial_kind":"offline_H3_native_exact_SAT_planning_NOT_PhysX_execution",
        "collision_checker_implementation":"native_batched_exact_SAT_no_quantization_or_envelope_change",
        "native_build":native_build,"native_initialization_time_s_excluded_from_method_budgets":native_warmup_s,
        "original_plan_method_bytecode_and_heuristic_unchanged":True,
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
    if continuous_binding is not None:
        metadata['continuous_task_search_binding']=continuous_binding
        metadata['planning_additional_execution_reserve_m']=args.execution_reserve_m
        metadata['P_spatial_candidates_are_shared_development_inputs_NOT_primary_end_to_end_timing']=True
    print("=== H3 FOUR-METHOD NATIVE EXACT-SAT OFFLINE PLANNING ===",flush=True)
    print("output:",output,flush=True)
    print("same H1/H2 algorithm/model/modes/heuristic + native exact SAT; per-method budget:",args.budget_s,"s",flush=True)
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
