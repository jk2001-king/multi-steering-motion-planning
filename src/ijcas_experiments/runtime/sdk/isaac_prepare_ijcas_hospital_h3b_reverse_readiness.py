"""Fresh STOPPED source -> frozen flush geometry/support for H3-B readiness.

Preparation ONLY. The old static gate must actually PASS next. This short
outside-door component is not H3-B mission execution or a dynamics plan.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _prepare_h3b_readiness():
    import json
    from pathlib import Path
    import runpy

    import omni.timeline
    import omni.usd
    from pxr import Usd

    if globals().pop("ijcas_hospital_h3b_prepare_condition",None)!="H3B_common_straight_reversal":
        raise RuntimeError("Explicitly select H3B_common_straight_reversal preparation")
    workspace=_ijcas_workspace()
    scripts=_ijcas_path(workspace, 'studies/IJCAS/scripts');entries=_ijcas_path(workspace, 'isaac/scripts')
    stage=omni.usd.get_context().get_stage();timeline=omni.timeline.get_timeline_interface()
    if stage is None or not timeline.is_stopped():raise RuntimeError("Fresh original hospital source must be STOPPED")
    before=stage.GetSessionLayer().ExportToString()
    for name,task in list(globals().items()):
        if name.startswith("ijcas_hospital_") and name.endswith("_task") and hasattr(task,"done") and not task.done():
            raise RuntimeError("A hospital trial is active")
    h=runpy.run_path(str(scripts/"hospital_h3_flush_support.py"))
    output=_ijcas_path(workspace, 'studies/IJCAS/results/hospital_paper_gate/hospital_h3b_readiness_preparation')
    output.mkdir(parents=True,exist_ok=True);number=1
    while (output/f"prepare_r{number:02d}.json").exists() or (output/f"prepare_r{number:02d}_failure.json").exists():number+=1
    result_path=output/f"prepare_r{number:02d}.json"
    try:
        globals()["ijcas_hospital_h3_environment_condition"]="H3_flush"
        path=entries/"isaac_prepare_ijcas_hospital_h3_flush_geometry.py"
        exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),globals())
        # The frozen geometry entry sets the explicit fresh setup directory.
        path=entries/"isaac_prepare_ijcas_hospital_h3_flush_static.py"
        exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),globals())
        marker=stage.GetPrimAtPath(h["COMPONENT"])
        if not marker.IsValid() or marker.GetCustomDataByKey("status")!="physics_prepared_static_pending":
            raise RuntimeError("Frozen flush support preparation did not complete")
        prep_path=Path(marker.GetCustomDataByKey("physics_preparation_result"))
        prep=json.loads(prep_path.read_bytes())
        task_path=_ijcas_path(workspace, 'studies/IJCAS/results/hospital_paper_gate/hospital_h3_turn_setup/setup_r01/hospital_h3_turn_geometry.json')
        task=json.loads(task_path.read_bytes());manifest=json.loads(Path(prep["geometry_manifest"]).read_bytes())
        if prep["start_xyyaw"]!=task["start_xyyaw"] or manifest["strict_map_digest"]!=task["planning_occupancy_digest"]:
            raise RuntimeError("Readiness environment/start differs from the unchanged H3 task")
        report={"status":"PREPARED_STATIC_AND_SHORT_REVERSAL_READINESS_PENDING", "role":"H3B_SHORT_STRAIGHT_REVERSAL_ONLY",
            "physics_preparation":str(prep_path),"physics_preparation_sha256":h["sha"](prep_path),
            "turning_task_geometry_reference":str(task_path),"turning_task_geometry_sha256":h["sha"](task_path),
            "mission_start_xyyaw_unchanged":task["start_xyyaw"],"mission_goal_xyyaw_unchanged":task["goal_xyyaw"],
            "this_component_is_not_a_mission_goal_redefinition":True,"PhysX_executed":False,"static_support_verified":False,
            "reverse_readiness_verified":False,"full_H3B_turning_path_verified":False,"source_USD_saved":False,
            "source_sha256":{"isaac_prepare_ijcas_hospital_h3b_reverse_readiness.py":h["sha"](entries/"isaac_prepare_ijcas_hospital_h3b_reverse_readiness.py"),
                "hospital_h3b_policy.py":h["sha"](scripts/"hospital_h3b_policy.py")}}
        with result_path.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            marker.SetCustomDataByKey("H3B_readiness_preparation",str(result_path))
            marker.SetCustomDataByKey("H3B_readiness_preparation_sha256",h["sha"](result_path))
    except Exception as error:
        stage.GetSessionLayer().ImportFromString(before)
        failure=result_path.with_name(result_path.stem+"_failure.json")
        with failure.open("x",encoding="utf-8") as stream:json.dump({"status":"FAILED_SESSION_ROLLED_BACK_NO_PLAY","error":str(error)},stream,indent=2)
        raise
    print("=== H3-B SHORT REVERSAL READINESS PREPARED ===")
    print("saved:",result_path)
    print("Timeline STOPPED. Next: isaac_run_ijcas_hospital_h3_flush_static_gate.py. No forward/reverse command yet.")
    print("After actual static PASS, keep PAUSED for the H3-B short reversal readiness script. Do not save the source USD.")


_prepare_h3b_readiness()
