"""Prepare the explicit H3_flush VISUAL/COLLISION variant and strict raster.

Fresh STOPPED hospital source only. Reuse the frozen fully-open door procedure,
then deactivate ONE selected threshold asset (visual AND collision) in Session
Layer. No floor/motor/material/robot/physics change or Play. Support continuity,
static contact and crossing must be checked in later gates before any driving.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _prepare_h3_flush_geometry():
    import hashlib
    import json
    from pathlib import Path
    import runpy
    import subprocess

    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics

    workspace=_ijcas_workspace()
    scripts=_ijcas_path(workspace, 'studies/IJCAS/scripts');isaac=_ijcas_path(workspace, 'isaac/scripts')
    gate=_ijcas_path(workspace, 'studies/IJCAS/results/hospital_paper_gate')
    if globals().pop("ijcas_hospital_h3_environment_condition",None)!="H3_flush":
        raise RuntimeError("Explicitly select ijcas_hospital_h3_environment_condition='H3_flush'")
    for name,task in list(globals().items()):
        if name.startswith("ijcas_hospital_") and name.endswith("_task") and hasattr(task,"done") and not task.done():
            raise RuntimeError("A hospital task is running; environment change refused: "+name)
    timeline=omni.timeline.get_timeline_interface();context=omni.usd.get_context();stage=context.get_stage()
    if not timeline.is_stopped() or stage is None:
        raise RuntimeError("STOP and reopen the unchanged hospital source without saving the old Session Layer")
    if Path(stage.GetRootLayer().realPath).resolve()!=_ijcas_path(workspace, 'isaac/scenes/hospital_experiment.usda'):
        raise RuntimeError("Open the original hospital_experiment.usda")
    for path in ("/World/IJCAS_HospitalH3DoorState","/World/IJCAS_HospitalH3TerrainComponent",
            "/World/IJCAS_HospitalH3FlushEnvironment","/World/IJCAS_HospitalH3FlushComponent"):
        if stage.GetPrimAtPath(path).IsValid():raise RuntimeError("An old H3 Session Layer is present; reopen source without saving")
    h=runpy.run_path(str(scripts/"hospital_h3_flush_geometry.py"))
    reference_path=gate/"hospital_h3_terrain_setup/setup_r01/terrain_component_setup.json"
    reference=json.loads(reference_path.read_bytes())
    original_door_path=Path(reference["fully_open_geometry_report"])
    original_door=json.loads(original_door_path.read_bytes())
    for name,digest in original_door["source_sha256"].items():
        folder=isaac if name.startswith("isaac_") else scripts
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Measured fully-open procedure changed: "+name)
    preserved={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (gate/"hospital_h3_terrain_execution").glob("hospital_h3_threshold_crossing_r*.json")}
    if len(preserved)<2:raise RuntimeError("Preserve the original threshold trial records before creating a new variant")
    output_root=gate/"hospital_h3_flush_geometry";output_root.mkdir(parents=True,exist_ok=True);i=1
    while (output_root/f"flush_r{i:02d}").exists():i+=1
    output=output_root/f"flush_r{i:02d}";output.mkdir()
    session=stage.GetSessionLayer();before=session.ExportToString()
    root_file=Path(stage.GetRootLayer().realPath);root_digest=hashlib.sha256(root_file.read_bytes()).hexdigest()
    report_path=output/"flush_geometry_report.json";audit_path=output/"hospital_asset_geometry_audit_flush.json"
    try:
        globals()["ijcas_hospital_h3_door_condition"]="fully_open_baseline"
        p=isaac/"isaac_prepare_ijcas_hospital_h3_fully_open.py"
        exec(compile(p.read_text(encoding="utf-8"),str(p),"exec"),globals())
        fully_open_path=Path(globals()["ijcas_hospital_h3_fully_open_directory"])/"doorway_geometry_audit.json"
        full_raw=fully_open_path.read_bytes();full=json.loads(full_raw)
        full_audit_path=Path(full["geometry_source_audit"]);full_audit=json.loads(full_audit_path.read_bytes())
        if not full.get("checks") or not all(full["checks"].values()):raise RuntimeError("Fully-open geometry checks did not pass")
        threshold=stage.GetPrimAtPath(h["THRESHOLD"])
        if not threshold.IsValid() or threshold.IsInstanceProxy() or not threshold.IsActive():
            raise RuntimeError("Expected the active selected threshold ROOT, not an instance-proxy child")
        if any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(threshold,Usd.TraverseInstanceProxies())):
            raise RuntimeError("A dynamic threshold needs a separate procedure")
        flags={}
        for a in full_audit["local_assets"]:
            for path in a["collision"]["collision_prim_paths"]:
                prim=stage.GetPrimAtPath(path)
                if not prim.IsValid() or not prim.IsActive():raise RuntimeError("Missing active source collider: "+path)
                flags[path]=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        with Usd.EditContext(stage,session):
            threshold.SetActive(False)
            # Hide only stale annotations which still depict the ORIGINAL sill.
            overlay=stage.GetPrimAtPath("/World/IJCAS_HospitalH3GeometryOverlay")
            if overlay.IsValid():overlay.SetActive(False)
        if threshold.IsActive():raise RuntimeError("Selected threshold visual/collision deactivation failed")
        bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
        for a in full_audit["local_assets"]:
            if a["path"]==h["THRESHOLD"]:continue
            prim=stage.GetPrimAtPath(a["path"])
            if not prim.IsValid() or not prim.IsActive():raise RuntimeError("Another hospital asset was deactivated: "+a["path"])
            bound=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            if any(not np.allclose(actual,a["world_aligned_visible_bound"][key],rtol=0,atol=2e-5)
                    for key,actual in (("minimum_m",tuple(bound.GetMin())),("maximum_m",tuple(bound.GetMax())))):
                raise RuntimeError("Another hospital asset moved: "+a["path"])
            if str(UsdGeom.Imageable(prim).ComputeVisibility())!=a["computed_visibility"]:
                raise RuntimeError("Another hospital asset visibility changed: "+a["path"])
            for path in a["collision"]["collision_prim_paths"]:
                prim=stage.GetPrimAtPath(path)
                if not prim.IsActive() or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()!=flags[path]:
                    raise RuntimeError("Another hospital collider changed: "+path)
        if hashlib.sha256(root_file.read_bytes()).hexdigest()!=root_digest:raise RuntimeError("Source USD changed")
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in preserved.items()):
            raise RuntimeError("An original threshold result changed")
        variant=h["variant_inventory"](full_audit,live_threshold_inactive=True,other_assets_verified=True)
        with audit_path.open("x",encoding="utf-8") as stream:json.dump(variant,stream,indent=2,allow_nan=False);stream.write("\n")
        checks={"only_selected_threshold_visual_and_collision_inactive":not threshold.IsActive(),
            "other_assets_bounds_visibility_and_collider_flags_unchanged":True,"fully_open_door_geometry_verified":True,
            "original_threshold_trial_files_preserved":True,"source_USD_file_unchanged":True}
        report={"schema_version":1,"condition":h["CONDITION"],"scenario":"H3_doorway_immediate_turn",
            "status":"FLUSH_GEOMETRY_RECORDED_SUPPORT_STATIC_AND_DRIVING_PENDING",
            "source_stage":str(root_file),"source_stage_sha256":root_digest,"live_geometry_verified":True,"checks":checks,
            "fully_open_geometry_report":str(fully_open_path),"fully_open_geometry_sha256":hashlib.sha256(full_raw).hexdigest(),
            "active_geometry_audit":str(audit_path),"suppressed_threshold_root":h["THRESHOLD"],
            "support_floor_top_z_m":h["SUPPORT_TOP_M"],"continuous_PhysX_support_prepared":False,
            "original_threshold_results_preserved":preserved,"source_USD_saved":False,"robot_moved":False,
            "physics_material_or_motor_settings_changed":False,"PhysX_executed":False,"static_support_verified":False,
            "component_crossing_verified":False,"H3_task_planning_verified":False,"Nav2_verified":False,
            "use_for_final_statistics":False,"floor_scope":"FLUSH_CONDITION_INTENDED_CANONICAL_SUPPORT_CONTINUITY_MUST_BE_VERIFIED_BEFORE_PLAY",
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (isaac/"isaac_prepare_ijcas_hospital_h3_flush_geometry.py",scripts/"hospital_h3_flush_geometry.py")},
            "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (fully_open_path,full_audit_path,audit_path,reference_path,root_file)}}
        with report_path.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
        built=subprocess.run(["/usr/bin/python3",str(scripts/"build_hospital_h3_flush_setup.py"),"--geometry-directory",str(output)],
            capture_output=True,text=True,check=False,timeout=120,cwd=str(workspace))
        print(built.stdout,end="")
        if built.returncode:print(built.stderr);raise RuntimeError("Flush strict-map preflight failed; no driving allowed")
        saved=[line.removeprefix("saved: ") for line in built.stdout.splitlines() if line.startswith("saved: ")]
        if len(saved)!=1:raise RuntimeError("Expected exactly one fresh flush setup manifest")
        manifest_path=Path(saved[0]).resolve()
        if not manifest_path.is_relative_to(gate/"hospital_h3_flush_setup"):raise RuntimeError("Unexpected setup path")
        with Usd.EditContext(stage,session):
            marker=UsdGeom.Xform.Define(stage,h["MARKER"]).GetPrim()
            marker.SetCustomDataByKey("condition",h["CONDITION"])
            marker.SetCustomDataByKey("geometry_report",str(report_path))
            marker.SetCustomDataByKey("geometry_report_sha256",hashlib.sha256(report_path.read_bytes()).hexdigest())
            marker.SetCustomDataByKey("setup_manifest",str(manifest_path))
            marker.SetCustomDataByKey("setup_manifest_sha256",hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            marker.SetCustomDataByKey("status","flush_geometry_ready_support_static_pending")
        globals()["ijcas_hospital_h3_flush_setup_directory"]=str(manifest_path.parent)
        globals()["ijcas_hospital_h3_flush_geometry_directory"]=str(output)
    except Exception as error:
        session.ImportFromString(before)
        with (output/"preparation_failure.json").open("x",encoding="utf-8") as stream:
            json.dump({"status":"FAILED_SESSION_ROLLED_BACK_NO_PHYSICS_OR_DRIVING","error":str(error)},stream,indent=2)
        raise
    context.get_selection().set_selected_prim_paths(["/World/Hospital/Geo_M1_DoorWall11",
        "/World/Hospital/Geo_M_DoorFrame43","/World/Hospital/SM_Door_02b9",
        "/World/Hospital/SM_HospitalBed_02d11","/World/Hospital/SM_HospitalBed_02d12"],True)
    print("=== H3 FLUSH GEOMETRY + STRICT MAP PREPARED ===")
    print("condition:",h["CONDITION"]);print("selected threshold visual AND collision inactive:",h["THRESHOLD"])
    print("wall/frame/bed/door width unchanged; original source and FAILs preserved")
    print("geometry report:",report_path);print("setup manifest:",manifest_path)
    print("Frame Selected (F), Top View; visually inspect the bedroom doorway floor.")
    print("Timeline STOPPED. Continuous support/static/crossing/H3 turning plan/Nav2 are NOT validated.")
    print("Do not press Play, save the source USD, or run the ORIGINAL threshold static/terrain scripts on this variant.")
    return str(output)


ijcas_hospital_h3_flush_geometry_directory=_prepare_h3_flush_geometry()
