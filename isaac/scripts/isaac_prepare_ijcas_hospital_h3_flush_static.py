"""H3_flush support preparation ONLY, in Session Layer on STOPPED geometry.

Use unchanged H1/H2 robot/contact/motor settings and one canonical flat support.
No Play, initial motion, controller hold gains, planner or Nav2. The selected
sill stays visually/physically inactive; every other obstacle stays intact.
"""


def _prepare_h3_flush_static():
    import hashlib
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    workspace = Path("/home/jk/Research/IJCAS_ws")
    scripts = workspace / "studies/IJCAS/scripts"
    helper_path = scripts / "hospital_h3_flush_support.py"
    h = runpy.run_path(str(helper_path))
    for name, task in list(globals().items()):
        if name.startswith("ijcas_hospital_") and name.endswith("_task") and hasattr(task,"done") and not task.done():
            raise RuntimeError("A hospital task is running; flush preparation refused")
    directory = globals().pop("ijcas_hospital_h3_flush_setup_directory",None)
    if not isinstance(directory,str) or not directory:
        raise RuntimeError("Explicitly set ijcas_hospital_h3_flush_setup_directory to the recorded flush setup")
    manifest_path, manifest, geometry, audit = h["load_setup"](directory,workspace)
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    if not timeline.is_stopped() or stage is None:
        raise RuntimeError("Keep the actual flush geometry STOPPED; do not prepare on a paused/used robot")
    if stage.GetPrimAtPath(h["COMPONENT"]).IsValid() or stage.GetPrimAtPath("/World/IJCAS_HospitalH3TerrainComponent").IsValid():
        raise RuntimeError("An old component is present; fresh source and flush geometry are required")
    env = stage.GetPrimAtPath(h["ENVIRONMENT"])
    if not env.IsValid() or Path(env.GetCustomDataByKey("setup_manifest") or "").resolve() != manifest_path.resolve():
        raise RuntimeError("Live flush setup does not match the explicitly selected manifest")
    h["live_geometry_check"](stage,manifest,geometry,audit)
    geo = runpy.run_path(str(scripts / "hospital_geometry_tools.py"))
    with np.load(manifest["occupancy_grid"]) as grid:
        occupied = grid["occupied"].copy()
        origin, resolution = grid["origin_xy_m"].copy(), float(grid["resolution_m"])
    if (hashlib.sha256(occupied.tobytes()).hexdigest() != manifest["strict_map_digest"]
            or geo["pose_collision"](occupied,manifest["start_xyyaw"],origin,resolution,.010)):
        raise RuntimeError("Flush staging is not clear on the unchanged strict map")
    rules_path = scripts / "hospital_contact_rules.py"
    floor_paths = runpy.run_path(str(rules_path))["exported_floor_paths"](audit)
    if not floor_paths or any(p.startswith(h["THRESHOLD"]+"/") for p in floor_paths):
        raise RuntimeError("Expected only audited flat duplicate floors")
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise RuntimeError("Expected one PhysicsScene")
    graphs = [p for p in stage.Traverse() if p.GetTypeName() == "OmniGraph"]
    robot_usd = workspace / "isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd"
    if h["sha"](robot_usd) != "ecca5698c07f70a39d242054573a1c3df7454ceff2df3c44af4ff821fa3eb74a":
        raise RuntimeError("The measured H1/H2 robot reference changed")
    root, ground_root, ground_path, material_path = h["ROBOT"],h["GROUND_ROOT"],h["GROUND"],h["MATERIAL"]
    output_root = workspace / "studies/IJCAS/results/hospital_paper_gate/hospital_h3_flush_static_preparation"
    output_root.mkdir(parents=True,exist_ok=True)
    number = 1
    while (output_root / f"hospital_h3_flush_physics_r{number:02d}.json").exists():
        number += 1
    output = output_root / f"hospital_h3_flush_physics_r{number:02d}.json"
    session = stage.GetSessionLayer()
    before = session.ExportToString()
    start = manifest["start_xyyaw"]
    try:
        with Usd.EditContext(stage,session):
            old = stage.GetPrimAtPath("/World/patient_transport")
            if old.IsValid():
                old.SetActive(False)
            robot = stage.GetPrimAtPath(root)
            if not robot.IsValid():
                robot = stage.DefinePrim(root,"Xform")
                robot.GetReferences().AddReference(str(robot_usd),root)
            base = stage.GetPrimAtPath(root+"/base_link")
            if not base.IsValid():
                raise RuntimeError("Missing IJCAS base_link")
            xf = UsdGeom.Xformable(robot)
            xf.ClearXformOpOrder()
            translation = xf.AddTranslateOp(opSuffix="hospitalH3Flush")
            rotation = xf.AddRotateZOp(opSuffix="hospitalH3Flush")
            translation.Set(Gf.Vec3d(0,0,0))
            rotation.Set(0.)
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(base)
            position = matrix.ExtractTranslation()
            forward = matrix.TransformDir(Gf.Vec3d(1,0,0))
            turn = start[2]-math.atan2(forward[1],forward[0])
            c, s = math.cos(turn),math.sin(turn)
            rotation.Set(math.degrees(turn))
            translation.Set(Gf.Vec3d(start[0]-(c*position[0]-s*position[1]),start[1]-(s*position[0]+c*position[1]),0.))
            disabled_graphs = []
            for prim in graphs:
                disabled_graphs.append(str(prim.GetPath()))
                prim.SetActive(False)
            disabled_floors, deinstanced = [],[]
            for path in floor_paths:
                for prefix in Sdf.Path(path).GetPrefixes():
                    ancestor = stage.GetPrimAtPath(prefix)
                    if ancestor.IsValid() and ancestor.IsInstance():
                        deinstanced.append(str(prefix))
                        ancestor.SetInstanceable(False)
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() or prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError("Flat floor instance override failed: " + path)
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
                disabled_floors.append(path)
            legacy = "/World/Hospital/GroundPlane/CollisionPlane"
            prim = stage.GetPrimAtPath(legacy)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
                disabled_floors.append(legacy)
            ground = stage.GetPrimAtPath(ground_root)
            if not ground.IsValid():
                ground = stage.DefinePrim(ground_root,"Xform")
                ground.GetReferences().AddReference(str(robot_usd),ground_root)
            xf = UsdGeom.Xformable(ground)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp(opSuffix="hospitalH3Flush").Set(Gf.Vec3d(start[0],start[1],.001))
            support = stage.GetPrimAtPath(ground_path)
            if not support.IsValid():
                raise RuntimeError("Missing canonical support floor")
            UsdPhysics.CollisionAPI.Apply(support).CreateCollisionEnabledAttr(True).Set(True)
            UsdGeom.Imageable(support).MakeInvisible()
            material = UsdShade.Material(stage.GetPrimAtPath(material_path))
            if not material:
                raise RuntimeError("Missing canonical support material")
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(.8).Set(.8)
            api.CreateDynamicFrictionAttr(.6).Set(.6)
            api.CreateRestitutionAttr(0.).Set(0.)
            physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            physx.CreateCompliantContactAccelerationSpringAttr(False).Set(False)
            physx.CreateCompliantContactStiffnessAttr(100000.).Set(100000.)
            physx.CreateCompliantContactDampingAttr(1500.).Set(1500.)
            physx.CreateFrictionCombineModeAttr("max").Set("max")
            for name in h["WHEELS"]:
                body = stage.GetPrimAtPath(root+"/"+name)
                colliders = [p for p in Usd.PrimRange(body,Usd.TraverseInstanceProxies())
                    if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
                if len(colliders) != 1 or colliders[0].GetTypeName() != "Capsule":
                    raise RuntimeError("Unexpected wheel collider: " + name)
                UsdShade.MaterialBindingAPI.Apply(body).Bind(material,UsdShade.Tokens.strongerThanDescendants,"physics")
            UsdShade.MaterialBindingAPI.Apply(support).Bind(material,UsdShade.Tokens.strongerThanDescendants,"physics")
            mass = UsdPhysics.MassAPI.Apply(base)
            old_mass, inertia = mass.GetMassAttr().Get(),mass.GetDiagonalInertiaAttr().Get()
            if old_mass is None or old_mass <= 0 or inertia is None:
                raise RuntimeError("Missing base mass/inertia")
            mass.CreateMassAttr(245.).Set(245.)
            mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v)*245./float(old_mass) for v in inertia)))
            mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.,0.,.270))
            for prim in Usd.PrimRange(robot):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False).Set(False)
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.).Set(0.)
            for name,(kp,kd,effort) in h["JOINT_SETTINGS"].items():
                prim = stage.GetPrimAtPath(root+"/joints/"+name)
                if not prim.IsA(UsdPhysics.RevoluteJoint):
                    raise RuntimeError("Missing active joint: " + name)
                drive = UsdPhysics.DriveAPI.Get(prim,"angular")
                if not drive:
                    raise RuntimeError("Missing angular drive: " + name)
                drive.CreateTypeAttr("force").Set("force")
                drive.CreateStiffnessAttr(kp).Set(kp)
                drive.CreateDampingAttr(kd).Set(kd)
                drive.CreateMaxForceAttr(effort).Set(effort)
                drive.CreateTargetPositionAttr(0.).Set(0.)
                drive.CreateTargetVelocityAttr(0.).Set(0.)
            scene = UsdPhysics.Scene(scenes[0])
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,0,-1))
            scene.CreateGravityMagnitudeAttr(9.81).Set(9.81)
            PhysxSchema.PhysxSceneAPI.Apply(scenes[0]).CreateTimeStepsPerSecondAttr(120).Set(120)
            h["live_geometry_check"](stage,manifest,geometry,audit,set(disabled_floors))
            for path in disabled_floors:
                if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get() is not False:
                    raise RuntimeError("A duplicate floor remains enabled")
            readback = h["support_readback"](stage,manifest)
            report = {"schema_version":1,"scenario":manifest["scenario"],"environment_condition":h["CONDITION"],
                "purpose":"H3_FLUSH_COMPONENT_STATIONARY_SUPPORT_ONLY_NOT_TURNING_TASK",
                "status":"FLUSH_PHYSICS_PREPARED_STATIC_PENDING","geometry_manifest":str(manifest_path),
                "geometry_manifest_sha256":h["sha"](manifest_path),"flush_geometry_report":manifest["flush_geometry_report"],
                "flush_geometry_sha256":manifest["flush_geometry_sha256"],"source_stage":manifest["source_stage"],
                "source_stage_sha256":geometry["source_stage_sha256"],"source_audit":manifest["source_audit"],
                "robot_reference":str(robot_usd),"robot_reference_sha256":h["sha"](robot_usd),
                "robot_base":root+"/base_link","start_xyyaw":start,"component_end_xyyaw":manifest["component_end_xyyaw"],
                "support_floor":ground_path,"support_floor_top_z_m":.001,"live_support_readback":readback,
                "simulation_mass_kg":readback["simulation_mass_kg"],"physics_hz":120,
                "disabled_legacy_floor_colliders":sorted(set(disabled_floors)),"audited_floor_count":len(floor_paths),
                "deinstanced_floor_roots":sorted(set(deinstanced)),"disabled_action_graphs":disabled_graphs,
                "selected_threshold_visual_and_collision_inactive":True,"other_hospital_obstacles_unchanged":True,
                "static_support_verified":False,"whole_component_PhysX_support_verified":False,
                "component_crossing_verified":False,"H3_task_planning_executed":False,"Nav2_executed":False,
                "caster_state_reset":False,"CICS_input_used":False,"source_USD_saved":False,"use_for_final_statistics":False,
                "source_sha256":{p.name:h["sha"](p) for p in (
                    workspace/"isaac/scripts/isaac_prepare_ijcas_hospital_h3_flush_static.py",helper_path,
                    rules_path,scripts/"hospital_geometry_tools.py")}}
            with output.open("x",encoding="utf-8") as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
                stream.write("\n")
            marker = UsdGeom.Xform.Define(stage,h["COMPONENT"]).GetPrim()
            marker.SetCustomDataByKey("condition",h["CONDITION"])
            marker.SetCustomDataByKey("status","physics_prepared_static_pending")
            marker.SetCustomDataByKey("setup_manifest",str(manifest_path))
            marker.SetCustomDataByKey("setup_manifest_sha256",report["geometry_manifest_sha256"])
            marker.SetCustomDataByKey("physics_preparation_result",str(output))
            marker.SetCustomDataByKey("physics_preparation_sha256",h["sha"](output))
    except Exception as error:
        session.ImportFromString(before)
        failure = output.with_name(output.stem+"_preparation_failure.json")
        with failure.open("x",encoding="utf-8") as stream:
            json.dump({"status":"FAILED_SESSION_ROLLED_BACK_NO_PLAY","error":str(error)},stream,indent=2)
        raise
    omni.usd.get_context().get_selection().set_selected_prim_paths([root,"/World/Hospital/Geo_M_DoorFrame43"],True)
    print("=== H3 FLUSH COMPONENT STATIC PREPARATION ===")
    print("static staging xy/yaw:",start,"(outside door; NOT the H3 turning task start)")
    print("mass/bodies:",readback["simulation_mass_kg"],"kg /",len(readback["rigid_bodies"]))
    print("same H1/H2 120 Hz, 0.8/0.6 friction and original velocity-mode motor settings")
    print("single flat support geometrically covers WHOLE component:",readback["support_coverage"]["authored_single_solid_cube_covers_whole_component"])
    print("selected threshold visual/collision INACTIVE; other hospital obstacles retained")
    print("duplicate floor colliders disabled:",len(set(disabled_floors)))
    print("saved:",output)
    print("Timeline STOPPED. Next: isaac_run_ijcas_hospital_h3_flush_static_gate.py ONLY.")
    print("Static support/whole passage/turning plan/Nav2 are not validated. Do not save the source USD.")


_prepare_h3_flush_static()
