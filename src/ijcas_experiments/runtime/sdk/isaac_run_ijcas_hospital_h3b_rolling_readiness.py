"""Representative rolling steering -> straight stop/hold/reverse readiness.

Separate from both stationary alignment and the saved full H3-B trajectory.
Frozen straight-reversal adapter safety, contacts, gain cleanup and handshake
are retained. New S steering reference and exactly two short motion legs.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _schedule_h3b_rolling_readiness():
    import asyncio
    from collections import Counter,deque
    import csv
    import hashlib
    import json
    import math
    from pathlib import Path
    import runpy
    import time

    import numpy as np
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.api.sensors import RigidContactView
    from isaacsim.core.prims import RigidPrim,SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf,PhysicsSchemaTools,Usd,UsdGeom,UsdPhysics,UsdUtils

    if globals().pop("ijcas_hospital_h3b_rolling_condition",None)!="H3B_rolling_steering_straight_cusp":
        raise RuntimeError("Explicitly select H3B_rolling_steering_straight_cusp; this is a representative component, not saved-route replay")
    for name,task in list(globals().items()):
        if name.startswith("ijcas_hospital_") and name.endswith("_task") and hasattr(task,"done") and not task.done():
            raise RuntimeError("Another hospital task is running")
    workspace=_ijcas_workspace()
    scripts=_ijcas_path(workspace, 'studies/IJCAS/scripts');results=_ijcas_path(workspace, 'studies/IJCAS/results/hospital_paper_gate')
    support=runpy.run_path(str(scripts/"hospital_h3_flush_support.py"))
    loader=runpy.run_path(str(scripts/"hospital_private_loader.py"))
    h=loader["load"](scripts/"hospital_h3b_rolling_readiness.py")
    hold=runpy.run_path(str(scripts/"hospital_drive_hold.py"))["drive_hold_gain_arrays"]
    static_rules=runpy.run_path(str(scripts/"hospital_static_gate.py"))
    collision=runpy.run_path(str(scripts/"hospital_geometry_tools.py"))["pose_collision"]
    timeline=omni.timeline.get_timeline_interface()
    if timeline.is_stopped() or timeline.is_playing():
        raise RuntimeError("Leave the actual H3_flush static PASS PAUSED; do not press Stop/Play")
    stage=omni.usd.get_context().get_stage()
    marker=stage.GetPrimAtPath(support["COMPONENT"]) if stage else None
    if (marker is None or not marker.IsValid() or marker.GetCustomDataByKey("condition")!=support["CONDITION"]
            or marker.GetCustomDataByKey("status")!="static_support_passed"):
        raise RuntimeError("A fresh actual dedicated H3_flush static PASS is required")
    wrapper_path=Path(marker.GetCustomDataByKey("H3B_readiness_preparation") or "")
    wrapper=json.loads(wrapper_path.read_bytes())
    if (support["sha"](wrapper_path)!=marker.GetCustomDataByKey("H3B_readiness_preparation_sha256")
            or wrapper.get("role")!="H3B_SHORT_STRAIGHT_REVERSAL_ONLY"
            or wrapper_path.with_name(wrapper_path.stem+"_failure.json").exists()):
        raise RuntimeError("Run the dedicated fresh H3-B preparation wrapper before the static gate")
    support["verify_files"]({wrapper["turning_task_geometry_reference"]:wrapper["turning_task_geometry_sha256"]})
    for name,expected in wrapper["source_sha256"].items():
        folder=_ijcas_path(workspace, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
        if support["sha"](folder/name)!=expected:raise RuntimeError("H3-B preparation source changed: "+name)
    rolling_path=Path(marker.GetCustomDataByKey("H3B_rolling_preparation") or "")
    rolling=json.loads(rolling_path.read_bytes())
    if (support["sha"](rolling_path)!=marker.GetCustomDataByKey("H3B_rolling_preparation_sha256")
            or rolling.get("role")!=h["ROLE"]
            or rolling_path.with_name(rolling_path.stem+"_failure.json").exists()
            or rolling["base_preparation_reference"]!=str(wrapper_path)
            or rolling["base_preparation_sha256"]!=support["sha"](wrapper_path)):
        raise RuntimeError("Fresh dedicated ROLLING component preparation required")
    support["verify_files"](rolling["measured_reverse_readiness"]["input_sha256"])
    warm_path=Path(rolling["reviewed_warm_start_audit"])
    if support["sha"](warm_path)!=rolling["reviewed_warm_start_audit_sha256"]:
        raise RuntimeError("Reviewed geometry-only warm-start audit changed")
    warm=json.loads(warm_path.read_bytes())
    support["verify_files"](warm["input_sha256"])
    for record in (rolling,warm):
        for name,expected in record["source_sha256"].items():
            folder=_ijcas_path(workspace, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
            if support["sha"](folder/name)!=expected:raise RuntimeError("Rolling component source changed: "+name)
    static_path=Path(marker.GetCustomDataByKey("static_gate_result") or "")
    static_raw=static_path.read_bytes();static=json.loads(static_raw)
    if (support["sha"](static_path)!=marker.GetCustomDataByKey("static_gate_sha256")
            or static.get("verdict")!="PASS" or static.get("environment_condition")!=support["CONDITION"]
            or not static.get("checks") or not all(static["checks"].values())
            or static.get("component_crossing_verified") is not False):
        raise RuntimeError("Invalid H3_flush static evidence, or a previously used start")
    static_csv=static_path.with_suffix(".csv")
    with static_csv.open(newline="",encoding="utf-8") as stream:
        static_rows=[{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
    if static_rules["evaluate_static"](static_rows,static["start_xyyaw"],static["simulation_mass_kg"],
            static["contact_report_counts_by_category"],static["contact_callback_errors"])["verdict"]!="PASS":
        raise RuntimeError("The raw flush static CSV does not reproduce PASS")
    prep=static["physics_preparation_snapshot"];prep_path=Path(static["physics_preparation"])
    if (support["sha"](prep_path)!=static["physics_preparation_sha256"]
            or json.loads(prep_path.read_bytes())!=prep
            or marker.GetCustomDataByKey("physics_preparation_sha256")!=static["physics_preparation_sha256"]
            or prep.get("environment_condition")!=support["CONDITION"]):
        raise RuntimeError("Flush preparation lineage changed")
    if str(prep_path)!=wrapper["physics_preparation"] or support["sha"](prep_path)!=wrapper["physics_preparation_sha256"]:
        raise RuntimeError("H3-B readiness wrapper/static preparation mismatch")
    for snapshot in (static,prep):
        for name,expected in snapshot["source_sha256"].items():
            folder=_ijcas_path(workspace, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
            if support["sha"](folder/name)!=expected:
                raise RuntimeError("A measured static/preparation source changed: "+name)
    manifest_path,manifest,geometry,audit=support["load_setup"](Path(prep["geometry_manifest"]).parent,workspace)
    if (support["sha"](manifest_path)!=prep["geometry_manifest_sha256"]
            or marker.GetCustomDataByKey("setup_manifest_sha256")!=prep["geometry_manifest_sha256"]
            or Path(marker.GetCustomDataByKey("setup_manifest") or "").resolve()!=manifest_path.resolve()):
        raise RuntimeError("Flush geometry manifest differs from the static test")
    support["live_geometry_check"](stage,manifest,geometry,audit,set(prep["disabled_legacy_floor_colliders"]))
    if support["support_readback"](stage,manifest)!=prep["live_support_readback"]:
        raise RuntimeError("Live support/contact/mass/motor baseline changed")
    if support["sha"](prep["robot_reference"])!=prep["robot_reference_sha256"]:
        raise RuntimeError("The robot USD changed")
    if any(p.GetTypeName()=="OmniGraph" for p in stage.Traverse()):
        raise RuntimeError("Competing action graph is active")
    for path in prep["disabled_legacy_floor_colliders"]:
        if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get() is not False:
            raise RuntimeError("A duplicate support floor remains enabled")
    door_geometry=json.loads(Path(manifest["fully_open_geometry_report"]).read_bytes())
    leaf=stage.GetPrimAtPath(door_geometry["assets"]["leaf"]["path"])
    leaf_matrix=UsdGeom.XformCache().GetLocalToWorldTransform(leaf)
    specification=door_geometry["door_state_specification"]
    if (not np.allclose(tuple(leaf_matrix.ExtractTranslation()),specification["hinge_world_origin_m"],rtol=0,atol=2e-5)
            or not np.allclose([tuple(leaf_matrix.TransformDir(axis)) for axis in
                (Gf.Vec3d(1,0,0),Gf.Vec3d(0,1,0),Gf.Vec3d(0,0,1))],specification["desired_root_world_axes"],rtol=0,atol=2e-5)):
        raise RuntimeError("The fully-open door hinge/orientation changed")
    # The suppressed sill's former span is a REFERENCE only, not a collider authorization.
    doorway_reference_bound=audit["explicitly_suppressed_assets"][0]["collision"]["bound"]
    component_path=results/"motion_stop/hospital_h1_ramp_stop_hold_r01.json"
    component_raw=component_path.read_bytes();component=json.loads(component_raw)
    if component.get("verdict")!="PASS" or not component.get("original_motor_gains_restored"):
        raise RuntimeError("Shared ordinary ramp/active-wheel hold evidence missing")
    if support["sha"](scripts/"hospital_drive_hold.py")!=component["source_sha256"]["hospital_drive_hold.py"]:
        raise RuntimeError("The verified actuator-hold helper changed")
    with np.load(manifest["occupancy_grid"]) as grid:
        occupied=grid["occupied"].copy()
        map_origin=grid["origin_xy_m"].copy();resolution=float(grid["resolution_m"])
    if (hashlib.sha256(occupied.tobytes()).hexdigest()!=manifest["strict_map_digest"]
            or occupied.dtype!=np.dtype(bool) or occupied.shape!=tuple(manifest["raster"]["shape_rows_columns"])
            or not np.array_equal(map_origin,manifest["raster"]["origin_xy_m"]) or resolution!=manifest["raster"]["resolution_m"]):
        raise RuntimeError("The actual NEW STRICT flush raster/frame changed")
    root=h["ROBOT_ROOT"];base=root+"/base_link"
    steering=("base_link_to_front_active_steering","base_link_to_rear_active_steering")
    drives=("front_active_steering_to_drive_wheel","rear_active_steering_to_drive_wheel")
    start=tuple(manifest["start_xyyaw"])
    goal=h["component_goal"](start)
    distance=.220  # forward component safety bound, NOT a doorway/turning target
    if abs(start[2]-math.pi)>1e-9 or list(goal)!=rolling["component_nominal_goal_world_xyyaw"]:
        raise RuntimeError("Fixed ORIGINAL-start rolling reference mismatch")
    output=results/"hospital_h3b_rolling_readiness";output.mkdir(parents=True,exist_ok=True);number=1
    while any((output/f"hospital_h3b_rolling_readiness_r{number:02d}{ext}").exists() for ext in (".json",".csv")):number+=1
    run_id=f"hospital_h3b_rolling_readiness_r{number:02d}"
    summary_path=output/(run_id+".json");trace_path=output/(run_id+".csv")
    clock={"time_s":0.};counts=Counter();pairs=Counter();callback_errors=[]

    def physics_step(dt):clock["time_s"]+=float(dt)

    def contacts(headers,data):
        del data
        try:
            for header in headers:
                pair=tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
                category=h["classify_contact"](pair,prep["support_floor"],prep["disabled_legacy_floor_colliders"])
                if category!="unrelated":counts[category]+=1;pairs[(pair,category)]+=1
        except Exception as error:callback_errors.append(str(error))

    def angles(quaternion):
        w,x,y,z=map(float,quaternion)
        return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))

    async def run():
        report={"schema_version":1,"scenario":"H3B_representative_rolling_steering_straight_cusp","environment_condition":support["CONDITION"],
            "run_id":run_id,"verdict":"FAIL",
            "trial_kind":h["ROLE"],
            "H3B_policy":dict(h["POLICY"]),"fixed_leg_targets":[list(v) for v in h["LEGS"]],
            "nonzero_stationary_alignment_physically_verified":False,
            "nonzero_steering_alignment_physically_verified":False,
            "full_H3B_turning_path_verified":False,"full_dynamics_plan_certified":False,"PhysX_executed":False,
            "representative_component_only_not_saved_warm_start_replay":True,
            "stationary_alignment_original_FAILs_remain_authoritative":True,
            "rolling_preparation_reference":str(rolling_path),"rolling_preparation_sha256":support["sha"](rolling_path),
            "reviewed_geometry_only_warm_start_reference":str(warm_path),
            "reviewed_geometry_only_warm_start_sha256":support["sha"](warm_path),
            "fixed_progress_curvature_knots":h["KNOTS"],
            "no_controller_gain_contact_or_existing_stop_gate_relaxation":True,
            "frozen_adapter_reference_sha256":"d9bce554fd582521f546495d711f6b61f9c0e1d481da5eaa3e4fe39487426dfe",
            "geometry_manifest":str(manifest_path),"geometry_manifest_sha256":prep["geometry_manifest_sha256"],
            "static_reference":str(static_path),"static_reference_sha256":support["sha"](static_path),
            "static_trace_sha256":support["sha"](static_csv),"physics_preparation_snapshot":prep,
            "H3B_preparation_reference":str(wrapper_path),"H3B_preparation_sha256":support["sha"](wrapper_path),
            "shared_hold_component_reference":str(component_path),"shared_hold_component_sha256":support["sha"](component_path),
            "shared_hold_scope":"H1_actuator_hold_component_only_NOT_flush_passage_evidence",
            "start_world_xyyaw":start,"target_world_xyyaw":goal,"net_target_distance_m":.170,"longitudinal_safety_upper_m":distance+.030,
            "command_limits":h["COMMAND_LIMITS"],"readiness":h["READINESS"],"guards":h["GUARDS"],"acceptance":h["ACCEPTANCE"],
            "authorized_threshold_colliders":[],"strict_flush_map_digest":manifest["strict_map_digest"],
            "doorway_reference_bound_m":doorway_reference_bound,"former_sill_span_used_only_as_clearance_reference":True,
            "component_crossing_verified":False,"retained_threshold_traversal_verified":False,"H3_task_execution_verified":False,
            "path_planner_executed":False,"Nav2_executed":False,"sensor_localization_used":False,"CICS_input_used":False,
            "body_pose_reset_during_run":False,"caster_state_reset_during_run":False,"material_modified_during_run":False,
            "source_USD_saved":False,"use_for_final_statistics":False,"fixed_hold_attempt_count":0,
            "source_sha256":{name:support["sha"](path) for name,path in (
                ("isaac_run_ijcas_hospital_h3b_rolling_readiness.py",_ijcas_path(workspace, 'isaac/scripts/isaac_run_ijcas_hospital_h3b_rolling_readiness.py')),
                *[(name,scripts/name) for name in ("hospital_h3b_rolling_readiness.py","hospital_h3b_readiness.py","hospital_h3b_policy.py","hospital_private_loader.py",
                    "hospital_path_execution.py","hospital_path_execution_v2.py","hospital_path_execution_v3.py",
                    "hospital_h3_flush_crossing.py","hospital_h3_flush_support.py",
                    "hospital_h3_terrain.py","hospital_drive_hold.py","hospital_stop_analysis.py",
                    "hospital_contact_rules.py","hospital_geometry_tools.py")])}}
        robot=motor=drive_indices=old_kp=old_kd=old_limits=old_targets=None
        hold_targets=None;rows=[];contact_views=[];pose_views={};radii={};history=deque()
        previous_pose=None;previous_time=None;phase="pre_hold";last_watch=-float("inf")
        physics_sub=contact_sub=None
        app=omni.kit.app.get_app();wall_start=time.perf_counter();gate_events=[];held=False
        steer_command=(0.,0.);last_command_time=clock["time_s"];requested_curvature=0.;wheel_command=np.zeros(2)
        try:
            physics_sub=omni.physx.get_physx_interface().subscribe_physics_step_events(physics_step)
            contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
            robot=SingleArticulation(base);robot.initialize()
            indices={name:i for i,name in enumerate(robot.dof_names)}
            if any(name not in indices for name in (*steering,*drives)):raise RuntimeError("Missing active joints")
            steer_indices=np.asarray([indices[name] for name in steering],dtype=np.int32)
            drive_indices=np.asarray([indices[name] for name in drives],dtype=np.int32)
            motor=robot.get_articulation_controller();kp,kd=motor.get_gains()
            old_kp,old_kd=np.asarray(kp).copy(),np.asarray(kd).copy();old_limits=np.asarray(motor.get_max_efforts()).copy()
            old_targets=np.asarray(robot.get_applied_action().joint_positions)[drive_indices].copy()
            if np.max(np.abs(old_kp[drive_indices]))>1e-6 or not np.isfinite(old_targets).all():
                raise RuntimeError("Original velocity-only drive gains and finite position targets required")
            cache=UsdGeom.XformCache()
            for i,name in enumerate(h["WHEELS"]):
                body_path=root+"/"+name;prim=stage.GetPrimAtPath(body_path)
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise RuntimeError("Missing measured wheel body")
                view=RigidContactView(prim_paths_expr=body_path,filter_paths_expr=[],name=run_id+"_force_"+str(i),
                    prepare_contact_sensors=False,disable_stablization=False)
                view.initialize();contact_views.append(view)
                pose_view=RigidPrim(prim_paths_expr=body_path,name=run_id+"_pose_"+str(i),reset_xform_properties=False,
                    prepare_contact_sensors=False,disable_stablization=False)
                pose_view.initialize()
                if not pose_view.is_physics_handle_valid():raise RuntimeError("Wheel poses must come from live PhysX, not USD fallback")
                pose_views[name]=pose_view
                colliders=[p for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
                if len(colliders)!=1 or not colliders[0].IsA(UsdGeom.Capsule):raise RuntimeError("Expected one capsule per wheel")
                shape=UsdGeom.Capsule(colliders[0]);local=cache.GetLocalToWorldTransform(colliders[0])*cache.GetLocalToWorldTransform(prim).GetInverse()
                scales=[local.TransformDir(axis).GetLength() for axis in (Gf.Vec3d(1,0,0),Gf.Vec3d(0,1,0),Gf.Vec3d(0,0,1))]
                radii[name]=(float(shape.GetRadiusAttr().Get())+float(shape.GetHeightAttr().Get())/2)*max(scales)+local.ExtractTranslation().GetLength()
            report["wheel_conservative_sphere_radii_m"]=radii
            sim=omni.physx.get_physx_simulation_interface()
            stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt();encoded=PhysicsSchemaTools.sdfPathToInt(base)

            def verify_gains(expected):
                kp,kd=motor.get_gains()
                if not (np.allclose(kp,expected,rtol=1e-6,atol=1e-6) and np.array_equal(kd,old_kd)
                        and np.array_equal(np.asarray(motor.get_max_efforts()),old_limits)):
                    raise RuntimeError("Motor/passive-joint gain or torque-limit guard")

            def set_hold(enabled):
                nonlocal hold_targets,held
                if enabled:
                    actual=np.asarray(robot.get_joint_positions(),dtype=float)[steer_indices]
                    body=np.asarray(robot.get_linear_velocity(),dtype=float)
                    if (max(abs(value) for value in steer_command)>1e-7
                            or np.max(np.abs(actual))>math.radians(2.5)
                            or math.hypot(body[0],body[1])>.005):
                        raise RuntimeError("Actual neutral steering and existing 5 mm/s stop required BEFORE drive hold; no held steering fallback")
                    hold_targets=np.asarray(robot.get_joint_positions(),dtype=float)[drive_indices].copy()
                    if not np.isfinite(hold_targets).all():raise RuntimeError("Non-finite captured wheel angles")
                    robot.apply_action(ArticulationAction(joint_positions=hold_targets.astype(np.float32),
                        joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
                    new_kp,new_kd=hold(old_kp,old_kd,drive_indices)
                else:new_kp,new_kd=old_kp.copy(),old_kd.copy()
                with Usd.EditContext(stage,stage.GetSessionLayer()):motor.set_gains(kps=new_kp,kds=new_kd,save_to_usd=False)
                verify_gains(new_kp)
                held=bool(enabled)

            def command(speed,held):
                nonlocal steer_command,last_command_time,requested_curvature,wheel_command
                if not math.isfinite(speed) or abs(speed)>.01000001 or (held and abs(speed)>1e-7):
                    raise RuntimeError("Rolling component signed-speed/hold command guard")
                now=clock["time_s"];dt=now-last_command_time
                position,quaternion=robot.get_world_pose()
                yaw=angles(quaternion)[2]
                progress=h["relative_pose"]((float(position[0]),float(position[1]),yaw),start)[0]
                requested_curvature=h["curvature_at"](progress,phase)
                actual=np.asarray(robot.get_joint_positions(),dtype=float)[steer_indices]
                steer_command,wheel_command=h["direct_joint_targets"](speed,"coordinated_counterphase",
                    requested_curvature,steer_command,dt,actual)
                if held and max(abs(value) for value in steer_command)>1e-7:
                    raise RuntimeError("Nonzero steering command during captured hold is NOT authorized")
                last_command_time=now
                robot.apply_action(ArticulationAction(joint_positions=np.asarray(steer_command,dtype=np.float32),joint_indices=steer_indices))
                robot.apply_action(ArticulationAction(joint_positions=hold_targets.astype(np.float32) if held else None,
                    joint_velocities=np.asarray(wheel_command,dtype=np.float32),joint_indices=drive_indices))

            def measure():
                position,quaternion=robot.get_world_pose();q=np.asarray(robot.get_joint_positions(),dtype=float)
                qd=np.asarray(robot.get_joint_velocities(),dtype=float);linear=np.asarray(robot.get_linear_velocity(),dtype=float)
                if not all(np.isfinite(value).all() for value in (position,quaternion,q,qd,linear)):raise RuntimeError("Non-finite body/joint state")
                roll,pitch,yaw=angles(quaternion)
                return (float(position[0]),float(position[1]),yaw),position,q,qd,linear,roll,pitch

            def environmental_guard():
                live=UsdGeom.XformCache().GetLocalToWorldTransform(leaf)
                if not np.allclose(np.asarray(live),np.asarray(leaf_matrix),rtol=0,atol=2e-5):
                    raise RuntimeError("Fully-open door pose changed during test")
                threshold=stage.GetPrimAtPath(support["THRESHOLD"])
                if not threshold.IsValid() or threshold.IsActive():
                    raise RuntimeError("The selected sill became active in the flush condition")
                if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(prep["support_floor"])).GetCollisionEnabledAttr().Get() is False:
                    raise RuntimeError("Canonical support floor became disabled")
                for path in prep["disabled_legacy_floor_colliders"]:
                    if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get() is not False:
                        raise RuntimeError("Duplicate floor became enabled")

            async def sample(speed,held,phase_start):
                nonlocal previous_pose,previous_time,last_watch
                command(speed,held)
                while True:
                    sim.wake_up(stage_id,encoded);await app.next_update_async()
                    if not timeline.is_playing():raise RuntimeError("Timeline controls changed during flush passage")
                    if time.perf_counter()-wall_start>h["GUARDS"]["maximum_wall_time_s"]:raise TimeoutError("Flush passage wall-time guard")
                    now=clock["time_s"]
                    if previous_time is None or now>previous_time:break
                if now>h["GUARDS"]["maximum_simulation_duration_s"] or (previous_time is not None and now-previous_time>.25):
                    raise TimeoutError("Flush passage simulation-duration/control-gap guard")
                pose,position,q,qd,linear,roll,pitch=measure();local=h["relative_pose"](pose,start)
                history.append((now,pose))
                while len(history)>2 and history[1][0]<=now-.20:history.popleft()
                dt=now-history[0][0]
                filtered=math.hypot(pose[0]-history[0][1][0],pose[1]-history[0][1][1])/dt if dt>0 else math.hypot(linear[0],linear[1])
                row={"time_s":now,"phase":phase,"phase_time_s":now-phase_start,"x_m":pose[0],"y_m":pose[1],"z_m":float(position[2]),
                    "yaw_rad":pose[2],"roll_deg":math.degrees(roll),"pitch_deg":math.degrees(pitch),
                    "longitudinal_m":local[0],"lateral_m":local[1],"yaw_error_deg":math.degrees(local[2]),
                    "vx_physx_m_s":float(linear[0]),"vy_physx_m_s":float(linear[1]),"vz_physx_m_s":float(linear[2]),
                    "pose_filtered_speed_m_s":filtered,"command_speed_m_s":speed,"position_hold_enabled":held,
                    "front_rad":float(q[steer_indices[0]]),"rear_rad":float(q[steer_indices[1]]),
                    "command_front_rad":float(steer_command[0]),"command_rear_rad":float(steer_command[1]),
                    "requested_curvature_1_m":float(requested_curvature),"command_time_s":float(last_command_time)}
                action=robot.get_applied_action();targets=np.asarray(action.joint_velocities)[drive_indices]
                positions=np.asarray(action.joint_positions)
                for i,axle in enumerate(("front","rear")):
                    row["drive_"+axle+"_rad"]=float(q[drive_indices[i]])
                    row["drive_"+axle+"_velocity_rad_s"]=float(qd[drive_indices[i]])
                    row["drive_"+axle+"_actual_velocity_target_rad_s"]=float(targets[i])
                for name,index in indices.items():row["joint_"+name+"_rad"]=float(q[index])
                for i,name in enumerate(h["WHEELS"]):
                    view=pose_views[name]
                    if not view.is_physics_handle_valid():raise RuntimeError("Live wheel pose handle was lost")
                    wheel_positions,_=view.get_world_poses();wp=np.asarray(wheel_positions,dtype=float).reshape(-1,3)[0]
                    force=float(np.asarray(contact_views[i].get_net_contact_forces(dt=1./120.),dtype=float).reshape(-1,3)[0,2])
                    if not np.isfinite(wp).all() or not math.isfinite(force):raise RuntimeError("Non-finite wheel pose/load")
                    for axis,value in zip(("x","y","z"),wp):row["wheel_"+name+"_"+axis+"_m"]=float(value)
                    row[f"wheel_{i}_fz_n"]=force
                # Keep the offending measured sample before evaluating safety;
                # a failed sample is NOT a completed-hold terminal certificate.
                rows.append(row)
                try:
                    verify_gains(hold(old_kp,old_kd,drive_indices)[0] if held else old_kp)
                    environmental_guard()
                    if not np.allclose(targets,wheel_command,rtol=0,atol=1e-6) or not np.allclose(positions[steer_indices],steer_command,rtol=0,atol=1e-6):
                        raise RuntimeError("Applied motor targets differ from the recorded command")
                    if held and not np.allclose(positions[drive_indices],hold_targets,rtol=0,atol=1e-6):raise RuntimeError("Hold target changed")
                    if callback_errors or any(counts[key] for key in h["BAD_CONTACTS"]):raise RuntimeError("Unexpected hospital obstacle/nonwheel/legacy contact")
                    if max(abs(roll),abs(pitch))>math.radians(2.) or math.hypot(linear[0],linear[1])>.080:
                        raise RuntimeError("Flush passage body tilt/speed safety guard")
                    if abs(local[1])>.030 or abs(local[2])>math.radians(2.) or not -.020<=local[0]<=distance+.030:
                        raise RuntimeError("Flush passage reference corridor/heading/progress safety guard")
                    last=previous_pose if previous_pose is not None else pose
                    turn=math.atan2(math.sin(pose[2]-last[2]),math.cos(pose[2]-last[2]))
                    n=max(1,math.ceil(math.hypot(pose[0]-last[0],pose[1]-last[1])/.005),math.ceil(abs(turn)/math.radians(.25)))
                    for j in range(1,n+1):
                        a=j/n;p=(last[0]+a*(pose[0]-last[0]),last[1]+a*(pose[1]-last[1]),last[2]+a*turn)
                        if collision(occupied,p,map_origin,resolution,.010):raise RuntimeError("Actual swept full-envelope STRICT flush raster guard")
                    if h["motion_stalled"](rows):raise RuntimeError("Five-second measured forward-progress stall")
                except Exception as error:
                    report["safety_abort"]={"reason":str(error),"actual_state":dict(row)}
                    raise
                previous_pose,previous_time=pose,now
                if time.perf_counter()-last_watch>=15.:
                    print(f"[H3-B ROLLING {run_id}] wall={time.perf_counter()-wall_start:.1f}s sim={now:.1f}s phase={phase}; "
                        f"progress={local[0]:.3f}/{distance:.3f}m speed command/observed={1000*speed:.2f}/{1000*filtered:.2f}mm/s; "
                        f"obstacle callbacks={counts['hospital_obstacle']}")
                    last_watch=time.perf_counter()
                return row,pose,linear

            pose,position,q,_,linear,_,_=measure();offset=h["relative_pose"](pose,start)
            static_end=static_rows[-1]
            if math.hypot(pose[0]-static_end["x_m"],pose[1]-static_end["y_m"])>h["READINESS"]["paused_static_pose_difference_m"]:
                raise RuntimeError("Robot moved after the paused static result; fresh preparation required")
            if math.hypot(offset[0],offset[1])>.010 or abs(offset[2])>math.radians(.5) or np.max(np.abs(q[steer_indices]))>math.radians(1.):
                raise RuntimeError("Actual H3_flush release pose/steering differs from the fixed component start")
            c,s=math.cos(start[2]),math.sin(start[2])
            for x,y,yaw in h["nominal_component"]():
                if collision(occupied,(start[0]+c*x-s*y,start[1]+s*x+c*y,start[2]+yaw),map_origin,resolution,.010):
                    raise RuntimeError("Representative rolling full-envelope component reference is not clear")
            report["observed_prehold_world_xyyaw"]=list(pose)
            set_hold(True);command(0.,True);timeline.play();report["PhysX_executed"]=True;pre_start=clock["time_s"]
            print("H3-B rolling phase: 2 s measured-wheel pre-hold; no forward command yet")
            while clock["time_s"]-pre_start<2.:row,pose,linear=await sample(0.,True,pre_start)
            offset=h["relative_pose"](pose,start)
            # Static acceptance remains 10 mm. Permit at most 3 mm MORE during
            # actuator engagement; fixed absolute start/end are never shifted.
            prepose=report["observed_prehold_world_xyyaw"]
            prehold_rows=[r for r in rows if r["phase"]=="pre_hold"]
            if (math.hypot(offset[0],offset[1])>h["READINESS"]["prehold_position_from_original_start_m"]
                    or any(math.hypot(r["x_m"]-prepose[0],r["y_m"]-prepose[1])>h["READINESS"]["prehold_displacement_m"] for r in prehold_rows)
                    or abs(offset[2])>math.radians(.5)
                    or max(row["pose_filtered_speed_m_s"],math.hypot(linear[0],linear[1]))>.005):
                raise RuntimeError("H3_flush pre-hold release-state readiness failed; no forward command issued")
            report["release_world_xyyaw"]=list(pose)
            initial_crossing=h["wheel_crossing_checks"](rows,doorway_reference_bound,radii)
            if not all(item["started_fully_east"] for item in initial_crossing.values()):raise RuntimeError("All wheels must start east of the doorway reference span")
            set_hold(False);motion_start=clock["time_s"]
            for leg_index,(direction,target_progress) in enumerate(h["LEGS"]):
                if leg_index:
                    gate=h["StopAlignGate"](direction,"coordinated_counterphase",0.,clock["time_s"])
                    gate_start=clock["time_s"];last_stage=None
                    while True:
                        measured=max(row["pose_filtered_speed_m_s"],math.hypot(linear[0],linear[1]))
                        info=gate.step(clock["time_s"],0.,measured,(row["front_rad"],row["rear_rad"]),held)
                        if info["drive_hold_requested"]!=held:set_hold(info["drive_hold_requested"])
                        phase=f"cusp_{leg_index}_{info['stage']}"
                        if info["stage"]!=last_stage:print("H3-B direction gate:",leg_index,info["stage"])
                        last_stage=info["stage"]
                        # Sample/verify actual actuator state after each requested
                        # change; a requested release is NOT an acknowledgement.
                        row,pose,linear=await sample(0.,held,gate_start)
                        if info["ready"]:
                            event={"direction":direction,"time_s":clock["time_s"],
                                "command_speed_m_s":0.,"measured_speed_m_s":max(row["pose_filtered_speed_m_s"],math.hypot(linear[0],linear[1])),
                                "captured_hold_acknowledged":gate.saw_hold,"hold_release_acknowledged":not held,
                                "steering_error_deg":math.degrees(max(abs(row["front_rad"]),abs(row["rear_rad"])))}
                            if event["measured_speed_m_s"]>.005 or event["steering_error_deg"]>2.5:
                                raise RuntimeError("H3-B release-state observation changed before motion")
                            gate_events.append(event);break
                leg_start=clock["time_s"]
                profile=h["SignedStopProfile"](direction,target_progress,leg_start)
                print(f"H3-B rolling component leg {leg_index}: direction={direction:+d}, fixed longitudinal target={target_progress:.3f} m")
                while True:
                    measured=max(row["pose_filtered_speed_m_s"],math.hypot(linear[0],linear[1]))
                    speed,ready=profile.step(clock["time_s"],row["longitudinal_m"],measured)
                    if ready:break
                    phase=f"leg_{leg_index}_braking" if profile.braking else f"leg_{leg_index}_motion"
                    row,pose,linear=await sample(speed,False,leg_start)
                if abs(speed)>1e-7 or measured>.005:raise RuntimeError("H3-B stop required before capture/reversal")
            set_hold(True);report["fixed_hold_attempt_count"]=1
            phase="terminal_hold";hold_start=clock["time_s"]
            print("H3-B rolling component: ONE 5 s hold; evaluate fixed final 3 s, then Pause/restore")
            while clock["time_s"]-hold_start<5.:row,pose,linear=await sample(0.,True,hold_start)
            report["direction_gate_events"]=gate_events
            report.update(h["evaluate_readiness"](rows,goal,static_rows,prep["simulation_mass_kg"],
                doorway_reference_bound,radii,counts,callback_errors,report["fixed_hold_attempt_count"],gate_events))
            report["motion_and_braking_time_s"]=hold_start-motion_start
        except (Exception,asyncio.CancelledError) as error:
            report["error"]=f"{type(error).__name__}: {error}";report.setdefault("checks",{})["readiness_and_hold_completed"]=False
            report["verdict"]="FAIL"
        finally:
            if robot is not None and drive_indices is not None:
                try:robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
                except Exception as error:report["zero_command_error"]=str(error);report.setdefault("checks",{})["final_zero_command_sent"]=False
            if timeline.is_playing():timeline.pause()
            if motor is not None and old_kp is not None:
                try:
                    with Usd.EditContext(stage,stage.GetSessionLayer()):motor.set_gains(kps=old_kp,kds=old_kd,save_to_usd=False)
                    kp,kd=motor.get_gains()
                    restored=bool(np.allclose(kp,old_kp,rtol=1e-6,atol=1e-6) and np.array_equal(kd,old_kd)
                        and np.array_equal(np.asarray(motor.get_max_efforts()),old_limits))
                    report["original_motor_gains_restored"]=restored
                    if not restored:raise RuntimeError("Original gains/damping/torque limits not restored")
                    if old_targets is not None and np.isfinite(old_targets).all():
                        robot.apply_action(ArticulationAction(joint_positions=old_targets.astype(np.float32),
                            joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
                except Exception as error:report["restore_error"]=str(error);report["original_motor_gains_restored"]=False
            report.setdefault("checks",{})["original_motor_gains_restored"]=report.get("original_motor_gains_restored",False)
            try:
                support["live_geometry_check"](stage,manifest,geometry,audit,set(prep["disabled_legacy_floor_colliders"]))
                report["checks"]["same_authored_support_contact_and_motor_baseline"]=support["support_readback"](stage,manifest)==prep["live_support_readback"]
                support["load_setup"](manifest_path.parent,workspace)
                report["checks"]["flush_environment_and_original_evidence_preserved"]=True
            except Exception as error:
                report["cleanup_environment_error"]=str(error)
                report["checks"]["flush_environment_and_original_evidence_preserved"]=False
            if callback_errors or any(counts[key] for key in h["BAD_CONTACTS"]):report["checks"]["no_obstacle_nonwheel_or_legacy_contacts"]=False
            if not all(report["checks"].values()):report["verdict"]="FAIL"
            physics_sub=contact_sub=None
            report.update(component_crossing_verified=False,H3B_rolling_readiness_verified=report["verdict"]=="PASS",
                direction_gate_events=gate_events,sample_count=len(rows),
                contact_report_counts_by_category=dict(counts),contact_callback_errors=callback_errors,
                contact_pairs_by_category=[{"pair":pair,"category":category,"callback_count":count} for (pair,category),count in sorted(pairs.items())],
                wall_time_s=time.perf_counter()-wall_start)
            if rows:
                report["last_recorded_state"]=dict(rows[-1])
                with trace_path.open("x",newline="",encoding="utf-8") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with summary_path.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                marker.SetCustomDataByKey("status","h3b_rolling_readiness_finished_fresh_setup_required")
                marker.SetCustomDataByKey("last_h3b_rolling_readiness_gate",str(summary_path))
            print("=== H3-B ROLLING STEERING + STRAIGHT-CUSP COMPONENT GATE ===")
            print("run:",run_id);print("contact categories:",dict(counts))
            if "terminal_position_error_m" in report:
                print("terminal error mm/deg:",round(1000*report["terminal_position_error_m"],3),round(report["terminal_heading_error_deg"],6))
                print("hold regression speed mm/s:",round(1000*report["fixed_terminal_hold_analysis"]["pose_regression_speed_m_s"],3))
            if "measured_moving_steering_peaks_deg" in report:
                print("moving actual front/rear angle ranges deg:",report["measured_moving_steering_peaks_deg"])
                print("maximum recorded steering command slew deg/s:",report["maximum_observed_command_slew_deg_s"])
            print("original motor gains restored:",report.get("original_motor_gains_restored",False))
            print("failed checks:",[key for key,value in report["checks"].items() if not value])
            if report.get("error"):print("error:",report["error"])
            if report.get("restore_error"):print("restore error:",report["restore_error"])
            if report.get("cleanup_environment_error"):print("environment error:",report["cleanup_environment_error"])
            print("VERDICT:",report["verdict"]);print("saved:",summary_path)
            print("Timeline PAUSED; original gains restoration shown above. Representative moving steering/straight cusp component ONLY; original stationary alignment, saved-route execution, dynamics/Nav2 remain pending.")
            print("Do not press Play or save the source USD. Fresh source/flush geometry/support/static PASS required for another attempt.")

    task=asyncio.ensure_future(run())
    print("H3-B representative rolling-steering readiness scheduled:",run_id)
    print("Automatic 2 s pre-hold -> forward <=0.22 m with S steering then straight -> measured STOP/HOLD/straight/RELEASE -> reverse 0.05 m -> ONE 5 s hold -> Pause/restore.")
    print("Body reference commands <=0.01 m/s; steering command slew <=8 deg/s; existing 5 mm/s stop gate; fixed ORIGINAL-start component targets; 15 s heartbeat. No CICS/Nav2.")
    print("Representative component only, NOT an identical segment/full replay of the saved warm-start candidate; original stationary FAILs remain.")
    print("Do not touch timeline controls. Preserve this attempt regardless of PASS/FAIL.")
    return task


ijcas_hospital_h3b_rolling_readiness_task=_schedule_h3b_rolling_readiness()

