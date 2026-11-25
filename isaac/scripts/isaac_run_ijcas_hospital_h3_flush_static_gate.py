"""H3_flush static support ONLY: automatic Play/1 s settle/2 s sample/Pause.

Same H1/H2 stationary limits and velocity-only motor baseline. No forward,
CICS, position-hold gain, path tracking, sensor localization or Nav2 command.
"""


def _schedule_h3_flush_static():
    import asyncio
    from collections import Counter
    import csv
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
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

    workspace = Path("/home/jk/Research/IJCAS_ws")
    scripts = workspace / "studies/IJCAS/scripts"
    h = runpy.run_path(str(scripts/"hospital_h3_flush_support.py"))
    rules = runpy.run_path(str(scripts/"hospital_static_gate.py"))
    classify = runpy.run_path(str(scripts/"hospital_contact_rules.py"))["classify_contact"]
    for name, task in list(globals().items()):
        if name.startswith("ijcas_hospital_") and name.endswith("_task") and hasattr(task,"done") and not task.done():
            raise RuntimeError("A hospital task is already running")
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    if not timeline.is_stopped() or stage is None:
        raise RuntimeError("Keep STOPPED immediately after the dedicated H3_flush static preparation")
    marker = stage.GetPrimAtPath(h["COMPONENT"])
    if (not marker.IsValid() or marker.GetCustomDataByKey("condition") != h["CONDITION"]
            or marker.GetCustomDataByKey("status") != "physics_prepared_static_pending"):
        raise RuntimeError("Run isaac_prepare_ijcas_hospital_h3_flush_static.py first; do not reuse old H3/H1/H2 starts")
    preparation_path = Path(marker.GetCustomDataByKey("physics_preparation_result") or "")
    prep = json.loads(preparation_path.read_bytes())
    if (h["sha"](preparation_path) != marker.GetCustomDataByKey("physics_preparation_sha256")
            or prep.get("environment_condition") != h["CONDITION"]
            or prep.get("selected_threshold_visual_and_collision_inactive") is not True
            or preparation_path.with_name(preparation_path.stem+"_preparation_failure.json").exists()):
        raise RuntimeError("Invalid/rolled-back H3_flush physics preparation")
    manifest_path, manifest, geometry, audit = h["load_setup"](Path(prep["geometry_manifest"]).parent,workspace)
    if (h["sha"](manifest_path) != prep["geometry_manifest_sha256"]
            or h["sha"](manifest_path) != marker.GetCustomDataByKey("setup_manifest_sha256")
            or Path(marker.GetCustomDataByKey("setup_manifest") or "").resolve() != manifest_path.resolve()):
        raise RuntimeError("H3_flush setup changed after preparation")
    for name, expected in prep["source_sha256"].items():
        folder = workspace/("isaac/scripts" if name.startswith("isaac_") else "studies/IJCAS/scripts")
        if h["sha"](folder/name) != expected:
            raise RuntimeError("Recorded preparation source changed: " + name)
    if h["sha"](prep["robot_reference"]) != prep["robot_reference_sha256"]:
        raise RuntimeError("Robot reference changed")
    h["live_geometry_check"](stage,manifest,geometry,audit,set(prep["disabled_legacy_floor_colliders"]))
    if h["support_readback"](stage,manifest) != prep["live_support_readback"]:
        raise RuntimeError("Actual floor/contact/motor/mass baseline changed")
    if any(p.GetTypeName() == "OmniGraph" for p in stage.Traverse()):
        raise RuntimeError("A competing action graph is active")
    for path in prep["disabled_legacy_floor_colliders"]:
        if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get() is not False:
            raise RuntimeError("A duplicate floor collider remains enabled")
    root, base = h["ROBOT"],h["ROBOT"]+"/base_link"
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(base))
    position, forward = matrix.ExtractTranslation(),matrix.TransformDir(Gf.Vec3d(1,0,0))
    yaw_error = math.atan2(forward[1],forward[0])-manifest["start_xyyaw"][2]
    if (math.hypot(position[0]-manifest["start_xyyaw"][0],position[1]-manifest["start_xyyaw"][1]) > .005
            or abs(math.atan2(math.sin(yaw_error),math.cos(yaw_error))) > math.radians(.5)):
        raise RuntimeError("Robot moved from the audited outside-door static staging pose")
    wheels = dict(zip(("active_front","active_rear","caster_fl","caster_fr","caster_rl","caster_rr"),h["WHEELS"]))
    steering = ("base_link_to_front_active_steering","base_link_to_rear_active_steering")
    drives = ("front_active_steering_to_drive_wheel","rear_active_steering_to_drive_wheel")
    output = workspace/"studies/IJCAS/results/hospital_paper_gate/hospital_h3_flush_static_contact"
    output.mkdir(parents=True,exist_ok=True)
    number = 1
    while any((output/f"hospital_h3_flush_static_r{number:02d}{extension}").exists() for extension in (".json",".csv")):
        number += 1
    run_id = f"hospital_h3_flush_static_r{number:02d}"
    summary_path, trace_path = output/(run_id+".json"),output/(run_id+".csv")
    clock = {"time_s":0.}
    counts, pairs, callback_errors = Counter(),Counter(),[]

    def physics_step(dt):
        clock["time_s"] += float(dt)

    def contacts(headers, data):
        del data
        try:
            for header in headers:
                pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
                category = classify(pair,{prep["support_floor"]},set(prep["disabled_legacy_floor_colliders"]))
                if category != "unrelated":
                    counts[category] += 1
                    pairs[(pair,category)] += 1
        except Exception as error:
            callback_errors.append(str(error))

    def angles(q):
        w,x,y,z = map(float,q)
        return (math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1.,min(1.,2*(w*y-z*x))))),
            math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))))

    async def run():
        report = {"schema_version":1,"run_id":run_id,"scenario":manifest["scenario"],"environment_condition":h["CONDITION"],
            "trial_kind":"H3_FLUSH_OUTSIDE_DOOR_STATIC_COMPONENT_ONLY_NOT_PASSAGE_OR_TURNING_TASK",
            "verdict":"FAIL","checks":{"stationary_observation_completed":False},
            "geometry_manifest":str(manifest_path),"geometry_manifest_sha256":h["sha"](manifest_path),
            "physics_preparation":str(preparation_path),"physics_preparation_sha256":h["sha"](preparation_path),
            "physics_preparation_snapshot":prep,"simulation_mass_kg":prep["simulation_mass_kg"],
            "start_xyyaw":manifest["start_xyyaw"],"settle_s":1.,"sampling_s":2.,
            "initial_motion_executed":False,"planned_path_executed":False,"component_crossing_verified":False,
            "whole_component_PhysX_support_verified":False,"Nav2_executed":False,"CICS_input_used":False,
            "motor_gains_modified_by_test":False,"caster_joint_state_reset_during_run":False,
            "source_stage_saved":False,"use_for_final_statistics":False,
            "source_sha256":{p.name:h["sha"](p) for p in (
                workspace/"isaac/scripts/isaac_run_ijcas_hospital_h3_flush_static_gate.py",scripts/"hospital_h3_flush_support.py",
                scripts/"hospital_static_gate.py",scripts/"hospital_contact_rules.py")}}
        rows, views = [],[]
        robot = motor = drive_indices = old_kp = old_kd = old_limits = None
        physics_sub = contact_sub = None
        app = omni.kit.app.get_app()
        wall_start = time.perf_counter()
        try:
            physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(physics_step)
            contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
            for label,name in wheels.items():
                views.append(RigidContactView(prim_paths_expr=root+"/"+name,filter_paths_expr=[],name=run_id+"_"+label,
                    prepare_contact_sensors=False,disable_stablization=False))
            timeline.play()
            for _ in range(5):
                await app.next_update_async()
            if not timeline.is_playing():
                raise RuntimeError("Timeline changed during flush static initialization")
            robot = SingleArticulation(base)
            robot.initialize()
            indices = {name:i for i,name in enumerate(robot.dof_names)}
            if any(name not in indices for name in (*steering,*drives)):
                raise RuntimeError("Missing active joints")
            steer_indices = np.asarray([indices[name] for name in steering],dtype=np.int32)
            drive_indices = np.asarray([indices[name] for name in drives],dtype=np.int32)
            motor = robot.get_articulation_controller()
            kp,kd = motor.get_gains()
            old_kp,old_kd = np.asarray(kp).copy(),np.asarray(kd).copy()
            old_limits = np.asarray(motor.get_max_efforts()).copy()
            if np.max(np.abs(old_kp[drive_indices])) > 1e-6:
                raise RuntimeError("Original velocity-only active drive gains required")

            def zero_command():
                robot.apply_action(ArticulationAction(joint_positions=np.zeros(2,dtype=np.float32),joint_indices=steer_indices))
                robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))

            for view in views:
                view.initialize()
            sim = omni.physx.get_physx_simulation_interface()
            stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            encoded = PhysicsSchemaTools.sdfPathToInt(base)
            began = previous = clock["time_s"]
            next_sample = 1.
            while clock["time_s"]-began < 3.:
                zero_command()
                sim.wake_up(stage_id,encoded)
                await app.next_update_async()
                if not timeline.is_playing():
                    raise RuntimeError("Timeline changed during flush static recording")
                if time.perf_counter()-wall_start > 120.:
                    raise TimeoutError("H3_flush static 120 s wall-time guard")
                now = clock["time_s"]
                if now-previous > .25:
                    raise TimeoutError("H3_flush static 0.25 s control-gap guard")
                previous = now
                position,quaternion = robot.get_world_pose()
                position = np.asarray(position,dtype=float)
                if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                    raise RuntimeError("Non-finite pose")
                if np.linalg.norm(position[:2]-np.asarray(manifest["start_xyyaw"][:2])) > .030:
                    raise RuntimeError("Zero-command drift exceeded 30 mm safety bound")
                roll,pitch,yaw = angles(quaternion)
                if max(abs(roll),abs(pitch)) > 2.:
                    raise RuntimeError("Static tilt safety guard")
                if callback_errors or any(counts[key] for key in rules["BAD_CONTACTS"]):
                    raise RuntimeError("Unexpected obstacle/legacy/nonwheel/unclassified contact")
                actual_kp,actual_kd = motor.get_gains()
                if not (np.array_equal(actual_kp,old_kp) and np.array_equal(actual_kd,old_kd)
                        and np.array_equal(np.asarray(motor.get_max_efforts()),old_limits)):
                    raise RuntimeError("Motor gains/damping/limits changed during static test")
                elapsed = now-began
                if elapsed < next_sample:
                    continue
                forces = [np.asarray(view.get_net_contact_forces(dt=1./120.),dtype=float).reshape(-1,3)[0] for view in views]
                if not np.isfinite(forces).all():
                    raise RuntimeError("Non-finite wheel force")
                row = {"time_s":elapsed,"x_m":float(position[0]),"y_m":float(position[1]),"z_m":float(position[2]),
                    "roll_deg":roll,"pitch_deg":pitch,"yaw_deg":yaw}
                row.update({label+"_fz_n":float(force[2]) for label,force in zip(wheels,forces)})
                rows.append(row)
                next_sample += 1./60.
            zero_command()
            report.update(rules["evaluate_static"](rows,manifest["start_xyyaw"],prep["simulation_mass_kg"],counts,callback_errors))
            report["checks"]["stationary_observation_completed"] = True
        except (Exception,asyncio.CancelledError) as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["verdict"] = "FAIL"
        finally:
            try:
                if robot is not None and drive_indices is not None and timeline.is_playing():
                    robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
            except Exception as error:
                report["checks"]["final_zero_command_sent"] = False
                report["zero_command_error"] = str(error)
            if timeline.is_playing():
                timeline.pause()
            physics_sub = contact_sub = None
            try:
                if motor is None or old_kp is None:
                    raise RuntimeError("Runtime motor readback unavailable")
                actual_kp,actual_kd = motor.get_gains()
                report["checks"]["original_motor_gains_and_limits_unchanged"] = bool(np.array_equal(actual_kp,old_kp)
                    and np.array_equal(actual_kd,old_kd) and np.array_equal(np.asarray(motor.get_max_efforts()),old_limits))
                h["live_geometry_check"](stage,manifest,geometry,audit,set(prep["disabled_legacy_floor_colliders"]))
                report["checks"]["flush_environment_and_original_evidence_preserved"] = True
                report["checks"]["same_authored_support_and_contact_baseline"] = h["support_readback"](stage,manifest) == prep["live_support_readback"]
                h["load_setup"](manifest_path.parent,workspace)
            except Exception as error:
                report["checks"]["cleanup_and_environment_readback_healthy"] = False
                report["cleanup_error"] = str(error)
            report["checks"]["contact_callback_monitor_healthy"] = not callback_errors
            report["checks"]["only_expected_external_support"] = not any(counts[key] for key in rules["BAD_CONTACTS"])
            if report.get("error") or not all(report["checks"].values()):
                report["verdict"] = "FAIL"
            report.update(contact_report_counts_by_category=dict(counts),contact_callback_errors=callback_errors,
                contact_pairs_by_category=[{"pair":pair,"category":category,"callback_count":count}
                    for (pair,category),count in sorted(pairs.items())],wall_time_s=time.perf_counter()-wall_start)
            if rows:
                with trace_path.open("x",newline="",encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with summary_path.open("x",encoding="utf-8") as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
                stream.write("\n")
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                marker.SetCustomDataByKey("status","static_support_passed" if report["verdict"] == "PASS" else "static_support_failed")
                marker.SetCustomDataByKey("static_gate_result",str(summary_path))
                marker.SetCustomDataByKey("static_gate_sha256",h["sha"](summary_path))
            print("=== H3 FLUSH COMPONENT STATIC CONTACT GATE ===")
            print("run:",run_id)
            for label,load in report.get("mean_wheel_vertical_loads_n",{}).items():
                print(f"{label}: {load:.2f} N")
            print("contact categories:",dict(counts))
            if "maximum_xy_drift_m" in report:
                print("maximum XY drift mm:",round(1000*report["maximum_xy_drift_m"],3))
                print("base Z span mm:",round(1000*report["base_z_span_m"],3))
                print("maximum roll/pitch deg:",round(report["maximum_abs_roll_pitch_deg"],3))
                print("vertical force / weight:",round(report["mean_vertical_force_to_weight_ratio"],4))
            print("failed checks:",[key for key,value in report["checks"].items() if not value])
            if report.get("error"):
                print("error:",report["error"])
            if report.get("cleanup_error"):
                print("cleanup error:",report["cleanup_error"])
            print("VERDICT:",report["verdict"])
            print("saved:",summary_path)
            print("Timeline PAUSED. Outside-door STATIC support only; whole passage, H3 turning plan and Nav2 have NOT run.")
            print("Do not press Play, save the source USD, or run the old retained-threshold controllers.")

    task = asyncio.ensure_future(run())
    print("H3_flush outside-door static component scheduled:",run_id)
    print("Automatic Play -> 1 s settling -> 2 s recording -> Pause. No forward/CICS/Nav2 command.")
    print("Do not touch timeline controls. Use the saved flush-specific result, not the original H3 static gate.")
    return task


ijcas_hospital_h3_flush_static_task = _schedule_h3_flush_static()
