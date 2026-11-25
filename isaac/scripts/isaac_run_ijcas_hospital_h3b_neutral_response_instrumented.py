"""ONE matched-length straight CP30 identification with passive wheel/body/friction data.

The primary control/guard/cleanup logic is unchanged in a separate cohort. All observation
capabilities are checked BEFORE scheduling it. No observer-issued action,
initialization, pose reset, material/gain change or stop-gate change is used.
"""


def _schedule_cp30_observation():
    import asyncio
    import csv
    import json
    import math
    from pathlib import Path
    import runpy

    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import PhysicsSchemaTools

    if globals().pop("ijcas_hospital_h3b_neutral_observer_condition",None)!="H3B_neutral_straight_identification_observation":
        raise RuntimeError("Explicitly select H3B_neutral_straight_identification_observation")
    for name,task in list(globals().items()):
        if name.startswith("ijcas_hospital_") and name.endswith("_task") and hasattr(task,"done") and not task.done():
            raise RuntimeError("A hospital trial is running")
    workspace=Path("/home/jk/Research/IJCAS_ws")
    scripts=workspace/"studies/IJCAS/scripts"
    gate=workspace/"studies/IJCAS/results/hospital_paper_gate"
    loader=runpy.run_path(str(scripts/"hospital_private_loader.py"))["load"]
    helper=loader(scripts/"hospital_h3b_neutral_response.py")
    rules=loader(scripts/"hospital_h3b_neutral_response.py")
    contacts_helper=runpy.run_path(str(scripts/"hospital_h3_instrumentation.py"))
    support=runpy.run_path(str(scripts/"hospital_h3_flush_support.py"))
    reviewed=helper["load_protocol"]()
    reference_path=helper["REFERENCE"]
    reference=json.loads(reference_path.read_bytes())
    output=helper["OBSERVER_OUTPUT"]
    if output.exists() or helper["OUTPUT"].exists():
        raise RuntimeError("One CP30 observation attempt already exists; preserve it; no automatic retry")
    timeline=omni.timeline.get_timeline_interface()
    stage=omni.usd.get_context().get_stage()
    if timeline.is_stopped() or timeline.is_playing():
        raise RuntimeError("Fresh original source -> CP30 observation preparation -> actual static PASS must remain PAUSED")
    marker=stage.GetPrimAtPath(support["COMPONENT"]) if stage else None
    if marker is None or not marker.IsValid() or marker.GetCustomDataByKey("status")!="static_support_passed":
        raise RuntimeError("Do not resume at the failed endpoint; require a fresh dedicated CP30 observation preparation/static PASS")
    static_path=Path(marker.GetCustomDataByKey("static_gate_result") or "")
    static=json.loads(static_path.read_bytes())
    if (static.get("verdict")!="PASS" or not static.get("checks") or not all(static["checks"].values())
            or support["sha"](static_path)!=marker.GetCustomDataByKey("static_gate_sha256")):
        raise RuntimeError("Fresh static PASS evidence missing")
    prep=static["physics_preparation_snapshot"]
    original=reference["physics_preparation_snapshot"]
    for key in ("environment_condition","source_stage_sha256","robot_reference_sha256","start_xyyaw","component_end_xyyaw",
            "simulation_mass_kg","live_support_readback","support_floor","disabled_legacy_floor_colliders"):
        if prep[key]!=original[key]:
            raise RuntimeError("Same-condition preparation differs: "+key)
    manifest=json.loads(Path(prep["geometry_manifest"]).read_bytes())
    if manifest["strict_map_digest"]!=reference["strict_flush_map_digest"]:
        raise RuntimeError("Same-condition strict map changed")
    for name,digest in reference["source_sha256"].items():
        folder=workspace/("isaac/scripts" if name.startswith("isaac_") else "studies/IJCAS/scripts")
        if support["sha"](folder/name)!=digest:
            raise RuntimeError("Frozen measured rolling controller changed: "+name)
    sim_view=SimulationManager.get_physics_sim_view()
    if sim_view is None:
        raise RuntimeError("No existing physics view; observer will not initialize/reset the scene")
    root_before,session_before=stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()
    view=sim_view.create_articulation_view(rules["ROBOT_ROOT"]+"/base_link")
    if view.count!=1:
        raise RuntimeError("Expected exactly one existing articulation")
    helper["wheel_body_snapshot"](view,0.)
    preparation_path=Path(marker.GetCustomDataByKey("H3B_mode_speed_preparation") or "")
    preparation=json.loads(preparation_path.read_bytes())
    if (support["sha"](preparation_path)!=marker.GetCustomDataByKey("H3B_mode_speed_preparation_sha256")
            or preparation.get("case")!=helper["CASE"]
            or preparation.get("neutral_identification_protocol_sha256")!=support["sha"](helper["PREPARATION_PROTOCOL_PATH"])):
        raise RuntimeError("Fresh observation-specific CP30 preparation required; not the old CP30 endpoint")
    # SAME_PHASE_HANDOFF_BEGIN
    receipt,_=helper["load_receipt"]()
    def current_joint_settings():
        settings={}
        for key,getter in (("kp",view.get_dof_stiffnesses),("kd",view.get_dof_dampings),
                ("force_limit",view.get_dof_max_forces),("velocity_target_rad_s",view.get_dof_velocity_targets)):
            settings[key]=dict(zip(list(view.shared_metatype.dof_names),helper["vector"](getter(),12)))
        return settings
    retained=helper["validate_handoff"](receipt,static_path,helper["vector"](view.get_root_transforms(),7),
        list(view.shared_metatype.dof_names),helper["vector"](view.get_dof_positions(),12),current_joint_settings())
    # SAME_PHASE_HANDOFF_END
    patterns=[rules["ROBOT_ROOT"]+"/"+name for name in rules["WHEELS"]]
    friction_view=sim_view.create_rigid_contact_view(patterns,[[prep["support_floor"]] for _ in patterns],max_contact_data_count=256)
    paths=[str(path) for path in friction_view.sensor_paths]
    if friction_view.sensor_count!=6 or friction_view.filter_count!=1 or len(paths)!=6:
        raise RuntimeError("Require six native wheel sensors with exactly one canonical-floor filter")
    matched=[]
    for path in paths:
        matches=[pattern for pattern in patterns if path==pattern or path.startswith(pattern+"/")]
        if len(matches)!=1:
            raise RuntimeError("Native contact sensor is not a unique expected wheel: "+path)
        matched.extend(matches)
    if set(matched)!=set(patterns) or len(set(matched))!=6:
        raise RuntimeError("Native wheel-contact sensor coverage is not one-to-one")
    # Native SDK only: no sensor API Apply/Set/initialize or stabilization change.
    helper["friction_snapshot"](friction_view,0.,1./120.,paths)
    if (stage.GetRootLayer().ExportToString()!=root_before or stage.GetSessionLayer().ExportToString()!=session_before
            or timeline.is_stopped() or timeline.is_playing()):
        raise RuntimeError("Observer capability check changed scene/timeline; primary NOT launched")
    # SAME_PHASE_RECHECK_BEGIN
    rechecked=helper["validate_handoff"](receipt,static_path,helper["vector"](view.get_root_transforms(),7),
        list(view.shared_metatype.dof_names),helper["vector"](view.get_dof_positions(),12),current_joint_settings())
    if rechecked!=retained:
        raise RuntimeError("Static body/joints changed during passive SDK preflight; primary NOT launched")
    print("H3-B SAME-PHASE static body/twelve joints: PASS; old 1mm/0.1deg/0.001rad tolerances retained.")
    print("Fresh actual static matched to recorded CP30 PRE-hold baseline; no post-Play comparison, caster assignment or tolerance waiver.")
    # SAME_PHASE_RECHECK_END
    clock={"time_s":0.,"next_friction_s":0.}
    joint_rows=[]
    friction_rows=[]
    anchors=[]
    raw_contacts=[]
    errors=[]

    def remember(error):
        message=f"{type(error).__name__}: {error}"
        if message not in errors and len(errors)<100:
            errors.append(message)

    def step(dt):
        try:
            dt=float(dt)
            if not math.isfinite(dt) or not 0.<dt<=.025:
                raise ValueError("Invalid observer physics timestep")
            clock["time_s"]+=dt
            if len(joint_rows)>=12000:
                raise RuntimeError("Observer joint capacity reached; primary policy is unchanged")
            joint_rows.append(helper["wheel_body_snapshot"](view,clock["time_s"]))
            if clock["time_s"]+1e-8>=clock["next_friction_s"]:
                clock["next_friction_s"]+=helper["FRICTION_PERIOD_S"]
                row,points=helper["friction_snapshot"](friction_view,clock["time_s"],dt,paths)
                if len(anchors)+len(points)>100000:
                    raise RuntimeError("Observer friction capacity reached; primary policy is unchanged")
                friction_rows.append(row)
                anchors.extend(points)
        except Exception as error:
            remember(error)

    def contacts(headers,data):
        try:
            points=contacts_helper["contact_points"](headers,data,clock["time_s"],PhysicsSchemaTools.intToSdfPath,
                lambda pair:rules["classify_contact"](pair,prep["support_floor"],prep["disabled_legacy_floor_colliders"]))
            if len(raw_contacts)+len(points)>500000:
                raise RuntimeError("Observer raw-contact capacity reached; primary policy is unchanged")
            raw_contacts.extend(points)
        except Exception as error:
            remember(error)

    physics_sub=omni.physx.get_physx_interface().subscribe_physics_step_events(step)
    contact_sub=None
    primary=None
    try:
        contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
        globals()["ijcas_hospital_h3b_neutral_component_condition"]="H3B_neutral_straight_component_v1"
        globals()["ijcas_hospital_h3b_neutral_observer_ready"]="NEUTRAL_OBSERVER_PREFLIGHT_COMPLETE"
        entry=workspace/"isaac/scripts/isaac_run_ijcas_hospital_h3b_neutral_response_primary.py"
        exec(compile(entry.read_text(encoding="utf-8"),str(entry),"exec"),globals())
        primary=globals()["ijcas_hospital_h3b_neutral_response_task"]
        state=primary.get_coro().cr_frame.f_locals
        run_id=state["run_id"]
        primary_path=Path(state["summary_path"])
        output.mkdir(parents=True,exist_ok=False)
    except Exception:
        # No event-loop yield: the primary has not issued any action/Play yet.
        if primary is not None and not primary.done():
            primary.cancel()
        physics_sub=contact_sub=None
        raise

    async def finish():
        nonlocal physics_sub,contact_sub
        report={"run_id":run_id,"role":"NEUTRAL_STRAIGHT_CP30_IDENTIFICATION_NOT_VALIDATION",
            "status":"INCOMPLETE","reference_trial":str(reference_path),"reference_trial_sha256":support["sha"](reference_path),
            "reviewed_paused_configuration":reviewed,"reference_input_sha256":reviewed["input_sha256"],
            "original_controller_result":str(primary_path),
            "same_condition_protocol":str(helper["PROTOCOL_PATH"]),"same_condition_protocol_sha256":support["sha"](helper["PROTOCOL_PATH"]),
            "initial_actual_static_body_and_twelve_joints_match":retained,
            # SAME_PHASE_PROVENANCE_BEGIN
            "CP30_baseline_static_readback":str(helper["READBACK"]),"CP30_baseline_static_readback_sha256":support["sha"](helper["READBACK"]),
            "primary_v2_base_protocol":str(helper["PREPARATION_PROTOCOL_PATH"]),
            "primary_v2_base_protocol_sha256":support["sha"](helper["PREPARATION_PROTOCOL_PATH"]),
            "CP30_actual_static_baseline_twelve_joint_match_required":True,
            "state_equality_tolerances_unchanged":helper["STATE_TOLERANCES_UNCHANGED"],
            # SAME_PHASE_PROVENANCE_END
            "six_wheel_transforms_COM_velocities_and_origin_velocities_recorded":True,
            "no_parameter_fit_or_whole_H3_drive_admission":True,"fresh_static_reference":str(static_path),
            "fresh_static_sha256":support["sha"](static_path),"comparison_CP30_controller_source_sha256":reference["source_sha256"],
            "control_or_physics_parameters_modified_by_observer":False,"original_guards_and_verdict_policy_unchanged":True,
            "joint_observer_nominal_hz":120.,"friction_observer_nominal_hz":30.,"native_friction_buffer_capacity":256,
            "native_sensor_paths_in_recorded_wheel_index_order":paths,"native_filter":prep["support_floor"],
            "forces_are_SDK_last_step_impulses_divided_by_physics_dt_NOT_observer_period":True,
            "SDK_world_vector_sign_convention_retained_not_assumed_motor_or_traction_sign":True,
            "projected_joint_effort_is_NOT_isolated_actuator_motor_torque":True,
            "callback_timestamp_uncertainty_s":1./120.,"callback_and_controller_samples_NOT_substep_synchronized":True,
            "SDK_root_velocity_at_COM_and_converted_link_origin_velocity_both_recorded":True,
            "saved_warm_start_trajectory_executed":False,"full_dynamics_plan_certified":False,
            "full_H3B_turning_path_verified":False,"use_for_final_statistics":False,
            "source_sha256":{p.name:support["sha"](p) for p in (workspace/"isaac/scripts/isaac_run_ijcas_hospital_h3b_neutral_response_instrumented.py",
                scripts/"hospital_h3b_neutral_response.py",scripts/"hospital_h3_instrumentation.py",scripts/"hospital_h3b_alignment_audit.py",
                scripts/"hospital_private_loader.py")}}
        original={}
        try:
            app=omni.kit.app.get_app()
            while not primary.done():
                await app.next_update_async()
            if primary.cancelled() or primary.exception() is not None:
                raise RuntimeError("Original rolling task ended abnormally; inspect its authoritative result")
            original=json.loads(primary_path.read_bytes())
            report["original_controller_verdict"]=original["verdict"]
            report["original_controller_source_sha256"]=original["source_sha256"]
            report["original_controller_result_sha256"]=support["sha"](primary_path)
            report["original_motor_gains_restored"]=original.get("original_motor_gains_restored",False)
            report["original_error"]=original.get("error")
            if helper["function_contract"]()!=reviewed["controller_contract"]:
                raise RuntimeError("Frozen control/guard/cleanup contract changed during observation")
            support["verify_files"](reviewed["input_sha256"])
        except (Exception,asyncio.CancelledError) as error:
            remember(error)
        finally:
            physics_sub=contact_sub=None
            for name,rows in (("joint_body_solver_trace.csv",joint_rows),("wheel_friction_summary.csv",friction_rows),
                    ("friction_anchors.csv",anchors),("raw_contact_points.csv",raw_contacts)):
                if rows:
                    with (output/name).open("x",newline="",encoding="utf-8") as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            report.update(helper["coverage"](joint_rows,friction_rows,anchors,raw_contacts,original,errors))
            report["telemetry_errors"]=errors
            report["saved_file_sha256"]={p.name:support["sha"](p) for p in output.glob("*.csv")}
            with (output/"instrumentation.json").open("x",encoding="utf-8") as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
                stream.write("\n")
            print("=== H3-B NEUTRAL CP30 IDENTIFICATION WHEEL/BODY/CONTACT INSTRUMENTATION ===")
            print("original controller verdict:",report.get("original_controller_verdict","NOT_AVAILABLE"))
            print("joint / friction samples / friction anchors:",len(joint_rows),len(friction_rows),len(anchors))
            print("each wheel has friction data:",report["each_wheel_has_friction_anchor_observations"])
            print("telemetry errors:",errors)
            print("STATUS:",report["status"])
            print("saved:",output/"instrumentation.json")
            print("Keep PAUSED. Original PASS/FAIL stays authoritative. Observer completion is NOT rolling success or an isolated cause.")

    print("H3-B NEW neutral input: ONLY curvature -> zero; same 0.65m/30mm_s/ramp/gains/guards and prescribed caster seed. Identification, not model validation.")
    print("Passive 120 Hz joint/body/SIX-WHEEL pose+velocity and 30 Hz native friction/raw contacts; no fit or model/H3 admission.")
    print("Another original FAIL is a valid outcome. No automatic additional attempt; do not touch timeline controls.")
    return asyncio.ensure_future(finish())


ijcas_hospital_h3b_neutral_observer_task=_schedule_cp30_observation()
