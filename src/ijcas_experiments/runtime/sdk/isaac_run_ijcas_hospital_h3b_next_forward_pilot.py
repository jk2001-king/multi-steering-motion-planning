"""ONE next counterphase FORWARD from the retained measured-reverse PASS.

Same hospital/map/goal/physics and actual caster history. Reuse the frozen
next-leg course-feedback prototype, not a new planner or reverse controller.
One stop/neutral/release handshake, one 5s hold, pause/restore; no later leg.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

def _schedule_next_forward_pilot():
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
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf,PhysicsSchemaTools,Usd,UsdGeom,UsdPhysics,UsdUtils

    if globals().pop('ijcas_hospital_h3b_next_forward_observer_ready',None)!='PASSIVE_OBSERVER_PREFLIGHT_COMPLETE':
        raise RuntimeError('Use the dedicated NEXT-forward instrumented wrapper; no uninstrumented retry')
    for name,task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task,'done') and not task.done():
            raise RuntimeError('Another hospital task is running')
    workspace=_ijcas_workspace();scripts=_ijcas_path(workspace, 'studies/IJCAS/scripts')
    loader=runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h=loader(scripts/'hospital_h3b_next_forward.py');info=h['load_admitted_profile']()
    support=runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    hold=runpy.run_path(str(scripts/'hospital_drive_hold.py'))['drive_hold_gain_arrays']
    collision=runpy.run_path(str(scripts/'hospital_geometry_tools.py'))['pose_collision']
    timeline=omni.timeline.get_timeline_interface();context=omni.usd.get_context();stage=context.get_stage()
    if timeline.is_playing() or timeline.is_stopped():raise RuntimeError('Keep the SAME measured-terminal reverse PASS PAUSED; no reopening/reset/Play')
    marker=stage.GetPrimAtPath(support['COMPONENT']) if stage else None
    first=info['preparation_record'];prior_result_path=info['reverse_result']
    if (not marker or not marker.IsValid()
            or marker.GetCustomDataByKey('status')!='h3b_reverse_closedloop_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_reverse_closedloop_pilot_gate')!=prior_result_path):
        raise RuntimeError('Need the actual SAME measured-terminal reverse-PASS scene; no substituted start')
    manifest_path,manifest,geometry,inventory=support['load_setup'](Path(first['geometry_manifest']).parent,workspace)
    prep=first['physics_preparation_snapshot']
    support['live_geometry_check'](stage,manifest,geometry,inventory,set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage,manifest)!=prep['live_support_readback']:
        raise RuntimeError('Original support/contact/motor baseline changed')
    if any(p.GetTypeName()=='OmniGraph' for p in stage.Traverse()):raise RuntimeError('Competing action graph is active')
    static_path=Path(first['static_reference']);static=json.loads(static_path.read_bytes())
    support['verify_files']({static_path:first['static_reference_sha256'],static_path.with_suffix('.csv'):first['static_trace_sha256'],
        prep['robot_reference']:prep['robot_reference_sha256']})
    with static_path.with_suffix('.csv').open(newline='') as stream:
        static_rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    if runpy.run_path(str(scripts/'hospital_static_gate.py'))['evaluate_static'](static_rows,static['start_xyyaw'],static['simulation_mass_kg'],
            static['contact_report_counts_by_category'],static['contact_callback_errors'])['verdict']!='PASS':
        raise RuntimeError('Original raw static support PASS does not reproduce')
    with np.load(manifest['occupancy_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    if (hashlib.sha256(occupied.tobytes()).hexdigest()!=info['strict_map_digest'] or info['strict_map_digest']!=manifest['strict_map_digest']
            or occupied.dtype!=np.dtype(bool) or occupied.shape!=tuple(manifest['raster']['shape_rows_columns'])):
        raise RuntimeError('Strict flush raster changed')
    door=json.loads(Path(manifest['fully_open_geometry_report']).read_bytes())
    leaf=stage.GetPrimAtPath(door['assets']['leaf']['path']);leaf_matrix=UsdGeom.XformCache().GetLocalToWorldTransform(leaf)
    root=support['ROBOT'];base=root+'/base_link';start=tuple(info['mission_start_xyyaw'])
    sx,sy,yaw=start;c,s=math.cos(yaw),math.sin(yaw)
    gx,gy,ga=info['next_cusp_local_xyyaw'];goal=(sx+c*gx-s*gy,sy+s*gx+c*gy,math.atan2(math.sin(yaw+ga),math.cos(yaw+ga)))
    def numeric(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        elif hasattr(value,'numpy'):value=value.numpy()
        result=np.asarray(value,dtype=float).reshape(-1)
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite live readback')
        return result
    sim_view=SimulationManager.get_physics_sim_view()
    if sim_view is None:raise RuntimeError('Existing simulation tensor view missing; no reset attempted')
    live=sim_view.create_articulation_view(base)
    if live.count!=1:raise RuntimeError('Need one existing articulation')
    names=list(live.shared_metatype.dof_names);q=numeric(live.get_dof_positions());tr=numeric(live.get_root_transforms())
    velocity=numeric(live.get_root_velocities());expected=info['retained_all_joint_positions_rad']
    if set(names)!=set(expected) or len(q)!=12 or len(tr)!=7 or len(velocity)!=6:raise RuntimeError('Unexpected retained body/joint state')
    x,y,z,w=tr[3:];actual_yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));saved=info['initial_world_xyyaw']
    if (math.hypot(tr[0]-saved[0],tr[1]-saved[1])>.001
            or abs(math.atan2(math.sin(actual_yaw-saved[2]),math.cos(actual_yaw-saved[2])))>math.radians(.1)
            or max(abs(value-expected[name]) for name,value in zip(names,q))>.001 or math.hypot(*velocity[:2])>.005):
        raise RuntimeError('Live reverse-PASS pose/12 joints/speed no longer match. Keep PAUSED; never reset to force a match')
    h['validate_live'](info,[float(tr[0]),float(tr[1]),actual_yaw],dict(zip(names,map(float,q))),velocity)
    if collision(occupied,(float(tr[0]),float(tr[1]),actual_yaw),origin,resolution,.010):
        raise RuntimeError('Actual retained reverse-PASS pose violates the strict +10mm envelope; no Play')
    reference=h['ForwardReference'](info)
    for p in reference.poses:
        if collision(occupied,(sx+c*p[0]-s*p[1],sy+s*p[0]+c*p[1],yaw+p[2]),origin,resolution,.010):
            raise RuntimeError('Frozen next forward geometry not clear')
    output=h['OUTPUT'];output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise RuntimeError('One prospective next-forward attempt per cohort; no retry or overwrite')
    run_id='hospital_h3b_next_forward_r01';summary_path=output/(run_id+'.json');trace_path=output/(run_id+'.csv')
    clock={'time_s':0.};counts=Counter();pairs=Counter();errors=[]
    def physics_step(dt):clock['time_s']+=float(dt)
    def contacts(headers,data):
        del data
        try:
            for header in headers:
                pair=tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
                category=h['classify_contact'](pair,prep['support_floor'],prep['disabled_legacy_floor_colliders'])
                if category!='unrelated':counts[category]+=1;pairs[(pair,category)]+=1
        except Exception as e:errors.append(str(e))
    def angles(quaternion):
        w,x,y,z=map(float,quaternion)
        return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))

    async def run():
        report=dict(schema_version=1,version=h['VERSION'],trial_kind=h['ROLE'],run_id=run_id,verdict='FAIL',
            execution_contract=info['intervention_contract'], next_forward_reference_adapter_NEW=True,
            inherited_course_feedback_step_and_gains_UNCHANGED=True,
            no_terminal_reapproach_or_reverse_command=True,
            original_first_forward_and_first_reverse_PASS_preserved=True,previous_FAILs_preserved=True,
            model_screens_are_NOT_PhysX_success=True,
            environment_condition=support['CONDITION'],H3B_policy=dict(h['POLICY']),
            component_mode='coordinated_counterphase',live_receipt=info['live_receipt'],live_receipt_sha256=info['live_receipt_sha256'],
            prior_reverse_result=info['reverse_result'],prior_reverse_result_sha256=info['reverse_result_sha256'],
            suffix_candidate=info['suffix_candidate'],suffix_candidate_sha256=info['suffix_candidate_sha256'],
            model_review=info['model_review'],model_review_sha256=info['model_review_sha256'],
            executor_admission=info['admission'],executor_admission_sha256=info['admission_sha256'],
            frozen_input_sha256=info['input_sha256'],
            geometry_manifest=str(manifest_path),geometry_manifest_sha256=support['sha'](manifest_path),
            physics_preparation_snapshot=prep,static_reference=str(static_path),
            static_reference_sha256=support['sha'](static_path),static_trace_sha256=support['sha'](static_path.with_suffix('.csv')),
            original_mission_start_xyyaw=list(start),mission_goal_world_xyyaw_UNCHANGED=info['mission_goal_xyyaw'],
            strict_map_digest=info['strict_map_digest'],
            initial_saved_world_xyyaw=saved,target_world_xyyaw=list(goal),fixed_next_cusp_local_xyyaw=info['next_cusp_local_xyyaw'],
            nominal_reference_forward_length_m=info['reference_length_m'],
            target_is_intermediate_cusp_NOT_full_mission_goal=True,expected_retained_joint_positions_rad=expected,
            NEW_reference_component_not_unchanged_frozen_four_method_comparison=True,
            terminal_pulses_used=False,terminal_reapproaches_used=0,
            body_pose_reset_during_run=False,caster_state_reset_during_run=False,material_modified_during_run=False,
            no_gain_damping_torque_or_stop_criterion_relaxation=True,source_USD_saved=False,
            PhysX_executed=False,full_H3B_turning_path_verified=False,full_suffix_execution_verified=False,
            next_front_only_reverse_verified=False,Nav2_executed=False,CICS_input_used=False,
            use_for_final_statistics=False,fixed_hold_attempt_count=0,
            source_sha256={p.name:support['sha'](p) for p in [_ijcas_path(workspace, 'isaac/scripts/isaac_run_ijcas_hospital_h3b_next_forward_pilot.py'),
                *[scripts/name for name in ('hospital_h3b_next_forward.py','review_hospital_h3b_post_reverse.py',
                    'hospital_h3b_course_tracking.py','hospital_h3b_first_leg_feedback_review.py',
                    'hospital_h3b_policy.py','hospital_drive_hold.py','hospital_path_execution.py',
                    'hospital_h3_flush_crossing.py','hospital_h3_flush_support.py','hospital_private_loader.py')]]})
        robot=motor=drive_indices=old_kp=old_kd=old_limits=old_targets=None
        hold_targets=None;held=False;rows=[];radii={};contact_views=[];pose_views={};history=deque()
        previous_pose=tuple(saved);previous_time=None;phase='pre_hold';last_watch=-float('inf');distance=0.;gate_events=[]
        physics_sub=contact_sub=None;steer_command=(0.,0.);last_command_time=0.;requested_curvature=0.;wheel_command=np.zeros(2)
        feedback_curvature=0.;app=omni.kit.app.get_app();wall_start=time.perf_counter();guard_reference=h['ForwardReference'](info)
        try:
            physics_sub=omni.physx.get_physx_interface().subscribe_physics_step_events(physics_step)
            contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
            robot=SingleArticulation(base);robot.initialize();indices={name:i for i,name in enumerate(robot.dof_names)}
            if set(indices)!=set(expected):raise RuntimeError('Articulation handle changed retained joint set')
            steering=('base_link_to_front_active_steering','base_link_to_rear_active_steering')
            drives=('front_active_steering_to_drive_wheel','rear_active_steering_to_drive_wheel')
            steer_indices=np.asarray([indices[name] for name in steering],dtype=np.int32)
            drive_indices=np.asarray([indices[name] for name in drives],dtype=np.int32)
            motor=robot.get_articulation_controller();kp,kd=motor.get_gains();old_kp=np.asarray(kp).copy();old_kd=np.asarray(kd).copy()
            old_limits=np.asarray(motor.get_max_efforts()).copy();old_targets=np.asarray(robot.get_applied_action().joint_positions)[drive_indices].copy()
            if np.max(abs(old_kp[drive_indices]))>1e-6 or not np.isfinite(old_targets).all():raise RuntimeError('Original velocity-mode drive state required')
            cache=UsdGeom.XformCache()
            for i,name in enumerate(h['WHEELS']):
                body_path=root+'/'+name;prim=stage.GetPrimAtPath(body_path)
                view=RigidContactView(prim_paths_expr=body_path,filter_paths_expr=[],name=run_id+'_force_'+str(i),
                    prepare_contact_sensors=False,disable_stablization=False);view.initialize();contact_views.append(view)
                pose_view=RigidPrim(prim_paths_expr=body_path,name=run_id+'_pose_'+str(i),reset_xform_properties=False,
                    prepare_contact_sensors=False,disable_stablization=False);pose_view.initialize();pose_views[name]=pose_view
                if not pose_view.is_physics_handle_valid():raise RuntimeError('Need live wheel PhysX poses')
                colliders=[p for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
                if len(colliders)!=1 or not colliders[0].IsA(UsdGeom.Capsule):raise RuntimeError('Unchanged wheel capsule required')
                shape=UsdGeom.Capsule(colliders[0]);local=cache.GetLocalToWorldTransform(colliders[0])*cache.GetLocalToWorldTransform(prim).GetInverse()
                scale=max(local.TransformDir(axis).GetLength() for axis in (Gf.Vec3d(1,0,0),Gf.Vec3d(0,1,0),Gf.Vec3d(0,0,1)))
                radii[name]=(float(shape.GetRadiusAttr().Get())+float(shape.GetHeightAttr().Get())/2)*scale+local.ExtractTranslation().GetLength()
            sim=omni.physx.get_physx_simulation_interface();stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            encoded=PhysicsSchemaTools.sdfPathToInt(base)
            def verify_gains(expected_kp):
                kp,kd=motor.get_gains()
                if not (np.allclose(kp,expected_kp,rtol=1e-6,atol=1e-6) and np.array_equal(kd,old_kd)
                        and np.array_equal(np.asarray(motor.get_max_efforts()),old_limits)):
                    raise RuntimeError('Original motor/passive gain, damping or torque-limit guard')
            def measure():
                position,quat=robot.get_world_pose();q=np.asarray(robot.get_joint_positions(),float)
                qd=np.asarray(robot.get_joint_velocities(),float);linear=np.asarray(robot.get_linear_velocity(),float)
                if not all(np.isfinite(v).all() for v in (position,quat,q,qd,linear)):raise RuntimeError('Nonfinite physical telemetry')
                roll,pitch,yaw=angles(quat)
                return (float(position[0]),float(position[1]),yaw),position,q,qd,linear,roll,pitch
            def set_hold(enabled):
                nonlocal held,hold_targets
                if enabled==held:return
                if enabled:
                    actual=np.asarray(robot.get_joint_positions(),float)[steer_indices];linear=np.asarray(robot.get_linear_velocity(),float)
                    if max(abs(v) for v in steer_command)>1e-7 or max(abs(actual))>math.radians(2.5) or math.hypot(*linear[:2])>.005:
                        raise RuntimeError('Actual neutral steering/5mm_s stop required BEFORE captured hold')
                    hold_targets=np.asarray(robot.get_joint_positions(),float)[drive_indices].copy()
                    if not np.isfinite(hold_targets).all():raise RuntimeError('Nonfinite measured hold targets')
                    robot.apply_action(ArticulationAction(joint_positions=hold_targets.astype(np.float32),joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
                    new_kp,new_kd=hold(old_kp,old_kd,drive_indices)
                else:new_kp,new_kd=old_kp.copy(),old_kd.copy()
                with Usd.EditContext(stage,stage.GetSessionLayer()):motor.set_gains(kps=new_kp,kds=new_kd,save_to_usd=False)
                verify_gains(new_kp);held=bool(enabled)
            def command(speed):
                nonlocal steer_command,last_command_time,requested_curvature,wheel_command
                if not math.isfinite(speed) or not 0.<=speed<=.050+1e-9 or (held and abs(speed)>1e-7):
                    raise RuntimeError('Only bounded next forward/zero hold command permitted')
                now=clock['time_s'];dt=now-last_command_time
                if not 0<=dt<=.25:raise RuntimeError('Command control-gap guard')
                requested_curvature=0. if held else feedback_curvature
                actual=np.asarray(robot.get_joint_positions(),float)[steer_indices]
                steer_command,wheel_command=h['direct_joint_targets'](speed,'coordinated_counterphase',requested_curvature,steer_command,dt,actual)
                if held and max(abs(v) for v in steer_command)>1e-7:raise RuntimeError('No nonzero commanded steering while held')
                last_command_time=now
                robot.apply_action(ArticulationAction(joint_positions=np.asarray(steer_command,dtype=np.float32),joint_indices=steer_indices))
                robot.apply_action(ArticulationAction(joint_positions=hold_targets.astype(np.float32) if held else None,
                    joint_velocities=np.asarray(wheel_command,dtype=np.float32),joint_indices=drive_indices))
            async def sample(speed,phase_start):
                nonlocal previous_pose,previous_time,last_watch,distance
                command(speed)
                while True:
                    sim.wake_up(stage_id,encoded);await app.next_update_async()
                    if not timeline.is_playing():raise RuntimeError('Timeline controls changed during next-forward trial')
                    if time.perf_counter()-wall_start>300.:raise TimeoutError('Existing 300s wall-time budget')
                    now=clock['time_s']
                    if previous_time is None or now>previous_time:break
                if now>90. or (previous_time is not None and now-previous_time>.25):raise TimeoutError('90s simulation/control-gap guard')
                pose,position,q,qd,linear,roll,pitch=measure();local=h['relative_pose'](pose,start)
                _,ref,error=guard_reference.project(local);heading=math.degrees(math.atan2(math.sin(local[2]-ref[2]),math.cos(local[2]-ref[2])))
                history.append((now,pose))
                while len(history)>2 and history[1][0]<=now-.20:history.popleft()
                dt=now-history[0][0];filtered=math.hypot(pose[0]-history[0][1][0],pose[1]-history[0][1][1])/dt if dt>0 else math.hypot(*linear[:2])
                row=dict(time_s=now,phase=phase,phase_time_s=now-phase_start,x_m=pose[0],y_m=pose[1],z_m=float(position[2]),yaw_rad=pose[2],
                    roll_deg=math.degrees(roll),pitch_deg=math.degrees(pitch),longitudinal_m=local[0],lateral_m=local[1],yaw_error_deg=math.degrees(local[2]),
                    vx_physx_m_s=float(linear[0]),vy_physx_m_s=float(linear[1]),vz_physx_m_s=float(linear[2]),pose_filtered_speed_m_s=filtered,
                    command_speed_m_s=float(speed),position_hold_enabled=held,front_rad=float(q[steer_indices[0]]),rear_rad=float(q[steer_indices[1]]),
                    command_front_rad=float(steer_command[0]),command_rear_rad=float(steer_command[1]),requested_curvature_1_m=float(requested_curvature),
                    command_time_s=float(last_command_time),command_mode='coordinated_counterphase',reference_progress_m=guard_reference.progress,
                    reference_position_error_m=error,reference_heading_error_deg=heading)
                for name,index in indices.items():row['joint_'+name+'_rad']=float(q[index])
                action=robot.get_applied_action();targets=np.asarray(action.joint_velocities)[drive_indices];positions=np.asarray(action.joint_positions)
                for i,axle in enumerate(('front','rear')):
                    row['drive_'+axle+'_rad']=float(q[drive_indices[i]]);row['drive_'+axle+'_velocity_rad_s']=float(qd[drive_indices[i]])
                    row['drive_'+axle+'_actual_velocity_target_rad_s']=float(targets[i])
                for i,name in enumerate(h['WHEELS']):
                    pv=pose_views[name]
                    if not pv.is_physics_handle_valid():raise RuntimeError('Lost live wheel pose handle')
                    wp=np.asarray(pv.get_world_poses()[0],float).reshape(-1,3)[0]
                    force=float(np.asarray(contact_views[i].get_net_contact_forces(dt=1/120.),float).reshape(-1,3)[0,2])
                    if not np.isfinite(wp).all() or not math.isfinite(force):raise RuntimeError('Nonfinite wheel pose/load')
                    for axis,value in zip(('x','y','z'),wp):row['wheel_'+name+'_'+axis+'_m']=float(value)
                    row[f'wheel_{i}_fz_n']=force
                rows.append(row)
                try:
                    verify_gains(hold(old_kp,old_kd,drive_indices)[0] if held else old_kp)
                    if not np.allclose(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(leaf)),np.asarray(leaf_matrix),rtol=0,atol=2e-5):raise RuntimeError('Fully-open door moved')
                    if stage.GetPrimAtPath(support['THRESHOLD']).IsActive():raise RuntimeError('Selected flush sill became active')
                    if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(prep['support_floor'])).GetCollisionEnabledAttr().Get() is False:raise RuntimeError('Canonical support disabled')
                    if any(UsdPhysics.CollisionAPI(stage.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get() is not False for p in prep['disabled_legacy_floor_colliders']):raise RuntimeError('Duplicate support became enabled')
                    if not np.allclose(targets,wheel_command,rtol=0,atol=1e-6) or not np.allclose(positions[steer_indices],steer_command,rtol=0,atol=1e-6):raise RuntimeError('Applied target acknowledgement differs')
                    if held and not np.allclose(positions[drive_indices],hold_targets,rtol=0,atol=1e-6):raise RuntimeError('Captured targets changed')
                    if held and max(abs(row[a+'_rad']) for a in ('front','rear'))>math.radians(2.5):raise RuntimeError('Lost actual neutral steering while held')
                    if errors or any(counts[k] for k in h['BAD_CONTACTS']):raise RuntimeError('Unexpected obstacle/nonwheel/legacy contact')
                    if max(abs(roll),abs(pitch))>math.radians(2) or math.hypot(*linear[:2])>.080:raise RuntimeError('Existing body tilt/speed guard')
                    if error>.080 or abs(heading)>10.:raise RuntimeError('Existing spatial-reference 80mm/10deg guard')
                    turn=math.atan2(math.sin(pose[2]-previous_pose[2]),math.cos(pose[2]-previous_pose[2]))
                    change=math.hypot(pose[0]-previous_pose[0],pose[1]-previous_pose[1]);distance+=change
                    if distance>info['intervention_contract']['maximum_measured_center_distance_m']:raise RuntimeError('Supplemental next-forward measured-distance guard')
                    n=max(1,math.ceil(change/.005),math.ceil(abs(turn)/math.radians(.25)))
                    for j in range(1,n+1):
                        a=j/n;p=(previous_pose[0]+a*(pose[0]-previous_pose[0]),previous_pose[1]+a*(pose[1]-previous_pose[1]),previous_pose[2]+a*turn)
                        if collision(occupied,p,origin,resolution,.010):raise RuntimeError('Actual swept full-envelope STRICT flush raster guard')
                    if h['motion_stalled'](rows):raise RuntimeError('Five-second measured forward-progress stall')
                except Exception as e:
                    report['safety_abort']=dict(reason=str(e),actual_state=dict(row));raise
                previous_pose,previous_time=pose,now
                if time.perf_counter()-last_watch>=15.:
                    print(f"[H3-B NEXT FORWARD {run_id}] wall={time.perf_counter()-wall_start:.1f}s sim={now:.1f}s phase={phase}; progress={guard_reference.progress:.3f}/{reference.arc[-1]:.3f}m command/observed={1000*speed:.2f}/{1000*filtered:.2f}mm/s reference={error*1000:.2f}mm/{heading:.3f}deg; obstacle callbacks={counts['hospital_obstacle']}")
                    last_watch=time.perf_counter()
                return row,pose,linear

            pose,_,q,_,linear,_,_=measure()
            if (math.hypot(pose[0]-saved[0],pose[1]-saved[1])>.001 or max(abs(q[indices[name]]-expected[name]) for name in indices)>.001
                    or abs(math.atan2(math.sin(pose[2]-saved[2]),math.cos(pose[2]-saved[2])))>math.radians(.1)):
                raise RuntimeError('Initialization changed saved body/12 joint state; no Play authorized')
            report['observed_prehold_world_xyyaw']=list(pose);report['initial_paused_state_continuity_verified']=True
            report['initial_measured_caster_joint_angles_rad']={name:float(q[index]) for name,index in indices.items() if name.endswith('_caster')}
            set_hold(True);command(0.);timeline.play();report['PhysX_executed']=True;pre_start=clock['time_s']
            print('H3-B next forward: 2s captured-wheel neutral pre-hold at ACTUAL reverse PASS; no reset')
            while clock['time_s']-pre_start<2.:row,pose,linear=await sample(0.,pre_start)
            pre=report['observed_prehold_world_xyyaw']
            if (any(math.hypot(r['x_m']-pre[0],r['y_m']-pre[1])>.003 for r in rows)
                    or abs(math.atan2(math.sin(pose[2]-saved[2]),math.cos(pose[2]-saved[2])))>math.radians(.5)
                    or max(row['pose_filtered_speed_m_s'],math.hypot(*linear[:2]))>.005):
                raise RuntimeError('Captured pre-hold 3mm / 0.5deg / 5mm_s release readiness failed')
            phase='entry_gate';gate_start=clock['time_s'];gate=h['StopAlignGate'](1,'coordinated_counterphase',0.,gate_start);last_stage=None
            while True:
                measured=max(row['pose_filtered_speed_m_s'],math.hypot(*linear[:2]))
                decision=gate.step(clock['time_s'],0.,measured,(row['front_rad'],row['rear_rad']),held)
                if decision['stage']!=last_stage:print('H3-B actual forward handoff:',decision['stage']);last_stage=decision['stage']
                if decision['ready']:
                    gate_events.append(dict(direction=1,captured_hold_acknowledged=gate.saw_hold,hold_release_acknowledged=not held,
                        command_speed_m_s=0.,measured_speed_m_s=measured,
                        steering_error_deg=max(abs(math.degrees(row[a+'_rad'])) for a in ('front','rear')),time_s=clock['time_s']))
                    break
                if decision['stage'] in ('settling','alignment'):set_hold(True)
                elif decision['stage']=='release_wait':set_hold(False)
                row,pose,linear=await sample(0.,gate_start)
            if held:raise RuntimeError('Measured hold release not acknowledged')
            report['release_world_xyyaw']=list(pose);motion_start=clock['time_s'];controller=h['ForwardProgress'](info,motion_start)
            print('H3-B NEXT forward ONLY: fixed 0.569m reference + SAME screened course law/gains; <=50mm/s; no terminal pulse or reverse')
            while True:
                out=controller.step(clock['time_s'],h['relative_pose'](pose,start),
                    max(row['pose_filtered_speed_m_s'],math.hypot(*linear[:2])),(row['front_rad'],row['rear_rad']),
                    h['body_velocity']((float(linear[0]),float(linear[1])),pose[2]))
                feedback_curvature=out['curvature_1_m'];speed=out['speed_m_s']
                if out['ready']:
                    if not all(h['arrival_checks'](h['relative_pose'](pose,start),info['next_cusp_local_xyyaw'],
                            max(row['pose_filtered_speed_m_s'],math.hypot(*linear[:2])),(row['front_rad'],row['rear_rad'])).values()):
                        raise RuntimeError('Measured fixed-cusp 30mm/2deg/5mm_s/neutral gate required BEFORE one final hold; no endpoint recapture')
                    report['terminal_arrival_before_ONE_hold']=True
                    break
                phase='forward_motion' if out['phase']=='tracking' else 'forward_braking'
                row,pose,linear=await sample(speed,motion_start)
            if abs(speed)>1e-7 or max(row['pose_filtered_speed_m_s'],math.hypot(*linear[:2]))>.005:raise RuntimeError('Actual stop required before final hold')
            feedback_curvature=0.;set_hold(True);report['fixed_hold_attempt_count']=1;phase='terminal_hold';hold_start=clock['time_s']
            print('H3-B next forward: ONE fixed 5s hold; fixed final 3s evaluation, not a PASS yet')
            while clock['time_s']-hold_start<5.:row,pose,linear=await sample(0.,hold_start)
            bound=inventory['explicitly_suppressed_assets'][0]['collision']['bound']
            report.update(h['evaluate_component'](rows,goal,static_rows,prep['simulation_mass_kg'],bound,radii,counts,errors,1,gate_events,info['reference_length_m']))
            report['motion_and_braking_time_s']=hold_start-motion_start
        except (Exception,asyncio.CancelledError) as e:
            report['error']=f'{type(e).__name__}: {e}';report.setdefault('checks',{})['next_forward_and_hold_completed']=False;report['verdict']='FAIL'
        finally:
            if robot is not None and drive_indices is not None:
                try:robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
                except Exception as e:report['zero_command_error']=str(e);report.setdefault('checks',{})['final_zero_command_sent']=False
            if timeline.is_playing():timeline.pause()
            if motor is not None and old_kp is not None:
                try:
                    with Usd.EditContext(stage,stage.GetSessionLayer()):motor.set_gains(kps=old_kp,kds=old_kd,save_to_usd=False)
                    kp,kd=motor.get_gains();report['original_motor_gains_restored']=bool(np.allclose(kp,old_kp,rtol=1e-6,atol=1e-6)
                        and np.array_equal(kd,old_kd) and np.array_equal(np.asarray(motor.get_max_efforts()),old_limits))
                    if old_targets is not None and np.isfinite(old_targets).all():robot.apply_action(ArticulationAction(joint_positions=old_targets.astype(np.float32),joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive_indices))
                except Exception as e:report['restore_error']=str(e);report['original_motor_gains_restored']=False
            report.setdefault('checks',{})['original_motor_gains_restored']=report.get('original_motor_gains_restored',False)
            try:
                support['verify_files'](info['input_sha256'])
                support['live_geometry_check'](stage,manifest,geometry,inventory,set(prep['disabled_legacy_floor_colliders']))
                report['checks']['same_authored_support_contact_and_motor_baseline']=support['support_readback'](stage,manifest)==prep['live_support_readback']
                report['checks']['original_PASS_and_frozen_inputs_preserved']=True
            except Exception as e:report['cleanup_environment_error']=str(e);report['checks']['original_PASS_and_frozen_inputs_preserved']=False
            if errors or any(counts[k] for k in h['BAD_CONTACTS']):report['checks']['no_obstacle_nonwheel_or_legacy_contacts']=False
            if not all(report['checks'].values()):report['verdict']='FAIL'
            physics_sub=contact_sub=None
            report.update(next_forward_component_verified=report['verdict']=='PASS',next_front_only_reverse_verified=False,full_suffix_execution_verified=False,full_H3B_turning_path_verified=False,
                sample_count=len(rows),contact_report_counts_by_category=dict(counts),contact_callback_errors=errors,
                contact_pairs_by_category=[dict(pair=pair,category=category,callback_count=count) for (pair,category),count in sorted(pairs.items())],
                direction_gate_events=gate_events,measured_center_distance_m=distance,wall_time_s=time.perf_counter()-wall_start)
            if rows:
                report['last_recorded_state']=dict(rows[-1]);report['wheel_conservative_sphere_radii_m']=radii
                with trace_path.open('x',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with summary_path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                marker.SetCustomDataByKey('status','h3b_next_forward_pilot_finished_wait_for_review')
                marker.SetCustomDataByKey('last_h3b_next_forward_pilot_gate',str(summary_path))
            print('=== H3-B RETAINED-STATE NEXT FORWARD COMPONENT GATE ===')
            print('run:',run_id);print('contact categories:',dict(counts));print('measured handoff gates:',len(gate_events))
            if 'terminal_position_error_m' in report:
                print('terminal error mm/deg:',round(1000*report['terminal_position_error_m'],3),round(report['terminal_heading_error_deg'],6))
                print('hold regression speed mm/s:',round(1000*report['fixed_terminal_hold_analysis']['pose_regression_speed_m_s'],3))
            print('original motor gains restored:',report.get('original_motor_gains_restored',False))
            print('failed checks:',[k for k,v in report['checks'].items() if not v])
            for key in ('error','restore_error','cleanup_environment_error'):
                if key in report:print(key+':',report[key])
            print('VERDICT:',report['verdict']);print('saved:',summary_path)
            print('Timeline PAUSED. NEXT forward component ONLY; second reverse/full H3/suffix/robustness/CICS/Nav2 remain unvalidated.')
            print('Keep PAUSED; no reset/reopen/Play/save or old controller. Review the actual endpoint before any next leg.')

    task=asyncio.ensure_future(run())
    print('H3-B retained-state NEXT FORWARD scheduled:',run_id)
    print('Automatic 2s neutral pre-hold -> measured STOP/HOLD/neutral/RELEASE -> next counterphase FORWARD ~0.569m <=50mm/s -> ONE 5s hold -> Pause/restore.')
    print('Retained actual reverse-PASS body/12 joints/caster history. Same course law/1.5+2 gains, 8deg/s steering, 5mm/s stop, strict +10mm envelope and 30mm/2deg terminal.')
    print('15s heartbeat; bounded 90s simulation/300s wall. No second reverse, CICS/Nav2, reset, terminal pulse or automatic retry. Do not touch timeline controls.')
    return task


ijcas_hospital_h3b_next_forward_pilot_task=_schedule_next_forward_pilot()

