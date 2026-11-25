"""Pure source/frame tests, not ROS/Isaac sensor or navigation trials."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import hospital_sensor_navigation as sensor


@pytest.mark.parametrize('yaw',[0.,.8,-math.pi/2])
def test_H5_replan_twist_transports_actual_CoM_velocity_to_base_origin(yaw):
    offset=np.array([.2,-.1,.05]);omega=np.array([0.,0.,.03])
    rotation=sensor.rotation_wxyz([math.cos(yaw/2),0.,0.,math.sin(yaw/2)])
    wanted=np.array([.002,.001,0.])
    velocity=rotation@wanted+np.cross(omega,rotation@offset)
    snapshot=dict(transform=[1.,2.,0.,0.,0.,math.sin(yaw/2),math.cos(yaw/2)],velocity=[*velocity,*omega])
    assert sensor.base_origin_body_twist(snapshot,offset)==pytest.approx([.002,.001,.03],abs=1e-14)
    assert np.linalg.norm(velocity[:2])>.005  # Raw CoM speed is NOT origin speed.


def test_H5_discovery_brakes_before_one_actual_quiet_stop_capture_and_keeps_steering():
    gate=sensor.H5DiscoveryStop(dict(speed_m_s=.05,mode='rear_only',curvature_1_m=.25),0.)
    previous=.05;ack=False;captured=0
    for i in range(1,801):
        t=i*.01;result=gate.step(t,previous,ack)
        assert abs(result['speed_m_s']-previous)<=.020*.01+1e-12
        assert result['freeze_steering_targets'] and result['mode']=='rear_only' and result['curvature_1_m']==.25
        if result['drive_hold_requested'] and not ack:
            captured+=1;assert t>=2.75-1e-10;assert previous<=.003
        ack=result['drive_hold_requested'];previous=result['speed_m_s']
        if gate.ready:break
    assert gate.ready and captured==1


@pytest.mark.parametrize('fault',['unexpected_ack','over_budget','NaN_speed','regressed_stamp','gap'])
def test_H5_discovery_stop_does_not_waive_bad_measurements_or_ACK(fault):
    gate=sensor.H5DiscoveryStop(dict(speed_m_s=.05,mode='coordinated_counterphase',curvature_1_m=0.),0.)
    args=dict(timestamp=.01,speed=.05,ack=False)
    if fault=='unexpected_ack':args['ack']=True
    elif fault=='over_budget':args['timestamp']=8.1
    elif fault=='NaN_speed':args['speed']=float('nan')
    elif fault=='regressed_stamp':args['timestamp']=-.01
    elif fault=='gap':args['timestamp']=.26
    with pytest.raises(RuntimeError):gate.step(**args)


def test_H5_two_Play_scope_is_exactly_bounded_and_not_a_generic_retry_exception():
    packet=dict(stopped_replanning=copy.deepcopy(sensor.H5_REPLAN_SCOPE),scenario='H5_obstacle_bypass_return',
        method='P_dynamics_full_envelope',maximum_Play_calls=2,simulation_budget_s=240.,maximum_wall_time_s=900.,
        maximum_Stop_reset_restaging_calls=0,automatic_retry=False)
    assert sensor.checked_h5_replanning_scope(packet)==sensor.H5_REPLAN_SCOPE
    for key,value in [('scenario','H6_integrated_transfer_mission'),('method','B0_fixed_counterphase_ramp'),
            ('maximum_Play_calls',3),('maximum_wall_time_s',1500.),('automatic_retry',True)]:
        changed=copy.deepcopy(packet);changed[key]=value
        with pytest.raises(ValueError):sensor.checked_h5_replanning_scope(changed)


def test_H5_replanning_uses_exact_original_common_controller_approval_and_separate_current_scope():
    import hospital_mission_sequence as seq
    folder=sensor.GATE/'hospital_h5_experiments/cohort_06_held_static_r01/route/P_dynamics_full_envelope'
    if not (folder/'replanning_sdk_binding.json').exists():pytest.skip('Saved unstarted preflight diagnostic only')
    old=json.loads((folder/'replanning_sdk_binding.json').read_bytes())
    source=json.loads((folder/'route_protocol.json').read_bytes())
    with pytest.raises(ValueError,match='same-limits retained-final-target'):seq.checked_route_lifecycle_contract(old)
    reviewed=copy.deepcopy(old);reviewed['explicit_route_approval']=source['explicit_route_approval']
    assert sensor.checked_h5_replanning_scope(reviewed)==sensor.H5_REPLAN_SCOPE
    contract=seq.checked_route_lifecycle_contract(reviewed)
    # A newer offline archive helper has its own source epoch; all controller
    # values must remain identical, but do not require the old code hash.
    current=contract['original_motion_contract'].pop('candidate_source_sha256')
    assert current==sensor.sha(seq.__file__)
    saved=copy.deepcopy(source['common_lifecycle_contract'])
    saved['original_motion_contract'].pop('candidate_source_sha256')
    assert contract==saved
    assert not (folder/'single_route_reserved.json').exists()


def test_H5_continuation_additions_leave_entire_nominal_controller_AST_exact():
    import hospital_mission_lifecycle as lifecycle
    import zipfile
    archive=sensor.GATE/'hospital_h6_experiments/cohort_20_held_static_r01/route/P_dynamics_full_envelope/route_sources.zip'
    with zipfile.ZipFile(archive) as z:before=z.read('studies/IJCAS/scripts/hospital_mission_execution.py')
    after=Path(lifecycle.motion.__file__).read_bytes()
    assert lifecycle.reviewed_mission_schema_extension(before,after)['whole_original_controller_and_all_guards_AST_identical']
    with pytest.raises(ValueError):lifecycle.reviewed_mission_schema_extension(before,after.replace(b"self.speed=0.",b"self.speed=.05",1))


def test_H5_saved_replanning_HOLD_archive_keeps_original_outcome_and_own_source_packet():
    import hospital_mission_sequence as seq
    result=sensor.GATE/'hospital_h5_experiments/cohort_06_held_static_r01/route/P_dynamics_full_envelope/first_mission_after_preflight_review_r01/route_readback.json'
    if not result.exists():pytest.skip('Saved actual H5 HOLD needed for archive regression')
    before=result.read_bytes();review=seq.saved_route_boundary_review(result)
    assert review['original_verdict']=='HOLD' and review['new_physical_trials']==0
    assert review['original_HOLD_FAIL_never_promoted'] and not review['actual_route_admission']
    assert result.read_bytes()==before


@pytest.mark.parametrize('key,value',[
    ('actual_Play_calls',3),('actual_Play_calls',True),('new_replanning_requests',2),
    ('additional_Stop_calls',1),('automatic_retry',True),('method','B0_fixed_counterphase_ramp')])
def test_H5_saved_replanning_archive_rejects_wrong_scope_counts_or_method(monkeypatch,key,value):
    import hospital_mission_sequence as seq
    result=sensor.GATE/'hospital_h5_experiments/cohort_06_held_static_r01/route/P_dynamics_full_envelope/first_mission_after_preflight_review_r01/route_readback.json'
    if not result.exists():pytest.skip('Saved actual H5 HOLD needed for archive regression')
    original=seq.retained.read;raw=copy.deepcopy(original(result));raw[key]=value
    monkeypatch.setattr(seq.retained,'read',lambda p:raw if Path(p).resolve()==result.resolve() else original(p))
    with pytest.raises(ValueError,match='bounded H5'):seq.saved_route_boundary_review(result)


@pytest.mark.parametrize('dt',[.020,.010])
def test_H5_retained_nonzero_steering_first_model_stamp_is_zero_interval_not_rounded_literal2(dt):
    from audit_hospital_paper_readiness import unified_mission_model_binding
    folder=sensor.GATE/'hospital_h5_experiments/cohort_06_held_static_r01/route/P_dynamics_full_envelope/first_mission_after_preflight_review_r01'
    if not (folder/'resumed_execution_plan.json').exists():pytest.skip('Saved measured H5 boundary needed for regression')
    plan=json.loads((folder/'resumed_execution_plan.json').read_bytes())
    geometry=json.loads((folder/'resumed_execution_geometry.json').read_bytes())
    run,execution,design=unified_mission_model_binding(plan['geometry_profile'],geometry,plan,simulation_budget_s=240.)
    original=execution.direct_joint_targets;calls=[]
    def observed(*args,**kwargs):
        result=original(*args,**kwargs);previous,interval=args[3],args[4]
        calls.append((interval,tuple(previous),tuple(result[0])))
        return result
    execution.direct_joint_targets=observed
    with np.load(geometry['occupancy_grid']) as grid:
        report,_=run(plan,plan['rigid_body_parameter_bounds'],dt,grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m']),design)
    assert calls[0][0]==0. and calls[0][1]==calls[0][2]
    assert abs(calls[0][1][1])>math.radians(10.)
    assert all(max(abs(a-b)/t for a,b in zip(before,after))<=math.radians(8.)+1e-9
        for t,before,after in calls if t>0.)
    assert report['model_status']=='MODEL_SCREEN_PASS'
    assert all(report['checks'].values())


def test_H5_actual_release_replays_held_targets_not_unreached_curvature_intention():
    import hospital_mission_sequence as seq
    from audit_hospital_paper_readiness import unified_wheelbase_connections
    folder=sensor.GATE/'hospital_h5_experiments/cohort_07_held_static_r01/route/P_dynamics_full_envelope'
    if not (folder/'route_readback.json').exists():pytest.skip('Saved actual release HOLD needed')
    raw=json.loads((folder/'route_readback.json').read_bytes());capture=raw['measured_capture']
    plan=json.loads((folder/'resumed_execution_plan.json').read_bytes())
    geometry=json.loads((folder/'resumed_execution_geometry.json').read_bytes())
    angles=unified_wheelbase_connections(plan['geometry_profile']).steering_angles
    measured=capture['front_rear_steering_rad'];held=capture['captured_steering_targets_rad']
    assert max(abs(a-b) for a,b in zip(measured,angles(capture['retained_curvature_1_m'],capture['retained_mode'])))>math.radians(6.)
    assert max(abs(a-b) for a,b in zip(measured,held))<math.radians(2.5)
    controller=seq.unified_geometry_lifecycle_controller(plan,geometry,profile=plan['geometry_profile'],simulation_budget_s=240.)
    controller.lifecycle_stage='route';controller.previous=capture['time_s']
    assert angles(controller.curvature,controller.mode)==pytest.approx(held,abs=1e-14)
    class Actuator:
        authorized=True;last_command_stamp=capture['time_s'];steering=np.array([0,1]);ack=True
        original=dict(positions=np.array(held));commands=[]
        def confirmation(self,stamp):return self.ack
        def measured_steering(self):return measured
        def request_hold(self,requested,stamp,*,command_speed_m_s,measured_speed_m_s):
            assert measured_speed_m_s<.005
            if self.ack:assert command_speed_m_s==0.
            self.ack=requested
        def command(self,steer,wheels,stamp):self.commands.append((list(steer),list(wheels)))
    actuator=Actuator();bridge=seq.RetainFinalTargetsBridge(controller,actuator,guard=lambda *a:None)
    speed=math.hypot(*capture['body_twist_u_v_r'][:2])
    first=bridge.step(capture['time_s']+1/60,capture['pose_world_xyyaw'],speed)
    assert first['speed_m_s']==0. and first['drive_hold_requested'] is False
    assert actuator.commands[0][0]==pytest.approx(held,abs=1e-14)
    second=bridge.step(capture['time_s']+2/60,capture['pose_world_xyyaw'],speed)
    assert controller.stage=='tracking' and second['speed_m_s']==0.
    third=bridge.step(capture['time_s']+3/60,capture['pose_world_xyyaw'],speed)
    assert 0.<third['speed_m_s']<=.020/60+1e-12
    assert raw['verdict']=='HOLD'  # SOFTWARE replay cannot promote the actual attempt.


@pytest.mark.parametrize('mode',['rear_only','front_only','coordinated_counterphase'])
def test_H5_release_command_inverse_rejects_incompatible_retained_mode(mode):
    source=sensor.GATE/'hospital_h5_experiments/cohort_07_held_static_r01/route/P_dynamics_full_envelope/resumed_execution_plan.json'
    if not source.exists():pytest.skip('Saved declared geometry profile needed')
    profile=json.loads(source.read_bytes())['geometry_profile']
    with pytest.raises(ValueError,match='reproduce'):
        sensor.h5_captured_command_curvature(dict(retained_mode=mode,captured_steering_targets_rad=[.1,.2]),profile)


PROTOCOL=sensor.GATE/'hospital_h5_experiments/cohort_01_P_development_r01/runtime_protocol.json'


@pytest.mark.parametrize('tolerance',[.030,.004,0.,float('nan'),True])
def test_H5_endpoint_margin_cannot_relax_original_planning_or_runtime_acceptance(tolerance):
    from audit_hospital_paper_readiness import unified_h3_guided_candidate
    with pytest.raises(ValueError,match='stricter'):
        unified_h3_guided_candidate(None,None,{},None,None,None,None,budget_s=1.,terminal_position_tolerance_m=tolerance)


def test_nominal_guided_planning_keeps_original25mm_endpoint_default():
    import inspect
    from audit_hospital_paper_readiness import unified_h3_guided_candidate
    assert inspect.signature(unified_h3_guided_candidate).parameters['terminal_position_tolerance_m'].default==.025
    failed=unified_h3_guided_candidate(None,None,{},None,None,None,None,budget_s=1.)
    assert failed['plan_success'] is False and failed['status']=='OWN_SPATIAL_SEED_NOT_FOUND_WITHIN_BUDGET'


def test_H5_goal_connected_distance_prevents_50mm_early_braking_without_relaxing25mm_handoff():
    from types import SimpleNamespace
    import hospital_mission_sequence as seq
    goal=(0.,0.,0.);endpoint=(.0225,0.,0.)
    tracker=SimpleNamespace(mission_final_leg=True,reference=SimpleNamespace(poses=[endpoint]))
    nominal=seq._tracking_floor_candidate_implementation(.020,physical_direction_conditioned=True)
    fixed=sensor.h5_goal_distance_tracking(nominal,goal)
    combine=fixed.__globals__['_mission_task_remaining']
    handoff=fixed.__globals__['_mission_handoff']
    pose=(.0474,0.,0.)
    assert handoff(tracker,pose,0.,.0249) is True
    assert combine(tracker,pose,.0249)==pytest.approx(.0474)
    assert handoff(tracker,pose,0.,combine(tracker,pose,.0249)) is False
    pose=(.0249,0.,0.)
    assert handoff(tracker,pose,0.,combine(tracker,pose,.0024)) is True
    tracker.mission_final_leg=False
    assert combine(tracker,(10.,20.,0.),.0024)==.0024
    assert fixed.mission_binding['original_braking_formula_and_speed_acceleration_limits_unchanged']
    assert fixed.mission_binding['additional_remaining_distance_expression_substitutions']==1
    assert 'goal_distance_handoff_policy' not in nominal.mission_binding


def geometry(**kwargs):
    args=dict(base_xyz=(1.,2.,0.),base_wxyz=(1.,0.,0.,0.),
        sensor_xyz_in_base=(0.,0.,.35),sensor_wxyz_in_base=(1.,0.,0.,0.),
        frame_id='hospital_laser',ray_count=4,angle_min=-math.pi,angle_increment=math.pi/2,
        range_min=.10,range_max=12.,rate_hz=5.)
    args.update(kwargs)
    return sensor.scan_geometry(**args)


def hit(ray,distance,path='/World/UNKNOWN_OBSTACLE'):
    point=np.asarray(ray['origin_world_m'])+distance*np.asarray(ray['direction_world'])
    return dict(collision=path,distance_m=distance,position_m=point.tolist(),normal=[0.,0.,1.])


def test_SE2_explicit_inverse_and_right_handed_yaw():
    pose=sensor.transform_pose((10.,20.,math.pi/2),(1.,0.,math.pi/2))
    assert pose[:2]==pytest.approx([10.,21.])
    assert abs(abs(pose[2])-math.pi)<1e-12
    assert sensor.transform_pose((10.,20.,math.pi/2),pose,inverse=True)==pytest.approx([1.,0.,math.pi/2])


@pytest.mark.parametrize('yaw',[0.,math.pi/2,-math.pi/2,math.pi,1.234])
def test_SE2_arbitrary_declared_map_frame_roundtrips_without_estimate(yaw):
    frame=(18.13497872201219,28.050273554946045,yaw);pose=(18.14,22.75,-1.58)
    mapped=sensor.transform_pose(frame,pose,inverse=True)
    assert sensor.transform_pose(frame,mapped)==pytest.approx(pose)


def test_full_circle_has_no_duplicate_endpoint_and_every_ray_has_same_sensor_origin():
    g=geometry(ray_count=360,angle_increment=2*math.pi/360)
    assert g['angle_max_rad']==pytest.approx(math.pi-2*math.pi/360)
    assert len(g['rays'])==360
    assert all(ray['origin_world_m']==[1.,2.,.35] for ray in g['rays'])
    assert np.linalg.norm(np.asarray(g['rays'][0]['direction_world'])-g['rays'][-1]['direction_world'])>0.
    assert g['time_increment_s']==0. and g['scan_time_s']==.2
    assert not g['old_perimeter_blind_zone_used'] and not g['ROS_publisher_or_PhysX_query_started']


def test_right_handed_angles_and_yaw_extrinsic_compose_in_3D():
    q=(math.sqrt(.5),0.,0.,math.sqrt(.5))
    g=geometry(base_wxyz=q,sensor_xyz_in_base=(.10,0.,.35),sensor_wxyz_in_base=q)
    assert g['sensor_origin_world_m']==pytest.approx([1.,2.1,.35])
    # Zero-angle beam rotates 90deg by the mount and another 90deg by the body.
    assert g['rays'][2]['direction_world']==pytest.approx([-1.,0.,0.],abs=1e-12)


def test_body_roll_pitch_and_sensor_height_are_not_flattened_to_world_plane():
    q=(math.sqrt(.5),math.sqrt(.5),0.,0.)
    g=geometry(base_xyz=(1.,2.,-.003),base_wxyz=q)
    assert g['sensor_origin_world_m']==pytest.approx([1.,1.65,-.003])
    assert g['rays'][3]['direction_world']==pytest.approx([0.,0.,1.],abs=1e-12)


@pytest.mark.parametrize('kwargs',[
    dict(base_xyz=(0.,float('nan'),0.)),dict(base_wxyz=(0.,0.,0.,0.)),
    dict(base_wxyz=(2.,0.,0.,0.)),dict(sensor_wxyz_in_base=(1.,0.,0.)),
    dict(frame_id='base_link'),dict(frame_id='/laser'),dict(frame_id=''),
    dict(frame_id='bad frame'),dict(ray_count=True),dict(ray_count=1),
    dict(ray_count=5),dict(angle_increment=0.),dict(range_min=0.),
    dict(range_min=12.),dict(range_max=float('inf')),dict(rate_hz=0.)])
def test_invalid_or_ambiguous_scan_contract_is_rejected_before_any_query(kwargs):
    with pytest.raises((ValueError,TypeError)):geometry(**kwargs)


def test_base_link_is_allowed_only_for_identity_sensor_extrinsics():
    assert geometry(frame_id='base_link',sensor_xyz_in_base=(0.,0.,0.))['frame_id']=='base_link'


def test_all_hits_preserved_and_only_exact_self_shape_filtered():
    ray=geometry()['rays'][2];own='/World/actor/base_link/collisions/mesh'
    raw=[hit(ray,1.5),hit(ray,.2,own),hit(ray,.45,'/World/actor/UNREVIEWED_COLLISION')]
    result=sensor.scan_return(ray,raw,own_collision_paths=[own],range_min=.1)
    assert result['raw_hits']==raw and result['exact_self_hits']==[raw[1]]
    assert result['all_external_hits']==[raw[0],raw[2]]
    assert result['nearest_external_hit']==raw[2] and result['valid_range_m']==.45
    assert result['status']=='VALID_RANGE' and result['publishable_without_silent_near_obstacle_loss']


def test_below_min_external_obstacle_is_never_replaced_by_free_space():
    ray=geometry()['rays'][0]
    result=sensor.scan_return(ray,[hit(ray,.05),hit(ray,1.)],own_collision_paths=['/World/actor/shape'],range_min=.1)
    assert result['status']=='BELOW_MIN_NONFREE' and result['nearest_external_hit']['distance_m']==.05
    assert result['valid_range_m'] is None and result['ROS_no_return_encoding'] is None
    assert not result['publishable_without_silent_near_obstacle_loss']


def test_no_return_has_explicit_JSON_safe_symbol_not_a_nonfinite_number():
    result=sensor.scan_return(geometry()['rays'][0],[],own_collision_paths=['/World/actor/shape'],range_min=.1)
    assert result['status']=='NO_RETURN' and result['ROS_no_return_encoding']=='positive_infinity'
    assert result['valid_range_m'] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('fault',['nan','negative','beyond','position','normal','path','direction'])
def test_bad_native_hit_or_ray_never_becomes_an_empty_free_ray(fault):
    ray=geometry()['rays'][0];raw=hit(ray,1.)
    if fault=='nan':raw['distance_m']=float('nan')
    elif fault=='negative':raw['distance_m']=-1.
    elif fault=='beyond':raw=hit(ray,13.)
    elif fault=='position':raw['position_m'][2]+=.01
    elif fault=='normal':raw['normal']=[0.,0.,0.]
    elif fault=='path':raw['collision']=''
    else:ray['direction_world']=[0.,0.,0.]
    with pytest.raises(ValueError):sensor.scan_return(ray,[raw],own_collision_paths=['/World/actor/shape'],range_min=.1)


def test_all_four_actual_H5_fixed_coordinate_representations_and_goal_are_preserved():
    p=json.loads(PROTOCOL.read_bytes());bundle=json.loads(Path(p['frozen_bundle']).read_bytes())
    for name in bundle['plans']:
        plan=json.loads(Path(name).read_bytes());before=copy.deepcopy(plan)
        contract=sensor.fixed_route_frames(plan,(0.,0.,0.))
        assert plan==before and contract['source_action_state_count']==108
        assert len(contract['fixed_reference_map_xyyaw'])==109
        assert contract['exact_task_goal_map_xyyaw']==pytest.approx(plan['goal_world_xyyaw'])
        assert not contract['goal_edge_appended'] and not contract['route_or_terminal_or_reverse_execution_admitted']
        shifted=sensor.fixed_route_frames(plan,(18.,25.,math.pi/3))
        assert shifted['fixed_reference_world_xyyaw']==contract['fixed_reference_world_xyyaw']


@pytest.mark.parametrize('fault',['world','local','scenario','empty','action_count','frame'])
def test_fixed_route_does_not_adopt_rebased_or_inconsistent_reference(fault):
    p=json.loads(PROTOCOL.read_bytes());plan=json.loads(Path(p['selected_plan']).read_bytes());frame=(0.,0.,0.)
    if fault=='world':plan['selected_states'][0]['world_xyyaw'][0]+=.001
    elif fault=='local':plan['selected_states'][0]['x_m']+=.001
    elif fault=='scenario':plan['scenario']='corridor_90_turn'
    elif fault=='empty':plan['selected_states']=[]
    elif fault=='action_count':plan['selected_actions'].pop()
    else:frame=(0.,float('inf'),0.)
    with pytest.raises(ValueError):sensor.fixed_route_frames(plan,frame)


def test_source_only_review_detects_real_legacy_gaps_but_does_not_relabel_old_results():
    before=set(sys.modules)
    report=sensor.legacy_adapter_review(PROTOCOL)
    assert report['legacy_scan']['actor']=='/World/patient_transport_lab_screening/base_link'
    assert report['legacy_scan']['frame']=='base_link'
    assert len(report['legacy_scan']['physics_write_calls'])==2
    assert report['legacy_scan']['perimeter_advance_used']
    assert report['legacy_runner']['first_estimate_route_reanchor']
    assert report['legacy_runner']['scenario_guard'] and report['legacy_runner']['positive_speed_guard']
    assert report['legacy_nav2']['desired_linear_vel_m_s']==.08
    assert not report['legacy_nav2']['allow_reversing']
    assert report['legacy_nav2']['controller_odom_topic']=='/sim/ground_truth/odom'
    assert report['legacy_nav2']['launch_scalar_overrides']['publish_map_to_odom']=='false'
    assert report['legacy_nav2']['launch_scalar_overrides']['initial_alignment_policy']=='configuration_adaptive'
    assert report['legacy_map_replanning']['fake_planner_call_counts']['after_changed_map_same_goal_default_period']==1
    assert report['legacy_map_replanning']['exact_reviewed_source_verified_BEFORE_fake_method_execution']
    assert len(report['fixed_H5_four_method_coordinate_checks'])==4
    assert report['original_simple_corridor_results_NOT_reclassified']
    assert report['new_trial_count']==0 and not report['actual_motion_authorized']
    assert not report['sensor_or_localization_or_Nav2_success']
    assert not any(n in sys.modules and n not in before for n in ('rclpy','omni.kit.app','omni.physx'))


def test_source_review_export_is_idempotent_and_preserves_existing_edits(tmp_path):
    path=sensor.export_legacy_review(PROTOCOL,tmp_path);before=path.read_bytes()
    assert sensor.export_legacy_review(PROTOCOL,tmp_path)==path and path.read_bytes()==before
    path.write_text('{"SOFTWARE_USER_EDIT": true}')
    with pytest.raises(ValueError,match='no overwrite'):sensor.export_legacy_review(PROTOCOL,tmp_path)
    assert json.loads(path.read_bytes())=={'SOFTWARE_USER_EDIT':True}


def test_source_review_does_not_accept_an_edited_workflow_definition(tmp_path):
    data=json.loads(PROTOCOL.read_bytes());data['command_limits']['SOFTWARE_UNREVIEWED_LIMIT']=1.
    path=tmp_path/'SOFTWARE_ONLY_PROTOCOL.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Own offline H5 definition'):sensor.legacy_adapter_review(path)


def test_current_source_fingerprint_cannot_overwrite_an_existing_frozen_input_pin(tmp_path):
    import hashlib
    data=json.loads(PROTOCOL.read_bytes())
    old=str(_ijcas_path(sensor.ROOT, 'isaac/scripts/isaac_prepare_ijcas_cpu_laserscan.py'))
    data['input_sha256'][old]='0'*64
    data['definition_sha256']=hashlib.sha256(json.dumps({k:v for k,v in data.items() if k!='definition_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    path=tmp_path/'SOFTWARE_ONLY_CONFLICTING_INPUT.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Conflicting original frozen source'):sensor.legacy_adapter_review(path)


def encoder_geometry():
    return dict(front_axis_xy_m=[.665,0.],rear_axis_xy_m=[-.665,0.],front_rolling_offset_xy_m=[0.,0.],
        rear_rolling_offset_xy_m=[0.,0.],front_radius_m=.074,rear_radius_m=.074,front_drive_sign=1,rear_drive_sign=1,
        front_steering_zero_rad=0.,rear_steering_zero_rad=0.,front_steering_sign=1,rear_steering_sign=1)


def planning_identity_inputs(**edits):
    args=dict(cells=[-1,0,100,0],shape_rows_columns=[2,2],origin_xyyaw_m_rad=[0.,0.,0.],resolution_m=.125,
        map_frame_id='SOFTWARE_MAP',world_from_map_xyyaw=[10.,20.,0.],goal_world_xyyaw=[11.,20.,0.],
        environment_sha256='a'*64,planner_contract_sha256='b'*64)
    args.update(edits);return args


def test_planning_identity_uses_whole_unknown_signed_grid_without_stamp_or_first_estimate_reanchor():
    args=planning_identity_inputs();before=copy.deepcopy(args)
    first=sensor.planning_input_identity(**args);second=sensor.planning_input_identity(**copy.deepcopy(args))
    assert args==before and first==second
    assert first['unknown_values_preserved'] and first['acquisition_stamp_intentionally_NOT_a_content_revision']
    assert first['inputs']['goal_world_xyyaw']==[11.,20.,0.]
    assert first['inputs']['world_from_map_xyyaw']==[10.,20.,0.]
    assert not first['source_provenance_or_freshness_or_map_validity_verified']
    assert not first['stop_or_replan_or_resume_or_command_replacement_authorized']


@pytest.mark.parametrize('edits',[
    dict(cells=[0,0,100,0]),dict(cells=[-1,100,100,0]),dict(cells=[-1,0,50,0]),
    dict(shape_rows_columns=[1,4]),dict(origin_xyyaw_m_rad=[.1,0.,0.]),dict(origin_xyyaw_m_rad=[0.,0.,.1]),
    dict(resolution_m=.25),dict(map_frame_id='DIFFERENT_FIXED_MAP'),
    dict(world_from_map_xyyaw=[10.,20.,.1]),dict(goal_world_xyyaw=[11.,20.,.2]),
    dict(goal_world_xyyaw=[11.1,20.,0.]),dict(environment_sha256='c'*64),dict(planner_contract_sha256='d'*64)])
def test_every_observed_cell_frame_goal_or_planner_revision_changes_planning_identity(edits):
    baseline=sensor.planning_input_identity(**planning_identity_inputs())
    revised=sensor.planning_input_identity(**planning_identity_inputs(**edits))
    assert baseline['identity_sha256']!=revised['identity_sha256']


@pytest.mark.parametrize('edits',[
    dict(cells=[-1,0,100]),dict(cells=[False,0,100,0]),dict(cells=[-2,0,100,0]),dict(cells=[-1,0,101,0]),
    dict(cells=[-1.,0,100,0]),dict(cells=(-1,0,100,0)),dict(shape_rows_columns=[True,4]),
    dict(shape_rows_columns=[0,4]),dict(shape_rows_columns=[2001,2001]),
    dict(origin_xyyaw_m_rad=[0.,0.,math.nan]),dict(world_from_map_xyyaw=[10.,20.]),
    dict(goal_world_xyyaw=[11.,math.inf,0.]),dict(resolution_m=0.),dict(resolution_m=True),
    dict(resolution_m=1e-50),dict(resolution_m=1e100),dict(map_frame_id='/map'),dict(map_frame_id=''),
    dict(environment_sha256='a'*63),dict(planner_contract_sha256='X'*64)])
def test_bad_or_ambiguous_planning_snapshot_does_not_get_an_identity(edits):
    with pytest.raises((ValueError,TypeError)):sensor.planning_input_identity(**planning_identity_inputs(**edits))


def test_legacy_map_cache_and_timer_are_reproduced_without_ROS_or_H5_trial():
    path=_ijcas_path(sensor.ROOT, 'src/patient_initial_stability/patient_initial_stability/occupancy_lattice_planner.py')
    original=path.read_bytes();before=set(sys.modules);r=sensor.legacy_replanning_source_review()
    assert path.read_bytes()==original and r['fake_planner_call_counts']==dict(initial=1,
        after_changed_map_same_goal_default_period=1,after_periodic_timer_at_fixture_speed50mm_s=2,
        after_new_goal_callback=3)
    assert r['map_callback_stores_content_but_does_not_invalidate_same_goal_default_cache']
    assert r['periodic_timer_by_itself_is_NOT_measured_standstill_gate']
    assert r['new_goal_callback_clears_cache_including_heading_change']
    assert r['fake_plan_None_is_NOT_any_method_no_path_or_failure_evidence']
    assert r['no_entire_runtime_safety_or_physical_cause_conclusion']
    assert r['new_trial_count']==0 and not r['actual_motion_authorized'] and not r['H5_online_replanning_success']
    assert not any(n in sys.modules and n not in before for n in ('rclpy','omni.kit.app','omni.physx'))


def test_legacy_cache_reproduction_cannot_be_reused_if_source_behavior_is_fixed(tmp_path):
    path=_ijcas_path(sensor.ROOT, 'src/patient_initial_stability/patient_initial_stability/occupancy_lattice_planner.py')
    source=path.read_text().replace('self.grid = message','self.grid = message\n        self.last_request = None')
    fixture=tmp_path/'SOFTWARE_CACHE_CHANGED_NOT_PRODUCTION.py';fixture.write_text(source)
    with pytest.raises(ValueError,match='behavior changed'):sensor.legacy_replanning_source_review(fixture)


def test_planning_identity_matches_installed_occupancy_grid_signed_CDR_without_context():
    serialization=pytest.importorskip('rclpy.serialization',reason='Matching ROS ABI only; no SDK/ROS node')
    from nav_msgs.msg import OccupancyGrid
    import rclpy
    msg=OccupancyGrid();msg.header.frame_id='SOFTWARE_MAP';msg.header.stamp.sec=1
    msg.info.resolution=.125;msg.info.width=msg.info.height=2;msg.info.origin.orientation.w=1.
    msg.data=[-1,0,100,0]
    wire=serialization.deserialize_message(serialization.serialize_message(msg),OccupancyGrid)
    def identity(message):
        args=planning_identity_inputs(cells=list(message.data),shape_rows_columns=[message.info.height,message.info.width],
            resolution_m=message.info.resolution,map_frame_id=message.header.frame_id)
        return sensor.planning_input_identity(**args)
    assert identity(wire)==sensor.planning_input_identity(**planning_identity_inputs())
    wire.header.stamp.sec=9
    assert identity(wire)['identity_sha256']==sensor.planning_input_identity(**planning_identity_inputs())['identity_sha256']
    assert not rclpy.ok()


def stopped_replan_fixture(*,method='P_dynamics_full_envelope'):
    """Invented four-cell grid, clock/measurements/audit: SOFTWARE ONLY."""
    from types import SimpleNamespace
    initial=sensor.planning_input_identity(**planning_identity_inputs())
    changed=sensor.planning_input_identity(**planning_identity_inputs(cells=[-1,100,100,0]))
    clock=[100.]
    policy=dict(stop_speed_m_s=.005,stop_yaw_rate_rad_s=.01,stop_observation_s=.1,
        maximum_sample_gap_s=.25,stop_timeout_s=1.,planning_wall_budget_s=2.,
        mission_sim_budget_s=10.,mission_wall_budget_s=20.,resume_position_m=.001,
        resume_heading_rad=.001,resume_steering_rad=.001,resume_body_twist=[.001,.001,.001],
        resume_drive_velocity_rad_s=.01)
    boundary=sensor.StoppedReplanBoundary(initial,method=method,policy=policy,allowed_environment_sha256=['a'*64],
        clock=lambda:clock[0],mission_start_sim_s=0.,mission_start_wall_s=100.)
    def row(t,**edits):
        result=dict(time_s=t,pose_world_xyyaw=[10.,20.,.2],body_twist_u_v_r=[.002,.001,.001],
            front_rear_steering_rad=[.17,-.09],front_rear_drive_velocity_rad_s=[.02,.03],
            zero_command_confirmed=True,hold_confirmed=False,retained_mode='rear_only',
            retained_direction=1,planner_started=True)
        result.update(edits);return result
    def send(t,identity=None,**edits):
        clock[0]=max(clock[0],100.+t)
        return boundary.observe(row(t,**edits),changed if identity is None else identity)
    def ready():
        assert send(0.,initial)['action']=='FOLLOW_EXISTING_REFERENCE'
        assert send(.05,body_twist_u_v_r=[.02,0.,0.],zero_command_confirmed=False)['action']=='ZERO_UNTIL_MEASURED_STOP'
        send(.1)
        assert send(.21)['action']=='REQUEST_CAPTURED_HOLD'
        assert send(.22,hold_confirmed=True)['phase']=='READY'
        return boundary.take_request()
    def plan(request):
        return dict(request_sha256=request['request_sha256'],method=method,
            initial_measurement=copy.deepcopy(request['payload']['measured_start']),
            goal_world_xyyaw=[11.,20.,0.],selected_actions=[dict(SOFTWARE_ONLY=True)],
            selected_states=[dict(SOFTWARE_ONLY=True)])
    def audit(candidate,request):
        return dict(plan_sha256=boundary._digest(candidate),request_sha256=request['request_sha256'],
            input_identity_sha256=request['payload']['input_identity']['identity_sha256'],
            checks={key:True for key in boundary.CHECKS})
    return SimpleNamespace(boundary=boundary,clock=clock,initial=initial,changed=changed,policy=policy,
        row=row,send=send,ready=ready,plan=plan,audit=audit)


@pytest.mark.parametrize('method',['B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'])
def test_same_replan_handshake_preserves_nonzero_actual_state_for_every_method_without_drive_authority(method):
    f=stopped_replan_fixture(method=method);request=f.ready();b=f.boundary
    assert request['payload']['measured_start']['front_rear_steering_rad']==[.17,-.09]
    assert request['payload']['measured_start']['body_twist_u_v_r']==[.002,.001,.001]
    assert request['payload']['measured_start']['planner_started'] is True
    candidate=f.plan(request);saved=copy.deepcopy(candidate)
    response=b.accept_result(candidate,validator=f.audit)
    assert response['action']=='KEEP_CAPTURED_HOLD_WAIT_FRESH_MEASUREMENT'
    with pytest.raises(RuntimeError,match='Fresh postvalidation'):b.take_resume()
    assert f.send(.23,hold_confirmed=True)['phase']=='RESUME_READY'
    result=b.take_resume()
    assert result['plan']==saved and candidate==saved and b.capture_count==1
    assert result['resume_measurement']['time_s']==.23 and result['resume_request_wall_s']==pytest.approx(100.23)
    assert result['resume_request_only_NOT_command_or_drive_admission']
    assert not result['actual_motion_authorized'] and not result['H5_online_replanning_success']
    assert [event['event'] for event in result['boundary_trace']]==['OBSERVED_CONTENT_CHANGE',
        'ONE_CAPTURE_REQUESTED','ACTUAL_HOLD_ACK','REQUEST_BOUND_VALIDATION_RETURNED',
        'FRESH_POSTVALIDATION_MEASUREMENT','ONE_RESUME_REQUEST_ONLY']
    with pytest.raises(RuntimeError):b.take_resume()


def test_identical_republished_content_never_triggers_stop_or_a_new_request():
    f=stopped_replan_fixture()
    for t in (0.,.1,.2):
        assert f.send(t,f.initial,zero_command_confirmed=False)['action']=='FOLLOW_EXISTING_REFERENCE'
    assert f.boundary.capture_count==0 and not f.boundary.events
    with pytest.raises(RuntimeError):f.boundary.take_request()


def test_transport_receipt_metadata_cannot_turn_identical_map_content_into_a_revision_during_planning():
    f=stopped_replan_fixture();f.ready();same=copy.deepcopy(f.changed)
    same['transport_receipt_NOT_part_of_content_identity']=dict(stamp_s=1000.)
    assert f.send(.23,same,hold_confirmed=True)['phase']=='PLANNING'
    assert f.boundary.capture_count==1


@pytest.mark.parametrize('edits',[
    dict(stop_speed_m_s=.0051),dict(maximum_sample_gap_s=.251),dict(stop_observation_s=0.),
    dict(stop_timeout_s=.05),dict(resume_position_m=.0031),dict(resume_heading_rad=math.radians(2.1)),
    dict(resume_body_twist=[.001,True,.001]),dict(resume_steering_rad=math.nan),
    dict(planning_wall_budget_s=-1.),dict(mission_sim_budget_s=False)])
def test_replanning_policy_is_explicit_finite_and_cannot_relax_original_guard_limits(edits):
    f=stopped_replan_fixture();policy={**f.policy,**edits}
    with pytest.raises(ValueError):sensor.StoppedReplanBoundary(f.initial,method='P_dynamics_full_envelope',
        policy=policy,allowed_environment_sha256=['a'*64],clock=lambda:100.,mission_start_sim_s=0.,mission_start_wall_s=100.)


def test_no_unreviewed_environment_digest_is_accepted_and_the_original_is_not_silently_replaced():
    f=stopped_replan_fixture()
    with pytest.raises(ValueError,match='Original environment'):sensor.StoppedReplanBoundary(f.initial,
        method='P_dynamics_full_envelope',policy=f.policy,allowed_environment_sha256=['c'*64],
        clock=lambda:100.,mission_start_sim_s=0.,mission_start_wall_s=100.)
    broken=copy.deepcopy(f.changed);broken['inputs']['goal_world_xyyaw'][0]+=.1
    with pytest.raises(ValueError,match='Unmodified'):f.send(.05,broken)
    assert f.boundary.phase=='HOLD'


@pytest.mark.parametrize('edits',[
    dict(body_twist_u_v_r=[.006,0.,0.]),dict(body_twist_u_v_r=[0.,0.,.02]),dict(zero_command_confirmed=False)])
def test_zero_request_or_stationary_pose_alone_never_substitutes_for_measured_standstill(edits):
    f=stopped_replan_fixture()
    for t in (.05,.15,.25,.35):assert f.send(t,**edits)['action']=='ZERO_UNTIL_MEASURED_STOP'
    assert f.boundary.capture_count==0
    with pytest.raises(RuntimeError):f.boundary.take_request()


def test_stop_window_is_contiguous_and_only_one_hold_request_is_allowed():
    f=stopped_replan_fixture();f.send(.05);f.send(.10,body_twist_u_v_r=[.02,0.,0.]);f.send(.15)
    assert f.send(.21)['action']=='ZERO_UNTIL_MEASURED_STOP'
    assert f.send(.26)['action']=='REQUEST_CAPTURED_HOLD'
    assert f.send(.27)['action']=='KEEP_ZERO_WAIT_HOLD_ACK'
    f.send(.28,hold_confirmed=True);request=f.boundary.take_request()
    with pytest.raises(RuntimeError):f.boundary.take_request()
    assert f.boundary.request==request and f.boundary.capture_count==1


@pytest.mark.parametrize('edits',[
    dict(pose_world_xyyaw=[10.002,20.,.2]),dict(pose_world_xyyaw=[10.,20.,.202]),
    dict(front_rear_steering_rad=[.172,-.09]),dict(front_rear_drive_velocity_rad_s=[.04,.03]),
    dict(body_twist_u_v_r=[.0001,.001,.001]),dict(retained_mode='front_only'),
    dict(retained_direction=-1),dict(planner_started=False),dict(hold_confirmed=False),
    dict(zero_command_confirmed=False),dict(body_twist_u_v_r=[.006,0.,0.])])
def test_changed_held_state_or_history_latches_hold_and_prevents_late_plan_recovery(edits):
    f=stopped_replan_fixture();request=f.ready()
    with pytest.raises((ValueError,RuntimeError)):f.send(.23,**{'hold_confirmed':True,**edits})
    assert f.boundary.phase=='HOLD'
    with pytest.raises(RuntimeError):f.boundary.accept_result(f.plan(request),validator=f.audit)


@pytest.mark.parametrize('when',['planning','validated'])
def test_new_map_revision_invalidates_pending_or_validated_plan_without_an_automatic_retry(when):
    f=stopped_replan_fixture();request=f.ready()
    if when=='validated':f.boundary.accept_result(f.plan(request),validator=f.audit)
    revised=sensor.planning_input_identity(**planning_identity_inputs(cells=[-1,100,100,100]))
    with pytest.raises(RuntimeError,match='New revision'):f.send(.23,revised,hold_confirmed=True)
    assert f.boundary.phase=='HOLD' and f.boundary.capture_count==1
    with pytest.raises(RuntimeError):f.boundary.take_resume()


@pytest.mark.parametrize('edits',[
    dict(goal_world_xyyaw=[11.1,20.,0.]),dict(world_from_map_xyyaw=[10.1,20.,0.]),
    dict(map_frame_id='ANOTHER_MAP'),dict(planner_contract_sha256='d'*64),
    dict(resolution_m=.25),dict(shape_rows_columns=[1,4]),dict(environment_sha256='c'*64)])
def test_fixed_task_frame_and_unreviewed_environment_cannot_be_rebaselined_as_an_observation(edits):
    f=stopped_replan_fixture();changed=sensor.planning_input_identity(**planning_identity_inputs(**edits))
    with pytest.raises(ValueError):f.send(.05,changed)
    assert f.boundary.phase=='HOLD' and f.boundary.capture_count==0


@pytest.mark.parametrize('fault',['wrong_request','nominal_zero_start','wrong_method','moved_goal','empty_actions','missing_state'])
def test_incomplete_foreign_or_nominal_start_plan_never_reaches_the_independent_validator(fault):
    f=stopped_replan_fixture();request=f.ready();candidate=f.plan(request);calls=[]
    if fault=='wrong_request':candidate['request_sha256']='0'*64
    elif fault=='nominal_zero_start':candidate['initial_measurement']['body_twist_u_v_r']=[0.,0.,0.]
    elif fault=='wrong_method':candidate['method']='B0_fixed_counterphase_ramp'
    elif fault=='moved_goal':candidate['goal_world_xyyaw'][0]+=.1
    elif fault=='empty_actions':candidate['selected_actions']=[]
    else:candidate['selected_states']=[]
    with pytest.raises(ValueError):f.boundary.accept_result(candidate,validator=lambda *args:calls.append(args))
    assert calls==[] and f.boundary.phase=='HOLD'


@pytest.mark.parametrize('fault',list(sensor.StoppedReplanBoundary.CHECKS)+['missing_check','stale_plan','stale_request','stale_map','truthy_not_bool'])
def test_all_independent_geometry_and_state_checks_are_required_and_bound_to_the_exact_result(fault):
    f=stopped_replan_fixture();request=f.ready()
    def validator(plan,req):
        report=f.audit(plan,req)
        if fault in f.boundary.CHECKS:report['checks'][fault]=False
        elif fault=='missing_check':report['checks'].pop('support')
        elif fault=='stale_plan':report['plan_sha256']='0'*64
        elif fault=='stale_request':report['request_sha256']='0'*64
        elif fault=='stale_map':report['input_identity_sha256']='0'*64
        else:report['checks']['support']=1
        return report
    with pytest.raises(ValueError):f.boundary.accept_result(f.plan(request),validator=validator)
    assert f.boundary.phase=='HOLD'


def test_validation_must_charge_its_own_elapsed_wall_time_and_cannot_recover_a_concurrent_invalidation():
    f=stopped_replan_fixture();request=f.ready()
    def slow(plan,req):f.clock[0]+=2.01;return f.audit(plan,req)
    with pytest.raises(TimeoutError):f.boundary.accept_result(f.plan(request),validator=slow)
    assert f.boundary.phase=='HOLD'
    f=stopped_replan_fixture();request=f.ready()
    def invalidated(plan,req):
        try:f.send(.23,hold_confirmed=False)
        except RuntimeError:pass
        return f.audit(plan,req)
    with pytest.raises(RuntimeError,match='invalidated during'):f.boundary.accept_result(f.plan(request),validator=invalidated)
    assert f.boundary.phase=='HOLD'


def test_callback_and_caller_copies_cannot_mutate_the_captured_request_plan_or_audit():
    f=stopped_replan_fixture();request=f.ready();candidate=f.plan(request);original=copy.deepcopy(candidate)
    request['payload']['measured_start']['body_twist_u_v_r']=[9.,9.,9.]
    def mutating(plan,req):
        audit=f.audit(plan,req);plan.clear();req.clear();return audit
    f.boundary.accept_result(candidate,validator=mutating);candidate.clear()
    f.send(.23,hold_confirmed=True);result=f.boundary.take_resume()
    assert result['plan']==original
    assert result['request']['payload']['measured_start']['body_twist_u_v_r']==[.002,.001,.001]


def test_explicit_no_path_is_retained_without_resume_or_global_impossibility_claim():
    f=stopped_replan_fixture();request=f.ready();result=f.boundary.no_path(request['request_sha256'])
    assert result['phase']=='NO_PATH' and result['action']=='KEEP_ZERO_NO_PATH_NO_RETRY'
    assert not result['actual_motion_authorized']
    with pytest.raises(RuntimeError):f.boundary.take_resume()
    with pytest.raises(RuntimeError):f.boundary.observe(f.row(.23,hold_confirmed=True),f.changed)


@pytest.mark.parametrize('stage',['request','resume'])
def test_stale_measurement_cannot_be_used_to_request_planning_or_resume(stage):
    f=stopped_replan_fixture();request=f.ready()
    if stage=='resume':
        f.boundary.accept_result(f.plan(request),validator=f.audit);f.send(.23,hold_confirmed=True)
    else:
        # Fresh fixture reaches actual ACK but deliberately does not take request.
        f=stopped_replan_fixture();f.send(.05);f.send(.16);f.send(.17,hold_confirmed=True)
    f.clock[0]+=.251
    with pytest.raises(RuntimeError,match='stale'):
        f.boundary.take_resume() if stage=='resume' else f.boundary.take_request()
    assert f.boundary.phase=='HOLD'


@pytest.mark.parametrize('fault',['sample_gap','repeated_sample','physics_reset','wall_reset','whole_wall_budget','invalid_ack','nonfinite','early_hold'])
def test_original_mission_clocks_and_backend_ack_schema_are_fail_closed(fault):
    f=stopped_replan_fixture();f.send(.05)
    sample=f.row(.1)
    if fault=='sample_gap':sample['time_s']=.31
    elif fault=='repeated_sample':sample['time_s']=.05
    elif fault=='physics_reset':sample['time_s']=0.
    elif fault=='wall_reset':f.clock[0]=99.
    elif fault=='whole_wall_budget':f.clock[0]=121.
    elif fault=='invalid_ack':sample['zero_command_confirmed']=1
    elif fault=='nonfinite':sample['front_rear_steering_rad'][0]=math.nan
    else:sample['hold_confirmed']=True
    with pytest.raises((ValueError,RuntimeError,TimeoutError)):f.boundary.observe(sample,f.changed)
    assert f.boundary.phase=='HOLD'


def measured_kernel_fixture(*,method='P_dynamics_full_envelope',rotated=False):
    """Invented held request and modeled prefix, NOT a live replan start."""
    from patient_initial_stability import steering_lattice as lattice
    f=stopped_replan_fixture(method=method);request=f.ready()
    if rotated:
        identity=sensor.planning_input_identity(**planning_identity_inputs(cells=[-1,100,100,0],
            world_from_map_xyyaw=[10.,20.,math.pi/2]))
        request['payload']['input_identity']=identity
        request['payload']['measured_start']['pose_world_xyyaw']=[10.,21.,.3]
        request['request_sha256']=sensor.StoppedReplanBoundary._digest(request['payload'])
    carried=lattice.State(9.,8.,.4,'rear_only',.5,.6,True,1,longitudinal_velocity=.03,
        lateral_velocity=.02,yaw_rate=.04,friction_utilization=.2,front_friction_utilization=.3,
        rear_friction_utilization=.1,minimum_normal_load_fraction=.7,abs_ltr=.15,
        actuator_utilization=.4,rigid_body_penalty_s=.8)
    return f,request,carried


@pytest.mark.parametrize('kernel_name',['search','refinement'])
@pytest.mark.parametrize('method',['B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'])
def test_both_original_planning_kernels_receive_measured_nonzero_state_without_forking_or_resetting(method,kernel_name):
    from dataclasses import asdict
    from types import FunctionType
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability import hierarchical_dynamics_planner as refinement
    _,request,carried=measured_kernel_fixture(method=method)
    original=lattice.plan if kernel_name=='search' else refinement.refine_speed_consistent_lattice_route
    before=dict(original.__globals__);captured=[];stop=TimeoutError('SOFTWARE_FIRST_PROPAGATION_ONLY')
    def propagation(state,*args,**kwargs):captured.append(state);raise stop
    private=FunctionType(original.__code__,dict(original.__globals__,propagate=propagation),original.__name__,original.__defaults__)
    private.__kwdefaults__=copy.deepcopy(original.__kwdefaults__)
    bound=sensor.measured_replanning_kernel(private,request,carried,carried_state_frame_id='SOFTWARE_MAP')
    receipt=bound.measured_replanning;goal=receipt['fixed_goal_local_xyyaw']
    with pytest.raises(TimeoutError) as caught:
        if kernel_name=='search':
            bound(goal,-10.,10.,collision_checker=lambda poses:False,node_limit=1,
                allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),dynamics=lattice.KINEMATIC_DYNAMICS)
        else:
            bound('SOFTWARE_ROUTE_NOT_H5',[dict(mode='rear_only',curvature_1_m=0.,direction=1)],
                goal,(.003,.05),object(),lambda poses:False,.005,.02,terminal_extension_edges=0,maximum_iterations=1)
    assert caught.value is stop and len(captured)==1
    actual=asdict(captured[0]);assert actual==receipt['initial_state']
    assert actual['x']==0. and actual['y']==0. and actual['yaw']==pytest.approx(.2)
    assert [actual['front'],actual['rear']]==[.17,-.09]
    assert [actual['longitudinal_velocity'],actual['lateral_velocity'],actual['yaw_rate']]==[.002,.001,.001]
    assert actual['started'] is True and actual['direction']==1 and actual['mode']=='rear_only'
    for key,value in receipt['preserved_prefix_fields'].items():assert actual[key]==getattr(carried,key)==value
    assert receipt['original_kernel_code_object_reused'] and not receipt['actual_motion_authorized']
    assert original.__globals__==before and carried.x==9. and carried.front==.5


def test_measured_pose_and_goal_use_the_same_fixed_map_transform_but_body_twist_is_not_rotated():
    from patient_initial_stability import steering_lattice as lattice
    _,request,carried=measured_kernel_fixture(rotated=True)
    bound=sensor.measured_replanning_kernel(lattice.plan,request,carried,carried_state_frame_id='SOFTWARE_MAP')
    receipt=bound.measured_replanning;seed=receipt['initial_state']
    assert [seed['x'],seed['y'],seed['yaw']]==pytest.approx([1.,0.,.3-math.pi/2],abs=1e-14)
    assert receipt['fixed_goal_local_xyyaw']==pytest.approx([0.,-1.,-math.pi/2],abs=1e-14)
    assert [seed['longitudinal_velocity'],seed['lateral_velocity'],seed['yaw_rate']]==[.002,.001,.001]
    assert receipt['unchanged_goal_world_xyyaw']==[11.,20.,0.]
    assert receipt['world_task_NOT_reanchored_to_current_pose'] and receipt['first_start_exception_NOT_reenabled']


def test_original_NE_search_and_speed_refinement_complete_one_software_edge_from_measured_state():
    """Real original math, saved H5 coefficients; invented open-space goal.

    The goal is chosen from one propagated edge only to test adapter wiring.
    The free checker is NOT an H5 map/support/envelope validation; no actual
    planner performance, H5 success or physical stopping evidence is produced.
    """
    from dataclasses import asdict,replace
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability import hierarchical_dynamics_planner as refinement
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    bundle=json.loads(Path(json.loads(PROTOCOL.read_text())['frozen_bundle']).read_text())
    source=next(Path(name) for name in bundle['plans'] if Path(name).stem=='P_dynamics_full_envelope')
    assert sensor.sha(source)==bundle['output_sha256'][str(source)]
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**json.loads(source.read_text())['rigid_body_parameter_bounds']))
    _,request,carried=measured_kernel_fixture()
    seed=replace(carried,x=0.,y=0.,yaw=.2,front=.17,rear=-.09,
        longitudinal_velocity=.002,lateral_velocity=.001,yaw_rate=.001)
    expected,poses,_=lattice.propagate(seed,'coordinated_counterphase',0.,1,speed_m_s=.003,
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=.003)
    assert expected.rigid_body_feasible and math.hypot(expected.longitudinal_velocity,expected.lateral_velocity)<=.005
    world_goal=sensor.transform_pose([10.,20.,0.],[expected.x,expected.y,expected.yaw])
    request['payload']['input_identity']=sensor.planning_input_identity(**planning_identity_inputs(
        cells=[-1,100,100,0],goal_world_xyyaw=world_goal))
    request['payload']['original_goal_world_xyyaw']=world_goal
    request['request_sha256']=sensor.StoppedReplanBoundary._digest(request['payload'])
    checked=[]
    def free_checker(samples):checked.append(list(samples));return False
    search=sensor.measured_replanning_kernel(lattice.plan,request,carried,carried_state_frame_id='SOFTWARE_MAP')
    goal=search.measured_replanning['fixed_goal_local_xyyaw']
    result=search(goal,-1.,1.,minimum_x=-1.,maximum_x=1.,collision_checker=free_checker,node_limit=4,
        allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),directions=(1,),
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,speed_m_s=.003,speed_levels_m_s=(.003,),
        terminal_speed_tolerance_m_s=.005,terminal_deceleration_m_s2=.02)
    assert result is not None and len(result.states)==1 and asdict(result.states[0])==asdict(expected)
    refine=sensor.measured_replanning_kernel(refinement.refine_speed_consistent_lattice_route,request,
        carried,carried_state_frame_id='SOFTWARE_MAP')
    refined,_=refine('SOFTWARE_SINGLE_EDGE_NOT_H5_RESULT',
        [dict(mode='coordinated_counterphase',curvature_1_m=0.,direction=1)],goal,(.003,),
        model,free_checker,.005,.02,terminal_extension_edges=0,maximum_iterations=1,
        dynamics=lattice.FULL_DYNAMICS,include_configuration_cost=True)
    assert refined is not None and len(refined.states)==1 and asdict(refined.states[0])==asdict(expected)
    _,reset_poses,_=lattice.propagate(replace(seed,longitudinal_velocity=0.,lateral_velocity=0.,yaw_rate=0.),
        'coordinated_counterphase',0.,1,speed_m_s=.003,dynamics=lattice.FULL_DYNAMICS,
        rigid_body_model=model,target_speed_m_s=.003)
    assert max(abs(a-b) for a,b in zip(poses[0],reset_poses[0]))>1e-6
    assert checked and any(len(samples)>1 for samples in checked)


def test_resource_state_class_and_consumed_prefix_fields_are_not_reconstructed_from_defaults():
    from dataclasses import field,make_dataclass
    from types import FunctionType
    from patient_initial_stability import steering_lattice as lattice
    _,request,_=measured_kernel_fixture()
    State=make_dataclass('SOFTWARE_RESOURCE_STATE',[('mission_reverse_total_bins',int,field(default=0)),
        ('mission_direction_changes',int,field(default=0))],bases=(lattice.State,),frozen=True)
    carried=State(.8,.1,.2,'rear_only',.2,.1,True,1,mission_reverse_total_bins=65,mission_direction_changes=3)
    private=FunctionType(lattice.plan.__code__,dict(lattice.plan.__globals__,State=State),lattice.plan.__name__,lattice.plan.__defaults__)
    bound=sensor.measured_replanning_kernel(private,request,carried,carried_state_frame_id='SOFTWARE_MAP')
    assert bound.measured_replanning['initial_state']['mission_reverse_total_bins']==65
    assert bound.measured_replanning['initial_state']['mission_direction_changes']==3
    assert carried.mission_reverse_total_bins==65
    # This test only proves constructor handoff, not downstream resource propagation.


def test_resource_handoff_refuses_a_downstream_original_propagator_that_drops_extended_history():
    from dataclasses import field,make_dataclass
    from types import FunctionType
    from patient_initial_stability import steering_lattice as lattice
    _,request,_=measured_kernel_fixture()
    State=make_dataclass('SOFTWARE_RESOURCE_DROP_GUARD',[('mission_reverse_total_bins',int,field(default=0))],
        bases=(lattice.State,),frozen=True)
    carried=State(.8,.1,.2,'rear_only',.2,.1,True,1,mission_reverse_total_bins=65)
    private=FunctionType(lattice.plan.__code__,dict(lattice.plan.__globals__,State=State),lattice.plan.__name__,lattice.plan.__defaults__)
    seed=State(0.,0.,.2,'rear_only',.17,-.09,True,1,longitudinal_velocity=.002,lateral_velocity=.001,
        yaw_rate=.001,mission_reverse_total_bins=65)
    end,_,_=lattice.propagate(seed,'coordinated_counterphase',0.,1,dynamics=lattice.KINEMATIC_DYNAMICS)
    assert type(end) is lattice.State  # Raw original kernel has no resource extension.
    goal=sensor.transform_pose([10.,20.,0.],[end.x,end.y,end.yaw])
    request['payload']['input_identity']=sensor.planning_input_identity(**planning_identity_inputs(
        cells=[-1,100,100,0],goal_world_xyyaw=goal))
    request['payload']['original_goal_world_xyyaw']=goal
    request['request_sha256']=sensor.StoppedReplanBoundary._digest(request['payload'])
    bound=sensor.measured_replanning_kernel(private,request,carried,carried_state_frame_id='SOFTWARE_MAP')
    with pytest.raises(RuntimeError,match='dropped prefix-state/resource'):
        bound(bound.measured_replanning['fixed_goal_local_xyyaw'],-1.,1.,minimum_x=-1.,maximum_x=1.,
            collision_checker=lambda poses:False,node_limit=3,allowed_modes=('coordinated_counterphase',),
            allowed_curvatures=(0.,),directions=(1,),dynamics=lattice.KINEMATIC_DYNAMICS)


@pytest.mark.parametrize('edits',[
    dict(initial_mode='front_only'),dict(initial_front_rad=0.),dict(initial_rear_rad=0.),
    dict(propagate_initial_steering=False),dict(propagate_initial_steering=1)])
def test_conflicting_nominal_initial_arguments_cannot_override_the_captured_measurement(edits):
    from patient_initial_stability import steering_lattice as lattice
    _,request,carried=measured_kernel_fixture()
    bound=sensor.measured_replanning_kernel(lattice.plan,request,carried,carried_state_frame_id='SOFTWARE_MAP')
    with pytest.raises(ValueError,match='contradicts'):
        bound(bound.measured_replanning['fixed_goal_local_xyyaw'],-1.,1.,collision_checker=lambda poses:False,**edits)


@pytest.mark.parametrize('fault',['request_digest','missing_payload','wrong_method','frame','missing_checker','goal',
    'hold_not_actual','zero_not_actual','still_moving','old_goal','history','started','nonfinite_prefix','foreign_kernel'])
def test_bad_measured_start_or_shifted_task_never_reaches_original_search(fault):
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    _,request,carried=measured_kernel_fixture();row=request['payload']['measured_start']
    original=lattice.plan;frame='SOFTWARE_MAP'
    if fault=='request_digest':request['request_sha256']='0'*64
    elif fault=='missing_payload':request['payload'].pop('event_observed_sim_s')
    elif fault=='wrong_method':request['payload']['method']='UNKNOWN'
    elif fault=='frame':frame='OTHER_FRAME'
    elif fault=='hold_not_actual':row['hold_confirmed']=False
    elif fault=='zero_not_actual':row['zero_command_confirmed']=1
    elif fault=='still_moving':row['body_twist_u_v_r']=[.006,0.,0.]
    elif fault=='old_goal':request['payload']['original_goal_world_xyyaw']=[99.,20.,0.]
    elif fault=='history':carried=replace(carried,mode='front_only')
    elif fault=='started':carried=replace(carried,started=False)
    elif fault=='nonfinite_prefix':carried=replace(carried,actuator_utilization=math.nan)
    elif fault=='foreign_kernel':original=lambda *args,**kwargs:None
    if fault not in ('request_digest',):request['request_sha256']=sensor.StoppedReplanBoundary._digest(request['payload'])
    with pytest.raises((ValueError,KeyError)):
        bound=sensor.measured_replanning_kernel(original,request,carried,carried_state_frame_id=frame)
        if fault=='missing_checker':bound(bound.measured_replanning['fixed_goal_local_xyyaw'],-1.,1.)
        elif fault=='goal':bound([2.,0.,0.],-1.,1.,collision_checker=lambda poses:False)


@pytest.mark.parametrize('u,v,r',[(.05,0.,0.),(-.05,0.,0.),(.04,.01,.02),(-.04,-.01,-.02),(0.,.02,.01),(0.,0.,0.)])
def test_encoder_inverse_recovers_independent_two_contact_velocities_without_command_or_mode_input(u,v,r):
    g=encoder_geometry();rolling=[];steer=[]
    for axle in ('front','rear'):
        contact=np.array([u,v+g[axle+'_axis_xy_m'][0]*r]);delta=math.atan2(contact[1],contact[0]) if np.linalg.norm(contact) else 0.
        rolling.append(float(np.linalg.norm(contact))/g[axle+'_radius_m']);steer.append(delta)
    result=sensor.centerline_encoder_twist(*rolling,*steer,geometry=g)
    assert result['body_twist_u_v_r']==pytest.approx([u,v,r],abs=1e-15)
    assert result['axle_longitudinal_incompatibility_m_s']==pytest.approx(0.,abs=1e-15)
    assert result['no_slip_is_an_assumption_NOT_observed_or_proved'] and not result['actual_motion_authorized']


def test_encoder_inverse_exposes_inconsistent_axles_and_cannot_detect_equal_common_slip():
    g=encoder_geometry();result=sensor.centerline_encoder_twist(1.,2.,0.,0.,geometry=g)
    assert result['body_twist_u_v_r']==pytest.approx([.111,0.,0.])
    assert result['axle_longitudinal_incompatibility_m_s']==pytest.approx(-.074)
    consistent=sensor.centerline_encoder_twist(1.,1.,0.,0.,geometry=g)
    assert consistent['axle_longitudinal_incompatibility_m_s']==0.
    assert consistent['zero_incompatibility_does_NOT_bound_common_slip']
    # A stationary body with two spinning wheels could yield this SAME input;
    # no hidden GT channel exists from which to disambiguate it here.
    assert consistent['body_twist_u_v_r']==pytest.approx([.074,0.,0.])


def test_encoder_inverse_is_equal_weight_four_equation_least_squares_not_a_fitted_response_scale():
    rng=np.random.default_rng(20261118);g=encoder_geometry();g['rear_axis_xy_m'][0]=-.62
    A=np.array([[1.,0.,0.],[0.,1.,.665],[1.,0.,0.],[0.,1.,-.62]])
    for _ in range(100):
        wf,wr,df,dr=rng.uniform(-1.,1.,4)
        expected=np.linalg.lstsq(A,np.array([.074*wf*math.cos(df),.074*wf*math.sin(df),
            .074*wr*math.cos(dr),.074*wr*math.sin(dr)]),rcond=None)[0]
        assert sensor.centerline_encoder_twist(wf,wr,df,dr,geometry=g)['body_twist_u_v_r']==pytest.approx(expected,abs=1e-14)


def test_encoder_inverse_requires_separate_measured_signs_zeros_and_radii_not_old_ROS_or_command_geometry():
    g=encoder_geometry();g.update(rear_drive_sign=-1,front_steering_zero_rad=math.pi/2,rear_steering_zero_rad=math.pi/2,
        rear_steering_sign=-1,rear_radius_m=.08)
    result=sensor.centerline_encoder_twist(.05/.074,-.05/.08,math.pi/2,math.pi/2,geometry=g)
    assert result['body_twist_u_v_r']==pytest.approx([.05,0.,0.])


@pytest.mark.parametrize('fault',['missing','extra','nan','bad_radius','bad_sign','bool_sign','same_axis','offcentre','trail','bad_zero'])
def test_encoder_geometry_or_measured_channel_ambiguity_does_not_get_silently_repaired(fault):
    g=encoder_geometry();channels=[1.,1.,0.,0.]
    if fault=='missing':g.pop('rear_drive_sign')
    elif fault=='extra':g['command_speed_m_s']=.05
    elif fault=='nan':channels[0]=math.nan
    elif fault=='bad_radius':g['front_radius_m']=0.
    elif fault=='bad_sign':g['rear_drive_sign']=0.
    elif fault=='bool_sign':g['rear_drive_sign']=True
    elif fault=='same_axis':g['rear_axis_xy_m'][0]=.665
    elif fault=='offcentre':g['rear_axis_xy_m'][1]=.01
    elif fault=='trail':g['front_rolling_offset_xy_m'][0]=.01
    else:g['rear_steering_zero_rad']=math.inf
    with pytest.raises(ValueError):sensor.centerline_encoder_twist(*channels,geometry=g)


@pytest.mark.parametrize('twist',[[.05,0.,0.],[.03,.01,.02],[-.03,.01,-.02],[0.,0.,0.],[.03,0.,1e-12]])
def test_constant_twist_integration_matches_analytic_SE2_with_explicit_non_GT_origin(twist):
    times=np.linspace(1.,3.,121);initial=[4.,5.,.7]
    poses=sensor.integrate_body_twist_samples(times,np.tile(twist,(len(times),1)),initial_xyyaw=initial,maximum_sample_gap_s=.25)
    u,v,r=twist;theta=2.*r
    sinc=math.sin(theta)/theta if theta else 1.;cosc=2*math.sin(theta/2)**2/theta if theta else 0.
    expected=sensor.transform_pose(initial,[2*(sinc*u-cosc*v),2*(cosc*u+sinc*v),theta])
    assert poses[-1]==pytest.approx(expected,abs=1e-13) and poses[0]==initial


@pytest.mark.parametrize('fault',['repeat','backward','gap','missing','nan','negative','limit'])
def test_integration_does_not_restart_smooth_or_skip_bad_measured_clock_samples(fault):
    t=[0.,.1,.2];q=[[.05,0.,0.]]*3;gap=.25
    if fault=='repeat':t[2]=t[1]
    elif fault=='backward':t[1]=-.1
    elif fault=='gap':t[2]=1.
    elif fault=='missing':q=q[:2]
    elif fault=='nan':q[0]=[math.nan,0.,0.]
    elif fault=='negative':t=[-.1,0.,.1]
    else:gap=True
    with pytest.raises(ValueError):sensor.integrate_body_twist_samples(t,q,initial_xyyaw=[0.,0.,0.],maximum_sample_gap_s=gap)


def saved_encoder_fixture():
    return [dict(time_s=i*.1,phase='pre_hold' if i==0 else 'path_tracking',drive_front_velocity_rad_s=.05/.074,
        drive_rear_velocity_rad_s=.05/.074,front_rad=0.,rear_rad=0.,drive_front_rad=i*.005/.074,
        drive_rear_rad=i*.005/.074,x_m=10.+i*.005,y_m=20.,yaw_rad=0.,vx_physx_m_s=.05,vy_physx_m_s=0.,
        command_speed_m_s=math.nan,target_front_rad=math.nan,target_rear_rad=math.nan,command_mode='NOT_READ_BY_ESTIMATOR')
        for i in range(6)]


def test_retrospective_encoder_review_preserves_ALL_phases_and_uses_no_command_or_GT_fitting():
    rows=saved_encoder_fixture();before=copy.deepcopy(rows)
    # NaN command/target values are deliberate poison fields: not inputs.
    result=sensor.review_saved_encoder_trace(rows,geometry=encoder_geometry(),fixed_start_world_xyyaw=[10.,20.,0.],maximum_sample_gap_s=.25)
    assert result['all_rows']['samples']==6 and set(result['every_saved_phase'])=={'pre_hold','path_tracking'}
    assert sum(value['samples'] for value in result['every_saved_phase'].values())==6
    assert result['all_rows']['integrated_position_error_max_m']<1e-13
    changed=copy.deepcopy(rows)
    for row in changed:
        row['x_m']+=10.;row['yaw_rad']+=.2;row['vx_physx_m_s']=1000.;row['command_speed_m_s']=123.
    other=sensor.review_saved_encoder_trace(changed,geometry=encoder_geometry(),fixed_start_world_xyyaw=[10.,20.,0.],maximum_sample_gap_s=.25)
    assert result['derived_twists_float64_sha256']==other['derived_twists_float64_sha256']
    assert result['integrated_poses_float64_sha256']==other['integrated_poses_float64_sha256']
    assert other['all_rows']['integrated_position_error_last_m']>9.
    assert result['raw_wheel_positions_NOT_differentiated_or_unwrapped']
    for old,current in zip(before,rows):
        assert old['x_m']==current['x_m'] and old['time_s']==current['time_s']


@pytest.mark.parametrize('fault',['missing_encoder','nonfinite_measured','bad_phase','old_clock'])
def test_retrospective_encoder_review_never_replaces_or_drops_missing_bad_measured_rows(fault):
    rows=saved_encoder_fixture()
    if fault=='missing_encoder':rows[3].pop('drive_front_velocity_rad_s')
    elif fault=='nonfinite_measured':rows[3]['drive_rear_velocity_rad_s']=math.nan
    elif fault=='bad_phase':rows[3]['phase']=''
    else:rows[3]['time_s']=rows[2]['time_s']
    with pytest.raises((ValueError,KeyError)):sensor.review_saved_encoder_trace(rows,geometry=encoder_geometry(),
        fixed_start_world_xyyaw=[10.,20.,0.],maximum_sample_gap_s=.25)


def test_wrapped_joint_increments_recover_both_directions_only_under_explicit_unique_displacement_bound():
    time=np.linspace(0.,2.,21);continuous=np.column_stack((5.9+3.*time,-5.8-4.*time));period=4.*math.pi
    wrapped=(continuous+period/2.)%period-period/2.;roundoff=float(8.*np.spacing(period))
    result=sensor.bounded_wrapped_joint_increments(time,wrapped,wrap_period_rad=period,
        maximum_abs_rate_rad_s=[4.,5.],position_roundoff_rad=roundoff)
    # Both synthetic endpoints were themselves computed modulo a float64
    # period; compare within the declared endpoint-roundoff budget, not an
    # unrelated sub-ULP tolerance. This is NOT a physical acceptance change.
    assert np.asarray(result['displacements_rad'])==pytest.approx(np.diff(continuous,axis=0),abs=2.*roundoff)
    assert result['wrap_correction_counts']==[1,1] and min(result['minimum_alias_margin_rad'])>5.
    assert result['conditional_on_actual_wrap_and_between_sample_rate_contract']
    assert result['native_joint_type_and_cap_enforcement_NOT_measured_here'] and not result['actual_motion_authorized']


@pytest.mark.parametrize('fault',['period','bool_period','rate','rate_shape','roundoff','range','nonfinite','duplicate',
    'negative_clock','shape','half_period','incompatible_displacement','wrong_period'])
def test_wrapped_position_recovery_rejects_aliases_and_bad_units_without_auto_period_or_turn_count_guess(fault):
    t=[0.,.1,.2];q=np.array([[5.9,-5.9],[6.,-6.],[6.1,-6.1]]);period=4*math.pi;rate=[2.,2.];roundoff=1e-9
    if fault=='period':period=math.inf
    elif fault=='bool_period':period=True
    elif fault=='rate':rate=[0.,2.]
    elif fault=='rate_shape':rate=[2.]
    elif fault=='roundoff':roundoff=-1.
    elif fault=='range':q[0,0]=20.
    elif fault=='nonfinite':q[1,0]=math.nan
    elif fault=='duplicate':t[2]=t[1]
    elif fault=='negative_clock':t[0]=-.1
    elif fault=='shape':q=q[:2]
    elif fault=='half_period':rate=[period/(2*.1),2.]
    elif fault=='incompatible_displacement':q[1,0]=6.25
    else:period=2*math.pi
    with pytest.raises(ValueError):sensor.bounded_wrapped_joint_increments(t,q,wrap_period_rad=period,
        maximum_abs_rate_rad_s=rate,position_roundoff_rad=roundoff)


def test_compatible_endpoints_do_not_claim_to_prove_native_cap_enforcement_or_absence_of_hidden_turns():
    # A physical wheel completing a whole period between these endpoints
    # would violate the GIVEN cap. Endpoints alone cannot disprove that;
    # retain the conditional nature instead of claiming actual alias proof.
    result=sensor.bounded_wrapped_joint_increments([0.,.1],[[0.],[0.]],wrap_period_rad=4*math.pi,
        maximum_abs_rate_rad_s=[1.],position_roundoff_rad=0.)
    assert result['displacements_rad']==[[0.]]
    assert result['native_joint_type_and_cap_enforcement_NOT_measured_here']


def test_position_increment_review_retains_all_endpoints_and_intervals_without_double_averaging_or_GT_fit():
    rows=saved_encoder_fixture();args=dict(geometry=encoder_geometry(),fixed_start_world_xyyaw=[10.,20.,0.],
        maximum_sample_gap_s=.25,wrap_period_rad=4*math.pi,maximum_abs_rate_rad_s=[25.,25.],position_roundoff_rad=1e-9)
    result=sensor.review_saved_encoder_position_trace(rows,**args)
    assert result['all_rows']['endpoint_samples']==6 and result['all_rows']['intervals']==5
    assert sum(v['endpoint_samples'] for v in result['every_saved_phase'].values())==6
    assert sum(v['intervals'] for v in result['every_saved_phase'].values())==5
    assert result['every_saved_phase']['pre_hold']['interval_RMSE_u_v_r_vs_midpoint_pose_differences'] is None
    assert result['all_rows']['integrated_position_error_max_m']<1e-13
    poisoned=copy.deepcopy(rows)
    for row in poisoned:
        row['x_m']+=10.;row['command_speed_m_s']=123.;row['drive_front_velocity_rad_s']=10.;row['drive_rear_velocity_rad_s']=-10.
    other=sensor.review_saved_encoder_position_trace(poisoned,**args)
    assert result['derived_interval_twists_float64_sha256']==other['derived_interval_twists_float64_sha256']
    assert result['integrated_poses_float64_sha256']==other['integrated_poses_float64_sha256']
    assert other['all_rows']['integrated_position_error_last_m']>9.
    assert other['all_rows']['interval_wheel_rate_RMSE_vs_mean_endpoint_instantaneous_rates_rad_s'][0]>9.
    assert not result['actual_motion_authorized'] and result['no_slip_and_midpoint_steering_are_still_model_assumptions']


@pytest.mark.parametrize('fault',['gap','velocity_cap','nonfinite','phase','missing_position','missing_rate','ambiguous_interval'])
def test_position_increment_trace_review_fails_closed_without_dropping_rejected_samples(fault):
    rows=saved_encoder_fixture();gap=.25;rate=[25.,25.]
    if fault=='gap':gap=.05
    elif fault=='velocity_cap':rows[2]['drive_front_velocity_rad_s']=100.
    elif fault=='nonfinite':rows[2]['drive_front_rad']=math.nan
    elif fault=='phase':rows[2]['phase']=''
    elif fault=='missing_position':rows[2].pop('drive_front_rad')
    elif fault=='missing_rate':rows[2].pop('drive_front_velocity_rad_s')
    else:rate=[100.,100.]
    with pytest.raises((ValueError,KeyError)):sensor.review_saved_encoder_position_trace(rows,geometry=encoder_geometry(),
        fixed_start_world_xyyaw=[10.,20.,0.],maximum_sample_gap_s=gap,wrap_period_rad=4*math.pi,
        maximum_abs_rate_rad_s=rate,position_roundoff_rad=1e-9)
