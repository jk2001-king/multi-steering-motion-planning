"""Shared timing prototype remains method-blind and cannot authorize motion."""

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

import hospital_timed_path_execution as timed
import hospital_mission_execution as mission
import hospital_h3b_final_suffix as evidence


def mission_fixture(directions=(1,)*10):
    """Synthetic interface fixture, NOT a real map/physical trial."""
    from hospital_h3_turn_planning import model_bounds
    actions=[];states=[];x=0.
    for direction in directions:
        x+=direction*.05
        actions.append(dict(mode='coordinated_counterphase',curvature_1_m=0.,direction=direction,target_speed_m_s=.03))
        states.append(dict(x_m=x,y_m=0.,yaw_deg=0.,world_xyyaw=[x,0.,0.],mode='coordinated_counterphase'))
    geometry=dict(scenario='H6_integrated_transfer_mission',full_body_rectangle_m=[1.630,.965],
        full_body_clearance_m=.010,start_xyyaw=[0.,0.,0.],goal_xyyaw=[x,0.,0.],
        planning_occupancy_digest='SYNTHETIC_NOT_A_MAP',actual_motion_authorized=False,use_for_final_statistics=False)
    plan=dict(scenario=geometry['scenario'],method='P_dynamics_full_envelope',plan_success=True,
        selected_actions=actions,selected_states=states,start_world_xyyaw=[0.,0.,0.],goal_world_xyyaw=[x,0.,0.],
        goal_local_xyyaw=[x,0.,0.],planning_occupancy_digest=geometry['planning_occupancy_digest'],
        full_envelope_clearance_m=.010,common_max_command_speed_m_s=.05,rigid_body_parameter_bounds=model_bounds(),
        initial_front_rad=0.,initial_rear_rad=0.,initial_u_v_r=[0.,0.,0.],
        allowed_modes=list(timed.MODES),planner_dynamics='full',physical_execution_verified=False,
        Nav2_execution_verified=False,integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True,
        actual_motion_authorized=False,use_for_final_statistics=False)
    return plan,geometry


def test_mission_tracking_binding_changes_only_handoff_and_preserves_original_forward_feedback():
    bound=mission._mission_tracking_implementation()
    assert bound.mission_binding['AST_handoff_substitutions']==1
    assert bound.mission_binding['undo_reproduces_entire_original_AST']
    assert bound.mission_binding['feedback_gains_path_guards_and_all_other_statements_unchanged']


@pytest.mark.parametrize('direction',[1,-1])
def test_mission_reverse_frame_changes_yaw_feedback_sign_not_body_or_joint_observations(direction):
    from hospital_path_execution import relative_pose,wrap
    plan,geometry=mission_fixture((direction,)*10);before=copy.deepcopy((plan,geometry))
    controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    controller.stage='tracking';controller.previous_time=0.;controller.started=0.;controller.speed=direction*.02
    pose=[direction*.05,.005,.02];observed=pose.copy();steer=(.01,-.02)
    out=controller.step(.02,pose,.02,steer,drive_hold_confirmed=False)
    _,reference=controller.reference.at(controller.reference.progress)
    expected=direction*1.5*wrap(reference[2]-pose[2])+2*relative_pose(reference,pose)[1]
    assert out['curvature_1_m']==pytest.approx(expected)
    assert direction*out['speed_m_s']>0. and abs(out['speed_m_s'])<=.0204+1e-12
    assert pose==observed and steer==(.01,-.02) and (plan,geometry)==before


def test_mission_initial_hold_neutral_alignment_AND_released_ACK_precede_any_motion():
    plan,geometry=mission_fixture();controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    now=0.
    for _ in range(5):
        now+=.1;out=controller.step(now,[0.,0.,0.],.007,(0.,0.),drive_hold_confirmed=False)
        assert out['speed_m_s']==0. and not out['drive_hold_requested']
    for _ in range(5):
        now+=.1;out=controller.step(now,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
        assert out['speed_m_s']==0. and out['drive_hold_requested'] and controller.leg_index==0
    for _ in range(10):
        now+=.1;out=controller.step(now,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)
        assert out['speed_m_s']==0. and controller.leg_index==0
    assert controller.stage=='release_hold' and not out['drive_hold_requested']
    now+=.1;out=controller.step(now,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    assert controller.stage=='tracking' and out['speed_m_s']==0.
    now+=.1;out=controller.step(now,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    assert out['speed_m_s']>0.


def test_mission_cusp_neutral_advance_retains_observed_pose_and_requires_release_before_reverse():
    plan,geometry=mission_fixture((1,1,-1,-1,1,1));before=copy.deepcopy((plan,geometry))
    controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    controller.stage='cusp_braking';controller.previous_time=0.;controller.started=controller.stage_started=0.
    controller.speed=.003;pose=[.1,0.,0.];now=.2
    out=controller.step(now,pose,.003,(.3,-.2),drive_hold_confirmed=False)
    assert controller.stage=='neutral' and out['drive_hold_requested'] and controller.leg_index==0
    for _ in range(3):
        now+=.1;out=controller.step(now,pose,0.,(.3,-.2),drive_hold_confirmed=True)
        assert controller.leg_index==0 and out['speed_m_s']==0.
    for _ in range(10):
        now+=.1;out=controller.step(now,pose,0.,(0.,0.),drive_hold_confirmed=True)
        assert out['speed_m_s']==0.
    assert controller.leg_index==1 and controller.stage=='release_hold'
    now+=.1;controller.step(now,pose,0.,(0.,0.),drive_hold_confirmed=False)
    now+=.1;out=controller.step(now,pose,0.,(0.,0.),drive_hold_confirmed=False)
    assert out['speed_m_s']<0. and out['direction']==-1
    assert pose==[.1,0.,0.] and (plan,geometry)==before
    assert [x['stage'] for x in controller.events].count('neutral')==1


def test_mission_reference_cannot_project_across_a_direction_change_or_invent_goal_orientation():
    plan,_=mission_fixture((1,1,-1,-1,1,1));reference=mission.MissionPathReference(plan)
    assert len(reference.legs)==3
    reverse=reference.legs[1]['view'];assert abs(reverse.poses[0][2])==pytest.approx(math.pi)
    assert reference.poses[2][2]==0. and all(a['direction']==1 for a in reverse.actions)
    for leg in reference.legs:
        assert len(leg['view'].actions)==2 and leg['view'].progress==0.
    assert reference.actions==plan['selected_actions'] and reference.poses[-1]==tuple(plan['goal_world_xyyaw'])


@pytest.mark.parametrize('damage',['direction_bool','direction_zero','speed','curvature','task','body','clearance',
    'world_frame','goal_frame','mode','method_dynamics','authority','physical_claim','acceleration','initial_steer','initial_velocity'])
def test_mission_native_validation_rejects_changed_or_unadmitted_input(damage):
    plan,geometry=mission_fixture()
    if damage=='direction_bool':plan['selected_actions'][0]['direction']=True
    elif damage=='direction_zero':plan['selected_actions'][0]['direction']=0
    elif damage=='speed':plan['selected_actions'][0]['target_speed_m_s']=.1
    elif damage=='curvature':plan['selected_actions'][0]['curvature_1_m']=float('nan')
    elif damage=='task':geometry['scenario']='H5_obstacle_bypass_return'
    elif damage=='body':geometry['full_body_rectangle_m'][0]=1.29
    elif damage=='clearance':plan['full_envelope_clearance_m']=0.
    elif damage=='world_frame':plan['selected_states'][0]['world_xyyaw'][0]+=.001
    elif damage=='goal_frame':plan['goal_local_xyyaw'][1]+=.001
    elif damage=='mode':plan['selected_actions'][0]['mode']='differential'
    elif damage=='method_dynamics':plan['planner_dynamics']='kinematic'
    elif damage=='authority':plan['actual_motion_authorized']=True
    elif damage=='physical_claim':plan['physical_execution_verified']=True
    elif damage=='initial_steer':plan['initial_front_rad']=.1
    elif damage=='initial_velocity':plan['initial_u_v_r'][0]=.01
    else:plan['rigid_body_parameter_bounds']['max_longitudinal_accel_m_s2']=.03
    with pytest.raises(ValueError):mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)


def test_mission_step_requires_explicit_ack_and_no_stale_or_nonfinite_observation():
    plan,geometry=mission_fixture();controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    with pytest.raises(TypeError):controller.step(0.,[0.,0.,0.],0.,(0.,0.))
    with pytest.raises(ValueError):controller.step(0.,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed='False')
    with pytest.raises(ValueError):controller.step(0.,[float('nan'),0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    controller.step(0.,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    with pytest.raises(ValueError):controller.step(.3,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)


def test_mission_model_contract_does_not_adopt_600s_live_budget_or_invent_physical_hold_forces():
    c=mission.mission_execution_contract(simulation_budget_s=600.)
    assert c['runtime_adapter_prepared'] is False and c['actual_motion_authorized'] is False
    assert c['ONE_fixed5s_hold_and_fixed_last3s_remain_external_runtime_checks']
    assert c['next_motion_requires_hold_ACK_then_neutral_then_alignment_then_released_ACK']
    assert c['physical_states_joints_velocities_casters_never_reset_or_invented']
    with pytest.raises(ValueError):mission.mission_execution_contract(simulation_budget_s=900.)


def test_mission_original_terminal_capture_needs_neutral_ack_and_cannot_reselect_after_drift():
    plan,geometry=mission_fixture();controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    controller.terminal=controller.raw(controller.reference,controller.goal)
    terminal=controller.terminal
    terminal.previous_time=0.;terminal.terminal_started=terminal.stage_started=0.
    terminal.terminal_stage='entry_braking';terminal.phase='terminal_braking'
    controller.stage='terminal';controller.previous_time=controller.started=0.
    pose=[plan['goal_world_xyyaw'][0]+.001,0.,0.];now=.1
    out=controller.step(now,pose,0.,(.3,0.),drive_hold_confirmed=False)
    assert out['drive_hold_requested'] and not out['hold_ready']
    for _ in range(10):
        now+=.1;out=controller.step(now,pose,0.,(.3,0.),drive_hold_confirmed=True)
        assert not out['hold_ready'] and out['speed_m_s']==0.
    assert controller.stage=='final_neutral' and controller.hold_entry_candidate is not None
    first=copy.deepcopy(controller.hold_entry_candidate)
    for _ in range(4):
        now+=.1;out=controller.step(now,pose,0.,(0.,0.),drive_hold_confirmed=True)
    assert out['hold_ready'] and controller.hold_entry_candidate==first
    assert controller.pulse_count==0 and out['curvature_1_m']==0.
    now+=.1
    with pytest.raises(RuntimeError,match='no retry'):
        controller.step(now,[plan['goal_world_xyyaw'][0]+.031,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)


def test_mission_terminal_cannot_advance_an_alignment_without_confirmed_hold():
    plan,geometry=mission_fixture();controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    controller.terminal=controller.raw(controller.reference,controller.goal)
    terminal=controller.terminal
    terminal.previous_time=0.;terminal.terminal_started=terminal.stage_started=0.
    terminal.terminal_stage='stationary_settle';terminal.phase='terminal_correction';terminal.drive_hold_requested=True
    controller.stage='terminal';controller.previous_time=controller.started=0.
    pose=[plan['goal_world_xyyaw'][0]-.05,0.,0.]
    for i in range(1,11):
        out=controller.step(.1*i,pose,0.,(0.,0.),drive_hold_confirmed=False)
        assert out['speed_m_s']==0. and out['drive_hold_requested'] and not out['hold_ready']
        assert terminal.terminal_stage=='stationary_settle' and terminal.pulse_count==0


def test_all_four_actual_r19_native_inputs_bind_one_prospective_controller_without_changing_saved_sources():
    import screen_hospital_h4_common_executor as original
    root=Path(__file__).resolve().parents[3]
    folder=root/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_planning/heading0_fresh_terminal_beam_matched_dev_r19'
    review=json.loads((folder/'planning_review.json').read_bytes());contract=json.loads(Path(review['protocol']).read_bytes())
    geometry=json.loads(Path(contract['common_geometry']).read_bytes());settings=[]
    for row in review['methods']:
        plan=json.loads(Path(row['plan_file']).read_bytes());before=copy.deepcopy((plan,geometry))
        controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
        settings.append(controller.tracker.config)
        for i in range(5):
            info=controller.step(.1*i,plan['start_world_xyyaw'],0.,(0.,0.),drive_hold_confirmed=i>0)
        assert controller.stage=='alignment' and info['curvature_1_m']==0.
        assert controller.leg_index==0 and info['speed_m_s']==0.
        run,execution,design=mission.mission_model_case_binding(geometry,plan,simulation_budget_s=600.)
        assert run.original_case_code is original.case.__code__ and run.plant_code_and_guards_unchanged
        assert execution.GUARDS==dict(original.execution.GUARDS,maximum_simulation_duration_s=600.)
        assert execution.ACCEPTANCE==original.execution.ACCEPTANCE and execution.COMMAND_LIMITS==original.execution.COMMAND_LIMITS
        assert len(controller.reference.legs)==(1 if row['method'].startswith(('B0_','B1_')) else 3)
        assert (plan,geometry)==before and design==geometry
    assert all(value==settings[0] for value in settings)


class _FakeTensorBackend:
    """SDK-shaped STATEFUL fake; not a physical articulation or hold force."""
    def __init__(self):
        from types import SimpleNamespace
        root=Path(__file__).resolve().parents[3]
        path=root/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_experiments/cohort_01_common_preparation_r01/reviewed_first_static/hospital_h6_reviewed_static_r01.json'
        self.baseline=json.loads(path.read_bytes())['runtime_motor_baseline']
        self.shared_metatype=SimpleNamespace(dof_names=self.baseline['joint_names'],link_names=['link_%d'%i for i in range(13)])
        self.count=1;self.max_dofs=12;self.max_links=13;self.valid=True;self.paused=False;self.context_valid=True
        self.data={k:np.asarray(self.baseline[name],np.float32).reshape(1,12).copy()
            for k,name in (('kp','kp'),('kd','kd'),('limits','max_efforts'))}
        self.data.update(positions=np.arange(12,dtype=np.float32).reshape(1,12)/10.,velocities=np.zeros((1,12),np.float32))
        self.data['positions'][0,[0,3]]=0.
        self.physical_q=np.arange(12,dtype=np.float32).reshape(1,12)+10.
        self.physical_q[0,[0,3]]=0.;self.writes=[];self.drop=None;self.fail=None
    def check(self):return self.valid
    def get_dof_positions(self):return self.physical_q.copy()
    def get_dof_stiffnesses(self):return self.data['kp'].copy()
    def get_dof_dampings(self):return self.data['kd'].copy()
    def get_dof_max_forces(self):return self.data['limits'].copy()
    def get_dof_position_targets(self):return self.data['positions'].copy()
    def get_dof_velocity_targets(self):return self.data['velocities'].copy()
    def _set(self,key,value,indices):
        assert value.shape==(1,12) and value.dtype==np.float32
        assert indices.shape==(1,) and indices.dtype==np.int32 and indices[0]==0
        self.writes.append((key,value.copy()))
        if self.fail==key:raise RuntimeError('INJECTED_SDK_SETTER_FAILURE_'+key)
        if self.drop!=key:self.data[key]=value.copy()
    def set_dof_stiffnesses(self,data,indices):self._set('kp',data,indices)
    def set_dof_dampings(self,data,indices):self._set('kd',data,indices)
    def set_dof_position_targets(self,data,indices):self._set('positions',data,indices)
    def set_dof_velocity_targets(self,data,indices):self._set('velocities',data,indices)


def _sdk_fixture(*,authorized=True):
    fake=_FakeTensorBackend()
    class BackendUtils:
        @staticmethod
        def convert(value,device,dtype,indexed):
            assert device=='SYNTHETIC_CPU' and indexed is False
            return np.asarray(value,dtype=np.float32 if dtype=='float32' else np.int32).copy()
    encoder=mission.retained_tensor_encoder(BackendUtils,'SYNTHETIC_CPU')
    adapter=mission.RetainedTensorDriveAdapter(fake,fake.baseline,encode=encoder,
        check_context=lambda:fake.context_valid,paused_check=lambda:fake.paused,physics_stamp=1.,authorized=authorized)
    return fake,adapter


def test_retained_adapter_constructor_is_read_only_and_requires_real_baseline_and_valid_singleton():
    fake,adapter=_sdk_fixture(authorized=False);q=fake.physical_q.copy()
    assert not fake.writes and adapter.confirmation(1.) is False
    for action in (lambda:adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=0.),
            lambda:adapter.command((0.,0.),(0.,0.),1.),adapter.emergency_zero,adapter.restore_original):
        with pytest.raises(RuntimeError):action()
    assert not fake.writes and np.array_equal(fake.physical_q,q)
    fake.valid=False
    with pytest.raises(RuntimeError):adapter.confirmation(1.1)


def test_real_backend_readback_and_later_physics_sample_not_request_echo_confirm_hold():
    fake,adapter=_sdk_fixture();q=fake.physical_q.copy();passive={k:v[:,adapter.passive].copy() for k,v in fake.data.items()}
    adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=.004)
    assert [name for name,value in fake.writes]==['velocities','positions','kp']
    assert adapter.confirmation(1.) is None and adapter.confirmed_hold is False
    assert adapter.confirmation(1.01) is True and adapter.capture_count==1
    assert np.array_equal(fake.data['positions'][0,adapter.drive],q[0,adapter.drive])
    fake.physical_q[0,adapter.drive]+=5.
    adapter.request_hold(True,1.01,command_speed_m_s=0.,measured_speed_m_s=0.)
    adapter.command((0.,0.),(0.,0.),1.01)
    assert adapter.capture_count==1 and np.array_equal(fake.data['positions'][0,adapter.drive],q[0,adapter.drive])
    for key,value in passive.items():assert np.array_equal(fake.data[key][:,adapter.passive],value)
    assert np.array_equal(fake.physical_q[0,adapter.steering],q[0,adapter.steering])


def test_release_must_restore_gain_and_be_read_back_before_any_nonzero_drive_command():
    fake,adapter=_sdk_fixture();adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=0.)
    adapter.confirmation(1.01);fake.writes.clear()
    adapter.request_hold(False,1.02,command_speed_m_s=0.,measured_speed_m_s=0.)
    assert [name for name,value in fake.writes]==['kp','positions','velocities']
    assert adapter.confirmation(1.02) is None
    with pytest.raises(RuntimeError):adapter.command((0.,0.),(.02/.074,.02/.074),1.02)
    assert adapter.confirmation(1.03) is False
    adapter.command((0.,0.),(.02/.074,.02/.074),1.03)
    assert np.all(fake.data['velocities'][0,adapter.drive]>0.)
    assert np.array_equal(fake.data['kp'][0],adapter.original['kp'])
    assert np.array_equal(fake.data['positions'][0,adapter.drive],adapter.original['positions'][adapter.drive])


@pytest.mark.parametrize('fault',['kp','positions','velocities','kd','limits','passive_target','stale_stamp','moving_hold'])
def test_sdk_confirmation_fails_closed_on_bad_readback_and_not_just_internal_boolean(fault):
    fake,adapter=_sdk_fixture()
    if fault=='moving_hold':
        with pytest.raises(RuntimeError):adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=.0051)
        assert not fake.writes;return
    if fault=='stale_stamp':
        adapter.confirmation(1.1)
        with pytest.raises(ValueError):adapter.confirmation(1.)
        return
    adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=0.)
    if fault=='passive_target':fake.data['positions'][0,adapter.passive[0]]+=.01
    else:fake.data[fault][0,adapter.drive[0]]+=1.
    with pytest.raises(RuntimeError):adapter.confirmation(1.01)
    assert adapter.confirmed_hold is False


def test_dropped_gain_write_can_never_be_certified_as_a_captured_hold():
    fake,adapter=_sdk_fixture();fake.drop='kp'
    with pytest.raises(RuntimeError):adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=0.)
    with pytest.raises(RuntimeError):adapter.confirmation(1.01)
    assert not adapter.confirmed_hold and adapter.dirty


def test_original_active_targets_and_gains_restore_only_after_actual_pause_without_state_or_force_writes():
    fake,adapter=_sdk_fixture();physical=fake.physical_q.copy();baseline=copy.deepcopy(fake.data)
    adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=.003);adapter.confirmation(1.01)
    adapter.command((.01,-.01),(0.,0.),1.1)
    count=len(fake.writes)
    with pytest.raises(RuntimeError):adapter.restore_original()
    assert len(fake.writes)==count
    fake.paused=True;report=adapter.restore_original()
    assert report['restored'] and all(report['checks'].values()) and adapter.closed
    assert all(np.array_equal(fake.data[k],v) for k,v in baseline.items())
    assert np.array_equal(fake.physical_q,physical)
    assert not any(name in ('limits','physical_positions','physical_velocities','root_transform') for name,value in fake.writes)
    with pytest.raises(RuntimeError):adapter.command((0.,0.),(0.,0.),1.2)


def test_partial_restoration_failure_still_attempts_other_independent_fields_and_is_never_green():
    fake,adapter=_sdk_fixture();adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=0.)
    fake.paused=True;fake.fail='kp';before=len(fake.writes)
    report=adapter.restore_original()
    assert not report['restored'] and report['errors'] and adapter.closed
    assert [k for k,v in fake.writes[before:]]==['velocities','kp','kd','positions']
    assert np.array_equal(fake.data['velocities'][0,adapter.active],adapter.original['velocities'][adapter.active])
    assert np.array_equal(fake.data['positions'][0,adapter.active],adapter.original['positions'][adapter.active])


def test_cleanup_does_not_silently_repair_an_unexpected_passive_target_edit():
    fake,adapter=_sdk_fixture();index=adapter.passive[0]
    fake.data['positions'][0,index]=.789;fake.paused=True
    report=adapter.restore_original()
    assert not report['restored'] and not report['checks']['positions']
    assert fake.data['positions'][0,index]==np.float32(.789)


def test_sdk_bridge_uses_backend_confirmation_for_capture_and_release_and_no_same_sample_double_command():
    fake,adapter=_sdk_fixture();plan,geometry=mission_fixture()
    controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    guards=[];bridge=mission.MissionSDKActuationBridge(controller,adapter,guard=lambda *a:guards.append(a))
    pose=[0.,0.,0.];now=1.
    first=bridge.step(now,pose,0.)
    assert first['SDK_hold_confirmation'] is None and first['drive_hold_requested']
    count=len(fake.writes);repeated=bridge.step(now,pose,0.)
    assert repeated['SDK_physics_sample_not_advanced'] and len(fake.writes)==count
    for _ in range(12):
        now+=.1;out=bridge.step(now,pose,0.)
    assert controller.stage=='tracking' and out['speed_m_s']>0. and adapter.confirmed_hold is False
    count=len(fake.writes);out=bridge.step(now,pose,0.)
    assert out['SDK_physics_sample_not_advanced'] and len(fake.writes)==count and not bridge.failed
    assert adapter.capture_count==1 and len(guards)>=15


def test_sdk_bridge_guard_failure_precedes_mutations_and_is_terminal_no_automatic_retry():
    fake,adapter=_sdk_fixture();plan,geometry=mission_fixture()
    def fail(*args):raise RuntimeError('ACTUAL_SAFETY_GUARD_FAILURE')
    bridge=mission.MissionSDKActuationBridge(mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.),adapter,guard=fail)
    with pytest.raises(RuntimeError,match='ACTUAL_SAFETY'):bridge.step(1.,[0.,0.,0.],0.)
    assert not fake.writes and bridge.failed
    with pytest.raises(RuntimeError,match='no automatic retry'):bridge.step(1.1,[0.,0.,0.],0.)


def test_sdk_bridge_rechecks_measured_stop_and_alignment_on_released_ack():
    fake,adapter=_sdk_fixture();plan,geometry=mission_fixture();controller=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    controller.stage='release_hold';controller.curvature=0.
    bridge=mission.MissionSDKActuationBridge(controller,adapter,guard=lambda *args:None)
    with pytest.raises(RuntimeError,match='stop/alignment'):bridge.step(1.,[0.,0.,0.],.006)
    assert not fake.writes


@pytest.mark.parametrize('setter',['set_dof_positions','set_dof_velocities','set_root_transforms','set_dof_max_forces'])
def test_even_private_sdk_writer_cannot_assign_physical_state_or_force_limits(setter):
    fake,adapter=_sdk_fixture()
    with pytest.raises(ValueError):adapter._set(setter,np.zeros(12,np.float32))
    assert not fake.writes


def test_multiple_holds_capture_measured_positions_once_each_and_keep_full_original_baseline():
    fake,adapter=_sdk_fixture();baseline=copy.deepcopy(fake.data);now=1.
    for number in range(3):
        fake.physical_q[0,adapter.drive]+=100.
        measured=fake.physical_q[0,adapter.drive].copy()
        adapter.request_hold(True,now,command_speed_m_s=0.,measured_speed_m_s=.003)
        assert adapter.confirmation(now) is None
        now+=.02;assert adapter.confirmation(now) is True
        assert np.array_equal(fake.data['positions'][0,adapter.drive],measured)
        adapter.command((0.,0.),(0.,0.),now)
        now+=.02;adapter.request_hold(False,now,command_speed_m_s=0.,measured_speed_m_s=.001)
        now+=.02;assert adapter.confirmation(now) is False
    assert adapter.capture_count==3
    fake.paused=True;assert adapter.restore_original()['restored']
    assert all(np.array_equal(fake.data[k],v) for k,v in baseline.items())


def test_installed_tensor_api_and_all_backend_converters_match_adapter_without_loading_Kit():
    sdk=Path('/home/jk/miniconda3/envs/isaac_env/lib/python3.11/site-packages/isaacsim')
    candidates=list((sdk/'extscache').glob('omni.physics.tensors-*/omni/physics/tensors/impl/api.py'))
    if len(candidates)!=1:pytest.skip('One local installed SDK source required for ABI inspection')
    utils=sdk/'exts/isaacsim.core.utils/isaacsim/core/utils'
    result=mission.inspect_installed_tensor_sdk(candidates[0],[utils/backend/'tensor.py' for backend in ('numpy','torch','warp')])
    assert result['no_Kit_import_or_physics_view_or_control_call'] and len(result['input_sha256'])==4
    assert result['actual_motion_authorized'] is False


def profile(mode='front_only', direction=-1, curvature=False):
    x = np.linspace(0., .2, 101)
    k = .55 * np.sin(2. * math.pi * x / x[-1]) if curvature else np.zeros_like(x)
    k[0] = k[-1] = 0.
    pose = np.zeros(3)
    poses = [pose.copy()]
    arm = {'front_only': .645, 'rear_only': -.645, 'coordinated_counterphase': 0.}[mode]
    for left, right, dx in zip(k[:-1], k[1:], np.diff(x)):
        ds = direction * dx
        middle = pose[2] + .25 * (left + right) * ds
        kmid = .5 * (left + right)
        pose[:2] += ds * np.array([math.cos(middle) - arm * kmid * math.sin(middle),
            math.sin(middle) + arm * kmid * math.cos(middle)])
        pose[2] += kmid * ds
        poses.append(pose.copy())
    poses = np.array(poses)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
    return dict(mode=mode, direction=direction, poses_local_xyyaw=poses.tolist(),
        center_arc_m=arc.tolist(), curvature_1_m=k.tolist(), next_cusp_local_xyyaw=poses[-1].tolist())


@pytest.mark.parametrize('mode', timed.MODES)
@pytest.mark.parametrize('direction', [-1, 1])
def test_same_timing_and_command_limits_for_each_existing_mode_and_direction(mode, direction):
    p = profile(mode, direction, curvature=True)
    before = copy.deepcopy(p)
    leg = timed.SpatialLeg(p)
    timing = timed.LegTiming(leg)
    assert all(timing.checks.values())
    assert p == before
    assert max(timing.speed) <= .05
    assert timing.speed[0] == timing.speed[-1] == 0.
    assert max(abs(timing.acceleration)) <= .020 + 1e-12
    assert max(timing.rate) <= math.radians(8.) + 1e-12
    if mode != 'coordinated_counterphase':
        assert leg.longitudinal[-1] < leg.arc[-1]
    assert timing.time[-1] > 0.
    controller = timed.TimedLegController(p, 0.)
    out = controller.step(.02, leg.poses[0], 0., (0., 0.))
    assert direction * out['speed_m_s'] > 0.  # no zero-profile launch deadlock
    assert abs(out['speed_m_s']) <= .020 * .02 + 1e-12
    assert type(out['ready']) is bool
    json.dumps(out, allow_nan=False)


def test_method_label_cannot_select_different_limits_or_commands():
    outputs = []
    for label in ('B0', 'B1', 'B2', 'P'):
        p = dict(profile(), method=label)
        outputs.append(timed.TimedLegController(p, 0.).step(.02, p['poses_local_xyyaw'][0], 0., (0., 0.)))
    assert all(out == outputs[0] for out in outputs)
    contract = timed.execution_contract()
    assert contract['method_identifier_used'] is False
    assert contract['same_executor_and_limits_required_for_all_future_baselines'] is True
    assert contract['old_results_preserved_do_NOT_pool_different_executor_versions'] is True


@pytest.mark.parametrize('damage', ['mode', 'direction', 'nan', 'arc', 'goal', 'reverse_length', 'end_k'])
def test_invalid_or_silently_changed_reference_is_rejected(damage):
    p = profile()
    if damage == 'mode': p['mode'] = 'differential'
    elif damage == 'direction': p['direction'] = True
    elif damage == 'nan': p['poses_local_xyyaw'][1][0] = float('nan')
    elif damage == 'arc': p['center_arc_m'][2] += .001
    elif damage == 'goal': p['next_cusp_local_xyyaw'][0] += .001
    elif damage == 'reverse_length': p['center_arc_m'][-1] = .501
    elif damage == 'end_k': p['curvature_1_m'][-1] = .1
    with pytest.raises(ValueError): timed.SpatialLeg(p)


def test_braking_keeps_frozen_curvature_instead_of_cutting_curve_to_zero():
    p = profile(curvature=True)
    controller = timed.TimedLegController(p, 0.)
    controller.reference.progress = controller.reference.arc[-1] - .04
    pose, k = controller.reference.at(controller.reference.progress)
    controller.speed = -.05
    out = controller.step(.02, pose, .05, tuple(timed.steering_angles(k, 'front_only')))
    assert out['phase'] == 'terminal_braking'
    assert out['curvature_1_m'] == pytest.approx(k)
    assert abs(k) > .1
    assert out['speed_m_s'] == pytest.approx(-.0496)
    assert out['ready'] is False


def test_neutral_and_arrival_are_required_before_single_hold_and_restart_is_forbidden():
    p = profile()
    controller = timed.TimedLegController(p, 0.)
    controller.phase = 'terminal_braking'
    controller.braking_started = 0.
    controller.reference.progress = controller.reference.arc[-1] - .01
    pose, _ = controller.reference.at(controller.reference.progress)
    out = controller.step(.02, pose, .001, (math.radians(6.), 0.))
    assert out['phase'] == 'neutral_settle' and not out['ready']
    assert out['curvature_1_m'] == 0. and out['speed_m_s'] == 0.
    out = controller.step(.04, pose, .001, (0., 0.))
    assert out['ready'] is True
    assert out['terminal_position_error_m'] < .030
    with pytest.raises(RuntimeError, match='cannot restart'):
        controller.step(.06, pose, .001, (0., 0.))


def test_same_direction_reapproach_is_bounded_and_does_not_recapture_goal():
    p = profile()
    controller = timed.TimedLegController(p, 0.)
    fixed_goal = controller.reference.goal.copy()
    controller.phase = 'neutral_settle'
    controller.braking_started = 0.
    controller.reference.progress = controller.reference.arc[-1] - .04
    pose, _ = controller.reference.at(controller.reference.progress)
    out = controller.step(.02, pose, 0., (0., 0.))
    assert out['terminal_reapproaches'] == 1 and out['phase'] == 'tracking'
    assert np.array_equal(controller.reference.goal, fixed_goal)
    controller.phase = 'neutral_settle'
    controller.braking_started = .02
    with pytest.raises(RuntimeError, match='reapproach exhausted'):
        controller.step(.04, pose, 0., (0., 0.))


@pytest.mark.parametrize('damage', ['clock', 'gap', 'speed', 'tracking', 'timeout'])
def test_existing_readiness_and_safety_guards_are_not_relaxed(damage):
    p = profile()
    controller = timed.TimedLegController(p, 0.)
    pose = np.array(p['poses_local_xyyaw'][0])
    time, speed = .02, 0.
    if damage == 'clock': time = -.01
    elif damage == 'gap': time = .251
    elif damage == 'speed': speed = float('nan')
    elif damage == 'tracking': pose[1] += .081
    elif damage == 'timeout': controller.started = -91.
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        controller.step(time, pose, speed, (0., 0.))


def test_prototype_is_not_a_cics_or_isaac_motion_adapter():
    source = Path(timed.__file__).read_text()
    tree = ast.parse(source)
    calls = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not calls.intersection({'play', 'pause', 'stop', 'set_world_pose', 'set_joint_positions', 'set_gains', 'apply_action', 'Save'})
    contract = timed.execution_contract()
    assert contract['CICS_input_used'] is False
    assert contract['spatial_feedback_or_yaw_controller_added'] is False
    assert contract['runtime_adapter_prepared'] is False
    assert contract['automatic_motion_authorized'] is False


def test_all_eight_saved_design_cases_preserve_old_hold_and_do_not_authorize_driving():
    path = evidence.GATE / 'hospital_h3b_shared_timed_executor/review_r02/timed_executor_review.json'
    report = json.loads(path.read_bytes())
    evidence.verify_files(report['input_sha256'])
    expected = {(d, r, s) for d in evidence.previous.STEPS for r in evidence.previous.RESIDUALS for s in evidence.previous.SCALES}
    actual = {(c['dt_s'], c['residual_case'], c['SYNTHETIC_body_target_scale']) for c in report['cases']}
    assert len(report['cases']) == 8 and expected == actual
    assert report['old_all_second_reverse_model_cases_pass'] is False
    assert report['all_timed_executor_design_cases_pass'] is True
    assert report['old_reference_unchanged'] is True
    assert report['new_suffix_spatial_replans_NOT_used'] is True
    assert report['reused_cases_for_design_NOT_independent_validation'] is True
    assert report['automatic_motion_authorized'] is False
    assert report['runtime_adapter_prepared'] is False
    assert report['PhysX_executed'] is False
    assert report['CICS_input_used'] is False
    assert len(report['retained_all_joint_positions_rad']) == 12
    for case in report['cases']:
        assert all(case['checks'].values())
        assert case['fixed_next_cusp_local_xyyaw'] == report['fixed_second_reverse']['next_cusp_local_xyyaw']
        assert case['model_parameters'] == evidence.previous.model_bounds()
        assert case['both_changes_are_combined_NOT_a_separate_causal_ablation'] is True
        assert case['caster_stiction_captured_hold_and_measured_handoff_NOT_modelled'] is True
        evidence.verify_files({case['trace_file']: case['trace_sha256']})
    old = evidence.previous.load_review(evidence.OLD_REVIEW)
    assert old['all_second_reverse_model_cases_pass'] is False
