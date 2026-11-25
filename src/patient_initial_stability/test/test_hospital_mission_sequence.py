"""Offline saved-evidence/packet tests; never start Isaac or a trial."""

import asyncio
import copy
import math
from pathlib import Path

import pytest

import hospital_mission_sequence as sequence
import hospital_mission_execution as mission
import hospital_timed_path_execution as timed
from test_hospital_timed_path_execution import mission_fixture


def test_feedback_slew_only_changes_desired_target_rate_not_state_or_original_clock():
    from test_hospital_timed_path_execution import _sdk_fixture
    backend,adapter=_sdk_fixture();backend.paused=True
    backend.physical_q[0,adapter.steering]=[.014,.46]
    physical=backend.physical_q.copy()
    c=sequence.FeedbackHeldNeutralResponseProbe([0.,0.,0.])
    b=sequence.FeedbackHeldNeutralBridge(c,adapter,guard=lambda *a:None)
    b.steering_command=sequence.prime_retained_steering_targets(adapter)
    previous=b.steering_command;previous_rate=0.;previous_stamp=b.last_stamp
    for i in range(1,220):
        stamp=1.+i*.05
        info=b.step(stamp,[0.,0.,0.],0.)
        target=backend.data['positions'][0,adapter.steering].copy()
        if info['freeze_steering_targets']:assert (target==previous).all()
        else:
            assert max(abs(a-d) for a,d in zip(target,previous))<=math.radians(4.)*(stamp-previous_stamp)+1e-7
            assert b.held_rate<=previous_rate+math.radians(1.)*(stamp-previous_stamp)+1e-12
        assert (backend.physical_q==physical).all() and c.previous==stamp
        assert not any(backend.data['velocities'][0,adapter.drive])
        previous=target;previous_rate=b.held_rate;previous_stamp=stamp
    assert sequence._feedback_held_steering_implementation().feedback_binding['undo_recovers_entire_original_AST']
    assert adapter.restore_original()['restored']


@pytest.mark.parametrize('speed,lead,expected',[(0.,0.,True),(.0035,0.,False),(.0049,0.,False),(0.,.05,False)])
def test_feedback_slew_stops_accumulating_targets_near_speed_or_steering_error(speed,lead,expected):
    from test_hospital_timed_path_execution import _sdk_fixture
    _,adapter=_sdk_fixture();c=sequence.FeedbackHeldNeutralResponseProbe([0.,0.,0.])
    b=sequence.FeedbackHeldNeutralBridge(c,adapter,guard=lambda *a:None)
    b.steering_command=(0.,.4);b.held_rate=math.radians(4.)
    dt=b.feedback_interpolation_dt(b.last_stamp+.05,speed,(0.,.4+lead),dict(drive_hold_requested=True))
    assert (dt>0.) is expected
    assert dt<=.05/2+1e-10


@pytest.mark.parametrize('fault',[None,'preflight','guard','cancel','hold_partial','restoration'])
def test_feedback_response_SOFTWARE_bounded_same_guards_and_cleanup(fault):
    from test_hospital_mission_lifecycle import SoftwareHost
    class Host(SoftwareHost):
        async def preflight(self):
            result=await super().preflight();self.raw={'runtime_motor_baseline':self.backend.baseline}
            sequence.bind_compliant_hold_adapter(self);self.actuator=self.adapter
            return result
    h=Host(fault=fault);h.adapter=h.actuator;h.backend.paused=True
    h.backend.physical_q[0,h.actuator.steering]=[.014,.46]
    h.command_readback=lambda:dict(actual_backend_hold_ACK=h.actuator.confirmation(h.clock))
    r=asyncio.run(sequence.observe_held_neutral_once(h,sequence.FeedbackHeldNeutralResponseProbe([0.,0.,0.]),authorized=True))
    assert h.events.count('actual_Pause')==h.events.count('durable_save_fixture')==1
    assert r['new_route_trials']==0 and not any(x['command_speed_m_s'] for x in h.saved_rows)
    assert r['verdict']==('PASS' if fault is None else 'HOLD'),r


def test_feedback_definition_keeps_latest_HOLD_exact_snapshot_and_no_route(monkeypatch):
    p=sequence.retained.GATE/'hospital_h6_experiments/cohort_15_held_static_r01/route/P_dynamics_full_envelope/compliant_held_neutral_response_probe_r01/probe_readback.json'
    writes={}
    exists=Path.exists;output=p.parent.parent/'feedback_held_neutral_response_probe_r01'
    monkeypatch.setattr(Path,'exists',lambda self:False if self==output else exists(self))
    monkeypatch.setattr(sequence.retained.evidence,'write_exclusive',lambda path,value:writes.update({str(path):value}))
    monkeypatch.setattr(sequence.lifecycle,'freeze_workflow_sources',lambda *a:None)
    target=sequence.export_feedback_held_neutral_probe(p,approval=sequence.ITERATIVE_FIX_APPROVAL)
    packet=writes[str(target)];raw=sequence.retained.read(p)
    assert packet['required_tensor_snapshot']==raw['actual_final_readback']['tensor']
    assert packet['required_PAUSED_timeline']['current_time_s']==pytest.approx(266.61666666675)
    assert packet['maximum_route_trials']==0 and not packet['actual_route_authorized']
    assert raw['verdict']=='HOLD' and packet['preceding_HOLD_preserved']
    with pytest.raises(ValueError):sequence.export_feedback_held_neutral_probe(p,approval=sequence.COMPLIANT_HOLD_APPROVAL)


def test_changed_feedback_export_uses_own_timeout_boundary_NOT_earlier_speed_HOLD(monkeypatch):
    p=sequence.retained.GATE/'hospital_h6_experiments/cohort_15_held_static_r01/route/P_dynamics_full_envelope/feedback_held_neutral_response_probe_r01/probe_readback.json'
    writes={};exists=Path.exists;output=p.parent.parent/'feedback_held_neutral_response_probe_r02'
    monkeypatch.setattr(Path,'exists',lambda self:False if self==output else exists(self))
    monkeypatch.setattr(sequence.retained.evidence,'write_exclusive',lambda path,value:writes.update({str(path):value}))
    monkeypatch.setattr(sequence.lifecycle,'freeze_workflow_sources',lambda *a:None)
    target=sequence.export_feedback_held_neutral_probe(p,approval=sequence.ITERATIVE_FIX_APPROVAL,revision=2)
    packet=writes[str(target)];raw=sequence.retained.read(p)
    assert packet['required_tensor_snapshot']==raw['actual_final_readback']['tensor']
    assert packet['required_PAUSED_timeline']['current_time_s']==pytest.approx(284.16666666674)
    assert packet['probe_contract']['bridge_binding']['maximum_held_target_lead_deg']==2.5
    assert packet['probe_contract']['bridge_binding']['held_max_target_rate_deg_s']==4.
    assert packet['maximum_route_trials']==0 and not packet['actual_route_authorized']


def test_feedback_target_unwind_removes_servo_error_before_original5mm_abort_without_reset():
    from test_hospital_timed_path_execution import _sdk_fixture
    backend,adapter=_sdk_fixture();c=sequence.FeedbackHeldNeutralResponseProbe([0.,0.,0.])
    b=sequence.FeedbackHeldNeutralBridge(c,adapter,guard=lambda *a:None)
    b.steering_command=(0.,.42);b.feedback_previous_speed=.001
    physical=backend.physical_q.copy()
    steer,wheels=b.feedback_targets(b.last_stamp+.05,.002,(0.,.45),dict(speed_m_s=0.,mode='coordinated_counterphase',curvature_1_m=0.,drive_hold_requested=True))
    assert .42<steer[1]<=.42+math.radians(8.)*.05+1e-12
    assert not any(wheels) and (backend.physical_q==physical).all() and not backend.writes


def test_feedback_capture_settling_is_not_fixed_stopped_window_and_no_window_restart():
    c=sequence.FeedbackHeldNeutralResponseProbe([0.,0.,0.])
    c.step(0.,[0.,0.,0.],0.,(0.,.45),drive_hold_confirmed=False)
    c.step(.25,[0.,0.,0.],0.,(0.,.45),drive_hold_confirmed=False)
    c.step(.30,[0.,0.,0.],.002,(0.,.45),drive_hold_confirmed=True)
    assert c.held_since==.30
    for i in range(1,20):
        info=c.step(.30+i*.05,[0.,0.,0.],.006,(0.,.45),drive_hold_confirmed=True)
        assert info['terminal_stage']=='captured_steering_hold' and info['freeze_steering_targets']
        assert c.held_since==.30
    with pytest.raises(RuntimeError,match='5mm/s'):
        c.step(1.31,[0.,0.,0.],.005001,(0.,.45),drive_hold_confirmed=True)
    assert c.held_since==.30 and c.stage=='held_neutral'


def test_actual_stop_connection_is_NE_and_geometry_verified_forward_and_neutral_NOT_live_success():
    path,review,plan,geometry=sequence.stop_connection_plan_inputs()
    assert path.exists() and review.exists() and plan['actual_motion_authorized'] is False
    assert plan['use_for_final_statistics'] is False
    assert all(a['direction']==1 for a in plan['selected_actions'])
    assert plan['selected_actions'][-1]['curvature_1_m']==0.
    assert max(abs(plan['selected_states'][-1][k]) for k in ('front_rad','rear_rad'))<=math.radians(2.5)
    mission.validate_mission_plan(plan,geometry)
    assert sequence.retained.read(review)['original_full_NE_and10mm_SAT_audit']['verdict']=='PASS'


def test_latest_failed_probe_can_only_supply_new_retirement_archive_NOT_original_PASS_condition():
    p=sequence.retained.GATE/'hospital_h6_experiments/cohort_15_held_static_r01/route/P_dynamics_full_envelope/feedback_held_neutral_response_probe_r04/probe_readback.json'
    value=sequence.saved_response_probe_preparation_handoff(p,archive_failed=True)
    assert value['status']=='EXACT_SAVED_FAILED_RESPONSE_BOUNDARY_FOR_NEW_ARCHIVE_NOT_PASS'
    assert value['original_verdict']=='HOLD' and value['new_live_actions']==0
    assert value['SDK_GET']['timeline']['current_time_s']==pytest.approx(305.83333333338)
    with pytest.raises(ValueError):sequence.saved_response_probe_preparation_handoff(p)
    assert sequence.lifecycle.checked_fresh_held_approval(sequence.ITERATIVE_FIX_APPROVAL)==sequence.ITERATIVE_FIX_APPROVAL


def test_new_iterative_authority_selects_same_original_directional_controller_not_failed_gain_slew():
    plan,geometry=mission_fixture()
    packet=dict(simulation_budget_s=600.,method='P_dynamics_full_envelope',
        explicit_route_approval=copy.deepcopy(sequence.ITERATIVE_FIX_APPROVAL),
        common_tracking_policy=sequence.DIRECTIONAL_QUIET_STOP_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        common_lifecycle_contract=sequence.directional_quiet_stop_lifecycle_contract(simulation_budget_s=600.))
    controller=sequence.route_controller(packet,plan,geometry)
    assert type(controller) is sequence.DirectionalQuietStopLifecycleCandidate
    assert controller.contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']
    packet['common_tracking_policy']=sequence.COMPLIANT_HOLD_POLICY
    with pytest.raises(ValueError):sequence.route_controller(packet,plan,geometry)


def test_current_iterative_authority_passes_REAL_host_scope_gate_SOFTWARE_SDK_only(tmp_path,monkeypatch):
    from test_hospital_mission_lifecycle import retained_host_fixture
    f=retained_host_fixture(tmp_path,monkeypatch)
    packet=copy.deepcopy(f.protocol)
    packet.update(explicit_route_approval=copy.deepcopy(sequence.ITERATIVE_FIX_APPROVAL),method='P_dynamics_full_envelope',
        common_tracking_policy=sequence.DIRECTIONAL_QUIET_STOP_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        common_lifecycle_contract=sequence.directional_quiet_stop_lifecycle_contract(simulation_budget_s=600.))
    target=tmp_path/'SOFTWARE_ONLY_iterative_authority.json'
    sequence.retained.evidence.write_exclusive(target,packet)
    old=f.host
    host=sequence.lifecycle.RetainedMissionSDKHost(old.sdk,packet,target,context=old.context,plan=old.plan,geometry=old.geometry,
        occupied=old.runtime_guard.occupied,origin=old.runtime_guard.origin,resolution=.010,authorized=True)
    assert host.adapter is None and not f.writes and not any(v in f.events for v in ('Play','Stop'))


@pytest.mark.parametrize('fault',[None,'preflight','guard','cancel','hold_partial','restoration'])
def test_final_target_hold_probe_SOFTWARE_never_moves_steering_or_promotes_old_route(fault):
    from test_hospital_mission_lifecycle import SoftwareHost
    h=SoftwareHost(fault=fault);h.adapter=h.actuator;h.backend.paused=True
    h.backend.physical_q[0,h.actuator.steering]=[-.013,-.65]
    original_measure=h.measure
    from hospital_static_gate import WHEEL_LABELS
    def SOFTWARE_complete_static_channels():
        row=original_measure();row.update(z_m=0.,roll_deg=0.,pitch_deg=0.,yaw_deg=math.degrees(row['yaw_rad']))
        row.update({label+'_fz_n':row[f'wheel_{i}_fz_n'] for i,label in enumerate(WHEEL_LABELS)})
        return row
    h.measure=SOFTWARE_complete_static_channels
    h.command_readback=lambda:dict(actual_backend_hold_ACK=h.actuator.confirmation(h.clock))
    c=sequence.RetainedFinalTargetHoldProbe([0.,0.,0.],[0.,0.,0.])
    r=asyncio.run(sequence.observe_held_neutral_once(h,c,authorized=True))
    assert r['new_route_trials']==0 and h.events.count('actual_Pause')==1
    assert not any(row['command_speed_m_s'] for row in h.saved_rows)
    if fault is None:
        assert r['verdict']=='PASS',r
        assert r['role']=='RETAINED_FINAL_TARGET_HOLD_NOT_H6_ROUTE_SUCCESS'
        assert r['retained_terminal_hold_component']['NOT_H6_whole_route_success']
        assert c.held_since is not None and h.clock-c.held_since>=5.
        assert h.actuator.capture_count==1 and r['restoration']['restored']
    else:assert r['verdict']=='HOLD'


def test_final_target_hold_export_captures_current_goal_reaching_route_HOLD_ONLY(monkeypatch):
    p=sequence.retained.GATE/'hospital_h6_experiments/cohort_16_held_static_r01/route/P_dynamics_full_envelope/route_readback.json'
    writes={};exists=Path.exists;output=p.parent/'final_target_hold_response_probe_r01'
    monkeypatch.setattr(Path,'exists',lambda self:False if self==output else exists(self))
    monkeypatch.setattr(sequence.retained.evidence,'write_exclusive',lambda path,value:writes.update({str(path):value}))
    monkeypatch.setattr(sequence.lifecycle,'freeze_workflow_sources',lambda *a:None)
    target=sequence.export_final_target_hold_probe(p,approval=sequence.ITERATIVE_FIX_APPROVAL)
    packet=writes[str(target)];raw=sequence.retained.read(p)
    assert packet['required_tensor_snapshot']==raw['actual_final_readback']['tensor']
    assert packet['required_PAUSED_timeline']['current_time_s']==pytest.approx(326.9666666667)
    assert packet['maximum_route_trials']==0 and packet['previous_route_HOLD_NEVER_promoted']
    assert packet['probe_contract']['original_fixed_tail_speed_m_s']==.005
    assert packet['probe_contract']['fixed_hold_s']==5. and packet['probe_contract']['fixed_tail_s']==3.


@pytest.mark.parametrize('bad',['none','speed','position','heading','ACK'])
def test_final_target_lifecycle_changes_only_unnecessary_goal_neutralization_not_body_acceptance(bad):
    plan,g=mission_fixture();c=sequence.FinalTargetHoldLifecycleCandidate(plan,g,simulation_budget_s=600.)
    c.lifecycle_stage='route';m=c.motion;m.stage='final_target_hold';m.previous_time=0.;m.started=0.;m.stage_started=0.
    m.speed=0.;m.curvature=.7;m.mode='rear_only';m.reference.progress=m.reference.arc[-1]
    pose=list(c.goal);speed=0.;ack=True
    if bad=='speed':speed=.005001
    elif bad=='position':pose[0]+=.030001
    elif bad=='heading':pose[2]+=math.radians(2.001)
    elif bad=='ACK':ack=False
    if bad in ('speed','position','heading'):
        with pytest.raises(RuntimeError):c.step(.05,pose,speed,(.01,-.65),drive_hold_confirmed=ack)
    else:
        info=c.step(.05,pose,speed,(.01,-.65),drive_hold_confirmed=ack)
        assert info['freeze_steering_targets'] and m.curvature==.7
        assert c.fixed_hold_attempts==(1 if ack else 0)
        assert c.contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']


def test_final_target_bridge_does_not_fake_actual_steering_or_write_a_new_angle_when_frozen():
    from test_hospital_timed_path_execution import _sdk_fixture
    backend,adapter=_sdk_fixture();backend.paused=True;backend.physical_q[0,adapter.steering]=[.01,-.65]
    c=sequence.RetainedFinalTargetHoldProbe([0.,0.,0.],[0.,0.,0.])
    b=sequence.RetainFinalTargetsBridge(c,adapter,guard=lambda *a:None)
    b.steering_command=sequence.prime_retained_steering_targets(adapter)
    before=backend.data['positions'][0,adapter.steering].copy()
    for i in range(1,35):
        b.step(1.+i*.05,[0.,0.,0.],0.)
        assert (backend.data['positions'][0,adapter.steering]==before).all()
    assert adapter.measured_steering()[1]==pytest.approx(-.65)
    assert adapter.restore_original()['restored']


def test_final_target_lifecycle_full_SOFTWARE_original5s_acceptance_and_cleanup():
    from test_hospital_mission_lifecycle import SoftwareHost
    h=SoftwareHost();h.adapter=h.actuator
    plan,g=mission_fixture((1,)*20)
    c=sequence.FinalTargetHoldLifecycleCandidate(plan,g,simulation_budget_s=600.,start_time_s=h.clock)
    r=asyncio.run(sequence.lifecycle.run_lifecycle_once(h,c,authorized=True,sdk_bridge_class=sequence.RetainFinalTargetsBridge))
    assert r['verdict']=='PASS',r
    assert c.fixed_hold_attempts==1 and h.events.count('actual_Pause')==1
    assert r['restoration']['restored']
    assert c.contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']


def test_final_target_protocol_requires_exact_new_scope_and_contract():
    p,g=mission_fixture()
    packet=dict(method='P_dynamics_full_envelope',simulation_budget_s=600.,explicit_route_approval=sequence.ITERATIVE_FIX_APPROVAL,
        common_tracking_policy=sequence.FINAL_TARGET_HOLD_POLICY,common_forward_tracking_floor_m_s=.010,
        common_reverse_tracking_floor_m_s=.020,common_lifecycle_contract=sequence.final_target_hold_lifecycle_contract(simulation_budget_s=600.))
    assert type(sequence.route_controller(packet,p,g)) is sequence.FinalTargetHoldLifecycleCandidate
    packet['explicit_route_approval']=sequence.DIRECTIONAL_QUIET_STOP_APPROVAL
    with pytest.raises(ValueError):sequence.route_controller(packet,p,g)


@pytest.mark.parametrize('method',sequence.METHODS)
def test_H3_whole_task_selects_identical_final_controller_for_every_method(method):
    from hospital_path_methods import method_settings
    p,g=mission_fixture();g['scenario']=p['scenario']='H3_doorway_immediate_turn'
    p['method']=method;p['planner_dynamics']=method_settings(method)['planner_dynamics']
    if method=='B0_fixed_counterphase_ramp':p['allowed_modes']=['coordinated_counterphase']
    packet=dict(scenario=g['scenario'],method=method,simulation_budget_s=300.,
        explicit_route_approval=sequence.H3_CONTINUOUS_APPROVAL,
        common_tracking_policy=sequence.FINAL_TARGET_HOLD_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        common_lifecycle_contract=sequence.final_target_hold_lifecycle_contract(simulation_budget_s=300.))
    c=sequence.route_controller(packet,p,g)
    assert type(c) is sequence.FinalTargetHoldLifecycleCandidate
    assert sequence.lifecycle.checked_fresh_held_approval(sequence.H3_CONTINUOUS_APPROVAL)==sequence.H3_CONTINUOUS_APPROVAL
    packet['scenario']='H6_integrated_transfer_mission'
    with pytest.raises(ValueError):sequence.route_controller(packet,p,g)


@pytest.mark.parametrize('fault',['goal','scenario','clearance','speed'])
def test_H3_schema_extension_does_not_relax_task_or_motion_bounds(fault):
    p,g=mission_fixture();g['scenario']=p['scenario']='H3_doorway_immediate_turn'
    if fault=='goal':g['goal_xyyaw']=[1.,0.,0.]
    elif fault=='scenario':p['scenario']='H6_integrated_transfer_mission'
    elif fault=='clearance':g['full_body_clearance_m']=.009
    else:p['common_max_command_speed_m_s']=.051
    with pytest.raises(ValueError):mission.validate_mission_plan(p,g)


def test_H3_historical_schema_extension_cannot_hide_any_controller_change():
    after=Path(mission.__file__).read_text()
    new="geometry.get('scenario') not in ('H6_integrated_transfer_mission','H3_doorway_immediate_turn','H1_bedside_departure','H2_bedside_lateral_alignment','H4_ward_corridor_turn','H5_obstacle_bypass_return')"
    old="geometry.get('scenario')!='H6_integrated_transfer_mission'"
    before=after.replace(new,old)
    assert before!=after
    proof=sequence.lifecycle.reviewed_mission_schema_extension(before,after)
    assert proof['whole_original_controller_and_all_guards_AST_identical']
    with pytest.raises(ValueError):
        sequence.lifecycle.reviewed_mission_schema_extension(before,after+'\nUNREVIEWED_CONTROL_CHANGE=True\n')


@pytest.mark.parametrize('task',tuple(sequence.TASK_SCENARIOS))
def test_remaining_tasks_reuse_controller_schema_but_keep_exact_task_budget(task):
    import numpy as np
    p,g=mission_fixture();g['scenario']=p['scenario']=sequence.TASK_SCENARIOS[task]
    mission.validate_mission_plan(p,g)
    budget=sequence.TASK_BUDGETS[task]
    from test_hospital_mission_lifecycle import guard_fixture
    g['same_three_authored_support_rectangles']=guard_fixture().supports
    guard=sequence.lifecycle.MissionRuntimeGuard(g,np.zeros((10,10),bool),[-1.,-1.],.010,
        simulation_budget_s=budget)
    assert guard.budget==budget
    with pytest.raises(ValueError):
        sequence.lifecycle.MissionRuntimeGuard(g,np.zeros((10,10),bool),[-1.,-1.],.010,
            simulation_budget_s=budget+60.)
    packet=dict(scenario=g['scenario'],method=p['method'],simulation_budget_s=budget,
        explicit_route_approval=sequence.REMAINING_NO_NAV2_APPROVAL,
        common_tracking_policy=sequence.FINAL_TARGET_HOLD_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020)
    assert sequence.checked_route_lifecycle_contract(packet)==sequence.final_target_hold_lifecycle_contract(simulation_budget_s=budget)
    packet['explicit_route_approval']=dict(scope='NAV2')
    with pytest.raises(ValueError):sequence.checked_route_lifecycle_contract(packet)


@pytest.mark.parametrize('measured_speed',[0.,.003,.005,.005001])
def test_compliant_hold_OFFLINE_candidate_changes_only_two_P_gains_and_keeps_exact_restore(measured_speed):
    import numpy as np
    from test_hospital_timed_path_execution import _sdk_fixture
    backend,old=_sdk_fixture();baseline=copy.deepcopy(backend.data);physical=backend.physical_q.copy()
    adapter=sequence.CompliantHoldDriveAdapter(backend,backend.baseline,encode=old.encode,
        check_context=old.check_context,paused_check=old.paused_check,physics_stamp=1.,authorized=True)
    assert not backend.writes
    if measured_speed>.005:
        with pytest.raises(RuntimeError,match='measured stop'):
            adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=measured_speed)
        assert not backend.writes;return
    adapter.request_hold(True,1.,command_speed_m_s=0.,measured_speed_m_s=measured_speed)
    assert np.array_equal(backend.physical_q,physical)
    assert np.array_equal(backend.data['kp'][0,adapter.drive],np.array([1000.,1000.]))
    for k in ('kd','limits'):assert np.array_equal(backend.data[k],baseline[k])
    for k in backend.data:assert np.array_equal(backend.data[k][0,adapter.passive],baseline[k][0,adapter.passive])
    assert adapter.confirmation(1.) is None and adapter.confirmation(1.1) is True
    adapter.request_hold(False,1.1,command_speed_m_s=0.,measured_speed_m_s=0.)
    assert adapter.confirmation(1.2) is False and np.array_equal(backend.data['kp'],baseline['kp'])
    backend.paused=True;assert adapter.restore_original()['restored']
    assert all(np.array_equal(backend.data[k][0],v) for k,v in adapter.original.items())
    binding=adapter._compliant_hold.compliance_binding
    assert binding['undo_recovers_entire_original_AST'] and binding['OFFLINE_candidate_NOT_wired_to_a_live_protocol']


@pytest.mark.parametrize('fault',[None,'preflight','guard','cancel','hold_partial','restoration'])
def test_compliant_response_SOFTWARE_fixed_capture_window_and_complete_original_cleanup(fault):
    from test_hospital_mission_lifecycle import SoftwareHost
    class Host(SoftwareHost):
        async def preflight(self):
            result=await super().preflight()
            self.raw={'runtime_motor_baseline':self.backend.baseline}
            sequence.bind_compliant_hold_adapter(self);self.actuator=self.adapter
            return result
    h=Host(fault=fault);h.adapter=h.actuator;h.backend.paused=True
    h.backend.physical_q[0,h.actuator.steering]=[.014,.461]
    h.command_readback=lambda:dict(actual_backend_hold_ACK=h.actuator.confirmation(h.clock),
        front_target=float(h.actuator.expected['positions'][h.actuator.steering[0]]),
        rear_target=float(h.actuator.expected['positions'][h.actuator.steering[1]]))
    c=sequence.CompliantHeldNeutralResponseProbe([0.,0.,0.])
    r=asyncio.run(sequence.observe_held_neutral_once(h,c,authorized=True))
    assert h.events.count('actual_Pause')==h.events.count('durable_save_fixture')==1
    assert r['new_route_trials']==0 and not any(x['command_speed_m_s'] for x in h.saved_rows)
    if fault is None:
        assert r['verdict']=='PASS',r
        held=[x for x in h.saved_rows if x['phase']=='captured_steering_hold']
        assert len(held)>=20 and max(x['rear_target'] for x in held)==min(x['rear_target'] for x in held)
        assert h.actuator.capture_count==1 and r['restoration']['restored']
        assert all((h.backend.data[k][0]==v).all() for k,v in h.actuator.original.items())
    else:assert r['verdict']=='HOLD'


def test_compliant_definition_uses_current_consumed_probe_not_stale_route_pose_SOFTWARE_export_only(monkeypatch):
    p=sequence.retained.GATE/'hospital_h6_experiments/cohort_15_held_static_r01/route/P_dynamics_full_envelope/held_neutral_response_probe_r01/probe_readback.json'
    writes={}
    exists=Path.exists;output=p.parent.parent/'compliant_held_neutral_response_probe_r01'
    monkeypatch.setattr(Path,'exists',lambda self:False if self==output else exists(self))
    monkeypatch.setattr(sequence.retained.evidence,'write_exclusive',lambda path,value:writes.update({str(path):value}))
    monkeypatch.setattr(sequence.lifecycle,'freeze_workflow_sources',lambda *a:None)
    path=sequence.export_compliant_held_neutral_probe(p,approval=sequence.COMPLIANT_HOLD_APPROVAL)
    packet=writes[str(path)];raw=sequence.retained.read(p)
    assert packet['required_PAUSED_timeline']==raw['actual_final_timeline']
    assert packet['required_tensor_snapshot']==raw['actual_final_readback']['tensor']
    assert packet['required_PAUSED_timeline']['current_time_s']==pytest.approx(264.)
    assert packet['actual_route_authorized'] is False and packet['maximum_route_trials']==0
    assert packet['probe_contract']['transient_common_hold_gain_candidate']['candidate_hold_kp_Nm_rad']==1000.
    assert packet['previous_diagnostic_boundary_review']['original_HOLD_unchanged']
    assert packet['previous_diagnostic_boundary_review']['all4997_source_environment_rows_STILL_require_new_preflight']
    with pytest.raises(ValueError):sequence.export_compliant_held_neutral_probe(p,approval=sequence.HELD_NEUTRAL_PROBE_APPROVAL)


def compliant_route_packet():
    return dict(simulation_budget_s=600.,method='P_dynamics_full_envelope',
        explicit_route_approval=copy.deepcopy(sequence.COMPLIANT_HOLD_APPROVAL),
        common_tracking_policy=sequence.COMPLIANT_HOLD_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        common_lifecycle_contract=sequence.compliant_hold_lifecycle_contract(simulation_budget_s=600.))


@pytest.mark.parametrize('fault',['approval','old_policy','method','missing_policy','contract'])
def test_compliant_route_requires_new_exact_scope_and_all_original_acceptance(fault):
    packet=compliant_route_packet();plan,geometry=mission_fixture()
    if fault=='approval':packet['explicit_route_approval']=sequence.DIRECTIONAL_QUIET_STOP_APPROVAL
    elif fault=='old_policy':packet['common_tracking_policy']=sequence.DIRECTIONAL_QUIET_STOP_POLICY
    elif fault=='method':packet['method']='B1_multimode_kinematic_ramp'
    elif fault=='missing_policy':del packet['common_tracking_policy']
    else:packet['common_lifecycle_contract']['acceptance']['unreviewed']=True
    with pytest.raises(ValueError):sequence.route_controller(packet,plan,geometry)


def test_conditional_new_source_REFUSES_without_actual_compliant_response_PASS():
    with pytest.raises(ValueError,match='response PASS'):
        sequence.lifecycle.fresh_held_workflow_definition(cohort='cohort_16_held_static_r01',approval=sequence.COMPLIANT_HOLD_APPROVAL)


def test_compliant_whole_lifecycle_SOFTWARE_one_hold_Pause_restore_and_original_criteria():
    from test_hospital_mission_lifecycle import SoftwareHost
    class Host(SoftwareHost):
        async def preflight(self):
            result=await super().preflight();self.raw={'runtime_motor_baseline':self.backend.baseline}
            sequence.bind_compliant_hold_adapter(self);self.actuator=self.adapter
            return result
    h=Host();h.adapter=h.actuator
    plan,geometry=mission_fixture((1,)*20+(-1,)*5+(1,)*6)
    packet=compliant_route_packet();selected=sequence.route_controller(packet,plan,geometry)
    c=type(selected)(plan,geometry,simulation_budget_s=600.,start_time_s=h.clock)
    assert type(c) is sequence.CompliantHoldLifecycleCandidate
    r=asyncio.run(sequence.lifecycle.run_lifecycle_once(h,c,authorized=True,sdk_bridge_class=sequence.HeldNeutralSlewBridge))
    assert r['verdict']=='PASS',r
    assert c.fixed_hold_attempts==1 and h.events.count('Play')==h.events.count('actual_Pause')==1
    assert r['restoration']['restored'] and c.contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']
    assert c.contract['route_own_measured2s_prehold_required_at_candidate_gain']
    assert sequence.lifecycle.checked_fresh_held_approval(sequence.COMPLIANT_HOLD_APPROVAL)==sequence.COMPLIANT_HOLD_APPROVAL


def test_held_neutral_bridge_changes_only_target_slew_not_clock_guards_or_measured_state():
    from test_hospital_timed_path_execution import _sdk_fixture
    backend,adapter=_sdk_fixture();backend.paused=True
    backend.physical_q[0,adapter.steering]=[.014,.542]
    physical=backend.physical_q.copy();baseline={k:v.copy() for k,v in backend.data.items()}
    initial=sequence.prime_retained_steering_targets(adapter)
    assert (backend.physical_q==physical).all()
    assert (backend.data['kp']==baseline['kp']).all() and (backend.data['kd']==baseline['kd']).all()
    assert (backend.data['limits']==baseline['limits']).all()
    assert (backend.data['positions'][0,adapter.passive]==baseline['positions'][0,adapter.passive]).all()
    controller=sequence.HeldNeutralResponseProbe([0.,0.,0.])
    bridge=sequence.HeldNeutralSlewBridge(controller,adapter,guard=lambda *a:None);bridge.steering_command=initial
    previous=initial
    for i in range(1,32):
        t=1.+i*.05
        info=bridge.step(t,[0.,0.,0.],0.)
        target=backend.data['positions'][0,adapter.steering].copy()
        if info['freeze_steering_targets']:assert (target==previous).all()
        else:assert max(abs(a-b) for a,b in zip(target,previous))<=math.radians(4)*.05+1e-7
        assert controller.previous==t and not any(backend.data['velocities'][0,adapter.drive])
        previous=target
    assert (backend.physical_q==physical).all()
    assert sequence._held_neutral_slew_implementation().slew_binding['undo_recovers_entire_original_AST']
    adapter.restore_original()
    assert all((backend.data[k][0]==v).all() for k,v in adapter.original.items())


@pytest.mark.parametrize('fault',['speed','distance','clock','deadline','settle_timeout','existing_hold'])
def test_held_neutral_response_keeps_original_guards_and_has_no_second_capture(fault):
    c=sequence.HeldNeutralResponseProbe([0.,0.,0.])
    if fault=='existing_hold':
        with pytest.raises(RuntimeError):c.step(0.,[0.,0.,0.],0.,(0.,.5),drive_hold_confirmed=True)
        return
    c.step(0.,[0.,0.,0.],0.,(0.,.5),drive_hold_confirmed=False)
    if fault=='speed':
        c.step(.25,[0.,0.,0.],0.,(0.,.5),drive_hold_confirmed=False)
        with pytest.raises(RuntimeError,match='5mm/s'):c.step(.30,[0.,0.,0.],.005001,(0.,.5),drive_hold_confirmed=True)
    elif fault=='distance':
        with pytest.raises(RuntimeError,match='30mm'):c.step(.1,[.030001,0.,0.],0.,(0.,.5),drive_hold_confirmed=False)
    elif fault=='clock':
        with pytest.raises(ValueError):c.step(.251,[0.,0.,0.],0.,(0.,.5),drive_hold_confirmed=False)
    elif fault=='deadline':
        c.previous=19.9
        with pytest.raises(TimeoutError,match='20s'):c.step(20.,[0.,0.,0.],0.,(0.,.5),drive_hold_confirmed=False)
    else:
        for i in range(1,33):c.step(i*.25,[0.,0.,0.],.006,(0.,.5),drive_hold_confirmed=False)
        with pytest.raises(TimeoutError,match='8s'):c.step(8.25,[0.,0.,0.],.006,(0.,.5),drive_hold_confirmed=False)


@pytest.mark.parametrize('fault',[None,'preflight','guard','cancel','hold_partial','restoration'])
def test_zero_neutral_response_SOFTWARE_one_trial_Pause_restore_and_no_retry(fault):
    from test_hospital_mission_lifecycle import SoftwareHost
    host=SoftwareHost(fault=fault);host.adapter=host.actuator;host.backend.paused=True
    host.backend.physical_q[0,host.actuator.steering]=[.014,.542]
    host.command_readback=lambda:dict(actual_backend_hold_ACK=host.actuator.confirmation(host.clock))
    c=sequence.HeldNeutralResponseProbe([0.,0.,0.])
    r=asyncio.run(sequence.observe_held_neutral_once(host,c,authorized=True))
    assert r['new_route_trials']==0 and r['new_response_probe_trials'] in (0,1)
    assert host.events.count('actual_Pause')==host.events.count('durable_save_fixture')==1
    assert not any(x['command_speed_m_s'] for x in host.saved_rows)
    if fault is None:
        assert r['verdict']=='PASS',r
        assert c.complete and host.actuator.capture_count==1 and r['restoration']['restored']
        assert host.clock<20. and c.travel==0.
    else:assert r['verdict']=='HOLD'
    saved=copy.deepcopy(host.saved)
    again=asyncio.run(sequence.observe_held_neutral_once(host,sequence.HeldNeutralResponseProbe([0.,0.,0.]),authorized=True))
    assert again['status']=='EXISTING_PROBE_RETAINED_NO_NEW_ACTION' and host.saved==saved


def test_neutral_probe_refuses_any_prior_approval_or_implicit_live_access():
    with pytest.raises(ValueError,match='Exact new ONE'):
        sequence.export_held_neutral_response_probe('/does/not/exist',approval=sequence.REVERSE_PROBE_APPROVAL)
    with pytest.raises(ValueError,match='Explicit ONE'):
        asyncio.run(sequence.run_held_neutral_response_probe_once('/does/not/exist',authorized=False))


@pytest.mark.parametrize('fault',['hold','nonfinite','negative_speed','gap','distance','deadline'])
def test_reverse_response_probe_rejects_unsafe_or_extended_observation(fault):
    controller=sequence.ReverseResponseProbe([0.,0.,0.],.477)
    controller.step(0.,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    pose=[0.,0.,0.];speed=0.;stamp=.1;ack=False
    if fault=='hold':ack=True
    elif fault=='nonfinite':pose[0]=float('nan')
    elif fault=='negative_speed':speed=-.001
    elif fault=='gap':stamp=.251
    elif fault=='distance':pose[0]=.030001
    else:controller.started=-15.
    with pytest.raises((ValueError,RuntimeError,TimeoutError)):
        controller.step(stamp,pose,speed,(0.,0.),drive_hold_confirmed=ack)


def test_reverse_response_probe_is_one_bounded_sequence_not_retry_until_move():
    controller=sequence.ReverseResponseProbe([0.,0.,0.],.477)
    angles=tuple(timed.steering_angles(.477,'rear_only'))
    outputs=[];previous=0.
    for i in range(1000):
        info=controller.step(i*.02,[0.,0.,0.],0.,angles,drive_hold_confirmed=False)
        assert -.030<=info['speed_m_s']<=0.
        assert abs(info['speed_m_s']-previous)<=.020*.02+1e-12
        assert not info['drive_hold_requested'] and info['mode']=='rear_only'
        outputs.append(info);previous=info['speed_m_s']
        if controller.complete:break
    assert controller.complete and outputs[-1]['speed_m_s']==0.
    assert {r['requested_level_m_s'] for r in outputs}>={-.010,-.020,-.030}
    assert outputs[-1]['phase_time_s']<15.
    assert controller.contract['new_route_trials']==0 and not controller.contract['use_for_final_statistics']
    with pytest.raises(ValueError):controller.step(i*.02+.02,[0.,0.,0.],0.,angles,drive_hold_confirmed=False)


def test_reverse_response_probe_brakes_before_30mm_and_does_not_realign_casters():
    controller=sequence.ReverseResponseProbe([0.,0.,0.],.477)
    angles=tuple(timed.steering_angles(.477,'rear_only'))
    x=speed=0.;braked=False
    for i in range(1000):
        info=controller.step(i*.01,[x,0.,0.],abs(speed),angles,drive_hold_confirmed=False)
        speed=info['speed_m_s'];x+=.01*speed
        braked=braked or info['terminal_stage']=='braking'
        assert abs(x)<.030
        if controller.complete:break
    assert controller.complete and braked and abs(x)>.005
    assert controller.travel<.030


def test_reverse_response_probe_alignment_still_requires_real_standstill():
    controller=sequence.ReverseResponseProbe([0.,0.,0.],.477)
    with pytest.raises(RuntimeError,match='standstill'):
        controller.step(0.,[0.,0.,0.],.005001,(0.,0.),drive_hold_confirmed=False)


def test_reverse_response_probe_requires_separate_actual_approval_before_evidence_or_sdk():
    with pytest.raises(ValueError,match='Separate short'):
        sequence.export_reverse_response_probe('/does/not/exist',approval=sequence.TRACKING_FLOOR_APPROVAL)
    with pytest.raises(ValueError,match='Separate ONE'):
        asyncio.run(sequence.run_reverse_response_probe_once('/does/not/exist',authorized=False))


@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('method',sequence.METHODS)
def test_measured20mm_candidate_is_common_bounded_ramp_not_method_specific(method,direction):
    import hospital_h4_path_execution as original
    plan,geometry=mission_fixture((direction,)*10)
    plan.update(original.method_settings(method))
    plan['allowed_modes']=['coordinated_counterphase'] if method.startswith('B0_') else list(timed.MODES)
    for action in plan['selected_actions']:action['target_speed_m_s']=.003
    saved=copy.deepcopy((plan,geometry))
    controller=sequence.MeasuredResponseFloorCandidate(plan,geometry,simulation_budget_s=600.)
    controller.stage='tracking';controller.previous_time=controller.started=0.
    last=0.
    for i in range(1,21):
        out=controller.step(i*.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
        assert abs(out['speed_m_s']-last)<=.002+1e-12
        assert direction*out['speed_m_s']>0. and abs(out['speed_m_s'])<=.020+1e-12
        last=out['speed_m_s']
    assert abs(last)==pytest.approx(.020) and (plan,geometry)==saved
    assert controller.tracking_step.mission_binding['original_endpoint_braking_cap_and_ramp_and_feedback_and_guards_unchanged']


def test_measured20mm_candidate_has_separate_contract_and_actual_fix_approval():
    plan,geometry=mission_fixture()
    contract=sequence.measured_response_lifecycle_contract(simulation_budget_s=600.)
    packet=dict(simulation_budget_s=600.,common_tracking_policy=sequence.MEASURED_RESPONSE_FLOOR_POLICY,
        common_tracking_speed_floor_m_s=.020,explicit_route_approval=sequence.MEASURED_RESPONSE_FIX_APPROVAL,
        method='P_dynamics_full_envelope',common_lifecycle_contract=contract)
    assert type(sequence.route_controller(packet,plan,geometry)) is sequence.MeasuredResponseLifecycleCandidate
    assert contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']
    for wrong in (sequence.TRACKING_FLOOR_APPROVAL,sequence.REVERSE_PROBE_APPROVAL):
        with pytest.raises(ValueError):sequence.route_controller(dict(packet,explicit_route_approval=wrong),plan,geometry)
    assert sequence.lifecycle.checked_fresh_held_approval(sequence.MEASURED_RESPONSE_FIX_APPROVAL)==sequence.MEASURED_RESPONSE_FIX_APPROVAL


def test_real_completed_probe_handoff_is_NOT_route_or_static_PASS_admission():
    path=sequence.retained.GATE/'hospital_h6_experiments/cohort_13_held_static_r01/route/P_dynamics_full_envelope/reverse_response_probe_r01/probe_readback.json'
    value=sequence.saved_response_probe_preparation_handoff(path)
    assert value['response_probe_PASS_is_NOT_H6_mission_PASS'] and value['old_route_HOLD_unchanged']
    assert len(value['SDK_GET']['actual_tensor_snapshot']['names'])==12
    assert value['SDK_GET']['timeline']['current_time_s']==pytest.approx(373.2)
    assert value['new_live_actions']==0


def test_preparation_review_cache_requires_both_complete_texts_and_copies_its_proof():
    calls=[]
    def classify(before,after):
        calls.append((dict(before),dict(after)))
        if after.get('Root')=='MASS_EDIT':raise ValueError('Protected Root changed')
        return dict(proof=['EXACT_PAIR'])
    cached=sequence.lifecycle.memoized_exact_layer_review(classify)
    before=dict(Root='R',Session='S');after=dict(Root='R',Session='VIEW_DEFAULT')
    result=cached(before,after);result['proof'].append('EXTERNAL_MUTATION')
    for _ in range(1000):assert cached(dict(before),dict(after))==dict(proof=['EXACT_PAIR'])
    assert len(calls)==1
    cached(dict(before,Session='DIFFERENT_BASELINE'),after)
    assert len(calls)==2
    for _ in range(2):
        with pytest.raises(ValueError):cached(before,dict(after,Root='MASS_EDIT'))
    assert len(calls)==4


def test_prepared_source_cache_never_skips_actual_context_or_changed_layer_bytes(monkeypatch):
    from types import SimpleNamespace
    host=object.__new__(sequence.lifecycle.PreparedHeldStaticSDKHost)
    values=dict(Root='same-root',Session='same-session');context_checks=[]
    host.same_context=lambda:context_checks.append(True)
    host.stage=SimpleNamespace(GetRootLayer=lambda:SimpleNamespace(ExportToString=lambda:values['Root']),
        GetSessionLayer=lambda:SimpleNamespace(ExportToString=lambda:values['Session']))
    host._last_prepared_boundary_texts=dict(values);host.prepared_layers=dict(values)
    def reject(*args):raise ValueError('RECLASSIFICATION_REQUIRED')
    monkeypatch.setattr(sequence.retained,'split_viewport_renderer_defaults',reject)
    assert host.check_prepared_layers()==values and context_checks==[True]
    values['Root']='mass or arbitrary byte edit'
    with pytest.raises(ValueError,match='RECLASSIFICATION'):host.check_prepared_layers()
    assert len(context_checks)==2


def test_remaining_first_static_never_opens_files_without_explicit_authority():
    with pytest.raises(ValueError,match='remaining FIRST static authority'):
        asyncio.run(sequence.lifecycle.finish_unstarted_staged_static_once('/does/not/exist'))


@pytest.mark.parametrize('direction,floor',[(1,.010),(-1,.020)])
@pytest.mark.parametrize('method',sequence.METHODS)
def test_directional_quiet_candidate_uses_physical_direction_not_method_id(method,direction,floor):
    import hospital_h4_path_execution as original
    plan,geometry=mission_fixture((direction,)*10)
    plan.update(original.method_settings(method))
    plan['allowed_modes']=['coordinated_counterphase'] if method.startswith('B0_') else list(timed.MODES)
    for a in plan['selected_actions']:a['target_speed_m_s']=.003
    c=sequence.DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=600.)
    c.stage='tracking';c.previous_time=c.started=0.
    last=0.
    for i in range(1,21):
        out=c.step(i*.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
        assert abs(out['speed_m_s']-last)<=.002+1e-12
        last=out['speed_m_s']
    assert last==pytest.approx(direction*floor)


def test_quiet_stop_requires_continuous_original_limit_AND_stricter_capture_speed():
    plan,geometry=mission_fixture();c=sequence.DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=600.)
    for i in range(7):
        out=c.step(i*.05,[0.,0.,0.],.0045,(0.,0.),drive_hold_confirmed=False)
        assert not out['drive_hold_requested']
    out=c.step(.35,[0.,0.,0.],.0025,(0.,0.),drive_hold_confirmed=False)
    assert out['drive_hold_requested'] and c.stage=='neutral'
    with pytest.raises(RuntimeError,match='not stopped'):
        c.step(.40,[0.,0.,0.],.0051,(0.,0.),drive_hold_confirmed=True)


def test_quiet_stop_resets_dwell_on_rebound_and_never_changes_original8s_timeout():
    plan,geometry=mission_fixture();c=sequence.DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=600.)
    for t,speed in [(0.,.00448),(.0167,.00441),(.0333,.00443),(.05,.00537883),(.1,.0025),(.2,.0025),(.3,.0025)]:
        out=c.step(t,[0.,0.,0.],speed,(0.,0.),drive_hold_confirmed=False)
        assert not out['drive_hold_requested']
    assert c.step(.36,[0.,0.,0.],.0025,(0.,0.),drive_hold_confirmed=False)['drive_hold_requested']
    stuck=sequence.DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=600.)
    for i in range(81):
        assert not stuck.step(i*.1,[0.,0.,0.],.006,(0.,0.),drive_hold_confirmed=False)['drive_hold_requested']
    with pytest.raises(TimeoutError,match='timeout8s'):
        stuck.step(8.1,[0.,0.,0.],.006,(0.,0.),drive_hold_confirmed=False)


def test_quiet_stop_gate_is_reset_at_each_new_cusp_and_contract_keeps_acceptance():
    plan,geometry=mission_fixture();c=sequence.DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=600.)
    c.quiet_stop_since=0.;c._stage('cusp_braking',10.,[0.,0.,0.])
    assert c.quiet_stop_since is None
    contract=sequence.directional_quiet_stop_contract(simulation_budget_s=600.)
    assert contract['original_terminal_m_deg_speed']==[.030,2.,.005]
    assert contract['stop_entry_binding']['undo_reproduces_entire_original_AST']
    assert contract['stop_entry_binding']['original8s_stop_timeout_and_postcapture5mm_s_guard_unchanged']
    assert contract['explicit_OFFLINE_candidate_NOT_wired_to_live_route']


def test_interior_cusp_neutralizes_BEFORE_capture_and_restarts_dwell_on_steering_rebound():
    import numpy as np
    plan,geometry=mission_fixture();c=sequence.DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=600.)
    c.stage='cusp_braking';c.stage_started=c.started=c.previous_time=0.;c.curvature=.6;c.speed=0.
    for t in [.1,.2,.3,.4]:
        out=c.step(t,[0.,0.,0.],.002,np.radians([-15.,-1.]),drive_hold_confirmed=False)
        assert not out['drive_hold_requested'] and c.curvature==0. and c.stage=='cusp_braking'
    for t,angles in [(.5,[2.,0.]),(.6,[3.,0.]),(.7,[2.,0.]),(.8,[1.5,0.]),(.9,[1.,0.])]:
        assert not c.step(t,[0.,0.,0.],.002,np.radians(angles),drive_hold_confirmed=False)['drive_hold_requested']
    assert c.step(1.,[0.,0.,0.],.002,np.radians([.1,0.]),drive_hold_confirmed=False)['drive_hold_requested']
    assert c.stage=='neutral'


def directional_live_packet():
    return dict(simulation_budget_s=600.,method='P_dynamics_full_envelope',
        common_tracking_policy=sequence.DIRECTIONAL_QUIET_STOP_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        explicit_route_approval=copy.deepcopy(sequence.DIRECTIONAL_QUIET_STOP_APPROVAL),
        common_lifecycle_contract=sequence.directional_quiet_stop_lifecycle_contract(simulation_budget_s=600.))


@pytest.mark.parametrize('fault',['forward','reverse','boolean','policy','approval','method','scalar',
    'missing_forward','missing_reverse','missing_policy','implicit','old10','old20','contract'])
def test_new_directional_route_rejects_wrong_scope_and_incomplete_or_changed_contract(fault):
    plan,geometry=mission_fixture();packet=directional_live_packet()
    if fault=='forward':packet['common_forward_tracking_floor_m_s']=.020
    elif fault=='reverse':packet['common_reverse_tracking_floor_m_s']=.010
    elif fault=='boolean':packet['common_forward_tracking_floor_m_s']=True
    elif fault=='policy':packet['common_tracking_policy']='UNREVIEWED'
    elif fault=='approval':packet['explicit_route_approval']=sequence.MEASURED_RESPONSE_FIX_APPROVAL
    elif fault=='method':packet['method']='B0_fixed_counterphase_ramp'
    elif fault=='scalar':packet['common_tracking_speed_floor_m_s']=.020
    elif fault.startswith('missing_'):
        del packet[{'missing_forward':'common_forward_tracking_floor_m_s',
            'missing_reverse':'common_reverse_tracking_floor_m_s','missing_policy':'common_tracking_policy'}[fault]]
    elif fault in ('implicit','old10','old20'):
        for k in ('common_tracking_policy','common_forward_tracking_floor_m_s','common_reverse_tracking_floor_m_s'):del packet[k]
        if fault!='implicit':packet.update(common_tracking_policy=sequence.TRACKING_FLOOR_POLICY if fault=='old10' else sequence.MEASURED_RESPONSE_FLOOR_POLICY,
            common_tracking_speed_floor_m_s=.010 if fault=='old10' else .020)
    else:packet['common_lifecycle_contract']['acceptance']['unreviewed']=True
    with pytest.raises(ValueError):sequence.route_controller(packet,plan,geometry)


def test_directional_route_uses_exact_new_approval_and_original_outer_lifecycle_SOFTWARE_only():
    from test_hospital_mission_lifecycle import SoftwareHost
    packet=directional_live_packet();plan,geometry=mission_fixture()
    assert sequence.lifecycle.checked_fresh_held_approval(sequence.DIRECTIONAL_QUIET_STOP_APPROVAL)==sequence.DIRECTIONAL_QUIET_STOP_APPROVAL
    for method in sequence.METHODS[1:]:
        with pytest.raises(ValueError,match='ONE P'):
            sequence.route_definition('/does/not/exist',method,approval=sequence.DIRECTIONAL_QUIET_STOP_APPROVAL)
    controller=sequence.route_controller(packet,plan,geometry)
    assert type(controller) is sequence.DirectionalQuietStopLifecycleCandidate
    assert type(controller.motion) is sequence.DirectionalQuietStopCandidate
    assert controller.contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']
    assert 'explicit_OFFLINE_candidate_NOT_wired_to_live_route' not in controller.contract['original_motion_contract']
    host=SoftwareHost()
    # This ideal SDK fixture was constructed at stamp1s. Match that clock
    # through the constructor, never move its retained backend stamp backward.
    controller=type(controller)(plan,geometry,simulation_budget_s=600.,start_time_s=host.clock)
    report=asyncio.run(sequence.lifecycle.run_lifecycle_once(host,controller,authorized=True))
    assert report['verdict']=='PASS' and report['new_route_trials']==1
    assert report['checks']['original_gains_targets_restored']
    assert host.events.count('Play')==host.events.count('actual_Pause')==1
    assert controller.complete and controller.fixed_hold_attempts==1


@pytest.mark.parametrize('value',[True,False,0.,.003,.005,.02,float('nan'),float('inf'),'0.01'])
def test_prospective_tracking_floor_requires_explicit_single_candidate_not_arbitrary_tuning(value):
    plan,geometry=mission_fixture()
    with pytest.raises(ValueError,match='prospective common10mm/s'):
        sequence.CommonTrackingFloorCandidate(plan,geometry,simulation_budget_s=600.,tracking_speed_floor_m_s=value)
    with pytest.raises(ValueError):
        sequence.tracking_floor_candidate_contract(simulation_budget_s=600.,tracking_speed_floor_m_s=value)


@pytest.mark.parametrize('direction',[1,-1])
@pytest.mark.parametrize('method',['B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'])
def test_prospective_common_floor_preserves_ramp_direction_reference_and_default_policy(method,direction):
    import hospital_h4_path_execution as original
    plan,geometry=mission_fixture((direction,)*10)
    plan.update(original.method_settings(method))
    plan['allowed_modes']=['coordinated_counterphase'] if method.startswith('B0_') else list(timed.MODES)
    for action in plan['selected_actions']:action['target_speed_m_s']=.003
    before=copy.deepcopy((plan,geometry))
    old=mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.)
    new=sequence.CommonTrackingFloorCandidate(plan,geometry,simulation_budget_s=600.,tracking_speed_floor_m_s=.010)
    for controller in (old,new):
        controller.stage='tracking';controller.previous_time=controller.started=0.
    previous=0.
    for i in range(1,21):
        old_out=old.step(i*.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
        out=new.step(i*.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
        assert abs(out['speed_m_s']-previous)<=.002+1e-12
        assert abs(out['speed_m_s'])<=.010+1e-12
        assert direction*out['speed_m_s']>0.
        assert not out['drive_hold_requested']
        previous=out['speed_m_s']
    assert abs(out['speed_m_s'])==pytest.approx(.010)
    assert abs(old_out['speed_m_s'])==pytest.approx(.003)
    assert (plan,geometry)==before
    assert new.reference.actions==plan['selected_actions']
    assert new.tracking_step.mission_binding['AST_tracking_target_substitutions']==1
    assert new.tracking_step.mission_binding['undo_reproduces_entire_original_AST']
    assert old.tracking_step.mission_binding['feedback_gains_path_guards_and_all_other_statements_unchanged']


def test_prospective_floor_does_not_override_original_endpoint_braking_cap():
    plan,geometry=mission_fixture()
    for action in plan['selected_actions']:action['target_speed_m_s']=.003
    controller=sequence.CommonTrackingFloorCandidate(plan,geometry,simulation_budget_s=600.,tracking_speed_floor_m_s=.010)
    controller.stage='tracking';controller.previous_time=controller.started=0.;controller.speed=.010
    # Project near the end, but outside the unchanged 50mm endpoint handoff.
    # Seed only fixture reference progress; this is not a physical state edit.
    controller.leg['view'].progress=.48
    out=controller.step(.1,[.488,.049,0.],.010,(0.,0.),drive_hold_confirmed=False)
    expected=math.sqrt(2*.020*.002)
    assert out['speed_m_s']==pytest.approx(expected) and out['speed_m_s']<.010
    assert out['phase']=='path_tracking'


@pytest.mark.parametrize('bad_pose',[[0.,.080001,0.],[0.,0.,math.radians(10.0001)]])
def test_prospective_floor_cannot_waive_original_tracking_guards(bad_pose):
    plan,geometry=mission_fixture()
    controller=sequence.CommonTrackingFloorCandidate(plan,geometry,simulation_budget_s=600.,tracking_speed_floor_m_s=.010)
    controller.stage='tracking';controller.previous_time=controller.started=0.
    with pytest.raises(RuntimeError,match='80 mm / 10 degrees'):
        controller.step(.1,bad_pose,0.,(0.,0.),drive_hold_confirmed=False)


def test_prospective_floor_does_not_add_motion_to_stop_hold_or_change_terminal_policy():
    plan,geometry=mission_fixture();outputs=[]
    for floor in (None,.010):
        controller=(mission.MissionTrackingController(plan,geometry,simulation_budget_s=600.) if floor is None else
            sequence.CommonTrackingFloorCandidate(plan,geometry,simulation_budget_s=600.))
        first=controller.step(0.,[0.,0.,0.],.006,(0.,0.),drive_hold_confirmed=False)
        assert first['speed_m_s']==0. and not first['drive_hold_requested']
        controller.terminal=controller.raw(controller.reference,controller.goal)
        controller.stage='terminal'
        controller.terminal.previous_time=controller.terminal.terminal_started=controller.terminal.stage_started=0.
        controller.terminal.terminal_stage='entry_braking';controller.terminal.phase='terminal_braking'
        outputs.append(controller.step(.1,[.49,0.,0.],0.,(0.,0.),drive_hold_confirmed=False))
    assert outputs[0]==outputs[1]
    contract=sequence.tracking_floor_candidate_contract(simulation_budget_s=600.,tracking_speed_floor_m_s=.010)
    assert contract['changed_common_controller_requires_prospective_matched_comparison']
    assert contract['original_terminal_m_deg_speed']==[.030,2.,.005]
    assert not contract['actual_motion_authorized'] and not contract['runtime_adapter_prepared']




def test_new_floor_approval_is_actual_user_message_and_never_reuses_a_question_answer():
    value=sequence.lifecycle.checked_fresh_held_approval(sequence.TRACKING_FLOOR_APPROVAL)
    assert value['approval_source']=='USER_MESSAGE' and value['user_message']=='진행하자'
    assert 'question_item_id' not in value
    with pytest.raises(ValueError):
        sequence.route_definition(source_result(),'B0_fixed_counterphase_ramp',approval=value)


@pytest.mark.parametrize('fault',['floor','policy','approval','method','missing_floor','missing_policy','contract'])
def test_candidate_live_selection_refuses_incomplete_or_changed_prospective_contract(fault):
    plan,geometry=mission_fixture()
    packet=dict(simulation_budget_s=600.,common_tracking_policy=sequence.TRACKING_FLOOR_POLICY,
        common_tracking_speed_floor_m_s=.010,explicit_route_approval=copy.deepcopy(sequence.TRACKING_FLOOR_APPROVAL),
        method='P_dynamics_full_envelope',common_lifecycle_contract=sequence.tracking_floor_lifecycle_contract(simulation_budget_s=600.))
    if fault=='floor':packet['common_tracking_speed_floor_m_s']=.02
    elif fault=='policy':packet['common_tracking_policy']='UNREVIEWED'
    elif fault=='approval':packet['explicit_route_approval']=sequence.REVALIDATION_APPROVAL
    elif fault=='method':packet['method']='B1_multimode_kinematic_ramp'
    elif fault=='missing_floor':del packet['common_tracking_speed_floor_m_s']
    elif fault=='missing_policy':del packet['common_tracking_policy']
    else:packet['common_lifecycle_contract']['acceptance']['unreviewed']=True
    with pytest.raises(ValueError):sequence.route_controller(packet,plan,geometry)


def test_candidate_outer_lifecycle_and_sdk_bridge_still_complete_one_fixed_hold_in_SOFTWARE_only():
    from test_hospital_mission_lifecycle import SoftwareHost
    plan,geometry=mission_fixture()
    for action in plan['selected_actions']:action['target_speed_m_s']=.003
    controller=sequence.TrackingFloorLifecycleCandidate(plan,geometry,simulation_budget_s=600.,start_time_s=1.)
    assert isinstance(controller,sequence.lifecycle.MissionLifecycleController)
    host=SoftwareHost()
    report=asyncio.run(sequence.lifecycle.run_lifecycle_once(host,controller,authorized=True))
    assert report['verdict']=='PASS' and report['new_route_trials']==1
    assert report['checks']['original_gains_targets_restored']
    assert host.events.count('Play')==host.events.count('actual_Pause')==1
    assert controller.fixed_hold_attempts==1 and controller.complete
    assert type(controller.motion) is sequence.CommonTrackingFloorCandidate
    assert controller.contract['acceptance']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)['acceptance']


def test_unselected_live_route_keeps_original_controller_and_contract():
    plan,geometry=mission_fixture()
    packet=dict(simulation_budget_s=600.,common_lifecycle_contract=sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.))
    controller=sequence.route_controller(packet,plan,geometry)
    assert type(controller) is sequence.lifecycle.MissionLifecycleController
    assert type(controller.motion) is mission.MissionTrackingController


@pytest.mark.parametrize('fault',[None,'Root_metadata','mass','camera_target','camera_metadata','picking_type'])
def test_saved_view_switch_review_is_archive_ONLY_and_rejects_other_scene_or_type_changes(fault):
    from pxr import Sdf
    p=sequence.retained.GATE/'hospital_h6_experiments/cohort_12_held_static_r01/route/P_dynamics_full_envelope/route_readback.json'
    raw=sequence.retained.read(p);packet=sequence.retained.read(raw['protocol'])
    original=sequence.retained.runtime.task.sha(p)
    before=sequence.retained.read(packet['original_static_record'])['actual_final_readback']['layer_texts']
    after=copy.deepcopy(raw['actual_final_readback']['layer_texts'])
    if fault is not None:
        key='Root' if fault in ('Root_metadata','mass','camera_target') else 'Session'
        layer=Sdf.Layer.CreateAnonymous('FAULT_COPY'+('.usda' if after[key].startswith('#usda') else '.sdf'))
        assert layer.ImportFromString(after[key])
        if fault=='Root_metadata':
            data=copy.deepcopy(layer.customLayerData);data['navmeshSettings']['agentRadius']+=.001;layer.customLayerData=data
        elif fault=='mass':
            prim=Sdf.CreatePrimInLayer(layer,packet['actor']+'/base_link')
            attr=layer.GetAttributeAtPath(prim.path.AppendProperty('physics:mass'))
            if attr is None:attr=Sdf.AttributeSpec(prim,'physics:mass',Sdf.ValueTypeNames.Float)
            attr.default=300.
        elif fault=='camera_target':
            rel=layer.GetRelationshipAtPath('/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0.camera')
            rel.targetPathList.explicitItems=[Sdf.Path('/World/UnreviewedSensorCamera')]
        elif fault=='camera_metadata':layer.GetPrimAtPath('/OmniverseKit_Top').documentation='UNREVIEWED'
        else:
            attr=layer.GetAttributeAtPath('/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0.viewPickingId')
            owner=attr.owner;name=attr.name;owner.RemoveProperty(attr)
            attr=Sdf.AttributeSpec(owner,name,Sdf.ValueTypeNames.Bool,Sdf.VariabilityVarying,False);attr.default=True
        after[key]=layer.ExportToString()
    if fault is None:
        result=sequence.saved_viewport_archive_review(before,after,packet['actor'],packet['required_tensor_snapshot'],raw['actual_final_readback']['tensor'])
        assert len(result['exact_saved_viewport_changes'])==3
        assert result['live_camera_or_notice_policy_NOT_changed'] and not result['actual_route_admission']
    else:
        with pytest.raises(ValueError):
            sequence.saved_viewport_archive_review(before,after,packet['actor'],packet['required_tensor_snapshot'],raw['actual_final_readback']['tensor'])
    assert sequence.retained.runtime.task.sha(p)==original and raw['verdict']=='HOLD'


def test_new_camera_interruption_revalidation_uses_new_message_not_consumed_prior_authority():
    new=sequence.lifecycle.checked_fresh_held_approval(sequence.CAMERA_INTERRUPTION_APPROVAL)
    assert new['user_message'].startswith('아 내가 잘 가는지 보려고')
    assert new!=sequence.TRACKING_FLOOR_APPROVAL
    assert new in sequence.TRACKING_FLOOR_APPROVALS and 'question_item_id' not in new


def source_result():
    return sequence.retained.GATE/'hospital_h6_experiments/cohort_07_held_static_r01/held_static/held_static_readback.json'


def consumed_route_result():
    return source_result().parent.parent/'route/P_dynamics_full_envelope/route_readback.json'


def actual_route_handoff():
    value=sequence.saved_route_preparation_handoff(consumed_route_result())
    # Source-addressed prospective receipts change after an offline helper
    # edit; never expect or overwrite an older consumed workflow's packet.
    path=sequence.export_route_preparation_handoff(consumed_route_result())
    assert sequence.retained.read(path)==value
    return path


def test_saved_camera_route_HOLD_and_full_actual_GET_are_not_promoted_or_replayed():
    result=consumed_route_result();digest=sequence.retained.runtime.task.sha(result)
    value=sequence.saved_route_boundary_review(result)
    assert value['original_verdict']=='HOLD' and value['new_physical_trials']==0
    assert value['original_HOLD_FAIL_never_promoted'] and not value['actual_route_admission']
    assert len(value['display_review']['exact_Top_default_differences'])==3
    handoff=sequence.retained.read(actual_route_handoff())
    raw=sequence.retained.read(result)
    assert handoff['SDK_GET']['actual_tensor_snapshot']==raw['actual_final_readback']['tensor']
    assert handoff['original_verdict']=='HOLD' and handoff['new_live_actions']==0
    assert sequence.retained.runtime.task.sha(result)==digest


def test_next_cohort_definition_and_existing_loader_require_fresh_static_and_original_start_without_SDK(tmp_path,monkeypatch):
    def forbidden():raise AssertionError('OFFLINE inspection cannot bind SDK')
    monkeypatch.setattr(sequence.retained.runtime,'live_sdk',forbidden)
    handoff=actual_route_handoff()
    monkeypatch.setattr(sequence.retained,'GATE',tmp_path)
    approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_USER_AUTHORITY',answer_ko='새 코호트 정지유지 1회 승인')
    path=sequence.lifecycle.export_fresh_held_workflow(cohort='cohort_12_held_static_r01',approval=approval,prior_handoff=handoff)
    context=sequence.lifecycle.load_fresh_held_workflow(path)
    p=context.protocol
    assert p['required_prior_result']==str(handoff)
    assert p['prior_verdict_preserved']=='HOLD' and p['user_approval']==approval
    assert p['maximum_Stop_calls']==p['maximum_static_trials']==1 and p['maximum_route_trials']==0
    assert p['start_xyyaw']!=p['required_prior_tensor']['transform'][:3]
    assert p['start_xyyaw']==sequence.retained.read(p['geometry_file'])['start_xyyaw']
    assert p['new_robot_root']!=p['retired_robot_root'] and not p['automatic_retry']


@pytest.mark.parametrize('fault',['tensor','timeline','Root','environment','original_verdict'])
def test_route_final_GET_handoff_refuses_unmatched_or_incomplete_saved_boundaries(monkeypatch,fault):
    original=sequence.retained.read
    def changed(path):
        value=original(path)
        if str(path).endswith('/final_boundary_GET.json'):
            value=copy.deepcopy(value)
            if fault=='tensor':value['actual_tensor_snapshot']['transform'][0]+=.001
            elif fault=='timeline':value['timeline']['current_time_s']+=1.
            elif fault=='Root':value['actual_layer_texts']['Root']+='\n'
            elif fault=='environment':value['all4997_rows145_source_settings_physics']['row_count']=4996
            elif fault=='original_verdict':value['original_verdict']='PASS'
        return value
    monkeypatch.setattr(sequence.retained,'read',changed)
    with pytest.raises(ValueError):sequence.saved_route_preparation_handoff(consumed_route_result())


def test_consumed_route_handoff_uses_the_original_Stop_constructor_display_pipeline_SOFTWARE_only(tmp_path,monkeypatch):
    from test_hospital_mission_lifecycle import _fresh_preparation_software_fixture
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch,prior_handoff=actual_route_handoff())
    prep=asyncio.run(f.operation(f.context))
    assert f.events==['ONE_Stop_requested','Stop_completed','FIRST_display_placeholder']
    assert prep['Play_or_static_or_drive_executed'] is False
    assert not (tmp_path/'held_static/held_static_readback.json').exists()


def test_same_static_adapter_preserves_actual_PASS_tensor_raw_CSV_and_original_bytes():
    source=source_result();sha=sequence.retained.runtime.task.sha(source)
    raw=sequence.retained.read(source)
    prep,adapter,replay=sequence.held_static_adapter(source)
    assert replay['verdict']==adapter['verdict']=='PASS'
    assert adapter['actual_post_static_tensor_snapshot']==raw['actual_final_readback']['tensor']
    assert adapter['trace']==raw['trace'] and adapter['trace_sha256']==raw['trace_sha256']
    assert adapter['runtime_motor_baseline']==prep['expected_nominal_motor_baseline']
    assert adapter['simulation_mass_kg']==265. and adapter['representation_only_static_binding']['new_trials']==0
    assert sequence.retained.runtime.task.sha(source)==sha


@pytest.mark.parametrize('method',sequence.METHODS)
def test_all_four_route_definitions_use_same600s_controller_safety_start_and_NO_Stop(method):
    packet,adapter,plan=sequence.route_definition(source_result(),method,approval=sequence.APPROVAL)
    assert packet['method']==plan['method']==method
    assert packet['maximum_Play_calls']==packet['maximum_route_trials']==1
    assert packet['maximum_Stop_reset_restaging_calls']==0 and packet['automatic_retry'] is False
    assert packet['simulation_budget_s']==600. and packet['maximum_wall_time_s']==900.
    assert packet['required_tensor_snapshot']==adapter['actual_post_static_tensor_snapshot']
    assert packet['common_lifecycle_contract']==sequence.lifecycle.lifecycle_contract(simulation_budget_s=600.)
    assert plan['start_world_xyyaw']==adapter['start_xyyaw']
    assert packet['use_for_final_statistics'] is False


def test_no_default_or_changed_authority_before_file_or_SDK_access(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No read or SDK before explicit scope')
    monkeypatch.setattr(sequence,'held_static_adapter',forbidden)
    with pytest.raises(ValueError):sequence.route_definition('/MUST_NOT_READ','P_dynamics_full_envelope')
    with pytest.raises(ValueError):sequence.route_definition('/MUST_NOT_READ','P_dynamics_full_envelope',approval=dict(sequence.APPROVAL,answer_ko='ASSUMED'))
    with pytest.raises(ValueError):sequence.schedule_route('/MUST_NOT_READ')
    with pytest.raises(ValueError):asyncio.run(sequence.run_route_once('/MUST_NOT_READ'))


def test_no_static_HOLD_is_promoted_or_used_for_route_admission():
    source=sequence.retained.GATE/'hospital_h6_experiments/cohort_06_held_static_r01/held_static/held_static_readback.json'
    with pytest.raises(ValueError):sequence.held_static_adapter(source)


def test_export_load_corruption_and_duplicate_guards_never_bind_SDK(tmp_path,monkeypatch):
    original=sequence.route_definition
    def definition(*args,**kwargs):
        packet,adapter,plan=original(*args,**kwargs)
        packet=copy.deepcopy(packet);packet['output_directory']=str(tmp_path/'route')
        packet['original_static_record']=str(tmp_path/'route/same_static_schema_binding.json')
        return packet,adapter,plan
    monkeypatch.setattr(sequence,'route_definition',definition)
    def forbidden():raise AssertionError('No SDK for offline packet tests or duplicate dispatch')
    monkeypatch.setattr(sequence.retained.runtime,'live_sdk',forbidden)
    path=sequence.export_route(source_result(),'P_dynamics_full_envelope',approval=sequence.APPROVAL)
    context,packet,binding,plan,geometry=sequence.load_route(path)
    assert context.stationary.RESULT==Path(packet['original_static_record'])
    assert context.stationary.TRACE==Path(packet['original_static_trace'])
    assert binding['version']==sequence.lifecycle.RetainedMissionSDKHost.VERSION
    prep,adapter,replay=context.operations.replay_static(context)
    assert adapter==sequence.retained.read(context.stationary.RESULT) and replay['verdict']=='PASS'
    with pytest.raises(ValueError,match='Existing route'):sequence.export_route(source_result(),'P_dynamics_full_envelope',approval=sequence.APPROVAL)
    reserve=path.parent/'single_route_reserved.json'
    sequence.retained.evidence.write_exclusive(reserve,dict(SOFTWARE_ONLY=True))
    assert asyncio.run(sequence.run_route_once(path,authorized=True))['status']=='PRIOR_ROUTE_DISPATCH_RETAINED_NO_NEW_ACTION'
    assert sequence.schedule_route(path,authorized=True) is None
    changed=sequence.retained.read(context.stationary.RESULT);changed['actual_post_static_tensor_snapshot']['velocity'][0]=1.
    context.stationary.RESULT.write_text(__import__('json').dumps(changed),encoding='utf-8')
    with pytest.raises(ValueError):sequence.load_route(path)
