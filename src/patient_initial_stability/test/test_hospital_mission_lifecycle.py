"""Software sequence/cleanup fixtures, NOT Isaac contacts or hold-force tests."""

import asyncio
import copy
import math
from pathlib import Path

import numpy as np
import pytest

import hospital_mission_lifecycle as lifecycle
import hospital_mission_execution as motion
from test_hospital_timed_path_execution import mission_fixture, _sdk_fixture


@pytest.fixture(autouse=True)
def relocated_saved_session_paths(monkeypatch):
    """Map archived file references in a read-only test view, not original logs.

    Physical guards and original byte hashes are unchanged. This fixture does not
    patch Path, synthesize observations, or admit a live actor.
    """
    from ijcas_paths import resolve_archived_path, relocate_json_view
    retained = lifecycle.retained
    read, sha = retained.read, retained.runtime.task.sha
    def read_view(path):
        original = read(resolve_archived_path(path))
        view = relocate_json_view(original)
        # Validate the historical contract first. A relocated software-only
        # representation needs its own digest; never bless a corrupt input.
        if isinstance(original, dict) and 'definition_sha256' in original:
            payload = {k: v for k, v in original.items() if k != 'definition_sha256'}
            if original['definition_sha256'] == retained.shared.digest(payload):
                view['definition_sha256'] = retained.shared.digest(
                    {k: v for k, v in view.items() if k != 'definition_sha256'})
        return view
    monkeypatch.setattr(retained, 'read', read_view)
    monkeypatch.setattr(retained.runtime.task, 'sha', lambda path: sha(resolve_archived_path(path)))


@pytest.mark.parametrize('speed',[.005001,.00663991272779625])
def test_measured_stop_NOT_zero_request_required_before_capture(speed):
    gate=lifecycle.MeasuredPreHold([0.,0.,0.])
    for i in range(10):
        info=gate.step(i*.05,[0.,0.,0.],speed,(0.,0.),drive_hold_confirmed=False)
        assert info['speed_m_s']==0. and not info['drive_hold_requested']
    info=gate.step(.5,[0.,0.,0.],.005,(0.,0.),drive_hold_confirmed=False)
    assert info['drive_hold_requested'] and not info['preparation_ready']


def test_actual_current_30mm_displaced_robot_rejected_not_rebased_or_admitted_by_prehold():
    gate=lifecycle.MeasuredPreHold([14.192329139709472,16.568600039835417,0.])
    with pytest.raises(RuntimeError,match='Original10mm'):
        gate.step(0.,[14.222394943237305,16.5685977935791,0.],.00663991272779625,(0.,0.),drive_hold_confirmed=False)
    assert gate.failed and gate.requested_at is None
    with pytest.raises(RuntimeError,match='already failed'):
        gate.step(.05,[*gate.start],0.,(0.,0.),drive_hold_confirmed=False)


def test_actual_later_ACK_and_fixed2s_no_hold_request_echo_or_better_window():
    gate=lifecycle.MeasuredPreHold([0.,0.,0.])
    info=gate.step(0.,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    assert info['drive_hold_requested'] and gate.held_since is None
    for i in range(1,6):
        info=gate.step(i*.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
        assert gate.held_since is None and not info['preparation_ready']
    gate.step(.6,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)
    for i in range(1,21):
        info=gate.step(.6+i*.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)
    assert info['preparation_ready'] and gate.ready_at==pytest.approx(2.6)
    with pytest.raises(RuntimeError,match='no recapture'):
        gate.step(2.7,[0.,0.,0.],.005001,(0.,0.),drive_hold_confirmed=True)
    assert gate.ready_at==pytest.approx(2.6) and gate.failed


@pytest.mark.parametrize('fault',['pose','heading','steering','gap','nonfinite','request_echo','ACK_lost'])
def test_initial_boundary_clock_and_hold_fail_terminally_without_physical_recovery(fault):
    gate=lifecycle.MeasuredPreHold([0.,0.,0.]);pose=[0.,0.,0.];steer=[0.,0.];stamp=0.;ack=False
    if fault=='pose':pose[0]=.010001
    elif fault=='heading':pose[2]=math.radians(.50001)
    elif fault=='steering':steer[0]=math.radians(1.00001)
    elif fault=='gap':stamp=.251
    elif fault=='nonfinite':pose[0]=math.nan
    elif fault=='request_echo':ack=None
    else:
        gate.step(0.,pose,0.,steer,drive_hold_confirmed=False)
        gate.step(.1,pose,0.,steer,drive_hold_confirmed=True);stamp=.2
    with pytest.raises((ValueError,RuntimeError)):
        gate.step(stamp,pose,0.,steer,drive_hold_confirmed=ack)
    assert gate.failed


def test_prehold_initial_stop_timeout_not_extended_by_favourable_late_sample():
    gate=lifecycle.MeasuredPreHold([0.,0.,0.])
    for i in range(33):gate.step(i*.25,[0.,0.,0.],.006,(0.,0.),drive_hold_confirmed=False)
    with pytest.raises(TimeoutError):gate.step(8.25,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)


class SoftwareHost:
    """Ideal kinematic/hold-response fixture, deliberately NOT PhysX.

    The REAL retained adapter reads/writes a stateful typed fake backend.
    Its hold ACK must follow actual getter equality at a later simulated
    clock, not a previous requested flag. Physical dynamics are NOT validated.
    """
    def __init__(self,*,fault=None):
        self.backend,self.actuator=_sdk_fixture();self.fault=fault;self.events=[];self.reserved=False
        self.clock=1.;self.pose=[0.,0.,0.];self.speed=0.;self.counts={'wheel_support':0};self.errors=[]
        self.saved=None;self.saved_rows=None;self.drive_qd=np.zeros(2);self.guard_calls=0

    def claim_once(self):
        self.events.append('claim_once')
        if self.reserved:return False
        self.reserved=True;return True

    async def preflight(self):
        self.events.append('preflight')
        if self.fault=='preflight':raise RuntimeError('SOFTWARE_PREFLIGHT_REFUSED')
        if self.fault=='bad_preflight_evidence':return {'unsupported_array':np.array([1.])}
        return {'software_fixture_only_NOT_all4997_145_live_proof':True}

    def subscribe(self):self.events.append('subscribe')

    def measure(self):
        row=dict(time_s=self.clock,x_m=self.pose[0],y_m=self.pose[1],yaw_rad=self.pose[2],
            vx_physx_m_s=self.speed*math.cos(self.pose[2]),vy_physx_m_s=self.speed*math.sin(self.pose[2]),
            vz_physx_m_s=0.,pose_filtered_speed_m_s=abs(self.speed),
            drive_front_rad=float(self.backend.physical_q[0,self.actuator.drive[0]]),
            drive_rear_rad=float(self.backend.physical_q[0,self.actuator.drive[1]]),
            drive_front_velocity_rad_s=float(self.drive_qd[0]),drive_rear_velocity_rad_s=float(self.drive_qd[1]))
        row.update({f'wheel_{i}_fz_n':265.*9.81/6. for i in range(6)})
        if self.fault=='nonfinite':row['wheel_2_fz_n']=math.nan
        if self.fault=='late_nonfinite' and self.clock>=1.25:row['wheel_2_fz_n']=math.nan
        if self.fault=='numpy_scalars':row={k:np.float64(v) for k,v in row.items()}
        return row

    def guard(self,row,info):
        self.guard_calls+=1
        if self.fault=='initial_guard':raise RuntimeError('SOFTWARE_INITIAL_FULL_GUARD_REFUSED')
        if self.fault=='guard' and self.clock>=1.25:raise RuntimeError('SOFTWARE_FULL_GUARD_REFUSED_BEFORE_WRITE')
        if self.errors or any(self.counts.get(k,0) for k in lifecycle.BAD_CONTACTS):raise RuntimeError('SOFTWARE_CONTACT_GUARD')
        assert math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])<=.080
        if info is not None:assert abs(info['speed_m_s'])<=.0500001

    def play(self):self.events.append('Play');self.backend.paused=False

    async def next_sample(self):
        if self.fault=='cancel':raise asyncio.CancelledError()
        self.clock+=.05
        self.backend.physical_q[0,self.actuator.steering]=self.backend.data['positions'][0,self.actuator.steering]
        steer=self.backend.physical_q[0,self.actuator.steering].astype(float)
        wheels=self.backend.data['velocities'][0,self.actuator.drive].astype(float)
        held=bool(np.any(self.backend.data['kp'][0,self.actuator.drive]>0.))
        self.speed=0. if held else float(np.mean(wheels*.074*np.cos(steer)))
        curvature=(math.tan(steer[0])-math.tan(steer[1]))/1.29
        self.pose[0]+=self.speed*math.cos(self.pose[2])*.05
        self.pose[1]+=self.speed*math.sin(self.pose[2])*.05
        self.pose[2]+=self.speed*curvature*.05
        self.drive_qd=np.zeros(2) if held else wheels
        self.backend.physical_q[0,self.actuator.drive]+=self.drive_qd*.05
        self.counts['wheel_support']+=6
        if self.fault=='zero_setter' and self.clock>=1.25:self.backend.fail='velocities'
        if self.fault=='hold_partial' and self.clock>=1.05:self.backend.drop='kp'

    async def confirm_pause(self):
        self.events.append('actual_Pause')
        if self.fault=='pause':raise RuntimeError('SOFTWARE_PAUSE_NOT_OBSERVED')
        self.backend.paused=True
        if self.fault=='pause_contact':self.counts['hospital_obstacle']=1
        if self.fault=='restoration':self.backend.fail='positions'
        return {'software_actual_paused':True}

    def unsubscribe(self):self.events.append('unsubscribe')

    async def postflight(self):
        self.events.append('postflight')
        if self.fault=='postflight':raise RuntimeError('SOFTWARE_POSTFLIGHT_FAILURE')
        original=all(np.array_equal(self.backend.data[k][0],v) for k,v in self.actuator.original.items())
        return {'checks':{'software_complete_motor_restoration':original}}

    def save(self,report,rows):
        self.events.append('durable_save_fixture')
        if self.fault=='save':raise OSError('SOFTWARE_EVIDENCE_WRITE_FAILURE_NO_REPLAY')
        self.saved=copy.deepcopy(report);self.saved_rows=copy.deepcopy(rows)


def controller_for(directions=(1,)*20):
    plan,geometry=mission_fixture(directions)
    return lifecycle.MissionLifecycleController(plan,geometry,simulation_budget_s=600.,start_time_s=1.)


@pytest.mark.parametrize('directions',[(1,)*20,(1,)*20+(-1,)*5+(1,)*6])
def test_original_controller_real_adapter_typed_ACK_prehold_whole_path_and_single_fixed_hold(directions):
    host=SoftwareHost();controller=controller_for(directions);physical_initial=host.backend.physical_q.copy()
    passive={k:v[:,host.actuator.passive].copy() for k,v in host.backend.data.items()}
    result=asyncio.run(lifecycle.run_lifecycle_once(host,controller,authorized=True))
    assert result['verdict']=='PASS',result
    assert result['fixed_hold_attempts']==1 and result['new_route_trials']==1 and not result['automatic_retry']
    assert host.events.count('Play')==host.events.count('actual_Pause')==host.events.count('durable_save_fixture')==1
    assert host.events.index('actual_Pause')<host.events.index('unsubscribe')<host.events.index('postflight')
    assert host.actuator.closed and host.backend.paused and host.actuator.restoration_report['restored']
    rows=host.saved_rows;pre=[r for r in rows if r['phase']=='pre_hold']
    assert pre and all(r['command_speed_m_s']==0. for r in pre)
    held=[r for r in rows if r['phase']=='terminal_hold']
    assert held[0]['phase_time_s']==0. and 5.<=held[-1]['phase_time_s']<5.1
    assert sum(e['event']=='HOLD_REQUESTED' for e in host.actuator.events)==host.actuator.capture_count
    # Only the SOFTWARE body/active joints integrate; production code has no
    # physical-state setter, no passive-target edit or instantaneous reset.
    assert np.array_equal(host.backend.physical_q[0,host.actuator.passive],physical_initial[0,host.actuator.passive])
    assert all(np.array_equal(host.backend.data[k][:,host.actuator.passive],v) for k,v in passive.items())
    assert all(np.array_equal(host.backend.data[k][0],v) for k,v in host.actuator.original.items())
    saved=copy.deepcopy(host.saved)
    duplicate=asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(directions),authorized=True))
    assert duplicate['status']=='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION' and host.saved==saved
    assert host.events.count('Play')==1


@pytest.mark.parametrize('fault',['preflight','guard','zero_setter','cancel','pause','pause_contact','restoration','postflight','hold_partial'])
def test_whole_lifecycle_independent_Pause_restoration_closeout_and_no_retry_on_fault(fault):
    host=SoftwareHost(fault=fault);controller=controller_for()
    result=asyncio.run(lifecycle.run_lifecycle_once(host,controller,authorized=True))
    assert result['verdict']=='HOLD' and not result['automatic_retry']
    assert host.events.count('Play')==(0 if fault=='preflight' else 1)
    assert host.events.count('actual_Pause')==1 and host.events.count('durable_save_fixture')==1
    assert host.events.index('actual_Pause')<host.events.index('unsubscribe')
    assert ('postflight' in host.events)==(fault not in ('preflight','pause'))
    if fault=='pause':assert not host.actuator.closed and 'restoration' not in result
    elif fault!='preflight':assert host.actuator.closed


def test_evidence_write_error_cannot_skip_Pause_restore_or_enable_a_second_trial():
    host=SoftwareHost(fault='save')
    with pytest.raises(OSError):asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(),authorized=True))
    assert host.backend.paused and host.actuator.closed and host.actuator.restoration_report['restored']
    assert asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(),authorized=True))['status']=='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION'
    assert host.events.count('Play')==1


def test_no_whole_route_authority_no_host_or_actuator_action():
    host=SoftwareHost()
    with pytest.raises(ValueError):asyncio.run(lifecycle.run_lifecycle_once(host,controller_for()))
    assert host.events==[] and host.backend.writes==[]


def test_whole_cap_includes_prehold_and_fixed_hold_not_only_tracking_time():
    controller=controller_for()
    controller.start_time=-599.;controller.previous=1.
    with pytest.raises(TimeoutError):controller.step(1.01,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    assert controller.failed


@pytest.mark.parametrize('fault',['speed','pose','force','contact','callback','second_hold','missing_wheel','window_restart'])
def test_fixed_last3s_original_acceptance_never_uses_a_better_window_or_missing_wheel_data(fault):
    host=SoftwareHost();controller=controller_for()
    result=asyncio.run(lifecycle.run_lifecycle_once(host,controller,authorized=True));assert result['verdict']=='PASS'
    rows=copy.deepcopy(host.saved_rows);tail=[r for r in rows if r['phase']=='terminal_hold' and r['phase_time_s']>=2.]
    counts=copy.deepcopy(host.counts);errors=[];attempts=1
    if fault=='speed':tail[0]['vx_physx_m_s']=.005001
    elif fault=='pose':tail[0]['x_m']+=.031
    elif fault=='force':
        for row in tail:row['wheel_0_fz_n']=0.
    elif fault=='contact':counts['hospital_obstacle']=1
    elif fault=='callback':errors=['SOFTWARE_CALLBACK_ERROR']
    elif fault=='second_hold':attempts=2
    elif fault=='missing_wheel':tail[0].pop('wheel_3_fz_n')
    else:
        for row in rows:
            if row['phase']=='terminal_hold':row['phase_time_s']+=.1
    if fault in ('second_hold','missing_wheel','window_restart'):
        with pytest.raises((KeyError,ValueError)):
            lifecycle.evaluate_terminal_hold(rows,controller.goal,controller.reference.arc[-1],counts,errors,fixed_hold_attempts=attempts)
    else:
        checked=lifecycle.evaluate_terminal_hold(rows,controller.goal,controller.reference.arc[-1],counts,errors,fixed_hold_attempts=attempts)
        assert checked['verdict']=='FAIL'


def test_guard_before_first_actuation_refusal_never_writes_gains_or_targets_even_in_cleanup():
    host=SoftwareHost(fault='initial_guard')
    report=asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(),authorized=True))
    assert report['verdict']=='HOLD' and report['new_route_trials']==0
    assert host.backend.writes==[] and host.events.count('Play')==0
    assert host.events.count('actual_Pause')==1 and host.events.count('postflight')==1
    assert report['checks']['no_motor_write_restoration_required']


def test_all_four_real_native_H6_plans_use_same_prehold_and_unmodified_original_controller():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    folder=root/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_planning/heading0_fresh_terminal_beam_matched_dev_r19'
    from ijcas_paths import load_archived_json
    review=load_archived_json(folder/'planning_review.json')
    protocol=load_archived_json(review['protocol'])
    geometry=json.loads(Path(protocol['common_geometry']).read_text());contracts=[]
    for row in review['methods']:
        plan=json.loads(Path(row['plan_file']).read_text());original=copy.deepcopy(plan)
        c=lifecycle.MissionLifecycleController(plan,geometry,simulation_budget_s=600.)
        contracts.append(c.contract)
        first=c.step(0.,plan['start_world_xyyaw'],.004,(0.,0.),drive_hold_confirmed=False)
        assert first['drive_hold_requested'] and first['speed_m_s']==0. and first['curvature_1_m']==0.
        assert type(c.motion) is motion.MissionTrackingController and c.motion.tracking_step.mission_binding['AST_handoff_substitutions']==1
        assert plan==original and not c.contract['actual_motion_authorized']
    assert len(contracts)==4 and all(c==contracts[0] for c in contracts)


def test_prehold_observes_one_fixed_window_with_original_release_check_not_new_all_window_speed_rule():
    gate=lifecycle.MeasuredPreHold([0.,0.,0.])
    gate.step(0.,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=False)
    gate.step(.1,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)
    for i in range(1,21):
        speed=.006 if i<10 else .004
        out=gate.step(.1+i*.1,[0.,0.,0.],speed,(0.,0.),drive_hold_confirmed=True)
    assert out['preparation_ready'] and gate.held_since==.1 and gate.ready_at==pytest.approx(2.1)


def test_terminal_first2s_settling_not_silently_strengthened_or_used_to_restart_fixed_last3s():
    controller=controller_for();controller.lifecycle_stage='fixed_hold';controller.hold_started=1.
    controller.fixed_hold_attempts=1;controller.previous=1.
    pose=[controller.goal[0]+.031,controller.goal[1],controller.goal[2]]
    out=controller.step(1.1,pose,.006,(0.,0.),drive_hold_confirmed=True)
    assert out['phase']=='terminal_hold' and out['phase_time_s']==pytest.approx(.1)
    assert out['speed_m_s']==0.
    assert controller.hold_started==1. and controller.fixed_hold_attempts==1 and not controller.failed


def test_lifecycle_module_never_imports_SDK_or_assigns_physical_states_or_exposes_timeline_retry():
    import ast,inspect
    tree=ast.parse(inspect.getsource(lifecycle))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls&{'stop','reset','initialize','set_dof_positions','set_dof_velocities','set_world_pose',
        'set_world_poses','Save','Reload','Set'}
    owners=[]
    for node in tree.body:
        if not isinstance(node,ast.AsyncFunctionDef):continue
        calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='play']
        if calls:
            assert len(calls)==1
            assert not any(calls[0] in list(ast.walk(n)) for n in ast.walk(node) if isinstance(n,(ast.For,ast.While)))
            owners.append(node.name)
    assert owners==['run_held_static_once','run_lifecycle_once']
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(name and name.startswith(('omni','isaacsim')) for name in imports)


def test_NumPy_float64_channels_and_completed_report_use_shared_exact_normalizer_before_writer():
    from hospital_runtime_evidence_io import json_value
    host=SoftwareHost(fault='numpy_scalars')
    result=asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(),authorized=True))
    assert result['verdict']=='PASS'
    assert json_value(result)==result and json_value(host.saved_rows)==host.saved_rows
    assert all(type(value) in (str,float,int,bool) for row in host.saved_rows for value in row.values())


@pytest.mark.parametrize('fault',['nonfinite','late_nonfinite'])
def test_bad_sensor_channel_is_rejected_before_next_target_write_but_still_durably_recordable(fault):
    from hospital_runtime_evidence_io import json_value
    host=SoftwareHost(fault=fault)
    result=asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(),authorized=True))
    assert result['verdict']=='HOLD'
    if fault=='nonfinite':assert host.backend.writes==[] and host.events.count('Play')==0
    else:assert host.events.count('Play')==1 and host.actuator.closed and host.actuator.restoration_report['restored']
    bad=result['last_attempted_observation_before_Pause']['wheel_2_fz_n']
    assert bad['kind']=='INVALID_REJECTED_CHANNEL_NOT_A_NUMERIC_MEASUREMENT' and bad['nonfinite_label']=='NaN'
    if fault=='nonfinite':assert result['sample_count']==0
    assert json_value(result)==result and host.saved==result


def test_unserializable_preflight_is_blocked_before_any_PD_target_or_Play_call():
    host=SoftwareHost(fault='bad_preflight_evidence')
    report=asyncio.run(lifecycle.run_lifecycle_once(host,controller_for(),authorized=True))
    assert report['verdict']=='HOLD' and 'Unsupported evidence type' in report['error']
    assert host.backend.writes==[] and host.events.count('Play')==0 and 'postflight' not in host.events


def guard_fixture(*,occupied=None):
    plan,g=mission_fixture((1,)*20)
    g['same_three_authored_support_rectangles']=[
        [[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]],
        [[2.,-2.],[3.,-2.],[3.,2.],[2.,2.]],
        [[3.,-2.],[4.,-2.],[4.,2.],[3.,2.]]]
    mask=np.zeros((400,400),bool) if occupied is None else occupied
    return lifecycle.MissionRuntimeGuard(g,mask,[-2.,-2.],.010,simulation_budget_s=600.)


def guard_row(t=0.,x=0.,y=0.,yaw=0.):
    return dict(time_s=t,x_m=x,y_m=y,yaw_rad=yaw,roll_deg=0.,pitch_deg=0.,vx_physx_m_s=0.,vy_physx_m_s=0.)


def guard_info(speed=0.,phase='path_tracking'):return dict(speed_m_s=speed,phase=phase)


@pytest.mark.parametrize('fault',['contact','callback','tilt','speed','cap','gap','teleport','same_clock_edit','terminal_neighbourhood','command'])
def test_outer_actual_guard_blocks_contact_body_clock_terminal_and_command_faults(fault):
    guard=guard_fixture();row=guard_row();counts={};errors=[];info=guard_info()
    guard(row,info,counts,errors);row=guard_row(.1)
    if fault=='contact':counts['hospital_obstacle']=1
    elif fault=='callback':errors=['actual monitor error']
    elif fault=='tilt':row['pitch_deg']=2.00001
    elif fault=='speed':row['vx_physx_m_s']=.080001
    elif fault=='cap':row['time_s']=600.0001
    elif fault=='gap':row['time_s']=.251
    elif fault=='teleport':row['x_m']=.100001
    elif fault=='same_clock_edit':row['time_s']=0.;row['x_m']=.001
    elif fault=='terminal_neighbourhood':info=guard_info(phase='terminal_braking')
    else:info=guard_info(.05001)
    with pytest.raises((RuntimeError,ValueError,TimeoutError)):guard(row,info,counts,errors)
    assert guard.failed and guard.summary()['rejected']
    with pytest.raises(RuntimeError,match='terminal'):guard(guard_row(.2),guard_info(),{},[])


def test_swept_geometry_tests_intermediate_rotations_not_just_two_endpoints():
    from hospital_geometry_tools import pose_collision
    mask=np.zeros((400,400),bool);mask[248,281]=True
    first=(0.,0.,math.radians(-10.));last=(0.,0.,math.radians(10.))
    assert not pose_collision(mask,first,[-2.,-2.],.01) and not pose_collision(mask,last,[-2.,-2.],.01)
    guard=guard_fixture(occupied=mask);guard(guard_row(yaw=first[2]),guard_info(),{},[])
    with pytest.raises(RuntimeError,match='swept full-envelope'):guard(guard_row(.1,yaw=last[2]),guard_info(),{},[])
    assert guard.pose_checks>2


def test_authored_support_guard_uses_full_body_not_center_or_10mm_long_old_bug():
    guard=guard_fixture();guard.supports=[[[0.,0.],[.1,0.],[.1,.1],[0.,.1]]]
    with pytest.raises(RuntimeError,match='full-body authored support'):
        guard(guard_row(x=.05,y=.05),guard_info(),{},[])


def test_reverse_accounting_charges_previous_interval_and_zero_does_not_split_episode():
    guard=guard_fixture();guard(guard_row(),guard_info(-.04),{},[])
    guard(guard_row(.25,x=-.01),guard_info(0.),{},[])
    guard(guard_row(.5,x=-.02),guard_info(0.),{},[])
    guard(guard_row(.75,x=-.02),guard_info(0.),{},[])
    assert guard.reverse_episode==pytest.approx(.02) and guard.reverse_total==pytest.approx(.02)
    guard(guard_row(1.,x=-.01),guard_info(.04),{},[])
    guard(guard_row(1.25,x=0.),guard_info(.04),{},[])
    assert guard.reverse_episode==0. and guard.reverse_total==pytest.approx(.02)
    assert guard.direction_changes==1 and guard.summary()['sampled_reverse_distance_NOT_continuous_arc_upper_bound']


@pytest.mark.parametrize('forward_command',[False,True])
def test_reverse_episode_guard_also_counts_uncommanded_backwards_drift(forward_command):
    guard=guard_fixture();command=.04 if forward_command else -.04
    guard(guard_row(),guard_info(command),{},[])
    for i in range(1,51):guard(guard_row(.25*i,x=-.01*i),guard_info(command),{},[])
    with pytest.raises(RuntimeError,match='episode0.5m'):
        guard(guard_row(12.75,x=-.51),guard_info(command),{},[])


def test_four_direction_changes_include_terminal_corrections_and_zeros_never_erase_history():
    guard=guard_fixture();guard(guard_row(),guard_info(.01),{},[])
    t=0.
    for i in range(4):
        t+=.05;guard(guard_row(t),guard_info(0.),{},[])
        t+=.05;guard(guard_row(t),guard_info(-.01 if i%2==0 else .01),{},[])
    assert guard.direction_changes==4
    with pytest.raises(RuntimeError,match='maximum4'):
        guard(guard_row(t+.05),guard_info(-.01),{},[])


def test_actual_host_refuses_no_consent_before_path_read_or_any_SDK_access():
    with pytest.raises(ValueError,match='authority'):
        lifecycle.RetainedMissionSDKHost(None,{},'/does/not/exist',context=None,plan={},geometry={},
            occupied=None,origin=None,resolution=.01)


def retained_host_fixture(tmp_path,monkeypatch,*,fault=None,port_kind='route',fresh_root=False,play_delay_updates=0,top_xy_policy=False):
    """Real host + actual USD NOTICE API on isolated COPIES + fake PhysX only.

    Synthetic static report/empty raster are in pytest's private temp folder.
    Original4997 rows/145settings are test data, NOT a live-stage verification.
    Contact forces and held/rolling response are IDEAL fixture values only.
    """
    pytest.importorskip('pxr')
    import json,hashlib
    from types import SimpleNamespace
    from pxr import Usd,UsdGeom,Gf,Tf
    from test_hospital_mission_runtime import _retained_GET_sdk_fixture
    r=lifecycle.retained;f=_retained_GET_sdk_fixture();snapshot=f.snapshot
    for key in ('velocity','joint_velocities_rad_s'):snapshot[key]=[0.]*len(snapshot[key])
    snapshot['all_thirteen_link_velocities_at_CoM']=[[0.]*6 for _ in range(13)]
    if port_kind=='held_static':
        start=f.raw['start_xyyaw'];delta=np.asarray(start[:2])-np.asarray(snapshot['transform'][:2])
        for pose in snapshot['all_thirteen_link_world_transforms_xyzw']:
            pose[0]=float(pose[0]+delta[0]);pose[1]=float(pose[1]+delta[1])
        snapshot['transform'][:2]=list(start[:2])
    if fault=='retained_actual_HOLD':
        current=lifecycle.retained.read(lifecycle.retained.REVIEWED_ZERO_FOLDER/'paused_closeout_GET.json')
        snapshot.clear();snapshot.update(copy.deepcopy(current['SDK_GET']['actual_tensor_snapshot']))
    elif fault=='retained_speed_HOLD':
        snapshot['velocity'][0]=float(np.float32(.005366004333059057))
        for value in snapshot['all_thirteen_link_velocities_at_CoM']:value[0]=snapshot['velocity'][0]
    raw=copy.deepcopy(f.raw);raw['run_id']='SOFTWARE_ONLY_SYNTHETIC_STATIC_NOT_AN_ACTUAL_TRIAL'
    raw['actual_post_static_tensor_snapshot']=copy.deepcopy(snapshot)
    actor=raw['physics_preparation_snapshot']['new_robot_root'];start=raw['start_xyyaw']
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World')
    UsdGeom.Xform.Define(stage,actor).AddTranslateOp().Set(Gf.Vec3d(start[0],start[1],0.))
    from pxr import Sdf
    for name in snapshot['link_names']:
        prim=UsdGeom.Xform.Define(stage,actor+'/'+name)
        prim.AddTranslateOp();prim.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        for attr in ('physics:velocity','physics:angularVelocity'):
            prim.GetPrim().CreateAttribute(attr,Sdf.ValueTypeNames.Vector3f,custom=False)
    for name in motion.RetainedTensorDriveAdapter.STEERING:
        prim=UsdGeom.Xform.Define(stage,actor+'/joints/'+name).GetPrim()
        for key in ('position','velocity'):prim.CreateAttribute('state:angular:physics:'+key,Sdf.ValueTypeNames.Float,custom=False)
    def write_state_COPY():
        for i,name in enumerate(snapshot['link_names']):
            prim=stage.GetPrimAtPath(actor+'/'+name);pose=snapshot['all_thirteen_link_world_transforms_xyzw'][i]
            prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(pose[0]-start[0],pose[1]-start[1],pose[2]))
            q=np.asarray(pose[3:],float);q/=np.linalg.norm(q)
            prim.GetAttribute('xformOp:orient').Set(Gf.Quatd(q[3],Gf.Vec3d(*q[:3])))
            v=snapshot['all_thirteen_link_velocities_at_CoM'][i]
            prim.GetAttribute('physics:velocity').Set(Gf.Vec3f(*v[:3]))
            prim.GetAttribute('physics:angularVelocity').Set(Gf.Vec3f(*np.degrees(v[3:])))
        for name in motion.RetainedTensorDriveAdapter.STEERING:
            index=snapshot['names'].index(name);prim=stage.GetPrimAtPath(actor+'/joints/'+name)
            prim.GetAttribute('state:angular:physics:position').Set(math.degrees(snapshot['positions'][index]))
            prim.GetAttribute('state:angular:physics:velocity').Set(math.degrees(snapshot['joint_velocities_rad_s'][index]))
    write_state_COPY()
    source_asset=None
    if fresh_root:
        assert port_kind=='held_static'
        # An anonymous reference mirrors the actual fresh source composition:
        # actor is in Session, with NO state opinions in Root before Play.
        source_asset=Usd.Stage.CreateInMemory()
        assert Sdf.CopySpec(stage.GetRootLayer(),actor,source_asset.GetRootLayer(),'/Source')
        remove=Sdf.BatchNamespaceEdit();remove.Add(actor,Sdf.Path.emptyPath);assert stage.GetRootLayer().Apply(remove)
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            stage.DefinePrim(actor,'Xform').GetReferences().AddReference(source_asset.GetRootLayer().identifier,'/Source')
    if top_xy_policy:
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            camera=UsdGeom.Camera.Define(stage,'/OmniverseKit_Top')
            camera.AddTransformOp().Set(Gf.Matrix4d(1.).SetTranslate(Gf.Vec3d(10.,12.,50001.)))
    renderer_layer=None;renderer_stage=None;renderer_attribute=None
    if fault and fault.startswith('renderer_'):
        from pxr import UsdRender
        renderer_layer=Sdf.Layer.CreateAnonymous('SOFTWARE_VIEWPORT_RENDERER.usda')
        renderer_stage=Usd.Stage.Open(renderer_layer)
        product=UsdRender.Product.Define(renderer_stage,'/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0').GetPrim()
        renderer_attribute=product.CreateAttribute('omni:rtx:rt:ecoMode:enabled',Sdf.ValueTypeNames.Bool,custom=False)
        renderer_attribute.Set(True)
        stage.OverridePrim('/Render').GetReferences().AddReference(renderer_layer.identifier,'/Render')
        if fault in ('renderer_eco_root','renderer_eco_fallback_root'):
            renderer_attribute=stage.GetAttributeAtPath(str(renderer_attribute.GetPath()))
            renderer_attribute.Set(True)
            if fault=='renderer_eco_fallback_root':renderer_attribute.Clear()
    sdk=f.sdk;callbacks={};state={'time':0. if port_kind=='held_static' else raw['actual_final_timeline']['current_time_s'],
        'playing':False,'stopped':port_kind=='held_static','pause_pending':False,'clock':0.,'play_pending':0}
    events=[];writes=[];port_box={}
    class Timeline:
        def is_playing(self):return state['playing']
        def is_stopped(self):return state['stopped']
        def get_current_time(self):return state['time']
        def play(self):
            events.append('Play')
            if play_delay_updates:state['play_pending']=play_delay_updates
            else:state['playing']=True;state['stopped']=False
        def pause(self):events.append('Pause_requested');state['pause_pending']=True
        def get_timeline_event_stream(self):
            return SimpleNamespace(create_subscription_to_pop=lambda fn,**kwargs:callbacks.update(timeline=fn) or object())
    timeline=Timeline()
    class App:
        async def next_update_async(self):
            if state['pause_pending']:
                state['playing']=False;state['pause_pending']=False;state['play_pending']=0;events.append('Pause_observed')
            if state['play_pending']:
                state['play_pending']-=1
                if not state['play_pending']:
                    state['playing']=True;state['stopped']=False
                    if 'timeline' in callbacks:callbacks['timeline'](SimpleNamespace(type=0))
            if not state['playing']:return
            host=port_box['host'];adapter=host.adapter
            steer_indices=adapter.steering if adapter else [snapshot['names'].index(n) for n in motion.RetainedTensorDriveAdapter.STEERING]
            drive_indices=adapter.drive if adapter else [snapshot['names'].index(n) for n in motion.RetainedTensorDriveAdapter.DRIVES]
            for _ in range(2):
                dt=1./120.;state['clock']+=dt;state['time']+=dt
                steer=np.array(snapshot['joint_position_targets_rad'])[steer_indices]
                wheels=np.array(snapshot['joint_velocity_targets_rad_s'])[drive_indices]
                held=bool(np.any(np.array(snapshot['gains']['kp'])[drive_indices]>0.))
                u=0. if held else float(np.mean(wheels*.074*np.cos(steer)))
                dx=u*dt;dy=0.
                snapshot['transform'][0]+=dx;snapshot['transform'][1]+=dy
                snapshot['velocity']=[u,0.,0.,0.,0.,0.]
                for i,pose in enumerate(snapshot['all_thirteen_link_world_transforms_xyzw']):
                    pose[0]+=dx;pose[1]+=dy;snapshot['all_thirteen_link_velocities_at_CoM'][i]=[u,0.,0.,0.,0.,0.]
                for i,value in zip(steer_indices,steer):snapshot['positions'][int(i)]=float(value)
                for i,value in zip(drive_indices,wheels):
                    value=0. if held else value;snapshot['joint_velocities_rad_s'][int(i)]=float(value)
                    snapshot['positions'][int(i)]+=float(value*dt)
                callbacks['step'](dt);write_state_COPY()
                collider=actor+'/'+next(iter(host.context.stationary.WHEELS.values()))
                if fault=='contact' and state['clock']>.25:collider=actor+'/base_link'
                callbacks['contacts']([SimpleNamespace(collider0=collider,collider1=host.context.stationary.SUPPORTS[0])],None)
                if fault=='USD_config' and state['clock']>.25 and not state.get('injected'):
                    state['injected']=True
                    stage.GetPrimAtPath(actor+'/joints/'+motion.RetainedTensorDriveAdapter.STEERING[0]).CreateAttribute(
                        'drive:angular:physics:stiffness',Sdf.ValueTypeNames.Float).Set(999.)
                if renderer_attribute and state['clock']>.025 and not state.get('renderer_injected'):
                    state['renderer_injected']=True
                    if fault=='renderer_metadata':renderer_attribute.SetDocumentation('NOT_A_DEFAULT')
                    elif fault=='renderer_other':product.CreateAttribute('omni:rtx:rt:UNREVIEWED',Sdf.ValueTypeNames.Bool).Set(False)
                    else:renderer_attribute.Set(False)
    app=App()
    for setter,key in (('set_dof_stiffnesses',('gains','kp')),('set_dof_dampings',('gains','kd')),
            ('set_dof_position_targets',('joint_position_targets_rad',)),('set_dof_velocity_targets',('joint_velocity_targets_rad_s',))):
        def put(data,indices,key=key,setter=setter):
            assert data.dtype==np.float32 and data.shape==(1,12) and indices.tolist()==[0]
            writes.append(setter)
            if len(key)==2:snapshot[key[0]][key[1]]=data[0].tolist()
            else:snapshot[key[0]]=data[0].tolist()
        setattr(f.view,setter,put)
    def contact_view(path):
        def forces(dt):
            assert dt==pytest.approx(1./120.)
            return np.array([[0.,0.,265.*9.81/6.]])
        return SimpleNamespace(sensor_count=1,sensor_paths=[path],get_net_contact_forces=forces)
    simulation=SimpleNamespace(create_articulation_view=lambda path:f.view,create_rigid_contact_view=contact_view,device='cpu')
    def physics_view():
        if port_kind=='held_static':assert state['playing'],'NEW physics views may bind ONLY after ONE approved Play'
        if fault=='binding_never':return None
        return simulation
    sdk.SimulationManager=SimpleNamespace(get_physics_sim_view=physics_view,get_backend=lambda:'numpy')
    sdk.omni.usd.get_context=lambda:SimpleNamespace(get_stage=lambda:stage)
    sdk.omni.timeline.get_timeline_interface=lambda:timeline
    sdk.omni.timeline.TimelineEventType=SimpleNamespace(PLAY=SimpleNamespace(value=0),PAUSE=SimpleNamespace(value=1),STOP=SimpleNamespace(value=2))
    sdk.omni.kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda:app))
    sim=SimpleNamespace(subscribe_contact_report_events=lambda fn:callbacks.update(contacts=fn) or object(),wake_up=lambda *args:None)
    sdk.omni.physx=SimpleNamespace(get_physx_interface=lambda:SimpleNamespace(
        subscribe_physics_step_events=lambda fn:callbacks.update(step=fn) or object()),get_physx_simulation_interface=lambda:sim)
    sdk.PhysicsSchemaTools=SimpleNamespace(sdfPathToInt=lambda v:v,intToSdfPath=lambda v:v)
    sdk.UsdUtils=SimpleNamespace(StageCache=SimpleNamespace(Get=lambda:SimpleNamespace(GetId=lambda st:SimpleNamespace(ToLongInt=lambda:1))))
    sdk.Tf=Tf;sdk.Usd=Usd
    import importlib
    original_import=importlib.import_module
    monkeypatch.setattr(importlib,'import_module',lambda name:SimpleNamespace(convert=lambda a,**kw:a.copy())
        if name=='isaacsim.core.utils.numpy' else original_import(name))
    context=r.reviewed_static_context(r.read(r.REVIEWED_STATIC_FOLDER/'static_protocol.json'),r.REVIEWED_STATIC_FOLDER/'static_protocol.json')
    context.preparation=SimpleNamespace(**vars(context.preparation));context.preparation.compact=SimpleNamespace(**vars(context.preparation.compact))
    compact=r.read(r.H6/'actual_common_environment_rows.json')['compact_rows']
    context.preparation.compact.stage_rows=lambda *args:iter(copy.deepcopy(compact).items())
    context.preparation.LIVE=tmp_path/'preparation.json'
    prep=copy.deepcopy(raw['physics_preparation_snapshot'])
    if port_kind=='held_static':
        prep.update(Root_sha256=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest(),
            Session_sha256=hashlib.sha256(stage.GetSessionLayer().ExportToString().encode()).hexdigest())
    r.evidence.write_exclusive(context.preparation.LIVE,prep)
    stationary=SimpleNamespace(**vars(context.stationary));context.stationary=stationary
    stationary.actor_configuration=lambda st:raw['runtime_anonymous_source_reference']['fixed_configuration_rows']
    stationary.critical_environment=lambda st,p:raw['environment_after_static']
    static_file=tmp_path/'SOFTWARE_ONLY_static.json';static_csv=tmp_path/'SOFTWARE_ONLY_static_trace.csv'
    static_csv.write_text('SOFTWARE FIXTURE; NOT AN ACTUAL STATIC TRIAL\n',encoding='utf-8')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    raw['trace_sha256']=sha(static_csv);r.evidence.write_exclusive(static_file,raw)
    stationary.RESULT=static_file;stationary.TRACE=static_csv
    def replay_fixture(unused):
        if fault=='raw_replay':raise ValueError('SOFTWARE_RAW_CSV_QUERY_REPLAY_FAILURE')
        return prep,raw,dict(verdict='PASS',SOFTWARE_FIXTURE_ONLY_NOT_ACTUAL_RAW_REPLAY=True)
    context.operations=SimpleNamespace(replay_static=replay_fixture)
    plan,g=mission_fixture((1,)*20)
    plan['start_world_xyyaw']=list(start);plan['goal_world_xyyaw']=[start[0]+1.,start[1],0.]
    g['start_xyyaw']=list(start);g['goal_xyyaw']=list(plan['goal_world_xyyaw'])
    for item in plan['selected_states']:item['world_xyyaw']=[start[0]+item['x_m'],start[1]+item['y_m'],0.]
    g['same_three_authored_support_rectangles']=guard_fixture().supports
    g['same_three_authored_support_rectangles']=[[[a+start[0],b+start[1]] for a,b in polygon]
        for polygon in g['same_three_authored_support_rectangles']]
    occupied=np.zeros((600,600),bool);origin=np.array([start[0]-2.,start[1]-2.]);resolution=.01
    digest=hashlib.sha256(occupied.tobytes()).hexdigest();plan['planning_occupancy_digest']=g['planning_occupancy_digest']=digest
    grid=tmp_path/'SOFTWARE_ONLY_empty_grid.npz';np.savez(grid,occupied=occupied,origin_xy_m=origin,resolution_m=resolution)
    geometry_file=tmp_path/'SOFTWARE_ONLY_geometry.json';plan_file=tmp_path/'SOFTWARE_ONLY_plan.json'
    r.evidence.write_exclusive(geometry_file,g);r.evidence.write_exclusive(plan_file,plan)
    source_files=(static_file,static_csv,context.preparation.LIVE,geometry_file,plan_file,grid,*lifecycle.host_source_files())
    protocol=dict(version=lifecycle.RetainedMissionSDKHost.VERSION,maximum_Play_calls=1,maximum_Stop_reset_restaging_calls=0,
        automatic_retry=False,actual_route_authorized=True,use_for_final_statistics=False,
        explicit_route_approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_REAL_USER_AUTHORITY',answer_ko='SOFTWARE ONLY'),
        common_lifecycle_contract=lifecycle.lifecycle_contract(simulation_budget_s=600.),simulation_budget_s=600.,
        actor=actor,plan_filename=plan_file.name,original_static_record=str(static_file),original_static_trace=str(static_csv),
        geometry_file=str(geometry_file),grid_file=str(grid),required_PAUSED_timeline=raw['actual_final_timeline'],
        required_tensor_snapshot=copy.deepcopy(snapshot),required_layer_sha256=[
            hashlib.sha256(layer.ExportToString().encode()).hexdigest() for layer in (stage.GetRootLayer(),stage.GetSessionLayer())],
        input_sha256={str(p):sha(p) for p in source_files})
    if top_xy_policy:
        protocol['reviewed_display_boundary']=dict(reference_Session_text=stage.GetSessionLayer().ExportToString(),
            unchanged_Root_sha256=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest(),
            reviewed_default_paths=list(r.TOP_XY_DISPLAY_PATHS),policy=r.TOP_XY_DISPLAY_POLICY)
    if port_kind=='held_static':
        layer_file=tmp_path/'SOFTWARE_ONLY_preparation_layers.json'
        r.evidence.write_exclusive(layer_file,dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString()))
        display_file=tmp_path/'SOFTWARE_ONLY_FIRST_display.json'
        r.evidence.write_exclusive(display_file,dict(actor=actor,registered_visual_root_count=13,visible_USD_shape_count=22,
            original_Root_and_Session_text_exactly_restored=True,timeline_before_and_after=dict(r.boundary.STOPPED),
            SOFTWARE_FIXTURE_NOT_VISIBLE_PIXEL_OR_ACTUAL_RENDER_REGISTRATION_PROOF=True))
        q=r.read(r.H6/'static_protocol.json');q['new_robot_root']=actor
        q['source_actor_node_paths']=list(prep['source_reference']['fixed_configuration_rows'])
        files=(context.preparation.LIVE,layer_file,display_file,r.H6/'actual_common_environment_rows.json',geometry_file,grid,
            context.preparation.ASSET,r.H6/'static_protocol.json',*lifecycle.host_source_files())
        protocol=dict(version=lifecycle.PreparedHeldStaticSDKHost.VERSION,maximum_Play_calls=1,maximum_Stop_reset_restaging_calls=0,
            maximum_route_trials=0,automatic_retry=False,actual_held_static_authorized=True,use_for_final_statistics=False,
            explicit_static_approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_REAL_USER_AUTHORITY',answer_ko='새 코호트 정지유지 1회 승인'),
            held_static_contract=lifecycle.held_static_contract(),required_STOPPED_timeline=dict(r.boundary.STOPPED),actor=actor,
            preparation_layers_file=str(layer_file),display_receipt=str(display_file),environment_rows_file=str(r.H6/'actual_common_environment_rows.json'),
            geometry_file=str(geometry_file),grid_file=str(grid),query_contract=q,input_sha256={str(p):sha(p) for p in files})
        sdk.carb=SimpleNamespace(Float3=lambda *args:args)
        monkeypatch.setattr(r.runtime,'query_functions',lambda scope:(lambda:copy.deepcopy(raw['actual_sampled_cooked_support_queries']),
            lambda:copy.deepcopy(raw['actual_disabled_subfloor_probes'])))
        monkeypatch.setattr(r.shared,'fresh_source_for_live_actor',lambda c,st:copy.deepcopy(prep['source_reference']))
        monkeypatch.setattr(r.shared.preparation.old,'constructor_for',lambda actor:SimpleNamespace(__globals__={
            'validate_new_robot':lambda *args:copy.deepcopy(prep['source_constructor_readback'])}))
        path=tmp_path/'SOFTWARE_ONLY_held_static_packet.json';r.evidence.write_exclusive(path,protocol)
        host=lifecycle.PreparedHeldStaticSDKHost(sdk,protocol,path,context=context,geometry=g,
            occupied=occupied,origin=origin,resolution=resolution,authorized=True)
        controller=lifecycle.HeldStaticController(start)
    else:
        path=tmp_path/'SOFTWARE_ONLY_route_packet.json';r.evidence.write_exclusive(path,protocol)
        host=lifecycle.RetainedMissionSDKHost(sdk,protocol,path,context=context,plan=plan,geometry=g,
            occupied=occupied,origin=origin,resolution=resolution,authorized=True)
        controller=lifecycle.MissionLifecycleController(plan,g,simulation_budget_s=600.)
    port_box['host']=host
    return SimpleNamespace(host=host,controller=controller,events=events,writes=writes,snapshot=snapshot,stage=stage,
        callbacks=callbacks,protocol=protocol,path=path,raw=raw,state=state,source_asset=source_asset,simulation=simulation,
        renderer_layer=renderer_layer,renderer_stage=renderer_stage)


def test_REAL_SDK_host_full4997_pipeline_actual_USD_notice_original_controller_and_backend_ACK_SOFTWARE_only(tmp_path,monkeypatch):
    f=retained_host_fixture(tmp_path,monkeypatch)
    result=asyncio.run(lifecycle.run_lifecycle_once(f.host,f.controller,authorized=True))
    assert result['verdict']=='PASS',result
    saved=lifecycle.retained.read(f.host.result)
    assert saved['actual_final_readback']['stable_PAUSED_tensor']
    assert saved['postflight']['checks']['all4997_environment_rows_and145_source_settings_and_physics_unchanged']
    assert saved['postflight']['exact_post_observation_boundary_audits']
    assert saved['runtime_guard']['swept_pose_checks']>1000
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert f.host.adapter.closed and f.host.notice_sub is f.host.physics_sub is f.host.contact_sub is None
    assert saved['ground_truth_development_NOT_sensor_Nav2_or_final_repetition']
    before=list(f.writes)
    duplicate=asyncio.run(lifecycle.run_lifecycle_once(f.host,f.controller,authorized=True))
    assert duplicate['status']=='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION' and f.writes==before


@pytest.mark.parametrize('fault',['contact','USD_config'])
def test_REAL_SDK_host_faults_preserve_Pause_gain_restoration_and_failure_readback_without_another_GET(tmp_path,monkeypatch,fault):
    f=retained_host_fixture(tmp_path,monkeypatch,fault=fault)
    report=asyncio.run(lifecycle.run_lifecycle_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='HOLD'
    saved=lifecycle.retained.read(f.host.result)
    assert f.events.count('Play')==f.events.count('Pause_observed')==1 and f.host.adapter.closed
    assert saved['actual_final_readback']['stable_PAUSED_tensor']
    assert saved['restoration']['restored']
    if fault=='contact':assert saved['contact_counts']['nonwheel_support']>0
    else:assert saved['callback_errors'] and 'postflight_error' in saved


@pytest.mark.parametrize('fault',['retained_actual_HOLD','retained_speed_HOLD'])
def test_REAL_SDK_host_rejects_actual_measured_HOLD_pose_or_speed_even_with_matching_raw_hashes_before_Play(tmp_path,monkeypatch,fault):
    f=retained_host_fixture(tmp_path,monkeypatch,fault=fault)
    result=asyncio.run(lifecycle.run_lifecycle_once(f.host,f.controller,authorized=True))
    assert result['verdict']=='HOLD' and 'release failed' in result['error']
    assert f.events==[] and f.writes==[] and result['new_route_trials']==0
    assert lifecycle.retained.read(f.host.result)['actual_final_readback']['stable_PAUSED_tensor']


def test_REAL_SDK_host_raw_static_replay_failure_prevents_any_control_or_Play(tmp_path,monkeypatch):
    f=retained_host_fixture(tmp_path,monkeypatch,fault='raw_replay')
    report=asyncio.run(lifecycle.run_lifecycle_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='HOLD' and 'RAW_CSV_QUERY' in report['error']
    assert f.writes==f.events==[] and report['new_route_trials']==0


@pytest.mark.parametrize('change',['XY','Z','rotation','metadata'])
def test_ROUTE_notice_and_complete_postflight_use_the_SAME_explicit_typed_camera_policy_SOFTWARE_only(tmp_path,monkeypatch,change):
    pytest.importorskip('pxr')
    from pxr import Gf,Usd
    f=retained_host_fixture(tmp_path,monkeypatch,top_xy_policy=True)
    app=f.host.app;original=app.next_update_async;injected=False
    async def changed_frame():
        nonlocal injected
        await original()
        if f.state['playing'] and f.state['clock']>.25 and not injected:
            injected=True
            with Usd.EditContext(f.stage,f.stage.GetSessionLayer()):
                attr=f.stage.GetAttributeAtPath('/OmniverseKit_Top.xformOp:transform')
                if change=='metadata':attr.SetDocumentation('UNREVIEWED')
                else:
                    values=[list(row) for row in attr.Get()]
                    if change=='XY':values[3][0]+=.4;values[3][1]-=.7
                    elif change=='Z':values[3][2]+=.1
                    elif change=='rotation':values[0][0]+=.1
                    assert attr.Set(Gf.Matrix4d(*values))
    monkeypatch.setattr(app,'next_update_async',changed_frame)
    report=asyncio.run(lifecycle.run_lifecycle_once(f.host,f.controller,authorized=True))
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert report['checks']['actual_Pause_confirmed'] and report['checks']['original_gains_targets_restored']
    if change=='XY':
        assert report['verdict']=='PASS',report
        assert f.host.notice_counts['reviewed_Top_XY_defaults']==1
        assert report['checks']['all4997_environment_rows_and145_source_settings_and_physics_unchanged']
        assert any(v['display_review']['exact_Top_default_differences'] for v in report['postflight']['exact_post_observation_boundary_audits'])
    else:
        assert report['verdict']=='HOLD' and report['callback_errors']
        assert not report['checks']['complete_postflight_verified']


@pytest.mark.parametrize('fresh_root',[False,True])
def test_PREPARED_SDK_host_own_STOPPED_preflight_FIRST_Play_binding_hold_window_and_closeout_SOFTWARE_only(tmp_path,monkeypatch,fresh_root):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=fresh_root)
    if fresh_root:assert not f.stage.GetRootLayer().GetPrimAtPath(f.protocol['actor'])
    result=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert result['verdict']=='PASS',result
    saved=lifecycle.retained.read(f.host.result)
    assert saved['method']=='COMMON_HELD_STATIC_ENGINEERING_NOT_ROUTE'
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert f.host.adapter.capture_count==1 and f.host.adapter.closed
    assert saved['preflight']['no_physics_view_initialized_before_Play']
    assert saved['postflight']['first_source_state_writeback_proofs']
    assert saved['postflight']['checks']['all4997_rows145_source_settings_and_original_physics_retained']
    assert saved['actual_final_readback']['stable_PAUSED_tensor']
    assert saved['new_route_trials']==0 and not saved['automatic_retry']
    replay=lifecycle.replay_held_static_evidence(f.host.result)
    assert replay['verdict']=='PASS' and replay['new_PhysX_trials']==0
    assert replay['raw_CSV_replay_NOT_an_independent_live_environment_or_SDK_revalidation']


@pytest.mark.parametrize('fault',['contact','USD_config'])
def test_PREPARED_SDK_host_actual_notice_or_contact_fault_durably_keeps_HOLD_and_Pause_restoration(tmp_path,monkeypatch,fault):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fault=fault)
    result=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert result['verdict']=='HOLD',result
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert f.host.adapter.closed and result['restoration']['restored']
    assert lifecycle.retained.read(f.host.result)['actual_final_readback']['stable_PAUSED_tensor']


def test_PREPARED_SDK_host_FIRST_binding_timeout_Pauses_without_motor_writes_or_another_Play(tmp_path,monkeypatch):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,fault='binding_never')
    result=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert result['verdict']=='HOLD' and 'Bounded initial physics binding' in result['error']
    assert f.events.count('Play')==f.events.count('Pause_observed')==1 and f.writes==[]
    saved=lifecycle.retained.read(f.host.result)
    assert saved['actual_final_readback']['tensor_NOT_bound_NO_fabricated_physical_snapshot']
    assert not saved['actual_final_timeline']['playing'] and f.host.clock<.30
    assert saved['sample_count']==0 and not saved['checks']['complete_postflight_verified']


def test_PREPARED_SDK_host_prePlay_opaque_Session_change_cannot_hide_behind_native_field_classifier(tmp_path,monkeypatch):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True)
    f.stage.GetSessionLayer().comment='UNREVIEWED_OPAQUE_EDIT_NOT_CAMERA_OR_STATE'
    result=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert result['verdict']=='HOLD' and any(text in result['error'] for text in ('canonical layer content','/:comment'))
    assert f.events==f.writes==[] and result['new_static_trials']==0
    saved=lifecycle.retained.read(f.host.result)
    assert saved['actual_final_timeline']['stopped'] and 'actual_final_readback' in saved


def test_PREPARED_SDK_host_no_default_authority_before_SDK_access_or_file_binding():
    with pytest.raises(ValueError,match='held-static authority'):
        lifecycle.PreparedHeldStaticSDKHost(None,{},'/MUST_NOT_READ',context=None,geometry=None,
            occupied=None,origin=None,resolution=None)


@pytest.mark.parametrize('port_kind',['route','held_static'])
@pytest.mark.parametrize('location',['external','Root','Root_composed_fallback'])
def test_VIEWPORT_renderer_ECO_default_in_own_layer_is_recorded_not_a_physics_HOLD(tmp_path,monkeypatch,port_kind,location):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind=port_kind,fresh_root=port_kind=='held_static',
        fault={'external':'renderer_eco','Root':'renderer_eco_root','Root_composed_fallback':'renderer_eco_fallback_root'}[location])
    runner=lifecycle.run_held_static_once if port_kind=='held_static' else lifecycle.run_lifecycle_once
    report=asyncio.run(runner(f.host,f.controller,authorized=True))
    assert report['verdict']=='PASS',report
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert f.host.notice_counts['reviewed_viewport_renderer_defaults']==1
    audit=[v for v in f.host.display_audits if v.get('classification')=='VIEWPORT_RENDERER_IDLE_FRAME_POLICY_NOT_PHYSICS_OR_GEOMETRY']
    assert len(audit)==1 and audit[0]['current_value'] is False
    assert audit[0]['authored_in_protected_layer']==(location!='external')
    assert f.host.adapter.closed and report['restoration']['restored']


def test_VIEWPORT_existing_Root_ECO_change_BEFORE_Play_is_copy_proved_and_not_a_preflight_HOLD(tmp_path,monkeypatch):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,fault='renderer_eco_root')
    f.stage.GetAttributeAtPath('/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0.omni:rtx:rt:ecoMode:enabled').Set(False)
    report=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='PASS',report
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert any(audit.get('exact_renderer_default_differences') for audit in f.host.display_audits)


@pytest.mark.parametrize('fault',['renderer_metadata','renderer_other'])
def test_VIEWPORT_ECO_classification_does_not_ignore_metadata_or_other_Render_changes(tmp_path,monkeypatch,fault):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,fault=fault)
    report=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='HOLD' and report['callback_errors']
    assert report['checks']['actual_Pause_confirmed'] and report['restoration']['restored']


@pytest.mark.parametrize('fault',['wrong_owner','wrong_type','time_samples','metadata','unrelated_path'])
def test_VIEWPORT_ECO_typed_exact_default_boundary_has_no_subtree_or_experiment_layer_waiver(fault):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdRender,Sdf
    stage=Usd.Stage.CreateInMemory();external=Sdf.Layer.CreateAnonymous('SOFTWARE_RENDERER_ONLY.usda')
    stage.GetSessionLayer().subLayerPaths.append(external.identifier)
    path='/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0'
    with Usd.EditContext(stage,external):
        prim=stage.DefinePrim(path,'Scope') if fault=='wrong_owner' else UsdRender.Product.Define(stage,path).GetPrim()
        attr=prim.CreateAttribute('omni:rtx:rt:ecoMode:enabled',Sdf.ValueTypeNames.String if fault=='wrong_type' else Sdf.ValueTypeNames.Bool)
        attr.Set('False' if fault=='wrong_type' else False)
        if fault=='time_samples':attr.Set(True,Usd.TimeCode(1.))
    fields=['documentation'] if fault=='metadata' else ['default']
    if fault=='unrelated_path':
        assert lifecycle.viewport_renderer_default_review(stage,'/World/robot.omni:rtx:rt:ecoMode:enabled',fields) is None
    else:
        with pytest.raises(ValueError):lifecycle.viewport_renderer_default_review(stage,str(attr.GetPath()),fields)


@pytest.mark.parametrize('delay',[1,3])
def test_PREPARED_SDK_host_deferred_Play_is_pending_NOT_an_unexpected_Stop(tmp_path,monkeypatch,delay):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,play_delay_updates=delay)
    report=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='PASS',report
    assert f.events.count('Play')==f.events.count('Pause_observed')==1
    assert f.host.adapter.capture_count==1 and f.host.adapter.closed
    assert report['new_route_trials']==0 and not f.state['playing'] and not f.state['play_pending']


@pytest.mark.parametrize('failure',['request_error','cancel_before_first_update','pending_timeout','wall_timeout'])
def test_PREPARED_pending_Play_failure_cancels_queue_without_ghost_Play_or_second_trial(tmp_path,monkeypatch,failure):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,play_delay_updates=1000)
    play=f.host.timeline.play
    def request():
        play()
        if failure=='request_error':raise RuntimeError('SOFTWARE_REQUEST_AFTER_QUEUE_ERROR')
        if failure=='wall_timeout':f.host.motion_wall-=16.
    monkeypatch.setattr(f.host.timeline,'play',request)
    if failure=='cancel_before_first_update':
        async def canceled_binding():raise asyncio.CancelledError('SOFTWARE_CANCEL_PENDING_PLAY')
        monkeypatch.setattr(f.host,'bind_after_play',canceled_binding)
    report=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='HOLD' and report['sample_count']==0
    assert f.events.count('Play')==f.events.count('Pause_requested')==1
    assert not f.state['play_pending'] and not f.state['playing'] and f.state['stopped']
    assert not report['checks']['actual_Pause_confirmed'] and 'STOPPED' in report['pause_error']
    assert f.writes==[] and f.host.adapter is None and f.host.tensor is None
    saved=lifecycle.retained.read(f.host.result)
    assert saved['first_Play_transition']['pause_cleanup']['actual_PAUSED_confirmed'] is False
    assert saved['first_Play_transition']['still_pending'] is False
    assert f.host.timeline_sub is None
    before=list(f.events)
    assert asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))['status']=='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION'
    assert f.events==before


def test_PREPARED_actual_Stop_after_Play_event_is_not_waived_as_pending(tmp_path,monkeypatch):
    from types import SimpleNamespace
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,play_delay_updates=1)
    update=f.host.app.next_update_async
    async def stop_after_frame():
        await update()
        if f.state['playing']:
            f.state.update(playing=False,stopped=True,time=0.)
            f.callbacks['timeline'](SimpleNamespace(type=2))
    monkeypatch.setattr(f.host.app,'next_update_async',stop_after_frame)
    report=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='HOLD' and 'Unexpected Stop' in report['error']
    assert f.events.count('Play')==1 and f.writes==[]
    assert not report['checks']['actual_Pause_confirmed']


def test_PREPARED_stage_change_during_pending_Play_does_not_command_another_stage(tmp_path,monkeypatch):
    from types import SimpleNamespace
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,play_delay_updates=3)
    play=f.host.timeline.play
    def replaced_stage():
        play();f.host.usd=SimpleNamespace(get_stage=lambda:None)
    monkeypatch.setattr(f.host.timeline,'play',replaced_stage)
    report=asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    assert report['verdict']=='HOLD' and 'Active stage changed' in report['error']
    assert f.events==['Play'] and f.writes==[]
    assert not report['checks']['actual_Pause_confirmed'] and 'Active stage changed' in report['pause_error']


@pytest.mark.parametrize('initial',['playing','paused','stopped','pending'])
def test_Pause_only_cleanup_stable_state_no_Play_Stop_or_physics_actions(initial):
    state=dict(playing=initial=='playing',stopped=initial in ('stopped','pending'),current_time_s=0. if initial in ('stopped','pending') else 2.)
    calls=[];pending=[]
    class Timeline:
        def is_playing(self):return state['playing']
        def is_stopped(self):return state['stopped']
        def get_current_time(self):return state['current_time_s']
        def pause(self):calls.append('Pause');pending.append(True)
    async def update():
        if pending:state['playing']=False;pending.clear()
    receipt=asyncio.run(lifecycle.pause_and_confirm_nonplaying(Timeline(),update,pending_play=initial=='pending'))
    assert calls==(['Pause'] if initial in ('playing','pending') else [])
    assert receipt['after']==state and receipt['updates_to_observe_stable_nonplaying']==3
    assert receipt['actual_PAUSED_confirmed']==(initial in ('playing','paused'))
    assert not receipt['any_Play_Stop_reset_or_trial']


def test_Pause_only_cleanup_stuck_Playback_times_out_without_retry():
    class Timeline:
        def is_playing(self):return True
        def is_stopped(self):return False
        def get_current_time(self):return 2.
        def pause(self):calls.append('Pause')
    calls=[]
    async def update():pass
    with pytest.raises(TimeoutError,match='NOT confirmed'):
        asyncio.run(lifecycle.pause_and_confirm_nonplaying(Timeline(),update,max_updates=3))
    assert calls==['Pause']


@pytest.mark.parametrize('stopped',[False,True])
def test_postabort_inspection_reads_same_existing_actor_without_another_static_or_constructor(tmp_path,monkeypatch,stopped):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,fault='binding_never')
    asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    if stopped:f.state.update(playing=False,stopped=True,time=0.)
    def cached_only():
        assert not stopped,'STOPPED inspection must not bind any physics view'
        return f.simulation
    monkeypatch.setattr(f.host.sdk.SimulationManager,'get_physics_sim_view',cached_only)
    before=list(f.events);writes=list(f.writes)
    saved_hash=lifecycle.retained.runtime.task.sha(f.host.result)
    result=asyncio.run(lifecycle.inspect_aborted_held_static(f.host.result,pause_cleanup_authorized=True,_sdk=f.host.sdk))
    assert result['status']=='H6_POSTABORT_PAUSE_AND_EXISTING_STATE_GET_NOT_STATIC_OR_DRIVE_ADMISSION',result
    assert result['new_Play_Stop_reset_static_route_trials']==0 and result['Root_Session_unchanged_during_GET']
    assert result['actual_final_timeline']['stopped']==stopped
    assert 'actual_tensor_snapshot' in result if not stopped else result['STOPPED_no_tensor_binding_or_fabricated_PAUSED_snapshot']
    assert f.events==before+(['Pause_requested','Pause_observed'] if stopped else []) and f.writes==writes
    assert lifecycle.retained.runtime.task.sha(f.host.result)==saved_hash
    before=list(f.events)
    duplicate=asyncio.run(lifecycle.inspect_aborted_held_static(f.host.result,pause_cleanup_authorized=True,_sdk=f.host.sdk))
    assert duplicate['status']=='PRIOR_INSPECTION_RETAINED_NOT_A_CURRENT_STATE_POLL' and f.events==before


def test_postabort_inspection_missing_expected_actor_refuses_Pause_on_another_stage(tmp_path,monkeypatch):
    f=retained_host_fixture(tmp_path,monkeypatch,port_kind='held_static',fresh_root=True,fault='binding_never')
    asyncio.run(lifecycle.run_held_static_once(f.host,f.controller,authorized=True))
    f.stage.GetPrimAtPath(f.protocol['actor']).SetActive(False)
    before=list(f.events)
    with pytest.raises(RuntimeError,match='actor missing'):
        asyncio.run(lifecycle.inspect_aborted_held_static(f.host.result,pause_cleanup_authorized=True,_sdk=f.host.sdk))
    assert f.events==before


def test_postabort_inspection_no_default_write_authority_or_new_trial():
    with pytest.raises(ValueError,match='authority'):
        asyncio.run(lifecycle.inspect_aborted_held_static('/MUST_NOT_READ'))


class SoftwareHeldStaticHost(SoftwareHost):
    """NEW STOPPED-source lifecycle interface, SOFTWARE plant/forces ONLY."""
    observation_kind='NEW_STOPPED_SOURCE_HELD_STATIC'

    async def bind_after_play(self):
        self.events.append('bind_NEW_source_after_ONE_Play')
        if self.fault=='bind':raise RuntimeError('SOFTWARE_NEW_VIEW_BIND_FAILED')
        await self.next_sample()

    def measure(self):
        row=super().measure()
        row.update(z_m=0.,roll_deg=0.,pitch_deg=0.,yaw_deg=math.degrees(self.pose[2]))
        from hospital_static_gate import WHEEL_LABELS
        for i,label in enumerate(WHEEL_LABELS):row[label+'_fz_n']=row[f'wheel_{i}_fz_n']
        if self.fault=='tail_speed' and self.clock>=3.:
            row['vx_physx_m_s']=.005001;row['pose_filtered_speed_m_s']=.005001
        return row

    async def postflight(self):
        if self.fault=='postflight_cancel':raise asyncio.CancelledError('SOFTWARE_POSTFLIGHT_CANCEL')
        return await super().postflight()

    def unsubscribe(self):
        super().unsubscribe()
        if self.fault=='unsubscribe_cancel':raise asyncio.CancelledError('SOFTWARE_UNSUBSCRIBE_CANCEL')


def test_held_static_one_measured_capture_actual_ACK_and_FIXED_one_plus_two_seconds_SOFTWARE_only():
    host=SoftwareHeldStaticHost();control=lifecycle.HeldStaticController([0.,0.,0.],start_time_s=1.)
    passive={k:v[:,host.actuator.passive].copy() for k,v in host.backend.data.items()}
    result=asyncio.run(lifecycle.run_held_static_once(host,control,authorized=True))
    assert result['verdict']=='PASS',result
    assert result['new_static_trials']==1 and result['new_route_trials']==0
    assert host.events.count('Play')==host.events.count('actual_Pause')==1
    assert host.events.index('preflight')<host.events.index('Play')<host.events.index('bind_NEW_source_after_ONE_Play')
    assert host.actuator.capture_count==1 and host.actuator.closed and host.actuator.restoration_report['restored']
    assert all(row['command_speed_m_s']==0. for row in host.saved_rows)
    static=result['static'];assert 1.9<=static['physical_stationary_evaluation']['observation_duration_s']<=2.05
    assert static['fixed_recording_first_s']>=control.held_since+1.
    assert 3.<=static['fixed_recording_last_s']-control.held_since<=3.05
    assert all(np.array_equal(host.backend.data[k][:,host.actuator.passive],v) for k,v in passive.items())
    before=list(host.backend.writes)
    duplicate=asyncio.run(lifecycle.run_held_static_once(host,control,authorized=True))
    assert duplicate['status']=='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION' and host.backend.writes==before


@pytest.mark.parametrize('fault',['preflight','bind','guard','cancel','pause','pause_contact','restoration','postflight','hold_partial','late_nonfinite','tail_speed','postflight_cancel','unsubscribe_cancel'])
def test_held_static_faults_preserve_single_trial_cleanup_original_limits_and_no_restart(fault):
    host=SoftwareHeldStaticHost(fault=fault);control=lifecycle.HeldStaticController([0.,0.,0.],start_time_s=1.)
    result=asyncio.run(lifecycle.run_held_static_once(host,control,authorized=True))
    assert result['verdict']=='HOLD',result
    assert host.events.count('Play')==(0 if fault=='preflight' else 1)
    assert host.events.count('actual_Pause')==host.events.count('durable_save_fixture')==1
    assert host.actuator.capture_count<=1 and result['new_route_trials']==0
    if fault=='tail_speed':
        assert control.complete and not result['static']['checks']['fixed_all_tail_physx_speed_under5mm_s']
        assert result['static']['fixed_recording_first_s']>=control.held_since+1.
    if fault=='pause':assert not host.actuator.closed and 'restoration' not in result
    assert not result['automatic_retry']


def test_held_static_never_uses_route_authority_or_binds_without_own_explicit_consent():
    host=SoftwareHeldStaticHost();control=lifecycle.HeldStaticController([0.,0.,0.],start_time_s=1.)
    with pytest.raises(ValueError,match='held-static authority'):asyncio.run(lifecycle.run_held_static_once(host,control))
    with pytest.raises(ValueError,match='whole-route authority'):asyncio.run(lifecycle.run_lifecycle_once(host,control))
    assert host.events==[] and host.backend.writes==[]


def test_held_static_rejects_existing_PAUSED_route_or_HOLD_port_before_even_claim_with_true_flag():
    host=SoftwareHost();control=lifecycle.HeldStaticController([0.,0.,0.],start_time_s=1.)
    with pytest.raises(ValueError,match='not a PAUSED'):
        asyncio.run(lifecycle.run_held_static_once(host,control,authorized=True))
    assert host.events==[] and host.backend.writes==[]


def test_held_static_actual_drift_HOLD_not_rebased_into_a_new_start_or_admitted_by_matching_hashes():
    control=lifecycle.HeldStaticController([14.192329139709472,16.568600039835417,0.])
    with pytest.raises(RuntimeError,match='30mm'):
        control.step(0.,[14.222394943237305,16.5685977935791,0.],.00663991272779625,(0.,0.),drive_hold_confirmed=False)
    assert control.failed and control.fixed_hold_attempts==0


def test_held_static_waits_for_measured_stop_and_later_ACK_does_not_invent_force_or_restart_window():
    control=lifecycle.HeldStaticController([0.,0.,0.])
    for i in range(6):
        info=control.step(i*.1,[0.,0.,0.],.005001,(0.,0.),drive_hold_confirmed=False)
        assert not info['drive_hold_requested']
    control.step(.6,[0.,0.,0.],.005,(0.,0.),drive_hold_confirmed=False)
    assert control.held_since is None and control.fixed_hold_attempts==1
    control.step(.7,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)
    for i in range(1,30):
        info=control.step(.7+i*.1,[0.,0.,0.],.01 if i<10 else 0.,(0.,0.),drive_hold_confirmed=True)
        assert not info['observation_complete']
    info=control.step(3.7,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)
    assert info['observation_complete'] and control.held_since==.7
    with pytest.raises(RuntimeError,match='terminal'):control.step(3.8,[0.,0.,0.],0.,(0.,0.),drive_hold_confirmed=True)


@pytest.mark.parametrize('fault',['extended','late_start','nonzero','force_alias','missing_force','second_capture','ACK_lost'])
def test_held_static_fixed_window_replay_rejects_shortcuts_not_better_samples_or_renamed_forces(fault):
    host=SoftwareHeldStaticHost();control=lifecycle.HeldStaticController([0.,0.,0.],start_time_s=1.)
    result=asyncio.run(lifecycle.run_held_static_once(host,control,authorized=True));assert result['verdict']=='PASS'
    rows=copy.deepcopy(host.saved_rows);capture=1
    if fault=='extended':rows[-1]['time_s']+=.26
    elif fault=='late_start':rows=[r for r in rows if r['time_s']>=control.held_since+1.3]
    elif fault=='nonzero':rows[0]['command_speed_m_s']=.001
    elif fault=='force_alias':rows[-1]['active_front_fz_n']+=1.
    elif fault=='missing_force':rows[-1].pop('wheel_0_fz_n')
    elif fault=='second_capture':capture=2
    else:rows[-1]['held_ACK_before_command']=False
    if fault=='ACK_lost':
        replay=lifecycle.evaluate_held_static(rows,control.start,host.counts,host.errors,held_since_s=control.held_since,capture_count=capture)
        assert replay['verdict']=='HOLD' and not replay['checks']['actual_hold_ACK_through_fixed_observation']
    else:
        with pytest.raises((ValueError,KeyError)):
            lifecycle.evaluate_held_static(rows,control.start,host.counts,host.errors,held_since_s=control.held_since,capture_count=capture)


def test_fresh_held_definition_preserves_actual_displaced_HOLD_but_keeps_original_fixed_start_and_no_default_authority():
    p=lifecycle.fresh_held_workflow_definition()
    assert p['status']=='PROSPECTIVE_NOT_RUNNABLE' and not p['actual_motion_authorized'] and p['user_approval'] is None
    assert (p['maximum_Stop_calls'],p['maximum_static_trials'],p['maximum_route_trials'])==(1,1,0)
    assert p['required_prior_timeline']['current_time_s']==pytest.approx(7.85)
    assert p['required_prior_tensor']['transform'][0]-p['start_xyyaw'][0]==pytest.approx(.030065803527833168)
    assert p['new_robot_root']!=p['retired_robot_root'] and p['prior_verdict_preserved']=='HOLD'
    assert p['start_xyyaw']==lifecycle.retained.read(lifecycle.retained.GEOMETRY)['start_xyyaw']


def test_fresh_held_missing_answer_never_accesses_SDK_or_exports_runnable_packet():
    with pytest.raises(ValueError,match='Answered explicit'):lifecycle.export_fresh_held_workflow()
    with pytest.raises(ValueError,match='authority required'):lifecycle.fresh_held_preparation_operations(None,None)
    with pytest.raises(ValueError,match='authority required'):asyncio.run(lifecycle.run_fresh_held_workflow('/MUST_NOT_READ'))


def test_fresh_held_historical_archive_cannot_waive_a_changed_original_controller():
    r=lifecycle.retained
    with pytest.raises(ValueError,match='physical/evidence dependency'):
        lifecycle.reviewed_prior_GET_dependencies({str(lifecycle.motion.__file__):'0'*64},r.REVIEWED_ZERO_FOLDER/'paused_closeout_GET_sources.zip')


def _postabort_handoff():
    return lifecycle.retained.ROOT/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_experiments/cohort_03_held_static_r01/held_static/paused_preparation_handoff_r01.json'


def test_new_fresh_definition_uses_actual213s_GET_NOT_old_consumed_zero_clock_or_prior_actor():
    p=lifecycle.fresh_held_workflow_definition(cohort='cohort_04_held_static_r01',prior_handoff=_postabort_handoff())
    assert p['required_prior_timeline']['current_time_s']==pytest.approx(213.033333333395)
    assert p['retired_robot_root']=='/World/patient_transport_h6_cohort_03_held_static_r01'
    assert p['new_robot_root']=='/World/patient_transport_h6_cohort_04_held_static_r01'
    assert not p['actual_motion_authorized'] and p['status']=='PROSPECTIVE_NOT_RUNNABLE'
    assert all(path.startswith(p['retired_robot_root']) for path in p['retiring_actor_fixed_configuration'])
    assert len(p['retiring_actor_fixed_configuration'])==145
    assert p['start_xyyaw'][:2]!=p['required_prior_tensor']['transform'][:2]
    assert p['start_xyyaw']==lifecycle.retained.read(lifecycle.retained.GEOMETRY)['start_xyyaw']
    assert (p['maximum_Stop_calls'],p['maximum_static_trials'],p['maximum_route_trials'])==(1,1,0)
    resolved=p['historical_source_resolution']['resolved_consumed_sources']
    assert any(row['executed_sha256']=='753420ba7700ab0935cb06aa04fb84d7d5083c37c96773dff9b4c407e11701c6' for row in resolved)


@pytest.mark.parametrize('fault',['snapshot','layers','gains','targets','timeline','source_hash'])
def test_postabort_boundary_cannot_invent_a_different_saved_state_or_waive_baseline(monkeypatch,fault):
    r=lifecycle.retained;original_read=r.read;path=_postabort_handoff()
    def changed(file):
        value=original_read(file)
        if Path(file).resolve()==path.resolve():
            value=copy.deepcopy(value)
            if fault=='snapshot':value['SDK_GET']['actual_tensor_snapshot']['transform'][0]+=.001
            elif fault=='layers':value['actual_layer_texts']['Session']+='\n'
            elif fault=='timeline':value['SDK_GET']['timeline']['current_time_s']=0.
            elif fault=='source_hash':value['source_GET_sha256']='0'*64
        elif fault in ('gains','targets') and str(file).endswith('postabort_pause_state_GET_r01.json'):
            value=copy.deepcopy(value)
            if fault=='gains':value['actual_tensor_snapshot']['gains']['kp'][6]=1.
            else:value['actual_tensor_snapshot']['joint_velocity_targets_rad_s'][6]=1.
        return value
    from pathlib import Path
    monkeypatch.setattr(r,'read',changed)
    with pytest.raises(ValueError,match='Exact unchanged|motor baseline'):
        lifecycle.fresh_held_workflow_definition(cohort='cohort_04_held_static_r01',prior_handoff=path)


def test_postabort_new_answer_export_load_remains_single_scope_with_no_SDK(tmp_path,monkeypatch):
    r=lifecycle.retained;monkeypatch.setattr(r,'GATE',tmp_path)
    def forbidden():raise AssertionError('OFFLINE export/load must not bind live SDK')
    monkeypatch.setattr(r.runtime,'live_sdk',forbidden)
    approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_USER_AUTHORITY',answer_ko='새 코호트 정지유지 1회 승인')
    path=lifecycle.export_fresh_held_workflow(cohort='cohort_04_held_static_r01',approval=approval,prior_handoff=_postabort_handoff())
    context=lifecycle.load_fresh_held_workflow(path)
    assert context.protocol['user_approval']==approval and context.protocol['prior_handoff_path']==str(_postabort_handoff())
    assert context.protocol['maximum_route_trials']==0 and context.source_archive.is_file()


def test_shared_fresh_scheduler_uses_exact_new_packet_once_without_reusing_consumed_authority(tmp_path,monkeypatch):
    import builtins
    r=lifecycle.retained;monkeypatch.setattr(r,'GATE',tmp_path)
    approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_USER_AUTHORITY',answer_ko='새 코호트 정지유지 1회 승인')
    path=lifecycle.export_fresh_held_workflow(cohort='cohort_04_held_static_r01',approval=approval,prior_handoff=_postabort_handoff())
    for name in (*r.boundary.ACTIVE_TASKS,'ijcas_h6_latest_postabort_inspection'):monkeypatch.setattr(builtins,name,None,raising=False)
    calls=[]
    async def software_only(packet,*,authorized):
        assert packet==path and authorized is True
        calls.append('ONE_SOFTWARE_DISPATCH_NOT_A_PHYSICAL_TRIAL')
        return dict(status='SOFTWARE_ONLY',route_trials=0)
    monkeypatch.setattr(lifecycle,'run_fresh_held_workflow',software_only)
    async def exercise():
        task=lifecycle.schedule_fresh_held_workflow(path,authorized=True)
        with pytest.raises(RuntimeError,match='Another research task'):
            lifecycle.schedule_fresh_held_workflow(path,authorized=True)
        await task
    asyncio.run(exercise())
    assert len(calls)==1
    with pytest.raises(ValueError,match='authority'):
        lifecycle.schedule_fresh_held_workflow('/MUST_NOT_READ')


def _fresh_preparation_software_fixture(tmp_path,monkeypatch,*,fault=None,prior_handoff=None):
    """Shared Stop/constructor code path on anonymous USD; FAKE SDK only.

    4997 protected row data, retiring tensor/config getters and display are
    explicit SOFTWARE fixtures. No Kit, PhysX, real stage or actual Stop.
    """
    pytest.importorskip('pxr')
    import json,sys
    from types import SimpleNamespace,ModuleType
    from pxr import Usd,UsdGeom,UsdShade,Sdf,UsdRender,Gf
    r=lifecycle.retained;shared=r.shared
    approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_USER_AUTHORITY',answer_ko='검증 후 새 코호트 정지 유지 1회 승인')
    p=lifecycle.fresh_held_workflow_definition(approval=approval,prior_handoff=prior_handoff);p['output_directory']=str(tmp_path)
    p['Stop_update_budget']=3 if fault=='Stop_timeout' else 240
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage,p['retired_robot_root'])
        camera=UsdGeom.Camera.Define(stage,'/OmniverseKit_Top');camera.GetHorizontalApertureAttr().Set(100.)
        if fault in ('camera_XY_before_archive','camera_XY_during_Stop','camera_Z_before_archive'):
            camera.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(1.,2.,100.)))
    UsdShade.Material.Define(stage,shared.preparation.old.constructor.MATERIAL)
    renderer_attribute=None
    if fault in ('renderer_during_Stop','renderer_before_archive'):
        product=UsdRender.Product.Define(stage,'/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0')
        renderer_attribute=product.GetPrim().CreateAttribute('omni:rtx:rt:ecoMode:enabled',Sdf.ValueTypeNames.Bool,custom=False)
        renderer_attribute.Set(True)
    p['required_prior_layer_texts']=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    if fault=='renderer_before_archive':renderer_attribute.Set(False)
    if fault in ('camera_XY_before_archive','camera_Z_before_archive'):
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            camera.GetPrim().GetAttribute('xformOp:transform').Set(Gf.Matrix4d(1).SetTranslate(
                Gf.Vec3d(3.,4.,101. if fault=='camera_Z_before_archive' else 100.)))
    state=copy.deepcopy(p['required_prior_timeline']);events=[];pending=False
    def stop():
        nonlocal pending
        events.append('ONE_Stop_requested');pending=True
        assert events.count('ONE_Stop_requested')==1
        assert (tmp_path/'actual_prior_state_before_Stop.json').exists(),'Archive must be durable BEFORE Stop'
    timeline=SimpleNamespace(is_playing=lambda:state['playing'],is_stopped=lambda:state['stopped'],get_current_time=lambda:state['current_time_s'],stop=stop)
    async def update():
        nonlocal pending
        if pending:
            if fault=='Stop_timeout':return
            state.update(r.boundary.STOPPED);pending=False;events.append('Stop_completed')
            if fault=='renderer_during_Stop':
                with Usd.EditContext(stage,stage.GetRootLayer()):renderer_attribute.Set(False)
            with Usd.EditContext(stage,stage.GetSessionLayer()):
                if fault=='camera_during_Stop':camera.GetHorizontalApertureAttr().Set(101.)
                if fault=='camera_XY_during_Stop':
                    camera.GetPrim().GetAttribute('xformOp:transform').Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(3.,4.,100.)))
                if fault=='foreign_Stop_edit':UsdGeom.Xform.Define(stage,'/World/UNREVIEWED')
    modules={}
    for name in ('omni','omni.usd','omni.timeline','omni.kit','omni.kit.app','isaacsim','isaacsim.core','isaacsim.core.simulation_manager'):
        module=ModuleType(name);module.__path__=[];modules[name]=module;monkeypatch.setitem(sys.modules,name,module)
        if '.' in name:setattr(modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],module)
    modules['omni.usd'].get_context=lambda:SimpleNamespace(get_stage=lambda:stage)
    modules['omni.timeline'].get_timeline_interface=lambda:timeline
    modules['omni.kit.app'].get_app=lambda:SimpleNamespace(next_update_async=update)
    modules['isaacsim.core.simulation_manager'].SimulationManager=SimpleNamespace(get_physics_sim_view=lambda:
        SimpleNamespace(create_articulation_view=lambda _:object()))
    sdk=SimpleNamespace(omni=modules['omni'])
    packet=tmp_path/'SOFTWARE_ONLY_workflow_protocol.json';r.evidence.write_exclusive(packet,p)
    context=r.geometry.common_preparation_context(p,packet)
    # The original preparation has a second JSON I/O boundary distinct from
    # retained.read. Use the same validated relocated software fixture view.
    context.preparation.read_small = r.read
    context.source_archive=tmp_path/'workflow_sources.zip';lifecycle.freeze_workflow_sources(context.source_archive,p['input_sha256'])
    source_rows=r.read(p['environment_rows_file'])['compact_rows']
    context.preparation.compact=SimpleNamespace(**vars(context.preparation.compact))
    def rows(*args):
        actual=copy.deepcopy(source_rows)
        if fault=='protected_row_changed':actual[next(iter(actual))]['row_sha256']='SOFTWARE_CHANGED_NOT_AN_ENVIRONMENT_WAIVER'
        return iter(actual.items())
    context.preparation.compact.stage_rows=rows
    monkeypatch.setattr(shared.preparation.old,'tensor_snapshot',lambda _:copy.deepcopy(p['required_prior_tensor']))
    monkeypatch.setattr(shared,'retiring_actor_configuration',lambda *args:copy.deepcopy(p['retiring_actor_fixed_configuration']))
    context.stationary.critical_environment=lambda *args:dict(SOFTWARE_ONLY_static_physics_getters=True)
    async def display(own,**kwargs):
        assert state==dict(r.boundary.STOPPED) and own is context
        if fault=='display_error':raise RuntimeError('SOFTWARE_FIRST_DISPLAY_FAILURE')
        if fault=='marker_child':UsdGeom.Cube.Define(stage,p['marker']+'/UNDECLARED')
        await fresh_environment_readback(own,stage,timeline,'before_renderer_initialization')
        events.append('FIRST_display_placeholder')
        r.evidence.write_exclusive(tmp_path/'display_initialization/renderer_initialization.json',dict(
            actor=p['new_robot_root'],registered_visual_root_count=13,visible_USD_shape_count=22,
            original_Root_and_Session_text_exactly_restored=True,timeline_before_and_after=dict(r.boundary.STOPPED),
            SOFTWARE_PLACEHOLDER_NOT_VISIBLE_PIXELS_OR_ACTUAL_RENDERER_PROOF=True))
    monkeypatch.setattr(shared,'initialize_fresh_staged_display',display)
    operation=lifecycle.fresh_held_preparation_operations(context,sdk,authorized=True)
    assert operation.__code__ is shared.prepare_fresh_live_trial.__code__
    return SimpleNamespace(context=context,sdk=sdk,operation=operation,stage=stage,state=state,events=events,protocol=p)


def test_actual_postabort_source_handoff_flows_through_same_Stop_constructor_display_code_SOFTWARE_only(tmp_path,monkeypatch):
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch,prior_handoff=_postabort_handoff())
    before=dict(f.state)
    prep=asyncio.run(f.operation(f.context))
    assert before['current_time_s']==pytest.approx(213.033333333395)
    assert f.events.count('ONE_Stop_requested')==1 and f.events.count('FIRST_display_placeholder')==1
    assert prep['retired_robot_root']=='/World/patient_transport_h6_cohort_03_held_static_r01'
    assert prep['Play_or_static_or_drive_executed'] is False
    assert prep['timeline']==dict(lifecycle.retained.boundary.STOPPED)


@pytest.mark.parametrize('fault',[None,'camera_during_Stop','camera_XY_before_archive','camera_XY_during_Stop','camera_Z_before_archive','renderer_during_Stop','renderer_before_archive','protected_row_changed','foreign_Stop_edit','Stop_timeout','display_error','marker_child'])
def test_fresh_held_reuses_original_ONE_Stop_and_real_native_constructor_without_environment_patch_or_any_Play(tmp_path,monkeypatch,fault):
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch,fault=fault)
    before_root=f.stage.GetRootLayer().ExportToString()
    if fault in (None,'camera_during_Stop','camera_XY_before_archive','camera_XY_during_Stop','renderer_during_Stop','renderer_before_archive'):
        prep=asyncio.run(f.operation(f.context))
        assert prep['timeline']==dict(lifecycle.retained.boundary.STOPPED) and not prep['Play_or_static_or_drive_executed']
        assert prep['simulation_mass_kg']==265. and len(prep['source_constructor_readback']['rigid_bodies'])==13
        assert len(prep['source_constructor_readback']['revolute_joint_names'])==12
        assert f.events==['ONE_Stop_requested','Stop_completed','FIRST_display_placeholder']
        review=lifecycle.retained.split_viewport_renderer_defaults(f.stage,
            dict(Root=before_root,Session=f.stage.GetSessionLayer().ExportToString()),
            dict(Root=f.stage.GetRootLayer().ExportToString(),Session=f.stage.GetSessionLayer().ExportToString()))
        assert review['normalized_after']['Root']==before_root
        assert not f.stage.GetPrimAtPath(f.protocol['retired_robot_root']).IsActive()
        layers=dict(Root=f.stage.GetRootLayer().ExportToString(),Session=f.stage.GetSessionLayer().ExportToString())
        bound,packet,path=lifecycle.bind_prepared_held_packet(f.context,prep,layers)
        assert packet['actor']==prep['new_robot_root'] and packet['maximum_route_trials']==0
        assert packet['maximum_Stop_reset_restaging_calls']==0 and path.exists()
        assert bound.preparation.LIVE==f.context.preparation.LIVE
    else:
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):asyncio.run(f.operation(f.context))
        assert f.events.count('ONE_Stop_requested')==(0 if fault in ('protected_row_changed','camera_Z_before_archive') else 1)
    if f.events.count('ONE_Stop_requested'):
        previous=list(f.events)
        with pytest.raises((ValueError,RuntimeError)):asyncio.run(f.operation(f.context))
        assert f.events==previous,'A consumed Stop/preparation must never be replayed'


def test_fresh_preparation_uses_one_bound_handle_when_default_factory_returns_unequal_aliases(tmp_path,monkeypatch):
    """SOFTWARE reproduction of wrapper identity, NOT actual Kit behavior."""
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch)
    captured=f.sdk.omni.timeline.get_timeline_interface();calls=[]
    class Alias:
        def __getattr__(self,name):return getattr(captured,name)
    def factory():calls.append('default_handle_reacquired');return Alias()
    monkeypatch.setattr(f.sdk.omni.timeline,'get_timeline_interface',factory)
    first,second=factory(),factory()
    assert first!=second and lifecycle.retained.evidence.timeline_state(first)==lifecycle.retained.evidence.timeline_state(second)
    calls.clear()
    prep=asyncio.run(f.operation(f.context))
    assert prep['timeline']==dict(lifecycle.retained.boundary.STOPPED)
    assert f.events==['ONE_Stop_requested','Stop_completed','FIRST_display_placeholder']
    assert calls==[],'Preparation/display must share the explicitly captured handle, not reacquire wrapper identities'


@pytest.mark.parametrize('fault',['context','active_stage','actual_clock'])
def test_bound_preparation_handle_cannot_hide_a_changed_context_stage_or_clock(tmp_path,monkeypatch,fault):
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch);context=f.context
    if fault=='context':context=copy.copy(context)
    elif fault=='active_stage':
        from pxr import Usd
        from types import SimpleNamespace
        foreign=Usd.Stage.CreateInMemory()
        monkeypatch.setattr(f.sdk.omni.usd,'get_context',lambda:SimpleNamespace(get_stage=lambda:foreign))
    else:f.state['current_time_s']+=.01
    with pytest.raises(ValueError):asyncio.run(f.operation(context))
    assert f.events==[] and not (tmp_path/'actual_prior_state_before_Stop.json').exists()


def test_bound_preparation_keeps_exact_supplied_timeline_identity_guard_and_shared_display_dependency(tmp_path,monkeypatch):
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch)
    scope=f.operation.__globals__;handles=f.operation._owned_sdk_handles;stage,timeline=handles(f.context)
    class ForeignAlias:
        def __getattr__(self,name):return getattr(timeline,name)
    # Values alone are NOT accepted by the protected environment callback.
    with pytest.raises(ValueError,match='Own fresh preparation stage'):
        asyncio.run(scope['fresh_environment_readback'](f.context,stage,ForeignAlias(),'before_archive'))
    display=scope['initialize_fresh_staged_display']
    closure=dict(zip(display.__code__.co_freevars,(c.cell_contents for c in display.__closure__)))
    assert closure['original_display']._owned_sdk_handles is handles
    assert handles(f.context)[0] is stage and handles(f.context)[1] is timeline
    assert f.events==[]


def test_owned_getter_imports_are_private_and_leave_real_modules_factories_and_builtins_untouched(tmp_path,monkeypatch):
    import builtins,sys
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch)
    real_import=builtins.__import__;real_timeline=f.sdk.omni.timeline.get_timeline_interface
    real_context=f.sdk.omni.usd.get_context
    private=f.operation.__builtins__['__import__']
    view=private('omni.timeline')
    stage,timeline=f.operation._owned_sdk_handles(f.context)
    assert view.timeline.get_timeline_interface() is timeline
    assert private('omni.usd').usd.get_context().get_stage() is stage
    assert private('omni.kit.app').kit is f.sdk.omni.kit
    assert private('math') is math
    assert private('omni.usd',fromlist=('get_context',)) is f.sdk.omni.usd
    assert f.sdk.omni.timeline.get_timeline_interface is real_timeline
    assert f.sdk.omni.usd.get_context is real_context
    assert sys.modules['omni'] is f.sdk.omni
    assert builtins.__import__ is real_import
    assert lifecycle.retained.shared.prepare_fresh_live_trial.__builtins__['__import__'] is not private
    assert f.events==[]


def test_owned_getter_import_refuses_a_real_active_stage_change_without_fabricating_an_owned_stage(tmp_path,monkeypatch):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd
    from types import SimpleNamespace
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch)
    private=f.operation.__builtins__['__import__'];view=private('omni.usd')
    foreign=Usd.Stage.CreateInMemory()
    monkeypatch.setattr(f.sdk.omni.usd,'get_context',lambda:SimpleNamespace(get_stage=lambda:foreign))
    with pytest.raises(ValueError,match='Own active fresh preparation context/stage'):
        view.usd.get_context().get_stage()
    with pytest.raises(ValueError,match='Own active fresh preparation context/stage'):
        private('omni.timeline').timeline.get_timeline_interface()
    assert f.events==[]


@pytest.mark.parametrize('fault',['active_stage','clock'])
def test_bound_handle_detects_actual_stage_or_clock_change_at_prearchive_UI_yield(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch)
    async def changed():
        if fault=='clock':f.state['current_time_s']+=.01
        else:
            from pxr import Usd
            foreign=Usd.Stage.CreateInMemory()
            monkeypatch.setattr(f.sdk.omni.usd,'get_context',lambda:SimpleNamespace(get_stage=lambda:foreign))
    monkeypatch.setattr(f.sdk.omni.kit.app,'get_app',lambda:SimpleNamespace(next_update_async=changed))
    with pytest.raises(RuntimeError,match='stage/clock changed'):asyncio.run(f.operation(f.context))
    assert f.events==[] and not (tmp_path/'actual_prior_state_before_Stop.json').exists()


def test_fresh_workflow_export_load_and_duplicate_dispatch_are_pinned_and_never_bind_SDK(tmp_path,monkeypatch):
    r=lifecycle.retained;monkeypatch.setattr(r,'GATE',tmp_path)
    approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_USER_AUTHORITY',answer_ko='검증 후 새 코호트 정지 유지 1회 승인')
    path=lifecycle.export_fresh_held_workflow(approval=approval)
    ctx=lifecycle.load_fresh_held_workflow(path)
    assert ctx.protocol['maximum_route_trials']==0 and ctx.source_archive.is_file()
    r.evidence.write_exclusive(ctx.output/'workflow_dispatch_reserved.json',dict(SOFTWARE_ONLY_prior_dispatch=True))
    def forbidden():raise AssertionError('A duplicate dispatch must NOT bind SDK or issue another Stop')
    monkeypatch.setattr(r.runtime,'live_sdk',forbidden)
    result=asyncio.run(lifecycle.run_fresh_held_workflow(path,authorized=True))
    assert result['status']=='PRIOR_WORKFLOW_DISPATCH_RETAINED_NO_ADDITIONAL_STOP_OR_ACTION'
    with pytest.raises(ValueError,match='Existing cohort'):lifecycle.export_fresh_held_workflow(approval=approval)


def _saved_cohort04_held_result():
    return lifecycle.retained.ROOT/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_experiments/cohort_04_held_static_r01/held_static/held_static_readback.json'


def test_saved_held_abort_handoff_is_exact_failed_evidence_not_a_new_GET_or_static():
    path=_saved_cohort04_held_result();before=lifecycle.retained.runtime.task.sha(path)
    value=lifecycle.saved_held_abort_boundary(path)
    assert value['status']=='OFFLINE_EXACT_SAVED_HELD_ABORT_HANDOFF_NOT_A_NEW_LIVE_OBSERVATION'
    assert value['SDK_GET']['timeline']==dict(playing=False,stopped=False,current_time_s=1/30)
    assert value['SDK_GET']['actor']=='/World/patient_transport_h6_cohort_04_held_static_r01'
    assert value['new_live_actions']==0 and value['original_HOLD_NOT_promoted_and_no_route_admission']
    assert lifecycle.retained.runtime.task.sha(path)==before


@pytest.mark.parametrize('fault',['Pause','restoration','verdict','actor','target'])
def test_saved_held_abort_handoff_rejects_unconfirmed_or_changed_saved_state(monkeypatch,fault):
    path=_saved_cohort04_held_result();read=lifecycle.retained.read
    raw=copy.deepcopy(read(path))
    if fault=='Pause':raw['checks']['actual_Pause_confirmed']=False
    elif fault=='restoration':raw['checks']['original_gains_targets_restored']=False
    elif fault=='verdict':raw['verdict']='PASS'
    elif fault=='actor':raw['preparation_record']=raw['protocol']
    else:raw['actual_final_readback']['tensor']['joint_velocity_targets_rad_s'][0]=.1
    monkeypatch.setattr(lifecycle.retained,'read',lambda p:raw if Path(p).resolve()==path else read(p))
    with pytest.raises((ValueError,KeyError)):lifecycle.saved_held_abort_boundary(path)


def test_operator_named_scope_command_refuses_missing_authority_before_files_or_SDK(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No files, SDK, export or scheduling before explicit scope flag')
    monkeypatch.setattr(lifecycle,'export_saved_held_abort_handoff',forbidden)
    monkeypatch.setattr(lifecycle,'schedule_fresh_held_workflow',forbidden)
    for flag in (False,None,1,'True'):
        with pytest.raises(ValueError,match='Explicit operator authorization'):
            lifecycle.authorize_and_schedule_held_static_from_saved_result('/MUST_NOT_READ',
                cohort='cohort_05_held_static_r01',approve_one_archive_Stop_new_source_and_held_static=flag)


def test_operator_named_scope_exports_exact_prior_source_and_no_route_packet_then_schedules_once(tmp_path,monkeypatch):
    monkeypatch.setattr(lifecycle.retained,'GATE',tmp_path)
    handoff=_saved_cohort04_held_result().parent/'saved_abort_preparation_handoff_r01.json'
    monkeypatch.setattr(lifecycle,'export_saved_held_abort_handoff',lambda p:handoff)
    scheduled=[]
    def software_schedule(path,*,authorized):
        context=lifecycle.load_fresh_held_workflow(path)
        assert authorized and context.protocol['maximum_route_trials']==0
        assert context.protocol['retired_robot_root']=='/World/patient_transport_h6_cohort_04_held_static_r01'
        assert context.protocol['required_prior_timeline']['current_time_s']==1/30
        assert context.protocol['user_approval']['explicit_flag'] is True
        assert 'question_item_id' not in context.protocol['user_approval']
        scheduled.append(path);return 'SOFTWARE_ONLY_NO_LIVE_SDK'
    monkeypatch.setattr(lifecycle,'schedule_fresh_held_workflow',software_schedule)
    kwargs=dict(cohort='cohort_05_held_static_r01',approve_one_archive_Stop_new_source_and_held_static=True)
    assert lifecycle.authorize_and_schedule_held_static_from_saved_result(_saved_cohort04_held_result(),**kwargs)=='SOFTWARE_ONLY_NO_LIVE_SDK'
    assert lifecycle.authorize_and_schedule_held_static_from_saved_result(_saved_cohort04_held_result(),**kwargs) is None
    assert len(scheduled)==1


def test_operator_approval_record_rejects_integer_flag_or_extra_scope():
    approval=dict(approval_source='USER_EXECUTED_SCRIPT_EDITOR_NAMED_SCOPE_FLAG',
        scope='ONE_ARCHIVE_STOP_NEW_SOURCE_HELD_STATIC_NO_ROUTE_NO_RETRY',explicit_flag=True)
    assert lifecycle.checked_fresh_held_approval(approval)==approval
    for changed in (dict(approval,explicit_flag=1),dict(approval,route=True),dict(approval,explicit_flag=False)):
        with pytest.raises(ValueError):lifecycle.checked_fresh_held_approval(changed)


def test_remaining_user_message_approval_preserves_static_subset_and_does_not_invent_an_answer():
    approval=lifecycle.remaining_validation_approval()
    assert approval['approval_source']=='USER_MESSAGE' and 'question_item_id' not in approval
    assert approval['scope']=='ONE_ARCHIVE_STOP_NEW_SOURCE_HELD_STATIC_NO_ROUTE_NO_RETRY'
    assert lifecycle.checked_fresh_held_approval(approval)==approval
    for value in (dict(approval,scope='RUN_UNTIL_PASS'),dict(approval,user_message='ASSUMED_APPROVAL')):
        with pytest.raises(ValueError):lifecycle.checked_fresh_held_approval(value)


def test_explicit_renderer_fix_revalidation_answer_keeps_ONE_static_NO_route_scope():
    approval=dict(question_item_id='["request_user_input_async","call_2f5870a3f4eb4c28a23490e454078641",0]',
        answer_ko='수정 검증 후 정지유지 재검증 1회 승인')
    assert lifecycle.checked_fresh_held_approval(approval)==approval
    assert lifecycle.held_static_contract()['maximum_static_trials']==1
    assert lifecycle.held_static_contract()['maximum_route_trials']==0


def test_fresh_workflow_rejects_corrupt_source_ZIP_before_any_SDK_access(tmp_path,monkeypatch):
    import zipfile
    r=lifecycle.retained;monkeypatch.setattr(r,'GATE',tmp_path)
    approval=dict(question_item_id='SOFTWARE_FIXTURE_NOT_USER_AUTHORITY',answer_ko='검증 후 새 코호트 정지 유지 1회 승인')
    path=lifecycle.export_fresh_held_workflow(approval=approval);ctx=lifecycle.load_fresh_held_workflow(path)
    with zipfile.ZipFile(ctx.source_archive,'a') as changed:changed.writestr('UNREVIEWED_NOT_A_SOURCE.py','INVALID_TEST_ONLY')
    with pytest.raises(ValueError,match='archive members'):lifecycle.load_fresh_held_workflow(path)


@pytest.mark.parametrize('static_verdict',['PASS','HOLD'])
def test_fresh_workflow_original_preparation_then_only_ONE_static_never_a_route(tmp_path,monkeypatch,static_verdict):
    from types import SimpleNamespace
    f=_fresh_preparation_software_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(lifecycle,'load_fresh_held_workflow',lambda path:f.context)
    monkeypatch.setattr(lifecycle.retained.runtime,'live_sdk',lambda:f.sdk)
    real_host=lifecycle.PreparedHeldStaticSDKHost
    class SoftwareStaticPort:
        VERSION=real_host.VERSION
        def __init__(self,sdk,packet,path,**kwargs):
            assert (tmp_path/'actual_prior_state_before_Stop.json').exists()
            assert (tmp_path/'actual_Stop_boundary.json').exists()
            assert (tmp_path/'display_initialization/renderer_initialization.json').exists()
            assert packet['maximum_route_trials']==0 and packet['maximum_Stop_reset_restaging_calls']==0
            self.result=path.parent/'held_static_readback.json';self.tensor=None
        def timeline_state(self):return copy.deepcopy(f.state)
    async def software_static(port,controller,*,authorized=False):
        assert authorized and isinstance(controller,lifecycle.HeldStaticController)
        f.events.append('SOFTWARE_static_ONLY_NOT_A_PhysX_trial')
        report=dict(verdict=static_verdict,SOFTWARE_FIXTURE_NOT_AN_ACTUAL_STATIC_RESULT=True)
        lifecycle.retained.evidence.write_exclusive(port.result,report);return report
    monkeypatch.setattr(lifecycle,'PreparedHeldStaticSDKHost',SoftwareStaticPort)
    monkeypatch.setattr(lifecycle,'run_held_static_once',software_static)
    monkeypatch.setattr(lifecycle,'replay_held_static_evidence',lambda path:dict(status='SOFTWARE_REPLAY_ONLY_NOT_ACTUAL_DATA',verdict=static_verdict))
    result=asyncio.run(lifecycle.run_fresh_held_workflow(f.context.protocol_path,authorized=True))
    assert result['status']=='ONE_FRESH_COMMON_HELD_STATIC_WORKFLOW_FINISHED' and result['static_verdict']==static_verdict
    assert result['route_trials']==0 and result['held_static_IS_NOT_H6_route_or_final_statistics']
    assert f.events==['ONE_Stop_requested','Stop_completed','FIRST_display_placeholder','SOFTWARE_static_ONLY_NOT_A_PhysX_trial']
    before=list(f.events)
    assert asyncio.run(lifecycle.run_fresh_held_workflow(f.context.protocol_path,authorized=True))['status'].startswith('PRIOR_WORKFLOW_DISPATCH')
    assert f.events==before
