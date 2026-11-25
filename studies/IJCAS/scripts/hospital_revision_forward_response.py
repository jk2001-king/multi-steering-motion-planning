"""One bounded neutral-steering forward response observation, not a route.

Reuses the retained-response SDK host, local full-envelope box, monitored
actuation bridge and original Pause/restoration transaction. No Stop, reset,
placement, source or passive-joint edit occurs. New holding uses the already
tested V4 damping. Fixed10/20mm/s levels are observations, not an identified
global minimum speed or a successful H4 mission.
"""
import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import hospital_revision_hold_damping as hold

runtime = hold.runtime
sequence = hold.base.goal.parent.sequence
OUTPUT = hold.OUTPUT.parent/'H4_retained_neutral_forward_response_r04_task_guard'
PRIOR = hold.OUTPUT.parent/'stopped_reference_end_paired_v5/live/H4_P_dynamics_full_envelope_s04/route/route_readback.json'
AUTHORITY = dict(scope='ONE retained neutral then bounded forward response, no route/reset/retry',
    user_request='P실패 보완과 필요한 검증을 자율 진행; 교수님 피드백과 공정 기준 유지')
VERSION = 'RETAINED_H4_NEUTRAL_FORWARD_RESPONSE_TASK_GUARD_15s_30mm_v4'


class ForwardResponse:
    def __init__(self, start, curvature=0.):
        if curvature != 0. or len(start) != 3 or not all(map(math.isfinite, start)):
            raise ValueError('Exact retained start and zero-curvature diagnostic required')
        self.start=tuple(start);self.mode='coordinated_counterphase';self.curvature=0.
        self.speed=0.;self.stage='capture';self.started=self.previous_time=None
        self.quiet_since=self.aligned_since=self.tracking_started=self.stopped_since=None
        self.previous_pose=None;self.travel=0.;self.complete=False
        self.contract=dict(role=VERSION, fixed_levels_m_s=[.010,.020],level_duration_s=2.,
            maximum_physics_duration_s=15.,maximum_actual_centre_polyline_m=.030,
            acceleration_m_s2=.020,steering_rate_limit_deg_s=8.,actual_route_trials=0,
            no_Stop_reset_or_retry=True,common_v4_hold_damping=hold.HOLD_DAMPING,
            one_local_observation_NOT_global_minimum_speed_identification=True)

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        from hospital_path_execution import ramp
        if (self.complete or type(drive_hold_confirmed) is not bool or measured_speed<0
                or len(pose)!=3 or len(measured_steer)!=2
                or not all(math.isfinite(v) for v in (timestamp,*pose,measured_speed,*measured_steer))):
            raise ValueError('Finite measured diagnostic state required')
        dt=0. if self.previous_time is None else timestamp-self.previous_time
        if not 0<=dt<=.25:raise RuntimeError('Original control-gap guard')
        self.previous_time=timestamp
        if self.started is None:self.started=timestamp
        if timestamp-self.started>=15.:raise TimeoutError('Fixed15s diagnostic cap')
        if self.previous_pose is not None:
            self.travel+=math.hypot(pose[0]-self.previous_pose[0],pose[1]-self.previous_pose[1])
        self.previous_pose=tuple(pose)
        if self.travel>=.030:raise RuntimeError('Fixed30mm measured travel cap')
        requested=0.;held=False
        if self.stage=='capture':
            if measured_speed<=.005:
                if self.quiet_since is None:self.quiet_since=timestamp
            else:self.quiet_since=None
            ready=self.quiet_since is not None and timestamp-self.quiet_since>=.25 and measured_speed<=.003
            held=ready
            if ready and drive_hold_confirmed:self.stage='alignment'
        if self.stage=='alignment':
            held=True
            if not drive_hold_confirmed:raise RuntimeError('Confirmed hold required during neutral alignment')
            if measured_speed>.005:raise RuntimeError('Original5mm/s held-neutral guard')
            if max(map(abs,measured_steer))<=math.radians(2.5):
                if self.aligned_since is None:self.aligned_since=timestamp
                if timestamp-self.aligned_since>=.25:self.stage='release';held=False
            else:self.aligned_since=None
        elif self.stage=='release':
            if not drive_hold_confirmed:self.stage='tracking';self.tracking_started=timestamp
        if self.stage=='tracking':
            if drive_hold_confirmed:raise RuntimeError('Measured release before forward command required')
            elapsed=timestamp-self.tracking_started
            reserve=max(abs(self.speed),measured_speed)**2/.040+.3*max(abs(self.speed),measured_speed)+.004
            if elapsed>=4. or timestamp-self.started>=12. or self.travel+reserve>=.030:
                self.stage='braking'
            else:requested=(.010,.020)[min(1,int(elapsed/2.))]
        self.speed=ramp(self.speed,requested,dt,.020)
        if self.stage=='braking' and abs(self.speed)<=1e-7 and measured_speed<=.005:
            if self.stopped_since is None:self.stopped_since=timestamp
            if timestamp-self.stopped_since>=.25:self.complete=True
        else:self.stopped_since=None
        return dict(phase='path_tracking',terminal_stage=self.stage,phase_time_s=timestamp-self.started,
            speed_m_s=self.speed,mode=self.mode,curvature_1_m=0.,drive_hold_requested=held,
            hold_ready=False,requested_level_m_s=requested,diagnostic_centre_polyline_m=self.travel,
            diagnostic_complete=self.complete)


def current_guard(packet):
    def guard(*args,simulation_budget_s,**kwargs):
        if simulation_budget_s!=600.:
            raise ValueError('Exact historical diagnostic guard construction anchor required')
        return sequence.lifecycle.MissionRuntimeGuard(*args,**kwargs,
            simulation_budget_s=packet['simulation_budget_s'])
    return guard


class ForwardHost(sequence.RetainedResponseSDKHost):
    PROBE_VERSION=VERSION
    APPROVAL_FIELD='explicit_forward_probe_approval'
    EXPECTED_APPROVAL=AUTHORITY
    def __init__(self,sdk,packet,path,**kwargs):
        private=SimpleNamespace(**dict(vars(sequence.lifecycle),MissionRuntimeGuard=current_guard(packet)))
        runtime.bind(sequence.RetainedResponseSDKHost.__init__,lifecycle=private)(self,sdk,packet,path,**kwargs)
    async def preflight(self):
        value=await super().preflight();hold.bind_adapter(self)
        return value


def canonical_source_packet(packet):
    """Exact file-pointer aliases from the SAME saved static record only."""
    static=runtime.read(packet['original_static_record'])
    result=dict(packet)
    for name,path in (('original_preparation',static['preparation_record']),
            ('original_static_packet',static['protocol'])):
        if name in packet and packet[name]!=path:
            raise ValueError('Existing source/static pointer may not be replaced')
        if not Path(path).is_file():raise ValueError('Actual saved source/static file required')
        result[name]=path
    return result


def current_source_context(protocol,path,static_folder,preparation_receipt):
    raw=runtime.read(PRIOR);prior=runtime.read(raw['protocol'])
    static=runtime.read(prior['original_static_record'])
    if (Path(preparation_receipt)!=Path(raw['preparation_record'])
            or Path(path)!=Path(static['protocol']) or protocol!=runtime.read(path)):
        raise ValueError('Exact saved current primary source/static context required')
    geometry=runtime.read(prior['geometry_file'])
    context=hold.base.shared.serialization.fixed_runtime().context_for(
        Path(preparation_receipt).parent,prior['actor'],geometry)
    if context.preparation.LIVE!=Path(preparation_receipt) or context.cohort.ROBOT!=prior['actor']:
        raise ValueError('Read-only current source identity must match; no constructor or restaging')
    return context


def export_protocol():
    raw=runtime.read(PRIOR);old=runtime.read(raw['protocol'])
    if (raw['verdict']!='HOLD' or not hold.base.shared.execution.restored_boundary(raw)
            or raw['error']!='TimeoutError: v2 terminal pulse stalled: under 1 mm movement in 5 simulated seconds'):
        raise ValueError('Exact restored latest stalled-pulse HOLD required')
    if any(not value for key,value in raw['checks'].items() if key not in
            ('whole_route_and_fixed_hold_completed','final_actual_goal_position_under30mm')):
        raise ValueError('Every original environment/contact/cleanup check must be true')
    final=raw['actual_final_readback'];snapshot=final['tensor']
    if math.hypot(*snapshot['velocity'][:2])>.005:raise ValueError('Actual retained measured stop required')
    pose=sequence.lifecycle.RetainedMissionSDKHost._pose(snapshot)
    layers=final['layer_texts'];pins=dict(old['input_sha256'])
    runtime.retained.runtime.task.verify_pins(pins)
    files=(Path(__file__),Path(hold.__file__),PRIOR,Path(raw['protocol']),Path(raw['trace']),
        *sequence.lifecycle.host_source_files())
    pins.update({str(p.resolve()):runtime.sha(p) for p in files})
    import hashlib
    packet=dict(canonical_source_packet(old),version=VERSION,original_route_result=str(PRIOR),original_route_result_sha256=runtime.sha(PRIOR),
        output_directory=str(OUTPUT),explicit_forward_probe_approval=AUTHORITY,
        required_PAUSED_timeline=raw['actual_final_timeline'],required_tensor_snapshot=snapshot,
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],
            unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(sequence.retained.TOP_XY_DISPLAY_PATHS),policy=sequence.retained.TOP_XY_DISPLAY_POLICY),
        start_actual_xyyaw=list(pose),probe_curvature_1_m=0.,probe_contract=ForwardResponse(pose).contract,
        maximum_wall_time_s=90.,maximum_physics_duration_s=15.,maximum_route_trials=0,
        actual_route_authorized=False,actual_response_probe_authorized=True,use_for_final_statistics=False,input_sha256=pins,
        previous_H4_HOLD_NOT_promoted=True)
    path=OUTPUT/'probe_protocol.json'
    if path.exists():
        if runtime.read(path)!=packet:raise ValueError('Frozen forward diagnostic inputs changed')
    else:
        runtime.evidence.write_exclusive(path,packet)
        sequence.lifecycle.freeze_workflow_sources(OUTPUT/'probe_sources.zip',pins)
    return path


async def run(parent,*,authorized=False):
    if authorized is not True:raise ValueError('Current autonomous one-probe authority required')
    import builtins
    previous=await asyncio.shield(parent)
    if (getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not parent
            or Path(previous['last_actual_report'])!=PRIOR):
        raise ValueError('Exact owned latest restored V5 HOLD required')
    path=export_protocol();lc=sequence.lifecycle
    def bridge(controller,actuator,*,guard):
        value=lc.motion.MissionSDKActuationBridge(controller,actuator,guard=guard)
        value.steering_command=tuple(sequence.prime_retained_steering_targets(actuator))
        return value
    async def finish(port,report,rows,**kwargs):
        kwargs['capture_count']=0 if port.adapter is None else port.actuator.capture_count
        return await lc._finish_observation(port,report,rows,**kwargs)
    private=SimpleNamespace(**dict(vars(lc),_finish_observation=finish,
        motion=SimpleNamespace(**dict(vars(lc.motion),MissionSDKActuationBridge=bridge))))
    original_read=sequence.retained.read
    prior_packet=Path(runtime.read(PRIOR)['protocol']).resolve()
    def checked_read(value):
        packet=original_read(value)
        return canonical_source_packet(packet) if Path(value).resolve()==prior_packet else packet
    retained=SimpleNamespace(**dict(vars(sequence.retained),read=checked_read,
        static_binding=current_source_context))
    call=runtime.bind(sequence.run_reverse_response_probe_once,lifecycle=private,
        retained=retained,RetainedReverseResponseSDKHost=ForwardHost,ReverseResponseProbe=ForwardResponse)
    current=asyncio.current_task();builtins.ijcas_hospital_latest_task_experiment=current
    builtins.ijcas_revision_primary_task=current
    report=await call(path,authorized=True)
    return dict(status='ONE_BOUNDED_FORWARD_RESPONSE_OBSERVATION_FINISHED',
        last_actual_report=str(OUTPUT/'probe_readback.json'),report_verdict=report['verdict'],
        actual_route_trials=0,previous_H4_HOLD_unchanged=True)
