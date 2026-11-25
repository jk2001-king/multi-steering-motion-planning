"""Common mission pre-hold / whole route / ONE terminal hold orchestration.

This shared component composes the frozen mission controller and retained
SDK bridge; it is not a task/run/error script clone, a physical hold-force
model or an adopted live protocol. The SDK host accepts an ALREADY OPEN app
and an explicitly admitted packet/context; it never starts an app, creates
an actor or supplies a default motion-authorized definition/entry command.
"""

import asyncio
import copy
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import hospital_mission_execution as motion
import hospital_mission_runtime as retained
from hospital_h4_subfloor_compact_path_launch import normalize_runtime_evidence
from hospital_h2_path_execution import ACCEPTANCE, COMMAND_LIMITS
from hospital_path_execution import relative_pose, wrap
from hospital_static_gate import BAD_CONTACTS
from hospital_stop_analysis import analyze_stop_trace
from hospital_geometry_tools import pose_collision
from hospital_h4_cohort_preparation_v4 import support_area_missing

# Preparation must separate viewport navigation from physical configuration.
# Every matrix change still passes the existing typed XY-only classifier;
# rotation, camera Z, clipping, metadata and all non-camera fields stay exact.
PREPARATION_DISPLAY_PATHS=frozenset((*retained.ZERO_DISPLAY_PATHS,
    retained.boundary.CAMERA+'.xformOp:transform'))


class MissionRuntimeGuard:
    """Original sampled swept-raster/support guards plus measured resources.

    Reverse centre distance is the measured sample polyline, charged to the
    PREVIOUS interval's commanded sign, including uncommanded backwards
    drift. It is NOT a continuous-time arc upper bound or a 3D guarantee.
    Zero commands do not split a reverse episode or erase direction history.
    All terminal correction commands count, not just planned reversals.
    """
    def __init__(self,geometry,occupied,origin,resolution,*,simulation_budget_s,static_observation=False):
        caps={'H6_integrated_transfer_mission':600.,'H3_doorway_immediate_turn':300.,
            'H1_bedside_departure':180.,'H2_bedside_lateral_alignment':180.,'H4_ward_corridor_turn':300.,'H5_obstacle_bypass_return':240.}
        if (geometry['scenario'] not in caps
                or geometry['full_body_rectangle_m']!=[1.630,.965] or geometry['full_body_clearance_m']!=.010
                or type(static_observation) is not bool or type(simulation_budget_s) not in (int,float)
                or simulation_budget_s not in ((held_static_contract()['maximum_physics_duration_s'],) if static_observation else (caps.get(geometry['scenario']),))):
            raise ValueError('Exact H6 full body/10mm margin and explicit common time cap required')
        self.occupied=np.asarray(occupied).copy();self.origin=np.asarray(origin,float).copy()
        if (self.occupied.ndim!=2 or self.occupied.dtype!=np.dtype(bool) or not all(self.occupied.shape)
                or self.origin.shape!=(2,) or not np.isfinite(self.origin).all() or resolution!=.010):
            raise ValueError('Exact finite10mm frozen raster required; no coarsening')
        self.resolution=float(resolution);self.budget=float(simulation_budget_s)
        self.supports=copy.deepcopy(geometry['same_three_authored_support_rectangles'])
        from hospital_mission_geometry import disjoint_support_rectangles
        if len(self.supports)!=3:raise ValueError('All three original authored support rectangles required')
        disjoint_support_rectangles(self.supports)
        self.goal=tuple(geometry['goal_xyyaw']);self.last=None;self.last_info=None
        self.previous_command=0.;self.last_nonzero_sign=0;self.direction_changes=0
        self.reverse_episode=0.;self.reverse_total=0.;self.maximum_episode=0.;self.pose_checks=0
        self.rejected=None;self.failed=False

    def __call__(self,row,info,counts,errors):
        if self.failed:raise RuntimeError('Mission guard terminal; no retry')
        try:self._check(row,info,counts,errors)
        except Exception as failure:
            self.failed=True;self.rejected=dict(error=type(failure).__name__+': '+str(failure),
                observation=rejected_observation_diagnostic(row),proposed_command=rejected_observation_diagnostic(info) if info else None)
            raise

    def _check(self,row,info,counts,errors):
        if errors or any(counts.get(k,0) for k in BAD_CONTACTS):raise RuntimeError('Unexpected contact or callback error')
        keys=('time_s','x_m','y_m','yaw_rad','roll_deg','pitch_deg','vx_physx_m_s','vy_physx_m_s')
        if any(not math.isfinite(float(row[k])) for k in keys):raise ValueError('Finite actual clock/body observation required')
        if max(abs(row['roll_deg']),abs(row['pitch_deg']))>2. or math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])>.080:
            raise RuntimeError('Original body tilt2deg/horizontal speed80mm/s guard')
        stamp=float(row['time_s']);pose=tuple(row[k] for k in ('x_m','y_m','yaw_rad'))
        if not 0.<=stamp<=self.budget:raise TimeoutError('Declared whole-mission physics budget')
        if self.last is not None:
            dt=stamp-self.last[0]
            if not 0.<=dt<=.25:raise RuntimeError('Original actual physics/control-gap guard0.25s')
            if dt==0.:
                if pose!=self.last[1]:raise RuntimeError('Body changed without a later physics observation')
            else:
                last=self.last[1];dx,dy=pose[0]-last[0],pose[1]-last[1];distance=math.hypot(dx,dy)
                if distance>.10:raise RuntimeError('Body observation jumped beyond original0.10m projection bound')
                heading=last[2]+.5*wrap(pose[2]-last[2]);backwards=dx*math.cos(heading)+dy*math.sin(heading)<-1e-12
                if self.previous_command<0. or backwards:
                    self.reverse_total+=distance;self.reverse_episode+=distance
                    self.maximum_episode=max(self.maximum_episode,self.reverse_episode)
                elif distance>1e-12 and self.previous_command>0.:self.reverse_episode=0.
                turn=wrap(pose[2]-last[2])
                steps=max(1,math.ceil(distance/.005),math.ceil(abs(turn)/math.radians(.25)))
                for i in range(1,steps+1):
                    f=i/steps;self._pose((last[0]+f*dx,last[1]+f*dy,last[2]+f*turn))
        else:self._pose(pose)
        if self.reverse_episode>.5+1e-12 or self.reverse_total>1.+1e-12:
            raise RuntimeError('Original reverse centre-polyline resource episode0.5m/total1m exceeded')
        if info is not None:
            command=info['speed_m_s']
            if not math.isfinite(command) or abs(command)>.0500001:raise ValueError('Original50mm/s command limit')
            sign=1 if command>1e-7 else -1 if command<-1e-7 else 0
            if sign and self.last_nonzero_sign and sign!=self.last_nonzero_sign:self.direction_changes+=1
            if self.direction_changes>4:raise RuntimeError('Original maximum4command direction changes exceeded')
            if sign:
                self.last_nonzero_sign=sign
            if info.get('phase') not in ('pre_hold','path_tracking') and math.hypot(pose[0]-self.goal[0],pose[1]-self.goal[1])>.12:
                raise RuntimeError('Original terminal120mm goal-neighbourhood guard')
            self.previous_command=float(command);self.last_info=copy.deepcopy(info)
        self.last=(stamp,pose)

    def _pose(self,pose):
        self.pose_checks+=1
        if pose_collision(self.occupied,pose,self.origin,self.resolution,clearance=.010):
            raise RuntimeError('Original sampled swept full-envelope10mm raster guard')
        if support_area_missing(pose,self.supports)>1e-9:
            raise RuntimeError('Original sampled swept full-body authored support-union guard')

    def summary(self):
        return dict(sampled_reverse_centre_polyline_total_m=self.reverse_total,
            maximum_sampled_reverse_episode_m=self.maximum_episode,command_direction_changes=self.direction_changes,
            swept_pose_checks=self.pose_checks,failed=self.failed,rejected=self.rejected,
            interpolated5mm_quarter_degree_raster_support_NOT_continuous3D_proof=True,
            sampled_reverse_distance_NOT_continuous_arc_upper_bound=True)


def checked_measurement(row):
    """Exact NumPy-scalar representation before control, NEVER state repair."""
    row=normalize_runtime_evidence(row)
    keys=('time_s','x_m','y_m','yaw_rad','vx_physx_m_s','vy_physx_m_s','vz_physx_m_s',
        'pose_filtered_speed_m_s','drive_front_rad','drive_rear_rad',
        'drive_front_velocity_rad_s','drive_rear_velocity_rad_s',*[f'wheel_{i}_fz_n' for i in range(6)])
    if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in keys):
        raise ValueError('Complete finite actual measurement channels required before any control write')
    return row


def viewport_renderer_default_review(stage,path,fields):
    """Separate the renderer's typed ECO default from experiment USD edits.

    Only the documented Boolean on a Kit viewport RenderProduct is eligible.
    No Render subtree, resync, metadata, material or physics exemption. Root
    opinions are valid in Kit and must also pass the separate typed-default
    COPY proof at the next decision. Read/record only; never set ECO.
    """
    import re
    path=str(path)
    if not re.fullmatch(r'/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_[0-9]+\.omni:rtx:rt:ecoMode:enabled',path):
        return None
    from pxr import Sdf,UsdRender
    if set(map(str,fields))!={'default'}:raise ValueError('Viewport ECO metadata/type change is not a display default')
    attribute=stage.GetAttributeAtPath(path)
    if (not attribute or not attribute.GetPrim().IsA(UsdRender.Product)
            or attribute.GetTypeName()!=Sdf.ValueTypeNames.Bool
            or type(attribute.Get()) is not bool or attribute.GetNumTimeSamples()!=0):
        raise ValueError('Exact typed Boolean viewport RenderProduct ECO default required')
    protected=(stage.GetRootLayer(),stage.GetSessionLayer())
    return dict(path=path,fields=['default'],current_value=attribute.Get(),
        classification='VIEWPORT_RENDERER_IDLE_FRAME_POLICY_NOT_PHYSICS_OR_GEOMETRY',
        source_layer_identifiers=[spec.layer.identifier for spec in attribute.GetPropertyStack()],
        authored_in_protected_layer=any(spec.layer in protected for spec in attribute.GetPropertyStack()),
        protected_layer_copy_proof_required=True,no_live_setting_or_layer_edit=True)


def rejected_observation_diagnostic(row):
    """Quarantine invalid channels as tagged ERROR evidence, not trace values."""
    if type(row) is not dict:return dict(invalid_container_type=type(row).__name__,NOT_an_accepted_observation=True)
    result={}
    for key,value in row.items():
        if type(key) is not str:raise TypeError('Diagnostic keys must remain strings')
        try:result[key]=normalize_runtime_evidence(value)
        except (TypeError,ValueError) as failure:
            label=None
            try:
                number=float(value)
                if not math.isfinite(number):label='NaN' if math.isnan(number) else '+Inf' if number>0 else '-Inf'
            except (TypeError,ValueError):pass
            result[key]=dict(kind='INVALID_REJECTED_CHANNEL_NOT_A_NUMERIC_MEASUREMENT',
                python_type=type(value).__name__,nonfinite_label=label,error=str(failure))
    return result


def lifecycle_contract(*,simulation_budget_s):
    contract=motion.mission_execution_contract(simulation_budget_s=simulation_budget_s)
    return dict(version='COMMON_MEASURED_PREHOLD_WHOLE_ROUTE_FIXED_HOLD_v1',
        role='PROSPECTIVE_SOFTWARE_COMPONENT_NOT_ADOPTED_LIVE_PROTOCOL',
        original_motion_contract=contract,simulation_budget_s=float(simulation_budget_s),wall_budget_s=900.,
        common_pre_hold_s=2.,initial_stop_timeout_s=8.,hold_ACK_timeout_s=16.,
        initial_position_m=.010,initial_heading_deg=.5,initial_steering_deg=1.,
        acceptance=dict(ACCEPTANCE),command_limits=dict(COMMAND_LIMITS),
        same_policy_for_all_four_methods=True,physical_state_or_start_reference_reset=False,
        hold_ACK_is_actual_backend_readback_NOT_request_echo=True,
        no_automatic_retry_rehold_or_best_window=True,actual_motion_authorized=False,
        complete_live_SDK_host_and_task_guards_STILL_required=True,use_for_final_statistics=False)


def host_source_files():
    """Executing common helpers, not a hand-picked optional source subset."""
    folder=Path(__file__).resolve().parent
    return tuple(folder/name for name in ('hospital_mission_lifecycle.py','hospital_mission_execution.py',
        'hospital_mission_runtime.py','hospital_h4_path_execution.py','hospital_h2_path_execution.py',
        'hospital_path_execution.py','hospital_path_execution_v2.py','hospital_path_execution_v3.py',
        'hospital_timed_path_execution.py','hospital_geometry_tools.py','hospital_h4_cohort_preparation_v4.py',
        'hospital_h4_cohort_preparation_v3.py','hospital_drive_hold.py','hospital_static_gate.py',
        'hospital_stop_analysis.py','hospital_runtime_evidence_io.py','hospital_h4_subfloor_compact_path_launch.py',
        'hospital_private_loader.py','hospital_h4_cohort_preparation.py','hospital_h4_subfloor_preparation.py',
        'hospital_h4_subfloor_remaining_staging.py','hospital_h4_subfloor_compact_static_gate.py','hospital_compact_usd_fingerprint.py',
        'hospital_mission_geometry.py','hospital_task_boundary.py','hospital_task_runtime.py','hospital_task_experiment.py',
        'hospital_h4_visual_instance_refresh.py','hospital_h4_root_difference_readback.py'))


def held_static_contract():
    """Prospective engineering observation, NOT an adopted experiment packet."""
    from hospital_static_gate import LIMITS
    return dict(version='COMMON_CAPTURED_WHEEL_HELD_STATIC_1_PLUS_2_v1',
        role='PROSPECTIVE_SOFTWARE_COMPONENT_NOT_ADOPTED_LIVE_PROTOCOL',
        measured_stop_limit_m_s=.005,initial_stop_timeout_s=8.,hold_ACK_timeout_s=16.,
        settling_after_actual_ACK_s=1.,fixed_recording_s=2.,maximum_physics_duration_s=27.25,
        wall_budget_s=120.,maximum_static_trials=1,maximum_route_trials=0,
        static_acceptance=dict(LIMITS),maximum_Stop_reset_restaging_calls=0,
        common_existing_captured_wheel_gain_policy=True,original_limits_and_passive_joints_retained=True,
        no_automatic_retry_rehold_or_best_window=True,actual_motion_authorized=False,
        held_static_is_NOT_original_released_static_or_route_or_final_repetition=True)


class HeldStaticController:
    """Measured stop -> one capture/real ACK -> fixed1s settling+2s recording.

    Uses the SAME retained adapter/hold policy as a route, never physical
    state assignment. The controller cannot admit a route, change a start,
    restart a window or turn an old released-static PASS into held evidence.
    """
    def __init__(self,start_xyyaw,*,start_time_s=0.):
        if len(start_xyyaw)!=3 or not all(math.isfinite(v) for v in (*start_xyyaw,start_time_s)):
            raise ValueError('Finite original fixed start/clock required')
        self.start=tuple(start_xyyaw);self.start_time=float(start_time_s);self.previous=None
        self.contract=held_static_contract();self.stage='initial_measured_stop'
        self.mode='coordinated_counterphase';self.curvature=0.;self.requested_at=self.held_since=None
        self.fixed_hold_attempts=0;self.complete=self.failed=False

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        if self.failed or self.complete:raise RuntimeError('Held static terminal; no retry or rehold')
        try:return self._step(timestamp,pose,measured_speed,measured_steer,drive_hold_confirmed)
        except Exception:
            self.failed=True;raise

    def _step(self,timestamp,pose,speed,steer,ack):
        previous=self.start_time if self.previous is None else self.previous
        if (type(ack) is not bool or len(pose)!=3 or len(steer)!=2 or speed<0.
                or not all(math.isfinite(v) for v in (timestamp,*pose,speed,*steer))
                or not 0.<=timestamp-previous<=.25 or self.previous is not None and timestamp<=self.previous):
            raise ValueError('Actual finite monotone held-static body/clock/ACK required')
        if timestamp-self.start_time>self.contract['maximum_physics_duration_s']:
            raise TimeoutError('Whole held-static clock includes measured stop and ACK; no extension')
        self.previous=timestamp;offset=relative_pose(pose,self.start)
        if math.hypot(*offset[:2])>.030:
            raise RuntimeError('Original30mm static drift guard; no start rebase')
        if self.stage=='initial_measured_stop':
            if ack:raise RuntimeError('Own original released gains required before ONE capture')
            if timestamp-self.start_time>8.:raise TimeoutError('Original measured-stop timeout8s')
            if speed<=.005:
                if (math.hypot(*offset[:2])>.010 or abs(offset[2])>math.radians(.5)
                        or max(abs(v) for v in steer)>math.radians(1.)):
                    raise RuntimeError('Original fixed-start10mm/.5deg/1deg lost before capture')
                self.requested_at=timestamp;self.stage='held_static_ACK';self.fixed_hold_attempts=1
        if self.stage=='held_static_ACK':
            if timestamp-self.requested_at>16.:raise TimeoutError('Actual captured-hold ACK timeout16s')
            if ack:self.held_since=timestamp;self.stage='held_static_settling'
        held=0. if self.held_since is None else timestamp-self.held_since
        if self.stage in ('held_static_settling','held_static_recording'):
            if not ack:raise RuntimeError('Actual captured-hold ACK lost; no recapture/window restart')
            if held>=1.:self.stage='held_static_recording'
            self.complete=held>=3.
        return dict(phase=self.stage,terminal_stage=self.stage,mode=self.mode,curvature_1_m=0.,speed_m_s=0.,
            drive_hold_requested=self.requested_at is not None,hold_ready=False,
            held_since_s=self.held_since,held_elapsed_s=held,phase_time_s=held-1. if held>=1. else 0.,
            record_static_sample=self.stage=='held_static_recording',observation_complete=self.complete,
            fixed_hold_attempts=self.fixed_hold_attempts,lifecycle_elapsed_s=timestamp-self.start_time)


def evaluate_held_static(rows,start,counts,errors,*,held_since_s,capture_count):
    """Replay original load/tilt/drift rules AND fixed-tail standstill.

    A successful getter ACK is not proof of a physical brake. Every actual
    force/velocity sample must pass in the single prospectively fixed window.
    """
    from hospital_static_gate import evaluate_static,WHEEL_LABELS
    if capture_count!=1 or not rows or held_since_s is None or not math.isfinite(held_since_s):
        raise ValueError('ONE actual captured hold and complete observation required')
    rows=[checked_measurement(row) for row in rows]
    actual_ACK_rows=[r for r in rows if r['held_ACK_before_command'] is True]
    if not actual_ACK_rows or actual_ACK_rows[0]['time_s']!=held_since_s:
        raise ValueError('Window must start at FIRST actual backend hold ACK, not a chosen later window')
    times=[row['time_s'] for row in rows]
    if any(not 0.<b-a<=.25 for a,b in zip(times,times[1:])):
        raise ValueError('Complete monotone held-static physics samples required')
    if not 3.<=times[-1]-held_since_s<=3.25:raise ValueError('Fixed1+2s observation incomplete or extended')
    tail=[r for r in rows if r['time_s']>=held_since_s+1.]
    if not tail or tail[0]['time_s']-(held_since_s+1.)>.25:
        raise ValueError('Fixed recording window start missing; no later/best window')
    for row in rows:
        if row['command_speed_m_s']!=0.:raise ValueError('Static approval never authorizes a nonzero command')
        for i,label in enumerate(WHEEL_LABELS):
            if row[label+'_fz_n']!=row[f'wheel_{i}_fz_n']:
                raise ValueError('Named and canonical actual wheel-force channels disagree')
    physical=evaluate_static(tail,start,265.,counts,errors);checks=dict(physical['checks'])
    checks.update(fixed_all_tail_physx_speed_under5mm_s=all(math.hypot(r['vx_physx_m_s'],r['vy_physx_m_s'])<=.005 for r in tail),
        fixed_all_tail_filtered_speed_under5mm_s=all(r['pose_filtered_speed_m_s']<=.005 for r in tail),
        final_fixed_start_heading_under_half_deg=abs(wrap(math.radians(tail[-1]['yaw_deg'])-start[2]))<=math.radians(.5),
        actual_hold_ACK_through_fixed_observation=all(r['held_ACK_before_command'] is True for r in rows if r['time_s']>=held_since_s),
        one_capture_no_rehold=capture_count==1)
    return normalize_runtime_evidence(dict(verdict='PASS' if all(checks.values()) else 'HOLD',checks=checks,
        physical_stationary_evaluation=physical,held_since_s=held_since_s,fixed_recording_first_s=tail[0]['time_s'],
        fixed_recording_last_s=tail[-1]['time_s'],fixed_recording_sample_count=len(tail),
        actual_drive_admission=False,use_for_final_statistics=False,
        held_static_NOT_original_released_static_or_complete_environment_proof=True))


class MeasuredPreHold:
    """Zero command -> measured stop -> capture once -> ACK -> fixed2s hold.

    The output requests a hold; ONLY the original retained SDK adapter can
    capture measured active-wheel angles and confirm gains/targets after a
    later physics stamp. This gate never assigns a physical position/speed,
    chooses a favourable observation window or translates the fixed start.
    """
    def __init__(self,start_xyyaw,*,start_time_s=0.):
        if len(start_xyyaw)!=3 or not all(math.isfinite(v) for v in (*start_xyyaw,start_time_s)):
            raise ValueError('Finite ORIGINAL fixed start and actual physics clock required')
        self.start=tuple(start_xyyaw);self.start_time=float(start_time_s);self.previous=None
        self.stage='initial_measured_stop';self.requested_at=self.held_since=self.ready_at=None
        self.failed=False

    def step(self,timestamp,pose,speed,steer,*,drive_hold_confirmed):
        if self.failed:raise RuntimeError('Pre-hold already failed; no automatic retry')
        try:return self._step(timestamp,pose,speed,steer,drive_hold_confirmed)
        except Exception:
            self.failed=True;raise

    def _step(self,timestamp,pose,speed,steer,ack):
        if (type(ack) is not bool or len(pose)!=3 or len(steer)!=2 or speed<0.
                or not all(math.isfinite(v) for v in (timestamp,*pose,speed,*steer))):
            raise ValueError('Actual finite body/steering/speed and backend hold ACK required')
        previous=self.start_time if self.previous is None else self.previous
        if not 0.<=timestamp-previous<=.25 or self.previous is not None and timestamp<=self.previous:
            raise ValueError('Increasing actual physics samples within original0.25s control gap required')
        self.previous=timestamp
        error=relative_pose(pose,self.start)
        if math.hypot(*error[:2])>.010 or abs(error[2])>math.radians(.5) or max(abs(v) for v in steer)>math.radians(1.):
            raise RuntimeError('Original10mm/.5deg/1deg initial boundary lost; no start translation or reset')
        if self.stage=='initial_measured_stop':
            if ack:raise RuntimeError('Original released motor baseline required before first hold request')
            if timestamp-self.start_time>8.:raise TimeoutError('Measured initial stopping timeout8s; no extended settling')
            if speed<=.005:
                self.requested_at=timestamp;self.stage='pre_hold_ACK'
        if self.stage=='pre_hold_ACK':
            if timestamp-self.requested_at>16.:raise TimeoutError('Actual hold ACK timeout16s')
            if ack:self.held_since=timestamp;self.stage='pre_hold'
        if self.stage in ('pre_hold','prepared'):
            if not ack:raise RuntimeError('Actual hold ACK lost; no recapture/window restart')
            if timestamp-self.held_since>=2. and self.ready_at is None:
                if speed>.005:raise RuntimeError('Original5mm/s pre-hold release failed; no recapture/window restart')
                self.ready_at=timestamp;self.stage='prepared'
            if self.ready_at is not None and speed>.005:
                raise RuntimeError('Original5mm/s prepared release lost; no recapture/window restart')
        return dict(phase='pre_hold',terminal_stage=self.stage,speed_m_s=0.,curvature_1_m=0.,
            drive_hold_requested=self.stage!='initial_measured_stop',hold_ready=False,
            preparation_ready=self.ready_at is not None,pre_hold_since_s=self.held_since,
            measured_speed_m_s=speed,physical_hold_ACK=ack)


class MissionLifecycleController:
    """Method-blind wrapper of the ORIGINAL mission tracking/terminal code.

    The outer cap includes pre-hold, every cusp/ACK and the full terminal5s.
    Reaching complete means observation END only, not acceptance or success.
    """
    def __init__(self,plan,geometry,*,simulation_budget_s,start_time_s=0.):
        self.contract=lifecycle_contract(simulation_budget_s=simulation_budget_s)
        self.motion=motion.MissionTrackingController(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.pre_hold=MeasuredPreHold(plan['start_world_xyyaw'],start_time_s=start_time_s)
        self.start_time=float(start_time_s);self.previous=None;self.lifecycle_stage='pre_hold'
        self.hold_started=None;self.fixed_hold_attempts=0;self.complete=False;self.failed=False;self.last_info=None

    @property
    def stage(self):return self.motion.stage if self.lifecycle_stage=='route' else self.lifecycle_stage
    @property
    def curvature(self):return self.motion.curvature if self.lifecycle_stage=='route' else 0.
    @property
    def mode(self):return self.motion.mode
    @property
    def reference(self):return self.motion.reference
    @property
    def goal(self):return self.motion.goal

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        if self.failed or self.complete:raise RuntimeError('Lifecycle terminal; no retry or second hold')
        try:return self._step(timestamp,pose,measured_speed,measured_steer,drive_hold_confirmed)
        except Exception:
            self.failed=True;raise

    def _step(self,timestamp,pose,speed,steer,ack):
        previous=self.start_time if self.previous is None else self.previous
        if (type(ack) is not bool or len(pose)!=3 or len(steer)!=2 or speed<0.
                or not all(math.isfinite(v) for v in (timestamp,*pose,speed,*steer))
                or not 0.<=timestamp-previous<=.25 or self.previous is not None and timestamp<=self.previous):
            raise ValueError('Actual finite monotone mission observations required')
        if timestamp-self.start_time>self.contract['simulation_budget_s']:
            raise TimeoutError('Declared whole-clock cap includes pre-hold/ACKs/terminal observation; no extension')
        self.previous=timestamp
        if self.lifecycle_stage=='pre_hold':
            info=self.pre_hold.step(timestamp,pose,speed,steer,drive_hold_confirmed=ack)
            if not info['preparation_ready']:
                return self._record(dict(info,mode=self.mode,progress_m=0.,action_index=0),timestamp,pose)
            self.lifecycle_stage='route'
        if self.lifecycle_stage=='route':
            info=self.motion.step(timestamp,pose,speed,steer,drive_hold_confirmed=ack)
            if not info['hold_ready']:return self._record(info,timestamp,pose)
            if self.fixed_hold_attempts:raise RuntimeError('No second terminal hold capture')
            error=relative_pose(self.goal,pose)
            if (not ack or speed>.005 or math.hypot(*error[:2])>.030 or abs(error[2])>math.radians(2.)
                    or max(abs(v) for v in steer)>math.radians(2.5)):
                raise RuntimeError('Original stopped terminal entry/neutral/ACK required before ONE hold')
            self.fixed_hold_attempts=1;self.hold_started=timestamp;self.lifecycle_stage='fixed_hold'
        if self.lifecycle_stage=='fixed_hold':
            if not ack:raise RuntimeError('Actual captured-hold ACK lost during ONE fixed5s observation')
            # Preserve the original acceptance definition: first stopped
            # capture, then ONE5s observation and its fixed LAST3s. Do not
            # silently add a stricter all5s standstill rule or restart a window.
            # The host's original80mm/s/tilt/contact/geometry guard stays active.
            self.complete=timestamp-self.hold_started>=5.
            info=dict(self.last_info or {},phase='terminal_hold',terminal_stage='fixed_hold',speed_m_s=0.,
                mode=self.mode,curvature_1_m=0.,drive_hold_requested=True,hold_ready=False,
                progress_m=self.reference.progress,action_index=self.reference.at(self.reference.progress)[0],
                phase_time_s=timestamp-self.hold_started,observation_complete=self.complete)
            return self._record(info,timestamp,pose)
        raise RuntimeError('Unknown lifecycle stage')

    def _record(self,info,timestamp,pose):
        error=relative_pose(self.goal,pose)
        self.last_info=dict(info,goal_position_error_m=math.hypot(*error[:2]),
            goal_heading_error_deg=math.degrees(error[2]),lifecycle_elapsed_s=timestamp-self.start_time,
            fixed_hold_attempts=self.fixed_hold_attempts)
        return dict(self.last_info)


def evaluate_terminal_hold(rows,goal,reference_length_m,contact_counts,callback_errors,*,fixed_hold_attempts):
    """Reuse exact original5s/last3s route acceptance; input provenance external.

    Actual wheel rotations, velocities and forces are REQUIRED. Never invent
    zero wheel data to turn a controller-only/MODEL trace into physical proof.
    """
    if (len(goal)!=3 or not all(math.isfinite(v) for v in (*goal,reference_length_m))
            or reference_length_m<=0. or fixed_hold_attempts!=1):raise ValueError('One complete fixed-goal hold required')
    held=[row for row in rows if row['phase']=='terminal_hold']
    if not held or held[0]['phase_time_s']!=0. or not 5.<=held[-1]['phase_time_s']<=5.25:
        raise ValueError('ONE entire original5s hold from its first capture required')
    times=[float(row['time_s']) for row in held];start=times[0]
    if any(not math.isfinite(v) for v in times) or any(not 0.<b-a<=.25 for a,b in zip(times,times[1:])):
        raise ValueError('Complete increasing actual hold physics timestamps required')
    if any(abs((row['time_s']-start)-row['phase_time_s'])>1e-8 for row in held):
        raise ValueError('Actual physics time and fixed hold origin disagree; no restarted window')
    tail=[row for row in held if row['phase_time_s']>=2.]
    analysis=analyze_stop_trace(tail)
    fields=('pose_filtered_speed_m_s','path_progress_m',*[f'wheel_{i}_fz_n' for i in range(6)])
    if any(not math.isfinite(float(row[k])) for row in tail for k in fields):
        raise ValueError('All actual six-wheel support/velocity/progress channels required')
    errors=[relative_pose(goal,(row['x_m'],row['y_m'],row['yaw_rad'])) for row in tail]
    balance=sum(sum(row[f'wheel_{i}_fz_n'] for i in range(6)) for row in tail)/len(tail)/(265.*9.81)
    final=errors[-1]
    checks=dict(path_execution_and_hold_completed=True,
        at_least95percent_reference_progress=tail[-1]['path_progress_m']>=.95*reference_length_m,
        terminal_position_under30mm=math.hypot(*final[:2])<=.030,terminal_heading_under2deg=abs(final[2])<=math.radians(2.),
        fixed_last3s_complete=analysis['duration_s']>=2.9,
        fixed_tail_position_under30mm=all(math.hypot(*e[:2])<=.030 for e in errors),
        fixed_tail_heading_under2deg=all(abs(e[2])<=math.radians(2.) for e in errors),
        fixed_tail_pose_standstill=analysis['development_standstill_check']=='PASS',
        fixed_tail_physx_speed_under5mm_s=all(math.hypot(r['vx_physx_m_s'],r['vy_physx_m_s'])<=.005 for r in tail),
        fixed_tail_filtered_speed_under5mm_s=all(r['pose_filtered_speed_m_s']<=.005 for r in tail),
        all_six_wheels_mean_load_over1N=all(sum(r[f'wheel_{i}_fz_n'] for r in tail)/len(tail)>1. for i in range(6)),
        vertical_force_weight_balance=.8<=balance<=1.2,
        only_expected_external_support=contact_counts.get('wheel_support',0)>0 and not any(contact_counts.get(k,0) for k in BAD_CONTACTS),
        contact_callback_monitor_healthy=not callback_errors,single_fixed_hold_attempt=fixed_hold_attempts==1)
    return dict(verdict='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        standstill_analysis=analysis,force_weight_ratio=balance,actual_motion_authorized=False,
        pure_acceptance_replay_NOT_input_provenance_or_physical_admission=True)


async def _finish_observation(port,report,rows,*,preflight,actuation,capture_count):
    """Shared independent zero/Pause/restore/evidence path for static+route."""
    if actuation:
        try:port.actuator.emergency_zero();report['checks']['final_zero_target_sent']=True
        except Exception as failure:report['zero_error']=str(failure);report['checks']['final_zero_target_sent']=False
    try:
        report['pause_confirmation']=await port.confirm_pause();report['checks']['actual_Pause_confirmed']=True
    except (Exception,asyncio.CancelledError) as failure:
        report['pause_error']=str(failure);report['checks']['actual_Pause_confirmed']=False
    try:port.unsubscribe()
    except (Exception,asyncio.CancelledError) as failure:
        report['unsubscribe_error']=type(failure).__name__+': '+str(failure);report['checks']['monitors_released']=False
    if preflight and report['checks']['actual_Pause_confirmed']:
        actuator=port.actuator
        if actuator is not None and actuator.dirty:
            try:
                report['restoration']=actuator.restore_original()
                report['checks']['original_gains_targets_restored']=report['restoration']['restored'] is True
            except (Exception,asyncio.CancelledError) as failure:
                report['restore_error']=type(failure).__name__+': '+str(failure);report['checks']['original_gains_targets_restored']=False
        else:report['checks']['no_motor_write_restoration_required']=True
        try:
            report['postflight']=normalize_runtime_evidence(await port.postflight());report['checks'].update(report['postflight']['checks'])
        except (Exception,asyncio.CancelledError) as failure:
            report['postflight_error']=type(failure).__name__+': '+str(failure);report['checks']['complete_postflight_verified']=False
    report.update(contact_counts=dict(port.counts),callback_errors=list(port.errors),sample_count=len(rows),fixed_hold_attempts=capture_count)
    report['checks']['contacts_healthy_through_actual_Pause']=not port.errors and not any(port.counts.get(k,0) for k in BAD_CONTACTS)
    if not report.get('error') and report['checks'] and all(report['checks'].values()):report['verdict']='PASS'
    report=normalize_runtime_evidence(report);rows=normalize_runtime_evidence(rows)
    port.save(report,rows)
    return report


async def pause_and_confirm_nonplaying(timeline,next_update,*,check_context=lambda:None,
        pending_play=False,max_updates=120,wall_budget_s=5.):
    """Pause-only cleanup, including an OWN queued Play not yet observable.

    A final STOPPED state is reported as STOPPED, never as an actual PAUSED
    static trial. No commit(), Play, Stop, clock assignment or physics init.
    The request is followed by three stable app boundaries, so a queued Play
    cannot be mistaken for a completed observation before its frame runs.
    """
    if type(pending_play) is not bool or max_updates<3 or wall_budget_s<=0.:
        raise ValueError('Explicit pending-Play ownership and bounded cleanup required')
    check_context();before=retained.evidence.timeline_state(timeline)
    if not math.isfinite(before['current_time_s']):raise ValueError('Finite timeline required')
    requested=before['playing'] or pending_play
    if requested:timeline.pause()
    began=time.monotonic();last=None;stable=0
    for updates in range(1,max_updates+1):
        remaining=wall_budget_s-(time.monotonic()-began)
        if remaining<=0.:break
        check_context();await asyncio.wait_for(next_update(),timeout=remaining);check_context()
        current=retained.evidence.timeline_state(timeline)
        if not math.isfinite(current['current_time_s']):raise ValueError('Finite cleanup clock required')
        stable=stable+1 if not current['playing'] and current==last else 1 if not current['playing'] else 0
        last=current
        if stable>=3:
            return dict(before=before,after=current,pause_requested_by_this_call=requested,
                own_pending_Play_cancellation_requested=pending_play,updates_to_observe_stable_nonplaying=updates,
                actual_PAUSED_confirmed=not current['stopped'],any_Play_Stop_reset_or_trial=False)
    raise TimeoutError('Stable nonplaying cleanup NOT confirmed; no new Play/Stop/reset')


async def inspect_aborted_held_static(result_path,*,pause_cleanup_authorized=False,_sdk=None):
    """Narrow current-state GET after a consumed first-static failure.

    The ONLY possible live write is one Pause canceling/completing the prior
    trial's cleanup. Uses an existing simulation view only when PAUSED. No
    source construction, view initialization, gains, motion or re-admission.
    Historical packets are read as evidence, never re-pinned or re-executed.
    """
    if pause_cleanup_authorized is not True:raise ValueError('Explicit Pause-only cleanup/GET authority required')
    result_path=Path(result_path).resolve();raw=retained.read(result_path)
    packet_path=Path(raw['protocol']);packet=retained.read(packet_path)
    if (raw['verdict']!='HOLD' or raw['method']!='COMMON_HELD_STATIC_ENGINEERING_NOT_ROUTE'
            or raw['automatic_static_Play_requested'] is not True or raw['new_route_trials']!=0
            or raw['protocol_sha256']!=retained.runtime.task.sha(packet_path)
            or packet['version']!=PreparedHeldStaticSDKHost.VERSION
            or result_path!=packet_path.parent/'held_static_readback.json'):
        raise ValueError('Consumed own held-static HOLD with a recorded Play request required')
    destination=result_path.parent/'postabort_pause_state_GET_r01.json'
    if destination.exists():
        print('[H6 CURRENT] Existing inspection retained; NO command repeated. saved:',destination,flush=True)
        return dict(status='PRIOR_INSPECTION_RETAINED_NOT_A_CURRENT_STATE_POLL',saved=str(destination))
    retained.no_other_task()
    sdk=retained.runtime.live_sdk() if _sdk is None else _sdk
    usd=sdk.omni.usd.get_context();stage=usd.get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    actor=packet['actor'];app=sdk.omni.kit.app.get_app()
    def same_source():
        if stage is None or usd.get_stage()!=stage:raise RuntimeError('Current stage changed/missing; no timeline command')
        prim=stage.GetPrimAtPath(actor)
        if not prim.IsValid() or not prim.IsActive() or not prim.IsLoaded():
            raise RuntimeError('Expected existing H6 actor missing; no timeline command')
    same_source()
    report=dict(status='H6_POSTABORT_PAUSE_AND_EXISTING_STATE_GET_NOT_STATIC_OR_DRIVE_ADMISSION',
        actor=actor,prior_result=str(result_path),prior_result_sha256=retained.runtime.task.sha(result_path),
        executed_current_source_sha256=retained.runtime.task.sha(__file__),
        new_Play_Stop_reset_static_route_trials=0,automatic_retry=False,actual_route_admission=False)
    try:
        report['cleanup']=await pause_and_confirm_nonplaying(timeline,app.next_update_async,
            check_context=same_source,pending_play=timeline.is_stopped())
        before=retained.evidence.timeline_state(timeline)
        layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
        report.update(timeline=before,actual_layer_texts=layers)
        if not before['stopped']:
            simulation=sdk.SimulationManager.get_physics_sim_view()
            if simulation is None:report['existing_physics_view_unavailable_NO_initialization']=True
            else:
                view=simulation.create_articulation_view(actor+'/base_link')
                if not bool(view.check()):raise ValueError('Existing articulation handle invalid; no initialization/recovery')
                first=retained.shared.preparation.old.tensor_snapshot(view)
                second=retained.shared.preparation.old.tensor_snapshot(view)
                same_source()
                if first!=second or before!=retained.evidence.timeline_state(timeline):
                    raise RuntimeError('Existing PAUSED tensor/clock changed during GET')
                if len(second['names'])!=12 or len(second['link_names'])!=13:
                    raise ValueError('All12 joints/all13 links required; no partial state handoff')
                report.update(actual_tensor_snapshot=second,tensor_stable_during_GET=True,
                    horizontal_speed_m_s=math.hypot(*second['velocity'][:2]))
        else:report['STOPPED_no_tensor_binding_or_fabricated_PAUSED_snapshot']=True
        same_source()
        report['Root_Session_unchanged_during_GET']=(layers==dict(
            Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString()))
        if not report['Root_Session_unchanged_during_GET'] or before!=retained.evidence.timeline_state(timeline):
            raise RuntimeError('Stage layers/clock changed during narrow GET')
    except (Exception,asyncio.CancelledError) as failure:
        report.update(status='HOLD_POSTABORT_CLEANUP_OR_GET_INCOMPLETE',error=type(failure).__name__+': '+str(failure))
    report['actual_final_timeline']=retained.evidence.timeline_state(timeline)
    retained.evidence.write_exclusive(destination,normalize_runtime_evidence(report))
    print('[H6 CURRENT]',report['status'],'timeline:',report['actual_final_timeline'],flush=True)
    print('[H6 CURRENT] speed m/s:',report.get('horizontal_speed_m_s'),'error:',report.get('error'),flush=True)
    print('[H6 CURRENT] saved:',destination,'; NO Play/Stop/reset/new trial.',flush=True)
    return report


def schedule_aborted_held_static_inspection(result_path,*,pause_cleanup_authorized=False):
    """Script Editor entry for one required cleanup/readback, NOT an experiment."""
    import builtins
    if pause_cleanup_authorized is not True:raise ValueError('Pause-only cleanup/GET authority required')
    for name in (*retained.boundary.ACTIVE_TASKS,'ijcas_h6_latest_postabort_inspection'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():raise ValueError('Another research task is active; no competing command')
    future=asyncio.ensure_future(inspect_aborted_held_static(result_path,pause_cleanup_authorized=True))
    def completed(task):
        try:task.result()
        except BaseException as failure:print('[H6 CURRENT] ERROR:',type(failure).__name__+': '+str(failure),flush=True)
    future.add_done_callback(completed);builtins.ijcas_h6_latest_postabort_inspection=future
    return future


async def run_held_static_once(port,controller,*,authorized=False):
    """ONE separately approved NEW STOPPED source, never resume the old HOLD.

    Host must verify the prepared STOPPED source/environment BEFORE Play;
    bind_after_play may only bind its new physics/contact views, not create,
    reset or reposition an actor. Binding clock is included in all guards.
    This component exports no live packet, consent or automatic caller.
    """
    if authorized is not True:raise ValueError('Separate ONE held-static authority required before any port call')
    if not isinstance(controller,HeldStaticController):raise ValueError('Explicit fixed1+2s held-static controller required')
    if (getattr(port,'observation_kind',None)!='NEW_STOPPED_SOURCE_HELD_STATIC'
            or not callable(getattr(port,'bind_after_play',None))):
        raise ValueError('Own NEW STOPPED-source static host required, not a PAUSED route/HOLD port')
    if not port.claim_once():return dict(status='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION')
    report=dict(verdict='HOLD',checks={},new_static_trials=0,new_route_trials=0,additional_Stop_calls=0,
        automatic_retry=False,use_for_final_statistics=False,actual_motion_authorized=False,
        component_verdict_NOT_host_provenance_or_independent_physical_verdict=True,contract=copy.deepcopy(controller.contract))
    rows=[];preflight=actuation=False;observed=None;began=watch=time.monotonic()
    try:
        report['preflight']=normalize_runtime_evidence(await port.preflight());preflight=True
        port.subscribe();report['automatic_static_Play_requested']=True;port.play();report['new_static_trials']=1
        await asyncio.wait_for(port.bind_after_play(),timeout=controller.contract['wall_budget_s'])
        bridge=motion.MissionSDKActuationBridge(controller,port.actuator,guard=lambda pose,speed,info:port.guard(observed,info))
        while not controller.complete:
            observed=port.measure();observed=checked_measurement(observed);port.guard(observed,None)
            speed=max(math.hypot(observed['vx_physx_m_s'],observed['vy_physx_m_s']),observed['pose_filtered_speed_m_s'])
            actuation=True
            info=bridge.step(observed['time_s'],(observed['x_m'],observed['y_m'],observed['yaw_rad']),speed)
            row=dict(observed,phase=info['phase'],phase_time_s=info['phase_time_s'],command_speed_m_s=info['speed_m_s'],
                held_ACK_before_command=info['SDK_controller_input_hold_confirmation'])
            if callable(getattr(port,'command_readback',None)):row.update(normalize_runtime_evidence(port.command_readback()))
            rows.append(row)
            if time.monotonic()-watch>=15.:
                print('[H6 HELD STATIC] sim=%.3fs phase=%s samples=%d; ZERO forward command'%(
                    row['time_s'],info['phase'],len(rows)),flush=True);watch=time.monotonic()
            if controller.complete:break
            remaining=controller.contract['wall_budget_s']-(time.monotonic()-began)
            if remaining<=0.:raise TimeoutError('Original120s held-static wall cap')
            await asyncio.wait_for(port.next_sample(),timeout=remaining)
        report['static']=evaluate_held_static(rows,controller.start,port.counts,port.errors,
            held_since_s=controller.held_since,capture_count=port.actuator.capture_count)
        report['checks'].update(report['static']['checks'])
    except (Exception,asyncio.CancelledError) as failure:
        report['error']=type(failure).__name__+': '+str(failure);report['checks']['fixed_held_static_observation_completed']=False
        if observed is not None:report['last_attempted_observation_before_Pause']=rejected_observation_diagnostic(observed)
    finally:
        report=await _finish_observation(port,report,rows,preflight=preflight,actuation=actuation,capture_count=controller.fixed_hold_attempts)
    return report


async def run_lifecycle_once(port,controller,*,authorized=False,sdk_bridge_class=None):
    """One separately admitted host transaction; no SDK binding in this module.

    Host responsibilities cannot be replaced by green software-fixture flags:
    claim_once pins/consent/exclusive reservation; preflight all4997/145/full
    actual tensor and fixed start; guard contacts/support/raster/tilt/resources;
    callbacks/readbacks actual clock/forces/targets; postflight full environment.
    No host implementation/actual route approval is supplied by this component.
    """
    if authorized is not True:raise ValueError('Separate verified whole-route authority required before any port call')
    if not isinstance(controller,MissionLifecycleController):raise ValueError('Original common lifecycle controller required')
    if sdk_bridge_class is None:sdk_bridge_class=motion.MissionSDKActuationBridge
    else:
        from hospital_mission_sequence import HeldNeutralSlewBridge,RetainFinalTargetsBridge
        if sdk_bridge_class not in (HeldNeutralSlewBridge,RetainFinalTargetsBridge):raise ValueError('Only explicitly declared original/target-slew/final-retention SDK bridge supported')
    if not port.claim_once():return dict(status='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION')
    report=dict(verdict='HOLD',checks={},new_route_trials=0,additional_Stop_calls=0,automatic_retry=False,
        use_for_final_statistics=False,actual_motion_authorized=False,
        component_verdict_NOT_host_provenance_or_independent_physical_verdict=True,
        contract=copy.deepcopy(controller.contract))
    rows=[];preflight=False;actuation=False;bridge=None;observed=None;began=None
    try:
        report['preflight']=normalize_runtime_evidence(await port.preflight());preflight=True
        port.subscribe()
        # guard observes the SAME full measurement used for each decision,
        # and runs BEFORE any target/gain write, including pre-Play capture.
        bridge=sdk_bridge_class(controller,port.actuator,
            guard=lambda pose,speed,info:port.guard(observed,info))
        observed=port.measure();observed=checked_measurement(observed);port.guard(observed,None)
        actuation=True
        def step(row):
            speed=max(math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s']),row['pose_filtered_speed_m_s'])
            return bridge.step(row['time_s'],(row['x_m'],row['y_m'],row['yaw_rad']),speed)
        step(observed)  # May request measured-position hold at actual standstill.
        began=watch=time.monotonic();report['new_route_trials']=1;port.play()
        while not controller.complete:
            remaining=controller.contract['wall_budget_s']-(time.monotonic()-began)
            if remaining<=0.:raise TimeoutError(f"Declared{controller.contract['wall_budget_s']:g}s wall cap")
            await asyncio.wait_for(port.next_sample(),timeout=remaining);observed=port.measure();observed=checked_measurement(observed)
            if time.monotonic()-began>controller.contract['wall_budget_s']:raise TimeoutError(f"Declared{controller.contract['wall_budget_s']:g}s wall cap")
            info=step(observed)
            row=dict(observed,phase=info['phase'],phase_time_s=info.get('phase_time_s',0.),
                command_speed_m_s=info['speed_m_s'],path_progress_m=info.get('progress_m',0.))
            if callable(getattr(port,'command_readback',None)):
                row.update(normalize_runtime_evidence(port.command_readback()))
            rows.append(row)
            if time.monotonic()-watch>=15.:
                print('[H6 MISSION] wall=%.1fs sim=%.3fs phase=%s/%s goal=%.2fmm speed=%.2fmm/s'%(
                    time.monotonic()-began,info['lifecycle_elapsed_s'],info['phase'],info['terminal_stage'],
                    1000*info['goal_position_error_m'],1000*math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])),flush=True)
                watch=time.monotonic()
        report['terminal']=evaluate_terminal_hold(rows,controller.goal,controller.reference.arc[-1],
            port.counts,port.errors,fixed_hold_attempts=controller.fixed_hold_attempts)
        report['checks'].update(report['terminal']['checks'])
    except (Exception,asyncio.CancelledError) as failure:
        report['error']=type(failure).__name__+': '+str(failure)
        report['checks']['whole_route_and_fixed_hold_completed']=False
        if observed is not None:report['last_attempted_observation_before_Pause']=rejected_observation_diagnostic(observed)
    finally:
        report=await _finish_observation(port,report,rows,preflight=preflight,actuation=actuation,capture_count=controller.fixed_hold_attempts)
    return report


class RetainedMissionSDKHost(retained.RetainedZeroSettleSDKPort):
    """Actual SDK binding for an explicitly admitted, already PAUSED actor.

    There is intentionally NO packet generator with an invented consent and
    NO automatic caller/SDK import entry here. A new separately reviewed live
    definition must provide an own STATIC PASS with original gains restored,
    exact stopped start/handoff, all frozen files and a matching context.
    Existing H6 zero-command HOLD is therefore not a runnable route handoff.
    """
    VERSION='H6_OWN_STATIC_RETAINED_COMMON_MISSION_SDK_HOST_v1'

    def __init__(self,sdk,protocol,path,*,context,plan,geometry,occupied,origin,resolution,authorized=False):
        if authorized is not True:raise ValueError('Separate explicit whole-route authority required before SDK binding')
        self.path=Path(path).resolve();self.output=self.path.parent
        if retained.read(self.path)!=protocol or protocol.get('version')!=self.VERSION:
            raise ValueError('Exact own frozen route packet required')
        from hospital_mission_sequence import checked_route_lifecycle_contract,TRACKING_FLOOR_APPROVALS,ITERATIVE_FIX_APPROVAL,H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL
        approval=protocol.get('explicit_route_approval',{})
        explicit_approval=(bool(approval.get('question_item_id') and approval.get('answer_ko'))
            or approval in (*TRACKING_FLOOR_APPROVALS,ITERATIVE_FIX_APPROVAL,H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL))
        replanning=protocol.get('stopped_replanning')
        if replanning is not None:
            from hospital_sensor_navigation import checked_h5_replanning_scope
            checked_h5_replanning_scope(protocol)
        if (protocol.get('maximum_Play_calls')!=(2 if replanning is not None else 1) or protocol.get('maximum_Stop_reset_restaging_calls')!=0
                or protocol.get('automatic_retry') is not False or protocol.get('actual_route_authorized') is not True
                or protocol.get('use_for_final_statistics') is not False
                or not explicit_approval):
            raise ValueError('Verified explicit ONE development route scope required; no default consent')
        if protocol.get('common_lifecycle_contract')!=checked_route_lifecycle_contract(protocol):
            raise ValueError('Exact common predeclared controller/runtime contract required')
        retained.runtime.task.verify_pins(protocol['input_sha256'])
        required=(self.path.parent/protocol['plan_filename'],Path(protocol['original_static_record']),
            Path(protocol['original_static_trace']),Path(context.preparation.LIVE),Path(protocol['geometry_file']),Path(protocol['grid_file']))
        for file in (*required,*host_source_files()):
            if protocol['input_sha256'].get(str(file.resolve()))!=retained.runtime.task.sha(file):
                raise ValueError('Complete source/preparation/static/plan/geometry/grid pins required: '+str(file))
        if retained.read(required[0])!=plan or retained.read(required[4])!=geometry:
            raise ValueError('Supplied controller plan/geometry differ from exact frozen files')
        motion.validate_mission_plan(plan,geometry)
        prep=retained.read(context.preparation.LIVE);raw=retained.read(protocol['original_static_record'])
        if (prep['new_robot_root']!=protocol['actor'] or context.cohort.ROBOT!=protocol['actor']
                or Path(context.stationary.RESULT).resolve()!=Path(protocol['original_static_record']).resolve()
                or Path(context.stationary.TRACE).resolve()!=Path(protocol['original_static_trace']).resolve()
                or raw['verdict']!='PASS' or not raw['checks'] or not all(raw['checks'].values())
                or raw['trace_sha256']!=retained.runtime.task.sha(protocol['original_static_trace'])
                or raw['actual_final_timeline']!=protocol['required_PAUSED_timeline']
                or raw['actual_post_static_tensor_snapshot']!=protocol['required_tensor_snapshot']
                or raw['start_xyyaw']!=plan['start_world_xyyaw'] or raw['simulation_mass_kg']!=265.
                or len(raw['runtime_anonymous_source_reference']['fixed_configuration_rows'])!=145):
            raise ValueError('Own source-cohort fixed-start STATIC PASS/current stopped handoff required; no reused HOLD')
        if len(raw['actual_post_static_tensor_snapshot']['names'])!=12:raise ValueError('Own complete12joint static snapshot required')
        self.plan=copy.deepcopy(plan);self.geometry=copy.deepcopy(geometry)
        self.runtime_guard=MissionRuntimeGuard(geometry,occupied,origin,resolution,simulation_budget_s=protocol['simulation_budget_s'])
        if hashlib.sha256(self.runtime_guard.occupied.tobytes()).hexdigest()!=plan['planning_occupancy_digest']:
            raise ValueError('Exact frozen raster digest required; no alternative map')
        with np.load(protocol['grid_file']) as grid:
            if (not np.array_equal(grid['occupied'],occupied) or not np.array_equal(grid['origin_xy_m'],origin)
                    or float(grid['resolution_m'])!=resolution):raise ValueError('Exact original grid coordinates and resolution required')
        super().__init__(sdk,protocol,context=context)
        self.notice_sub=None;self.notice_counts={'allowed_state_paths':0,'reviewed_aperture_paths':0,'unreviewed_paths':0}
        self.claimed=False;self.reservation=self.output/'single_route_reserved.json';self.result=self.output/'route_readback.json'
        self.trace=self.output/'route_trace.csv'

    @property
    def actuator(self):return self.adapter

    def claim_once(self):
        if any(p.exists() for p in (self.reservation,self.result,self.trace)):
            print('[H6 MISSION] Existing dispatch/evidence retained; NO second trial.',flush=True);return False
        retained.no_other_task();retained.runtime.task.verify_pins(self.protocol['input_sha256'])
        retained.evidence.write_exclusive(self.reservation,dict(protocol=str(self.path),protocol_sha256=retained.runtime.task.sha(self.path),
            explicit_route_approval=self.protocol['explicit_route_approval'],maximum_Play_calls=self.protocol['maximum_Play_calls'],
            maximum_Stop_reset_restaging_calls=0,automatic_retry=False,no_Play_yet=True))
        self.claimed=True;return True

    async def preflight(self):
        if not self.claimed:raise RuntimeError('Exact durable ONE route reservation required before preflight')
        prep,raw,replay=self.context.operations.replay_static(self.context)
        if (replay.get('verdict')!='PASS' or prep!=self.prep
                or raw['run_id']!=self.raw['run_id']
                or raw['trace_sha256']!=self.raw['trace_sha256']
                or raw['actual_post_static_tensor_snapshot']!=self.protocol['required_tensor_snapshot']):
            raise ValueError('Own raw static CSV/query/contact/clock/stopped replay required before route')
        result=await super().preflight()
        snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        pose=self._pose(snapshot);start=self.plan['start_world_xyyaw'];offset=relative_pose(pose,start)
        if (math.hypot(*offset[:2])>.010 or abs(offset[2])>math.radians(.5)
                or math.hypot(*snapshot['velocity'][:2])>.005
                or max(abs(snapshot['positions'][i]) for i in self.adapter.steering)>math.radians(1.)):
            raise ValueError('Original fixed-start10mm/.5deg/5mm_s/nominal1deg release failed; no route or reset')
        # Paused cached forces use ONLY the verified configured physics period.
        # Actual callbacks replace this dt before the first dynamic sample.
        hz=self.raw['environment_after_static']['physics_hz']
        if hz!=120:raise ValueError('Original verified120Hz physics setting required')
        self.last_dt=1./hz
        return dict(result,original_fixed_start_and_stopped_release_verified=True,
            own_static_raw_replay=replay,
            cached_prePlay_force_dt_s=self.last_dt,cached_force_IS_NOT_a_new_support_trial=True,
            all_inputs_verified_before_Play=True)

    @staticmethod
    def _pose(snapshot):
        x,y,z,w=snapshot['transform'][3:]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        return (*snapshot['transform'][:2],yaw)

    def subscribe(self):
        super().subscribe()
        Tf=getattr(self.sdk,'Tf',None)
        if Tf is None:
            from pxr import Tf
        names=self.protocol['required_tensor_snapshot']['link_names'];actor=self.protocol['actor']
        state={actor+'/'+name+'.'+field for name in names for field in
            ('xformOp:translate','xformOp:orient','physics:velocity','physics:angularVelocity')}
        state.update(actor+'/joints/'+name+'.state:angular:physics:'+field
            for name in motion.RetainedTensorDriveAdapter.STEERING for field in ('position','velocity'))
        def changed(notice,sender):
            try:
                resynced=list(notice.GetResyncedPaths())
                if resynced:raise RuntimeError('Unexpected stage resync during admitted route: '+str(resynced[:4]))
                for item in notice.GetChangedInfoOnlyPaths():
                    path=str(item)
                    display=viewport_renderer_default_review(self.stage,path,notice.GetChangedFields(item))
                    if display is not None:
                        self.display_audits.append(display)
                        self.notice_counts['reviewed_viewport_renderer_defaults']=self.notice_counts.get('reviewed_viewport_renderer_defaults',0)+1
                        continue
                    if hasattr(notice,'GetChangedFields') and set(map(str,notice.GetChangedFields(item)))-{'default'}:
                        raise RuntimeError('Non-default USD metadata edit during admitted route: '+path)
                    if path in state:self.notice_counts['allowed_state_paths']+=1
                    elif path in retained.ZERO_DISPLAY_PATHS:self.notice_counts['reviewed_aperture_paths']+=1
                    elif (path=='/OmniverseKit_Top.xformOp:transform'
                            and 'reviewed_display_boundary' in self.protocol
                            and path in retained.reviewed_display_paths(self.protocol)):
                        # A camera path match alone is insufficient: prove the
                        # entire Session on COPIES and require the SAME matrix
                        # orientation/Z/affine values before accepting XY only.
                        review=retained.review_zero_display_layers(self.protocol,dict(
                            Root=self.preflight_capture['actual_layer_texts']['Root'],
                            Session=self.stage.GetSessionLayer().ExportToString()))
                        self.display_audits.append(dict(typed_Top_XY_notice=review))
                        self.notice_counts['reviewed_Top_XY_defaults']=self.notice_counts.get('reviewed_Top_XY_defaults',0)+1
                    else:
                        self.notice_counts['unreviewed_paths']+=1
                        raise RuntimeError('Unreviewed non-state USD edit during route: '+path)
            except Exception as failure:self.errors.append(type(failure).__name__+': '+str(failure))
        # Tf keeps a weak Python callback reference; retaining only the
        # registration key silently expires this local callable after return.
        self._notice_callback=changed
        self.notice_sub=Tf.Notice.Register(self.sdk.Usd.Notice.ObjectsChanged,self._notice_callback,self.stage)

    async def next_sample(self):
        previous=self.clock
        while self.clock<=previous:
            self.same_context();self.sim.wake_up(self.stage_id,self.encoded)
            await self.app.next_update_async()
            if not self.timeline.is_playing():raise RuntimeError('Timeline controls changed during whole mission')
            if self.errors:raise RuntimeError('Physics/contact/USD monitor error: '+str(self.errors))
            wall_limit=self.protocol.get('maximum_wall_time_s',900.)
            if time.monotonic()-self.motion_wall>wall_limit:
                raise TimeoutError(f'Declared{wall_limit:g}s whole-route wall cap')

    def measure(self):
        row=super().measure();snapshot=self.last_snapshot
        row['yaw_rad']=math.radians(row['yaw_deg']);row['force_dt_s']=self.last_dt
        row['actual_timeline_s']=self.timeline.get_current_time()
        for i,label in enumerate(self.context.stationary.WHEELS):row[f'wheel_{i}_fz_n']=row[label+'_fz_n']
        for axis,i in zip(('front','rear'),self.adapter.drive):
            row['drive_'+axis+'_rad']=snapshot['positions'][i]
            row['drive_'+axis+'_velocity_rad_s']=snapshot['joint_velocities_rad_s'][i]
        for name,q,qd in zip(snapshot['names'],snapshot['positions'],snapshot['joint_velocities_rad_s']):
            row['joint_'+name+'_rad']=q;row['joint_'+name+'_velocity_rad_s']=qd
        row.update(front_rad=snapshot['positions'][self.adapter.steering[0]],rear_rad=snapshot['positions'][self.adapter.steering[1]])
        return checked_measurement(row)

    def guard(self,row,info):
        self.same_context();self.runtime_guard(row,info,self.counts,self.errors)

    def command_readback(self):
        ack=self.adapter.confirmation(self.clock);motor=self.adapter._motor()
        result=dict(actual_backend_hold_ACK=ack,observations_precede_same_stamp_command=True)
        for axis,index in zip(('front','rear'),self.adapter.drive):
            result['drive_'+axis+'_actual_velocity_target_rad_s']=float(motor['velocities'][index])
            result['drive_'+axis+'_actual_position_target_rad']=float(motor['positions'][index])
            result['drive_'+axis+'_actual_kp']=float(motor['kp'][index])
        for axis,index in zip(('front','rear'),self.adapter.steering):
            result['target_'+axis+'_rad']=float(motor['positions'][index])
        return result

    def unsubscribe(self):
        try:
            if self.notice_sub is not None:self.notice_sub.Revoke()
        finally:
            self.notice_sub=None;self._notice_callback=None;super().unsubscribe()

    async def postflight(self):
        self.post_observation_snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        self.post_observation_boundary_audits=[]
        result=await self.environment(post_observation=True)
        snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        if snapshot!=self.post_observation_snapshot:raise ValueError('Actual PAUSED state changed during full mission postflight')
        motor=self.adapter._motor()
        motors=all(np.array_equal(motor[k],v) for k,v in self.adapter.original.items())
        retained.runtime.task.verify_pins(self.protocol['input_sha256'])
        final=self.measure();last_info=self.runtime_guard.last_info
        if not self.runtime_guard.failed:self.runtime_guard(final,last_info,self.counts,self.errors)
        offset=relative_pose(self._pose(snapshot),self.plan['goal_world_xyyaw'])
        return dict(environment=result,actual_post_mission_tensor_snapshot=snapshot,
            exact_post_observation_boundary_audits=self.post_observation_boundary_audits,
            USD_notice_counts=self.notice_counts,measured_resources=self.runtime_guard.summary(),
            checks=dict(all4997_environment_rows_and145_source_settings_and_physics_unchanged=True,
                original_all12joint_gains_targets_and_limits_restored=motors,all_frozen_inputs_unchanged=True,
                final_actual_goal_position_under30mm=math.hypot(*offset[:2])<=.030,
                final_actual_goal_heading_under2deg=abs(offset[2])<=math.radians(2.),
                final_actual_speed_under5mm_s=math.hypot(*snapshot['velocity'][:2])<=.005,
                actual_Pause_before_common_physics_cap=self.clock<=self.protocol['simulation_budget_s'],
                no_unreviewed_USD_edits_or_callback_errors=not self.errors and not self.notice_counts['unreviewed_paths'],
                all_sampled_geometry_support_and_resource_guards_completed=not self.runtime_guard.failed))

    def save(self,report,rows):
        import csv,os
        if hasattr(self,'play_pending'):
            report=dict(report,first_Play_transition=dict(requested=self.play_requested,observed=self.play_observed,
                still_pending=self.play_pending,binding_updates=self.binding_updates,timeline_events=list(self.timeline_events),
                pause_cleanup=getattr(self,'pause_cleanup_receipt',None)))
        report=normalize_runtime_evidence(dict(report,scenario=self.protocol.get('scenario','H6_integrated_transfer_mission'),method=self.report_method(),
            protocol=str(self.path),protocol_sha256=retained.runtime.task.sha(self.path),actual_final_timeline=self.timeline_state(),
            preparation_record=str(self.context.preparation.LIVE),
            runtime_guard=self.runtime_guard.summary(),notice_counts=self.notice_counts,
            ground_truth_development_NOT_sensor_Nav2_or_final_repetition=True))
        # Preserve actual failure fields/tensor in THIS report, rather than
        # asking the user for another live GET just because postflight failed.
        try:
            before=self.timeline_state()
            layers=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
            report['actual_final_readback']=dict(timeline=before,layer_texts=layers,
                narrow_GET_NOT_a_substitute_for_full_environment_postflight=True)
            if getattr(self,'tensor',None) is not None:
                first=retained.shared.preparation.old.tensor_snapshot(self.tensor)
                second=retained.shared.preparation.old.tensor_snapshot(self.tensor)
                report['actual_final_readback'].update(tensor=second,
                    stable_PAUSED_tensor=not before['playing'] and not before['stopped'] and before==self.timeline_state() and first==second)
            else:report['actual_final_readback']['tensor_NOT_bound_NO_fabricated_physical_snapshot']=True
        except Exception as failure:report['final_readback_error']=type(failure).__name__+': '+str(failure)
        if rows:
            try:
                schema={}
                if any(set(row)!=set(rows[0]) for row in rows):raise ValueError('Raw trace columns changed during one observation')
                for field in rows[0]:
                    types={type(row[field]) for row in rows}
                    if types<={bool,type(None)}:schema[field]='optional_bool' if type(None) in types else 'bool'
                    elif types<={int,float}:schema[field]='finite_number'
                    elif types=={str}:schema[field]='string'
                    else:raise ValueError('Unsupported heterogeneous raw trace channel: '+field)
                with self.trace.open('x',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    stream.flush();os.fsync(stream.fileno())
                report.update(trace=str(self.trace),trace_sha256=retained.runtime.task.sha(self.trace),trace_schema=schema)
            except Exception as failure:
                report['trace_write_error']=type(failure).__name__+': '+str(failure)
                report['verdict']='HOLD';report['checks']['durable_complete_trace']=False
        retained.evidence.write_exclusive(self.result,report)
        print('=== H6 ONE OBSERVATION / '+self.report_method()+' ===',flush=True)
        print('verdict:',report['verdict'],'samples:',len(rows),'timeline:',report['actual_final_timeline'],flush=True)
        print('failed checks:',[k for k,v in report['checks'].items() if not v],'error:',report.get('error'),flush=True)
        print('saved:',self.result,'; NO automatic retry,Stop/reset or next method.',flush=True)

    def report_method(self):return self.plan['method']


class PreparedHeldStaticSDKHost(RetainedMissionSDKHost):
    """Own NEW STOPPED source only; first physics views bind AFTER ONE Play.

    The caller must already have separately archived/stopped/staged/displayed
    the new source and frozen the resulting receipt. This class cannot do any
    of those actions, recreate an old HOLD, generate consent or admit a route.
    No released-static report is fabricated as an initialization baseline.
    """
    VERSION='H6_NEW_STOPPED_SOURCE_HELD_STATIC_SDK_HOST_v1'
    observation_kind='NEW_STOPPED_SOURCE_HELD_STATIC'

    def __init__(self,sdk,protocol,path,*,context,geometry,occupied,origin,resolution,authorized=False):
        if authorized is not True:raise ValueError('Separate explicit held-static authority required before SDK binding')
        self.path=Path(path).resolve();self.output=self.path.parent
        if retained.read(self.path)!=protocol or protocol.get('version')!=self.VERSION:
            raise ValueError('Exact frozen NEW source held-static packet required')
        if (protocol.get('maximum_Play_calls')!=1 or protocol.get('maximum_Stop_reset_restaging_calls')!=0
                or protocol.get('maximum_route_trials')!=0 or protocol.get('automatic_retry') is not False
                or protocol.get('actual_held_static_authorized') is not True or protocol.get('use_for_final_statistics') is not False
                or protocol.get('held_static_contract')!=held_static_contract()
                or protocol.get('required_STOPPED_timeline')!=dict(retained.boundary.STOPPED)):
            raise ValueError('Exact separately approved ONE static/no route/no second Stop scope required')
        checked_fresh_held_approval(protocol.get('explicit_static_approval'))
        retained.runtime.task.verify_pins(protocol['input_sha256'])
        required=tuple(Path(p) for p in (context.preparation.LIVE,protocol['preparation_layers_file'],protocol['display_receipt'],
            protocol['environment_rows_file'],protocol['geometry_file'],protocol['grid_file'],context.preparation.ASSET,retained.H6/'static_protocol.json'))
        for file in (*required,*host_source_files()):
            if protocol['input_sha256'].get(str(file.resolve()))!=retained.runtime.task.sha(file):
                raise ValueError('Complete own preparation/display/environment/query/current-source pins required: '+str(file))
        if Path(protocol['environment_rows_file']).resolve()!=retained.H6/'actual_common_environment_rows.json':
            raise ValueError('Same adopted H6 common environment required; no new environment version')
        prep=retained.read(context.preparation.LIVE);self.prepared_layers=retained.read(protocol['preparation_layers_file'])
        self.link_names=[row['path'].removeprefix(protocol['actor']+'/') for row in prep['source_constructor_readback']['rigid_bodies']]
        self.joint_names=prep['source_constructor_readback']['revolute_joint_names']
        if (prep['new_robot_root']!=protocol['actor'] or context.cohort.ROBOT!=protocol['actor']
                or prep['Play_or_static_or_drive_executed'] is not False or prep['timeline']!=dict(retained.boundary.STOPPED)
                or prep['simulation_mass_kg']!=265. or prep['start_xyyaw']!=geometry['start_xyyaw']
                or prep['live_new_robot_readback']!=prep['source_constructor_readback']
                or prep['source_reference']['source_constructor_readback']!=prep['source_constructor_readback']
                or len(prep['source_reference']['fixed_configuration_rows'])!=145
                or len(self.link_names)!=13 or len(set(self.link_names))!=13 or any('/' in n for n in self.link_names)
                or len(self.joint_names)!=12 or len(set(self.joint_names))!=12
                or set(self.prepared_layers)!={'Root','Session'}
                or any(hashlib.sha256(text.encode()).hexdigest()!=prep[k+'_sha256'] for k,text in self.prepared_layers.items())):
            raise ValueError('Own untouched265kg/13link/12joint source/display/STOPPED preparation required')
        display=retained.read(protocol['display_receipt'])
        if (display['actor']!=protocol['actor'] or display['registered_visual_root_count']!=13 or display['visible_USD_shape_count']!=22
                or display['original_Root_and_Session_text_exactly_restored'] is not True
                or display['timeline_before_and_after']!=dict(retained.boundary.STOPPED)):
            raise ValueError('Own FIRST display registration receipt required; not visible-pixel proof')
        self.query=protocol['query_contract'];old_query=retained.read(protocol.get('static_query_contract_file',retained.H6/'static_protocol.json'))
        for key in ('sampled_cooked_support_queries','supports','support_top_m','additional_sampled_query_limits',
                'approved_disabled_subfloor_paths','original_disabled_collider_paths','additional_disabled_subfloor_probes',
                'auxiliary_probe_known_nonself_colliders','query_known_enabled_colliders','selected_original_cart'):
            if self.query[key]!=old_query[key]:raise ValueError('Original27 rays/four probes/identity limits changed: '+key)
        if (self.query['new_robot_root']!=protocol['actor']
                or set(self.query['source_actor_node_paths'])!=set(prep['source_reference']['fixed_configuration_rows'])):
            raise ValueError('Exact own actor query partition required')
        if retained.read(protocol['geometry_file'])!=geometry:raise ValueError('Exact frozen geometry required')
        with np.load(protocol['grid_file']) as grid:
            if not np.array_equal(grid['occupied'],occupied) or not np.array_equal(grid['origin_xy_m'],origin) or float(grid['resolution_m'])!=resolution:
                raise ValueError('Exact frozen full-envelope10mm raster required')
        self.runtime_guard=MissionRuntimeGuard(geometry,occupied,origin,resolution,
            simulation_budget_s=held_static_contract()['maximum_physics_duration_s'],static_observation=True)
        if hashlib.sha256(self.runtime_guard.occupied.tobytes()).hexdigest()!=geometry['planning_occupancy_digest']:
            raise ValueError('Original frozen raster digest required')
        self._bind_sdk(sdk,protocol,context);self.prep=prep;self.source=prep['source_reference']['fixed_configuration_rows']
        self.tensor=None;self.claimed=self.preflight_completed=self.play_requested=False
        self.play_pending=self.play_observed=self.pause_cleanup_started=False
        self.timeline_sub=None;self.timeline_events=[];self.binding_updates=0
        self.notice_sub=None;self._notice_callback=None;self.pending_state_notice=False
        self.notice_counts=dict(allowed_state_paths=0,reviewed_aperture_paths=0,unreviewed_paths=0)
        self.state_types=None;self.reservation=self.output/'single_held_static_reserved.json'
        self.result=self.output/'held_static_readback.json';self.trace=self.output/'held_static_trace.csv'

    def report_method(self):return 'COMMON_HELD_STATIC_ENGINEERING_NOT_ROUTE'

    def claim_once(self):
        if any(p.exists() for p in (self.reservation,self.result,self.trace)):return False
        retained.no_other_task();retained.runtime.task.verify_pins(self.protocol['input_sha256'])
        retained.evidence.write_exclusive(self.reservation,dict(protocol=str(self.path),protocol_sha256=retained.runtime.task.sha(self.path),
            explicit_static_approval=self.protocol['explicit_static_approval'],maximum_Play_calls=1,
            maximum_Stop_reset_restaging_calls=0,maximum_route_trials=0,automatic_retry=False,no_Play_yet=True))
        self.claimed=True;return True

    def same_source(self):
        if self.usd.get_stage()!=self.stage:
            raise RuntimeError('Active stage changed; no reset/recovery')
        prim=self.stage.GetPrimAtPath(self.protocol['actor'])
        if not prim.IsValid() or not prim.IsActive() or not prim.IsLoaded():raise RuntimeError('Own active loaded NEW source required')

    def same_context(self):
        self.same_source()
        if self.play_requested and self.timeline.is_playing():
            self.play_observed=True;self.play_pending=False
        if self.play_requested and self.timeline.is_stopped():
            # Timeline mutations are frame-deferred. This narrow exception
            # covers ONLY our first pending Play at the exact zero-time source;
            # it does not admit a Stop after playback or reset/recreate a view.
            if not (self.play_pending and not self.play_observed and self.clock==0.
                    and self.timeline.get_current_time()==0.):
                raise RuntimeError('Unexpected Stop after first Play request; no reset/recovery')

    def check_prepared_layers(self):
        self.same_context();texts=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
        if texts==getattr(self,'_last_prepared_boundary_texts',None):return texts
        renderer=retained.split_viewport_renderer_defaults(self.stage,self.prepared_layers,texts)
        normalized=renderer['normalized_after']
        if normalized['Root']!=self.prepared_layers['Root']:raise ValueError('Prepared Root changed before first Play')
        if texts!=getattr(self,'_last_prepared_boundary_texts',None):
            review=retained.canonical_display_only_review(self.prepared_layers['Session'],normalized['Session'])
            deltas=review['exact_Top_default_differences']
            if any(row['path'] not in PREPARATION_DISPLAY_PATHS for row in deltas):raise ValueError('Non-display prepared Session edit')
            audit=dict(reviewed_aperture_defaults=deltas,lossless_canonical_copy_review=review,
                exact_renderer_default_differences=renderer['exact_renderer_default_differences'])
            if not self.display_audits or audit!=self.display_audits[-1]:self.display_audits.append(audit)
            self._last_prepared_boundary_texts=texts
        return texts

    async def environment(self,*,post_observation=False):
        self.same_context();state=self.timeline_state();started=watch=time.monotonic();rows={}
        if state['playing'] or state['stopped'] is post_observation:raise ValueError('Own STOPPED preflight/PAUSED postflight required')
        def stable():
            self.same_context()
            if self.timeline_state()!=state:raise RuntimeError('Nonplaying clock changed during held-static environment GET')
            if post_observation:self.check_post_observation_layers()
            else:self.check_prepared_layers()
            if time.monotonic()-started>90.:raise TimeoutError('Original bounded90s environment GET')
        p=self.context.preparation
        for path,row in p.compact.stage_rows(self.stage,p.SELECTED,p.old.original):
            if path in rows:raise ValueError('Duplicate protected row')
            rows[path]=row
            if len(rows)%16==0:stable();await self.app.next_update_async();stable()
            if time.monotonic()-watch>=15.:
                print('[H6 HELD STATIC] protected rows=%d/4997; NO motion'%len(rows),flush=True);watch=time.monotonic()
        stable();expected=copy.deepcopy(retained.read(self.protocol['environment_rows_file'])['compact_rows'])
        for path in p.SELECTED:
            if expected[path]['collision_enabled'] is not False:raise ValueError('Exactly four originally disabled subfloor identities required')
            expected[path]['collision_enabled']=True
        digest=p.compact.compare_rows(expected,rows,p.SELECTED,after_flags=True)
        if len(rows)!=4997 or digest!=self.prep['protected_environment_sha256']:raise ValueError('Same4997 original H6 rows required')
        if self.context.stationary.actor_configuration(self.stage)!=self.source:raise ValueError('All145 source settings changed')
        if self.context.stationary.critical_environment(self.stage,self.prep)!=self.prep['critical_environment']:
            raise ValueError('Original physics/material/support/mass settings changed')
        return dict(row_count=4997,sha256=digest,all145_source_settings_and_critical_environment_verified=True)

    async def preflight(self):
        if not self.claimed or self.timeline_state()!=dict(retained.boundary.STOPPED):
            raise ValueError('Own durable claim and untouched STOPPED zero-time source required before Play')
        self.check_prepared_layers()
        source=retained.shared.fresh_source_for_live_actor(self.context,self.stage)
        if source!=self.prep['source_reference']:raise ValueError('Exact same-SDK fresh source reference changed')
        constructor=retained.shared.preparation.old.constructor_for(self.protocol['actor']).__globals__['validate_new_robot']
        if constructor(self.stage,self.sdk.Usd.Stage.Open(str(self.context.preparation.ASSET)),self.prep['start_xyyaw'])!=self.prep['source_constructor_readback']:
            raise ValueError('Original source pose/seed/12joints/13links changed')
        self.state_types=retained.prepared_source_state_types(self.stage,self.protocol['actor'],self.link_names,self.joint_names)
        checked=await self.environment();self.preflight_layers=self.check_prepared_layers()
        if self.timeline_state()!=dict(retained.boundary.STOPPED):raise ValueError('Own STOPPED boundary lost before Play')
        self.sim=self.sdk.omni.physx.get_physx_simulation_interface()
        self.stage_id=self.sdk.UsdUtils.StageCache.Get().GetId(self.stage).ToLongInt()
        self.encoded=self.sdk.PhysicsSchemaTools.sdfPathToInt(self.protocol['actor']+'/base_link')
        self.preflight_completed=True
        return dict(environment=checked,original_source_seed_and_constructor_verified=True,
            prepared_STOPPED_source_verified=True,no_physics_view_initialized_before_Play=True,prePlay_source_state_types=self.state_types)

    def subscribe(self):
        if not self.preflight_completed:raise ValueError('Complete prepared-source preflight required before monitors')
        retained.RetainedZeroSettleSDKPort.subscribe(self)
        Tf=getattr(self.sdk,'Tf',None)
        if Tf is None:from pxr import Tf
        from pxr import Sdf
        ancestors=set()
        for path in self.state_types:
            p=Sdf.Path(path).GetPrimPath()
            while str(p).startswith(self.protocol['actor']):ancestors.add(str(p));p=p.GetParentPath()
        def changed(notice,sender):
            try:
                for path in notice.GetResyncedPaths():
                    if str(path) not in ancestors and str(path) not in self.state_types:
                        raise RuntimeError('Unreviewed first-physics resync: '+str(path))
                    self.pending_state_notice=True
                for path in notice.GetChangedInfoOnlyPaths():
                    display=viewport_renderer_default_review(self.stage,path,notice.GetChangedFields(path))
                    if display is not None:
                        self.display_audits.append(display);self.pending_state_notice=True
                        self.notice_counts['reviewed_viewport_renderer_defaults']=self.notice_counts.get('reviewed_viewport_renderer_defaults',0)+1
                        continue
                    if str(path) in self.state_types:
                        self.notice_counts['allowed_state_paths']+=1
                        if set(map(str,notice.GetChangedFields(path)))-{'default'}:self.pending_state_notice=True
                    elif str(path) in ancestors:
                        # USD can notify an EXISTING composed ancestor when
                        # its first local state Over is authored. This is NOT
                        # a waiver: full canonical typed+SDK proof is mandatory
                        # at the next decision BEFORE any actuator write.
                        self.pending_state_notice=True
                    elif str(path) in retained.ZERO_DISPLAY_PATHS:
                        if set(map(str,notice.GetChangedFields(path)))-{'default'}:raise ValueError('Aperture metadata changed')
                        self.notice_counts['reviewed_aperture_paths']+=1
                    else:raise RuntimeError('Unreviewed USD edit during held static: '+str(path))
            except Exception as failure:
                self.notice_counts['unreviewed_paths']+=1;self.errors.append(type(failure).__name__+': '+str(failure))
        self._notice_callback=changed;self.notice_sub=Tf.Notice.Register(self.sdk.Usd.Notice.ObjectsChanged,changed,self.stage)
        kinds=self.sdk.omni.timeline.TimelineEventType
        def timeline_changed(event):
            if not self.play_requested:return
            kind=int(event.type)
            if kind not in (int(kinds.PLAY.value),int(kinds.PAUSE.value),int(kinds.STOP.value)):return
            self.timeline_events.append(kind)
            if kind==int(kinds.PLAY.value):
                if self.timeline_events.count(kind)>1:self.errors.append('Unexpected additional Play event')
                self.play_observed=True;self.play_pending=False
            elif not self.pause_cleanup_started:
                self.play_pending=False;self.errors.append('Unexpected Pause/Stop during first static')
        self._timeline_callback=timeline_changed
        self.timeline_sub=self.timeline.get_timeline_event_stream().create_subscription_to_pop(
            timeline_changed,name='H6 first-static owned Play transition')

    def unsubscribe(self):
        self.timeline_sub=None;self._timeline_callback=None
        super().unsubscribe()

    def play(self):
        if not self.preflight_completed or self.play_requested or self.timeline_state()!=dict(retained.boundary.STOPPED):
            raise ValueError('ONE first Play only after untouched STOPPED preflight; no resume/retry')
        self.check_prepared_layers();self.play_requested=True;self.play_pending=True
        self.motion_wall=time.monotonic();self.timeline.play()

    async def bind_after_play(self):
        import importlib
        if not self.play_requested:raise ValueError('Own authorized ONE Play required before new tensor binding')
        while True:
            self.same_context()
            remaining=15.-(time.monotonic()-self.motion_wall)
            if self.clock>.25 or remaining<=0. or self.binding_updates>=240:
                raise TimeoutError('Bounded initial physics binding0.25s/15s; no recovery')
            if self.errors:raise RuntimeError('First physics initialization monitor error: '+str(self.errors))
            await asyncio.wait_for(self.app.next_update_async(),timeout=remaining)
            self.binding_updates+=1;self.same_context()
            if self.clock>.25 or time.monotonic()-self.motion_wall>15.:
                raise TimeoutError('Bounded initial physics binding0.25s/15s; no recovery')
            if self.errors:raise RuntimeError('First physics initialization monitor error: '+str(self.errors))
            if not self.timeline.is_playing():
                if self.play_pending:continue
                raise RuntimeError('First physics initialization changed timeline state')
            simulation=self.sdk.SimulationManager.get_physics_sim_view()
            if simulation is not None and self.clock>0.:break
        self.simulation=simulation;self.tensor=simulation.create_articulation_view(self.protocol['actor']+'/base_link')
        snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        if set(snapshot['link_names'])!=set(self.link_names) or set(snapshot['names'])!=set(self.joint_names):
            raise ValueError('Own source13link/12joint tensor correspondence required')
        backend=self.sdk.SimulationManager.get_backend()
        if backend not in ('numpy','torch','warp'):raise ValueError('Verified native SDK encoder required')
        utils=importlib.import_module('isaacsim.core.utils.'+backend)
        self.adapter=motion.RetainedTensorDriveAdapter(self.tensor,self.prep['expected_nominal_motor_baseline'],
            encode=motion.retained_tensor_encoder(utils,simulation.device),check_context=self.same_context,
            paused_check=lambda:not self.timeline.is_playing() and not self.timeline.is_stopped(),physics_stamp=self.clock,authorized=True)
        for name in self.context.stationary.WHEELS.values():
            path=self.protocol['actor']+'/'+name;view=simulation.create_rigid_contact_view(path)
            if view.sensor_count!=1 or list(view.sensor_paths)!=[path]:raise ValueError('Own single original wheel contact view required')
            self.views.append(view)
        self.initial_binding_snapshot=snapshot;self.pending_state_notice=True

    async def confirm_pause(self):
        if not self.play_requested:return await super().confirm_pause()
        self.pause_cleanup_started=True
        receipt=await pause_and_confirm_nonplaying(self.timeline,self.app.next_update_async,
            check_context=self.same_source,pending_play=self.play_pending)
        self.play_pending=False;self.pause_cleanup_receipt=receipt
        if not receipt['actual_PAUSED_confirmed']:
            raise RuntimeError('Owned Play canceled at STOPPED; no actual PAUSED observation or static PASS')
        return receipt

    def guard(self,row,info):
        self.same_context();self.runtime_guard(row,None,self.counts,self.errors)
        metrics=retained.zero_sample_safety_metrics(row,self.prep['start_xyyaw'])
        if metrics['exceeded']:raise RuntimeError('Original held-static safety guard: '+', '.join(metrics['exceeded']))
        if info is not None and info['speed_m_s']!=0.:raise ValueError('Static authority excludes all nonzero commands')
        if self.pending_state_notice:
            texts=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
            self.last_first_state_proof=retained.prepared_source_writeback_review(self.stage,self.preflight_layers,texts,
                self.last_snapshot,self.protocol['actor'],self.state_types)
            self.pending_state_notice=False

    def check_post_observation_layers(self):
        snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        if snapshot!=self.post_observation_snapshot:raise ValueError('Actual PAUSED held-static tensor changed')
        texts=dict(Root=self.stage.GetRootLayer().ExportToString(),Session=self.stage.GetSessionLayer().ExportToString())
        if texts==getattr(self,'_last_post_observation_layer_texts',None):return
        proof=retained.prepared_source_writeback_review(self.stage,self.preflight_layers,texts,snapshot,self.protocol['actor'],self.state_types)
        self.post_observation_boundary_audits.append(proof);self._last_post_observation_layer_texts=texts

    async def postflight(self):
        if self.tensor is None:raise ValueError('New tensor initialization incomplete; no fabricated static postflight')
        self.post_observation_snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor);self.post_observation_boundary_audits=[]
        checked=await self.environment(post_observation=True)
        import importlib
        carb=getattr(self.sdk,'carb',None) or importlib.import_module('carb')
        collect,probe=retained.runtime.query_functions(dict(carb=carb,omni=self.sdk.omni,c=self.context.cohort,protocol=self.query))
        support,absence=collect(),probe();positive=retained.support_evaluation(self.query,support)
        negative=retained.subfloor_evaluation(self.query,support,absence)
        snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        if snapshot!=self.post_observation_snapshot:raise ValueError('PAUSED state changed during cached static queries')
        self.check_post_observation_layers()
        if not self.runtime_guard.failed:self.runtime_guard(self.measure(),None,self.counts,self.errors)
        motor=self.adapter._motor();restored=all(np.array_equal(motor[k],v) for k,v in self.adapter.original.items())
        retained.runtime.task.verify_pins(self.protocol['input_sha256']);offset=relative_pose(self._pose(snapshot),self.prep['start_xyyaw'])
        checks=dict(all4997_rows145_source_settings_and_original_physics_retained=True,
            original_all12joint_gains_targets_and_limits_restored=restored,all_frozen_inputs_retained=True,
            actual_final_start_under10mm=math.hypot(*offset[:2])<=.010,actual_final_heading_under_half_deg=abs(offset[2])<=math.radians(.5),
            actual_final_speed_under5mm_s=math.hypot(*snapshot['velocity'][:2])<=.005,
            actual_final_nominal_steering_under1deg=max(abs(snapshot['positions'][i]) for i in self.adapter.steering)<=math.radians(1.),
            actual_Pause_before_whole_static_cap=self.clock<=held_static_contract()['maximum_physics_duration_s'],
            actual_geometry_support_and_monitors_healthy=not self.runtime_guard.failed and not self.errors,
            self_query_hits_are_exact_own_source_nodes=all(h['collision'] in self.source for r in support for h in r['own_robot_occlusion_hits']))
        checks.update({'support_query_'+k:bool(v) for k,v in positive['checks'].items()});checks.update(negative['checks'])
        return dict(checks=checks,environment=checked,actual_post_held_static_tensor_snapshot=snapshot,
            first_source_state_writeback_proofs=self.post_observation_boundary_audits,
            actual_sampled_cooked_support_queries=support,actual_disabled_subfloor_probes=absence,
            support_query_evaluation=positive,absence_probe_evaluation=negative,
            held_static_NOT_released_static_route_or_final_statistics=True)


FRESH_HELD_VERSION='H6_UNCHANGED_COMMON_ENVIRONMENT_FRESH_HELD_STATIC_v1'


def reviewed_mission_schema_extension(before,after):
    """Prove ONLY the explicit H3 schema tag was added to the frozen kernel.

    Both complete source ASTs must otherwise be identical. This cannot waive
    a changed propagator, controller, mode, gain, collision or acceptance.
    Historical bytes/digests remain in their original archives and reports.
    """
    import ast
    before_tree=ast.parse(before);after_tree=ast.parse(after)
    if 'measured_replan_start' in (after.decode() if isinstance(after,bytes) else after):
        from hospital_sensor_navigation import original_nominal_execution_ast
        after_tree=original_nominal_execution_ast(after_tree)
    if 'measured_replan_start' in (before.decode() if isinstance(before,bytes) else before):
        from hospital_sensor_navigation import original_nominal_execution_ast
        before_tree=original_nominal_execution_ast(before_tree)
    old=ast.parse("geometry.get('scenario')!='H6_integrated_transfer_mission'",mode='eval').body
    scenarios=('H6_integrated_transfer_mission','H3_doorway_immediate_turn','H1_bedside_departure',
        'H2_bedside_lateral_alignment','H4_ward_corridor_turn','H5_obstacle_bypass_return')
    variants=[ast.parse("geometry.get('scenario') not in "+repr(tags),mode='eval').body
        for tags in (scenarios[:2],scenarios)]
    wanted={ast.dump(node,include_attributes=False) for node in variants};changed=[]
    class Undo(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False) in wanted:
                changed.append(node);return copy.deepcopy(old)
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('(180.,240.,300.,600.)',mode='eval').body,include_attributes=False):
                return ast.copy_location(ast.parse('(300.,600.)',mode='eval').body,node)
            return self.generic_visit(node)
    restored=Undo().visit(after_tree);after_changes=len(changed);baseline=Undo().visit(before_tree)
    if after_changes!=1 or ast.dump(restored,include_attributes=False)!=ast.dump(baseline,include_attributes=False):
        raise ValueError('Prior physical/evidence dependency changed beyond the explicit H3 schema identity')
    return dict(only_H3_schema_identity_added=True,explicit_task_schema_and_declared_budget_whitelist_only=True,
        whole_original_controller_and_all_guards_AST_identical=True)


def reviewed_prior_GET_dependencies(mapping,archive_path):
    """Resolve ONLY consumed runtime/test bytes from their exact saved ZIP.

    Does not permit rerunning old code, changing a physical kernel, replacing
    evidence or ignoring its checksum. New workflow code gets current pins.
    """
    import zipfile
    evolving={Path(retained.__file__).resolve(),Path(__file__).resolve(),
        retained.ROOT/'studies/IJCAS/scripts/hospital_mission_sequence.py',
        retained.ROOT/'studies/IJCAS/scripts/hospital_mission_execution.py',
        retained.ROOT/'studies/IJCAS/scripts/audit_hospital_paper_readiness.py',
        retained.ROOT/'studies/IJCAS/scripts/hospital_sensor_navigation.py',
        retained.ROOT/'studies/IJCAS/scripts/hospital_mission_planning.py',
        retained.ROOT/'studies/IJCAS/scripts/plan_hospital_h3_native.py',
        retained.ROOT/'studies/IJCAS/scripts/screen_hospital_h4_common_executor.py',
        retained.ROOT/'src/patient_initial_stability/test/test_hospital_paper_protocol.py',
        retained.ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',
        retained.ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py'}
    resolved=[]
    with zipfile.ZipFile(archive_path) as archive:
        for name,digest in mapping.items():
            path=Path(name)
            if retained.runtime.task.sha(path)==digest:continue
            if path not in evolving:raise ValueError('Prior physical/evidence dependency changed: '+name)
            member=path.relative_to(retained.ROOT).as_posix()
            archived=archive.read(member)
            if hashlib.sha256(archived).hexdigest()!=digest:raise ValueError('Prior physical/evidence dependency has no exact archived source: '+name)
            if path==Path(motion.__file__).resolve():reviewed_mission_schema_extension(archived,path.read_bytes())
            resolved.append(dict(path=name,executed_sha256=digest,archive=str(archive_path),member=member))
    return dict(verified_pins=len(mapping),resolved_consumed_sources=resolved,
        historical_source_resolution_is_NOT_rerun_or_new_launch_authority=True)


def checked_fresh_held_approval(approval):
    # This record is created ONLY inside the user-executed, explicitly named
    # scope-flag entry below, not exported as a pretend answered chat question.
    from hospital_mission_sequence import TRACKING_FLOOR_APPROVALS,DIRECTIONAL_QUIET_STOP_APPROVAL,COMPLIANT_HOLD_APPROVAL,ITERATIVE_FIX_APPROVAL,H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL
    from hospital_sensor_navigation import H5_REPLAN_SCOPE
    if type(approval) is dict and approval==H5_REPLAN_SCOPE['approval']:return dict(approval)
    if type(approval) is dict and approval in (*TRACKING_FLOOR_APPROVALS,DIRECTIONAL_QUIET_STOP_APPROVAL,COMPLIANT_HOLD_APPROVAL,ITERATIVE_FIX_APPROVAL,H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL):return dict(approval)
    direct=dict(approval_source='USER_EXECUTED_SCRIPT_EDITOR_NAMED_SCOPE_FLAG',
        scope='ONE_ARCHIVE_STOP_NEW_SOURCE_HELD_STATIC_NO_ROUTE_NO_RETRY',explicit_flag=True)
    if (type(approval) is dict and approval==direct
            and approval.get('explicit_flag') is True):return dict(direct)
    batch=remaining_validation_approval()
    if type(approval) is dict and approval==batch:return dict(batch)
    if (type(approval) is not dict or set(approval)!={'question_item_id','answer_ko'}
            or not isinstance(approval['question_item_id'],str) or not approval['question_item_id'].strip()
            or approval['answer_ko'] not in ('검증 후 새 코호트 정지 유지 1회 승인','새 코호트 정지유지 1회 승인',
                '수정 검증 후 정지유지 1회 승인','수정 검증 후 정지유지 재검증 1회 승인',
                '수정 검증 후 새 코호트·정적·P 재검증 1회 승인')):
        raise ValueError('Answered explicit ONE fresh-source held-static approval required; no inferred consent')
    return dict(approval)


def remaining_validation_approval():
    """Actual user message authorizes sequencing, not fabricated Q/A consent.

    This record grants only the held-static subset when used by this module.
    Every other cell still needs its own frozen definition and safety checks;
    failures are retained, not retried until success or silently replaced.
    """
    return dict(approval_source='USER_MESSAGE',
        scope='ONE_ARCHIVE_STOP_NEW_SOURCE_HELD_STATIC_NO_ROUTE_NO_RETRY',
        user_message='그러면 지금 남은 시험들 있잖아 아직 완료 안된거 이거 하나하나씩 자동실험 될때까지 싹 돌리는거 가능하나?\n중간중간 실험 하나 끝날 때 결과 보고만 해주고 가능?')


def saved_held_abort_boundary(result_path):
    """Exact saved Pause/restoration/tensor -> next archive boundary, OFFLINE.

    This is not another GET, a successful static, or permission to resume.
    All original failed evidence and its executed-source archive stay intact.
    """
    result_path=Path(result_path).resolve();raw=retained.read(result_path)
    packet_path=Path(raw['protocol']);packet=retained.read(packet_path)
    prep_path=Path(raw['preparation_record']);prep=retained.read(prep_path)
    final=raw['actual_final_readback'];state=raw['actual_final_timeline']
    if (raw['verdict']!='HOLD' or raw['new_route_trials']!=0 or raw['automatic_static_Play_requested'] is not True
            or raw['protocol_sha256']!=retained.runtime.task.sha(packet_path)
            or raw['checks'].get('actual_Pause_confirmed') is not True
            or raw['checks'].get('original_gains_targets_restored') is not True
            or final.get('stable_PAUSED_tensor') is not True or final['timeline']!=state
            or state['playing'] or state['stopped'] or raw['pause_confirmation']['after']!=state
            or result_path!=packet_path.parent/'held_static_readback.json'
            or prep['new_robot_root']!=packet['actor']):
        raise ValueError('Own saved HOLD with confirmed stable Pause and exact restoration required')
    snapshot=final['tensor'];baseline=prep['expected_nominal_motor_baseline']
    if (len(snapshot['names'])!=12 or len(snapshot['link_names'])!=13 or snapshot['names']!=baseline['joint_names']
            or any(snapshot['gains'][k]!=baseline[k] for k in ('kp','kd','max_efforts'))
            or snapshot['joint_position_targets_rad']!=[0.]*12 or snapshot['joint_velocity_targets_rad_s']!=[0.]*12):
        raise ValueError('All12 restored motor settings/zero targets and13 links required')
    workflow_path=Path(packet['workflow_protocol']);workflow=retained.read(workflow_path)
    archive=workflow_path.parent/'workflow_sources.zip'
    if packet['input_sha256'].get(str(workflow_path))!=retained.runtime.task.sha(workflow_path):
        raise ValueError('Own original workflow lineage changed')
    verify_workflow_sources(archive,workflow['input_sha256'])
    reviewed_prior_GET_dependencies(workflow['input_sha256'],archive)
    files=(result_path,packet_path,prep_path,workflow_path,archive)
    pins=dict(workflow['input_sha256']);pins.update({str(p):retained.runtime.task.sha(p) for p in files})
    return normalize_runtime_evidence(dict(status='OFFLINE_EXACT_SAVED_HELD_ABORT_HANDOFF_NOT_A_NEW_LIVE_OBSERVATION',
        source_result=str(result_path),source_result_sha256=retained.runtime.task.sha(result_path),
        source_archive=str(archive),preparation_record=str(prep_path),original_failed_result=str(result_path),
        actual_layer_texts=final['layer_texts'],same_stage=True,Root_Session_unchanged_during_GET=True,
        SDK_GET=dict(actor=packet['actor'],timeline=state,actual_tensor_snapshot=snapshot,tensor_stable_during_GET=True,
            observed_hold_confirmation=False,original_motor_baseline_backend_readback_verified=True),
        legacy_SDK_GET_fields_derived_from_saved_final_readback_NOT_a_new_GET=True,
        original_HOLD_NOT_promoted_and_no_route_admission=True,new_live_actions=0,input_sha256=pins))


def export_saved_held_abort_handoff(result_path):
    """Append-only exact derivation of existing bytes; no SDK action."""
    value=saved_held_abort_boundary(result_path)
    target=Path(result_path).resolve().parent/'saved_abort_preparation_handoff_r01.json'
    if target.exists():
        if retained.read(target)!=value:raise ValueError('Existing handoff differs; preserve it, no overwrite')
    else:retained.evidence.write_exclusive(target,value)
    return target


def fresh_held_workflow_definition(*,cohort='cohort_02_held_static_r01',approval=None,prior_handoff=None,task_binding=None):
    """OFFLINE definition for the actual saved H6 HOLD; NO output/SDK action.

    Without an answered approval the result is explicitly NON-runnable.
    The current displaced pose is archived, NEVER substituted for the fixed
    nominal start. Existing scene environment/old FAILs are not modified.
    """
    import re
    from hospital_mission_sequence import COMPLIANT_HOLD_APPROVAL,checked_compliant_probe_condition,ITERATIVE_FIX_APPROVAL,checked_stop_connection_model_condition
    if approval==COMPLIANT_HOLD_APPROVAL:checked_compliant_probe_condition(prior_handoff)
    if approval==ITERATIVE_FIX_APPROVAL:checked_stop_connection_model_condition()
    if re.fullmatch(r'cohort_[0-9]{2}_held_static_r[0-9]{2}',cohort) is None:
        raise ValueError('Explicit fresh held-static cohort name required')
    static=retained.read(retained.REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json');route_handoff=False
    if prior_handoff is None:
        folder=retained.REVIEWED_ZERO_FOLDER;get_path=folder/'paused_closeout_GET.json'
        prior=retained.read(get_path);zero=retained.read(folder/'zero_settling_readback.json')
        archive=folder/'paused_closeout_GET_sources.zip'
        if retained.runtime.task.sha(get_path)!='1e65865852b264276e87c2332fc55fa433649d0ebfcd6cd9d5c71da234a2f3fe':
            raise ValueError('Exact saved current HOLD GET required; no replaced receipt')
        fields=static['runtime_anonymous_source_reference']['fixed_configuration_rows']
        prior_preparation=None
    elif retained.read(prior_handoff).get('status')=='OFFLINE_EXACT_SAVED_HELD_ABORT_HANDOFF_NOT_A_NEW_LIVE_OBSERVATION':
        get_path=Path(prior_handoff).resolve();folder=get_path.parent;prior=retained.read(get_path)
        if prior!=saved_held_abort_boundary(prior['source_result']):
            raise ValueError('Saved final Pause/tensor handoff changed; no rebaseline')
        prior_preparation=retained.read(prior['preparation_record']);zero=retained.read(prior['source_result'])
        archive=Path(prior['source_archive']);fields=prior_preparation['source_reference']['fixed_configuration_rows']
    elif retained.read(prior_handoff).get('status') in ('EXACT_SAVED_RESPONSE_PROBE_BOUNDARY_FOR_NEW_ARCHIVE_NOT_ROUTE_COMPLETION',
            'EXACT_SAVED_FAILED_RESPONSE_BOUNDARY_FOR_NEW_ARCHIVE_NOT_PASS'):
        from hospital_mission_sequence import saved_response_probe_preparation_handoff
        get_path=Path(prior_handoff).resolve();folder=get_path.parent;prior=retained.read(get_path)
        failed=prior['status']=='EXACT_SAVED_FAILED_RESPONSE_BOUNDARY_FOR_NEW_ARCHIVE_NOT_PASS'
        if failed and approval!=ITERATIVE_FIX_APPROVAL:raise ValueError('Failed response only retires under new evidence-led fix authority')
        if prior!=saved_response_probe_preparation_handoff(prior['source_result'],archive_failed=failed):
            raise ValueError('Own exact completed response probe/Pause/restoration/full environment required')
        prior_preparation=retained.read(prior['preparation_record']);zero=retained.read(prior['source_result'])
        archive=Path(prior['source_archive']);fields=prior_preparation['source_reference']['fixed_configuration_rows']
        route_handoff=True
    elif retained.read(prior_handoff).get('status')=='EXACT_SAVED_ROUTE_AND_COMPLETE_FINAL_GET_FOR_NEW_ARCHIVE_NOT_ROUTE_COMPLETION':
        # Consumed route outcomes remain their original PASS/FAIL/HOLD. Only
        # the independently verified final GET supplies the next archive
        # boundary; never resume the old dispatch or reuse its STATIC PASS.
        from hospital_mission_sequence import saved_route_preparation_handoff
        get_path=Path(prior_handoff).resolve();folder=get_path.parent;prior=retained.read(get_path)
        if prior!=saved_route_preparation_handoff(prior['source_result']):
            raise ValueError('Exact own saved route and complete actual final GET required')
        prior_preparation=retained.read(prior['preparation_record']);zero=retained.read(prior['source_result'])
        archive=Path(prior['source_archive']);fields=prior_preparation['source_reference']['fixed_configuration_rows']
        route_handoff=True
    else:
        get_path=Path(prior_handoff).resolve();folder=get_path.parent;prior=retained.read(get_path)
        if prior['status']!='OFFLINE_EXACT_SAVED_POSTABORT_GET_HANDOFF_NOT_A_NEW_LIVE_OBSERVATION':
            raise ValueError('Exact saved postabort handoff required, not a successful static or rebaseline')
        observed_path=Path(prior['source_GET']);observed=retained.read(observed_path)
        prior_preparation=retained.read(prior['preparation_record']);zero=retained.read(prior['original_failed_result'])
        archive=folder/'postabort_GET_source_r01.zip'
        if (get_path.name!='paused_preparation_handoff_r01.json' or observed_path.parent!=folder
                or observed_path.name!='postabort_pause_state_GET_r01.json'
                or retained.runtime.task.sha(observed_path)!=prior['source_GET_sha256']
                or observed['prior_result']!=prior['original_failed_result']
                or retained.runtime.task.sha(prior['original_failed_result'])!=observed['prior_result_sha256']
                or observed['status']!='H6_POSTABORT_PAUSE_AND_EXISTING_STATE_GET_NOT_STATIC_OR_DRIVE_ADMISSION'
                or observed.get('error') or observed['tensor_stable_during_GET'] is not True
                or observed['Root_Session_unchanged_during_GET'] is not True
                or prior['SDK_GET']['timeline']!=observed['actual_final_timeline']
                or prior['SDK_GET']['actual_tensor_snapshot']!=observed['actual_tensor_snapshot']
                or prior['actual_layer_texts']!=observed['actual_layer_texts']
                or prior['SDK_GET']['actor']!=observed['actor'] or observed['actor']!=prior_preparation['new_robot_root']
                or zero['automatic_static_Play_requested'] is not True or zero['new_route_trials']!=0):
            raise ValueError('Exact unchanged saved PAUSED postabort GET and own consumed HOLD required')
        # The old HOLD's final saved clock preceded its deferred Play. Keep
        # that record intact; the newer GET, not the old zero clock, is the
        # actual retiring boundary for this separately approved fresh source.
        snapshot=observed['actual_tensor_snapshot'];baseline=prior_preparation['expected_nominal_motor_baseline']
        if (snapshot['names']!=baseline['joint_names']
                or any(snapshot['gains'][k]!=baseline[k] for k in ('kp','kd','max_efforts'))
                or snapshot['joint_position_targets_rad']!=[0.]*12
                or snapshot['joint_velocity_targets_rad_s']!=[0.]*12
                or baseline!=static['runtime_motor_baseline']):
            raise ValueError('Exact original motor baseline/zero targets required; no automatic repair')
        fields=prior_preparation['source_reference']['fixed_configuration_rows']
    snapshot=prior['SDK_GET']['actual_tensor_snapshot'];timeline=prior['SDK_GET']['timeline']
    if ((zero['verdict'] not in ('PASS','FAIL','HOLD') if route_handoff else zero['verdict']!='HOLD')
            or (prior_handoff is None and timeline!=zero['actual_final_timeline']) or timeline['playing'] or timeline['stopped']
            or prior['SDK_GET']['observed_hold_confirmation'] is not False
            or prior['SDK_GET']['tensor_stable_during_GET'] is not True
            or prior['Root_Session_unchanged_during_GET'] is not True or prior['same_stage'] is not True
            or prior['SDK_GET']['original_motor_baseline_backend_readback_verified'] is not True
            or len(snapshot['names'])!=12 or len(snapshot['link_names'])!=13):
        raise ValueError('Exact PAUSED complete current HOLD/original motor baseline required, NOT a route release')
    history=reviewed_prior_GET_dependencies(prior['input_sha256'],archive)
    g=retained.read(retained.GEOMETRY);output=retained.GATE/'hospital_h6_experiments'/cohort
    robot='/World/patient_transport_h6_'+cohort;marker='/World/IJCAS_HospitalH6_'+cohort
    if robot==prior['SDK_GET']['actor']:raise ValueError('Never reuse the displaced retiring actor')
    if len(fields)!=145 or static['simulation_mass_kg']!=265. or static['start_xyyaw']!=g['start_xyyaw']:
        raise ValueError('Complete original fixed-start H6 source definition required')
    geometry_path=Path(retained.GEOMETRY);grid_path=Path(g['occupancy_grid'])
    with np.load(grid_path) as grid:
        MissionRuntimeGuard(g,grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m']),
            simulation_budget_s=held_static_contract()['maximum_physics_duration_s'],static_observation=True)
        if hashlib.sha256(grid['occupied'].tobytes()).hexdigest()!=g['planning_occupancy_digest']:
            raise ValueError('Frozen original H6 raster digest changed')
    user_approval=None if approval is None else checked_fresh_held_approval(approval)
    files=[get_path,archive,
        retained.REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json',retained.H6/'static_protocol.json',
        retained.H6/'actual_common_environment_rows.json',geometry_path,grid_path,retained.shared.preparation.ASSET,*host_source_files()]
    if prior_handoff is None:files.extend((folder/'protocol.json',folder/'zero_settling_readback.json',folder/'zero_settling_trace.csv'))
    else:files.extend((Path(prior.get('source_GET',prior.get('source_result'))),Path(prior['preparation_record']),
        Path(prior.get('original_failed_result',prior.get('source_result')))))
    # Historical evidence keeps old source pins in its own saved report/ZIP;
    # this NEW code packet uses current source bytes, never rewrites the old.
    active_pins=dict(prior['input_sha256'])
    for row in history['resolved_consumed_sources']:active_pins.pop(row['path'])
    active_pins.update({str(Path(p).resolve()):retained.runtime.task.sha(p) for p in files})
    result=dict(version=FRESH_HELD_VERSION,status='PROSPECTIVE_NOT_RUNNABLE' if user_approval is None else 'EXPLICIT_ONE_WORKFLOW_DEFINITION_NOT_EXECUTED',
        scenario='H6_integrated_transfer_mission',cohort=cohort,method='COMMON_CAPTURED_WHEEL_HELD_STATIC_NOT_ROUTE',
        output_directory=str(output),new_robot_root=robot,marker=marker,retired_robot_root=prior['SDK_GET']['actor'],
        required_prior_result=str(get_path),required_prior_tensor=snapshot,required_prior_timeline=timeline,
        required_prior_layer_texts=prior['actual_layer_texts'],prior_GET_source_archive=str(archive),
        prior_verdict_preserved=zero['verdict'],retiring_actor_fixed_configuration=fields,
        expected_nominal_motor_baseline=static['runtime_motor_baseline'],
        start_xyyaw=g['start_xyyaw'],goal_xyyaw=g['goal_xyyaw'],geometry_file=str(geometry_path),grid_file=str(grid_path),
        protected_environment_sha256=static['physics_preparation_snapshot']['protected_environment_sha256'],
        environment_rows_file=str(retained.H6/'actual_common_environment_rows.json'),
        initialize_new_actor_visuals_before_static=True,Stop_wall_budget_s=15.,Stop_update_budget=240,
        compact_readback_wall_budget_s=180.,maximum_Stop_calls=1,maximum_static_trials=1,maximum_route_trials=0,
        automatic_retry=False,actual_motion_authorized=user_approval is not None,use_for_final_statistics=False,
        held_static_contract=held_static_contract(),user_approval=user_approval,
        historical_source_resolution=history,input_sha256=active_pins)
    if prior_handoff is not None:result['prior_handoff_path']=str(get_path)
    if task_binding is not None:
        from hospital_mission_sequence import checked_h3_task_binding,H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL
        if approval not in (H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL):raise ValueError('Separate actual whole-task authority required')
        task=checked_h3_task_binding(task_binding)
        if task['approval']!=approval:raise ValueError('Exact task approval lineage required')
        task_geometry=retained.read(task['geometry_file'])
        task_key=task['scenario'].split('_')[0]
        result.update(scenario=task['scenario'],output_directory=str(retained.GATE/('hospital_'+task_key.lower()+'_experiments')/cohort),
            new_robot_root='/World/patient_transport_'+task_key.lower()+'_'+cohort,marker='/World/IJCAS_Hospital'+task_key+'_'+cohort,
            start_xyyaw=task_geometry['start_xyyaw'],goal_xyyaw=task_geometry['goal_xyyaw'],
            geometry_file=task['geometry_file'],grid_file=task_geometry['occupancy_grid'],
            task_binding_path=str(Path(task_binding).resolve()),static_query_contract_file=task['static_query_contract_file'])
        result['input_sha256'].update(task['input_sha256'])
        result['input_sha256'][str(Path(task_binding).resolve())]=retained.runtime.task.sha(task_binding)
    return normalize_runtime_evidence(result)


def fresh_held_preparation_operations(context,sdk,*,authorized=False):
    """Reuse original archive/ONE Stop/constructor/FIRST-display code object.

    No environment patch: all4997 rows must remain the SAME already adopted
    H6 environment. Only two existing Top aperture defaults are classified
    separately during preparation. Never replace an unreviewed layer baseline.
    """
    if authorized is not True:raise ValueError('Separate fresh preparation+held-static authority required before SDK binding')
    protocol=context.protocol
    if (protocol.get('version')!=FRESH_HELD_VERSION or protocol.get('actual_motion_authorized') is not True
            or (protocol.get('maximum_Stop_calls'),protocol.get('maximum_static_trials'),protocol.get('maximum_route_trials'))!=(1,1,0)
            or protocol.get('automatic_retry') is not False or protocol.get('held_static_contract')!=held_static_contract()):
        raise ValueError('Exact ONE Stop/ONE held-static/NO route scope required')
    checked_fresh_held_approval(protocol.get('user_approval'))
    from pxr import Sdf,UsdGeom
    shared=retained.shared;p=context.preparation;protocol=context.protocol;memory={}
    usd=sdk.omni.usd.get_context();stage=usd.get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    def texts():return dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    def layers(reference,current):
        if reference==current:
            # Full byte equality is stronger than reparsing the same large
            # Session hundreds of times at UI yields. No hashes/field masks:
            # every current Root/Session byte is still compared on each call.
            return dict(exact_Top_default_differences=[],exact_renderer_default_differences=[],
                exact_Root_Session_bytes_unchanged=True)
        renderer=retained.split_viewport_renderer_defaults(stage,reference,current)
        normalized=renderer['normalized_after']
        if normalized['Root']!=reference['Root']:raise ValueError('Root changed during nonplaying fresh preparation')
        review=retained.canonical_display_only_review(reference['Session'],normalized['Session'])
        if any(row['path'] not in PREPARATION_DISPLAY_PATHS for row in review['exact_Top_default_differences']):
            raise ValueError('Only declared Top XY/aperture defaults may change during fresh preparation')
        return dict(review,exact_renderer_default_differences=renderer['exact_renderer_default_differences'])
    layers=memoized_exact_layer_review(layers)
    def marker():
        prim=stage.GetPrimAtPath(context.cohort.MARKER)
        if (not prim.IsA(UsdGeom.Xform) or prim.GetChildren() or prim.GetAppliedSchemas()
                or prim.GetAuthoredAttributes() or prim.GetAuthoredRelationships()
                or prim.GetCustomData()!=dict(robot_root=context.cohort.ROBOT,status='fresh_static_pending_NO_PLAY',cohort=protocol['cohort'])):
            raise ValueError('Exact nonphysical marker required; no extra marker child/attribute/API')
    def source():return shared.fresh_source_for_live_actor(context,stage)
    def owned_handles(own):
        # Keep the SAME captured objects through preparation AND display.
        # The original identity guard below remains intact. Do not compare
        # separately acquired native wrappers or replace checks by values.
        if own is not context or sdk.omni.usd.get_context().get_stage()!=stage:
            raise ValueError('Own active fresh preparation context/stage required')
        return stage,timeline
    class GetterView:
        """Private, read-only factory view; never patch a global SDK module."""
        def __init__(self,original,**overrides):self.original,self.overrides=original,overrides
        def __getattr__(self,name):
            if name in self.overrides:return self.overrides[name]
            return getattr(self.original,name)
    owned_usd=GetterView(usd,get_stage=lambda:owned_handles(context)[0])
    owned_omni=GetterView(sdk.omni,
        usd=GetterView(sdk.omni.usd,get_context=lambda:owned_usd),
        timeline=GetterView(sdk.omni.timeline,get_timeline_interface=lambda:owned_handles(context)[1]))
    def bind_owned_getters(function,**bindings):
        # Preserve the original shared source and bytecode, including all
        # exact field/control/Stop guards. Only these TWO bound functions'
        # private imports see the owned getters. Other imported symbols,
        # native modules, host builtins and other tasks are untouched.
        original_builtins=function.__globals__['__builtins__']
        originals=original_builtins if isinstance(original_builtins,dict) else vars(original_builtins)
        normal_import=originals['__import__']
        def private_import(name,globals=None,locals=None,fromlist=(),level=0):
            imported=normal_import(name,globals,locals,fromlist,level)
            if level==0 and not fromlist and name in ('omni.usd','omni.timeline','omni.kit.app'):
                return owned_omni
            return imported
        bound=shared._bound(function,__builtins__={**originals,'__import__':private_import},**bindings)
        bound._owned_sdk_handles=owned_handles
        return bound
    async def environment(own,supplied_stage,supplied_timeline,phase):
        if own is not context or supplied_stage!=stage or supplied_timeline!=timeline:raise ValueError('Own fresh preparation stage required')
        expected=protocol['required_prior_timeline'] if phase=='before_archive' else dict(retained.boundary.STOPPED)
        first=texts();started=watch=time.monotonic();rows={}
        def stable():
            if (usd.get_stage()!=stage or sdk.omni.usd.get_context().get_stage()!=stage
                    or retained.evidence.timeline_state(timeline)!=expected):
                raise RuntimeError('Own nonplaying stage/clock changed during preparation')
            layers(first,texts())
            if time.monotonic()-started>protocol['compact_readback_wall_budget_s']:raise TimeoutError('Bounded preparation row GET')
        if phase=='before_archive':layers(protocol['required_prior_layer_texts'],first)
        if phase=='before_renderer_initialization':
            # A full unchanged environment scan was completed immediately
            # after Stop. Prove only the declared source/marker addition here
            # to avoid UI frames inside the renderer's exact snapshot.
            reference=memory['completed_Stop_layers'];current=texts()
            normalized=retained.split_viewport_renderer_defaults(stage,reference,current)['normalized_after']
            retained.boundary.staging_scope_review(context,reference,normalized);marker();source()
            layers(dict(Root=reference['Root'],Session=current['Session']),current)
            return dict(row_count=4997,sha256=protocol['protected_environment_sha256'])
        if phase not in ('before_archive','completed_Stop','new_source_staged'):raise ValueError('Unknown fresh phase')
        for path,row in p.compact.stage_rows(stage,p.SELECTED,p.old.original):
            if path in rows:raise ValueError('Duplicate protected environment row')
            rows[path]=row
            if len(rows)%16==0:stable();await sdk.omni.kit.app.get_app().next_update_async();stable()
            if time.monotonic()-watch>=15.:
                print('[H6 HELD PREPARATION] %s rows=%d/4997; NO Play'%(phase,len(rows)),flush=True);watch=time.monotonic()
        stable();baseline=copy.deepcopy(retained.read(protocol['environment_rows_file'])['compact_rows'])
        for path in p.SELECTED:
            if baseline[path]['collision_enabled'] is not False:raise ValueError('Original four disabled identities changed')
            baseline[path]['collision_enabled']=True
        digest=p.compact.compare_rows(baseline,rows,p.SELECTED,after_flags=True)
        if len(rows)!=4997 or digest!=protocol['protected_environment_sha256']:raise ValueError('Same complete H6 environment required')
        memory[phase+'_layers']=texts()
        if phase=='new_source_staged':marker();source()
        return dict(row_count=4997,sha256=digest)
    def review_Stop(before,after,robot):
        if before==memory['before_archive_layers']['Root']:
            baseline=dict(Root=before,Session=memory['before_archive_layers']['Session'])
            renderer=retained.split_viewport_renderer_defaults(stage,baseline,dict(baseline,Root=after))
            return dict(retiring_actor_motion_fields=shared.review_fresh_Stop_motion_fields(before,renderer['normalized_after']['Root'],robot),
                exact_renderer_default_differences=renderer['exact_renderer_default_differences'])
        if before!=memory['before_archive_layers']['Session']:raise ValueError('Unknown pre-Stop layer boundary')
        baseline=dict(Root=memory['before_archive_layers']['Root'],Session=before)
        renderer=retained.split_viewport_renderer_defaults(stage,baseline,dict(baseline,Session=after))
        after=renderer['normalized_after']['Session']
        suffix='.usda' if before.lstrip().startswith('#usda') else '.sdf'
        first,last=[Sdf.Layer.CreateAnonymous('H6_STOP_DISPLAY_COPY'+suffix) for _ in range(2)]
        if not first.ImportFromString(before) or not last.ImportFromString(after):raise ValueError('Cannot parse Stop layer COPIES')
        camera=[]
        for row in shared.layer_field_differences(before,after):
            if row['path'] in PREPARATION_DISPLAY_PATHS:
                retained.boundary._validate_display_row(row);camera.append(row)
                first.GetAttributeAtPath(row['path']).default=last.GetAttributeAtPath(row['path']).default
        return dict(retiring_actor_motion_fields=shared.review_fresh_Stop_motion_fields(first.ExportToString(),last.ExportToString(),robot),
            exact_Top_aperture_default_differences=camera,exact_renderer_default_differences=renderer['exact_renderer_default_differences'])
    original_display=bind_owned_getters(shared.initialize_fresh_staged_display,preparation=p,
        fresh_source_for_live_actor=shared.fresh_source_for_live_actor,fresh_environment_readback=environment)
    async def display(own):
        first=texts();last=first;stable=0
        for _ in range(60):
            if retained.evidence.timeline_state(timeline)!=dict(retained.boundary.STOPPED):raise ValueError('Stopped display preparation required')
            layers(first,texts());await sdk.omni.kit.app.get_app().next_update_async();current=texts();layers(first,current)
            stable=stable+1 if last==current else 0;last=current
            if stable>=3:break
        else:raise TimeoutError('Aperture defaults failed to settle before exact FIRST display')
        return await original_display(own)
    prior_map=retained.read(protocol['required_prior_result'])['input_sha256'];original_verify=p.verify
    def verify(mapping):
        if mapping==prior_map:return reviewed_prior_GET_dependencies(mapping,protocol['prior_GET_source_archive'])
        return original_verify(mapping)
    p.verify=verify
    return bind_owned_getters(shared.prepare_fresh_live_trial,preparation=p,
        fresh_environment_readback=environment,initialize_fresh_staged_display=display,
        fresh_source_for_live_actor=shared.fresh_source_for_live_actor,retiring_actor_configuration=shared.retiring_actor_configuration,
        review_fresh_Stop_motion_fields=review_Stop,print=lambda *a,**kw:print(*[v.replace('[H4 FRESH]','[H6 HELD PREPARATION]').replace(
            'Completed prior H4 trial','Current H6 HOLD') if isinstance(v,str) else v for v in a],**kw))


def memoized_exact_layer_review(classifier):
    """Reuse a proof ONLY when both complete old/current text dictionaries match.

    Keep one pair, not an allowlist or hash-only approximation. Every changed
    byte invokes the original classifier; failed proofs are never cached.
    Copy returned proofs so external mutation cannot corrupt a later audit.
    """
    saved=None
    def checked(before,after):
        nonlocal saved
        if saved is not None and before==saved[0] and after==saved[1]:return copy.deepcopy(saved[2])
        proof=classifier(before,after)
        saved=(dict(before),dict(after),copy.deepcopy(proof))
        return proof
    return checked


def export_fresh_held_workflow(*,approval=None,cohort='cohort_02_held_static_r01',prior_handoff=None,task_binding=None):
    """OFFLINE export ONLY after an actual answered approval; never run Kit."""
    approval=checked_fresh_held_approval(approval)
    protocol=fresh_held_workflow_definition(cohort=cohort,approval=approval,prior_handoff=prior_handoff,task_binding=task_binding)
    output=Path(protocol['output_directory']);path=output/'workflow_protocol.json'
    if output.exists():raise ValueError('Existing cohort/partial export preserved; no overwrite or second definition')
    retained.evidence.write_exclusive(path,protocol)
    archive=output/'workflow_sources.zip'
    freeze_workflow_sources(archive,protocol['input_sha256'])
    return path


def freeze_workflow_sources(archive,pins):
    """Append-only exact code ZIP; no code loading or live operation."""
    import os,zipfile
    archive=Path(archive)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as frozen:
        for name in sorted(pins):
            file=Path(name)
            if file.suffix=='.py' and file.is_relative_to(retained.ROOT):frozen.write(file,file.relative_to(retained.ROOT).as_posix())
    with archive.open('rb') as stream:os.fsync(stream.fileno())
    return verify_workflow_sources(archive,pins)


def verify_workflow_sources(archive,pins):
    import zipfile
    expected={Path(name).relative_to(retained.ROOT).as_posix():digest for name,digest in pins.items()
        if Path(name).suffix=='.py' and Path(name).is_relative_to(retained.ROOT)}
    with zipfile.ZipFile(archive) as frozen:
        entries=frozen.infolist()
        if (len(entries)!=len(expected) or set(frozen.namelist())!=set(expected)
                or any(item.file_size>2*1024*1024 for item in entries) or sum(item.file_size for item in entries)>16*1024*1024):
            raise ValueError('Exact bounded workflow source archive members required')
        for name,digest in expected.items():
            if hashlib.sha256(frozen.read(name)).hexdigest()!=digest:raise ValueError('Exact workflow source archive failed verification; no launch')
    return dict(source_members=len(expected),source_archive=str(archive),source_archive_sha256=retained.runtime.task.sha(archive))


def load_fresh_held_workflow(path):
    path=Path(path).resolve();protocol=retained.read(path)
    if path!=Path(protocol['output_directory'])/'workflow_protocol.json':raise ValueError('Exact own workflow packet path required')
    if protocol.get('actual_motion_authorized') is not True:raise ValueError('Prospective packet grants no live authority')
    expected=fresh_held_workflow_definition(cohort=protocol['cohort'],approval=protocol['user_approval'],
        prior_handoff=protocol.get('prior_handoff_path'),task_binding=protocol.get('task_binding_path'))
    if protocol!=expected:raise ValueError('Exact approved current-HOLD/fixed-start/environment/workflow definition required')
    retained.runtime.task.verify_pins(protocol['input_sha256'])
    context=retained.geometry.common_preparation_context(protocol,path)
    context.source_archive=path.parent/'workflow_sources.zip'
    verify_workflow_sources(context.source_archive,protocol['input_sha256'])
    return context


async def finish_unstarted_staged_static_once(path,*,authorized=False):
    """Finish ONE already-staged source after the exact pre-Play row timeout.

    No original dispatch is replayed. No Stop, constructor, display refresh,
    physical-state restoration or second static exists here. The old failure
    and source ZIP remain immutable; new executing helpers get their own pins.
    """
    if authorized is not True:raise ValueError('Explicit remaining FIRST static authority required')
    path=Path(path).resolve();protocol=retained.read(path);output=path.parent
    if path!=Path(protocol['output_directory'])/'workflow_protocol.json':raise ValueError('Own original workflow required')
    checked_fresh_held_approval(protocol['user_approval'])
    error_path=output/'workflow_error.json';error=retained.read(error_path)
    display_path=output/'display_initialization/renderer_initialization.json';display=retained.read(display_path)
    stop_path=output/'actual_Stop_boundary.json';stop=retained.read(stop_path)
    archive=output/'workflow_sources.zip';verify_workflow_sources(archive,protocol['input_sha256'])
    reviewed_prior_GET_dependencies(protocol['input_sha256'],archive)
    if (error.get('error')!='TimeoutError: Bounded preparation row GET'
            or error.get('phase')!='actual_HOLD_archive_ONE_Stop_new_source_FIRST_display'
            or error['actual_failure_readback']['timeline']!=dict(retained.boundary.STOPPED)
            or stop['actual_timeline']!=dict(retained.boundary.STOPPED) or stop['Stop_called_once'] is not True
            or display['actor']!=protocol['new_robot_root'] or display['timeline_before_and_after']!=dict(retained.boundary.STOPPED)
            or display['registered_visual_root_count']!=13 or display['visible_USD_shape_count']!=22
            or display['original_Root_and_Session_text_exactly_restored'] is not True):
        raise ValueError('Exact completed ONE Stop/FIRST display/pre-Play row timeout required')
    context=retained.geometry.common_preparation_context(protocol,path);context.source_archive=archive
    reservation=output/'remaining_preparation_reserved.json';closeout=output/'remaining_static_closeout.json'
    if any(p.exists() for p in (reservation,closeout,context.preparation.LIVE,output/'held_static/static_protocol.json')):
        return dict(status='REMAINING_FIRST_STATIC_ALREADY_RESERVED_NO_ACTION')
    retained.no_other_task();sdk=retained.runtime.live_sdk();stage=sdk.omni.usd.get_context().get_stage()
    timeline=sdk.omni.timeline.get_timeline_interface()
    if retained.evidence.timeline_state(timeline)!=dict(retained.boundary.STOPPED):
        raise ValueError('Same unplayed STOPPED zero-time source required')
    before=error['actual_failure_readback']['layer_texts']
    current=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    renderer=retained.split_viewport_renderer_defaults(stage,before,current);normalized=renderer['normalized_after']
    if normalized['Root']!=before['Root']:raise ValueError('Unreviewed Root edit since staged abort')
    display_review=retained.canonical_display_only_review(before['Session'],normalized['Session'])
    if any(r['path'] not in PREPARATION_DISPLAY_PATHS for r in display_review['exact_Top_default_differences']):
        raise ValueError('Unreviewed Session edit since staged abort')
    reference=retained.shared.fresh_source_for_live_actor(context,stage)
    made=display['source_actor_before_and_after']['constructor']
    validator=retained.shared.preparation.old.constructor_for(context.cohort.ROBOT).__globals__['validate_new_robot']
    if (reference['source_constructor_readback']!=made
            or validator(stage,sdk.Usd.Stage.Open(str(context.preparation.ASSET)),protocol['start_xyyaw'])!=made):
        raise ValueError('Own actual same-SDK265kg/13body/12joint pose and seed changed')
    retained.evidence.write_exclusive(reservation,dict(original_workflow=str(path),original_error=str(error_path),
        original_error_sha256=retained.runtime.task.sha(error_path),same_unplayed_actor=context.cohort.ROBOT,
        user_approval=protocol['user_approval'],maximum_additional_Stop_constructor_display_calls=0,
        maximum_first_static_trials=1,automatic_retry=False))
    phase='remaining_full_environment_GET';host=None
    try:
        operation=fresh_held_preparation_operations(context,sdk,authorized=True)
        owned_stage,owned_timeline=operation._owned_sdk_handles(context)
        environment=operation.__globals__['fresh_environment_readback']
        protected=await environment(context,owned_stage,owned_timeline,'new_source_staged')
        prep=dict(cohort=protocol['cohort'],new_robot_root=context.cohort.ROBOT,
            retired_robot_root=protocol['retired_robot_root'],start_xyyaw=protocol['start_xyyaw'],simulation_mass_kg=265.,
            live_new_robot_readback=made,source_constructor_readback=made,source_reference=reference,
            protected_environment_sha256=protected['sha256'],expected_nominal_motor_baseline=protocol['expected_nominal_motor_baseline'],
            timeline=retained.evidence.timeline_state(timeline),old_physical_continuity_NOT_restored=True,
            archive=str(output/'actual_prior_state_before_Stop.json'),
            archive_sha256=retained.runtime.task.sha(output/'actual_prior_state_before_Stop.json'),
            Stop_boundary=str(stop_path),Stop_boundary_sha256=retained.runtime.task.sha(stop_path),Play_or_static_or_drive_executed=False)
        prep['critical_environment']=context.stationary.critical_environment(stage,prep)
        layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
        prep.update({k+'_sha256':hashlib.sha256(v.encode()).hexdigest() for k,v in layers.items()})
        retained.evidence.write_exclusive(context.preparation.LIVE,prep)
        context.continuation_record=reservation
        bound,packet,static_path=bind_prepared_held_packet(context,prep,layers)
        geometry=retained.read(packet['geometry_file'])
        with np.load(packet['grid_file']) as grid:
            host=PreparedHeldStaticSDKHost(sdk,packet,static_path,context=bound,geometry=geometry,
                occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),resolution=float(grid['resolution_m']),authorized=True)
        phase='ONE_UNSTARTED_FIRST_STATIC'
        report=await run_held_static_once(host,HeldStaticController(protocol['start_xyyaw']),authorized=True)
        replay=replay_held_static_evidence(host.result)
        retained.evidence.write_exclusive(host.result.parent/'raw_replay.json',replay)
        result=dict(status='SAME_SOURCE_REMAINING_FIRST_STATIC_FINISHED',static_verdict=report['verdict'],
            static_result=str(host.result),raw_replay_verdict=replay['verdict'],additional_Stop_calls=0,
            additional_constructor_or_display_calls=0,route_trials=0,automatic_retry=False,original_error_preserved=True)
        retained.evidence.write_exclusive(closeout,result)
        print('[H6 REMAINING FIRST STATIC]',result,flush=True);return result
    except (Exception,asyncio.CancelledError) as failure:
        result=dict(status='REMAINING_FIRST_STATIC_ABORTED_NO_RETRY',phase=phase,
            error=type(failure).__name__+': '+str(failure),actual_failure_readback=capture_fresh_workflow_failure(context,sdk,host),
            additional_Stop_calls=0,automatic_retry=False)
        retained.evidence.write_exclusive(output/'remaining_static_error.json',result)
        print('[H6 REMAINING FIRST STATIC]',result['error'],flush=True);return result


def bind_prepared_held_packet(context,preparation,layers):
    """Freeze the SAME newly completed preparation; no relaxed rebaseline.

    Called synchronously immediately after staging, before another UI yield.
    The resulting file still needs the ordinary full live stopped preflight.
    """
    protocol=context.protocol;output=context.output
    if (preparation!=retained.read(context.preparation.LIVE)
            or preparation['new_robot_root']!=context.cohort.ROBOT or preparation['timeline']!=dict(retained.boundary.STOPPED)
            or preparation['Play_or_static_or_drive_executed'] is not False
            or any(hashlib.sha256(text.encode()).hexdigest()!=preparation[k+'_sha256'] for k,text in layers.items())
            or set(layers)!={'Root','Session'}):
        raise ValueError('Own exact first preparation/layer receipt required; no replacement baseline')
    for field in ('archive','Stop_boundary'):
        path=Path(preparation[field])
        if path.parent!=output or preparation[field+'_sha256']!=retained.runtime.task.sha(path):
            raise ValueError('Own durable archive/ONE Stop lineage required')
    old=retained.read(preparation['archive']);stop=retained.read(preparation['Stop_boundary'])
    if (old['actual_state']!=protocol['required_prior_tensor'] or old['timeline']!=protocol['required_prior_timeline']
            or old['prior_result']!=protocol['required_prior_result'] or old['prior_result_sha256']!=retained.runtime.task.sha(protocol['required_prior_result'])
            or stop['Stop_called_once'] is not True or stop['actual_timeline']!=dict(retained.boundary.STOPPED)
            or preparation['protected_environment_sha256']!=protocol['protected_environment_sha256']):
        raise ValueError('Exact preserved old HOLD/ONE completed Stop/same environment lineage required')
    layer_file=output/'prepared_layer_texts.json';retained.evidence.write_exclusive(layer_file,layers)
    query=copy.deepcopy(retained.read(protocol.get('static_query_contract_file',retained.H6/'static_protocol.json')))
    query['new_robot_root']=context.cohort.ROBOT;query['source_actor_node_paths']=list(preparation['source_reference']['fixed_configuration_rows'])
    static_path=output/'held_static'/'static_protocol.json'
    extra=[context.protocol_path,context.preparation.LIVE,layer_file,Path(preparation['archive']),Path(preparation['Stop_boundary']),
        output/'display_initialization/renderer_initialization.json',context.source_archive]
    verify_workflow_sources(context.source_archive,protocol['input_sha256'])
    pins=dict(protocol['input_sha256'])
    if hasattr(context,'continuation_record'):
        history=reviewed_prior_GET_dependencies(pins,context.source_archive)
        for row in history['resolved_consumed_sources']:pins.pop(row['path'])
        extra.extend((context.continuation_record,*host_source_files(),
            retained.ROOT/'studies/IJCAS/scripts/hospital_mission_sequence.py',
            retained.ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py'))
    pins.update({str(p):retained.runtime.task.sha(p) for p in extra})
    packet=dict(version=PreparedHeldStaticSDKHost.VERSION,cohort=protocol['cohort'],output_directory=str(output),
        actor=context.cohort.ROBOT,new_robot_root=context.cohort.ROBOT,retired_robot_root=protocol['retired_robot_root'],
        marker=context.cohort.MARKER,start_xyyaw=protocol['start_xyyaw'],run_id='hospital_h6_common_held_static_r01',
        maximum_Play_calls=1,maximum_Stop_reset_restaging_calls=0,maximum_route_trials=0,automatic_retry=False,
        actual_held_static_authorized=True,use_for_final_statistics=False,explicit_static_approval=protocol['user_approval'],
        held_static_contract=held_static_contract(),required_STOPPED_timeline=dict(retained.boundary.STOPPED),
        preparation_layers_file=str(layer_file),display_receipt=str(output/'display_initialization/renderer_initialization.json'),
        environment_rows_file=protocol['environment_rows_file'],geometry_file=protocol['geometry_file'],grid_file=protocol['grid_file'],
        query_contract=query,workflow_protocol=str(context.protocol_path),input_sha256=pins)
    if 'task_binding_path' in protocol:
        packet.update(scenario=protocol['scenario'],task_binding_path=protocol['task_binding_path'],
            static_query_contract_file=protocol['static_query_contract_file'],run_id='hospital_'+protocol['scenario'].split('_')[0].lower()+'_common_held_static_r01')
    retained.evidence.write_exclusive(static_path,packet)
    bound=retained.static_binding(packet,static_path,static_path.parent,context.preparation.LIVE)
    return bound,packet,static_path


def capture_fresh_workflow_failure(context,sdk,host=None):
    """Current non-mutating boundary in the SAME error file, not a new trial."""
    if sdk is None:return dict(SDK_not_bound=True,no_recovery_or_timeline_command=True)
    timeline=sdk.omni.timeline.get_timeline_interface();stage=sdk.omni.usd.get_context().get_stage()
    state=retained.evidence.timeline_state(timeline)
    result=dict(timeline=state,no_recovery_or_timeline_command=True)
    if stage is None:return dict(result,stage_missing=True)
    result['layer_texts']=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    try:
        tensor=None
        if host is not None:tensor=host.tensor
        elif not state['playing'] and not state['stopped']:
            existing=sdk.SimulationManager.get_physics_sim_view()
            if existing is not None:tensor=existing.create_articulation_view(context.protocol['retired_robot_root']+'/base_link')
        if tensor is not None and not state['playing']:
            first=retained.shared.preparation.old.tensor_snapshot(tensor);second=retained.shared.preparation.old.tensor_snapshot(tensor)
            result.update(tensor=second,stable_nonplaying_tensor=first==second and state==retained.evidence.timeline_state(timeline))
        else:result['no_retained_tensor_bound_no_fabricated_state']=True
    except Exception as failure:result['tensor_readback_error']=type(failure).__name__+': '+str(failure)
    return normalize_runtime_evidence(result)


def replay_held_static_evidence(result_path):
    """Read same saved CSV/query transactions; never repeat physics or rescore a HOLD.

    Recomputes load/standstill/fixed-window/query decisions. This is NOT an
    independent recreation of PhysX, all4997 live rows or an SDK-to-USD view.
    The saved live postflight and immutable own lineage remain separate proof.
    """
    import csv
    result_path=Path(result_path).resolve();report=retained.read(result_path);packet_path=Path(report['protocol'])
    from hospital_mission_sequence import TASK_SCENARIOS
    if (report['method']!='COMMON_HELD_STATIC_ENGINEERING_NOT_ROUTE' or report['scenario'] not in TASK_SCENARIOS.values()
            or report['protocol_sha256']!=retained.runtime.task.sha(packet_path)
            or report['contract']!=held_static_contract() or report['new_route_trials']!=0 or report['automatic_retry'] is not False):
        raise ValueError('Exact saved common held-static record required, not a route/other static or replacement')
    packet=retained.read(packet_path)
    # Replaying immutable evidence is not executing its old host. Resolve only
    # the already-declared evolving orchestration files from the SAME saved
    # source ZIP; original model/controller/data pins remain exact and live.
    try:
        retained.runtime.task.verify_pins(packet['input_sha256'])
        history=dict(verified_pins=len(packet['input_sha256']),resolved_consumed_sources=[])
    except ValueError:
        workflow_path=Path(packet['workflow_protocol']);workflow=retained.read(workflow_path)
        archive=workflow_path.parent/'workflow_sources.zip'
        verify_workflow_sources(archive,workflow['input_sha256'])
        history=reviewed_prior_GET_dependencies(packet['input_sha256'],archive)
    if packet['version']!=PreparedHeldStaticSDKHost.VERSION or result_path!=packet_path.parent/'held_static_readback.json':
        raise ValueError('Own declared first held-static packet/result required')
    prep_path=Path(report['preparation_record']);prep=retained.read(prep_path)
    if packet['input_sha256'].get(str(prep_path.resolve()))!=retained.runtime.task.sha(prep_path):
        raise ValueError('Own frozen fresh preparation required')
    rows=[]
    if report['sample_count']:
        trace=Path(report['trace']);schema=report['trace_schema']
        if trace!=result_path.with_name('held_static_trace.csv') or report['trace_sha256']!=retained.runtime.task.sha(trace):
            raise ValueError('Exact saved raw trace digest required')
        with trace.open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=list(schema) or len(reader.fieldnames)!=len(set(reader.fieldnames)):
                raise ValueError('Exact declared raw channel order required')
            for source in reader:
                if set(source)!=set(schema):raise ValueError('Malformed raw trace row')
                row={}
                for field,kind in schema.items():
                    value=source[field]
                    if kind in ('bool','optional_bool'):
                        if value not in (('True','False','') if kind=='optional_bool' else ('True','False')):
                            raise ValueError('Explicit raw Boolean/ACK channel required')
                        row[field]=None if value=='' else value=='True'
                    elif kind=='finite_number':
                        row[field]=float(value)
                        if not math.isfinite(row[field]):raise ValueError('Nonfinite saved raw channel')
                    elif kind=='string':row[field]=value
                    else:raise ValueError('Unknown saved raw channel type')
                rows.append(checked_measurement(row))
        if len(rows)!=report['sample_count']:raise ValueError('Raw sample count mismatch')
    if report['verdict']!='PASS':
        return dict(status='SAVED_HOLD_RETAINED_NOT_PROMOTED_TO_COMPLETED_STATIC',verdict=report['verdict'],
            raw_sample_count=len(rows),new_PhysX_trials=0,actual_route_admission=False,use_for_final_statistics=False)
    if not report['checks'] or not all(report['checks'].values()) or report.get('error') or report.get('postflight_error'):
        raise ValueError('Saved PASS with failed/incomplete checks is not an admissible completed static')
    replay=evaluate_held_static(rows,prep['start_xyyaw'],report['contact_counts'],report['callback_errors'],
        held_since_s=report['static']['held_since_s'],capture_count=report['fixed_hold_attempts'])
    if replay['checks']!=report['static']['checks'] or replay['verdict']!='PASS':raise ValueError('Original raw load/stop/window decision does not reproduce')
    post=report['postflight'];support=post['actual_sampled_cooked_support_queries'];absence=post['actual_disabled_subfloor_probes']
    positive=retained.support_evaluation(packet['query_contract'],support)
    negative=retained.subfloor_evaluation(packet['query_contract'],support,absence)
    if (positive['checks']!=post['support_query_evaluation']['checks'] or negative['checks']!=post['absence_probe_evaluation']['checks']
            or not all(positive['checks'].values()) or not all(negative['checks'].values())):
        raise ValueError('Raw27 support/four identity-query decisions do not reproduce')
    final=report['actual_final_readback'];state=report['actual_final_timeline']
    if (not final['stable_PAUSED_tensor'] or final['timeline']!=state or state['playing'] or state['stopped']
            or final['tensor']!=post['actual_post_held_static_tensor_snapshot']):
        raise ValueError('Own actual stable PAUSED final tensor record required')
    return normalize_runtime_evidence(dict(status='SAME_HELD_STATIC_RAW_CSV_AND_CACHED_QUERY_DECISIONS_REPRODUCED',verdict='PASS',
        raw_sample_count=len(rows),raw_static_evaluation=replay,raw_support_query_evaluation=positive,raw_absence_query_evaluation=negative,
        original_result=str(result_path),original_result_sha256=retained.runtime.task.sha(result_path),
        raw_CSV_replay_NOT_an_independent_live_environment_or_SDK_revalidation=True,
        historical_source_resolution=history,
        new_PhysX_trials=0,actual_route_admission=False,use_for_final_statistics=False))


async def run_fresh_held_workflow(path,*,authorized=False):
    """ONE authorized archive/Stop/new-source/FIRST-display/held-static only.

    No route, reset, environment patch, automatic retry or interrupted-phase
    resumption. A partial dispatch is always returned without another SDK call.
    """
    if authorized is not True:raise ValueError('Separate explicit fresh held-static workflow authority required before any file/SDK access')
    context=load_fresh_held_workflow(path);protocol=context.protocol;output=context.output
    reserve=output/'workflow_dispatch_reserved.json';result_path=output/'workflow_closeout.json';error_path=output/'workflow_error.json'
    touched=(reserve,result_path,error_path,output/'workflow_reserved.json',context.preparation.LIVE,
        output/'actual_prior_state_before_Stop.json',output/'actual_Stop_boundary.json')
    if any(p.exists() for p in touched):return dict(status='PRIOR_WORKFLOW_DISPATCH_RETAINED_NO_ADDITIONAL_STOP_OR_ACTION')
    retained.no_other_task()
    retained.evidence.write_exclusive(reserve,dict(protocol=str(context.protocol_path),protocol_sha256=retained.runtime.task.sha(context.protocol_path),
        explicit_approval=protocol['user_approval'],maximum_Stop_calls=1,maximum_static_trials=1,maximum_route_trials=0,automatic_retry=False))
    sdk=host=None;phase='actual_HOLD_archive_ONE_Stop_new_source_FIRST_display'
    try:
        sdk=retained.runtime.live_sdk()
        operation=fresh_held_preparation_operations(context,sdk,authorized=True)
        preparation=await operation(context)
        stage=sdk.omni.usd.get_context().get_stage()
        layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
        phase='freeze_exact_own_staged_source_held_static_binding'
        bound,packet,static_path=bind_prepared_held_packet(context,preparation,layers)
        g=retained.read(packet['geometry_file'])
        with np.load(packet['grid_file']) as grid:
            occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
        phase='ONE_first_held_static_NO_route'
        host=PreparedHeldStaticSDKHost(sdk,packet,static_path,context=bound,geometry=g,occupied=occupied,
            origin=origin,resolution=resolution,authorized=True)
        report=await run_held_static_once(host,HeldStaticController(protocol['start_xyyaw']),authorized=True)
        phase='same_trial_raw_static_CSV_query_replay_NO_new_physics'
        replay=replay_held_static_evidence(host.result);replay_path=host.result.parent/'raw_replay.json'
        retained.evidence.write_exclusive(replay_path,replay)
        result=dict(status='ONE_FRESH_COMMON_HELD_STATIC_WORKFLOW_FINISHED',new_robot_root=context.cohort.ROBOT,
            static_verdict=report['verdict'],static_result=str(host.result),static_result_sha256=retained.runtime.task.sha(host.result),
            actual_final_timeline=host.timeline_state(),ONE_Stop_completion_record=str(output/'actual_Stop_boundary.json'),
            raw_replay=str(replay_path),raw_replay_sha256=retained.runtime.task.sha(replay_path),
            route_trials=0,automatic_retry=False,original_HOLD_preserved=True,
            held_static_IS_NOT_H6_route_or_final_statistics=True,actual_route_admission=False)
        retained.evidence.write_exclusive(result_path,result)
        print('[H6 HELD WORKFLOW]',result['status'],'static:',result['static_verdict'],'; NO route/retry.',flush=True)
        return result
    except (Exception,asyncio.CancelledError) as failure:
        report=dict(status='HOLD_FRESH_HELD_WORKFLOW_ABORTED_NO_AUTOMATIC_RECOVERY',phase=phase,
            error=type(failure).__name__+': '+str(failure),original_HOLD_preserved=True,automatic_retry=False,
            route_trials=0,Stop_calls_NOT_inferred_from_missing_completion_receipt=True,
            Stop_completion_record_exists=(output/'actual_Stop_boundary.json').exists(),
            actual_route_admission=False,use_for_final_statistics=False)
        try:report['actual_failure_readback']=capture_fresh_workflow_failure(context,sdk,host)
        except Exception as readback_error:report['failure_readback_error']=type(readback_error).__name__+': '+str(readback_error)
        retained.evidence.write_exclusive(error_path,report)
        print('[H6 HELD WORKFLOW] Aborted in',phase,report['error'],'; saved:',error_path,'; NO retry/second Stop.',flush=True)
        return report


def schedule_fresh_held_workflow(path,*,authorized=False):
    """Shared Script Editor entry; exact approved packet, never a script clone."""
    import builtins
    if authorized is not True:raise ValueError('Explicit fresh held-static workflow authority required')
    for name in (*retained.boundary.ACTIVE_TASKS,'ijcas_h6_latest_postabort_inspection'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():raise RuntimeError('Another research task is active; no competing command')
    context=load_fresh_held_workflow(path)
    if any((context.output/name).exists() for name in ('workflow_dispatch_reserved.json','workflow_closeout.json','workflow_error.json')):
        print('[H6 HELD WORKFLOW] Prior dispatch retained; NO command repeated.',flush=True)
        return None
    future=asyncio.ensure_future(run_fresh_held_workflow(path,authorized=True))
    def finished(task):
        try:
            result=task.result()
            print('[H6 HELD WORKFLOW] Final:',{k:result[k] for k in ('status','static_verdict','phase','error','route_trials') if k in result},flush=True)
        except BaseException as failure:print('[H6 HELD WORKFLOW] ERROR:',type(failure).__name__+': '+str(failure),'; NO retry.',flush=True)
    future.add_done_callback(finished);builtins.ijcas_hospital_latest_task_experiment=future
    print('[H6 HELD WORKFLOW] ONE approved archive/Stop -> NEW source/FIRST display -> ONE held-static scheduled. '
        'ZERO forward; no route/reset/retry. Do not navigate camera or touch timeline controls.',flush=True)
    return future


def authorize_and_schedule_held_static_from_saved_result(result_path,*,cohort,
        approve_one_archive_Stop_new_source_and_held_static=False):
    """Operator command is the NEW explicit authorization, not an old answer.

    Default is read/write/SDK refusal. The supplied Script Editor instruction
    declares the complete scope, and the operator explicitly executes its
    named True flag. No authority is attributed to a past consumed approval.
    Packet export/claim still precede every live action, with no rerun.
    """
    import builtins
    if approve_one_archive_Stop_new_source_and_held_static is not True:
        raise ValueError('Explicit operator authorization for ONE named scope required before any file/SDK action')
    for name in (*retained.boundary.ACTIVE_TASKS,'ijcas_h6_latest_postabort_inspection'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():raise RuntimeError('Another research task is active; no competing command')
    if (retained.GATE/'hospital_h6_experiments'/cohort).exists():
        print('[H6 HELD WORKFLOW] Existing cohort/packet retained; NO command repeated.',flush=True)
        return None
    handoff=export_saved_held_abort_handoff(result_path)
    approval=dict(approval_source='USER_EXECUTED_SCRIPT_EDITOR_NAMED_SCOPE_FLAG',
        scope='ONE_ARCHIVE_STOP_NEW_SOURCE_HELD_STATIC_NO_ROUTE_NO_RETRY',explicit_flag=True)
    packet=export_fresh_held_workflow(cohort=cohort,approval=approval,prior_handoff=handoff)
    return schedule_fresh_held_workflow(packet,authorized=True)
