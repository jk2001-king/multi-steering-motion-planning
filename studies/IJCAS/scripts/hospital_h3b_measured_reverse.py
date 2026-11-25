"""ONE measured-state first reverse component, not full H3 execution.

NEW spatial-progress feedforward executor. No forward course-feedback law is
silently reused in reverse. Curvature is the frozen suffix's first reverse
leg; only progression/braking uses measured pose. Original terminal, contact,
stop, slew and torque limits remain. Captured hold only with neutral steering.
"""

import json
import math
from pathlib import Path

import numpy as np

from hospital_h3_flush_support import sha, verify_files
from hospital_h3_flush_crossing import (ROBOT_ROOT, WHEELS, BAD_CONTACTS, classify_contact,
    relative_pose, evaluate_passage)
from hospital_h3b_policy import POLICY, StopAlignGate
from hospital_path_execution import direct_joint_targets, ramp, wrap


WORKSPACE=Path(__file__).resolve().parents[3]
GATE=WORKSPACE/'studies/IJCAS/results/hospital_paper_gate'
LIVE=GATE/'hospital_h3b_measured_suffix_live_reviews/live_r01/live_suffix_review.json'
ADMISSION=GATE/'hospital_h3b_measured_reverse_model/review_r01/reverse_executor_review.json'
OUTPUT=GATE/'hospital_h3b_measured_reverse_pilot/cohort_01'
VERSION='H3B_measured_suffix_first_reverse_spatial_progress_v1'
ROLE='MEASURED_FIRST_REVERSE_COMPONENT_NOT_FULL_SUFFIX_OR_FINAL_COMPARISON'
TRACKING={'maximum_reference_position_error_m':.080,'maximum_reference_heading_error_deg':10.}
GUARDS={'maximum_simulation_duration_s':90.,'maximum_wall_time_s':300.}


def load_inputs():
    live=json.loads(LIVE.read_bytes())
    if (live['status']!='PAUSED_SAVED_STATE_CONTINUITY_AND_NEW_SUFFIX_OVERLAY_RECORDED_NO_DRIVE_AUTHORITY'
            or live.get('automatic_reverse_motion_authorized') is not False
            or live.get('physical_state_reset') is not False
            or live['position_difference_m']>.001 or abs(live['yaw_difference_deg'])>.1
            or live['maximum_joint_difference_rad']>.001 or live['measured_live_speed_m_s']>.005):
        raise ValueError('Actual PAUSED first-PASS live-state continuity required')
    verify_files({live[k]:live[k+'_sha256'] for k in ('first_result','suffix_candidate','model_review')})
    first=json.loads(Path(live['first_result']).read_bytes())
    suffix=json.loads(Path(live['suffix_candidate']).read_bytes())
    review=json.loads(Path(live['model_review']).read_bytes())
    for item in (suffix,review):verify_files(item['input_sha256'])
    verify_files(first['frozen_input_sha256'])
    if (first['verdict']!='PASS' or first['original_motor_gains_restored'] is not True
            or suffix['geometry_audit']['verdict']!='PASS' or not all(suffix['geometry_audit']['checks'].values())
            or not all(suffix['nominal_timing']['checks'].values())
            or review['all_first_reverse_model_cases_admitted'] is not True
            or sorted(c['dt_s'] for c in review['model_cases'])!=[.01,.02]
            or any(not all(c['checks'].values()) for c in review['model_cases'])
            or suffix['original_mission_goal_xyyaw']!=first['mission_goal_world_xyyaw_UNCHANGED']
            or suffix['strict_map_digest']!=first['strict_flush_map_digest']):
        raise ValueError('Frozen initial PASS/NEW suffix/original goal and map lineage changed')
    for name,digest in first['source_sha256'].items():
        folder=WORKSPACE/('isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
        verify_files({folder/name:digest})
    for c in review['model_cases']:verify_files({c['trace_file']:c['trace_sha256']})
    verify_files({WORKSPACE/'isaac/scripts/isaac_show_ijcas_hospital_h3b_measured_suffix.py':live['script_sha256']})
    g=suffix['geometry_audit'];end=suffix['nominal_timing']['cusp_sample_indices'][0]
    if (any(d!=-1 for d in g['interval_directions'][:end])
            or any(m!='front_only' for m in g['interval_modes'][:end])
            or not np.allclose(g['poses_local_xyyaw'][0],suffix['planning_initial_in_ORIGINAL_mission_frame'],rtol=0,atol=1e-12)):
        raise ValueError('Only complete FIRST reverse leg of the admitted suffix is authorized')
    inputs={**suffix['input_sha256'],**review['input_sha256'],str(LIVE):sha(LIVE),
        **{live[k]:live[k+'_sha256'] for k in ('first_result','suffix_candidate','model_review')}}
    info=dict(version=VERSION,live_receipt=str(LIVE),live_receipt_sha256=sha(LIVE),first_record=first,
        suffix_candidate=live['suffix_candidate'],suffix_candidate_sha256=live['suffix_candidate_sha256'],
        model_review=live['model_review'],model_review_sha256=live['model_review_sha256'],
        input_sha256=inputs,mission_start_xyyaw=suffix['original_mission_start_xyyaw'],
        mission_goal_xyyaw=suffix['original_mission_goal_xyyaw'],strict_map_digest=suffix['strict_map_digest'],
        initial_world_xyyaw=first['final_world_xyyaw'],expected_saved_last_state=first['last_recorded_state'],
        retained_all_joint_positions_rad=live['retained_all_joint_positions_rad'],
        poses_local_xyyaw=g['poses_local_xyyaw'][:end+1],
        curvature_1_m=g['sample_curvature_1_m'][:end+1],
        center_arc_m=g['travelled_arc_m'][:end+1],next_cusp_local_xyyaw=g['poses_local_xyyaw'][end],
        nominal_reference_reverse_m=g['leg_lengths_m'][0],
        curvature_is_frozen_spatial_feedforward_NOT_reverse_course_or_yaw_feedback=True,
        actual_caster_state_never_assigned=True)
    FirstReverseReference(info)
    return info


def load_admitted_profile():
    info=load_inputs();record=json.loads(ADMISSION.read_bytes());verify_files(record['input_sha256'])
    if (record.get('version')!=VERSION or record.get('all_cases_admitted') is not True
            or record.get('PhysX_executed') is not False or record['live_receipt_sha256']!=info['live_receipt_sha256']
            or sorted(c['dt_s'] for c in record['cases'])!=[.01,.02]
            or any(not all(c['checks'].values()) for c in record['cases'])):
        raise ValueError('Both exact NEW reverse-executor MODEL cases must be admitted before a physical attempt')
    verify_files({c['trace_file']:c['trace_sha256'] for c in record['cases']})
    info.update(admission=str(ADMISSION),admission_sha256=sha(ADMISSION),
        input_sha256={**info['input_sha256'],**record['input_sha256'],str(ADMISSION):sha(ADMISSION)})
    return info


class FirstReverseReference:
    def __init__(self,info):
        self.poses=np.asarray(info['poses_local_xyyaw'],float);self.arc=np.asarray(info['center_arc_m'],float)
        self.k=np.asarray(info['curvature_1_m'],float);self.progress=0.
        if (self.poses.shape!=(len(self.arc),3) or self.k.shape!=self.arc.shape or len(self.arc)<3
                or not all(np.isfinite(v).all() for v in (self.poses,self.arc,self.k))
                or self.arc[0]!=0. or np.any(np.diff(self.arc)<=0) or not .05<self.arc[-1]<=.5
                or max(abs(self.k))>.7+1e-10 or abs(self.k[0])+abs(self.k[-1])>1e-10):
            raise ValueError('Finite bounded original-frame first-reverse spatial reference required')

    def at(self,s):
        return np.array([np.interp(s,self.arc,self.poses[:,i]) for i in range(3)])

    def project(self,pose):
        pose=np.asarray(pose,float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():raise ValueError('Finite measured pose required')
        a=self.poses[:-1,:2];d=np.diff(self.poses[:,:2],axis=0)
        t=np.clip(np.sum((pose[:2]-a)*d,axis=1)/np.sum(d*d,axis=1),0,1)
        projected=a+t[:,None]*d;distance=np.linalg.norm(pose[:2]-projected,axis=1)
        candidates=self.arc[:-1]+t*np.diff(self.arc)
        distance[candidates<self.progress-.005]=np.inf
        index=int(np.argmin(distance))
        if not np.isfinite(distance[index]):raise RuntimeError('Measured reference projection lost')
        self.progress=max(self.progress,float(candidates[index]))
        ref=self.at(self.progress)
        return ref,float(np.linalg.norm(pose[:2]-ref[:2])),float(np.interp(self.progress,self.arc,self.k))


class FirstReverseProgress:
    def __init__(self,info,started):
        if not math.isfinite(started):raise ValueError('Finite command clock required')
        self.reference=FirstReverseReference(info);self.previous_time=started
        self.speed=0.;self.phase='tracking';self.braking_started=None

    def step(self,timestamp,pose,measured_speed,actual_steering):
        if (not math.isfinite(timestamp) or not math.isfinite(measured_speed) or measured_speed<0
                or len(actual_steering)!=2 or not all(math.isfinite(v) for v in actual_steering)):
            raise ValueError('Finite actual speed/steering required')
        dt=timestamp-self.previous_time
        if not 0<=dt<=.25:raise RuntimeError('Existing 0.25s control-gap guard')
        self.previous_time=timestamp;ref,error,k=self.reference.project(pose)
        heading=math.degrees(wrap(pose[2]-ref[2]));remaining=self.reference.arc[-1]-self.reference.progress
        if error>.080 or abs(heading)>10.:raise RuntimeError('Existing curved-reference 80mm/10deg guard')
        if self.phase=='tracking' and remaining<=.005+self.speed*self.speed/(2*.020):
            self.phase='braking';self.braking_started=timestamp
        if self.phase=='braking' and timestamp-self.braking_started>8.:
            raise TimeoutError('Existing 8s measured braking/neutral guard')
        self.speed=ramp(self.speed,-.010 if self.phase=='tracking' else 0.,dt,.020)
        if self.phase=='braking':k=0.
        ready=(self.phase=='braking' and abs(self.speed)<1e-7 and measured_speed<=.005
            and max(abs(v) for v in actual_steering)<=math.radians(2.5))
        return dict(speed_m_s=self.speed,curvature_1_m=k,ready=ready,phase=self.phase,
            progress_m=self.reference.progress,remaining_m=remaining,path_error_m=error,
            path_heading_error_deg=heading,reference_local_xyyaw=ref.tolist())


def motion_stalled(rows):
    moving=[r for r in rows if r['phase']=='reverse_motion']
    if not moving:return False
    tail=[r for r in moving if r['time_s']>=moving[-1]['time_s']-5.]
    return (len(tail)>1 and tail[-1]['time_s']-tail[0]['time_s']>=4.95
        and all(r['command_speed_m_s']<=-.006 for r in tail)
        and max(r['reference_progress_m'] for r in tail)-min(r['reference_progress_m'] for r in tail)<.001)


def evaluate_component(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events):
    common=evaluate_passage(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count)
    excluded={'all_six_wheels_started_east_and_finished_west','ordinary_positive_ramp_within_30mm_s','passage_and_hold_completed'}
    checks={k:v for k,v in common['checks'].items() if k not in excluded}
    moving=[r for r in rows if r['phase']=='reverse_motion' and r['command_speed_m_s']<-.003]
    maximum_slew=0.
    for a,b in zip(rows,rows[1:]):
        dt=b['command_time_s']-a['command_time_s']
        if dt<0:raise ValueError('Command clock went backwards')
        for axle in ('front','rear'):
            change=abs(b['command_'+axle+'_rad']-a['command_'+axle+'_rad'])
            if dt==0 and change>1e-12:raise ValueError('Steering changed with no elapsed command time')
            maximum_slew=max(maximum_slew,change/dt if dt else 0.)
    gate=gate_events[0] if len(gate_events)==1 else {}
    checks.update(first_reverse_and_single_hold_completed=True,
        one_measured_stop_hold_neutral_release_before_reverse=len(gate_events)==1
            and gate.get('direction')==-1 and gate.get('captured_hold_acknowledged') is True
            and gate.get('hold_release_acknowledged') is True and gate.get('measured_speed_m_s',1.)<=.005
            and gate.get('steering_error_deg',180.)<=2.5,
        only_front_only_reverse_commands=all(r['command_mode']=='front_only' and -.010-1e-9<=r['command_speed_m_s']<=1e-9 for r in rows),
        measured_reverse_progress_over_50mm=bool(moving) and moving[-1]['reference_progress_m']-moving[0]['reference_progress_m']>.050,
        measured_center_distance_under_original_0p5m=sum(math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']) for a,b in zip(rows,rows[1:]))<=.5,
        held_commands_zero=all(abs(r['command_speed_m_s'])<1e-7 and max(abs(r['command_'+a+'_rad']) for a in ('front','rear'))<1e-7 for r in rows if r['position_hold_enabled']),
        held_actual_steering_neutral=all(max(abs(r[a+'_rad']) for a in ('front','rear'))<=math.radians(2.5) for r in rows if r['position_hold_enabled']),
        command_slew_under_8deg_s=maximum_slew<=math.radians(8.)+1e-5,
        existing_curved_reference_bounds=all(r['reference_position_error_m']<=.080 and abs(r['reference_heading_error_deg'])<=10. for r in rows))
    return dict(checks={k:bool(v) for k,v in checks.items()},verdict='PASS' if all(checks.values()) else 'FAIL',
        **{k:common[k] for k in ('terminal_position_error_m','terminal_heading_error_deg','final_world_xyyaw','fixed_terminal_hold_analysis','final_force_weight_ratio')},
        maximum_observed_command_slew_deg_s=math.degrees(maximum_slew),
        first_reverse_component_verified=all(checks.values()),full_H3B_turning_path_verified=False,
        full_suffix_execution_verified=False,Nav2_executed=False,CICS_input_used=False,use_for_final_statistics=False)
