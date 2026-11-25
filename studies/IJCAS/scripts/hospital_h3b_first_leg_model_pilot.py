"""Prospective first FORWARD leg model check, not whole H3 execution.

Uses a frozen model-admitted command profile and fixed predicted first-stop
target. No actual pose recapture, path feedback, reverse, caster reset or CICS.
The straight-S 30 mm / 2 deg corridor is NOT applicable to a turning task;
inherit H1's 80 mm / 10 deg reference-tracking guard, retaining the SAME
30 mm / 2 deg TERMINAL acceptance and strict actual full-envelope raster guard.
"""

import json
import math
from pathlib import Path

import numpy as np

from hospital_h3b_mode_speed_review import (POLICY, READINESS, ACCEPTANCE, BAD_CONTACTS,
    ROBOT_ROOT,WHEELS,GUARDS,classify_contact,relative_pose,wheel_crossing_checks,
    motion_stalled,direct_joint_targets)
from hospital_h3_turn_planning import GATE,SUPPORT,WORKSPACE,verify_sources,load_setup,model_bounds
from hospital_h3b_mode_speed_evidence import read_trace
from hospital_h3_flush_crossing import evaluate_passage
from hospital_path_execution import ramp,wrap


CONDITION='H3B_first_forward_leg_model_validation_v1'
ROLE='PROSPECTIVE_FIRST_FORWARD_LEG_MODEL_VALIDATION_NOT_FULL_H3_OR_REVERSE'
PROFILE=GATE/'hospital_h3b_model_profile_refinement/refinement_r01/refinement.json'
OUTPUT=GATE/'hospital_h3b_first_leg_model_pilot/cohort_01'
TRACKING={'maximum_reference_position_error_m':.080,'maximum_reference_heading_error_deg':10.}
COMMAND_LIMITS={'forward_speed_m_s':.050,'reverse_speed_m_s':.010,'acceleration_m_s2':.020,
    'steering_command_slew_deg_s':8.,'curvature_limit_1_m':.7}


def load_profile():
    record=json.loads(PROFILE.read_bytes());verify_sources(record)
    SUPPORT['verify_files'](record['input_sha256'])
    if (record.get('model_profile_admitted_at_both_resolutions') is not True
            or record.get('PhysX_executed') is not False or record.get('full_dynamics_plan_certified') is not False
            or record.get('P_full_dynamics_plan_selected') is not False or record['model_parameters']!=model_bounds()):
        raise ValueError('Need frozen independently checked MODEL profile, not a whole-route execution PASS')
    baseline_path=Path(record['baseline_diagnostic']);baseline=json.loads(baseline_path.read_bytes());verify_sources(baseline)
    SUPPORT['verify_files']({baseline_path:record['baseline_sha256']})
    SUPPORT['verify_files'](baseline['input_sha256'])
    SUPPORT['verify_files']({c['trace_file']:c['trace_sha256'] for c in baseline['cases']})
    SUPPORT['verify_files']({c['trace_file']:c['trace_sha256'] for c in record['model_cases']})
    if sorted(c['dt_s'] for c in record['model_cases'])!=[.01,.02] or any(not all(c['model_checks'].values()) for c in record['model_cases']):
        raise ValueError('Both predeclared MODEL resolutions must be admitted')
    fine=next(c for c in record['model_cases'] if c['dt_s']==.01)
    rows=read_trace(fine['trace_file'])
    prefix=[]
    for row in rows:
        if row['leg_index']!=0:break
        prefix.append(row)
    event=fine['model_stop_events'][0]
    if (not prefix or prefix[-1]['time_s']!=event['model_ready_time_s']
            or event['body_velocity_or_steering_reset'] or event['model_horizontal_speed_m_s']>.005
            or event['model_active_steering_error_deg']>2.5
            or any(row['command_mode']!='coordinated_counterphase' or not 0<=row['command_speed_m_s']<=.05+1e-9 for row in prefix)
            or not np.allclose([prefix[-1][k] for k in ('x_m','y_m','yaw_rad')],event['model_pose_after_settle'],rtol=0,atol=1e-12)):
        raise ValueError('Expected a bounded counterphase FORWARD first leg ending at the frozen model stop')
    audit_path=Path(record['spatial_audit']);audit=json.loads(audit_path.read_bytes());verify_sources(audit)
    SUPPORT['verify_files'](audit['input_sha256'])
    manifest,_,_,_,_=load_setup(audit['geometry_manifest'],audit['preview_receipt'])
    if (manifest['start_xyyaw']!=record['fixed_start_xyyaw'] or manifest['goal_xyyaw']!=record['fixed_goal_xyyaw']
            or manifest['planning_occupancy_digest']!=record['strict_map_digest']):
        raise ValueError('Original H3 task/map changed')
    return {'profile':str(PROFILE),'profile_sha256':SUPPORT['sha'](PROFILE),
        'input_sha256':{**record['input_sha256'],str(PROFILE):SUPPORT['sha'](PROFILE),
            **{c['trace_file']:c['trace_sha256'] for c in record['model_cases']}},
        'mode':'coordinated_counterphase','duration_s':prefix[-1]['time_s'],
        'first_stop_model_local_xyyaw':event['model_pose_after_settle'],
        'model_prefix_rows':prefix,'maximum_reference_x_m':max(r['x_m'] for r in prefix),
        'mission_start_xyyaw':manifest['start_xyyaw'],'mission_goal_xyyaw':manifest['goal_xyyaw'],
        'strict_map_digest':record['strict_map_digest'],'model_admission_is_not_PhysX_success':True}


def interpolation_table(info):
    if '_interpolation_table' not in info:
        keys=('time_s','command_speed_m_s','requested_curvature_1_m','x_m','y_m','yaw_rad')
        table={key:np.array([r[key] for r in info['model_prefix_rows']],float) for key in keys}
        if (not len(table['time_s']) or any(not np.isfinite(v).all() for v in table.values())
                or table['time_s'][0]!=0. or np.any(np.diff(table['time_s'])<=0)
                or table['time_s'][-1]!=info['duration_s']
                or np.any(table['command_speed_m_s']<0) or np.any(table['command_speed_m_s']>.05+1e-9)
                or np.max(np.abs(table['requested_curvature_1_m']))>.7+1e-9):
            raise ValueError('Finite bounded first-leg time series required')
        info['_interpolation_table']=table
    return info['_interpolation_table']


def command_at(info,elapsed):
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Finite elapsed model-reference time required')
    table=interpolation_table(info)
    if elapsed>=info['duration_s']:return 0.,0.
    return tuple(float(np.interp(elapsed,table['time_s'],table[key])) for key in ('command_speed_m_s','requested_curvature_1_m'))


def reference_at(info,elapsed):
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Finite elapsed model-reference time required')
    table=interpolation_table(info)
    return tuple(float(np.interp(elapsed,table['time_s'],table[key])) for key in ('x_m','y_m','yaw_rad'))


def component_goal(start,info):
    x,y,yaw=info['first_stop_model_local_xyyaw'];c,s=math.cos(start[2]),math.sin(start[2])
    return start[0]+c*x-s*y,start[1]+s*x+c*y,wrap(start[2]+yaw)


class ForwardReplayProfile:
    def __init__(self,info,started):
        if not math.isfinite(started):raise ValueError('Finite motion start time required')
        self.info,self.started,self.previous_time=info,started,started
        self.command=0.;self.braking=False

    def step(self,timestamp,measured_speed,measured_steer):
        if not all(math.isfinite(v) for v in (timestamp,measured_speed,*measured_steer)) or measured_speed<0 or len(measured_steer)!=2:
            raise ValueError('Finite actual body/steering observations required')
        dt=timestamp-self.previous_time
        if not 0<=dt<=.25:raise RuntimeError('Existing 0.25 s control-gap guard')
        self.previous_time=timestamp;elapsed=timestamp-self.started
        target,_=command_at(self.info,elapsed)
        self.command=ramp(self.command,target,dt,.02)
        self.braking=elapsed>=self.info['duration_s']
        if elapsed>self.info['duration_s']+POLICY['braking_timeout_s']:
            raise TimeoutError('First-leg measured stop/neutral timeout; no reset or held steering fallback')
        ready=(self.braking and abs(self.command)<1e-7 and measured_speed<=.005
            and max(abs(v) for v in measured_steer)<=math.radians(2.5))
        return self.command,ready


def evaluate_readiness(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events):
    common=evaluate_passage(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count)
    excluded={'all_six_wheels_started_east_and_finished_west','ordinary_positive_ramp_within_30mm_s','passage_and_hold_completed'}
    checks={k:v for k,v in common['checks'].items() if k not in excluded}
    values=('front_rad','rear_rad','command_front_rad','command_rear_rad','command_time_s',
        'model_reference_position_error_m','model_reference_heading_error_deg')
    if any(not math.isfinite(float(r[k])) for r in rows for k in values):raise ValueError('Finite actual/reference telemetry required')
    moving=[r for r in rows if r['phase'].startswith('leg_0_') and r['command_speed_m_s']>.003]
    slew=[]
    for a,b in zip(rows,rows[1:]):
        dt=b['command_time_s']-a['command_time_s']
        if dt<0:raise ValueError('Command clock went backwards')
        for axle in ('front','rear'):
            delta=abs(b['command_'+axle+'_rad']-a['command_'+axle+'_rad'])
            if dt==0 and delta>1e-12:raise ValueError('Steering changed without elapsed time')
            slew.append(delta/dt if dt else 0.)
    checks.update(first_forward_leg_and_single_hold_completed=True,
        no_reverse_or_cusp_execution=not gate_events and all(not r['phase'].startswith(('leg_1_','cusp_')) for r in rows),
        actual_forward_displacement_over_one_m=bool(moving) and moving[-1]['longitudinal_m']-moving[0]['longitudinal_m']>1.,
        both_active_steering_responses_observed=bool(moving) and all(max(abs(r[a+'_rad']) for r in moving)>math.radians(5.) for a in ('front','rear')),
        selected_counterphase_forward_commands=all(r['command_mode']=='coordinated_counterphase' and 0<=r['command_speed_m_s']<=.05+1e-9 for r in rows),
        command_steering_slew_within_8deg_s=max(slew,default=0.)<=math.radians(8.)+1e-5,
        model_reference_tracking_within_existing_H1_bounds=all(r['model_reference_position_error_m']<=.080
            and r['model_reference_position_error_m']>=0 and abs(r['model_reference_heading_error_deg'])<=10. for r in rows),
        all_hold_commands_zero=all(abs(r['command_speed_m_s'])<1e-7
            and max(abs(r['command_'+a+'_rad']) for a in ('front','rear'))<1e-7 for r in rows if r['position_hold_enabled']),
        all_held_actual_steering_neutral=all(max(abs(r[a+'_rad']) for a in ('front','rear'))<=math.radians(2.5)
            for r in rows if r['position_hold_enabled']))
    return {'checks':{k:bool(v) for k,v in checks.items()},'verdict':'PASS' if all(checks.values()) else 'FAIL',
        **{k:common[k] for k in ('terminal_position_error_m','terminal_heading_error_deg','final_world_xyyaw',
            'fixed_terminal_hold_analysis','final_force_weight_ratio')},
        'first_forward_leg_model_pilot_verified':all(checks.values()),'reverse_turning_verified':False,
        'full_H3B_turning_path_verified':False,'full_dynamics_plan_certified':False,'use_for_final_statistics':False}
