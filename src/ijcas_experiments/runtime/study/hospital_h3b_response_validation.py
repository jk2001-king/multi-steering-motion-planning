"""Prospective MIRRORED small-angle response validation, not H3 execution.

Primary fit frozen BEFORE this trial; it NEVER changes commands/physical gains.
The motion adapter retains the existing .65m ramp/stop/strict-envelope guards.
Only sign of the original S excitation is reversed, with its fixed nominal
endpoint mirrored explicitly. Original H3 mission start/goal remain unchanged.
"""

import json
import math

import numpy as np

from hospital_h3b_mode_speed_review import (POLICY,READINESS,ACCEPTANCE,BAD_CONTACTS,
    ROBOT_ROOT,WHEELS,GUARDS,COMMAND_LIMITS,DISTANCE,LEGS,ForwardStopProfile,
    classify_contact,relative_pose,wheel_crossing_checks,motion_stalled,direct_joint_targets,
    KNOTS as ORIGINAL_KNOTS,curvature_at as original_curvature,nominal_component as original_nominal,
    evaluate_readiness as original_evaluate)
from hospital_h3_turn_planning import GATE,SUPPORT,verify_sources
from hospital_steering_response_identification import PRIMARY,FAMILIES,VALIDATION_CRITERIA,prepare_trace,predict,error_metrics


VERSION=CONDITION='H3B_frozen_response_mirrored_S_validation_v1'
ROLE='PROSPECTIVE_FROZEN_MODEL_RESPONSE_VALIDATION_NOT_H3_OR_METHOD_COMPARISON'
CASE='coordinated_counterphase_50mm_s'
KNOTS=tuple((x,-k) for x,k in ORIGINAL_KNOTS)
MODEL=GATE/'hospital_h3b_steering_response_identification/fit_r01/identified_response.json'
OUTPUT=GATE/'hospital_h3b_steering_response_validation/cohort_01'


def load_frozen_fit():
    record=json.loads(MODEL.read_bytes());verify_sources(record);SUPPORT['verify_files'](record['input_sha256'])
    SUPPORT['verify_files']({record['training_prediction_trace']:record['training_prediction_trace_sha256']})
    if (record.get('status')!='FROZEN_SEPARATE_RESPONSE_CANDIDATE_PROSPECTIVE_MIRRORED_S_VALIDATION_PENDING'
            or record['primary_family']!=PRIMARY or record.get('new_PhysX_validation_performed') is not False
            or record.get('new_model_applied_to_planner_or_drive_commands') is not False
            or record['prospective_validation_condition']!=CONDITION
            or record['prospective_knots']!=[list(v) for v in KNOTS]
            or not all(record['prospective_geometry_and_rate_preflight'].values())
            or record['fit']['prospective_validation_criteria']!=VALIDATION_CRITERIA):
        raise ValueError('Expected frozen primary response + mirrored-S protocol; never refit from prospective data')
    return record


def require_case(case):
    if case!=CASE:raise ValueError('Only the predeclared mirrored counterphase 50mm_s validation case is permitted')
    return 'coordinated_counterphase',.050


def curvature_at(progress,phase):return -original_curvature(progress,phase)


def nominal_component(mode,step=.001):
    if mode!='coordinated_counterphase':raise ValueError('Mirrored validation is counterphase only')
    return [[x,-y,-yaw] for x,y,yaw in original_nominal(mode,step)]


def component_goal(start,mode):
    x,y,yaw=nominal_component(mode)[-1];c,s=math.cos(start[2]),math.sin(start[2])
    return start[0]+c*x-s*y,start[1]+s*x+c*y,start[2]+yaw


def evaluate_readiness(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,events,case):
    require_case(case)
    result=original_evaluate(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,events,CASE)
    result.pop('mode_speed_component_verified',None)
    result.update(mirrored_response_excitation_completed=result['verdict']=='PASS',
        frozen_response_prediction_verified=False,response_validation_pending_external_analysis=True,
        use_for_final_statistics=False)
    return result


def evaluate_predictions(rows,frozen):
    time,command,actual,_,moving=prepare_trace(rows)
    if frozen['primary_family']!=PRIMARY:raise ValueError('Primary candidate changed after freeze')
    metrics={};checks={};predictions={}
    for j,axle in enumerate(('front','rear')):
        metrics[axle]={};predictions[axle]={}
        for family in FAMILIES:
            parameters=frozen['fit']['axles'][axle]['models'][family]['parameters']
            predicted=predict(time,command[:,j],actual[0,j],family,parameters)
            predictions[axle][family]=predicted
            metrics[axle][family]={'all_samples':error_metrics(actual[:,j],predicted),
                'moving_samples':error_metrics(actual[:,j],predicted,moving)}
        primary=metrics[axle][PRIMARY]['moving_samples'];old=metrics[axle]['legacy_minus1deg_slew']['moving_samples']
        ratio=primary['rmse_deg']/max(old['rmse_deg'],1e-12)
        checks.update({axle+'_moving_rmse_under_1deg':primary['rmse_deg']<=VALIDATION_CRITERIA['moving_rmse_deg_max'],
            axle+'_moving_p95_under_2p5deg':primary['p95_absolute_error_deg']<=VALIDATION_CRITERIA['moving_p95_absolute_error_deg_max'],
            axle+'_moving_max_under_5deg':primary['max_absolute_error_deg']<=VALIDATION_CRITERIA['moving_max_absolute_error_deg_max'],
            axle+'_moving_rmse_at_most_80percent_legacy':ratio<=VALIDATION_CRITERIA['moving_rmse_ratio_to_legacy_max']})
        metrics[axle]['primary_rmse_ratio_to_legacy']=ratio
    return {'primary_family':PRIMARY,'criteria':VALIDATION_CRITERIA,'metrics':metrics,'checks':{k:bool(v) for k,v in checks.items()},
        'model_prediction_verdict':'PASS' if all(checks.values()) else 'FAIL',
        'future_validation_samples_used_to_refit_parameters':False,'observed_state_reset_after_first_sample':False,
        'conditional_on_recorded_applied_commands_NOT_full_navigation_prediction':True,
        'full_H3B_turning_path_verified':False,'large_angle_or_reverse_response_verified':False},predictions
