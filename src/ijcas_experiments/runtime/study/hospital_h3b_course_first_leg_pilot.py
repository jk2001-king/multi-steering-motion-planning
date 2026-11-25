"""Separate shared-executor diagnostic; original first-leg FAIL is immutable.

Same fixed path/cusp/body model/P1/acceptance and original physical settings.
The disclosed change is body-yaw -> measured-course feedback while moving,
not new planning, gain fitting, reverse or a terminal correction manoeuvre.
"""

import json
from pathlib import Path

from hospital_h3b_response_first_leg_pilot import (POLICY, READINESS, ACCEPTANCE, BAD_CONTACTS,
    ROBOT_ROOT, WHEELS, GUARDS, TRACKING, COMMAND_LIMITS, TRACKING_GAINS,
    classify_contact, relative_pose, wheel_crossing_checks, motion_stalled,
    direct_joint_targets, component_goal, interpolation_table, reference_at_pose,
    load_profile as baseline_profile, evaluate_readiness as baseline_evaluate)
from hospital_h3b_course_tracking import CourseFirstLegFeedback as FirstLegFeedback, body_velocity, VERSION
from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources


CONDITION='H3B_shared_course_feedback_first_leg_v1'
ROLE='PROSPECTIVE_SHARED_COURSE_FEEDBACK_FIRST_LEG_NOT_PLANNER_OR_FULL_H3'
PROFILE=GATE/'hospital_h3b_course_tracking_reviews/review_r01/course_tracking_review.json'
OUTPUT=GATE/'hospital_h3b_course_first_leg_pilot/cohort_01'


def load_profile():
    info=baseline_profile()
    report=json.loads(PROFILE.read_bytes()); verify_sources(report); SUPPORT['verify_files'](report['input_sha256'])
    nominal=[r for r in report['model_cases'] if r['residual_case']=='frozen_nominal']
    if (report.get('version')!=VERSION
            or report.get('status')!='NOMINAL_SHARED_COURSE_COMPONENT_MODEL_ADMITTED_PhysX_PENDING'
            or report.get('nominal_component_model_admitted') is not True
            or report.get('PhysX_executed') is not False
            or report.get('automatic_PhysX_execution_authorized') is not False
            or report.get('original_measured_FAIL_remains_FAIL') is not True
            or len(nominal)!=6 or any(not r['all_model_checks_pass'] for r in nominal)
            or {(r['dt_s'],r['initial_case']) for r in nominal}!={
                (dt,initial) for dt in (.01,.02) for initial in report['declared_initial_cases']}
            or report['fixed_goal_local_xyyaw']!=info['first_stop_model_local_xyyaw']
            or report['strict_map_digest']!=info['strict_map_digest']
            or report['fixed_original_mission_start_xyyaw']!=info['mission_start_xyyaw']
            or report['fixed_original_mission_goal_xyyaw']!=info['mission_goal_xyyaw']
            or report['tracking_gains']!=TRACKING_GAINS):
        raise ValueError('Need complete nominal course-executor model admission on the SAME path/goal/map/gains')
    SUPPORT['verify_files']({c['trace_file']:c['trace_sha256'] for c in report['model_cases']})
    failure_audit=Path(report['original_first_leg_audit'])
    audit=json.loads(failure_audit.read_bytes());verify_sources(audit);SUPPORT['verify_files'](audit['input_sha256'])
    if audit['original_verdict_unchanged']!='FAIL':raise ValueError('Do not erase/relabel the original 34mm FAIL')
    info.update(profile=str(PROFILE),profile_sha256=SUPPORT['sha'](PROFILE),
        input_sha256={**info['input_sha256'],**report['input_sha256'],str(PROFILE):SUPPORT['sha'](PROFILE)},
        duration_s=next(c['model_duration_s']-5. for c in nominal if c['dt_s']==.01 and c['initial_case']=='model_origin'),
        synthetic_stress_checks_all_pass=report['all_synthetic_cases_pass'],
        original_failed_audit=str(failure_audit),controller_revision=VERSION,
        exact_old_yaw_only_tracking_law_unchanged=False,shared_course_feedback_revision_disclosed=True,
        future_comparison_requires_same_executor_for_ALL_methods_in_NEW_cohort=True)
    return info


def evaluate_readiness(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events):
    result=baseline_evaluate(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events)
    result.pop('first_forward_spatial_feedback_component_verified',None)
    result.update(course_feedback_first_forward_component_verified=result['verdict']=='PASS',
        changed_shared_executor_not_a_like_for_like_original_trial=True,
        original_yaw_only_first_leg_FAIL_preserved=True,
        full_H3B_turning_path_verified=False,reverse_turning_verified=False,use_for_final_statistics=False)
    return result
