"""Fresh physical reproduction of the SAME successful forward prefix.

Separate cohort/output and purpose only. Controller, path, nominal first cusp,
gains, physical settings and acceptance are inherited unchanged. Never reset a
paused robot to the old actual endpoint, or treat this prefix as rear success.
"""

import json
from pathlib import Path

from hospital_h3b_course_first_leg_pilot import (POLICY,READINESS,ACCEPTANCE,BAD_CONTACTS,
    ROBOT_ROOT,WHEELS,GUARDS,TRACKING,COMMAND_LIMITS,TRACKING_GAINS,
    classify_contact,relative_pose,wheel_crossing_checks,motion_stalled,direct_joint_targets,
    component_goal,interpolation_table,reference_at_pose,FirstLegFeedback,body_velocity,
    load_profile as original_profile,evaluate_readiness as original_evaluate)
from hospital_h3_flush_support import sha,verify_files


WORKSPACE=Path(__file__).resolve().parents[3]
GATE=WORKSPACE/'studies/IJCAS/results/hospital_paper_gate'
OUTPUT=GATE/'hospital_h3b_rear_prefix_pilot/cohort_01'
CONDITION='H3B_rear_reverse_prefix_reproduction_v1'
ROLE='FRESH_IDENTICAL_COURSE_FIRST_FORWARD_PREFIX_ONLY_FOR_REAR_REVERSE_REVIEW'
MODEL_REVIEW=GATE/'hospital_h3b_rear_reverse_model/review_r01/rear_reverse_model_review.json'
ORIGINAL_FIRST=GATE/'hospital_h3b_course_first_leg_pilot/cohort_01/hospital_h3b_course_first_leg_r01.json'


def load_profile():
    info=original_profile();first=json.loads(ORIGINAL_FIRST.read_bytes())
    if first.get('verdict')!='PASS' or first.get('original_motor_gains_restored') is not True:
        raise ValueError('The original fixed-course first-forward PASS must remain preserved')
    for name,digest in first['source_sha256'].items():
        folder=WORKSPACE/('isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
        verify_files({folder/name:digest})
    verify_files(first['frozen_input_sha256'])
    model=json.loads(MODEL_REVIEW.read_bytes());verify_files(model['input_sha256'])
    if (model.get('version')!='H3B_saved_cusp_rear_reverse_existing_executor_model_screen_v1'
            or model.get('all_predeclared_model_cases_admitted') is not True or model.get('PhysX_executed') is not False
            or model.get('automatic_PhysX_execution_authorized') is not False
            or model['original_mission_goal_xyyaw']!=info['mission_goal_xyyaw']
            or model['strict_map_digest']!=info['strict_map_digest'] or len(model['cases'])!=4
            or {(c['dt_s'],c['residual_case']) for c in model['cases']}!={
                (dt,residual) for dt in (.01,.02) for residual in ('frozen_nominal','observed_prior_neutral_offset_synthetic')}
            or any(not all(c['checks'].values()) for c in model['cases'])):
        raise ValueError('All four rear model screens must be retained; none is a physical rear PASS')
    verify_files({c['trace_file']:c['trace_sha256'] for c in model['cases']})
    info.update(input_sha256={**info['input_sha256'],**model['input_sha256'],
        str(MODEL_REVIEW):sha(MODEL_REVIEW),str(ORIGINAL_FIRST):sha(ORIGINAL_FIRST),
        str(Path(__file__)):sha(__file__)},
        fresh_prefix_reproduction_for_rear_review=True,
        original_first_PASS_reference=str(ORIGINAL_FIRST),original_first_PASS_sha256=sha(ORIGINAL_FIRST),
        rear_model_screen=str(MODEL_REVIEW),rear_model_screen_sha256=sha(MODEL_REVIEW),
        actual_new_prefix_endpoint_requires_review_before_reverse=True,
        fixed_prefix_goal_NOT_recaptured_old_actual_endpoint=True,
        forward_controller_path_and_acceptance_UNCHANGED=True,
        rear_reverse_execution_authorized=False)
    return info


def evaluate_readiness(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events):
    result=original_evaluate(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events)
    result.update(fresh_first_forward_prefix_reproduced=result['verdict']=='PASS',
        rear_reverse_verified=False,rear_reverse_execution_authorized=False,
        actual_new_prefix_endpoint_requires_review_before_reverse=True)
    return result
