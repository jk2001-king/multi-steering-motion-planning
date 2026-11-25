"""Unchanged forward prefix for ONE separately approved speed-cap probe."""

import json
from pathlib import Path

from hospital_h3b_reverse_force_prefix import (
    POLICY, READINESS, ACCEPTANCE, BAD_CONTACTS, ROBOT_ROOT, WHEELS, GUARDS,
    TRACKING, COMMAND_LIMITS, TRACKING_GAINS, classify_contact, relative_pose,
    wheel_crossing_checks, motion_stalled, direct_joint_targets, component_goal,
    interpolation_table, reference_at_pose, FirstLegFeedback, body_velocity,
    evaluate_readiness, load_profile as baseline_profile,
)
from hospital_h3_flush_support import sha, verify_files
from hospital_h3_turn_planning import verify_sources
from hospital_h3b_reverse_speed50 import VERSION, verify_speed_only_change


WORKSPACE = Path(__file__).resolve().parents[3]
GATE = WORKSPACE/'studies/IJCAS/results/hospital_paper_gate'
COHORT = GATE/'hospital_h3b_reverse_speed50_pilot/cohort_01'
OUTPUT = COHORT/'prefix'
CONDITION = 'H3B_rear_reverse_prefix_reproduction_v1'
ROLE = 'UNCHANGED_FORWARD_PREFIX_FOR_SEPARATE_REVERSE_SPEED50_INTERVENTION'
PRE_MODEL = GATE/'hospital_h3b_reverse_speed50_model/review_r01/speed50_model_review.json'


def validate_model(record):
    expected = {(dt, residual) for dt in (.01, .02) for residual in
                ('frozen_nominal', 'observed_prior_neutral_offset_synthetic')}
    if (record.get('version') != VERSION or record.get('PhysX_executed') is not False
            or record.get('automatic_PhysX_execution_authorized') is not False
            or record.get('intervention_contract') != verify_speed_only_change()
            or not record.get('all_predeclared_model_cases_admitted', record.get('all_cases_admitted', False))
            or len(record.get('cases', [])) != 4
            or {(c['dt_s'], c['residual_case']) for c in record['cases']} != expected
            or any(not c['checks'] or not all(c['checks'].values()) for c in record['cases'])):
        raise ValueError('All four unchanged-guard speed50 MODEL cases required; not physical success')
    verify_sources(record); verify_files(record['input_sha256'])
    verify_files({c['trace_file']: c['trace_sha256'] for c in record['cases']})


def load_profile():
    info = baseline_profile(); model = json.loads(PRE_MODEL.read_bytes()); validate_model(model)
    if (model['original_mission_goal_xyyaw'] != info['mission_goal_xyyaw']
            or model['strict_map_digest'] != info['strict_map_digest']):
        raise ValueError('Original goal and strict map must remain unchanged')
    info.pop('one_instrumented_rear_repeat_preparation_only', None)
    info.update(input_sha256={**info['input_sha256'], **model['input_sha256'],
                             str(PRE_MODEL): sha(PRE_MODEL), str(Path(__file__)): sha(__file__)},
                separate_speed50_probe_preparation_only=True,
                forward_controller_path_and_acceptance_UNCHANGED=True,
                prefix_PASS_does_not_automatically_launch_reverse=True)
    return info
