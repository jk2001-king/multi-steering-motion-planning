"""Unchanged forward reproduction before ONE NEW terminal-executor trial."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
from pathlib import Path

from hospital_h3b_reverse_speed50_prefix import (
    POLICY, READINESS, ACCEPTANCE, BAD_CONTACTS, ROBOT_ROOT, WHEELS, GUARDS,
    TRACKING, COMMAND_LIMITS, TRACKING_GAINS, classify_contact, relative_pose,
    wheel_crossing_checks, motion_stalled, direct_joint_targets, component_goal,
    interpolation_table, reference_at_pose, FirstLegFeedback, body_velocity,
    evaluate_readiness, load_profile as baseline_profile,
)
from hospital_h3_flush_support import sha, verify_files
from hospital_h3_turn_planning import verify_sources
from hospital_h3b_reverse_closedloop import VERSION, execution_contract


WORKSPACE = _ijcas_workspace()
GATE = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
COHORT = GATE/'hospital_h3b_reverse_closedloop_pilot/cohort_01'
OUTPUT = COHORT/'prefix'
CONDITION = 'H3B_rear_reverse_prefix_reproduction_v1'
ROLE = 'UNCHANGED_FORWARD_PREFIX_FOR_NEW_MEASURED_TERMINAL_EXECUTOR'
PRE_MODEL = GATE/'hospital_h3b_reverse_closedloop_model/review_r02/closedloop_model_review.json'


def validate_model(record):
    expected = {(dt, residual, scale) for dt in (.01, .02)
                for residual in ('frozen_nominal', 'observed_prior_neutral_offset_synthetic') for scale in (1., .6)}
    if (record.get('version') != VERSION or record.get('PhysX_executed') is not False
            or record.get('automatic_PhysX_execution_authorized') is not False
            or record.get('intervention_contract') != execution_contract()
            or record.get('all_cases_admitted') is not True or len(record.get('cases', [])) != 8
            or {(c['dt_s'], c['residual_case'], c['synthetic_body_target_response_scale']) for c in record['cases']} != expected
            or any(not c['checks'] or not all(c['checks'].values()) for c in record['cases'])):
        raise ValueError('All eight predeclared NEW terminal MODEL cases required; no PhysX success implied')
    verify_sources(record); verify_files(record['input_sha256'])
    verify_files({c['trace_file']:c['trace_sha256'] for c in record['cases']})


def load_profile():
    info = baseline_profile(); model = json.loads(PRE_MODEL.read_bytes()); validate_model(model)
    if (model['original_mission_goal_xyyaw'] != info['mission_goal_xyyaw']
            or model['strict_map_digest'] != info['strict_map_digest']):
        raise ValueError('Original fixed goal and strict map changed')
    info.pop('separate_speed50_probe_preparation_only', None)
    info.update(input_sha256={**info['input_sha256'], **model['input_sha256'],
                             str(PRE_MODEL):sha(PRE_MODEL), str(Path(__file__)):sha(__file__)},
                NEW_measured_terminal_executor_preparation_only=True,
                forward_controller_path_and_acceptance_UNCHANGED=True,
                prefix_PASS_does_not_automatically_launch_reverse=True)
    return info
