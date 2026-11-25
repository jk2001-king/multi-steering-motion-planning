"""User-adopted supplementary query definition; separate retrospective review.

The actual original combined static result remains FAIL. Existing physical
data are rechecked under the adopted query definition, NOT called a new trial
or independent validation. This module cannot authorize Play or driving.
"""

import csv
import json
from pathlib import Path

import numpy as np

import hospital_h4_support_query_contract as contract


diagnosis, static, cohort = contract.diagnosis, contract.static, contract.cohort
OUTPUT = cohort.GATE/'hospital_h4_support_query_contract/adopted_r01'
ADOPTION = OUTPUT/'user_definition_adoption.json'
TARGET = OUTPUT/'recorded_static_and_live_query_reassessment.json'
VERSION = 'H4_adopted_same_26_dual_channel_query_separate_retrospective_review_v1'
APPROVAL_ANSWER = '새 보조 검사 정의 채택 (권장)'
REASSESSMENT_CHECKS = frozenset(('original_combined_static_FAIL_reproduces',
    'original_physical_stationary_limits_from_actual_trace_satisfied',
    'recorded_all_hit_queries_satisfy_adopted_definition',
    'retained_live_all_hit_queries_satisfy_adopted_definition',
    'actual_body_twelve_joints_gains_and_environment_continuity_verified',
    'same_original_obstacle_recorded_saved_and_live'))
LIVE_CHECKS = frozenset(('actual_failed_static_body_and_twelve_joint_continuity',
    'all_original_query_transactions_available', 'all_original_top_supports_recorded',
    'every_occluder_stays_an_original_mapped_blocked_obstacle',
    'same_failed_auxiliary_point_and_obstacle_retained', 'Root_Session_timeline_body_joints_gains_unchanged'))


def build_adoption():
    proposal = json.loads(contract.TARGET.read_bytes()); cohort.verify(proposal['input_sha256'])
    if proposal != contract.build():
        raise ValueError('The exact proposal shown to the user changed; adoption cannot substitute another definition')
    return dict(version=VERSION, status='USER_ADOPTED_SUPPLEMENTARY_QUERY_DEFINITION_ONLY',
        user_reply=APPROVAL_ANSWER, approval_scope='new supplementary query definition ONLY; no new physical trial or route',
        proposal=str(contract.TARGET), proposal_sha256=cohort.completion.sha(contract.TARGET),
        adoption_authorized=True, original_raw_static_stays_FAIL=True,
        original_26_queries_limits_physics_and_runtime_safety_unchanged=True,
        original_proposal_and_failed_result_NOT_overwritten=True,
        new_static_trial_authorized=False, actual_drive_admission=False,
        input_sha256={**proposal['input_sha256'], str(contract.TARGET): cohort.completion.sha(contract.TARGET),
            str(Path(__file__)): cohort.completion.sha(Path(__file__))})


def validate_adoption(record):
    required = dict(version=VERSION, status='USER_ADOPTED_SUPPLEMENTARY_QUERY_DEFINITION_ONLY',
        user_reply=APPROVAL_ANSWER, approval_scope='new supplementary query definition ONLY; no new physical trial or route',
        proposal=str(contract.TARGET), proposal_sha256=cohort.completion.sha(contract.TARGET),
        adoption_authorized=True, original_raw_static_stays_FAIL=True,
        original_26_queries_limits_physics_and_runtime_safety_unchanged=True,
        original_proposal_and_failed_result_NOT_overwritten=True,
        new_static_trial_authorized=False, actual_drive_admission=False)
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError('Adoption is definition-only; no implicit physics or drive authority')
    cohort.verify(record['input_sha256'])


def load_adoption():
    record = json.loads(ADOPTION.read_bytes()); validate_adoption(record)
    return record


def evaluate_observations(protocol, observations, reviewed):
    inventory = json.loads(cohort.support.floor.inventory.TARGET.read_bytes())
    with np.load(diagnosis.launch.cohort.MAP) as grid:
        return contract.evaluate_example(protocol['sampled_cooked_support_queries'], observations,
            protocol['support_top_m'], inventory['collision_prims'], grid['occupied'], grid['origin_xy_m'],
            float(grid['resolution_m']), {reviewed['original_existing_occluding_obstacle']: reviewed['sampled_envelope_screens']})


def build_reassessment():
    adoption = load_adoption()
    reviewed = diagnosis.load_review()
    if not diagnosis.LIVE.is_file():
        raise ValueError('Actual PAUSED same-state cached query review required; no Play to create it')
    live = json.loads(diagnosis.LIVE.read_bytes()); cohort.verify(live['input_sha256'])
    required = dict(version=diagnosis.VERSION, status='LIVE_SAME_FAILED_STATIC_QUERY_DIAGNOSIS_VERIFIED_NO_ADMISSION',
        actual_static_verdict='FAIL', actual_static_result=str(static.RESULT), static_relabelled_or_criteria_changed=False,
        original_obstacle_never_removed_filtered_or_exempted=True,
        additional_Play_Stop_Pause_reset_update_reopen_or_save=False, actual_drive_admission=False)
    if (any(live.get(key) != value for key, value in required.items()) or set(live.get('checks', {})) != LIVE_CHECKS
            or not all(live['checks'].values())):
        raise ValueError('Actual same-state live diagnostic incomplete or inconsistent; no failure waiver')
    prep = static.load_actual_preparation(); protocol = static.load_protocol()
    raw = json.loads(static.RESULT.read_bytes()); cohort.verify(raw['input_sha256'])
    if (raw['version'] != static.VERSION or raw['verdict'] != 'FAIL' or raw.get('error') or raw.get('evaluation_error')
            or raw['physics_preparation_snapshot'] != prep or raw['trace_sha256'] != cohort.completion.sha(static.TRACE)
            or {key for key, ok in raw['checks'].items() if not ok} != {
                'all_sampled_cooked_support_hits_at_original_top_and_upward_normal'}):
        raise ValueError('Only the exact supplementary-query original FAIL can be reassessed separately')
    with static.TRACE.open(newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    original = static.evaluate(rows, raw['start_xyyaw'], raw['contact_report_counts_by_category'], raw['contact_callback_errors'],
        protocol['sampled_cooked_support_queries'], raw['actual_sampled_cooked_support_queries'], protocol['support_top_m'])
    core = static.original.original.evaluate_static(rows, raw['start_xyyaw'], 265.,
        raw['contact_report_counts_by_category'], raw['contact_callback_errors'])
    saved_query = evaluate_observations(protocol, raw['actual_sampled_cooked_support_queries'], reviewed)
    live_query = evaluate_observations(protocol, live['actual_cached_queries'], reviewed)
    checks = dict(original_combined_static_FAIL_reproduces=original['verdict'] == 'FAIL'
            and all(raw['checks'].get(key) == value for key, value in original['checks'].items()),
        original_physical_stationary_limits_from_actual_trace_satisfied=core['verdict'] == 'PASS',
        recorded_all_hit_queries_satisfy_adopted_definition=saved_query['diagnostic_example_satisfies_proposed_query_contract'],
        retained_live_all_hit_queries_satisfy_adopted_definition=live_query['diagnostic_example_satisfies_proposed_query_contract'],
        actual_body_twelve_joints_gains_and_environment_continuity_verified=all(live['checks'].values()),
        same_original_obstacle_recorded_saved_and_live=(saved_query['original_obstacles_retained_and_recorded'] ==
            live_query['original_obstacles_retained_and_recorded'] == [reviewed['original_existing_occluding_obstacle']]))
    pins = {**adoption['input_sha256'], **raw['input_sha256'], **live['input_sha256'],
        **{str(path): cohort.completion.sha(path) for path in (ADOPTION, diagnosis.LIVE, static.RESULT, static.TRACE,
            static.PROTOCOL, cohort.LIVE, diagnosis.TARGET)}}
    cohort.verify(pins)
    return cohort.canonical(dict(version=VERSION,
        status='ADOPTED_QUERY_REASSESSMENT_SATISFIED_SEPARATE_DRIVE_AUTHORITY_REQUIRED' if all(checks.values()) else
            'HOLD_ADOPTED_QUERY_REASSESSMENT_INCOMPLETE', checks=checks,
        definition_adoption=str(ADOPTION), original_raw_static_result=str(static.RESULT), original_raw_static_verdict='FAIL',
        original_combined_evaluator_replay=original, original_physical_stationary_evaluator_replay=core,
        recorded_query_reassessment=saved_query, retained_live_query_reassessment=live_query,
        live_diagnostic=str(diagnosis.LIVE), actual_static_body_joint_continuity=live['live_static_continuity'],
        evidence_kind='RETROSPECTIVE_REVIEW_OF_EXISTING_ACTUAL_DATA_PLUS_GET_ONLY_LIVE_CONFIRMATION',
        original_result_relabelled=False, new_actual_static_trial_executed=False,
        prospective_independent_query_validation_claimed=False,
        no_statistical_or_physical_performance_result_created=True,
        next_route_if_separately_authorized='one disclosed development pilot; common original v3 controls and strict obstacle/full-envelope guards; no final comparison claim',
        fresh_final_comparison_requires_definition_frozen_before_new_trial_collection=True,
        new_static_trial_authorized=False, actual_drive_admission=False, use_for_final_statistics=False,
        CICS_runtime_integration=False, real_robot_required=False, input_sha256=pins))


def load_reassessment():
    report = json.loads(TARGET.read_bytes()); cohort.verify(report['input_sha256'])
    if (report['version'] != VERSION
            or report['status'] != 'ADOPTED_QUERY_REASSESSMENT_SATISFIED_SEPARATE_DRIVE_AUTHORITY_REQUIRED'
            or set(report.get('checks', {})) != REASSESSMENT_CHECKS
            or not all(report['checks'].values()) or report['original_raw_static_verdict'] != 'FAIL'
            or report['original_result_relabelled'] is not False or report['new_actual_static_trial_executed'] is not False
            or report['actual_drive_admission'] is not False or report['new_static_trial_authorized'] is not False):
        raise ValueError('Adopted support review missing/failed; old FAIL is never a raw PASS')
    load_adoption()
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-user-definition-adoption', action='store_true')
    args = parser.parse_args()
    if not args.record_user_definition_adoption:
        raise ValueError('Only the actual definition-only user reply may be recorded; no automatic adoption')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if ADOPTION.exists():
        load_adoption()
    else:
        cohort.save_exclusive(ADOPTION, build_adoption())
    if TARGET.exists():
        report = load_reassessment()
    else:
        report = build_reassessment(); cohort.save_exclusive(TARGET, report)
    print('=== H4 USER-ADOPTED QUERY DEFINITION / SEPARATE RECORDED-DATA REASSESSMENT ===')
    print('Original raw actual static: FAIL; unchanged')
    print('Physical stationary trace / saved and live adopted query checks:', report['checks'])
    print('STATUS:', report['status'])
    print('saved adoption:', ADOPTION); print('saved reassessment:', TARGET)
    print('No new static trial, retrospective raw PASS relabel, independent validation or drive authorization.')


if __name__ == '__main__':
    main()
