"""Prospective fresh-cohort static gate, using the adopted dual-query rule.

Old physical limits, numerical query tolerances and zero-command controller
are unchanged. The query definition is frozen BEFORE this new trial. Old raw
FAILs remain FAILs; a new static PASS would not clear the known route-map gap.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hospital_h4_reopened_preparation as preparation
import hospital_h4_static_gate as old
import hospital_static_gate as physical
import hospital_h4_adopted_support_review as adopted


cohort = SimpleNamespace(**vars(preparation))
cohort.completion = preparation.original.completion
cohort.SETUP = preparation.original.SETUP
cohort.validate_new_robot = preparation.original.validate_new_robot
ROOT, GATE, ROBOT, MARKER = preparation.ROOT, preparation.GATE, preparation.ROBOT, preparation.MARKER
VERSION = 'H4_reopened_cohort_02_original_static_control_prospective_adopted_dual_query_v1'
OUTPUT = GATE/'hospital_h4_static_contact/cohort_02_reopened_r01'
RUN_ID = 'hospital_h4_reopened_static_r01'
RESULT, TRACE = OUTPUT/(RUN_ID+'.json'), OUTPUT/(RUN_ID+'.csv')
RESERVATION = OUTPUT/'single_attempt_reserved.json'
PROTOCOL = GATE/'hospital_h4_static_admission/cohort_02_reopened_r01/static_protocol.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_reopened_static_gate.py')
SETTLE_S, RECORD_S, PHYSICS_HZ = old.SETTLE_S, old.RECORD_S, old.PHYSICS_HZ
GUARDS, LIMITS, QUERY_LIMITS = old.GUARDS, old.LIMITS, old.QUERY_LIMITS
BAD_CONTACTS, WHEELS, SUPPORTS, KNOWN_CATEGORIES = old.BAD_CONTACTS, old.WHEELS, old.SUPPORTS, old.KNOWN_CATEGORIES
classify_contact, validate_environment, merge_evaluation = old.classify_contact, old.validate_environment, old.merge_evaluation


def prepare_protocol():
    prep = preparation.load_protocol()
    definition = adopted.load_adoption()
    baseline = old.load_protocol()
    paths = (Path(__file__), ENTRY, preparation.PREFLIGHT, adopted.ADOPTION, old.PROTOCOL,
        _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_completed_stop_static_gate.py'))
    return dict(version=VERSION, run_id=RUN_ID, cohort=preparation.COHORT,
        status='PROSPECTIVE_STATIC_PROTOCOL_ONLY_FRESH_ACTUAL_STAGING_AND_USER_PLAY_AUTHORITY_REQUIRED',
        required_actual_preparation=str(preparation.LIVE), new_robot_root=ROBOT,
        start_xyyaw=baseline['start_xyyaw'], goal_xyyaw=baseline['goal_xyyaw'],
        settle_s=SETTLE_S, record_s=RECORD_S, physics_hz=PHYSICS_HZ,
        unchanged_original_static_limits=LIMITS, guards=GUARDS, supports=list(SUPPORTS),
        additional_sampled_query_limits=QUERY_LIMITS,
        sampled_cooked_support_queries=baseline['sampled_cooked_support_queries'],
        support_top_m=baseline['support_top_m'],
        query_definition_adoption=str(adopted.ADOPTION), query_definition_frozen_before_new_trial=True,
        query_definition='same 26 all-hit rays: exact support presence AND separate mapped-obstacle/full-envelope exclusion; no hidden hit filtering',
        original_nearest_hit_rule_recorded_as_diagnostic_NOT_new_trial_acceptance=True,
        original_physical_static_evaluator_and_zero_command_control_unchanged=True,
        old_static_and_contact_aborted_path_FAILs_remain_immutable=True,
        required_actual_H3_or_H4_dynamic_state_restore=False,
        exactly_one_attempt_no_automatic_retry=True, explicit_user_new_static_authorization_required=True,
        automatic_Play_1s_settling_2s_recording_then_Pause_only_after_user_launch=True,
        all_unknown_disabled_legacy_or_unmapped_ray_hits_fail_closed=True,
        known_low_collider_gap_blocks_old_route_even_if_new_start_static_PASS=True,
        no_obstacle_contact_exception_or_collider_gain_material_change=True,
        actual_static_trial_executed=False, actual_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**prep['input_sha256'], **definition['input_sha256'], **baseline['input_sha256'],
            **{str(p): preparation.sha(p) for p in paths}})


def load_protocol():
    protocol = json.loads(PROTOCOL.read_bytes()); preparation.verify(protocol['input_sha256'])
    if protocol != prepare_protocol():
        raise ValueError('Prospective reopened static protocol changed; no Play')
    return protocol


def load_actual_preparation():
    if not preparation.LIVE.is_file():
        raise ValueError('Actual reopened Session staging receipt required; do NOT Play to create it')
    report = json.loads(preparation.LIVE.read_bytes()); preparation.verify(report['input_sha256'])
    if (report.get('version') != preparation.VERSION or report.get('cohort') != preparation.COHORT
            or report.get('status') != 'REOPENED_H4_NOMINAL_SESSION_STAGED_NEW_STATIC_PENDING'
            or report.get('preparation_protocol') != str(preparation.PREFLIGHT)
            or report.get('old_solver_contact_cache_or_physical_state_restored') is not False
            or report.get('Root_and_source_file_unchanged') is not True
            or report.get('actual_drive_admission') is not False
            or report.get('timeline') != dict(playing=False, stopped=True, current_time_s=0.)
            or report['start_xyyaw'] != load_protocol()['start_xyyaw']):
        raise ValueError('Exact successful reopened nominal staging required; no old receipt substitution')
    return report


def evaluate(rows, start, counts, errors, specifications, observations, top):
    result = physical.evaluate_static(rows, start, 265., counts, errors)
    original_physical_verdict = result['verdict']
    reviewed = adopted.diagnosis.load_review()
    query = adopted.evaluate_observations(dict(sampled_cooked_support_queries=specifications,
        support_top_m=top), observations, reviewed)
    result['checks'].update({'adopted_query_'+key: value for key, value in query['checks'].items()})
    result['checks']['only_known_contact_categories_recorded'] = not (set(counts)-KNOWN_CATEGORIES)
    result['verdict'] = 'PASS' if all(result['checks'].values()) else 'FAIL'
    result.update(physical_stationary_verdict=original_physical_verdict,
        prospective_adopted_query_observation=query,
        original_nearest_hit_diagnostic_checks_NOT_new_acceptance=old.query_checks(specifications, observations, top),
        known_old_route_still_NOT_admitted=True)
    return result


def passive_final_link_readback(view):
    """Post-Pause tensor GETs only; never substitutes for core static checks."""
    if view is None or view.count != 1 or view.max_links != 13:
        raise ValueError('One current thirteen-body H4 articulation required for optional readback')
    names = list(view.shared_metatype.link_names)
    value = view.get_link_transforms()
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    values = np.asarray(value, float)
    if values.shape != (1, 13, 7) or not np.isfinite(values).all() or len(names) != 13 or len(set(names)) != 13:
        raise ValueError('Complete finite current thirteen-link transforms required')
    if np.any(np.abs(np.linalg.norm(values[0, :, 3:], axis=1)-1.) > 1e-4):
        raise ValueError('Current link quaternion is not normalized XYZW')
    lookup = {name.rsplit('/', 1)[-1]: index for index, name in enumerate(names)}
    if not set(WHEELS.values()) <= set(lookup):
        raise ValueError('All six current wheel body names must be available')
    return dict(link_names=names, all_thirteen_link_world_transforms_xyzw=values[0].tolist(),
        wheel_body_transforms_xyzw={name: values[0, lookup[name]].tolist() for name in WHEELS.values()},
        phase='AFTER_THIS_NEW_STATIC_PAUSE_NOT_OLD_PATH_CONTACT_TIME',
        measured_cooked_shape_contact_offsets_or_historical_stall_forces=False,
        optional_observer_completion_is_NOT_static_or_route_success=True)


def main():
    report = prepare_protocol()
    if PROTOCOL.exists():
        load_protocol()
    else:
        preparation.write_exclusive(PROTOCOL, report)
    print('=== H4 REOPENED PROSPECTIVE STATIC PROTOCOL PREPARED (OFFLINE) ===')
    print('same physical limits/controller/rays/tolerances; adopted query definition fixed BEFORE trial')
    print('saved:', PROTOCOL)
    print('No actual staging claim, static Play authority, drive authority or old FAIL relabeling.')


if __name__ == '__main__':
    main()
