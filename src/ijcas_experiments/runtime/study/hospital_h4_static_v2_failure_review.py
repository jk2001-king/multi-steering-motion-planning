"""Audit the actual V2 finalization failure; NEVER relabel it or retry physics."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_subfloor_compact_static_gate_v2 as stationary
import hospital_runtime_evidence_io as evidence
import hospital_static_gate as physical


ROOT, GATE = stationary.ROOT, stationary.GATE
OUTPUT = GATE / 'hospital_h4_static_v2_failure_review/review_r01'
SAVED_REVIEW = OUTPUT / 'saved_data_review.json'
LIVE = OUTPUT / 'live_paused_closeout.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_closeout_ijcas_hospital_h4_static_v2_failure.py')
VERSION = 'H4_actual_static_v2_deferred_Pause_and_JSON_representation_failure_review_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_records():
    raw = stationary.preparation.read_small(stationary.RESULT)
    protocol = stationary.preparation.read_small(stationary.PROTOCOL)
    reserved = stationary.preparation.read_small(stationary.RESERVATION)
    stationary.preparation.verify(raw['input_sha256'])
    if (raw.get('version') != stationary.VERSION or raw.get('run_id') != stationary.RUN_ID
            or raw.get('static_protocol_sha256') != sha(stationary.PROTOCOL)
            or reserved.get('run_id') != stationary.RUN_ID
            or reserved.get('protocol_sha256') != sha(stationary.PROTOCOL)
            or reserved.get('actual_preparation_sha256') != sha(stationary.preparation.LIVE)
            or reserved.get('user_authorized') is not True
            or stationary.PREFLIGHT_HOLD.exists()):
        raise ValueError('Exact own actual V2 reservation/protocol/result required')
    if raw.get('trace_sha256') != sha(stationary.TRACE):
        raise ValueError('Actual static CSV hash mismatch')
    with stationary.TRACE.open(newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    return raw, protocol, rows


def analyze_records(raw, protocol, rows, result_bytes):
    """Separate observed physics, missed postconditions and representation.

    No missing query is fabricated from the visible floor or older cohort.
    No post-monitor interval is certified by this recorded-data audit.
    """
    if raw.get('verdict') != 'FAIL':
        raise ValueError('This review must preserve the actual V2 FAIL')
    if raw.get('checks', {}).get('stationary_observation_completed') is not True:
        raise ValueError('A completed recorded observation is required for this diagnosis')
    if raw.get('evaluation_error') != 'RuntimeError: Never perform full environment hashing during the physics observation window':
        raise ValueError('Unexpected finalization error; review separately')
    if (raw.get('actual_final_timeline', {}).get('playing') is not True
            or raw.get('compact_environment_after_Pause') is not None
            or raw.get('actual_sampled_cooked_support_queries') != []
            or raw.get('actual_disabled_subfloor_probes') != []):
        raise ValueError('Actual evidence does not match this precise missed-Pause boundary')
    observed_counts = Counter()
    for row in raw['contact_pairs_by_category']:
        pair = row['pair']
        if (not isinstance(pair, list) or len(pair) != 2 or pair != sorted(pair)
                or type(row['callback_count']) is not int or row['callback_count'] <= 0
                or row['category'] not in stationary.KNOWN_CATEGORIES):
            raise ValueError('Malformed recorded contact pair')
        category = stationary.classify_contact(tuple(pair), raw['environment_after_static']['disabled_legacy_colliders'])
        if category != row['category']:
            raise ValueError('Contact classification no longer reproduces')
        observed_counts[category] += row['callback_count']
    if dict(observed_counts) != raw['contact_report_counts_by_category']:
        raise ValueError('Recorded contact pair totals do not match callback counts')
    subset = physical.evaluate_static(rows, raw['start_xyyaw'], raw['simulation_mass_kg'],
        raw['contact_report_counts_by_category'], raw['contact_callback_errors'])
    reconstructed = dict(raw, contact_pairs_by_category=[dict(row, pair=tuple(row['pair']))
        for row in raw['contact_pairs_by_category']])
    dumped = json.dumps(reconstructed, indent=2, allow_nan=False).encode('utf-8')
    if dumped != result_bytes:
        raise ValueError('Tuple-only reconstruction does not reproduce exact original JSON bytes')
    return dict(original_static_verdict=raw['verdict'], original_evaluation_error=raw['evaluation_error'],
        original_static_result_NOT_relabelled=True, actual_raw_JSON_valid=True,
        recorded_physical_subset=subset,
        serialization_diagnosis=dict(exact_original_result_bytes_reproduced=True,
            contact_pair_tuples_serialized_as_JSON_arrays=len(raw['contact_pairs_by_category']),
            old_Python_object_equality_is_false=json.loads(dumped) != reconstructed,
            JSON_content_roundtrip_is_stable=json.loads(json.dumps(raw, allow_nan=False)) == raw,
            actual_file_corruption_demonstrated=False),
        missing_postconditions=dict(confirmed_Pause=False, complete_post_Pause_4997_row_environment=False,
            sampled_26_support_rays=False, four_disabled_identity_probes=False),
        original_report_timeline=raw['actual_final_timeline'],
        observed_settle_s=protocol['settle_s'], observed_record_duration_s=subset['observation_duration_s'],
        no_contact_or_force_claim_for_unmonitored_post_recording_interval=True,
        physical_subset_PASS_does_NOT_make_complete_static_PASS=True,
        actual_static_admission=False, actual_drive_admission=False, automatic_retry=False)


def build_saved_review():
    raw, protocol, rows = saved_records()
    analysis = analyze_records(raw, protocol, rows, stationary.RESULT.read_bytes())
    inputs = dict(raw['input_sha256'])
    for path in (stationary.RESULT, stationary.TRACE, stationary.RESERVATION,
            Path(__file__), Path(evidence.__file__), Path(physical.__file__), ENTRY):
        inputs[str(path)] = sha(path)
    return dict(version=VERSION, status='ACTUAL_PHYSICAL_SUBSET_AUDITED_STATIC_FINALIZATION_FAILED_NO_ADMISSION',
        **analysis, actual_static_result=str(stationary.RESULT), actual_raw_CSV=str(stationary.TRACE),
        input_sha256=inputs, new_physics_or_scene_operation_performed=False,
        all_old_PASS_FAIL_HOLDs_preserved=True)


def load_saved_review():
    report = stationary.preparation.read_small(SAVED_REVIEW)
    stationary.preparation.verify(report['input_sha256'])
    if report != build_saved_review():
        raise ValueError('Saved failure audit changed; no new trial or admission')
    return report


def frozen_query_functions(scope):
    """Compile ONLY the two immutable GET-only ray functions, not the entry.

    Their actual source bytes are covered by the original trial input hashes.
    No scheduling, controls or original top-level entry code is executed.
    """
    tree = ast.parse(stationary.ENTRY.read_text())
    names = ('sampled_queries', 'sampled_subfloor_queries')
    functions = [next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        and node.name == name) for name in names]
    forbidden = {'play', 'stop', 'pause', 'reset', 'initialize', 'wake_up', 'apply_action', 'Set', 'Save'}
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden for function in functions for node in ast.walk(function)):
        raise ValueError('Frozen ray body unexpectedly contains a mutation')
    namespace = dict(scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
        str(stationary.ENTRY), 'exec'), namespace)
    return [namespace[name] for name in names]


def continuity_diagnosis(raw, actual):
    """Report deviations, not exceptions that discard the current state."""
    previous = raw['passive_post_static_link_readback']
    if (actual['names'] != raw['final_actual_joint_names'] or actual['link_names'] != previous['link_names']):
        raise ValueError('Saved/live twelve-joint and thirteen-link ordering must match')
    old = np.asarray(previous['all_thirteen_link_world_transforms_xyzw'], float)
    now = np.asarray(actual['all_thirteen_link_world_transforms_xyzw'], float)
    q = np.asarray(actual['positions'], float)
    qd = np.asarray(actual['joint_velocities_rad_s'], float)
    if (old.shape != (13, 7) or now.shape != (13, 7) or q.shape != (12,) or qd.shape != (12,)
            or not all(np.isfinite(v).all() for v in (old, now, q, qd))):
        raise ValueError('Complete finite saved/current tensor state required')
    for transforms in (old, now):
        if not np.allclose(np.linalg.norm(transforms[:, 3:], axis=1), 1., rtol=0, atol=1e-4):
            raise ValueError('Unit XYZW link quaternions required')
    dot = np.sum((old[:, 3:]/np.linalg.norm(old[:, 3:], axis=1)[:, None]) *
        (now[:, 3:]/np.linalg.norm(now[:, 3:], axis=1)[:, None]), axis=1)
    position = float(np.max(np.linalg.norm(now[:, :3] - old[:, :3], axis=1)))
    orientation = float(np.max(np.degrees(2*np.arccos(np.clip(np.abs(dot), 0., 1.)))))
    joints = float(np.max(np.abs(q - np.asarray(raw['final_actual_twelve_joint_positions_rad']))))
    gains = raw['runtime_motor_baseline']
    same_gains = (actual['names'] == gains['joint_names'] and
        all(actual['gains'][key] == gains[key] for key in ('kp', 'kd', 'max_efforts')))
    return dict(maximum_link_position_difference_m=position, maximum_link_orientation_difference_deg=orientation,
        maximum_joint_difference_rad=joints, runtime_motor_baseline_retained=same_gains,
        twelve_joint_velocities_exactly_retained=qd.tolist() == raw['final_actual_twelve_joint_velocities_rad_s'],
        old_1mm_0p1deg_0p001rad_continuity=position <= .001 and orientation <= .1 and joints <= .001,
        passing_continuity_would_NOT_certify_unmonitored_physics_or_change_original_FAIL=True)


def main():
    report = build_saved_review()
    if SAVED_REVIEW.exists():
        load_saved_review()
    else:
        evidence.write_exclusive(SAVED_REVIEW, report)
    print('Original complete static verdict: FAIL (preserved)')
    print('Recorded physical subset:', report['recorded_physical_subset']['verdict'],
        'samples:', report['recorded_physical_subset']['sample_count'])
    print('Exact tuple-only serialization failure reproduced:', report['serialization_diagnosis'])
    print('Actual Pause / missing live queries remain unverified. No retry or drive admission.')
    print('saved:', SAVED_REVIEW)


if __name__ == '__main__':
    main()
