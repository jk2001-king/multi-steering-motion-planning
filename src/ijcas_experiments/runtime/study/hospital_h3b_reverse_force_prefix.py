"""SAME course-feedback prefix, separate cohort for ONE reverse observation.

Only evidence/output role changes. No controller, path, caster, gain, material,
speed or gate change. Prefix PASS alone never launches reverse automatically.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
from pathlib import Path

from hospital_h3b_rear_prefix_pilot import (
    POLICY, READINESS, ACCEPTANCE, BAD_CONTACTS, ROBOT_ROOT, WHEELS, GUARDS,
    TRACKING, COMMAND_LIMITS, TRACKING_GAINS, classify_contact, relative_pose,
    wheel_crossing_checks, motion_stalled, direct_joint_targets, component_goal,
    interpolation_table, reference_at_pose, FirstLegFeedback, body_velocity,
    evaluate_readiness, load_profile as baseline_load_profile,
)
from hospital_h3_flush_support import sha, verify_files


WORKSPACE = _ijcas_workspace()
GATE = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
COHORT = GATE/'hospital_h3b_reverse_force_observation/cohort_01'
OUTPUT = COHORT/'prefix'
CONDITION = 'H3B_rear_reverse_prefix_reproduction_v1'
ROLE = 'SAME_PREFIX_FOR_ONE_UNCHANGED_REAR_REVERSE_FORCE_OBSERVATION'
REFERENCE = GATE/'hospital_h3b_rear_reverse_pilot/cohort_01/hospital_h3b_rear_reverse_r01.json'
CACHE = GATE/'hospital_h3b_rear_reverse_pilot/paused_readbacks/rear_reverse_paused_forces_r01.json'


def review_reference():
    record = json.loads(REFERENCE.read_bytes()); cache = json.loads(CACHE.read_bytes())
    if (record.get('verdict') != 'FAIL' or record.get('component_mode') != 'rear_only'
            or record.get('error') != 'RuntimeError: Five-second measured reverse-progress stall'
            or record.get('original_motor_gains_restored') is not True
            or cache.get('original_verdict_unchanged') != 'FAIL'
            or cache.get('stage_root_session_timeline_and_body_unchanged') is not True
            or cache.get('new_PhysX_trial_performed') is not False
            or cache['friction_cache']['status'] != 'CACHED_ANCHOR_COVERAGE_UNAVAILABLE_DO_NOT_PRESS_PLAY'
            or cache['source_trial_sha256'] != sha(REFERENCE)):
        raise ValueError('Need the original actual rear stall and unchanged, unavailable paused cache receipt')
    verify_files(record['frozen_input_sha256']); verify_files(cache['input_sha256'])
    for report in (record, cache):
        for name, digest in report['source_sha256'].items():
            folder = _ijcas_path(WORKSPACE, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
            verify_files({folder/name: digest})
    return {**record['frozen_input_sha256'], **cache['input_sha256'], str(REFERENCE): sha(REFERENCE),
            str(REFERENCE.with_suffix('.csv')): sha(REFERENCE.with_suffix('.csv')), str(CACHE): sha(CACHE)}


def load_profile():
    info = baseline_load_profile()
    inputs = review_reference()
    info.update(input_sha256={**info['input_sha256'], **inputs, str(Path(__file__)): sha(__file__)},
        one_instrumented_rear_repeat_preparation_only=True,
        original_rear_stall_reference=str(REFERENCE), original_rear_stall_sha256=sha(REFERENCE),
        same_prefix_executor_and_controls_unchanged=True,
        prefix_PASS_does_not_automatically_launch_reverse=True)
    return info
