"""New preparation identity binding; ALL original static checks/control kept."""

import json
from pathlib import Path

import hospital_h4_completed_stop_preparation as cohort
import hospital_h4_static_gate as original


ROOT, GATE, ROBOT, MARKER = cohort.ROOT, cohort.GATE, cohort.ROBOT, cohort.MARKER
VERSION = 'H4_original_static_contract_completed_Stop_staging_binding_v1'
OUTPUT = GATE/'hospital_h4_static_contact/cohort_01_completed_stop_continuation'
RUN_ID = 'hospital_h4_completed_stop_static_r01'
RESULT, TRACE = OUTPUT/(RUN_ID+'.json'), OUTPUT/(RUN_ID+'.csv')
RESERVATION = OUTPUT/'single_attempt_reserved.json'
PROTOCOL = GATE/'hospital_h4_static_admission/completed_stop_continuation_r01/static_protocol.json'
ENTRY = ROOT/'isaac/scripts/isaac_run_ijcas_hospital_h4_completed_stop_static_gate.py'
SETTLE_S, RECORD_S, PHYSICS_HZ = original.SETTLE_S, original.RECORD_S, original.PHYSICS_HZ
GUARDS, LIMITS, QUERY_LIMITS = original.GUARDS, original.LIMITS, original.QUERY_LIMITS
BAD_CONTACTS, WHEELS, SUPPORTS, KNOWN_CATEGORIES = original.BAD_CONTACTS, original.WHEELS, original.SUPPORTS, original.KNOWN_CATEGORIES
classify_contact, floor_query_samples, query_checks = original.classify_contact, original.floor_query_samples, original.query_checks
evaluate, merge_evaluation, validate_environment = original.evaluate, original.merge_evaluation, original.validate_environment
load_actual_preparation = cohort.load_actual_preparation


def prepare_protocol():
    continuation = cohort.load_protocol()
    baseline = original.load_protocol()
    # Explicitly disclose schema/root binding; do not relabel old failed v4.
    result = dict(baseline)
    result.update(version=VERSION, run_id=RUN_ID, required_actual_preparation=str(cohort.LIVE),
        original_static_protocol=str(original.PROTOCOL), original_static_control_and_acceptance_unchanged=True,
        preparation_binding_only_new_version=True, original_failed_v4_receipt_is_NOT_successful=True,
        reviewed_completed_Stop_protocol=str(cohort.PREFLIGHT))
    sources = (Path(__file__), ENTRY, cohort.LAUNCH)
    result['input_sha256'] = {**baseline['input_sha256'], **continuation['input_sha256'],
        str(original.PROTOCOL): cohort.completion.sha(original.PROTOCOL),
        str(cohort.PREFLIGHT): cohort.completion.sha(cohort.PREFLIGHT),
        **{str(path): cohort.completion.sha(path) for path in sources}}
    cohort.verify(result['input_sha256'])
    return cohort.canonical(result)


def load_protocol():
    result = json.loads(PROTOCOL.read_bytes()); cohort.verify(result['input_sha256'])
    if result != prepare_protocol():
        raise ValueError('New static binding or original checks changed; no Play')
    return result
