"""Complete ONLY the remaining approved staging after the ACTUAL prior Stop.

No new Stop, Play, reset, dynamic-state restore or giant JSON read. The old
HOLD stays authoritative. Exactly four added float/zero steering-state specs
are reviewed explicitly; no tolerance or blanket runtime-field waiver.
"""

import hashlib
import json
from pathlib import Path
from types import FunctionType

import hospital_compact_usd_fingerprint as compact
import hospital_h4_subfloor_preparation as old


ROOT, GATE = old.ROOT, old.GATE
SOURCE, SOURCE_SHA, ASSET, ASSET_SHA = old.SOURCE, old.SOURCE_SHA, old.ASSET, old.ASSET_SHA
ROBOT, OLD_ROBOT, MARKER, COHORT = old.ROBOT, old.OLD_ROBOT, old.MARKER, old.COHORT
SELECTED = old.SELECTED
OUTPUT = old.OUTPUT/'completed_stop_compact_preparation_r01'
MANIFEST = OUTPUT/'archive_manifest/compact_archived_environment.json'
DIAGNOSIS = old.OUTPUT/'failure_diagnosis_r01/actual_archive_and_Stop_review.json'
PROTOCOL, LIVE, FAILURE, RESERVATION, SNAPSHOTS = (OUTPUT/name for name in (
    'preparation_protocol.json', 'live_h4_preparation.json', 'preparation_failure.json',
    'single_remaining_staging_reserved.json', 'actual_compact_environment_readbacks.json'))
ENTRY = ROOT/'isaac/scripts/isaac_prepare_ijcas_hospital_h4_subfloor_remaining_staging.py'
VERSION = 'H4_exact_reviewed_completed_Stop_remaining_four_flags_and_nominal_cohort_COMPACT_ONLY_v1'
sha, verify = old.sha, old.verify
MAX_SMALL_JSON_BYTES = 4*1024*1024


def read_small(path):
    path = Path(path)
    if path.stat().st_size > MAX_SMALL_JSON_BYTES:
        raise ValueError('Small prepared artifact exceeds explicit 4 MiB budget: '+str(path))
    return json.loads(path.read_bytes())


def write_exclusive(path, value):
    if len(json.dumps(value, indent=2, allow_nan=False).encode()) > MAX_SMALL_JSON_BYTES:
        raise ValueError('Live output exceeds explicit 4 MiB budget; never expand geometry arrays')
    old.write_exclusive(path, value)


def layer_summary(layer):
    """Keep a failure receipt small even if an unexpected edit added a mesh."""
    text = layer.ExportToString()
    encoded = text.encode('utf-8')
    return dict(identifier=layer.identifier, text_sha256=hashlib.sha256(encoded).hexdigest(),
        text_bytes=len(encoded), text_if_under_64KiB=text if len(encoded) <= 65536 else None,
        oversized_layer_text_deliberately_NOT_embedded=len(encoded) > 65536)


def load_review(verify_inputs=True):
    review, boundary, failure = map(read_small, (DIAGNOSIS, old.BOUNDARY, old.FAILURE))
    if verify_inputs:
        verify(review['input_sha256'])
    before = review['small_archived_actual_state_and_pre_Stop_Root_Session_fields']
    if (review['status'] != 'PREPARATION_HOLD_DIAGNOSED_SAVED_BOUNDARY_ONLY_NO_LIVE_CONTINUATION_ADMISSION'
            or not all(review['checks'].values()) or review['original_archive'] != str(old.ARCHIVE)
            or boundary['actual_timeline'] != dict(playing=False, stopped=True, current_time_s=0.)
            or boundary['Stop_called_once'] is not True
            or failure['status'] != 'PREPARATION_HOLD_REVIEW_ACTUAL_BOUNDARY_NO_AUTOMATIC_RECOVERY'
            or failure['actual_root_layer_text'] != boundary['actual_root_layer_text']
            or failure['actual_session_layer_text'] != boundary['actual_session_layer_text']
            or before['root_layer_text_before_Stop'] != boundary['actual_root_layer_text']):
        raise ValueError('Exact actual completed Stop and untouched original HOLD required')
    # This exact native-Sdf diff was already computed offline in the pinned
    # diagnosis. Recheck every reported spec, never re-import a live layer.
    changes = {r['path']:(r['before'],r['after']) for r in review['exact_Stop_Session_spec_differences']}
    expected = {OLD_ROBOT+'/joints/'+joint+'.state:angular:physics:'+key
        for joint in ('base_link_to_front_active_steering', 'base_link_to_rear_active_steering')
        for key in ('position', 'velocity')}
    target = dict(spec_type='AttributeSpec', info=dict(custom=False, default=0., typeName='float',
        variability=dict(type='Variability', value='Sdf.VariabilityVarying')))
    if (len(review['exact_Stop_Session_spec_differences']) != 4 or set(changes) != expected
            or any(a is not None or b != target for a,b in changes.values())):
        raise ValueError('Only the four EXACT actual added zero/float/varying joint specs have been reviewed')
    old.validate_actual_state(read_small(old.static.RESULT), before['actual_state'])
    return review, boundary


def load_manifest(verify_inputs=True, review_context=None):
    report = read_small(MANIFEST)
    if verify_inputs:
        verify(report['input_sha256'])
    _, boundary = review_context if review_context is not None else load_review(verify_inputs)
    state = old.ARCHIVE.stat()  # metadata only; never load/hash this giant file inside Kit
    if (report['status'] != 'FULL_ORIGINAL_ARCHIVE_HASH_VERIFIED_OFFLINE_COMPACT_MANIFEST_NOT_LIVE_STAGING'
            or report['codec'] != compact.FORMAT
            or report['original_archive_sha256'] != boundary['actual_prior_archive_sha256']
            or report['original_archive_bytes'] != state.st_size
            or report['original_archive_mtime_ns'] != state.st_mtime_ns
            or not report['full_original_archive_sha_matches_original_Stop_boundary']
            or not report['all_original_protected_prim_rows_included']
            or report['prim_row_count'] != len(report['compact_environment_rows'])
            or report['approved_disabled_paths'] != list(SELECTED)):
        raise ValueError('Entire-original-archive verified OFFLINE manifest required, never a reduced ROI substitute')
    return report


def _protocol_from_context(prior, review, boundary, manifest):
    paths = (Path(__file__), Path(compact.__file__), ENTRY, MANIFEST, DIAGNOSIS, old.BOUNDARY, old.FAILURE, old.PROTOCOL)
    return dict(version=VERSION, cohort=COHORT,
        status='REVIEWED_COMPLETED_STOP_REMAINING_PREVIOUSLY_APPROVED_STAGING_ONLY',
        original_start_xyyaw=prior['original_start_xyyaw'], unchanged_goal_xyyaw=prior['unchanged_goal_xyyaw'],
        new_robot_root=ROBOT, original_preparation_HOLD_preserved=True,
        existing_Stop_completed_no_additional_Stop_authorized=True,
        exact_four_added_zero_specs_reviewed_NOT_a_tolerance_waiver=True,
        original_large_archive=str(old.ARCHIVE), original_large_archive_preserved=True,
        original_archive_will_NOT_be_redecoded_or_hashed_inside_Kit=True,
        original_archive_sha256_verified_OFFLINE=manifest['original_archive_sha256'],
        compact_manifest=str(MANIFEST), codec=compact.FORMAT,
        all_original_protected_prim_rows_required=manifest['prim_row_count'],
        protected_Root_text_sha256=hashlib.sha256(boundary['actual_root_layer_text'].encode()).hexdigest(),
        required_pre_staging_Session_sha256=hashlib.sha256(boundary['actual_session_layer_text'].encode()).hexdigest(),
        approved_collision_enabled_false_paths=list(SELECTED),
        same_original_nominal_constructor_source_seed_mass_gains_material_support_and_task=True,
        no_full_Hospital_attribute_JSON_expansion_or_importer_mutation=True,
        UI_yield_max_rows=16, UI_yield_wall_interval_s=.05, heartbeat_s=15., preparation_wall_budget_s=180.,
        maximum_each_live_JSON_bytes=MAX_SMALL_JSON_BYTES,
        no_Play_Stop_pause_reset_reopen_save_physics_initialization_or_trial=True,
        all_four_methods_require_same_new_environment_fingerprint=True,
        original_1029_dynamics_full_envelope_multi_steering_planning_unchanged=True,
        existing_H3_H4_PASS_FAIL_HOLDs_preserved_NOT_pooled_or_relabelled=True,
        fresh_actual_static_and_separate_trial_approval_still_required=True,
        actual_static_or_drive_admission=False, CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**prior['input_sha256'], **manifest['input_sha256'], **review['input_sha256'],
            **{str(p):sha(p) for p in paths}})


def prepare_protocol():
    prior = old.load_protocol()
    context = load_review()
    manifest = load_manifest(review_context=context)
    return _protocol_from_context(prior, *context, manifest)


def load_protocol():
    protocol = read_small(PROTOCOL); verify(protocol['input_sha256'])
    # All source/evidence hashes were checked above ONCE. Reconstruct only
    # this small protocol, not the old nested offline builders inside Kit.
    context = load_review(verify_inputs=False)
    manifest = load_manifest(verify_inputs=False, review_context=context)
    prior = read_small(old.PROTOCOL)
    if protocol != _protocol_from_context(prior, *context, manifest):
        raise ValueError('Exact compact remaining-staging protocol changed')
    if str(old.ARCHIVE) in protocol['input_sha256']:
        raise ValueError('Do not pull the giant archive into live general-purpose hash checks')
    return protocol


def author_session(stage, protocol):
    """Identical immutable author bytecode; only NEW receipt bindings differ."""
    scope = dict(old.author_session.__globals__, PROTOCOL=PROTOCOL, LIVE=LIVE)
    function = old.author_session
    return FunctionType(function.__code__, scope, function.__name__, function.__defaults__, function.__closure__)(stage, protocol)


def main():
    report = prepare_protocol()
    if PROTOCOL.exists():
        load_protocol()
    else:
        write_exclusive(PROTOCOL, report)
    print('Completed Stop reviewed. COMPACT remaining staging prepared; NO additional Stop/Play/static/drive.')
    print('All original protected rows:', report['all_original_protected_prim_rows_required'])
    print('saved:', PROTOCOL)


if __name__ == '__main__':
    main()
