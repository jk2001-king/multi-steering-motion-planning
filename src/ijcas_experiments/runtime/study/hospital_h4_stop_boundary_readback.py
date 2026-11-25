"""Inspect the actual archived-H3 / deferred-Stop preparation boundary only.

NO timeline control, event pumping, physics initialization or scene changes.
The immutable failed preparation and its original actual archive stay intact.
Even a clean STOPPED readback is NOT staging, static or drive admission.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
from pathlib import Path

import hospital_h4_cohort_preparation_v4 as cohort


OUTPUT = cohort.GATE/'hospital_h4_cohort_preparation/stop_boundary_readback_r01'
RESULT = OUTPUT/'live_stop_boundary.json'
ENTRY = _ijcas_path(cohort.ROOT, 'isaac/scripts/isaac_review_ijcas_hospital_h4_stop_boundary.py')
TIMELINE_DOC = Path('/home/jk/miniconda3/envs/isaac_env/lib/python3.11/site-packages/isaacsim/extscache/'
    'omni.timeline-1.0.14+8131b85d.lx64.r.cp311/docs/FRAME_INTEGRITY.md')
EXPECTED_FAILURE = 'RuntimeError: Authorized Stop did not complete; no new robot may be staged'


def load_context():
    failure = json.loads(cohort.FAILURE.read_bytes())
    path = cohort.ARCHIVE/'actual_H3_final_state.json'
    archived = json.loads(path.read_bytes()); cohort.verify(archived['input_sha256'])
    preflight = json.loads(cohort.PREFLIGHT.read_bytes()); cohort.verify(preflight['input_sha256'])
    if (failure.get('error') != EXPECTED_FAILURE or failure.get('authorized_Stop_performed') is not True
            or failure.get('automatic_retry_or_restore') is not False
            or failure.get('actual_H3_archive') != str(path)
            or archived['status'] != 'ACTUAL_H3_FINAL_ARCHIVED_BEFORE_AUTHORIZED_STOP'
            or archived['source_sha256'] != cohort.SOURCE_SHA or archived['version'] != cohort.VERSION
            or len(archived['actual_state']['names']) != 12 or len(archived['actual_state']['link_names']) != 13):
        raise ValueError('Only the exact archived-before-Stop / immediate state-query failure is in scope')
    return failure, archived, preflight


def assess(checks, timeline):
    required = dict(checks)
    required['timeline_is_ACTUALLY_STOPPED_not_merely_requested'] = bool(timeline['stopped'] and not timeline['playing'])
    return dict(checks=required,
        status='STOP_BOUNDARY_READBACK_CONSISTENT_NO_STAGING_OR_STATIC_ADMISSION' if all(required.values()) else
            'HOLD_STOP_BOUNDARY_LIVE_STATE_REQUIRES_REVIEW',
        actual_H4_staging_or_static_or_drive_admission=False,
        additional_Stop_Play_reset_or_restore_performed=False,
        original_H3_contact_solver_state_retention_or_reconstruction_claimed=False)


def inspect(stage, timeline):
    from pxr import UsdGeom
    failure, archived, preflight = load_context()
    if stage is None:
        raise ValueError('An existing live hospital stage is required; never reopen it')
    root_before, session_before = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    before = dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
        current_time_s=float(timeline.get_current_time()))
    old, new, marker = (stage.GetPrimAtPath(p) for p in (cohort.OLD_ROBOT, cohort.ROBOT, cohort.MARKER))
    support = cohort.verify_actual_support(stage)
    # Stop is allowed to restore the OLD actor's USD transforms. Never compare
    # them to saved physical joint states or recreate a disposed tensor view.
    h = cohort.completion
    info, _, manifest, geometry, inventory, closed = h.load_closeout(h.DIRECTORY)
    import runpy
    original = runpy.run_path(str(_ijcas_path(cohort.ROOT, 'studies/IJCAS/scripts/hospital_h3_flush_support.py')))
    prep = info['final_forward_record']['physics_preparation_snapshot']
    original['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    original_support = original['support_readback'](stage, manifest)
    old_marker = stage.GetPrimAtPath(original['COMPONENT'])
    physical_activity_absent = not any((cohort.GATE/path).exists() for path in (
        'hospital_h4_static_contact/cohort_01/single_attempt_reserved.json',
        'hospital_h4_static_contact/cohort_01/hospital_h4_static_r01.json',
        'hospital_h4_path_execution_v3/cohort_01/single_static_handoff_attempt_reserved.json'))
    checks = dict(original_source_path_and_bytes_unchanged=Path(stage.GetRootLayer().realPath).resolve() == cohort.SOURCE.resolve()
            and h.sha(cohort.SOURCE) == cohort.SOURCE_SHA,
        root_layer_matches_actual_pre_Stop_archive=root_before == archived['root_layer_text_before_Stop'],
        protected_session_matches_actual_pre_Stop_archive=cohort.protected_session_text(session_before) ==
            cohort.protected_session_text(archived['session_layer_text_before_Stop']),
        old_actor_exists_and_not_yet_retired=old.IsValid() and old.IsActive(),
        no_new_H4_actor_or_marker_created=not new.IsValid() and not marker.IsValid(),
        no_successful_preparation_or_static_or_path_started=not cohort.LIVE.exists() and physical_activity_absent,
        original_H3_final_marker_unchanged=old_marker.IsValid()
            and old_marker.GetCustomDataByKey('status') == 'h3b_final_response_pilot_finished_wait_for_review'
            and old_marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') == archived['final_response_result'],
        exact_H4_support_points_topology_material_unchanged=all(support['checks'].values()),
        canonical_material_gravity_masses_drives_and_old_support_unchanged=original_support == prep['live_support_readback'],
        archived_five_component_verdicts_retained=closed['chain_audit']['all_five_raw_component_verdicts'] == 'PASS',
        current_H3_raw_final_receipt_matches_archive=info['final_forward_result'] == archived['final_response_result']
            and h.sha(info['final_forward_result']) == archived['final_response_result_sha256'])
    after = dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
        current_time_s=float(timeline.get_current_time()))
    checks['timeline_root_and_session_unchanged_by_readback'] = before == after and root_before == stage.GetRootLayer().ExportToString() \
        and session_before == stage.GetSessionLayer().ExportToString()
    base = stage.GetPrimAtPath(cohort.OLD_ROBOT+'/base_link')
    usd_origin = list(UsdGeom.XformCache().GetLocalToWorldTransform(base).ExtractTranslation())
    pins = {**archived['input_sha256'], **preflight['input_sha256'],
        **{str(p): h.sha(p) for p in (Path(__file__), ENTRY, TIMELINE_DOC, cohort.FAILURE,
            cohort.ARCHIVE/'actual_H3_final_state.json')}}
    h.verify_files(pins)
    return dict(version='H4_exact_deferred_Stop_failure_get_only_live_boundary_readback_v1', **assess(checks, before),
        actual_timeline=before, original_preparation_failure=failure, actual_H3_archive=str(cohort.ARCHIVE/'actual_H3_final_state.json'),
        actual_H3_archive_sha256=h.sha(cohort.ARCHIVE/'actual_H3_final_state.json'),
        old_actor_USD_origin_m=list(map(float, usd_origin)),
        old_actor_USD_origin_is_NOT_retained_physical_final_pose=True,
        exact_H4_support_readback=support, static_trial_user_approval_is_conditional_on_successful_future_preparation=True,
        H4_static_or_PhysX_not_executed_by_this_readback=True,
        no_caster_assignment_gain_material_change_or_source_save=True,
        prior_H3_MODEL_PASS_count=2, prior_H3_MODEL_case_count=8,
        independent_fresh_frozen_whole_H3_repeats=0, use_for_final_statistics=False,
        CICS_runtime_integration=False, input_sha256=pins)
