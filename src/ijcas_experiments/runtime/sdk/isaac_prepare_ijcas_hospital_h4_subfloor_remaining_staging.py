"""Finish the prior approved staging, ONLY after its exact completed Stop.

No additional Stop/Play, physics initialization, pose restore or giant JSON.
Full archived coverage is checked using small, lossless per-prim hashes.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


async def _stage_h4_remaining_compact():
    import json
    from pathlib import Path
    import runpy
    import time

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Usd

    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    r = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load'](scripts/'hospital_h4_subfloor_remaining_staging.py')
    if any(p.exists() for p in (r['RESERVATION'], r['LIVE'], r['FAILURE'], r['SNAPSHOTS'])):
        raise RuntimeError('This remaining-staging version is already reserved/recorded; no overwrite/retry')
    protocol = r['load_protocol']()
    manifest = r['load_manifest'](verify_inputs=False)
    _, boundary = r['load_review'](verify_inputs=False)
    context, timeline = omni.usd.get_context(), omni.timeline.get_timeline_interface()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError('Existing completed-Stop stage required; never reopen/reset to create it')
    root = boundary['actual_root_layer_text']
    session = boundary['actual_session_layer_text']
    began = heartbeat = time.monotonic()
    phase, row_count = 'preflight', 0

    def unchanged(expected_session):
        if (context.get_stage() != stage or timeline.is_playing() or not timeline.is_stopped()
                or float(timeline.get_current_time()) != 0.
                or stage.GetRootLayer().ExportToString() != root
                or stage.GetSessionLayer().ExportToString() != expected_session):
            raise RuntimeError('Actual STOPPED Root/Session boundary changed; no reset/recovery')
        if time.monotonic()-began > protocol['preparation_wall_budget_s']:
            raise TimeoutError('Compact staging inspection budget exceeded; no automatic retry')

    unchanged(session)
    if any(stage.GetPrimAtPath(p).IsValid() for p in (r['ROBOT'], r['MARKER'])):
        raise RuntimeError('New actor/marker already exists; never overwrite')
    environment = r['old'].static.validate_environment(stage)
    old_raw = r['read_small'](r['old'].static.RESULT)
    if environment != old_raw['environment_after_static']:
        raise RuntimeError('Original authored environment/material/masses/gains differ after Stop')
    r['write_exclusive'](r['RESERVATION'], dict(version=r['VERSION'], cohort=r['COHORT'],
        previously_approved_remaining_staging_only=True, new_Stop_or_Play_or_trial_authorized=False,
        protocol_sha256=r['sha'](r['PROTOCOL'])))
    app = omni.kit.app.get_app()

    async def snapshot(expected_session):
        nonlocal heartbeat, row_count
        rows = {}; yielded = time.monotonic()
        for path, row in r['compact'].stage_rows(stage, r['SELECTED'], r['old'].original):
            if path in rows:
                raise RuntimeError('Duplicate protected prim identity')
            rows[path] = row; row_count = len(rows)
            wall = time.monotonic()
            if wall-heartbeat >= protocol['heartbeat_s']:
                print('[H4 COMPACT PREP] phase=%s rows=%d/%d wall=%.1fs; STOPPED, no motion' %
                    (phase, row_count, protocol['all_original_protected_prim_rows_required'], wall-began), flush=True)
                heartbeat = wall
            if row_count % protocol['UI_yield_max_rows'] == 0 or wall-yielded >= protocol['UI_yield_wall_interval_s']:
                unchanged(expected_session)
                await app.next_update_async()  # UI refresh only; timeline MUST stay STOPPED
                unchanged(expected_session)
                yielded = time.monotonic()
        unchanged(expected_session)
        return rows

    try:
        phase = 'compare_full_archived_environment_before_changes'
        print('H4 compact remaining staging: checking ALL archived collider/material rows; no additional Stop/Play.', flush=True)
        before = await snapshot(session)
        expected = manifest['compact_environment_rows']
        before_digest = r['compact'].compare_rows(expected, before, r['SELECTED'], after_flags=False)
        unchanged(session)
        phase = 'authorized_four_flags_and_nominal_actor'
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            actual = r['author_session'](stage, protocol)
        staged_session = stage.GetSessionLayer().ExportToString()
        scope = r['old'].check_staging_layers(session, staged_session, actual['deinstanced_mesh_ancestors'])
        phase = 'compare_full_archived_environment_after_four_flags'
        after = await snapshot(staged_session)
        after_digest = r['compact'].compare_rows(expected, after, r['SELECTED'], after_flags=True)
        r['verify'](protocol['input_sha256'])
        unchanged(staged_session)
        snapshots = dict(codec=r['compact'].FORMAT, before=before, after=after,
            before_sha256=before_digest, after_sha256=after_digest,
            every_original_protected_row_verified=True, row_count=len(after),
            large_original_archive_NOT_loaded_or_hashed_inside_Kit=True,
            timeline_stayed_STOPPED_without_any_additional_Stop_or_Play=True)
        r['write_exclusive'](r['SNAPSHOTS'], snapshots)
        report = dict(version=r['VERSION'], cohort=r['COHORT'],
            status='REVIEWED_COMPLETED_STOP_COMPACT_SUBFLOOR_COHORT_STAGED_STATIC_PENDING',
            new_robot_root=r['ROBOT'], retired_robot_root=r['OLD_ROBOT'],
            preparation_protocol=str(r['PROTOCOL']), **actual,
            start_xyyaw=protocol['original_start_xyyaw'], unchanged_goal_xyyaw=protocol['unchanged_goal_xyyaw'],
            original_preparation_FAILURE_and_HOLD_preserved=True,
            original_archived_body_twelve_joints_thirteen_links_preserved_NOT_restored=True,
            original_archive_full_sha256_verified_OFFLINE=manifest['original_archive_sha256'],
            compact_original_archive_manifest=str(r['MANIFEST']), codec=r['compact'].FORMAT,
            actual_compact_readbacks=str(r['SNAPSHOTS']), protected_environment_sha256=after_digest,
            all_original_protected_row_count=len(after), all_rows_match_original_except_exact_four_flags=True,
            all_four_methods_require_same_new_environment_fingerprint=True,
            explicit_Session_field_deltas=scope, original_Root_and_source_USD_file_unchanged=True,
            timeline=dict(playing=False, stopped=True, current_time_s=0.),
            any_additional_Stop_Play_pause_reset_reopen_save_physics_initialization_or_trial=False,
            old_physical_continuation_ended_at_PRIOR_Stop_NOT_restored=True,
            large_original_archive_redecoded_or_hashed_inside_Kit=False,
            fresh_static_support_or_drive_validated=False, actual_drive_admission=False,
            CICS_runtime_integration=False, use_for_final_statistics=False,
            wall_time_s=time.monotonic()-began,
            input_sha256={**protocol['input_sha256'], str(r['PROTOCOL']):r['sha'](r['PROTOCOL']),
                str(r['SNAPSHOTS']):r['sha'](r['SNAPSHOTS'])})
        r['write_exclusive'](r['LIVE'], report)
    except Exception as error:
        # No rollback/import of an old physical state. Inhibit only a partially
        # created NEW actor; retain the exact original HOLD and all archives.
        if timeline.is_stopped() and stage.GetPrimAtPath(r['ROBOT']).IsValid():
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                stage.GetPrimAtPath(r['ROBOT']).SetActive(False)
        r['write_exclusive'](r['FAILURE'], dict(version=r['VERSION'], cohort=r['COHORT'],
            status='REMAINING_STAGING_HOLD_NO_AUTOMATIC_RETRY_OR_RESTORE', phase=phase,
            error=type(error).__name__+': '+str(error), last_compact_row_count=row_count,
            actual_timeline=dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
                current_time_s=float(timeline.get_current_time())),
            actual_Root_summary=r['layer_summary'](stage.GetRootLayer()),
            actual_Session_summary=r['layer_summary'](stage.GetSessionLayer()),
            any_additional_Stop_Play_reset_or_trial=False, actual_drive_admission=False))
        raise
    context.get_selection().set_selected_prim_paths([r['ROBOT']], True)
    print('=== H4 REVIEWED COMPLETED-STOP COMPACT REMAINING STAGING ===')
    print('original HOLD/large archive preserved; additional Stop/Play: NONE')
    print('all archived collider/material rows verified:', len(after))
    print('exact disabled subfloor meshes:', len(actual['disabled_subfloor_meshes']))
    print('new nominal actor:', r['ROBOT'], '; 265kg / 13 bodies / 12 source-authored joints')
    print('saved:', r['LIVE'])
    print('STOPPED. F then Top View. Staging ONLY; separate static approval still required. No manual Play/save/retry.')


if globals().get('ijcas_hospital_h4_subfloor_remaining_staging_authorized') is not True:
    raise RuntimeError('Explicit flag for the previously approved REMAINING staging only is required; no action taken')
import asyncio
_ijcas_h4_compact_remaining_staging_task = asyncio.ensure_future(_stage_h4_remaining_compact())
print('H4 COMPACT remaining staging scheduled; current completed Stop retained. NO additional Stop, Play, reset or trial.')
