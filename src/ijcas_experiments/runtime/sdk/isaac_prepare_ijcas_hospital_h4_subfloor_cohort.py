"""Explicitly approved preparation boundary: archive / ONE Stop / new cohort.

No Play, physics initialization, static trial, drive, retry or solver restore.
The async update wait is essential: timeline.stop() may complete next frame.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


async def _prepare_h4_subfloor_cohort():
    import json
    from pathlib import Path
    import runpy
    import time

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Usd

    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_subfloor_preparation.py')
    protocol = r['load_protocol']()
    if any(p.exists() for p in (r['RESERVATION'], r['ARCHIVE'], r['BOUNDARY'], r['LIVE'], r['FAILURE'])):
        raise RuntimeError('This single preparation is already reserved/recorded; no overwrite or automatic retry')
    context, timeline = omni.usd.get_context(), omni.timeline.get_timeline_interface()
    stage = context.get_stage()

    def timeline_state():
        return dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
            current_time_s=float(timeline.get_current_time()))

    before = timeline_state()
    if stage is None or before['playing'] or before['stopped']:
        raise RuntimeError('Exact actual reopened-static PASS must remain PAUSED; no reset or Play to recover it')
    if any(stage.GetPrimAtPath(p).IsValid() for p in (r['ROBOT'], r['MARKER'])):
        raise RuntimeError('New cohort namespace already exists; no overwrite')
    old_marker = stage.GetPrimAtPath(r['OLD_MARKER'])
    if (not old_marker.IsValid() or old_marker.GetCustomDataByKey('status') != 'static_support_passed'
            or old_marker.GetCustomDataByKey('static_gate_result') != str(r['static'].RESULT)
            or old_marker.GetCustomDataByKey('static_gate_sha256') != r['sha'](r['static'].RESULT)):
        raise RuntimeError('Exact actual reopened-static PASS marker required')
    raw = json.loads(r['static'].RESULT.read_bytes())
    environment = r['static'].validate_environment(stage)
    if environment != raw['environment_after_static']:
        raise RuntimeError('Original environment/physics baseline changed before archival')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Retained actual tensor view missing; never initialize or Play')
    tensor = sim.create_articulation_view(r['OLD_ROBOT']+'/base_link')
    actual = r['tensor_snapshot'](tensor)
    continuity = r['validate_actual_state'](raw, actual)
    root, session = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    protected = r['environment_snapshot'](stage)
    # Validate the exact four enabled starting flags BEFORE any reservation/Stop.
    expected = json.loads(json.dumps(protected))
    for path in r['SELECTED']:
        expected[path]['attributes']['physics:collisionEnabled']['value'] = False
    r['check_environment_delta'](protected, expected)
    r['verify'](protocol['input_sha256'])
    r['write_exclusive'](r['RESERVATION'], dict(version=r['VERSION'], cohort=r['COHORT'],
        user_authorized_archive_one_Stop_and_new_Session_cohort=True,
        Play_static_trial_drive_or_retry_authorized=False,
        protocol_sha256=r['sha'](r['PROTOCOL'])))
    stop_requested = False
    try:
        archive = dict(version=r['VERSION'], cohort=r['COHORT'],
            status='ACTUAL_REOPENED_H4_STATIC_PASS_ARCHIVED_BEFORE_AUTHORIZED_STOP',
            actual_state=actual, live_continuity=continuity, timeline_before_Stop=before,
            actual_static_result=str(r['static'].RESULT), actual_static_sha256=r['sha'](r['static'].RESULT),
            root_layer_text_before_Stop=root, session_layer_text_before_Stop=session,
            protected_environment_before_Stop=protected, environment_after_actual_static=environment,
            source_USD_file_sha256=r['sha'](r['SOURCE']),
            physical_solver_contact_state_NOT_serialized_or_restoreable=True,
            original_H3_H4_PASS_FAIL_HOLD_records_preserved=True,
            input_sha256={**protocol['input_sha256'], str(r['PROTOCOL']): r['sha'](r['PROTOCOL'])})
        r['write_exclusive'](r['ARCHIVE'], archive)
        if (actual != r['tensor_snapshot'](tensor) or before != timeline_state()
                or context.get_stage() != stage or root != stage.GetRootLayer().ExportToString()
                or session != stage.GetSessionLayer().ExportToString()
                or protected != r['environment_snapshot'](stage)):
            raise RuntimeError('Durable archive/live continuity failed BEFORE Stop; no staging')
        print('H4 actual static PASS body/12 joints/13 links and Root/Session durably archived BEFORE Stop.', flush=True)
        started = time.monotonic()
        # ONE authorized call only. Keep writeback directed at Session while
        # awaiting deferred cleanup, and always restore the prior EditTarget.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            stop_requested = True
            timeline.stop()
            app = omni.kit.app.get_app()
            for update_count in range(1, protocol['Stop_update_budget']+1):
                await app.next_update_async()
                if context.get_stage() != stage:
                    raise RuntimeError('Stage replaced while awaiting Stop; no staging')
                state = timeline_state()
                if state['playing'] or time.monotonic()-started > protocol['Stop_wall_budget_s']:
                    raise RuntimeError('Authorized Stop did not complete within its bounded wait; no staging/retry')
                if state == dict(playing=False, stopped=True, current_time_s=0.):
                    break
            else:
                raise RuntimeError('Authorized Stop update budget exhausted; no staging/retry')
            # Flush another frame of deferred cleanup before exact field checks.
            await app.next_update_async()
        state = timeline_state()
        post_root, post_session = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
        r['write_exclusive'](r['BOUNDARY'], dict(version=r['VERSION'], actual_timeline=state,
            Stop_called_once=True, awaited_update_count=update_count+1,
            actual_root_layer_text=post_root, actual_session_layer_text=post_session,
            actual_prior_archive=str(r['ARCHIVE']), actual_prior_archive_sha256=r['sha'](r['ARCHIVE']),
            no_physical_state_restore_or_Play=True))
        if state != dict(playing=False, stopped=True, current_time_s=0.):
            raise RuntimeError('Timeline changed after completed Stop; no staging')
        stop_delta = r['check_stop_layers'](root, post_root, session, post_session)
        if protected != r['environment_snapshot'](stage):
            raise RuntimeError('Protected environment changed during Stop; no staging')
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            staged = r['author_session'](stage, protocol)
        scope = r['check_staging_layers'](post_session, stage.GetSessionLayer().ExportToString(), staged['deinstanced_mesh_ancestors'])
        fingerprint = r['check_environment_delta'](protected, r['environment_snapshot'](stage))
        if (root != stage.GetRootLayer().ExportToString() or context.get_stage() != stage
                or state != timeline_state() or r['sha'](r['SOURCE']) != r['SOURCE_SHA']):
            raise RuntimeError('Source/Root/stage/timeline changed during Session staging')
        r['verify'](protocol['input_sha256'])
        report = dict(version=r['VERSION'], cohort=r['COHORT'],
            status='NEW_SUBFLOOR_ENVIRONMENT_COHORT_STAGED_STOPPED_STATIC_PENDING',
            new_robot_root=r['ROBOT'], retired_robot_root=r['OLD_ROBOT'],
            actual_prior_static_archive=str(r['ARCHIVE']), actual_prior_static_archive_sha256=r['sha'](r['ARCHIVE']),
            archive_durably_verified_before_Stop=True, Stop_called_once_and_deferred_completion_verified=True,
            old_physical_continuation_ended_NOT_restored=True, timeline=state, **staged,
            start_xyyaw=protocol['original_start_xyyaw'], unchanged_goal_xyyaw=protocol['unchanged_goal_xyyaw'],
            preparation_protocol=str(r['PROTOCOL']), protected_environment_sha256=fingerprint,
            same_environment_fingerprint_required_for_all_four_methods=True,
            Stop_Session_runtime_field_deltas=stop_delta, explicit_staging_Session_field_deltas=scope,
            original_root_and_source_USD_file_unchanged=True,
            old_environment_outcomes_preserved_NOT_pooled_or_relabelled=True,
            original_planning_map_changed_cells=0, fresh_static_support_validated=False,
            Play_static_or_drive_executed=False, actual_drive_admission=False,
            caster_state_assigned_or_selected=False, CICS_runtime_integration=False, use_for_final_statistics=False,
            input_sha256={**archive['input_sha256'], str(r['ARCHIVE']): r['sha'](r['ARCHIVE']),
                str(r['BOUNDARY']): r['sha'](r['BOUNDARY'])})
        r['write_exclusive'](r['LIVE'], report)
    except Exception as error:
        # Never restore archived transforms, reset, import a live layer or retry.
        # Inhibit a partially-created NEW actor only; do not touch the old one.
        if stop_requested and timeline.is_stopped() and stage.GetPrimAtPath(r['ROBOT']).IsValid():
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                stage.GetPrimAtPath(r['ROBOT']).SetActive(False)
        r['write_exclusive'](r['FAILURE'], dict(version=r['VERSION'], cohort=r['COHORT'],
            status='PREPARATION_HOLD_REVIEW_ACTUAL_BOUNDARY_NO_AUTOMATIC_RECOVERY',
            error=type(error).__name__+': '+str(error), Stop_requested=stop_requested,
            actual_timeline=timeline_state(), archive_exists=r['ARCHIVE'].exists(),
            actual_root_layer_text=stage.GetRootLayer().ExportToString(),
            actual_session_layer_text=stage.GetSessionLayer().ExportToString(),
            Play_or_static_or_drive_executed=False, actual_drive_admission=False))
        raise
    context.get_selection().set_selected_prim_paths([r['ROBOT']], True)
    print('=== H4 APPROVED SUBFLOOR SESSION VARIANT / NEW COHORT PREPARED ===')
    print('archived actual prior static PASS; ONE completed Stop; old H3/H4 PASS/FAIL/HOLD preserved')
    print('disabled exact subfloor meshes:', len(staged['disabled_subfloor_meshes']))
    print('new actor:', r['ROBOT'], '; same 265kg / thirteen bodies / twelve source-authored joints')
    print('environment fingerprint:', fingerprint)
    print('saved:', r['LIVE'])
    print('STOPPED. Preparation ONLY; no Play/static trial/drive/retry. F then Top View.')


if globals().get('ijcas_hospital_h4_subfloor_cohort_authorized') is not True:
    raise RuntimeError('Explicit user-authorized new environment/archive/Stop/staging flag required; no action taken')
import asyncio
_ijcas_h4_subfloor_preparation_task = asyncio.ensure_future(_prepare_h4_subfloor_cohort())
print('H4 approved preparation scheduled: archive -> ONE deferred Stop -> four Session flags -> NEW nominal cohort. NO Play.')
