"""Finish ONLY the outstanding Pause and preserve delayed GET-only evidence.

No Play, Stop, reset, preparation, controls, source save or trial is allowed.
The original V2 FAIL and its raw JSON/CSV remain immutable, even if all delayed
queries pass. Contact monitoring did not cover the intervening interval.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import asyncio
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import runpy
import time

import carb
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd


print('[H4 STATIC CLOSEOUT] Pause confirmation + saved/current evidence ONLY; NO new trial.', flush=True)
_context = omni.usd.get_context()
_stage = _context.get_stage()
_timeline = omni.timeline.get_timeline_interface()
if _stage is None:
    raise RuntimeError('Current stage required; do not reopen/Play to create it')
# Safety first, before protocol loading/hashing or asynchronous scheduling.
if _timeline.is_playing():
    _timeline.pause()


async def _closeout_h4_static_v2():
    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    # This tiny helper has no USD/Kit imports or file/protocol traversal.
    io = runpy.run_path(str(scripts/'hospital_runtime_evidence_io.py'))
    app = omni.kit.app.get_app()

    def same_stage():
        if _context.get_stage() != _stage:
            raise RuntimeError('Stage changed; no recovery or new static trial')

    pause = await io['confirm_pause'](_timeline, app.next_update_async, check_context=same_stage)
    print('[H4 STATIC CLOSEOUT] Actual PAUSED confirmed:', pause['after'], flush=True)
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h = load(scripts/'hospital_h4_static_v2_failure_review.py')
    saved = h['load_saved_review']()
    s = h['stationary']
    c = s.preparation
    # Refuse to overwrite a completed delayed closeout; never start a trial.
    if h['LIVE'].exists():
        print('[H4 STATIC CLOSEOUT] Existing closeout preserved:', h['LIVE'], flush=True)
        return
    raw, protocol, unused_rows = h['saved_records']()
    root = _stage.GetRootLayer().ExportToString()
    session = _stage.GetSessionLayer().ExportToString()
    retained_time = pause['after']
    report = dict(version=h['VERSION'], original_static_verdict=raw['verdict'],
        original_static_result_NOT_relabelled=True, pause_confirmation=pause,
        status='DELAYED_GET_ONLY_CLOSEOUT_INCOMPLETE_NO_ADMISSION',
        saved_data_review=str(h['SAVED_REVIEW']), saved_data_review_sha256=h['sha'](h['SAVED_REVIEW']),
        original_result_sha256=h['sha'](s.RESULT), original_trace_sha256=h['sha'](s.TRACE),
        delayed_from_original_recorded_timeline_s=retained_time['current_time_s']-raw['actual_final_timeline']['current_time_s'],
        intervening_contact_force_and_safety_history_NOT_observed=True,
        full_static_PASS_or_drive_admission=False, actual_drive_admission=False,
        automatic_retry=False, any_Play_Stop_reset_control_or_stage_edit=False,
        original_Root_text_at_delayed_closeout=root, original_Session_text_at_delayed_closeout=session,
        checks={}, errors=[], input_sha256=dict(saved['input_sha256'], **{str(h['SAVED_REVIEW']):h['sha'](h['SAVED_REVIEW'])}))

    def unchanged():
        same_stage()
        if (io['timeline_state'](_timeline) != retained_time
                or _stage.GetRootLayer().ExportToString() != root
                or _stage.GetSessionLayer().ExportToString() != session):
            raise RuntimeError('Paused timeline/Root/Session changed during delayed GET-only closeout')

    try:
        unchanged()
        # No articulation initialization or Play to manufacture a tensor view.
        try:
            from isaacsim.core.simulation_manager import SimulationManager
            view = SimulationManager.get_physics_sim_view()
            if view is None:
                raise RuntimeError('Current tensor view unavailable; no Play to recreate it')
            tensor = view.create_articulation_view(c.ROBOT+'/base_link')
            report['actual_current_tensor_snapshot'] = c.old.tensor_snapshot(tensor)
            report['continuity_diagnosis'] = h['continuity_diagnosis'](raw, report['actual_current_tensor_snapshot'])
        except Exception as error:
            report['errors'].append('tensor: '+type(error).__name__+': '+str(error))
        print('[H4 STATIC CLOSEOUT] Checking retained source settings and all 4997 protected rows...', flush=True)
        actual_actor = s.actor_configuration(_stage)
        source_actor = raw['runtime_anonymous_source_reference']['fixed_configuration_rows']
        report['same_actual_trial_SDK_actor_rows'] = dict(expected=source_actor, actual=actual_actor)
        report['checks']['actor_settings_equal_own_same_SDK_PRE_Play_source_reference'] = actual_actor == source_actor
        report['critical_environment'] = s.critical_environment(_stage, raw['physics_preparation_snapshot'])
        manifest = c.load_manifest(verify_inputs=False)
        environment_rows = {}
        began = heartbeat = yielded = time.monotonic()
        for path, row in c.compact.stage_rows(_stage, c.SELECTED, c.old.original):
            if path in environment_rows:
                raise ValueError('Duplicate protected row identity')
            environment_rows[path] = row
            now = time.monotonic()
            if now-began > protocol['compact_readback_wall_budget_s']:
                raise TimeoutError('Delayed nonmoving environment readback budget exceeded')
            if now-heartbeat >= protocol['compact_readback_heartbeat_s']:
                print('[H4 STATIC CLOSEOUT] Protected rows %d/%d; PAUSED, no trial.' %
                    (len(environment_rows), protocol['all_archived_environment_rows_required']), flush=True)
                heartbeat = now
            if (len(environment_rows) % protocol['compact_readback_UI_yield_max_rows'] == 0
                    or now-yielded >= protocol['compact_readback_UI_yield_wall_interval_s']):
                unchanged()
                await app.next_update_async()
                unchanged()
                yielded = time.monotonic()
        fingerprint = c.compact.compare_rows(manifest['compact_environment_rows'], environment_rows, c.SELECTED, after_flags=True)
        report['protected_environment_delayed_readback'] = dict(row_count=len(environment_rows), sha256=fingerprint)
        report['checks']['all_4997_original_rows_match_except_exact_four_approved_flags'] = (
            len(environment_rows) == protocol['all_archived_environment_rows_required']
            and fingerprint == raw['physics_preparation_snapshot']['protected_environment_sha256'])
        support_queries, identity_queries = h['frozen_query_functions'](dict(carb=carb, omni=omni,
            c=s.cohort, protocol=protocol))
        unchanged()
        report['delayed_26_support_queries'] = supports = support_queries()
        report['delayed_four_identity_probes'] = probes = identity_queries()
        report['delayed_query_evaluation'] = s.evaluate(unused_rows, raw['start_xyyaw'],
            raw['contact_report_counts_by_category'], raw['contact_callback_errors'],
            protocol['sampled_cooked_support_queries'], supports, protocol['support_top_m'])
        report['delayed_probe_evaluation'] = s.evaluate_subfloor_queries(protocol, supports, probes)
        # These are separately timed diagnostic queries, NOT original static checks.
        report['checks']['delayed_query_transactions_and_limits'] = all(report['delayed_query_evaluation']['checks'].values())
        report['checks']['delayed_disabled_identity_probes'] = all(report['delayed_probe_evaluation']['checks'].values())
        unchanged()
        report['checks']['Root_Session_and_paused_time_unchanged_during_GETs'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    report['actual_final_timeline'] = io['timeline_state'](_timeline)
    report['checks']['original_trial_files_unchanged'] = (
        h['sha'](s.RESULT) == report['original_result_sha256'] and h['sha'](s.TRACE) == report['original_trace_sha256'])
    if not report['errors'] and report['checks'] and all(report['checks'].values()):
        report['status'] = 'DELAYED_GET_ONLY_CLOSEOUT_RECORDED_ORIGINAL_STATIC_FAIL_AND_NO_ADMISSION'
    report['recorded_UTC'] = datetime.now(timezone.utc).isoformat()
    io['write_exclusive'](h['LIVE'], report)
    print('=== H4 ACTUAL STATIC V2 FAILURE / DELAYED PAUSED CLOSEOUT ===', flush=True)
    print('actual timeline:', report['actual_final_timeline'], flush=True)
    print('original complete static verdict: FAIL — preserved; NO reclassification', flush=True)
    print('saved/live continuity:', report.get('continuity_diagnosis'), flush=True)
    print('failed delayed checks:', [key for key, ok in report['checks'].items() if not ok], flush=True)
    print('errors:', report['errors'], flush=True)
    print('STATUS:', report['status'], flush=True)
    print('saved:', h['LIVE'], flush=True)
    print('No new trial or route admission. Do not rerun static/Play/Stop/reset/save.', flush=True)


def _report_closeout_error(task):
    if task.cancelled():
        print('[H4 STATIC CLOSEOUT] Cancelled; no automatic retry.', flush=True)
        return
    error = task.exception()
    if error is not None:
        import traceback
        traceback.print_exception(type(error), error, error.__traceback__)
        print('[H4 STATIC CLOSEOUT] Error preserved; no automatic retry or trial.', flush=True)


ijcas_hospital_h4_static_v2_closeout_task = asyncio.ensure_future(_closeout_h4_static_v2())
ijcas_hospital_h4_static_v2_closeout_task.add_done_callback(_report_closeout_error)
