"""User-requested reopened H4 nominal setup. Session ONLY, no timeline action."""


def _prepare_h4_reopened_cohort():
    import hashlib
    from pathlib import Path
    import runpy

    import omni.timeline
    import omni.usd
    from pxr import Usd

    scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_reopened_preparation.py')
    protocol, readback = r['load_protocol'](), r['load_source_readback']()
    if any(path.exists() for path in (r['LIVE'], r['FAILURE'], r['RESERVATION'])):
        raise RuntimeError('This reopened cohort already reserved/staged/failed; no overwrite or retry')
    stage, timeline = omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface()
    before = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    if stage is None or before != (False, True, 0.):
        raise RuntimeError('Actual inspected source must remain STOPPED at zero; do not reset/Play to create it')
    root, session = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    if (hashlib.sha256(root.encode()).hexdigest() != readback['Root_text_sha256']
            or hashlib.sha256(session.encode()).hexdigest() != readback['Session_text_sha256']):
        raise RuntimeError('Root/Session differs from the actual v2 fresh-source readback; no implicit recovery')
    current = r['restart'].inspect_fresh_source(stage, *before)
    if current['checks'] != {k: v for k, v in readback['checks'].items()
            if k != 'Root_Session_and_timeline_unchanged_by_readback'} or not all(current['checks'].values()):
        raise RuntimeError('Original Hospital collider/flag/geometry continuity changed before setup')
    r['write_exclusive'](r['RESERVATION'], dict(cohort=r['COHORT'], nominal_setup_requested_by_user=True,
        no_Play_or_static_or_drive_authority=True, protocol_sha256=r['sha'](r['PREFLIGHT'])))
    try:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            actual = r['author_session'](stage, protocol)
        delta = r['session_delta'](session, stage.GetSessionLayer().ExportToString(),
            protocol['disabled_duplicate_floor_paths'], actual['deinstanced_floor_roots'], actual['physics_scene'])
        after = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
        if (root != stage.GetRootLayer().ExportToString() or after != before
                or r['sha'](r['SOURCE']) != r['SOURCE_SHA']):
            raise RuntimeError('Source/Root/timeline changed during Session-only setup')
        r['verify'](protocol['input_sha256'])
        report = dict(version=r['VERSION'], status='REOPENED_H4_NOMINAL_SESSION_STAGED_NEW_STATIC_PENDING',
            cohort=r['COHORT'], new_robot_root=r['ROBOT'], start_xyyaw=protocol['original_start_xyyaw'],
            unchanged_goal_xyyaw=protocol['unchanged_goal_xyyaw'], preparation_protocol=str(r['PREFLIGHT']),
            actual_source_readback=str(r['restart'].RESTART), **actual,
            Root_and_source_file_unchanged=True, timeline=dict(zip(('playing', 'stopped', 'current_time_s'), after)),
            explicit_Session_change_scope=delta, old_H3_H4_records_preserved=True,
            old_solver_contact_cache_or_physical_state_restored=False,
            fresh_static_support_or_drive_validated=False, actual_drive_admission=False,
            no_Play_Stop_Pause_reset_or_source_save_performed=True, CICS_runtime_integration=False,
            known_low_collider_map_height_gap_still_blocks_old_route_drive=True, use_for_final_statistics=False,
            input_sha256={**protocol['input_sha256'], str(r['PREFLIGHT']): r['sha'](r['PREFLIGHT'])})
        r['write_exclusive'](r['LIVE'], report)
    except Exception as error:
        stage.GetSessionLayer().ImportFromString(session)
        r['write_exclusive'](r['FAILURE'], dict(version=r['VERSION'], cohort=r['COHORT'],
            status='PREPARATION_FAILED_SESSION_ROLLED_BACK_NO_PLAY', error=type(error).__name__+': '+str(error),
            actual_static_or_drive_admission=False))
        raise
    omni.usd.get_context().get_selection().set_selected_prim_paths([r['ROBOT']], True)
    print('=== H4 REOPENED FRESH NOMINAL COHORT PREPARED (NO PLAY) ===')
    print('new result cohort:', r['COHORT'], '; recreated robot namespace:', r['ROBOT'])
    print('mass/body/joints:', actual['simulation_mass_kg'], len(actual['all_active_rigid_body_paths']),
        len(actual['live_new_robot_readback']['revolute_joint_names']))
    print('same material/gains/source seed/support/start/goal; old H3/H4 PASS/FAIL/HOLD preserved')
    print('saved:', r['LIVE'])
    print('STOPPED. F then Top View. New static is pending; old contact-failed route remains HOLD.')
    print('Do not Play/save/reset or run old static/path controllers. No motion was authorized by this setup.')


_prepare_h4_reopened_cohort()
