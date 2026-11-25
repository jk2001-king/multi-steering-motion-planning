"""GET ONLY v2: handle missing historical bounds of disabled colliders."""


def _inspect_h4_source_v2():
    import hashlib
    from pathlib import Path
    import runpy

    import omni.timeline
    import omni.usd

    scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_restart_review_v2.py')
    protocol = r['load_protocol']()
    if r['RESTART'].exists():
        raise RuntimeError('The v2 fresh-source inspection is already saved; no overwrite')
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None:
        raise RuntimeError('Open the unchanged hospital_experiment.usda first')
    root, session = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    before = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    report = r['inspect_fresh_source'](stage, *before)
    after = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    report['checks']['Root_Session_and_timeline_unchanged_by_readback'] = (
        before == after and root == stage.GetRootLayer().ExportToString()
        and session == stage.GetSessionLayer().ExportToString())
    report.update(version=r['VERSION'],
        status='FRESH_SOURCE_AUTHORED_READBACK_COMPLETE_SETUP_PENDING' if all(report['checks'].values()) else
            'HOLD_FRESH_SOURCE_COLLIDER_OR_ASSET_DIFFERENCES',
        old_actual_state_ended_due_to_application_close_NOT_restored=True,
        old_H4_FAIL_and_old_H3_archives_preserved=True,
        old_reader_KeyError_repair_is_NOT_a_physical_retry=True,
        prior_contact_review=str(r['original'].REVIEW), prior_contact_review_sha256=r['sha'](r['original'].REVIEW),
        source_stage=str(r['SOURCE']), source_file_sha256=r['sha'](r['SOURCE']),
        Root_text_sha256=hashlib.sha256(root.encode()).hexdigest(),
        Session_text_sha256=hashlib.sha256(session.encode()).hexdigest(),
        timeline=dict(zip(('playing', 'stopped', 'current_time_s'), before)),
        inspection_protocol=str(r['PROTOCOL']), inspection_protocol_sha256=r['sha'](r['PROTOCOL']),
        input_sha256={**protocol['input_sha256'], str(r['PROTOCOL']): r['sha'](r['PROTOCOL'])})
    r['verify'](report['input_sha256'])
    r['write_exclusive'](r['RESTART'], report)
    print('=== H4 REOPENED ORIGINAL SOURCE READBACK V2 (NO MOTION) ===')
    print('original enabled Hospital ROI colliders checked:', sum(
        row['recorded_classification'] == 'enabled_in_ROI' for row in report['collision_comparisons']))
    print('previously disabled colliders: current flags recorded, historical bounds NOT invented')
    print('previously disabled colliders currently enabled:', len(report['previously_disabled_colliders_currently_enabled']))
    print('known contact mesh unchanged:', report['checks']['contact_collider_authored_mesh_matches_recorded'])
    print('failed checks:', [key for key, ok in report['checks'].items() if not ok])
    print('STATUS:', report['status']); print('saved:', r['RESTART'])
    print('STOPPED. Readback ONLY. No reset/Play/reopen/save, H4 staging/static or drive.')


_inspect_h4_source_v2()
