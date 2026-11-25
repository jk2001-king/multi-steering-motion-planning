"""Read-only fresh-source restart boundary. NO Play/Stop/reset/staging."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _inspect_h4_reopened_source():
    import hashlib
    from pathlib import Path
    import runpy

    import omni.timeline
    import omni.usd

    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_restart_review.py')
    review = r['load_review']()
    if r['RESTART'].exists():
        raise RuntimeError('Fresh-source readback already saved; no overwrite or repeated setup')
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None:
        raise RuntimeError('Open the original hospital_experiment.usda first')
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
        prior_contact_review=str(r['REVIEW']), prior_contact_review_sha256=r['sha'](r['REVIEW']),
        source_stage=str(r['SOURCE']), source_file_sha256=r['sha'](r['SOURCE']),
        Root_text_sha256=hashlib.sha256(root.encode()).hexdigest(),
        Session_text_sha256=hashlib.sha256(session.encode()).hexdigest(),
        timeline=dict(zip(('playing', 'stopped', 'current_time_s'), before)),
        input_sha256={**review['input_sha256'], str(r['REVIEW']): r['sha'](r['REVIEW'])})
    r['verify'](report['input_sha256'])
    r['write_exclusive'](r['RESTART'], report)
    print('=== H4 REOPENED ORIGINAL SOURCE READBACK (NO MOTION) ===')
    print('source SHA:', report['source_file_sha256'])
    print('old live solver/contact state restored:', False)
    print('known low contact collider unchanged:', report['checks']['contact_collider_authored_mesh_matches_recorded'])
    print('failed checks:', [key for key, ok in report['checks'].items() if not ok])
    print('STATUS:', report['status']); print('saved:', r['RESTART'])
    print('STOPPED. Source inspection ONLY. No H4 staging/static/drive or old-controller admission.')


_inspect_h4_reopened_source()
