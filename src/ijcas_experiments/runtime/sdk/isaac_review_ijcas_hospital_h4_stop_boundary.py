"""GET ONLY after archived-H3 / immediate Stop-state-query preparation failure.

No Play, Stop, pause, time setting, commit, update, reset, reopen or staging.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _review_h4_stop_boundary():
    import json
    from pathlib import Path
    import runpy
    import omni.timeline
    import omni.usd

    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_stop_boundary_readback.py')
    if r['RESULT'].exists():
        raise RuntimeError('A live boundary readback already exists; no overwrite or automatic follow-up')
    report = r['inspect'](omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface())
    r['OUTPUT'].mkdir(parents=True, exist_ok=False)
    with r['RESULT'].open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 ARCHIVED-H3 / DEFERRED STOP LIVE BOUNDARY READBACK (GET ONLY) ===')
    print('actual timeline:', report['actual_timeline'])
    print('H3 archive: verified 13 bodies / 12 joints; no physical-state restore claim')
    print('new H4 actor and marker absent:', report['checks']['no_new_H4_actor_or_marker_created'])
    print('protected source / Session unchanged:', report['checks']['root_layer_matches_actual_pre_Stop_archive'],
        report['checks']['protected_session_matches_actual_pre_Stop_archive'])
    print('failed checks:', [key for key, ok in report['checks'].items() if not ok])
    print('STATUS:', report['status']); print('saved:', r['RESULT'])
    print('No timeline/robot/material change. Do not rerun v4 or start static/drive; continuation requires review.')


_review_h4_stop_boundary()
