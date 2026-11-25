"""Capture actual in-memory Root/Session and exact field differences, GET ONLY."""


def _review_h4_root_difference():
    import json
    from pathlib import Path
    import runpy
    import omni.timeline
    import omni.usd

    helpers = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(helpers/'hospital_private_loader.py'))['load']
    r = load(helpers/'hospital_h4_root_difference_readback.py')
    if r['RESULT'].exists():
        raise RuntimeError('Actual Root difference report already exists; no overwrite or automatic follow-up')
    report = r['inspect'](omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface())
    r['OUTPUT'].mkdir(parents=True, exist_ok=False)
    with r['RESULT'].open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    diff = report['root_difference']
    print('=== H4 COMPLETED-STOP ACTUAL ROOT FIELD DIFF (GET ONLY) ===')
    print('actual timeline:', report['actual_timeline'])
    print('original source file unchanged:', report['source_USD_file_sha256'] == r['cohort'].SOURCE_SHA)
    print('Root text / parsed field equality:', diff['raw_root_text_equal'], diff['parsed_layer_field_values_equal'])
    print('changed fields:', diff['changed_field_count'])
    for row in diff['changed_fields'][:80]:
        print(row['path'], row['field'], row['diagnostic_classification'],
            'matches disk:', row['current_matches_unchanged_disk_authored_field'])
    print('full actual Root/Session text and values saved:', r['RESULT'])
    print('No timeline/scene/robot/source mutation or HOLD waiver. No v4/static/drive execution.')


_review_h4_root_difference()
