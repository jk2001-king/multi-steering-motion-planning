"""Append-only provenance handoff of completed nominal model observations.

The shared live runner expects an executor_model_screen packet beside its
reference. Construct it from the exact owned screen and independent audit,
not from a different plan's model outcome. No model rerun or simulator action
is performed, and existing packets are never replaced.
"""
from pathlib import Path

import hospital_revision_followup as followup

runtime = followup.runtime


def publish(plan_folder, model_report, *, controller_version):
    folder = Path(plan_folder).resolve(); report_path = Path(model_report).resolve()
    summary = runtime.read(folder/'summary.json')
    audit = runtime.read(folder/'independent_plan_audit.json')
    model = runtime.read(report_path); plan = Path(summary['plan']).resolve()
    if (not summary['plan_success'] or audit['verdict'] != 'PASS'
            or Path(model['plan']).resolve() != plan
            or model['plan_sha256'] != runtime.sha(plan)
            or audit['plan_sha256'] != runtime.sha(plan)
            or model['controller_version'] != controller_version):
        raise ValueError('Exact accepted reference, independent audit and owned controller screen required')
    cases = model['cases']
    if (len(cases) != 2 or sorted(c['integration_dt_s'] for c in cases) != [.010, .020]
            or any(c.get('new_PhysX_trials', 0) for c in cases)):
        raise ValueError('Exactly the two original nominal resolutions, no physical trials')
    inputs = {str(p): runtime.sha(p) for p in (plan, folder/'summary.json',
        folder/'independent_plan_audit.json', folder/'task_geometry.json',
        folder/'planning_occupancy.npz', report_path, Path(__file__).resolve())}
    outputs = {}
    for row in cases:
        path = Path(row['trace']).resolve()
        if runtime.sha(path) != row['trace_sha256']:
            raise ValueError('Actual saved nominal trace changed')
        outputs[str(path)] = row['trace_sha256']
    value = dict(version='EXACT_COMPLETED_MODEL_ARTIFACT_HANDOFF_NOT_NEW_EXPERIMENT_v1',
        plan=str(plan), plan_sha256=runtime.sha(plan), controller_version=controller_version,
        source_model_report=str(report_path), source_model_report_sha256=runtime.sha(report_path),
        both_nominal_resolutions_pass=model['both_nominal_resolutions_pass'],
        model_statuses=[c['model_status'] for c in cases],
        input_sha256=inputs, output_sha256=outputs, actual_motion_trials=0,
        nominal_screen_is_NOT_physical_drive_or_hold_PD_validation=True,
        independent_geometry_and_actual_own_static_checks_STILL_required=True)
    target = folder/'executor_model_screen/model_screen.json'
    if target.exists():
        if runtime.read(target) != value:
            raise FileExistsError('Existing model packet never replaced; use a distinct owned reference folder')
    else:
        runtime.evidence.write_exclusive(target, value)
    return target
