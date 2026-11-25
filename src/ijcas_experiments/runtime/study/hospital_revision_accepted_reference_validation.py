"""Admit independently checked new references through the shared live tools.

No controller, stage, physics, speed, goal or safety policy is implemented
here. Reuse the already paired v4 host; substitute only the explicit owned
planning folders. All observations are development, never n5 replacements.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_hold_damping as control

runtime = control.runtime
OUTPUT = control.OUTPUT.parent / 'final_H3_native_reference_validation_v1'
PROTOCOL = OUTPUT / 'protocol.json'
PLAN_FOLDERS = {'H3_P_dynamics_full_envelope_s01': control.OUTPUT.parent /
    'final_beam_strict_goal_rank_v3_distributed_pairs/planning/carry/H3_P_dynamics_full_envelope_s01'}
PREDECESSOR_POLICY = 'six-cell paired v4 block must finish with all three P cases complete'


class ReferenceFolders:
    def __init__(self, folders):
        self.folders = {name: Path(folder) for name, folder in folders.items()}

    def __truediv__(self, name):
        if name not in self.folders:
            raise ValueError('Only explicitly declared independently audited references')
        return self.folders[name]


def planning_namespace():
    return SimpleNamespace(**dict(vars(control.base.shared.planning),
        OUTPUT=ReferenceFolders(PLAN_FOLDERS)))


def export_protocol():
    original = runtime.read(control.base.shared.campaign.PROTOCOL)
    inputs = {}
    for name, folder in PLAN_FOLDERS.items():
        summary = runtime.read(folder/'summary.json'); audit = runtime.read(folder/'independent_plan_audit.json')
        plan = Path(summary['plan'])
        if not summary['plan_success'] or audit['verdict'] != 'PASS' or audit['plan_sha256'] != runtime.sha(plan):
            raise ValueError('Own unchanged independently audited plan required')
        for path in (plan, folder/'summary.json', folder/'independent_plan_audit.json',
                folder/'task_geometry.json', folder/'planning_occupancy.npz'):
            inputs[str(path.resolve())] = runtime.sha(path)
    value = dict(version='OWN_ACCEPTED_NATIVE_REFERENCE_COMMON_V4_ACTUAL_VALIDATION_v1',
        main_cells=[c for c in original['main_cells'] if c['cell_id'] in PLAN_FOLDERS],
        tasks=original['tasks'], common_speed_policy=original['common_speed_policy'],
        controller_version=control.VERSION, reference_inputs_sha256=inputs,
        common_controller_candidate=control.hold_implementation().damping_binding,
        predecessor=PREDECESSOR_POLICY,
        actual_limits_and_completion_criteria_unchanged=True,
        own_two_resolution_nominal_screen_and_actual_static_PASS_required=True,
        hold_PD_not_represented_by_nominal_motion_model=True,
        one_fresh_source_one_static_conditional_one_route_per_reference=True,
        old_failed_attempts_and_original_n5_never_replaced=True,
        development_NOT_final_repetition=True, automatic_physical_retry=False,
        photo_deferred_by_author=True, no_manuscript_or_sensor_changes=True)
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Prospective reference validation definition frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        files = (Path(__file__), Path(control.__file__), Path(control.base.__file__),
            Path(control.base.shared.__file__), Path(control.base.goal.__file__),
            *getattr(control, 'extra_source_paths', ()))
        pins = {str(p.resolve()): runtime.sha(p) for p in files}
        runtime.evidence.write_exclusive(OUTPUT/'sources.json', pins)
        runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip', pins)
    return PROTOCOL


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'sources.json'))
    runtime.retained.runtime.task.verify_pins(runtime.read(PROTOCOL)['reference_inputs_sha256'])


def model_binding(*args, **kwargs):
    run, execution, design = control.base.model_binding(*args, **kwargs)
    def measured(*a, **k):
        value, rows = run(*a, **k)
        value.update(controller_version=control.VERSION,
            nominal_model_has_NO_captured_hold_PD_dynamics=True)
        return value, rows
    return measured, execution, design


def model_screen(name):
    shared = SimpleNamespace(**dict(vars(control.base.shared), planning=planning_namespace()))
    return runtime.bind(control.base.model_screen, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
        VERSION=control.VERSION, CELL_IDS=tuple(PLAN_FOLDERS), shared=shared, export_protocol=export_protocol,
        verify_sources=verify_sources, model_binding=model_binding)(name)


def bound_cell():
    cell = runtime.bind(control.bound_cell, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL)()
    return runtime.bind(cell, planning=planning_namespace())


def policy_file():
    # The shared base expects a candidate namespace, unlike the intervening
    # convenience factory. Bind that exact interface, without a global patch.
    return runtime.bind(control.base.shared.policy_file, OUTPUT=OUTPUT,
        PROTOCOL=PROTOCOL, candidate=control.candidate())()


def predecessor_admitted(previous):
    rows = [r for r in previous.get('results', []) if '_P_' in r['cell_id']]
    return (previous['status'] == 'FINITE_PAIRED_REMAINING_STOP_VALIDATION_FINISHED'
        and len(rows) == 3 and all(r['actual_verdict'] == 'PASS' for r in rows))


async def after_current(parent, *, authorized=False):
    if authorized is not True:
        raise ValueError('Current explicit autonomous validation authority required')
    import builtins
    previous = await asyncio.shield(parent)
    if getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not parent:
        raise ValueError('Exact owned predecessor required; no intervening motion')
    last = Path(previous['last_actual_report'])
    if not predecessor_admitted(previous):
        value = dict(status='NOT_ADMITTED_PREDECESSOR_P_BLOCK_NOT_COMPLETE',
            physical_trials=0, last_actual_report=str(last))
        runtime.evidence.write_exclusive(OUTPUT/'not_admitted.json', value)
        return value
    verify_sources()
    if not control.base.shared.execution.restored_boundary(runtime.read(last)):
        raise ValueError('Actual Pause and original gains/targets restoration required')
    policy = policy_file(); results = []
    current = asyncio.current_task()
    builtins.ijcas_hospital_latest_task_experiment = current
    builtins.ijcas_revision_primary_task = current
    try:
        for name in PLAN_FOLDERS:
            screen = runtime.read(OUTPUT/'model'/name/'model_screen.json')
            if '_P_' in name and not screen['both_nominal_resolutions_pass']:
                raise ValueError('Both original nominal resolutions must pass before motion')
            verify_sources()
            outcome = await bound_cell()(name, prior_result=last, authorized=True,
                model_policy_path=policy)
            last = Path(outcome['report'])
            if not control.base.shared.execution.restored_boundary(runtime.read(last)):
                raise ValueError('Actual restoration missing; no continuation')
            value = dict(outcome, controller_version=control.VERSION,
                development_NOT_final_repetition=True,
                metrics=control.base.shared.metrics.metrics(last) if outcome['actual_route_started'] else None)
            runtime.evidence.write_exclusive(OUTPUT/'outcomes'/(name+'.json'), value)
            results.append(value)
            if outcome['actual_verdict'] != 'PASS':
                stopped = dict(status='MEASURED_NEW_REFERENCE_NONCOMPLETION_NO_AUTO_RETRY',
                    results=results, last_actual_report=str(last))
                runtime.evidence.write_exclusive(OUTPUT/'stopped.json', stopped)
                return stopped
        value = dict(status='OWN_ACCEPTED_REFERENCES_ACTUAL_VALIDATION_FINISHED',
            results=results, last_actual_report=str(last), automatic_physical_retry=False)
        runtime.evidence.write_exclusive(OUTPUT/'completed.json', value)
        return value
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT/'workflow_error.json', dict(
            error=type(error).__name__+': '+str(error), results=results,
            last_actual_report=str(last), automatic_physical_retry=False))
        raise


def dispatch_after_current():
    export_protocol(); verify_sources()
    for name in PLAN_FOLDERS:
        if '_P_' in name and not runtime.read(OUTPUT/'model'/name/'model_screen.json')['both_nominal_resolutions_pass']:
            raise ValueError('Both nominal model resolutions required before reservation')
    source = ("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "parent=getattr(builtins,'ijcas_revision_hold_damping_v4_policy_checked',None)\n"
        "assert parent is not None,'Owned v4 paired predecessor required'\n"
        "assert not hasattr(builtins,'ijcas_revision_H3_accepted_reference'),'Already reserved'\n"
        f"p=Path({str(Path(__file__).resolve())!r})\n"
        "m=types.ModuleType('hospital_revision_accepted_reference_validation');m.__file__=str(p)\n"
        "sys.modules[m.__name__]=m\nexec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.after_current(parent,authorized=True))\n"
        "builtins.ijcas_revision_H3_accepted_reference=job\n"
        "print({'checked_native_H3_reference_waiting':True,'concurrent_motion':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'reservation.json', dict(source=source,
        maximum_routes=1, automatic_retry=False, no_concurrent_motion=True))
    return control.base.shared.runner.remote(source)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('models', 'dispatch'))
    args = parser.parse_args()
    if args.action == 'models':
        for name in PLAN_FOLDERS:
            model_screen(name)
    else:
        print(dispatch_after_current())
