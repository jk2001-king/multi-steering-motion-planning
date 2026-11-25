"""Finite paired validation of unresolved stop/ACK failures.

Reuse the existing common compliant-hold adapter (1000 instead of10000
Nm/rad) and the independently tested v2 goal/cusp handoff. No planner,
speed, damping, torque, geometry, contact, safety or success limit changes.
All prior outcomes stay intact. These are seen-condition development pairs,
not replacements of the original matched n5 repetitions.
"""
import asyncio
import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import hospital_revision_goal_handoff as goal
import hospital_revision_cusp_pilot as shared

runtime = goal.runtime
VERSION = 'COMMON1330_ACTUAL_GOAL_v2_COMPLIANT_HOLD1000_v3'
OUTPUT = shared.OUTPUT.parent / 'remaining_stop_ACK_paired_v3'
PROTOCOL = OUTPUT / 'protocol.json'
PAIR_IDS = ('H2_s01', 'H4_s02', 'H4_s04')
CELL_IDS = tuple(task + '_' + method + '_' + repeat
    for paired in PAIR_IDS for task, repeat in (paired.split('_'),)
    for method in (shared.planning.campaign.planning.P, shared.planning.campaign.planning.B2))


def contract(profile, *, simulation_budget_s, wall_budget_s=900.):
    value = copy.deepcopy(goal.contract(profile, simulation_budget_s=simulation_budget_s,
        wall_budget_s=wall_budget_s))
    value.update(version=VERSION, parent_controller_version=goal.VERSION,
        common_transient_drive_position_hold_kp_Nm_rad=1000.,
        original_transient_drive_position_hold_kp_Nm_rad=10000.,
        hold_adapter_binding=goal.parent.sequence._compliant_hold_implementation().compliance_binding,
        same_gain_candidate_for_all_methods=True,
        original_damping_max_efforts_passive_joints_and_final_restore_unchanged=True,
        same_original_measured_2s_route_prehold_and_first_fixed5s_final_hold=True,
        gain_change_requires_new_actual_validation_NOT_counterfactual_replay=True)
    return value


def controller(plan, geometry, *, profile, simulation_budget_s, start_time_s=0., wall_budget_s=900.):
    value = goal.controller(plan, geometry, profile=profile,
        simulation_budget_s=simulation_budget_s, start_time_s=start_time_s, wall_budget_s=wall_budget_s)
    value.contract = contract(profile, simulation_budget_s=simulation_budget_s, wall_budget_s=wall_budget_s)
    return value


def private_sequence():
    value = goal.private_sequence()
    return SimpleNamespace(**dict(vars(value), unified_geometry_lifecycle_controller=controller,
        unified_geometry_lifecycle_contract=contract))


def model_binding(*args, **kwargs):
    # Existing kinematic/response model has no captured-wheel contact-force
    # integration. Do not claim it validates this changed physical gain.
    run, execution, design = goal.model_binding(*args, **kwargs)
    def measured(*a, **k):
        value, rows = run(*a, **k)
        value.update(controller_version=VERSION, nominal_model_does_NOT_model_hold_gain=True,
            actual_static_and_route_stop_response_STILL_required=True)
        return value, rows
    return measured, execution, design


def candidate():
    return SimpleNamespace(**dict(vars(shared.candidate), VERSION=VERSION,
        private_sequence=private_sequence, contract=contract, model_binding=model_binding))


def export_protocol():
    parent = runtime.read(shared.campaign.PROTOCOL)
    cells = [copy.deepcopy(next(c for c in parent['main_cells'] if c['cell_id'] == name))
        for name in CELL_IDS]
    value = dict(version=VERSION, main_cells=cells, tasks=parent['tasks'],
        common_speed_policy=parent['common_speed_policy'],
        unchanged_frozen_original_reference_plans=True,
        sole_additional_actual_change='common transient drive position hold stiffness10000->1000Nm/rad',
        parent_controller_version=goal.VERSION,
        unchanged_actual_acceptance_position_m=.030, unchanged_actual_acceptance_heading_deg=2.,
        unchanged_actual_acceptance_speed_m_s=.005, unchanged_first_fixed_hold_s=5.,
        unchanged_damping_effort_limits_mass_material_geometry_and_collision_policy=True,
        common_candidate_applied_to_B2_and_P=True,
        main_unresolved_conditions=list(PAIR_IDS), paired_B2_controls_included=True,
        source_static_gain_policy='unchanged original10000 support screen for both methods',
        own_route_prehold_gain_policy='new1000; own original measured2s prehold mandatory',
        model_hold_is_diagnostic_not_a_physical_failure=True,
        independent_full_envelope_audit_and_own_actual_static_PASS_required=True,
        known_condition_development_NOT_independent_final_repetition=True,
        original_main_n5_and_all_FAIL_HOLD_records_preserved=True,
        no_successful_original_P_condition_repeated=True,
        maximum_static_trials_per_cell=1, maximum_route_trials_per_cell=1,
        automatic_physical_retry=False, stop_queue_on_any_P_noncompletion=True,
        authority='현재 사용자: 남은P실패 보완을 자율 진행하되 교수님 피드백과 공정 비교 유지',
        photo_deferred_by_author=True, manuscript_modified=False)
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('This finite paired definition is frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        sources = (Path(__file__), Path(goal.__file__), Path(shared.__file__),
            Path(shared.live.__file__), Path(shared.serialization.__file__),
            Path(goal.parent.sequence.__file__))
        pins = {str(p.resolve()): runtime.sha(p) for p in sources}
        runtime.evidence.write_exclusive(OUTPUT / 'sources.json', pins)
        runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip', pins)
    return PROTOCOL


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT / 'sources.json'))


def model_screen(cell_id):
    export_protocol(); verify_sources()
    if cell_id not in CELL_IDS:
        raise ValueError('Only the declared failed P conditions and paired B2 controls')
    folder = shared.planning.OUTPUT / cell_id
    summary = runtime.read(folder / 'summary.json')
    audit = runtime.read(folder / 'independent_plan_audit.json')
    plan_path = Path(summary['plan']); plan = runtime.read(plan_path)
    if (not summary['plan_success'] or audit['verdict'] != 'PASS'
            or audit['plan_sha256'] != runtime.sha(plan_path)):
        raise ValueError('Exact own independently audited plan required')
    output = OUTPUT / 'model' / cell_id
    if (output / 'model_screen.json').exists():
        value = runtime.read(output / 'model_screen.json')
        if value['plan_sha256'] != runtime.sha(plan_path):
            raise ValueError('Frozen model input changed')
        for row in value['cases']:
            if runtime.sha(row['trace']) != row['trace_sha256']:
                raise ValueError('Frozen model trace changed')
        return value
    output.mkdir(parents=True, exist_ok=False)
    geometry = runtime.read(folder / 'task_geometry.json')
    with np.load(folder / 'planning_occupancy.npz') as grid:
        occupied, origin = grid['occupied'].copy(), grid['origin_xy_m'].copy()
        resolution = float(grid['resolution_m'])
    task = cell_id.split('_')[0]
    budget = runtime.read(PROTOCOL)['tasks'][task]['simulation_budget_s']
    reports = []
    for dt in (.020, .010):
        run, unused, design = model_binding(plan['geometry_profile'], geometry, plan,
            simulation_budget_s=budget)
        report, rows = run(plan, plan['rigid_body_parameter_bounds'], dt,
            occupied, origin, resolution, design)
        trace = output / (str(round(dt * 1000)) + 'ms.csv')
        with trace.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        report.update(trace=str(trace), trace_sha256=runtime.sha(trace))
        reports.append(runtime.lc.normalize_runtime_evidence(report))
        print('[REMAINING STOP MODEL]', cell_id, dt, report['model_status'], report['error'], flush=True)
    value = dict(controller_version=VERSION, plan=str(plan_path), plan_sha256=runtime.sha(plan_path),
        cases=reports, both_nominal_resolutions_pass=all(r['model_status'] == 'MODEL_SCREEN_PASS' for r in reports),
        gain_stability_NOT_predicted_by_nominal_model=True, actual_trials=0)
    runtime.evidence.write_exclusive(output / 'model_screen.json', value)
    return value


def bound_cell():
    call = runtime.bind(shared.bound_cell, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, candidate=candidate())()
    prior = call.__globals__['runtime']
    original_context = prior.context_for
    private = SimpleNamespace(**dict(vars(prior), context_for=lambda folder, actor, geometry:
        original_context(folder, actor + '_compliant1000_v3', geometry)))
    parent = call.__globals__['PrimaryRouteHost']
    class CommonCompliantHost(parent):
        async def preflight(self):
            value = await super().preflight()
            goal.parent.sequence.bind_compliant_hold_adapter(self)
            return dict(value, explicitly_bound_transient_drive_hold_kp_Nm_rad=1000.,
                original_damping_effort_limits_passive_states_and_final_restore_unchanged=True)
    return runtime.bind(call, runtime=private, PrimaryRouteHost=CommonCompliantHost)


def policy_file():
    return runtime.bind(shared.policy_file, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, candidate=candidate())()


async def run(parent, *, authorized=False):
    if authorized is not True:
        raise ValueError('Current autonomous paired development authority required')
    import builtins
    previous = await asyncio.shield(parent)
    if getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not parent:
        raise ValueError('Current exact restored parent required')
    last = Path(previous['last_actual_report'])
    if not shared.execution.restored_boundary(runtime.read(last)):
        raise ValueError('Actual Pause/restoration required')
    verify_sources()
    current = asyncio.current_task()
    builtins.ijcas_hospital_latest_task_experiment = current
    builtins.ijcas_revision_primary_task = current
    results = []
    try:
        for name in CELL_IDS:
            screen = runtime.read(OUTPUT / 'model' / name / 'model_screen.json')
            if '_P_' in name and not screen['both_nominal_resolutions_pass']:
                raise ValueError('Own P nominal models not both PASS; no speculative motion')
            verify_sources()
            outcome = await bound_cell()(name, prior_result=last, authorized=True,
                model_policy_path=policy_file())
            last = Path(outcome['report'])
            if not shared.execution.restored_boundary(runtime.read(last)):
                raise ValueError('Actual Pause/restoration missing; stop queue')
            result = dict(outcome, controller_version=VERSION, development_NOT_final_repetition=True,
                metrics=shared.metrics.metrics(last) if outcome['actual_route_started'] else None)
            runtime.evidence.write_exclusive(OUTPUT / 'outcomes' / (name + '.json'), result)
            results.append(result)
            print('[REMAINING STOP RESULT]', result, flush=True)
            if '_P_' in name and outcome['actual_verdict'] != 'PASS':
                value = dict(status='MEASURED_P_NONCOMPLETION_REQUIRES_DIAGNOSIS_NO_AUTO_RETRY',
                    results=results, last_actual_report=str(last))
                runtime.evidence.write_exclusive(OUTPUT / 'stopped_on_P.json', value)
                return value
        value = dict(status='FINITE_PAIRED_REMAINING_STOP_VALIDATION_FINISHED', results=results,
            last_actual_report=str(last), automatic_physical_retry=False)
        runtime.evidence.write_exclusive(OUTPUT / 'completed.json', value)
        return value
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT / 'workflow_error.json', dict(
            error=type(error).__name__ + ': ' + str(error), results=results,
            last_actual_report=str(last), automatic_physical_retry=False))
        raise


def dispatch():
    export_protocol(); verify_sources()
    for name in CELL_IDS:
        value = runtime.read(OUTPUT / 'model' / name / 'model_screen.json')
        if '_P_' in name and not value['both_nominal_resolutions_pass']:
            raise ValueError('Model validation first')
    if (OUTPUT / 'live').exists():
        raise ValueError('Consumed finite physical block; no redispatch')
    source = ("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "parent=getattr(builtins,'ijcas_revision_actual_goal_validation',None)\n"
        "assert parent is not None and parent.done() and not parent.exception(),'Completed exact parent required'\n"
        "assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is parent,'No intervening motion'\n"
        "assert not hasattr(builtins,'ijcas_revision_remaining_stop_v3'),'Already reserved'\n"
        f"p=Path({str(Path(__file__).resolve())!r})\n"
        "m=types.ModuleType('hospital_revision_remaining_validation');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.run(parent,authorized=True))\n"
        "builtins.ijcas_revision_remaining_stop_v3=job\n"
        "print({'paired_remaining_stop_validation_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT / 'reservation.json', dict(source=source,
        maximum_pairs=3, old_failures_preserved=True, automatic_physical_retry=False))
    return shared.runner.remote(source)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('models', 'dispatch'))
    args = parser.parse_args()
    if args.action == 'models':
        for cell_id in CELL_IDS:
            model_screen(cell_id)
    else:
        print(dispatch())
