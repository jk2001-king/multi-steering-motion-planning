"""Finite H4 common-controller pilot, using existing paired B2/P plans.

One fresh trial per method after the current queue has fully restored Pause.
Not a replacement of old failures, not a new final repetition campaign.
"""
import asyncio
import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_cusp_candidate as candidate
import hospital_revision_live as live
import hospital_revision_serialization as serialization
import hospital_revision_execution as execution
import hospital_revision_metrics as metrics
import hospital_revision_runner as runner

runtime = candidate.runtime
campaign = candidate.review.followup.campaign
planning = candidate.review.followup.planning
OUTPUT = candidate.review.followup.OUTPUT / 'common_cusp_pilot_v1'
PROTOCOL = OUTPUT / 'protocol.json'
CELL_IDS = ('H4_P_dynamics_full_envelope_s03', 'H4_B2_multimode_transition_cost_ramp_s03')


def export_protocol():
    parent = runtime.read(campaign.PROTOCOL)
    value = dict(version='PAIRED_H4_STOPPED_CUSP_CONTROLLER_DEVELOPMENT_v1',
        controller_version=candidate.VERSION,
        main_cells=[c for name in CELL_IDS for c in parent['main_cells'] if c['cell_id'] == name],
        tasks=parent['tasks'], common_speed_policy=parent['common_speed_policy'],
        parent_protocol=str(campaign.PROTOCOL), parent_protocol_sha256=runtime.sha(campaign.PROTOCOL),
        current_authority='P를 개선하고 공통 조건 변경 시 공정한 재검증을 자율 진행하라는 현재 사용자 요청',
        unchanged_frozen_paired_original_plans=True,
        changed_common_execution_controller=candidate.VERSION,
        changed_physics_actuation_limits_collision_or_final_success_criteria=False,
        new_trials_are_development_not_final_repetitions=True,
        previously_seen_conditions_not_held_out=True,
        maximum_static_trials_per_cell=1, maximum_route_trials_per_cell=1,
        automatic_physical_retry=False, old_failed_outcomes_never_replaced=True)
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Pilot definition already frozen')
        return PROTOCOL
    runtime.evidence.write_exclusive(PROTOCOL, value)
    sources = (Path(__file__), Path(candidate.__file__), Path(candidate.review.__file__),
        Path(live.__file__), Path(serialization.__file__), Path(metrics.__file__))
    pins = {str(p.resolve()): runtime.sha(p) for p in sources}
    runtime.evidence.write_exclusive(OUTPUT / 'sources.json', pins)
    runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip', pins)
    return PROTOCOL


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT / 'sources.json'))


def model_screen(cell_id):
    export_protocol(); verify_sources()
    if cell_id not in CELL_IDS:
        raise ValueError('Only declared matched pilot cells')
    folder = planning.OUTPUT / cell_id
    summary = runtime.read(folder / 'summary.json')
    plan_path = Path(summary['plan']); plan = runtime.read(plan_path)
    geometry = runtime.read(folder / 'task_geometry.json')
    audit = runtime.read(folder / 'independent_plan_audit.json')
    if not summary['plan_success'] or audit['verdict'] != 'PASS' or audit['plan_sha256'] != runtime.sha(plan_path):
        raise ValueError('Own frozen independently checked reference required')
    import numpy as np
    with np.load(folder / 'planning_occupancy.npz') as grid:
        occupied = grid['occupied'].copy(); origin = grid['origin_xy_m'].copy()
        resolution = float(grid['resolution_m'])
    output = OUTPUT / 'model' / cell_id
    output.mkdir(parents=True, exist_ok=False)
    rows_out = []
    for dt in (.020, .010):
        run, unused, design = candidate.model_binding(plan['geometry_profile'], geometry,
            plan, simulation_budget_s=300.)
        report, rows = run(plan, plan['rigid_body_parameter_bounds'], dt,
            occupied, origin, resolution, design)
        trace = output / (str(round(dt * 1000)) + 'ms.csv')
        with trace.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        report.update(trace=str(trace), trace_sha256=runtime.sha(trace))
        runtime.evidence.write_exclusive(trace.with_suffix('.json'),
            runtime.lc.normalize_runtime_evidence(report))
        rows_out.append(report)
        print('[CUSP CANDIDATE MODEL]', cell_id, dt, report['model_status'], report['error'], flush=True)
    verify_sources()
    result = dict(controller_version=candidate.VERSION, plan=str(plan_path),
        plan_sha256=runtime.sha(plan_path), cases=rows_out,
        both_nominal_resolutions_pass=all(r['model_status'] == 'MODEL_SCREEN_PASS' for r in rows_out),
        no_physical_success_or_RMS_claim=True, actual_trials=0)
    runtime.evidence.write_exclusive(output / 'model_screen.json', runtime.lc.normalize_runtime_evidence(result))
    return result


def bound_cell():
    """One private versioned factory/contract; keep original complete hosts."""
    seq = candidate.private_sequence()
    contract = runtime.bind(runtime.controlled_contract,
        sequence=SimpleNamespace(**dict(vars(seq),
            checked_route_lifecycle_contract=runtime.bind(seq.checked_route_lifecycle_contract,
                unified_geometry_lifecycle_contract=candidate.contract))))
    old = serialization.fixed_runtime()
    private = SimpleNamespace(**dict(vars(old), sequence=seq, controlled_contract=contract))
    private.route_packet = runtime.bind(old.route_packet, sequence=seq, controlled_contract=contract)
    private.context_for = lambda folder, actor, geometry: old.context_for(
        folder, actor + '_stopped_cusp_v1', geometry)
    initializer = runtime.bind(live.primary_initializer, runtime=private)()

    class PilotHost(live.PrimaryRouteHost):
        __init__ = initializer

        def save(self, report, rows):
            return super().save(dict(report, use_for_final_statistics=False,
                development_trial_NOT_final_repetition=True,
                controller_candidate_version=candidate.VERSION,
                old_negative_result_NOT_replaced=True), rows)

    base_packet = runtime.bind(live.task_route_packet, runtime=private)

    def packet(*args, **kwargs):
        path, value, plan = base_packet(*args, **kwargs)
        value = copy.deepcopy(value)
        value['input_sha256'].update(runtime.read(OUTPUT / 'sources.json'))
        value['input_sha256'][str(PROTOCOL)] = runtime.sha(PROTOCOL)
        return path, value, plan

    scoped = SimpleNamespace(**dict(vars(campaign), PROTOCOL=PROTOCOL, OUTPUT=OUTPUT))
    return runtime.bind(live.run_cell, campaign=scoped, runtime=private,
        checked_model_policy=runtime.bind(live.checked_model_policy, campaign=scoped),
        task_route_packet=packet, PrimaryRouteHost=PilotHost, __file__=__file__)


def policy_file():
    target = OUTPUT / 'diagnostic_model_policy.json'
    value = dict(version='COMMON_CUSP_PILOT_DIAGNOSTIC_MODEL_v1',
        user_authority='뭐가 가장 베스트의 방향성인지 판단하고 진행해',
        protocol_sha256=runtime.sha(PROTOCOL), model_is_prediction_not_actual_failure=True,
        independent_full_envelope_audit_required=True, actual_static_PASS_required=True,
        unchanged_actual_controller_limits_and_success_criteria=True,
        applies_to_all_methods_and_history_ablation=True, no_physical_retry=True,
        controller_implementation_changed_explicitly=candidate.VERSION,
        no_claim_of_equal_controller_to_original_campaign=True)
    if target.exists():
        if runtime.read(target) != value:
            raise ValueError('Pilot diagnostic policy frozen')
    else:
        runtime.evidence.write_exclusive(target, value)
    return target


async def after_current(parent, *, authorized=False):
    if authorized is not True:
        raise ValueError('Explicit current autonomous development authority required')
    import builtins
    result = await asyncio.shield(parent)
    if getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not parent:
        raise ValueError('Intervening trial replaced the owned queue')
    last = Path(result['last_actual_report'])
    if not execution.restored_boundary(runtime.read(last)):
        raise ValueError('Actual Pause/restoration required')
    verify_sources()
    current = asyncio.current_task()
    builtins.ijcas_hospital_latest_task_experiment = current
    builtins.ijcas_revision_primary_task = current
    outcomes = []
    try:
        for cell_id in CELL_IDS:
            screened = runtime.read(OUTPUT / 'model' / cell_id / 'model_screen.json')
            # P pilot must first pass both nominal MODEL resolutions. A baseline
            # model HOLD is diagnostic exactly as in the common original block.
            if cell_id == CELL_IDS[0] and not screened['both_nominal_resolutions_pass']:
                raise ValueError('Candidate P model feasibility not demonstrated; no pilot motion')
            if (OUTPUT / 'live' / cell_id).exists():
                raise FileExistsError('Consumed pilot trial; no repeat')
            verify_sources()
            outcome = await bound_cell()(cell_id, prior_result=last,
                model_policy_path=policy_file(), authorized=True)
            last = Path(outcome['report'])
            report = runtime.read(last)
            if not execution.restored_boundary(report):
                raise ValueError('Actual Pause/restoration missing; queue stops')
            actual = metrics.metrics(last) if outcome['actual_route_started'] else None
            value = dict(outcome, metrics=actual, controller_version=candidate.VERSION,
                development_not_final_repetition=True, old_outcome_not_replaced=True)
            runtime.evidence.write_exclusive(OUTPUT / 'outcomes' / (cell_id + '.json'), value)
            outcomes.append(value)
            print('[CUSP PILOT RESULT]', outcome, flush=True)
        result = dict(status='FINITE_PAIRED_COMMON_CUSP_PILOT_FINISHED',
            outcomes=outcomes, last_actual_report=str(last))
        runtime.evidence.write_exclusive(OUTPUT / 'completed.json', result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT / 'stopped.json', dict(
            error=type(error).__name__ + ': ' + str(error), outcomes=outcomes,
            last_actual_report=str(last), automatic_retry=False))
        raise


def dispatch_after_current():
    export_protocol(); verify_sources()
    for cell_id in CELL_IDS:
        runtime.read(OUTPUT / 'model' / cell_id / 'model_screen.json')
    source = ('import sys,types,asyncio,builtins\nfrom pathlib import Path\n'
        "parent=getattr(builtins,'ijcas_revision_primary_task',None)\n"
        "assert parent is not None,'Owned finite queue required'\n"
        "assert not hasattr(builtins,'ijcas_revision_common_cusp_pilot'),'Pilot already reserved'\n")
    for name in ('hospital_revision_cusp_candidate', 'hospital_revision_cusp_pilot'):
        path = Path(__file__).with_name(name + '.py').resolve()
        source += (f'p=Path({str(path)!r})\nm=types.ModuleType({name!r});m.__file__=str(p);sys.modules[m.__name__]=m\n'
            "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n")
    source += ("job=asyncio.ensure_future(m.after_current(parent,authorized=True))\n"
        "builtins.ijcas_revision_common_cusp_pilot=job\n"
        "print({'bounded_common_cusp_pilot_queued':True,'concurrent_motion':False})\n")
    runtime.evidence.write_exclusive(OUTPUT / 'dispatch.json', dict(source=source, automatic_retry=False))
    return runner.remote(source)
