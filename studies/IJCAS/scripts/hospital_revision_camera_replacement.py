"""One explicitly approved replacement of an externally interrupted B1 trial.

Reuse the frozen physical runner, plan, source perturbation, controller and
criteria. Only output namespace, fresh actor name and provenance are adapted.
Never replace an algorithmic failure, overwrite the interrupted observation,
add an independent condition, or automatically retry the replacement.
"""
import argparse
import asyncio
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import hospital_revision_camera_retire as primary

experiment = primary.experiment
runtime = primary.runtime
CELL = primary.INITIAL
OUTPUT = experiment.OUTPUT / 'technical_camera_replacement_r01'
KEY = 'ijcas_revision_n3_camera_replacement_r01'
AUTHORITY = '사용자 승인: 같은 조건 대체 실행 1회 진행'


def cell_definition():
    block = experiment.campaign()
    return copy.deepcopy(next(c for c in runtime.read(block.EXECUTION_PROTOCOL)['main_cells'] if c['cell_id'] == CELL))


def bound_replacement():
    block = experiment.campaign()
    cell = cell_definition()
    original = block.bound_cell(cell)
    old_runtime = original.__globals__['runtime']
    old_context = old_runtime.context_for

    def context(folder, actor, geometry):
        return old_context(folder, actor + '_camera_replacement_r01', geometry)

    parent = original.__globals__['PrimaryRouteHost']

    class ReplacementHost(parent):
        def save(self, report, rows):
            return super().save(dict(report,
                technical_replacement_of_cell_id=CELL,
                technical_replacement_authority=AUTHORITY,
                extra_independent_repetition=False,
                original_interrupted_record_preserved=True), rows)

    private_runtime = SimpleNamespace(**dict(vars(old_runtime), context_for=context))
    campaign = SimpleNamespace(**dict(vars(original.__globals__['campaign']), OUTPUT=OUTPUT))
    return runtime.bind(original, runtime=private_runtime, campaign=campaign,
        PrimaryRouteHost=ReplacementHost, __file__=__file__)


def prepare():
    block = experiment.campaign()
    block.verify_sources()
    old_outcome = block.OUTPUT / 'outcomes' / (CELL + '.json')
    observation = runtime.read(old_outcome)
    proof = Path(observation['separate_camera_boundary_GET'])
    raw = runtime.read(observation['report'])
    packet = runtime.read(raw['protocol'])
    primary.check_camera_report(raw, packet)
    if (observation.get('technical_interruption') is not True
            or observation.get('completion_evaluable') is not False
            or runtime.read(proof)['original_report_sha256'] != runtime.sha(observation['report'])):
        raise ValueError('Only the confirmed external camera interruption may be replaced')
    cell = cell_definition()
    folder = block.folder_for(CELL)
    inputs = [Path(__file__).resolve(), old_outcome, proof,
        Path(observation['report']), Path(raw['protocol']),
        block.PROTOCOL, block.EXECUTION_PROTOCOL, folder / 'summary.json',
        folder / 'independent_plan_audit.json', folder / 'task_geometry.json',
        folder / 'planning_occupancy.npz', Path(runtime.read(folder / 'summary.json')['plan'])]
    value = dict(authority=AUTHORITY, original_declared_cell=cell,
        original_technical_outcome=str(old_outcome), original_camera_proof=str(proof),
        original_technical_record_NOT_deleted_or_promoted=True,
        selected_comparison_conditions_remain_three=True,
        maximum_replacement_static_trials=1, maximum_replacement_route_trials=1,
        automatic_retry=False, no_new_planning_or_tuning=True,
        same_plan_start_goal_joint_draw_controller_speed_physics_guards_and_budgets=True,
        only_output_namespace_actor_name_and_provenance_adapted=True,
        start_only_after_original_all78_queue_finished=True,
        input_sha256={str(p.resolve()): runtime.sha(p) for p in inputs})
    runtime.evidence.write_exclusive(OUTPUT / 'definition.json', value)
    runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip', value['input_sha256'])
    return OUTPUT / 'definition.json'


def verify():
    experiment.campaign().verify_sources()
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT / 'definition.json')['input_sha256'])


async def run(parent, *, authorized=False):
    import builtins
    last = None
    try:
        verify()
        block = experiment.campaign()
        if (authorized is not True or not parent.done() or parent.exception() is not None
                or getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not parent
                or not (block.OUTPUT / 'all_steps_completed.json').is_file()):
            raise ValueError('Complete exact primary owner and explicit replacement authority required')
        complete = runtime.read(block.OUTPUT / 'completed.json')
        if len(complete['results']) != 78:
            raise ValueError('Every original declared outcome must be retained first')
        last = Path(complete['last_actual_report'])
        raw = runtime.read(last)
        if not block.safe_restored_boundary(raw):
            last_cell = runtime.read(raw['protocol'])['primary_cell']
            await runtime.bind(primary.previous.prove_paused_boundary, OUTPUT=OUTPUT)(last, last_cell)
        job = asyncio.current_task()
        builtins.ijcas_hospital_latest_task_experiment = job
        builtins.ijcas_revision_primary_task = job
        runtime.evidence.write_exclusive(OUTPUT / 'execution_reservation.json', dict(
            authority=AUTHORITY, original_technical_record_preserved=True,
            previous_actual_report=str(last), maximum_additional_routes=1,
            no_automatic_retry=True))
        print('[N3 CAMERA REPLACEMENT START]', CELL, flush=True)
        value = await bound_replacement()(CELL, prior_result=last, authorized=True,
            model_policy_path=block.policy_file())
        last = Path(value['report'])
        raw = runtime.read(last)
        if not block.safe_restored_boundary(raw):
            await runtime.bind(primary.previous.prove_paused_boundary, OUTPUT=OUTPUT)(last, cell_definition())
        metrics = block.metrics.metrics(last) if value['actual_route_started'] else None
        result = dict(value, metrics=metrics,
            technical_replacement_of_cell_id=CELL, authority=AUTHORITY,
            independent_repetition_increment=0,
            original_technical_outcome=str(block.OUTPUT / 'outcomes' / (CELL + '.json')),
            valid_same_condition_replacement=True,
            experiment_role=cell_definition()['experiment_role'],
            predictive_history_policy=cell_definition()['predictive_history_policy'],
            controller_version=block.control.VERSION, final_fixed_version_trial=True)
        runtime.evidence.write_exclusive(OUTPUT / 'outcome.json', result)
        runtime.evidence.write_exclusive(OUTPUT / 'completed.json', dict(
            status='ONE_APPROVED_TECHNICAL_REPLACEMENT_FINISHED', outcome=str(OUTPUT / 'outcome.json'),
            actual_verdict=value['actual_verdict'], last_actual_report=str(last),
            original_camera_interruption_preserved=True, automatic_retry=False))
        print('[N3 CAMERA REPLACEMENT RESULT]', value['actual_verdict'], flush=True)
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT / 'stopped.json', dict(
            error=type(error).__name__ + ': ' + str(error),
            last_actual_report=str(last) if last else None, automatic_retry=False))
        raise


def dispatch():
    verify()
    if not (experiment.OUTPUT / 'all_steps_completed.json').exists():
        raise ValueError('Do not interrupt the original queue')
    source = ("import importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        "import hospital_revision_camera_replacement as _replacement\n"
        "_replacement=importlib.reload(_replacement)\n"
        f"assert not hasattr(builtins,{KEY!r}),'One replacement already reserved'\n"
        f"builtins.{KEY}=asyncio.ensure_future(_replacement.run(builtins.{primary.KEY},authorized=True))\n"
        "print({'one_approved_camera_replacement_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT / 'dispatch.json', dict(source=source, authority=AUTHORITY))
    return experiment.original.predecessor.control.base.shared.runner.remote(source)


def watch():
    verify()
    while not (experiment.OUTPUT / 'all_steps_completed.json').is_file():
        if (primary.OUTPUT / 'stopped.json').exists():
            raise ValueError('Original queue unresolved; no replacement dispatch')
        time.sleep(15.)
    print(dispatch(), flush=True)
    while not (OUTPUT / 'completed.json').is_file():
        if (OUTPUT / 'stopped.json').exists():
            raise ValueError('Replacement interrupted; no automatic retry')
        time.sleep(15.)
    print('[N3 CAMERA REPLACEMENT COMPLETE]', OUTPUT / 'completed.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'watch', 'dispatch'))
    args = parser.parse_args()
    result = {'prepare': prepare, 'watch': watch, 'dispatch': dispatch}[args.action]()
    if result is not None:
        print(result, flush=True)
