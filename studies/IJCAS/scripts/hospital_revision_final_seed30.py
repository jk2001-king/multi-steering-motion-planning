"""Pre-FIRST-drive seed-budget amendment; reuse the shared n3 campaign.

Only P's B0 seed slice changes (20 -> 30 s inside the same 45 s portfolio).
All 24 P/history plans are regenerated. All 54 unchanged baseline attempts,
including failures, are linked with verified provenance, never selected for
success. No planner, physics controller, guard, metric or SDK loop is copied.
The inspected n3 inputs are NOT an untouched confirmatory test set.
"""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace
import xml.etree.ElementTree as ET

import hospital_revision_final_campaign as original
import hospital_revision_final_manager_resume as envelope_manager

runtime = original.runtime
OUTPUT = original.OUTPUT.parent/'revision_final_n3_seed30_v2_20261122'
AUTHORITY = '2026-11-22 사용자: 추천 탐색 배분 보완 승인; 3조건으로 계속 진행'
KEY = 'ijcas_revision_final_n3_seed30_v2'
VERSION = 'FINAL_N3_COMMON_V7_UNIFORM_P_B0_SEED30_PREPHYSICAL_AMENDMENT_v2'


def amended_definition():
    value = copy.deepcopy(runtime.read(original.PROTOCOL))
    value.update(version=VERSION, user_authority=AUTHORITY,
        amendment=dict(sole_algorithm_change='P B0 seed slice 20 to 30 seconds within common45/180s',
            applied_to_all_tasks_and_history_ablation=True,
            original_protocol=str(original.PROTOCOL), original_protocol_sha256=runtime.sha(original.PROTOCOL),
            original_P_planning_failure_preserved='H3_P_dynamics_full_envelope_f03',
            conditions_inspected_before_amendment=True, untouched_held_out_confirmation=False,
            physical_trials_before_amendment=0, baseline_attempts_reused_including_failures=54,
            P_and_history_planning_attempts_regenerated=24, no_additional_actual_comparison_cells=True),
        fresh_draws_NOT_seen_development_replays=False,
        draws_originally_generated_before_v1_planning_outcomes=True,
        treatment_frozen_before_new_outcomes=False,
        v2_frozen_before_v2_planning_and_ALL_actual_trials=True,
        no_retuning_or_successful_retry_replacement_within_block=True,
        target_NOT_guarantee=dict(P_completion_numerator=15, P_declared_denominator=18,
            compare_completion_and_position_yaw_RMS_with_time_cost=True,
            no_exclusion_or_replacement_to_meet_target=True))
    value['P_seed_portfolio'][1]['maximum_s'] = 30.
    return value


def campaign(output=OUTPUT):
    """Rebind orchestration functions to one private shared namespace.

    Function bodies are those of the pinned original campaign, not script
    text clones. No imported planner/controller globals are monkey-patched.
    """
    output = Path(output)
    scope = dict(vars(original), OUTPUT=output, PROTOCOL=output/'protocol.json',
        EXECUTION_PROTOCOL=output/'execution_protocol.json', AUTHORITY=AUTHORITY,
        __name__=__name__, __file__=__file__)
    for name, function in vars(original).items():
        if isinstance(function, FunctionType) and function.__module__ == original.__name__:
            bound = FunctionType(function.__code__, scope, function.__name__,
                function.__defaults__, function.__closure__)
            bound.__kwdefaults__ = function.__kwdefaults__
            scope[name] = bound

    def implementation_sources():
        return sorted(set(original.implementation_sources()) | {
            Path(__file__).resolve(), Path(envelope_manager.__file__).resolve()})

    def verify_sources():
        original.verify_sources()
        runtime.retained.runtime.task.verify_pins(runtime.read(output/'sources.json'))
        runtime.retained.runtime.task.verify_pins(runtime.read(scope['PROTOCOL'])['original_evidence_inputs_sha256'])
        ledger = output/'baseline_reuse.json'
        if ledger.exists():
            runtime.retained.runtime.task.verify_pins(runtime.read(ledger)['input_sha256'])

    def seed_binding():
        seed = runtime.bind(original.seed_order.base.followup.seed_portfolio,
            PROTOCOL=scope['EXECUTION_PROTOCOL'])
        def labelled(*args, **kwargs):
            value = seed(*args, **kwargs)
            return dict(value, inherited_seed_policy_text_NOT_actual_order=value.get('owned_seed_policy'),
                owned_seed_policy='COMMON_B1_10_B0_30_B2_REMAINDER45_v2')
        return labelled

    scope.update(definition=amended_definition, implementation_sources=implementation_sources,
        verify_sources=verify_sources, seed_binding=seed_binding)
    value = SimpleNamespace(**scope)
    # All mutually referenced functions share scope, including these wrappers.
    scope['dispatch'] = lambda: dispatch(value)
    scope['current_get'] = lambda **kwargs: current_get(**kwargs)
    value.dispatch = scope['dispatch']
    value.current_get = scope['current_get']
    return value


def require_unstarted_first_block(block):
    for folder in (original.OUTPUT, block.OUTPUT):
        for name in ('dispatch.json','execution_reservation.json','live'):
            if (folder/name).exists():
                raise ValueError('Prephysical amendment requires no consumed final actual block: '+str(folder/name))


def reuse_baselines(block):
    """Read-only links, with every original attempt in the reused denominator."""
    require_unstarted_first_block(block)
    old_cells = {c['cell_id']: c for c in runtime.read(original.EXECUTION_PROTOCOL)['main_cells']}
    rows = []; pins = {}
    for cell in runtime.read(block.EXECUTION_PROTOCOL)['main_cells']:
        if cell['method'] == original.design.original.planning.P:
            continue
        if cell != old_cells[cell['cell_id']]:
            raise ValueError('A reused baseline must have exactly the same input/role')
        name = cell['cell_id']; source = original.OUTPUT/'planning'/name
        old = runtime.read(original.OUTPUT/'offline_outcomes'/(name+'.json'))
        for path in source.rglob('*'):
            if path.is_file(): pins[str(path.resolve())] = runtime.sha(path)
        target = block.OUTPUT/'planning'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve(): raise ValueError('Baseline link changed')
        elif target.exists():
            raise ValueError('Baseline slot cannot replace other work')
        else:
            target.symlink_to(source.resolve(), target_is_directory=True)
        raw = dict(old, exact_original_offline_outcome=str(original.OUTPUT/'offline_outcomes'/(name+'.json')),
            exact_original_outcome_sha256=runtime.sha(original.OUTPUT/'offline_outcomes'/(name+'.json')),
            reused_unchanged_baseline_attempt=True, new_planning_or_model_trial=False,
            baseline_plan_and_model_NOT_regenerated_or_relabelled=True,
            final_scientific_protocol=str(block.PROTOCOL))
        pins[raw['exact_original_offline_outcome']] = raw['exact_original_outcome_sha256']
        runtime.evidence.write_exclusive(block.OUTPUT/'offline_outcomes'/(name+'.json'), raw)
        rows.append(raw)
    if len(rows) != 54: raise ValueError('All54 baseline attempts including every failure required')
    runtime.evidence.write_exclusive(block.OUTPUT/'baseline_reuse.json',dict(
        rows=rows, input_sha256=pins, unchanged_controller_version=original.control.VERSION,
        unchanged_baseline_inputs_algorithms_and_timing_method=True,
        baseline_failures_not_excluded=True, new_PhysX_trials=0))
    return rows


def regression(block):
    prior = runtime.read(original.OUTPUT/'regression.json')
    command = prior['command'][:-1]+[
        'src/patient_initial_stability/test/test_hospital_revision_final_manager_resume.py',
        'src/patient_initial_stability/test/test_hospital_revision_final_seed30.py',
        '--junitxml='+str(block.OUTPUT/'regression_junit.xml')]
    process = original.subprocess.run(command,text=True,capture_output=True,
        cwd=original.design.original.ROOT,timeout=180.)
    suites = ET.parse(block.OUTPUT/'regression_junit.xml').getroot().findall('testsuite')
    totals = {k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    raw = dict(command=command,exit_status=process.returncode,tests=totals['tests'],
        failed=totals['failures'],errors=totals['errors'],skipped=totals['skipped'],
        stdout=process.stdout,stderr=process.stderr,
        implementation_sources_sha256=runtime.read(block.OUTPUT/'sources.json'),
        unit_tests_NOT_physical_success_proof=True)
    runtime.evidence.write_exclusive(block.OUTPUT/'regression.json',raw)
    if process.returncode or any(totals[k] for k in ('failures','errors','skipped')):
        raise ValueError('Amended frozen-source regression must fully pass')
    return raw


def offline(block):
    require_unstarted_first_block(block)
    block.export_protocol(); block.verify_sources(); block.build_assets(); regression(block)
    before = original.current_get()
    runtime.evidence.write_exclusive(block.OUTPUT/'before_offline_live_GET.json',before)
    runtime.evidence.write_exclusive(block.OUTPUT/'offline_reservation.json',dict(
        P_and_history_plans=24, reused_baseline_attempts=54,offline_envelope_plans=12,
        single_worker_actual_timeline_PAUSED=True,one_attempt_per_v2_input=True,
        new_PhysX_trials=0, user_authority=AUTHORITY))
    reuse_baselines(block)
    # Check the previously starved seed first without adding a second attempt.
    cells = runtime.read(block.EXECUTION_PROTOCOL)['main_cells']
    pending = [c for c in cells if c['method']==original.design.original.planning.P]
    pending.sort(key=lambda c:c['cell_id']!='H3_P_dynamics_full_envelope_f03')
    for index,cell in enumerate(pending,1):
        name = cell['cell_id']; value = block.plan_cell(name); model = block.model_cell(name)
        row = dict(value,model_report=str(block.OUTPUT/'model'/name/'model_screen.json') if model else None,
            both_nominal_resolutions_pass=model['both_nominal_resolutions_pass'] if model else None)
        runtime.evidence.write_exclusive(block.OUTPUT/'offline_outcomes'/(name+'.json'),row)
        print('[SEED30 OFFLINE]',index,'/24',name,value['plan_success'],row['both_nominal_resolutions_pass'],flush=True)
    extra=[]
    for cell in cells:
        if cell['experiment_role']!='MAIN' or cell['method']!=original.design.original.planning.P or cell['task'] not in ('H2','H3'):
            continue
        for variant in ('point','endpoints'):
            try:
                value=block.plan_cell(cell['cell_id'],envelope_variant=variant)
            except ValueError as error:
                if not envelope_manager.expected_offline_rejection(error): raise
                value=runtime.bind(envelope_manager.negative_receipt,base=block,__file__=__file__)(
                    cell['cell_id'],variant,error)
            extra.append(value)
            print('[SEED30 ENVELOPE]',len(extra),'/12',cell['cell_id'],variant,value['status'],flush=True)
    rows=[runtime.read(block.OUTPUT/'offline_outcomes'/(c['cell_id']+'.json')) for c in cells]
    after=original.current_get()
    if before!=after: raise ValueError('Admitted actual state changed during offline work')
    raw=dict(status='FINAL_N3_OFFLINE_FINISHED',rows=rows,offline_envelope=extra,
        actual_motion_trials=0,baseline_attempts_reused=54,P_and_history_attempts_regenerated=24,
        original_v1_negative_results_preserved=True,inputs_inspected_NOT_untouched_holdout=True)
    runtime.evidence.write_exclusive(block.OUTPUT/'offline_completed.json',raw)
    return raw


def dispatch(block):
    require_unstarted_first_block(block);block.verify_sources()
    if not (block.OUTPUT/'offline_completed.json').exists(): raise ValueError('All offline outcomes required')
    check=runtime.read(block.OUTPUT/'regression.json')
    if check['exit_status'] or check['failed'] or check['errors'] or check['skipped']:
        raise ValueError('Fully passing current-source regression required')
    runtime.retained.runtime.task.verify_pins(check['implementation_sources_sha256'])
    source=("import sys,types,asyncio,builtins,importlib\nfrom pathlib import Path\n"
        "importlib.invalidate_caches()\n"
        "_n3_parent=getattr(builtins,'ijcas_revision_targeted_joint_v7',None)\n"
        "assert _n3_parent is not None and _n3_parent.done() and not _n3_parent.exception(),'Exact completed predecessor required'\n"
        "assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is _n3_parent,'Intervening motion'\n"
        "assert not hasattr(builtins,'ijcas_revision_final_n3'),'Original block must remain unstarted'\n"
        f"assert not hasattr(builtins,{KEY!r}),'Amended block already reserved'\n"
        f"_n3_path=Path({str(Path(__file__).resolve())!r})\n"
        "_n3_module=types.ModuleType('hospital_revision_final_seed30');_n3_module.__file__=str(_n3_path)\n"
        "sys.modules[_n3_module.__name__]=_n3_module\n"
        "exec(compile(_n3_path.read_bytes(),str(_n3_path),'exec'),_n3_module.__dict__)\n"
        "_n3_job=asyncio.ensure_future(_n3_module.campaign().run(_n3_parent,authorized=True))\n"
        f"builtins.{KEY}=_n3_job\n"
        "print({'amended_final_n3_main72_ablation6_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(block.OUTPUT/'dispatch.json',dict(source=source,
        protocol_sha256=runtime.sha(block.PROTOCOL),authority=AUTHORITY,automatic_retry=False))
    return original.predecessor.control.base.shared.runner.remote(source)


def current_get(*,final=False):
    if not final:return original.current_get()
    fn=original.current_get
    # Same read-only ownership/timeline check; substitute only the owned key.
    class Runner:
        @staticmethod
        def remote(source):
            return original.predecessor.control.base.shared.runner.remote(source.replace(
                repr('ijcas_revision_final_n3'),repr(KEY)))
    prior=original.predecessor
    shared=SimpleNamespace(**dict(vars(prior.control.base.shared),runner=Runner))
    control=SimpleNamespace(**dict(vars(prior.control),base=SimpleNamespace(
        **dict(vars(prior.control.base),shared=shared))))
    return runtime.bind(fn,predecessor=SimpleNamespace(**dict(vars(prior),control=control)))(final=True)


def all_steps():
    block=campaign();offline(block);print(dispatch(block),flush=True)
    previous=-1
    while not (block.OUTPUT/'completed.json').exists():
        if (block.OUTPUT/'stopped.json').exists():
            raise ValueError('Actual boundary unresolved; no automatic physical retry')
        snapshot=block.results_snapshot()
        if snapshot['accounted_cells']!=previous:
            previous=snapshot['accounted_cells'];print('[SEED30 ACTUAL PROGRESS]',previous,'/78',flush=True)
            for group in snapshot['groups']:
                if group['accounted_n']==group['declared_n']:print('[SEED30 GROUP]',group,flush=True)
        time.sleep(30.)
    completed=runtime.read(block.OUTPUT/'completed.json');time.sleep(.5)
    final=current_get(final=True)
    if abs(final['current_time_s']-runtime.read(completed['last_actual_report'])['actual_final_timeline']['current_time_s'])>1e-6:
        raise ValueError('Final actual Pause does not match saved last report')
    runtime.evidence.write_exclusive(block.OUTPUT/'final_live_GET.json',final)
    folder=block.export_results()
    runtime.evidence.write_exclusive(block.OUTPUT/'all_steps_completed.json',dict(
        status='FINAL_N3_EXECUTION_AND_REPORTING_FINISHED',analysis_folder=str(folder),
        all_failures_preserved=True,inspected_inputs_not_untouched_holdout=True))
    print('[SEED30 FINISHED]',folder,flush=True)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('definition','offline','all','results','status'))
    args=parser.parse_args()
    block=campaign()
    if args.action=='definition':print(block.export_protocol())
    elif args.action=='offline':offline(block)
    elif args.action=='all':all_steps()
    elif args.action=='results':print(block.export_results())
    else:print(block.results_snapshot())
