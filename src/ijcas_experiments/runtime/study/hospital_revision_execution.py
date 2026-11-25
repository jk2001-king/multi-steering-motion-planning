"""Finite primary/ablation execution queue; one actual trial per frozen cell.

Predicted model HOLDs are not execution failures. Collision-invalid reference
paths never enter physics. Every new actor still passes its own actual static
test, and every route uses the original physical guards and verdict code.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
import argparse
import asyncio
import copy
from pathlib import Path

import hospital_revision_campaign as campaign
import hospital_revision_live as live
import hospital_revision_metrics as metrics
import hospital_revision_worker as offline

runtime=campaign.runtime
POLICY=campaign.OUTPUT/'diagnostic_model_admission_v2.json'
QUEUE=campaign.OUTPUT/'execution_queue_v2'


def export_policy():
    value=dict(version='COMMON_DIAGNOSTIC_MODEL_v2',
        user_authority='뭐가 가장 베스트의 방향성인지 판단하고 진행해',
        purpose='Measure actual matched baseline and P outcomes without filtering on model predictions',
        protocol=str(campaign.PROTOCOL),protocol_sha256=runtime.sha(campaign.PROTOCOL),
        model_is_prediction_not_actual_failure=True,
        independent_full_envelope_audit_required=True,actual_static_PASS_required=True,
        unchanged_actual_controller_limits_and_success_criteria=True,
        applies_to_all_methods_and_history_ablation=True,no_physical_retry=True,
        previously_measured_negative_outcomes_preserved=True,
        changed_admission_is_NOT_a_change_in_physical_success_threshold=True,
        no_reference_collision_waiver=True)
    if POLICY.exists():
        if runtime.read(POLICY)!=value:raise ValueError('Prospective policy already frozen')
    else:runtime.evidence.write_exclusive(POLICY,value)
    return POLICY


def candidates(protocol):
    for cell in offline.queue(protocol):
        yield cell,False
        if cell['task'] in ('H2','H3') and cell['method']==campaign.planning.P:
            yield cell,True


def eligibility(cell,no_history):
    label=cell['cell_id']+('_no_entry_history' if no_history else '')
    folder=live.planning.OUTPUT/label
    summary=runtime.read(folder/'summary.json')
    if not summary['plan_success']:return 'PLANNING_FAILURE',folder
    audit=runtime.read(folder/'independent_plan_audit.json')
    if audit['verdict']!='PASS':return 'INDEPENDENT_REFERENCE_AUDIT_REJECTED',folder
    # A diagnostic HOLD is allowed; the unaltered report and pins remain.
    model=runtime.read(folder/'executor_model_screen/model_screen.json')
    runtime.retained.runtime.task.verify_pins(model['input_sha256'])
    runtime.retained.runtime.task.verify_pins(model['output_sha256'])
    return 'ELIGIBLE_FOR_OWN_FRESH_STATIC',folder


def restored_boundary(report):
    return (report.get('actual_final_timeline',{}).get('playing') is False
            and report.get('actual_final_timeline',{}).get('stopped') is False
            and report.get('checks',{}).get('original_gains_targets_restored') is True
            and bool(report.get('actual_final_readback',{}).get('tensor')))


async def run(*,prior_result,authorized=False,tasks=None):
    if authorized is not True:raise ValueError('Explicit reviewer-campaign execution authority required')
    live.checked_model_policy(POLICY)
    protocol=runtime.read(campaign.PROTOCOL)
    if not (campaign.OUTPUT/'offline_worker_complete.json').exists():
        raise ValueError('Frozen finite offline queue must be complete before actual dispatch')
    QUEUE.mkdir(parents=True,exist_ok=False)
    receipt=dict(protocol=str(campaign.PROTOCOL),protocol_sha256=runtime.sha(campaign.PROTOCOL),
        policy=str(POLICY),policy_sha256=runtime.sha(POLICY),prior_result=str(prior_result),
        tasks=tasks,one_attempt_per_cell=True,automatic_retry=False,
        shared_physical_limits_and_success_criteria_unchanged=True)
    files=(Path(__file__),Path(live.__file__),Path(metrics.__file__))
    receipt['source_sha256']={str(p.resolve()):runtime.sha(p) for p in files}
    runtime.evidence.write_exclusive(QUEUE/'reservation.json',receipt)
    runtime.lc.freeze_workflow_sources(QUEUE/'execution_sources.zip',receipt['source_sha256'])
    last=Path(prior_result).resolve()
    if not restored_boundary(runtime.read(last)):
        raise ValueError('Exact restored actual prior boundary required')
    outcomes=[]
    try:
        for cell,no_history in candidates(protocol):
            if tasks is not None and cell['task'] not in tasks:continue
            label=cell['cell_id']+('_no_entry_history' if no_history else '')
            old=campaign.OUTPUT/'live'/label
            if old.exists():
                # Never resume/replace a consumed attempt. Old actual outcomes
                # remain part of the denominator and detailed reporting.
                if not (old/'trial_outcome.json').exists():
                    raise ValueError('Consumed cell has unresolved infrastructure boundary: '+label)
                outcome=dict(runtime.read(old/'trial_outcome.json'),previously_consumed=True)
            else:
                reason,folder=eligibility(cell,no_history)
                if reason!='ELIGIBLE_FOR_OWN_FRESH_STATIC':
                    outcome=dict(cell_id=label,status=reason,actual_route_started=False,
                        attempt_replaced=False,planning_summary=str(folder/'summary.json'))
                else:
                    print('[REVIEW ACTUAL START]',label,flush=True)
                    outcome=await live.run_cell(cell['cell_id'],prior_result=last,authorized=True,
                        model_policy_path=POLICY,no_entry_history=no_history)
                    report_path=Path(outcome['report'])
                    report=runtime.read(report_path)
                    if not restored_boundary(report):
                        raise ValueError('Actual Pause/restoration not verified; queue stops: '+label)
                    last=report_path
                    if outcome.get('actual_route_started'):
                        try:
                            target=report_path.parent/'controlled_actual_metrics.json'
                            if not target.exists():runtime.evidence.write_exclusive(target,metrics.metrics(report_path))
                        except Exception as error:
                            runtime.evidence.write_exclusive(report_path.parent/'metrics_unavailable.json',dict(
                                error=type(error).__name__+': '+str(error),actual_result_unchanged=True))
            outcomes.append(outcome)
            runtime.evidence.write_exclusive(QUEUE/'outcomes'/(label+'.json'),outcome)
            print('[REVIEW ACTUAL RESULT]',outcome,flush=True)
            # Allow the existing app/event/render loop to run between cells.
            await asyncio.sleep(0)
        result=dict(status='FINITE_QUEUE_FINISHED',outcomes=outcomes,last_actual_report=str(last),
                    all_negative_outcomes_retained=True,no_success_replacement=True)
        runtime.evidence.write_exclusive(QUEUE/'completed.json',result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(QUEUE/'stopped.json',dict(
            error=type(error).__name__+': '+str(error),last_actual_report=str(last),
            completed_cells=len(outcomes),no_automatic_recovery_or_retry=True))
        raise


def dispatch(prior_result,tasks=None):
    import hospital_revision_runner as runner
    import hospital_revision_envelope as envelope
    export_policy()
    files=[Path(live.planning.__file__),Path(runner.__file__),Path(envelope.__file__),
           Path(offline.__file__),Path(live.__file__),Path(metrics.__file__),Path(__file__)]
    source="import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
    source+="old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\nassert old is None or old.done(),'Another experiment is active'\n"
    for path in files:
        name=path.stem
        source+=(f"p=Path({str(path.resolve())!r})\nm=types.ModuleType({name!r});m.__file__=str(p);sys.modules[{name!r}]=m\n"
                 "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n")
    source+=(f"job=asyncio.ensure_future(m.run(prior_result=Path({str(Path(prior_result).resolve())!r}),authorized=True,tasks={tasks!r}))\n"
             "builtins.ijcas_hospital_latest_task_experiment=job\nbuiltins.ijcas_revision_primary_task=job\n"
             "print({'finite_queue_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(campaign.OUTPUT/'execution_queue_v2_dispatch.json',dict(
        exact_command=source,policy=str(POLICY),entry_sha256=runtime.sha(Path(__file__))))
    return runner.remote(source)


async def resume_typed_writeback_block(*,prior_result,authorized=False):
    """Continue AFTER the consumed route; never repeat its failed pre-hold."""
    if authorized is not True:raise ValueError('Current reviewer-campaign authority required')
    import hospital_revision_serialization as serial
    import hospital_revision_initial_conditions as initial
    from types import SimpleNamespace
    prior_result=Path(prior_result).resolve()
    report=runtime.read(prior_result)
    if not restored_boundary(report):raise ValueError('Actual Pause/restoration required before continuing')
    output=serial.ROOT
    runtime.evidence.write_exclusive(output/'queue_resume_reservation.json',dict(
        prior_result=str(prior_result),prior_result_sha256=runtime.sha(prior_result),
        failed_current_route_preserved_NOT_retried=True,
        change='Analysis only: absent tracking interval yields null RMS, not zero or a new trial',
        source_sha256={str(Path(p).resolve()):runtime.sha(p) for p in (__file__,metrics.__file__,serial.__file__)}))
    runtime.lc.freeze_workflow_sources(output/'queue_resume_sources.zip',
        {str(Path(p).resolve()):runtime.sha(p) for p in (__file__,metrics.__file__,serial.__file__)})
    try:
        target=prior_result.parent/'controlled_actual_metrics.json'
        if not target.exists():runtime.evidence.write_exclusive(target,metrics.metrics(prior_result))
        proxy=SimpleNamespace(**dict(vars(live),run_cell=runtime.bind(live.run_cell,
            runtime=serial.fixed_runtime(),__file__=serial.__file__)))
        main=await runtime.bind(run,live=proxy,runtime=serial.queue_runtime(),QUEUE=serial.QUEUE)(
            prior_result=prior_result,authorized=True)
        if main['status']!='FINITE_QUEUE_FINISHED':raise ValueError('Main queue did not finish normally')
        joint=await runtime.bind(initial.run,cell_bindings=serial.initial_cell_binding,__file__=serial.__file__)(
            prior_result=Path(main['last_actual_report']),authorized=True)
        runtime.evidence.write_exclusive(initial.OUTPUT/'after_primary_result.json',joint)
        result=dict(status='ALL_DECLARED_REVISION_PHYSICAL_BLOCKS_FINISHED',
            main=main,initial_conditions=joint,all_negative_outcomes_preserved=True)
        runtime.evidence.write_exclusive(output/'all_blocks_completed.json',result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(output/'queue_resume_stopped.json',dict(
            error=type(error).__name__+': '+str(error),no_automatic_retry=True))
        raise


def dispatch_typed_writeback_resume(prior_result):
    import hospital_revision_runner as runner
    import hospital_revision_serialization as serial
    files=(Path(metrics.__file__).resolve(),Path(__file__).resolve(),Path(serial.__file__).resolve())
    source="import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
    source+="old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\nassert old is None or old.done(),'Another experiment is active'\n"
    for p in files:
        source+=(f"p=Path({str(p)!r})\nm=types.ModuleType({p.stem!r});m.__file__=str(p);sys.modules[m.__name__]=m\n"
            "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n")
    source+=(f"job=asyncio.ensure_future(sys.modules['hospital_revision_execution'].resume_typed_writeback_block(prior_result=Path({str(Path(prior_result).resolve())!r}),authorized=True))\n"
        "builtins.ijcas_hospital_latest_task_experiment=job\nbuiltins.ijcas_revision_primary_task=job\n"
        "print({'remaining_finite_queue_resumed':True,'consumed_route_retried':False,'absent_RMS_not_fabricated_as_zero':True})\n")
    runtime.evidence.write_exclusive(_ijcas_path(serial.ROOT, 'queue_resume_dispatch.json'),dict(
        exact_command=source,entry_source_sha256=runtime.sha(Path(__file__))))
    return runner.remote(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-result',type=Path,required=True)
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--resume-typed-writeback',action='store_true')
    args=parser.parse_args()
    print(dispatch_typed_writeback_resume(args.prior_result) if args.resume_typed_writeback else dispatch(args.prior_result,args.tasks))
