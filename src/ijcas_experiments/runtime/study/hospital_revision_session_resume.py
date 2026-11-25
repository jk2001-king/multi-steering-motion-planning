"""Resume the user-closed Isaac process, not a failed method route.

All planned inputs and shared control remain fixed. The sole consumed cell
was interrupted inside its initial zero-command static screen, with no
static result and no method route packet/trace. Keep that attempt intact.
Start a fresh app/source, reconstruct the exact static environment, and
record one explicitly authorized infrastructure restart of that preparation.
Never fabricate restoration of the lost dynamic state or an extra repeat.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import hospital_revision_final_seed30 as experiment
import hospital_revision_restart_app as launcher
import hospital_session_bootstrap as bootstrap

runtime=experiment.runtime
OUTPUT=launcher.OUTPUT
FIRST='H1_B0_fixed_counterphase_ramp_f01'
KEY='ijcas_revision_final_n3_session_resume_r01'
AUTHORITY='2026-11-22 사용자: 아이작 꺼짐 다시 켜서 이어서 진행해'


def verify_interruption():
    block=experiment.campaign();block.verify_sources()
    existing=list((block.OUTPUT/'outcomes').glob('*.json'))
    live=list((block.OUTPUT/'live').iterdir())
    expected=block.OUTPUT/'live'/FIRST
    if existing or live!=[expected]:raise ValueError('Only the diagnosed first pre-route process interruption')
    if any((expected/p).exists() for p in ('route','static/held_static_readback.json','trial_outcome.json')):
        raise ValueError('A completed static/method route cannot be silently restarted')
    receipt=runtime.read(expected/'primary_trial_definition.json')
    if receipt['cell']['cell_id']!=FIRST or not (expected/'static/single_held_static_reserved.json').exists():
        raise ValueError('Exact incomplete zero-command static attempt required')
    if (block.OUTPUT/'completed.json').exists() or (block.OUTPUT/'stopped.json').exists():
        raise ValueError('No completed/diagnosed negative route replacement')
    return dict(cell_id=FIRST,original_consumed_preparation=str(expected),
        original_preparation_sha256={str(p):runtime.sha(p) for p in expected.rglob('*') if p.is_file()},
        original_static_verdict='UNAVAILABLE_PROCESS_TERMINATED',original_static_fixed_recording_completion='UNCONFIRMED',
        method_route_packets_or_traces=0,new_independent_repetition_count=0,
        user_reported_process_closed=True,old_dynamic_state_NOT_restored=True,
        lost_process_Pause_and_gain_restoration_NOT_confirmed=True,
        one_explicit_infrastructure_static_restart_authorized=True,
        original_files_not_moved_deleted_or_relabelled=True,
        original_main_and_history_protocol_unchanged=True,user_authority=AUTHORITY)


def freeze():
    value=verify_interruption()
    runtime.evidence.write_exclusive(OUTPUT/'interruption.json',value)
    pins={str(Path(p).resolve()):runtime.sha(p) for p in (__file__,bootstrap.__file__,launcher.__file__)}
    runtime.evidence.write_exclusive(OUTPUT/'sources.json',pins)
    runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',pins)
    return value


def verify():
    experiment.campaign().verify_sources()
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'sources.json'))
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'interruption.json')['original_preparation_sha256'])


async def prepare_environment():
    import builtins
    import omni.usd
    import omni.timeline
    import hospital_h4_cohort_preparation as constructor
    context=omni.usd.get_context();timeline=omni.timeline.get_timeline_interface()
    if timeline.is_playing() or not timeline.is_stopped() or timeline.get_current_time()!=0.:
        raise ValueError('Only a fresh STOPPED process may reconstruct the environment')
    if getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not None:
        raise ValueError('Never replace another experiment')
    stage=context.get_stage()
    if stage and not stage.GetRootLayer().anonymous:
        if stage.GetRootLayer().identifier!=str(constructor.SOURCE):
            raise ValueError('Unexpected open saved stage; do not overwrite')
    if not stage or stage.GetRootLayer().identifier!=str(constructor.SOURCE):
        success,error=await context.open_stage_async(str(constructor.SOURCE))
        if not success:raise RuntimeError('Cannot open preserved original source: '+str(error))
    # Allow ordinary first-view metadata to settle BEFORE taking strict
    # Root/Session baselines; this sends no camera or timeline command.
    import omni.kit.app
    for unused in range(30):
        await omni.kit.app.get_app().next_update_async()
        if timeline.is_playing() or not timeline.is_stopped() or timeline.get_current_time()!=0.:
            raise ValueError('Fresh app timeline changed before environment verification')
    call=runtime.bind(bootstrap.prepare_environment_once,OUTPUT=OUTPUT/'environment')
    await call(authorized=True)
    return runtime.read(OUTPUT/'environment/environment_prepared.json')


def checked_cell_binding(block,cell,*,first):
    call=block.bound_cell(cell)
    campaign=call.__globals__['campaign']
    if first:
        campaign=SimpleNamespace(**dict(vars(campaign),OUTPUT=OUTPUT/'restarted_preparation'))
    return runtime.bind(call,campaign=campaign,__file__=__file__)


async def run(*,authorized=False):
    if authorized is not True:raise ValueError('Current explicit process-restart authority required')
    import builtins
    verify();block=experiment.campaign()
    ready=runtime.read(OUTPUT/'environment/environment_prepared.json')
    if ready['status']!='FRESH_COMMON_ENVIRONMENT_VERIFIED_NOT_STATIC_OR_DRIVE_ADMISSION':
        raise ValueError('Verified fresh common environment required')
    task=asyncio.current_task()
    if getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not None:
        raise ValueError('No concurrent owned experiment permitted')
    runtime.evidence.write_exclusive(OUTPUT/'execution_reservation.json',dict(
        user_authority=AUTHORITY,original_protocol=str(block.PROTOCOL),
        same_78_declared_cells_not_extra_independent_repetitions=True,
        old_interrupted_static_retained=True,one_infrastructure_restart_ONLY=True))
    builtins.ijcas_hospital_latest_task_experiment=task
    builtins.ijcas_revision_primary_task=task
    last=None;results=[]
    try:
        for index,cell in enumerate(runtime.read(block.EXECUTION_PROTOCOL)['main_cells']):
            verify();name=cell['cell_id'];folder=block.folder_for(name)
            if getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not task:
                raise ValueError('Another owner changed the live experiment')
            planned=runtime.read(folder/'final_planning_receipt.json')
            if not planned['independently_valid_reference']:
                result=dict(cell_id=name,status='PLANNING_FAILURE_NO_PHYSICAL_RUN' if not planned['plan_success']
                    else 'INDEPENDENT_REFERENCE_AUDIT_REJECTED',actual_route_started=False,
                    actual_verdict='PLAN_FAILURE' if not planned['plan_success'] else 'REFERENCE_REJECTED',
                    planning_summary=str(folder/'summary.json'),metrics=None)
            else:
                if index==0 and name!=FIRST:raise ValueError('Original fixed execution order changed')
                print('[N3 RESUME ACTUAL START]',name,flush=True)
                result=await checked_cell_binding(block,cell,first=index==0)(name,prior_result=last,
                    authorized=True,model_policy_path=block.policy_file())
                last=Path(result['report']);raw=runtime.read(last)
                if not block.safe_restored_boundary(raw):
                    raise ValueError('Unresolved actual/restoration boundary; no automatic retry or next trial')
                result=dict(result,metrics=block.metrics.metrics(last) if result['actual_route_started'] else None)
            result.update(experiment_role=cell['experiment_role'],controller_version=block.control.VERSION,
                predictive_history_policy=cell['predictive_history_policy'],final_fixed_version_trial=True,
                attempt_replaced=False,infrastructure_restart_before_first_method_route=index==0,
                interruption_record=str(OUTPUT/'interruption.json') if index==0 else None,
                independent_repetitions_added_by_process_restart=0)
            runtime.evidence.write_exclusive(block.OUTPUT/'outcomes'/(name+'.json'),result);results.append(result)
            print('[N3 RESUME RESULT]',name,result['actual_verdict'],flush=True)
            await asyncio.sleep(0)
        value=dict(status='FINAL_N3_ALL_DECLARED_CELLS_FINISHED',results=results,
            last_actual_report=str(last),all_negative_results_preserved=True,automatic_retry=False,
            process_interruption_record=str(OUTPUT/'interruption.json'))
        runtime.evidence.write_exclusive(block.OUTPUT/'completed.json',value)
        return value
    except BaseException as error:
        runtime.evidence.write_exclusive(block.OUTPUT/'stopped.json',dict(
            error=type(error).__name__+': '+str(error),results=results,last_actual_report=str(last),
            no_automatic_retry=True,process_interruption_record=str(OUTPUT/'interruption.json')))
        raise


def dispatch():
    verify()
    if not (OUTPUT/'environment/environment_prepared.json').exists():raise ValueError('Fresh environment not prepared')
    source=("import importlib,sys,asyncio,builtins\nimportlib.invalidate_caches()\n"
        f"sys.path.insert(0,{str(Path(__file__).parent.resolve())!r})\n"
        "import hospital_revision_session_resume as _resume\n"
        f"assert not hasattr(builtins,{KEY!r}),'Resume already reserved'\n"
        "_resume_task=asyncio.ensure_future(_resume.run(authorized=True))\n"
        f"builtins.{KEY}=_resume_task\n"
        "print({'fixed_n3_session_resume_scheduled':True,'blind_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch.json',dict(source=source,user_authority=AUTHORITY))
    return experiment.original.predecessor.control.base.shared.runner.remote(source)


def prepare_dispatch():
    verify()
    runner=experiment.original.predecessor.control.base.shared.runner
    source=("import sys,importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        f"sys.path.insert(0,{str(Path(__file__).parent.resolve())!r})\n"
        "import hospital_revision_session_resume as _resume\n"
        "assert not hasattr(builtins,'ijcas_revision_restart_environment'),'Environment preparation already reserved'\n"
        "builtins.ijcas_revision_restart_environment=asyncio.ensure_future(_resume.prepare_environment())\n"
        "print({'fresh_environment_preparation_scheduled':True,'Play_or_drive':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'environment_dispatch.json',dict(source=source,user_authority=AUTHORITY))
    return runner.remote(source)


def all_steps():
    freeze();runner=experiment.original.predecessor.control.base.shared.runner
    begun=time.monotonic()
    while True:
        try:
            with socket.create_connection(('127.0.0.1',8226),timeout=1.):break
        except OSError:
            if time.monotonic()-begun>180.:raise TimeoutError('Restarted Isaac execution extension not ready')
            time.sleep(5.)
    print(prepare_dispatch(),flush=True)
    while not (OUTPUT/'environment/environment_prepared.json').exists():
        if (OUTPUT/'environment/error.json').exists():raise ValueError('Fresh environment reconstruction failed')
        result=runner.remote("import builtins\n_r=builtins.ijcas_revision_restart_environment\nprint({'done':_r.done(),'error':str(_r.exception()) if _r.done() and _r.exception() else None})\n")
        if "'error': None" not in result['output']:raise ValueError('Fresh environment job failed: '+result['output'])
        print('[N3 RESTART ENVIRONMENT]',result['output'].strip(),flush=True);time.sleep(15.)
    print(dispatch(),flush=True);block=experiment.campaign();n=-1
    while not (block.OUTPUT/'completed.json').exists():
        if (block.OUTPUT/'stopped.json').exists():raise ValueError('Resumed block stopped; inspect without automatic retry')
        snapshot=block.results_snapshot()
        if snapshot['accounted_cells']!=n:
            n=snapshot['accounted_cells'];print('[N3 RESUMED PROGRESS]',n,'/78',flush=True)
        time.sleep(30.)
    completed=runtime.read(block.OUTPUT/'completed.json')
    source=("import builtins,json,omni.timeline\n"
        f"_job=getattr(builtins,{KEY!r},None)\n"
        "_t=omni.timeline.get_timeline_interface()\n"
        "print(json.dumps({'job_done':bool(_job and _job.done()),"
        "'is_latest':getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is _job,"
        "'error':str(_job.exception()) if _job and _job.done() and _job.exception() else None,"
        "'playing':_t.is_playing(),'stopped':_t.is_stopped(),'current_time_s':_t.get_current_time()}))\n")
    import json
    final=json.loads(runner.remote(source)['output'])
    if (not final['job_done'] or not final['is_latest'] or final['error'] is not None
            or final['playing'] or final['stopped'] or abs(final['current_time_s']-
                runtime.read(completed['last_actual_report'])['actual_final_timeline']['current_time_s'])>1e-6):
        raise ValueError('Final actual Pause/ownership does not match the saved last report')
    runtime.evidence.write_exclusive(block.OUTPUT/'final_live_GET.json',final)
    folder=block.export_results()
    import hospital_revision_final_comparison_analysis as analysis
    detail=analysis.export()
    runtime.evidence.write_exclusive(block.OUTPUT/'all_steps_completed.json',dict(
        status='FINAL_N3_EXECUTION_AND_REPORTING_FINISHED',analysis_folder=str(folder),
        detailed_feedback_analysis=str(detail),all_failures_preserved=True,
        process_interruption_record=str(OUTPUT/'interruption.json')))
    print('[N3 RESUMED FINISHED]',folder,detail,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('all','prepare','dispatch'))
    args=parser.parse_args()
    if args.action=='all':all_steps()
    elif args.action=='prepare':print(prepare_dispatch())
    else:print(dispatch())
