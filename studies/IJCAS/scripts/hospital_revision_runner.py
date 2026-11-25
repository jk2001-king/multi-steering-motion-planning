"""Finite reviewer-campaign runner; no implicit retries or source USD saves."""
import argparse
import json
from pathlib import Path
import socket
import time

import hospital_revision_campaign as campaign
import hospital_revision_planning as planning

runtime=campaign.runtime


def model_screen(cell_id):
    folder=planning.OUTPUT/cell_id
    summary=runtime.read(folder/'summary.json')
    if not summary['plan_success']:
        return dict(status='NO_PLAN_NO_MODEL_TRIAL')
    output=folder/'executor_model_screen'
    target=output/'model_screen.json'
    if target.exists():
        value=runtime.read(target)
        runtime.retained.runtime.task.verify_pins(value['input_sha256'])
        runtime.retained.runtime.task.verify_pins(value['output_sha256'])
        return value
    if output.exists():
        raise FileExistsError('Partial model attempt preserved; no retry')
    plan_path=Path(summary['plan'])
    plan=runtime.read(plan_path)
    task=plan['scenario'].split('_')[0]
    protocol=runtime.read(campaign.PROTOCOL)
    evaluation=folder/'common_controller_model_binding.json'
    runtime.evidence.write_exclusive(evaluation,dict(profile=plan['geometry_profile'],
        primary_protocol=str(campaign.PROTOCOL),primary_protocol_sha256=runtime.sha(campaign.PROTOCOL),
        plan=str(plan_path),plan_sha256=runtime.sha(plan_path),physical_trials=0))
    campaign.planning.shared.write_unified_mission_model_screen(
        evaluation,plan_path,plan['geometry_manifest'],folder/'planning_occupancy.npz',
        output=output,simulation_budget_s=protocol['tasks'][task]['simulation_budget_s'])
    return runtime.read(target)


def remote(source, *, timeout=25.):
    with socket.create_connection(('127.0.0.1',8226),timeout=10) as channel:
        channel.sendall(source.encode())
        channel.settimeout(timeout)
        blocks=[]
        while True:
            block=channel.recv(65536)
            if not block:break
            blocks.append(block)
    response=json.loads(b''.join(blocks).decode())
    if response.get('status')!='ok':
        raise RuntimeError('Isaac command failed: '+str(response))
    return response


def dispatch_cell(cell_id,prior_result):
    protocol=runtime.read(campaign.PROTOCOL)
    if cell_id not in {r['cell_id'] for r in protocol['main_cells']}:
        raise ValueError('Only a predeclared primary cell can be dispatched')
    output=campaign.OUTPUT/'live'/cell_id
    if output.exists():
        raise FileExistsError('Consumed physical cell; no retry')
    module_path=Path(__file__).with_name('hospital_revision_live.py').resolve()
    source=("import sys, types, asyncio, builtins\nfrom pathlib import Path\n"
        "old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\n"
        "assert old is None or old.done(), 'Another experiment is active'\n"
        f"p=Path({str(module_path)!r})\n"
        "live=types.ModuleType('hospital_revision_live');live.__file__=str(p);sys.modules['hospital_revision_live']=live\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),live.__dict__)\n"
        f"job=asyncio.ensure_future(live.run_cell({cell_id!r},prior_result=Path({str(Path(prior_result).resolve())!r}),authorized=True))\n"
        "builtins.ijcas_hospital_latest_task_experiment=job\n"
        "builtins.ijcas_revision_primary_task=job\n"
        "print({'scheduled':True,'automatic_retry':False})\n")
    audit=campaign.OUTPUT/'dispatch'/cell_id/'command.json'
    runtime.evidence.write_exclusive(audit,dict(cell_id=cell_id,primary_protocol=str(campaign.PROTOCOL),
        primary_protocol_sha256=runtime.sha(campaign.PROTOCOL),prior_result=str(prior_result),
        exact_command_source=source,entry_source=str(module_path),entry_source_sha256=runtime.sha(module_path),
        current_authority=runtime.APPROVAL,no_automatic_retry=True))
    result=remote(source)
    print('[REVIEW DISPATCH]',cell_id,result['output'].strip(),flush=True)
    return result


def task_status():
    source=("import builtins, json\njob=getattr(builtins,'ijcas_revision_primary_task',None)\n"
        "print(json.dumps({'exists':job is not None,'done':job.done() if job else False,"
        "'exception':str(job.exception()) if job and job.done() and not job.cancelled() and job.exception() else None,"
        "'result':job.result() if job and job.done() and not job.cancelled() and job.exception() is None else None}))\n")
    return json.loads(remote(source)['output'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['model','dispatch','status'])
    parser.add_argument('cell_id',nargs='?')
    parser.add_argument('--prior-result',type=Path)
    args=parser.parse_args()
    if args.action=='model':
        result=model_screen(args.cell_id)
        print('MODEL BOTH PASS:',result.get('both_nominal_resolutions_pass'))
    elif args.action=='dispatch':
        if args.prior_result is None:parser.error('--prior-result required')
        dispatch_cell(args.cell_id,args.prior_result)
    else:print(json.dumps(task_status(),indent=2))
