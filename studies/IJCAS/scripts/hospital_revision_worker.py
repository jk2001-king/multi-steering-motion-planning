"""Finite offline campaign queue. Never imports SDK or dispatches motion."""
import argparse
from pathlib import Path
import traceback

import hospital_revision_campaign as campaign
import hospital_revision_planning as planning
import hospital_revision_runner as runner
import hospital_revision_envelope as envelope

runtime=campaign.runtime


def queue(protocol,task_order=('H2','H1','H3','H4','H5','H6')):
    rows=protocol['main_cells']
    return [r for task in task_order for r in rows if r['task']==task]


def run_one(cell):
    cell_id=cell['cell_id']
    target=campaign.OUTPUT/'offline_outcomes'/cell_id/'outcome.json'
    if target.exists():
        return runtime.read(target)
    folder=planning.OUTPUT/cell_id
    if (folder/'summary.json').exists():
        result=runtime.read(folder/'summary.json')
    else:
        result=planning.run_cell(cell_id)
    status='PLANNING_FAILURE'
    model=None
    if result['plan_success']:
        if result['independent_model_consistency']!='PASS':
            status='INDEPENDENT_AUDIT_REJECTED'
        else:
            model=runner.model_screen(cell_id)
            status='READY_FOR_OWN_FRESH_STATIC' if model.get('both_nominal_resolutions_pass') else 'MODEL_ADMISSION_REJECTED'
    outcome=dict(cell_id=cell_id,status=status,planning_summary=str(folder/'summary.json'),
        model_screen=str(folder/'executor_model_screen/model_screen.json') if model else None,
        actual_route_trials=0,not_a_physical_execution_failure=True,attempt_replaced=False)
    runtime.evidence.write_exclusive(target,outcome)
    print('[REVIEW OFFLINE CELL]',outcome,flush=True)
    return outcome


def ablation(cell):
    cell_id=cell['cell_id']
    target=campaign.OUTPUT/'offline_outcomes'/(cell_id+'_no_entry_history')/'outcome.json'
    if not target.exists():
        folder=planning.OUTPUT/(cell_id+'_no_entry_history')
        result=runtime.read(folder/'summary.json') if (folder/'summary.json').exists() else planning.run_cell(cell_id,no_entry_history=True)
        model=None
        if result['plan_success'] and result['independent_model_consistency']=='PASS':
            model=runner.model_screen(cell_id+'_no_entry_history')
        value=dict(cell_id=cell_id+'_no_entry_history',
            status='READY_FOR_OWN_FRESH_STATIC' if model and model['both_nominal_resolutions_pass'] else
                   'MODEL_ADMISSION_REJECTED' if model else 'PLANNING_FAILURE',
            planning_summary=str(folder/'summary.json'),actual_route_trials=0,attempt_replaced=False)
        runtime.evidence.write_exclusive(target,value)
        print('[REVIEW ABLATION CELL]',value,flush=True)
    point=envelope.OUTPUT/cell_id/'envelope_ablation_result.json'
    if not point.exists():envelope.run_cell(cell_id)


def run(*,tasks=('H2','H1','H3','H4','H5','H6'),include_ablations=True):
    protocol=runtime.read(campaign.PROTOCOL)
    marker=campaign.OUTPUT/'offline_worker_reserved.json'
    if marker.exists():
        raise FileExistsError('Worker previously dispatched; inspect outcomes instead of blindly restarting')
    runtime.evidence.write_exclusive(marker,dict(
        protocol=str(campaign.PROTOCOL),protocol_sha256=runtime.sha(campaign.PROTOCOL),
        task_order=list(tasks),include_predeclared_ablations=include_ablations,
        bounded_main_cells=120,no_motion_or_SDK_or_scene_access=True,
        no_case_retry_or_success_replacement=True))
    runtime.lc.freeze_workflow_sources(campaign.OUTPUT/'offline_worker_sources.zip',
        {str(Path(p).resolve()):runtime.sha(p) for p in (__file__,planning.__file__,runner.__file__,envelope.__file__)})
    completed=[]
    try:
        for cell in queue(protocol,tasks):
            result=run_one(cell)
            completed.append(result)
            if include_ablations and cell['task'] in ('H2','H3') and cell['method']==campaign.planning.P:
                ablation(cell)
        runtime.evidence.write_exclusive(campaign.OUTPUT/'offline_worker_complete.json',dict(
            outcomes=completed,actual_PhysX_trials=0,independent_physical_repetitions_NOT_claimed=True))
    except BaseException as error:
        runtime.evidence.write_exclusive(campaign.OUTPUT/'offline_worker_error.json',dict(
            error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),
            completed_main_cells=len(completed),no_automatic_retry=True,actual_PhysX_trials=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks',nargs='+',choices=['H1','H2','H3','H4','H5','H6'],default=['H2','H1','H3','H4','H5','H6'])
    parser.add_argument('--without-ablations',action='store_true')
    args=parser.parse_args()
    run(tasks=tuple(args.tasks),include_ablations=not args.without_ablations)
