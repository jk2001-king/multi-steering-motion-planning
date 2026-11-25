"""Resume only unfinished offline envelope checks before the FIRST n3 drives.

The frozen main experiment/planners/controllers are not edited. A deliberately
weakened offline checker can produce a seed rejected by the unchanged shared
reference-replay predicate. That is a negative ablation outcome, not permission
to bypass the predicate or rerun the consumed attempt. All other errors stop.
"""
import argparse
from pathlib import Path
import time

import hospital_revision_final_campaign as base

runtime=base.runtime
GUARD_ERROR='Own spatial reference no longer passes original map/resource predicates'
CONSUMED=('H3_P_dynamics_full_envelope_f01','endpoints')


def expected_offline_rejection(error):
    return type(error) is ValueError and str(error)==GUARD_ERROR


def negative_receipt(name,variant,error,*,already_observed=False):
    if variant not in ('point','endpoints') or not expected_offline_rejection(error):
        raise error
    folder=base.OUTPUT/'offline_envelope'/variant/name
    definition=runtime.read(folder/'definition.json')
    seed=runtime.read(folder/'own_seed.json')
    if not seed['plan_success'] or (folder/'summary.json').exists():
        raise ValueError('Only the consumed, unreturned offline candidate can be classified')
    runtime.retained.runtime.task.verify_pins(definition['input_sha256'])
    value=dict(cell_id=name,method=base.design.original.planning.P,
        experiment_role='OFFLINE_ENVELOPE_ONLY',offline_envelope_variant=variant,
        plan=None,plan_success=False,status='EXISTING_REFERENCE_REPLAY_GUARD_REJECTED',
        error='ValueError: '+str(error),planning_time_s=None,
        planning_time_NOT_recorded_by_original_uncaught_error=True,
        measured_owned_seed_time_s=seed.get('owned_seed_total_time_s'),
        independent_model_consistency='NOT_RUN_NO_ACCEPTED_CANDIDATE',
        independently_valid_reference=False,full_envelope_colliding_samples=None,
        predicate_may_reject_geometry_OR_resources_not_disambiguated=True,
        no_claim_of_measured_collision_or_actual_execution_failure=True,
        guard_NOT_waived_or_changed=True,negative_NOT_relabelled_PASS=True,
        original_already_observed_rejection=already_observed,
        consumed_attempt_NOT_repeated=True,new_PhysX_trials=0,
        unsafe_paths_NOT_physically_executed=True,
        source_definition=str(folder/'definition.json'),source_seed=str(folder/'own_seed.json'),
        frozen_main_protocol=str(base.PROTOCOL),manager_source_sha256=runtime.sha(__file__))
    runtime.evidence.write_exclusive(folder/'final_planning_receipt.json',value)
    return value


def remaining_envelopes():
    results=[]
    for cell in runtime.read(base.PROTOCOL)['main_cells']:
        if cell['method']!=base.design.original.planning.P or cell['task'] not in ('H2','H3'):
            continue
        for variant in ('point','endpoints'):
            name=cell['cell_id'];folder=base.OUTPUT/'offline_envelope'/variant/name
            target=folder/'final_planning_receipt.json'
            if target.exists():
                value=runtime.read(target)
            elif folder.exists():
                if (name,variant)!=CONSUMED:
                    raise ValueError('Unreviewed partial offline folder; no automatic retry: '+str(folder))
                value=negative_receipt(name,variant,ValueError(GUARD_ERROR),already_observed=True)
            else:
                try:
                    value=base.plan_cell(name,envelope_variant=variant)
                except ValueError as error:
                    if not expected_offline_rejection(error): raise
                    value=negative_receipt(name,variant,error)
            results.append(value)
            print('[N3 REMAINING ENVELOPE]',len(results),'/12',name,variant,
                value['plan_success'],value['status'],flush=True)
    if len(results)!=12: raise ValueError('Exactly twelve predeclared offline envelope comparisons required')
    return results


def finish_offline():
    base.verify_sources()
    if (base.OUTPUT/'dispatch.json').exists() or (base.OUTPUT/'execution_reservation.json').exists():
        raise ValueError('This adapter admits only an unstarted FIRST physical block')
    cells=runtime.read(base.EXECUTION_PROTOCOL)['main_cells']
    rows=[runtime.read(base.OUTPUT/'offline_outcomes'/(c['cell_id']+'.json')) for c in cells]
    if len(rows)!=78 or {r['cell_id'] for r in rows}!={c['cell_id'] for c in cells}:
        raise ValueError('All original seventy-eight offline outcomes must already be preserved')
    before=base.current_get()
    runtime.evidence.write_exclusive(base.OUTPUT/'offline_manager_recovery/source_abort.json',dict(
        original_uncaught_error='ValueError: '+GUARD_ERROR,
        original_source='hospital_mission_planning.py:3556',
        failed_offline_variant=CONSUMED,main_offline_outcomes_already_complete=78,
        actual_route_trials=0,no_main_outcome_controller_or_protocol_change=True,
        no_repeated_consumed_planning_attempt=True,before_live_GET=before,
        manager_source_sha256=runtime.sha(__file__)))
    sources={str(Path(__file__).resolve()):runtime.sha(__file__)}
    runtime.lc.freeze_workflow_sources(base.OUTPUT/'offline_manager_recovery/manager_sources.zip',sources)
    extra=remaining_envelopes()
    base.verify_sources()
    value=dict(status='FINAL_N3_OFFLINE_FINISHED',rows=rows,offline_envelope=extra,
        model_HOLD_is_diagnostic_for_ALL=True,actual_motion_trials=0,
        original_expected_ablation_rejection_preserved=True,
        manager_recovery=str(base.OUTPUT/'offline_manager_recovery/source_abort.json'))
    runtime.evidence.write_exclusive(base.OUTPUT/'offline_completed.json',value)
    return value


def run():
    finish_offline()
    print(base.dispatch(),flush=True)
    n=-1
    while not (base.OUTPUT/'completed.json').exists():
        if (base.OUTPUT/'stopped.json').exists():
            raise ValueError('Actual final block stopped; inspect without automatic physical retry')
        snapshot=base.results_snapshot()
        if snapshot['accounted_cells']!=n:
            n=snapshot['accounted_cells'];print('[FINAL N3 PROGRESS]',n,'/78',flush=True)
            if n:
                row=next(r for r in reversed(snapshot['rows']) if r['accounted'])
                print('[FINAL N3 LATEST]',row,flush=True)
        time.sleep(30.)
    completed=runtime.read(base.OUTPUT/'completed.json');time.sleep(.5)
    final=base.current_get(final=True);last=completed['last_actual_report']
    if abs(final['current_time_s']-runtime.read(last)['actual_final_timeline']['current_time_s'])>1e-6:
        raise ValueError('Current final timeline differs from actual saved final report')
    final['final_report_sha256']=runtime.sha(last)
    runtime.evidence.write_exclusive(base.OUTPUT/'final_live_GET.json',final)
    folder=base.export_results()
    runtime.evidence.write_exclusive(base.OUTPUT/'all_steps_completed.json',dict(
        status='FINAL_N3_EXECUTION_AND_REPORTING_FINISHED',analysis_folder=str(folder),
        final_live_GET=str(base.OUTPUT/'final_live_GET.json'),all_failures_preserved=True,
        offline_manager_expected_rejection_handling_only=True))
    print('[FINAL N3 FINISHED]',folder,flush=True)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('offline','all'))
    args=parser.parse_args()
    try:
        print(finish_offline() if args.action=='offline' else run(),flush=True)
    except BaseException as error:
        runtime.evidence.write_exclusive(base.OUTPUT/'offline_manager_recovery/manager_error.json',dict(
            error=type(error).__name__+': '+str(error),no_automatic_retry=True))
        raise
