"""Append-only reviewer evidence ledger/statistics from stored logs only."""
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import hospital_revision_campaign as campaign
import hospital_revision_metrics as actual

runtime=campaign.runtime


def primary_queue():
    root=campaign.OUTPUT
    if (root/'typed_writeback_resolution_policy.json').exists():
        return root/'execution_queue_v4_typed_writeback'
    continuation=root/'execution_queue_v3_wrapped_placement'
    return continuation if continuation.exists() else root/'execution_queue_v2'


def initial_queue():
    root=campaign.OUTPUT/'steering_initial_supplement'
    policy=root/'joint_writeback_resolution_policy.json'
    return Path(runtime.read(policy)['queue']) if policy.exists() else root/'execution'


def block_state(folder):
    if (folder/'completed.json').exists():return 'FINISHED'
    if (folder/'stopped.json').exists():return 'STOPPED_FOR_INFRASTRUCTURE_REVIEW'
    if (folder/'reservation.json').exists():return 'RUNNING'
    return 'NOT_STARTED'


def measured_entry_conditions(report_path):
    """Distinguish authored initialization from the angles entering tracking."""
    saved=runtime.read(report_path)
    packet=runtime.read(saved['protocol'])
    snapshot=packet['required_tensor_snapshot']
    names=('base_link_to_front_active_steering','base_link_to_rear_active_steering',
           'base_link_to_front_left_caster','base_link_to_front_right_caster',
           'base_link_to_rear_left_caster','base_link_to_rear_right_caster')
    before={name:math.degrees(snapshot['positions'][snapshot['names'].index(name)]) for name in names}
    if runtime.sha(saved['trace'])!=saved['trace_sha256']:
        raise ValueError('Entry-angle trace digest changed')
    first=None
    with Path(saved['trace']).open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['phase']=='path_tracking':
                first=row
                break
    return dict(before_route_pre_hold_angles_deg=before,
        first_path_tracking_time_s=float(first['time_s']) if first else None,
        first_path_tracking_angles_deg={name:math.degrees(float(first['joint_'+name+'_rad']))
                                        for name in names} if first else None,
        authored_angles_are_NOT_assumed_to_persist_after_static_or_pre_hold=True,
        static_and_pre_hold_policies_NOT_bypassed=True)


def describe(values):
    values=[float(v) for v in values if v is not None and math.isfinite(float(v))]
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                sample_standard_deviation=statistics.stdev(values) if len(values)>1 else None,
                standard_deviation_ddof=1)


def wilson95(success,n):
    if not n:return None
    z=1.959963984540054;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def cell_record(cell,no_history=False,*,planning_label=None,live_root=None):
    label=cell['cell_id']+('_no_entry_history' if no_history else '')
    folder=campaign.OUTPUT/'planning'/(planning_label or label)
    summary=runtime.read(folder/'summary.json')
    output=(Path(live_root) if live_root is not None else campaign.OUTPUT/'live')/label
    incident=None
    policy_path=campaign.OUTPUT/'typed_writeback_resolution_policy.json'
    if live_root is None and policy_path.exists():
        policy=runtime.read(policy_path)
        if label==policy['cell_id']:
            original=runtime.read(output/'trial_outcome.json')
            if original.get('actual_route_started') or policy['original_route_attempts']!=0:
                raise ValueError('A started method route cannot be resolved as a setup-only incident')
            if runtime.sha(policy['source_static'])!=policy['source_static_sha256']:
                raise ValueError('Preserved failed setup bytes changed')
            incident=policy['source_static']
            output=Path(policy['qualified_output'])
    joint_policy=campaign.OUTPUT/'steering_initial_supplement/joint_writeback_resolution_policy.json'
    if live_root is not None and joint_policy.exists():
        policy=runtime.read(joint_policy)
        if label==policy['cell_id']:
            original=runtime.read(output/'trial_outcome.json')
            if original.get('actual_route_started') or policy['original_route_attempts']!=0:
                raise ValueError('A started joint-sensitivity route cannot be replaced by setup qualification')
            if runtime.sha(policy['source_static'])!=policy['source_static_sha256']:
                raise ValueError('Original joint-initialization setup evidence changed')
            incident=policy['source_static'];policy_path=joint_policy
            output=Path(policy['qualified_output'])
    row=dict(cell_id=label,task=cell['task'],method='P_no_entry_state_history' if no_history else cell['method'],
        paired_condition_id=cell['paired_condition_id'],
        planning_success=summary['plan_success'],planning_time_s=summary['planning_time_s'],
        planning_status=summary['status'],
        planning_summary=str(folder/'summary.json'),independent_audit=summary['independent_model_consistency'],
        actual_route_started=False,execution_outcome='PENDING',actual_verdict=None,
        actual_success=False,actual_metrics=None)
    if incident:
        row.update(technical_setup_incident_report=incident,setup_resolution_policy=str(policy_path),
            prior_method_route_attempts=0,method_route_attempt_NOT_replaced=True)
    model=folder/'executor_model_screen/model_screen.json'
    row['model_prediction_both_resolutions_pass']=runtime.read(model)['both_nominal_resolutions_pass'] if model.exists() else None
    if not summary['plan_success']:row['execution_outcome']='PLANNING_FAILURE_NO_ROUTE'
    elif summary['independent_model_consistency']!='PASS':row['execution_outcome']='INDEPENDENT_AUDIT_REJECTED_NO_ROUTE'
    if (output/'trial_outcome.json').exists():
        outcome=runtime.read(output/'trial_outcome.json')
        row.update(execution_outcome=outcome['status'],actual_route_started=outcome.get('actual_route_started',False),
                   actual_verdict=outcome.get('actual_verdict'),actual_report=outcome.get('report'))
        if row['actual_route_started']:
            report=runtime.read(outcome['report'])
            row['actual_success']=report['verdict']=='PASS' and all(report['checks'].values())
            row['failed_checks']=[k for k,v in report['checks'].items() if not v]
            row['actual_error']=report.get('error')
            try:
                cached=Path(outcome['report']).parent/'controlled_actual_metrics.json'
                if cached.exists():
                    m=runtime.read(cached)
                    if m['report_sha256']!=runtime.sha(outcome['report']) or m['trace_sha256']!=runtime.sha(m['trace']):
                        raise ValueError('Saved actual metrics or trace digest changed')
                    row['actual_metrics']=m
                else:row['actual_metrics']=actual.metrics(outcome['report'])
            except Exception as e:row['metrics_unavailable']=str(e)
    return row


def tables(rows):
    groups=[]
    for task in ('H1','H2','H3','H4','H5','H6'):
        for method in (*campaign.METHODS,'P_no_entry_state_history'):
            subset=[r for r in rows if r['task']==task and r['method']==method]
            if not subset:continue
            finished=[r for r in subset if r['execution_outcome'] not in ('PENDING','INFRASTRUCTURE_ABORT_BEFORE_ROUTE')]
            success=[r for r in subset if r['actual_success']]
            partial=[r for r in subset if r['actual_route_started'] and not r['actual_success'] and r['actual_metrics']]
            groups.append(dict(task=task,method=method,planned_n=len(subset),resolved_n=len(finished),
                planning_success_n=sum(r['planning_success'] for r in subset),
                actual_route_n=sum(r['actual_route_started'] for r in subset),completed_n=len(success),
                no_route_static_rejection_n=sum(r['execution_outcome']=='STATIC_NOT_PASS_NO_ROUTE' for r in subset),
                infrastructure_abort_n=sum(r['execution_outcome']=='INFRASTRUCTURE_ABORT_BEFORE_ROUTE' for r in subset),
                technical_setup_incident_n=sum(bool(r.get('technical_setup_incident_report')) for r in subset),
                pending_n=len(subset)-len(finished),
                complete_campaign_success_fraction=len(success)/len(subset) if len(finished)==len(subset) else None,
                complete_campaign_success_Wilson95=wilson95(len(success),len(subset)) if len(finished)==len(subset) else None,
                success_fraction_scope='All declared task attempts, not only physically admitted/generated paths; setup rejections separately counted',
                actual_route_completion_fraction=len(success)/sum(r['actual_route_started'] for r in subset)
                    if any(r['actual_route_started'] for r in subset) else None,
                provisional_success_fraction_among_resolved=len(success)/len(finished) if finished else None,
                planning_time_s_all_prespecified_attempts=describe([r['planning_time_s'] for r in subset]),
                all_attempt_planning_time_mean_is_computation_effort_NOT_mean_time_to_solution=True,
                successful_plan_time_s=describe([r['planning_time_s'] for r in subset if r['planning_success']]),
                planning_failure_n=sum(not r['planning_success'] for r in subset),
                planning_failures_elapsed_times_NOT_successful_plan_times=[r['planning_time_s'] for r in subset if not r['planning_success']],
                completed_tracking_position_RMS_mm=describe([r['actual_metrics']['tracking_position_RMS_mm'] for r in success]),
                completed_tracking_yaw_RMS_deg=describe([r['actual_metrics']['tracking_yaw_RMS_deg'] for r in success]),
                completed_travel_sim_s=describe([r['actual_metrics']['controller_elapsed_simulation_s'] for r in success]),
                interrupted_tracking_metrics_NOT_ranked_as_completed_cases=[dict(cell_id=r['cell_id'],
                    position_RMS_mm=r['actual_metrics']['tracking_position_RMS_mm'],
                    yaw_RMS_deg=r['actual_metrics']['tracking_yaw_RMS_deg'],
                    duration_s=r['actual_metrics']['tracking_duration_s']) for r in partial]))
    return groups


def paired_completed(rows):
    result=[]
    for task in ('H1','H2','H3','H4','H5','H6'):
        for comparator in (campaign.planning.B2,'P_no_entry_state_history'):
            pairs=[]
            for name in sorted({r['paired_condition_id'] for r in rows if r['task']==task}):
                selected=[r for r in rows if r['task']==task and r['paired_condition_id']==name]
                p=next((r for r in selected if r['method']==campaign.planning.P),None)
                b=next((r for r in selected if r['method']==comparator),None)
                if not p or not b or not p['actual_success'] or not b['actual_success']:continue
                pm,bm=p['actual_metrics'],b['actual_metrics']
                pairs.append(dict(paired_condition_id=name,P_cell=p['cell_id'],comparator_cell=b['cell_id'],
                    position_RMS_difference_P_minus_comparator_mm=pm['tracking_position_RMS_mm']-bm['tracking_position_RMS_mm'],
                    yaw_RMS_difference_P_minus_comparator_deg=pm['tracking_yaw_RMS_deg']-bm['tracking_yaw_RMS_deg'],
                    position_RMS_reduction_percent=100*(1-pm['tracking_position_RMS_mm']/bm['tracking_position_RMS_mm'])
                        if bm['tracking_position_RMS_mm']>0 else None,
                    yaw_RMS_reduction_percent=100*(1-pm['tracking_yaw_RMS_deg']/bm['tracking_yaw_RMS_deg'])
                        if bm['tracking_yaw_RMS_deg']>0 else None,
                    P_mean_tracking_command_mm_s=pm['tracking_time_weighted_mean_absolute_speed_mm_s']['command_speed_m_s'],
                    comparator_mean_tracking_command_mm_s=bm['tracking_time_weighted_mean_absolute_speed_mm_s']['command_speed_m_s'],
                    P_mean_observed_planar_speed_mm_s=pm['tracking_time_weighted_mean_absolute_speed_mm_s']['observed_planar_speed_m_s'],
                    comparator_mean_observed_planar_speed_mm_s=bm['tracking_time_weighted_mean_absolute_speed_mm_s']['observed_planar_speed_m_s']))
            result.append(dict(task=task,comparator=comparator,paired_completed_n=len(pairs),pairs=pairs,
                position_RMS_reduction_percent=describe([p['position_RMS_reduction_percent'] for p in pairs]),
                yaw_RMS_reduction_percent=describe([p['yaw_RMS_reduction_percent'] for p in pairs]),
                not_an_unpaired_or_all_attempt_completion_analysis=True))
    return result


def collect():
    protocol=runtime.read(campaign.PROTOCOL)
    rows=[]
    for cell in protocol['main_cells']:
        rows.append(cell_record(cell))
        if cell['task'] in ('H2','H3') and cell['method']==campaign.planning.P:
            rows.append(cell_record(cell,True))
    supplemental_root=campaign.OUTPUT/'steering_initial_supplement'
    primary=primary_queue()
    main_done=(primary/'completed.json').exists()
    joint_done=(supplemental_root/'after_primary_result.json').exists()
    supplemental=[]
    if (supplemental_root/'protocol.json').exists():
        for cell in runtime.read(supplemental_root/'protocol.json')['main_cells']:
            row=cell_record(cell,planning_label=cell['parent_main_cell_id'],live_root=supplemental_root/'live')
            row.update(reused_plan_time_reference_s=row['planning_time_s'],planning_time_s=None,
                       no_new_independent_planning_time_measurement=True,
                       prescribed_initial_angles_deg=cell['initial_joint_angles_deg'])
            source=supplemental_root/'live'/cell['cell_id']/'static/initial_joint_initialization_readback.json'
            if row.get('technical_setup_incident_report'):
                source=Path(runtime.read(supplemental_root/'joint_writeback_resolution_policy.json')['qualified_output'])/'static/initial_joint_initialization_readback.json'
            row['actual_initial_condition_readback']=str(source) if source.exists() else None
            if row['actual_route_started']:
                row['measured_route_entry_conditions']=measured_entry_conditions(row['actual_report'])
            supplemental.append(row)
    envelope=[runtime.read(p) for p in sorted((campaign.OUTPUT/'point_reference_planning').glob('*/envelope_ablation_result.json'))]
    activity=[]
    for cell in protocol['main_cells']:
        if cell['method']!=campaign.planning.P:continue
        value=runtime.read(campaign.OUTPUT/'planning'/cell['cell_id']/'summary.json')
        activity.append(dict(cell_id=cell['cell_id'],**value['model_activity']))
    return dict(date_timezone='Asia/Seoul',data_are_actual_saved_logs_NOT_new_experiments=True,
        main_cells=120,history_ablation_cells=10,records=rows,group_statistics=tables(rows),
        paired_completed_accuracy_comparisons=paired_completed(rows),
        steering_initial_supplement=dict(records=supplemental,
            group_statistics=tables(supplemental),paired_completed_accuracy=paired_completed(supplemental),
            separate_from_the_original_n5_pose_only_block=True,
            planning_times_reused_inputs_NOT_new_timing_repetitions=True),
        envelope_ablation=envelope,force_load_candidate_activity=activity,
        reviewer_requirements=dict(
            physical_platform_photo='PENDING_AUTHOR_ORIGINAL_PHOTO; no real-robot driving claim',
            equal_speed='COMMON30MM_S_COMMAND_RULE; report measured command/actual speed, not exactly identical actual velocities',
            dynamic_history_ablation='TEN_DECLARED_PAIRED_CELLS; same controller/coefficients/constraints except five predictive entry states',
            repeated_pose_conditions='FIVE_PAIRED_DRAWS_X_SIX_TASKS_X_FOUR_METHODS; include all negative outcomes',
            initial_steering='SEPARATE_FIVE_PAIRED_JOINT_DRAWS_PER_TASK_FOR_B2_AND_P',
            main_and_history_physical_blocks_status=('SETUP_REQUALIFICATION' if
                primary.name=='execution_queue_v4_typed_writeback' and not primary.exists()
                else block_state(primary)),
            initial_steering_physical_block_status='FINISHED' if joint_done else
                block_state(initial_queue()) if (initial_queue()/'reservation.json').exists()
                else 'SCHEDULED_AFTER_PRIMARY',
            full_envelope_offline_block_status='FINISHED_TEN_DECLARED_CONDITIONS',
            force_load_offline_block_status='FINISHED_THIRTY_PRIMARY_P_ATTEMPTS; NO_FORCE_LOAD_REJECTIONS_OBSERVED',
            manuscript_and_figure_updates='DEFERRED_UNTIL_EXPERIMENT_RESULTS_AS_REQUESTED',
            envelope='H2/H3 matched point-reference planning independently checked with full rectangle',
            force_load='Report activity; no causal benefit or patient/motor safety claim when inactive',
            common_H6='ALL_FOUR_METHODS_1500WALL_600SIM; P_LOOKAHEAD8_ALL_TASKS',
            terminology=dict(development_execution='selected simulation case / prospective repeated trial as applicable',
                PASS_PASS='navigation-action completion and task-completion criteria satisfied',
                hold_entry_target='position tolerance for entering stationary holding')),
        statistics_scope='Descriptive small-sample estimates of starting-condition variation; no invented power/significance/real-world reliability claims',
        original_manuscript_and_old_evidence_unchanged=True)


def export():
    value=collect()
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
    folder=campaign.OUTPUT/'analysis_snapshots'/stamp
    runtime.evidence.write_exclusive(folder/'reviewer_evidence.json',value)
    keys=['cell_id','task','method','paired_condition_id','planning_success','planning_time_s',
          'independent_audit','model_prediction_both_resolutions_pass','execution_outcome',
          'actual_route_started','actual_verdict','actual_success','actual_report','technical_setup_incident_report']
    with (folder/'all_attempts.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(value['records'])
    def summary_csv(path,groups):
        columns=['task','method','planned_n','resolved_n','actual_route_n','completed_n','pending_n',
            'planning_success_n','task_completion_fraction','task_completion_Wilson95_low','task_completion_Wilson95_high',
            'successful_plan_n','successful_plan_time_mean_s','successful_plan_time_sample_SD_s',
            'completed_RMS_n','position_RMS_mean_mm','position_RMS_sample_SD_mm',
            'yaw_RMS_mean_deg','yaw_RMS_sample_SD_deg','travel_sim_mean_s','travel_sim_sample_SD_s']
        with path.open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for group in groups:
                item={k:group[k] for k in columns if k in group}
                ci=group['complete_campaign_success_Wilson95']
                item.update(task_completion_fraction=group['complete_campaign_success_fraction'],
                    task_completion_Wilson95_low=ci[0] if ci else None,
                    task_completion_Wilson95_high=ci[1] if ci else None,
                    successful_plan_n=group['successful_plan_time_s']['n'],
                    successful_plan_time_mean_s=group['successful_plan_time_s']['mean'],
                    successful_plan_time_sample_SD_s=group['successful_plan_time_s']['sample_standard_deviation'],
                    completed_RMS_n=group['completed_tracking_position_RMS_mm']['n'])
                for name,source in [('position_RMS','completed_tracking_position_RMS_mm'),
                                    ('yaw_RMS','completed_tracking_yaw_RMS_deg'),('travel_sim','completed_travel_sim_s')]:
                    unit='mm' if name=='position_RMS' else 'deg' if name=='yaw_RMS' else 's'
                    item[name+'_mean_'+unit]=group[source]['mean']
                    item[name+'_sample_SD_'+unit]=group[source]['sample_standard_deviation']
                writer.writerow(item)
    summary_csv(folder/'main_and_ablation_statistics.csv',value['group_statistics'])
    summary_csv(folder/'joint_initial_statistics.csv',value['steering_initial_supplement']['group_statistics'])
    return folder


def attach_completion_reports():
    """Analysis-only callbacks; no experiment/timeline/target command."""
    from hospital_revision_runner import remote
    path=Path(__file__).resolve()
    source=("import sys,types,builtins\nfrom pathlib import Path\n"
        "assert not hasattr(builtins,'ijcas_revision_final_tables_attached'),'Callbacks already attached'\n"
        f"p=Path({str(path)!r})\n"
        "analysis=types.ModuleType('hospital_revision_reporting');analysis.__file__=str(p);sys.modules[analysis.__name__]=analysis\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),analysis.__dict__)\n"
        "def export_after_revision_block(ignored):\n"
        "    try: print('[REVIEW BLOCK TABLES]',analysis.export(),flush=True)\n"
        "    except Exception as error: print('[REVIEW BLOCK TABLES ERROR]',repr(error),flush=True)\n"
        "for name in ('ijcas_revision_primary_task','ijcas_revision_joint_supplement_waiter'):\n"
        "    job=getattr(builtins,name,None)\n"
        "    assert job is not None,'Own declared block required'\n"
        "    job.add_done_callback(export_after_revision_block)\n"
        "builtins.ijcas_revision_final_tables_attached=True\n"
        "print({'analysis_callbacks_attached':True,'new_physical_trials':0})\n")
    runtime.evidence.write_exclusive(campaign.OUTPUT/'analysis_completion_callbacks.json',dict(
        exact_command=source,source_sha256=runtime.sha(path),no_motion_or_setting_changes=True))
    runtime.lc.freeze_workflow_sources(campaign.OUTPUT/'analysis_completion_sources.zip',
        {str(path):runtime.sha(path),str(Path(actual.__file__).resolve()):runtime.sha(Path(actual.__file__))})
    return remote(source)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attach-completion',action='store_true')
    args=parser.parse_args()
    print(attach_completion_reports() if args.attach_completion else export())
