"""Read-only paired/ablation/constraint analysis of the complete n3 block.

No actuation, replanning, record replacement, or manuscript mutation. Full
mission RMS is compared only when BOTH paired methods completed. All declared
failures stay in completion denominators. Cruise-only results are a labelled
sensitivity analysis, not a replacement for primary whole-tracking RMS.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import hospital_revision_final_seed30 as experiment
import hospital_revision_cruise_metrics as cruise

runtime=experiment.runtime
FIELDS=('tracking_position_RMS_mm','tracking_yaw_RMS_deg','travel_sim_s','planning_time_s')


def technical_interruption_evidence(rows):
    """Keep externally interrupted observations out of method-failure counts."""
    result=[]
    for row in rows:
        row.update(technical_interruption=False,completion_evaluable=True)
        outcome=runtime.read(row['outcome_path'])
        if not outcome.get('technical_interruption'):
            continue
        proof_path=Path(outcome['separate_camera_boundary_GET'])
        proof=runtime.read(proof_path)
        if (outcome['actual_verdict']!='HOLD' or row['completed']
                or outcome.get('completion_evaluable') is not False
                or proof['status']!='CAMERA_TECHNICAL_INTERRUPTION_RETAINED_NOT_SUCCESS_OR_METHOD_FAILURE'
                or proof['original_report']!=row['report']
                or proof['original_report_sha256']!=runtime.sha(row['report'])
                or proof['original_trace_sha256']!=runtime.sha(row['trace'])):
            raise ValueError('Exact retained camera-interruption proof required')
        row.update(technical_interruption=True,completion_evaluable=False,
            publication_outcome='TECHNICAL_CAMERA_INTERRUPTION_NOT_METHOD_FAILURE',
            technical_proof=str(proof_path),technical_proof_sha256=runtime.sha(proof_path))
        result.append(dict(cell_id=row['cell_id'],paired_condition_id=row['paired_condition_id'],
            outcome=str(row['outcome_path']),proof=str(proof_path),proof_sha256=runtime.sha(proof_path),
            raw_HOLD_and_trace_preserved=True,not_counted_as_algorithmic_failure=True,
            not_counted_as_completed_or_independent_repetition=True))
    return result


def paired_record(candidate,reference):
    completed=bool(candidate['completed'] and reference['completed'])
    return dict(task=candidate['task'],paired_condition_id=candidate['paired_condition_id'],
        candidate=candidate['cell_id'],reference=reference['cell_id'],
        candidate_completed=candidate['completed'],reference_completed=reference['completed'],
        both_completed=completed,
        relative_reductions_percent={key:100*(1-candidate[key]/reference[key])
            if completed and candidate.get(key) is not None and reference.get(key) not in (None,0) else None
            for key in FIELDS},
        reference_interrupted_or_failed_NOT_ranked_by_partial_RMS=not reference['completed'])


def rankings(rows):
    result=[]
    for proposed in rows:
        if proposed['experiment_role']!='MAIN' or proposed['method']!=experiment.original.design.original.planning.P:
            continue
        completed=[r for r in rows if r['experiment_role']=='MAIN'
            and r['paired_condition_id']==proposed['paired_condition_id'] and r['completed']]
        baselines=[r for r in completed if r['method']!=proposed['method']]
        result.append(dict(task=proposed['task'],paired_condition_id=proposed['paired_condition_id'],
            P_completed=proposed['completed'],completed_baseline_count=len(baselines),
            best_position_among_completed=bool(proposed['completed'] and all(
                proposed['tracking_position_RMS_mm']<=r['tracking_position_RMS_mm'] for r in baselines))
                if baselines else None,
            best_yaw_among_completed=bool(proposed['completed'] and all(
                proposed['tracking_yaw_RMS_deg']<=r['tracking_yaw_RMS_deg'] for r in baselines))
                if baselines else None,
            P_only_completed_case_is_NOT_a_measured_RMS_superiority_comparison=not baselines))
    return result


def analyze(block):
    snapshot=block.results_snapshot()
    if snapshot['status']!='COMPLETE':raise ValueError('All78 declared outcomes required before final analysis')
    block.verify_sources()
    rows=snapshot['rows'];paired=[];history=[];cruise_rows=[];constraints=[]
    technical=technical_interruption_evidence(rows)
    for group in snapshot['groups']:
        subset=[r for r in rows if (r['task'],r['experiment_role'],r['method'])==
            (group['task'],group['experiment_role'],group['method'])]
        group['technical_interrupted_n']=sum(r['technical_interruption'] for r in subset)
        group['execution_evaluable_n']=group['accounted_n']-group['technical_interrupted_n']
        group['algorithmic_noncompletion_n']=group['execution_evaluable_n']-group['completed_n']
    P=experiment.original.design.original.planning.P
    B2=experiment.original.design.original.planning.B2
    for row in rows:
        if row['experiment_role']=='HISTORY_ABLATION':
            parent=next(r for r in rows if r['cell_id']==row['cell_id'].removesuffix('_steady_both'))
            history.append(paired_record(parent,row))
        elif row['method']==P:
            for baseline in rows:
                if baseline['experiment_role']=='MAIN' and baseline['method']!=P and baseline['paired_condition_id']==row['paired_condition_id']:
                    paired.append(paired_record(row,baseline))
        if row['completed'] and row['experiment_role']=='MAIN' and row['method'] in (P,B2):
            cruise_rows.append(dict(cell_id=row['cell_id'],task=row['task'],method=row['method'],
                paired_condition_id=row['paired_condition_id'],**cruise.measure(row['report'])))
        if row['method']==P:
            summary=runtime.read(row['planning_summary']);path=Path(summary['plan'])
            plan=runtime.read(path);states=plan.get('selected_states',[])
            activity=plan.get('planning_candidate_model_activity',{})
            constraints.append(dict(cell_id=row['cell_id'],experiment_role=row['experiment_role'],
                plan=str(path),plan_sha256=runtime.sha(path),plan_success=summary['plan_success'],
                selected_primitive_count=len(plan.get('selected_actions',[])),
                maximum_selected_friction_utilization=max((s['friction_utilization'] for s in states),default=None),
                maximum_selected_abs_LTR=max((s['abs_ltr'] for s in states),default=None),
                candidate_activity=activity,threshold_friction_utilization=.8,threshold_abs_LTR=.8,
                excludes_actual_terminal_correction_and_hold=True,
                stored_primitive_peak_values_NOT_motor_performance_guarantees=True))
    P_rows=[r for r in rows if r['experiment_role']=='MAIN' and r['method']==P]
    rank=rankings(rows)
    p_completed=sum(r['completed'] for r in P_rows)
    counts={m:dict(declared_n=18,accounted_n=sum(r['accounted'] for r in rows if r['experiment_role']=='MAIN' and r['method']==m),
        completed_n=sum(r['completed'] for r in rows if r['experiment_role']=='MAIN' and r['method']==m),
        technical_interrupted_n=sum(r['technical_interruption'] for r in rows if r['experiment_role']=='MAIN' and r['method']==m),
        execution_evaluable_n=sum(r['completion_evaluable'] for r in rows if r['experiment_role']=='MAIN' and r['method']==m))
        for m in experiment.original.design.original.METHODS}
    extra=runtime.read(block.OUTPUT/'offline_completed.json')['offline_envelope']
    return dict(protocol=str(block.PROTOCOL),protocol_sha256=runtime.sha(block.PROTOCOL),
        all_declared_counts=counts,main_rows=snapshot['rows'],task_statistics=snapshot['groups'],
        technical_interruptions=technical,
        technical_replacements=snapshot.get('technical_replacements',[]),
        excluded_technical_attempts=snapshot.get('excluded_technical_attempts',[]),
        P_goal=dict(target=15,declared_n=18,completed_n=p_completed,completion_fraction=p_completed/18,
            target_met=p_completed>=15,not_a_guarantee_or_exclusion_rule=True),
        paired_completed_only=paired,history_ablation_pairs=history,
        per_condition_rankings=rank,
        ranking_counts=dict(completed_P=p_completed,
            P_with_at_least_one_completed_baseline=sum(r['P_completed'] and r['completed_baseline_count']>0 for r in rank),
            P_best_position_when_comparable=sum(r['best_position_among_completed'] is True for r in rank),
            P_best_yaw_when_comparable=sum(r['best_yaw_among_completed'] is True for r in rank)),
        cruise_only_sensitivity=cruise_rows,force_load_constraints=constraints,
        offline_envelope_counterfactuals=extra,
        process_interruption_record=str(block.OUTPUT/'process_restart_r01/interruption.json')
            if (block.OUTPUT/'process_restart_r01/interruption.json').exists() else None,
        technical_restarted_static_is_NOT_an_extra_independent_repetition=True,
        interpretation=dict(no_partial_failure_RMS_vs_complete_mission_ranking=True,
            primary_RMS_NOT_replaced_by_selected_cruise_intervals=True,
            common_command_policy_does_NOT_imply_identical_observed_speed_everywhere=True,
            history_comparison_is_same_seed_coefficients_speeds_constraints_and_controller=True,
            envelope_unsafe_candidates_not_physically_driven=True,
            inactive_force_load_screen_does_NOT_prove_tracking_improvement_or_patient_safety=True,
            inspected_benchmark_NOT_untouched_holdout=True,
            three_bounded_initial_conditions_descriptive_NOT_powered_clinical_reliability=True,
            failed_previous_versions_preserved_not_pooled=True,
            actual_platform_photo_deferred_by_author_not_yet_supplied=True,
            manuscript_not_modified=True),source_sha256={str(Path(__file__).resolve()):runtime.sha(__file__)})


def export(block=None,*,output_root=None):
    block=block or experiment.campaign();raw=analyze(block)
    root=Path(output_root) if output_root is not None else block.OUTPUT
    path=root/'feedback_analysis'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')/'paired_feedback_evidence.json'
    runtime.evidence.write_exclusive(path,raw)
    snippets=manuscript_table_rows(raw)
    for name,body in snippets.items():
        with (path.parent/name).open('x',encoding='utf-8') as stream:stream.write(body)
    runtime.evidence.write_exclusive(path.parent/'table_provenance.json',dict(
        source=str(path),source_sha256=runtime.sha(path),
        generated_tex_sha256={name:runtime.sha(path.parent/name) for name in snippets},
        manuscript_NOT_modified=True,all78_outcomes_required=True,
        SD_denominator='n-1',RMS_and_travel_conditional_on_completion=True,
        technical_interruptions_preserved_and_excluded_from_method_failure_denominator=True,
        dagger_in_completion_cell='Technical interruption omitted from evaluable denominator; original record retained',
        SD_for_n_less_than_two_NOT_zero=True,rounding='two decimal places for display only'))
    return path


def statistic_tex(stat,*,include_n=True):
    """A missing statistic or undefined SD is a dash, never a fabricated 0."""
    n=stat['n'];mean=stat['mean'];sd=stat['sample_SD']
    if n==0:
        if mean is not None or sd is not None:raise ValueError('Empty sample must remain unmeasured')
        body=r'\text{--}'
    else:
        if mean is None or (n<2 and sd is not None) or (n>=2 and sd is None):
            raise ValueError('Undefined mean/SD provenance')
        body=f'{mean:.2f}'+r'\pm'+(f'{sd:.2f}' if sd is not None else r'\text{--}')
    return '$'+body+(r'\, ['+str(n)+']' if include_n else '')+'$'


def manuscript_table_rows(raw):
    """Only TeX snippets; the draft remains a separate author-reviewed file."""
    rows=raw['main_rows']
    if len(rows)!=78 or not all(r.get('accounted') for r in rows):
        raise ValueError('All78 accounted outcomes required for final table snippets')
    primary=[];history=[]
    for group in raw['task_statistics']:
        subset=[r for r in rows if (r['task'],r['experiment_role'],r['method'])==
            (group['task'],group['experiment_role'],group['method'])]
        if len(subset)!=3 or group['accounted_n']!=3 or group['declared_n']!=3:
            raise ValueError('Every final group must preserve all three declared conditions')
        pos=statistic_tex(group['completed_tracking_position_RMS_mm'])
        yaw=statistic_tex(group['completed_tracking_yaw_RMS_deg'])
        method=group['method'].split('_')[0]
        if group['experiment_role']=='MAIN':
            planned=sum(r['plan_success'] is True for r in subset)
            technical=sum(r.get('technical_interruption',False) for r in subset)
            completion=f"{group['completed_n']}/{3-technical}"+(r'$^{\dagger}$' if technical else '')
            parts=[group['task'],method,f'{planned}/3',completion,
                statistic_tex(group['planning_time_all_declared_attempts_s'],include_n=False),pos,yaw,
                statistic_tex(group['completed_travel_sim_s'])]
            primary.append(' & '.join(parts)+r'\\')
        else:
            history.append(' & '.join([group['task'],'No history',f"{group['completed_n']}/3",pos,yaw])+r'\\')
        if group['experiment_role']=='MAIN' and method=='P' and group['task'] in ('H2','H3'):
            history.append(' & '.join([group['task'],'P',f"{group['completed_n']}/3",pos,yaw])+r'\\')
    if len(primary)!=24 or len(history)!=4:raise ValueError('Complete primary/history row rosters required')
    return {'primary_rows.tex':'\n'.join(primary)+'\n','history_rows.tex':'\n'.join(history)+'\n'}


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(export())
