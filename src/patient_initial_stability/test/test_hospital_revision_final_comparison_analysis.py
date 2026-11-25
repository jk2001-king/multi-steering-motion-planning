from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_final_comparison_analysis as module


def row(name='P',completed=True,position=10.,yaw=1.):
    return dict(cell_id=name,task='H1',paired_condition_id='H1_f01',method=name,
        experiment_role='MAIN',completed=completed,tracking_position_RMS_mm=position,
        tracking_yaw_RMS_deg=yaw,travel_sim_s=100.,planning_time_s=10.)


def test_partial_failure_never_gets_an_RMS_reduction():
    pair=module.paired_record(row(),row('B2',False,1.,.1))
    assert not pair['both_completed']
    assert all(v is None for v in pair['relative_reductions_percent'].values())


def test_paired_reductions_keep_worse_results_negative():
    pair=module.paired_record(row(position=10.,yaw=2.),row('B2',position=20.,yaw=1.))
    assert pair['relative_reductions_percent']['tracking_position_RMS_mm']==50.
    assert pair['relative_reductions_percent']['tracking_yaw_RMS_deg']==-100.


def test_only_P_success_is_not_a_numeric_accuracy_superiority_comparison():
    P=module.experiment.original.design.original.planning.P
    value=module.rankings([row(P),row('B2',False)])[0]
    assert value['P_completed']
    assert value['best_position_among_completed'] is None
    assert value['best_yaw_among_completed'] is None


def test_ranking_does_not_hide_yaw_tradeoff():
    P=module.experiment.original.design.original.planning.P
    value=module.rankings([row(P,position=10.,yaw=2.),row('B2',position=20.,yaw=1.)])[0]
    assert value['best_position_among_completed']
    assert not value['best_yaw_among_completed']


def test_incomplete_block_cannot_be_published_as_final():
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='All78'):
        module.analyze(SimpleNamespace(results_snapshot=lambda:dict(status='IN_PROGRESS')))


def test_tex_statistics_never_replace_unmeasured_or_undefined_SD_with_zero():
    assert module.statistic_tex(dict(n=0,mean=None,sample_SD=None))==r'$\text{--}\, [0]$'
    assert module.statistic_tex(dict(n=1,mean=1.234,sample_SD=None))==r'$1.23\pm\text{--}\, [1]$'
    assert module.statistic_tex(dict(n=3,mean=1.234,sample_SD=.321),include_n=False)==r'$1.23\pm0.32$'


def test_table_snippets_require_all78_not_a_success_only_subset():
    with pytest.raises(ValueError,match='All78'):
        module.manuscript_table_rows(dict(main_rows=[dict(accounted=True)]*77))


def test_table_snippet_generation_from_all_groups(tmp_path):
    import copy
    import hospital_revision_final_seed30 as e
    snapshot=e.campaign().results_snapshot()
    snapshot=copy.deepcopy(snapshot)
    for r in snapshot['rows']:r['accounted']=True
    for g in snapshot['groups']:g['accounted_n']=3
    value=module.manuscript_table_rows(dict(main_rows=snapshot['rows'],task_statistics=snapshot['groups']))
    assert len(value['primary_rows.tex'].splitlines())==24
    assert len(value['history_rows.tex'].splitlines())==4
    assert not value['primary_rows.tex'].endswith('\n\n')


def test_postanalysis_does_not_change_frozen_experiment_sources():
    module.experiment.campaign().verify_sources()


def test_camera_interruption_is_not_promoted_or_counted_as_method_failure(monkeypatch):
    observation=dict(cell_id='H6_B1_f02',paired_condition_id='H6_f02',
        outcome_path='outcome',report='report',trace='trace',completed=False)
    records={
        'outcome':dict(technical_interruption=True,actual_verdict='HOLD',
            completion_evaluable=False,separate_camera_boundary_GET='proof'),
        'proof':dict(status='CAMERA_TECHNICAL_INTERRUPTION_RETAINED_NOT_SUCCESS_OR_METHOD_FAILURE',
            original_report='report',original_report_sha256='report-sha',original_trace_sha256='trace-sha')}
    monkeypatch.setattr(module.runtime,'read',lambda p:records[str(p)])
    monkeypatch.setattr(module.runtime,'sha',lambda p:str(p)+'-sha')
    events=module.technical_interruption_evidence([observation])
    assert observation['technical_interruption'] and not observation['completion_evaluable']
    assert not observation['completed']
    assert records['outcome']['actual_verdict']=='HOLD'
    assert events[0]['not_counted_as_algorithmic_failure']
    records['proof']['original_report_sha256']='changed'
    with pytest.raises(ValueError,match='proof'):
        module.technical_interruption_evidence([observation])


def test_completion_table_shows_technical_missingness_explicitly():
    import copy
    snapshot=copy.deepcopy(module.experiment.campaign().results_snapshot())
    for r in snapshot['rows']:r['accounted']=True
    for g in snapshot['groups']:g['accounted_n']=3
    record=next(r for r in snapshot['rows'] if r['cell_id']=='H6_B1_multimode_kinematic_ramp_f02')
    record['technical_interruption']=True
    value=module.manuscript_table_rows(dict(main_rows=snapshot['rows'],task_statistics=snapshot['groups']))
    line=next(s for s in value['primary_rows.tex'].splitlines() if s.startswith('H6 & B1 &'))
    assert r'/2$^{\dagger}$' in line
