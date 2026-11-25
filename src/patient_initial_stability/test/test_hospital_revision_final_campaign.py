from pathlib import Path
import copy
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_final_campaign as module


def test_approved_n3_has_all_tasks_methods_and_matched_ablation():
    value=module.definition()
    assert value['status']=='APPROVED_PREDECLARED_NOT_COMPLETED_RESULTS'
    assert value['repeats_per_task_method']==3
    assert len(value['main_cells'])==72 and len(value['history_ablation_cells'])==6
    assert value['uniform_P_lookahead_edges_all_tasks']==4
    assert value['tasks']['H6']['wall_budget_s']==1500.
    assert value['tasks']['H6']['simulation_budget_s']==600.
    cells=module.execution_cells(value)
    assert len(cells)==78 and len({c['cell_id'] for c in cells})==78
    assert [c['task'] for c in cells]==sorted(c['task'] for c in cells)
    for index,cell in enumerate(cells):
        if cell['experiment_role']=='HISTORY_ABLATION':
            parent=cells[index-1]
            assert parent['cell_id']==cell['parent_main_cell_id']
            assert parent['predictive_history_policy']=='carry'
            assert cell['predictive_history_policy']=='steady_both'
            assert cell['start_xyyaw']==parent['start_xyyaw']
            assert cell['initial_joint_angles_deg']==parent['initial_joint_angles_deg']


def test_source_and_goal_acceptance_not_relaxed_for_P_or_ablation():
    value=module.definition()
    assert value['common_actual_goal_position_m']==.030
    assert value['common_actual_goal_heading_deg']==2.
    assert value['common_actual_stopped_speed_m_s']==.005
    assert value['common_fixed_stationary_hold_s']==5.
    assert value['nominal_execution_model_prediction_is_diagnostic_for_ALL_methods_and_ablation']
    assert value['clean_restored_execution_noncompletion_continues_for_ALL_methods_including_P']
    assert value['no_retuning_or_successful_retry_replacement_within_block']
    assert not value['no_P_PASS_or_metric_dominance_guarantee'] is False


def test_seed_order_is_the_validated_common_B1_B0_B2_rule(tmp_path,monkeypatch):
    path=tmp_path/'execution.json'
    path.write_text(json.dumps(module.definition()))
    monkeypatch.setattr(module,'EXECUTION_PROTOCOL',path)
    calls=[]
    def method(name,*args):
        calls.append((name,args[-3],args[-2]))
        return dict(plan_success=len(calls)==2,status='TEST',planning_time_s=.001,expanded_nodes=2)
    value=module.seed_binding()(method,None,None,None,None,None,budget_s=180.,node_limit=60000,policy={})
    assert [c[0] for c in calls]==[module.design.original.METHODS[1],module.design.original.METHODS[0]]
    assert calls[0][1]<=10. and calls[1][1]<=20.
    assert value['owned_seed_policy']=='COMMON_B1_10_B0_20_B2_REMAINDER45_v1'
    assert value['no_cached_baseline_plan_or_measured_state_used']


def test_frozen_protocol_and_sources_cannot_be_replaced(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'OUTPUT',tmp_path)
    monkeypatch.setattr(module,'PROTOCOL',tmp_path/'protocol.json')
    monkeypatch.setattr(module,'EXECUTION_PROTOCOL',tmp_path/'execution.json')
    source=tmp_path/'source.py';source.write_text('original')
    monkeypatch.setattr(module,'implementation_sources',lambda:[source])
    monkeypatch.setattr(module.runtime.lc,'freeze_workflow_sources',lambda *a,**k:None)
    module.export_protocol();module.export_protocol()
    assert len(module.runtime.read(module.PROTOCOL)['main_cells'])==72
    assert len(module.runtime.read(module.EXECUTION_PROTOCOL)['main_cells'])==78
    source.write_text('changed')
    with pytest.raises(ValueError,match='implementation is frozen'):module.export_protocol()


def test_actual_negative_can_continue_only_with_complete_restoration(monkeypatch):
    monkeypatch.setattr(module.predecessor.control.base.shared.execution,'restored_boundary',lambda raw:raw['paused'])
    keys=('all4997_environment_rows_and145_source_settings_and_physics_unchanged',
        'original_all12joint_gains_targets_and_limits_restored','all_frozen_inputs_unchanged',
        'no_unreviewed_USD_edits_or_callback_errors','contacts_healthy_through_actual_Pause')
    raw=dict(paused=True,verdict='HOLD',checks={k:True for k in keys})
    assert module.safe_restored_boundary(raw)
    for key in keys:
        changed=copy.deepcopy(raw);changed['checks'][key]=False
        assert not module.safe_restored_boundary(changed)


def test_predecessor_requires_nine_actual_P_passes(tmp_path,monkeypatch):
    monkeypatch.setattr(module.predecessor,'OUTPUT',tmp_path)
    path=tmp_path/'completed.json'
    monkeypatch.setattr(module.predecessor.control.base.shared.execution,'restored_boundary',lambda raw:True)
    report=tmp_path/'last.json';report.write_text('{}')
    rows=[dict(cell_id=f'H1_P_dynamics_full_envelope_{n}',actual_verdict='PASS') for n in range(9)]
    assert not module.predecessor_ready()
    path.write_text(json.dumps(dict(results=rows,last_actual_report=str(report))))
    assert module.predecessor_ready()
    rows[-1]['actual_verdict']='HOLD'
    path.write_text(json.dumps(dict(results=rows,last_actual_report=str(report))))
    assert not module.predecessor_ready()


def test_final_host_uses_actual_primary_saver_not_development_labels(monkeypatch):
    def prepared():pass
    prepared.__globals__['runtime']=SimpleNamespace(context_for=lambda *a:None,prepare_source=lambda *a:None)
    monkeypatch.setattr(module,'source_binding',lambda cell:prepared)
    cell=module.execution_cells(module.definition())[0]
    bound=module.bound_cell(cell)
    host=bound.__globals__['PrimaryRouteHost']
    recorded={}
    monkeypatch.setattr(module.live.PrimaryRouteHost,'save',lambda self,raw,rows:recorded.update(raw))
    host.save(SimpleNamespace(),dict(verdict='HOLD',checks={'unchanged':False}),[])
    assert recorded['verdict']=='HOLD' and recorded['checks']=={'unchanged':False}
    assert recorded['use_for_final_statistics'] is True
    assert recorded['development_trial_NOT_final_repetition'] is False
    assert recorded['controller_candidate_version']==module.control.VERSION
    assert recorded['predictive_history_policy']=='carry'


def test_summary_keeps_failures_in_denominator_and_excludes_partial_RMS(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'OUTPUT',tmp_path)
    monkeypatch.setattr(module,'PROTOCOL',tmp_path/'protocol.json')
    monkeypatch.setattr(module,'EXECUTION_PROTOCOL',tmp_path/'execution.json')
    value=module.definition();cells=module.execution_cells(value)
    module.PROTOCOL.write_text(json.dumps(value))
    module.EXECUTION_PROTOCOL.write_text(json.dumps(dict(value,main_cells=cells)))
    chosen=[c for c in cells if c['task']=='H1' and c['method']==module.design.original.planning.P]
    for index,cell in enumerate(chosen):
        folder=tmp_path/'planning'/cell['cell_id'];folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(dict(plan_success=True,planning_time_s=index+1)))
        report=tmp_path/(cell['cell_id']+'.json');report.write_text('{}')
        path=tmp_path/'outcomes'/(cell['cell_id']+'.json');path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(actual_verdict='PASS' if index<2 else 'HOLD',
            actual_route_started=True,report=str(report),metrics=dict(
                stationary_hold_and_all_final_checks_satisfied=index<2,
                tracking_position_RMS_mm=[10.,20.,1000.][index],tracking_yaw_RMS_deg=1.,controller_elapsed_simulation_s=50.))))
    snapshot=module.results_snapshot()
    group=next(g for g in snapshot['groups'] if g['task']=='H1' and g['method']==module.design.original.planning.P)
    assert group['declared_n']==3 and group['completed_n']==2
    assert group['completion_fraction_all_declared']==pytest.approx(2/3)
    assert group['completed_tracking_position_RMS_mm']['n']==2
    assert group['completed_tracking_position_RMS_mm']['mean']==15.
    assert group['completed_tracking_position_RMS_mm']['sample_SD']==pytest.approx(2**.5*5)
    assert group['planning_time_all_declared_attempts_s']['n']==3
    assert snapshot['status']=='IN_PROGRESS'
    assert next(r for r in snapshot['rows'] if r['cell_id']==chosen[2]['cell_id'])['tracking_position_RMS_mm']==1000.


def test_single_completed_trial_has_no_invented_sample_SD():
    assert module.describe([None,2.])==dict(n=1,mean=2.,sample_SD=None,ddof=1)


def test_no_implicit_authority_or_physical_loop():
    import asyncio
    with pytest.raises(ValueError,match='Explicit user-approved'):
        asyncio.run(module.run(None,authorized=False))
