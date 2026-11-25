from pathlib import Path
import copy
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_final_seed30 as module


def test_only_P_seed_slice_and_transparent_provenance_change():
    old=module.runtime.read(module.original.PROTOCOL);new=module.amended_definition()
    assert new['main_cells']==old['main_cells']
    assert new['history_ablation_cells']==old['history_ablation_cells']
    assert new['tasks']==old['tasks'] and new['common_speed_policy']==old['common_speed_policy']
    assert new['common_execution_controller_version']==old['common_execution_controller_version']
    assert new['P_seed_portfolio'][0]==old['P_seed_portfolio'][0]
    assert new['P_seed_portfolio'][1]['maximum_s']==30.
    assert new['P_seed_portfolio'][2]==old['P_seed_portfolio'][2]
    assert new['seed_total_budget_s']==45. and new['planning_total_budget_s']==180.
    assert not new['amendment']['untouched_held_out_confirmation']
    assert not new['treatment_frozen_before_new_outcomes']
    assert new['v2_frozen_before_v2_planning_and_ALL_actual_trials']
    assert new['target_NOT_guarantee']['no_exclusion_or_replacement_to_meet_target']


def test_shared_function_bodies_and_private_paths_not_global_patches(tmp_path):
    block=module.campaign(tmp_path)
    assert block.OUTPUT==tmp_path and block.folder_for.__globals__['OUTPUT']==tmp_path
    assert block.plan_cell.__code__ is module.original.plan_cell.__code__
    assert block.run.__code__ is module.original.run.__code__
    assert block.bound_cell.__code__ is module.original.bound_cell.__code__
    assert block.model_cell.__code__ is module.original.model_cell.__code__
    assert block.results_snapshot.__code__ is module.original.results_snapshot.__code__
    assert module.original.OUTPUT!=tmp_path
    assert block.plan_cell.__globals__ is block.run.__globals__
    assert block.plan_cell.__globals__['definition'] is module.amended_definition


def test_actual_seed_budget_and_label_are_30_without_hidden_path_reuse(tmp_path):
    block=module.campaign(tmp_path)
    (tmp_path/'execution_protocol.json').write_text(json.dumps(module.amended_definition()))
    calls=[]
    def search(method,*args):
        calls.append((method,args[-3]))
        return dict(plan_success=len(calls)==2,status='TEST',planning_time_s=.001,expanded_nodes=2)
    value=block.seed_binding()(search,None,None,None,None,None,budget_s=180.,node_limit=60000,policy={})
    assert calls[0][1]<=10. and calls[1][1]==pytest.approx(30.)
    assert value['owned_seed_policy']=='COMMON_B1_10_B0_30_B2_REMAINDER45_v2'
    assert value['no_cached_baseline_plan_or_measured_state_used']


def test_reused_cell_match_includes_failures_before_any_physical_trial(tmp_path,monkeypatch):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    monkeypatch.setattr(module.original,'OUTPUT',old)
    cell=dict(cell_id='H1_B0_f01',method='B0',experiment_role='MAIN')
    cells=[dict(cell,cell_id=f'H1_B0_f{n:02d}') for n in range(54)]
    old_protocol=old/'execution.json';old_protocol.write_text(json.dumps(dict(main_cells=cells)))
    monkeypatch.setattr(module.original,'EXECUTION_PROTOCOL',old_protocol)
    protocol=new/'execution.json';protocol.write_text(json.dumps(dict(main_cells=cells)))
    for index,c in enumerate(cells):
        source=old/'planning'/c['cell_id'];source.mkdir(parents=True)
        (source/'summary.json').write_text(json.dumps(dict(plan_success=index%2==0)))
        target=old/'offline_outcomes'/(c['cell_id']+'.json');target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(dict(cell_id=c['cell_id'],plan_success=index%2==0)))
    block=module.campaign(new);block.EXECUTION_PROTOCOL=protocol;block.PROTOCOL=protocol
    rows=module.reuse_baselines(block)
    assert len(rows)==54 and sum(r['plan_success'] for r in rows)==27
    assert all(not r['new_planning_or_model_trial'] for r in rows)
    assert (new/'planning'/cells[0]['cell_id']).is_symlink()
    assert module.runtime.read(new/'baseline_reuse.json')['baseline_failures_not_excluded']


def test_no_amendment_after_any_final_actual_dispatch(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();new=tmp_path/'new';new.mkdir()
    monkeypatch.setattr(module.original,'OUTPUT',old)
    block=module.campaign(new)
    module.require_unstarted_first_block(block)
    for folder in (old,new):
        marker=folder/'dispatch.json';marker.write_text('{}')
        with pytest.raises(ValueError,match='no consumed final actual'):module.require_unstarted_first_block(block)
        marker.unlink()


def test_targets_do_not_replace_all_declared_denominators_or_guards():
    new=module.amended_definition()
    assert new['all_main_and_ablation_attempts']==78
    assert new['common_actual_goal_position_m']==.030
    assert new['common_actual_goal_heading_deg']==2.
    assert new['common_actual_stopped_speed_m_s']==.005
    assert new['no_retuning_or_successful_retry_replacement_within_block']
    assert new['no_P_PASS_or_metric_dominance_guarantee']
    assert new['clean_restored_execution_noncompletion_continues_for_ALL_methods_including_P']
