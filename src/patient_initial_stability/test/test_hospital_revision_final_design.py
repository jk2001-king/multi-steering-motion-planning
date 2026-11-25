from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_final_design as module


@pytest.mark.parametrize('n',[3,5])
def test_all_feedback_factors_shared_with_fresh_combined_draws(n):
    value=module.definition(n)
    assert len(value['main_cells'])==24*n and len(value['history_ablation_cells'])==2*n
    assert value['status']=='PROPOSAL_NOT_APPROVED_NOT_DISPATCHED_NOT_RESULTS'
    assert value['actual_route_trials']==0 and not value['automatic_dispatch']
    assert value['uniform_P_lookahead_edges_all_tasks']==4
    assert value['tasks']['H6']['wall_budget_s']==1500.
    assert value['tasks']['H6']['simulation_budget_s']==600.
    for pair in {c['paired_condition_id'] for c in value['main_cells']}:
        cells=[c for c in value['main_cells'] if c['paired_condition_id']==pair]
        assert len(cells)==4 and len({c['method'] for c in cells})==4
        assert all(c['start_xyyaw']==cells[0]['start_xyyaw'] for c in cells)
        assert all(c['initial_joint_angles_deg']==cells[0]['initial_joint_angles_deg'] for c in cells)
        assert all(abs(cells[0]['initial_joint_angles_deg'][name])<=.5 for name in module.initial.ACTIVE)
        assert all(abs(cells[0]['initial_joint_angles_deg'][name])<=20. for name in module.initial.CASTERS)
    assert value['nominal_execution_model_prediction_is_diagnostic_for_ALL_methods_and_ablation']
    assert value['no_P_PASS_or_metric_dominance_guarantee']
    assert value['no_retuning_or_successful_retry_replacement_within_block']


def test_rejects_undefined_count_or_reused_development_seed():
    with pytest.raises(ValueError):module.definition(2)
    with pytest.raises(ValueError):module.definition(5,seed=20261121)
    with pytest.raises(ValueError):module.definition(5,seed=2026112102)
