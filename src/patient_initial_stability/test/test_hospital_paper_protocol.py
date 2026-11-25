
from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
from pathlib import Path
import importlib.util
import copy
import csv
import json

import pytest

import yaml


WORKSPACE = _ijcas_workspace()
PROTOCOL = (
    _ijcas_path(WORKSPACE, 'studies/IJCAS/config/hospital_paper_validation_protocol.yaml')
)
MEASUREMENT = (
    _ijcas_path(WORKSPACE, 'studies/IJCAS/config/hospital_environment_measurement_template.yaml')
)


def test_hospital_paper_protocol_has_predeclared_matrix():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "draft_pre_registered"
    assert len(protocol["methods"]) == 4
    assert len(protocol["scenarios"]) == 6
    assert "H2_planning_dynamics" in protocol["hypotheses"]
    assert "H1_initial_transient" not in protocol["hypotheses"]
    assert "H4_integrated_mission" in protocol["hypotheses"]
    final = protocol["trial_stages"]["final_matched_comparison"]
    assert final["repeats_per_method_scenario"] == 10
    assert final["paired_seed_required"] is True
    assert final["method_freeze_required"] is True
    assert final["geometry_freeze_required"] is True


def test_scope_keeps_cics_results_separate_and_matches_execution_controllers():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["schema_version"] == 2
    assert protocol["research_scope"]["new_forward_micro_yaw_controller_required"] is False
    assert protocol["research_scope"]["controller_planner_joint_optimization_required"] is False
    assert protocol["scope_revision"]["professor_has_explicitly_confirmed_this_interpretation"] is False
    assert "include_existing_initial_motion_research_in_the_manuscript" in protocol["professor_direction"]["requirements"]
    assert all(method["initial_motion"] == "acceleration_limited_ramp" for method in protocol["methods"].values())
    assert len({method["terminal_controller"] for method in protocol["methods"].values()}) == 1
    archive = yaml.safe_load((PROTOCOL.parent / protocol["supersedes_draft"]).read_text(encoding="utf-8"))
    assert archive["schema_version"] == 1
    assert archive["methods"]["P_full_integrated"]["initial_motion"] == "micro_yaw_initiation_and_body_yaw_recovery"


def test_readiness_audit_does_not_require_the_removed_controller_coupling():
    path = _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/audit_hospital_paper_readiness.py')
    spec = importlib.util.spec_from_file_location("hospital_paper_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert all(audit.protocol_checks_for(protocol).values())
    protocol["methods"]["P_dynamics_full_envelope"]["initial_motion"] = "micro_yaw_initiation_and_body_yaw_recovery"
    assert not audit.protocol_checks_for(protocol)["matched_initial_controller"]


def test_measurement_template_blocks_paper_trials_until_populated():
    measurement = yaml.safe_load(MEASUREMENT.read_text(encoding="utf-8"))
    assert measurement["status"] == "measurement_required"
    assert measurement["bed"]["overall_length_m"] is None
    assert measurement["doorway"]["clear_width_m"] is None
    assert measurement["ward_corridor"]["clear_width_m"] is None
    assert measurement["freeze"]["geometry_frozen"] is False
    audit = measurement["asset_and_collision_audit"]
    assert audit["planner_geometry_generated_from_this_file"] is False
    assert audit["isaac_geometry_generated_from_this_file"] is False
    assert audit["nav2_map_generated_from_this_file"] is False


@pytest.fixture
def ledger_module():
    path=_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/audit_hospital_paper_readiness.py')
    spec=importlib.util.spec_from_file_location('hospital_ledger_software_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _write_test_result(path,*,method='P_dynamics_full_envelope',verdict='FAIL',started=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(run_id=path.stem,method=method,scenario='SOFTWARE_H4',execution_version='SOFTWARE_v3',
        execution_role='development',execution_cohort='SOFTWARE_COHORT',verdict=verdict,
        path_execution_started=started,use_for_final_statistics=False,checks={'completed':verdict=='PASS'},sample_count=0)
    path.write_text(json.dumps(data))
    return data


@pytest.mark.parametrize('role,started,expected',[
    ('method_path',False,'SETUP_OR_PREPATH_NO_PLANNING_PERFORMANCE_OUTCOME'),
    ('method_path',True,'ACTUAL_PATH_OUTCOME'),
    ('static',False,'ENGINEERING_OUTCOME_NOT_PLANNING_COMPARISON'),
    ('initial_motion',True,'ENGINEERING_OUTCOME_NOT_PLANNING_COMPARISON'),
    ('segmented_component',True,'COMPONENT_OUTCOME_NOT_WHOLE_SEQUENCE')])
def test_ledger_does_not_mix_engineering_component_and_method_outcomes(ledger_module,tmp_path,role,started,expected):
    path=tmp_path/'SOFTWARE_NOT_ACTUAL.json'
    _write_test_result(path,started=started)
    row=ledger_module.inspect_execution_record(path,role)
    assert row['reported_verdict']=='FAIL'
    assert row['outcome_stage']==expected
    assert row['final_comparison_qualification_NOT_inferred']
    assert not row['statistical_superiority_established']


def test_ledger_retains_each_attempt_and_separates_environment_and_control_versions(ledger_module,tmp_path):
    rows=[]
    for index,method in enumerate(ledger_module.METHODS):
        path=tmp_path/f'SOFTWARE_{index}.json'
        _write_test_result(path,method=method)
        rows.append(ledger_module.inspect_execution_record(path,'method_path'))
    repeat=copy.deepcopy(rows[-1]);repeat['run_id']='SOFTWARE_second_attempt_PASS';repeat['reported_verdict']='PASS'
    rows.append(repeat)
    groups=ledger_module.development_comparison_groups(rows)
    assert len(groups)==1
    assert groups[0]['method_run_counts']['P_dynamics_full_envelope']==2
    assert groups[0]['actual_path_outcomes']['P_dynamics_full_envelope']==['FAIL','PASS']
    assert not groups[0]['final_comparison_qualification']
    changed=copy.deepcopy(rows[-1]);changed['environment_variant']='OTHER_FLOOR_VERSION'
    assert len(ledger_module.development_comparison_groups([*rows,changed]))==2
    changed=copy.deepcopy(rows[-1]);changed['execution_version']='OTHER_CONTROLLER'
    assert len(ledger_module.development_comparison_groups([*rows,changed]))==2


def test_orphan_motion_CSV_is_pending_not_silently_dropped_or_declared_success(ledger_module,tmp_path):
    directory=tmp_path/'hospital_h4_initial_motion/SOFTWARE_cohort';directory.mkdir(parents=True)
    trace=directory/'hospital_h4_initial_motion_r01.csv'
    with trace.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['time_s','phase','phase_time_s','path_progress_m'])
        writer.writeheader();writer.writerows([dict(time_s=1.,phase='path_tracking',phase_time_s=0.,path_progress_m=.1),
            dict(time_s=2.,phase='terminal_hold',phase_time_s=2.,path_progress_m=.64)])
    before=trace.read_bytes()
    data=ledger_module.build_execution_ledger(tmp_path)
    assert data['records']==[] and len(data['pending_finalization'])==1
    pending=data['pending_finalization'][0]
    assert pending['reported_verdict'] is None and not pending['new_trial_or_success_inferred']
    assert pending['observations']['sample_count']==2
    assert data['qualified_final_cells_proven_by_this_audit']==0 and not data['actual_motion_authorized']
    assert trace.read_bytes()==before


def test_ledger_does_not_trust_final_statistics_flag_or_overwrite_raw_or_duplicate_snapshots(ledger_module,tmp_path):
    path=tmp_path/'hospital_h4_path_execution_v3/SOFTWARE_cohort/hospital_h4_SOFTWARE_r01.json'
    raw=_write_test_result(path,verdict='PASS')
    raw['use_for_final_statistics']=True
    path.write_text(json.dumps(raw));before=path.read_bytes()
    data=ledger_module.build_execution_ledger(tmp_path)
    assert data['claimed_final_records_requiring_separate_frozen_contract_audit']==[str(path)]
    assert data['qualified_final_cells_proven_by_this_audit']==0
    first=ledger_module.write_execution_ledger(tmp_path)
    second=ledger_module.write_execution_ledger(tmp_path)
    assert first==second and len(list(first.parent.glob('*.json')))==1
    assert path.read_bytes()==before


def test_ledger_hash_mismatch_and_nonmonotone_trace_not_a_new_physical_outcome(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_TRACE.json';raw=_write_test_result(path)
    trace=path.with_suffix('.csv');trace.write_text('time_s,phase\n1,path_tracking\n1,path_tracking\n')
    raw.update(trace_sha256='0'*64,sample_count=2);path.write_text(json.dumps(raw))
    row=ledger_module.inspect_execution_record(path,'method_path')
    assert row['reported_verdict']=='FAIL' and row['trace_pin_status']=='TRACE_HASH_MISMATCH'
    assert row['trace_observations']['CSV_structure_errors']==['nonfinite_or_nonincreasing_time']
    assert not row['statistical_superiority_established']


def test_ledger_flags_untraced_method_PASS_without_relabelling_it(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_NO_TRACE.json'
    _write_test_result(path,verdict='PASS',started=False)
    row=ledger_module.inspect_execution_record(path,'method_path')
    assert row['reported_verdict']=='PASS'
    assert set(row['integrity_errors'])=={'PASS_without_physical_trace_samples',
        'method_PASS_without_path_execution_evidence'}
    assert 'physics_baseline_sha256' in row['missing_comparison_controls']
    assert 'source_sha256' in row['missing_comparison_controls']
    assert row['final_comparison_qualification_NOT_inferred']


@pytest.mark.parametrize('field',['physics_baseline_sha256','source_sha256','physical_condition',
    'source_authored_caster_seed_policy','planning_occupancy_digest','shared_candidate_pool_sha256'])
def test_development_groups_expose_physics_source_seed_and_pool_mismatches(ledger_module,tmp_path,field):
    path=tmp_path/'SOFTWARE_MISMATCH.json';_write_test_result(path)
    first=ledger_module.inspect_execution_record(path,'method_path')
    second=copy.deepcopy(first);second['run_id']='SOFTWARE_other';second['comparison_controls'][field]='DIFFERENT'
    group=ledger_module.development_comparison_groups([first,second])[0]
    assert group['control_mismatches']['SOFTWARE_other']==[field]
    assert not group['final_comparison_qualification']


def _software_resource_inputs(module,root):
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path
    cases=[];outputs={};plans=[]
    for index,method in enumerate(module.METHODS):
        plan=write(_ijcas_path(root, 'SOFTWARE_PLAN_' + str(index) + '.json'),dict(method=method,
            reference_center_arc_m=7.6,common_max_command_speed_m_s=.05,common_max_acceleration_m_s2=.02))
        plans.append(str(plan));outputs[str(plan)]=module.file_sha(plan)
        for dt in (.02,.01):
            trace=_ijcas_path(root, f'SOFTWARE_MODEL_{index}_{dt}.csv');trace.write_text('SOFTWARE_ONLY\n')
            cases.append(dict(method=method,integration_dt_s=dt,simulation_duration_s=170.+index,
                hold_entry_model_time_s=165.+index,terminal_pulse_count=0,model_status='MODEL_SCREEN_PASS',
                model_trace=str(trace),model_trace_sha256=module.file_sha(trace)))
    write(_ijcas_path(root, 'hospital_h4_execution_bundle/bundle_r01/execution_bundle.json'),dict(plans=plans,output_sha256=outputs))
    model=write(_ijcas_path(root, 'hospital_h4_common_executor_screen/screen_r01/common_executor_model_screen.json'),dict(cases=cases))
    write(_ijcas_path(root, 'hospital_h4_path_execution_v3/cohort_01_adopted_query_r01/hospital_h4_shared_v3_P_dynamics_full_envelope_r01.json'),
        dict(guards={'maximum_simulation_duration_s':180.,'maximum_wall_time_s':900.},
            acceptance={'hold_observation_s':5.},terminal_controller_config={'terminal_timeout_s':90.}))
    return model,plans


def test_H4_resource_review_keeps_all_methods_and_never_extends_terminal_allowance(ledger_module,tmp_path):
    _software_resource_inputs(ledger_module,tmp_path)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report,pins=ledger_module.frozen_h4_resource_review(tmp_path)
    assert len(report['cases'])==8 and len(report['frozen_plans'])==4
    assert report['minimum_nominal_MODEL_whole_trial_margin_s']==7.
    assert report['terminal_timeout_is_NOT_an_additional_budget_after_tracking']
    assert report['completed_initial_component_changes_entry_requires_new_cohort_for_full_task']
    assert not report['actual_drive_admission']
    assert pins and all(ledger_module.file_sha(Path(path))==sha for path,sha in pins.items())
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['tamper_trace','missing_method','duplicate_case','changed_plan'])
def test_H4_resource_review_never_accepts_changed_or_partial_frozen_references(ledger_module,tmp_path,fault):
    model,plans=_software_resource_inputs(ledger_module,tmp_path)
    data=json.loads(model.read_bytes())
    if fault=='tamper_trace':Path(data['cases'][0]['model_trace']).write_text('CHANGED')
    elif fault=='changed_plan':Path(plans[0]).write_text('{}')
    else:
        if fault=='missing_method':data['cases'].pop()
        else:data['cases'].append(data['cases'][0])
        model.write_text(json.dumps(data))
    with pytest.raises(ValueError):ledger_module.frozen_h4_resource_review(tmp_path)


def _write_metric_trace(path,rows):
    fields=['time_s','phase','phase_time_s','x_m','y_m','path_error_m','path_heading_error_deg',
        'goal_position_error_m','goal_heading_error_deg','command_speed_m_s','vx_physx_m_s','vy_physx_m_s']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({**dict(phase_time_s=0.,x_m=0.,y_m=0.,path_error_m=0.,path_heading_error_deg=0.,
                goal_position_error_m=.001,goal_heading_error_deg=.1,command_speed_m_s=0.,vx_physx_m_s=0.,vy_physx_m_s=0.),**row})


def test_trajectory_metrics_separate_tracking_error_from_terminal_rescue_and_use_fixed_tail(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_TRACE.csv'
    _write_metric_trace(path,[dict(time_s=0.,phase='pre_hold'),
        dict(time_s=1.,phase='path_tracking',x_m=.05,path_error_m=.010,path_heading_error_deg=1.),
        dict(time_s=2.,phase='path_tracking',x_m=.10,path_error_m=.030,path_heading_error_deg=3.),
        dict(time_s=3.,phase='terminal_correction',x_m=.11),
        dict(time_s=4.,phase='terminal_hold',x_m=.11,phase_time_s=0.),
        dict(time_s=6.,phase='terminal_hold',x_m=.11,phase_time_s=2.),
        dict(time_s=9.,phase='terminal_hold',x_m=.11,phase_time_s=5.)])
    metrics=ledger_module.trace_metrics(path)
    assert metrics['tracking_only_position_error_mm']['sample_count']==2
    assert metrics['tracking_only_position_error_mm']['rmse']==pytest.approx(500.**.5)
    assert metrics['tracking_only_position_error_mm']['absolute_p95']==pytest.approx(29.)
    assert metrics['tracking_only_time_weighted_position_RMS_mm']==pytest.approx(500.**.5)
    assert metrics['tracking_only_time_weighted_covered_duration_s']==1.
    assert sum(metrics['sampled_phase_duration_s'].values())==9.
    assert metrics['sampled_phase_travel_m']['terminal_correction']==pytest.approx(.01)
    assert metrics['terminal_correction_is_NOT_included_in_tracking_error']
    assert metrics['terminal_hold_episode_count']==1 and metrics['fixed_tail_duration_s']==3.
    assert metrics['fixed_tail_observations']['regression_speed_mm_s']==0.
    assert metrics['fixed_tail_observations']['maximum_goal_error_mm']==1.
    assert metrics['CSV_structure_errors']==[]


def test_time_weighted_error_does_not_mistake_more_render_frames_for_more_successes(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_NONUNIFORM.csv'
    _write_metric_trace(path,[dict(time_s=0.,phase='path_tracking',path_error_m=.001),
        dict(time_s=.01,phase='path_tracking',path_error_m=.001),
        dict(time_s=1.,phase='path_tracking',path_error_m=.003)])
    metrics=ledger_module.trace_metrics(path)
    assert metrics['tracking_only_time_weighted_position_RMS_mm']==pytest.approx(4.96**.5)
    assert metrics['tracking_only_position_error_mm']['rmse']==pytest.approx((11./3.)**.5)
    assert metrics['telemetry_rows_are_NOT_independent_trials']
    assert metrics['fixed_tail_observations'] is None


def test_multiple_hold_episodes_never_get_merged_into_one_favorable_final_window(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_MULTIPLE_HOLDS.csv'
    _write_metric_trace(path,[dict(time_s=0.,phase='terminal_hold',phase_time_s=2.),
        dict(time_s=3.,phase='terminal_hold',phase_time_s=5.),
        dict(time_s=4.,phase='terminal_correction'),
        dict(time_s=5.,phase='terminal_hold',phase_time_s=2.),
        dict(time_s=8.,phase='terminal_hold',phase_time_s=5.)])
    metrics=ledger_module.trace_metrics(path)
    assert metrics['terminal_hold_episode_count']==2
    assert metrics['fixed_tail_duration_s'] is metrics['fixed_tail_observations'] is None
    assert metrics['CSV_structure_errors']==['multiple_terminal_hold_episodes']


@pytest.fixture(scope='module')
def original_H4_native_plans():
    bundle=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_execution_bundle/bundle_r01/execution_bundle.json')
    if not bundle.is_file():pytest.skip('Frozen H4 planning evidence unavailable')
    return [json.loads(Path(path).read_bytes()) for path in json.loads(bundle.read_bytes())['plans']]


def test_native_planning_review_does_not_turn_inactive_force_stop_predicates_into_causal_performance(
        ledger_module,original_H4_native_plans):
    plans=copy.deepcopy(original_H4_native_plans);before=copy.deepcopy(plans)
    report=ledger_module.native_planning_mechanism_review(plans)
    records={r['method']:r for r in report['methods']};p=records['P_dynamics_full_envelope']
    assert plans==before and len(records)==4
    assert p['independent_predicate_counts']['original_force_load_actuator_limits']==dict(
        pass_count=26,fail_count=0,not_recorded_count=0)
    assert p['independent_predicate_counts']['stopping_reachability']['fail_count']==0
    assert p['independent_predicate_counts']['sampled_full_envelope_clear']['fail_count']==2
    assert p['selected_state_native_MODEL_ranges']['actuator_utilization']['maximum']==1.
    assert records['B1_multimode_kinematic_ramp']['independent_predicate_counts']['stopping_reachability']['not_recorded_count']==26
    assert p['selected_mode_action_counts']=={'coordinated_counterphase':152}
    assert records['B0_fixed_counterphase_ramp']['selected_mode_action_counts']=={'coordinated_counterphase':153}
    assert records['B1_multimode_kinematic_ramp']['planned_mode_switches']==2
    assert not report['common_control_mismatches'] and not report['missing_common_controls']
    assert report['zero_force_or_stop_rejections_is_NOT_proof_of_causal_safety_improvement']
    assert report['selected_state_constraint_saturation_is_NOT_a_rejected_candidate']
    assert not report['actual_drive_admission'] and not report['statistical_superiority_established']


def test_native_cross_model_contrast_uses_same_candidate_not_each_selectors_favorable_choice(
        ledger_module,original_H4_native_plans):
    report=ledger_module.native_planning_mechanism_review(original_H4_native_plans)
    choice=next(r for r in report['selected_candidate_cross_model_observations'] if r['selected_by_method']=='B0_fixed_counterphase_ramp')
    views=choice['native_evaluators']
    assert views['B0_fixed_counterphase_ramp']['admitted_by_native_model'] is True
    assert views['P_dynamics_full_envelope']['admitted_by_native_model'] is False
    assert views['P_dynamics_full_envelope']['position_error_m']>.30
    assert views['B0_fixed_counterphase_ramp']['terminal_speed_m_s'] is None
    assert report['native_models_are_NOT_an_independent_common_model_comparison']
    assert report['model_predictions_NOT_actual_measured_response_validation']


@pytest.mark.parametrize('fault',['missing_method','duplicate_candidate','partial_pool','wrong_pool_id',
    'incorrect_admitted_count','unadmitted_selection','missing_action','nonfinite_model_metric'])
def test_native_planning_review_rejects_missing_or_inconsistent_frozen_evidence(
        ledger_module,original_H4_native_plans,fault):
    plans=copy.deepcopy(original_H4_native_plans);p=plans[-1]
    if fault=='missing_method':plans.pop()
    elif fault=='duplicate_candidate':p['native_candidate_evaluations'][-1]['candidate_id']=p['native_candidate_evaluations'][0]['candidate_id']
    elif fault=='partial_pool':p['native_candidate_evaluations'].pop()
    elif fault=='wrong_pool_id':p['native_candidate_evaluations'][0]['candidate_id']='UNRELATED_CANDIDATE'
    elif fault=='incorrect_admitted_count':p['eligible_candidate_count']+=1
    elif fault=='unadmitted_selection':p['selected_candidate_id']=next(r['candidate_id'] for r in p['native_candidate_evaluations'] if not r['admitted_by_native_model'])
    elif fault=='missing_action':p['selected_actions'].pop()
    else:p['selected_states'][0]['friction_utilization']=float('nan')
    with pytest.raises(ValueError):ledger_module.native_planning_mechanism_review(plans)


def test_native_planning_review_exposes_unmatched_or_absent_controls_without_inventing_equality(
        ledger_module,original_H4_native_plans):
    plans=copy.deepcopy(original_H4_native_plans)
    plans[-1]['shared_controller_config']={'changed':'controller'}
    plans[-1].pop('source_authored_caster_seed_policy')
    result=ledger_module.native_planning_mechanism_review(plans)
    assert set(result['common_control_mismatches']['P_dynamics_full_envelope'])=={
        'shared_controller_config','source_authored_caster_seed_policy'}
    assert result['missing_common_controls']['P_dynamics_full_envelope']==['source_authored_caster_seed_policy']
    assert not result['statistical_superiority_established']


def test_missing_native_planning_details_are_unknown_not_force_constraint_PASS(ledger_module):
    report=ledger_module.native_planning_mechanism_review([{'method':m} for m in ledger_module.METHODS])
    assert report['status']=='NATIVE_PLANNING_DETAILS_NOT_AVAILABLE_NOT_INVENTED'
    assert 'methods' not in report and not report['statistical_superiority_established']


def test_incomplete_hold_statistics_are_diagnostic_not_required_fixed_tail_acceptance(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_DURATION_ABORT.csv'
    _write_metric_trace(path,[dict(time_s=176.9,phase='terminal_hold',phase_time_s=.0166667),
        dict(time_s=178.9,phase='terminal_hold',phase_time_s=2.0166667),
        dict(time_s=180.,phase='terminal_hold',phase_time_s=3.1166667)])
    metrics=ledger_module.trace_metrics(path)
    assert not metrics['fixed_hold_and_tail_observation_complete']
    assert metrics['fixed_hold_remaining_to_5s_s']==pytest.approx(1.8833333)
    assert metrics['fixed_tail_observations'] is None
    assert metrics['partial_hold_tail_observations_NOT_acceptance']['regression_speed_mm_s']==0.
    raw=dict(verdict='FAIL',error='TimeoutError: simulated-duration/control-gap guard',
        guards=dict(maximum_simulation_duration_s=180.))
    original=copy.deepcopy(raw)
    report=ledger_module.execution_duration_diagnosis(raw,metrics)
    assert report['original_verdict']=='FAIL' and not report['completed_fixed_hold_and_required_tail_observed']
    assert report['partial_tail_does_NOT_establish_required_five_second_standstill']
    assert raw==original


def test_actual_H4_timeout_and_common_resource_option_remain_new_design_not_relabelled_PASS(ledger_module):
    root=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
    path=_ijcas_path(root, 'hospital_h4_experiments/cohort_04_full_task_r01/route/hospital_h4_cohort_04_full_task_r01_P_dynamics_full_envelope_r01.json')
    if not path.exists():pytest.skip('Actual fresh H4 first route not recorded')
    before=path.read_bytes();row=ledger_module.inspect_execution_record(path,'method_path')
    assert row['reported_verdict']=='FAIL' and row['outcome_stage']=='ACTUAL_PATH_OUTCOME'
    assert row['trace_observations']['fixed_tail_observations'] is None
    assert row['trace_observations']['fixed_tail_duration_s']==pytest.approx(1.1166667249053717)
    assert row['duration_diagnosis']['recorded_fixed_hold_elapsed_s']==pytest.approx(3.1166668292135)
    assert row['duration_diagnosis']['last_goal_position_error_mm']==pytest.approx(25.466778021853632)
    report,_=ledger_module.frozen_h4_resource_review(root)
    option=report['prospective_common_resource_option']
    assert report['common_simulation_budget_s']==option['original_budget_s']==180.
    assert option['proposed_common_simulation_budget_s']==300.
    assert option['maximum_nominal_component_envelope_s']==pytest.approx(265.3333333333333)
    assert len(option['declared_component_envelopes'])==4
    assert not option['actual_motion_authorized'] and not option['protocol_or_runtime_modified']
    assert path.read_bytes()==before


def _configured_resource_fixture(module,root,cohort,limit,*,common=True):
    path=_ijcas_path(root, 'hospital_h4_experiments', cohort, 'protocol.json');path.parent.mkdir(parents=True)
    guards=dict(maximum_simulation_duration_s=limit,maximum_wall_time_s=900.,maximum_control_gap_s=.25)
    data=dict(version='H4_configured_fresh_single_whole_task_v1',cohort=cohort,
        method=module.METHODS[-1],environment_variant='SOFTWARE_SAME_ENV',guards=guards,
        command_limits={'speed_m_s':.05},acceptance={'hold_observation_s':5.},
        terminal_controller_config={'terminal_timeout_s':90.})
    if common:
        data['common_resource_contract']=dict(version='SOFTWARE_ALL_FOUR_300',methods=list(module.METHODS),
            common_to_all_methods_NOT_P_specific=True,maximum_simulation_duration_s=limit,maximum_wall_time_s=900.,
            initial_pre_hold_s=2.,common_terminal_allocation_s=90.,fixed_terminal_hold_s=5.,
            nominal_action_schedule_s={method:168. for method in module.METHODS},
            round_up_to_whole_minute_s=60.,largest_nominal_plus_common_allocations_s=265.)
    path.write_text(json.dumps(data));return path,data


def _configured_result_fixture(module,protocol,data,name,*,started):
    path=protocol.parent/'route'/name/(f'hospital_h4_{name}_r01.json')
    raw=_write_test_result(path,started=started,verdict='FAIL' if started else 'HOLD')
    for key in ('guards','command_limits','acceptance','terminal_controller_config','environment_variant'):
        raw[key]=copy.deepcopy(data[key])
    raw['execution_cohort']=data['cohort'];path.write_text(json.dumps(raw))
    return module.inspect_execution_record(path,'method_path')


def test_configured_resources_separate_180_and_300_and_prepath_HOLD_from_actual_outcome(ledger_module,tmp_path):
    old,prior=_configured_resource_fixture(ledger_module,tmp_path,'SOFTWARE_180',180.,common=False)
    new,current=_configured_resource_fixture(ledger_module,tmp_path,'SOFTWARE_300',300.)
    rows=[_configured_result_fixture(ledger_module,old,prior,'old_actual_FAIL',started=True),
        _configured_result_fixture(ledger_module,new,current,'prepath_HOLD',started=False)]
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report,pins=ledger_module.configured_h4_resource_inventory(tmp_path,rows)
    versions={row['cohort']:row for row in report['configured_versions']}
    assert len(versions)==2 and not report['unreadable_protocols']
    assert versions['SOFTWARE_180']['declared_resource_limits']['maximum_simulation_duration_s']==180.
    assert versions['SOFTWARE_180']['actual_path_outcome_counts'][ledger_module.METHODS[-1]]==1
    assert not versions['SOFTWARE_180']['all_four_common_resource_declaration_consistent']
    present=versions['SOFTWARE_300']
    assert present['all_four_common_resource_declaration_consistent']
    assert sum(present['actual_path_outcome_counts'].values())==0
    assert present['methods_without_actual_path_outcome']==list(ledger_module.METHODS)
    assert len(present['setup_or_prepath_attempts_NOT_performance_outcomes'])==1
    assert not present['actual_result_contract_mismatches']
    assert not present['actual_motion_authorized'] and not present['final_comparison_qualification']
    assert set(pins)=={str(old),str(new)}
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',[
    'missing_method','duplicate_method','P_only','wrong_guard','missing_schedule','wrong_envelope',
    'wrong_rounding','wrong_terminal','wrong_hold','nonfinite_schedule'])
def test_configured_all_four_contract_does_not_trust_a_resource_version_label(ledger_module,tmp_path,fault):
    path,data=_configured_resource_fixture(ledger_module,tmp_path,'SOFTWARE_invalid',300.)
    contract=data['common_resource_contract']
    if fault=='missing_method':contract['methods'].pop()
    elif fault=='duplicate_method':contract['methods'][-1]=contract['methods'][0]
    elif fault=='P_only':contract['common_to_all_methods_NOT_P_specific']=False
    elif fault=='wrong_guard':contract['maximum_wall_time_s']=901.
    elif fault=='missing_schedule':contract['nominal_action_schedule_s'].pop(ledger_module.METHODS[0])
    elif fault=='wrong_envelope':contract['largest_nominal_plus_common_allocations_s']=264.
    elif fault=='wrong_rounding':contract['round_up_to_whole_minute_s']=59.
    elif fault=='wrong_terminal':data['terminal_controller_config']['terminal_timeout_s']=91.
    elif fault=='wrong_hold':data['acceptance']['hold_observation_s']=6.
    elif fault=='nonfinite_schedule':contract['nominal_action_schedule_s'][ledger_module.METHODS[0]]=float('nan')
    path.write_text(json.dumps(data));before=path.read_bytes()
    report,_=ledger_module.configured_h4_resource_inventory(tmp_path,[])
    row=report['configured_versions'][0]
    assert not row['all_four_common_resource_declaration_consistent']
    assert row['declaration_integrity_errors'] and not row['actual_motion_authorized']
    json.dumps(report,allow_nan=False)
    assert path.read_bytes()==before


def test_configured_result_uses_actual_resource_limits_not_the_declared_version_label(ledger_module,tmp_path):
    path,data=_configured_resource_fixture(ledger_module,tmp_path,'SOFTWARE_mismatch',300.)
    row=_configured_result_fixture(ledger_module,path,data,'wrong_runtime_limit',started=True)
    row['comparison_controls']['guards']['maximum_simulation_duration_s']=180.
    report,_=ledger_module.configured_h4_resource_inventory(tmp_path,[row])
    current=report['configured_versions'][0]
    assert current['actual_result_contract_mismatches']=={row['run_id']:['guards']}
    assert current['actual_path_outcomes_by_method'][row['method']][0]['verdict']=='FAIL'
    assert not current['final_comparison_qualification']


@pytest.mark.parametrize('bad_limit',[True,0.,-1.,float('nan'),float('inf'),None])
def test_unreadable_resource_limits_are_unknown_not_physical_failure(ledger_module,tmp_path,bad_limit):
    path,data=_configured_resource_fixture(ledger_module,tmp_path,'SOFTWARE_bad_limit',300.)
    data['guards']['maximum_simulation_duration_s']=bad_limit;path.write_text(json.dumps(data))
    report,_=ledger_module.configured_h4_resource_inventory(tmp_path,[])
    assert not report['configured_versions'] and len(report['unreadable_protocols'])==1
    assert not report['unreadable_protocols'][0]['physical_outcome_inferred']
    json.dumps(report,allow_nan=False)


def _measured_H4_envelope_fixture(module,tmp_path):
    root=tmp_path/'hospital_paper_gate'
    manifest=_ijcas_path(root, 'hospital_h4_setup/candidate_r01/hospital_h4_geometry_candidate.json')
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(footprint=dict(length_m=1.630,width_m=.965,clearance_m=.010),
        candidate_occupancy_digest='SOFTWARE_MAP_PIN',obstacle_height_slab_m=[.006,.450],
        included_collision_AABBs=[dict(path='/SOFTWARE_ONLY/obstacle',
            world_aabb=dict(minimum_m=[1.5,-.2,.006],maximum_m=[2.,.2,.450]))])))
    path=_ijcas_path(root, 'hospital_h4_experiments/SOFTWARE_ENVELOPE/route/hospital_h4_SOFTWARE_r01.json')
    raw=_write_test_result(path,started=True)
    trace=path.with_suffix('.csv')
    trace.write_text('time_s,phase,phase_time_s,x_m,y_m,yaw_rad\n'
        '1,path_tracking,0,-0.1,0,0\n2,path_tracking,1,0,0,0\n')
    raw.update(scenario='H4_ward_corridor_turn',sample_count=2,trace_sha256=module.file_sha(trace),
        geometry_manifest_sha256=module.file_sha(manifest),planning_occupancy_digest='SOFTWARE_MAP_PIN',
        input_sha256={str(manifest):module.file_sha(manifest)})
    path.write_text(json.dumps(raw))
    return path,manifest,raw


def test_actual_logged_poses_yield_conditional_full_body_gap_not_reference_or_cooked_clearance(ledger_module,tmp_path):
    path,manifest,raw=_measured_H4_envelope_fixture(ledger_module,tmp_path)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    row=ledger_module.inspect_execution_record(path,'method_path')
    observation=row['sampled_authored_envelope_observations']
    assert row['reported_verdict']=='FAIL'
    assert observation['measured_pose_count']==2 and observation['original_mapped_obstacle_count']==1
    assert observation['raw_body_footprint']['minimum_sampled_rectangle_to_authored_AABB_distance_m']==pytest.approx(.685)
    assert observation['expanded_10mm_footprint']['minimum_sampled_rectangle_to_authored_AABB_distance_m']==pytest.approx(.675)
    assert observation['actual_contacts_remain_a_separate_authoritative_channel']
    assert observation['upper_loaded_patient_3D_clearance_NOT_certified']
    assert observation['input_sha256'][str(manifest)]==ledger_module.file_sha(manifest)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['changed_trace','changed_manifest','wrong_own_manifest_pin','different_occupancy','different_footprint'])
def test_actual_clearance_proxy_requires_same_raw_trace_and_frozen_full_footprint(ledger_module,tmp_path,fault):
    path,manifest,raw=_measured_H4_envelope_fixture(ledger_module,tmp_path)
    if fault=='changed_trace':path.with_suffix('.csv').write_text('time_s,phase\n1,path_tracking\n')
    elif fault=='changed_manifest':manifest.write_text('{}')
    elif fault=='wrong_own_manifest_pin':raw['input_sha256'][str(manifest)]='0'*64
    elif fault=='different_occupancy':raw['planning_occupancy_digest']='OTHER_MAP'
    elif fault=='different_footprint':
        setup=json.loads(manifest.read_bytes());setup['footprint']['width_m']=.9
        manifest.write_text(json.dumps(setup));raw['input_sha256'][str(manifest)]=ledger_module.file_sha(manifest)
        raw['geometry_manifest_sha256']=ledger_module.file_sha(manifest)
    path.write_text(json.dumps(raw))
    row=ledger_module.inspect_execution_record(path,'method_path')
    assert row['reported_verdict']=='FAIL'
    observation=row['sampled_authored_envelope_observations']
    assert 'NO_CLEARANCE_COMPUTED' in observation['status']
    assert 'raw_body_footprint' not in observation


def test_saved_prepath_hold_never_receives_nominal_reference_clearance_as_measured_clearance(ledger_module,tmp_path):
    path,manifest,raw=_measured_H4_envelope_fixture(ledger_module,tmp_path)
    path.with_suffix('.csv').unlink()
    raw.update(verdict='HOLD',path_execution_started=False,sample_count=0)
    path.write_text(json.dumps(raw))
    row=ledger_module.inspect_execution_record(path,'method_path')
    assert row['sampled_authored_envelope_observations']=={'status':'NO_ACTUAL_POSE_TRACE_NO_CLEARANCE_INVENTED'}
    assert row['outcome_stage']=='SETUP_OR_PREPATH_NO_PLANNING_PERFORMANCE_OUTCOME'


def test_receipt_and_trace_copies_are_not_counted_as_independent_paired_repeats(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_original.json';_write_test_result(path)
    row=ledger_module.inspect_execution_record(path,'method_path');row['trace']='SOFTWARE_original.csv'
    alias=copy.deepcopy(row);alias['result']='SOFTWARE_readiness_copy.json';alias['trace']='SOFTWARE_copy.csv'
    before=copy.deepcopy([row,alias])
    result=ledger_module.execution_artifact_identity_review([row,alias],
        {'SOFTWARE_original.csv':'SAME_TRACE_BYTES','SOFTWARE_copy.csv':'SAME_TRACE_BYTES'})
    assert result['method_result_artifact_count']==2 and result['distinct_saved_run_id_count']==1
    assert len(result['repeated_run_id_artifacts'])==1
    assert len(result['identical_trace_byte_groups'])==1
    assert not result['final_comparison_qualification']
    assert [row,alias]==before


def test_different_ids_with_identical_deterministic_trace_do_not_prove_same_or_independent_trials(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_first.json';_write_test_result(path)
    row=ledger_module.inspect_execution_record(path,'method_path');row['trace']='first.csv'
    second=copy.deepcopy(row);second.update(run_id='SOFTWARE_second',result='second.json',trace='second.csv')
    report=ledger_module.execution_artifact_identity_review([row,second],{'first.csv':'SAME','second.csv':'SAME'})
    assert report['distinct_saved_run_id_count']==2 and not report['repeated_run_id_artifacts']
    group=report['identical_trace_byte_groups'][0]
    assert group['duplicate_physical_trial_or_independence_NOT_proven_by_bytes_alone']
    assert not report['final_comparison_qualification']


def test_six_task_inventory_adds_H5_and_H6_without_promoting_prepath_or_static(ledger_module,tmp_path):
    for task,role in (('H5','route'),('H6','route'),('H6','static')):
        path=tmp_path/f'hospital_{task.lower()}_experiments/SOFTWARE_cohort/{role}/hospital_{task.lower()}_SOFTWARE_{"static_" if role=="static" else ""}r01.json'
        raw=_write_test_result(path,started=role=='route')
        raw['scenario']='H5_obstacle_bypass_return' if task=='H5' else 'H6_integrated_transfer_mission'
        path.write_text(json.dumps(raw))
    data=ledger_module.build_full_scope_inventory(tmp_path)
    assert len(data['cells'])==24 and len(data['tasks'])==6 and len(data['methods'])==4
    assert data['counts_by_role']=={'METHOD_PATH_DEVELOPMENT':2,'ENGINEERING_STATIC':1}
    for task in ('H5_obstacle_bypass_return','H6_integrated_transfer_mission'):
        cell=next(c for c in data['cells'] if c['task']==task and c['method']==ledger_module.METHODS[-1])
        assert len(cell['saved_path_outcome_artifacts'])==1
        assert not cell['final_matched_cell_qualification']
    assert not data['unresolved_saved_artifacts']
    assert data['final_matched_cells_proven_by_this_inventory']==0
    assert not data['remaining_drive_count_or_53_cell_protocol_adoption_inferred']


def test_six_task_inventory_preserves_all_H5_fail_hold_pass_artifacts_and_versions(ledger_module,tmp_path):
    for index,(verdict,started) in enumerate((('FAIL',True),('HOLD',False),('PASS',True))):
        path=tmp_path/f'hospital_h5_experiments/cohort_{index}/route/hospital_h5_SOFTWARE_r01.json'
        raw=_write_test_result(path,verdict=verdict,started=started)
        raw.update(scenario='H5_obstacle_bypass_return',execution_version=f'SOFTWARE_v{index}')
        path.write_text(json.dumps(raw))
    data=ledger_module.build_full_scope_inventory(tmp_path)
    cell=next(c for c in data['cells'] if c['task']=='H5_obstacle_bypass_return' and c['method']==ledger_module.METHODS[-1])
    assert [r['reported_verdict'] for r in cell['saved_path_outcome_artifacts']]==['FAIL','PASS']
    assert len(cell['setup_or_prepath_artifacts'])==1
    assert len(data['development_groups'])==3
    assert 'PASS_without_physical_trace_samples' in cell['saved_path_outcome_artifacts'][1]['integrity_errors']


def test_six_task_inventory_never_runs_clearance_queries_for_artifact_accounting(ledger_module,tmp_path,monkeypatch):
    path=tmp_path/'hospital_h4_experiments/SOFTWARE/route/hospital_h4_SOFTWARE_r01.json'
    raw=_write_test_result(path);raw['scenario']='H4_ward_corridor_turn';path.write_text(json.dumps(raw))
    def forbidden(*args,**kwargs):raise AssertionError('No geometric re-evaluation in saved-artifact coverage')
    monkeypatch.setattr(ledger_module,'actual_h4_authored_envelope_observations',forbidden)
    assert ledger_module.build_full_scope_inventory(tmp_path)['counts_by_role']=={'METHOD_PATH_DEVELOPMENT':1}


def test_six_task_inventory_exposes_incomplete_result_without_physical_or_live_inference(ledger_module,tmp_path):
    path=tmp_path/'hospital_h6_experiments/SOFTWARE/route/hospital_h6_SOFTWARE_r01.json'
    path.parent.mkdir(parents=True);path.write_text('{"verdict":')
    before=path.read_bytes();data=ledger_module.build_full_scope_inventory(tmp_path)
    assert len(data['unresolved_saved_artifacts'])==1 and not data['records']
    assert data['file_absence_NOT_live_state_or_process_completion_evidence']
    assert not data['actual_motion_authorized'] and path.read_bytes()==before


def test_six_task_inventory_zero_command_trace_is_engineering_not_H6_route(ledger_module,tmp_path):
    path=tmp_path/'hospital_h6_experiments/SOFTWARE/retained_zero_command_settling_r01/zero_settling_readback.json'
    path.parent.mkdir(parents=True);trace=path.with_name('zero_settling_trace.csv')
    trace.write_text('time_s,x_m,y_m\n0,1,2\n1,1,2\n')
    path.write_text(json.dumps(dict(verdict='HOLD',sample_count=2,trace=str(trace),
        trace_sha256=ledger_module.file_sha(trace),checks={'complete':False},error='SOFTWARE_ONLY')))
    data=ledger_module.build_full_scope_inventory(tmp_path)
    zero=data['H6_zero_command_engineering'][0]
    assert zero['sample_count']==2 and not zero['integrity_errors']
    assert not zero['new_trial_identity_or_route_success_inferred']
    assert not data['records'] and not data['unresolved_saved_artifacts']
    assert all(not c['saved_path_outcome_artifacts'] for c in data['cells'])


def test_six_task_inventory_pins_raw_inputs_and_is_idempotent(ledger_module,tmp_path):
    path=tmp_path/'hospital_h5_experiments/SOFTWARE/route/hospital_h5_SOFTWARE_r01.json'
    _write_test_result(path);before=path.read_bytes()
    first=ledger_module.write_full_scope_inventory(tmp_path)
    assert ledger_module.write_full_scope_inventory(tmp_path)==first
    data=json.loads(first.read_bytes())
    assert data['input_sha256'][str(path)]==ledger_module.file_sha(path)
    assert len(list(first.parent.iterdir()))==2 and path.read_bytes()==before
    summary=next(first.parent.glob('*_ko.txt')).read_text()
    assert '48+5=53은 여전히 미채택 계획안' in summary
    assert '독립 실험 횟수·성공률·최종 반복 통계가 아니다' in summary


def _software_scope_plans(module,root):
    folder=_ijcas_path(root, 'hospital_h6_planning/heading0_fresh_terminal_beam_matched_dev_r19')
    folder.mkdir(parents=True);methods=[];outputs={}
    for method in module.METHODS:
        path=folder/(method+'.json');path.write_text(json.dumps(dict(method=method,plan_success=True)))
        audit=folder/(method+'_independent_audit.json')
        audit.write_text(json.dumps(dict(plan=str(path),plan_sha256=module.file_sha(path),verdict='PASS')))
        outputs.update({str(p):module.file_sha(p) for p in (path,audit)})
        methods.append(dict(method=method,plan_file=str(path),plan_success=True,planning_time_s=1.,
            path_length_m=1.,action_count=1,mode_switch_count=0,independent_OFFLINE_audit='PASS',
            accepted_offline_start_goal_plan=True))
    path=folder/'planning_review.json';path.write_text(json.dumps(dict(methods=methods,output_sha256=outputs)))
    return path


@pytest.mark.parametrize('mutation',['none','plan','audit','missing_method'])
def test_six_task_inventory_offline_plans_and_audits_never_count_as_physical_drives(ledger_module,tmp_path,mutation):
    path=_software_scope_plans(ledger_module,tmp_path)
    if mutation in ('plan','audit'):
        target=path.parent/(ledger_module.METHODS[0]+('_independent_audit' if mutation=='audit' else '')+'.json')
        target.write_text(target.read_text()+' ')
    if mutation=='missing_method':
        raw=json.loads(path.read_bytes());raw['methods'].pop();path.write_text(json.dumps(raw))
    data=ledger_module.build_full_scope_inventory(tmp_path)
    assert not data['records'] and all(not c['saved_path_outcome_artifacts'] for c in data['cells'])
    if mutation=='none':
        assert len(data['H6_declared_offline_planning']['methods'])==4 and not data['unresolved_saved_artifacts']
        assert not data['H6_declared_offline_planning']['actual_drive_success_inferred']
    else:assert len(data['unresolved_saved_artifacts'])==1


def test_six_task_inventory_catches_trace_mutation_during_inspection(ledger_module,tmp_path,monkeypatch):
    path=tmp_path/'hospital_h5_experiments/SOFTWARE/route/hospital_h5_SOFTWARE_r01.json'
    _write_test_result(path);trace=path.with_suffix('.csv');trace.write_text('time_s,phase\n0,path_tracking\n')
    original=ledger_module.inspect_execution_record
    def changed(*args,**kwargs):
        row=original(*args,**kwargs);trace.write_text(trace.read_text()+'1,path_tracking\n');return row
    monkeypatch.setattr(ledger_module,'inspect_execution_record',changed)
    data=ledger_module.build_full_scope_inventory(tmp_path)
    assert not data['records'] and 'Trace changed' in data['unresolved_saved_artifacts'][0]['error']


def test_one_actual_development_figure_is_pinned_readonly_and_idempotent(ledger_module,tmp_path,monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'matplotlib_cache'))
    pytest.importorskip('matplotlib')
    result=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if not result.exists():pytest.skip('Actual completed H4 P not recorded')
    before={path:ledger_module.file_sha(path) for path in (result,result.with_suffix('.csv'))}
    output=tmp_path/'derived_figure'
    manifest=ledger_module.write_execution_figure(result,output)
    saved=json.loads(manifest.read_bytes())
    assert saved['raw_verdict']=='PASS' and saved['measured_pose_count']>0
    assert saved['trace_observations']['fixed_hold_and_tail_observation_complete']
    assert not saved['use_for_final_statistics'] and not saved['original_results_modified']
    assert saved['no_independent_repeat_or_statistical_superiority_inferred']
    assert len(saved['output_sha256'])==2
    assert all(Path(p).stat().st_size>1000 and ledger_module.file_sha(p)==sha for p,sha in saved['output_sha256'].items())
    first_outputs={p:ledger_module.file_sha(p) for p in output.iterdir()}
    assert ledger_module.write_execution_figure(result,output)==manifest
    assert first_outputs=={p:ledger_module.file_sha(p) for p in output.iterdir()}
    assert before=={p:ledger_module.file_sha(p) for p in before}


def test_prepath_HOLD_never_gets_a_measured_route_figure(ledger_module,tmp_path,monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'matplotlib_cache'))
    pytest.importorskip('matplotlib')
    path=tmp_path/'SOFTWARE_HOLD.json';raw=_write_test_result(path,started=False,verdict='HOLD')
    raw['scenario']='H4_ward_corridor_turn';path.write_text(json.dumps(raw));before=path.read_bytes()
    output=tmp_path/'never_created'
    with pytest.raises(ValueError,match='actual H4 outcome'):ledger_module.write_execution_figure(path,output)
    assert not output.exists() and path.read_bytes()==before


def _clock_overlap_rows(times,*,x_offset=0.,yaw=0.):
    import math
    measured=[dict(time_s=t,x_m=t+x_offset,y_m=0.,yaw_rad=yaw,
        vx_physx_m_s=.03*math.cos(yaw),vy_physx_m_s=.03*math.sin(yaw)) for t in times]
    nominal=[dict(time_s=t,x_m=t,y_m=0.,yaw_rad=yaw,u_m_s=.03) for t in times]
    return measured,nominal


def test_equal_clock_residual_keeps_initial_offset_and_never_counts_it_as_tracking_error(ledger_module):
    measured,nominal=_clock_overlap_rows([0.,1.,3.],x_offset=.02,yaw=1.2)
    report,rows=ledger_module.nominal_measured_clock_overlap(measured,nominal)
    assert report['time_weighted_equal_clock_position_RMS_mm']==pytest.approx(20.)
    assert report['time_weighted_body_forward_velocity_residual_RMS_mm_s']==pytest.approx(0.,abs=1e-10)
    assert report['initial_measured_minus_nominal_world_xy_mm']==[20.,0.]
    assert report['absolute_clocks_used_without_time_shift_or_pose_bias_removal']
    assert report['equal_clock_error_includes_progress_timing_error_NOT_path_tracking_error']
    assert report['closed_loop_commands_can_differ_NOT_identical_input_system_identification']
    assert len(rows)==3 and all(row['position_difference_mm']==pytest.approx(20.) for row in rows)


def test_clock_overlap_does_not_extrapolate_a_model_endpoint_over_unpredicted_physical_time(ledger_module):
    measured,_=_clock_overlap_rows([0.,.5,1.,2.,5.]);_,nominal=_clock_overlap_rows([0.,.5,1.])
    report,rows=ledger_module.nominal_measured_clock_overlap(measured,nominal)
    assert report['overlap_route_clock_s']==[0.,1.]
    assert report['measured_samples_outside_model_support']==2
    assert report['measured_end_after_nominal_end_s']==4.
    assert [r['time_s'] for r in rows]==[0.,.5,1.]
    assert report['time_weighted_equal_clock_position_RMS_mm']==0.


def test_equal_clock_heading_interpolation_unwraps_model_and_wraps_the_residual(ledger_module):
    import math
    measured,_=_clock_overlap_rows([0.,1.,2.]);_,nominal=_clock_overlap_rows([0.,2.])
    for row,angle in zip(measured,[179.,180.,-179.]):row['yaw_rad']=math.radians(angle)
    for row,angle in zip(nominal,[179.,-179.]):row['yaw_rad']=math.radians(angle)
    report,rows=ledger_module.nominal_measured_clock_overlap(measured,nominal)
    assert report['time_weighted_equal_clock_wrapped_heading_RMS_deg']==pytest.approx(0.,abs=1e-10)
    assert rows[1]['wrapped_heading_difference_deg']==pytest.approx(0.,abs=1e-10)


@pytest.mark.parametrize('fault',['duplicate_time','backward_time','nonfinite_time','nonfinite_pose','missing_velocity','no_overlap'])
def test_clock_overlap_rejects_missing_nonfinite_nonmonotonic_or_unshared_data(ledger_module,fault):
    measured,nominal=_clock_overlap_rows([0.,1.,2.])
    if fault=='duplicate_time':measured[1]['time_s']=0.
    elif fault=='backward_time':nominal[2]['time_s']=.5
    elif fault=='nonfinite_time':nominal[1]['time_s']=float('nan')
    elif fault=='nonfinite_pose':measured[1]['x_m']=float('nan')
    elif fault=='missing_velocity':del measured[1]['vx_physx_m_s']
    else:
        for row in measured:row['time_s']+=10.
    with pytest.raises(ValueError):ledger_module.nominal_measured_clock_overlap(measured,nominal)


def test_actual_frozen_prediction_review_retains_both_original_resolutions_and_raw_outcome(ledger_module,tmp_path,monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'matplotlib_cache'))
    pytest.importorskip('matplotlib')
    result=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if not result.exists():pytest.skip('Actual P record not available')
    pins={p:ledger_module.file_sha(p) for p in (result,result.with_suffix('.csv'))}
    output=tmp_path/'derived_prediction'
    target=ledger_module.write_frozen_prediction_review(result,output)
    report=json.loads(target.read_bytes())
    assert report['raw_verdict']=='PASS' and report['actual_complete_fixed_hold']
    assert len(report['model_resolution_rows'])==2
    assert {r['integration_dt_s'] for r in report['model_resolution_rows']}=={.01,.02}
    assert report['actual_terminal_pulse_count']==2
    assert all(r['nominal_terminal_pulse_count']==0 for r in report['model_resolution_rows'])
    assert all(r['actual_end_minus_nominal_end_s']>19. for r in report['model_resolution_rows'])
    assert all(r['equal_clock_overlap']['measured_samples_outside_model_support']>0 for r in report['model_resolution_rows'])
    assert not report['actual_model_admission'] and not report['use_for_final_statistics']
    assert report['new_PhysX_trial_count']==0 and report['prediction_archives_pre_pinned_in_executed_protocol_NOT_refitted']
    assert len(report['output_sha256'])==4
    outputs={p:ledger_module.file_sha(p) for p in output.iterdir()}
    assert ledger_module.write_frozen_prediction_review(result,output)==target
    assert outputs=={p:ledger_module.file_sha(p) for p in output.iterdir()}
    assert pins=={p:ledger_module.file_sha(p) for p in pins}


def test_prediction_review_never_promotes_an_unstarted_HOLD(ledger_module,tmp_path):
    path=tmp_path/'SOFTWARE_HOLD.json';raw=_write_test_result(path,started=False,verdict='HOLD')
    raw['scenario']='H4_ward_corridor_turn';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='actual H4 outcome'):
        ledger_module.write_frozen_prediction_review(path,tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


@pytest.mark.parametrize('fault',['screen_not_prepinned','changed_control_source','protocol_pin','plan_pin','trace_pin'])
def test_frozen_prediction_review_requires_its_own_preexecuted_pins_and_common_control(ledger_module,tmp_path,fault):
    original=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if not original.exists():pytest.skip('Actual P record not available')
    raw=json.loads(original.read_bytes());result=tmp_path/'SOFTWARE_evidence_copy.json'
    trace=result.with_suffix('.csv');trace.write_bytes(original.with_suffix('.csv').read_bytes())
    if fault=='screen_not_prepinned':
        screen=ledger_module.RESULT/'hospital_h4_common_executor_screen/screen_r01/common_executor_model_screen.json'
        raw['input_sha256'].pop(str(screen))
    elif fault=='changed_control_source':raw['source_sha256']['hospital_h4_path_execution.py']='0'*64
    elif fault=='protocol_pin':raw['path_protocol_sha256']='0'*64
    elif fault=='plan_pin':raw['plan_sha256']='0'*64
    else:raw['trace_sha256']='0'*64
    result.write_text(json.dumps(raw));output=tmp_path/'not_created'
    with pytest.raises(ValueError):ledger_module.write_frozen_prediction_review(result,output)
    assert not output.exists()


def test_failed_180s_actual_outcome_is_not_promoted_by_a_completed_nominal_prediction(ledger_module,tmp_path,monkeypatch):
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'matplotlib_cache'))
    pytest.importorskip('matplotlib')
    original=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_04_full_task_r01/route/hospital_h4_cohort_04_full_task_r01_P_dynamics_full_envelope_r01.json')
    if not original.exists():pytest.skip('Actual original 180s FAIL not available')
    before=original.read_bytes()
    target=ledger_module.write_frozen_prediction_review(original,tmp_path/'failed_actual_review')
    report=json.loads(target.read_bytes())
    assert report['raw_verdict']=='FAIL' and not report['actual_complete_fixed_hold']
    assert all(r['endpoints_after_own_completed_holds'] is None for r in report['model_resolution_rows'])
    assert all(r['incomplete_actual_or_model_hold_is_NOT_compared_as_completed_endpoints'] for r in report['model_resolution_rows'])
    assert not report['actual_model_admission'] and report['original_controller_verdict_NOT_relabelled']
    assert before==original.read_bytes()


def test_explicit_development_export_never_promotes_one_run_to_a_four_method_comparison(ledger_module,tmp_path):
    result=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if not result.exists():pytest.skip('Actual P not present')
    before=result.read_bytes();output=tmp_path/'one_run'
    path=ledger_module.write_explicit_development_comparison([result],output)
    report=json.loads(path.read_bytes())
    assert len(report['actual_development_rows'])==1
    assert not report['configuration_matching_checks_passed'] and report['P_vs_B2_individual_development_contrast'] is None
    assert len(report['missing_methods'])==3 and report['qualified_final_comparison_cells']==0
    assert not report['independent_paired_repeats_established'] and not report['statistical_superiority_established']
    assert before==result.read_bytes()
    assert ledger_module.write_explicit_development_comparison([result],output)==path


def test_explicit_development_export_rejects_duplicate_receipts_and_unstarted_HOLD(ledger_module,tmp_path):
    result=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if result.exists():
        with pytest.raises(ValueError,match='distinct actual result paths'):
            ledger_module.write_explicit_development_comparison([result,result],tmp_path/'duplicate')
    path=tmp_path/'SOFTWARE_HOLD.json';raw=_write_test_result(path,started=False,verdict='HOLD')
    raw['scenario']='H4_ward_corridor_turn';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='Actual intact H4 outcomes'):
        ledger_module.write_explicit_development_comparison([path],tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


def test_explicit_development_export_does_not_pool_old_180s_FAIL_with_new_300s_PASS(ledger_module,tmp_path):
    root=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments')
    old=_ijcas_path(root, 'cohort_04_full_task_r01/route/hospital_h4_cohort_04_full_task_r01_P_dynamics_full_envelope_r01.json')
    new=_ijcas_path(root, 'cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if not old.exists() or not new.exists():pytest.skip('Actual separate resource-version outcomes not present')
    target=ledger_module.write_explicit_development_comparison([old,new],tmp_path/'different_versions')
    report=json.loads(target.read_bytes())
    assert {r['raw_verdict'] for r in report['actual_development_rows']}=={'PASS','FAIL'}
    assert not report['configuration_matching_checks_passed']
    assert any('guards' in fields for fields in report['physical_control_or_domain_mismatches'].values())
    assert report['P_vs_B2_individual_development_contrast'] is None
    assert report['original_180s_and_other_environment_versions_NOT_pooled']


def test_explicit_comparison_does_not_relabel_a_P_trace_as_a_B2_method_trial(ledger_module,tmp_path):
    original=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json')
    if not original.exists():pytest.skip('Actual P not present')
    raw=json.loads(original.read_bytes());raw['method']='B2_multimode_transition_cost_ramp'
    path=tmp_path/'SOFTWARE_relabeled.json';path.write_text(json.dumps(raw))
    path.with_suffix('.csv').write_bytes(original.with_suffix('.csv').read_bytes())
    with pytest.raises(ValueError,match='own method/plan identity'):
        ledger_module.write_explicit_development_comparison([path],tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


def _phase_response_rows():
    import math
    rows=[dict(time_s=t,phase='path_tracking',action_index=index,command_speed_m_s=.01,
        vx_physx_m_s=0.,vy_physx_m_s=.01,yaw_rad=math.pi/2,path_error_m=error,
        path_heading_error_deg=.5,command_curvature_1_m=.8)
        for t,index,error in [(.1,0,.01),(.3,0,.01),(1.3,1,.02),(2.3,1,.02)]]
    actions=[dict(curvature_1_m=0.),dict(curvature_1_m=.2)]
    return rows,actions


def test_phase_response_projects_world_velocity_and_uses_frozen_not_feedback_curvature(ledger_module):
    rows,actions=_phase_response_rows();review=ledger_module.measured_phase_response(rows,actions)
    groups={r['segment']:r for r in review['segments']}
    assert groups['tracking_straight']['duration_s']==pytest.approx(.2)
    assert groups['tracking_curve']['duration_s']==pytest.approx(1.)
    assert groups['boundary_crossing']['duration_s']==pytest.approx(1.)
    assert groups['tracking_straight']['tracking_position_RMS_mm']==pytest.approx(10.)
    assert groups['tracking_curve']['tracking_position_RMS_mm']==pytest.approx(20.)
    assert groups['boundary_crossing']['tracking_position_RMS_mm'] is None
    for group in groups.values():
        assert group['mean_measured_body_forward_speed_mm_s']==pytest.approx(10.)
        assert group['time_weighted_speed_residual_RMS_mm_s']==pytest.approx(0.,abs=1e-10)
    assert review['observed_duration_s']==pytest.approx(2.2)
    assert review['partition_duration_s']==pytest.approx(2.2)
    assert sum(group['interval_count'] for group in groups.values())==len(rows)-1
    assert review['all_original_adjacent_intervals_accounted_for']
    assert review['startup_before_first_CSV_sample_NOT_invented']


def test_phase_response_retains_phase_transition_duration_without_polluting_tracking_RMS(ledger_module):
    rows,actions=_phase_response_rows();rows[2]['phase']='terminal_braking';rows[3]['phase']='terminal_braking'
    rows[2]['path_error_m']=100.;rows[3]['path_error_m']=100.
    review=ledger_module.measured_phase_response(rows,actions)
    groups={r['segment']:r for r in review['segments']}
    assert review['phase_duration_right_endpoint_s']['terminal_braking']==pytest.approx(2.)
    assert groups['terminal_braking']['tracking_position_RMS_mm'] is None
    assert groups['boundary_crossing']['tracking_position_RMS_mm'] is None
    assert groups['tracking_straight']['tracking_position_RMS_mm']==pytest.approx(10.)


@pytest.mark.parametrize('fault',['missing_trace','empty_actions','time_duplicate','time_backwards',
    'nonfinite_time','nonfinite_speed','nonfinite_yaw','nonfinite_curvature','bad_index','fractional_index','unknown_phase'])
def test_phase_response_rejects_invalid_clock_channels_or_reference_classification(ledger_module,fault):
    rows,actions=_phase_response_rows()
    if fault=='missing_trace':rows=rows[:1]
    elif fault=='empty_actions':actions=[]
    elif fault=='time_duplicate':rows[1]['time_s']=rows[0]['time_s']
    elif fault=='time_backwards':rows[2]['time_s']=-1.
    elif fault=='nonfinite_time':rows[1]['time_s']=float('nan')
    elif fault=='nonfinite_speed':rows[1]['vy_physx_m_s']=float('inf')
    elif fault=='nonfinite_yaw':rows[1]['yaw_rad']=float('nan')
    elif fault=='nonfinite_curvature':actions[0]['curvature_1_m']=float('nan')
    elif fault=='bad_index':rows[1]['action_index']=-1
    elif fault=='fractional_index':rows[1]['action_index']=.5
    else:rows[1]['phase']='tracking_curve'
    with pytest.raises(ValueError):ledger_module.measured_phase_response(rows,actions)


def test_actual_P_B2_phase_diagnosis_retains_own_traces_and_never_creates_a_physical_trial(ledger_module,tmp_path):
    base=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments')
    proposed=base/'cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json'
    baseline=base/'cohort_06_full_task_r01/route/hospital_h4_cohort_06_full_task_r01_B2_multimode_transition_cost_ramp_r01.json'
    if not all(p.exists() for p in (proposed,baseline)):pytest.skip('Both explicit actual development outcomes required')
    pins={p:ledger_module.file_sha(p) for result in (proposed,baseline) for p in (result,result.with_suffix('.csv'))}
    target=ledger_module.write_phase_response_review([proposed,baseline],tmp_path/'derived')
    report=json.loads(target.read_bytes());assert report['configuration_matching_checks_passed']
    assert [r['raw_verdict'] for r in report['runs']]==['PASS','PASS']
    assert report['P_minus_B2_phase_duration_s']['terminal_correction']==pytest.approx(11.2,abs=1e-5)
    groups=[{s['segment']:s for s in run['response']['segments']} for run in report['runs']]
    assert groups[0]['tracking_curve']['tracking_position_RMS_mm']<groups[1]['tracking_curve']['tracking_position_RMS_mm']
    assert groups[0]['tracking_straight']['tracking_position_RMS_mm']>groups[1]['tracking_straight']['tracking_position_RMS_mm']
    assert report['new_PhysX_trial_count']==0 and not report['use_for_final_statistics']
    assert all(r['response']['all_original_adjacent_intervals_accounted_for'] for r in report['runs'])
    outputs={p:ledger_module.file_sha(p) for p in target.parent.iterdir()}
    assert ledger_module.write_phase_response_review([proposed,baseline],target.parent)==target
    assert outputs=={p:ledger_module.file_sha(p) for p in target.parent.iterdir()}
    assert pins=={p:ledger_module.file_sha(p) for p in pins}
    table=target.parent/'measured_phase_response.csv';table.write_bytes(b'SOFTWARE_CORRUPTED')
    with pytest.raises(ValueError,match='no overwrite'):
        ledger_module.write_phase_response_review([proposed,baseline],target.parent)
    assert table.read_bytes()==b'SOFTWARE_CORRUPTED'


def _software_orientation_rows(plan,offset_deg=.75):
    import math
    from hospital_h4_path_execution import PathReference,wrap
    reference=PathReference(plan);rows=[]
    for time,progress in ((.1,.01),(.2,.02),(.3,.03)):
        index,pose=reference.at(progress);yaw=wrap(pose[2]-math.radians(offset_deg))
        rows.append(dict(time_s=time,phase='path_tracking',action_index=index,path_progress_m=progress,
            yaw_rad=yaw,goal_heading_error_deg=math.degrees(wrap(plan['goal_world_xyyaw'][2]-yaw)),
            path_heading_error_deg=offset_deg,front_rad=.12,target_front_rad=.1,rear_rad=-.01,target_rear_rad=0.))
    return rows


@pytest.mark.parametrize('offset',[.75,-.75,179.9,-179.9])
def test_orientation_segments_keep_signed_wrapped_goal_identity_without_fitting(
        ledger_module,original_H4_native_plans,offset):
    import math
    plan=original_H4_native_plans[-1];rows=_software_orientation_rows(plan,offset)
    before=copy.deepcopy((rows,plan));report=ledger_module.measured_orientation_segments(rows,plan)
    assert (rows,plan)==before and report['tracking_sample_count']==3 and report['all_tracking_samples_accounted_for']
    assert len(report['segments'])==1
    segment=report['segments'][0]
    assert segment['first']['reference_minus_actual_deg']==pytest.approx(offset)
    assert segment['maximum_absolute_wrapped_decomposition_closure_deg']<1e-10
    assert segment['maximum_absolute_logged_vs_progress_sampled_path_heading_deg']<1e-10
    assert segment['steering_target_tracking_observations']['front_actual_minus_target_deg']['time_weighted_RMS']==pytest.approx(math.degrees(.02))
    assert segment['steering_target_tracking_observations']['rear_actual_minus_target_deg']['time_weighted_mean']==pytest.approx(math.degrees(-.01))
    assert report['target_tracking_observations_NOT_identical_input_force_load_or_caster_causality']


@pytest.mark.parametrize('fault',['one_row','duplicate_time','nonfinite_time','unknown_phase','bad_index',
    'fractional_index','progress_outside','progress_backwards','missing_target','nonfinite_target','wrong_goal_frame'])
def test_orientation_segments_reject_invalid_channels_frame_or_reference(
        ledger_module,original_H4_native_plans,fault):
    plan=original_H4_native_plans[-1];rows=_software_orientation_rows(plan)
    if fault=='one_row':rows=rows[:1]
    elif fault=='duplicate_time':rows[1]['time_s']=rows[0]['time_s']
    elif fault=='nonfinite_time':rows[1]['time_s']=float('nan')
    elif fault=='unknown_phase':rows[1]['phase']='nominal_model'
    elif fault=='bad_index':rows[1]['action_index']=-1
    elif fault=='fractional_index':rows[1]['action_index']=.5
    elif fault=='progress_outside':rows[1]['path_progress_m']=100.
    elif fault=='progress_backwards':rows[1]['path_progress_m']=0.
    elif fault=='missing_target':del rows[1]['target_front_rad']
    elif fault=='nonfinite_target':rows[1]['target_front_rad']=float('inf')
    else:rows[1]['goal_heading_error_deg']+=1.
    with pytest.raises((ValueError,KeyError)):ledger_module.measured_orientation_segments(rows,plan)


def test_orientation_segments_do_not_merge_separate_tracking_episodes_or_hide_unlogged_error(
        ledger_module,original_H4_native_plans):
    plan=original_H4_native_plans[-1];rows=_software_orientation_rows(plan)
    rows[1]['phase']='terminal_braking';rows[2]['path_heading_error_deg']+=.01
    report=ledger_module.measured_orientation_segments(rows,plan)
    assert report['tracking_sample_count']==2 and report['other_original_phase_sample_count']==1
    assert len(report['segments'])==2
    assert all(r['steering_target_tracking_observations']['front_actual_minus_target_deg']['time_weighted_RMS'] is None for r in report['segments'])
    assert report['segments'][-1]['maximum_absolute_logged_vs_progress_sampled_path_heading_deg']==pytest.approx(.01)


def test_actual_old_and_new_P_orientation_review_retains_versions_and_exact_raw_outcomes(ledger_module,tmp_path):
    base=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments')
    old=base/'cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json'
    new=base/'cohort_09_full_task_r01/route/hospital_h4_cohort_09_full_task_r01_P_dynamics_full_envelope_r01.json'
    if not all(p.is_file() for p in (old,new)):pytest.skip('Both explicit actual development P outcomes required')
    protected={p:ledger_module.file_sha(p) for result in (old,new) for p in (result,result.with_suffix('.csv'))}
    target=ledger_module.write_orientation_response_review([old,new],tmp_path/'orientation')
    report=json.loads(target.read_bytes());assert not report['configuration_matching_checks_passed']
    assert [r['raw_verdict'] for r in report['runs']]==['PASS','PASS']
    assert all(len(run['orientation']['segments'])==3 for run in report['runs'])
    end=report['runs'][1]['orientation']['segments'][-1]['last']
    assert end['goal_minus_reference_deg']==pytest.approx(.9301739284804641,abs=1e-7)
    assert end['reference_minus_actual_deg']==pytest.approx(1.0465811191828932,abs=1e-7)
    assert end['recomputed_goal_minus_actual_deg']==pytest.approx(1.9767550476633573,abs=1e-7)
    assert report['new_PhysX_trial_count']==0 and not report['use_for_final_statistics']
    assert report['geometric_heading_decomposition_is_NOT_a_physical_cause_or_predictive_fit']
    assert protected=={p:ledger_module.file_sha(p) for p in protected}
    assert ledger_module.write_orientation_response_review([old,new],target.parent)==target
    table=target.parent/'measured_orientation_segments.csv';table.write_bytes(b'SOFTWARE_CORRUPTED')
    with pytest.raises(ValueError,match='no overwrite'):ledger_module.write_orientation_response_review([old,new],target.parent)


def _explicit_hospital_development_results():
    gate=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
    paths=[]
    for scenario,folder in [('h1','path_execution_v3'),('h2','hospital_h2_path_execution_v3')]:
        paths.extend(gate/folder/'cohort_01'/f'hospital_{scenario}_stopalign_v3_{name}_path_r01.json'
            for name in ('b0_fixed','b1_kinematic','b2_transition','proposed'))
    paths.extend([gate/'hospital_h4_experiments/cohort_05_full_task_r01/route/first_motion_after_prepath_hold/hospital_h4_cohort_05_full_task_r01_P_dynamics_full_envelope_after_prepath_hold_r01.json',
        gate/'hospital_h4_experiments/cohort_06_full_task_r01/route/hospital_h4_cohort_06_full_task_r01_B2_multimode_transition_cost_ramp_r01.json'])
    if not all(p.exists() for p in paths):pytest.skip('Explicit H1/H2/H4 development results unavailable')
    return paths


def test_manuscript_development_table_retains_FAILs_and_legacy_provenance_without_final_claims(ledger_module,tmp_path):
    paths=_explicit_hospital_development_results()
    pins={p:ledger_module.file_sha(p) for result in paths for p in (result,result.with_suffix('.csv'))}
    target=ledger_module.write_development_manuscript_table(paths,tmp_path)
    report=json.loads(target.read_bytes());rows=report['rows']
    assert len(rows)==10 and {r['task'] for r in rows}=={'H1','H2','H4'}
    assert len([r for r in rows if r['raw_verdict']=='FAIL'])==2
    for row in rows:
        if row['raw_verdict']=='FAIL':
            assert row['successful_route_clock_end_s'] is None
            assert row['completed_tracking_position_RMS_mm'] is None
            assert row['observed_until_last_CSV_sample_s']>0.
            assert row['partial_trace_tracking_position_RMS_mm'] is not None
        else:
            assert row['successful_route_clock_end_s']>0. and row['complete_fixed_hold']
        if row['task'] in ('H1','H2'):
            assert row['original_trace_provenance']=='LEGACY_UNPINNED_TRACE_CURRENT_SNAPSHOT_ONLY'
        else:assert row['original_trace_provenance']=='ORIGINAL_TRACE_HASH_VERIFIED'
    assert not report['statistical_superiority_established'] and not report['independent_paired_repeats_verified']
    assert report['qualifying_final_comparison_cells_proven']==0 and not report['use_for_final_statistics']
    tex=target.with_suffix('.tex').read_text()
    assert 'not independent paired repetitions' in tex and 'Failed partial traces are not ranked' in tex
    assert '\\begin{tabular}' in tex and '\\end{tabular}' in tex
    outputs={p:ledger_module.file_sha(p) for p in tmp_path.iterdir()}
    assert ledger_module.write_development_manuscript_table(paths,tmp_path)==target
    assert outputs=={p:ledger_module.file_sha(p) for p in tmp_path.iterdir()}
    assert pins=={p:ledger_module.file_sha(p) for p in pins}


@pytest.mark.parametrize('fault',['duplicate_path','same_task_method','unstarted','method_relabel'])
def test_manuscript_table_rejects_duplicate_unstarted_or_mislabelled_records(ledger_module,tmp_path,fault):
    paths=_explicit_hospital_development_results();original=paths[0]
    if fault=='duplicate_path':inputs=[original,original]
    else:
        raw=json.loads(original.read_bytes());changed=tmp_path/'SOFTWARE_copy.json'
        if fault=='same_task_method':raw['run_id']='SOFTWARE_NOT_AN_INDEPENDENT_REPEAT'
        elif fault=='unstarted':raw.update(verdict='HOLD',path_execution_started=False,sample_count=0)
        else:raw['method']='P_dynamics_full_envelope'
        changed.write_text(json.dumps(raw));changed.with_suffix('.csv').write_bytes(original.with_suffix('.csv').read_bytes())
        inputs=[original,changed] if fault=='same_task_method' else [changed]
    output=tmp_path/'not_created'
    with pytest.raises(ValueError):ledger_module.write_development_manuscript_table(inputs,output)
    assert not output.exists()


def test_terminal_reference_geometry_keeps_endpoint_and_ideal_handoff_distinct(ledger_module):
    plan_path=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_execution_bundle/bundle_r01/P_dynamics_full_envelope.json')
    if not plan_path.exists():pytest.skip('Original frozen H4 reference unavailable')
    plan=json.loads(plan_path.read_bytes());before=copy.deepcopy(plan)
    result=ledger_module.terminal_reference_geometry(plan,.025,.030,2.)
    assert result['planned_endpoint']['goal_position_error_mm']==pytest.approx(22.1623377692)
    assert result['ideal_on_reference_handoff_pose']['goal_position_error_mm']==pytest.approx(38.3621,abs=1e-4)
    assert result['planned_endpoint']['inside_stationary_pose_band_only']
    assert not result['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only']
    assert result['outside_pose_band_is_NOT_an_original_motion_FAIL']
    assert result['inside_pose_band_is_NOT_a_measured_stop_or_fixed_hold_PASS']
    assert result['velocity_braking_steering_tracking_errors_and_contact_NOT_simulated']
    assert plan==before


@pytest.mark.parametrize('distance,position,heading',[(float('nan'),.03,2.),(-.1,.03,2.),(1e8,.03,2.),(.025,0.,2.),(.025,.03,float('inf'))])
def test_terminal_reference_geometry_rejects_unknown_or_invalid_bands(ledger_module,distance,position,heading):
    plan_path=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_execution_bundle/bundle_r01/P_dynamics_full_envelope.json')
    if not plan_path.exists():pytest.skip('Original frozen H4 reference unavailable')
    with pytest.raises(ValueError):ledger_module.terminal_reference_geometry(json.loads(plan_path.read_bytes()),distance,position,heading)


def test_terminal_reference_review_keeps_all_four_methods_without_inventing_B1_motion(ledger_module,tmp_path):
    paths=_explicit_hospital_development_results()
    b0=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_07_full_task_r01/route/hospital_h4_cohort_07_full_task_r01_B0_fixed_counterphase_ramp_r01.json')
    if not b0.exists():pytest.skip('Actual B0 not yet recorded')
    inputs=[b0,*paths[-2:]];pins={p:ledger_module.file_sha(p) for r in inputs for p in (r,r.with_suffix('.csv'))}
    target=ledger_module.write_terminal_reference_review(inputs,tmp_path/'derived')
    report=json.loads(target.read_bytes());methods={r['method']:r for r in report['methods']}
    assert set(methods)==set(ledger_module.METHODS)
    assert report['original_remaining_distance_literal_m']==.025
    assert report['original_stationary_position_band_m']==.03
    assert report['original_stationary_heading_band_deg']==2.
    assert not methods['B1_multimode_kinematic_ramp']['actual_result_supplied']
    assert methods['B1_multimode_kinematic_ramp']['actual_handoff'] is None
    assert methods['P_dynamics_full_envelope']['original_actual_verdict']=='PASS'
    assert not methods['P_dynamics_full_envelope']['reference_geometry']['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only']
    assert methods['B0_fixed_counterphase_ramp']['reference_geometry']['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only']
    assert report['future_planner_cost_or_goal_connection_amendment_NOT_adopted']
    assert report['new_PhysX_trial_count']==0 and not report['use_for_final_statistics']
    assert ledger_module.write_terminal_reference_review(inputs,target.parent)==target
    assert pins=={p:ledger_module.file_sha(p) for p in pins}


def _contrast_fixture(module):
    return [dict(method=method,run_id=f'SOFTWARE_{method}',raw_verdict='PASS',complete_fixed_hold=True,
        total_logged_route_clock_s=10.,tracking_only_time_weighted_position_RMS_mm=20.,
        tracking_only_time_weighted_heading_RMS_deg=1.,terminal_position_error_mm=10.,
        terminal_heading_error_deg=-.5,terminal_pulse_count=0) for method in module.METHODS]


def test_descriptive_contrast_keeps_all_baselines_and_error_direction_without_superiority_claim(ledger_module):
    rows=_contrast_fixture(ledger_module)
    rows[-1].update(total_logged_route_clock_s=12.,tracking_only_time_weighted_position_RMS_mm=10.,
        terminal_heading_error_deg=-1.)
    before=copy.deepcopy(rows)
    contrasts=ledger_module.individual_development_contrasts(rows,True)
    assert len(contrasts)==3
    assert [row['baseline_method'] for row in contrasts]==list(ledger_module.METHODS[:3])
    for item in contrasts:
        assert item['P_minus_baseline']['total_logged_route_clock_s']==2.
        assert item['P_minus_baseline']['terminal_absolute_heading_error_deg']==.5
        assert item['P_minus_baseline_percent']['tracking_only_time_weighted_position_RMS_mm']==-50.
        assert item['P_minus_baseline_percent']['terminal_pulse_count'] is None
        assert not item['lower_error_or_duration_for_P']['terminal_absolute_heading_error_deg']
        assert not item['P_strictly_lower_on_all_listed_metrics']
        assert not item['independent_paired_repetition_or_statistical_superiority_claimed']
    assert rows==before


@pytest.mark.parametrize('fault',['missing','duplicate','unmatched','FAIL','partial_hold'])
def test_descriptive_contrast_never_selects_repeats_or_ranks_incomplete_failure(ledger_module,fault):
    rows=_contrast_fixture(ledger_module)
    if fault=='missing':del rows[0]
    elif fault=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif fault=='FAIL':rows[0].update(raw_verdict='FAIL',total_logged_route_clock_s=.1)
    elif fault=='partial_hold':rows[0]['complete_fixed_hold']=False
    result=ledger_module.individual_development_contrasts(rows,fault!='unmatched')[0]
    assert result['P_minus_baseline'] is None
    assert result['P_minus_baseline_percent'] is None
    assert result['P_strictly_lower_on_all_listed_metrics'] is None
    assert result['status']!='DESCRIPTIVE_COMPLETED_INDIVIDUAL_PAIR_NOT_FINAL_STATISTICS'


@pytest.mark.parametrize('invalid',[None,float('nan'),float('inf'),-.1,True])
def test_descriptive_completed_contrast_rejects_missing_or_invalid_metrics(ledger_module,invalid):
    rows=_contrast_fixture(ledger_module);rows[-1]['total_logged_route_clock_s']=invalid
    with pytest.raises(ValueError,match='Finite nonnegative'):
        ledger_module.individual_development_contrasts(rows,True)


@pytest.mark.parametrize('invalid',[None,float('nan'),float('inf'),True,'-0.5'])
def test_descriptive_contrast_rejects_invalid_signed_heading_before_magnitude_conversion(ledger_module,invalid):
    rows=_contrast_fixture(ledger_module);rows[-1]['terminal_heading_error_deg']=invalid
    with pytest.raises(ValueError,match='heading magnitude'):
        ledger_module.individual_development_contrasts(rows,True)


def test_matched_development_figure_preserves_exact_traces_and_does_not_infer_final_statistics(ledger_module,tmp_path,monkeypatch):
    inputs=_explicit_hospital_development_results()[-2:]
    pins={p:ledger_module.file_sha(p) for r in inputs for p in (r,r.with_suffix('.csv'))}
    exporter=ledger_module.write_explicit_development_comparison
    monkeypatch.setattr(ledger_module,'write_explicit_development_comparison',
        lambda results:exporter(results,tmp_path/'comparison'))
    target=ledger_module.write_development_comparison_figure(inputs,tmp_path/'figure')
    report=json.loads(target.read_bytes())
    assert report['methods']==['B2_multimode_transition_cost_ramp','P_dynamics_full_envelope']
    assert report['raw_verdicts_retained']==['PASS','PASS']
    assert not report['use_for_final_statistics'] and report['new_PhysX_trial_count']==0
    for path,sha in report['output_sha256'].items():
        assert ledger_module.file_sha(Path(path))==sha
        if path.endswith('.png'):assert Path(path).read_bytes().startswith(b'\x89PNG')
        else:assert Path(path).read_bytes().startswith(b'%PDF')
    assert ledger_module.write_development_comparison_figure(inputs,target.parent)==target
    assert pins=={p:ledger_module.file_sha(p) for p in pins}
    image=target.parent/'matched_development.png';image.write_bytes(b'CORRUPTED_SOFTWARE_TEST')
    with pytest.raises(ValueError,match='no overwrite'):
        ledger_module.write_development_comparison_figure(inputs,target.parent)


@pytest.mark.parametrize('fault',['HOLD','partial_hold','duplicate_method','unmatched'])
def test_matched_figure_rejects_incomplete_or_unmatched_comparisons(ledger_module,tmp_path,monkeypatch,fault):
    rows=_contrast_fixture(ledger_module)[:2]
    if fault=='HOLD':rows[0]['raw_verdict']='HOLD'
    elif fault=='partial_hold':rows[0]['complete_fixed_hold']=False
    elif fault=='duplicate_method':rows[1]['method']=rows[0]['method']
    report=tmp_path/'SOFTWARE_comparison.json'
    report.write_text(json.dumps(dict(actual_development_rows=rows,configuration_matching_checks_passed=fault!='unmatched')))
    monkeypatch.setattr(ledger_module,'write_explicit_development_comparison',lambda results:report)
    with pytest.raises(ValueError,match='complete PASS holds'):
        ledger_module.write_development_comparison_figure([],tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


def test_prospective_terminal_connection_filter_uses_original_cost_after_common_geometry_check(ledger_module):
    def row(name,cost,inside,original=True):
        return dict(candidate_id=name,original_native_admitted=original,original_native_cost_s=cost,
            reference_geometry=dict(planned_endpoint=dict(goal_position_error_mm=6.),
                ideal_on_reference_handoff_pose=dict(inside_stationary_pose_band_only=inside)))
    rows=[row('old_cheapest_outside_pose_band',1.,False),row('new_original_cost_winner',2.,True),
        row('more_expensive',3.,True),row('native_rejected_not_resurrected',0.,True,False)]
    before=copy.deepcopy(rows)
    assert ledger_module.choose_terminal_connection_candidate(rows)['candidate_id']=='new_original_cost_winner'
    assert rows==before
    assert ledger_module.choose_terminal_connection_candidate([rows[0],rows[-1]]) is None


@pytest.mark.parametrize('fault',['empty','duplicate','nonfinite_cost'])
def test_terminal_connection_filter_cannot_invent_or_duplicate_candidates_or_refit_costs(ledger_module,fault):
    row=dict(candidate_id='SOFTWARE_ONLY',original_native_admitted=True,original_native_cost_s=1.,
        reference_geometry=dict(planned_endpoint=dict(goal_position_error_mm=1.),
            ideal_on_reference_handoff_pose=dict(inside_stationary_pose_band_only=True)))
    rows=[row]
    if fault=='empty':rows=[]
    elif fault=='duplicate':rows.append(copy.deepcopy(row))
    else:row['original_native_cost_s']=float('nan')
    with pytest.raises(ValueError):ledger_module.choose_terminal_connection_candidate(rows)


def test_all_four_prospective_terminal_candidates_replay_original_models_without_adopting_new_motion(ledger_module,tmp_path):
    base=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_execution_bundle/bundle_r01')
    if not base.exists():pytest.skip('Original frozen H4 pool unavailable')
    protected={p:ledger_module.file_sha(p) for p in base.glob('*.json')}
    target=ledger_module.write_terminal_connection_candidate(tmp_path/'candidate')
    report=json.loads(target.read_bytes());methods={row['method']:row for row in report['methods']}
    assert set(methods)==set(ledger_module.METHODS)
    assert report['original_native_selected_decisions_and_costs_exactly_reproduced']
    for method,row in methods.items():
        assert len(row['complete_original_candidate_decisions'])==26
        selected=row['candidate_selected_by_original_native_cost']
        assert selected['original_native_admitted']
        assert selected['reference_geometry']['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only']
        assert row['selected_candidate_prediction']['selected_states']
        if method!='P_dynamics_full_envelope':assert not row['changed_from_original_selection']
    p=methods['P_dynamics_full_envelope']
    assert p['changed_from_original_selection']
    assert p['candidate_selected_by_original_native_cost']['reference_geometry']['ideal_on_reference_handoff_pose']['goal_position_error_mm']<30.
    assert p['original_selected_reference_geometry']['ideal_on_reference_handoff_pose']['goal_position_error_mm']>30.
    assert report['new_PhysX_trial_count']==0
    assert not report['adopted_by_any_live_runner'] and not report['actual_motion_authorized']
    assert report['pose_only_filter_NOT_tracking_braking_steering_uncertainty_or_fixed_hold_validation']
    assert ledger_module.write_terminal_connection_candidate(target.parent)==target
    assert protected=={p:ledger_module.file_sha(p) for p in protected}


@pytest.fixture
def original_common_interface_archive():
    path=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_terminal_connection_candidates/3828a169cafbb9c3/terminal_connection_candidate.json')
    if not path.is_file():pytest.skip('Original prospective candidate archive unavailable')
    return path,json.loads(path.read_bytes())


def test_same_margin_grid_keeps_all_four_no_candidate_cells_and_original_costs(
        ledger_module,original_common_interface_archive):
    _,candidate=original_common_interface_archive;before=copy.deepcopy(candidate)
    rows=ledger_module.common_terminal_margin_sensitivity(candidate)
    assert candidate==before and len(rows)==48
    assert {row['method'] for row in rows}==set(ledger_module.METHODS)
    assert len({(r['method'],r['pose_position_limit_mm'],r['pose_heading_limit_deg']) for r in rows})==48
    for row in rows:
        method=next(m for m in candidate['methods'] if m['method']==row['method'])
        if row['selected_candidate_id'] is not None:
            original=next(r for r in method['complete_original_candidate_decisions'] if r['candidate_id']==row['selected_candidate_id'])
            assert row['original_native_cost_s']==original['original_native_cost_s']
        if (row['pose_position_limit_mm'],row['pose_heading_limit_deg'])==(30.,2.):
            assert row['selected_candidate_id']==method['candidate_selected_by_original_native_cost']['candidate_id']
    narrow=[row for row in rows if row['pose_position_limit_mm']==30. and row['pose_heading_limit_deg']==.5]
    b0=next(row for row in narrow if row['method'].startswith('B0'))
    proposed=next(row for row in narrow if row['method'].startswith('P_'))
    assert b0['eligible_candidate_count']==0 and b0['selected_candidate_id'] is None
    assert proposed['selected_candidate_id']=='rear_only_k+0.50_S50_A65_S38'


@pytest.mark.parametrize('fault',['band','method_subset','duplicate_method','pool_subset','duplicate_candidate',
    'different_pool','admission_not_bool','cost_nonfinite','geometry_nonfinite','reference_distance','rejected_imputation','different_choice'])
def test_margin_sensitivity_rejects_missing_invented_or_retuned_candidate_inputs(
        ledger_module,original_common_interface_archive,fault):
    _,candidate=original_common_interface_archive;candidate=copy.deepcopy(candidate)
    method=candidate['methods'][0];rows=method['complete_original_candidate_decisions']
    good=next(r for r in rows if r['original_native_admitted'])
    rejected=next(r for r in rows if not r['original_native_admitted'])
    if fault=='band':candidate['common_heading_band_deg']=3.
    elif fault=='method_subset':candidate['methods'].pop()
    elif fault=='duplicate_method':candidate['methods'][-1]=copy.deepcopy(method)
    elif fault=='pool_subset':rows.pop()
    elif fault=='duplicate_candidate':rows[-1]['candidate_id']=rows[0]['candidate_id']
    elif fault=='different_pool':rejected['candidate_id']='SOFTWARE_NOT_ORIGINAL'
    elif fault=='admission_not_bool':good['original_native_admitted']=1
    elif fault=='cost_nonfinite':good['original_native_cost_s']=float('nan')
    elif fault=='geometry_nonfinite':good['reference_geometry']['ideal_on_reference_handoff_pose']['goal_position_error_mm']=float('inf')
    elif fault=='reference_distance':good['reference_geometry']['remaining_reference_distance_m']=.01
    elif fault=='rejected_imputation':rejected['original_native_cost_s']=0.
    else:method['candidate_selected_by_original_native_cost']={'candidate_id':'SOFTWARE_DIFFERENT'}
    with pytest.raises(ValueError):ledger_module.common_terminal_margin_sensitivity(candidate)


def test_all_four_margin_export_preserves_raw_inputs_and_does_not_select_or_adopt_a_policy(
        ledger_module,original_common_interface_archive,tmp_path):
    path,_=original_common_interface_archive;before=ledger_module.file_sha(path)
    target=ledger_module.write_common_terminal_margin_sensitivity(path,tmp_path/'derived_sensitivity')
    saved=json.loads(target.read_bytes())
    assert saved['grid_cell_count']==48 and saved['all_no_candidate_cells_retained']
    assert saved['new_PhysX_trial_count']==0 and not saved['actual_motion_authorized']
    assert saved['no_new_policy_selected_or_executor_plan_acceptance_physics_changed']
    assert ledger_module.file_sha(path)==before
    assert ledger_module.write_common_terminal_margin_sensitivity(path,target.parent)==target
    table=target.parent/'all_four_margin_grid.csv';table.write_bytes(b'SOFTWARE_CORRUPTED')
    with pytest.raises(ValueError,match='no overwrite'):ledger_module.write_common_terminal_margin_sensitivity(path,target.parent)


def test_actual_B1_FAIL_stays_in_matched_figure_without_completed_bar_or_goal_invention(ledger_module,tmp_path,monkeypatch):
    b1=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_experiments/cohort_08_full_task_r01/route/hospital_h4_cohort_08_full_task_r01_B1_multimode_kinematic_ramp_r01.json')
    if not b1.exists():pytest.skip('Actual B1 outcome not recorded')
    raw=json.loads(b1.read_bytes())
    if raw['verdict']!='FAIL':pytest.skip('Fixture requires actual unmodified failed outcome')
    inputs=[b1,*_explicit_hospital_development_results()[-2:]]
    exporter=ledger_module.write_explicit_development_comparison
    monkeypatch.setattr(ledger_module,'write_explicit_development_comparison',lambda paths:exporter(paths,tmp_path/'comparison'))
    target=ledger_module.write_development_comparison_figure(inputs,tmp_path/'figure')
    report=json.loads(target.read_bytes())
    assert report['raw_verdicts_retained']==['FAIL','PASS','PASS']
    assert report['all_supplied_original_FAILs_retained_in_trajectory_and_manifest']
    assert report['partial_FAILs_NOT_ranked_as_completed_performance']
    assert json.loads(b1.read_bytes())==raw


def test_connection_nominal_review_reuses_identical_cases_and_only_executes_changed_plan(ledger_module,tmp_path,monkeypatch):
    import screen_hospital_h4_common_executor as screen
    if not screen.TARGET.exists():pytest.skip('Original nominal diagnostic unavailable')
    candidate=ledger_module.write_terminal_connection_candidate(tmp_path/'candidate')
    original=json.loads(screen.TARGET.read_bytes());calls=[]
    def software_fixture_case(plan,bounds,dt,occupied,origin,resolution,design):
        calls.append((plan['method'],dt))
        row=next(row for row in original['cases'] if row['method']==plan['method'] and row['integration_dt_s']==dt)
        # Exercise orchestration/representation only, not a new MODEL result.
        report={key:value for key,value in row.items() if key not in ('model_trace','model_trace_sha256')}
        return report,[dict(time_s=0.,software_fixture_not_physical=True),dict(time_s=1.,software_fixture_not_physical=True)]
    monkeypatch.setattr(screen,'case',software_fixture_case)
    output=ledger_module.write_terminal_connection_model_review(candidate,tmp_path/'software_only_nominal')
    report=json.loads(output.read_bytes())
    assert calls==[('P_dynamics_full_envelope',dt) for dt in screen.TIMESTEPS_S]
    assert report['newly_executed_nominal_case_count']==2
    assert report['exact_unchanged_original_case_reuse_count']==6
    assert report['original_B1_nominal_PASS_is_NOT_its_actual_success']
    assert report['new_PhysX_trial_count']==0 and not report['actual_motion_authorized']
    assert ledger_module.write_terminal_connection_model_review(candidate,output.parent)==output
    assert len(calls)==2
    trace=Path(next(iter(report['output_sha256'])))
    trace.write_bytes(b'CORRUPTED_SOFTWARE_ONLY')
    with pytest.raises(ValueError,match='no overwrite'):
        ledger_module.write_terminal_connection_model_review(candidate,output.parent)


def test_one_actual_outcome_can_review_own_all_four_references_without_inventing_other_runs(ledger_module,tmp_path):
    proposed=_explicit_hospital_development_results()[-2]
    target=ledger_module.write_terminal_reference_review([proposed],tmp_path/'one_actual')
    report=json.loads(target.read_bytes())
    assert len(report['methods'])==4
    assert sum(row['actual_result_supplied'] for row in report['methods'])==1
    assert all(row['actual_handoff'] is None for row in report['methods'] if not row['actual_result_supplied'])
    assert report['own_frozen_bundle']==str(Path(json.loads(proposed.read_bytes())['plan_path']).parent/'execution_bundle.json')


@pytest.mark.parametrize('fault',[None,'new_FAIL','unmatched_gain','cross_method','different_budget','reversed_versions'])
def test_planner_version_contrast_orchestration_preserves_failures_and_rejects_unreviewed_changes(ledger_module,tmp_path,monkeypatch,fault):
    original=_explicit_hospital_development_results()[-2]
    raw=json.loads(original.read_bytes())
    candidate_root=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h4_terminal_connection_candidates/3828a169cafbb9c3/execution_bundle/cbce49e1430f2237')
    plan=candidate_root/'P_dynamics_full_envelope.json'
    if not plan.exists():pytest.skip('Explicit prospective native plan unavailable')
    old=tmp_path/'SOFTWARE_OLD.json';new=tmp_path/'SOFTWARE_NEW.json'
    old_protocol=tmp_path/'SOFTWARE_OLD_protocol.json';new_protocol=tmp_path/'SOFTWARE_NEW_protocol.json'
    old_protocol.write_text('{}');new_protocol.write_text(json.dumps(dict(terminal_connection_bundle=str(candidate_root/'execution_bundle.json'))))
    first=copy.deepcopy(raw);last=copy.deepcopy(raw)
    first['path_protocol']=str(old_protocol);last.update(path_protocol=str(new_protocol),plan_path=str(plan))
    if fault=='cross_method':last['method']='B1_multimode_kinematic_ramp'
    if fault=='different_budget':last['guards']['maximum_simulation_duration_s']=180.
    if fault=='reversed_versions':first['path_protocol'],last['path_protocol']=last['path_protocol'],first['path_protocol']
    old.write_text(json.dumps(first));new.write_text(json.dumps(last))
    rows=[]
    for index in range(2):
        complete=not(fault=='new_FAIL' and index==1)
        rows.append(dict(run_id='SOFTWARE_NOT_PHYSICAL_'+str(index),method='P_dynamics_full_envelope',
            raw_verdict='PASS' if complete else 'FAIL',complete_fixed_hold=complete,
            total_logged_route_clock_s=100.+index,tracking_only_time_weighted_position_RMS_mm=10.-index,
            tracking_only_time_weighted_heading_RMS_deg=1.,terminal_position_error_mm=5. if complete else None,
            terminal_pulse_count=0))
    inputs={str(path):ledger_module.file_sha(path) for path in (old,new,old_protocol,new_protocol,plan)}
    comparison=tmp_path/'SOFTWARE_COMPARISON.json'
    comparison.write_text(json.dumps(dict(actual_development_rows=rows,input_sha256=inputs,
        physical_control_or_domain_mismatches={'SOFTWARE_NEW':['generator_selector_version','common_resource_contract']+
            (['source_sha256.hospital_path_execution.py'] if fault=='unmatched_gain' else [])})))
    monkeypatch.setattr(ledger_module,'write_explicit_development_comparison',lambda paths:comparison)
    if fault not in (None,'new_FAIL'):
        with pytest.raises(ValueError):ledger_module.write_planner_version_development_contrast(old,new,tmp_path/'not_created')
        assert not (tmp_path/'not_created').exists()
    else:
        target=ledger_module.write_planner_version_development_contrast(old,new,tmp_path/'software_only_output')
        report=json.loads(target.read_bytes())
        assert len(report['original_and_prospective'])==2
        assert not report['use_for_final_statistics'] and report['new_PhysX_trial_count']==0
        assert not report['independent_paired_repeats_or_statistical_superiority_established']
        if fault=='new_FAIL':
            assert not report['both_complete_PASS'] and report['new_minus_original_if_both_complete'] is None
        else:
            assert report['both_complete_PASS']
            assert report['new_minus_original_if_both_complete']['total_logged_route_clock_s']==1.
        assert ledger_module.write_planner_version_development_contrast(old,new,target.parent)==target


def test_existing_initial_motion_replay_preserves_all_135_runs_definitions_and_missing_metadata(ledger_module):
    path=_ijcas_path(WORKSPACE, 'studies/CICS26/paper_assets/patient_transport/final9/patient_transport_final9_metrics.csv')
    if not path.is_file():pytest.skip('Existing initial-motion evidence unavailable')
    before=ledger_module.file_sha(path);report=ledger_module.initial_motion_reuse_evidence()
    assert report['recorded_run_count']==report['reached_count']==135
    assert len(report['cells'])==45 and len(report['median_of_nine_condition_summaries'])==5
    assert report['legacy_saved_table_numerically_reproduced']
    assert report['original_response_detected_count']==27
    assert report['original_response_selected_rates_rad_s']==[.0080078125]
    assert len(report['input_sha256'])==185
    assert report['original_trace_provenance']=='CURRENT_BYTE_SNAPSHOTS_NO_ORIGINAL_FREEZE_HASH_INVENTED'
    missing=report['missing_legacy_metadata_retained_NOT_filled']
    assert len(missing)==3 and all(row['actual_value'] is None for row in missing)
    assert all('265center_lateral_lowbed_frontpos_B2_v5' in row['summary'] for row in missing)
    assert not report['CICS_runtime_integration'] and report['new_PhysX_trial_count']==0
    assert ledger_module.file_sha(path)==before


def _manuscript_sources():
    gate=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
    return (gate/'paper_evidence_inventory/manuscript_tables/hospital_development_6abbfc85692e6f0c.json',
        gate/'hospital_h4_experiments/cohort_09_full_task_r01/route/offline_planner_version_contrast/planner_version_contrast.json')


@pytest.mark.skipif(not (_ijcas_path(WORKSPACE, 'studies/IJCAS/paper')).is_dir(), reason='Publication assets intentionally excluded from the technical workspace')
def test_native_manuscript_keeps_prior_work_failed_outcomes_and_separate_P_versions(ledger_module,tmp_path):
    import re
    first,second=_manuscript_sources()
    if not all(path.is_file() for path in (first,second)):pytest.skip('Explicit development inputs unavailable')
    protected={path:ledger_module.file_sha(path) for path in (first,second)}
    target=ledger_module.write_manuscript_development_assembly(first,second,tmp_path/'native_manuscript')
    report=json.loads(target.read_bytes());text=(target.parent/'manuscript_development.tex').read_text()
    assert not report['submission_ready'] and not report['final_hospital_evaluation_complete']
    assert not report['latex_compilation_verified'] and not report['use_for_final_statistics']
    assert report['original_initial_motion_runs_replayed']==135
    assert report['original_hospital_FAILs_and_planner_versions_preserved']
    assert not report['real_robot_trials_required_for_this_scope'] and not report['CICS_runtime_integration']
    assert 'H4 & B1 & FAIL & -- & -- & -- & --' in text
    assert '$P_{\\mathrm{v1}}$ & PASS & 181.87' in text
    assert '$P_{\\mathrm{v2}}$ & PASS & 170.65' in text
    assert '$I_R$ & Response-step + hold' in text
    assert '27/27' in text and '135 saved traces' in text and 'four verified' in text
    assert 'zero\nindependent fresh frozen whole-sequence repeats' in text
    assert 'original whole-sequence\nmodel remains 2/8 PASS' in text
    assert '@@' not in text and 'dtype' not in text and 'KeyError' not in text
    # Structural static checks, explicitly NOT a claim of TeX compilation.
    stack=[]
    for kind,name in re.findall(r'\\(begin|end)\{([^{}]+)\}',text):
        if kind=='begin':stack.append(name)
        else:assert stack and stack.pop()==name
    assert not stack and len(re.findall(r'(?<!\\)\$',text))%2==0
    relative=re.search(r'\\includegraphics\[[^]]+\]\{\\detokenize\{([^}]+)\}\}',text).group(1)
    assert (target.parent/relative).resolve().is_file()
    assert protected=={path:ledger_module.file_sha(path) for path in protected}
    assert ledger_module.write_manuscript_development_assembly(first,second,target.parent)==target
    (target.parent/'manuscript_development.tex').write_text('SOFTWARE_CORRUPTED')
    with pytest.raises(ValueError,match='no overwrite'):
        ledger_module.write_manuscript_development_assembly(first,second,target.parent)


@pytest.mark.parametrize('fault',['missing_failure_row','renamed_original_P','relabeled_final_statistics','changed_input_sha'])
@pytest.mark.skipif(not (_ijcas_path(WORKSPACE, 'studies/IJCAS/paper')).is_dir(), reason='Publication assets intentionally excluded from the technical workspace')
def test_manuscript_does_not_silently_rewrite_or_promote_development_inputs(ledger_module,tmp_path,fault):
    first,second=_manuscript_sources()
    if not all(path.is_file() for path in (first,second)):pytest.skip('Explicit development inputs unavailable')
    table=json.loads(first.read_bytes())
    if fault=='missing_failure_row':table['rows']=[row for row in table['rows'] if row['raw_verdict']!='FAIL']
    elif fault=='renamed_original_P':
        next(row for row in table['rows'] if row['task']=='H4' and row['method'].startswith('P_'))['run_id']='SOFTWARE_RENAMED'
    elif fault=='relabeled_final_statistics':table['use_for_final_statistics']=True
    else:table['input_sha256'][str(first)]='0'*64
    source=tmp_path/'SOFTWARE_TABLE.json';source.write_text(json.dumps(table))
    with pytest.raises(ValueError):ledger_module.write_manuscript_development_assembly(source,second,tmp_path/'not_created')
    assert not (tmp_path/'not_created').exists()


@pytest.mark.skipif(not (_ijcas_path(WORKSPACE, 'studies/IJCAS/paper')).is_dir(), reason='Publication assets intentionally excluded from the technical workspace')
def test_korean_first_review_preserves_full_scope_CICS_distinction_and_editable_document(ledger_module,tmp_path):
    import zipfile
    import xml.etree.ElementTree as ET
    source=_ijcas_path(WORKSPACE, 'studies/IJCAS/paper/korean_manuscript.txt')
    before=source.read_bytes()
    target=ledger_module.write_korean_manuscript_review(tmp_path/'korean_review')
    report=json.loads(target.read_bytes());root=target.parent
    assert report['existing_patient_initial_motion_runs_replayed']==135
    assert report['CICS_reference_pages']==2
    assert report['full_H1_to_H6_and_hospital_sensor_Nav2_scope_retained']
    assert report['generic_English_article_draft_NOT_current_author_review_source']
    assert report['unknown_development_or_static_trials_NOT_hidden_in_a_fixed_remaining_drive_count']
    assert report['translate_to_IJCAS_only_after_author_Korean_logic_results_figures_approval']
    assert not report['submission_ready'] and not report['actual_motion_authorized']
    assert report['new_PhysX_trial_count']==0
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(_ijcas_path(root, 'korean_manuscript.docx')) as z:
        assert z.testzip() is None
        for name in z.namelist():ET.fromstring(z.read(name))
        document=ET.fromstring(z.read('word/document.xml'))
        words='\n'.join(n.text or '' for n in document.findall('.//w:t',ns))
        assert all(name in words for name in ('H1','H2','H3','H4','H5','H6','Nav2','센서','CICS'))
        assert '전체 48개' not in words and '총 48개 비교 셀' in words
        assert '53개 최종 평가 셀' in words
        assert '135개' in words and '총 240회' in words
        assert '원래 P는 약 181.87초' in words and '별도 보완 P는 약 170.65초' in words
        assert 'B1의 종단 정렬 시간초과 FAIL' in words
        assert len(document.findall('.//w:tbl',ns))==3
        styles=ET.fromstring(z.read('word/styles.xml'))
        assert any(n.get('{'+ns['w']+'}eastAsia')=='Noto Sans CJK KR' for n in styles.findall('.//w:rFonts',ns))
    for name,count in [('comparison_methods.csv',4),('hospital_scenarios.csv',6),('figure_plan.csv',8)]:
        with (_ijcas_path(root, name)).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==count
    assert source.read_bytes()==before
    assert ledger_module.write_korean_manuscript_review(root)==target
    (_ijcas_path(root, 'korean_manuscript.docx')).write_bytes(b'AUTHOR_EDIT_MUST_BE_PRESERVED')
    with pytest.raises(ValueError,match='no overwrite'):
        ledger_module.write_korean_manuscript_review(root)
    assert (_ijcas_path(root, 'korean_manuscript.docx')).read_bytes()==b'AUTHOR_EDIT_MUST_BE_PRESERVED'


@pytest.fixture
def platform_geometry_software_inputs():
    """Numeric source-geometry fixtures, not new native or physical evidence."""
    def source(current):
        active=.665 if current else .645;caster=.660 if current else .645
        side=.355 if current else .3365;trail=.045 if current else .095
        joints={}
        for axle,sign in [('front',1),('rear',-1)]:
            for role,z,axis in [('steering',.174,'Z'),('rolling',.074,'Y')]:
                joints[axle+'_'+role]=dict(body0_anchor_in_base_m=[sign*active,0.,z],axis=axis)
        for corner,sx,sy in [('fl',1,1),('fr',1,-1),('rl',-1,1),('rr',-1,-1)]:
            joints[corner+'_swivel']=dict(body0_anchor_in_base_m=[sx*caster,sy*side,.150],axis='Z')
            joints[corner+'_rolling']=dict(body0_anchor_in_base_m=[sx*caster+(trail if current else 0.),
                sy*side+(0. if current else sy*trail),.0475],axis='Y')
        return dict(rigid_body_count=13,revolute_joint_count=12,joints=joints)
    nominal=dict(body_length_m=1.63,body_width_m=.965,wheelbase_m=1.29,front_length_m=.645,rear_length_m=.645,
        mode_curvature_response_scale=dict(coordinated_counterphase=.9618004068327801,front_only=.96545,rear_only=.9643),
        mode_lateral_response_scale=dict(coordinated_counterphase=.92434,front_only=1.00542,rear_only=.99375))
    bounds=dict(front_support_x_m=.665,rear_support_x_m=-.665,track_width_m=.710)
    measurements={'physical_geometry_evidence':{'unresolved':['SOFTWARE_declared_platform_versions_NOT_new_measurement']}}
    return source(False),source(True),nominal,bounds,measurements


def test_platform_source_nominal_and_load_geometries_are_separate_not_refitted(ledger_module,platform_geometry_software_inputs):
    import math
    inputs=platform_geometry_software_inputs;before=copy.deepcopy(inputs)
    r=ledger_module.platform_geometry_comparison(*inputs)
    assert r['legacy_CAD_source']['active_pivot_span_m']==pytest.approx(1.290)
    assert r['current_hospital_USD_source']['active_pivot_span_m']==pytest.approx(1.330)
    assert r['nominal_wheelbase_minus_current_source_span_m']==pytest.approx(-.040)
    assert r['frozen_support_positions_match_current_source_to_float_precision']
    assert r['frozen_Newton_Euler_support_parameters']==inputs[3]
    assert r['preserved_nominal_steering']==inputs[2]
    assert r['legacy_CAD_source']['passive_caster_planar_pivot_to_wheel_distance_m']['fl']==pytest.approx(.095)
    assert r['current_hospital_USD_source']['passive_caster_planar_pivot_to_wheel_distance_m']['fl']==pytest.approx(.045)
    assert r['nominal_command_to_source_geometry_only_curvature_ratio']==pytest.approx(1.29/1.33)
    angles=r['illustrative_nominal_target_angles'];assert len(angles)==3
    for row in angles:
        assert row['geometry_only_current_source_curvature_1_m']==pytest.approx(.5*1.29/1.33)
        assert row['not_measured_response_not_initial_joint_assignment']
    assert angles[0]['target_front_rad']==-angles[0]['target_rear_rad']
    assert angles[1]['target_front_rad']==math.atan(1.29*.5) and angles[1]['target_rear_rad']==0.
    assert angles[2]['target_front_rad']==0. and angles[2]['target_rear_rad']==-math.atan(1.29*.5)
    assert r['geometry_ratio_is_NOT_a_response_fit_or_proven_cause_of_identified_coefficients']
    assert not r['body_response_validation_complete'] and not r['actual_motion_authorized']
    assert r['new_PhysX_trial_count']==0 and not r['use_for_final_statistics']
    assert inputs==before


@pytest.mark.parametrize('fault',['body_count','joint_count','missing_role','axis','nan','bool','nominal_sum','nominal_extra','zero_length','bounds_nan'])
def test_platform_geometry_review_rejects_missing_or_invalid_source(ledger_module,platform_geometry_software_inputs,fault):
    legacy,current,nominal,bounds,measurements=platform_geometry_software_inputs
    if fault=='body_count':current['rigid_body_count']=12
    elif fault=='joint_count':current['revolute_joint_count']=11
    elif fault=='missing_role':del current['joints']['fl_rolling']
    elif fault=='axis':current['joints']['front_steering']['axis']='X'
    elif fault=='nan':current['joints']['front_steering']['body0_anchor_in_base_m'][0]=float('nan')
    elif fault=='bool':current['joints']['fl_swivel']['body0_anchor_in_base_m'][1]=True
    elif fault=='nominal_sum':nominal['wheelbase_m']+=.01
    elif fault=='nominal_extra':nominal['silently_fitted_parameter']=1.
    elif fault=='zero_length':nominal['body_length_m']=0.
    else:bounds['track_width_m']=float('nan')
    with pytest.raises(ValueError):ledger_module.platform_geometry_comparison(legacy,current,nominal,bounds,measurements)


def test_source_geometry_disagreement_is_reported_not_waived_or_fixed(ledger_module,platform_geometry_software_inputs):
    legacy,current,nominal,bounds,measurements=platform_geometry_software_inputs
    bounds['front_support_x_m']+=.010;before=copy.deepcopy(bounds)
    r=ledger_module.platform_geometry_comparison(legacy,current,nominal,bounds,measurements)
    assert not r['frozen_support_positions_match_current_source_to_float_precision']
    assert r['frozen_support_vs_current_source_residuals_m']['front_support_minus_source_axis_m']==pytest.approx(.010)
    assert bounds==before and r['preserved_nominal_steering']['wheelbase_m']==1.29
    assert not r['body_response_validation_complete']


def test_platform_figure_export_preserves_author_files_and_never_overwrites(ledger_module,platform_geometry_software_inputs,tmp_path,monkeypatch):
    import inspect
    legacy,current,*_=platform_geometry_software_inputs
    def read_source(asset,robot,joint_roles):
        r=copy.deepcopy(current if 'ijcas' in str(asset) else legacy)
        r.update(source_asset=str(Path(asset).resolve()),source_actor=robot,
            input_sha256={str(Path(asset).resolve()):ledger_module.file_sha(asset)},SOFTWARE_FIXTURE_NOT_NATIVE_EVIDENCE=True)
        return r
    def draw(report,paths):
        for path in paths:path.write_bytes(b'SOFTWARE_FIGURE_NOT_RENDERED')
    monkeypatch.setattr(ledger_module,'read_platform_joint_geometry',read_source)
    monkeypatch.setattr(ledger_module,'draw_platform_geometry_figure',draw)
    doc=tmp_path/'author_document.docx';doc.write_bytes(b'AUTHOR_DOCUMENT_FIXTURE');before=ledger_module.file_sha(doc)
    target=ledger_module.write_platform_geometry_review(tmp_path/'figure')
    report=json.loads(target.read_bytes());preserved=target.read_bytes()
    assert report['analysis_source'][read_source.__name__]==inspect.getsource(read_source)
    assert len(report['output_sha256'])==5
    assert ledger_module.write_platform_geometry_review(target.parent)==target
    assert target.read_bytes()==preserved and ledger_module.file_sha(doc)==before
    with (target.parent/'source_geometry.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
    assert len(rows)==7 and all(len(row)==6 for row in rows)
    assert rows[0][3:5]==['명목 조향 기하','하중 모델']
    assert '캐스터를 구동한 결과가 아니다' in (target.parent/'figure_02_caption_ko.txt').read_text()
    edited=target.parent/'figure_02_platform_geometry.svg';edited.write_bytes(b'AUTHOR_FIGURE_EDIT')
    with pytest.raises(ValueError,match='no overwrite'):ledger_module.write_platform_geometry_review(target.parent)
    assert edited.read_bytes()==b'AUTHOR_FIGURE_EDIT' and ledger_module.file_sha(doc)==before


def test_platform_source_reader_and_diagram_cannot_edit_live_state_or_refit_models(ledger_module):
    import ast,inspect
    source='\n'.join(inspect.getsource(f) for f in (ledger_module.read_platform_joint_geometry,
        ledger_module.platform_geometry_comparison,ledger_module.draw_platform_geometry_figure,ledger_module.write_platform_geometry_review))
    calls={n.func.attr for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not calls & {'Set','SetActive','SetEditTarget','DefinePrim','RemovePrim','play','pause','stop','reset','initialize',
        'set_joint_positions','set_world_pose','set_gains','apply_action','Save','Reload','open_stage'}
    assert 'SimulationApp' not in source and 'omni.kit' not in source
    assert 'GetLocalToWorldTransform' in calls and 'GetBody0Rel' in calls and 'GetBody1Rel' in calls
    assert 'metadata' in source and 'source_authored_pose_NOT_a_live_measured_endpoint' in source


@pytest.fixture(scope='module')
def actual_H5_fixed_mode_witness_inputs():
    import numpy as np
    gate=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h5_geometry_candidates')
    source=json.loads((gate/'native_planning_probes/17ead2e292304982/B0_fixed_counterphase_ramp.json').read_bytes())
    folder=gate/'common_map/e16a2898d38c414b';manifest=json.loads((folder/'common_map_candidate.json').read_bytes())
    with np.load(folder/'common_occupancy.npz') as grid:
        return source,manifest,grid['occupied'].copy(),grid['origin_xy_m'].copy(),float(grid['resolution_m'])


def test_actual_B0_candidate_refutes_structural_geometric_infeasibility_for_B1_B2_only(ledger_module,actual_H5_fixed_mode_witness_inputs):
    inputs=actual_H5_fixed_mode_witness_inputs;before=copy.deepcopy(inputs[:2])
    report=ledger_module.fixed_mode_candidate_permission_witness(*inputs)
    assert [r['method'] for r in report['methods']]==['B1_multimode_kinematic_ramp','B2_multimode_transition_cost_ramp']
    for row in report['methods']:
        assert row['permitted_geometric_candidate_exists'] and all(row['checks'].values())
        assert row['action_count']==108 and row['substep_count']==1080 and row['colliding_substeps']==0
        assert row['witness_was_NOT_found_by_this_methods_recorded_search']
        assert not row['physical_execution_feasibility_claimed']
    assert report['methods'][1]['original_method_candidate_cost_s']>report['methods'][0]['original_method_candidate_cost_s']
    assert report['witness_NOT_final_B1_B2_comparison_plan']
    assert report['P_feasible_set_NOT_assumed_to_contain_ideal_baseline_sets']
    assert not report['actual_motion_authorized'] and not report['use_for_final_statistics'] and report['new_PhysX_trial_count']==0
    assert inputs[:2]==before


@pytest.mark.parametrize('fault',['wrong_source','incomplete','action_mode','direction','speed','started'])
def test_fixed_mode_permission_proof_cannot_use_invalid_or_substituted_source(ledger_module,actual_H5_fixed_mode_witness_inputs,fault):
    source,manifest,occupied,origin,resolution=copy.deepcopy(actual_H5_fixed_mode_witness_inputs)
    if fault=='wrong_source':source['method']='P_dynamics_full_envelope'
    elif fault=='incomplete':source['selected_states'].pop()
    elif fault=='action_mode':source['selected_actions'][0]['mode']='front_only'
    elif fault=='direction':source['selected_actions'][0]['direction']=-1
    elif fault=='speed':source['selected_actions'][0]['target_speed_m_s']=float('nan')
    else:source['initial_steering_propagated']=True
    with pytest.raises(ValueError):ledger_module.fixed_mode_candidate_permission_witness(source,manifest,occupied,origin,resolution)


@pytest.mark.parametrize('fault',['blocked_map','different_goal','changed_saved_state'])
def test_fixed_mode_witness_does_not_waive_collision_goal_or_repropagation(ledger_module,actual_H5_fixed_mode_witness_inputs,fault):
    source,manifest,occupied,origin,resolution=copy.deepcopy(actual_H5_fixed_mode_witness_inputs)
    if fault=='blocked_map':occupied[:]=True
    elif fault=='different_goal':manifest['local_goal_xyyaw'][0]+=1.
    else:source['selected_states'][0]['x_m']+=.1
    report=ledger_module.fixed_mode_candidate_permission_witness(source,manifest,occupied,origin,resolution)
    assert all(not row['permitted_geometric_candidate_exists'] for row in report['methods'])
    assert all(not row['physical_execution_feasibility_claimed'] for row in report['methods'])


def test_permission_review_is_idempotent_and_does_not_relabel_actual_search_failures(ledger_module,tmp_path):
    native=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate/hospital_h5_geometry_candidates/native_planning_probes/17ead2e292304982/planning_probe.json')
    before=native.read_bytes();target=ledger_module.write_fixed_mode_candidate_permission_review(native,tmp_path/'witness')
    report=json.loads(target.read_bytes());saved=target.read_bytes()
    assert all(not row['plan_success'] for row in report['actual_recorded_search_outcomes'])
    assert all(row['permitted_geometric_candidate_exists'] for row in report['methods'])
    assert native.read_bytes()==before
    assert ledger_module.write_fixed_mode_candidate_permission_review(native,target.parent)==target and target.read_bytes()==saved
    target.write_bytes(b'AUTHOR_EDIT_PRESERVED')
    with pytest.raises(ValueError,match='no overwrite'):ledger_module.write_fixed_mode_candidate_permission_review(native,target.parent)
    assert target.read_bytes()==b'AUTHOR_EDIT_PRESERVED' and native.read_bytes()==before


@pytest.mark.parametrize('raw,expected,success',[
    (dict(verdict='PASS',new_route_trials=1),'TASK_COMPLETED',1),
    (dict(verdict='HOLD',new_route_trials=1,error='RuntimeError: H1 path tracking deviation exceeded 80 mm / 10 degrees'),
        'EXECUTION_ABORTED_AT_DECLARED_LIMIT',0),
    (dict(verdict='HOLD',new_route_trials=1,error='RuntimeError: Original sampled swept full-envelope10mm raster guard'),
        'EXECUTION_ABORTED_AT_DECLARED_LIMIT',0),
    (dict(verdict='HOLD',new_route_trials=0,error='Layer changed before Play'),
        'NOT_QUALIFIED_AS_BINARY_METHOD_OUTCOME',None),
    (dict(verdict='HOLD',new_route_trials=1,error='TypeError: Unsupported evidence type'),
        'NOT_QUALIFIED_AS_BINARY_METHOD_OUTCOME',None),
])
def test_paper_outcome_does_not_confuse_raw_HOLD_with_planning_failure_or_software_failure(ledger_module,raw,expected,success):
    before=copy.deepcopy(raw);row=ledger_module.comparison_outcome(raw)
    assert row['outcome']==expected and row['mission_success']==success
    assert raw==before and row['raw_status']==raw['verdict']


def test_bounded_planning_failure_has_no_invented_execution_or_infeasibility(ledger_module):
    raw=dict(plan_success=False,status='PLANNING_TIMEOUT_NOT_PROVEN_INFEASIBLE')
    row=ledger_module.comparison_outcome(raw,planning_only=True)
    assert row['mission_success']==0 and row['execution_success'] is None and row['execution_time_s'] is None
    assert not row['physical_infeasibility_proven']


def test_explicit24_cell_common_protocol_review_is_not_an_adoption_or_rerun_order(ledger_module):
    report=ledger_module.build_common_protocol_review()
    assert report['counts']==dict(comparison_cells=24,explicit_physical_route_references=22,planning_only_references=2,
        physical_references_matching_current_controller=2,
        physical_references_matching_candidate30mm_s_AND_controller=1,
        physical_references_not_matching_candidate30mm_s_AND_controller=21,existing_final_cells_admitted_by_this_audit=0)
    assert len({(r['task'],r['method']) for r in report['references']})==24
    assert not report['actual_motion_authorized'] and report['new_PhysX_trials']==0
    assert not any(r['final_cell_admission'] for r in report['references'])
    assert report['remaining_drive_count_NOT_determined_before_protocol_choice_and_replanning']
    assert report['parameter_mismatch_is_NOT_an_automatic_order_to_repeat_every_trial']
    b0=next(r for r in report['references'] if r['task']=='H3' and r['method'].startswith('B0'))
    assert b0['outcome']['raw_status']=='HOLD' and b0['outcome']['mission_success']==0
    assert b0['selected_target_speeds_m_s']==[.05]


@pytest.fixture
def unified_wheelbase_software_profile(ledger_module,platform_geometry_software_inputs):
    geometry=ledger_module.platform_geometry_comparison(*platform_geometry_software_inputs)
    return ledger_module.unified_wheelbase_profile(geometry)


def test_unified_wheelbase_has_one_forward_geometry_and_preserves_legacy_globals(
        ledger_module,unified_wheelbase_software_profile):
    import math
    import patient_initial_stability.steering_lattice as lattice
    p=unified_wheelbase_software_profile;before=(lattice.WHEELBASE_M,dict(lattice.MODE_CURVATURE_RESPONSE_SCALE))
    kernel=ledger_module.unified_wheelbase_kernel(p)
    assert p['geometry']['wheelbase_m']==1.330 and p['geometry']['front_length_m']==.665
    assert p['geometry']['frame']=='base_link_NOT_CoM'
    assert not p['new_live_profile_installed'] and not p['actual_motion_authorized']
    assert p['transfer']['algebraic_reparameterization_NOT_a_new_identification_or_physical_PASS']
    for mode in lattice.ALL_MODES:
        for front in range(-45,46,5):
            for rear in range(-45,46,5):
                diff=math.tan(math.radians(front))-math.tan(math.radians(rear))
                old=diff/lattice.WHEELBASE_M*lattice.MODE_CURVATURE_RESPONSE_SCALE[mode]
                new=diff/1.33*p['mode_curvature_response_scale'][mode]
                assert new==pytest.approx(old,abs=1e-12)
    assert kernel.steering_targets('front_only',.5)==(math.atan(1.33*.5),0.)
    assert kernel.steering_targets('coordinated_counterphase',.5)==(math.atan(.665*.5),-math.atan(.665*.5))
    assert before==(lattice.WHEELBASE_M,lattice.MODE_CURVATURE_RESPONSE_SCALE)


@pytest.mark.parametrize('fault',['asymmetric_axes','old_asymmetric','missing_gain','negative_gain'])
def test_unified_wheelbase_rejects_unreviewed_geometry_or_fit(ledger_module,platform_geometry_software_inputs,fault):
    r=ledger_module.platform_geometry_comparison(*platform_geometry_software_inputs)
    if fault=='asymmetric_axes':r['current_hospital_USD_source']['active_steering_axes_in_base_m']['front'][0]+=.01
    elif fault=='old_asymmetric':r['preserved_nominal_steering']['front_length_m']+=.01
    elif fault=='missing_gain':del r['preserved_nominal_steering']['mode_curvature_response_scale']['rear_only']
    else:r['preserved_nominal_steering']['mode_lateral_response_scale']['front_only']=-1.
    with pytest.raises(ValueError):ledger_module.unified_wheelbase_profile(r)


@pytest.mark.parametrize('fault',['geometry','gain','lateral_gain','admission'])
def test_unified_wheelbase_private_kernel_cannot_accept_manual_tuning_or_motion_admission(
        ledger_module,unified_wheelbase_software_profile,fault):
    p=copy.deepcopy(unified_wheelbase_software_profile)
    if fault=='geometry':p['geometry']['wheelbase_m']=1.29
    elif fault=='gain':p['mode_curvature_response_scale']['front_only']*=1.01
    elif fault=='lateral_gain':p['mode_lateral_response_scale']['rear_only']*=1.01
    else:p['actual_motion_authorized']=True
    with pytest.raises(ValueError):ledger_module.unified_wheelbase_kernel(p)


def test_unified_wheelbase_replay_is_a_changed_plan_diagnostic_not_new_physical_PASS(
        ledger_module,unified_wheelbase_software_profile):
    r=json.loads((ledger_module.RESULT/'hospital_h3_experiments/cohort_07_held_static_r01/route/P_dynamics_full_envelope/route_readback.json').read_bytes())
    packet=json.loads(Path(r['protocol']).read_bytes());source=Path(packet['original_plan'])
    before=source.read_bytes();plan=json.loads(before)
    result=ledger_module.wheelbase_route_replay(plan,unified_wheelbase_software_profile)
    assert result['status']=='BOTH_PARAMETERIZATIONS_REPLAYED_OFFLINE_ONLY'
    assert result['maximum_legacy_state_difference']<1e-9
    assert result['legacy']['original_25mm_2deg_goal_test']
    assert not result['unified1330']['original_25mm_2deg_goal_test']
    assert result['changed_endpoint_distance_m']>.03
    assert result['existing_verdict_NOT_changed'] and not result['actual_motion_authorized']
    assert source.read_bytes()==before and r['verdict']=='PASS'


def test_unified_wheelbase_legacy_H4_requires_original_nominal_context_not_invented_physical_state(
        ledger_module,unified_wheelbase_software_profile):
    plan=json.loads((ledger_module.RESULT/'hospital_h4_execution_bundle/bundle_r01/B0_fixed_counterphase_ramp.json').read_bytes())
    before=copy.deepcopy(plan);p=unified_wheelbase_software_profile
    assert ledger_module.wheelbase_route_replay(plan,p)['status']=='UNREVIEWED_LEGACY_INITIAL_SCHEMA_NO_MIGRATION_CONCLUSION'
    context=json.loads((ledger_module.RESULT/'hospital_h4_shared_model_seed_screen/screen_r01/shared_model_seed_screen.json').read_bytes())
    result=ledger_module.wheelbase_route_replay(plan,p,legacy_h4_nominal_context=context)
    assert result['status']=='BOTH_PARAMETERIZATIONS_REPLAYED_OFFLINE_ONLY'
    assert result['initial_state_basis']=='SOURCE_BOUND_ORIGINAL_H4_NOMINAL_ZERO_INITIALIZATION_NOT_PHYSICAL_STATE'
    assert plan==before
    plan['initial_front_rad']=.1
    assert ledger_module.wheelbase_route_replay(plan,p,legacy_h4_nominal_context=context)['status']=='UNREVIEWED_LEGACY_INITIAL_SCHEMA_NO_MIGRATION_CONCLUSION'


def test_unified_wheelbase_bad_saved_state_is_quarantined_not_a_migrated_plan(
        ledger_module,unified_wheelbase_software_profile):
    plan=json.loads((ledger_module.RESULT/'hospital_h4_execution_bundle/bundle_r01/B0_fixed_counterphase_ramp.json').read_bytes())
    context=json.loads((ledger_module.RESULT/'hospital_h4_shared_model_seed_screen/screen_r01/shared_model_seed_screen.json').read_bytes())
    plan['selected_states'][0]['x_m']+=.01
    result=ledger_module.wheelbase_route_replay(plan,unified_wheelbase_software_profile,legacy_h4_nominal_context=context)
    assert result['status']=='LEGACY_STATE_REPLAY_NOT_VERIFIED_NO_MIGRATION_CONCLUSION'
    assert 'unified1330' not in result and not result['actual_motion_authorized']


@pytest.mark.parametrize('mode',['coordinated_counterphase','front_only','rear_only'])
def test_unified_wheelbase_command_vector_and_nominal_timing_use_one_private_geometry(
        ledger_module,unified_wheelbase_software_profile,mode):
    import math
    import numpy as np
    import hospital_path_execution as original
    before=original.direct_joint_targets
    b=ledger_module.unified_wheelbase_connections(unified_wheelbase_software_profile)
    for curvature in (-.7,-.5,0.,.25,.5,.7):
        expected=b.python.steering_targets(mode,curvature)
        assert b.steering_geometry(mode,curvature)==expected
        assert np.allclose(b.steering_angles(curvature,mode),expected,rtol=0.,atol=1e-15)
        for speed in (-.05,-.03,.03,.05):
            targets,wheels=b.direct_joint_targets(speed,mode,curvature,expected,.1,expected)
            assert targets==expected
            assert wheels==pytest.approx(tuple(speed/(.074*math.cos(a)) for a in expected))
    assert original.direct_joint_targets is before
    assert b.compiled is None and not b.live_controller_installed and not b.actual_motion_authorized
    assert len(b.bindings)==5


def test_unified_wheelbase_compiled_fallback_encoder_and_timing_verification_is_only_software(
        ledger_module,unified_wheelbase_software_profile):
    pytest.importorskip('numba')
    import patient_initial_stability.steering_lattice as lattice
    from hospital_h3_turn_planning import model_bounds
    before=(lattice.WHEELBASE_M,lattice.FRONT_LENGTH_M,lattice.REAR_LENGTH_M,lattice.propagate,
        dict(lattice.MODE_CURVATURE_RESPONSE_SCALE))
    r=ledger_module.verify_unified_wheelbase_connections(unified_wheelbase_software_profile,
        parameter_bounds=model_bounds())
    assert r['command_queries']==168 and r['native_primitive_queries']==168
    assert r['synthetic_nominal_timing_queries']==6 and r['explicit_source_encoder_queries']==36
    assert r['native_fallback_retains_new_geometry'] and r['exact_native_state_pose_time_comparison']
    assert r['maximum_synthetic_encoder_twist_error']<1e-12
    assert r['new_hospital_plans_generated']==r['new_PhysX_trials']==0
    assert not r['current_live_executor_installed'] and not r['actual_motion_authorized']
    assert before==(lattice.WHEELBASE_M,lattice.FRONT_LENGTH_M,lattice.REAR_LENGTH_M,lattice.propagate,
        lattice.MODE_CURVATURE_RESPONSE_SCALE)


def test_unified_wheelbase_command_binding_rejects_changed_source_anchors(
        ledger_module,unified_wheelbase_software_profile,monkeypatch):
    import hospital_path_execution as direct
    def altered(speed,mode,curvature,previous_steer,dt,measured_steer):
        return ((1.29*curvature,.645*curvature),(0.,0.))
    monkeypatch.setattr(direct,'direct_joint_targets',altered)
    with pytest.raises(ValueError,match='literal anchors'):
        ledger_module.unified_wheelbase_connections(unified_wheelbase_software_profile)


@pytest.mark.parametrize('raw,planning_only,label,score',[
    (dict(verdict='PASS',new_route_trials=1),False,'SUCCESS',1),
    (dict(verdict='HOLD',new_route_trials=1,error='RuntimeError: H1 path tracking deviation exceeded 80 mm / 10 degrees'),
        False,'EXEC_FAIL',0),
    (dict(verdict='FAIL',path_execution_started=True,error='TimeoutError: v2 steering alignment timeout: remaining error 2.51 deg'),
        False,'EXEC_FAIL',0),
    (dict(verdict='FAIL',path_execution_started=True,error='TimeoutError: v2 terminal pulse stalled: under 1 mm movement in 5 simulated seconds'),
        False,'EXEC_FAIL',0),
    (dict(verdict='HOLD',new_route_trials=0,error='Prepared Session boundary changed'),
        False,'ERR_OR_BLOCKED_PENDING_REVIEW',None),
    (dict(verdict='HOLD',new_route_trials=1,error='TypeError: Unsupported evidence type: float64'),
        False,'ERR_OR_BLOCKED_PENDING_REVIEW',None),
    (dict(verdict='FAIL',new_route_trials=1,error='Unreviewed SDK backend exception'),
        False,'PENDING_CAUSE_REVIEW',None),
    (dict(plan_success=False,status='PLANNING_TIMEOUT_NOT_PROVEN_INFEASIBLE'),
        True,'PLAN_FAIL_WITHIN_BUDGET',0),
])
def test_paper_HOLD_display_never_invents_zero_metrics_or_confuses_engineering_and_mission_failure(
        ledger_module,raw,planning_only,label,score):
    before=copy.deepcopy(raw);r=ledger_module.paper_outcome_presentation(raw,planning_only=planning_only)
    assert r['table_label']==label and r['mission_success']==score and r['table_missing_value']=='—'
    assert r['algorithm_success_denominator_eligible']==(score is not None)
    assert r['task_completion_time_available']==(score==1)
    if score is None:assert r['graph_style']=='grey_hatched_NOT_zero_height'
    elif score==0:assert r['graph_style']=='x_at_abort_or_timeout_NO_extension'
    assert r['engineering_exclusions_and_all_original_attempts_must_be_disclosed']
    assert raw==before


def test_unified_h3_whole_search_refiner_and_independent_audit_share_new_geometry_without_global_edit(
        ledger_module,unified_wheelbase_software_profile):
    import math
    import plan_hospital_h1 as original
    import patient_initial_stability.steering_lattice as lattice
    from hospital_h3_turn_planning import load_setup
    old=json.loads((ledger_module.RESULT/'planning/hospital_h3_native_planning_r03/summary.json').read_bytes())
    manifest,occupied,origin,resolution,_=load_setup(old['geometry_manifest'],old['preview_receipt'])
    before=(lattice.WHEELBASE_M,lattice.propagate,original.plan_method)
    method,auditor,connections,parameters=ledger_module.unified_h3_planner_binding(
        unified_wheelbase_software_profile,manifest,occupied,origin,resolution)
    refiner=method.__globals__['refine_speed_consistent_lattice_route']
    assert refiner.__globals__['steering_targets']('front_only',.5)==(math.atan(1.330*.5),0.)
    base_audit=auditor.__globals__['audit_actions']
    assert base_audit.__globals__['propagate'] is connections.python.propagate
    assert parameters['canonical_geometry']['wheelbase_m']==1.330
    assert parameters['private_geometry_binding_anchors']==dict(lattice=1,hierarchy=1,native=1)
    assert parameters['common_additional_planning_reserve_m']==.050
    assert parameters['all_search_refinement_command_geometry_uses_unified_profile']
    assert before==(lattice.WHEELBASE_M,lattice.propagate,original.plan_method)


def test_unified_wheelbase_cannot_refit_legacy_and_transferred_coefficients_together(
        ledger_module,unified_wheelbase_software_profile):
    p=copy.deepcopy(unified_wheelbase_software_profile)
    p['mode_curvature_response_scale']['front_only']*=1.1
    p['transfer']['old_geometry_and_gains_for_provenance_only']['mode_curvature_response_scale']['front_only']*=1.1
    with pytest.raises(ValueError,match='identified coefficients'):
        ledger_module.unified_wheelbase_kernel(p)


@pytest.mark.parametrize('directions,expected',[
    ([1,1],(.05,.03,.01)),([-1,-1],(.05,.03)),([1,-1,1],(.05,.03))])
def test_common_automatic_cruise_candidates_are_direction_based_not_method_based(ledger_module,directions,expected):
    import inspect
    actions=[dict(direction=d) for d in directions]
    assert ledger_module.common_cruise_profile_caps(actions)==expected
    assert 'method' not in inspect.signature(ledger_module.common_cruise_profile_caps).parameters
    assert actions==[dict(direction=d) for d in directions]


@pytest.mark.parametrize('actions,levels',[
    ([],(.03,.05)),([dict(direction=True)],(.03,.05)),([dict(direction=0)],(.03,.05)),
    ([dict(direction=-1)],(.003,.01)),([dict(direction=1)],(.03,float('nan')))])
def test_common_cruise_policy_rejects_invalid_or_below_directional_floor_candidates(ledger_module,actions,levels):
    with pytest.raises(ValueError):ledger_module.common_cruise_profile_caps(actions,levels)


def test_guided_P_never_turns_missing_own_seed_into_cached_route_or_motion_PASS(ledger_module):
    seed=dict(plan_success=False)
    row=ledger_module.unified_h3_guided_candidate(None,seed,None,None,None,None,None,budget_s=10.)
    assert not row['plan_success'] and row['status']=='OWN_SPATIAL_SEED_NOT_FOUND_WITHIN_BUDGET'
    assert row['selected_actions']==row['selected_states']==[]
    assert not row['cached_other_trial_or_measured_state_used'] and not row['actual_motion_authorized']


@pytest.mark.parametrize('first_success',[True,False])
def test_owned_multimode_seed_has_same_fixed_total_budget_and_no_borrowed_pool(ledger_module,first_success):
    calls=[]
    def method(family,*args):
        budget,node_limit,pool=args[-3:]
        calls.append((family,budget,node_limit,pool))
        return dict(method=family,plan_success=first_success or len(calls)==2,
            status='FOUND' if first_success or len(calls)==2 else 'BOUNDED_NO_PLAN',planning_time_s=.1)
    row=ledger_module.unified_P_owned_seed(method,None,None,None,None,None,
        budget_s=180.,node_limit=60000,policy=ledger_module.UNIFIED_P_SEED_POLICY)
    assert calls[0][0]==ledger_module.METHODS[1] and calls[0][1]<=30.
    assert len(calls)==(1 if first_success else 2)
    if len(calls)==2:assert calls[1][0]==ledger_module.METHODS[0] and calls[1][1]<=45.
    assert all(c[2]==60000 and c[3]=={} for c in calls)
    assert row['plan_success'] and row['no_cached_baseline_plan_or_measured_state_used']


@pytest.mark.parametrize('policy,budget,nodes',[
    ('task_specific_H1',180.,60000),('FIXED_COUNTERPHASE_v1',181.,60000),
    ('FIXED_COUNTERPHASE_v1',180.,60001),('FIXED_COUNTERPHASE_v1',float('nan'),60000)])
def test_owned_seed_rejects_task_specific_policy_or_expanded_budget(ledger_module,policy,budget,nodes):
    with pytest.raises(ValueError):ledger_module.unified_P_owned_seed(
        None,None,None,None,None,None,budget_s=budget,node_limit=nodes,policy=policy)


@pytest.mark.parametrize('times,budget,expected',[
    ([5.]*38,180.,False),([5./3]*38,180.,True),([1.,2.,3.],180.,True),([100.,100.],240.,True)])
def test_nominal_P_profile_clock_rejects_already_over_budget_without_changing_limits(ledger_module,times,budget,expected):
    r=ledger_module.nominal_execution_clock(times,budget)
    assert r['admissible'] is expected and r['declared_simulation_budget_s']==budget
    assert r['nominal_edges_plus_fixed_prehold_and_hold_s']==pytest.approx(sum(times)+7.)
    assert r['NOT_a_physical_completion_time_bound_or_full_executor_model_PASS']


@pytest.mark.parametrize('times,budget',[([],180.),([0.],180.),([float('nan')],180.),([1.],601.)])
def test_nominal_profile_clock_does_not_invent_a_longer_task_budget(ledger_module,times,budget):
    with pytest.raises(ValueError):ledger_module.nominal_execution_clock(times,budget)


@pytest.mark.parametrize('budget',[0.,-1.,181.,float('inf'),float('nan')])
def test_guided_P_cannot_expand_its_remaining_common_compute_budget(ledger_module,budget):
    with pytest.raises(ValueError,match='180s budget'):
        ledger_module.unified_h3_guided_candidate(None,None,None,None,None,None,None,budget_s=budget)


def test_unified_mission_model_binds_command_alignment_and_plant_without_legacy_global_edits(
        ledger_module,unified_wheelbase_software_profile):
    import math
    import hospital_mission_execution as mission
    import hospital_path_execution as direct
    import screen_hospital_h4_common_executor as screen
    from test_hospital_timed_path_execution import mission_fixture
    p,g=mission_fixture();p['geometry_profile']=unified_wheelbase_software_profile
    g['same_three_authored_support_rectangles']=[
        [[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]],
        [[2.,-2.],[4.,-2.],[4.,2.],[2.,2.]],
        [[4.,-2.],[6.,-2.],[6.,2.],[4.,2.]]]
    before=(direct.direct_joint_targets,mission.steering_angles,screen.case,dict(screen.MODE_CURVATURE_RESPONSE_SCALE))
    original=copy.deepcopy((p,g))
    run,execution,design=ledger_module.unified_mission_model_binding(
        unified_wheelbase_software_profile,g,p,simulation_budget_s=600.)
    c=execution.H4TrackingController(p)
    assert c._quiet_step.__globals__['steering_angles'](.5,'front_only')[0]==pytest.approx(math.atan(1.33*.5))
    assert c.tracker.step.__func__.__globals__['steering_geometry']('front_only',.5)==(math.atan(1.33*.5),0.)
    terminal=c.raw(c.reference,c.goal)
    assert terminal.step.__func__.__globals__['steering_geometry']('rear_only',.5)==(0.,-math.atan(1.33*.5))
    assert execution.direct_joint_targets is run.connections.direct_joint_targets
    assert c.tracking_step.mission_binding['method_identifier_NOT_used_for_control_policy']
    assert c._quiet_step.stop_binding['original8s_stop_timeout_and_postcapture5mm_s_guard_unchanged']
    assert c.tracker.config['stationary_hold_entry_position_m']==.030
    assert design==g and (p,g)==original
    assert before==(direct.direct_joint_targets,mission.steering_angles,screen.case,screen.MODE_CURVATURE_RESPONSE_SCALE)


@pytest.mark.parametrize('fault',['profile','physical_claim','geometry','budget'])
def test_unified_mission_model_cannot_invent_admission_or_change_task_limits(
        ledger_module,unified_wheelbase_software_profile,fault):
    from test_hospital_timed_path_execution import mission_fixture
    p,g=mission_fixture();p['geometry_profile']=copy.deepcopy(unified_wheelbase_software_profile);budget=600.
    if fault=='profile':p['geometry_profile']['geometry']['wheelbase_m']=1.29
    elif fault=='physical_claim':p['physical_execution_verified']=True
    elif fault=='geometry':g['goal_xyyaw'][0]+=.01
    else:budget=900.
    with pytest.raises(ValueError):ledger_module.unified_mission_model_binding(
        unified_wheelbase_software_profile,g,p,simulation_budget_s=budget)


def test_unified_common_lifecycle_keeps_actual_ACK_contract_and_new_terminal_geometry(
        ledger_module,unified_wheelbase_software_profile):
    import math
    import hospital_mission_sequence as sequence
    from test_hospital_timed_path_execution import mission_fixture
    p,g=mission_fixture();p['geometry_profile']=unified_wheelbase_software_profile
    c=sequence.unified_geometry_lifecycle_controller(p,g,profile=unified_wheelbase_software_profile,
        simulation_budget_s=600.)
    assert c.contract['hold_ACK_is_actual_backend_readback_NOT_request_echo']
    assert c.contract['acceptance']==sequence.lifecycle.ACCEPTANCE
    assert c.contract['command_limits']==sequence.lifecycle.COMMAND_LIMITS
    assert not c.contract['actual_motion_authorized']
    assert c.motion.raw(c.motion.reference,c.motion.goal).step.__func__.__globals__[
        'steering_geometry']('front_only',.5)==(math.atan(1.33*.5),0.)
    with pytest.raises(ValueError,match='legacy route binding prohibited'):
        sequence.route_controller({},p,g)


def test_unified_SDK_bridge_replaces_in_function_legacy_import_without_state_or_gain_edits(
        ledger_module,unified_wheelbase_software_profile):
    import math
    import numpy as np
    from types import SimpleNamespace
    import hospital_mission_sequence as sequence
    from test_hospital_timed_path_execution import _sdk_fixture
    backend,adapter=_sdk_fixture();physical=backend.physical_q.copy()
    gains={k:backend.data[k].copy() for k in ('kp','kd')}
    def step(*args,**kw):
        assert kw['drive_hold_confirmed'] is False
        return dict(speed_m_s=.03,mode='front_only',curvature_1_m=.5,drive_hold_requested=False)
    c=SimpleNamespace(geometry_profile=unified_wheelbase_software_profile,
        contract=dict(geometry_profile=unified_wheelbase_software_profile),stage='tracking',step=step)
    bridge=sequence.RetainFinalTargetsBridge(c,adapter,guard=lambda *a:None)
    for i in range(1,48):bridge.step(1.+i*.1,[0.,0.,0.],.03)
    assert bridge.steering_command==(math.atan(1.33*.5),0.)
    assert np.array_equal(backend.physical_q,physical)
    assert all(np.array_equal(backend.data[k],v) for k,v in gains.items())
    assert bridge._candidate_step.geometry_binding['undo_recovers_entire_original_SDK_AST']
    assert bridge._candidate_step.geometry_binding['actual_ACKs_and_gains_state_clock_and_guards_unchanged']


@pytest.mark.parametrize('method',[
    'B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'])
def test_unified_H3_runtime_profile_is_explicit_and_identical_for_all_four_methods(
        ledger_module,unified_wheelbase_software_profile,method):
    import hospital_mission_sequence as sequence
    import hospital_h4_path_execution as original
    from test_hospital_timed_path_execution import mission_fixture
    p,g=mission_fixture();p['scenario']=g['scenario']='H3_doorway_immediate_turn'
    p.update(method=method,planner_dynamics=original.method_settings(method)['planner_dynamics'],
        allowed_modes=['coordinated_counterphase'] if method.startswith('B0_') else list(original.MODES),
        geometry_profile=unified_wheelbase_software_profile)
    packet=dict(scenario=p['scenario'],method=method,simulation_budget_s=300.,
        explicit_route_approval=sequence.H3_CONTINUOUS_APPROVAL,
        geometry_profile=unified_wheelbase_software_profile,common_tracking_policy=sequence.FINAL_TARGET_HOLD_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020)
    packet['common_lifecycle_contract']=sequence.checked_route_lifecycle_contract(packet)
    c=sequence.route_controller(packet,p,g)
    assert c.contract==packet['common_lifecycle_contract']
    assert c.geometry_profile==p['geometry_profile'] and c.motion.geometry_profile==p['geometry_profile']
    assert c.contract['same_policy_for_all_four_methods'] and not c.contract['actual_motion_authorized']
    bad=copy.deepcopy(packet);bad.pop('geometry_profile')
    with pytest.raises(ValueError,match='legacy route binding prohibited'):sequence.route_controller(bad,p,g)


def test_unified_MODEL_guard_counts_terminal_reversals_without_fabricated_physical_channels(
        ledger_module,unified_wheelbase_software_profile):
    import numpy as np
    from test_hospital_timed_path_execution import mission_fixture
    p,g=mission_fixture();p['geometry_profile']=unified_wheelbase_software_profile
    g['same_three_authored_support_rectangles']=[
        [[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]],[[2.,-2.],[4.,-2.],[4.,2.],[2.,2.]],
        [[4.,-2.],[6.,-2.],[6.,2.],[4.,2.]]]
    run,_,_=ledger_module.unified_mission_model_binding(unified_wheelbase_software_profile,g,p,simulation_budget_s=600.)
    guard=run.model_guard_class(g,np.zeros((400,800),bool),np.array([-2.,-2.]),.010,simulation_budget_s=600.)
    for i,sign in enumerate((1,-1,1,-1,1)):
        guard(dict(time_s=i*.02,x_m=g['goal_xyyaw'][0],y_m=0.,yaw_rad=0.),
            dict(speed_m_s=sign*.01,phase='terminal_correction'),{},[])
    assert guard.direction_changes==4
    with pytest.raises(RuntimeError,match='maximum4command direction changes'):
        guard(dict(time_s=.1,x_m=g['goal_xyyaw'][0],y_m=0.,yaw_rad=0.),dict(speed_m_s=-.01,phase='terminal_correction'),{},[])
    assert guard.failed and guard.direction_changes==5


def test_unified_H3_planning_reserves_two_of_four_changes_without_relaxing_physics_or_runtime(
        ledger_module,unified_wheelbase_software_profile):
    import hospital_mission_planning as resource
    from hospital_h3_turn_planning import load_setup
    old=json.loads((ledger_module.RESULT/'planning/hospital_h3_native_planning_r03/summary.json').read_bytes())
    manifest,occupied,origin,resolution,_=load_setup(old['geometry_manifest'],old['preview_receipt'])
    method,_,_,parameters=ledger_module.unified_h3_planner_binding(
        unified_wheelbase_software_profile,manifest,occupied,origin,resolution,terminal_direction_change_reserve=2)
    refine=method.__globals__['refine_speed_consistent_lattice_route'];ns=refine.__globals__
    state=ns['State'](0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    feasible=[]
    for direction in (1,-1,1,-1):
        state,poses,_=ns['propagate'](state,'coordinated_counterphase',0.,direction,speed_m_s=.05)
        feasible.append(poses.resource_checks['direction_changes'])
    assert feasible==[True,True,True,False]
    assert parameters['maximum_direction_changes']==2 and parameters['reserved_direction_changes_for_terminal']==2
    assert parameters['original_TOTAL_runtime_direction_change_cap']==4
    assert resource.MotionResourceLimits().maximum_direction_changes==4


@pytest.mark.parametrize('reserve',[True,-1,1,3,4])
def test_unified_H3_reserve_does_not_allow_arbitrary_or_over_budget_values(
        ledger_module,unified_wheelbase_software_profile,reserve):
    with pytest.raises(ValueError,match='original runtime maximum4'):
        ledger_module.unified_h3_planner_binding(unified_wheelbase_software_profile,None,None,None,None,
            terminal_direction_change_reserve=reserve)


def test_verified_direction_resource_abort_is_execution_failure_not_a_technical_HOLD(ledger_module):
    raw=dict(verdict='HOLD',new_route_trials=1,error='RuntimeError: Original maximum4command direction changes exceeded')
    row=ledger_module.paper_outcome_presentation(raw)
    assert row['mission_success']==0 and row['table_label']=='EXEC_FAIL'
    assert row['raw_status']=='HOLD' and not row['task_completion_time_available']


def test_actual_alignment_stop_guard_abort_is_not_a_camera_or_serialization_error(ledger_module):
    raw=dict(verdict='HOLD',new_route_trials=1,error='RuntimeError: Alignment requires measured stop')
    row=ledger_module.paper_outcome_presentation(raw)
    assert row['raw_status']=='HOLD' and row['table_label']=='EXEC_FAIL'
    assert row['mission_success']==0 and row['failed_route_partial_metrics_must_be_labelled_at_abort']


@pytest.mark.parametrize('method',[
    'B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'])
def test_reviewed_H6_wall_budget_is_identical_for_all_methods_and_changes_NO_motion(
        ledger_module,unified_wheelbase_software_profile,method):
    import hospital_mission_sequence as sequence
    import hospital_h4_path_execution as original
    from test_hospital_timed_path_execution import mission_fixture
    p,g=mission_fixture();p['scenario']=g['scenario']=sequence.TASK_SCENARIOS['H6']
    p.update(method=method,planner_dynamics=original.method_settings(method)['planner_dynamics'],
        allowed_modes=['coordinated_counterphase'] if method.startswith('B0_') else list(original.MODES),
        geometry_profile=unified_wheelbase_software_profile)
    packet=dict(scenario=p['scenario'],method=method,simulation_budget_s=600.,maximum_wall_time_s=1500.,
        explicit_route_approval=sequence.REMAINING_NO_NAV2_APPROVAL,
        common_wall_budget_review=copy.deepcopy(sequence.H6_COMMON_WALL_REVIEW),
        geometry_profile=unified_wheelbase_software_profile,common_tracking_policy=sequence.FINAL_TARGET_HOLD_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020)
    packet['common_lifecycle_contract']=sequence.checked_route_lifecycle_contract(packet)
    assert sequence.checked_common_wall_budget(packet)==1500.
    c=sequence.route_controller(packet,p,g)
    legacy=sequence.unified_geometry_lifecycle_controller(p,g,profile=unified_wheelbase_software_profile,
        simulation_budget_s=600.)
    old=copy.deepcopy(legacy.contract);new=copy.deepcopy(c.contract)
    assert new.pop('additional_reviewed_common_PC_wall_allowance_only_changed') is True
    assert new.pop('wall_budget_s')==1500. and old.pop('wall_budget_s')==900.
    assert new.pop('original_common_controller_and_limits_changed_ONLY_geometry_bindings') is False
    assert old.pop('original_common_controller_and_limits_changed_ONLY_geometry_bindings') is True
    assert new==old
    assert c.contract['original_motion_contract']==legacy.contract['original_motion_contract']
    assert c.motion._quiet_step.__code__==legacy.motion._quiet_step.__code__
    assert c.motion.geometry_profile==legacy.motion.geometry_profile
    assert c.contract['acceptance']==legacy.contract['acceptance']
    assert c.contract['command_limits']==legacy.contract['command_limits']
    assert not c.contract['actual_motion_authorized'] and c.contract['simulation_budget_s']==600.


@pytest.mark.parametrize('fault',['missing_review','changed_approval','P_only','scenario','simulation',
    'method','unreviewed900','nan','inf','bool','zero','arbitrary'])
def test_reviewed_H6_wall_budget_refuses_implicit_extensions_and_changed_scope(fault):
    import hospital_mission_sequence as sequence
    packet=dict(scenario=sequence.TASK_SCENARIOS['H6'],simulation_budget_s=600.,maximum_wall_time_s=1500.,
        method=sequence.METHODS[0],common_wall_budget_review=copy.deepcopy(sequence.H6_COMMON_WALL_REVIEW))
    if fault=='missing_review':packet.pop('common_wall_budget_review')
    elif fault=='changed_approval':packet['common_wall_budget_review']['approval']['user_message']='ASSUMED'
    elif fault=='P_only':packet['common_wall_budget_review']['methods']=[sequence.METHODS[0]]
    elif fault=='scenario':packet['scenario']=sequence.TASK_SCENARIOS['H5']
    elif fault=='simulation':packet['simulation_budget_s']=900.
    elif fault=='method':packet['method']='UNKNOWN'
    elif fault=='unreviewed900':packet['maximum_wall_time_s']=900.
    else:packet['maximum_wall_time_s']={'nan':float('nan'),'inf':float('inf'),'bool':True,'zero':0.,'arbitrary':1501.}[fault]
    with pytest.raises(ValueError):sequence.checked_common_wall_budget(packet)


def test_legacy_common_wall_default_and_explicit900_are_unchanged():
    import hospital_mission_sequence as sequence
    assert sequence.checked_common_wall_budget({})==900.
    assert sequence.checked_common_wall_budget(dict(maximum_wall_time_s=900.))==900.
    assert sequence.checked_common_wall_budget(dict(maximum_wall_time_s=900))==900.


@pytest.mark.parametrize('limit,elapsed,expired',[(None,1200.,True),(900.,1200.,True),(1500.,1200.,False),(1500.,1500.1,True)])
def test_actual_retained_host_uses_reviewed_protocol_wall_limit_NOT_hidden900(monkeypatch,limit,elapsed,expired):
    import asyncio
    from types import SimpleNamespace
    import hospital_mission_lifecycle as lifecycle
    host=SimpleNamespace(clock=0.,motion_wall=0.,protocol={} if limit is None else dict(maximum_wall_time_s=limit),
        stage_id=1,encoded=2,errors=[],same_context=lambda:None,
        sim=SimpleNamespace(wake_up=lambda *a:None),timeline=SimpleNamespace(is_playing=lambda:True))
    async def tick():host.clock=1.
    host.app=SimpleNamespace(next_update_async=tick)
    monkeypatch.setattr(lifecycle.time,'monotonic',lambda:elapsed)
    if expired:
        with pytest.raises(TimeoutError,match='Declared'+str(int(900. if limit is None else limit))+'s whole-route wall cap'):
            asyncio.run(lifecycle.RetainedMissionSDKHost.next_sample(host))
    else:asyncio.run(lifecycle.RetainedMissionSDKHost.next_sample(host))
    assert host.clock==1.

