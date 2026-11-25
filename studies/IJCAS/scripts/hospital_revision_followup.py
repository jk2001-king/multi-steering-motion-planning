"""Targeted reviewer follow-up, never a replacement of the completed campaign.

Uses the existing planning/static/guarded execution tools. No common physical
controller, speed policy, threshold, environment or prior outcome is changed.
Development conditions are explicitly NOT held-out confirmation repetitions.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace

import hospital_revision_planning as planning
import hospital_revision_mechanisms as mechanisms

campaign = planning.campaign
runtime = planning.runtime
OUTPUT = campaign.GATE/'revision_followup_20261122'
PROTOCOL = OUTPUT/'protocol.json'
LEDGER = campaign.OUTPUT/'final_evidence/20261121_233144_359073/complete_evidence_ledger.json'
PILOT_IDS = tuple(task+'_P_dynamics_full_envelope_'+repeat for task,repeat in (
    ('H2','s02'),('H3','s01'),('H3','s05'),('H4','s03'),('H6','s03')))


def definition():
    original = runtime.read(campaign.PROTOCOL)
    cells = [copy.deepcopy(c) for c in original['main_cells'] if c['cell_id'] in PILOT_IDS]
    return dict(version='TARGETED_P_AND_MECHANISM_REVIEW_v1',
        main_cells=cells, tasks=original['tasks'],
        original_evidence_inputs_sha256=original['original_evidence_inputs_sha256'],
        common_speed_policy=original['common_speed_policy'],
        parent_protocol=str(campaign.PROTOCOL), parent_protocol_sha256=runtime.sha(campaign.PROTOCOL),
        old_final_ledger=str(LEDGER), old_final_ledger_sha256=runtime.sha(LEDGER),
        scope='bounded development and mechanism follow-up, not a new all-method campaign',
        development_conditions_already_seen_NOT_held_out=True,
        old_120_main_10_history_60_joint_results_never_overwritten_or_pooled=True,
        P_seed_portfolio=[dict(method=m,maximum_s=s) for m,s in (
            (campaign.planning.B2,10.),(campaign.METHODS[0],20.),(campaign.METHODS[1],45.))],
        seed_total_budget_s=45., planning_total_budget_s=180., lattice_expansion_limit=60000,
        uniform_P_lookahead_edges_all_tasks=4, beam_width=64, local_iterations=24,
        ablation=mechanisms.ablation_definition(),
        no_common_controller_speed_geometry_or_success_criteria_changes=True,
        no_other_methods_automatically_rerun=True,
        unsafe_reference_paths_not_executed=True,
        physical_trial_maximum_per_development_cell=1,
        no_automatic_physical_retry_or_success_replacement=True,
        current_user_authority='보완·검증을 자율 진행하되 변경 때문에 전체 비교군·시나리오를 처음부터 재실험하는 것은 피한다',
        photograph_deferred_by_author=True, manuscript_edits=False)


def export_protocol():
    value = runtime.evidence.json_value(definition())
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL)!=value:
            raise ValueError('Prospective follow-up definition already frozen')
        return PROTOCOL
    runtime.evidence.write_exclusive(PROTOCOL,value)
    files = (Path(__file__),Path(mechanisms.__file__),Path(planning.__file__))
    runtime.lc.freeze_workflow_sources(OUTPUT/'design_sources.zip',
        {str(p.resolve()):runtime.sha(p) for p in files})
    return PROTOCOL


def seed_portfolio(method,manifest,occupied,origin,resolution,model,*,budget_s,node_limit,policy):
    """Prefer transition-aware geometry, with a bounded simpler-mode fallback.

    Same schedule for every task, no task ID or prior successful path reuse.
    The three slice limits are ceilings within ONE45s seed/180s total clock.
    """
    definition = runtime.read(PROTOCOL)
    start=time.perf_counter();attempts=[];seed=None;expanded=0
    for item in definition['P_seed_portfolio']:
        remaining=min(45.,budget_s)-(time.perf_counter()-start)
        if remaining<=0. or expanded>=node_limit:
            break
        limit=min(item['maximum_s'],remaining)
        seed=method(item['method'],manifest,occupied,origin,resolution,model,
            limit,node_limit-expanded,{})
        count=int(seed.get('expanded_nodes',0))
        expanded+=count
        attempts.append(dict(family=item['method'],budget_s=limit,status=seed['status'],
            planning_time_s=seed['planning_time_s'],plan_success=seed['plan_success'],
            reported_expanded_nodes=count))
        if seed['plan_success']:
            break
    if seed is None:
        raise RuntimeError('No seed search dispatched')
    return dict(seed,owned_seed_policy='COMMON_B2_10_B0_20_B1_REMAINDER45_v1',
        owned_seed_attempts=attempts,owned_seed_total_time_s=time.perf_counter()-start,
        no_cached_baseline_plan_or_measured_state_used=True)


def planner_bindings(history_policy):
    def bind(profile,*args,**kwargs):
        kernel=planning.shared.unified_wheelbase_kernel(profile)
        def wrap(original,*,no_entry_history,activity):
            if no_entry_history:
                raise ValueError('Do not confuse legacy zero resets with steady-state ablation')
            return mechanisms.bind_propagation(original,kernel=kernel,profile=profile,
                history_policy=history_policy,activity=activity)
        return runtime.bind(planning.planner_bindings,propagation_binding=wrap)(profile,*args,**kwargs)
    return bind


def run_plan(cell_id, *, history_policy='carry'):
    export_protocol()
    if history_policy not in mechanisms.HISTORY_POLICIES:
        raise ValueError('Explicit history intervention required')
    private_campaign=SimpleNamespace(**dict(vars(campaign),PROTOCOL=PROTOCOL,OUTPUT=OUTPUT))
    private_shared=SimpleNamespace(**dict(vars(planning.shared),unified_P_owned_seed=seed_portfolio))
    folder=OUTPUT/'planning'/history_policy
    bound=runtime.bind(planning.run_cell,campaign=private_campaign,shared=private_shared,
        OUTPUT=folder,planner_bindings=planner_bindings(history_policy),__file__=__file__)
    result=bound(cell_id,lookahead_edges=4)
    result=dict(result,predictive_history_policy=history_policy,
        prospective_definition=str(PROTOCOL),development_NOT_final_repetition=True,
        old_negative_result_NOT_replaced=True)
    runtime.evidence.write_exclusive(folder/cell_id/'followup_summary.json',result)
    return result


def mechanism_review():
    """Source/log review and fixed-input sensitivity, not physical validation."""
    export_protocol()
    value=runtime.read(LEDGER)
    activity=value['force_load_candidate_activity']
    rows=[];maxima=[];weights=[]
    for row in value['records']:
        if row['method']!=campaign.planning.P or not row['planning_success']:
            continue
        summary=runtime.read(row['planning_summary'])
        plan=runtime.read(summary['plan'])
        bounds=plan['rigid_body_parameter_bounds']
        weights.append({k:v for k,v in bounds.items() if k.endswith('cost_weight_s')})
        maxima.append(dict(task=row['task'],condition=row['paired_condition_id'],
            friction=max(s['friction_utilization'] for s in plan['selected_states']),
            LTR=max(s['abs_ltr'] for s in plan['selected_states'])))
        # Pair the four interventions on the EXACT same frozen controls.
        kernel=planning.shared.unified_wheelbase_kernel(plan['geometry_profile'])
        from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
        from patient_initial_stability.steering_lattice import State,FULL_DYNAMICS
        model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
        for variant in mechanisms.HISTORY_POLICIES:
            propagate=mechanisms.bind_propagation(kernel.propagate,kernel=kernel,
                profile=plan['geometry_profile'],history_policy=variant)
            state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
            elapsed=0.;max_saved_difference=0.;infeasible=0
            for action,saved in zip(plan['selected_actions'],plan['selected_states']):
                state,poses,dt=propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
                    speed_m_s=.05,dynamics=FULL_DYNAMICS,rigid_body_model=model,
                    target_speed_m_s=action['target_speed_m_s'])
                elapsed+=dt;infeasible+=int(not state.rigid_body_feasible)
                if variant=='carry':
                    import math
                    max_saved_difference=max(max_saved_difference,abs(state.x-saved['x_m']),
                        abs(state.y-saved['y_m']),abs(math.atan2(math.sin(state.yaw-math.radians(saved['yaw_deg'])),
                            math.cos(state.yaw-math.radians(saved['yaw_deg'])))))
            import math
            goal=plan['goal_local_xyyaw']
            if variant=='carry' and max_saved_difference>1e-8:
                raise ValueError('Original full-history reference replay did not reproduce saved states')
            rows.append(dict(task=row['task'],condition=row['paired_condition_id'],variant=variant,
                final_position_error_mm=1000*math.hypot(state.x-goal[0],state.y-goal[1]),
                final_yaw_error_deg=math.degrees(math.atan2(math.sin(state.yaw-goal[2]),math.cos(state.yaw-goal[2]))),
                model_duration_s=elapsed,infeasible_primitive_count=infeasible,
                carry_saved_state_maximum_difference=max_saved_difference if variant=='carry' else None,
                frozen_plan=summary['plan'],not_actual_tracking_RMS=True,not_new_planning_repetition=True))
    report=dict(version='REVIEWER_REQUIREMENTS_AND_MECHANISMS_AUDIT_v1',
        old_ledger=str(LEDGER),old_ledger_sha256=runtime.sha(LEDGER),
        ablation_review=dict(legacy_variant='P_no_entry_state_history',
            legacy_issue='zero-entry every50mm adds repeated-start and segmentation effects; not a clean instantaneous-response comparison',
            legacy_results_preserved=True,corrected_definition=mechanisms.ablation_definition(),
            fixed_control_rollouts=rows,physical_replanned_ablation_still_pending=True),
        force_load=dict(candidate_activity=activity,selected_path_peaks=maxima,
            actual_configured_cost_weights=weights,
            formulation_and_implemented_screen_may_be_retained=True,
            clinical_motor_or_causal_completion_guarantees_not_supported=True,
            inactive_screen_is_not_evidence_of_a_path_selection_effect=True,
            artificially_lowering_limits_to_manufacture_effect_is_not_allowed=True),
        required_scope_corrections=dict(
            common_command_policy_NOT_identical_measured_velocity=True,
            joint_block_is_reused_plan_execution_sensitivity_NOT_repeated_initial_state_planning=True,
            zero_success_ablation_cannot_supply_completed_pair_accuracy_reduction=True,
            full_envelope_offline_violation_NOT_measured_physical_collision=True,
            old_selected_case_tables_NOT_success_rates=True,
            new_main_all_H6_methods_wall1500_and_all_tasks_P_lookahead8=True,
            new_P_v2_lookahead4_uniform_is_a_separate_planner_version=True,
            physical_photo_waiting_for_author_not_an_experiment_omission=True),
        common_controller_and_prior_experiment_records_modified=False,
        new_motion_trials=0,manuscript_modified=False)
    target=OUTPUT/'mechanism_review.json'
    runtime.evidence.write_exclusive(target,report)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['definition','review','plan'])
    parser.add_argument('cell_id',nargs='?')
    parser.add_argument('--history-policy',choices=mechanisms.HISTORY_POLICIES,default='carry')
    args=parser.parse_args()
    print(export_protocol() if args.action=='definition' else mechanism_review() if args.action=='review'
          else run_plan(args.cell_id,history_policy=args.history_policy))
