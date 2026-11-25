"""Pure final-block PROPOSAL, not dispatch or an approved/completed protocol.

Fresh combined pose/joint draws avoid promoting seen development replays.
One fixed execution/search version must be used by all declared conditions.
Export/admission is deliberately separate: importing or inspecting this
proposal cannot open a stage, plan a path, or command a simulator.
"""
import copy
import math

import numpy as np
import hospital_revision_campaign as original
import hospital_revision_initial_conditions as initial
import hospital_revision_terminal_geometry as control


def definition(repeats=5, *, seed=2026112205):
    if type(repeats) is not int or repeats not in (3,5):
        raise ValueError('Explicit three- or five-condition matched proposal required')
    if type(seed) is not int or seed in (20261121,2026112102):
        raise ValueError('Fresh fixed seed, not either development-condition seed')
    previous=original.runtime.read(original.PROTOCOL)
    rng=np.random.default_rng(seed);cells=[]
    for number in range(1,7):
        task=f'H{number}';spec=previous['tasks'][task]
        x,y,yaw=spec['original_start_xyyaw']
        for repeat in range(repeats):
            dx,dy=map(float,rng.uniform(-.010,.010,2))
            dyaw=float(rng.uniform(-math.radians(.5),math.radians(.5)))
            start=[x+math.cos(yaw)*dx-math.sin(yaw)*dy,
                y+math.sin(yaw)*dx+math.cos(yaw)*dy,yaw+dyaw]
            angles=dict(zip((*initial.ACTIVE,*initial.CASTERS),map(float,
                np.r_[rng.uniform(-.5,.5,2),rng.uniform(-20.,20.,4)])))
            shift=(repeat+number-1)%4
            order=original.METHODS[shift:]+original.METHODS[:shift]
            for index,method in enumerate(order):
                cells.append(dict(cell_id=f'{task}_{method}_f{repeat+1:02d}',task=task,method=method,
                    paired_condition_id=f'{task}_final_f{repeat+1:02d}',order_within_condition=index,
                    start_xyyaw=list(start),goal_xyyaw=copy.deepcopy(spec['goal_xyyaw']),
                    body_pose_perturbation=[dx,dy,dyaw],initial_joint_angles_deg=copy.deepcopy(angles),
                    initial_joint_velocities_deg_s=0.,
                    initial_active_steering_rule='fresh source perturbation; measure static and tracking entry; retain common release tolerance'))
    ablations=[]
    for cell in cells:
        if cell['method']==original.planning.P and cell['task'] in ('H2','H3'):
            ablations.append(dict(copy.deepcopy(cell),parent_main_cell_id=cell['cell_id'],
                cell_id=cell['cell_id']+'_steady_both',variant='P_steady_current_input_entry_projection',
                remove_predictive_entry_history_only=True,do_NOT_reset_physical_state_or_pose=True))
    return dict(version='FINAL_COMBINED_POSE_JOINT_MATCHED_DESIGN_PROPOSAL_v1',
        status='PROPOSAL_NOT_APPROVED_NOT_DISPATCHED_NOT_RESULTS',repeats_per_task_method=repeats,
        random_generator=f'numpy.default_rng({seed})',fresh_draws_NOT_seen_development_replays=True,
        draws_NOT_resampled_or_selected_using_P_outcomes=True,
        main_cells=cells,history_ablation_cells=ablations,tasks=copy.deepcopy(previous['tasks']),
        original_evidence_inputs_sha256=copy.deepcopy(previous['original_evidence_inputs_sha256']),
        main_planned_cells=24*repeats,additional_history_ablation_cells=2*repeats,
        maximum_conditional_routes=26*repeats,counts_NOT_approved_or_guaranteed_drive_counts=True,
        combined_pose_and_joint_variation_shared_by_ALL_four_methods=True,
        separate_new60_joint_block_NOT_required_by_this_combined_design=True,
        nominal_planning_initial_active_angles_deg=[0.,0.],
        physical_joint_draws_are_bounded_initial_state_sensitivity_NOT_perfectly_known_planner_states=True,
        actual_first_bound_and_tracking_entry_joint_angles_required=True,
        common_speed_policy=copy.deepcopy(previous['common_speed_policy']),
        common_execution_controller_version=control.VERSION,
        P_seed_portfolio=[dict(method=m,maximum_s=s) for m,s in (
            (original.METHODS[1],10.),(original.METHODS[0],20.),(original.planning.B2,45.))],
        seed_total_budget_s=45.,planning_total_budget_s=180.,
        lattice_expansion_limit_per_search_call=60000,uniform_P_lookahead_edges_all_tasks=4,
        beam_width=64,local_iterations=24,
        P_refinement='same strict joint-goal beam ranking and native distributed-index local search for every task',
        nominal_execution_model_prediction_is_diagnostic_for_ALL_methods_and_ablation=True,
        own_independent_full_envelope_reference_audit_and_actual_static_PASS_required=True,
        actual_collision_tracking_stop_and_resource_guards_NOT_disabled=True,
        common_actual_goal_position_m=.030,common_actual_goal_heading_deg=2.,
        common_actual_stopped_speed_m_s=.005,common_fixed_stationary_hold_s=5.,
        actual_Pause_and_original_all_joint_gains_targets_restoration_required=True,
        history_ablation=dict(tasks=['H2','H3'],same_main_draws=True,
            sole_prediction_change='project front/rear steering and u/v/r to current-input steady state at each primitive entry',
            coefficient_seed_beam_speed_constraints_controller_environment_limits_all_identical=True,
            no_repeated_zero_velocity_reset_confound=True),
        envelope_ablation=dict(tasks=['H2','H3'],same_draws=True,
            point_reference_and_transition_endpoint_only_OFFLINE_comparisons=True,
            independently_audit_full_sampled_envelope=True,known_unsafe_paths_NOT_physically_executed=True),
        force_load=dict(record_all_rejected_and_accepted_candidate_activity=True,
            narrow_causal_and_safety_claims_if_inactive=True,no_artificially_lowered_limits=True,
            no_extra_high_speed_or_low_friction_trials_required=True),
        statistics=dict(all_declared_planning_failures_and_actual_noncompletions_in_denominator=True,
            completed_mission_RMS_with_explicit_valid_n=True,partial_RMS_not_ranked_as_full_tasks=True,
            sample_SD_ddof=1,SD_not_fabricated_for_n_under2=True,
            descriptive_small_sample_NOT_clinical_or_powered_confirmatory_reliability=True),
        treatment_frozen_before_new_outcomes=True,no_retuning_or_successful_retry_replacement_within_block=True,
        no_P_PASS_or_metric_dominance_guarantee=True,no_success_only_final_block_selection=True,
        original120_main60_joint_and_all_development_FAIL_HOLD_records_preserved=True,
        actual_static_trials=0,actual_route_trials=0,automatic_dispatch=False,
        manuscript_edits=False,real_robot_trials=False,sensor_Nav2_or_CICS_runtime_work=False,
        real_platform_photo_deferred_by_author=True)
