"""Close out cohort_03's five SAME-input response observations, without motion.

Retain the original raw evaluators and sampled actual-envelope audit. Known
input reproductions, even with continuous physical handoffs, do not validate
an open-loop body model or admit a fresh autonomous whole-sequence replay.
"""

import json
from pathlib import Path

import hospital_h3b_final_response_observation as final
import hospital_h3b_post_final_forward as original
from hospital_h3_flush_support import load_setup
from hospital_h3_turn_planning import verify_sources


WORKSPACE, BASE = final.WORKSPACE, final.BASE
VERSION = 'H3B_cohort03_five_response_observations_closeout_v1'
STATUS = 'FIVE_SAME_INPUT_RESPONSES_AND_SEGMENTED_GOAL_AUDITED_WHOLE_MODEL_HOLD'
DIRECTORY = BASE/'response_completion/closeout_r01'
TARGET = DIRECTORY/'response_completion_audit.json'
LIVE = DIRECTORY/'live_final_response_goal.json'
SUMMARY = BASE/'measurement_chain/chain_r04/five_component_response_review.json'
SCRIPT = WORKSPACE/'isaac/scripts/isaac_review_ijcas_hospital_h3b_final_response_goal.py'
WHOLE_MODEL = final.GATE/'hospital_h3b_replanned_whole_sequence/review_r01/sequence_model_review.json'
ANALYSES = (
    BASE/'analyses/analysis_r02/actual_prefix_response_review.json',
    BASE/'reverse_analyses/analysis_r02/actual_reverse_response_review.json',
    BASE/'next_forward_analyses/analysis_r01/actual_next_forward_response_review.json',
    BASE/'second_reverse_analyses/analysis_r01/actual_second_reverse_response_review.json',
    BASE/'final_forward_analyses/analysis_r01/actual_final_forward_response_review.json',
)
ROLES = ('first_forward', 'first_reverse', 'next_forward', 'second_reverse', 'final_forward')
IDENTITIES = ('prefix', 'reverse', 'next_forward', 'second_reverse', 'final_forward')
sha, verify_files, read_trace = final.sha, final.verify_files, final.read_trace
joints_from_last, audit_chain = original.joints_from_last, original.audit_chain
validate_tensor = final.validate_tensor


def inspect_final_response():
    """Re-evaluate the authoritative actual FINAL result, never relabel it."""
    info = final.load_admitted_profile()
    record = final.exact_history(final.RESULT, final.REFERENCE)
    expected = dict(version=final.VERSION, trial_kind=final.ROLE, verdict='PASS',
        PhysX_executed=True, original_motor_gains_restored=True,
        final_forward_component_verified=True, original_final_goal_reached_in_segmented_development=True,
        body_pose_reset_during_run=False, caster_state_reset_during_run=False, material_modified_during_run=False,
        source_USD_saved=False, CICS_input_used=False, Nav2_executed=False, use_for_final_statistics=False,
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        initial_saved_world_xyyaw=info['initial_world_xyyaw'],
        expected_retained_joint_positions_rad=info['retained_all_joint_positions_rad'],
        prior_second_reverse_result=info['second_reverse_result'],
        prior_second_reverse_result_sha256=info['second_reverse_result_sha256'],
        target_world_xyyaw=info['mission_goal_xyyaw'], mission_goal_world_xyyaw_UNCHANGED=info['mission_goal_xyyaw'],
        target_is_EXACT_original_final_mission_goal=True, strict_map_digest=info['strict_map_digest'],
        intervention_contract=info['intervention_contract'], selected_model_design_version=final.DESIGN_VERSION,
        planned_speed_cap_m_s=.030, physical_speed_cap_m_s=.050,
        runtime_motor_baseline_matches_actual_prior_PASS=True,
        same_original_final_forward_controller_AND_evaluator=True,
        passive_six_wheel_response_observation_only=True, new_body_or_steering_model_deployed=False)
    if any(record.get(k) != v for k, v in expected.items()) or record.get('error') is not None:
        raise ValueError('Exact original FINAL goal, SAME executor and retained actual history required')
    final.validate_motor_baseline(info, record['runtime_motor_baseline'])
    rows = read_trace(final.RESULT.with_suffix('.csv'))
    _, manifest, geometry, inventory = load_setup(Path(record['geometry_manifest']).parent, WORKSPACE)
    actual = final.evaluate_component(rows, info['mission_goal_xyyaw'],
        read_trace(Path(record['static_reference']).with_suffix('.csv')),
        record['physics_preparation_snapshot']['simulation_mass_kg'],
        inventory['explicitly_suppressed_assets'][0]['collision']['bound'], record['wheel_conservative_sphere_radii_m'],
        record['contact_report_counts_by_category'], record['contact_callback_errors'],
        record['fixed_hold_attempt_count'], record['direction_gate_events'], info)
    if (actual['verdict'] != 'PASS' or any(record['checks'].get(k) != v for k, v in actual['checks'].items())
            or actual['command_reproduction'] != record['command_reproduction']
            or abs(actual['measured_center_distance_m']-record['measured_center_distance_m']) > 1e-10
            or any(record[k] != actual[k] for k in ('cumulative_direction_changes', 'cumulative_measured_reverse_m'))):
        raise ValueError('Original raw FINAL commands, fixed hold, resources or terminal verdict did not reproduce')
    observer_path = final.OBSERVER_OUTPUT/'instrumentation.json'
    observer = json.loads(observer_path.read_bytes())
    verify_sources(observer); verify_files(observer['reference_input_sha256'])
    files = {str(observer_path.parent/name): digest for name, digest in observer['saved_file_sha256'].items()}
    verify_files(files)
    if (observer['original_controller_result_sha256'] != sha(final.RESULT) or observer['telemetry_errors']
            or observer.get('control_or_physics_parameters_modified_by_observer') is not False
            or observer['original_controller_source_sha256'] != record['source_sha256']):
        raise ValueError('SAME final observer/primary source lineage and complete passive data required')
    audit = dict(original_verdict_unchanged='PASS', primary_check_count=len(actual['checks']),
        saved_check_count=len(record['checks']), command_reproduction=actual['command_reproduction'],
        exact_original_goal_world_xyyaw=info['mission_goal_xyyaw'],
        terminal_position_error_m=actual['terminal_position_error_m'],
        terminal_heading_error_deg=actual['terminal_heading_error_deg'],
        pose_regression_speed_m_s=actual['fixed_terminal_hold_analysis']['pose_regression_speed_m_s'],
        actual_fixed_hold_attempt_count=record['fixed_hold_attempt_count'],
        cumulative_measured_reverse_m=actual['cumulative_measured_reverse_m'],
        cumulative_direction_changes=actual['cumulative_direction_changes'])
    info.update(final_forward_result=str(final.RESULT), final_forward_result_sha256=sha(final.RESULT),
        final_forward_record=record, initial_world_xyyaw=actual['final_world_xyyaw'],
        expected_saved_last_state=rows[-1], retained_all_joint_positions_rad=joints_from_last(record),
        input_sha256={**info['input_sha256'], **record['frozen_input_sha256'],
            str(final.RESULT): sha(final.RESULT), str(final.RESULT.with_suffix('.csv')): sha(final.RESULT.with_suffix('.csv')),
            str(observer_path): sha(observer_path), **files})
    return info, audit, manifest, geometry, inventory


def response_synthesis(info, chain):
    """Join authenticated analyses; accounting differences are not causal fits."""
    inputs = dict(info['input_sha256']); components = []
    for role, identity, path, segment in zip(ROLES, IDENTITIES, ANALYSES, chain['segments']):
        report = json.loads(path.read_bytes()); verify_sources(report); verify_files(report['input_sha256'])
        verify_files({c['trace_file']: c['trace_sha256'] for c in report['body_replays']})
        coverage = report['coverage']
        result_path = Path(segment['result']); trace_path = result_path.with_suffix('.csv')
        if (report['original_actual_verdict_unchanged'] != 'PASS' or report['original_error_unchanged'] is not None
                or report['original_failed_checks_unchanged'] or report['telemetry_errors_unchanged']
                or report.get('primary_CSV_byte_identical_to_previous_actual_'+identity) is not True
                or report['input_sha256'].get(str(result_path)) != sha(result_path)
                or report['input_sha256'].get(str(trace_path)) != sha(trace_path)
                or report['whole_H3_drive_admission'] is not False
                or not all(coverage['each_wheel_has_friction_anchor_observations'])
                or len(coverage['each_wheel_has_friction_anchor_observations']) != 6
                or not all(coverage[k] for k in ('telemetry_covers_controller_start', 'telemetry_covers_controller_end',
                    'raw_contact_points_complete', 'six_wheel_transforms_and_COM_and_origin_velocities_recorded'))):
            raise ValueError('Complete same-input raw/analysis/coverage lineage required: '+role)
        cases = report['body_replays']
        if {c['variant'] for c in cases} != {'frozen_P1_steering', 'recorded_steering_interval_mean'} or len(cases) != 2:
            raise ValueError('Both unchanged conditional body replay variants required: '+role)
        for case in cases:
            if (case['body_state_initialization_count'] != 1 or case['body_state_corrections_after_initialization'] != 0
                    or case['fitted_speed_scale_used'] is not False or case['speed_scale_exactly_one'] is not True
                    or case['actual_drive_admission'] is not False
                    or case['conditional_reused_data_NOT_independent_validation'] is not True
                    or case['observed_future_velocities_are_metrics_ONLY_no_body_state_injection'] is not True):
                raise ValueError('No parameter fit, future-state injection or model admission permitted: '+role)
        wheels = report['wheels']
        if len(wheels) != 6 or len({w['wheel'] for w in wheels}) != 6:
            raise ValueError('All six wheels required; no selective wheel accounting')
        components.append(dict(component=role, result=str(result_path), result_sha256=sha(result_path),
            response_analysis=str(path), response_analysis_sha256=sha(path),
            original_motion_verdict='PASS', actual_terminal_position_error_mm=1000*segment['measured_terminal_position_error_m'],
            actual_terminal_heading_error_deg=segment['measured_terminal_heading_error_deg'],
            primary_CSV_byte_identical_to_previous_actual_component=True, telemetry_errors=[], coverage=coverage,
            conditional_body_replays=[dict(variant=c['variant'],
                final_position_gap_mm=c['metrics']['position_error_mm']['final_signed'],
                final_heading_gap_deg=c['metrics']['heading_error_deg']['final_signed'],
                active_u_RMSE_mm_s=c['active_command_velocity_errors']['u_error_mm_s']['rmse'],
                body_corrections_after_initialization=0) for c in cases],
            all_six_wheel_phase_accounting=[dict(wheel=w['wheel'], phases=w['phases']) for w in wheels],
            maximum_native_float32_anchor_sum_error_n=report['maximum_native_float32_anchor_sum_error_n'],
            maximum_COM_origin_velocity_conversion_error_m_s=report['maximum_COM_origin_velocity_conversion_error_m_s']))
        inputs.update(report['input_sha256']); inputs[str(path)] = sha(path)
        inputs.update({c['trace_file']: c['trace_sha256'] for c in cases})
    whole = json.loads(WHOLE_MODEL.read_bytes()); verify_sources(whole); verify_files(whole['input_sha256'])
    count = sum(c['error'] is None and bool(c['checks']) and all(c['checks'].values()) for c in whole['cases'])
    if (len(whole['cases']) != 8 or count != 2 or whole['whole_sequence_MODEL_PASS_count'] != count
            or whole['actual_drive_admission'] is not False or whole['automatic_motion_authorized'] is not False):
        raise ValueError('Original 2/8 whole-model PASS and six HOLDs must remain authoritative')
    inputs.update(whole['input_sha256']); inputs[str(WHOLE_MODEL)] = sha(WHOLE_MODEL)
    return dict(version=VERSION, status=STATUS, components=components,
        total_joint_body_six_wheel_samples=sum(c['coverage']['joint_sample_count'] for c in components),
        total_friction_samples=sum(c['coverage']['friction_sample_count'] for c in components),
        total_friction_anchors=sum(c['coverage']['friction_anchor_count'] for c in components),
        total_raw_contact_points=sum(c['coverage']['raw_contact_point_count'] for c in components),
        original_final_goal_xyyaw=info['mission_goal_xyyaw'],
        four_actual_saved_body_and_twelve_joint_handoffs_verified=True,
        actual_component_terminal_errors_NOT_open_loop_body_prediction_errors=True,
        prediction_gaps_are_full_released_pre_hold_conditional_replays_NOT_terminal_acceptance_errors=True,
        original_success_recipe_was_developed_segment_by_segment_and_replanned_from_measured_state=True,
        this_cohort_replays_known_successful_components_with_manual_reviews_NOT_a_fresh_single_orchestrator_trial=True,
        sampled_actual_envelope_check=dict(clearance_m=.010, maximum_translation_spacing_m=.005,
            maximum_heading_spacing_deg=.25, analytic_continuous_collision_proof=False),
        no_parameter_fit_or_new_control_or_physics_change=True,
        shared_previous_data_NOT_independent_predictive_validation=True,
        no_unique_caster_friction_motor_or_solver_cause_identified=True,
        no_scalar_speed_gain_or_radius_change_justified_by_this_accounting=True,
        whole_H3_original_model_PASS_count=count, whole_H3_original_model_case_count=8,
        all_six_original_whole_H3_HOLDs_preserved=True, whole_H3_drive_admission=False,
        independent_fresh_frozen_whole_sequence_repeats=0,
        current_observations_are_NOT_four_method_or_robustness_statistics=True,
        next_gate='READ_ONLY_ACTUAL_FINAL_STATE_CLOSEOUT_THEN_OFFLINE_BODY_RESPONSE_MODEL_REVIEW',
        no_sixth_leg_or_automatic_retry=True, new_PhysX_executed=False, CICS_input_used=False,
        use_for_final_statistics=False, input_sha256=inputs)


def build_closeout():
    info, audit, manifest, geometry, inventory = inspect_final_response()
    chain = audit_chain(info, manifest)  # Original sampled envelope and exact four-handoff code, unchanged.
    synthesis = response_synthesis(info, chain)
    sources = [Path(__file__), SCRIPT, WORKSPACE/'studies/IJCAS/scripts/audit_hospital_h3b_response_completion.py']
    inputs = {**synthesis['input_sha256'], **{str(p): sha(p) for p in sources}}
    verify_files(inputs)
    report = dict(version=VERSION, status=STATUS, final_forward_result=info['final_forward_result'],
        final_forward_result_sha256=info['final_forward_result_sha256'],
        final_actual_world_xyyaw=info['initial_world_xyyaw'], original_goal_world_xyyaw=info['mission_goal_xyyaw'],
        final_all_twelve_joint_positions_rad=info['retained_all_joint_positions_rad'],
        final_forward_raw_audit=audit, chain_audit=chain, response_synthesis=synthesis,
        input_sha256=inputs, source_sha256={p.name: sha(p) for p in sources},
        new_PhysX_motion_executed=False, automatic_motion_authorized=False,
        whole_H3_drive_admission=False, CICS_input_used=False, use_for_final_statistics=False)
    return info, audit, manifest, geometry, inventory, report


def load_closeout(directory):
    if Path(directory).resolve() != DIRECTORY.resolve():
        raise ValueError('Explicit cohort_03 closeout directory required; no alternate scene/receipt')
    report = json.loads(TARGET.read_bytes()); verify_sources(report); verify_files(report['input_sha256'])
    values = build_closeout()
    if report != values[-1] or json.loads(SUMMARY.read_bytes()) != report['response_synthesis']:
        raise ValueError('Raw chain, six-wheel response analyses or scope changed after offline closeout')
    return values
