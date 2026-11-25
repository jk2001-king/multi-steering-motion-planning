"""Dedicated FINAL forward adapter admission and raw acceptance; no scene action.

Reuse the EXACT selected cap30 course/timing class. All prior HOLDs/FAILs,
the nominal spatial endpoint, exact mission goal and twelve joints remain.
The eight synthetic cases are reused DESIGN evidence, not independent
robustness, caster/contact validation or whole-H3 statistical evidence.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_post_second_reverse as previous
import hospital_h3b_final_forward_review_gate as review_gate
from hospital_h3b_timed_course_speed_profile import CappedTimedCourseForward, TRACKING_GAINS, PLANNED_SPEED_CAP_M_S, VERSION as DESIGN_VERSION
from hospital_h3b_timed_second_reverse import terminal_hold_checks, arrival_checks
from hospital_h3b_next_forward import body_velocity, motion_stalled
from hospital_h3b_policy import StopAlignGate, POLICY as ORIGINAL_POLICY
from hospital_h3_flush_crossing import ROBOT_ROOT, WHEELS, BAD_CONTACTS, classify_contact, evaluate_passage
from hospital_path_execution import direct_joint_targets, relative_pose, ramp, wrap
from hospital_timed_path_execution import SpatialLeg, execution_contract as timing_contract


VERSION = 'H3B_actual_second_reverse_PASS_FINAL_forward_cap30_physical_adapter_v1'
ROLE = 'ONE_FINAL_FORWARD_COMPONENT_NOT_FROZEN_WHOLE_H3_COMPARISON'
WORKSPACE, GATE = previous.WORKSPACE, previous.GATE
HERE = Path(__file__).resolve().parent
REVIEW_DIR = previous.BASE/'review_r01'
MODEL_REVIEW = REVIEW_DIR/'final_forward_review.json'
LIVE = REVIEW_DIR/'live_post_second_reverse.json'
COHORT = GATE/'hospital_h3b_final_forward_pilot/cohort_01'
ADMISSION = COHORT/'admission/executor_admission.json'
OUTPUT = COHORT/'forward'
POLICY = dict(ORIGINAL_POLICY, version=VERSION, forward_planned_speed_limit_m_s=.030)
sha, verify_files, read_trace = previous.sha, previous.verify_files, previous.read_trace
validate_live = review_gate.validate_live


def execution_contract(length):
    return dict(version=VERSION, selected_design_version=DESIGN_VERSION,
        fixed_mode='coordinated_counterphase', fixed_direction=1, reference_length_m=length,
        planned_speed_cap_m_s=.030, physical_speed_cap_m_s=.050, tracking_gains=TRACKING_GAINS,
        shared_timing_prototype_contract=timing_contract(),
        existing_course_equation_with_NEW_timing_composition_disclosed=True,
        nominal_profile_slope_NOT_a_certificate_of_feedback_command_slew=True,
        maximum_measured_forward_distance_m=length+.160,
        maximum_direction_changes=4, previous_reverse_total_limit_m=1.,
        controller_internal_goal_is_frozen_nominal_endpoint=True,
        exact_original_mission_goal_is_physical_acceptance_target=True,
        commanded_steering_zero_AND_actual_neutral_required_before_ONE_hold=True,
        maximum_same_direction_terminal_reapproaches=1, automatic_retry_or_next_leg=False,
        CICS_runtime_technical_integration=False, synthetic_design_NOT_independent_validation=True)


def load_profile():
    info, audit, *_ = previous.inspect_second_reverse()
    report = review_gate.load_review(REVIEW_DIR, info)
    selected = [c for c in report['cases'] if c['design_variant'] == review_gate.VARIANTS[-1]]
    expected_timing = CappedTimedCourseForward(info['final_leg'], 0.).timing.report()
    if (report['all_selected_final_forward_design_cases_pass'] is not True
            or len(selected) != 8 or any(not all(c['checks'].values())
                or c.get('composition_version') != DESIGN_VERSION
                or c.get('nominal_timing') != expected_timing
                or c.get('exact_command_neutral_before_hold_modelled') is not True
                or c.get('caster_stiction_captured_hold_and_measured_handoff_NOT_modelled') is not True for c in selected)):
        raise ValueError('All EIGHT exact cap30/neutral-handoff MODEL design cases must pass')
    live = json.loads(LIVE.read_bytes()); verify_files(live['input_sha256'])
    expected = dict(status='ACTUAL_SECOND_REVERSE_PASS_CONTINUITY_FINAL_FORWARD_ADAPTER_PENDING_NO_MOTION',
        second_reverse_result=info['second_reverse_result'], second_reverse_result_sha256=info['second_reverse_result_sha256'],
        offline_review=str(MODEL_REVIEW), offline_review_sha256=sha(MODEL_REVIEW),
        all_selected_final_forward_design_cases_pass=True, selected_planned_speed_cap_m_s=.030,
        original_mission_goal_xyyaw=info['mission_goal_xyyaw'], physical_state_reset=False,
        new_PhysX_motion_executed=False, automatic_motion_authorized=False,
        final_forward_executor_admitted=False, root_session_and_timeline_unchanged=True, CICS_input_used=False)
    if any(live.get(k) != v for k, v in expected.items()):
        raise ValueError('Exact actual SECOND-reverse PAUSED continuity receipt required; no substitute scene')
    script = _ijcas_path(WORKSPACE, 'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_second_reverse.py')
    verify_files({script: live['script_sha256']})
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    leg = copy.deepcopy(info['final_leg'])
    if leg['mode'] != 'coordinated_counterphase' or leg['direction'] != 1 or leg['original_suffix_leg_index'] != 3:
        raise ValueError('Only the complete frozen FINAL forward may be connected')
    info.update(leg=leg, next_cusp_local_xyyaw=leg['next_cusp_local_xyyaw'],
        exact_mission_goal_local_xyyaw=leg['exact_mission_goal_local_xyyaw'], reference_length_m=leg['reference_length_m'],
        preparation_record=info['second_reverse_record'], raw_second_reverse_audit=audit,
        model_review=str(MODEL_REVIEW), model_review_sha256=sha(MODEL_REVIEW),
        live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
        intervention_contract=execution_contract(leg['reference_length_m']),
        input_sha256={**info['input_sha256'], **report['input_sha256'], **live['input_sha256'],
            str(MODEL_REVIEW):sha(MODEL_REVIEW), str(LIVE):sha(LIVE),
            **{c['trace_file']: c['trace_sha256'] for c in report['cases']}})
    return info


def _evaluate_core(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events):
    common = evaluate_passage(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count)
    excluded = {'all_six_wheels_started_east_and_finished_west', 'ordinary_positive_ramp_within_30mm_s', 'passage_and_hold_completed'}
    checks = {k: v for k, v in common['checks'].items() if k not in excluded}
    maximum_slew = 0.
    for a, b in zip(rows, rows[1:]):
        dt = b['command_time_s'] - a['command_time_s']
        if dt < 0: raise ValueError('Command clock went backwards')
        for axle in ('front', 'rear'):
            change = abs(b['command_' + axle + '_rad'] - a['command_' + axle + '_rad'])
            if dt == 0 and change > 1e-12: raise ValueError('Steering changed with no elapsed command time')
            maximum_slew = max(maximum_slew, change / dt if dt else 0.)
    gate = gate_events[0] if len(gate_events) == 1 else {}
    first_hold = next((r for r in rows if r['phase'] == 'terminal_hold'), None)
    entered_at_goal = first_hold is not None and all(arrival_checks(
        [first_hold[k] for k in ('x_m', 'y_m', 'yaw_rad')], goal,
        max(first_hold['pose_filtered_speed_m_s'], math.hypot(first_hold['vx_physx_m_s'], first_hold['vy_physx_m_s'])),
        (first_hold['front_rad'], first_hold['rear_rad'])).values())
    reapproaches = [(a, b) for a, b in zip(rows, rows[1:])
                   if a['phase'] == 'forward_braking' and b['phase'] == 'forward_motion']
    maximum_acceleration = 0.
    for a, b in zip(rows, rows[1:]):
        dt = b['command_time_s']-a['command_time_s']
        change = abs(b['command_speed_m_s']-a['command_speed_m_s'])
        if dt == 0. and change > 1e-12: raise ValueError('Speed command changed without elapsed command time')
        maximum_acceleration = max(maximum_acceleration, change/dt if dt else 0.)
    checks.update(final_forward_and_single_hold_completed=True,
        measured_terminal_arrival_before_one_fixed_hold=entered_at_goal,
        at_most_one_same_direction_terminal_reapproach=len(reapproaches) <= 1,
        reapproach_entry_command_zero_and_measured_neutral=all(
            abs(b['command_speed_m_s']) < 1e-7
            and max(b['pose_filtered_speed_m_s'], math.hypot(b['vx_physx_m_s'], b['vy_physx_m_s'])) <= .005
            and max(abs(b[a+'_rad']) for a in ('front', 'rear')) <= math.radians(2.5)
            for _, b in reapproaches),
        command_acceleration_under_existing_20mm_s2=maximum_acceleration <= .020+1e-7,
        one_measured_stop_hold_neutral_release_before_final_forward=len(gate_events) == 1
            and gate.get('direction') == 1 and gate.get('captured_hold_acknowledged') is True
            and gate.get('hold_release_acknowledged') is True and gate.get('measured_speed_m_s', 1.) <= .005
            and gate.get('steering_error_deg', 180.) <= 2.5,
        only_counterphase_forward_commands_with_planned_cap30=all(r['command_mode'] == 'coordinated_counterphase' and -1e-9 <= r['command_speed_m_s'] <= .030+1e-9 for r in rows),
        measured_forward_progress_over_50mm=forward_episode_progress(rows) > .050,
        held_commands_zero=all(abs(r['command_speed_m_s']) < 1e-7 and max(abs(r['command_' + a + '_rad']) for a in ('front', 'rear')) < 1e-7 for r in rows if r['position_hold_enabled']),
        held_actual_steering_neutral=all(max(abs(r[a + '_rad']) for a in ('front', 'rear')) <= math.radians(2.5) for r in rows if r['position_hold_enabled']),
        final_hold_actually_uses_captured_position_targets=first_hold is not None
            and all(r['position_hold_enabled'] for r in rows if r['phase']=='terminal_hold'),
        command_slew_under_8deg_s=maximum_slew <= math.radians(8.) + 1e-5,
        existing_curved_reference_bounds=all(r['reference_position_error_m'] <= .080 and abs(r['reference_heading_error_deg']) <= 10. for r in rows))
    return dict(checks={k: bool(v) for k, v in checks.items()}, verdict='PASS' if all(checks.values()) else 'FAIL',
        **{k: common[k] for k in ('terminal_position_error_m', 'terminal_heading_error_deg', 'final_world_xyyaw', 'fixed_terminal_hold_analysis', 'final_force_weight_ratio')},
        maximum_observed_command_slew_deg_s=math.degrees(maximum_slew),
        final_forward_component_verified=all(checks.values()),
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        Nav2_executed=False, CICS_input_used=False, use_for_final_statistics=False)


def replay_commands(info, rows, gate_events):
    """Reproduce exact high-level, steering and applied wheel commands."""
    if len(gate_events) != 1 or not rows:
        raise ValueError('One measured release and complete raw rows required')
    controller = Progress(info, gate_events[0]['time_s'])
    command_error = steering_error = wheel_error = 0.
    ready_count = 0
    for before, row in zip(rows, rows[1:]):
        first_hold = row['phase'] == 'terminal_hold' and before['phase'] != 'terminal_hold'
        if row['phase'] not in ('forward_motion', 'forward_braking') and not first_hold:
            continue
        pose = [before[k] for k in ('x_m', 'y_m', 'yaw_rad')]
        if not controller.finished:
            out = controller.step(row['command_time_s'], relative_pose(pose, info['mission_start_xyyaw']),
                max(before['pose_filtered_speed_m_s'], math.hypot(before['vx_physx_m_s'], before['vy_physx_m_s'])),
                (before['front_rad'], before['rear_rad']),
                body_velocity((before['vx_physx_m_s'],before['vy_physx_m_s']), before['yaw_rad']))
        else:
            # The controller's arrival has already been acknowledged. Only
            # zero commands may drain the inherited low-level steering slew.
            out = dict(speed_m_s=0., curvature_1_m=0., ready=False)
        command_error = max(command_error, abs(out['speed_m_s'] - row['command_speed_m_s']),
            abs(out['curvature_1_m'] - row['requested_curvature_1_m']))
        angles, wheels = direct_joint_targets(out['speed_m_s'], 'coordinated_counterphase', out['curvature_1_m'],
            (before['command_front_rad'], before['command_rear_rad']),
            row['command_time_s'] - before['command_time_s'], (before['front_rad'], before['rear_rad']))
        steering_error = max(steering_error, *(abs(angles[i] - row['command_'+a+'_rad']) for i, a in enumerate(('front', 'rear'))))
        wheel_error = max(wheel_error, *(abs(wheels[i] - row['drive_'+a+'_actual_velocity_target_rad_s']) for i, a in enumerate(('front', 'rear'))))
        ready_count += int(out['ready'])
    return dict(maximum_controller_command_error=command_error, maximum_steering_target_error_rad=steering_error,
        maximum_applied_wheel_target_error_rad_s=wheel_error, terminal_ready_count=ready_count,
        exact_frozen_controller_and_mapping_reproduced=command_error <= 1e-10 and steering_error <= 1e-10
            and wheel_error <= 1e-6 and ready_count == 1)


def Reference(info):
    return SpatialLeg(info['leg'])


def Progress(info, started):
    return CappedTimedCourseForward(info['leg'], started)


def validate_motor_baseline(info, actual):
    expected = info['second_reverse_record']['runtime_motor_baseline']
    if (actual['joint_names'] != expected['joint_names']
            or any(not np.isfinite(actual[k]).all() for k in ('stiffness','damping','effort_limits'))
            or not np.allclose(actual['stiffness'], expected['stiffness'], rtol=1e-6, atol=1e-6)
            or any(not np.array_equal(actual[k], expected[k]) for k in ('damping','effort_limits'))):
        raise ValueError('Actual motor/passive stiffness, damping and torque limits must match prior PASS')


def controller_and_adapter_contract():
    entries = _ijcas_path(WORKSPACE, 'isaac/scripts')
    old = entries/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_pilot.py'
    new = entries/'isaac_run_ijcas_hospital_h3b_final_forward_pilot.py'
    functions = [{n.name:n for n in ast.walk(ast.parse(p.read_text()))
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for p in (old,new)]
    unchanged = ('numeric','physics_step','contacts','angles','verify_gains','measure','set_hold')
    for name in unchanged:
        if ast.dump(functions[0][name], include_attributes=False) != ast.dump(functions[1][name], include_attributes=False):
            raise ValueError('Low-level measurement/hold baseline changed: '+name)
    # Only direction, mode, stricter positive speed bound and explanatory
    # message differ; target mapping, slew and action calls are identical.
    expected = ast.get_source_segment(old.read_text(), functions[0]['command'])
    expected = expected.replace('not -.050-1e-9<=speed<=1e-9','not -1e-9<=speed<=.030+1e-9')
    expected = expected.replace("'front_only'", "'coordinated_counterphase'")
    expected = expected.replace('Only bounded first reverse/zero hold command permitted',
        'Only bounded cap30 FINAL forward/zero hold command permitted')
    expected = ast.parse(expected).body[0]
    if ast.dump(expected, include_attributes=False) != ast.dump(functions[1]['command'], include_attributes=False):
        raise ValueError('Undisclosed change in low-level FINAL-forward command mapping')
    sources = [old,new,Path(__file__), HERE/'hospital_h3b_final_forward_observation.py',
        HERE/'prepare_hospital_h3b_final_forward.py', HERE/'hospital_h3b_final_forward_cap30_model.py',
        HERE/'hospital_h3b_timed_course_speed_profile.py', HERE/'hospital_h3b_timed_course_forward.py',
        HERE/'hospital_timed_path_execution.py', HERE/'hospital_h3b_next_forward.py',
        HERE/'hospital_h3b_course_tracking.py', HERE/'hospital_path_execution.py', HERE/'hospital_h3b_policy.py',
        HERE/'hospital_drive_hold.py', HERE/'hospital_private_loader.py',
        entries/'isaac_run_ijcas_hospital_h3b_final_forward_instrumented.py']
    return dict(unchanged_AST_functions=list(unchanged),
        low_level_command_AST_equal_except_disclosed_forward_mode_cap30_and_message=True,
        controller_class_used_by_selected_model_and_runtime='hospital_h3b_timed_course_speed_profile.CappedTimedCourseForward',
        existing_course_step_and_1p5_2_gains_unchanged=True,
        actual_body_velocity_passed_to_course_feedback=True,
        exact_original_goal_checked_before_ONE_hold_and_final_tail=True,
        no_reverse_reset_CICS_or_automatic_next_leg=True,
        physical_contact_caster_mass_gains_and_safety_limits_unchanged=True,
        input_sha256={str(p):sha(p) for p in sources})


def admission_payload(info):
    contract = controller_and_adapter_contract()
    return dict(version=VERSION, status='FINAL_FORWARD_CAP30_ADAPTER_PREPARED_RUNTIME_CONTINUITY_REQUIRED_NO_MOTION',
        second_reverse_result=info['second_reverse_result'], second_reverse_result_sha256=info['second_reverse_result_sha256'],
        model_review=info['model_review'], model_review_sha256=info['model_review_sha256'],
        live_receipt=info['live_receipt'], live_receipt_sha256=info['live_receipt_sha256'],
        actual_start_world_xyyaw=info['initial_world_xyyaw'], retained_all_joint_positions_rad=info['retained_all_joint_positions_rad'],
        fixed_final_forward=info['leg'], original_mission_goal_xyyaw=info['mission_goal_xyyaw'],
        strict_map_digest=info['strict_map_digest'], intervention_contract=info['intervention_contract'],
        controller_contract=contract, input_sha256={**info['input_sha256'], **contract['input_sha256']},
        synthetic_design_cases_reused_NOT_independent_validation=True,
        PhysX_executed=False, automatic_motion_authorized=False, CICS_input_used=False,
        full_H3B_turning_path_verified=False, use_for_final_statistics=False)


def load_admitted_profile():
    info = load_profile()
    report = json.loads(ADMISSION.read_bytes()); verify_files(report['input_sha256'])
    if report != admission_payload(info):
        raise ValueError('Exact prepared FINAL-forward adapter contract required; no overwrite or bypass')
    info.update(admission=str(ADMISSION), admission_sha256=sha(ADMISSION),
        input_sha256={**report['input_sha256'], str(ADMISSION):sha(ADMISSION)})
    return info


def forward_episode_progress(rows):
    moving = [r for r in rows if r['phase'] in ('forward_motion','forward_braking')
        and not r['position_hold_enabled'] and r['command_speed_m_s'] > .003]
    return moving[-1]['reference_progress_m']-moving[0]['reference_progress_m'] if moving else 0.


def evaluate_component(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events, info):
    numeric = ('time_s','command_time_s','command_speed_m_s','command_front_rad','command_rear_rad',
        'front_rad','rear_rad','reference_progress_m','reference_position_error_m','reference_heading_error_deg',
        'x_m','y_m','yaw_rad','vx_physx_m_s','vy_physx_m_s','pose_filtered_speed_m_s')
    if not rows or any(not math.isfinite(r[k]) for r in rows for k in numeric):
        raise ValueError('Complete finite FINAL-forward rows required')
    if not np.allclose(goal, info['mission_goal_xyyaw'], rtol=0., atol=1e-12):
        raise ValueError('Acceptance goal must be the EXACT original mission goal')
    result = _evaluate_core(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events)
    replay = replay_commands(info, rows, gate_events)
    initial = info['initial_world_xyyaw']
    distance = math.hypot(rows[0]['x_m']-initial[0], rows[0]['y_m']-initial[1])
    distance += sum(math.hypot(b['x_m']-a['x_m'], b['y_m']-a['y_m']) for a,b in zip(rows,rows[1:]))
    first_hold = next((r for r in rows if r['phase']=='terminal_hold'),None)
    phases = [r['phase'] for r in rows]
    allowed = {'pre_hold','entry_gate','forward_motion','forward_braking','terminal_hold'}
    hold_index = next((i for i,r in enumerate(rows) if r['phase']=='terminal_hold'),len(rows))
    result['checks'].update(
        exact_screened_controller_and_applied_targets_reproduced=replay['exact_frozen_controller_and_mapping_reproduced'],
        full_measured_forward_distance_under_reference_plus_160mm=distance <= info['reference_length_m']+.160,
        cumulative_measured_reverse_UNCHANGED_under_1m=info['previous_measured_reverse_total_m'] <= 1.,
        cumulative_direction_changes_under_four=info['completed_direction_changes']+len(gate_events) <= POLICY['maximum_direction_changes'],
        reference_progress_reached_original_final_end_neighbourhood=rows[-1]['reference_progress_m'] >= info['reference_length_m']-.050,
        only_forward_episode_then_ONE_contiguous_hold=all(p in allowed for p in phases)
            and 'forward_motion' in phases and hold_index<len(rows)
            and all(p=='terminal_hold' for p in phases[hold_index:])
            and all(p in ('forward_motion','forward_braking') for p in phases[next((i for i,p in enumerate(phases) if p=='forward_motion'),len(rows)):hold_index]),
        exact_command_neutral_before_captured_hold=first_hold is not None and all(terminal_hold_checks(
            [first_hold[k] for k in ('x_m','y_m','yaw_rad')], goal,
            max(first_hold['pose_filtered_speed_m_s'], math.hypot(first_hold['vx_physx_m_s'],first_hold['vy_physx_m_s'])),
            (first_hold['front_rad'],first_hold['rear_rad']),
            (first_hold['command_front_rad'],first_hold['command_rear_rad'])).values()))
    passed = all(result['checks'].values())
    result.update(command_reproduction=replay, measured_center_distance_m=distance,
        cumulative_measured_reverse_m=info['previous_measured_reverse_total_m'],
        cumulative_direction_changes=info['completed_direction_changes']+len(gate_events),
        final_forward_component_verified=passed, original_final_goal_reached_in_segmented_development=passed,
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        CICS_input_used=False, Nav2_executed=False, use_for_final_statistics=False,
        verdict='PASS' if passed else 'FAIL')
    return result
