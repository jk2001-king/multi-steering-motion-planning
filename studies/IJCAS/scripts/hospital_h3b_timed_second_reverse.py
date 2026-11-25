"""ONE frozen SECOND reverse from the retained actual next-forward PASS.

Use the EXACT shared timing class screened offline, not a new step law.
CICS is manuscript context only: no differential/micro-yaw/caster alignment
input or handoff is introduced. All previous PASS/FAIL/HOLD records remain.
"""

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_post_forward as previous
from hospital_timed_path_execution import TimedLegController, SpatialLeg, LegTiming, VERSION, execution_contract
from hospital_h3b_policy import StopAlignGate, POLICY as ORIGINAL_POLICY
from hospital_h3b_reverse_closedloop import arrival_checks, motion_stalled, reverse_episode_progress
from hospital_h3_flush_crossing import ROBOT_ROOT, WHEELS, BAD_CONTACTS, classify_contact, evaluate_passage
from hospital_path_execution import direct_joint_targets, relative_pose, ramp, wrap
from hospital_h3_turn_planning import verify_sources, model_bounds


WORKSPACE, GATE = previous.WORKSPACE, previous.GATE
HERE = Path(__file__).resolve().parent
MODEL_REVIEW = GATE/'hospital_h3b_shared_timed_executor/review_r02/timed_executor_review.json'
OLD_REVIEW = previous.BASE/'review_r01'
OLD_LIVE = OLD_REVIEW/'live_post_forward.json'
COHORT = GATE/'hospital_h3b_timed_second_reverse/cohort_01'
ADMISSION = COHORT/'admission/executor_admission.json'
ADAPTER_MODEL = COHORT/'admission/adapter_model_review.json'
LIVE = COHORT/'handoff/live_handoff.json'
OUTPUT = COHORT/'reverse'
ROLE = 'ONE_FROZEN_SECOND_REVERSE_WITH_SHARED_TIMING_NOT_FULL_H3_OR_FINAL_COMPARISON'
POLICY = dict(ORIGINAL_POLICY, version=VERSION, reverse_speed_limit_m_s=.050)
sha, verify_files, read_trace = previous.sha, previous.verify_files, previous.read_trace
validate_live = previous.forward.validate_live


def validate_model_report(info, report):
    verify_sources(report)
    verify_files(report['input_sha256'])
    expected = {(d, r, s) for d in previous.STEPS for r in previous.RESIDUALS for s in previous.SCALES}
    cases = report.get('cases', [])
    if (report.get('version') != VERSION or report.get('all_timed_executor_design_cases_pass') is not True
            or report.get('status') != 'COMMON_TIMING_MODEL_DESIGN_PASS_NO_PHYSICAL_ADAPTER_OR_DRIVE_AUTHORITY'
            or report.get('PhysX_executed') is not False or report.get('automatic_motion_authorized') is not False
            or report.get('runtime_adapter_prepared') is not False or report.get('CICS_input_used') is not False
            or report.get('original_forward_result') != info['forward_result']
            or report.get('original_forward_result_sha256') != info['forward_result_sha256']
            or report.get('exact_actual_initial_world_xyyaw') != info['initial_world_xyyaw']
            or report.get('retained_all_joint_positions_rad') != info['retained_all_joint_positions_rad']
            or report.get('original_mission_start_xyyaw') != info['mission_start_xyyaw']
            or report.get('original_mission_goal_xyyaw') != info['mission_goal_xyyaw']
            or report.get('strict_map_digest') != info['strict_map_digest']
            or report.get('fixed_second_reverse') != info['second_reverse']
            or report.get('executor_contract') != execution_contract()
            or report.get('old_reference_unchanged') is not True
            or report.get('new_suffix_spatial_replans_NOT_used') is not True
            or report.get('reused_cases_for_design_NOT_independent_validation') is not True
            or len(cases) != 8
            or {(c['dt_s'], c['residual_case'], c['SYNTHETIC_body_target_scale']) for c in cases} != expected
            or any(not c.get('checks') or not all(c['checks'].values())
                or c['model_contract'] != execution_contract() or c['model_parameters'] != model_bounds()
                or c['fixed_next_cusp_local_xyyaw'] != info['second_reverse']['next_cusp_local_xyyaw']
                or c.get('no_reverse_course_feedback_or_new_gains') is not True for c in cases)):
        raise ValueError('Exact complete EIGHT-case NEW shared-timing model PASS required; never omit a HOLD')
    verify_files({c['trace_file']: c['trace_sha256'] for c in cases})
    timing = LegTiming(SpatialLeg(info['second_reverse'])).report()
    if report['nominal_timing'] != timing or any(c['nominal_timing'] != timing for c in cases):
        raise ValueError('Frozen reference timing no longer reproduces')
    if info['completed_direction_changes'] + 1 > POLICY['maximum_direction_changes']:
        raise ValueError('Original cumulative direction-change resource exhausted')


def load_profile():
    info, audit, *_ = previous.inspect_forward()
    old = previous.load_review(OLD_REVIEW)
    live = json.loads(OLD_LIVE.read_bytes())
    verify_files(live['input_sha256'])
    if (live.get('status') != 'ACTUAL_NEXT_FORWARD_PASS_PAUSED_CONTINUITY_RECORDED_NOT_DRIVE_AUTHORITY'
            or live.get('forward_result_sha256') != info['forward_result_sha256']
            or live.get('physical_state_reset') is not False or live.get('new_PhysX_motion_executed') is not False
            or live.get('automatic_motion_authorized') is not False
            or live.get('root_session_and_timeline_unchanged') is not True
            or live.get('offline_review_sha256') != sha(OLD_REVIEW/'post_forward_review.json')
            or live.get('all_second_reverse_model_cases_pass') != old['all_second_reverse_model_cases_pass']):
        raise ValueError('Retained actual next-forward PASS and old truthful HOLD continuity required')
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    verify_files({WORKSPACE/'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_forward.py': live['script_sha256']})
    report = json.loads(MODEL_REVIEW.read_bytes())
    validate_model_report(info, report)
    leg = copy.deepcopy(info['second_reverse'])
    if leg['mode'] != 'front_only' or leg['direction'] != -1:
        raise ValueError('Only the complete frozen SECOND front-only reverse may be connected')
    info.update(leg=leg, next_cusp_local_xyyaw=leg['next_cusp_local_xyyaw'],
        reference_length_m=leg['reference_length_m'], raw_forward_audit=audit,
        preparation_record=info['forward_record'],
        model_review=str(MODEL_REVIEW), model_review_sha256=sha(MODEL_REVIEW),
        old_second_reverse_HOLD_preserved=True, intervention_contract=execution_contract(),
        input_sha256={**info['input_sha256'], **old['input_sha256'], **live['input_sha256'], **report['input_sha256'],
            str(OLD_LIVE): sha(OLD_LIVE), str(OLD_REVIEW/'post_forward_review.json'): sha(OLD_REVIEW/'post_forward_review.json'),
            str(MODEL_REVIEW): sha(MODEL_REVIEW), **{c['trace_file']: c['trace_sha256'] for c in report['cases']}})
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
                   if a['phase'] == 'reverse_braking' and b['phase'] == 'reverse_motion']
    maximum_acceleration = 0.
    for a, b in zip(rows, rows[1:]):
        dt = b['command_time_s']-a['command_time_s']
        change = abs(b['command_speed_m_s']-a['command_speed_m_s'])
        if dt == 0. and change > 1e-12: raise ValueError('Speed command changed without elapsed command time')
        maximum_acceleration = max(maximum_acceleration, change/dt if dt else 0.)
    checks.update(second_reverse_and_single_hold_completed=True,
        measured_terminal_arrival_before_one_fixed_hold=entered_at_goal,
        at_most_one_same_direction_terminal_reapproach=len(reapproaches) <= 1,
        reapproach_entry_command_zero_and_measured_neutral=all(
            abs(b['command_speed_m_s']) < 1e-7
            and max(b['pose_filtered_speed_m_s'], math.hypot(b['vx_physx_m_s'], b['vy_physx_m_s'])) <= .005
            and max(abs(b[a+'_rad']) for a in ('front', 'rear')) <= math.radians(2.5)
            for _, b in reapproaches),
        command_acceleration_under_existing_20mm_s2=maximum_acceleration <= .020+1e-7,
        one_measured_stop_hold_neutral_release_before_reverse=len(gate_events) == 1
            and gate.get('direction') == -1 and gate.get('captured_hold_acknowledged') is True
            and gate.get('hold_release_acknowledged') is True and gate.get('measured_speed_m_s', 1.) <= .005
            and gate.get('steering_error_deg', 180.) <= 2.5,
        only_front_only_reverse_commands=all(r['command_mode'] == 'front_only' and -.050 - 1e-9 <= r['command_speed_m_s'] <= 1e-9 for r in rows),
        measured_reverse_progress_over_50mm=reverse_episode_progress(rows) > .050,
        measured_center_distance_under_original_0p5m=sum(math.hypot(b['x_m'] - a['x_m'], b['y_m'] - a['y_m']) for a, b in zip(rows, rows[1:])) <= .5,
        held_commands_zero=all(abs(r['command_speed_m_s']) < 1e-7 and max(abs(r['command_' + a + '_rad']) for a in ('front', 'rear')) < 1e-7 for r in rows if r['position_hold_enabled']),
        held_actual_steering_neutral=all(max(abs(r[a + '_rad']) for a in ('front', 'rear')) <= math.radians(2.5) for r in rows if r['position_hold_enabled']),
        command_slew_under_8deg_s=maximum_slew <= math.radians(8.) + 1e-5,
        existing_curved_reference_bounds=all(r['reference_position_error_m'] <= .080 and abs(r['reference_heading_error_deg']) <= 10. for r in rows))
    return dict(checks={k: bool(v) for k, v in checks.items()}, verdict='PASS' if all(checks.values()) else 'FAIL',
        **{k: common[k] for k in ('terminal_position_error_m', 'terminal_heading_error_deg', 'final_world_xyyaw', 'fixed_terminal_hold_analysis', 'final_force_weight_ratio')},
        maximum_observed_command_slew_deg_s=math.degrees(maximum_slew),
        second_reverse_component_verified=all(checks.values()), front_only_reverse_component_verified=all(checks.values()),
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        Nav2_executed=False, CICS_input_used=False, use_for_final_statistics=False)


def Reference(info):
    return SpatialLeg(info['leg'])


def Progress(info, started):
    return TimedLegController(info['leg'], started)


def terminal_hold_checks(pose, goal, measured_speed, actual_steering, commanded_steering):
    if len(commanded_steering) != 2 or not all(math.isfinite(v) for v in commanded_steering):
        raise ValueError('Finite paired steering commands required before captured hold')
    checks = arrival_checks(pose, goal, measured_speed, actual_steering)
    checks['commanded_steering_exactly_neutral'] = max(abs(v) for v in commanded_steering) <= 1e-7
    return {k: bool(v) for k, v in checks.items()}


def load_adapter_reviewed_profile():
    info = load_profile()
    report = json.loads(ADAPTER_MODEL.read_bytes())
    verify_sources(report);verify_files(report['input_sha256'])
    cases = report.get('cases', [])
    expected = {(d, r, s) for d in previous.STEPS for r in previous.RESIDUALS for s in previous.SCALES}
    if (report.get('version') != VERSION or report.get('all_adapter_model_cases_pass') is not True
            or report.get('PhysX_executed') is not False or report.get('automatic_motion_authorized') is not False
            or report.get('forward_result_sha256') != info['forward_result_sha256']
            or report.get('fixed_second_reverse') != info['leg']
            or report.get('actual_start_world_xyyaw') != info['initial_world_xyyaw']
            or report.get('retained_all_joint_positions_rad') != info['retained_all_joint_positions_rad']
            or report.get('controller_contract') != controller_and_adapter_contract()
            or len(cases) != 8 or {(c['dt_s'], c['residual_case'], c['SYNTHETIC_body_target_scale']) for c in cases} != expected
            or any(not all(c['checks'].values()) or c.get('exact_command_neutral_before_hold_modelled') is not True for c in cases)):
        raise ValueError('All EIGHT exact neutral-to-captured-hold ADAPTER model cases must pass')
    verify_files({c['trace_file']: c['trace_sha256'] for c in cases})
    info.update(adapter_model=str(ADAPTER_MODEL), adapter_model_sha256=sha(ADAPTER_MODEL),
        input_sha256={**info['input_sha256'], **report['input_sha256'], str(ADAPTER_MODEL): sha(ADAPTER_MODEL),
            **{c['trace_file']: c['trace_sha256'] for c in cases}})
    return info


def controller_and_adapter_contract():
    entries = WORKSPACE/'isaac/scripts'
    old = entries/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_pilot.py'
    new = entries/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_pilot.py'
    functions = [{n.name: n for n in ast.walk(ast.parse(p.read_text()))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in (old, new)]
    unchanged = ('numeric', 'physics_step', 'contacts', 'angles', 'verify_gains', 'measure', 'set_hold')
    for name in unchanged:
        if ast.dump(functions[0][name], include_attributes=False) != ast.dump(functions[1][name], include_attributes=False):
            raise ValueError('Shared low-level measurement/hold baseline changed: '+name)
    normalized = copy.deepcopy(functions[1]['command'])
    for node in ast.walk(normalized):
        if isinstance(node, ast.Constant) and node.value == 'front_only': node.value = 'rear_only'
    if ast.dump(normalized, include_attributes=False) != ast.dump(functions[0]['command'], include_attributes=False):
        raise ValueError('Only the disclosed front-only mode may differ in the low-level command adapter')
    sources = [old, new, Path(__file__), HERE/'hospital_timed_path_execution.py', HERE/'hospital_h3b_timed_executor_model.py',
        HERE/'hospital_h3b_timed_second_reverse_observation.py', HERE/'prepare_hospital_h3b_timed_second_reverse.py',
        HERE/'hospital_path_execution.py', HERE/'hospital_h3b_policy.py', HERE/'hospital_drive_hold.py',
        HERE/'hospital_private_loader.py', HERE/'hospital_h3b_timed_second_reverse_model.py',
        entries/'isaac_review_ijcas_hospital_h3b_timed_second_reverse.py',
        entries/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_instrumented.py']
    return dict(unchanged_AST_functions=list(unchanged),
        low_level_command_AST_equal_except_disclosed_rear_to_front_mode=True,
        controller_class_used_by_model_and_runtime='hospital_timed_path_execution.TimedLegController',
        NEW_timing_and_braking_sequence_NOT_a_new_caster_or_yaw_controller=True,
        original_actual_start_12_joints_final_goal_and_frozen_second_reverse_kept=True,
        second_reverse_ONLY_no_final_forward_or_automatic_retry=True,
        original_stall_contact_swept_envelope_and_terminal_limits_kept=True,
        cumulative_reverse_and_direction_change_guards_added_not_relaxed=True,
        CICS_runtime_technical_integration=False,
        input_sha256={str(p): sha(p) for p in sources})


def admission_payload(info):
    contract = controller_and_adapter_contract()
    return dict(version=VERSION, status='SECOND_REVERSE_ADAPTER_PREPARED_LIVE_REVIEW_REQUIRED_NO_MOTION',
        forward_result=info['forward_result'], forward_result_sha256=info['forward_result_sha256'],
        model_review=info['model_review'], model_review_sha256=info['model_review_sha256'],
        adapter_model=info['adapter_model'], adapter_model_sha256=info['adapter_model_sha256'],
        actual_start_world_xyyaw=info['initial_world_xyyaw'], retained_all_joint_positions_rad=info['retained_all_joint_positions_rad'],
        fixed_second_reverse=info['leg'], original_mission_goal_xyyaw=info['mission_goal_xyyaw'],
        strict_map_digest=info['strict_map_digest'], execution_contract=execution_contract(), controller_contract=contract,
        input_sha256={**info['input_sha256'], **contract['input_sha256']},
        PhysX_executed=False, automatic_motion_authorized=False, CICS_input_used=False,
        full_H3B_turning_path_verified=False, use_for_final_statistics=False)


def load_prepared_profile():
    info = load_adapter_reviewed_profile()
    report = json.loads(ADMISSION.read_bytes())
    verify_files(report['input_sha256'])
    if report != admission_payload(info):
        raise ValueError('Exact prepared SECOND-reverse adapter contract required')
    info.update(admission=str(ADMISSION), admission_sha256=sha(ADMISSION),
        input_sha256={**report['input_sha256'], str(ADMISSION): sha(ADMISSION)})
    return info


def load_admitted_profile():
    info = load_prepared_profile()
    live = json.loads(LIVE.read_bytes())
    verify_files(live['input_sha256'])
    if (live.get('status') != 'TIMED_SECOND_REVERSE_ACTUAL_FORWARD_PASS_CONTINUITY_RECORDED_NO_MOTION'
            or live.get('forward_result_sha256') != info['forward_result_sha256']
            or live.get('executor_admission') != info['admission']
            or live.get('executor_admission_sha256') != info['admission_sha256']
            or live.get('model_review') != info['model_review']
            or live.get('model_review_sha256') != info['model_review_sha256']
            or live.get('all_timed_executor_design_cases_pass') is not True
            or live.get('CICS_input_used') is not False
            or live.get('physical_state_reset') is not False or live.get('new_PhysX_motion_executed') is not False
            or live.get('root_session_and_timeline_unchanged') is not True
            or live.get('automatic_motion_authorized') is not False):
        raise ValueError('Fresh dedicated PAUSED live continuity review required before one SECOND reverse')
    verify_files({WORKSPACE/'isaac/scripts/isaac_review_ijcas_hospital_h3b_timed_second_reverse.py': live['script_sha256']})
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    info.update(live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
        input_sha256={**info['input_sha256'], **live['input_sha256'], str(LIVE): sha(LIVE)})
    return info


def replay_commands(info, rows, gate_events):
    """Reproduce exact high-level, steering and applied wheel commands."""
    if len(gate_events) != 1 or not rows:
        raise ValueError('One measured release and complete raw rows required')
    controller = Progress(info, gate_events[0]['time_s'])
    command_error = steering_error = wheel_error = 0.
    ready_count = 0
    for before, row in zip(rows, rows[1:]):
        first_hold = row['phase'] == 'terminal_hold' and before['phase'] != 'terminal_hold'
        if row['phase'] not in ('reverse_motion', 'reverse_braking') and not first_hold:
            continue
        pose = [before[k] for k in ('x_m', 'y_m', 'yaw_rad')]
        if not controller.finished:
            out = controller.step(row['command_time_s'], relative_pose(pose, info['mission_start_xyyaw']),
                max(before['pose_filtered_speed_m_s'], math.hypot(before['vx_physx_m_s'], before['vy_physx_m_s'])),
                (before['front_rad'], before['rear_rad']))
        else:
            # The controller's arrival has already been acknowledged. Only
            # zero commands may drain the inherited low-level steering slew.
            out = dict(speed_m_s=0., curvature_1_m=0., ready=False)
        command_error = max(command_error, abs(out['speed_m_s'] - row['command_speed_m_s']),
            abs(out['curvature_1_m'] - row['requested_curvature_1_m']))
        angles, wheels = direct_joint_targets(out['speed_m_s'], 'front_only', out['curvature_1_m'],
            (before['command_front_rad'], before['command_rear_rad']),
            row['command_time_s'] - before['command_time_s'], (before['front_rad'], before['rear_rad']))
        steering_error = max(steering_error, *(abs(angles[i] - row['command_'+a+'_rad']) for i, a in enumerate(('front', 'rear'))))
        wheel_error = max(wheel_error, *(abs(wheels[i] - row['drive_'+a+'_actual_velocity_target_rad_s']) for i, a in enumerate(('front', 'rear'))))
        ready_count += int(out['ready'])
    return dict(maximum_controller_command_error=command_error, maximum_steering_target_error_rad=steering_error,
        maximum_applied_wheel_target_error_rad_s=wheel_error, terminal_ready_count=ready_count,
        exact_frozen_controller_and_mapping_reproduced=command_error <= 1e-10 and steering_error <= 1e-10
            and wheel_error <= 1e-6 and ready_count == 1)


def evaluate_component(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events, info):
    numeric = ('command_time_s', 'command_speed_m_s', 'command_front_rad', 'command_rear_rad',
        'front_rad', 'rear_rad', 'reference_progress_m', 'reference_position_error_m', 'reference_heading_error_deg')
    if not rows or any(not math.isfinite(r[k]) for r in rows for k in numeric):
        raise ValueError('Complete finite SECOND-reverse rows required')
    result = _evaluate_core(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events)
    replay = replay_commands(info, rows, gate_events)
    initial = info['initial_world_xyyaw']
    distance = math.hypot(rows[0]['x_m'] - initial[0], rows[0]['y_m'] - initial[1])
    distance += sum(math.hypot(b['x_m'] - a['x_m'], b['y_m'] - a['y_m']) for a, b in zip(rows, rows[1:]))
    result['checks'].update(exact_screened_timed_controller_commands_reproduced=replay['exact_frozen_controller_and_mapping_reproduced'],
        full_measured_reverse_episode_including_prehold_under_0p5m=distance <= .5,
        cumulative_measured_reverse_under_1m=info['previous_measured_reverse_total_m'] + distance <= 1.,
        cumulative_direction_changes_under_four=info['completed_direction_changes'] + len(gate_events) <= POLICY['maximum_direction_changes'])
    first_hold = next((r for r in rows if r['phase'] == 'terminal_hold'), None)
    result['checks']['exact_command_neutral_before_captured_hold'] = first_hold is not None and all(terminal_hold_checks(
        [first_hold[k] for k in ('x_m', 'y_m', 'yaw_rad')], goal,
        max(first_hold['pose_filtered_speed_m_s'], math.hypot(first_hold['vx_physx_m_s'], first_hold['vy_physx_m_s'])),
        (first_hold['front_rad'], first_hold['rear_rad']),
        (first_hold['command_front_rad'], first_hold['command_rear_rad'])).values())
    result.update(command_reproduction=replay, measured_center_distance_m=distance,
        second_reverse_component_verified=all(result['checks'].values()),
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        final_forward_component_verified=False, CICS_input_used=False, use_for_final_statistics=False)
    result['verdict'] = 'PASS' if all(result['checks'].values()) else 'FAIL'
    return result
