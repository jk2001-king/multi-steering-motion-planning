"""New actual-prefix admission for the measured braking/terminal executor.

No recovery of the failed endpoint. Require raw new prefix PASS, all 12 joints,
the same fixed path/goal/map, a fresh model review and a read-only live receipt.
"""

import json
import math
from pathlib import Path

from hospital_h3b_reverse_force_pilot import (
    load_profile as baseline_profile, read_trace, joints_from_last,
    StopAlignGate, FirstReverseReference, ROBOT_ROOT, WHEELS, BAD_CONTACTS,
    classify_contact, relative_pose, evaluate_passage, direct_joint_targets,
    ramp, wrap, motion_stalled, validate_live, evaluate_prefix,
    TRACKING, GUARDS, sha, verify_files, POLICY as ORIGINAL_POLICY,
)
from hospital_h3b_reverse_closedloop import (
    VERSION, FirstReverseProgress, execution_contract, motion_stalled,
    reverse_episode_progress, arrival_checks,
)
from hospital_h3b_reverse_closedloop_prefix import WORKSPACE, GATE, COHORT, PRE_MODEL, validate_model, ROLE as PREFIX_ROLE
from hospital_h3_turn_planning import verify_sources


FIRST = COHORT/'prefix/hospital_h3b_reverse_closedloop_prefix_r01.json'
ADMISSION = COHORT/'admission/closedloop_model_review.json'
LIVE = COHORT/'handoff/live_handoff.json'
OUTPUT = COHORT/'reverse'
ROLE = 'ONE_NEW_MEASURED_TERMINAL_REAR_REVERSE_EXECUTOR_NOT_FINAL_COMPARISON'
POLICY = dict(ORIGINAL_POLICY, version=VERSION, reverse_speed_limit_m_s=.050)


def load_profile():
    info = baseline_profile(); first = json.loads(FIRST.read_bytes())
    if (first.get('verdict') != 'PASS' or first.get('trial_kind') != PREFIX_ROLE
            or first.get('original_motor_gains_restored') is not True
            or not first.get('checks') or not all(first['checks'].values())
            or first.get('fresh_first_forward_prefix_reproduced') is not True
            or first.get('component_condition') != 'H3B_rear_reverse_prefix_reproduction_v1'
            or first.get('rear_reverse_verified') is not False):
        raise ValueError('New physical prefix PASS required; never reuse the old failed scene')
    verify_sources(first); verify_files(first['frozen_input_sha256'])
    old = info['first_record']; joints = joints_from_last(first)
    if (first['final_world_xyyaw'] != old['final_world_xyyaw']
            or joints != info['retained_all_joint_positions_rad']
            or first['start_world_xyyaw'] != old['start_world_xyyaw']
            or first['mission_goal_world_xyyaw_UNCHANGED'] != old['mission_goal_world_xyyaw_UNCHANGED']
            or first['strict_flush_map_digest'] != info['strict_map_digest']):
        raise ValueError('New endpoint/history differs: HOLD for a new review, never reset it to match')
    for key in ('environment_condition', 'source_stage_sha256', 'robot_reference_sha256',
                'start_xyyaw', 'component_end_xyyaw', 'simulation_mass_kg',
                'live_support_readback', 'support_floor', 'disabled_legacy_floor_colliders'):
        if first['physics_preparation_snapshot'][key] != old['physics_preparation_snapshot'][key]:
            raise ValueError('Only the disclosed terminal executor may change; preparation differs: '+key)
    for key in ('geometry_manifest', 'static_reference'):
        verify_files({first[key]: first[key+'_sha256']})
    rows = read_trace(FIRST.with_suffix('.csv'))
    static_path = Path(first['static_reference'])
    verify_files({static_path.with_suffix('.csv'): first['static_trace_sha256']})
    if len(rows) != first['sample_count'] or rows[-1] != first['last_recorded_state']:
        raise ValueError('New actual prefix CSV and last-state receipt disagree')
    actual = evaluate_prefix(rows, first['target_world_xyyaw'], read_trace(static_path.with_suffix('.csv')),
        first['physics_preparation_snapshot']['simulation_mass_kg'], first['doorway_reference_bound_m'],
        first['wheel_conservative_sphere_radii_m'], first['contact_report_counts_by_category'],
        first['contact_callback_errors'], first['fixed_hold_attempt_count'], first['direction_gate_events'])
    if actual['verdict'] != 'PASS': raise ValueError('New prefix raw PASS does not reproduce')
    info.update(version=VERSION, first_record=first, first_result=str(FIRST), first_result_sha256=sha(FIRST),
                initial_world_xyyaw=first['final_world_xyyaw'], expected_saved_last_state=first['last_recorded_state'],
                retained_all_joint_positions_rad=joints, intervention_contract=execution_contract(),
                input_sha256={**info['input_sha256'], **first['frozen_input_sha256'],
                              str(FIRST):sha(FIRST), str(FIRST.with_suffix('.csv')):sha(FIRST.with_suffix('.csv')),
                              str(Path(__file__)):sha(__file__),
                              str(Path(__file__).with_name('hospital_h3b_reverse_closedloop.py')):sha(Path(__file__).with_name('hospital_h3b_reverse_closedloop.py'))})
    return info


def load_reviewed_profile():
    info = load_profile(); record = json.loads(ADMISSION.read_bytes()); validate_model(record)
    if (record.get('fresh_prefix_review') is not True
            or record['first_result'] != info['first_result']
            or record['first_result_sha256'] != info['first_result_sha256']
            or record['initial_actual_world_xyyaw'] != info['initial_world_xyyaw']
            or record['retained_all_joint_positions_rad'] != info['retained_all_joint_positions_rad']
            or record['suffix_candidate_sha256'] != info['suffix_candidate_sha256']
            or record['strict_map_digest'] != info['strict_map_digest']):
        raise ValueError('Fresh NEW terminal-executor model review does not describe this actual prefix state')
    info.update(admission=str(ADMISSION), admission_sha256=sha(ADMISSION),
                input_sha256={**info['input_sha256'], **record['input_sha256'], str(ADMISSION):sha(ADMISSION)})
    return info


def load_admitted_profile():
    info = load_reviewed_profile(); live = json.loads(LIVE.read_bytes()); verify_files(live['input_sha256'])
    if (live.get('status') != 'PAUSED_FRESH_PREFIX_REAR_HANDOFF_REVIEWED_NO_MOTION'
            or live.get('physical_state_reset') is not False or live.get('rear_reverse_PhysX_executed') is not False
            or live['first_result_sha256'] != info['first_result_sha256']
            or live['model_admission_sha256'] != info['admission_sha256']):
        raise ValueError('Fresh live no-motion handoff required before the NEW measured terminal executor')
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    info.update(live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
                input_sha256={**info['input_sha256'], **live['input_sha256'], str(LIVE):sha(LIVE)})
    return info


def evaluate_component(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events):
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
    checks.update(first_reverse_and_single_hold_completed=True,
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
        only_rear_only_reverse_commands=all(r['command_mode'] == 'rear_only' and -.050 - 1e-9 <= r['command_speed_m_s'] <= 1e-9 for r in rows),
        measured_reverse_progress_over_50mm=reverse_episode_progress(rows) > .050,
        measured_center_distance_under_original_0p5m=sum(math.hypot(b['x_m'] - a['x_m'], b['y_m'] - a['y_m']) for a, b in zip(rows, rows[1:])) <= .5,
        held_commands_zero=all(abs(r['command_speed_m_s']) < 1e-7 and max(abs(r['command_' + a + '_rad']) for a in ('front', 'rear')) < 1e-7 for r in rows if r['position_hold_enabled']),
        held_actual_steering_neutral=all(max(abs(r[a + '_rad']) for a in ('front', 'rear')) <= math.radians(2.5) for r in rows if r['position_hold_enabled']),
        command_slew_under_8deg_s=maximum_slew <= math.radians(8.) + 1e-5,
        existing_curved_reference_bounds=all(r['reference_position_error_m'] <= .080 and abs(r['reference_heading_error_deg']) <= 10. for r in rows))
    return dict(checks={k: bool(v) for k, v in checks.items()}, verdict='PASS' if all(checks.values()) else 'FAIL',
        **{k: common[k] for k in ('terminal_position_error_m', 'terminal_heading_error_deg', 'final_world_xyyaw', 'fixed_terminal_hold_analysis', 'final_force_weight_ratio')},
        maximum_observed_command_slew_deg_s=math.degrees(maximum_slew),
        first_reverse_component_verified=all(checks.values()), rear_only_reverse_component_verified=all(checks.values()),
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        Nav2_executed=False, CICS_input_used=False, use_for_final_statistics=False)
