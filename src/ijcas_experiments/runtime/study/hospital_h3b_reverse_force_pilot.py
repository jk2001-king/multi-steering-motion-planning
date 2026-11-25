"""Rear-only FIRST reverse from the freshly reproduced, actual prefix PASS.

No old scene recovery, pose recapture, caster initialization, reverse feedback
retuning or broader route authorization. Original generic reverse progress,
stop/hold, contact, slew and terminal rules are reused. A separate read-only
live continuity receipt is required before the one manually scheduled trial.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import json
import math
from pathlib import Path

import numpy as np

from hospital_h3b_measured_reverse import (
    POLICY, StopAlignGate, FirstReverseReference, FirstReverseProgress,
    ROBOT_ROOT, WHEELS, BAD_CONTACTS, classify_contact, relative_pose,
    evaluate_passage, direct_joint_targets, ramp, wrap, motion_stalled,
)
from hospital_h3_flush_support import sha, verify_files
from hospital_h3b_rear_prefix_pilot import evaluate_readiness as evaluate_prefix


WORKSPACE = _ijcas_workspace()
GATE = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
FIRST = GATE / 'hospital_h3b_reverse_force_observation/cohort_01/prefix/hospital_h3b_reverse_force_prefix_r01.json'
ORIGINAL = GATE / 'hospital_h3b_course_first_leg_pilot/cohort_01/hospital_h3b_course_first_leg_r01.json'
MODEL = GATE / 'hospital_h3b_rear_reverse_model/review_r01/rear_reverse_model_review.json'
ADMISSION = GATE / 'hospital_h3b_rear_reverse_fresh_review/review_r01/fresh_prefix_rear_review.json'
LIVE = GATE / 'hospital_h3b_reverse_force_observation/cohort_01/handoff/live_handoff.json'
OUTPUT = GATE / 'hospital_h3b_reverse_force_observation/cohort_01/reverse'
VERSION = 'H3B_fresh_prefix_rear_only_first_reverse_v1'
ROLE = 'ONE_SAME_CONDITION_REAR_REVERSE_WITH_PASSIVE_FORCE_OBSERVER_NOT_FINAL_COMPARISON'
TRACKING = {'maximum_reference_position_error_m': .080, 'maximum_reference_heading_error_deg': 10.}
GUARDS = {'maximum_simulation_duration_s': 90., 'maximum_wall_time_s': 300.}


def read_trace(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return [{k: (v if k in ('phase', 'command_mode') else
                     v == 'True' if v in ('True', 'False') else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def joints_from_last(first):
    last = first['last_recorded_state']
    joints = {k[len('joint_'):-len('_rad')]: float(v) for k, v in last.items()
              if k.startswith('joint_') and k.endswith('_rad')}
    if len(joints) != 12 or not all(math.isfinite(v) for v in joints.values()):
        raise ValueError('All twelve retained physical joint positions required')
    return joints


def load_profile():
    first = json.loads(FIRST.read_bytes()); original = json.loads(ORIGINAL.read_bytes())
    model = json.loads(MODEL.read_bytes()); verify_files(model['input_sha256'])
    for record in (first, original):
        if (record.get('verdict') != 'PASS' or record.get('original_motor_gains_restored') is not True
                or not record.get('checks') or not all(record['checks'].values())):
            raise ValueError('Both original and freshly reproduced prefix must remain actual PASS')
        verify_files(record['frozen_input_sha256'])
        for name, digest in record['source_sha256'].items():
            folder = _ijcas_path(WORKSPACE, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
            verify_files({folder / name: digest})
    if (first.get('fresh_first_forward_prefix_reproduced') is not True
            or first.get('rear_reverse_verified') is not False
            or first['component_condition'] != 'H3B_rear_reverse_prefix_reproduction_v1'
            or first['mission_goal_world_xyyaw_UNCHANGED'] != original['mission_goal_world_xyyaw_UNCHANGED']
            or first['start_world_xyyaw'] != original['start_world_xyyaw']
            or first['strict_flush_map_digest'] != original['strict_flush_map_digest']):
        raise ValueError('Fresh prefix, unchanged original task or strict map lineage mismatch')
    if (model.get('all_predeclared_model_cases_admitted') is not True
            or model.get('PhysX_executed') is not False
            or len(model['cases']) != 4
            or {(c['dt_s'], c['residual_case']) for c in model['cases']} != {
                (dt, residual) for dt in (.01, .02) for residual in
                ('frozen_nominal', 'observed_prior_neutral_offset_synthetic')}
            or any(not all(c['checks'].values()) for c in model['cases'])):
        raise ValueError('All four retained model screens required; no physical rear PASS implied')
    verify_files({c['trace_file']: c['trace_sha256'] for c in model['cases']})
    joints = joints_from_last(first); old_joints = joints_from_last(original)
    state_keys = ('x_m', 'y_m', 'z_m', 'yaw_rad', 'front_rad', 'rear_rad',
                  'vx_physx_m_s', 'vy_physx_m_s', 'command_front_rad', 'command_rear_rad')
    delta = max(abs(first['last_recorded_state'][k] - original['last_recorded_state'][k]) for k in state_keys)
    joint_delta = max(abs(joints[k] - old_joints[k]) for k in joints)
    if delta > 1e-9 or joint_delta > 1e-9:
        raise ValueError('Fresh actual endpoint differs from screened start; HOLD, never reset/shift it to match')
    for key in ('geometry_manifest', 'static_reference'):
        verify_files({first[key]: first[key + '_sha256']})
    static_path = Path(first['static_reference']); static = json.loads(static_path.read_bytes())
    verify_files({static_path.with_suffix('.csv'): first['static_trace_sha256']})
    prefix_rows = read_trace(FIRST.with_suffix('.csv'))
    static_rows = read_trace(static_path.with_suffix('.csv'))
    prep = first['physics_preparation_snapshot']
    bound = first['doorway_reference_bound_m']
    actual = evaluate_prefix(prefix_rows, first['target_world_xyyaw'], static_rows,
                             prep['simulation_mass_kg'], bound, first['wheel_conservative_sphere_radii_m'],
                             first['contact_report_counts_by_category'], first['contact_callback_errors'],
                             first['fixed_hold_attempt_count'], first['direction_gate_events'])
    if actual['verdict'] != 'PASS':
        raise ValueError('Fresh actual prefix CSV does not reproduce its reported PASS')
    for key in ('x_m', 'y_m', 'yaw_rad'):
        if abs(prefix_rows[-1][key] - first['last_recorded_state'][key]) > 1e-12:
            raise ValueError('Fresh CSV tail differs from saved final body state')
    candidate_path = Path(model['candidate']); verify_files({candidate_path: model['candidate_sha256']})
    suffix = json.loads(candidate_path.read_bytes()); verify_files(suffix['input_sha256'])
    g = suffix['geometry_audit']; end = suffix['nominal_timing']['cusp_sample_indices'][0]
    local_start = relative_pose(first['final_world_xyyaw'], first['start_world_xyyaw'])
    if (g['verdict'] != 'PASS' or not all(g['checks'].values())
            or not all(suffix['nominal_timing']['checks'].values())
            or any(v != -1 for v in g['interval_directions'][:end])
            or any(v != 'rear_only' for v in g['interval_modes'][:end])
            or not np.allclose(g['poses_local_xyyaw'][0], local_start, rtol=0, atol=1e-12)
            or suffix['strict_map_digest'] != first['strict_flush_map_digest']
            or suffix['original_mission_goal_xyyaw'] != first['mission_goal_world_xyyaw_UNCHANGED']):
        raise ValueError('Rear-first frozen candidate does not start at fresh actual prefix endpoint')
    inputs = {**model['input_sha256'], **suffix['input_sha256'], **first['frozen_input_sha256'],
              str(FIRST): sha(FIRST), str(FIRST.with_suffix('.csv')): sha(FIRST.with_suffix('.csv')),
              str(ORIGINAL): sha(ORIGINAL), str(MODEL): sha(MODEL), str(candidate_path): sha(candidate_path),
              first['geometry_manifest']: first['geometry_manifest_sha256'],
              first['static_reference']: first['static_reference_sha256'],
              str(static_path.with_suffix('.csv')): first['static_trace_sha256'], str(Path(__file__)): sha(__file__)}
    info = dict(version=VERSION, mode='rear_only', first_record=first, first_result=str(FIRST),
                first_result_sha256=sha(FIRST), input_sha256=inputs,
                original_first_PASS_reference=str(ORIGINAL), original_first_PASS_sha256=sha(ORIGINAL),
                original_rear_model=str(MODEL), original_rear_model_sha256=sha(MODEL),
                suffix_candidate=str(candidate_path), suffix_candidate_sha256=sha(candidate_path),
                mission_start_xyyaw=first['start_world_xyyaw'],
                mission_goal_xyyaw=first['mission_goal_world_xyyaw_UNCHANGED'],
                strict_map_digest=first['strict_flush_map_digest'],
                initial_world_xyyaw=first['final_world_xyyaw'],
                expected_saved_last_state=first['last_recorded_state'],
                retained_all_joint_positions_rad=joints,
                poses_local_xyyaw=g['poses_local_xyyaw'][:end + 1],
                curvature_1_m=g['sample_curvature_1_m'][:end + 1],
                center_arc_m=g['travelled_arc_m'][:end + 1],
                next_cusp_local_xyyaw=g['poses_local_xyyaw'][end],
                nominal_reference_reverse_m=g['leg_lengths_m'][0],
                fresh_saved_body_state_max_difference=delta,
                fresh_saved_joint_max_difference_rad=joint_delta,
                raw_prefix_PASS_reproduced=True,
                curvature_is_frozen_spatial_feedforward_NOT_reverse_course_or_yaw_feedback=True,
                actual_caster_state_never_assigned=True)
    FirstReverseReference(info)
    return info


def load_reviewed_profile():
    # Reuse retained MODEL screens only after an EXACT actual-state match.
    # This is not a new model run or a claim of reverse PhysX feasibility.
    info = load_profile(); record = json.loads(ADMISSION.read_bytes()); verify_files(record['input_sha256'])
    from hospital_h3b_reverse_force_prefix import review_reference
    baseline = json.loads(Path(record['first_result']).read_bytes())
    if (record.get('version') != VERSION or record.get('all_cases_admitted') is not True
            or record.get('PhysX_executed') is not False or record.get('automatic_PhysX_execution_authorized') is not False
            or sha(record['first_result']) != record['first_result_sha256']
            or baseline.get('verdict') != 'PASS' or len(record['cases']) != 4
            or any(not all(c['checks'].values()) for c in record['cases'])
            or baseline['final_world_xyyaw'] != info['initial_world_xyyaw']
            or joints_from_last(baseline) != info['retained_all_joint_positions_rad']
            or record['strict_map_digest'] != info['strict_map_digest']):
        raise ValueError('Retained model review requires the SAME actual body/12 joints and unchanged map; no reset or selection retry')
    verify_files({c['trace_file']: c['trace_sha256'] for c in record['cases']})
    info.update(admission=str(ADMISSION), admission_sha256=sha(ADMISSION),
                model_admission_reused_ONLY_for_exact_actual_state_equivalence=True,
                MODEL_screens_NOT_recomputed_or_PhysX_success=True,
                input_sha256={**info['input_sha256'], **record['input_sha256'],
                              **review_reference(), str(ADMISSION): sha(ADMISSION)})
    return info


def validate_live(info, pose, joints, velocity):
    pose = np.asarray(pose, float); velocity = np.asarray(velocity, float)
    expected = info['retained_all_joint_positions_rad']
    if (pose.shape != (3,) or velocity.shape != (6,) or not np.isfinite(pose).all()
            or not np.isfinite(velocity).all() or set(joints) != set(expected)
            or not all(math.isfinite(v) for v in joints.values())):
        raise ValueError('Finite live body/six velocities/all twelve joints required')
    saved = info['initial_world_xyyaw']
    position = float(np.linalg.norm(pose[:2] - saved[:2]))
    yaw = math.degrees(wrap(pose[2] - saved[2]))
    joint = max(abs(joints[k] - expected[k]) for k in expected)
    speed = float(np.linalg.norm(velocity[:2]))
    if position > .001 or abs(yaw) > .1 or joint > .001 or speed > .005:
        raise ValueError('Live fresh-PASS body/12 joints/speed differs; keep PAUSED; never reset to force a match')
    if max(abs(joints['base_link_to_' + axle + '_active_steering']) for axle in ('front', 'rear')) > math.radians(2.5):
        raise ValueError('Existing 2.5deg actual-neutral gate not satisfied')
    return dict(position_difference_m=position, yaw_difference_deg=yaw,
                maximum_joint_difference_rad=joint, measured_live_speed_m_s=speed)


def load_admitted_profile():
    info = load_reviewed_profile(); live = json.loads(LIVE.read_bytes()); verify_files(live['input_sha256'])
    if (live.get('status') != 'PAUSED_FRESH_PREFIX_REAR_HANDOFF_REVIEWED_NO_MOTION'
            or live.get('physical_state_reset') is not False or live.get('rear_reverse_PhysX_executed') is not False
            or live['first_result_sha256'] != info['first_result_sha256']
            or live['model_admission_sha256'] != info['admission_sha256']):
        raise ValueError('Fresh read-only live handoff receipt required; no old first-PASS/stalled scene accepted')
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    info.update(live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
                input_sha256={**info['input_sha256'], **live['input_sha256'], str(LIVE): sha(LIVE)})
    return info


def evaluate_component(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events):
    common = evaluate_passage(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count)
    excluded = {'all_six_wheels_started_east_and_finished_west', 'ordinary_positive_ramp_within_30mm_s', 'passage_and_hold_completed'}
    checks = {k: v for k, v in common['checks'].items() if k not in excluded}
    moving = [r for r in rows if r['phase'] == 'reverse_motion' and r['command_speed_m_s'] < -.003]
    maximum_slew = 0.
    for a, b in zip(rows, rows[1:]):
        dt = b['command_time_s'] - a['command_time_s']
        if dt < 0: raise ValueError('Command clock went backwards')
        for axle in ('front', 'rear'):
            change = abs(b['command_' + axle + '_rad'] - a['command_' + axle + '_rad'])
            if dt == 0 and change > 1e-12: raise ValueError('Steering changed with no elapsed command time')
            maximum_slew = max(maximum_slew, change / dt if dt else 0.)
    gate = gate_events[0] if len(gate_events) == 1 else {}
    checks.update(first_reverse_and_single_hold_completed=True,
        one_measured_stop_hold_neutral_release_before_reverse=len(gate_events) == 1
            and gate.get('direction') == -1 and gate.get('captured_hold_acknowledged') is True
            and gate.get('hold_release_acknowledged') is True and gate.get('measured_speed_m_s', 1.) <= .005
            and gate.get('steering_error_deg', 180.) <= 2.5,
        only_rear_only_reverse_commands=all(r['command_mode'] == 'rear_only' and -.010 - 1e-9 <= r['command_speed_m_s'] <= 1e-9 for r in rows),
        measured_reverse_progress_over_50mm=bool(moving) and moving[-1]['reference_progress_m'] - moving[0]['reference_progress_m'] > .050,
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
