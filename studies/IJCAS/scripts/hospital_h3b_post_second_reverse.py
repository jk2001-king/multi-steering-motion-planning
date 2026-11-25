"""Audit actual timed SECOND-reverse PASS; extract frozen FINAL forward only.

No scene action, model fitting, caster assignment or old outcome replacement.
The exact original mission goal remains the acceptance target. The frozen
path's numerical endpoint residual is reported, never silently corrected.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_timed_second_reverse as reverse
from hospital_h3b_timed_second_reverse_observation import coverage
from hospital_h3b_rolling_force_audit import numeric_rows, anchor_sums_match
from hospital_h3_turn_planning import verify_sources, model_bounds
from hospital_h3_flush_support import load_setup
from hospital_timed_path_execution import SpatialLeg, LegTiming, TimedLegController, execution_contract
from hospital_native_envelope import NativeHospitalEnvelopeChecker


WORKSPACE, GATE = reverse.WORKSPACE, reverse.GATE
RESULT = reverse.OUTPUT/'hospital_h3b_timed_second_reverse_r01.json'
BASE = GATE/'hospital_h3b_post_second_reverse_reviews'
VERSION = 'H3B_actual_timed_second_reverse_PASS_final_forward_MODEL_review_v1'
STEPS, RESIDUALS, SCALES = reverse.previous.STEPS, reverse.previous.RESIDUALS, reverse.previous.SCALES
sha, verify_files = reverse.sha, reverse.verify_files


def read_trace(path):
    """Versioned reader: the new timed_executor_phase column is textual."""
    text_fields = {'phase', 'command_mode', 'timed_executor_phase', 'controller_phase'}
    rows = []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            converted = {}
            for key, value in row.items():
                if key in text_fields: converted[key] = value
                elif value in ('True', 'False'): converted[key] = value == 'True'
                else:
                    converted[key] = float(value)
                    if not math.isfinite(converted[key]): raise ValueError('Nonfinite raw trace: '+key)
            rows.append(converted)
    return rows


def final_leg(candidate, mission_start, mission_goal):
    g, timing = candidate['geometry_audit'], candidate['nominal_timing']
    if (g.get('verdict') != 'PASS' or not all(g['checks'].values()) or not all(timing['checks'].values())
            or g['leg_directions'] != [-1, 1, -1, 1]
            or g['leg_modes'] != ['rear_only', 'coordinated_counterphase', 'front_only', 'coordinated_counterphase']):
        raise ValueError('The SAME complete frozen rear-first/front-second four-leg suffix is required')
    left, right = timing['cusp_sample_indices'][-1], len(g['poses_local_xyyaw'])-1
    arc = np.asarray(g['travelled_arc_m'][left:right+1], float);arc -= arc[0]
    profile = dict(original_suffix_leg_index=3, original_sample_indices_inclusive=[left, right],
        mode='coordinated_counterphase', direction=1,
        poses_local_xyyaw=g['poses_local_xyyaw'][left:right+1],
        center_arc_m=arc.tolist(), curvature_1_m=g['sample_curvature_1_m'][left:right+1],
        next_cusp_local_xyyaw=g['poses_local_xyyaw'][right], reference_length_m=float(arc[-1]))
    if (not 0 < left < right or any(d != 1 for d in g['interval_directions'][left:right])
            or any(m != 'coordinated_counterphase' for m in g['interval_modes'][left:right])):
        raise ValueError('Only the complete FINAL existing counterphase FORWARD leg may be reviewed')
    leg = SpatialLeg(profile)
    exact_goal = np.asarray(reverse.relative_pose(mission_goal, mission_start), float)
    position = float(np.linalg.norm(leg.goal[:2]-exact_goal[:2]))
    heading = abs(math.degrees(reverse.wrap(leg.goal[2]-exact_goal[2])))
    if position > .002 or heading > .2:
        raise ValueError('Frozen endpoint no longer agrees with the ORIGINAL mission goal')
    profile.update(exact_mission_goal_local_xyyaw=exact_goal.tolist(),
        frozen_endpoint_numerical_residual_m=position, frozen_endpoint_numerical_residual_deg=heading,
        exact_original_mission_goal_is_physical_acceptance_target=True)
    return profile


def inspect_second_reverse():
    info = reverse.load_admitted_profile()
    record = json.loads(RESULT.read_bytes())
    verify_sources(record);verify_files(record['frozen_input_sha256'])
    if (record.get('verdict') != 'PASS' or record.get('trial_kind') != reverse.ROLE
            or record.get('PhysX_executed') is not True or record.get('error') is not None
            or not record.get('checks') or not all(record['checks'].values())
            or record.get('original_motor_gains_restored') is not True
            or record.get('second_reverse_component_verified') is not True
            or record.get('full_suffix_execution_verified') is not False
            or record.get('body_pose_reset_during_run') is not False
            or record.get('caster_state_reset_during_run') is not False
            or record.get('CICS_input_used') is not False
            or record['initial_saved_world_xyyaw'] != info['initial_world_xyyaw']
            or record['expected_retained_joint_positions_rad'] != info['retained_all_joint_positions_rad']
            or record['strict_map_digest'] != info['strict_map_digest']
            or record['suffix_candidate_sha256'] != info['suffix_candidate_sha256']
            or record['mission_goal_world_xyyaw_UNCHANGED'] != info['mission_goal_xyyaw']
            or record['intervention_contract'] != execution_contract()):
        raise ValueError('Exact actual shared-timing SECOND-reverse PASS required, never an older first reverse')
    rows = read_trace(RESULT.with_suffix('.csv'))
    if len(rows) != record['sample_count'] or rows[-1] != record['last_recorded_state']:
        raise ValueError('Timed second-reverse raw CSV and saved final state disagree')
    _, manifest, geometry, inventory = load_setup(Path(record['geometry_manifest']).parent, WORKSPACE)
    actual = reverse.evaluate_component(rows, record['target_world_xyyaw'],
        read_trace(Path(record['static_reference']).with_suffix('.csv')),
        record['physics_preparation_snapshot']['simulation_mass_kg'],
        inventory['explicitly_suppressed_assets'][0]['collision']['bound'], record['wheel_conservative_sphere_radii_m'],
        record['contact_report_counts_by_category'], record['contact_callback_errors'],
        record['fixed_hold_attempt_count'], record['direction_gate_events'], info)
    if (actual['verdict'] != 'PASS' or any(record['checks'].get(k) != v for k, v in actual['checks'].items())
            or actual['command_reproduction'] != record['command_reproduction']
            or abs(actual['measured_center_distance_m']-record['measured_center_distance_m']) > 1e-10):
        raise ValueError('Raw acceptance/commands/captured hold/resource PASS does not reproduce')
    observer_path = reverse.COHORT/'instrumentation/instrumentation.json'
    observer = json.loads(observer_path.read_bytes());verify_sources(observer)
    verify_files(observer['reference_input_sha256'])
    files = {str(observer_path.parent/n): digest for n, digest in observer['saved_file_sha256'].items()}
    verify_files(files)
    if observer['original_controller_result_sha256'] != sha(RESULT) or observer['telemetry_errors']:
        raise ValueError('Passive instrumentation/primary lineage mismatch or telemetry error')
    joints = numeric_rows(observer_path.parent/'joint_body_solver_trace.csv')
    friction = numeric_rows(observer_path.parent/'wheel_friction_summary.csv')
    with (observer_path.parent/'friction_anchors.csv').open(newline='') as stream: anchors = list(csv.DictReader(stream))
    with (observer_path.parent/'raw_contact_points.csv').open(newline='') as stream:
        contacts = [dict(r, complete_point_data=r['complete_point_data']=='True') for r in csv.DictReader(stream)]
    measured = coverage(joints, friction, anchors, contacts, record, [])
    if any(observer.get(k) != v for k, v in measured.items()):
        raise ValueError('All six wheel observer coverage does not reproduce')
    anchor_error = anchor_sums_match(friction, anchors, observer['native_sensor_paths_in_recorded_wheel_index_order'])
    q = reverse.previous.forward.reverse.joints_from_last(record)
    candidate = json.loads(Path(info['suffix_candidate']).read_bytes())
    leg = final_leg(candidate, info['mission_start_xyyaw'], info['mission_goal_xyyaw'])
    reverse_total = info['previous_measured_reverse_total_m'] + actual['measured_center_distance_m']
    completed_changes = info['completed_direction_changes'] + len(record['direction_gate_events'])
    if len(q) != 12 or reverse_total > 1. or completed_changes + 1 > reverse.POLICY['maximum_direction_changes']:
        raise ValueError('All retained joints and ORIGINAL cumulative resources are required')
    info.update(second_reverse_result=str(RESULT), second_reverse_result_sha256=sha(RESULT), second_reverse_record=record,
        initial_world_xyyaw=record['final_world_xyyaw'], expected_saved_last_state=rows[-1],
        retained_all_joint_positions_rad=q, final_leg=leg,
        previous_measured_reverse_total_m=reverse_total, completed_direction_changes=completed_changes,
        input_sha256={**info['input_sha256'], **record['frozen_input_sha256'], str(RESULT):sha(RESULT),
            str(RESULT.with_suffix('.csv')):sha(RESULT.with_suffix('.csv')), str(observer_path):sha(observer_path), **files})
    audit = dict(original_verdict_unchanged='PASS', primary_check_count=len(actual['checks']),
        command_reproduction=actual['command_reproduction'], actual_fixed_hold_attempt_count=record['fixed_hold_attempt_count'],
        terminal_position_error_m=actual['terminal_position_error_m'], terminal_heading_error_deg=actual['terminal_heading_error_deg'],
        measured_reverse_episode_m=actual['measured_center_distance_m'], cumulative_measured_reverse_m=reverse_total,
        completed_direction_changes=completed_changes, raw_observer_coverage_reproduced=True,
        maximum_raw_anchor_sum_error_n=anchor_error, observer_does_NOT_isolate_unique_motor_or_caster_cause=True)
    return info, audit, manifest, geometry, inventory
