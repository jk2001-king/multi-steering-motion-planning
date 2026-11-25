#!/usr/bin/env python3
"""Audit the measured-terminal PASS, then screen ONLY the next forward leg.

Offline only. Keep the measured reverse endpoint, all 12 joints, fixed next
cusp, final goal, map, model and existing course-feedback gains. The new
reference adapter is a MODEL probe, not an admitted physical executor. Neither
this report nor the read-only live check authorizes another motion command.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_reverse_closedloop_pilot as reverse
from hospital_h3b_reverse_closedloop_observation import coverage
from hospital_h3b_rolling_force_audit import numeric_rows, anchor_sums_match
from hospital_h3b_course_tracking import CourseFirstLegFeedback, body_velocity, TRACKING_GAINS
from hospital_h3b_response_link import FrozenSteeringResponse, load_validated_response
from hospital_h3_flush_support import load_setup
from hospital_h3_turn_planning import model_bounds, verify_sources
from hospital_path_execution import PathReference, direct_joint_targets, relative_pose, wrap
from hospital_native_envelope import NativeHospitalEnvelopeChecker


VERSION = 'H3B_post_measured_reverse_next_forward_MODEL_review_v1'
RESULT = reverse.OUTPUT/'hospital_h3b_reverse_closedloop_rear_r01.json'
BASE = reverse.GATE/'hospital_h3b_post_reverse_reviews'
STEPS = (.02, .01)
RESIDUALS = ('frozen_nominal', 'observed_neutral_offset_SYNTHETIC')


def inspect_reverse():
    """Reproduce the actual primary verdict and commands; never edit evidence."""
    info = reverse.load_admitted_profile()
    record = json.loads(RESULT.read_bytes())
    verify_sources(record); reverse.verify_files(record['frozen_input_sha256'])
    if (record.get('trial_kind') != reverse.ROLE or record.get('verdict') != 'PASS'
            or record.get('error') is not None or not record.get('checks')
            or not all(record['checks'].values())
            or record.get('original_motor_gains_restored') is not True
            or record.get('full_suffix_execution_verified') is not False
            or record['initial_saved_world_xyyaw'] != info['initial_world_xyyaw']):
        raise ValueError('Need the NEW measured-terminal actual first-reverse PASS, not a stalled/old scene')
    rows = reverse.read_trace(RESULT.with_suffix('.csv'))
    if len(rows) != record['sample_count'] or rows[-1] != record['last_recorded_state']:
        raise ValueError('Actual reverse CSV / last-state receipt mismatch')
    _, manifest, geometry, inventory = load_setup(Path(record['geometry_manifest']).parent, reverse.WORKSPACE)
    bound = inventory['explicitly_suppressed_assets'][0]['collision']['bound']
    actual = reverse.evaluate_component(rows, record['target_world_xyyaw'],
        reverse.read_trace(Path(record['static_reference']).with_suffix('.csv')),
        record['physics_preparation_snapshot']['simulation_mass_kg'], bound,
        record['wheel_conservative_sphere_radii_m'], record['contact_report_counts_by_category'],
        record['contact_callback_errors'], record['fixed_hold_attempt_count'], record['direction_gate_events'])
    if actual['verdict'] != 'PASS' or any(record['checks'].get(k) != v for k, v in actual['checks'].items()):
        raise ValueError('Raw 30mm/2deg, fixed hold and stop/support PASS does not reproduce')
    controller = reverse.FirstReverseProgress(info, record['direction_gate_events'][0]['time_s'])
    command_error = target_error = 0.; ready_count = 0; events = []
    for previous, row in zip(rows, rows[1:]):
        first_hold = row['phase'] == 'terminal_hold' and previous['phase'] != 'terminal_hold'
        if row['phase'] not in ('reverse_motion', 'reverse_braking') and not first_hold:
            continue
        out = controller.step(row['command_time_s'],
            relative_pose([previous[k] for k in ('x_m', 'y_m', 'yaw_rad')], info['mission_start_xyyaw']),
            max(previous['pose_filtered_speed_m_s'], math.hypot(previous['vx_physx_m_s'], previous['vy_physx_m_s'])),
            (previous['front_rad'], previous['rear_rad']))
        command_error = max(command_error, abs(out['speed_m_s']-row['command_speed_m_s']),
                            abs(out['curvature_1_m']-row['requested_curvature_1_m']))
        angles, wheels = direct_joint_targets(out['speed_m_s'], 'rear_only', out['curvature_1_m'],
            (previous['command_front_rad'], previous['command_rear_rad']),
            row['command_time_s']-previous['command_time_s'], (previous['front_rad'], previous['rear_rad']))
        target_error = max(target_error, *(abs(angles[i]-row['command_'+a+'_rad'])
            for i, a in enumerate(('front', 'rear'))))
        ready_count += int(out['ready'])
        if out['transition']: events.append(dict(command_time_s=row['command_time_s'], **out))
    if (max(command_error, target_error) > 1e-10 or ready_count != 1
            or controller.reapproaches != record['terminal_reapproaches_used']
            or events != record['terminal_control_events']):
        raise ValueError('Recorded terminal controller/steering/arrival events do not reproduce')
    observer_path = reverse.COHORT/'instrumentation/instrumentation.json'
    observer = json.loads(observer_path.read_bytes()); verify_sources(observer)
    reverse.verify_files(observer['reference_input_sha256'])
    files = {str(observer_path.parent/name): digest for name, digest in observer['saved_file_sha256'].items()}
    reverse.verify_files(files)
    if observer['original_controller_result_sha256'] != reverse.sha(RESULT) or observer['telemetry_errors']:
        raise ValueError('Primary/observer lineage or coverage errors')
    joint = numeric_rows(observer_path.parent/'joint_body_solver_trace.csv')
    friction = numeric_rows(observer_path.parent/'wheel_friction_summary.csv')
    with (observer_path.parent/'friction_anchors.csv').open(newline='') as stream:
        anchors = list(csv.DictReader(stream))
    with (observer_path.parent/'raw_contact_points.csv').open(newline='') as stream:
        contacts = [dict(r, complete_point_data=r['complete_point_data'] == 'True') for r in csv.DictReader(stream)]
    measured_coverage = coverage(joint, friction, anchors, contacts, record, [])
    if any(observer.get(k) != v for k, v in measured_coverage.items()):
        raise ValueError('Observer raw coverage does not reproduce')
    anchor_error = anchor_sums_match(friction, anchors, observer['native_sensor_paths_in_recorded_wheel_index_order'])
    saved_joints = reverse.joints_from_last(record)
    if len(saved_joints) != 12: raise ValueError('All twelve actual post-reverse joints required')
    info.update(reverse_result=str(RESULT), reverse_result_sha256=reverse.sha(RESULT),
        reverse_record=record, expected_saved_last_state=rows[-1],
        initial_world_xyyaw=record['final_world_xyyaw'], retained_all_joint_positions_rad=saved_joints,
        input_sha256={**info['input_sha256'], **record['frozen_input_sha256'],
            str(RESULT):reverse.sha(RESULT), str(RESULT.with_suffix('.csv')):reverse.sha(RESULT.with_suffix('.csv')),
            str(observer_path):reverse.sha(observer_path), **files})
    audit = dict(original_verdict_unchanged='PASS', raw_primary_checks_reproduced=True,
        primary_check_count=len(actual['checks']), maximum_command_reproduction_error=command_error,
        maximum_steering_target_reproduction_error_rad=target_error,
        terminal_ready_event_count=ready_count, terminal_reapproaches=controller.reapproaches,
        actual_fixed_hold_attempt_count=record['fixed_hold_attempt_count'],
        terminal_position_error_m=actual['terminal_position_error_m'],
        terminal_heading_error_deg=actual['terminal_heading_error_deg'],
        actual_reverse_episode_progress_m=reverse.reverse_episode_progress(rows),
        observer_coverage_reproduced=True, maximum_raw_anchor_sum_error_n=anchor_error,
        observer_does_not_establish_unique_force_or_motor_cause=True)
    return info, audit, manifest, geometry, inventory


def next_leg(candidate):
    """Extract the fixed second SUFFIX leg, not the old first mission leg."""
    g, timing = candidate['geometry_audit'], candidate['nominal_timing']
    if (g['leg_directions'] != [-1, 1, -1, 1]
            or g['leg_modes'] != ['rear_only', 'coordinated_counterphase', 'front_only', 'coordinated_counterphase']
            or g.get('verdict') != 'PASS' or not all(g['checks'].values())):
        raise ValueError('Need the same four-leg rear-first spatial candidate; no alternative selected')
    left, right = timing['cusp_sample_indices'][:2]
    poses = np.array(g['poses_local_xyyaw'][left:right+1], float)
    arc = np.array(g['travelled_arc_m'][left:right+1], float)
    arc -= arc[0]
    k = np.array(g['sample_curvature_1_m'][left:right+1], float)
    if (poses.shape != (len(arc), 3) or len(arc) < 3 or not 0 < left < right
            or not all(np.isfinite(v).all() for v in (poses, arc, k))
            or np.any(np.diff(arc) <= 0) or max(abs(k)) > .7+1e-9
            or abs(k[0])+abs(k[-1]) > 1e-9
            or any(d != 1 for d in g['interval_directions'][left:right])
            or any(m != 'coordinated_counterphase' for m in g['interval_modes'][left:right])):
        raise ValueError('Need the complete straight-cusp-ended next counterphase FORWARD reference')
    return dict(original_suffix_leg_index=1, original_sample_indices_inclusive=[left, right],
        poses_local_xyyaw=poses.tolist(), center_arc_m=arc.tolist(), curvature_1_m=k.tolist(),
        mode='coordinated_counterphase', direction=1, length_m=float(arc[-1]),
        fixed_goal_local_xyyaw=poses[-1].tolist())


class NextForwardReference(PathReference):
    """Same H1 projection/interpolation; disclosed next-leg MODEL adapter."""
    def __init__(self, leg):
        arc = np.asarray(leg['center_arc_m']); points = np.asarray(leg['poses_local_xyyaw'])
        self.arc = np.linspace(0., arc[-1], math.ceil(arc[-1]/.025)+1).tolist()
        self.poses = [tuple(float(np.interp(s, arc, points[:,j])) for j in range(3)) for s in self.arc]
        mid = (np.array(self.arc[:-1])+self.arc[1:])/2
        self.actions = [dict(mode=leg['mode'], direction=1, target_speed_m_s=.050,
            curvature_1_m=float(k)) for k in np.interp(mid, arc, leg['curvature_1_m'])]
        self.progress = 0.


class NextForwardModelController(CourseFirstLegFeedback):
    """Inherit the existing step() law/gains; NOT a physical-run adapter."""
    def __init__(self, leg, started):
        if not math.isfinite(started): raise ValueError('Finite clock required')
        self.reference = NextForwardReference(leg); self.goal = tuple(leg['fixed_goal_local_xyyaw'])
        self.speed = 0.; self.phase = 'tracking'; self.previous_time = started; self.braking_started = None
        self.curvature = 0.; self.last_result = None


def model_case(info, leg, dt, residual_case):
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import MODE_CURVATURE_RESPONSE_SCALE, MODE_LATERAL_RESPONSE_SCALE
    if dt not in STEPS or residual_case not in RESIDUALS: raise ValueError('Both declared clocks/residuals required')
    frozen, _ = load_validated_response(); last = info['expected_saved_last_state']
    pose = np.array(relative_pose(info['initial_world_xyyaw'], info['mission_start_xyyaw']))
    u, v = body_velocity((last['vx_physx_m_s'], last['vy_physx_m_s']), last['yaw_rad'])
    actual_rows = reverse.read_trace(RESULT.with_suffix('.csv'))
    previous = actual_rows[-2]
    yaw_rate = wrap(last['yaw_rad']-previous['yaw_rad'])/(last['time_s']-previous['time_s'])
    velocity = np.array((u, v, yaw_rate)); initial_pose = pose.copy(); initial_velocity = velocity.copy()
    steer = (last['front_rad'], last['rear_rad'])
    residual = (0., 0.) if residual_case == 'frozen_nominal' else tuple(map(math.degrees, steer))
    response = FrozenSteeringResponse(frozen, steer, residual_deg=residual)
    commanded = tuple(last['command_'+a+'_rad'] for a in ('front', 'rear'))
    if max(abs(a) for a in commanded) > 1e-7: raise ValueError('Actual post-reverse commands must already be neutral')
    body = MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    now = 0.; tail = None; controller = None; trace = []; error = None
    feasible = True; maximum_slew = maximum_acceleration = previous_speed = 0.
    try:
        for _ in range(math.ceil(90./dt)):
            if now < 2.75-1e-10:
                speed = k = 0.; phase = 'model_neutral_prelude_NOT_measured_handoff'
            else:
                if controller is None: controller = NextForwardModelController(leg, now)
                if tail is None:
                    out = controller.step(now, pose, float(np.linalg.norm(velocity[:2])), response.actual, velocity[:2])
                    speed, k = out['speed_m_s'], out['curvature_1_m']; phase = 'next_forward_'+out['phase']
                    if out['ready']: tail = now
                if tail is not None: speed = k = 0.; phase = 'model_zero_command_tail_NOT_PhysX_hold'
            old = commanded
            commanded, _ = direct_joint_targets(speed, leg['mode'], k, commanded, dt, response.actual)
            maximum_slew = max(maximum_slew, *(abs(a-b)/dt for a,b in zip(old, commanded)))
            maximum_acceleration = max(maximum_acceleration, abs(speed-previous_speed)/dt); previous_speed = speed
            mid, _ = response.step(commanded, dt); tf, tr = np.tan(mid)
            curvature = (tf-tr)/1.29*MODE_CURVATURE_RESPONSE_SCALE[leg['mode']]
            lateral_ratio = (tf+tr)/2*MODE_LATERAL_RESPONSE_SCALE[leg['mode']]
            state = body.step(*velocity, speed, speed*lateral_ratio, speed*curvature, dt)
            updated = np.array((state.longitudinal_velocity_m_s, state.lateral_velocity_m_s, state.yaw_rate_rad_s))
            ux, vy, rate = (velocity+updated)/2; angle = pose[2]+rate*dt/2
            pose += dt*np.array((ux*math.cos(angle)-vy*math.sin(angle), ux*math.sin(angle)+vy*math.cos(angle), rate))
            velocity = updated; now += dt; feasible &= bool(state.feasible)
            if controller is None: ref = np.array(leg['poses_local_xyyaw'][0])
            else: _, ref = controller.reference.at(controller.reference.progress); ref = np.array(ref)
            path_error = float(np.linalg.norm(pose[:2]-ref[:2])); heading = math.degrees(wrap(pose[2]-ref[2]))
            goal = np.array(leg['fixed_goal_local_xyyaw'])
            trace.append(dict(time_s=now, phase=phase, leg_index=1, command_mode=leg['mode'],
                command_speed_m_s=speed, requested_curvature_1_m=k,
                x_m=float(pose[0]), y_m=float(pose[1]), yaw_rad=float(pose[2]),
                front_rad=float(response.actual[0]), rear_rad=float(response.actual[1]),
                command_front_rad=commanded[0], command_rear_rad=commanded[1],
                reference_position_error_m=path_error, reference_heading_error_deg=heading,
                terminal_position_error_m=float(np.linalg.norm(pose[:2]-goal[:2])),
                terminal_heading_error_deg=math.degrees(wrap(pose[2]-goal[2])),
                model_body_speed_m_s=float(np.linalg.norm(velocity[:2])),
                phase_time_s=now-tail if tail is not None else 0.))
            if path_error > .080 or abs(heading) > 10.: raise RuntimeError('Existing 80mm/10deg spatial-reference guard')
            if tail is not None and now-tail >= 5.-1e-10: break
        else: raise TimeoutError('90s MODEL budget')
    except (ValueError, RuntimeError, TimeoutError) as exc: error = f'{type(exc).__name__}: {exc}'
    fixed = [r for r in trace if r['phase'] == 'model_zero_command_tail_NOT_PhysX_hold' and r['phase_time_s'] >= 2.]
    position = float(np.linalg.norm(pose[:2]-np.array(leg['fixed_goal_local_xyyaw'][:2])))
    heading = math.degrees(wrap(pose[2]-leg['fixed_goal_local_xyyaw'][2]))
    checks = dict(sequence_completed=error is None and tail is not None,
        position_under_30mm=position <= .030, heading_under_2deg=abs(heading) <= 2.,
        fixed_tail_complete=bool(fixed) and fixed[-1]['time_s']-fixed[0]['time_s'] >= 2.9,
        fixed_tail_position_under_30mm=bool(fixed) and all(r['terminal_position_error_m'] <= .030 for r in fixed),
        fixed_tail_heading_under_2deg=bool(fixed) and all(abs(r['terminal_heading_error_deg']) <= 2. for r in fixed),
        fixed_tail_speed_under_5mm_s=bool(fixed) and all(r['model_body_speed_m_s'] <= .005 for r in fixed),
        actual_neutral_under_2p5deg=max(abs(a) for a in response.actual) <= math.radians(2.5),
        existing_force_limits=bool(feasible), command_slew_under_8deg_s=maximum_slew <= math.radians(8.)+1e-9,
        command_acceleration_under_20mm_s2=maximum_acceleration <= .020+1e-9,
        forward_commands_under_50mm_s=all(0 <= r['command_speed_m_s'] <= .050+1e-9 for r in trace))
    return dict(dt_s=dt, residual_case=residual_case, error=error, checks={k:bool(v) for k,v in checks.items()},
        duration_s=now, initial_actual_local_xyyaw=initial_pose.tolist(),
        initial_actual_body_velocity_with_pose_derived_yaw_rate=initial_velocity.tolist(),
        initial_actual_steering_rad=list(steer), synthetic_additive_steering_residual_deg=list(residual),
        final_model_local_xyyaw=pose.tolist(), terminal_position_error_m=position, terminal_heading_error_deg=heading,
        fixed_goal_local_xyyaw=leg['fixed_goal_local_xyyaw'], model_parameters=model_bounds(), tracking_gains=TRACKING_GAINS,
        new_next_leg_reference_adapter_MODEL_only=True, same_inherited_course_step_law=True,
        speed_profile_is_progress_based_course_controller_NOT_saved_nominal_timing_replay=True,
        caster_stiction_captured_hold_and_live_handoff_NOT_modelled=True,
        synthetic_residual_is_NOT_identified_uncertainty_or_robustness_proof=True,
        PhysX_executed=False, automatic_motion_authorized=False), trace


def load_review(directory):
    path = Path(directory)/'post_reverse_review.json'; report = json.loads(path.read_bytes())
    verify_sources(report); reverse.verify_files(report['input_sha256'])
    reverse.verify_files({c['trace_file']:c['trace_sha256'] for c in report['cases']})
    if (report['version'] != VERSION or report['reverse_result'] != str(RESULT)
            or report['reverse_result_sha256'] != reverse.sha(RESULT)
            or report['PhysX_executed'] is not False or report['automatic_motion_authorized'] is not False
            or {(c['dt_s'], c['residual_case']) for c in report['cases']} != {(d,r) for d in STEPS for r in RESIDUALS}
            or len(report['cases']) != 4):
        raise ValueError('Complete new post-reverse read-only MODEL review required')
    expected = all(all(c['checks'].values()) for c in report['cases'])
    if expected != report['all_next_forward_model_cases_pass']:
        raise ValueError('Stored next-forward MODEL summary disagrees with case checks')
    primary = json.loads(RESULT.read_bytes())
    candidate = json.loads(Path(report['suffix_candidate']).read_bytes())
    if (report['actual_start_world_xyyaw'] != primary['final_world_xyyaw']
            or report['retained_all_joint_positions_rad'] != reverse.joints_from_last(primary)
            or report['original_mission_goal_xyyaw'] != primary['mission_goal_world_xyyaw_UNCHANGED']
            or report['original_mission_start_xyyaw'] != primary['original_mission_start_xyyaw']
            or report['suffix_candidate'] != primary['suffix_candidate']
            or report['suffix_candidate_sha256'] != primary['suffix_candidate_sha256']
            or report['strict_map_digest'] != candidate['strict_map_digest']
            or report['next_leg'] != next_leg(candidate)):
        raise ValueError('Actual reverse endpoint, twelve joints, fixed next cusp, goal or map changed')
    return report


def main():
    # CLI-only siblings are intentionally lazy: the embedded Kit private
    # loader resolves hospital_* helpers, not arbitrary audit_/review_ names.
    from audit_hospital_h3b_directional_playback import collision_diagnostic
    from review_hospital_h3b_reserve_execution import save_trace

    info, audit, manifest, _, _ = inspect_reverse()
    candidate = json.loads(Path(info['suffix_candidate']).read_bytes()); leg = next_leg(candidate)
    with np.load(manifest['occupancy_grid']) as grid:
        occupied = grid['occupied'].copy(); origin = grid['origin_xy_m'].copy(); resolution = float(grid['resolution_m'])
    checker = NativeHospitalEnvelopeChecker(occupied, origin, resolution, info['mission_start_xyyaw'])
    actual_pose = relative_pose(info['initial_world_xyyaw'], info['mission_start_xyyaw'])
    if checker([actual_pose]) or checker(leg['poses_local_xyyaw']):
        raise ValueError('Actual post-reverse pose or frozen nominal next leg violates strict +10mm envelope')
    number = 1
    while (output := BASE/f'review_r{number:02d}').exists(): number += 1
    output.mkdir(parents=True)
    dependencies = [Path(__file__), Path(__file__).with_name('hospital_h3b_course_tracking.py'),
        Path(__file__).with_name('hospital_h3b_first_leg_feedback_review.py'),
        reverse.WORKSPACE/'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_reverse.py']
    header = dict(version=VERSION, role='POST_REVERSE_RAW_PASS_AND_NEXT_FORWARD_MODEL_ONLY',
        reverse_result=info['reverse_result'], reverse_result_sha256=info['reverse_result_sha256'],
        raw_reverse_audit=audit, actual_start_world_xyyaw=info['initial_world_xyyaw'],
        retained_all_joint_positions_rad=info['retained_all_joint_positions_rad'],
        original_mission_start_xyyaw=info['mission_start_xyyaw'], original_mission_goal_xyyaw=info['mission_goal_xyyaw'],
        strict_map_digest=info['strict_map_digest'], suffix_candidate=info['suffix_candidate'],
        suffix_candidate_sha256=info['suffix_candidate_sha256'], next_leg=leg,
        actual_start_deviation_from_nominal_cusp_m=float(np.linalg.norm(np.array(actual_pose[:2])-leg['poses_local_xyyaw'][0][:2])),
        actual_start_pose_and_nominal_next_leg_envelopes_clear=True,
        no_pose_caster_goal_or_path_reset=True, previous_FAILs_and_new_primary_PASS_preserved=True,
        next_stage='READ_ONLY_LIVE_STATE_REVIEW_THEN_DEDICATED_FORWARD_EXECUTOR_REVIEW_NOT_DRIVE',
        input_sha256={**info['input_sha256'], **{str(p):reverse.sha(p) for p in dependencies}},
        source_sha256={p.name:reverse.sha(p) for p in dependencies},
        PhysX_executed=False, automatic_motion_authorized=False, full_H3B_turning_path_verified=False,
        use_for_final_statistics=False)
    cases = []
    for residual in RESIDUALS:
        for dt in STEPS:
            result, trace = model_case(info, leg, dt, residual)
            diag = collision_diagnostic(trace, checker, occupied, origin, resolution, info['mission_start_xyyaw'])
            result['sampled_envelope_diagnostic'] = diag
            result['checks']['strict_10mm_model_envelope_clear'] = diag['sampled_model_envelope_clear']
            save_trace(output, f'{residual}_{round(dt*1000)}ms', result, trace); cases.append(result)
            print('NEXT FORWARD MODEL', residual, dt, 'ADMIT' if all(result['checks'].values()) else 'HOLD',
                'mm/deg:', round(1000*result['terminal_position_error_m'],3), round(result['terminal_heading_error_deg'],6),
                'failed:', [k for k,v in result['checks'].items() if not v], flush=True)
    admitted = all(all(c['checks'].values()) for c in cases)
    report = dict(header, cases=cases, all_next_forward_model_cases_pass=admitted,
        status='NEXT_FORWARD_MODEL_SCREEN_PASSED_LIVE_AND_EXECUTOR_PENDING' if admitted else 'HOLD_NEXT_FORWARD_MODEL_SCREEN')
    reverse.verify_files(header['input_sha256'])
    with (output/'post_reverse_review.json').open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print('RAW FIRST REVERSE:', audit['original_verdict_unchanged'], '; next forward length m:', round(leg['length_m'],6))
    print('STATUS:', report['status']); print('saved:', output/'post_reverse_review.json')
    print('Keep the SAME actual reverse-PASS scene PAUSED. No motion, reset, refit, new goal or whole-H3 result.')


if __name__ == '__main__': main()
