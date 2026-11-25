#!/usr/bin/env python3
"""Nominal closed-loop MODEL diagnostics of FOUR frozen H4 references.

Original v3 control, legacy steering bias/rate and Newton-Euler bounds only.
No actual body/joints, fit, gain change, caster assignment or Isaac control.
Physical captured-wheel hold forces, caster swivel/stiction and cooked contact
are NOT simulated. Even 8/8 software cases would NOT admit a physical drive.
"""

from collections import deque
import csv
import json
import math
from pathlib import Path

import numpy as np

import build_hospital_h4_execution_bundle as bundle
import hospital_h4_path_execution as execution
from hospital_geometry_tools import pose_collision
from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
from patient_initial_stability.steering_lattice import (MODE_CURVATURE_RESPONSE_SCALE,
    MODE_LATERAL_RESPONSE_SCALE, STEERING_TRACKING_BIAS_RAD)


OUTPUT = bundle.cohort.GATE/'hospital_h4_common_executor_screen/screen_r01'
TARGET = OUTPUT/'common_executor_model_screen.json'
VERSION = 'H4_shared_original_v3_nominal_legacy_response_model_diagnostic_v1'
TIMESTEPS_S = (.020, .010)


def case(plan, bounds, dt, occupied, origin, resolution, design):
    if dt not in TIMESTEPS_S:
        raise ValueError('Only the prescribed 10/20ms resolution screens are allowed')
    controller = execution.H4TrackingController(plan)
    model = MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
    pose = np.asarray(plan['start_world_xyyaw'], float).copy()
    velocity, actual, commanded = np.zeros(3), np.zeros(2), (0., 0.)
    covers = bundle.cohort.support_covers(design)
    now, next_record, hold_start, last_command_time = 0., 0., None, 2.
    rows, events, history = [], [], deque([(now, pose.copy())])
    maximum_path_error = maximum_path_heading = maximum_friction = maximum_ltr = maximum_actuator = 0.
    minimum_load, maximum_command_slew = math.inf, 0.
    feasible, complete, error = True, False, None
    last_guard = pose.copy()
    info = dict(speed_m_s=0., mode=plan['selected_actions'][0]['mode'], curvature_1_m=0.,
        terminal_stage='pre_hold', drive_hold_requested=True, progress_m=0., path_error_m=0., path_heading_error_deg=0.)

    def record():
        nonlocal next_record
        e = execution.relative_pose(plan['goal_world_xyyaw'], pose)
        rows.append(dict(time_s=now, phase='model_hold_NOT_physical_hold' if hold_start is not None else info['terminal_stage'],
            x_m=float(pose[0]), y_m=float(pose[1]), yaw_rad=float(pose[2]),
            u_m_s=float(velocity[0]), v_m_s=float(velocity[1]), yaw_rate_rad_s=float(velocity[2]),
            command_speed_m_s=float(info['speed_m_s']), command_mode=info['mode'],
            command_curvature_1_m=float(info['curvature_1_m']), command_front_rad=float(commanded[0]),
            command_rear_rad=float(commanded[1]), front_rad=float(actual[0]), rear_rad=float(actual[1]),
            requested_drive_hold=bool(info['drive_hold_requested']), reference_progress_m=float(controller.reference.progress),
            goal_position_error_m=math.hypot(e[0], e[1]), goal_heading_error_deg=math.degrees(e[2])))
        next_record = now+.05

    try:
        record()
        while True:
            speed = float(np.linalg.norm(velocity[:2]))
            filtered = (float(np.linalg.norm(pose[:2]-history[0][1][:2]))/(now-history[0][0])
                if now > history[0][0] else speed)
            measured = max(speed, filtered)
            if now >= 2.-1e-10 and hold_start is None:
                if controller.previous_time is None:
                    controller.previous_time = now
                info = controller.step(now, pose, measured, tuple(actual))
                maximum_path_error = max(maximum_path_error, info['path_error_m'])
                maximum_path_heading = max(maximum_path_heading, abs(info['path_heading_error_deg']))
                if info['drive_hold_requested'] and (abs(info['speed_m_s']) > 1e-7 or measured > .005):
                    raise RuntimeError('MODEL drive-hold request before commanded/measured stop')
                if info['hold_ready']:
                    hold_start = now
                    events.append(dict(event='MODEL_fixed_hold_entry_NOT_physical_acknowledgement', time_s=now,
                        controller_entry_candidate=controller.hold_entry_candidate, pulse_count=controller.pulse_count))
                else:
                    command_dt = max(0., now-last_command_time)
                    old = commanded
                    commanded, _ = execution.direct_joint_targets(info['speed_m_s'], info['mode'], info['curvature_1_m'],
                        commanded, command_dt, actual)
                    if command_dt:
                        maximum_command_slew = max(maximum_command_slew,
                            *(abs(a-b)/command_dt for a, b in zip(commanded, old)))
                    last_command_time = now
            if hold_start is not None:
                info = {**info, 'speed_m_s': 0., 'drive_hold_requested': True}
                if now-hold_start >= 5.-1e-10:
                    complete = True; break
            if now+dt > execution.GUARDS['maximum_simulation_duration_s']+1e-9:
                raise TimeoutError('Inherited 180s MODEL whole-trial resource limit; no budget increase')
            target = np.asarray(commanded)+[STEERING_TRACKING_BIAS_RAD[a] for a in ('front', 'rear')]
            middle = actual+np.clip(target-actual, -math.radians(8.)*dt/2, math.radians(8.)*dt/2)
            actual = middle+np.clip(target-middle, -math.radians(8.)*dt/2, math.radians(8.)*dt/2)
            tf, tr = np.tan(middle)
            curvature = (tf-tr)/1.29*MODE_CURVATURE_RESPONSE_SCALE[info['mode']]
            lateral = (.645*tf+.645*tr)/1.29*MODE_LATERAL_RESPONSE_SCALE[info['mode']]
            # Hold is zero-target legacy NE settling, NEVER instant velocity
            # cancellation or a fictitious captured-wheel force model.
            step = model.step(*velocity, info['speed_m_s'], info['speed_m_s']*lateral,
                info['speed_m_s']*curvature, dt)
            next_velocity = np.array([step.longitudinal_velocity_m_s, step.lateral_velocity_m_s, step.yaw_rate_rad_s])
            u, v, yaw_rate = (velocity+next_velocity)/2
            pose += dt*np.array([u*math.cos(pose[2])-v*math.sin(pose[2]),
                u*math.sin(pose[2])+v*math.cos(pose[2]), yaw_rate])
            velocity = next_velocity; now += dt
            feasible = feasible and bool(step.feasible)
            maximum_friction = max(maximum_friction, step.friction_utilization)
            maximum_ltr = max(maximum_ltr, step.abs_ltr)
            maximum_actuator = max(maximum_actuator, step.actuator_utilization)
            minimum_load = min(minimum_load, step.minimum_normal_load_fraction)
            if not step.feasible:
                raise RuntimeError('Inherited force/load MODEL feasibility constraint')
            if float(np.linalg.norm(velocity[:2])) > execution.GUARDS['body_speed_m_s']:
                raise RuntimeError('Inherited MODEL body speed guard')
            if controller.phase != 'path_tracking' and np.linalg.norm(pose[:2]-controller.goal[:2]) > .12:
                raise RuntimeError('Inherited MODEL terminal 120mm neighbourhood guard')
            displacement = float(np.linalg.norm(pose[:2]-last_guard[:2]))
            turn = execution.wrap(float(pose[2]-last_guard[2]))
            if displacement >= .005-1e-12 or abs(turn) >= math.radians(.25)-1e-12 or hold_start is not None:
                count = max(1, math.ceil(displacement/.005), math.ceil(abs(turn)/math.radians(.25)))
                for j in range(1, count+1):
                    t = j/count; p = [last_guard[0]+t*(pose[0]-last_guard[0]), last_guard[1]+t*(pose[1]-last_guard[1]), last_guard[2]+t*turn]
                    if pose_collision(occupied, p, origin, resolution, clearance=.010):
                        raise RuntimeError('MODEL full-envelope authored-raster guard')
                    if bundle.cohort.support_area_missing(p, covers) > 1e-9:
                        raise RuntimeError('MODEL TRUE full-footprint authored-support guard')
                last_guard = pose.copy()
            history.append((now, pose.copy()))
            while len(history) > 2 and history[1][0] <= now-.20:
                history.popleft()
            if now >= next_record-1e-10:
                record()
    except (RuntimeError, TimeoutError, ValueError) as failure:
        error = type(failure).__name__+': '+str(failure)
    if not rows or rows[-1]['time_s'] != now:
        record()
    final = execution.relative_pose(plan['goal_world_xyyaw'], pose)
    tail = [r for r in rows if hold_start is not None and r['time_s'] >= hold_start+2.-1e-9]
    regression = drift = None
    if len(tail) >= 10:
        times = np.array([r['time_s'] for r in tail]); centered = times-times.mean()
        xy = np.array([[r['x_m'], r['y_m']] for r in tail])
        regression = float(np.linalg.norm(centered@(xy-xy.mean(axis=0))/(centered@centered)))
        drift = float(np.max(np.linalg.norm(xy-xy[0], axis=1)))
    checks = dict(reference_and_ONE_model_hold_completed=complete,
        at_least_95_percent_reference_progress=controller.reference.progress >= .95*controller.reference.arc[-1],
        terminal_position_under_30mm=math.hypot(final[0], final[1]) <= .030,
        terminal_heading_under_2deg=abs(math.degrees(final[2])) <= 2.,
        final_horizontal_speed_under_5mm_s=float(np.linalg.norm(velocity[:2])) <= .005,
        fixed_tail_at_least_2p9s=bool(tail and tail[-1]['time_s']-tail[0]['time_s'] >= 2.9),
        fixed_tail_every_position_under_30mm=bool(tail and all(r['goal_position_error_m'] <= .030 for r in tail)),
        fixed_tail_every_heading_under_2deg=bool(tail and all(abs(r['goal_heading_error_deg']) <= 2. for r in tail)),
        fixed_tail_regression_under_1mm_s=regression is not None and regression <= .001,
        fixed_tail_drift_under_3mm=drift is not None and drift <= .003,
        inherited_force_load_MODEL_constraints=bool(feasible),
        commanded_steering_slew_under_8deg_s=maximum_command_slew <= math.radians(8.)+1e-9,
        no_inherited_guard_error=error is None)
    report = dict(method=plan['method'], integration_dt_s=dt, model_status='MODEL_SCREEN_PASS' if all(checks.values()) else 'MODEL_SCREEN_HOLD',
        checks={k: bool(v) for k, v in checks.items()}, error=error, simulation_duration_s=now,
        final_position_error_m=math.hypot(final[0], final[1]), final_heading_error_deg=math.degrees(final[2]),
        final_world_xyyaw=pose.tolist(), terminal_pulse_count=controller.pulse_count,
        hold_entry_model_time_s=hold_start, model_tail_regression_speed_m_s=regression,
        model_tail_drift_m=drift, maximum_tracking_error_m=maximum_path_error,
        maximum_tracking_heading_error_deg=maximum_path_heading, maximum_friction_utilization=maximum_friction,
        maximum_abs_ltr=maximum_ltr, maximum_actuator_utilization=maximum_actuator,
        minimum_normal_load_fraction=minimum_load if math.isfinite(minimum_load) else None,
        events=events, physical_drive_hold_caster_stiction_and_contact_NOT_simulated=True,
        native_model_not_independently_validated_in_this_H4_domain=True,
        physical_motion_or_whole_route_admission=False, use_for_final_statistics=False)
    return report, rows


def main():
    if TARGET.exists():
        report = json.loads(TARGET.read_bytes()); bundle.cohort.verify(report['input_sha256']); bundle.cohort.verify(report['output_sha256'])
        print('Existing immutable H4 common-executor MODEL screen verified:', TARGET)
        return
    frozen = json.loads(bundle.TARGET.read_bytes())
    bundle.cohort.verify(frozen['input_sha256']); bundle.cohort.verify(frozen['output_sha256'])
    plans = [json.loads(Path(path).read_bytes()) for path in frozen['plans']]
    native = json.loads(bundle.cohort.MODELS.read_bytes())
    design = bundle.cohort.support.load_design()
    with np.load(bundle.cohort.MAP) as grid:
        occupied, origin, resolution = grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m'])
    OUTPUT.mkdir(parents=True, exist_ok=False)
    cases, paths = [], []
    for plan in plans:
        for dt in TIMESTEPS_S:
            report, rows = case(plan, native['rigid_body_parameter_bounds'], dt, occupied, origin, resolution, design)
            path = OUTPUT/(plan['method']+'_'+str(round(1000*dt))+'ms.csv'); paths.append(path)
            with path.open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            report.update(model_trace=str(path), model_trace_sha256=bundle.cohort.completion.sha(path)); cases.append(report)
            print(plan['method'], dt, report['model_status'], 'mm/deg:', 1000*report['final_position_error_m'],
                report['final_heading_error_deg'], 's:', report['simulation_duration_s'], 'error:', report['error'], flush=True)
    count = sum(c['model_status'] == 'MODEL_SCREEN_PASS' for c in cases)
    report = dict(version=VERSION, status='NOMINAL_SOFTWARE_MODEL_SCREEN_ONLY_NOT_PHYSICAL_ADMISSION',
        cases=cases, nominal_software_MODEL_PASS_count=count, case_count=len(cases),
        original_models_parameters_v3_controller_guards_and_acceptance_unchanged=True,
        physical_hold_force_caster_swivel_stiction_and_cooked_contacts_NOT_simulated=True,
        software_step_resolution_check_is_NOT_independent_data_or_robustness=True,
        prior_whole_H3_MODEL_PASS_count=2, prior_whole_H3_MODEL_case_count=8,
        old_PASS_FAIL_HOLDs_retained=True, live_stage_or_robot_accessed=False,
        new_PhysX_executed=False, CICS_runtime_integration=False, actual_drive_admission=False, use_for_final_statistics=False,
        input_sha256={**frozen['input_sha256'], **frozen['output_sha256'], str(bundle.TARGET): bundle.cohort.completion.sha(bundle.TARGET),
            str(Path(__file__)): bundle.cohort.completion.sha(Path(__file__))},
        output_sha256={str(path): bundle.cohort.completion.sha(path) for path in paths})
    with TARGET.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('saved:', TARGET)
    print('No physical admission: MODEL resolution cases are not validation/robustness or new H3 success.')


if __name__ == '__main__':
    main()
