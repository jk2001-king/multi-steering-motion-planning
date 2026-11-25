"""Separate FINAL forward MODEL: same course/timing step, planned cap30.

No controller refit, reference recapture, new feedback law, CICS or scene action.
Evaluate the EXACT original mission goal, not a newly selected intermediate
target. All eight predeclared synthetic DESIGN cases are retained.
"""

import math
import numpy as np

import hospital_h3b_post_second_reverse as data
from hospital_h3b_post_second_reverse import RESULT, STEPS, RESIDUALS, SCALES, read_trace, model_bounds
from hospital_h3b_timed_second_reverse import terminal_hold_checks, arrival_checks
from hospital_h3b_reverse_closedloop import terminal_errors
from hospital_timed_path_execution import SpatialLeg, LegTiming, execution_contract
from hospital_h3b_timed_course_speed_profile import CappedTimedCourseForward as TimedCourseForward, TRACKING_GAINS, VERSION as COMPOSITION_VERSION, PLANNED_SPEED_CAP_M_S

forward = data.reverse.previous.forward
FrozenSteeringResponse = data.reverse.previous.FrozenSteeringResponse
load_validated_response = data.reverse.previous.load_validated_response
motion_stalled = forward.motion_stalled


def forward_episode_progress(rows):
    moving = [r for r in rows if r['phase'] in ('forward_motion', 'forward_braking')
        and not r['position_hold_enabled'] and r['command_speed_m_s'] > .003]
    return moving[-1]['reference_progress_m']-moving[0]['reference_progress_m'] if moving else 0.


def model_case(info,dt,residual_case,response_scale):
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import MODE_CURVATURE_RESPONSE_SCALE,MODE_LATERAL_RESPONSE_SCALE
    if dt not in STEPS or residual_case not in RESIDUALS or response_scale not in SCALES:
        raise ValueError('Only the eight declared clocks/residuals/synthetic scales may be screened')
    leg=dict(info['final_leg']);mode=leg['mode'];frozen,_=load_validated_response();last=info['expected_saved_last_state']
    pose=np.array(forward.relative_pose(info['initial_world_xyyaw'],info['mission_start_xyyaw']))
    u,v=forward.body_velocity((last['vx_physx_m_s'],last['vy_physx_m_s']),last['yaw_rad'])
    previous=read_trace(RESULT.with_suffix('.csv'))[-2]
    yaw_rate=forward.wrap(last['yaw_rad']-previous['yaw_rad'])/(last['time_s']-previous['time_s'])
    velocity=np.array((u,v,yaw_rate));initial_pose=pose.copy();initial_velocity=velocity.copy()
    initial_steer=(last['front_rad'],last['rear_rad'])
    residual=(0.,0.) if residual_case=='frozen_nominal' else tuple(map(math.degrees,initial_steer))
    response=FrozenSteeringResponse(frozen,initial_steer,residual_deg=residual)
    commanded=tuple(last['command_'+a+'_rad'] for a in ('front','rear'))
    if max(abs(v) for v in commanded)>1e-7:raise ValueError('Actual saved commands must already be neutral')
    body=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    reference=SpatialLeg(leg);goal=np.array(leg['exact_mission_goal_local_xyyaw'])
    now=0.;trace=[];feasible=True;max_slew=max_acceleration=previous_speed=0.
    controller=None;controller_ready=False;tail=None;error=None;entry_checks={};events=[]
    try:
        for _ in range(math.ceil(90./dt)):
            if now<2.75-1e-10:speed=k=0.;phase='pre_hold'
            else:
                if controller is None:controller=TimedCourseForward(leg,now)
                if tail is None:
                    if not controller_ready:
                        out=controller.step(now,pose,float(np.linalg.norm(velocity[:2])),response.actual,velocity[:2])
                        speed,k=out['speed_m_s'],out['curvature_1_m'];phase='forward_motion' if out['phase']=='tracking' else 'forward_braking'
                        if out['transition']:events.append(dict(time_s=now,**out))
                        controller_ready=out['ready']
                    else:speed=k=0.;phase='forward_braking'
                    if controller_ready:
                        speed=k=0.;phase='forward_braking'
                        gate=terminal_hold_checks(pose,goal,float(np.linalg.norm(velocity[:2])),response.actual,commanded)
                        if not gate['terminal_position_under_30mm'] or not gate['terminal_heading_under_2deg']:
                            raise RuntimeError('Original mission-goal acceptance lost during bounded neutral handoff')
                        if all(gate.values()):entry_checks=gate;tail=now
                        elif now-controller.braking_started>8.:raise TimeoutError('Existing 8s braking/neutral guard')
                if tail is not None:speed=k=0.;phase='terminal_hold'
            old=commanded
            commanded,_=forward.direct_joint_targets(speed,mode,k,commanded,dt,response.actual)
            max_slew=max(max_slew,*(abs(a-b)/dt for a,b in zip(old,commanded)))
            max_acceleration=max(max_acceleration,abs(speed-previous_speed)/dt);previous_speed=speed
            mid,_=response.step(commanded,dt);tf,tr=np.tan(mid)
            curvature=(tf-tr)/1.29*MODE_CURVATURE_RESPONSE_SCALE[mode]
            ratio=(tf+tr)/2*MODE_LATERAL_RESPONSE_SCALE[mode]
            target=speed*response_scale
            state=body.step(*velocity,target,target*ratio,target*curvature,dt)
            new=np.array((state.longitudinal_velocity_m_s,state.lateral_velocity_m_s,state.yaw_rate_rad_s))
            ux,vy,rate=(velocity+new)/2;angle=pose[2]+rate*dt/2
            pose+=dt*np.array((ux*math.cos(angle)-vy*math.sin(angle),ux*math.sin(angle)+vy*math.cos(angle),rate))
            velocity=new;now+=dt;feasible&=bool(state.feasible)
            ref,path_error,_=reference.project(pose);heading=math.degrees(forward.wrap(pose[2]-ref[2]))
            position,yaw=terminal_errors(pose,goal)
            trace.append(dict(time_s=now,phase=phase,phase_time_s=now-tail if tail is not None else 0.,leg_index=3,
                command_mode=mode,command_speed_m_s=speed,requested_curvature_1_m=k,
                x_m=float(pose[0]),y_m=float(pose[1]),yaw_rad=float(pose[2]),
                front_rad=float(response.actual[0]),rear_rad=float(response.actual[1]),
                command_front_rad=commanded[0],command_rear_rad=commanded[1],reference_progress_m=reference.progress,
                position_hold_enabled=tail is not None,reference_position_error_m=path_error,reference_heading_error_deg=heading,
                terminal_position_error_m=position,terminal_heading_error_deg=yaw,model_body_speed_m_s=float(np.linalg.norm(velocity[:2])),
                controller_phase='pre_hold' if controller is None else ('post_ready_neutral' if controller_ready else controller.phase),
                scheduled_body_u_m_s=0. if controller is None else out['scheduled_body_u_m_s']))
            if path_error>.080 or abs(heading)>10.:raise RuntimeError('Existing 80mm/10deg curved-reference guard')
            if motion_stalled(trace):raise RuntimeError('Existing five-second reverse-progress stall')
            if tail is not None and now-tail>=5.-1e-10:break
        else:raise TimeoutError('Existing 90s model budget')
    except (ValueError,RuntimeError,TimeoutError) as exc:error=f'{type(exc).__name__}: {exc}'
    position,yaw=terminal_errors(pose,goal)
    fixed=[r for r in trace if r['phase']=='terminal_hold' and r['phase_time_s']>=2.]
    distance=sum(math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']) for a,b in zip(trace,trace[1:]))
    if len(fixed)>=2:
        t=np.array([r['time_s'] for r in fixed]);t-=t.mean();xy=np.array([[r['x_m'],r['y_m']] for r in fixed])
        regression=float(np.linalg.norm((t[:,None]*xy).sum(axis=0)/(t*t).sum()));drift=float(np.linalg.norm(xy-xy[0],axis=1).max())
    else:regression=drift=None
    checks=dict(sequence_completed=error is None and tail is not None,
        terminal_arrival_before_ONE_hold=bool(entry_checks) and all(entry_checks.values()),
        captured_hold_entered_with_exact_command_neutral=entry_checks.get('commanded_steering_exactly_neutral',False),
        position_under_30mm=position<=.030,heading_under_2deg=abs(yaw)<=2.,
        fixed_tail_complete=bool(fixed) and fixed[-1]['time_s']-fixed[0]['time_s']>=2.9,
        fixed_tail_position_under_30mm=bool(fixed) and all(r['terminal_position_error_m']<=.030 for r in fixed),
        fixed_tail_heading_under_2deg=bool(fixed) and all(abs(r['terminal_heading_error_deg'])<=2. for r in fixed),
        fixed_tail_speed_under_5mm_s=bool(fixed) and all(r['model_body_speed_m_s']<=.005 for r in fixed),
        fixed_tail_regression_under_1mm_s=regression is not None and regression<=.001,
        fixed_tail_drift_under_3mm=drift is not None and drift<=.003,
        existing_force_limits=bool(feasible),command_slew_under_8deg_s=max_slew<=math.radians(8.)+1e-9,
        command_acceleration_under_20mm_s2=max_acceleration<=.020+1e-9,
        only_forward_commands_under_50mm_s=all(0.<=r['command_speed_m_s']<=.050+1e-9 for r in trace),
        forward_distance_under_reference_plus_160mm=distance<=leg['reference_length_m']+.160,
        total_reverse_with_prior_measured_episode_under_1m=info['previous_measured_reverse_total_m']<=1.,
        next_direction_change_within_existing_four_limit=info['completed_direction_changes']+1<=forward.POLICY['maximum_direction_changes'],
        forward_progress_over_50mm=forward_episode_progress(trace)>.050,
        at_most_one_same_direction_terminal_reapproach=controller is not None and controller.reapproaches<=1)
    return dict(dt_s=dt,residual_case=residual_case,SYNTHETIC_body_target_scale=response_scale,error=error,
        checks={k:bool(v) for k,v in checks.items()},duration_s=now,initial_actual_local_xyyaw=initial_pose.tolist(),
        initial_actual_body_velocity_with_pose_derived_yaw_rate=initial_velocity.tolist(),initial_actual_steering_rad=list(initial_steer),
        synthetic_additive_steering_residual_deg=list(residual),fixed_next_cusp_local_xyyaw=goal.tolist(),
        terminal_position_error_m=position,terminal_heading_error_deg=yaw,model_measured_center_distance_m=distance,
        previous_measured_reverse_total_m=info['previous_measured_reverse_total_m'],terminal_events=events,
        maximum_terminal_reapproaches=controller.reapproaches if controller else 0,
        model_parameters=model_bounds(),model_contract=execution_contract(),
        nominal_timing=TimedCourseForward(leg,0.).timing.report(),
        planned_speed_cap_m_s=PLANNED_SPEED_CAP_M_S,physical_speed_cap_m_s=.050,
        same_existing_course_step_and_gains_no_override=True,
        exact_command_neutral_before_hold_modelled=True,
        NEW_shared_timing_and_braking_curvature_sequence=True,
        both_changes_are_combined_NOT_a_separate_causal_ablation=True,
        NEW_composition_existing_forward_course_equation_with_shared_timing=True,
        composition_version=COMPOSITION_VERSION,tracking_gains=TRACKING_GAINS,
        no_new_gain_fit_or_reverse_feedback_transplant=True,actual_endpoint_NOT_reset_to_nominal_cusp=True,
        synthetic_cases_NOT_fitted_uncertainty_or_statistical_robustness=True,
        caster_stiction_captured_hold_and_measured_handoff_NOT_modelled=True,
        final_forward_PhysX_validation_still_pending=True,
        PhysX_executed=False,automatic_motion_authorized=False),trace
