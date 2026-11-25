"""Raw next-forward PASS audit and SECOND reverse MODEL screen, no motion.

Keep actual endpoint/12 joints/caster history. Extract the fixed front-only
reverse leg without recapturing any cusp/goal. Reuse the measured-terminal
reverse controller, explicitly in a NEW mode/length/history domain. No PhysX
admission is implied by nominal or synthetic model screens.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_next_forward as forward
from hospital_h3b_next_forward_observation import coverage
from hospital_h3b_rolling_force_audit import numeric_rows, anchor_sums_match
from hospital_h3b_reverse_closedloop import (FirstReverseProgress, FirstReverseReference,
    arrival_checks, reverse_episode_progress, motion_stalled, terminal_errors, execution_contract)
from hospital_h3b_response_link import FrozenSteeringResponse, load_validated_response
from hospital_h3_flush_support import load_setup
from hospital_h3_turn_planning import model_bounds, verify_sources
from hospital_native_envelope import NativeHospitalEnvelopeChecker


WORKSPACE, GATE = forward.WORKSPACE, forward.GATE
RESULT = forward.OUTPUT/'hospital_h3b_next_forward_r01.json'
BASE = GATE/'hospital_h3b_post_forward_reviews'
VERSION = 'H3B_actual_next_forward_PASS_second_reverse_MODEL_review_v1'
STEPS = (.02,.01)
RESIDUALS = ('frozen_nominal','observed_neutral_offset_SYNTHETIC')
SCALES = (1.,.6)
sha, verify_files, read_trace = forward.sha, forward.verify_files, forward.read_trace


def extract_second_reverse(candidate):
    g,t=candidate['geometry_audit'],candidate['nominal_timing']
    if (g['leg_directions']!=[-1,1,-1,1]
            or g['leg_modes']!=['rear_only','coordinated_counterphase','front_only','coordinated_counterphase']
            or g.get('verdict')!='PASS' or not all(g['checks'].values()) or not all(t['checks'].values())):
        raise ValueError('Need the frozen same rear-first four-leg suffix, not a new selected family')
    left,right=t['cusp_sample_indices'][1:3]
    arc=np.array(g['travelled_arc_m'][left:right+1],float);arc-=arc[0]
    leg=dict(original_suffix_leg_index=2,original_sample_indices_inclusive=[left,right],
        mode='front_only',direction=-1,poses_local_xyyaw=g['poses_local_xyyaw'][left:right+1],
        center_arc_m=arc.tolist(),curvature_1_m=g['sample_curvature_1_m'][left:right+1],
        next_cusp_local_xyyaw=g['poses_local_xyyaw'][right],reference_length_m=float(arc[-1]))
    if (not 0<left<right or any(v!=-1 for v in g['interval_directions'][left:right])
            or any(v!='front_only' for v in g['interval_modes'][left:right])):
        raise ValueError('Only the complete SECOND front-only reverse leg is screened')
    FirstReverseReference(leg)
    return leg


def inspect_forward():
    info=forward.load_admitted_profile();record=json.loads(RESULT.read_bytes())
    verify_sources(record);verify_files(record['frozen_input_sha256'])
    if (record.get('trial_kind')!=forward.ROLE or record.get('verdict')!='PASS'
            or record.get('PhysX_executed') is not True or record.get('error') is not None
            or record.get('original_motor_gains_restored') is not True or not record.get('checks')
            or not all(record['checks'].values()) or record.get('next_forward_component_verified') is not True
            or record.get('full_suffix_execution_verified') is not False
            or record['initial_saved_world_xyyaw']!=info['initial_world_xyyaw']
            or record['expected_retained_joint_positions_rad']!=info['retained_all_joint_positions_rad']
            or record['fixed_next_cusp_local_xyyaw']!=info['next_cusp_local_xyyaw']
            or record['prior_reverse_result_sha256']!=info['reverse_result_sha256']
            or record['strict_map_digest']!=info['strict_map_digest']
            or record['suffix_candidate_sha256']!=info['suffix_candidate_sha256']
            or record['mission_goal_world_xyyaw_UNCHANGED']!=info['mission_goal_xyyaw']
            or record['execution_contract']!=info['intervention_contract']
            or record['physics_preparation_snapshot']!=info['preparation_record']['physics_preparation_snapshot']):
        raise ValueError('Exact NEW actual next-forward PASS required; no prefix/failed-state substitution')
    rows=read_trace(RESULT.with_suffix('.csv'))
    if len(rows)!=record['sample_count'] or rows[-1]!=record['last_recorded_state']:
        raise ValueError('Actual next-forward CSV/last state disagree')
    _,manifest,geometry,inventory=load_setup(Path(record['geometry_manifest']).parent,WORKSPACE)
    actual=forward.evaluate_component(rows,record['target_world_xyyaw'],read_trace(Path(record['static_reference']).with_suffix('.csv')),
        record['physics_preparation_snapshot']['simulation_mass_kg'],inventory['explicitly_suppressed_assets'][0]['collision']['bound'],
        record['wheel_conservative_sphere_radii_m'],record['contact_report_counts_by_category'],record['contact_callback_errors'],
        record['fixed_hold_attempt_count'],record['direction_gate_events'],info['reference_length_m'])
    if actual['verdict']!='PASS' or any(record['checks'].get(k)!=v for k,v in actual['checks'].items()):
        raise ValueError('Raw next-forward 30mm/2deg, fixed hold, support and stop PASS does not reproduce')
    c=forward.ForwardProgress(info,record['direction_gate_events'][0]['time_s'])
    command_error=steer_error=wheel_error=0.;ready_count=0
    for previous,row in zip(rows,rows[1:]):
        first_hold=row['phase']=='terminal_hold' and previous['phase']!='terminal_hold'
        if row['phase'] not in ('forward_motion','forward_braking') and not first_hold:continue
        pose=[previous[k] for k in ('x_m','y_m','yaw_rad')]
        out=c.step(row['command_time_s'],forward.relative_pose(pose,info['mission_start_xyyaw']),
            max(previous['pose_filtered_speed_m_s'],math.hypot(previous['vx_physx_m_s'],previous['vy_physx_m_s'])),
            (previous['front_rad'],previous['rear_rad']),
            forward.body_velocity((previous['vx_physx_m_s'],previous['vy_physx_m_s']),pose[2]))
        command_error=max(command_error,abs(out['speed_m_s']-row['command_speed_m_s']),
            abs(out['curvature_1_m']-row['requested_curvature_1_m']))
        angles,wheels=forward.direct_joint_targets(out['speed_m_s'],'coordinated_counterphase',out['curvature_1_m'],
            (previous['command_front_rad'],previous['command_rear_rad']),row['command_time_s']-previous['command_time_s'],
            (previous['front_rad'],previous['rear_rad']))
        steer_error=max(steer_error,*(abs(angles[i]-row['command_'+a+'_rad']) for i,a in enumerate(('front','rear'))))
        wheel_error=max(wheel_error,*(abs(wheels[i]-row['drive_'+a+'_actual_velocity_target_rad_s']) for i,a in enumerate(('front','rear'))))
        ready_count+=int(out['ready'])
    if max(command_error,steer_error)>1e-10 or wheel_error>1e-6 or ready_count!=1:
        raise ValueError('Next-forward course/steering/applied wheel targets or ONE arrival do not reproduce')
    observer_path=forward.COHORT/'instrumentation/instrumentation.json';observer=json.loads(observer_path.read_bytes())
    verify_sources(observer);verify_files(observer['reference_input_sha256'])
    files={str(observer_path.parent/n):v for n,v in observer['saved_file_sha256'].items()};verify_files(files)
    if observer['original_controller_result_sha256']!=sha(RESULT) or observer['telemetry_errors']:
        raise ValueError('Primary/observer lineage or telemetry error')
    j=numeric_rows(observer_path.parent/'joint_body_solver_trace.csv');f=numeric_rows(observer_path.parent/'wheel_friction_summary.csv')
    with (observer_path.parent/'friction_anchors.csv').open(newline='') as stream:anchors=list(csv.DictReader(stream))
    with (observer_path.parent/'raw_contact_points.csv').open(newline='') as stream:
        contacts=[dict(r,complete_point_data=r['complete_point_data']=='True') for r in csv.DictReader(stream)]
    covered=coverage(j,f,anchors,contacts,record,[])
    if any(observer.get(k)!=v for k,v in covered.items()):raise ValueError('Raw observer coverage mismatch')
    anchor_error=anchor_sums_match(f,anchors,observer['native_sensor_paths_in_recorded_wheel_index_order'])
    joints=forward.reverse.joints_from_last(record)
    if len(joints)!=12:raise ValueError('All 12 actual post-forward joints required')
    candidate=json.loads(Path(info['suffix_candidate']).read_bytes());leg=extract_second_reverse(candidate)
    previous_reverse=info['reverse_record']['measured_center_distance_m']
    # Both preceding measured gate records were independently reproduced:
    # forward -> first reverse -> this forward = two direction changes.
    completed_changes=len(info['reverse_record']['direction_gate_events'])+len(record['direction_gate_events'])
    info.update(forward_result=str(RESULT),forward_result_sha256=sha(RESULT),forward_record=record,
        initial_world_xyyaw=record['final_world_xyyaw'],expected_saved_last_state=rows[-1],
        retained_all_joint_positions_rad=joints,second_reverse=leg,
        previous_measured_reverse_total_m=previous_reverse,
        completed_direction_changes=completed_changes,
        input_sha256={**info['input_sha256'],**record['frozen_input_sha256'],
            str(RESULT):sha(RESULT),str(RESULT.with_suffix('.csv')):sha(RESULT.with_suffix('.csv')),
            str(observer_path):sha(observer_path),**files})
    audit=dict(original_verdict_unchanged='PASS',raw_primary_checks_reproduced=True,primary_check_count=len(actual['checks']),
        maximum_course_command_reproduction_error=command_error,maximum_steering_target_error_rad=steer_error,
        maximum_applied_wheel_target_error_rad_s=wheel_error,terminal_ready_count=ready_count,
        fixed_hold_attempt_count=record['fixed_hold_attempt_count'],terminal_pulses_used=record['terminal_pulses_used'],
        terminal_position_error_m=actual['terminal_position_error_m'],terminal_heading_error_deg=actual['terminal_heading_error_deg'],
        raw_observer_coverage_reproduced=True,maximum_raw_anchor_sum_error_n=anchor_error,
        observer_does_NOT_identify_unique_traction_or_motor_cause=True)
    return info,audit,manifest,geometry,inventory


def second_reverse_profile(info):
    # The actual start is intentionally NOT copied from poses[0]; it comes
    # from the raw held endpoint, including its nonzero position/yaw error.
    return {**info['second_reverse']}


def model_case(info,dt,residual_case,response_scale):
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import MODE_CURVATURE_RESPONSE_SCALE,MODE_LATERAL_RESPONSE_SCALE
    if dt not in STEPS or residual_case not in RESIDUALS or response_scale not in SCALES:
        raise ValueError('Only the eight declared clocks/residuals/synthetic scales may be screened')
    leg=second_reverse_profile(info);mode='front_only';frozen,_=load_validated_response();last=info['expected_saved_last_state']
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
    reference=FirstReverseReference(leg);goal=np.array(leg['next_cusp_local_xyyaw'])
    now=0.;trace=[];feasible=True;max_slew=max_acceleration=previous_speed=0.
    controller=None;tail=None;error=None;entry_checks={};events=[]
    try:
        for _ in range(math.ceil(90./dt)):
            if now<2.75-1e-10:speed=k=0.;phase='pre_hold'
            else:
                if controller is None:controller=FirstReverseProgress(leg,now)
                if tail is None:
                    out=controller.step(now,pose,float(np.linalg.norm(velocity[:2])),response.actual)
                    speed,k=out['speed_m_s'],out['curvature_1_m'];phase='reverse_motion' if out['phase']=='tracking' else 'reverse_braking'
                    if out['transition']:events.append(dict(time_s=now,**out))
                    if out['ready']:
                        entry_checks=arrival_checks(pose,goal,float(np.linalg.norm(velocity[:2])),response.actual);tail=now
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
            trace.append(dict(time_s=now,phase=phase,phase_time_s=now-tail if tail is not None else 0.,leg_index=2,
                command_mode=mode,command_speed_m_s=speed,requested_curvature_1_m=k,
                x_m=float(pose[0]),y_m=float(pose[1]),yaw_rad=float(pose[2]),
                front_rad=float(response.actual[0]),rear_rad=float(response.actual[1]),
                command_front_rad=commanded[0],command_rear_rad=commanded[1],reference_progress_m=reference.progress,
                position_hold_enabled=tail is not None,reference_position_error_m=path_error,reference_heading_error_deg=heading,
                terminal_position_error_m=position,terminal_heading_error_deg=yaw,model_body_speed_m_s=float(np.linalg.norm(velocity[:2]))))
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
        position_under_30mm=position<=.030,heading_under_2deg=abs(yaw)<=2.,
        fixed_tail_complete=bool(fixed) and fixed[-1]['time_s']-fixed[0]['time_s']>=2.9,
        fixed_tail_position_under_30mm=bool(fixed) and all(r['terminal_position_error_m']<=.030 for r in fixed),
        fixed_tail_heading_under_2deg=bool(fixed) and all(abs(r['terminal_heading_error_deg'])<=2. for r in fixed),
        fixed_tail_speed_under_5mm_s=bool(fixed) and all(r['model_body_speed_m_s']<=.005 for r in fixed),
        fixed_tail_regression_under_1mm_s=regression is not None and regression<=.001,
        fixed_tail_drift_under_3mm=drift is not None and drift<=.003,
        existing_force_limits=bool(feasible),command_slew_under_8deg_s=max_slew<=math.radians(8.)+1e-9,
        command_acceleration_under_20mm_s2=max_acceleration<=.020+1e-9,
        only_reverse_commands_under_50mm_s=all(-.050-1e-9<=r['command_speed_m_s']<=1e-9 for r in trace),
        reverse_episode_under_UNCHANGED_0p5m=distance<=.5,
        total_reverse_with_prior_measured_episode_under_1m=info['previous_measured_reverse_total_m']+distance<=1.,
        next_direction_change_within_existing_four_limit=info['completed_direction_changes']+1<=forward.POLICY['maximum_direction_changes'],
        reverse_progress_over_50mm=reverse_episode_progress(trace)>.050,
        at_most_one_same_direction_terminal_reapproach=controller is not None and controller.reapproaches<=1)
    return dict(dt_s=dt,residual_case=residual_case,SYNTHETIC_body_target_scale=response_scale,error=error,
        checks={k:bool(v) for k,v in checks.items()},duration_s=now,initial_actual_local_xyyaw=initial_pose.tolist(),
        initial_actual_body_velocity_with_pose_derived_yaw_rate=initial_velocity.tolist(),initial_actual_steering_rad=list(initial_steer),
        synthetic_additive_steering_residual_deg=list(residual),fixed_next_cusp_local_xyyaw=goal.tolist(),
        terminal_position_error_m=position,terminal_heading_error_deg=yaw,model_measured_center_distance_m=distance,
        previous_measured_reverse_total_m=info['previous_measured_reverse_total_m'],terminal_events=events,
        maximum_terminal_reapproaches=controller.reapproaches if controller else 0,
        model_parameters=model_bounds(),model_contract=execution_contract(),
        exact_same_measured_terminal_controller_NEW_front_only_length_history_domain=True,
        no_reverse_course_feedback_or_new_gains=True,actual_endpoint_NOT_reset_to_nominal_cusp=True,
        synthetic_cases_NOT_fitted_uncertainty_or_statistical_robustness=True,
        caster_stiction_captured_hold_and_measured_handoff_NOT_modelled=True,
        front_only_large_angle_PhysX_validation_still_pending=True,
        PhysX_executed=False,automatic_motion_authorized=False),trace


def load_review(directory):
    report=json.loads((Path(directory)/'post_forward_review.json').read_bytes())
    verify_sources(report);verify_files(report['input_sha256'])
    verify_files({c['trace_file']:c['trace_sha256'] for c in report['cases']})
    expected={(d,r,s) for d in STEPS for r in RESIDUALS for s in SCALES}
    if (report['version']!=VERSION or report['forward_result']!=str(RESULT)
            or report['forward_result_sha256']!=sha(RESULT)
            or report['PhysX_executed'] is not False or report['automatic_motion_authorized'] is not False
            or len(report['cases'])!=8
            or {(c['dt_s'],c['residual_case'],c['SYNTHETIC_body_target_scale']) for c in report['cases']}!=expected
            or report['all_second_reverse_model_cases_pass']!=(all(all(c['checks'].values()) for c in report['cases'])
                and report['actual_forward_swept_strict_10mm_raster_reproduced_clear']
                and report['actual_start_strict_10mm_clear'] and report['frozen_nominal_second_reverse_strict_10mm_clear'])):
        raise ValueError('Complete truthful post-forward EIGHT-case MODEL review required; never omit a HOLD')
    primary=json.loads(RESULT.read_bytes());candidate=json.loads(Path(report['suffix_candidate']).read_bytes())
    if (report['actual_start_world_xyyaw']!=primary['final_world_xyyaw']
            or report['retained_all_joint_positions_rad']!=forward.reverse.joints_from_last(primary)
            or report['second_reverse']!=extract_second_reverse(candidate)
            or report['strict_map_digest']!=primary['strict_map_digest']
            or report['original_mission_goal_xyyaw']!=primary['mission_goal_world_xyyaw_UNCHANGED']):
        raise ValueError('Actual post-forward state, original map/goal or second reverse reference changed')
    return report


def run_review(output):
    # CLI-only siblings must not be imported at Kit helper-load time.
    from audit_hospital_h3b_directional_playback import collision_diagnostic
    from review_hospital_h3b_reserve_execution import save_trace

    output=Path(output)
    if output.exists():raise FileExistsError('Preserve every previous screen, including HOLD; no overwrite')
    info,audit,manifest,_,_=inspect_forward()
    with np.load(manifest['occupancy_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    checker=NativeHospitalEnvelopeChecker(occupied,origin,resolution,info['mission_start_xyyaw'])
    local=forward.relative_pose(info['initial_world_xyyaw'],info['mission_start_xyyaw'])
    # Independently recheck the complete actual forward swept raster, not just
    # the primary's successful termination/contact count.
    rows=read_trace(RESULT.with_suffix('.csv'));last=info['forward_record']['initial_saved_world_xyyaw'];swept=[]
    for row in rows:
        pose=[row[k] for k in ('x_m','y_m','yaw_rad')]
        turn=forward.wrap(pose[2]-last[2]);length=math.hypot(pose[0]-last[0],pose[1]-last[1])
        count=max(1,math.ceil(length/.005),math.ceil(abs(turn)/math.radians(.25)))
        swept.extend(forward.relative_pose([last[0]+j/count*(pose[0]-last[0]),last[1]+j/count*(pose[1]-last[1]),last[2]+j/count*turn],
            info['mission_start_xyyaw']) for j in range(1,count+1));last=pose
    actual_clear=not checker(swept)
    nominal_clear=not checker(info['second_reverse']['poses_local_xyyaw'])
    start_clear=not checker([local])
    sources=[Path(__file__),_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/review_hospital_h3b_post_forward.py'),
        _ijcas_path(WORKSPACE, 'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_forward.py')]
    header=dict(version=VERSION,role='RAW_FORWARD_PASS_AND_SECOND_REVERSE_MODEL_ONLY',
        forward_result=str(RESULT),forward_result_sha256=sha(RESULT),raw_forward_audit=audit,
        actual_start_world_xyyaw=info['initial_world_xyyaw'],retained_all_joint_positions_rad=info['retained_all_joint_positions_rad'],
        original_mission_start_xyyaw=info['mission_start_xyyaw'],original_mission_goal_xyyaw=info['mission_goal_xyyaw'],
        suffix_candidate=info['suffix_candidate'],suffix_candidate_sha256=info['suffix_candidate_sha256'],
        strict_map_digest=info['strict_map_digest'],second_reverse=info['second_reverse'],
        previous_measured_reverse_total_m=info['previous_measured_reverse_total_m'],completed_direction_changes=info['completed_direction_changes'],
        actual_forward_swept_strict_10mm_raster_reproduced_clear=actual_clear,
        actual_start_strict_10mm_clear=start_clear,frozen_nominal_second_reverse_strict_10mm_clear=nominal_clear,
        declared_dt_s=STEPS,declared_residual_cases=RESIDUALS,declared_SYNTHETIC_body_target_scales=SCALES,
        synthetic_conditions_NOT_identified_uncertainty_or_statistical_robustness=True,
        physical_caster_state_goal_acceptance_model_or_terrain_changed=False,
        new_front_only_50mm_s_domain_is_NOT_rear_reverse_PhysX_certificate=True,
        previous_measured_PASS_and_FAIL_outcomes_preserved=True,
        PhysX_executed=False,automatic_motion_authorized=False,full_H3B_turning_path_verified=False,use_for_final_statistics=False,
        input_sha256={**info['input_sha256'],**{str(p):sha(p) for p in sources}},
        source_sha256={p.name:sha(p) for p in sources})
    output.mkdir(parents=True)
    with (output/'contract.json').open('x') as stream:json.dump(header,stream,indent=2,allow_nan=False)
    cases=[]
    for scale in SCALES:
        for residual in RESIDUALS:
            for dt in STEPS:
                result,trace=model_case(info,dt,residual,scale)
                diag=collision_diagnostic(trace,checker,occupied,origin,resolution,info['mission_start_xyyaw'])
                result['sampled_envelope_diagnostic']=diag
                result['checks']['strict_10mm_model_envelope_clear']=diag['sampled_model_envelope_clear']
                save_trace(output,f'scale{round(scale*100)}_{residual}_{round(dt*1000)}ms',result,trace);cases.append(result)
                print('SECOND REVERSE MODEL',scale,residual,dt,'ADMIT' if all(result['checks'].values()) else 'HOLD',
                    'mm/deg:',round(1000*result['terminal_position_error_m'],3),round(result['terminal_heading_error_deg'],6),
                    'distance m:',round(result['model_measured_center_distance_m'],6),flush=True)
    admitted=all(all(c['checks'].values()) for c in cases) and actual_clear and start_clear and nominal_clear
    report=dict(header,cases=cases,all_second_reverse_model_cases_pass=admitted,
        status='SECOND_REVERSE_MODEL_SCREEN_PASS_LIVE_AND_EXECUTOR_PENDING' if admitted else 'HOLD_SECOND_REVERSE_MODEL_SENSITIVITY_OR_RESOURCE_OR_GEOMETRY',
        next_stage='READ_ONLY_ACTUAL_FORWARD_PASS_CONTINUITY_THEN_REFERENCE_MARGIN_REVIEW_NOT_DRIVE')
    verify_files(header['input_sha256'])
    with (output/'post_forward_review.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print('RAW FORWARD:',audit['original_verdict_unchanged'],'; MODEL cases passing:',sum(all(c['checks'].values()) for c in cases),'/ 8')
    print('STATUS:',report['status']);print('saved:',output/'post_forward_review.json')
    print('Keep actual next-forward PASS PAUSED. No reset/manual Play/old controller. Model HOLD is NOT a measured failure or global impossibility.')
    return report
