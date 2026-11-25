"""Offline hospital frame/scan contracts; never import or start ROS/Isaac.

Keep the old simple-corridor publishers and results unchanged. These pure
functions prepare explicit fixed-map poses and sensor-origin rays for a
future separately authorized adapter. They are NOT a sensor publisher,
localizer, Nav2 controller, motion admission or physical validation.
"""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

import hospital_runtime_evidence_io as evidence


ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT/'studies/IJCAS/results/hospital_paper_gate'
H5_REPLAN_CART='/World/Hospital/SM_SupplyCart_02a7/SM_SupplyCart_02a'
VERSION = 'HOSPITAL_FIXED_FRAME_AND_SENSOR_ORIGIN_OFFLINE_CONTRACT_v1'
SCENARIOS = ('H1_bedside_departure','H2_bedside_lateral_alignment',
    'H3_doorway_immediate_turn','H4_ward_corridor_turn',
    'H5_obstacle_bypass_return','H6_integrated_transfer_mission')
STANDARDS = dict(
    laser_scan='https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/LaserScan.msg',
    axes='https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst',
    frames='https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst',
    nav2_tf='https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/transformation/setup_transforms/')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value,size,name):
    array=np.asarray(value,dtype=float)
    if array.shape!=(size,) or not np.isfinite(array).all():
        raise ValueError('Finite '+str(size)+'-vector required: '+name)
    return array


def wrap(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


def transform_pose(parent_from_child,pose_in_child,*,inverse=False):
    """Explicit SI/right-handed SE(2); no estimate-derived route anchoring."""
    x,y,yaw=vector(parent_from_child,3,'parent_from_child_xyyaw')
    a,b,heading=vector(pose_in_child,3,'pose_xyyaw')
    c,s=math.cos(yaw),math.sin(yaw)
    if inverse:
        a,b=a-x,b-y
        return [float(c*a+s*b),float(-s*a+c*b),wrap(heading-yaw)]
    return [float(x+c*a-s*b),float(y+s*a+c*b),wrap(yaw+heading)]


def fixed_route_frames(plan,world_from_map_xyyaw):
    """Check both saved coordinate representations; never append a goal edge.

    The map alignment must be declared separately. An AMCL first estimate is
    NOT an argument. This is geometry accounting, including for signed H3;
    it does not make a forward-only Nav2 controller support reverse motion.
    """
    if plan.get('scenario') not in SCENARIOS:
        raise ValueError('Explicit hospital task required; not the old simple corridor')
    frame=vector(world_from_map_xyyaw,3,'declared_world_from_map')
    start=vector(plan['start_world_xyyaw'],3,'fixed_start_world')
    goal=vector(plan['goal_world_xyyaw'],3,'fixed_goal_world')
    states,actions=plan['selected_states'],plan['selected_actions']
    if not states or len(states)!=len(actions):
        raise ValueError('Original nonempty one-to-one action/state correspondence required')
    world=[start.tolist()]
    maximum=0.
    for state in states:
        saved=vector(state['world_xyyaw'],3,'saved_world_state')
        reconstructed=transform_pose(start,(state['x_m'],state['y_m'],math.radians(state['yaw_deg'])))
        error=max(abs(saved[0]-reconstructed[0]),abs(saved[1]-reconstructed[1]),abs(wrap(saved[2]-reconstructed[2])))
        if error>1e-8:raise ValueError('Saved fixed local/world reference disagrees; do not rebase it')
        maximum=max(maximum,error);world.append(saved.tolist())
    mapped=[transform_pose(frame,pose,inverse=True) for pose in world]
    for original,mapped_pose in zip(world,mapped):
        back=transform_pose(frame,mapped_pose)
        if max(abs(original[0]-back[0]),abs(original[1]-back[1]),abs(wrap(original[2]-back[2])))>1e-8:
            raise ValueError('Declared fixed world/map roundtrip failed')
    return dict(scenario=plan['scenario'],declared_world_from_map_xyyaw=frame.tolist(),
        fixed_reference_world_xyyaw=world,fixed_reference_map_xyyaw=mapped,
        exact_task_goal_world_xyyaw=goal.tolist(),exact_task_goal_map_xyyaw=transform_pose(frame,goal,inverse=True),
        maximum_local_world_coordinate_difference=maximum,
        source_action_state_count=len(actions),goal_edge_appended=False,
        source_actions_or_states_modified=False,first_sensor_estimate_used_to_move_task=False,
        route_or_terminal_or_reverse_execution_admitted=False)


def rotation_wxyz(quaternion):
    q=vector(quaternion,4,'explicit_WXYZ_quaternion');norm=float(np.linalg.norm(q))
    if abs(norm-1.)>1e-6:raise ValueError('Unit WXYZ quaternion required; no arbitrary pose normalization')
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def scan_geometry(base_xyz,base_wxyz,sensor_xyz_in_base,sensor_wxyz_in_base,*,
        frame_id,ray_count,angle_min,angle_increment,range_min,range_max,rate_hz):
    """Instantaneous planar virtual scan: one explicit sensor origin.

    All beams have the same acquisition instant. A future ROS publisher must
    stamp that instant, set time_increment=0 and supply the exact base/sensor
    TF. No perimeter advance, yaw-only flattening or world-height override.
    Range/noise/odometry assumptions still need a separately frozen design.
    """
    base=vector(base_xyz,3,'base_xyz');offset=vector(sensor_xyz_in_base,3,'sensor_xyz_in_base')
    rb,rs=rotation_wxyz(base_wxyz),rotation_wxyz(sensor_wxyz_in_base)
    if (not isinstance(frame_id,str) or not frame_id or frame_id.startswith('/')
            or any(ch.isspace() for ch in frame_id) or type(ray_count) is not int or not 2<=ray_count<=65536):
        raise ValueError('Explicit unprefixed sensor frame and bounded integer ray count required')
    values=(angle_min,angle_increment,range_min,range_max,rate_hz)
    if (not all(math.isfinite(float(v)) for v in values) or angle_increment<=0.
            or not 0.<range_min<range_max or rate_hz<=0.
            or (ray_count-1)*angle_increment>=2*math.pi):
        raise ValueError('Finite positive range/rate/increment and no duplicate full-circle endpoint required')
    if frame_id=='base_link' and (np.linalg.norm(offset)>1e-12 or not np.allclose(rs,np.eye(3),rtol=0.,atol=1e-12)):
        raise ValueError('Nonidentity laser extrinsics require their own frame, not base_link')
    origin=base+rb@offset;rotation=rb@rs
    rays=[]
    for i in range(ray_count):
        angle=float(angle_min+i*angle_increment)
        direction=rotation@np.array([math.cos(angle),math.sin(angle),0.])
        rays.append(dict(index=i,angle_rad=angle,origin_world_m=origin.tolist(),
            direction_world=direction.tolist(),length_m=float(range_max)))
    return dict(frame_id=frame_id,base_to_sensor_translation_m=offset.tolist(),
        base_to_sensor_rotation_wxyz=vector(sensor_wxyz_in_base,4,'sensor_wxyz').tolist(),
        sensor_origin_world_m=origin.tolist(),angle_min_rad=float(angle_min),
        angle_max_rad=float(angle_min+(ray_count-1)*angle_increment),angle_increment_rad=float(angle_increment),
        range_min_m=float(range_min),range_max_m=float(range_max),scan_time_s=1./rate_hz,
        time_increment_s=0.,all_beams_acquired_at_one_instant=True,rays=rays,
        old_perimeter_blind_zone_used=False,ROS_publisher_or_PhysX_query_started=False)


def scan_return(ray,hits,*,own_collision_paths,range_min):
    """Decode ALL raw hits from one sensor-origin ray without silent omissions.

    Self filtering uses exact shape identities only. An unknown path under
    the robot prefix is NOT discarded. A below-min external hit blocks this
    proposed scan instead of being converted into free space. JSON output
    encodes no-return symbolically, never with a non-finite numeric value.
    """
    origin=vector(ray['origin_world_m'],3,'ray_origin');direction=vector(ray['direction_world'],3,'ray_direction')
    length=float(ray['length_m'])
    if (abs(float(np.linalg.norm(direction))-1.)>1e-6 or not math.isfinite(length)
            or not math.isfinite(range_min) or not 0.<range_min<length):
        raise ValueError('Unit ray and finite increasing range bounds required')
    own=list(own_collision_paths)
    if (not own or len(set(own))!=len(own) or any(not isinstance(p,str) or not p.startswith('/World/') for p in own)):
        raise ValueError('Exact nonempty unique own collision identities required')
    raw=[];self_hits=[];external=[]
    for hit in hits:
        path=hit['collision'];distance=float(hit['distance_m'])
        point=vector(hit['position_m'],3,'raw_hit_position');normal=vector(hit['normal'],3,'raw_hit_normal')
        if (not isinstance(path,str) or not path.startswith('/World/') or not math.isfinite(distance)
                or not 0.<=distance<=length or abs(float(np.linalg.norm(normal))-1.)>1e-4
                or np.max(np.abs(point-(origin+distance*direction)))>1e-4):
            raise ValueError('Malformed/nonfinite/off-ray native hit; no free-space substitution')
        saved=dict(collision=path,distance_m=distance,position_m=point.tolist(),normal=normal.tolist())
        raw.append(saved)
        (self_hits if path in own else external).append(saved)
    nearest=min(external,key=lambda hit:hit['distance_m']) if external else None
    status='NO_RETURN' if nearest is None else 'BELOW_MIN_NONFREE' if nearest['distance_m']<range_min else 'VALID_RANGE'
    return dict(raw_hits=raw,exact_self_hits=self_hits,all_external_hits=external,
        nearest_external_hit=nearest,status=status,
        valid_range_m=nearest['distance_m'] if status=='VALID_RANGE' else None,
        ROS_no_return_encoding='positive_infinity' if status=='NO_RETURN' else None,
        publishable_without_silent_near_obstacle_loss=status!='BELOW_MIN_NONFREE',
        sensor_observation_or_motion_started=False)


def centerline_encoder_twist(front_velocity_rad_s,rear_velocity_rad_s,front_steering_rad,rear_steering_rad,*,geometry):
    """Conditional, mode-blind TWO-active-wheel rolling kinematics, not truth.

    Require explicit source geometry, encoder signs and raw steering zeros;
    never borrow the old ROS rear-drive sign or1.290m command wheelbase.
    Each steering pivot and rolling centre must coincide in the planar view.
    At x=a,b the wheel velocity is [u,v+x*r]. Equal-weight least squares of
    BOTH rolling vectors gives u,v,r and exposes longitudinal incompatibility.
    Passive caster forces, tyre deformation, slip and dynamic response are
    NOT estimated. A small residual does not certify absence of common slip.
    No commands, modes, GT poses, fitted response scales or filters are inputs.
    """
    required={'front_axis_xy_m','rear_axis_xy_m','front_rolling_offset_xy_m','rear_rolling_offset_xy_m',
        'front_radius_m','rear_radius_m','front_drive_sign','rear_drive_sign',
        'front_steering_zero_rad','rear_steering_zero_rad','front_steering_sign','rear_steering_sign'}
    if type(geometry) is not dict or set(geometry)!=required:raise ValueError('Complete explicit encoder geometry/sign/zero definition required')
    a,b=(vector(geometry[k],2,k) for k in ('front_axis_xy_m','rear_axis_xy_m'))
    if not b[0]<0.<a[0] or a[1]!=0. or b[1]!=0.:raise ValueError('Distinct front/rear axes on the declared body centreline required')
    for k in ('front_rolling_offset_xy_m','rear_rolling_offset_xy_m'):
        if np.any(vector(geometry[k],2,k)!=0.):raise ValueError('Nonzero steering-to-rolling planar offset needs a different odometry model')
    for k in ('front_radius_m','rear_radius_m'):
        if type(geometry[k]) not in (int,float) or not math.isfinite(geometry[k]) or geometry[k]<=0.:
            raise ValueError('Positive finite explicitly declared rolling radii required')
    for k in ('front_drive_sign','rear_drive_sign','front_steering_sign','rear_steering_sign'):
        if type(geometry[k]) not in (int,float) or geometry[k] not in (-1,1):raise ValueError('Explicit unit encoder/steering signs required')
    wf,wr,df,dr,zf,zr=vector([front_velocity_rad_s,rear_velocity_rad_s,front_steering_rad,rear_steering_rad,
        geometry['front_steering_zero_rad'],geometry['rear_steering_zero_rad']],6,'measured encoder channels and zeros')
    df=geometry['front_steering_sign']*(df-zf);dr=geometry['rear_steering_sign']*(dr-zr)
    sf=geometry['front_radius_m']*geometry['front_drive_sign']*wf
    sr=geometry['rear_radius_m']*geometry['rear_drive_sign']*wr
    xf,yf=sf*math.cos(df),sf*math.sin(df);xr,yr=sr*math.cos(dr),sr*math.sin(dr)
    u=(xf+xr)/2.;yaw=(yf-yr)/(a[0]-b[0]);v=(a[0]*yr-b[0]*yf)/(a[0]-b[0])
    result=vector([u,v,yaw,xf-xr],4,'finite conditional wheel twist and incompatibility')
    return dict(body_twist_u_v_r=result[:3].tolist(),axle_longitudinal_incompatibility_m_s=float(result[3]),
        measured_rolling_vectors_front_rear_m_s=[[float(xf),float(yf)],[float(xr),float(yr)]],
        no_slip_is_an_assumption_NOT_observed_or_proved=True,
        zero_incompatibility_does_NOT_bound_common_slip=True,
        command_or_ground_truth_used_as_encoder_input=False,actual_motion_authorized=False)


def integrate_body_twist_samples(time_s,body_twists,*,initial_xyyaw,maximum_sample_gap_s):
    """Trapezoid body-twist samples + constant-twist SE(2) step per interval.

    No endpoint fit, pose reset, route tracking, filter, GT feedback or wrapped
    wheel-position differentiation. This numerical diagnostic is not a drift
    guarantee. It begins at the FIRST supplied stamp; no missing first edge.
    """
    stamps=np.asarray(time_s,dtype=float);twists=np.asarray(body_twists,dtype=float)
    if (stamps.ndim!=1 or len(stamps)<2 or twists.shape!=(len(stamps),3) or not np.isfinite(stamps).all()
            or not np.isfinite(twists).all() or type(maximum_sample_gap_s) not in (int,float)
            or not math.isfinite(maximum_sample_gap_s) or maximum_sample_gap_s<=0.
            or np.any(stamps<0.) or np.any(np.diff(stamps)<=0.) or np.any(np.diff(stamps)>maximum_sample_gap_s)):
        raise ValueError('Complete finite increasing clock/twist pairs and explicit gap limit required')
    poses=[vector(initial_xyyaw,3,'declared initial odometry pose').tolist()]
    for i,dt in enumerate(np.diff(stamps)):
        u,v,r=(twists[i]+twists[i+1])/2.;theta=float(r*dt)
        sinc=math.sin(theta)/theta if theta else 1.;cosc=2.*math.sin(theta/2.)**2/theta if theta else 0.
        increment=[float(dt*(sinc*u-cosc*v)),float(dt*(cosc*u+sinc*v)),theta]
        poses.append(transform_pose(poses[-1],increment))
    return poses


def bounded_wrapped_joint_increments(time_s,positions_rad,*,wrap_period_rad,maximum_abs_rate_rad_s,position_roundoff_rad):
    """Unique principal increments CONDITIONAL on a declared wrap/rate model.

    Every interval must satisfy rate_cap*dt+2*roundoff < half the declared
    period. Endpoints must lie in the declared centred representation range,
    and the recovered displacement must fit the SAME bound. Reject rather
    than infer a period, reset, turn count, best window or missing sample.
    Source-authored caps and compatible endpoint velocities are NOT proof of
    actual between-sample cap enforcement or of a native articulation type.
    """
    time=np.asarray(time_s,dtype=float);positions=np.asarray(positions_rad,dtype=float)
    if (time.ndim!=1 or len(time)<2 or positions.ndim!=2 or positions.shape[0]!=len(time)
            or positions.shape[1]<1 or not np.isfinite(time).all() or not np.isfinite(positions).all()
            or np.any(time<0.) or np.any(np.diff(time)<=0.)):
        raise ValueError('Complete finite strictly advancing clock and joint-position channels required')
    period=wrap_period_rad;roundoff=position_roundoff_rad
    if (type(period) not in (int,float) or not math.isfinite(period) or period<=0.
            or type(roundoff) not in (int,float) or not math.isfinite(roundoff) or roundoff<0.):
        raise ValueError('Explicit positive wrap period and nonnegative per-position roundoff required')
    rate=vector(maximum_abs_rate_rad_s,positions.shape[1],'declared absolute joint-rate caps')
    if np.any(rate<=0.) or np.any(np.abs(positions)>period/2.+roundoff):
        raise ValueError('Positive finite rate caps and declared centred wrapped-position range required')
    bound=np.nextafter(np.nextafter(np.diff(time)[:,None]*rate[None,:],math.inf)+2.*roundoff,math.inf)
    half_lower=math.nextafter(period/2.,-math.inf)
    if not np.isfinite(bound).all() or np.any(bound>=half_lower):
        raise ValueError('Aliased/ambiguous interval: strict half-period displacement bound not satisfied')
    raw=np.diff(positions,axis=0)
    recovered=np.array([[math.remainder(float(value),period) for value in row] for row in raw])
    if np.any(np.abs(recovered)>bound):raise ValueError('Measured angle displacement violates declared rate/roundoff contract')
    wraps=np.rint((raw-recovered)/period).astype(np.int64)
    return dict(displacements_rad=recovered.tolist(),interval_mean_rates_rad_s=(recovered/np.diff(time)[:,None]).tolist(),
        integer_wrap_corrections=wraps.tolist(),wrap_correction_counts=np.sum(wraps!=0,axis=0).tolist(),
        minimum_alias_margin_rad=np.min(np.nextafter(half_lower-bound,-math.inf),axis=0).tolist(),
        maximum_absolute_interval_displacements_rad=np.max(np.abs(recovered),axis=0).tolist(),
        wrap_period_rad=float(period),maximum_abs_rate_rad_s=rate.tolist(),position_roundoff_rad=float(roundoff),
        conditional_on_actual_wrap_and_between_sample_rate_contract=True,
        native_joint_type_and_cap_enforcement_NOT_measured_here=True,actual_motion_authorized=False)


def review_saved_encoder_position_trace(rows,*,geometry,fixed_start_world_xyyaw,maximum_sample_gap_s,
        wrap_period_rad,maximum_abs_rate_rad_s,position_roundoff_rad):
    """Retrospective alternative: wrapped measured positions, NOT target rates.

    Midpoint measured steering approximates the wheel direction in each
    interval. Rate-cap/wrap assumptions remain explicit, and all positions,
    phases and intervals survive. No tuning, low-pass window, GT fit or new
    physical run. Native instantaneous rates are checked against the declared
    cap and compared diagnostically, never substituted for angle increments.
    """
    if not isinstance(rows,(list,tuple)) or len(rows)<2:raise ValueError('Complete saved trace required')
    fields=('time_s','drive_front_rad','drive_rear_rad','front_rad','rear_rad',
        'drive_front_velocity_rad_s','drive_rear_velocity_rad_s','x_m','y_m','yaw_rad')
    values=np.array([[float(row[k]) for k in fields] for row in rows],dtype=float)
    if not np.isfinite(values).all() or any(not isinstance(row['phase'],str) or not row['phase'] for row in rows):
        raise ValueError('Finite complete measured fields and phases required; no omitted bad row')
    time=values[:,0];dt=np.diff(time)
    if (type(maximum_sample_gap_s) not in (int,float) or not math.isfinite(maximum_sample_gap_s)
            or maximum_sample_gap_s<=0. or np.any(dt>maximum_sample_gap_s)):
        raise ValueError('Explicit original sample-gap contract required')
    increments=bounded_wrapped_joint_increments(time,values[:,1:3],wrap_period_rad=wrap_period_rad,
        maximum_abs_rate_rad_s=maximum_abs_rate_rad_s,position_roundoff_rad=position_roundoff_rad)
    rate_caps=np.array(increments['maximum_abs_rate_rad_s'])
    if np.any(np.abs(values[:,5:7])>rate_caps):raise ValueError('Saved instantaneous joint velocity violates declared source cap')
    rates=np.array(increments['interval_mean_rates_rad_s']);steering=(values[:-1,3:5]+values[1:,3:5])/2.
    twists=[];incompatibility=[];poses=[vector(fixed_start_world_xyyaw,3,'fixed nominal start').tolist()]
    for index,(rate,steer) in enumerate(zip(rates,steering)):
        estimate=centerline_encoder_twist(*rate,*steer,geometry=geometry);twist=estimate['body_twist_u_v_r']
        twists.append(twist);incompatibility.append(estimate['axle_longitudinal_incompatibility_m_s'])
        # Constant interval-mean twist, NOT trapezoiding it again or inventing
        # endpoint instantaneous measurements. Reuse the same SE2 integrator.
        poses.append(integrate_body_twist_samples([0.,float(dt[index])],[twist,twist],
            initial_xyyaw=poses[-1],maximum_sample_gap_s=maximum_sample_gap_s)[-1])
    poses=np.asarray(poses);twists=np.asarray(twists);yaw=values[:,9]
    position=np.hypot(poses[:,0]-values[:,7],poses[:,1]-values[:,8])
    heading=np.abs(np.arctan2(np.sin(poses[:,2]-yaw),np.cos(poses[:,2]-yaw)))
    phase=np.array([row['phase'] for row in rows]);incompatibility=np.abs(incompatibility)
    # Independent GT endpoint differences in the midpoint heading frame,
    # labelled as finite differences rather than native angular velocities.
    dyaw=np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw)));midyaw=yaw[:-1]+dyaw/2.
    dx=np.diff(values[:,7])/dt;dy=np.diff(values[:,8])/dt;c,s=np.cos(midyaw),np.sin(midyaw)
    reference=np.column_stack((c*dx+s*dy,-s*dx+c*dy,dyaw/dt))
    mean_instant_rates=(values[:-1,5:7]+values[1:,5:7])/2.
    def summary(mask):
        indices=np.flatnonzero(mask);interval_mask=mask[1:];last=indices[-1]
        return dict(endpoint_samples=int(mask.sum()),intervals=int(interval_mask.sum()),
            first_time_s=float(time[indices[0]]),last_time_s=float(time[last]),
            integrated_position_error_last_m=float(position[last]),integrated_position_error_max_m=float(np.max(position[mask])),
            integrated_heading_error_last_deg=float(np.degrees(heading[last])),integrated_heading_error_max_deg=float(np.degrees(np.max(heading[mask]))),
            interval_RMSE_u_v_r_vs_midpoint_pose_differences=None if not interval_mask.any() else np.sqrt(np.mean((twists[interval_mask]-reference[interval_mask])**2,axis=0)).tolist(),
            interval_wheel_rate_RMSE_vs_mean_endpoint_instantaneous_rates_rad_s=None if not interval_mask.any() else np.sqrt(np.mean((rates[interval_mask]-mean_instant_rates[interval_mask])**2,axis=0)).tolist(),
            axle_incompatibility_abs_p95_m_s=None if not interval_mask.any() else float(np.quantile(incompatibility[interval_mask],.95)))
    bound_summary={k:v for k,v in increments.items() if k not in ('displacements_rad','interval_mean_rates_rad_s','integer_wrap_corrections')}
    return evidence.json_value(dict(version='SAVED_ALL_PHASES_BOUNDED_WRAPPED_POSITION_ODOMETRY_DIAGNOSTIC_v1',
        status='CONDITIONAL_POSITION_INCREMENT_REPLAY_NOT_NATIVE_ENCODER_OR_NAVIGATION_ADMISSION',
        fixed_nominal_start_world_xyyaw=list(map(float,fixed_start_world_xyyaw)),geometry=geometry,
        wrap_and_rate_contract=bound_summary,interval_phase_is_DESTINATION_record_phase=True,
        all_rows=summary(np.ones(len(rows),dtype=bool)),every_saved_phase={p:summary(phase==p) for p in dict.fromkeys(phase.tolist())},
        derived_interval_twists_float64_sha256=hashlib.sha256(twists.tobytes()).hexdigest(),
        integrated_poses_float64_sha256=hashlib.sha256(poses.tobytes()).hexdigest(),
        estimator_measured_inputs=['time_s','drive_front_rad','drive_rear_rad','front_rad','rear_rad'],
        instantaneous_rates_used_only_for_cap_check_and_diagnostic_not_estimator=True,
        no_command_or_GT_state_used_to_fit_or_correct_estimator=True,
        unique_causal_separation_of_solver_sensor_sampling_slip_and_calibration_NOT_established=True,
        no_slip_and_midpoint_steering_are_still_model_assumptions=True,
        actual_motion_authorized=False,new_PhysX_trials=0,use_for_final_statistics=False))


def review_saved_encoder_trace(rows,*,geometry,fixed_start_world_xyyaw,maximum_sample_gap_s):
    """ALL saved rows, retrospective diagnostic, no calibration or admission.

    Derive twists ONLY from measured rolling velocities and measured steering.
    The declared nominal start projects integrated odometry into world for
    evaluation; neither the first GT observation nor the final endpoint fits
    that frame. GT channels are used AFTER estimation to describe disagreement.
    They are not encoder inputs, gain fitting or a sensor-mode controller.
    Every recorded phase remains separate; unequal aborted run lengths must
    not be pooled into a matched performance or localization-success claim.
    """
    if not isinstance(rows,(list,tuple)) or len(rows)<2:raise ValueError('Complete nonempty saved trace required')
    numeric=('time_s','drive_front_velocity_rad_s','drive_rear_velocity_rad_s','front_rad','rear_rad',
        'drive_front_rad','drive_rear_rad','x_m','y_m','yaw_rad','vx_physx_m_s','vy_physx_m_s')
    values=np.asarray([[float(row[key]) for key in numeric] for row in rows],dtype=float)
    if not np.isfinite(values).all() or any(not isinstance(row['phase'],str) or not row['phase'] for row in rows):
        raise ValueError('Complete finite observed channels and phase labels required; no dropped bad rows')
    by_name={key:values[:,i] for i,key in enumerate(numeric)};time=by_name['time_s']
    twists=[];incompatibility=[]
    for row in rows:
        result=centerline_encoder_twist(*(float(row[key]) for key in numeric[1:5]),geometry=geometry)
        twists.append(result['body_twist_u_v_r']);incompatibility.append(result['axle_longitudinal_incompatibility_m_s'])
    twists=np.asarray(twists,dtype=np.float64)
    poses=np.asarray(integrate_body_twist_samples(time,twists,initial_xyyaw=fixed_start_world_xyyaw,
        maximum_sample_gap_s=maximum_sample_gap_s),dtype=np.float64)
    yaw=by_name['yaw_rad'];c,s=np.cos(yaw),np.sin(yaw)
    observed_u=c*by_name['vx_physx_m_s']+s*by_name['vy_physx_m_s']
    observed_v=-s*by_name['vx_physx_m_s']+c*by_name['vy_physx_m_s']
    # Raw clock finite differences, including both endpoints; NOT a new yaw
    # rate sensor or a fitted smoothing window. Root/CoM velocity references
    # are not silently equated: also retain direct base-pose derivatives.
    dx=np.gradient(by_name['x_m'],time);dy=np.gradient(by_name['y_m'],time)
    observed_r=np.gradient(np.unwrap(yaw),time)
    pose_velocity=np.column_stack((c*dx+s*dy,-s*dx+c*dy,observed_r))
    channel_velocity=np.column_stack((observed_u,observed_v,observed_r))
    position=np.hypot(poses[:,0]-by_name['x_m'],poses[:,1]-by_name['y_m'])
    heading=np.abs(np.arctan2(np.sin(poses[:,2]-yaw),np.cos(poses[:,2]-yaw)))
    incompatible=np.abs(incompatibility);phases=[row['phase'] for row in rows]
    def summary(mask):
        indices=np.flatnonzero(mask);last=indices[-1]
        return dict(samples=len(indices),first_time_s=float(time[indices[0]]),last_time_s=float(time[last]),
            RMSE_body_u_v_r_vs_saved_PhysX_linear_and_pose_yaw_derivative=np.sqrt(np.mean((twists[mask]-channel_velocity[mask])**2,axis=0)).tolist(),
            RMSE_body_u_v_r_vs_saved_base_pose_derivatives=np.sqrt(np.mean((twists[mask]-pose_velocity[mask])**2,axis=0)).tolist(),
            axle_longitudinal_incompatibility_abs_p95_m_s=float(np.quantile(incompatible[mask],.95)),
            axle_longitudinal_incompatibility_abs_max_m_s=float(np.max(incompatible[mask])),
            integrated_position_error_p95_m=float(np.quantile(position[mask],.95)),
            integrated_position_error_max_m=float(np.max(position[mask])),integrated_position_error_last_m=float(position[last]),
            integrated_heading_error_p95_deg=float(np.degrees(np.quantile(heading[mask],.95))),
            integrated_heading_error_max_deg=float(np.degrees(np.max(heading[mask]))),integrated_heading_error_last_deg=float(np.degrees(heading[last])))
    angles=np.column_stack((by_name['drive_front_rad'],by_name['drive_rear_rad']));jumps=np.abs(np.diff(angles,axis=0))
    output=dict(version='SAVED_ALL_PHASES_TWO_ACTIVE_WHEEL_ENCODER_DIAGNOSTIC_v1',
        status='RETROSPECTIVE_CONDITIONAL_KINEMATIC_DISAGREEMENT_NOT_ENCODER_OR_NAVIGATION_ADMISSION',
        geometry=evidence.json_value(geometry),fixed_nominal_start_world_xyyaw=list(map(float,fixed_start_world_xyyaw)),
        declared_initial_time_s=float(time[0]),maximum_sample_gap_s=float(maximum_sample_gap_s),
        estimator_measured_input_columns=list(numeric[1:5]),command_mode_speed_targets_or_GT_used_in_estimator=False,
        derived_twists_float64_sha256=hashlib.sha256(twists.tobytes()).hexdigest(),
        integrated_poses_float64_sha256=hashlib.sha256(poses.tobytes()).hexdigest(),
        numerical_integration='trapezoid_body_twist_then_constant_twist_SE2_exponential_per_interval',
        all_rows=summary(np.ones(len(rows),dtype=bool)),
        every_saved_phase={phase:summary(np.asarray([value==phase for value in phases])) for phase in dict.fromkeys(phases)},
        raw_drive_position_max_single_sample_jump_rad=np.max(jumps,axis=0).tolist(),
        raw_drive_position_jumps_above_pi_count=np.sum(jumps>math.pi,axis=0).tolist(),
        raw_wheel_positions_NOT_differentiated_or_unwrapped=True,
        wheel_angle_wrap_period_and_unaliased_increment_bound_NOT_inferred=True,
        initial_frame_or_radius_signs_or_mode_coefficients_fitted_to_GT=False,
        GT_yaw_rate_is_only_unsmoothed_pose_finite_difference_NOT_a_native_angular_velocity_channel=True,
        saved_PhysX_linear_channel_and_base_origin_derivative_kept_separate=True,
        residual_or_disagreement_does_NOT_identify_unique_slip_noise_calibration_or_physics_cause=True,
        unequal_phase_and_run_duration_NOT_a_paired_method_performance_ranking=True,
        no_slip_assumption_or_reported_covariance_NOT_validated_as_actual_odometry=True,
        actual_motion_authorized=False,new_PhysX_trials=0,use_for_final_statistics=False)
    return evidence.json_value(output)


def _constant(tree,name):
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Expected exact legacy scalar declaration missing: '+name)


def _source_of(tree,source,predicate):
    return [dict(line=node.lineno,source=ast.get_source_segment(source,node)) for node in ast.walk(tree) if predicate(node)]


def legacy_adapter_review(h5_protocol):
    """Source-only review of current legacy adapters; no imports of entries."""
    import yaml
    path=Path(h5_protocol).resolve();p=json.loads(path.read_bytes())
    definition=hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='definition_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (p.get('scenario')!='H5_obstacle_bypass_return' or p.get('actual_motion_authorized') is not False
            or p.get('use_for_final_statistics') is not False or p.get('definition_sha256')!=definition):
        raise ValueError('Own offline H5 definition required; not a live outcome')
    scripts=ROOT/'isaac/scripts';package=ROOT/'src/patient_initial_stability'
    files=dict(scan=scripts/'isaac_prepare_ijcas_cpu_laserscan.py',
        runner=package/'patient_initial_stability/proposed_nav2_path_runner.py',
        map_planner=package/'patient_initial_stability/occupancy_lattice_planner.py',
        localization_launch=package/'launch/sim_cpu_scan_localization_nav2.launch.py',
        nav2=package/'config/nav2_sim.yaml',amcl=package/'config/amcl_cpu_scan.yaml',
        module=Path(__file__).resolve(),workflow=path)
    pins=dict(p['input_sha256'])
    for filename in files.values():
        name,value=str(filename),sha(filename)
        if name in pins and pins[name]!=value:raise ValueError('Conflicting original frozen source: '+name)
        pins[name]=value
    archive=Path(p['workflow_runtime_sources'])
    if sha(archive)!=p['workflow_runtime_sources_sha256']:
        raise ValueError('Original frozen workflow archive changed')
    with zipfile.ZipFile(archive) as saved:
        if set(saved.namelist())!={'studies/IJCAS/scripts/hospital_task_runtime.py',
                'isaac/scripts/isaac_run_ijcas_hospital_task_experiment.py'}:
            raise ValueError('Exact reviewed workflow archive members required')
        for member in saved.namelist():
            name=str(ROOT/member);value=hashlib.sha256(saved.read(member)).hexdigest()
            if name in pins and pins[name]!=value:raise ValueError('Conflicting archived workflow source: '+name)
            pins[name]=value
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Frozen source/evidence changed: '+name)
    texts={name:filename.read_text() for name,filename in files.items() if filename.suffix=='.py'}
    scan,runner,launch=(ast.parse(texts[name]) for name in ('scan','runner','localization_launch'))
    nav2=yaml.safe_load(files['nav2'].read_bytes());amcl=yaml.safe_load(files['amcl'].read_bytes())
    physics_calls=_source_of(scan,texts['scan'],lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('CreateEnableGPUDynamicsAttr','CreateBroadphaseTypeAttr'))
    reanchor=_source_of(runner,texts['runner'],lambda n:isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='origin'
        for t in n.targets) and isinstance(n.value,ast.Name) and n.value.id=='pose')
    scenario_guard=_source_of(runner,texts['runner'],lambda n:isinstance(n,ast.Compare)
        and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get'
        and n.left.args and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='scenario')
    speed_guard=_source_of(runner,texts['runner'],lambda n:isinstance(n,ast.Compare)
        and any(isinstance(c,ast.Constant) and c.value=='target_speed_m_s' for c in ast.walk(n)))
    overrides={}
    for node in ast.walk(launch):
        if isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value in ('publish_map_to_odom','path_tracking_model','initial_alignment_policy'):
                    overrides[key.value]=ast.literal_eval(value)
    controller=nav2['controller_server']['ros__parameters'];follow=controller['FollowPath']
    bundle=json.loads(Path(p['frozen_bundle']).read_bytes())
    frames=[]
    for name in bundle['plans']:
        plan=json.loads(Path(name).read_bytes());frame=fixed_route_frames(plan,(0.,0.,0.))
        frames.append(dict(method=plan['method'],source_plan=str(name),source_sha256=sha(name),
            reference_pose_count=len(frame['fixed_reference_world_xyyaw']),
            maximum_local_world_coordinate_difference=frame['maximum_local_world_coordinate_difference'],
            unchanged_reference_and_goal_in_declared_identity_world_map=True,
            no_goal_edge_appended=not frame['goal_edge_appended']))
    result=dict(version=VERSION,status='OFFLINE_SOURCE_GAPS_AND_FIXED_COORDINATES_RECORDED_NOT_SENSOR_OR_NAV2_VALIDATION',
        scope='Static source patterns and pure geometry only; not whole-program verification or runtime diagnosis',
        legacy_sources={k:str(v) for k,v in files.items()},
        legacy_scan=dict(actor=_constant(scan,'ARTICULATION_ROOT'),frame=_constant(scan,'FRAME'),
            rate_hz=_constant(scan,'PUBLISH_RATE_HZ'),ray_count=_constant(scan,'RAY_COUNT'),
            world_fixed_scan_height_m=_constant(scan,'SCAN_HEIGHT_M'),
            physics_write_calls=physics_calls,
            perimeter_advance_used=any(isinstance(n,ast.FunctionDef) and n.name=='_distance_to_envelope' for n in scan.body),
            not_a_measured_hospital_sensor_or_3D_extrinsic_validation=True),
        legacy_runner=dict(scenario_guard=scenario_guard,first_estimate_route_reanchor=reanchor,
            positive_speed_guard=speed_guard,goal_source=_source_of(runner,texts['runner'],lambda n:isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='GOAL_LOCAL' for t in n.targets))),
        legacy_nav2=dict(desired_linear_vel_m_s=follow['desired_linear_vel'],
            velocity_smoother_max_velocity=nav2['velocity_smoother']['ros__parameters']['max_velocity'],
            allow_reversing=follow['allow_reversing'],controller_odom_topic=controller['odom_topic'],
            bt_odom_topic=nav2['bt_navigator']['ros__parameters']['odom_topic'],
            amcl_initial_pose=amcl['amcl']['ros__parameters']['initial_pose'],launch_scalar_overrides=overrides),
        legacy_map_replanning=legacy_replanning_source_review(files['map_planner']),
        fixed_H5_four_method_coordinate_checks=frames,
        hospital_reference_contract=dict(fixed_start_world_xyyaw=p['start_xyyaw'],fixed_goal_world_xyyaw=p['goal_xyyaw'],
            command_limits=p['command_limits'],guards=p['guards'],acceptance=p['acceptance'],
            source_controller_config=p['terminal_controller_config'],
            NOT_changed_or_admitted_by_this_review=True),
        frozen_H5_runtime_archive_members_match_active_files=True,
        required_separate_hospital_adapter_work=[
            'Bind exact admitted hospital actor/session and keep all existing physics settings unchanged.',
            'Use explicit sensor extrinsics/TF and same-origin full-3D ray geometry; do not silently retain the old perimeter blind region.',
            'Bind the fixed authored world/map transform; do not move the task or path using the first AMCL estimate.',
            'Do not equate the old Nav2 controller with common v3; freeze and revalidate the selected sensor/controller interface without weakening physical safety limits.',
            'Declare a scan-consistent localization map separately from the conservative full-body planning/safety map.',
            'Declare ground-truth odometry assistance separately; do not claim wheel/IMU odometry has been validated.',
            'Provide one map->odom authority plus odom->base_link and exact base_link->sensor transforms.',
            'Validate reverse handoff and continuous integrated execution separately; a forward-only corridor runner cannot admit H3/H6.',
            'Map-content revision detection and measured zero-command standstill/replan/resume need an explicit common hospital policy; a periodic timer alone is not that validation.',
            'Obtain separate setup/motion approval before any bridge, graph, ROS launch, Play, staging or navigation run.'],
        original_simple_corridor_results_NOT_reclassified=True,
        original_H5_workflow_and_all_legacy_files_unchanged=True,
        actual_live_stage_or_ROS_access=False,new_trial_count=0,actual_motion_authorized=False,
        sensor_or_localization_or_Nav2_success=False,use_for_final_statistics=False,
        standards=STANDARDS,input_sha256=pins)
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Source changed during offline review: '+name)
    return evidence.json_value(result)


def export_legacy_review(h5_protocol,output_base=None):
    result=legacy_adapter_review(h5_protocol)
    digest=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    folder=(GATE/'hospital_sensor_navigation/offline_adapter_review' if output_base is None else Path(output_base))/digest[:16]
    path=folder/'review.json'
    if path.exists():
        if json.loads(path.read_bytes())!=result:raise ValueError('Existing source review differs; no overwrite')
    else:evidence.write_exclusive(path,result)
    return path


def planning_input_identity(cells,*,shape_rows_columns,origin_xyyaw_m_rad,resolution_m,
        map_frame_id,world_from_map_xyyaw,goal_world_xyyaw,environment_sha256,planner_contract_sha256):
    """PURE identity of an explicit planning snapshot, never a sensor map.

    The whole signed OccupancyGrid content participates, including UNKNOWN;
    republishing identical content with another timestamp is NOT a revision.
    Fixed frame/grid origin, exact goal and declared environment/planner
    revisions also participate. No first-estimate reanchor or probability
    threshold/unknown-to-free conversion. This digest does NOT authenticate
    provenance, establish freshness, check collision/support, stop a robot,
    plan a route, or authorize publishing/replacing a command.
    """
    shape=list(shape_rows_columns)
    if (len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape) or math.prod(shape)>4_000_000
            or type(cells) is not list or len(cells)!=math.prod(shape)
            or any(type(v) is not int or not -1<=v<=100 for v in cells)):
        raise ValueError('Bounded explicit signed row-major planning occupancy required; no unknown-to-free conversion')
    if (type(resolution_m) not in (int,float) or not math.isfinite(resolution_m) or resolution_m<=0.
            or resolution_m>float(np.finfo(np.float32).max)
            or float(np.float32(resolution_m))==0.
            or not isinstance(map_frame_id,str) or not map_frame_id or map_frame_id.startswith('/')
            or any(c.isspace() for c in map_frame_id)):
        raise ValueError('Explicit finite metric grid resolution and fixed map frame required')
    for value in (environment_sha256,planner_contract_sha256):
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Exact environment and prospective planner-contract SHA256 required')
    record=dict(version='PLANNING_INPUT_CONTENT_IDENTITY_NOT_EXECUTION_GATE_v1',map_frame_id=map_frame_id,
        shape_rows_columns=shape,origin_xyyaw_m_rad=vector(origin_xyyaw_m_rad,3,'fixed grid origin').tolist(),
        resolution_m=float(resolution_m),world_from_map_xyyaw=vector(world_from_map_xyyaw,3,'fixed world<-map').tolist(),
        goal_world_xyyaw=vector(goal_world_xyyaw,3,'unchanged world task goal').tolist(),
        environment_sha256=environment_sha256,planner_contract_sha256=planner_contract_sha256,
        signed_row_major_cells_sha256=hashlib.sha256(np.asarray(cells,dtype=np.int8).tobytes()).hexdigest())
    encoded=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(identity_sha256=hashlib.sha256(encoded).hexdigest(),inputs=record,
        acquisition_stamp_intentionally_NOT_a_content_revision=True,unknown_values_preserved=True,
        source_provenance_or_freshness_or_map_validity_verified=False,
        stop_or_replan_or_resume_or_command_replacement_authorized=False)


class StoppedReplanBoundary:
    """Pure ONE-event stop/hold/plan/revalidate/resume handshake.

    No SDK, ROS, map acquisition, planner, geometric validator or actuator is
    implemented here. Outputs are requests, NEVER command/drive authority.
    The future host must retain its ordinary physical watchdog throughout
    planning, authenticate fresh measurements, and supply the real independent
    full-state/envelope/support/stop/terminal validator. A digest or a caller's
    asserted check is not physical verification. No adopted H5 event/policy or
    successful replanning experiment is implied by this component.
    """

    CHECKS=('measured_initial_state','steering_continuity','body_dynamic_state',
        'full_envelope','support','stopping','terminal')
    POLICY_KEYS={'stop_speed_m_s','stop_yaw_rate_rad_s','stop_observation_s',
        'maximum_sample_gap_s','stop_timeout_s','planning_wall_budget_s',
        'mission_sim_budget_s','mission_wall_budget_s','resume_position_m',
        'resume_heading_rad','resume_steering_rad','resume_body_twist',
        'resume_drive_velocity_rad_s'}
    SAMPLE_KEYS={'time_s','pose_world_xyyaw','body_twist_u_v_r',
        'front_rear_steering_rad','front_rear_drive_velocity_rad_s',
        'zero_command_confirmed','hold_confirmed','retained_mode','retained_direction','planner_started'}

    @staticmethod
    def _number(value,name):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('Explicit finite scalar required: '+name)
        return float(value)

    @classmethod
    def _vector(cls,value,size,name):
        if type(value) not in (list,tuple) or len(value)!=size:
            raise ValueError('Explicit vector required: '+name)
        return [cls._number(v,name) for v in value]

    @staticmethod
    def _digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _sha(value):
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Exact lowercase SHA256 required')
        return value

    @classmethod
    def _identity(cls,identity):
        # Revalidate metadata/digest, NOT the unprovided grid or its provenance.
        value=copy.deepcopy(identity);record=value['inputs']
        expected={'version','map_frame_id','shape_rows_columns','origin_xyyaw_m_rad','resolution_m',
            'world_from_map_xyyaw','goal_world_xyyaw','environment_sha256','planner_contract_sha256',
            'signed_row_major_cells_sha256'}
        if (set(record)!=expected or record['version']!='PLANNING_INPUT_CONTENT_IDENTITY_NOT_EXECUTION_GATE_v1'
                or cls._sha(value['identity_sha256'])!=cls._digest(record)):
            raise ValueError('Unmodified complete planning identity required')
        shape=record['shape_rows_columns'];resolution=cls._number(record['resolution_m'],'resolution_m')
        frame=record['map_frame_id']
        if (type(shape) is not list or len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape)
                or math.prod(shape)>4_000_000 or not 0.<resolution<=float(np.finfo(np.float32).max)
                or float(np.float32(resolution))==0. or type(frame) is not str or not frame
                or frame.startswith('/') or any(c.isspace() for c in frame)):
            raise ValueError('Explicit bounded fixed planning frame/grid required')
        for key in ('origin_xyyaw_m_rad','world_from_map_xyyaw','goal_world_xyyaw'):cls._vector(record[key],3,key)
        for key in ('environment_sha256','planner_contract_sha256','signed_row_major_cells_sha256'):cls._sha(record[key])
        return value

    def __init__(self,identity,*,method,policy,allowed_environment_sha256,clock,
            mission_start_sim_s,mission_start_wall_s):
        if method not in ('B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
                'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'):
            raise ValueError('One explicit common-comparison method required')
        if type(policy) is not dict or set(policy)!=self.POLICY_KEYS or not callable(clock):
            raise ValueError('Complete explicit prospective policy and monotonic clock required')
        self.policy=copy.deepcopy(policy)
        for key,value in self.policy.items():
            values=self._vector(value,3,key) if key=='resume_body_twist' else [self._number(value,key)]
            if any(v<=0. for v in values):raise ValueError('Positive explicitly declared policy required')
        if (self.policy['stop_speed_m_s']>.005 or self.policy['maximum_sample_gap_s']>.25
                or self.policy['stop_timeout_s']<=self.policy['stop_observation_s']
                or self.policy['resume_position_m']>.003 or self.policy['resume_heading_rad']>math.radians(2.)):
            raise ValueError('Cannot weaken original stop/hold/control-gap limits in a replan adapter')
        self.initial=self._identity(identity);self.method=method
        if (type(allowed_environment_sha256) is not list or not allowed_environment_sha256
                or len(set(allowed_environment_sha256))!=len(allowed_environment_sha256)):
            raise ValueError('Explicit allowed environment revisions required, not a mutable baseline')
        self.environments=frozenset(self._sha(v) for v in allowed_environment_sha256)
        if self.initial['inputs']['environment_sha256'] not in self.environments:
            raise ValueError('Original environment must be explicitly retained')
        self.clock=clock;self.start_sim=self._number(mission_start_sim_s,'mission_start_sim_s')
        self.start_wall=self._number(mission_start_wall_s,'mission_start_wall_s')
        if self.start_sim<0. or self.start_wall<0.:raise ValueError('Nonnegative original mission clocks required')
        self.previous_sim=self.previous_wall=self.last_measurement_wall=self.latest_sample=None
        self.phase='FOLLOWING';self.failure=None;self.event=self.event_sim=self.event_wall=None
        self.stop_since=None;self.capture_count=0;self.request=None;self.request_wall=None
        self.plan=self.validation=self.snapshot=None;self.events=[]

    def _fail(self,error):
        self.phase='HOLD';self.failure=type(error).__name__+': '+str(error)
        self.events.append(dict(event='TERMINAL_HOLD',error=self.failure))

    def _wall(self):
        now=self._number(self.clock(),'monotonic_wall_s')
        if now<self.start_wall or (self.previous_wall is not None and now<self.previous_wall):
            raise RuntimeError('Wall clock regressed; no rebasing')
        self.previous_wall=now
        if now-self.start_wall>self.policy['mission_wall_budget_s']:raise TimeoutError('Whole mission wall budget includes replanning')
        if self.request_wall is not None and now-self.request_wall>self.policy['planning_wall_budget_s'] and self.phase=='PLANNING':
            raise TimeoutError('Planning/revalidation wall budget exhausted; no extension')
        return now

    def _same_start(self,sample):
        if self.snapshot is None:return
        initial=self.snapshot
        p,q=sample['pose_world_xyyaw'],initial['pose_world_xyyaw']
        if (math.hypot(p[0]-q[0],p[1]-q[1])>self.policy['resume_position_m']
                or abs(wrap(p[2]-q[2]))>self.policy['resume_heading_rad']
                or max(abs(a-b) for a,b in zip(sample['front_rear_steering_rad'],initial['front_rear_steering_rad']))>self.policy['resume_steering_rad']
                or max(abs(a-b) for a,b in zip(sample['front_rear_drive_velocity_rad_s'],initial['front_rear_drive_velocity_rad_s']))>self.policy['resume_drive_velocity_rad_s']
                or any(abs(a-b)>limit for a,b,limit in zip(sample['body_twist_u_v_r'],initial['body_twist_u_v_r'],self.policy['resume_body_twist']))):
            raise RuntimeError('Captured full planning start changed; no nominal-zero substitution or automatic replan')
        if any(sample[key]!=initial[key] for key in ('retained_mode','retained_direction','planner_started')):
            raise RuntimeError('Retained mode/direction/initialization history changed while planning')

    def observe(self,sample,identity):
        if self.phase in ('HOLD','NO_PATH','RESUMED'):raise RuntimeError('ONE replan boundary terminal; no automatic retry')
        try:return self._observe(sample,identity)
        except Exception as error:self._fail(error);raise

    def _observe(self,sample,identity):
        now=self._wall();row=copy.deepcopy(sample)
        if type(row) is not dict or set(row)!=self.SAMPLE_KEYS:raise ValueError('Complete actual measurement schema required')
        t=self._number(row['time_s'],'measurement_time_s')
        for key,size in (('pose_world_xyyaw',3),('body_twist_u_v_r',3),('front_rear_steering_rad',2),('front_rear_drive_velocity_rad_s',2)):
            row[key]=self._vector(row[key],size,key)
        if any(type(row[key]) is not bool for key in ('zero_command_confirmed','hold_confirmed')):
            raise ValueError('Actual backend ACK booleans required, not command-request echoes')
        if (row['retained_mode'] not in ('coordinated_counterphase','front_only','rear_only')
                or type(row['retained_direction']) is not int or row['retained_direction'] not in (-1,0,1)
                or row['planner_started'] is not True):
            raise ValueError('Explicit retained control history required; no first-start exception at a replanning boundary')
        if (t<self.start_sim or (self.previous_sim is not None and not 0.<t-self.previous_sim<=self.policy['maximum_sample_gap_s'])
                or t-self.start_sim>self.policy['mission_sim_budget_s']):
            raise RuntimeError('Whole mission physics clock/gap changed; no reset or omitted planning time')
        self.previous_sim=t;self.last_measurement_wall=now;self.latest_sample=copy.deepcopy(row)
        current=self._identity(identity)
        for key,value in self.initial['inputs'].items():
            if key not in ('signed_row_major_cells_sha256','environment_sha256') and current['inputs'][key]!=value:
                raise ValueError('Fixed map frame/goal/planner contract changed; do not reanchor the task')
        if current['inputs']['environment_sha256'] not in self.environments:raise ValueError('Unreviewed environment revision')
        if self.phase=='FOLLOWING':
            if current['identity_sha256']==self.initial['identity_sha256']:return self._output('FOLLOW_EXISTING_REFERENCE')
            self.event=current;self.event_sim=t;self.event_wall=now;self.phase='STOPPING'
            self.events.append(dict(event='OBSERVED_CONTENT_CHANGE',time_s=t,wall_s=now,identity_sha256=current['identity_sha256']))
        if current['identity_sha256']!=self.event['identity_sha256']:
            raise RuntimeError('New revision invalidates pending result; hold without automatic retry')
        speed=math.hypot(*row['body_twist_u_v_r'][:2]);yaw_rate=abs(row['body_twist_u_v_r'][2])
        stopped=row['zero_command_confirmed'] and speed<=self.policy['stop_speed_m_s'] and yaw_rate<=self.policy['stop_yaw_rate_rad_s']
        if self.phase=='STOPPING':
            if row['hold_confirmed']:raise RuntimeError('Unexpected hold ACK before this event capture request')
            if t-self.event_sim>self.policy['stop_timeout_s']:raise TimeoutError('Measured event stopping timeout')
            self.stop_since=t if stopped and self.stop_since is None else self.stop_since if stopped else None
            if self.stop_since is not None and t-self.stop_since>=self.policy['stop_observation_s']:
                self.capture_count+=1;self.phase='HOLD_REQUESTED'
                self.events.append(dict(event='ONE_CAPTURE_REQUESTED',time_s=t,wall_s=now))
                return self._output('REQUEST_CAPTURED_HOLD')
            return self._output('ZERO_UNTIL_MEASURED_STOP')
        if not stopped:raise RuntimeError('Measured standstill/zero acknowledgement lost while planning or validating')
        if self.phase=='HOLD_REQUESTED':
            if t-self.event_sim>self.policy['stop_timeout_s']:raise TimeoutError('Event hold ACK timeout')
            if not row['hold_confirmed']:return self._output('KEEP_ZERO_WAIT_HOLD_ACK')
            self.snapshot=row;self.phase='READY'
            self.events.append(dict(event='ACTUAL_HOLD_ACK',time_s=t,wall_s=now))
        if not row['hold_confirmed']:raise RuntimeError('Actual hold ACK lost; no recapture')
        self._same_start(row)
        if self.phase=='VALIDATED':
            self.phase='RESUME_READY';self.events.append(dict(event='FRESH_POSTVALIDATION_MEASUREMENT',time_s=t,wall_s=now))
        return self._output('RESUME_READY' if self.phase=='RESUME_READY' else 'KEEP_CAPTURED_HOLD')

    def take_request(self):
        if self.phase!='READY':raise RuntimeError('Measured ONE capture/ACK required before ONE planning request')
        try:
            now=self._wall()
            if now-self.last_measurement_wall>self.policy['maximum_sample_gap_s']:raise RuntimeError('Planning start observation stale')
            request=dict(method=self.method,input_identity=self.event,measured_start=self.snapshot,
                original_goal_world_xyyaw=self.initial['inputs']['goal_world_xyyaw'],
                event_observed_sim_s=self.event_sim,event_observed_wall_s=self.event_wall)
            self.request=dict(payload=copy.deepcopy(request),request_sha256=self._digest(request))
            self.request_wall=now;self.phase='PLANNING'
            return copy.deepcopy(self.request)
        except Exception as error:self._fail(error);raise

    def accept_result(self,plan,*,validator):
        if self.phase!='PLANNING':raise RuntimeError('Only own pending planning request may return; no late/duplicate result')
        try:
            self._wall()
            if type(plan) is not dict or plan.get('request_sha256')!=self.request['request_sha256'] or not callable(validator):
                raise ValueError('Exact own request-bound plan and independent validator required')
            candidate=copy.deepcopy(plan);digest=self._digest(candidate)
            if (candidate.get('method')!=self.method or candidate.get('initial_measurement')!=self.snapshot
                    or candidate.get('goal_world_xyyaw')!=self.initial['inputs']['goal_world_xyyaw']
                    or type(candidate.get('selected_actions')) is not list or not candidate['selected_actions']
                    or type(candidate.get('selected_states')) is not list
                    or len(candidate['selected_states'])!=len(candidate['selected_actions'])):
                raise ValueError('Complete own-method route with exact measured initial state and unchanged task goal required')
            audit=validator(copy.deepcopy(candidate),copy.deepcopy(self.request))
            self._wall()  # Validation duration also consumes the original budget.
            if self.phase!='PLANNING' or self.failure is not None:
                raise RuntimeError('Boundary invalidated during validation; no recovery by a late successful result')
            if (type(audit) is not dict or set(audit)!={'plan_sha256','request_sha256','input_identity_sha256','checks'}
                    or audit['plan_sha256']!=digest or audit['request_sha256']!=self.request['request_sha256']
                    or audit['input_identity_sha256']!=self.event['identity_sha256']
                    or type(audit['checks']) is not dict or set(audit['checks'])!=set(self.CHECKS)
                    or any(value is not True for value in audit['checks'].values())):
                raise ValueError('Full-state/envelope/support/stopping/terminal validation incomplete or stale')
            self.plan=candidate;self.validation=copy.deepcopy(audit);self.phase='VALIDATED'
            self.events.append(dict(event='REQUEST_BOUND_VALIDATION_RETURNED',wall_s=self.previous_wall,plan_sha256=digest))
            return self._output('KEEP_CAPTURED_HOLD_WAIT_FRESH_MEASUREMENT')
        except Exception as error:self._fail(error);raise

    def no_path(self,request_sha256):
        if self.phase!='PLANNING':raise RuntimeError('No pending planning request')
        try:
            self._wall()
            if request_sha256!=self.request['request_sha256']:raise ValueError('Foreign or stale no-path result')
            self.phase='NO_PATH';self.events.append(dict(event='ONE_PLAN_NO_PATH_NOT_GLOBAL_INFEASIBILITY',wall_s=self.previous_wall))
            return self._output('KEEP_ZERO_NO_PATH_NO_RETRY')
        except Exception as error:self._fail(error);raise

    def take_resume(self):
        if self.phase!='RESUME_READY':raise RuntimeError('Fresh postvalidation full state required before ONE resume request')
        try:
            now=self._wall()
            if now-self.last_measurement_wall>self.policy['maximum_sample_gap_s']:raise RuntimeError('Resume observation stale')
            self.phase='RESUMED';self.events.append(dict(event='ONE_RESUME_REQUEST_ONLY',wall_s=now))
            return dict(plan=copy.deepcopy(self.plan),request=copy.deepcopy(self.request),validation=copy.deepcopy(self.validation),
                resume_measurement=copy.deepcopy(self.latest_sample),resume_request_wall_s=now,
                boundary_trace=copy.deepcopy(self.events),resume_request_only_NOT_command_or_drive_admission=True,
                actual_motion_authorized=False,H5_online_replanning_success=False)
        except Exception as error:self._fail(error);raise

    def _output(self,action):
        return dict(phase=self.phase,action=action,capture_requests=self.capture_count,
            zero_command_requested=self.phase!='FOLLOWING',actual_motion_authorized=False,
            independent_validator_and_host_physical_watchdog_still_required=True)


def measured_replanning_kernel(original,request,carried_state,*,carried_state_frame_id):
    """Bind the ORIGINAL search/refinement to one measured, fixed-map start.

    Reuse the existing complete-state continuation helper; do not fork either
    original algorithm or change its propagation/keys/objective/goal. This is
    a kernel adapter, NOT a complete matched H5 planner, state estimator,
    prefix audit or collision/support validator. Extra modeled force/resource
    fields are CARRIED, not reclassified as measured forces or reset defaults.
    The host must supply the correctly framed/authenticated prefix state and
    retain all ordinary map/resource/deadline checks and candidate accounting.
    """
    from dataclasses import asdict,fields,is_dataclass,replace
    import inspect
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability import hierarchical_dynamics_planner as refinement
    from hospital_mission_planning import continue_original_lattice_from_state

    if (not inspect.isfunction(original) or all(original.__code__ is not code for code in
            (lattice.plan.__code__,refinement.refine_speed_consistent_lattice_route.__code__))
            or original.__closure__):
        raise ValueError('Exact original standalone lattice/search or speed-consistent refinement kernel required')
    if (type(request) is not dict or set(request)!={'payload','request_sha256'}
            or StoppedReplanBoundary._sha(request['request_sha256'])!=StoppedReplanBoundary._digest(request['payload'])):
        raise ValueError('Complete exact pending measured-state request required')
    payload=copy.deepcopy(request['payload'])
    if set(payload)!={'method','input_identity','measured_start','original_goal_world_xyyaw','event_observed_sim_s','event_observed_wall_s'}:
        raise ValueError('Original event/task/state payload required')
    if payload['method'] not in ('B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
            'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'):
        raise ValueError('Explicit matched-comparison method required')
    identity=StoppedReplanBoundary._identity(payload['input_identity']);meta=identity['inputs']
    row=payload['measured_start']
    if (type(row) is not dict or set(row)!=StoppedReplanBoundary.SAMPLE_KEYS
            or row['zero_command_confirmed'] is not True or row['hold_confirmed'] is not True
            or row['planner_started'] is not True or carried_state_frame_id!=meta['map_frame_id']):
        raise ValueError('Held actual measurement and exact fixed planning frame required, not nominal origin')
    pose=StoppedReplanBoundary._vector(row['pose_world_xyyaw'],3,'measured world pose')
    twist=StoppedReplanBoundary._vector(row['body_twist_u_v_r'],3,'measured BODY-frame twist')
    steer=StoppedReplanBoundary._vector(row['front_rear_steering_rad'],2,'measured axle steering')
    StoppedReplanBoundary._vector(row['front_rear_drive_velocity_rad_s'],2,'measured drive velocities')
    stamp=StoppedReplanBoundary._number(row['time_s'],'measurement_time_s')
    event=StoppedReplanBoundary._number(payload['event_observed_sim_s'],'event_time_s')
    if (not 0.<=event<=stamp or math.hypot(*twist[:2])>.005
            or StoppedReplanBoundary._number(payload['event_observed_wall_s'],'event_wall_s')<0.
            or payload['original_goal_world_xyyaw']!=meta['goal_world_xyyaw']):
        raise ValueError('Exact stopped event measurement and original world goal required')
    state_type=original.__globals__['State']
    if (not is_dataclass(carried_state) or type(carried_state) is not state_type
            or not carried_state.__dataclass_params__.frozen
            or carried_state.rigid_body_feasible is not True):
        raise ValueError('Own complete frozen feasible prefix-state class required; no reconstructed default history')
    if (row['retained_mode'] not in lattice.ALL_MODES or type(row['retained_direction']) is not int
            or row['retained_direction'] not in (-1,0,1)
            or type(carried_state.direction) is not int or carried_state.started is not True
            or (carried_state.mode,carried_state.direction,carried_state.started)!=
                (row['retained_mode'],row['retained_direction'],True)):
        raise ValueError('Measured request and retained mode/direction/started history disagree')
    for field in fields(carried_state):
        value=getattr(carried_state,field.name)
        if field.name in ('mode','started','direction','rigid_body_feasible'):continue
        StoppedReplanBoundary._number(value,'complete prefix '+field.name)
    local=transform_pose(meta['world_from_map_xyyaw'],pose,inverse=True)
    goal=transform_pose(meta['world_from_map_xyyaw'],meta['goal_world_xyyaw'],inverse=True)
    # Pose changes coordinates; body twist/steering MUST NOT be map-rotated.
    refreshed=dict(x=local[0],y=local[1],yaw=local[2],front=steer[0],rear=steer[1],
        longitudinal_velocity=twist[0],lateral_velocity=twist[1],yaw_rate=twist[2])
    seed=replace(carried_state,**refreshed)
    continued=continue_original_lattice_from_state(original,seed)
    signature=inspect.signature(continued)
    initial=dict(initial_mode=seed.mode,initial_front_rad=seed.front,initial_rear_rad=seed.rear,propagate_initial_steering=True)
    fixed_goal=tuple(goal)

    def bound(*args,**kwargs):
        values=signature.bind(*args,**kwargs)
        if tuple(StoppedReplanBoundary._vector(values.arguments.get('goal'),3,'original fixed-map goal'))!=fixed_goal:
            raise ValueError('Measured start binding cannot move the fixed task goal')
        for key,value in initial.items():
            if key in values.arguments and (values.arguments[key]!=value or
                    key=='propagate_initial_steering' and values.arguments[key] is not True):
                raise ValueError('Explicit initial-state argument contradicts the measured request: '+key)
            values.arguments[key]=value
        values.apply_defaults()
        if not callable(values.arguments.get('collision_checker')):
            raise ValueError('Original complete collision/support checker binding required, not bounds-only planning')
        result=continued(*values.args,**values.kwargs)
        route=result[0] if original.__code__ is refinement.refine_speed_consistent_lattice_route.__code__ else result
        if route is not None and any(type(state) is not state_type for state in route.states):
            raise RuntimeError('Downstream propagator dropped prefix-state/resource fields; bind the original resource adapter first')
        return result

    bound.measured_replanning=dict(request_sha256=request['request_sha256'],method=payload['method'],
        kernel_name=original.__name__,fixed_map_frame_id=meta['map_frame_id'],
        world_from_map_xyyaw=copy.deepcopy(meta['world_from_map_xyyaw']),
        unchanged_goal_world_xyyaw=copy.deepcopy(meta['goal_world_xyyaw']),fixed_goal_local_xyyaw=list(fixed_goal),
        initial_state=asdict(seed),refreshed_measurement_fields=sorted(refreshed),
        preserved_prefix_fields={field.name:getattr(carried_state,field.name) for field in fields(carried_state) if field.name not in refreshed},
        original_kernel_code_object_reused=continued.__code__ is original.__code__,
        body_twist_NOT_rotated_to_map_axes=True,world_task_NOT_reanchored_to_current_pose=True,
        first_start_exception_NOT_reenabled=True,actual_motion_authorized=False,
        prefix_provenance_resource_feasibility_and_independent_route_audit_NOT_proven_here=True)
    return bound


def h5_cart_information_maps(geometry,cart_inventory,occupied,origin,resolution):
    """Remove ONLY exclusive cart information, never physics or overlapping cells.

    Reconstruct the selected-cart patch from the original authenticated AABBs
    AND authored support. Reject disagreement with the current complete map;
    no generic rectangle erase, unknown-to-free or obstacle/collider edit.
    """
    from hospital_geometry_tools import polygon_cells
    from hospital_mission_geometry import disjoint_support_rectangles
    full=np.asarray(occupied)
    if (full.dtype!=np.dtype(bool) or full.ndim!=2 or resolution!=.010
            or hashlib.sha256(full.tobytes()).hexdigest()!=geometry['planning_occupancy_digest']
            or geometry['scenario']!='H5_obstacle_bypass_return'):
        raise ValueError('Exact complete current H5 raster required')
    rows=cart_inventory['included_current_obstacle_AABBs']
    selected=[r for r in rows if r['path']==H5_REPLAN_CART]
    if len(selected)!=1:raise ValueError('One authenticated original stationary cart required')
    def cells(row):
        b=row['world_aligned_bound'];a,z=b['minimum_m'],b['maximum_m']
        return polygon_cells([(a[0],a[1]),(z[0],a[1]),(z[0],z[1]),(a[0],z[1])],origin,resolution,full.shape)
    mask=np.zeros(full.shape,bool);rr,cc=cells(selected[0]);mask[rr,cc]=True
    others=np.zeros(full.shape,bool)
    for row in rows:
        if row['path']!=H5_REPLAN_CART:
            r,c=cells(row);others[r,c]=True
    # Whole cells, including rectangle joins, must remain fully supported.
    x=origin[0]+cc*resolution;y=origin[1]+rr*resolution;area=np.zeros(len(rr))
    for lo,hi in disjoint_support_rectangles(geometry['same_three_authored_support_rectangles']):
        area+=np.maximum(0.,np.minimum(x+resolution,hi[0])-np.maximum(x,lo[0]))*np.maximum(0.,np.minimum(y+resolution,hi[1])-np.maximum(y,lo[1]))
    removable=~others[rr,cc]&(area>=resolution**2-1e-12)
    if not full[rr,cc].all() or not removable.any():
        raise ValueError('Current selected-cart contribution disagrees or has no exclusive known cells')
    initial=full.copy();initial[rr[removable],cc[removable]]=False
    if (np.any(initial&~full) or np.any((initial!=full)&~mask)):
        raise RuntimeError('Only selected-cart information may differ')
    return initial,dict(cart=copy.deepcopy(selected[0]),cart_cells=int(mask.sum()),
        exclusive_cart_information_cells_removed=int(removable.sum()),
        overlapping_obstacles_and_unsupported_cells_preserved=True,
        initial_digest=hashlib.sha256(initial.tobytes()).hexdigest(),
        discovered_digest=hashlib.sha256(full.tobytes()).hexdigest(),actual_scene_unchanged=True)


def h5_measured_guided_plan(geometry,profile,occupied,origin,resolution,initial_state=None,*,budget_s=180.):
    """Own bounded P seed/refinement from a complete fixed-frame state.

    No measured state is invented here. The host supplies its recorded prefix
    resource state. Existing geometry/NE/key/goal/resource rules are retained.
    This helper runs offline or in a separate worker, never inside a Kit tick.
    """
    import time
    from dataclasses import asdict,fields
    from types import FunctionType
    import audit_hospital_paper_readiness as audit
    import hospital_mission_planning as resource
    import plan_hospital_h1 as original
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    method,_,connections,parameters=audit.unified_h3_planner_binding(profile,geometry,occupied,origin,resolution,
        terminal_direction_change_reserve=2,initial_state=initial_state)
    refine=method.__globals__['refine_speed_consistent_lattice_route'];State=refine.__globals__['State']
    if initial_state is not None and set(initial_state)!={f.name for f in fields(State)}:
        raise ValueError('Every complete prefix resource/body/steering field is required; no default history')
    initial=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True) if initial_state is None else State(**initial_state)
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    began=time.perf_counter()
    seed=audit.unified_P_owned_seed(method,geometry,occupied,origin,resolution,model,
        budget_s=budget_s,node_limit=60000,policy=audit.UNIFIED_P_SEED_POLICY)
    remaining=budget_s-(time.perf_counter()-began)
    if remaining<=0.:raise TimeoutError('Own shared planning budget consumed by seed')
    result=audit.unified_h3_guided_candidate(method,seed,geometry,occupied,origin,resolution,model,
        budget_s=remaining,execution_budget_s=240.,initial_state=initial if initial_state is not None else None)
    result.update(scenario=geometry['scenario'],geometry_profile=copy.deepcopy(profile),
        start_world_xyyaw=geometry['start_xyyaw'],goal_world_xyyaw=geometry['goal_xyyaw'],
        goal_local_xyyaw=geometry['local_goal_xyyaw'],planning_occupancy_digest=hashlib.sha256(occupied.tobytes()).hexdigest(),
        full_envelope_clearance_m=.010,common_max_command_speed_m_s=.05,
        rigid_body_parameter_bounds=asdict(model.bounds),actual_motion_authorized=False,use_for_final_statistics=False,
        physical_execution_verified=False,Nav2_execution_verified=False,
        integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True,
        planning_time_s=time.perf_counter()-began,planning_wall_budget_s=budget_s,
        initial_complete_state=asdict(initial),seed_attempts=seed['owned_seed_attempts'],
        fixed_planning_frame_world_xyyaw=list(geometry['start_xyyaw']),no_physical_state_reset=True)
    return result


def audit_h5_measured_plan(plan,geometry,occupied,origin,resolution):
    """Independent ORIGINAL Python NE/SAT/support/resource replay from capture."""
    from dataclasses import asdict
    import hospital_mission_planning as resource
    from audit_hospital_paper_readiness import unified_wheelbase_kernel
    from hospital_geometry_tools import pose_collision
    from hospital_h4_cohort_preparation_v4 import support_area_missing
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    if not plan.get('plan_success') or plan['goal_world_xyyaw']!=geometry['goal_xyyaw']:
        raise ValueError('Own complete plan to the unchanged original H5 goal required')
    kernel=unified_wheelbase_kernel(plan['geometry_profile'])
    State,propagate,_=resource.resource_propagation_binding(lattice.State,kernel.propagate,resource.MotionResourceLimits(maximum_direction_changes=2))
    state=State(**plan['initial_complete_state']);model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**plan['rigid_body_parameter_bounds']))
    collisions=unsupported=forces=resources=samples=0;error=0.;states=[]
    for action,saved in zip(plan['selected_actions'],plan['selected_states']):
        state,poses,_=propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
            speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=action['target_speed_m_s'])
        forces+=not state.rigid_body_feasible;resources+=not poses.resource_feasible
        for pose in poses:
            world=transform_pose(geometry['start_xyyaw'],pose);samples+=1
            collisions+=pose_collision(occupied,world,origin,resolution,.060)
            unsupported+=support_area_missing(world,geometry['same_three_authored_support_rectangles'])>1e-9
        error=max(error,abs(state.x-saved['x_m']),abs(state.y-saved['y_m']),abs(wrap(state.yaw-math.radians(saved['yaw_deg']))),
            abs(state.front-saved['front_rad']),abs(state.rear-saved['rear_rad']),
            abs(state.longitudinal_velocity-saved['u_m_s']),abs(state.lateral_velocity-saved['v_m_s']),abs(state.yaw_rate-saved['yaw_rate_rad_s']))
        states.append(asdict(state))
    goal=geometry['local_goal_xyyaw'];checks=dict(complete_actions=len(plan['selected_actions'])==len(plan['selected_states'])>0,
        original_python_NE_matches=error<=1e-10,full_SAT_10mm_plus50mm_planning_reserve=collisions==0,
        full_authored_support=unsupported==0,all_NE_edges_feasible=forces==0,retained_resource_history_feasible=resources==0,
        unchanged_goal_position_25mm=math.hypot(state.x-goal[0],state.y-goal[1])<=.025,
        unchanged_goal_heading_2deg=abs(wrap(state.yaw-goal[2]))<=math.radians(2.),
        planning_terminal_speed_5mm_s=math.hypot(state.longitudinal_velocity,state.lateral_velocity)<=.005)
    return dict(verdict='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},
        independent_substeps=samples,maximum_state_difference=error,complete_model_states=states,
        NOT_PhysX_or_sensor_or_Nav2_validation=True)


H5_REPLAN_SCOPE=dict(version='H5_ONE_STATIC_CART_DISCOVERY_MEASURED_STOP_PAUSED_REPLAN_RESUME_v1',
    approval=dict(approval_source='USER_MESSAGE',user_message='별도 H5 정지 후 재계획 검증은 남아 있고 이것만 빠르게 일단은 진행을 하자'),
    cart=H5_REPLAN_CART,discovery_full_body_separation_m=.75,simulation_budget_s=240.,wall_budget_s=900.,
    planning_wall_budget_s=180.,maximum_discoveries=1,maximum_replans=1,maximum_resumes=1,
    actual_cart_stationary_and_collision_enabled_throughout=True,
    initial_map_omits_ONLY_cart_information=True,full_map_physical_guard_active_from_start=True,
    real_Pause_during_planning=True,whole_clock_NOT_reset_on_resume=True,
    retain_actual_pose_steering_twist_and_prefix_resources=True,
    final_goal_and_controller_and_original_safety_unchanged=True,automatic_retry=False,
    ground_truth_query_NOT_sensor_localization_Nav2_or_moving_obstacle_validation=True,
    separate_functional_development_case_NOT_final_repetition_or_four_method_comparison=True)


def original_nominal_execution_ast(tree):
    """Prove old plans' full controller AST by undoing exact opt-in additions.

    Used ONLY to resolve consumed source archives. It does not admit a route,
    waive a physical change, or turn historical data into a new execution.
    """
    def dump(node):return ast.dump(node,include_attributes=False)
    def expr(text):return ast.parse(text,mode='eval').body
    added=ast.parse('''measured=plan.get('measured_replan_start')
nominal=(plan.get('initial_front_rad')==0. and plan.get('initial_rear_rad')==0. and plan.get('initial_u_v_r')==[0.,0.,0.])
if measured is not None:
    from hospital_sensor_navigation import checked_h5_execution_replan_start
    checked_h5_execution_replan_start(plan,geometry)
''').body
    ctor=ast.parse('''if plan.get('measured_replan_start') is not None:
    captured=plan['measured_replan_start']
    self.mode=captured['retained_mode'];self.curvature=captured['retained_curvature_1_m']
    self.stage='release_hold'
''').body[0]
    held_ctor=ast.parse('''if plan.get('measured_replan_start') is not None:
    captured=plan['measured_replan_start']
    from hospital_sensor_navigation import h5_captured_command_curvature
    self.mode=captured['retained_mode'];self.curvature=h5_captured_command_curvature(captured,plan['geometry_profile'])
    self.stage='release_hold'
''').body[0]
    removed=[];predicates=0;ack=0
    class Undo(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            nonlocal removed
            if node.name=='validate_mission_plan':
                for wanted in added:
                    matches=[i for i,n in enumerate(node.body) if dump(n)==dump(wanted)]
                    if len(matches)!=1:raise ValueError('Exact opt-in H5 validator additions required')
                    node.body.pop(matches[0]);removed.append('validator')
            for i,n in reversed(list(enumerate(node.body))):
                if dump(n) in (dump(ctor),dump(held_ctor)):node.body.pop(i);removed.append('constructor')
            return self.generic_visit(node)
        def visit_BoolOp(self,node):
            nonlocal predicates
            if isinstance(node.op,ast.Or):
                values=[]
                for child in node.values:
                    if dump(child)==dump(expr('not nominal and measured is None')):
                        predicates+=1
                        values.extend(expr(t) for t in ("plan.get('initial_front_rad')!=0.","plan.get('initial_rear_rad')!=0.","plan.get('initial_u_v_r')!=[0.,0.,0.]"))
                    else:values.append(child)
                node.values=values
            return self.generic_visit(node)
        def visit_Assign(self,node):
            nonlocal ack
            if dump(node.value)==dump(expr("candidate.get('measured_replan_start') is not None")):
                if dump(node.targets[0])!=dump(expr('self.software_previous_request_ACK')):
                    # AST assignment targets have Store context.
                    if not isinstance(node.targets[0],ast.Attribute) or node.targets[0].attr!='software_previous_request_ACK':
                        raise ValueError('Only explicit model ACK initialization addition allowed')
                node.value=ast.Constant(False);ack+=1
            return self.generic_visit(node)
    tree=Undo().visit(copy.deepcopy(tree))
    if sorted(removed)!=['constructor','validator','validator','validator'] or predicates!=1 or ack!=1:
        raise ValueError('Exact complete H5 opt-in additions required; no broad source exemption')
    return tree


def checked_h5_replanning_scope(protocol):
    if (protocol.get('stopped_replanning')!=H5_REPLAN_SCOPE
            or protocol.get('scenario')!='H5_obstacle_bypass_return'
            or protocol.get('method')!='P_dynamics_full_envelope'
            or protocol.get('maximum_Play_calls')!=2 or protocol.get('simulation_budget_s')!=240.
            or protocol.get('maximum_wall_time_s')!=900.
            or protocol.get('maximum_Stop_reset_restaging_calls')!=0
            or protocol.get('automatic_retry') is not False):
        raise ValueError('Exact separately authorized ONE H5 stop/replan/resume scope required')
    return copy.deepcopy(H5_REPLAN_SCOPE)


def base_origin_body_twist(snapshot,root_com_local_m):
    """PhysX CoM WORLD velocity -> base_link ORIGIN BODY twist (SI)."""
    v=vector(snapshot['velocity'],6,'actual root CoM twist')
    q=vector(snapshot['transform'][3:],4,'actual root XYZW quaternion')
    rotation=rotation_wxyz([q[3],*q[:3]])
    offset=rotation@vector(root_com_local_m,3,'SDK root local CoM')
    origin_world=v[:3]-np.cross(v[3:],offset)
    body=rotation.T@origin_world
    return [float(body[0]),float(body[1]),float((rotation.T@v[3:])[2])]


def checked_h5_execution_replan_start(plan,geometry):
    h=plan.get('measured_replan_start',{});initial=h.get('initial_complete_state',{})
    if (h.get('version')!='H5_MEASURED_CAPTURE_EXECUTION_VIEW_v1'
            or plan.get('scenario')!='H5_obstacle_bypass_return' or plan.get('method')!='P_dynamics_full_envelope'
            or h.get('same_original_goal_world_xyyaw')!=plan.get('goal_world_xyyaw')
            or geometry.get('goal_xyyaw')!=plan.get('goal_world_xyyaw')
            or h.get('actual_PAUSED_capture') is not True or h.get('no_physical_reset') is not True
            or h.get('model_audit_verdict')!='PASS'
            or plan.get('start_world_xyyaw')!=h.get('captured_world_xyyaw')
            or plan.get('initial_front_rad')!=initial.get('front') or plan.get('initial_rear_rad')!=initial.get('rear')
            or plan.get('initial_u_v_r')!=[initial.get(k) for k in ('longitudinal_velocity','lateral_velocity','yaw_rate')]
            or initial.get('started') is not True or initial.get('rigid_body_feasible') is not True
            or math.hypot(*plan['initial_u_v_r'][:2])>.005):
        raise ValueError('Exact stopped measured replan execution view required; no nominal initial substitution')
    for key in ('request_sha256','fixed_frame_plan_sha256','prefix_trace_sha256'):
        StoppedReplanBoundary._sha(h[key])
    if (h.get('retained_mode') not in ('coordinated_counterphase','front_only','rear_only')
            or not math.isfinite(h.get('retained_curvature_1_m',float('nan')))
            or abs(h['retained_curvature_1_m'])>.7
            or max(abs(v) for v in vector(h.get('captured_steering_targets_rad'),2,'actual captured steering targets'))>math.radians(60.)):
        raise ValueError('Actual retained steering/mode/curvature required')
    if max(abs(v) for v in vector(plan['initial_u_v_r'],3,'measured initial body twist'))>.050:
        raise ValueError('Invalid measured initial body state')
    policy=h.get('goal_distance_handoff_policy')
    if policy is not None and policy!='H5_FINAL_REMAINING_MAX_REFERENCE_AND_ACTUAL_TASK_GOAL_v1':
        raise ValueError('Only exact explicit H5 goal-distance handoff policy allowed')
    return h


def h5_goal_distance_tracking(function,goal):
    """One opt-in final-leg distance binding; every control rule is retained.

    A valid plan may end up to25mm from the TASK goal. Treating its last25mm
    as the last25mm to that goal can brake about50mm early. Use the maximum
    of remaining reference arc and measured task-goal distance for the SAME
    original handoff and braking formulas. No reference extension, extra
    actuator command, new gain or relaxed acceptance is introduced.
    """
    goal=vector(goal,3,'unchanged actual H5 task goal')
    from hospital_mission_sequence import _tracking_floor_candidate_implementation
    original=_tracking_floor_candidate_implementation(.020,physical_direction_conditioned=True)
    if function.__code__!=original.__code__ or function.mission_binding!=original.mission_binding:
        raise ValueError('Exact original directional common tracking function required')
    return _tracking_floor_candidate_implementation(.020,physical_direction_conditioned=True,task_goal_xyyaw=goal)


def h5_captured_command_curvature(capture,profile):
    """Recover the held command, NOT the unslewed tracking intention.

    Discovery froze the two actual backend steering targets. Their curvature
    may differ from the tracker request because finite steering slew was in
    progress. Resume the release handshake at those SAME targets; do not
    reinterpret a request as an achieved command or change measured state.
    A pair incompatible with its retained mode must remain blocked.
    """
    from audit_hospital_paper_readiness import unified_wheelbase_connections
    targets=vector(capture.get('captured_steering_targets_rad'),2,'captured backend steering targets')
    mode=capture.get('retained_mode')
    if (mode not in ('coordinated_counterphase','front_only','rear_only')
            or max(abs(v) for v in targets)>math.radians(60.)):
        raise ValueError('Bounded actual held targets and retained steering mode required')
    connections=unified_wheelbase_connections(profile)
    wheelbase=profile['geometry']['wheelbase_m']
    curvature=float((math.tan(targets[0])-math.tan(targets[1]))/wheelbase)
    reproduced=connections.steering_angles(curvature,mode)
    if (not math.isfinite(curvature) or abs(curvature)>.7
            or max(abs(a-b) for a,b in zip(targets,reproduced))>1e-10):
        raise ValueError('Captured target pair does not reproduce its retained mode; no fabricated alignment')
    return curvature


def h5_execution_view(plan,geometry,capture,*,request_sha256,prefix_trace_sha256,audit):
    """Re-express a FIXED-frame plan for tracking, never reanchor its search."""
    if audit.get('verdict')!='PASS':raise ValueError('Independent measured-start plan audit required')
    initial=plan['initial_complete_state']
    measured_world=transform_pose(geometry['start_xyyaw'],[initial['x'],initial['y'],initial['yaw']])
    if (max(abs(a-b) for a,b in zip(measured_world,capture['pose_world_xyyaw']))>1e-10
            or [initial['front'],initial['rear']]!=capture['front_rear_steering_rad']
            or [initial[k] for k in ('longitudinal_velocity','lateral_velocity','yaw_rate')]!=capture['body_twist_u_v_r']):
        raise ValueError('Returned full initial state must exactly match its actual capture')
    pose=list(capture['pose_world_xyyaw']);view=copy.deepcopy(plan);g=copy.deepcopy(geometry)
    view['start_world_xyyaw']=g['start_xyyaw']=pose
    view['goal_local_xyyaw']=g['local_goal_xyyaw']=transform_pose(pose,geometry['goal_xyyaw'],inverse=True)
    for state in view['selected_states']:
        local=transform_pose(pose,state['world_xyyaw'],inverse=True)
        state.update(x_m=local[0],y_m=local[1],yaw_deg=math.degrees(local[2]))
    view['measured_replan_start']=dict(version='H5_MEASURED_CAPTURE_EXECUTION_VIEW_v1',
        captured_world_xyyaw=pose,same_original_goal_world_xyyaw=geometry['goal_xyyaw'],
        initial_complete_state=copy.deepcopy(plan['initial_complete_state']),
        fixed_frame_plan_sha256=StoppedReplanBoundary._digest(plan),request_sha256=request_sha256,
        prefix_trace_sha256=prefix_trace_sha256,actual_PAUSED_capture=True,no_physical_reset=True,
        model_audit_verdict=audit['verdict'],execution_coordinates_only_changed_NOT_planning_frame=True)
    view['measured_replan_start'].update(retained_mode=capture['retained_mode'],
        retained_curvature_1_m=capture['retained_curvature_1_m'],
        captured_steering_targets_rad=list(capture['captured_steering_targets_rad']),
        goal_distance_handoff_policy='H5_FINAL_REMAINING_MAX_REFERENCE_AND_ACTUAL_TASK_GOAL_v1')
    from hospital_mission_execution import validate_mission_plan
    validate_mission_plan(view,g)
    return view,g


class H5DiscoveryStop:
    """One bounded deceleration and measured capture, retaining steering."""
    def __init__(self,info,timestamp):
        self.previous=float(timestamp);self.began=float(timestamp);self.speed=float(info['speed_m_s'])
        self.mode=info['mode'];self.curvature=info['curvature_1_m'];self.quiet=None
        self.capture_requested=False;self.capture_stamp=None;self.ready=False

    def step(self,timestamp,speed,ack):
        from hospital_path_execution import ramp
        dt=timestamp-self.previous
        if (type(ack) is not bool or not all(math.isfinite(v) for v in (timestamp,speed)) or speed<0.
                or not 0.<dt<=.25 or timestamp-self.began>8.):raise RuntimeError('Original8s discovery stop/control-gap guard')
        if ack and not self.capture_requested:raise RuntimeError('Unexpected hold ACK before measured discovery capture')
        self.previous=timestamp;self.speed=ramp(self.speed,0.,dt,.020)
        if not self.capture_requested:
            stopped=abs(self.speed)<1e-7 and speed<=.005
            self.quiet=timestamp if stopped and self.quiet is None else self.quiet if stopped else None
            if self.quiet is not None and timestamp-self.quiet>=.25 and speed<=.003:
                self.capture_requested=True;self.capture_stamp=timestamp
        elif speed>.005:raise RuntimeError('Discovery captured hold lost original5mm/s standstill')
        self.ready=self.capture_requested and ack is True and timestamp>self.capture_stamp
        return dict(phase='path_tracking',terminal_stage='discovery_captured_hold' if self.capture_requested else 'discovery_braking',
            speed_m_s=self.speed,mode=self.mode,curvature_1_m=self.curvature,
            drive_hold_requested=self.capture_requested,freeze_steering_targets=True,
            hold_ready=False,progress_m=0.,goal_position_error_m=0.,goal_heading_error_deg=0.,lifecycle_elapsed_s=timestamp)


def h5_prefix_carried_state(initial_plan,initial_audit,rows,capture,resources):
    """Measured fields replace predictions; prefix resource history never resets.

    Force/utilization fields are the carried MODEL prefix, not measured motor
    guarantees. Actual full-prefix contacts/resources remain separately guarded.
    """
    if not rows or initial_audit['verdict']!='PASS':raise ValueError('Actual prefix and audited own model required')
    progress=max(r['path_progress_m'] for r in rows)
    from hospital_mission_execution import MissionPathReference
    reference=MissionPathReference(initial_plan)
    index=max(0,min(len(initial_plan['selected_actions'])-1,int(np.searchsorted(reference.arc,progress)-1)))
    state=copy.deepcopy(initial_audit['complete_model_states'][index])
    local=transform_pose(initial_plan['start_world_xyyaw'],capture['pose_world_xyyaw'],inverse=True)
    state.update(x=local[0],y=local[1],yaw=local[2],front=capture['front_rear_steering_rad'][0],
        rear=capture['front_rear_steering_rad'][1],longitudinal_velocity=capture['body_twist_u_v_r'][0],
        lateral_velocity=capture['body_twist_u_v_r'][1],yaw_rate=capture['body_twist_u_v_r'][2],
        mode=capture['retained_mode'],direction=capture['retained_direction'],started=True,
        mission_reverse_episode_bins=math.ceil(resources['maximum_sampled_reverse_episode_m']/.005),
        mission_reverse_total_bins=math.ceil(resources['sampled_reverse_centre_polyline_total_m']/.005),
        mission_direction_changes=resources['command_direction_changes'])
    return state


def h5_worker(request_file):
    """Separate native process: one measured suffix, independent audit, 2 dt screens."""
    import csv,time
    from audit_hospital_paper_readiness import unified_mission_model_binding
    request_file=Path(request_file).resolve();request=json.loads(request_file.read_bytes());out=request_file.parent
    began=time.monotonic();g=json.loads(Path(request['geometry_file']).read_bytes())
    for file,digest in request['input_sha256'].items():
        if sha(file)!=digest:raise ValueError('Replanning input changed: '+file)
    with np.load(g['occupancy_grid']) as z: occupied=z['occupied'].copy();origin=z['origin_xy_m'].copy();res=float(z['resolution_m'])
    # Leave a bounded part of the SAME180s clock for audit/model/revalidation.
    plan=h5_measured_guided_plan(g,request['profile'],occupied,origin,res,request['carried_state'],budget_s=150.)
    evidence.write_exclusive(out/'fixed_frame_replan.json',plan)
    if not plan['plan_success']:raise RuntimeError('No measured-start P suffix inside own planning budget')
    audit=audit_h5_measured_plan(plan,g,occupied,origin,res)
    evidence.write_exclusive(out/'independent_replan_audit.json',audit)
    view,view_geometry=h5_execution_view(plan,g,request['capture'],request_sha256=sha(request_file),
        prefix_trace_sha256=request['prefix_trace_sha256'],audit=audit)
    evidence.write_exclusive(out/'resumed_execution_plan.json',view)
    evidence.write_exclusive(out/'resumed_execution_geometry.json',view_geometry)
    models=[]
    for dt in (.020,.010):
        run,_,design=unified_mission_model_binding(request['profile'],view_geometry,view,simulation_budget_s=240.)
        report,rows=run(view,view['rigid_body_parameter_bounds'],dt,occupied,origin,res,design)
        trace=out/('resume_model_'+str(round(dt*1000))+'ms.csv');h5_write_trace(trace,rows)
        report.update(trace=str(trace),trace_sha256=sha(trace),
            original_mission_prefix_plus_nominal_suffix_s=request['capture']['time_s']+report['simulation_duration_s'],
            captured_wheel_contact_forces_NOT_modelled=True)
        from hospital_mission_lifecycle import normalize_runtime_evidence
        report=normalize_runtime_evidence(report)
        evidence.write_exclusive(trace.with_suffix('.json'),report);models.append(report)
        if report['model_status']!='MODEL_SCREEN_PASS':raise RuntimeError('Own measured suffix executor model did not pass')
        if report['original_mission_prefix_plus_nominal_suffix_s']>240.:raise TimeoutError('Original whole H5 physics clock has insufficient nominal time remaining')
    elapsed=time.monotonic()-began
    if elapsed>180.:raise TimeoutError('Replan+audit+model exceeded same180s wall allowance')
    for file,digest in request['input_sha256'].items():
        if sha(file)!=digest:raise ValueError('Replanning input changed during worker')
    files=[out/'fixed_frame_replan.json',out/'independent_replan_audit.json',out/'resumed_execution_plan.json',out/'resumed_execution_geometry.json',
        *[out/('resume_model_'+str(dt)+'ms'+suffix) for dt in (20,10) for suffix in ('.csv','.json')]]
    evidence.write_exclusive(out/'worker_complete.json',dict(status='MEASURED_SUFFIX_AND_INDEPENDENT_AUDIT_AND_TWO_MODEL_SCREENS_PASS',
        request_sha256=sha(request_file),planning_audit_model_wall_s=elapsed,
        output_sha256={str(p):sha(p) for p in files},model_cases=models,actual_motion_authorized=False))
    print('[H5 REPLAN] measured suffix + independent audit + both model resolutions PASS',flush=True)


def h5_write_trace(path,rows):
    import csv,os
    if not rows or any(set(r)!=set(rows[0]) for r in rows):raise ValueError('Complete consistent evidence trace required')
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())


def h5_query_selected_cart(port,pose,cart_bound):
    """Actual cached PhysX ray, NOT a simulated sensor/localization signal."""
    import carb
    from hospital_h4_geometry_metrics import rectangle_aabb_distances
    separation=float(rectangle_aabb_distances([pose],cart_bound,clearance=0.)[0])
    lo,hi=cart_bound['minimum_m'],cart_bound['maximum_m']
    target=np.array([(lo[0]+hi[0])/2.,(lo[1]+hi[1])/2.,.35])
    origin=np.array([pose[0],pose[1],.35]);delta=target-origin;length=float(np.linalg.norm(delta));direction=delta/length
    hits=[];errors=[]
    def receive(hit):
        try:
            row=dict(collision=str(hit.collision),distance_m=float(hit.distance),position_m=list(map(float,hit.position)))
            if not math.isfinite(row['distance_m']) or not np.isfinite(row['position_m']).all():raise ValueError('Nonfinite actual cached hit')
            hits.append(row)
        except Exception as error:errors.append(str(error))
        return True
    port.sdk.omni.physx.get_physx_scene_query_interface().raycast_all(carb.Float3(*origin),carb.Float3(*direction),length+.05,receive)
    if errors:raise RuntimeError('Actual cart query failed: '+str(errors))
    seen=any(h['collision']==H5_REPLAN_CART or h['collision'].startswith(H5_REPLAN_CART+'/') for h in hits)
    return dict(time_s=port.clock,origin_world_m=origin.tolist(),direction_world=direction.tolist(),length_m=length+.05,
        all_hits=sorted(hits,key=lambda h:(h['distance_m'],h['collision'])),selected_cart_hit=seen,
        full_body_to_selected_cart_AABB_m=separation,NOT_sensor_or_visibility_occlusion_validation=True)


async def run_h5_stopped_replanning_once(route_path,initial_plan_path,initial_audit_path,worker_prefix,*,authorized=False,
        unstarted_preflight_review=None):
    """ONE continuous H5 mission: approach/stop/Pause/one plan/resume/one hold.

    The initial information map is NEVER the physical safety map. No Stop,
    reset, actor construction or gain restoration occurs at the replan seam.
    The original wall deadline includes planning; simulation clock is retained.
    """
    import asyncio,time,subprocess
    import hospital_mission_sequence as seq
    lc=seq.lifecycle;rt=seq.retained
    if authorized is not True:raise ValueError('Explicit separate H5 replan validation authority required')
    context,packet,binding,_,_=seq.load_route(route_path)
    initial_plan_path=Path(initial_plan_path).resolve();initial=json.loads(initial_plan_path.read_bytes())
    initial_audit_path=Path(initial_audit_path).resolve();initial_audit=json.loads(initial_audit_path.read_bytes())
    full_geometry=json.loads(Path(packet['geometry_file']).read_bytes());geometry=copy.deepcopy(full_geometry)
    geometry['planning_occupancy_digest']=initial['planning_occupancy_digest']
    inv_path=GATE/'hospital_h5_geometry_candidates/common_map/e16a2898d38c414b/common_map_candidate.json'
    inventory=json.loads(inv_path.read_bytes())
    with np.load(packet['grid_file']) as z: full=z['occupied'].copy();origin=z['origin_xy_m'].copy();res=float(z['resolution_m'])
    omitted,map_review=h5_cart_information_maps(full_geometry,inventory,full,origin,res)
    if (initial_audit['verdict']!='PASS' or initial['planning_occupancy_digest']!=map_review['initial_digest']
            or initial['start_world_xyyaw']!=full_geometry['start_xyyaw']
            or initial['goal_world_xyyaw']!=full_geometry['goal_xyyaw']):
        raise ValueError('Audited own unseen-cart initial plan on exact original H5 task required')
    output=Path(packet['output_directory']);output.mkdir(parents=True,exist_ok=True)
    if unstarted_preflight_review is not None:
        review_path=Path(unstarted_preflight_review).resolve();review=json.loads(review_path.read_bytes())
        old=output/'replanning_sdk_binding.json';blocked=Path(review['original_block_record'])
        if (review.get('version')!='H5_UNSTARTED_APPROVAL_REPRESENTATION_REVIEW_v1'
                or review.get('original_binding_sha256')!=sha(old)
                or json.loads(blocked.read_bytes()).get('error')!='ValueError: Explicit same-limits retained-final-target development protocol required'
                or any((output/name).exists() for name in ('single_route_reserved.json','route_readback.json','route_trace.csv'))
                or json.loads(old.read_bytes()).get('stopped_replanning')!=H5_REPLAN_SCOPE):
            raise ValueError('Only the exact UNSTARTED preflight block may continue; no physical retry')
        output=output/'first_mission_after_preflight_review_r01';output.mkdir(exist_ok=False)
        binding.update(output_directory=str(output),unstarted_approval_representation_review=str(review_path))
    if (output/'single_route_reserved.json').exists():return dict(status='PRIOR_H5_REPLAN_DISPATCH_RETAINED_NO_NEW_ACTION')
    rt.no_other_task()
    plan_file=output/'initial_information_plan.json';geometry_file=output/'initial_information_geometry.json';grid_file=output/'initial_information_grid.npz'
    evidence.write_exclusive(plan_file,initial);evidence.write_exclusive(geometry_file,geometry)
    np.savez_compressed(grid_file,occupied=omitted,origin_xy_m=origin,resolution_m=res)
    # This SDK packet explicitly owns TWO Play transitions in ONE mission.
    # The frozen initial map is used for planning only; physical full-map
    # guards are independently bound below BEFORE any actuation/preflight.
    binding.update(plan_filename=plan_file.name,geometry_file=str(geometry_file),grid_file=str(grid_file),
        maximum_Play_calls=2,stopped_replanning=copy.deepcopy(H5_REPLAN_SCOPE),
        physical_complete_grid_file=packet['grid_file'],physical_complete_geometry_file=packet['geometry_file'])
    files=[plan_file,geometry_file,grid_file,initial_plan_path,initial_audit_path,inv_path,Path(__file__).resolve(),
        ROOT/'studies/IJCAS/scripts/hospital_mission_planning.py',ROOT/'studies/IJCAS/scripts/plan_hospital_h3_native.py',
        ROOT/'studies/IJCAS/scripts/audit_hospital_paper_readiness.py',*lc.host_source_files()]
    if unstarted_preflight_review is not None:files.append(review_path)
    binding['input_sha256'].update({str(p.resolve()):sha(p) for p in files})
    path=output/'replanning_sdk_binding.json';evidence.write_exclusive(path,binding)
    checked_h5_replanning_scope(binding);lc.freeze_workflow_sources(output/'replanning_sources.zip',binding['input_sha256'])
    sdk=rt.runtime.live_sdk()
    port=lc.RetainedMissionSDKHost(sdk,binding,path,context=context,plan=initial,geometry=geometry,
        occupied=omitted,origin=origin,resolution=res,authorized=True)
    # Runtime map is always COMPLETE, including the as-yet unobserved cart.
    port.runtime_guard=lc.MissionRuntimeGuard(full_geometry,full,origin,res,simulation_budget_s=240.)
    controller=seq.unified_geometry_lifecycle_controller(initial,geometry,profile=packet['geometry_profile'],simulation_budget_s=240.)
    class Switch:
        def __init__(self):self.inner=controller;self.stop=None;self.contract=controller.contract;self.geometry_profile=controller.geometry_profile
        @property
        def stage(self):return 'discovery_stop' if self.stop is not None else self.inner.stage
        @property
        def curvature(self):return self.stop.curvature if self.stop else self.inner.curvature
        @property
        def mode(self):return self.stop.mode if self.stop else self.inner.mode
        def step(self,t,pose,speed,steer,*,drive_hold_confirmed):
            if self.stop is None:return self.inner.step(t,pose,speed,steer,drive_hold_confirmed=drive_hold_confirmed)
            result=self.stop.step(t,speed,drive_hold_confirmed)
            error=seq.lifecycle.relative_pose(self.inner.goal,pose)
            result.update(goal_position_error_m=math.hypot(*error[:2]),goal_heading_error_deg=math.degrees(error[2]),
                progress_m=self.inner.reference.progress)
            return result
    switch=Switch();rows=[];query_rows=[];observed=None;bridge=None;process=None;preflight=actuation=False
    report=dict(verdict='HOLD',checks={},new_route_trials=0,additional_Stop_calls=0,automatic_retry=False,
        use_for_final_statistics=False,contract=copy.deepcopy(H5_REPLAN_SCOPE),new_replanning_requests=0,actual_Play_calls=0,
        initial_cart_information_map_review=map_review,initial_plan=str(initial_plan_path),initial_audit=str(initial_audit_path),
        frozen_initial_plan_NOT_a_full_map_bypass=False,actual_physical_guard_uses_complete_map_from_start=True)
    if not port.claim_once():return dict(status='PRIOR_H5_REPLAN_DISPATCH_RETAINED_NO_NEW_ACTION')
    last_query=None;began=None;phase='initial_approach';fixed_hold_attempts=0
    try:
        report['preflight']=lc.normalize_runtime_evidence(await port.preflight());preflight=True;port.subscribe()
        def step(row):
            return bridge.step(row['time_s'],(row['x_m'],row['y_m'],row['yaw_rad']),
                max(math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s']),row['pose_filtered_speed_m_s']))
        bridge=seq.RetainFinalTargetsBridge(switch,port.actuator,guard=lambda pose,speed,info:port.guard(observed,info))
        observed=lc.checked_measurement(port.measure());port.guard(observed,None);actuation=True;step(observed)
        began=watch=time.monotonic();report['new_route_trials']=1;port.play();report['actual_Play_calls']=1
        while not switch.inner.complete:
            remaining=900.-(time.monotonic()-began)
            if remaining<=0.:raise TimeoutError('Original900s WHOLE H5 wall cap includes planning')
            await asyncio.wait_for(port.next_sample(),timeout=remaining)
            observed=lc.checked_measurement(port.measure());pose=[observed[k] for k in ('x_m','y_m','yaw_rad')]
            port.guard(observed,None)
            if phase=='initial_approach' and (last_query is None or port.clock-last_query['time_s']>=.1):
                last_query=h5_query_selected_cart(port,pose,map_review['cart']['world_aligned_bound']);query_rows.append(last_query)
                if last_query['full_body_to_selected_cart_AABB_m']<=.75:
                    if not last_query['selected_cart_hit']:raise RuntimeError('Declared discovery standoff reached without authenticated cart query')
                    if bridge.last_info is None or bridge.last_info['phase']!='path_tracking':raise RuntimeError('Expected ordinary initial path tracking at discovery')
                    switch.stop=H5DiscoveryStop(bridge.last_info,bridge.last_stamp);phase='discovery_stop'
                    report['discovery']=copy.deepcopy(last_query);print('[H5 REPLAN] Cart discovered; measured braking/hold ONLY.',flush=True)
            info=step(observed)
            row=dict(observed,phase=info['phase'],phase_time_s=info.get('phase_time_s',0.),command_speed_m_s=info['speed_m_s'],
                path_progress_m=info.get('progress_m',0.),replanning_phase=phase,
                cart_query_hit=bool(last_query and last_query['selected_cart_hit']),
                cart_query_stamp_s=-1. if last_query is None else last_query['time_s'])
            row.update(lc.normalize_runtime_evidence(port.command_readback()));rows.append(row)
            if phase=='discovery_stop' and switch.stop.ready:
                report['intermediate_Pause']=await port.confirm_pause()
                if port.timeline.is_playing() or port.timeline.is_stopped():raise RuntimeError('Actual retained PAUSED planning boundary required')
                observed=lc.checked_measurement(port.measure());port.guard(observed,None)
                tensor=rt.shared.preparation.old.tensor_snapshot(port.tensor);motor=port.adapter._motor()
                def cpu(value):
                    if hasattr(value,'detach'):return value.detach().cpu().numpy()
                    return value.numpy() if hasattr(value,'numpy') else np.asarray(value)
                coms=cpu(port.tensor.get_coms());root_index=tensor['link_names'].index('base_link')
                com_local=coms[0,root_index,:3].tolist();twist=base_origin_body_twist(tensor,com_local)
                raw_speed=math.hypot(*tensor['velocity'][:2]);body_speed=math.hypot(*twist[:2])
                if max(raw_speed,body_speed,observed['pose_filtered_speed_m_s'])>.005 or body_speed>.003:
                    raise RuntimeError('Actual PAUSED capture not within original quiet-stop/3mm_s capture limits')
                if port.adapter.confirmation(port.clock) is not True:raise RuntimeError('Actual captured-wheel hold ACK required')
                capture=dict(time_s=port.clock,pose_world_xyyaw=[observed[k] for k in ('x_m','y_m','yaw_rad')],
                    body_twist_u_v_r=twist,front_rear_steering_rad=[observed['front_rad'],observed['rear_rad']],
                    front_rear_drive_velocity_rad_s=[observed['drive_front_velocity_rad_s'],observed['drive_rear_velocity_rad_s']],
                    retained_mode=switch.mode,retained_direction=port.runtime_guard.last_nonzero_sign,
                    retained_curvature_1_m=switch.curvature,planner_started=True,actual_hold_ACK=True,
                    captured_steering_targets_rad=[float(motor['positions'][i]) for i in port.adapter.steering],
                    SDK_root_local_CoM_m=com_local,raw_CoM_horizontal_speed_m_s=raw_speed,
                    zero_drive_targets_confirmed=all(abs(float(motor['velocities'][i]))<1e-10 for i in port.adapter.drive))
                if not capture['zero_drive_targets_confirmed']:raise RuntimeError('Actual drive targets must be zero throughout planning')
                prefix=output/'executed_prefix.csv';h5_write_trace(prefix,rows)
                carried=h5_prefix_carried_state(initial,initial_audit,rows,capture,port.runtime_guard.summary())
                request=dict(version=H5_REPLAN_SCOPE['version'],capture=capture,carried_state=carried,
                    profile=packet['geometry_profile'],geometry_file=packet['geometry_file'],prefix_trace=str(prefix),
                    prefix_trace_sha256=sha(prefix),input_sha256=dict(binding['input_sha256']))
                request['input_sha256'][str(prefix)]=sha(prefix)
                request_path=output/'measured_replan_request.json';evidence.write_exclusive(request_path,request)
                report.update(new_replanning_requests=1,measured_capture=capture,measured_request=str(request_path),
                    measured_request_sha256=sha(request_path),executed_prefix=str(prefix),executed_prefix_sha256=sha(prefix))
                print('[H5 REPLAN] Actual stop/Pause captured; ONE measured-state plan. Physics clock retained.',flush=True)
                log=(output/'planner_worker.log').open('x',encoding='utf-8')
                command=list(worker_prefix)+['-c','from hospital_sensor_navigation import h5_worker; import sys; h5_worker(sys.argv[1])',str(request_path)]
                process=await asyncio.create_subprocess_exec(*command,cwd=str(ROOT),stdout=log,stderr=asyncio.subprocess.STDOUT,stdin=asyncio.subprocess.DEVNULL)
                started=time.monotonic();paused_clock=port.clock;paused_state=port.timeline_state();monitor_checks=0
                while process.returncode is None:
                    await port.app.next_update_async();port.same_context()
                    if (port.timeline_state()!=paused_state or port.clock!=paused_clock or port.errors
                            or rt.shared.preparation.old.tensor_snapshot(port.tensor)!=tensor):
                        raise RuntimeError('Actual captured full state/clock/monitor changed during PAUSED planning')
                    current_motor=port.adapter._motor()
                    if any(not np.array_equal(current_motor[k],v) for k,v in motor.items()):raise RuntimeError('Captured motor/gain/targets changed during planning')
                    monitor_checks+=1
                    if time.monotonic()-started>180. or time.monotonic()-began>900.:
                        raise TimeoutError('Original planning180s/whole900s deadline; no retry')
                    await asyncio.sleep(.02)
                log.close()
                if process.returncode!=0:raise RuntimeError('Measured replan worker failed; see planner_worker.log; no resume')
                done=json.loads((output/'worker_complete.json').read_bytes())
                if done['request_sha256']!=report['measured_request_sha256'] or sha(request_path)!=report['measured_request_sha256']:
                    raise ValueError('Foreign/stale/modified measured replan request/result')
                for file,digest in done['output_sha256'].items():
                    if sha(file)!=digest:raise ValueError('Returned replan changed')
                rt.runtime.task.verify_pins(binding['input_sha256'])
                plan=json.loads((output/'resumed_execution_plan.json').read_bytes());g=json.loads((output/'resumed_execution_geometry.json').read_bytes())
                if (plan['initial_complete_state']!=carried
                        or plan['measured_replan_start']['request_sha256']!=report['measured_request_sha256']
                        or plan['measured_replan_start']['captured_world_xyyaw']!=capture['pose_world_xyyaw']):
                    raise ValueError('Returned plan does not continue the exact own captured prefix')
                if rt.shared.preparation.old.tensor_snapshot(port.tensor)!=tensor or port.timeline_state()!=paused_state or port.errors:
                    raise RuntimeError('Fresh exact captured-state revalidation failed; no resume')
                observed=lc.checked_measurement(port.measure());port.guard(observed,None)
                resumed=seq.unified_geometry_lifecycle_controller(plan,g,profile=packet['geometry_profile'],simulation_budget_s=240.,start_time_s=0.)
                resumed.lifecycle_stage='route';resumed.previous=port.clock
                switch.inner=resumed;switch.stop=None;phase='resumed_suffix'
                if time.monotonic()-started>180. or time.monotonic()-began>900.:
                    raise TimeoutError('Original replan/revalidation180s and whole900s clocks required before resume')
                # Same adapter/targets, same resource guard, same clocks, ONE
                # authorized second Play; do not call the fresh-start play().
                port.same_context();port.timeline.play();report['actual_Play_calls']=2
                report['replan_completion']=dict(done,paused_watchdog_checks=monitor_checks,
                    complete_actual_tensor_exactly_retained_through_planning=True,
                    original_mission_clock_and_resources_retained_on_resume=True)
                print('[H5 REPLAN] Replan/audit/models PASS; SAME robot resumes to SAME goal.',flush=True)
            if time.monotonic()-watch>=15.:
                print('[H5 REPLAN] wall=%.1fs sim=%.2fs phase=%s goal=%.2fmm'%(
                    time.monotonic()-began,port.clock,phase,1000*info['goal_position_error_m']),flush=True);watch=time.monotonic()
        fixed_hold_attempts=switch.inner.fixed_hold_attempts
        if phase!='resumed_suffix':raise RuntimeError('Goal completion without the required actual stop/replan/resume event')
        report['terminal']=lc.evaluate_terminal_hold(rows,switch.inner.goal,switch.inner.reference.arc[-1],port.counts,port.errors,fixed_hold_attempts=fixed_hold_attempts)
        report['checks'].update(report['terminal']['checks'])
        report['checks'].update(one_actual_discovery_and_replan=report['new_replanning_requests']==1,
            one_initial_Play_and_one_resume=report['actual_Play_calls']==2,
            real_intermediate_PAUSED_state_and_exact_full_state_retained=True,
            no_Stop_reset_or_replacement_at_replanning=True,physical_full_map_guard_never_omitted_cart=True)
    except (Exception,asyncio.CancelledError) as failure:
        report['error']=type(failure).__name__+': '+str(failure);report['checks']['whole_stop_replan_resume_and_final_hold_completed']=False
    finally:
        fixed_hold_attempts=switch.inner.fixed_hold_attempts
        if process is not None and process.returncode is None:
            process.terminate()
            try:await asyncio.wait_for(process.wait(),timeout=3.)
            except asyncio.TimeoutError:process.kill();await process.wait()
        report['cart_queries']=query_rows
        report['whole_mission_wall_s']=None if began is None else time.monotonic()-began
        report=await lc._finish_observation(port,report,rows,preflight=preflight,actuation=actuation,capture_count=fixed_hold_attempts)
    return json.loads(port.result.read_bytes())


def legacy_replanning_source_review(source=None):
    """Reproduce this ONE adapter's cache/timer behavior on fake ROS data.

    Compile only original AST function bodies; no ROS/SDK imports or Node.
    Planning is replaced by an explicit count-only fixture returning None.
    This does not mean that an H5 method failed to find a path, nor that an
    entire runtime lacks safety checks. Never adopt periodic replanning or
    the old center-cell collision checker as the hospital execution policy.
    """
    from types import SimpleNamespace
    source=Path(source or ROOT/'src/patient_initial_stability/patient_initial_stability/occupancy_lattice_planner.py').resolve()
    text=source.read_text();before=sha(source)
    if before!='5aa3906a191333055edc0b1ba5b728c4caaa4168b02bb511df8514735dda3668':
        raise ValueError('Legacy adapter source behavior changed; review exact source before fixture execution')
    tree=ast.parse(text)
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OccupancyLatticePlanner']
    if len(classes)!=1:raise ValueError('Exact original replanning adapter class required')
    cls=classes[0]
    names=('_on_map','_on_goal','_try_plan');methods={}
    for name in names:
        found=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(found)!=1:raise ValueError('Exact original replanning adapter method required: '+name)
        methods[name]=found[0]
    calls=[];clock=[10.];logs=[]
    def plan_fixture(*args,**kwargs):
        calls.append(dict(started_wall_s=clock[0],goal_local=list(args[0]),
            collision_source='SOFTWARE_PLACEHOLDER_NOT_ACTUAL_COLLISION_VALIDATION'))
        return None
    namespace=dict(math=math,time=SimpleNamespace(monotonic=lambda:clock[0],perf_counter=lambda:clock[0]),
        plan=plan_fixture,GridCollisionChecker=lambda *args:object(),
        yaw_from_odometry=lambda odom:odom.fixture_yaw,yaw_from_pose=lambda pose:pose.fixture_yaw,wrap=wrap)
    for method in methods.values():
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    def header():return SimpleNamespace(frame_id='SOFTWARE_FIXED_MAP',stamp=SimpleNamespace(sec=1,nanosec=0))
    grid=SimpleNamespace(header=header(),data=[0,0,0,0])
    pose=SimpleNamespace(position=SimpleNamespace(x=1.,y=0.),fixture_yaw=0.)
    odom=SimpleNamespace(header=header(),pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=0.,y=0.))),
        fixture_yaw=0.,twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=.05,y=0.),angular=SimpleNamespace(z=0.))))
    logger=SimpleNamespace(error=lambda *a,**kw:logs.append(a[0]))
    node=SimpleNamespace(grid=grid,goal=SimpleNamespace(header=header(),pose=pose),odom=odom,
        require_steering_state=False,steering_deg=None,replan_period=0.,last_request=None,last_plan_time=None,
        clearance=.010,occupied_threshold=50,speed=.05,initial_mode='coordinated_counterphase',get_logger=lambda:logger)
    namespace['_try_plan'](node);first=len(calls)
    changed=SimpleNamespace(header=header(),data=[0,100,0,0])
    namespace['_on_map'](node,changed);namespace['_try_plan'](node);second=len(calls)
    node.replan_period=1.;clock[0]=12.;namespace['_try_plan'](node);third=len(calls)
    # A genuinely new goal callback DOES clear the cache; do not mistake the
    # request tuple's omitted yaw for a normal goal-callback failure.
    replacement=SimpleNamespace(header=header(),pose=SimpleNamespace(position=pose.position,fixture_yaw=.2))
    namespace['_on_goal'](node,replacement);namespace['_try_plan'](node);fourth=len(calls)
    if (first,second,third,fourth)!=(1,1,2,3):
        raise ValueError('Legacy adapter behavior changed; review findings instead of reusing stale diagnosis')
    request=methods['_try_plan']
    attributes={n.attr for n in ast.walk(request) if isinstance(n,ast.Attribute)}
    if sha(source)!=before:raise ValueError('Original planner source changed during GET-only source review')
    return dict(version='LEGACY_ROS_MAP_CACHE_TIMER_SOFTWARE_REPRODUCTION_v1',
        status='SOURCE_ONLY_GAPS_NOT_H5_REPLANNING_OR_PHYSICAL_FAILURE',source=str(source),source_sha256=before,
        source_methods=[dict(name=n,start_line=m.lineno,end_line=m.end_lineno,
            AST_source_sha256=hashlib.sha256(ast.get_source_segment(text,m).encode()).hexdigest()) for n,m in methods.items()],
        fake_planner_call_counts=dict(initial=first,after_changed_map_same_goal_default_period=second,
            after_periodic_timer_at_fixture_speed50mm_s=third,after_new_goal_callback=fourth),
        SOFTWARE_fixture_planner_calls=calls,SOFTWARE_fixture_odometry_speed_m_s=.05,
        map_callback_stores_content_but_does_not_invalidate_same_goal_default_cache=True,
        periodic_timer_by_itself_is_NOT_measured_standstill_gate='twist' not in attributes,
        new_goal_callback_clears_cache_including_heading_change=True,
        original_collision_checker_was_NOT_used_or_validated=True,
        exact_reviewed_source_verified_BEFORE_fake_method_execution=True,
        fake_plan_None_is_NOT_any_method_no_path_or_failure_evidence=True,
        no_entire_runtime_safety_or_physical_cause_conclusion=True,
        required_separate_hospital_work=['freeze same observed-change policy and task/map/compute revisions for all methods',
            'request zero command and confirm measured standstill without simulation Stop/reset',
            'freeze measured steering/body entry and reject stale map/state planning results',
            'use full-cell/full-envelope support-aware planning, not legacy occupied cell centers',
            'charge observation-to-stop/replan/revalidation/resume and preserve all failures',
            'separately authorize actual map/environment event, bridge, controller and motion'],
        no_original_adapter_or_H5_result_modified=True,no_ROS_context_or_SDK_or_node=True,
        new_trial_count=0,actual_motion_authorized=False,H5_online_replanning_success=False)
