"""Open-loop MODEL diagnostic of a frozen signed H3 reference, NOT execution.

Uses the inherited Newton-Euler response/wrench model without retuning. Body
velocity, yaw rate and steering states are NEVER reset at a cusp. Additional
zero-command model settling is allowed before changing direction; it is NOT a
measured captured-wheel hold/actuator acknowledgement or a physical gate PASS.
No caster swivel/stiction states or path/terminal feedback are invented here.
"""

import math

import numpy as np

from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
from patient_initial_stability.steering_lattice import (MODE_CURVATURE_RESPONSE_SCALE,
    MODE_LATERAL_RESPONSE_SCALE, STEERING_TRACKING_BIAS_RAD)
from hospital_path_execution import MODES, direct_joint_targets, wrap
from hospital_h3b_policy import POLICY


VERSION='H3B_frozen_reference_directional_open_loop_model_v1'


class Reference:
    def __init__(self,geometry,timing):
        self.s=np.asarray(geometry['travelled_longitudinal_parameter_m'],float)
        self.pose=np.asarray(geometry['poses_local_xyyaw'],float)
        self.k=np.asarray(geometry['sample_curvature_1_m'],float)
        self.v=np.asarray(timing['sample_unsigned_reference_speed_m_s'],float)
        self.dt=np.asarray(timing['interval_motion_duration_s'],float)
        self.direction=np.asarray(geometry['interval_directions'])
        self.mode=tuple(geometry['interval_modes'])
        n=len(self.s)
        if (n<3 or self.pose.shape!=(n,3) or self.k.shape!=(n,) or self.v.shape!=(n,)
                or self.dt.shape!=(n-1,) or self.direction.shape!=(n-1,) or len(self.mode)!=n-1
                or not all(np.isfinite(a).all() for a in (self.s,self.pose,self.k,self.v,self.dt))
                or abs(self.s[0])>1e-12 or np.any(np.diff(self.s)<=0) or np.any(self.dt<=0)
                or np.any(self.v<0) or max(abs(self.k))>.7+1e-10
                or not np.isin(self.direction,(-1,1)).all() or any(m not in MODES for m in self.mode)):
            raise ValueError('Complete finite bounded signed reference required')
        dx=np.diff(self.s)
        if not np.allclose((self.v[:-1]+self.v[1:])*self.dt/2,dx,rtol=1e-10,atol=1e-12):
            raise ValueError('Nominal time/speed/distance mismatch')
        cap=np.where(self.direction>0,POLICY['forward_speed_limit_m_s'],POLICY['reverse_speed_limit_m_s'])
        if (np.any(np.maximum(self.v[:-1],self.v[1:])>cap+1e-10)
                or np.any(abs(np.diff(self.v)/self.dt)>POLICY['acceleration_m_s2']+1e-10)):
            raise ValueError('Reference exceeds existing signed speed or acceleration bounds')
        self.cusps=np.flatnonzero(np.diff(self.direction))+1
        self.stops=np.r_[0,self.cusps,n-1]
        if len(self.cusps)>4 or np.any(self.v[self.stops]>1e-12) or np.any(abs(self.k[self.stops])>1e-10):
            raise ValueError('Existing straight/start/cusp/end stop contract required')
        changes=[i for i in range(1,len(self.mode)) if self.mode[i]!=self.mode[i-1]]
        if any(i not in self.cusps for i in changes):
            raise ValueError('This reference permits mode changes only at straight direction cusps')

    def at(self,index,elapsed):
        duration=self.dt[index]
        if not math.isfinite(elapsed) or not -1e-12<=elapsed<=duration+1e-12:
            raise ValueError('Finite in-interval reference time required')
        elapsed=min(duration,max(0.,elapsed))
        acceleration=(self.v[index+1]-self.v[index])/duration
        travelled=self.v[index]*elapsed+.5*acceleration*elapsed**2
        fraction=float(np.clip(travelled/(self.s[index+1]-self.s[index]),0.,1.))
        pose=self.pose[index]+fraction*(self.pose[index+1]-self.pose[index])
        return (int(self.direction[index])*(self.v[index]+acceleration*elapsed),
            float(self.k[index]+fraction*(self.k[index+1]-self.k[index])),pose)


def playback(geometry,timing,bounds,goal,dt=.01):
    if not math.isfinite(dt) or not 0<dt<=.02 or len(goal)!=3 or not all(math.isfinite(v) for v in goal):
        raise ValueError('Fine integration and finite unchanged goal required')
    reference=Reference(geometry,timing)
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
    pose=reference.pose[0].copy();velocity=np.zeros(3);actual=np.zeros(2);commanded=(0.,0.)
    now=0.;rows=[];events=[];feasible=True
    max_friction=max_ltr=max_actuator=max_command_rate=0.
    reverse_distance=reverse_episode=maximum_episode=0.
    phase='initial';leg=0

    def row(speed,k,mode,ref,phase):
        return dict(time_s=now,phase=phase,leg_index=leg,command_mode=mode,
            command_speed_m_s=float(speed),requested_curvature_1_m=float(k),
            x_m=float(pose[0]),y_m=float(pose[1]),yaw_rad=float(pose[2]),
            u_m_s=float(velocity[0]),v_m_s=float(velocity[1]),r_rad_s=float(velocity[2]),
            front_rad=float(actual[0]),rear_rad=float(actual[1]),
            command_front_rad=float(commanded[0]),command_rear_rad=float(commanded[1]),
            reference_x_m=float(ref[0]),reference_y_m=float(ref[1]),reference_yaw_rad=float(ref[2]),
            reference_position_error_m=float(np.linalg.norm(pose[:2]-ref[:2])),
            reference_heading_error_deg=math.degrees(wrap(float(pose[2]-ref[2]))))

    rows.append(row(0.,0.,reference.mode[0],reference.pose[0],phase))

    def tick(h,speed,k,mode,ref,phase):
        nonlocal now,commanded,actual,velocity,pose,feasible,max_friction,max_ltr,max_actuator,max_command_rate
        nonlocal reverse_distance,reverse_episode,maximum_episode
        old_command=commanded
        commanded,_=direct_joint_targets(float(speed),mode,float(k),commanded,h,actual)
        max_command_rate=max(max_command_rate,*(abs(a-b)/h for a,b in zip(commanded,old_command)))
        target=np.array(commanded)+[STEERING_TRACKING_BIAS_RAD[a] for a in ('front','rear')]
        mid=actual+np.clip(target-actual,-math.radians(8.)*h/2,math.radians(8.)*h/2)
        actual=mid+np.clip(target-mid,-math.radians(8.)*h/2,math.radians(8.)*h/2)
        tf,tr=np.tan(mid)
        curvature=(tf-tr)/1.29*MODE_CURVATURE_RESPONSE_SCALE[mode]
        ratio=(.645*tf+.645*tr)/1.29*MODE_LATERAL_RESPONSE_SCALE[mode]
        result=model.step(*velocity,float(speed),float(speed)*ratio,float(speed)*curvature,h)
        new_velocity=np.array([result.longitudinal_velocity_m_s,result.lateral_velocity_m_s,result.yaw_rate_rad_s])
        u,v,r=(velocity+new_velocity)/2
        mid_yaw=pose[2]+r*h/2
        increment=h*np.array([u*math.cos(mid_yaw)-v*math.sin(mid_yaw),u*math.sin(mid_yaw)+v*math.cos(mid_yaw),r])
        pose+=increment;velocity=new_velocity;now+=h
        distance=float(np.linalg.norm(increment[:2]))
        if u< -1e-8:
            reverse_distance+=distance;reverse_episode+=distance;maximum_episode=max(maximum_episode,reverse_episode)
        elif u>1e-8:reverse_episode=0.
        feasible &= bool(result.feasible)
        max_friction=max(max_friction,result.friction_utilization)
        max_ltr=max(max_ltr,result.abs_ltr)
        max_actuator=max(max_actuator,result.actuator_utilization)
        rows.append(row(speed,k,mode,ref,phase))

    def stop_and_settle(mode,ref,number):
        start=now;stable_since=None;first_stopped=None
        before=pose.copy();incoming=velocity.copy()
        while True:
            speed=float(np.linalg.norm(velocity[:2]))
            neutral=max(abs(actual))<=math.radians(POLICY['alignment_tolerance_deg'])
            stopped=speed<=POLICY['stopped_speed_m_s']
            if stopped and first_stopped is None:first_stopped=now
            if not stopped and now-start>POLICY['braking_timeout_s']:
                raise TimeoutError('Inherited MODEL braking timeout; no state reset permitted')
            if stopped and neutral and max(abs(v) for v in commanded)<1e-10:
                if stable_since is None:stable_since=now
                if now-stable_since>=POLICY['stationary_settle_s']+POLICY['alignment_dwell_s']-1e-12:break
            else:stable_since=None
            if now-start>POLICY['braking_timeout_s']+POLICY['alignment_timeout_s']:
                raise TimeoutError('Inherited MODEL settle/alignment timeout')
            tick(dt,0.,0.,mode,ref,f'stop_{number}_model_zero_command_settle')
        events.append({'stop_number':number,'reference_pose':ref.tolist(),'model_start_time_s':start,'model_ready_time_s':now,
            'model_first_stopped_time_s':first_stopped,'model_pose_before_settle':before.tolist(),
            'model_pose_after_settle':pose.tolist(),'model_incoming_velocity':incoming.tolist(),
            'model_outgoing_velocity':velocity.tolist(),'model_horizontal_speed_m_s':float(np.linalg.norm(velocity[:2])),
            'model_active_steering_error_deg':math.degrees(max(abs(actual))),
            'body_velocity_or_steering_reset':False,'physical_hold_release_acknowledgement_verified':False})

    for leg,(left,right) in enumerate(zip(reference.stops[:-1],reference.stops[1:])):
        mode=reference.mode[left]
        for index in range(left,right):
            duration=reference.dt[index]
            # Equal subdivisions avoid tiny remainder steps affecting slew.
            count=math.ceil(duration/dt);h=duration/count
            for j in range(count):
                speed,k,_=reference.at(index,(j+.5)*h)
                _,_,ref=reference.at(index,(j+1)*h)
                tick(h,speed,k,mode,ref,f'leg_{leg}_reference_motion')
        stop_and_settle(mode,reference.pose[right],leg+1)
    tail_start=now
    for _ in range(math.ceil(5./dt)):
        tick(dt,0.,0.,reference.mode[-1],reference.pose[-1],'model_zero_command_tail_NOT_PhysX_hold')
    error=float(np.linalg.norm(pose[:2]-np.array(goal[:2])))
    heading=math.degrees(wrap(float(pose[2]-goal[2])))
    checks={'inherited_force_load_limits':bool(feasible),'reference_and_model_stop_count_match':len(events)==len(reference.stops)-1,
        'commanded_steering_slew_under_8deg_s':max_command_rate<=math.radians(8.)+1e-9,
        'model_stopped_before_each_direction_change':all(e['model_horizontal_speed_m_s']<=.005 for e in events),
        'model_neutral_before_each_direction_change':all(e['model_active_steering_error_deg']<=2.5 for e in events),
        'model_reverse_episode_under_0p5m':maximum_episode<=.5+1e-9,
        'model_total_reverse_under_1m':reverse_distance<=1.+1e-9,
        'model_terminal_position_under_30mm':error<=.030,'model_terminal_heading_under_2deg':abs(heading)<=2.}
    return {'version':VERSION,'diagnostic_role':'FROZEN_REFERENCE_OPEN_LOOP_MODEL_NOT_CLOSED_LOOP_EXECUTION_GATE',
        'dt_s':dt,'model_checks':{k:bool(v) for k,v in checks.items()},'model_stop_events':events,
        'final_model_local_xyyaw':pose.tolist(),'model_terminal_position_error_m':error,
        'model_terminal_heading_error_deg':heading,'model_duration_s':now,'model_zero_command_tail_s':now-tail_start,
        'maximum_friction_utilization':max_friction,'maximum_abs_ltr':max_ltr,'maximum_actuator_utilization':max_actuator,
        'maximum_command_slew_deg_s':math.degrees(max_command_rate),'model_reverse_distance_m':reverse_distance,
        'model_maximum_reverse_episode_m':maximum_episode,
        'maximum_reference_position_error_m':max(r['reference_position_error_m'] for r in rows),
        'no_body_velocity_or_steering_resets':True,'path_or_terminal_feedback_used':False,
        'model_has_no_explicit_caster_swivel_or_stiction_state':True,
        'physical_hold_release_acknowledgement_verified':False,'PhysX_executed':False,
        'full_dynamics_plan_certified':False,'full_H3B_turning_path_verified':False,
        'model_violation_is_NOT_measured_PhysX_failure_or_global_infeasibility':True},rows
