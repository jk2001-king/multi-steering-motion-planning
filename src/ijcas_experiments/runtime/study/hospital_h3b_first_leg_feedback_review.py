"""First-leg progress feedback MODEL review, NOT a new planner/physical PASS.

Reuse H1/H2 PathReference projection and EXACT tracking equation/gains. No
terminal pulse, reverse, stationary nonzero steering, caster assignment or
model retuning. Neutral braking and fixed first-stop target remain separate
from tracking. Synthetic steering-bias cases are NOT an identified uncertainty
set or substitutes for physical caster/contact dynamics.
"""

import math

import numpy as np

from hospital_h3b_first_leg_model_pilot import load_profile,component_goal,relative_pose,COMMAND_LIMITS,TRACKING
from hospital_path_execution import PathReference,direct_joint_targets,ramp,wrap
from hospital_h3_turn_planning import model_bounds


VERSION='H3B_first_forward_leg_existing_H1_tracking_model_review_v1'
ROLE='MODEL_ONLY_FIRST_FORWARD_LEG_PROGRESS_FEEDBACK_NOT_FULL_H3_OR_PhysX_PASS'
TRACKING_GAINS={'yaw_error_gain_1_m_rad':1.5,'cross_error_gain_1_m2':2.}
BIAS_CASES={
    'inherited_minus1_deg':(-1.,-1.),
    'zero_bias_synthetic':(0.,0.),
    'common_plus1p5_deg_synthetic':(1.5,1.5),
    'front_plus_rear_minus_1p5_deg_synthetic':(1.5,-1.5),
    'front_minus_rear_plus_1p5_deg_synthetic':(-1.5,1.5)}
INITIAL_CASES={
    'model_origin':((0.,0.,0.),(0.,0.,0.)),
    'positive_lateral_prehold_envelope':((.0095,.003,math.radians(.5)),(.0045,0.,0.)),
    'negative_lateral_prehold_envelope':((.0095,-.003,-math.radians(.5)),(.0045,0.,0.))}


class FirstLegReference(PathReference):
    """Same project()/at() implementation, no false H1 plan-success schema."""
    def __init__(self,info):
        rows=info['model_prefix_rows']
        points=np.array([[r[k] for k in ('x_m','y_m','yaw_rad')] for r in rows])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))]
        if (not np.isfinite(points).all() or arc[-1]<1. or any(r['leg_index']!=0 for r in rows)
                or any(r['command_mode']!='coordinated_counterphase' for r in rows)):
            raise ValueError('Need bounded original first forward leg, not a different route')
        source=np.r_[True,np.diff(arc)>1e-10];arc,points=arc[source],points[source]
        self.arc=np.linspace(0.,arc[-1],math.ceil(arc[-1]/.025)+1).tolist()
        self.poses=[tuple(float(np.interp(s,arc,points[:,j])) for j in range(3)) for s in self.arc]
        mid=(np.array(self.arc[:-1])+self.arc[1:])/2
        self.actions=[{'mode':'coordinated_counterphase','curvature_1_m':float(k),
            'target_speed_m_s':float(v),'direction':1} for k,v in zip(
                np.interp(mid,arc,np.array([r['requested_curvature_1_m'] for r in rows])[source]),
                np.interp(mid,arc,np.array([r['command_speed_m_s'] for r in rows])[source]))]
        self.progress=0.


class FirstLegFeedback:
    """H1 tracking only, followed by measured neutral stop. NEVER reverses."""
    def __init__(self,info,started):
        if not math.isfinite(started):raise ValueError('Finite clock required')
        self.reference=FirstLegReference(info);self.goal=tuple(info['first_stop_model_local_xyyaw'])
        self.speed=0.;self.phase='tracking';self.previous_time=started;self.braking_started=None
        self.curvature=0.;self.last_result=None

    def step(self,timestamp,pose,measured_speed,measured_steer):
        if (len(pose)!=3 or len(measured_steer)!=2 or measured_speed<0
                or not all(math.isfinite(v) for v in (timestamp,*pose,measured_speed,*measured_steer))):
            raise ValueError('Finite actual pose/steering/speed required')
        dt=timestamp-self.previous_time
        if not 0<=dt<=.25:raise RuntimeError('Existing 0.25 s control-gap guard')
        self.previous_time=timestamp
        index,reference,path_error=self.reference.project(pose)
        yaw_error=wrap(reference[2]-pose[2]);remaining=self.reference.arc[-1]-self.reference.progress
        goal_error=math.hypot(pose[0]-self.goal[0],pose[1]-self.goal[1])
        if path_error>.080 or abs(yaw_error)>math.radians(10.):
            raise RuntimeError('Existing H1 reference tracking 80 mm / 10 deg guard')
        if self.phase=='tracking' and remaining<=.025 and goal_error<=.05:
            self.phase='braking';self.braking_started=timestamp
        if self.phase=='tracking':
            action=self.reference.actions[index]
            _,cross_error,_=relative_pose(reference,pose)
            # Exact inherited equation; no new feedback gain fitting.
            self.curvature=max(-.70,min(.70,action['curvature_1_m']+1.5*yaw_error+2.*cross_error))
            target=min(action['target_speed_m_s'],math.sqrt(2*.02*max(.000225,remaining-.010)))
        else:
            target=0.;self.curvature=0.
            if timestamp-self.braking_started>8.:raise TimeoutError('Existing measured braking/neutral timeout')
        self.speed=ramp(self.speed,target,dt,.02)
        ready=(self.phase=='braking' and abs(self.speed)<1e-7 and measured_speed<=.005
            and max(abs(v) for v in measured_steer)<=math.radians(2.5))
        self.last_result={'speed_m_s':self.speed,'curvature_1_m':self.curvature,'ready':ready,
            'phase':self.phase,'path_error_m':path_error,'path_heading_error_deg':math.degrees(yaw_error),
            'reference_local_xyyaw':list(reference),'progress_m':self.reference.progress,'remaining_m':remaining,
            'goal_error_m':goal_error,'goal_heading_error_deg':math.degrees(wrap(pose[2]-self.goal[2]))}
        return dict(self.last_result)


def model_probe(info,dt=.01,bias_case='inherited_minus1_deg',initial_case='model_origin'):
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import MODE_CURVATURE_RESPONSE_SCALE,MODE_LATERAL_RESPONSE_SCALE
    if not math.isfinite(dt) or not 0<dt<=.02 or bias_case not in BIAS_CASES or initial_case not in INITIAL_CASES:
        raise ValueError('Only predeclared fine model cases are supported')
    bounds=model_bounds();plant=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
    pose,velocity=[np.array(v,float) for v in INITIAL_CASES[initial_case]]
    bias=np.radians(BIAS_CASES[bias_case]);actual=np.zeros(2);commanded=(0.,0.)
    controller=FirstLegFeedback(info,0.);now=0.;rows=[];error=None;tail_start=None
    force_feasible=True;maximum_friction=maximum_ltr=maximum_actuator=maximum_slew=0.
    try:
        for _ in range(math.ceil(90./dt)):
            if tail_start is None:
                result=controller.step(now,pose,float(np.linalg.norm(velocity[:2])),actual)
                if result['ready']:tail_start=now
            if tail_start is not None:
                speed,curvature=0.,0.
            else:speed,curvature=result['speed_m_s'],result['curvature_1_m']
            previous=commanded
            commanded,_=direct_joint_targets(speed,'coordinated_counterphase',curvature,commanded,dt,actual)
            maximum_slew=max(maximum_slew,*(abs(a-b)/dt for a,b in zip(commanded,previous)))
            target=np.array(commanded)+bias
            mid=actual+np.clip(target-actual,-math.radians(8.)*dt/2,math.radians(8.)*dt/2)
            actual=mid+np.clip(target-mid,-math.radians(8.)*dt/2,math.radians(8.)*dt/2)
            tf,tr=np.tan(mid);mode='coordinated_counterphase'
            k=(tf-tr)/1.29*MODE_CURVATURE_RESPONSE_SCALE[mode]
            ratio=(tf+tr)/2*MODE_LATERAL_RESPONSE_SCALE[mode]
            state=plant.step(*velocity,speed,speed*ratio,speed*k,dt)
            updated=np.array([state.longitudinal_velocity_m_s,state.lateral_velocity_m_s,state.yaw_rate_rad_s])
            u,v,yaw_rate=(velocity+updated)/2;mid_yaw=pose[2]+yaw_rate*dt/2
            pose+=dt*np.array([u*math.cos(mid_yaw)-v*math.sin(mid_yaw),u*math.sin(mid_yaw)+v*math.cos(mid_yaw),yaw_rate])
            velocity=updated;now+=dt
            force_feasible &= bool(state.feasible)
            maximum_friction=max(maximum_friction,state.friction_utilization)
            maximum_ltr=max(maximum_ltr,state.abs_ltr);maximum_actuator=max(maximum_actuator,state.actuator_utilization)
            rows.append({'time_s':now,'phase':'model_zero_command_tail_NOT_PhysX_hold' if tail_start is not None else controller.phase,
                'leg_index':0,'command_mode':'coordinated_counterphase',
                'x_m':float(pose[0]),'y_m':float(pose[1]),'yaw_rad':float(pose[2]),
                'u_m_s':float(velocity[0]),'v_m_s':float(velocity[1]),'r_rad_s':float(velocity[2]),
                'command_speed_m_s':speed,'curvature_1_m':curvature,'front_rad':float(actual[0]),'rear_rad':float(actual[1]),
                'command_front_rad':commanded[0],'command_rear_rad':commanded[1],
                'reference_position_error_m':math.hypot(pose[0]-result['reference_local_xyyaw'][0],pose[1]-result['reference_local_xyyaw'][1]),
                'path_error_m':result['path_error_m'],'path_heading_error_deg':result['path_heading_error_deg']})
            if tail_start is not None and now-tail_start>=5.-1e-9:break
        else:raise TimeoutError('Existing 90 simulated-second model guard')
    except (RuntimeError,ValueError,TimeoutError) as exception:error=f'{type(exception).__name__}: {exception}'
    position=float(np.linalg.norm(pose[:2]-np.array(info['first_stop_model_local_xyyaw'][:2])))
    heading=math.degrees(wrap(pose[2]-info['first_stop_model_local_xyyaw'][2]))
    checks={'model_sequence_completed':error is None and tail_start is not None,
        'model_terminal_under_30mm':position<=.030,'model_terminal_under_2deg':abs(heading)<=2.,
        'model_stopped_under_5mm_s':np.linalg.norm(velocity[:2])<=.005,
        'model_inherited_force_limits':force_feasible,'command_slew_under_8deg_s':maximum_slew<=math.radians(8.)+1e-9}
    return {'version':VERSION,'role':ROLE,'dt_s':dt,'bias_case':bias_case,'initial_case':initial_case,
        'synthetic_bias_deg':BIAS_CASES[bias_case],'initial_pose_and_velocity':INITIAL_CASES[initial_case],
        'model_parameters':bounds,'tracking_gains':TRACKING_GAINS,'model_checks':{k:bool(v) for k,v in checks.items()},
        'terminal_position_error_m':position,'terminal_heading_error_deg':heading,'final_model_local_xyyaw':pose.tolist(),
        'maximum_friction_utilization':maximum_friction,'maximum_abs_ltr':maximum_ltr,'maximum_actuator_utilization':maximum_actuator,
        'model_duration_s':now,'model_zero_command_tail_s':None if tail_start is None else now-tail_start,
        'error':error,'no_velocity_steering_or_pose_reset':True,'caster_swivel_or_stiction_modelled':False,
        'synthetic_cases_are_NOT_an_identified_uncertainty_set_or_robustness_proof':True,
        'physical_hold_release_acknowledgement_verified':False,'PhysX_executed':False,
        'full_H3B_turning_path_verified':False,'reverse_turning_verified':False,'use_for_final_statistics':False},rows
