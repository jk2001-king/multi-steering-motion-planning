"""New shared spatial-candidate families using EXISTING steering modes only.

Mode-dependent body-lateral motion is integrated, not a label substitution on
the old counterphase path. Components motivate two predeclared seed families;
they do NOT establish reverse-turning or large-curvature execution feasibility.
All later method comparisons need the same candidate representation/resources.
The old seed, measured outcomes and rigid-body model remain unchanged.
"""

import math
import time

import numpy as np
from scipy.optimize import least_squares

from hospital_h3b_warm_start import SeedModel, Objective, audit as old_geometry_audit
from hospital_h3b_policy import POLICY


FAMILIES = ('front_only_all_legs', 'counterphase_forward_front_only_reverse')
CONTRACT = {'version':'H3B_mode_aware_continuous_spatial_seed_v1',
    'role':'SHARED_SPATIAL_CANDIDATES_NOT_SELECTED_FULL_DYNAMICS_PLAN',
    'families':FAMILIES,'existing_modes_only':True,'mode_changes_at_straight_cusps_only':True,
    'body_lateral_motion_propagated':True,'same_search_revision_required_in_future_comparison':True,
    'measured_forward_component_is_not_a_reverse_or_large_curvature_certificate':True,
    'Newton_Euler_or_caster_state_propagated_here':False,'PhysX_executed':False}


class ModeSeedModel(SeedModel):
    def __init__(self, actions, family, initial_parameters=None):
        super().__init__(actions, straight_cusps=True)
        if family not in FAMILIES:
            raise ValueError('Select a predeclared existing-mode family')
        self.family = family
        self.leg_modes = ['front_only' if family==FAMILIES[0] or d<0 else 'coordinated_counterphase'
                          for d in self.leg_directions]
        self.modes = [self.leg_modes[i] for i in self.leg]
        self.initial_parameters = None if initial_parameters is None else np.asarray(initial_parameters,float).copy()
        if self.initial_parameters is not None:
            self.unpack(self.initial_parameters)

    def initial(self, extended_reverse=False):
        if extended_reverse:
            raise ValueError('No unreported initialization change in this family')
        return self.initial_parameters.copy() if self.initial_parameters is not None else super().initial()

    def integrate(self, z, substeps=2, jacobian=False, independent=False):
        if not isinstance(substeps,int) or substeps<1:
            raise ValueError('Positive integration subdivision required')
        knots,lengths = self.unpack(z)
        pose=np.zeros(3); derivative=np.zeros((3,self.p))
        poses=[pose.copy()]; derivatives=[derivative.copy()]
        distance=[0.]; curvature=[knots[0]]; controls=[-1]
        for i in range(self.n):
            arm = .645 if self.modes[i]=='front_only' else 0.
            count=max(substeps,math.ceil(lengths[self.leg[i]]/self.leg_counts[self.leg[i]]/.0025)) if independent else substeps
            ds=lengths[self.leg[i]]/self.leg_counts[self.leg[i]]/count*self.direction[i]
            dds=np.zeros(self.p);dds[len(self.free)+self.leg[i]]=self.direction[i]/self.leg_counts[self.leg[i]]/count
            for j in range(count):
                a,b=j/count,(j+1)/count
                left=(1-a)*knots[i]+a*knots[i+1]
                right=(1-b)*knots[i]+b*knots[i+1]
                q=.375*left+.125*right
                middle=pose[2]+q*ds; end=pose[2]+.5*(left+right)*ds
                kmid=.5*(left+right)
                def tangent(yaw,k):
                    c,s=math.cos(yaw),math.sin(yaw)
                    return np.array([c-arm*k*s,s+arm*k*c])
                if independent:
                    pose[:2]+=ds/6*(tangent(pose[2],left)+4*tangent(middle,kmid)+tangent(end,right))
                else:
                    pose[:2]+=ds*tangent(middle,kmid)
                if jacobian:
                    dl=(1-a)*self.k_jac[i]+a*self.k_jac[i+1]
                    dr=(1-b)*self.k_jac[i]+b*self.k_jac[i+1]
                    dm=derivative[2]+ds*(.375*dl+.125*dr)+q*dds
                    dk=.5*(dl+dr);c,s=math.cos(middle),math.sin(middle)
                    derivative[0]+=dds*(c-arm*kmid*s)+ds*((-s-arm*kmid*c)*dm-arm*s*dk)
                    derivative[1]+=dds*(s+arm*kmid*c)+ds*((c-arm*kmid*s)*dm+arm*c*dk)
                    derivative[2]+=ds*.5*(dl+dr)+.5*(left+right)*dds
                pose[2]=end;poses.append(pose.copy())
                if jacobian:derivatives.append(derivative.copy())
                distance.append(distance[-1]+abs(ds));curvature.append(right);controls.append(i)
        return np.array(poses),np.array(derivatives) if jacobian else None,np.array(distance),np.array(curvature),np.array(controls)


def audit_geometry(model,z,manifest,occupied,origin,resolution):
    result=old_geometry_audit(model,z,manifest,occupied,origin,resolution)
    poses,_,longitudinal,k,controls=model.integrate(z,independent=True)
    interval_direction=model.direction[controls[1:]]
    interval_modes=[model.modes[i] for i in controls[1:]]
    center_ds=np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)
    cusps=np.flatnonzero(np.diff(interval_direction))+1
    stops=np.r_[0,cusps,len(poses)-1]
    legs=[float(sum(center_ds[a:b])) for a,b in zip(stops[:-1],stops[1:])]
    reverse=[length for length,d in zip(legs,model.leg_directions) if d<0]
    # Existing distance resources refer to actual body-center path distance;
    # do not silently substitute the shorter longitudinal parameter in a
    # single-axle mode with nonzero body-lateral velocity.
    result.update(travelled_longitudinal_parameter_m=longitudinal.tolist(),
        travelled_arc_m=np.r_[0.,np.cumsum(center_ds)].tolist(),total_path_length_m=float(sum(center_ds)),
        total_longitudinal_travel_m=float(longitudinal[-1]),total_reverse_m=sum(reverse),
        leg_longitudinal_distances_m=model.unpack(z)[1].tolist(),leg_lengths_m=legs,
        leg_modes=model.leg_modes,interval_modes=interval_modes,interval_directions=interval_direction.tolist(),
        family=model.family,body_lateral_motion_propagated=True)
    result['checks'].update(original_reverse_episode_bound=max(reverse,default=0.)<=.5+1e-9,
        original_total_reverse_bound=sum(reverse)<=1.+1e-9,
        all_mode_changes_nominally_straight=all(abs(k[i])<1e-12 for i in cusps))
    result['verdict']='PASS' if all(result['checks'].values()) else 'FAIL'
    return result


def nominal_timing(geometry):
    x=np.array(geometry['travelled_longitudinal_parameter_m']);k=np.array(geometry['sample_curvature_1_m'])
    direction=np.array(geometry['interval_directions']);modes=geometry['interval_modes']
    if len(x)<3 or len(k)!=len(x) or len(direction)!=len(x)-1 or len(modes)!=len(direction):
        raise ValueError('Complete sampled mode-aware geometry required')
    dx=np.diff(x)
    if not np.isfinite(x).all() or not np.isfinite(k).all() or np.any(dx<=0) or max(abs(k))>.7+1e-10:
        raise ValueError('Finite bounded curvature and increasing longitudinal parameter required')
    arms=np.array([1.29 if m=='front_only' else .645 for m in modes])
    minimum=np.minimum(abs(k[:-1]),abs(k[1:]));minimum[k[:-1]*k[1:]<=0]=0.
    slope=arms*abs(np.diff(k)/dx)/(1+(arms*minimum)**2)
    caps=np.where(direction>0,.05,.01)
    moving=slope>1e-15;caps[moving]=np.minimum(caps[moving],math.radians(8.)/slope[moving])
    v=np.r_[caps[0],np.minimum(caps[:-1],caps[1:]),caps[-1]]
    cusps=np.flatnonzero(np.diff(direction))+1;stops=np.r_[0,cusps,len(x)-1]
    if np.any(abs(k[stops])>1e-10):raise ValueError('Mode switches/cusps/start/end must be nominally straight')
    v[stops]=0.
    for i,d in enumerate(dx):v[i+1]=min(v[i+1],math.sqrt(v[i]**2+2*.02*d))
    for i in range(len(dx)-1,-1,-1):v[i]=min(v[i],math.sqrt(v[i+1]**2+2*.02*dx[i]))
    if np.any(v[:-1]+v[1:]<=0):raise ValueError('Intermediate samples required between stops')
    dt=2*dx/(v[:-1]+v[1:]);time=np.r_[0.,np.cumsum(dt)]
    for i in cusps:time[i+1:]+=.75
    rate=slope*np.maximum(v[:-1],v[1:]);accel=np.diff(v*v)/(2*dx)
    checks={'nominal_speed_caps':bool(np.all(np.maximum(v[:-1],v[1:])<=caps+1e-12)),
        'nominal_acceleration_cap':bool(max(abs(accel))<=.02+1e-12),
        'mode_aware_nominal_slew_cap':bool(max(rate)<=math.radians(8.)+1e-12),
        'zero_reference_speed_and_curvature_at_stops':bool(np.all(v[stops]==0.) and np.all(abs(k[stops])<1e-10))}
    return {'status':'MODE_AWARE_NOMINAL_TIMING_ONLY_DYNAMICS_AND_EXECUTION_PENDING',
        'checks':checks,'reference_motion_s':float(sum(dt)),
        'reference_motion_plus_nominal_cusp_dwell_s':float(time[-1]),
        'maximum_reference_steering_slew_deg_s_upper_bound':math.degrees(max(rate)),
        'sample_unsigned_reference_speed_m_s':v.tolist(),'time_arrival_s':time.tolist(),
        'interval_motion_duration_s':dt.tolist(),'interval_directions':direction.tolist(),
        'interval_modes':modes,'cusp_sample_indices':cusps.tolist(),
        'reverse_turning_PhysX_verified':False,'directional_rigid_body_model_propagated':False,
        'measured_stop_hold_alignment_release_verified':False,'PhysX_executed':False}


def optimize(model,manifest,occupied,origin,resolution,budget_s=60.,heartbeat=None):
    if not math.isfinite(budget_s) or not 0<budget_s<=90.:raise ValueError('Bounded offline budget <=90 s required')
    started=time.perf_counter()
    objective=Objective(model,manifest,occupied,origin,resolution,started+budget_s,heartbeat)
    initial=model.initial();status='OPTIMIZER_COMPLETED_NOT_A_CERTIFICATE';optimizer=None
    try:
        optimizer=least_squares(lambda z:objective.evaluate(z)[0],initial,jac=lambda z:objective.evaluate(z)[1],
            bounds=model.bounds(),method='trf',tr_solver='lsmr',x_scale='jac',max_nfev=160,
            ftol=1e-7,xtol=1e-7,gtol=1e-7)
    except TimeoutError:status='TIME_BUDGET_EXHAUSTED_NOT_GLOBAL_INFEASIBILITY'
    z=objective.best[1] if objective.best is not None else initial
    geometry=audit_geometry(model,z,manifest,occupied,origin,resolution)
    timing=nominal_timing(geometry) if geometry['verdict']=='PASS' else None
    admitted=geometry['verdict']=='PASS' and all(timing['checks'].values())
    return {'status':'MODE_AWARE_SPATIAL_AND_NOMINAL_TIMING_PASS_DYNAMIC_GATE_PENDING' if admitted else status,
        'geometry_audit':geometry,'nominal_timing':timing,'parameters':z.tolist(),
        'objective_evaluations':objective.calls,'elapsed_s':time.perf_counter()-started,'budget_s':budget_s,
        'optimizer_message':str(optimizer.message) if optimizer is not None else None,
        'global_infeasibility_proven':False,'dynamic_feasibility_certified':False,
        'P_full_dynamics_plan_selected':False,'PhysX_executed':False,'use_for_final_statistics':False}
