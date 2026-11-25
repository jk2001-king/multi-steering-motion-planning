"""Separate steering-response candidates; NEVER mutate legacy model constants.

Fit JOINT ANGLES only, not H3 endpoint/contacts. Applied command for row i is
held over (t[i-1],t[i]]; no measured-state resets beyond the initial sample.
Neutral offset is anchored to training pre-hold, not fitted from turning error.
Select the simplest adequate lag candidate from TRAINING only and freeze it
before prospective validation. Second-order unit-gain response and legacy
-1deg/slew are retained comparisons, never selected from an H3 endpoint.
No fit is a prospective validation or a caster/contact causality proof.
"""

import math

import numpy as np


VERSION='H3B_separate_joint_response_identification_v1'
PRIMARY='first_order_unit_gain'
SECOND_ORDER='second_order_unit_gain'
FAMILIES=('legacy_minus1deg_slew','neutral_offset_slew',PRIMARY,SECOND_ORDER)
PARAMETER_BOUNDS={'first_order_unit_gain':{'tau_s':(.02,2.)},
    SECOND_ORDER:{'natural_frequency_rad_s':(.5,25.),'damping_ratio':(.05,2.)}}
VALIDATION_CRITERIA={'moving_rmse_deg_max':1.,'moving_p95_absolute_error_deg_max':2.5,
    'moving_max_absolute_error_deg_max':5.,'moving_rmse_ratio_to_legacy_max':.80,
    'scope':'predeclared_simulation_model_acceptance_not_clinical_or_whole_route_limit'}


def prepare_trace(rows):
    if len(rows)<3:raise ValueError('Complete applied-command/measured-angle trace required')
    time=np.array([r['time_s'] for r in rows],float)
    command_time=np.array([r['command_time_s'] for r in rows],float)
    if (not np.isfinite(time).all() or np.any(np.diff(time)<=0) or np.max(np.diff(time))>.25
            or not np.isfinite(command_time).all()
            or not np.allclose(command_time[1:],time[:-1],rtol=0,atol=1e-7)):
        raise ValueError('Applied-command/observation chronology or existing 0.25 s gap differs')
    pre=np.array([r['phase']=='pre_hold' and r['phase_time_s']>=1. for r in rows])
    moving=np.array([r['phase'].startswith('leg_0_') and r['command_speed_m_s']>.003 for r in rows])
    if np.sum(pre)<10 or np.sum(moving)<20:raise ValueError('Need predeclared neutral anchor and moving response samples')
    actual=np.array([[r[a+'_rad'] for a in ('front','rear')] for r in rows],float)
    command=np.array([[r['command_'+a+'_rad'] for a in ('front','rear')] for r in rows],float)
    if not np.isfinite(actual).all() or not np.isfinite(command).all():raise ValueError('Finite joint telemetry required')
    if np.max(abs(command[pre]))>math.radians(.1):raise ValueError('Training neutral offset must use zero steering commands')
    return time,command,actual,pre,moving


def transition_second_order(omega,zeta,dt):
    if not all(math.isfinite(v) for v in (omega,zeta,dt)) or omega<=0 or zeta<=0 or not 0<dt<=.25:
        raise ValueError('Finite stable model and bounded positive dt required')
    a=np.array([[0.,1.],[-omega*omega,-2*zeta*omega]])
    shifted=a+zeta*omega*np.eye(2);d=zeta*zeta-1.
    if abs(d)<1e-9:return math.exp(-zeta*omega*dt)*(np.eye(2)+shifted*dt)
    q=omega*math.sqrt(abs(d));angle=q*dt
    if d<0:return math.exp(-zeta*omega*dt)*(np.eye(2)*math.cos(angle)+shifted*(math.sin(angle)/q))
    return math.exp(-zeta*omega*dt)*(np.eye(2)*math.cosh(angle)+shifted*(math.sinh(angle)/q))


def predict(time,command,initial_angle,family,parameters):
    time,command=np.asarray(time,float),np.asarray(command,float)
    if (time.ndim!=1 or command.shape!=time.shape or len(time)<2 or not np.isfinite(time).all()
            or not np.isfinite(command).all() or np.any(np.diff(time)<=0) or np.max(np.diff(time))>.25
            or not math.isfinite(initial_angle) or family not in FAMILIES):
        raise ValueError('Finite chronological bounded response inputs required')
    bias=float(parameters.get('neutral_offset_rad',0.))
    if not math.isfinite(bias) or abs(bias)>math.radians(5.):raise ValueError('Finite independently recorded neutral anchor required')
    output=np.empty(len(time));output[0]=initial_angle
    if family in ('legacy_minus1deg_slew','neutral_offset_slew'):
        bias=math.radians(-1.) if family=='legacy_minus1deg_slew' else bias
        for i,dt in enumerate(np.diff(time),1):
            output[i]=output[i-1]+np.clip(command[i]+bias-output[i-1],-math.radians(8.)*dt,math.radians(8.)*dt)
    elif family=='first_order_unit_gain':
        tau=float(parameters['tau_s'])
        if not .02<=tau<=2.:raise ValueError('Predeclared lag-time bounds exceeded')
        for i,dt in enumerate(np.diff(time),1):
            target=command[i]+bias;output[i]=target+(output[i-1]-target)*math.exp(-dt/tau)
    else:
        omega,zeta=float(parameters['natural_frequency_rad_s']),float(parameters['damping_ratio'])
        if not .5<=omega<=25. or not .05<=zeta<=2.:raise ValueError('Predeclared second-order bounds exceeded')
        state=np.array([initial_angle,0.]);cache={}
        for i,dt in enumerate(np.diff(time),1):
            key=round(float(dt),10)
            if key not in cache:cache[key]=transition_second_order(omega,zeta,float(dt))
            target=np.array([command[i]+bias,0.]);state=target+cache[key]@(state-target);output[i]=state[0]
    if not np.isfinite(output).all():raise ValueError('Unstable/nonfinite response prediction')
    return output


def error_metrics(observed,predicted,mask=None):
    observed,predicted=np.asarray(observed,float),np.asarray(predicted,float)
    if observed.shape!=predicted.shape or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('Finite aligned prediction/observation arrays required')
    error=np.degrees(predicted-observed)
    if mask is not None:error=error[np.asarray(mask,bool)]
    if not len(error):raise ValueError('No samples for the predeclared metric window')
    return {'sample_count':len(error),'rmse_deg':float(np.sqrt(np.mean(error*error))),
        'mean_error_deg':float(np.mean(error)),'p95_absolute_error_deg':float(np.quantile(abs(error),.95)),
        'max_absolute_error_deg':float(np.max(abs(error)))}


def fit_training(rows):
    from scipy.optimize import least_squares
    time,command,actual,pre,moving=prepare_trace(rows)
    report={};predictions={}
    for column,axle in enumerate(('front','rear')):
        offset=float(np.mean(actual[pre,column]-command[pre,column]));models={};predictions[axle]={}
        for family in FAMILIES:
            parameters={'neutral_offset_rad':math.radians(-1.) if family=='legacy_minus1deg_slew' else offset};optimization=None
            if family=='first_order_unit_gain':
                def residual(z):return predict(time,command[:,column],actual[0,column],family,{**parameters,'tau_s':z[0]})-actual[:,column]
                optimization=least_squares(residual,[.3],bounds=([.02],[2.]),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                parameters['tau_s']=float(optimization.x[0])
            elif family==SECOND_ORDER:
                def residual(z):return predict(time,command[:,column],actual[0,column],family,
                    {**parameters,'natural_frequency_rad_s':z[0],'damping_ratio':z[1]})-actual[:,column]
                optimization=least_squares(residual,[4.,.7],bounds=([.5,.05],[25.,2.]),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                parameters.update(natural_frequency_rad_s=float(optimization.x[0]),damping_ratio=float(optimization.x[1]))
            predicted=predict(time,command[:,column],actual[0,column],family,parameters)
            models[family]={'parameters':parameters,'all_training_samples':error_metrics(actual[:,column],predicted),
                'moving_training_samples':error_metrics(actual[:,column],predicted,moving),
                'optimizer_success':None if optimization is None else bool(optimization.success),
                'optimizer_nfev':None if optimization is None else int(optimization.nfev)}
            predictions[axle][family]=predicted
        report[axle]={'neutral_anchor_sample_count':int(np.sum(pre)),'measured_neutral_offset_rad':offset,'models':models}
    return {'version':VERSION,'primary_family_frozen_before_prospective_validation':PRIMARY,'axles':report,
        'primary_family_selected_using_training_only':True,
        'primary_selection_reason':'first-order lag is the simpler candidate; no second-order training-error advantage; prospective validation still mandatory',
        'parameter_bounds':PARAMETER_BOUNDS,'prospective_validation_criteria':VALIDATION_CRITERIA,
        'DC_gain_fixed_to_one':True,'initial_rate_fixed_to_zero':True,'fit_objective':'all_training_joint_angle_samples_only',
        'H3_endpoint_or_contact_outcome_used_in_fit':False,'fit_used_future_validation':False,
        'legacy_constants_or_physical_gains_changed':False,'new_model_deployed_to_planner_or_controller':False,
        'identified_scope':'single_counterphase_50mm_s_moderate_S_training_trace_only_not_reverse_or_large_angle',
        'status':'CANDIDATES_FROZEN_PROSPECTIVE_VALIDATION_PENDING_NOT_MODEL_OR_H3_PASS'},predictions
