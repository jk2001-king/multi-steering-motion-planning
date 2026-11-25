"""Pure full-ROI task binding of existing lattice/planner code.

No new propagator, method-specific bounds, fallback path or live executor.
Direction choices are REQUIRED explicit development inputs, not inferred
from H1's forward-only call. This module grants no motion/reverse authority.
"""

import math
from dataclasses import dataclass, fields, make_dataclass, replace
from types import FunctionType


def planar_reference_to_com_review(velocity,body_derivative,com_xy_m,*,yaw_radius_com_m):
    """OFFLINE point/frame audit, NOT a replacement of the frozen NE kernel.

    O and G have the SAME body-axis orientation. velocity=(u_O,v_O,r),
    body_derivative=(du_O/dt,dv_O/dt,dr/dt); c=OG is fixed in body axes.
    F/M is acceleration of G, while moments about G and O differ by c x F/M.
    No identified residual box, contact allocation or model feasibility is
    transferred/certified by this identity. In particular a residual fitted
    at O cannot silently become an independently validated G-frame bound.
    """
    u,v,r=_finite(velocity,3,'reference-point body velocity')
    du,dv,alpha=_finite(body_derivative,3,'body-axis velocity derivative')
    cx,cy=_finite(com_xy_m,2,'fixed reference-to-CoM vector')
    if (type(yaw_radius_com_m) not in (int,float) or not math.isfinite(yaw_radius_com_m)
            or yaw_radius_com_m<=0.):
        raise ValueError('Positive finite CoM yaw radius required')
    ug,vg=u-r*cy,v+r*cx
    dug,dvg=du-alpha*cy,dv+alpha*cx
    axo,ayo=du-r*v,dv+r*u
    axg,ayg=axo-alpha*cy-r*r*cx,ayo+alpha*cx-r*r*cy
    ng=yaw_radius_com_m**2*alpha
    no=ng+cx*ayg-cy*axg
    return dict(version='OFFLINE_PLANAR_REFERENCE_TO_COM_IDENTITY_v1',
        reference_body_velocity=[u,v,r],com_body_velocity=[ug,vg,r],
        reference_body_velocity_derivative=[du,dv,alpha],
        com_body_velocity_derivative=[dug,dvg,alpha],reference_to_com_xy_m=[cx,cy],
        reference_point_inertial_accel_in_body_m_s2=[axo,ayo],
        com_specific_force_in_body_m_s2=[axg,ayg],
        omitted_reference_to_com_specific_force_m_s2=[axg-axo,ayg-ayo],
        specific_yaw_moment_about_com_m2_s2=ng,
        specific_yaw_moment_about_reference_m2_s2=no,
        parallel_axis_yaw_radius_squared_m2=yaw_radius_com_m**2+cx*cx+cy*cy,
        frozen_planning_propagation_and_acceptance_NOT_changed=True,
        not_a_contact_motor_robustness_or_physical_validation=True,
        residual_bounds_require_their_own_frame_and_statistical_scope=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def _frame_interval_result(lower,upper):
    """Outward-rounded enclosure; fail closed on arithmetic overflow."""
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError('Finite frame-interval arithmetic domain exceeded')
    result=math.nextafter(lower,-math.inf),math.nextafter(upper,math.inf)
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Outward frame-interval arithmetic domain exceeded')
    return result


def _frame_interval_add(first,second):
    return _frame_interval_result(first[0]+second[0],first[1]+second[1])


def _frame_interval_sub(first,second):
    return _frame_interval_result(first[0]-second[1],first[1]-second[0])


def _frame_interval_mul(first,second):
    corners=[a*b for a in first for b in second]
    return _frame_interval_result(min(corners),max(corners))


def _frame_interval_div(first,positive_second):
    if positive_second[0]<=0. or positive_second[0]>positive_second[1]:
        raise ValueError('Positive interval denominator required')
    inverse=_frame_interval_result(1./positive_second[1],1./positive_second[0])
    return _frame_interval_mul(first,inverse)


class CoMTransportAugmentedEdgeModel:
    """EXPLICIT prospective CoM guard, NEVER the default/frozen force model.

    Keep the original reference-point response, original bounds and original
    acceptance. Additionally enclose CoM acceleration, aggregate support-line
    loads/friction and LTR using outward-rounded interval arithmetic. Thus no
    originally rejected step becomes accepted. This is a per-step CONDITIONAL
    model check, not uncertain-state propagation, individual motor capacity,
    six-wheel contact reconstruction, all-time residual or physical proof.

    Residual bounds were fitted to body-derivative error AT THE SAME measured
    reference state. The stored yaw-moment residual is rho_id^2 * delta_alpha,
    not an independently measured caster torque. Require that convention and
    rho_id explicitly; transport the derivative box BEFORE load allocation.
    Legacy fields named longitudinal/lateral_accel retain their original
    reference-specific-force meanings for schema compatibility. CoM values
    and the enclosure assumptions are exposed by review_step instead.

    Composition, not inheritance, is intentional: the original compiled NE
    adapter must fall back to Python, never silently skip this new guard.
    """
    revision='PROSPECTIVE_REFERENCE_RESPONSE_COM_INTERVAL_AUGMENTED_GUARD_v1'

    def __init__(self,bounds,*,residual_reference_definition,identification_yaw_radius_m):
        from patient_initial_stability.rigid_body_edge import (
            MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds,GRAVITY_M_S2)
        if type(bounds) is not RigidBodyParameterBounds:
            raise ValueError('Exact original bounded-parameter schema required')
        if any(type(getattr(bounds,f.name)) not in (int,float) or not math.isfinite(getattr(bounds,f.name))
                for f in fields(bounds)):
            raise ValueError('Finite explicit original parameters required')
        if residual_reference_definition!='base_reference_body_derivative_residuals_v1':
            raise ValueError('Explicit fitted reference-point derivative residual convention required')
        if (type(identification_yaw_radius_m) not in (int,float)
                or not math.isfinite(identification_yaw_radius_m) or identification_yaw_radius_m<=0.):
            raise ValueError('Positive finite ORIGINAL identification yaw radius required')
        self.bounds=bounds;self.original=MassNormalizedRigidBodyEdgeModel(bounds)
        self.gravity=GRAVITY_M_S2;self.identification_yaw_radius_m=float(identification_yaw_radius_m)
        try:scale=identification_yaw_radius_m**2
        except OverflowError:raise ValueError('Finite positive identification yaw scale required') from None
        if not math.isfinite(scale) or scale<=0.:raise ValueError('Finite positive identification yaw scale required')
        alpha_bound=math.nextafter(bounds.caster_specific_yaw_moment_abs_max_m2_s2/scale,math.inf)
        if not math.isfinite(alpha_bound):raise ValueError('Finite identified yaw-derivative interval required')
        self.residual_derivative_box=(bounds.caster_specific_force_x_abs_max_m_s2,
            bounds.caster_specific_force_y_abs_max_m_s2,alpha_bound)
        self.statistics=dict(step_calls=0,original_rejected_steps=0,additional_guard_rejected_steps=0)
        self.parameters=dict(revision=self.revision,residual_reference_definition=residual_reference_definition,
            identification_yaw_radius_m=float(identification_yaw_radius_m),
            identification_specific_yaw_scale_m2=scale,
            reference_body_derivative_residual_box=list(self.residual_derivative_box),
            outward_rounded_intervals_enclose_declared_continuous_parameter_box=True,
            independent_interval_overapproximation_NOT_exact_correlated_maximum=True,
            original_body_response_and_acceptance_retained=True,
            empirical_box_coverage_NOT_all_time_or_new_hospital_guarantee=True,
            residual_transport_conditioned_on_SAME_current_reference_state=True,
            nominal_acceleration_saturation_NOT_uncertain_motor_capacity_proof=True,
            no_uncertain_state_tube_or_contact_motor_or_physical_guarantee=True,
            original_compiled_model_type_guard_must_fall_back=True,
            not_default_or_adopted_runtime_or_final_comparison=True,
            actual_motion_authorized=False,use_for_final_statistics=False)

    def review_step(self,u,v,r,target_u,target_v,target_r,dt_s):
        values=_finite((u,v,r,target_u,target_v,target_r,dt_s),7,'finite prospective response inputs')
        u,v,r,target_u,target_v,target_r,dt_s=values
        if dt_s<=0.:raise ValueError('Positive finite step duration required')
        legacy=self.original.step(u,v,r,target_u,target_v,target_r,dt_s)
        b=self.bounds;g=self.gravity
        point=lambda value:(float(value),float(value))
        add,sub,mul=_frame_interval_add,_frame_interval_sub,_frame_interval_mul
        divide=lambda value,denominator:_frame_interval_div(value,point(denominator))
        absolute=lambda value:max(abs(value[0]),abs(value[1]))
        cx=(b.com_longitudinal_min_m,b.com_longitudinal_max_m)
        cy=(b.com_lateral_min_m,b.com_lateral_max_m)
        height=(b.com_height_min_m,b.com_height_max_m)
        rho2=_frame_interval_result(b.yaw_radius_min_m**2,b.yaw_radius_max_m**2)
        eu,ev,ea=self.residual_derivative_box
        alpha=add(point(legacy.yaw_accel_rad_s2),(-ea,ea))
        fx_o=add(point(legacy.longitudinal_accel_m_s2),(-eu,eu))
        fy_o=add(point(legacy.lateral_accel_m_s2),(-ev,ev))
        r2=mul(point(r),point(r))
        fx_g=sub(sub(fx_o,mul(alpha,cy)),mul(r2,cx))
        fy_g=sub(add(fy_o,mul(alpha,cx)),mul(r2,cy))
        ng=mul(rho2,alpha)
        af=sub(point(b.front_support_x_m),cx);ar=sub(cx,point(b.rear_support_x_m))
        length=sub(point(b.front_support_x_m),point(b.rear_support_x_m))
        nf=_frame_interval_div(sub(mul(point(g),ar),mul(height,fx_g)),length)
        nr=sub(point(g),nf)
        lateral=add(cy,divide(mul(height,fy_g),g))
        ltr=math.nextafter(2.*absolute(lateral)/b.track_width_m,math.inf)
        side_lower=math.nextafter(.5*(1.-ltr),-math.inf)
        minimum_normal=mul(divide(point(min(nf[0],nr[0])),g),point(max(0.,side_lower)))[0]
        fy_front=_frame_interval_div(add(ng,mul(ar,fy_g)),length)
        fy_rear=_frame_interval_div(sub(mul(af,fy_g),ng),length)
        if min(nf[0],nr[0])<=0.:
            front_util=rear_util=math.inf
        else:
            # Fx_i=(F_i_z/g)*fx_G in the declared support-line allocation.
            # Cancel its OWN positive normal before bounding the norm; do not
            # divide a maximum longitudinal load by an unrelated minimum.
            x_ratio=divide(point(absolute(fx_g)),g)[1]
            front_ratio=divide(point(absolute(fy_front)),nf[0])[1]
            rear_ratio=divide(point(absolute(fy_rear)),nr[0])[1]
            norm_upper=lambda lateral:math.nextafter(math.hypot(x_ratio,lateral),math.inf)
            front_util=divide(point(norm_upper(front_ratio)),b.friction_min)[1]
            rear_util=divide(point(norm_upper(rear_ratio)),b.friction_min)[1]
        friction=max(front_util,rear_util)
        support=min(nf[0],nr[0])>0. and side_lower>0.
        accepted=(support and friction<=b.max_friction_utilization and ltr<=b.max_abs_ltr)
        return legacy,dict(reference_specific_force_x_interval=fx_o,reference_specific_force_y_interval=fy_o,
            yaw_acceleration_interval=alpha,com_specific_force_x_interval=fx_g,com_specific_force_y_interval=fy_g,
            specific_yaw_moment_about_com_interval=ng,front_normal_interval=nf,rear_normal_interval=nr,
            front_lateral_specific_force_interval=fy_front,rear_lateral_specific_force_interval=fy_rear,
            front_friction_upper=front_util,rear_friction_upper=rear_util,friction_upper=friction,
            abs_ltr_upper=ltr,minimum_normal_fraction_lower=minimum_normal,support_positive=support,
            conditional_interval_guard_pass=accepted,interval_dependency_overapproximation=True,
            correlated_extrema_NOT_assumed_to_occur_together=True,
            legacy_step_pass=legacy.feasible,accepted_by_BOTH_guards=legacy.feasible and accepted)

    def step(self,u,v,r,target_u,target_v,target_r,dt_s):
        legacy,review=self.review_step(u,v,r,target_u,target_v,target_r,dt_s)
        self.statistics['step_calls']+=1
        self.statistics['original_rejected_steps']+=int(not legacy.feasible)
        self.statistics['additional_guard_rejected_steps']+=int(legacy.feasible and not review['conditional_interval_guard_pass'])
        b=self.bounds;front=max(legacy.front_friction_utilization,review['front_friction_upper'])
        rear=max(legacy.rear_friction_utilization,review['rear_friction_upper'])
        friction=max(front,rear);ltr=max(legacy.abs_ltr,review['abs_ltr_upper'])
        # Preserve +infinity as an unbounded rejected-candidate quantity;
        # avoid 0*infinity -> NaN for an explicitly disabled cost term.
        weighted=lambda weight,value:weight*value**2 if weight else 0.
        penalty=(weighted(b.friction_cost_weight_s,friction)+weighted(b.ltr_cost_weight_s,ltr)
            +weighted(b.actuator_cost_weight_s,legacy.actuator_utilization))*dt_s
        return replace(legacy,front_friction_utilization=front,rear_friction_utilization=rear,
            friction_utilization=friction,abs_ltr=ltr,
            minimum_normal_load_fraction=min(legacy.minimum_normal_load_fraction,review['minimum_normal_fraction_lower']),
            feasible=bool(review['accepted_by_BOTH_guards']),penalty_s=max(legacy.penalty_s,penalty))


def conditional_tracking_reserve_definition(*,position_error_m,heading_error_deg):
    """Conservative rectangle envelope for an explicitly ASSUMED tracking tube.

    For every point of the original body+10mm outline, bound rotated body
    coordinates over +/-heading_error, then add the translational disk's
    axis bounds. Uniform enlargement encloses that whole set. This is only
    conditional geometry: neither controller tracking nor physical contact
    is verified, and the existing80mm/10deg abort guards remain unchanged.
    """
    for value,maximum,name in ((position_error_m,.080,'position'),(heading_error_deg,10.,'heading')):
        if type(value) not in (int,float) or not math.isfinite(value) or not 0.<=value<=maximum:
            raise ValueError('Explicit nonnegative tracking '+name+' within original abort guard required')
    a,b=1.630/2+.010,.965/2+.010;angle=math.radians(heading_error_deg)
    def extent(first,second):
        candidates=(0.,angle,min(angle,math.atan2(second,first)))
        return max(first*math.cos(v)+second*math.sin(v) for v in candidates)
    x_extra=extent(a,b)-a;y_extra=extent(b,a)-b
    exact=float(position_error_m)+max(x_extra,y_extra)
    quantum=.005;reserved=math.ceil(exact/quantum)*quantum
    return dict(version='CONDITIONAL_ORIGINAL_BODY_TRACKING_TUBE_PLANNING_RESERVE_v1',
        physical_body_m=[1.630,.965],unchanged_required_safety_clearance_m=.010,
        assumed_position_error_norm_m=float(position_error_m),assumed_heading_error_deg=float(heading_error_deg),
        rotated_outline_extra_body_x_m=x_extra,rotated_outline_extra_body_y_m=y_extra,
        exact_uniform_additional_reserve_m=exact,upward_reserve_quantum_m=quantum,
        planning_additional_execution_reserve_m=reserved,planning_total_clearance_m=.010+reserved,
        original_abort_guards=dict(path_error_m=.080,path_heading_error_deg=10.),
        conditional_geometric_containment_NOT_proven_controller_tracking=True,
        original_abort_guards_and_static_terminal_acceptance_NOT_changed=True,
        all_four_methods_must_use_same_development_definition=True,
        actual_motion_authorized=False,use_for_final_statistics=False)


def goal_predicate_separating_state_key(original_key,goal,*,terminal_speed_m_s):
    """Keep an exact accepted-goal label separate from binned non-goal labels.

    Original25mm XY/1deg yaw/10mm_s velocity bins can straddle the unchanged
    25mm/2deg/5mm_s terminal boundary. A cheaper outside-goal label in the
    SAME bin must not erase a valid goal label. Add one boolean before the
    resource wrapper appends its three history fields; change no edge/bin,
    tolerance, dynamics or original source. This is an explicit search-key
    revision, NOT a completeness/optimality guarantee.
    """
    goal=_finite(goal,3,'fixed key goal')
    if not callable(original_key) or type(terminal_speed_m_s) not in (int,float) or terminal_speed_m_s!=.005:
        raise ValueError('Original key and unchanged5mm_s MODEL goal speed required')
    counters=dict(key_calls=0,exact_accepted_goal_key_calls=0)
    def accepted(state,include_rigid_body):
        return bool(state.started and math.hypot(state.x-goal[0],state.y-goal[1])<=.025
            and abs(math.atan2(math.sin(state.yaw-goal[2]),math.cos(state.yaw-goal[2])))<=math.radians(2.)
            and (not include_rigid_body or math.hypot(state.longitudinal_velocity,state.lateral_velocity)<=terminal_speed_m_s))
    def key(state,include_rigid_body=False):
        base=original_key(state,include_rigid_body);at_goal=accepted(state,include_rigid_body)
        counters['key_calls']+=1;counters['exact_accepted_goal_key_calls']+=int(at_goal)
        return (*base,at_goal)
    key.parameters=dict(revision='original_physical_bins_plus_exact_goal_predicate_v1',
        original_state_bins_unchanged=True,original_edge_and_goal_constraints_unchanged=True,
        no_global_completeness_or_optimality_guarantee=True)
    key.counters=counters;key.accepted_goal=accepted
    return key


def first_terminal_neighborhood_prefix(records,goal,*,entry_radius_m):
    """Cut an OWN planning parent chain at first entry, never reset its state.

    Selection only: original full-state NE replay remains mandatory. An old
    cached chain stays a cached diagnosis, not free fresh planning input. In
    a fresh pipeline the whole generation/time/node cost must still be paid.
    This does not impose a new goal steering angle/speed or grant a physical
    mid-mission restart. The radius is an explicit soft search-stage policy.
    """
    from copy import deepcopy
    goal=_finite(goal,3,'fixed terminal-neighborhood goal')
    if (type(entry_radius_m) not in (int,float) or not math.isfinite(entry_radius_m)
            or not .025<entry_radius_m<=1.63 or not records or records[0].get('action') is not None):
        raise ValueError('Original rooted parent chain and explicit25mm<radius<=1.63m required')
    selected=None;previous=None
    for index,row in enumerate(records):
        state=row['state'];_finite([state[k] for k in ('x','y','yaw','front','rear',
            'longitudinal_velocity','lateral_velocity','yaw_rate')],8,'complete finite terminal-entry state')
        if (type(row.get('node_id')) is not int or not isinstance(row.get('cost_s'),(int,float))
                or not math.isfinite(row['cost_s']) or row['cost_s']<0.
                or previous is not None and (row['node_id']<=previous['node_id'] or row['cost_s']<previous['cost_s'])):
            raise ValueError('Ordered original node identities and cumulative costs required')
        distance=math.hypot(state['x']-goal[0],state['y']-goal[1])
        if selected is None and distance<=entry_radius_m:selected=(index,distance)
        previous=row
    if selected is None:raise ValueError('Actual parent chain never enters the declared terminal neighborhood')
    index,distance=selected
    if index==0:raise ValueError('Do not replace the original whole-mission start with a terminal root')
    prefix=deepcopy(records[:index+1])
    return prefix,dict(revision='own_parent_chain_first_terminal_neighborhood_v1',
        entry_radius_m=float(entry_radius_m),source_chain_edges=len(records)-1,selected_prefix_edges=index,
        source_node_id=prefix[-1]['node_id'],source_prefix_cost_s=prefix[-1]['cost_s'],
        selected_distance_m=distance,complete_incoming_state=deepcopy(prefix[-1]['state']),
        original_world_frame_goal_tolerances_and_all_full_states_unchanged=True,
        no_new_goal_steering_angle_or_velocity_requirement=True,
        whole_fresh_generation_cost_must_be_charged=True,cached_input_is_NOT_primary_success=True,
        original_NE_prefix_replay_STILL_required=True,actual_motion_authorized=False)


def replay_saved_complete_NE_prefix(records,State,propagate,checker,model):
    """Authenticate a saved parent chain with ORIGINAL complete-state NE replay.

    Exact field equality, original50mm primitives, resource history and every
    swept collision check are required. This does not restore any live robot
    or promote a partial prefix into a completed planning/physical result.
    """
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    if not records or records[0].get('action') is not None:
        raise ValueError('Complete saved prefix beginning at its original root required')
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    if asdict(state)!=records[0]['state'] or records[0]['cost_s']!=0.:
        raise ValueError('Exact original nominal root including all resource fields required')
    if checker([(state.x,state.y,state.yaw)]):raise ValueError('Saved prefix root footprint blocked')
    times=[];previous=records[0]
    for index,row in enumerate(records[1:],1):
        action=row.get('action')
        if (not isinstance(action,list) or len(action)!=4 or action[0] not in lattice.ALL_MODES
                or type(action[2]) is not int or action[2] not in (-1,1)
                or action[1] not in lattice.CURVATURES or action[3] not in (.003,.01,.03,.05)
                or row['node_id']<=previous['node_id'] or row['cost_s']<previous['cost_s']):
            raise ValueError('Original ordered NE controls/parent costs required')
        state,poses,elapsed=propagate(state,*action[:3],speed_m_s=.050,dynamics=lattice.FULL_DYNAMICS,
            rigid_body_model=model,target_speed_m_s=action[3])
        if (asdict(state)!=row['state'] or state.rigid_body_feasible is not True
                or not math.isfinite(elapsed) or elapsed<=0. or not poses.resource_feasible
                or checker(poses)):
            raise ValueError('Original complete-state NE/swept/resource replay differs at prefix edge '+str(index))
        times.append(elapsed);previous=row
    return state,dict(verified_edges=len(times),original_edge_times_s=times,
        complete_final_state=asdict(state),exact_all_physical_steering_velocity_resource_fields=True,
        cached_prefix_is_NOT_free_primary_planning_input=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def audit_complete_NE_terminal_segment(initial,actions,saved_states,propagate,model,checker,goal,*,step_m):
    """Independent full-state replay of an explicitly length-tagged local segment.

    Preserve original25mm/2deg/5mm_s MODEL goal and stopping/resource checks.
    A25mm edge diagnostic cannot be fed into the old50mm action serializer or
    called a fresh matched primary mission/physical success.
    """
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    goal=_finite(goal,3,'unchanged local MODEL goal')
    if type(step_m) is not float or step_m not in (.025,.050) or len(actions)!=len(saved_states) or not actions:
        raise ValueError('Explicit25/50mm segment with complete states required')
    state=initial;times=[];original=asdict(initial);maximum_stopping_violation=0.
    if any(name not in original for name in ('mission_reverse_episode_bins','mission_reverse_total_bins','mission_direction_changes')):
        raise ValueError('Complete inherited reverse history required')
    for index,(action,saved) in enumerate(zip(actions,saved_states)):
        if (len(action)!=4 or action[0] not in lattice.ALL_MODES or action[1] not in lattice.CURVATURES
                or type(action[2]) is not int or action[2] not in (-1,1) or action[3] not in (.003,.01,.03,.05)):
            raise ValueError('Original finite NE control set required')
        state,poses,elapsed=propagate(state,*action[:3],step_m=step_m,speed_m_s=.05,
            dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=action[3])
        if (asdict(state)!=asdict(saved) or not math.isfinite(elapsed) or elapsed<=0.
                or not state.rigid_body_feasible or not poses.resource_feasible or checker(poses)):
            raise ValueError('Independent complete-state/geometry/force/resource segment replay failed at '+str(index))
        distance=math.hypot(state.x-goal[0],state.y-goal[1])
        bound=math.sqrt(.005**2+2*.020*max(0.,distance-.025))
        maximum_stopping_violation=max(maximum_stopping_violation,math.hypot(state.longitudinal_velocity,state.lateral_velocity)-bound)
        times.append(elapsed)
    position=math.hypot(state.x-goal[0],state.y-goal[1])
    heading=math.degrees(math.atan2(math.sin(state.yaw-goal[2]),math.cos(state.yaw-goal[2])))
    speed=math.hypot(state.longitudinal_velocity,state.lateral_velocity)
    checks=dict(complete_state_geometry_force_and_resource_replay=True,
        original_stopping_reachability=max(maximum_stopping_violation,0.)<=1e-12,
        original_goal_position_under25mm=position<=.025,original_goal_heading_under2deg=abs(heading)<=2.,
        original_goal_body_speed_under5mm_s=speed<=.005,incoming_state_unmodified=asdict(initial)==original)
    return dict(verdict='PASS' if all(checks.values()) else 'FAIL',checks=checks,primitive_step_m=step_m,
        complete_incoming_state=original,complete_final_state=asdict(state),edge_times_s=times,
        terminal_position_error_m=position,terminal_heading_error_deg=heading,terminal_body_speed_m_s=speed,
        maximum_stopping_reachability_violation_m_s=max(maximum_stopping_violation,0.),
        original50mm_whole_mission_schema_NOT_used=True,cached_prefix_is_NOT_free_primary_planning_input=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def additional_planning_reserve_checker(original_checker,*,reserve_m):
    """Original checker/code/map, with an explicit ADDITIONAL planning margin.

    No occupied cells are erased or source geometry edited. The original
    physical safety definition is10mm; this larger reference constraint is
    separately reported and never silently presented as the old algorithm.
    """
    if type(reserve_m) not in (int,float) or not math.isfinite(reserve_m) or not 0.<=reserve_m<=.250:
        raise ValueError('Explicit bounded nonnegative ADDITIONAL planning reserve required')
    class Checker(original_checker):
        planning_execution_reserve_m=float(reserve_m)
        original_physical_safety_clearance_m=.010
        def __init__(self,*args,clearance=.010,**kwargs):
            if clearance!=.010:raise ValueError('Do not replace or double-count original10mm safety clearance')
            super().__init__(*args,clearance=.010+float(reserve_m),**kwargs)
    return Checker


def additional_planning_reserve_audit(original_audit,*,reserve_m):
    """Original independent repropagation, private additional-margin SAT only."""
    if type(reserve_m) not in (int,float) or not math.isfinite(reserve_m) or not 0.<=reserve_m<=.250:
        raise ValueError('Explicit bounded nonnegative ADDITIONAL audit reserve required')
    collision=original_audit.__globals__['pose_collision']
    def enlarged(occupied,pose,origin,resolution,clearance=.010):
        if clearance!=.010:raise ValueError('Independent physical safety clearance must remain10mm')
        return collision(occupied,pose,origin,resolution,.010+float(reserve_m))
    result=FunctionType(original_audit.__code__,dict(original_audit.__globals__,pose_collision=enlarged),
        original_audit.__name__,original_audit.__defaults__,original_audit.__closure__)
    result.__kwdefaults__=None if original_audit.__kwdefaults__ is None else dict(original_audit.__kwdefaults__)
    return result


def forward_common_v3_execution_binding(geometry, *, simulation_budget_s):
    """H6 schema/frame adapter to the ORIGINAL forward-reference v3 code.

    This is a prospective offline binding, not an executable live workflow.
    Missing H4 reporting fields are derived from the explicit H6 geometry and
    original NE acceleration limit, never from changed paths or safety bands.
    Even a forward reference can request reverse terminal correction in v3;
    this adapter does NOT define or authorize a physical reverse policy.
    """
    import copy
    from types import SimpleNamespace
    import hospital_h4_path_execution as original
    scenario='H6_integrated_transfer_mission'
    geometry=copy.deepcopy(geometry)
    if (geometry.get('scenario')!=scenario
            or geometry.get('full_body_rectangle_m')!=[1.630,.965]
            or geometry.get('full_body_clearance_m')!=.010
            or geometry.get('actual_motion_authorized') is not False
            or geometry.get('use_for_final_statistics') is not False):
        raise ValueError('Explicit H6 original-body MODEL geometry required')
    start=_finite(geometry['start_xyyaw'],3,'H6 start')
    goal=_finite(geometry['goal_xyyaw'],3,'H6 goal')
    if type(simulation_budget_s) not in (int,float) or simulation_budget_s not in (180.,240.,300.):
        raise ValueError('Explicit common prospective 180/240/300s MODEL resource bound required')
    def bind(function,**values):
        result=FunctionType(function.__code__,dict(function.__globals__,**values),
            function.__name__,function.__defaults__,function.__closure__)
        result.__kwdefaults__=None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
        return result
    legacy_validate=bind(original.validate_plan,SCENARIO=scenario)
    def validate(plan):
        if (plan.get('scenario')!=scenario or tuple(plan.get('start_world_xyyaw',()))!=start
                or tuple(plan.get('goal_world_xyyaw',()))!=goal
                or plan.get('planning_occupancy_digest')!=geometry['planning_occupancy_digest']
                or plan.get('physical_execution_verified') is not False
                or plan.get('Nav2_execution_verified') is not False
                or plan.get('integrated_stage_portal_bypass_and_continuous_executor_validation_pending') is not True):
            raise ValueError('Own H6 fixed-frame/occupancy MODEL plan required, not physical admission')
        if (plan.get('rigid_body_parameter_bounds',{}).get('max_longitudinal_accel_m_s2')!=.020
                or any(type(a.get('direction')) is not int or a['direction']!=1
                    for a in plan.get('selected_actions',[]))):
            raise ValueError('Only unchanged-acceleration forward H6 references are implemented')
        if any(a.get('mode')!=s.get('mode') for a,s in zip(
                plan.get('selected_actions',[]),plan.get('selected_states',[]))):
            raise ValueError('Native H6 action/state mode mismatch')
        for state in plan.get('selected_states',[]):
            _finite([state['x_m'],state['y_m'],state['yaw_deg']],3,'native H6 local state')
        added=dict(common_max_acceleration_m_s2=.020,footprint=dict(original.FOOTPRINT),
            native_MODEL_success_is_NOT_actual_support_or_drive_admission=True)
        if any(k in plan and plan[k]!=v for k,v in added.items()):
            raise ValueError('Conflicting explicit execution fields must not be replaced')
        legacy_validate(dict(plan,**added))
    Reference=type('MissionForwardPathReference',(original.PathReference,),{
        '__init__':bind(original.PathReference.__init__,validate_plan=validate)})
    config=dict(original.CONTROLLER_CONFIG)
    Controller=type('MissionForwardTrackingController',(original.H4TrackingController,),{
        '__init__':bind(original.H4TrackingController.__init__,PathReference=Reference,CONTROLLER_CONFIG=config)})
    result={k:v for k,v in vars(original).items() if not k.startswith('__')}
    result.update(SCENARIO=scenario,EXECUTION_VERSION='stop_align_v3_H6_forward_schema_MODEL_binding_v1',
        validate_plan=validate,PathReference=Reference,H4TrackingController=Controller,TrackingController=Controller,
        CONTROLLER_CONFIG=config,GUARDS=dict(original.GUARDS,maximum_simulation_duration_s=float(simulation_budget_s)),
        ACCEPTANCE=dict(original.ACCEPTANCE),COMMAND_LIMITS=dict(original.COMMAND_LIMITS),
        original_reference_projection_and_control_method_code_unchanged=True,
        physical_reverse_and_direction_change_policy_NOT_implemented=True,
        current_binding_is_NOT_a_motion_authorization=True)
    return SimpleNamespace(**result)


def mission_nominal_v3_case_binding(geometry, *, simulation_budget_s):
    """Unchanged original MODEL case with exact H6 support union/task binding.

    No fitted response, caster reset, physical wheel-hold or contact model is
    added. A MODEL case never reads a live articulation or timeline. Support
    subtraction retains the verified 1.630 x .965m +10mm full rectangle.
    """
    import copy
    from types import SimpleNamespace
    import screen_hospital_h4_common_executor as original
    from hospital_mission_geometry import disjoint_support_rectangles
    execution=forward_common_v3_execution_binding(geometry,simulation_budget_s=simulation_budget_s)
    design=copy.deepcopy(geometry);polygons=design['same_three_authored_support_rectangles']
    if len(polygons)!=3:raise ValueError('All three original authored support rectangles required')
    disjoint_support_rectangles(polygons)  # rejects crossed/nonfinite/nonrectangle geometry
    def covers(candidate):
        if candidate!=design:raise ValueError('H6 support geometry changed within the MODEL case')
        return copy.deepcopy(polygons)
    cohort=SimpleNamespace(support_covers=covers,
        support_area_missing=original.bundle.cohort.support_area_missing)
    bundle=SimpleNamespace(cohort=cohort)
    run=FunctionType(original.case.__code__,dict(original.case.__globals__,execution=execution,bundle=bundle),
        original.case.__name__,original.case.__defaults__,original.case.__closure__)
    run.__kwdefaults__=None if original.case.__kwdefaults__ is None else dict(original.case.__kwdefaults__)
    return run,execution,design


def export_common_v3_mission_model_screen(planning_review_path,preparation_protocol_path,*,
                                        simulation_budget_s,output):
    """Append-only FOUR-method, two-resolution offline H6 MODEL diagnostic.

    The declared duration here is explicitly prospective MODEL-only, not an
    adopted H6 execution contract. Every HOLD, failed guard and raw trace is
    retained. Existing H4/H5/controller/planning/preparation files are read-only.
    """
    import csv,hashlib,json,sys,zipfile
    from pathlib import Path
    import numpy as np
    import screen_hospital_h4_common_executor as original
    from hospital_runtime_evidence_io import write_exclusive
    from hospital_h4_subfloor_compact_path_launch import normalize_runtime_evidence
    if any(name in sys.modules for name in ('isaacsim','omni.kit.app','omni.physx')):
        raise RuntimeError('Mission MODEL exporter is offline-only; no Kit entry')
    planning_review_path=Path(planning_review_path).resolve()
    preparation_protocol_path=Path(preparation_protocol_path).resolve()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    review=json.loads(planning_review_path.read_bytes())
    prep=json.loads(preparation_protocol_path.read_bytes())
    if (prep.get('scenario')!='H6_integrated_transfer_mission' or prep.get('preparation_only') is not True
            or (prep.get('maximum_static_trials'),prep.get('maximum_route_trials'))!=(0,0)
            or review.get('actual_motion_authorized') is not False
            or review.get('use_for_final_statistics') is not False):
        raise ValueError('Explicit frozen H6 planning and prepare-only provenance required')
    contract_path=Path(review['protocol']);contract=json.loads(contract_path.read_bytes())
    geometry_path=Path(contract['common_geometry']);geometry=json.loads(geometry_path.read_bytes())
    plans=[Path(row['plan_file']) for row in review['methods']]
    if ([row['method'] for row in review['methods']]!=list(original.execution.METHODS)
            or not all(row['accepted_offline_start_goal_plan'] is True for row in review['methods'])
            or sha(geometry_path)!=contract['common_geometry_sha256']
            or prep['start_xyyaw']!=geometry['start_xyyaw'] or prep['goal_xyyaw']!=geometry['goal_xyyaw']):
        raise ValueError('All four independently accepted same-fixed-task references required')
    pins={}
    def add(collection):
        for name,digest in collection.items():
            if name in pins and pins[name]!=digest:raise ValueError('Contradictory MODEL source pin: '+name)
            if sha(name)!=digest:raise ValueError('Changed frozen MODEL input: '+name)
            pins[name]=digest
    add(prep['input_sha256']);add(review['output_sha256'])
    current=[planning_review_path,preparation_protocol_path,geometry_path,Path(__file__).resolve(),
        Path(__file__).resolve().parents[3]/'src/patient_initial_stability/test/test_hospital_mission_planning.py']
    add({str(p):sha(p) for p in current})
    native=json.loads(contract_path.read_bytes())['rigid_body_parameter_bounds']
    for plan_path in plans:
        plan=json.loads(plan_path.read_bytes())
        if plan['geometry_manifest_sha256']!=sha(geometry_path) or plan['rigid_body_parameter_bounds']!=native:
            raise ValueError('Same exact H6 geometry and NE parameters required for every method')
    run,execution,design=mission_nominal_v3_case_binding(geometry,simulation_budget_s=simulation_budget_s)
    for plan_path in plans:execution.validate_plan(json.loads(plan_path.read_bytes()))
    grid_path=planning_review_path.parent/'planning_occupancy.npz'
    if str(grid_path) not in pins:raise ValueError('Exact frozen planning raster required')
    with np.load(grid_path) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    if (occupied.ndim!=2 or occupied.dtype!=bool or origin.shape!=(2,)
            or not np.isfinite(origin).all() or resolution!=.010):
        raise ValueError('Original finite boolean H6 10mm raster required without coarsening')
    definition=dict(version='H6_ORIGINAL_V3_FORWARD_NOMINAL_MODEL_DIAGNOSTIC_v1',
        planning_review=str(planning_review_path),preparation_protocol=str(preparation_protocol_path),
        same_approved_common_geometry=str(geometry_path),methods=list(original.execution.METHODS),
        integration_steps_s=list(original.TIMESTEPS_S),
        common_prospective_MODEL_simulation_limit_s=execution.GUARDS['maximum_simulation_duration_s'],
        this_limit_is_NOT_an_adopted_live_H6_execution_policy=True,
        exact_original_MODEL_case_and_v3_control_and_reference_method_code_unchanged=True,
        full_footprint_original_authored_support_union_and_raster_checks_retained=True,
        physical_reverse_and_direction_change_execution_policy_NOT_verified=True,
        nominal_initial_zero_MODEL_velocity_and_steering_NOT_actual_caster_or_retained_state=True,
        physical_wheel_hold_caster_stiction_and_cooked_contacts_NOT_simulated=True,
        software_resolutions_NOT_independent_repetitions_or_robustness_or_physical_model_validation=True,
        actual_motion_authorized=False,new_PhysX_trial_count=0,use_for_final_statistics=False,
        input_sha256=pins)
    output=Path(output).resolve();target=output/'common_executor_model_screen.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()):raise ValueError('Existing MODEL definition differs; no overwrite')
        for name,digest in saved['output_sha256'].items():
            if sha(name)!=digest:raise ValueError('Existing MODEL artifact changed')
        return target
    if output.exists() and any(output.iterdir()):raise ValueError('Unfinalized MODEL diagnostic retained; no automatic retry')
    output.mkdir(parents=True,exist_ok=True)
    reservation=output/'model_definition.json';write_exclusive(reservation,definition)
    artifacts=[reservation];archive_path=output/'analysis_sources.zip'
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        root=Path(__file__).resolve().parents[3]
        for path in current[-2:]:archive.write(path,str(path.relative_to(root)))
    artifacts.append(archive_path);cases=[]
    for plan_path in plans:
        plan=json.loads(plan_path.read_bytes());execution.validate_plan(plan)
        for dt in original.TIMESTEPS_S:
            print('[H6 NOMINAL MODEL]',plan['method'],'dt=',dt,'; original v3, no Isaac',flush=True)
            report,rows=run(plan,native,dt,occupied,origin,resolution,design)
            report.update(scenario=geometry['scenario'],plan_path=str(plan_path),plan_sha256=sha(plan_path),
                actual_configured_MODEL_limit_s=float(simulation_budget_s),
                inherited_error_text_may_name_original_180s_limit=bool(report['error'] and '180s' in report['error']),
                actual_motion_authorized=False,new_PhysX_trial_count=0)
            trace=output/(plan['method']+'_%dms.csv'%round(dt*1000))
            with trace.open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            artifacts.append(trace);report.update(model_trace=str(trace),model_trace_sha256=sha(trace))
            conversions=[];report=normalize_runtime_evidence(report,conversions)
            report['representation_only_scalar_conversions']=conversions
            case_path=trace.with_suffix('.json');write_exclusive(case_path,report)
            artifacts.append(case_path);cases.append(report)
            print('[H6 NOMINAL MODEL]',plan['method'],dt,report['model_status'],
                's=',report['simulation_duration_s'],'goal_mm=',1000*report['final_position_error_m'],
                'error=',report['error'],flush=True)
    add(pins.copy())
    report=dict(**definition,status='NOMINAL_MODEL_DIAGNOSTIC_RECORDED_NOT_ACTUAL_H6_ADMISSION',cases=cases,
        case_count=len(cases),MODEL_PASS_count=sum(r['model_status']=='MODEL_SCREEN_PASS' for r in cases),
        live_scene_body_joint_or_timeline_accessed=False,old_FAIL_HOLDs_preserved=True,
        output_sha256={str(path):sha(path) for path in artifacts})
    write_exclusive(target,report)
    return target


def review_saved_mission_model_trace_boundaries(folder):
    """Read existing traces ONLY; never rerun an interrupted MODEL exporter.

    For an export interrupted after CSV creation, unavailable force/hold
    summary fields are NOT reconstructed or invented. Recompute geometric
    witnesses directly from final recorded body poses on the frozen raster.
    Resolve only this shared helper/test's historical source from their exact
    executed ZIP; every other immutable input must still match its real file.
    """
    import csv,hashlib,json,zipfile
    from pathlib import Path
    import numpy as np
    from hospital_geometry_tools import footprint_polygon,polygon_cells,pose_collision
    from hospital_runtime_evidence_io import write_exclusive
    folder=Path(folder).resolve();definition_path=folder/'model_definition.json'
    definition=json.loads(definition_path.read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if definition.get('version')!='H6_ORIGINAL_V3_FORWARD_NOMINAL_MODEL_DIAGNOSTIC_v1':
        raise ValueError('Explicit completed-trace H6 MODEL definition required')
    root=Path(__file__).resolve().parents[3]
    own=(Path(__file__).resolve(),root/'src/patient_initial_stability/test/test_hospital_mission_planning.py')
    source_archive=folder/'analysis_sources.zip';historical={}
    with zipfile.ZipFile(source_archive) as archive:
        if sorted(archive.namelist())!=sorted(str(p.relative_to(root)) for p in own):
            raise ValueError('Exact two-file executed analysis source archive required')
        for path in own:
            digest=hashlib.sha256(archive.read(str(path.relative_to(root)))).hexdigest()
            if digest!=definition['input_sha256'][str(path)]:raise ValueError('Executed analysis source archive digest mismatch')
            historical[str(path)]=dict(executed_sha256=digest,current_sha256=sha(path),
                historical_bytes_verified_from=str(source_archive))
    for name,digest in definition['input_sha256'].items():
        if name not in historical and sha(name)!=digest:raise ValueError('Frozen MODEL input changed: '+name)
    review_path=Path(definition['planning_review']);review=json.loads(review_path.read_bytes())
    plans=[Path(row['plan_file']) for row in review['methods']]
    geometry_path=Path(definition['same_approved_common_geometry']);geometry=json.loads(geometry_path.read_bytes())
    execution=forward_common_v3_execution_binding(geometry,
        simulation_budget_s=definition['common_prospective_MODEL_simulation_limit_s'])
    grid_path=review_path.parent/'planning_occupancy.npz'
    with np.load(grid_path) as grid:
        masks={key:grid[key].copy() for key in grid.files if grid[key].shape==grid['occupied'].shape and grid[key].dtype==bool}
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    # Reuse the original full-body support-union evaluator, not a center test.
    import screen_hospital_h4_common_executor as original
    support_area=original.bundle.cohort.support_area_missing
    covers=geometry['same_three_authored_support_rectangles'];cases=[];artifacts={}
    for plan_path in plans:
        plan=json.loads(plan_path.read_bytes());reference=execution.PathReference(plan)
        for dt in definition['integration_steps_s']:
            trace=folder/(plan['method']+'_%dms.csv'%round(1000*dt))
            with trace.open(newline='') as stream:rows=list(csv.DictReader(stream))
            if len(rows)<2:raise ValueError('At least two existing MODEL trace rows required')
            numeric=[{k:float(row[k]) for k in ('time_s','x_m','y_m','yaw_rad','reference_progress_m',
                'goal_position_error_m','goal_heading_error_deg')} for row in rows]
            if any(not all(math.isfinite(v) for v in row.values()) for row in numeric):raise ValueError('Nonfinite saved MODEL trace')
            if any(b['time_s']<=a['time_s'] for a,b in zip(numeric,numeric[1:])):raise ValueError('Nonmonotone MODEL trace times')
            last=numeric[-1];pose=[last[k] for k in ('x_m','y_m','yaw_rad')]
            final_error=execution.relative_pose(plan['goal_world_xyyaw'],pose)
            if (abs(math.hypot(*final_error[:2])-last['goal_position_error_m'])>1e-10
                    or abs(math.degrees(final_error[2])-last['goal_heading_error_deg'])>1e-9):
                raise ValueError('Saved MODEL pose and error fields disagree')
            index,expected=reference.at(last['reference_progress_m'])
            relative=execution.relative_pose(pose,expected)
            rr,cc=polygon_cells(footprint_polygon(pose,clearance=.010),origin,resolution,occupied.shape)
            hits=occupied[rr,cc];hit_rows,hit_columns=rr[hits],cc[hits]
            categories={name:int(np.count_nonzero(mask[hit_rows,hit_columns])) for name,mask in masks.items()
                if name!='occupied' and np.any(mask[hit_rows,hit_columns])}
            witness=[dict(row=int(r),column=int(c),center_world_xy_m=[float(origin[0]+(c+.5)*resolution),
                float(origin[1]+(r+.5)*resolution)]) for r,c in list(zip(hit_rows,hit_columns))[:4]]
            cases.append(dict(method=plan['method'],integration_dt_s=dt,saved_trace=str(trace),
                trace_sha256=sha(trace),sample_count=len(rows),final_recorded_time_s=last['time_s'],
                final_recorded_phase=rows[-1]['phase'],final_recorded_world_xyyaw=pose,
                final_recorded_progress_m=last['reference_progress_m'],reference_polyline_m=reference.arc[-1],
                final_reference_action_index=index,reference_pose_at_recorded_progress=list(expected),
                body_vs_reference_position_error_m=math.hypot(*relative[:2]),
                body_vs_reference_heading_error_deg=math.degrees(relative[2]),
                final_body_pose_intersects_frozen_full_envelope_raster=pose_collision(occupied,pose,origin,resolution,clearance=.010),
                reference_pose_at_same_progress_intersects_raster=pose_collision(occupied,expected,origin,resolution,clearance=.010),
                body_support_uncovered_area_m2=support_area(pose,covers),
                occupied_touched_cell_count=int(len(hit_rows)),first_four_occupied_cell_witnesses=witness,
                saved_raster_component_membership=categories,
                final_recorded_goal_position_error_m=last['goal_position_error_m'],
                final_recorded_goal_heading_error_deg=last['goal_heading_error_deg'],
                this_is_a_geometric_MODEL_trace_witness_NOT_PhysX_contact=True,
                unavailable_original_force_utilization_and_hold_summary_NOT_invented=True))
            artifacts[str(trace)]=sha(trace)
    artifacts.update({str(p):sha(p) for p in (definition_path,source_archive,Path(__file__).resolve(),own[1])})
    report=dict(version='H6_SAVED_NOMINAL_MODEL_TRACE_BOUNDARY_REVIEW_v1',
        status='SAVED_TRACE_GEOMETRIC_WITNESSES_RECORDED_NOT_A_NEW_MODEL_OR_PHYSICAL_TRIAL',
        original_model_definition=str(definition_path),
        original_aggregate_model_report_exists=(folder/'common_executor_model_screen.json').exists(),
        original_interrupted_aggregate_metrics_NOT_reconstructed=True,
        all_eight_expected_traces_read=len(cases)==8,cases=cases,
        recorded_body_raster_witness_count=sum(r['final_body_pose_intersects_frozen_full_envelope_raster'] for r in cases),
        reference_same_progress_raster_witness_count=sum(r['reference_pose_at_same_progress_intersects_raster'] for r in cases),
        all_final_body_footprints_inside_original_authored_support=all(r['body_support_uncovered_area_m2']<=1e-9 for r in cases),
        executed_analysis_source_history=historical,verified_input_sha256=dict(definition['input_sha256']),
        review_input_sha256=artifacts,no_original_artifact_modified=True,
        new_MODEL_simulations=0,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    target=folder/'saved_trace_boundary_review.json'
    if target.exists():
        if json.loads(target.read_bytes())!=report:raise ValueError('Existing saved-trace review differs; no overwrite')
    else:write_exclusive(target,report)
    return target


def model_clearance_pose_comparison(occupied,origin,resolution,actual,reference):
    """Fixed-raster SAT counterfactuals, NOT a controller/contact experiment.

    Keep original physical dimensions and required10mm margin. A zero-margin
    comparison distinguishes the expanded safety outline from the raw body,
    but never admits a drive or waives the adopted10mm criterion.
    """
    from hospital_geometry_tools import pose_collision
    actual=_finite(actual,3,'recorded MODEL body pose');reference=_finite(reference,3,'reference pose')
    result={}
    for role,pose in (('recorded_body',actual),('same_progress_reference',reference),
                      ('translation_only',(*actual[:2],reference[2])),
                      ('heading_only',(*reference[:2],actual[2]))):
        zero=pose_collision(occupied,pose,origin,resolution,clearance=0.)
        required=pose_collision(occupied,pose,origin,resolution,clearance=.010)
        lo,hi=0.,.100
        if zero:lo=hi=0.
        elif pose_collision(occupied,pose,origin,resolution,clearance=hi):
            for _ in range(16):
                middle=(lo+hi)/2
                if pose_collision(occupied,pose,origin,resolution,clearance=middle):hi=middle
                else:lo=middle
        else:lo=hi=.100
        result[role]=dict(raw_body_intersects_recorded_raster=zero,
            required_10mm_expanded_outline_intersects_recorded_raster=required,
            sampled_SAT_expansion_free_lower_bound_m=lo,
            sampled_SAT_expansion_collision_upper_bound_m=hi,
            search_cap_m=.100,bound_is_raster_expansion_NOT_true_mesh_or_cooked_clearance=True)
    return result


def export_saved_mission_clearance_witness_review(folder):
    """Use only already-saved poses; no additional nominal MODEL simulation."""
    import hashlib,json,zipfile
    from pathlib import Path
    import numpy as np
    from hospital_geometry_tools import footprint_polygon,polygon_cells
    from hospital_runtime_evidence_io import write_exclusive
    folder=Path(folder).resolve();boundary_path=folder/'saved_trace_boundary_review.json'
    report=json.loads(boundary_path.read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    root=Path(__file__).resolve().parents[3];own_names={str(root/rel):rel for rel in (
        'studies/IJCAS/scripts/hospital_mission_planning.py',
        'src/patient_initial_stability/test/test_hospital_mission_planning.py')}
    with zipfile.ZipFile(folder/'trace_review_sources.zip') as archive:
        for name,digest in report['review_input_sha256'].items():
            value=hashlib.sha256(archive.read(own_names[name])).hexdigest() if name in own_names else sha(name)
            if value!=digest:raise ValueError('Saved-trace review provenance changed')
    definition=json.loads(Path(report['original_model_definition']).read_bytes())
    geometry_path=Path(definition['same_approved_common_geometry']);geometry=json.loads(geometry_path.read_bytes())
    frozen=Path(definition['planning_review']).parent/'planning_occupancy.npz'
    component_grid=Path(geometry['occupancy_grid'])
    if (sha(geometry_path)!=definition['input_sha256'][str(geometry_path)]
            or sha(component_grid)!=geometry['output_sha256'][str(component_grid)]):
        raise ValueError('Exact original H6 geometry/component raster required')
    with np.load(component_grid) as grid,np.load(frozen) as original:
        if any(not np.array_equal(grid[k],original[k]) for k in ('occupied','origin_xy_m','resolution_m')):
            raise ValueError('Component attribution must use the exact same frozen MODEL raster')
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
        components={k:grid[k].copy() for k in grid.files if k!='occupied' and grid[k].shape==occupied.shape and grid[k].dtype==bool}
    cases=[]
    for row in report['cases']:
        actual=row['final_recorded_world_xyyaw'];reference=row['reference_pose_at_recorded_progress']
        rr,cc=polygon_cells(footprint_polygon(actual,clearance=.010),origin,resolution,occupied.shape)
        hit=occupied[rr,cc];rr,cc=rr[hit],cc[hit]
        categories={k:int(np.count_nonzero(mask[rr,cc])) for k,mask in components.items() if np.any(mask[rr,cc])}
        cases.append(dict(method=row['method'],integration_dt_s=row['integration_dt_s'],
            recorded_time_s=row['final_recorded_time_s'],recorded_progress_m=row['final_recorded_progress_m'],
            body_vs_reference_position_error_m=row['body_vs_reference_position_error_m'],
            body_vs_reference_heading_error_deg=row['body_vs_reference_heading_error_deg'],
            exact_common_grid_component_membership_at_expanded_body_witness=categories,
            pose_counterfactuals=model_clearance_pose_comparison(occupied,origin,resolution,actual,reference)))
    inputs=[boundary_path,geometry_path,component_grid,frozen,folder/'trace_review_sources.zip',
        Path(__file__).resolve(),root/'src/patient_initial_stability/test/test_hospital_mission_planning.py']
    result=dict(version='H6_SAVED_MODEL_POSE_EXPANDED_OUTLINE_WITNESS_REVIEW_v1',
        status='SAME_SAVED_POSES_RASTER_SAFETY_MARGIN_AND_COUNTERFACTUALS_ONLY',cases=cases,
        component_raster_bitwise_equal_frozen_simulated_raster=True,
        original_10mm_margin_retained_zero_margin_IS_NOT_waiver=True,
        translation_and_heading_counterfactuals_NOT_causal_identification_of_controller_or_physics=True,
        actual_contact_or_cooked_clearance_or_sensor_validation_NOT_claimed=True,
        new_MODEL_simulations=0,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={str(path):sha(path) for path in inputs})
    target=folder/'clearance_witness_review.json'
    if target.exists():
        if json.loads(target.read_bytes())!=result:raise ValueError('Existing clearance witness review differs; no overwrite')
    else:write_exclusive(target,result)
    return target


def _finite(values, count, name):
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError('Finite '+name+' required')
    return tuple(map(float, values))


def mission_search_bounds(roi_xy, start_xyyaw, goal_xyyaw):
    """Local AABB of the ENTIRE fixed ROI; original SAT still bounds edges."""
    roi = _finite(roi_xy, 4, 'fixed ROI')
    start = _finite(start_xyyaw, 3, 'fixed start')
    goal = _finite(goal_xyyaw, 3, 'fixed goal')
    if not roi[0] < roi[1] or not roi[2] < roi[3]:
        raise ValueError('Positive fixed ROI required')
    if any(not (roi[0] <= p[0] <= roi[1] and roi[2] <= p[1] <= roi[3]) for p in (start, goal)):
        raise ValueError('Both fixed task coordinates must lie in reviewed ROI')
    c, s = math.cos(start[2]), math.sin(start[2])
    def local(x,y):
        dx, dy = x-start[0], y-start[1]
        return c*dx+s*dy, -s*dx+c*dy
    corners = [local(x,y) for x in roi[:2] for y in roi[2:]]
    gx, gy = local(*goal[:2])
    yaw = math.atan2(math.sin(goal[2]-start[2]), math.cos(goal[2]-start[2]))
    return dict(minimum_x=min(x for x,y in corners), maximum_x=max(x for x,y in corners),
        minimum_y=min(y for x,y in corners), maximum_y=max(y for x,y in corners),
        local_goal_xyyaw=[gx,gy,yaw], entire_original_ROI_retained=True,
        local_AABB_is_NOT_free_space_or_a_collision_certificate=True)


def common_v3_terminal_geometry_review(plan):
    """Read-only endpoint coupling screen using the EXISTING path reference.

    Alias only the legacy reference reader's H1 schema label in a new dict;
    do not modify a stored H6 plan or run/admit its controller. Sample the
    existing25mm path-end handoff and the endpoint. Neither an ideal pose in
    the capture band nor a model endpoint is an actual stopped5s hold PASS.
    """
    from hospital_path_execution import PathReference,relative_pose
    from hospital_path_execution_v3 import CONTROLLER_CONFIG
    if plan.get('scenario')!='H6_integrated_transfer_mission':
        raise ValueError('Explicit H6 model plan required; no task relabel or runtime admission')
    reference=PathReference({**plan,'scenario':'H1_bedside_departure'})
    if reference.arc[-1]<=.025:raise ValueError('H6 whole route must exceed the existing25mm handoff distance')
    result=[]
    for label,arc in (('existing_25mm_reference_handoff',reference.arc[-1]-.025),
                       ('reference_endpoint',reference.arc[-1])):
        index,pose=reference.at(arc);error=relative_pose(plan['goal_world_xyyaw'],pose)
        position=math.hypot(*error[:2]);heading=math.degrees(error[2])
        result.append(dict(role=label,reference_arc_m=arc,reference_world_xyyaw=list(pose),
            reference_action_index=index,goal_position_error_m=position,goal_heading_error_deg=heading,
            ideal_reference_pose_inside_existing_stopped_capture_band=(
                position<=CONTROLLER_CONFIG['stationary_hold_entry_position_m']
                and abs(heading)<=CONTROLLER_CONFIG['stationary_hold_entry_heading_deg'])))
    return dict(method=plan['method'],reference_polyline_m=reference.arc[-1],samples=result,
        existing_v3_config=dict(CONTROLLER_CONFIG),reference_schema_alias_ONLY_no_saved_plan_edit=True,
        braking_tracking_alignment_pulses_and_actual_hold_NOT_simulated=True,
        out_of_band_handoff_NOT_automatic_FAIL_existing_bounded_terminal_closure_still_required=True,
        in_band_model_endpoint_NOT_actual_hold_PASS=True,
        no_controller_change_or_timed_executor_substitution=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def bounded_lattice_function(original, bounds, *, directions):
    """Replace only legacy H1 bounds and explicitly declared directions.

    Does not implement or waive reverse-distance, hold or live-resource
    guards. A planning direction set is never physical reverse permission.
    """
    keys = ('minimum_x','maximum_x','minimum_y','maximum_y')
    _finite([bounds[k] for k in keys],4,'mission bounds')
    if not bounds['minimum_x'] < bounds['maximum_x'] or not bounds['minimum_y'] < bounds['maximum_y']:
        raise ValueError('Nondegenerate full mission bounds required')
    directions = tuple(directions)
    if not directions or len(set(directions)) != len(directions) or any(type(v) is not int or v not in (-1,1) for v in directions):
        raise ValueError('Explicit unique signed planning directions required')
    expected = _finite(bounds['local_goal_xyyaw'],3,'fixed local goal')
    def run(goal, legacy_minimum_y, legacy_maximum_y, **kwargs):
        if _finite(goal,3,'local goal') != expected:
            raise ValueError('Mission binding cannot silently move the fixed goal')
        if not callable(kwargs.get('collision_checker')):
            raise ValueError('Original full-envelope map checker required, not bounds-only walls')
        scoped = dict(kwargs, minimum_x=bounds['minimum_x'], maximum_x=bounds['maximum_x'], directions=directions)
        return original(goal,bounds['minimum_y'],bounds['maximum_y'],**scoped)
    return run


def bind_plan_method(original_method, original_lattice, bounds, *, checker_class, guidance, directions):
    """Reuse exact method code in a PRIVATE namespace, never global patches."""
    namespace = dict(original_method.__globals__)
    namespace.update(plan=bounded_lattice_function(original_lattice,bounds,directions=directions),
        HospitalEnvelopeChecker=checker_class,pose_guidance=guidance,
        # H1's specialised same-heading offset connector is not an H6 path.
        # Full original primitives/modes remain; no fixed-route fallback.
        counterphase_connector=lambda goal,checker:None)
    result = FunctionType(original_method.__code__,namespace,original_method.__name__,
        original_method.__defaults__,original_method.__closure__)
    result.__kwdefaults__ = None if original_method.__kwdefaults__ is None else dict(original_method.__kwdefaults__)
    return result


@dataclass(frozen=True)
class MotionResourceLimits:
    """Prospective planning bookkeeping, not a relaxed physical stop guard."""
    maximum_reverse_episode_m: float = .5
    maximum_total_reverse_m: float = 1.
    maximum_direction_changes: int = 4
    distance_quantum_m: float = .005

    def __post_init__(self):
        lengths=(self.maximum_reverse_episode_m,self.maximum_total_reverse_m,self.distance_quantum_m)
        if (not all(math.isfinite(v) and v>0. for v in lengths)
                or self.maximum_reverse_episode_m>self.maximum_total_reverse_m
                or type(self.maximum_direction_changes) is not int or self.maximum_direction_changes<0
                or self.distance_quantum_m>self.maximum_reverse_episode_m):
            raise ValueError('Explicit positive bounded planning resource limits required')


class ResourcePoseSamples(list):
    """Original unmodified swept poses tagged with a SEPARATE resource guard."""
    pass


def resource_propagation_binding(original_state, original_propagate, limits):
    """Carry reverse history without changing any original body/steering field.

    Round every reverse-edge centre arc UP to 5mm accounting bins (1pm
    arithmetic tolerance). The original exact physical limits remain .5/1m.
    Adding history to dominance keys prevents merging distinct resource states.
    """
    if not isinstance(limits,MotionResourceLimits):
        raise ValueError('Explicit resource limits required')
    names=('mission_reverse_episode_bins','mission_reverse_total_bins','mission_direction_changes')
    State=make_dataclass('MissionResourceState',[(name,int,0) for name in names],
        bases=(original_state,),frozen=True)
    State.__module__=__name__
    original_fields=tuple(f.name for f in fields(original_state))
    def propagate(state,mode,curvature,direction,*args,**kwargs):
        if type(direction) is not int or direction not in (-1,1):
            raise ValueError('Exact signed primitive direction required')
        result,poses,elapsed=original_propagate(state,mode,curvature,direction,*args,**kwargs)
        samples=ResourcePoseSamples(poses)
        previous=(state.x,state.y);arc=0.
        for pose in samples:
            if len(pose)!=3 or not all(math.isfinite(v) for v in pose):
                raise ValueError('Complete finite original swept poses required')
            arc+=math.hypot(pose[0]-previous[0],pose[1]-previous[1]);previous=pose[:2]
        old_episode,old_total,old_switches=(getattr(state,name,0) for name in names)
        if any(type(v) is not int or v<0 for v in (old_episode,old_total,old_switches)):
            raise ValueError('Nonnegative integer resource history required')
        bins=max(0,math.ceil((arc-1e-12)/limits.distance_quantum_m)) if direction<0 else 0
        episode=(old_episode if state.direction<0 else 0)+bins if direction<0 else 0
        total=old_total+bins
        changes=old_switches+int(state.started and state.direction!=0 and direction!=state.direction)
        samples.resource_checks=dict(
            reverse_episode=episode*limits.distance_quantum_m<=limits.maximum_reverse_episode_m+1e-12,
            reverse_total=total*limits.distance_quantum_m<=limits.maximum_total_reverse_m+1e-12,
            direction_changes=changes<=limits.maximum_direction_changes)
        samples.resource_feasible=all(samples.resource_checks.values())
        samples.actual_edge_center_arc_m=arc
        values={name:getattr(result,name) for name in original_fields}
        values.update(zip(names,(episode,total,changes)))
        return State(**values),samples,elapsed
    def state_key(original_key):
        def key(state,*args,**kwargs):
            return (*original_key(state,*args,**kwargs),*(getattr(state,name,0) for name in names))
        return key
    return State,propagate,state_key


def resource_checker_class(original_checker):
    """Reject tagged resource overflow, explicitly NOT an obstacle collision."""
    class Checker(original_checker):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.statistics['motion_resource_rejected_edges_NOT_obstacle_collisions']=0
            for name in ('reverse_episode','reverse_total','direction_changes'):
                self.statistics['motion_resource_'+name+'_rejections']=0
        def __call__(self,poses):
            if not getattr(poses,'resource_feasible',True):
                self.check_deadline()
                self.statistics['motion_resource_rejected_edges_NOT_obstacle_collisions']+=1
                for name,passed in poses.resource_checks.items():
                    self.statistics['motion_resource_'+name+'_rejections']+=int(not passed)
                return True
            return super().__call__(poses)
    return Checker


def bind_resource_kernels(original_lattice, original_refinement, original_state,
        original_propagate, original_key, limits):
    """Reuse both original code objects; private history/guard bindings only."""
    State,propagate,key=resource_propagation_binding(original_state,original_propagate,limits)
    def clone(function,**values):
        result=FunctionType(function.__code__,dict(function.__globals__,**values),
            function.__name__,function.__defaults__,function.__closure__)
        result.__kwdefaults__=None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
        return result
    lattice=clone(original_lattice,State=State,propagate=propagate,_state_key=key(original_key))
    refinement=clone(original_refinement,State=State,propagate=propagate)
    return lattice,refinement


def bidirectional_search_guidance(original_guidance):
    """Common heuristic only: inspect forward/reverse VIRTUAL headings.

    Never return the virtual pose to propagation, the controller or the SDK.
    Fixed actual goal/heading/observed state and all collision checks remain.
    """
    def guidance(state,goal):
        wrap=lambda angle:math.atan2(math.sin(angle),math.cos(angle))
        virtual=replace(state,yaw=wrap(state.yaw+math.pi))
        virtual_goal=(goal[0],goal[1],wrap(goal[2]+math.pi))
        return min(original_guidance(state,goal),original_guidance(virtual,virtual_goal))
    return guidance


def with_direction_change_cost(result, penalty_s):
    """Add the baseline's existing reversal cost to a refinement result.

    The frozen forward-only refiner had no direction-switch cost. This is an
    explicit common mission cost overlay, NOT a body/steering model edit.
    """
    if not math.isfinite(penalty_s) or penalty_s<0.:
        raise ValueError('Finite nonnegative common direction-switch cost required')
    if result is None:return None,0
    previous=0;count=0
    for action in result.actions:
        direction=action['direction']
        if type(direction) is not int or direction not in (-1,1):
            raise ValueError('Exact signed action required for common switch cost')
        count+=int(previous!=0 and direction!=previous);previous=direction
    return replace(result,cost_s=result.cost_s+penalty_s*count),count


class ExactRigidBodyStepCache:
    """Bounded exact-input reuse of the ORIGINAL stateless NE step.

    No quantisation, approximate state key, coordinate transform, new plant
    equation or skipped constraint. Seven IEEE-754 input bit patterns are the
    entire key (including signed zero); each miss delegates to the unchanged
    original model. A private frozen copy of explicit bounds owns the cache.
    This is a computation adapter, not a feasibility or motion certificate.
    """
    __slots__=('_bounds','_model','_cached','_bypass')

    def __init__(self,bounds,*,maxsize=65536):
        import struct
        from functools import lru_cache
        from patient_initial_stability.rigid_body_edge import (
            RigidBodyParameterBounds,MassNormalizedRigidBodyEdgeModel)
        if type(bounds) is not RigidBodyParameterBounds or type(maxsize) is not int or maxsize<=0:
            raise ValueError('Original explicit immutable bounds and positive bounded cache size required')
        self._bounds=replace(bounds)
        self._model=MassNormalizedRigidBodyEdgeModel(self._bounds)
        self._bypass=0
        model=self._model
        @lru_cache(maxsize=maxsize)
        def cached(key):
            return model.step(*struct.unpack('!7d',key))
        self._cached=cached

    @property
    def bounds(self):return self._bounds

    def step(self,u,v,r,target_u,target_v,target_r,dt):
        import struct
        # Non-Python-float/nonfinite inputs retain original model behaviour,
        # but never enter a cache whose key promises exact float bit patterns.
        if (type(u) is type(v) is type(r) is type(target_u) is type(target_v) is type(target_r) is type(dt) is float
                and math.isfinite(u) and math.isfinite(v) and math.isfinite(r)
                and math.isfinite(target_u) and math.isfinite(target_v) and math.isfinite(target_r) and math.isfinite(dt)):
            return self._cached(struct.pack('!7d',u,v,r,target_u,target_v,target_r,dt))
        self._bypass+=1
        return self._model.step(u,v,r,target_u,target_v,target_r,dt)

    def cache_info(self):
        values=self._cached.cache_info()
        return dict(version='exact_IEEE754_7input_original_NE_step_cache_v1',
            hits=values.hits,misses=values.misses,maxsize=values.maxsize,currsize=values.currsize,
            original_behaviour_uncached_bypasses=self._bypass,
            input_quantisation_used=False,signed_zero_preserved=True,
            privately_owned_frozen_parameter_copy=True,original_NE_equations_and_returns_unchanged=True)


def _cpython311_hypot2(first,second):
    """Two-coordinate CPython 3.11 vector_norm operation order, no FMA.

    Numerical attribution: Python Software Foundation, CPython v3.11.14,
    Modules/mathmodule.c vector_norm, PSF License; see
    https://github.com/python/cpython/blob/v3.11.14/Modules/mathmodule.c
    Kept here because Numba/libm hypot differs from Python math.hypot by
    one ULP for some original wrench inputs. This is numeric compatibility,
    not a change to the original wrench equation or feasibility threshold.
    """
    values=(abs(first),abs(second));maximum=0.
    for value in values:
        if value>maximum:maximum=value
    if math.isinf(maximum):return maximum
    if math.isnan(first) or math.isnan(second):return math.nan
    if maximum==0.:return maximum
    _,exponent=math.frexp(maximum)
    csum=1.;frac1=frac2=frac3=0.
    if exponent>=-1023:
        scale=math.ldexp(1.,-exponent)
        for value in values:
            value*=scale;t=value*134217729.;hi=t-(t-value);lo=value-hi
            value=hi*hi;old=csum;csum+=value;frac1+=(old-csum)+value
            value=2.*hi*lo;old=csum;csum+=value;frac2+=(old-csum)+value
            frac3+=lo*lo
        h=math.sqrt(csum-1.+(frac1+frac2+frac3))
        t=h*134217729.;hi=t-(t-h);lo=h-hi
        value=-hi*hi;old=csum;csum+=value;frac1+=(old-csum)+value
        value=-2.*hi*lo;old=csum;csum+=value;frac2+=(old-csum)+value
        value=-lo*lo;old=csum;csum+=value;frac3+=(old-csum)+value
        value=csum-1.+(frac1+frac2+frac3)
        return (h+value/(2.*h))/scale
    for value in values:
        value/=maximum;value=value*value;old=csum;csum+=value;frac1+=(old-csum)+value
    return maximum*math.sqrt(csum-1.+frac1)


def compiled_original_NE_propagation():
    """EXPERIMENTAL private compilation of the original complete NE edge.

    Derive the numerical bodies from the original Python AST, not a second
    hand-written plant. Changes are representation-only: immutable tuples
    replace dataclasses/model objects, original map values are explicit
    scalars, and the original NE method is an in-process compiled call.
    libm pow and ordered min/max preserve Python's finite-float operation
    semantics. Unsupported inputs delegate to the original function. No
    fast-math, reduced precision, skipped poses/guards, global patch, disk
    compilation cache or runtime adoption. Equivalence MUST be verified by
    the caller before any planning benchmark; this factory grants no motion.
    """
    import ast
    import ctypes
    import hashlib
    import inspect
    import textwrap
    from collections import namedtuple
    from numba import njit
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability import rigid_body_edge as rigid

    original=lattice.propagate
    State=namedtuple('CompiledOriginalState',[f.name for f in fields(lattice.State)])
    Dynamics=namedtuple('CompiledOriginalDynamics',[f.name for f in fields(lattice.PlannerDynamics)])
    Bounds=namedtuple('CompiledOriginalBounds',[f.name for f in fields(rigid.RigidBodyParameterBounds)])
    Step=namedtuple('CompiledOriginalStep',[f.name for f in fields(rigid.RigidBodyStep)])
    Model=namedtuple('CompiledOriginalModel','bounds support_length support_arms com_heights yaw_radius_squared')
    options=dict(fastmath=False,error_model='python',cache=False)
    # Python float ** delegates to libm pow. An opaque typed call prevents
    # LLVM from replacing **2 by a differently-rounded multiplication.
    library=ctypes.CDLL(None);power=library.pow
    power.argtypes=(ctypes.c_double,ctypes.c_double);power.restype=ctypes.c_double
    def maximum2(a,b):return b if b>a else a
    def minimum2(a,b):return b if b<a else a
    maximum2=njit(**options)(maximum2);minimum2=njit(**options)(minimum2)
    def maximum3(a,b,c):return maximum2(maximum2(a,b),c)
    def minimum3(a,b,c):return minimum2(minimum2(a,b),c)
    namespace=dict(math=math,GRAVITY_M_S2=rigid.GRAVITY_M_S2,RigidBodyStep=Step,
        State=State,_libm_pow=power,_ordered_max2=maximum2,_ordered_min2=minimum2,
        _ordered_max3=njit(**options)(maximum3),_ordered_min3=njit(**options)(minimum3),
        _original_hypot2=njit(**options)(_cpython311_hypot2))
    # Keep scalar CPython/libm trig calls opaque as well. LLVM may otherwise
    # combine sin/cos into sincos with a different last bit on some inputs.
    for name in ('sin','cos','tan','atan'):
        function=getattr(library,name);function.argtypes=(ctypes.c_double,);function.restype=ctypes.c_double
        namespace['_libm_'+name]=function
    sources={};transforms={}
    class Representation(ast.NodeTransformer):
        def visit_BinOp(self,node):
            self.generic_visit(node)
            if isinstance(node.op,ast.Pow):
                return ast.copy_location(ast.Call(ast.Name('_libm_pow',ast.Load()),[node.left,node.right],[]),node)
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id=='math' and node.func.attr=='hypot'):
                if len(node.args)!=2 or node.keywords:raise ValueError('Original two-coordinate norm required')
                transforms['original_hypot2_calls']=transforms.get('original_hypot2_calls',0)+1
                node.func=ast.Name('_original_hypot2',ast.Load())
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id=='math' and node.func.attr in ('sin','cos','tan','atan')):
                node.func=ast.Name('_libm_'+node.func.attr,ast.Load())
            if isinstance(node.func,ast.Name) and node.func.id in ('min','max'):
                if node.keywords or len(node.args) not in (2,3):raise ValueError('Reviewed ordered min/max arity required')
                node.func.id='_ordered_'+node.func.id+str(len(node.args))
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id=='rigid_body_model' and node.func.attr=='step'):
                transforms['NE_method_calls']=transforms.get('NE_method_calls',0)+1
                node.func=ast.Name('_compiled_NE_step',ast.Load());node.args.insert(0,ast.Name('rigid_body_model',ast.Load()))
            return node
        def visit_Attribute(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='self':
                node.attr={'_support_length':'support_length','_support_arms':'support_arms',
                    '_com_heights':'com_heights','_yaw_radius_squared':'yaw_radius_squared'}.get(node.attr,node.attr)
            return node
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='response_scales' for t in node.targets):
                transforms['response_map_assignment']=transforms.get('response_map_assignment',0)+1
                node.value=ast.Constant(None)
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name):
                name=node.value.id
                if name in ('response_scales','MODE_LATERAL_RESPONSE_SCALE'):
                    transforms[name]=transforms.get(name,0)+1
                    return ast.copy_location(ast.Name('compiled_response_scale' if name=='response_scales'
                        else 'compiled_lateral_scale',ast.Load()),node)
                if name=='STEERING_TRACKING_BIAS_RAD':
                    transforms[name]=transforms.get(name,0)+1
                    return ast.copy_location(ast.IfExp(ast.Compare(ast.Name('axle',ast.Load()),[ast.Eq()],
                        [ast.Constant('front')]),ast.Constant(lattice.STEERING_TRACKING_BIAS_RAD['front']),
                        ast.Constant(lattice.STEERING_TRACKING_BIAS_RAD['rear'])),node)
            return node
    def build(function,*,propagation=False):
        source=textwrap.dedent(inspect.getsource(function));sources[function.__qualname__]=hashlib.sha256(source.encode()).hexdigest()
        tree=ast.parse(source);definition=tree.body[0]
        if not isinstance(definition,ast.FunctionDef) or definition.decorator_list:
            raise ValueError('Undecorated original numeric function required')
        tree=Representation().visit(tree)
        if propagation:
            for name in ('compiled_response_scale','compiled_lateral_scale'):
                definition.args.args.append(ast.arg(name));definition.args.defaults.append(ast.Constant(None))
        ast.fix_missing_locations(tree)
        exec(compile(tree,'<private_original_NE_AST>', 'exec'),namespace)
        return njit(**options)(namespace[definition.name])
    constants={name:getattr(lattice,name) for name in
        ('WHEELBASE_M','FRONT_LENGTH_M','REAR_LENGTH_M','SUBSTEPS','STEERING_RATE_RAD_S',
         'STEP_M','SPEED_M_S','FULL_DYNAMICS')}
    namespace.update(constants)
    for name,function in (('_clamp',rigid._clamp),('_toward',rigid._toward),
            ('move_toward',lattice.move_toward),('steering_targets',lattice.steering_targets),
            ('_tracked_steering_target',lattice._tracked_steering_target)):
        namespace[name]=build(function)
    namespace['_compiled_NE_step']=build(rigid.MassNormalizedRigidBodyEdgeModel.step)
    numerical=build(original,propagation=True)
    if transforms!={'STEERING_TRACKING_BIAS_RAD':1,'NE_method_calls':2,'original_hypot2_calls':2,
            'response_map_assignment':1,'response_scales':1,'MODE_LATERAL_RESPONSE_SCALE':1}:
        raise ValueError('Original representation anchors changed; no unreviewed compilation')
    original_signature=inspect.signature(original)
    bias=dict(lattice.STEERING_TRACKING_BIAS_RAD)
    model_fields=tuple(f.name for f in fields(rigid.RigidBodyParameterBounds))
    state_fields=tuple(f.name for f in fields(lattice.State))
    float_state_fields=tuple(name for name in state_fields if name not in
        ('mode','started','direction','rigid_body_feasible'))
    counts=dict(compiled_calls=0,original_fallback_calls=0)
    def propagate(*args,**kwargs):
        bound=original_signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        state=v['state'];model=v['rigid_body_model'];dynamics=v['dynamics'];mode=v['mode']
        supported=(type(model) is rigid.MassNormalizedRigidBodyEdgeModel
            and type(model.bounds) is rigid.RigidBodyParameterBounds
            and isinstance(state,lattice.State) and type(dynamics) is lattice.PlannerDynamics
            and mode in lattice.ALL_MODES and type(v['direction']) is int and v['direction'] in (-1,1)
            and type(state.started) is bool and type(state.rigid_body_feasible) is bool
            and type(state.direction) is int and lattice.STEERING_TRACKING_BIAS_RAD==bias
            and all(getattr(lattice,k)==value for k,value in constants.items())
            and all(type(getattr(state,k)) is float and math.isfinite(getattr(state,k))
                and abs(getattr(state,k))<=1000. for k in float_state_fields)
            and all(type(getattr(model.bounds,k)) is float and math.isfinite(getattr(model.bounds,k))
                and abs(getattr(model.bounds,k))<=1000. for k in model_fields)
            and all(type(v[k]) is float and math.isfinite(v[k]) and abs(v[k])<=1000.
                for k in ('curvature','step_m','speed_m_s'))
            and (v['target_speed_m_s'] is None or type(v['target_speed_m_s']) is float
                and math.isfinite(v['target_speed_m_s']) and abs(v['target_speed_m_s'])<=1000.)
            and v['response_scale_by_mode'] is None)
        if not supported:
            counts['original_fallback_calls']+=1;return original(*args,**kwargs)
        parameters=Model(Bounds(*(getattr(model.bounds,k) for k in model_fields)),model._support_length,
            model._support_arms,model._com_heights,model._yaw_radius_squared)
        result,poses,elapsed=numerical(State(*(getattr(state,k) for k in state_fields)),mode,
            v['curvature'],v['direction'],v['step_m'],v['speed_m_s'],
            Dynamics(*(getattr(dynamics,f.name) for f in fields(lattice.PlannerDynamics))),
            None,parameters,v['target_speed_m_s'],lattice.MODE_CURVATURE_RESPONSE_SCALE[mode],
            lattice.MODE_LATERAL_RESPONSE_SCALE[mode])
        counts['compiled_calls']+=1
        return lattice.State(*result),poses,elapsed
    propagate.mission_compilation=dict(revision='original_complete_NE_AST_numba_representation_v1',
        numba_options=options,original_function_source_sha256=sources,representation_anchors=transforms,
        original_module_and_global_functions_unchanged=True,original_pow_libm_call_retained=True,
        original_CPython311_two_coordinate_norm_operation_order_retained=True,
        original_scalar_libm_trigonometric_calls_retained=True,
        original_ordered_minmax_retained=True,unsupported_inputs_delegate_original=True,
        finite_numeric_adapter_domain_is_NOT_a_planning_constraint=True,
        equivalence_NOT_assumed_or_motion_admitted=True)
    propagate.compilation_statistics=counts
    propagate.compiled_numeric_function=numerical
    propagate.compiled_norm_function=namespace['_original_hypot2']
    propagate.libm_library=library
    return propagate


def row_skipping_native_checker(original_module):
    """Original finite-cell SAT with ONE exact empty-row broad-phase skip.

    Reuse the original integral occupancy table: a zero row count proves
    there are no occupied cells to test. All nonempty-row loops, cell order,
    double equations, boundary predicates and counters are unchanged. The
    original C/Python files and global native function are never patched.
    Build only in an isolated temporary directory, from compiler stdin.
    """
    import ctypes
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    source=original_module.SOURCE.read_text(encoding='utf-8')
    anchor='        for (int yy = y0; yy <= y1; ++yy) {\n'
    addition=('            const int64_t row_n = integral[(yy+1)*stride+x1+1] - integral[yy*stride+x1+1]\n'
        '                                - integral[(yy+1)*stride+x0] + integral[yy*stride+x0];\n'
        '            if (row_n == 0) continue;\n')
    if source.count(anchor)!=1 or addition in source:
        raise ValueError('Exactly one original row loop and no prior skip required')
    derived=source.replace(anchor,anchor+addition)
    if derived.replace(anchor+addition,anchor)!=source:
        raise ValueError('Original SAT source changed outside one row-count skip')
    original_build=original_module.initialize_native()
    directory=tempfile.TemporaryDirectory(prefix='ijcas_exact_row_skip_')
    target=Path(directory.name)/'hospital_exact_row_skip.so'
    try:
        built=subprocess.run(['cc',*original_module.FLAGS,'-x','c','-','-o',str(target),'-lm'],
            input=derived,capture_output=True,text=True,check=False,timeout=60)
        if built.returncode:raise RuntimeError('Exact row-skip native compilation failed: '+built.stderr)
        library=ctypes.CDLL(str(target));function=library.hospital_sat_batch
        function.argtypes=original_module._NATIVE[2].argtypes
        function.restype=original_module._NATIVE[2].restype
        metadata=dict(original_build,revision='exact_empty_row_integral_broadphase_v1',
            source_sha256=hashlib.sha256(derived.encode()).hexdigest(),
            original_C_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            compiled_C_source_sha256=hashlib.sha256(derived.encode()).hexdigest(),
            original_row_loop_insertions=1,undo_reproduces_entire_original_C_source=True,
            all_nonempty_row_cell_order_SAT_equations_and_boundary_predicates_unchanged=True,
            original_Python_batch_code_object_reused=True,original_global_native_function_unmodified=True,
            no_extra_occupancy_approximation_or_new_grid=True,original_compile_flags=list(original_module.FLAGS))
        native=(directory,library,function,metadata)
        original=original_module.NativeHospitalEnvelopeChecker
        batch=original._batch
        scoped=FunctionType(batch.__code__,dict(batch.__globals__,_NATIVE=native),
            batch.__name__,batch.__defaults__,batch.__closure__)
        scoped.__kwdefaults__=batch.__kwdefaults__
        class Checker(original):pass
        Checker._batch=scoped
        Checker.mission_collision_binding=metadata
        return Checker
    except Exception:
        directory.cleanup();raise


def sparse_cell_native_checker(original_module):
    """Exact SAME occupied-cell SAT/order, using per-row occupied indices.

    Replace only iteration over empty cells, not geometry, clearance, grid,
    samples, float equations or early-exit order. Every occupied cell in the
    original AABB is visited in identical ascending (row,column) order.
    Original C/Python files and the global native library remain untouched.
    Each checker owns an immutable occupancy/integral/index snapshot.
    """
    import ctypes
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    import numpy as np

    source=original_module.SOURCE.read_text(encoding='utf-8')
    signature='                      const double *poses, int count, uint64_t *stats) {'
    replacement=('                      const int64_t *row_offsets, const int32_t *occupied_cols,\n'+signature)
    loop=('        for (int yy = y0; yy <= y1; ++yy) {\n'
        '            for (int xx = x0; xx <= x1; ++xx) {\n'
        '                if (!occupied[yy*cols+xx]) continue;\n')
    indexed=('        for (int yy = y0; yy <= y1; ++yy) {\n'
        '            int64_t first = row_offsets[yy], last = row_offsets[yy+1];\n'
        '            while (first < last) {\n'
        '                const int64_t middle = first + (last-first)/2;\n'
        '                if (occupied_cols[middle] < x0) first = middle+1; else last = middle;\n'
        '            }\n'
        '            for (int64_t entry = first; entry < row_offsets[yy+1] && occupied_cols[entry] <= x1; ++entry) {\n'
        '                const int xx = occupied_cols[entry];\n')
    if source.count(signature)!=1 or source.count(loop)!=1:raise ValueError('Exact original SAT signature/loop required')
    derived=source.replace(signature,replacement).replace(loop,indexed)
    if derived.replace(replacement,signature).replace(indexed,loop)!=source:
        raise ValueError('Original SAT changed outside exact occupied-cell iteration')
    original_build=original_module.initialize_native()
    directory=tempfile.TemporaryDirectory(prefix='ijcas_exact_sparse_cells_')
    target=Path(directory.name)/'hospital_exact_sparse_cells.so'
    try:
        built=subprocess.run(['cc',*original_module.FLAGS,'-x','c','-','-o',str(target),'-lm'],
            input=derived,capture_output=True,text=True,check=False,timeout=60)
        if built.returncode:raise RuntimeError('Exact occupied-cell native compilation failed: '+built.stderr)
        library=ctypes.CDLL(str(target));function=library.hospital_sat_batch
        types=list(original_module._NATIVE[2].argtypes)
        function.argtypes=types[:-3]+[ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int32)]+types[-3:]
        function.restype=ctypes.c_int
        metadata=dict(original_build,revision='exact_sparse_occupied_cell_iteration_v1',
            original_C_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            compiled_C_source_sha256=hashlib.sha256(derived.encode()).hexdigest(),
            undo_reproduces_entire_original_C_source=True,
            identical_occupied_cell_order_and_SAT_equations_and_boundary_predicates=True,
            original_global_native_function_unmodified=True,no_grid_or_pose_quantization=True)
        original=original_module.NativeHospitalEnvelopeChecker
        class Checker(original):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.occupancy=np.array(self.occupancy,dtype=bool,order='C',copy=True)
                if self.occupancy.shape[1]>np.iinfo(np.int32).max:raise ValueError('Bounded exact column indices required')
                self.integral=np.ascontiguousarray(np.pad(self.occupancy.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1))
                self._row_offsets=np.r_[0,np.cumsum(self.occupancy.sum(axis=1,dtype=np.int64))].astype(np.int64)
                self._occupied_cols=np.ascontiguousarray(np.nonzero(self.occupancy)[1],dtype=np.int32)
                for array in (self.occupancy,self.integral,self._row_offsets,self._occupied_cols):array.setflags(write=False)
                self._occupied_pointer=self.occupancy.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                self._integral_pointer=self.integral.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
                self._row_pointer=self._row_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
                self._column_pointer=self._occupied_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                self._private_library=native

            def _batch(self,poses,reference):
                self.check_deadline()
                values=np.asarray(poses if isinstance(poses,np.ndarray) else list(poses),dtype=np.float64)
                if not values.size:return False
                if values.ndim!=2 or values.shape[1]!=3:raise ValueError('Expected finite pose triples')
                values=np.ascontiguousarray(values)
                result=function(self._occupied_pointer,self._integral_pointer,*self.occupancy.shape,
                    *self.origin,self.resolution,*reference,self.clearance,self._row_pointer,self._column_pointer,
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(values),
                    self._counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
                for key,value in zip(('pose_queries','empty_aabb','exact_sat'),self._counts):self.statistics[key]+=int(value)
                self.statistics['native_batches']+=1
                if result<0:raise ValueError('Non-finite hospital pose')
                return bool(result)
        native=(directory,library,function,metadata)
        Checker.mission_collision_binding=metadata
        return Checker
    except Exception:
        directory.cleanup();raise


def sampled_body_pose_masks(occupied,origin,resolution,*,stride=5,orientation_bins=18):
    """Sample ORIGINAL rectangular cell-SAT configurations for GUIDANCE only.

    FFT counts integer occupied-cell overlaps with the same four SAT axes.
    Values are checked against integer rounding before use. Outside-map cells
    stay occupied. Only sampled poses are represented; graph connections are
    NOT continuous sweeps, steering primitives, or edge/collision authority.
    """
    import numpy as np
    from scipy.signal import fftconvolve
    if (not isinstance(occupied,np.ndarray) or occupied.dtype!=np.dtype(bool)
            or occupied.ndim!=2 or not all(occupied.shape) or occupied.size>10000000
            or type(stride) is not int or stride<1 or type(orientation_bins) is not int
            or not 4<=orientation_bins<=72 or not math.isfinite(resolution) or resolution<=0.):
        raise ValueError('Bounded original boolean map and explicit guidance sampling required')
    origin=_finite(origin,2,'original map origin');resolution=float(resolution)
    ys=np.arange(stride//2,occupied.shape[0],stride);xs=np.arange(stride//2,occupied.shape[1],stride)
    if not len(ys) or not len(xs) or orientation_bins*len(ys)*len(xs)>4000000:
        raise ValueError('Bounded full-map guidance graph required; no automatic map crop/coarsening')
    a,b=1.630/2+.010,.965/2+.010;half=resolution/2
    padding=int(math.ceil(math.hypot(a,b)/resolution))+2
    padded=np.pad(occupied,padding,constant_values=True).astype(np.float64)
    yy,xx=np.mgrid[-padding:padding+1,-padding:padding+1]
    dx=xx*resolution;dy=yy*resolution
    free=np.empty((orientation_bins,len(ys),len(xs)),dtype=bool);maximum_residual=0.
    qx=origin[0]+(xs+.5)*resolution;qy=origin[1]+(ys+.5)*resolution
    maximum=(origin[0]+occupied.shape[1]*resolution,origin[1]+occupied.shape[0]*resolution)
    for k in range(orientation_bins):
        yaw=k*math.pi/orientation_bins;c,s=math.cos(yaw),math.sin(yaw)
        ex=abs(c)*a+abs(s)*b;ey=abs(s)*a+abs(c)*b;support=half*(abs(c)+abs(s))
        kernel=((np.abs(dx)<=ex+half+1e-12)&(np.abs(dy)<=ey+half+1e-12)
            &(np.abs(c*dx+s*dy)<=a+support+1e-12)&(np.abs(-s*dx+c*dy)<=b+support+1e-12))
        count=fftconvolve(padded,kernel.astype(np.float64),mode='same')
        count=count[padding:padding+occupied.shape[0],padding:padding+occupied.shape[1]]
        nearest=np.rint(count);residual=float(np.max(np.abs(count-nearest)))
        if not np.isfinite(count).all() or residual>1e-6 or np.any(nearest<0):
            raise ValueError('Guidance overlap counts numerically ambiguous; no false free cells')
        maximum_residual=max(maximum_residual,residual)
        inside=((qy[:,None]-ey>=origin[1])&(qy[:,None]+ey<=maximum[1])
            &(qx[None,:]-ex>=origin[0])&(qx[None,:]+ex<=maximum[0]))
        free[k]=(nearest[np.ix_(ys,xs)]==0)&inside
    return free,dict(sample_origin_xy_m=[float(qx[0]),float(qy[0])],
        sample_step_m=resolution*stride,angle_step_rad=math.pi/orientation_bins,
        shape=list(free.shape),maximum_FFT_integer_residual=maximum_residual,
        original_rectangle_half_lengths_m=[a,b],outside_original_map_occupied=True,
        same_original_10mm_clearance=True,sampled_geometry_only_NOT_edge_admission=True)


class SampledBodyPoseGuidance:
    """Common position/rectangle-orientation heuristic, NEVER a route.

    A relaxed holonomic graph joins sampled configurations. Its distance and
    lookahead orientation rank ORIGINAL NE/lattice edges; sideways graph
    moves and in-place graph turns are never commands or accepted primitives.
    Same fixed task, map, body+10mm, actions, true costs and safety predicates.
    Conservative resource availability changes guidance only, not any edge.
    """
    def __init__(self,occupied,origin,resolution,start,goal_world,fallback,*,
            directions,limits,speed_m_s=.05,maximum_curvature_1_m=.7,heading_weight=.4,
            stride=5,orientation_bins=18):
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy.ndimage import distance_transform_edt
        if not callable(fallback) or not isinstance(limits,MotionResourceLimits):
            raise ValueError('Original fallback and explicit resource limits required')
        if (not math.isfinite(speed_m_s) or speed_m_s<=0. or not math.isfinite(maximum_curvature_1_m)
                or maximum_curvature_1_m<=0. or not math.isfinite(heading_weight) or heading_weight<0.):
            raise ValueError('Original finite guidance coefficients required')
        directions=tuple(directions)
        if not directions or len(set(directions))!=len(directions) or any(type(d) is not int or d not in (-1,1) for d in directions):
            raise ValueError('Explicit unchanged signed planning directions required')
        self.start=_finite(start,3,'fixed reference');self.goal=_finite(goal_world,3,'fixed world goal')
        c,s=math.cos(self.start[2]),math.sin(self.start[2]);self.c,self.s=c,s
        dx,dy=self.goal[0]-self.start[0],self.goal[1]-self.start[1]
        self.local_goal=(c*dx+s*dy,-s*dx+c*dy,math.atan2(math.sin(self.goal[2]-self.start[2]),math.cos(self.goal[2]-self.start[2])))
        self.fallback=fallback;self.directions=directions;self.limits=limits
        self.speed=speed_m_s;self.curvature=maximum_curvature_1_m;self.weight=heading_weight
        self.radius=math.hypot(1.630/2+.010,.965/2+.010);self.lookahead=1.630/2
        free,metadata=sampled_body_pose_masks(occupied,origin,resolution,stride=stride,orientation_bins=orientation_bins)
        self.free=free;self.origin=metadata['sample_origin_xy_m'];self.step=metadata['sample_step_m']
        self.angle=metadata['angle_step_rad'];K,R,C=free.shape;N=free.size
        self.parameters=dict(metadata,kind='sampled_full_rectangle_SE2_relaxed_holonomic_guidance_NOT_a_plan',
            same_guidance_for_all_methods=True,never_used_as_collision_checker=True,
            resource_limits_change_guidance_only=True,optimality_NOT_claimed=True,
            all_graph_preprocessing_charged_to_primary_clock=True,available=False)
        ids=np.arange(N,dtype=np.int32).reshape(free.shape);sources=[];destinations=[];weights=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dy or dx):continue
                a_y=slice(max(0,-dy),min(R,R-dy));b_y=slice(max(0,dy),min(R,R+dy))
                a_x=slice(max(0,-dx),min(C,C-dx));b_x=slice(max(0,dx),min(C,C+dx))
                valid=free[:,a_y,a_x]&free[:,b_y,b_x]
                if dy and dx:valid&=free[:,a_y,b_x]&free[:,b_y,a_x]
                a=ids[:,a_y,a_x][valid];b=ids[:,b_y,b_x][valid]
                sources.append(a);destinations.append(b);weights.append(np.full(len(a),math.hypot(dx,dy)*self.step))
        for shift in (-1,1):
            next_free=np.roll(free,-shift,axis=0);next_ids=np.roll(ids,-shift,axis=0);valid=free&next_free
            a=ids[valid];b=next_ids[valid];sources.append(a);destinations.append(b)
            weights.append(np.full(len(a),self.radius*self.angle))
        gx=int(math.floor((self.goal[0]-self.origin[0])/self.step+.5))
        gy=int(math.floor((self.goal[1]-self.origin[1])/self.step+.5))
        gk=int(math.floor((self.goal[2]%math.pi)/self.angle+.5))%K
        goal_nodes=[];goal_cost=[]
        for yy in range(max(0,gy-1),min(R,gy+2)):
            for xx in range(max(0,gx-1),min(C,gx+2)):
                if free[gk,yy,xx]:
                    goal_nodes.append(int(ids[gk,yy,xx]));goal_cost.append(math.hypot(
                        self.origin[0]+xx*self.step-self.goal[0],self.origin[1]+yy*self.step-self.goal[1]))
        self.parameters['sampled_goal_seed_count']=len(goal_nodes)
        if not goal_nodes:self.available=False;return
        sources.append(np.full(len(goal_nodes),N,dtype=np.int32));destinations.append(np.asarray(goal_nodes,dtype=np.int32))
        weights.append(np.asarray(goal_cost,dtype=float))
        graph=coo_matrix((np.concatenate(weights),(np.concatenate(sources),np.concatenate(destinations))),shape=(N+1,N+1)).tocsr()
        distance,previous=dijkstra(graph,directed=True,indices=N,return_predecessors=True)
        finite=np.isfinite(distance[:-1]);distance_grid=distance[:-1].reshape(free.shape)
        targets=np.arange(N,dtype=np.int32)
        for _ in range(int(math.ceil(self.lookahead/min(self.step,self.radius*self.angle)))+2):
            valid=finite&(targets<N)
            progress=np.zeros(N);progress[valid]=distance[:-1][valid]-distance[targets[valid]]
            active=valid&(progress<self.lookahead)&(previous[targets.clip(max=N)]>=0)
            if not np.any(active):break
            targets[active]=previous[targets[active]]
        self.distance=distance_grid;self.targets=targets.reshape(free.shape)
        extended=np.tile(finite.reshape(free.shape),(3,1,1))
        nearest=distance_transform_edt(~extended,sampling=(self.radius*self.angle,self.step,self.step),
            return_distances=False,return_indices=True)[:,K:2*K]
        nearest[0]%=K;self.nearest=nearest
        self.parameters.update(available=True,graph_vertices=N+1,graph_arcs=int(graph.nnz),
            reachable_sampled_configurations=int(finite.sum()),free_sampled_configurations=int(free.sum()),
            goal_seeds_are_ranking_only_NOT_physical_connectors=True,
            modulo_pi_rectangle_symmetry_NOT_a_goal_heading_change=True)
        self.available=True

    def __call__(self,state,goal):
        if _finite(goal,3,'fixed local goal')!=self.local_goal:
            raise ValueError('Configuration guidance cannot change the fixed task')
        if not self.available:return self.fallback(state,goal)
        x,y,yaw=_finite((state.x,state.y,state.yaw),3,'original planning state')
        remaining=math.hypot(x-goal[0],y-goal[1])
        if remaining<=.025:return self.fallback(state,goal)
        wx=self.start[0]+self.c*x-self.s*y;wy=self.start[1]+self.s*x+self.c*y;world_yaw=yaw+self.start[2]
        K,R,C=self.free.shape
        xx=int(math.floor((wx-self.origin[0])/self.step+.5));yy=int(math.floor((wy-self.origin[1])/self.step+.5))
        kk=int(math.floor((world_yaw%math.pi)/self.angle+.5))%K
        if not (0<=xx<C and 0<=yy<R):return self.fallback(state,goal)
        k,ay,ax=map(int,self.nearest[:,kk,yy,xx]);anchor_x=self.origin[0]+ax*self.step;anchor_y=self.origin[1]+ay*self.step
        projection=math.hypot(anchor_x-wx,anchor_y-wy)
        if projection>2*self.step:return self.fallback(state,goal)
        target=int(self.targets[k,ay,ax]);N=K*R*C
        if target==N:tx,ty,theta=self.goal
        else:
            tk,flat=divmod(target,R*C);tyi,txi=divmod(flat,C)
            tx=self.origin[0]+txi*self.step;ty=self.origin[1]+tyi*self.step;theta=tk*self.angle
        wrap=lambda value:math.atan2(math.sin(value),math.cos(value))
        align=abs(wrap(2*(theta-world_yaw)))/2
        changes=getattr(state,'mission_direction_changes',0);total=getattr(state,'mission_reverse_total_bins',0)
        episode=getattr(state,'mission_reverse_episode_bins',0)
        if any(type(v) is not int or v<0 for v in (changes,total,episode)):
            raise ValueError('Original nonnegative integer resource state required')
        directions=self.directions
        if state.started and state.direction and changes>=self.limits.maximum_direction_changes:
            directions=tuple(d for d in directions if d==state.direction)
        if total*self.limits.distance_quantum_m>=self.limits.maximum_total_reverse_m-1e-12:
            directions=tuple(d for d in directions if d>0)
        if state.direction<0 and episode*self.limits.distance_quantum_m>=self.limits.maximum_reverse_episode_m-1e-12:
            directions=tuple(d for d in directions if d>0)
        if not directions:return self.fallback(state,goal)
        turn=0.
        if math.hypot(tx-wx,ty-wy)>self.step/4:
            bearing=math.atan2(ty-wy,tx-wx)
            turn=min(self.weight*abs(wrap(bearing-world_yaw-(math.pi if d<0 else 0.)))/(self.curvature*self.speed)
                +(1. if state.started and state.direction and d!=state.direction else 0.) for d in directions)
        terminal=max(0.,1.-remaining/1.630)*abs(wrap(goal[2]-yaw))
        return (float(self.distance[k,ay,ax])+projection)/self.speed+turn+self.weight*(align+terminal)/(self.curvature*self.speed)


def search_priority_weight(original_search, *, heuristic_weight):
    """Explicit common weighted-search priority, never a model/edge waiver.

    The original queue already supports a heuristic weight. This private
    call adapter supplies the declared common value while retaining every
    state, action, cost, feasibility predicate and returned original cost.
    Non-admissible mission guidance gives NO optimality guarantee at any
    weight. Changed weights require a separate development result/version.
    """
    if (not callable(original_search) or type(heuristic_weight) not in (int,float)
            or not math.isfinite(heuristic_weight) or heuristic_weight<1.):
        raise ValueError('Explicit finite common heuristic weight >=1 required')
    def run(*args,**kwargs):
        return original_search(*args,**dict(kwargs,heuristic_weight=heuristic_weight))
    run.__dict__.update(getattr(original_search,'__dict__',{}))
    run.mission_search_priority=dict(heuristic_weight=heuristic_weight,
        original_queue_and_costs_and_all_constraints_unchanged=True,
        optimality_NOT_claimed=True,common_to_declared_comparison_methods=True)
    return run


class NEStraightArcTimeGuidance:
    """Relaxed distance/speed time-to-go table, NEVER an actual NE edge.

    Nominal straight-edge transitions are evaluated with the supplied
    ORIGINAL model/propagator. Backward DP includes the original terminal
    speed instead of pricing every remaining metre at maximum speed. Turns,
    lateral dynamics, reverse changes and steering histories are NOT solved
    by this table. Interpolation is heuristic-only, NOT state interpolation
    in the actual search or an optimal/admissible stopping certificate.
    Construction is charged to the caller's one planning deadline.
    """
    def __init__(self,propagate,model,*,dynamics,speed_levels_m_s,maximum_speed_m_s,
            terminal_speed_m_s,step_m,maximum_distance_m,check_deadline):
        import numpy as np
        from patient_initial_stability.steering_lattice import State

        levels=tuple(speed_levels_m_s)
        if (not callable(propagate) or model is None or not callable(check_deadline)
                or not levels or any(not math.isfinite(v) or v<=0. for v in levels)
                or max(levels)!=maximum_speed_m_s or not math.isfinite(terminal_speed_m_s)
                or not 0.<terminal_speed_m_s<maximum_speed_m_s
                or not math.isfinite(step_m) or step_m<=0.
                or not math.isfinite(maximum_distance_m) or not 0.<maximum_distance_m<=100.):
            raise ValueError('Explicit original NE and finite bounded arc-time guidance required')
        self.velocity=np.linspace(0.,maximum_speed_m_s*1.2,61)
        endpoints=np.empty((61,len(levels)));times=np.empty_like(endpoints)
        for i,v in enumerate(self.velocity):
            check_deadline()
            state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,1,longitudinal_velocity=float(v))
            for j,target in enumerate(levels):
                result,_,elapsed=propagate(state,'coordinated_counterphase',0.,1,
                    step_m=step_m,speed_m_s=maximum_speed_m_s,dynamics=dynamics,rigid_body_model=model,target_speed_m_s=target)
                endpoints[i,j]=math.hypot(result.longitudinal_velocity,result.lateral_velocity)
                times[i,j]=elapsed
        if (not np.isfinite(endpoints).all() or not np.isfinite(times).all()
                or (times<=0.).any() or (endpoints<0.).any() or (endpoints>self.velocity[-1]).any()):
            raise ValueError('Finite bounded original straight-edge time samples required')
        count=math.ceil(maximum_distance_m/step_m)
        if count>10000:raise ValueError('Bounded arc-time table required')
        self.table=np.empty((count+1,61));self.table[0]=np.where(self.velocity<=terminal_speed_m_s,0.,np.inf)
        for i in range(1,count+1):
            check_deadline()
            options=times+np.interp(endpoints,self.velocity,self.table[i-1])
            self.table[i]=options.min(axis=1)
        self.step=step_m;self.maximum_speed=maximum_speed_m_s;self.terminal_speed=terminal_speed_m_s
        self.slow_edge_time=step_m/min(levels)
        self.parameters=dict(revision='original_NE_straight_arc_speed_time_DP_guidance_v1',
            velocity_samples=61,maximum_table_speed_m_s=float(self.velocity[-1]),distance_step_m=step_m,
            maximum_table_distance_m=count*step_m,original_speed_levels_m_s=list(levels),
            original_terminal_speed_m_s=terminal_speed_m_s,original_NE_edges_sampled=61*len(levels),
            only_relaxed_ranking_NOT_actual_state_or_edge_or_stopping_admission=True)
        self.velocity.setflags(write=False);self.table.setflags(write=False)

    def __call__(self,distance,speed):
        if not math.isfinite(distance) or distance<0. or not math.isfinite(speed) or speed<0.:
            raise ValueError('Finite nonnegative actual distance/speed query required')
        import numpy as np
        n=distance/self.step;lo=min(int(n),len(self.table)-1);hi=min(lo+1,len(self.table)-1)
        ratio=min(1.,n-lo);speed=min(speed,float(self.velocity[-1]))
        values=[]
        for i in (lo,hi):
            # An unreachable relaxed table entry receives a finite ranking
            # penalty. It NEVER removes an actual search state/action.
            row=np.where(np.isfinite(self.table[i]),self.table[i],
                i*self.step/self.maximum_speed+self.slow_edge_time+self.velocity/self.maximum_speed*self.slow_edge_time)
            values.append(float(np.interp(speed,self.velocity,row)))
        return values[0]*(1.-ratio)+values[1]*ratio+max(0.,distance-(len(self.table)-1)*self.step)/self.maximum_speed


def reference_corridor_checker(original_checker,candidate,*,position_radius_m,heading_radius_rad):
    """Optional FIRST search stage around an OWN spatial reference.

    An explicit METHOD search restriction, NOT an obstacle/contact/safety
    predicate or a completeness claim. It tests only edge-end association to
    the union of reference segments. Every admitted edge still receives the
    ORIGINAL full substep envelope/resource check. All primitive modes,
    directions, speeds and actual NE states remain unmodified. A caller may
    spend its remaining total budget on unrestricted search if this fails.
    """
    import numpy as np
    if (not isinstance(candidate,dict) or candidate.get('plan_success') is not True
            or not candidate.get('selected_states')
            or not math.isfinite(position_radius_m) or position_radius_m<=0.
            or not math.isfinite(heading_radius_rad) or not 0.<heading_radius_rad<=math.pi):
        raise ValueError('Complete own reference and explicit finite method-corridor bounds required')
    poses=np.asarray([(0.,0.,0.)]+[_finite((s['x_m'],s['y_m'],math.radians(s['yaw_deg'])),3,
        'own reference pose') for s in candidate['selected_states']],dtype=float)
    delta=np.diff(poses[:,:2],axis=0);squared=np.einsum('ij,ij->i',delta,delta)
    if (squared<=0.).any():raise ValueError('Nonzero original reference segments required')
    dyaw=np.arctan2(np.sin(np.diff(poses[:,2])),np.cos(np.diff(poses[:,2])))
    for array in (poses,delta,squared,dyaw):array.setflags(write=False)
    class Checker(original_checker):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.statistics['method_reference_corridor_rejections_NOT_obstacle_collisions']=0
        def __call__(self,samples):
            self.check_deadline()
            if samples and getattr(samples,'resource_feasible',True):
                x,y,yaw=_finite(samples[-1],3,'actual NE edge-end pose')
                offset=np.asarray([x,y])-poses[:-1,:2]
                fraction=np.clip(np.einsum('ij,ij->i',offset,delta)/squared,0.,1.)
                error=offset-fraction[:,None]*delta
                within=np.einsum('ij,ij->i',error,error)<=position_radius_m**2
                angles=poses[:-1,2]+fraction*dyaw-yaw
                within&=np.abs(np.arctan2(np.sin(angles),np.cos(angles)))<=heading_radius_rad
                if not within.any():
                    self.statistics['method_reference_corridor_rejections_NOT_obstacle_collisions']+=1
                    return True
            return super().__call__(samples)
    Checker.mission_reference_corridor=dict(position_radius_m=position_radius_m,
        heading_radius_rad=heading_radius_rad,search_restriction_NOT_obstacle_or_physical_guard=True,
        original_substep_envelope_resource_model_and_states_unchanged=True,
        no_fixed_action_sequence_or_direction_or_mode_filter=True,unrestricted_fallback_must_share_original_total_budget=True)
    return Checker


def installed_reeds_shepp_metric():
    """Optional installed OMPL geometric distance, SOFT GUIDANCE ONLY.

    Link the existing library in a temporary C ABI shim, without installing
    packages or starting ROS/Kit. Its instantaneous steering and unrestricted
    reverse motion are NOT the original NE dynamics/resources. Neither its
    paths nor interpolated states are inserted in the original planner.
    """
    import ctypes
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    import numpy as np
    include=Path('/opt/ros/humble/include/ompl-1.7')
    header=include/'ompl/base/spaces/ReedsSheppStateSpace.h'
    library_path=Path('/opt/ros/humble/lib/x86_64-linux-gnu/libompl.so.1.7.0')
    if not header.is_file() or not library_path.is_file():raise RuntimeError('Optional installed OMPL1.7 metric unavailable; no auto-install or silent fallback')
    source=r'''
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <cmath>
extern "C" int hospital_rs_batch(double radius, const double *from,
                                const double *targets, int count, double *lengths) {
    if (!std::isfinite(radius) || radius <= 0. || count < 0) return -1;
    for (int j=0;j<3;++j) if (!std::isfinite(from[j])) return -1;
    try {
        ompl::base::ReedsSheppStateSpace space(radius);
        auto *a=space.allocState()->as<ompl::base::SE2StateSpace::StateType>();
        auto *b=space.allocState()->as<ompl::base::SE2StateSpace::StateType>();
        a->setX(from[0]);a->setY(from[1]);a->setYaw(from[2]);
        for (int i=0;i<count;++i) {
            for (int j=0;j<3;++j) if (!std::isfinite(targets[3*i+j])) {
                space.freeState(a);space.freeState(b);return -1;
            }
            b->setX(targets[3*i]);b->setY(targets[3*i+1]);b->setYaw(targets[3*i+2]);
            lengths[i]=space.distance(a,b);
        }
        space.freeState(a);space.freeState(b);return 0;
    } catch (...) { return -2; }
}
'''
    directory=tempfile.TemporaryDirectory(prefix='ijcas_ompl_metric_');target=Path(directory.name)/'metric.so'
    flags=['-O3','-std=c++17','-shared','-fPIC','-ffp-contract=off']
    try:
        built=subprocess.run(['c++',*flags,'-I'+str(include),'-I/usr/include/eigen3','-x','c++','-',
            '-x','none',str(library_path),'-Wl,-rpath,'+str(library_path.parent),'-o',str(target)],
            input=source,capture_output=True,text=True,check=False,timeout=60)
        if built.returncode:raise RuntimeError('Installed geometric metric shim compilation failed: '+built.stderr)
        library=ctypes.CDLL(str(target));function=library.hospital_rs_batch
        pointer=ctypes.POINTER(ctypes.c_double)
        function.argtypes=[ctypes.c_double,pointer,pointer,ctypes.c_int,pointer];function.restype=ctypes.c_int
        def distance(initial,targets,turning_radius_m):
            initial=np.ascontiguousarray(_finite(initial,3,'actual metric query pose'),dtype=np.float64)
            targets=np.asarray(targets,dtype=np.float64)
            if (targets.ndim!=2 or targets.shape[1]!=3 or not np.isfinite(targets).all()
                    or not math.isfinite(turning_radius_m) or turning_radius_m<=0.):
                raise ValueError('Finite oriented targets and positive radius required')
            targets=np.ascontiguousarray(targets);out=np.empty(len(targets))
            result=function(turning_radius_m,initial.ctypes.data_as(pointer),targets.ctypes.data_as(pointer),len(targets),out.ctypes.data_as(pointer))
            if result or not np.isfinite(out).all() or (out<0.).any():raise RuntimeError('Installed geometric metric failed; never accept a false connection')
            return out
        distance.parameters=dict(revision='installed_OMPL1p7_reeds_shepp_soft_distance_v1',
            header=str(header),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
            library=str(library_path),library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
            shim_source_sha256=hashlib.sha256(source.encode()).hexdigest(),compile_flags=flags,
            original_NE_and_resource_guard_NOT_replaced=True,geometric_distance_NOT_edge_or_collision_admission=True,
            unbounded_reverse_and_instant_steering_in_metric_ONLY=True,original_goal_and_actual_states_unchanged=True)
        distance.private_library=(directory,library)
        return distance
    except Exception:
        directory.cleanup();raise


class NETerminalTimeGuidance:
    """Rank terminal NE states by oriented distance AND relaxed stop time.

    This is an explicit search revision, not a new plant, edge, goal predicate
    or motion primitive. The supplied original-model straight-edge table is
    a relaxation: steering, lateral response, reverse resource and obstacles
    still require original full-state propagation/checks. No admissibility or
    optimality claim is made. Construction belongs to the caller's budget.
    """
    def __init__(self,goal,*,connection_metric,arc_time_guidance,accepted_goal,
            maximum_speed_m_s,maximum_curvature_1_m):
        self.goal=_finite(goal,3,'fixed NE terminal goal')
        if (not all(callable(v) for v in (connection_metric,arc_time_guidance,accepted_goal))
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0.
                    for v in (maximum_speed_m_s,maximum_curvature_1_m))):
            raise ValueError('Explicit metric, original-model stop-time guidance, complete goal and positive scales required')
        self.metric=connection_metric;self.arc_time=arc_time_guidance;self.accepted=accepted_goal
        self.speed=float(maximum_speed_m_s);self.radius=1./maximum_curvature_1_m
        self.parameters=dict(revision='original_NE_terminal_oriented_distance_and_stop_time_priority_v1',
            combination='max(oriented_distance/maximum_speed,original_NE_relaxed_arc_stop_time)',
            metric=getattr(connection_metric,'parameters',None),
            stop_time=getattr(arc_time_guidance,'parameters',None),
            maximum_speed_m_s=self.speed,maximum_curvature_1_m=float(maximum_curvature_1_m),
            exact_complete_original_goal_only_receives_zero_priority=True,
            states_edges_costs_keys_resources_and_goal_constraints_unchanged=True,
            no_action_filter_state_insertion_or_physical_authority=True,
            heuristic_NOT_a_feasible_connection_or_admissibility_or_optimality_proof=True)
        self.statistics=dict(queries=0,complete_goal_queries=0,stop_time_priority_raised=0)

    def __call__(self,state,goal):
        if _finite(goal,3,'NE terminal query goal')!=self.goal:
            raise ValueError('NE terminal guidance cannot move the fixed goal')
        x,y,yaw,u,v=_finite((state.x,state.y,state.yaw,state.longitudinal_velocity,
            state.lateral_velocity),5,'actual full-state terminal pose and body velocity')
        self.statistics['queries']+=1
        if self.accepted(state,True):
            self.statistics['complete_goal_queries']+=1
            return 0.
        values=self.metric((x,y,yaw),[self.goal],self.radius)
        if len(values)!=1 or not math.isfinite(float(values[0])) or float(values[0])<0.:
            raise ValueError('One finite nonnegative terminal connection metric required')
        distance=float(values[0]);geometric=distance/self.speed
        stop=float(self.arc_time(distance,math.hypot(u,v)))
        if not math.isfinite(stop) or stop<0.:
            raise ValueError('Finite nonnegative original-model stop-time priority required')
        self.statistics['stop_time_priority_raised']+=int(stop>geometric)
        return max(geometric,stop)


class TerminalPoseConnectionGuidance:
    """Common terminal queue ranking, never a connection or goal certificate.

    Nearest-reference position can be cheap even with the wrong final BODY
    yaw. Inside one explicitly declared terminal zone, rank by the maximum
    of the existing heuristic and an oriented geometric goal connection.
    The metric may assume instantaneous steering/unlimited reverse: ORIGINAL
    NE states, costs, collision, resources and goal tests still decide every
    actual edge. Neither admissibility nor optimality is claimed.
    """
    def __init__(self,fallback,goal,*,connection_metric,speed_m_s,
            maximum_curvature_1_m,terminal_zone_m):
        self.goal=_finite(goal,3,'fixed terminal ranking goal')
        values=(speed_m_s,maximum_curvature_1_m,terminal_zone_m)
        if (not callable(fallback) or not callable(connection_metric)
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0. for v in values)
                or terminal_zone_m<=.025):
            raise ValueError('Existing fallback, oriented metric and explicit positive terminal ranking bounds required')
        self.fallback=fallback;self.metric=connection_metric
        self.speed=float(speed_m_s);self.radius=1./maximum_curvature_1_m
        self.zone=float(terminal_zone_m)
        definition=dict(revision='common_terminal_oriented_connection_priority_v1',
            speed_m_s=float(speed_m_s),maximum_curvature_1_m=float(maximum_curvature_1_m),
            terminal_zone_m=self.zone,original_pose_goal_tolerance_m_deg=[.025,2.],
            combination='max(existing_priority,geometric_connection_length/speed)',
            accepted_pose_retains_original_fallback_and_NE_speed_braking_priority=True,
            same_rule_for_all_four_methods_and_owned_P_reference=True,
            original_states_edges_costs_bins_goal_constraints_and_resources_NOT_changed=True,
            metric_is_NOT_a_feasible_path_or_NE_lower_bound_or_optimality_proof=True,
            metric=getattr(connection_metric,'parameters',None))
        self.parameters=dict(getattr(fallback,'parameters',{}),terminal_pose_connection=definition)
        self.statistics=dict(queries=0,metric_queries=0,priority_raised=0)

    def __call__(self,state,goal):
        if _finite(goal,3,'terminal ranking query goal')!=self.goal:
            raise ValueError('Terminal ranking cannot move the fixed goal')
        x,y,yaw=_finite((state.x,state.y,state.yaw),3,'observed complete-search body pose')
        previous=float(self.fallback(state,goal))
        if not math.isfinite(previous) or previous<0.:
            raise ValueError('Finite nonnegative existing priority required; no silent metric fallback')
        self.statistics['queries']+=1
        distance=math.hypot(x-self.goal[0],y-self.goal[1])
        heading=abs(math.atan2(math.sin(yaw-self.goal[2]),math.cos(yaw-self.goal[2])))
        # A valid pose is not necessarily an accepted NE speed/state: retain
        # its original braking priority instead of declaring it a goal here.
        if distance>self.zone or distance<=.025 and heading<=math.radians(2.):return previous
        values=self.metric((x,y,yaw),[self.goal],self.radius)
        if len(values)!=1 or not math.isfinite(float(values[0])) or float(values[0])<0.:
            raise ValueError('One finite nonnegative geometric connection distance required')
        self.statistics['metric_queries']+=1
        result=max(previous,float(values[0])/self.speed)
        self.statistics['priority_raised']+=int(result>previous)
        return result

    def wrap_reference(self,reference):
        """Same common terminal rule after OWN fresh-reference construction."""
        return type(self)(reference,self.goal,connection_metric=self.metric,speed_m_s=self.speed,
            maximum_curvature_1_m=1./self.radius,terminal_zone_m=self.zone)


class OwnedSpatialReferenceGuidance:
    """Soft ranking from a caller's freshly generated car-kinematic route.

    This is NOT an NE seed, hard corridor, waypoint constraint, collision
    certificate, action sequence, or cost overlay. No reference pose or
    velocity is ever inserted into the actual search. The caller must charge
    reference generation and this construction to the SAME search budget.
    The original full-state NE search remains free to leave this reference.
    """
    def __init__(self, candidate, goal, fallback, *, speed_m_s,
            maximum_curvature_1_m, heading_weight, arc_time_guidance=None,
            forecast_distance_m=0.,forecast_geometry=None,connection_metric=None):
        import numpy as np

        self.goal=_finite(goal,3,'unchanged reference goal')
        if (not callable(fallback) or not math.isfinite(speed_m_s) or speed_m_s<=0.
                or not math.isfinite(maximum_curvature_1_m) or maximum_curvature_1_m<=0.
                or not math.isfinite(heading_weight) or heading_weight<0.
                or not isinstance(candidate,dict) or candidate.get('plan_success') is not True
                or not candidate.get('selected_states') or not candidate.get('selected_actions')
                or len(candidate['selected_states'])!=len(candidate['selected_actions'])):
            raise ValueError('Complete owned spatial reference and explicit finite ranking parameters required')
        if arc_time_guidance is not None and not callable(arc_time_guidance):raise ValueError('Callable arc-time ranking required')
        if connection_metric is not None and not callable(connection_metric):raise ValueError('Callable oriented connection metric required')
        if connection_metric is not None and forecast_distance_m:
            raise ValueError('Select one connection ranking revision, not an unused forecast')
        if not math.isfinite(forecast_distance_m) or not 0.<=forecast_distance_m<=1.63:
            raise ValueError('Explicit bounded soft steering-forecast distance required')
        if forecast_distance_m:
            required=('wheelbase_m','front_length_m','rear_length_m','curvature_scales','lateral_scales')
            if not isinstance(forecast_geometry,dict) or set(forecast_geometry)!=set(required):
                raise ValueError('Complete original geometry/response constants required for soft forecast')
            for name in required[:3]:
                if not math.isfinite(forecast_geometry[name]) or forecast_geometry[name]<=0.:
                    raise ValueError('Positive original steering geometry required')
            for name in required[3:]:
                if (set(forecast_geometry[name])!={'coordinated_counterphase','front_only','rear_only'}
                        or any(not math.isfinite(v) or v<=0. for v in forecast_geometry[name].values())):
                    raise ValueError('Complete finite original mode-response constants required')
        wrap=lambda a:math.atan2(math.sin(a),math.cos(a))
        poses=[(0.,0.,0.)]+[_finite((s['x_m'],s['y_m'],math.radians(s['yaw_deg'])),3,
            'owned reference pose') for s in candidate['selected_states']]
        if (math.hypot(poses[-1][0]-self.goal[0],poses[-1][1]-self.goal[1])>.025
                or abs(wrap(poses[-1][2]-self.goal[2]))>math.radians(2.)):
            raise ValueError('Owned reference must reach the ORIGINAL fixed spatial goal')
        self.poses=np.asarray(poses,dtype=float)
        self.delta=np.diff(self.poses[:,:2],axis=0)
        self.length=np.hypot(self.delta[:,0],self.delta[:,1])
        if (not np.isfinite(self.length).all() or (self.length<=0.).any()):
            raise ValueError('Nonzero finite owned reference segments required')
        self.yaw_delta=np.arctan2(np.sin(np.diff(self.poses[:,2])),np.cos(np.diff(self.poses[:,2])))
        self.suffix=np.r_[np.cumsum(self.length[::-1])[::-1],0.]
        self.end_gap=math.hypot(poses[-1][0]-self.goal[0],poses[-1][1]-self.goal[1])
        self.speed=float(speed_m_s)
        self.angular_scale=heading_weight/(maximum_curvature_1_m*speed_m_s)
        self.fallback=fallback
        self.arc_time=arc_time_guidance
        self.connection_metric=connection_metric
        self.connection_radius=1./maximum_curvature_1_m
        self.forecast_distance=float(forecast_distance_m)
        self.forecast_geometry=None if not forecast_distance_m else dict(forecast_geometry,
            curvature_scales=dict(forecast_geometry['curvature_scales']),lateral_scales=dict(forecast_geometry['lateral_scales']))
        self.progress=np.r_[0.,np.cumsum(self.length)]
        self.directions=tuple(a.get('direction') if isinstance(a,dict) else a[2] for a in candidate['selected_actions'])
        if forecast_distance_m and any(type(d) is not int or d not in (-1,1) for d in self.directions):
            raise ValueError('Explicit original reference directions required for reversal-safe forecast ranking')
        end=len(self.length);self.direction_end=np.empty(len(self.length),dtype=int)
        for i in range(len(self.length)-1,-1,-1):
            if i<len(self.length)-1 and self.directions[i]!=self.directions[i+1]:end=i+1
            self.direction_end[i]=end
        for array in (self.poses,self.delta,self.length,self.yaw_delta,self.suffix):array.setflags(write=False)
        self.parameters=dict(revision='owned_spatial_projection_soft_ranking_v1',
            segment_count=len(self.length),reference_length_m=float(self.suffix[0]),
            speed_m_s=speed_m_s,maximum_curvature_1_m=maximum_curvature_1_m,heading_weight=heading_weight,
            reference_changes_priority_only=True,reference_states_never_inserted=True,
            no_hard_corridor_or_waypoint_or_action_filter=True,goal_position_and_full_body_yaw_unchanged=True,
            arc_time_guidance=getattr(arc_time_guidance,'parameters',None),
            connection_metric=getattr(connection_metric,'parameters',None),
            forecast_distance_m=forecast_distance_m,forecast_geometry=self.forecast_geometry,
            forecast_is_held_measured_steering_ranking_NOT_NE_or_state_insertion=bool(forecast_distance_m),
            no_optimality_or_NE_or_physical_admission=True)

    def __call__(self,state,goal):
        import numpy as np

        if _finite(goal,3,'reference query goal')!=self.goal:
            raise ValueError('Owned reference cannot silently move the fixed goal')
        x,y,yaw=_finite((state.x,state.y,state.yaw),3,'actual NE pose')
        if math.hypot(x-self.goal[0],y-self.goal[1])<=.025:
            return self.fallback(state,goal)
        offset=np.array([x,y])-self.poses[:-1,:2]
        fraction=np.clip(np.einsum('ij,ij->i',offset,self.delta)/(self.length*self.length),0.,1.)
        error=offset-fraction[:,None]*self.delta
        distance=np.hypot(error[:,0],error[:,1])
        minimum=float(distance.min())
        # Coincident/retraced segments are disambiguated by ACTUAL body yaw,
        # then shorter suffix. This is memoryless ranking, not progress reset.
        near=np.flatnonzero(distance<=minimum+1e-12)
        desired=self.poses[near,2]+fraction[near]*self.yaw_delta[near]
        heading=np.abs(np.arctan2(np.sin(desired-yaw),np.cos(desired-yaw)))
        i=int(near[np.lexsort((self.suffix[near],heading))[0]])
        angle=abs(math.atan2(math.sin(self.poses[i,2]+fraction[i]*self.yaw_delta[i]-yaw),
            math.cos(self.poses[i,2]+fraction[i]*self.yaw_delta[i]-yaw)))
        remaining=self.suffix[i]-fraction[i]*self.length[i]+self.end_gap
        time=remaining/self.speed if self.arc_time is None else self.arc_time(remaining,
            math.hypot(state.longitudinal_velocity,state.lateral_velocity))
        cross=float(distance[i])
        if self.connection_metric is not None:
            desired_yaw=self.poses[i,2]+fraction[i]*self.yaw_delta[i]
            target=self.poses[i,:2]+fraction[i]*self.delta[i]
            connector=float(self.connection_metric((x,y,yaw),[(target[0],target[1],desired_yaw)],self.connection_radius)[0])
            # Replace Euclidean sideways+independent yaw repair with ONE
            # oriented geometric connection. It affects queue order only.
            return time+connector/self.speed
        if self.forecast_distance:
            current=self.progress[i]+fraction[i]*self.length[i]
            look=min(self.forecast_distance,self.progress[self.direction_end[i]]-current)
            target=min(float(self.progress[-1]),current+look)
            j=min(len(self.length)-1,int(np.searchsorted(self.progress,target,side='right')-1))
            part=max(0.,min(1.,(target-self.progress[j])/self.length[j]))
            desired=self.poses[j,:2]+part*self.delta[j]
            desired_yaw=self.poses[j,2]+part*self.yaw_delta[j]
            predicted=self.forecast_pose(state,look,self.directions[i])
            cross=max(cross,math.hypot(predicted[0]-desired[0],predicted[1]-desired[1]))
            angle=max(angle,abs(math.atan2(math.sin(predicted[2]-desired_yaw),math.cos(predicted[2]-desired_yaw))))
        return time+cross/self.speed+self.angular_scale*angle

    def forecast_pose(self,state,distance,reference_direction):
        """Analytic held-measured-steering GEOMETRY for ranking, never NE."""
        g=self.forecast_geometry
        if g is None:raise ValueError('No steering forecast selected')
        front,rear=math.tan(state.front),math.tan(state.rear)
        curvature=(front-rear)/g['wheelbase_m']*g['curvature_scales'][state.mode]
        ratio=(g['rear_length_m']*front+g['front_length_m']*rear)/g['wheelbase_m']*g['lateral_scales'][state.mode]
        signed=distance*(state.direction if state.direction in (-1,1) else reference_direction)
        yaw=state.yaw+curvature*signed
        if abs(curvature)>1e-8:
            dx=(math.sin(yaw)-math.sin(state.yaw)+ratio*(math.cos(yaw)-math.cos(state.yaw)))/curvature
            dy=(-math.cos(yaw)+math.cos(state.yaw)+ratio*(math.sin(yaw)-math.sin(state.yaw)))/curvature
        else:
            dx=signed*(math.cos(state.yaw)-ratio*math.sin(state.yaw))
            dy=signed*(math.sin(state.yaw)+ratio*math.cos(state.yaw))
        return state.x+dx,state.y+dy,yaw


def plan_NE_mission_candidate(original_search, bounds, goal, checker, model, guidance, *,
        directions, speed_levels_m_s, maximum_speed_m_s,
        terminal_speed_m_s, deceleration_m_s2, node_limit, diagnostics):
    """Bind the EXISTING joint NE lattice, not a kinematic post-repair.

    Original propagation, all original modes/curvatures, physical state bins,
    costs, edge constraints and terminal predicates are reused. Callers own
    the deadline and must charge this complete search/preparation to their
    end-to-end budget. No cached route input, reset or motion authority.
    """
    from patient_initial_stability import steering_lattice as lattice
    if getattr(original_search,'mission_search_binding',{}).get('intermediate_region_NOT_mission_goal'):
        raise ValueError('Intermediate region search cannot certify the whole mission goal')
    if getattr(original_search,'mission_rolling_resolution',None) is not None:
        raise ValueError('Diagnostic rolling resolution cannot use the original whole-mission action serializer')
    levels=tuple(speed_levels_m_s)
    if (model is None or not callable(original_search) or not callable(guidance)
            or not callable(checker) or not callable(getattr(checker,'check_deadline',None))
            or type(node_limit) is not int or node_limit<=0 or not isinstance(diagnostics,dict)
            or not levels or len(set(levels))!=len(levels)
            or not all(math.isfinite(v) and v>0. for v in
                (*levels,maximum_speed_m_s,terminal_speed_m_s,deceleration_m_s2))
            or max(levels)!=maximum_speed_m_s):
        raise ValueError('Explicit original NE model, finite speed/stop bounds and search budget required')
    search=bounded_lattice_function(original_search,bounds,directions=directions)
    checker.check_deadline()
    if checker([(0.,0.,0.),_finite(goal,3,'unchanged fixed NE goal')]):
        raise ValueError('Original full start/goal envelope is blocked')
    return search(goal,bounds['minimum_y'],bounds['maximum_y'],
        robust_clearance=.010,allowed_modes=lattice.ALL_MODES,allowed_curvatures=lattice.CURVATURES,
        speed_m_s=maximum_speed_m_s,node_limit=node_limit,collision_checker=checker,
        initial_mode='coordinated_counterphase',initial_front_rad=0.,initial_rear_rad=0.,
        propagate_initial_steering=True,dynamics=lattice.FULL_DYNAMICS,
        rigid_body_model=model,speed_levels_m_s=levels,
        terminal_speed_tolerance_m_s=terminal_speed_m_s,terminal_deceleration_m_s2=deceleration_m_s2,
        goal_heading_tolerance_rad=math.radians(2.),heuristic_weight=1.5,
        heuristic_function=lambda state:(checker.check_deadline() or guidance(state,goal)),diagnostics=diagnostics)


def bounded_NE_terminal_beam(initial,propagate,state_key,bounds,goal,checker,model,accepted_goal,*,
        beam_width,maximum_depth,node_limit,diagnostics,executable_stop_connections=False,goal_handoff_radius_m=None):
    """Bounded terminal candidate search; original50mm NE controls/guards.

    Layer pruning is an APPROXIMATE SEARCH POLICY, never exact dominance,
    infeasibility or optimality proof. The incoming state/history is kept
    verbatim. Every original mode/curvature/direction/speed is generated at
    each retained parent; no reference or goal pose is inserted as a state.
    Original objective is accumulated for audit, but goal-normalized pose and
    speed error ranks the beam instead of pretending to optimize that cost.
    Caller owns one deadline/node budget and independent full-chain replay.
    """
    from patient_initial_stability import steering_lattice as lattice
    goal=_finite(goal,3,'fixed original NE beam goal')
    if (any(not callable(v) for v in (propagate,state_key,checker,accepted_goal))
            or not callable(getattr(checker,'check_deadline',None)) or model is None
            or type(beam_width) is not int or not 1<=beam_width<=128
            or type(maximum_depth) is not int or not 1<=maximum_depth<=32
            or type(node_limit) is not int or node_limit<=0 or not isinstance(diagnostics,dict)):
        raise ValueError('Explicit bounded terminal beam, original model/checker and remaining node budget required')
    if type(executable_stop_connections) is not bool:raise ValueError('Explicit common stop-connection policy required')
    if goal_handoff_radius_m is not None and (not executable_stop_connections
            or type(goal_handoff_radius_m) not in (int,float) or goal_handoff_radius_m not in (.020,.030)):
        raise ValueError('Explicit stronger20/30mm final25mm-before-end handoff required')
    limits=_finite([bounds[k] for k in ('minimum_x','maximum_x','minimum_y','maximum_y')],4,'original search bounds')
    if limits[0]>limits[1] or limits[2]>limits[3]:raise ValueError('Ordered original search bounds required')
    diagnostics.update(revision='original_NE_goal_normalized_terminal_beam_v1',
        beam_width=beam_width,maximum_depth=maximum_depth,expanded_nodes=0,generated=0,
        rigid_body=0,bounds=0,resource=0,stopping_reachability=0,collision=0,completed_layers=0,
        approximate_layer_pruning_NOT_exact_dominance=True,original_objective_optimized_NOT_claimed=True,
        incoming_state_and_reverse_history_never_reset=True,no_original_control_family_filtered_before_guards=True,
        original50mm_step_25mm_2deg_5mm_s_goal_and_20mm_s2_stop_guard_unchanged=True)
    if executable_stop_connections:
        diagnostics.update(executable_stop_connection_revision='COMMON_NEUTRAL_STOP_AND_ZERO_CURVATURE_RESTART_v1',
            additional_goal_steering_tolerance_deg=2.5,final_action_curvature_must_be_zero=True,
            direction_change_requires_previous_neutral_and_under5mm_s_and_next_zero_curvature=True,
            physical_stop_and_steering_response_STILL_require_SDK_validation=True,
            stronger_connection_condition_NOT_relaxed_original_goal_or_new_plant=True,
            cusp_connection=0)
    if goal_handoff_radius_m is not None:
        diagnostics.update(additional_original_controller25mm_before_end_capture_radius_m=goal_handoff_radius_m,
            original_goal_pose_NOT_shifted_and_original30mm_execution_acceptance_NOT_relaxed=True)
    def handoff_distance(state):
        return math.hypot(state.x-.025*math.cos(state.yaw)-goal[0],state.y-.025*math.sin(state.yaw)-goal[1])
    def rank(state):
        distance=math.hypot(state.x-goal[0],state.y-goal[1])
        steering=(max(0.,1.-distance/.50)*max(abs(state.front),abs(state.rear))/math.radians(2.5)
            if executable_stop_connections else 0.)
        capture=(max(0.,handoff_distance(state)-goal_handoff_radius_m)/.010
            if goal_handoff_radius_m is not None else 0.)
        return (distance/.025+steering+capture
            +abs(lattice.wrap(state.yaw-goal[2]))/math.radians(2.)
            +max(0.,math.hypot(state.longitudinal_velocity,state.lateral_velocity)/.005-1.))
    def complete(state,curvature):
        return accepted_goal(state,True) and (not executable_stop_connections
            or curvature==0. and max(abs(state.front),abs(state.rear))<=math.radians(2.5)) and (
                goal_handoff_radius_m is None or handoff_distance(state)<=goal_handoff_radius_m)
    checker.check_deadline()
    if checker([(initial.x,initial.y,initial.yaw)]):raise ValueError('Incoming full-state footprint blocked')
    if complete(initial,0.):return lattice.PlanResult((),(),0,0.,())
    beam=[(initial,(),(),0.)];best=rank(initial)
    diagnostics['best_goal_normalized_error']=best
    for depth in range(maximum_depth):
        labels={};serial=0
        for previous,states,actions,cost in beam:
            checker.check_deadline()
            if diagnostics['expanded_nodes']>=node_limit:
                diagnostics['termination']='node_limit';return None
            diagnostics['expanded_nodes']+=1
            for mode in lattice.ALL_MODES:
                for curvature in lattice.CURVATURES:
                    for direction in (1,-1):
                        for speed in (.003,.01,.03,.05):
                            checker.check_deadline();diagnostics['generated']+=1
                            state,poses,elapsed=propagate(previous,mode,curvature,direction,
                                speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,
                                rigid_body_model=model,target_speed_m_s=speed)
                            if not math.isfinite(elapsed) or elapsed<=0.:
                                raise ValueError('Finite positive original NE transition time required')
                            reason=None
                            if not state.rigid_body_feasible:reason='rigid_body'
                            elif not limits[0]<=state.x<=limits[1] or not limits[2]<=state.y<=limits[3]:reason='bounds'
                            elif not getattr(poses,'resource_feasible',False):reason='resource'
                            elif (executable_stop_connections and previous.started and previous.direction
                                    and previous.direction!=direction and (curvature!=0.
                                        or max(abs(previous.front),abs(previous.rear))>math.radians(2.5)
                                        or math.hypot(previous.longitudinal_velocity,previous.lateral_velocity)>.005)):
                                reason='cusp_connection'
                            elif math.hypot(state.longitudinal_velocity,state.lateral_velocity)>math.sqrt(
                                    .005**2+2*.020*max(0.,math.hypot(state.x-goal[0],state.y-goal[1])-.025))+1e-12:
                                reason='stopping_reachability'
                            elif checker(poses):reason='collision'
                            if reason is not None:
                                diagnostics[reason]+=1;continue
                            target=lattice.steering_targets(mode,curvature)
                            new_cost=(cost+elapsed+(lattice.MODE_SWITCH_PENALTY_S if previous.started and mode!=previous.mode else 0.)
                                +(lattice.DIRECTION_SWITCH_PENALTY_S if previous.started and previous.direction!=0 and direction!=previous.direction else 0.)
                                +(lattice.STEERING_TRANSIENT_PENALTY_WEIGHT*max(abs(target[0]-previous.front),abs(target[1]-previous.rear))
                                    /lattice.STEERING_RATE_RAD_S if previous.started else 0.)
                                +.02*abs(curvature)+state.rigid_body_penalty_s)
                            action=(mode,curvature,direction,speed)
                            candidate=(state,(*states,state),(*actions,action),new_cost)
                            if complete(state,curvature):
                                diagnostics.update(termination='complete_original_goal',completed_layers=depth+1)
                                return lattice.PlanResult(candidate[1],candidate[2],diagnostics['expanded_nodes'],new_cost,())
                            score=rank(state);serial+=1
                            if score<best:
                                best=score;diagnostics.update(best_goal_normalized_error=score,
                                    best_goal_position_error_m=math.hypot(state.x-goal[0],state.y-goal[1]),
                                    best_goal_heading_error_deg=math.degrees(lattice.wrap(state.yaw-goal[2])),
                                    best_goal_speed_m_s=math.hypot(state.longitudinal_velocity,state.lateral_velocity))
                            key=state_key(state,True);old=labels.get(key)
                            entry=(score,new_cost,serial,candidate)
                            if old is None or entry[:2]<old[:2]:labels[key]=entry
        diagnostics['completed_layers']=depth+1
        if not labels:
            diagnostics['termination']='frontier_exhausted_NOT_global_infeasibility';return None
        beam=[row[3] for row in sorted(labels.values(),key=lambda row:row[:3])[:beam_width]]
    diagnostics['termination']='depth_limit_NOT_global_infeasibility'
    return None


def mission_local_heading_guidance(field, fallback, goal_local_xyyaw, *,
        directions, speed_m_s, maximum_curvature_1_m, heading_weight,
        terminal_zone_m, terminal_position_tolerance_m):
    """Common SEARCH heuristic: local travel first, final yaw near the goal.

    The legacy lookahead heuristic also penalised the difference between
    EVERY intermediate bearing and the FINAL goal yaw. In a multi-door
    mission that can create an avoidable priority barrier before the first
    turn, even along a collision-free prefix. Here final BODY yaw is blended
    only inside an explicit terminal zone, not interpreted as travel yaw.

    This is non-admissible guidance, never a path/edge approval, constraint,
    state reset, mode filter, or permission to reverse. Every real edge and
    terminal test still belongs to the unchanged original lattice/NE/SAT.
    Missing relaxed guidance falls back to the original heuristic. A finite
    but malformed field fails instead of disguising corruption as fallback.
    """
    import numpy as np

    goal = _finite(goal_local_xyyaw,3,'fixed local goal')
    directions = tuple(directions)
    if (not directions or len(set(directions))!=len(directions)
            or any(type(v) is not int or v not in (-1,1) for v in directions)):
        raise ValueError('Explicit unique signed planning directions required')
    values=(speed_m_s,maximum_curvature_1_m,terminal_zone_m,terminal_position_tolerance_m)
    if (not all(math.isfinite(v) and v>0. for v in values)
            or not math.isfinite(heading_weight) or heading_weight<0.
            or terminal_zone_m<=terminal_position_tolerance_m or not callable(fallback)):
        raise ValueError('Explicit finite mission guidance parameters required')
    start = _finite(field.start,3,'fixed guidance reference')
    origin = np.asarray(field.origin,dtype=float)
    distance = np.asarray(field.distance)
    targets = np.asarray(field.targets)
    resolution = float(field.resolution)
    if (origin.shape!=(2,) or not np.isfinite(origin).all()
            or distance.ndim!=2 or not all(distance.shape)
            or targets.shape!=(*distance.shape,2)
            or not math.isfinite(resolution) or resolution<=0.):
        raise ValueError('Complete finite relaxed guidance geometry required')
    c,s=math.cos(start[2]),math.sin(start[2])
    wrap=lambda angle:math.atan2(math.sin(angle),math.cos(angle))
    field_goal=_finite(field.goal,3,'fixed guidance goal')
    dx,dy=field_goal[0]-start[0],field_goal[1]-start[1]
    derived_goal=(c*dx+s*dy,-s*dx+c*dy,wrap(field_goal[2]-start[2]))
    if any(abs(a-b)>1e-12 for a,b in zip(derived_goal,goal)):
        raise ValueError('Relaxed guidance field must use the same fixed task')

    def guidance(state, supplied_goal):
        if _finite(supplied_goal,3,'fixed local goal')!=goal:
            raise ValueError('Mission guidance cannot move the fixed goal')
        x_local,y_local,yaw=_finite((state.x,state.y,state.yaw),3,'observed planning pose')
        remaining=math.hypot(x_local-goal[0],y_local-goal[1])
        if remaining<=terminal_position_tolerance_m:
            return fallback(state,goal)
        wx=start[0]+c*x_local-s*y_local
        wy=start[1]+s*x_local+c*y_local
        x,y=np.floor((np.asarray([wx,wy])-origin)/resolution).astype(int)
        if not (0<=y<distance.shape[0] and 0<=x<distance.shape[1]):
            return fallback(state,goal)
        distance_m=float(distance[y,x])
        if math.isinf(distance_m) and distance_m>0.:
            return fallback(state,goal)
        if not math.isfinite(distance_m) or distance_m<0.:
            raise ValueError('Invalid finite relaxed guidance distance')
        tx,ty=_finite(targets[y,x],2,'relaxed lookahead target')
        bearing=math.atan2(ty-wy,tx-wx)-start[2]
        local_turn=min(abs(wrap(bearing-yaw-(math.pi if direction<0 else 0.)))
            for direction in directions)
        # BODY orientation must match the real goal even when moving reverse.
        final_body_yaw_error=abs(wrap(goal[2]-yaw))
        terminal_weight=max(0.,1.-remaining/terminal_zone_m)
        return distance_m/speed_m_s+heading_weight*(local_turn+
            terminal_weight*final_body_yaw_error)/(maximum_curvature_1_m*speed_m_s)

    return guidance


class ResourceParetoBest:
    """Cost/resource labels WITHIN each original discrete physical key.

    Retain incomparable cheap/high-resource and costly/low-resource labels.
    A label dominates only with no greater cost AND no greater consumption
    of EVERY resource. Mode, direction, steering and body-key components are
    never merged here; the original spatial/angle quantisation is unchanged.
    This inherits the lattice's approximation, NOT global completeness or
    optimality. Actual states/resources and all edge guards remain intact.
    """

    def __init__(self, initial=None):
        self._buckets={}
        self._retained=0
        self.statistics=dict(queries=0,strict_resource_dominance_hits=0,
            labels_inserted=0,dominated_labels_removed=0,
            dominated_insertions_ignored=0,maximum_labels_in_one_physical_bin=0)
        if initial is not None:
            for key,cost in initial.items():self[key]=cost

    @staticmethod
    def _split(key):
        if not isinstance(key,tuple) or len(key)<4:
            raise ValueError('Physical key plus THREE resource counters required')
        physical,resources=key[:-3],key[-3:]
        if any(type(v) is not int or v<0 for v in resources):
            raise ValueError('Nonnegative integer resource counters required')
        if resources[0]>resources[1]:
            raise ValueError('Reverse episode cannot exceed total reverse consumption')
        try:hash(physical)
        except TypeError as error:raise ValueError('Hashable original physical key required') from error
        return physical,resources

    def get(self, key, default=None):
        physical,resources=self._split(key)
        self.statistics['queries']+=1
        # Sorted by cost: first dominating resource label is the cheapest.
        for cost,used in self._buckets.get(physical,()):
            if all(a<=b for a,b in zip(used,resources)):
                self.statistics['strict_resource_dominance_hits']+=int(used!=resources)
                return cost
        return default

    def __setitem__(self, key, cost):
        physical,resources=self._split(key)
        if isinstance(cost,bool) or not math.isfinite(cost) or cost<0.:
            raise ValueError('Finite nonnegative original path cost required')
        cost=float(cost)
        old=self._buckets.get(physical,())
        if any(c<=cost and all(a<=b for a,b in zip(r,resources)) for c,r in old):
            self.statistics['dominated_insertions_ignored']+=1
            return
        kept=[(c,r) for c,r in old
            if not (cost<=c and all(a<=b for a,b in zip(resources,r)))]
        removed=len(old)-len(kept)
        kept.append((cost,resources));kept.sort()
        self._buckets[physical]=kept
        self._retained+=1-removed
        self.statistics['labels_inserted']+=1
        self.statistics['dominated_labels_removed']+=removed
        self.statistics['maximum_labels_in_one_physical_bin']=max(
            self.statistics['maximum_labels_in_one_physical_bin'],len(kept))

    def __len__(self):
        return self._retained

    def snapshot(self):
        return dict(self.statistics,retained_labels=self._retained,
            original_physical_key_bins=len(self._buckets))


def diagnostic_rolling_resolution(original, *, step_m):
    """Private moving-edge resolution probe, NOT a new admitted mission plan.

    Reuse the original search code and resource propagator with an explicit
    25mm or original50mm travelled-distance edge. Never add stationary
    steering, reset a state, edit a kernel, change key bins or waive guards.
    The original per-edge cost formula is retained, so costs across different
    resolutions are NOT matched performance comparisons. Returned action
    tuples omit length; this binding is therefore rejected by the original
    whole-mission serializer. Local diagnostic segments MUST store step_m
    explicitly and be independently replayed with that same distance.
    """
    from patient_initial_stability import steering_lattice as lattice
    if (type(step_m) is not float or step_m not in (.025,lattice.STEP_M)
            or not isinstance(original,FunctionType) or original.__closure__
            or not callable(original.__globals__.get('propagate'))
            or getattr(original,'mission_rolling_resolution',None) is not None):
        raise ValueError('Explicit original standalone search and diagnostic25/50mm rolling step required')
    propagate=original.__globals__['propagate']
    def scoped(*args,**kwargs):
        if 'step_m' in kwargs and kwargs['step_m']!=step_m:
            raise ValueError('Diagnostic moving-edge distance cannot change inside a search')
        return propagate(*args,**dict(kwargs,step_m=step_m))
    run=FunctionType(original.__code__,dict(original.__globals__,propagate=scoped),
        original.__name__,original.__defaults__,original.__closure__)
    run.__kwdefaults__=None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    run.mission_rolling_resolution=dict(step_m=step_m,
        diagnostic_only_NOT_original_whole_mission_action_schema=True,
        original_search_code_object_and_state_key_bins_retained=True,
        original_NE_constraints_and_per_edge_cost_formula_retained=True,
        costs_across_resolutions_NOT_matched_performance=True,
        stationary_reconfiguration_added=False)
    return run


def continue_original_lattice_from_state(original, initial_state):
    """Original local-frame search from a COMPLETE already propagated state.

    OFFLINE continuation only, not restoration of any physical robot. All
    pose/steering/velocity/force/resource fields and the coordinate frame are
    carried unchanged. The caller must retain/audit/charge the prefix and
    add its original cost; a cached prefix is NEVER a free primary input.
    This replaces ONLY the original one-time start constructor in a private
    namespace. Original propagation, keys, guards and goal stay untouched.
    """
    from dataclasses import fields,is_dataclass
    if (not isinstance(original,FunctionType) or original.__closure__
            or not is_dataclass(initial_state) or 'State' not in original.__globals__):
        raise ValueError('Original standalone lattice and complete propagated dataclass state required')
    values={f.name:getattr(initial_state,f.name) for f in fields(initial_state)}
    if any(isinstance(v,float) and not math.isfinite(v) for v in values.values()):
        raise ValueError('Finite complete continuation state required')
    state_type=original.__globals__['State']
    if not isinstance(initial_state,state_type):raise ValueError('Exact original resource-state class required')
    def constructor(*args,**kwargs):
        nominal=state_type(*args,**kwargs)
        if (nominal.x,nominal.y,nominal.yaw)!=(0.,0.,0.):
            raise ValueError('Continuation only replaces the original zero-pose start constructor')
        return initial_state
    run=FunctionType(original.__code__,dict(original.__globals__,State=constructor),original.__name__,
        original.__defaults__,original.__closure__)
    run.__kwdefaults__=None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    run.mission_continuation=dict(exact_initial_state=values,original_search_code_object_reused=True,
        no_coordinate_or_pose_velocity_steering_resource_reset=True,prefix_must_be_audited_and_charged=True)
    if getattr(original,'mission_rolling_resolution',None) is not None:
        run.mission_rolling_resolution=dict(original.mission_rolling_resolution)
    return run


def copy_original_search_prefixes(observed):
    """Copy real parent chains for diagnosis; never manufacture a PlanResult.

    Read only original search-frame objects. Incomplete, uninitialised and
    stale queue-head states are labelled explicitly. No source node, queue,
    cost, counter or state is changed. Complete body/steering/resource fields
    are retained so a later audit need not guess missing velocities.
    """
    from dataclasses import fields

    nodes=observed.get('nodes',[])
    result=dict(diagnostic_only_NOT_a_complete_plan=True,node_count=len(nodes),prefixes={})
    if not nodes:return result
    identities={id(node['state']):i for i,node in enumerate(nodes)}
    selected={name:observed.get(key) for name,key in
        (('goal_closest','best_goal_state'),('currently_expanded','state'))}
    queue=observed.get('queue',[])
    if queue:
        result['raw_queue_head_priority']=queue[0][0]
        result['raw_queue_head_may_be_stale']=True
        selected['raw_queue_head']=nodes[queue[0][2]]['state']
    for name,state in selected.items():
        if state is None:continue
        index=identities.get(id(state))
        if index is None:raise ValueError('Observed state identity absent from original nodes')
        endpoint=index;chain=[]
        while True:
            node=nodes[index]
            chain.append(dict(node_id=index,state={f.name:getattr(node['state'],f.name) for f in fields(node['state'])},
                action=None if node['action'] is None else list(node['action']),cost_s=node['cost']))
            parent=node['parent']
            if parent is None:break
            if type(parent) is not int or not 0<=parent<index:
                raise ValueError('Original parent chain is not a finite backward tree')
            index=parent
        result['prefixes'][name]=dict(endpoint_node_id=endpoint,edge_count=len(chain)-1,
            records=list(reversed(chain)),goal_completion_NOT_claimed=True)
    return result


def pareto_resource_lattice_function(original, *, capture_prefixes=False, stage_acceptance=None,
                                     generated_goal_exit=False):
    """Replace ONLY the original `best` dictionary constructor privately.

    The search code object is explicitly recompiled, not claimed identical.
    A checked one-node AST substitution keeps every original propagation,
    action, collision, cost, goal and queue statement unchanged. Undoing that
    substitution must reproduce the entire original function AST exactly.
    Optional explicit INTERMEDIATE-region termination is a SEPARATE search
    method change, never a whole-goal certificate. Its callback receives the
    actual state and original whole-goal predicate. All original propagation,
    global-goal stopping reachability, resource guards and costs stay intact.
    An optional explicit GENERATED-goal exit copies the ORIGINAL complete
    goal predicate and parent-chain return AFTER a validated child is stored.
    This permits a feasible result before a later queue pop; it is not an
    optimality claim. The default remains the original pop-goal policy.
    No original module/file/global is patched. Intended input is the private
    resource-aware lattice from bind_resource_kernels, NOT a live executor.

    Timeout diagnostics are copies of the ORIGINAL search frame, never a
    fabricated result or swallowed exception. Each call owns its label store
    and namespace, so concurrent calls do not share diagnostic state.
    """
    import ast
    import copy
    import hashlib
    import inspect
    from collections.abc import MutableMapping

    if type(capture_prefixes) is not bool:raise ValueError('Explicit boolean diagnostic prefix selection required')
    if type(generated_goal_exit) is not bool or generated_goal_exit and stage_acceptance is not None:
        raise ValueError('Explicit generated whole-goal exit cannot certify an intermediate region')
    if stage_acceptance is not None and not callable(stage_acceptance):
        raise ValueError('Explicit callable intermediate region or None required')
    if original.__closure__:
        raise ValueError('Explicit standalone original lattice function required')
    source=inspect.getsource(original)
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise ValueError('One original lattice function required')
    if tree.body[0].name!=original.__name__ or tree.body[0].decorator_list:
        raise ValueError('Unmodified standalone lattice definition required')
    symbol='__mission_resource_pareto_best';stage_symbol='__mission_intermediate_region'
    if any(isinstance(n,ast.Name) and n.id in (symbol,stage_symbol) for n in ast.walk(tree)):
        raise ValueError('Resource dominance cannot be applied twice')
    before=ast.dump(tree,include_attributes=False)
    generated_marker='__mission_generated_complete_goal'
    generated_count=[]
    if generated_goal_exit:
        # Reuse the ORIGINAL predicate/velocity test/chain return, not a
        # separately relaxed approximation. Existing statements stay exact.
        goal_nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If)
            and any(isinstance(v,ast.Name) and v.id=='terminal_speed_ok' for v in ast.walk(n.test))
            and any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='PlanResult'
                for v in ast.walk(n))]
        speed_nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='terminal_speed_ok' for t in n.targets)]
        if len(goal_nodes)!=1 or len(speed_nodes)!=1:
            raise ValueError('One exact original goal predicate and terminal-speed expression required')
        class ChildPredicate(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='terminal_speed_ok':return self.visit(copy.deepcopy(speed_nodes[0].value))
                if node.id=='state':return ast.copy_location(ast.Name(id='new_state',ctx=node.ctx),node)
                return node
        predicate=ChildPredicate().visit(copy.deepcopy(goal_nodes[0].test))
        body=ast.parse('node_id = new_id\ncost = new_cost').body+copy.deepcopy(goal_nodes[0].body)
        injected=ast.If(test=predicate,body=body,orelse=[])
        # Reversible marker exists only in a private AST; it has no runtime
        # call, state assignment or constraint effect outside this exit.
        injected._mission_generated_goal_marker=generated_marker
        class GeneratedGoal(ast.NodeTransformer):
            def visit_Assign(self,node):
                if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                        and node.targets[0].id=='new_id'):
                    expected=ast.parse('new_id = len(nodes) - 1').body[0]
                    if ast.dump(node,include_attributes=False)!=ast.dump(expected,include_attributes=False):
                        raise ValueError('Original validated-child index statement changed')
                    generated_count.append(node)
                    return [node,copy.deepcopy(injected)]
                return node
        tree=GeneratedGoal().visit(tree)
        if len(generated_count)!=1:raise ValueError('Exactly one stored validated-child exit point required')
    matches=[]
    class Constructor(ast.NodeTransformer):
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=='best' and isinstance(node.value,ast.Dict)):
                matches.append(node)
                node.value=ast.copy_location(ast.Call(func=ast.Name(id=symbol,ctx=ast.Load()),
                    args=[node.value],keywords=[]),node.value)
            return node
    tree=Constructor().visit(tree)
    if len(matches)!=1:
        raise ValueError('Exactly ONE original best-dictionary constructor required')
    stages=[]
    if stage_acceptance is not None:
        class Region(ast.NodeTransformer):
            def visit_If(self,node):
                node=self.generic_visit(node)
                if (any(isinstance(n,ast.Name) and n.id=='terminal_speed_ok' for n in ast.walk(node.test))
                        and any(isinstance(n,ast.Return) and isinstance(n.value,ast.Call)
                            and isinstance(n.value.func,ast.Name) and n.value.func.id=='PlanResult'
                            for n in ast.walk(node))):
                    stages.append(node)
                    node.test=ast.copy_location(ast.Call(func=ast.Name(id=stage_symbol,ctx=ast.Load()),
                        args=[ast.Name(id='state',ctx=ast.Load()),node.test],keywords=[]),node.test)
                return node
        tree=Region().visit(tree)
        if len(stages)!=1:raise ValueError('Exactly ONE original goal predicate required for intermediate search')
    after=ast.dump(tree,include_attributes=False)
    class Undo(ast.NodeTransformer):
        def visit_If(self,node):
            if getattr(node,'_mission_generated_goal_marker',None)==generated_marker:return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id==stage_symbol:
                if len(node.args)!=2 or node.keywords:raise ValueError('Unexpected intermediate AST substitution')
                return self.visit(node.args[1])
            if isinstance(node.func,ast.Name) and node.func.id==symbol:
                if len(node.args)!=1 or node.keywords:raise ValueError('Unexpected AST substitution')
                return node.args[0]
            return self.generic_visit(node)
    if ast.dump(Undo().visit(copy.deepcopy(tree)),include_attributes=False)!=before:
        raise ValueError('An original search/guard statement changed outside declared private substitutions')
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),original.__code__.co_filename+'#resource_pareto_v1','exec'),namespace)
    compiled=namespace[original.__name__]
    compiled.__defaults__=original.__defaults__
    compiled.__kwdefaults__=None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    signature=inspect.signature(original)

    def run(*args,**kwargs):
        diagnostics=signature.bind_partial(*args,**kwargs).arguments.get('diagnostics')
        if diagnostics is not None and not isinstance(diagnostics,MutableMapping):
            raise ValueError('Mutable original diagnostics mapping required')
        stores=[]
        def constructor(initial):
            store=ResourceParetoBest(initial);stores.append(store);return store
        call=FunctionType(compiled.__code__,dict(compiled.__globals__,**{symbol:constructor,stage_symbol:stage_acceptance}),
            compiled.__name__,compiled.__defaults__,compiled.__closure__)
        call.__kwdefaults__=compiled.__kwdefaults__
        try:
            return call(*args,**kwargs)
        except TimeoutError as error:
            trace=error.__traceback__;frame=None
            while trace is not None:
                if trace.tb_frame.f_code is call.__code__:frame=trace.tb_frame;break
                trace=trace.tb_next
            if diagnostics is not None and frame is not None:
                observed=frame.f_locals
                counts=observed.get('rejection_counts')
                if counts is not None:diagnostics.update(dict(counts))
                if 'expanded' in observed:diagnostics['expanded_nodes']=observed['expanded']
                if 'best_goal_position_error' in observed:
                    diagnostics['best_goal_position_error_m']=observed['best_goal_position_error']
                state=observed.get('best_goal_state')
                goal=observed.get('goal')
                if state is not None and goal is not None:
                    yaw=math.atan2(math.sin(state.yaw-goal[2]),math.cos(state.yaw-goal[2]))
                    diagnostics.update(best_goal_heading_error_deg=math.degrees(yaw),
                        best_goal_speed_m_s=math.hypot(state.longitudinal_velocity,state.lateral_velocity),
                        observed_best_goal_state=dict(x_m=state.x,y_m=state.y,yaw_rad=state.yaw,
                            mode=state.mode,front_rad=state.front,rear_rad=state.rear,direction=state.direction,
                            reverse_episode_bins=state.mission_reverse_episode_bins,
                            reverse_total_bins=state.mission_reverse_total_bins,
                            direction_changes=state.mission_direction_changes))
                diagnostics.update(original_timeout_exception=str(error),
                    timeout_diagnostics_source='ORIGINAL_SEARCH_FRAME_NOT_A_REPLAY_OR_PLAN',
                    search_initialisation_complete='expanded' in observed)
                if capture_prefixes:
                    try:diagnostics['original_search_prefixes']=copy_original_search_prefixes(observed)
                    except (ValueError,TypeError,KeyError,IndexError) as capture_error:
                        diagnostics['prefix_capture_error']=type(capture_error).__name__+': '+str(capture_error)
            # Do not retain a failed search's potentially large node list via
            # our diagnostic frame/local references after the original catch.
            trace=frame=None
            observed=counts=state=goal=None
            raise
        finally:
            if diagnostics is not None and stores:
                diagnostics['resource_pareto']=stores[0].snapshot()

    run.__name__=original.__name__
    run.mission_search_binding=dict(revision='resource_pareto_first_stored_goal_v2' if generated_goal_exit else
        'resource_pareto_best_constructor_only_v1',
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_AST_sha256=hashlib.sha256(before.encode()).hexdigest(),
        modified_AST_sha256=hashlib.sha256(after.encode()).hexdigest(),
        AST_constructor_substitutions=1,AST_intermediate_goal_substitutions=len(stages),undo_reproduces_entire_original_AST=True,
        AST_generated_complete_goal_exits=len(generated_count),
        generated_goal_policy='first_stored_fully_validated_goal_not_optimality_claim' if generated_goal_exit else 'original_queue_pop_goal',
        search_code_object_identical=False,original_lattice_source_file_unchanged=True,
        all_original_propagation_collision_cost_goal_and_primitive_statements_unchanged=stage_acceptance is None,
        intermediate_region_NOT_mission_goal=stage_acceptance is not None,
        original_global_goal_stopping_resources_costs_and_primitives_unchanged=True,
        original_physical_key_quantisation_retained_NOT_global_completeness=True,
        counters_and_physical_states_never_reset=True,original_timeout_rethrown=True)
    run.mission_search_binding['diagnostic_prefix_capture_enabled']=capture_prefixes
    if getattr(original,'mission_rolling_resolution',None) is not None:
        run.mission_rolling_resolution=dict(original.mission_rolling_resolution)
    return run


def plan_NE_region_connection(original, initial_state, bounds, goal, checker, model, guidance, *,
        region, node_limit, diagnostics):
    """Explicit intermediate search; a returned segment is NOT a mission PASS.

    The ORIGINAL final goal remains the stopping-reachability target. Region
    termination alone may retain nonzero velocity. No waypoint state, zero
    velocity or steering is inserted. The complete incoming state belongs to
    the original resource-state class and all segment costs remain original.
    """
    from patient_initial_stability import steering_lattice as lattice
    if not callable(region):raise ValueError('Explicit intermediate region required')
    if type(node_limit) is not int or node_limit<=0:raise ValueError('Bounded region node budget required')
    if not isinstance(diagnostics,dict) or not callable(guidance) or model is None:
        raise ValueError('Explicit region diagnostics, guidance and original model required')
    continued=continue_original_lattice_from_state(original,initial_state)
    history=(initial_state.mission_reverse_episode_bins,initial_state.mission_reverse_total_bins,
        initial_state.mission_direction_changes)
    if (initial_state.rigid_body_feasible is not True or any(type(v) is not int or v<0 for v in history)
            or history[0]*.005>.5+1e-12 or history[1]*.005>1.+1e-12 or history[2]>4):
        raise ValueError('Feasible complete incoming model/resource state required before intermediate acceptance')
    search=pareto_resource_lattice_function(continued,
        capture_prefixes=True,stage_acceptance=lambda state,whole_goal:bool(whole_goal or region(state)))
    checker.check_deadline()
    if checker([(initial_state.x,initial_state.y,initial_state.yaw),_finite(goal,3,'unchanged final mission goal')]):
        raise ValueError('Original incoming-state/final-goal envelope blocked')
    bounded=bounded_lattice_function(search,bounds,directions=(1,-1))
    segment=bounded(goal,bounds['minimum_y'],bounds['maximum_y'],
        robust_clearance=.010,allowed_modes=lattice.ALL_MODES,allowed_curvatures=lattice.CURVATURES,
        speed_m_s=.050,node_limit=node_limit,collision_checker=checker,
        initial_mode='coordinated_counterphase',initial_front_rad=0.,initial_rear_rad=0.,
        propagate_initial_steering=True,dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,
        speed_levels_m_s=(.003,.01,.03,.05),terminal_speed_tolerance_m_s=.005,
        terminal_deceleration_m_s2=.020,goal_heading_tolerance_rad=math.radians(2.),heuristic_weight=1.5,
        heuristic_function=lambda state:(checker.check_deadline() or guidance(state,goal)),diagnostics=diagnostics)
    return dict(segment=segment,intermediate_segment_NOT_mission_goal_certificate=True,
        complete_initial_state=initial_state,original_final_stopping_goal=tuple(goal),
        primitive_step_m=getattr(original,'mission_rolling_resolution',{}).get('step_m',lattice.STEP_M))


def rate_aware_spatial_reference(original, bounds, goal, checker, guidance, *, node_limit, diagnostics):
    """OWN P candidate generator using original finite-rate geometric model.

    Reuse FULL_DYNAMICS with NO rigid-body model: steering rate/identified
    geometric response/state/costs retained, but no body inertia/forces or
    stopping-speed certificate. NEVER a returned full NE or physical plan.
    Caller must charge this generation to the complete P time/node budget.
    """
    from patient_initial_stability import steering_lattice as lattice
    if type(node_limit) is not int or node_limit<=0 or not isinstance(diagnostics,dict) or not callable(guidance):
        raise ValueError('Explicit bounded own finite-rate spatial search required')
    checker.check_deadline()
    if checker([(0.,0.,0.),_finite(goal,3,'unchanged full spatial goal')]):
        raise ValueError('Original full start/goal envelope blocked')
    search=pareto_resource_lattice_function(original,capture_prefixes=True)
    result=bounded_lattice_function(search,bounds,directions=(1,-1))(
        goal,bounds['minimum_y'],bounds['maximum_y'],robust_clearance=.010,
        allowed_modes=lattice.ALL_MODES,allowed_curvatures=lattice.CURVATURES,speed_m_s=.05,
        node_limit=node_limit,collision_checker=checker,initial_mode='coordinated_counterphase',
        initial_front_rad=0.,initial_rear_rad=0.,propagate_initial_steering=True,
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=None,heuristic_weight=1.5,
        goal_heading_tolerance_rad=math.radians(2.),
        heuristic_function=lambda state:(checker.check_deadline() or guidance(state,goal)),diagnostics=diagnostics)
    return dict(spatial_result=result,reference_ONLY_NOT_NE_goal_or_physical_admission=True,
        original_finite_steering_rate_and_identified_geometric_response=True,
        rigid_body_forces_inertia_terminal_speed_NOT_evaluated=True)


class ReferenceSteeringPreparation:
    """Method arrival targets from original finite steering-rate information.

    Arc-weighted future tracked commands, truncated before direction change.
    The preview distance is original maximum speed times the largest adjacent
    reference joint change / original8deg/s rate. No state is assigned, no new
    actual control primitive is created, and this is NOT a reachability proof.
    """
    def __init__(self,candidate,*,maximum_speed_m_s=.05):
        import numpy as np
        from patient_initial_stability import steering_lattice as lattice
        if (not isinstance(candidate,dict) or candidate.get('plan_success') is not True
                or not math.isfinite(maximum_speed_m_s) or maximum_speed_m_s<=0.):
            raise ValueError('Complete own reference and finite positive preview speed required')
        states=candidate.get('selected_states',[]);actions=candidate.get('selected_actions',[])
        if not states or len(states)!=len(actions):raise ValueError('Complete reference states/actions required')
        xy=np.asarray([(0.,0.)]+[_finite((s['x_m'],s['y_m']),2,'reference XY') for s in states])
        lengths=np.hypot(*np.diff(xy,axis=0).T)
        if (lengths<=0.).any():raise ValueError('Positive reference centre arcs required')
        targets=[];directions=[]
        for a in actions:
            if (a.get('mode') not in lattice.ALL_MODES or a.get('curvature_1_m') not in lattice.CURVATURES
                    or type(a.get('direction')) is not int or a['direction'] not in (-1,1)):
                raise ValueError('Only original modes/curvatures/directions may define preparation')
            f,r=lattice.steering_targets(a['mode'],a['curvature_1_m'])
            targets.append((f+lattice.STEERING_TRACKING_BIAS_RAD['front'],r+lattice.STEERING_TRACKING_BIAS_RAD['rear']))
            directions.append(a['direction'])
        self.arc=np.r_[0.,np.cumsum(lengths)];self.targets=np.asarray(targets)
        maximum_change=float(np.max(np.abs(np.diff(np.vstack(([0.,0.],self.targets)),axis=0))))
        self.preview_distance=maximum_speed_m_s*maximum_change/lattice.STEERING_RATE_RAD_S
        self.direction_end=np.empty(len(actions),dtype=int);end=len(actions)
        for i in range(len(actions)-1,-1,-1):
            if i<len(actions)-1 and directions[i]!=directions[i+1]:end=i+1
            self.direction_end[i]=end
        self.parameters=dict(revision='original_rate_derived_arc_weighted_future_joint_arrival_v1',
            maximum_adjacent_joint_change_rad=maximum_change,original_rate_rad_s=lattice.STEERING_RATE_RAD_S,
            maximum_speed_m_s=maximum_speed_m_s,derived_preview_distance_m=self.preview_distance,
            actual_joint_region_radius_rad=lattice.STEERING_RESOLUTION_RAD,
            method_arrival_restriction_NOT_physical_tolerance_or_reachability_certificate=True,
            original_controls_model_and_state_not_modified=True)
        for array in (self.arc,self.targets,self.direction_end):array.setflags(write=False)

    def at(self,distance):
        import numpy as np
        if not math.isfinite(distance) or not 0.<=distance<=float(self.arc[-1]):
            raise ValueError('Reference preparation query must lie inside its finite arc')
        i=min(len(self.targets)-1,int(np.searchsorted(self.arc,distance,side='right')-1))
        stop=min(float(self.arc[-1]),distance+self.preview_distance,float(self.arc[self.direction_end[i]]))
        widths=np.maximum(0.,np.minimum(self.arc[1:],stop)-np.maximum(self.arc[:-1],distance))
        target=(widths@self.targets/widths.sum()) if widths.sum()>1e-12 else self.targets[i]
        return dict(front=float(target[0]),rear=float(target[1]),preview_distance_used_m=max(0.,stop-distance))


def hierarchical_NE_reference_connections(original, initial_state, bounds, goal, candidate,
        checker_factory, model, terminal_guidance, *, deadline, node_limit,
        clock, gate_spacing_m=.5, gate_radius_m=.08, gate_heading_rad=math.radians(5.),
        stage_wall_s=2., stage_node_limit=200, alternatives_per_gate=4, observe=None, steering_ready=False):
    """OFFLINE own-reference decomposition with full-state backtracking.

    This is an explicit new SEARCH method, not the old global A* with a new
    name, a completeness claim or a physical execution. Intermediate gates
    restrict the search method; they do not alter obstacles/goal tolerances.
    Only a complete original terminal predicate can return a whole PlanResult.
    All explored alternatives consume the single caller-owned time/node cap.
    Caller must generate and charge its OWN spatial reference in a primary.
    """
    import numpy as np
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    if type(steering_ready) is not bool:raise ValueError('Explicit boolean steering-ready method selection required')
    values=(gate_spacing_m,gate_radius_m,gate_heading_rad,stage_wall_s)
    if (not all(math.isfinite(v) and v>0. for v in values) or gate_heading_rad>math.pi
            or gate_radius_m>=gate_spacing_m/2 or not math.isfinite(deadline)
            or any(type(v) is not int or v<=0 for v in (node_limit,stage_node_limit,alternatives_per_gate))
            or not callable(clock) or not callable(checker_factory) or not callable(terminal_guidance)):
        raise ValueError('Explicit bounded hierarchical search resources and gate definition required')
    goal=_finite(goal,3,'unchanged hierarchical goal')
    if goal!=tuple(bounds['local_goal_xyyaw']):raise ValueError('Hierarchical search cannot move the final goal')
    if not isinstance(candidate,dict) or candidate.get('plan_success') is not True:
        raise ValueError('Caller-owned complete spatial reference required')
    states=candidate.get('selected_states',[]);actions=candidate.get('selected_actions',[])
    if not states or len(states)!=len(actions):raise ValueError('Complete original reference actions/states required')
    poses=np.asarray([(0.,0.,0.)]+[_finite((s['x_m'],s['y_m'],math.radians(s['yaw_deg'])),3,
        'own hierarchical reference pose') for s in states])
    lengths=np.hypot(*np.diff(poses[:,:2],axis=0).T);arc=np.r_[0.,np.cumsum(lengths)]
    if not np.isfinite(arc).all() or (lengths<=0.).any():raise ValueError('Positive finite reference segments required')
    if math.hypot(*(poses[-1,:2]-goal[:2]))>.025 or abs(lattice.wrap(poses[-1,2]-goal[2]))>math.radians(2.):
        raise ValueError('Reference must retain the original whole task goal')
    preparation=ReferenceSteeringPreparation(candidate) if steering_ready else None
    gates=[]
    for distance in np.arange(gate_spacing_m,max(gate_spacing_m,arc[-1]-gate_spacing_m),gate_spacing_m):
        index=min(len(lengths)-1,int(np.searchsorted(arc,distance,side='right')-1))
        fraction=(distance-arc[index])/lengths[index]
        pose=poses[index]+fraction*np.r_[poses[index+1,:2]-poses[index,:2],lattice.wrap(poses[index+1,2]-poses[index,2])]
        a=actions[index];front,rear=lattice.steering_targets(a['mode'],a['curvature_1_m'])
        gate=dict(pose=tuple(map(float,pose)),front=front+lattice.STEERING_TRACKING_BIAS_RAD['front'],
            rear=rear+lattice.STEERING_TRACKING_BIAS_RAD['rear'])
        if preparation is not None:gate.update(preparation.at(float(distance)))
        gates.append(gate)
    gates.append(None)
    key=original.__globals__['_state_key'];excluded=[set() for _ in gates]
    chosen=[];endpoints=[initial_state];events=[];expanded=0;current=initial_state
    def whole(state):
        return (state.started and math.hypot(state.x-goal[0],state.y-goal[1])<=.025
            and abs(lattice.wrap(state.yaw-goal[2]))<=math.radians(2.)
            and math.hypot(state.longitudinal_velocity,state.lateral_velocity)<=.005)
    while clock()<deadline and expanded<node_limit:
        index=len(chosen);gate=gates[index];diag={};cap=min(stage_node_limit,node_limit-expanded)
        checker=checker_factory(min(deadline,clock()+stage_wall_s));result=None;error=None
        tick=clock()
        try:
            if gate is None:
                search=pareto_resource_lattice_function(continue_original_lattice_from_state(original,current),capture_prefixes=True)
                result=plan_NE_mission_candidate(search,bounds,goal,checker,model,terminal_guidance,
                    directions=(1,-1),speed_levels_m_s=(.003,.01,.03,.05),maximum_speed_m_s=.05,
                    terminal_speed_m_s=.005,deceleration_m_s2=.020,node_limit=cap,diagnostics=diag)
            else:
                target=gate['pose']
                def region(state):
                    return (math.hypot(state.x-target[0],state.y-target[1])<=gate_radius_m
                        and abs(lattice.wrap(state.yaw-target[2]))<=gate_heading_rad
                        and (not steering_ready or max(abs(state.front-gate['front']),
                            abs(state.rear-gate['rear']))<=lattice.STEERING_RESOLUTION_RAD)
                        and key(state,True) not in excluded[index])
                def guide(state,unused_goal):
                    position=max(0.,math.hypot(state.x-target[0],state.y-target[1])-gate_radius_m)/.05
                    heading=max(0.,abs(lattice.wrap(state.yaw-target[2]))-gate_heading_rad)/(.7*.05)
                    steering=max(abs(state.front-gate['front']),abs(state.rear-gate['rear']))/lattice.STEERING_RATE_RAD_S
                    return max(position,heading)+lattice.STEERING_TRANSIENT_PENALTY_WEIGHT*steering
                result=plan_NE_region_connection(original,current,bounds,goal,checker,model,guide,
                    region=region,node_limit=cap,diagnostics=diag)['segment']
        except TimeoutError as caught:error=str(caught)
        used=diag.get('expanded_nodes',0)
        if type(used) is not int or not 0<=used<=cap:raise ValueError('Actual local expansion accounting required')
        expanded+=used
        event=dict(gate_index=index,is_final_original_goal=gate is None,found_segment=result is not None,
            expanded_nodes=used,expanded_total=expanded,elapsed_s=clock()-tick,error=error,
            incoming_state=current,endpoint=None if result is None else (result.states[-1] if result.states else current),
            diagnostics=diag,checker_statistics=dict(checker.statistics),
            selected_segment_actions=[] if result is None else [list(action) for action in result.actions],
            selected_segment_full_states=[] if result is None else [asdict(state) for state in result.states],
            selected_segment_original_cost_s=None if result is None else result.cost,
            method_arrival_gate=gate,steering_ready_method_region=steering_ready)
        events.append(event)
        if observe is not None:observe(event)
        if result is not None:
            endpoint=result.states[-1] if result.states else current
            chosen.append(result);endpoints.append(endpoint);current=endpoint
            if whole(current):
                return lattice.PlanResult(tuple(s for part in chosen for s in part.states),
                    tuple(a for part in chosen for a in part.actions),expanded,sum(part.cost for part in chosen)),dict(
                        status='COMPLETE_ORIGINAL_MODEL_GOAL_NOT_PHYSX',events=events,expanded_nodes=expanded,
                        local_deadline_respected=clock()<=deadline,
                        steering_preparation_parameters=None if preparation is None else preparation.parameters,
                        no_pose_velocity_steering_force_resource_reset=True,all_original_global_stopping_and_edge_guards_retained=True)
            if gate is None:raise ValueError('Original terminal search returned a nonterminal state')
            continue
        if not chosen:break
        # Discard a computed hypothesis, never reset a physical/surviving-path
        # state. The next alternative starts from its EXACT computed parent.
        while chosen:
            chosen.pop();endpoint=endpoints.pop();index=len(chosen)
            excluded[index].add(key(endpoint,True))
            for later in range(index+1,len(excluded)):excluded[later].clear()
            current=endpoints[-1]
            if len(excluded[index])<alternatives_per_gate:break
        else:break
    return None,dict(status='LOCAL_HIERARCHICAL_SEARCH_EXHAUSTED_OR_BUDGET_NOT_GLOBAL_INFEASIBILITY',
        events=events,expanded_nodes=expanded,retained_gate_count=len(chosen),whole_goal_NOT_reached=True,
        steering_preparation_parameters=None if preparation is None else preparation.parameters,
        no_pose_velocity_steering_force_resource_reset=True,all_original_global_stopping_and_edge_guards_retained=True)


def owned_candidate_portfolio(generate, evaluate, *, sources, clock,
        budget_s, generation_cap_s, refinement_reserve_s, node_limit,
        observe=None):
    """Anytime OWN-candidate/NE scheduling; no model, SDK or cached input.

    Priority is an explicit prospective source order, not a result-selected
    route. Every source remains eligible, every actual generation consumes
    the SAME cumulative node/time budget, and NE starts as soon as an owned
    candidate is available. Keep the cheapest feasible evaluated incumbent;
    incomplete families are explicitly pending, never called infeasible.
    The callbacks own all original model/geometry/goal checks. Returning a
    row here is not physical admission or global route optimality.
    """
    sources=tuple(sources)
    if (not sources or any(not isinstance(s,str) or not s for s in sources)
            or len(set(sources))!=len(sources) or not callable(generate)
            or not callable(evaluate) or not callable(clock)
            or (observe is not None and not callable(observe))):
        raise ValueError('Explicit unique own-source callbacks required')
    values=(budget_s,generation_cap_s,refinement_reserve_s)
    if (not all(math.isfinite(v) and v>0. for v in values)
            or generation_cap_s>budget_s or refinement_reserve_s>=budget_s
            or type(node_limit) is not int or node_limit<=0):
        raise ValueError('Explicit finite cumulative portfolio budgets required')
    start=float(clock());last=start
    if not math.isfinite(start):raise ValueError('Finite monotonic clock required')
    def now():
        nonlocal last
        value=float(clock())
        if not math.isfinite(value) or value<last:
            raise ValueError('Portfolio clock cannot reverse or become nonfinite')
        last=value;return value
    def remaining():return max(0.,budget_s-(now()-start))
    generated=[];evaluated=[];nodes=0;generation_wall=0.;refinement_wall=0.
    incumbent=None;incumbent_source=None
    for source in sources:
        available=min(generation_cap_s-generation_wall,remaining()-refinement_reserve_s)
        if available<=0. or nodes>=node_limit:break
        allocation=available;node_allocation=node_limit-nodes;tick=now()
        candidate,used_nodes=generate(source,allocation,node_allocation)
        elapsed=now()-tick;generation_wall+=elapsed
        if (type(used_nodes) is not int or not 0<=used_nodes<=node_allocation
                or not isinstance(candidate,dict) or type(candidate.get('plan_success')) is not bool):
            raise ValueError('Actual own generation result and bounded node count required')
        nodes+=used_nodes
        found=candidate['plan_success']
        if found and not candidate.get('selected_actions'):
            raise ValueError('A found own candidate must contain original actions')
        event=dict(source_method=source,status=candidate.get('status'),plan_found=found,
            elapsed_s=elapsed,allocated_wall_s=allocation,node_cap=node_allocation,expanded_nodes=used_nodes)
        generated.append(event)
        if observe is not None:observe('generation',source,candidate,event)
        if not found:continue
        allocation=remaining()
        if allocation<=0.:break
        tick=now();profile=evaluate(source,candidate,allocation);elapsed=now()-tick
        refinement_wall+=elapsed
        if not isinstance(profile,dict) or type(profile.get('plan_success')) is not bool:
            raise ValueError('Original NE evaluation result required')
        feasible=profile['plan_success']
        if feasible:
            cost=profile.get('search_cost_s')
            if (not isinstance(cost,(int,float)) or isinstance(cost,bool)
                    or not math.isfinite(cost) or cost<0.
                    or not profile.get('selected_actions') or not profile.get('selected_states')):
                raise ValueError('Full original feasible NE profile/cost required')
            if incumbent is None or cost<incumbent['search_cost_s']:
                incumbent=profile;incumbent_source=source
        event=dict(source_method=source,status=profile.get('status'),profile_found=feasible,
            elapsed_s=elapsed,allocated_wall_s=allocation)
        evaluated.append(event)
        if observe is not None:observe('refinement',source,profile,event)
    elapsed=now()-start
    pending=[s for s in sources if s not in {x['source_method'] for x in generated}]
    return dict(selected_profile=incumbent,selected_source=incumbent_source,
        generator_stages=generated,refinement_stages=evaluated,
        pipeline_candidate_generation_time_s=generation_wall,
        pipeline_NE_evaluation_time_s=refinement_wall,expanded_nodes_total=nodes,
        elapsed_s=elapsed,end_to_end_budget_respected=elapsed<=budget_s,
        cumulative_generation_cap_respected=generation_wall<=generation_cap_s,
        all_candidate_families_attempted=not pending,not_attempted_candidate_families=pending,
        all_generated_candidates_NE_evaluated=sum(x['plan_found'] for x in generated)==len(evaluated),
        no_global_optimality_or_infeasibility_claim=True,
        completed_external_baseline_paths_used=False)


def restore_NE_feasible_prefix(actions, speeds, initial_state, propagate,
        checker, rigid_body_model, dynamics, *, curvatures, step_m,
        maximum_speed_m_s, terminal_speed_m_s, deceleration_m_s2,
        lookback_edges=12, maximum_iterations=12, progress_fraction_quantum=.001,
        allow_mode_changes=False, allowed_modes=None, progress=None):
    """Recover a feasible NE SEED, never declare a final goal/hold PASS.

    Infeasible *search proposals* can be ranked by their strictly valid prefix.
    No rejected edge/state enters the reusable cache. Only a fully valid seed
    is returned, and the ORIGINAL strict goal refiner must still accept it.
    Curvature-only by default; opt-in mode edits retain all original steering
    choices. The source direction sequence is never changed. Cached states
    include ALL steering/body/resource history; no phase/state is restarted.
    """
    if (not actions or len(actions)!=len(speeds) or not callable(propagate)
            or not callable(getattr(checker,'check_deadline',None))
            or type(lookback_edges) is not int or lookback_edges<=0
            or type(maximum_iterations) is not int or maximum_iterations<=0):
        raise ValueError('Explicit bounded NE-prefix restoration inputs required')
    if (not all(math.isfinite(v) and v>0. for v in
            (step_m,maximum_speed_m_s,terminal_speed_m_s,deceleration_m_s2))
            or any(not math.isfinite(s) or not 0.<s<=maximum_speed_m_s for s in speeds)
            or not math.isfinite(progress_fraction_quantum) or not 0.<progress_fraction_quantum<1.):
        raise ValueError('Original finite speed/stopping parameters required')
    curvatures=tuple(curvatures)
    if not curvatures or len(set(curvatures))!=len(curvatures) or not all(math.isfinite(k) for k in curvatures):
        raise ValueError('Original finite curvature lattice required')
    controls=[dict(a) for a in actions]
    from patient_initial_stability.steering_lattice import ALL_MODES
    modes=tuple(ALL_MODES if allowed_modes is None else allowed_modes)
    if (type(allow_mode_changes) is not bool or not modes or len(set(modes))!=len(modes)
            or any(m not in ALL_MODES for m in modes)
            or any(a.get('mode') not in modes for a in controls)):
        raise ValueError('Explicit original steering-mode scope required')
    source_modes=tuple(a['mode'] for a in controls)
    if any(a.get('curvature_1_m') not in curvatures or type(a.get('direction')) is not int
            or a['direction'] not in (-1,1) for a in controls):
        raise ValueError('Original signed source primitives required')
    log={} if progress is None else progress
    log.update(profile_evaluations=0,original_NE_edges_evaluated=0,changes=[],
        complete_seed_found=False,mode_direction_sequence_unchanged=True,
        rejected_edges_never_cached=True,final_goal_NOT_certified_here=True,
        progress_fraction_quantum_SEARCH_RANKING_ONLY=progress_fraction_quantum,
        adjacent_two_curvature_moves_eligible=True,
        original_mode_search_enabled=allow_mode_changes,allowed_original_modes=list(modes),
        direction_sequence_unchanged=True)
    if checker([(initial_state.x,initial_state.y,initial_state.yaw)]):
        log['initial_state_blocked']=True
        return None,log

    def rollout(candidate,changed=0,base=None):
        checker.check_deadline();log['profile_evaluations']+=1
        if base is not None and changed>0:
            if changed>len(base['states']) or candidate[:changed]!=base['actions'][:changed]:
                raise ValueError('Cannot reuse a changed or unverified NE prefix')
            states=list(base['states'][:changed]);state=states[-1]
        else:changed=0;states=[];state=initial_state
        for i in range(changed,len(candidate)):
            checker.check_deadline();previous=state;action=candidate[i]
            state,poses,_=propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
                step_m=step_m,speed_m_s=maximum_speed_m_s,dynamics=dynamics,
                rigid_body_model=rigid_body_model,target_speed_m_s=speeds[i])
            log['original_NE_edges_evaluated']+=1
            reason=None;fraction=0.
            if not state.rigid_body_feasible:reason='rigid_body'
            elif not getattr(poses,'resource_feasible',True):reason='motion_resource'
            elif math.hypot(state.longitudinal_velocity,state.lateral_velocity)>math.sqrt(
                    terminal_speed_m_s**2+2*deceleration_m_s2*(len(candidate)-i-1)*step_m)+1e-12:
                reason='stopping_reachability'
            elif checker(poses):
                reason='geometry';point=(previous.x,previous.y);total=clear=0.;blocked=False
                for pose in poses:
                    distance=math.hypot(pose[0]-point[0],pose[1]-point[1]);total+=distance;point=pose[:2]
                    if not blocked:
                        if checker([pose]):blocked=True
                        else:clear+=distance
                if not blocked:raise ValueError('Batch/point geometry predicates disagree')
                fraction=clear/total if total>0. else 0.
                # Ignore sub-numeric progress in SEARCH RANKING only. This
                # does not change a single collision/force/goal predicate.
                fraction=math.floor(fraction/progress_fraction_quantum+1e-12)*progress_fraction_quantum
            if reason:
                return dict(actions=candidate,states=states,complete=False,
                    score=len(states)+fraction,first_rejected_edge=i,reason=reason)
            states.append(state)
        return dict(actions=candidate,states=states,complete=True,
            score=float(len(candidate)),first_rejected_edge=None,reason=None)

    current=rollout(controls)
    log.update(initial_valid_edges=len(current['states']),initial_progress=current['score'],
        initial_rejection=current['reason'])
    for iteration in range(maximum_iterations):
        log.update(last_valid_edges=len(current['states']),last_progress=current['score'],
            last_rejection=current['reason'],iterations_completed=iteration)
        if current['complete']:break
        edge=current['first_rejected_edge'];best=current;edit=None
        # Recent actuator history first; wider existing prefixes remain eligible
        # if the local pass cannot improve. This is not an edge-feasibility gate.
        lower=max(0,edge-lookback_edges+1)
        for phase in ('recent_single','recent_pair','earlier_single'):
            if phase=='recent_pair':
                for index in reversed(range(lower,edge)):
                    first=controls[index]['curvature_1_m'];second=controls[index+1]['curvature_1_m']
                    for a in sorted(curvatures,key=lambda k:(abs(k-first),k)):
                        if a==first:continue
                        for b in sorted(curvatures,key=lambda k:(abs(k-second),k)):
                            if b==second:continue
                            proposed=[dict(v) for v in controls]
                            proposed[index]['curvature_1_m']=a;proposed[index+1]['curvature_1_m']=b
                            candidate=rollout(proposed,index,current)
                            if candidate['score']>best['score']+1e-10:
                                best=candidate;edit=[(index,a,controls[index]['mode']),(index+1,b,controls[index+1]['mode'])]
                            if best['complete']:break
                        if best['complete']:break
                    if best['complete']:break
                if edit is not None:break
                continue
            window=range(lower,edge+1) if phase=='recent_single' else range(0,lower)
            for index in reversed(tuple(window)):
                old=controls[index]['curvature_1_m']
                current_mode=controls[index]['mode']
                choices=(current_mode,)+tuple(m for m in modes if m!=current_mode) if allow_mode_changes else (current_mode,)
                for mode in choices:
                    for k in sorted(curvatures,key=lambda value:(abs(value-old),value)):
                        if k==old and mode==current_mode:continue
                        proposed=[dict(a) for a in controls]
                        proposed[index].update(curvature_1_m=k,mode=mode)
                        candidate=rollout(proposed,index,current)
                        if candidate['score']>best['score']+1e-10:
                            best=candidate;edit=[(index,k,mode)]
                        if best['complete']:break
                    if best['complete']:break
                if best['complete']:break
            if edit is not None:break
        if edit is None:break
        controls=[dict(a) for a in best['actions']];current=best
        log['changes'].append(dict(action_index=edit[0][0],curvature_1_m=edit[0][1],
            mode=edit[0][2],edits=[dict(action_index=i,curvature_1_m=k,mode=m) for i,k,m in edit],
            valid_edges=len(current['states']),valid_progress=current['score']))
        log['mode_direction_sequence_unchanged']=tuple(a['mode'] for a in controls)==source_modes
        log['source_mode_entries_changed']=sum(a['mode']!=m for a,m in zip(controls,source_modes))
    log.update(last_valid_edges=len(current['states']),last_progress=current['score'],
        last_rejection=current['reason'],iterations_completed=len(log['changes']),
        complete_seed_found=current['complete'])
    if current['complete']:
        state=current['states'][-1]
        log['final_seed_pose_local_m_rad']=[state.x,state.y,state.yaw]
        return controls,log
    return None,log


def refine_with_NE_prefix_restoration(original_refine, *args, event_sink=None,
        allow_restoration_mode_changes=False, **kwargs):
    """Original strict refiner -> bounded NE seed recovery -> SAME refiner.

    No final constraint is weakened. A failed original proposal is preserved;
    an unchanged seed is never optimised a second time as a retry. Timeout is
    rethrown with partial diagnostics copied to the explicit caller sink.
    """
    import inspect
    if event_sink is not None and not callable(event_sink):raise ValueError('Explicit diagnostic sink required')
    values=inspect.signature(original_refine).bind(*args,**kwargs)
    values.apply_defaults();v=values.arguments
    if v['terminal_extension_edges']!=0:
        raise ValueError('This reviewed restoration requires unchanged zero-extension task')
    record=dict(source_route_id=v['route_id'],original_refiner_unchanged=True,
        final_constraints_unchanged=True,restoration_attempted=False,
        allow_original_mode_restoration=allow_restoration_mode_changes)
    progress={}
    try:
        result,audit=original_refine(*args,**kwargs)
        record['original_seed_refinement_diagnostics']=audit
        if result is not None or not audit.get('collision'):
            return result,audit
        record['restoration_attempted']=True
        State=original_refine.__globals__['State']
        initial=State(0.,0.,0.,v['initial_mode'],v['initial_front_rad'],v['initial_rear_rad'],v['propagate_initial_steering'],0)
        actions,_=restore_NE_feasible_prefix(v['actions'],audit['reference_speed_profile_m_s'],initial,
            original_refine.__globals__['propagate'],v['collision_checker'],v['rigid_body_model'],v['dynamics'],
            curvatures=original_refine.__globals__['CURVATURES'],step_m=v['step_m'],
            maximum_speed_m_s=max(v['speed_levels_m_s']),terminal_speed_m_s=v['terminal_speed_tolerance_m_s'],
            deceleration_m_s2=v['terminal_deceleration_m_s2'],progress=progress,
            allow_mode_changes=allow_restoration_mode_changes)
        record['restoration']=progress
        if actions is None or actions==[dict(a) for a in v['actions']]:
            return None,dict(audit,strict_NE_prefix_restoration=record)
        updated=dict(v);updated['actions']=actions
        result,final_audit=original_refine(**updated)
        record['final_original_refinement_diagnostics']=final_audit
        return result,dict(final_audit,strict_NE_prefix_restoration=record)
    except TimeoutError:
        record.update(original_timeout_rethrown=True,partial_restoration=dict(progress))
        raise
    finally:
        if event_sink is not None:event_sink(dict(record))


def recover_NE_guided_beam(actions, speeds, references, initial_state,
        propagate, checker, rigid_body_model, dynamics, *, curvatures,
        allowed_modes, state_key, goal, step_m, maximum_speed_m_s,
        terminal_speed_m_s, deceleration_m_s2, body_radius_m,
        beam_width=12, lookahead_edges=0, progress=None, stop_at_direction_changes=False):
    """Approximate search around an OWN spatial route using ORIGINAL NE edges.

    References rank proposals, never reset or constrain the physical state.
    Only fully valid edges enter the finite beam; the complete resulting seed
    must still pass the unchanged original strict refiner. Quantised keys are
    explicitly approximate beam pruning, NOT exact physical-state dominance.
    No change to direction sequence, primitive length, speed, model or guards.
    """
    from patient_initial_stability.steering_lattice import ALL_MODES
    if type(stop_at_direction_changes) is not bool:raise ValueError('Explicit direction-stop connection policy required')
    controls=[dict(a) for a in actions];curvatures=tuple(curvatures);modes=tuple(allowed_modes)
    goal=_finite(goal,3,'unchanged NE beam goal')
    if (not controls or len(controls)!=len(speeds) or len(controls)!=len(references)
            or type(beam_width) is not int or beam_width<=0
            or type(lookahead_edges) is not int or lookahead_edges<0 or not callable(propagate)
            or not callable(state_key) or not callable(getattr(checker,'check_deadline',None))
            or not curvatures or len(set(curvatures))!=len(curvatures)
            or not all(math.isfinite(k) for k in curvatures)
            or not modes or len(set(modes))!=len(modes) or any(m not in ALL_MODES for m in modes)):
        raise ValueError('Explicit original NE primitives and bounded beam required')
    if (not all(math.isfinite(v) and v>0. for v in
            (step_m,maximum_speed_m_s,terminal_speed_m_s,deceleration_m_s2,body_radius_m))
            or any(not math.isfinite(s) or not 0.<s<=maximum_speed_m_s for s in speeds)
            or any(a.get('mode') not in modes or a.get('curvature_1_m') not in curvatures
                or type(a.get('direction')) is not int or a['direction'] not in (-1,1) for a in controls)):
        raise ValueError('Unchanged finite NE speed/stopping/action scope required')
    reference_poses=[_finite((s.x,s.y,s.yaw),3,'spatial guidance pose') for s in references]
    log={} if progress is None else progress
    log.update(beam_width=beam_width,lookahead_edges=lookahead_edges,
        lookahead_shortlist_factor=2,lookahead_shortlist_minimum_original_control_count=len(modes)*len(curvatures),
        lookahead_NE_edges_evaluated=0,
        predicted_states_never_inserted_in_beam=True,completed_layers=0,NE_edges_evaluated=0,
        rejected=dict(rigid_body=0,motion_resource=0,stopping_reachability=0,geometry=0),
        complete_seed_found=False,final_goal_NOT_certified_here=True,
        reference_poses_are_ranking_only_never_state_resets=True,
        direction_sequence_unchanged=True,approximate_beam_pruning_NOT_exact_dominance=True,
        all_original_modes=list(modes),all_original_curvatures=list(curvatures))
    if stop_at_direction_changes:log['rejected']['cusp_connection']=0
    if checker([(initial_state.x,initial_state.y,initial_state.yaw)]):
        log['initial_state_blocked']=True;return None,log
    wrap=lambda value:math.atan2(math.sin(value),math.cos(value))
    # (full original state, complete parent controls, accumulated tracking cost,
    #  exact elapsed time). No state field is reconstructed from a reference.
    beam=[(initial_state,(),0.,0.)]
    for index,(source,speed,reference) in enumerate(zip(controls,speeds,reference_poses)):
        checker.check_deadline();labels={}
        choices=(source['mode'],)+tuple(m for m in modes if m!=source['mode'])
        values=sorted(curvatures,key=lambda k:(abs(k-source['curvature_1_m']),k))
        for previous,chain,tracking,elapsed in beam:
            for mode in choices:
                for curvature in values:
                    checker.check_deadline()
                    state,poses,edge_time=propagate(previous,mode,curvature,source['direction'],
                        step_m=step_m,speed_m_s=maximum_speed_m_s,dynamics=dynamics,
                        rigid_body_model=rigid_body_model,target_speed_m_s=speed)
                    log['NE_edges_evaluated']+=1
                    if not math.isfinite(edge_time) or edge_time<=0.:
                        raise ValueError('Original NE edge elapsed time must be finite positive')
                    reason=None
                    if not state.rigid_body_feasible:reason='rigid_body'
                    elif not getattr(poses,'resource_feasible',True):reason='motion_resource'
                    elif math.hypot(state.longitudinal_velocity,state.lateral_velocity)>math.sqrt(
                            terminal_speed_m_s**2+2*deceleration_m_s2*(direction_remaining_arc(controls,index,step_m)
                                if stop_at_direction_changes else (len(controls)-index-1)*step_m))+1e-12:
                        reason='stopping_reachability'
                    elif stop_at_direction_changes and not stop_connected_edge(previous,state,controls,index,curvature_1_m=curvature):reason='cusp_connection'
                    elif checker(poses):reason='geometry'
                    if reason:log['rejected'][reason]+=1;continue
                    pose=_finite((state.x,state.y,state.yaw),3,'original NE edge pose')
                    error=(pose[0]-reference[0])**2+(pose[1]-reference[1])**2+(
                        body_radius_m*wrap(pose[2]-reference[2]))**2
                    accumulated=tracking+step_m*error
                    ranking=error+accumulated
                    key=(state.mode,state.direction,*state_key(state),
                        *(getattr(state,name,0) for name in
                          ('mission_reverse_episode_bins','mission_reverse_total_bins','mission_direction_changes')))
                    action=dict(mode=mode,curvature_1_m=curvature,direction=source['direction'])
                    entry=(ranking,accumulated,elapsed+edge_time,state,(*chain,action))
                    old=labels.get(key)
                    if old is None or entry[:3]<old[:3]:labels[key]=entry
        if not labels:
            log['frontier_exhausted_at_edge']=index;return None,log
        ordered=sorted(labels.values(),key=lambda e:e[:3])
        if lookahead_edges:
            # Bounded anticipatory ranking ONLY: predict the original future
            # source controls. A blocked prediction does not reject the actual
            # current edge or prove a frontier infeasible. Predicted states are
            # NEVER inserted in the beam or used as a reset/acceptance proof.
            def forecast(entry):
                state=entry[3];valid=0
                for j in range(index+1,min(len(controls),index+1+lookahead_edges)):
                    checker.check_deadline();a=controls[j]
                    state,poses,_=propagate(state,a['mode'],a['curvature_1_m'],a['direction'],
                        step_m=step_m,speed_m_s=maximum_speed_m_s,dynamics=dynamics,
                        rigid_body_model=rigid_body_model,target_speed_m_s=speeds[j])
                    log['lookahead_NE_edges_evaluated']+=1
                    if (not state.rigid_body_feasible or not getattr(poses,'resource_feasible',True)
                            or math.hypot(state.longitudinal_velocity,state.lateral_velocity)>math.sqrt(
                                terminal_speed_m_s**2+2*deceleration_m_s2*(len(controls)-j-1)*step_m)+1e-12
                            or checker(poses)):break
                    valid+=1
                return valid
            shortlist=ordered[:max(2*beam_width,len(modes)*len(curvatures))]
            ranked=[(-forecast(e),e[:3],i,e) for i,e in enumerate(shortlist)]
            kept=[item[-1] for item in sorted(ranked,key=lambda item:item[:3])[:beam_width]]
        else:kept=ordered[:beam_width]
        beam=[(e[3],e[4],e[1],e[2]) for e in kept]
        log.update(completed_layers=index+1,last_frontier_unique_labels=len(labels),
            last_best_tracking_rank=kept[0][0])
    selected=min(beam,key=lambda e:(100.*math.hypot(e[0].x-goal[0],e[0].y-goal[1])+
        60.*abs(wrap(e[0].yaw-goal[2])),e[2],e[3]))
    state,chain,_,_=selected
    log.update(complete_seed_found=True,final_seed_pose_local_m_rad=[state.x,state.y,state.yaw],
        final_seed_position_error_m=math.hypot(state.x-goal[0],state.y-goal[1]),
        final_seed_heading_error_rad=wrap(state.yaw-goal[2]),
        final_seed_speed_m_s=math.hypot(state.longitudinal_velocity,state.lateral_velocity))
    return [dict(a) for a in chain],log


def neutral_direction_bridges(actions):
    """Explicit native straight out/back pair at each non-neutral cusp.

    These are REAL propagated candidate controls, not a zero-motion state
    reset or a changed goal. In the spatial candidate the equal opposite
    zero-curvature steps close at the same pose. NE refinement must still
    repropagate the whole chain, brake at every cusp and pass all original
    force, resource, geometry and terminal checks. The extra reverse travel
    is charged normally; a resource-exceeding candidate cannot be admitted.
    """
    from patient_initial_stability.steering_lattice import ALL_MODES,CURVATURES
    if not actions:raise ValueError('Complete native candidate actions required')
    controls=[dict(a) for a in actions]
    if any(a.get('mode') not in ALL_MODES or a.get('curvature_1_m') not in CURVATURES
            or type(a.get('direction')) is not int or a['direction'] not in (-1,1) for a in controls):
        raise ValueError('Original mode/curvature/direction primitives required')
    result=[];inserted=[]
    for index,action in enumerate(controls):
        if (index and controls[index-1]['direction']!=action['direction']
                and (controls[index-1]['curvature_1_m']!=0. or action['curvature_1_m']!=0.)):
            inserted.append(dict(original_direction_change_index=index,new_pair_first_index=len(result)))
            result.extend((dict(controls[index-1],curvature_1_m=0.),dict(action,curvature_1_m=0.)))
        result.append(action)
    return result,dict(inserted_zero_curvature_pairs=inserted,additional_native_edges=2*len(inserted),
        same_spatial_goal_NOT_a_state_injection=True,NE_repropagation_and_all_original_checks_STILL_required=True)


def direction_remaining_arc(actions,index,step_m):
    """Remaining nominal arc to the NEXT direction change, not just the goal."""
    end=index+1
    while end<len(actions) and actions[end]['direction']==actions[index]['direction']:end+=1
    return (end-index-1)*step_m


def stop_connected_edge(previous,state,actions,index,*,curvature_1_m=None):
    """Original stop/neutral bounds at a fixed, continuously propagated cusp."""
    def stopped(s):
        return (math.hypot(s.longitudinal_velocity,s.lateral_velocity)<=.005+1e-12
            and max(abs(s.front),abs(s.rear))<=math.radians(2.5)+1e-12)
    entering=index>0 and actions[index-1]['direction']!=actions[index]['direction']
    leaving=index+1<len(actions) and actions[index+1]['direction']!=actions[index]['direction']
    curvature=actions[index]['curvature_1_m'] if curvature_1_m is None else curvature_1_m
    return not ((entering and (not stopped(previous) or curvature!=0.))
        or (leaving and (not stopped(state) or curvature!=0.)))


def mode_curvature_refinement_binding(original_refine, *, allowed_modes,stop_at_direction_changes=False):
    """Explore native MODE/curvature pairs; preserve the entire NE evaluator.

    The old local refiner could edit curvature but froze the seed's modes.
    This optional planning-only neighbourhood permits the original modes,
    with original propagation, speed profile, SAT, force/resource checks and
    strict goal predicate unchanged. It does not change a live controller.
    """
    import ast,copy,hashlib,inspect,textwrap
    from patient_initial_stability.steering_lattice import ALL_MODES
    modes=tuple(allowed_modes)
    if type(stop_at_direction_changes) is not bool:raise ValueError('Explicit cusp-stop refinement policy required')
    if not modes or len(set(modes))!=len(modes) or any(m not in ALL_MODES for m in modes):
        raise ValueError('Explicit subset of original steering modes required')
    source=textwrap.dedent(inspect.getsource(original_refine));tree=ast.parse(source)
    found=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
        and n.target.id=='curvature' and isinstance(n.iter,ast.Name) and n.iter.id=='CURVATURES']
    if len(found)!=1:raise ValueError('One original local curvature neighbourhood required')
    before=found[0];inner=copy.deepcopy(before)
    expected=ast.dump(ast.parse('curvature == current_curvature',mode='eval').body,include_attributes=False)
    checks=[n for n in ast.walk(inner) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==expected]
    assignments=[n for n in ast.walk(inner) if isinstance(n,ast.Assign) and len(n.targets)==1
        and ast.dump(n.targets[0],include_attributes=False)==ast.dump(ast.parse('candidate_actions[action_index]["curvature_1_m"] = curvature').body[0].targets[0],include_attributes=False)]
    changes=[n for n in ast.walk(inner) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='best_change' for t in n.targets)]
    if len(checks)!=1 or len(assignments)!=1 or len(changes)!=1:
        raise ValueError('Exact original equality/edit/diagnostic statements required')
    checks[0].test=ast.parse('curvature == current_curvature and _refine_mode == refined_actions[action_index]["mode"]',mode='eval').body
    edit=assignments[0];index=inner.body.index(edit)
    inner.body.insert(index+1,ast.parse('candidate_actions[action_index]["mode"] = _refine_mode').body[0])
    changes[0].value=ast.parse('(action_index, _refine_mode, curvature)',mode='eval').body
    outer=ast.For(target=ast.Name(id='_refine_mode',ctx=ast.Store()),
        iter=ast.parse('_native_refinement_modes(refined_actions[action_index]["mode"])',mode='eval').body,
        body=[inner],orelse=[])
    class Replace(ast.NodeTransformer):
        def generic_visit(self,node):
            if node is before:return outer
            return super().generic_visit(node)
    tree=Replace().visit(tree)
    cusp_edits=[];cusp_guard=None
    if stop_at_direction_changes:
        for node in ast.walk(tree):
            if (isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                    and node.targets[0].id=='remaining_arc_m'):
                for variable in ('refined_actions','candidate_actions'):
                    value=ast.parse('(len('+variable+') - action_index - 1) * step_m',mode='eval').body
                    if ast.dump(node.value,include_attributes=False)==ast.dump(value,include_attributes=False):
                        cusp_edits.append((node,node.value))
                        node.value=ast.parse('_mission_remaining_arc('+variable+',action_index,step_m)',mode='eval').body
        evaluation=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate']
        if len(cusp_edits)!=3 or len(evaluation)!=1:raise ValueError('All three original arc/braking decisions and one evaluator required')
        loop=[n for n in ast.walk(evaluation[0]) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='action_index']
        if len(loop)!=1:raise ValueError('One original propagated-edge evaluator loop required')
        conditions=[n for n in loop[0].body if isinstance(n,ast.If)
            and ast.dump(n.test,include_attributes=False)==ast.dump(ast.parse('not state.rigid_body_feasible',mode='eval').body,include_attributes=False)]
        if len(conditions)!=1:raise ValueError('Original force guard required before adding a stricter cusp connection')
        cusp_guard=ast.parse('if not _mission_stop_connected(previous_state,state,candidate_actions,action_index):\n diagnostics["cusp_connection"]=diagnostics.get("cusp_connection",0)+1\n valid=False\n break').body[0]
        loop[0].body.insert(loop[0].body.index(conditions[0]),cusp_guard)
    def family(current):
        if current not in modes:raise ValueError('Seed mode outside the explicit original mode family')
        return (current,)+tuple(m for m in modes if m!=current)
    namespace=dict(original_refine.__globals__,_native_refinement_modes=family,
        _mission_remaining_arc=direction_remaining_arc,_mission_stop_connected=stop_connected_edge)
    exec(compile(ast.fix_missing_locations(tree),'<private_native_mode_curvature_refinement>','exec'),namespace)
    result=namespace[original_refine.__name__]
    class Undo(ast.NodeTransformer):
        def generic_visit(self,node):
            if node is outer:return before
            if node is cusp_guard:return None
            return super().generic_visit(node)
    restored=Undo().visit(tree)
    for node,value in cusp_edits:node.value=value
    if ast.dump(restored,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
        raise ValueError('Undo must recover the entire original refinement AST')
    result.mode_refinement_binding=dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        allowed_native_modes=list(modes),only_local_mode_curvature_neighbourhood_changed=True,
        undo_recovers_entire_original_refiner_AST=True,original_evaluator_and_strict_goal_unchanged=not stop_at_direction_changes,
        original_NE_SAT_force_and_strict_final_goal_unchanged=True,
        stricter_cusp_stop_connections=stop_at_direction_changes,
        all_three_original_arc_decisions_bound_to_next_cusp=stop_at_direction_changes,
        no_actuation_or_new_motion_authority=True)
    return result


def refine_with_NE_guided_beam(original_refine, *args, source_dynamics,
        body_radius_m, beam_width=12, lookahead_edges=0, event_sink=None, mode_refinement=False,
        stop_at_direction_changes=False, initial_state=None, **kwargs):
    """Own spatial reference -> bounded NE beam seed -> ORIGINAL strict refiner.

    No cached external baseline input or budget is introduced here. The caller
    must charge own generation, reference replay, beam and final refinement to
    its SAME end-to-end deadline. This helper grants no live motion authority.
    """
    import inspect
    from patient_initial_stability.steering_lattice import ALL_MODES
    if type(mode_refinement) is not bool:raise ValueError('Explicit mode-refinement selection required')
    if type(stop_at_direction_changes) is not bool or stop_at_direction_changes and not mode_refinement:
        raise ValueError('Explicit mode-aware refinement required for the stricter cusp-stop candidate')
    if event_sink is not None and not callable(event_sink):raise ValueError('Explicit diagnostic sink required')
    values=inspect.signature(original_refine).bind(*args,**kwargs);values.apply_defaults();v=values.arguments
    if v['terminal_extension_edges']!=0:raise ValueError('No hidden terminal extension in guided NE search')
    ns=original_refine.__globals__;State=ns['State'];propagate=ns['propagate']
    controls=[dict(mode=a['mode'],curvature_1_m=a['curvature_1_m'],direction=a['direction']) for a in v['actions']]
    record=dict(source_route_id=v['route_id'],original_model_and_final_refiner_unchanged=not mode_refinement,
        original_model_and_strict_final_goal_unchanged=True,
        additional_cusp_stop_connections=stop_at_direction_changes,
        reference_from_same_own_actions_only=True,source_dynamics=source_dynamics.name,
        reference_replay_charged_to_deadline=True,final_constraints_unchanged=True)
    progress={}
    try:
        reference=State(0.,0.,0.,v['initial_mode'],v['initial_front_rad'],v['initial_rear_rad'],False,0)
        initial=State(0.,0.,0.,v['initial_mode'],v['initial_front_rad'],v['initial_rear_rad'],v['propagate_initial_steering'],0)
        if initial_state is not None:
            if (type(initial_state) is not State or initial_state.started is not True
                    or initial_state.rigid_body_feasible is not True
                    or (v['initial_mode'],v['initial_front_rad'],v['initial_rear_rad'],v['propagate_initial_steering'])
                    !=(initial_state.mode,initial_state.front,initial_state.rear,True)):
                raise ValueError('Exact complete continued NE/resource state and matching steering required')
            initial=reference=initial_state
            record['continued_complete_initial_state']=__import__('dataclasses').asdict(initial_state)
        maximum=max(v['speed_levels_m_s']);references=[];speeds=[];checker=v['collision_checker']
        for i,a in enumerate(controls):
            checker.check_deadline()
            reference,poses,_=propagate(reference,a['mode'],a['curvature_1_m'],a['direction'],
                step_m=v['step_m'],speed_m_s=maximum,dynamics=source_dynamics,
                rigid_body_model=None,target_speed_m_s=maximum)
            if not getattr(poses,'resource_feasible',True) or checker(poses):
                raise ValueError('Own spatial reference no longer passes original map/resource predicates')
            references.append(reference)
            remaining=(direction_remaining_arc(controls,i,v['step_m']) if stop_at_direction_changes else (len(controls)-i-1)*v['step_m'])
            cap=math.sqrt(v['terminal_speed_tolerance_m_s']**2+2*v['terminal_deceleration_m_s2']*remaining)
            levels=[s for s in v['speed_levels_m_s'] if s<=cap+1e-12]
            speeds.append(max(levels) if levels else min(v['speed_levels_m_s']))
        actions,_=recover_NE_guided_beam(controls,speeds,references,initial,propagate,
            checker,v['rigid_body_model'],v['dynamics'],curvatures=ns['CURVATURES'],
            allowed_modes=ALL_MODES,
            state_key=ns['_state_key'],goal=v['goal'],step_m=v['step_m'],maximum_speed_m_s=maximum,
            terminal_speed_m_s=v['terminal_speed_tolerance_m_s'],deceleration_m_s2=v['terminal_deceleration_m_s2'],
            body_radius_m=body_radius_m,beam_width=beam_width,lookahead_edges=lookahead_edges,progress=progress,
            stop_at_direction_changes=stop_at_direction_changes)
        record['beam']=progress
        if actions is None:return None,dict(strict_NE_guided_beam=record)
        updated=dict(v);updated['actions']=actions
        continued=(continue_original_lattice_from_state(original_refine,initial)
            if initial_state is not None else original_refine)
        refine=mode_curvature_refinement_binding(continued,allowed_modes=ALL_MODES,
            stop_at_direction_changes=stop_at_direction_changes) if mode_refinement else continued
        if mode_refinement:record['mode_refinement_binding']=refine.mode_refinement_binding
        result,audit=refine(**updated)
        record['final_original_refinement_diagnostics']=audit
        return result,dict(audit,strict_NE_guided_beam=record)
    except TimeoutError:
        record.update(original_timeout_rethrown=True,partial_beam=dict(progress));raise
    finally:
        if event_sink is not None:event_sink(dict(record))
