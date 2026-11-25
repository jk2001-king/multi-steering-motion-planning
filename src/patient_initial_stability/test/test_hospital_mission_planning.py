"""Mission scope bindings, not a steering/physical mission certificate."""

import math
import pytest

from hospital_mission_planning import mission_search_bounds,bounded_lattice_function,bind_plan_method
from hospital_mission_planning import (MotionResourceLimits,resource_propagation_binding,
    resource_checker_class,bind_resource_kernels,bidirectional_search_guidance,with_direction_change_cost)
from hospital_mission_planning import mission_local_heading_guidance
from hospital_mission_planning import ResourceParetoBest,pareto_resource_lattice_function
from hospital_mission_planning import owned_candidate_portfolio
from hospital_mission_planning import restore_NE_feasible_prefix,refine_with_NE_prefix_restoration
from hospital_mission_planning import recover_NE_guided_beam,refine_with_NE_guided_beam
from hospital_mission_planning import mode_curvature_refinement_binding


def test_mode_refinement_changes_only_native_planning_neighbourhood():
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route as original
    from patient_initial_stability.steering_lattice import ALL_MODES
    bound=mode_curvature_refinement_binding(original,allowed_modes=ALL_MODES)
    assert bound.mode_refinement_binding['undo_recovers_entire_original_refiner_AST']
    assert bound.mode_refinement_binding['original_evaluator_and_strict_goal_unchanged']
    with pytest.raises(ValueError,match='empty'):
        bound('SOFTWARE_ONLY',[],(0.,0.,0.),(.003,.05),None,lambda poses:False,.005,.020)


@pytest.mark.parametrize('modes',[(),('NOT_A_NATIVE_MODE',),('front_only','front_only')])
def test_mode_refinement_rejects_undeclared_or_duplicate_mode_family(modes):
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route as original
    with pytest.raises(ValueError):mode_curvature_refinement_binding(original,allowed_modes=modes)


def test_internal_stop_refinement_keeps_original_guards_and_does_not_reset_state():
    from types import SimpleNamespace
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route as original
    from patient_initial_stability.steering_lattice import ALL_MODES
    from hospital_mission_planning import direction_remaining_arc,stop_connected_edge
    bound=mode_curvature_refinement_binding(original,allowed_modes=ALL_MODES,stop_at_direction_changes=True)
    assert bound.mode_refinement_binding['undo_recovers_entire_original_refiner_AST']
    assert bound.mode_refinement_binding['original_NE_SAT_force_and_strict_final_goal_unchanged']
    actions=[dict(direction=d,curvature_1_m=0.) for d in (1,1,-1,-1,1)]
    assert [direction_remaining_arc(actions,i,.05) for i in range(5)]==[.05,0.,.05,0.,0.]
    state=SimpleNamespace(longitudinal_velocity=.003,lateral_velocity=0.,front=0.,rear=0.)
    moving=SimpleNamespace(longitudinal_velocity=.050,lateral_velocity=0.,front=0.,rear=0.)
    assert stop_connected_edge(state,state,actions,1)
    assert not stop_connected_edge(state,moving,actions,1)
    assert not stop_connected_edge(moving,state,actions,2)
    assert stop_connected_edge(state,state,actions,2)
    actions[1]['curvature_1_m']=-.7
    assert not stop_connected_edge(state,state,actions,1)
    assert stop_connected_edge(state,state,actions,1,curvature_1_m=0.)
    assert moving.longitudinal_velocity==.050


def test_neutral_bridge_is_explicit_closed_native_controls_not_state_snapping():
    from hospital_mission_planning import neutral_direction_bridges
    from patient_initial_stability import steering_lattice as lattice
    source=([dict(mode='coordinated_counterphase',curvature_1_m=-.5,direction=1)]*10
        +[dict(mode='front_only',curvature_1_m=.5,direction=-1)]*3
        +[dict(mode='coordinated_counterphase',curvature_1_m=-.5,direction=1)]*4)
    controls,proof=neutral_direction_bridges(source)
    assert proof['additional_native_edges']==4 and len(controls)==len(source)+4
    assert neutral_direction_bridges(controls)[0]==controls
    def endpoint(actions):
        state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
        for action in actions:
            state,_,_=lattice.propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
                speed_m_s=.05,dynamics=lattice.KINEMATIC_DYNAMICS)
        return state
    before,after=endpoint(source),endpoint(controls)
    assert (before.x,before.y,before.yaw)==pytest.approx((after.x,after.y,after.yaw),abs=1e-12)
from hospital_mission_planning import plan_NE_mission_candidate
from hospital_mission_planning import ExactRigidBodyStepCache
from hospital_mission_planning import row_skipping_native_checker
from hospital_mission_planning import compiled_original_NE_propagation
from hospital_mission_planning import OwnedSpatialReferenceGuidance,NEStraightArcTimeGuidance
from hospital_mission_planning import TerminalPoseConnectionGuidance
from hospital_mission_planning import NETerminalTimeGuidance
from hospital_mission_planning import bounded_NE_terminal_beam
from hospital_mission_planning import copy_original_search_prefixes
from hospital_mission_planning import sparse_cell_native_checker
from hospital_mission_planning import continue_original_lattice_from_state
from hospital_mission_planning import installed_reeds_shepp_metric
from hospital_mission_planning import reference_corridor_checker
from hospital_mission_planning import plan_NE_region_connection,hierarchical_NE_reference_connections
from hospital_mission_planning import ReferenceSteeringPreparation
from hospital_mission_planning import rate_aware_spatial_reference
from hospital_mission_planning import diagnostic_rolling_resolution
from hospital_mission_planning import common_v3_terminal_geometry_review
from hospital_mission_planning import (forward_common_v3_execution_binding,mission_nominal_v3_case_binding,
    model_clearance_pose_comparison,conditional_tracking_reserve_definition,
    additional_planning_reserve_checker,additional_planning_reserve_audit,replay_saved_complete_NE_prefix,
    audit_complete_NE_terminal_segment,goal_predicate_separating_state_key)
from hospital_mission_planning import planar_reference_to_com_review
from hospital_mission_planning import CoMTransportAugmentedEdgeModel
from hospital_mission_planning import first_terminal_neighborhood_prefix


def _augmented_com_fixture(**changes):
    from dataclasses import replace
    from patient_initial_stability.rigid_body_edge import RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    b=replace(RigidBodyParameterBounds(**model_bounds()),**changes)
    return CoMTransportAugmentedEdgeModel(b,
        residual_reference_definition='base_reference_body_derivative_residuals_v1',
        identification_yaw_radius_m=.533)


@pytest.mark.parametrize('operation',['add','sub','mul','div'])
def test_com_interval_scalar_operations_enclose_exact_rational_corner_results(operation):
    import random
    from fractions import Fraction
    from hospital_mission_planning import (_frame_interval_add,_frame_interval_sub,
        _frame_interval_mul,_frame_interval_div)
    generator=random.Random(681)
    function={'add':_frame_interval_add,'sub':_frame_interval_sub,
        'mul':_frame_interval_mul,'div':_frame_interval_div}[operation]
    for _ in range(100):
        first=tuple(sorted(generator.uniform(-2.,2.) for _ in range(2)))
        second=tuple(sorted(generator.uniform(.001,2.) if operation=='div' else generator.uniform(-2.,2.) for _ in range(2)))
        low,high=function(first,second);corners=[]
        for a in first:
            for b in second:
                x,y=Fraction(a),Fraction(b)
                corners.append({'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'div':lambda:x/y}[operation]())
        assert Fraction(low)<=min(corners) and Fraction(high)>=max(corners)


@pytest.mark.parametrize('motion',[(.03,.001,.035,.05,.005,.04,.05),
    (.00,.00,.00,.05,.00,.03,.02),(-.03,.005,-.2,-.05,-.01,-.3,.01),
    (.05,-.02,.2,.00,.02,-.03,.05)])
def test_augmented_com_intervals_enclose_independent_continuous_parameter_and_residual_samples(motion):
    import itertools
    import numpy as np
    from patient_initial_stability.rigid_body_edge import GRAVITY_M_S2
    model=_augmented_com_fixture();legacy,review=model.review_step(*motion);b=model.bounds
    r=motion[2];g=GRAVITY_M_S2;J=np.array([[0.,-1.],[1.,0.]])
    random=np.random.default_rng(7128)
    ranges=[(b.com_longitudinal_min_m,b.com_longitudinal_max_m),
        (b.com_lateral_min_m,b.com_lateral_max_m),(b.com_height_min_m,b.com_height_max_m),
        (b.yaw_radius_min_m,b.yaw_radius_max_m)]
    ranges+= [(-v,v) for v in model.residual_derivative_box]
    samples=list(itertools.product(*ranges))
    samples.extend([[float(random.uniform(lo,hi)) for lo,hi in ranges] for _ in range(100)])
    contains=lambda interval,value:interval[0]-1e-12<=value<=interval[1]+1e-12
    for cx,cy,height,rho,eu,ev,ea in samples:
        alpha=legacy.yaw_accel_rad_s2+ea;c=np.array([cx,cy])
        # Independent vector cross-product point transport and two equilibrium
        # linear solves, not the interval evaluator's component expressions.
        reference=np.array([legacy.longitudinal_accel_m_s2+eu,legacy.lateral_accel_m_s2+ev])
        force=reference+alpha*(J@c)+r*(J@(r*(J@c)))
        ng=rho*rho*alpha;af=b.front_support_x_m-cx;ar=cx-b.rear_support_x_m
        matrix=np.array([[1.,1.],[af,-ar]])
        normals=np.linalg.solve(matrix,np.array([g,-height*force[0]]))
        lateral=np.linalg.solve(matrix,np.array([force[1],ng]))
        assert contains(review['com_specific_force_x_interval'],force[0])
        assert contains(review['com_specific_force_y_interval'],force[1])
        assert contains(review['specific_yaw_moment_about_com_interval'],ng)
        assert contains(review['front_normal_interval'],normals[0])
        assert contains(review['rear_normal_interval'],normals[1])
        assert contains(review['front_lateral_specific_force_interval'],lateral[0])
        assert contains(review['rear_lateral_specific_force_interval'],lateral[1])
        if min(normals)>0.:
            longitudinal=force[0]*normals/g
            util=np.hypot(longitudinal,lateral)/(b.friction_min*normals)
            assert review['front_friction_upper']>=util[0]-1e-12
            assert review['rear_friction_upper']>=util[1]-1e-12
        ltr=2./b.track_width_m*abs(cy+height*force[1]/g)
        assert review['abs_ltr_upper']>=ltr-1e-12
        if min(normals)>0. and ltr<=1.:
            fraction=min(normals)/g*.5*(1.-ltr)
            assert review['minimum_normal_fraction_lower']<=fraction+1e-12


@pytest.mark.parametrize('direction',[-1.,1.])
def test_augmented_com_preserves_exact_response_and_never_weakens_original_acceptance(direction):
    import numpy as np
    random=np.random.default_rng(318);model=_augmented_com_fixture()
    fields=('longitudinal_velocity_m_s','lateral_velocity_m_s','yaw_rate_rad_s',
        'longitudinal_accel_m_s2','lateral_accel_m_s2','yaw_accel_rad_s2','actuator_utilization')
    for _ in range(40):
        args=(*[float(v) for v in random.uniform(-.1,.1,3)],direction*.05,.01,.03,.025)
        old=model.original.step(*args);new=model.step(*args)
        assert all(getattr(new,name)==getattr(old,name) for name in fields)
        assert not new.feasible or old.feasible
        assert new.friction_utilization>=old.friction_utilization
        assert new.front_friction_utilization>=old.front_friction_utilization
        assert new.rear_friction_utilization>=old.rear_friction_utilization
        assert new.abs_ltr>=old.abs_ltr and new.penalty_s>=old.penalty_s
        assert new.minimum_normal_load_fraction<=old.minimum_normal_load_fraction
    assert model.statistics['step_calls']==40
    assert model.parameters['not_default_or_adopted_runtime_or_final_comparison']


def test_augmented_com_zero_offset_zero_residual_agrees_with_centered_aggregate_model():
    changes=dict(com_longitudinal_min_m=0.,com_longitudinal_max_m=0.,com_lateral_min_m=0.,com_lateral_max_m=0.,
        com_height_min_m=.27,com_height_max_m=.27,yaw_radius_min_m=.533,yaw_radius_max_m=.533,
        caster_specific_force_x_abs_max_m_s2=0.,caster_specific_force_y_abs_max_m_s2=0.,
        caster_specific_yaw_moment_abs_max_m2_s2=0.)
    model=_augmented_com_fixture(**changes);args=(.03,.005,.035,.05,.008,.04,.05)
    old=model.original.step(*args);new=model.step(*args)
    assert old.feasible and new.feasible
    for field in ('friction_utilization','front_friction_utilization','rear_friction_utilization',
        'abs_ltr','minimum_normal_load_fraction','penalty_s'):
        assert getattr(old,field)==pytest.approx(getattr(new,field),rel=1e-12,abs=1e-14)


def test_augmented_com_detects_synthetic_offset_angular_acceleration_that_original_ltr_omits():
    # Intentionally tiny LTR threshold is a synthetic mechanism witness,
    # NOT a claim of activation at the hospital's unchanged0.8 threshold.
    model=_augmented_com_fixture(com_longitudinal_min_m=.09,com_longitudinal_max_m=.09,
        com_lateral_min_m=0.,com_lateral_max_m=0.,max_abs_ltr=.001,
        caster_specific_force_x_abs_max_m_s2=0.,caster_specific_force_y_abs_max_m_s2=0.,
        caster_specific_yaw_moment_abs_max_m2_s2=0.)
    args=(0.,0.,0.,0.,0.,.25,.02)
    old=model.original.step(*args);new=model.step(*args)
    assert old.feasible and old.abs_ltr==0.
    assert not new.feasible and new.abs_ltr>.001
    assert model.statistics['additional_guard_rejected_steps']==1


def test_augmented_com_unbounded_support_candidate_is_rejected_without_zero_weight_nan():
    model=_augmented_com_fixture(com_height_min_m=100.,com_height_max_m=100.,
        friction_cost_weight_s=0.,ltr_cost_weight_s=0.,actuator_cost_weight_s=0.)
    step=model.step(0.,0.,.8,.05,0.,.9,.05)
    assert not step.feasible and math.isinf(step.friction_utilization)
    assert step.minimum_normal_load_fraction<=0. and step.penalty_s==0. and not math.isnan(step.penalty_s)


@pytest.mark.parametrize('fault',['wrong_convention','zero_radius','nan_radius','boolean_radius','tiny_radius','nan_bounds','zero_dt','nan_velocity'])
def test_augmented_com_requires_explicit_finite_residual_convention_and_inputs(fault):
    from dataclasses import replace
    from hospital_h3_turn_planning import model_bounds
    from patient_initial_stability.rigid_body_edge import RigidBodyParameterBounds
    b=RigidBodyParameterBounds(**model_bounds());definition='base_reference_body_derivative_residuals_v1';radius=.533
    if fault=='wrong_convention':definition='unmeasured_external_caster_force'
    elif fault=='zero_radius':radius=0.
    elif fault=='nan_radius':radius=float('nan')
    elif fault=='boolean_radius':radius=True
    elif fault=='tiny_radius':radius=1e-300
    elif fault=='nan_bounds':b=replace(b,friction_min=float('nan'))
    with pytest.raises(ValueError):
        model=CoMTransportAugmentedEdgeModel(b,residual_reference_definition=definition,identification_yaw_radius_m=radius)
        model.step(float('nan') if fault=='nan_velocity' else .03,0.,.03,.05,0.,.03,0. if fault=='zero_dt' else .05)


def test_augmented_com_same_controls_preserve_original_nominal_pose_time_and_steering():
    from patient_initial_stability import steering_lattice as lattice
    model=_augmented_com_fixture();state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    old=state;new=state
    for mode,k,d,speed in [('front_only',.5,1,.03),('rear_only',-.25,1,.03),
            ('coordinated_counterphase',.0,-1,.003)]:
        old,old_poses,old_time=lattice.propagate(old,mode,k,d,speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,
            rigid_body_model=model.original,target_speed_m_s=speed)
        new,new_poses,new_time=lattice.propagate(new,mode,k,d,speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,
            rigid_body_model=model,target_speed_m_s=speed)
        assert old.rigid_body_feasible and new.rigid_body_feasible
        assert new_poses==old_poses and new_time==old_time
        assert all(getattr(old,name)==getattr(new,name) for name in ('x','y','yaw','front','rear',
            'longitudinal_velocity','lateral_velocity','yaw_rate','mode','direction','started'))


def test_original_compiled_adapter_cannot_silently_bypass_augmented_com_guard():
    pytest.importorskip('numba')
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel
    compiled=compiled_original_NE_propagation();model=_augmented_com_fixture()
    assert not isinstance(model,MassNormalizedRigidBodyEdgeModel)
    state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    result,_,_=compiled(state,'front_only',.5,1,speed_m_s=.05,
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=.03)
    assert result.rigid_body_feasible and model.statistics['step_calls']>0
    assert compiled.compilation_statistics==dict(compiled_calls=0,original_fallback_calls=1)


@pytest.mark.parametrize('offset',[(0.,0.),(.093,-.05),(-.091,.05)])
@pytest.mark.parametrize('yaw',[-1.2,0.,.7])
def test_point_frame_review_matches_independent_world_position_second_difference(offset,yaw):
    import numpy as np
    # Construct G in WORLD coordinates independently, then differentiate its
    # position; do not assert the implementation against the same formula.
    w=.035;alpha=.020;world_v=np.array([.03,.005]);world_a=np.array([.020,.003])
    rotate=lambda angle:np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    uv=rotate(yaw).T@world_v;aO=rotate(yaw).T@world_a
    body_dot=aO-w*np.array([-uv[1],uv[0]])
    result=planar_reference_to_com_review([*uv,w],[*body_dot,alpha],offset,yaw_radius_com_m=.533)
    def position(t):return world_v*t+.5*world_a*t*t+rotate(yaw+w*t+.5*alpha*t*t)@np.array(offset)
    h=1e-4;acc=(position(h)-2*position(0.)+position(-h))/(h*h)
    vel=(position(h)-position(-h))/(2*h)
    assert result['com_specific_force_in_body_m_s2']==pytest.approx(rotate(yaw).T@acc,abs=1e-8)
    assert result['com_body_velocity'][:2]==pytest.approx(rotate(yaw).T@vel,abs=1e-10)
    cx,cy=offset;fx,fy=result['com_specific_force_in_body_m_s2']
    assert result['specific_yaw_moment_about_reference_m2_s2']==pytest.approx(.533**2*alpha+cx*fy-cy*fx)
    assert result['specific_yaw_moment_about_reference_m2_s2']==pytest.approx(
        result['parallel_axis_yaw_radius_squared_m2']*alpha+cx*aO[1]-cy*aO[0])
    assert not result['actual_motion_authorized'] and result['frozen_planning_propagation_and_acceptance_NOT_changed']


def test_point_frame_review_round_trip_and_nonzero_centripetal_and_angular_acceleration_terms():
    first=planar_reference_to_com_review((.03,-.005,.035),(.02,.003,.02),(.09,-.05),yaw_radius_com_m=.533)
    reverse=planar_reference_to_com_review(first['com_body_velocity'],first['com_body_velocity_derivative'],
        (-.09,.05),yaw_radius_com_m=.533)
    assert reverse['com_body_velocity']==pytest.approx(first['reference_body_velocity'])
    assert reverse['com_body_velocity_derivative']==pytest.approx(first['reference_body_velocity_derivative'])
    assert reverse['com_specific_force_in_body_m_s2']==pytest.approx(first['reference_point_inertial_accel_in_body_m_s2'])
    assert first['omitted_reference_to_com_specific_force_m_s2']==pytest.approx([.00088975,.00186125])
    # Pure rotation at constant rate still needs centripetal CoM acceleration.
    pure=planar_reference_to_com_review((0.,0.,.1),(0.,0.,0.),(.1,.05),yaw_radius_com_m=.533)
    assert pure['com_specific_force_in_body_m_s2']==pytest.approx([-.001,-.0005])
    assert pure['specific_yaw_moment_about_reference_m2_s2']==pytest.approx(0.,abs=1e-18)


@pytest.mark.parametrize('fault',['short','nan','infinite','bad_offset','zero_radius','boolean_radius'])
def test_point_frame_review_rejects_invalid_or_implicit_inputs(fault):
    velocity=(.03,0.,.03);derivative=(.02,0.,.02);offset=(.09,.05);radius=.533
    if fault=='short':velocity=(.03,0.)
    elif fault=='nan':derivative=(.02,float('nan'),.02)
    elif fault=='infinite':velocity=(.03,0.,float('inf'))
    elif fault=='bad_offset':offset=(.09,)
    elif fault=='zero_radius':radius=0.
    else:radius=True
    with pytest.raises(ValueError):planar_reference_to_com_review(velocity,derivative,offset,yaw_radius_com_m=radius)


@pytest.mark.parametrize('boundary',['position','heading','speed'])
def test_original_quantized_key_can_merge_a_valid_MODEL_goal_with_non_goal_but_split_key_cannot(boundary):
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    goal=(10.12581432198926,-4.694855006220115,0.)
    inside=lattice.State(goal[0],goal[1],0.,'coordinated_counterphase',0.,0.,True,1,
        longitudinal_velocity=.003)
    if boundary=='position':
        inside=replace(inside,x=goal[0]-.0249);outside=replace(inside,x=goal[0]-.0251)
    elif boundary=='heading':
        inside=replace(inside,yaw=math.radians(1.99));outside=replace(inside,yaw=math.radians(2.01))
    else:
        inside=replace(inside,longitudinal_velocity=.0049,lateral_velocity=0.)
        outside=replace(inside,lateral_velocity=.0011)
    assert lattice._state_key(inside,True)==lattice._state_key(outside,True)
    key=goal_predicate_separating_state_key(lattice._state_key,goal,terminal_speed_m_s=.005)
    assert key(inside,True)[:-1]==lattice._state_key(inside,True)
    assert key(inside,True)[-1] is True and key(outside,True)[-1] is False
    assert key(inside,True)!=key(outside,True)
    # Concrete original-dictionary dominance counterexample, not proof that
    # this happened during a particular recorded hospital search.
    old={lattice._state_key(outside,True):1.}
    assert 2.>=old[lattice._state_key(inside,True)]
    new={key(outside,True):1.};assert key(inside,True) not in new
    assert key.parameters['original_edge_and_goal_constraints_unchanged']


def test_goal_key_split_keeps_reverse_history_at_the_END_for_original_resource_Pareto_logic():
    from patient_initial_stability import steering_lattice as lattice
    State,propagate,make_key=resource_propagation_binding(lattice.State,lattice.propagate,MotionResourceLimits())
    physical=goal_predicate_separating_state_key(lattice._state_key,(.3,0.,0.),terminal_speed_m_s=.005)
    key=make_key(physical);state=State(.3,0.,0.,'coordinated_counterphase',0.,0.,True,1,
        mission_reverse_episode_bins=0,mission_reverse_total_bins=7,mission_direction_changes=2)
    assert key(state,True)[-4:] == (True,0,7,2)
    assert physical.counters['exact_accepted_goal_key_calls']==1


@pytest.mark.parametrize('step_m',[.025,.050])
def test_independent_terminal_segment_length_tag_replay_keeps_original_MODEL_goal_and_resources(step_m):
    from copy import deepcopy
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    _,State,propagate,model,_=_saved_complete_NE_prefix_fixture()
    initial=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0);state=initial
    actions=[];states=[]
    for i in range(round(.10/step_m)):
        action=('coordinated_counterphase',0.,1,.003)
        state,_,_=propagate(state,*action[:3],step_m=step_m,speed_m_s=.05,
            dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=action[3])
        actions.append(action);states.append(state)
    before=deepcopy(asdict(initial))
    report=audit_complete_NE_terminal_segment(initial,actions,states,propagate,model,lambda p:False,(.10,0.,0.),step_m=step_m)
    assert report['verdict']=='PASS' and all(report['checks'].values())
    assert asdict(initial)==before and report['primitive_step_m']==step_m
    assert report['original50mm_whole_mission_schema_NOT_used'] and not report['actual_motion_authorized']
    failed=audit_complete_NE_terminal_segment(initial,actions,states,propagate,model,lambda p:False,(.10,.050,0.),step_m=step_m)
    assert failed['verdict']=='FAIL' and not failed['checks']['original_goal_position_under25mm']
    with pytest.raises(ValueError):audit_complete_NE_terminal_segment(initial,actions,states,propagate,model,lambda p:True,(.10,0.,0.),step_m=step_m)


def _saved_complete_NE_prefix_fixture():
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import RigidBodyParameterBounds,MassNormalizedRigidBodyEdgeModel
    from hospital_h3_turn_planning import model_bounds
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,
        lattice.propagate,lattice._state_key,MotionResourceLimits())
    State=base.__globals__['State'];propagate=base.__globals__['propagate']
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    records=[dict(node_id=0,action=None,cost_s=0.,state=asdict(state))]
    for i in range(3):
        action=['coordinated_counterphase',0.,1,.03]
        state,_,elapsed=propagate(state,*action[:3],speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,
            rigid_body_model=model,target_speed_m_s=action[3])
        records.append(dict(node_id=i+1,action=action,cost_s=records[-1]['cost_s']+elapsed,state=asdict(state)))
    return records,State,propagate,model,state


def test_NE_terminal_time_guidance_prices_measured_speed_and_only_complete_goal_gets_zero():
    from dataclasses import replace,asdict
    from patient_initial_stability.steering_lattice import State
    calls=[];goal=(.1,0.,0.)
    def metric(pose,goals,radius):
        calls.append(('metric',pose,goals,radius));return [math.hypot(pose[0]-.1,pose[1])]
    def time(distance,speed):
        calls.append(('time',distance,speed));return distance/.05+max(0.,speed-.005)/.02
    def accepted(s,rigid):
        return rigid and math.hypot(s.x-.1,s.y)<=.025 and abs(s.yaw)<=math.radians(2.) and math.hypot(s.longitudinal_velocity,s.lateral_velocity)<=.005
    guide=NETerminalTimeGuidance(goal,connection_metric=metric,arc_time_guidance=time,
        accepted_goal=accepted,maximum_speed_m_s=.05,maximum_curvature_1_m=.7)
    fast=State(0.,0.,0.,'front_only',.2,-.1,True,1,.05,.001,.01)
    original=asdict(fast);slow=replace(fast,longitudinal_velocity=.003)
    assert guide(fast,goal)>guide(slow,goal)>0.
    assert ('time',.1,math.hypot(.05,.001)) in calls and asdict(fast)==original
    moving_goal=replace(fast,x=.1)
    assert guide(moving_goal,goal)>0.
    n=len(calls);assert guide(replace(slow,x=.1),goal)==0. and len(calls)==n
    assert guide.statistics['stop_time_priority_raised']==2  # Only the two moving-fast queries need braking.
    assert guide.statistics['complete_goal_queries']==1
    assert guide.parameters['states_edges_costs_keys_resources_and_goal_constraints_unchanged']


@pytest.mark.parametrize('invalid',[[],[-.1],[float('nan')],[float('inf')],[.1,.2]])
def test_NE_terminal_time_guidance_rejects_invalid_metric_without_silent_fallback(invalid):
    from patient_initial_stability.steering_lattice import State
    guide=NETerminalTimeGuidance((1.,0.,0.),connection_metric=lambda *a:invalid,
        arc_time_guidance=lambda *a:pytest.fail('No time query after invalid metric'),
        accepted_goal=lambda *a:False,maximum_speed_m_s=.05,maximum_curvature_1_m=.7)
    with pytest.raises(ValueError):guide(State(0.,0.,0.,'front_only',0.,0.,True),(1.,0.,0.))


@pytest.mark.parametrize('invalid',[-.1,float('nan'),float('inf')])
def test_NE_terminal_time_guidance_rejects_invalid_stop_time(invalid):
    from patient_initial_stability.steering_lattice import State
    guide=NETerminalTimeGuidance((1.,0.,0.),connection_metric=lambda *a:[1.],
        arc_time_guidance=lambda *a:invalid,accepted_goal=lambda *a:False,
        maximum_speed_m_s=.05,maximum_curvature_1_m=.7)
    with pytest.raises(ValueError):guide(State(0.,0.,0.,'front_only',0.,0.,True),(1.,0.,0.))


def test_NE_terminal_time_guidance_keeps_goal_and_body_velocity_inputs_explicit():
    from dataclasses import replace
    from patient_initial_stability.steering_lattice import State
    guide=NETerminalTimeGuidance((1.,0.,0.),connection_metric=lambda *a:[1.],
        arc_time_guidance=lambda *a:20.,accepted_goal=lambda *a:False,
        maximum_speed_m_s=.05,maximum_curvature_1_m=.7)
    s=State(0.,0.,0.,'front_only',0.,0.,True)
    with pytest.raises(ValueError):guide(s,(1.,.1,0.))
    with pytest.raises(ValueError):guide(replace(s,lateral_velocity=float('nan')),(1.,0.,0.))
    with pytest.raises(ValueError):NETerminalTimeGuidance((1.,0.,0.),connection_metric=None,
        arc_time_guidance=lambda *a:0.,accepted_goal=lambda *a:False,
        maximum_speed_m_s=.05,maximum_curvature_1_m=.7)


@pytest.mark.parametrize('guard',['rigid_body','bounds','resource','stopping_reachability','collision'])
def test_terminal_beam_cannot_bypass_any_original_edge_guard(guard):
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    initial=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True);calls=[]
    class Samples(list):resource_feasible=guard!='resource'
    class Checker(FreeCheckerFixture):
        def __call__(self,poses):return guard=='collision' and isinstance(poses,Samples)
    def propagate(previous,mode,curvature,direction,**kw):
        calls.append((mode,curvature,direction,kw['target_speed_m_s']))
        state=replace(previous,x=999. if guard=='bounds' else .05*direction,
            rigid_body_feasible=guard!='rigid_body',longitudinal_velocity=1. if guard=='stopping_reachability' else .003)
        return state,Samples([(state.x,state.y,state.yaw)]),1.
    diagnostics={};bounds=dict(minimum_x=-1.,maximum_x=1.,minimum_y=-1.,maximum_y=1.)
    result=bounded_NE_terminal_beam(initial,propagate,lattice._state_key,bounds,(.1,0.,0.),Checker(),object(),
        lambda state,rigid:state is not initial,beam_width=4,maximum_depth=1,node_limit=5,diagnostics=diagnostics)
    assert result is None and len(calls)==168 and diagnostics[guard]==168
    assert set(calls)=={(m,k,d,s) for m in lattice.ALL_MODES for k in lattice.CURVATURES
        for d in (1,-1) for s in (.003,.01,.03,.05)}


def test_terminal_beam_preserves_actual_deadline_exception_and_does_not_return_partial_success():
    from patient_initial_stability import steering_lattice as lattice
    error=TimeoutError('ORIGINAL_CALLER_DEADLINE');calls=[]
    class Checker(FreeCheckerFixture):
        def check_deadline(self):raise error
    with pytest.raises(TimeoutError) as caught:
        bounded_NE_terminal_beam(lattice.State(0.,0.,0.,'front_only',0.,0.,True),
            lambda *a,**kw:calls.append(a),lattice._state_key,
            dict(minimum_x=-1.,maximum_x=1.,minimum_y=-1.,maximum_y=1.),(.1,0.,0.),
            Checker(),object(),lambda *a:False,beam_width=4,maximum_depth=2,node_limit=10,diagnostics={})
    assert caught.value is error and not calls


def test_actual_original_NE_terminal_beam_keeps_incoming_history_and_independent_goal_replay(compiled_NE_edge):
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    State,propagate,_=resource_propagation_binding(lattice.State,compiled_NE_edge,MotionResourceLimits())
    initial=State(.02,.03,0.,'coordinated_counterphase',0.,0.,True,1,
        longitudinal_velocity=.001,mission_reverse_total_bins=7,mission_direction_changes=2)
    before=asdict(initial);model,_=_NE_cache_models();goal=(.12,.03,0.)
    predicate=goal_predicate_separating_state_key(lattice._state_key,goal,terminal_speed_m_s=.005)
    diagnostics={};bounds=dict(minimum_x=-1.,maximum_x=1.,minimum_y=-1.,maximum_y=1.)
    answer=bounded_NE_terminal_beam(initial,propagate,predicate,bounds,goal,FreeCheckerFixture(),model,
        predicate.accepted_goal,beam_width=4,maximum_depth=4,node_limit=16,diagnostics=diagnostics)
    assert answer is not None and answer.actions and asdict(initial)==before
    proof=audit_complete_NE_terminal_segment(initial,answer.actions,answer.states,propagate,model,
        lambda poses:False,goal,step_m=.050)
    assert proof['verdict']=='PASS' and all(proof['checks'].values())
    assert diagnostics['termination']=='complete_original_goal'
    assert diagnostics['expanded_nodes']<=16 and len(answer.actions)<=4
    assert answer.states[-1].mission_reverse_total_bins>=7
    assert answer.states[-1].mission_direction_changes>=2


def test_executable_stop_beam_adds_neutral_goal_without_weakening_original_NE_limits(compiled_NE_edge):
    from patient_initial_stability import steering_lattice as lattice
    State,propagate,_=resource_propagation_binding(lattice.State,compiled_NE_edge,MotionResourceLimits())
    initial=State(.02,.03,0.,'coordinated_counterphase',0.,0.,True,1,longitudinal_velocity=.001)
    model,_=_NE_cache_models();goal=(.12,.03,0.)
    predicate=goal_predicate_separating_state_key(lattice._state_key,goal,terminal_speed_m_s=.005)
    diagnostics={};bounds=dict(minimum_x=-1.,maximum_x=1.,minimum_y=-1.,maximum_y=1.)
    answer=bounded_NE_terminal_beam(initial,propagate,predicate,bounds,goal,FreeCheckerFixture(),model,
        predicate.accepted_goal,beam_width=4,maximum_depth=4,node_limit=16,diagnostics=diagnostics,
        executable_stop_connections=True)
    assert answer is not None and predicate.accepted_goal(answer.states[-1],True)
    assert max(abs(answer.states[-1].front),abs(answer.states[-1].rear))<=math.radians(2.5)
    assert answer.actions[-1][1]==0.
    assert diagnostics['stronger_connection_condition_NOT_relaxed_original_goal_or_new_plant']


def test_saved_NE_prefix_replay_retains_every_physical_and_resource_field_before_continuation():
    from dataclasses import asdict
    from copy import deepcopy
    records,State,propagate,model,expected=_saved_complete_NE_prefix_fixture();saved=deepcopy(records)
    state,proof=replay_saved_complete_NE_prefix(records,State,propagate,lambda p:False,model)
    assert asdict(state)==asdict(expected) and records==saved
    assert proof['verified_edges']==3 and proof['exact_all_physical_steering_velocity_resource_fields']
    assert state.longitudinal_velocity>0. and state.front!=0.
    assert proof['cached_prefix_is_NOT_free_primary_planning_input'] and not proof['actual_motion_authorized']


def test_terminal_entry_selects_first_not_nearest_and_replays_exact_original_full_state():
    from copy import deepcopy
    from dataclasses import asdict
    records,State,propagate,model,last=_saved_complete_NE_prefix_fixture();before=deepcopy(records)
    goal=(last.x,last.y,last.yaw)
    distances=[math.hypot(row['state']['x']-goal[0],row['state']['y']-goal[1]) for row in records]
    radius=(distances[0]+distances[1])/2.
    selected,review=first_terminal_neighborhood_prefix(records,goal,entry_radius_m=radius)
    assert selected==records[:2] and review['selected_prefix_edges']==1
    assert review['source_chain_edges']==3 and review['complete_incoming_state']==records[1]['state']
    state,proof=replay_saved_complete_NE_prefix(selected,State,propagate,lambda p:False,model)
    assert asdict(state)==records[1]['state'] and state.longitudinal_velocity>0.
    assert review['whole_fresh_generation_cost_must_be_charged']
    assert review['cached_input_is_NOT_primary_success'] and not review['actual_motion_authorized']
    selected[-1]['state']['mission_reverse_total_bins']=99
    review['complete_incoming_state']['front']=99.
    assert records==before and proof['verified_edges']==1


@pytest.mark.parametrize('radius',[0.,.025,-.1,1.64,True,float('nan'),float('inf')])
def test_terminal_entry_policy_requires_explicit_finite_bounded_radius(radius):
    records,_,_,_,last=_saved_complete_NE_prefix_fixture()
    with pytest.raises(ValueError):
        first_terminal_neighborhood_prefix(records,(last.x,last.y,last.yaw),entry_radius_m=radius)


@pytest.mark.parametrize('fault',['missing_root','root_inside','no_entry','unordered_node','decreasing_cost','nonfinite_state'])
def test_terminal_entry_selection_cannot_hide_incomplete_or_changed_parent_chain(fault):
    records,_,_,_,last=_saved_complete_NE_prefix_fixture();goal=(last.x,last.y,last.yaw);radius=.1
    if fault=='missing_root':records=records[1:]
    elif fault=='root_inside':radius=1.
    elif fault=='no_entry':goal=(5.,5.,0.)
    elif fault=='unordered_node':records[-1]['node_id']=records[-2]['node_id']
    elif fault=='decreasing_cost':records[-1]['cost_s']=0.
    else:records[-1]['state']['yaw_rate']=float('nan')
    with pytest.raises(ValueError):first_terminal_neighborhood_prefix(records,goal,entry_radius_m=radius)


@pytest.mark.parametrize('fault',['root_velocity','missing_resource','late_steering','late_velocity',
    'resource_history','parent_order','changed_action','collision'])
def test_saved_NE_prefix_replay_rejects_guessed_or_reset_states_and_blocked_sweeps(fault):
    records,State,propagate,model,_=_saved_complete_NE_prefix_fixture();checker=lambda p:False
    if fault=='root_velocity':records[0]['state']['longitudinal_velocity']=.001
    elif fault=='missing_resource':del records[0]['state']['mission_reverse_total_bins']
    elif fault=='late_steering':records[2]['state']['front']=0.
    elif fault=='late_velocity':records[2]['state']['longitudinal_velocity']=0.
    elif fault=='resource_history':records[2]['state']['mission_direction_changes']=1
    elif fault=='parent_order':records[2]['node_id']=records[1]['node_id']
    elif fault=='changed_action':records[2]['action'][3]=.05
    else:checker=lambda p:True
    with pytest.raises(ValueError):replay_saved_complete_NE_prefix(records,State,propagate,checker,model)


def _fresh_terminal_pipeline_fixture(driver):
    """Real complete original-model states, FAKE bounded search/clock only."""
    from copy import deepcopy
    from dataclasses import asdict
    from types import FunctionType,SimpleNamespace
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    import hospital_mission_planning as binding
    f=driver['run_fresh_prefix_terminal_NE_pipeline'];now=[0.];calls=[];events=[]
    _,State,propagate,model,_=_saved_complete_NE_prefix_fixture()
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,
        lattice.State,lattice.propagate,lattice._state_key,MotionResourceLimits())
    # Bind the exact class/propagator to the original code object. This is not
    # a physical state restoration and no actual lattice search is executed.
    base=FunctionType(base.__code__,dict(base.__globals__,State=State,propagate=propagate),
        base.__name__,base.__defaults__);base.__kwdefaults__=dict(lattice.plan.__kwdefaults__ or {})
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    root=state;records=[dict(node_id=0,action=None,cost_s=0.,state=asdict(state))]
    actions=[];states=[];costs=[];cost=0.
    for i in range(2):
        a=('coordinated_counterphase',0.,1,.003);previous=state
        state,_,elapsed=propagate(state,*a[:3],speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,
            rigid_body_model=model,target_speed_m_s=a[3])
        target=lattice.steering_targets(a[0],a[1])
        increment=elapsed+lattice.STEERING_TRANSIENT_PENALTY_WEIGHT*max(
            abs(target[0]-previous.front),abs(target[1]-previous.rear))/lattice.STEERING_RATE_RAD_S+state.rigid_body_penalty_s
        cost+=increment;costs.append(increment);actions.append(a);states.append(state)
        records.append(dict(node_id=i+1,action=list(a),cost_s=cost,state=asdict(state)))
    stage=dict(method=f.__globals__['METHODS'][-1],plan_success=False,status='SOFTWARE_JOINT_STAGE_TIMEOUT',
        error='TimeoutError: ORIGINAL_FAKE_STAGE_CLOCK',planning_time_s=150.,
        selected_actions=[],selected_states=[],P_joint_NE_lattice=True,
        candidate_generation_diagnostics=dict(expanded_nodes=100,original_search_prefixes={
            'prefixes':{'goal_closest':{'records':deepcopy(records[:2])}}}))
    result=lattice.PlanResult((states[-1],),(actions[-1],),7,costs[-1],())
    search=SimpleNamespace(mission_unwrapped_resource_search=base,
        mission_complete_goal_predicate=goal_predicate_separating_state_key(lattice._state_key,(.10,0.,0.),
            terminal_speed_m_s=.005).accepted_goal)
    def joint(*a,**kw):
        calls.append(('joint',a[-1],kw));now[0]=150.;return deepcopy(stage)
    def observe(*args):
        events.append(args);now[0]+=1.
        # An observer must not get a mutable reference to the authenticated
        # chain. The durable JSON writer can serialize but cannot rewrite it.
        snapshot=args[2]['candidate_generation_diagnostics']['joint_stage']['original_search_prefixes']['prefixes']['goal_closest']['records']
        if len(snapshot)>1:snapshot[1]['state']['x']=999.
    def selected(*a,**kw):
        calls.append(('terminal',a,kw));now[0]+=5.
        kw['diagnostics'].update(expanded_nodes=7,generated=20);return result
    class Metric:
        parameters={'role':'FAKE_DISTANCE_NOT_PATH_OR_SUCCESS'}
        def __call__(self,*a):return [0.]
    class Checker(FreeCheckerFixture):
        def __init__(self,*a,deadline=None,**kw):super().__init__();self.deadline=deadline
        def check_deadline(self):
            if now[0]>=self.deadline:raise TimeoutError('SOFTWARE_ONLY_CHECKER_DEADLINE')
        def __call__(self,poses):self.check_deadline();return False
    fake_binding=SimpleNamespace(**{name:getattr(binding,name) for name in (
        'replay_saved_complete_NE_prefix','continue_original_lattice_from_state',
        'pareto_resource_lattice_function','audit_complete_NE_terminal_segment')},
        installed_reeds_shepp_metric=lambda:Metric(),plan_NE_mission_candidate=selected,
        bounded_NE_terminal_beam=selected)
    namespace=dict(f.__globals__,time=SimpleNamespace(perf_counter=lambda:now[0]),
        run_joint_NE_pipeline=joint,binding=fake_binding)
    function=FunctionType(f.__code__,namespace,f.__name__)
    function.__kwdefaults__=dict(f.__kwdefaults__ or {})
    args=(search,'BOUNDS','GUIDANCE',Checker,dict(start_xyyaw=[0.,0.,0.],local_goal_xyyaw=[.10,0.,0.]),
        None,None,None,model,180.)
    return SimpleNamespace(function=function,args=args,namespace=namespace,clock=now,calls=calls,events=events,
        stage=stage,observe=observe,binding=fake_binding,result=result,records=records,states=states,
        model=model,root=root,actions=actions,cost=cost)


def test_fresh_terminal_pipeline_uses_only_own_parent_chain_one_deadline_and_exact_all_state_replay(mission_offline_probe_driver):
    from dataclasses import asdict
    f=_fresh_terminal_pipeline_fixture(mission_offline_probe_driver)
    row=f.function(*f.args,node_limit=20000,observe=f.observe)
    assert row['plan_success'] and row['status']=='OFFLINE_FRESH_NE_PREFIX_AND_TERMINAL_PLAN_FOUND_EXECUTION_PENDING'
    assert row['action_count']==2 and row['search_cost_s']==f.cost and 'error' not in row
    assert row['candidate_generation_diagnostics']['expanded_nodes']==107
    accounting=row['P_fresh_terminal_accounting']
    assert accounting['joint_stage']['error']=='TimeoutError: ORIGINAL_FAKE_STAGE_CLOCK'
    assert accounting['terminal_search_calls']==1 and accounting['terminal_stage']['node_cap']==10000
    assert accounting['terminal_stage']['complete_incoming_state']==asdict(f.states[0])
    assert accounting['whole_fresh_chain_replay']['verdict']=='PASS'
    assert accounting['end_to_end_budget_respected'] and accounting['old_cached_diagnostic_chain_is_NOT_an_input']
    assert f.calls[0]==('joint',150.,dict(node_limit=20000))
    assert f.calls[1][2]['node_limit']==10000 and f.calls[1][2]['directions']==(1,-1)
    assert len(f.events)==1 and f.events[0][0]=='NE_prefix'
    assert row['candidate_generation_diagnostics']['joint_stage']['original_search_prefixes']['prefixes']['goal_closest']['records'][1]['state']['x']!=999.


def test_fresh_terminal_beam_charges_own_prefix_and_one_connection_without_borrowed_metric_or_Astar(mission_offline_probe_driver):
    from dataclasses import asdict
    f=_fresh_terminal_pipeline_fixture(mission_offline_probe_driver)
    f.namespace['P_TERMINAL_SEARCH_REVISION']='bounded_terminal_beam_v1'
    f.binding.installed_reeds_shepp_metric=lambda:pytest.fail('Unused geometric metric must not be constructed')
    f.binding.plan_NE_mission_candidate=lambda *a,**kw:pytest.fail('No second Astar or alternative retry')
    row=f.function(*f.args,node_limit=20000,observe=f.observe)
    assert row['plan_success'] and row['search_cost_s']==f.cost and row['action_count']==2
    assert row['candidate_generation_diagnostics']['expanded_nodes']==107
    accounting=row['P_fresh_terminal_accounting']
    assert accounting['terminal_search_revision']=='bounded_terminal_beam_v1'
    assert accounting['terminal_search_calls']==1 and accounting['end_to_end_budget_respected']
    assert accounting['terminal_stage']['complete_incoming_state']==asdict(f.states[0])
    assert accounting['terminal_stage']['metric_parameters'] is None
    assert f.calls[1][2]['beam_width']==32 and f.calls[1][2]['maximum_depth']==20
    assert f.calls[1][2]['node_limit']==10000 and len(f.events)==1
    assert accounting['whole_fresh_chain_replay']['verdict']=='PASS'


@pytest.mark.parametrize('terminal_strategy',['original_Astar_v1','bounded_terminal_beam_v1'])
@pytest.mark.parametrize('fault',['missing_prefix','nodes_exhausted','root_velocity','prefix_steering',
    'prefix_cost','suffix_velocity','suffix_resource','suffix_collision','full_replay_goal',
    'late_observer','late_terminal','late_serialization'])
def test_fresh_terminal_pipeline_never_promotes_incomplete_corrupt_or_late_chain(mission_offline_probe_driver,fault,terminal_strategy):
    from dataclasses import replace
    from types import SimpleNamespace
    f=_fresh_terminal_pipeline_fixture(mission_offline_probe_driver)
    f.namespace['P_TERMINAL_SEARCH_REVISION']=terminal_strategy
    selected_name='bounded_NE_terminal_beam' if terminal_strategy=='bounded_terminal_beam_v1' else 'plan_NE_mission_candidate'
    prefix=f.stage['candidate_generation_diagnostics']['original_search_prefixes']['prefixes']['goal_closest']['records']
    expected_calls=1
    if fault=='missing_prefix':prefix.clear();expected_calls=0
    elif fault=='nodes_exhausted':f.stage['candidate_generation_diagnostics']['expanded_nodes']=20000;expected_calls=0
    elif fault=='root_velocity':prefix[0]['state']['longitudinal_velocity']=.001;expected_calls=0
    elif fault=='prefix_steering':prefix[1]['state']['front']=0.;expected_calls=0
    elif fault=='prefix_cost':prefix[1]['cost_s']+=1.
    elif fault in ('suffix_velocity','suffix_resource'):
        changed=replace(f.result.states[0],**({'longitudinal_velocity':0.} if fault=='suffix_velocity' else {'mission_direction_changes':1}))
        bad=replace(f.result,states=(changed,));old=getattr(f.binding,selected_name)
        def selected(*a,**kw):old(*a,**kw);return bad
        setattr(f.binding,selected_name,selected)
    elif fault=='suffix_collision':
        class Collision(f.args[3]):
            def __call__(self,poses):
                return self.deadline==180. or super().__call__(poses)
        f.args=(*f.args[:3],Collision,*f.args[4:])
    elif fault=='full_replay_goal':f.args[4]['local_goal_xyyaw']=[.20,0.,0.]
    elif fault=='late_observer':
        old=f.observe
        def observe(*args):old(*args);f.clock[0]=179.
        f.observe=observe;expected_calls=0
    elif fault=='late_terminal':
        old=getattr(f.binding,selected_name)
        def selected(*a,**kw):result=old(*a,**kw);f.clock[0]=179.;return result
        setattr(f.binding,selected_name,selected)
    else:
        old=f.namespace['original'].state_record
        def serialize(*a,**kw):f.clock[0]=181.;return old(*a,**kw)
        f.namespace['original']=SimpleNamespace(**{k:getattr(f.namespace['original'],k) for k in
            ('SPEED_LEVELS','SPEED','TERMINAL_SPEED','DECELERATION')},state_record=serialize)
    row=f.function(*f.args,node_limit=20000,observe=f.observe)
    assert not row['plan_success'] and not row['selected_actions'] and not row['selected_states']
    assert row['P_fresh_terminal_accounting']['terminal_search_calls']==expected_calls
    assert row['candidate_generation_diagnostics']['expanded_nodes']<=20000
    assert 'search_cost_s' not in row and 'predicted_motion_time_s' not in row


def test_fresh_terminal_pipeline_complete_joint_stage_never_starts_a_second_search(mission_offline_probe_driver):
    f=_fresh_terminal_pipeline_fixture(mission_offline_probe_driver)
    f.stage.update(plan_success=True,status='SOFTWARE_ORIGINAL_COMPLETE_PLAN')
    row=f.function(*f.args,node_limit=20000,observe=f.observe)
    assert row['plan_success'] and row['status']=='SOFTWARE_ORIGINAL_COMPLETE_PLAN'
    assert row['P_fresh_terminal_accounting']['terminal_search_calls']==0 and not f.events
    assert len(f.calls)==1


@pytest.mark.parametrize('fault',['valid','old_policy','P_only','wrong_compute','wrong_guidance','600s'])
def test_fresh_terminal_reserve_requires_new_all_four180s_explicit_contract(mission_offline_probe_driver,fault):
    from types import FunctionType
    f=mission_offline_probe_driver['validate_probe_selection'];old=f.__globals__
    values=dict(old,P_PIPELINE_REVISION='owned_reference_terminal_reserve_NE_v1',
        GUIDANCE_REVISION='local_heading_terminal_blend_v1',SEARCH_REVISION='resource_pareto_first_stored_goal_v2',
        COMPUTE_REVISION='compiled_original_NE_and_exact_sparse_cell_SAT_v1',RUN_METHODS=old['METHODS'],
        OUTPUT=old['GATE']/'hospital_h6_planning/software_fresh_terminal_fixture_NOT_run')
    if fault=='old_policy':values['SEARCH_REVISION']='resource_pareto_best_constructor_only_v1'
    elif fault=='P_only':values['RUN_METHODS']=(old['METHODS'][-1],)
    elif fault=='wrong_compute':values['COMPUTE_REVISION']='original_exact_computation_v1'
    elif fault=='wrong_guidance':values['GUIDANCE_REVISION']='terminal_pose_connection_v1'
    elif fault=='600s':values['TOTAL_WALL_S']=600.
    function=FunctionType(f.__code__,values,f.__name__)
    if fault=='valid':function()
    else:
        with pytest.raises(ValueError):function()


@pytest.mark.parametrize('position,heading',[(0.,0.),(.010,.5),(.020,2.),(.080,10.)])
def test_conditional_tracking_reserve_contains_rotated_translated_ORIGINAL_body_outline(position,heading):
    import numpy as np
    from hospital_geometry_tools import footprint_polygon
    definition=conditional_tracking_reserve_definition(position_error_m=position,heading_error_deg=heading)
    extra=definition['planning_additional_execution_reserve_m']
    assert extra>=definition['exact_uniform_additional_reserve_m']
    assert extra-definition['exact_uniform_additional_reserve_m']<.005+1e-12
    for angle in np.linspace(-math.radians(heading),math.radians(heading),51):
        for direction in np.linspace(-math.pi,math.pi,25):
            pose=[position*math.cos(direction),position*math.sin(direction),angle]
            corners=footprint_polygon(pose,clearance=.010)
            assert (np.abs(corners[:,0])<=1.630/2+.010+extra+1e-12).all()
            assert (np.abs(corners[:,1])<=.965/2+.010+extra+1e-12).all()
    assert definition['original_abort_guards']==dict(path_error_m=.080,path_heading_error_deg=10.)
    assert definition['conditional_geometric_containment_NOT_proven_controller_tracking']
    assert not definition['actual_motion_authorized'] and not definition['use_for_final_statistics']


def test_20mm_2degree_conditional_reserve_is_50mm_not_a_10mm_safety_waiver():
    d=conditional_tracking_reserve_definition(position_error_m=.020,heading_error_deg=2.)
    assert d['planning_additional_execution_reserve_m']==.050
    assert d['exact_uniform_additional_reserve_m']==pytest.approx(.048492079)
    assert d['planning_total_clearance_m']==pytest.approx(.060)
    assert d['unchanged_required_safety_clearance_m']==.010


@pytest.mark.parametrize('position,heading',[(True,2.),(.081,2.),(-.001,2.),(.02,True),(.02,11.),(.02,float('nan'))])
def test_conditional_reserve_rejects_invalid_or_beyond_original_guards(position,heading):
    with pytest.raises(ValueError):conditional_tracking_reserve_definition(position_error_m=position,heading_error_deg=heading)


@pytest.mark.parametrize('kind',['python','native'])
def test_additional_planning_reserve_uses_original_map_SAT_and_shared_extra_not_body_or_physics_edit(kind):
    import numpy as np
    from hospital_geometry_tools import pose_collision
    from hospital_planning_tools import HospitalEnvelopeChecker
    from hospital_native_envelope import NativeHospitalEnvelopeChecker
    original=HospitalEnvelopeChecker if kind=='python' else NativeHospitalEnvelopeChecker
    scoped=additional_planning_reserve_checker(original,reserve_m=.050)
    grid=np.zeros((400,400),bool);grid[:,200:]=True;before=grid.copy()
    checker=scoped(grid,[-2.,-2.],.010,[0.,0.,0.]);old=original(grid,[-2.,-2.],.010,[0.,0.,0.])
    assert checker.clearance==pytest.approx(.060) and old.clearance==.010
    assert scoped.world_collision is original.world_collision
    assert checker.planning_execution_reserve_m==.050 and checker.original_physical_safety_clearance_m==.010
    for x in np.linspace(-1.15,-.80,20):
        for yaw in np.linspace(-.2,.2,5):
            assert checker.world_collision([x,0.,yaw])==pose_collision(grid,[x,0.,yaw],[-2.,-2.],.010,.060)
    assert np.array_equal(before,grid)
    assert not old.world_collision([-.85,0.,0.]) and checker.world_collision([-.85,0.,0.])
    with pytest.raises(ValueError):scoped(grid,[-2.,-2.],.010,[0.,0.,0.],clearance=.060)


def test_independent_extra_reserve_audit_reuses_original_code_and_globals_privately():
    import numpy as np
    import audit_hospital_h1_plan as original
    before=dict(original.audit.__globals__)
    run=additional_planning_reserve_audit(original.audit,reserve_m=.050)
    assert run.__code__ is original.audit.__code__ and original.audit.__globals__==before
    grid=np.zeros((400,400),bool);grid[:,200:]=True
    args=(grid,[-.85,0.,0.],[-2.,-2.],.010,.010)
    assert not original.pose_collision(*args)
    assert run.__globals__['pose_collision'](*args)
    with pytest.raises(ValueError):run.__globals__['pose_collision'](*args[:-1],.060)


def test_matched_driver_extra_reserve_is_explicit_common_50mm_and_never_old_folder_or_P_only():
    import runpy
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    driver=root/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_planning/review_matched_probe_driver.txt'
    target=driver.parent/'heading0_execution_reserve_matched_dev_r14'
    scope=runpy.run_path(str(driver),run_name='offline_DRIVER_DEFINITION_ONLY',init_globals=dict(
        h6_offline_additional_execution_reserve_m=.050,h6_offline_output=target))
    scope['validate_probe_selection']()
    assert scope['TRACKING_RESERVE_DEFINITION']['planning_additional_execution_reserve_m']==.050
    scope['validate_probe_selection'].__globals__['RUN_METHODS']=(scope['METHODS'][-1],)
    with pytest.raises(ValueError):scope['validate_probe_selection']()


@pytest.mark.parametrize('fault',[None,'missing_assumption','inconsistent_margin','P_only','old_folder'])
def test_new_common20mm_reserve_requires_explicit10mm_halfdegree_assumption_and_all_methods(fault):
    import runpy
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    driver=root/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_planning/review_matched_probe_driver.txt'
    values=dict(h6_offline_additional_execution_reserve_m=.020,
        h6_offline_assumed_tracking_position_m=.010,h6_offline_assumed_tracking_heading_deg=.5,
        h6_offline_output=driver.parent/'SOFTWARE_DEFINITION_ONLY_NOT_EXECUTED')
    if fault=='missing_assumption':values.pop('h6_offline_assumed_tracking_heading_deg')
    elif fault=='inconsistent_margin':values['h6_offline_additional_execution_reserve_m']=.015
    elif fault=='P_only':values['h6_offline_methods']=('P_dynamics_full_envelope',)
    elif fault=='old_folder':values['h6_offline_output']=driver.parent/'heading0_first_matched_dev_r01'
    scope=runpy.run_path(str(driver),run_name='SOFTWARE_DRIVER_DEFINITION_ONLY',init_globals=values)
    if fault:
        with pytest.raises(ValueError):scope['validate_probe_selection']()
    else:
        scope['validate_probe_selection']();d=scope['TRACKING_RESERVE_DEFINITION']
        assert d['planning_additional_execution_reserve_m']==.020
        assert d['exact_uniform_additional_reserve_m']==pytest.approx(.017180638895262827)
        assert d['unchanged_required_safety_clearance_m']==.010
        assert d['planning_total_clearance_m']==.030
        assert d['conditional_geometric_containment_NOT_proven_controller_tracking']
        assert not d['actual_motion_authorized'] and not d['use_for_final_statistics']


def _terminal_metric_guide(fallback=lambda s,g:1.,metric=None,**settings):
    metric=(lambda pose,goals,radius:[.2]) if metric is None else metric
    values=dict(connection_metric=metric,speed_m_s=.05,maximum_curvature_1_m=.7,terminal_zone_m=1.63)
    values.update(settings)
    return TerminalPoseConnectionGuidance(fallback,(1.,0.,0.),**values)


def test_terminal_connection_priority_sees_wrong_BODY_yaw_at_correct_position_without_accepting_or_mutating_state():
    from types import SimpleNamespace
    from copy import deepcopy
    calls=[]
    def metric(pose,goals,radius):calls.append((pose,goals,radius));return [.2]
    guide=_terminal_metric_guide(metric=metric)
    state=SimpleNamespace(x=1.,y=0.,yaw=math.radians(6.5),longitudinal_velocity=.003,
        lateral_velocity=.001,front=.1,rear=-.1,direction=-1)
    before=deepcopy(vars(state));assert guide(state,(1.,0.,0.))==4.
    assert vars(state)==before and calls==[((state.x,state.y,state.yaw),[(1.,0.,0.)],1./.7)]
    assert guide.statistics==dict(queries=1,metric_queries=1,priority_raised=1)
    d=guide.parameters['terminal_pose_connection']
    assert d['original_states_edges_costs_bins_goal_constraints_and_resources_NOT_changed']
    assert d['metric_is_NOT_a_feasible_path_or_NE_lower_bound_or_optimality_proof']


@pytest.mark.parametrize('pose',[(3.,0.,0.),(1.,0.,0.),(1.,0.,math.radians(1.99))])
def test_far_or_accepted_pose_preserves_original_fallback_including_NE_braking_priority(pose):
    from types import SimpleNamespace
    def forbidden(*args):pytest.fail('No needless metric or replacement of the original accepted-pose braking priority')
    guide=_terminal_metric_guide(fallback=lambda s,g:7.5,metric=forbidden)
    state=SimpleNamespace(x=pose[0],y=pose[1],yaw=pose[2],longitudinal_velocity=.05)
    assert guide(state,(1.,0.,0.))==7.5 and guide.statistics['metric_queries']==0


def test_terminal_ranking_max_never_reduces_existing_priority_and_identical_rule_wraps_own_P_reference():
    from types import SimpleNamespace
    def fallback(s,g):return 8.
    fallback.parameters=dict(reference_changes_priority_only=True,reference_states_never_inserted=True)
    common=_terminal_metric_guide(fallback=fallback);owned=common.wrap_reference(fallback)
    state=SimpleNamespace(x=.8,y=0.,yaw=.1)
    assert owned is not common and owned.metric is common.metric and owned.fallback is fallback
    assert owned(state,(1.,0.,0.))==common(state,(1.,0.,0.))==8.
    assert owned.parameters['terminal_pose_connection']==common.parameters['terminal_pose_connection']
    assert owned.parameters['reference_changes_priority_only'] and owned.parameters['reference_states_never_inserted']


@pytest.mark.parametrize('settings',[dict(speed_m_s=True),dict(speed_m_s=0.),
    dict(maximum_curvature_1_m=math.nan),dict(terminal_zone_m=.025),dict(connection_metric=None)])
def test_terminal_ranking_rejects_invalid_definition(settings):
    with pytest.raises(ValueError):_terminal_metric_guide(**settings)


@pytest.mark.parametrize('answer',[[],[.2,.3],[-.1],[math.nan],[math.inf]])
def test_terminal_ranking_metric_corruption_does_not_disguise_itself_as_existing_priority(answer):
    from types import SimpleNamespace
    guide=_terminal_metric_guide(metric=lambda *args:answer)
    with pytest.raises(ValueError):guide(SimpleNamespace(x=1.,y=0.,yaw=.1),(1.,0.,0.))
    with pytest.raises(ValueError,match='fixed goal'):guide(SimpleNamespace(x=1.,y=0.,yaw=.1),(1.,.1,0.))


def test_installed_terminal_metric_ranks_saved_r15_wrong_heading_as_geometric_connection_NOT_a_real_plan():
    import json
    from pathlib import Path
    from types import SimpleNamespace
    metric=installed_reeds_shepp_metric()
    path=Path(__file__).resolve().parents[3]/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_planning/heading0_tight_tracking_reserve_matched_dev_r15/P_dynamics_full_envelope.json'
    saved=json.loads(path.read_bytes());goal=saved['goal_local_xyyaw']
    row=saved['candidate_generation_diagnostics']['original_search_prefixes']['prefixes']['goal_closest']['records'][-1]['state']
    state=SimpleNamespace(**row)
    guide=TerminalPoseConnectionGuidance(lambda s,g:1.,goal,connection_metric=metric,
        speed_m_s=.05,maximum_curvature_1_m=.7,terminal_zone_m=1.63)
    assert guide(state,goal)>1.
    assert guide.statistics['priority_raised']==1
    state.yaw=goal[2]
    assert guide(state,goal)==1.  # SOFTWARE counterfactual; never inserted into a planner or saved trial.
    assert metric.parameters['original_NE_and_resource_guard_NOT_replaced']


@pytest.mark.parametrize('scope',['all_four','P_only','extended_budget'])
def test_new_terminal_guidance_requires_one_fresh_all_four180s_scope(mission_offline_probe_driver,scope):
    from types import FunctionType
    f=mission_offline_probe_driver['validate_probe_selection'];old=f.__globals__
    namespace=dict(old,GUIDANCE_REVISION='terminal_pose_connection_v1',
        SEARCH_REVISION='resource_pareto_best_constructor_only_v1',P_PIPELINE_REVISION='owned_reference_joint_NE_v1',
        RUN_METHODS=old['METHODS'],TOTAL_WALL_S=180.,OUTPUT=old['GATE']/'hospital_h6_planning/SOFTWARE_TERMINAL_DEFINITION_ONLY')
    if scope=='P_only':namespace['RUN_METHODS']=(old['METHODS'][-1],)
    elif scope=='extended_budget':namespace.update(RUN_METHODS=(old['METHODS'][-1],),TOTAL_WALL_S=600.)
    run=FunctionType(f.__code__,namespace,f.__name__)
    if scope=='all_four':run()
    else:
        with pytest.raises(ValueError):run()


def _forward_v3_geometry_and_plan(method='P_dynamics_full_envelope'):
    from hospital_h4_path_execution import method_settings,MODES,METHODS
    geometry=dict(scenario='H6_integrated_transfer_mission',full_body_rectangle_m=[1.630,.965],
        full_body_clearance_m=.010,start_xyyaw=[0.,0.,0.],goal_xyyaw=[.30,0.,0.],
        planning_occupancy_digest='fixture-raster-NOT-a-live-map',actual_motion_authorized=False,
        use_for_final_statistics=False,same_three_authored_support_rectangles=[
            [[-2.,-2.],[2.,-2.],[2.,2.],[-2.,2.]],
            [[2.,-2.],[4.,-2.],[4.,2.],[2.,2.]],
            [[4.,-2.],[6.,-2.],[6.,2.],[4.,2.]]])
    plan=dict(scenario=geometry['scenario'],method=method,plan_success=True,
        planner_dynamics=method_settings(method)['planner_dynamics'],
        allowed_modes=['coordinated_counterphase'] if method==METHODS[0] else list(MODES),
        start_world_xyyaw=list(geometry['start_xyyaw']),goal_world_xyyaw=list(geometry['goal_xyyaw']),
        goal_local_xyyaw=list(geometry['goal_xyyaw']),planning_occupancy_digest=geometry['planning_occupancy_digest'],
        selected_actions=[dict(mode='coordinated_counterphase',direction=1,curvature_1_m=0.,
            target_speed_m_s=.05) for _ in range(6)],
        selected_states=[dict(mode='coordinated_counterphase',world_xyyaw=[.05*i,0.,0.],
            x_m=.05*i,y_m=0.,yaw_deg=0.) for i in range(1,7)],
        common_max_command_speed_m_s=.05,full_envelope_clearance_m=.010,
        rigid_body_parameter_bounds=dict(max_longitudinal_accel_m_s2=.020),
        physical_execution_verified=False,Nav2_execution_verified=False,
        integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True,
        actual_motion_authorized=False,use_for_final_statistics=False)
    return geometry,plan


@pytest.mark.parametrize('method',['B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope'])
def test_H6_forward_adapter_keeps_native_plan_and_exact_original_code_globals(method):
    import copy
    import hospital_h4_path_execution as original
    geometry,plan=_forward_v3_geometry_and_plan(method);saved=copy.deepcopy((geometry,plan))
    before=dict(original.PathReference.__init__.__globals__)
    binding=forward_common_v3_execution_binding(geometry,simulation_budget_s=300.)
    controller=binding.TrackingController(plan)
    assert (geometry,plan)==saved and controller.reference.actions is plan['selected_actions']
    assert binding.PathReference.__init__.__code__ is original.PathReference.__init__.__code__
    assert binding.TrackingController.__init__.__code__ is original.H4TrackingController.__init__.__code__
    assert binding.TrackingController.step is original.H4TrackingController.step
    assert binding.PathReference.project is original.PathReference.project
    assert binding.PathReference.at is original.PathReference.at
    assert original.PathReference.__init__.__globals__==before
    assert binding.GUARDS==dict(original.GUARDS,maximum_simulation_duration_s=300.)
    assert binding.CONTROLLER_CONFIG==original.CONTROLLER_CONFIG
    assert binding.ACCEPTANCE==original.ACCEPTANCE and binding.COMMAND_LIMITS==original.COMMAND_LIMITS
    assert original.GUARDS['maximum_simulation_duration_s']==180.
    assert binding.physical_reverse_and_direction_change_policy_NOT_implemented
    assert binding.current_binding_is_NOT_a_motion_authorization


@pytest.mark.parametrize('fault',['scenario','world_start','world_goal','local_frame','raster','reverse',
    'bool_direction','float_direction','speed','acceleration','state_mode','yaw','curvature','empty',
    'physical_claim','sensor_claim','pending_claim','authorization','statistics','explicit_footprint'])
def test_H6_forward_adapter_rejects_mutations_without_overriding_the_original_plan(fault):
    import copy
    geometry,plan=_forward_v3_geometry_and_plan()
    if fault=='scenario':plan['scenario']='H5_obstacle_bypass_return'
    elif fault=='world_start':plan['start_world_xyyaw'][0]=.001
    elif fault=='world_goal':plan['goal_world_xyyaw'][0]=.301
    elif fault=='local_frame':plan['goal_local_xyyaw'][0]=.301
    elif fault=='raster':plan['planning_occupancy_digest']='other'
    elif fault=='reverse':plan['selected_actions'][0]['direction']=-1
    elif fault=='bool_direction':plan['selected_actions'][0]['direction']=True
    elif fault=='float_direction':plan['selected_actions'][0]['direction']=1.
    elif fault=='speed':plan['selected_actions'][0]['target_speed_m_s']=.051
    elif fault=='acceleration':plan['rigid_body_parameter_bounds']['max_longitudinal_accel_m_s2']=.021
    elif fault=='state_mode':plan['selected_states'][0]['mode']='front_only'
    elif fault=='yaw':plan['selected_states'][0]['yaw_deg']=float('nan')
    elif fault=='curvature':plan['selected_actions'][0]['curvature_1_m']=float('inf')
    elif fault=='empty':plan['selected_actions']=[]
    elif fault=='physical_claim':plan['physical_execution_verified']=True
    elif fault=='sensor_claim':plan['Nav2_execution_verified']=True
    elif fault=='pending_claim':plan['integrated_stage_portal_bypass_and_continuous_executor_validation_pending']=False
    elif fault=='authorization':plan['actual_motion_authorized']=True
    elif fault=='statistics':plan['use_for_final_statistics']=True
    else:plan['footprint']=dict(length_m=.010,width_m=.965,clearance_m=.010)
    saved=copy.deepcopy(plan);binding=forward_common_v3_execution_binding(geometry,simulation_budget_s=300.)
    with pytest.raises(ValueError):binding.TrackingController(plan)
    # NaN is not equal to itself; compare its faithful textual representation.
    assert repr(plan)==repr(saved)


@pytest.mark.parametrize('duration',[True,0.,179.,301.,600.,float('inf'),float('nan')])
def test_H6_forward_MODEL_duration_requires_explicit_common_resource_binding(duration):
    geometry,_=_forward_v3_geometry_and_plan()
    with pytest.raises(ValueError):forward_common_v3_execution_binding(geometry,simulation_budget_s=duration)


def test_H6_nominal_case_uses_exact_original_full_footprint_support_subtraction_and_case_code():
    import screen_hospital_h4_common_executor as original
    geometry,plan=_forward_v3_geometry_and_plan()
    before=dict(original.case.__globals__)
    run,binding,design=mission_nominal_v3_case_binding(geometry,simulation_budget_s=300.)
    assert run.__code__ is original.case.__code__ and original.case.__globals__==before
    covers=run.__globals__['bundle'].cohort.support_covers(design)
    area=run.__globals__['bundle'].cohort.support_area_missing
    assert area is original.bundle.cohort.support_area_missing
    assert area([0.,0.,0.],covers)<1e-9
    assert area([0.,2.,0.],covers)>.5  # full body, not a tiny positional .010m footprint
    assert covers==geometry['same_three_authored_support_rectangles']
    covers[0][0][0]=999.
    assert geometry['same_three_authored_support_rectangles'][0][0][0]==-2.
    with pytest.raises(ValueError):run.__globals__['bundle'].cohort.support_covers({})
    binding.validate_plan(plan)


def test_H6_nominal_binding_rejects_crossed_or_missing_authored_support_geometry():
    geometry,_=_forward_v3_geometry_and_plan()
    geometry['same_three_authored_support_rectangles']=geometry['same_three_authored_support_rectangles'][:2]
    with pytest.raises(ValueError):mission_nominal_v3_case_binding(geometry,simulation_budget_s=300.)
    geometry,_=_forward_v3_geometry_and_plan()
    geometry['same_three_authored_support_rectangles'][0]=[[-2.,-2.],[2.,2.],[2.,-2.],[-2.,2.]]
    with pytest.raises(ValueError):mission_nominal_v3_case_binding(geometry,simulation_budget_s=300.)


@pytest.mark.parametrize('fault',['scenario','body','clearance','live_claim','statistics'])
def test_H6_forward_binding_rejects_geometry_scope_or_footprint_waivers(fault):
    geometry,_=_forward_v3_geometry_and_plan()
    if fault=='scenario':geometry['scenario']='H4_ward_corridor_turn'
    elif fault=='body':geometry['full_body_rectangle_m']=[.010,.965]
    elif fault=='clearance':geometry['full_body_clearance_m']=.009
    elif fault=='live_claim':geometry['actual_motion_authorized']=True
    else:geometry['use_for_final_statistics']=True
    with pytest.raises(ValueError):forward_common_v3_execution_binding(geometry,simulation_budget_s=300.)


def test_H6_MODEL_export_is_offline_only_before_any_file_or_output_access(monkeypatch,tmp_path):
    import sys
    from hospital_mission_planning import export_common_v3_mission_model_screen
    monkeypatch.setitem(sys.modules,'omni.kit.app',object())
    with pytest.raises(RuntimeError,match='offline-only'):
        export_common_v3_mission_model_screen(tmp_path/'absent-plan',tmp_path/'absent-prep',
            simulation_budget_s=300.,output=tmp_path/'must-not-reserve')
    assert list(tmp_path.iterdir())==[]


def test_H6_original_nominal_case_scalar_normalization_preserves_ALL_outcomes_and_raw_values(tmp_path):
    import numpy as np
    from hospital_h3_turn_planning import model_bounds
    from hospital_h4_subfloor_compact_path_launch import normalize_runtime_evidence
    from hospital_runtime_evidence_io import json_value,write_exclusive
    geometry,plan=_forward_v3_geometry_and_plan()
    run,_,design=mission_nominal_v3_case_binding(geometry,simulation_budget_s=300.)
    report,rows=run(plan,model_bounds(),.020,np.zeros((400,400),bool),np.array([-4.,-4.]),.020,design)
    with pytest.raises(TypeError,match='float64'):json_value(report)
    conversions=[];normalized=normalize_runtime_evidence(report,conversions)
    assert normalized['model_status']==report['model_status']
    assert normalized['checks']==report['checks'] and normalized['error']==report['error']
    assert len(rows)>2 and conversions and all(c['no_rounding_or_value_change'] for c in conversions)
    assert {c['path'] for c in conversions}=={'$.maximum_friction_utilization','$.maximum_abs_ltr',
        '$.maximum_actuator_utilization','$.minimum_normal_load_fraction'}
    for c in conversions:assert c['value']==report[c['path'][2:]]
    assert write_exclusive(tmp_path/'MODEL-only-scalar-fixture.json',normalized)==normalized
    assert normalized['physical_motion_or_whole_route_admission'] is False


def test_MODEL_outline_diagnostic_separates_10mm_safety_margin_from_raw_body_without_waiver():
    import numpy as np
    grid=np.zeros((400,400),bool);grid[:,200:]=True
    # Occupied half-plane starts at x=0; raw body ends at -5mm,
    # original +10mm safety outline crosses it. No motion is executed.
    result=model_clearance_pose_comparison(grid,[-2.,-2.],.010,[-.820,0.,0.],[-.840,0.,0.])
    body,reference=result['recorded_body'],result['same_progress_reference']
    assert not body['raw_body_intersects_recorded_raster']
    assert body['required_10mm_expanded_outline_intersects_recorded_raster']
    assert not reference['required_10mm_expanded_outline_intersects_recorded_raster']
    assert body['sampled_SAT_expansion_free_lower_bound_m']<=.005
    assert body['sampled_SAT_expansion_collision_upper_bound_m']>=.005-1e-10
    assert body['bound_is_raster_expansion_NOT_true_mesh_or_cooked_clearance']
    assert result['translation_only']==body and result['heading_only']==reference


def test_H6_binding_tracking_outputs_equal_original_v3_for_identical_spatial_reference():
    import hospital_h4_path_execution as original
    geometry,plan=_forward_v3_geometry_and_plan()
    binding=forward_common_v3_execution_binding(geometry,simulation_budget_s=300.)
    other=dict(plan,scenario=original.SCENARIO,footprint=dict(original.FOOTPRINT),
        common_max_acceleration_m_s2=.02,native_MODEL_success_is_NOT_actual_support_or_drive_admission=True)
    first,second=binding.TrackingController(plan),original.H4TrackingController(other)
    for i in range(20):
        args=(i*.02,(i*.0001,0.,0.),.005,(0.,0.))
        assert first.step(*args)==second.step(*args)


def _handoff_geometry_plan():
    return dict(method='P_dynamics_full_envelope',scenario='H6_integrated_transfer_mission',plan_success=True,
        start_world_xyyaw=[0.,0.,0.],goal_world_xyyaw=[.11,0.,0.],
        selected_actions=[dict(mode='coordinated_counterphase',direction=1,curvature_1_m=0.,target_speed_m_s=.05)]*2,
        selected_states=[dict(world_xyyaw=[.05,0.,0.]),dict(world_xyyaw=[.10,0.,0.])])


def test_v3_terminal_geometry_keeps_original25mm_handoff_and_distinguishes_capture_from_hold():
    import copy
    plan=_handoff_geometry_plan();before=copy.deepcopy(plan);report=common_v3_terminal_geometry_review(plan)
    handoff,endpoint=report['samples'];assert plan==before
    assert handoff['goal_position_error_m']==pytest.approx(.035)
    assert not handoff['ideal_reference_pose_inside_existing_stopped_capture_band']
    assert endpoint['goal_position_error_m']==pytest.approx(.01)
    assert endpoint['ideal_reference_pose_inside_existing_stopped_capture_band']
    assert report['existing_v3_config']['stationary_hold_entry_position_m']==.030
    assert report['existing_v3_config']['stationary_hold_entry_heading_deg']==2.
    assert report['out_of_band_handoff_NOT_automatic_FAIL_existing_bounded_terminal_closure_still_required']
    assert report['in_band_model_endpoint_NOT_actual_hold_PASS'] and not report['actual_motion_authorized']
    assert report['braking_tracking_alignment_pulses_and_actual_hold_NOT_simulated']


@pytest.mark.parametrize('fault',['scenario','missing_state','nonfinite_goal','reverse'])
def test_v3_terminal_geometry_refuses_unsupported_or_malformed_model_inputs(fault):
    plan=_handoff_geometry_plan()
    if fault=='scenario':plan['scenario']='H5_obstacle_bypass_return'
    elif fault=='missing_state':plan['selected_states']=[]
    elif fault=='nonfinite_goal':plan['goal_world_xyyaw'][0]=float('nan')
    else:plan['selected_actions']=[dict(a,direction=-1) for a in plan['selected_actions']]
    with pytest.raises(ValueError):common_v3_terminal_geometry_review(plan)


def test_whole_H6_ROI_differs_from_legacy_numeric_bounds_not_a_collision_failure():
    start=[14.192329139709472,16.568600039835417,0.]
    goal=[24.31814346169873,11.873745033615302,0.]
    b=mission_search_bounds([11.624561730535497,27.055793535543966,11.055421818866616,31.96082240859503],start,goal)
    assert b['local_goal_xyyaw'][1]<-2.
    assert b['minimum_y']<b['local_goal_xyyaw'][1]<b['maximum_y']
    assert b['minimum_x']<0.<b['local_goal_xyyaw'][0]<b['maximum_x']
    assert b['entire_original_ROI_retained'] and b['local_AABB_is_NOT_free_space_or_a_collision_certificate']


def test_original_lattice_uses_map_checker_instead_of_numeric_Y_wall_bounds():
    # Merely being outside H1's numeric +/-2m Y arguments does NOT imply
    # failure when an explicit map checker is supplied. X is still bounded.
    from patient_initial_stability.steering_lattice import plan,KINEMATIC_DYNAMICS
    result=plan((.15,0.,0.),1.,2.,allowed_modes=('coordinated_counterphase',),
        allowed_curvatures=(0.,),dynamics=KINEMATIC_DYNAMICS,node_limit=50,
        minimum_x=0.,maximum_x=1.,collision_checker=lambda poses:False)
    assert result is not None


@pytest.mark.parametrize('yaw',[0.,.7,math.pi/2,math.pi,-math.pi/2])
def test_all_original_ROI_corners_retained_in_rotated_local_bounds(yaw):
    roi=[1.,9.,2.,12.];start=[3.,5.,yaw];goal=[7.,10.,yaw]
    b=mission_search_bounds(roi,start,goal);c,s=math.cos(yaw),math.sin(yaw)
    for x in roi[:2]:
        for y in roi[2:]:
            u,v=c*(x-start[0])+s*(y-start[1]),-s*(x-start[0])+c*(y-start[1])
            assert b['minimum_x']-1e-12<=u<=b['maximum_x']+1e-12
            assert b['minimum_y']-1e-12<=v<=b['maximum_y']+1e-12


@pytest.mark.parametrize('roi,start,goal',[
    ([2.,1.,0.,2.],[1.5,1.,0.],[1.5,1.,0.]),
    ([0.,2.,0.,2.],[-.1,1.,0.],[1.,1.,0.]),
    ([0.,2.,0.,2.],[1.,1.,0.],[2.1,1.,0.]),
    ([0.,float('inf'),0.,2.],[1.,1.,0.],[1.,1.,0.]),
    ([0.,2.,0.,2.],[1.,1.,float('nan')],[1.,1.,0.])])
def test_invalid_scope_fails_without_changing_goal_or_ROI(roi,start,goal):
    with pytest.raises(ValueError):mission_search_bounds(roi,start,goal)


@pytest.mark.parametrize('directions',[(1,),(-1,),(1,-1)])
def test_explicit_common_direction_set_and_all_non_scope_parameters_preserved(directions):
    b=mission_search_bounds([0.,20.,0.,20.],[1.,1.,0.],[12.,14.,0.]);calls=[]
    def old(*args,**kwargs):calls.append((args,kwargs));return 'SOFTWARE_FIXTURE'
    run=bounded_lattice_function(old,b,directions=directions)
    checker=lambda poses:False
    kwargs=dict(minimum_x=-.3,maximum_x=11.4,directions=(1,),collision_checker=checker,
        allowed_modes=('rear_only','front_only'),allowed_curvatures=(-.7,0.,.7),speed_m_s=.05,node_limit=20000)
    before=dict(kwargs)
    assert run(tuple(b['local_goal_xyyaw']),-2.,2.,**kwargs)=='SOFTWARE_FIXTURE'
    assert kwargs==before and calls[0][0][1:]==(b['minimum_y'],b['maximum_y'])
    observed=calls[0][1]
    assert observed['directions']==directions and observed['minimum_x']==b['minimum_x'] and observed['maximum_x']==b['maximum_x']
    assert all(observed[k]==kwargs[k] for k in ('collision_checker','allowed_modes','allowed_curvatures','speed_m_s','node_limit'))


@pytest.mark.parametrize('directions',[(),(0,),(True,),(1,1),('1',)])
def test_no_implicit_or_malformed_direction_set(directions):
    b=mission_search_bounds([0.,2.,0.,2.],[.5,.5,0.],[1.,1.,0.])
    with pytest.raises(ValueError):bounded_lattice_function(lambda *a,**k:None,b,directions=directions)


def test_no_bounds_only_checker_or_changed_goal_or_original_global_mutation():
    b=mission_search_bounds([0.,20.,0.,20.],[1.,1.,0.],[12.,14.,0.])
    calls=[]
    def lattice(*a,**kw):calls.append(1)
    run=bounded_lattice_function(lattice,b,directions=(1,-1))
    with pytest.raises(ValueError):run(b['local_goal_xyyaw'],-2.,2.)
    with pytest.raises(ValueError):run([10.,13.,0.],-2.,2.,collision_checker=lambda p:False)
    assert not calls
    def method():return plan,HospitalEnvelopeChecker,pose_guidance,counterphase_connector
    before=dict(method.__globals__)
    bound=bind_plan_method(method,lattice,b,checker_class='FIXTURE',guidance='FIXTURE',directions=(1,-1))
    assert bound.__code__ is method.__code__ and method.__globals__==before
    assert bound()[1:3]==('FIXTURE','FIXTURE') and bound()[3](None,None) is None


@pytest.fixture
def resource_binding():
    from patient_initial_stability import steering_lattice as lattice
    State,propagate,key=resource_propagation_binding(lattice.State,lattice.propagate,MotionResourceLimits())
    return lattice,State,propagate,key


@pytest.mark.parametrize('mode',['coordinated_counterphase','front_only','rear_only'])
@pytest.mark.parametrize('direction',[-1,1])
def test_resource_wrapper_keeps_every_original_kinematic_field_pose_and_time_exact(resource_binding,mode,direction):
    from dataclasses import fields
    lattice,State,propagate,_=resource_binding
    original=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    expected,poses,duration=lattice.propagate(original,mode,.7,direction,dynamics=lattice.KINEMATIC_DYNAMICS)
    actual,samples,elapsed=propagate(state,mode,.7,direction,dynamics=lattice.KINEMATIC_DYNAMICS)
    assert all(getattr(actual,f.name)==getattr(expected,f.name) for f in fields(lattice.State))
    assert list(samples)==poses and duration==elapsed and samples.resource_feasible
    if direction<0:
        assert actual.mission_reverse_episode_bins*.005>=samples.actual_edge_center_arc_m-1e-12
    else:assert actual.mission_reverse_episode_bins==actual.mission_reverse_total_bins==0


def test_reverse_episode_total_and_direction_history_persist_without_resets(resource_binding):
    lattice,State,propagate,_=resource_binding
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    def step(direction,count):
        nonlocal state
        for _ in range(count):state,samples,_=propagate(state,'coordinated_counterphase',0.,direction,dynamics=lattice.KINEMATIC_DYNAMICS)
        return samples
    assert step(-1,10).resource_feasible
    assert state.mission_reverse_episode_bins==100 and state.mission_reverse_total_bins==100
    assert step(1,1).resource_feasible and state.mission_reverse_episode_bins==0
    assert step(-1,10).resource_feasible and state.mission_reverse_total_bins==200
    assert step(1,1).resource_feasible
    samples=step(-1,1)
    assert samples.resource_checks==dict(reverse_episode=True,reverse_total=False,direction_changes=True)
    assert state.mission_direction_changes==4 and state.rigid_body_feasible
    samples=step(1,1)
    assert not samples.resource_checks['direction_changes']


def test_reverse_mode_change_does_not_recapture_episode_or_total(resource_binding):
    lattice,State,propagate,_=resource_binding
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    for _ in range(10):state,_,_=propagate(state,'coordinated_counterphase',0.,-1,dynamics=lattice.KINEMATIC_DYNAMICS)
    state,poses,_=propagate(state,'rear_only',0.,-1,dynamics=lattice.KINEMATIC_DYNAMICS)
    assert not poses.resource_checks['reverse_episode'] and state.mission_reverse_episode_bins==110
    assert state.mission_direction_changes==0


def test_resource_histories_are_part_of_dominance_not_merged_by_same_body_pose(resource_binding):
    from dataclasses import replace
    lattice,State,_,key=resource_binding
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    other=replace(state,mission_reverse_total_bins=10)
    assert lattice._state_key(state)==lattice._state_key(other)
    assert key(lattice._state_key)(state)!=key(lattice._state_key)(other)


class FreeCheckerFixture:
    def __init__(self,*a,**k):self.statistics={};self.map_calls=0;self.deadline_calls=0
    def check_deadline(self):self.deadline_calls+=1
    def __call__(self,poses):self.map_calls+=1;return False


def test_resource_guard_rejections_are_not_reported_as_obstacle_or_rigid_body_failures(resource_binding):
    lattice,State,propagate,_=resource_binding
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
    for _ in range(11):state,samples,_=propagate(state,'coordinated_counterphase',0.,-1,dynamics=lattice.KINEMATIC_DYNAMICS)
    checker=resource_checker_class(FreeCheckerFixture)()
    assert checker(samples) and checker.map_calls==0 and checker.deadline_calls==1
    assert checker.statistics['motion_resource_rejected_edges_NOT_obstacle_collisions']==1
    assert checker.statistics['motion_resource_reverse_episode_rejections']==1 and state.rigid_body_feasible
    assert checker([(0.,0.,0.)]) is False and checker.map_calls==1


def test_resource_history_is_used_inside_original_P_refinement_code_not_only_after_selection(resource_binding):
    lattice,_,_,_=resource_binding
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route as refine
    original_globals=dict(refine.__globals__)
    search,bound=bind_resource_kernels(lattice.plan,refine,lattice.State,lattice.propagate,lattice._state_key,MotionResourceLimits())
    assert search.__code__ is lattice.plan.__code__ and bound.__code__ is refine.__code__
    actions=[dict(mode='coordinated_counterphase',curvature_1_m=0.,direction=-1)]*11
    args=('fixture',actions,(-.55,0.,0.),(.003,.05),None)
    kw=dict(terminal_speed_tolerance_m_s=.005,terminal_deceleration_m_s2=.02,
        step_m=.05,terminal_extension_edges=0,maximum_iterations=1,dynamics=lattice.KINEMATIC_DYNAMICS)
    plain,_=refine(*args,collision_checker=FreeCheckerFixture(),**kw)
    checker=resource_checker_class(FreeCheckerFixture)()
    restricted,diagnostic=bound(*args,collision_checker=checker,**kw)
    assert plain is not None and restricted is None
    assert diagnostic['collision']>0 and checker.statistics['motion_resource_reverse_episode_rejections']>0
    assert refine.__globals__==original_globals


@pytest.mark.parametrize('kw',[
    {'maximum_reverse_episode_m':0.},{'maximum_total_reverse_m':.4},
    {'maximum_direction_changes':True},{'distance_quantum_m':float('nan')},
    {'distance_quantum_m':.6}])
def test_bad_resource_definitions_rejected_without_any_search(kw):
    with pytest.raises(ValueError):MotionResourceLimits(**kw)


def test_reverse_heuristic_uses_virtual_copy_only_and_does_not_move_real_pose_or_goal():
    from patient_initial_stability.steering_lattice import State
    state=State(2.,3.,math.pi,'coordinated_counterphase',0.,0.,True)
    goal=(4.,3.,math.pi);calls=[]
    def original(s,g):calls.append((s,g));return abs(s.yaw)+abs(g[2])
    score=bidirectional_search_guidance(original)(state,goal)
    assert score==pytest.approx(0.,abs=1e-12) and len(calls)==2
    assert calls[0]==(state,goal) and calls[1][0] is not state
    assert calls[1][0].x==state.x and calls[1][0].y==state.y and state.yaw==goal[2]==math.pi


def test_refinement_switch_cost_overlay_does_not_change_actions_states_or_edge_times():
    from patient_initial_stability.hierarchical_dynamics_planner import DynamicRouteResult
    actions=tuple(dict(direction=d) for d in (1,-1,-1,1))
    old=DynamicRouteResult('fixture',(),actions,(.01,)*4,(1.,)*4,10.,())
    new,count=with_direction_change_cost(old,1.)
    assert count==2 and new.cost_s==12. and old.cost_s==10.
    assert new.actions is old.actions and new.states is old.states and new.edge_times_s is old.edge_times_s
    assert with_direction_change_cost(None,1.)==(None,0)


@pytest.mark.parametrize('direction',[-1,1])
def test_wrapper_keeps_full_Newton_Euler_edge_and_steering_state_exact(resource_binding,direction):
    from dataclasses import fields
    from hospital_h3_turn_planning import model_bounds
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    lattice,State,propagate,_=resource_binding
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    old=lattice.State(0.,0.,0.,'coordinated_counterphase',.1,-.1,True,1,.025,0.,0.)
    wrapped=State(**{f.name:getattr(old,f.name) for f in fields(lattice.State)})
    kw=dict(dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=.03)
    expected,poses,elapsed=lattice.propagate(old,'rear_only',.25,direction,**kw)
    actual,samples,duration=propagate(wrapped,'rear_only',.25,direction,**kw)
    assert all(getattr(actual,f.name)==getattr(expected,f.name) for f in fields(lattice.State))
    assert list(samples)==poses and duration==elapsed


def _mission_guidance_fixture(goal=(10.,0.,0.),start=(0.,0.,0.),**settings):
    from types import SimpleNamespace
    import numpy as np
    from patient_initial_stability.steering_lattice import State
    c,s=math.cos(start[2]),math.sin(start[2])
    world_goal=(start[0]+c*goal[0]-s*goal[1],start[1]+s*goal[0]+c*goal[1],start[2]+goal[2])
    field=SimpleNamespace(start=start,goal=world_goal,origin=np.array([-20.,-20.]),resolution=1.,
        distance=np.full((60,60),10.),targets=np.tile([1.,0.],(60,60,1)))
    calls=[]
    def fallback(state,actual_goal):calls.append((state,actual_goal));return 999.
    def make(**overrides):
        values=dict(directions=(1,-1),speed_m_s=.05,maximum_curvature_1_m=.7,heading_weight=.4,
            terminal_zone_m=1.63,terminal_position_tolerance_m=.025)
        values.update(settings);values.update(overrides)
        return mission_local_heading_guidance(field,fallback,goal,**values)
    return field,State(0.,0.,0.,'coordinated_counterphase',0.,0.,False),calls,make


def test_mission_far_field_uses_local_bearing_not_premature_final_heading():
    for final_yaw in (0.,math.pi/2,math.pi):
        field,state,calls,make=_mission_guidance_fixture(goal=(10.,0.,final_yaw))
        assert make()(state,(10.,0.,final_yaw))==pytest.approx(200.)
        assert not calls and state.yaw==0. and field.goal[2]==final_yaw


def test_mission_final_BODY_yaw_blends_near_goal_not_reverse_travel_yaw():
    from dataclasses import replace
    field,state,calls,make=_mission_guidance_fixture(goal=(1.,0.,0.))
    field.targets[:]=[-1.,0.]
    actual=replace(state,x=.5)
    both=make()(actual,(1.,0.,0.))
    forward=make(directions=(1,))(actual,(1.,0.,0.))
    assert both==pytest.approx(200.) and forward>both and not calls
    # Reverse travel facing -X is not a BODY goal-yaw waiver.
    opposite=replace(actual,yaw=math.pi)
    assert make()(opposite,(1.,0.,0.))>both
    assert opposite.yaw==math.pi and actual.yaw==0.


def test_mission_terminal_tolerance_preserves_exact_original_goal_heuristic():
    from dataclasses import replace
    field,state,calls,make=_mission_guidance_fixture(goal=(1.,0.,0.))
    actual=replace(state,x=.99,yaw=1.)
    assert make()(actual,(1.,0.,0.))==999. and calls==[(actual,(1.,0.,0.))]


def test_mission_guidance_does_not_mutate_state_field_goal_or_original_fallback():
    import numpy as np
    from dataclasses import asdict
    field,state,calls,make=_mission_guidance_fixture()
    before=asdict(state);distance=field.distance.copy();targets=field.targets.copy()
    first=make()(state,(10.,0.,0.));second=make()(state,(10.,0.,0.))
    assert first==second and asdict(state)==before and not calls
    assert np.array_equal(field.distance,distance) and np.array_equal(field.targets,targets)


@pytest.mark.parametrize('where',['outside','unreachable'])
def test_mission_missing_relaxed_guidance_uses_original_not_a_false_collision(where):
    from dataclasses import replace
    field,state,calls,make=_mission_guidance_fixture()
    if where=='outside':state=replace(state,x=100.)
    else:field.distance[:]=math.inf
    assert make()(state,(10.,0.,0.))==999. and calls==[(state,(10.,0.,0.))]


@pytest.mark.parametrize('corrupt',['nan_distance','negative_distance','negative_infinity','nan_target'])
def test_mission_corrupt_finite_field_fails_instead_of_hidden_fallback(corrupt):
    field,state,calls,make=_mission_guidance_fixture()
    if corrupt=='nan_target':field.targets[:]=math.nan
    else:field.distance[:]=dict(nan_distance=math.nan,negative_distance=-1.,negative_infinity=-math.inf)[corrupt]
    with pytest.raises(ValueError):make()(state,(10.,0.,0.))
    assert not calls


@pytest.mark.parametrize('settings',[
    dict(directions=()),dict(directions=(True,)),dict(directions=(1,1)),dict(directions=(0,)),
    dict(speed_m_s=0.),dict(maximum_curvature_1_m=math.inf),dict(heading_weight=-.1),
    dict(terminal_zone_m=.025),dict(terminal_position_tolerance_m=0.)])
def test_mission_guidance_requires_explicit_valid_resources_and_scope(settings):
    _,_,_,make=_mission_guidance_fixture(**settings)
    with pytest.raises(ValueError):make()


def test_mission_guidance_rejects_wrong_fixed_goal_reference_and_nonfinite_state():
    from dataclasses import replace
    field,state,calls,make=_mission_guidance_fixture()
    run=make()
    with pytest.raises(ValueError):run(state,(9.,0.,0.))
    with pytest.raises(ValueError):run(replace(state,x=math.nan),(10.,0.,0.))
    field.goal=(9.,0.,0.)
    with pytest.raises(ValueError):make()
    assert not calls


def test_mission_rotated_world_reference_uses_local_heading_and_same_fixed_goal():
    field,state,calls,make=_mission_guidance_fixture(start=(5.,7.,math.pi/2))
    field.targets[:]=[5.,8.]
    assert make()(state,(10.,0.,0.))==pytest.approx(200.) and not calls


def test_mission_priority_barrier_reduction_is_guidance_only_not_a_terminal_waiver():
    # Reproduce two fixed H6 straight-prefix observations without a new search.
    # The original finite-cell SAT separately verified all2m of this prefix.
    old=[];new=[]
    for edge_count,distance_m,bearing_deg in [(10,11.929991334482208,-9.97971038879206),
            (25,11.169991334482225,-45.98582915990483)]:
        field,state,calls,make=_mission_guidance_fixture()
        field.distance[:]=distance_m
        angle=math.radians(bearing_deg);field.targets[:]=[math.cos(angle),math.sin(angle)]
        legacy=distance_m/.05+.4*(2*abs(angle))/(.7*.05)
        old.append(edge_count+1.5*legacy)
        new.append(edge_count+1.5*make()(state,(10.,0.,0.)))
        assert not calls
    assert old[1]-old[0]>13. and new[1]-new[0]<3.


@pytest.fixture
def mission_offline_probe_driver():
    import runpy
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    path=root/'studies/IJCAS/results/hospital_paper_gate/hospital_h6_planning/review_matched_probe_driver.txt'
    return runpy.run_path(str(path),run_name='software_fixture_NO_main')


@pytest.mark.parametrize('selection',['unknown_revision','new_in_original_folder','outside_evidence_parent'])
def test_offline_probe_invalid_selection_cannot_reuse_old_outcomes_or_start_search(mission_offline_probe_driver,selection):
    from types import FunctionType
    from pathlib import Path
    original=mission_offline_probe_driver['validate_probe_selection'];namespace=dict(original.__globals__)
    if selection=='unknown_revision':namespace['GUIDANCE_REVISION']='unsupported'
    elif selection=='new_in_original_folder':namespace['GUIDANCE_REVISION']='local_heading_terminal_blend_v1'
    else:namespace['OUTPUT']=Path('/tmp/not_the_planning_evidence_parent')
    run=FunctionType(original.__code__,namespace,original.__name__)
    with pytest.raises(ValueError):run()
    assert original.__globals__['GUIDANCE_REVISION']=='legacy_bidirectional_virtual_heading_v1'


def test_offline_probe_new_guidance_parameters_shared_without_model_or_global_mutation(mission_offline_probe_driver):
    from types import FunctionType,SimpleNamespace
    original=mission_offline_probe_driver['make_common_guidance'];before=dict(original.__globals__);calls=[]
    def factory(*a,**kw):calls.append((a,kw));return 'GUIDANCE_FIXTURE_NOT_PLAN'
    namespace=dict(before,GUIDANCE_REVISION='local_heading_terminal_blend_v1',
        binding=SimpleNamespace(mission_local_heading_guidance=factory))
    run=FunctionType(original.__code__,namespace,original.__name__)
    run.__kwdefaults__=dict(original.__kwdefaults__)
    goal=[10.,-4.,0.];field=object()
    assert run(field,{'local_goal_xyyaw':goal})=='GUIDANCE_FIXTURE_NOT_PLAN'
    assert calls==[((field,before['original'].pose_guidance,goal),before['LOCAL_GUIDANCE_PARAMETERS'])]
    assert original.__globals__==before


def test_completed_original_offline_probe_is_verified_not_run_again(mission_offline_probe_driver):
    # Fail before calling main if the historical fixture is absent. It is
    # never permissible for a missing test fixture to launch a real search.
    root=mission_offline_probe_driver['OUTPUT']
    assert (root/'planning_review.json').exists()
    mission_offline_probe_driver['main']()


def test_original_geometry_binding_default_and_explicit_adopted_variant_are_separate(mission_offline_probe_driver):
    import json
    from types import FunctionType
    driver=mission_offline_probe_driver;function=driver['validate_geometry_binding']
    original=json.loads(driver['ORIGINAL_GEOMETRY'].read_bytes())
    assert function(original) is None and driver['GEOMETRY']==driver['ORIGINAL_GEOMETRY']
    path=driver['GATE']/'hospital_h6_task_geometry/0e5f25f64d2256c2/common_geometry.json'
    assert path.exists(),'Only read the completed offline map fixture; NEVER launch its exporter from a test'
    scoped=FunctionType(function.__code__,dict(function.__globals__,GEOMETRY=path),function.__name__)
    result=scoped(json.loads(path.read_bytes()))
    assert result['effective_opening_m']==1.5
    assert result['NEW_common_environment_NOT_original_task_success_or_physical_admission']
    assert result['old_baseline_reference_is_kernel_lineage_NOT_matched_performance_under_this_new_layout']


@pytest.mark.parametrize('fault',['start','goal','body','clearance','support','ROI','raster','methods',
    'original_hash','definition_hash','unknown_definition','narrow','nonfinite','claim_actual','missing_new_scope'])
def test_new_layout_binding_rejects_unapproved_goal_model_map_or_scope_changes(mission_offline_probe_driver,fault):
    import json
    from types import FunctionType
    driver=mission_offline_probe_driver;function=driver['validate_geometry_binding']
    path=driver['GATE']/'hospital_h6_task_geometry/0e5f25f64d2256c2/common_geometry.json'
    assert path.exists();manifest=json.loads(path.read_bytes())
    if fault=='start':manifest['start_xyyaw'][0]+=.1
    elif fault=='goal':manifest['goal_xyyaw'][1]+=.1
    elif fault=='body':manifest['full_body_rectangle_m'][1]=.76
    elif fault=='clearance':manifest['full_body_clearance_m']=0.
    elif fault=='support':manifest['fixed_unknown_support_and_ROI_cell_count']=0
    elif fault=='ROI':manifest['source_ROI_xy_m'][0]-=1.
    elif fault=='raster':manifest['raster']['resolution_m']=.02
    elif fault=='methods':manifest['common_methods']=manifest['common_methods'][-1:]
    elif fault=='original_hash':manifest['original_common_geometry_sha256']='0'*64
    elif fault=='definition_hash':manifest['task_definition_sha256']='0'*64
    elif fault=='unknown_definition':manifest['task_definition']='/tmp/unadopted_definition.json'
    elif fault=='narrow':manifest['effective_destination_opening_m']=1.49
    elif fault=='nonfinite':manifest['effective_destination_opening_m']=float('nan')
    elif fault=='claim_actual':manifest['actual_motion_authorized']=True
    else:manifest.pop('environment_variant_is_NEW_not_original_task_success')
    scoped=FunctionType(function.__code__,dict(function.__globals__,GEOMETRY=path),function.__name__)
    with pytest.raises(ValueError):scoped(manifest)


def test_completed_original_probe_cannot_be_reused_for_a_new_environment(mission_offline_probe_driver):
    from types import FunctionType
    driver=mission_offline_probe_driver;function=driver['validate_probe_selection']
    path=driver['GATE']/'hospital_h6_task_geometry/0e5f25f64d2256c2/common_geometry.json'
    scoped=FunctionType(function.__code__,dict(function.__globals__,GEOMETRY=path),function.__name__)
    with pytest.raises(ValueError,match='original-environment'):scoped()


def _resource_label(episode,total,changes,physical='pose',direction=1):
    return (physical,direction,episode,total,changes)


def test_resource_pareto_removes_only_no_cheaper_no_better_resource_label():
    high=_resource_label(0,10,2);low=_resource_label(0,5,1)
    labels=ResourceParetoBest({high:12.})
    labels[low]=11.
    assert len(labels)==1 and labels.get(high)==11. and labels.get(low)==11.
    assert labels.snapshot()['dominated_labels_removed']==1
    assert labels.snapshot()['strict_resource_dominance_hits']==1
    labels[high]=13.
    assert len(labels)==1 and labels.snapshot()['dominated_insertions_ignored']==1


def test_resource_pareto_retains_cheaper_high_resource_and_costlier_low_resource_alternatives():
    high=_resource_label(0,10,2);low=_resource_label(0,5,1)
    labels=ResourceParetoBest({high:10.,low:12.})
    assert len(labels)==2 and labels.get(high)==10. and labels.get(low)==12.
    # A cheaper path with a worse individual resource must not dominate.
    episode=_resource_label(2,5,0);switches=_resource_label(0,5,1)
    other=ResourceParetoBest({episode:5.,switches:6.})
    assert len(other)==2 and other.get(switches)==6.


def test_resource_pareto_never_merges_original_physical_mode_direction_or_steering_keys():
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    State,_,key=resource_propagation_binding(lattice.State,lattice.propagate,MotionResourceLimits())
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,1)
    original_key=key(lattice._state_key);labels=ResourceParetoBest({original_key(state):1.})
    variants=[replace(state,x=.05),replace(state,yaw=.1),replace(state,mode='rear_only'),
        replace(state,front=.2),replace(state,rear=.2),replace(state,direction=-1)]
    for other in variants:assert labels.get(original_key(other)) is None
    assert state.mission_reverse_total_bins==state.mission_direction_changes==0


@pytest.mark.parametrize('key',[(),('p',0,0),('p',True,1,0),('p',-1,0,0),
    ('p',2,1,0),('p',0.,0,0),(['unhashable'],0,0,0)])
def test_resource_pareto_malformed_or_reset_counter_keys_fail_closed(key):
    with pytest.raises(ValueError):ResourceParetoBest().__setitem__(key,1.)


@pytest.mark.parametrize('cost',[-1.,math.nan,math.inf,True])
def test_resource_pareto_never_changes_invalid_path_cost_into_an_accepted_label(cost):
    with pytest.raises(ValueError):ResourceParetoBest({_resource_label(0,0,0):cost})


@pytest.mark.parametrize('seed',range(30))
def test_resource_pareto_matches_exhaustive_label_Dijkstra_with_reverse_episode_resets(seed):
    import heapq,itertools,random
    random_source=random.Random(seed)
    graph={node:[] for node in range(5)}
    for node in graph:
        for target in graph:
            if random_source.random()<.45:
                graph[node].append((target,random_source.choice((-1,1)),random_source.randint(1,2),random_source.randint(1,5)))
    def solve(best):
        initial=(0,0,0,0,0);best[initial]=0.;serial=itertools.count();queue=[(0.,next(serial),initial)]
        while queue:
            cost,_,key=heapq.heappop(queue)
            if cost>best.get(key,math.inf):continue
            node,previous,episode,total,changes=key
            if node==4:return cost
            for target,direction,distance,edge_cost in graph[node]:
                e=(episode if previous<0 else 0)+distance if direction<0 else 0
                t=total+(distance if direction<0 else 0)
                c=changes+int(previous!=0 and previous!=direction)
                if e>2 or t>4 or c>3:continue
                new=(target,direction,e,t,c);value=cost+edge_cost
                if value>=best.get(new,math.inf):continue
                best[new]=value;heapq.heappush(queue,(value,next(serial),new))
        return None
    assert solve(ResourceParetoBest())==solve({})


@pytest.fixture
def pareto_original_lattice():
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,
        lattice.State,lattice.propagate,lattice._state_key,MotionResourceLimits(maximum_total_reverse_m=.5))
    return lattice,base,pareto_resource_lattice_function(base)


@pytest.mark.parametrize('goal',[(.2,0.,0.),(-.25,0.,0.)])
def test_pareto_actual_original_straight_lattice_matches_geometry_cost_and_full_resource_state(pareto_original_lattice,goal):
    from dataclasses import asdict
    lattice,base,corrected=pareto_original_lattice
    args=dict(allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),speed_m_s=.05,
        node_limit=1000,dynamics=lattice.KINEMATIC_DYNAMICS,directions=(1,-1),minimum_x=-1.,maximum_x=1.)
    before={};after={};Checker=resource_checker_class(FreeCheckerFixture)
    plain=base(goal,-2.,2.,collision_checker=Checker(),diagnostics=before,**args)
    result=corrected(goal,-2.,2.,collision_checker=Checker(),diagnostics=after,**args)
    assert plain is not None and result is not None and result.cost==plain.cost
    assert result.actions==plain.actions
    assert [asdict(s) for s in result.states]==[asdict(s) for s in plain.states]
    assert after['resource_pareto']['labels_inserted']>0
    assert corrected.mission_search_binding['AST_constructor_substitutions']==1
    assert corrected.mission_search_binding['undo_reproduces_entire_original_AST']
    assert corrected.mission_search_binding['search_code_object_identical'] is False


@pytest.mark.parametrize('full_NE',[False,True])
def test_complete_goal_exists_before_queue_pop_node_cap_and_generation_exit_keeps_exact_state_actions_and_cost(full_NE):
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,
        lattice.State,lattice.propagate,lattice._state_key,MotionResourceLimits())
    normal=pareto_resource_lattice_function(base)
    generated=pareto_resource_lattice_function(base,generated_goal_exit=True)
    options=dict(allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),speed_m_s=.05,
        directions=(1,),collision_checker=resource_checker_class(FreeCheckerFixture)(),
        dynamics=lattice.FULL_DYNAMICS if full_NE else lattice.KINEMATIC_DYNAMICS,
        goal_heading_tolerance_rad=math.radians(2.))
    if full_NE:options.update(rigid_body_model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds())),
        speed_levels_m_s=(.003,),terminal_speed_tolerance_m_s=.005,terminal_deceleration_m_s2=.020,
        propagate_initial_steering=True)
    assert normal((.10,0.,0.),-1.,1.,node_limit=2,**options) is None
    result=generated((.10,0.,0.),-1.,1.,node_limit=2,**options)
    popped=normal((.10,0.,0.),-1.,1.,node_limit=20,**options)
    assert result is not None and popped is not None
    assert result.expanded_nodes==2 and popped.expanded_nodes>result.expanded_nodes
    assert result.actions==popped.actions and result.cost==popped.cost
    assert [asdict(s) for s in result.states]==[asdict(s) for s in popped.states]
    info=generated.mission_search_binding
    assert info['AST_generated_complete_goal_exits']==1 and info['undo_reproduces_entire_original_AST']
    assert info['all_original_propagation_collision_cost_goal_and_primitive_statements_unchanged']
    if full_NE:assert math.hypot(result.states[-1].longitudinal_velocity,result.states[-1].lateral_velocity)<=.005


@pytest.mark.parametrize('fault',['collision','heading','speed','reverse_resource'])
def test_generated_goal_exit_still_rejects_original_geometry_heading_speed_and_reverse_constraints(fault):
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,
        lattice.State,lattice.propagate,lattice._state_key,MotionResourceLimits(maximum_total_reverse_m=.5))
    run=pareto_resource_lattice_function(base,generated_goal_exit=True)
    goal=(.10,0.,math.pi/2 if fault=='heading' else 0.)
    checker=resource_checker_class(FreeCheckerFixture)()
    if fault=='collision':checker=lambda poses:any(p[0]>=.075 for p in poses)
    options=dict(allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),speed_m_s=.05,
        directions=(1,),collision_checker=checker,dynamics=lattice.KINEMATIC_DYNAMICS,
        goal_heading_tolerance_rad=math.radians(2.))
    if fault=='speed':options.update(dynamics=lattice.FULL_DYNAMICS,
        rigid_body_model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds())),
        speed_levels_m_s=(.05,),terminal_speed_tolerance_m_s=.005,terminal_deceleration_m_s2=.020,
        propagate_initial_steering=True)
    if fault=='reverse_resource':goal=(-.55,0.,0.);options['directions']=(-1,)
    assert run(goal,-1.,1.,node_limit=20,**options) is None


def test_generated_goal_exit_cannot_relabel_an_intermediate_region(pareto_original_lattice):
    _,base,_=pareto_original_lattice
    with pytest.raises(ValueError,match='intermediate'):
        pareto_resource_lattice_function(base,generated_goal_exit=True,stage_acceptance=lambda state,whole:True)
    with pytest.raises(ValueError):pareto_resource_lattice_function(base,generated_goal_exit='yes')


@pytest.mark.parametrize('scope',['all_four','P_only','extended_budget'])
def test_new_first_generated_goal_policy_is_explicit_common_all_four180s(mission_offline_probe_driver,scope):
    from types import FunctionType
    f=mission_offline_probe_driver['validate_probe_selection'];old=f.__globals__
    values=dict(old,GUIDANCE_REVISION='local_heading_terminal_blend_v1',
        SEARCH_REVISION='resource_pareto_first_stored_goal_v2',P_PIPELINE_REVISION='owned_reference_joint_NE_v1',
        RUN_METHODS=old['METHODS'],TOTAL_WALL_S=180.,OUTPUT=old['GATE']/'hospital_h6_planning/SOFTWARE_FIRST_GOAL_NOT_EXECUTED')
    if scope=='P_only':values['RUN_METHODS']=(old['METHODS'][-1],)
    elif scope=='extended_budget':values.update(RUN_METHODS=(old['METHODS'][-1],),TOTAL_WALL_S=600.)
    run=FunctionType(f.__code__,values,f.__name__)
    if scope=='all_four':run()
    else:
        with pytest.raises(ValueError):run()


def test_driver_generated_goal_binding_keeps_goal_flag_before_all_three_resource_bins(mission_offline_probe_driver):
    from types import FunctionType
    from patient_initial_stability import steering_lattice as lattice
    f=mission_offline_probe_driver['make_resource_search'];old=f.__globals__
    values=dict(old,SEARCH_REVISION='resource_pareto_first_stored_goal_v2',
        COMPUTE_REVISION='original_exact_computation_v1',P_PIPELINE_REVISION='owned_reference_joint_NE_v1')
    search,_=FunctionType(f.__code__,values,f.__name__)()
    assert search.mission_search_binding['AST_generated_complete_goal_exits']==1
    assert search.mission_search_binding['undo_reproduces_entire_original_AST']
    assert search.mission_complete_goal_key['original_edge_and_goal_constraints_unchanged']


def test_pareto_search_does_not_waive_500mm_total_reverse_bound(pareto_original_lattice):
    lattice,_,corrected=pareto_original_lattice;diagnostics={}
    Checker=resource_checker_class(FreeCheckerFixture);checker=Checker()
    result=corrected((-.55,0.,0.),-2.,2.,allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),
        speed_m_s=.05,node_limit=1000,dynamics=lattice.KINEMATIC_DYNAMICS,directions=(1,-1),
        minimum_x=-1.,maximum_x=1.,collision_checker=checker,diagnostics=diagnostics)
    assert result is None and checker.statistics['motion_resource_reverse_total_rejections']>0
    assert diagnostics['resource_pareto']['strict_resource_dominance_hits']>0


@pytest.mark.parametrize('raise_after',[1,5])
def test_original_timeout_is_rethrown_with_actual_search_frame_not_fabricated_success(pareto_original_lattice,raise_after):
    lattice,base,corrected=pareto_original_lattice
    def attempt(function):
        error=TimeoutError('ORIGINAL_FIXED_DIAGNOSTIC_DEADLINE');calls=[];diagnostics={}
        def heuristic(state):
            calls.append(state)
            if len(calls)>=raise_after:raise error
            return 0.
        with pytest.raises(TimeoutError) as caught:
            function((.5,0.,0.),-2.,2.,allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),
                speed_m_s=.05,node_limit=1000,dynamics=lattice.KINEMATIC_DYNAMICS,directions=(1,-1),
                minimum_x=-1.,maximum_x=1.,collision_checker=resource_checker_class(FreeCheckerFixture)(),
                diagnostics=diagnostics,heuristic_function=heuristic)
        assert caught.value is error
        return diagnostics
    before=attempt(base);after=attempt(corrected)
    assert before=={} and after['original_timeout_exception']=='ORIGINAL_FIXED_DIAGNOSTIC_DEADLINE'
    assert after['timeout_diagnostics_source']=='ORIGINAL_SEARCH_FRAME_NOT_A_REPLAY_OR_PLAN'
    assert after['search_initialisation_complete']==(raise_after>1)
    if raise_after>1:
        assert after['expanded_nodes']>=1 and 'best_goal_position_error_m' in after
        assert after['resource_pareto']['labels_inserted']>0
        assert after['observed_best_goal_state']['reverse_total_bins']>=0


def test_pareto_adapter_never_mutates_original_module_code_or_previous_private_binding(pareto_original_lattice):
    from patient_initial_stability import steering_lattice as lattice
    _,base,corrected=pareto_original_lattice
    before=dict(lattice.plan.__globals__);private=dict(base.__globals__)
    assert lattice.plan.__code__ is base.__code__
    assert corrected.__code__ is not base.__code__
    assert lattice.plan.__globals__==before and base.__globals__==private


def _unsupported_zero_best_fixture():
    return None


def _unsupported_two_best_fixture():
    best={}
    best={}
    return best


@pytest.mark.parametrize('function',[_unsupported_zero_best_fixture,_unsupported_two_best_fixture])
def test_pareto_transform_rejects_unreviewed_search_structure(function):
    with pytest.raises(ValueError):pareto_resource_lattice_function(function)


def test_pareto_concurrent_calls_own_separate_stores_and_diagnostics(pareto_original_lattice):
    from concurrent.futures import ThreadPoolExecutor
    lattice,_,corrected=pareto_original_lattice
    def evaluate(goal):
        diagnostics={};result=corrected(goal,-2.,2.,allowed_modes=('coordinated_counterphase',),
            allowed_curvatures=(0.,),speed_m_s=.05,node_limit=1000,dynamics=lattice.KINEMATIC_DYNAMICS,
            directions=(1,-1),minimum_x=-1.,maximum_x=1.,
            collision_checker=resource_checker_class(FreeCheckerFixture)(),diagnostics=diagnostics)
        return result,diagnostics
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(evaluate,[(.2,0.,0.),(-.25,0.,0.)]))
    assert all(result is not None for result,_ in results)
    assert results[0][1]['resource_pareto'] is not results[1][1]['resource_pareto']
    assert results[0][0].states[-1].mission_reverse_total_bins==0
    assert results[1][0].states[-1].mission_reverse_total_bins>0


@pytest.mark.parametrize('selection',['unknown_search','wrong_guidance','old_second_folder'])
def test_pareto_probe_requires_reviewed_revision_and_a_separate_new_evidence_folder(mission_offline_probe_driver,selection):
    from types import FunctionType
    original=mission_offline_probe_driver['validate_probe_selection']
    namespace=dict(original.__globals__,GUIDANCE_REVISION='local_heading_terminal_blend_v1',
        SEARCH_REVISION='resource_pareto_best_constructor_only_v1',
        OUTPUT=original.__globals__['GATE']/'hospital_h6_planning/software_fixture_NOT_executed')
    if selection=='unknown_search':namespace['SEARCH_REVISION']='unknown'
    elif selection=='wrong_guidance':namespace['GUIDANCE_REVISION']='legacy_bidirectional_virtual_heading_v1'
    else:namespace['OUTPUT']=original.__globals__['GATE']/'hospital_h6_planning/heading0_local_heading_matched_dev_r02'
    run=FunctionType(original.__code__,namespace,original.__name__)
    with pytest.raises(ValueError):run()


def test_pareto_probe_wraps_only_common_lattice_not_fixed_candidate_NE_refinement(mission_offline_probe_driver):
    from types import FunctionType,SimpleNamespace
    original=mission_offline_probe_driver['make_resource_search'];calls=[]
    def kernels(*args):calls.append(args);return 'ORIGINAL_RESOURCE_LATTICE','SAME_NE_REFINEMENT'
    def wrapper(search):assert search=='ORIGINAL_RESOURCE_LATTICE';return 'PARETO_STORAGE_ONLY'
    namespace=dict(original.__globals__,SEARCH_REVISION='resource_pareto_best_constructor_only_v1',
        binding=SimpleNamespace(bind_resource_kernels=kernels,pareto_resource_lattice_function=wrapper))
    run=FunctionType(original.__code__,namespace,original.__name__)
    assert run()==('PARETO_STORAGE_ONLY','SAME_NE_REFINEMENT') and len(calls)==1
    assert calls[0][0] is original.__globals__['lattice'].plan
    assert original.__globals__['SEARCH_REVISION']=='exact_resource_history_keys_v1'


class _PortfolioClock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value


def _owned_candidate(found=True):
    return dict(plan_success=found,status='SOFTWARE_CANDIDATE_FIXTURE',selected_actions=[{'fixture':True}] if found else [])


def _NE_profile(found=True,cost=10.):
    return dict(plan_success=found,status='SOFTWARE_NE_FIXTURE',search_cost_s=cost,
        selected_actions=[{'fixture':True}] if found else [],selected_states=[{'fixture':True}] if found else [])


def test_owned_portfolio_gives_long_primary_own_search_then_immediate_NE_without_free_baseline_path():
    clock=_PortfolioClock();calls=[];owned={}
    def generate(source,seconds,nodes):
        calls.append(('generate',source,seconds,nodes))
        found=source=='transition' and seconds>=84.
        clock.value+=84. if found else seconds
        candidate=_owned_candidate(found);owned[source]=candidate
        return candidate,min(nodes,17680)
    def evaluate(source,candidate,seconds):
        assert candidate is owned[source]
        calls.append(('NE',source,seconds));clock.value+=20.;return _NE_profile()
    result=owned_candidate_portfolio(generate,evaluate,sources=('transition','fixed','kinematic'),clock=clock,
        budget_s=180.,generation_cap_s=120.,refinement_reserve_s=60.,node_limit=60000)
    assert calls[0][0:2]==('generate','transition') and calls[0][2]==120.
    assert calls[1][0:2]==('NE','transition') and calls[1][2]==96.
    assert result['selected_source']=='transition' and result['selected_profile']['plan_success']
    assert result['end_to_end_budget_respected'] and not result['completed_external_baseline_paths_used']
    assert result['not_attempted_candidate_families']==['kinematic']


def test_owned_portfolio_retains_all_families_and_selects_best_evaluated_NE_cost_not_first_success():
    clock=_PortfolioClock();objects={};events=[];costs={'transition':10.,'fixed':12.,'kinematic':5.}
    def generate(source,seconds,nodes):
        clock.value+=3.;objects[source]=_owned_candidate();return objects[source],2
    def evaluate(source,candidate,seconds):
        assert candidate is objects[source];clock.value+=2.;return _NE_profile(cost=costs[source])
    result=owned_candidate_portfolio(generate,evaluate,sources=tuple(costs),clock=clock,budget_s=30.,
        generation_cap_s=20.,refinement_reserve_s=5.,node_limit=10,
        observe=lambda kind,source,row,event:events.append((kind,source)))
    assert result['selected_source']=='kinematic' and result['selected_profile']['search_cost_s']==5.
    assert result['all_candidate_families_attempted'] and result['all_generated_candidates_NE_evaluated']
    assert result['expanded_nodes_total']==6 and result['elapsed_s']==15.
    assert events==[(kind,source) for source in costs for kind in ('generation','refinement')]


def test_owned_portfolio_charges_aggregate_nodes_not_a_fresh_cap_for_every_source():
    clock=_PortfolioClock();allocations=[]
    def generate(source,seconds,nodes):
        allocations.append(nodes);clock.value+=1.;return _owned_candidate(False),min(nodes,6)
    result=owned_candidate_portfolio(generate,lambda *a:pytest.fail('No candidate means no NE'),
        sources=('a','b','c'),clock=clock,budget_s=30.,generation_cap_s=20.,refinement_reserve_s=5.,node_limit=10)
    assert allocations==[10,4] and result['expanded_nodes_total']==10
    assert result['not_attempted_candidate_families']==['c'] and result['selected_profile'] is None


def test_owned_portfolio_observer_writes_consume_whole_pipeline_time():
    clock=_PortfolioClock()
    def generate(*args):clock.value+=3.;return _owned_candidate(),1
    def evaluate(*args):clock.value+=2.;return _NE_profile()
    def observe(*args):clock.value+=1.
    result=owned_candidate_portfolio(generate,evaluate,sources=('a',),clock=clock,budget_s=20.,
        generation_cap_s=10.,refinement_reserve_s=5.,node_limit=10,observe=observe)
    assert result['elapsed_s']==7. and result['pipeline_candidate_generation_time_s']==3.
    assert result['pipeline_NE_evaluation_time_s']==2.


def test_owned_portfolio_preserves_actual_overrun_flag_instead_of_promoting_late_profile():
    clock=_PortfolioClock()
    def generate(*args):clock.value+=1.;return _owned_candidate(),1
    def evaluate(*args):clock.value+=100.;return _NE_profile()
    result=owned_candidate_portfolio(generate,evaluate,sources=('a',),clock=clock,budget_s=20.,
        generation_cap_s=10.,refinement_reserve_s=5.,node_limit=10)
    assert result['selected_profile']['plan_success'] and not result['end_to_end_budget_respected']
    assert result['elapsed_s']==101.


def test_owned_portfolio_original_callback_error_is_not_swallowed_or_retried():
    clock=_PortfolioClock();error=RuntimeError('ORIGINAL_GENERATOR_ERROR');calls=[]
    def generate(*args):calls.append(args);raise error
    with pytest.raises(RuntimeError) as caught:
        owned_candidate_portfolio(generate,lambda *a:None,sources=('a','b'),clock=clock,budget_s=20.,
            generation_cap_s=10.,refinement_reserve_s=5.,node_limit=10)
    assert caught.value is error and len(calls)==1


@pytest.mark.parametrize('kind',['bad_nodes','missing_actions','missing_NE_states','nonfinite_NE_cost','clock_reverse'])
def test_owned_portfolio_rejects_unverified_or_malformed_callback_results(kind):
    clock=_PortfolioClock()
    def generate(source,seconds,nodes):
        if kind=='clock_reverse':clock.value=-1.
        else:clock.value+=1.
        candidate=_owned_candidate()
        if kind=='missing_actions':candidate['selected_actions']=[]
        return candidate,nodes+1 if kind=='bad_nodes' else 1
    def evaluate(*args):
        profile=_NE_profile()
        if kind=='missing_NE_states':profile['selected_states']=[]
        if kind=='nonfinite_NE_cost':profile['search_cost_s']=math.nan
        return profile
    with pytest.raises(ValueError):owned_candidate_portfolio(generate,evaluate,sources=('a',),clock=clock,
        budget_s=20.,generation_cap_s=10.,refinement_reserve_s=5.,node_limit=10)


@pytest.mark.parametrize('settings',[dict(sources=()),dict(sources=('a','a')),dict(budget_s=0.),
    dict(generation_cap_s=math.inf),dict(refinement_reserve_s=30.),dict(node_limit=True)])
def test_owned_portfolio_requires_explicit_valid_budgets_and_families(settings):
    params=dict(sources=('a','b'),clock=_PortfolioClock(),budget_s=20.,generation_cap_s=10.,refinement_reserve_s=5.,node_limit=10)
    params.update(settings)
    with pytest.raises(ValueError):owned_candidate_portfolio(lambda *a:None,lambda *a:None,**params)


@pytest.mark.parametrize('case',['unknown_policy','P_only_legacy','reuse_four_method_result','wrong_search'])
def test_owned_P_driver_selection_cannot_omit_or_overwrite_original_comparisons(mission_offline_probe_driver,case):
    from types import FunctionType
    original=mission_offline_probe_driver['validate_probe_selection'];old=original.__globals__
    namespace=dict(old,GUIDANCE_REVISION='local_heading_terminal_blend_v1',
        SEARCH_REVISION='resource_pareto_best_constructor_only_v1',
        P_PIPELINE_REVISION='owned_transition_priority_reserve_v1',RUN_METHODS=(old['METHODS'][-1],),
        OUTPUT=old['GATE']/'hospital_h6_planning/software_fixture_owned_NOT_executed')
    if case=='unknown_policy':namespace['P_PIPELINE_REVISION']='unknown'
    elif case=='P_only_legacy':namespace['P_PIPELINE_REVISION']='equal_three_stage_v1'
    elif case=='reuse_four_method_result':namespace['OUTPUT']=old['BASELINE_REFERENCE'].parent
    else:namespace['SEARCH_REVISION']='exact_resource_history_keys_v1'
    run=FunctionType(original.__code__,namespace,original.__name__)
    with pytest.raises(ValueError):run()


@pytest.mark.parametrize('case',['duplicate','missing','unknown','valid_B0_first'])
def test_owned_P_priority_is_an_explicit_permutation_not_a_hidden_family_removal(mission_offline_probe_driver,case):
    from types import FunctionType
    original=mission_offline_probe_driver['validate_probe_selection'];old=original.__globals__
    priority=(old['METHODS'][0],old['METHODS'][2],old['METHODS'][1])
    if case=='duplicate':priority=(priority[0],priority[0],priority[2])
    elif case=='missing':priority=priority[:-1]
    elif case=='unknown':priority=(priority[0],priority[1],'invented')
    namespace=dict(old,GUIDANCE_REVISION='local_heading_terminal_blend_v1',
        SEARCH_REVISION='resource_pareto_best_constructor_only_v1',
        P_PIPELINE_REVISION='owned_transition_priority_reserve_v1',RUN_METHODS=(old['METHODS'][-1],),
        P_SOURCE_PRIORITY=priority,OUTPUT=old['GATE']/'hospital_h6_planning/software_fixture_B0_priority_NOT_executed')
    run=FunctionType(original.__code__,namespace,original.__name__)
    if case=='valid_B0_first':run()
    else:
        with pytest.raises(ValueError):run()


def test_owned_P_description_cannot_claim_a_hardcoded_source_order(mission_offline_probe_driver):
    # The actual selector and recorded list are authoritative. A human label
    # must not contradict them when a different explicit priority is reviewed.
    strings=[s for s in mission_offline_probe_driver['main'].__code__.co_consts if isinstance(s,str)]
    assert any('see P_source_priority' in s for s in strings)
    assert not any('Own priority B2/B0/B1' in s for s in strings)


@pytest.mark.parametrize('NE_success',[False,True])
def test_owned_P_driver_uses_new_own_actions_and_preserves_original_NE_outcome(mission_offline_probe_driver,NE_success):
    from types import FunctionType,SimpleNamespace
    original=mission_offline_probe_driver['run_owned_portfolio_pipeline'];clock=_PortfolioClock();calls=[];owned={};observed=[]
    def run(method,manifest,occupied,origin,resolution,model,seconds,nodes,pool):
        calls.append((method,seconds,nodes,pool))
        if method==original.__globals__['METHODS'][-1]:
            source=original.__globals__['P_SOURCE_PRIORITY'][0]
            assert pool=={source:owned['selected_actions']}
            clock.value+=10.;result=_NE_profile(NE_success)
            result.update(method=method,refinement_attempts=[{'original_marker':True}])
            return result
        if not owned:
            clock.value+=84.;candidate=_owned_candidate()
            candidate.update(method=method,candidate_generation_diagnostics={'expanded_nodes':17680})
            owned.update(candidate);return candidate
        clock.value+=seconds;candidate=_owned_candidate(False)
        candidate.update(method=method,candidate_generation_diagnostics={'expanded_nodes':1000});return candidate
    namespace=dict(original.__globals__,time=SimpleNamespace(perf_counter=clock))
    function=FunctionType(original.__code__,namespace,original.__name__)
    result,summary=function(run,{},None,None,None,None,178.,lambda *args:observed.append(args))
    assert [v[0] for v in calls][:2]==[original.__globals__['METHODS'][2],original.__globals__['METHODS'][-1]]
    assert calls[0][3]=={} and len(summary['refinement_stages'])==1
    assert summary['expanded_nodes_total']==18680 and not summary['all_candidate_families_attempted']
    assert result['P_full_NE_refinement_executed'] and result['plan_success']==NE_success
    assert result['refinement_attempts']==[{'original_marker':True}]
    if not NE_success:
        assert result['original_last_NE_status']=='SOFTWARE_NE_FIXTURE'
        assert result['status']=='NO_ACCEPTED_NE_PROFILE_IN_EVALUATED_OWN_CANDIDATES_NOT_GLOBAL_INFEASIBILITY'


class _RestorePoses(list):
    resource_feasible=True


def _restore_fake_propagate(state,mode,curvature,direction,**kwargs):
    # Pure unit geometry. These integer-X edges are NOT robot metres or NE
    # validation; all production rollouts use the unchanged original model.
    from dataclasses import replace
    new=replace(state,x=state.x+direction,y=state.y+curvature,front=state.front+curvature,
        rear=state.rear-curvature,started=True,direction=direction)
    return new,_RestorePoses([(new.x,new.y,new.yaw)]),1.


class _RestoreChecker:
    def __init__(self):self.checks=0;self.timeout=None;self.force_invalid=False
    def check_deadline(self):
        self.checks+=1
        if self.timeout is not None and self.checks>=self.timeout[0]:raise self.timeout[1]
    def __call__(self,poses):
        self.check_deadline()
        return any((abs(x-3.)<.01 and y<.20) or (abs(x-4.)<.01 and abs(y)>1e-12)
            or self.force_invalid for x,y,yaw in poses)


def _restore_arguments():
    from patient_initial_stability import steering_lattice as lattice
    return dict(actions=[dict(mode='coordinated_counterphase',curvature_1_m=0.,direction=1)]*5,
        speeds=[.05]*5,initial_state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True),
        propagate=_restore_fake_propagate,checker=_RestoreChecker(),rigid_body_model=None,
        dynamics=lattice.FULL_DYNAMICS,curvatures=lattice.CURVATURES,step_m=.05,
        maximum_speed_m_s=.05,terminal_speed_m_s=.005,deceleration_m_s2=.02)


def test_NE_prefix_restoration_can_cross_two_infeasible_search_steps_but_returns_only_full_safe_seed():
    args=_restore_arguments();old=[dict(a) for a in args['actions']]
    repaired,log=restore_NE_feasible_prefix(**args)
    assert repaired is not None and log['complete_seed_found'] and len(log['changes'])==2
    assert log['initial_valid_edges']==2 and log['last_valid_edges']==5
    assert args['actions']==old and args['initial_state'].x==0.
    state=args['initial_state']
    for action in repaired:
        state,poses,_=_restore_fake_propagate(state,action['mode'],action['curvature_1_m'],action['direction'])
        assert not args['checker'](poses)
    assert [state.x,state.y,state.yaw]==log['final_seed_pose_local_m_rad']==[5.,0.,0.]
    assert state.front==state.rear==0.
    assert all(a['mode']==b['mode'] and a['direction']==b['direction'] for a,b in zip(old,repaired))
    assert log['final_goal_NOT_certified_here']


@pytest.mark.parametrize('guard',['rigid','resource','stopping'])
def test_NE_prefix_restoration_never_waives_a_non_geometry_edge_constraint(guard):
    from dataclasses import replace
    args=_restore_arguments()
    def guarded(*a,**kw):
        state,poses,t=_restore_fake_propagate(*a,**kw)
        if guard=='rigid':state=replace(state,rigid_body_feasible=False)
        elif guard=='resource':poses.resource_feasible=False
        else:state=replace(state,longitudinal_velocity=1.)
        return state,poses,t
    args['propagate']=guarded
    result,log=restore_NE_feasible_prefix(**args)
    assert result is None and not log['complete_seed_found'] and log['last_valid_edges']==0
    assert log['last_rejection']==dict(rigid='rigid_body',resource='motion_resource',stopping='stopping_reachability')[guard]


def test_NE_prefix_restoration_rethrows_original_timeout_and_retains_progress_only():
    args=_restore_arguments();error=TimeoutError('ORIGINAL_RESTORATION_DEADLINE');args['checker'].timeout=(12,error);progress={}
    with pytest.raises(TimeoutError) as caught:restore_NE_feasible_prefix(**args,progress=progress)
    assert caught.value is error and progress['profile_evaluations']>0
    assert not progress['complete_seed_found']


def test_NE_prefix_restoration_does_not_approve_blocked_initial_pose():
    args=_restore_arguments();args['checker'].force_invalid=True
    result,log=restore_NE_feasible_prefix(**args)
    assert result is None and log['initial_state_blocked'] and log['profile_evaluations']==0


def test_original_strict_refiner_still_rechecks_repaired_seed_and_final_goal():
    from types import FunctionType
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route as original
    namespace=dict(original.__globals__,propagate=_restore_fake_propagate)
    refiner=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__)
    refiner.__kwdefaults__=original.__kwdefaults__
    args=_restore_arguments();events=[]
    before=dict(original.__globals__)
    plain,diagnostics=refiner('unit',args['actions'],(5.,0.,0.),(.003,.05),None,args['checker'],.005,.02,
        terminal_extension_edges=0,propagate_initial_steering=True)
    assert plain is None and diagnostics['collision']>0
    result,audit=refine_with_NE_prefix_restoration(refiner,'unit',args['actions'],(5.,0.,0.),(.003,.05),None,
        _RestoreChecker(),.005,.02,terminal_extension_edges=0,propagate_initial_steering=True,event_sink=events.append)
    assert result is not None and result.states[-1].x==5. and result.states[-1].y==0.
    assert events[0]['restoration']['complete_seed_found'] and events[0]['final_constraints_unchanged']
    assert audit['terminal_position_error_m']==0. and original.__globals__==before


def test_NE_restoration_can_change_adjacent_controls_when_no_single_control_improves_prefix():
    from dataclasses import replace
    args=_restore_arguments()
    def pair_required(state,mode,curvature,direction,**kw):
        y=1. if state.front>.2 and curvature>.2 else state.y
        new=replace(state,x=state.x+1.,y=y,front=curvature,started=True,direction=direction)
        return new,_RestorePoses([(new.x,new.y,new.yaw)]),1.
    class Checker(_RestoreChecker):
        def __call__(self,poses):
            self.check_deadline();return any(abs(x-3.)<.01 and y<.5 for x,y,yaw in poses)
    args.update(propagate=pair_required,checker=Checker())
    repaired,log=restore_NE_feasible_prefix(**args)
    assert repaired is not None and log['complete_seed_found'] and len(log['changes'])==1
    assert len(log['changes'][0]['edits'])==2 and log['final_goal_NOT_certified_here']


@pytest.mark.parametrize('quantum',[0.,1.,math.nan])
def test_NE_prefix_search_ranking_quantum_cannot_be_a_guard_waiver(quantum):
    with pytest.raises(ValueError):restore_NE_feasible_prefix(**_restore_arguments(),progress_fraction_quantum=quantum)


def test_NE_restoration_original_modes_are_eligible_without_direction_or_state_reset():
    from dataclasses import replace
    args=_restore_arguments()
    def needs_modes(state,mode,curvature,direction,**kw):
        dy=.25 if mode=='front_only' else -.25 if mode=='rear_only' else 0.
        new=replace(state,x=state.x+1.,y=state.y+dy,mode=mode,started=True,direction=direction,
            front=state.front+dy,rear=state.rear-dy)
        return new,_RestorePoses([(new.x,new.y,new.yaw)]),1.
    args['propagate']=needs_modes
    fixed,old=restore_NE_feasible_prefix(**args)
    assert fixed is None and not old['complete_seed_found']
    repaired,log=restore_NE_feasible_prefix(**args,allow_mode_changes=True)
    assert repaired is not None and log['complete_seed_found'] and log['direction_sequence_unchanged']
    assert not log['mode_direction_sequence_unchanged'] and log['source_mode_entries_changed']==2
    assert {a['mode'] for a in repaired}<={'coordinated_counterphase','front_only','rear_only'}
    state=args['initial_state']
    for a in repaired:
        state,poses,_=needs_modes(state,a['mode'],a['curvature_1_m'],a['direction'])
        assert not args['checker'](poses)
    assert state.front==state.rear==0. and [state.x,state.y,state.yaw]==[5.,0.,0.]


@pytest.mark.parametrize('values',[dict(allow_mode_changes='yes'),dict(allowed_modes=('UNKNOWN',)),dict(allowed_modes=())])
def test_NE_restoration_cannot_invent_or_silently_select_an_unreviewed_mode(values):
    with pytest.raises(ValueError):restore_NE_feasible_prefix(**_restore_arguments(),**values)


def _beam_arguments():
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    args=_restore_arguments()
    args.pop('lookback_edges',None)
    args.update(references=[replace(args['initial_state'],x=float(i+1)) for i in range(5)],
        allowed_modes=lattice.ALL_MODES,goal=(5.,0.,0.),body_radius_m=1.,
        state_key=lambda s:(round(s.x,2),round(s.y,2),round(s.front,2),round(s.rear,2)),beam_width=12)
    return args


def test_NE_guided_beam_keeps_real_history_and_can_avoid_a_later_dead_end():
    args=_beam_arguments();old=[dict(a) for a in args['actions']]
    result,log=recover_NE_guided_beam(**args)
    assert result is not None and log['completed_layers']==5 and log['complete_seed_found']
    assert log['final_seed_position_error_m']==0. and log['final_goal_NOT_certified_here']
    state=args['initial_state']
    for action in result:
        previous=state
        state,poses,_=args['propagate'](state,action['mode'],action['curvature_1_m'],action['direction'])
        assert not args['checker'](poses)
        assert state.front==previous.front+action['curvature_1_m']
        assert state.rear==previous.rear-action['curvature_1_m']
    assert args['actions']==old and state.x==5. and state.y==0.
    assert log['approximate_beam_pruning_NOT_exact_dominance']


@pytest.mark.parametrize('guard',['rigid','resource','stopping'])
def test_NE_guided_beam_rejects_every_invalid_edge_before_caching(guard):
    from dataclasses import replace
    args=_beam_arguments()
    def guarded(*a,**kw):
        state,poses,t=_restore_fake_propagate(*a,**kw)
        if guard=='rigid':state=replace(state,rigid_body_feasible=False)
        elif guard=='resource':poses.resource_feasible=False
        else:state=replace(state,longitudinal_velocity=1.)
        return state,poses,t
    args['propagate']=guarded
    result,log=recover_NE_guided_beam(**args)
    assert result is None and log['completed_layers']==0 and not log['complete_seed_found']
    reason=dict(rigid='rigid_body',resource='motion_resource',stopping='stopping_reachability')[guard]
    assert log['rejected'][reason]==21


def test_NE_guided_beam_ranking_reference_is_not_a_state_or_goal_reset():
    from dataclasses import replace
    args=_beam_arguments();args['references']=[replace(s,x=100.+i,y=100.) for i,s in enumerate(args['references'])]
    class Clear(_RestoreChecker):
        def __call__(self,poses):self.check_deadline();return False
    args['checker']=Clear();result,log=recover_NE_guided_beam(**args)
    assert result is not None and log['complete_seed_found']
    assert log['final_seed_pose_local_m_rad'][0]==5. and log['final_goal_NOT_certified_here']
    assert all(a['direction']==1 for a in result)


def test_NE_guided_beam_rethrows_same_timeout_and_preserves_only_completed_layers():
    args=_beam_arguments();error=TimeoutError('ORIGINAL_BEAM_DEADLINE');args['checker'].timeout=(70,error);progress={}
    with pytest.raises(TimeoutError) as caught:recover_NE_guided_beam(**args,progress=progress)
    assert caught.value is error and progress['NE_edges_evaluated']>0
    assert not progress['complete_seed_found'] and progress['completed_layers']<5


@pytest.mark.parametrize('extra',[dict(beam_width=0),dict(beam_width=True),dict(body_radius_m=math.nan),
    dict(allowed_modes=('INVENTED_MODE',)),dict(speeds=[.1]*5),dict(lookahead_edges=-1),dict(lookahead_edges=True)])
def test_NE_guided_beam_does_not_silently_expand_primitives_or_limits(extra):
    args=_beam_arguments();args.update(extra)
    with pytest.raises(ValueError):recover_NE_guided_beam(**args)


def test_NE_guided_wrapper_original_strict_refiner_rechecks_all_states_and_final_goal():
    from types import FunctionType
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route as original
    class Clear(_RestoreChecker):
        def __call__(self,poses):self.check_deadline();return False
    namespace=dict(original.__globals__,propagate=_restore_fake_propagate)
    refiner=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__)
    args=_beam_arguments();events=[];before=dict(original.__globals__)
    result,audit=refine_with_NE_guided_beam(refiner,'unit',args['actions'],(5.,0.,0.),(.003,.05),None,
        Clear(),.005,.02,terminal_extension_edges=0,propagate_initial_steering=True,
        source_dynamics=lattice.KINEMATIC_DYNAMICS,body_radius_m=1.,event_sink=events.append)
    assert result is not None and audit['terminal_position_error_m']==0.
    assert events[0]['beam']['complete_seed_found'] and events[0]['final_constraints_unchanged']
    assert original.__globals__==before
    # A complete valid seed cannot make a WRONG endpoint into a final PASS.
    bad,_=refine_with_NE_guided_beam(refiner,'unit',args['actions'],(100.,0.,0.),(.003,.05),None,
        Clear(),.005,.02,terminal_extension_edges=0,propagate_initial_steering=True,
        source_dynamics=lattice.KINEMATIC_DYNAMICS,body_radius_m=1.)
    assert bad is None


def test_NE_beam_anticipatory_rank_preserves_delayed_response_without_reusing_forecast_states():
    from dataclasses import replace
    args=_beam_arguments();args['beam_width']=1
    def delayed(state,mode,curvature,direction,**kw):
        # Integer unit-X, artificial one-edge actuator delay; NOT a plant fit.
        new=replace(state,x=state.x+direction,y=state.y+state.front,front=curvature,
            mode=mode,started=True,direction=direction)
        return new,_RestorePoses([(new.x,new.y,new.yaw)]),1.
    class Wall(_RestoreChecker):
        def __call__(self,poses):
            self.check_deadline();return any(abs(x-3.)<.01 and y<.2 for x,y,yaw in poses)
    args.update(propagate=delayed,checker=Wall())
    no_preview,first=recover_NE_guided_beam(**args)
    assert no_preview is None and first['frontier_exhausted_at_edge']==2
    args['checker']=Wall();result,log=recover_NE_guided_beam(**args,lookahead_edges=2)
    assert result is not None and result[0]['curvature_1_m']>0.
    state=args['initial_state']
    for a in result:
        state,poses,_=delayed(state,a['mode'],a['curvature_1_m'],a['direction'])
        assert not args['checker'](poses)
    assert [state.x,state.y,state.yaw]==log['final_seed_pose_local_m_rad']
    assert log['lookahead_NE_edges_evaluated']>0 and log['predicted_states_never_inserted_in_beam']
    assert log['final_goal_NOT_certified_here']


def _joint_NE_fixture_arguments():
    bounds=mission_search_bounds([-1.,1.,-1.,1.],[0.,0.,0.],[.05,0.,0.])
    return dict(bounds=bounds,goal=(.05,0.,0.),checker=FreeCheckerFixture(),model=object(),
        guidance=lambda state,goal:0.,directions=(1,-1),speed_levels_m_s=(.003,.01,.03,.05),
        maximum_speed_m_s=.05,terminal_speed_m_s=.005,deceleration_m_s2=.02,node_limit=60000,diagnostics={})


def test_joint_NE_binding_keeps_original_model_modes_speeds_and_all_state_constraints():
    from patient_initial_stability import steering_lattice as lattice
    args=_joint_NE_fixture_arguments();calls=[];marker=object()
    def original(*a,**kw):calls.append((a,kw));return marker
    before=dict(lattice.plan.__globals__)
    assert plan_NE_mission_candidate(original,**args) is marker
    positional,kw=calls[0]
    assert tuple(positional[0])==args['goal'] and positional[1:]==(-1.,1.)
    assert kw['allowed_modes']==lattice.ALL_MODES and kw['allowed_curvatures']==lattice.CURVATURES
    assert kw['dynamics'] is lattice.FULL_DYNAMICS and kw['rigid_body_model'] is args['model']
    assert kw['speed_levels_m_s']==args['speed_levels_m_s'] and kw['directions']==(1,-1)
    assert kw['propagate_initial_steering'] and kw['initial_front_rad']==kw['initial_rear_rad']==0.
    assert kw['terminal_speed_tolerance_m_s']==.005 and kw['terminal_deceleration_m_s2']==.02
    assert kw['heuristic_weight']==1.5 and kw['node_limit']==60000 and kw['robust_clearance']==.01
    assert lattice.plan.__globals__==before


@pytest.mark.parametrize('change',[dict(model=None),dict(node_limit=True),dict(terminal_speed_m_s=math.nan),
    dict(speed_levels_m_s=(.003,.01,.03,.05,.1)),dict(speed_levels_m_s=(.05,.05)),dict(goal=(.06,0.,0.))])
def test_joint_NE_binding_rejects_unreviewed_limits_or_a_changed_fixed_goal(change):
    args=_joint_NE_fixture_arguments();args.update(change)
    with pytest.raises(ValueError):plan_NE_mission_candidate(lambda *a,**k:pytest.fail('Must fail before search'),**args)


@pytest.mark.parametrize('direction',[1,-1])
def test_joint_NE_binding_uses_real_original_NE_from_standstill_and_stops_at_goal(direction):
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,
        lattice.State,lattice.propagate,lattice._state_key,MotionResourceLimits())
    search=pareto_resource_lattice_function(base);args=_joint_NE_fixture_arguments()
    args.update(model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds())),
        directions=(direction,),goal=(direction*.05,0.,0.),
        bounds=mission_search_bounds([-1.,1.,-1.,1.],[0.,0.,0.],[direction*.05,0.,0.]),node_limit=200)
    result=plan_NE_mission_candidate(search,**args)
    assert result is not None and len(result.actions)==1
    final=result.states[-1]
    assert final.rigid_body_feasible and math.hypot(final.x-args['goal'][0],final.y)<=.025
    assert math.hypot(final.longitudinal_velocity,final.lateral_velocity)<=.005
    assert final.started and final.direction==direction and result.actions[0][3] in args['speed_levels_m_s']
    state=base.__globals__['State'](0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    a=result.actions[0]
    replay,poses,_=base.__globals__['propagate'](state,*a[:3],speed_m_s=.05,
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=args['model'],target_speed_m_s=a[3])
    assert replay==final and poses.resource_feasible


@pytest.mark.parametrize('case',['valid','wrong_search','overwrite_baseline','legacy_guidance'])
def test_joint_NE_driver_requires_an_explicit_isolated_revision(mission_offline_probe_driver,case):
    from types import FunctionType
    original=mission_offline_probe_driver['validate_probe_selection'];old=original.__globals__
    namespace=dict(old,GUIDANCE_REVISION='local_heading_terminal_blend_v1',
        SEARCH_REVISION='resource_pareto_best_constructor_only_v1',P_PIPELINE_REVISION='joint_NE_lattice_v1',
        RUN_METHODS=(old['METHODS'][-1],),OUTPUT=old['GATE']/'hospital_h6_planning/software_fixture_joint_NE_NOT_executed')
    if case=='wrong_search':namespace['SEARCH_REVISION']='exact_resource_history_keys_v1'
    elif case=='overwrite_baseline':namespace['OUTPUT']=old['BASELINE_REFERENCE'].parent
    elif case=='legacy_guidance':namespace['GUIDANCE_REVISION']='legacy_bidirectional_virtual_heading_v1'
    run=FunctionType(original.__code__,namespace,original.__name__)
    if case=='valid':run()
    else:
        with pytest.raises(ValueError):run()


def test_joint_NE_driver_reports_actual_timeout_without_a_fabricated_plan(mission_offline_probe_driver):
    from types import FunctionType,SimpleNamespace
    original=mission_offline_probe_driver['run_joint_NE_pipeline'];calls=[]
    class Checker(FreeCheckerFixture):pass
    def timeout(*args,**kwargs):
        calls.append((args,kwargs));kwargs['diagnostics'].update(generated=17,expanded_nodes=3)
        raise TimeoutError('ORIGINAL_JOINT_DEADLINE')
    function=FunctionType(original.__code__,dict(original.__globals__,
        binding=SimpleNamespace(plan_NE_mission_candidate=timeout)),original.__name__)
    function.__kwdefaults__=dict(original.__kwdefaults__)
    manifest=dict(start_xyyaw=[0.,0.,0.],local_goal_xyyaw=[10.,-4.,0.])
    row=function('SEARCH','BOUNDS','GUIDANCE',Checker,manifest,None,None,None,'MODEL',123.)
    assert not row['plan_success'] and not row['selected_actions'] and not row['selected_states']
    assert row['P_NE_evaluated_during_search'] and not row['P_full_NE_refinement_executed']
    assert row['candidate_generation_diagnostics']==dict(generated=17,expanded_nodes=3)
    assert row['error']=='TimeoutError: ORIGINAL_JOINT_DEADLINE' and len(calls)==1
    assert calls[0][1]['directions']==(1,-1) and calls[0][1]['node_limit']==60000


def test_joint_NE_success_serialization_and_independent_original_stopping_resource_audit(mission_offline_probe_driver):
    from types import FunctionType,SimpleNamespace
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    initial=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    final,_,elapsed=lattice.propagate(initial,'coordinated_counterphase',0.,1,
        speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=.003)
    result=lattice.PlanResult((final,),(('coordinated_counterphase',0.,1,.003),),2,elapsed,())
    def selected(*a,**kw):kw['diagnostics'].update(generated=84,expanded_nodes=2);return result
    original=mission_offline_probe_driver['run_joint_NE_pipeline']
    function=FunctionType(original.__code__,dict(original.__globals__,binding=SimpleNamespace(plan_NE_mission_candidate=selected)),original.__name__)
    function.__kwdefaults__=dict(original.__kwdefaults__)
    class Checker(FreeCheckerFixture):pass
    manifest=dict(start_xyyaw=[0.,0.,0.],local_goal_xyyaw=[.05,0.,0.])
    row=function('SEARCH','BOUNDS','GUIDANCE',Checker,manifest,None,None,None,model,123.)
    assert row['plan_success'] and row['predicted_motion_time_s']==elapsed
    assert row['selected_actions'][0]['target_speed_m_s']==.003
    assert row['selected_states'][0]['u_m_s']==final.longitudinal_velocity
    row.update(goal_local_xyyaw=manifest['local_goal_xyyaw'],rigid_body_parameter_bounds=asdict(model.bounds))
    audit=mission_offline_probe_driver['resource_audit'](row)
    assert audit['verdict']=='PASS' and audit['checks']['every_selected_edge_original_stopping_reachability']
    # Corrupt the ACTUAL selected speed, not the acceptance threshold.
    row['selected_actions'][0]['target_speed_m_s']=.05
    rejected=mission_offline_probe_driver['resource_audit'](row)
    assert rejected['verdict']=='FAIL' and not rejected['checks']['every_selected_edge_original_stopping_reachability']


def _NE_cache_models(maxsize=65536):
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    bounds=RigidBodyParameterBounds(**model_bounds())
    return MassNormalizedRigidBodyEdgeModel(bounds),ExactRigidBodyStepCache(bounds,maxsize=maxsize)


def _raw_numeric_fields(value):
    import struct
    from dataclasses import fields
    return tuple((f.name,type(getattr(value,f.name)),struct.pack('!d',getattr(value,f.name))
        if type(getattr(value,f.name)) is float else getattr(value,f.name)) for f in fields(value))


def test_NE_step_cache_reuses_only_identical_IEEE_inputs_including_signed_zero():
    from dataclasses import FrozenInstanceError
    plain,cached=_NE_cache_models();args=(0.,0.,0.,.05,0.,0.,.05)
    first=cached.step(*args);again=cached.step(*args)
    assert first is again and _raw_numeric_fields(first)==_raw_numeric_fields(plain.step(*args))
    negative=(-0.,*args[1:]);near=(math.nextafter(0.,1.),*args[1:])
    for values in (negative,near):assert _raw_numeric_fields(cached.step(*values))==_raw_numeric_fields(plain.step(*values))
    info=cached.cache_info();assert info['misses']==3 and info['hits']==1
    assert not info['input_quantisation_used'] and info['signed_zero_preserved']
    with pytest.raises(FrozenInstanceError):first.feasible=False


def test_NE_cache_owns_an_immutable_parameter_copy_and_independent_bounded_storage():
    from dataclasses import replace,FrozenInstanceError
    plain,cached=_NE_cache_models(maxsize=2)
    assert cached.bounds==plain.bounds and cached.bounds is not plain.bounds
    with pytest.raises(AttributeError):cached.bounds=replace(plain.bounds,friction_min=.2)
    with pytest.raises(FrozenInstanceError):cached.bounds.friction_min=.2
    for speed in (.01,.03,.05):cached.step(0.,0.,0.,speed,0.,0.,.05)
    assert cached.cache_info()['currsize']==2
    cached.step(0.,0.,0.,.01,0.,0.,.05)
    assert cached.cache_info()['misses']==4 and cached.cache_info()['currsize']==2
    _,other=_NE_cache_models();assert other.cache_info()['hits']==other.cache_info()['misses']==0


@pytest.mark.parametrize('maxsize',[0,-1,True,1.5])
def test_NE_cache_rejects_unbounded_or_noninteger_storage(maxsize):
    with pytest.raises(ValueError):_NE_cache_models(maxsize=maxsize)


def test_NE_cache_preserves_original_errors_and_nonfloat_input_behaviour():
    plain,cached=_NE_cache_models();args=(0,0,0,.05,0,0,.05)
    assert _raw_numeric_fields(cached.step(*args))==_raw_numeric_fields(plain.step(*args))
    assert cached.cache_info()['original_behaviour_uncached_bypasses']==1
    for model in (plain,cached):
        with pytest.raises(ValueError,match='dt_s must be positive'):model.step(0.,0.,0.,.05,0.,0.,0.)
    assert cached.cache_info()['currsize']==0


@pytest.mark.parametrize('mode',['coordinated_counterphase','front_only','rear_only'])
@pytest.mark.parametrize('curvature',[-.7,0.,.7])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('speed',[.003,.01,.03,.05])
def test_NE_step_cache_entire_original_propagation_is_bitwise_identical(mode,curvature,direction,speed):
    import struct
    from patient_initial_stability import steering_lattice as lattice
    plain,cached=_NE_cache_models()
    for initial_speed in (0.,.02,-.01):
        state=lattice.State(.123,-.321,.4,'coordinated_counterphase',.07,-.03,True,1,
            initial_speed,.001,-.002)
        args=(state,mode,curvature,direction)
        kw=dict(speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,target_speed_m_s=speed)
        expected,poses,elapsed=lattice.propagate(*args,rigid_body_model=plain,**kw)
        result,actual,seconds=lattice.propagate(*args,rigid_body_model=cached,**kw)
        assert _raw_numeric_fields(result)==_raw_numeric_fields(expected)
        assert struct.pack('!d',seconds)==struct.pack('!d',elapsed)
        assert b''.join(struct.pack('!3d',*p) for p in actual)==b''.join(struct.pack('!3d',*p) for p in poses)


@pytest.fixture(scope='module')
def exact_row_skip_checker():
    import hospital_native_envelope as native
    before=native.initialize_native();original=native._NATIVE
    derived=row_skipping_native_checker(native)
    assert native._NATIVE is original and native.initialize_native()==before
    assert derived._batch.__code__ is native.NativeHospitalEnvelopeChecker._batch.__code__
    return native,derived


@pytest.mark.parametrize('mask',['empty','sparse_row','random'])
def test_exact_row_skip_matches_original_C_numpy_and_counters_without_mutation(exact_row_skip_checker,mask):
    import numpy as np
    from hospital_geometry_tools import pose_collision
    native,Derived=exact_row_skip_checker;rng=np.random.default_rng(39173)
    occupied=np.zeros((500,600),dtype=bool);origin=np.array([-3.,-2.5]);resolution=.01
    if mask=='sparse_row':occupied[202,155:445]=True
    elif mask=='random':occupied=rng.random(occupied.shape)<.002
    original_grid=occupied.copy()
    a=native.NativeHospitalEnvelopeChecker(occupied,origin,resolution,(.2,-.1,.3))
    b=Derived(occupied,origin,resolution,(.2,-.1,.3))
    for pose in rng.uniform([-3.2,-2.7,-math.pi],[3.2,2.7,math.pi],size=(150,3)):
        x=a.world_collision(pose);y=b.world_collision(pose)
        assert x==y==pose_collision(occupied,pose,origin,resolution,.01)
    local=rng.uniform([-1.,-1.,-1.],[1.,1.,1.],size=(30,3))
    assert a(local)==b(local) and a.statistics==b.statistics
    assert np.array_equal(occupied,original_grid)
    assert np.array_equal(a.integral,b.integral)
    assert b.mission_collision_binding['undo_reproduces_entire_original_C_source']
    assert not b.mission_collision_binding['fast_math_used']


def test_exact_row_skip_keeps_grazing_and_outside_boundary_predicates(exact_row_skip_checker):
    import numpy as np
    native,Derived=exact_row_skip_checker;occupied=np.zeros((400,400),dtype=bool);occupied[200,300]=True
    args=(occupied,np.array([-2.,-2.]),.01,(0.,0.,0.))
    a=native.NativeHospitalEnvelopeChecker(*args);b=Derived(*args)
    for x in (1.005-.825-.005,1.005+.825+.005,-2.,2.):
        for delta in (-1e-9,-1e-12,0.,1e-12,1e-9):
            assert a.world_collision((x+delta,.005,0.))==b.world_collision((x+delta,.005,0.))
    assert a.statistics==b.statistics


def test_exact_row_skip_preserves_deadline_guard(exact_row_skip_checker):
    import numpy as np
    native,Derived=exact_row_skip_checker
    for cls in (native.NativeHospitalEnvelopeChecker,Derived):
        checker=cls(np.zeros((400,400),dtype=bool),np.array([-2.,-2.]),.01,(0.,0.,0.),deadline=0.)
        with pytest.raises(TimeoutError):checker([(0.,0.,0.)])


def test_probe_compute_revision_is_explicit_and_never_changes_original_defaults(mission_offline_probe_driver):
    from types import FunctionType
    original=mission_offline_probe_driver['validate_probe_selection'];before=dict(original.__globals__)
    function=FunctionType(original.__code__,dict(before,COMPUTE_REVISION='unknown_accelerator'),original.__name__)
    with pytest.raises(ValueError,match='computation revision'):function()
    assert original.__globals__==before and before['COMPUTE_REVISION']=='original_exact_computation_v1'


@pytest.mark.parametrize('revision',['original_exact_computation_v1','exact_NE_cache_and_empty_row_skip_v1',
    'compiled_original_NE_and_empty_row_skip_v1','compiled_original_NE_and_exact_sparse_cell_SAT_v1'])
def test_probe_compute_adapter_binding_preserves_parameter_values_and_original_native_global(mission_offline_probe_driver,revision):
    from types import FunctionType
    from dataclasses import asdict
    from hospital_h3_turn_planning import model_bounds
    import hospital_native_envelope as native
    original=mission_offline_probe_driver['make_compute_bindings'];function=FunctionType(original.__code__,
        dict(original.__globals__,COMPUTE_REVISION=revision),original.__name__)
    native.initialize_native();before=native._NATIVE
    Checker,model,build=function(model_bounds())
    assert asdict(model.bounds)==model_bounds() and native._NATIVE is before
    assert issubclass(Checker,native.NativeHospitalEnvelopeChecker)
    if revision=='original_exact_computation_v1':
        assert not hasattr(model,'cache_info') and build==native.initialize_native()
    elif revision=='exact_NE_cache_and_empty_row_skip_v1':
        assert isinstance(model,ExactRigidBodyStepCache) and model.cache_info()['maxsize']==65536
        assert build['revision']=='exact_empty_row_integral_broadphase_v1'
    else:
        assert not hasattr(model,'cache_info')
        assert build['revision']==('exact_sparse_occupied_cell_iteration_v1' if revision=='compiled_original_NE_and_exact_sparse_cell_SAT_v1' else 'exact_empty_row_integral_broadphase_v1')


@pytest.fixture(scope='module')
def compiled_NE_edge():
    pytest.importorskip('numba')
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability import rigid_body_edge as rigid
    before=(lattice.propagate.__code__,rigid.MassNormalizedRigidBodyEdgeModel.step.__code__)
    result=compiled_original_NE_propagation()
    yield result
    assert before==(lattice.propagate.__code__,rigid.MassNormalizedRigidBodyEdgeModel.step.__code__)


def _assert_original_NE_edge_bits(left,right):
    import struct
    assert _raw_numeric_fields(left[0])==_raw_numeric_fields(right[0])
    assert struct.pack('!d',left[2])==struct.pack('!d',right[2])
    assert b''.join(struct.pack('!3d',*p) for p in left[1])==b''.join(struct.pack('!3d',*p) for p in right[1])


@pytest.mark.parametrize('mode',['coordinated_counterphase','front_only','rear_only'])
@pytest.mark.parametrize('curvature',[-.7,-.5,-.25,0.,.25,.5,.7])
@pytest.mark.parametrize('direction',[-1,1])
@pytest.mark.parametrize('speed',[.003,.01,.03,.05])
def test_compiled_original_NE_all_primitive_state_pose_time_bits(compiled_NE_edge,mode,curvature,direction,speed):
    from patient_initial_stability import steering_lattice as lattice
    model,_=_NE_cache_models()
    for initial_speed in (0.,.02,-.01):
        state=lattice.State(.123,-.321,.4,'coordinated_counterphase',.07,-.03,True,1,
            initial_speed,.001,-.002)
        kw=dict(speed_m_s=.05,rigid_body_model=model,target_speed_m_s=speed)
        _assert_original_NE_edge_bits(compiled_NE_edge(state,mode,curvature,direction,**kw),
            lattice.propagate(state,mode,curvature,direction,**kw))
    assert compiled_NE_edge.compilation_statistics['compiled_calls']>0


def test_compiled_original_NE_parameter_force_corner_and_velocity_contrasts(compiled_NE_edge):
    import random
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel
    source,_=_NE_cache_models();rng=random.Random(20261117)
    for index in range(120):
        bounds=replace(source.bounds,friction_min=rng.uniform(.01,.7),max_abs_ltr=rng.uniform(.05,.9),
            com_height_max_m=rng.uniform(.6,1.1),caster_specific_force_x_abs_max_m_s2=rng.uniform(0.,.8),
            caster_specific_force_y_abs_max_m_s2=rng.uniform(0.,.8),
            caster_specific_yaw_moment_abs_max_m2_s2=rng.uniform(0.,.8))
        model=MassNormalizedRigidBodyEdgeModel(bounds)
        state=lattice.State(rng.uniform(-2.,2.),rng.uniform(-2.,2.),rng.uniform(-3.,3.),
            rng.choice(lattice.ALL_MODES),rng.uniform(-.8,.8),rng.uniform(-.8,.8),bool(index%2),
            rng.choice((-1,0,1)),rng.uniform(-.08,.08),rng.uniform(-.02,.02),rng.uniform(-.1,.1))
        args=(state,rng.choice(lattice.ALL_MODES),rng.choice(lattice.CURVATURES),rng.choice((-1,1)))
        kw=dict(speed_m_s=.05,rigid_body_model=model,target_speed_m_s=rng.choice((.003,.01,.03,.05)))
        _assert_original_NE_edge_bits(compiled_NE_edge(*args,**kw),lattice.propagate(*args,**kw))


def test_compiled_original_NE_continuous_state_and_resource_history_retained(compiled_NE_edge):
    from patient_initial_stability import steering_lattice as lattice
    model,_=_NE_cache_models()
    _,plain,key=resource_propagation_binding(lattice.State,lattice.propagate,MotionResourceLimits())
    _,compiled,_=resource_propagation_binding(lattice.State,compiled_NE_edge,MotionResourceLimits())
    a=b=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    for index in range(45):
        action=(lattice.ALL_MODES[index%3],lattice.CURVATURES[index%7],-1 if index%9==0 else 1)
        kw=dict(speed_m_s=.05,rigid_body_model=model,target_speed_m_s=(.01,.03,.05)[index%3])
        left=plain(a,*action,**kw);right=compiled(b,*action,**kw)
        _assert_original_NE_edge_bits(left,right)
        assert vars(left[1])==vars(right[1])
        assert key(lattice._state_key)(left[0],True)==key(lattice._state_key)(right[0],True)
        a,b=left[0],right[0]


def test_compiled_original_NE_unsupported_inputs_and_errors_reuse_original(compiled_NE_edge,monkeypatch):
    from patient_initial_stability import steering_lattice as lattice
    model,_=_NE_cache_models();state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    before=compiled_NE_edge.compilation_statistics['original_fallback_calls']
    cases=[dict(rigid_body_model=None),dict(rigid_body_model=model,response_scale_by_mode={m:1. for m in lattice.ALL_MODES})]
    for kw in cases:
        _assert_original_NE_edge_bits(compiled_NE_edge(state,lattice.ALL_MODES[0],0.,1,**kw),
            lattice.propagate(state,lattice.ALL_MODES[0],0.,1,**kw))
    monkeypatch.setattr(lattice,'STEERING_RATE_RAD_S',.1)
    kw=dict(rigid_body_model=model,speed_m_s=.05)
    _assert_original_NE_edge_bits(compiled_NE_edge(state,lattice.ALL_MODES[0],.7,1,**kw),
        lattice.propagate(state,lattice.ALL_MODES[0],.7,1,**kw))
    assert compiled_NE_edge.compilation_statistics['original_fallback_calls']==before+3
    for f in (compiled_NE_edge,lattice.propagate):
        with pytest.raises(ValueError,match='must be positive'):f(state,lattice.ALL_MODES[0],0.,1,step_m=0.,**kw)


def test_CPython311_two_coordinate_norm_bits_including_subnormals_and_specials():
    import random,struct
    from hospital_mission_planning import _cpython311_hypot2
    rng=random.Random(31114)
    cases=[(a,b) for a in (0.,-0.,1e-320,1e-308,1.,1e308,math.inf,math.nan)
        for b in (0.,-0.,1e-320,1e-308,-1.,1e308,math.inf,math.nan)]
    cases.extend((math.ldexp(rng.uniform(-1.,1.),rng.randrange(-1070,1020)),
        math.ldexp(rng.uniform(-1.,1.),rng.randrange(-1070,1020))) for _ in range(2000))
    for a,b in cases:
        expected=math.hypot(a,b);actual=_cpython311_hypot2(a,b)
        assert math.isnan(actual) if math.isnan(expected) else struct.pack('!d',actual)==struct.pack('!d',expected)


def test_compiled_CPython311_norm_preserves_binary_results(compiled_NE_edge):
    import random,struct
    rng=random.Random(131114)
    cases=[(a,b) for a in (0.,-0.,1e-320,1e-308,1.,1e308,math.inf,math.nan)
        for b in (0.,-0.,1e-320,1e-308,-1.,1e308,math.inf,math.nan)]
    cases.extend((rng.uniform(-100.,100.),rng.uniform(-100.,100.)) for _ in range(2000))
    for a,b in cases:
        expected=math.hypot(a,b);actual=compiled_NE_edge.compiled_norm_function(a,b)
        assert math.isnan(actual) if math.isnan(expected) else struct.pack('!d',actual)==struct.pack('!d',expected)


def test_compiled_original_NE_saved_search_sincos_rounding_regression(compiled_NE_edge):
    import struct
    from patient_initial_stability import steering_lattice as lattice
    number=lambda bits:struct.unpack('!d',bytes.fromhex(bits))[0]
    state=lattice.State(*map(number,('3fecbb3b37db75a4','bfa730d88442edb7','bfc56f262c025c76')),
        'coordinated_counterphase',number('bfd515418e59dcc4'),number('3fd2d958ba0f891c'),True,1,
        *map(number,('3fa9999999999996','bf4d2f566d76d765','bf98a15ed6e21950')))
    model,_=_NE_cache_models();kw=dict(speed_m_s=.05,rigid_body_model=model,target_speed_m_s=.003)
    _assert_original_NE_edge_bits(compiled_NE_edge(state,'rear_only',-.7,-1,**kw),
        lattice.propagate(state,'rear_only',-.7,-1,**kw))


@pytest.mark.parametrize('weight',[0.,.5,True,math.inf,math.nan,'3'])
def test_search_priority_rejects_invalid_or_implicit_weights(weight):
    from hospital_mission_planning import search_priority_weight
    with pytest.raises(ValueError,match='heuristic weight'):search_priority_weight(lambda:None,heuristic_weight=weight)


def test_search_priority_only_overrides_explicit_queue_weight_and_preserves_metadata():
    from hospital_mission_planning import search_priority_weight
    received=[];result=object()
    def original(*args,**kw):received.append((args,kw));return result
    original.mission_compilation_statistics={'compiled_calls':1};original.mission_search_binding={'unchanged':True}
    weighted=search_priority_weight(original,heuristic_weight=3.)
    kwargs=dict(heuristic_weight=1.5,speed_levels_m_s=(.003,.01,.03,.05),node_limit=60000)
    assert weighted('fixed goal',**kwargs) is result
    assert received==[(('fixed goal',),dict(kwargs,heuristic_weight=3.))]
    assert kwargs['heuristic_weight']==1.5
    assert weighted.mission_compilation_statistics is original.mission_compilation_statistics
    assert weighted.mission_search_binding is original.mission_search_binding
    assert weighted.mission_search_priority['optimality_NOT_claimed']


def test_probe_default_search_priority_unchanged_and_other_weight_cannot_reuse_completed_output(mission_offline_probe_driver,tmp_path):
    import json
    from types import FunctionType
    ns=mission_offline_probe_driver;assert ns['SEARCH_HEURISTIC_WEIGHT']==1.5
    output=tmp_path/'completed';output.mkdir()
    (output/'planning_review.json').write_text(json.dumps({
        'common_guidance_revision':ns['GUIDANCE_REVISION'],'common_resource_search_revision':ns['SEARCH_REVISION'],
        'common_compute_revision':ns['COMPUTE_REVISION'],'common_search_heuristic_weight':1.5}))
    f=ns['main'];scoped=dict(f.__globals__,OUTPUT=output,SEARCH_HEURISTIC_WEIGHT=3.,validate_probe_selection=lambda:None)
    function=FunctionType(f.__code__,scoped,f.__name__,f.__defaults__,f.__closure__)
    with pytest.raises(ValueError,match='another search priority'):function()


@pytest.mark.parametrize('kind',['empty','sparse'])
def test_sampled_body_pose_FFT_counts_match_original_finite_cell_SAT(kind):
    import numpy as np
    from hospital_mission_planning import sampled_body_pose_masks
    from hospital_planning_tools import HospitalEnvelopeChecker
    rng=np.random.default_rng(641337);grid=np.zeros((150,180),dtype=bool)
    if kind=='sparse':grid=rng.random(grid.shape)<.0008
    original=grid.copy();origin=np.array([-2.,-1.]);resolution=.04
    free,meta=sampled_body_pose_masks(grid,origin,resolution,stride=3,orientation_bins=8)
    checker=HospitalEnvelopeChecker(grid,origin,resolution,(0.,0.,0.))
    for k in range(8):
        for _ in range(40):
            y=int(rng.integers(free.shape[1]));x=int(rng.integers(free.shape[2]))
            pose=(meta['sample_origin_xy_m'][0]+x*meta['sample_step_m'],
                meta['sample_origin_xy_m'][1]+y*meta['sample_step_m'],k*meta['angle_step_rad'])
            assert bool(free[k,y,x])==(not checker.world_collision(pose))
    assert np.array_equal(grid,original) and meta['maximum_FFT_integer_residual']<1e-6
    assert meta['same_original_10mm_clearance'] and meta['sampled_geometry_only_NOT_edge_admission']


@pytest.mark.parametrize('kwargs',[{'stride':0},{'stride':True},{'orientation_bins':3},{'orientation_bins':73}])
def test_sampled_body_pose_sampling_requires_explicit_bounded_parameters(kwargs):
    import numpy as np
    from hospital_mission_planning import sampled_body_pose_masks
    with pytest.raises(ValueError):sampled_body_pose_masks(np.zeros((80,80),dtype=bool),(0.,0.),.05,**kwargs)


def test_sampled_configuration_guidance_keeps_original_door_and_needs_aligned_body():
    import numpy as np
    from dataclasses import replace
    from hospital_mission_planning import SampledBodyPoseGuidance
    from patient_initial_stability import steering_lattice as lattice
    grid=np.zeros((180,200),dtype=bool);grid[:,100]=True;grid[77:103,100]=False
    before=grid.copy();start=(2.,4.5,0.);goal=(8.,4.5,0.)
    fallback=lambda state,local:math.hypot(state.x-local[0],state.y-local[1])/.05
    field=SampledBodyPoseGuidance(grid,(0.,0.),.05,start,goal,fallback,
        directions=(1,-1),limits=MotionResourceLimits(),stride=2,orientation_bins=8)
    assert field.available and field.parameters['never_used_as_collision_checker']
    x=int(round((5.-field.origin[0])/field.step));y=int(round((4.5-field.origin[1])/field.step))
    assert field.free[0,y,x] and not field.free[4,y,x]
    state=lattice.State(2.4,0.,0.,'coordinated_counterphase',0.,0.,True,1)
    aligned=field(state,(6.,0.,0.));turned=field(replace(state,yaw=math.pi/2),(6.,0.,0.))
    assert math.isfinite(aligned) and math.isfinite(turned) and aligned<turned
    assert np.array_equal(grid,before)
    with pytest.raises(ValueError,match='fixed task'):field(state,(6.,0.,math.pi))
    assert field.parameters['modulo_pi_rectangle_symmetry_NOT_a_goal_heading_change']


def test_sampled_configuration_resource_priority_never_drops_actual_actions():
    import numpy as np
    from dataclasses import replace
    from hospital_mission_planning import SampledBodyPoseGuidance
    from patient_initial_stability import steering_lattice as lattice
    limits=MotionResourceLimits()
    State,_,_=resource_propagation_binding(lattice.State,lattice.propagate,limits)
    field=SampledBodyPoseGuidance(np.zeros((160,180),dtype=bool),(0.,0.),.05,(6.,4.,0.),(3.,4.,0.),
        lambda state,goal:100.,directions=(1,-1),limits=limits,stride=3,orientation_bins=6)
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,1)
    free=field(state,(-3.,0.,0.));exhausted=field(replace(state,mission_direction_changes=4),(-3.,0.,0.))
    no_reverse=field(replace(state,mission_reverse_total_bins=200),(-3.,0.,0.))
    assert free<exhausted and free<no_reverse
    assert state.mission_direction_changes==0 and field.directions==(1,-1)
    assert field.parameters['resource_limits_change_guidance_only']


def test_sampled_configuration_unavailable_or_outside_uses_original_fallback():
    import numpy as np
    from hospital_mission_planning import SampledBodyPoseGuidance
    from patient_initial_stability import steering_lattice as lattice
    field=SampledBodyPoseGuidance(np.ones((50,50),dtype=bool),(0.,0.),.05,(1.,1.,0.),(1.5,1.,0.),
        lambda state,goal:123.,directions=(1,-1),limits=MotionResourceLimits(),stride=2,orientation_bins=4)
    state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    assert not field.available and field(state,(.5,0.,0.))==123.


def test_configuration_probe_guidance_requires_original_map_not_a_baseline_path(mission_offline_probe_driver):
    from types import FunctionType,SimpleNamespace
    f=mission_offline_probe_driver['make_common_guidance']
    fake=SimpleNamespace(mission_local_heading_guidance=lambda *a,**k:(lambda state,goal:42.))
    scoped=dict(f.__globals__,GUIDANCE_REVISION='sampled_body_pose_resource_v1',binding=fake)
    function=FunctionType(f.__code__,scoped,f.__name__,f.__defaults__,f.__closure__)
    function.__kwdefaults__=dict(f.__kwdefaults__)
    with pytest.raises(ValueError,match='complete original map'):function(object(),{'local_goal_xyyaw':[1.,0.,0.]})


def test_configuration_probe_guidance_binds_same_start_goal_limits_and_complete_map(mission_offline_probe_driver):
    import numpy as np
    from types import FunctionType,SimpleNamespace
    calls=[];fallback=object();field=object()
    f=mission_offline_probe_driver['make_common_guidance'];grid=np.zeros((20,30),dtype=bool);origin=(1.,2.)
    def build(*args,**kw):calls.append((args,kw));return field
    fake=SimpleNamespace(mission_local_heading_guidance=lambda *a,**k:fallback,SampledBodyPoseGuidance=build)
    function=FunctionType(f.__code__,dict(f.__globals__,GUIDANCE_REVISION='sampled_body_pose_resource_v1',binding=fake),
        f.__name__,f.__defaults__,f.__closure__);function.__kwdefaults__=dict(f.__kwdefaults__)
    manifest={'local_goal_xyyaw':[1.,0.,0.],'start_xyyaw':[3.,4.,0.],'goal_xyyaw':[4.,4.,0.]}
    assert function(object(),manifest,occupied=grid,origin=origin,resolution=.01) is field
    args,kw=calls[0];assert args[0] is grid and args[1] is origin and args[3:]==(manifest['start_xyyaw'],manifest['goal_xyyaw'],fallback)
    assert kw['directions']==(1,-1) and kw['limits']==MotionResourceLimits()
    assert kw['stride']==5 and kw['orientation_bins']==18


def _owned_reference_fixture(points=None,goal=(1.,0.,0.),**settings):
    points=points or [(.25,0.,0.),(.5,0.,0.),(.75,0.,0.),(1.,0.,0.)]
    candidate=dict(plan_success=True,selected_actions=[dict(mode='coordinated_counterphase',direction=1) for _ in points],
        selected_states=[dict(x_m=x,y_m=y,yaw_deg=math.degrees(yaw)) for x,y,yaw in points])
    values=dict(speed_m_s=.05,maximum_curvature_1_m=.7,heading_weight=.4);values.update(settings)
    return candidate,OwnedSpatialReferenceGuidance(candidate,goal,lambda state,goal:123.,**values)


def test_owned_reference_guides_steady_progress_without_inserting_or_resetting_any_state():
    from types import SimpleNamespace
    from copy import deepcopy
    candidate,guide=_owned_reference_fixture();before=deepcopy(candidate)
    scores=[]
    for x in (0.,.1,.3,.6,.9):
        state=SimpleNamespace(x=x,y=0.,yaw=0.,front=.2,longitudinal_velocity=.04)
        old=vars(state).copy();scores.append(guide(state,(1.,0.,0.)))
        assert vars(state)==old and scores[-1]==pytest.approx((1.-x)/.05)
    assert candidate==before and all(a>b for a,b in zip(scores,scores[1:]))
    assert guide.parameters['reference_states_never_inserted']
    assert guide.parameters['no_hard_corridor_or_waypoint_or_action_filter']
    with pytest.raises(ValueError):guide.poses[0,0]=99.


def test_owned_reference_is_soft_at_arbitrarily_far_actual_states_not_a_corridor_rejection():
    from types import SimpleNamespace
    _,guide=_owned_reference_fixture()
    state=SimpleNamespace(x=.5,y=20.,yaw=math.pi)
    assert math.isfinite(guide(state,(1.,0.,0.))) and guide(state,(1.,0.,0.))>400.
    with pytest.raises(ValueError,match='fixed goal'):guide(state,(1.001,0.,0.))
    state.x=math.nan
    with pytest.raises(ValueError):guide(state,(1.,0.,0.))


def test_owned_reference_preserves_full_terminal_body_heading_and_original_fallback():
    from types import SimpleNamespace
    _,guide=_owned_reference_fixture()
    at_goal=SimpleNamespace(x=1.,y=0.,yaw=math.pi)
    assert guide(at_goal,(1.,0.,0.))==123.
    with pytest.raises(ValueError,match='ORIGINAL fixed spatial goal'):
        _owned_reference_fixture(points=[(1.,0.,math.pi)])


@pytest.mark.parametrize('settings',[dict(speed_m_s=0.),dict(maximum_curvature_1_m=math.nan),dict(heading_weight=-1.)])
def test_owned_reference_rejects_invalid_ranking_inputs(settings):
    with pytest.raises(ValueError):_owned_reference_fixture(**settings)


def test_owned_reference_can_rank_retraced_reverse_segments_without_forcing_reference_direction():
    from types import SimpleNamespace
    candidate,guide=_owned_reference_fixture(points=[(-.1,0.,0.),(0.,0.,0.),(.5,0.,0.),(1.,0.,0.)])
    candidate['selected_actions'][0]['direction']=-1
    state=SimpleNamespace(x=0.,y=0.,yaw=0.,direction=1)
    assert guide(state,(1.,0.,0.))==pytest.approx(20.)
    state.direction=-1
    assert guide(state,(1.,0.,0.))==pytest.approx(20.)


@pytest.mark.parametrize('budget',[180.,600.])
@pytest.mark.parametrize('found',[True,False])
@pytest.mark.parametrize('terminal',[False,True])
def test_owned_reference_driver_charges_fresh_generation_and_NE_nodes_to_one_budget(mission_offline_probe_driver,found,budget,terminal):
    from types import FunctionType,SimpleNamespace
    from copy import deepcopy
    f=mission_offline_probe_driver['run_owned_reference_NE_pipeline'];before=dict(f.__globals__)
    now=[0.];calls=[];events=[];candidate,_=_owned_reference_fixture()
    candidate.update(method=before['METHODS'][0],status='SOFTWARE_REFERENCE_NOT_NE_PLAN',plan_success=found,
        candidate_generation_diagnostics=dict(expanded_nodes=17000))
    def generate(*args):
        calls.append(args);now[0]+=20.;return deepcopy(candidate)
    def search(*args,**kw):
        calls.append(('NE',kw));now[0]+=3.
        return dict(plan_success=False,candidate_generation_diagnostics=dict(expanded_nodes=2000))
    def observe(kind,source,row,event):events.append((kind,source));now[0]+=1.
    fake_binding=SimpleNamespace(**{k:getattr(before['binding'],k) for k in ('OwnedSpatialReferenceGuidance',)})
    fake_binding.NEStraightArcTimeGuidance=lambda *a,**kw:(lambda distance,speed:distance/.05)
    wrapped=[]
    class CommonGuide:
        def __call__(self,state,goal):return 0.
        def wrap_reference(self,reference):
            wrapped.append(reference)
            return TerminalPoseConnectionGuidance(reference,(1.,0.,0.),connection_metric=lambda *args:[0.],
                speed_m_s=.05,maximum_curvature_1_m=.7,terminal_zone_m=1.63)
    namespace=dict(before,TOTAL_WALL_S=budget,time=SimpleNamespace(perf_counter=lambda:now[0]),binding=fake_binding,
        GUIDANCE_REVISION='terminal_pose_connection_v1' if terminal else 'local_heading_terminal_blend_v1',
        P_SOURCE_PRIORITY=(candidate['method'],),run_joint_NE_pipeline=search)
    function=FunctionType(f.__code__,namespace,f.__name__)
    row,account=function(generate,SimpleNamespace(mission_original_NE_propagate=object()),'BOUNDS',CommonGuide(),'CHECKER',
        dict(local_goal_xyyaw=[1.,0.,0.]),None,None,None,'MODEL',budget,observe)
    assert calls[0][-3:]==(budget*2/3,60000,{}) and events==[('generation',candidate['method'])]
    assert account['generator_expanded_nodes']==17000 and account['NE_node_cap']==43000
    assert account['elapsed_s']==(24. if found else 21.)
    if found:
        assert calls[1]==('NE',dict(node_limit=43000)) and account['expanded_nodes_total']==19000
        assert row['P_owned_reference_parameters']['reference_changes_priority_only']
    else:
        assert len(calls)==1 and account['NE_expanded_nodes']==0 and not row['plan_success']
    assert f.__globals__==before
    assert len(wrapped)==int(terminal and found)


def test_owned_reference_driver_binds_remaining_NE_limit_and_keeps_all_original_actions(mission_offline_probe_driver):
    from types import FunctionType,SimpleNamespace
    f=mission_offline_probe_driver['run_joint_NE_pipeline'];seen=[]
    def empty(*a,**kw):seen.append(kw);return None
    function=FunctionType(f.__code__,dict(f.__globals__,binding=SimpleNamespace(plan_NE_mission_candidate=empty)),f.__name__)
    function.__kwdefaults__=dict(f.__kwdefaults__)
    row=function('SEARCH','BOUNDS','GUIDANCE',FreeCheckerFixture,
        dict(start_xyyaw=[0.,0.,0.],local_goal_xyyaw=[1.,0.,0.]),None,None,None,'MODEL',123.,node_limit=1234)
    assert seen[0]['node_limit']==1234 and seen[0]['directions']==(1,-1)
    assert seen[0]['speed_levels_m_s']==(.003,.01,.03,.05) and row['allowed_modes']==f.__globals__['lattice'].ALL_MODES


@pytest.mark.parametrize('case',['valid','missing_family','wrong_guidance'])
def test_owned_reference_probe_requires_explicit_isolated_source_families(mission_offline_probe_driver,case):
    from types import FunctionType
    f=mission_offline_probe_driver['validate_probe_selection'];old=f.__globals__
    values=dict(old,P_PIPELINE_REVISION='owned_reference_joint_NE_v1',
        GUIDANCE_REVISION='local_heading_terminal_blend_v1',SEARCH_REVISION='resource_pareto_best_constructor_only_v1',
        RUN_METHODS=(old['METHODS'][-1],),P_SOURCE_PRIORITY=(old['METHODS'][0],old['METHODS'][2],old['METHODS'][1]),
        OUTPUT=old['GATE']/'hospital_h6_planning/software_fixture_owned_NE_NOT_run')
    if case=='missing_family':values['P_SOURCE_PRIORITY']=old['METHODS'][:1]
    elif case=='wrong_guidance':values['GUIDANCE_REVISION']='sampled_body_pose_resource_v1'
    function=FunctionType(f.__code__,values,f.__name__)
    if case=='valid':function()
    else:
        with pytest.raises(ValueError):function()


@pytest.mark.parametrize('budget',[180.,600.,0.,-1.,math.nan,math.inf,True,'600',3600.])
def test_offline_extended_budget_is_explicit_and_never_silent_or_unbounded(mission_offline_probe_driver,budget):
    from types import FunctionType
    f=mission_offline_probe_driver['validate_probe_selection'];old=f.__globals__
    values=dict(old,TOTAL_WALL_S=budget,P_PIPELINE_REVISION='owned_reference_joint_NE_v1',
        GUIDANCE_REVISION='local_heading_terminal_blend_v1',SEARCH_REVISION='resource_pareto_best_constructor_only_v1',
        RUN_METHODS=(old['METHODS'][-1],),OUTPUT=old['GATE']/'hospital_h6_planning/software_budget_fixture_NOT_run')
    function=FunctionType(f.__code__,values,f.__name__)
    if type(budget) in (int,float) and budget in (180.,600.):function()
    else:
        with pytest.raises(ValueError,match='budget'):function()


def test_extended_P_diagnosis_cannot_be_mislabelled_as_existing_matched_comparison(mission_offline_probe_driver):
    from types import FunctionType
    f=mission_offline_probe_driver['validate_probe_selection'];old=f.__globals__
    values=dict(old,TOTAL_WALL_S=600.,P_PIPELINE_REVISION='owned_reference_joint_NE_v1',RUN_METHODS=old['METHODS'])
    with pytest.raises(ValueError,match='P-only'):FunctionType(f.__code__,values,f.__name__)()


def test_extended_budget_cannot_reuse_or_overwrite_completed180s_result(mission_offline_probe_driver,tmp_path):
    import json
    from types import FunctionType
    ns=mission_offline_probe_driver;output=tmp_path/'completed';output.mkdir()
    protocol=output/'protocol.json';protocol.write_text(json.dumps({'common_end_to_end_planning_wall_budget_s':180.}))
    (output/'planning_review.json').write_text(json.dumps(dict(protocol=str(protocol),
        common_guidance_revision=ns['GUIDANCE_REVISION'],common_resource_search_revision=ns['SEARCH_REVISION'],
        common_compute_revision=ns['COMPUTE_REVISION'])))
    f=ns['main'];scoped=dict(f.__globals__,OUTPUT=output,TOTAL_WALL_S=600.,validate_probe_selection=lambda:None)
    with pytest.raises(ValueError,match='another planning budget'):FunctionType(f.__code__,scoped,f.__name__)()
    assert json.loads(protocol.read_text())['common_end_to_end_planning_wall_budget_s']==180.


@pytest.mark.parametrize('direction',[1,-1])
def test_owned_reference_actual_original_NE_search_keeps_continuous_state_and_original_terminal_guard(compiled_NE_edge,direction):
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    goal=(direction*.3,0.,0.)
    source=lattice.plan(goal,-1.,1.,minimum_x=-1.,maximum_x=1.,
        allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),directions=(direction,),
        collision_checker=lambda poses:False,dynamics=lattice.KINEMATIC_DYNAMICS,node_limit=30)
    assert source is not None
    candidate=dict(plan_success=True,selected_actions=list(source.actions),selected_states=[
        dict(x_m=s.x,y_m=s.y,yaw_deg=math.degrees(s.yaw)) for s in source.states])
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    times=NEStraightArcTimeGuidance(compiled_NE_edge,model,dynamics=lattice.FULL_DYNAMICS,
        speed_levels_m_s=(.003,.01,.03,.05),maximum_speed_m_s=.05,terminal_speed_m_s=.005,
        step_m=.05,maximum_distance_m=.5,check_deadline=lambda:None)
    guide=OwnedSpatialReferenceGuidance(candidate,goal,lambda state,goal:0.,speed_m_s=.05,
        maximum_curvature_1_m=.7,heading_weight=.4,arc_time_guidance=times)
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,
        lattice.State,compiled_NE_edge,lattice._state_key,MotionResourceLimits())
    args=_joint_NE_fixture_arguments()
    args.update(bounds=mission_search_bounds([-1.,1.,-1.,1.],[0.,0.,0.],goal),goal=goal,
        guidance=guide,model=model,
        directions=(1,-1),node_limit=300)
    result=plan_NE_mission_candidate(pareto_resource_lattice_function(base),**args)
    assert result is not None and result.actions and result.states
    final=result.states[-1]
    assert math.hypot(final.x-goal[0],final.y-goal[1])<=.025
    assert abs(lattice.wrap(final.yaw-goal[2]))<=math.radians(2.)
    assert math.hypot(final.longitudinal_velocity,final.lateral_velocity)<=.005
    state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True,0)
    for action,saved in zip(result.actions,result.states):
        state,poses,_=lattice.propagate(state,*action[:3],speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,
            target_speed_m_s=action[3],rigid_body_model=args['model'])
        for name in lattice.State.__dataclass_fields__:assert getattr(state,name)==getattr(saved,name)
        assert saved.mission_reverse_total_bins*.005<=1. and saved.mission_direction_changes<=4


def test_NE_arc_time_prices_original_terminal_speed_instead_of_constant_maximum_speed():
    from types import SimpleNamespace
    seen=[]
    def sample(state,mode,curvature,direction,**kw):
        seen.append((state,mode,curvature,direction,kw))
        return SimpleNamespace(longitudinal_velocity=kw['target_speed_m_s'],lateral_velocity=0.),[],kw['step_m']/kw['target_speed_m_s']
    table=NEStraightArcTimeGuidance(sample,'ORIGINAL_MODEL',dynamics='ORIGINAL_DYNAMICS',
        speed_levels_m_s=(.003,.01,.03,.05),maximum_speed_m_s=.05,terminal_speed_m_s=.005,
        step_m=.05,maximum_distance_m=.5,check_deadline=lambda:None)
    assert len(seen)==244 and table(.05,.05)==pytest.approx(.05/.003)
    assert table(.1,.05)==pytest.approx(1.+.05/.003)
    assert table(.3,.05)>.3/.05 and table(.3,.05)<table(.35,.05)
    assert all(c[-1]['rigid_body_model']=='ORIGINAL_MODEL' and c[-1]['dynamics']=='ORIGINAL_DYNAMICS' for c in seen)
    assert math.isfinite(table(20.,.08)) and table.parameters['only_relaxed_ranking_NOT_actual_state_or_edge_or_stopping_admission']
    with pytest.raises(ValueError):table(-1.,.05)
    with pytest.raises(ValueError):table(.05,math.nan)


def test_NE_arc_time_keeps_actual_deadline_exception_without_approving_a_partial_table():
    error=TimeoutError('ORIGINAL_TOTAL_BUDGET_DURING_GUIDANCE')
    def deadline():raise error
    with pytest.raises(TimeoutError) as caught:
        NEStraightArcTimeGuidance(lambda *a,**kw:pytest.fail('No edge after deadline'),'MODEL',dynamics='DYNAMICS',
            speed_levels_m_s=(.003,.01,.03,.05),maximum_speed_m_s=.05,terminal_speed_m_s=.005,
            step_m=.05,maximum_distance_m=.5,check_deadline=deadline)
    assert caught.value is error


def test_original_prefix_copy_keeps_every_real_state_parent_action_and_cost_without_mutation():
    from dataclasses import asdict
    from copy import deepcopy
    from patient_initial_stability.steering_lattice import State
    states=[State(i*.05,0.,0.,'front_only',.1,.2,True,1,longitudinal_velocity=.03,
        lateral_velocity=.002,yaw_rate=.01) for i in range(3)]
    nodes=[dict(state=s,parent=i-1 if i else None,action=('front_only',.25,1,.03) if i else None,cost=float(i))
        for i,s in enumerate(states)]
    observed=dict(nodes=nodes,state=states[1],best_goal_state=states[2],queue=[(123.,7,2)])
    before=deepcopy(observed);out=copy_original_search_prefixes(observed)
    assert observed==before and out['raw_queue_head_may_be_stale']
    for name,count in [('goal_closest',2),('currently_expanded',1),('raw_queue_head',2)]:
        prefix=out['prefixes'][name];assert prefix['edge_count']==count and prefix['goal_completion_NOT_claimed']
        for i,record in enumerate(prefix['records']):
            assert record['state']==asdict(states[i]) and record['cost_s']==float(i)
    out['prefixes']['goal_closest']['records'][2]['state']['x']=99.
    assert states[2].x==.1


@pytest.mark.parametrize('raise_after',[1,5,30])
def test_prefix_diagnosis_rethrows_same_original_timeout_and_never_changes_search_work(pareto_original_lattice,raise_after):
    lattice,base,_=pareto_original_lattice
    def attempt(capture):
        function=pareto_resource_lattice_function(base,capture_prefixes=capture)
        diagnostic={};calls=[];error=TimeoutError('SAME_FIXED_WORK_LIMIT_NOT_A_NEW_PRIMARY')
        def estimate(state):
            calls.append(state)
            if len(calls)==raise_after:raise error
            return math.hypot(state.x-.5,state.y-.3)/.05
        with pytest.raises(TimeoutError) as caught:
            function((.5,.3,0.),-2.,2.,allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),
                speed_m_s=.05,node_limit=1000,dynamics=lattice.KINEMATIC_DYNAMICS,directions=(1,-1),
                minimum_x=-1.,maximum_x=1.,collision_checker=resource_checker_class(FreeCheckerFixture)(),
                diagnostics=diagnostic,heuristic_function=estimate)
        assert caught.value is error
        return diagnostic,calls
    plain,plain_calls=attempt(False);captured,captured_calls=attempt(True)
    prefixes=captured.pop('original_search_prefixes')
    assert plain==captured and plain_calls==captured_calls
    assert prefixes['diagnostic_only_NOT_a_complete_plan']
    for prefix in prefixes['prefixes'].values():
        assert prefix['records'][0]['action'] is None and prefix['records'][0]['cost_s']==0.
        assert all('longitudinal_velocity' in r['state'] and 'mission_reverse_total_bins' in r['state'] for r in prefix['records'])


def test_original_prefix_copy_rejects_a_parent_cycle_instead_of_inventing_a_repaired_history():
    from patient_initial_stability.steering_lattice import State
    state=State(0.,0.,0.,'front_only',0.,0.,True)
    with pytest.raises(ValueError,match='backward tree'):
        copy_original_search_prefixes(dict(nodes=[dict(state=state,parent=0,action=None,cost=0.)],best_goal_state=state))


def test_prefix_capture_selector_cannot_enable_implicitly(pareto_original_lattice):
    _,base,_=pareto_original_lattice
    with pytest.raises(ValueError,match='boolean'):pareto_resource_lattice_function(base,capture_prefixes=1)


@pytest.fixture(scope='module')
def exact_sparse_cell_checker():
    import hospital_native_envelope as native
    before=native.initialize_native();original=native._NATIVE
    derived=sparse_cell_native_checker(native)
    assert native._NATIVE is original and native.initialize_native()==before
    return native,derived


@pytest.mark.parametrize('mask',['empty','sparse_row','sparse_column','random','dense'])
def test_exact_sparse_cells_match_original_C_numpy_and_counters(exact_sparse_cell_checker,mask):
    import numpy as np
    from hospital_geometry_tools import pose_collision
    native,Derived=exact_sparse_cell_checker;rng=np.random.default_rng(409184)
    occupied=np.zeros((500,600),dtype=bool);origin=np.array([-3.,-2.5]);resolution=.01
    if mask=='sparse_row':occupied[202,155:445]=True
    elif mask=='sparse_column':occupied[155:445,202]=True
    elif mask=='random':occupied=rng.random(occupied.shape)<.002
    elif mask=='dense':occupied=rng.random(occupied.shape)<.6
    original_grid=occupied.copy()
    a=native.NativeHospitalEnvelopeChecker(occupied,origin,resolution,(.2,-.1,.3));b=Derived(occupied,origin,resolution,(.2,-.1,.3))
    for pose in rng.uniform([-3.2,-2.7,-math.pi],[3.2,2.7,math.pi],size=(150,3)):
        assert a.world_collision(pose)==b.world_collision(pose)==pose_collision(occupied,pose,origin,resolution,.01)
    local=rng.uniform([-1.,-1.,-1.],[1.,1.,1.],size=(30,3))
    assert a(local)==b(local) and a.statistics==b.statistics
    assert np.array_equal(a.integral,b.integral) and np.array_equal(occupied,original_grid)
    assert b.mission_collision_binding['undo_reproduces_entire_original_C_source']
    assert not b.mission_collision_binding['fast_math_used']


def test_sparse_cell_snapshot_cannot_be_desynchronized_by_later_caller_array_writes(exact_sparse_cell_checker):
    import numpy as np
    _,Derived=exact_sparse_cell_checker;occupied=np.zeros((400,400),dtype=bool)
    checker=Derived(occupied,np.array([-2.,-2.]),.01,(0.,0.,0.))
    assert not checker([(0.,0.,0.)]);occupied[:]=True
    assert not checker([(0.,0.,0.)]) and not checker.occupancy.any()
    for array in (checker.occupancy,checker.integral,checker._row_offsets,checker._occupied_cols):
        assert not array.flags.writeable


def test_exact_sparse_cells_keep_boundary_nonfinite_and_deadline_behavior(exact_sparse_cell_checker):
    import numpy as np
    native,Derived=exact_sparse_cell_checker;occupied=np.zeros((400,400),dtype=bool);occupied[200,300]=True
    args=(occupied,np.array([-2.,-2.]),.01,(0.,0.,0.))
    a=native.NativeHospitalEnvelopeChecker(*args);b=Derived(*args)
    for x in (1.005-.825-.005,1.005+.825+.005,-2.,2.):
        for delta in (-1e-9,-1e-12,0.,1e-12,1e-9):
            assert a.world_collision((x+delta,.005,0.))==b.world_collision((x+delta,.005,0.))
    assert a.statistics==b.statistics
    for cls in (native.NativeHospitalEnvelopeChecker,Derived):
        checker=cls(*args,deadline=0.)
        with pytest.raises(TimeoutError):checker([(0.,0.,0.)])
        checker=cls(*args)
        with pytest.raises(ValueError):checker([(math.nan,0.,0.)])


def test_original_continuation_carries_full_pose_steering_velocity_and_resource_history(pareto_original_lattice):
    from dataclasses import asdict
    lattice,base,_=pareto_original_lattice;State=base.__globals__['State']
    initial=State(.1,.02,.01,'front_only',.12,.03,True,1,longitudinal_velocity=.04,
        lateral_velocity=.002,yaw_rate=.01,mission_reverse_total_bins=60,mission_direction_changes=2)
    before=dict(base.__globals__);continued=continue_original_lattice_from_state(base,initial)
    assert continued.__code__ is base.__code__ and base.__globals__==before
    captured=[]
    error=TimeoutError('OBSERVE_EXACT_INITIAL_STATE_ONLY')
    def heuristic(state):captured.append(state);raise error
    with pytest.raises(TimeoutError) as caught:
        continued((.3,0.,0.),-1.,1.,collision_checker=lambda poses:False,heuristic_function=heuristic)
    assert caught.value is error and captured==[initial] and captured[0] is initial
    assert continued.mission_continuation['exact_initial_state']==asdict(initial)


def test_original_continuation_does_not_reset_consumed_reverse_budget(pareto_original_lattice):
    lattice,base,_=pareto_original_lattice;State=base.__globals__['State']
    initial=State(-.05,0.,0.,'coordinated_counterphase',0.,0.,True,-1,
        mission_reverse_episode_bins=100,mission_reverse_total_bins=100)
    continued=continue_original_lattice_from_state(base,initial)
    checker=resource_checker_class(FreeCheckerFixture)();diagnostics={}
    result=continued((-.1,0.,0.),-1.,1.,minimum_x=-1.,maximum_x=1.,
        allowed_modes=('coordinated_counterphase',),allowed_curvatures=(0.,),directions=(-1,),
        dynamics=lattice.KINEMATIC_DYNAMICS,collision_checker=checker,node_limit=30,diagnostics=diagnostics)
    assert result is None and checker.statistics['motion_resource_reverse_episode_rejections']>0


def _original_forecast_geometry():
    from patient_initial_stability import steering_lattice as l
    return dict(wheelbase_m=l.WHEELBASE_M,front_length_m=l.FRONT_LENGTH_M,rear_length_m=l.REAR_LENGTH_M,
        curvature_scales=dict(l.MODE_CURVATURE_RESPONSE_SCALE),lateral_scales=dict(l.MODE_LATERAL_RESPONSE_SCALE))


@pytest.mark.parametrize('direction',[1,-1])
def test_soft_measured_steering_forecast_matches_closed_form_geometry_but_never_changes_actual_state(direction):
    from types import SimpleNamespace
    _,guide=_owned_reference_fixture(forecast_distance_m=.4,forecast_geometry=_original_forecast_geometry())
    state=SimpleNamespace(x=0.,y=0.,yaw=0.,front=.2,rear=-.2,direction=direction,mode='coordinated_counterphase')
    before=vars(state).copy();x,y,yaw=guide.forecast_pose(state,.4,1)
    g=_original_forecast_geometry();k=2*math.tan(.2)/g['wheelbase_m']*g['curvature_scales'][state.mode]
    assert (x,y,yaw)==pytest.approx((math.sin(k*.4*direction)/k,(1-math.cos(k*.4*direction))/k,k*.4*direction))
    assert vars(state)==before and guide.parameters['forecast_is_held_measured_steering_ranking_NOT_NE_or_state_insertion']


def test_soft_forecast_truncates_at_reference_direction_reversal_not_a_fictitious_long_reverse():
    from types import SimpleNamespace
    candidate,plain=_owned_reference_fixture(points=[(-.1,0.,0.),(0.,0.,math.pi/2),(.5,.4,.4),(1.,0.,0.)])
    candidate['selected_actions']=[dict(direction=-1),dict(direction=1),dict(direction=1),dict(direction=1)]
    guide=OwnedSpatialReferenceGuidance(candidate,(1.,0.,0.),lambda s,g:0.,speed_m_s=.05,
        maximum_curvature_1_m=.7,heading_weight=.4,forecast_distance_m=.4,forecast_geometry=_original_forecast_geometry())
    seen=[]
    def observation(state,distance,direction):seen.append((distance,direction));return (state.x,state.y,state.yaw)
    guide.forecast_pose=observation
    state=SimpleNamespace(x=-.05,y=0.,yaw=0.,longitudinal_velocity=0.,lateral_velocity=0.)
    assert math.isfinite(guide(state,(1.,0.,0.)))
    assert seen==[(pytest.approx(.05),-1)]


def test_soft_forecast_rewards_early_appropriate_steering_instead_of_waiting_for_body_yaw_error():
    from types import SimpleNamespace
    g=_original_forecast_geometry();k=.5
    points=[(math.sin(k*s)/k,(1-math.cos(k*s))/k,k*s) for s in (.1,.2,.4,.6,.8,1.)]
    goal=points[-1];candidate,guide=_owned_reference_fixture(points=points,goal=goal,
        forecast_distance_m=.4,forecast_geometry=g)
    angle=math.atan(k*g['wheelbase_m']/2/g['curvature_scales']['coordinated_counterphase'])
    def state(front,rear):return SimpleNamespace(x=0.,y=0.,yaw=0.,front=front,rear=rear,direction=1,
        mode='coordinated_counterphase',longitudinal_velocity=.05,lateral_velocity=0.)
    assert guide(state(angle,-angle),goal)<guide(state(0.,0.),goal)
    assert guide.parameters['reference_states_never_inserted']


def test_soft_forecast_requires_explicit_original_coefficients_without_changing_default():
    with pytest.raises(ValueError,match='original geometry'):_owned_reference_fixture(forecast_distance_m=.4)
    with pytest.raises(ValueError,match='bounded'):_owned_reference_fixture(forecast_distance_m=2.)
    _,guide=_owned_reference_fixture()
    assert guide.forecast_distance==0. and guide.forecast_geometry is None


@pytest.fixture(scope='module')
def installed_RS_metric():
    from pathlib import Path
    if not Path('/opt/ros/humble/lib/x86_64-linux-gnu/libompl.so.1.7.0').is_file():
        pytest.skip('Optional installed OMPL metric not available; no package installation in tests')
    return installed_reeds_shepp_metric()


def test_installed_RS_metric_matches_straight_rotation_symmetry_and_curvature_units(installed_RS_metric):
    import numpy as np
    metric=installed_RS_metric
    assert metric((0.,0.,0.),[(0.,0.,0.),(1.,0.,0.),(-1.,0.,0.)],1.4)==pytest.approx([0.,1.,1.])
    poses=np.array([[.3,.1,.2],[.4,-.2,-.3],[0.,.1,0.],[0.,0.,.4]])
    lengths=metric((0.,0.,0.),poses,1.4)
    assert (lengths>=np.hypot(poses[:,0],poses[:,1])-1e-12).all()
    assert lengths[-1]>=1.4*.4-1e-12
    for pose,length in zip(poses,lengths):
        assert metric(pose,[(0.,0.,0.)],1.4)[0]==pytest.approx(length)
    assert metric((0.,0.,0.),poses*2,2.8)[0]!=pytest.approx(lengths[0]*2) # yaw is not a length
    scaled=poses.copy();scaled[:,:2]*=2
    assert metric((0.,0.,0.),scaled,2.8)==pytest.approx(lengths*2)
    assert metric((0.,0.,0.),[(0.,.1,0.)],1.4)[0]>.3
    assert metric.parameters['original_NE_and_resource_guard_NOT_replaced']


def test_installed_RS_metric_rejects_nonfinite_or_invalid_radius(installed_RS_metric):
    with pytest.raises(ValueError):installed_RS_metric((0.,0.,0.),[(0.,math.nan,0.)],1.)
    with pytest.raises(ValueError):installed_RS_metric((0.,0.,0.),[(0.,0.,0.)],0.)


def test_oriented_reference_connection_replaces_sideways_and_independent_yaw_cost_without_state_reset(installed_RS_metric):
    from types import SimpleNamespace
    _,plain=_owned_reference_fixture();_,oriented=_owned_reference_fixture(connection_metric=installed_RS_metric)
    state=SimpleNamespace(x=.4,y=.1,yaw=0.)
    before=vars(state).copy()
    assert oriented(state,(1.,0.,0.))>plain(state,(1.,0.,0.))+4.
    assert vars(state)==before
    state.y=0.
    assert oriented(state,(1.,0.,0.))==pytest.approx(plain(state,(1.,0.,0.)))
    with pytest.raises(ValueError,match='one connection'):
        _owned_reference_fixture(connection_metric=installed_RS_metric,forecast_distance_m=.4,forecast_geometry=_original_forecast_geometry())


def test_reference_corridor_is_explicit_method_restriction_not_false_obstacle_or_state_reset():
    candidate,_=_owned_reference_fixture();seen=[]
    class Original(FreeCheckerFixture):
        def __call__(self,poses):seen.append(poses);return False
    Checker=reference_corridor_checker(Original,candidate,position_radius_m=.08,heading_radius_rad=math.radians(10.))
    checker=Checker();valid=[(.3,.01,0.)];outside=[(.3,.081,0.)]
    assert checker(valid) is False and seen==[valid]
    assert checker(outside) is True and seen==[valid] and outside==[(.3,.081,0.)]
    assert checker([(.3,0.,math.radians(10.1))]) is True
    assert checker.statistics['method_reference_corridor_rejections_NOT_obstacle_collisions']==2
    assert checker.mission_reference_corridor['search_restriction_NOT_obstacle_or_physical_guard']


def test_reference_corridor_never_waives_original_substep_obstacle_or_resource_guard():
    candidate,_=_owned_reference_fixture();calls=[]
    class Original(FreeCheckerFixture):
        def __call__(self,poses):calls.append(poses);return True
    Checker=reference_corridor_checker(Original,candidate,position_radius_m=.08,heading_radius_rad=math.radians(10.))
    checker=Checker();poses=[(.1,10.,0.),(.3,0.,0.)]
    assert checker(poses) is True and calls==[poses] # Endpoint association NEVER certifies the swept edge.
    from hospital_mission_planning import ResourcePoseSamples
    exhausted=ResourcePoseSamples([(.3,.1,0.)]);exhausted.resource_feasible=False
    assert checker(exhausted) is True and calls[-1] is exhausted


def test_intermediate_region_ast_is_explicit_and_cannot_certify_whole_mission():
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,lattice.propagate,
        lattice._state_key,MotionResourceLimits())
    local=pareto_resource_lattice_function(base,stage_acceptance=lambda state,whole:True)
    assert local.mission_search_binding['AST_intermediate_goal_substitutions']==1
    assert local.mission_search_binding['undo_reproduces_entire_original_AST']
    assert local.mission_search_binding['intermediate_region_NOT_mission_goal']
    with pytest.raises(ValueError,match='cannot certify'):plan_NE_mission_candidate(local,**_joint_NE_fixture_arguments())


@pytest.mark.parametrize('invalid',[dict(rigid_body_feasible=False),dict(mission_reverse_episode_bins=101),
    dict(mission_reverse_total_bins=201),dict(mission_direction_changes=5)])
def test_intermediate_region_cannot_bypass_invalid_incoming_physical_or_resource_state(invalid):
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,lattice.propagate,
        lattice._state_key,MotionResourceLimits());State=base.__globals__['State']
    incoming=replace(State(0.,0.,0.,'coordinated_counterphase',0.,0.,True),**invalid)
    bounds=mission_search_bounds([-1.,2.,-1.,1.],[0.,0.,0.],[1.,0.,0.])
    with pytest.raises(ValueError,match='Feasible complete incoming'):
        plan_NE_region_connection(base,incoming,bounds,(1.,0.,0.),FreeCheckerFixture(),object(),lambda s,g:0.,
            region=lambda s:True,node_limit=1,diagnostics={})


def test_actual_original_NE_region_retains_nonzero_speed_and_carries_every_state_field(compiled_NE_edge):
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,compiled_NE_edge,
        lattice._state_key,MotionResourceLimits());State=base.__globals__['State']
    from hospital_h3_turn_planning import model_bounds
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    bounds=mission_search_bounds([-1.,2.,-1.,1.],[0.,0.,0.],[1.,0.,0.])
    initial=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    checker=resource_checker_class(FreeCheckerFixture)()
    first=plan_NE_region_connection(base,initial,bounds,(1.,0.,0.),checker,model,
        lambda s,g:abs(.05-s.x)/.05,region=lambda s:s.x>=.049,node_limit=30,diagnostics={})
    result=first['segment'];assert result is not None and len(result.actions)==1
    state=result.states[-1];assert math.hypot(state.longitudinal_velocity,state.lateral_velocity)>.005
    second=plan_NE_region_connection(base,state,bounds,(1.,0.,0.),checker,model,
        lambda s,g:abs(.1-s.x)/.05,region=lambda s:s.x>=.099,node_limit=30,diagnostics={})
    assert second['complete_initial_state'] is state and second['segment'] is not None
    replay,_,_=base.__globals__['propagate'](state,*second['segment'].actions[0][:3],speed_m_s=.05,
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=second['segment'].actions[0][3])
    assert asdict(replay)==asdict(second['segment'].states[0])
    assert first['intermediate_segment_NOT_mission_goal_certificate']


def test_hierarchical_backtracking_discards_hypothesis_without_resetting_surviving_state(monkeypatch):
    import hospital_mission_planning as binding
    from dataclasses import replace
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,lattice.propagate,
        lattice._state_key,MotionResourceLimits());State=base.__globals__['State']
    initial=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    candidate,_=_owned_reference_fixture(points=[(.5,0.,0.),(1.,0.,0.),(2.,0.,0.)],goal=(2.,0.,0.))
    for action in candidate['selected_actions']:action['curvature_1_m']=0.
    bounds=mission_search_bounds([-1.,3.,-1.,1.],[0.,0.,0.],[2.,0.,0.]);now=[0.];calls=[];attempts=[0]
    def local(original,current,bounds,goal,checker,model,guide,**kw):
        now[0]+=.1;kw['diagnostics']['expanded_nodes']=1;calls.append(current)
        if current.x==0.:
            attempts[0]+=1
            end=replace(current,x=.5,front=.1*attempts[0],longitudinal_velocity=.02,
                direction=1,mission_reverse_total_bins=4)
            assert kw['region'](end)
        elif current.front==.1:return dict(segment=None)
        else:end=replace(current,x=1.,lateral_velocity=.001)
        return dict(segment=lattice.PlanResult((end,),(('coordinated_counterphase',0.,1,.05),),1,2.))
    def terminal(search,bounds,goal,checker,model,guide,**kw):
        now[0]+=.1;kw['diagnostics']['expanded_nodes']=1;parent=search.mission_test_parent
        end=replace(parent,x=2.,longitudinal_velocity=0.,lateral_velocity=0.)
        return lattice.PlanResult((end,),(('coordinated_counterphase',0.,1,.003),),1,3.)
    original_pareto=binding.pareto_resource_lattice_function
    def scoped(original,**kw):
        result=original_pareto(original,**kw)
        result.mission_test_parent=State(**original.mission_continuation['exact_initial_state'])
        return result
    monkeypatch.setattr(binding,'plan_NE_region_connection',local)
    monkeypatch.setattr(binding,'plan_NE_mission_candidate',terminal)
    monkeypatch.setattr(binding,'pareto_resource_lattice_function',scoped)
    result,log=hierarchical_NE_reference_connections(base,initial,bounds,(2.,0.,0.),candidate,
        lambda deadline:FreeCheckerFixture(),object(),lambda s,g:0.,deadline=100.,node_limit=20,clock=lambda:now[0])
    assert result is not None and [s.x for s in result.states]==[.5,1.,2.]
    assert result.states[0].front==.2 and result.states[1].mission_reverse_total_bins==4
    assert result.states[1].longitudinal_velocity==.02 and result.cost==7.
    assert log['expanded_nodes']==5 and calls[2] is initial
    assert log['no_pose_velocity_steering_force_resource_reset']


def test_steering_preparation_uses_original_joint_rate_and_future_arc_mean_without_state_assignment():
    from copy import deepcopy
    from patient_initial_stability import steering_lattice as lattice
    candidate,_=_owned_reference_fixture(points=[(.2,0.,0.),(.4,0.,0.),(.6,0.,0.)],goal=(.6,0.,0.))
    for a,k in zip(candidate['selected_actions'],(0.,.7,.7)):a['curvature_1_m']=k
    before=deepcopy(candidate);preview=ReferenceSteeringPreparation(candidate)
    target=lattice.steering_targets('coordinated_counterphase',.7)[0]
    assert preview.preview_distance==pytest.approx(.05*target/lattice.STEERING_RATE_RAD_S)
    now=preview.at(.2)
    assert now['front']==pytest.approx(target+lattice.STEERING_TRACKING_BIAS_RAD['front'])
    assert candidate==before and not preview.targets.flags.writeable
    assert preview.parameters['method_arrival_restriction_NOT_physical_tolerance_or_reachability_certificate']


def test_steering_preparation_stops_at_reference_reversal_and_does_not_average_across_it():
    candidate,_=_owned_reference_fixture(points=[(.2,0.,0.),(.4,0.,0.),(.2,0.,0.),(.6,0.,0.)],goal=(.6,0.,0.))
    for a,k,d in zip(candidate['selected_actions'],(0.,0.,-.7,.7),(1,1,-1,1)):
        a.update(curvature_1_m=k,direction=d)
    preview=ReferenceSteeringPreparation(candidate);at=preview.at(.39)
    assert at['preview_distance_used_m']==pytest.approx(.01)
    assert at['front']==pytest.approx(math.radians(-1.))
    with pytest.raises(ValueError):preview.at(-.01)
    with pytest.raises(ValueError):preview.at(math.nan)


def test_steering_ready_region_rejects_unprepared_joints_without_mutating_model_state(monkeypatch):
    import hospital_mission_planning as binding
    from dataclasses import replace,asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,lattice.propagate,
        lattice._state_key,MotionResourceLimits());State=base.__globals__['State']
    initial=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    candidate,_=_owned_reference_fixture(points=[(.5,0.,0.),(1.,0.,0.),(2.,0.,0.)],goal=(2.,0.,0.))
    for a in candidate['selected_actions']:a['curvature_1_m']=0.
    seen=[]
    def connection(original,current,bounds,goal,checker,model,guide,**kw):
        kw['diagnostics']['expanded_nodes']=1
        wrong=replace(current,x=current.x+.5,front=.5);snapshot=asdict(wrong)
        assert not kw['region'](wrong) and asdict(wrong)==snapshot
        ready=replace(wrong,front=math.radians(-1.),rear=math.radians(-1.))
        assert kw['region'](ready);seen.append(current)
        return dict(segment=lattice.PlanResult((ready,),(('coordinated_counterphase',0.,1,.05),),1,1.))
    def final(*a,**kw):raise RuntimeError('ORIGINAL_FINAL_CHECK_STILL_REQUIRED')
    monkeypatch.setattr(binding,'plan_NE_region_connection',connection)
    monkeypatch.setattr(binding,'plan_NE_mission_candidate',final)
    bounds=mission_search_bounds([-1.,3.,-1.,1.],[0.,0.,0.],[2.,0.,0.])
    with pytest.raises(RuntimeError,match='ORIGINAL_FINAL_CHECK_STILL_REQUIRED'):
        hierarchical_NE_reference_connections(base,initial,bounds,(2.,0.,0.),candidate,lambda deadline:FreeCheckerFixture(),
            object(),lambda s,g:0.,deadline=10.,node_limit=10,clock=lambda:0.,steering_ready=True)
    assert len(seen)==2 and seen[0] is initial


def test_rate_aware_reference_is_original_finite_joint_geometry_not_a_full_NE_certificate(monkeypatch):
    import hospital_mission_planning as binding
    from patient_initial_stability import steering_lattice as lattice
    marker=object();calls=[]
    def original(*args,**kwargs):calls.append((args,kwargs));return marker
    monkeypatch.setattr(binding,'pareto_resource_lattice_function',lambda f,**kw:f)
    bounds=mission_search_bounds([-1.,2.,-1.,1.],[0.,0.,0.],[1.,0.,0.])
    out=rate_aware_spatial_reference(original,bounds,(1.,0.,0.),FreeCheckerFixture(),lambda s,g:0.,node_limit=123,diagnostics={})
    args,kw=calls[0]
    assert out['spatial_result'] is marker and out['reference_ONLY_NOT_NE_goal_or_physical_admission']
    assert out['rigid_body_forces_inertia_terminal_speed_NOT_evaluated']
    assert kw['dynamics'] is lattice.FULL_DYNAMICS and kw['rigid_body_model'] is None
    assert kw['propagate_initial_steering'] and kw['initial_front_rad']==kw['initial_rear_rad']==0.
    assert kw['allowed_modes']==lattice.ALL_MODES and kw['allowed_curvatures']==lattice.CURVATURES
    assert kw['directions']==(1,-1) and kw['node_limit']==123
    assert 'terminal_speed_tolerance_m_s' not in kw and args[0]==(1.,0.,0.)


def test_actual_rate_aware_spatial_reference_replays_original_joint_state_without_force_admission():
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,lattice.propagate,
        lattice._state_key,MotionResourceLimits());State=base.__globals__['State']
    bounds=mission_search_bounds([-1.,1.,-1.,1.],[0.,0.,0.],[.3,0.,0.])
    answer=rate_aware_spatial_reference(base,bounds,(.3,0.,0.),resource_checker_class(FreeCheckerFixture)(),
        lambda s,g:math.hypot(s.x-g[0],s.y-g[1])/.05,node_limit=200,diagnostics={})
    result=answer['spatial_result'];assert result is not None
    state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    for action,saved in zip(result.actions,result.states):
        new,poses,elapsed=base.__globals__['propagate'](state,*action,speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS)
        assert asdict(new)==asdict(saved) and poses.resource_feasible
        assert max(abs(new.front-state.front),abs(new.rear-state.rear))<=lattice.STEERING_RATE_RAD_S*elapsed+1e-12
        state=new
    assert answer['reference_ONLY_NOT_NE_goal_or_physical_admission']


@pytest.mark.parametrize('step',[0.,-.025,.01,.1,True,25,float('nan'),float('inf')])
def test_rolling_resolution_requires_an_explicit_bounded_diagnostic_distance(step):
    from patient_initial_stability import steering_lattice as lattice
    with pytest.raises(ValueError,match='diagnostic25/50mm'):
        diagnostic_rolling_resolution(lattice.plan,step_m=step)


def test_rolling_resolution_keeps_original_code_globals_keys_and_explicit_replay_distance():
    from patient_initial_stability import steering_lattice as lattice
    before=dict(lattice.plan.__globals__)
    run=diagnostic_rolling_resolution(lattice.plan,step_m=.025)
    assert run.__code__ is lattice.plan.__code__ and lattice.plan.__globals__==before
    assert run.__globals__['_state_key'] is lattice._state_key
    assert not run.mission_rolling_resolution['stationary_reconfiguration_added']
    initial=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    scoped=run.__globals__['propagate']
    assert scoped(initial,'front_only',.5,1)==lattice.propagate(initial,'front_only',.5,1,step_m=.025)
    with pytest.raises(ValueError,match='cannot change'):
        scoped(initial,'front_only',.5,1,step_m=.05)
    with pytest.raises(ValueError):diagnostic_rolling_resolution(run,step_m=.025)


def test_rolling_resolution_metadata_survives_private_wrappers_and_blocks_original_whole_serializer():
    from patient_initial_stability import steering_lattice as lattice
    run=diagnostic_rolling_resolution(lattice.plan,step_m=.025)
    incoming=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
    continued=continue_original_lattice_from_state(run,incoming)
    wrapped=pareto_resource_lattice_function(continued)
    assert continued.mission_rolling_resolution==wrapped.mission_rolling_resolution==run.mission_rolling_resolution
    with pytest.raises(ValueError,match='action serializer'):
        plan_NE_mission_candidate(wrapped,**_joint_NE_fixture_arguments())


@pytest.mark.parametrize('distance',[.025,.05])
def test_actual_short_rolling_NE_connection_retains_state_resources_and_exact_original_edge(distance):
    from dataclasses import asdict
    from patient_initial_stability import steering_lattice as lattice
    from patient_initial_stability.hierarchical_dynamics_planner import refine_speed_consistent_lattice_route
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from hospital_h3_turn_planning import model_bounds
    base,_=bind_resource_kernels(lattice.plan,refine_speed_consistent_lattice_route,lattice.State,lattice.propagate,
        lattice._state_key,MotionResourceLimits());State=base.__globals__['State']
    run=diagnostic_rolling_resolution(base,step_m=distance)
    incoming=State(0.,0.,0.,'coordinated_counterphase',.1,-.1,True,1,.02,.001,.002,
        mission_reverse_total_bins=7,mission_direction_changes=2)
    before=asdict(incoming);model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    bounds=mission_search_bounds([-1.,2.,-1.,1.],[0.,0.,0.],[1.,0.,0.])
    out=plan_NE_region_connection(run,incoming,bounds,(1.,0.,0.),resource_checker_class(FreeCheckerFixture)(),
        model,lambda state,goal:abs(distance-state.x)/.05,region=lambda state:state.x>=distance-.005,
        node_limit=15,diagnostics={})
    result=out['segment'];assert result is not None and len(result.actions)==1
    assert out['primitive_step_m']==distance and out['complete_initial_state'] is incoming and asdict(incoming)==before
    action=result.actions[0]
    replay,poses,elapsed=base.__globals__['propagate'](incoming,*action[:3],step_m=distance,speed_m_s=.05,
        dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=action[3])
    assert asdict(replay)==asdict(result.states[0]) and poses.resource_feasible and elapsed>0.
    assert replay.mission_reverse_total_bins==7 and replay.mission_direction_changes==2
    assert math.hypot(replay.longitudinal_velocity,replay.lateral_velocity)>.005


