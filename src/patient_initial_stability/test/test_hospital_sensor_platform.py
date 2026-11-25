"""Pure proposal computations, NOT a physical or Nav2 integration PASS."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import hospital_sensor_platform as platform
import hospital_sensor_messages as messages
from ijcas_paths import load_archived_json
from test_hospital_sensor_runtime import encoder_inputs

BASE=Path(__file__).resolve().parents[3]/'studies/IJCAS/results/hospital_paper_gate'


def definition_fixture():
    sample,settings=encoder_inputs()
    root=BASE/'hospital_h5_experiments/cohort_10_held_static_r01/route/P_dynamics_full_envelope'
    route=root/'route_protocol.json'
    r=load_archived_json(route)
    files=dict(controller_definition=route,frozen_plan=root/'frozen_plan.json',geometry=Path(r['geometry_file']),
        encoder_geometry=BASE/'hospital_sensor_navigation/encoder_diagnostic/retrospective_h5_r01/source_geometry_GET.json',
        encoder_wrap_contract=BASE/'hospital_sensor_navigation/encoder_diagnostic/wrapped_positions_h5_r01/source_wrap_and_cap_contract.json')
    value=dict(version=platform.VERSION,scenario='H5_obstacle_bypass_return',method='P_dynamics_full_envelope',
        sensor_speed_window_s=.2,control_minimum_period_s=.05,maximum_control_gap_s=.25,simulation_budget_s=240.,
        maximum_wall_time_s=900.,maximum_native_route_trials=1,automatic_retry=False,
        ground_truth_control_pose_or_velocity_inputs=False,actor_root=sample['actor_root'],
        native_joint_names=sample['joint_names'],original_native_position_targets_rad=[0.]*12,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
        **{k:str(v) for k,v in files.items()})
    return value,sample


def test_H6_sensor3000s_is_separate_from_original1500s_and_exact_active_math_is_reviewed():
    value,_=definition_fixture()
    root=BASE/'hospital_h6_experiments/cohort_20_held_static_r01/route/P_dynamics_full_envelope'
    route=load_archived_json(root/'route_protocol.json')
    value.update(scenario='H6_integrated_transfer_mission',simulation_budget_s=600.,maximum_wall_time_s=3000.,
        sensor_wall_budget_review=copy.deepcopy(platform.H6_SENSOR_WALL_REVIEW))
    for key,path in dict(controller_definition=root/'route_protocol.json',frozen_plan=root/'frozen_plan.json',
            geometry=Path(route['geometry_file'])).items():
        value[key]=str(path);value['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert platform.checked_definition(value)['scenario']=='H6_integrated_transfer_mission'
    original_bytes=(root/'route_protocol.json').read_bytes()
    controller=platform.MeasuredPlatformController(value)
    review=controller.controller_source_review
    assert review['exact_active_generated_tracking_code_and_binding']
    assert review['all_other_module_AST_exact'] and review['inactive_H5_branch_removed_recovers_original_AST']
    assert review['original_packet_and_result_NOT_modified']
    assert (root/'route_protocol.json').read_bytes()==original_bytes
    assert route['maximum_wall_time_s']==1500.
    limits=platform.task_limits(value['scenario'])
    assert limits['maximum_sensor_frames']==3010 and limits['maximum_control_records']==12010
    value['simulation_budget_s']=240.
    with pytest.raises(ValueError):platform.checked_definition(value)


def test_strict15mm_capture_changes_ONLY_sensor_instances_and_future_terminal_factory():
    definition,_=definition_fixture()
    legacy=platform.MeasuredPlatformController(definition)
    strict_definition=copy.deepcopy(definition)
    strict_definition['sensor_stationary_capture_policy']=copy.deepcopy(platform.SENSOR_CAPTURE_POLICY)
    strict=platform.MeasuredPlatformController(strict_definition)
    baseline=legacy.controller.motion.tracker.config
    tracker=strict.controller.motion.tracker
    assert baseline['stationary_hold_entry_position_m']==.030
    assert tracker.config==dict(baseline,stationary_hold_entry_position_m=.015)
    view=strict.controller.motion.leg['view']
    future=strict.controller.motion.raw(view,view.poses[-1])
    assert future.config==tracker.config
    assert platform.MeasuredPlatformController(definition).controller.motion.tracker.config==baseline
    for candidate in (legacy.controller.motion.tracker,tracker,future):
        candidate.terminal_stage='stationary_settle';candidate.previous_time=1.;candidate.stage_started=0.
        goal=candidate.goal
        assert candidate._inside_capture([goal[0]+.014,goal[1],goal[2]])
        assert candidate._inside_capture([goal[0]+.024,goal[1],goal[2]]) is (candidate is legacy.controller.motion.tracker)


def test_strict_policy_is_explicit_and_never_changes_actual30mm_scoring_or_goal():
    value,_=definition_fixture();value['sensor_stationary_capture_policy']=dict(platform.SENSOR_CAPTURE_POLICY)
    assert platform.checked_definition(value)['sensor_stationary_capture_policy']['original_world_scoring_position_m']==.030
    for key,change in [('stationary_hold_entry_position_m',.020),('goal_or_estimated_pose_shifted',True)]:
        bad=copy.deepcopy(value);bad['sensor_stationary_capture_policy'][key]=change
        with pytest.raises(ValueError):platform.checked_definition(bad)


def test_H5_bounds_are_unchanged_and_no_other_scenarios_silently_admitted():
    assert platform.task_limits('H5_obstacle_bypass_return')==dict(simulation_budget_s=240.,
        maximum_wall_time_s=900.,maximum_sensor_frames=1210,maximum_control_records=4810)
    with pytest.raises(ValueError):platform.task_limits('H1_bedside_departure')


def prepared_fixture():
    d,s=definition_fixture();controller=platform.MeasuredPlatformController(d)
    controller.observe_encoder(s)
    controller.accept_amcl_fix(dict(stamp=messages.stamp_parts(s['stamp_ns']),
        xyyaw=controller.plan['start_world_xyyaw'],covariance=[0.]*36))  # SOFTWARE ONLY.
    return controller,s


def test_original_P_controller_generates_zero_prehold_request_but_NEVER_invents_native_ACK():
    controller,s=prepared_fixture()
    result=controller.propose(s,relative_physics_time_s=0.,actual_native_hold_ack=False)
    assert result['requested_hold'] is True
    assert result['proposed_body_twist_u_v_r']==[0.,0.,0.]
    assert result['proposal_only_NO_SDK_actuation_or_physical_admission']
    assert controller.endpoint.confirmation(0.) is False  # Request is NOT confirmation.
    assert result['source_original_bridge_input_ack'] is False
    assert controller.controller.geometry_profile==controller.plan['geometry_profile']


def test_actual_pending_hold_ACK_stays_pending_without_new_platform_write_or_fake_confirmation():
    controller,s=prepared_fixture();controller.propose(s,relative_physics_time_s=0.,actual_native_hold_ack=False)
    s=copy.deepcopy(s);s['stamp_ns']+=50000000
    result=controller.propose(s,relative_physics_time_s=.05,actual_native_hold_ack=None)
    assert result['actuator_write_required'] is False
    assert result['source_original_bridge_input_ack'] is None
    assert controller.endpoint.confirmation(.05) is None


def test_original_AMCL_cancellation_residual_is_preserved_not_projected_or_used_as_a_confidence_bound():
    import numpy as np
    covariance=[0.]*36
    covariance[0]=-1.3102408047416247e-10;covariance[1]=covariance[6]=6.411937647499144e-11
    covariance[7]=-1.3665157894138247e-10;covariance[35]=-3.3528735343676915e-13
    fix=dict(xyyaw=[18.119397186745847,28.055191856391836,-1.5696879162528472],covariance=covariance)
    original=copy.deepcopy(fix);result=platform.review_amcl_covariance(fix)
    assert result['tiny_negative_roundoff_present'] and not result['raw_covariance_modified']
    assert fix==original and result['covariance_NOT_used_as_calibrated_control_or_safety_bound']
    fix['covariance'][0]=-.001
    with pytest.raises(ValueError):platform.review_amcl_covariance(fix)


@pytest.mark.parametrize('fault',['changed_same_stamp','GT_field','stamp_regression','gap','stale_fix','source_flag'])
def test_measured_platform_rejects_ambiguous_or_GT_or_stale_input_without_recovery(fault):
    controller,s=prepared_fixture();controller.propose(s,relative_physics_time_s=0.,actual_native_hold_ack=False)
    s=copy.deepcopy(s)
    if fault=='changed_same_stamp':s['positions_rad'][0]=.001
    elif fault=='GT_field':s['ground_truth_pose']=[999.,999.,0.]
    elif fault=='stamp_regression':s['stamp_ns']-=1
    elif fault=='gap':s['stamp_ns']+=251000000
    elif fault=='source_flag':s['commands_modes_or_ground_truth_used_as_encoder_inputs']=True
    else:
        for i in range(1,6):
            s['stamp_ns']=1000000000+i*200000000;controller.observe_encoder(s)
        s['stamp_ns']+=50000000
    with pytest.raises((ValueError,RuntimeError)):
        controller.propose(s,relative_physics_time_s=(s['stamp_ns']-1000000000)/1e9,actual_native_hold_ack=False)
    assert controller.failed
