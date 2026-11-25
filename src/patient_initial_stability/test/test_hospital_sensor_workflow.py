"""Software boundary tests only; these do not assert a physical PASS."""
import copy
import gzip
import json
from pathlib import Path
import pytest
import hospital_sensor_workflow as workflow
import hospital_sensor_transport as transport


def response_fixture():
    return dict(stamp_ns=100,prehold_ZERO_only_before_Nav2=False,
        actual_plugin_ack_source_stamp_ns=100,
        proposal=dict(stamp_ns=100,proposal_only_NO_SDK_actuation_or_physical_admission=True,
            proposed_body_twist_u_v_r=[.03,.01,.02],
            proposed_joint_targets=dict(steering_targets_rad=[.1,.2],wheel_velocity_targets_rad_s=[.4,.5])),
        actual_nav2_command=dict(linear_m_s=[.03,.01,0.],angular_rad_s=[0.,0.,.02]))


def test_actual_Nav2_gate_accepts_only_matching_proposal_and_uniform_speed_reduction():
    r=response_fixture();targets,scale=workflow.validate_actual_command(r,dict(stamp_ns=100))
    assert scale==1. and targets==r['proposal']['proposed_joint_targets']
    r['actual_nav2_command']=dict(linear_m_s=[.015,.005,0.],angular_rad_s=[0.,0.,.01])
    assert workflow.validate_actual_command(r,dict(stamp_ns=100))[1]==.5


@pytest.mark.parametrize('fault',['stamp','ack','amplify','different_curvature','nonplanar','NaN','no_actual_output'])
def test_native_write_boundary_rejects_unmatched_or_modified_Nav2_output(fault):
    r=response_fixture()
    if fault=='stamp':r['stamp_ns']=99
    elif fault=='ack':r['actual_plugin_ack_source_stamp_ns']=99
    elif fault=='amplify':r['actual_nav2_command']['linear_m_s'][0]=.06
    elif fault=='different_curvature':r['actual_nav2_command']['angular_rad_s'][2]=.01
    elif fault=='nonplanar':r['actual_nav2_command']['linear_m_s'][2]=.001
    elif fault=='NaN':r['actual_nav2_command']['linear_m_s'][0]=float('nan')
    else:r['actual_nav2_command']=None
    with pytest.raises(ValueError):workflow.validate_actual_command(r,dict(stamp_ns=100))


def test_prehold_cannot_supply_direct_forward_fallback():
    r=response_fixture();r['prehold_ZERO_only_before_Nav2']=True;r['actual_nav2_command']=None
    with pytest.raises(ValueError):workflow.validate_actual_command(r,dict(stamp_ns=100))
    r['proposal']['proposed_body_twist_u_v_r']=[0.,0.,0.]
    assert workflow.validate_actual_command(r,dict(stamp_ns=100))[1]==1.


def test_private_authorization_binding_keeps_physical_code_and_does_not_modify_original_globals():
    original=workflow.lc.fresh_held_preparation_operations
    bound=workflow._authorization_binding(original)
    assert bound.__code__ is original.__code__
    assert original.__globals__['checked_fresh_held_approval'] is not workflow.checked_approval
    assert bound.__globals__['checked_fresh_held_approval'] is workflow.checked_approval
    with pytest.raises(ValueError):workflow.checked_approval(dict(question_item_id='invented'))


def test_followup_revalidation_uses_actual_user_reply_and_an_explicit_optin_sensor_policy():
    approved=workflow.checked_approval(workflow.REVALIDATION_APPROVAL)
    assert approved['answer_ko']=='수정 검증 후 H5 재검증 1회 진행'
    assert approved!=workflow.APPROVAL
    assert 'call_230dd05850ca41c6a0235c8b53f7d456' in approved['question_item_id']
    assert workflow.SURFACE_POLICY['endpoint_to_source_spatial_tolerance_m']==.0001
    assert workflow.SURFACE_POLICY['original_native_range_NOT_changed']
    assert workflow.SURFACE_POLICY['along_ray_range_error_NOT_claimed_under100um']
    bridge_fix=workflow.checked_approval(workflow.BRIDGE_FIX_APPROVAL)
    assert bridge_fix['answer_ko']=='수정 검증 후 H5 추가 1회 승인'
    assert 'call_67ff2607960b407db49762f9f5c252b2' in bridge_fix['question_item_id']
    assert bridge_fix!=approved


def test_strict15mm_and_conditional_H6_authority_are_actual_latest_replies_not_old_retry_consent():
    approved=workflow.checked_approval(workflow.STRICT_CAPTURE_APPROVAL)
    assert 'call_8162fcaf2aee49c1aa85ec013828324e' in approved['question_item_id']
    assert approved['answer_ko']=='검증 후 강화 설정 H5 1회·통과 시 H6 진행'
    h6=workflow.checked_approval(workflow.H6_STRICT_APPROVAL)
    assert h6['capture_approval']==approved
    assert 'call_10e7e20f9340442dad04ee6df19c5ba7' in h6['wall_budget_approval']['question_item_id']
    assert h6['wall_budget_approval']['sensor_orchestration_wall_s']==3000.
    assert 'ONLY_AFTER_OWN_H5_SENSOR_PASS' in h6['scope']
    assert approved!=workflow.BRIDGE_FIX_APPROVAL
    guard=workflow.checked_approval(workflow.H6_GUARD_REVALIDATION_APPROVAL)
    assert 'call_5f134e94653f4fa2ae637af9396e1f6c' in guard['question_item_id']
    assert guard['answer_ko']=='수정 검증 후 H6 추가 1회 승인'
    assert guard['inherited_strict_capture_and_wall_approval']==h6
    cell=workflow.checked_approval(workflow.H6_CELL_GUARD_APPROVAL)
    assert cell['approval_source']=='USER_MESSAGE'
    assert cell['user_message']=='바로가 pass 뜰 때 까지 멈추지말고'
    assert cell['inherited_strict_capture_and_wall_approval']==h6
    assert cell!=guard and 'ONE_FRESH_STATIC_CONDITIONAL_H6_ROUTE' in cell['approved_proposal']


def test_receipt_archive_replays_exact_bytes_and_is_exclusive(tmp_path):
    p=tmp_path/'receipts.jsonl.gz';archive=transport.BoundedJSONLArchive(p,maximum_records=3)
    values=[dict(stamp_ns=10,actual_ack=False),dict(stamp_ns=20,actual_ack=None)]
    for value in values:archive(value)
    receipt=archive.finish()
    assert receipt['records']==2 and receipt['exact_original_records_durably_verified']
    with gzip.open(p,'rt') as stream:assert [json.loads(line) for line in stream]==values
    with pytest.raises(FileExistsError):transport.BoundedJSONLArchive(p,maximum_records=3)
    with pytest.raises(ValueError):archive.finish()


def test_known_start_prior_is_declared_task_pose_not_current_GT_and_short_prior_stays_unchanged():
    import math
    from hospital_sensor_ros_bridge import initial_prior_covariance
    from test_hospital_sensor_platform import definition_fixture
    definition,_=definition_fixture();plan=json.loads(Path(definition['frozen_plan']).read_text())
    config=dict(mission_platform_definition=definition,initial_prior_xyyaw=plan['start_world_xyyaw'],
        known_fixed_start_local_prior=dict(position_std_m=.003,yaw_std_deg=.1))
    assert initial_prior_covariance(config)[0]==.003**2
    config['initial_prior_xyyaw']=list(config['initial_prior_xyyaw']);config['initial_prior_xyyaw'][0]+=.01
    with pytest.raises(ValueError):initial_prior_covariance(config)
    assert initial_prior_covariance({})[0]==.15**2
    assert initial_prior_covariance({})[35]==math.radians(10.)**2


def test_H6_ROS_goal_array_is_all_double_without_changing_the_declared_zero_yaw_goal(tmp_path):
    from hospital_sensor_ros_bridge import platform_nav2_parameters
    from test_hospital_sensor_platform import definition_fixture,BASE
    import hashlib
    import hospital_sensor_platform as platform
    definition,_=definition_fixture()
    root=BASE/'hospital_h6_experiments/cohort_20_held_static_r01/route/P_dynamics_full_envelope'
    from ijcas_paths import load_archived_json
    route=load_archived_json(root/'route_protocol.json')
    definition.update(scenario='H6_integrated_transfer_mission',simulation_budget_s=600.,maximum_wall_time_s=3000.,
        sensor_wall_budget_review=copy.deepcopy(platform.H6_SENSOR_WALL_REVIEW))
    for key,path in dict(controller_definition=root/'route_protocol.json',frozen_plan=root/'frozen_plan.json',geometry=Path(route['geometry_file'])).items():
        definition[key]=str(path);definition['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    params=platform_nav2_parameters('/ijcas_sensor_dev/software_H6',definition)
    assert params['/ijcas_sensor_dev/software_H6/controller_server']['ros__parameters']['FollowPath.footprint_guard_version']=='CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1'
    goal=params['/ijcas_sensor_dev/software_H6/controller_server']['ros__parameters']['goal_checker.declared_task_goal_xyyaw']
    original=json.loads(Path(definition['frozen_plan']).read_text())['goal_world_xyyaw']
    assert goal==original and all(type(x) is float for x in goal) and goal[2]==0.
    utilities=pytest.importorskip('rclpy.utilities')
    file=tmp_path/'complete_H6_parameters.json'
    file.write_text(json.dumps(params),encoding='utf-8')
    assert utilities.remove_ros_args(args=['controller_server','--ros-args','--params-file',str(file)])==['controller_server']
    goal[2]=0  # Reproduce the ACTUAL heterogeneous-array startup defect.
    bad=tmp_path/'original_mixed_H6_parameters.json'
    bad.write_text(json.dumps(params),encoding='utf-8')
    with pytest.raises(Exception,match='same type|integer|parse params'):
        utilities.remove_ros_args(args=['controller_server','--ros-args','--params-file',str(bad)])


def test_actual_locked_costmap_rejection_parser_retains_all_cells_and_never_relabels_success():
    from hospital_sensor_ros_bridge import costmap_guard_log_record
    value=dict(version='ACTUAL_LOCKED_NAV2_COSTMAP_REJECTION_v1',proposal_stamp_ns=1000,pose_stamp_ns=990,
        frame_id='ijcas_sensor_dev_odom',projection_step=3,cost=254.,projected_xyyaw=[1.,2.,.1],
        origin_xy_m=[0.,0.],resolution_m=.02,shape_rows_columns=[2,3],
        padded_footprint_xy_m=[[.825,.4925],[.825,-.4925],[-.825,-.4925],[-.825,.4925]],
        complete_grid_snapshot=True,cost_data=[0,253,254,255,0,0])
    marker='IJCAS_COSTMAP_GUARD_DIAGNOSTIC '
    text='[ERROR] [owned controller]: '+marker+json.dumps(value)+'\x1b[0m\n'
    assert costmap_guard_log_record(text)==value
    geometry=copy.deepcopy(value)
    geometry.update(geometry_guard_version='CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1',rejected_cell_column_row=[2,0])
    assert costmap_guard_log_record(marker+json.dumps(geometry))==geometry
    bad_cell=copy.deepcopy(geometry);bad_cell['rejected_cell_column_row']=[3,0]
    with pytest.raises(ValueError):costmap_guard_log_record(marker+json.dumps(bad_cell))
    bad_cell['rejected_cell_column_row']=[-1,0]
    with pytest.raises(ValueError):costmap_guard_log_record(marker+json.dumps(bad_cell))
    assert costmap_guard_log_record('old generic rejection WITHOUT raster') is None
    with pytest.raises(ValueError):costmap_guard_log_record(text+text)
    for mutate in ('drop','byte','step','snapshot'):
        bad=copy.deepcopy(value)
        if mutate=='drop':bad['cost_data'].pop()
        if mutate=='byte':bad['cost_data'][0]=256
        if mutate=='step':bad['projection_step']=6
        if mutate=='snapshot':bad['complete_grid_snapshot']=False
        with pytest.raises(ValueError):costmap_guard_log_record(marker+json.dumps(bad))


def test_actual_ROS_encoder_header_assignments_cannot_mutate_retained_TF():
    import ast
    import inspect
    import hospital_sensor_ros_bridge as ros
    geometry=pytest.importorskip('geometry_msgs.msg')
    sensor=pytest.importorskip('sensor_msgs.msg')
    nav=pytest.importorskip('nav_msgs.msg')
    tree=ast.parse(inspect.getsource(ros.run_bridge))
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='publish_encoder')
    expressions={}
    for node in ast.walk(function):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target=ast.unparse(node.targets[0])
            if target in ('odom.header','joints.header'):expressions[target]=node.value
    assert set(expressions)=={'odom.header','joints.header'}
    tf=geometry.TransformStamped();tf.header.frame_id='measured_odom';tf.child_frame_id='base_link'
    odom=nav.Odometry();joints=sensor.JointState()
    for target,expression in expressions.items():
        assert ast.unparse(expression)=='copy.deepcopy(tf.header)'
        value=eval(compile(ast.Expression(expression),'<actual_header_assignment>','eval'),dict(copy=copy,tf=tf))
        setattr(odom if target.startswith('odom') else joints,'header',value)
    joints.header.frame_id='base_link'
    assert tf.header.frame_id=='measured_odom' and odom.header.frame_id=='measured_odom'
    assert tf.header is not odom.header and tf.header is not joints.header
