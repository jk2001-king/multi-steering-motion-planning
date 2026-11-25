"""Shared fresh-source sensor/Nav2 workflow; original SDK/control stays intact.

No old questionnaire answer is reused as new motion consent. Private function
bindings extend ONLY the authorization record accepted by the original fresh
preparation and held-static constructor; their physical code/guards are the
same code objects. No global SDK/module monkeypatch, actor restore or retry.
"""

import copy
import hashlib
import math
import time
from pathlib import Path
from types import FunctionType

import numpy as np
import hospital_runtime_evidence_io as evidence
import hospital_sensor_runtime as native
import hospital_mission_sequence as sequence
import hospital_sensor_platform as platform_math

lc=sequence.lifecycle
BASE=sequence.retained.GATE/'hospital_sensor_navigation'
APPROVAL=dict(approval_source='USER_MESSAGE',user_message='바로 진행해 완료 될때까ㅈ;',
    prior_user_request='음.. 일단 그러면 빠르게 결과 뽑을 수 있도록 바로 가보자 너 혼자 쫘라락 검증도 하면서 테스트 할 수 있자나 이제',
    scope='ONE_FRESH_H5_STATIC_THEN_CONDITIONAL_FROZEN_P_SENSOR_AMCL_NAV2_WHOLE_ROUTE_NO_AUTOMATIC_PHYSICAL_RETRY')
REVALIDATION_APPROVAL=dict(approval_source='USER_QUESTION_REPLY',
    question_item_id='["request_user_input_async","call_230dd05850ca41c6a0235c8b53f7d456",0]',
    answer_ko='수정 검증 후 H5 재검증 1회 진행',
    scope='PRESERVE_HOLD_ONE_Stop_NEW_H5_STATIC_THEN_ONLY_IF_PASS_ONE_P_SENSOR_AMCL_NAV2_REVALIDATION_NO_AUTOMATIC_PHYSICAL_RETRY')
BRIDGE_FIX_APPROVAL=dict(approval_source='USER_QUESTION_REPLY',
    question_item_id='["request_user_input_async","call_67ff2607960b407db49762f9f5c252b2",0]',
    answer_ko='수정 검증 후 H5 추가 1회 승인',
    scope='PRESERVE_HOLD_ONE_Stop_NEW_H5_STATIC_THEN_ONLY_IF_PASS_ONE_P_SENSOR_AMCL_NAV2_BRIDGE_FIX_REVALIDATION_NO_AUTOMATIC_PHYSICAL_RETRY')
H6_COMPLETION_APPROVAL=dict(approval_source='USER_MESSAGE',
    user_message='이거 완료될때까지 멈추지말고 쭉 진행해',
    explicitly_retained_cases=['H5','H6'],
    scope='ONE_FRESH_H6_STATIC_THEN_ONLY_IF_PASS_ONE_FROZEN_P_SENSOR_AMCL_NAV2_WHOLE_ROUTE_NO_AUTOMATIC_PHYSICAL_RETRY')
STRICT_CAPTURE_APPROVAL=dict(approval_source='USER_QUESTION_REPLY',
    question_item_id='["request_user_input_async","call_8162fcaf2aee49c1aa85ec013828324e",0]',
    answer_ko='검증 후 강화 설정 H5 1회·통과 시 H6 진행',
    scope='PRESERVE_HOLD_ONE_Stop_NEW_H5_STATIC_THEN_ONLY_IF_PASS_ONE_STRICT15MM_SENSOR_P_ROUTE_NO_AUTOMATIC_PHYSICAL_RETRY')
H6_STRICT_APPROVAL=dict(approval_source='USER_QUESTION_REPLIES',
    capture_approval=copy.deepcopy(STRICT_CAPTURE_APPROVAL),
    wall_budget_approval=copy.deepcopy(platform_math.H6_SENSOR_WALL_REVIEW),
    scope='ONLY_AFTER_OWN_H5_SENSOR_PASS_ONE_Stop_NEW_H6_STATIC_THEN_ONLY_IF_PASS_ONE_STRICT15MM_SENSOR_P_ROUTE_3000S_NO_AUTOMATIC_PHYSICAL_RETRY')
H6_GUARD_REVALIDATION_APPROVAL=dict(approval_source='USER_QUESTION_REPLY',
    question_item_id='["request_user_input_async","call_5f134e94653f4fa2ae637af9396e1f6c",0]',
    answer_ko='수정 검증 후 H6 추가 1회 승인',
    scope='PRESERVE_H6_HOLD_ONE_Stop_NEW_H6_STATIC_THEN_ONLY_IF_PASS_ONE_SAME_PATH_SENSOR_GUARD_REVALIDATION_NO_AUTOMATIC_PHYSICAL_RETRY',
    inherited_strict_capture_and_wall_approval=copy.deepcopy(H6_STRICT_APPROVAL))
H6_CELL_GUARD_APPROVAL=dict(approval_source='USER_MESSAGE',
    user_message='바로가 pass 뜰 때 까지 멈추지말고',
    approved_proposal='SAME_PADDED_FOOTPRINT_AND_ORIGINAL_CELLS_FULL_POLYGON_INTERSECTION_REVIEW_THEN_ONE_FRESH_STATIC_CONDITIONAL_H6_ROUTE',
    scope='PRESERVE_H6_HOLD_ONE_Stop_NEW_H6_STATIC_THEN_ONLY_IF_PASS_ONE_CLOSED_CELL_GUARD_SENSOR_ROUTE_NO_BLIND_PHYSICAL_RETRY',
    inherited_strict_capture_and_wall_approval=copy.deepcopy(H6_STRICT_APPROVAL))
SURFACE_POLICY=dict(version='EXPLICIT_FINITE_SOURCE_TRIANGLE_ENDPOINT100UM_v1',
    first_external_return_only=True,endpoint_surface_consistency=True,
    endpoint_to_source_spatial_tolerance_m=.0001,original_native_range_NOT_changed=True,
    along_ray_range_error_NOT_claimed_under100um=True,original_full_body_and_goal_stop_limits_unchanged=True)


def checked_approval(value):
    if value not in (APPROVAL,REVALIDATION_APPROVAL,BRIDGE_FIX_APPROVAL,H6_COMPLETION_APPROVAL,
            STRICT_CAPTURE_APPROVAL,H6_STRICT_APPROVAL,H6_GUARD_REVALIDATION_APPROVAL,H6_CELL_GUARD_APPROVAL):raise ValueError('Exact current user sensor/Nav2 workflow authority required')
    return copy.deepcopy(value)


def _authorization_binding(function):
    result=FunctionType(function.__code__,dict(function.__globals__,checked_fresh_held_approval=checked_approval),
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    if result.__code__ is not function.__code__:raise ValueError('Original physical code object must be retained')
    return result


class SensorPreparedHeldStaticHost(lc.PreparedHeldStaticSDKHost):
    __init__=_authorization_binding(lc.PreparedHeldStaticSDKHost.__init__)


async def capture_current_handoff(route_path,reference_folder,source_observation,output,*,authorized=False):
    """GET-only complete current boundary; no zero observer or motion call."""
    if authorized is not True:raise ValueError('Explicit current-boundary GET required')
    output=Path(output).resolve();source_observation=Path(source_observation).resolve()
    original=evidence.json_value(sequence.retained.read(source_observation))
    if original.get('verdict')!='PASS' or original.get('short_sensor_Nav2_interface_PASS') is not True:
        raise ValueError('Own completed and restored actual short-interface PASS required')
    port,read,_,_=native.bind_retained_zero_sensor_host(route_path,reference_folder,output,
        authorization=dict(APPROVAL,subscope='GET_ONLY_COMPLETE_CURRENT_SENSOR_PASS_BOUNDARY_NO_TRIAL'),
        prior_observation=source_observation,use_common_hold=True,authorized=True)
    before=read();checked=await port.preflight();after=read()
    snapshot=sequence.retained.shared.preparation.old.tensor_snapshot(port.tensor)
    if (before!=after or snapshot!=original['postflight']['actual_post_observation_tensor_snapshot']
            or port.adapter.dirty or port.physics_sub is not None or port.contact_sub is not None):
        raise ValueError('Complete unchanged PAUSED source before fresh archive required')
    # The binding creates a raw evidence sink; close the UNUSED sink with its
    # truthful zero-frame count. This GET does not acquire a scan or trial.
    unused=port.frame_archive.finish()
    pins={str(source_observation):sequence.retained.runtime.task.sha(source_observation),
        str(Path(original['protocol'])):sequence.retained.runtime.task.sha(original['protocol']),
        str(Path(original['protocol']).parent/'executing_sources.zip'):sequence.retained.runtime.task.sha(Path(original['protocol']).parent/'executing_sources.zip')}
    for file in (*lc.host_source_files(),Path(__file__).resolve()):pins[str(file)]=sequence.retained.runtime.task.sha(file)
    record=dict(status='SENSOR_PASS_CURRENT_COMPLETE_PAUSED_BOUNDARY_GET_NOT_A_TRIAL_OR_MISSION_PASS',
        source_report=str(source_observation),source_report_sha256=sequence.retained.runtime.task.sha(source_observation),
        preserved_short_verdict='PASS',actor=port.protocol['actor'],timeline=port.timeline_state(),
        actual_tensor_snapshot=snapshot,actual_layer_texts=dict(Root=port.stage.GetRootLayer().ExportToString(),Session=port.stage.GetSessionLayer().ExportToString()),
        all4997_rows145_settings_physics=checked['environment'],source_fixed_configuration=port.source,
        protected_environment_sha256=port.prep['protected_environment_sha256'],
        zero_sensor_captures_or_PhysX_trials=0,unused_sink=unused,input_sha256=pins,
        Root_Session_tensor_and_clock_unchanged_during_GET=True,original_targets_and_gains_not_written=True)
    target=output/'complete_current_handoff.json';evidence.write_exclusive(target,record)
    lc.freeze_workflow_sources(output/'GET_sources.zip',pins)
    return record


async def capture_failed_route_handoff(host,output,*,authorized=False):
    """GET-only archive readiness, NEVER retry or relabel an incomplete run."""
    return await capture_route_handoff(host,output,authorized=authorized,required_verdict='HOLD')


async def capture_route_handoff(host,output,*,authorized=False,required_verdict):
    """Complete current GET of an own completed/restored route; no motion."""
    if authorized is not True:raise ValueError('Explicit read-only failed-route handoff required')
    source=host.output/'route_readback.json';raw=sequence.retained.read(source)
    if (required_verdict not in ('HOLD','PASS') or raw['verdict']!=required_verdict or raw['new_route_trials']!=1
            or raw['checks'].get('actual_Pause_confirmed') is not True
            or raw['checks'].get('original_gains_targets_restored') is not True
            or raw['checks'].get('contacts_healthy_through_actual_Pause') is not True
            or raw['protocol']!=str(host.path)):
        raise ValueError('Own incomplete actual trial, confirmed Pause and original restoration required')
    if required_verdict=='PASS' and (not all(raw['checks'].values()) or raw.get('error')):
        raise ValueError('A completed verified own route PASS is required, not a verdict override')
    lc.verify_workflow_sources(host.output/'executing_sources.zip',host.protocol['input_sha256'])
    before=host.timeline_state()
    if before!=raw['actual_final_timeline'] or before['playing'] or before['stopped']:
        raise ValueError('Exact own retained actual PAUSED failure boundary required')
    snapshot=sequence.retained.shared.preparation.old.tensor_snapshot(host.tensor)
    layers_before=dict(Root=host.stage.GetRootLayer().ExportToString(),Session=host.stage.GetSessionLayer().ExportToString())
    environment=await host.environment(post_observation=True)
    motor=host.adapter._motor()
    if (not all(np.array_equal(motor[k],v) for k,v in host.adapter.original.items())
            or host.timeline_state()!=before
            or layers_before!=dict(Root=host.stage.GetRootLayer().ExportToString(),Session=host.stage.GetSessionLayer().ExportToString())
            or sequence.retained.shared.preparation.old.tensor_snapshot(host.tensor)!=snapshot):
        raise ValueError('All twelve original gains/targets/limits and current thirteen links must be retained')
    output=Path(output).resolve()
    files=(*lc.host_source_files(),Path(__file__).resolve(),Path(native.__file__).resolve(),source,host.path,
        host.output/'executing_sources.zip',Path(raw['original_sensor_frame_archive']['path']))
    proof=host.output/'explicit_surface_consistency_GET.json'
    if proof.exists():files=(*files,proof)
    pins={str(p):sequence.retained.runtime.task.sha(p) for p in files}
    record=dict(status='SENSOR_'+required_verdict+'_CURRENT_COMPLETE_PAUSED_BOUNDARY_GET_NOT_A_TRIAL_OR_MISSION_PASS',
        source_report=str(source),source_report_sha256=sequence.retained.runtime.task.sha(source),
        preserved_route_verdict=required_verdict,actor=host.protocol['actor'],timeline=before,
        actual_tensor_snapshot=snapshot,actual_layer_texts=dict(Root=host.stage.GetRootLayer().ExportToString(),Session=host.stage.GetSessionLayer().ExportToString()),
        all4997_rows145_settings_physics=environment,source_fixed_configuration=host.source,
        protected_environment_sha256=host.prep['protected_environment_sha256'],
        zero_sensor_captures_or_PhysX_trials=0,input_sha256=pins,
        Root_Session_tensor_and_clock_unchanged_during_GET=True,original_targets_and_gains_not_written=True,
        original_HOLD_NOT_promoted_to_static_route_or_sensor_PASS=True)
    evidence.write_exclusive(output/'complete_current_handoff.json',record)
    lc.freeze_workflow_sources(output/'GET_sources.zip',pins)
    return record


def preparation_definition(handoff_path,output,*,cohort='sensor_nav2_cohort_01_r01',approval,scenario='H5_obstacle_bypass_return'):
    import re
    if not re.fullmatch(r'sensor_nav2_cohort_[0-9]{2}_r[0-9]{2}',cohort):
        raise ValueError('Exact dedicated sensor cohort name required')
    checked_approval(approval)
    platform_math.task_limits(scenario)
    task=scenario.split('_')[0]
    if (scenario=='H6_integrated_transfer_mission')!=(approval in (H6_COMPLETION_APPROVAL,H6_STRICT_APPROVAL,H6_GUARD_REVALIDATION_APPROVAL,H6_CELL_GUARD_APPROVAL)):
        raise ValueError('Exact separate H6 whole-route or H5 workflow authority required')
    handoff_path=Path(handoff_path).resolve();handoff=sequence.retained.read(handoff_path)
    output=Path(output).resolve()
    if (not output.is_relative_to(BASE/('whole_'+task+'_P_sensor_nav2')) or output.exists()
            or handoff.get('status') not in ('SENSOR_PASS_CURRENT_COMPLETE_PAUSED_BOUNDARY_GET_NOT_A_TRIAL_OR_MISSION_PASS',
                'SENSOR_HOLD_CURRENT_COMPLETE_PAUSED_BOUNDARY_GET_NOT_A_TRIAL_OR_MISSION_PASS')
            or handoff['zero_sensor_captures_or_PhysX_trials']!=0
            or not handoff['Root_Session_tensor_and_clock_unchanged_during_GET']
            or not handoff['original_targets_and_gains_not_written']):
        raise ValueError('Exact own current GET and new exclusive full-sensor output required')
    sequence.retained.runtime.task.verify_pins(handoff['input_sha256'])
    if approval==H6_STRICT_APPROVAL:
        prior=sequence.retained.read(handoff['source_report'])
        prior_protocol=sequence.retained.read(prior['protocol'])
        if (handoff.get('preserved_route_verdict')!='PASS'
                or prior_protocol.get('scenario')!='H5_obstacle_bypass_return'
                or prior_protocol.get('sensor_stationary_capture_policy')!=platform_math.SENSOR_CAPTURE_POLICY
                or not all(prior['checks'].values()) or prior.get('error')):
            raise ValueError('Own completed strict15mm H5 sensor PASS required before the conditional H6 case')
    if approval==H6_GUARD_REVALIDATION_APPROVAL:
        prior=sequence.retained.read(handoff['source_report']);prior_protocol=sequence.retained.read(prior['protocol'])
        if (handoff.get('preserved_route_verdict')!='HOLD' or prior['new_route_trials']!=1
                or prior_protocol.get('scenario')!='H6_integrated_transfer_mission'
                or prior_protocol.get('explicit_sensor_workflow_approval')!=H6_STRICT_APPROVAL
                or prior_protocol.get('sensor_stationary_capture_policy')!=platform_math.SENSOR_CAPTURE_POLICY
                or prior_protocol.get('simulation_budget_s')!=600. or prior_protocol.get('maximum_wall_time_s')!=3000.):
            raise ValueError('Exact own completed original H6 sensor HOLD and unchanged limits required')
    if approval==H6_CELL_GUARD_APPROVAL:
        prior=sequence.retained.read(handoff['source_report']);prior_protocol=sequence.retained.read(prior['protocol'])
        if (handoff.get('preserved_route_verdict')!='HOLD' or prior['new_route_trials']!=1
                or prior_protocol.get('scenario')!='H6_integrated_transfer_mission'
                or prior_protocol.get('explicit_sensor_workflow_approval')!=H6_GUARD_REVALIDATION_APPROVAL
                or prior_protocol.get('sensor_stationary_capture_policy')!=platform_math.SENSOR_CAPTURE_POLICY
                or prior_protocol.get('simulation_budget_s')!=600. or prior_protocol.get('maximum_wall_time_s')!=3000.):
            raise ValueError('Own locked-raster H6 sensor HOLD and same original physical limits required')
    template_path=sequence.retained.GATE/('hospital_'+task.lower()+'_experiments')/(
        'cohort_10_held_static_r01' if task=='H5' else 'cohort_20_held_static_r01')/'workflow_protocol.json'
    template=sequence.retained.read(template_path)
    history=lc.reviewed_prior_GET_dependencies(template['input_sha256'],template_path.parent/'workflow_sources.zip')
    protocol=copy.deepcopy(template)
    if (protocol['scenario']!=scenario or protocol['maximum_Stop_calls']!=1
            or protocol['maximum_static_trials']!=1 or protocol['maximum_route_trials']!=0
            or protocol['protected_environment_sha256']!=handoff['protected_environment_sha256']):
        raise ValueError('Same adopted H5 source/environment/preparation template required')
    protocol.update(cohort=cohort,output_directory=str(output),
        new_robot_root='/World/patient_transport_'+task.lower()+'_'+cohort,marker='/World/IJCAS_Hospital'+task+'_'+cohort,
        retired_robot_root=handoff['actor'],required_prior_result=str(handoff_path),
        required_prior_tensor=handoff['actual_tensor_snapshot'],required_prior_timeline=handoff['timeline'],
        required_prior_layer_texts=handoff['actual_layer_texts'],
        prior_GET_source_archive=str(handoff_path.parent/'GET_sources.zip'),
        prior_verdict_preserved=handoff.get('preserved_route_verdict',handoff.get('preserved_short_verdict')),
        retiring_actor_fixed_configuration=handoff['source_fixed_configuration'],user_approval=checked_approval(approval),
        status='ONE_SENSOR_WORKFLOW_PREPARATION_DEFINITION_NOT_YET_EXECUTED',
        sensor_integration_NOT_a_new_primary_matched_comparison=True,
        original_physical_preparation_code_and_all_guards_unchanged=True,
        only_authorization_record_type_extended_in_private_binding=True)
    protocol.pop('prior_handoff_path',None)
    pins=dict(template['input_sha256'])
    for row in history['resolved_consumed_sources']:pins.pop(row['path'])
    for file in (*lc.host_source_files(),Path(sequence.__file__).resolve(),Path(__file__).resolve(),template_path,
            handoff_path,handoff_path.parent/'GET_sources.zip'):
        pins[str(file)]=sequence.retained.runtime.task.sha(file)
    pins.update(handoff['input_sha256'])
    protocol['input_sha256']=pins
    return evidence.json_value(protocol)


def export_preparation(handoff_path,output,*,approval,cohort='sensor_nav2_cohort_01_r01',scenario='H5_obstacle_bypass_return'):
    definition=preparation_definition(handoff_path,output,approval=approval,cohort=cohort,scenario=scenario)
    file=Path(output)/'workflow_protocol.json';evidence.write_exclusive(file,definition)
    lc.freeze_workflow_sources(file.parent/'workflow_sources.zip',definition['input_sha256'])
    return file


async def prepare_and_static_once(protocol_path,*,authorized=False):
    """ONE original archive/Stop/new-source/display/static, no route here."""
    if authorized is not True:raise ValueError('Explicit ONE sensor preparation/static authority required')
    path=Path(protocol_path).resolve();p=sequence.retained.read(path)
    checked_approval(p['user_approval']);sequence.retained.runtime.task.verify_pins(p['input_sha256'])
    if path!=Path(p['output_directory'])/'workflow_protocol.json':raise ValueError('Own exclusive preparation packet required')
    context=sequence.retained.geometry.common_preparation_context(p,path)
    context.source_archive=path.parent/'workflow_sources.zip';lc.verify_workflow_sources(context.source_archive,p['input_sha256'])
    reserve=context.output/'sensor_workflow_reserved.json'
    if reserve.exists():return dict(status='EXISTING_SENSOR_WORKFLOW_PRESERVED_NO_SECOND_STOP_OR_STATIC')
    evidence.write_exclusive(reserve,dict(protocol=str(path),authorization=checked_approval(p['user_approval']),
        maximum_Stop_calls=1,maximum_static_trials=1,automatic_retry=False,no_route_in_this_operation=True))
    phase='archive_ONE_Stop_source_display';host=None
    try:
        sdk=sequence.retained.runtime.live_sdk()
        operations=_authorization_binding(lc.fresh_held_preparation_operations)(context,sdk,authorized=True)
        prepared=await operations(context)
        stage=sdk.omni.usd.get_context().get_stage()
        layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
        bound,packet,static_path=lc.bind_prepared_held_packet(context,prepared,layers)
        g=sequence.retained.read(packet['geometry_file'])
        with np.load(packet['grid_file']) as grid:
            host=SensorPreparedHeldStaticHost(sdk,packet,static_path,context=bound,geometry=g,
                occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),resolution=float(grid['resolution_m']),authorized=True)
        phase='ONE_own_original_held_static'
        result=await lc.run_held_static_once(host,lc.HeldStaticController(p['start_xyyaw']),authorized=True)
        replay=lc.replay_held_static_evidence(host.result)
        evidence.write_exclusive(host.result.parent/'sensor_static_raw_replay.json',replay)
        receipt=dict(status='ONE_FRESH_SENSOR_STATIC_FINISHED_NOT_ROUTE',static_verdict=result['verdict'],
            raw_replay_verdict=replay['verdict'],static_result=str(host.result),preparation=str(context.preparation.LIVE),
            protocol=str(path),actor=p['new_robot_root'],actual_final_timeline=host.timeline_state(),
            route_trials=0,automatic_retry=False)
        evidence.write_exclusive(context.output/'sensor_preparation_closeout.json',receipt)
        return receipt
    except BaseException as failure:
        value=dict(status='SENSOR_FRESH_PREPARATION_HOLD_NOT_DRIVE_ADMISSION',phase=phase,
            error=type(failure).__name__+': '+str(failure),automatic_retry=False,route_trials=0)
        try:value['actual_failure_boundary']=lc.capture_fresh_workflow_failure(context,sdk,host)
        except Exception as error:value['boundary_error']=type(error).__name__+': '+str(error)
        evidence.write_exclusive(context.output/'sensor_preparation_error.json',value)
        return value


def bridge_definition(route_path,*,actor,snapshot,map_report,plugin_prefix,capture_id,initial_stamp_ns,
        initial_prior_xyyaw,topic_name,diagnostic=False,sensor_capture_policy=None):
    """Pure explicit configuration; no bridge process or native action."""
    import hospital_sensor_platform as platform
    import hospital_sensor_ros_bridge as ros
    import hospital_sensor_transport as transport
    route_path=Path(route_path).resolve();route=sequence.retained.read(route_path)
    limits=platform_math.task_limits(route['scenario'])
    encoder=BASE/'encoder_diagnostic'
    files=dict(controller_definition=route_path,frozen_plan=route_path.parent/'frozen_plan.json',
        geometry=Path(route['geometry_file']),encoder_geometry=encoder/'retrospective_h5_r01/source_geometry_GET.json',
        encoder_wrap_contract=encoder/'wrapped_positions_h5_r01/source_wrap_and_cap_contract.json')
    if route['scenario']=='H6_integrated_transfer_mission':files['original_controller_source_archive']=route_path.parent/'route_sources.zip'
    pins={str(p.resolve()):sequence.retained.runtime.task.sha(p) for p in files.values()}
    definition=dict(version=platform.VERSION,scenario=route['scenario'],method='P_dynamics_full_envelope',
        actor_root=actor,native_joint_names=snapshot['names'],original_native_position_targets_rad=snapshot['joint_position_targets_rad'],
        sensor_speed_window_s=.2,control_minimum_period_s=.05,maximum_control_gap_s=.25,simulation_budget_s=limits['simulation_budget_s'],
        maximum_wall_time_s=limits['maximum_wall_time_s'],maximum_native_route_trials=1,automatic_retry=False,
        ground_truth_control_pose_or_velocity_inputs=False,input_sha256=dict(pins),**{k:str(v.resolve()) for k,v in files.items()})
    if sensor_capture_policy is not None:
        if sensor_capture_policy!=platform_math.SENSOR_CAPTURE_POLICY:raise ValueError('Exact reviewed strict15mm SENSOR policy required')
        definition['sensor_stationary_capture_policy']=copy.deepcopy(sensor_capture_policy)
    if route['scenario']=='H6_integrated_transfer_mission':
        definition['sensor_wall_budget_review']=copy.deepcopy(platform_math.H6_SENSOR_WALL_REVIEW)
    plugin_prefix=Path(plugin_prefix).resolve();map_report=Path(map_report).resolve()
    plugin_source=sequence.retained.ROOT/'src/ijcas_nav2_sensor_adapter'
    for file in (map_report,Path(sequence.retained.read(map_report)['map_data_file']) if 'map_data_file' in sequence.retained.read(map_report) else map_report.with_suffix('.npz'),
            plugin_prefix/'lib/libijcas_sensor_nav2_plugins.so',plugin_prefix/'share/ijcas_nav2_sensor_adapter/plugins.xml',
            plugin_source/'src/plugins.cpp',plugin_source/'include/ijcas_nav2_sensor_adapter/contracts.hpp',
            plugin_source/'CMakeLists.txt',plugin_source/'test/contracts.cpp',
            *lc.host_source_files(),Path(sequence.__file__).resolve(),Path(__file__).resolve(),
            Path(native.__file__).resolve(),Path(ros.__file__).resolve(),Path(platform.__file__).resolve(),
            Path(transport.__file__).resolve(),Path(native.messages.__file__).resolve()):
        pins[str(file)]=sequence.retained.runtime.task.sha(file)
    config=dict(version='ISOLATED_NATIVE_SENSOR_AMCL_BRIDGE_v1',authorized_data_only=True,actuator_connection=False,
        capture_kind='LIVE_ACTUAL_ACQUISITION',domain_id=79,topic_prefix='/ijcas_sensor_dev/'+topic_name,
        map_frame_id='ijcas_sensor_dev_map',odom_frame_id='ijcas_sensor_dev_odom',
        initial_prior_xyyaw=list(initial_prior_xyyaw),initial_native_stamp_ns=initial_stamp_ns,
        force_stationary_updates=True,maximum_frames=limits['maximum_sensor_frames'],actor_root=actor,capture_id=capture_id,
        input_sha256=pins,map_report=str(map_report),encoder_geometry=str(files['encoder_geometry']),
        encoder_wrap_contract=str(files['encoder_wrap_contract']),mission_platform_definition=definition,
        compact_raw_source_frames=True,nav2_plugin_prefix=str(plugin_prefix),
        initial_prior_is_disclosed_fixed_task_map_prior_NOT_GT_feedback=True)
    if diagnostic:config['diagnostic_zero_plugin_smoke']=True
    return ros.checked_configuration(config)


def validate_actual_command(response,encoder):
    """Fail-closed actuator boundary: actual Nav2 output, not a proposal bypass."""
    import numpy as np
    p=response['proposal'];stamp=encoder['stamp_ns']
    if (response['stamp_ns']!=stamp or p['stamp_ns']!=stamp
            or p['proposal_only_NO_SDK_actuation_or_physical_admission'] is not True):
        raise ValueError('Exact independently admitted native encoder proposal required')
    requested=np.asarray(p['proposed_body_twist_u_v_r'],float)
    command=response['actual_nav2_command']
    if response['prehold_ZERO_only_before_Nav2']:
        if command is not None or np.max(abs(requested))>1e-10:raise ValueError('Prehold allows ZERO preparation only')
        return p['proposed_joint_targets'],1.
    if response.get('fixed_hold_complete_ZERO_only'):
        if response['actual_nav2_action_status']!=4 or np.max(abs(requested))>1e-10:
            raise ValueError('Actual Nav2 completion and zero required')
        return p['proposed_joint_targets'],1.
    if response.get('actual_plugin_ack_source_stamp_ns')!=stamp or command is None:
        raise ValueError('Actual Nav2 plugin/command acknowledgement required before native write')
    actual=np.array([command['linear_m_s'][0],command['linear_m_s'][1],command['angular_rad_s'][2]])
    if not np.isfinite(actual).all() or max(abs(command['linear_m_s'][2]),abs(command['angular_rad_s'][0]),abs(command['angular_rad_s'][1]))>1e-10:
        raise ValueError('Finite planar actual Nav2 command required')
    nz=np.flatnonzero(abs(requested)>1e-9)
    scale=1. if not len(nz) else float(actual[nz[0]]/requested[nz[0]])
    if not 0.<=scale<=1.000000001 or np.max(abs(actual-scale*requested))>1e-9:
        raise ValueError('Nav2 may only uniformly reduce the exact multi-steering proposal')
    return p['proposed_joint_targets'],min(1.,scale)


class WholeSensorMissionHost(lc.RetainedMissionSDKHost):
    """New scope binding, SAME physical guards/readbacks/cleanup code."""
    async def preflight(self):
        prep,raw,replay=sequence.held_static_adapter(self.protocol['own_fresh_held_static'])
        if replay['verdict']!='PASS' or raw!=self.raw or prep!=self.prep:
            raise ValueError('Own new static raw replay required; no inherited static PASS')
        result=await sequence.retained.RetainedZeroSettleSDKPort.preflight(self)
        pose=self._pose(self.protocol['required_tensor_snapshot']);start=self.plan['start_world_xyyaw']
        snapshot=sequence.retained.shared.preparation.old.tensor_snapshot(self.tensor)
        if (math.hypot(pose[0]-start[0],pose[1]-start[1])>.010
                or abs(lc.wrap(pose[2]-start[2]))>math.radians(.5) or math.hypot(*snapshot['velocity'][:2])>.005
                or max(abs(snapshot['positions'][i]) for i in self.adapter.steering)>math.radians(1.)):
            raise ValueError('Original10mm/.5deg/5mm_s/nominal1deg fixed-start release failed')
        if self.raw['environment_after_static']['physics_hz']!=120:raise ValueError('Original120Hz required')
        self.last_dt=1./120.
        return dict(result,own_static_replay=replay,original_fixed_start_release_verified=True)

    async def confirm_pause(self):
        result=await super().confirm_pause()
        # Close owned ROS before lengthy raw replay, but never turn a ROS
        # close error into a false Pause result that would skip restoration.
        try:self.ros_close_receipt=self.bridge.close()
        except Exception as error:self.ros_close_error=type(error).__name__+': '+str(error)
        return result

    async def postflight(self):
        import asyncio
        result=await super().postflight();self.physical_postflight_before_ROS_gate=result;before=self.timeline_state()
        snapshot=result['actual_post_mission_tensor_snapshot']
        self.archive_receipt=await asyncio.to_thread(self.frame_archive.finish)
        self.control_receipt=await asyncio.to_thread(self.control_archive.finish)
        if before!=self.timeline_state() or sequence.retained.shared.preparation.old.tensor_snapshot(self.tensor)!=snapshot:
            raise ValueError('Actual PAUSED state changed during lossless evidence replay')
        bridge_result=Path(self.bridge.ready['configuration_path']).parent/'bridge_result.json'
        began=time.monotonic()
        while not bridge_result.exists() and time.monotonic()-began<10.:await self.app.next_update_async()
        ros=sequence.retained.read(bridge_result)
        if ros.get('error') or ros.get('actual_Nav2_result_status')!=4 or ros.get('frame_count')!=self.frame_archive.count:
            raise ValueError('Complete actual ROS/Nav2 success/count readback required: '+str(ros.get('error')))
        result['actual_ROS_result']=str(bridge_result)
        result['checks'].update(lossless_all_original_sensor_frames_verified=True,
            exact_all_control_and_AMCL_receipts_verified=True,actual_Nav2_FollowPath_status4=True,
            only_measured_AMCL_encoder_inputs_used_for_control=True)
        return result

    def save(self,report,rows):
        import csv,os
        for attr,key in (('frame_archive','original_sensor_frame_archive'),('control_archive','actual_control_receipts')):
            archive=getattr(self,attr)
            try:
                receipt=(getattr(self,'archive_receipt' if attr=='frame_archive' else 'control_receipt',None)
                    or archive.finish())
                report[key]=receipt
            except Exception as error:
                report['verdict']='HOLD';report[key+'_error']=type(error).__name__+': '+str(error)
        if rows:
            keys=list(dict.fromkeys(k for row in rows for k in row))
            with (self.output/'physics_trace.csv').open('x',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows);stream.flush();os.fsync(stream.fileno())
        report.update(protocol=str(self.path),actual_final_timeline=self.timeline_state(),
            scenario=self.plan['scenario'],actual_original_frozen_P_full_route_SENSOR_NAV2_PASS=report['verdict']=='PASS',
            actual_original_frozen_P_H5_full_route_SENSOR_NAV2_PASS=report['verdict']=='PASS' and self.plan['scenario']=='H5_obstacle_bypass_return',
            actual_original_frozen_P_H6_full_route_SENSOR_NAV2_PASS=report['verdict']=='PASS' and self.plan['scenario']=='H6_integrated_transfer_mission',
            primary_matched_comparison_or_all_H1_to_H6_Nav2_claim=False,
            actual_ROS_close_error=getattr(self,'ros_close_error',None))
        if hasattr(self,'physical_postflight_before_ROS_gate'):
            report['physical_postflight_before_separate_ROS_success_gate']=self.physical_postflight_before_ROS_gate
        evidence.write_exclusive(self.output/'route_readback.json',report)


def bind_whole_sensor_host(static_result,route_path,output,*,approved=False,approval=APPROVAL):
    if approved is not True:raise ValueError('Current explicit whole sensor workflow authority required')
    import hospital_sensor_transport as transport
    output=Path(output).resolve();static_result=Path(static_result).resolve();route_path=Path(route_path).resolve()
    original=sequence.retained.read(route_path);plan=sequence.retained.read(route_path.parent/'frozen_plan.json')
    limits=platform_math.task_limits(original['scenario']);task=original['scenario'].split('_')[0]
    if (task=='H6')!=(approval in (H6_COMPLETION_APPROVAL,H6_STRICT_APPROVAL,H6_GUARD_REVALIDATION_APPROVAL,H6_CELL_GUARD_APPROVAL)):raise ValueError('Exact H5/H6 whole route authority required')
    if not output.is_relative_to(BASE/('whole_'+task+'_P_sensor_nav2')) or output.exists():raise ValueError('New exact full sensor output required')
    prep,raw,replay=sequence.held_static_adapter(static_result)
    geometry=sequence.retained.read(original['geometry_file']);lc.motion.validate_mission_plan(plan,geometry)
    packet_path=Path(raw['protocol']);packet=sequence.retained.read(packet_path)
    if packet['explicit_static_approval']!=checked_approval(approval):raise ValueError('Same own fresh static and exact route authorization required')
    context=sequence.retained.static_binding(packet,packet_path,static_result.parent,Path(raw['preparation_record']))
    sdk=sequence.retained.runtime.live_sdk();stage=sdk.omni.usd.get_context().get_stage()
    timeline=sequence.retained.evidence.timeline_state(sdk.omni.timeline.get_timeline_interface())
    if timeline!=raw['actual_final_timeline'] or timeline['playing'] or timeline['stopped']:
        raise ValueError('Exact own fresh STATIC PAUSED boundary required')
    source=output/'own_static_schema_binding.json';evidence.write_exclusive(source,raw)
    layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    protocol=dict(original,version='FRESH_'+task+'_FROZEN_P_MEASURED_SENSOR_NAV2_HOST_v1',
        actor=prep['new_robot_root'],output_directory=str(output),own_fresh_held_static=str(static_result),
        original_static_record=str(source),original_static_trace=raw['trace'],
        required_PAUSED_timeline=timeline,required_tensor_snapshot=raw['actual_post_static_tensor_snapshot'],
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(sequence.retained.TOP_XY_DISPLAY_PATHS),policy=sequence.retained.TOP_XY_DISPLAY_POLICY),
        explicit_sensor_workflow_approval=checked_approval(approval),maximum_Play_calls=1,
        maximum_Stop_reset_restaging_calls=0,automatic_retry=False,use_for_final_statistics=False)
    protocol['maximum_wall_time_s']=limits['maximum_wall_time_s']
    if task=='H6':protocol['sensor_wall_budget_review']=copy.deepcopy(platform_math.H6_SENSOR_WALL_REVIEW)
    if approval in (STRICT_CAPTURE_APPROVAL,H6_STRICT_APPROVAL,H6_GUARD_REVALIDATION_APPROVAL,H6_CELL_GUARD_APPROVAL):
        protocol['sensor_stationary_capture_policy']=copy.deepcopy(platform_math.SENSOR_CAPTURE_POLICY)
    if approval==H6_GUARD_REVALIDATION_APPROVAL:
        protocol['Nav2_guard_read_review']=dict(whole6pose_check_under_original_costmap_mutex=True,
            actual_locked_rejection_grid_snapshot_bounded65536cells=True,lethal254_unknown255_and_outside_rejection_unchanged=True,
            original_padded_body_and250ms_projection_unchanged=True,prior_HOLD_cause_NOT_proven_by_this_change=True)
    if approval==H6_CELL_GUARD_APPROVAL:
        protocol['Nav2_cell_geometry_guard_review']=dict(version='CLOSED_CONVEX_FULL_POLYGON_CELL_SAT_v1',
            same_original_padded_body_and_every_original_cell_square=True,interior_cells_included=True,
            boundary_touch_lethal254_unknown255_outside_and_invalid_geometry_fail_closed=True,
            no_cell_erosion_relabelling_margin_reduction_or_guard_bypass=True,
            whole6pose_check_under_original_costmap_mutex=True,original250ms_projection_unchanged=True,
            independent_native_full_body_contact_goal_stop_guards_unchanged=True,
            guard_fix_does_NOT_assert_whole_route_success=True)
    if approval in (REVALIDATION_APPROVAL,BRIDGE_FIX_APPROVAL,H6_COMPLETION_APPROVAL,STRICT_CAPTURE_APPROVAL,H6_STRICT_APPROVAL,H6_GUARD_REVALIDATION_APPROVAL,H6_CELL_GUARD_APPROVAL):protocol['native_source_surface_consistency_policy']=copy.deepcopy(SURFACE_POLICY)
    if approval in (BRIDGE_FIX_APPROVAL,STRICT_CAPTURE_APPROVAL,H6_STRICT_APPROVAL,H6_GUARD_REVALIDATION_APPROVAL,H6_CELL_GUARD_APPROVAL):
        protocol['Nav2_bridge_fix_review']=dict(independent_ROS_message_headers=True,
            maximum_proposal_input_roundoff_m_s=1e-8,actual_output_linear_component_cap_m_s=.05,
            actual_output_yaw_cap_deg_s=8.,only_uniform_twist_reduction=True,
            original_source_timestamps_and250ms_age_bound_unchanged=True,
            physical_collision_goal_stop_and_time_criteria_NOT_relaxed=True)
    pins={str(p):sequence.retained.runtime.task.sha(p) for p in (*lc.host_source_files(),Path(sequence.__file__).resolve(),
        Path(__file__).resolve(),Path(native.__file__).resolve(),static_result,packet_path,Path(raw['preparation_record']),source,route_path,
        route_path.parent/'frozen_plan.json',Path(protocol['geometry_file']),Path(protocol['grid_file']))}
    protocol['input_sha256']=pins;path=output/'protocol.json';evidence.write_exclusive(path,protocol)
    lc.freeze_workflow_sources(output/'executing_sources.zip',pins)
    host=object.__new__(WholeSensorMissionHost)
    sequence.retained.RetainedZeroSettleSDKPort.__init__(host,sdk,protocol,context=context)
    host.path=path;host.output=output;host.plan=plan;host.geometry=geometry;host.notice_sub=None;host._notice_callback=None
    host.notice_counts=dict(allowed_state_paths=0,reviewed_aperture_paths=0,unreviewed_paths=0)
    with np.load(protocol['grid_file']) as grid:
        host.runtime_guard=lc.MissionRuntimeGuard(geometry,grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m']),simulation_budget_s=limits['simulation_budget_s'])
    host.frame_archive=transport.RawSourceFrameArchive(output/'original_native_frames.jsonl.gz',maximum_frames=limits['maximum_sensor_frames'],maximum_compressed_bytes=(256 if task=='H5' else 640)*1024*1024)
    host.control_archive=transport.BoundedJSONLArchive(output/'actual_control_receipts.jsonl.gz',maximum_records=limits['maximum_control_records'])
    return host


async def run_whole_sensor_once(host,sensor,bridge,*,authorized=False):
    """ONE actual P route; original P math in ROS, actual Nav2 gates writes."""
    import asyncio
    if authorized is not True or bridge.configuration.get('diagnostic_zero_plugin_smoke'):
        raise ValueError('Own actual sensor/Nav2 whole route, never a diagnostic, required')
    checked_approval(host.protocol['explicit_sensor_workflow_approval'])
    if (host.protocol.get('sensor_stationary_capture_policy')!=bridge.configuration['mission_platform_definition'].get('sensor_stationary_capture_policy')
            or host.protocol.get('sensor_wall_budget_review')!=bridge.configuration['mission_platform_definition'].get('sensor_wall_budget_review')):
        raise ValueError('Same explicitly reviewed sensor capture and wall policy required at BOTH boundaries')
    reserved=host.output/'single_route_reserved.json'
    if reserved.exists():return dict(status='EXISTING_FULL_SENSOR_ROUTE_RETAINED_NO_RETRY')
    evidence.write_exclusive(reserved,dict(approval=host.protocol['explicit_sensor_workflow_approval'],maximum_Play_calls=1,automatic_retry=False))
    host.bridge=bridge;report=dict(verdict='HOLD',checks={},new_route_trials=0,automatic_retry=False)
    rows=[];preflight=actuation=False;response=None;held_origin=None;last_control=None
    reference=lc.motion.MissionPathReference(host.plan).legs[0]['view'];watch=time.monotonic()
    actual_origin=host.timeline_state()['current_time_s'];initial_stamp=round(actual_origin*1e9)
    try:
        report['preflight']=await host.preflight();preflight=True;host.subscribe()
        def capture():
            frame=sensor.capture();host.frame_archive(frame);return bridge.frame(frame)
        first=capture();last_scan=first['stamp_ns']
        report['actual_Nav2_preparation']=bridge.prepare_controller()
        report['checks']['actual_Nav2_parameters_and_binary_verified_before_Play']=True
        row=host.measure()
        while True:
            host.guard(row,None);now=row['actual_timeline_s'];stamp=round(now*1e9)
            if stamp-last_scan>=200000000:
                receipt=capture();last_scan=receipt['stamp_ns']
            if last_control is None or stamp-last_control>=50000000:
                encoder=sensor.encoders()
                response=bridge.platform_control(encoder,(stamp-initial_stamp)/1e9,host.adapter.confirmation(host.clock))
                targets,scale=validate_actual_command(response,encoder)
                host.control_archive(response);last_control=stamp
                proposal=response['proposal'];info=proposal['controller_info'];host.guard(row,info)
                requested=proposal['requested_hold'];speed=math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])
                if requested is not None and speed<=.005:
                    actuation=True;host.adapter.request_hold(requested,host.clock,command_speed_m_s=info['speed_m_s'],measured_speed_m_s=speed)
                if proposal['actuator_write_required']:
                    actuation=True
                    # Position targets are the EXACT profile-bound original P
                    # result. No differential-drive steering approximation.
                    host.adapter.command(targets['steering_targets_rad'],
                        [scale*v for v in targets['wheel_velocity_targets_rad_s']],host.clock)
            proposal=response['proposal'];info=proposal['controller_info']
            requested=proposal['requested_hold'];speed=math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])
            if requested is not None and speed<=.005:
                actuation=True;host.adapter.request_hold(requested,host.clock,command_speed_m_s=info['speed_m_s'],measured_speed_m_s=speed)
            pose=(row['x_m'],row['y_m'],row['yaw_rad']);_,path_pose,deviation=reference.project(pose)
            if info['phase']=='path_tracking' and (deviation>.080 or abs(lc.wrap(pose[2]-path_pose[2]))>math.radians(10.)):
                raise RuntimeError('Original independent actual path deviation80mm/10deg guard')
            phase=info['phase']
            if phase=='terminal_hold' and held_origin is None:held_origin=row['time_s']
            feedback=proposal['matched_sensor_feedback'];estimated=feedback['estimated_map_xyyaw']
            row.update(phase=phase,phase_time_s=0. if held_origin is None else row['time_s']-held_origin,
                path_progress_m=reference.progress,command_speed_m_s=info['speed_m_s'],
                sensor_feedback_x_m=estimated[0],sensor_feedback_y_m=estimated[1],sensor_feedback_yaw_rad=estimated[2],
                AMCL_fix_stamp_ns=feedback['fix_stamp_ns'],control_encoder_stamp_ns=last_control,
                localization_position_error_m=math.hypot(estimated[0]-pose[0],estimated[1]-pose[1]),
                localization_heading_error_deg=math.degrees(lc.wrap(estimated[2]-pose[2])))
            rows.append(row)
            if time.monotonic()-watch>=15.:
                print('['+host.plan['scenario'].split('_')[0]+' SENSOR NAV2] sim=%.2fs phase=%s progress=%.3f/%.3fm goal=%.1fmm'%(host.clock,phase,reference.progress,reference.arc[-1],
                    1000*math.hypot(pose[0]-host.plan['goal_world_xyyaw'][0],pose[1]-host.plan['goal_world_xyyaw'][1])),flush=True);watch=time.monotonic()
            if proposal['complete']:
                report['terminal']=lc.evaluate_terminal_hold(rows,host.plan['goal_world_xyyaw'],reference.arc[-1],host.counts,host.errors,
                    fixed_hold_attempts=proposal['fixed_hold_attempts'])
                report['checks'].update(report['terminal']['checks']);break
            if report['new_route_trials']==0:report['new_route_trials']=1;host.play()
            await host.next_sample();row=host.measure()
    except (Exception,asyncio.CancelledError) as error:
        report['error']=type(error).__name__+': '+str(error);report['checks']['whole_route_and_fixed_hold_completed']=False
        if response is not None:report['last_actual_control_receipt']=response
    finally:
        report=await lc._finish_observation(host,report,rows,preflight=preflight,actuation=actuation,
            capture_count=0 if response is None else response['proposal']['fixed_hold_attempts'])
    return report
