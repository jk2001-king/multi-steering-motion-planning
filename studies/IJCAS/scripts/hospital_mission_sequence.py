"""Frozen one-cell continuations using the existing mission SDK/controller.

Import/export never touch Isaac. A held-static schema adapter is an explicitly
labelled representation of the SAME immutable report/CSV, not another trial.
The actual shared SDK port, controller, guards, Pause and restoration stay
unchanged. No retries, method-specific control settings or success selection.
"""

import asyncio
import copy
import hashlib
from pathlib import Path

import numpy as np

import hospital_mission_lifecycle as lifecycle


retained=lifecycle.retained
ROOT=retained.ROOT
PLANNING=retained.GATE/'hospital_h6_planning/heading0_fresh_terminal_beam_matched_dev_r19'
STOP_CONNECTION=retained.GATE/'hospital_h6_planning/neutral_stop_connection_model_review_r01'
STOP_CONNECTION_MODEL=STOP_CONNECTION/'executor_model_retained_final_targets'
METHODS=('P_dynamics_full_envelope','B0_fixed_counterphase_ramp',
    'B1_multimode_kinematic_ramp','B2_multimode_transition_cost_ramp')
APPROVAL=dict(question_item_id='["request_user_input_async","call_1f3fed6c1c1a4cc38fec374545f35919",0]',
    answer_ko='H6 공통 600초·미완료 칸 1회씩 자동 진행 승인')
REVALIDATION_APPROVAL=dict(question_item_id='["request_user_input_async","call_52bb9011adc749ae84f803d9d2a0a0de",0]',
    answer_ko='수정 검증 후 새 코호트·정적·P 재검증 1회 승인')
TRACKING_FLOOR_APPROVAL=dict(approval_source='USER_MESSAGE',user_message='진행하자',
    scope='ONE_NEW_SOURCE_STATIC_THEN_CONDITIONAL_P_COMMON10mm_s_TRACKING_FLOOR_NO_RETRY',
    prior_explained_limits='600s_sim_900s_wall_50mm_s_8deg_s_original_safety_terminal_hold')
CAMERA_INTERRUPTION_APPROVAL=dict(TRACKING_FLOOR_APPROVAL,
    user_message='아 내가 잘 가는지 보려고 시점 변경을 해서 생긴 문제네 다시 진행하면 문제없을듯',
    scope='ONE_FRESH_STATIC_AND_SAME_P_CANDIDATE_AFTER_ACKNOWLEDGED_CAMERA_INTERRUPTION_NO_RETRY')
MEASURED_RESPONSE_FIX_APPROVAL=dict(approval_source='USER_MESSAGE',user_message='그럼 해결하고 바로 진행해',
    scope='ONE_FRESH_STATIC_THEN_CONDITIONAL_P_MEASURED20mm_s_TRACKING_FLOOR_DEVELOPMENT_NO_RETRY',
    prior_explained_limits='600s_sim_900s_wall_50mm_s_8deg_s_original_safety_terminal_hold')
TRACKING_FLOOR_APPROVALS=(TRACKING_FLOOR_APPROVAL,CAMERA_INTERRUPTION_APPROVAL,MEASURED_RESPONSE_FIX_APPROVAL)
DIRECTIONAL_QUIET_STOP_APPROVAL=dict(
    question_item_id='["request_user_input_async","call_6c504af3f5734cb29f425935f6d4addd",0]',
    answer_ko='검증 후 새 코호트·정적·보완 P 1회 승인')
REVERSE_PROBE_APPROVAL=dict(
    question_item_id='["request_user_input_async","call_30aa82a10454455885db20261874e377",0]',
    answer_ko='검증 후 짧은 후진 응답 시험 1회 승인')
HELD_NEUTRAL_PROBE_APPROVAL=dict(approval_source='USER_MESSAGE',
    user_message='바로 진행해 오늘 정말 끝을 볼거니까',
    scope='ONE_RETAINED_ZERO_COMMAND_HELD_NEUTRAL_RESPONSE_PROBE_AFTER_OFFLINE_FIX_NO_ROUTE_NO_RETRY')
COMPLIANT_HOLD_APPROVAL=dict(
    question_item_id='["request_user_input_async","call_bd17a2f896594ec9afea662f059acb88",0]',
    answer_ko='수정 검증 후 짧은 시험·조건부 정적·P 1회 승인')
ITERATIVE_FIX_APPROVAL=dict(approval_source='USER_MESSAGE',
    user_message='왜 자꾸 hold가 나는거야 할거면 좀 확실하게 문제 고치면서 pass가 나와야지 바로 진행해 pass 될 때까지',
    scope='EVIDENCE_LED_BOUNDED_H6_FIX_AND_REVALIDATION_PRESERVE_ALL_LIMITS_AND_FAILURES',
    no_unchanged_blind_retries_or_automatic_acceptance_relaxation=True)
TRACKING_FLOOR_POLICY='COMMON_TRACKING_TARGET_FLOOR_10mm_s_WITH_ORIGINAL_BRAKING_CAP_v1'
MEASURED_RESPONSE_FLOOR_POLICY='COMMON_TRACKING_TARGET_FLOOR_20mm_s_WITH_ORIGINAL_BRAKING_CAP_DEV_v1'
DIRECTIONAL_QUIET_STOP_POLICY='COMMON_FORWARD10_REVERSE20mm_s_250ms_STOP_AND3mm_s_CAPTURE_DEV_v1'
COMPLIANT_HOLD_POLICY='COMMON_DIRECTIONAL_QUIET_STOP_HELD4deg_s_KP1000_DEV_v1'
FINAL_TARGET_HOLD_POLICY='COMMON_DIRECTIONAL_QUIET_STOP_RETAIN_FINAL_TARGETS_DEV_v1'
VERSION='H6_SAME_COMPLETED_HELD_STATIC_ONE_COMMON_ROUTE_TYPED_TOP_XY_v2'
H3_CONTINUOUS_APPROVAL=dict(approval_source='USER_MESSAGE',
    user_message='바로 이제 1. H3 검증 시작하자 1. 완료될때까지 계속 검증하고 결과 나오면 알려줘',
    scope='H3_FOUR_METHOD_FRESH_CONTINUOUS_WHOLE_TASK_VALIDATION_WITH_EVIDENCE_LED_FIXES',
    preserve_all_failures_and_original_safety_limits=True,no_unchanged_blind_retries=True)
REMAINING_NO_NAV2_APPROVAL=dict(approval_source='USER_MESSAGE',
    user_message='오케이 그러면 일단 nav2 연동을 배제한 나머지 부분 전부 결과 나올때까지 진행하자 쭉',
    scope='ONE_COMMON1330_CASE_PER_REMAINING_TASK_METHOD_NO_NAV2_NO_BLIND_RETRY',
    preserve_all_failures_and_original_safety_limits=True)
TASK_SCENARIOS=dict(H1='H1_bedside_departure',H2='H2_bedside_lateral_alignment',
    H3='H3_doorway_immediate_turn',H4='H4_ward_corridor_turn',H5='H5_obstacle_bypass_return',H6='H6_integrated_transfer_mission')
TASK_BUDGETS=dict(H1=180.,H2=180.,H3=300.,H4=300.,H5=240.,H6=600.)

# Prospective common PC-runtime allowance, not a motion/acceptance relaxation.
# The original 900s comparison and its failures remain immutable evidence.
H6_COMMON_WALL_REVIEW=dict(
    version='COMMON_H6_PC_WALL1500_PLAN180_SIM600_DEVELOPMENT_v1',
    approval=dict(approval_source='USER_MESSAGE',user_message='추천안으로 가는게 좋아보이네',
        question_item_id='["request_user_input_async","call_401643cd21f24768a5d03ae6f6fcd864",0]'),
    methods=list(METHODS),planning_wall_budget_s=180.,simulation_budget_s=600.,maximum_wall_time_s=1500.,
    only_actual_PC_wall_allowance_changed=True,physics_controller_acceptance_unchanged=True,
    old900s_truncation_NOT_reusable_as_same_condition_outcome=True,automatic_retry=False)


def checked_common_wall_budget(packet):
    """Exact reviewed H6 allowance for ALL methods; legacy 900s is unchanged."""
    value=packet.get('maximum_wall_time_s',900.)
    if type(value) not in (int,float) or value not in (900.,1500.):
        raise ValueError('Only the original or explicitly reviewed common wall budget is allowed')
    review=packet.get('common_wall_budget_review')
    if value==900.:
        if review is not None:raise ValueError('No unrequested wall-policy opinion in an original900s packet')
        return 900.
    if (packet.get('scenario')!=TASK_SCENARIOS['H6'] or packet.get('simulation_budget_s')!=600.
            or ('method' in packet and packet['method'] not in METHODS)
            or review!=H6_COMMON_WALL_REVIEW):
        raise ValueError('Exact approved H6 common1500s review required; no implicit extension')
    return 1500.


def saved_viewport_archive_review(before,after,actor,before_tensor,after_tensor):
    """COPY-only camera review for ARCHIVING, never a live notice exemption.

    Normalize only existing typed viewport properties, the ONE camera target,
    and customLayerData.cameraSettings.boundCamera. Everything else must pass
    the original lossless Session and exact actor physics-writeback proofs.
    Old route HOLD/postflight errors remain immutable; no drive is admitted.
    """
    import math
    from pxr import Sdf
    suffix=lambda text:'.usda' if text.lstrip().startswith('#usda') else '.sdf'
    layers={k:Sdf.Layer.CreateAnonymous('VIEWPORT_ARCHIVE_COPY'+suffix(after[k])) for k in ('Root','Session')}
    baseline={k:Sdf.Layer.CreateAnonymous('VIEWPORT_ARCHIVE_BASE'+suffix(before[k])) for k in ('Root','Session')}
    for k in layers:
        if not layers[k].ImportFromString(after[k]) or not baseline[k].ImportFromString(before[k]):
            raise ValueError('Exact saved layer COPIES required')
    product='/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0'
    cameras=('/OmniverseKit_Top','/OmniverseKit_Persp');reviews=[]
    fields={product+'.viewPickingId',*(c+'.'+a for c in ('/OmniverseKit_Persp',) for a in (
        'horizontalAperture','verticalAperture','xformOp:transform','xformOp:translate',
        'xformOp:rotateXYZ','omni:kit:centerOfInterest'))}
    for key in ('Root','Session'):
        for row in retained.shared.layer_field_differences(before[key],after[key]):
            path=row['path'];field=row['field']
            if key=='Root' and path=='/' and field=='customLayerData':
                old=copy.deepcopy(baseline[key].customLayerData);new=copy.deepcopy(layers[key].customLayerData)
                a=old.get('cameraSettings',{}).get('boundCamera');b=new.get('cameraSettings',{}).get('boundCamera')
                if a not in cameras or b not in cameras:raise ValueError('Only known saved viewport camera binding may be reviewed')
                new['cameraSettings']['boundCamera']=a
                if new!=old:raise ValueError('Non-camera Root customLayerData changed; no archive rebaseline')
                layers[key].customLayerData=new
            elif key=='Root' and path==product+'.camera' and field=='targetPaths':
                a=baseline[key].GetRelationshipAtPath(path);b=layers[key].GetRelationshipAtPath(path)
                if (a is None or b is None or baseline[key].GetPrimAtPath(product).typeName!='RenderProduct'
                        or layers[key].GetPrimAtPath(product).typeName!='RenderProduct'
                        or any(list(v.targetPathList.explicitItems) not in [[Sdf.Path(c)] for c in cameras] for v in (a,b))):
                    raise ValueError('Existing typed viewport RenderProduct camera target required')
                if not a.GetInfo('targetPaths').isExplicit or not b.GetInfo('targetPaths').isExplicit:
                    raise ValueError('Only existing explicit viewport camera target lists may be reviewed')
                b.targetPathList.explicitItems=list(a.targetPathList.explicitItems)
            elif path in fields and field=='default':
                a=baseline[key].GetAttributeAtPath(path);b=layers[key].GetAttributeAtPath(path)
                if a is None or b is None or a.typeName!=b.typeName or not a.HasInfo('default') or not b.HasInfo('default'):
                    raise ValueError('Existing SAME typed saved viewport defaults required')
                if path.endswith('.viewPickingId'):
                    if str(a.typeName)!='uint64' or any(type(v) is not int or not 0<=v<2**64 for v in (a.default,b.default)):
                        raise ValueError('Exact uint64 viewport picking ID required')
                else:
                    def finite(value):
                        if type(value) in (int,float):return math.isfinite(value)
                        try:return all(finite(v) for v in value)
                        except TypeError:return False
                    if not all(finite(v) for v in (a.default,b.default)):raise ValueError('Finite original camera defaults required')
                b.SetInfo('default',a.GetInfo('default'))
            else:continue
            reviews.append(dict(layer=key,**row))
    normalized={k:layer.ExportToString() for k,layer in layers.items()}
    display=retained.canonical_display_only_review(before['Session'],normalized['Session'])
    physics=retained.recorded_physics_writeback_review(dict(Root=before['Root'],Session=normalized['Session']),normalized,
        before_tensor,after_tensor,actor)
    return dict(exact_saved_viewport_changes=reviews,remaining_Session_review=display,
        remaining_exact_physics_writeback_review=physics,anonymous_copies_only=True,
        live_camera_or_notice_policy_NOT_changed=True,old_HOLD_NOT_promoted=True,actual_route_admission=False)


def saved_route_boundary_review(result_path):
    """Review the saved attempt WITHOUT promoting HOLD or rerunning physics."""
    import zipfile
    result_path=Path(result_path).resolve();raw=retained.read(result_path)
    path=Path(raw['protocol']);packet=retained.read(path);output=result_path.parent
    replanning='stopped_replanning' in packet
    if replanning:
        from hospital_sensor_navigation import checked_h5_replanning_scope
        scope=checked_h5_replanning_scope(packet)
        if (raw.get('contract')!=scope or raw.get('method')!=packet['method']
                or raw.get('scenario')!=packet['scenario']
                or type(raw.get('actual_Play_calls')) is not int or not 1<=raw['actual_Play_calls']<=2
                or type(raw.get('new_replanning_requests')) is not int or not 0<=raw['new_replanning_requests']<=1
                or raw.get('additional_Stop_calls')!=0 or raw.get('automatic_retry') is not False):
            raise ValueError('Own bounded H5 stopped-replanning archive required; no retry or outcome promotion')
    binding_name='replanning_sdk_binding.json' if replanning else 'sdk_binding.json'
    if (result_path!=output/'route_readback.json' or path!=output/binding_name
            or raw['protocol_sha256']!=retained.runtime.task.sha(path)
            or raw['scenario'] not in tuple(TASK_SCENARIOS.values()) or raw['new_route_trials']!=1
            or raw['checks'].get('actual_Pause_confirmed') is not True
            or raw['checks'].get('original_gains_targets_restored') is not True
            or raw['trace_sha256']!=retained.runtime.task.sha(raw['trace'])):
        raise ValueError('Own consumed route with exact trace/Pause/restoration required')
    final=raw['actual_final_readback'];state=raw['actual_final_timeline']
    if (not final['stable_PAUSED_tensor'] or final['timeline']!=state or state['playing'] or state['stopped']):
        raise ValueError('Own actual stable PAUSED route boundary required')
    archive=output/('replanning_sources.zip' if replanning else 'route_sources.zip')
    lifecycle.verify_workflow_sources(archive,packet['input_sha256'])
    evolving={Path(__file__).resolve(),Path(lifecycle.__file__).resolve(),Path(retained.__file__).resolve(),
        ROOT/'studies/IJCAS/scripts/hospital_mission_execution.py',
        ROOT/'studies/IJCAS/scripts/audit_hospital_paper_readiness.py',
        ROOT/'studies/IJCAS/scripts/hospital_sensor_navigation.py',
        ROOT/'studies/IJCAS/scripts/hospital_mission_planning.py',
        ROOT/'studies/IJCAS/scripts/plan_hospital_h3_native.py',
        ROOT/'studies/IJCAS/scripts/screen_hospital_h4_common_executor.py',
        ROOT/'src/patient_initial_stability/test/test_hospital_paper_protocol.py',
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py'}
    history=[]
    with zipfile.ZipFile(archive) as frozen:
        for name,digest in packet['input_sha256'].items():
            source=Path(name)
            if retained.runtime.task.sha(source)==digest:continue
            member=source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else None
            if source not in evolving or member is None or hashlib.sha256(frozen.read(member)).hexdigest()!=digest:
                raise ValueError('Original physical/data/evidence dependency changed: '+name)
            if source==Path(lifecycle.motion.__file__).resolve():
                lifecycle.reviewed_mission_schema_extension(frozen.read(member),source.read_bytes())
            history.append(dict(path=name,executed_sha256=digest,archive=str(archive),member=member))
    static=retained.read(packet['original_static_record']);before=static['actual_final_readback']['layer_texts']
    after=final['layer_texts']
    try:
        display=retained.canonical_display_only_review(before['Session'],after['Session'])
        writeback=retained.recorded_physics_writeback_review(dict(Root=before['Root'],Session=after['Session']),after,
            packet['required_tensor_snapshot'],final['tensor'],packet['actor'])
    except ValueError:
        if raw['checks'].get('complete_postflight_verified') is not False:raise
        display=saved_viewport_archive_review(before,after,packet['actor'],packet['required_tensor_snapshot'],final['tensor'])
        writeback=display['remaining_exact_physics_writeback_review']
    return dict(status='SAVED_ROUTE_BOUNDARY_COPY_REVIEW_NOT_A_COMPLETED_ROUTE_OR_NEW_LIVE_GET',
        original_result=str(result_path),original_result_sha256=retained.runtime.task.sha(result_path),
        original_verdict=raw['verdict'],original_error=raw.get('error'),display_review=display,
        exact_recorded_actor_writeback=writeback,historical_source_resolution=history,
        actual_route_admission=False,new_physical_trials=0,original_HOLD_FAIL_never_promoted=True)


async def inspect_route_final_boundary(result_path):
    """GET ONLY all4997 rows,145 settings and SAME final tensor; no actuator.

    A diagnostic receipt can support the next archive, not complete the
    interrupted route or resume its consumed dispatch. No Play/Stop/Pause,
    source construction, cooking, target/gain/pose or display write exists.
    """
    result_path=Path(result_path).resolve();target=result_path.parent/'final_boundary_GET.json'
    if target.exists():return dict(status='EXISTING_GET_RETAINED_NOT_A_CURRENT_POLL',saved=str(target))
    offline=saved_route_boundary_review(result_path);raw=retained.read(result_path)
    packet=retained.read(raw['protocol']);final=raw['actual_final_readback'];static=retained.read(packet['original_static_packet'])
    context=retained.static_binding(static,Path(packet['original_static_packet']),Path(packet['original_static_record']).parent,
        Path(packet['original_preparation']))
    context.stationary.RESULT=Path(packet['original_static_record']);context.stationary.TRACE=Path(packet['original_static_trace'])
    get_packet=dict(packet,required_PAUSED_timeline=raw['actual_final_timeline'],required_tensor_snapshot=final['tensor'],
        reviewed_display_boundary=dict(reference_Session_text=final['layer_texts']['Session'],
            unchanged_Root_sha256=hashlib.sha256(final['layer_texts']['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY))
    sdk=retained.runtime.live_sdk();stage=sdk.omni.usd.get_context().get_stage()
    current=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    archive_view=saved_viewport_archive_review(final['layer_texts'],current,packet['actor'],final['tensor'],final['tensor'])
    get_packet['reviewed_display_boundary'].update(reference_Session_text=current['Session'],
        unchanged_Root_sha256=hashlib.sha256(current['Root'].encode()).hexdigest())
    port=retained.RetainedZeroSettleSDKPort(sdk,get_packet,context=context)
    if port.timeline_state()!=raw['actual_final_timeline']:raise ValueError('Exact saved current PAUSED timeline required; no restoration')
    simulation=sdk.SimulationManager.get_physics_sim_view()
    if simulation is None:raise ValueError('Existing physics view required; no initialization')
    port.tensor=simulation.create_articulation_view(packet['actor']+'/base_link')
    snapshot=retained.shared.preparation.old.tensor_snapshot(port.tensor)
    if snapshot!=final['tensor']:raise ValueError('Same saved actual body/all12joints/all13links/gains/targets required')
    port.check_display_layers();environment=await port.environment();port.check_display_layers()
    if (port.timeline_state()!=raw['actual_final_timeline']
            or retained.shared.preparation.old.tensor_snapshot(port.tensor)!=snapshot or port.adapter is not None):
        raise ValueError('Read-only final-state transaction changed; no archive admission')
    layers=dict(Root=port.stage.GetRootLayer().ExportToString(),Session=port.stage.GetSessionLayer().ExportToString())
    result=dict(status='SAME_ROUTE_FINAL_BOUNDARY_ENVIRONMENT_VERIFIED_GET_ONLY_NOT_ROUTE_COMPLETION',
        original_result=str(result_path),original_result_sha256=retained.runtime.task.sha(result_path),
        original_verdict=raw['verdict'],actor=packet['actor'],timeline=port.timeline_state(),actual_tensor_snapshot=snapshot,
        actual_layer_texts=layers,all4997_rows145_source_settings_physics=environment,display_audits=port.display_audits,
        saved_boundary_review=offline,saved_to_current_viewport_archive_review=archive_view,
        no_timeline_actuator_scene_or_new_trial=True,actual_route_admission=False)
    result['GET_source_sha256']={str(Path(p).resolve()):retained.runtime.task.sha(p)
        for p in (__file__,lifecycle.__file__,retained.__file__)}
    retained.evidence.write_exclusive(target,result)
    print('[H6 FINAL GET]',result['status'],'original verdict:',raw['verdict'],'rows:',environment['row_count'],flush=True)
    return result


def saved_route_preparation_handoff(result_path):
    """One canonical OFFLINE view of a consumed route + actual complete GET."""
    result_path=Path(result_path).resolve();review=saved_route_boundary_review(result_path)
    raw=retained.read(result_path);packet=retained.read(raw['protocol']);folder=result_path.parent
    get_path=folder/'final_boundary_GET.json';get_archive=folder/'final_boundary_GET_sources.zip';actual=retained.read(get_path)
    lifecycle.verify_workflow_sources(get_archive,actual['GET_source_sha256'])
    final=raw['actual_final_readback'];prep=retained.read(raw['preparation_record']);snapshot=final['tensor']
    if (actual['status']!='SAME_ROUTE_FINAL_BOUNDARY_ENVIRONMENT_VERIFIED_GET_ONLY_NOT_ROUTE_COMPLETION'
            or actual['original_result']!=str(result_path) or actual['original_result_sha256']!=retained.runtime.task.sha(result_path)
            or actual['original_verdict']!=raw['verdict'] or actual['actor']!=packet['actor']
            or actual['timeline']!=raw['actual_final_timeline'] or actual['actual_tensor_snapshot']!=snapshot
            or actual['all4997_rows145_source_settings_physics']!=dict(row_count=4997,
                sha256=prep['protected_environment_sha256'],all145_source_settings_and_critical_environment_verified=True)
            or not actual['no_timeline_actuator_scene_or_new_trial'] or actual['actual_route_admission'] is not False):
        raise ValueError('Own complete actual final GET/same restored tensor/environment required')
    archive_review=saved_viewport_archive_review(final['layer_texts'],actual['actual_layer_texts'],packet['actor'],snapshot,snapshot)
    if 'saved_to_current_viewport_archive_review' in actual:
        if actual['saved_to_current_viewport_archive_review']!=archive_review:
            raise ValueError('Exact captured viewport archive review required; no replacement')
    elif actual['actual_layer_texts']['Root']!=final['layer_texts']['Root']:
        raise ValueError('Legacy final GET requires exact saved Root bytes')
    baseline=prep['expected_nominal_motor_baseline']
    if (len(snapshot['names'])!=12 or len(snapshot['link_names'])!=13 or snapshot['names']!=baseline['joint_names']
            or any(snapshot['gains'][k]!=baseline[k] for k in ('kp','kd','max_efforts'))
            or snapshot['joint_position_targets_rad']!=[0.]*12 or snapshot['joint_velocity_targets_rad_s']!=[0.]*12):
        raise ValueError('Same final Root/all12 original settings/zero targets/all13 links required')
    pins=dict(packet['input_sha256'])
    for row in review['historical_source_resolution']:pins.pop(row['path'])
    route_archive=folder/('replanning_sources.zip' if 'stopped_replanning' in packet else 'route_sources.zip')
    files=(result_path,Path(raw['protocol']),Path(raw['trace']),get_path,get_archive,route_archive,
        Path(raw['preparation_record']),Path(__file__).resolve(),*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    import json
    revision=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()[:16]
    return dict(status='EXACT_SAVED_ROUTE_AND_COMPLETE_FINAL_GET_FOR_NEW_ARCHIVE_NOT_ROUTE_COMPLETION',
        source_result=str(result_path),source_result_sha256=retained.runtime.task.sha(result_path),
        source_GET=str(get_path),source_GET_sha256=retained.runtime.task.sha(get_path),
        preparation_record=raw['preparation_record'],source_archive=str(folder/'next_cohort_handoff'/revision/'sources.zip'),
        original_verdict=raw['verdict'],original_route_HOLD_FAIL_not_promoted=True,new_live_actions=0,
        actual_layer_texts=actual['actual_layer_texts'],same_stage=True,Root_Session_unchanged_during_GET=True,
        SDK_GET=dict(actor=packet['actor'],timeline=actual['timeline'],actual_tensor_snapshot=snapshot,
            tensor_stable_during_GET=True,observed_hold_confirmation=False,original_motor_baseline_backend_readback_verified=True),
        legacy_field_view_is_saved_actual_GET_NOT_a_new_static_or_drive_admission=True,input_sha256=pins)


def export_route_preparation_handoff(result_path):
    value=saved_route_preparation_handoff(result_path);target=Path(value['source_archive']).with_name('handoff.json')
    if target.exists():
        if retained.read(target)!=value:raise ValueError('Original handoff retained; no overwrite')
    else:retained.evidence.write_exclusive(target,value)
    archive=Path(value['source_archive'])
    if not archive.exists():lifecycle.freeze_workflow_sources(archive,value['input_sha256'])
    else:lifecycle.verify_workflow_sources(archive,value['input_sha256'])
    return target


def held_static_adapter(result_path):
    """Reproduce raw acceptance, then expose existing static-port field names.

    Derived fields come only from the own frozen preparation and saved final
    SDK snapshot. The original report/trace bytes are never edited or replaced.
    A HOLD, incomplete postflight, changed source or restoration is rejected.
    """
    result_path=Path(result_path).resolve();raw=retained.read(result_path)
    replay=lifecycle.replay_held_static_evidence(result_path)
    prep=retained.read(raw['preparation_record']);packet=retained.read(raw['protocol'])
    final=raw['actual_final_readback'];post=raw['postflight'];snapshot=final['tensor']
    if (raw['verdict']!='PASS' or replay['verdict']!='PASS' or not all(raw['checks'].values())
            or raw.get('error') or raw.get('postflight_error')
            or not final['stable_PAUSED_tensor'] or final['timeline']!=raw['actual_final_timeline']
            or raw['actual_final_timeline']['playing'] or raw['actual_final_timeline']['stopped']
            or snapshot!=post['actual_post_held_static_tensor_snapshot']
            or prep['new_robot_root']!=packet['actor'] or prep['simulation_mass_kg']!=265.
            or len(snapshot['names'])!=12 or len(snapshot['link_names'])!=13
            or len(prep['source_reference']['fixed_configuration_rows'])!=145
            or raw['checks']['original_gains_targets_restored'] is not True
            or result_path!=Path(raw['protocol']).parent/'held_static_readback.json'):
        raise ValueError('Own complete raw-replayed held-static PASS and restored PAUSED tensor required')
    derived=copy.deepcopy(raw)
    derived.update(run_id=packet['run_id'],start_xyyaw=prep['start_xyyaw'],simulation_mass_kg=prep['simulation_mass_kg'],
        runtime_anonymous_source_reference=prep['source_reference'],runtime_motor_baseline=prep['expected_nominal_motor_baseline'],
        physics_preparation_snapshot=prep,actual_post_static_tensor_snapshot=snapshot,
        environment_after_static=prep['critical_environment'],
        representation_only_static_binding=dict(source=str(result_path),source_sha256=retained.runtime.task.sha(result_path),
            original_verdict=raw['verdict'],new_trials=0,
            held_static_NOT_relabelled_as_released_static=True,
            static_schema_aliases_are_NOT_new_physical_evidence=True))
    return prep,derived,replay


def checked_h3_task_binding(path):
    """Read-only task/schema boundary; never infer admission from a plan."""
    path=Path(path).resolve();task=retained.read(path)
    common=task.get('version') in ('UNIFIED1330_ORIGINAL_TASK_CURRENT_ENVIRONMENT_BINDING_v1',
        'UNIFIED1330_REVIEWED_COMMON_WALL_TASK_BINDING_v2')
    wall=checked_common_wall_budget(task)
    if ((wall==1500.)!=(task.get('version')=='UNIFIED1330_REVIEWED_COMMON_WALL_TASK_BINDING_v2')):
        raise ValueError('Reviewed wall allowance requires its explicit new task-binding version')
    key=task.get('scenario','').split('_')[0]
    valid_common=(common and key in TASK_SCENARIOS and task['scenario']==TASK_SCENARIOS[key]
        and task.get('approval')==REMAINING_NO_NAV2_APPROVAL and task.get('geometry_profile') is not None
        and task.get('simulation_budget_s')==TASK_BUDGETS[key])
    if ((not valid_common and (task.get('version')!='H3_ORIGINAL_TASK_CURRENT_ENVIRONMENT_CONTINUOUS_BINDING_v1'
            or task.get('scenario')!='H3_doorway_immediate_turn'
            or task.get('approval')!=H3_CONTINUOUS_APPROVAL
            or task.get('simulation_budget_s')!=300.))
            or task.get('automatic_retry') is not False
            or task.get('maximum_wall_time_s')!=wall or task.get('use_for_final_statistics') is not False):
        raise ValueError('Exact approved unchanged H3 whole-task binding required')
    retained.runtime.task.verify_pins(task['input_sha256'])
    g=retained.read(task['geometry_file']);original=retained.read(task['original_task_geometry'])
    current=retained.read(retained.GEOMETRY)
    if (g['scenario']!=task['scenario'] or any(g[k]!=original[k] for k in ('start_xyyaw','goal_xyyaw','local_goal_xyyaw'))
            or any(g[k]!=current[k] for k in ('occupancy_grid','planning_occupancy_digest',
                'same_three_authored_support_rectangles','full_body_rectangle_m','full_body_clearance_m'))):
        raise ValueError('Original H3 poses and SAME current complete environment/map required')
    for method,plan_path in task['plans'].items():
        plan=retained.read(plan_path);lifecycle.motion.validate_mission_plan(plan,g)
        if plan.get('geometry_profile')!=task.get('geometry_profile'):
            raise ValueError('Same explicit geometry profile for every admitted task plan required')
        audit=retained.read(task['plan_audits'][method])
        if audit['verdict']!='PASS' or audit['plan_sha256']!=retained.runtime.task.sha(plan_path):
            raise ValueError('Own original-SAT repropagation on current full map required')
    return task


def export_unified_task_binding(planning_folder,*,model_screen=None,common_wall_budget_review=None):
    """Same current map/old task poses, NEW planning and source-pinned common runtime."""
    import json,shutil
    from types import FunctionType
    import audit_hospital_h1_plan as independent
    from audit_hospital_paper_readiness import unified_wheelbase_connections
    folder=Path(planning_folder).resolve();summary=folder/'summary.json';raw=retained.read(summary)
    original_path=Path(raw['geometry_manifest']);g=retained.read(original_path)
    key=g['scenario'].split('_')[0]
    if (key not in TASK_SCENARIOS or g['scenario']!=TASK_SCENARIOS[key]
            or raw.get('fresh_planning_on_exact_recorded_common_map') is not True
            or raw.get('geometry_profile') is None):raise ValueError('Own unified current-map task planning required')
    wall=900.
    if common_wall_budget_review is not None:
        wall=checked_common_wall_budget(dict(scenario=g['scenario'],simulation_budget_s=TASK_BUDGETS[key],
            maximum_wall_time_s=1500.,common_wall_budget_review=common_wall_budget_review))
    connections=unified_wheelbase_connections(raw['geometry_profile']);function=independent.audit
    audit_fn=FunctionType(function.__code__,dict(function.__globals__,propagate=connections.python.propagate),
        function.__name__,function.__defaults__,function.__closure__)
    pins={str(p.resolve()):retained.runtime.task.sha(p) for p in
        (summary,original_path,Path(retained.GEOMETRY),Path(g['occupancy_grid']),Path(__file__),*lifecycle.host_source_files())}
    model_path=Path(model_screen).resolve() if model_screen is not None else folder/'executor_model_screen/model_screen.json';model=None
    proposed=retained.read(folder/'P_dynamics_full_envelope.json')
    if proposed['plan_success']:
        model=retained.read(model_path)
        if not model.get('both_nominal_resolutions_pass'):raise ValueError('Own proposed common-executor MODEL must PASS before binding')
        retained.runtime.task.verify_pins(model['input_sha256']);retained.runtime.task.verify_pins(model['output_sha256'])
        modeled=retained.read(model['plan'])
        if any(modeled.get(k)!=proposed.get(k) for k in ('selected_actions','selected_states','geometry_profile','planning_occupancy_digest')):
            raise ValueError('Exact own proposed plan MODEL required')
        pins.update(model['input_sha256']);pins.update(model['output_sha256']);pins[str(model_path)]=retained.runtime.task.sha(model_path)
    for method in METHODS:
        p=folder/(method+'.json');pins[str(p)]=retained.runtime.task.sha(p)
    stamp_input=pins if common_wall_budget_review is None else dict(input_sha256=pins,
        common_wall_budget_review=common_wall_budget_review)
    stamp=hashlib.sha256(json.dumps(stamp_input,sort_keys=True).encode()).hexdigest()[:16]
    output=retained.GATE/('hospital_'+key.lower()+'_experiments')/'task_bindings'/stamp
    target=output/'task_binding.json'
    if target.exists():checked_h3_task_binding(target);return target
    if output.exists():raise ValueError('Partial task binding retained; no overwrite')
    output.mkdir(parents=True)
    shutil.copyfile(g['occupancy_grid'],output/'planning_occupancy.npz')
    query=copy.deepcopy(retained.read(retained.H6/'static_protocol.json'))
    query.update(scenario=g['scenario'],start_xyyaw=g['start_xyyaw'],goal_xyyaw=g['goal_xyyaw'],
        sampled_cooked_support_queries=retained.static_query_specs(g,query))
    query_path=output/'static_query_contract.json';retained.evidence.write_exclusive(query_path,query)
    plans={};audits={}
    for method in METHODS:
        value=retained.read(folder/(method+'.json'))
        if not value['plan_success']:continue
        value=dict(value,physical_execution_verified=False,Nav2_execution_verified=False,
            integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True)
        path=output/(method+'.json');retained.evidence.write_exclusive(path,value)
        checked=audit_fn(path);ap=output/(method+'_current_map_audit.json');retained.evidence.write_exclusive(ap,checked)
        if checked['verdict']!='PASS':raise ValueError('Exact independent original SAT replay must PASS')
        lifecycle.motion.validate_mission_plan(value,g);plans[method]=str(path);audits[method]=str(ap)
    for p in (output/'planning_occupancy.npz',query_path,*map(Path,plans.values()),*map(Path,audits.values())):pins[str(p)]=retained.runtime.task.sha(p)
    task=dict(version='UNIFIED1330_ORIGINAL_TASK_CURRENT_ENVIRONMENT_BINDING_v1',scenario=g['scenario'],
        approval=REMAINING_NO_NAV2_APPROVAL,geometry_file=str(original_path),static_query_contract_file=str(query_path),
        original_task_geometry=str(original_path),original_planning_summary=str(summary),plans=plans,plan_audits=audits,
        simulation_budget_s=TASK_BUDGETS[key],maximum_wall_time_s=wall,geometry_profile=raw['geometry_profile'],
        all_methods_share_controller_environment_and_acceptance=True,automatic_retry=False,use_for_final_statistics=False,
        input_sha256=pins)
    if common_wall_budget_review is not None:
        task.update(version='UNIFIED1330_REVIEWED_COMMON_WALL_TASK_BINDING_v2',
            common_wall_budget_review=copy.deepcopy(common_wall_budget_review))
    retained.evidence.write_exclusive(target,task);checked_h3_task_binding(target);return target


def export_h3_task_binding(planning_folder):
    """Bind native H3 references to the CURRENT map, without a scene edit.

    Original search data, all failed plans, and the old H3 map remain intact.
    Repropagate every action through the original model/SAT on the full
    already-adopted map. This is development compatibility, NOT a fresh
    end-to-end planning-time comparison or a measured whole-task success.
    """
    import json,shutil
    import audit_hospital_h1_plan as independent
    from hospital_geometry_tools import pose_collision
    from hospital_h4_cohort_preparation_v4 import support_area_missing
    planning_folder=Path(planning_folder).resolve();summary=planning_folder/'summary.json'
    raw=retained.read(summary)
    original_path=Path(raw['geometry_manifest']);original=retained.read(original_path)
    if (raw['scenario']!='H3_doorway_immediate_turn'
            or not raw.get('continuous_task_search_binding',{}).get('original_H1_forward_only_call_not_used')):
        raise ValueError('Explicit native continuous-task search, not measured component stitching, required')
    profile=raw.get('geometry_profile');model_receipt=None
    if profile is not None:
        from types import FunctionType
        from audit_hospital_paper_readiness import unified_wheelbase_connections
        connections=unified_wheelbase_connections(profile)
        function=independent.audit
        independent_audit=FunctionType(function.__code__,dict(function.__globals__,
            propagate=connections.python.propagate),function.__name__,function.__defaults__,function.__closure__)
        model_receipt=planning_folder/'executor_model_screen/model_screen.json'
        model=retained.read(model_receipt)
        if (raw.get('fresh_planning_on_exact_recorded_common_map') is not True
                or model.get('both_nominal_resolutions_pass') is not True
                or model.get('integration_steps_s')!=[.020,.010]
                or model.get('actual_motion_authorized') is not False
                or model.get('new_PhysX_trials')!=0):
            raise ValueError('Own current-map unified planning and two-resolution common-executor MODEL PASS required')
        retained.runtime.task.verify_pins(model['input_sha256']);retained.runtime.task.verify_pins(model['output_sha256'])
        modeled=retained.read(model['plan']);proposed=retained.read(planning_folder/'P_dynamics_full_envelope.json')
        if any(modeled.get(k)!=proposed.get(k) for k in ('geometry_profile','selected_actions','selected_states',
                'start_world_xyyaw','goal_world_xyyaw','planning_occupancy_digest','rigid_body_parameter_bounds')):
            raise ValueError('MODEL screen must belong to the exact own fresh P plan, not another candidate')
    else:independent_audit=independent.audit
    pins={str(p):retained.runtime.task.sha(p) for p in (summary,original_path,Path(retained.GEOMETRY))}
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in (*lifecycle.host_source_files(),Path(__file__))})
    if profile is not None:
        pins.update(model['input_sha256']);pins.update(model['output_sha256'])
        pins[str(model_receipt)]=retained.runtime.task.sha(model_receipt)
    current=retained.read(retained.GEOMETRY);grid_path=Path(current['occupancy_grid'])
    pins[str(grid_path)]=retained.runtime.task.sha(grid_path)
    rows={}
    for method in METHODS:
        source=planning_folder/(method+'.json')
        value=retained.read(source);pins[str(source)]=retained.runtime.task.sha(source)
        if value['plan_success']:rows[method]=(source,value)
    key=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()[:16]
    folder=retained.GATE/'hospital_h3_experiments'/'task_bindings'/key;target=folder/'task_binding.json'
    if target.exists():checked_h3_task_binding(target);return target
    if folder.exists():raise ValueError('Partial H3 task export retained; no overwrite')
    g=copy.deepcopy(current)
    g.update(scenario=original['scenario'],start_xyyaw=original['start_xyyaw'],goal_xyyaw=original['goal_xyyaw'],
        local_goal_xyyaw=original['local_goal_xyyaw'],original_H3_task_geometry=str(original_path),
        current_H6_environment_retained_NO_new_scene_edit=True,
        different_destination_door_geometry_is_NOT_an_H3_doorway_change=True,
        actual_motion_authorized=False,use_for_final_statistics=False)
    with np.load(grid_path) as grid:
        for role in ('start','goal'):
            pose=g[role+'_xyyaw']
            if (pose_collision(grid['occupied'],pose,grid['origin_xy_m'],float(grid['resolution_m']),clearance=.010)
                    or support_area_missing(pose,g['same_three_authored_support_rectangles'])>1e-9):
                raise ValueError('Unchanged H3 endpoint not safe on current complete map/support')
    folder.mkdir(parents=True,exist_ok=False)
    geometry_path=folder/'task_geometry.json';retained.evidence.write_exclusive(geometry_path,g)
    shutil.copyfile(grid_path,folder/'planning_occupancy.npz')
    query=copy.deepcopy(retained.read(retained.H6/'static_protocol.json'))
    query.update(scenario=g['scenario'],start_xyyaw=g['start_xyyaw'],goal_xyyaw=g['goal_xyyaw'],
        sampled_cooked_support_queries=retained.static_query_specs(g,query))
    query_path=folder/'static_query_contract.json';retained.evidence.write_exclusive(query_path,query)
    plans={};audits={};blocked={}
    for method,(source,value) in rows.items():
        value=copy.deepcopy(value)
        value.update(planning_occupancy_digest=g['planning_occupancy_digest'],
            integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True,
            original_H3_search_result=str(source),original_H3_search_result_sha256=retained.runtime.task.sha(source),
            current_environment_action_repropagation_NOT_new_search_or_timing=True,
            actual_motion_authorized=False,physical_execution_verified=False,Nav2_execution_verified=False)
        if profile is not None:
            value.update(geometry_manifest=str(geometry_path),geometry_manifest_sha256=retained.runtime.task.sha(geometry_path),
                current_environment_action_repropagation_NOT_new_search_or_timing=False,
                current_map_fresh_planning_time_preserved=True)
        plan=folder/(method+'.json');retained.evidence.write_exclusive(plan,value)
        audit=independent_audit(plan);audit_path=folder/(method+'_current_map_audit.json')
        retained.evidence.write_exclusive(audit_path,audit)
        if audit['verdict']=='PASS':
            lifecycle.motion.validate_mission_plan(value,g);plans[method]=str(plan);audits[method]=str(audit_path)
        else:blocked[method]=dict(audit=str(audit_path),reason='CURRENT_FULL_MAP_REPROPAGATION_NOT_PASS')
    files=[geometry_path,query_path,folder/'planning_occupancy.npz',*[Path(v) for v in plans.values()],
        *[Path(v) for v in audits.values()],*lifecycle.host_source_files(),Path(__file__).resolve()]
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    task=dict(version='H3_ORIGINAL_TASK_CURRENT_ENVIRONMENT_CONTINUOUS_BINDING_v1',scenario=g['scenario'],
        approval=H3_CONTINUOUS_APPROVAL,geometry_file=str(geometry_path),static_query_contract_file=str(query_path),
        original_task_geometry=str(original_path),original_planning_summary=str(summary),plans=plans,plan_audits=audits,
        blocked_plan_bindings=blocked,simulation_budget_s=300.,maximum_wall_time_s=900.,
        all_methods_share_controller_environment_and_acceptance=True,automatic_retry=False,use_for_final_statistics=False,
        no_actual_scene_or_timeline_access=True,new_physical_trials=0,input_sha256=pins)
    if profile is not None:task.update(geometry_profile=profile,own_unified_P_executor_model_screen=str(model_receipt))
    retained.evidence.write_exclusive(target,task);checked_h3_task_binding(target)
    return target


def h3_route_definition(static_result,method):
    """Own H3 static -> one route using exactly the shared final controller."""
    if method not in METHODS:raise ValueError('One original comparison method required')
    static_result=Path(static_result).resolve();prep,adapter,replay=held_static_adapter(static_result)
    static_packet=retained.read(adapter['protocol']);workflow=Path(static_packet['workflow_protocol'])
    workflow_raw=retained.read(workflow);task_path=Path(workflow_raw['task_binding_path']);task=checked_h3_task_binding(task_path)
    if (workflow_raw['user_approval']!=task['approval'] or workflow.parent!=static_result.parent.parent
            or workflow_raw['scenario']!=task['scenario'] or method not in task['plans']):
        raise ValueError('Own H3 fresh static/approved whole-task plan required')
    lifecycle.verify_workflow_sources(workflow.parent/'workflow_sources.zip',workflow_raw['input_sha256'])
    plan_path=Path(task['plans'][method]);plan=retained.read(plan_path)
    geometry_path=Path(static_packet['geometry_file']);geometry=retained.read(geometry_path)
    lifecycle.motion.validate_mission_plan(plan,geometry)
    if plan['start_world_xyyaw']!=prep['start_xyyaw']:raise ValueError('Never recenter the fixed H3 task start')
    output=workflow.parent/'route'/method;pins=dict(static_packet['input_sha256'])
    files=(static_result,Path(adapter['trace']),Path(adapter['protocol']),workflow,workflow.parent/'workflow_sources.zip',
        Path(adapter['preparation_record']),task_path,plan_path,geometry_path,Path(static_packet['grid_file']),
        Path(__file__).resolve(),*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    layers=adapter['actual_final_readback']['layer_texts']
    packet=dict(version='H3_FRESH_CONTINUOUS_SHARED_FINAL_CONTROLLER_ROUTE_v1',scenario=task['scenario'],method=method,
        output_directory=str(output),cohort=static_packet['cohort'],new_robot_root=static_packet['actor'],actor=static_packet['actor'],
        marker=static_packet['marker'],run_id='hospital_'+task['scenario'].split('_')[0].lower()+'_'+static_packet['cohort']+'_'+method,
        original_held_static_result=str(static_result),original_static_record=str(output/'same_static_schema_binding.json'),
        original_static_trace=adapter['trace'],plan_filename='frozen_plan.json',original_plan=str(plan_path),
        original_preparation=adapter['preparation_record'],original_static_packet=adapter['protocol'],
        geometry_file=str(geometry_path),grid_file=static_packet['grid_file'],environment_rows_file=static_packet['environment_rows_file'],
        required_PAUSED_timeline=adapter['actual_final_timeline'],required_tensor_snapshot=adapter['actual_post_static_tensor_snapshot'],
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],
            unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        simulation_budget_s=task['simulation_budget_s'],maximum_wall_time_s=task['maximum_wall_time_s'],
        maximum_Play_calls=1,maximum_Stop_reset_restaging_calls=0,maximum_route_trials=1,automatic_retry=False,
        actual_route_authorized=True,use_for_final_statistics=False,explicit_route_approval=task['approval'],
        common_tracking_policy=FINAL_TARGET_HOLD_POLICY,common_forward_tracking_floor_m_s=.010,
        common_reverse_tracking_floor_m_s=.020,
        common_lifecycle_contract=final_target_hold_lifecycle_contract(simulation_budget_s=task['simulation_budget_s']),
        raw_static_replay=replay,held_static_binding_is_representation_only=True,input_sha256=pins)
    if task.get('geometry_profile') is not None:
        packet.update(geometry_profile=task['geometry_profile'],
            common_lifecycle_contract=unified_geometry_lifecycle_contract(task['geometry_profile'],
                simulation_budget_s=task['simulation_budget_s'],wall_budget_s=checked_common_wall_budget(task)))
    if task.get('common_wall_budget_review') is not None:
        packet['common_wall_budget_review']=copy.deepcopy(task['common_wall_budget_review'])
    return packet,adapter,plan


def route_definition(static_result,method,*,approval=None):
    if approval in (H3_CONTINUOUS_APPROVAL,REMAINING_NO_NAV2_APPROVAL):
        return h3_route_definition(static_result,method)
    if approval not in (APPROVAL,REVALIDATION_APPROVAL,*TRACKING_FLOOR_APPROVALS,DIRECTIONAL_QUIET_STOP_APPROVAL,COMPLIANT_HOLD_APPROVAL,ITERATIVE_FIX_APPROVAL):raise ValueError('Actual approved common600s remaining-cell or explicit revalidation scope required')
    if method not in METHODS:raise ValueError('One declared H6 method required')
    compliant=approval==COMPLIANT_HOLD_APPROVAL
    stop_connection=approval==ITERATIVE_FIX_APPROVAL
    directional=approval==DIRECTIONAL_QUIET_STOP_APPROVAL or compliant or stop_connection
    candidate=approval in TRACKING_FLOOR_APPROVALS or directional
    if candidate and method!='P_dynamics_full_envelope':raise ValueError('This new user approval covers ONE P development route only')
    static_result=Path(static_result).resolve();prep,adapter,replay=held_static_adapter(static_result)
    static_packet=retained.read(adapter['protocol']);workflow=Path(static_packet['workflow_protocol'])
    workflow_raw=retained.read(workflow)
    if compliant:
        if workflow_raw.get('user_approval')!=COMPLIANT_HOLD_APPROVAL:raise ValueError('Own conditional compliant workflow approval required')
        checked_compliant_probe_condition(workflow_raw.get('prior_handoff_path'))
    if workflow.parent!=static_result.parent.parent:raise ValueError('Own source-cohort/static lineage required')
    lifecycle.verify_workflow_sources(workflow.parent/'workflow_sources.zip',workflow_raw['input_sha256'])
    if stop_connection:
        if workflow_raw.get('user_approval')!=ITERATIVE_FIX_APPROVAL:raise ValueError('Own changed-plan workflow authority required')
        plan_path,review_path=checked_stop_connection_model_condition()
        review=retained.read(review_path)
    else:
        review_path=PLANNING/'planning_review.json';review=retained.read(review_path)
        for mapping in ('input_sha256','output_sha256'):retained.runtime.task.verify_pins(review[mapping])
        plan_path=PLANNING/(method+'.json')
    plan=retained.read(plan_path)
    geometry_path=Path(static_packet['geometry_file']);geometry=retained.read(geometry_path)
    lifecycle.motion.validate_mission_plan(plan,geometry)
    if plan['start_world_xyyaw']!=prep['start_xyyaw']:raise ValueError('Never replace the fixed source start by a measured pose')
    output=workflow.parent/'route'/method
    pins=dict(static_packet['input_sha256'])
    files=(static_result,Path(adapter['trace']),Path(adapter['protocol']),workflow,workflow.parent/'workflow_sources.zip',
        Path(adapter['preparation_record']),review_path,plan_path,geometry_path,Path(static_packet['grid_file']),
        Path(__file__).resolve(),*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    layers=adapter['actual_final_readback']['layer_texts']
    packet=dict(version=VERSION,scenario='H6_integrated_transfer_mission',method=method,output_directory=str(output),
        cohort=static_packet['cohort'],new_robot_root=static_packet['actor'],actor=static_packet['actor'],marker=static_packet['marker'],
        run_id='hospital_h6_'+static_packet['cohort']+'_'+method,
        original_held_static_result=str(static_result),original_static_record=str(output/'same_static_schema_binding.json'),
        original_static_trace=adapter['trace'],plan_filename='frozen_plan.json',original_plan=str(plan_path),
        original_preparation=adapter['preparation_record'],original_static_packet=adapter['protocol'],
        geometry_file=str(geometry_path),grid_file=static_packet['grid_file'],environment_rows_file=static_packet['environment_rows_file'],
        required_PAUSED_timeline=adapter['actual_final_timeline'],required_tensor_snapshot=adapter['actual_post_static_tensor_snapshot'],
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        simulation_budget_s=600.,maximum_wall_time_s=900.,maximum_Play_calls=1,maximum_Stop_reset_restaging_calls=0,
        maximum_route_trials=1,automatic_retry=False,actual_route_authorized=True,use_for_final_statistics=False,
        explicit_route_approval=approval,common_lifecycle_contract=lifecycle.lifecycle_contract(simulation_budget_s=600.),
        raw_static_replay=replay,held_static_binding_is_representation_only=True,input_sha256=pins)
    if directional:
        packet.update(version='H6_ONE_DIRECTIONAL_QUIET_STOP_DEVELOPMENT_ROUTE_v1',
            common_tracking_policy=DIRECTIONAL_QUIET_STOP_POLICY,
            common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
            common_lifecycle_contract=directional_quiet_stop_lifecycle_contract(simulation_budget_s=600.),
            prior_comparison_controller_version_NOT_matched_to_this_development_trial=True,
            actual_frozen_plan_low_speed_profile_NOT_preserved=True)
        if stop_connection:packet.update(version='H6_ONE_NEUTRAL_STOP_CONNECTION_P_DEVELOPMENT_ROUTE_v1',
            common_tracking_policy=FINAL_TARGET_HOLD_POLICY,
            common_lifecycle_contract=final_target_hold_lifecycle_contract(simulation_budget_s=600.),
            neutral_stop_connection_MODEL_condition=str(review_path),
            changed_P_own_cached_terminal_plan_NOT_primary_matched_planning_runtime=True,
            original10000_hold_and_original8deg_s_bridge_restored_NOT_failed_compliant_candidate=True,
            original_planned615s_schedule_NOT_the10mm_s_development_executor_schedule=True)
        if compliant:packet.update(version='H6_ONE_CONDITIONAL_COMPLIANT_HOLD_P_DEVELOPMENT_ROUTE_v1',
            common_tracking_policy=COMPLIANT_HOLD_POLICY,
            common_lifecycle_contract=compliant_hold_lifecycle_contract(simulation_budget_s=600.),
            static_original10000_hold_IS_engineering_support_check_NOT_this1000_route_prehold=True)
    elif candidate:
        measured=approval==MEASURED_RESPONSE_FIX_APPROVAL
        packet.update(version='H6_ONE_COMMON_TRACKING_FLOOR_DEVELOPMENT_ROUTE_v1',
            common_tracking_policy=MEASURED_RESPONSE_FLOOR_POLICY if measured else TRACKING_FLOOR_POLICY,
            common_tracking_speed_floor_m_s=.020 if measured else .010,
            common_lifecycle_contract=(measured_response_lifecycle_contract if measured else tracking_floor_lifecycle_contract)(simulation_budget_s=600.),
            prior_comparison_controller_version_NOT_matched_to_this_development_trial=True,
            actual_frozen_plan_low_speed_profile_NOT_preserved=True)
    return packet,adapter,plan


def export_route(static_result,method,*,approval=None):
    packet,adapter,plan=route_definition(static_result,method,approval=approval)
    output=Path(packet['output_directory'])
    if output.exists():raise ValueError('Existing route packet/evidence retained; no overwrite or second definition')
    retained.evidence.write_exclusive(output/'same_static_schema_binding.json',adapter)
    retained.evidence.write_exclusive(output/packet['plan_filename'],plan)
    for path in (output/'same_static_schema_binding.json',output/packet['plan_filename']):
        packet['input_sha256'][str(path)]=retained.runtime.task.sha(path)
    path=output/'route_protocol.json';retained.evidence.write_exclusive(path,packet)
    lifecycle.freeze_workflow_sources(output/'route_sources.zip',packet['input_sha256'])
    return path


def load_route(path):
    path=Path(path).resolve();packet=retained.read(path)
    expected,adapter,plan=route_definition(packet['original_held_static_result'],packet['method'],approval=packet['explicit_route_approval'])
    output=Path(expected['output_directory'])
    for target,value in ((output/'same_static_schema_binding.json',adapter),(output/expected['plan_filename'],plan)):
        if retained.read(target)!=value:raise ValueError('Exact same raw static aliases/frozen original plan required')
        expected['input_sha256'][str(target)]=retained.runtime.task.sha(target)
    if packet!=expected or path!=output/'route_protocol.json':raise ValueError('Exact one-cell frozen continuation required')
    retained.runtime.task.verify_pins(packet['input_sha256'])
    lifecycle.verify_workflow_sources(output/'route_sources.zip',packet['input_sha256'])
    static=retained.read(packet['original_static_packet'])
    context=retained.static_binding(static,Path(packet['original_static_packet']),Path(packet['original_held_static_result']).parent,
        Path(packet['original_preparation']))
    context.stationary.RESULT=Path(packet['original_static_record']);context.stationary.TRACE=Path(packet['original_static_trace'])
    context.operations.replay_static=lambda _:held_static_adapter(packet['original_held_static_result'])
    # Existing port validates its exact version; all scope and provenance
    # checks above remain attached to this new explicit representation layer.
    binding=dict(packet,version=lifecycle.RetainedMissionSDKHost.VERSION)
    return context,packet,binding,plan,retained.read(packet['geometry_file'])


async def run_route_once(path,*,authorized=False):
    if authorized is not True:raise ValueError('Explicit one remaining route authority required before SDK or file access')
    context,packet,binding,plan,geometry=load_route(path)
    output=Path(packet['output_directory'])
    if any((output/name).exists() for name in ('single_route_reserved.json','route_readback.json','dispatch_error.json')):
        return dict(status='PRIOR_ROUTE_DISPATCH_RETAINED_NO_NEW_ACTION')
    retained.no_other_task()
    # The SDK host requires bytes equal to the supplied binding. Store that
    # separate typed-view packet; do not rewrite the user's frozen definition.
    bound_path=output/'sdk_binding.json'
    if not bound_path.exists():retained.evidence.write_exclusive(bound_path,binding)
    elif retained.read(bound_path)!=binding:raise ValueError('Existing SDK binding changed; no overwrite')
    sdk=retained.runtime.live_sdk()
    with np.load(packet['grid_file']) as grid:
        host_class=CompliantRetainedMissionSDKHost if packet.get('common_tracking_policy')==COMPLIANT_HOLD_POLICY else lifecycle.RetainedMissionSDKHost
        port=host_class(sdk,binding,bound_path,context=context,plan=plan,geometry=geometry,
            occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),resolution=float(grid['resolution_m']),authorized=True)
    controller=route_controller(packet,plan,geometry)
    return await lifecycle.run_lifecycle_once(port,controller,authorized=True,
        sdk_bridge_class=RetainFinalTargetsBridge if packet.get('common_tracking_policy')==FINAL_TARGET_HOLD_POLICY else
            HeldNeutralSlewBridge if packet.get('common_tracking_policy')==COMPLIANT_HOLD_POLICY else None)


def schedule_route(path,*,authorized=False):
    import builtins
    if authorized is not True:raise ValueError('Explicit one remaining route authority required')
    for name in (*retained.boundary.ACTIVE_TASKS,'ijcas_h6_latest_postabort_inspection'):
        task=getattr(builtins,name,None)
        if task is not None and not task.done():raise RuntimeError('Another trial/task active; no concurrent drive')
    context,packet,_,_,_=load_route(path)
    output=Path(packet['output_directory'])
    if any((output/name).exists() for name in ('single_route_reserved.json','route_readback.json','dispatch_error.json')):
        print('[H6 SEQUENCE] Existing route dispatch retained; NO new action.',flush=True);return None
    async def once():
        try:return await run_route_once(path,authorized=True)
        except Exception as failure:
            result=dict(status='ROUTE_DISPATCH_ERROR_NO_AUTOMATIC_RETRY',error=type(failure).__name__+': '+str(failure))
            if not (output/'dispatch_error.json').exists():retained.evidence.write_exclusive(output/'dispatch_error.json',result)
            print('[H6 SEQUENCE]',result,flush=True);return result
    task=asyncio.ensure_future(once());builtins.ijcas_hospital_latest_task_experiment=task
    print('[H6 SEQUENCE] ONE',packet['method'],'route scheduled; SAME static-PASS actor,600s/900s,NO Stop/reset/retry.',flush=True)
    return task


def _tracking_floor_candidate_implementation(value,*,physical_direction_conditioned=False,task_goal_xyyaw=None):
    """Explicit OFFLINE common candidate, never an adopted SDK policy.

    10mm/s is the existing terminal-pulse command, not an identified minimum
    viable plant speed. Preserve the frozen controller file and its default
    live binding. Change one tracking target in a private function only;
    reuse the exact original mission handoff, ramp, guards and terminal code.
    """
    import ast
    import inspect
    import textwrap
    import hospital_h4_path_execution as original
    if type(value) not in (int,float) or value not in (.010,.020):
        raise ValueError('Only the explicit prospective common10/20mm/s candidates are supported')
    if type(physical_direction_conditioned) is not bool or physical_direction_conditioned and value!=.020:
        raise ValueError('Explicit10mm/s forward20mm/s reverse candidate required')
    if task_goal_xyyaw is not None:
        if (len(task_goal_xyyaw)!=3 or not all(np.isfinite(v) for v in task_goal_xyyaw)
                or not physical_direction_conditioned):
            raise ValueError('Exact finite task goal and existing directional floor required')
        task_goal_xyyaw=tuple(float(v) for v in task_goal_xyyaw)
    old=lifecycle.motion._mission_tracking_implementation()
    native=original.common._v3['_v2']['_legacy']['H1TrackingController'].step
    source=textwrap.dedent(inspect.getsource(native));tree=ast.parse(source)
    matches=[]
    predicates=ast.dump(ast.parse('remaining <= .025 and math.hypot(pose[0]-self.reference.poses[-1][0], pose[1]-self.reference.poses[-1][1]) <= .05',mode='eval').body,include_attributes=False)
    target=ast.dump(ast.parse('min(action["target_speed_m_s"], math.sqrt(2*.02*max(.000225, remaining-.010)))',mode='eval').body,include_attributes=False)
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.dump(node.test,include_attributes=False)==predicates:
            matches.append((node,'test',node.test));node.test=ast.parse('_mission_handoff(self,pose,measured_speed,remaining)',mode='eval').body
        elif isinstance(node,ast.Assign) and ast.dump(node.value,include_attributes=False)==target:
            matches.append((node,'value',node.value))
            # The existing endpoint braking cap remains OUTSIDE the floor.
            # The existing ramp still limits command acceleration afterwards.
            floor='_mission_floor_for_direction(self)' if physical_direction_conditioned else '_mission_tracking_floor'
            node.value=ast.parse('min(max(action["target_speed_m_s"], '+floor+'), math.sqrt(2*.02*max(.000225, remaining-.010)))',mode='eval').body
        elif (task_goal_xyyaw is not None and isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='remaining'
                and ast.dump(node.value,include_attributes=False)==ast.dump(ast.parse(
                    'self.reference.arc[-1]-self.reference.progress',mode='eval').body,include_attributes=False)):
            matches.append((node,'value',node.value))
            node.value=ast.parse('_mission_task_remaining(self,pose,self.reference.arc[-1]-self.reference.progress)',mode='eval').body
    if sorted(field for _,field,_ in matches)!=(['test','value'] if task_goal_xyyaw is None else ['test','value','value']):
        raise ValueError('Exactly one original mission handoff and tracking target required')
    def directional_floor(tracker):
        if type(tracker.mission_travel_direction) is not int or tracker.mission_travel_direction not in (-1,1):
            raise ValueError('Actual declared physical leg direction required')
        return .010 if tracker.mission_travel_direction==1 else .020
    def task_remaining(tracker,pose,reference):
        if not tracker.mission_final_leg:return reference
        return max(reference,float(np.hypot(pose[0]-task_goal_xyyaw[0],pose[1]-task_goal_xyyaw[1])))
    namespace=dict(old.__globals__,_mission_tracking_floor=float(value),_mission_floor_for_direction=directional_floor,
        _mission_task_remaining=task_remaining)
    exec(compile(ast.fix_missing_locations(tree),'<OFFLINE_common_tracking_floor_candidate>','exec'),namespace)
    result=namespace[native.__name__]
    for node,field,before in matches:setattr(node,field,before)
    if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
        raise ValueError('Undo must recover entire original tracking AST')
    result.mission_binding=dict(old.mission_binding,
        feedback_gains_path_guards_and_all_other_statements_unchanged=False,
        AST_tracking_target_substitutions=1,common_tracking_speed_floor_m_s=float(value),
        original_endpoint_braking_cap_and_ramp_and_feedback_and_guards_unchanged=True,
        planned_low_speed_profile_is_NOT_preserved_or_repropagated=True,
        floor_NOT_an_identified_minimum_viable_PhysX_speed=True,
        explicit_OFFLINE_candidate_NOT_adopted_live_policy=True)
    if physical_direction_conditioned:
        result.mission_binding.update(physical_direction_conditioned=True,
            common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
            method_identifier_NOT_used_for_control_policy=True)
    if task_goal_xyyaw is not None:
        result.mission_binding.update(goal_distance_handoff_policy='H5_FINAL_REMAINING_MAX_REFERENCE_AND_ACTUAL_TASK_GOAL_v1',
            actual_task_goal_xyyaw=list(task_goal_xyyaw),additional_remaining_distance_expression_substitutions=1,
            original_braking_formula_and_speed_acceleration_limits_unchanged=True,
            original_endpoint_braking_cap_and_ramp_and_feedback_and_guards_unchanged=False,
            remaining_distance_is_max_reference_arc_and_measured_task_goal=True)
    return result


class CommonTrackingFloorCandidate(lifecycle.motion.MissionTrackingController):
    """Private common-model alternative, deliberately NOT wired into live runs."""
    def __init__(self,plan,geometry,*,simulation_budget_s,tracking_speed_floor_m_s=.010):
        if type(tracking_speed_floor_m_s) not in (int,float) or tracking_speed_floor_m_s!=.010:
            raise ValueError('Explicit prospective common10mm/s candidate required')
        candidate=_tracking_floor_candidate_implementation(tracking_speed_floor_m_s)
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.tracking_step=candidate


def checked_route_lifecycle_contract(packet):
    """Explicit selection, never infer a policy from a plan/method/outcome."""
    wall=checked_common_wall_budget(packet)
    if wall==1500. and (packet.get('geometry_profile') is None
            or packet.get('common_tracking_policy')!=FINAL_TARGET_HOLD_POLICY):
        raise ValueError('Reviewed wall allowance requires the explicit unchanged unified controller')
    directional_keys=('common_forward_tracking_floor_m_s','common_reverse_tracking_floor_m_s')
    policy=packet.get('common_tracking_policy');compliant=policy==COMPLIANT_HOLD_POLICY
    if policy==FINAL_TARGET_HOLD_POLICY:
        h3=(packet.get('method') in METHODS and ((packet.get('explicit_route_approval')==H3_CONTINUOUS_APPROVAL
            and packet.get('scenario')=='H3_doorway_immediate_turn') or
            (packet.get('explicit_route_approval')==REMAINING_NO_NAV2_APPROVAL
            and packet.get('scenario') in TASK_SCENARIOS.values()
            and packet.get('simulation_budget_s')==TASK_BUDGETS[packet['scenario'].split('_')[0]])))
        if ((not h3 and (packet.get('explicit_route_approval')!=ITERATIVE_FIX_APPROVAL or packet.get('method')!='P_dynamics_full_envelope'))
                or packet.get(directional_keys[0])!=.010 or packet.get(directional_keys[1])!=.020
                or 'common_tracking_speed_floor_m_s' in packet):
            raise ValueError('Explicit same-limits retained-final-target development protocol required')
        if packet.get('geometry_profile') is not None:
            if not h3:raise ValueError('Unified runtime profile is currently bound only to the explicit H3 task')
            return unified_geometry_lifecycle_contract(packet['geometry_profile'],simulation_budget_s=packet['simulation_budget_s'],
                wall_budget_s=wall)
        return final_target_hold_lifecycle_contract(simulation_budget_s=packet['simulation_budget_s'])
    if policy in (DIRECTIONAL_QUIET_STOP_POLICY,COMPLIANT_HOLD_POLICY) or any(k in packet for k in directional_keys):
        if (policy not in (DIRECTIONAL_QUIET_STOP_POLICY,COMPLIANT_HOLD_POLICY)
                or any(type(packet.get(k)) not in (int,float) for k in directional_keys)
                or packet.get(directional_keys[0])!=.010 or packet.get(directional_keys[1])!=.020
                or 'common_tracking_speed_floor_m_s' in packet
                or (packet.get('explicit_route_approval')!=COMPLIANT_HOLD_APPROVAL if compliant else
                    packet.get('explicit_route_approval') not in (DIRECTIONAL_QUIET_STOP_APPROVAL,ITERATIVE_FIX_APPROVAL))
                or packet.get('method')!='P_dynamics_full_envelope'):
            raise ValueError('Exact newly approved directional quiet-stop P development protocol required')
        return (compliant_hold_lifecycle_contract if compliant else directional_quiet_stop_lifecycle_contract)(simulation_budget_s=packet['simulation_budget_s'])
    if packet.get('explicit_route_approval') in (DIRECTIONAL_QUIET_STOP_APPROVAL,COMPLIANT_HOLD_APPROVAL):
        raise ValueError('New approval may not select a previous or implicit controller policy')
    if 'common_tracking_policy' not in packet and 'common_tracking_speed_floor_m_s' not in packet:
        return lifecycle.lifecycle_contract(simulation_budget_s=packet['simulation_budget_s'])
    if (packet.get('common_tracking_policy')==MEASURED_RESPONSE_FLOOR_POLICY
            and type(packet.get('common_tracking_speed_floor_m_s')) in (int,float)
            and packet['common_tracking_speed_floor_m_s']==.020
            and packet.get('explicit_route_approval')==MEASURED_RESPONSE_FIX_APPROVAL
            and packet.get('method')=='P_dynamics_full_envelope'):
        return measured_response_lifecycle_contract(simulation_budget_s=packet['simulation_budget_s'])
    if (packet.get('common_tracking_policy')!=TRACKING_FLOOR_POLICY
            or type(packet.get('common_tracking_speed_floor_m_s')) not in (int,float)
            or packet['common_tracking_speed_floor_m_s']!=.010
            or packet.get('explicit_route_approval') not in TRACKING_FLOOR_APPROVALS
            or packet.get('method')!='P_dynamics_full_envelope'):
        raise ValueError('Exact separately approved prospective common tracking-floor protocol required')
    return tracking_floor_lifecycle_contract(simulation_budget_s=packet['simulation_budget_s'])


def tracking_floor_lifecycle_contract(*,simulation_budget_s):
    contract=lifecycle.lifecycle_contract(simulation_budget_s=simulation_budget_s)
    contract.update(version='COMMON_MEASURED_LIFECYCLE_TRACKING_FLOOR_DEV_v1',
        original_motion_contract=tracking_floor_candidate_contract(simulation_budget_s=simulation_budget_s),
        common_tracking_policy=TRACKING_FLOOR_POLICY,common_tracking_speed_floor_m_s=.010,
        changed_controller_requires_same_version_for_final_matched_comparison=True)
    return contract


class TrackingFloorLifecycleCandidate(lifecycle.MissionLifecycleController):
    """Original outer sequencing/guards with the explicitly selected candidate."""
    def __init__(self,plan,geometry,*,simulation_budget_s,start_time_s=0.):
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s,start_time_s=start_time_s)
        self.motion=CommonTrackingFloorCandidate(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.contract=tracking_floor_lifecycle_contract(simulation_budget_s=simulation_budget_s)


def route_controller(packet,plan,geometry):
    profile=plan.get('geometry_profile')
    if profile!=packet.get('geometry_profile') or profile is not None and packet.get('common_tracking_policy')!=FINAL_TARGET_HOLD_POLICY:
        # Never silently run a new1.330m reference through the historical
        # command/alignment geometry. A separate exact packet binding is
        # required; merely passing the old version's scope checks is not it.
        raise ValueError('Unified geometry plan requires its explicit profile-bound runtime packet; legacy route binding prohibited')
    expected=checked_route_lifecycle_contract(packet)
    if packet.get('common_lifecycle_contract')!=expected:
        raise ValueError('Same declared route controller contract required before SDK actuation')
    if profile is not None:
        return unified_geometry_lifecycle_controller(plan,geometry,profile=profile,
            simulation_budget_s=packet['simulation_budget_s'],wall_budget_s=checked_common_wall_budget(packet))
    cls=(FinalTargetHoldLifecycleCandidate if packet.get('common_tracking_policy')==FINAL_TARGET_HOLD_POLICY else
        CompliantHoldLifecycleCandidate if packet.get('common_tracking_policy')==COMPLIANT_HOLD_POLICY else
        DirectionalQuietStopLifecycleCandidate if packet.get('common_tracking_policy')==DIRECTIONAL_QUIET_STOP_POLICY else
        MeasuredResponseLifecycleCandidate if packet.get('common_tracking_policy')==MEASURED_RESPONSE_FLOOR_POLICY else
        TrackingFloorLifecycleCandidate if 'common_tracking_policy' in packet else lifecycle.MissionLifecycleController)
    return cls(plan,geometry,simulation_budget_s=packet['simulation_budget_s'])


def tracking_floor_candidate_contract(*,simulation_budget_s,tracking_speed_floor_m_s=.010):
    if type(tracking_speed_floor_m_s) not in (int,float) or tracking_speed_floor_m_s!=.010:
        raise ValueError('Explicit prospective common10mm/s contract required')
    return _floor_candidate_contract(simulation_budget_s=simulation_budget_s,tracking_speed_floor_m_s=tracking_speed_floor_m_s)


def _floor_candidate_contract(*,simulation_budget_s,tracking_speed_floor_m_s):
    contract=lifecycle.motion.mission_execution_contract(simulation_budget_s=simulation_budget_s)
    contract.update(version='shared_bidirectional_v3_common_tracking_floor_OFFLINE_candidate_v1',
        tracking_binding=_tracking_floor_candidate_implementation(tracking_speed_floor_m_s).mission_binding,
        common_tracking_speed_floor_m_s=float(tracking_speed_floor_m_s),
        low_command_stall_mitigation_hypothesis_NOT_PhysX_validation=True,
        original_frozen_planned_speed_profile_NOT_the_executed_tracking_profile=True,
        nominal_force_and_collision_guards_rechecked_during_MODEL_rollout=True,
        changed_common_controller_requires_prospective_matched_comparison=True,
        candidate_source=str(Path(__file__).resolve()),candidate_source_sha256=retained.runtime.task.sha(__file__))
    return contract


class MeasuredResponseFloorCandidate(lifecycle.motion.MissionTrackingController):
    """New common20mm/s hypothesis from ONE local PhysX response probe.

    Not a globally identified speed threshold or a changed physical model.
    Same target law for either direction/all methods; original endpoint cap,
    acceleration, guards, mode families and final acceptance remain intact.
    """
    def __init__(self,plan,geometry,*,simulation_budget_s):
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.tracking_step=_tracking_floor_candidate_implementation(.020)


def measured_response_candidate_contract(*,simulation_budget_s):
    contract=_floor_candidate_contract(simulation_budget_s=simulation_budget_s,tracking_speed_floor_m_s=.020)
    contract.update(version='common20mm_s_tracking_floor_LOCAL_RESPONSE_DEV_v1',
        one_local_probe_NOT_global_minimum_speed_identification=True,
        profile_changes_require_same_final_controller_and_cross_task_regression=True)
    return contract


def measured_response_lifecycle_contract(*,simulation_budget_s):
    contract=lifecycle.lifecycle_contract(simulation_budget_s=simulation_budget_s)
    contract.update(version='COMMON_MEASURED_LIFECYCLE_20mm_s_TRACKING_FLOOR_DEV_v1',
        original_motion_contract=measured_response_candidate_contract(simulation_budget_s=simulation_budget_s),
        common_tracking_policy=MEASURED_RESPONSE_FLOOR_POLICY,common_tracking_speed_floor_m_s=.020,
        changed_controller_requires_same_version_for_final_matched_comparison=True)
    return contract


class MeasuredResponseLifecycleCandidate(lifecycle.MissionLifecycleController):
    def __init__(self,plan,geometry,*,simulation_budget_s,start_time_s=0.):
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s,start_time_s=start_time_s)
        self.motion=MeasuredResponseFloorCandidate(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.contract=measured_response_lifecycle_contract(simulation_budget_s=simulation_budget_s)


def _quiet_stop_candidate_implementation(*,retain_final_targets=False):
    """Strengthen ONLY the original initial/cusp position-hold entry gate.

    Require continuous under5mm/s for250ms AND under3mm/s at capture. At
    an interior cusp, measured steering must also remain within2.5deg of
    neutral during that dwell BEFORE wheel position hold is captured. The
    original8s stopping timeout and every post-capture5mm/s guard remain.
    Physical measurements, commands, gains, force limits and state are not
    replaced. This is not a proof that a captured hold can never rebound.
    """
    import ast,inspect,textwrap
    function=lifecycle.motion.MissionTrackingController.step
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
    expected=ast.dump(ast.parse('abs(self.speed)<1e-7 and measured_speed<=.005',mode='eval').body,include_attributes=False)
    changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.dump(node.test,include_attributes=False)==expected:
            changed.append((node,node.test));node.test=ast.parse('_quiet_stop_ready(self,timestamp,measured_speed,measured_steer)',mode='eval').body
    if len(changed)!=1:raise ValueError('Exactly one original initial/cusp stop-entry predicate required')
    braking_block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and changed[0][0] in n.body)
    neutral_command=ast.parse("if self.stage=='cusp_braking': self.curvature=0.").body[0]
    braking_block.body.insert(0,neutral_command)
    extra=[];labels=[]
    if retain_final_targets:
        block=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==
            ast.dump(ast.parse("self.stage=='final_neutral'",mode='eval').body,include_attributes=False))
        for statement in block.body:
            if isinstance(statement,ast.Assign) and len(statement.targets)==1:
                target=statement.targets[0]
                if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self' and target.attr=='curvature':
                    extra.append((statement,statement.value));statement.value=ast.parse('self.curvature',mode='eval').body
                elif isinstance(target,ast.Name) and target.id=='ready':
                    extra.append((statement,statement.value));statement.value=ast.Name(id='drive_hold_confirmed',ctx=ast.Load())
        if len(extra)!=2:raise ValueError('Exactly final steering rewrite and neutral-ready predicate required')
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and node.value=='final_neutral':labels.append(node);node.value='final_target_hold'
        if len(labels)!=3:raise ValueError('Exactly the three original final-neutral stage labels required')
    def ready(controller,timestamp,measured_speed,actual_steering):
        cusp_aligned=(controller.stage!='cusp_braking' or
            max(abs(float(value)) for value in actual_steering)<=np.radians(2.5))
        if abs(controller.speed)>=1e-7 or measured_speed>.005 or not cusp_aligned:
            controller.quiet_stop_since=None;return False
        if controller.quiet_stop_since is None:controller.quiet_stop_since=timestamp
        return timestamp-controller.quiet_stop_since>=.25 and measured_speed<=.003
    namespace=dict(function.__globals__,_quiet_stop_ready=ready)
    exec(compile(ast.fix_missing_locations(tree),'<private_stricter_measured_stop_entry>','exec'),namespace)
    result=namespace[function.__name__]
    for node,value in extra:node.value=value
    for node in labels:node.value='final_neutral'
    braking_block.body.remove(neutral_command)
    changed[0][0].test=changed[0][1]
    if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
        raise ValueError('Undo must recover the entire original mission-step AST')
    result.stop_binding=dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        AST_stop_entry_substitutions=1,undo_reproduces_entire_original_AST=True,
        continuous_original5mm_s_observation_s=.25,additional_capture_speed_m_s=.003,
        interior_cusp_measured_neutral_before_capture_deg=2.5,
        interior_cusp_neutral_required_through_entire_quiet_dwell=True,
        interior_cusp_neutral_command_before_hold_capture=True,
        original8s_stop_timeout_and_postcapture5mm_s_guard_unchanged=True,
        measured_physical_state_NEVER_fabricated_or_overwritten=True)
    if retain_final_targets:result.stop_binding.update(final_target_hold_revision='PRESERVE_FINAL_STEERING_TARGETS_NO_ADDED_NEUTRAL_MOTION_v1',
        final_only_AST_statement_substitutions=2,final_only_stage_labels=3,
        body_goal30mm2deg5mm_s_and_actual_ACK_entry_guards_unchanged=True,
        initial_and_interior_cusp_neutral_and_alignment_guards_unchanged=True)
    return result


def stop_connection_plan_inputs():
    """Verify the NEW own-prefix MODEL candidate, never a physical PASS."""
    report_path=STOP_CONNECTION/'connection_diagnosis.json';report=retained.read(report_path)
    retained.runtime.task.verify_pins(report['output_sha256'])
    plan_path=Path(report['combined_chain']);plan=retained.read(plan_path)
    if (report.get('common_executable_stop_connections') is not True or not report['connection_found']
            or any(report[k]['verdict']!='PASS' for k in ('original_full_NE_and10mm_SAT_audit',
                'additional_planning_reserve_audit','original_reverse_resource_audit'))
            or plan.get('common_executable_stop_connections') is not True
            or plan['selected_actions'][-1]['curvature_1_m']!=0.
            or any(a['direction']!=1 for a in plan['selected_actions'])
            or max(abs(plan['selected_states'][-1][k]) for k in ('front_rad','rear_rad'))>np.deg2rad(2.5)):
        raise ValueError('Own independently verified neutral/forward-only MODEL connection required')
    if retained.runtime.task.sha(plan['geometry_manifest'])!=plan['geometry_manifest_sha256']:
        raise ValueError('Exact unchanged approved geometry required')
    geometry=retained.read(plan['geometry_manifest']);lifecycle.motion.validate_mission_plan(plan,geometry)
    return plan_path,report_path,plan,geometry


def checked_stop_connection_model_condition():
    plan_path,_,_,_=stop_connection_plan_inputs()
    component,_=checked_final_target_hold_probe_condition()
    path=STOP_CONNECTION_MODEL/'model_screen.json';report=retained.read(path)
    retained.runtime.task.verify_pins(report['input_sha256']);retained.runtime.task.verify_pins(report['output_sha256'])
    if (report['plan']!=str(plan_path) or report['actual_motion_authorized'] is not False
            or report['integration_steps_s']!=[.020,.010] or len(report['cases'])!=2
            or any(row['model_status']!='MODEL_SCREEN_PASS' or row['error'] is not None for row in report['cases'])
            or report.get('final_target_hold_contract')!=final_target_hold_lifecycle_contract(simulation_budget_s=600.)
            or report.get('own_actual_retained_target_hold_PASS')!=str(component)
            or report['same_directional_quiet_stop_controller_and_original_plant'] is not True):
        raise ValueError('Own two-resolution current-controller MODEL PASS required, not physical evidence')
    return plan_path,path


def export_stop_connection_model_screen():
    """Two P-only engineering cases; NOT ablations/final repetitions/PhysX."""
    import csv,sys
    if any(name in sys.modules for name in ('isaacsim','omni.kit.app','omni.physx')):
        raise RuntimeError('Offline-only changed-plan screen')
    plan_path,diagnosis_path,plan,geometry=stop_connection_plan_inputs()
    component,component_raw=checked_final_target_hold_probe_condition()
    output=STOP_CONNECTION_MODEL
    if output.exists():raise ValueError('Existing model screen retained; no overwrite/retry')
    sources=tuple(dict.fromkeys((Path(__file__).resolve(),*lifecycle.host_source_files(),
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',
        ROOT/'studies/IJCAS/scripts/screen_hospital_h4_common_executor.py',
        ROOT/'src/patient_initial_stability/patient_initial_stability/rigid_body_edge.py',
        ROOT/'src/patient_initial_stability/patient_initial_stability/steering_lattice.py')))
    paths=(*sources,plan_path,diagnosis_path,STOP_CONNECTION/'planning_occupancy.npz',component,
        Path(component_raw['protocol']),Path(component_raw['trace']),component.parent/'probe_sources.zip')
    pins={str(p):retained.runtime.task.sha(p) for p in paths}
    definition=dict(role='CHANGED_OWN_PREFIX_NEUTRAL_STOP_PLAN_P_ONLY_ENGINEERING_MODEL',
        plan=str(plan_path),integration_steps_s=[.020,.010],input_sha256=pins,
        cached_generation_NOT_primary_matched_planning_performance=True,
        same_directional_quiet_stop_controller_and_original_plant=True,
        preceding_field_applies_to_tracking_and_plant_NOT_unchanged_terminal_reconfiguration=True,
        final_target_hold_contract=final_target_hold_lifecycle_contract(simulation_budget_s=600.),
        own_actual_retained_target_hold_PASS=str(component),
        NOT_a_manuscript_component_ablation_or_independent_repeat=True,
        original_measured30mm2deg5mm_s5s_hold_and600s900s_limits_unchanged=True,
        source10000_hold_and_original8deg_s_SDK_bridge_intended=True,
        actual_motion_authorized=False,new_PhysX_trial_count=0,use_for_final_statistics=False)
    retained.evidence.write_exclusive(output/'prospective_model_definition.json',definition)
    lifecycle.freeze_workflow_sources(output/'model_sources.zip',{str(p):pins[str(p)] for p in sources})
    files=[output/'prospective_model_definition.json',output/'model_sources.zip'];cases=[]
    with np.load(STOP_CONNECTION/'planning_occupancy.npz') as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    for dt in definition['integration_steps_s']:
        run,_,design=tracking_floor_model_case_binding(geometry,plan,simulation_budget_s=600.,final_target_hold=True)
        report,rows=run(plan,plan['rigid_body_parameter_bounds'],dt,occupied,origin,resolution,design)
        trace=output/('P_'+str(round(dt*1000))+'ms.csv')
        with trace.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report.update(trace=str(trace),trace_sha256=retained.runtime.task.sha(trace));cases.append(report);files.append(trace)
        result=trace.with_suffix('.json');retained.evidence.write_exclusive(result,lifecycle.normalize_runtime_evidence(report))
        files.append(result)
        print('[H6 NEUTRAL STOP MODEL]',dt,report['model_status'],report['error'],flush=True)
    retained.runtime.task.verify_pins(pins)
    result=dict(definition,cases=cases,output_sha256={str(p):retained.runtime.task.sha(p) for p in files})
    retained.evidence.write_exclusive(output/'model_screen.json',lifecycle.normalize_runtime_evidence(result))
    return output/'model_screen.json'


class DirectionalQuietStopCandidate(lifecycle.motion.MissionTrackingController):
    """Method-blind direction response floor plus stricter measured stop entry."""
    def __init__(self,plan,geometry,*,simulation_budget_s):
        self.quiet_stop_since=None
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.tracking_step=_tracking_floor_candidate_implementation(.020,physical_direction_conditioned=True)
        self._quiet_step=_quiet_stop_candidate_implementation()

    def _make_tracker(self):
        super()._make_tracker();self.tracker.mission_travel_direction=self.leg['direction']
        self.quiet_stop_since=None

    def _stage(self,stage,timestamp,pose):
        if stage in ('initial_stop','cusp_braking'):self.quiet_stop_since=None
        return super()._stage(stage,timestamp,pose)

    def step(self,*args,**kwargs):return self._quiet_step(self,*args,**kwargs)


def directional_quiet_stop_contract(*,simulation_budget_s):
    contract=measured_response_candidate_contract(simulation_budget_s=simulation_budget_s)
    contract.update(version='common_directional10_20mm_s_and_quiet_stop_OFFLINE_candidate_v1',
        tracking_binding=_tracking_floor_candidate_implementation(.020,physical_direction_conditioned=True).mission_binding,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        stop_entry_binding=_quiet_stop_candidate_implementation().stop_binding,
        local_response_and_prior_development_evidence_NOT_global_guarantees=True,
        explicit_OFFLINE_candidate_NOT_wired_to_live_route=True)
    contract.pop('common_tracking_speed_floor_m_s',None)
    return contract


def directional_quiet_stop_lifecycle_contract(*,simulation_budget_s):
    contract=lifecycle.lifecycle_contract(simulation_budget_s=simulation_budget_s)
    motion_contract=directional_quiet_stop_contract(simulation_budget_s=simulation_budget_s)
    motion_contract.pop('explicit_OFFLINE_candidate_NOT_wired_to_live_route')
    motion_contract.update(version='common_directional10_20mm_s_and_quiet_stop_APPROVED_DEV_v1',
        separately_approved_ONE_PhysX_development_route_NOT_final_validation=True)
    contract.update(version='COMMON_LIFECYCLE_DIRECTIONAL_QUIET_STOP_APPROVED_DEV_v1',
        original_motion_contract=motion_contract,common_tracking_policy=DIRECTIONAL_QUIET_STOP_POLICY,
        common_forward_tracking_floor_m_s=.010,common_reverse_tracking_floor_m_s=.020,
        changed_controller_requires_same_version_for_final_matched_comparison=True)
    return contract


class DirectionalQuietStopLifecycleCandidate(lifecycle.MissionLifecycleController):
    """Exact tested candidate inside unchanged outer safeguards and cleanup."""
    def __init__(self,plan,geometry,*,simulation_budget_s,start_time_s=0.):
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s,start_time_s=start_time_s)
        self.motion=DirectionalQuietStopCandidate(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.contract=directional_quiet_stop_lifecycle_contract(simulation_budget_s=simulation_budget_s)


class FinalTargetHoldCandidate(DirectionalQuietStopCandidate):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.final_targets_retained=False
        self._quiet_step=_quiet_stop_candidate_implementation(retain_final_targets=True)

    def step(self,*args,**kwargs):
        info=self._quiet_step(self,*args,**kwargs)
        if self.stage=='final_target_hold':
            self.final_targets_retained=True
            info.update(freeze_steering_targets=True,final_target_hold_NO_added_steering_motion=True)
        return info


def unified_geometry_motion_class(profile):
    """Private1.330m version of the SAME common motion controller.

    No SDK, measurements, hold ACK, target, gain or authority is supplied.
    Bind both cusp alignment and the original v2 terminal pulse alignment,
    including trackers constructed later, without mutating historical code.
    """
    from types import FunctionType,MethodType
    from audit_hospital_paper_readiness import unified_wheelbase_connections
    connections=unified_wheelbase_connections(profile)
    def bind(function,**updates):
        result=FunctionType(function.__code__,dict(function.__globals__,**updates),
            function.__name__,function.__defaults__,function.__closure__)
        result.__kwdefaults__=function.__kwdefaults__;result.__dict__.update(function.__dict__)
        return result
    class Controller(FinalTargetHoldCandidate):
        def __init__(self,plan,*args,**kwargs):
            if plan.get('geometry_profile')!=profile:raise ValueError('Exact frozen plan/profile required')
            super().__init__(plan,*args,**kwargs)
            self.geometry_profile=copy.deepcopy(profile)
            self._quiet_step=bind(self._quiet_step,steering_angles=connections.steering_angles)
            original_raw=self.raw
            def terminal(controller):
                function=controller.step.__func__
                if function.__globals__.get('steering_geometry') is None:
                    raise ValueError('Original terminal alignment anchor required')
                controller.step=MethodType(bind(function,steering_geometry=connections.steering_geometry),controller)
                return controller
            self.raw=lambda *a,**kw:terminal(original_raw(*a,**kw))
            terminal(self.tracker)
            if plan.get('measured_replan_start',{}).get('goal_distance_handoff_policy') is not None:
                from hospital_sensor_navigation import h5_goal_distance_tracking
                self.tracking_step=h5_goal_distance_tracking(self.tracking_step,plan['goal_world_xyyaw'])
    Controller.geometry_connections=connections
    return Controller


def unified_geometry_lifecycle_controller(plan,geometry,*,profile,simulation_budget_s,start_time_s=0.,wall_budget_s=900.):
    """Component construction ONLY, never a launch or old-packet admission."""
    cls=unified_geometry_motion_class(profile)
    controller=FinalTargetHoldLifecycleCandidate(plan,geometry,
        simulation_budget_s=simulation_budget_s,start_time_s=start_time_s)
    controller.motion=cls(plan,geometry,simulation_budget_s=simulation_budget_s)
    controller.geometry_profile=copy.deepcopy(profile)
    controller.contract=unified_geometry_lifecycle_contract(profile,simulation_budget_s=simulation_budget_s,wall_budget_s=wall_budget_s)
    return controller


def unified_geometry_lifecycle_contract(profile,*,simulation_budget_s,wall_budget_s=900.):
    from audit_hospital_paper_readiness import unified_wheelbase_kernel
    unified_wheelbase_kernel(profile)
    contract=final_target_hold_lifecycle_contract(simulation_budget_s=simulation_budget_s)
    contract.update(version='COMMON_LIFECYCLE_UNIFIED1330_RETAIN_FINAL_TARGETS_v1',
        geometry_profile=copy.deepcopy(profile),
        original_common_controller_and_limits_changed_ONLY_geometry_bindings=True,
        separate_profile_bound_runtime_packet_STILL_required=True)
    if type(wall_budget_s) not in (int,float) or wall_budget_s not in (900.,1500.):
        raise ValueError('Exact bounded common wall allowance required')
    if wall_budget_s==1500.:
        if simulation_budget_s!=600.:raise ValueError('Reviewed1500s component is H6-only; keep original600s physics cap')
        contract.update(wall_budget_s=1500.,
            original_common_controller_and_limits_changed_ONLY_geometry_bindings=False,
            additional_reviewed_common_PC_wall_allowance_only_changed=True)
    return contract


def final_target_hold_motion_contract(*,simulation_budget_s):
    c=directional_quiet_stop_contract(simulation_budget_s=simulation_budget_s)
    c.update(version='COMMON_RETAIN_FINAL_STEERING_TARGETS_DEV_v1',
        stop_entry_binding=_quiet_stop_candidate_implementation(retain_final_targets=True).stop_binding,
        final_goal_is_BODY_pose_and_fixed5s_stop_NOT_zero_steering_angle=True,
        source10000_hold_and_original8deg_s_actuator_limits_unchanged=True)
    return c


def final_target_hold_lifecycle_contract(*,simulation_budget_s):
    c=directional_quiet_stop_lifecycle_contract(simulation_budget_s=simulation_budget_s)
    c.update(version='COMMON_LIFECYCLE_RETAIN_FINAL_STEERING_TARGETS_DEV_v1',common_tracking_policy=FINAL_TARGET_HOLD_POLICY,
        original_motion_contract=final_target_hold_motion_contract(simulation_budget_s=simulation_budget_s),
        final_steering_target_retention_is_NOT_faked_neutral_measurement=True,
        original30mm2deg5mm_s5s_last3s_and_progress_acceptance_unchanged=True,
        original10000_hold_and_original8deg_s_bridge_except_explicit_final_target_freeze=True)
    return c


def _final_target_lifecycle_step():
    import ast,inspect,textwrap
    f=lifecycle.MissionLifecycleController._step;source=textwrap.dedent(inspect.getsource(f));tree=ast.parse(source)
    original=ast.parse('max(abs(v) for v in steer)>math.radians(2.5)',mode='eval').body
    expected=ast.dump(original,include_attributes=False);replacements=[]
    class Replace(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==expected:
                new=ast.parse('not self.motion.final_targets_retained',mode='eval').body
                replacements.append((new,node));return new
            return self.generic_visit(node)
    tree=Replace().visit(tree)
    if len(replacements)!=1:raise ValueError('Exactly original final-neutral admission predicate required')
    namespace=dict(f.__globals__);exec(compile(ast.fix_missing_locations(tree),'<private_final_target_hold_lifecycle>','exec'),namespace)
    result=namespace[f.__name__]
    class Undo(ast.NodeTransformer):
        def generic_visit(self,node):
            if node is replacements[0][0]:return replacements[0][1]
            return super().generic_visit(node)
    restored=Undo().visit(tree)
    if ast.dump(restored,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('Whole original outer lifecycle AST must recover')
    return result


class FinalTargetHoldLifecycleCandidate(lifecycle.MissionLifecycleController):
    def __init__(self,plan,geometry,*,simulation_budget_s,start_time_s=0.):
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s,start_time_s=start_time_s)
        self.motion=FinalTargetHoldCandidate(plan,geometry,simulation_budget_s=simulation_budget_s)
        self.contract=final_target_hold_lifecycle_contract(simulation_budget_s=simulation_budget_s)
        self._retained_step=_final_target_lifecycle_step()

    def _step(self,*args,**kwargs):
        info=self._retained_step(self,*args,**kwargs)
        if self.lifecycle_stage=='fixed_hold':
            if not self.motion.final_targets_retained:raise RuntimeError('Explicit actual final-target hold entry required')
            info=dict(info,freeze_steering_targets=True,final_target_hold_NO_added_steering_motion=True)
            self.last_info=dict(info)
        return info


def tracking_floor_model_case_binding(geometry,plan,*,simulation_budget_s,measured_response=False,directional_quiet=False,final_target_hold=False):
    """Same original case/plant/guards; replace ONLY the declared controller."""
    from types import FunctionType
    function=lifecycle.motion.mission_model_case_binding
    bound=FunctionType(function.__code__,dict(function.__globals__,
        MissionTrackingController=FinalTargetHoldCandidate if final_target_hold else DirectionalQuietStopCandidate if directional_quiet else MeasuredResponseFloorCandidate if measured_response else CommonTrackingFloorCandidate,
        mission_execution_contract=final_target_hold_motion_contract if final_target_hold else directional_quiet_stop_contract if directional_quiet else measured_response_candidate_contract if measured_response else tracking_floor_candidate_contract),function.__name__,function.__defaults__)
    bound.__kwdefaults__=function.__kwdefaults__
    return bound(geometry,plan,simulation_budget_s=simulation_budget_s)


def export_tracking_floor_model_screen(planning_review_path,*,simulation_budget_s,output,measured_response=False,directional_quiet=False):
    """All-four/two-resolution OFFLINE hypothesis screen, no new live entry.

    The original archive records its unchanged model/exporter. A separate
    candidate archive records this explicitly different common controller.
    Neither software resolution is an independent physical repetition.
    """
    import zipfile
    from types import FunctionType
    function=lifecycle.motion.export_mission_executor_model_screen
    model_case=lambda *args,**kw:tracking_floor_model_case_binding(*args,**kw,measured_response=measured_response,directional_quiet=directional_quiet)
    bound=FunctionType(function.__code__,dict(function.__globals__,
        mission_execution_contract=directional_quiet_stop_contract if directional_quiet else measured_response_candidate_contract if measured_response else tracking_floor_candidate_contract,
        mission_model_case_binding=model_case),function.__name__,function.__defaults__)
    bound.__kwdefaults__=function.__kwdefaults__
    source=Path(__file__).resolve();test=ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py'
    sources={str(p):retained.runtime.task.sha(p) for p in (source,test)}
    result=bound(planning_review_path,simulation_budget_s=simulation_budget_s,output=output)
    retained.runtime.task.verify_pins(sources)
    archive=result.parent/'candidate_sources.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as saved:
        for name in sources:saved.write(name,Path(name).relative_to(ROOT).as_posix())
    receipt=dict(status='OFFLINE_COMMON_TRACKING_FLOOR_CANDIDATE_NOT_PHYSX_OR_LIVE_ADMISSION',
        model_result=str(result),model_result_sha256=retained.runtime.task.sha(result),
        candidate_source_sha256=sources,candidate_archive=str(archive),candidate_archive_sha256=retained.runtime.task.sha(archive),
        original_model_exporter_and_controller_file_unchanged=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    retained.evidence.write_exclusive(result.parent/'candidate_closeout.json',receipt)
    return result


class ReverseResponseProbe:
    """One diagnostic, not route recovery or a terminal success controller.

    Retain the failed reverse steering target; ramp desired velocity only.
    Distance/time guards end the observation, never extend a level until it
    moves. Early braking reserves nominal .020m/s2 stopping distance, .3s
    response and4mm; actual30mm is an independent hard observation guard.
    No identified friction/torque model or robust braking claim is made.
    """
    def __init__(self,start,curvature):
        import math
        if len(start)!=3 or not all(math.isfinite(v) for v in (*start,curvature)) or abs(curvature)>.7:
            raise ValueError('Finite retained pose and original reverse curvature required')
        self.start=tuple(start);self.curvature=float(curvature);self.mode='rear_only'
        self.stage='alignment';self.previous_time=None;self.started=None;self.aligned_since=None
        self.tracking_started=None;self.stopped_since=None;self.speed=0.;self.complete=False
        self.previous_pose=None;self.travel=0.;self.last_info=None
        self.contract=dict(role='SEPARATELY_APPROVED_REVERSE_RESPONSE_DIAGNOSTIC_NOT_H6_ROUTE',
            speed_levels_m_s=[-.010,-.020,-.030],level_duration_s=2.,maximum_physics_duration_s=15.,
            maximum_actual_centre_polyline_m=.030,maximum_command_m_s=.030,acceleration_m_s2=.020,
            nominal_braking_response_reserve_s=.3,nominal_braking_position_reserve_m=.004,
            stopping_reserve_NOT_a_robust_physical_distance_guarantee=True,
            automatic_retry=False,maximum_Play_calls=1,maximum_Stop_reset_calls=0,
            new_route_trials=0,use_for_final_statistics=False)

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        import math
        from hospital_path_execution import ramp
        from hospital_timed_path_execution import steering_angles
        if (self.complete or type(drive_hold_confirmed) is not bool or drive_hold_confirmed
                or len(pose)!=3 or len(measured_steer)!=2 or measured_speed<0.
                or not all(math.isfinite(v) for v in (timestamp,*pose,measured_speed,*measured_steer))):
            raise ValueError('Open diagnostic with finite actual unheld observations required')
        dt=0. if self.previous_time is None else timestamp-self.previous_time
        if not 0.<=dt<=.25:raise RuntimeError('Original0.25s control-gap guard')
        self.previous_time=timestamp
        if self.started is None:self.started=timestamp
        if timestamp-self.started>=15.:raise TimeoutError('Approved15s diagnostic cap; no extension')
        if self.previous_pose is not None:self.travel+=math.hypot(pose[0]-self.previous_pose[0],pose[1]-self.previous_pose[1])
        self.previous_pose=tuple(pose)
        if self.travel>=.030:raise RuntimeError('Approved30mm measured centre-polyline cap')
        target=0.
        if self.stage=='alignment':
            if measured_speed>.005:raise RuntimeError('Measured standstill required before reverse probe alignment')
            desired=steering_angles(self.curvature,self.mode)
            if max(abs(a-b) for a,b in zip(measured_steer,desired))<=math.radians(2.5):
                if self.aligned_since is None:self.aligned_since=timestamp
                if timestamp-self.aligned_since>=.25:
                    self.stage='tracking';self.tracking_started=timestamp
            else:self.aligned_since=None
            if timestamp-self.started>7.:raise TimeoutError('Bounded diagnostic steering alignment')
        if self.stage=='tracking':
            elapsed=timestamp-self.tracking_started
            nominal_reserve=max(abs(self.speed),measured_speed)**2/.040+.3*max(abs(self.speed),measured_speed)+.004
            if elapsed>=6. or timestamp-self.started>=12. or self.travel+nominal_reserve>=.030:
                self.stage='braking'
            else:target=(-.010,-.020,-.030)[min(2,int(elapsed/2.))]
        self.speed=ramp(self.speed,target,dt,.020)
        if self.stage=='braking' and abs(self.speed)<=1e-7 and measured_speed<=.005:
            if self.stopped_since is None:self.stopped_since=timestamp
            if timestamp-self.stopped_since>=.25:self.complete=True
        else:self.stopped_since=None
        self.last_info=dict(phase='path_tracking',terminal_stage=self.stage,phase_time_s=timestamp-self.started,
            speed_m_s=self.speed,mode=self.mode,curvature_1_m=self.curvature,
            drive_hold_requested=False,hold_ready=False,diagnostic_centre_polyline_m=self.travel,
            requested_level_m_s=target,diagnostic_complete=self.complete)
        return dict(self.last_info)


def export_reverse_response_probe(result_path,*,approval):
    """Freeze ONE own post-route diagnostic; never manufacture a STATIC PASS."""
    import csv
    import math
    if approval!=REVERSE_PROBE_APPROVAL:raise ValueError('Separate short reverse-response approval required')
    result_path=Path(result_path).resolve();review=saved_route_boundary_review(result_path)
    raw=retained.read(result_path);prior=retained.read(raw['protocol'])
    if (raw['verdict']!='HOLD' or raw.get('error')!='TimeoutError: Declared900s whole-route wall cap'
            or not raw['checks']['all4997_environment_rows_and145_source_settings_and_physics_unchanged']
            or not raw['checks']['no_unreviewed_USD_edits_or_callback_errors']):
        raise ValueError('Own fully closed post-route stalled HOLD required, not a current motion or camera abort')
    final=raw['actual_final_readback'];snapshot=final['tensor']
    if math.hypot(*snapshot['velocity'][:2])>.005:raise ValueError('Actual retained under5mm/s required')
    with Path(raw['trace']).open(newline='') as stream:
        last=None
        for last in csv.DictReader(stream):pass
    if last is None or float(last['command_speed_m_s'])!=-.010:
        raise ValueError('Original recorded reverse plateau required')
    curvature=-math.tan(float(last['target_rear_rad']))/1.29
    if abs(float(last['target_front_rad']))>1e-6:raise ValueError('Same rear-only target required')
    pose=lifecycle.RetainedMissionSDKHost._pose(snapshot)
    probe=ReverseResponseProbe(pose,curvature)
    output=result_path.parent/'reverse_response_probe_r01'
    if output.exists():raise ValueError('Existing diagnostic definition retained; no overwrite/retry')
    layers=final['layer_texts'];pins=dict(prior['input_sha256'])
    for row in review['historical_source_resolution']:pins.pop(row['path'])
    files=(result_path,Path(raw['protocol']),Path(raw['trace']),result_path.parent/'route_sources.zip',
        Path(__file__).resolve(),ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    packet=dict(prior,version='H6_RETAINED_REVERSE_RESPONSE_DIAGNOSTIC_v1',
        original_route_result=str(result_path),original_route_result_sha256=retained.runtime.task.sha(result_path),
        output_directory=str(output),explicit_reverse_probe_approval=dict(approval),
        required_PAUSED_timeline=raw['actual_final_timeline'],required_tensor_snapshot=snapshot,
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],
            unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        start_actual_xyyaw=list(pose),probe_curvature_1_m=curvature,probe_contract=probe.contract,
        maximum_wall_time_s=90.,maximum_physics_duration_s=15.,maximum_route_trials=0,
        actual_route_authorized=False,actual_response_probe_authorized=True,
        input_sha256=pins,original_static_used_ONLY_for_unchanged_source_configuration=True)
    path=output/'probe_protocol.json';retained.evidence.write_exclusive(path,packet)
    lifecycle.freeze_workflow_sources(output/'probe_sources.zip',packet['input_sha256'])
    return path


class RetainedResponseSDKHost(lifecycle.RetainedMissionSDKHost):
    """Existing SDK views/guards, independently admitted current-state probe.

    Deliberately does NOT call the route host's fixed-start/static admission
    or pretend that the final pose is its original start. Only the explicit
    diagnostic protocol, exact consumed route snapshot and full environment
    GET authorize this short response observation.
    """
    PROBE_VERSION='H6_RETAINED_REVERSE_RESPONSE_DIAGNOSTIC_v1'
    APPROVAL_FIELD='explicit_reverse_probe_approval'
    EXPECTED_APPROVAL=REVERSE_PROBE_APPROVAL
    PHYSICS_CAP=15.
    INITIAL_RELEASED_SPEED_CAP=.005

    def __init__(self,sdk,packet,path,*,context,geometry,occupied,origin,resolution):
        self.path=Path(path).resolve();self.output=self.path.parent
        if (retained.read(self.path)!=packet or packet['version']!=self.PROBE_VERSION
                or packet[self.APPROVAL_FIELD]!=self.EXPECTED_APPROVAL
                or packet['maximum_route_trials']!=0 or packet['actual_route_authorized'] is not False
                or packet['actual_response_probe_authorized'] is not True
                or packet['maximum_physics_duration_s']!=self.PHYSICS_CAP or packet['maximum_wall_time_s']!=90.):
            raise ValueError('Exact separately approved response probe required, never route admission')
        retained.runtime.task.verify_pins(packet['input_sha256'])
        lifecycle.verify_workflow_sources(self.output/'probe_sources.zip',packet['input_sha256'])
        self.geometry=geometry
        self.runtime_guard=lifecycle.MissionRuntimeGuard(geometry,occupied,origin,resolution,simulation_budget_s=600.)
        self.plan=dict(method='RETAINED_RESPONSE_DIAGNOSTIC_NOT_H6_MISSION')
        retained.RetainedZeroSettleSDKPort.__init__(self,sdk,packet,context=context)
        self.notice_sub=None;self.notice_counts={'allowed_state_paths':0,'reviewed_aperture_paths':0,'unreviewed_paths':0}
        self.claimed=False;self.reservation=self.output/'single_probe_reserved.json'
        self.result=self.output/'probe_readback.json';self.trace=self.output/'probe_trace.csv'

    async def preflight(self):
        import math
        if not self.claimed:raise ValueError('Exclusive diagnostic reservation required')
        result=await retained.RetainedZeroSettleSDKPort.preflight(self)
        self.last_dt=1./120.
        if self.raw['environment_after_static']['physics_hz']!=120:raise ValueError('Original120Hz required')
        pose=self._pose(self.protocol['required_tensor_snapshot'])
        # Check every sampled pose in a conservative60mm/3deg local box,
        # including braking margin; the live30mm cap remains independent.
        for ix in range(-12,13):
            for iy in range(-12,13):
                for angle in (-3.,0.,3.):
                    self.runtime_guard._pose((pose[0]+ix*.005,pose[1]+iy*.005,pose[2]+math.radians(angle)))
        if math.hypot(*self.protocol['required_tensor_snapshot']['velocity'][:2])>self.INITIAL_RELEASED_SPEED_CAP:
            raise ValueError('Own retained measured initial released-speed boundary required')
        return dict(result,scope='DIAGNOSTIC_NOT_STATIC_OR_ROUTE_ADMISSION',
            sampled_local60mm3deg_clearance_verified=True,source_static_NOT_reused_as_new_start=True)

    def claim_once(self):
        if any(p.exists() for p in (self.reservation,self.result,self.trace)):return False
        retained.no_other_task();retained.runtime.task.verify_pins(self.protocol['input_sha256'])
        retained.evidence.write_exclusive(self.reservation,dict(protocol=str(self.path),
            explicit_response_probe_approval=self.EXPECTED_APPROVAL,maximum_Play_calls=1,
            maximum_route_trials=0,maximum_Stop_reset_calls=0,automatic_retry=False))
        self.claimed=True;return True

    async def next_sample(self):
        await retained.RetainedZeroSettleSDKPort.next_sample(self)
        if self.clock>=self.PHYSICS_CAP:raise TimeoutError('Approved diagnostic physics cap')

    def command_readback(self):
        result=super().command_readback()
        forces=self.tensor.get_dof_projected_joint_forces()
        if hasattr(forces,'detach'):forces=forces.detach().cpu().numpy()
        elif hasattr(forces,'numpy'):forces=forces.numpy()
        forces=np.asarray(forces,float)
        if forces.shape!=(1,12) or not np.isfinite(forces).all():
            raise ValueError('Finite measured projected joint reactions required')
        for axis,index in zip(('front','rear'),self.adapter.drive):
            result['drive_'+axis+'_projected_joint_reaction_NOT_motor_torque']=float(forces[0,index])
        result['base_link_PhysX_sleeping_GET']=bool(self.sim.is_sleeping(self.stage_id,self.encoded))
        return result

    async def postflight(self):
        self.post_observation_snapshot=retained.shared.preparation.old.tensor_snapshot(self.tensor)
        self.post_observation_boundary_audits=[]
        environment=await self.environment(post_observation=True)
        if retained.shared.preparation.old.tensor_snapshot(self.tensor)!=self.post_observation_snapshot:
            raise ValueError('Actual PAUSED diagnostic state changed during full postflight')
        motors=self.adapter._motor()
        restored=all(np.array_equal(motors[k],v) for k,v in self.adapter.original.items())
        retained.runtime.task.verify_pins(self.protocol['input_sha256'])
        return dict(environment=environment,actual_post_diagnostic_tensor_snapshot=self.post_observation_snapshot,
            exact_post_observation_boundary_audits=self.post_observation_boundary_audits,
            checks=dict(all4997_rows145_settings_and_physics_unchanged=True,
                original_gains_targets_and_limits_restored=restored,all_frozen_inputs_unchanged=True,
                no_unreviewed_USD_edits_or_callback_errors=not self.errors and not self.notice_counts['unreviewed_paths'],
                diagnostic_geometry_and_resource_guards_healthy=not self.runtime_guard.failed))


class RetainedReverseResponseSDKHost(RetainedResponseSDKHost):
    """Original reverse-only scope; sharing host checks grants no new action."""


def _held_neutral_slew_implementation(*,retain_final_targets_only=False):
    """Change ONLY steering-target interpolation dt, not observed physics dt.

    Held steering is4deg/s instead of8; actual controller clocks, commands,
    measured5mm/s guards, torque limits and original cleanup remain unchanged.
    A retained diagnostic may freeze its INITIAL measured steering targets
    while the released zero-command body settles. Neither option sets state.
    """
    import ast,inspect,textwrap
    function=lifecycle.motion.MissionSDKActuationBridge._step
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='direct_joint_targets']
    if len(calls)!=1 or len(calls[0].args)!=6:raise ValueError('One original target interpolation call required')
    node=calls[0];old=node.args[4]
    expected=ast.parse('stamp-self.last_stamp',mode='eval').body
    if ast.dump(old,include_attributes=False)!=ast.dump(expected,include_attributes=False):raise ValueError('Original interpolation dt required')
    expression=("0. if info.get('freeze_steering_targets',False) else stamp-self.last_stamp" if retain_final_targets_only else
        "0. if info.get('freeze_steering_targets',False) else (0.5 if info['drive_hold_requested'] else 1.)*(stamp-self.last_stamp)")
    node.args[4]=ast.parse(expression,mode='eval').body
    namespace=dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<private_held_neutral_4deg_s_target_slew>','exec'),namespace)
    result=namespace[function.__name__];node.args[4]=old
    if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('Entire original bridge AST must recover')
    result.slew_binding=dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        AST_target_interpolation_substitutions=1,undo_recovers_entire_original_AST=True,
        held_steering_target_slew_deg_s=8. if retain_final_targets_only else 4.,released_steering_target_slew_deg_s=8.,
        measured_controller_and_SDK_clock_gaps_NEVER_rescaled=True,
        original5mm_s_hold_guard_and_gains_and_torque_limits_unchanged=True)
    return result


class HeldNeutralSlewBridge(lifecycle.motion.MissionSDKActuationBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._candidate_step=_held_neutral_slew_implementation()

    def _step(self,*args,**kwargs):return self._candidate_step(self,*args,**kwargs)


class RetainFinalTargetsBridge(lifecycle.motion.MissionSDKActuationBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._candidate_step=_held_neutral_slew_implementation(retain_final_targets_only=True)
        profile=getattr(self.controller,'geometry_profile',None)
        if profile is not None:
            if self.controller.contract.get('geometry_profile')!=profile:
                raise ValueError('Explicit same geometry in lifecycle contract required before any SDK write')
            self._candidate_step=unified_geometry_final_bridge_implementation(profile)

    def _step(self,*args,**kwargs):return self._candidate_step(self,*args,**kwargs)


def unified_geometry_final_bridge_implementation(profile):
    """Original retained-target SDK transaction, ONLY geometry rebound.

    In-function direct-target import must be replaced too: changing globals
    alone would silently reimport1.290m. No state/gain/ACK/clock guard changes.
    """
    import ast,inspect,textwrap
    from audit_hospital_paper_readiness import unified_wheelbase_connections
    connections=unified_wheelbase_connections(profile)
    source=textwrap.dedent(inspect.getsource(lifecycle.motion.MissionSDKActuationBridge._step))
    tree=ast.parse(source);imports=[];targets=[];changes=[]
    class Bind(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module=='hospital_path_execution' and len(node.names)==1 and node.names[0].name=='direct_joint_targets':
                imports.append(True)
                replacement=ast.copy_location(ast.Assign([ast.Name('direct_joint_targets',ast.Store())],
                    ast.Name('_unified_direct_joint_targets',ast.Load())),node)
                changes.append((replacement,node));return replacement
            return node
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='direct_joint_targets':
                if len(node.args)!=6 or ast.dump(node.args[4],include_attributes=False)!=ast.dump(
                        ast.parse('stamp-self.last_stamp',mode='eval').body,include_attributes=False):
                    raise ValueError('Original measured-clock interpolation anchor required')
                targets.append((node,node.args[4]))
                node.args[4]=ast.parse("0. if info.get('freeze_steering_targets',False) else stamp-self.last_stamp",mode='eval').body
            return self.generic_visit(node)
    tree=Bind().visit(tree)
    if len(imports)!=1 or len(targets)!=1:raise ValueError('One original direct-target import/call required')
    namespace=dict(lifecycle.motion.MissionSDKActuationBridge._step.__globals__,
        _unified_direct_joint_targets=connections.direct_joint_targets,steering_angles=connections.steering_angles)
    exec(compile(ast.fix_missing_locations(tree),'<private_unified1330_retained_target_bridge>','exec'),namespace)
    result=namespace['_step']
    targets[0][0].args[4]=targets[0][1]
    class Undo(ast.NodeTransformer):
        def generic_visit(self,node):
            if node is changes[0][0]:return changes[0][1]
            return super().generic_visit(node)
    if ast.dump(Undo().visit(tree),include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):
        raise ValueError('Entire original SDK bridge AST must be recovered')
    result.geometry_binding=dict(wheelbase_m=1.330,
        command_import_substitutions=1,original_final_only_target_freeze_substitutions=1,
        undo_recovers_entire_original_SDK_AST=True,actual_ACKs_and_gains_state_clock_and_guards_unchanged=True,
        no_runtime_packet_or_motion_authority_supplied=True)
    return result


def _feedback_held_steering_implementation():
    """Change target interpolation ONLY; retain original transaction/guards.

    The rate governor is a candidate, NOT a proven stopping invariant. Actual
    measured5mm/s still aborts independently, including after rate reduction.
    Neither the physics clock nor any measured state is rescaled or reset.
    """
    import ast,inspect,textwrap
    function=lifecycle.motion.MissionSDKActuationBridge._step
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='direct_joint_targets']
    if len(calls)!=1 or len(calls[0].args)!=6:raise ValueError('One original target interpolation call required')
    node=calls[0];old=copy.deepcopy(node)
    if ast.dump(node.args[4],include_attributes=False)!=ast.dump(ast.parse('stamp-self.last_stamp',mode='eval').body,include_attributes=False):
        raise ValueError('Original interpolation clock required')
    replacement=ast.parse('self.feedback_targets(stamp,measured_speed,actual,info)',mode='eval').body
    node.func=replacement.func;node.args=replacement.args;node.keywords=[]
    namespace=dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<private_feedback_held_steering>','exec'),namespace)
    result=namespace[function.__name__];node.func=old.func;node.args=old.args;node.keywords=old.keywords
    if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('Entire original bridge AST must recover')
    result.feedback_binding=dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        AST_target_interpolation_substitutions=1,undo_recovers_entire_original_AST=True,
        held_max_target_rate_deg_s=4.,held_target_acceleration_deg_s2=1.,
        full_rate_below_body_speed_m_s=.002,zero_rate_above_body_speed_m_s=.0035,
        maximum_held_target_lead_deg=2.5,released_rate_deg_s=8.,
        revision='PREDICTIVE_SPEED_FEEDBACK_WITH_TARGET_ERROR_UNWIND_v3',
        speed_prediction_horizon_s=.15,error_unwind_starts_predicted_speed_m_s=.001,
        zero_target_error_at_predicted_speed_m_s=.0025,
        target_error_unwind_max_rate_deg_s=8.,
        original_measured5mm_s_abort_and16s_neutral_timeout_unchanged=True,
        feedback_is_NOT_a_proven_invariant_or_a_physical_model=True,
        measured_controller_and_SDK_clock_gaps_NEVER_rescaled=True)
    return result


class FeedbackHeldNeutralBridge(lifecycle.motion.MissionSDKActuationBridge):
    """Method-blind slew/anti-windup candidate, no extra actuator writes."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.held_rate=0.
        self.feedback_previous_speed=None
        self._candidate_step=_feedback_held_steering_implementation()

    def feedback_targets(self,stamp,speed,actual,info):
        import math
        from hospital_path_execution import direct_joint_targets,ramp
        dt=stamp-self.last_stamp
        interpolation=self.feedback_interpolation_dt(stamp,speed,actual,info)
        steer,wheels=direct_joint_targets(info['speed_m_s'],info['mode'],info['curvature_1_m'],
            self.steering_command,interpolation,actual)
        if info['drive_hold_requested'] and not info.get('freeze_steering_targets',False):
            previous=speed if self.feedback_previous_speed is None else self.feedback_previous_speed
            predicted=speed+max(0.,speed-previous)/max(dt,1e-12)*.15
            error_cap=math.radians(2.5)*max(0.,min(1.,(.0025-predicted)/.0015))
            # Freezing the old target retains its restoring torque. Unwind
            # that desired-target error instead, at the ORIGINAL8deg/s bound.
            desired=tuple(max(a-error_cap,min(a+error_cap,s)) for a,s in zip(actual,steer))
            steer=tuple(ramp(old,new,dt,math.radians(8.)) for old,new in zip(self.steering_command,desired))
        self.feedback_previous_speed=speed
        return steer,wheels

    def feedback_interpolation_dt(self,stamp,speed,actual,info):
        import math
        dt=stamp-self.last_stamp
        if not math.isfinite(dt) or not 0.<=dt<=.25:raise ValueError('Original actual clock gap required')
        if info.get('freeze_steering_targets',False):self.held_rate=0.;return 0.
        if not info['drive_hold_requested']:self.held_rate=0.;return dt
        # Do not accumulate desired-target error while the tire/steering is
        # sticking, or continue adding steering displacement near the guard.
        lead=max(abs(a-b) for a,b in zip(actual,self.steering_command))
        scale=max(0.,min(1.,(.0035-speed)/.0015))
        desired=math.radians(4.)*scale if lead<math.radians(2.5) else 0.
        if desired<self.held_rate:self.held_rate=desired
        else:self.held_rate=min(desired,self.held_rate+math.radians(1.)*dt)
        return dt*self.held_rate/math.radians(8.)

    def _step(self,*args,**kwargs):return self._candidate_step(self,*args,**kwargs)


def _compliant_hold_implementation():
    """OFFLINE candidate: ONLY the transient captured-position P gain changes.

    This is a contact-response tuning hypothesis, not identified hardware
    parameters or proof of stability. Keep original damping, effort limits,
    measured standstill requirements, captured targets and final restoration.
    """
    import ast,inspect,textwrap
    function=lifecycle.motion.RetainedTensorDriveAdapter.request_hold
    source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='drive_hold_gain_arrays']
    if len(calls)!=1 or calls[0].keywords:raise ValueError('Exactly one original default captured-wheel gain call required')
    node=calls[0];node.keywords=[ast.keyword(arg='hold_kp',value=ast.Constant(value=1000.))]
    namespace=dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<private_common_compliant1000_hold_candidate>','exec'),namespace)
    result=namespace[function.__name__];node.keywords=[]
    if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('Entire original hold transaction AST must recover')
    result.compliance_binding=dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        AST_hold_stiffness_keyword_substitutions=1,undo_recovers_entire_original_AST=True,
        original_development_hold_kp_Nm_rad=10000.,candidate_hold_kp_Nm_rad=1000.,
        original_source_damping_effort_limits_passive_entries_and_restore_unchanged=True,
        original5mm_s_capture_release_guard_and_zero_command_unchanged=True,
        OFFLINE_candidate_NOT_wired_to_a_live_protocol=True)
    return result


class CompliantHoldDriveAdapter(lifecycle.motion.RetainedTensorDriveAdapter):
    """Method-blind transient gain candidate; construction is still read-only."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._compliant_hold=_compliant_hold_implementation()

    def request_hold(self,*args,**kwargs):return self._compliant_hold(self,*args,**kwargs)


def compliant_hold_lifecycle_contract(*,simulation_budget_s):
    contract=directional_quiet_stop_lifecycle_contract(simulation_budget_s=simulation_budget_s)
    binding=copy.deepcopy(_compliant_hold_implementation().compliance_binding)
    binding.pop('OFFLINE_candidate_NOT_wired_to_a_live_protocol')
    contract.update(version='COMMON_COMPLIANT1000_HELD4deg_s_DIRECTIONAL_QUIET_STOP_DEV_v1',
        common_tracking_policy=COMPLIANT_HOLD_POLICY,transient_common_hold_gain_candidate=binding,
        actual_SDK_bridge_binding=_held_neutral_slew_implementation().slew_binding,
        original_static10000_hold_NOT_this1000_route_prehold=True,
        route_own_measured2s_prehold_required_at_candidate_gain=True)
    return contract


class CompliantHoldLifecycleCandidate(DirectionalQuietStopLifecycleCandidate):
    def __init__(self,plan,geometry,*,simulation_budget_s,start_time_s=0.):
        super().__init__(plan,geometry,simulation_budget_s=simulation_budget_s,start_time_s=start_time_s)
        self.contract=compliant_hold_lifecycle_contract(simulation_budget_s=simulation_budget_s)


class CompliantRetainedMissionSDKHost(lifecycle.RetainedMissionSDKHost):
    async def preflight(self):
        if self.protocol.get('common_tracking_policy')!=COMPLIANT_HOLD_POLICY:raise ValueError('Exact approved compliant route packet required')
        result=await super().preflight();bind_compliant_hold_adapter(self)
        return dict(result,explicit_common_transient_hold_kp_Nm_rad=1000.,original_source_damping_effort_limits_unchanged=True)


def prime_retained_steering_targets(actuator):
    """PAUSED desired-target capture, NOT a joint-position/velocity reset.

    The old route restored unused zero targets while leaving physical steering
    at31deg. A retained response test must start from the MEASURED angles, not
    impose an instantaneous31deg desired-target jump before its slew limiter.
    Original complete targets are preserved in the adapter for final restore.
    """
    if actuator.paused_check() is not True or actuator.confirmation(actuator.last_stamp) is not False:
        raise ValueError('Same PAUSED released original motor baseline required')
    actual=actuator.measured_steering();targets=actuator.expected['positions'].copy()
    targets[actuator.steering]=np.asarray(actual,np.float32)
    actuator.expected['positions']=targets
    actuator._set('set_dof_position_targets',targets,paused=True)
    actuator._verify(actuator._motor())
    return actual


class HeldNeutralResponseProbe:
    """ONE fixed zero-command settle/capture/neutral/2s observation sequence."""
    def __init__(self,start,*,captured_steering_observation_s=0.):
        import math
        if len(start)!=3 or not all(math.isfinite(v) for v in start):raise ValueError('Finite retained actual pose required')
        if type(captured_steering_observation_s) not in (int,float) or captured_steering_observation_s not in (0.,1.):
            raise ValueError('Only original or declared fixed1s capture observation supported')
        self.capture_observation_s=float(captured_steering_observation_s)
        self.start=tuple(start);self.stage='released_settle';self.mode='coordinated_counterphase';self.curvature=0.
        self.previous=None;self.previous_pose=None;self.travel=0.;self.quiet_since=None;self.stage_started=0.
        self.neutral_since=None;self.complete=False;self.last_info=None;self.held_since=None
        self.contract=dict(role='ONE_HELD_NEUTRAL_RESPONSE_NOT_STATIC_OR_ROUTE',
            maximum_physics_duration_s=20.,maximum_wall_time_s=90.,maximum_centre_polyline_m=.030,
            zero_drive_velocity_command_only=True,initial_released_settling_cap_s=8.,
            continuous_under5mm_s_before_capture_s=.25,capture_speed_limit_m_s=.003,
            held_speed_limit_m_s=.005,held_steering_target_slew_deg_s=4.,
            neutral_tolerance_deg=2.5,neutral_dwell_s=.25,fixed_final_observation_s=2.,
            captured_steering_response_observation_s=self.capture_observation_s,
            maximum_Play_calls=1,maximum_Stop_reset_calls=0,new_route_trials=0,
            gains_torque_limits_physical_model_and_passive_joints_unchanged=True,
            automatic_retry=False,use_for_final_statistics=False,
            bridge_binding=_held_neutral_slew_implementation().slew_binding)

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        import math
        if (self.complete or type(drive_hold_confirmed) is not bool or len(pose)!=3 or len(measured_steer)!=2
                or measured_speed<0. or not all(math.isfinite(v) for v in (timestamp,*pose,measured_speed,*measured_steer))):
            raise ValueError('Finite open diagnostic observations required')
        if self.previous is not None and not 0.<timestamp-self.previous<=.25:raise ValueError('Original control-gap guard')
        self.previous=timestamp
        if timestamp>=20.:raise TimeoutError('Fixed20s diagnostic cap')
        if self.previous_pose is not None:self.travel+=math.hypot(pose[0]-self.previous_pose[0],pose[1]-self.previous_pose[1])
        self.previous_pose=tuple(pose)
        if self.travel>=.030:raise RuntimeError('Fixed30mm diagnostic polyline cap')
        if self.stage=='released_settle':
            if drive_hold_confirmed:raise RuntimeError('No pre-existing hold or recapture')
            if timestamp>8.:raise TimeoutError('Original8s measured settling cap')
            if measured_speed>.005:self.quiet_since=None
            elif self.quiet_since is None:self.quiet_since=timestamp
            if self.quiet_since is not None and timestamp-self.quiet_since>=.25 and measured_speed<=.003:
                self.stage='captured_steering_hold' if self.capture_observation_s else 'held_neutral';self.stage_started=timestamp
        if self.stage in ('captured_steering_hold','held_neutral','fixed_observation'):
            if measured_speed>.005:raise RuntimeError('Original5mm/s held-neutral speed guard')
            if drive_hold_confirmed and self.held_since is None:self.held_since=timestamp
            if self.stage=='captured_steering_hold':
                if self.held_since is not None and not drive_hold_confirmed:raise RuntimeError('Original hold ACK lost; no recapture')
                if self.held_since is not None and timestamp-self.held_since>=self.capture_observation_s:
                    self.stage='held_neutral';self.stage_started=timestamp
            elif self.stage=='held_neutral':
                if timestamp-self.stage_started>16.:raise TimeoutError('Original16s neutral/ACK cap')
                if drive_hold_confirmed and max(abs(v) for v in measured_steer)<=math.radians(2.5):
                    if self.neutral_since is None:self.neutral_since=timestamp
                    if timestamp-self.neutral_since>=.25:self.stage='fixed_observation';self.stage_started=timestamp
                else:self.neutral_since=None
            elif not drive_hold_confirmed:raise RuntimeError('Original hold ACK lost; no second capture')
            elif timestamp-self.stage_started>=2.:self.complete=True
        self.last_info=dict(phase='path_tracking',terminal_stage=self.stage,phase_time_s=timestamp-self.stage_started,
            speed_m_s=0.,mode=self.mode,curvature_1_m=0.,drive_hold_requested=self.stage!='released_settle',
            freeze_steering_targets=self.stage in ('released_settle','captured_steering_hold'),hold_ready=False,
            diagnostic_centre_polyline_m=self.travel,diagnostic_complete=self.complete)
        return dict(self.last_info)


class RetainedHeldNeutralSDKHost(RetainedResponseSDKHost):
    PROBE_VERSION='H6_RETAINED_HELD_NEUTRAL_RESPONSE_DIAGNOSTIC_v1'
    APPROVAL_FIELD='explicit_response_probe_approval'
    EXPECTED_APPROVAL=HELD_NEUTRAL_PROBE_APPROVAL
    PHYSICS_CAP=20.
    INITIAL_RELEASED_SPEED_CAP=.008


class CompliantHeldNeutralResponseProbe(HeldNeutralResponseProbe):
    def __init__(self,start):
        super().__init__(start,captured_steering_observation_s=1.)
        self.contract.pop('gains_torque_limits_physical_model_and_passive_joints_unchanged')
        binding=copy.deepcopy(_compliant_hold_implementation().compliance_binding)
        binding.pop('OFFLINE_candidate_NOT_wired_to_a_live_protocol')
        self.contract.update(role='ONE_COMMON_COMPLIANT_HELD_NEUTRAL_RESPONSE_NOT_ROUTE',
            transient_common_hold_gain_candidate=binding,
            source_motor_baseline_damping_torque_limits_physical_model_passive_joints_unchanged=True)


class FeedbackHeldNeutralResponseProbe(CompliantHeldNeutralResponseProbe):
    def __init__(self,start):
        super().__init__(start)
        self.contract.update(role='ONE_FEEDBACK_HELD_NEUTRAL_RESPONSE_NOT_ROUTE',
            held_steering_target_slew_deg_s=4.,
            bridge_binding=_feedback_held_steering_implementation().feedback_binding,
            initial1s_after_actual_capture_ACK_is_SETTLING_not_stopped_evidence=True,
            original_runtime80mm_s_tilt_geometry_contact_guards_apply_during_settling=True,
            unchanged5mm_s_required_after_fixed1s_settling_and_through_neutral_and_fixed_observation=True,
            no_further_gain_change_from_preceding1000_candidate=True)

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        # The original held-static lifecycle explicitly settles for1s BEFORE
        # fixed recording. Do not call a capture transient a failed completed
        # standstill. Keep RAW measured speeds; no clipping/averaging/restart.
        if not hasattr(self,'_settling_step'):
            import ast,inspect,textwrap
            function=HeldNeutralResponseProbe.step
            source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source)
            expected=ast.dump(ast.parse('measured_speed>.005',mode='eval').body,include_attributes=False)
            found=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==expected]
            if len(found)!=2:raise ValueError('Exactly released quiet-capture and held-speed predicates required')
            node=next(n for n in found if any(isinstance(v,ast.Raise) for v in n.body));old=node.test
            node.test=ast.parse("self.stage!='captured_steering_hold' and measured_speed>.005",mode='eval').body
            namespace=dict(function.__globals__)
            exec(compile(ast.fix_missing_locations(tree),'<private_fixed1s_capture_settling>','exec'),namespace)
            self._settling_step=namespace[function.__name__];node.test=old
            if ast.dump(tree,include_attributes=False)!=ast.dump(ast.parse(source),include_attributes=False):raise ValueError('Whole original response AST must recover')
        info=self._settling_step(self,timestamp,pose,measured_speed,measured_steer,drive_hold_confirmed=drive_hold_confirmed)
        if self.stage!='captured_steering_hold' and info['drive_hold_requested'] and measured_speed>.005:
            raise RuntimeError('Original5mm/s held-neutral speed guard')
        return info


class RetainedFinalTargetHoldProbe:
    """Zero command, ONE measured capture, original5s/fixed-last3s hold.

    Preserve the measured steering targets, not a fictitious neutral angle.
    This engineering component NEVER completes/promotes the old H6 route.
    """
    def __init__(self,start,goal):
        import math
        if any(len(p)!=3 or not all(math.isfinite(v) for v in p) for p in (start,goal)):
            raise ValueError('Finite actual start and UNCHANGED original goal required')
        self.start=tuple(start);self.goal=tuple(goal);self.stage='released_settle'
        self.mode='coordinated_counterphase';self.curvature=0.;self.previous=None
        self.previous_pose=None;self.travel=0.;self.quiet_since=None;self.held_since=None
        self.complete=False;self.last_info=None
        self.contract=dict(role='RETAINED_FINAL_TARGET_HOLD_ENGINEERING_NOT_H6_SUCCESS',
            zero_drive_command_and_no_new_steering_target_motion=True,maximum_physics_duration_s=20.,
            maximum_wall_time_s=90.,maximum_centre_polyline_m=.030,initial_released_settling_cap_s=8.,
            capture_under5mm_s_continuous_s=.25,capture_instant_speed_m_s=.003,
            fixed_hold_s=5.,fixed_tail_s=3.,original_goal_tolerance_m_deg=[.030,2.],
            original_fixed_tail_speed_m_s=.005,source_original10000_hold_and_damping_effort_limits=True,
            no_rehold_window_restart_or_route_admission=True,new_route_trials=0,use_for_final_statistics=False)

    def step(self,timestamp,pose,measured_speed,measured_steer,*,drive_hold_confirmed):
        import math
        if (self.complete or type(drive_hold_confirmed) is not bool or len(pose)!=3 or len(measured_steer)!=2
                or measured_speed<0. or not all(math.isfinite(v) for v in (timestamp,*pose,measured_speed,*measured_steer))):
            raise ValueError('Finite open retained-hold observations required')
        if self.previous is not None and not 0.<timestamp-self.previous<=.25:raise ValueError('Original control-gap guard')
        self.previous=timestamp
        if timestamp>=20.:raise TimeoutError('Fixed20s diagnostic cap')
        if self.previous_pose is not None:self.travel+=math.hypot(pose[0]-self.previous_pose[0],pose[1]-self.previous_pose[1])
        self.previous_pose=tuple(pose)
        if self.travel>=.030:raise RuntimeError('Fixed30mm diagnostic travel cap')
        error=lifecycle.relative_pose(self.goal,pose)
        if math.hypot(*error[:2])>.030 or abs(error[2])>math.radians(2.):raise RuntimeError('Original terminal30mm/2deg boundary')
        if self.stage=='released_settle':
            if drive_hold_confirmed:raise RuntimeError('Original released baseline required')
            if timestamp>8.:raise TimeoutError('Original8s measured stop cap')
            if measured_speed>.005:self.quiet_since=None
            elif self.quiet_since is None:self.quiet_since=timestamp
            if self.quiet_since is not None and timestamp-self.quiet_since>=.25 and measured_speed<=.003:
                self.stage='hold_ACK'
        if self.stage=='hold_ACK' and drive_hold_confirmed:self.held_since=timestamp;self.stage='terminal_hold'
        elapsed=0. if self.held_since is None else timestamp-self.held_since
        if self.stage=='terminal_hold':
            if not drive_hold_confirmed:raise RuntimeError('Hold ACK lost; no recapture/window restart')
            if elapsed>=2. and measured_speed>.005:raise RuntimeError('Original fixed-tail5mm/s guard')
            self.complete=elapsed>=5.
        self.last_info=dict(phase=self.stage,terminal_stage=self.stage,phase_time_s=elapsed,
            speed_m_s=0.,mode=self.mode,curvature_1_m=0.,drive_hold_requested=self.stage!='released_settle',
            freeze_steering_targets=True,hold_ready=False,diagnostic_complete=self.complete,
            diagnostic_centre_polyline_m=self.travel)
        return dict(self.last_info)


def bind_compliant_hold_adapter(port):
    """Rebind SAME tensor; constructors only read/verify the original baseline."""
    old=port.adapter
    if old.dirty or old.capture_count or old.pending_stamp is not None:raise ValueError('Untouched preflight baseline required')
    port.adapter=CompliantHoldDriveAdapter(old.view,port.raw['runtime_motor_baseline'],encode=old.encode,
        check_context=old.check_context,paused_check=old.paused_check,physics_stamp=old.last_stamp,authorized=True)


class RetainedCompliantHeldNeutralSDKHost(RetainedHeldNeutralSDKHost):
    PROBE_VERSION='H6_RETAINED_COMPLIANT1000_HELD_NEUTRAL_RESPONSE_DIAGNOSTIC_v1'
    EXPECTED_APPROVAL=COMPLIANT_HOLD_APPROVAL

    async def preflight(self):
        result=await super().preflight();bind_compliant_hold_adapter(self)
        return dict(result,explicit_common_transient_hold_kp_Nm_rad=1000.,source_gains_damping_limits_unchanged=True)


class RetainedFeedbackHeldNeutralSDKHost(RetainedCompliantHeldNeutralSDKHost):
    PROBE_VERSION='H6_RETAINED_FEEDBACK_HELD_NEUTRAL_RESPONSE_DIAGNOSTIC_v1'
    EXPECTED_APPROVAL=ITERATIVE_FIX_APPROVAL


class RetainedFinalTargetHoldSDKHost(RetainedHeldNeutralSDKHost):
    PROBE_VERSION='H6_RETAINED_FINAL_TARGET_HOLD_ENGINEERING_v1'
    EXPECTED_APPROVAL=ITERATIVE_FIX_APPROVAL


def export_final_target_hold_probe(result_path,*,approval):
    if approval!=ITERATIVE_FIX_APPROVAL:raise ValueError('Current evidence-led correction authority required')
    result_path=Path(result_path).resolve();review=saved_route_boundary_review(result_path)
    raw=retained.read(result_path);prior=retained.read(raw['protocol'])
    required=('actual_Pause_confirmed','original_gains_targets_restored',
        'all4997_environment_rows_and145_source_settings_and_physics_unchanged',
        'original_all12joint_gains_targets_and_limits_restored','all_frozen_inputs_unchanged',
        'final_actual_goal_position_under30mm','final_actual_goal_heading_under2deg',
        'no_unreviewed_USD_edits_or_callback_errors','all_sampled_geometry_support_and_resource_guards_completed',
        'contacts_healthy_through_actual_Pause')
    if (raw['verdict']!='HOLD' or raw.get('error')!='RuntimeError: First terminal capture left original30mm/2deg/5mm_s before fixed hold; no retry'
            or not all(raw['checks'].get(k) is True for k in required)):
        raise ValueError('Own fully closed goal-reaching terminal-neutralization HOLD required')
    final=raw['actual_final_readback'];snapshot=final['tensor'];state=raw['actual_final_timeline']
    if not final['stable_PAUSED_tensor'] or final['timeline']!=state:raise ValueError('Same stable actual final snapshot required')
    pose=lifecycle.RetainedMissionSDKHost._pose(snapshot);goal=retained.read(prior['geometry_file'])['goal_xyyaw']
    output=result_path.parent/'final_target_hold_response_probe_r01'
    if output.exists():raise ValueError('Existing retained hold evidence preserved; no unchanged retry')
    pins=dict(prior['input_sha256'])
    for row in review['historical_source_resolution']:pins.pop(row['path'])
    files=(result_path,Path(raw['protocol']),Path(raw['trace']),result_path.parent/'route_sources.zip',
        Path(__file__).resolve(),ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    layers=final['layer_texts']
    packet=dict(prior,version=RetainedFinalTargetHoldSDKHost.PROBE_VERSION,
        original_route_result=str(result_path),original_route_result_sha256=retained.runtime.task.sha(result_path),
        output_directory=str(output),explicit_response_probe_approval=dict(approval),
        required_PAUSED_timeline=state,required_tensor_snapshot=snapshot,
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        start_actual_xyyaw=list(pose),original_goal_xyyaw=goal,
        probe_contract=RetainedFinalTargetHoldProbe(pose,goal).contract,
        maximum_wall_time_s=90.,maximum_physics_duration_s=20.,maximum_route_trials=0,
        actual_route_authorized=False,actual_response_probe_authorized=True,
        previous_route_HOLD_NEVER_promoted=True,input_sha256=pins)
    path=output/'probe_protocol.json';retained.evidence.write_exclusive(path,packet)
    lifecycle.freeze_workflow_sources(output/'probe_sources.zip',pins)
    return path


def export_feedback_held_neutral_probe(result_path,*,approval,revision=1):
    """Prospective changed candidate from the latest fully closed HOLD.

    This exports a NEW bounded diagnostic, not a retry of a consumed packet,
    a PASS handoff, or permission to skip actual source/environment checks.
    """
    import math
    if approval!=ITERATIVE_FIX_APPROVAL:raise ValueError('Actual current iterative-fix authority required')
    result_path=Path(result_path).resolve();raw=retained.read(result_path);prior=retained.read(raw['protocol'])
    allowed=((RetainedCompliantHeldNeutralSDKHost.PROBE_VERSION,COMPLIANT_HOLD_APPROVAL),
        (RetainedFeedbackHeldNeutralSDKHost.PROBE_VERSION,ITERATIVE_FIX_APPROVAL))
    if (type(revision) is not int or not 1<=revision<=4
            or (prior.get('version'),prior.get('explicit_response_probe_approval')) not in allowed
            or raw.get('role')!='HELD_NEUTRAL_RESPONSE_NOT_ROUTE_SUCCESS' or raw.get('verdict')!='HOLD'
            or raw.get('error') not in ('RuntimeError: Original5mm/s held-neutral speed guard','TimeoutError: Original16s neutral/ACK cap')
            or raw.get('new_route_trials')!=0 or raw.get('new_response_probe_trials')!=1
            or raw['protocol_sha256']!=retained.runtime.task.sha(raw['protocol'])
            or raw['trace_sha256']!=retained.runtime.task.sha(raw['trace'])
            or not raw['restoration']['restored']
            or any(not v for k,v in raw['checks'].items() if k!='approved_bounded_observation_completed')):
        raise ValueError('Own latest fully closed speed-guard HOLD required, never waive an incomplete cleanup')
    archive=result_path.parent/'probe_sources.zip'
    lifecycle.verify_workflow_sources(archive,prior['input_sha256'])
    history=lifecycle.reviewed_prior_GET_dependencies(prior['input_sha256'],archive)
    final=raw['actual_final_readback'];snapshot=final['tensor'];state=raw['actual_final_timeline']
    prep=retained.read(raw['preparation_record'])
    if (not final['stable_PAUSED_tensor'] or final['timeline']!=state or state['playing'] or state['stopped']
            or raw['postflight']['environment']!=dict(row_count=4997,sha256=prep['protected_environment_sha256'],
                all145_source_settings_and_critical_environment_verified=True)
            or math.hypot(*snapshot['velocity'][:2])>.008):
        raise ValueError('Same exact PAUSED tensor and complete unchanged environment required')
    output=result_path.parent.parent/('feedback_held_neutral_response_probe_r'+str(revision).zfill(2))
    if output.exists():raise ValueError('Existing feedback diagnostic preserved; no overwrite/retry')
    pins=dict(prior['input_sha256'])
    for row in history['resolved_consumed_sources']:pins.pop(row['path'])
    files=(result_path,Path(raw['protocol']),Path(raw['trace']),archive,Path(__file__).resolve(),
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    pose=lifecycle.RetainedMissionSDKHost._pose(snapshot);layers=final['layer_texts']
    if prior['probe_contract']==FeedbackHeldNeutralResponseProbe(pose).contract:
        raise ValueError('Unchanged feedback candidate cannot be automatically retried')
    packet=dict(prior,version=RetainedFeedbackHeldNeutralSDKHost.PROBE_VERSION,
        retained_response_source=str(result_path),retained_response_source_sha256=retained.runtime.task.sha(result_path),
        output_directory=str(output),explicit_response_probe_approval=dict(approval),
        required_PAUSED_timeline=state,required_tensor_snapshot=snapshot,
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        start_actual_xyyaw=list(pose),probe_contract=FeedbackHeldNeutralResponseProbe(pose).contract,
        preceding_HOLD_preserved=True,input_sha256=pins)
    path=output/'probe_protocol.json';retained.evidence.write_exclusive(path,packet)
    lifecycle.freeze_workflow_sources(output/'probe_sources.zip',pins)
    return path


def saved_held_neutral_probe_boundary_review(result_path):
    """COPY proof for a consumed diagnostic, never promote its HOLD."""
    result_path=Path(result_path).resolve();raw=retained.read(result_path);packet=retained.read(raw['protocol'])
    archive=result_path.parent/'probe_sources.zip'
    lifecycle.verify_workflow_sources(archive,packet['input_sha256'])
    history=lifecycle.reviewed_prior_GET_dependencies(packet['input_sha256'],archive)
    if (packet.get('version')!=RetainedHeldNeutralSDKHost.PROBE_VERSION
            or packet.get('explicit_response_probe_approval')!=HELD_NEUTRAL_PROBE_APPROVAL
            or raw.get('role')!='HELD_NEUTRAL_RESPONSE_NOT_ROUTE_SUCCESS'
            or raw['new_route_trials']!=0 or raw['new_response_probe_trials']!=1
            or not raw['checks']['actual_Pause_confirmed'] or not raw['restoration']['restored']
            or raw['trace_sha256']!=retained.runtime.task.sha(raw['trace'])
            or raw['protocol_sha256']!=retained.runtime.task.sha(raw['protocol'])):
        raise ValueError('Own completed one diagnostic/Pause/restoration required')
    previous=retained.read(packet['original_route_result'])['actual_final_readback'];final=raw['actual_final_readback']
    if not final['stable_PAUSED_tensor'] or final['timeline']!=raw['actual_final_timeline']:
        raise ValueError('Own exact stable PAUSED retained snapshot required')
    proof=retained.recorded_physics_writeback_review(previous['layer_texts'],final['layer_texts'],previous['tensor'],final['tensor'],packet['actor'])
    return dict(saved_state_writeback_proof=proof,historical_source_resolution=history,
        original_HOLD_unchanged=True,all4997_source_environment_rows_STILL_require_new_preflight=True,
        actual_route_admission=False)


def export_compliant_held_neutral_probe(result_path,*,approval):
    import math
    if approval!=COMPLIANT_HOLD_APPROVAL:raise ValueError('Exact new compliant-probe conditional-workflow approval required')
    result_path=Path(result_path).resolve();review=saved_held_neutral_probe_boundary_review(result_path)
    raw=retained.read(result_path);prior=retained.read(raw['protocol'])
    if raw['verdict']!='HOLD' or raw.get('error')!='RuntimeError: Original5mm/s held-neutral speed guard':
        raise ValueError('Own preserved held-neutral physical HOLD required')
    snapshot=raw['actual_final_readback']['tensor'];pose=lifecycle.RetainedMissionSDKHost._pose(snapshot)
    if math.hypot(*snapshot['velocity'][:2])>.008:raise ValueError('Bounded released initial speed before zero-only settling required')
    probe=CompliantHeldNeutralResponseProbe(pose);output=result_path.parent.parent/'compliant_held_neutral_response_probe_r01'
    if output.exists():raise ValueError('Existing compliant diagnostic preserved; no overwrite/retry')
    pins=dict(prior['input_sha256'])
    for row in review['historical_source_resolution']['resolved_consumed_sources']:pins.pop(row['path'])
    files=(result_path,Path(raw['protocol']),Path(raw['trace']),result_path.parent/'probe_sources.zip',Path(__file__).resolve(),
        ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',ROOT/'src/patient_initial_stability/test/test_hospital_mission_runtime.py',*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    layers=raw['actual_final_readback']['layer_texts']
    packet=dict(prior,version=RetainedCompliantHeldNeutralSDKHost.PROBE_VERSION,
        retained_response_source=str(result_path),retained_response_source_sha256=retained.runtime.task.sha(result_path),
        output_directory=str(output),explicit_response_probe_approval=dict(approval),
        required_PAUSED_timeline=raw['actual_final_timeline'],required_tensor_snapshot=snapshot,
        required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        start_actual_xyyaw=list(pose),probe_contract=probe.contract,previous_diagnostic_boundary_review=review,input_sha256=pins)
    path=output/'probe_protocol.json';retained.evidence.write_exclusive(path,packet)
    lifecycle.freeze_workflow_sources(output/'probe_sources.zip',packet['input_sha256'])
    return path


def export_held_neutral_response_probe(result_path,*,approval):
    import math
    if approval!=HELD_NEUTRAL_PROBE_APPROVAL:raise ValueError('Exact new ONE held-neutral response approval required')
    result_path=Path(result_path).resolve();review=saved_route_boundary_review(result_path)
    raw=retained.read(result_path);prior=retained.read(raw['protocol'])
    if (raw['verdict']!='HOLD' or raw.get('error')!='RuntimeError: Hold requested while actual body is not stopped'
            or not raw['checks']['all4997_environment_rows_and145_source_settings_and_physics_unchanged']
            or not raw['checks']['no_unreviewed_USD_edits_or_callback_errors']):raise ValueError('Own fully closed held-neutral HOLD required')
    snapshot=raw['actual_final_readback']['tensor'];pose=lifecycle.RetainedMissionSDKHost._pose(snapshot)
    if math.hypot(*snapshot['velocity'][:2])>.008:raise ValueError('Bounded retained released body speed required before zero-only settling')
    probe=HeldNeutralResponseProbe(pose);output=result_path.parent/'held_neutral_response_probe_r01'
    if output.exists():raise ValueError('Existing response probe preserved; no overwrite/retry')
    layers=raw['actual_final_readback']['layer_texts'];pins=dict(prior['input_sha256'])
    for row in review['historical_source_resolution']:pins.pop(row['path'])
    files=(result_path,Path(raw['protocol']),Path(raw['trace']),result_path.parent/'route_sources.zip',
        Path(__file__).resolve(),ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    packet=dict(prior,version=RetainedHeldNeutralSDKHost.PROBE_VERSION,original_route_result=str(result_path),
        original_route_result_sha256=retained.runtime.task.sha(result_path),output_directory=str(output),
        explicit_response_probe_approval=dict(approval),required_PAUSED_timeline=raw['actual_final_timeline'],
        required_tensor_snapshot=snapshot,required_layer_sha256=[hashlib.sha256(layers[k].encode()).hexdigest() for k in ('Root','Session')],
        reviewed_display_boundary=dict(reference_Session_text=layers['Session'],unchanged_Root_sha256=hashlib.sha256(layers['Root'].encode()).hexdigest(),
            reviewed_default_paths=list(retained.TOP_XY_DISPLAY_PATHS),policy=retained.TOP_XY_DISPLAY_POLICY),
        start_actual_xyyaw=list(pose),probe_contract=probe.contract,maximum_wall_time_s=90.,maximum_physics_duration_s=20.,
        maximum_route_trials=0,actual_route_authorized=False,actual_response_probe_authorized=True,input_sha256=pins,
        original_static_used_ONLY_for_unchanged_source_configuration=True)
    path=output/'probe_protocol.json';retained.evidence.write_exclusive(path,packet)
    lifecycle.freeze_workflow_sources(output/'probe_sources.zip',packet['input_sha256'])
    return path


async def run_held_neutral_response_probe_once(path,*,authorized=False):
    import math,time
    if authorized is not True:raise ValueError('Explicit ONE held-neutral response authority required')
    path=Path(path).resolve();packet=retained.read(path)
    source=packet.get('retained_response_source',packet['original_route_result'])
    raw=retained.read(source);prior=retained.read(raw['protocol'])
    if (retained.runtime.task.sha(source)!=packet.get('retained_response_source_sha256',packet['original_route_result_sha256'])
            or packet['required_tensor_snapshot']!=raw['actual_final_readback']['tensor']
            or packet['required_PAUSED_timeline']!=raw['actual_final_timeline']
            or packet['start_actual_xyyaw']!=list(lifecycle.RetainedMissionSDKHost._pose(raw['actual_final_readback']['tensor']))
            or any(packet[k]!=prior[k] for k in ('actor','original_static_record','original_preparation','original_static_packet','geometry_file','grid_file','simulation_budget_s'))):
        raise ValueError('Own exact retained route boundary/configuration required')
    static=retained.read(prior['original_static_packet'])
    context=retained.static_binding(static,Path(prior['original_static_packet']),Path(prior['original_static_record']).parent,Path(prior['original_preparation']))
    geometry=retained.read(packet['geometry_file']);sdk=retained.runtime.live_sdk()
    choices={RetainedHeldNeutralSDKHost.PROBE_VERSION:(RetainedHeldNeutralSDKHost,HeldNeutralResponseProbe),
        RetainedCompliantHeldNeutralSDKHost.PROBE_VERSION:(RetainedCompliantHeldNeutralSDKHost,CompliantHeldNeutralResponseProbe),
        RetainedFeedbackHeldNeutralSDKHost.PROBE_VERSION:(RetainedFeedbackHeldNeutralSDKHost,FeedbackHeldNeutralResponseProbe),
        RetainedFinalTargetHoldSDKHost.PROBE_VERSION:(RetainedFinalTargetHoldSDKHost,RetainedFinalTargetHoldProbe)}
    if packet.get('version') not in choices:raise ValueError('Exact known response packet required')
    host_class,controller_class=choices[packet['version']]
    with np.load(packet['grid_file']) as grid:
        port=host_class(sdk,packet,path,context=context,geometry=geometry,
            occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),resolution=float(grid['resolution_m']))
    controller=(controller_class(packet['start_actual_xyyaw'],packet['original_goal_xyyaw'])
        if controller_class is RetainedFinalTargetHoldProbe else controller_class(packet['start_actual_xyyaw']))
    if controller.contract!=packet['probe_contract']:raise ValueError('Exact frozen response controller contract required')
    return await observe_held_neutral_once(port,controller,authorized=True)


async def observe_held_neutral_once(port,controller,*,authorized=False):
    """One admitted host transaction; also testable with labelled SOFTWARE ports."""
    import math,time
    if authorized is not True or type(controller) not in (HeldNeutralResponseProbe,CompliantHeldNeutralResponseProbe,FeedbackHeldNeutralResponseProbe,RetainedFinalTargetHoldProbe):
        raise ValueError('Exact separately authorized ONE response component required')
    if not port.claim_once():return dict(status='EXISTING_PROBE_RETAINED_NO_NEW_ACTION')
    report=dict(verdict='HOLD',role='HELD_NEUTRAL_RESPONSE_NOT_ROUTE_SUCCESS',checks={},new_route_trials=0,
        new_response_probe_trials=0,automatic_retry=False,use_for_final_statistics=False,contract=controller.contract)
    rows=[];preflight=actuation=False;observed=None
    try:
        report['preflight']=await port.preflight();preflight=True;port.subscribe()
        actuation=True;initial=prime_retained_steering_targets(port.actuator)
        bridge_class=FeedbackHeldNeutralBridge if type(controller) is FeedbackHeldNeutralResponseProbe else HeldNeutralSlewBridge
        bridge=bridge_class(controller,port.actuator,guard=lambda pose,speed,info:port.guard(observed,info))
        bridge.steering_command=tuple(initial)
        observed=lifecycle.checked_measurement(port.measure());port.guard(observed,None)
        speed=max(math.hypot(observed['vx_physx_m_s'],observed['vy_physx_m_s']),observed['pose_filtered_speed_m_s'])
        bridge.step(observed['time_s'],(observed['x_m'],observed['y_m'],observed['yaw_rad']),speed)
        began=time.monotonic();port.play();report['new_response_probe_trials']=1
        while not controller.complete:
            await asyncio.wait_for(port.next_sample(),timeout=max(.01,90.-(time.monotonic()-began)))
            if time.monotonic()-began>90.:raise TimeoutError('Fixed90s diagnostic wall cap')
            observed=lifecycle.checked_measurement(port.measure())
            speed=max(math.hypot(observed['vx_physx_m_s'],observed['vy_physx_m_s']),observed['pose_filtered_speed_m_s'])
            info=bridge.step(observed['time_s'],(observed['x_m'],observed['y_m'],observed['yaw_rad']),speed)
            rows.append(dict(observed,command_speed_m_s=0.,phase=info['terminal_stage'],
                phase_time_s=info['phase_time_s'],diagnostic_centre_polyline_m=controller.travel,
                held_steering_target_rate_rad_s=getattr(bridge,'held_rate',None),**port.command_readback()))
        if type(controller) is RetainedFinalTargetHoldProbe:
            from hospital_static_gate import evaluate_static
            held=[r for r in rows if r['phase']=='terminal_hold'];tail=[r for r in held if r['phase_time_s']>=2.]
            if not held or held[0]['phase_time_s']!=0. or held[-1]['phase_time_s']<5.:raise ValueError('Whole fixed5s from FIRST ACK required')
            static=evaluate_static(tail,[held[0][k] for k in ('x_m','y_m','yaw_rad')],265.,port.counts,port.errors)
            stop=lifecycle.analyze_stop_trace(tail)
            errors=[lifecycle.relative_pose(controller.goal,[r[k] for k in ('x_m','y_m','yaw_rad')]) for r in tail]
            checks=dict(static['checks'],fixed_last3s_complete=stop['duration_s']>=2.9,
                original_pose_standstill=stop['development_standstill_check']=='PASS',
                all_fixed_tail_physx_under5mm_s=all(math.hypot(r['vx_physx_m_s'],r['vy_physx_m_s'])<=.005 for r in tail),
                all_fixed_tail_filtered_under5mm_s=all(r['pose_filtered_speed_m_s']<=.005 for r in tail),
                all_fixed_tail_goal_under30mm2deg=all(math.hypot(*e[:2])<=.030 and abs(e[2])<=math.radians(2.) for e in errors))
            report['checks'].update(checks);report['retained_terminal_hold_component']=dict(checks=checks,standstill=stop,
                static=static,old_route_HOLD_unchanged=True,NOT_H6_whole_route_success=True)
            report['role']='RETAINED_FINAL_TARGET_HOLD_NOT_H6_ROUTE_SUCCESS'
        report['checks'].update(approved_bounded_observation_completed=True,final_measured_speed_under5mm_s=speed<=.005,
            actual_centre_polyline_under30mm=controller.travel<.030,one_hold_capture=port.actuator.capture_count==1,
            fixed2s_observation_completed=controller.complete)
    except (Exception,asyncio.CancelledError) as failure:
        report['error']=type(failure).__name__+': '+str(failure);report['checks']['approved_bounded_observation_completed']=False
        if observed is not None:report['last_attempted_observation_before_Pause']=lifecycle.rejected_observation_diagnostic(observed)
    finally:
        report=await lifecycle._finish_observation(port,report,rows,preflight=preflight,actuation=actuation,
            capture_count=0 if port.adapter is None else port.actuator.capture_count)
    return report


async def run_reverse_response_probe_once(path,*,authorized=False):
    import math,time
    if authorized is not True:raise ValueError('Separate ONE reverse response probe authority required')
    path=Path(path).resolve();packet=retained.read(path)
    raw=retained.read(packet['original_route_result']);prior=retained.read(raw['protocol'])
    if retained.runtime.task.sha(packet['original_route_result'])!=packet['original_route_result_sha256']:
        raise ValueError('Own immutable prior route evidence required')
    if (packet['required_tensor_snapshot']!=raw['actual_final_readback']['tensor']
            or packet['required_PAUSED_timeline']!=raw['actual_final_timeline']
            or packet['start_actual_xyyaw']!=list(lifecycle.RetainedMissionSDKHost._pose(raw['actual_final_readback']['tensor']))
            or any(packet[k]!=prior[k] for k in ('actor','original_static_record','original_preparation',
                'original_static_packet','geometry_file','grid_file','simulation_budget_s'))):
        raise ValueError('Exact own post-route boundary and original configuration files required')
    static=retained.read(prior['original_static_packet'])
    context=retained.static_binding(static,Path(prior['original_static_packet']),Path(prior['original_static_record']).parent,
        Path(prior['original_preparation']))
    geometry=retained.read(packet['geometry_file']);sdk=retained.runtime.live_sdk()
    with np.load(packet['grid_file']) as grid:
        port=RetainedReverseResponseSDKHost(sdk,packet,path,context=context,geometry=geometry,
            occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),resolution=float(grid['resolution_m']))
    controller=ReverseResponseProbe(packet['start_actual_xyyaw'],packet['probe_curvature_1_m'])
    if controller.contract!=packet['probe_contract']:raise ValueError('Exact predeclared diagnostic controller required')
    if not port.claim_once():return dict(status='EXISTING_PROBE_RETAINED_NO_NEW_ACTION')
    report=dict(verdict='HOLD',role='RESPONSE_DIAGNOSTIC_NOT_ROUTE_SUCCESS',checks={},new_route_trials=0,
        new_response_probe_trials=0,automatic_retry=False,use_for_final_statistics=False,contract=controller.contract)
    rows=[];preflight=actuation=False;observed=None
    try:
        report['preflight']=await port.preflight();preflight=True;port.subscribe()
        bridge=lifecycle.motion.MissionSDKActuationBridge(controller,port.actuator,
            guard=lambda pose,speed,info:port.guard(observed,info))
        observed=port.measure();port.guard(observed,None)
        speed=max(math.hypot(observed['vx_physx_m_s'],observed['vy_physx_m_s']),observed['pose_filtered_speed_m_s'])
        actuation=True;bridge.step(observed['time_s'],(observed['x_m'],observed['y_m'],observed['yaw_rad']),speed)
        began=time.monotonic();port.play();report['new_response_probe_trials']=1
        while not controller.complete:
            await asyncio.wait_for(port.next_sample(),timeout=max(.01,90.-(time.monotonic()-began)))
            observed=port.measure();observed=lifecycle.checked_measurement(observed)
            speed=max(math.hypot(observed['vx_physx_m_s'],observed['vy_physx_m_s']),observed['pose_filtered_speed_m_s'])
            info=bridge.step(observed['time_s'],(observed['x_m'],observed['y_m'],observed['yaw_rad']),speed)
            rows.append(dict(observed,command_speed_m_s=info['speed_m_s'],phase=info['terminal_stage'],
                requested_level_m_s=info['requested_level_m_s'],diagnostic_centre_polyline_m=info['diagnostic_centre_polyline_m'],
                **port.command_readback()))
        report['checks'].update(approved_bounded_observation_completed=True,
            final_measured_speed_under5mm_s=speed<=.005,actual_centre_polyline_under30mm=controller.travel<.030)
    except (Exception,asyncio.CancelledError) as failure:
        report['error']=type(failure).__name__+': '+str(failure)
        report['checks']['approved_bounded_observation_completed']=False
        if observed is not None:report['last_attempted_observation_before_Pause']=lifecycle.rejected_observation_diagnostic(observed)
    finally:
        report=await lifecycle._finish_observation(port,report,rows,preflight=preflight,actuation=actuation,capture_count=0)
    return report


def saved_response_probe_preparation_handoff(result_path,*,archive_failed=False):
    """Archive a completed probe's real state, never relabel it as a route."""
    import json
    result_path=Path(result_path).resolve();raw=retained.read(result_path)
    packet_path=Path(raw['protocol']);packet=retained.read(packet_path)
    archive=packet_path.parent/'probe_sources.zip'
    lifecycle.verify_workflow_sources(archive,packet['input_sha256'])
    history=lifecycle.reviewed_prior_GET_dependencies(packet['input_sha256'],archive)
    feedback=packet.get('version')==RetainedFeedbackHeldNeutralSDKHost.PROBE_VERSION
    targethold=packet.get('version')==RetainedFinalTargetHoldSDKHost.PROBE_VERSION
    compliant=packet.get('version')==RetainedCompliantHeldNeutralSDKHost.PROBE_VERSION
    approval_ok=(packet.get('explicit_response_probe_approval')==ITERATIVE_FIX_APPROVAL if feedback or targethold else
        packet.get('explicit_response_probe_approval')==COMPLIANT_HOLD_APPROVAL if compliant
        else packet.get('explicit_reverse_probe_approval')==REVERSE_PROBE_APPROVAL)
    if type(archive_failed) is not bool:raise ValueError('Explicit failed-boundary ARCHIVE-only selection required')
    outcome_ok=(feedback and raw['verdict']=='HOLD'
        and raw.get('error') in ('RuntimeError: Original5mm/s held-neutral speed guard','TimeoutError: Original16s neutral/ACK cap')
        and all(v for k,v in raw['checks'].items() if k!='approved_bounded_observation_completed')
        if archive_failed else raw['verdict']=='PASS' and all(raw['checks'].values()))
    if (result_path!=packet_path.parent/'probe_readback.json' or raw['protocol_sha256']!=retained.runtime.task.sha(packet_path)
            or not approval_ok
            or raw['role']!=('RETAINED_FINAL_TARGET_HOLD_NOT_H6_ROUTE_SUCCESS' if targethold else
                'HELD_NEUTRAL_RESPONSE_NOT_ROUTE_SUCCESS' if compliant or feedback else 'RESPONSE_DIAGNOSTIC_NOT_ROUTE_SUCCESS') or raw['new_route_trials']!=0
            or raw['new_response_probe_trials']!=1 or not outcome_ok
            or not raw['checks'] or not raw['restoration']['restored']
            or raw['trace_sha256']!=retained.runtime.task.sha(raw['trace'])):
        raise ValueError('Own one completed bounded diagnostic with complete cleanup required')
    prep=retained.read(raw['preparation_record']);final=raw['actual_final_readback'];state=raw['actual_final_timeline']
    snapshot=final['tensor'];baseline=prep['expected_nominal_motor_baseline']
    if (not final['stable_PAUSED_tensor'] or final['timeline']!=state or state['playing'] or state['stopped']
            or len(snapshot['names'])!=12 or len(snapshot['link_names'])!=13
            or snapshot['names']!=baseline['joint_names']
            or any(snapshot['gains'][k]!=baseline[k] for k in ('kp','kd','max_efforts'))
            or snapshot['joint_position_targets_rad']!=[0.]*12 or snapshot['joint_velocity_targets_rad_s']!=[0.]*12
            or raw['postflight']['environment']!=dict(row_count=4997,sha256=prep['protected_environment_sha256'],
                all145_source_settings_and_critical_environment_verified=True)):
        raise ValueError('Exact stable actual PAUSED12joint13link/original baseline/full environment required')
    pins=dict(packet['input_sha256'])
    for row in history['resolved_consumed_sources']:pins.pop(row['path'])
    files=(result_path,packet_path,Path(raw['trace']),archive,Path(raw['preparation_record']),
        Path(__file__).resolve(),ROOT/'src/patient_initial_stability/test/test_hospital_mission_sequence.py',*lifecycle.host_source_files())
    pins.update({str(p.resolve()):retained.runtime.task.sha(p) for p in files})
    revision=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()[:16]
    return dict(status=('EXACT_SAVED_FAILED_RESPONSE_BOUNDARY_FOR_NEW_ARCHIVE_NOT_PASS' if archive_failed else
            'EXACT_SAVED_RESPONSE_PROBE_BOUNDARY_FOR_NEW_ARCHIVE_NOT_ROUTE_COMPLETION'),
        source_result=str(result_path),source_result_sha256=retained.runtime.task.sha(result_path),
        source_archive=str(result_path.parent/'next_cohort_handoff'/revision/'sources.zip'),
        preparation_record=raw['preparation_record'],original_verdict=raw['verdict'],
        response_probe_PASS_is_NOT_H6_mission_PASS=True,old_route_HOLD_unchanged=True,
        actual_layer_texts=final['layer_texts'],same_stage=True,Root_Session_unchanged_during_GET=True,
        SDK_GET=dict(actor=packet['actor'],timeline=state,actual_tensor_snapshot=snapshot,
            tensor_stable_during_GET=True,observed_hold_confirmation=False,
            original_motor_baseline_backend_readback_verified=True),
        fields_derived_from_saved_actual_postflight_NOT_a_new_static_or_route_admission=True,
        new_live_actions=0,input_sha256=pins)


def checked_compliant_probe_condition(handoff_path):
    """Actual PASS condition, not a user intention or SOFTWARE/MODEL result."""
    if handoff_path is None:raise ValueError('Completed compliant response PASS handoff required')
    handoff=retained.read(handoff_path);result_path=Path(handoff['source_result'])
    if handoff!=saved_response_probe_preparation_handoff(result_path):raise ValueError('Exact completed response handoff required')
    raw=retained.read(result_path);packet=retained.read(raw['protocol'])
    if (packet.get('version')!=RetainedCompliantHeldNeutralSDKHost.PROBE_VERSION
            or packet.get('explicit_response_probe_approval')!=COMPLIANT_HOLD_APPROVAL or raw['verdict']!='PASS'):
        raise ValueError('Conditional approval requires own actual compliant probe PASS')
    return handoff


def checked_final_target_hold_probe_condition():
    path=retained.GATE/'hospital_h6_experiments/cohort_16_held_static_r01/route/P_dynamics_full_envelope/final_target_hold_response_probe_r01/probe_readback.json'
    raw=retained.read(path);packet=retained.read(raw['protocol'])
    saved_response_probe_preparation_handoff(path)
    if (raw['verdict']!='PASS' or raw.get('error') or packet['version']!=RetainedFinalTargetHoldSDKHost.PROBE_VERSION
            or packet['explicit_response_probe_approval']!=ITERATIVE_FIX_APPROVAL
            or raw['role']!='RETAINED_FINAL_TARGET_HOLD_NOT_H6_ROUTE_SUCCESS'
            or raw['new_route_trials']!=0 or not all(raw['checks'].values())
            or not raw['retained_terminal_hold_component']['NOT_H6_whole_route_success']):
        raise ValueError('Own actual fixed5s/final3s retained-target hold PASS required; old H6 HOLD unchanged')
    return path,raw


def export_response_probe_preparation_handoff(result_path,*,archive_failed=False):
    value=saved_response_probe_preparation_handoff(result_path,archive_failed=archive_failed)
    path=Path(value['source_archive']).with_name('handoff.json')
    if path.exists():
        if retained.read(path)!=value:raise ValueError('Existing probe handoff retained; no replacement')
    else:
        retained.evidence.write_exclusive(path,value)
        lifecycle.freeze_workflow_sources(Path(value['source_archive']),value['input_sha256'])
    return path
