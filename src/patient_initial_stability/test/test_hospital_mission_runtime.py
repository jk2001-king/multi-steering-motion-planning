"""Offline software tests; fixtures are never presented as actual H6 trials."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import asyncio
import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hospital_mission_runtime as mission


@pytest.fixture(scope='module')
def protocol():
    return mission.static_definition()


@pytest.fixture
def context(protocol):
    return mission.static_context(copy.deepcopy(protocol), mission.H6/'static_protocol.json')


def test_prospective_definition_uses_actual_own_preparation_not_old_h5_trial(protocol):
    assert protocol['new_robot_root'] == '/World/patient_transport_h6_cohort_01_common_preparation_r01'
    assert protocol['maximum_static_trials'] == 1
    assert protocol['maximum_Stop_calls'] == protocol['maximum_route_trials'] == 0
    assert protocol['user_approval'] == mission.APPROVAL
    assert protocol['automatic_retry'] is protocol['actual_motion_authorized'] is False
    assert protocol['unchanged_original_static_limits'] == mission.shared.stationary.LIMITS
    assert protocol['unchanged_original_static_guards'] == mission.shared.stationary.GUARDS
    assert protocol['additional_sampled_query_limits'] == mission.shared.stationary.QUERY_LIMITS
    assert protocol['supports'] == list(mission.shared.stationary.SUPPORTS)
    assert protocol['protected_environment_sha256'] == '3a29fcefdc87751ad38304071370d0107a9e3d4400e3cbfc7273873e3aa9bd25'
    assert len(protocol['source_actor_node_paths']) == 145
    mission.runtime.task.verify_pins(protocol['input_sha256'])


def test_static_force_contact_zero_pause_and_raw_replay_use_exact_original_code(context):
    static = context.operations.static
    assert static.__code__ is mission.shared.run_retained_static.__code__
    assert context.operations.replay_static.__code__ is mission.shared.load_fresh_static_inputs.__code__
    scope = static.__globals__
    assert scope['STATIC_RUN_ID'] == 'hospital_h6_static_r01'
    assert scope['STATIC_DESIGN'] == mission.H6/'static_protocol.json'
    assert scope['preparation'] is context.preparation
    assert scope['cohort'] is context.cohort
    assert scope['completed_static'].physical.evaluate_static is mission.shared.completed_static.physical.evaluate_static
    assert scope['evidence'].confirm_pause is mission.shared.evidence.confirm_pause
    assert scope['completed_static'].frozen_query_functions is mission.runtime.query_functions
    assert scope['evaluate_support_queries_for_actor'] is mission.support_evaluation
    assert mission.shared.STATIC_RUN_ID != context.static_run_id
    assert mission.shared.preparation.ROBOT != context.preparation.ROBOT
    assert not any(hasattr(context.operations, k) for k in ('stage', 'route', 'initialize_staged_display'))


def test_actual_4997_h6_rows_pass_original_exact_comparator(context, protocol):
    actual = mission.read(mission.H6/'actual_common_environment_rows.json')['compact_rows']
    archive = context.preparation.load_manifest()['compact_environment_rows']
    compact = context.preparation.compact
    assert compact.compare_rows(archive, actual, context.preparation.SELECTED, True) == protocol['protected_environment_sha256']
    for name in context.preparation.SELECTED:
        assert archive[name]['collision_enabled'] is True
        assert actual[name]['collision_enabled'] is False


@pytest.mark.parametrize('damage', ['missing', 'new', 'digest', 're_enabled_flag'])
def test_any_other_environment_row_or_four_flags_change_fails(context, damage):
    actual = copy.deepcopy(mission.read(mission.H6/'actual_common_environment_rows.json')['compact_rows'])
    archive = context.preparation.load_manifest()['compact_environment_rows']
    name = next(iter(actual))
    if damage == 'missing': actual.pop(name)
    elif damage == 'new': actual['/World/NOT_AUTHORIZED'] = actual[name]
    elif damage == 'digest': actual[name]['row_sha256'] = '0'*64
    else: actual[context.preparation.SELECTED[0]]['collision_enabled'] = True
    with pytest.raises(ValueError):
        context.preparation.compact.compare_rows(archive, actual, context.preparation.SELECTED, True)


def test_h6_27_queries_are_fixed_current_endpoints_and_start_interior_not_h5_coordinates(protocol):
    rows = protocol['sampled_cooked_support_queries']
    assert len(rows) == len({r['id'] for r in rows}) == 27
    assert rows[0]['xy_m'] == protocol['start_xyyaw'][:2]
    assert rows[5]['xy_m'] == protocol['goal_xyyaw'][:2]
    assert len([r for r in rows if r['kind'] == 'endpoint']) == 10
    assert len([r for r in rows if r['kind'] == 'initial_footprint']) == 16
    assert rows[1]['xy_m'] == [protocol['start_xyyaw'][0]+.825, protocol['start_xyyaw'][1]+.4925]
    assert rows[-1]['xy_m'] == [17.646392822265625, 25.0766455078125]
    assert protocol['selected_original_cart'] == '/World/Hospital/SM_SupplyCart_02a7'
    assert all('SM_SupplyCart_01e14' not in p for p in protocol['query_known_enabled_colliders'])
    assert len(protocol['additional_disabled_subfloor_probes']) == 4


def synthetic_hits(protocol):
    """Explicit software ray fixture, NOT observations from the live scene."""
    result = []
    for spec in protocol['sampled_cooked_support_queries']:
        hits = [dict(collision=protocol['supports'][0], position_m=[*spec['xy_m'], protocol['support_top_m']],
            normal=[0., 0., 1.], distance_m=spec['origin_z_m']-protocol['support_top_m'])]
        if spec['kind'] == 'cart_control':
            hits.append(dict(collision=protocol['selected_original_cart']+'/SM_SupplyCart_02a',
                position_m=[*spec['xy_m'], .4], normal=[0., 0., 1.], distance_m=1.6))
        result.append(dict(id=spec['id'], all_hits=hits, errors=[]))
    return result


class Rays:
    def __init__(self, rows): self.rows, self.calls = rows, []
    def raycast_all(self, origin, direction, length, callback):
        row = self.rows[len(self.calls)]
        self.calls.append((origin, direction, length))
        for hit in row['all_hits']:
            assert callback(SimpleNamespace(collision=hit['collision'], position=hit['position_m'],
                normal=hit['normal'], distance=hit['distance_m'])) is True


def collected(protocol, raw):
    rays = Rays(raw)
    rows = mission.runtime.collect_task_queries(protocol, rays, lambda *v: list(v))
    assert len(rows) == len(rays.calls) == 27
    return rows


def test_all_hit_collector_and_two_channel_contract_reuse_with_h6_configuration(protocol):
    raw = synthetic_hits(protocol)
    rows = collected(protocol, raw)
    assert [r['all_hits'] for r in rows] == [r['all_hits'] for r in raw]
    result = mission.support_evaluation(protocol, rows)
    assert all(result['checks'].values())
    assert result['original_evaluator_code_unchanged'] is True
    assert result['whole_path_cooked_support_or_static_or_drive_admission'] is False


@pytest.mark.parametrize('fault', ['missing_floor', 'unknown', 'unknown_self_prefix', 'disabled',
    'off_ray', 'wrong_floor_height', 'bad_normal', 'endpoint_cart', 'missing_cart', 'query_error'])
def test_query_faults_cannot_turn_into_support_or_contact_policy_exemptions(protocol, fault):
    raw = synthetic_hits(protocol)
    hit = copy.deepcopy(raw[0]['all_hits'][0])
    if fault == 'missing_floor': raw[0]['all_hits'] = []
    elif fault == 'missing_cart': raw[-1]['all_hits'] = raw[-1]['all_hits'][:1]
    elif fault == 'query_error': raw[0]['errors'] = ['software query exception']
    else:
        if fault == 'unknown': hit['collision'] = '/World/UNKNOWN'
        elif fault == 'unknown_self_prefix': hit['collision'] = protocol['new_robot_root']+'/UNVERIFIED'
        elif fault == 'disabled': hit['collision'] = protocol['approved_disabled_subfloor_paths'][0]
        elif fault == 'off_ray': hit['position_m'][0] += .001
        elif fault == 'wrong_floor_height': hit['position_m'][2] += .001
        elif fault == 'bad_normal': hit['normal'][2] = .98
        elif fault == 'endpoint_cart': hit['collision'] = protocol['selected_original_cart']+'/SM_SupplyCart_02a'
        raw[0]['all_hits'] = [hit]
    # Preserve supplied transaction errors after the SDK collector; the latter
    # records SDK exceptions itself rather than reading any software fixture.
    rows = collected(protocol, raw)
    if fault == 'query_error': rows[0]['errors'] = raw[0]['errors']
    result = mission.support_evaluation(protocol, rows)
    assert not all(result['checks'].values())


def test_exact_source_self_hits_retained_but_never_replace_floor(protocol):
    raw = synthetic_hits(protocol)
    own = copy.deepcopy(raw[0]['all_hits'][0])
    own['collision'] = protocol['new_robot_root']+'/base_link'
    raw[0]['all_hits'].insert(0, own)
    rows = collected(protocol, raw)
    assert rows[0]['own_robot_occlusion_hits'] == [own]
    assert all(mission.support_evaluation(protocol, rows)['checks'].values())
    rows[0]['all_hits'] = [own]
    rows[0]['all_nonself_hits'] = []
    assert not all(mission.support_evaluation(protocol, rows)['checks'].values())


def test_contact_identity_binding_never_exempts_hospital_obstacles(context):
    classify = context.stationary.classify_contact
    wheel = context.cohort.ROBOT+'/'+context.stationary.WHEELS['active_front']+'/collisions/mesh_0/capsule'
    assert classify([wheel, context.stationary.SUPPORTS[0]]) == 'wheel_support'
    assert classify([context.cohort.ROBOT+'/base_link/collisions/a', context.stationary.SUPPORTS[0]]) == 'nonwheel_support'
    assert classify([wheel, '/World/Hospital/SM_Door_02b3/Mesh']) == 'hospital_obstacle'
    assert classify([wheel, '/World/UNKNOWN']) == 'unclassified_external'


def test_refuses_consent_before_sdk_import(monkeypatch):
    monkeypatch.setattr(mission.runtime, 'live_sdk', lambda: pytest.fail('No SDK access without consent'))
    with pytest.raises(ValueError, match='consent'):
        asyncio.run(mission.run_static_once('/nonexistent', authorized=False))
    with pytest.raises(ValueError, match='consent'):
        mission.schedule_static('/nonexistent', authorized=False)


@pytest.mark.parametrize('field,value', [('maximum_Stop_calls', 1), ('maximum_route_trials', 1),
    ('maximum_static_trials', 2), ('automatic_retry', True), ('user_approval', {})])
def test_static_context_refuses_scope_expansion(protocol, field, value):
    p = copy.deepcopy(protocol); p[field] = value
    with pytest.raises(ValueError): mission.static_context(p, mission.H6/'static_protocol.json')


def test_no_second_dispatch_even_before_sdk(monkeypatch, context, tmp_path):
    context.output = tmp_path
    (tmp_path/'static').mkdir()
    (tmp_path/'static/dispatch_reserved.json').touch()
    monkeypatch.setattr(mission, 'load_static', lambda path: context)
    monkeypatch.setattr(mission.runtime, 'live_sdk', lambda: pytest.fail('Duplicate dispatch must not touch SDK'))
    result = asyncio.run(mission.run_static_once('software_fixture', authorized=True))
    assert result['status'] == 'PRIOR_STATIC_DISPATCH_RETAINED_NO_NEW_ACTION'


def test_library_import_export_and_live_entry_have_no_actor_camera_stop_or_route_command():
    import ast
    entries = (mission.static_context, mission.run_static_once, mission.schedule_static,
               mission.run_boundary_get, mission.schedule_boundary_get,
               mission.run_reviewed_static_once,mission.schedule_reviewed_static)
    tree = ast.parse('\n'.join(inspect.getsource(f) for f in entries))
    forbidden = {'play', 'pause', 'stop', 'reset', 'Set', 'Save', 'initialize_staged_display',
                 'constructor_for', 'prepare_fresh_live_trial', 'stage_after_archive', 'SimulationApp'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden
                   for n in ast.walk(tree))
    assert 'static_binding' in inspect.getsource(mission.static_context)
    assert 'run_retained_static' in inspect.getsource(mission.static_binding)


def test_recorded_actual_preplay_abort_is_authenticated_with_exact_executed_source_zip():
    protocol, history = mission.recorded_unstarted_static()
    assert protocol['new_robot_root'].endswith('h6_cohort_01_common_preparation_r01')
    assert history['phase_before_static_operation_verified'] is True
    assert history['no_physical_static_reservation_or_result'] is True
    assert len(history['exact_executed_source_resolution']) == 2
    assert {Path(r['path']).name for r in history['exact_executed_source_resolution']} == {
        'hospital_mission_runtime.py', 'test_hospital_mission_runtime.py'}


def test_native_snapshot_delta_and_literal_transaction_replay_do_not_skip_metadata():
    first = {'/': dict(spec_type='PseudoRootSpec', info={})}
    last = copy.deepcopy(first)
    last['/World/A'] = dict(spec_type='PrimSpec', info=dict(specifier='over'))
    last['/World/A.p'] = dict(spec_type='AttributeSpec', info=dict(default=.1, typeName='float', custom=False))
    changes = mission.snapshot_deltas(first, last)
    assert len(changes) == 6
    assert mission.apply_recorded_field_changes(first, changes) == last
    assert first == {'/': dict(spec_type='PseudoRootSpec', info={})}
    bad = copy.deepcopy(changes); bad[-1]['before'] = 'invented'
    with pytest.raises(ValueError, match='apply exactly'):
        mission.apply_recorded_field_changes(first, bad)


def test_offline_declared_preparation_fields_are_explicit_not_fabricated_missing_session_text():
    pytest.importorskip('pxr.PhysxSchema')
    snapshots, proof = mission.expected_staged_native_fields()
    assert len(snapshots['Root']) == 1162
    assert len(snapshots['Session']) == 1339
    assert proof['recorded_environment_field_transactions'] == 232
    assert proof['prepared_Session_text_was_NOT_saved_or_recovered'] is True
    assert proof['camera_reference_is_actual_completed_Stop_NOT_assumed_final_preparation'] is True
    assert proof['live_GET_must_independently_compare_all145_settings_to_own_saved_SAME_SDK_live_reference'] is True
    p = mission.read(mission.H6/'preparation_protocol.json')
    assert snapshots['Session'][p['retired_robot_root']]['info']['active'] is False
    assert snapshots['Session'][p['marker']]['info']['customData'] == dict(
        robot_root=p['new_robot_root'], status='fresh_static_pending_NO_PLAY', cohort=p['cohort'])
    # Changes to physics, metadata, other cameras or the actor can never be
    # classified as Top XY/aperture-only display differences.
    changed = copy.deepcopy(snapshots['Session'])
    changed[p['new_robot_root']+'.isaac:namespace']['info']['default'] = 'not_unvalued'
    rows = mission.snapshot_deltas(snapshots['Session'], changed)
    assert len(rows) == 1
    with pytest.raises(ValueError): mission.boundary._validate_display_row(rows[0])


def test_actual_root9_session10_metadata_differences_are_not_falsely_waived_as_current_edits():
    pytest.importorskip('pxr.PhysxSchema')
    result = mission.review_recorded_boundary_get()
    assert result['Root_actual_bytes_identical_to_actual_completed_Stop_and_preparation'] is True
    assert result['prior_Root9_and_Session10_UI_metadata_missing_fields_were_cross_registry_comparison_artifacts'] is True
    assert result['same_stopped_actor_all145_settings_13bodies_12joints_and4997_rows_verified'] is True
    assert result['Session_review']['full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults'] is True
    assert len(result['Session_review']['exact_Top_default_differences']) == 3
    assert result['actual_motion_authorized'] is False
    assert result['static_continuation_requires_separate_user_launch_approval'] is True


@pytest.fixture
def opaque_camera_text():
    return '''#sdf 1.4.32
def Camera "OmniverseKit_Top" (
    hide_in_stage_window = true
    no_delete = true
)
{
    float horizontalAperture = 146.3383
    float verticalAperture = 120.088295
    float2 clippingRange = (1, 10000000)
    matrix4d xformOp:transform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (14,16,50001,1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
'''


def test_canonical_review_uses_typed_float_values_and_preserves_opaque_metadata(opaque_camera_text):
    pytest.importorskip('pxr.Usd')
    current = opaque_camera_text.replace('146.3383', '251.62486').replace('120.088295', '225.37486')
    current = current.replace('(14,16,50001,1)', '(17,16.5,50001,1)')
    result = mission.canonical_display_only_review(opaque_camera_text, current)
    assert len(result['exact_Top_default_differences']) == 3
    assert result['full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults'] is True


@pytest.mark.parametrize('fault', ['opaque_hide', 'opaque_delete', 'clip', 'Z', 'orientation', 'added_prim', 'type'])
def test_canonical_review_rejects_opaque_or_non_display_changes_not_just_iterated_known_fields(opaque_camera_text, fault):
    pytest.importorskip('pxr.Usd')
    current = opaque_camera_text
    if fault == 'opaque_hide': current = current.replace('hide_in_stage_window = true', 'hide_in_stage_window = false')
    elif fault == 'opaque_delete': current = current.replace('no_delete = true', 'no_delete = false')
    elif fault == 'clip': current = current.replace('(1, 10000000)', '(1, 100)')
    elif fault == 'Z': current = current.replace('(14,16,50001,1)', '(14,16,8,1)')
    elif fault == 'orientation': current = current.replace('(1,0,0,0)', '(0,1,0,0)')
    elif fault == 'added_prim': current += '\ndef Xform "UNREVIEWED" {}\n'
    elif fault == 'type': current = current.replace('float horizontalAperture', 'double horizontalAperture')
    with pytest.raises(ValueError): mission.canonical_display_only_review(opaque_camera_text, current)


@pytest.fixture(scope='module')
def revised_protocol():
    pytest.importorskip('pxr.PhysxSchema')
    return mission.reviewed_static_definition()


def test_reviewed_static_private_binding_reuses_original_observation_and_replay_not_old_dispatch(revised_protocol):
    path=mission.REVIEWED_STATIC_FOLDER/'static_protocol.json'
    c=mission.reviewed_static_context(revised_protocol,path)
    assert c.operations.static.__code__ is mission.shared.run_retained_static.__code__
    assert c.operations.replay_static.__code__ is mission.shared.load_fresh_static_inputs.__code__
    assert c.cohort.ROBOT==revised_protocol['new_robot_root']
    assert c.preparation.LIVE==mission.REVIEWED_STATIC_FOLDER/'same_actor_handoff.json'
    assert c.static_folder==mission.REVIEWED_STATIC_FOLDER
    assert c.operations.static.__globals__['STATIC_DESIGN']==path
    assert c.operations.static.__globals__['STATIC_OUTPUT']==c.static_folder
    assert c.protocol_path==path
    assert c.preparation.LIVE != mission.H6/'preparation.json'
    assert c.static_folder != mission.H6/'static'
    assert revised_protocol['launch_requires_NEW_explicit_single_static_consent'] is True
    assert revised_protocol['actual_motion_authorized'] is False
    original=mission.read(mission.H6/'static_protocol.json')
    for key in ('start_xyyaw','supports','support_top_m','sampled_cooked_support_queries',
        'expected_nominal_motor_baseline','unchanged_original_static_limits','unchanged_original_static_guards',
        'protected_environment_sha256'):
        assert revised_protocol[key]==original[key]
    mission.runtime.task.verify_pins(revised_protocol['input_sha256'])


@pytest.mark.parametrize('field,value', [('maximum_static_trials',2),('maximum_route_trials',1),
    ('maximum_Stop_calls',1),('automatic_retry',True),('start_xyyaw',[0.,0.,0.]),
    ('supports',['/World/UNREVIEWED_SUPPORT']),('support_top_m',.002),('launch_requires_NEW_explicit_single_static_consent',False)])
def test_reviewed_scope_and_physical_changes_rejected_even_with_recomputed_definition_digest(revised_protocol,field,value):
    p=copy.deepcopy(revised_protocol);p[field]=value;p.pop('definition_sha256');p['definition_sha256']=mission.shared.digest(p)
    with pytest.raises(ValueError):mission.reviewed_static_context(p,mission.REVIEWED_STATIC_FOLDER/'static_protocol.json')


def test_same_actor_handoff_is_separate_receipt_and_preserves_every_original_physical_field():
    prepared=mission.read(mission.H6/'preparation.json');before=copy.deepcopy(prepared)
    live=mission.read(mission.H6/'static_boundary_GET/live_boundary.json')
    reviewed=dict(full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults=True,
        exact_Top_default_differences=[])
    handoff=mission.same_actor_handoff(prepared,live,live['actual_layer_texts'],reviewed)
    assert prepared==before
    for key,value in before.items():
        if key not in ('Root_sha256','Session_sha256'):assert handoff[key]==value
    assert handoff['Root_sha256']==before['Root_sha256']
    assert handoff['Session_sha256']==live['actual_layer_sha256']['Session']
    assert handoff['reviewed_display_handoff_NOT_new_staging']['original_prepared_Session_sha256']==before['Session_sha256']
    assert handoff['reviewed_display_handoff_NOT_new_staging']['original_preparation_fields_NOT_overwritten'] is True


def test_live_display_bridge_does_not_waive_Root_even_for_UI_metadata(opaque_camera_text):
    before=dict(Root='the exact recorded Root',Session=opaque_camera_text)
    with pytest.raises(ValueError,match='Root bytes'):
        mission.review_live_display_texts(before,dict(before,Root='changed Root UI or physics'))


@pytest.mark.parametrize('authorized,approval', [(False,None),(True,None),(True,mission.APPROVAL),
    (True,dict(answer_ko='검증 후 같은 로봇 정적 최초 1회 진행 승인',question_item_id=''))])
def test_original_static_approval_never_automatically_authorizes_revised_launch(monkeypatch,authorized,approval):
    monkeypatch.setattr(mission,'load_reviewed_static',lambda _:pytest.fail('No protocol or SDK before NEW consent'))
    monkeypatch.setattr(mission.runtime,'live_sdk',lambda:pytest.fail('No live SDK without NEW consent'))
    with pytest.raises(ValueError,match='NEW explicit'):
        asyncio.run(mission.run_reviewed_static_once('not_loaded',authorized=authorized,approval=approval))
    with pytest.raises(ValueError,match='NEW explicit'):
        mission.schedule_reviewed_static('not_loaded',authorized=authorized,approval=approval)


@pytest.mark.parametrize('outcome',['PASS','FAIL','static_error','replay_error'])
def test_revised_orchestration_exactly_once_and_preserves_errors_with_SOFTWARE_SDK_ONLY(monkeypatch,tmp_path,outcome):
    """No Isaac SDK/physics is imported; code-object reuse is tested separately."""
    import sys
    from types import ModuleType
    root=tmp_path/'SOFTWARE_ONLY_NOT_A_TRIAL';root.mkdir()
    protocol_path=_ijcas_path(root, 'protocol.json');mission.evidence.write_exclusive(protocol_path,dict(software_fixture=True))
    live=mission.read(mission.H6/'static_boundary_GET/live_boundary.json');clock=dict(mission.boundary.STOPPED)
    class Layer:
        def __init__(self,key):self.key=key
        def ExportToString(self):return live['actual_layer_texts'][self.key]
    class Stage:
        def GetRootLayer(self):return Layer('Root')
        def GetSessionLayer(self):return Layer('Session')
    class Timeline:
        # Deliberately no play/stop/pause/reset method on this SOFTWARE SDK.
        def is_playing(self):return clock['playing']
        def is_stopped(self):return clock['stopped']
        def get_current_time(self):return clock['current_time_s']
    class App:
        async def next_update_async(self):pass
    stage=Stage();timeline=Timeline();counts=dict(static=0,replay=0)
    async def static(context):
        counts['static']+=1
        if outcome=='static_error':raise RuntimeError('SOFTWARE_INJECTED_STATIC_ERROR')
        clock.update(playing=False,stopped=False,current_time_s=3.1)
        return dict(verdict='FAIL' if outcome=='FAIL' else 'PASS')
    def replay(context):
        counts['replay']+=1
        if outcome=='replay_error':raise RuntimeError('SOFTWARE_INJECTED_REPLAY_ERROR')
        return None,None,dict(verdict='PASS')
    context=SimpleNamespace(static_folder=root,protocol_path=protocol_path,
        protocol=dict(saved_verified_GET=str(mission.H6/'static_boundary_GET/live_boundary.json'),required_display_settings={}),
        preparation=SimpleNamespace(LIVE=_ijcas_path(root, 'handoff.json')),stationary=SimpleNamespace(RESULT=_ijcas_path(root, 'SOFTWARE_RESULT.json')),
        operations=SimpleNamespace(static=static,replay_static=replay))
    sdk=SimpleNamespace(omni=SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:stage)),
        timeline=SimpleNamespace(get_timeline_interface=lambda:timeline),
        kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda:App()))))
    carb=ModuleType('carb');carb.__path__=[];settings=ModuleType('carb.settings')
    settings.get_settings=lambda:SimpleNamespace(get=lambda key:None);carb.settings=settings
    monkeypatch.setitem(sys.modules,'carb',carb);monkeypatch.setitem(sys.modules,'carb.settings',settings)
    monkeypatch.setattr(mission,'load_reviewed_static',lambda _:context)
    monkeypatch.setattr(mission,'recorded_unstarted_static',lambda:None)
    monkeypatch.setattr(mission.runtime,'live_sdk',lambda:sdk)
    monkeypatch.setattr(mission,'review_live_display_texts',lambda a,b:dict(
        full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults=True,
        exact_Top_default_differences=[]))
    def software_equal_text_difference(before,after):
        # Native parsing/opaque-field faults have separate real-Sdf tests.
        # This orchestration fixture intentionally runs without the Isaac SDK.
        assert before==after
        return []
    monkeypatch.setattr(mission.shared,'layer_field_differences',software_equal_text_difference)
    approval=dict(question_item_id='call_SOFTWARE_FIXTURE_NOT_USER_AUTHORIZATION:0',
        answer_ko='검증 후 같은 로봇 정적 최초 1회 진행 승인')
    if outcome.endswith('error'):
        with pytest.raises(RuntimeError,match='SOFTWARE_INJECTED'):
            asyncio.run(mission.run_reviewed_static_once(protocol_path,authorized=True,approval=approval))
        error=mission.read(_ijcas_path(root, 'static_operation_error.json'))
        assert error['actual_current_layer_texts']==live['actual_layer_texts']
        assert error['exact_changes_since_entry']==dict(Root=[],Session=[])
        assert error['automatic_retry'] is False
        assert not (_ijcas_path(root, 'static_only_closeout.json')).exists()
    else:
        result=asyncio.run(mission.run_reviewed_static_once(protocol_path,authorized=True,approval=approval))
        assert result['static_verdict']==outcome
        assert result['own_raw_replay_verdict']==('PASS' if outcome=='PASS' else None)
        assert result['additional_Stop_calls']==result['new_route_trials']==0
    assert counts['static']==1
    assert counts['replay']==int(outcome in ('PASS','replay_error'))
    assert mission.read(_ijcas_path(root, 'actual_layers_before_preflight.json'))['actual_layer_texts']==live['actual_layer_texts']
    duplicate=asyncio.run(mission.run_reviewed_static_once(protocol_path,authorized=True,approval=approval))
    assert duplicate['status']=='PRIOR_REVIEWED_STATIC_DISPATCH_RETAINED_NO_NEW_ACTION'
    assert counts['static']==1


def test_same_sdk_anonymous_source_actor_code_and_all_145_settings(context, protocol):
    pytest.importorskip('pxr.PhysxSchema')
    from pxr import Usd, UsdShade
    stage = Usd.Stage.CreateInMemory('H6_STATIC_SOFTWARE_FIXTURE_NOT_LIVE.usda')
    p = context.preparation
    UsdShade.Material.Define(stage, p.old.constructor.MATERIAL)
    result = p.old.constructor_for(context.cohort.ROBOT)(stage, protocol['start_xyyaw'])
    actual = context.operations.source_for_live_actor(context, stage)
    assert actual['source_constructor_readback'] == result
    assert len(actual['fixed_configuration_rows']) == 145
    assert len(result['rigid_bodies']) == 13
    assert len(result['revolute_joint_names']) == 12
    stage.GetPrimAtPath(context.cohort.ROBOT+'/base_link').GetAttribute('physics:mass').Set(246.)
    with pytest.raises(ValueError, match='configuration'):
        context.operations.source_for_live_actor(context, stage)


def test_actual_saved_static_pass_does_not_imply_stricter_stopped_release_pass():
    raw=mission.read(mission.REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json')
    assert raw['verdict']=='PASS' and raw['sample_count']==121
    normalized,review=mission.saved_post_static_release_review(raw)
    assert normalized['actual_post_static_tensor_snapshot']==raw['actual_post_static_tensor_snapshot']
    assert review['status']=='SAVED_POST_STATIC_RELEASE_HOLD'
    assert review['horizontal_speed_m_s']==pytest.approx(.005366004333059057,abs=1e-15)
    assert review['original_horizontal_stop_limit_m_s']==.005
    assert [k for k,v in review['checks'].items() if not v]==['horizontal_speed_within_original_stop_limit']
    assert review['saved_snapshot_compared_to_ITSELF_not_fresh_live_state'] is True


@pytest.mark.parametrize('speed,expected',[(.004999,'PASS'),(.005,'PASS'),(.005001,'HOLD')])
def test_saved_release_review_preserves_exact5mm_s_boundary_on_SOFTWARE_modified_copies(speed,expected):
    raw=copy.deepcopy(mission.read(mission.REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json'))
    raw['actual_post_static_tensor_snapshot']['velocity'][:2]=[speed,0.]
    _,review=mission.saved_post_static_release_review(raw)
    assert review['status']=='SAVED_POST_STATIC_RELEASE_'+expected
    assert review['original_horizontal_stop_limit_m_s']==.005
    assert raw['actual_post_static_tensor_snapshot']['velocity'][0]==speed


@pytest.mark.parametrize('fault',['gains','quaternion','shape','nonfinite','joint_names'])
def test_saved_release_diagnosis_never_waives_other_recorded_gate_faults(fault):
    raw=copy.deepcopy(mission.read(mission.REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json'))
    a=raw['actual_post_static_tensor_snapshot'];a['velocity'][:2]=[.004,0.]
    if fault=='gains':a['gains']['kd'][0]+=1.
    elif fault=='quaternion':a['transform'][6]=.5
    elif fault=='shape':a['all_thirteen_link_velocities_at_CoM'].pop()
    elif fault=='nonfinite':a['positions'][0]=float('nan')
    else:a['names'][0]='UNREVIEWED_JOINT'
    if fault in ('gains','quaternion'):
        _,review=mission.saved_post_static_release_review(raw)
        assert review['status']=='SAVED_POST_STATIC_RELEASE_HOLD'
    else:
        with pytest.raises(ValueError):mission.saved_post_static_release_review(raw)


def test_actual_original_physical_replay_pass_and_release_hold_reproduce_without_any_live_SDK(monkeypatch):
    monkeypatch.setattr(mission.runtime,'live_sdk',lambda:pytest.fail('Saved review must not touch live SDK'))
    previous=mission.shared.preparation.verify
    review=mission.review_completed_reviewed_static()
    assert mission.shared.preparation.verify is previous
    assert review['status']=='ACTUAL_H6_STATIC_PASS_RAW_PHYSICAL_REPLAY_PASS_RETAINED_RELEASE_HOLD'
    assert review['physical_evaluation_reproduced_exactly'] is True
    assert review['source145_environment4997_contact_counts_clocks_gains_and_complete_tensor_pass_before_original_release_gate'] is True
    assert review['additional_Stop_static_or_route_trials']==0
    assert review['actual_motion_authorized'] is False
    assert len(review['executed_input_resolution']['historical_source_resolution'])==2


def test_historical_resolution_does_not_exempt_any_other_input(tmp_path):
    path=tmp_path/'CHANGED_SOFTWARE_INPUT.txt';path.write_text('new bytes')
    with pytest.raises(ValueError,match='physical dependency changed'):
        mission.verify_reviewed_executed_pins({str(path):'0'*64})


def test_exact_numpy_scalar_normalization_is_available_without_loosening_strict_writer(tmp_path):
    path = tmp_path/'evidence.json'
    mission.shared.write_runtime_evidence(path, dict(a=np.float64(.125), b=np.int64(12)))
    saved = json.loads(path.read_bytes())
    assert saved['a'] == .125 and saved['b'] == 12
    with pytest.raises((TypeError, ValueError)):
        mission.evidence.write_exclusive(tmp_path/'bad.json', dict(a=np.float64(.125)))
    assert not (tmp_path/'bad.json').exists()


def _retained_GET_sdk_fixture():
    """Getter-only facade backed by SAVED data, never a current live trial."""
    raw=mission.read(mission.REVIEWED_STATIC_FOLDER/'hospital_h6_reviewed_static_r01.json')
    snapshot=copy.deepcopy(raw['actual_post_static_tensor_snapshot'])
    class View:
        count=1;max_dofs=12;max_links=13
        shared_metatype=SimpleNamespace(dof_names=snapshot['names'],link_names=snapshot['link_names'])
        def check(self):return True
        def __getattr__(self,name):
            mapping=dict(get_root_transforms=('transform',(1,7)),get_root_velocities=('velocity',(1,6)),
                get_dof_positions=('positions',(1,12)),get_dof_velocities=('joint_velocities_rad_s',(1,12)),
                get_dof_position_targets=('joint_position_targets_rad',(1,12)),
                get_dof_velocity_targets=('joint_velocity_targets_rad_s',(1,12)),
                get_link_transforms=('all_thirteen_link_world_transforms_xyzw',(1,13,7)),
                get_link_velocities=('all_thirteen_link_velocities_at_CoM',(1,13,6)))
            gains=dict(get_dof_stiffnesses='kp',get_dof_dampings='kd',get_dof_max_forces='max_efforts')
            if name in mapping:
                key,shape=mapping[name];return lambda:np.asarray(snapshot[key],np.float32).reshape(shape).copy()
            if name in gains:return lambda:np.asarray(snapshot['gains'][gains[name]],np.float32).reshape(1,12).copy()
            raise AssertionError('Forbidden or unsupported SDK operation: '+name)
    class Prim:
        def IsValid(self):return True
        def IsActive(self):return True
        def IsLoaded(self):return True
    class Layer:
        def ExportToString(self):return 'UNCHANGED_SYNTHETIC_LAYER_NOT_A_USD_STAGE'
    class Stage:
        def GetPrimAtPath(self,path):
            assert path=='/World/patient_transport_h6_cohort_01_common_preparation_r01';return Prim()
        def GetRootLayer(self):return Layer()
        def GetSessionLayer(self):return Layer()
    class Timeline:
        playing=False;stopped=False
        def is_playing(self):return self.playing
        def is_stopped(self):return self.stopped
        def get_current_time(self):return raw['actual_final_timeline']['current_time_s']
        def __getattr__(self,name):raise AssertionError('Timeline command forbidden: '+name)
    timeline=Timeline();stage=Stage();view=View();created=[]
    class Simulation:
        def create_articulation_view(self,path):
            assert path=='/World/patient_transport_h6_cohort_01_common_preparation_r01/base_link'
            created.append(path);return view
    manager=SimpleNamespace(get_physics_sim_view=lambda:Simulation())
    sdk=SimpleNamespace(omni=SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:stage)),
        timeline=SimpleNamespace(get_timeline_interface=lambda:timeline)),SimulationManager=manager)
    return SimpleNamespace(sdk=sdk,raw=raw,snapshot=snapshot,timeline=timeline,stage=stage,view=view,created=created)


def test_retained_SDK_GET_reads_same_all12_all13_state_but_preserves_distinct_release_HOLD():
    f=_retained_GET_sdk_fixture();before=copy.deepcopy(f.snapshot)
    r=mission.retained_sdk_get_observation(f.sdk,f.raw)
    assert r['tensor_stable_during_GET'] and r['Root_Session_unchanged_during_GET'] and r['same_saved_PAUSED_timeline']
    assert r['original_motor_baseline_backend_readback_verified'] and r['observed_hold_confirmation'] is False
    assert r['metrics']['root_position_difference_m']==0.
    assert r['metrics']['maximum_joint_position_difference_rad']==0.
    assert r['metrics']['maximum_link_position_difference_m']==0.
    assert r['metrics']['current_original_stopped_speed_check'] is False
    assert r['metrics']['current_horizontal_speed_m_s']==pytest.approx(.005366004333149632)
    assert r['new_trial_count']==0 and not r['any_Play_Stop_Pause_reset_control_gain_or_scene_command']
    assert not r['actual_motion_authorized'] and f.snapshot==before and len(f.created)==1


@pytest.mark.parametrize('state',['playing','stopped','no_physics_view','bad_gain'])
def test_retained_sdk_GET_never_recovers_or_writes_to_obtain_a_snapshot(state):
    f=_retained_GET_sdk_fixture()
    if state=='playing':f.timeline.playing=True
    elif state=='stopped':f.timeline.stopped=True
    elif state=='no_physics_view':f.sdk.SimulationManager.get_physics_sim_view=lambda:None
    else:f.snapshot['gains']['kp'][6]=10000.
    with pytest.raises((ValueError,RuntimeError)):mission.retained_sdk_get_observation(f.sdk,f.raw)
    if state!='bad_gain':assert not f.created


def test_new_sdk_get_packet_runs_once_and_never_uses_duplicate_command_to_probe_again(tmp_path,monkeypatch):
    monkeypatch.setattr(mission,'SDK_GET_FOLDER',tmp_path/'SDK_GET_ONLY')
    path=mission.export_retained_sdk_get();f=_retained_GET_sdk_fixture()
    contract=mission.read(path)
    assert contract['maximum_Play_Stop_Pause_reset_static_route_or_gain_commands']==0
    with pytest.raises(ValueError):mission.run_retained_sdk_get(path,_sdk=f.sdk)
    assert not f.created
    r=mission.run_retained_sdk_get(path,read_only_authorized=True,_sdk=f.sdk)
    assert 'error' not in r and r['actual_motion_authorized'] is False and len(f.created)==1
    saved=Path(contract['result']).read_bytes()
    assert mission.run_retained_sdk_get(path,read_only_authorized=True,_sdk=f.sdk)==r
    assert Path(contract['result']).read_bytes()==saved and len(f.created)==1


def test_sdk_get_error_is_durably_reported_without_any_new_trial_or_control(tmp_path,monkeypatch):
    monkeypatch.setattr(mission,'SDK_GET_FOLDER',tmp_path/'SDK_GET_ONLY_ERROR')
    path=mission.export_retained_sdk_get();f=_retained_GET_sdk_fixture()
    f.sdk.SimulationManager.get_physics_sim_view=lambda:None
    r=mission.run_retained_sdk_get(path,read_only_authorized=True,_sdk=f.sdk)
    assert r['status']=='READ_ONLY_GET_ERROR_NO_RECOVERY' and 'error' in r
    assert not r['any_Play_Stop_Pause_reset_control_gain_or_scene_command'] and r['new_trial_count']==0
    assert not f.created


def test_current_sdk_GET_entries_contain_no_scene_control_or_physical_state_setter_calls():
    import ast
    functions=(mission.retained_sdk_get_observation,mission.run_retained_sdk_get,
        mission.zero_observation_paused_closeout_get)
    tree=ast.parse('\n'.join(inspect.getsource(f) for f in functions))
    calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    forbidden={'play','pause','stop','reset','initialize','set_gains','apply_action','set_dof_positions',
        'set_dof_velocities','set_dof_stiffnesses','set_dof_dampings','set_dof_position_targets',
        'set_dof_velocity_targets','set_world_pose','set_world_poses','Save','Reload','Load'}
    assert not calls&forbidden
    source=inspect.getsource(mission.retained_sdk_get_observation)
    assert 'authorized=False' in source and 'blocked_encoder' in source


def _zero_rows(protocol,*,speed=0.):
    raw=mission.read(protocol['original_static_record']);x,y,yaw=raw['start_xyyaw']
    return [dict(time_s=i/20.,x_m=x,y_m=y,z_m=0.,yaw_deg=np.degrees(yaw),roll_deg=0.,pitch_deg=0.,
        command_speed_m_s=0.,vx_physx_m_s=speed,vy_physx_m_s=0.,vz_physx_m_s=0.,
        pose_filtered_speed_m_s=speed,**{label+'_fz_n':265.*9.81/6 for label in mission.shared.stationary.WHEELS})
        for i in range(1,101)]


def test_completed_actual_GET_is_resolved_from_original_executing_archive_after_shared_code_evolves():
    path,r=mission.verified_retained_sdk_get()
    assert path.exists() and r['metrics']['current_original_stopped_speed_check'] is False
    assert r['metrics']['current_horizontal_speed_m_s']==pytest.approx(.005366004333059057)
    p=mission.zero_settle_definition()
    assert (p['settling_s'],p['recording_s'],p['maximum_Play_calls'])==(3.,2.,1)
    assert p['maximum_Stop_reset_restaging_route_or_gain_changes']==0
    assert p['stop_speed_limit_m_s']==.005 and p['required_PAUSED_timeline']==r['timeline']


@pytest.mark.parametrize('speed,verdict',[(0.,'PASS'),(.005,'PASS'),(.005001,'HOLD'),(.005366004333,'HOLD')])
def test_fixed_zero_tail_never_relaxes_original_speed_or_replaces_old_static(speed,verdict):
    p=mission.zero_settle_definition();rows=_zero_rows(p,speed=speed)
    r=mission.evaluate_fixed_zero_settling(rows,p,{'wheel_support':100},[])
    assert r['verdict']==verdict and r['fixed_tail_first_s']==3. and r['fixed_tail_last_s']==5.
    assert r['fixed_tail_sample_count']==41 and not r['actual_drive_admission']
    assert r['original_static_result_NOT_replaced']


@pytest.mark.parametrize('field,value',[('vx_physx_m_s',.00501),('pose_filtered_speed_m_s',.00501),('active_front_fz_n',0.)])
def test_single_bad_fixed_tail_sample_is_not_removed_by_selecting_a_better_window(field,value):
    p=mission.zero_settle_definition();rows=_zero_rows(p)
    if field.endswith('_fz_n'):
        for row in rows:row[field]=value
    else:rows[62][field]=value
    assert mission.evaluate_fixed_zero_settling(rows,p,{'wheel_support':100},[])['verdict']=='HOLD'


@pytest.mark.parametrize('mutation',['nonzero','NaN','missing_tail','extended','reversed_clock'])
def test_fixed_settling_rejects_command_clock_or_window_manipulation(mutation):
    p=mission.zero_settle_definition();rows=_zero_rows(p)
    if mutation=='nonzero':rows[0]['command_speed_m_s']=.001
    elif mutation=='NaN':rows[0]['command_speed_m_s']=float('nan')
    elif mutation=='missing_tail':rows.pop()
    elif mutation=='extended':rows[-1]['time_s']=5.3
    else:rows[2]['time_s']=rows[1]['time_s']
    with pytest.raises(ValueError):mission.evaluate_fixed_zero_settling(rows,p,{'wheel_support':100},[])


class _ZeroLifecycleSoftwarePort:
    """Protocol/cleanup fault injector; NOT Isaac or a physical plant."""
    def __init__(self,protocol,fail=None):
        self.rows=_zero_rows(protocol);self.index=-1;self.events=[];self.fail=fail
        self.counts={'wheel_support':100};self.errors=[];self.playing=False
    async def preflight(self):
        self.events.append('preflight')
        if self.fail=='preflight':raise RuntimeError('INJECTED_PREFLIGHT')
        return {'software_only':True}
    def subscribe(self):self.events.append('subscribe')
    def zero(self):
        self.events.append('zero')
        if self.playing and self.fail=='zero':raise RuntimeError('INJECTED_ZERO_SETTER')
    def play(self):self.events.append('Play');self.playing=True
    async def next_sample(self):
        self.index+=1
        if self.fail=='cancel' and self.index==2:raise asyncio.CancelledError()
    def measure(self):
        row=dict(self.rows[self.index])
        if self.fail=='gap' and self.index==2:row['time_s']+=.3
        return row
    def guard(self,row):
        if self.fail=='contact' and self.index==2:raise RuntimeError('INJECTED_CONTACT')
    async def confirm_pause(self):
        self.events.append('Pause_confirmation')
        if self.fail=='pause':raise RuntimeError('INJECTED_PAUSE_NOT_CONFIRMED')
        self.playing=False
        if self.fail=='pause_contact':self.counts['hospital_obstacle']=1
        return {'actual_software_PAUSED':True}
    def unsubscribe(self):self.events.append('unsubscribe')
    async def postflight(self):
        self.events.append('postflight')
        if self.fail=='postflight':raise RuntimeError('INJECTED_POSTFLIGHT')
        return {'checks':{'software_postflight':True}}
    def timeline_state(self):return {'playing':self.playing,'stopped':False,'current_time_s':max(0.,self.index/20.)}


def test_zero_lifecycle_requires_new_consent_before_any_port_action():
    p=mission.zero_settle_definition();port=_ZeroLifecycleSoftwarePort(p)
    with pytest.raises(ValueError):asyncio.run(mission.observe_zero_settle_once(port,p))
    assert port.events==[]


@pytest.mark.parametrize('failure',[None,'preflight','contact','zero','pause','pause_contact','postflight','cancel','gap'])
def test_zero_lifecycle_always_attempts_independent_Pause_and_never_retries_or_restores_state(failure):
    p=mission.zero_settle_definition();port=_ZeroLifecycleSoftwarePort(p,fail=failure)
    report,rows=asyncio.run(mission.observe_zero_settle_once(port,p,authorized=True))
    assert port.events.count('Play')==(0 if failure=='preflight' else 1)
    assert port.events.count('Pause_confirmation')==1
    assert port.events.index('Pause_confirmation')<port.events.index('unsubscribe')
    assert ('postflight' in port.events)==(failure not in ('pause','preflight'))
    assert report['verdict']==('PASS' if failure is None else 'HOLD')
    assert report['new_route_trials']==0 and not report['automatic_retry'] and not report['actual_drive_admission']
    if failure=='pause':assert report['actual_final_timeline']['playing']
    if failure=='preflight':
        assert 'postflight_error' not in report and 'postflight_not_attempted' in report


def test_zero_live_port_reuses_low_level_views_and_never_invokes_gain_steering_or_physical_state_setters(monkeypatch):
    """Execute the REAL port against a stateful software SDK facade ONLY."""
    import importlib
    from collections import Counter
    f=_retained_GET_sdk_fixture();p=mission.zero_settle_definition();events=[];callbacks={};clock={'s':0.}
    p['required_tensor_snapshot']=copy.deepcopy(f.snapshot)
    layer=f.stage.GetRootLayer().ExportToString()
    p['required_layer_sha256']=[mission.hashlib.sha256(layer.encode()).hexdigest()]*2
    p['input_sha256']={}
    class Timeline:
        playing=False;stopped=False;pending=False
        def is_playing(self):return self.playing
        def is_stopped(self):return self.stopped
        def get_current_time(self):return f.raw['actual_final_timeline']['current_time_s']+clock['s']
        def play(self):events.append('Play');self.playing=True
        def pause(self):events.append('Pause_requested');self.pending=True
    timeline=Timeline()
    class App:
        async def next_update_async(self):
            if timeline.pending:timeline.playing=False;timeline.pending=False;events.append('Pause_observed')
            if timeline.playing:
                clock['s']+=.05;callbacks['step'](.05)
                f.snapshot['velocity'][:3]=[.004*max(0.,1-clock['s']/3.),0.,0.]
                callbacks['contact']([SimpleNamespace(collider0=p['actor']+'/'+next(iter(port.context.stationary.WHEELS.values())),
                    collider1=port.context.stationary.SUPPORTS[0])],None)
    app=App()
    class Physics:
        def subscribe_physics_step_events(self,fn):callbacks['step']=fn;return object()
    class SimulationEvents:
        def subscribe_contact_report_events(self,fn):callbacks['contact']=fn;return object()
        def wake_up(self,*args):pass
    sim_events=SimulationEvents()
    def set_velocity(data,indices):
        assert data.shape==(1,12) and indices.tolist()==[0]
        events.append('velocity_target_ONLY');f.snapshot['joint_velocity_targets_rad_s']=data[0].tolist()
    f.view.set_dof_velocity_targets=set_velocity
    def contact_view(path):
        return SimpleNamespace(sensor_count=1,sensor_paths=[path],get_net_contact_forces=lambda dt:np.array([[0.,0.,265.*9.81/6.]]))
    simulation=SimpleNamespace(create_articulation_view=lambda path:f.view,create_rigid_contact_view=contact_view,device='cpu')
    f.sdk.SimulationManager=SimpleNamespace(get_physics_sim_view=lambda:simulation,get_backend=lambda:'numpy')
    f.sdk.omni.timeline.get_timeline_interface=lambda:timeline
    f.sdk.omni.kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda:app))
    f.sdk.omni.physx=SimpleNamespace(get_physx_interface=lambda:Physics(),get_physx_simulation_interface=lambda:sim_events)
    f.sdk.PhysicsSchemaTools=SimpleNamespace(sdfPathToInt=lambda v:v,intToSdfPath=lambda v:v)
    f.sdk.UsdUtils=SimpleNamespace(StageCache=SimpleNamespace(Get=lambda:SimpleNamespace(GetId=lambda stage:SimpleNamespace(ToLongInt=lambda:1))))
    utils=SimpleNamespace(convert=lambda data,**kwargs:data.copy())
    real_import=importlib.import_module
    monkeypatch.setattr(importlib,'import_module',lambda name:utils if name=='isaacsim.core.utils.numpy' else real_import(name))
    async def checked_environment(self,**kwargs):return {'software_only_NOT4997_live_rows':True}
    monkeypatch.setattr(mission.RetainedZeroSettleSDKPort,'environment',checked_environment)
    port=mission.RetainedZeroSettleSDKPort(f.sdk,p)
    report,rows=asyncio.run(mission.observe_zero_settle_once(port,p,authorized=True))
    assert report['verdict']=='PASS',report
    assert events.count('Play')==events.count('Pause_requested')==events.count('Pause_observed')==1
    assert events.count('velocity_target_ONLY')>100 and all(r['command_speed_m_s']==0. for r in rows)
    assert not report['actual_drive_admission'] and not timeline.playing and port.adapter.capture_count==0
    assert port.physics_sub is port.contact_sub is None
    assert f.snapshot['gains']==f.raw['actual_post_static_tensor_snapshot']['gains']


def test_zero_settle_runtime_has_no_actor_creation_gain_state_renderer_or_Stop_reset_calls():
    import ast
    tree=ast.parse(inspect.getsource(mission.RetainedZeroSettleSDKPort))
    names={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not names&{'initialize','SingleArticulation','set_gains','set_dof_stiffnesses','set_dof_dampings',
        'set_dof_positions','set_dof_velocities','set_dof_position_targets','set_world_pose','set_world_poses',
        'Set','Save','Reload','reset','stop','request_hold','restore_original'}
    assert sum(n.func.attr=='play' for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute))==1


@pytest.mark.parametrize('fault',[None,'missing','duplicate','row_digest','source145','critical_environment','layer','timeline'])
def test_zero_preflight_requires_all4997_rows_145_fields_and_exact_stable_PAUSED_boundary(monkeypatch,fault):
    f=_retained_GET_sdk_fixture();p=mission.zero_settle_definition()
    port=object.__new__(mission.RetainedZeroSettleSDKPort)
    port.protocol=p;port.stage=f.stage;port.timeline=f.timeline
    port.usd=f.sdk.omni.usd.get_context()
    port.context=mission.reviewed_static_context(mission.read(mission.REVIEWED_STATIC_FOLDER/'static_protocol.json'),
        mission.REVIEWED_STATIC_FOLDER/'static_protocol.json')
    port.raw=f.raw;port.prep=mission.read(port.context.preparation.LIVE)
    port.source=port.raw['runtime_anonymous_source_reference']['fixed_configuration_rows']
    compact=SimpleNamespace(**vars(port.context.preparation.compact));port.context.preparation.compact=compact
    rows=list(copy.deepcopy(mission.read(mission.H6/'actual_common_environment_rows.json')['compact_rows']).items())
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows.append(rows[0])
    elif fault=='row_digest':rows[0][1]['row_sha256']='0'*64
    monkeypatch.setattr(compact,'stage_rows',lambda *args:iter(rows))
    port.context.stationary=SimpleNamespace(actor_configuration=lambda stage:{} if fault=='source145' else port.source,
        critical_environment=lambda stage,prep:{} if fault=='critical_environment' else port.raw['environment_after_static'])
    state={'text':f.stage.GetSessionLayer().ExportToString()}
    monkeypatch.setattr(f.stage,'GetSessionLayer',lambda:SimpleNamespace(ExportToString=lambda:state['text']))
    async def update():
        if fault=='layer':state['text']='UNREVIEWED_CHANGED_LAYER'
        if fault=='timeline':f.timeline.playing=True
    port.app=SimpleNamespace(next_update_async=update)
    if fault:
        with pytest.raises((ValueError,RuntimeError)):asyncio.run(port.environment())
    else:
        checked=asyncio.run(port.environment())
        assert checked['row_count']==4997 and checked['sha256']==port.prep['protected_environment_sha256']
        assert checked['all145_source_settings_and_critical_environment_verified'] is True


@pytest.mark.parametrize('fault',[None,'SDK_binding','trace_write'])
def test_zero_launch_packet_durable_outcomes_and_duplicate_rejection_with_software_port_only(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(mission,'ZERO_SETTLE_FOLDER',tmp_path/'single_zero_observation')
    path=mission.export_zero_settle();p=mission.read(path);calls=[]
    def sdk():calls.append('SDK');return SimpleNamespace()
    monkeypatch.setattr(mission.runtime,'live_sdk',sdk)
    def port(sdk,protocol):
        if fault=='SDK_binding':raise RuntimeError('INJECTED_READ_ONLY_SDK_BIND_FAILURE')
        return _ZeroLifecycleSoftwarePort(protocol)
    monkeypatch.setattr(mission,'RetainedZeroSettleSDKPort',port)
    original_open=Path.open
    def open_file(self,*args,**kwargs):
        if fault=='trace_write' and self.name=='zero_settling_trace.csv' and args and args[0]=='x':
            raise OSError('INJECTED_TRACE_WRITE_FAILURE')
        return original_open(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',open_file)
    report=asyncio.run(mission.run_zero_settle_once(path,authorized=True))
    saved=path.parent/'zero_settling_readback.json';before=saved.read_bytes()
    assert mission.read(saved)==report
    assert report['verdict']==('PASS' if fault is None else 'HOLD')
    if fault=='trace_write':assert 'trace_write_error' in report and not report['checks']['durable_complete_trace']
    if fault=='SDK_binding':assert report['new_zero_command_observations']==0
    assert calls==['SDK']
    assert asyncio.run(mission.run_zero_settle_once(path,authorized=True)) is None
    assert saved.read_bytes()==before and calls==['SDK']


def test_zero_live_entry_requires_separate_authorization_before_reading_protocol_or_importing_SDK(monkeypatch):
    monkeypatch.setattr(mission.runtime,'live_sdk',lambda:pytest.fail('No SDK before separate consent'))
    with pytest.raises(ValueError):asyncio.run(mission.run_zero_settle_once('/does/not/exist'))


def test_preplay_failure_retains_full_actual_boundary_capture_without_uninitialized_postflight():
    p=mission.zero_settle_definition();port=_ZeroLifecycleSoftwarePort(p,fail='preflight')
    port.preflight_capture={'actual_layer_texts':{'Root':'SOFTWARE_ROOT','Session':'SOFTWARE_SESSION'},
        'no_live_layer_or_control_edit':True}
    report,rows=asyncio.run(mission.observe_zero_settle_once(port,p,authorized=True))
    assert rows==[] and report['actual_preflight_boundary_capture']==port.preflight_capture
    assert 'postflight' not in port.events and 'postflight_error' not in report
    assert report['new_zero_command_observations']==0 and report['verdict']=='HOLD'


def test_reviewed_zero_definition_requires_actual_unstarted_abort_and_exact_saved_two_aperture_only_changes():
    pytest.importorskip('pxr')
    p=mission.reviewed_zero_settle_definition()
    assert p['approval']==mission.REVIEWED_ZERO_APPROVAL
    assert p['original_physical_observation_NOT_started'] and p['maximum_Play_calls']==1
    assert p['maximum_Stop_reset_restaging_route_or_gain_changes']==0
    assert p['settling_s']==3. and p['recording_s']==2. and p['stop_speed_limit_m_s']==.005
    assert p['reviewed_display_boundary']['reviewed_default_paths']==list(mission.ZERO_DISPLAY_PATHS)
    assert len(p['exact_actual_display_review']['exact_Top_default_differences'])==2
    assert p['exact_actual_display_review']['full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults']


@pytest.mark.parametrize('change',[None,'apertures','Root','camera_XY','clipping','new_prim'])
def test_display_aware_zero_guard_allows_ONLY_two_apertures_without_writing_any_live_layer(change):
    pytest.importorskip('pxr')
    from pxr import Sdf,Usd,Gf
    p=mission.reviewed_zero_settle_definition()
    actual=mission.read(mission.ZERO_SETTLE_FOLDER/'preplay_layer_fields_GET.json')
    texts=copy.deepcopy(actual['actual_layer_texts']);before=copy.deepcopy(texts)
    layer=Sdf.Layer.CreateAnonymous('ONLY_A_TEST_COPY.sdf');assert layer.ImportFromString(texts['Session'])
    stage=Usd.Stage.Open(layer)
    if change=='apertures':
        for path in mission.ZERO_DISPLAY_PATHS:
            attr=stage.GetAttributeAtPath(path);assert attr.Set(float(attr.Get())+1.)
    elif change=='Root':texts['Root']+='\n# UNREVIEWED_ROOT_EDIT\n'
    elif change=='camera_XY':
        attr=stage.GetAttributeAtPath('/OmniverseKit_Top.xformOp:transform');matrix=Gf.Matrix4d(attr.Get());matrix[3]=Gf.Vec4d(99.,16.,50001.05441966978,1.)
        assert attr.Set(matrix)
    elif change=='clipping':
        attr=stage.GetPrimAtPath('/OmniverseKit_Top').CreateAttribute('clippingRange',Sdf.ValueTypeNames.Float2)
        assert attr.Set(Gf.Vec2f(.2,100.))
    elif change=='new_prim':stage.DefinePrim('/World/UNREVIEWED_NEW_OBJECT','Xform')
    if change not in (None,'Root'):texts['Session']=layer.ExportToString()
    if change in (None,'apertures'):
        review=mission.review_zero_display_layers(p,texts)
        assert review['full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults']
    else:
        with pytest.raises((ValueError,RuntimeError)):mission.review_zero_display_layers(p,texts)
    assert actual['actual_layer_texts']==before


@pytest.mark.parametrize('change',['XY','apertures','Z','rotation','affine','clipping','metadata','new_prim','Root','missing_policy','changed_scope'])
def test_prospective_Top_XY_policy_keeps_all_non_display_fields_exact_on_COPIES(change):
    from pxr import Sdf,Usd,Gf
    p=mission.reviewed_zero_settle_definition()
    original=mission.read(mission.ZERO_SETTLE_FOLDER/'preplay_layer_fields_GET.json')['actual_layer_texts']
    texts=copy.deepcopy(original)
    p['reviewed_display_boundary'].update(reviewed_default_paths=list(mission.TOP_XY_DISPLAY_PATHS),policy=mission.TOP_XY_DISPLAY_POLICY)
    layer=Sdf.Layer.CreateAnonymous('PROSPECTIVE_DISPLAY_COPY.sdf');assert layer.ImportFromString(texts['Session'])
    stage=Usd.Stage.Open(layer);attr=stage.GetAttributeAtPath('/OmniverseKit_Top.xformOp:transform')
    matrix=Gf.Matrix4d(attr.Get());values=[list(row) for row in matrix]
    if change=='XY':values[3][0]+=.4;values[3][1]-=.7
    elif change=='Z':values[3][2]+=.1
    elif change=='rotation':values[0][0]+=.1
    elif change=='affine':values[3][3]=.9
    elif change=='apertures':
        for path in mission.ZERO_DISPLAY_PATHS:
            item=stage.GetAttributeAtPath(path);assert item.Set(float(item.Get())+1.)
    elif change=='clipping':stage.GetAttributeAtPath('/OmniverseKit_Top.clippingRange').Set(Gf.Vec2f(.2,100.))
    elif change=='metadata':attr.SetDocumentation('UNREVIEWED_METADATA')
    elif change=='new_prim':stage.DefinePrim('/UNREVIEWED','Xform')
    elif change=='Root':texts['Root']+='\n# UNREVIEWED_ROOT_BYTES\n'
    elif change=='missing_policy':p['reviewed_display_boundary'].pop('policy')
    elif change=='changed_scope':p['reviewed_display_boundary']['reviewed_default_paths'].append('/UNREVIEWED.x')
    if change in ('XY','Z','rotation','affine'):assert attr.Set(Gf.Matrix4d(*values))
    texts['Session']=layer.ExportToString()
    saved=copy.deepcopy(texts)
    if change in ('XY','apertures'):
        assert mission.review_zero_display_layers(p,texts)['full_canonical_layer_including_unregistered_metadata_identical_except_reviewed_defaults']
    else:
        with pytest.raises((ValueError,RuntimeError)):mission.review_zero_display_layers(p,texts)
    assert texts==saved and mission.read(mission.ZERO_SETTLE_FOLDER/'preplay_layer_fields_GET.json')['actual_layer_texts']==original


def test_boundary_capture_logs_exact_actual_camera_fields_but_does_not_admit_play():
    pytest.importorskip('pxr')
    p=mission.zero_settle_definition();actual=mission.read(mission.ZERO_SETTLE_FOLDER/'preplay_layer_fields_GET.json')
    capture=mission.zero_preflight_boundary_capture(p,actual['actual_layer_texts'],actual['timeline'])
    assert capture['exact_GET_hash_matches']==[True,False]
    assert {r['path'] for r in capture['Session_fields_vs_exact_GET_baseline']}==set(mission.ZERO_DISPLAY_PATHS)
    assert capture['no_live_layer_or_control_edit']


@pytest.mark.parametrize('field,value,exceeded',[
    ('x_m',.030,[]),('x_m',.030001,['position_drift_m']),
    ('roll_deg',2.,[]),('pitch_deg',2.00001,['maximum_tilt_deg']),
    ('vx_physx_m_s',.080,[]),('vy_physx_m_s',.080001,['horizontal_speed_m_s'])])
def test_zero_guard_reports_exact_original_limit_without_inventing_trigger(field,value,exceeded):
    row=dict(x_m=0.,y_m=0.,roll_deg=0.,pitch_deg=0.,vx_physx_m_s=0.,vy_physx_m_s=0.)
    row[field]=value
    result=mission.zero_sample_safety_metrics(row,[0.,0.,0.])
    assert result['exceeded']==exceeded
    assert result['unchanged_limits']==dict(position_drift_m=.030,maximum_tilt_deg=2.,horizontal_speed_m_s=.080)
    row[field]=math.nan
    with pytest.raises(ValueError):mission.zero_sample_safety_metrics(row,[0.,0.,0.])


def test_rejected_zero_sample_and_exact_failure_saved_separately_without_trace_repair():
    p=mission.zero_settle_definition();port=_ZeroLifecycleSoftwarePort(p)
    def guard(row):
        if port.index==2:
            error=RuntimeError('SOFTWARE_DRIFT_GUARD')
            error.zero_safety_metrics=dict(measured={'position_drift_m':.0301},exceeded=['position_drift_m'])
            raise error
    port.guard=guard
    report,rows=asyncio.run(mission.observe_zero_settle_once(port,p,authorized=True))
    assert len(rows)==2 and report['rejected_observation_before_Pause']['time_s']==pytest.approx(.15)
    assert report['exact_zero_safety_guard_failure']['exceeded']==['position_drift_m']
    assert report['verdict']=='HOLD' and port.events.count('Play')==port.events.count('Pause_confirmation')==1
    assert report['new_zero_command_observations']==1 and not report['automatic_retry']


def test_saved_actual_zero_creep_review_keeps_incomplete_original_HOLD_and_missing_trigger_honest():
    path=mission.REVIEWED_ZERO_FOLDER/'protocol.json'
    files=[path,path.parent/'zero_settling_readback.json',path.parent/'zero_settling_trace.csv',path.parent/'executed_sources.zip']
    before=[p.read_bytes() for p in files]
    review=mission.review_completed_zero_observation(path)
    assert review['original_verdict']=='HOLD' and review['sample_count']==285
    assert review['additional_displacement_during_saved_observation_m']==pytest.approx(.02040290841434875)
    assert review['last_accepted_safety_metrics']['measured']['position_drift_m']==pytest.approx(.02996185311154641)
    assert review['last_accepted_safety_metrics']['exceeded']==[]
    assert not review['exact_trigger_sample_available'] and not review['partial_fixed_tail']['full_fixed2s_window_completed']
    assert review['partial_fixed_tail']['physx_speed_over_original5mm_s']==39
    assert review['partial_fixed_tail']['filtered_speed_over_original5mm_s']==32
    assert review['actual_Pause_confirmed'] and not review['actual_motion_authorized']
    assert review['postflight_environment_NOT_verified']
    assert [p.read_bytes() for p in files]==before


@pytest.mark.parametrize('fault',[None,'bad_gain','clock','duplicate'])
def test_zero_closeout_GET_keeps_all_deltas_and_never_repairs_state_or_repeats(tmp_path,monkeypatch,fault):
    pytest.importorskip('pxr')
    from pxr import Sdf
    import shutil
    original=mission.REVIEWED_ZERO_FOLDER
    folder=tmp_path/'consumed_zero';shutil.copytree(original,folder,ignore=shutil.ignore_patterns('paused_closeout_GET.json'))
    # Keep the actual packet bytes and source archive intact in the isolated
    # copy. Only the expected path is rebound; no historical evidence edited.
    monkeypatch.setattr(mission,'REVIEWED_ZERO_FOLDER',folder)
    protocol=mission.read(folder/'protocol.json');saved=mission.read(folder/'zero_settling_readback.json')
    f=_retained_GET_sdk_fixture()
    f.timeline.get_current_time=lambda:saved['actual_final_timeline']['current_time_s']+(1. if fault=='clock' else 0.)
    baseline=saved['actual_preflight_boundary_capture']['actual_layer_texts']
    layers={}
    for name in ('Root','Session'):
        extension='.usda' if baseline[name].startswith('#usda ') else '.sdf'
        layer=Sdf.Layer.CreateAnonymous('SOFTWARE_ONLY_'+name+extension)
        assert layer.ImportFromString(baseline[name]);layers[name]=layer
    layers['Root'].comment='UNREVIEWED_SOFTWARE_TEST_METADATA_NOT_WAIVED'
    monkeypatch.setattr(f.stage,'GetRootLayer',lambda:layers['Root'])
    monkeypatch.setattr(f.stage,'GetSessionLayer',lambda:layers['Session'])
    if fault=='bad_gain':f.snapshot['gains']['kp'][6]=10000.
    before=copy.deepcopy(f.snapshot);source_bytes=[(folder/name).read_bytes() for name in
        ('protocol.json','zero_settling_readback.json','zero_settling_trace.csv','executed_sources.zip')]
    if fault=='clock':
        with pytest.raises(ValueError,match='PAUSED clock'):
            mission.zero_observation_paused_closeout_get(folder/'protocol.json',read_only_authorized=True,_sdk=f.sdk)
        assert not f.created and not (folder/'paused_closeout_GET.json').exists()
        return
    result=mission.zero_observation_paused_closeout_get(folder/'protocol.json',read_only_authorized=True,_sdk=f.sdk)
    assert result['Root_Session_unchanged_during_GET'] and result['actual_PAUSED_timeline_unchanged'] and result['same_stage']
    assert len(result['native_layer_differences']['Root'])==1 and not result['native_layer_differences']['Session']
    assert result['every_recorded_field_difference_is_UNREVIEWED_NOT_waived']
    assert result['complete4997_environment_rows_NOT_rechecked']
    assert not result['any_Play_Stop_Pause_reset_control_gain_or_scene_command'] and not result['actual_motion_authorized']
    if fault=='bad_gain':assert result['SDK_error'] and result['actual_tensor_before_SDK_rejection']
    else:assert result['SDK_GET']['original_motor_baseline_backend_readback_verified']
    assert f.snapshot==before and len(f.created)==1
    if fault=='duplicate':
        assert mission.zero_observation_paused_closeout_get(folder/'protocol.json',read_only_authorized=True,_sdk=f.sdk)==result
        assert len(f.created)==1
    assert [(folder/name).read_bytes() for name in
        ('protocol.json','zero_settling_readback.json','zero_settling_trace.csv','executed_sources.zip')]==source_bytes


def test_zero_closeout_GET_no_permission_no_SDK_or_path_read(monkeypatch):
    monkeypatch.setattr(mission.runtime,'live_sdk',lambda:pytest.fail('No SDK call allowed'))
    with pytest.raises(ValueError,match='GET-only'):
        mission.zero_observation_paused_closeout_get('/does/not/exist')


def test_actual_zero_closeout_full_file_proves_state_writeback_but_keeps_real_drift_and_release_HOLD():
    pytest.importorskip('pxr')
    folder=mission.REVIEWED_ZERO_FOLDER
    paths=[folder/name for name in ('paused_closeout_GET.json','zero_settling_readback.json','protocol.json')]
    before=[p.read_bytes() for p in paths]
    result=mission.review_zero_paused_closeout()
    metrics=result['current_paused_safety_metrics']
    assert metrics['measured']['position_drift_m']==pytest.approx(.030065803611743572)
    assert metrics['exceeded']==['position_drift_m']
    assert metrics['measured']['horizontal_speed_m_s']==pytest.approx(.00663991272779625)
    assert not result['current_horizontal_stopped_check'] and result['original_zero_verdict']=='HOLD'
    review=result['recorded_state_writeback_review']
    assert review['changed_native_state_defaults']==56 and review['Session_text_exactly_identical']
    assert review['entire_canonical_Root_except_tensor_matched_state_defaults_identical']
    assert review['maximum_cross_representation_errors']['world_position_m']<1e-9
    assert review['all13links_pose_and_velocity_and_two_steering_states_matched']
    assert result['full4997_environment_postflight_STILL_not_rechecked']
    assert not result['actual_motion_authorized'] and result['new_GETs']==result['new_PhysX_trials']==0
    assert [p.read_bytes() for p in paths]==before


@pytest.mark.parametrize('fault',['Root_gain','Root_metadata','Root_extra_prim','pose','velocity','joint_degrees','SDK_tensor','Session'])
def test_recorded_writeback_never_exempts_arbitrary_actor_defaults_or_rebases_tensors(fault):
    pytest.importorskip('pxr')
    from pxr import Sdf,Gf
    folder=mission.REVIEWED_ZERO_FOLDER
    zero=mission.read(folder/'zero_settling_readback.json');current=mission.read(folder/'paused_closeout_GET.json')
    protocol=mission.read(folder/'protocol.json');actor=protocol['actor']
    before=copy.deepcopy(zero['actual_preflight_boundary_capture']['actual_layer_texts'])
    after=copy.deepcopy(current['actual_layer_texts']);s0=copy.deepcopy(protocol['required_tensor_snapshot'])
    s1=copy.deepcopy(current['SDK_GET']['actual_tensor_snapshot']);unchanged=copy.deepcopy((before,after,s0,s1))
    layer=Sdf.Layer.CreateAnonymous('OFFLINE_REJECTION_TEST.usda');assert layer.ImportFromString(after['Root'])
    if fault=='Root_gain':
        prim=layer.GetPrimAtPath(actor+'/joints/base_link_to_front_active_steering')
        Sdf.AttributeSpec(prim,'drive:angular:physics:stiffness',Sdf.ValueTypeNames.Float).default=999.
    elif fault=='Root_metadata':layer.comment='UNREVIEWED_METADATA_NOT_STATE'
    elif fault=='Root_extra_prim':Sdf.CreatePrimInLayer(layer,actor+'/UNREVIEWED_NEW_BODY')
    elif fault=='pose':layer.GetAttributeAtPath(actor+'/base_link.xformOp:translate').default=Gf.Vec3d(999.,0.,0.)
    elif fault=='velocity':layer.GetAttributeAtPath(actor+'/base_link.physics:velocity').default=Gf.Vec3f(0.,0.,0.)
    elif fault=='joint_degrees':
        layer.GetAttributeAtPath(actor+'/joints/base_link_to_front_active_steering.state:angular:physics:position').default=s1['positions'][0]
    elif fault=='SDK_tensor':s1['all_thirteen_link_world_transforms_xyzw'][5][0]+=.010
    else:after['Session']+='\n# UNREVIEWED_SESSION_EDIT\n'
    if fault not in ('SDK_tensor','Session'):after['Root']=layer.ExportToString()
    with pytest.raises(ValueError):mission.recorded_physics_writeback_review(before,after,s0,s1,actor)
    assert before==unchanged[0]
    assert (zero['actual_preflight_boundary_capture']['actual_layer_texts'],current['actual_layer_texts'],
        protocol['required_tensor_snapshot'],current['SDK_GET']['actual_tensor_snapshot'])==unchanged


@pytest.mark.parametrize('fault',[None,'USD_angular_velocity','SDK_angular_velocity'])
def test_real_held_neutral_HOLD_exact_vector3f_degree_roundtrip_is_not_a_physical_tolerance_waiver(fault):
    from pxr import Sdf,Gf
    folder=mission.retained.GATE if hasattr(mission,'retained') else _ijcas_path(mission.ROOT, 'studies/IJCAS/results/hospital_paper_gate')
    p=folder/'hospital_h6_experiments/cohort_15_held_static_r01/route/P_dynamics_full_envelope'
    prior=mission.read(p/'route_readback.json');raw=mission.read(p/'held_neutral_response_probe_r01/probe_readback.json')
    before=copy.deepcopy(prior['actual_final_readback']['layer_texts']);after=copy.deepcopy(raw['actual_final_readback']['layer_texts'])
    a=copy.deepcopy(prior['actual_final_readback']['tensor']);b=copy.deepcopy(raw['actual_final_readback']['tensor'])
    actor='/World/patient_transport_h6_cohort_15_held_static_r01'
    if fault=='USD_angular_velocity':
        layer=Sdf.Layer.CreateAnonymous('ROUNDTRIP_REJECT_COPY.usda');assert layer.ImportFromString(after['Root'])
        attr=layer.GetAttributeAtPath(actor+'/rear_active_drive_wheel_link.physics:angularVelocity')
        v=list(attr.default);v[1]+=.01;attr.default=Gf.Vec3f(*v);after['Root']=layer.ExportToString()
    elif fault=='SDK_angular_velocity':b['all_thirteen_link_velocities_at_CoM'][b['link_names'].index('rear_active_drive_wheel_link')][4]+=.001
    if fault:
        with pytest.raises(ValueError):mission.recorded_physics_writeback_review(before,after,a,b,actor)
    else:
        proof=mission.recorded_physics_writeback_review(before,after,a,b,actor)
        assert proof['exact_typed_float32_degree_serialization_proof']['maximum_native_roundtrip_error_rad_s']>1e-7
        assert proof['SDK_float_readback_tolerances_NOT_physical_acceptance_relaxation']['angular_velocity_rad_s']==1e-7
        assert raw['verdict']=='HOLD' and raw['checks']['approved_bounded_observation_completed'] is False
        assert not proof['actual_motion_authorized']


@pytest.mark.parametrize('fault',[None,'USD_angular_velocity','SDK_angular_velocity'])
def test_real_H3_cusp_HOLD_proves_exact_native_float32_radian_to_degree_multiply(fault):
    from pxr import Sdf,Gf
    folder=_ijcas_path(mission.ROOT, 'studies/IJCAS/results/hospital_paper_gate/hospital_h3_experiments/cohort_06_held_static_r01/route/P_dynamics_full_envelope')
    raw=mission.read(folder/'route_readback.json');protocol=mission.read(raw['protocol'])
    static=mission.read(protocol['original_static_record'])
    before=copy.deepcopy(static['actual_final_readback']['layer_texts']);after=copy.deepcopy(raw['actual_final_readback']['layer_texts'])
    a=copy.deepcopy(protocol['required_tensor_snapshot']);b=copy.deepcopy(raw['actual_final_readback']['tensor'])
    actor=protocol['actor'];name='front_active_drive_wheel_link'
    if fault=='USD_angular_velocity':
        layer=Sdf.Layer.CreateAnonymous('H3_NATIVE_MULTIPLY_REJECT_COPY.usda');assert layer.ImportFromString(after['Root'])
        attr=layer.GetAttributeAtPath(actor+'/'+name+'.physics:angularVelocity');v=list(attr.default)
        v[0]+=.01;attr.default=Gf.Vec3f(*v);after['Root']=layer.ExportToString()
    elif fault=='SDK_angular_velocity':b['all_thirteen_link_velocities_at_CoM'][b['link_names'].index(name)][3]+=.001
    if fault:
        with pytest.raises(ValueError):mission.recorded_physics_writeback_review(before,after,a,b,actor)
    else:
        proof=mission.recorded_physics_writeback_review(before,after,a,b,actor)
        assert proof['exact_typed_float32_degree_serialization_proof']['exact_float32_multiply_vectors']>0
        assert proof['SDK_float_readback_tolerances_NOT_physical_acceptance_relaxation']['angular_velocity_rad_s']==1e-7
        assert raw['verdict']=='HOLD' and not proof['actual_motion_authorized']


@pytest.mark.parametrize('fault',[None,'Root_gain','tensor_edit','Session_clip'])
def test_real_post_observation_layer_guard_uses_tensor_matched_state_not_frozen_preplay_root(fault):
    pytest.importorskip('pxr')
    from pxr import Sdf,Gf
    folder=mission.REVIEWED_ZERO_FOLDER
    zero=mission.read(folder/'zero_settling_readback.json');current=mission.read(folder/'paused_closeout_GET.json')
    protocol=mission.read(folder/'protocol.json');f=_retained_GET_sdk_fixture()
    snapshot=copy.deepcopy(current['SDK_GET']['actual_tensor_snapshot'])
    f.snapshot.clear();f.snapshot.update(copy.deepcopy(snapshot))
    layers={}
    for name,text in current['actual_layer_texts'].items():
        layer=Sdf.Layer.CreateAnonymous('POSTFLIGHT_SOFTWARE_COPY'+('.usda' if text.startswith('#usda') else '.sdf'))
        assert layer.ImportFromString(text);layers[name]=layer
    port=object.__new__(mission.RetainedZeroSettleSDKPort)
    port.protocol=protocol;port.tensor=f.view;port.preflight_capture=zero['actual_preflight_boundary_capture']
    port.post_observation_snapshot=snapshot;port.post_observation_boundary_audits=[]
    port.stage=SimpleNamespace(GetRootLayer=lambda:layers['Root'],GetSessionLayer=lambda:layers['Session'])
    texts_before={name:layer.ExportToString() for name,layer in layers.items()}
    port.check_post_observation_layers()
    assert len(port.post_observation_boundary_audits)==1
    assert port.post_observation_boundary_audits[0]['state_writeback']['changed_native_state_defaults']==56
    port.check_post_observation_layers();assert len(port.post_observation_boundary_audits)==1
    if fault=='Root_gain':
        prim=layers['Root'].GetPrimAtPath(protocol['actor']+'/joints/base_link_to_front_active_steering')
        Sdf.AttributeSpec(prim,'drive:angular:physics:stiffness',Sdf.ValueTypeNames.Float).default=999.
    elif fault=='tensor_edit':f.snapshot['positions'][0]+=.001
    elif fault=='Session_clip':
        prim=layers['Session'].GetPrimAtPath('/OmniverseKit_Top')
        prim.attributes['clippingRange'].default=Gf.Vec2f(.2,50.)
    if fault:
        with pytest.raises(ValueError):port.check_post_observation_layers()
    else:assert {name:layer.ExportToString() for name,layer in layers.items()}==texts_before


def _first_source_writeback_fixture(*,target_layer='Root',renderer_layer=None):
    """Anonymous USD reference + SYNTHETIC state arrays, never a live stage."""
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,Sdf,Gf
    snapshot=copy.deepcopy(mission.read(mission.REVIEWED_ZERO_FOLDER/'protocol.json')['required_tensor_snapshot'])
    actor='/World/SOFTWARE_ONLY_new_source';asset=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(asset,'/Robot');snapshot['velocity']=[.001,0.,0.,0.,0.,0.]
    for i,name in enumerate(snapshot['link_names']):
        pose=snapshot['all_thirteen_link_world_transforms_xyzw'][i]
        prim=UsdGeom.Xform.Define(asset,'/Robot/'+name);prim.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        q=np.asarray(pose[3:],float);q/=np.linalg.norm(q);pose[3:]=q.tolist()
        prim.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[3],Gf.Vec3d(*q[:3])))
        for field in ('physics:velocity','physics:angularVelocity'):
            prim.GetPrim().CreateAttribute(field,Sdf.ValueTypeNames.Vector3f,custom=False).Set(Gf.Vec3f(0.))
        pose[0]+=.002;snapshot['all_thirteen_link_velocities_at_CoM'][i]=[.001,0.,0.,0.,0.,0.]
    steering=('base_link_to_front_active_steering','base_link_to_rear_active_steering')
    for name in steering:
        prim=UsdGeom.Xform.Define(asset,'/Robot/joints/'+name).GetPrim();index=snapshot['names'].index(name)
        for field in ('position','velocity'):prim.CreateAttribute('state:angular:physics:'+field,Sdf.ValueTypeNames.Float,custom=False).Set(0.)
        snapshot['positions'][index]=.001;snapshot['joint_velocities_rad_s'][index]=.002
    root=Sdf.Layer.CreateAnonymous('FRESH_SOFTWARE_ROOT.usda');session=Sdf.Layer.CreateAnonymous('FRESH_SOFTWARE_SESSION.usda')
    stage=Usd.Stage.Open(root,session);UsdGeom.Xform.Define(stage,'/World')
    with Usd.EditContext(stage,session):
        stage.DefinePrim(actor,'Xform').GetReferences().AddReference(asset.GetRootLayer().identifier,'/Robot')
        camera=UsdGeom.Camera.Define(stage,'/OmniverseKit_Top')
        camera.GetHorizontalApertureAttr().Set(100.);camera.GetVerticalApertureAttr().Set(60.)
    source_types=mission.prepared_source_state_types(stage,actor,snapshot['link_names'],snapshot['names'])
    if renderer_layer is not None:
        from pxr import UsdRender
        with Usd.EditContext(stage,root if renderer_layer=='Root' else session):
            product=UsdRender.Product.Define(stage,'/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0').GetPrim()
            product.CreateAttribute('omni:rtx:rt:ecoMode:enabled',Sdf.ValueTypeNames.Bool,custom=False).Set(True)
    before=dict(Root=root.ExportToString(),Session=session.ExportToString())
    with Usd.EditContext(stage,root if target_layer=='Root' else session):
        for i,name in enumerate(snapshot['link_names']):
            prim=stage.GetPrimAtPath(actor+'/'+name);pose=snapshot['all_thirteen_link_world_transforms_xyzw'][i]
            prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*pose[:3]))
            prim.GetAttribute('xformOp:orient').Set(Gf.Quatd(pose[6],Gf.Vec3d(*pose[3:6])))
            prim.GetAttribute('physics:velocity').Set(Gf.Vec3f(.001,0.,0.))
            prim.GetAttribute('physics:angularVelocity').Set(Gf.Vec3f(0.))
        for name in steering:
            prim=stage.GetPrimAtPath(actor+'/joints/'+name);index=snapshot['names'].index(name)
            for field,value in (('position',snapshot['positions'][index]),('velocity',snapshot['joint_velocities_rad_s'][index])):
                prim.GetAttribute('state:angular:physics:'+field).Set(math.degrees(value))
    snapshot['transform']=copy.deepcopy(snapshot['all_thirteen_link_world_transforms_xyzw'][snapshot['link_names'].index('base_link')])
    def texts():return dict(Root=root.ExportToString(),Session=session.ExportToString())
    return SimpleNamespace(asset=asset,stage=stage,actor=actor,snapshot=snapshot,types=source_types,before=before,texts=texts)


@pytest.mark.parametrize('renderer_layer',['Root','Session'])
def test_renderer_existing_ROOT_or_SESSION_Boolean_default_split_keeps_entire_first_physics_proof(renderer_layer):
    from pxr import Usd
    f=_first_source_writeback_fixture(renderer_layer=renderer_layer)
    path='/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0.omni:rtx:rt:ecoMode:enabled'
    with Usd.EditContext(f.stage,f.stage.GetRootLayer() if renderer_layer=='Root' else f.stage.GetSessionLayer()):
        f.stage.GetAttributeAtPath(path).Set(False)
    current=f.texts();result=mission.prepared_source_writeback_review(f.stage,f.before,current,f.snapshot,f.actor,f.types)
    rows=result['exact_renderer_default_differences']
    assert len(rows)==1 and rows[0]['layer']==renderer_layer and rows[0]['field']=='default'
    assert rows[0]['before'] is True and rows[0]['after'] is False
    assert result['canonical_nonstate_Root_and_Session_retained'] and result['no_live_layer_edit']
    assert f.texts()==current


@pytest.mark.parametrize('fault',['gain','metadata','new_ECO_property','other_Render','camera_Z','opaque_Root'])
def test_renderer_Boolean_split_cannot_hide_physical_or_unreviewed_Render_change(fault):
    from pxr import Usd,Sdf,UsdGeom
    f=_first_source_writeback_fixture(renderer_layer='Root')
    path='/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0.omni:rtx:rt:ecoMode:enabled'
    attr=f.stage.GetAttributeAtPath(path);attr.Set(False)
    if fault=='gain':f.stage.GetPrimAtPath(f.actor+'/base_link').CreateAttribute('drive:angular:physics:stiffness',Sdf.ValueTypeNames.Float).Set(999.)
    elif fault=='metadata':attr.SetDocumentation('UNREVIEWED')
    elif fault=='new_ECO_property':
        from pxr import UsdRender
        prim=UsdRender.Product.Define(f.stage,'/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_99').GetPrim()
        prim.CreateAttribute('omni:rtx:rt:ecoMode:enabled',Sdf.ValueTypeNames.Bool).Set(False)
    elif fault=='other_Render':UsdGeom.Xform.Define(f.stage,'/Render/UNREVIEWED')
    elif fault=='camera_Z':
        with Usd.EditContext(f.stage,f.stage.GetSessionLayer()):
            UsdGeom.Xformable(f.stage.GetPrimAtPath('/OmniverseKit_Top')).AddTranslateOp().Set((0.,0.,8.))
    else:f.stage.GetRootLayer().comment='UNREVIEWED_OPAQUE_CHANGE'
    with pytest.raises(ValueError):mission.prepared_source_writeback_review(f.stage,f.before,f.texts(),f.snapshot,f.actor,f.types)


@pytest.mark.parametrize('layer',['Root','Session'])
def test_FIRST_source_writeback_all56_new_overrides_and_minimal_ancestors_not_a_whole_Root_hash_invariant(layer):
    f=_first_source_writeback_fixture(target_layer=layer);unchanged=copy.deepcopy((f.before,f.snapshot))
    if layer=='Root':assert f.actor not in f.before['Root']
    current=f.texts();result=mission.prepared_source_writeback_review(f.stage,f.before,current,f.snapshot,f.actor,f.types)
    assert result['canonical_nonstate_Root_and_Session_retained'] and result['no_live_layer_edit']
    assert result['prePlay_source_NOT_fabricated_as_a_measured_SDK_snapshot']
    assert result['maximum_cross_representation_errors']['world_position_m']<1e-10
    assert f.texts()==current and (f.before,f.snapshot)==unchanged


@pytest.mark.parametrize('fault',['gain','new_prim','ancestor_type','opaque_Root','opaque_Session','state_metadata','SDK_pose','SDK_velocity','SDK_joint','camera_Z','bad_type'])
def test_FIRST_source_writeback_rejects_configuration_metadata_pose_velocity_joint_and_camera_changes(fault):
    pytest.importorskip('pxr')
    from pxr import Sdf,Usd,UsdGeom,Gf
    f=_first_source_writeback_fixture();root=f.stage.GetRootLayer();base=f.stage.GetPrimAtPath(f.actor+'/base_link')
    if fault=='gain':base.CreateAttribute('drive:angular:physics:stiffness',Sdf.ValueTypeNames.Float).Set(999.)
    elif fault=='new_prim':UsdGeom.Xform.Define(f.stage,f.actor+'/UNREVIEWED')
    elif fault=='ancestor_type':root.GetPrimAtPath(f.actor).typeName='Xform'
    elif fault=='opaque_Root':root.comment='UNREVIEWED_OPAQUE_CHANGE'
    elif fault=='opaque_Session':f.stage.GetSessionLayer().comment='UNREVIEWED_OPAQUE_CHANGE'
    elif fault=='state_metadata':base.GetAttribute('physics:velocity').SetDocumentation('NOT_A_STATE_VALUE')
    elif fault=='SDK_pose':f.snapshot['all_thirteen_link_world_transforms_xyzw'][2][0]+=.001
    elif fault=='SDK_velocity':f.snapshot['all_thirteen_link_velocities_at_CoM'][3][0]+=.001
    elif fault=='SDK_joint':f.snapshot['positions'][f.snapshot['names'].index('base_link_to_front_active_steering')]+=.001
    elif fault=='camera_Z':
        with Usd.EditContext(f.stage,f.stage.GetSessionLayer()):
            UsdGeom.Xformable(f.stage.GetPrimAtPath('/OmniverseKit_Top')).AddTranslateOp().Set(Gf.Vec3d(0.,0.,8.))
    else:root.GetAttributeAtPath(f.actor+'/base_link.physics:velocity').SetInfo('typeName','double3')
    with pytest.raises(ValueError):mission.prepared_source_writeback_review(f.stage,f.before,f.texts(),f.snapshot,f.actor,f.types)


def test_FIRST_source_writeback_records_only_existing_positive_aperture_defaults_without_live_edit():
    pytest.importorskip('pxr')
    from pxr import UsdGeom,Usd
    f=_first_source_writeback_fixture()
    with Usd.EditContext(f.stage,f.stage.GetSessionLayer()):
        UsdGeom.Camera(f.stage.GetPrimAtPath('/OmniverseKit_Top')).GetHorizontalApertureAttr().Set(101.)
    result=mission.prepared_source_writeback_review(f.stage,f.before,f.texts(),f.snapshot,f.actor,f.types)
    assert len(result['reviewed_aperture_defaults'])==1
