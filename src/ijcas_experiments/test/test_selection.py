
from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ijcas_experiments import selection, workflow
from ijcas_experiments import sdk_runner


@pytest.mark.parametrize('scenario', ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'])
@pytest.mark.parametrize('method', ['B0', 'B1', 'B2', 'P'])
def test_all_main_selections_exactly_match_registered_conditions(scenario, method):
    for condition in (1, 2, 3):
        spec = selection.resolve(scenario=scenario, method=method, condition=condition)
        assert spec['cell']['cell_id'] == f'{scenario}_{selection.METHODS[method]}_f{condition:02d}'
        assert not spec['missing_inputs']
        assert spec['lookahead_edges'] == 4
        assert spec['common_speed_policy']['cruise_m_s'] == .03
        assert spec['stage'] == 'check' and spec['allow_motion'] is False
        assert spec['physical_execution_verified'] is False
        assert spec['task']['wall_budget_s'] == (1500. if scenario == 'H6' else 900.)


@pytest.mark.parametrize('scenario', ['H2', 'H3'])
def test_history_is_actual_steady_entry_variant_not_zero_reset(scenario):
    spec = selection.resolve(scenario=scenario, method='P_no_history', condition=2)
    assert spec['cell']['predictive_history_policy'] == 'steady_both'
    assert spec['cell']['experiment_role'] == 'HISTORY_ABLATION'
    assert spec['cell']['cell_id'].endswith('_steady_both')


@pytest.mark.parametrize('options', [
    dict(environment='imaginary'), dict(scenario='H7'), dict(method='P_best'),
    dict(condition=0), dict(condition=4), dict(condition=True), dict(condition=1.0),
    dict(method='P_no_history', scenario='H1'), dict(feedback='auto'),
    dict(stage='run'), dict(stage='run', allow_motion='yes'), dict(headless='no'),
    dict(feedback='amcl_encoder', scenario='H2'),
    dict(feedback='amcl_encoder', scenario='H5', method='B2'),
    dict(feedback='amcl_encoder', scenario='H5', condition=2),
    dict(unknown='setting'),
])
def test_unsupported_or_unsafe_combinations_fail_closed(options):
    with pytest.raises(ValueError):
        selection.resolve(**options)


def test_yaml_precedence_and_unknown_keys(tmp_path):
    path = tmp_path / 'settings.yaml'
    path.write_text(yaml.safe_dump(dict(scenario='H3', method='B2', condition=2, headless=True)))
    spec = selection.resolve(path, scenario='H6', condition='3')
    assert (spec['scenario'], spec['method'], spec['condition'], spec['headless']) == ('H6', 'B2', 3, True)
    path.write_text('cruise_speed: 0.001\n')
    with pytest.raises(ValueError, match='Unknown'):
        selection.resolve(path)


def test_check_has_no_backend_or_writes(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('Check must not plan, reserve outputs or start a simulator')
    monkeypatch.setattr(workflow, 'reserve', forbidden)
    monkeypatch.setattr(workflow, 'plan_selected', forbidden)
    monkeypatch.setattr(workflow.subprocess, 'run', forbidden)
    result = workflow.execute(selection.resolve())
    assert result['ready'] and result['new_trials'] == 0
    assert not result['simulator_started']


def test_same_condition_start_and_joints_are_method_blind():
    specs = [selection.resolve(scenario='H6', method=m, condition=2) for m in ('B0', 'B1', 'B2', 'P')]
    for spec in specs[1:]:
        for key in ('start_xyyaw', 'goal_xyyaw', 'initial_joint_angles_deg'):
            assert spec['cell'][key] == specs[0]['cell'][key]


def test_archive_analysis_selects_one_without_new_sd_or_denominator(tmp_path, monkeypatch):
    spec = selection.resolve(scenario='H2', method='P', condition=3)
    row = dict(cell_id=spec['cell']['cell_id'], planning_time_s=12., completed=True)
    from ijcas_experiments import reproduce
    monkeypatch.setattr(reproduce, 'main_results', lambda: dict(rows=[row, dict(row, cell_id='other')]))
    result = workflow.analyze_selected(spec, tmp_path)
    assert result['result'] == row and result['sample_SD'] is None and result['new_trials'] == 0
    assert json.loads((tmp_path / 'metrics.json').read_text()) == row


def test_output_is_exclusive_and_never_in_immutable_evidence(tmp_path):
    spec = selection.resolve(output=str(tmp_path / 'new'))
    out = workflow.reserve(spec)
    assert (out / 'selection.json').is_file()
    with pytest.raises(FileExistsError):
        workflow.reserve(spec)
    from ijcas_paths import ROOT
    with pytest.raises(ValueError):
        workflow.reserve(dict(spec, output=str(_ijcas_path(ROOT, 'studies/IJCAS/results/new'))))


def test_fresh_protocol_does_not_grant_motion_or_relabel_paper_results(tmp_path, monkeypatch):
    spec = selection.resolve(scenario='H1', method='B0')
    monkeypatch.setattr(workflow, 'source_pins', lambda: {})
    value = workflow.prepare_protocol(spec, tmp_path)
    assert value['main_cells'] == [spec['cell']]
    assert value['current_motion_authorized'] is False
    assert value['status'] == 'NEW_SELECTION_NOT_A_PUBLISHED_RESULT'
    assert value['old_results_never_replaced'] is True


def test_sdk_authority_is_current_and_never_an_old_approval(tmp_path):
    spec = selection.resolve(stage='run', allow_motion=True)
    authority = sdk_runner.current_authority(spec, tmp_path)
    assert authority['approval_source'] == 'CURRENT_EXPLICIT_CLI_SELECTION'
    with pytest.raises(ValueError):
        sdk_runner.current_authority(dict(spec, allow_motion=False), tmp_path)
    sensor = selection.resolve(scenario='H5', method='P', feedback='amcl_encoder', stage='run', allow_motion=True)
    assert sdk_runner.current_authority(sensor, tmp_path)['approval_source'] == 'CURRENT_EXPLICIT_CLI_SELECTION'
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(
        current_motion_authorized=False, main_cells=[spec['cell']])))
    with pytest.raises(ValueError, match='frozen'):
        sdk_runner.current_authority(spec, tmp_path)


def test_sdk_python_must_exist(tmp_path):
    spec = selection.resolve(stage='run', allow_motion=True, sdk_python=str(tmp_path / 'missing'))
    with pytest.raises(ValueError):
        workflow.sdk_command(spec, tmp_path)


@pytest.mark.parametrize('headless', [True, False])
def test_sdk_uses_original_full_app_clock_not_python_default(headless):
    assert sdk_runner.app_configuration(dict(headless=headless)) == dict(
        headless=headless, extra_args=['--/app/player/useFixedTimeStepping=true'])


def test_cli_defaults_and_bad_argument_exit(capsys):
    assert workflow.main(['--list']) == 0
    assert 'environments' in capsys.readouterr().out
    assert workflow.main(['--scenario', 'H2', '--method', 'B2']) == 0
    assert workflow.main(['--condition', '4']) == 2


def test_source_binding_uses_new_output_and_current_authority(tmp_path, monkeypatch):
    # Pure host construction test: no pxr, SimulationApp, Play or SDK access.
    import hospital_revision_final_campaign as final
    def seed(path, angles):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test-only source placeholder, never opened by a simulator')
        return dict(asset=str(path), sha256=workflow.sha(path), initial_angles_deg=angles)
    monkeypatch.setattr(final.initial, 'seed_asset', seed)
    spec = selection.resolve(stage='run', allow_motion=True)
    workflow.prepare_protocol(spec, tmp_path)
    call, authority = sdk_runner.build_live_binding(spec, tmp_path)
    assert call.__globals__['campaign'].OUTPUT == tmp_path
    runtime = call.__globals__['runtime']
    assert runtime.APPROVAL == authority
    assert runtime.checked_approval(authority) == authority
    with pytest.raises(ValueError):
        runtime.checked_approval(final.runtime.APPROVAL)
    assert list((tmp_path / 'source_conditions').glob('*.usda'))


def test_error_does_not_publish_completion_or_retry(tmp_path, monkeypatch):
    @workflow.contextmanager
    def unlocked():
        yield
    monkeypatch.setattr(workflow, 'single_worker', unlocked)
    calls = []
    def broken(spec, output):
        calls.append(1)
        raise RuntimeError('test failure')
    monkeypatch.setattr(workflow, 'native_plan', broken)
    monkeypatch.setattr(workflow, 'sdk_python', lambda spec: '/test/sdk/python')
    target = tmp_path / 'failed'
    spec = selection.resolve(stage='plan', output=str(target))
    with pytest.raises(RuntimeError):
        workflow.execute(spec)
    assert calls == [1]
    assert (target / 'error.json').is_file() and not (target / 'completed.json').exists()


def test_run_handoff_is_once_in_an_isolated_python_environment(tmp_path, monkeypatch):
    @workflow.contextmanager
    def unlocked():
        yield
    monkeypatch.setattr(workflow, 'single_worker', unlocked)
    monkeypatch.setattr(workflow, 'sdk_python', lambda spec: '/test/sdk/python')
    monkeypatch.setattr(workflow, 'native_plan', lambda spec, out: dict(independently_valid_reference=True))
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        assert 'PYTHONPATH' not in kwargs['env']
        assert kwargs['check'] is False
        out = Path(command[-1])
        workflow.write(out / 'execution_result.json', dict(status='TEST_ONLY_NOT_PHYSICAL', simulator_started=False))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess, 'run', child)
    spec = selection.resolve(stage='run', allow_motion=True, output=str(tmp_path / 'one'))
    result = workflow.execute(spec)
    assert len(calls) == 1
    assert result['result']['status'] == 'TEST_ONLY_NOT_PHYSICAL'
    assert (tmp_path / 'one/completed.json').is_file()


@pytest.mark.parametrize('started,verdict', [(False, 'HOLD'), (True, 'HOLD'), (True, 'PASS')])
def test_missing_sensor_metrics_never_relabel_static_hold_or_invent_success(tmp_path, monkeypatch, started, verdict):
    from ijcas_experiments import reproduce
    @workflow.contextmanager
    def unlocked():
        yield
    monkeypatch.setattr(workflow, 'single_worker', unlocked)
    monkeypatch.setattr(workflow, 'sdk_python', lambda spec: '/test/sdk/python')
    monkeypatch.setattr(workflow, 'native_plan', lambda spec, out: dict(independently_valid_reference=True))
    calls = []
    def missing(*args, **kwargs):
        calls.append(1)
        raise ValueError('Incomplete test-only paired observations')
    monkeypatch.setattr(reproduce, 'raw_sensor_results', missing)
    def child(command, **kwargs):
        out = Path(command[-1])
        workflow.write(out / 'execution_result.json', dict(actual_verdict=verdict,
            actual_route_started=started, report=str(out / 'route/route_readback.json')))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess, 'run', child)
    target = tmp_path / 'sensor'
    spec = selection.resolve(scenario='H5', feedback='amcl_encoder',
        stage='run', allow_motion=True, output=str(target))
    if verdict == 'PASS':
        with pytest.raises(ValueError, match='Incomplete test-only'):
            workflow.execute(spec)
        assert not (target / 'completed.json').exists()
    else:
        result = workflow.execute(spec)
        assert result['result']['actual_verdict'] == 'HOLD'
        assert calls == ([1] if started else [])
        assert (target / 'sensor_metrics_error.json').exists() == started


@pytest.mark.parametrize('task,wall', [('H5', 900.), ('H6', 3000.)])
def test_sensor_selection_uses_separate_nominal_case_not_main_draw(task, wall):
    sensor = selection.resolve(scenario=task, method='P', feedback='amcl_encoder')
    main = selection.resolve(scenario=task, method='P')
    assert sensor['cell']['cell_id'] != main['cell']['cell_id']
    assert sensor['cell']['start_xyyaw'] == sensor['sensor_case']['start_xyyaw']
    assert sensor['task']['wall_budget_s'] == wall
    assert sensor['recorded_sensor_case_is_not_a_main_repetition'] is True
    assert sensor['lookahead_edges'] is None and sensor['original_P_seed_portfolio'] is None
    assert sensor['common_speed_policy']['version'] == 'RETAINED_SENSOR_PATH_NOT_MAIN_COMMON_SPEED_PROTOCOL'
    assert not sensor['missing_inputs']


def test_sensor_bridge_command_does_not_mix_sdk_and_ros(monkeypatch, tmp_path):
    from ijcas_experiments.sensor_runner import bridge_command
    monkeypatch.setenv('PYTHONPATH', '/sdk/py311')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/sdk/bin')
    command, env = bridge_command(tmp_path / 'configuration.json')
    assert 'PYTHONPATH' not in env and 'LD_LIBRARY_PATH' not in env
    assert command[-1] == str(tmp_path / 'configuration.json')
    assert 'install/setup.bash' in command[4] and '/usr/bin/python3' in command[4]


def test_child_zero_exit_with_error_receipt_never_publishes_completion(tmp_path, monkeypatch):
    @workflow.contextmanager
    def unlocked():
        yield
    monkeypatch.setattr(workflow, 'single_worker', unlocked)
    monkeypatch.setattr(workflow, 'sdk_python', lambda spec: '/test/sdk/python')
    monkeypatch.setattr(workflow, 'native_plan', lambda spec, out: dict(independently_valid_reference=True))
    def child(command, **kwargs):
        workflow.write(Path(command[-1]) / 'sdk_error.json', dict(error='test-only startup error'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess, 'run', child)
    target = tmp_path / 'failed-child'
    with pytest.raises(RuntimeError, match='test-only startup error'):
        workflow.execute(selection.resolve(stage='run', allow_motion=True, output=str(target)))
    assert not (target / 'completed.json').exists()


@pytest.mark.parametrize('verdict,expected', [('PASS', 0), ('HOLD', 3), ('PLAN_FAILURE', 3)])
def test_run_exit_code_reports_noncompletion(monkeypatch, verdict, expected):
    monkeypatch.setattr(workflow, 'execute', lambda spec: dict(result=dict(actual_verdict=verdict)))
    assert workflow.main(['--stage', 'run', '--allow-motion', 'true']) == expected


@pytest.mark.parametrize('task', ['H5', 'H6'])
def test_sensor_runner_restores_full_paper_prior_instead_of_broad_probe_default(task):
    from ijcas_experiments.sensor_runner import restore_paper_bridge_settings
    from ijcas_paths import load_archived_json
    from hospital_sensor_ros_bridge import initial_prior_covariance
    spec = selection.resolve(scenario=task, feedback='amcl_encoder')
    archived = load_archived_json(spec['sensor_case']['bridge_configuration'])
    generated = deepcopy(archived)
    for key in ('known_fixed_start_local_prior', 'not_unknown_pose_global_relocalization_case',
                'native_source_surface_consistency_policy'):
        generated.pop(key, None)
    assert initial_prior_covariance(generated)[0] == .15**2
    restored, parity = restore_paper_bridge_settings(generated, archived)
    assert initial_prior_covariance(restored)[0] == .003**2
    assert restored == archived
    assert parity['all_nonidentity_top_level_settings_match_paper']
    bad = deepcopy(generated)
    bad['initial_prior_xyyaw'][0] += .01
    with pytest.raises(ValueError, match='Unreviewed sensor setting'):
        restore_paper_bridge_settings(bad, archived)
    bad = deepcopy(generated)
    bad['mission_platform_definition']['maximum_control_gap_s'] = .5
    with pytest.raises(ValueError, match='limits/policy'):
        restore_paper_bridge_settings(bad, archived)


@pytest.mark.parametrize('task', ['H5', 'H6'])
def test_scan_configuration_comes_from_actual_paper_frame_not_old_paused_probe(task):
    import gzip
    from ijcas_experiments.sensor_runner import paper_scan_settings
    from hospital_sensor_messages import FIRST_RETURN_MODEL
    spec = selection.resolve(scenario=task, feedback='amcl_encoder')
    path = spec['sensor_case']['native_frames']
    with gzip.open(path, 'rt') as stream:
        original = json.loads(next(stream))['scan_source']
    config = original['configuration']
    actor = original['acquisition']['actor_root']
    own = [p.replace(actor, '/World/new_sensor_actor', 1) for p in config['own_collision_paths']]
    result = paper_scan_settings(path, own)
    assert result['return_model'] == FIRST_RETURN_MODEL
    assert result['own_collision_paths'] == own
    assert {k: v for k, v in result.items() if k != 'own_collision_paths'} == {
        k: v for k, v in config.items() if k != 'own_collision_paths'}
    with pytest.raises(ValueError, match='collider suffixes'):
        paper_scan_settings(path, own[:-1])
