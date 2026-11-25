"""Shared selection, new offline planning, SDK handoff and archived analysis."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys

from ijcas_paths import ROOT, load_archived_json, resolve_archived_path
from . import selection


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def reserve(spec):
    if spec['missing_inputs']:
        raise FileNotFoundError('Missing experiment inputs: ' + repr(spec['missing_inputs']))
    output = Path(spec['output']) if spec['output'] else _ijcas_path(ROOT, 'results/experiments', datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + spec['cell']['cell_id'])
    output = output.resolve()
    for folder in ('src', 'studies', 'isaac', 'config', 'tools', 'migration', 'archive', 'build', 'install'):
        protected = _ijcas_path(ROOT, folder)
        if output == ROOT or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Output must not overwrite code, assets or archived evidence')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'selection.json', spec)
    return output


@contextmanager
def single_worker():
    folder = _ijcas_path(ROOT, 'results')
    folder.mkdir(exist_ok=True)
    with (folder / '.experiment.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another selected experiment is active; no concurrent planning/driving') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_pins():
    paths = []
    for folder in ('studies/IJCAS/scripts', 'isaac/scripts',
                   'src/patient_initial_stability/patient_initial_stability',
                   'src/ijcas_experiments/ijcas_experiments'):
        paths.extend((_ijcas_path(ROOT, folder)).glob('*.py'))
    paths.extend((Path(selection.default_catalog()), _ijcas_path(ROOT, 'tools/ijcas_paths.py')))
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def prepare_protocol(spec, output):
    parent = load_archived_json(spec['execution_protocol'])
    inherited = ('tasks', 'common_speed_policy', 'P_seed_portfolio',
        'original_evidence_inputs_sha256', 'uniform_P_lookahead_edges_all_tasks',
        'seed_total_budget_s', 'planning_total_budget_s', 'lattice_expansion_limit_per_search_call',
        'beam_width', 'local_iterations', 'common_execution_controller_version',
        'common_controller_contract', 'common_actual_goal_position_m', 'common_actual_goal_heading_deg',
        'common_actual_stopped_speed_m_s', 'common_fixed_stationary_hold_s')
    value = {key: deepcopy(parent[key]) for key in inherited if key in parent}
    # These values identify inputs/algorithms, not current execution authority.
    value = dict(value, version='SELECTED_FRESH_RUN_v1',
        parent_protocol=spec['execution_protocol'], parent_protocol_sha256=sha(spec['execution_protocol']),
        parent_protocol_version=parent['version'],
        selection_sha256=sha(output / 'selection.json') if (output / 'selection.json').exists() else None,
        main_cells=[deepcopy(spec['cell'])], main_comparison_cells=[deepcopy(spec['cell'])],
        status='NEW_SELECTION_NOT_A_PUBLISHED_RESULT', old_results_never_replaced=True,
        automatic_dispatch=False, user_authority='Explicit current CLI allow_motion flag only',
        current_motion_authorized=spec['stage'] == 'run' and spec['allow_motion'])
    write(output / 'protocol.json', value)
    pins = source_pins()
    if (output / 'selection.json').exists():
        pins[str(output / 'selection.json')] = value['selection_sha256']
    write(output / 'sources.json', pins)
    return value


def private_planner(spec, output):
    """Reuse the final planner's function bodies; only bind I/O and one new output."""
    import hospital_revision_final_campaign as final
    import hospital_revision_planning as planning
    from hospital_controlled_runtime import bind
    protocol_path = output / 'protocol.json'
    io = SimpleNamespace(**dict(vars(final.runtime), read=load_archived_json,
        sha=lambda p: sha(resolve_archived_path(p))))
    campaign = SimpleNamespace(**dict(vars(final.design.original), PROTOCOL=protocol_path))
    seed = bind(final.seed_order.base.followup.seed_portfolio, PROTOCOL=protocol_path)
    shared = SimpleNamespace(**dict(vars(planning.shared), unified_P_owned_seed=seed))
    call = bind(planning.run_cell, runtime=io, campaign=campaign, OUTPUT=output / 'planning',
                __file__=planning.__file__)
    private = SimpleNamespace(**dict(vars(planning), run_cell=call, runtime=io, shared=shared))
    # Preserve steady_both, strict beam ranking and distributed local refinement.
    def verify():
        for path, expected in json.loads((output / 'sources.json').read_text()).items():
            if sha(path) != expected:
                raise ValueError('Selected implementation changed during execution: ' + path)
    plan = bind(final.plan_cell, OUTPUT=output, PROTOCOL=protocol_path,
        EXECUTION_PROTOCOL=protocol_path, runtime=io, planning=private,
        seed_binding=lambda: seed, verify_sources=verify, __file__=final.__file__)
    return plan, verify


def plan_selected(spec, output):
    prepare_protocol(spec, output)
    call, verify = private_planner(spec, output)
    verify()
    result = call(spec['cell']['cell_id'])
    verify()
    write(output / 'planning_result.json', dict(result, new_simulator_trials=0))
    return result


def analyze_selected(spec, output):
    from .reproduce import main_results, sensor_results, raw_sensor_results, same_number, describe, write_csv
    if spec['feedback'] == 'amcl_encoder':
        value = sensor_results()[spec['scenario']]
        raw = raw_sensor_results(spec['scenario'], output=output / 'raw_sensor_metrics')
        for kind in ('localization', 'actual_tracking'):
            for key in ('position_RMS_mm', 'yaw_RMS_deg'):
                same_number(raw[kind][key], value[kind+'_'+key], 'raw_sensor_'+kind+'_'+key)
        write(output / 'sensor_metrics.json', value)
        return dict(kind='ARCHIVED_SENSOR_CASE', metrics=value, raw_metrics=raw, new_trials=0)
    full = main_results()
    cell_id = spec['cell']['cell_id']
    rows = [r for r in full['rows'] if r['cell_id'] == cell_id]
    if len(rows) != 1:
        raise ValueError('One fixed selected result required')
    row = rows[0]
    write_csv(output / 'individual.csv', rows)
    write(output / 'metrics.json', row)
    return dict(kind='ARCHIVED_MAIN_OR_HISTORY_CASE', result=row, new_trials=0,
        result_selection_unchanged=True, sample_SD=describe([row.get('planning_time_s')])['sample_SD'])


def sdk_python(spec):
    python = spec['sdk_python'] or os.environ.get('ISAAC_SIM_PYTHON', '')
    if not python:
        entry = shutil.which('isaacsim')
        candidates = ([Path(entry).resolve().with_name('python')] if entry else [])
        candidates += [Path.home() / 'miniconda3/envs/isaac_env/bin/python']
        python = next((str(p) for p in candidates if p.is_file()), '')
    if not python or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError('Set sdk_python to Isaac Sim Python, not ROS/system Python')
    return python


def sdk_command(spec, output):
    python = sdk_python(spec)
    worker = Path(__file__).with_name('sdk_runner.py')
    return [python, str(worker), str(output / 'selection.json'), str(output)]


def native_plan(spec, output):
    # The final compiled NE/Numba implementation belongs to SDK Python 3.11,
    # not ROS Humble Python 3.10. Starting this worker does NOT start Isaac.
    command = [sdk_python(spec), str(Path(__file__).with_name('native_worker.py')),
               str(output / 'selection.json'), str(output)]
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    with (output / 'planning.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output / 'planning_result.json').read_text())


def execute(spec):
    if spec['stage'] == 'check':
        return dict(selection=spec, ready=not spec['missing_inputs'], simulator_started=False,
                    new_trials=0, check_is_not_physical_validation=True)
    # Validate the SDK path before consuming an output or doing a new plan.
    if spec['stage'] in ('plan', 'run'):
        sdk_python(spec)
    with single_worker():
        output = reserve(spec)
        try:
            if spec['stage'] == 'analyze':
                result = analyze_selected(spec, output)
            else:
                result = native_plan(spec, output)
                if spec['stage'] == 'run' and result['independently_valid_reference']:
                    command = sdk_command(spec, output)
                    # ROS Humble and SDK binary Python libraries must never mix.
                    environment = dict(os.environ)
                    environment.pop('PYTHONPATH', None)
                    with (output / 'simulator.log').open('x') as log:
                        child = subprocess.run(command, env=environment, cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, check=False)
                    if (output / 'sdk_error.json').is_file():
                        failure = json.loads((output / 'sdk_error.json').read_text())
                        raise RuntimeError('SDK attempt failed: ' + failure['error'] +
                            '; retained evidence: ' + str(output))
                    if child.returncode or not (output / 'execution_result.json').is_file():
                        raise RuntimeError('SDK closed without a completion receipt '
                            f'(exit={child.returncode}); inspect {output / "simulator.log"}')
                    result = json.loads((output / 'execution_result.json').read_text())
                    if spec['feedback'] == 'amcl_encoder' and result.get('report') and result.get('actual_route_started'):
                        from .reproduce import raw_sensor_results
                        try:
                            scores = raw_sensor_results(spec['scenario'],
                                route=Path(result['report']).parent, output=output / 'sensor_metrics')
                        except (ValueError, FileNotFoundError, EOFError) as error:
                            # Preserve a physical HOLD as HOLD, even if its
                            # interrupted trace cannot support paired metrics.
                            # A purported PASS must still fail closed.
                            if result.get('actual_verdict') == 'PASS':
                                raise
                            write(output / 'sensor_metrics_error.json', dict(
                                error=type(error).__name__ + ': ' + str(error),
                                incomplete_mission_verdict_unchanged=result.get('actual_verdict'),
                                missing_metrics_not_invented=True))
                            result['raw_sensor_metrics'] = None
                            result['raw_sensor_metric_status'] = 'UNAVAILABLE_INCOMPLETE_RAW_OBSERVATION'
                        else:
                            result['raw_sensor_metrics'] = str(output / 'sensor_metrics/metrics.json')
                            result['all_physical_and_ROS_completion_checks_passed'] = scores[
                                'all_physical_and_ROS_completion_checks_passed']
                elif spec['stage'] == 'run':
                    result = dict(result, simulator_started=False, new_trials=0,
                                  status='NO_VALID_PLAN_NO_PHYSICAL_RUN')
            write(output / 'completed.json', dict(stage=spec['stage'], result=result,
                does_not_replace_original_paper_results=True))
            return dict(output=str(output), stage=spec['stage'], result=result)
        except BaseException as error:
            write(output / 'error.json', dict(error=type(error).__name__ + ': ' + str(error),
                automatic_retry=False, original_records_unchanged=True))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Select an IJCAS environment/task/method/condition; default is check only')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--config', default=None)
    for name in sorted(selection.OPTIONS):
        parser.add_argument('--' + name.replace('_', '-'), default=None)
    args, remaining = parser.parse_known_args(argv)
    # launch_ros appends the ROS node name/remaps to console entry points.
    if remaining and remaining[0] != '--ros-args':
        parser.error('Unknown arguments: ' + ' '.join(remaining))
    try:
        if args.list:
            print(json.dumps(selection.available(), indent=2))
            return 0
        options = {k: v for k, v in vars(args).items() if k in selection.OPTIONS}
        spec = selection.resolve(args.config or None, **options)
        result = execute(spec)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if spec['stage'] == 'run':
            return 0 if result['result'].get('actual_verdict') == 'PASS' else 3
        return int(spec['stage'] == 'check' and not result['ready'])
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
