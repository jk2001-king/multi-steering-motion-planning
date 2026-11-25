"""Resolve one explicit experiment selection without starting any backend."""
from copy import deepcopy
from pathlib import Path
import math

import yaml
from ijcas_paths import ROOT, load_archived_json, resolve_archived_path

METHODS = {
    'B0': 'B0_fixed_counterphase_ramp',
    'B1': 'B1_multimode_kinematic_ramp',
    'B2': 'B2_multimode_transition_cost_ramp',
    'P': 'P_dynamics_full_envelope',
    'P_no_history': 'P_dynamics_full_envelope',
}
STAGES = ('check', 'plan', 'run', 'analyze')
OPTIONS = {'environment', 'scenario', 'method', 'condition', 'feedback', 'stage',
           'output', 'sdk_python', 'headless', 'allow_motion'}


def boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    raise ValueError('Boolean must be true or false')


def default_catalog():
    source = Path(__file__).resolve().parents[1] / 'config/experiments.yaml'
    if source.is_file():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('ijcas_experiments')) / 'config/experiments.yaml'


def read_yaml(path):
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a YAML mapping')
    return value


def workspace_path(value):
    path = Path(value).expanduser()
    return (ROOT / path if not path.is_absolute() else path).resolve()


def resolve(config=None, **overrides):
    catalog = read_yaml(default_catalog())
    options = deepcopy(catalog['defaults'])
    if config:
        supplied = read_yaml(config)
        unknown = set(supplied) - OPTIONS
        if unknown:
            raise ValueError('Unknown experiment settings: ' + ', '.join(sorted(unknown)))
        options.update(supplied)
    for key, value in overrides.items():
        if key not in OPTIONS:
            raise ValueError('Unknown argument: ' + key)
        if value is not None and value != '':
            options[key] = value
    environment = options['environment']
    if environment not in catalog['environments']:
        raise ValueError('Unknown environment; available: ' + ', '.join(catalog['environments']))
    env = catalog['environments'][environment]
    task, method, feedback, stage = (str(options[k]) for k in ('scenario', 'method', 'feedback', 'stage'))
    if task not in env['tasks'] or method not in METHODS or stage not in STAGES:
        raise ValueError('Choose a registered H1-H6 task, B0/B1/B2/P/P_no_history and check/plan/run/analyze')
    condition = options['condition']
    if isinstance(condition, bool) or str(condition) not in ('1', '2', '3'):
        raise ValueError('This fixed protocol has condition 1, 2 or 3; not a new random draw')
    condition = int(condition)
    if feedback not in ('ground_truth', 'amcl_encoder'):
        raise ValueError('Feedback must be ground_truth or amcl_encoder; no automatic fallback')
    if method == 'P_no_history' and task not in ('H2', 'H3'):
        raise ValueError('The retained steady-entry history ablation is defined only for H2/H3')
    if feedback == 'amcl_encoder' and (task not in ('H5', 'H6') or method != 'P' or condition != 1):
        raise ValueError('AMCL/encoder evidence is one separate P integration case each in H5/H6; use condition=1')
    for name in ('headless', 'allow_motion'):
        options[name] = boolean(options[name])
    if stage == 'run' and not options['allow_motion']:
        raise ValueError('run starts a NEW simulated trial; explicitly pass allow_motion=true')
    protocol_path = workspace_path(env['execution_protocol'])
    if not protocol_path.is_file():
        raise FileNotFoundError('Fixed condition inputs missing: ' + str(protocol_path))
    protocol = load_archived_json(protocol_path)
    suffix = '_steady_both' if method == 'P_no_history' else ''
    cell_id = f'{task}_{METHODS[method]}_f{condition:02d}{suffix}'
    matches = [c for c in protocol['main_cells'] if c['cell_id'] == cell_id]
    if len(matches) != 1:
        raise ValueError('Exactly one stored condition is required: ' + cell_id)
    cell = deepcopy(matches[0])
    spec = deepcopy(protocol['tasks'][task])
    declared = env['tasks'][task]
    if any(float(spec[k]) != float(declared[k]) for k in ('simulation_budget_s', 'wall_budget_s')):
        raise ValueError('Catalog clock limits differ from the fixed task protocol')
    if not all(math.isfinite(float(v)) for v in (*cell['start_xyyaw'], *cell['goal_xyyaw'])):
        raise ValueError('Finite start and goal poses required')
    scene = workspace_path(env['scene'])
    original_geometry = resolve_archived_path(spec['original_geometry'])
    missing = [str(p) for p in (scene, original_geometry) if not p.is_file()]
    geometry = load_archived_json(original_geometry) if original_geometry.is_file() else {}
    grid = resolve_archived_path(geometry.get('occupancy_grid', 'MISSING_OCCUPANCY'))
    if not grid.is_file():
        missing.append(str(grid))
    sensor_case = None
    if feedback == 'amcl_encoder':
        from .sensor_runner import sensor_selection
        sensor_case = sensor_selection(task)
        cell = dict(cell_id=sensor_case['archived_case_id'], task=task, method=METHODS[method],
            start_xyyaw=sensor_case['start_xyyaw'], goal_xyyaw=sensor_case['goal_xyyaw'],
            experiment_role='SEPARATE_SENSOR_FUNCTIONAL_REPLAY_NOT_MAIN_REPETITION')
        spec = dict(spec, simulation_budget_s=sensor_case['simulation_budget_s'],
            wall_budget_s=sensor_case['wall_budget_s'], original_geometry=sensor_case['geometry'])
        grid = Path(sensor_case['grid'])
        missing = [str(p) for p in (scene, grid, *(Path(sensor_case[k]) for k in
            ('route_template', 'frozen_plan', 'map_report', 'geometry', 'controller_definition',
             'bridge_configuration', 'native_frames'))) if not p.is_file()]
    return dict(schema=1, environment=environment, scenario=task, method=method,
        condition=condition, feedback=feedback, stage=stage, cell=cell,
        task=spec, scene=str(scene), grid=str(grid),
        execution_protocol=sensor_case['controller_definition'] if sensor_case else str(protocol_path),
        selection_protocol_version='SEPARATE_RETAINED_SENSOR_CASE' if sensor_case else protocol['version'],
        common_speed_policy=sensor_case['speed_policy'] if sensor_case else protocol['common_speed_policy'],
        lookahead_edges=None if sensor_case else protocol['uniform_P_lookahead_edges_all_tasks'],
        original_P_seed_portfolio=None if sensor_case else protocol['P_seed_portfolio'],
        output=str(workspace_path(options['output'])) if options.get('output') else None,
        sdk_python=str(workspace_path(options['sdk_python'])) if options.get('sdk_python') else '',
        headless=options['headless'], allow_motion=options['allow_motion'],
        missing_inputs=missing, physical_execution_verified=False,
        archived_approvals_are_not_motion_authority=True,
        sensor_case=sensor_case, fresh_sensor_execution_supported=True,
        recorded_sensor_case_is_not_a_main_repetition=(feedback == 'amcl_encoder'))


def available():
    catalog = read_yaml(default_catalog())
    return dict(environments={name: list(value['tasks']) for name, value in catalog['environments'].items()},
        methods=list(METHODS), conditions=[1, 2, 3], stages=list(STAGES),
        feedback=dict(ground_truth='H1-H6, all registered methods (history ablation H2/H3 only)',
                      amcl_encoder='separate retained P H5/H6 case, condition=1; plan audits its frozen path'),
        default_stage='check', default_motion_authority=False)
