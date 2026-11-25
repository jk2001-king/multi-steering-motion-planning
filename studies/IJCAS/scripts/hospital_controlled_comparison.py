"""Controlled-speed hospital experiments using the existing shared kernels.

Import is pure: no Kit, sockets, timeline, actor, or experiment reservation.
The CLI performs one bounded OFFLINE H2 planning pair, never a live trial.
This is an engineering pilot, not an independently repeated final sample.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from types import FunctionType

import numpy as np

import audit_hospital_paper_readiness as shared
import hospital_mission_planning as mission
from hospital_runtime_evidence_io import write_exclusive


ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / 'studies/IJCAS/results/hospital_paper_gate'
REFERENCE = GATE / 'planning/hospital_h2_unified_planning_r03/P_dynamics_full_envelope.json'
OUTPUT = GATE / 'controlled_comparison_20261120/H2/engineering_pair_r02'
VERSION = 'HOSPITAL_SHARED_CRUISE30_ORIGINAL_SPATIAL_SEARCH_CLOCK_ENGINEERING_PAIR_v2'
SPATIAL_SEARCH_NOMINAL_SPEED_M_S = .050
B2, P = shared.METHODS[2:]


def sha(path):
    return shared.file_sha(path)


@dataclass(frozen=True)
class CommonSpeedPolicy:
    cruise_m_s: float = .030
    levels_m_s: tuple = (.003, .010, .030)
    acceleration_m_s2: float = .020
    terminal_model_speed_m_s: float = .005
    primitive_m: float = .050
    forward_tracking_floor_m_s: float = .010
    reverse_tracking_floor_m_s: float = .020

    def validate(self):
        if self != CommonSpeedPolicy():
            raise ValueError('This first engineering pair is frozen at the declared common30 policy')
        return self

    def profile(self, actions):
        self.validate()
        if not actions or any(type(a.get('direction')) is not int or a['direction'] not in (-1, 1)
                              for a in actions):
            raise ValueError('Nonempty signed primitive sequence required')
        values = []
        for index in range(len(actions)):
            remaining = mission.direction_remaining_arc(actions, index, self.primitive_m)
            cap = min(self.cruise_m_s, math.sqrt(self.terminal_model_speed_m_s**2
                       + 2 * self.acceleration_m_s2 * remaining))
            eligible = [v for v in self.levels_m_s if v <= cap + 1e-12]
            if not eligible:
                raise ValueError('Common braking profile has no admissible speed')
            values.append(max(eligible))
        return values


def bind(function, **values):
    result = FunctionType(function.__code__, dict(function.__globals__, **values),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def common_caps(actions, levels=(.003, .010, .030, .050), *,
                forward_floor=.010, reverse_floor=.020):
    """One cap, independent of model score, method, and observed outcome."""
    policy = CommonSpeedPolicy().validate()
    policy.profile(actions)
    if max(forward_floor, reverse_floor) > policy.cruise_m_s:
        raise ValueError('Common cruise must not be below either tracking floor')
    return (policy.cruise_m_s,)


def validate_profile(row):
    expected = CommonSpeedPolicy().profile(row['selected_actions'])
    actual = [a['target_speed_m_s'] for a in row['selected_actions']]
    if len(actual) != len(expected) or any(abs(a-b) > 1e-12 for a, b in zip(actual, expected)):
        raise ValueError('A method selected a speed outside the common method-blind profile')
    return dict(rule_identical=True, maximum_target_m_s=max(actual),
                distinct_targets_m_s=sorted(set(actual)), primitive_count=len(actual),
                exact_observed_speed_matching_NOT_yet_measured=True)


def load_inputs():
    reference = json.loads(REFERENCE.read_bytes())
    geometry_path = Path(reference['geometry_manifest'])
    if sha(geometry_path) != reference['geometry_manifest_sha256']:
        raise ValueError('The original H2 task geometry changed')
    geometry = json.loads(geometry_path.read_bytes())
    if (geometry['scenario'] != 'H2_bedside_lateral_alignment'
            or geometry['start_xyyaw'] != reference['start_world_xyyaw']
            or geometry['goal_xyyaw'] != reference['goal_world_xyyaw']):
        raise ValueError('Exact original H2 endpoints required')
    grid_path = Path(geometry['occupancy_grid'])
    with np.load(grid_path) as grid:
        occupied = grid['occupied'].copy()
        origin = grid['origin_xy_m'].copy()
        resolution = float(grid['resolution_m'])
    if (occupied.dtype != np.dtype(bool) or resolution != .010
            or hashlib.sha256(occupied.tobytes()).hexdigest() != reference['planning_occupancy_digest']):
        raise ValueError('The exact recorded common hospital raster is required')
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
    model = MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**reference['rigid_body_parameter_bounds']))
    return reference, geometry, geometry_path, grid_path, occupied, origin, resolution, model


def run_offline_pair(output=OUTPUT):
    """One new planning attempt per method; all outcomes retained, no retry."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Consumed engineering-pair output; do not retry: ' + str(output))
    policy = CommonSpeedPolicy().validate()
    reference, geometry, geometry_path, grid_path, occupied, origin, resolution, model = load_inputs()
    begun = time.perf_counter()
    method, auditor, connections, parameters = shared.unified_h3_planner_binding(
        reference['geometry_profile'], geometry, occupied, origin, resolution,
        reserve_m=.050, terminal_direction_change_reserve=2)
    # Keep the original spatial search/heuristic clock consistent for BOTH
    # methods. Kinematic seed geometry is speed-invariant. The prior pilot
    # changed the search clock to30 while its guidance retained50, exhausting
    # P's seed budget. Actual candidate propagation/execution still uses the
    # fixed method-blind30 profile, NOT this seed-search normalization.
    method = bind(method, SPEED=SPATIAL_SEARCH_NOMINAL_SPEED_M_S)
    guided = bind(shared.unified_h3_guided_candidate, common_cruise_profile_caps=common_caps)
    from patient_initial_stability import steering_lattice as lattice
    connections.compiled(lattice.State(0., 0., 0., 'coordinated_counterphase', 0., 0., True),
        'coordinated_counterphase', 0., 1, speed_m_s=.05, rigid_body_model=model,
        target_speed_m_s=policy.cruise_m_s)
    common_initialization_s = time.perf_counter() - begun
    # The H3-specific receipt is NOT forged into an H2 admission. Use its
    # original independent action/NE/SAT auditor after explicit task/map checks.
    action_auditor = auditor.__globals__['audit_actions']
    scripts = ROOT / 'studies/IJCAS/scripts'
    sources = [Path(__file__), Path(shared.__file__), Path(mission.__file__), Path(lattice.__file__),
        scripts/'plan_hospital_h1.py', scripts/'plan_hospital_h3_native.py',
        scripts/'hospital_native_envelope.py', scripts/'hospital_exact_sat.c',
        scripts/'hospital_planning_tools.py', scripts/'hospital_geometry_tools.py',
        scripts/'audit_hospital_h1_plan.py', scripts/'hospital_runtime_evidence_io.py',
        ROOT/'src/patient_initial_stability/patient_initial_stability/hierarchical_dynamics_planner.py',
        ROOT/'src/patient_initial_stability/patient_initial_stability/rigid_body_edge.py']
    pins = {str(p.resolve()): sha(p) for p in [REFERENCE, geometry_path, grid_path, *sources]}
    metadata = {k: reference[k] for k in ('scenario', 'environment_condition', 'geometry_profile',
        'geometry_manifest', 'geometry_manifest_sha256', 'start_world_xyyaw', 'goal_world_xyyaw',
        'goal_local_xyyaw', 'planning_occupancy_digest', 'rigid_body_parameter_bounds')}
    metadata.update(schema_version=1, version=VERSION, trial_kind='OFFLINE_H2_COMMON_SPEED_ENGINEERING_PAIR',
        common_speed_policy=asdict(policy), common_max_command_speed_m_s=.050,
        spatial_search_nominal_speed_m_s=SPATIAL_SEARCH_NOMINAL_SPEED_M_S,
        spatial_search_normalization_is_NOT_the_execution_speed=True,
        prior_seed_timeout_attempt_preserved=str(OUTPUT.with_name('engineering_pair_r01')),
        common_max_acceleration_m_s2=.020, planning_terminal_speed_tolerance_m_s=.005,
        full_envelope_clearance_m=.010, planning_additional_execution_reserve_m=.050,
        continuous_task_search_binding=parameters, common_initialization_time_s=common_initialization_s,
        common_initialization_excluded_from_method_clock_but_reported=True,
        input_sha256=pins, initial_state_rule='same nominal standstill; no initial-state perturbation in this pilot',
        per_method_planning_wall_limit_s=180., lattice_expansion_limit=60000,
        per_method_execution_simulation_limit_s=180., per_method_execution_wall_limit_s=900.,
        outcome_based_cruise_selection=False, current_user_requested_experiments=True,
        physical_trials_executed=0, Isaac_execution_verified=False, Nav2_execution_verified=False,
        actual_motion_authorized=False, use_for_final_statistics=False,
        live_admission='separate actual environment/static/controller-model validation required')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'planning_occupancy.npz', occupied=occupied,
                        origin_xy_m=origin, resolution_m=resolution)
    write_exclusive(output/'definition.json', dict(metadata, methods=[B2, P],
        explicit_preexecution_role='engineering condition/model check, not a final repetition',
        no_automatic_retry=True, manuscript_modified=False))
    from hospital_mission_lifecycle import freeze_workflow_sources
    freeze_workflow_sources(output/'sources.zip', {str(p.resolve()): pins[str(p.resolve())] for p in sources})
    print('[CONTROLLED H2] output:', output, flush=True)
    rows = []
    for name in (B2, P):
        print('[CONTROLLED H2] START', name, 'common30 policy; budget180s', flush=True)
        start = time.perf_counter()
        if name == P:
            seed = shared.unified_P_owned_seed(method, geometry, occupied, origin, resolution, model,
                budget_s=180., node_limit=60000, policy=shared.UNIFIED_P_SEED_POLICY)
            write_exclusive(output/'P_own_fresh_seed.json', dict(metadata, **seed))
            remaining = 180. - (time.perf_counter() - start)
            if remaining > 0:
                row = guided(method, seed, geometry, occupied, origin, resolution, model,
                             budget_s=remaining, execution_budget_s=180.)
            else:
                row = dict(method=name, plan_success=False, status='PLANNING_BUDGET_EXHAUSTED',
                           selected_actions=[], selected_states=[])
            row.update(own_seed_time_s=seed['owned_seed_total_time_s'],
                       own_seed_attempts=seed['owned_seed_attempts'])
        else:
            row = method(name, geometry, occupied, origin, resolution, model, 180., 60000, {})
            if row['plan_success']:
                speeds = policy.profile(row['selected_actions'])
                row['selected_actions'] = [dict(a, target_speed_m_s=s) for a, s in zip(row['selected_actions'], speeds)]
                row['baseline_execution_schedule'] = 'same predetermined common30 direction-section braking profile as P-CS'
        row.update(planning_time_s=time.perf_counter()-start, wall_clock_budget_s=180.,
                   controlled_variant='B2-CS' if name == B2 else 'P-CS')
        if row['plan_success']:
            row['common_profile_check'] = validate_profile(row)
        target = output/(name+'.json')
        write_exclusive(target, dict(metadata, **row))
        checked = None
        if row['plan_success']:
            checked = action_auditor(target)
            write_exclusive(output/'offline_audits'/(name+'.json'), checked)
        result = {k: row.get(k) for k in ('method', 'controlled_variant', 'plan_success',
                    'status', 'planning_time_s', 'action_count', 'path_length_m')}
        result['independent_offline_audit'] = checked['verdict'] if checked else 'NOT_RUN_NO_PLAN'
        result['failed_audit_checks'] = [k for k,v in checked['checks'].items() if not v] if checked else []
        rows.append(result)
        print('[CONTROLLED H2] RESULT', result, flush=True)
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError('Input changed during controlled planning: '+path)
    summary = dict(metadata, outcomes=rows, all_attempts_retained=True,
                   status='OFFLINE_PAIR_RECORDED_NOT_A_DRIVING_RESULT', no_automatic_retry=True)
    write_exclusive(output/'summary.json', summary)
    print('[CONTROLLED H2] saved:', output/'summary.json', flush=True)
    return summary


def run_model_screens(folder=OUTPUT):
    """Existing common controller, both nominal resolutions; no SDK action."""
    folder = Path(folder).resolve()
    summary = json.loads((folder/'summary.json').read_bytes())
    target = folder/'model_evaluation_binding.json'
    if target.exists():
        raise FileExistsError('Consumed model-screen definition; no automatic retry')
    write_exclusive(target, dict(profile=summary['geometry_profile'],
        planning_summary=str(folder/'summary.json'),
        planning_summary_sha256=sha(folder/'summary.json'),
        source_sha256=sha(Path(__file__)),
        kind='DERIVED_COMMON_SPEED_MODEL_BINDING_NOT_A_PHYSICAL_TRIAL',
        integration_steps_s=[.020, .010], physical_trials=0))
    reports = {}
    for method in (B2, P):
        plan = folder/(method+'.json')
        if not json.loads(plan.read_bytes())['plan_success']:
            reports[method] = 'NOT_RUN_NO_PLAN'
            continue
        reports[method] = str(shared.write_unified_mission_model_screen(
            target, plan, summary['geometry_manifest'], folder/'planning_occupancy.npz',
            output=folder/'executor_model_screen'/method, simulation_budget_s=180.))
    write_exclusive(folder/'model_screens_summary.json', dict(reports=reports,
        physical_trials=0, software_resolution_cases_NOT_independent_experiments=True))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--model-screens-only', action='store_true')
    args = parser.parse_args()
    if args.model_screens_only:
        run_model_screens(args.output)
    else:
        run_offline_pair(args.output)


if __name__ == '__main__':
    main()
