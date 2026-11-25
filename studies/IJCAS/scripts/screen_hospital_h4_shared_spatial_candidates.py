#!/usr/bin/env python3
"""Finite H4 straight/arc/straight candidates on the PROVISIONAL authored map.

This is a disclosed NEW spatial generator, not an unchanged H1 lattice search,
complete search, cooked-collision certificate or a paper comparison. All four
selectors receive the same outcome-independent geometry-generated pool; B0 is
restricted to counterphase. P reuses the original H1 full-model refinement.
No original plan/model/result, live state, physics or safety limit is changed.
"""

from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

import build_hospital_h4_setup_candidate as setup_builder
from hospital_planning_tools import HospitalEnvelopeChecker, local_to_world
import patient_initial_stability.hierarchical_dynamics_planner as refinement
import patient_initial_stability.steering_lattice as lattice
import patient_initial_stability.rigid_body_edge as rigid
import plan_hospital_h1 as shared


ROOT, GATE = setup_builder.ROOT, setup_builder.BASE
OUTPUT = GATE/'hospital_h4_shared_candidate_screen/screen_r01'
TARGET = OUTPUT/'shared_candidate_screen.json'
MODEL = GATE/'planning/hospital_h1_planning_r06/P_dynamics_full_envelope.json'
MODEL_SHA = '1d3f6978a5bb3bb6085a2d8882fe3855f1811599197cbc4365a223d874b292ac'
VERSION = 'H4_shared_forward_SAS_spatial_family_original_model_candidate_map_v1'
GENERATOR_BUDGET_S = 45.
PROPOSED_BUDGET_S = 60.
MAXIMUM_STEPS = 200
POSITION_TOLERANCE_M = .025
HEADING_TOLERANCE_RAD = math.radians(2.)
STRAIGHT_COUNT_NEIGHBOURHOOD = (-1, 0, 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(pins):
    for name, digest in pins.items():
        if sha(name) != digest:
            raise ValueError('Frozen H4 candidate/model/source changed: '+name)


def load_inputs():
    setup = json.loads(setup_builder.TARGET.read_bytes())
    verify(setup['input_sha256']); verify(setup['output_sha256'])
    if (setup['version'] != setup_builder.VERSION or not all(setup['checks'].values())
            or setup['complete_stage_collision_coverage_verified'] is not False
            or setup['strict_map_or_final_task_drive_admission'] is not False
            or setup['footprint'] != dict(length_m=1.630, width_m=.965, clearance_m=.010)):
        raise ValueError('Exact PROVISIONAL geometry-derived H4 setup required')
    with np.load(setup['candidate_grid']) as grid:
        occupied = grid['occupied'].copy()
        origin = grid['origin_xy_m'].copy()
        resolution = float(grid['resolution_m'])
    if (occupied.dtype != np.dtype(bool)
            or list(occupied.shape) != setup['raster']['shape_rows_columns']
            or not np.array_equal(origin, setup['raster']['origin_xy_m'])
            or resolution != setup['raster']['resolution_m']
            or hashlib.sha256(occupied.tobytes()).hexdigest() != setup['candidate_occupancy_digest']):
        raise ValueError('Exact unchanged candidate occupancy required')
    if sha(MODEL) != MODEL_SHA:
        raise ValueError('Original H1 Newton-Euler model reference changed')
    parameters = json.loads(MODEL.read_bytes())['rigid_body_parameter_bounds']
    model = rigid.MassNormalizedRigidBodyEdgeModel(rigid.RigidBodyParameterBounds(**parameters))
    sources = [Path(__file__), Path(setup_builder.__file__), Path(shared.__file__),
        Path(refinement.__file__), Path(lattice.__file__), Path(rigid.__file__),
        ROOT/'studies/IJCAS/scripts/hospital_planning_tools.py',
        ROOT/'studies/IJCAS/scripts/hospital_geometry_tools.py']
    pins = {**setup['input_sha256'], **setup['output_sha256'],
        **{str(path): sha(path) for path in (*sources, setup_builder.TARGET, MODEL)}}
    return setup, occupied, origin, resolution, model, pins


def actions_for(arc_mode, curvature, before, turn, after):
    if (arc_mode not in lattice.ALL_MODES or curvature not in lattice.CURVATURES or curvature == 0
            or any(not isinstance(value, int) or value < 0 for value in (before, turn, after))
            or turn == 0 or before+turn+after > MAXIMUM_STEPS):
        raise ValueError('Bounded original-mode/curvature forward SAS family required')
    return ([dict(mode='coordinated_counterphase', curvature_1_m=0., direction=1)]*before
        + [dict(mode=arc_mode, curvature_1_m=float(curvature), direction=1)]*turn
        + [dict(mode='coordinated_counterphase', curvature_1_m=0., direction=1)]*after)


def replay(actions, dynamics, checker=None):
    """Exactly the existing baseline propagator and lattice edge cost."""
    state = lattice.State(0., 0., 0., 'coordinated_counterphase', 0., 0., False)
    states = []; poses = []; cost = 0.
    for action in actions:
        previous = state
        state, edge, elapsed = lattice.propagate(state, action['mode'], action['curvature_1_m'],
            action['direction'], speed_m_s=shared.SPEED, dynamics=dynamics)
        target_front, target_rear = lattice.steering_targets(action['mode'], action['curvature_1_m'])
        cost += elapsed+.02*abs(action['curvature_1_m'])
        if previous.started and dynamics.include_mode_switch_cost and previous.mode != action['mode']:
            cost += lattice.MODE_SWITCH_PENALTY_S
        if previous.started and dynamics.include_steering_transient_cost:
            cost += lattice.STEERING_TRANSIENT_PENALTY_WEIGHT*max(
                abs(target_front-previous.front), abs(target_rear-previous.rear))/lattice.STEERING_RATE_RAD_S
        states.append(state); poses.extend(edge)
        if checker is not None and checker(edge):
            return None, states, poses
    return cost, states, poses


def terminal_errors(state, goal):
    return math.hypot(state.x-goal[0], state.y-goal[1]), abs(lattice.wrap(state.yaw-goal[2]))


def generate_pool(goal, checker):
    """Finite, goal-derived counts; no successful physical endpoint is used.

    Enumerate original positive/negative curvature lattice and all modes.
    Arc counts must meet the original 2deg heading tolerance. Solve the two
    straight lengths in the resulting exact discrete-propagator frame, then
    enumerate the predetermined +/-one-edge rounding neighbourhood. Replay
    every retained candidate using the original propagator and full envelope.
    """
    if not np.isfinite(goal).all() or abs(math.sin(goal[2])) < .1:
        raise ValueError('Nonparallel finite H4 turning goal required')
    pool = []; seen = set()
    diagnostics = dict(arc_count_pairs=0, count_neighbours=0, count_or_length_rejected=0,
        endpoint_rejected=0, full_envelope_rejected=0, accepted=0, budget_exhausted=False)
    try:
        for mode in lattice.ALL_MODES:
            for curvature in lattice.CURVATURES:
                if curvature*goal[2] <= 0:
                    continue
                for turn in range(1, MAXIMUM_STEPS+1):
                    checker.check_deadline()
                    if abs(lattice.wrap(curvature*turn*lattice.STEP_M-goal[2])) > HEADING_TOLERANCE_RAD:
                        continue
                    diagnostics['arc_count_pairs'] += 1
                    _, arc_states, _ = replay(actions_for(mode, curvature, 0, turn, 0), lattice.KINEMATIC_DYNAMICS)
                    arc = arc_states[-1]
                    after_m = (goal[1]-arc.y)/math.sin(arc.yaw)
                    before_m = goal[0]-arc.x-after_m*math.cos(arc.yaw)
                    before0, after0 = round(before_m/lattice.STEP_M), round(after_m/lattice.STEP_M)
                    for db, da in itertools.product(STRAIGHT_COUNT_NEIGHBOURHOOD, repeat=2):
                        checker.check_deadline()
                        diagnostics['count_neighbours'] += 1
                        before, after = before0+db, after0+da
                        if min(before, after) < 0 or before+turn+after > MAXIMUM_STEPS:
                            diagnostics['count_or_length_rejected'] += 1
                            continue
                        key = (mode, curvature, before, turn, after)
                        if key in seen:
                            continue
                        seen.add(key)
                        actions = actions_for(*key)
                        cost, states, _ = replay(actions, lattice.KINEMATIC_DYNAMICS)
                        position, heading = terminal_errors(states[-1], goal)
                        if position > POSITION_TOLERANCE_M or heading > HEADING_TOLERANCE_RAD:
                            diagnostics['endpoint_rejected'] += 1
                            continue
                        _, _, poses = replay(actions, lattice.KINEMATIC_DYNAMICS)
                        if checker([(0., 0., 0.), *poses]):
                            diagnostics['full_envelope_rejected'] += 1
                            continue
                        name = f'{mode}_k{curvature:+.2f}_S{before}_A{turn}_S{after}'
                        pool.append(dict(candidate_id=name, family='forward_straight_arc_straight',
                            arc_mode=mode, curvature_1_m=curvature, straight_before_count=before,
                            arc_count=turn, straight_after_count=after, actions=actions,
                            kinematic_cost_s=cost, path_length_m=len(actions)*lattice.STEP_M,
                            terminal_position_error_m=position, terminal_heading_error_deg=math.degrees(heading)))
                        diagnostics['accepted'] += 1
    except TimeoutError:
        diagnostics['budget_exhausted'] = True
    return pool, diagnostics


def baseline_select(method, pool, goal, start):
    modes, dynamics = shared.METHODS[method]
    candidates = []; begun = time.perf_counter()
    for item in pool:
        if any(action['mode'] not in modes for action in item['actions']):
            continue
        cost, states, _ = replay(item['actions'], dynamics)
        position, heading = terminal_errors(states[-1], goal)
        candidates.append((cost, position, item['candidate_id'], states, item))
    result = dict(method=method, plan_success=bool(candidates), planner_dynamics=dynamics.name,
        allowed_modes=list(modes), eligible_candidate_count=len(candidates),
        selected_actions=[], selected_states=[], initial_steering_propagated=False,
        predicted_terminal_speed_m_s=None, actual_motion_authorized=False,
        use_for_final_statistics=False, physical_execution_verified=False,
        search_scope='same finite SAS pool; NOT weighted lattice/global optimality',
        planning_time_s=time.perf_counter()-begun)
    if not candidates:
        result['status'] = 'NO_CANDIDATE_IN_FINITE_SHARED_FAMILY_NOT_GLOBAL_INFEASIBILITY'
        return result
    cost, position, name, states, item = min(candidates, key=lambda row: row[:3])
    result.update(status='PROVISIONAL_MAP_NATIVE_PLAN_ONLY_NOT_DRIVE_ADMISSION',
        selected_candidate_id=name, selected_actions=[{**action, 'target_speed_m_s': shared.SPEED} for action in item['actions']],
        selected_states=[shared.state_record(state, start, False) for state in states],
        action_count=len(states), path_length_m=len(states)*lattice.STEP_M,
        mode_switch_count=sum(a['mode'] != b['mode'] for a, b in zip(item['actions'], item['actions'][1:])),
        search_cost_s=cost, nominal_constant_speed_travel_time_s=len(states)*lattice.STEP_M/shared.SPEED,
        predicted_terminal_position_error_m=position,
        predicted_terminal_heading_error_deg=math.degrees(lattice.wrap(states[-1].yaw-goal[2])),
        kinematic_zero_state_velocities_are_NOT_stop_evidence=True)
    return result


def analyze():
    setup, occupied, origin, resolution, model, pins = load_inputs()
    begun = time.perf_counter()
    checker = HospitalEnvelopeChecker(occupied, origin, resolution, setup['start_xyyaw'],
        deadline=begun+GENERATOR_BUDGET_S)
    pool, diagnostics = generate_pool(setup['local_goal_xyyaw'], checker)
    generation_time = time.perf_counter()-begun
    print('H4 shared SAS generator:', len(pool), 'candidates;', diagnostics, flush=True)
    rows = [baseline_select(method, pool, setup['local_goal_xyyaw'], setup['start_xyyaw'])
            for method in list(shared.METHODS)[:3]]
    if pool:
        p = shared.plan_method('P_dynamics_full_envelope', setup, occupied, origin, resolution,
            model, PROPOSED_BUDGET_S, 20000, {row['candidate_id']: row['actions'] for row in pool})
        p.update(search_scope='NEW shared finite SAS pool followed by unchanged H1 refinement; not full lattice search',
            actual_motion_authorized=False, use_for_final_statistics=False,
            provisional_map_and_unvalidated_H4_body_model=True)
    else:
        p = dict(method='P_dynamics_full_envelope', plan_success=False,
            status='NO_SHARED_SPATIAL_CANDIDATE_NOT_GLOBAL_INFEASIBILITY', selected_actions=[], selected_states=[],
            actual_motion_authorized=False, use_for_final_statistics=False)
    rows.append(p)
    for row in rows:
        print(row['method'], row['status'], 'actions:', row.get('action_count'), flush=True)
    verify(pins)
    return dict(version=VERSION, scenario=setup['scenario'],
        status='PROVISIONAL_MAP_FOUR_METHOD_SHARED_FAMILY_SCREEN_NOT_PHYSICAL_OR_FINAL_COMPARISON',
        task_candidate=str(setup_builder.TARGET), same_fixed_task_and_candidate_map_for_all_methods=True,
        spatial_generator_changed_relative_to_H1_H2=True, original_propagators_model_limits_and_gains_changed=False,
        shared_pool_is_available_before_any_selector_outcome=True,
        B0_only_counterphase_filter_is_prescribed_method_constraint=True,
        P_receives_all_spatially_eligible_pool_candidates_not_only_other_selected_winners=True,
        spatial_generation_budget_s=GENERATOR_BUDGET_S, proposed_refinement_budget_s=PROPOSED_BUDGET_S,
        shared_spatial_generation_time_s=generation_time, generation_diagnostics=diagnostics,
        maximum_primitive_count=MAXIMUM_STEPS, straight_rounding_neighbourhood=list(STRAIGHT_COUNT_NEIGHBOURHOOD),
        full_envelope_clearance_m=.010, common_max_command_speed_m_s=shared.SPEED,
        common_max_acceleration_m_s2=shared.DECELERATION, planning_terminal_speed_tolerance_m_s=shared.TERMINAL_SPEED,
        rigid_body_parameter_bounds=asdict(model.bounds), model_reference=str(MODEL), model_reference_sha256=MODEL_SHA,
        initial_pose_velocity_and_steering='nominal planning origin/standstill/zero joints; not an actual robot reset',
        start_world_xyyaw=setup['start_xyyaw'], goal_world_xyyaw=setup['goal_xyyaw'],
        goal_local_xyyaw=setup['local_goal_xyyaw'], candidate_occupancy_digest=setup['candidate_occupancy_digest'],
        finite_family_exhaustion_or_refinement_timeout_does_NOT_prove_global_infeasibility=True,
        sampled_swept_envelope_not_analytic_continuous_collision_proof=True,
        strict_complete_collider_map_verified=False, cooked_collision_or_support_verified=False,
        full_H4_body_response_model_independently_validated=False,
        force_load_safety_constraints_are_not_isolated_as_cause_of_improvement=True,
        candidate_pool=pool, methods=rows, model_fit_or_CICS_runtime_integration=False,
        new_PhysX_executed=False, H3_final_comparison_complete=False, actual_motion_authorized=False,
        use_for_final_statistics=False, input_sha256=pins)


def draw(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from hospital_geometry_tools import footprint_polygon
    setup, occupied, origin, resolution, _, _ = load_inputs()
    fig, ax = plt.subplots(figsize=(8, 9), layout='constrained')
    ax.imshow(occupied, cmap='Greys', origin='lower', extent=[*([origin[0], origin[0]+occupied.shape[1]*resolution]),
        origin[1], origin[1]+occupied.shape[0]*resolution])
    colors = ('#d6a000', '#4b9ac6', '#b36cb1', '#16824f')
    for row, color in zip(report['methods'], colors):
        if not row['plan_success']:
            continue
        poses = [setup['start_xyyaw']]+[state['world_xyyaw'] for state in row['selected_states']]
        xy = np.asarray(poses)
        ax.plot(xy[:, 0], xy[:, 1], color=color, label=row['method'], linewidth=1.6)
        for pose in poses[::12]:
            ax.add_patch(Polygon(footprint_polygon(pose), fill=False, edgecolor=color, alpha=.22, linewidth=.5))
    for key in ('start_xyyaw', 'goal_xyyaw'):
        x, y, yaw = setup[key]
        ax.plot(x, y, 'ko', markersize=4)
        ax.arrow(x, y, .4*math.cos(yaw), .4*math.sin(yaw), width=.02, color='black')
    ax.set(aspect='equal', xlabel='Hospital world X (m)', ylabel='Hospital world Y (m)',
        title='H4 shared finite SAS family: provisional authored map\nNative plan predictions, not physical executions')
    ax.legend(fontsize=7, loc='upper right')
    fig.savefig(OUTPUT/'shared_candidate_preview.png', dpi=160)
    plt.close(fig)


def main():
    if TARGET.exists():
        report = json.loads(TARGET.read_bytes()); verify(report['input_sha256']); verify(report['output_sha256'])
        print('Existing H4 shared candidate screen verified; no overwrite:', TARGET)
        return
    report = analyze()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    draw(report)
    report['output_sha256'] = {str(OUTPUT/'shared_candidate_preview.png'): sha(OUTPUT/'shared_candidate_preview.png')}
    with TARGET.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('saved:', TARGET, flush=True)
    print('No complete/cooked map, static support, execution, statistics or H3 whole-route admission.', flush=True)


if __name__ == '__main__':
    main()
