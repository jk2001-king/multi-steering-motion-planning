#!/usr/bin/env python3
"""NEW shared H4 SAS pool augmented by original-full-model control search.

Only integer primitive counts are searched; response/force/load/model constants
are never fitted. The kinematic-only screen and its 14 P refinement failures
remain immutable. Every selector receives the same union pool and must pass
its OWN native propagation/endpoint/geometry checks; B0 stays counterphase.
This development screen is not the unchanged H1 algorithm, held-out response
validation, a physical drive admission, or final paper comparison.
"""

from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

import audit_hospital_h4_complete_map_support as map_review
import screen_hospital_h4_shared_spatial_candidates as previous


ROOT, GATE = previous.ROOT, previous.GATE
OUTPUT = GATE/'hospital_h4_shared_model_seed_screen/screen_r01'
TARGET = OUTPUT/'shared_model_seed_screen.json'
VERSION = 'H4_shared_kinematic_and_full_model_SAS_control_count_search_v2'
GENERATION_BUDGET_S = 90.
MAXIMUM_FULL_ROLLOUTS = 400
MAXIMUM_NEWTON_ITERATIONS = 4
MAXIMUM_COUNT_CHANGE_PER_ITERATION = 8
NEIGHBOURHOOD = (-1, 0, 1)


def speed_schedule(count):
    if not isinstance(count, int) or count <= 0:
        raise ValueError('Positive original primitive count required')
    return [max(value for value in previous.shared.SPEED_LEVELS
            if value <= math.sqrt(previous.shared.TERMINAL_SPEED**2
                + 2*previous.shared.DECELERATION*(count-index-1)*previous.lattice.STEP_M)+1e-12)
        for index in range(count)]


def full_rollout(actions, model, checker=None, deadline=None):
    """Original carried steering/velocity/Newton-Euler states, no endpoint reset."""
    lattice = previous.lattice
    state = lattice.State(0., 0., 0., 'coordinated_counterphase', 0., 0., True)
    states = []; all_poses = []; times = []; cost = 0.; minimum_stopping_margin = math.inf
    speeds = speed_schedule(len(actions))
    for index, (action, speed) in enumerate(zip(actions, speeds)):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('H4 shared model-seed generation wall-clock budget')
        old = state
        state, poses, elapsed = lattice.propagate(state, action['mode'], action['curvature_1_m'], action['direction'],
            speed_m_s=previous.shared.SPEED, dynamics=lattice.FULL_DYNAMICS,
            rigid_body_model=model, target_speed_m_s=speed)
        configuration = 0.
        if old.started:
            if action['mode'] != old.mode:
                configuration += lattice.MODE_SWITCH_PENALTY_S
            front, rear = lattice.steering_targets(action['mode'], action['curvature_1_m'])
            configuration += lattice.STEERING_TRANSIENT_PENALTY_WEIGHT*max(abs(front-old.front), abs(rear-old.rear))/lattice.STEERING_RATE_RAD_S
        cost += elapsed+state.rigid_body_penalty_s+configuration
        limit = math.sqrt(previous.shared.TERMINAL_SPEED**2
            + 2*previous.shared.DECELERATION*(len(actions)-index-1)*lattice.STEP_M)
        minimum_stopping_margin = min(minimum_stopping_margin, limit-math.hypot(state.longitudinal_velocity, state.lateral_velocity))
        states.append(state); all_poses.extend(poses); times.append(elapsed)
    geometry_clear = None if checker is None else not checker([(0., 0., 0.), *all_poses])
    return dict(states=states, poses=all_poses, speeds=speeds, edge_times=times,
        cost_s=cost, original_force_load_actuator_limits=all(state.rigid_body_feasible for state in states),
        minimum_stopping_margin_m_s=minimum_stopping_margin,
        stopping_reachability=minimum_stopping_margin >= -1e-12,
        sampled_full_envelope_clear=geometry_clear)


def summary_of(result, goal):
    last = result['states'][-1]
    position, heading = previous.terminal_errors(last, goal)
    return dict(terminal_local_xyyaw=[last.x, last.y, last.yaw],
        position_error_m=position, heading_error_deg=math.degrees(heading),
        terminal_speed_m_s=math.hypot(last.longitudinal_velocity, last.lateral_velocity),
        original_force_load_actuator_limits=result['original_force_load_actuator_limits'],
        minimum_stopping_margin_m_s=result['minimum_stopping_margin_m_s'],
        stopping_reachability=result['stopping_reachability'])


def accepted(summary):
    return (summary['position_error_m'] <= previous.POSITION_TOLERANCE_M
        and summary['heading_error_deg'] <= math.degrees(previous.HEADING_TOLERANCE_RAD)
        and summary['terminal_speed_m_s'] <= previous.shared.TERMINAL_SPEED
        and summary['original_force_load_actuator_limits'] and summary['stopping_reachability'])


def dynamic_candidates(old_pool, goal, model, checker, deadline):
    """Solve CONTROL counts, not response-model parameters or measured endpoints.

    One seed per original arc-mode/curvature pair, chosen by kinematic endpoint
    error before any full-model selector outcome. Four bounded finite-difference
    Newton iterations followed by a fixed +/-one-edge integer neighbourhood.
    The complete legacy model is independently replayed from standstill for
    every trial. No actual saved body, steering or caster history is injected.
    """
    groups = {}
    for item in old_pool:
        key = item['arc_mode'], item['curvature_1_m']
        if key not in groups or (item['terminal_position_error_m'], item['candidate_id']) < (
                groups[key]['terminal_position_error_m'], groups[key]['candidate_id']):
            groups[key] = item
    cache = {}; traces = []; retained = []; iterations = []; exhausted = False

    def evaluate(mode, curvature, counts):
        counts = tuple(int(value) for value in counts)
        key = mode, curvature, *counts
        if key in cache:
            return cache[key]
        if time.perf_counter() >= deadline or len(traces) >= MAXIMUM_FULL_ROLLOUTS:
            raise TimeoutError('H4 finite model-seed evaluation budget exhausted')
        if min(counts) < 0 or counts[1] == 0 or sum(counts) > previous.MAXIMUM_STEPS:
            return None
        actions = previous.actions_for(mode, curvature, *counts)
        result = full_rollout(actions, model, deadline=deadline)
        summary = summary_of(result, goal)
        traces.append(dict(arc_mode=mode, curvature_1_m=curvature, counts=list(counts), **summary))
        # Cache endpoints/diagnostics only; later geometry and selectors replay
        # the same carried-state trajectory independently.
        cache[key] = summary
        return summary

    try:
        for (mode, curvature), seed in sorted(groups.items()):
            counts = np.array([seed['straight_before_count'], seed['arc_count'], seed['straight_after_count']], dtype=int)
            for iteration in range(MAXIMUM_NEWTON_ITERATIONS):
                base = evaluate(mode, curvature, counts)
                if base is None:
                    break
                endpoint = np.asarray(base['terminal_local_xyyaw'])
                residual = np.asarray([goal[0]-endpoint[0], goal[1]-endpoint[1], previous.lattice.wrap(goal[2]-endpoint[2])])
                jacobian = np.empty((3, 3))
                valid = True
                for column in range(3):
                    trial = counts.copy(); trial[column] += 1
                    point = evaluate(mode, curvature, trial)
                    if point is None:
                        valid = False; break
                    delta = np.asarray(point['terminal_local_xyyaw'])-endpoint
                    delta[2] = previous.lattice.wrap(delta[2])
                    jacobian[:, column] = delta
                if not valid:
                    break
                try:
                    change = np.rint(np.clip(np.linalg.solve(jacobian, residual),
                        -MAXIMUM_COUNT_CHANGE_PER_ITERATION, MAXIMUM_COUNT_CHANGE_PER_ITERATION)).astype(int)
                except np.linalg.LinAlgError:
                    iterations.append(dict(seed=seed['candidate_id'], iteration=iteration, status='SINGULAR_CONTROL_JACOBIAN'))
                    break
                iterations.append(dict(seed=seed['candidate_id'], iteration=iteration, counts=counts.tolist(),
                    endpoint_residual=residual.tolist(), prescribed_bounded_count_update=change.tolist()))
                if not np.any(change):
                    break
                updated = counts+change
                if min(updated) < 0 or updated[1] == 0 or sum(updated) > previous.MAXIMUM_STEPS:
                    break
                counts = updated
            for delta in itertools.product(NEIGHBOURHOOD, repeat=3):
                trial = counts+np.asarray(delta, dtype=int)
                summary = evaluate(mode, curvature, trial)
                if summary is None or not accepted(summary):
                    continue
                actions = previous.actions_for(mode, curvature, *map(int, trial))
                result = full_rollout(actions, model, checker, deadline)
                if not result['sampled_full_envelope_clear']:
                    continue
                before, turn, after = map(int, trial)
                name = f'{mode}_k{curvature:+.2f}_S{before}_A{turn}_S{after}'
                retained.append(dict(candidate_id=name, family='forward_straight_arc_straight',
                    arc_mode=mode, curvature_1_m=curvature, straight_before_count=before, arc_count=turn,
                    straight_after_count=after, actions=actions,
                    generation_source='frozen_full_model_CONTROL_count_search', full_model_seed_summary=summary))
            print('H4 model-seed family', mode, curvature, 'rollouts:', len(traces), 'retained:', len(retained), flush=True)
    except TimeoutError:
        exhausted = True
    unique = {item['candidate_id']: item for item in retained}
    return list(unique.values()), dict(seed_count=len(groups), full_rollout_evaluations=len(traces),
        budget_exhausted=exhausted, iteration_trace=iterations, endpoint_evaluation_trace=traces,
        retained_dynamic_candidate_count=len(unique))


def select(method, pool, goal, start, model, checker):
    modes, dynamics = previous.shared.METHODS[method]
    full = method == 'P_dynamics_full_envelope'
    begun = time.perf_counter(); evaluations = []; eligible = []
    for item in pool:
        if any(action['mode'] not in modes for action in item['actions']):
            evaluations.append(dict(candidate_id=item['candidate_id'], admitted_by_native_model=False, reason='METHOD_MODE_CONSTRAINT'))
            continue
        actions = item['actions']
        if full:
            result = full_rollout(actions, model, checker)
            summary = summary_of(result, goal)
            valid = accepted(summary) and result['sampled_full_envelope_clear']
            cost, states, speeds = result['cost_s'], result['states'], result['speeds']
            geometry_clear = result['sampled_full_envelope_clear']
        else:
            cost, states, poses = previous.replay(actions, dynamics)
            position, heading = previous.terminal_errors(states[-1], goal)
            geometry_clear = not checker([(0., 0., 0.), *poses])
            summary = dict(position_error_m=position, heading_error_deg=math.degrees(heading),
                terminal_speed_m_s=None, kinematic_zero_state_velocity_is_NOT_stop_evidence=True)
            valid = position <= .025 and heading <= math.radians(2.) and geometry_clear
            speeds = [previous.shared.SPEED]*len(actions)
        evaluations.append(dict(candidate_id=item['candidate_id'], admitted_by_native_model=bool(valid),
            sampled_full_envelope_clear=geometry_clear, **summary))
        if valid:
            eligible.append((cost, summary['position_error_m'], item['candidate_id'], states, speeds, item))
    row = dict(method=method, allowed_modes=list(modes), planner_dynamics=dynamics.name,
        plan_success=bool(eligible), initial_steering_propagated=full,
        all_pool_candidates_considered=True, native_candidate_evaluations=evaluations,
        eligible_candidate_count=len(eligible), selected_actions=[], selected_states=[],
        status='NATIVE_MODEL_CANDIDATE_FOUND_NOT_PHYSICAL_OR_SUPPORT_ADMISSION' if eligible else 'NO_NATIVE_ADMITTED_CANDIDATE_IN_FINITE_POOL_NOT_GLOBAL_INFEASIBILITY',
        selection_wall_time_s=time.perf_counter()-begun, actual_motion_authorized=False, use_for_final_statistics=False,
        executor_or_gains_changed=False, full_state_model_independently_validated=False,
        selector_scope='NEW shared union pool; fixed original curvature/steering/velocity/Newton-Euler models, no extra H1 curvature refinement')
    if not eligible:
        return row
    cost, position, name, states, speeds, item = min(eligible, key=lambda entry: entry[:3])
    row.update(selected_candidate_id=name,
        selected_actions=[{**action, 'target_speed_m_s': speed} for action, speed in zip(item['actions'], speeds)],
        selected_states=[previous.shared.state_record(state, start, full) for state in states],
        action_count=len(states), path_length_m=len(states)*previous.lattice.STEP_M,
        native_cost_s=cost, costs_NOT_directly_comparable_across_fidelity_models=True,
        predicted_terminal_position_error_m=position,
        predicted_terminal_heading_error_deg=math.degrees(previous.lattice.wrap(states[-1].yaw-goal[2])),
        predicted_terminal_speed_m_s=math.hypot(states[-1].longitudinal_velocity, states[-1].lateral_velocity) if full else None)
    return row


def analyze():
    setup, _, origin, resolution, model, pins = previous.load_inputs()
    old = json.loads(previous.TARGET.read_bytes()); previous.verify(old['input_sha256']); previous.verify(old['output_sha256'])
    audit = json.loads(map_review.TARGET.read_bytes()); previous.verify(audit['input_sha256']); previous.verify(audit['output_sha256'])
    with np.load(map_review.GRID) as grid:
        occupied = grid['occupied'].copy()
    if (audit['occupancy_digest'] != setup['candidate_occupancy_digest']
            or audit['original_candidate_grid_changed_cell_count'] != 0
            or not audit['complete_USD_authored_collision_inventory_verified']):
        raise ValueError('Require actual all-stage map reconciliation; do not silently change geometry or goal')
    checker = previous.HospitalEnvelopeChecker(occupied, origin, resolution, setup['start_xyyaw'])
    begun = time.perf_counter()
    new, diagnostics = dynamic_candidates(old['candidate_pool'], setup['local_goal_xyyaw'], model, checker, begun+GENERATION_BUDGET_S)
    pool = {item['candidate_id']: {**item, 'generation_source': 'immutable_kinematic_geometry_pool_v1'} for item in old['candidate_pool']}
    for item in new:
        if item['candidate_id'] not in pool:
            pool[item['candidate_id']] = item
    pool = list(pool.values())
    generation_time = time.perf_counter()-begun
    rows = [select(method, pool, setup['local_goal_xyyaw'], setup['start_xyyaw'], model, checker)
        for method in previous.shared.METHODS]
    sources = [Path(__file__), Path(previous.__file__), Path(map_review.__file__)]
    pins.update({**old['input_sha256'], **old['output_sha256'], **audit['input_sha256'], **audit['output_sha256'],
        **{str(path): previous.sha(path) for path in (*sources, previous.TARGET, map_review.TARGET)}})
    previous.verify(pins)
    return dict(version=VERSION, status='DEVELOPMENT_SHARED_MODEL_SEED_SCREEN_H4_SUPPORT_AND_PHYSICAL_ADMISSION_STILL_HOLD',
        task_candidate=str(previous.setup_builder.TARGET), actual_all_stage_map_audit=str(map_review.TARGET),
        original_kinematic_only_screen_retained=str(previous.TARGET),
        earlier_P_refinement_failure_count=len(old['methods'][-1]['refinement_attempts']),
        all_actual_and_model_prior_PASS_FAIL_HOLDs_retained=True,
        spatial_generator_AND_finite_candidate_selector_are_disclosed_NEW_development_versions=True,
        original_H1_H2_algorithm_unchanged_claimed=False, general_lattice_completeness_or_global_optimality_claimed=False,
        only_CONTROL_counts_searched_NOT_response_model_fit=True,
        original_model_parameters_curvatures_modes_speed_levels_limits_unchanged=True,
        no_actual_saved_body_caster_or_joint_state_injection=True,
        shared_union_pool_fixed_before_any_final_method_selection=True,
        every_method_must_pass_its_OWN_native_model_endpoint_and_envelope=True,
        all_generation_time_must_be_charged_to_EVERY_method=True,
        shared_generation_wall_time_s=generation_time, shared_generation_budget_s=GENERATION_BUDGET_S,
        maximum_full_rollouts=MAXIMUM_FULL_ROLLOUTS, maximum_Newton_iterations=MAXIMUM_NEWTON_ITERATIONS,
        maximum_count_change_per_iteration=MAXIMUM_COUNT_CHANGE_PER_ITERATION,
        final_integer_neighbourhood=list(NEIGHBOURHOOD), generation_diagnostics=diagnostics,
        start_world_xyyaw=setup['start_xyyaw'], goal_world_xyyaw=setup['goal_xyyaw'], goal_local_xyyaw=setup['local_goal_xyyaw'],
        full_envelope_clearance_m=.010, original_planning_position_tolerance_m=.025,
        original_heading_tolerance_deg=2., original_terminal_speed_tolerance_m_s=.005,
        rigid_body_parameter_bounds=asdict(model.bounds), model_reference_sha256=previous.MODEL_SHA,
        candidate_pool=pool, methods=rows,
        model_success_is_NOT_full_body_response_validation_or_force_load_causality=True,
        authored_inventory_complete_but_cooked_equivalence_not_proven=True,
        current_H4_floor_support_shortfall_HOLD_preserved=True,
        current_H3_final_state_not_accessed_or_changed=True,
        source_USD_or_live_stage_modified=False, real_robot_required=False, CICS_runtime_integration=False,
        new_PhysX_executed=False, actual_motion_authorized=False, use_for_final_statistics=False,
        input_sha256=pins)


def main():
    if TARGET.exists():
        report = json.loads(TARGET.read_bytes()); previous.verify(report['input_sha256'])
        print('Existing shared model-seed screen verified; not overwritten:', TARGET)
        return
    report = analyze()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with TARGET.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 NEW SHARED KINEMATIC / ORIGINAL-MODEL CONTROL-SEED SCREEN ===')
    print('union candidates:', len(report['candidate_pool']))
    for row in report['methods']:
        print(row['method'], row['status'], 'mm/deg:',
            None if not row['plan_success'] else row['predicted_terminal_position_error_m']*1000,
            row.get('predicted_terminal_heading_error_deg'))
    print('H4 floor support remains HOLD; no physical drive or final comparison admission.')
    print('saved:', TARGET)


if __name__ == '__main__':
    main()
