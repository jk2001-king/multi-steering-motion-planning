"""Prospective offline goal-feasibility ranking, using the shared refiner.

Weighted pose error can prefer an almost exact position outside the yaw
tolerance to a jointly admissible pose. Rank the largest normalized goal
residual first, with a small normalized sum to resolve ties. This changes
search ranking ONLY, not propagation, any predicate, speed or actual control.
"""
import ast
import hashlib
import inspect
import linecache
from pathlib import Path
import textwrap
from types import SimpleNamespace

import hospital_revision_followup as followup

runtime = followup.runtime
planning = followup.planning
OUTPUT = followup.OUTPUT / 'strict_goal_feasibility_rank_v1'
PROTOCOL = OUTPUT / 'protocol.json'
CELL_ID = 'H3_P_dynamics_full_envelope_s01'
SCORE = ('max(position_error/goal_position_tolerance_m, '
    'heading_error/goal_heading_tolerance_rad, terminal_speed/terminal_speed_tolerance_m_s) '
    '+ .001*(position_error/goal_position_tolerance_m '
    '+ heading_error/goal_heading_tolerance_rad + terminal_speed/terminal_speed_tolerance_m_s)')


def ranked_refiner(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'score']
    expected = ast.parse('100.0 * position_error + heading_objective_weight * heading_error', mode='eval').body
    if len(matches) != 1 or ast.dump(matches[0].value) != ast.dump(expected):
        raise ValueError('Exactly one original local pose-ranking assignment required')
    before = matches[0].value
    matches[0].value = ast.parse(SCORE, mode='eval').body
    normalized = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    matches[0].value = before
    if ast.dump(tree) != ast.dump(ast.parse(source)):
        raise ValueError('Undo must recover every original evaluator/acceptance statement')
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    filename = '<strict_goal_feasibility_' + digest + '>'
    # Standard inspect support for the existing mode/curvature wrapper. No
    # original source file, loaded module or simulator state is modified.
    linecache.cache[filename] = (len(normalized), None, normalized.splitlines(True), filename)
    namespace = dict(original.__globals__)
    exec(compile(normalized, filename, 'exec'), namespace)
    result = namespace[original.__name__]
    result.__dict__.update(original.__dict__)
    result.goal_feasibility_rank = dict(expression=SCORE,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_source_sha256=digest, all_original_nonranking_AST_unchanged=True,
        original_strict_goal_and_every_edge_predicate_unchanged=True)
    return result


def guided_binding(*, lookahead_edges=4):
    original = planning.mission.mode_curvature_refinement_binding

    def factory(refine, **kwargs):
        ranked = ranked_refiner(refine)
        result = original(ranked, **kwargs)
        result.mode_refinement_binding['goal_feasibility_rank'] = ranked.goal_feasibility_rank
        return result

    refine = runtime.bind(planning.mission.refine_with_NE_guided_beam,
        mode_curvature_refinement_binding=factory)
    mission = SimpleNamespace(**dict(vars(planning.mission), refine_with_NE_guided_beam=refine))
    return runtime.bind(planning.guided_binding, mission=mission)(lookahead_edges=lookahead_edges)


def export_protocol():
    parent = runtime.read(followup.PROTOCOL)
    main = runtime.read(followup.campaign.PROTOCOL)
    value = dict(version='P_STRICT_GOAL_NORMALIZED_LOCAL_RANK_v1',
        main_cells=[c for c in main['main_cells'] if c['cell_id'] == CELL_ID],
        tasks=main['tasks'], common_speed_policy=main['common_speed_policy'],
        original_evidence_inputs_sha256=main['original_evidence_inputs_sha256'],
        P_seed_portfolio=parent['P_seed_portfolio'], seed_total_budget_s=45.,
        uniform_P_lookahead_edges_all_tasks=4, beam_width=64, local_iterations=24,
        planning_total_budget_s=180., lattice_expansion_limit=60000,
        sole_additional_algorithm_change=dict(local_goal_ranking=SCORE),
        all_edge_model_map_resource_cusp_stop_and_final_predicates_unchanged=True,
        not_an_objective_optimality_or_physical_success_claim=True,
        known_condition_development_NOT_independent_repetition=True,
        old_failed_attempts_preserved=True, physical_trials=0, motion_authorized=False)
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value: raise ValueError('Prospective goal-ranking protocol frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        files = (Path(__file__), Path(followup.__file__), Path(planning.__file__))
        runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip',
            {str(p.resolve()): runtime.sha(p) for p in files})
    return PROTOCOL


def run():
    export_protocol()
    campaign = SimpleNamespace(**dict(vars(followup.campaign), PROTOCOL=PROTOCOL, OUTPUT=OUTPUT))
    seed = runtime.bind(followup.seed_portfolio, PROTOCOL=PROTOCOL)
    shared = SimpleNamespace(**dict(vars(planning.shared), unified_P_owned_seed=seed))
    call = runtime.bind(planning.run_cell, campaign=campaign, shared=shared,
        OUTPUT=OUTPUT / 'planning/carry', planner_bindings=followup.planner_bindings('carry'),
        guided_binding=guided_binding, __file__=__file__)
    return call(CELL_ID, lookahead_edges=4)


if __name__ == '__main__':
    run()
