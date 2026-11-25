"""Offline, uniform all-task compatibility screen BEFORE costly reruns.

Reuse the30 original P initial conditions with one frozen owned-seed/beam/
native-refinement rule. These already-seen development conditions are NOT
new final repetitions and contain no Isaac or timeline operation. This
screen checks whether a locally successful search fix generalizes at all.
"""
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_beam_goal_rank as search

runtime = search.runtime
followup = search.parent.followup
planning = followup.planning
OUTPUT = followup.OUTPUT/'uniform_v3_P_compatibility_screen_v1'
PROTOCOL = OUTPUT/'protocol.json'


def export_protocol():
    original = runtime.read(followup.campaign.PROTOCOL)
    rule = runtime.read(search.PROTOCOL)
    value = dict(version='UNIFORM30_SEEN_CONDITION_NATIVE_P_SEARCH_COMPATIBILITY_v1',
        main_cells=[c for c in original['main_cells'] if c['method'] == followup.campaign.planning.P],
        tasks=original['tasks'], common_speed_policy=original['common_speed_policy'],
        original_evidence_inputs_sha256=original['original_evidence_inputs_sha256'],
        P_seed_portfolio=rule['P_seed_portfolio'], seed_total_budget_s=45.,
        planning_total_budget_s=180., lattice_expansion_limit_per_search_call=60000,
        uniform_P_lookahead_edges_all_tasks=4, beam_width=64, local_iterations=24,
        final_native_refinement=rule['final_strict_failure_fallback'],
        final_beam_goal_ranking=rule['sole_additional_change'],
        history_policy='carry', map_speed_geometry_goal_and_constraints_unchanged=True,
        previously_seen_initial_conditions_NOT_held_out=True,
        one_planning_attempt_per_declared_condition=True,
        original_all_method_n5_records_never_overwritten=True,
        final_statistics=False, actual_motion_trials=0,
        concurrent_live_work_possible_wall_times_NOT_a_matched_planning_time_benchmark=True,
        conditional_future_actual_reruns_are_NOT_authorized_by_this_file=True)
    if len(value['main_cells']) != 30:
        raise ValueError('Exactly original six tasks times five P conditions required')
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Uniform compatibility definition already frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        files = (Path(__file__), Path(search.__file__), Path(search.terminal.__file__),
            Path(followup.__file__), Path(planning.__file__), Path(search.parent.__file__))
        pins = {str(p.resolve()): runtime.sha(p) for p in files}
        runtime.evidence.write_exclusive(OUTPUT/'sources.json', pins)
        runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip', pins)
    return PROTOCOL


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'sources.json'))


def run_cell(name):
    verify_sources()
    scoped = SimpleNamespace(**dict(vars(followup.campaign), PROTOCOL=PROTOCOL, OUTPUT=OUTPUT))
    seed = runtime.bind(followup.seed_portfolio, PROTOCOL=PROTOCOL)
    shared = SimpleNamespace(**dict(vars(planning.shared), unified_P_owned_seed=seed))
    call = runtime.bind(planning.run_cell, campaign=scoped, shared=shared,
        OUTPUT=OUTPUT/'planning', planner_bindings=followup.planner_bindings('carry'),
        guided_binding=search.guided_binding, __file__=__file__)
    return call(name, lookahead_edges=4)


def run():
    export_protocol(); verify_sources()
    if (OUTPUT/'reserved.json').exists():
        raise ValueError('Consumed finite offline screen; no automatic repeated attempts')
    runtime.evidence.write_exclusive(OUTPUT/'reserved.json', dict(actual_motion_trials=0,
        independent_new_repetitions=0, maximum_offline_planning_attempts=30))
    results = []
    for cell in runtime.read(PROTOCOL)['main_cells']:
        name = cell['cell_id']; verify_sources()
        value = run_cell(name)
        results.append(value)
        print('[UNIFORM P COMPATIBILITY]', len(results), '/30', name,
            value['plan_success'], value['status'], flush=True)
    value = dict(status='UNIFORM30_OFFLINE_COMPATIBILITY_SCREEN_FINISHED',
        rows=results, planned_n=sum(r['plan_success'] for r in results), declared_n=30,
        actual_motion_trials=0, not_independent_final_repetitions=True,
        all_rows_and_old_failures_preserved=True)
    runtime.evidence.write_exclusive(OUTPUT/'completed.json', value)
    return value


if __name__ == '__main__':
    run()
