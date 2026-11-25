"""Strict-goal ranking of the existing final NE beam, offline only.

The previous final selection minimized a weighted pose score, which need
not prefer a jointly admissible position/yaw endpoint. Rank the SAME valid
beam states by their original normalized goal conditions instead. No new
primitive, state, curvature, goal, or acceptance predicate is introduced.
"""
import ast
import inspect
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace

import hospital_revision_goal_feasibility as parent
import hospital_revision_terminal_refinement as terminal

runtime = parent.runtime
OUTPUT = parent.followup.OUTPUT / 'final_beam_strict_goal_rank_v3_distributed_pairs'
PROTOCOL = OUTPUT / 'protocol.json'


def strict_goal_rank(entry, goal, position_tolerance, heading_tolerance, speed_tolerance):
    state = entry[0]
    ep = math.hypot(state.x-goal[0], state.y-goal[1])/position_tolerance
    ey = abs(math.atan2(math.sin(state.yaw-goal[2]), math.cos(state.yaw-goal[2])))/heading_tolerance
    speed = math.hypot(state.longitudinal_velocity, state.lateral_velocity)/speed_tolerance
    return (max(ep, ey, speed) > 1., max(ep, ey, speed), ep+ey+speed,
        100*ep*position_tolerance+60*ey*heading_tolerance, entry[2], entry[3])


def ranked_beam(*, position_tolerance=.025, heading_tolerance=math.radians(2)):
    if position_tolerance <= 0 or heading_tolerance <= 0:
        raise ValueError('Original positive goal tolerances required')
    original = parent.planning.mission.recover_NE_guided_beam
    source = textwrap.dedent(inspect.getsource(original)); tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and
        len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'selected']
    if len(nodes) != 1 or not isinstance(nodes[0].value, ast.Call):
        raise ValueError('Exactly the original one final beam selection required')
    node = nodes[0]; before = node.value
    node.value = ast.parse('min(beam, key=lambda e: _strict_goal_rank(e, goal, '
        '_position_tolerance, _heading_tolerance, terminal_speed_m_s))', mode='eval').body
    namespace = dict(original.__globals__, _strict_goal_rank=strict_goal_rank,
        _position_tolerance=position_tolerance, _heading_tolerance=heading_tolerance)
    exec(compile(ast.fix_missing_locations(tree), '<final_beam_original_strict_goal_ranking>', 'exec'), namespace)
    value = namespace[original.__name__]
    node.value = before
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(source), include_attributes=False):
        raise ValueError('Undo must recover every original expansion/propagation/geometry predicate')
    value.final_goal_binding = dict(only_final_selection_rank_changed=True,
        every_original_expansion_and_acceptance_AST_recovered_on_undo=True,
        position_tolerance_m=position_tolerance, heading_tolerance_rad=heading_tolerance,
        same_native_modes_curvatures_speeds_resources_and_full_envelope=True)
    return value


def distributed_native_pairs():
    """Same finite pair checker; expose early controls as well as the tail."""
    original = terminal.bounded_pairs
    source = textwrap.dedent(inspect.getsource(original)); tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name) and n.target.id == 'index']
    if len(nodes) != 1:
        raise ValueError('Exactly one original single-coordinate enumeration required')
    before = nodes[0].iter
    nodes[0].iter = ast.parse('range(len(actions)) if len(actions)<=96 else '
        'sorted(set(list(range(0,len(actions),max(1,len(actions)//16))) + '
        'list(range(max(0,len(actions)-12),len(actions)))))', mode='eval').body
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<distributed_original_native_pairs>', 'exec'), namespace)
    value = namespace[original.__name__]
    nodes[0].iter = before
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(source), include_attributes=False):
        raise ValueError('Undo must recover original evaluator, pair shortlist and guards')
    return value


def guided_binding(*, lookahead_edges=4):
    mission = parent.planning.mission
    beam = ranked_beam()
    original = mission.refine_with_NE_guided_beam
    source = textwrap.dedent(inspect.getsource(original)); tree = ast.parse(source)
    nodes = [n for n in tree.body[0].body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'ns' for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError('Original bound-argument boundary required')
    index = tree.body[0].body.index(nodes[0])
    guard = ast.parse("if abs(v['goal_position_tolerance_m']-.025)>1e-12 or "
        "abs(v['goal_heading_tolerance_rad']-math.radians(2))>1e-12:\n "
        "raise ValueError('Final beam ranking must match ORIGINAL strict goal conditions')").body[0]
    tree.body[0].body.insert(index, guard)
    namespace = dict(original.__globals__, recover_NE_guided_beam=beam)
    exec(compile(ast.fix_missing_locations(tree), '<checked_final_beam_goal_ranking>', 'exec'), namespace)
    checked = namespace[original.__name__]
    tree.body[0].body.remove(guard)
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(source), include_attributes=False):
        raise ValueError('Only a stricter parameter-consistency guard may be added')
    original_mode = mission.mode_curvature_refinement_binding
    def paired_mode(refine, **kwargs):
        factory = runtime.bind(terminal.paired_binding, bounded_pairs=distributed_native_pairs())
        return factory(original_mode(refine, **kwargs),
            window=12, shortlist=24, goal_aware_ranking=True)
    private_mission = SimpleNamespace(**dict(vars(mission),
        refine_with_NE_guided_beam=checked, recover_NE_guided_beam=beam,
        mode_curvature_refinement_binding=paired_mode))
    planning = SimpleNamespace(**dict(vars(parent.planning), mission=private_mission))
    return runtime.bind(parent.guided_binding, planning=planning)(lookahead_edges=lookahead_edges)


def export_protocol():
    value = dict(runtime.read(parent.PROTOCOL), version='FINAL_NE_BEAM_STRICT_GOAL_RANK_DISTRIBUTED_PAIRS_v3',
        sole_additional_change=ranked_beam().final_goal_binding,
        final_strict_failure_fallback='original native single/pair checker; all<=96 controls, distributed16+last12 otherwise; shortlist24',
        same_actual_goal_and_strict_acceptance=True,
        continuous_curvature_polishing_NOT_used=True,
        native_discrete_actions_only=True, actual_trials=0, motion_authorized=False)
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Prospective final-beam selection policy frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        files = (Path(__file__), Path(parent.__file__), Path(terminal.__file__), Path(parent.planning.__file__),
            Path(parent.planning.mission.__file__))
        runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',
            {str(p.resolve()): runtime.sha(p) for p in files})
    return PROTOCOL


def run():
    return runtime.bind(parent.run, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
        export_protocol=export_protocol, guided_binding=guided_binding, __file__=__file__)()


if __name__ == '__main__':
    run()
