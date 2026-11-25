"""Versioned, private common-controller stopped-cusp correction.

Pure factories; importing does not install a controller or operate Isaac.
Only the interior-leg handoff predicate changes. Original braking, ACK,
alignment, final correction/hold, collision and hardware limits remain.
"""
import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace

import hospital_revision_cusp_review as review

runtime = review.runtime
sequence = runtime.sequence
VERSION = 'COMMON1330_STOPPED_INTERIOR_CUSP_RESERVE_v1'


def bind_motion(controller):
    """Change one private function global, never the historical module."""
    original = controller.tracking_step
    handoff = original.__globals__.get('_mission_handoff')
    if handoff is None or getattr(original, 'cusp_reserve_version', None):
        raise ValueError('One original unmodified mission handoff required')
    controller.stopped_cusp_handoff_events = []

    def checked_handoff(tracker, pose, speed, remaining):
        old = handoff(tracker, pose, speed, remaining)
        new = review.stopped_handoff_candidate(lambda *args: old,
            tracker, pose, speed, remaining)
        if new and not old:
            controller.stopped_cusp_handoff_events.append(dict(
                remaining_m=remaining, measured_speed_m_s=speed,
                interior_leg_index=controller.leg_index))
        return new

    candidate = runtime.bind(original, _mission_handoff=checked_handoff)
    candidate.__dict__.update(original.__dict__)
    candidate.cusp_reserve_version = VERSION
    controller.tracking_step = candidate
    return controller


def motion_class(profile):
    original = sequence.unified_geometry_motion_class(profile)

    class CommonStoppedCuspController(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            bind_motion(self)

    return CommonStoppedCuspController


def contract(profile, *, simulation_budget_s, wall_budget_s=900.):
    value = copy.deepcopy(sequence.unified_geometry_lifecycle_contract(profile,
        simulation_budget_s=simulation_budget_s, wall_budget_s=wall_budget_s))
    value.update(version=VERSION,
        parent_controller_version=value['version'],
        original_common_controller_and_limits_changed_ONLY_geometry_bindings=False,
        additional_change='interior stopped-cusp handoff reserve only',
        maximum_stopped_interior_handoff_remaining_m=review.STOPPED_HANDOFF_RESERVE_M,
        additional_handoff_measured_speed_limit_m_s=.005,
        final_leg_handoff_goal_hold_collision_and_actuation_limits_unchanged=True,
        same_policy_for_all_methods_and_tasks=True)
    return value


def controller(plan, geometry, *, profile, simulation_budget_s,
               start_time_s=0., wall_budget_s=900.):
    result = sequence.unified_geometry_lifecycle_controller(plan, geometry,
        profile=profile, simulation_budget_s=simulation_budget_s,
        start_time_s=start_time_s, wall_budget_s=wall_budget_s)
    bind_motion(result.motion)
    result.contract = contract(profile, simulation_budget_s=simulation_budget_s,
        wall_budget_s=wall_budget_s)
    return result


def private_sequence():
    return SimpleNamespace(**dict(vars(sequence),
        unified_geometry_motion_class=motion_class,
        unified_geometry_lifecycle_controller=controller,
        unified_geometry_lifecycle_contract=contract))


def model_binding(profile, geometry, plan, *, simulation_budget_s):
    """Use the exact existing plant and guard, substituting its one factory."""
    original = review.followup.planning.shared.unified_mission_model_binding
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    count = 0
    for index, node in enumerate(tree.body[0].body):
        if (isinstance(node, ast.Import) and len(node.names) == 1
                and node.names[0].name == 'hospital_mission_sequence'
                and node.names[0].asname == 'sequence'):
            tree.body[0].body[index] = ast.Assign(
                [ast.Name('sequence', ast.Store())], ast.Name('_candidate_sequence', ast.Load()))
            count += 1
    if count != 1:
        raise ValueError('Exactly one original model controller import required')
    namespace = dict(original.__globals__, _candidate_sequence=private_sequence())
    exec(compile(ast.fix_missing_locations(tree), '<private_stopped_cusp_model_factory>', 'exec'), namespace)
    run, execution, design = namespace[original.__name__](profile, geometry, plan,
        simulation_budget_s=simulation_budget_s)

    def measured_run(*args, **kwargs):
        report, rows = run(*args, **kwargs)
        report.update(controller_candidate_version=VERSION,
            original_common_controller_except_geometry_unchanged=False,
            common_stopped_interior_handoff_reserve_m=review.STOPPED_HANDOFF_RESERVE_M,
            final_goal_hold_actual_guards_and_limits_unchanged=True)
        return report, rows

    return measured_run, execution, design
