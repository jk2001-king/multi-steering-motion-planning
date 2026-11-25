"""Common stopped-near-reference handoff into the original goal corrector.

Arrival is NOT asserted here. A robot may stop43mm from an accepted
reference endpoint yet remain56mm from the actual goal. Let the existing
goal corrector act once within the unchanged50mm reference-end neighborhood,
only on the final leg with measured stopped speed and aligned heading.
All actual goal, hold, collision, actuation and time limits stay unchanged.
"""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_hold_damping as parent

runtime = parent.runtime
VERSION = 'COMMON1330_HOLD_DAMPING_v4_STOPPED_REFERENCE_END_HANDOFF_v5'
OUTPUT = parent.OUTPUT.parent/'stopped_reference_end_paired_v5'
PROTOCOL = OUTPUT/'protocol.json'
hold_implementation = parent.hold_implementation


def stopped_near_reference_end(motion, tracker, virtual_pose, speed, remaining):
    if not tracker.mission_final_leg:
        return False
    length = float(motion.reference.arc[-1])
    if (length <= 0 or motion.reference.progress < .95*length
            or not 0 <= speed <= .005 or not 0 <= remaining <= .050):
        return False
    actual_yaw = virtual_pose[2] - (math.pi if motion.leg['direction'] < 0 else 0.)
    end = motion.reference.poses[-1]
    return (math.hypot(virtual_pose[0]-end[0], virtual_pose[1]-end[1]) <= .050
        and abs(math.atan2(math.sin(actual_yaw-end[2]), math.cos(actual_yaw-end[2]))) <= math.radians(2))


def bind_motion(motion):
    original = motion.tracking_step
    if (getattr(original, 'actual_goal_handoff_version', None) != parent.base.goal.VERSION
            or getattr(original, 'stopped_reference_end_version', None)):
        raise ValueError('Exact once-bound actual-goal v2 predecessor required')
    old = original.__globals__['_mission_handoff']
    motion.stopped_reference_end_events = []
    def handoff(tracker, pose, speed, remaining):
        previous = old(tracker, pose, speed, remaining)
        candidate = previous or stopped_near_reference_end(motion, tracker, pose, speed, remaining)
        if candidate and not previous:
            motion.stopped_reference_end_events.append(dict(remaining_reference_m=remaining,
                measured_speed_m_s=speed, reference_endpoint=list(motion.reference.poses[-1]),
                actual_goal=list(motion.goal), pose=list(pose),
                enters_original_goal_corrector_NOT_arrival_or_hold_success=True))
        return candidate
    value = runtime.bind(original, _mission_handoff=handoff)
    value.__dict__.update(original.__dict__)
    value.stopped_reference_end_version = VERSION
    motion.tracking_step = value
    return motion


def contract(profile, *, simulation_budget_s, wall_budget_s=900.):
    value = copy.deepcopy(parent.contract(profile, simulation_budget_s=simulation_budget_s,
        wall_budget_s=wall_budget_s))
    value.update(version=VERSION, parent_controller_version=parent.VERSION,
        stopped_final_reference_handoff=dict(max_remaining_reference_m=.050,
            max_endpoint_distance_m=.050, max_endpoint_heading_deg=2.,
            measured_speed_limit_m_s=.005, minimum_total_progress_fraction=.95,
            original_goal_corrector_still_required=True),
        actual_goal_30mm_2deg_5mm_s_and_fixed5s_hold_NOT_relaxed=True,
        common_v4_hold_gains_and_all_original_limits_unchanged=True,
        no_new_goal_capture_retry_or_time_extension=True)
    return value


def controller(plan, geometry, *, profile, simulation_budget_s, start_time_s=0., wall_budget_s=900.):
    value = parent.controller(plan, geometry, profile=profile, simulation_budget_s=simulation_budget_s,
        start_time_s=start_time_s, wall_budget_s=wall_budget_s)
    bind_motion(value.motion)
    value.contract = contract(profile, simulation_budget_s=simulation_budget_s, wall_budget_s=wall_budget_s)
    return value


def private_sequence():
    value = parent.private_sequence()
    return SimpleNamespace(**dict(vars(value), unified_geometry_lifecycle_controller=controller,
        unified_geometry_lifecycle_contract=contract))


def model_binding(*args, **kwargs):
    # Same nominal plant and lifecycle factory, now with the explicit V5
    # phase connection. Captured-wheel contact PD is still NOT in that model.
    return runtime.bind(parent.base.goal.model_binding,
        private_sequence=private_sequence, VERSION=VERSION)(*args, **kwargs)


base = SimpleNamespace(**dict(vars(parent.base), model_binding=model_binding))
extra_source_paths = (Path(parent.__file__),)


def candidate():
    return SimpleNamespace(**dict(vars(parent.candidate()), VERSION=VERSION,
        private_sequence=private_sequence, contract=contract, model_binding=model_binding))


def bound_cell():
    original = runtime.bind(parent.bound_cell, candidate=candidate,
        OUTPUT=OUTPUT, PROTOCOL=PROTOCOL)()
    prior = original.__globals__['runtime']; context = prior.context_for
    private = SimpleNamespace(**dict(vars(prior), context_for=lambda folder, actor, geometry:
        context(folder, actor+'_stopped_reference_end_v5', geometry)))
    return runtime.bind(original, runtime=private)


def replay(report_path):
    """Exact old controller replay; candidate predicate observed, not applied."""
    holder = []
    def make(*args, **kwargs):
        value = parent.controller(*args, **kwargs); holder.append(value)
        return value
    def compare(old, tracker, pose, speed, remaining):
        return old(tracker, pose, speed, remaining) or stopped_near_reference_end(
            holder[0].motion, tracker, pose, speed, remaining)
    review = parent.base.goal.parent.review
    seq = SimpleNamespace(**dict(vars(parent.private_sequence()), unified_geometry_lifecycle_controller=make))
    return runtime.bind(review.replay, sequence=seq, stopped_handoff_candidate=compare)(report_path)
