import math

import pytest

from patient_initial_stability.steering_lattice import (
    FULL_DYNAMICS,
    KINEMATIC_DYNAMICS,
    State,
    plan,
    propagate,
)


@pytest.mark.parametrize(
    "goal,bounds,expected_mode",
    [
        ((0.6451846534, 0.3411371489, math.radians(20.0)), (-0.80, 1.30), "front_only"),
        ((0.7229260505, -0.1000785532, math.radians(20.0)), (-1.00, 0.80), "rear_only"),
    ],
)
def test_online_planner_solves_both_end_constraints(goal, bounds, expected_mode):
    result = plan(goal, bounds[0], bounds[1], robust_clearance=0.010)
    assert result is not None
    assert len(result.actions) == 14
    assert {action[0] for action in result.actions} == {expected_mode}
    final = result.states[-1]
    assert math.hypot(final.x - goal[0], final.y - goal[1]) <= 0.025
    assert abs(math.degrees(final.yaw - goal[2])) <= 2.0


def test_legacy_frame_width_corridor_is_rejected_by_full_envelope():
    """The old 0.760 m frame-only gate must not return as collision evidence."""
    goal = (0.6451846534, 0.3411371489, math.radians(20.0))
    assert plan(goal, -0.40, 1.05, robust_clearance=0.010) is None


def test_dynamics_ablation_changes_steering_transition_propagation():
    state = State(0.0, 0.0, 0.0, "coordinated_counterphase", 0.0, 0.0, True, 1)
    kinematic, _, _ = propagate(
        state, "front_only", 0.70, 1, dynamics=KINEMATIC_DYNAMICS
    )
    dynamic, _, _ = propagate(
        state, "front_only", 0.70, 1, dynamics=FULL_DYNAMICS
    )
    assert kinematic.front > dynamic.front
    assert kinematic.rear == pytest.approx(0.0)
    assert dynamic.yaw != pytest.approx(kinematic.yaw)
