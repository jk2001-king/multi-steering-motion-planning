"""Policy-independent general-navigation scenarios for the IJCAS study.

These are simulation screening geometries, not replicas of the KTL docking
sequence.  Each scenario represents a recurring navigation pattern for a long
mobile platform.  Obstacles are closed axis-aligned rectangles
``(min_x, max_x, min_y, max_y)`` in metres in the initial base frame.
"""

import math


GENERAL_NAVIGATION_SCENARIOS = {
    "corridor_90_turn": {
        "goal": (1.50, 2.20, math.radians(90.0)),
        "obstacles": (
            # Inside corner of an L-shaped passage.
            (-2.0, -0.20, 1.05, 3.5),
            (2.30, 3.5, -1.00, 3.5),
        ),
        "bounds_x": (-0.8, 2.8),
        "description": "Traverse a straight corridor and immediately turn 90 degrees.",
    },
    "s_corridor": {
        "goal": (5.80, 0.0, 0.0),
        "obstacles": (
            # The original first offset began too close to the initial pose
            # for the 1.63 m body to establish the required lateral offset.
            # These approach and inter-obstacle distances are scaled to the
            # full body length and the 0.70 1/m maximum curvature.
            (2.00, 2.35, 0.25, 2.2),
            (4.20, 4.55, -2.2, -0.25),
        ),
        "bounds_x": (-0.8, 6.4),
        "description": "Negotiate two opposite offsets and return to the initial heading.",
    },
    "narrow_t_intersection": {
        "goal": (1.10, 2.15, math.radians(90.0)),
        "obstacles": (
            (-2.0, 0.05, 1.00, 3.5),
            (2.15, 3.5, 1.00, 3.5),
        ),
        "bounds_x": (-0.8, 2.6),
        "description": "Turn into the side branch of a narrow T intersection.",
    },
    "doorway_immediate_turn": {
        "goal": (2.00, 1.30, math.radians(65.0)),
        "obstacles": (
            (0.70, 0.82, -2.0, -0.80),
            (0.70, 0.82, 0.80, 2.0),
            # Keep the terminal pose outside the wall by more than the
            # full-envelope 10 mm planning clearance.  With the previous
            # x=2.60 m bound, the nominal 65 deg goal pose itself collided
            # with this wall, so no policy could produce a valid plan.
            (2.90, 3.5, -0.10, 3.0),
        ),
        "bounds_x": (-0.8, 3.0),
        "description": "Pass a doorway and begin a turn before the long body fully clears it.",
    },
    "obstacle_bypass_return": {
        "goal": (4.50, 0.0, 0.0),
        "obstacles": (
            # Reserve enough approach distance for the long front overhang to
            # clear the obstacle while the body develops a lateral offset.
            (2.30, 2.70, -0.25, 0.25),
        ),
        "bounds_x": (-0.8, 5.1),
        "description": "Bypass a centred obstacle and return to the original path and heading.",
    },
}


SCENARIO_GEOMETRY_STATUS = "provisional_general_navigation_screening"

