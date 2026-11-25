"""Geometry-based support identification and explicit hospital contact classes."""

import math


ROBOT_ROOT = "/World/patient_transport_lab_screening"
WHEEL_NAMES = (
    "front_active_drive_wheel_link", "rear_active_drive_wheel_link",
    "front_left_caster_wheel_link", "front_right_caster_wheel_link",
    "rear_left_caster_wheel_link", "rear_right_caster_wheel_link",
)


def is_flat_floor_bound(bound, floor_top_m=.001):
    try:
        lo, hi = bound["minimum_m"], bound["maximum_m"]
        if len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in (*lo,*hi)):
            return False
        return (abs(hi[2]-floor_top_m) <= .005 and -.000001 <= hi[2]-lo[2] < .010
                and hi[0] > lo[0] and hi[1] > lo[1])
    except (KeyError, TypeError, ValueError):
        return False


def exported_floor_paths(audit):
    """Use exported collision geometry, including paths inside USD instances.

    A floor-like name alone is insufficient: 25 mm doorway thresholds must
    remain active and be treated as terrain, not silently deleted as floors.
    """
    paths = []
    for asset in audit.get("local_assets", []):
        name = asset["path"].rsplit("/",1)[-1].lower()
        if "floor" not in name or not asset["path"].startswith("/World/Hospital/"):
            continue
        for primitive in asset["collision"]["primitives"]:
            if is_flat_floor_bound(primitive["world_aligned_bound"]):
                paths.append(primitive["path"])
    return sorted(set(paths))


def classify_contact(pair, expected_supports, disabled_legacy_supports):
    if len(pair) != 2:
        return "unclassified_external"
    is_robot = [path.startswith(ROBOT_ROOT+"/") for path in pair]
    if not any(is_robot):
        return "unrelated"
    if all(is_robot):
        return "robot_internal"
    robot_path = pair[0] if is_robot[0] else pair[1]
    other_path = pair[1] if is_robot[0] else pair[0]
    wheel_contact = any(robot_path == ROBOT_ROOT+"/"+name or
                        robot_path.startswith(ROBOT_ROOT+"/"+name+"/") for name in WHEEL_NAMES)
    if other_path in expected_supports:
        return "wheel_support" if wheel_contact else "nonwheel_support"
    if other_path in disabled_legacy_supports:
        return "unexpected_legacy_support"
    if other_path.startswith("/World/Hospital/"):
        return "hospital_obstacle"
    return "unclassified_external"
