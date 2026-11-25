#!/usr/bin/env python3
"""Audit the existing H3 door/threshold WITHOUT opening or removing anything.

Authored triangle-mesh cross sections are not cooked PhysX geometry. A failed
constant-Y, zero-yaw necessary condition does not rule out a turning path.
No H3 task poses, terrain permission, planning or driving are created here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from hospital_geometry_tools import combine_material_meshes, line_free_intervals


WORKSPACE = Path(__file__).resolve().parents[3]
GATE = WORKSPACE / "studies/IJCAS/results/hospital_paper_gate"
ASSETS = {
    "wall": "/World/Hospital/Geo_M1_DoorWall11",
    "frame": "/World/Hospital/Geo_M_DoorFrame43",
    "threshold": "/World/Hospital/Geo_M_DoorFloor20",
    "leaf": "/World/Hospital/SM_Door_02b9",
}
HEIGHTS = (.100, .250, .426)
SUPPORT_TOP_M = .001
FOOTPRINT = {"length_m": 1.630, "width_m": .965, "clearance_m": .010}


def intersect_intervals(first, second):
    result = []
    for a, b in first:
        for c, d in second:
            lo, hi = max(a, c), min(b, d)
            if hi >= lo:
                result.append([lo, hi])
    return result


def erode_intervals(intervals, half_width):
    return [[lo+half_width, hi-half_width] for lo, hi in intervals
            if hi-lo >= 2*half_width]


def _asset(audit, path):
    matches = [row for row in audit["local_assets"] if row["path"] == path]
    if len(matches) != 1:
        raise ValueError("Missing/duplicate audited H3 asset: "+path)
    row = matches[0]
    bound = row["world_aligned_visible_bound"]
    if (not all(len(bound[key]) == 3 and all(math.isfinite(v) for v in bound[key])
            for key in ("minimum_m", "maximum_m", "center_m", "size_m"))
            or any(b <= a for a, b in zip(bound["minimum_m"], bound["maximum_m"]))):
        raise ValueError("Invalid H3 asset bound: "+path)
    primitives = row["collision"]["primitives"]
    if not primitives or row["collision"]["collision_prim_count"] != len(primitives):
        raise ValueError("Missing H3 collision evidence: "+path)
    for primitive in primitives:
        mesh = primitive.get("authored_mesh")
        if (mesh is None or mesh.get("physics_approximation") != "none"
                or not mesh.get("face_vertex_counts") or any(n != 3 for n in mesh["face_vertex_counts"])
                or len(mesh["face_vertex_indices"]) != 3*len(mesh["face_vertex_counts"])
                or not mesh.get("world_points_m")
                or not all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in mesh["world_points_m"])
                or any(not isinstance(i, int) or i < 0 or i >= len(mesh["world_points_m"])
                    for i in mesh["face_vertex_indices"])):
            raise ValueError("Need finite authored triangle meshes with approximation=none: "+path)
    return row


def analyze_doorway(audit):
    if (audit.get("schema_version") != 2 or audit.get("candidate_export_complete") is not True
            or audit.get("coordinate_units") != "metres_converted_from_stage_units"
            or audit.get("up_axis") != "Z"):
        raise ValueError("Need a complete metre-converted Z-up v02 audit")
    assets = {key: _asset(audit, path) for key, path in ASSETS.items()}
    meshes = {key: combine_material_meshes([p["authored_mesh"] for p in row["collision"]["primitives"]])
              for key, row in assets.items() if key != "threshold"}
    wall, frame = (assets[key]["world_aligned_visible_bound"] for key in ("wall", "frame"))
    if wall["size_m"][0] >= wall["size_m"][1] or wall["size_m"][0] > .3:
        raise ValueError("This diagnostic requires the audited thin world-X wall")
    x = wall["center_m"][0]
    y_min, y_max = wall["minimum_m"][1], wall["maximum_m"][1]
    frame_y = frame["center_m"][1]
    half_width = FOOTPRINT["width_m"]/2+FOOTPRINT["clearance_m"]
    leaf = assets["leaf"]["world_aligned_visible_bound"]
    # Cross the CURRENT leaf as well as the wall. A constant-Y crossing has to
    # clear every one of these slices; no alternative door state is invented.
    leaf_x = [leaf["minimum_m"][0]+fraction*leaf["size_m"][0] for fraction in (.25, .5, .75)]
    sections, errors = [], []
    straight_candidates = None
    for height in HEIGHTS:
        try:
            structural = line_free_intervals([meshes["wall"], meshes["frame"]], x, y_min, y_max, height)
            opening = [interval for interval in structural if interval[0] <= frame_y <= interval[1]]
            if len(opening) != 1:
                raise ValueError("No unique structural opening containing the frame center")
            allowed = erode_intervals(opening, half_width)
            probes = []
            for probe_x in sorted(set([x, *leaf_x])):
                free = line_free_intervals(list(meshes.values()), probe_x, y_min, y_max, height)
                centers = erode_intervals(free, half_width)
                allowed = intersect_intervals(allowed, centers)
                probes.append({"x_m": probe_x, "free_y_intervals_m": free,
                    "robot_center_y_intervals_m": centers})
            straight_candidates = allowed if straight_candidates is None else intersect_intervals(straight_candidates, allowed)
            sections.append({"height_m": height, "wall_x_m": x,
                "structural_opening_y_m": opening[0], "structural_width_m": opening[0][1]-opening[0][0],
                "wall_frame_leaf_probes": probes, "constant_Y_zero_yaw_center_intervals_m": allowed})
        except ValueError as error:
            errors.append({"height_m": height, "error": str(error)})
    # Collision bounds, not the visible floor image, determine the authored step.
    threshold = assets["threshold"]["collision"]["bound"]
    step = threshold["maximum_m"][2]-SUPPORT_TOP_M
    return {
        "schema_version": 1, "scenario": "H3_doorway_immediate_turn",
        "status": "HOLD_H3_geometry_and_terrain_readiness_NOT_execution_failure",
        "analysis_complete": not errors,
        "source_stage": audit["stage_root_layer"], "door_state": "existing_authored_pose_unchanged",
        "assets": {key: {"path": row["path"], "visible_bound_m": row["world_aligned_visible_bound"],
            "collision_bound_m": row["collision"]["bound"],
            "collision_prim_paths": row["collision"]["collision_prim_paths"],
            "computed_visibility_in_export": row["computed_visibility"]} for key, row in assets.items()},
        "footprint": dict(FOOTPRINT), "required_width_with_clearance_m": 2*half_width,
        "sampled_height_sections": sections, "section_errors": errors,
        "straight_constant_Y_zero_yaw_necessary_condition": {
            "result": "UNKNOWN" if errors else "NOT_RULED_OUT_NOT_PROVEN" if straight_candidates else "BLOCKED_AT_SAMPLED_SLICES",
            "possible_center_y_intervals_m": [] if errors else straight_candidates,
            "curved_or_rotating_path_feasibility": "NOT_TESTED_do_not_generalize_straight_path_blockage",
            "door_leaf_included": True, "threshold_traversal_included": False,
        },
        "threshold": {"path": ASSETS["threshold"], "authored_collision_top_z_m": threshold["maximum_m"][2],
            "canonical_support_top_z_m": SUPPORT_TOP_M, "authored_step_height_m": step,
            "support_height_scope": "existing_H1_H2_canonical_support_setting_not_new_H3_PhysX_measurement",
            "wheel_terrain_validation": "PENDING", "collision_disabled": False},
        "source_stage_modified": False, "robot_moved": False, "door_moved": False,
        "collision_or_terrain_policy_changed": False, "H1_H2_raster_modified": False,
        "start_goal_poses_created": False, "path_planning_executed": False,
        "PhysX_or_Nav2_executed": False, "cooked_collision_verified": False,
        "use_for_final_statistics": False,
        "next_gate": "review_visible_geometry_overlay_then_select_explicit_door_state_and_validate_threshold_before_planning",
        "limitations": ["authored_mesh_cross_sections_not_cooked_PhysX_proof",
            "sampled_lines_are_necessary_checks_not_a_full_swept_envelope_certificate",
            "mobile_base_width_only_loaded_upper_envelope_pending",
            "do_not_remove_leaf_or_threshold_to_silently_make_a_route_feasible"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, default=GATE / "hospital_asset_geometry_audit_v02.json")
    args = parser.parse_args()
    raw = args.source_audit.read_bytes()
    manifest_path = GATE / "hospital_h1_geometry.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if hashlib.sha256(raw).hexdigest() != manifest["source_sha256"]:
        raise ValueError("The frozen H1 hospital audit changed; do not replace existing evidence")
    report = analyze_doorway(json.loads(raw))
    report["input_sha256"] = {str(args.source_audit.resolve()): hashlib.sha256(raw).hexdigest(),
        str(manifest_path): hashlib.sha256(manifest_raw).hexdigest()}
    report["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    folder = GATE / "hospital_h3_doorway_geometry"
    folder.mkdir(parents=True, exist_ok=True)
    number = 1
    while (folder / f"audit_r{number:02d}").exists():
        number += 1
    output = folder / f"audit_r{number:02d}"
    output.mkdir()
    target = output / "doorway_geometry_audit.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("=== H3 EXISTING DOORWAY GEOMETRY AUDIT ===")
    for section in report["sampled_height_sections"]:
        print("height / structural width mm:", 1000*section["height_m"], round(1000*section["structural_width_m"], 3))
    print("required robot width + clearance mm:", 1000*report["required_width_with_clearance_m"])
    print("current leaf / zero-yaw constant-Y crossing:", report["straight_constant_Y_zero_yaw_necessary_condition"]["result"])
    print("curved-path feasibility: NOT TESTED")
    print("authored step above canonical support mm:", round(1000*report["threshold"]["authored_step_height_m"], 3))
    print("threshold terrain validation: PENDING; no collision disabled")
    print("STATUS:", report["status"])
    print("saved:", target)
    print("No robot, door, physics, map, H1/H2 result or source USD changed.")


if __name__ == "__main__":
    main()
