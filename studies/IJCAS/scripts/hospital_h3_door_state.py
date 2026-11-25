"""Pure H3 fully-open door geometry, NOT an Isaac or traversal result.

Use the user-reported hinge review to fail closed on a different asset/pose.
Keep the hinge translation/Z and rotate its local negative-X leaf into west.
This helper never edits files, USD stages, collision flags or existing maps.
"""

import builtins
import copy
import math
from pathlib import Path
import runpy
from types import ModuleType

import numpy as np


def _load_frozen_doorway_audit():
    """Load both frozen project dependencies without a path/module-cache edit.

    Isaac's import hooks may not discover a just-added sys.path directory.
    The historical audit must retain its original bytes/hash, so give ONLY
    that module a private import binding for its one project dependency.
    Neither global builtins nor sys.path/sys.modules are changed.
    """
    here = Path(__file__).resolve().parent
    geometry = ModuleType("_ijcas_h3_frozen_geometry")
    geometry.__dict__.update(runpy.run_path(str(here / "hospital_geometry_tools.py")))
    standard_import = builtins.__import__

    def dependency_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "hospital_geometry_tools":
            return geometry
        return standard_import(name, globals, locals, fromlist, level)

    path = here / "audit_hospital_h3_doorway.py"
    namespace = {"__name__": "_ijcas_h3_frozen_doorway_audit", "__file__": str(path),
        "__package__": "", "__builtins__": {**vars(builtins), "__import__": dependency_import}}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


_frozen_audit = _load_frozen_doorway_audit()
ASSETS, analyze_doorway = _frozen_audit["ASSETS"], _frozen_audit["analyze_doorway"]


CONDITION = "fully_open_baseline_session_only"
OBSERVED_ORIGIN_M = (17.218881835937502, 15.177172241210938, 1.127729278564453)
OBSERVED_AXES = ((.5000055210271341, -.8660222161944716, 0.),
    (.8660222161944716, .5000055210271341, 0.), (0., 0., 1.))
OBSERVED_LOCAL_MIN = (-1.232165881272424, -.11701817250956381, -1.0909781402875893)
OBSERVED_LOCAL_MAX = (2.400173926275784e-7, .04765299690545177, 1.090978140287589)


def opening_specification(origin, axes, local_min, local_max):
    arrays = [np.asarray(value, dtype=float) for value in (origin, axes, local_min, local_max)]
    origin, axes, lo, hi = arrays
    if (origin.shape != (3,) or axes.shape != (3,3) or lo.shape != (3,) or hi.shape != (3,)
            or not all(np.isfinite(value).all() for value in arrays)):
        raise ValueError("Non-finite or incomplete H3 hinge review")
    if (not np.allclose(origin, OBSERVED_ORIGIN_M, rtol=0, atol=2e-5)
            or not np.allclose(axes, OBSERVED_AXES, rtol=0, atol=2e-5)
            or not np.allclose(lo, OBSERVED_LOCAL_MIN, rtol=0, atol=2e-5)
            or not np.allclose(hi, OBSERVED_LOCAL_MAX, rtol=0, atol=2e-5)):
        raise ValueError("The live hinge/door differs from the user-reviewed partial-open state")
    if (not np.allclose(axes @ axes.T, np.eye(3), rtol=0, atol=1e-5)
            or np.linalg.det(axes) < .9999 or not np.allclose(axes[2], [0,0,1], atol=1e-6)
            or abs(hi[0]) > .01 or lo[0] >= -1. or hi[1]-lo[1] >= .3):
        raise ValueError("Need a unit-scale upright leaf with its hinge at local X=0")
    old_yaw = math.atan2(axes[0,1], axes[0,0])
    delta = -old_yaw
    c, s = math.cos(delta), math.sin(delta)
    rotation = np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
    if not np.allclose(axes @ rotation.T, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("The desired pure-Z rotation would not yield the reviewed fully-open axes")
    return {"condition": CONDITION, "hinge_world_origin_m": origin.tolist(),
        "original_root_yaw_deg": math.degrees(old_yaw), "additional_world_Z_rotation_deg": math.degrees(delta),
        "desired_root_world_axes": np.eye(3).tolist(),
        "leaf_extension_direction": "negative_world_X_into_existing_room",
        "definition": "leaf_long_axis_perpendicular_to_world_Y_door_wall_not_a_90deg_delta",
        "root_translation_and_Z_must_remain_fixed": True, "rotation_matrix": rotation.tolist()}


def rotate_world_points(points, specification):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite nonempty 3-D door vertices")
    origin = np.asarray(specification["hinge_world_origin_m"])
    rotation = np.asarray(specification["rotation_matrix"])
    return ((values-origin) @ rotation.T+origin).tolist()


def points_bound(points):
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite points to bound")
    lo, hi = values.min(axis=0), values.max(axis=0)
    return {"minimum_m": lo.tolist(), "maximum_m": hi.tolist(), "size_m": (hi-lo).tolist(),
        "center_m": ((hi+lo)/2).tolist()}


def predicted_variant(source, specification):
    """Rigidly transform vertices, NEVER a rotated old world-AABB.

    The measured original world bound is looser than the world vertex bound.
    This MODEL-ONLY projection is for checking live export after rotation.
    """
    variant = copy.deepcopy(source)
    matches = [r for r in variant["local_assets"] if r["path"] == ASSETS["leaf"]]
    if len(matches) != 1:
        raise ValueError("Missing/duplicate source leaf")
    leaf = matches[0]
    all_points = []
    for primitive in leaf["collision"]["primitives"]:
        mesh = primitive["authored_mesh"]
        mesh["world_points_m"] = rotate_world_points(mesh["world_points_m"], specification)
        primitive["world_aligned_bound"] = points_bound(mesh["world_points_m"])
        all_points.extend(mesh["world_points_m"])
    leaf["collision"]["bound"] = points_bound(all_points)
    leaf["world_aligned_visible_bound"] = points_bound(all_points)
    variant["geometry_variant"] = {"condition": CONDITION,
        "scope": "MODEL_ONLY_vertex_rotation_check_not_live_stage_or_PhysX_execution"}
    return variant


def export_mismatches(source, actual, specification, tolerance=2e-5):
    """Only the leaf may move; preserve asset/collider identities and topology."""
    expected = predicted_variant(source, specification)
    old = {r["path"]:r for r in source["local_assets"]}
    new = {r["path"]:r for r in actual["local_assets"]}
    predicted = next(r for r in expected["local_assets"] if r["path"] == ASSETS["leaf"])
    errors = []
    if len(old) != len(source["local_assets"]) or len(new) != len(actual["local_assets"]) or set(old) != set(new):
        return ["asset_inventory"]
    for path, row in old.items():
        current = new[path]
        if path != ASSETS["leaf"]:
            if row != current:
                errors.append("non_leaf_asset_changed:"+path)
            continue
        if (row["collision"]["collision_prim_paths"] != current["collision"]["collision_prim_paths"]
                or row["collision"]["collision_prim_count"] != current["collision"]["collision_prim_count"]):
            errors.append("leaf_collision_inventory")
            continue
        for before, after, wanted in zip(row["collision"]["primitives"], current["collision"]["primitives"],
                predicted["collision"]["primitives"]):
            a, b, p = before["authored_mesh"], after["authored_mesh"], wanted["authored_mesh"]
            if (before["path"] != after["path"] or a["face_vertex_indices"] != b["face_vertex_indices"]
                    or a["face_vertex_counts"] != b["face_vertex_counts"]
                    or a["physics_approximation"] != b["physics_approximation"]):
                errors.append("leaf_topology_or_approximation")
            points, desired = np.asarray(b["world_points_m"]), np.asarray(p["world_points_m"])
            if points.shape != desired.shape or not np.allclose(points, desired, rtol=0, atol=tolerance):
                errors.append("live_leaf_vertices_not_the_hinge_rotation")
        for key in ("minimum_m", "maximum_m"):
            if not np.allclose(current["collision"]["bound"][key], predicted["collision"]["bound"][key], rtol=0, atol=tolerance):
                errors.append("live_leaf_collision_bound")
    return errors


def analyze_open_variant(actual):
    variant = actual.get("geometry_variant", {})
    if (variant.get("condition") != CONDITION or not variant.get("actual_new_leaf_vertices_exported")
            or not variant.get("only_leaf_transform_changed") or not variant.get("collider_flags_unchanged")):
        raise ValueError("A model-only rotation cannot be labelled as live fully-open preparation")
    result = analyze_doorway(actual)
    result.update(door_state=CONDITION, door_moved=True, session_layer_door_transform_modified=True,
        source_USD_file_saved=False,
        status="FULLY_OPEN_GEOMETRY_RECORDED_HOLD_TERRAIN_AND_TASK_READINESS",
        next_gate="validate_threshold_support_and_terrain_then_define_fresh_H3_task_and_map")
    return result
