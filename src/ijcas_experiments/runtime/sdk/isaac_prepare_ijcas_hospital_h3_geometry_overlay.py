"""H3 geometry diagnostic ONLY: raised XY annotations in Session Layer.

No robot/door movement, collision/visibility/material change to hospital assets,
physics, planning, Nav2 or source-stage save. Run STOPPED on the source stage.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

for _h3_task_name in ("ijcas_hospital_h1_static_task", "ijcas_hospital_h1_initial_task", "ijcas_hospital_h1_zero_task",
        "ijcas_hospital_h1_motion_stop_task", "ijcas_hospital_h1_path_task", "ijcas_hospital_h2_static_task",
        "ijcas_hospital_h2_path_task"):
    _h3_task = globals().get(_h3_task_name)
    if _h3_task is not None and not _h3_task.done():
        raise RuntimeError("A hospital task is still running; H3 overlay refused")

import hashlib
import json
import math
from pathlib import Path

import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


WORKSPACE = _ijcas_workspace()
directory = globals().pop("ijcas_hospital_h3_geometry_directory", None)
if not isinstance(directory, str) or not directory:
    raise RuntimeError("Explicitly set ijcas_hospital_h3_geometry_directory")
report_path = Path(directory) / "doorway_geometry_audit.json"
report = json.loads(report_path.read_bytes())
fully_open_variant = report.get("door_state") == "fully_open_baseline_session_only"
if (report.get("scenario") != "H3_doorway_immediate_turn" or not report.get("analysis_complete")
        or report.get("source_stage_modified") or (report.get("door_moved") and not fully_open_variant)
        or report.get("collision_or_terrain_policy_changed")):
    raise RuntimeError("Need a complete original or explicitly prepared fully-open H3 geometry audit")
for name, expected in report["input_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("H3 source evidence changed: "+name)
script = _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/audit_hospital_h3_doorway.py')
if hashlib.sha256(script.read_bytes()).hexdigest() != report["audit_script_sha256"]:
    raise RuntimeError("H3 audit code changed; regenerate the geometry diagnostic")
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_stopped():
    raise RuntimeError("Stop and reopen the original hospital source USD before this overlay")
context = omni.usd.get_context()
stage = context.get_stage()
if stage is None or Path(stage.GetRootLayer().realPath).resolve() != Path(report["source_stage"]).resolve():
    raise RuntimeError("Open the audited hospital source stage")
if not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.) or str(UsdGeom.GetStageUpAxis(stage)) != "Z":
    raise RuntimeError("The diagnostic requires a metre-based Z-up stage")
if fully_open_variant:
    marker = stage.GetPrimAtPath("/World/IJCAS_HospitalH3DoorState")
    if (not marker.IsValid() or marker.GetCustomDataByKey("condition") != report["door_state"]
            or marker.GetCustomDataByKey("geometry_audit") != str(report_path)
            or marker.GetCustomDataByKey("geometry_audit_sha256") != hashlib.sha256(report_path.read_bytes()).hexdigest()
            or not report.get("session_layer_door_transform_modified")
            or not report.get("checks") or not all(report["checks"].values())):
        raise RuntimeError("The live fully-open door preparation does not match this variant")
    for name, expected in report["source_sha256"].items():
        folder = "isaac/scripts" if name.startswith("isaac_") else "studies/IJCAS/scripts"
        if hashlib.sha256((_ijcas_path(WORKSPACE, folder, name)).read_bytes()).hexdigest() != expected:
            raise RuntimeError("The door-state preparation code changed: "+name)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
    UsdGeom.Tokens.proxy], useExtentsHint=False)
for key, asset in report["assets"].items():
    prim = stage.GetPrimAtPath(asset["path"])
    if not prim.IsValid():
        raise RuntimeError("Missing H3 asset: "+asset["path"])
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    expected = asset["visible_bound_m"]
    if any(not math.isclose(float(actual), saved, rel_tol=0., abs_tol=1e-5)
            for actuals, saved_values in ((bounds.GetMin(), expected["minimum_m"]), (bounds.GetMax(), expected["maximum_m"]))
            for actual, saved in zip(actuals, saved_values)):
        raise RuntimeError("The live H3 asset moved/changed; do not reuse this audit: "+key)
geometry_source = report.get("geometry_source_audit") or next(name for name in report["input_sha256"] if name.endswith("hospital_asset_geometry_audit_v02.json"))
source = json.loads(Path(geometry_source).read_bytes())
assets = {row["path"]: row for row in source["local_assets"]}
ROOT = "/World/IJCAS_HospitalH3GeometryOverlay"
DRAW_Z = 3.2  # XY annotations above wall tops; NOT physical geometry at this Z.


def _h3_plane_segments(asset, height=.250):
    """Cross authored triangles, not AABBs, at one declared height."""
    segments = []
    for primitive in asset["collision"]["primitives"]:
        mesh = primitive["authored_mesh"]
        points, indices = mesh["world_points_m"], mesh["face_vertex_indices"]
        for offset in range(0, len(indices), 3):
            triangle = [points[i] for i in indices[offset:offset+3]]
            hits = []
            for a, b in zip(triangle, triangle[1:]+triangle[:1]):
                da, db = a[2]-height, b[2]-height
                if da*db < 0:
                    fraction = da/(da-db)
                    hits.append((a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1])))
            if len(hits) == 2:
                segments.append(hits)
    if not segments:
        raise RuntimeError("No triangle cross section for "+asset["path"])
    return segments


def _h3_curve(name, segments, color, elevation=DRAW_Z):
    curve = UsdGeom.BasisCurves.Define(stage, ROOT+"/"+name)
    curve.CreateTypeAttr("linear")
    curve.CreateWrapAttr("nonperiodic")
    curve.CreatePointsAttr([Gf.Vec3f(float(x), float(y), elevation) for segment in segments for x, y in segment])
    curve.CreateCurveVertexCountsAttr([len(segment) for segment in segments])
    curve.CreateWidthsAttr([.012])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    material = UsdShade.Material.Define(stage, ROOT+"/AnnotationMaterials/"+name)
    shader = UsdShade.Shader.Define(stage, str(material.GetPath())+"/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0., 0., 0.))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)


session = stage.GetSessionLayer()
before = session.ExportToString()
try:
    with Usd.EditContext(stage, session):
        overlay = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
        overlay.SetCustomDataByKey("scope", "raised_XY_geometry_annotations_ONLY_NOT_H3_setup_or_path")
        overlay.SetCustomDataByKey("geometry_audit", str(report_path))
        overlay.SetCustomDataByKey("geometry_audit_sha256", hashlib.sha256(report_path.read_bytes()).hexdigest())
        for key, label, color in (("wall", "WallSectionCyan", (.05, .7, 1.)),
                ("frame", "FrameSectionGreen", (.1, 1., .2)),
                ("leaf", "ExistingLeafSectionMagenta", (1., .05, .45))):
            _h3_curve(label, _h3_plane_segments(assets[report["assets"][key]["path"]]), color)
        bound = report["assets"]["threshold"]["collision_bound_m"]
        x0, y0 = bound["minimum_m"][:2]
        x1, y1 = bound["maximum_m"][:2]
        _h3_curve("ThresholdAABBYellow", [[(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]], (1., .65, .02), DRAW_Z+.02)
        section = report["sampled_height_sections"][0]
        lo, hi = section["structural_opening_y_m"]
        x = section["wall_x_m"]+.35
        _h3_curve("StructuralWidthOnlyGreen", [[(x,lo),(x,hi)], [(x-.05,lo),(x+.05,lo)],
            [(x-.05,hi),(x+.05,hi)]], (.1, 1., .2))
        half_width = report["required_width_with_clearance_m"]/2
        center = (lo+hi)/2
        x += .3
        _h3_curve("RobotWidthReferenceOnlyBlue", [[(x,center-half_width),(x,center+half_width)],
            [(x-.05,center-half_width),(x+.05,center-half_width)],
            [(x-.05,center+half_width),(x+.05,center+half_width)]], (.15, .2, 1.))
except Exception:
    session.ImportFromString(before)
    raise
context.get_selection().set_selected_prim_paths([ROOT], True)
leaf_prim = stage.GetPrimAtPath(report["assets"]["leaf"]["path"])
leaf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
leaf_world = leaf_cache.GetLocalToWorldTransform(leaf_prim)
leaf_inverse = leaf_world.GetInverse()
local_points = []
for primitive_path in report["assets"]["leaf"]["collision_prim_paths"]:
    primitive = stage.GetPrimAtPath(primitive_path)
    mesh = UsdGeom.Mesh(primitive)
    if not mesh:
        raise RuntimeError("The door hinge review requires the audited triangle mesh")
    mesh_world = leaf_cache.GetLocalToWorldTransform(primitive)
    for point in mesh.GetPointsAttr().Get():
        local_points.append(leaf_inverse.Transform(mesh_world.Transform(Gf.Vec3d(*point))))
if not local_points:
    raise RuntimeError("No door collider points available for hinge review")
print("=== H3 LIVE DOOR HINGE REVIEW (READ ONLY) ===")
print("leaf root world origin m:", tuple(leaf_world.ExtractTranslation()))
for name, axis in (("X", Gf.Vec3d(1., 0., 0.)), ("Y", Gf.Vec3d(0., 1., 0.)), ("Z", Gf.Vec3d(0., 0., 1.))):
    print("leaf root world axis "+name+":", tuple(leaf_world.TransformDir(axis)))
print("leaf collider in root frame minimum:", tuple(min(float(p[i]) for p in local_points) for i in range(3)))
print("leaf collider in root frame maximum:", tuple(max(float(p[i]) for p in local_points) for i in range(3)))
print("leaf root authored xform ops:", [(op.GetOpName(), str(op.Get())) for op in UsdGeom.Xformable(leaf_prim).GetOrderedXformOps()])
if fully_open_variant:
    print("Fully-open baseline applied in Session Layer; hinge fixed; threshold retained. No physics or driving.")
else:
    print("Fully-open BASELINE selected by user; door rotation NOT applied yet: hinge origin/orientation review required.")
print("=== H3 DOORWAY GEOMETRY OVERLAY ONLY ===")
print("door condition:", report["door_state"])
print("cyan=wall, green=frame; both are authored mesh cross sections at z=0.250 m")
print("magenta=current door leaf cross section; yellow=threshold collision AABB")
print("green side bar=structural opening ONLY; blue bar=robot width + 10 mm each side")
print("Annotations are raised XY projections at z=3.2 m for visibility, NOT physical objects or task poses.")
print("Select", ROOT, "then move the pointer over the viewport, F and Top View.")
print("Current leaf straight-crossing diagnostic:", report["straight_constant_Y_zero_yaw_necessary_condition"]["result"])
print("Curved path NOT tested; threshold PhysX validation PENDING. H3 is NOT approved to drive.")
print("Timeline STOPPED. Session Layer annotations only; do not save the source USD or press Play.")
