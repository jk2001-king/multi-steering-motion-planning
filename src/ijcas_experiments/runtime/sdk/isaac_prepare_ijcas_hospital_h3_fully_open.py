"""Apply the explicitly selected fully-open H3 door BASELINE in Session Layer.

STOPPED only. Keep the measured hinge translation and all collider flags.
Export the ACTUAL new door vertices to a new variant, never overwrite H1/H2
maps, the source USD or the partial-open audit. No robot, physics or driving.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

for _h3open_name in ("ijcas_hospital_h1_static_task", "ijcas_hospital_h1_initial_task", "ijcas_hospital_h1_zero_task",
        "ijcas_hospital_h1_motion_stop_task", "ijcas_hospital_h1_path_task", "ijcas_hospital_h2_static_task",
        "ijcas_hospital_h2_path_task"):
    _h3open_task = globals().get(_h3open_name)
    if _h3open_task is not None and not _h3open_task.done():
        raise RuntimeError("A hospital task is running; do not change its scene")


def _prepare_h3_fully_open():
    import copy
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    workspace = _ijcas_workspace()
    scripts = _ijcas_path(workspace, 'studies/IJCAS/scripts')
    gate = _ijcas_path(workspace, 'studies/IJCAS/results/hospital_paper_gate')
    if globals().pop("ijcas_hospital_h3_door_condition", None) != "fully_open_baseline":
        raise RuntimeError("Explicitly set ijcas_hospital_h3_door_condition='fully_open_baseline'")
    # Every project dependency is loaded by file path; no import-path/cache
    # mutation, persistent Script Editor imports or inherited __file__ needed.
    helpers = runpy.run_path(str(scripts / "hospital_h3_door_state.py"))
    partial_path = gate / "hospital_h3_doorway_geometry/audit_r01/doorway_geometry_audit.json"
    partial_raw = partial_path.read_bytes()
    partial = json.loads(partial_raw)
    for name, expected in partial["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen H1/H3 evidence changed: "+name)
    if hashlib.sha256((scripts / "audit_hospital_h3_doorway.py").read_bytes()).hexdigest() != partial["audit_script_sha256"]:
        raise RuntimeError("The original partial-open diagnostic code changed")
    source_path = gate / "hospital_asset_geometry_audit_v02.json"
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw)
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        raise RuntimeError("STOP and reopen the hospital source USD before changing the door baseline")
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None or Path(stage.GetRootLayer().realPath).resolve() != Path(partial["source_stage"]).resolve():
        raise RuntimeError("Open the audited hospital_experiment.usda")
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) > 1e-9 or str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        raise RuntimeError("Expected metre-based Z-up hospital geometry")
    marker_path = "/World/IJCAS_HospitalH3DoorState"
    if stage.GetPrimAtPath(marker_path).IsValid():
        raise RuntimeError("An H3 door variant is already prepared; reopen the unchanged source before another preparation")
    root_file = Path(stage.GetRootLayer().realPath)
    root_hash = hashlib.sha256(root_file.read_bytes()).hexdigest()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy], useExtentsHint=False)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())

    def bound(prim):
        value = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = [float(x) for x in value.GetMin()], [float(x) for x in value.GetMax()]
        return {"minimum_m": lo, "maximum_m": hi, "size_m": [b-a for a,b in zip(lo,hi)],
            "center_m": [(a+b)/2 for a,b in zip(lo,hi)]}

    flags = {}
    for asset in source["local_assets"]:
        prim = stage.GetPrimAtPath(asset["path"])
        if not prim.IsValid():
            raise RuntimeError("Missing audited asset: "+asset["path"])
        actual = bound(prim)
        for key in ("minimum_m", "maximum_m"):
            if not np.allclose(actual[key], asset["world_aligned_visible_bound"][key], rtol=0, atol=2e-5):
                raise RuntimeError("A hospital asset differs from the partial-open source: "+asset["path"])
        if str(UsdGeom.Imageable(prim).ComputeVisibility()) != asset["computed_visibility"]:
            raise RuntimeError("An audited hospital asset visibility changed: "+asset["path"])
        for collider_path in asset["collision"]["collision_prim_paths"]:
            collider = stage.GetPrimAtPath(collider_path)
            if not collider.IsValid() or not collider.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError("Missing source collision prim: "+collider_path)
            enabled = UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
            if enabled is False:
                raise RuntimeError("An audited source collider was disabled: "+collider_path)
            flags[collider_path] = enabled
    leaf_path = helpers["ASSETS"]["leaf"]
    leaf = stage.GetPrimAtPath(leaf_path)
    old_leaf = next(row for row in source["local_assets"] if row["path"] == leaf_path)
    # Geometry baseline only: do not relocate a simulated or joint-driven door.
    for prim in Usd.PrimRange(leaf, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
            raise RuntimeError("A dynamic door requires a separate joint/physics procedure")
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            if any(str(target) == leaf_path or str(target).startswith(leaf_path+"/")
                    for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()) for target in rel.GetTargets()):
                raise RuntimeError("A joint constrains this door; geometry-only rotation refused")
    leaf_xform = UsdGeom.Xformable(leaf)
    if leaf_xform.GetResetXformStack():
        raise RuntimeError("Unexpected reset-stack door transform")
    old_world = transforms.GetLocalToWorldTransform(leaf)
    old_inverse = old_world.GetInverse()
    local_points = []

    def read_mesh(prim):
        if not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("Door collider is not a triangle mesh")
        mesh = UsdGeom.Mesh(prim)
        matrix = transforms.GetLocalToWorldTransform(prim)
        return {"world_points_m": [[float(x) for x in matrix.Transform(Gf.Vec3d(*point))] for point in mesh.GetPointsAttr().Get()],
            "face_vertex_counts": [int(v) for v in mesh.GetFaceVertexCountsAttr().Get()],
            "face_vertex_indices": [int(v) for v in mesh.GetFaceVertexIndicesAttr().Get()],
            "physics_approximation": str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None,
            "meaning": "authored_mesh_not_cooked_PhysX_collision_shape"}

    for primitive in old_leaf["collision"]["primitives"]:
        mesh = read_mesh(stage.GetPrimAtPath(primitive["path"]))
        before_mesh = primitive["authored_mesh"]
        if (mesh["face_vertex_counts"] != before_mesh["face_vertex_counts"]
                or mesh["face_vertex_indices"] != before_mesh["face_vertex_indices"]
                or mesh["physics_approximation"] != before_mesh["physics_approximation"]
                or not np.allclose(mesh["world_points_m"], before_mesh["world_points_m"], rtol=0, atol=2e-5)):
            raise RuntimeError("The original door mesh differs from the partial-open audit")
        local_points.extend(old_inverse.Transform(Gf.Vec3d(*point)) for point in mesh["world_points_m"])
    origin = [float(x) for x in old_world.ExtractTranslation()]
    axes = [[float(x) for x in old_world.TransformDir(axis)] for axis in
        (Gf.Vec3d(1,0,0), Gf.Vec3d(0,1,0), Gf.Vec3d(0,0,1))]
    lo = [min(float(p[i]) for p in local_points) for i in range(3)]
    hi = [max(float(p[i]) for p in local_points) for i in range(3)]
    specification = helpers["opening_specification"](origin, axes, lo, hi)
    parent_world = transforms.GetLocalToWorldTransform(leaf.GetParent())
    desired_world = Gf.Matrix4d(1.)
    desired_world.SetTranslate(Gf.Vec3d(*origin))
    desired_local = desired_world * parent_world.GetInverse()
    folder = gate / "hospital_h3_doorway_geometry"
    number = 1
    while (folder / f"fully_open_r{number:02d}").exists():
        number += 1
    output = folder / f"fully_open_r{number:02d}"
    output.mkdir()
    variant_path = output / "hospital_asset_geometry_audit_fully_open.json"
    audit_path = output / "doorway_geometry_audit.json"
    session = stage.GetSessionLayer()
    session_before = session.ExportToString()
    try:
        with Usd.EditContext(stage, session):
            leaf_xform.ClearXformOpOrder()
            leaf_xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "hospitalH3FullyOpen").Set(desired_local)
        bbox.Clear()
        transforms.Clear()
        actual_world = transforms.GetLocalToWorldTransform(leaf)
        if (not np.allclose(tuple(actual_world.ExtractTranslation()), origin, rtol=0, atol=1e-8)
                or any(not np.allclose(tuple(actual_world.TransformDir(axis)), wanted, rtol=0, atol=1e-8)
                    for axis,wanted in ((Gf.Vec3d(1,0,0),(1,0,0)), (Gf.Vec3d(0,1,0),(0,1,0)), (Gf.Vec3d(0,0,1),(0,0,1))))):
            raise RuntimeError("The live fully-open hinge/axes do not match the intended transform")
        variant = copy.deepcopy(source)
        variant["created_utc"] = datetime.now(timezone.utc).isoformat()
        updated_leaf = next(row for row in variant["local_assets"] if row["path"] == leaf_path)
        updated_leaf["world_aligned_visible_bound"] = bound(leaf)
        updated_leaf["computed_visibility"] = str(UsdGeom.Imageable(leaf).ComputeVisibility())
        all_points = []
        for primitive in updated_leaf["collision"]["primitives"]:
            collider = stage.GetPrimAtPath(primitive["path"])
            primitive["authored_mesh"] = read_mesh(collider)
            primitive["world_aligned_bound"] = bound(collider)
            all_points.extend(primitive["authored_mesh"]["world_points_m"])
        updated_leaf["collision"]["bound"] = helpers["points_bound"](all_points)
        for asset in source["local_assets"]:
            if asset["path"] != leaf_path:
                actual = bound(stage.GetPrimAtPath(asset["path"]))
                if any(not np.allclose(actual[key], asset["world_aligned_visible_bound"][key], rtol=0, atol=2e-5)
                        for key in ("minimum_m", "maximum_m")):
                    raise RuntimeError("A non-door hospital asset moved during preparation")
        for path, enabled in flags.items():
            if UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get() != enabled:
                raise RuntimeError("A hospital collision flag changed: "+path)
        mismatches = helpers["export_mismatches"](source, variant, specification)
        if mismatches:
            raise RuntimeError("Actual door export mismatch: "+str(mismatches))
        # Only leaf-related candidate bounds change; the full inventory remains
        # the original export, with explicitly checked non-leaf live bounds.
        for candidate in variant["candidate_paths"]:
            if candidate["path"] == leaf_path or candidate["path"].startswith(leaf_path+"/"):
                candidate["world_aligned_bound"] = bound(stage.GetPrimAtPath(candidate["path"]))
        for row in variant["targets"].values():
            if row.get("path") == leaf_path:
                row.update(world_aligned_visible_bound=updated_leaf["world_aligned_visible_bound"], collision=updated_leaf["collision"])
        variant["geometry_variant"] = {"condition": helpers["CONDITION"], "leaf_path": leaf_path,
            "base_partial_open_audit": str(source_path), "base_partial_open_audit_sha256": hashlib.sha256(source_raw).hexdigest(),
            "only_leaf_transform_changed": True, "live_non_leaf_asset_bounds_checked": True,
            "full_inventory_reexported": False, "actual_new_leaf_vertices_exported": True,
            "collider_flags_unchanged": True, "door_state_specification": specification,
            "source_USD_saved": False, "physics_execution": False}
        report = helpers["analyze_open_variant"](variant)
        report["geometry_source_audit"] = str(variant_path)
        report["partial_open_geometry_reference"] = str(partial_path)
        report["partial_open_geometry_sha256"] = hashlib.sha256(partial_raw).hexdigest()
        report["door_state_specification"] = specification
        report["checks"] = {"hinge_translation_preserved": True, "fully_open_axes_verified": True,
            "live_leaf_mesh_matches_rigid_rotation": not mismatches, "all_audited_collider_flags_unchanged": True,
            "other_asset_live_bounds_unchanged": True, "source_USD_file_unchanged": hashlib.sha256(root_file.read_bytes()).hexdigest() == root_hash}
        if not all(report["checks"].values()):
            raise RuntimeError("Source-file/door geometry preparation guard failed")
        with variant_path.open("x", encoding="utf-8") as stream:
            json.dump(variant, stream, indent=2, allow_nan=False)
            stream.write("\n")
        report["input_sha256"] = {**partial["input_sha256"], str(partial_path): hashlib.sha256(partial_raw).hexdigest(),
            str(variant_path): hashlib.sha256(variant_path.read_bytes()).hexdigest()}
        report["audit_script_sha256"] = partial["audit_script_sha256"]
        report["source_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in (
            ("hospital_h3_door_state.py", scripts / "hospital_h3_door_state.py"),
            ("isaac_prepare_ijcas_hospital_h3_fully_open.py", _ijcas_path(workspace, 'isaac/scripts/isaac_prepare_ijcas_hospital_h3_fully_open.py')))}
        with audit_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with Usd.EditContext(stage, session):
            marker = UsdGeom.Xform.Define(stage, marker_path).GetPrim()
            marker.SetCustomDataByKey("condition", helpers["CONDITION"])
            marker.SetCustomDataByKey("geometry_audit", str(audit_path))
            marker.SetCustomDataByKey("geometry_audit_sha256", hashlib.sha256(audit_path.read_bytes()).hexdigest())
            marker.SetCustomDataByKey("status", "fully_open_geometry_only_threshold_validation_pending")
    except Exception as error:
        session.ImportFromString(session_before)
        with (output / "preparation_failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "FAILED_SESSION_ROLLED_BACK_NOT_DRIVING", "error": str(error),
                "source_USD_saved": False, "PhysX_executed": False}, stream, indent=2)
        raise
    print("=== H3 FULLY OPEN DOOR BASELINE GEOMETRY PREPARED ===")
    print("additional world-Z rotation deg:", round(specification["additional_world_Z_rotation_deg"], 6))
    print("hinge translation preserved: True; all audited collider flags unchanged: True")
    print("current leaf straight-crossing diagnostic:", report["straight_constant_Y_zero_yaw_necessary_condition"]["result"])
    print("threshold collision retained; step above canonical support mm:", round(1000*report["threshold"]["authored_step_height_m"], 3))
    print("geometry checks:", report["checks"])
    print("STATUS:", report["status"])
    print("saved:", audit_path)
    print("No robot motion, physics/terrain test, path planning or Nav2. Do not press Play or save the source USD.")
    return str(output)


ijcas_hospital_h3_fully_open_directory = _prepare_h3_fully_open()
ijcas_hospital_h3_geometry_directory = ijcas_hospital_h3_fully_open_directory
from pathlib import Path as _H3OpenPath
_h3_open_overlay = _H3OpenPath("/home/jk/Research/IJCAS_ws/isaac/scripts/isaac_prepare_ijcas_hospital_h3_geometry_overlay.py")
exec(compile(_h3_open_overlay.read_text(encoding="utf-8"), str(_h3_open_overlay), "exec"), globals())
