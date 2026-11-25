"""Typed support readback fix: original Vec3d -> world -> JSON, exact equality.

The v3 reader supplied Vec3f to Matrix4d.Transform, selecting its Vec3f-return
overload and rounding WORLD coordinates before comparison. The original live
support preparation used explicit Vec3d, retaining double translation. Use
that exact original readback convention; do not change support or tolerances.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
from pathlib import Path

import numpy as np

import hospital_h4_cohort_preparation_v3 as previous


ROOT, GATE, support, completion = previous.ROOT, previous.GATE, previous.support, previous.completion
OLD_ROBOT, ROBOT, MARKER = previous.OLD_ROBOT, previous.ROBOT, previous.MARKER
ASSET, ASSET_SHA = previous.ASSET, previous.ASSET_SHA
SETUP, MAP_REVIEW, MAP, MODELS = previous.SETUP, previous.MAP_REVIEW, previous.MAP, previous.MODELS
OUTPUT, ARCHIVE, LIVE, FAILURE = previous.OUTPUT, previous.ARCHIVE, previous.LIVE, previous.FAILURE
PREFLIGHT = OUTPUT/'preflight_v4/cohort_preflight.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_prepare_ijcas_hospital_h4_fresh_cohort_v4.py')
SOURCE, SOURCE_SHA = previous.SOURCE, previous.SOURCE_SHA
VERSION = previous.VERSION+'_exact_original_Vec3d_support_readback_v4'
MATERIAL, GROUND, WHEELS, JOINT_SETTINGS = previous.MATERIAL, previous.GROUND, previous.WHEELS, previous.JOINT_SETTINGS
canonical, verify, support_covers = previous.canonical, previous.verify, previous.support_covers
joint_seed, validate_new_robot, author_new_robot = previous.joint_seed, previous.validate_new_robot, previous.author_new_robot
protected_session_text, FOOTPRINT = previous.protected_session_text, previous.FOOTPRINT
full_footprint, support_area_missing = previous.full_footprint, previous.support_area_missing
sampled_support_screen = previous.sampled_support_screen


def support_readback_checks(stage):
    """Read-only, independently named checks; never relax exact saved equality."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    design = support.load_design()
    receipt = json.loads(support.LIVE.read_bytes())
    prim = stage.GetPrimAtPath(support.PATCH_MESH)
    if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
        raise ValueError('Actual H4 Session support is missing or not an active Mesh')
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    mesh = UsdGeom.Mesh(prim)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    # Match the immutable original support author's readback byte-for-byte.
    points = [list(map(float, matrix.Transform(Gf.Vec3d(*point)))) for point in mesh.GetPointsAttr().Get()]
    checks = dict(
        world_points_EXACTLY_match_saved_original_Vec3d_readback=points == receipt['actual_world_points_m'],
        canonical_material_binding_matches=bool(material and str(material.GetPath()) == MATERIAL),
        support_collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True,
        static_triangle_mesh_approximation_none=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()) == 'none',
        face_vertex_counts_EXACTLY_match=list(mesh.GetFaceVertexCountsAttr().Get()) == design['patch_world_mesh']['face_vertex_counts'],
        face_vertex_indices_EXACTLY_match=list(mesh.GetFaceVertexIndicesAttr().Get()) == design['patch_world_mesh']['face_vertex_indices'],
        support_is_static_not_a_rigid_body=not prim.HasAPI(UsdPhysics.RigidBodyAPI))
    difference = None
    if np.shape(points) == np.shape(receipt['actual_world_points_m']):
        difference = float(np.max(np.abs(np.asarray(points)-receipt['actual_world_points_m'])))
    return dict(checks=checks, actual_world_points_m=points, maximum_saved_coordinate_difference_m=difference,
        explicit_original_readback_type='Gf.Vec3d', absolute_or_relative_equality_tolerance_added=False)


def verify_actual_support(stage):
    result = support_readback_checks(stage)
    failed = [key for key, ok in result['checks'].items() if not ok]
    if failed:
        raise ValueError('Actual H4 support mismatch: '+', '.join(failed)+
            '; maximum coordinate difference m='+str(result['maximum_saved_coordinate_difference_m']))
    return result


def prepare():
    report = previous.prepare()
    report.update(version=VERSION,
        support_readback_uses_exact_original_Gf_Vec3d_transform_overload=True,
        original_saved_support_coordinates_topology_material_and_exact_equality_preserved=True,
        prior_reported_v3_error_was_before_actual_H3_archive_or_timeline_Stop=True,
        no_automatic_recovery_or_physical_retry_added=True)
    paths = (Path(__file__), ENTRY, previous.PREFLIGHT)
    report['input_sha256'].update({str(p): completion.sha(p) for p in paths})
    report['source_sha256'].update({p.name: completion.sha(p) for p in (Path(__file__), ENTRY)})
    verify(report['input_sha256'])
    return canonical(report)


def load_preflight():
    report = json.loads(PREFLIGHT.read_bytes()); verify(report['input_sha256'])
    if report != prepare():
        raise ValueError('Exact typed-readback H4 preflight changed; no substitute support or constraints')
    return report
