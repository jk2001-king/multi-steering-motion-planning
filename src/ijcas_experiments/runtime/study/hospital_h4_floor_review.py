"""Targeted original visual-floor geometry review; no live scene mutation."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
import math
from pathlib import Path

import hospital_h4_collision_inventory as inventory


ROOT = inventory.ROOT
GATE = inventory.retained.completion.final.GATE
AUDIT = GATE/'hospital_h4_complete_map_support/review_r01/complete_map_support_review.json'
PREFLIGHT = GATE/'hospital_h4_geometry_review/floor_preparation_r01/floor_readback_preflight.json'
TARGET = GATE/'hospital_h4_geometry_review/floor_surfaces_r01/live_floor_surfaces.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_review_ijcas_hospital_h4_floor_surfaces.py')
FLOORS = ('/World/Hospital/Geo_Floor_Costum18_24', '/World/Hospital/Geo_Floor_Costum19_27')
VERSION = 'H4_original_floor_surface_topology_and_canonical_support_get_only_v1'


def mesh_summary(mesh):
    """Validate authored flat floor topology, never substitute a root AABB.

    Polygon areas are summed diagnostics, NOT union-coverage/hole proofs.
    Later support design must test the actual triangulated surface union.
    """
    points = mesh['world_points_m']; counts = mesh['face_vertex_counts']; indices = mesh['face_vertex_indices']
    if (not points or not counts or any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in points)
            or any(not isinstance(count, int) or count < 3 for count in counts)
            or sum(counts) != len(indices)
            or any(not isinstance(index, int) or not 0 <= index < len(points) for index in indices)):
        raise ValueError('Complete finite original floor topology required')
    lower = [min(point[i] for point in points) for i in range(3)]
    upper = [max(point[i] for point in points) for i in range(3)]
    if upper[2]-lower[2] > 2e-5 or max(abs(value-.001) for value in (lower[2], upper[2])) > 2e-5:
        raise ValueError('Original corridor floor is not horizontal at the common support height')
    area = 0.; offset = 0
    for count in counts:
        face = [points[index] for index in indices[offset:offset+count]]; offset += count
        area += abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(face, face[1:]+face[:1])))/2
    if area <= 0:
        raise ValueError('Positive projected floor polygon area required')
    return dict(point_count=len(points), polygon_count=len(counts),
        world_bound=dict(minimum_m=lower, maximum_m=upper),
        projected_polygon_area_sum_m2=area,
        polygon_area_sum_is_NOT_union_or_hole_free_coverage=True,
        flat_at_original_canonical_top=True,
        physical_floor_support_or_drive_admission=False)


def prepare():
    h = inventory.retained.completion
    audit = json.loads(AUDIT.read_bytes())
    h.verify_files(audit['input_sha256']); h.verify_files(audit['output_sha256'])
    if (audit['status'] != 'HOLD_H4_CANONICAL_FLOOR_DOES_NOT_COVER_TASK'
            or not audit['complete_USD_authored_collision_inventory_verified']
            or not audit['every_cached_miss_is_outside_current_canonical_floor']
            or audit['any_floor_extension_applied'] is not False
            or audit['H4_static_or_drive_admission'] is not False):
        raise ValueError('Actual support-shortfall audit required; no missing-floor waiver')
    floors = {row['path']: row for row in audit['visual_floor_bounds']}
    if set(floors) != set(FLOORS):
        raise ValueError('Exactly the two unchanged original corridor floor roots required')
    pins = {**audit['input_sha256'], **audit['output_sha256'],
        **{str(path): h.sha(path) for path in (AUDIT, Path(__file__), ENTRY)}}
    return dict(version=VERSION, status='ORIGINAL_VISUAL_FLOOR_TOPOLOGY_GET_ONLY_PREPARED',
        original_floor_roots=floors, canonical_support=audit['canonical_support'],
        current_H3_FINAL_body_12_joints_gains_root_session_source_and_timeline_must_remain_retained=True,
        any_floor_CollisionAPI_or_material_change_permitted=False,
        any_robot_pose_caster_reset_or_Play_permitted=False,
        new_static_or_drive_admission=False, CICS_runtime_integration=False,
        use_for_final_statistics=False, input_sha256=pins,
        source_sha256={path.name: h.sha(path) for path in (Path(__file__), ENTRY)})


def load_preflight():
    saved = json.loads(PREFLIGHT.read_bytes())
    inventory.retained.completion.verify_files(saved['input_sha256'])
    inventory.retained.completion.verify_sources(saved)
    if saved != prepare():
        raise ValueError('Original-floor preflight changed; no reset, Play or substitute floor')
    return saved
