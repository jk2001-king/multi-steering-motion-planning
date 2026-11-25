"""Audited local H4 flat support design from actual ORIGINAL floor triangles.

The current H3 floor and all obstacle/caster/gain settings remain unchanged.
Only a NEW Session subtree may later be added, after this offline design is
verified. No hole/stair area is filled: planar asset join seams alone may be
welded within an explicit 50-micrometre topology tolerance, not a safety waiver.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
import math
from pathlib import Path

import hospital_h4_floor_review as floor
from hospital_geometry_tools import footprint_polygon


ROOT = floor.ROOT
GATE = floor.GATE
OUTPUT = GATE/'hospital_h4_support/design_r01'
TARGET = OUTPUT/'support_design.json'
PATCH_ROOT = '/World/IJCAS_HospitalH4SupportR01'
PATCH_MESH = PATCH_ROOT+'/CollisionMesh'
LIVE = GATE/'hospital_h4_support/session_r01/live_support_preparation.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_prepare_ijcas_hospital_h4_local_support.py')
VERSION = 'H4_local_original_floor_L_support_same_top_material_session_only_v1'
GEOMETRY_INSET_M = .001
STAIR_REFERENCE_RESERVE_M = .350
MAXIMUM_PLANAR_JOIN_WELD_M = .000050
AREA_TOLERANCE_M2 = 1e-9


def area(polygon):
    if len(polygon) < 3:
        return 0.
    x, y = polygon[0]
    return sum((a[0]-x)*(b[1]-y)-(b[0]-x)*(a[1]-y)
        for a, b in zip(polygon, polygon[1:]+polygon[:1]))/2


def ccw(polygon):
    return list(polygon) if area(polygon) >= 0 else list(reversed(polygon))


def rectangle(left, right, bottom, top):
    if not all(math.isfinite(value) for value in (left, right, bottom, top)) or left >= right or bottom >= top:
        raise ValueError('Finite positive support rectangle required')
    return [(left, bottom), (right, bottom), (right, top), (left, top)]


def clip_halfplane(polygon, a, b, inside=True):
    """Convex clipping with exact shared-line partitions (no support inflation)."""
    if not polygon:
        return []
    def distance(point):
        value = (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
        return value if inside else -value
    result = []
    last = polygon[-1]; dl = distance(last)
    for point in polygon:
        dp = distance(point)
        if (dl >= 0) != (dp >= 0):
            fraction = dl/(dl-dp)
            result.append((last[0]+fraction*(point[0]-last[0]), last[1]+fraction*(point[1]-last[1])))
        if dp >= 0:
            result.append(tuple(point))
        last, dl = point, dp
    clean = []
    for point in result:
        if not clean or math.hypot(point[0]-clean[-1][0], point[1]-clean[-1][1]) > 1e-12:
            clean.append(point)
    if len(clean) > 1 and math.hypot(clean[0][0]-clean[-1][0], clean[0][1]-clean[-1][1]) <= 1e-12:
        clean.pop()
    return clean if len(clean) >= 3 and abs(area(clean)) > 1e-13 else []


def subtract_convex(subject, cover):
    remainder = list(subject); outside = []
    boundary = ccw(cover)
    for a, b in zip(boundary, boundary[1:]+boundary[:1]):
        fragment = clip_halfplane(remainder, a, b, inside=False)
        if fragment:
            outside.append(fragment)
        remainder = clip_halfplane(remainder, a, b, inside=True)
        if not remainder:
            break
    return outside


def uncovered(subjects, covers):
    """Actual planar union subtraction, NOT sum-of-areas or corner sampling."""
    pieces = [ccw(poly) for poly in subjects]
    for cover in covers:
        pieces = [piece for poly in pieces for piece in subtract_convex(poly, cover)]
        if not pieces:
            break
    return pieces


def floor_triangles(record):
    triangles = []; bounds = {}
    for root in record['original_floor_roots']:
        points = []
        for mesh in root['meshes']:
            if not mesh['is_active'] or floor.mesh_summary(mesh) != mesh['flat_mesh_summary']:
                raise ValueError('Exact active original flat floor mesh required')
            if any(count != 3 for count in mesh['face_vertex_counts']):
                raise ValueError('Use the actual triangulated floor, not an invented polygon fan')
            values = mesh['world_points_m']; points.extend(values)
            for index in range(0, len(mesh['face_vertex_indices']), 3):
                triangle = ccw([tuple(values[i][:2]) for i in mesh['face_vertex_indices'][index:index+3]])
                if area(triangle) <= 1e-12:
                    raise ValueError('Nondegenerate original floor triangle required')
                triangles.append(triangle)
        bounds[root['path']] = dict(minimum_m=[min(point[i] for point in points) for i in range(3)],
            maximum_m=[max(point[i] for point in points) for i in range(3)], points=points)
    return triangles, bounds


def ear_triangles(polygon):
    """Triangulate the fixed simple CCW L polygon without internal T-junctions."""
    remaining = list(range(len(polygon))); triangles = []
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def in_triangle(point, triangle):
        return all(cross(a, b, point) >= -1e-12 for a, b in zip(triangle, triangle[1:]+triangle[:1]))
    while len(remaining) > 3:
        for at, index in enumerate(remaining):
            previous = remaining[at-1]; following = remaining[(at+1) % len(remaining)]
            triangle = [polygon[previous], polygon[index], polygon[following]]
            if cross(*triangle) <= 1e-12:
                continue
            if any(in_triangle(polygon[other], triangle) for other in remaining if other not in (previous, index, following)):
                continue
            triangles.append([previous, index, following]); remaining.pop(at)
            break
        else:
            raise ValueError('Support L polygon must be simple and triangulable')
    triangles.append(remaining)
    return triangles


def solid_mesh(polygon, top, bottom):
    if area(polygon) <= 0 or not math.isfinite(top) or not math.isfinite(bottom) or top <= bottom:
        raise ValueError('CCW finite closed solid support required')
    count = len(polygon)
    points = [[x, y, top] for x, y in polygon]+[[x, y, bottom] for x, y in polygon]
    upper = ear_triangles(polygon)
    faces = upper+[[index+count for index in reversed(triangle)] for triangle in upper]
    for index in range(count):
        following = (index+1) % count
        faces += [[index, index+count, following+count], [index, following+count, following]]
    edges = {}
    for face in faces:
        for a, b in zip(face, face[1:]+face[:1]):
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append((a, b))
    if any(len(values) != 2 or values[0] != tuple(reversed(values[1])) for values in edges.values()):
        raise ValueError('Every support edge must have two oppositely oriented incident faces')
    return dict(world_points_m=points, face_vertex_counts=[3]*len(faces),
        face_vertex_indices=[index for face in faces for index in face],
        closed_manifold_edge_count=len(edges), closed_oriented_surface=True,
        top_triangle_count=len(upper), support_area_m2=area(polygon))


def build():
    h = floor.inventory.retained.completion
    record = json.loads(floor.TARGET.read_bytes()); h.verify_files(record['input_sha256'])
    if (record['version'] != floor.VERSION
            or not record['H3_body_12_joints_settings_root_session_source_and_timeline_unchanged']
            or not record['visual_floors_remain_without_enabled_collision']
            or not record['no_floor_extension_collider_or_material_assignment_performed']
            or record['canonical_support_resolved_material'] != '/World/PatientTransportLabGround/WheelGroundContact'):
        raise ValueError('Actual get-only original floor surfaces and retained H3 required')
    audit = json.loads(floor.AUDIT.read_bytes()); h.verify_files(audit['input_sha256']); h.verify_files(audit['output_sha256'])
    setup_path = floor.inventory.SETUP
    setup = json.loads(setup_path.read_bytes()); h.verify_files(setup['input_sha256']); h.verify_files(setup['output_sha256'])
    triangles, bounds = floor_triangles(record)
    first, second = (bounds[path] for path in floor.FLOORS)
    corners = record['canonical_support_world_corners_m']
    old_min = [min(point[i] for point in corners) for i in range(3)]
    old_max = [max(point[i] for point in corners) for i in range(3)]
    left = first['minimum_m'][0]+GEOMETRY_INSET_M
    right = min(first['maximum_m'][0], second['maximum_m'][0])-GEOMETRY_INSET_M
    cross_left = max(setup['raster']['origin_xy_m'][0], second['minimum_m'][0])+GEOMETRY_INSET_M
    original_join_reference = (first['maximum_m'][1]+second['minimum_m'][1])/2
    north = setup['geometry_anchors']['stairs']['minimum_m'][1]-STAIR_REFERENCE_RESERVE_M
    seam_low = min(point[1] for point in first['points'] if abs(point[1]-original_join_reference) < .001)
    seam_high = max(point[1] for point in second['points'] if abs(point[1]-original_join_reference) < .001)
    if not 0 <= seam_high-seam_low <= MAXIMUM_PLANAR_JOIN_WELD_M:
        raise ValueError('Original-floor join mismatch exceeds the explicit 50 micrometre topology tolerance')
    # The cross-corridor's southern boundary is slightly skewed. Inset its
    # rectangle from the maximum ACTUAL boundary Y, not an AABB midpoint that
    # would silently extend a few micrometres into the room-side boundary.
    join = seam_high+GEOMETRY_INSET_M
    stem = rectangle(left, right, old_max[1], join)
    cross = rectangle(cross_left, right, join, north)
    polygon = [stem[0], stem[1], cross[2], cross[3], cross[0], stem[3]]
    seam = rectangle(left, right, seam_low, seam_high)
    missing_raw = uncovered([stem, cross], triangles)
    missing_after_weld = uncovered([stem, cross], [*triangles, seam])
    if sum(abs(area(poly)) for poly in missing_after_weld) > AREA_TOLERANCE_M2:
        raise ValueError('Support would fill an original floor hole or extend outside floor topology')
    canonical = rectangle(old_min[0], old_max[0], old_min[1], old_max[1])
    endpoint_checks = {}
    for key in ('start_xyyaw', 'goal_xyyaw'):
        remainder = uncovered([list(map(tuple, footprint_polygon(setup[key])))], [canonical, stem, cross])
        endpoint_checks[key] = sum(abs(area(poly)) for poly in remainder) <= AREA_TOLERANCE_M2
    if not all(endpoint_checks.values()):
        raise ValueError('Full expanded endpoint footprint not covered by canonical plus local support')
    collision = json.loads(floor.inventory.TARGET.read_bytes())
    stair_checks = []
    for row in collision['collision_prims']:
        if row['classification'] != 'enabled_in_ROI' or 'Stairs' not in row['path']:
            continue
        bound = row['world_aligned_bound']; lo, hi = bound['minimum_m'], bound['maximum_m']
        projection = rectangle(lo[0], hi[0], lo[1], hi[1])
        overlap = [list(poly) for poly in (stem, cross)]
        outside = uncovered(overlap, [projection])
        intersection_area = area(stem)+area(cross)-sum(abs(area(poly)) for poly in outside)
        stair_checks.append(dict(path=row['path'], support_overlap_area_m2=max(0., intersection_area)))
    if any(row['support_overlap_area_m2'] > AREA_TOLERANCE_M2 for row in stair_checks):
        raise ValueError('Support patch would project into an existing stair/rail asset')
    mesh = solid_mesh(polygon, old_max[2], old_min[2])
    pins = {**record['input_sha256'], **audit['input_sha256'], **audit['output_sha256'],
        **setup['input_sha256'], **setup['output_sha256'],
        **{str(path): h.sha(path) for path in (floor.TARGET, floor.AUDIT, setup_path, floor.inventory.TARGET,
            Path(__file__), ENTRY, _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_geometry_tools.py'))}}
    h.verify_files(pins)
    return dict(version=VERSION, status='LOCAL_SUPPORT_GEOMETRY_PROVEN_SESSION_AUTHORING_PREPARED_NOT_PhysX',
        user_selected_scope='H4 전용 지지면 준비 진행 (권장)',
        source_floor_surfaces=str(floor.TARGET), original_floor_triangle_count=len(triangles),
        patch_root=PATCH_ROOT, patch_mesh=PATCH_MESH,
        patch_planar_polygon=polygon, nonoverlapping_support_rectangles=[stem, cross],
        patch_world_mesh=mesh, canonical_unchanged_support_bound=dict(minimum_m=old_min, maximum_m=old_max),
        canonical_support_material=record['canonical_support_resolved_material'],
        canonical_material_readback=record['unchanged_H3_support_readback']['material_readback'],
        top_height_m=old_max[2], bottom_height_m=old_min[2],
        contact_shape_is_NEW_closed_concave_static_triangle_mesh_NOT_original_cube=True,
        original_visual_floor_topology_preserved_except_explicit_submillimetre_planar_join_weld=True,
        original_floor_topology_without_weld_uncovered_area_m2=sum(abs(area(poly)) for poly in missing_raw),
        maximum_planar_join_weld_m=MAXIMUM_PLANAR_JOIN_WELD_M,
        actual_join_strip_height_m=seam_high-seam_low, explicit_join_strip=seam,
        topology_union_subtraction_uncovered_area_m2=sum(abs(area(poly)) for poly in missing_after_weld),
        topology_check_uses_polygon_union_subtraction_NOT_corner_or_AABB_only=True,
        full_expanded_endpoint_support_checks=endpoint_checks, stair_overlap_checks=stair_checks,
        new_support_does_NOT_overlap_current_canonical_floor_positive_area=True,
        old_H3_support_settings_and_robot_state_must_remain_unchanged=True,
        existing_hospital_floor_wall_cart_stair_door_or_threshold_attributes_may_NOT_change=True,
        only_authorized_session_prim_subtree=PATCH_ROOT,
        Play_Stop_reset_reopen_source_save_or_drive_permitted=False,
        static_support_and_cooked_collision_NOT_validated=True,
        independent_H3_whole_sequence_comparison_complete=False, CICS_runtime_integration=False,
        real_robot_required=False, use_for_final_statistics=False,
        input_sha256=pins, source_sha256={path.name: h.sha(path) for path in (Path(__file__), ENTRY)})


def load_design():
    h = floor.inventory.retained.completion
    report = json.loads(TARGET.read_bytes()); h.verify_files(report['input_sha256']); h.verify_sources(report)
    if report != build():
        raise ValueError('Local support design changed; do not author a substitute patch')
    return report
