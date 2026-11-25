#!/usr/bin/env python3
"""Reconcile actual all-stage H4 inventory, authored map and cached floor hits.

Complete USD collider coverage is NOT cooked-shape equivalence. The H3 support
box ends before the H4 goal: missing sampled support must be preserved as HOLD,
not converted to empty obstacle cells and misreported as traversable floor.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import build_hospital_h4_setup_candidate as setup_builder
import hospital_h4_collision_inventory as inventory
from hospital_geometry_tools import polygon_cells, pose_collision


ROOT, GATE = setup_builder.ROOT, setup_builder.BASE
OUTPUT = GATE/'hospital_h4_complete_map_support/review_r01'
TARGET = OUTPUT/'complete_map_support_review.json'
GRID = OUTPUT/'complete_authored_AABB_occupancy.npz'
VERSION = 'H4_actual_all_stage_authored_inventory_map_cached_floor_reconciliation_v1'
GROUND = '/World/PatientTransportLabGround/CollisionBox'
FLOOR_ROOTS = ('/World/Hospital/Geo_Floor_Costum18_24', '/World/Hospital/Geo_Floor_Costum19_27')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Recorded H4 evidence changed: '+path)


def inside_xy(point, bound, tolerance=0.):
    return all(bound['minimum_m'][i]-tolerance <= point[i] <= bound['maximum_m'][i]+tolerance for i in range(2))


def project(rows, setup):
    raster = setup['raster']; lower, upper = setup['obstacle_height_slab_m']
    occupied = np.zeros(raster['shape_rows_columns'], dtype=bool)
    included = []; excluded = []
    paths = set()
    for row in rows:
        if row['classification'] != 'enabled_in_ROI':
            continue
        if row['path'] in paths:
            raise ValueError('Duplicate enabled collider path')
        paths.add(row['path'])
        bounds = row['world_aligned_bound']; lo, hi = bounds['minimum_m'], bounds['maximum_m']
        if (not all(math.isfinite(value) for value in (*lo, *hi)) or any(a > b for a, b in zip(lo, hi))):
            raise ValueError('Finite ordered collider bounds required')
        if hi[2] < lower or lo[2] > upper:
            excluded.append(dict(path=row['path'], reason='outside_original_height_slab'))
            continue
        rr, cc = polygon_cells([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])],
            raster['origin_xy_m'], raster['resolution_m'], occupied.shape)
        occupied[rr, cc] = True
        included.append(dict(path=row['path'], world_aabb=bounds, occupied_cell_count=len(rr)))
    return occupied, included, excluded


def reconcile_floor_queries(queries, ground, visual_floors):
    expected = {(name, sample) for name in ('start_xyyaw', 'goal_xyyaw')
        for sample in ('center', 'front_left', 'front_right', 'rear_left', 'rear_right')}
    rays = queries['endpoint_rays']
    if (len(rays) != 10 or {(row['task_pose'], row['sample']) for row in rays} != expected
            or not queries['available'] or not queries['positive_control_hit'] or queries['errors']):
        raise ValueError('Complete successful cached-query transaction and positive control required')
    rows = []
    for ray in rays:
        inside = inside_xy(ray['xy_m'], ground['world_aligned_bound'])
        hit_ground = bool(ray['hit'] and ray.get('collision') == GROUND)
        rows.append(dict(task_pose=ray['task_pose'], sample=ray['sample'], xy_m=ray['xy_m'],
            inside_current_canonical_support_xy=inside,
            actual_cached_PhysX_hit=bool(ray['hit']), actual_hit_collider=ray.get('collision'),
            actual_hit_height_m=ray.get('position_m', [None, None, None])[2],
            actual_hit_matches_canonical_box_prediction=inside == hit_ground,
            missing_canonical_support_distance_y_m=max(0., ray['xy_m'][1]-ground['world_aligned_bound']['maximum_m'][1]),
            containing_visual_floor_ROOT_AABBs_NOT_surface_proof=[item['path'] for item in visual_floors
                if inside_xy(ray['xy_m'], item['world_aligned_visible_bound'])]))
    return rows


def analyze():
    data = json.loads(inventory.TARGET.read_bytes()); verify(data['input_sha256'])
    setup = json.loads(setup_builder.TARGET.read_bytes()); verify(setup['input_sha256']); verify(setup['output_sha256'])
    old = json.loads(setup_builder.SOURCE.read_bytes()); verify(old['input_sha256'])
    summary = inventory.inventory_summary(data['collision_prims'],
        data['inventory_summary']['active_unloaded_payload_paths'], data['inventory_summary']['errors'])
    if (summary != data['inventory_summary'] or not summary['authored_inventory_coverage_complete']
            or not data['original_H3_body_12_joints_settings_root_session_source_and_timeline_unchanged']
            or data['candidate_task'] != str(setup_builder.TARGET)):
        raise ValueError('Actual complete all-stage collision inventory with H3 continuity required')
    grounds = [row for row in data['collision_prims'] if row['path'] == GROUND]
    if len(grounds) != 1 or grounds[0]['classification'] != 'enabled_in_ROI' or grounds[0]['type_name'] != 'Cube':
        raise ValueError('Exactly one original enabled canonical H3 support box required')
    ground = grounds[0]
    floors = [row for row in old['assets'] if row['path'] in FLOOR_ROOTS]
    if len(floors) != 2 or any(row['collision']['enabled_collision_prim_count'] != 0 for row in floors):
        raise ValueError('Original visual-only corridor floor roots changed')
    occupied, included, excluded = project(data['collision_prims'], setup)
    with np.load(setup['candidate_grid']) as grid:
        old_grid = grid['occupied'].copy()
    raster = setup['raster']
    endpoint_clear = {key: not pose_collision(occupied, setup[key], raster['origin_xy_m'], raster['resolution_m'])
        for key in ('start_xyyaw', 'goal_xyyaw')}
    rows = reconcile_floor_queries(data['sampled_cached_PhysX_queries'], ground, floors)
    missing = [row for row in rows if not row['actual_cached_PhysX_hit']]
    if not all(row['actual_hit_matches_canonical_box_prediction'] for row in rows):
        raise ValueError('Observed support mismatch not explained by canonical box; retain unisolated HOLD')
    pins = {**data['input_sha256'], **setup['input_sha256'], **setup['output_sha256'],
        **{str(path): sha(path) for path in (inventory.TARGET, setup_builder.TARGET, setup_builder.SOURCE,
            Path(__file__), Path(inventory.__file__), ROOT/'studies/IJCAS/scripts/hospital_geometry_tools.py')}}
    report = dict(version=VERSION, status='HOLD_H4_CANONICAL_FLOOR_DOES_NOT_COVER_TASK',
        collision_inventory=str(inventory.TARGET), inventory_summary=summary,
        complete_USD_authored_collision_inventory_verified=True,
        cooked_shapes_equal_authored_AABBs_verified=False,
        map_scope='all enabled ROI colliders; conservative authored AABB obstacle projection, NOT a support or cooked collision certificate',
        original_candidate_grid_changed_cell_count=int(np.count_nonzero(occupied != old_grid)),
        old_obstacle_cells_removed=int(np.count_nonzero(old_grid & ~occupied)),
        newly_detected_obstacle_cells=int(np.count_nonzero(occupied & ~old_grid)),
        all_original_thresholds_stairs_walls_and_cart_kept=True,
        included_collision_AABBs=included, excluded_by_height=excluded,
        complete_authored_grid=str(GRID), occupancy_digest=hashlib.sha256(occupied.tobytes()).hexdigest(),
        raster=raster, obstacle_height_slab_m=setup['obstacle_height_slab_m'],
        full_envelope_endpoint_obstacle_clear=endpoint_clear,
        canonical_support=ground,
        endpoint_support_samples=rows, cached_endpoint_floor_hit_count=sum(row['actual_cached_PhysX_hit'] for row in rows),
        cached_endpoint_floor_miss_count=len(missing),
        every_cached_miss_is_outside_current_canonical_floor=True,
        visual_floor_bounds=floors,
        authored_visual_floor_AABB_is_NOT_triangle_coverage_or_physical_support_proof=True,
        proposed_next_action='read original corridor-floor meshes/material and canonical box transforms; design a separate same-top/material H4 support setup only after no-stair/no-obstacle-change audit',
        geometry_safety_and_body_response_planning_HOLDs_are_separate=True,
        current_H3_state_support_settings_and_all_PRIOR_PASS_FAIL_HOLDs_preserved=True,
        source_USD_or_live_stage_modified=False, any_floor_extension_applied=False,
        real_robot_required=False, CICS_runtime_integration=False, new_PhysX_executed=False,
        H4_static_or_drive_admission=False, use_for_final_statistics=False,
        input_sha256=pins)
    return report, occupied


def main():
    if TARGET.exists():
        saved = json.loads(TARGET.read_bytes()); verify(saved['input_sha256']); verify(saved['output_sha256'])
        print('Existing H4 complete-map/support review verified; not overwritten:', TARGET)
        return
    report, occupied = analyze()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(GRID, occupied=occupied, origin_xy_m=report['raster']['origin_xy_m'], resolution_m=report['raster']['resolution_m'])
    report['output_sha256'] = {str(GRID): sha(GRID)}
    with TARGET.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 ACTUAL COMPLETE COLLIDER MAP / FLOOR SUPPORT AUDIT ===')
    print('map cells changed / old obstacles removed:', report['original_candidate_grid_changed_cell_count'], report['old_obstacle_cells_removed'])
    print('endpoint obstacle clearance:', report['full_envelope_endpoint_obstacle_clear'])
    print('canonical floor Y maximum:', report['canonical_support']['world_aligned_bound']['maximum_m'][1])
    print('cached support hits / misses:', report['cached_endpoint_floor_hit_count'], report['cached_endpoint_floor_miss_count'])
    print('all misses outside canonical floor:', report['every_cached_miss_is_outside_current_canonical_floor'])
    print('STATUS:', report['status'])
    print('saved:', TARGET)
    print('No floor change, reset, Play, static/drive admission or final statistical trial.')


if __name__ == '__main__':
    main()
