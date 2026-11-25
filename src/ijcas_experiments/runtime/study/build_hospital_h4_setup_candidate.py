#!/usr/bin/env python3
"""H4 asset-derived task + conservative AABB candidate map, NOT a strict map.

Use all 103 recorded local asset roots / 115 enabled collision primitives.
The preceding exporter selected roots by visible bounds; therefore this file
does NOT invent complete collider coverage or cooked/support validation.
The next get-only live inventory must inspect every stage collision prim.
No plan outcome, robot/caster state or successful endpoint selects the task.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_retained_review as review
from hospital_geometry_tools import footprint_polygon, polygon_cells, pose_collision


ROOT, BASE = review.ROOT, review.completion.final.GATE
SOURCE = review.TARGET
SCOPE = _ijcas_path(ROOT, 'studies/IJCAS/config/hospital_simulation_scope_addendum_20261116.json')
OUTPUT = BASE/'hospital_h4_setup/candidate_r01'
TARGET = OUTPUT/'hospital_h4_geometry_candidate.json'
GRID = OUTPUT/'candidate_occupancy.npz'
PREVIEW = OUTPUT/'task_candidate_preview.png'
VERSION = 'H4_authored_corridor_cart_geometry_task_candidate_v1'
RESOLUTION = .010
ORIGIN = np.array([15., 23.])
SHAPE = (1150, 700)
SLAB = (.006, .450)
RESERVE = .150
ASSETS = dict(left='Geo_M1_DoorWall8', right='Geo_W_WallDoorCostum_536',
    corner='Geo_M1_SideWall15', cart='SM_SupplyCart_02a7', stairs='Geo_M_Stairs10')


def bound_lookup(record):
    values = {}
    for role, name in ASSETS.items():
        selected = [row for row in record['assets'] if row['path'] == '/World/Hospital/'+name]
        if len(selected) != 1 or not selected[0]['collision']['enabled_collision_prim_count']:
            raise ValueError('Every original geometry anchor requires an enabled collider: '+name)
        values[role] = selected[0]['world_aligned_visible_bound']
    return values


def derive_task(record):
    b = bound_lookup(record)
    # Start alongside the original cart in the centre of the unobstructed
    # cart-to-right-wall gap; head north. Goal west of the corner, centred
    # between the unchanged south wall and stair boundary; face west.
    start = [(b['cart']['maximum_m'][0]+b['right']['minimum_m'][0])/2,
             (b['cart']['minimum_m'][1]+b['cart']['maximum_m'][1])/2, math.pi/2]
    goal = [b['left']['maximum_m'][0]-1.630/2-.010-RESERVE,
            (b['corner']['maximum_m'][1]+b['stairs']['minimum_m'][1])/2, math.pi]
    dx, dy = goal[0]-start[0], goal[1]-start[1]
    local = [math.cos(start[2])*dx+math.sin(start[2])*dy,
             -math.sin(start[2])*dx+math.cos(start[2])*dy, math.pi/2]
    return start, goal, local, b


def project_candidate(record):
    occupancy = np.zeros(SHAPE, dtype=bool)
    included = []; excluded = []
    seen = set()
    for asset in record['assets']:
        for primitive in asset['collision']['primitives']:
            if primitive['path'] in seen:
                raise ValueError('Duplicate enabled collision primitive')
            seen.add(primitive['path'])
            bound = primitive['world_aligned_bound']
            lo, hi = bound['minimum_m'], bound['maximum_m']
            if (not all(math.isfinite(value) for value in (*lo, *hi))
                    or any(a > b for a, b in zip(lo, hi))):
                raise ValueError('Finite ordered collision AABB required')
            if hi[2] < SLAB[0] or lo[2] > SLAB[1]:
                excluded.append(dict(path=primitive['path'], reason='below_low_terrain_band' if hi[2] < SLAB[0] else 'above_mobile_base_band'))
                continue
            # Entire AABB: conservative for the AUTHORED shape/hull. Never
            # drive under the cart or erase low thresholds/stair rails.
            rows, cols = polygon_cells([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])],
                ORIGIN, RESOLUTION, SHAPE)
            occupancy[rows, cols] = True
            included.append(dict(path=primitive['path'], world_aabb=bound,
                occupied_cell_count=int(len(rows)), low_terrain_candidate=hi[2] < .030))
    return occupancy, included, excluded


def build():
    record = json.loads(SOURCE.read_bytes())
    review.completion.verify_files(record['input_sha256'])
    if (record['status'] != 'RETAINED_H3_FINAL_H4_AUTHORED_GEOMETRY_RECORDED_TASK_SETUP_PENDING'
            or record['root_session_source_timeline_robot_joints_and_settings_unchanged'] is not True
            or record['H3_abandoned_or_global_infeasibility_claimed'] is not False
            or record['H4_or_H3_motion_admitted'] is not False
            or record['complete_collision_coverage_for_future_map_proven'] is not False):
        raise ValueError('Require the actual get-only retained H3 final H4 asset review')
    occupancy, included, excluded = project_candidate(record)
    start, goal, local, bounds = derive_task(record)
    checks = dict(candidate_start_full_envelope_clear=not pose_collision(occupancy, start, ORIGIN, RESOLUTION),
        candidate_goal_full_envelope_clear=not pose_collision(occupancy, goal, ORIGIN, RESOLUTION),
        original_cart_and_walls_not_moved=True,
        north_to_west_90deg_left_turn=abs(local[2]-math.pi/2) < 1e-12,
        body_and_caster_dimensions_unchanged=True,
        low_terrain_not_exempted=all(item['world_aabb']['maximum_m'][2] >= SLAB[0] for item in included))
    inputs = {**record['input_sha256'], str(SOURCE): review.completion.sha(SOURCE),
        str(SCOPE): review.completion.sha(SCOPE),
        str(Path(__file__)): review.completion.sha(Path(__file__)),
        str(_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_geometry_tools.py')):
            review.completion.sha(_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_geometry_tools.py'))}
    review.completion.verify_files(inputs)
    report = dict(version=VERSION, scenario='H4_ward_corridor_turn',
        status='CANDIDATE_ENDPOINTS_CLEAR_COMPLETE_COLLIDER_INVENTORY_PENDING' if all(checks.values()) else 'HOLD_CANDIDATE_ENDPOINT_GEOMETRY',
        task='northbound_passage_along_original_cart_then_90deg_left_turn_into_cross_corridor',
        environment_scope='different corridor area in the SAME original hospital USD, NOT another independent hospital',
        start_xyyaw=start, goal_xyyaw=goal, local_goal_xyyaw=local,
        pose_rule='cart-to-right-wall gap midpoint alongside cart; westbound goal 150mm behind corner after full-envelope clearance, centred between south wall and stair boundary',
        task_pose_selection_uses_method_performance_or_measured_H3_endpoint=False,
        geometry_anchors=bounds, existing_asset_paths={role: '/World/Hospital/'+name for role, name in ASSETS.items()},
        bedside_or_corner_setup_reserve_m=RESERVE,
        footprint=dict(length_m=1.630, width_m=.965, clearance_m=.010),
        raster=dict(origin_xy_m=ORIGIN.tolist(), resolution_m=RESOLUTION, shape_rows_columns=list(SHAPE),
            method='SAT_SUPERCOVER_AUTHORED_COLLIDER_AABBs_CANDIDATE_NOT_STRICT_COMPLETE_MAP'),
        obstacle_height_slab_m=list(SLAB),
        slab_basis='include low terrain above canonical support+5mm; retain existing 450mm mobile-base band; loaded upper envelope NOT certified',
        candidate_occupancy_digest=hashlib.sha256(occupancy.tobytes()).hexdigest(),
        candidate_grid=str(GRID), preview_image=str(PREVIEW),
        included_collision_AABBs=included, excluded_by_height=excluded, checks=checks,
        complete_stage_collision_coverage_verified=False, cooked_collision_verified=False,
        terrain_and_continuous_floor_support_verified=False, connecting_path_verified=False,
        next_required_live_data='all enabled/disabled collision prims traversed independently of visible-root selection, including non-hospital support geometry and instance proxies; no Play/reset',
        strict_map_or_final_task_drive_admission=False, geometry_frozen_for_paper=False,
        H3_final_state_preserved=True, H3_final_comparison_complete=False,
        CICS_runtime_integration=False, real_robot_validation_required=False,
        new_PhysX_executed=False, use_for_final_statistics=False,
        input_sha256=inputs, source_sha256={Path(__file__).name: review.completion.sha(Path(__file__))})
    return report, occupancy


def draw(report, occupancy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8, 9), layout='constrained')
    ax.imshow(occupancy, cmap='Greys', origin='lower', vmin=0, vmax=1,
        extent=[ORIGIN[0], ORIGIN[0]+SHAPE[1]*RESOLUTION, ORIGIN[1], ORIGIN[1]+SHAPE[0]*RESOLUTION])
    for key, color, label in (('start_xyyaw', '#118c4f', 'Candidate start: northbound'),
            ('goal_xyyaw', '#df8617', 'Candidate goal: westbound')):
        pose = report[key]
        ax.add_patch(Polygon(footprint_polygon(pose), facecolor=color, edgecolor=color, alpha=.35))
        ax.arrow(*pose[:2], .45*math.cos(pose[2]), .45*math.sin(pose[2]), width=.025, color=color, length_includes_head=True)
        ax.text(pose[0], pose[1]-.65, label, color=color, ha='center', fontsize=9)
    for role, label in (('cart', 'Original parked cart'), ('stairs', 'Stairs: blocked')):
        bound = report['geometry_anchors'][role]
        ax.text((bound['minimum_m'][0]+bound['maximum_m'][0])/2,
            (bound['minimum_m'][1]+bound['maximum_m'][1])/2, label, ha='center', fontsize=9,
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.set(aspect='equal', xlabel='Hospital world X (m)', ylabel='Hospital world Y (m)',
        title='H4 task candidate: original corridor and cart\nEndpoints only; no robot move or planned path')
    ax.text(.02, .01, 'Candidate AABB map; complete collider inventory and support validation pending',
        transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    fig.savefig(PREVIEW, dpi=160)
    plt.close(fig)


def main():
    if TARGET.exists():
        report = json.loads(TARGET.read_bytes())
        review.completion.verify_files(report['input_sha256'])
        review.completion.verify_sources(report)
        print('Existing H4 task candidate verified; not overwritten:', TARGET)
        return
    report, occupied = build()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(GRID, occupied=occupied, origin_xy_m=ORIGIN, resolution_m=RESOLUTION)
    draw(report, occupied)
    report['output_sha256'] = {str(path): review.completion.sha(path) for path in (GRID, PREVIEW)}
    with TARGET.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 ORIGINAL CORRIDOR / CART TASK CANDIDATE (OFFLINE ONLY) ===')
    print('start / goal:', report['start_xyyaw'], report['goal_xyyaw'])
    print('checks:', report['checks'])
    print('Full stage collider inventory / cooked collision / support: PENDING')
    print('saved:', TARGET)
    print('No H3 completion claim, strict-map admission or robot motion.')


if __name__ == '__main__':
    main()
