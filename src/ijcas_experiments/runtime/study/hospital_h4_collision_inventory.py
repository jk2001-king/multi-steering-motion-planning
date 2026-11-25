"""Get-only H4 collider-coverage and cached-query preparation.

No current robot state, collider or material may be changed. Complete authored
inventory and sampled cached PhysX queries remain separate from static support,
path execution, or a continuous collision certificate.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
import math
from pathlib import Path

import hospital_h4_retained_review as retained


ROOT, BASE = retained.ROOT, retained.BASE
SETUP = retained.completion.final.GATE/'hospital_h4_setup/candidate_r01/hospital_h4_geometry_candidate.json'
PREFLIGHT = BASE/'inventory_preparation_r01/collision_inventory_preflight.json'
TARGET = BASE/'complete_inventory_r01/live_collision_inventory.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_review_ijcas_hospital_h4_collision_inventory.py')
VERSION = 'H4_all_stage_USD_collision_inventory_and_cached_queries_from_retained_H3_v1'


def intersects_roi(bound, roi):
    if bound is None:
        return False
    lo, hi = bound['minimum_m'], bound['maximum_m']
    if (len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in (*lo, *hi))
            or any(a > b for a, b in zip(lo, hi))):
        raise ValueError('Finite ordered collider bound required')
    return not (hi[0] < roi[0] or lo[0] > roi[1] or hi[1] < roi[2] or lo[1] > roi[3])


def inventory_summary(rows, unloaded, errors):
    if len({row['path'] for row in rows}) != len(rows):
        raise ValueError('Duplicate collider paths; no partial inventory promotion')
    roles = ('inactive', 'disabled', 'robot_self_excluded', 'outside_ROI', 'enabled_in_ROI', 'bound_unavailable')
    if any(row['classification'] not in roles for row in rows):
        raise ValueError('Every traversed collision prim needs an explicit classification')
    counts = {role: sum(row['classification'] == role for row in rows) for role in roles}
    return dict(collision_prim_count=len(rows), classification_counts=counts,
        no_visible_root_selection_filter=True, instance_proxies_traversed=True,
        active_unloaded_payload_paths=list(unloaded), errors=list(errors),
        authored_inventory_coverage_complete=not unloaded and not errors and counts['bound_unavailable'] == 0,
        cooked_shape_equivalence_or_continuous_floor_support_proven=False,
        physical_drive_admission=False)


def endpoint_samples(setup):
    length, width = setup['footprint']['length_m'], setup['footprint']['width_m']
    points = []
    for name in ('start_xyyaw', 'goal_xyyaw'):
        x, y, yaw = setup[name]
        for label, u, v in (('center', 0., 0.), ('front_left', length/2, width/2),
                ('front_right', length/2, -width/2), ('rear_left', -length/2, width/2),
                ('rear_right', -length/2, -width/2)):
            points.append(dict(task_pose=name, sample=label,
                xy_m=[x+math.cos(yaw)*u-math.sin(yaw)*v, y+math.sin(yaw)*u+math.cos(yaw)*v]))
    return points


def prepare():
    retained.load_preflight()
    live = json.loads(retained.TARGET.read_bytes())
    retained.completion.verify_files(live['input_sha256'])
    setup = json.loads(SETUP.read_bytes())
    retained.completion.verify_files(setup['input_sha256'])
    retained.completion.verify_files(setup['output_sha256'])
    if (setup['version'] != 'H4_authored_corridor_cart_geometry_task_candidate_v1'
            or not all(setup['checks'].values()) or setup['strict_map_or_final_task_drive_admission'] is not False
            or setup['task_pose_selection_uses_method_performance_or_measured_H3_endpoint'] is not False
            or live['root_session_source_timeline_robot_joints_and_settings_unchanged'] is not True):
        raise ValueError('Exact geometry-derived H4 candidate and actual retained H3 state required')
    raster = setup['raster']; origin = raster['origin_xy_m']; rows, cols = raster['shape_rows_columns']
    roi = [origin[0], origin[0]+cols*raster['resolution_m'], origin[1], origin[1]+rows*raster['resolution_m']]
    pins = {**setup['input_sha256'], **setup['output_sha256'], **live['input_sha256'],
        **{str(path): retained.completion.sha(path) for path in
            (SETUP, retained.TARGET, retained.PREFLIGHT, Path(__file__), ENTRY)}}
    return dict(version=VERSION, status='GET_ONLY_COMPLETE_COLLIDER_INVENTORY_PREPARED_NOT_DRIVE_ADMISSION',
        task_candidate=str(SETUP), task_candidate_sha256=retained.completion.sha(SETUP),
        review_roi_xy_m=roi, candidate_start_xyyaw=setup['start_xyyaw'], candidate_goal_xyyaw=setup['goal_xyyaw'],
        footprint=setup['footprint'], query_height_slab_m=setup['obstacle_height_slab_m'],
        prescribed_endpoint_ray_samples=endpoint_samples(setup),
        collision_query_positive_control='existing cart top triangle centroid; query its actual cooked collider without adding/moving anything',
        cached_PhysX_query_failure_is_NOT_permission_to_Play_or_initialize=True,
        all_stage_collision_prims_not_only_visible_roots=True,
        current_H3_FINAL_body_12_joints_and_settings_must_remain_retained=True,
        root_session_source_robot_and_timeline_writes_permitted=False,
        no_H4_static_or_motion_admitted=True, CICS_runtime_integration=False,
        use_for_final_statistics=False, input_sha256=pins,
        source_sha256={path.name: retained.completion.sha(path) for path in (Path(__file__), ENTRY)})


def load_preflight():
    report = json.loads(PREFLIGHT.read_bytes())
    retained.completion.verify_files(report['input_sha256'])
    retained.completion.verify_sources(report)
    if report != prepare():
        raise ValueError('Collider-inventory preflight changed; do not substitute a scene or reset robot')
    return report
