"""Preserve the failed H4 pilot and inspect a fresh source WITHOUT physics.

No replay, state restoration, contact exemption or changed collision flag.
The observed contact is authoritative; authored geometry can explain a map
coverage gap but cannot identify a cooked penetration, force or SDK margin.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hospital_geometry_tools import pose_collision
from hospital_h4_geometry_metrics import densify_poses, rectangle_aabb_distances


ROOT = _ijcas_workspace()
GATE = _ijcas_path(ROOT, 'studies/IJCAS/results/hospital_paper_gate')
SOURCE = _ijcas_path(ROOT, 'isaac/scenes/hospital_experiment.usda')
SOURCE_SHA = '9130af19a583e157b48137ed6ada13ac8540d1f26583f218bd3ed903906988a5'
ROBOT = '/World/patient_transport_h4_cohort_01'
GROUND = '/World/PatientTransportLabGround/CollisionBox'
RESULT = GATE/'hospital_h4_path_execution_v3/cohort_01_adopted_query_r01/hospital_h4_shared_v3_P_dynamics_full_envelope_r01.json'
INVENTORY = GATE/'hospital_h4_geometry_review/complete_inventory_r01/live_collision_inventory.json'
MAP_REVIEW = GATE/'hospital_h4_complete_map_support/review_r01/complete_map_support_review.json'
GRID = MAP_REVIEW.parent/'complete_authored_AABB_occupancy.npz'
CONTACT = '/World/Hospital/Geo_W_WallBase33/Geo_W_WallBase/Geo_W_WallBase'
REVIEW = GATE/'hospital_h4_restart_review/failure_r01/path_contact_review.json'
RESTART = GATE/'hospital_h4_restart_review/source_readback_r01/live_fresh_source.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_review_ijcas_hospital_h4_reopened_source.py')
VERSION = 'H4_saved_contact_failure_and_fresh_source_restart_inspection_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(pins):
    if not pins:
        raise ValueError('Nonempty frozen input pins required')
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Frozen evidence changed: '+str(path))


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def diagnose(raw, rows, plan, inventory, map_review, occupied, origin, resolution):
    """Pure recorded-data diagnosis; FAIL must never be promoted to PASS."""
    if (raw.get('verdict') != 'FAIL' or raw.get('path_executed') is not False
            or raw.get('path_execution_started') is not True
            or raw.get('original_motor_gains_restored') is not True
            or raw.get('contact_monitor_errors') != []
            or raw.get('terminal_handoff') is not None
            or raw.get('fixed_hold_attempt_count') != 0):
        raise ValueError('Exact incomplete contact-aborted pilot required, not a completed/terminal run')
    counts = raw['contact_report_counts_by_category']
    contacts = [p for p in raw['contact_pairs_by_category'] if p['category'] != 'wheel_support']
    if (counts != {'wheel_support': 53221, 'hospital_obstacle': 1}
            or len(contacts) != 1 or contacts[0]['category'] != 'hospital_obstacle'
            or contacts[0]['callback_count'] != 1 or CONTACT not in contacts[0]['pair']
            or ROBOT+'/front_active_drive_wheel_link/collisions/mesh_0/capsule' not in contacts[0]['pair']):
        raise ValueError('Recorded contact identity/count differs; do not substitute another failure')
    if not rows or len(rows) != raw['sample_count'] or any(r['phase'] not in ('pre_hold', 'path_tracking') for r in rows):
        raise ValueError('Complete PRE-terminal trace required')
    keys = ('time_s', 'x_m', 'y_m', 'yaw_rad', 'goal_position_error_m',
        'goal_heading_error_deg', 'path_progress_m', 'path_error_m', 'path_heading_error_deg',
        'command_speed_m_s', 'vx_physx_m_s', 'vy_physx_m_s')
    if not all(math.isfinite(float(r[k])) for r in rows for k in keys):
        raise ValueError('Finite raw samples required')
    if any(float(b['time_s']) <= float(a['time_s']) for a, b in zip(rows, rows[1:])):
        raise ValueError('Raw timestamps must increase strictly')
    if raw['goal_world_xyyaw'] != plan['goal_world_xyyaw'] or raw['start_world_xyyaw'] != plan['start_world_xyyaw']:
        raise ValueError('Original task poses changed')
    goal = plan['goal_world_xyyaw']
    for row in rows:
        error = math.hypot(float(row['x_m'])-goal[0], float(row['y_m'])-goal[1])
        angle = math.degrees(math.atan2(math.sin(goal[2]-float(row['yaw_rad'])),
            math.cos(goal[2]-float(row['yaw_rad']))))
        if abs(error-float(row['goal_position_error_m'])) > 1e-9 or abs(angle-float(row['goal_heading_error_deg'])) > 1e-9:
            raise ValueError('Raw pose and saved goal-error columns disagree')
    colliders = [r for r in inventory if r['path'] == CONTACT]
    omitted = [r for r in map_review['excluded_by_height'] if r['path'] == CONTACT]
    if (len(colliders) != 1 or colliders[0]['classification'] != 'enabled_in_ROI'
            or colliders[0]['collision_enabled_value'] is not True or len(omitted) != 1
            or omitted[0]['reason'] != 'outside_original_height_slab'):
        raise ValueError('The original inventory and original map exclusion must agree')
    collider = colliders[0]
    bound, slab = collider['world_aligned_bound'], map_review['obstacle_height_slab_m']
    if bound['maximum_m'][2] >= slab[0]:
        raise ValueError('Recorded collider is not below the lower map slab')
    poses = [[float(r[k]) for k in ('x_m', 'y_m', 'yaw_rad')] for r in rows]
    final = raw['final_actual_world_xyyaw']
    if len(final) != 3 or not all(math.isfinite(v) for v in final):
        raise ValueError('Finite final actual readback required')
    expanded = rectangle_aabb_distances(poses, bound, clearance=.010)
    overlap_indices = np.flatnonzero(expanded == 0.)
    sampled = densify_poses([*poses, final])
    raster_intersections = sum(pose_collision(occupied, p, origin, resolution, clearance=.010) for p in sampled)
    planned = densify_poses([plan['start_world_xyyaw']]+[s['world_xyyaw'] for s in plan['selected_states']])
    planned_overlap = rectangle_aabb_distances(planned, bound, clearance=.010) == 0.
    tracking = [r for r in rows if r['phase'] == 'path_tracking']
    return dict(
        raw_verdict='FAIL', raw_error=raw['error'], raw_failed_checks=[k for k, ok in raw['checks'].items() if not ok],
        contact_pair=contacts[0], original_contact_collider=collider,
        original_map_height_slab_m=slab, original_map_exclusion=omitted[0],
        collider_authored_top_m=bound['maximum_m'][2],
        height_gap_to_original_map_lower_slab_m=slab[0]-bound['maximum_m'][2],
        all_recorded_pose_samples=len(rows), last_recorded_simulation_time_s=float(rows[-1]['time_s']),
        final_actual_world_xyyaw=final, last_recorded_state=raw['last_recorded_state'],
        recorded_path_progress_m=raw['path_progress_m'], reference_polyline_length_m=raw['reference_polyline_length_m'],
        recorded_path_progress_fraction=raw['path_progress_m']/raw['reference_polyline_length_m'],
        terminal_correction_or_fixed_hold_reached=False,
        terminal_position_error_m=None, terminal_heading_error_deg=None,
        tracking_maximum_recorded_path_error_m=max(float(r['path_error_m']) for r in tracking),
        tracking_maximum_recorded_heading_error_deg=max(abs(float(r['path_heading_error_deg'])) for r in tracking),
        strict_original_raster_replayed_at_recorded_and_interpolated_poses=dict(
            pose_sample_count=len(sampled), intersecting_pose_samples=int(raster_intersections),
            sample_spacing_m_deg=[.005, .25], full_footprint_m=[1.630, .965], clearance_m=.010),
        body_XY_projection_intersects_low_collider_AABB_samples=int(len(overlap_indices)),
        first_recorded_XY_projection_overlap_simulation_s=(float(rows[int(overlap_indices[0])]['time_s']) if len(overlap_indices) else None),
        frozen_native_plan_XY_projection_intersects_same_low_collider=bool(np.any(planned_overlap)),
        native_plan_XY_projection_intersecting_samples=int(np.count_nonzero(planned_overlap)),
        XY_projection_overlap_is_NOT_physical_penetration_or_contact_time=True,
        last_saved_row_precedes_guard_rejected_sample=True,
        precise_first_contact_time_force_penetration_and_cooked_margin_NOT_recorded=True,
        conclusion='Actual wheel/obstacle contact exposes a wheel-floor-height domain outside the frozen 6-450mm obstacle projection. The old raster cannot certify absence of this contact.',
        no_obstacle_disabled_filtered_relabelled_as_support_or_exempted=True,
        next_safe_scope='Fresh source authored-collider/configuration inspection; no stale cached query or actual physical-state restoration.',
        new_model_fit_or_control_change=False, actual_drive_admission=False,
        use_for_final_statistics=False, qualified_final_comparison_cells=0,
        source_stage_or_PhysX_modified=False)


def build():
    raw = json.loads(RESULT.read_bytes())
    verify(raw['input_sha256'])
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Original source bytes changed; do not reopen a modified scene as the baseline')
    trace = RESULT.with_suffix('.csv')
    if sha(trace) != raw['trace_sha256'] or sha(raw['plan_path']) != raw['plan_sha256']:
        raise ValueError('Failed pilot trace or frozen plan changed')
    for name, expected in raw['source_sha256'].items():
        path = _ijcas_path(ROOT, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts', name)
        if sha(path) != expected:
            raise ValueError('Runtime source changed: '+name)
    inventory = json.loads(INVENTORY.read_bytes()); verify(inventory['input_sha256'])
    mapping = json.loads(MAP_REVIEW.read_bytes()); verify(mapping['input_sha256']); verify(mapping['output_sha256'])
    plan = json.loads(Path(raw['plan_path']).read_bytes())
    with trace.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(GRID) as grid:
        occupied, origin, resolution = grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m'])
        if hashlib.sha256(occupied.tobytes()).hexdigest() != raw['planning_occupancy_digest']:
            raise ValueError('Original strict raster changed')
        report = diagnose(raw, rows, plan, inventory['collision_prims'], mapping, occupied, origin, resolution)
    static = json.loads(Path(raw['static_gate_reference']).read_bytes())
    if static['verdict'] != 'FAIL' or raw['original_combined_static_raw_verdict'] != 'FAIL':
        raise ValueError('Original static FAIL must stay preserved, not relabelled')
    paths = (Path(__file__), ENTRY, _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_geometry_metrics.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_geometry_tools.py'), RESULT, trace, INVENTORY, MAP_REVIEW, GRID)
    pins = {**raw['input_sha256'], **{str(p): sha(p) for p in paths}}
    verify(pins)
    return dict(version=VERSION, status='RECORDED_H4_WHEEL_CONTACT_AND_MAP_HEIGHT_DOMAIN_GAP_NO_RETRY_ADMISSION',
        original_result=str(RESULT), original_result_sha256=sha(RESULT), original_static_verdict='FAIL',
        prior_actual_H3_H4_records_preserved=True,
        independent_fresh_frozen_whole_H3_repeats=0,
        CICS_runtime_integration=False, real_robot_required=False, **report, input_sha256=pins)


def load_review():
    data = json.loads(REVIEW.read_bytes()); verify(data['input_sha256'])
    if data['version'] != VERSION or data['raw_verdict'] != 'FAIL' or data['actual_drive_admission'] is not False:
        raise ValueError('Exact saved offline contact review required')
    return data


def inspect_fresh_source(stage, playing, stopped, current_time):
    """Get-only USD inspection. Does not touch a PhysicsScene/tensor/target."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if stage is None or playing or not stopped or current_time != 0.:
        raise ValueError('Fresh original source at STOPPED time zero required; do not Play or reset to satisfy this')
    if Path(stage.GetRootLayer().realPath).resolve() != SOURCE.resolve() or sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Open the exact unchanged hospital_experiment.usda source')
    for path in (ROBOT, '/World/IJCAS_HospitalH4Cohort01', '/World/IJCAS_HospitalH4SupportR01'):
        if stage.GetPrimAtPath(path).IsValid():
            raise ValueError('Prior H4 Session actor/support/marker present; this is not an untouched reopened source')
    hospital = stage.GetPrimAtPath('/World/Hospital')
    if not hospital.IsValid() or not hospital.IsActive() or not hospital.IsLoaded():
        raise ValueError('Hospital asset is not loaded; do not replace it with a proxy')
    data = json.loads(INVENTORY.read_bytes()); verify(data['input_sha256'])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)
    comparisons = []
    for saved in data['collision_prims']:
        if saved['classification'] not in ('enabled_in_ROI', 'disabled') or saved['path'] == GROUND:
            continue
        prim = stage.GetPrimAtPath(saved['path'])
        observed = dict(path=saved['path'], recorded_classification=saved['classification'],
            valid=bool(prim.IsValid()), active=bool(prim.IsValid() and prim.IsActive()))
        if prim.IsValid() and prim.IsActive():
            observed['collision_enabled'] = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() if prim.HasAPI(UsdPhysics.CollisionAPI) else None
            bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            observed['world_aligned_bound'] = dict(minimum_m=list(map(float, bound.GetMin())), maximum_m=list(map(float, bound.GetMax())))
            observed['bound_matches_recorded_20micrometre_check'] = all(np.allclose(
                observed['world_aligned_bound'][key], saved['world_aligned_bound'][key], rtol=0, atol=2e-5)
                for key in ('minimum_m', 'maximum_m'))
            if saved['path'] == CONTACT and prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim); matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                points = [list(map(float, matrix.Transform(Gf.Vec3d(*p)))) for p in mesh.GetPointsAttr().Get()]
                old = saved['authored_mesh']
                observed['contact_mesh_topology_and_world_points_match'] = bool(
                    list(mesh.GetFaceVertexCountsAttr().Get()) == old['face_vertex_counts']
                    and list(mesh.GetFaceVertexIndicesAttr().Get()) == old['face_vertex_indices']
                    and np.shape(points) == np.shape(old['world_points_m'])
                    and np.allclose(points, old['world_points_m'], rtol=0, atol=2e-5))
                observed['authored_contact_rest_offsets'] = {name: prim.GetAttribute(name).Get()
                    for name in ('physxCollision:contactOffset', 'physxCollision:restOffset') if prim.GetAttribute(name)}
        comparisons.append(observed)
    expected = [r for r in comparisons if r['recorded_classification'] == 'enabled_in_ROI']
    hit = next(r for r in comparisons if r['path'] == CONTACT)
    checks = dict(original_source_file_unchanged=True, new_session_not_an_actual_state_restore=True,
        every_recorded_enabled_Hospital_ROI_collider_available=bool(expected) and all(
            r['valid'] and r['active'] and r.get('collision_enabled') is not False for r in expected),
        every_recorded_enabled_Hospital_ROI_bound_unchanged=bool(expected) and all(
            r.get('bound_matches_recorded_20micrometre_check') is True for r in expected),
        contact_collider_authored_mesh_matches_recorded=hit.get('contact_mesh_topology_and_world_points_match') is True)
    return dict(checks=checks, collision_comparisons=comparisons, contact_collider_readback=hit,
        existing_rigid_body_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)],
        existing_graph_paths=[str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'OmniGraph'],
        old_solver_contact_cache_or_physical_state_restored=False,
        cooked_collision_queries_attempted=False, runtime_physics_initialized=False,
        new_H4_robot_or_support_staged=False, actual_static_or_drive_admission=False,
        actual_motion_executed=False, use_for_final_statistics=False)


def main():
    if REVIEW.exists():
        report = load_review()
    else:
        report = build(); write_exclusive(REVIEW, report)
    print('=== H4 RECORDED PATH CONTACT / RESTART REVIEW (OFFLINE) ===')
    print('original verdict:', report['raw_verdict'], '; contact:', CONTACT)
    print('progress m:', report['recorded_path_progress_m'], '/', report['reference_polyline_length_m'])
    print('collider authored top / map lower slab mm:', 1000*report['collider_authored_top_m'], 1000*report['original_map_height_slab_m'][0])
    print('old raster intersecting saved/interpolated poses:', report['strict_original_raster_replayed_at_recorded_and_interpolated_poses']['intersecting_pose_samples'])
    print('status:', report['status']); print('saved:', REVIEW)
    print('No retry, physics, state restore, collider/material/gain change or failure relabeling.')


if __name__ == '__main__':
    main()
