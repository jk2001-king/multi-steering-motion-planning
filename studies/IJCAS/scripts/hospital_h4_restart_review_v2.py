"""Get-only restart reader: disabled inventory entries have NO saved bounds.

Keep the pinned v1 source/result unchanged. Enabled ROI colliders still need
their original finite bounds, enabled flags and 20 micrometre comparison.
For previously disabled colliders, record current flags/bounds as NEW data;
never invent old geometry or claim that the unprepared scene is drive-ready.
"""

import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_restart_review as original


ROOT, GATE = original.ROOT, original.GATE
SOURCE, SOURCE_SHA = original.SOURCE, original.SOURCE_SHA
ROBOT, GROUND, CONTACT, INVENTORY = original.ROBOT, original.GROUND, original.CONTACT, original.INVENTORY
sha, verify, write_exclusive = original.sha, original.verify, original.write_exclusive
VERSION = 'H4_fresh_source_reader_disabled_inventory_has_no_historical_bounds_v2'
OUTPUT = GATE/'hospital_h4_restart_review/source_readback_v2_r01'
PROTOCOL, RESTART = OUTPUT/'inspection_protocol.json', OUTPUT/'live_fresh_source.json'
ENTRY = ROOT/'isaac/scripts/isaac_review_ijcas_hospital_h4_reopened_source_v2.py'


def compare_bound(saved, live_bound):
    """None means NOT RECORDED, never equality success for missing evidence."""
    if saved['classification'] == 'disabled':
        return dict(historical_bound_comparison_applicable=False,
            historical_bound_comparison_status='NOT_RECORDED_FOR_DISABLED_COLLIDER',
            bound_matches_recorded_20micrometre_check=None)
    if saved['classification'] != 'enabled_in_ROI':
        raise ValueError('Only enabled ROI or explicitly disabled collider rows are in scope')
    if 'world_aligned_bound' not in saved:
        raise ValueError('Enabled collider is missing REQUIRED historical bounds: '+saved['path'])
    expected = saved['world_aligned_bound']
    for key in ('minimum_m', 'maximum_m'):
        if len(expected[key]) != 3 or not all(math.isfinite(v) for v in expected[key]):
            raise ValueError('Enabled collider historical bounds must be finite three-vectors')
    if any(a > b for a, b in zip(expected['minimum_m'], expected['maximum_m'])):
        raise ValueError('Enabled collider historical bounds are not ordered')
    match = live_bound is not None and all(
        len(live_bound[key]) == 3 and all(math.isfinite(v) for v in live_bound[key])
        and np.allclose(live_bound[key], expected[key], rtol=0, atol=2e-5)
        for key in ('minimum_m', 'maximum_m'))
    return dict(historical_bound_comparison_applicable=True,
        historical_bound_comparison_status='MATCH' if match else 'MISSING_OR_DIFFERENT',
        bound_matches_recorded_20micrometre_check=bool(match))


def prepare_protocol():
    previous = original.load_review()
    inventory = json.loads(INVENTORY.read_bytes()); verify(inventory['input_sha256'])
    disabled = [r for r in inventory['collision_prims'] if r['classification'] == 'disabled']
    if len(disabled) != 10 or any('world_aligned_bound' in row for row in disabled):
        raise ValueError('Expected the original ten disabled entries with no recorded bounds')
    paths = (Path(__file__), ENTRY, original.REVIEW)
    return dict(version=VERSION, status='GET_ONLY_READER_REPAIR_PREPARED_NO_SETUP_OR_MOTION',
        original_failure_review=str(original.REVIEW),
        v1_reader_preserved=str(Path(original.__file__)),
        user_reported_v1_exception="KeyError: 'world_aligned_bound' at disabled inventory comparison",
        user_error_log_is_NOT_a_new_physical_failure_or_new_validation=True,
        disabled_collider_historical_bound_not_available_count=len(disabled),
        enabled_collider_bounds_flags_and_tolerance_NOT_relaxed=True,
        disabled_live_geometry_recording_is_NOT_a_claim_of_historical_equality=True,
        actual_source_inspection_completed=False, actual_static_or_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**previous['input_sha256'], **{str(p): sha(p) for p in paths}})


def load_protocol():
    protocol = json.loads(PROTOCOL.read_bytes()); verify(protocol['input_sha256'])
    if protocol != prepare_protocol():
        raise ValueError('Exact read-only v2 protocol changed')
    return protocol


def inspect_fresh_source(stage, playing, stopped, current_time):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if stage is None or playing or not stopped or current_time != 0.:
        raise ValueError('Untouched reopened source must be STOPPED at time zero; no reset/Play to create it')
    if Path(stage.GetRootLayer().realPath).resolve() != SOURCE.resolve() or sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Open the unchanged hospital_experiment.usda source')
    for path in (ROBOT, '/World/IJCAS_HospitalH4Cohort01', '/World/IJCAS_HospitalH4SupportR01'):
        if stage.GetPrimAtPath(path).IsValid():
            raise ValueError('Old H4 Session actor/support/marker is present; not a fresh source')
    hospital = stage.GetPrimAtPath('/World/Hospital')
    if not hospital.IsValid() or not hospital.IsActive() or not hospital.IsLoaded():
        raise ValueError('Original Hospital must be loaded; no replacement asset or proxy')
    inventory = json.loads(INVENTORY.read_bytes()); verify(inventory['input_sha256'])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)
    comparisons = []
    for saved in inventory['collision_prims']:
        if saved['classification'] not in ('enabled_in_ROI', 'disabled') or saved['path'] == GROUND:
            continue
        prim = stage.GetPrimAtPath(saved['path'])
        row = dict(path=saved['path'], recorded_classification=saved['classification'],
            valid=bool(prim.IsValid()), active=bool(prim.IsValid() and prim.IsActive()))
        bound = None
        if row['active']:
            row['collision_enabled'] = (UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if prim.HasAPI(UsdPhysics.CollisionAPI) else None)
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            values = list(map(float, box.GetMin())), list(map(float, box.GetMax()))
            if not box.IsEmpty() and all(math.isfinite(v) for vector in values for v in vector):
                bound = dict(minimum_m=values[0], maximum_m=values[1])
            row['world_aligned_bound'] = bound
            if saved['path'] == CONTACT and prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim); matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                points = [list(map(float, matrix.Transform(Gf.Vec3d(*p)))) for p in mesh.GetPointsAttr().Get()]
                old = saved['authored_mesh']
                row['contact_mesh_topology_and_world_points_match'] = bool(
                    list(mesh.GetFaceVertexCountsAttr().Get()) == old['face_vertex_counts']
                    and list(mesh.GetFaceVertexIndicesAttr().Get()) == old['face_vertex_indices']
                    and np.shape(points) == np.shape(old['world_points_m'])
                    and np.allclose(points, old['world_points_m'], rtol=0, atol=2e-5))
                row['authored_contact_rest_offsets'] = {name: prim.GetAttribute(name).Get()
                    for name in ('physxCollision:contactOffset', 'physxCollision:restOffset') if prim.GetAttribute(name)}
        row.update(compare_bound(saved, bound))
        comparisons.append(row)
    enabled = [r for r in comparisons if r['recorded_classification'] == 'enabled_in_ROI']
    disabled = [r for r in comparisons if r['recorded_classification'] == 'disabled']
    hit = next(r for r in comparisons if r['path'] == CONTACT)
    checks = dict(original_source_file_unchanged=True, new_session_not_an_actual_state_restore=True,
        every_recorded_enabled_Hospital_ROI_collider_available=bool(enabled) and all(
            r['valid'] and r['active'] and r.get('collision_enabled') is True for r in enabled),
        every_recorded_enabled_Hospital_ROI_bound_unchanged=bool(enabled) and all(
            r['bound_matches_recorded_20micrometre_check'] is True for r in enabled),
        contact_collider_authored_mesh_matches_recorded=hit.get('contact_mesh_topology_and_world_points_match') is True,
        all_ten_previously_disabled_collider_flags_recorded_without_invented_old_bounds=len(disabled) == 10 and all(
            r['historical_bound_comparison_applicable'] is False
            and r['bound_matches_recorded_20micrometre_check'] is None for r in disabled))
    return dict(checks=checks, collision_comparisons=comparisons, contact_collider_readback=hit,
        previously_disabled_colliders_currently_enabled=[r['path'] for r in disabled
            if r['active'] and r.get('collision_enabled') is True],
        disabled_collider_flags_must_be_bound_in_future_explicit_Session_setup=True,
        disabled_bounds_missing_is_NOT_a_geometry_equality_PASS=True,
        existing_rigid_body_paths=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)],
        existing_graph_paths=[str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == 'OmniGraph'],
        old_solver_contact_cache_or_physical_state_restored=False,
        cooked_collision_queries_attempted=False, runtime_physics_initialized=False,
        new_H4_robot_or_support_staged=False, actual_static_or_drive_admission=False,
        actual_motion_executed=False, use_for_final_statistics=False)


def main():
    report = prepare_protocol()
    if PROTOCOL.exists():
        load_protocol()
    else:
        write_exclusive(PROTOCOL, report)
    print('=== H4 READ-ONLY RESTART READER V2 PREPARED ===')
    print('original disabled collider rows without old bounds:', report['disabled_collider_historical_bound_not_available_count'])
    print('original enabled geometry/flags/20-micrometre check retained:', True)
    print('saved:', PROTOCOL)
    print('No live scene access, setup, physics or drive admission.')


if __name__ == '__main__':
    main()
