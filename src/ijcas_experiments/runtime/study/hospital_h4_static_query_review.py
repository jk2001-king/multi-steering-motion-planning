"""Diagnose the actual FAIL without changing any gate, query point or collider.

All ray hits were already recorded. Floor-under-an-obstacle is NOT traversable
free space. Separate observations, retain the original FAIL, admit NO motion.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_completed_stop_path_launch as launch
from hospital_geometry_tools import footprint_polygon


static, cohort = launch.stationary, launch.preparation
OUTPUT = cohort.GATE/'hospital_h4_static_query_review/review_r01'
TARGET, LIVE = OUTPUT/'static_query_review.json', OUTPUT/'live_failed_static_query_review.json'
ENTRY = _ijcas_path(cohort.ROOT, 'isaac/scripts/isaac_review_ijcas_hospital_h4_failed_static_queries.py')
VERSION = 'H4_actual_failed_static_all_ray_hits_map_and_envelope_diagnosis_v1'


def point_cell(point, occupied, origin, resolution):
    if len(point) != 2 or not np.isfinite(point).all():
        raise ValueError('Finite query XY required')
    col, row = np.floor((np.asarray(point)-origin)/resolution).astype(int)
    inside = 0 <= row < occupied.shape[0] and 0 <= col < occupied.shape[1]
    return dict(row=int(row), column=int(col), inside_map=bool(inside),
        occupied=bool(occupied[row, col]) if inside else None)


def hit_is_support(hit, spec, top):
    limits = static.QUERY_LIMITS
    pos, normal = hit.get('position_m', []), hit.get('normal', [])
    return bool(hit.get('collision') in spec['allowed_supports'] and len(pos) == 3 and len(normal) == 3
        and np.isfinite([*pos, *normal]).all()
        and np.allclose(pos[:2], spec['xy_m'], rtol=0, atol=limits['horizontal_tolerance_m'])
        and abs(pos[2]-top) <= limits['height_tolerance_m'] and normal[2] >= limits['minimum_upward_normal_z'])


def diagnose_queries(specifications, observations, top, inventory, occupied, origin, resolution):
    """Do not skip occluders, self-unrelated obstacles, unknown hits or misses."""
    expected = {s['id']: s for s in specifications}
    if len(observations) != len(expected) or {r.get('id') for r in observations} != set(expected):
        raise ValueError('All exact original query identities required, with no duplicates')
    colliders = {r['path']: r for r in inventory if r['classification'] == 'enabled_in_ROI'}
    rows = []
    for observed in observations:
        spec = expected[observed['id']]
        hits = observed.get('all_nonself_hits', [])
        complete = bool(observed.get('hit') and not observed.get('errors') and hits)
        supports = [hit for hit in hits if hit_is_support(hit, spec, top)]
        occluders = []
        for hit in hits:
            if hit.get('collision') in static.SUPPORTS:
                continue
            pos = hit.get('position_m', [])
            record = colliders.get(hit.get('collision'))
            finite = len(pos) == 3 and bool(np.isfinite(pos).all())
            bound = record['world_aligned_bound'] if record else None
            mapped = bool(record and record['collision_enabled_value'] is not False and finite
                and all(bound['minimum_m'][i]-.0005 <= pos[i] <= bound['maximum_m'][i]+.0005 for i in range(3)))
            cell = point_cell(pos[:2], occupied, origin, resolution) if finite else None
            occluders.append(dict(collision=hit.get('collision'), position_m=pos,
                is_original_enabled_inventory_collider=mapped, raster_cell=cell,
                mapped_blocked=bool(mapped and cell and cell['inside_map'] and cell['occupied'])))
        original = static.query_checks([spec], [observed], top)
        rows.append(dict(id=spec['id'], xy_m=spec['xy_m'], original_closest_hit_check=all(original.values()),
            original_collision=observed.get('collision'), original_height_m=observed.get('position_m', [None]*3)[2],
            original_height_error_m=observed['position_m'][2]-top if observed.get('hit') else None,
            all_hit_transaction_complete=complete, prescribed_top_support_hits=supports,
            support_at_original_height_recorded=bool(complete and supports), occluders=occluders,
            every_non_support_hit_is_original_mapped_blocked_obstacle=all(o['mapped_blocked'] for o in occluders),
            query_points_or_allowed_supports_or_tolerances_modified=False))
    return rows


def sample_poses(poses):
    """Same 5 mm / 0.25 degree spatial diagnostic sampling, NOT continuous proof."""
    if not poses or not np.isfinite(poses).all():
        raise ValueError('Complete finite saved pose sequence required')
    out = [poses[0]]
    for first, last in zip(poses, poses[1:]):
        turn = launch.execution.wrap(last[2]-first[2])
        n = max(1, math.ceil(math.hypot(last[0]-first[0], last[1]-first[1])/.005),
            math.ceil(abs(turn)/math.radians(.25)))
        out.extend([[first[0]+i/n*(last[0]-first[0]), first[1]+i/n*(last[1]-first[1]), first[2]+i/n*turn]
            for i in range(1, n+1)])
    return np.asarray(out)


def obstacle_envelope_check(poses, bound, query_xy):
    """Vectorized SAT vs original conservative AABB; expanded FULL rectangle."""
    values = sample_poses(poses)
    c, s = np.cos(values[:, 2]), np.sin(values[:, 2])
    a, b = 1.630/2+.010, .965/2+.010
    lo, hi = np.asarray(bound['minimum_m'][:2]), np.asarray(bound['maximum_m'][:2])
    center, half = (lo+hi)/2, (hi-lo)/2
    delta = center-values[:, :2]
    gaps = np.column_stack((np.abs(delta[:, 0])-(a*np.abs(c)+b*np.abs(s)+half[0]),
        np.abs(delta[:, 1])-(a*np.abs(s)+b*np.abs(c)+half[1]),
        np.abs(delta[:, 0]*c+delta[:, 1]*s)-(a+half[0]*np.abs(c)+half[1]*np.abs(s)),
        np.abs(-delta[:, 0]*s+delta[:, 1]*c)-(b+half[0]*np.abs(s)+half[1]*np.abs(c))))
    separation = np.max(gaps, axis=1)
    point_delta = np.asarray(query_xy)-values[:, :2]
    contains = (np.abs(point_delta[:, 0]*c+point_delta[:, 1]*s) <= a) & (
        np.abs(-point_delta[:, 0]*s+point_delta[:, 1]*c) <= b)
    return dict(pose_samples=len(values), expanded_full_rectangle_area_m2=1.650*.985,
        colliding_samples_with_original_obstacle_AABB=int(np.count_nonzero(separation <= 0.)),
        query_point_inside_expanded_rectangle_samples=int(np.count_nonzero(contains)),
        minimum_SAT_separating_projection_gap_m=float(np.min(separation)),
        sampling_m_deg=[.005, .25], NOT_actual_path_or_continuous_cooked_collision_proof=True)


def build():
    prep = static.load_actual_preparation()
    protocol = static.load_protocol()
    raw = json.loads(static.RESULT.read_bytes()); cohort.verify(raw['input_sha256'])
    if (raw['version'] != static.VERSION or raw['verdict'] != 'FAIL'
            or raw.get('error') or raw.get('evaluation_error') or raw['physics_preparation_snapshot'] != prep
            or raw['trace_sha256'] != cohort.completion.sha(static.TRACE)
            or [key for key, ok in raw['checks'].items() if not ok] != [
                'all_sampled_cooked_support_hits_at_original_top_and_upward_normal']):
        raise ValueError('Only this exact original supplementary-ray FAIL is in scope; no broad failure waiver')
    with static.TRACE.open(newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    replay = static.evaluate(rows, raw['start_xyyaw'], raw['contact_report_counts_by_category'],
        raw['contact_callback_errors'], protocol['sampled_cooked_support_queries'],
        raw['actual_sampled_cooked_support_queries'], protocol['support_top_m'])
    if replay['verdict'] != 'FAIL' or any(raw['checks'].get(k) != v for k, v in replay['checks'].items()):
        raise ValueError('Original raw FAIL must reproduce without changing any criterion')
    core = static.original.original.evaluate_static(rows, raw['start_xyyaw'], 265.,
        raw['contact_report_counts_by_category'], raw['contact_callback_errors'])
    inventory_path = cohort.support.floor.inventory.TARGET
    inventory = json.loads(inventory_path.read_bytes()); cohort.verify(inventory['input_sha256'])
    with np.load(launch.cohort.MAP) as grid:
        occupied, origin, resolution = grid['occupied'].copy(), grid['origin_xy_m'].copy(), float(grid['resolution_m'])
    diagnostics = diagnose_queries(protocol['sampled_cooked_support_queries'], raw['actual_sampled_cooked_support_queries'],
        protocol['support_top_m'], inventory['collision_prims'], occupied, origin, resolution)
    failures = [r for r in diagnostics if not r['original_closest_hit_check']]
    if len(failures) != 1 or len(failures[0]['occluders']) != 1:
        raise ValueError('The observed single mapped-obstacle occlusion was not reproduced')
    failed = failures[0]; obstacle = failed['occluders'][0]['collision']
    record = next(row for row in inventory['collision_prims'] if row['path'] == obstacle)
    bundle, model, plans = launch.load_frozen_bundle()
    screens = []
    for name, plan in plans.items():
        poses = [plan['start_world_xyyaw']]+[state['world_xyyaw'] for state in plan['selected_states']]
        screens.append(dict(source='frozen_native_reference', method=name,
            **obstacle_envelope_check(poses, record['world_aligned_bound'], failed['xy_m'])))
    for case in model['cases']:
        with Path(case['model_trace']).open(newline='') as stream:
            poses = [[float(row[key]) for key in ('x_m', 'y_m', 'yaw_rad')] for row in csv.DictReader(stream)]
        screens.append(dict(source='nominal_closed_loop_MODEL_trace', method=case['method'],
            integration_dt_s=case['integration_dt_s'],
            **obstacle_envelope_check(poses, record['world_aligned_bound'], failed['xy_m'])))
    checks = dict(original_raw_FAIL_reproduced_unchanged=replay['verdict'] == 'FAIL',
        independent_original_stationary_load_pose_contact_checks_pass=core['verdict'] == 'PASS',
        all_26_original_query_transactions_complete=all(r['all_hit_transaction_complete'] for r in diagnostics),
        original_height_support_recorded_in_all_26_all_hit_lists=all(r['support_at_original_height_recorded'] for r in diagnostics),
        every_occluder_is_an_original_enabled_mapped_blocked_obstacle=all(
            r['every_non_support_hit_is_original_mapped_blocked_obstacle'] for r in diagnostics),
        all_four_native_and_eight_MODEL_sampled_envelopes_avoid_this_AABB=all(
            s['colliding_samples_with_original_obstacle_AABB'] == 0 for s in screens),
        failed_auxiliary_query_point_outside_all_sampled_envelopes=all(
            s['query_point_inside_expanded_rectangle_samples'] == 0 for s in screens))
    sources = (Path(__file__), ENTRY, static.RESULT, static.TRACE, inventory_path, launch.BUNDLE,
        launch.MODEL_SCREEN, launch.cohort.MAP)
    pins = {**raw['input_sha256'], **bundle['input_sha256'], **bundle['output_sha256'],
        **model['input_sha256'], **model['output_sha256'], **{str(p): cohort.completion.sha(p) for p in sources}}
    cohort.verify(pins)
    return cohort.canonical(dict(version=VERSION,
        status='SAVED_STATIC_RAY_OCCLUSION_DIAGNOSED_NO_ADMISSION' if all(checks.values()) else
            'HOLD_STATIC_RAY_DIAGNOSIS_INCOMPLETE', checks=checks,
        original_static_result=str(static.RESULT), original_static_raw_verdict='FAIL',
        original_static_raw_FAIL_and_all_criteria_immutable=True,
        core_stationary_replay=core, original_combined_replay=replay, query_diagnostics=diagnostics,
        failed_query_id=failed['id'], original_existing_occluding_obstacle=obstacle,
        original_obstacle_bound=record['world_aligned_bound'], sampled_envelope_screens=screens,
        floor_presence_is_NOT_traversable_free_space=True,
        candidate_revised_query_contract_NOT_authorized_or_applied=dict(
            keep_all_original_points_all_ray_hits_and_tolerances=True,
            support_channel='exact audited support at original height/normal, record occluders rather than erase them',
            obstacle_channel='every occluder remains an active strict-map obstacle; no path through its full-envelope exclusion',
            original_first_hit_query_contract_is_different_and_original_FAIL_stays=True,
            separate_version_and_explicit_review_required_before_any_reassessment_or_new_trial=True),
        live_PAUSED_state_and_cached_queries_recheck_pending=True,
        old_or_new_path_gate_must_reject_original_raw_static_FAIL=True,
        new_static_PASS_claimed=False, actual_drive_admission=False, new_PhysX_executed=False,
        CICS_runtime_integration=False, use_for_final_statistics=False, input_sha256=pins))


def load_review():
    result = json.loads(TARGET.read_bytes()); cohort.verify(result['input_sha256'])
    return result


def main():
    if TARGET.exists():
        result = load_review()
    else:
        result = build(); OUTPUT.mkdir(parents=True, exist_ok=False)
        cohort.save_exclusive(TARGET, result)
    print('=== H4 ACTUAL STATIC FAIL / ALL-RAY-HIT DIAGNOSIS (NO MOTION) ===')
    print('original raw static:', result['original_static_raw_verdict'], '; unchanged')
    print('failed auxiliary query:', result['failed_query_id'])
    print('existing occluder:', result['original_existing_occluding_obstacle'])
    print('diagnostic checks:', result['checks'])
    print('STATUS:', result['status']); print('saved:', TARGET)
    print('No new static PASS, drive admission, physics run, obstacle exemption or old result edit.')


if __name__ == '__main__':
    main()
