"""Import-only repair: consume the frozen ray protocol, not offline planners.

The pinned v1 script/protocol and all physical outcomes remain untouched.
No sys.path/module/importer mutation. Four queries, tolerances, continuity
validator and GET-only runtime are unchanged. No motion or scene authority.
"""

import json
from pathlib import Path

import numpy as np

import hospital_h4_reopened_static_gate as static
import hospital_h4_path_launch as continuity


ROOT, GATE = static.ROOT, static.GATE
PRIOR = GATE/'hospital_h4_near_floor_coverage/cached_live_r01/get_only_protocol.json'
PRIOR_LIVE = PRIOR.parent/'live_cached_low_wall.json'
OUTPUT = GATE/'hospital_h4_near_floor_coverage/cached_live_v2_r01'
PROTOCOL, LIVE = OUTPUT/'get_only_protocol.json', OUTPUT/'live_cached_low_wall.json'
ENTRY = ROOT/'isaac/scripts/isaac_review_ijcas_hospital_h4_cached_low_wall_v2.py'
VERSION = 'H4_retained_new_static_PASS_cached_low_wall_floor_geometry_GET_only_import_repair_v2'
sha, verify, write_exclusive = static.preparation.sha, static.preparation.verify, static.preparation.write_exclusive


def load_prior():
    data = json.loads(PRIOR.read_bytes()); verify(data['input_sha256'])
    raw = json.loads(static.RESULT.read_bytes()); verify(raw['input_sha256'])
    if (data['version'] != 'H4_retained_new_static_PASS_cached_low_wall_floor_geometry_GET_only_v1'
            or data['status'] != 'CACHED_GET_ONLY_REVIEW_PREPARED_NO_DRIVE_OR_PHYSICS_AUTHORITY'
            or data['actual_static_result'] != str(static.RESULT)
            or data['actual_static_sha256'] != sha(static.RESULT)
            or data['original_guard_tolerances'] != continuity.LIVE_TOLERANCES
            or len(data['sampled_rays']) != 4 or data['query_position_tolerance_m'] != .0005
            or data['support_normal_z_minimum'] != .99 or data['actual_drive_admission'] is not False
            or raw['verdict'] != 'PASS' or not all(raw['checks'].values())):
        raise ValueError('Exact frozen v1 query design and actual new static PASS required')
    if PRIOR_LIVE.exists():
        raise ValueError('The v1 live observation already exists; review it, do not duplicate the GET-only trial')
    return data


def prepare_protocol():
    prior = load_prior()
    paths = (Path(__file__), ENTRY, PRIOR, ROOT/'studies/IJCAS/scripts/hospital_private_loader.py')
    return dict(prior, version=VERSION,
        import_only_repair=True,
        user_reported_v1_error="ModuleNotFoundError: No module named 'audit_hospital_h4_complete_map_support'",
        failure_was_during_helper_import_before_get_stage_or_live_query=True,
        old_static_PASS_and_old_static_path_FAILs_preserved=True,
        prior_frozen_protocol=str(PRIOR), prior_frozen_protocol_sha256=sha(PRIOR),
        all_four_ray_specifications_tolerances_and_all_hit_policy_UNCHANGED=True,
        legacy_private_loader_and_prior_sources_unchanged=True,
        runtime_imports_only_hospital_helpers_and_standard_third_party_libraries=True,
        offline_map_generation_planning_or_fitting_code_NOT_imported_by_live_entry=True,
        additional_Play_Stop_Pause_reset_or_static_retry_authorized=False,
        input_sha256={**prior['input_sha256'], **{str(p): sha(p) for p in paths}})


def load_protocol():
    data = json.loads(PROTOCOL.read_bytes()); verify(data['input_sha256'])
    if data != prepare_protocol():
        raise ValueError('Exact frozen import-only repaired GET protocol required')
    return data


def evaluate(protocol, observations):
    specs = protocol['sampled_rays']
    if len(observations) != len(specs) or [r['id'] for r in observations] != [r['id'] for r in specs]:
        raise ValueError('Every prescribed cached ray transaction must be retained')
    rows = []
    tolerance = protocol['query_position_tolerance_m']
    top = protocol['support_top_m']
    for spec, observation in zip(specs, observations):
        hits = observation['all_hits']
        complete = not observation['errors']
        for hit in hits:
            values = [*hit['position_m'], *hit['normal'], hit['distance_m']]
            if len(hit['position_m']) != 3 or len(hit['normal']) != 3 or not np.isfinite(values).all() or not 0. <= hit['distance_m'] <= spec['distance_m']+tolerance:
                complete = False
            if hit['collision'] not in protocol['allowed_hit_collider_paths'] and not hit['collision'].startswith(static.ROBOT+'/'):
                complete = False
        target = [h for h in hits if h['collision'] == spec['expected_collider']]
        target_seen = any(np.linalg.norm(np.asarray(h['position_m'])-spec['expected_target_position_m']) <= tolerance
            and np.dot(h['normal'], spec['expected_target_normal']) >= protocol['support_normal_z_minimum'] for h in target) if complete else False
        support_seen = any(h['collision'] in static.SUPPORTS and abs(h['position_m'][2]-top) <= tolerance
            and h['normal'][2] >= protocol['support_normal_z_minimum'] for h in hits) if complete else False
        rows.append(dict(id=spec['id'], kind=spec['kind'], query_transaction_complete=complete,
            expected_original_collider_surface_recorded=bool(target_seen), original_top_support_recorded=bool(support_seen),
            required_surfaces_recorded=bool(target_seen and (spec['kind'] == 'horizontal_below_floor_wall' or support_seen))))
    return dict(checks=dict(all_cached_query_transactions_available=all(r['query_transaction_complete'] for r in rows),
        all_expected_floor_and_original_wall_surfaces_recorded=all(r['required_surfaces_recorded'] for r in rows)),
        query_diagnostics=rows, scope='CACHED_COOKED_SURFACE_RAYS_ONLY_NOT_CONTACT_OFFSET_FORCE_OR_PENETRATION')


def main():
    if PROTOCOL.exists():
        load_protocol()
    else:
        write_exclusive(PROTOCOL, prepare_protocol())
    print('GET-only import repair v2 frozen; same four rays/guards, no physics or scene changes.')
    print('saved:', PROTOCOL)


if __name__ == '__main__':
    main()
