"""PROPOSAL ONLY: separate floor presence from mapped obstacle exclusion.

Changes the supplemental query definition, NOT the physical static limits.
No adoption, reinterpretation of the original FAIL, physical retry or driving
is authorized by producing or testing this prospective contract.
"""

import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_static_query_review as diagnosis


static, cohort = diagnosis.static, diagnosis.cohort
OUTPUT = cohort.GATE/'hospital_h4_support_query_contract/proposal_r01'
TARGET = OUTPUT/'query_contract.json'
VERSION = 'H4_same_26_all_hit_floor_and_mapped_obstacle_query_contract_PROPOSAL_v1'


def envelope_screens_complete(screens):
    expected = {(source, method, dt) for method in diagnosis.launch.execution.METHODS
        for source, dt in (('frozen_native_reference', None), ('nominal_closed_loop_MODEL_trace', .02),
            ('nominal_closed_loop_MODEL_trace', .01))}
    actual = [(row.get('source'), row.get('method'), row.get('integration_dt_s')) for row in screens]
    return bool(len(actual) == len(expected) and len(set(actual)) == len(expected) and set(actual) == expected
        and all(row.get('pose_samples', 0) >= 1 and row.get('expanded_full_rectangle_area_m2') == 1.650*.985
            and row.get('sampling_m_deg') == [.005, .25]
            and row.get('colliding_samples_with_original_obstacle_AABB') == 0
            and row.get('query_point_inside_expanded_rectangle_samples') == 0
            and math.isfinite(row.get('minimum_SAT_separating_projection_gap_m', math.nan))
            and row['minimum_SAT_separating_projection_gap_m'] > 0.
            and row.get('NOT_actual_path_or_continuous_cooked_collision_proof') is True for row in screens))


def evaluate_example(specifications, observations, top, inventory, occupied, origin, resolution,
        screens_by_obstacle):
    """Pure diagnostic example. A satisfied example is NEVER an admission."""
    rows = diagnosis.diagnose_queries(specifications, observations, top, inventory, occupied, origin, resolution)
    specs = {spec['id']: spec for spec in specifications}
    consistent = True
    for observed in observations:
        spec = specs[observed['id']]
        hits = observed.get('all_nonself_hits', [])
        own = observed.get('own_robot_occlusion_hits', [])
        if (not observed.get('hit') or observed.get('errors') or not hits
                or observed.get('xy_m') != spec['xy_m']
                or any(not str(hit.get('collision', '')).startswith(cohort.ROBOT+'/') for hit in own)):
            consistent = False
            continue
        distances = []
        for hit in hits:
            pos, normal, distance = hit.get('position_m', []), hit.get('normal', []), hit.get('distance_m', math.nan)
            if (len(pos) != 3 or len(normal) != 3 or not np.isfinite([*pos, *normal, distance]).all()
                    or not 0. <= distance <= .8
                    or not np.allclose(pos[:2], spec['xy_m'], rtol=0, atol=static.QUERY_LIMITS['horizontal_tolerance_m'])
                    or abs(pos[2]+distance-.6) > static.QUERY_LIMITS['height_tolerance_m']):
                consistent = False
            distances.append(distance)
        if (any(a > b for a, b in zip(distances, distances[1:]))
                or any(observed.get(key) != hits[0].get(key) for key in ('collision', 'position_m', 'normal', 'distance_m'))):
            consistent = False
    occluders = {item['collision'] for row in rows for item in row['occluders']}
    screens = bool(all(path in screens_by_obstacle and envelope_screens_complete(screens_by_obstacle[path])
        for path in occluders))
    checks = dict(original_query_identity_and_all_hit_transactions_complete=consistent
            and all(row['all_hit_transaction_complete'] for row in rows),
        prescribed_support_present_at_original_height_XY_and_normal=all(row['support_at_original_height_recorded'] for row in rows),
        no_unknown_disabled_legacy_or_unmapped_non_support_hit=all(
            row['every_non_support_hit_is_original_mapped_blocked_obstacle'] for row in rows),
        every_occluder_has_all_four_reference_and_eight_MODEL_full_envelope_exclusion_screens=screens)
    return dict(checks=checks, diagnostic_example_satisfies_proposed_query_contract=all(checks.values()),
        original_nearest_hit_contract_pass_count=sum(row['original_closest_hit_check'] for row in rows),
        query_count=len(rows), original_obstacles_retained_and_recorded=sorted(occluders),
        original_static_verdict_relabelled=False, actual_drive_admission=False,
        independent_static_or_model_validation_claimed=False)


def build():
    reviewed = diagnosis.load_review()
    if not all(reviewed['checks'].values()) or reviewed['original_static_raw_verdict'] != 'FAIL':
        raise ValueError('The exact actual FAIL diagnosis must be retained before drafting a different contract')
    raw = json.loads(static.RESULT.read_bytes())
    protocol = json.loads(static.PROTOCOL.read_bytes())
    inventory_path = cohort.support.floor.inventory.TARGET
    inventory = json.loads(inventory_path.read_bytes())
    with np.load(diagnosis.launch.cohort.MAP) as grid:
        example = evaluate_example(protocol['sampled_cooked_support_queries'], raw['actual_sampled_cooked_support_queries'],
            protocol['support_top_m'], inventory['collision_prims'], grid['occupied'], grid['origin_xy_m'],
            float(grid['resolution_m']), {reviewed['original_existing_occluding_obstacle']: reviewed['sampled_envelope_screens']})
    if not example['diagnostic_example_satisfies_proposed_query_contract']:
        raise ValueError('Even the disclosed diagnostic example fails the proposed query contract')
    sources = (Path(__file__), diagnosis.TARGET, static.RESULT, static.TRACE, static.PROTOCOL,
        inventory_path, diagnosis.launch.cohort.MAP)
    pins = {**reviewed['input_sha256'], **{str(path): cohort.completion.sha(path) for path in sources}}
    cohort.verify(pins)
    return cohort.canonical(dict(version=VERSION, status='PROPOSAL_ONLY_EXPLICIT_ADOPTION_DECISION_REQUIRED',
        original_raw_static_result=str(static.RESULT), original_raw_static_verdict='FAIL',
        original_FAIL_and_first_hit_evaluator_and_receipts_remain_immutable=True,
        supplemental_query_definition_changed=True,
        old_definition='nearest non-self ray hit must be an exact support at every endpoint/interior patch sample',
        proposed_definition='same all-hit rays: support presence at prescribed height AND separate strict mapped-obstacle exclusion',
        all_original_26_query_points_and_ray_height_direction_length_preserved=True,
        original_query_limits=static.QUERY_LIMITS,
        original_static_limits=static.LIMITS, original_static_safety_guards=static.GUARDS,
        unchanged_physical_requirements=['all six wheels carry load', 'vertical load/weight balance and stability',
            '10mm stationary drift and original tilt/Z limits', 'zero obstacle/nonwheel/legacy contact',
            'original material, 265kg, gains, source-authored initial condition policy',
            '30mm/2deg terminal, 5mm/s stopped, +10mm full-envelope driving guards'],
        proposed_obstacle_channel=['record EVERY non-support hit, never filter hospital objects out of the ray list',
            'require original enabled collider, matching bounds and occupied strict-map cell',
            'require all FOUR frozen-reference and EIGHT nominal MODEL sampled full-envelope exclusion cases',
            'reject unknown/disabled/unmapped obstacles and missing/changed case data',
            'retain runtime contact/full-envelope guards without exceptions'],
        support_presence_is_NOT_free_space_or_whole_path_support_certificate=True,
        sampled_envelope_screens_are_NOT_continuous_collision_or_body_response_validation=True,
        existing_recorded_data_example_ONLY=example,
        retrospective_example_is_NOT_independent_validation=True,
        original_failed_static_is_NOT_new_actual_static_PASS=True,
        default_next_action='Await user decision on the changed supplementary query definition and actual PAUSED continuity review; do not launch another static or route.',
        explicit_user_adoption_required=True, explicit_user_new_PhysX_trial_authority_required=True,
        current_live_diagnostic_receipt_required=str(diagnosis.LIVE),
        adoption_authorized=False, new_static_trial_authorized=False, actual_drive_admission=False,
        new_PhysX_executed=False, original_query_or_static_code_modified=False,
        CICS_runtime_integration=False, real_robot_required=False, use_for_final_statistics=False,
        input_sha256=pins))


def main():
    report = build()
    if TARGET.exists():
        if json.loads(TARGET.read_bytes()) != report:
            raise ValueError('Existing proposal changed; no overwrite')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        cohort.save_exclusive(TARGET, report)
    print('=== H4 SEPARATED FLOOR / OBSTACLE QUERY CONTRACT PROPOSAL (OFFLINE ONLY) ===')
    print('Original actual static: FAIL, retained; no reassessment as a new actual PASS')
    print('Change: supplementary query definition only; same 26 points and numerical tolerances')
    print('Physical gates, strict raster, obstacle colliders, gains and materials unchanged')
    print('Recorded-data example:', report['existing_recorded_data_example_ONLY'])
    print('Adoption/static retry/drive authorized:', False, False, False)
    print('saved:', TARGET)


if __name__ == '__main__':
    main()
