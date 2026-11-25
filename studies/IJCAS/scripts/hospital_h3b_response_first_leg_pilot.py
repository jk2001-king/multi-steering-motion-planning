"""ONE prospective reserve-FIRST-forward-leg spatial feedback pilot.

The frozen response is used in OFFLINE admission only, not actuator inversion.
Reuse the exact existing H1 tracking law with its original gains. Different
reference/controller from the failed time replay is EXPLICIT, not a relabelled
repeat or an unchanged four-method comparison. No reverse/terminal pulses.
"""

import json
import math

from hospital_h3b_first_leg_model_pilot import (POLICY, READINESS, ACCEPTANCE, BAD_CONTACTS,
    ROBOT_ROOT, WHEELS, GUARDS, TRACKING, COMMAND_LIMITS, classify_contact, relative_pose,
    wheel_crossing_checks, motion_stalled, direct_joint_targets, component_goal,
    evaluate_readiness as original_evaluate)
from hospital_h3b_first_leg_feedback_review import FirstLegReference, FirstLegFeedback, TRACKING_GAINS
from hospital_h3b_response_link import load_validated_response, VERSION as RESPONSE_VERSION
from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources, model_bounds


CONDITION = 'H3B_frozen_response_reserve_first_leg_spatial_feedback_v1'
ROLE = 'PROSPECTIVE_FIRST_FORWARD_SPATIAL_FEEDBACK_COMPONENT_NOT_FULL_H3_OR_METHOD_COMPARISON'
PROFILE = GATE/'hospital_h3b_response_link_reviews/review_r02/response_link_review.json'
OUTPUT = GATE/'hospital_h3b_response_first_leg_pilot/cohort_01'


def load_profile():
    load_validated_response()
    record = json.loads(PROFILE.read_bytes()); verify_sources(record)
    SUPPORT['verify_files'](record['input_sha256'])
    cases = record['first_leg_model_cases']
    nominal = [c for c in cases if c['residual_case'] == 'frozen_nominal']
    if (record['version'] != RESPONSE_VERSION
            or record.get('status') != 'NOMINAL_FIRST_COMPONENT_MODEL_ADMITTED_STILL_NEEDS_PROSPECTIVE_PhysX'
            or record.get('nominal_first_forward_component_model_admitted') is not True
            or record.get('PhysX_executed') is not False
            or record.get('automatic_PhysX_execution_authorized') is not False
            or len(nominal) != 6 or any(not c['all_model_checks_pass'] or c['model_parameters'] != model_bounds() for c in nominal)
            or {(c['dt_s'], c['initial_case']) for c in nominal} != {
                (dt, initial) for dt in (.02, .01) for initial in record['declared_initial_cases']}):
        raise ValueError('Require ALL nominal first-leg dt/initial/reserve model cases; never select a favourable case')
    SUPPORT['verify_files']({c['trace_file']: c['trace_sha256'] for c in cases})
    from pathlib import Path
    candidate_path = record['candidate']; candidate = json.loads(Path(candidate_path).read_bytes())
    verify_sources(candidate); SUPPORT['verify_files'](candidate['input_sha256'])
    if SUPPORT['sha'](candidate_path) != record['candidate_sha256']:
        raise ValueError('Fixed reserve candidate changed')
    geometry, timing = candidate['geometry_audit'], candidate['nominal_timing']
    stop = timing['cusp_sample_indices'][0]
    if (any(d != 1 for d in geometry['interval_directions'][:stop])
            or any(m != 'coordinated_counterphase' for m in geometry['interval_modes'][:stop])
            or abs(geometry['sample_curvature_1_m'][stop]) > 1e-10
            or candidate['unchanged_start_xyyaw'] != record['original_mission_start_xyyaw']
            or candidate['unchanged_goal_xyyaw'] != record['original_mission_goal_xyyaw']
            or candidate['unchanged_strict_map_digest'] != record['strict_map_digest']):
        raise ValueError('Original task/map and forward counterphase first-leg contract required')
    points = geometry['poses_local_xyyaw']
    prefix = [dict(leg_index=0, command_mode='coordinated_counterphase',
        command_speed_m_s=timing['sample_unsigned_reference_speed_m_s'][i],
        requested_curvature_1_m=geometry['sample_curvature_1_m'][i],
        x_m=points[i][0], y_m=points[i][1], yaw_rad=points[i][2]) for i in range(stop+1)]
    info = dict(profile=str(PROFILE), profile_sha256=SUPPORT['sha'](PROFILE),
        input_sha256={**record['input_sha256'], str(PROFILE): SUPPORT['sha'](PROFILE),
            **{c['trace_file']: c['trace_sha256'] for c in cases}},
        mode='coordinated_counterphase', duration_s=next(c['model_duration_s']-5. for c in nominal if c['dt_s']==.01),
        first_stop_model_local_xyyaw=points[stop], model_prefix_rows=prefix,
        maximum_reference_x_m=max(p[0] for p in points[:stop+1]),
        mission_start_xyyaw=record['original_mission_start_xyyaw'], mission_goal_xyyaw=record['original_mission_goal_xyyaw'],
        strict_map_digest=record['strict_map_digest'], model_admission_is_not_PhysX_success=True,
        reference_kind='FIXED_NEW_SPATIAL_PLAN_NOT_MODEL_PREDICTED_TRACE',
        legacy_model_field_names_are_schema_adapters_not_time_replay=True,
        model_does_NOT_invert_or_modify_physical_drive_gains=True,
        synthetic_stress_checks_all_pass=record['all_synthetic_stress_cases_pass'])
    if info['first_stop_model_local_xyyaw'] != record['fixed_first_nominal_cusp_local_xyyaw']:
        raise ValueError('Fixed nominal first-stop target changed')
    interpolation_table(info)
    return info


def interpolation_table(info):
    # Kept name only to make the inherited entry adapter auditable.
    return FirstLegReference(info)


def reference_at_pose(info, local_pose):
    if '_guard_spatial_reference' not in info:
        info['_guard_spatial_reference'] = FirstLegReference(info)
    _, reference, _ = info['_guard_spatial_reference'].project(local_pose)
    return reference


def evaluate_readiness(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events):
    result = original_evaluate(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events)
    result.pop('first_forward_leg_model_pilot_verified', None)
    result.update(first_forward_spatial_feedback_component_verified=result['verdict']=='PASS',
        legacy_model_reference_fields_represent_SPATIAL_plan_tracking=True,
        body_response_or_large_angle_joint_model_validation_NOT_implied_by_motion_PASS=True,
        full_H3B_turning_path_verified=False, reverse_turning_verified=False, use_for_final_statistics=False)
    return result
