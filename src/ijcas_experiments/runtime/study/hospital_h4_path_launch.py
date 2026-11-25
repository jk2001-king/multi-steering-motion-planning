"""Prospective H4 single-route development pilot, no implicit drive permission.

An exact actual static PASS is re-evaluated from raw CSV and sampled queries.
Live PAUSED body/twelve-joint/gain continuity is checked independently. Model
8/8 software cases are NOT treated as actual contact/hold/body validation.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_static_gate as stationary
import hospital_h4_path_execution as execution


cohort = stationary.cohort
ROOT, GATE = cohort.ROOT, cohort.GATE
VERSION = 'H4_single_frozen_route_original_v3_ground_truth_pilot_admission_v1'
BUNDLE = GATE/'hospital_h4_execution_bundle/bundle_r01/execution_bundle.json'
MODEL_SCREEN = GATE/'hospital_h4_common_executor_screen/screen_r01/common_executor_model_screen.json'
PROTOCOL = GATE/'hospital_h4_path_admission/design_r01/path_protocol.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_path_gate.py')
OUTPUT = GATE/'hospital_h4_path_execution_v3/cohort_01'
RESERVATION = OUTPUT/'single_static_handoff_attempt_reserved.json'
BENCHMARK = GATE/'motion_stop/hospital_h1_ramp_stop_hold_r01.json'
LIVE_TOLERANCES = dict(body_position_m=.001, yaw_deg=.1, joint_rad=.001, horizontal_stop_m_s=.005)


def load_frozen_bundle():
    report = json.loads(BUNDLE.read_bytes()); cohort.verify(report['input_sha256']); cohort.verify(report['output_sha256'])
    model = json.loads(MODEL_SCREEN.read_bytes()); cohort.verify(model['input_sha256']); cohort.verify(model['output_sha256'])
    if (len(report['plans']) != 4 or model['case_count'] != 8
            or model['nominal_software_MODEL_PASS_count'] != 8
            or any(r['model_status'] != 'MODEL_SCREEN_PASS' or not all(r['checks'].values()) for r in model['cases'])
            or model['actual_drive_admission'] is not False or report['actual_motion_authorized'] is not False):
        raise ValueError('Exact four native references and disclosed nominal software screens required, not physical admission')
    plans = {p['method']: p for p in (json.loads(Path(path).read_bytes()) for path in report['plans'])}
    if tuple(plans) != execution.METHODS:
        raise ValueError('All four methods must retain the same frozen bundle')
    for plan in plans.values():
        execution.validate_plan(plan)
    return report, model, plans


def prepare_protocol():
    bundle, model, plans = load_frozen_bundle()
    static_protocol = stationary.load_protocol()
    benchmark = json.loads(BENCHMARK.read_bytes())
    if benchmark['verdict'] != 'PASS' or benchmark['original_motor_gains_restored'] is not True:
        raise ValueError('Original H1 actuator/hold benchmark must remain intact; it is not H4 evidence')
    sources = (Path(__file__), ENTRY, Path(execution.__file__), Path(stationary.__file__),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/prepare_hospital_h4_path_gate.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_drive_hold.py'), _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_stop_analysis.py'))
    return cohort.canonical(dict(version=VERSION, status='CONDITIONAL_PROTOCOL_ONLY_ACTUAL_STATIC_AND_USER_LAUNCH_PENDING',
        required_actual_static_result=str(stationary.RESULT), required_actual_static_trace=str(stationary.TRACE),
        frozen_bundle=str(BUNDLE), frozen_native_methods=list(plans),
        explicit_user_selected_method_required=True, explicit_user_single_route_motion_authorization_required=True,
        one_frozen_route_only_no_automatic_second_method_or_retry=True,
        original_v3_tracking_projection_terminal_and_hold_gain_policy_unchanged=True,
        old_H1_H2_source_or_results_modified=False, initial_pre_hold_s=2., final_hold_s=5., fixed_tail_s=3.,
        command_limits=execution.COMMAND_LIMITS, guards=execution.GUARDS, acceptance=execution.ACCEPTANCE,
        terminal_controller_config=execution.CONTROLLER_CONFIG,
        live_static_handoff_tolerances=LIVE_TOLERANCES,
        two_active_drive_hold_stiffnesses_temporarily_10000_Nm_rad_then_original_restored=True,
        damping_torque_limits_and_all_passive_caster_gains_unchanged=True,
        contact_policy='six wheels on the two exact audited supports only; no obstacle/threshold/legacy exception',
        actual_swept_full_envelope_and_authored_support_sampling_m_deg=[.005, .25],
        common_static_source_authored_seed_policy='retained completed static state; never assign or select caster angles',
        B0_fixed_constraint_is_planned_path_only_common_terminal_may_change_modes=True,
        terminal_stabilization_logged_separately_from_planner_selected_modes=True,
        H1_component_benchmark_scope='shared actuator/hold precedent, NOT H4 path/traction/body/contact proof',
        baseline_launch_does_NOT_require_proposed_method_to_have_passed=True,
        old_development_trial_outcomes_are_NOT_matched_final_statistics=True,
        model_software_resolution_cases_are_NOT_independent_response_or_robustness_trials=True,
        independent_fresh_frozen_whole_H3_repeats=0, original_H3_MODEL_PASS_count=2, original_H3_MODEL_case_count=8,
        CICS_runtime_integration=False, sensor_or_Nav2_execution=False, real_robot_required=False,
        actual_static_verified=False, actual_motion_authorized=False, use_for_final_statistics=False,
        input_sha256={**bundle['input_sha256'], **bundle['output_sha256'], **model['input_sha256'],
            **model['output_sha256'], **static_protocol['input_sha256'],
            **{str(p): cohort.completion.sha(p) for p in (*sources, BUNDLE, MODEL_SCREEN, stationary.PROTOCOL, BENCHMARK)}}))


def load_protocol():
    report = json.loads(PROTOCOL.read_bytes()); cohort.verify(report['input_sha256'])
    if report != prepare_protocol():
        raise ValueError('Prospective H4 route protocol changed; no new controls/guard or model substitution')
    return report


def replay_actual_static():
    if not stationary.RESULT.is_file() or not stationary.TRACE.is_file():
        raise ValueError('Actual H4 static JSON AND raw CSV required; software MODEL 8/8 cannot substitute')
    prep = stationary.load_actual_preparation()
    protocol = stationary.load_protocol()
    raw = json.loads(stationary.RESULT.read_bytes()); cohort.verify(raw['input_sha256'])
    expected = dict(version=stationary.VERSION, scenario=execution.SCENARIO, verdict='PASS',
        physics_preparation=str(cohort.LIVE), physics_preparation_sha256=cohort.completion.sha(cohort.LIVE),
        physics_preparation_snapshot=prep, start_xyyaw=protocol['start_xyyaw'], simulation_mass_kg=265.,
        static_protocol=str(stationary.PROTOCOL), static_protocol_sha256=cohort.completion.sha(stationary.PROTOCOL),
        initial_motion_executed=False, planned_path_executed=False, CICS_input_used=False, Nav2_executed=False,
        caster_joint_state_reset_during_run=False, material_modified_during_run=False, motor_gains_modified_by_test=False,
        source_stage_saved=False, use_for_final_statistics=False, actual_drive_admission=False)
    if (any(raw.get(k) != v for k, v in expected.items()) or raw.get('error') or raw.get('evaluation_error')
            or not raw.get('checks') or not all(raw['checks'].values())
            or raw['trace_sha256'] != cohort.completion.sha(stationary.TRACE)):
        raise ValueError('Exact successful actual H4 static receipt is missing, changed or failed')
    with stationary.TRACE.open(newline='') as stream:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    computed = stationary.evaluate(rows, raw['start_xyyaw'], raw['contact_report_counts_by_category'],
        raw['contact_callback_errors'], protocol['sampled_cooked_support_queries'],
        raw['actual_sampled_cooked_support_queries'], protocol['support_top_m'])
    if computed['verdict'] != 'PASS' or any(raw['checks'].get(k) != value for k, value in computed['checks'].items()):
        raise ValueError('Original H4 static PASS does not reproduce from its actual raw data')
    names = raw['final_actual_joint_names']; q = raw['final_actual_twelve_joint_positions_rad']
    if (len(names) != 12 or len(set(names)) != 12 or len(q) != 12
            or len(raw['final_actual_world_xyyaw']) != 3
            or not np.isfinite(q).all() or not np.isfinite(raw['final_actual_world_xyyaw']).all()):
        raise ValueError('Complete finite POST-static body/twelve-joint snapshot required')
    return prep, raw, computed


def load_actual_inputs(method):
    if method not in execution.METHODS:
        raise ValueError('Select one exact frozen H4 method')
    protocol = load_protocol(); bundle, model, plans = load_frozen_bundle()
    prep, static, replay = replay_actual_static()
    plan = plans[method]
    if plan['start_world_xyyaw'] != static['start_xyyaw']:
        raise ValueError('Frozen H4 plan and own completed static start differ; no translation/recapture')
    return dict(protocol=protocol, bundle=bundle, model=model, plan=plan, preparation=prep, static=static, static_replay=replay)


def validate_tensor(static, transforms, names, positions, velocity, gains):
    transform = np.asarray(transforms, float).reshape(-1)
    speed = np.asarray(velocity, float).reshape(-1)
    q = np.asarray(positions, float).reshape(-1)
    names = list(names)
    if (transform.shape != (7,) or speed.shape != (6,) or q.shape != (12,)
            or not all(np.isfinite(v).all() for v in (transform, speed, q))
            or len(names) != 12 or len(set(names)) != 12
            or set(names) != set(static['final_actual_joint_names'])):
        raise ValueError('Finite actual retained POST-static body/twelve joints required')
    expected = static['final_actual_world_xyyaw']
    x, y, z, w = transform[3:]
    if abs(np.linalg.norm((x, y, z, w))-1.) > 1e-4:
        raise ValueError('Actual root quaternion is not normalized')
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    position_error = math.hypot(transform[0]-expected[0], transform[1]-expected[1])
    yaw_error = math.degrees(execution.wrap(yaw-expected[2]))
    expected_q = dict(zip(static['final_actual_joint_names'], static['final_actual_twelve_joint_positions_rad']))
    joint_error = max(abs(value-expected_q[name]) for name, value in zip(names, q))
    baseline = static['runtime_motor_baseline']; recorded_names = baseline['joint_names']
    if len(recorded_names) != 12 or set(recorded_names) != set(names):
        raise ValueError('Complete POST-static motor baseline required')
    settings_match = True
    for key in ('kp', 'kd', 'max_efforts'):
        recorded = dict(zip(recorded_names, baseline[key]))
        observed = gains.get(key)
        if observed is None or len(observed) != len(names):
            settings_match = False; continue
        target = [math.inf if recorded[n] == 'positive_infinity' else float(recorded[n]) for n in names]
        if not np.array_equal(np.asarray(observed, float), np.asarray(target, float)):
            settings_match = False
    if (position_error > LIVE_TOLERANCES['body_position_m'] or abs(yaw_error) > LIVE_TOLERANCES['yaw_deg']
            or joint_error > LIVE_TOLERANCES['joint_rad'] or np.linalg.norm(speed[:2]) > LIVE_TOLERANCES['horizontal_stop_m_s']
            or not settings_match):
        raise ValueError('Live H4 state differs from its completed POST-static snapshot; no reset/recovery permitted')
    return dict(position_difference_m=position_error, heading_difference_deg=yaw_error,
        maximum_joint_difference_rad=joint_error, horizontal_speed_m_s=float(np.linalg.norm(speed[:2])),
        original_all_twelve_motor_settings_match=True,
        compared_to_COMPLETED_static_not_source_seed_or_first_post_Play_row=True)


def physics_baseline_digest(preparation, environment, runtime_motor_baseline=None):
    """Normalize robot paths only, never hash an evidence filename as physics."""
    bodies = [{**row, 'path': row['path'].replace(cohort.ROBOT+'/', 'ROBOT/', 1)} for row in environment['rigid_bodies']]
    normal = dict(simulation_mass_kg=environment['simulation_mass_kg'], rigid_bodies=bodies,
        physics_hz=environment['physics_hz'], material_readback=environment['material_readback'],
        original_joint_settings=cohort.JOINT_SETTINGS, support_colliders=preparation['support_colliders'],
        all_twelve_runtime_motor_settings=runtime_motor_baseline,
        support_top_m=preparation['support_top_m'], source_robot_asset_sha256=cohort.ASSET_SHA,
        disabled_legacy_colliders=environment['disabled_legacy_colliders'])
    return hashlib.sha256(json.dumps(normal, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
