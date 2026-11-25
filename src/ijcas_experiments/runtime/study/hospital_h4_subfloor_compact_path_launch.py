"""Bind unchanged H4 v3 routes to the NEW compact subfloor static trial.

Preparation, a nominal software screen and display recovery are not drive
permission. Require this cohort's actual static PASS/raw CSV, four identity
probes, full protected environment and retained thirteen-link/twelve-joint
state. Never decode the giant old archive or relabel/pool old outcomes.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import csv
import hashlib
import json
import math
import io
import zipfile
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

import hospital_h4_subfloor_compact_static_gate as stationary
import hospital_h4_path_execution as execution
import hospital_h4_completed_stop_path_launch as original
import hospital_runtime_evidence_io as evidence
import hospital_h4_static_v2_failure_review as completed_static


preparation = stationary.preparation
cohort = SimpleNamespace(**vars(stationary.cohort))
cohort.MAP = preparation.old.original.MAP
cohort.support_covers = preparation.old.original.support_covers
cohort.support_area_missing = preparation.old.original.support_area_missing
cohort.support = preparation.old.original.support
cohort.WHEELS = tuple(stationary.WHEELS.values())
ROOT, GATE = preparation.ROOT, preparation.GATE
VERSION = 'H4_single_original_v3_route_compact_subfloor_actual_static_binding_v1'
EXECUTION_COHORT = preparation.COHORT + '_compact_static_r01'
BUNDLE, MODEL_SCREEN, BENCHMARK = original.BUNDLE, original.MODEL_SCREEN, original.BENCHMARK
DESIGN = GATE/'hospital_h4_path_admission'/preparation.COHORT/'compact_static_r01'
PROTOCOL = DESIGN/'path_protocol.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_subfloor_compact_path_gate.py')
OUTPUT = GATE/'hospital_h4_path_execution_v3'/EXECUTION_COHORT
RESERVATION = OUTPUT/'single_static_handoff_attempt_reserved.json'
CLOSEOUT_REVIEW = DESIGN/'recorded_static_closeout_review.json'
EXPERIMENT_ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_experiment.py')
STATIC_DESIGN = DESIGN/'retained_corrected_static_protocol.json'
RUNTIME_SNAPSHOT = DESIGN/'runtime_source_snapshot.zip'
STATIC_OUTPUT = GATE/'hospital_h4_static_contact'/preparation.COHORT/'retained_corrected_static_r01'
STATIC_RUN_ID = 'hospital_h4_retained_corrected_static_r01'
INITIAL_DESIGN = DESIGN/'initial_motion_design_0p65m.json'
INITIAL_PROTOCOL = DESIGN/'initial_motion_protocol.json'
INITIAL_SOURCES = DESIGN/'initial_motion_runtime_sources.zip'
INITIAL_OUTPUT = GATE/'hospital_h4_initial_motion'/preparation.COHORT/'retained_static_r01'
INITIAL_RESERVATION = INITIAL_OUTPUT/'single_initial_motion_reserved.json'
INITIAL_RUN_ID = 'hospital_h4_initial_motion_r01'
INITIAL_GENERATOR = _ijcas_path(ROOT, 'studies/IJCAS/scripts/prepare_hospital_h4_subfloor_compact_path_gate.py')
FRESH_TRIAL_ROOT = GATE/'hospital_h4_experiments'
LIVE_TOLERANCES = dict(original.LIVE_TOLERANCES)
load_frozen_bundle = original.load_frozen_bundle
validate_tensor = original.validate_tensor


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def evaluate_support_queries_for_actor(protocol, observations):
    """Identical frozen all-hit rules with the DECLARED current actor identity.

    The old pure evaluator captured the retired actor's namespace in globals.
    Rebind ONLY that configuration value. Do not edit any saved observation,
    omit a hit, change a tolerance, or treat a different actor as self.
    """
    robot = protocol['new_robot_root']
    if robot != cohort.ROBOT or robot != preparation.ROBOT:
        raise ValueError('Exact protocol/actual cohort actor identity required')
    adopted = completed_static.stationary.previous.prior.previous.adopted
    contract = adopted.contract
    original = contract.evaluate_example
    bound = FunctionType(original.__code__, dict(original.__globals__, cohort=SimpleNamespace(ROBOT=robot)),
        original.__name__, original.__defaults__, original.__closure__)
    reviewed = adopted.diagnosis.load_review()
    inventory = json.loads(adopted.cohort.support.floor.inventory.TARGET.read_bytes())
    with np.load(cohort.MAP) as grid:
        result = bound(protocol['sampled_cooked_support_queries'], observations, protocol['support_top_m'],
            inventory['collision_prims'], grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m']),
            {reviewed['original_existing_occluding_obstacle']: reviewed['sampled_envelope_screens']})
    return dict(result, actor_identity_binding=dict(previous_actor_root=contract.cohort.ROBOT,
        declared_actual_actor_root=robot, original_evaluator_code_object_unchanged=bound.__code__ is original.__code__,
        tolerances_observations_map_obstacles_and_envelope_screens_unchanged=True))


def review_recorded_static_closeout():
    """Saved-data retrospective completion, NOT a new raw static PASS/admission.

    Only this exact infrastructure-failed trial is in scope. Require ZERO
    elapsed simulation time, the entire retained tensor state/environment,
    all original physical limits, and unchanged 26+4 native ray records.
    """
    baseline = completed_static.load_saved_review()
    raw, definition, rows = completed_static.saved_records()
    path = completed_static.LIVE
    if not path.is_file():
        raise ValueError('Actual delayed PAUSED closeout required; no Play to create it')
    live = preparation.read_small(path)
    preparation.verify(live['input_sha256'])
    expected = dict(version=completed_static.VERSION, original_static_verdict='FAIL',
        original_static_result_NOT_relabelled=True, actual_drive_admission=False,
        automatic_retry=False, any_Play_Stop_reset_control_or_stage_edit=False,
        status='DELAYED_GET_ONLY_CLOSEOUT_INCOMPLETE_NO_ADMISSION', errors=[])
    if (any(live.get(k) != v for k,v in expected.items())
            or live.get('original_result_sha256') != preparation.sha(completed_static.stationary.RESULT)
            or live.get('original_trace_sha256') != preparation.sha(completed_static.stationary.TRACE)
            or live.get('saved_data_review_sha256') != preparation.sha(completed_static.SAVED_REVIEW)
            or live.get('delayed_from_original_recorded_timeline_s') != 0.
            or live.get('actual_final_timeline') != dict(raw['actual_final_timeline'], playing=False)
            or live.get('pause_confirmation',{}).get('after') != live['actual_final_timeline']):
        raise ValueError('Exact zero-time retained failed-static closeout required, not another trial/state')
    required = {'actor_settings_equal_own_same_SDK_PRE_Play_source_reference',
        'all_4997_original_rows_match_except_exact_four_approved_flags',
        'delayed_disabled_identity_probes', 'Root_Session_and_paused_time_unchanged_during_GETs',
        'original_trial_files_unchanged'}
    if (set(live['checks']) != required | {'delayed_query_transactions_and_limits'}
            or any(live['checks'].get(k) is not True for k in required)
            or live['checks']['delayed_query_transactions_and_limits'] is not False):
        raise ValueError('Only the exact own-actor query identity error may be reviewed; no broad failure waiver')
    old_queries = live['delayed_query_evaluation']
    expected_failed = 'adopted_query_original_query_identity_and_all_hit_transactions_complete'
    if {k for k,v in old_queries['checks'].items() if not v} != {expected_failed}:
        raise ValueError('A physical/support/obstacle failure cannot be waived as an actor identity bug')
    actual = live['actual_current_tensor_snapshot']
    continuity = completed_static.continuity_diagnosis(raw, actual)
    if (continuity['maximum_link_position_difference_m'] != 0.
            or continuity['maximum_link_orientation_difference_deg'] > .1
            or continuity['maximum_joint_difference_rad'] != 0.
            or continuity['runtime_motor_baseline_retained'] is not True
            or continuity['twelve_joint_velocities_exactly_retained'] is not True):
        raise ValueError('Own recorded body/all 12 joints/13 links/gains not retained; no recovery')
    tensor = live['same_actual_trial_SDK_actor_rows']
    if (tensor['actual'] != tensor['expected']
            or tensor['expected'] != raw['runtime_anonymous_source_reference']['fixed_configuration_rows']
            or len(tensor['actual']) != 145):
        raise ValueError('All own same-SDK source actor parameters required')
    prep = raw['physics_preparation_snapshot']
    before = raw['compact_environment_before_Play']
    after = live['protected_environment_delayed_readback']
    if (before['all_original_row_count'] != 4997 or after['row_count'] != 4997
            or before['protected_environment_sha256'] != prep['protected_environment_sha256']
            or after['sha256'] != prep['protected_environment_sha256']
            or live['critical_environment'] != raw['environment_after_static']):
        raise ValueError('Entire original protected environment and physics must match')
    supports, probes = live['delayed_26_support_queries'], live['delayed_four_identity_probes']
    current = evaluate_support_queries_for_actor(definition, supports)
    absence = completed_static.stationary.evaluate_subfloor_queries(definition, supports, probes)
    physical = completed_static.physical.evaluate_static(rows, raw['start_xyyaw'], 265.,
        raw['contact_report_counts_by_category'], raw['contact_callback_errors'])
    own = [hit for row in supports for hit in row['own_robot_occlusion_hits']]
    if (not own or any(hit['collision'] not in tensor['actual'] for hit in own)
            or any(not np.isfinite([*hit['position_m'],*hit['normal'],hit['distance_m']]).all() for hit in own)):
        raise ValueError('Every retained self-ray hit must name an exact known current actor node')
    checks = dict(recorded_original_2s_physical_limits=physical['verdict']=='PASS',
        unchanged_26_all_hit_queries_with_exact_current_actor=all(current['checks'].values()),
        unchanged_four_disabled_identity_probes=all(absence['checks'].values()),
        exact_body_twelve_joint_state_gains_and_13_links_retained=True,
        no_elapsed_simulation_between_result_and_delayed_closeout=True,
        same_4997_row_environment_and_same_SDK_actor_settings=True,
        original_raw_FAIL_and_delayed_incomplete_record_preserved=True)
    files=(completed_static.SAVED_REVIEW, path, completed_static.stationary.RESULT,
        completed_static.stationary.TRACE)
    pins={**baseline['input_sha256'], **live['input_sha256'], **{str(p):preparation.sha(p) for p in files}}
    preparation.verify(pins)
    return dict(version=VERSION, status=('SAME_TIMESTAMP_QUERY_IDENTITY_DIAGNOSIS_SATISFIED_NO_STATIC_OR_DRIVE_ADMISSION'
        if all(checks.values()) else 'HOLD_RECORDED_STATIC_COMPLETION_REVIEW_FAILED'), checks=checks,
        evidence_kind='RETROSPECTIVE_REVIEW_OF_ALREADY_RECORDED_PHYSICAL_TRACE_AND_SAME_TIMESTAMP_CACHED_QUERIES',
        original_static_result=str(completed_static.stationary.RESULT), original_static_verdict='FAIL',
        original_delayed_closeout=str(path), original_delayed_closeout_status=live['status'],
        physical_stationary_replay=physical, current_actor_query_replay=current, four_probe_replay=absence,
        retained_physical_state_diagnosis=continuity, observed_own_actor_hit_count=len(own),
        actual_trial_count_added=0, new_static_PASS_created=False, original_outcomes_modified=False,
        prospective_independent_static_validation_claimed=False, final_comparison_cells_added=0,
        root_sha256=hashlib.sha256(live['original_Root_text_at_delayed_closeout'].encode()).hexdigest(),
        session_sha256=hashlib.sha256(live['original_Session_text_at_delayed_closeout'].encode()).hexdigest(),
        prospective_corrected_static_and_short_initial_motion_checks_required=True,
        actual_drive_admission=False, automatic_retry=False, use_for_final_statistics=False,
        analysis_source_sha256={str(Path(__file__)):preparation.sha(Path(__file__))},
        input_sha256=pins)


def prepare_protocol():
    bundle, model, plans = load_frozen_bundle()
    static = stationary.load_protocol()
    benchmark = preparation.read_small(BENCHMARK)
    if benchmark['verdict'] != 'PASS' or benchmark['original_motor_gains_restored'] is not True:
        raise ValueError('Preserved H1 actuator precedent required; it is NOT H4 route validation')
    sources = (Path(__file__), ENTRY, Path(original.__file__), Path(execution.__file__),
        Path(evidence.__file__),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/prepare_hospital_h4_subfloor_compact_path_gate.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py'),
        _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_adopted_query_path_gate.py'))
    return canonical(dict(version=VERSION, execution_cohort=EXECUTION_COHORT,
        status='CONDITIONAL_DESIGN_ONLY_NEW_ACTUAL_STATIC_PASS_AND_SEPARATE_ROUTE_CONSENT_REQUIRED',
        required_actual_static_result=str(stationary.RESULT), required_actual_static_trace=str(stationary.TRACE),
        required_actual_preparation=str(preparation.LIVE), static_protocol=str(stationary.PROTOCOL),
        new_robot_root=cohort.ROBOT, marker=cohort.MARKER,
        environment_variant='four_exact_original_subfloor_collision_flags_disabled_Session_only',
        exact_disabled_subfloor_paths=list(preparation.SELECTED),
        environment_fingerprint_required_for_all_four_methods=True,
        old_environment_static_and_route_FAILs_are_NOT_relabelled_or_pooled=True,
        original_planning_map_changed_cells=0, frozen_bundle=str(BUNDLE), frozen_native_methods=list(plans),
        explicit_user_selected_method_required=True, explicit_user_single_route_motion_authorization_required=True,
        one_frozen_route_only_no_automatic_second_method_or_retry=True,
        original_v3_tracking_projection_terminal_and_hold_gain_policy_unchanged=True,
        common_evidence_writer_and_confirmed_Pause_required=True,
        JSON_tuple_to_array_representation_verified_before_exclusive_creation=True,
        deferred_Pause_must_complete_before_environment_readback=True,
        initial_pre_hold_s=2., final_hold_s=5., fixed_tail_s=3.,
        command_limits=execution.COMMAND_LIMITS, guards=execution.GUARDS, acceptance=execution.ACCEPTANCE,
        terminal_controller_config=execution.CONTROLLER_CONFIG, live_static_handoff_tolerances=LIVE_TOLERANCES,
        retained_all_thirteen_links_required=True, retained_twelve_joint_velocities_must_match_exactly=True,
        missing_optional_static_link_snapshot_blocks_drive_NOT_relabels_static=True,
        two_active_drive_hold_stiffnesses_temporarily_10000_Nm_rad_then_original_restored=True,
        damping_torque_limits_and_all_passive_caster_gains_unchanged=True,
        contact_policy='six wheels on two exact supports only; ANY disabled mesh or obstacle contact still FAIL',
        actual_swept_full_envelope_and_authored_support_sampling_m_deg=[.005, .25],
        common_static_source_authored_seed_policy='retained completed static state; never assign or select caster angles',
        B0_fixed_constraint_is_planned_path_only_common_terminal_may_change_modes=True,
        terminal_stabilization_logged_separately_from_planner_selected_modes=True,
        H1_component_benchmark_scope='actuator/hold precedent NOT H4 moving-response/support/contact validation',
        baseline_launch_does_NOT_require_proposed_method_to_have_passed=True,
        compact_readback_UI_yield_max_rows=static['compact_readback_UI_yield_max_rows'],
        compact_readback_UI_yield_wall_interval_s=static['compact_readback_UI_yield_wall_interval_s'],
        compact_readback_heartbeat_s=static['compact_readback_heartbeat_s'],
        compact_readback_wall_budget_s=static['compact_readback_wall_budget_s'],
        all_archived_environment_rows_required=static['all_archived_environment_rows_required'],
        full_environment_checks_only_PAUSED_before_Play_and_after_Pause=True,
        large_original_archive_never_read_redecoded_or_hashed_inside_Kit=True,
        old_development_trial_outcomes_are_NOT_matched_final_statistics=True,
        model_software_resolution_cases_are_NOT_independent_response_or_robustness_trials=True,
        independent_fresh_frozen_whole_H3_repeats=0, original_H3_MODEL_PASS_count=2, original_H3_MODEL_case_count=8,
        actual_static_verified=False, actual_motion_authorized=False, use_for_final_statistics=False,
        CICS_runtime_integration=False, sensor_or_Nav2_execution=False, real_robot_required=False,
        input_sha256={**bundle['input_sha256'], **bundle['output_sha256'], **model['input_sha256'],
            **model['output_sha256'], **static['input_sha256'],
            **{str(p):preparation.sha(p) for p in (*sources, BUNDLE, MODEL_SCREEN, BENCHMARK, stationary.PROTOCOL)}}))


def load_protocol():
    report = preparation.read_small(PROTOCOL)
    preparation.verify(report['input_sha256'])
    if report != prepare_protocol() or str(preparation.old.ARCHIVE) in report['input_sha256']:
        raise ValueError('Frozen compact route design changed or attempts to hash giant archive')
    return report


def validate_static_record(raw, rows, prep, protocol):
    """Re-evaluate actual records; tmp synthetic fixtures are tests ONLY."""
    expected = dict(version=stationary.VERSION, cohort=preparation.COHORT,
        scenario=execution.SCENARIO, run_id=stationary.RUN_ID, verdict='PASS',
        physics_preparation=str(preparation.LIVE), physics_preparation_sha256=preparation.sha(preparation.LIVE),
        physics_preparation_snapshot=prep, start_xyyaw=protocol['start_xyyaw'], simulation_mass_kg=265.,
        static_protocol=str(stationary.PROTOCOL), static_protocol_sha256=preparation.sha(stationary.PROTOCOL),
        settle_s=stationary.SETTLE_S, sampling_s=stationary.RECORD_S,
        automatic_static_Play_requested=True, source_robot_seed_checked_before_Play=True,
        initial_motion_executed=False, planned_path_executed=False, CICS_input_used=False, Nav2_executed=False,
        caster_joint_state_reset_during_run=False, material_modified_during_run=False,
        motor_gains_modified_by_test=False, source_stage_saved=False,
        query_definition_frozen_before_new_trial=True, exactly_one_reserved_attempt_no_automatic_retry=True,
        use_for_final_statistics=False, actual_drive_admission=False,
        approved_disabled_subfloor_paths=list(preparation.SELECTED), prior_environment_outcomes_NOT_pooled_or_relabelled=True)
    if (any(raw.get(k) != v for k,v in expected.items()) or raw.get('error') or raw.get('evaluation_error')
            or not raw.get('checks') or any(value is not True for value in raw['checks'].values())):
        raise ValueError('Exact successful NEW compact static receipt required; old PASS/FAIL/model cannot substitute')
    final = raw.get('actual_final_timeline', {})
    if (final.get('playing') is not False or final.get('stopped') is not False
            or not math.isfinite(final.get('current_time_s', math.nan)) or final['current_time_s'] <= 0.):
        raise ValueError('Own actual completed static must have ended PAUSED')
    for key in ('compact_environment_before_Play', 'compact_environment_after_Pause'):
        value = raw.get(key, {})
        if (value.get('all_original_row_count') != protocol['all_archived_environment_rows_required']
                or value.get('protected_environment_sha256') != prep['protected_environment_sha256']
                or value.get('only_exact_four_approved_collision_flags_differ_from_original_archive') is not True
                or value.get('timeline_and_Root_Session_unchanged_during_readback') is not True
                or value.get('whole_archive_redecoded_or_hashed_inside_Kit') is not False):
            raise ValueError('Both complete 4997-row compact environment checks required')
    evaluated = stationary.evaluate(rows, raw['start_xyyaw'], raw['contact_report_counts_by_category'],
        raw['contact_callback_errors'], protocol['sampled_cooked_support_queries'],
        raw['actual_sampled_cooked_support_queries'], protocol['support_top_m'])
    probes = stationary.evaluate_subfloor_queries(protocol, raw['actual_sampled_cooked_support_queries'],
        raw['actual_disabled_subfloor_probes'])
    checks = {**evaluated['checks'], **probes['checks']}
    required = ('stationary_observation_completed', 'original_motor_gains_and_limits_unchanged',
        'all_4997_original_environment_rows_unchanged_except_approved_four_flags',
        'nominal_actor_fixed_parameters_unchanged', 'original_support_material_scene_and_hospital_colliders_unchanged',
        'all_original_input_evidence_unchanged', 'contact_callback_monitor_healthy',
        'only_expected_external_support', 'only_known_contact_categories_recorded')
    if (evaluated['verdict'] != 'PASS' or not all(checks.values())
            or any(raw['checks'].get(k) is not True for k in (*checks, *required))):
        raise ValueError('NEW static physical/26-ray/four-probe PASS does not reproduce from actual raw data')
    names = raw.get('final_actual_joint_names', [])
    q, qd = (np.asarray(raw.get(key, []), float) for key in
        ('final_actual_twelve_joint_positions_rad', 'final_actual_twelve_joint_velocities_rad_s'))
    links = raw.get('passive_post_static_link_readback')
    if (len(names) != 12 or len(set(names)) != 12 or q.shape != (12,) or qd.shape != (12,)
            or not np.isfinite(q).all() or not np.isfinite(qd).all()
            or np.asarray(raw.get('final_actual_world_xyyaw', [])).shape != (3,)
            or not np.isfinite(raw['final_actual_world_xyyaw']).all()
            or not links or raw.get('passive_post_static_link_readback_errors') != []):
        raise ValueError('Complete own POST-static body/twelve joints and observed thirteen links required for drive')
    # Check measured gains, poses, quaternion normalization and all 13 links
    # with the same immutable earlier continuity evaluator, using saved values
    # as their own reference here (live values are checked separately).
    transforms = np.asarray(links.get('all_thirteen_link_world_transforms_xyzw', []), float)
    link_names = links.get('link_names', [])
    if transforms.shape != (13,7) or len(link_names) != 13 or len(set(link_names)) != 13 or not np.isfinite(transforms).all():
        raise ValueError('Complete finite thirteen-link static snapshot required')
    if not np.allclose(np.linalg.norm(transforms[:,3:], axis=1),1.,rtol=0,atol=1e-4):
        raise ValueError('Normalized thirteen-link static quaternions required')
    if raw['environment_after_static']['protected_environment_sha256'] != prep['protected_environment_sha256']:
        raise ValueError('Static environment fingerprint differs from its own preparation')
    return dict(verdict='PASS', checks=checks,
        scope='RAW_NEW_START_STATIC_AND_SAMPLED_QUERIES_ONLY_NOT_ROUTE_OR_TRACTION_VALIDATION')


def replay_actual_static():
    if not stationary.RESULT.is_file() or not stationary.TRACE.is_file():
        raise ValueError('Actual NEW compact static JSON AND raw CSV required; model/display/preparation are insufficient')
    if not stationary.RESERVATION.is_file() or stationary.PREFLIGHT_HOLD.exists():
        raise ValueError('Own complete single static reservation/result required; HOLD cannot admit drive')
    prep, protocol = stationary.load_actual_preparation(), stationary.load_protocol()
    reservation = preparation.read_small(stationary.RESERVATION)
    if (reservation.get('run_id') != stationary.RUN_ID or reservation.get('user_authorized') is not True
            or reservation.get('protocol_sha256') != preparation.sha(stationary.PROTOCOL)
            or reservation.get('actual_preparation_sha256') != preparation.sha(preparation.LIVE)):
        raise ValueError('Exact actual single static reservation required')
    raw = preparation.read_small(stationary.RESULT)
    preparation.verify(raw['input_sha256'])
    if raw.get('trace_sha256') != preparation.sha(stationary.TRACE):
        raise ValueError('Actual static raw CSV changed')
    with stationary.TRACE.open(newline='') as stream:
        rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    replay = validate_static_record(raw, rows, prep, protocol)
    return prep, raw, replay


def load_actual_inputs(method):
    if method not in execution.METHODS:
        raise ValueError('Select one exact frozen H4 method')
    protocol = load_protocol()
    bundle, model, plans = load_frozen_bundle()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('This actual static handoff already reserved/ran a route; no retry or second method')
    prep, static, replay = replay_actual_static()
    plan = plans[method]
    if plan['start_world_xyyaw'] != static['start_xyyaw']:
        raise ValueError('Frozen H4 plan and own static start differ; no translation or endpoint recapture')
    return dict(protocol=protocol, bundle=bundle, model=model, plan=plan,
        preparation=prep, static=static, static_replay=replay)


def load_recorded_closeout_review():
    if not CLOSEOUT_REVIEW.is_file():
        raise ValueError('Exact saved retrospective completion review required; no automatic motion')
    record = preparation.read_small(CLOSEOUT_REVIEW)
    preparation.verify(record['input_sha256'])
    if (record != review_recorded_static_closeout() or not all(record['checks'].values())
            or record['actual_drive_admission'] is not False or record['new_static_PASS_created'] is not False):
        raise ValueError('Retrospective review incomplete; never promote a failed raw static record to PASS')
    return record


def prepare_retained_static_protocol():
    review = review_recorded_static_closeout()
    if not all(review['checks'].values()):
        raise ValueError('Current retained state/query diagnosis incomplete; no new static setup')
    original = preparation.read_small(completed_static.stationary.PROTOCOL)
    files=(Path(evidence.__file__),completed_static.LIVE)
    bundle=runtime_source_bundle()
    return canonical(dict(original, version='H4_common_retained_state_prospective_zero_command_static_v1',
        run_id=STATIC_RUN_ID, trial_role='engineering_static_support_NOT_path_performance_or_final_statistics',
        actual_start_history='retained prior zero-command static; NOT a fresh source-authored or independent random entry',
        required_paused_closeout=str(completed_static.LIVE), required_root_sha256=review['root_sha256'],
        required_session_sha256=review['session_sha256'], output=str(STATIC_OUTPUT),
        original_failed_static_and_incomplete_closeout_preserved=True,
        corrected_current_actor_identity_in_unchanged_26_ray_rule=True,
        confirmed_Pause_before_postprocessing_and_common_durable_writer=True,
        initial_motion_and_route_still_require_separate_actual_verification_and_authorization=True,
        one_separately_authorized_trial_only_no_automatic_retry=True,
        source_seed_assignment_or_body_joint_reset=False, new_motion_authorized=False,
        runtime_source_snapshot=str(RUNTIME_SNAPSHOT), runtime_source_snapshot_sha256=hashlib.sha256(bundle).hexdigest(),
        runtime_source_sha256={str(p):preparation.sha(p) for p in (Path(__file__),EXPERIMENT_ENTRY)},
        historical_evidence_pins_immutable_source_snapshot_NOT_mutable_common_runner=True,
        input_sha256={**original['input_sha256'],**review['input_sha256'],
            **{str(path):preparation.sha(path) for path in files},str(RUNTIME_SNAPSHOT):hashlib.sha256(bundle).hexdigest()}))


def runtime_source_bundle():
    """Small deterministic source archive, not another executable version.

    A future fix to the active common runner must not invalidate evidence
    from an already completed trial. Raw records pin this immutable archive;
    live admission separately checks the active source against the protocol.
    """
    buffer=io.BytesIO()
    # STORED avoids zlib-version-dependent bytes between terminal Python and
    # Isaac's embedded Python. The two text sources are only tens of KiB.
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in (Path(__file__),EXPERIMENT_ENTRY):
            info=zipfile.ZipInfo(path.name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o100644<<16
            archive.writestr(info,path.read_bytes())
    return buffer.getvalue()


def write_runtime_source_bundle():
    import os
    payload=runtime_source_bundle()
    if RUNTIME_SNAPSHOT.exists():
        if RUNTIME_SNAPSHOT.read_bytes()!=payload:
            raise ValueError('This trial already froze different runtime sources; no overwrite')
        return
    RUNTIME_SNAPSHOT.parent.mkdir(parents=True,exist_ok=True)
    with RUNTIME_SNAPSHOT.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    if RUNTIME_SNAPSHOT.read_bytes()!=payload:
        raise RuntimeError('Runtime source archive verification failed; archive preserved')


def load_retained_static_protocol():
    protocol=preparation.read_small(STATIC_DESIGN)
    preparation.verify(protocol['input_sha256'])
    if protocol!=prepare_retained_static_protocol():
        raise ValueError('Exact prospective retained-static definition changed; no Play')
    return protocol


async def run_retained_static(fresh=None):
    """ONE prospectively corrected static trial; never a route or a retry loop.

    Shared runtime entry, explicit consent, immutable prior evidence. Keep
    native contact/physics monitors through the CONFIRMED Pause transition.
    All full-environment hashing is outside the physics recording window.
    """
    import asyncio
    from collections import Counter
    import os
    import time
    import carb
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.api.sensors import RigidContactView
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import PhysicsSchemaTools,UsdUtils

    if fresh is None:
        protocol=load_retained_static_protocol()
        raw,_,_=completed_static.saved_records()
        prior=preparation.read_small(completed_static.LIVE)
        prep=raw['physics_preparation_snapshot']
        s=completed_static.stationary
        expected_timeline=prior['actual_final_timeline']
        expected_root,expected_session=protocol['required_root_sha256'],protocol['required_session_sha256']
    else:
        protocol=fresh.protocol
        prior=preparation.read_small(fresh.preparation.LIVE);prep=prior
        raw=dict(final_actual_joint_names=prior['source_constructor_readback']['revolute_joint_names'],
            runtime_motor_baseline=prior['expected_nominal_motor_baseline'])
        s=fresh.stationary;expected_timeline=prior['timeline']
        expected_root,expected_session=prior['Root_sha256'],prior['Session_sha256']
    app=omni.kit.app.get_app(); context=omni.usd.get_context()
    stage=context.get_stage(); timeline=omni.timeline.get_timeline_interface()
    if stage is None or evidence.timeline_state(timeline)!=expected_timeline:
        raise RuntimeError('Exact own pre-static state required; no Play/reset/recovery to create it')
    if (hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest()!=expected_root
            or hashlib.sha256(stage.GetSessionLayer().ExportToString().encode()).hexdigest()!=expected_session):
        raise RuntimeError('Actual retained Root/Session changed; no state restoration')
    result=STATIC_OUTPUT/(STATIC_RUN_ID+'.json'); trace=STATIC_OUTPUT/(STATIC_RUN_ID+'.csv')
    reservation=STATIC_OUTPUT/'single_attempt_reserved.json'
    if any(path.exists() for path in (result,trace,reservation)):
        raise RuntimeError('This single static attempt already reserved/recorded; no overwrite or automatic retry')
    if fresh is None:
        existing=SimulationManager.get_physics_sim_view()
        if existing is None:
            raise RuntimeError('Existing paused tensor view required; no Play to create it')
        validate_complete_tensor(raw,existing.create_articulation_view(cohort.ROBOT+'/base_link'))
        source=raw['runtime_anonymous_source_reference']['fixed_configuration_rows']
    else:
        source=fresh_source_for_live_actor(fresh,stage)['fixed_configuration_rows']
        from pxr import Usd
        made=preparation.old.constructor_for(cohort.ROBOT).__globals__['validate_new_robot'](
            stage,Usd.Stage.Open(str(preparation.ASSET)),protocol['start_xyyaw'])
        if made!=prior['source_constructor_readback']:
            raise RuntimeError('Fresh source start/seed/constructor changed before static Play')
    if s.actor_configuration(stage)!=source:
        raise RuntimeError('All current same-SDK source actor settings must be retained')

    async def environment_readback(phase):
        state=evidence.timeline_state(timeline)
        if state['playing'] or fresh is None and state['stopped']:
            raise RuntimeError('Full environment readback requires own nonplaying state')
        root,session=stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()
        began=heartbeat=yielded=time.monotonic(); rows={}
        def unchanged():
            if (context.get_stage()!=stage or evidence.timeline_state(timeline)!=state
                    or stage.GetRootLayer().ExportToString()!=root or stage.GetSessionLayer().ExportToString()!=session):
                raise RuntimeError('Paused state/layers changed during full environment readback')
            if time.monotonic()-began>protocol['compact_readback_wall_budget_s']:
                raise TimeoutError('Nonmoving environment readback wall budget exceeded')
        for path,row in preparation.compact.stage_rows(stage,preparation.SELECTED,preparation.old.original):
            if path in rows: raise RuntimeError('Duplicate protected environment row')
            rows[path]=row; now=time.monotonic()
            if now-heartbeat>=15.:
                print('[H4 STATIC] %s rows=%d/4997; PAUSED, no motion' % (phase,len(rows)),flush=True)
                heartbeat=now
            if len(rows)%16==0 or now-yielded>=.05:
                unchanged(); await app.next_update_async(); unchanged(); yielded=time.monotonic()
        unchanged()
        manifest=preparation.load_manifest(verify_inputs=False)
        fingerprint=preparation.compact.compare_rows(manifest['compact_environment_rows'],rows,preparation.SELECTED,after_flags=True)
        if len(rows)!=4997 or fingerprint!=prep['protected_environment_sha256']:
            raise RuntimeError('Protected environment differs beyond exact four approved flags')
        if s.critical_environment(stage,prep)!=prior['critical_environment']:
            raise RuntimeError('Nominal masses/material/gravity/support/actor configuration changed')
        return dict(row_count=len(rows),sha256=fingerprint)

    print('[H4 STATIC] Checking retained state + all protected geometry BEFORE Play.',flush=True)
    before=await environment_readback('before_static')
    if fresh is None:
        validate_complete_tensor(raw,existing.create_articulation_view(cohort.ROBOT+'/base_link'))
    elif (evidence.timeline_state(timeline)!=expected_timeline
            or s.actor_configuration(stage)!=source):
        raise RuntimeError('Fresh source state changed during complete preflight')
    evidence.write_exclusive(reservation,dict(run_id=STATIC_RUN_ID,explicit_user_single_static_consent=True,
        protocol_sha256=preparation.sha(STATIC_DESIGN), no_physics_yet=True,no_automatic_retry=True))
    report=dict(version=protocol['version'],run_id=STATIC_RUN_ID,cohort=preparation.COHORT,
        trial_kind=protocol['trial_role'],verdict='HOLD',checks={},automatic_static_Play_requested=False,
        start_xyyaw=protocol['start_xyyaw'],simulation_mass_kg=265.,settle_s=1.,sampling_s=2.,
        actual_drive_admission=False,use_for_final_statistics=False,initial_motion_executed=False,
        planned_path_executed=False,source_seed_or_caster_angle_assigned=False,body_or_joint_reset=False,
        original_static_verdict='FAIL' if fresh is None else 'NOT_REUSED_NEW_SOURCE_STATIC_TRIAL',
        original_static_result=str(completed_static.stationary.RESULT) if fresh is None else None,
        static_protocol=str(STATIC_DESIGN),static_protocol_sha256=preparation.sha(STATIC_DESIGN),
        compact_environment_before_Play=before,input_sha256=dict(protocol['input_sha256'],
            **{str(STATIC_DESIGN):preparation.sha(STATIC_DESIGN)}))
    if fresh is not None:
        report.update(physics_preparation_snapshot=prior,
            runtime_anonymous_source_reference=fresh_source_for_live_actor(fresh,stage),
            environment_after_static=prior['critical_environment'],
            source_authored_seed_verified_before_Play=True,
            prior_component_result_preserved_NOT_a_continuation=True)
        report['input_sha256'][str(fresh.preparation.LIVE)]=preparation.sha(fresh.preparation.LIVE)
    clock={'s':0.}; counts=Counter(); pairs=Counter(); errors=[]
    rows=[];views=[];robot=motor=steer=drive=kp0=kd0=limits0=None
    physics_sub=contact_sub=None; wall_start=time.monotonic()
    def step(dt):
        if not math.isfinite(float(dt)) or dt<=0.:errors.append('Invalid physics step duration')
        else:clock['s']+=float(dt)
    def contacts(headers,data):
        try:
            for header in headers:
                pair=tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
                category=s.classify_contact(pair,prior['critical_environment']['disabled_legacy_colliders'])
                if category not in s.KNOWN_CATEGORIES:raise RuntimeError('Unknown contact category')
                if category!='unrelated':counts[category]+=1;pairs[(pair,category)]+=1
        except Exception as error:errors.append(type(error).__name__+': '+str(error))
    def zero():
        robot.apply_action(ArticulationAction(joint_positions=np.zeros(2,dtype=np.float32),joint_indices=steer))
        robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2,dtype=np.float32),joint_indices=drive))
    def same_gains():
        kp,kd=motor.get_gains()
        return np.array_equal(kp,kp0) and np.array_equal(kd,kd0) and np.array_equal(motor.get_max_efforts(),limits0)
    def pose():
        xyz,quat=robot.get_world_pose();w,x,y,z=map(float,quat)
        if not np.isfinite([*xyz,*quat]).all():raise RuntimeError('Non-finite body pose')
        roll=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)))
        pitch=math.degrees(math.asin(np.clip(2*(w*y-z*x),-1.,1.)))
        yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        return xyz,roll,pitch,yaw
    try:
        physics_sub=omni.physx.get_physx_interface().subscribe_physics_step_events(step)
        contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
        views=[RigidContactView(prim_paths_expr=cohort.ROBOT+'/'+name,filter_paths_expr=[],
            name=STATIC_RUN_ID+'_'+label,prepare_contact_sensors=False,disable_stablization=False) for label,name in s.WHEELS.items()]
        report['automatic_static_Play_requested']=True
        timeline.play()
        for _ in range(5):await app.next_update_async()
        if not timeline.is_playing():raise RuntimeError('Timeline changed during static initialization')
        robot=SingleArticulation(cohort.ROBOT+'/base_link');robot.initialize()
        names=list(robot.dof_names);indices={name:i for i,name in enumerate(names)}
        if len(names)!=12 or set(names)!=set(raw['final_actual_joint_names']):raise RuntimeError('All twelve joints required')
        steer=np.asarray([indices['base_link_to_front_active_steering'],indices['base_link_to_rear_active_steering']],dtype=np.int32)
        drive=np.asarray([indices['front_active_steering_to_drive_wheel'],indices['rear_active_steering_to_drive_wheel']],dtype=np.int32)
        motor=robot.get_articulation_controller();kp,kd=motor.get_gains()
        kp0,kd0,limits0=np.asarray(kp).copy(),np.asarray(kd).copy(),np.asarray(motor.get_max_efforts()).copy()
        if (not np.isfinite(kp0).all() or not np.isfinite(kd0).all()
                or np.isnan(limits0).any() or np.isneginf(limits0).any()):
            raise RuntimeError('Invalid original motor gains/force limits')
        baseline=dict(joint_names=names,kp=kp0.tolist(),kd=kd0.tolist(),
            max_efforts=[float(v) if np.isfinite(v) else 'positive_infinity' for v in limits0])
        if baseline!=raw['runtime_motor_baseline']:raise RuntimeError('Retained original motor baseline mismatch')
        report['runtime_motor_baseline']=baseline
        for view in views:view.initialize()
        sim=omni.physx.get_physx_simulation_interface();stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        encoded=PhysicsSchemaTools.sdfPathToInt(cohort.ROBOT+'/base_link')
        began=previous=clock['s'];next_sample=1.;heartbeat=time.monotonic()
        while clock['s']-began<3.:
            zero();sim.wake_up(stage_id,encoded);await app.next_update_async()
            now=clock['s'];elapsed=now-began
            if context.get_stage()!=stage:raise RuntimeError('Stage changed during static observation')
            if not timeline.is_playing():raise RuntimeError('Timeline changed during static observation')
            if time.monotonic()-wall_start>120. or now-previous>.25:raise TimeoutError('Original static wall/control-gap guard')
            previous=now;xyz,roll,pitch,yaw=pose()
            if math.hypot(xyz[0]-protocol['start_xyyaw'][0],xyz[1]-protocol['start_xyyaw'][1])>.030 or max(abs(roll),abs(pitch))>2.:
                raise RuntimeError('Original static 30mm / 2deg safety guard')
            if errors or any(counts[k] for k in s.BAD_CONTACTS):raise RuntimeError('Unexpected obstacle/nonwheel/legacy contact')
            if not same_gains():raise RuntimeError('Original motor gains/limits changed')
            if time.monotonic()-heartbeat>=15.:
                print('[H4 STATIC] sim=%.3fs samples=%d; zero command, contacts=%s' % (elapsed,len(rows),dict(counts)),flush=True)
                heartbeat=time.monotonic()
            if elapsed<next_sample:continue
            forces=[np.asarray(view.get_net_contact_forces(dt=1./120.),float).reshape(-1,3)[0,2] for view in views]
            if not np.isfinite(forces).all():raise RuntimeError('Non-finite wheel support forces')
            row=dict(time_s=elapsed,x_m=float(xyz[0]),y_m=float(xyz[1]),z_m=float(xyz[2]),roll_deg=roll,pitch_deg=pitch,yaw_deg=yaw)
            row.update({label+'_fz_n':float(value) for label,value in zip(s.WHEELS,forces)})
            rows.append(row);next_sample+=1./60.
        report['checks']['stationary_observation_completed']=True
    except (Exception,asyncio.CancelledError) as error:
        report['error']=type(error).__name__+': '+str(error)
        report['checks']['stationary_observation_completed']=False
    finally:
        try:
            if robot is not None and drive is not None:zero()
        except Exception as error:report['checks']['final_zero_command_sent']=False;report['zero_command_error']=str(error)
        try:
            report['pause_confirmation']=await evidence.confirm_pause(timeline,app.next_update_async)
            report['checks']['actual_Pause_confirmed']=True
        except Exception as error:
            report['pause_error']=str(error);report['checks']['actual_Pause_confirmed']=False
        physics_sub=contact_sub=None
        try:
            if not report['checks']['actual_Pause_confirmed']:raise RuntimeError('Pause not confirmed; no postprocessing')
            report['compact_environment_after_Pause']=await environment_readback('after_static')
            report['checks']['source_actor_parameters_unchanged']=s.actor_configuration(stage)==source
            report['checks']['original_motor_gains_and_limits_unchanged']=bool(motor is not None and same_gains())
            tensor=SimulationManager.get_physics_sim_view().create_articulation_view(cohort.ROBOT+'/base_link')
            report['actual_post_static_tensor_snapshot']=preparation.old.tensor_snapshot(tensor)
            queries,probes=completed_static.frozen_query_functions(dict(carb=carb,omni=omni,c=cohort,protocol=protocol))
            report['actual_sampled_cooked_support_queries']=support=queries()
            report['actual_disabled_subfloor_probes']=absence=probes()
            report['checks']['self_ray_hits_are_exact_known_current_actor_nodes']=all(
                hit['collision'] in source for row in support for hit in row['own_robot_occlusion_hits'])
            report['support_query_evaluation']=query=evaluate_support_queries_for_actor(protocol,support)
            report['absence_probe_evaluation']=negative=s.evaluate_subfloor_queries(protocol,support,absence)
            report['checks'].update({'support_query_'+k:bool(v) for k,v in query['checks'].items()})
            report['checks'].update(negative['checks'])
            if rows:
                physical=completed_static.physical.evaluate_static(rows,protocol['start_xyyaw'],265.,counts,errors)
                report['physical_stationary_evaluation']=physical;report['checks'].update(physical['checks'])
            else:report['checks']['complete_stationary_observation']=False
            preparation.verify(report['input_sha256']);report['checks']['all_frozen_inputs_unchanged']=True
        except Exception as error:
            report['postprocessing_error']=type(error).__name__+': '+str(error)
            report['checks']['postprocessing_completed']=False
        report['checks']['only_expected_external_support']=counts['wheel_support']>0 and not any(counts[k] for k in s.BAD_CONTACTS)
        report['checks']['contact_monitor_healthy']=not errors
        observed=clock['s']>0.
        report['verdict']=('PASS' if not report.get('error') and all(report['checks'].values()) else 'FAIL') if observed else 'HOLD'
        report.update(actual_final_timeline=evidence.timeline_state(timeline),actual_physics_callback_duration_s=clock['s'],
            contact_report_counts_by_category=dict(counts),contact_callback_errors=errors,
            contact_pairs_by_category=[dict(pair=pair,category=category,callback_count=count) for (pair,category),count in sorted(pairs.items())],
            sample_count=len(rows),wall_time_s=time.monotonic()-wall_start,actual_motion_component_or_path_success=False)
        if rows:
            with trace.open('x',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                stream.flush();os.fsync(stream.fileno())
            report['trace_sha256']=preparation.sha(trace)
        write_runtime_evidence(result,report)
        print('=== H4 FRESH FIXED-START STATIC ===' if fresh is not None else '=== H4 PROSPECTIVE CORRECTED RETAINED-STATE STATIC ===',flush=True)
        print('run:',STATIC_RUN_ID,'VERDICT:',report['verdict'],'samples:',len(rows),flush=True)
        print('actual timeline:',report['actual_final_timeline'],flush=True)
        print('failed checks:',[k for k,v in report['checks'].items() if not v],flush=True)
        print('errors:',report.get('error'),report.get('postprocessing_error'),report.get('pause_error'),flush=True)
        print('saved:',result,flush=True)
        print('Static only; no automatic retry. Original FAILs unchanged.',flush=True)
    return report


def validate_complete_tensor(static, tensor):
    actual = preparation.old.tensor_snapshot(tensor)
    result = preparation.old.validate_actual_state(static, actual)
    expected = dict(zip(static['final_actual_joint_names'], static['final_actual_twelve_joint_velocities_rad_s']))
    if not np.array_equal(actual['joint_velocities_rad_s'], [expected[n] for n in actual['names']]):
        raise ValueError('Paused twelve joint velocities changed; no Play/reset to recover')
    return dict(result, all_twelve_joint_velocities_exactly_retained=True,
        all_thirteen_links_checked_against_own_POST_static_not_source_seed=True)


def load_verified_current_static():
    """Replay actual corrected static data, not a marker or old raw FAIL.

    Historical evidence pins its executed-source ZIP. Active shared code may
    now advance to the next action without invalidating that source archive.
    The returned legacy-schema view is in memory only; raw files stay exact.
    """
    path=STATIC_OUTPUT/(STATIC_RUN_ID+'.json');trace=STATIC_OUTPUT/(STATIC_RUN_ID+'.csv')
    reservation=STATIC_OUTPUT/'single_attempt_reserved.json'
    if not all(p.is_file() for p in (path,trace,reservation)):
        raise ValueError('Actual corrected static JSON, CSV and its exclusive reservation required')
    raw=preparation.read_small(path);protocol=preparation.read_small(STATIC_DESIGN)
    preparation.verify(raw['input_sha256']);preparation.verify(protocol['input_sha256'])
    receipt=preparation.read_small(reservation)
    if (raw['run_id']!=STATIC_RUN_ID or raw['version']!=protocol['version']
            or raw['cohort']!=preparation.COHORT or raw['trial_kind']!=protocol['trial_role']
            or raw['verdict']!='PASS' or not raw['checks'] or any(v is not True for v in raw['checks'].values())
            or raw['static_protocol_sha256']!=preparation.sha(STATIC_DESIGN)
            or raw['trace_sha256']!=preparation.sha(trace)
            or receipt.get('run_id')!=STATIC_RUN_ID or receipt.get('explicit_user_single_static_consent') is not True
            or receipt.get('protocol_sha256')!=preparation.sha(STATIC_DESIGN)
            or (raw['settle_s'],raw['sampling_s'],raw['simulation_mass_kg'])!=(1.,2.,265.)
            or raw['actual_drive_admission'] is not False or raw['initial_motion_executed'] is not False
            or raw['planned_path_executed'] is not False or raw['use_for_final_statistics'] is not False
            or raw['contact_callback_errors'] or raw.get('error') or raw.get('postprocessing_error') or raw.get('pause_error')):
        raise ValueError('Own corrected actual static PASS and unchanged raw evidence required; no relabelling')
    state=raw['actual_final_timeline']
    if (state['playing'] or state['stopped'] or raw['pause_confirmation']['after']!=state
            or not math.isfinite(state['current_time_s']) or raw['actual_physics_callback_duration_s']<3.):
        raise ValueError('Completed physics observation and confirmed actual PAUSED boundary required')
    old,_,_=completed_static.saved_records()
    # The compact receipt stores mass inside the constructor readback, while
    # the shared path engine expects its explicit scalar. Bind the verified
    # actual static value in memory; never edit the preparation/raw receipt.
    prep=dict(old['physics_preparation_snapshot'],simulation_mass_kg=raw['simulation_mass_kg'])
    s=completed_static.stationary
    for key in ('compact_environment_before_Play','compact_environment_after_Pause'):
        if raw[key]!=dict(row_count=4997,sha256=prep['protected_environment_sha256']):
            raise ValueError('All protected rows required both before and after the corrected static')
    with trace.open(newline='') as stream:
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    physical=completed_static.physical.evaluate_static(rows,raw['start_xyyaw'],265.,
        raw['contact_report_counts_by_category'],raw['contact_callback_errors'])
    support=evaluate_support_queries_for_actor(protocol,raw['actual_sampled_cooked_support_queries'])
    probes=s.evaluate_subfloor_queries(protocol,raw['actual_sampled_cooked_support_queries'],raw['actual_disabled_subfloor_probes'])
    required={'stationary_observation_completed','actual_Pause_confirmed','source_actor_parameters_unchanged',
        'original_motor_gains_and_limits_unchanged','self_ray_hits_are_exact_known_current_actor_nodes',
        'all_frozen_inputs_unchanged','only_expected_external_support','contact_monitor_healthy',
        *physical['checks'],*('support_query_'+k for k in support['checks']),*probes['checks']}
    if (len(rows)!=raw['sample_count'] or physical['verdict']!='PASS'
            or any(raw['checks'].get(k) is not True for k in required)
            or raw['start_xyyaw']!=old['start_xyyaw']
            or not all(support['checks'].values()) or not all(probes['checks'].values())
            or set(raw['contact_report_counts_by_category'])-set(s.KNOWN_CATEGORIES)
            or raw['runtime_motor_baseline']!=old['runtime_motor_baseline']):
        raise ValueError('Actual static CSV/support/probe/contact/gain replay failed')
    counts={}
    critical=preparation.read_small(completed_static.LIVE)['critical_environment']
    for row in raw['contact_pairs_by_category']:
        pair=row['pair'];category=row['category'];count=row['callback_count']
        if (len(pair)!=2 or not isinstance(count,int) or isinstance(count,bool) or count<=0
                or s.classify_contact(pair,critical['disabled_legacy_colliders'])!=category):
            raise ValueError('Actual static contact identity/count mismatch')
        counts[category]=counts.get(category,0)+count
    if counts!=raw['contact_report_counts_by_category']:
        raise ValueError('Actual static contact pair totals disagree with its monitor')
    actual=raw['actual_post_static_tensor_snapshot']
    if (len(set(actual['names']))!=12 or set(actual['names'])!=set(old['final_actual_joint_names'])
            or len(set(actual['link_names']))!=13 or actual['link_names']!=old['passive_post_static_link_readback']['link_names']):
        raise ValueError('Complete actual twelve-joint / thirteen-link correspondence required')
    for key,shape in (('transform',(7,)),('velocity',(6,)),('positions',(12,)),
            ('joint_velocities_rad_s',(12,)),('joint_position_targets_rad',(12,)),('joint_velocity_targets_rad_s',(12,)),
            ('all_thirteen_link_world_transforms_xyzw',(13,7)),
            ('all_thirteen_link_velocities_at_CoM',(13,6))):
        values=np.asarray(actual[key],float)
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError('Complete finite own-static tensor required: '+key)
    if (not np.allclose(np.linalg.norm(np.asarray(actual['all_thirteen_link_world_transforms_xyzw'])[:,3:],axis=1),1.,atol=1e-4,rtol=0)
            or not math.isclose(float(np.linalg.norm(actual['transform'][3:])),1.,abs_tol=1e-4)):
        raise ValueError('Normalized measured quaternions required')
    x,y,z,qx,qy,qz,qw=actual['transform']
    yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    normalized=dict(raw,final_actual_world_xyyaw=[x,y,yaw],final_actual_joint_names=actual['names'],
        final_actual_twelve_joint_positions_rad=actual['positions'],
        final_actual_twelve_joint_velocities_rad_s=actual['joint_velocities_rad_s'],
        passive_post_static_link_readback=actual,physics_preparation_snapshot=prep,
        runtime_anonymous_source_reference=old['runtime_anonymous_source_reference'],
        environment_after_static=critical)
    # Reuse the old strict continuity/gain evaluator against THIS own current
    # post-static snapshot, never the retired source seed or old failed pose.
    preparation.old.validate_actual_state(normalized,actual)
    return prep,normalized,dict(verdict='PASS',physical=physical,support=support,probes=probes,
        result=str(path),trace=str(trace),actual_trial_count=1,qualified_final_cells=0,
        scope='OWN_CORRECTED_STATIC_ONLY_NOT_INITIAL_MOTION_OR_PATH_ADMISSION')


def initial_motion_source_bundle():
    buffer=io.BytesIO()
    sources=(Path(__file__),EXPERIMENT_ENTRY,ENTRY,INITIAL_GENERATOR)
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in sources:
            info=zipfile.ZipInfo(path.name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
            info.external_attr=0o100644<<16;archive.writestr(info,path.read_bytes())
    return buffer.getvalue()


def prepare_initial_motion_protocol():
    prep,static,replay=load_verified_current_static()
    design=preparation.read_small(INITIAL_DESIGN);preparation.verify(design['input_sha256'])
    if (design['status']!='OFFLINE_COMPONENT_MODEL_SCREEN_PASSED_ACTUAL_STATIC_LIVE_ADAPTER_AND_CONSENT_PENDING'
            or design['distance_m']!=.65 or design['actual_motion_authorized'] is not False
            or any(c['model_status']!='MODEL_SCREEN_PASS' for c in design['original_common_executor_MODEL_cases'])):
        raise ValueError('Exact fixed common-prefix MODEL design required; no substitute or length search')
    original=preparation.read_small(STATIC_DESIGN)
    source_files=(Path(__file__),EXPERIMENT_ENTRY,ENTRY,INITIAL_GENERATOR)
    return canonical(dict(version='H4_one_shared_v3_common_prefix_initial_motion_admission_v1',
        role='engineering_0p65m_initial_motion_NOT_full_route_or_method_performance',
        new_robot_root=cohort.ROBOT,required_paused_timeline=static['actual_final_timeline'],
        required_actual_static_result=replay['result'],required_actual_static_trace=replay['trace'],
        diagnostic_design=str(INITIAL_DESIGN),diagnostic_only=True,reference_distance_m=.65,
        separate_explicit_user_motion_consent_required=True,one_component_only_no_automatic_retry=True,
        command_limits=execution.COMMAND_LIMITS,guards=execution.GUARDS,acceptance=execution.ACCEPTANCE,
        terminal_controller_config=execution.CONTROLLER_CONFIG,initial_pre_hold_s=2.,final_hold_s=5.,fixed_tail_s=3.,
        existing_v3_bounded_terminal_corrections_logged_separately=True,
        environment_variant='four_exact_original_subfloor_collision_flags_disabled_Session_only',
        compact_readback_wall_budget_s=original['compact_readback_wall_budget_s'],
        compact_readback_heartbeat_s=original['compact_readback_heartbeat_s'],
        compact_readback_UI_yield_max_rows=original['compact_readback_UI_yield_max_rows'],
        compact_readback_UI_yield_wall_interval_s=original['compact_readback_UI_yield_wall_interval_s'],
        all_archived_environment_rows_required=4997,protected_environment_sha256=prep['protected_environment_sha256'],
        runtime_source_snapshot=str(INITIAL_SOURCES),
        runtime_source_sha256={str(p):preparation.sha(p) for p in source_files},
        source_seed_or_pose_joint_assignment=False,actual_motion_authorized=False,
        use_for_final_statistics=False,CICS_runtime_integration=False,whole_route_admission=False,
        input_sha256={**static['input_sha256'],**design['input_sha256'],
            **{str(Path(p)):preparation.sha(Path(p)) for p in (replay['result'],replay['trace'],INITIAL_DESIGN)},
            str(INITIAL_SOURCES):hashlib.sha256(initial_motion_source_bundle()).hexdigest()}))


def load_initial_motion_inputs():
    protocol=preparation.read_small(INITIAL_PROTOCOL);preparation.verify(protocol['input_sha256'])
    if protocol!=prepare_initial_motion_protocol():
        raise ValueError('Frozen initial-motion protocol/current runtime changed; no Play')
    if INITIAL_OUTPUT.exists() and any(INITIAL_OUTPUT.iterdir()):
        raise ValueError('Single initial-motion attempt already reserved/recorded; no retry')
    prep,static,replay=load_verified_current_static()
    bundle,model,_=load_frozen_bundle()
    design=preparation.read_small(INITIAL_DESIGN)
    return dict(protocol=protocol,bundle=bundle,model=model,plan=design['diagnostic_reference'],
        plan_path=INITIAL_DESIGN,preparation=prep,static=static,static_replay=replay,
        static_protocol=preparation.read_small(STATIC_DESIGN))


def validate_initial_motion_environment(stage,prep,protocol):
    """Same 4997 protected rows + 145 actor parameters, not a stale Root hash.

    Physics legitimately updated body state in the new static trial. Compare
    actual dynamics/configuration with that trial, not pre-first-static text.
    The caller separately checks all 13 links / 12 q/qd and paused time.
    """
    old,_,_=completed_static.saved_records()
    s=completed_static.stationary
    if (str(stage.GetRootLayer().realPath)!=str(preparation.SOURCE)
            or s.actor_configuration(stage)!=old['runtime_anonymous_source_reference']['fixed_configuration_rows']):
        raise ValueError('Original source stage and all own same-SDK actor parameters required')
    return s.critical_environment(stage,prep)


def normalize_runtime_evidence(value, conversions=None, path='$'):
    """Represent finite NumPy scalar observations as exact JSON primitives.

    No rounding, default=str, array expansion, gain alteration or PASS/FAIL
    change. Keep the historical strict writer untouched. float32/float64
    values embed exactly in Python's float; wider/complex values are rejected.
    """
    if conversions is None:conversions=[]
    if isinstance(value,np.generic):
        if (not isinstance(value,(np.floating,np.integer,np.bool_))
                or isinstance(value,np.floating) and value.dtype.itemsize>8):
            raise TypeError('Unsupported NumPy evidence scalar at '+path+': '+str(value.dtype))
        original=value;value=value.item()
        if type(value) not in (float,int,bool) or type(value) is float and not math.isfinite(value):
            raise ValueError('Non-finite/nonprimitive measured scalar at '+path)
        if not bool(original==value):raise ValueError('Inexact measured scalar conversion at '+path)
        conversions.append(dict(path=path,from_type=type(original).__module__+'.'+type(original).__name__,
            to_type=type(value).__name__,value=value,no_rounding_or_value_change=True))
    if type(value) is dict:
        if any(type(key) is not str for key in value):raise TypeError('Evidence keys must remain strings')
        return {key:normalize_runtime_evidence(item,conversions,path+'.'+key) for key,item in value.items()}
    if type(value) in (list,tuple):
        return [normalize_runtime_evidence(item,conversions,path+'['+str(i)+']') for i,item in enumerate(value)]
    return evidence.json_value(value)


def write_runtime_evidence(path,value):
    """Shared adapter for later admitted trials; not a physics change."""
    return evidence.write_exclusive(path,normalize_runtime_evidence(value))


def initial_motion_trace_replay(report,rows):
    """Recheck the already saved fixed tail without executing another hold."""
    import hospital_stop_analysis as stop
    if len(rows)!=report['sample_count'] or len(rows)<10:
        raise ValueError('Saved CSV must contain every completed sample')
    times=[float(r['time_s']) for r in rows]
    if any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Actual increasing physics timestamps required')
    tail=[r for r in rows if r['phase']=='terminal_hold' and float(r['phase_time_s'])>=2.]
    if report['verdict']!='PASS':
        return dict(original_controller_verdict=report['verdict'],all_rows_preserved=True,
            scope='No failed original verdict upgraded; saved CSV inspected only')
    analysis=stop.analyze_stop_trace(tail)
    checks=dict(one_fixed_5s_hold=report['fixed_hold_attempt_count']==1,
        fixed_tail_covers_3s=analysis['duration_s']>=2.9,
        endpoint_under_30mm=all(float(r['goal_position_error_m'])<=.030 for r in tail),
        heading_under_2deg=all(abs(float(r['goal_heading_error_deg']))<=2. for r in tail),
        measured_stop_under_5mm_s=all(math.hypot(float(r['vx_physx_m_s']),float(r['vy_physx_m_s']))<=.005 for r in tail),
        filtered_stop_under_5mm_s=all(float(r['pose_filtered_speed_m_s'])<=.005 for r in tail),
        fixed_tail_standstill=analysis['development_standstill_check']=='PASS',
        progress_over_95percent=float(rows[-1]['path_progress_m'])>=.95*report['reference_polyline_length_m'],
        original_controller_checks_all_true=bool(report['checks']) and all(v is True for v in report['checks'].values()))
    if not all(checks.values()):raise ValueError('Saved actual component CSV contradicts its in-memory verdict: '+str(checks))
    return dict(original_controller_verdict='PASS',checks=checks,standstill_analysis=analysis,
        last_position_error_m=float(rows[-1]['goal_position_error_m']),
        last_heading_error_deg=float(rows[-1]['goal_heading_error_deg']),
        actual_new_trial_or_hold=False,whole_H4_task_or_final_statistics=False)


def _initial_motion_failure_frame(tasks=None):
    """Pin ONLY this already-finished task and its original exception frame."""
    import asyncio
    import builtins
    import gc
    if tasks is None:
        tasks=getattr(builtins,'ijcas_h4_completed_motion_tasks',None)
    if tasks is None:
        # Exact type avoids inspecting torch's unrelated deprecated proxy
        # objects while walking the GC cache. Do not collect or cancel tasks.
        tasks=[obj for obj in gc.get_objects() if type(obj) is asyncio.Task]
    candidates=[]
    for task in tasks:
        if not task.done() or task.cancelled():continue
        error=task.exception()
        if not isinstance(error,TypeError) or not str(error).startswith('Unsupported evidence type: '):continue
        tb=error.__traceback__
        while tb is not None:
            frame=tb.tb_frame;report=frame.f_locals.get('report')
            if (frame.f_code.co_name=='_run_h4_path' and type(report) is dict
                    and report.get('run_id')==INITIAL_RUN_ID):
                if all(frame is not saved[1] for saved in candidates):candidates.append((task,frame,error))
            tb=tb.tb_next
    if len(candidates)!=1:
        raise ValueError('Exactly ONE retained completed initial-motion serialization exception is required; no replay/recovery')
    builtins.ijcas_h4_initial_motion_serialization_exception=candidates[0]
    return candidates[0]


async def finalize_initial_motion_serialization(tasks=None):
    """GET-only closeout of the SAME completed trial; no timeline command.

    Capture the untouched completed report before inspection, independently
    replay its CSV, require its exact actual PAUSED timestamp and whole tensor,
    and repeat the existing full-environment GET. Create the originally absent
    JSON once. Never reinterpret a failed verdict or issue another command.
    """
    import csv
    task,frame,error=_initial_motion_failure_frame(tasks)
    local,scope=frame.f_locals,frame.f_globals
    original=local['report'];conversions=[]
    report=normalize_runtime_evidence(original,conversions)
    result=Path(scope['summary_path']);trace=Path(scope['trace_path'])
    expected=INITIAL_OUTPUT/(INITIAL_RUN_ID+'.json')
    if (result!=expected or trace!=expected.with_suffix('.csv')
            or report.get('trial_kind')!='H4_0p65m_engineering_initial_motion_and_measured_stop'
            or report.get('use_for_final_statistics') is not False
            or report.get('selected_frozen_H4_route_PhysX_verified') is not False):
        raise ValueError('Exact own initial-motion result/CSV role required; no old-route substitution')
    if result.exists():raise ValueError('Original result already exists; no overwrite or second finalization')
    source_text=Path(__file__).read_text(encoding='utf-8')
    capture=result.parent/'completed_report_before_JSON_finalization.json'
    archived=dict(status='SAME_COMPLETED_REPORT_CAPTURED_NOT_YET_LIVE_CLOSEOUT_VERIFIED',
        original_serialization_error=type(error).__name__+': '+str(error),
        original_controller_verdict=report['verdict'],completed_report=report,scalar_representation_changes=conversions,
        no_new_motion_or_timeline_command=True,no_values_thresholds_or_verdict_revised=True,
        closeout_source=dict(path=str(Path(__file__)),sha256=hashlib.sha256(source_text.encode()).hexdigest(),source_text=source_text))
    if capture.exists():
        if preparation.read_small(capture)!=archived:raise ValueError('A different capture already exists; preserve it')
    else:evidence.write_exclusive(capture,archived)
    stage,timeline=scope['stage'],scope['timeline']
    before=evidence.timeline_state(timeline)
    if (before['playing'] or before['stopped'] or before!=report['actual_final_timeline']
            or report.get('checks',{}).get('actual_route_Pause_confirmed') is not True
            or report.get('original_motor_gains_restored') is not True):
        raise ValueError('Own recorded PAUSED timestamp and restored gains required; no Pause/Play/reset recovery')
    if preparation.sha(trace)!=report['trace_sha256']:
        raise ValueError('Original physical CSV bytes changed')
    with trace.open(newline='') as stream:rows=list(csv.DictReader(stream))
    replay=initial_motion_trace_replay(report,rows)
    preparation.verify(report['input_sha256'])
    original_root,original_session=stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()
    tensor=scope['SimulationManager'].get_physics_sim_view()
    if tensor is None:raise ValueError('Existing retained tensor view required; no initialization')
    actual=scope['c'].old.tensor_snapshot(tensor.create_articulation_view(scope['BASE']))
    if normalize_runtime_evidence(actual)!=report['actual_post_initial_motion_tensor_snapshot']:
        raise ValueError('Own actual completed 13 links / 12 joints / gains changed; no restoration')
    print('[H4 SAME-TRIAL CLOSEOUT] Report captured; checking all original environment rows, GET ONLY.',flush=True)
    environment=await scope['compact_environment_readback']('same_trial_serialization_closeout')
    if (environment['all_original_row_count']!=4997
            or environment['protected_environment_sha256']!=scope['preparation']['protected_environment_sha256']
            or scope['launch']['validate_environment'](stage,scope['preparation'],scope['static_protocol'])!=scope['environment']
            or evidence.timeline_state(timeline)!=before
            or stage.GetRootLayer().ExportToString()!=original_root
            or stage.GetSessionLayer().ExportToString()!=original_session
            or normalize_runtime_evidence(scope['c'].old.tensor_snapshot(tensor.create_articulation_view(scope['BASE'])))!=report['actual_post_initial_motion_tensor_snapshot']):
        raise ValueError('Unchanged actual environment/layers/timestamp/tensor required; no command permitted')
    report['serialization_closeout']=dict(original_error=type(error).__name__+': '+str(error),
        original_completed_report_capture=str(capture),capture_sha256=preparation.sha(capture),
        scalar_representation_changes=conversions,no_rounding_fitting_or_verdict_change=True,
        original_CSV_byte_sha256=report['trace_sha256'],same_trial_fixed_tail_replay=replay,
        whole_protected_environment_rechecked=environment,actual_PAUSED_time_unchanged=before,
        Root_Session_13_links_12_joints_velocities_targets_and_gains_unchanged=True,
        new_trial_count=0,Play_Pause_Stop_reset_or_control_command_executed=False,
        whole_H4_or_final_comparison_success_claimed=False)
    evidence.write_exclusive(result,report)
    print('=== H4 SAME INITIAL-MOTION TRIAL / REPRESENTATION-ONLY FINALIZATION ===',flush=True)
    print('Original controller VERDICT:',report['verdict'],flush=True)
    print('terminal mm/deg:',1000*report.get('terminal_position_error_m',0.),report.get('terminal_heading_error_deg'),flush=True)
    print('failed checks:',[k for k,v in report['checks'].items() if not v],flush=True)
    print('same actual timeline:',before,'; NEW trial count: 0',flush=True)
    print('scalar fields normalized:',[r['path'] for r in conversions],flush=True)
    print('saved:',result,flush=True)
    print('No new motion, retry, full H4 route or final statistical claim.',flush=True)
    return report


def schedule_initial_motion_serialization_closeout():
    import asyncio
    import builtins
    import traceback
    pinned=_initial_motion_failure_frame()
    task=getattr(builtins,'ijcas_h4_initial_motion_serialization_closeout_task',None)
    if task is not None and not task.done():raise RuntimeError('Same-trial closeout already active; no duplicate')
    task=asyncio.ensure_future(finalize_initial_motion_serialization([pinned[0]]))
    builtins.ijcas_h4_initial_motion_serialization_closeout_task=task
    def done(future):
        if future.cancelled():print('Same-trial finalization cancelled; captured evidence retained.',flush=True);return
        error=future.exception()
        if error is not None:traceback.print_exception(type(error),error,error.__traceback__)
    task.add_done_callback(done)
    print('[H4 SAME-TRIAL CLOSEOUT] Scheduled saved-report + actual PAUSED GETs ONLY; no trial or timeline command.',flush=True)
    return task


def validate_environment(stage, prep, protocol):
    expected_root = protocol.get('required_actual_Root_sha256') or preparation.read_small(preparation.PROTOCOL)['protected_Root_text_sha256']
    if hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest() != expected_root:
        raise ValueError('Original Root text changed; do not save/reopen to recover')
    if stationary.actor_configuration(stage) != protocol['nominal_actor_fixed_configuration']:
        raise ValueError('Nominal source actor fixed parameters changed; no radius/gain/scale waiver')
    return stationary.critical_environment(stage, prep)


def physics_baseline_digest(prep, environment, runtime_motor_baseline=None, *, source_settings=None):
    """Bind physics and full environment, not a receipt filename or verdict."""
    bodies = [{**row, 'path':row['path'].replace(cohort.ROBOT+'/', 'ROBOT/', 1)}
        for row in environment['rigid_bodies']]
    return digest(dict(simulation_mass_kg=environment['simulation_mass_kg'], rigid_bodies=bodies,
        physics_hz=environment['physics_hz'], source_robot_asset_sha256=preparation.ASSET_SHA,
        nominal_actor_recipe_sha256=preparation.sha(stationary.ACTOR_RECIPE) if source_settings is None else digest(source_settings),
        protected_full_environment_sha256=environment['protected_environment_sha256'],
        approved_disabled_subfloor_colliders=environment['approved_disabled_subfloor_colliders'],
        disabled_legacy_colliders=environment['disabled_legacy_colliders'],
        all_twelve_runtime_motor_settings=runtime_motor_baseline))


def _bound(function, **bindings):
    """Private callable with explicit cohort identities; no module mutation."""
    bound=FunctionType(function.__code__,dict(function.__globals__,**bindings),
        function.__name__,function.__defaults__,function.__closure__)
    bound.__kwdefaults__=None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    return bound


def fresh_trial_context(protocol):
    """Bind namespaces, not controller/evaluator bytecode or physical limits."""
    import re
    name=protocol['cohort']
    if not re.fullmatch(r'cohort_[0-9]{2}_full_task_r[0-9]{2}',name):
        raise ValueError('Explicit bounded fresh H4 cohort identifier required')
    target=FRESH_TRIAL_ROOT/name
    robot='/World/patient_transport_h4_'+name
    marker='/World/IJCAS_HospitalH4_'+name
    if (protocol['new_robot_root']!=robot or protocol['marker']!=marker
            or Path(protocol['output_directory'])!=target
            or protocol['method'] not in execution.METHODS):
        raise ValueError('Fresh cohort paths/method do not match their frozen identities')
    p=SimpleNamespace(**vars(preparation))
    p.ROBOT,p.MARKER,p.OLD_ROBOT,p.COHORT=robot,marker,protocol['retired_robot_root'],name
    p.LIVE=target/'preparation.json'
    c=SimpleNamespace(**vars(cohort));c.ROBOT,c.MARKER,c.COHORT=robot,marker,name
    c.LIVE=p.LIVE
    s=SimpleNamespace(**vars(stationary));s.ROBOT,s.MARKER,s.preparation=robot,marker,p
    s.actor_configuration=_bound(stationary.actor_configuration,ROBOT=robot)
    s.critical_environment=_bound(stationary.critical_environment,ROBOT=robot,preparation=p)
    s.classify_contact=_bound(stationary.classify_contact,ROBOT=robot)
    s.RESULT=target/'static'/'hospital_h4_fresh_static_r01.json';s.TRACE=s.RESULT.with_suffix('.csv')
    s.PROTOCOL=target/'protocol.json'
    query=_bound(evaluate_support_queries_for_actor,cohort=c,preparation=p)
    return SimpleNamespace(protocol=protocol,output=target,preparation=p,cohort=c,stationary=s,
        support_evaluator=query,static_run_id=s.RESULT.stem,
        route_run_id='hospital_h4_'+name+'_'+protocol['method']+'_r01',
        protocol_path=target/'protocol.json',source_archive=target/'runtime_sources.zip')


def fresh_source_reference(context, namespace_present=False):
    """Same-SDK source reconstruction, no live stage or physics access."""
    from pxr import Usd,UsdShade
    stage=Usd.Stage.CreateInMemory('H4_FRESH_SOURCE_REFERENCE_ONLY.usda')
    UsdShade.Material.Define(stage,preparation.old.constructor.MATERIAL)
    made=preparation.old.constructor_for(context.cohort.ROBOT)(stage,context.protocol['start_xyyaw'])
    if namespace_present:
        helper=completed_static.stationary.reference
        _bound(helper.add_metadata_to_anonymous_layer,ROBOT=context.cohort.ROBOT)(stage.GetRootLayer())
    return dict(source_constructor_readback=made,
        fixed_configuration_rows=context.stationary.actor_configuration(stage),
        namespace_unvalued_declaration_in_reference=namespace_present,
        created_in_anonymous_noncontext_stage_only=True)


def fresh_source_for_live_actor(context,stage):
    """Allow only the previously reviewed exact unvalued SDK namespace spec.

    Every physical/default/sample/relationship remains in the full same-SDK
    actor comparison. Do not add, erase or repair a live namespace attribute.
    """
    from pxr import Sdf
    path=context.cohort.ROBOT+'.isaac:namespace'
    actual=None
    for layer in stage.GetLayerStack():
        spec=layer.GetPropertyAtPath(path)
        if spec:
            info={key:preparation.old.fields.semantic_value(spec.GetInfo(key)) for key in spec.ListInfoKeys()}
            expected={'custom':False,'displayName':'Namespace',
                'documentation':'Namespace of the prim in Isaac Sim','typeName':'string',
                'variability':preparation.old.fields.semantic_value(Sdf.VariabilityVarying)}
            if info!=expected:
                raise ValueError('Unreviewed namespace value/samples/metadata; no actor parameter waiver')
            actual=spec
    reference=fresh_source_reference(context,actual is not None)
    if context.stationary.actor_configuration(stage)!=reference['fixed_configuration_rows']:
        raise ValueError('New actor differs from complete same-SDK source configuration')
    return reference


def fresh_runtime_source_bundle(*,include_visual=False):
    """Small immutable runtime archive; old trials keep their own source ZIP."""
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        sources=(Path(__file__),EXPERIMENT_ENTRY,ENTRY,INITIAL_GENERATOR)
        if include_visual:
            sources+= (_ijcas_path(ROOT, 'isaac/scripts/isaac_refresh_ijcas_hospital_h4_visual_instances_once.py'),
                _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_visual_instance_refresh.py'))
        for source in sources:
            item=zipfile.ZipInfo(str(source.relative_to(ROOT)),date_time=(1980,1,1,0,0,0))
            item.external_attr=0o100644<<16
            archive.writestr(item,source.read_bytes())
    return buffer.getvalue()


def common_resource_contract(simulation_budget_s,plans):
    """Prospective resource version shared by ALL four unchanged methods.

    Round the largest frozen nominal schedule + common pre-hold/terminal/hold
    allocation to whole minutes. This is not a physical completion guarantee
    or a post-hoc extension of any original 180 s outcome.
    """
    if type(simulation_budget_s) not in (float,int) or simulation_budget_s!=300.:
        raise ValueError('Only the separately approved common 300 s resource version is supported')
    if set(plans)!=set(execution.METHODS):
        raise ValueError('All four frozen methods required for a common resource contract')
    schedules={method:sum(.05/action['target_speed_m_s'] for action in plan['selected_actions'])
        for method,plan in plans.items()}
    terminal=execution.CONTROLLER_CONFIG['terminal_timeout_s']
    total=max(schedules.values())+2.+terminal+5.
    if 60.*math.ceil(total/60.)!=simulation_budget_s:
        raise ValueError('Common whole-minute resource derivation differs from the approved definition')
    return canonical(dict(version='H4_ALL_FOUR_COMMON_300SIM_900WALL_DEVELOPMENT_V1',
        methods=list(execution.METHODS),maximum_simulation_duration_s=300.,maximum_wall_time_s=900.,
        prior_development_simulation_limit_s=180.,original_180s_outcomes_preserved_NOT_pooled=True,
        nominal_action_schedule_s=schedules,initial_pre_hold_s=2.,
        common_terminal_allocation_s=terminal,fixed_terminal_hold_s=5.,
        largest_nominal_plus_common_allocations_s=total,round_up_to_whole_minute_s=60.,
        common_to_all_methods_NOT_P_specific=True,physical_worst_case_or_success_guarantee=False,
        original_plan_control_gains_physics_safety_acceptance_unchanged=True))


def load_terminal_connection_bundle(path):
    """Separate prospective planner inputs; never replace the original bundle."""
    path=Path(path).resolve();report=preparation.read_small(path)
    if (report.get('version')!='H4_ALL_FOUR_COMMON_TERMINAL_INTERFACE_NATIVE_BUNDLE_V2'
            or report.get('actual_motion_authorized') is not False
            or report.get('use_for_final_statistics') is not False
            or report.get('original_bundle')!=str(BUNDLE)
            or report.get('additional_common_pose_interface_predicate_is_NEW_planning_version') is not True):
        raise ValueError('Explicit separate all-four terminal connection bundle required')
    preparation.verify(report['input_sha256']);preparation.verify(report['output_sha256'])
    old_bundle,_,old_plans=load_frozen_bundle()
    candidate=preparation.read_small(report['terminal_connection_candidate'])
    nominal=preparation.read_small(report['nominal_connection_review'])
    if (candidate['version']!='prospective_common_terminal_pose_interface_candidate_v1'
            or candidate['adopted_by_any_live_runner'] or candidate['actual_motion_authorized']
            or candidate['new_PhysX_trial_count']!=0 or len(candidate['methods'])!=4
            or (candidate['common_remaining_reference_distance_m'],candidate['common_pose_band_m'],candidate['common_heading_band_deg'])!=(.025,.030,2.)
            or candidate['same_original_pool_and_own_native_cost_for_all_four'] is not True
            or candidate['no_actual_response_or_outcome_used_in_candidate_selection'] is not True
            or nominal['version']!='prospective_terminal_connection_original_nominal_executor_review_v1'
            or nominal['actual_motion_authorized'] or nominal['new_PhysX_trial_count']!=0
            or len(nominal['cases'])!=8
            or {(row['method'],row['integration_dt_s']) for row in nominal['cases']}!=
                {(method,dt) for method in execution.METHODS for dt in (.010,.020)}
            or nominal['input_sha256'].get(report['terminal_connection_candidate'])!=preparation.sha(report['terminal_connection_candidate'])
            or any(row['model_status']!='MODEL_SCREEN_PASS' or not all(row['checks'].values()) for row in nominal['cases'])):
        raise ValueError('All original-model/native and nominal checks required, not motion admission')
    preparation.verify(candidate['input_sha256']);preparation.verify(nominal['input_sha256'])
    preparation.verify(nominal['output_sha256'])
    proposals={row['method']:row for row in candidate['methods']}
    plans={p['method']:p for p in (preparation.read_small(path) for path in report['plans'])}
    if tuple(plans)!=execution.METHODS or set(proposals)!=set(plans):
        raise ValueError('No missing/repeated method or P-only geometry rule')
    invariant=('allowed_modes','planner_dynamics','start_world_xyyaw','goal_world_xyyaw','goal_local_xyyaw',
        'common_max_command_speed_m_s','common_max_acceleration_m_s2','full_envelope_clearance_m','footprint',
        'planning_occupancy_digest','geometry_manifest','geometry_manifest_sha256','shared_candidate_pool_sha256',
        'shared_candidate_count','shared_executor_version','shared_controller_config','shared_command_limits',
        'shared_guards','shared_acceptance','source_authored_caster_seed_policy')
    for method,plan in plans.items():
        row=proposals[method];selected=row['candidate_selected_by_original_native_cost']
        if (selected is None or selected['original_native_admitted'] is not True
                or not selected['reference_geometry']['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only']
                or any(plan[key]!=old_plans[method][key] for key in invariant)
                or plan['selected_candidate_id']!=selected['candidate_id']
                or plan['native_cost_s']!=selected['original_native_cost_s']
                or any(plan[key]!=row['selected_candidate_prediction'][key] for key in ('selected_actions','selected_states'))
                or plan['generator_selector_version']!=report['version']):
            raise ValueError('Only the exact verified own-native common-interface candidate may be bound')
        execution.validate_plan(plan)
    model=preparation.read_small(report['model_screen'])
    if (model['case_count']!=8 or model['nominal_software_MODEL_PASS_count']!=8
            or model['actual_drive_admission'] is not False or model['cases']!=nominal['cases']):
        raise ValueError('Exact prospective nominal diagnostic required, not actual validation')
    return report,model,plans


def bundle_for_protocol(protocol):
    path=protocol.get('terminal_connection_bundle')
    if path is None:return load_frozen_bundle()
    if protocol.get('frozen_bundle')!=path:
        raise ValueError('Prospective bundle identity must be explicit in its own protocol')
    return load_terminal_connection_bundle(path)


def resource_contract_for_protocol(protocol,plans):
    contract=common_resource_contract(300.,plans)
    if protocol.get('terminal_connection_bundle') is not None:
        contract.update(version='H4_ALL_FOUR_COMMON_300SIM_900WALL_TERMINAL_CONNECTION_V2',
            original_plan_control_gains_physics_safety_acceptance_unchanged=False,
            NEW_common_terminal_connection_plans_same_controls_physics_safety_acceptance=True)
    return contract


def route_resource_guards(protocol):
    """Bind the validated resource version without mutating shared controls."""
    guards=dict(protocol['guards']);expected=dict(execution.GUARDS)
    resource=protocol.get('common_resource_contract')
    if resource is not None:
        _,_,plans=bundle_for_protocol(protocol)
        if resource!=resource_contract_for_protocol(protocol,plans):
            raise ValueError('Exact all-four common resource definition required')
        expected['maximum_simulation_duration_s']=300.
    if guards!=expected:
        raise ValueError('Unreviewed physical guard/resource change; no route')
    return guards


def prepare_fresh_trial_protocol(name='cohort_04_full_task_r01',method='P_dynamics_full_envelope',
        *,previous_route=None,simulation_budget_s=None,initialize_visuals_before_static=False,
        terminal_connection_bundle=None):
    """Offline whole-task design only, independent fresh static still required."""
    initial_path=INITIAL_OUTPUT/(INITIAL_RUN_ID+'.json')
    if previous_route is None:
        raw=preparation.read_small(initial_path);preparation.verify(raw['input_sha256'])
        if (raw['verdict']!='PASS' or not raw['checks'] or any(v is not True for v in raw['checks'].values())
                or raw['trial_kind']!='H4_0p65m_engineering_initial_motion_and_measured_stop'
                or raw['use_for_final_statistics'] is not False
                or raw['trace_sha256']!=preparation.sha(initial_path.with_suffix('.csv'))
                or raw['serialization_closeout']['new_trial_count']!=0):
            raise ValueError('Completed own initial-motion evidence required, not a reservation/display or retry')
        with initial_path.with_suffix('.csv').open(newline='') as stream:
            initial_motion_trace_replay(raw,list(csv.DictReader(stream)))
        _,nominal,_=load_verified_current_static()
        if raw['observed_original_motor_settings']!=nominal['runtime_motor_baseline']:
            raise ValueError('Initial motion must retain the exact original nominal motor baseline')
        prior=dict(retired_robot_root=cohort.ROBOT,required_prior_result=str(initial_path),
            required_prior_tensor=raw['actual_post_initial_motion_tensor_snapshot'],
            retiring_actor_fixed_configuration=nominal['runtime_anonymous_source_reference']['fixed_configuration_rows'],
            expected_nominal_motor_baseline=nominal['runtime_motor_baseline'],
            required_prior_timeline=raw['actual_final_timeline'],
            protected_environment_sha256=raw['serialization_closeout']['whole_protected_environment_rechecked']['protected_environment_sha256'])
        prior_pins={**raw['input_sha256'],str(initial_path):preparation.sha(initial_path),
            str(initial_path.with_suffix('.csv')):preparation.sha(initial_path.with_suffix('.csv'))}
    else:
        boundary=completed_route_boundary(previous_route)
        prior={key:boundary[key] for key in ('retired_robot_root','required_prior_tensor',
            'retiring_actor_fixed_configuration','expected_nominal_motor_baseline',
            'required_prior_timeline','protected_environment_sha256')}
        prior['required_prior_result']=boundary['prior_result'];prior_pins=boundary['input_sha256']
    bundle,model,plans=(load_frozen_bundle() if terminal_connection_bundle is None else
        load_terminal_connection_bundle(terminal_connection_bundle))
    if method not in plans:raise ValueError('One exact frozen method required')
    static=preparation.read_small(STATIC_DESIGN)
    # Historical source archives, geometry/model definitions and current
    # small runtime archive are pinned; active source files may later evolve.
    pins={**prior_pins,**bundle['input_sha256'],**bundle['output_sha256'],
        **model['input_sha256'],**model['output_sha256']}
    target=FRESH_TRIAL_ROOT/name
    p=dict(version='H4_configured_fresh_single_whole_task_v1',cohort=name,method=method,
        new_robot_root='/World/patient_transport_h4_'+name,marker='/World/IJCAS_HospitalH4_'+name,
        **prior,output_directory=str(target),start_xyyaw=plans[method]['start_world_xyyaw'],
        goal_xyyaw=plans[method]['goal_world_xyyaw'],frozen_bundle=str(BUNDLE if terminal_connection_bundle is None else Path(terminal_connection_bundle).resolve()),
        environment_variant='four_exact_original_subfloor_collision_flags_disabled_Session_only',
        approved_disabled_subfloor_paths=list(preparation.SELECTED),
        command_limits=execution.COMMAND_LIMITS,guards=execution.GUARDS,acceptance=execution.ACCEPTANCE,
        terminal_controller_config=execution.CONTROLLER_CONFIG,
        trial_role='fresh_source_fixed_start_zero_command_static_NOT_path_performance',
        sampled_cooked_support_queries=static['sampled_cooked_support_queries'],
        additional_disabled_subfloor_probes=static['additional_disabled_subfloor_probes'],
        auxiliary_probe_known_nonself_colliders=static['auxiliary_probe_known_nonself_colliders'],
        support_top_m=static['support_top_m'],new_robot_root_in_support_protocol=True,
        automatic_Stop_count_limit=1,Stop_wall_budget_s=15.,Stop_update_budget=240,
        compact_readback_wall_budget_s=180.,compact_readback_heartbeat_s=15.,
        compact_readback_UI_yield_max_rows=16,compact_readback_UI_yield_wall_interval_s=.05,
        all_archived_environment_rows_required=4997,
        static_settle_s=1.,static_record_s=2.,source_authored_seed_no_angle_selection=True,
        same_original_fixed_start_goal_plan_gains_material_and_contact_guards=True,
        full_task_includes_ordinary_startup_not_continuation_from_initial_component=True,
        maximum_static_attempts=1,maximum_route_attempts=1,automatic_retry=False,
        no_CICS_Nav2_or_sensor_navigation=True,use_for_final_statistics=False,
        actual_motion_authorized=False,status='OFFLINE_DESIGN_EXPLICIT_WORKFLOW_CONSENT_AND_LIVE_PREFLIGHT_REQUIRED',
        input_sha256={**pins,str(target/'runtime_sources.zip'):hashlib.sha256(
            fresh_runtime_source_bundle(include_visual=simulation_budget_s is not None)).hexdigest()})
    if terminal_connection_bundle is not None:
        if previous_route is None or simulation_budget_s!=300. or not initialize_visuals_before_static:
            raise ValueError('Separate fresh 300s/early-display workflow required for a new planner version')
        p.update(terminal_connection_bundle=str(Path(terminal_connection_bundle).resolve()),
            separate_terminal_connection_candidate_consent_required=True,
            same_original_fixed_start_goal_plan_gains_material_and_contact_guards=False,
            same_original_fixed_start_goal_gains_material_contact_guards_but_NEW_planner_reference=True,
            original_planner_version_outcomes_NOT_pooled_or_relabelled=True)
        p['input_sha256'][p['terminal_connection_bundle']]=preparation.sha(p['terminal_connection_bundle'])
    if previous_route is not None:
        p['prior_completed_route_boundary']=boundary
    if simulation_budget_s is not None:
        p['common_resource_contract']=resource_contract_for_protocol(p,plans)
        p['guards']=dict(execution.GUARDS,maximum_simulation_duration_s=float(simulation_budget_s))
        p['restore_new_actor_visual_cache_after_static_only']=True
    if initialize_visuals_before_static:
        if simulation_budget_s is None:raise ValueError('Configured common-resource workflow required for early renderer initialization')
        p['initialize_new_actor_visuals_before_static']=True
        p['new_actor_unvalued_namespace_authored_in_Session_before_UI_frames']=True
        p['restore_new_actor_visual_cache_after_static_only']=False
    route_resource_guards(p)
    fresh_trial_context(p)  # Validate all paths before any exclusive write.
    return canonical(p)


def load_fresh_trial_protocol(path):
    path=Path(path);protocol=preparation.read_small(path);context=fresh_trial_context(protocol)
    if protocol.get('staged_boundary_continuation') is not None:
        return load_staged_workflow_continuation(path)
    if path!=context.protocol_path:raise ValueError('Exact own frozen fresh-trial protocol path required')
    preparation.verify(protocol['input_sha256'])
    previous=protocol.get('prior_completed_route_boundary',{}).get('prior_result')
    budget=protocol.get('common_resource_contract',{}).get('maximum_simulation_duration_s')
    if (protocol!=prepare_fresh_trial_protocol(protocol['cohort'],protocol['method'],
            previous_route=previous,simulation_budget_s=budget,
            initialize_visuals_before_static=protocol.get('initialize_new_actor_visuals_before_static') is True,
            terminal_connection_bundle=protocol.get('terminal_connection_bundle'))
            or context.source_archive.read_bytes()!=fresh_runtime_source_bundle(
                include_visual=(protocol.get('restore_new_actor_visual_cache_after_static_only') is True
                    or protocol.get('initialize_new_actor_visuals_before_static') is True))):
        raise ValueError('Fresh trial configuration or executed runtime source changed before launch')
    return context


def load_archived_fresh_trial_protocol(path):
    """Read an executed immutable definition, not permission to rerun it.

    Its original runtime ZIP remains pinned even after a reviewed software
    amendment. Only the new continuation loader can admit the first route.
    """
    path=Path(path);protocol=preparation.read_small(path);context=fresh_trial_context(protocol)
    if path!=context.protocol_path:
        raise ValueError('Exact own archived fresh-trial protocol required')
    preparation.verify(protocol['input_sha256'])
    expected={str(source.relative_to(ROOT)) for source in
        (Path(__file__),EXPERIMENT_ENTRY,ENTRY,INITIAL_GENERATOR)}
    if (protocol.get('restore_new_actor_visual_cache_after_static_only') is True
            or protocol.get('initialize_new_actor_visuals_before_static') is True):
        expected.update(('isaac/scripts/isaac_refresh_ijcas_hospital_h4_visual_instances_once.py',
            'studies/IJCAS/scripts/hospital_h4_visual_instance_refresh.py'))
    with zipfile.ZipFile(context.source_archive) as archive:
        if len(archive.namelist())!=len(expected) or set(archive.namelist())!=expected:
            raise ValueError('Original complete workflow source archive required')
    return context


def layer_field_differences(before,after):
    """Exact native field deltas on anonymous COPIES; no live layer edit."""
    first,last=map(preparation.old.fields.layer_snapshot,(before,after))
    missing={'field_is_not_authored':True};changes=[]
    for path in sorted(set(first)|set(last)):
        a,b=first.get(path,{}),last.get(path,{})
        if a.get('spec_type')!=b.get('spec_type'):
            changes.append(dict(path=path,field='__spec_type__',
                before=a.get('spec_type',missing),after=b.get('spec_type',missing)))
        ia,ib=a.get('info',{}),b.get('info',{})
        for key in sorted(set(ia)|set(ib)):
            if ia.get(key,missing)!=ib.get(key,missing):
                changes.append(dict(path=path,field=key,before=ia.get(key,missing),after=ib.get(key,missing)))
    return changes


async def review_fresh_prepath_hold(path):
    """GET-only diagnosis of an unstarted route; never retry, relax or repair.

    The old failure did not retain its changed layer fields. Observe the
    current PAUSED state and full protected environment without pretending
    later observations reconstruct those missing historical differences.
    """
    import time
    import omni.usd
    import omni.timeline
    import omni.kit.app
    from isaacsim.core.simulation_manager import SimulationManager
    context=load_archived_fresh_trial_protocol(path)
    result=context.output/'route'/(context.route_run_id+'.json')
    raw=preparation.read_small(result);preparation.verify(raw['input_sha256'])
    prep,static,replay=load_fresh_static_inputs(context)
    if (raw['verdict']!='HOLD' or raw['path_execution_started'] is not False
            or raw['automatic_route_Play_requested'] is not False or raw['sample_count']!=0
            or raw['actual_final_timeline']!=static['actual_final_timeline']
            or result.with_suffix('.csv').exists() or raw['static_gate_sha256']!=preparation.sha(context.stationary.RESULT)):
        raise ValueError('Only this own unstarted HOLD at the same completed static boundary may be reviewed')
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    expected=static['actual_final_timeline']
    if stage is None or expected['playing'] or expected['stopped'] or evidence.timeline_state(timeline)!=expected:
        raise ValueError('Exact retained own PAUSED static required; no recovery')
    view=SimulationManager.get_physics_sim_view()
    if view is None:raise ValueError('Existing static tensor view required; no Play to obtain it')
    tensor=view.create_articulation_view(context.cohort.ROBOT+'/base_link')
    def actual():return normalize_runtime_evidence(preparation.old.tensor_snapshot(tensor))
    before=actual()
    if before!=static['actual_post_static_tensor_snapshot']:
        raise ValueError('Own body/12 joints/13 links/gains/targets changed; no reset or review-based admission')
    source=fresh_source_for_live_actor(context,stage)
    output=context.output/'preflight_hold_review/live_review.json'
    if output.exists():
        old=preparation.read_small(output);preparation.verify(old['input_sha256'])
        if old['actual_tensor_before_and_after']!=before:raise ValueError('Recorded diagnostic state no longer retained')
        print('[H4 GET REVIEW] Existing read-only review retained; no overwrite, trial or retry:',output,flush=True)
        return old
    layers=lambda:dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    first=previous=layers();deltas=[];rows={};began=heartbeat=yielded=time.monotonic();frames=0
    async def frame():
        nonlocal previous,frames
        await omni.kit.app.get_app().next_update_async();frames+=1
        if usd.get_stage()!=stage or evidence.timeline_state(timeline)!=expected or actual()!=before:
            raise ValueError('PAUSED stage/physical state changed during GET-only review; no repair')
        current=layers()
        for name in current:
            if current[name]!=previous[name]:
                deltas.append(dict(frame=frames,layer=name,
                    changed_fields=layer_field_differences(previous[name],current[name])))
        previous=current
        if time.monotonic()-began>context.protocol['compact_readback_wall_budget_s']:
            raise TimeoutError('GET-only review exceeded existing wall budget; no retry')
    print('[H4 GET REVIEW] Own static body/12 joints/13 links retained; checking 4997 rows and exact layer deltas. NO Play.',flush=True)
    for name,row in preparation.compact.stage_rows(stage,preparation.SELECTED,preparation.old.original):
        if name in rows:raise ValueError('Duplicate protected environment row')
        rows[name]=row;now=time.monotonic()
        if now-heartbeat>=15.:
            print('[H4 GET REVIEW] protected rows=%d/4997; NO motion' % len(rows),flush=True);heartbeat=now
        if len(rows)%16==0 or now-yielded>=.05:
            await frame();yielded=time.monotonic()
    await frame()
    manifest=preparation.load_manifest(verify_inputs=False)
    fingerprint=preparation.compact.compare_rows(manifest['compact_environment_rows'],rows,preparation.SELECTED,after_flags=True)
    if (len(rows)!=4997 or fingerprint!=context.protocol['protected_environment_sha256']
            or actual()!=before or fresh_source_for_live_actor(context,stage)!=source
            or context.stationary.critical_environment(stage,prep)!=static['environment_after_static']):
        raise ValueError('Full protected environment/source/physical state not retained; no waiver')
    files=(result,context.protocol_path,context.source_archive,context.stationary.RESULT,context.stationary.TRACE)
    report=dict(status='RETAINED_PREPATH_HOLD_GET_ONLY_FIELD_REVIEW_NOT_DRIVE_ADMISSION',
        original_route_result=str(result),original_route_verdict='HOLD',original_HOLD_NOT_relabelled=True,
        own_static_raw_replay=replay,actual_tensor_before_and_after=before,actual_timeline_before_and_after=expected,
        all_4997_protected_environment_rows_verified=True,protected_environment_sha256=fingerprint,
        full_source_actor_configuration_unchanged=True,observed_UI_frames=frames,
        Root_Session_text_before=first,Root_Session_text_after=previous,observed_current_layer_deltas=deltas,
        original_failed_transaction_exact_field_deltas_NOT_recorded_and_NOT_invented=True,
        current_deltas_NOT_claimed_as_the_historical_isolated_cause=True,
        original_no_initialization_gain_restore_false_is_NOT_a_measured_gain_change=True,
        raw_static_and_HOLD_and_source_archives_preserved=True,
        new_trial_count=0,any_Play_Stop_Pause_reset_or_control_or_scene_edit=False,
        actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={str(p):preparation.sha(p) for p in files},
        analysis_source_sha256={str(Path(__file__)):preparation.sha(Path(__file__))})
    write_runtime_evidence(output,report)
    print('=== H4 PREPATH HOLD GET-ONLY REVIEW ===',flush=True)
    print('own static / body / 12 joints / 13 links / gains / all protected rows: VERIFIED',flush=True)
    print('current layer change transactions:',len(deltas),flush=True)
    for change in deltas:
        for item in change['changed_fields'][:12]:print(change['layer'],item['path'],item['field'],flush=True)
    print('saved:',output,flush=True)
    print('NO new static/route/Play/Stop/reset, waiver or motion admission. Original HOLD preserved.',flush=True)
    return report


async def fresh_environment_readback(context,stage,timeline,phase):
    """All original rows, with UI yields only while physics is not playing."""
    import time
    import omni.usd
    import omni.kit.app
    state=evidence.timeline_state(timeline)
    if state['playing']:raise RuntimeError('No full environment hashing during physics')
    root,session=stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()
    usd=omni.usd.get_context();began=heartbeat=yielded=time.monotonic();rows={}
    def unchanged():
        if (usd.get_stage()!=stage or evidence.timeline_state(timeline)!=state
                or stage.GetRootLayer().ExportToString()!=root or stage.GetSessionLayer().ExportToString()!=session):
            raise RuntimeError('Stage/layers/nonplaying state changed during fresh preflight')
        if time.monotonic()-began>context.protocol['compact_readback_wall_budget_s']:
            raise TimeoutError('Fresh environment readback budget exceeded; no automatic retry')
    for path,row in preparation.compact.stage_rows(stage,preparation.SELECTED,preparation.old.original):
        if path in rows:raise RuntimeError('Duplicate protected environment row')
        rows[path]=row;now=time.monotonic()
        if now-heartbeat>=15.:
            print('[H4 FRESH] %s rows=%d/4997; no physics' % (phase,len(rows)),flush=True);heartbeat=now
        if len(rows)%16==0 or now-yielded>=.05:
            unchanged();await omni.kit.app.get_app().next_update_async();unchanged();yielded=time.monotonic()
    unchanged();manifest=preparation.load_manifest(verify_inputs=False)
    fingerprint=preparation.compact.compare_rows(manifest['compact_environment_rows'],rows,preparation.SELECTED,after_flags=True)
    if len(rows)!=4997 or fingerprint!=context.protocol['protected_environment_sha256']:
        raise ValueError('The entire original environment must retain the same four-flag variant')
    return dict(row_count=len(rows),sha256=fingerprint)


def review_fresh_Stop_motion_fields(before,after,robot):
    """Permit only released motion state of the archived retiring actor.

    No scale, collider, gain, material, scene or other actor is exempted.
    Native fields and exact before/after values are retained, never repaired.
    """
    from pxr import Sdf
    first,last=map(preparation.old.fields.layer_snapshot,(before,after));changes=[]
    motion={'physics:velocity','physics:angularVelocity','xformOp:translate','xformOp:orient',
        'xformOp:transform','xformOpOrder','state:angular:physics:position','state:angular:physics:velocity'}
    for path in sorted(set(first)|set(last)):
        a,b=first.get(path),last.get(path)
        if a==b:continue
        parsed=Sdf.Path(path)
        if not parsed.GetPrimPath().HasPrefix(Sdf.Path(robot)):
            raise ValueError('Stop changed a field outside the archived retiring actor: '+path)
        if parsed.IsPropertyPath():
            if parsed.name not in motion:raise ValueError('Stop changed a fixed actor field: '+path)
            info_a,info_b=(a or {}).get('info',{}),(b or {}).get('info',{})
            changed={k for k in set(info_a)|set(info_b) if info_a.get(k)!=info_b.get(k)}
            allowed={'default'} if a and b else {'default','typeName','custom','variability'}
            if not changed<=allowed or any('timeSamples' in info for info in (info_a,info_b)):
                raise ValueError('Unknown Stop motion-state metadata: '+path)
        else:
            ia,ib=(a or {}).get('info',{}),(b or {}).get('info',{})
            if ({k for k in set(ia)|set(ib) if ia.get(k)!=ib.get(k)}-{'specifier'}):
                raise ValueError('Stop changed fixed prim metadata: '+path)
        changes.append(dict(path=path,before=a,after=b))
    return changes


async def prepare_fresh_live_trial(context):
    """Archive -> ONE confirmed Stop -> source-authored new actor; no Play."""
    import time
    import omni.usd
    import omni.timeline
    import omni.kit.app
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Usd,UsdGeom
    protocol=context.protocol;stage=omni.usd.get_context().get_stage();timeline=omni.timeline.get_timeline_interface()
    output=context.output;reservation=output/'workflow_reserved.json'
    archive=output/'actual_prior_state_before_Stop.json';boundary=output/'actual_Stop_boundary.json'
    if any(p.exists() for p in (reservation,archive,boundary,context.preparation.LIVE)):
        raise ValueError('This fresh workflow already reserved/began; preserve it, no replay or overwrite')
    if stage is None or evidence.timeline_state(timeline)!=protocol['required_prior_timeline']:
        raise ValueError('Exact completed prior H4 trial PAUSED boundary required before archiving')
    prior=preparation.read_small(Path(protocol['required_prior_result']))
    preparation.verify(prior['input_sha256'])
    view=SimulationManager.get_physics_sim_view()
    if view is None:raise ValueError('Existing completed prior H4 trial tensor required')
    actual=normalize_runtime_evidence(preparation.old.tensor_snapshot(view.create_articulation_view(protocol['retired_robot_root']+'/base_link')))
    if actual!=protocol['required_prior_tensor']:
        raise ValueError('Prior complete body/joints/links/targets/gains changed; no restoration')
    before=await fresh_environment_readback(context,stage,timeline,'before_archive')
    source=retiring_actor_configuration(context,stage)
    if source!=protocol['retiring_actor_fixed_configuration']:
        raise ValueError('Archived actor fixed configuration changed before Stop')
    if any(stage.GetPrimAtPath(p).IsValid() for p in (context.cohort.ROBOT,context.cohort.MARKER)):
        raise ValueError('New actor/marker already exists; never replace it')
    actual2=normalize_runtime_evidence(preparation.old.tensor_snapshot(view.create_articulation_view(protocol['retired_robot_root']+'/base_link')))
    if actual2!=actual:raise ValueError('Prior actor changed during full readback')
    root,session=stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()
    write_runtime_evidence(reservation,dict(cohort=protocol['cohort'],method=protocol['method'],
        explicit_workflow_consent_checked_by_entry=True,protocol_sha256=preparation.sha(context.protocol_path),
        no_Stop_or_Play_yet=True,no_automatic_retry=True))
    write_runtime_evidence(archive,dict(actual_state=actual,timeline=evidence.timeline_state(timeline),
        Root_text=root,Session_text=session,protected_environment=before,
        prior_result=protocol['required_prior_result'],prior_result_sha256=preparation.sha(protocol['required_prior_result']),
        solver_contact_state_NOT_serialized_or_restoreable=True,physical_continuity_ends_at_authorized_Stop=True))
    app=omni.kit.app.get_app();began=time.monotonic();updates=0
    print('[H4 FRESH] Completed prior H4 trial body/12 joints/13 links durably archived; requesting ONE Stop.',flush=True)
    # Keep the edit target stable throughout the deferred Stop transition.
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        timeline.stop()
        while not timeline.is_stopped():
            if updates>=protocol['Stop_update_budget'] or time.monotonic()-began>protocol['Stop_wall_budget_s']:
                raise TimeoutError('Authorized Stop not confirmed; no fresh staging or second Stop')
            await app.next_update_async();updates+=1
            if omni.usd.get_context().get_stage()!=stage:raise ValueError('Stage changed during authorized Stop')
        await app.next_update_async();updates+=1
        if evidence.timeline_state(timeline)!=dict(playing=False,stopped=True,current_time_s=0.):
            raise ValueError('Completed stable STOPPED zero-time boundary required')
        root_after,session_after=stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()
        deltas=dict(Root=review_fresh_Stop_motion_fields(root,root_after,protocol['retired_robot_root']),
            Session=review_fresh_Stop_motion_fields(session,session_after,protocol['retired_robot_root']))
        if retiring_actor_configuration(context,stage)!=source:
            raise ValueError('Stop altered fixed actor settings; no staging or repair')
        write_runtime_evidence(boundary,dict(actual_timeline=evidence.timeline_state(timeline),Stop_called_once=True,
            waited_updates=updates,exact_motion_field_differences=deltas,
            Root_text=root_after,Session_text=session_after,no_physical_state_restore=True))
        await fresh_environment_readback(context,stage,timeline,'completed_Stop')
        stage.GetPrimAtPath(protocol['retired_robot_root']).SetActive(False)
        made=preparation.old.constructor_for(context.cohort.ROBOT)(stage,protocol['start_xyyaw'])
        marker=UsdGeom.Xform.Define(stage,context.cohort.MARKER).GetPrim()
        marker.SetCustomDataByKey('robot_root',context.cohort.ROBOT)
        marker.SetCustomDataByKey('status','fresh_static_pending_NO_PLAY')
        marker.SetCustomDataByKey('cohort',protocol['cohort'])
        if protocol.get('initialize_new_actor_visuals_before_static') is True:
            from pxr import Sdf
            # Create this exact already-reviewed UNVALUED SDK declaration in
            # the new actor's Session, before property widgets get UI frames.
            # No default, drive, pose, seed, material or collider is assigned.
            namespace=stage.GetPrimAtPath(context.cohort.ROBOT).CreateAttribute(
                'isaac:namespace',Sdf.ValueTypeNames.String,custom=False,variability=Sdf.VariabilityVarying)
            namespace.SetDisplayName('Namespace')
            namespace.SetDocumentation('Namespace of the prim in Isaac Sim')
            await initialize_fresh_staged_display(context)
    protected=await fresh_environment_readback(context,stage,timeline,'new_source_staged')
    reference=fresh_source_for_live_actor(context,stage)
    prep=dict(cohort=protocol['cohort'],new_robot_root=context.cohort.ROBOT,
        retired_robot_root=protocol['retired_robot_root'],start_xyyaw=protocol['start_xyyaw'],
        simulation_mass_kg=265.,live_new_robot_readback=made,source_constructor_readback=made,
        protected_environment_sha256=protected['sha256'],source_reference=reference,
        expected_nominal_motor_baseline=protocol['expected_nominal_motor_baseline'],
        timeline=evidence.timeline_state(timeline),old_physical_continuity_NOT_restored=True,
        archive=str(archive),archive_sha256=preparation.sha(archive),Stop_boundary=str(boundary),
        Stop_boundary_sha256=preparation.sha(boundary),Play_or_static_or_drive_executed=False)
    prep['critical_environment']=context.stationary.critical_environment(stage,prep)
    prep.update(Root_sha256=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest(),
        Session_sha256=hashlib.sha256(stage.GetSessionLayer().ExportToString().encode()).hexdigest())
    write_runtime_evidence(context.preparation.LIVE,prep)
    print('[H4 FRESH] NEW fixed-start source actor staged; all original environment rows unchanged. NO Play yet.',flush=True)
    return prep


async def initialize_fresh_staged_display(context,*,recorded_preparation=None):
    """A stopped renderer-only initialization, also usable after preflight abort.

    Original preparation hashes are never replaced or adopted as a new static
    baseline. A late operation records exact Root differences and grants no
    static/route continuation. New workflows call it BEFORE that baseline.
    """
    import time
    import runpy
    import omni.usd
    import omni.timeline
    import omni.kit.app
    import carb.settings
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Usd
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    expected=dict(playing=False,stopped=True,current_time_s=0.)
    if stage is None or evidence.timeline_state(timeline)!=expected:
        raise ValueError('Exact stopped new-actor boundary required; no Play/Stop/reset')
    for folder in (context.output/'static',context.output/'route'):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('New actor already has a reserved/recorded physical attempt; no early display initialization')
    refresh=runpy.run_path(str(_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py')))['load'](
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_visual_instance_refresh.py'))
    source=fresh_source_for_live_actor(context,stage)
    validate=preparation.old.constructor_for(context.cohort.ROBOT).__globals__['validate_new_robot']
    asset=Usd.Stage.Open(str(preparation.ASSET))
    def physical():
        if usd.get_stage()!=stage:raise ValueError('Stage changed during renderer initialization')
        # The full 145-node source is compared before and AFTER restoration.
        # During the transaction ONLY its thirteen proved nonphysical visual
        # subtrees may have transient instanceable metadata. The visual helper
        # independently fingerprints every shape/material/default/sample.
        fixed=context.stationary.actor_configuration(stage)
        fixed={path:value for path,value in fixed.items() if 'visuals' not in path[len(context.cohort.ROBOT):].split('/')}
        return dict(nonvisual_fixed_configuration_rows=fixed,
            constructor=validate(stage,asset,context.protocol['start_xyyaw']))
    before=physical()
    if recorded_preparation is not None and before['constructor']!=recorded_preparation['source_constructor_readback']:
        raise ValueError('Recorded new actor pose/all twelve source joints/physical settings changed')
    folder=context.output/'display_initialization';output=folder/'renderer_initialization.json'
    reserve=folder/'single_renderer_initialization_reserved.json'
    if output.exists():
        saved=preparation.read_small(output)
        if saved['source_actor_before_and_after']!=before:raise ValueError('Recorded display source state no longer retained')
        print('[H4 DISPLAY INIT] Existing one-time receipt retained; no second refresh:',output,flush=True)
        return saved
    if reserve.exists():raise ValueError('Display initialization already attempted; no automatic retry')
    root_before=stage.GetRootLayer().ExportToString();session_before=stage.GetSessionLayer().ExportToString()
    diagnostic={}
    if recorded_preparation is not None:
        boundary=preparation.read_small(recorded_preparation['Stop_boundary'])
        if hashlib.sha256(boundary['Root_text'].encode()).hexdigest()!=recorded_preparation['Root_sha256']:
            raise ValueError('Original preparatory Root baseline cannot be reconstructed exactly')
        diagnostic=dict(original_preparation_Root_sha256=recorded_preparation['Root_sha256'],
            actual_Root_sha256=hashlib.sha256(root_before.encode()).hexdigest(),
            exact_Root_field_differences=layer_field_differences(boundary['Root_text'],root_before),
            Session_matches_original_preparation=hashlib.sha256(session_before.encode()).hexdigest()==recorded_preparation['Session_sha256'],
            missing_original_staged_Session_text_NOT_invented=True,
            original_preflight_abort_NOT_relabelled_or_admitted=True)
    protected=await fresh_environment_readback(context,stage,timeline,'before_renderer_initialization')
    if physical()!=before:raise ValueError('Source state changed during complete protected-environment review')
    write_runtime_evidence(reserve,dict(actor=context.cohort.ROBOT,cohort=context.protocol['cohort'],
        renderer_only=True,actual_static_or_drive_admission=False,no_timeline_command=True))
    try:
        result=await refresh['initialize_stopped_actor'](stage,context.cohort.ROBOT,
            state=lambda:evidence.timeline_state(timeline),next_frame=omni.kit.app.get_app().next_update_async,
            physical_readback=physical,viewport=get_active_viewport(),settings=carb.settings.get_settings())
        if (stage.GetRootLayer().ExportToString()!=root_before
                or stage.GetSessionLayer().ExportToString()!=session_before or physical()!=before
                or fresh_source_for_live_actor(context,stage)!=source):
            raise ValueError('Exact original stopped source/layer state not retained after renderer initialization')
        result.update(protected_environment=protected,original_preparation_diagnosis=diagnostic,
            input_sha256={str(context.protocol_path):preparation.sha(context.protocol_path)},
            use_for_final_statistics=False)
        write_runtime_evidence(output,result)
        print('=== H4 NEW-ACTOR DISPLAY INITIALIZATION BEFORE STATIC ===',flush=True)
        print('render roots registered / USD shapes:',result['registered_visual_root_count'],result['visible_USD_shape_count'],flush=True)
        print('source body / twelve joints / layers / stopped timeline retained: True',flush=True)
        if diagnostic:print('original Root field changes:',len(diagnostic['exact_Root_field_differences']),flush=True)
        print('saved:',output,flush=True)
        print('No Play/Stop/static/route. Inspect pixels without F; renderer registration is not drive admission.',flush=True)
        return result
    except BaseException as error:
        write_runtime_evidence(folder/'renderer_initialization_error.json',dict(
            error=type(error).__name__+': '+str(error),original_preparation_diagnosis=diagnostic,
            actual_static_or_drive_admission=False,automatic_retry=False))
        raise


async def initialize_recorded_staged_display(path):
    context=load_archived_fresh_trial_protocol(path)
    prep=preparation.read_small(context.preparation.LIVE)
    if (prep['cohort']!=context.protocol['cohort'] or prep['new_robot_root']!=context.cohort.ROBOT
            or prep['Play_or_static_or_drive_executed'] is not False):
        raise ValueError('Own completed staging receipt without a physical trial required')
    return await initialize_fresh_staged_display(context,recorded_preparation=prep)


def reconstruct_staged_layer_reference(path):
    """OFFLINE anonymous USD replay, accepted ONLY on the original two hashes.

    This does not infer the historical failed transaction. The exact recorded
    Stop layers plus the original deterministic staging operations reproduce
    a missing staged Session text; byte equality with its original receipt is
    mandatory. Never call this composition replay against a live Kit stage.
    """
    from pxr import Sdf,Usd,UsdGeom
    context=load_archived_fresh_trial_protocol(path)
    prep=preparation.read_small(context.preparation.LIVE)
    boundary=Path(prep['Stop_boundary'])
    if (prep['cohort']!=context.protocol['cohort'] or prep['new_robot_root']!=context.cohort.ROBOT
            or prep['Play_or_static_or_drive_executed'] is not False
            or prep['Stop_boundary_sha256']!=preparation.sha(boundary)
            or context.protocol.get('initialize_new_actor_visuals_before_static') is True):
        raise ValueError('Exact original pre-Play staging receipt and archived Stop required')
    stopped=preparation.read_small(boundary)
    if (stopped['actual_timeline']!=dict(playing=False,stopped=True,current_time_s=0.)
            or stopped['Stop_called_once'] is not True):
        raise ValueError('Own completed single Stop boundary required for offline reconstruction')
    def layer(text,label):
        suffix='.usda' if text.lstrip().startswith('#usda') else '.sdf'
        result=Sdf.Layer.CreateAnonymous(label+suffix)
        if not result.ImportFromString(text):raise ValueError('Cannot parse archived layer COPY')
        return result
    root=layer(stopped['Root_text'],'H4_OFFLINE_RECONSTRUCTION_ROOT')
    session=layer(stopped['Session_text'],'H4_OFFLINE_RECONSTRUCTION_SESSION')
    stage=Usd.Stage.Open(root,session)
    with Usd.EditContext(stage,session):
        stage.GetPrimAtPath(context.protocol['retired_robot_root']).SetActive(False)
        made=preparation.old.constructor_for(context.cohort.ROBOT)(stage,context.protocol['start_xyyaw'])
        marker=UsdGeom.Xform.Define(stage,context.cohort.MARKER).GetPrim()
        marker.SetCustomDataByKey('robot_root',context.cohort.ROBOT)
        marker.SetCustomDataByKey('status','fresh_static_pending_NO_PLAY')
        marker.SetCustomDataByKey('cohort',context.protocol['cohort'])
    texts=dict(Root=root.ExportToString(),Session=session.ExportToString())
    if (made!=prep['source_constructor_readback'] or any(
            hashlib.sha256(text.encode()).hexdigest()!=prep[name+'_sha256'] for name,text in texts.items())):
        raise ValueError('Offline reconstruction does not exactly match original staged hashes; no inferred baseline')
    files=(context.protocol_path,context.source_archive,context.preparation.LIVE,boundary)
    return dict(status='OFFLINE_RECONSTRUCTED_STAGED_LAYERS_BYTE_IDENTICAL_TO_ORIGINAL_RECEIPT',
        original_protocol=str(context.protocol_path),original_preparation=str(context.preparation.LIVE),
        original_staged_layer_text=texts,source_constructor_readback=made,
        historical_failed_transaction_NOT_reconstructed=True,actual_motion_authorized=False,
        input_sha256={str(p):preparation.sha(p) for p in files})


async def review_recorded_staged_boundary(path):
    """GET current stopped layers/source/environment; NO admission or resync.

    In particular, this does not adopt a new Session hash, classify unknown
    changes as harmless, or call the display initializer a second time.
    """
    import omni.usd
    import omni.timeline
    context=load_archived_fresh_trial_protocol(path)
    prep=preparation.read_small(context.preparation.LIVE)
    folder=context.output/'staged_boundary_review'
    reference_path=folder/'offline_original_layers.json';output=folder/'live_review.json'
    reference=preparation.read_small(reference_path);preparation.verify(reference['input_sha256'])
    if (reference['status']!='OFFLINE_RECONSTRUCTED_STAGED_LAYERS_BYTE_IDENTICAL_TO_ORIGINAL_RECEIPT'
            or reference['original_protocol']!=str(context.protocol_path)
            or reference['original_preparation']!=str(context.preparation.LIVE)
            or reference['source_constructor_readback']!=prep['source_constructor_readback']
            or prep['new_robot_root']!=context.cohort.ROBOT
            or prep['Play_or_static_or_drive_executed'] is not False):
        raise ValueError('Own immutable original preparation/reference required for GET-only review')
    for name,text in reference['original_staged_layer_text'].items():
        if name not in ('Root','Session') or hashlib.sha256(text.encode()).hexdigest()!=prep[name+'_sha256']:
            raise ValueError('Reference layer must exactly match original receipt, not a replacement hash')
    if set(reference['original_staged_layer_text'])!={'Root','Session'}:
        raise ValueError('Both original staged layer texts required')
    for subfolder in (context.output/'static',context.output/'route'):
        if subfolder.exists() and any(subfolder.iterdir()):
            raise ValueError('A physical attempt is already reserved/recorded; no staged-start review')
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    state=dict(playing=False,stopped=True,current_time_s=0.)
    if stage is None or evidence.timeline_state(timeline)!=state:
        raise ValueError('Own original STOPPED zero-time actor required; no timeline command')
    layers=lambda:dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    before=layers();source=fresh_source_for_live_actor(context,stage)
    # Validate the constructor without composing another Hospital stage.
    from pxr import Usd
    made=preparation.old.constructor_for(context.cohort.ROBOT).__globals__['validate_new_robot'](
        stage,Usd.Stage.Open(str(preparation.ASSET)),context.protocol['start_xyyaw'])
    if made!=prep['source_constructor_readback']:
        raise ValueError('Actual stopped source pose/twelve joints/gains/physical configuration changed')
    if output.exists():
        saved=preparation.read_small(output);preparation.verify(saved['input_sha256'])
        if saved['actual_layer_text']!=before or saved['actual_source_reference']!=source:
            raise ValueError('Previously reviewed boundary is no longer retained; no automatic rebaseline')
        print('[H4 STAGED REVIEW] Existing exact GET-only receipt retained:',output,flush=True)
        return saved
    print('[H4 STAGED REVIEW] Same stopped actor; exact Root/Session fields + all 4997 protected rows. GET ONLY.',flush=True)
    protected=None;error=None
    try:
        protected=await fresh_environment_readback(context,stage,timeline,'staged_boundary_GET_only')
        if (usd.get_stage()!=stage or evidence.timeline_state(timeline)!=state or layers()!=before
                or fresh_source_for_live_actor(context,stage)!=source):
            raise ValueError('Current stopped state/layers/source changed during GETs')
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    after=layers()
    changes={name:layer_field_differences(reference['original_staged_layer_text'][name],before[name])
        for name in ('Root','Session')}
    during={name:layer_field_differences(before[name],after[name]) for name in ('Root','Session')}
    files=(context.protocol_path,context.preparation.LIVE,reference_path,
        context.output/'workflow_error.json',context.output/'display_initialization/renderer_initialization.json')
    report=dict(status=('STOPPED_STAGED_BOUNDARY_GET_ONLY_REVIEW_COMPLETE_NOT_ADMISSION' if error is None
            else 'HOLD_STOPPED_STAGED_BOUNDARY_REVIEW_INCOMPLETE_NOT_ADMISSION'),
        original_protocol=str(context.protocol_path),new_robot_root=context.cohort.ROBOT,
        actual_timeline_before=state,actual_timeline_after=evidence.timeline_state(timeline),
        actual_layer_text=before,actual_layer_text_after=after,
        exact_original_to_current_field_differences=changes,exact_during_GET_field_differences=during,
        actual_source_reference=source,source_constructor_readback=made,protected_environment=protected,
        error=error,original_preparation_and_workflow_error_preserved=True,
        original_failed_transaction_exact_differences_NOT_invented=True,
        no_field_waived_or_baseline_replaced=True,any_stage_render_camera_or_timeline_edit=False,
        new_physical_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={**reference['input_sha256'],**{str(p):preparation.sha(p) for p in files}})
    write_runtime_evidence(output,report)
    print('=== H4 SAME STAGED ACTOR / EXACT NATIVE LAYER FIELD REVIEW ===',flush=True)
    print('Root / Session changed fields:',len(changes['Root']),len(changes['Session']),flush=True)
    for name,items in changes.items():
        for item in items:print('FIELD:',name,item['path'],item['field'],'before:',item['before'],'now:',item['after'],flush=True)
    print('protected environment / GET-time layer changes / error:',
        protected,sum(map(len,during.values())),error,flush=True)
    print('saved:',output,flush=True)
    print('NO Play/Stop/reset/refresh or static/route reservation. Differences require offline review.',flush=True)
    return report


def staged_view_only_review(context):
    """Validate the actual two aperture fields, not a general camera waiver."""
    folder=context.output/'staged_boundary_review'
    review=preparation.read_small(folder/'live_review.json')
    reference=preparation.read_small(folder/'offline_original_layers.json')
    prep=preparation.read_small(context.preparation.LIVE)
    for record in (review,reference):preparation.verify(record['input_sha256'])
    expected=[dict(path='/OmniverseKit_Top.'+name,field='default',before=old,after=new)
        for name,old,new in (('horizontalAperture',156.63197326660156,195.34104919433594),
            ('verticalAperture',130.38197326660156,169.09104919433594))]
    actual=review['actual_layer_text'];state=dict(playing=False,stopped=True,current_time_s=0.)
    changes={name:layer_field_differences(reference['original_staged_layer_text'][name],actual[name])
        for name in ('Root','Session')}
    if (review['status']!='STOPPED_STAGED_BOUNDARY_GET_ONLY_REVIEW_COMPLETE_NOT_ADMISSION'
            or review['original_protocol']!=str(context.protocol_path)
            or review['new_robot_root']!=context.cohort.ROBOT
            or review['actual_timeline_before']!=state or review['actual_timeline_after']!=state
            or review['actual_layer_text_after']!=actual or review['error'] is not None
            or changes!={'Root':[],'Session':expected}
            or review['exact_original_to_current_field_differences']!=changes
            or review['exact_during_GET_field_differences']!={'Root':[],'Session':[]}
            or review['protected_environment']!=dict(row_count=4997,sha256=context.protocol['protected_environment_sha256'])
            or review['actual_source_reference']!=prep['source_reference']
            or review['source_constructor_readback']!=prep['source_constructor_readback']
            or review['new_physical_trial_count']!=0 or review['any_stage_render_camera_or_timeline_edit'] is not False
            or review['no_field_waived_or_baseline_replaced'] is not True
            or any(hashlib.sha256(text.encode()).hexdigest()!=prep[name+'_sha256']
                for name,text in reference['original_staged_layer_text'].items())):
        raise ValueError('Only the exact GET-reviewed two aperture changes may define this pre-static continuation')
    display_path=context.output/'display_initialization/renderer_initialization.json'
    display=preparation.read_small(display_path)
    if (display['registered_visual_root_count']!=13 or display['visible_USD_shape_count']!=22
            or display['original_Root_and_Session_text_exactly_restored'] is not True
            or display['new_physical_trial_count']!=0
            or display['source_actor_before_and_after']['constructor']!=prep['source_constructor_readback']):
        raise ValueError('Own completed pre-static renderer registration required; do not repeat it')
    return prep,review,expected


def verify_staged_workflow_sources(context,new_archive):
    """Unchanged static, route control and physics; admission/display only."""
    import ast
    allowed={'prepare_fresh_trial_protocol','load_fresh_trial_protocol','load_archived_fresh_trial_protocol',
        'prepare_fresh_live_trial','completed_route_boundary'}
    checked={}
    with zipfile.ZipFile(context.source_archive) as old,zipfile.ZipFile(io.BytesIO(new_archive)) as new:
        if set(old.namelist())!=set(new.namelist()) or len(old.namelist())!=len(new.namelist()):
            raise ValueError('Complete original six-source membership required in the new archive')
        for path in (Path(__file__),ENTRY):
            key=str(path.relative_to(ROOT));a,b=(ast.parse(z.read(key)) for z in (old,new))
            functions=lambda tree:{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            first,last=functions(a),functions(b)
            for name,node in first.items():
                if path==Path(__file__) and name in allowed:continue
                if name not in last or ast.dump(node)!=ast.dump(last[name]):
                    raise ValueError('Unreviewed original physical/runtime function change: '+name)
            if path==Path(__file__):
                constants=lambda tree:[ast.dump(n) for n in tree.body
                    if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                if constants(a)!=constants(b):raise ValueError('Original physical module configuration changed')
            elif old.read(key)!=new.read(key):
                raise ValueError('The entire actual route entry must be byte-identical, not just its controller functions')
            checked[path.name]=sorted(set(first)-(allowed if path==Path(__file__) else set()))
    return dict(unchanged_existing_physical_functions_AST=checked,
        entire_route_entry_byte_identical=True,static_function_bytecode_source_unchanged=True,
        no_physical_tolerance_model_controller_gain_material_or_guard_change=True)


def staged_continuation_definition(context,source_bytes):
    """Deterministic prospective definition; no files or live state modified."""
    prep,review,changes=staged_view_only_review(context)
    folder=context.output/'staged_static_continuation'
    original_protocol=context.protocol_path;original_preparation=context.preparation.LIVE
    baseline=dict(prep,Session_sha256=hashlib.sha256(review['actual_layer_text']['Session'].encode()).hexdigest(),
        pre_static_view_only_amendment=dict(original_preparation=str(original_preparation),
            original_preparation_sha256=preparation.sha(original_preparation),
            reviewed_actual_stopped_state=str(context.output/'staged_boundary_review/live_review.json'),
            exact_changed_fields=changes,source_pose_joints_and_environment_unchanged=True,
            no_restaging_or_new_Stop_or_physical_trial=True))
    if baseline['Root_sha256']!=hashlib.sha256(review['actual_layer_text']['Root'].encode()).hexdigest():
        raise ValueError('Original Root must remain exact; no Root rebaseline')
    source_review=verify_staged_workflow_sources(context,source_bytes)
    files=(original_protocol,context.source_archive,original_preparation,
        context.output/'workflow_reserved.json',context.output/'workflow_error.json',
        context.output/'staged_boundary_review/offline_original_layers.json',
        context.output/'staged_boundary_review/live_review.json',
        context.output/'display_initialization/renderer_initialization.json')
    pins={**context.protocol['input_sha256'],**{str(p):preparation.sha(p) for p in files},
        str(folder/'preparation.json'):hashlib.sha256((json.dumps(normalize_runtime_evidence(baseline),indent=2,allow_nan=False)+'\n').encode()).hexdigest(),
        str(folder/'runtime_sources.zip'):hashlib.sha256(source_bytes).hexdigest()}
    protocol=dict(context.protocol,input_sha256=pins,restore_new_actor_visual_cache_after_static_only=False,
        staged_boundary_continuation=dict(version='H4_EXACT_TWO_APERTURE_PRE_STATIC_CONTINUATION_V1',
            original_protocol=str(original_protocol),original_preparation=str(original_preparation),
            amended_preparation=str(folder/'preparation.json'),protocol=str(folder/'protocol.json'),
            executed_source_archive=str(folder/'runtime_sources.zip'),
            source_amendment_review=source_review,exact_reviewed_nonphysical_fields=changes,
            additional_Stop_commands=0,additional_restaging=0,additional_display_refresh=0,
            remaining_original_static_attempts=1,remaining_original_conditional_route_attempts=1,
            original_workflow_error_and_preparation_preserved=True,automatic_retry=False))
    return normalize_runtime_evidence(protocol),normalize_runtime_evidence(baseline)


def prepare_staged_workflow_continuation(path):
    """Prepare ONLY the unused remainder of the already approved B2 attempt."""
    context=load_archived_fresh_trial_protocol(path)
    failure=preparation.read_small(context.output/'workflow_error.json')
    if (failure['status']!='WORKFLOW_STOPPED_NO_RETRY'
            or failure['error']!='RuntimeError: Paused state/layers changed during full environment readback'):
        raise ValueError('Exact unused pre-static workflow abort required, not a failed physical trial')
    for folder in (context.output/'static',context.output/'route'):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Original static/route already reserved or recorded; cannot continue or retry')
    source=fresh_runtime_source_bundle(include_visual=True)
    protocol,baseline=staged_continuation_definition(context,source)
    return protocol,baseline,source


def load_staged_workflow_continuation(path,*,active_source_required=True):
    path=Path(path);protocol=preparation.read_small(path)
    amendment=protocol.get('staged_boundary_continuation',{})
    context=load_archived_fresh_trial_protocol(amendment['original_protocol'])
    expected=context.output/'staged_static_continuation/protocol.json'
    if path!=expected:raise ValueError('Exact own staged continuation protocol required')
    preparation.verify(protocol['input_sha256'])
    archive=context.output/'staged_static_continuation/runtime_sources.zip';source=archive.read_bytes()
    definition,baseline=staged_continuation_definition(context,source)
    if (protocol!=definition or preparation.read_small(amendment['amended_preparation'])!=baseline
            or active_source_required and source!=fresh_runtime_source_bundle(include_visual=True)):
        raise ValueError('Reviewed staged remainder/source archive/baseline changed; no trial')
    context.protocol=protocol;context.protocol_path=path;context.source_archive=archive
    context.preparation.LIVE=Path(amendment['amended_preparation'])
    context.cohort.LIVE=context.preparation.LIVE
    context.stationary.PROTOCOL=path
    return context


async def run_staged_workflow_continuation(path):
    """No Stop/restage/resync: original static once, then its conditional B2."""
    import runpy
    import builtins
    context=load_staged_workflow_continuation(path)
    folder=context.protocol_path.parent;reservation=folder/'workflow_reserved.json'
    for directory in (context.output/'static',context.output/'route'):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Original static/route already reserved/recorded; no automatic retry')
    if reservation.exists():raise ValueError('This remainder already dispatched; no second attempt')
    write_runtime_evidence(reservation,dict(protocol_sha256=preparation.sha(context.protocol_path),
        remaining_original_static_attempts=1,remaining_original_conditional_route_attempts=1,
        no_Stop_restaging_renderer_refresh_or_retry=True))
    try:
        static=_bound(run_retained_static,preparation=context.preparation,cohort=context.cohort,
            STATIC_OUTPUT=context.stationary.RESULT.parent,STATIC_RUN_ID=context.static_run_id,
            STATIC_DESIGN=context.protocol_path,evaluate_support_queries_for_actor=context.support_evaluator)
        result=await static(fresh=context)
        if result['verdict']!='PASS':
            print('[H4 REMAINDER] Static did not PASS; B2 NOT started. No retry.',flush=True)
            return result
        load_fresh_static_inputs(context)
        dispatched=runpy.run_path(str(ENTRY),init_globals={
            'ijcas_hospital_h4_subfloor_compact_path_authorized':True,
            'ijcas_hospital_h4_fresh_trial_protocol':str(context.protocol_path),
            'ijcas_hospital_h4_method':context.protocol['method']},run_name='__main__')
        task=dispatched['ijcas_hospital_h4_path_task'];builtins.ijcas_h4_latest_path_task=task
        await task
        return preparation.read_small(context.output/'route'/(context.route_run_id+'.json'))
    except BaseException as error:
        write_runtime_evidence(folder/'workflow_error.json',dict(status='REMAINDER_STOPPED_NO_RETRY',
            error=type(error).__name__+': '+str(error),original_failure_preserved=True))
        raise


def retiring_actor_configuration(context,stage):
    """GET the actual retiring actor, not the old cohort-03 module constant."""
    return _bound(stationary.actor_configuration,ROBOT=context.protocol['retired_robot_root'])(stage)


def completed_route_boundary(path):
    """Read an archivable completed route boundary, whether PASS or FAIL.

    This is not a new experiment definition, permission to Stop, a retry or
    relabelling of the route. It supplies exact saved state for a separately
    approved later fresh-cohort workflow. The live state must still be checked.
    """
    path=Path(path).resolve();raw=preparation.read_small(path)
    cohort_path=FRESH_TRIAL_ROOT/raw['execution_cohort']/'protocol.json'
    context=load_archived_fresh_trial_protocol(cohort_path)
    if Path(raw.get('path_protocol',''))==context.output/'staged_static_continuation/protocol.json':
        context=load_staged_workflow_continuation(raw['path_protocol'],active_source_required=False)
    if Path(raw.get('path_protocol','')).parent==context.output/'prepath_hold_continuation':
        amendment=preparation.read_small(raw['path_protocol']);preparation.verify(amendment['input_sha256'])
        expected=prepath_motion_identity(context)
        if (amendment['version']!='H4_RETAINED_STATIC_FIRST_MOTION_AFTER_PREPATH_HOLD_V1'
                or amendment['original_protocol']!=str(context.protocol_path)
                or amendment['route_output']!=str(expected[0]) or amendment['route_run_id']!=expected[1]
                or amendment['new_robot_root']!=context.cohort.ROBOT):
            raise ValueError('Exact own archived prepath continuation identity required')
        context.route_output,context.route_run_id=expected
    route_output=getattr(context,'route_output',context.output/'route')
    if path!=route_output/(context.route_run_id+'.json'):
        raise ValueError('Exact own completed fresh route path required, not a selected unrelated result')
    preparation.verify(raw['input_sha256'])
    infrastructure=('actual_route_Pause_confirmed','original_motor_gains_restored',
        'physical_environment_unchanged','all_frozen_input_evidence_unchanged')
    if (raw['verdict'] not in ('PASS','FAIL') or raw['path_execution_started'] is not True
            or raw['automatic_route_Play_requested'] is not True
            or raw['execution_role']!='fresh_fixed_start_whole_H4_development_pilot'
            or raw['original_motor_gains_restored'] is not True
            or any(raw['checks'].get(key) is not True for key in infrastructure)
            or raw['actual_final_timeline']!=raw['pause_confirmation']['after']
            or raw['actual_final_timeline']['playing'] or raw['actual_final_timeline']['stopped']
            or raw['trace_sha256']!=preparation.sha(path.with_suffix('.csv'))
            or raw['use_for_final_statistics'] is not False):
        raise ValueError('Own finished PAUSED route with restored gains/environment required before a later fresh cohort')
    prep,static,replay=load_fresh_static_inputs(context)
    if (raw['static_gate_reference']!=str(context.stationary.RESULT)
            or raw['static_gate_sha256']!=preparation.sha(context.stationary.RESULT)
            or raw['physics_preparation_snapshot']!=prep
            or raw['compact_environment_after_route_Pause']['all_original_row_count']!=4997
            or raw['compact_environment_after_route_Pause']['protected_environment_sha256']!=context.protocol['protected_environment_sha256']):
        raise ValueError('Completed route must retain its own actual static and whole protected environment')
    actual=raw['actual_post_route_tensor_snapshot']
    for key,shape in (('transform',(7,)),('velocity',(6,)),('positions',(12,)),
            ('joint_velocities_rad_s',(12,)),('joint_position_targets_rad',(12,)),('joint_velocity_targets_rad_s',(12,)),
            ('all_thirteen_link_world_transforms_xyzw',(13,7)),('all_thirteen_link_velocities_at_CoM',(13,6))):
        values=np.asarray(actual[key],float)
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError('Complete finite saved post-route state required: '+key)
    baseline=raw['observed_original_motor_settings']
    if (baseline!=static['runtime_motor_baseline'] or actual['names']!=baseline['joint_names']
            or actual['gains']!={key:baseline[key] for key in ('kp','kd','max_efforts')}
            or len(actual['link_names'])!=13 or len(set(actual['link_names']))!=13
            or actual['positions']!=raw['final_actual_twelve_joint_positions_rad']
            or actual['joint_velocities_rad_s']!=raw['final_actual_twelve_joint_velocities_rad_s']):
        raise ValueError('Own post-route joint/gain/link correspondence differs from the saved closeout')
    with path.with_suffix('.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    times=np.asarray([float(row['time_s']) for row in rows])
    if (len(rows)!=raw['sample_count'] or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0.) or not any(row['phase']=='path_tracking' for row in rows)):
        raise ValueError('Own actual monotonic route trace required, not a setup/reservation-only receipt')
    pins={**raw['input_sha256'],str(path):preparation.sha(path),
        str(path.with_suffix('.csv')):preparation.sha(path.with_suffix('.csv')),
        str(context.protocol_path):preparation.sha(context.protocol_path),
        str(context.stationary.RESULT):preparation.sha(context.stationary.RESULT)}
    return canonical(dict(status='SAVED_COMPLETED_ROUTE_BOUNDARY_NOT_LIVE_OR_NEW_TRIAL_ADMISSION',
        prior_result=str(path),prior_verdict=raw['verdict'],retired_robot_root=context.cohort.ROBOT,
        required_prior_tensor=actual,required_prior_timeline=raw['actual_final_timeline'],
        retiring_actor_fixed_configuration=static['runtime_anonymous_source_reference']['fixed_configuration_rows'],
        expected_nominal_motor_baseline=baseline,
        protected_environment_sha256=context.protocol['protected_environment_sha256'],
        prior_result_NOT_relabelled=True,live_state_NOT_accessed_or_restored=True,
        no_new_experiment_or_Stop_authority=True,actual_motion_authorized=False,input_sha256=pins))


def load_fresh_static_inputs(context):
    """Replay this NEW static's own measurements, never a translated old PASS."""
    s=context.stationary;raw=preparation.read_small(s.RESULT);prep=preparation.read_small(context.preparation.LIVE)
    protocol=context.protocol
    reservation=preparation.read_small(s.RESULT.parent/'single_attempt_reserved.json')
    if (raw['run_id']!=context.static_run_id or raw['cohort']!=protocol['cohort']
            or raw['trial_kind']!=protocol['trial_role'] or raw['verdict']!='PASS'
            or not raw['checks'] or any(v is not True for v in raw['checks'].values())
            or raw['trace_sha256']!=preparation.sha(s.TRACE)
            or raw['static_protocol_sha256']!=preparation.sha(context.protocol_path)
            or reservation['run_id']!=context.static_run_id
            or reservation['explicit_user_single_static_consent'] is not True
            or reservation['protocol_sha256']!=preparation.sha(context.protocol_path)
            or (raw['settle_s'],raw['sampling_s'],raw['simulation_mass_kg'])!=(1.,2.,265.)
            or raw['source_authored_seed_verified_before_Play'] is not True
            or raw['physics_preparation_snapshot']!=prep
            or raw['use_for_final_statistics'] is not False
            or raw.get('error') or raw.get('postprocessing_error') or raw.get('pause_error')):
        raise ValueError('Exact own fresh actual static PASS required before a whole route')
    preparation.verify(raw['input_sha256'])
    for key in ('compact_environment_before_Play','compact_environment_after_Pause'):
        if raw[key]!=dict(row_count=4997,sha256=protocol['protected_environment_sha256']):
            raise ValueError('Both complete environment checks required for the own fresh static')
    with s.TRACE.open(newline='') as stream:rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    physical=completed_static.physical.evaluate_static(rows,protocol['start_xyyaw'],265.,
        raw['contact_report_counts_by_category'],raw['contact_callback_errors'])
    support=context.support_evaluator(protocol,raw['actual_sampled_cooked_support_queries'])
    probes=s.evaluate_subfloor_queries(protocol,raw['actual_sampled_cooked_support_queries'],raw['actual_disabled_subfloor_probes'])
    if (len(rows)!=raw['sample_count'] or physical['verdict']!='PASS'
            or not all(support['checks'].values()) or not all(probes['checks'].values())
            or raw['runtime_motor_baseline']!=protocol['expected_nominal_motor_baseline']
            or raw['actual_final_timeline']!=raw['pause_confirmation']['after']
            or raw['actual_final_timeline']['playing'] or raw['actual_final_timeline']['stopped']):
        raise ValueError('Fresh own physical CSV/query/gain/Pause replay failed')
    clocks=review_fresh_static_clocks(prep,raw,rows)
    counts={}
    for row in raw['contact_pairs_by_category']:
        if (type(row['callback_count']) is not int or row['callback_count']<=0
                or s.classify_contact(row['pair'],prep['critical_environment']['disabled_legacy_colliders'])!=row['category']):
            raise ValueError('Fresh contact identity or count invalid')
        counts[row['category']]=counts.get(row['category'],0)+row['callback_count']
    if counts!=raw['contact_report_counts_by_category']:
        raise ValueError('Fresh contact totals disagree')
    actual=raw['actual_post_static_tensor_snapshot'];qx,qy,qz,qw=actual['transform'][3:]
    for key,shape in (('transform',(7,)),('velocity',(6,)),('positions',(12,)),
            ('joint_velocities_rad_s',(12,)),('joint_position_targets_rad',(12,)),('joint_velocity_targets_rad_s',(12,)),
            ('all_thirteen_link_world_transforms_xyzw',(13,7)),('all_thirteen_link_velocities_at_CoM',(13,6))):
        values=np.asarray(actual[key],float)
        if values.shape!=shape or not np.isfinite(values).all():
            raise ValueError('Own fresh finite complete tensor required: '+key)
    if (set(actual['names'])!=set(protocol['expected_nominal_motor_baseline']['joint_names'])
            or len(actual['names'])!=12 or len(set(actual['link_names']))!=13):
        raise ValueError('Own fresh complete twelve-joint/thirteen-link correspondence required')
    yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    normalized=dict(raw,final_actual_world_xyyaw=[*actual['transform'][:2],yaw],
        final_actual_joint_names=actual['names'],final_actual_twelve_joint_positions_rad=actual['positions'],
        final_actual_twelve_joint_velocities_rad_s=actual['joint_velocities_rad_s'],
        passive_post_static_link_readback=actual)
    preparation.old.validate_actual_state(normalized,actual)
    return prep,normalized,dict(verdict='PASS',physical=physical,support=support,probes=probes,
        separate_clock_review=clocks)


def review_fresh_static_clocks(prep,raw,rows):
    """Do not equate Kit timeline, callback total and post-init CSV clocks.

    The monitor is attached before Play; CSV time starts after initialization;
    callback accounting continues through confirmed Pause. Freshness comes
    from the own hashed STOPPED-zero preparation, identity/reservation/source
    checks, not equality of two independently recorded runtime clocks. This
    does not alter any physical acceptance or replayed measurement.
    """
    if prep['timeline']!=dict(playing=False,stopped=True,current_time_s=0.):
        raise ValueError('Own fresh static preparation must be STOPPED at zero; not a retained trial')
    duration=raw['settle_s']+raw['sampling_s']
    timeline=raw['actual_final_timeline']['current_time_s']
    callbacks=raw['actual_physics_callback_duration_s']
    times=np.asarray([row['time_s'] for row in rows],float)
    pause=raw['pause_confirmation']
    if (not all(type(value) in (float,int) and math.isfinite(value) and value>=duration
                for value in (timeline,callbacks))
            or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0.)
            or times[0]<raw['settle_s'] or times[-1]<duration or times[-1]>callbacks
            or pause['before']['playing'] is not True or pause['before']['stopped'] is not False
            or pause['pause_requested_by_this_call'] is not True
            or type(pause['updates_to_observe_stable_pause']) is not int
            or pause['updates_to_observe_stable_pause']<1
            or pause['any_Play_Stop_reset_or_trial'] is not False
            or not math.isfinite(pause['before']['current_time_s'])
            or pause['before']['current_time_s']>timeline):
        raise ValueError('Own bounded monotonic fresh observation and confirmed Pause clocks required')
    return dict(status='OWN_ZERO_START_OBSERVATION_AND_PAUSE_VERIFIED_CLOCKS_NOT_EQUATED',
        own_preparation_timeline=prep['timeline'],final_timeline_s=timeline,
        callback_total_s=callbacks,callback_minus_timeline_s=callbacks-timeline,
        CSV_first_last_s=[float(times[0]),float(times[-1])],
        difference_diagnostic_only_NOT_an_isolated_callback_cause=True,
        physical_acceptance_tolerances_and_raw_verdict_unchanged=True)


def load_fresh_route_inputs(path,method,*,continuation_path=None):
    context=(load_fresh_trial_protocol(path) if continuation_path is None else
        load_first_route_continuation(continuation_path))
    if Path(path)!=context.protocol_path:
        raise ValueError('Route continuation must refer to its own original fresh protocol')
    if method!=context.protocol['method']:raise ValueError('Only the separately consented frozen method may run')
    bundle,model,plans=bundle_for_protocol(context.protocol);prep,static,replay=load_fresh_static_inputs(context)
    canonical_root='/World/patient_transport_h4_nominal_recipe'
    canonical_context=SimpleNamespace(protocol=context.protocol,cohort=SimpleNamespace(ROBOT=canonical_root),
        stationary=SimpleNamespace(actor_configuration=_bound(stationary.actor_configuration,ROBOT=canonical_root)))
    source=fresh_source_reference(canonical_context,namespace_present=True)
    return dict(protocol=context.protocol,bundle=bundle,model=model,plan=plans[method],preparation=prep,
        static=static,static_replay=replay,context=context,
        canonical_actor_configuration=source['fixed_configuration_rows'])


def verify_first_route_source_amendment(context):
    """Prove unchanged physical functions; only admission/metadata may differ."""
    import ast
    allowed={'load_fresh_static_inputs','load_fresh_route_inputs'}
    checked={}
    with zipfile.ZipFile(context.source_archive) as archive:
        for path in (Path(__file__),ENTRY):
            old=ast.parse(archive.read(str(path.relative_to(ROOT))).decode())
            new=ast.parse(path.read_text())
            old_functions={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
            new_functions={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
            for name,node in old_functions.items():
                if path==Path(__file__) and name in allowed:continue
                if name not in new_functions or ast.dump(node)!=ast.dump(new_functions[name]):
                    raise ValueError('Original physical/runtime function changed: '+path.name+':'+name)
            # No existing module-level constants/imports may be retuned in
            # the helper. New functions supply the separate admission path.
            if path==Path(__file__):
                nonfunctions=lambda tree:[ast.dump(n) for n in tree.body
                    if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                if nonfunctions(old)!=nonfunctions(new):
                    raise ValueError('Original module-level physical configuration changed')
            checked[path.name]=sorted(set(old_functions)-allowed if path==Path(__file__) else old_functions)
    return dict(original_physical_and_static_functions_AST_exact=checked,
        changed_existing_helper_functions=sorted(allowed),
        original_shared_v3_controller_and_all_safety_limits_pinned_by_original_inputs=True)


def prepath_motion_identity(context):
    """A separate output for the explicitly approved first actual motion."""
    return (context.output/'route'/'first_motion_after_prepath_hold',
        context.route_run_id.removesuffix('_r01')+'_after_prepath_hold_r01')


def verify_prepath_instrumentation_amendment(context):
    """Prove the original controller/guard body is unchanged, not just assert it."""
    import ast,copy
    def functions(tree):
        return {node.name:node for node in ast.walk(tree)
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    with zipfile.ZipFile(context.source_archive) as archive:
        old=ast.parse(archive.read(str(ENTRY.relative_to(ROOT))).decode())
        new=ast.parse(ENTRY.read_text());a,b=functions(old),functions(new)
        exact=('_h4_path_contacts','_h4_path_step','_h4_path_angles','numeric','live_static_check',
            'measure','verify_motor_settings','set_hold','command','guard','sample')
        for name in exact:
            if ast.dump(a[name])!=ast.dump(b[name]):raise ValueError('Original physical function changed: '+name)
        old_run,new_run=a['_run_h4_path'],b['_run_h4_path']
        old_try=next(n for n in old_run.body if isinstance(n,ast.Try))
        new_try=next(n for n in new_run.body if isinstance(n,ast.Try))
        if ast.dump(ast.Module(body=old_try.body,type_ignores=[]))!=ast.dump(ast.Module(body=new_try.body,type_ignores=[])):
            raise ValueError('Original tracking/terminal/fixed-hold operative body changed')
        mutating={'play','pause','stop','reset','apply_action','set_gains','confirm_pause','SetCustomDataByKey'}
        actions=lambda node:[ast.dump(n) for n in ast.walk(node) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr in mutating]
        if actions(old_run)!=actions(new_run):raise ValueError('Original physical/timeline/cleanup commands changed')
        compact=copy.deepcopy(b['compact_environment_readback'])
        old_nested=functions(a['compact_environment_readback'])['unchanged']
        new_nested=functions(compact)['unchanged']
        if ast.dump(old_nested.body[0].test)!=ast.dump(new_nested.body[0].test):
            raise ValueError('Original strict stage/timeline/whole-layer rejection predicate changed')
        if actions(new_nested.body[0]):raise ValueError('Boundary diagnostics may not command or repair the scene')
        new_nested.body[0].body=copy.deepcopy(old_nested.body[0].body)
        if ast.dump(compact)!=ast.dump(a['compact_environment_readback']):
            raise ValueError('Original full environment scan or guard changed beyond failure diagnostics')
        old_helper=ast.parse(archive.read(str(Path(__file__).relative_to(ROOT))).decode())
        new_helper=ast.parse(Path(__file__).read_text())
        allowed={'load_fresh_route_inputs','prepare_first_route_continuation','load_first_route_continuation','completed_route_boundary'}
        ah,bh=functions(old_helper),functions(new_helper)
        for name in (n.name for n in old_helper.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))):
            if name not in allowed and ast.dump(ah[name])!=ast.dump(bh[name]):
                raise ValueError('Original physical/static/resource helper changed: '+name)
    return dict(original_physical_functions_AST_exact=list(exact),
        original_tracking_terminal_and_single_fixed_hold_body_AST_exact=True,
        original_strict_stage_timeline_layer_predicate_and_full_environment_scan_preserved=True,
        original_physical_timeline_and_gain_cleanup_commands_AST_exact=True,
        changed_scope='admission/output binding, exact rejected-field evidence, uninitialized GET-only closeout',
        runtime_safety_or_physics_or_control_or_resource_limits_relaxed=False)


def prepare_prepath_hold_continuation(context):
    """New explicitly approved dispatch; original no-Play HOLD remains immutable."""
    result=context.output/'route'/(context.route_run_id+'.json')
    review_path=context.output/'preflight_hold_review/live_review.json'
    raw=preparation.read_small(result);review=preparation.read_small(review_path)
    preparation.verify(raw['input_sha256']);preparation.verify(review['input_sha256'])
    prep,static,replay=load_fresh_static_inputs(context)
    route_output,run_id=prepath_motion_identity(context)
    reservation=preparation.read_small(result.parent/'single_route_reserved.json')
    if (raw['verdict']!='HOLD' or raw['automatic_route_Play_requested'] is not False
            or raw['path_execution_started'] is not False or raw['sample_count']!=0
            or raw['fixed_hold_attempt_count']!=0 or result.with_suffix('.csv').exists()
            or raw['error']!='RuntimeError: Stage or paused timeline changed during full compact environment readback'
            or raw['actual_final_timeline']!=static['actual_final_timeline']
            or raw['static_gate_sha256']!=preparation.sha(context.stationary.RESULT)
            or reservation['run_id']!=context.route_run_id
            or review['status']!='RETAINED_PREPATH_HOLD_GET_ONLY_FIELD_REVIEW_NOT_DRIVE_ADMISSION'
            or review['original_route_result']!=str(result) or review['original_route_verdict']!='HOLD'
            or review['observed_current_layer_deltas']!=[]
            or review['all_4997_protected_environment_rows_verified'] is not True
            or review['full_source_actor_configuration_unchanged'] is not True
            or review['protected_environment_sha256']!=context.protocol['protected_environment_sha256']
            or review['actual_tensor_before_and_after']!=static['actual_post_static_tensor_snapshot']
            or review['actual_timeline_before_and_after']!=static['actual_final_timeline']
            or review['new_trial_count']!=0 or review['any_Play_Stop_Pause_reset_or_control_or_scene_edit'] is not False):
        raise ValueError('Exact own no-Play HOLD plus unchanged raw static/GET review required, not a physical retry')
    if any(p.is_file() for p in route_output.rglob('*')):
        raise ValueError('The separately approved first-motion dispatch already reserved/recorded; no retry')
    source_review=verify_prepath_instrumentation_amendment(context)
    target=context.output/'prepath_hold_continuation'
    files=(result,result.parent/'single_route_reserved.json',review_path,context.protocol_path,
        context.source_archive,context.preparation.LIVE,context.stationary.RESULT,context.stationary.TRACE)
    return canonical(dict(version='H4_RETAINED_STATIC_FIRST_MOTION_AFTER_PREPATH_HOLD_V1',
        original_protocol=str(context.protocol_path),cohort=context.protocol['cohort'],method=context.protocol['method'],
        new_robot_root=context.cohort.ROBOT,original_prepath_HOLD=str(result),original_HOLD_NOT_relabelled=True,
        protocol=str(target/'protocol.json'),new_runtime_archive=str(target/'runtime_sources.zip'),
        runtime_archive_includes_visual_sources=True,route_output=str(route_output),route_run_id=run_id,
        required_retained_tensor=review['actual_tensor_before_and_after'],required_retained_timeline=review['actual_timeline_before_and_after'],
        static_replay=replay,source_amendment_review=source_review,
        remaining_authorized_route_attempts=1,additional_static_attempts=0,additional_Stop_commands=0,
        additional_actor_staging_or_visual_refresh=0,automatic_retry=False,
        original_plan_controller_physics_guards_and_acceptance_unchanged=True,
        historical_changed_fields_unknown_NOT_waived_or_claimed_reconstructed=True,
        explicit_new_single_motion_consent_required=True,actual_drive_admission=False,use_for_final_statistics=False,
        input_sha256={**context.protocol['input_sha256'],**static['input_sha256'],
            **{str(p):preparation.sha(p) for p in files},
            str(target/'runtime_sources.zip'):hashlib.sha256(fresh_runtime_source_bundle(include_visual=True)).hexdigest()}))


def prepare_first_route_continuation(path):
    """Offline definition for the unconsumed first route; never another static."""
    context=load_archived_fresh_trial_protocol(path)
    if (context.output/'preflight_hold_review/live_review.json').exists():
        return prepare_prepath_hold_continuation(context)
    workflow=preparation.read_small(context.output/'workflow_reserved.json')
    error=preparation.read_small(context.output/'workflow_error.json')
    if (workflow['cohort']!=context.protocol['cohort'] or workflow['method']!=context.protocol['method']
            or workflow['protocol_sha256']!=preparation.sha(context.protocol_path)
            or workflow['explicit_workflow_consent_checked_by_entry'] is not True
            or workflow['no_automatic_retry'] is not True
            or error!=dict(status='WORKFLOW_STOPPED_NO_RETRY',
                error='ValueError: This fresh static must start at its own STOPPED zero time, not a retained prior trial',
                no_second_Stop_reset_or_recovery_command=True,
                static_or_path_outcome_NOT_inferred_from_this_wrapper=True)
            or context.protocol['maximum_route_attempts']!=1
            or context.protocol['maximum_static_attempts']!=1):
        raise ValueError('Only the original clock-loader abort before its first route may be continued')
    route=context.output/'route'
    if any(p.is_file() for p in route.rglob('*')):
        raise ValueError('First route is already reserved/started/recorded; no continuation or retry')
    prep,static,replay=load_fresh_static_inputs(context)
    boundary=context.output/'actual_Stop_boundary.json';archive=context.output/'actual_prior_state_before_Stop.json'
    stopped=preparation.read_small(boundary);prior=preparation.read_small(archive)
    if (prep['cohort']!=context.protocol['cohort'] or prep['new_robot_root']!=context.cohort.ROBOT
            or prep['retired_robot_root']!=context.protocol['retired_robot_root']
            or prep['start_xyyaw']!=context.protocol['start_xyyaw'] or prep['simulation_mass_kg']!=265.
            or prep['archive']!=str(archive) or prep['archive_sha256']!=preparation.sha(archive)
            or prep['Stop_boundary']!=str(boundary) or prep['Stop_boundary_sha256']!=preparation.sha(boundary)
            or stopped['Stop_called_once'] is not True
            or stopped['actual_timeline']!=dict(playing=False,stopped=True,current_time_s=0.)
            or prior['actual_state']!=context.protocol['required_prior_tensor']
            or prior['prior_result']!=context.protocol['required_prior_result']
            or prior['prior_result_sha256']!=preparation.sha(Path(prior['prior_result']))
            or prior['physical_continuity_ends_at_authorized_Stop'] is not True):
        raise ValueError('Own archived prior state -> single completed Stop -> fresh static identity chain required')
    source_review=verify_first_route_source_amendment(context)
    target=context.output/'first_route_continuation'
    files=(context.protocol_path,context.source_archive,context.preparation.LIVE,context.stationary.RESULT,
        context.stationary.TRACE,context.stationary.RESULT.parent/'single_attempt_reserved.json',
        context.output/'workflow_reserved.json',context.output/'workflow_error.json',
        context.output/'actual_Stop_boundary.json',context.output/'actual_prior_state_before_Stop.json')
    pins={**context.protocol['input_sha256'],**static['input_sha256'],
        **{str(item):preparation.sha(item) for item in files},
        str(target/'runtime_sources.zip'):hashlib.sha256(fresh_runtime_source_bundle()).hexdigest()}
    return canonical(dict(version='H4_SAME_FRESH_STATIC_FIRST_ROUTE_CLOCK_ADMISSION_AMENDMENT',
        role='software_only_pre_route_continuation_NOT_a_retrial_or_new_physical_cohort',
        original_protocol=str(context.protocol_path),cohort=context.protocol['cohort'],
        method=context.protocol['method'],new_robot_root=context.cohort.ROBOT,
        original_fresh_workflow_error_preserved=True,original_static_PASS_and_CSV_unchanged=True,
        old_runtime_archive_preserved=True,new_runtime_archive=str(target/'runtime_sources.zip'),
        required_retained_timeline=static['actual_final_timeline'],
        required_retained_tensor=static['actual_post_static_tensor_snapshot'],
        source_amendment_review=source_review,static_replay=replay,
        protocol=str(target/'protocol.json'),route_output=str(route),route_run_id=context.route_run_id,
        remaining_authorized_route_attempts=1,additional_static_attempts=0,additional_Stop_commands=0,
        reset_or_pose_restore=False,automatic_retry=False,
        original_plan_controller_physics_guards_and_acceptance_unchanged=True,
        actual_drive_admission=False,use_for_final_statistics=False,input_sha256=pins))


def load_first_route_continuation(path):
    path=Path(path);record=preparation.read_small(path)
    context=load_archived_fresh_trial_protocol(record['original_protocol'])
    if record.get('version')=='H4_RETAINED_STATIC_FIRST_MOTION_AFTER_PREPATH_HOLD_V1':
        if path!=context.output/'prepath_hold_continuation/protocol.json' or record!=prepare_prepath_hold_continuation(context):
            raise ValueError('Exact separately approved unchanged-static first-motion continuation required')
        preparation.verify(record['input_sha256'])
        if Path(record['new_runtime_archive']).read_bytes()!=fresh_runtime_source_bundle(include_visual=True):
            raise ValueError('Reviewed first-motion runtime changed before dispatch')
        context.route_continuation_path=path;context.route_continuation=record
        context.route_output=Path(record['route_output']);context.route_run_id=record['route_run_id']
        return context
    expected=context.output/'first_route_continuation'/'protocol.json'
    if path!=expected or record!=prepare_first_route_continuation(context.protocol_path):
        raise ValueError('Exact reviewed same-static first-route continuation required')
    preparation.verify(record['input_sha256'])
    if Path(record['new_runtime_archive']).read_bytes()!=fresh_runtime_source_bundle():
        raise ValueError('Amended runtime source changed; no route')
    context.route_continuation_path=path;context.route_continuation=record
    return context


async def run_fresh_whole_trial(path):
    """One approved fresh preparation/static/route workflow; no retry branch."""
    import runpy
    context=load_fresh_trial_protocol(path)
    try:
        await prepare_fresh_live_trial(context)
        static=_bound(run_retained_static,preparation=context.preparation,cohort=context.cohort,
            STATIC_OUTPUT=context.stationary.RESULT.parent,STATIC_RUN_ID=context.static_run_id,
            STATIC_DESIGN=context.protocol_path,evaluate_support_queries_for_actor=context.support_evaluator)
        result=await static(fresh=context)
        if result['verdict']!='PASS':
            print('[H4 FRESH] Static did not PASS; whole route NOT started. No retry.',flush=True)
            return result
        load_fresh_static_inputs(context)
        if context.protocol.get('restore_new_actor_visual_cache_after_static_only') is True:
            # Reuse the already verified visual-only transaction, while the
            # OWN completed static remains PAUSED. This child is awaited;
            # no competing controller, new static or timeline command.
            import asyncio
            display=runpy.run_path(str(_ijcas_path(ROOT, 'isaac/scripts/isaac_refresh_ijcas_hospital_h4_visual_instances_once.py')),
                init_globals={'ijcas_hospital_h4_visual_instance_refresh_authorized':True,
                    'ijcas_hospital_h4_fresh_static_protocol':str(context.protocol_path),
                    'ijcas_hospital_h4_visual_refresh_workflow_owner':asyncio.current_task()},run_name='__main__')
            await display['ijcas_hospital_h4_visual_refresh_task']
            load_fresh_static_inputs(context)
        dispatched=runpy.run_path(str(ENTRY),init_globals={
            'ijcas_hospital_h4_subfloor_compact_path_authorized':True,
            'ijcas_hospital_h4_fresh_trial_protocol':str(context.protocol_path),
            'ijcas_hospital_h4_method':context.protocol['method']},run_name='__main__')
        task=dispatched['ijcas_hospital_h4_path_task']
        import builtins
        builtins.ijcas_h4_latest_path_task=task
        await task
        return preparation.read_small(context.output/'route'/(context.route_run_id+'.json'))
    except Exception as error:
        hold=context.output/'workflow_error.json'
        if not hold.exists():
            write_runtime_evidence(hold,dict(status='WORKFLOW_STOPPED_NO_RETRY',
                error=type(error).__name__+': '+str(error),
                no_second_Stop_reset_or_recovery_command=True,
                static_or_path_outcome_NOT_inferred_from_this_wrapper=True))
        raise
