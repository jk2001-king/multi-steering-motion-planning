"""Audit the actual FINAL forward and five-leg segmented development lineage.

No actuation, old-result edits, replanning, model fit or retrospective
statistical success. A frozen replay INPUT specification is not a physical
executor admission. Different per-leg executor versions remain explicit.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_final_forward_pilot as forward
from hospital_h3b_final_forward_observation import coverage
from hospital_h3b_rolling_force_audit import numeric_rows, anchor_sums_match
from hospital_h3_turn_planning import verify_sources
from hospital_h3_flush_support import load_setup
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from hospital_h3b_reverse_closedloop_prefix import load_profile as load_prefix_profile


WORKSPACE, GATE = forward.WORKSPACE, forward.GATE
RESULT = forward.OUTPUT/'hospital_h3b_final_forward_r01.json'
BASE = GATE/'hospital_h3b_segmented_completion'
VERSION = 'H3B_final_goal_segmented_development_raw_audit_v1'
sha, verify_files, read_trace = forward.sha, forward.verify_files, forward.read_trace
validate_live = forward.validate_live


def joints_from_last(record):
    result = {k[len('joint_'):-len('_rad')]:v for k,v in record['last_recorded_state'].items()
        if k.startswith('joint_') and k.endswith('_rad')}
    if len(result)!=12 or not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Complete finite twelve-joint saved state required')
    return result


def inspect_final_forward():
    info = forward.load_admitted_profile()
    record = json.loads(RESULT.read_bytes())
    verify_sources(record);verify_files(record['frozen_input_sha256'])
    expected = dict(version=forward.VERSION, trial_kind=forward.ROLE, verdict='PASS',
        PhysX_executed=True, original_motor_gains_restored=True, final_forward_component_verified=True,
        original_final_goal_reached_in_segmented_development=True,
        body_pose_reset_during_run=False, caster_state_reset_during_run=False, material_modified_during_run=False,
        source_USD_saved=False, CICS_input_used=False, Nav2_executed=False, use_for_final_statistics=False,
        full_H3B_turning_path_verified=False, full_suffix_execution_verified=False,
        initial_saved_world_xyyaw=info['initial_world_xyyaw'], expected_retained_joint_positions_rad=info['retained_all_joint_positions_rad'],
        prior_second_reverse_result=info['second_reverse_result'], prior_second_reverse_result_sha256=info['second_reverse_result_sha256'],
        target_world_xyyaw=info['mission_goal_xyyaw'], mission_goal_world_xyyaw_UNCHANGED=info['mission_goal_xyyaw'],
        target_is_EXACT_original_final_mission_goal=True, strict_map_digest=info['strict_map_digest'],
        intervention_contract=info['intervention_contract'], selected_model_design_version=forward.DESIGN_VERSION,
        planned_speed_cap_m_s=.030, physical_speed_cap_m_s=.050,
        runtime_motor_baseline_matches_actual_prior_PASS=True)
    if (any(record.get(k)!=v for k,v in expected.items()) or record.get('error') is not None
            or not record.get('checks') or not all(record['checks'].values())):
        raise ValueError('Exact admitted FINAL-forward raw PASS and unchanged original goal/history required')
    forward.validate_motor_baseline(info, record['runtime_motor_baseline'])
    rows = read_trace(RESULT.with_suffix('.csv'))
    if len(rows)!=record['sample_count'] or rows[-1]!=record['last_recorded_state']:
        raise ValueError('FINAL-forward raw CSV / last state disagree')
    _,manifest,geometry,inventory = load_setup(Path(record['geometry_manifest']).parent, WORKSPACE)
    actual = forward.evaluate_component(rows, info['mission_goal_xyyaw'],
        read_trace(Path(record['static_reference']).with_suffix('.csv')),
        record['physics_preparation_snapshot']['simulation_mass_kg'],
        inventory['explicitly_suppressed_assets'][0]['collision']['bound'], record['wheel_conservative_sphere_radii_m'],
        record['contact_report_counts_by_category'], record['contact_callback_errors'],
        record['fixed_hold_attempt_count'], record['direction_gate_events'], info)
    if (actual['verdict']!='PASS' or any(record['checks'].get(k)!=v for k,v in actual['checks'].items())
            or actual['command_reproduction']!=record['command_reproduction']
            or abs(actual['measured_center_distance_m']-record['measured_center_distance_m'])>1e-10
            or record['cumulative_direction_changes']!=actual['cumulative_direction_changes']
            or record['cumulative_measured_reverse_m']!=actual['cumulative_measured_reverse_m']):
        raise ValueError('Final goal, commands, single fixed hold or cumulative resource verdict does not reproduce')
    observer_path = forward.COHORT/'instrumentation/instrumentation.json'
    observer = json.loads(observer_path.read_bytes());verify_sources(observer);verify_files(observer['reference_input_sha256'])
    files = {str(observer_path.parent/n):digest for n,digest in observer['saved_file_sha256'].items()}
    verify_files(files)
    if (observer['original_controller_result_sha256']!=sha(RESULT) or observer['telemetry_errors']
            or observer.get('control_or_physics_parameters_modified_by_observer') is not False):
        raise ValueError('Final observer/primary lineage mismatch or telemetry error')
    joints = numeric_rows(observer_path.parent/'joint_body_solver_trace.csv')
    friction = numeric_rows(observer_path.parent/'wheel_friction_summary.csv')
    with (observer_path.parent/'friction_anchors.csv').open(newline='') as stream:anchors = list(csv.DictReader(stream))
    with (observer_path.parent/'raw_contact_points.csv').open(newline='') as stream:
        contacts = [dict(r,complete_point_data=r['complete_point_data']=='True') for r in csv.DictReader(stream)]
    measured = coverage(joints,friction,anchors,contacts,record,[])
    if any(observer.get(k)!=v for k,v in measured.items()):
        raise ValueError('Final passive six-wheel coverage does not reproduce')
    anchor_error = anchor_sums_match(friction,anchors,observer['native_sensor_paths_in_recorded_wheel_index_order'])
    audit = dict(original_verdict_unchanged='PASS', primary_check_count=len(actual['checks']),
        saved_check_count=len(record['checks']), command_reproduction=actual['command_reproduction'],
        exact_original_goal_world_xyyaw=info['mission_goal_xyyaw'],
        terminal_position_error_m=actual['terminal_position_error_m'], terminal_heading_error_deg=actual['terminal_heading_error_deg'],
        pose_regression_speed_m_s=actual['fixed_terminal_hold_analysis']['pose_regression_speed_m_s'],
        raw_observer_coverage_reproduced=True, maximum_raw_anchor_sum_error_n=anchor_error,
        actual_fixed_hold_attempt_count=record['fixed_hold_attempt_count'],
        cumulative_measured_reverse_m=actual['cumulative_measured_reverse_m'],
        cumulative_direction_changes=actual['cumulative_direction_changes'])
    info.update(final_forward_result=str(RESULT), final_forward_result_sha256=sha(RESULT), final_forward_record=record,
        initial_world_xyyaw=actual['final_world_xyyaw'], expected_saved_last_state=rows[-1],
        retained_all_joint_positions_rad=joints_from_last(record),
        input_sha256={**info['input_sha256'],**record['frozen_input_sha256'],
            str(RESULT):sha(RESULT),str(RESULT.with_suffix('.csv')):sha(RESULT.with_suffix('.csv')),
            str(observer_path):sha(observer_path),**files})
    return info,audit,manifest,geometry,inventory


def audit_chain(info, manifest):
    """Verify four saved PAUSED handoffs, never concatenate unrelated PASSes."""
    names = ('first','reverse','forward','second_reverse','final_forward')
    roles = ('first_forward','first_reverse','next_forward','second_reverse','final_forward')
    directions = (1,-1,1,-1,1)
    modes = ('coordinated_counterphase','rear_only','coordinated_counterphase','front_only','coordinated_counterphase')
    records = [info[n+'_record'] for n in names]
    paths = [Path(info[n+'_result']) for n in names]
    traces = [read_trace(p.with_suffix('.csv')) for p in paths]
    with np.load(manifest['occupancy_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    if sha_bytes(occupied.tobytes())!=info['strict_map_digest'] or occupied.dtype!=np.dtype(bool):
        raise ValueError('Unchanged STRICT flush occupancy required')
    checker = NativeHospitalEnvelopeChecker(occupied,origin,resolution,(0.,0.,0.))
    handoffs=[];segments=[];sampled_total=0.
    for i,(role,direction,mode,record,path,rows) in enumerate(zip(roles,directions,modes,records,paths,traces)):
        if (record['verdict']!='PASS' or not all(record['checks'].values()) or record['fixed_hold_attempt_count']!=1
                or record['original_motor_gains_restored'] is not True
                or any(record.get(k) is not False for k in ('body_pose_reset_during_run','caster_state_reset_during_run','CICS_input_used','Nav2_executed'))
                or record['mission_goal_world_xyyaw_UNCHANGED']!=info['mission_goal_xyyaw']
                or record.get('strict_map_digest',record.get('strict_flush_map_digest',manifest['strict_map_digest']))!=info['strict_map_digest']
                or record['geometry_manifest']!=records[0]['geometry_manifest']
                or record['physics_preparation_snapshot']!=records[0]['physics_preparation_snapshot']
                or len(rows)!=record['sample_count'] or rows[-1]!=record['last_recorded_state']):
            raise ValueError('Five-leg history/physics/goal mismatch: '+role)
        if i:
            previous=records[i-1]
            q=joints_from_last(previous)
            if (record['initial_saved_world_xyyaw']!=previous['final_world_xyyaw']
                    or record['expected_retained_joint_positions_rad']!=q
                    or record.get('initial_paused_state_continuity_verified') is not True):
                raise ValueError('Cannot join reset, substituted or unmatched body/12-joint endpoints: '+role)
            handoffs.append(dict(from_run_id=previous['run_id'],to_run_id=record['run_id'],
                saved_body_difference_m=0.,saved_yaw_difference_deg=0.,maximum_saved_joint_difference_rad=0.,
                all_twelve_joints_and_caster_history_retained=True))
        initial = record.get('initial_saved_world_xyyaw',record.get('observed_prehold_world_xyyaw'))
        if initial is None:initial=[rows[0][k] for k in ('x_m','y_m','yaw_rad')]
        poses=[list(initial)];distance=0.
        for row in rows:
            pose=[row[k] for k in ('x_m','y_m','yaw_rad')];before=poses[-1]
            delta=forward.wrap(pose[2]-before[2]);ds=math.hypot(pose[0]-before[0],pose[1]-before[1]);distance+=ds
            n=max(1,math.ceil(ds/.005),math.ceil(abs(delta)/math.radians(.25)))
            poses.extend([[before[0]+j/n*(pose[0]-before[0]),before[1]+j/n*(pose[1]-before[1]),before[2]+j/n*delta] for j in range(1,n+1)])
        if checker(poses):raise ValueError('Actual interpolated strict +10mm envelope violation: '+role)
        sampled_total+=distance
        bad=sum(record['contact_report_counts_by_category'].get(k,0) for k in forward.BAD_CONTACTS)
        if bad or record['contact_callback_errors']:raise ValueError('Unexpected external contact in chain: '+role)
        segments.append(dict(index=i,role=role,run_id=record['run_id'],result=str(path),result_sha256=sha(path),
            trace_sha256=sha(path.with_suffix('.csv')),direction=direction,mode=mode,
            executor_version=record.get('version','H3B_shared_course_feedback_candidate_v1'),
            measured_terminal_position_error_m=record['terminal_position_error_m'],
            measured_terminal_heading_error_deg=record['terminal_heading_error_deg'],
            measured_center_distance_m=distance,reported_component_distance_m=record.get('measured_center_distance_m'),
            recorded_simulation_s=rows[-1]['time_s'],motion_and_braking_s=record.get('motion_and_braking_time_s'),
            single_fixed_hold_attempts=record['fixed_hold_attempt_count'],
            strict_interpolated_10mm_envelope_clear=True,unexpected_external_contact_callbacks=bad))
    reverse_total = sum(r['measured_center_distance_m'] for r,d in zip(records,directions) if d<0)
    if reverse_total!=info['final_forward_record']['cumulative_measured_reverse_m'] or reverse_total>1.:
        raise ValueError('Measured reverse resource mismatch')
    if info['final_forward_record']['cumulative_direction_changes']!=4:
        raise ValueError('Exactly four retained-state direction changes required')
    return dict(segments=segments,handoffs=handoffs,recorded_segmented_development_chains=1,
        segmented_development_final_goal_reached=True,all_five_raw_component_verdicts='PASS',
        saved_body_and_12_joint_handoffs_verified=len(handoffs),
        cumulative_measured_reverse_m=reverse_total,cumulative_direction_changes=4,
        measured_center_distance_including_all_segment_preholds_m=sampled_total,
        sum_recorded_segment_simulation_s=sum(s['recorded_simulation_s'] for s in segments),
        inter_segment_review_wall_time_NOT_a_traversal_metric=True,
        sequential_development_changes_and_measured_state_replanning_disclosed=True,
        one_fixed_executor_from_start_was_NOT_prospectively_tested=True,
        independent_complete_mission_repeats=0,comparison_success_rate_NOT_computed=True,
        full_H3B_turning_path_verified=False,full_suffix_execution_verified=False,
        statistical_superiority_established=False,robustness_established=False,
        CICS_input_used=False,Nav2_executed=False,use_for_final_statistics=False)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def replay_input_spec(info, chain):
    """Freeze known-successful INPUTS; no generated or admitted new controller."""
    candidate=json.loads(Path(info['suffix_candidate']).read_bytes())
    g,t=candidate['geometry_audit'],candidate['nominal_timing']
    cuts=[0,*t['cusp_sample_indices'],len(g['poses_local_xyyaw'])-1]
    profiles=[]
    for i,(left,right) in enumerate(zip(cuts,cuts[1:])):
        arc=np.array(g['travelled_arc_m'][left:right+1]);arc-=arc[0]
        profiles.append(dict(original_suffix_leg_index=i,original_sample_indices_inclusive=[left,right],
            mode=g['leg_modes'][i],direction=g['leg_directions'][i],
            poses_local_xyyaw=g['poses_local_xyyaw'][left:right+1],center_arc_m=arc.tolist(),
            curvature_1_m=g['sample_curvature_1_m'][left:right+1],reference_length_m=float(arc[-1]),
            next_cusp_local_xyyaw=g['poses_local_xyyaw'][right]))
    # The suffix-admission schema intentionally dropped the legacy prefix
    # table. Reload its original frozen producer, not a measured trace or a
    # newly reconstructed path that could silently change the first leg.
    prefix_info=load_prefix_profile()
    if any(prefix_info[k]!=info[k] for k in ('mission_start_xyyaw','mission_goal_xyyaw','strict_map_digest')):
        raise ValueError('Original frozen prefix task/environment changed')
    prefix=prefix_info['model_prefix_rows']
    endpoint=prefix[-1]
    nominal=[endpoint[k] for k in ('x_m','y_m','yaw_rad')]
    anchor=forward.relative_pose(info['first_record']['final_world_xyyaw'],info['mission_start_xyyaw'])
    return dict(version='H3B_successful_segmented_recipe_INPUT_FREEZE_v1',
        status='FROZEN_DEVELOPMENT_REPLAY_INPUTS_NOT_RUNTIME_ADMISSION',
        original_start_xyyaw=info['mission_start_xyyaw'],original_goal_xyyaw=info['mission_goal_xyyaw'],
        strict_map_digest=info['strict_map_digest'],environment_condition=info['final_forward_record']['environment_condition'],
        original_frozen_prefix_profile=prefix_info['profile'],original_frozen_prefix_profile_sha256=prefix_info['profile_sha256'],
        fixed_prefix_legacy_spatial_rows=prefix,fixed_suffix_spatial_legs=profiles,
        frozen_development_executors=chain['segments'],
        planned_speed_caps_m_s_by_leg=[.050,.050,.050,.050,.030],
        existing_course_gains={'yaw_error_gain_1_m_rad':1.5,'cross_error_gain_1_m2':2.},
        original_prefix_nominal_cusp_local_xyyaw=nominal,
        measured_anchor_used_to_design_the_fixed_suffix_local_xyyaw=list(anchor),
        prefix_nominal_to_suffix_design_anchor_gap_m=math.hypot(anchor[0]-nominal[0],anchor[1]-nominal[1]),
        prefix_nominal_to_suffix_design_anchor_heading_deg=math.degrees(forward.wrap(anchor[2]-nominal[2])),
        suffix_was_designed_from_a_measured_development_endpoint_NOT_an_independent_full_route=True,
        do_NOT_snap_recenter_or_edit_waypoints_to_join_nominal_endpoints=True,
        next_gate='ONE_FRESH_ORIGINAL_START_FULL_SEQUENCE_DEVELOPMENT_PILOT_AFTER_ORCHESTRATOR_ADMISSION',
        must_freeze_before_first_new_attempt=['single_orchestrator_and_per_leg_dispatch',
            'fixed_references_and_targets','unchanged_start_environment_and_natural_caster_seed',
            'measured_handoff_and_endpoint_acceptance','all_attempt_recording_and_time_resource_budgets'],
        must_check_before_new_motion=['whole_sequence_model_and_strict_swept_envelope',
            'nominal_prefix_to_measured_suffix_anchor_handoff_without_pose_or_joint_reset',
            'fresh_geometry_physics_and_raw_static_PASS','dedicated_runtime_preflight'],
        failure_policy='Record every preflight/drive/terminal failure separately; no automatic retry, success filtering or new endpoint recapture',
        future_comparisons_require_same_prospectively_frozen_executor_and_initial_conditions=True,
        ordinary_initial_ramp_and_common_stop_hold_are_NOT_CICS=True,
        CICS_runtime_technical_integration=False,full_sequence_runtime_prepared=False,
        new_PhysX_motion_executed=False,automatic_motion_authorized=False,use_for_final_statistics=False)


def load_audit(directory):
    directory=Path(directory).resolve()
    report=json.loads((directory/'segmented_completion_audit.json').read_bytes())
    verify_sources(report);verify_files(report['input_sha256'])
    info,audit,manifest,geometry,inventory=inspect_final_forward()
    chain=audit_chain(info,manifest)
    if (report.get('version')!=VERSION
            or report.get('status')!='SEGMENTED_DEVELOPMENT_FINAL_GOAL_RAW_PASS_FRESH_FULL_SEQUENCE_PENDING'
            or report.get('final_forward_result_sha256')!=info['final_forward_result_sha256']
            or report.get('final_actual_world_xyyaw')!=info['initial_world_xyyaw']
            or report.get('original_goal_world_xyyaw')!=info['mission_goal_xyyaw']
            or report.get('final_all_twelve_joint_positions_rad')!=info['retained_all_joint_positions_rad']
            or report.get('final_forward_raw_audit')!=audit or report.get('chain_audit')!=chain
            or report.get('new_PhysX_motion_executed') is not False
            or report.get('automatic_motion_authorized') is not False
            or report.get('CICS_input_used') is not False or report.get('use_for_final_statistics') is not False):
        raise ValueError('Exact raw five-leg segmented-development closeout required; not statistical or motion authority')
    frozen=json.loads((directory/'frozen_replay_inputs.json').read_bytes())
    expected=replay_input_spec(info,chain)
    if any(frozen.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen replay INPUTS changed; never treat them as an admitted executor')
    verify_files(frozen['input_sha256'])
    return info,audit,manifest,geometry,inventory,report
