#!/usr/bin/env python3
"""Audit whether the hospital-paper experiment can leave the setup phase."""

from __future__ import annotations

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import math
from pathlib import Path

import yaml


WORKSPACE = _ijcas_workspace()
CONFIG = _ijcas_path(WORKSPACE, 'studies/IJCAS/config')
RESULT = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
PROTOCOL_PATH = CONFIG / "hospital_paper_validation_protocol.yaml"
MEASUREMENT_PATH = CONFIG / "hospital_environment_measurements.yaml"
MEASUREMENT_TEMPLATE = CONFIG / "hospital_environment_measurement_template.yaml"


def load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def value_at(data, dotted_path):
    value = data
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def protocol_checks_for(protocol):
    methods = protocol.get("methods", {})
    return {
        "six_scenarios_declared": len(protocol.get("scenarios", {})) == 6,
        "four_planning_methods_declared": set(methods) == {
            "B0_fixed_counterphase_ramp", "B1_multimode_kinematic_ramp",
            "B2_multimode_transition_cost_ramp", "P_dynamics_full_envelope",
        },
        "cics_is_separate_prior_work": value_at(protocol, "research_scope.cics_role")
        == "prior_work_and_separately_identified_existing_results",
        "forward_micro_yaw_is_not_required": value_at(
            protocol, "research_scope.new_forward_micro_yaw_controller_required") is False,
        "matched_initial_controller": bool(methods) and all(
            method.get("initial_motion") == "acceleration_limited_ramp" for method in methods.values()),
        "matched_terminal_controller": bool(methods) and all(
            method.get("terminal_controller") == "common_bounded_2d_mode_switching" for method in methods.values()),
        "planning_hypothesis_declared": "H2_planning_dynamics" in protocol.get("hypotheses", {}),
        "final_repeat_count_is_ten": value_at(
            protocol, "trial_stages.final_matched_comparison.repeats_per_method_scenario") == 10,
        "paired_seed_required": value_at(
            protocol, "trial_stages.final_matched_comparison.paired_seed_required") is True,
    }


METHODS=('B0_fixed_counterphase_ramp','B1_multimode_kinematic_ramp',
    'B2_multimode_transition_cost_ramp','P_dynamics_full_envelope')


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def trace_metrics(path):
    """Descriptive observations only; no telemetry-row pseudo-replication."""
    phases=Counter();count=0;first=last=None;previous=None;tail=[];errors=[]
    elapsed=Counter();distance=Counter();tracking_position=[];tracking_heading=[]
    weighted_position=weighted_heading=weighted_seconds=0.;previous_row=None
    hold_episodes=0;commands=[];body_speeds=[];hold_phase_times=[]
    def number(row,key,scale=1.):
        if key not in row or row[key] in ('',None):return None
        try:
            value=float(row[key])*scale
            if not math.isfinite(value):raise ValueError('nonfinite')
            return value
        except (TypeError,ValueError):
            message='invalid_numeric_metric:'+key
            if message not in errors:errors.append(message)
            return None
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            count+=1;phase=row.get('phase','static_observation');phases[phase]+=1
            try:
                t=float(row['time_s'])
                if not math.isfinite(t) or previous is not None and t<=previous:
                    raise ValueError('nonfinite_or_nonincreasing_time')
                if phase=='terminal_hold' and (previous_row is None or previous_row.get('phase')!='terminal_hold'):
                    hold_episodes+=1
                if previous is not None:
                    dt=t-previous;elapsed[phase]+=dt
                    coords=[number(point,key) for point in (previous_row,row) for key in ('x_m','y_m')]
                    if all(value is not None for value in coords):
                        distance[phase]+=math.hypot(coords[2]-coords[0],coords[3]-coords[1])
                    if phase=='path_tracking' and previous_row.get('phase')=='path_tracking':
                        a,b=number(previous_row,'path_error_m',1000.),number(row,'path_error_m',1000.)
                        c,d=number(previous_row,'path_heading_error_deg'),number(row,'path_heading_error_deg')
                        if all(value is not None for value in (a,b,c,d)):
                            weighted_position+=dt*(a*a+b*b)/2
                            weighted_heading+=dt*(c*c+d*d)/2;weighted_seconds+=dt
                if phase=='path_tracking':
                    position=number(row,'path_error_m',1000.);heading=number(row,'path_heading_error_deg')
                    if position is not None:tracking_position.append(position)
                    if heading is not None:tracking_heading.append(heading)
                command=number(row,'command_speed_m_s',1000.)
                if command is not None:commands.append(abs(command))
                vx,vy=number(row,'vx_physx_m_s',1000.),number(row,'vy_physx_m_s',1000.)
                if vx is not None and vy is not None:body_speeds.append(math.hypot(vx,vy))
                previous=t;previous_row=row
                if first is None:first=t
                last=row
                if phase=='terminal_hold':
                    phase_time=number(row,'phase_time_s')
                    if phase_time is not None:
                        hold_phase_times.append(phase_time)
                        if phase_time>=2.:tail.append(row)
            except (KeyError,ValueError,TypeError) as error:
                if str(error) not in errors:errors.append(str(error))
    def finite(key,scale=1.):
        try:
            value=float(last[key])*scale
            return value if math.isfinite(value) else None
        except (TypeError,KeyError,ValueError):return None
    tail_duration=None
    if hold_episodes>1:errors.append('multiple_terminal_hold_episodes')
    if tail and hold_episodes==1:tail_duration=float(tail[-1]['time_s'])-float(tail[0]['time_s'])
    def summary(values):
        if not values:return dict(sample_count=0,rmse=None,absolute_p95=None,maximum_absolute=None)
        ordered=sorted(abs(v) for v in values);rank=.95*(len(ordered)-1)
        low=int(math.floor(rank));high=int(math.ceil(rank));fraction=rank-low
        return dict(sample_count=len(values),rmse=math.sqrt(math.fsum(v*v for v in values)/len(values)),
            absolute_p95=ordered[low]*(1-fraction)+ordered[high]*fraction,maximum_absolute=ordered[-1])
    fixed_tail=None
    if tail_duration is not None and len(tail)>=2 and tail_duration>0.:
        times=[float(row['time_s']) for row in tail];mean_t=math.fsum(times)/len(times)
        x=[number(row,'x_m',1000.) for row in tail];y=[number(row,'y_m',1000.) for row in tail]
        positions=[number(row,'goal_position_error_m',1000.) for row in tail]
        headings=[number(row,'goal_heading_error_deg') for row in tail]
        velocities=[(number(row,'vx_physx_m_s',1000.),number(row,'vy_physx_m_s',1000.)) for row in tail]
        if all(value is not None for value in (*x,*y,*positions,*headings)):
            centered=[t-mean_t for t in times];denom=math.fsum(t*t for t in centered)
            mean_x,mean_y=math.fsum(x)/len(x),math.fsum(y)/len(y)
            sx=math.fsum(t*(v-mean_x) for t,v in zip(centered,x))/denom
            sy=math.fsum(t*(v-mean_y) for t,v in zip(centered,y))/denom
            fixed_tail=dict(regression_speed_mm_s=math.hypot(sx,sy),
                maximum_displacement_from_first_tail_pose_mm=max(math.hypot(a-x[0],b-y[0]) for a,b in zip(x,y)),
                maximum_goal_error_mm=max(positions),maximum_absolute_heading_error_deg=max(map(abs,headings)),
                maximum_measured_horizontal_speed_mm_s=max(math.hypot(a,b) for a,b in velocities)
                    if all(a is not None and b is not None for a,b in velocities) else None)
    hold_elapsed=max(hold_phase_times) if hold_phase_times and hold_episodes==1 else None
    complete=bool(hold_elapsed is not None and hold_elapsed>=5.
        and tail_duration is not None and tail_duration>=2.9)
    return dict(sample_count=count,phases=dict(phases),first_physics_time_s=first,
        last_physics_time_s=finite('time_s') if last else None,
        observed_duration_s=finite('time_s')-first if last and first is not None and finite('time_s') is not None else None,
        last_position_error_mm=finite('goal_position_error_m',1000.) if last else None,
        last_heading_error_deg=finite('goal_heading_error_deg') if last else None,
        last_path_progress_m=finite('path_progress_m') if last else None,
        fixed_tail_sample_count=len(tail),fixed_tail_duration_s=tail_duration,
        terminal_hold_episode_count=hold_episodes,fixed_hold_observed_phase_elapsed_s=hold_elapsed,
        fixed_hold_remaining_to_5s_s=max(0.,5.-hold_elapsed) if hold_elapsed is not None else None,
        fixed_hold_and_tail_observation_complete=complete,
        fixed_tail_observations=fixed_tail if complete else None,
        partial_hold_tail_observations_NOT_acceptance=fixed_tail if not complete else None,
        tracking_only_position_error_mm=summary(tracking_position),tracking_only_heading_error_deg=summary(tracking_heading),
        tracking_only_time_weighted_position_RMS_mm=math.sqrt(weighted_position/weighted_seconds) if weighted_seconds else None,
        tracking_only_time_weighted_heading_RMS_deg=math.sqrt(weighted_heading/weighted_seconds) if weighted_seconds else None,
        tracking_only_time_weighted_covered_duration_s=weighted_seconds,
        sampled_phase_duration_s=dict(elapsed),sampled_phase_travel_m=dict(distance),
        maximum_logged_command_speed_mm_s=max(commands) if commands else None,
        maximum_logged_measured_horizontal_speed_mm_s=max(body_speeds) if body_speeds else None,
        phase_duration_convention='Observed adjacent intervals assigned to right-endpoint phase; startup outside CSV is not invented.',
        time_weighted_error_convention='Trapezoidal squared logged errors within tracking-only intervals; no continuous safety bound.',
        terminal_correction_is_NOT_included_in_tracking_error=True,
        fixed_tail_convention='Exactly the single terminal_hold phase at phase_time_s>=2; no later/best window selection.',
        CSV_structure_errors=errors,telemetry_rows_are_NOT_independent_trials=True)


def execution_duration_diagnosis(raw,metrics):
    """Keep incomplete holds and duration guards distinct from full task PASS."""
    limit=raw.get('guards',{}).get('maximum_simulation_duration_s')
    if limit is None:return None
    entry=raw.get('fixed_hold_entry_candidate')
    return dict(original_verdict=raw['verdict'],original_error=raw.get('error'),
        original_whole_trial_simulation_budget_s=limit,
        last_logged_route_clock_s=metrics.get('last_physics_time_s'),
        whole_trial_includes_prehold_tracking_terminal_and_hold=True,
        fixed_hold_entry_candidate_NOT_success=entry,
        completed_fixed_hold_and_required_tail_observed=metrics.get('fixed_hold_and_tail_observation_complete',False),
        recorded_fixed_hold_elapsed_s=metrics.get('fixed_hold_observed_phase_elapsed_s'),
        missing_hold_duration_at_last_CSV_sample_s=metrics.get('fixed_hold_remaining_to_5s_s'),
        last_goal_position_error_mm=metrics.get('last_position_error_mm'),
        last_goal_heading_error_deg=metrics.get('last_heading_error_deg'),
        partial_tail_does_NOT_establish_required_five_second_standstill=True,
        compound_duration_or_gap_error_not_rewritten_as_an_isolated_cause=True,
        no_extra_time_appended_to_original_trial=True,
        no_new_or_relabelled_verdict=True)


def classify_execution_record(raw,metrics,expected_role):
    """Do not turn setup/software failures into planning-performance limits."""
    if expected_role=='initial_motion':role='ENGINEERING_INITIAL_MOTION'
    elif expected_role=='static':role='ENGINEERING_STATIC'
    elif expected_role=='segmented_component':role='SEGMENTED_COMPONENT_NOT_WHOLE_ROUTE'
    elif raw.get('method') in METHODS:role='METHOD_PATH_DEVELOPMENT'
    else:role='UNCLASSIFIED_NOT_FINAL'
    phases=metrics.get('phases',{})
    reached=bool(raw.get('path_execution_started') or phases.get('path_tracking',0))
    if role=='METHOD_PATH_DEVELOPMENT':
        stage='ACTUAL_PATH_OUTCOME' if reached else 'SETUP_OR_PREPATH_NO_PLANNING_PERFORMANCE_OUTCOME'
    elif role=='SEGMENTED_COMPONENT_NOT_WHOLE_ROUTE':stage='COMPONENT_OUTCOME_NOT_WHOLE_SEQUENCE'
    else:stage='ENGINEERING_OUTCOME_NOT_PLANNING_COMPARISON'
    return role,stage


def actual_h4_authored_envelope_observations(path,raw,metrics):
    """Sample measured poses against the SAME frozen 2D authored map bounds.

    This is a descriptive clearance proxy, not cooked mesh clearance, a
    continuous/loaded-patient certificate, or replacement for actual contacts.
    Never substitute a nominal reference trajectory for missing measurements.
    """
    if raw.get('scenario')!='H4_ward_corridor_turn':return None
    trace=Path(path).with_suffix('.csv')
    if not trace.is_file() or metrics.get('sample_count',0)==0:
        return dict(status='NO_ACTUAL_POSE_TRACE_NO_CLEARANCE_INVENTED')
    if metrics.get('CSV_structure_errors'):
        return dict(status='INVALID_ACTUAL_TRACE_NO_CLEARANCE_COMPUTED')
    # These are the original planner's stored bounds, not queried or modified
    # scene geometry. The actual outcome must explicitly pin this SAME map.
    root=next((p for p in Path(path).parents if p.name=='hospital_paper_gate'),None)
    if root is None:return dict(status='OWN_FROZEN_GEOMETRY_REFERENCE_UNAVAILABLE')
    manifest=_ijcas_path(root, 'hospital_h4_setup/candidate_r01/hospital_h4_geometry_candidate.json')
    if (not manifest.is_file() or raw.get('trace_sha256')!=file_sha(trace)
            or raw.get('input_sha256',{}).get(str(manifest))!=file_sha(manifest)
            or raw.get('geometry_manifest_sha256')!=file_sha(manifest)):
        return dict(status='MEASURED_TRACE_OR_OWN_GEOMETRY_PIN_UNVERIFIED_NO_CLEARANCE_COMPUTED')
    setup=json.loads(manifest.read_bytes())
    if (raw.get('planning_occupancy_digest')!=setup.get('candidate_occupancy_digest')
            or setup.get('footprint')!={'length_m':1.630,'width_m':.965,'clearance_m':.010}):
        return dict(status='FROZEN_MAP_OR_FULL_FOOTPRINT_MISMATCH_NO_CLEARANCE_COMPUTED')
    import hospital_h4_geometry_metrics as geometry
    colliders=[dict(path=row['path'],world_aligned_bound=row['world_aabb'])
        for row in setup['included_collision_AABBs']]
    with trace.open(newline='') as stream:
        poses=[[float(row[key]) for key in ('x_m','y_m','yaw_rad')] for row in csv.DictReader(stream)]
    if len(poses)!=metrics['sample_count']:raise ValueError('Actual pose trace count changed during envelope review')
    result=dict(status='ACTUAL_POSE_SAMPLED_AUTHORED_AABB_SEPARATION_NOT_COOKED_CLEARANCE',
        measured_pose_count=len(poses),original_mapped_obstacle_count=len(colliders),
        obstacle_height_slab_m=setup['obstacle_height_slab_m'],
        all_logged_phases_included_NOT_best_window=True,
        raw_body_footprint=geometry.sampled_separation(poses,colliders),
        expanded_10mm_footprint=geometry.sampled_separation(poses,colliders,clearance=.010),
        missing_unlogged_initial_or_final_motion_NOT_reconstructed=True,
        sampled_interpolation_is_NOT_continuous_collision_or_contact_margin_evidence=True,
        actual_contacts_remain_a_separate_authoritative_channel=True,
        upper_loaded_patient_3D_clearance_NOT_certified=True,
        no_safety_or_statistical_superiority_admission=True,
        input_sha256={str(manifest):file_sha(manifest),str(trace):file_sha(trace),
            str(Path(geometry.__file__)):file_sha(geometry.__file__)})
    if (result['input_sha256'][str(trace)]!=raw['trace_sha256']
            or result['input_sha256'][str(manifest)]!=raw['geometry_manifest_sha256']):
        raise ValueError('Measured trace or frozen geometry changed during envelope review')
    return result


def inspect_execution_record(path,expected_role,*,include_envelope_observations=True):
    path=Path(path)
    if path.stat().st_size>4*1024*1024:raise ValueError('Ledger never reads giant scene archives')
    raw=json.loads(path.read_bytes());trace=path.with_suffix('.csv');errors=[]
    if raw.get('verdict') not in ('PASS','FAIL','HOLD'):raise ValueError('Not an actual outcome record: '+str(path))
    metrics=trace_metrics(trace) if trace.exists() else dict(sample_count=0,phases={})
    errors.extend('CSV:'+message for message in metrics.get('CSV_structure_errors',[]))
    if trace.exists() and raw.get('trace_sha256'):
        pin='ORIGINAL_TRACE_HASH_VERIFIED' if file_sha(trace)==raw['trace_sha256'] else 'TRACE_HASH_MISMATCH'
        if pin=='TRACE_HASH_MISMATCH':errors.append('original_trace_hash_mismatch')
    elif trace.exists():pin='LEGACY_UNPINNED_TRACE_CURRENT_SNAPSHOT_ONLY'
    else:pin='TRACE_ABSENT'
    role,stage=classify_execution_record(raw,metrics,expected_role)
    if raw['verdict']=='PASS' and (not raw.get('checks') or any(v is not True for v in raw['checks'].values())):
        errors.append('PASS_without_all_original_checks_true')
    if raw['verdict']=='PASS' and metrics['sample_count']==0:
        errors.append('PASS_without_physical_trace_samples')
    if raw['verdict']=='PASS' and role=='METHOD_PATH_DEVELOPMENT' and stage!='ACTUAL_PATH_OUTCOME':
        errors.append('method_PASS_without_path_execution_evidence')
    if raw.get('sample_count') is not None and metrics['sample_count']!=raw['sample_count']:
        errors.append('raw_sample_count_differs_from_CSV')
    required=['scenario','execution_version','execution_role','execution_cohort','geometry_manifest_sha256',
        'start_world_xyyaw','goal_world_xyyaw','command_limits','guards','acceptance','terminal_controller_config','feedback_source',
        'planning_occupancy_digest','shared_candidate_pool_sha256','generator_selector_version',
        'source_authored_caster_seed_policy','physical_condition','physics_baseline_sha256','source_sha256']
    controls={key:raw.get(key) for key in required}
    envelope=actual_h4_authored_envelope_observations(path,raw,metrics) if (
        role=='METHOD_PATH_DEVELOPMENT' and include_envelope_observations) else None
    return dict(run_id=raw['run_id'],result=str(path),reported_verdict=raw['verdict'],
        evidence_role=role,outcome_stage=stage,method=raw.get('method'),scenario=raw.get('scenario'),
        execution_cohort=raw.get('execution_cohort'),execution_version=raw.get('execution_version'),
        environment_variant=raw.get('environment_variant','ORIGINAL_OR_UNSPECIFIED_NOT_POOLED'),
        physical_condition=raw.get('physical_condition'),physics_baseline_sha256=raw.get('physics_baseline_sha256'),
        reported_error=raw.get('error'),evaluation_error=raw.get('evaluation_error'),
        failed_checks=[k for k,v in raw.get('checks',{}).items() if v is not True],
        trace=str(trace) if trace.exists() else None,trace_pin_status=pin,trace_observations=metrics,
        before_common_terminal=raw.get('terminal_handoff'),terminal_pulses=raw.get('terminal_pulse_count'),
        fixed_hold_entry=raw.get('fixed_hold_entry_candidate'),duration_diagnosis=execution_duration_diagnosis(raw,metrics),
        terminal_mode_switches=raw.get('terminal_execution_mode_switches'),
        reported_position_error_mm=1000*raw['terminal_position_error_m'] if raw.get('terminal_position_error_m') is not None else None,
        reported_heading_error_deg=raw.get('terminal_heading_error_deg'),
        motion_plus_terminal_time_s=raw.get('motion_and_terminal_correction_time_s'),
        obstacle_callback_count=raw.get('contact_report_counts_by_category',{}).get('hospital_obstacle',0),
        original_motor_gains_restored=raw.get('original_motor_gains_restored'),
        reported_final_statistics_flag=raw.get('use_for_final_statistics',False),
        final_comparison_qualification_NOT_inferred=True,statistical_superiority_established=False,
        original_controller_verdict_NOT_relabelled=True,comparison_controls=controls,
        sampled_authored_envelope_observations=envelope,
        missing_comparison_controls=[k for k,v in controls.items() if v is None],integrity_errors=errors)


def development_comparison_groups(records):
    """Keep controller/cohort/environment identities; never select best/latest."""
    grouped=defaultdict(list)
    for row in records:
        if row['evidence_role']=='METHOD_PATH_DEVELOPMENT':
            grouped[(row['scenario'],row['execution_cohort'],row['execution_version'],row['environment_variant'])].append(row)
    groups=[]
    for identity,rows in sorted(grouped.items(),key=lambda item:repr(item[0])):
        reference=rows[0]['comparison_controls'];present={r['method'] for r in rows}
        mismatches={r['run_id']:[key for key in reference if r['comparison_controls'][key]!=reference[key]] for r in rows}
        repeats={method:sum(r['method']==method for r in rows) for method in METHODS}
        groups.append(dict(scenario=identity[0],cohort=identity[1],execution_version=identity[2],environment_variant=identity[3],
            method_run_counts=repeats,missing_methods=sorted(set(METHODS)-present),
            actual_path_outcomes={method:[r['reported_verdict'] for r in rows if r['method']==method
                and r['outcome_stage']=='ACTUAL_PATH_OUTCOME'] for method in METHODS},
            control_mismatches={k:v for k,v in mismatches.items() if v},
            missing_controls_by_run={r['run_id']:r['missing_comparison_controls'] for r in rows if r['missing_comparison_controls']},
            all_attempts_retained_no_best_or_latest_selection=True,
            independent_paired_entry_history_NOT_established_by_this_ledger=True,
            confidence_intervals_or_significance_NOT_inferred_from_pilot_or_telemetry_counts=True,
            final_comparison_qualification=False))
    return groups


def execution_artifact_identity_review(records,pins):
    """Expose copied receipts without assuming byte-identical runs independent."""
    methods=[row for row in records if row['evidence_role']=='METHOD_PATH_DEVELOPMENT']
    identities=defaultdict(list);traces=defaultdict(list)
    for row in methods:
        identities[row['run_id']].append(row)
        if row.get('trace') and row['trace'] in pins:traces[pins[row['trace']]].append(row)
    return dict(method_result_artifact_count=len(methods),distinct_saved_run_id_count=len(identities),
        unique_trace_byte_count_NOT_independent_repeat_count=len(traces),
        repeated_run_id_artifacts=[dict(run_id=run_id,results=[row['result'] for row in rows],
            trace_sha256_by_result={row['result']:pins.get(row.get('trace')) for row in rows},
            new_independent_trial_count_NOT_inferred=True) for run_id,rows in sorted(identities.items()) if len(rows)>1],
        identical_trace_byte_groups=[dict(trace_sha256=sha,results=[row['result'] for row in rows],
            run_ids=[row['run_id'] for row in rows],
            duplicate_physical_trial_or_independence_NOT_proven_by_bytes_alone=True)
            for sha,rows in sorted(traces.items()) if len(rows)>1],
        all_result_artifacts_retained_no_best_latest_or_success_selection=True,
        artifact_or_unique_trace_count_does_NOT_establish_independent_paired_repeats=True,
        final_comparison_qualification=False)


def native_planning_mechanism_review(plans):
    """Describe ALL frozen selectors without attributing MODEL differences to physics.

    Rejected predicates can overlap. A passing force/load constraint is not
    evidence that it caused selection, and a kinematic terminal velocity of
    None is not stopping evidence. Native costs use different fidelities and
    must not be ranked as common executed time.
    """
    required=('method','selected_actions','selected_states','selected_candidate_id',
        'native_candidate_evaluations','eligible_candidate_count','shared_candidate_count',
        'all_pool_candidates_considered','planner_dynamics','initial_steering_propagated')
    if any(any(key not in plan for key in required) for plan in plans):
        return dict(status='NATIVE_PLANNING_DETAILS_NOT_AVAILABLE_NOT_INVENTED',
            actual_drive_admission=False,statistical_superiority_established=False)
    if len(plans)!=4 or {p['method'] for p in plans}!=set(METHODS):
        raise ValueError('All four native selectors required; no successful-method selection')
    controls=('scenario','start_world_xyyaw','goal_world_xyyaw','footprint',
        'planning_occupancy_digest','geometry_manifest_sha256','shared_candidate_pool_sha256',
        'generator_selector_version','source_authored_caster_seed_policy','shared_executor_version',
        'shared_controller_config','shared_command_limits','shared_guards','shared_acceptance')
    first=plans[0];pool=None;by_method={};missing={};mismatches={};records=[]
    for plan in plans:
        method=plan['method'];actions=plan['selected_actions'];evaluations=plan['native_candidate_evaluations']
        ids=[row['candidate_id'] for row in evaluations]
        if (len(ids)!=len(set(ids)) or len(ids)!=plan['shared_candidate_count']
                or plan['all_pool_candidates_considered'] is not True
                or any(type(row.get('admitted_by_native_model')) is not bool for row in evaluations)
                or plan['eligible_candidate_count']!=sum(row['admitted_by_native_model'] for row in evaluations)
                or not actions or len(actions)!=len(plan['selected_states'])
                or plan.get('action_count',len(actions))!=len(actions)):
            raise ValueError('Native pool coverage/admission/action correspondence inconsistent: '+method)
        candidates={row['candidate_id']:row for row in evaluations};by_method[method]=candidates
        if pool is None:pool=set(ids)
        elif pool!=set(ids):raise ValueError('Methods did not evaluate the identical candidate identities')
        if (plan['selected_candidate_id'] not in candidates
                or candidates[plan['selected_candidate_id']]['admitted_by_native_model'] is not True):
            raise ValueError('Native selected candidate is not admitted by its own model')
        missing[method]=[key for key in controls if plan.get(key) is None]
        mismatches[method]=[key for key in controls if plan.get(key)!=first.get(key)]
        segments=[]
        for index,action in enumerate(actions):
            command={key:action[key] for key in ('mode','curvature_1_m','direction','target_speed_m_s')}
            if segments and segments[-1]['command']==command:segments[-1]['step_count']+=1
            else:segments.append(dict(first_action_index=index,step_count=1,command=command))
        predicates={}
        for key in ('sampled_full_envelope_clear','original_force_load_actuator_limits','stopping_reachability'):
            values=[row.get(key) for row in evaluations]
            if any(value is not None and type(value) is not bool for value in values):
                raise ValueError('Non-boolean native predicate: '+key)
            predicates[key]=dict(pass_count=sum(value is True for value in values),
                fail_count=sum(value is False for value in values),
                not_recorded_count=sum(value is None for value in values))
        state_ranges={}
        for key in ('friction_utilization','minimum_normal_load_fraction','abs_ltr','actuator_utilization'):
            values=[state[key] for state in plan['selected_states'] if state.get(key) is not None]
            if any(not isinstance(value,(float,int)) or not math.isfinite(value) for value in values):
                raise ValueError('Non-finite native selected-state metric: '+key)
            state_ranges[key]=dict(recorded_count=len(values),minimum=min(values) if values else None,
                maximum=max(values) if values else None)
        records.append(dict(method=method,planner_dynamics=plan['planner_dynamics'],
            initial_steering_propagated=plan['initial_steering_propagated'],
            selected_candidate_id=plan['selected_candidate_id'],eligible_candidate_count=plan['eligible_candidate_count'],
            total_candidates_evaluated=len(ids),selected_mode_action_counts=dict(Counter(a['mode'] for a in actions)),
            planned_mode_switches=sum(a['mode']!=b['mode'] for a,b in zip(actions,actions[1:])),
            planned_direction_reversals=sum(a['direction']!=b['direction'] for a,b in zip(actions,actions[1:])),
            command_segments=segments,reference_center_arc_m=plan.get('reference_center_arc_m'),
            native_cost_s_NOT_comparable_between_models=plan.get('native_cost_s'),
            selected_state_native_MODEL_ranges=state_ranges,independent_predicate_counts=predicates,
            native_selector_wall_s=plan.get('selection_wall_time_s'),
            shared_generation_charged_wall_s=plan.get('total_shared_generation_wall_s_charged_to_this_method'),
            generation_plus_selector_wall_s=plan.get('total_generation_plus_native_selection_wall_s')))
    contrasts=[]
    for plan in plans:
        candidate=plan['selected_candidate_id'];evaluators={}
        for method in METHODS:
            observation=by_method[method][candidate]
            evaluators[method]={key:observation.get(key) for key in ('admitted_by_native_model',
                'position_error_m','heading_error_deg','terminal_speed_m_s',
                'sampled_full_envelope_clear','original_force_load_actuator_limits','stopping_reachability')}
        contrasts.append(dict(selected_by_method=plan['method'],candidate_id=candidate,native_evaluators=evaluators))
    return dict(status='FROZEN_NATIVE_MODEL_MECHANISM_REVIEW_NOT_ACTUAL_PERFORMANCE',
        methods=records,selected_candidate_cross_model_observations=contrasts,
        common_control_mismatches={k:v for k,v in mismatches.items() if v},
        missing_common_controls={k:v for k,v in missing.items() if v},
        same_candidate_identities_evaluated_by_all_four_models=True,
        native_models_are_NOT_an_independent_common_model_comparison=True,
        predicate_failure_counts_overlap_NOT_disjoint_or_isolated_causal_attribution=True,
        zero_force_or_stop_rejections_is_NOT_proof_of_causal_safety_improvement=True,
        selected_state_constraint_saturation_is_NOT_a_rejected_candidate=True,
        predicted_terminal_speed_None_is_NOT_stopping_evidence=True,
        terminal_controller_rescue_must_be_analyzed_separately_in_actual_trace=True,
        native_costs_NOT_ranked_as_executed_time=True,
        model_predictions_NOT_actual_measured_response_validation=True,
        actual_drive_admission=False,statistical_superiority_established=False)


def frozen_h4_resource_review(root):
    """Read saved MODEL timing, never expand a live or method-specific budget.

    The archived full-route resource contract is inspected without importing
    an executor or rerunning its model. Nominal software margin is NOT a
    measured physical runtime guarantee or permission to reuse a moved start.
    """
    root=Path(root)
    bundle_path=_ijcas_path(root, 'hospital_h4_execution_bundle/bundle_r01/execution_bundle.json')
    model_path=_ijcas_path(root, 'hospital_h4_common_executor_screen/screen_r01/common_executor_model_screen.json')
    if not bundle_path.is_file() or not model_path.is_file():
        return dict(status='FROZEN_H4_INPUTS_NOT_AVAILABLE',actual_drive_admission=False),{}
    frozen=json.loads(bundle_path.read_bytes());model=json.loads(model_path.read_bytes())
    pins={str(path):file_sha(path) for path in (bundle_path,model_path)}
    # Read the common limits already used in the original execution report;
    # absence is unknown, not an invitation to invent a more generous budget.
    prior_path=_ijcas_path(root, 'hospital_h4_path_execution_v3/cohort_01_adopted_query_r01/hospital_h4_shared_v3_P_dynamics_full_envelope_r01.json')
    if not prior_path.is_file():
        return dict(status='COMMON_EXECUTED_RESOURCE_CONTRACT_NOT_AVAILABLE',actual_drive_admission=False),pins
    prior=json.loads(prior_path.read_bytes());pins[str(prior_path)]=file_sha(prior_path)
    guards=prior['guards'];acceptance=prior['acceptance'];config=prior['terminal_controller_config']
    limit=float(guards['maximum_simulation_duration_s']);hold=float(acceptance['hold_observation_s'])
    if not math.isfinite(limit) or not math.isfinite(hold) or limit<=hold or hold<=0:
        raise ValueError('Finite original full-H4 resource contract required')
    cases=[];seen=set()
    expected={(method,dt) for method in METHODS for dt in (.02,.01)}
    for case in model['cases']:
        identity=(case['method'],case['integration_dt_s'])
        if identity not in expected or identity in seen:
            raise ValueError('Exactly the original four methods/two MODEL resolutions required')
        seen.add(identity)
        duration=float(case['simulation_duration_s']);entry=case['hold_entry_model_time_s']
        if not math.isfinite(duration) or duration<0:
            raise ValueError('Finite saved MODEL timing required')
        trace=Path(case['model_trace'])
        if not trace.is_file() or file_sha(trace)!=case['model_trace_sha256']:
            raise ValueError('Original H4 MODEL trace missing or changed: '+str(trace))
        pins[str(trace)]=case['model_trace_sha256']
        cases.append(dict(method=identity[0],integration_dt_s=identity[1],
            original_model_status=case['model_status'],model_total_duration_s=duration,
            model_fixed_hold_entry_s=entry,model_remaining_whole_trial_budget_s=limit-duration,
            original_terminal_pulse_count=case['terminal_pulse_count']))
    if seen!=expected:raise ValueError('Missing original H4 MODEL resolution/method case')
    plans=[];native_plans=[]
    for path in map(Path,frozen['plans']):
        if file_sha(path)!=frozen['output_sha256'].get(str(path)):
            raise ValueError('Original H4 selected plan changed: '+str(path))
        plan=json.loads(path.read_bytes());pins[str(path)]=file_sha(path);native_plans.append(plan)
        plans.append(dict(method=plan['method'],reference_center_arc_m=plan['reference_center_arc_m'],
            command_speed_cap_m_s=plan['common_max_command_speed_m_s'],
            command_acceleration_cap_m_s2=plan['common_max_acceleration_m_s2']))
    if len(plans)!=4 or {p['method'] for p in plans}!=set(METHODS):
        raise ValueError('All four original H4 plans required, not just a successful method')
    resource_option=dict(status='FULL_NOMINAL_RESOURCE_ENVELOPE_UNAVAILABLE_NOT_INVENTED')
    original_protocol=Path(prior['path_protocol']) if prior.get('path_protocol') else None
    if original_protocol is not None:
        if not original_protocol.is_file() or file_sha(original_protocol)!=prior.get('path_protocol_sha256'):
            raise ValueError('Original common resource protocol is missing or changed')
        definition=json.loads(original_protocol.read_bytes());pins[str(original_protocol)]=file_sha(original_protocol)
        startup=definition['initial_pre_hold_s']
        schedules=[p.get('nominal_longitudinal_schedule_duration_s') for p in native_plans]
        if all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in (startup,*schedules)):
            bounds=[dict(method=p['method'],nominal_schedule_s=nominal,
                common_prehold_s=startup,common_terminal_timeout_s=config['terminal_timeout_s'],
                required_fixed_hold_s=hold,
                sum_of_declared_components_s=startup+nominal+config['terminal_timeout_s']+hold)
                for p,nominal in zip(native_plans,schedules)]
            maximum=max(row['sum_of_declared_components_s'] for row in bounds)
            resource_option=dict(status='PROSPECTIVE_COMMON_RESOURCE_OPTION_REQUIRES_EXPLICIT_APPROVAL',
                original_budget_s=limit,declared_component_envelopes=bounds,
                maximum_nominal_component_envelope_s=maximum,
                proposed_common_simulation_budget_s=60.*math.ceil(maximum/60.),
                derivation='Round up the largest ALL-FOUR frozen nominal schedule + common prehold + common terminal timeout + required hold to a whole minute.',
                proposed_common_wall_budget_s=guards['maximum_wall_time_s'],
                apply_same_contract_to_all_four_methods_before_new_cohort=True,
                nominal_sum_NOT_a_physical_worst_case_or_success_guarantee=True,
                existing_180s_FAILs_preserved_NOT_relabelled_or_pooled=True,
                changed_resource_contract_is_a_new_development_version_NOT_same_trial_completion=True,
                exact_old_endpoint_must_NOT_be_resumed_or_time_appended=True,
                new_live_state_archive_Stop_fresh_start_static_and_motion_need_explicit_authority=True,
                actual_motion_authorized=False,protocol_or_runtime_modified=False)
    return dict(status='ARCHIVED_COMMON_RESOURCE_REVIEW_ONLY_NOT_NEW_DRIVE_ADMISSION',
        common_simulation_budget_s=limit,common_wall_budget_s=guards['maximum_wall_time_s'],
        common_terminal_controller_timeout_s=config['terminal_timeout_s'],fixed_hold_s=hold,
        whole_trial_budget_includes_tracking_terminal_and_hold=True,
        terminal_timeout_is_NOT_an_additional_budget_after_tracking=True,
        minimum_nominal_MODEL_whole_trial_margin_s=min(c['model_remaining_whole_trial_budget_s'] for c in cases),
        cases=cases,frozen_plans=plans,
        native_planning_mechanism=native_planning_mechanism_review(native_plans),
        prospective_common_resource_option=resource_option,
        MODEL_resolution_cases_NOT_independent_physical_repeats=True,
        physical_runtime_safety_or_success_NOT_predicted_by_nominal_margin=True,
        no_method_specific_budget_increase_or_retuning=True,
        original_environment_route_FAIL_NOT_pooled_with_subfloor_variant=True,
        completed_initial_component_changes_entry_requires_new_cohort_for_full_task=True,
        no_automatic_continuation_from_moved_initial_component_endpoint=True,
        actual_drive_admission=False),pins


def configured_h4_resource_inventory(root,records):
    """Keep configured fresh resource versions separate from actual outcomes.

    A four-method declaration is not four prepared methods, a live task,
    permission, or measured performance. Never pool 180s and 300s outcomes.
    This reads saved JSON only, without loading or changing a live executor.
    """
    root=Path(root);versions=[];pins={};unreadable=[]
    for path in sorted(root.glob('hospital_h4_experiments/*/protocol.json')):
        try:
            if path.stat().st_size>4*1024*1024:raise ValueError('Oversized protocol is not read')
            data=json.loads(path.read_bytes())
            if data.get('version')!='H4_configured_fresh_single_whole_task_v1':continue
            pins[str(path)]=file_sha(path)
            guards=data['guards'];limits={key:guards.get(key) for key in
                ('maximum_simulation_duration_s','maximum_wall_time_s','maximum_control_gap_s')}
            errors=[]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in limits.values()):
                raise ValueError('invalid_declared_resource_limits')
            contract=data.get('common_resource_contract');common_verified=False
            if contract is not None:
                if not isinstance(contract,dict):
                    errors.append('common_resource_contract_is_not_an_object')
                else:
                    methods=contract.get('methods',[]);schedules=contract.get('nominal_action_schedule_s',{})
                    if not isinstance(methods,list) or len(methods)!=4 or set(methods)!=set(METHODS):
                        errors.append('all_four_methods_not_declared_exactly_once')
                    if contract.get('common_to_all_methods_NOT_P_specific') is not True:
                        errors.append('method_blind_resource_contract_not_declared')
                    for key in ('maximum_simulation_duration_s','maximum_wall_time_s'):
                        if contract.get(key)!=limits[key]:errors.append('contract_guard_mismatch:'+key)
                    components=(contract.get('initial_pre_hold_s'),contract.get('common_terminal_allocation_s'),
                        contract.get('fixed_terminal_hold_s'),contract.get('round_up_to_whole_minute_s'))
                    valid_components=all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in components)
                    valid_schedules=(isinstance(schedules,dict) and set(schedules)==set(METHODS)
                        and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in schedules.values()))
                    if not valid_components or not valid_schedules:
                        errors.append('incomplete_or_nonfinite_common_resource_derivation')
                    else:
                        startup,terminal,hold,rounding=components
                        envelope=max(schedules.values())+startup+terminal+hold
                        if envelope!=contract.get('largest_nominal_plus_common_allocations_s'):
                            errors.append('declared_nominal_component_envelope_mismatch')
                        if rounding*math.ceil(envelope/rounding)!=limits['maximum_simulation_duration_s']:
                            errors.append('rounded_common_simulation_limit_mismatch')
                        if terminal!=data.get('terminal_controller_config',{}).get('terminal_timeout_s'):
                            errors.append('common_terminal_allowance_mismatch')
                        if hold!=data.get('acceptance',{}).get('hold_observation_s'):
                            errors.append('required_fixed_hold_allowance_mismatch')
                    common_verified=not errors
            rows=[row for row in records if row['evidence_role']=='METHOD_PATH_DEVELOPMENT'
                and row['execution_cohort']==data.get('cohort')
                and Path(row['result']).is_relative_to(path.parent/'route')]
            mismatches={}
            for row in rows:
                keys=[key for key in ('guards','command_limits','acceptance','terminal_controller_config')
                    if row['comparison_controls'].get(key)!=data.get(key)]
                if row['environment_variant']!=data.get('environment_variant'):keys.append('environment_variant')
                if keys:mismatches[row['run_id']]=keys
            actual=[row for row in rows if row['outcome_stage']=='ACTUAL_PATH_OUTCOME']
            counts={method:sum(row['method']==method for row in actual) for method in METHODS}
            versions.append(dict(protocol=str(path),cohort=data.get('cohort'),configured_method=data.get('method'),
                environment_variant=data.get('environment_variant'),declared_resource_limits=limits,
                common_resource_version=contract.get('version') if isinstance(contract,dict) else None,
                all_four_common_resource_declaration_consistent=common_verified,
                declaration_integrity_errors=errors,actual_result_contract_mismatches=mismatches,
                actual_path_outcome_counts=counts,
                actual_path_outcomes_by_method={method:[dict(run_id=row['run_id'],verdict=row['reported_verdict'])
                    for row in actual if row['method']==method] for method in METHODS},
                methods_without_actual_path_outcome=[method for method in METHODS if counts[method]==0],
                setup_or_prepath_attempts_NOT_performance_outcomes=[row['run_id'] for row in rows
                    if row['outcome_stage']!='ACTUAL_PATH_OUTCOME'],
                declaration_does_NOT_establish_prepared_or_executed_all_four_methods=True,
                version_is_NOT_pooled_with_different_resource_contracts=True,
                live_task_status_or_current_state_NOT_inferred=True,actual_motion_authorized=False,
                final_comparison_qualification=False))
        except (ValueError,TypeError,KeyError,OSError,OverflowError) as error:
            unreadable.append(dict(protocol=str(path),error=str(error),physical_outcome_inferred=False))
    return dict(status='SAVED_FRESH_RESOURCE_VERSIONS_NOT_LIVE_STATE_OR_FINAL_COMPARISON',
        configured_versions=versions,unreadable_protocols=unreadable,
        historical_resource_option_is_NOT_current_execution_limit=True,
        runtime_protocol_or_original_results_modified=False,actual_motion_authorized=False),pins


def build_execution_ledger(root=RESULT):
    root=Path(root);paths={}
    patterns=(('path_execution*/**/hospital_h1_*path_r*.json','method_path'),
        ('hospital_h2_path_execution*/**/hospital_h2_*path_r*.json','method_path'),
        ('hospital_h4_path_execution*/**/hospital_h4_*_r*.json','method_path'),
        ('hospital_h4_experiments/*/route/**/hospital_h4_*_r*.json','method_path'),
        ('hospital_h4_experiments/*/static/hospital_h4_fresh_static_r*.json','static'),
        ('hospital_h4_static_contact/**/hospital_h4_*_r*.json','static'),
        ('hospital_h4_initial_motion/**/hospital_h4_initial_motion_r*.json','initial_motion'),
        ('hospital_h3b_prefix_response_observation/cohort_03/*/hospital_h3b_*_r01.json','segmented_component'))
    for pattern,role in patterns:
        for path in root.glob(pattern):paths[path]=role
    records=[];pins={};unreadable=[]
    for path,role in sorted(paths.items()):
        try:
            row=inspect_execution_record(path,role);records.append(row);pins[str(path)]=file_sha(path)
            if row['trace']:pins[row['trace']]=file_sha(row['trace'])
            envelope=row['sampled_authored_envelope_observations']
            if envelope:pins.update(envelope.get('input_sha256',{}))
        except (ValueError,KeyError,OSError) as error:
            # An incomplete write is NOT a terminal process/physical failure.
            unreadable.append(dict(path=str(path),status='UNREADABLE_OR_INCOMPLETE_NOT_A_PHYSICAL_VERDICT',error=str(error)))
    pending=[]
    # Record an orphaned commissioning CSV rather than silently omitting the
    # completed physical trace or calling its reservation a motion success.
    for trace in sorted(root.glob('hospital_h4_initial_motion/**/hospital_h4_initial_motion_r*.csv')):
        if trace.with_suffix('.json').exists():continue
        pending.append(dict(trace=str(trace),status='CSV_PRESENT_RESULT_NOT_FINALIZED',
            reported_verdict=None,observations=trace_metrics(trace),new_trial_or_success_inferred=False))
        pins[str(trace)]=file_sha(trace)
    resources,resource_pins=frozen_h4_resource_review(root)
    pins.update(resource_pins)
    configured,configured_pins=configured_h4_resource_inventory(root,records)
    pins.update(configured_pins)
    return dict(version='hospital_execution_scope_ledger_v1',
        status='SAVED_EVIDENCE_SCOPE_ACCOUNTING_NOT_LIVE_MONITOR_OR_FINAL_COMPARISON',
        audited_scope='selected H1/H2/H4 path cohorts, H4 static/initial commissioning and H3 cohort03 response components; NOT all historical experiments',
        records=records,counts_by_role=dict(Counter(r['evidence_role'] for r in records)),
        method_artifact_identity_review=execution_artifact_identity_review(records,pins),
        development_groups=development_comparison_groups(records),pending_finalization=pending,unreadable_records=unreadable,
        frozen_H4_common_resource_review=resources,
        configured_H4_fresh_resource_versions=configured,
        claimed_final_records_requiring_separate_frozen_contract_audit=[r['result'] for r in records if r['reported_final_statistics_flag']],
        qualified_final_cells_proven_by_this_audit=0,
        qualification_reason='No pilot/setup flag promotes a trial; frozen study, independent paired histories and matched conditions require a separate final-cohort audit.',
        engineering_or_serialization_failures_NOT_algorithmic_performance_limitations=True,
        actual_execution_failures_preserved_in_their_own_environment_controller_cohorts=True,
        deterministic_replays_and_telemetry_samples_NOT_independent_robustness_trials=True,
        no_mandatory_hardware_or_CICS_runtime_added=True,actual_motion_authorized=False,
        original_raw_files_modified=False,input_sha256=pins)


def write_execution_ledger(root=RESULT):
    """Content-addressed derived snapshots; unchanged inputs create no clones."""
    from hospital_runtime_evidence_io import write_exclusive
    record=build_execution_ledger(root)
    record['analysis_source_sha256']={str(Path(__file__)):file_sha(Path(__file__))}
    payload=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    identity=hashlib.sha256(payload).hexdigest()[:16]
    output=Path(root)/'paper_evidence_inventory/execution_ledger'
    target=output/('execution_ledger_'+identity+'.json')
    if target.exists():
        if json.loads(target.read_bytes())!=record:raise ValueError('Existing ledger snapshot differs; no overwrite')
    else:write_exclusive(target,record)
    print('Saved execution ledger:',target)
    print('Evidence roles:',record['counts_by_role'])
    print('CSV without finalized result:',len(record['pending_finalization']))
    print('Qualified final comparison cells proven here: 0; no motion or raw-result change.')
    return target


def build_full_scope_inventory(root=RESULT):
    """Six-task saved-artifact coverage, never a final-cohort or live-state gate.

    Bounded discovery includes earlier FAIL/HOLDs and receipt copies. Do not
    choose a latest/best run, pool changed planners, count CSV rows as repeats,
    or infer that an absent report means a live task has stopped. The older
    setup inventory and execution-ledger snapshots remain untouched.
    """
    root=Path(root).resolve();pins={};issues=[];paths={};records=[]
    tasks=('H1_bedside_departure','H2_bedside_lateral_alignment',
        'H3_doorway_immediate_turn','H4_ward_corridor_turn',
        'H5_obstacle_bypass_return','H6_integrated_transfer_mission')
    patterns=(
        ('path_execution*/**/hospital_h1_*path_r*.json','method_path'),
        ('hospital_h2_path_execution*/**/hospital_h2_*path_r*.json','method_path'),
        ('hospital_h3_path_execution*/**/hospital_h3_*_r*.json','method_path'),
        ('hospital_h4_path_execution*/**/hospital_h4_*_r*.json','method_path'),
        ('hospital_h4_experiments/*/route/**/hospital_h4_*_r*.json','method_path'),
        ('hospital_h5_experiments/*/route/**/hospital_h5_*_r*.json','method_path'),
        ('hospital_h6_experiments/*/route/**/hospital_h6_*_r*.json','method_path'),
        ('hospital_h4_experiments/*/static/hospital_h4_fresh_static_r*.json','static'),
        ('hospital_h4_static_contact/**/hospital_h4_*_r*.json','static'),
        ('hospital_h4_initial_motion/**/hospital_h4_initial_motion_r*.json','initial_motion'),
        ('hospital_h5_experiments/*/static/hospital_h5_*static_r*.json','static'),
        ('hospital_h6_experiments/*/**/hospital_h6_*static_r*.json','static'),
        ('hospital_h3b_prefix_response_observation/cohort_03/*/hospital_h3b_*_r01.json','segmented_component'))

    def load(path):
        path=Path(path)
        if path.stat().st_size>4*1024*1024:raise ValueError('No giant scene archives in scope inventory')
        payload=path.read_bytes();pins[str(path)]=hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def failure(path,error):
        issues.append(dict(path=str(path),error=str(error),
            status='UNREADABLE_OR_INCONSISTENT_SAVED_ARTIFACT_NOT_A_NEW_PHYSICAL_VERDICT'))

    for pattern,role in patterns:
        for path in root.glob(pattern):paths[path]=role
    for path,role in sorted(paths.items()):
        try:
            # Pin before and after reading: never silently accept a changed file.
            before=file_sha(path);trace=path.with_suffix('.csv')
            trace_before=file_sha(trace) if trace.exists() else None
            row=inspect_execution_record(path,role,include_envelope_observations=False)
            if file_sha(path)!=before:raise ValueError('Result changed during saved-artifact inspection')
            trace_after=file_sha(trace) if trace.exists() else None
            if trace_after!=trace_before:raise ValueError('Trace changed during saved-artifact inspection')
            records.append(row);pins[str(path)]=before
            if row['trace']:pins[row['trace']]=trace_after
        except (ValueError,KeyError,OSError,TypeError) as error:failure(path,error)

    # Zero-command engineering reports have an explicit trace path and no run_id;
    # do not invent a trial identity or mistake them for H6 integrated driving.
    zero=[]
    for path in sorted(root.glob('hospital_h6_experiments/*/retained_zero_command_settling*/**/zero_settling_readback.json')):
        try:
            raw=load(path)
            if raw.get('verdict') not in ('PASS','FAIL','HOLD'):raise ValueError('Not a zero-command outcome')
            trace=Path(raw['trace']) if raw.get('trace') else None;metrics=None;integrity=[]
            if trace is not None:
                if not trace.resolve().is_relative_to(root):raise ValueError('Engineering trace outside declared evidence root')
                digest=file_sha(trace);metrics=trace_metrics(trace)
                if file_sha(trace)!=digest:raise ValueError('Zero-command trace changed during inspection')
                pins[str(trace)]=digest
                if digest!=raw.get('trace_sha256'):integrity.append('original_trace_hash_mismatch_or_absent')
                if metrics['sample_count']!=raw.get('sample_count'):integrity.append('raw_sample_count_differs_from_CSV')
                integrity.extend(metrics['CSV_structure_errors'])
            elif raw.get('sample_count',0):integrity.append('nonzero_samples_without_trace')
            zero.append(dict(result=str(path),reported_verdict=raw['verdict'],
                evidence_role='ENGINEERING_ZERO_COMMAND_NOT_INTEGRATED_ROUTE',
                sample_count=raw.get('sample_count'),trace=str(trace) if trace else None,
                trace_observations=metrics,integrity_errors=integrity,
                failed_checks=[k for k,v in raw.get('checks',{}).items() if v is not True],
                reported_error=raw.get('error'),new_trial_identity_or_route_success_inferred=False))
        except (ValueError,KeyError,OSError,TypeError) as error:failure(path,error)

    # Explicit historical segmented closeout, not an automatically selected plan.
    h3path=_ijcas_path(root, 'hospital_h3b_prefix_response_observation/response_completion/closeout_r01/response_completion_audit.json')
    h3=dict(status='SAVED_SEGMENTED_CLOSEOUT_NOT_FOUND_IN_THIS_ROOT',whole_mission_success_inferred=False)
    if h3path.exists():
        try:
            raw=load(h3path);synthesis=raw['response_synthesis'];components=[]
            for item in synthesis['components']:
                path=Path(item['result']);component=load(path)
                if pins[str(path)]!=item['result_sha256'] or component['verdict']!=item['original_motion_verdict']:
                    raise ValueError('Segmented closeout/raw component identity or verdict mismatch')
                components.append(dict(component=item['component'],result=str(path),reported_verdict=component['verdict']))
            h3=dict(status='SAVED_SEGMENTED_COMPONENTS_NOT_FRESH_CONTINUOUS_WHOLE_MISSION',
                result=str(h3path),components=components,
                reported_all_four_saved_handoffs_verified=synthesis['four_actual_saved_body_and_twelve_joint_handoffs_verified'],
                reported_fresh_whole_sequence_repeats=synthesis['independent_fresh_frozen_whole_sequence_repeats'],
                reported_MODEL_PASS_count=synthesis['whole_H3_original_model_PASS_count'],
                reported_MODEL_case_count=synthesis['whole_H3_original_model_case_count'],
                MODEL_cases_and_actual_components_NOT_pooled=True,whole_mission_success_inferred=False)
        except (ValueError,KeyError,OSError,TypeError) as error:failure(h3path,error)

    # Name the current OFFLINE planning revision explicitly; no best/latest
    # search or claim that all historical planning attempts were re-audited.
    h6path=_ijcas_path(root, 'hospital_h6_planning/heading0_fresh_terminal_beam_matched_dev_r19/planning_review.json')
    h6=dict(status='DECLARED_r19_OFFLINE_PLANNING_REVIEW_NOT_FOUND_IN_THIS_ROOT',methods=[])
    if h6path.exists():
        try:
            raw=load(h6path);rows=[]
            if len(raw['methods'])!=4 or {item['method'] for item in raw['methods']}!=set(METHODS):
                raise ValueError('Exactly four declared planning methods required')
            for item in raw['methods']:
                path=Path(item['plan_file']);plan=load(path)
                expected=raw['output_sha256'].get(str(path))
                if (pins[str(path)]!=expected or plan['method']!=item['method']
                        or plan['plan_success']!=item['plan_success']):
                    raise ValueError('Declared offline plan hash/identity/outcome mismatch')
                auditpath=path.with_name(path.stem+'_independent_audit.json');audit=load(auditpath)
                if (pins[str(auditpath)]!=raw['output_sha256'].get(str(auditpath))
                        or audit['plan_sha256']!=pins[str(path)] or Path(audit['plan'])!=path
                        or audit['verdict']!=item['independent_OFFLINE_audit']):
                    raise ValueError('Declared independent OFFLINE audit hash/plan/verdict mismatch')
                rows.append({k:item[k] for k in ('method','plan_success','planning_time_s','path_length_m',
                    'action_count','mode_switch_count','independent_OFFLINE_audit','accepted_offline_start_goal_plan')})
            h6=dict(status='EXPLICIT_r19_OFFLINE_PLANS_NOT_ACTUAL_INTEGRATED_DRIVE',result=str(h6path),methods=rows,
                historical_search_attempts_NOT_exhaustively_reaudited=True,
                internal_P_generators_NOT_independent_comparison_trials=True,actual_drive_success_inferred=False)
        except (ValueError,KeyError,OSError,TypeError) as error:failure(h6path,error)

    cells=[]
    for task in tasks:
        for method in METHODS:
            rows=[r for r in records if r['evidence_role']=='METHOD_PATH_DEVELOPMENT'
                and r['scenario']==task and r['method']==method]
            actual=[r for r in rows if r['outcome_stage']=='ACTUAL_PATH_OUTCOME']
            cells.append(dict(task=task,method=method,
                saved_path_outcome_artifacts=[dict(result=r['result'],run_id=r['run_id'],
                    reported_verdict=r['reported_verdict'],execution_version=r['execution_version'],
                    execution_cohort=r['execution_cohort'],environment_variant=r['environment_variant'],
                    trace_pin_status=r['trace_pin_status'],integrity_errors=r['integrity_errors']) for r in actual],
                setup_or_prepath_artifacts=[r['result'] for r in rows if r not in actual],
                saved_actual_outcome_presence_NOT_independent_repeats=bool(actual),
                final_matched_cell_qualification=False))
    for path,digest in pins.items():
        if file_sha(path)!=digest:raise ValueError('Inventory input changed during inspection: '+path)
    return dict(version='six_task_saved_evidence_scope_inventory_v1',
        status='DESCRIPTIVE_DEVELOPMENT_COVERAGE_NOT_FINAL_REPETITIONS_OR_LIVE_STATE',
        audited_root=str(root),discovery_patterns=[dict(pattern=p,role=r) for p,r in patterns],
        historical_scope_limit='Declared bounded raw-route patterns, H6 zero-command reports, one named H3 closeout and H6 r19 planning revision; NOT every historical planning/MODEL/sensor artifact.',
        tasks=list(tasks),methods=list(METHODS),cells=cells,records=records,
        counts_by_role=dict(Counter(r['evidence_role'] for r in records)),
        development_groups=development_comparison_groups(records),
        artifact_identity_review=execution_artifact_identity_review(records,pins),
        H3_segmented_closeout=h3,H6_declared_offline_planning=h6,H6_zero_command_engineering=zero,
        unresolved_saved_artifacts=issues,
        required_evidence_NOT_certified_by_this_inventory=[
            'Fresh continuous whole-sequence H3 without intermediate state reset',
            'H5 declared replanning event/observation/compute/stop-resume/completion beyond static-cart bypass',
            'H6 actual continuous integrated transfer with shared bidirectional executor',
            'Hospital sensor/localization/Nav2 H1-H4 and H6 integrated cases',
            'Frozen reduced matched design with independent predeclared entry histories',
            'Mechanism ablations and condition sensitivity; no multi-factor attribution',
            'Author-reviewed Korean manuscript/figures, then same-content English and IJCAS layout'],
        engineering_errors_NOT_algorithmic_performance_limits=True,
        no_automatic_latest_best_or_success_selection=True,
        file_absence_NOT_live_state_or_process_completion_evidence=True,
        final_matched_cells_proven_by_this_inventory=0,
        remaining_drive_count_or_53_cell_protocol_adoption_inferred=False,
        actual_motion_authorized=False,original_results_modified=False,input_sha256=pins)


def full_scope_summary_ko(record):
    """Small author-facing summary, explicitly counting artifacts not repeats."""
    lines=['H1–H6 저장 근거 현황 — 개발 판정파일과 최종 검증의 구분',
        '', '아래는 선언된 검색 범위의 저장 파일 목록이다. 독립 실험 횟수·성공률·최종 반복 통계가 아니다.',
        '이전 실패·준비 HOLD·바뀐 제어/환경/계획 버전과 복사된 확인용 영수증도 삭제하지 않는다.',
        '날짜순 정렬이나 최신/최고 결과 선택을 하지 않는다. 방법별 비교는 JSON의 조건별 그룹과 원자료를 함께 확인한다.',
        '', '과제 | B0 | B1 | B2 | P (각 숫자는 실제 경로 결과가 기록된 판정파일 수)']
    for task in record['tasks']:
        labels=[]
        for cell in [c for c in record['cells'] if c['task']==task]:
            counts=Counter(r['reported_verdict'] for r in cell['saved_path_outcome_artifacts'])
            labels.append('·'.join(f'{v}×{counts[v]}' for v in ('PASS','FAIL','HOLD') if counts[v]) or '해당 범위의 주행 결과 없음')
        lines.append(task.split('_')[0]+' | '+' | '.join(labels))
    h3=record['H3_segmented_closeout'];h6=record['H6_declared_offline_planning']
    lines.extend(['', 'H3: 지정한 과거 closeout의 분할 구간 근거는 전체 연속 임무 검증으로 대체하지 않는다.',
        'H3 분할 구간 저장 판정: '+', '.join(c['component']+'='+c['reported_verdict'] for c in h3.get('components',[])),
        'H6: 지정한 r19의 네 방법 오프라인 경로·독립 OFFLINE 감사는 실제 통합 주행 성공이 아니다.',
        'H6 r19 저장 계획 판정: '+', '.join(c['method'].split('_')[0]+'='+str(c['plan_success']) for c in h6['methods']),
        'H6의 정적 지지·0명령 관측은 별도 공학 자료다. HOLD나 정적 PASS를 주행 성공에 합산하지 않는다.',
        'H5 고정 카트 우회 결과는 선언한 환경 변경 이벤트의 온라인 재계획 검증이 아니다.',
        '센서 지도·TF·메시지 SOFTWARE 검사는 실제 병원 센서/위치추정/Nav2 성공이 아니다.',
        '', '확인용 중복 영수증 그룹: '+str(len(record['artifact_identity_review']['repeated_run_id_artifacts'])),
        '동일 CSV 바이트 그룹: '+str(len(record['artifact_identity_review']['identical_trace_byte_groups'])),
        '서로 다른 run ID라도 같은 CSV만으로 중복 물리 시험 또는 독립 반복을 증명할 수 없다.',
        '읽기 불가/불일치 저장 자료: '+str(len(record['unresolved_saved_artifacts'])),
        '이 목록이 입증한 최종 비교 셀: 0. 48+5=53은 여전히 미채택 계획안이며 남은 주행 횟수가 아니다.',
        '', '이후에도 필요한 범위:',*(' - '+item for item in record['required_evidence_NOT_certified_by_this_inventory']),
        '', '현재 살아 있는 Isaac 작업/타임라인은 조회하지 않았다. 파일 부재로 미실행·중지·완료를 단정하지 않는다.',
        '원본 결과·기존 원고·실행 명령을 변경하지 않았다. 새 주행이나 자동 재시도를 승인하지 않는다.', ''])
    return '\n'.join(lines)


def write_full_scope_inventory(root=RESULT):
    """One content-addressed JSON snapshot; no source/result replacement."""
    from hospital_runtime_evidence_io import write_exclusive
    record=build_full_scope_inventory(root)
    record['analysis_source_sha256']={str(Path(__file__).resolve()):file_sha(__file__)}
    payload=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    identity=hashlib.sha256(payload).hexdigest()[:16]
    target=Path(root)/'paper_evidence_inventory/full_scope'/('scope_inventory_'+identity+'.json')
    if target.exists():
        if json.loads(target.read_bytes())!=record:raise ValueError('Existing full-scope inventory differs; no overwrite')
    else:write_exclusive(target,record)
    summary=target.with_name('scope_summary_'+identity+'_ko.txt');payload=full_scope_summary_ko(record).encode('utf-8')
    if summary.exists():
        if summary.read_bytes()!=payload:raise ValueError('Existing Korean scope summary differs; no overwrite')
    else:
        with summary.open('xb') as stream:stream.write(payload)
    print('Saved six-task scope inventory:',target)
    print('Saved artifact roles (NOT independent trials):',record['counts_by_role'])
    print('Unresolved saved artifacts:',len(record['unresolved_saved_artifacts']))
    print('No live operation or final repetition qualification.')
    return target


def write_execution_figure(result,output=None):
    """Explicit one-trial measured figure; no nominal/selected-run statistics."""
    import inspect
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Rectangle
    from hospital_runtime_evidence_io import write_exclusive
    result=Path(result).resolve();raw=json.loads(result.read_bytes())
    record=inspect_execution_record(result,'method_path');metrics=record['trace_observations']
    if (raw.get('scenario')!='H4_ward_corridor_turn' or record['outcome_stage']!='ACTUAL_PATH_OUTCOME'
            or record['trace_pin_status']!='ORIGINAL_TRACE_HASH_VERIFIED' or record['integrity_errors']
            or metrics.get('sample_count',0)<2):
        raise ValueError('Explicit actual H4 outcome and intact measured trace required; no setup/nominal figure')
    plan_path=Path(raw['plan_path']);plan=json.loads(plan_path.read_bytes())
    geometry_path=Path(plan['geometry_manifest']);setup=json.loads(geometry_path.read_bytes())
    if (file_sha(plan_path)!=raw['plan_sha256'] or plan['method']!=raw['method']
            or file_sha(geometry_path)!=raw['geometry_manifest_sha256']
            or plan['geometry_manifest_sha256']!=raw['geometry_manifest_sha256']
            or plan['planning_occupancy_digest']!=raw['planning_occupancy_digest']
            or setup['footprint']!={'length_m':1.630,'width_m':.965,'clearance_m':.010}):
        raise ValueError('Same own frozen plan/map/full footprint required; no replacement reference')
    trace=Path(record['trace'])
    pins={str(p):file_sha(p) for p in (result,trace,plan_path,geometry_path)}
    definition=dict(version='one_measured_H4_development_trial_figure_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_execution_figure).encode()).hexdigest(),
        matplotlib_version=matplotlib.__version__)
    output=Path(output) if output is not None else result.parent/'offline_summary'
    manifest=output/'figure_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()):
            raise ValueError('Existing derived figure uses other inputs/exporter; no overwrite')
        if any(not Path(p).is_file() or file_sha(p)!=sha for p,sha in saved['output_sha256'].items()):
            raise ValueError('Existing derived figure changed or missing; no silent replacement')
        return manifest
    output.mkdir(parents=True,exist_ok=True)
    images=[output/'measured_execution.png',output/'measured_execution.pdf']
    if any(p.exists() for p in images):raise ValueError('Unfinalized figure output exists; no overwrite')
    with trace.open(newline='') as stream:rows=list(csv.DictReader(stream))
    def values(key):
        array=np.asarray([float(row[key]) for row in rows])
        if not np.isfinite(array).all():raise ValueError('Finite measured figure channel required: '+key)
        return array
    t=values('time_s');x=values('x_m');y=values('y_m');yaw=values('yaw_rad')
    command=1000*values('command_speed_m_s')
    u=1000*(values('vx_physx_m_s')*np.cos(yaw)+values('vy_physx_m_s')*np.sin(yaw))
    phases=np.asarray([row['phase'] for row in rows]);tracking=phases=='path_tracking'
    terminal=np.asarray([phase.startswith('terminal_') for phase in phases])
    if not tracking.any():raise ValueError('Actual tracking measurements required')
    position=1000*values('goal_position_error_m');heading=np.abs(values('goal_heading_error_deg'))
    track_position=1000*values('path_error_m');track_heading=np.abs(values('path_heading_error_deg'))
    reference=np.asarray([plan['start_world_xyyaw']]+[state['world_xyyaw'] for state in plan['selected_states']])
    start=np.asarray(plan['start_world_xyyaw']);goal=np.asarray(plan['goal_world_xyyaw'])
    style={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'figure.dpi':160}
    with plt.rc_context(style):
        fig,axes=plt.subplots(2,2,figsize=(12.8,8.7))
        map_ax,speed_ax,error_ax,terminal_ax=axes.flat
        lo=np.minimum(np.min(np.c_[x,y],axis=0),reference[:,:2].min(axis=0))-1.1
        hi=np.maximum(np.max(np.c_[x,y],axis=0),reference[:,:2].max(axis=0))+1.1
        for obstacle in setup['included_collision_AABBs']:
            bound=obstacle['world_aabb'];a=np.asarray(bound['minimum_m'][:2]);b=np.asarray(bound['maximum_m'][:2])
            if np.all(b>=lo) and np.all(a<=hi):
                map_ax.add_patch(Rectangle(a,*(b-a),facecolor='#d5d9dc',edgecolor='#9aa1a7',linewidth=.4))
        map_ax.plot(reference[:,0],reference[:,1],'--',color='#dc8b28',lw=1.5,label='Frozen planned reference')
        map_ax.plot(x,y,color='#087e91',lw=1.4,label='Measured base trajectory')
        map_ax.scatter(*start[:2],marker='o',color='#253746',s=35,label='Fixed start',zorder=5)
        map_ax.scatter(*goal[:2],marker='*',color='#b33e4a',s=85,label='Fixed goal',zorder=5)
        arc=np.r_[0.,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
        corners=np.asarray([[.815,.4825],[.815,-.4825],[-.815,-.4825],[-.815,.4825]])
        for index in sorted(set(np.searchsorted(arc,arc[-1]*np.asarray([0.,.5,1.])).tolist())):
            c,s=np.cos(yaw[index]),np.sin(yaw[index]);rotation=np.asarray([[c,-s],[s,c]])
            map_ax.add_patch(Polygon(corners@rotation.T+[x[index],y[index]],closed=True,
                fill=False,edgecolor='#087e91',lw=1.,alpha=.6))
        map_ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World x (m)',ylabel='World y (m)',
            title='(a) Measured route and body outlines at 3 poses')
        map_ax.set_aspect('equal',adjustable='box');map_ax.legend(loc='best',fontsize=7)
        speed_ax.plot(t,command,color='#dc8b28',lw=1.2,label='Signed speed command')
        speed_ax.plot(t,u,color='#087e91',lw=.65,alpha=.85,label='Measured body-forward velocity')
        speed_ax.set(xlabel='Route clock (s)',ylabel='Longitudinal velocity (mm/s)',
            title='(b) Ordinary ramp, tracking and terminal control')
        speed_ax.legend(loc='upper left',fontsize=7)
        inset=speed_ax.inset_axes([.59,.52,.38,.34]);initial=t<=8.
        inset.plot(t[initial],command[initial],color='#dc8b28',lw=1.)
        inset.plot(t[initial],u[initial],color='#087e91',lw=.6)
        inset.set(xlim=(0.,8.),title='Startup: first 8 s');inset.tick_params(labelsize=6)
        inset.title.set_fontsize(7)
        error_ax.plot(t[tracking],track_position[tracking],color='#087e91',lw=1.,label='Position deviation')
        error_heading=error_ax.twinx()
        error_heading.plot(t[tracking],track_heading[tracking],color='#a74c76',lw=.8,label='Absolute heading deviation')
        error_ax.set(xlabel='Route clock (s)',ylabel='Path deviation (mm)',title='(c) Tracking only; terminal correction excluded')
        error_heading.set_ylabel('Path heading deviation (deg)',color='#a74c76')
        error_ax.text(.02,.96,f"Time-weighted RMS: {metrics['tracking_only_time_weighted_position_RMS_mm']:.2f} mm / "
            f"{metrics['tracking_only_time_weighted_heading_RMS_deg']:.3f} deg",transform=error_ax.transAxes,va='top',fontsize=8)
        terminal_heading=terminal_ax.twinx()
        if terminal.any():
            terminal_ax.plot(t[terminal],position[terminal],color='#087e91',lw=1.,label='Goal position error')
            terminal_heading.plot(t[terminal],heading[terminal],color='#a74c76',lw=1.,label='Absolute goal heading error')
            hold=phases=='terminal_hold';tail=hold&(values('phase_time_s')>=2.)
            if hold.any():terminal_ax.axvspan(t[hold][0],t[hold][-1],color='#4b9d7a',alpha=.14,label='Observed fixed hold')
            if tail.any():terminal_ax.axvspan(t[tail][0],t[tail][-1],color='#4b9d7a',alpha=.2,label='Observed final 3 s window')
            terminal_ax.axhline(1000*raw['acceptance']['position_error_m'],color='#087e91',ls='--',lw=.8)
            terminal_heading.axhline(raw['acceptance']['heading_error_deg'],color='#a74c76',ls='--',lw=.8)
            terminal_ax.set_xlim(t[terminal][0],t[-1]+.5)
            terminal_ax.legend(loc='best',fontsize=7)
        else:terminal_ax.text(.5,.5,'No terminal measurements recorded',ha='center',transform=terminal_ax.transAxes)
        terminal_ax.set(xlabel='Route clock (s)',ylabel='Goal position error (mm)',
            title='(d) Common terminal correction and fixed hold')
        terminal_heading.set_ylabel('Absolute goal heading error (deg)',color='#a74c76')
        for axis in axes.flat:axis.grid(alpha=.18);axis.set_axisbelow(True)
        label={'P_dynamics_full_envelope':'P','B2_multimode_transition_cost_ramp':'B2',
            'B1_multimode_kinematic_ramp':'B1','B0_fixed_counterphase_ramp':'B0'}[raw['method']]
        fig.suptitle(f"H4 ward-corridor turn | {label} | raw verdict: {raw['verdict']} | one development trial",fontsize=13)
        fig.text(.5,.025,'Ground-truth Isaac Sim record, not final repeated comparison or sensor/Nav2 validation.\n'
            'Map AABBs and body outlines are 2D proxies, not continuous cooked-mesh or loaded-patient 3D clearance.',
            ha='center',fontsize=8,color='#4b5563')
        fig.tight_layout(rect=(0.,.07,1.,.95))
        try:
            for image_path in images:
                with image_path.open('xb') as stream:
                    kwargs={'metadata':{'CreationDate':None,'ModDate':None}} if image_path.suffix=='.pdf' else {}
                    fig.savefig(stream,format=image_path.suffix[1:],facecolor='white',**kwargs)
        finally:plt.close(fig)
    if any(file_sha(Path(p))!=sha for p,sha in pins.items()):raise ValueError('Figure input changed during export')
    write_exclusive(manifest,dict(**definition,run_id=raw['run_id'],raw_verdict=raw['verdict'],
        measured_pose_count=len(rows),trace_observations=metrics,
        no_nominal_reference_used_as_measured_response=True,
        no_independent_repeat_or_statistical_superiority_inferred=True,
        no_actual_scene_or_timeline_access=True,original_results_modified=False,
        use_for_final_statistics=False,output_sha256={str(p):file_sha(p) for p in images}))
    print('Saved measured one-trial figure:',images[0])
    print('Source/units/scope manifest:',manifest)
    return manifest


def nominal_measured_clock_overlap(measured,nominal):
    """No shift, rigid alignment, DTW, bias removal or extrapolation.

    This compares two CLOSED-LOOP responses at the same route clock, not
    identical-input open-loop identification and not path-tracking error.
    Metrics integrate squared residuals on the observed measured-time grid.
    """
    import numpy as np
    def channel(rows,key):
        try:values=np.asarray([float(row[key]) for row in rows],dtype=float)
        except (KeyError,ValueError,TypeError) as error:raise ValueError('Missing/non-numeric trace channel: '+key) from error
        if not np.isfinite(values).all():raise ValueError('Non-finite trace channel: '+key)
        return values
    a,b=channel(measured,'time_s'),channel(nominal,'time_s')
    if (len(a)<2 or len(b)<2 or np.any(np.diff(a)<=0.) or np.any(np.diff(b)<=0.)
            or a[0]<0. or b[0]<0.):
        raise ValueError('Both complete finite strictly increasing route clocks required')
    select=(a>=b[0])&(a<=b[-1]);t=a[select]
    if len(t)<2:raise ValueError('At least two measured samples inside the model clock support required')
    ax,ay,ah=(channel(measured,k)[select] for k in ('x_m','y_m','yaw_rad'))
    bx,by,bh=(channel(nominal,k) for k in ('x_m','y_m','yaw_rad'))
    px,py=np.interp(t,b,bx),np.interp(t,b,by)
    ph=np.interp(t,b,np.unwrap(bh))
    du=(channel(measured,'vx_physx_m_s')[select]*np.cos(ah)
        +channel(measured,'vy_physx_m_s')[select]*np.sin(ah)-np.interp(t,b,channel(nominal,'u_m_s')))
    position=np.hypot(ax-px,ay-py)
    heading=np.degrees(np.arctan2(np.sin(ah-ph),np.cos(ah-ph)))
    duration=float(t[-1]-t[0])
    rms=lambda v:float(np.sqrt(np.trapz(v*v,t)/duration))
    summary=dict(overlap_route_clock_s=[float(t[0]),float(t[-1])],overlap_sample_count=len(t),
        measured_route_clock_support_s=[float(a[0]),float(a[-1])],
        nominal_route_clock_support_s=[float(b[0]),float(b[-1])],
        time_weighted_equal_clock_position_RMS_mm=1000*rms(position),
        maximum_sampled_equal_clock_position_difference_mm=1000*float(position.max()),
        time_weighted_equal_clock_wrapped_heading_RMS_deg=rms(heading),
        time_weighted_body_forward_velocity_residual_RMS_mm_s=1000*rms(du),
        initial_measured_minus_nominal_world_xy_mm=[1000*float(ax[0]-px[0]),1000*float(ay[0]-py[0])],
        initial_measured_minus_nominal_heading_deg=float(heading[0]),
        measured_samples_outside_model_support=int((~select).sum()),
        measured_end_after_nominal_end_s=max(0.,float(a[-1]-b[-1])),
        absolute_clocks_used_without_time_shift_or_pose_bias_removal=True,
        no_model_extrapolation_or_endpoint_clamping_outside_support=True,
        equal_clock_error_includes_progress_timing_error_NOT_path_tracking_error=True,
        closed_loop_commands_can_differ_NOT_identical_input_system_identification=True,
        telemetry_rows_NOT_independent_trials=True)
    residuals=[dict(time_s=float(time),position_difference_mm=1000*float(dp),
        wrapped_heading_difference_deg=float(dh),body_forward_velocity_residual_mm_s=1000*float(dv))
        for time,dp,dh,dv in zip(t,position,heading,du)]
    return summary,residuals


def write_frozen_prediction_review(result,output=None):
    """Compare an explicit actual outcome to BOTH previously pinned screens.

    Never select the best step size, fit the response, relabel a raw outcome,
    or claim that a nominal caster/hold-free MODEL is actual validation.
    """
    import inspect
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from hospital_h4_path_execution import COMMON_SOURCE_FILES
    from hospital_runtime_evidence_io import write_exclusive
    result=Path(result).resolve();raw=json.loads(result.read_bytes())
    record=inspect_execution_record(result,'method_path')
    if (raw.get('scenario')!='H4_ward_corridor_turn' or record['outcome_stage']!='ACTUAL_PATH_OUTCOME'
            or record['trace_pin_status']!='ORIGINAL_TRACE_HASH_VERIFIED' or record['integrity_errors']):
        raise ValueError('Explicit intact actual H4 outcome required, not a setup/HOLD or nominal result')
    plan_path=Path(raw['plan_path']);bundle_path=plan_path.resolve().parent/'execution_bundle.json'
    bundle=json.loads(bundle_path.read_bytes())
    if (str(plan_path.resolve()) not in bundle['plans']
            or bundle['output_sha256'].get(str(plan_path.resolve()))!=raw['plan_sha256']
            or raw['input_sha256'].get(str(bundle_path))!=file_sha(bundle_path)):
        raise ValueError('Actual prediction review must use its own already-pinned planner bundle')
    screen_path=(Path(bundle['model_screen']) if bundle.get('additional_common_pose_interface_predicate_is_NEW_planning_version') is True else
        RESULT/'hospital_h4_common_executor_screen/screen_r01/common_executor_model_screen.json')
    screen=json.loads(screen_path.read_bytes());executed_path=Path(raw['path_protocol'])
    executed=json.loads(executed_path.read_bytes())
    if (file_sha(executed_path)!=raw['path_protocol_sha256']
            or raw['input_sha256'].get(str(screen_path))!=file_sha(screen_path)
            or executed['input_sha256'].get(str(screen_path))!=file_sha(screen_path)):
        raise ValueError('Executed protocol must already pin this exact pre-existing prediction screen')
    plan=json.loads(plan_path.read_bytes())
    if (file_sha(plan_path)!=raw['plan_sha256'] or plan['method']!=raw['method']
            or plan['start_world_xyyaw']!=raw['start_world_xyyaw']
            or plan['goal_world_xyyaw']!=raw['goal_world_xyyaw']):
        raise ValueError('Own unchanged plan and world frame required; no fitted reference')
    source_checks={name:screen['input_sha256'].get(str(_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts', name)))
        for name in COMMON_SOURCE_FILES}
    if any(not sha or sha!=raw['source_sha256'].get(name) for name,sha in source_checks.items()):
        raise ValueError('Prediction and actual outcome must use the same shared physical/control source files')
    cases=[case for case in screen['cases'] if case['method']==raw['method']]
    if len(cases)!=2 or {c['integration_dt_s'] for c in cases}!={.01,.02}:
        raise ValueError('Both original 10ms and 20ms MODEL screens required; no favorable step-size selection')
    trace=result.with_suffix('.csv')
    pins={str(p):file_sha(p) for p in (result,trace,screen_path,executed_path,plan_path,bundle_path)}
    models=[]
    with trace.open(newline='') as stream:measured=list(csv.DictReader(stream))
    for case in cases:
        path=Path(case['model_trace']);sha=file_sha(path)
        trace_owner_pin=screen['output_sha256'].get(str(path))
        if case.get('exact_original_case_reused') is True:
            old_path=Path(plan['original_plan']);old_plan=json.loads(old_path.read_bytes())
            if (raw['input_sha256'].get(str(old_path))!=file_sha(old_path)
                    or any(plan[key]!=old_plan[key] for key in ('selected_candidate_id','selected_actions','selected_states'))):
                raise ValueError('A reused nominal trace must retain its exact original reference, not merely the same method name')
            pins[str(old_path)]=file_sha(old_path);trace_owner_pin=screen['input_sha256'].get(str(path))
        elif case.get('new_nominal_case_executed') is True and case.get('selected_candidate_id')!=plan['selected_candidate_id']:
            raise ValueError('Prospective nominal trace belongs to another selected candidate')
        if (case['model_trace_sha256']!=sha or raw['input_sha256'].get(str(path))!=sha
                or executed['input_sha256'].get(str(path))!=sha
                or trace_owner_pin!=sha):
            raise ValueError('Every model trace must be pinned in the already executed protocol')
        with path.open(newline='') as stream:nominal=list(csv.DictReader(stream))
        if (float(nominal[0]['time_s'])!=0.
                or [float(nominal[0][k]) for k in ('x_m','y_m','yaw_rad')]!=plan['start_world_xyyaw']
                or float(nominal[-1]['time_s'])!=case['simulation_duration_s']):
            raise ValueError('Original nominal zero-start/frame/duration must match its pinned trace')
        summary,residuals=nominal_measured_clock_overlap(measured,nominal)
        complete=record['trace_observations']['fixed_hold_and_tail_observation_complete']
        model_complete=(case['checks'].get('reference_and_ONE_model_hold_completed') is True
            and case['checks'].get('fixed_tail_at_least_2p9s') is True)
        last_actual=np.asarray([float(measured[-1][k]) for k in ('x_m','y_m','yaw_rad')])
        last_model=np.asarray([float(nominal[-1][k]) for k in ('x_m','y_m','yaw_rad')])
        endpoint=dict(position_difference_mm=1000*float(np.linalg.norm(last_actual[:2]-last_model[:2])),
            absolute_wrapped_heading_difference_deg=abs(float(np.degrees(np.arctan2(
                np.sin(last_actual[2]-last_model[2]),np.cos(last_actual[2]-last_model[2])))))) if complete and model_complete else None
        row=dict(integration_dt_s=case['integration_dt_s'],model_status=case['model_status'],
            nominal_total_duration_s=case['simulation_duration_s'],nominal_terminal_pulse_count=case['terminal_pulse_count'],
            actual_end_minus_nominal_end_s=float(measured[-1]['time_s'])-case['simulation_duration_s'],
            duration_difference_is_NOT_common_180s_versus_300s_success_comparison=True,
            nominal_terminal_position_error_mm=1000*case['final_position_error_m'],
            nominal_terminal_absolute_heading_error_deg=abs(case['final_heading_error_deg']),
            nominal_fixed_MODEL_hold_completed=model_complete,
            endpoints_after_own_completed_holds=endpoint,
            incomplete_actual_or_model_hold_is_NOT_compared_as_completed_endpoints=not (complete and model_complete),
            equal_clock_overlap=summary,model_trace=str(path),model_trace_sha256=sha)
        models.append(dict(row=row,nominal=nominal,residuals=residuals));pins[str(path)]=sha
    definition=dict(version='H4_pinned_frozen_nominal_vs_actual_closed_loop_response_review_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_frozen_prediction_review).encode()).hexdigest(),
        metric_function_sha256=hashlib.sha256(inspect.getsource(nominal_measured_clock_overlap).encode()).hexdigest(),
        matplotlib_version=matplotlib.__version__,numpy_version=np.__version__)
    output=Path(output) if output is not None else result.parent/'frozen_prediction_review'
    target=output/'prediction_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()):raise ValueError('Existing derived review has other inputs/exporter; no overwrite')
        if any(not Path(p).is_file() or file_sha(Path(p))!=sha for p,sha in saved['output_sha256'].items()):
            raise ValueError('Existing prediction-review output changed/missing; no silent replacement')
        return target
    output.mkdir(parents=True,exist_ok=True)
    images=[output/'frozen_vs_measured.png',output/'frozen_vs_measured.pdf']
    tables=[output/('clock_overlap_%dms.csv'%round(1000*m['row']['integration_dt_s'])) for m in models]
    if any(p.exists() for p in (*images,*tables)):raise ValueError('Unfinalized derived output exists; no overwrite')
    for model,path in zip(models,tables):
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(model['residuals'][0]));writer.writeheader();writer.writerows(model['residuals'])
    t=np.asarray([float(r['time_s']) for r in measured]);yaw=np.asarray([float(r['yaw_rad']) for r in measured])
    u=1000*(np.asarray([float(r['vx_physx_m_s']) for r in measured])*np.cos(yaw)
        +np.asarray([float(r['vy_physx_m_s']) for r in measured])*np.sin(yaw))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':160}):
        fig,axes=plt.subplots(3,1,figsize=(10.5,9.),sharex=True)
        progress,speed,error=axes;heading_axis=error.twinx()
        progress.plot(t,[float(r['path_progress_m']) for r in measured],color='#087e91',lw=1.6,label='Measured Isaac progress')
        speed.plot(t,u,color='#087e91',lw=.6,alpha=.85,label='Measured body-forward velocity')
        for model,color in zip(models,('#dc8b28','#a74c76')):
            nominal=model['nominal'];dt=model['row']['integration_dt_s'];label='Frozen nominal MODEL (%d ms)'%round(1000*dt)
            mt=[float(r['time_s']) for r in nominal]
            progress.plot(mt,[float(r['reference_progress_m']) for r in nominal],color=color,ls='--',lw=1.,label=label)
            speed.plot(mt,[1000*float(r['u_m_s']) for r in nominal],color=color,ls='--',lw=.9,label=label)
            residuals=model['residuals'];rt=[r['time_s'] for r in residuals]
            error.plot(rt,[r['position_difference_mm'] for r in residuals],color=color,lw=1.,label='XY discrepancy (%d ms)'%round(1000*dt))
            heading_axis.plot(rt,[abs(r['wrapped_heading_difference_deg']) for r in residuals],color=color,ls=':',lw=.8)
        progress.set(ylabel='Reference progress (m)',title='(a) Frozen nominal predictions and one actual development run')
        speed.set(ylabel='Signed velocity (mm/s)',title='(b) Closed-loop body response, not an identical-input identification')
        error.set(xlabel='Unshifted route clock (s)',ylabel='Equal-clock XY discrepancy (mm)',
            title='(c) Same-clock trajectory discrepancy includes timing lag; NOT tracking error')
        heading_axis.set_ylabel('Absolute heading discrepancy (deg; dotted)')
        for axis in axes:axis.grid(alpha=.18);axis.legend(loc='upper left',fontsize=8)
        fig.suptitle('H4 | '+raw['method'].split('_')[0]+' | raw actual verdict: '+raw['verdict']+' | both original MODEL resolutions',fontsize=12)
        fig.text(.5,.025,'No fitting, time shift, pose alignment or extrapolation. MODEL caster/physical-hold/contact effects are not simulated.\n'
            'One development record; not an independent model admission, paired-method comparison or final repeated statistic.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0.,.07,1.,.96))
        try:
            for path in images:
                with path.open('xb') as stream:
                    options={'metadata':{'CreationDate':None,'ModDate':None}} if path.suffix=='.pdf' else {}
                    fig.savefig(stream,format=path.suffix[1:],facecolor='white',**options)
        finally:plt.close(fig)
    if any(file_sha(Path(p))!=sha for p,sha in pins.items()):raise ValueError('Input changed during derived review; raw evidence untouched')
    write_exclusive(target,dict(**definition,status='PINNED_NOMINAL_VS_MEASURED_DIAGNOSTIC_NOT_MODEL_ADMISSION',
        run_id=raw['run_id'],method=raw['method'],raw_verdict=raw['verdict'],original_controller_verdict_NOT_relabelled=True,
        actual_total_observed_route_clock_s=float(t[-1]),actual_terminal_pulse_count=raw.get('terminal_pulse_count'),
        actual_terminal_position_error_mm=record['reported_position_error_mm'],
        actual_terminal_absolute_heading_error_deg=abs(record['reported_heading_error_deg']) if record['reported_heading_error_deg'] is not None else None,
        actual_complete_fixed_hold=record['trace_observations']['fixed_hold_and_tail_observation_complete'],
        actual_tracking_only_position_RMS_mm=record['trace_observations']['tracking_only_time_weighted_position_RMS_mm'],
        both_original_resolutions_retained_NOT_independent_repeats=True,
        prediction_archives_pre_pinned_in_executed_protocol_NOT_refitted=True,common_control_source_hashes=source_checks,
        model_resolution_rows=[m['row'] for m in models],
        physical_caster_stiction_contact_and_captured_wheel_hold_NOT_in_nominal_screen=True,
        original_nominal_screen_180s_limit_NOT_relabelled_as_actual_common_300s_trial=True,
        no_model_parameter_gain_safety_or_raw_file_change=True,no_live_stage_or_timeline_access=True,
        new_PhysX_trial_count=0,actual_model_admission=False,use_for_final_statistics=False,
        output_sha256={str(p):file_sha(p) for p in (*images,*tables)}))
    print('Saved frozen-prediction/actual diagnostic:',target)
    return target


def individual_development_contrasts(table,configuration_matching):
    """Describe unique completed P/baseline pairs, never rank partial FAILs.

    Each metric is lower-is-better. Signed terminal heading is converted to
    magnitude before comparison; a negative heading is not a better error.
    No confidence interval, pooled repetitions or causal claim is inferred.
    """
    metrics=('total_logged_route_clock_s','tracking_only_time_weighted_position_RMS_mm',
        'tracking_only_time_weighted_heading_RMS_deg','terminal_position_error_mm',
        'terminal_absolute_heading_error_deg','terminal_pulse_count')
    def values(row):
        result={key:row.get(key) for key in metrics}
        heading=row.get('terminal_heading_error_deg')
        if type(heading) not in (int,float) or not math.isfinite(heading):
            raise ValueError('Finite nonnegative completed contrast metrics required after heading magnitude conversion')
        result['terminal_absolute_heading_error_deg']=abs(heading)
        if any(type(value) not in (int,float) or not math.isfinite(value) or value<0.
                for value in result.values()):
            raise ValueError('Finite nonnegative completed contrast metrics required')
        return result
    proposed=[row for row in table if row['method']=='P_dynamics_full_envelope']
    contrasts=[]
    for method in METHODS[:3]:
        baseline=[row for row in table if row['method']==method]
        item=dict(baseline_method=method,P_observation_count=len(proposed),
            baseline_observation_count=len(baseline),P_minus_baseline=None,
            P_minus_baseline_percent=None,lower_error_or_duration_for_P=None,
            P_strictly_lower_on_all_listed_metrics=None,
            independent_paired_repetition_or_statistical_superiority_claimed=False)
        if len(proposed)!=1 or len(baseline)!=1:
            item['status']='MISSING_OR_NONUNIQUE_PAIR_NO_SELECTION'
        elif not configuration_matching:
            item['status']='UNMATCHED_CONFIGURATION_NOT_RANKED'
        else:
            p,b=proposed[0],baseline[0]
            item.update(P_run_id=p['run_id'],baseline_run_id=b['run_id'])
            if not all(row['raw_verdict']=='PASS' and row['complete_fixed_hold'] for row in (p,b)):
                item['status']='FAILED_OR_INCOMPLETE_PAIR_NOT_RANKED'
            else:
                pvalue,bvalue=values(p),values(b)
                item.update(status='DESCRIPTIVE_COMPLETED_INDIVIDUAL_PAIR_NOT_FINAL_STATISTICS',
                    P_minus_baseline={key:pvalue[key]-bvalue[key] for key in metrics},
                    P_minus_baseline_percent={key:100.*(pvalue[key]-bvalue[key])/bvalue[key]
                        if bvalue[key]>0. else None for key in metrics},
                    lower_error_or_duration_for_P={key:pvalue[key]<bvalue[key] for key in metrics},
                    P_strictly_lower_on_all_listed_metrics=all(pvalue[key]<bvalue[key] for key in metrics))
        contrasts.append(item)
    return contrasts


def write_explicit_development_comparison(results,output=None):
    """Explicit actual trials, not best/latest selection or paired statistics.

    Per-attempt actor/cohort namespaces differ intentionally. Every physical,
    reference-domain, control, guard and source identity is compared before
    reporting a descriptive P/B2 difference. A fixed source seed alone does
    NOT establish independently sampled, paired entry-history repetitions.
    """
    import inspect
    from hospital_h4_path_execution import COMMON_SOURCE_FILES
    from hospital_runtime_evidence_io import write_exclusive
    paths=[Path(p).resolve() for p in results]
    if not paths or len(set(paths))!=len(paths):raise ValueError('Explicit distinct actual result paths required')
    records=[];raws=[];pins={}
    for path in paths:
        row=inspect_execution_record(path,'method_path');raw=json.loads(path.read_bytes())
        if (row['scenario']!='H4_ward_corridor_turn' or row['outcome_stage']!='ACTUAL_PATH_OUTCOME'
                or row['trace_pin_status']!='ORIGINAL_TRACE_HASH_VERIFIED' or row['integrity_errors']):
            raise ValueError('Actual intact H4 outcomes only; setup and prepath HOLDs are not method performance')
        plan_path=Path(raw['plan_path']);plan=json.loads(plan_path.read_bytes())
        if (file_sha(plan_path)!=raw['plan_sha256'] or plan['method']!=raw['method']
                or raw['method'] not in METHODS):raise ValueError('Every outcome must retain its own method/plan identity')
        protocol_path=Path(raw['path_protocol']);protocol=json.loads(protocol_path.read_bytes())
        reservation_path=path.parent/'single_route_reserved.json';reservation=json.loads(reservation_path.read_bytes())
        if (file_sha(protocol_path)!=raw['path_protocol_sha256'] or protocol.get('method')!=raw['method']
                or protocol['input_sha256'].get(str(plan_path))!=raw['plan_sha256']
                or reservation['method']!=raw['method'] or reservation['run_id']!=raw['run_id']
                or reservation['protocol_sha256']!=raw['path_protocol_sha256']
                or reservation['actual_static_sha256']!=raw['static_gate_sha256']):
            raise ValueError('Own executed method/protocol and exclusive route reservation required')
        records.append(row);raws.append(raw)
        pins.update({str(p):file_sha(p) for p in (path,path.with_suffix('.csv'),plan_path,protocol_path,reservation_path)})
    if len({r['run_id'] for r in records})!=len(records):
        raise ValueError('Copied outcome receipts are not separate development runs')
    controls=[dict(row['comparison_controls'],environment_variant=raw.get('environment_variant'),
        common_resource_contract=raw.get('common_resource_contract')) for row,raw in zip(records,raws)]
    compare_fields=sorted(set(controls[0])-{'execution_cohort','source_sha256'})
    mismatches={}
    for row,raw,current in zip(records,raws,controls):
        changed=[k for k in compare_fields if controls[0].get(k) is None or current.get(k)!=controls[0].get(k)]
        for name in (*COMMON_SOURCE_FILES,'isaac_run_ijcas_hospital_h4_subfloor_compact_path_gate.py'):
            expected=raws[0].get('source_sha256',{}).get(name)
            if not expected or raw.get('source_sha256',{}).get(name)!=expected:changed.append('source_sha256.'+name)
        for key in ('caster_state_reset_during_run','body_pose_reset_during_run','material_modified_during_run',
                'CICS_input_used','Nav2_executed','sensor_closed_loop_verified'):
            if raw.get(key) is not False:changed.append('required_false.'+key)
        if changed:mismatches[row['run_id']]=changed
    table=[]
    for row in records:
        metric=row['trace_observations'];gap=row.get('sampled_authored_envelope_observations') or {}
        table.append(dict(run_id=row['run_id'],method=row['method'],cohort=row['execution_cohort'],
            raw_verdict=row['reported_verdict'],total_logged_route_clock_s=metric['last_physics_time_s'],
            tracking_only_time_weighted_position_RMS_mm=metric['tracking_only_time_weighted_position_RMS_mm'],
            tracking_only_time_weighted_heading_RMS_deg=metric['tracking_only_time_weighted_heading_RMS_deg'],
            terminal_position_error_mm=row['reported_position_error_mm'],terminal_heading_error_deg=row['reported_heading_error_deg'],
            terminal_pulse_count=row['terminal_pulses'],obstacle_callback_count=row['obstacle_callback_count'],
            complete_fixed_hold=metric['fixed_hold_and_tail_observation_complete'],
            raw_sampled_authored_AABB_body_gap_mm=(1000*gap['raw_body_footprint']['minimum_sampled_rectangle_to_authored_AABB_distance_m']
                if 'raw_body_footprint' in gap else None),result=row['result']))
    difference=None
    proposed=[r for r in table if r['method']=='P_dynamics_full_envelope']
    baseline=[r for r in table if r['method']=='B2_multimode_transition_cost_ramp']
    if not mismatches and len(proposed)==len(baseline)==1:
        p,b=proposed[0],baseline[0]
        both=p['raw_verdict']==b['raw_verdict']=='PASS' and p['complete_fixed_hold'] and b['complete_fixed_hold']
        metrics=('total_logged_route_clock_s','tracking_only_time_weighted_position_RMS_mm',
            'tracking_only_time_weighted_heading_RMS_deg','terminal_position_error_mm','terminal_pulse_count')
        difference=dict(P_run_id=p['run_id'],B2_run_id=b['run_id'],both_actual_runs_PASS_with_completed_hold=both,
            P_minus_B2={key:p[key]-b[key] for key in metrics} if both else None,
            incomplete_or_failed_time_NOT_ranked_as_a_successful_completion_time=True,
            descriptive_individual_trial_difference_NOT_paired_estimate_or_causal_mechanism=True)
    contrasts=individual_development_contrasts(table,not mismatches)
    definition=dict(version='explicit_actual_H4_development_comparison_NOT_final_statistics_v1',input_sha256=pins,
        contrast_function_sha256=hashlib.sha256(inspect.getsource(individual_development_contrasts).encode()).hexdigest(),
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_explicit_development_comparison).encode()).hexdigest())
    key=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=Path(output) if output is not None else RESULT/'paper_evidence_inventory/development_comparisons'/key
    target=output/'development_comparison.json';csv_path=output/'actual_development_rows.csv'
    if target.exists():
        old=json.loads(target.read_bytes())
        if any(old.get(k)!=v for k,v in definition.items()) or file_sha(csv_path)!=old['output_sha256'][str(csv_path)]:
            raise ValueError('Existing explicit comparison changed; no overwrite')
        return target
    output.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if any(file_sha(Path(p))!=sha for p,sha in pins.items()):raise ValueError('Explicit input changed during review')
    write_exclusive(target,dict(**definition,actual_development_rows=table,
        configuration_matching_checks_passed=len(table)>=2 and not mismatches,physical_control_or_domain_mismatches=mismatches,
        per_attempt_cohort_namespace_difference_is_NOT_an_initial_history_or_randomization_proof=True,
        source_authored_seed_policy_is_NOT_a_measured_equal_caster_history_claim=True,
        compared_common_source_files=[*COMMON_SOURCE_FILES,'isaac_run_ijcas_hospital_h4_subfloor_compact_path_gate.py'],
        missing_methods=sorted(set(METHODS)-{row['method'] for row in table}),
        P_vs_B2_individual_development_contrast=difference,
        P_vs_each_baseline_individual_development_contrasts=contrasts,
        all_supplied_original_outcomes_retained_no_best_or_latest_selection_by_this_exporter=True,
        explicit_subset_does_NOT_claim_to_be_all_historical_attempts=True,
        all_four_method_final_comparison_complete=False,independent_paired_repeats_established=False,
        qualified_final_comparison_cells=0,statistical_superiority_established=False,
        original_180s_and_other_environment_versions_NOT_pooled=True,
        no_actual_scene_or_timeline_access=True,new_PhysX_trial_count=0,use_for_final_statistics=False,
        output_sha256={str(csv_path):file_sha(csv_path)}))
    print('Saved explicit development comparison (not final statistics):',target)
    return target


def write_development_comparison_figure(results,output=None):
    """Matched development outcomes, retaining FAILs without ranking partials."""
    import inspect
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from hospital_runtime_evidence_io import write_exclusive
    comparison=write_explicit_development_comparison(results)
    review=json.loads(comparison.read_bytes());table=review['actual_development_rows']
    if (not review['configuration_matching_checks_passed']
            or len({row['method'] for row in table})!=len(table)
            or any(row['raw_verdict'] not in ('PASS','FAIL')
                or row['raw_verdict']=='PASS' and not row['complete_fixed_hold'] for row in table)
            or not any(row['raw_verdict']=='PASS' for row in table)):
        raise ValueError('Unique configuration-matched actual PASS/FAIL outcomes with complete PASS holds required')
    table=sorted(table,key=lambda row:METHODS.index(row['method']))
    raws=[json.loads(Path(row['result']).read_bytes()) for row in table]
    plan=json.loads(Path(raws[0]['plan_path']).read_bytes())
    geometry=Path(plan['geometry_manifest']);setup=json.loads(geometry.read_bytes())
    if any(raw['geometry_manifest_sha256']!=file_sha(geometry) for raw in raws):
        raise ValueError('Own shared original geometry required')
    pins={**review['input_sha256'],str(comparison):file_sha(comparison),str(geometry):file_sha(geometry)}
    definition=dict(version='matched_H4_development_figure_FAILs_retained_NOT_final_v2',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_development_comparison_figure).encode()).hexdigest(),
        matplotlib_version=matplotlib.__version__)
    output=(Path(output) if output is not None else comparison.parent/'figure').resolve()
    target=output/'comparison_figure.json'
    if target.exists():
        old=json.loads(target.read_bytes())
        if (any(old.get(key)!=value for key,value in definition.items())
                or any(not Path(p).is_file() or file_sha(p)!=sha for p,sha in old['output_sha256'].items())):
            raise ValueError('Existing comparison figure changed; no overwrite')
        return target
    traces=[]
    for row in table:
        with Path(row['result']).with_suffix('.csv').open(newline='') as stream:
            trace=list(csv.DictReader(stream))
        xy=np.asarray([[float(sample['x_m']),float(sample['y_m'])] for sample in trace])
        if not np.isfinite(xy).all():raise ValueError('Finite measured coordinates required')
        traces.append(xy)
    pairs=[dict(first=table[i]['method'],second=table[j]['method'])
        for i in range(len(table)) for j in range(i+1,len(table))
        if file_sha(Path(table[i]['result']).with_suffix('.csv'))==file_sha(Path(table[j]['result']).with_suffix('.csv'))]
    output.mkdir(parents=True,exist_ok=True)
    images=[output/'matched_development.png',output/'matched_development.pdf']
    if any(path.exists() for path in images):raise ValueError('Unfinalized comparison figure exists; no overwrite')
    labels=[row['method'].split('_')[0] for row in table]
    palette={'B0':'#4477aa','B1':'#ee7733','B2':'#aa3377','P':'#228833'}
    colors=[palette[name] for name in labels]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'figure.dpi':160}):
        fig,axes=plt.subplots(2,2,figsize=(11.5,8.7))
        try:
            map_ax,rms_ax,time_ax,goal_ax=axes.flat
            points=np.concatenate(traces);lo=points.min(axis=0)-1.;hi=points.max(axis=0)+1.
            for obstacle in setup['included_collision_AABBs']:
                bounds=obstacle['world_aabb'];a=np.asarray(bounds['minimum_m'][:2]);b=np.asarray(bounds['maximum_m'][:2])
                if np.all(b>=lo) and np.all(a<=hi):
                    map_ax.add_patch(Rectangle(a,*(b-a),facecolor='#e1e4e7',edgecolor='#a0a6aa',linewidth=.4))
            for label,color,xy in zip(labels,colors,traces):
                verdict=next(row['raw_verdict'] for row in table if row['method'].split('_')[0]==label)
                map_ax.plot(xy[:,0],xy[:,1],color=color,lw=1.3,linestyle='--' if label=='B2' else '-',
                    label=label+(' (FAIL)' if verdict=='FAIL' else ''))
            for key,marker,label in [('start_world_xyyaw','o','Fixed start'),('goal_world_xyyaw','*','Fixed goal')]:
                point=raws[0][key];map_ax.scatter(*point[:2],marker=marker,color='#222222',s=55,label=label,zorder=5)
            map_ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World x (m)',ylabel='World y (m)',
                title='(a) Actual base trajectories; authored AABBs in grey')
            map_ax.set_aspect('equal',adjustable='box');map_ax.legend(fontsize=7,loc='best')
            completed=[row['raw_verdict']=='PASS' and row['complete_fixed_hold'] for row in table]
            rms=[row['tracking_only_time_weighted_position_RMS_mm'] if good else None for row,good in zip(table,completed)]
            elapsed=[row['total_logged_route_clock_s'] if good else None for row,good in zip(table,completed)]
            for ax,values,title,ylabel in [(rms_ax,rms,'(b) Tracking-only deviation from each own reference','Time-weighted position RMS (mm)'),
                    (time_ax,elapsed,'(c) Complete route clock, including terminal hold','Elapsed simulation route clock (s)')]:
                bars=ax.bar(labels,[value if value is not None else 0. for value in values],color=colors,width=.6)
                for bar,value in zip(bars,values):
                    ax.annotate(f'{value:.2f}' if value is not None else 'FAIL\nnot completed',
                        (bar.get_x()+bar.get_width()/2,value if value is not None else 0.),xytext=(0,4),
                        textcoords='offset points',ha='center',fontsize=9)
                ax.set(title=title,ylabel=ylabel,ylim=(0.,max(value for value in values if value is not None)*1.18))
                ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            acceptance=raws[0]['acceptance']
            position_band=1000*acceptance['position_error_m']
            heading_band=acceptance['heading_error_deg']
            goal_ax.axvline(position_band,color='#888888',ls='--',lw=1.,label='Original pose bands')
            goal_ax.axhline(heading_band,color='#888888',ls='--',lw=1.)
            for index,(row,label,color) in enumerate(zip(table,labels,colors)):
                if not completed[index]:continue
                x=row['terminal_position_error_mm'];y=abs(row['terminal_heading_error_deg'])
                goal_ax.scatter(x,y,color=color,s=65,marker='s' if label=='B2' else 'o',facecolors='none' if label=='B2' else color)
                goal_ax.annotate(label,(x,y),xytext=(7,7 if label!='B2' else -13),textcoords='offset points',color=color)
            goal_ax.set(xlabel='Terminal position error (mm)',ylabel='Absolute terminal heading error (deg)',
                xlim=(0.,max(position_band,max(row['terminal_position_error_mm'] for row,good in zip(table,completed) if good))*1.15),
                ylim=(0.,max(heading_band,max(abs(row['terminal_heading_error_deg']) for row,good in zip(table,completed) if good))*1.15),
                title='(d) Completed terminal result; original verdicts retained')
            failed=[label for label,good in zip(labels,completed) if not good]
            if failed:goal_ax.text(.98,.06,', '.join(failed)+': FAIL, no completed hold',transform=goal_ax.transAxes,ha='right',color='#a33333',fontsize=8)
            goal_ax.legend(fontsize=7,loc='upper left');goal_ax.grid(alpha=.2)
            identity='; B0/B2 CSV bytes identical' if any({p['first'].split('_')[0],p['second'].split('_')[0]}=={'B0','B2'} for p in pairs) else ''
            fig.suptitle('H4 hospital development comparison: one supplied actual outcome per method',fontsize=12)
            fig.text(.5,.018,'Development only, not independent paired repeats or final statistics'+identity+'\n'
                'FAIL traces retained in (a), not ranked as completed bars/results. Authored AABBs are not cooked/continuous proof.',
                ha='center',fontsize=8)
            fig.tight_layout(rect=(0.,.07,1.,.96))
            for image in images:
                with image.open('xb') as stream:
                    fig.savefig(stream,format=image.suffix[1:],facecolor='white',bbox_inches='tight')
        finally:plt.close(fig)
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Comparison figure input changed')
    write_exclusive(target,dict(**definition,methods=[row['method'] for row in table],
        raw_verdicts_retained=[row['raw_verdict'] for row in table],byte_identical_trace_pairs=pairs,
        all_supplied_original_FAILs_retained_in_trajectory_and_manifest=True,
        no_duplicate_trace_independence_or_paired_statistics_claimed=True,
        partial_FAILs_NOT_ranked_as_completed_performance=True,
        original_pose_bands_NOT_new_motion_or_support_admission=True,
        no_live_scene_or_timeline_access=True,new_PhysX_trial_count=0,use_for_final_statistics=False,
        output_sha256={str(path):file_sha(path) for path in images}))
    print('Saved matched completed development figure:',target)
    return target


def measured_phase_response(rows,actions):
    """Account for every logged interval without fitting or changing a reference.

    Straight/curve labels come from each method's frozen spatial actions,
    NOT feedback curvature, a chosen time window or measured performance.
    Intervals crossing labels are retained separately, not silently removed.
    Body-forward velocity is projected from world velocity, not its magnitude.
    """
    if len(rows)<2 or not actions:raise ValueError('Complete measured clock and frozen spatial actions required')
    channels=('time_s','command_speed_m_s','vx_physx_m_s','vy_physx_m_s','yaw_rad',
        'path_error_m','path_heading_error_deg')
    parsed=[]
    for row in rows:
        value={key:float(row[key]) for key in channels}
        if not all(math.isfinite(x) for x in value.values()):raise ValueError('Finite measured phase channels required')
        phase=row['phase']
        if phase not in ('pre_hold','path_tracking','terminal_braking','terminal_correction','terminal_hold'):
            raise ValueError('Explicit original H4 logged phase required')
        label=phase
        if phase=='path_tracking':
            index=float(row['action_index'])
            if not math.isfinite(index) or not index.is_integer() or not 0<=index<len(actions):
                raise ValueError('Own valid frozen action index required for tracking classification')
            curvature=float(actions[int(index)]['curvature_1_m'])
            if not math.isfinite(curvature):raise ValueError('Finite frozen spatial curvature required')
            label='tracking_straight' if curvature==0. else 'tracking_curve'
        if parsed and value['time_s']<=parsed[-1]['time_s']:raise ValueError('Strictly increasing measured clock required')
        value.update(phase=phase,label=label,body_forward_speed_m_s=
            value['vx_physx_m_s']*math.cos(value['yaw_rad'])+value['vy_physx_m_s']*math.sin(value['yaw_rad']))
        parsed.append(value)
    phase_duration=Counter();groups={}
    for first,last in zip(parsed,parsed[1:]):
        dt=last['time_s']-first['time_s'];phase_duration[last['phase']]+=dt
        key=last['label'] if first['label']==last['label'] else 'boundary_crossing'
        group=groups.setdefault(key,dict(interval_count=0,duration_s=0.,command_integral_m=0.,
            body_forward_integral_m=0.,speed_error_squared_integral=0.,position_squared_integral=0.,heading_squared_integral=0.))
        group['interval_count']+=1;group['duration_s']+=dt
        group['command_integral_m']+=dt*(first['command_speed_m_s']+last['command_speed_m_s'])/2
        group['body_forward_integral_m']+=dt*(first['body_forward_speed_m_s']+last['body_forward_speed_m_s'])/2
        group['speed_error_squared_integral']+=dt*sum((point['command_speed_m_s']-point['body_forward_speed_m_s'])**2
            for point in (first,last))/2
        group['position_squared_integral']+=dt*sum(point['path_error_m']**2 for point in (first,last))/2
        group['heading_squared_integral']+=dt*sum(point['path_heading_error_deg']**2 for point in (first,last))/2
    table=[]
    for label,group in groups.items():
        duration=group['duration_s'];tracking=label in ('tracking_straight','tracking_curve')
        table.append(dict(segment=label,interval_count=group['interval_count'],duration_s=duration,
            mean_command_speed_mm_s=1000*group['command_integral_m']/duration,
            mean_measured_body_forward_speed_mm_s=1000*group['body_forward_integral_m']/duration,
            mean_command_minus_body_forward_speed_mm_s=1000*(group['command_integral_m']-group['body_forward_integral_m'])/duration,
            time_weighted_speed_residual_RMS_mm_s=1000*math.sqrt(group['speed_error_squared_integral']/duration),
            signed_integrated_body_forward_distance_m=group['body_forward_integral_m'],
            tracking_position_RMS_mm=1000*math.sqrt(group['position_squared_integral']/duration) if tracking else None,
            tracking_heading_RMS_deg=math.sqrt(group['heading_squared_integral']/duration) if tracking else None))
    observed=parsed[-1]['time_s']-parsed[0]['time_s'];partition=math.fsum(row['duration_s'] for row in table)
    if sum(row['interval_count'] for row in table)!=len(rows)-1 or not math.isclose(partition,observed,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Every original adjacent interval must remain accounted for')
    return dict(first_recorded_time_s=parsed[0]['time_s'],last_recorded_time_s=parsed[-1]['time_s'],
        observed_duration_s=observed,partition_duration_s=partition,phase_duration_right_endpoint_s=dict(phase_duration),
        segments=table,all_original_adjacent_intervals_accounted_for=True,
        startup_before_first_CSV_sample_NOT_invented=True,
        label_rule='Exact zero/nonzero curvature of own pinned spatial action; feedback corrections do not relabel the reference.',
        boundary_rule='Every label-crossing interval retained as boundary_crossing, not attributed to either within-segment RMS.',
        integration_rule='Trapezoidal logged samples; phase duration separately assigned to right-endpoint phase.',
        different_methods_can_have_different_curve_lengths_and_states_NOT_identical_geometry=True,
        no_fit_alignment_clock_shift_or_caster_causality=True,telemetry_rows_NOT_independent_trials=True)


def write_phase_response_review(results,output=None):
    """Derived event accounting for explicit actual development outcomes only."""
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    comparison_path=write_explicit_development_comparison(results)
    comparison=json.loads(comparison_path.read_bytes())
    pins={**comparison['input_sha256'],str(comparison_path):file_sha(comparison_path)};runs=[];table=[]
    for item in comparison['actual_development_rows']:
        result=Path(item['result']);raw=json.loads(result.read_bytes())
        plan=json.loads(Path(raw['plan_path']).read_bytes())
        with result.with_suffix('.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        review=measured_phase_response(rows,plan['selected_actions'])
        runs.append(dict(run_id=raw['run_id'],method=raw['method'],raw_verdict=raw['verdict'],
            complete_fixed_hold=item['complete_fixed_hold'],terminal_pulse_count=item['terminal_pulse_count'],
            result=str(result),response=review))
        table.extend(dict(run_id=raw['run_id'],method=raw['method'],**row) for row in review['segments'])
    difference=None
    p=[r for r in runs if r['method']=='P_dynamics_full_envelope'];b=[r for r in runs if r['method']=='B2_multimode_transition_cost_ramp']
    if comparison['configuration_matching_checks_passed'] and len(p)==len(b)==1 and all(
            r['raw_verdict']=='PASS' and r['complete_fixed_hold'] for r in (*p,*b)):
        first,last=(r['response']['phase_duration_right_endpoint_s'] for r in (p[0],b[0]))
        difference={phase:first.get(phase,0.)-last.get(phase,0.) for phase in sorted(set(first)|set(last))}
    definition=dict(version='actual_H4_segment_response_and_terminal_event_accounting_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_phase_response_review).encode()).hexdigest(),
        metric_function_sha256=hashlib.sha256(inspect.getsource(measured_phase_response).encode()).hexdigest())
    output=Path(output) if output is not None else comparison_path.parent/'phase_response_review'
    target=output/'phase_response_review.json';csv_path=output/'measured_phase_response.csv'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()) or file_sha(csv_path)!=saved['output_sha256'][str(csv_path)]:
            raise ValueError('Existing derived phase review changed; no overwrite')
        return target
    if any(file_sha(Path(p))!=sha for p,sha in pins.items()):raise ValueError('Original comparison input changed during phase review')
    output.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if any(file_sha(Path(p))!=sha for p,sha in pins.items()):raise ValueError('Original input changed during phase export')
    write_exclusive(target,dict(**definition,status='MEASURED_PHASE_RESPONSE_DIAGNOSIS_NOT_CAUSAL_OR_FINAL_VALIDATION',
        configuration_matching_checks_passed=comparison['configuration_matching_checks_passed'],runs=runs,
        P_minus_B2_phase_duration_s=difference,
        phase_differences_are_event_accounting_NOT_isolated_controller_or_model_causality=True,
        original_PASS_FAILs_and_entire_measured_traces_retained=True,
        no_response_parameter_gain_guard_or_original_result_change=True,
        no_live_stage_or_timeline_access=True,new_PhysX_trial_count=0,use_for_final_statistics=False,
        output_sha256={str(csv_path):file_sha(csv_path)}))
    print('Saved measured phase response review:',target)
    return target


def measured_orientation_segments(rows,plan):
    """Decompose signed heading errors on the OWN frozen reference, no fit.

    Contiguous groups use original spatial mode/curvature/direction, not a
    selected time window or feedback command. Logged progress is sampled as
    recorded; this is not a new closest-point projection or a time alignment.
    Joint target discrepancies are observations, NOT isolated force causes.
    """
    from hospital_h4_path_execution import PathReference,wrap
    reference=PathReference(plan);groups=[];previous_time=None;tracking_count=0;tracking_contiguous=False
    previous_progress=None
    known={'pre_hold','path_tracking','terminal_braking','terminal_correction','terminal_hold'}
    channels=('time_s','yaw_rad','path_progress_m','goal_heading_error_deg','path_heading_error_deg',
        'front_rad','rear_rad','target_front_rad','target_rear_rad')
    for row in rows:
        time=float(row['time_s']);phase=row['phase']
        if not math.isfinite(time) or (previous_time is not None and time<=previous_time) or phase not in known:
            raise ValueError('Finite monotone clock and original H4 phase required')
        previous_time=time
        if phase!='path_tracking':
            tracking_contiguous=False
            continue
        point={key:float(row[key]) for key in channels}
        index=float(row['action_index'])
        if (not all(math.isfinite(x) for x in point.values()) or not math.isfinite(index)
                or not index.is_integer() or not 0<=index<len(reference.actions)
                or not 0<=point['path_progress_m']<=reference.arc[-1]
                or previous_progress is not None and point['path_progress_m']<previous_progress):
            raise ValueError('Finite tracking channels and valid own action/progress required')
        previous_progress=point['path_progress_m']
        action=reference.actions[int(index)]
        key=(action['mode'],action['curvature_1_m'],action['direction'])
        _,pose=reference.at(point['path_progress_m']);actual=point['yaw_rad'];goal=plan['goal_world_xyyaw'][2]
        goal_minus_reference=wrap(goal-pose[2]);reference_minus_actual=wrap(pose[2]-actual)
        goal_minus_actual=wrap(goal-actual)
        if abs(wrap(math.radians(point['goal_heading_error_deg'])-goal_minus_actual))>1e-8:
            raise ValueError('Logged goal heading differs from its own saved pose/frame')
        point.update(action_index=int(index),reference_yaw_rad=pose[2],
            goal_minus_reference_deg=math.degrees(goal_minus_reference),
            reference_minus_actual_deg=math.degrees(reference_minus_actual),
            recomputed_goal_minus_actual_deg=math.degrees(goal_minus_actual),
            wrapped_decomposition_closure_deg=math.degrees(wrap(goal_minus_reference+reference_minus_actual-goal_minus_actual)),
            logged_minus_progress_sampled_path_heading_deg=math.degrees(wrap(
                math.radians(point['path_heading_error_deg'])-reference_minus_actual)),
            front_actual_minus_target_deg=math.degrees(point['front_rad']-point['target_front_rad']),
            rear_actual_minus_target_deg=math.degrees(point['rear_rad']-point['target_rear_rad']))
        if not tracking_contiguous or groups[-1]['key']!=key:
            groups.append(dict(key=key,points=[]))
        groups[-1]['points'].append(point);tracking_count+=1;tracking_contiguous=True
    if len(rows)<2 or tracking_count<2:raise ValueError('At least two actual tracking samples required')
    segments=[]
    for group in groups:
        points=group['points'];start,end=points[0],points[-1];duration=end['time_s']-start['time_s']
        metrics={}
        for channel in ('front_actual_minus_target_deg','rear_actual_minus_target_deg'):
            area=math.fsum((b['time_s']-a['time_s'])*(a[channel]+b[channel])/2 for a,b in zip(points,points[1:]))
            square=math.fsum((b['time_s']-a['time_s'])*(a[channel]**2+b[channel]**2)/2 for a,b in zip(points,points[1:]))
            metrics[channel]=dict(time_weighted_mean=area/duration if duration else None,
                time_weighted_RMS=math.sqrt(square/duration) if duration else None,
                maximum_sampled_absolute=max(abs(p[channel]) for p in points))
        segments.append(dict(segment_index=len(segments),mode=group['key'][0],frozen_curvature_1_m=group['key'][1],
            direction=group['key'][2],sample_count=len(points),observed_within_segment_duration_s=duration,
            first=start,last=end,steering_target_tracking_observations=metrics,
            maximum_absolute_wrapped_decomposition_closure_deg=max(abs(p['wrapped_decomposition_closure_deg']) for p in points),
            maximum_absolute_logged_vs_progress_sampled_path_heading_deg=max(abs(p['logged_minus_progress_sampled_path_heading_deg']) for p in points)))
    return dict(tracking_sample_count=tracking_count,other_original_phase_sample_count=len(rows)-tracking_count,
        segments=segments,all_tracking_samples_accounted_for=sum(s['sample_count'] for s in segments)==tracking_count,
        source_label_rule='Contiguous own frozen spatial mode/curvature/direction; no best-time or measured-performance segmentation.',
        signed_heading_identity='wrap(goal-reference + reference-actual) = wrap(goal-actual)',
        logged_progress_used_without_new_projection_or_time_shift=True,
        logged_reference_error_can_precede_current_logged_progress_one_control_update=True,
        target_tracking_observations_NOT_identical_input_force_load_or_caster_causality=True,
        target_values_NOT_reassigned_or_used_to_fit_a_new_model=True)


def write_orientation_response_review(results,output=None):
    """Append-only measured reference/heading/steering diagnostic, not a trial."""
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    comparison_path=write_explicit_development_comparison(results)
    comparison=json.loads(comparison_path.read_bytes());runs=[];table=[]
    pins={**comparison['input_sha256'],str(comparison_path):file_sha(comparison_path)}
    for item in comparison['actual_development_rows']:
        result=Path(item['result']);raw=json.loads(result.read_bytes())
        plan=json.loads(Path(raw['plan_path']).read_bytes())
        with result.with_suffix('.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        review=measured_orientation_segments(rows,plan)
        runs.append(dict(run_id=raw['run_id'],method=raw['method'],raw_verdict=raw['verdict'],
            complete_fixed_hold=item['complete_fixed_hold'],selected_candidate_id=plan['selected_candidate_id'],
            original_handoff=raw.get('terminal_handoff'),orientation=review))
        for segment in review['segments']:
            first,last=segment['first'],segment['last']
            table.append(dict(run_id=raw['run_id'],method=raw['method'],raw_verdict=raw['verdict'],
                segment_index=segment['segment_index'],mode=segment['mode'],frozen_curvature_1_m=segment['frozen_curvature_1_m'],
                sample_count=segment['sample_count'],first_time_s=first['time_s'],last_time_s=last['time_s'],
                final_goal_minus_reference_deg=last['goal_minus_reference_deg'],
                final_reference_minus_actual_deg=last['reference_minus_actual_deg'],
                final_goal_minus_actual_deg=last['recomputed_goal_minus_actual_deg'],
                front_actual_minus_target_RMS_deg=segment['steering_target_tracking_observations']['front_actual_minus_target_deg']['time_weighted_RMS'],
                rear_actual_minus_target_RMS_deg=segment['steering_target_tracking_observations']['rear_actual_minus_target_deg']['time_weighted_RMS']))
    definition=dict(version='actual_own_reference_heading_and_steering_segment_diagnostic_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_orientation_response_review).encode()).hexdigest(),
        metric_function_sha256=hashlib.sha256(inspect.getsource(measured_orientation_segments).encode()).hexdigest())
    output=(Path(output) if output is not None else comparison_path.parent/'orientation_response_review').resolve()
    target=output/'orientation_response_review.json';csv_path=output/'measured_orientation_segments.csv'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (any(saved.get(k)!=v for k,v in definition.items()) or not csv_path.is_file()
                or file_sha(csv_path)!=saved['output_sha256'].get(str(csv_path))):
            raise ValueError('Existing orientation review differs; no overwrite')
        return target
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Orientation input changed')
    output.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Orientation input changed during export')
    write_exclusive(target,dict(**definition,status='MEASURED_ORIENTATION_DIAGNOSTIC_NOT_ISOLATED_CAUSALITY_OR_FINAL_VALIDATION',
        configuration_matching_checks_passed=comparison['configuration_matching_checks_passed'],
        physical_control_or_domain_mismatches=comparison['physical_control_or_domain_mismatches'],runs=runs,
        all_supplied_original_PASS_FAILs_retained=True,partial_FAILs_NOT_ranked_as_completed=True,
        geometric_heading_decomposition_is_NOT_a_physical_cause_or_predictive_fit=True,
        no_control_gain_physics_pose_caster_guard_or_original_result_change=True,
        no_live_stage_or_timeline_access=True,new_PhysX_trial_count=0,use_for_final_statistics=False,
        output_sha256={str(csv_path):file_sha(csv_path)}))
    print('Saved measured own-reference orientation diagnosis:',target)
    return target


def write_development_manuscript_table(results,output=None):
    """Explicit development rows for manuscript assembly, not final evidence.

    Never choose a best/latest trial or rank partial failed trajectories.
    Older H1/H2 CSVs have no original trace pin: retain that provenance limit,
    snapshot current bytes and do not invent historical hash verification.
    """
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    paths=[Path(path).resolve() for path in results]
    if not paths or len(set(paths))!=len(paths):raise ValueError('Explicit distinct actual development results required')
    rows=[];pins={};identities=set();run_ids=set()
    for path in paths:
        record=inspect_execution_record(path,'method_path');raw=json.loads(path.read_bytes())
        scene=str(record['scenario']).split('_',1)[0];method=record['method'];metrics=record['trace_observations']
        if (scene not in ('H1','H2','H4') or method not in METHODS
                or record['outcome_stage']!='ACTUAL_PATH_OUTCOME' or record['integrity_errors']
                or metrics['sample_count']<2 or record['trace_pin_status'] not in
                    ('ORIGINAL_TRACE_HASH_VERIFIED','LEGACY_UNPINNED_TRACE_CURRENT_SNAPSHOT_ONLY')
                or raw.get('use_for_final_statistics') is not False):
            raise ValueError('Intact explicit actual DEVELOPMENT path outcomes only; no setup, nominal or final relabelling')
        if (scene,method) in identities or record['run_id'] in run_ids:
            raise ValueError('One explicit development outcome per task/method; no repeat pooling or best-run selection')
        identities.add((scene,method));run_ids.add(record['run_id'])
        plan_path=Path(raw['plan_path']);plan=json.loads(plan_path.read_bytes())
        if file_sha(plan_path)!=raw['plan_sha256'] or plan.get('method')!=method:
            raise ValueError('Own original method and plan identity required for manuscript table')
        completed=record['reported_verdict']=='PASS' and metrics['fixed_hold_and_tail_observation_complete']
        if record['reported_verdict']=='PASS' and not completed:raise ValueError('PASS requires its own complete fixed hold')
        if completed and any(value is None for value in (metrics['last_physics_time_s'],
                metrics['tracking_only_time_weighted_position_RMS_mm'],record['reported_position_error_mm'],
                record['reported_heading_error_deg'])):
            raise ValueError('Completed rows require their actual clock, tracking and terminal metrics')
        row=dict(task=scene,method=method,run_id=record['run_id'],cohort=record['execution_cohort'],
            raw_verdict=record['reported_verdict'],complete_fixed_hold=metrics['fixed_hold_and_tail_observation_complete'],
            successful_route_clock_end_s=metrics['last_physics_time_s'] if completed else None,
            observed_until_last_CSV_sample_s=metrics['last_physics_time_s'],
            completed_tracking_position_RMS_mm=metrics['tracking_only_time_weighted_position_RMS_mm'] if completed else None,
            partial_trace_tracking_position_RMS_mm=metrics['tracking_only_time_weighted_position_RMS_mm'] if not completed else None,
            terminal_position_error_mm=record['reported_position_error_mm'] if completed else None,
            terminal_absolute_heading_error_deg=abs(record['reported_heading_error_deg']) if completed else None,
            original_trace_provenance=record['trace_pin_status'],original_error=record['reported_error'],result=str(path))
        if any(isinstance(value,float) and not math.isfinite(value) for value in row.values()):
            raise ValueError('Finite report metrics required')
        rows.append(row)
        pins.update({str(item):file_sha(item) for item in (path,path.with_suffix('.csv'),plan_path)})
    definition=dict(version='explicit_hospital_development_manuscript_table_NOT_final_comparison_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_development_manuscript_table).encode()).hexdigest(),
        metric_function_sha256=hashlib.sha256(inspect.getsource(trace_metrics).encode()).hexdigest())
    identity=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=Path(output) if output is not None else RESULT/'paper_evidence_inventory/manuscript_tables'
    stem=output/('hospital_development_'+identity)
    target=stem.with_suffix('.json');csv_path=stem.with_suffix('.csv');tex_path=stem.with_suffix('.tex')
    if target.exists():
        saved=json.loads(target.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()) or any(
                not Path(path).exists() or file_sha(Path(path))!=sha for path,sha in saved['output_sha256'].items()):
            raise ValueError('Existing manuscript development table changed; no overwrite')
        return target
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Explicit raw input changed during table review')
    output.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[r'% Derived from the adjacent JSON provenance manifest; not final repeated evidence.',
        r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Individual hospital development executions. These are not independent paired repetitions or final comparative statistics.}',
        r'\label{tab:hospital-development}',r'\begin{tabular}{lllrrrr}',r'\hline',
        r'Task & Method & Result & $t_{\rm end}$ (s) & Tracking RMS (mm) & Goal error (mm) & $|e_\psi|$ ($^\circ$) \\',r'\hline']
    def cell(value,digits=2):return '--' if value is None else f'{value:.{digits}f}'
    for row in rows:
        lines.append(' & '.join((row['task'],row['method'].split('_',1)[0],row['raw_verdict'],
            cell(row['successful_route_clock_end_s']),cell(row['completed_tracking_position_RMS_mm']),
            cell(row['terminal_position_error_mm']),cell(row['terminal_absolute_heading_error_deg'],3)))+r' \\')
    lines.extend((r'\hline',r'\end{tabular}',r'\par\smallskip',r'\parbox{\linewidth}{\footnotesize '
        r'Time is the recorded route-clock end, not wall time. Tracking excludes terminal correction. '
        r'Failed partial traces are not ranked as completed runs. H1/H2 legacy CSVs are current-byte snapshots without original trace hashes; '
        r'H4 uses pinned traces. Conditions and entry histories are not established as paired across methods. '
        r'Only the explicitly listed development cohorts are shown; older environment/time-limit outcomes are preserved separately.}',
        r'\end{table}'))
    with tex_path.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Raw input changed during manuscript table export')
    write_exclusive(target,dict(**definition,status='MANUSCRIPT_ASSEMBLY_DEVELOPMENT_TABLE_NOT_FINAL_EVALUATION',rows=rows,
        explicit_input_selection_NOT_best_or_latest_search=True,
        no_historical_hash_invented_for_legacy_traces=True,failed_partial_time_or_tracking_NOT_ranked_as_success=True,
        independent_paired_repeats_verified=False,statistical_superiority_established=False,
        identical_physical_conditions_across_all_rows_verified=False,
        qualifying_final_comparison_cells_proven=0,no_new_PhysX_or_live_stage_access=True,
        use_for_final_statistics=False,output_sha256={str(path):file_sha(path) for path in (csv_path,tex_path)}))
    print('Saved manuscript development table:',target)
    return target


def terminal_reference_geometry(plan,remaining_m,position_band_m,heading_band_deg):
    """Pose-only samples of the original reference, NOT a stopping prediction."""
    from hospital_h4_path_execution import PathReference
    values=(remaining_m,position_band_m,heading_band_deg)
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
        raise ValueError('Finite original handoff distance and pose bands required')
    reference=PathReference(plan)
    if not 0<=remaining_m<=reference.arc[-1] or position_band_m<=0 or heading_band_deg<=0:
        raise ValueError('Original bounded reference and positive pose bands required')
    goal=plan['goal_world_xyyaw']
    def describe(pose):
        distance=math.dist(pose[:2],goal[:2])
        angle=abs(math.degrees(math.atan2(math.sin(pose[2]-goal[2]),math.cos(pose[2]-goal[2]))))
        return dict(world_xyyaw=list(pose),goal_position_error_mm=1000*distance,
            goal_absolute_heading_error_deg=angle,
            inside_stationary_pose_band_only=distance<=position_band_m and angle<=heading_band_deg)
    _,sample=reference.at(reference.arc[-1]-remaining_m)
    return dict(reference_arc_m=reference.arc[-1],remaining_reference_distance_m=remaining_m,
        planned_endpoint=describe(reference.poses[-1]),ideal_on_reference_handoff_pose=describe(sample),
        velocity_braking_steering_tracking_errors_and_contact_NOT_simulated=True,
        outside_pose_band_is_NOT_an_original_motion_FAIL=True,
        inside_pose_band_is_NOT_a_measured_stop_or_fixed_hold_PASS=True,
        reference_geometry_is_NOT_autonomous_response_model_validation=True)


def write_terminal_reference_review(results,output=None):
    """Audit the same reference/terminal interface for ALL FOUR frozen plans.

    This is a post-development diagnostic, not a changed admission criterion,
    a fitted cost, or a replacement for the original common terminal policy.
    """
    import ast
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    comparison_path=write_explicit_development_comparison(results)
    comparison=json.loads(comparison_path.read_bytes())
    if not comparison['actual_development_rows'] or comparison['physical_control_or_domain_mismatches']:
        raise ValueError('Intact own-version actual outcomes required; no mixed physical/control/planner contracts')
    pins={**comparison['input_sha256'],str(comparison_path):file_sha(comparison_path)}
    raws={row['method']:json.loads(Path(row['result']).read_bytes()) for row in comparison['actual_development_rows']}
    if len(raws)!=len(comparison['actual_development_rows']):raise ValueError('No repeated-method or selected-best handoff review')
    source=_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_path_execution.py');source_sha=file_sha(source)
    if any(raw['source_sha256'].get(source.name)!=source_sha for raw in raws.values()):
        raise ValueError('Exact shared executed tracking source required for the handoff constant')
    tree=ast.parse(source.read_text());thresholds=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='remaining'
                and len(node.ops)==len(node.comparators)==1 and isinstance(node.ops[0],ast.LtE)
                and isinstance(node.comparators[0],ast.Constant)):
            thresholds.append(node.comparators[0].value)
    if len(thresholds)!=1 or type(thresholds[0]) not in (float,int):raise ValueError('One explicit original remaining-distance handoff literal required')
    remaining=thresholds[0];first=next(iter(raws.values()));config=first['terminal_controller_config']
    position=config['stationary_hold_entry_position_m'];heading=config['stationary_hold_entry_heading_deg']
    if position!=first['acceptance']['position_error_m'] or heading!=first['acceptance']['heading_error_deg']:
        raise ValueError('Original stationary pose and final acceptance bands must agree')
    bundle_path=Path(first['plan_path']).resolve().parent/'execution_bundle.json'
    bundle=json.loads(bundle_path.read_bytes());pins.update({str(source):source_sha,str(bundle_path):file_sha(bundle_path)})
    if any(raw['input_sha256'].get(str(bundle_path))!=file_sha(bundle_path)
            or Path(raw['plan_path']).resolve().parent!=bundle_path.parent for raw in raws.values()):
        raise ValueError('Same already executed own frozen four-method bundle required; no mixed planner versions')
    rows=[];seen=set()
    for path in map(Path,bundle['plans']):
        if file_sha(path)!=bundle['output_sha256'].get(str(path)):raise ValueError('Original frozen plan changed')
        plan=json.loads(path.read_bytes());method=plan['method']
        if method not in METHODS or method in seen or plan['goal_world_xyyaw']!=first['goal_world_xyyaw']:
            raise ValueError('All distinct original method references and same task goal required')
        seen.add(method);pins[str(path)]=file_sha(path)
        raw=raws.get(method);handoff=raw.get('terminal_handoff') if raw else None
        if raw and (Path(raw['plan_path']).resolve()!=path.resolve() or raw['plan_sha256']!=file_sha(path)):
            raise ValueError('Actual handoff must name its own exact selected reference, not a prior planner version')
        rows.append(dict(method=method,selected_candidate_id=plan['selected_candidate_id'],plan=str(path),
            reference_geometry=terminal_reference_geometry(plan,remaining,position,heading),
            actual_result_supplied=raw is not None,original_actual_verdict=raw['verdict'] if raw else None,
            actual_handoff=handoff,actual_terminal_pulse_count=raw.get('terminal_pulse_count') if raw else None))
    if seen!=set(METHODS):raise ValueError('No selected-method-only terminal geometry screen')
    definition=dict(version='H4_ALL_FOUR_frozen_reference_terminal_pose_interface_diagnostic_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_terminal_reference_review).encode()).hexdigest(),
        metric_function_sha256=hashlib.sha256(inspect.getsource(terminal_reference_geometry).encode()).hexdigest())
    output=Path(output) if output is not None else comparison_path.parent/'terminal_reference_review'
    target=output/'terminal_reference_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()):raise ValueError('Existing reference review changed; no overwrite')
        return target
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Original input changed during reference diagnosis')
    write_exclusive(target,dict(**definition,status='POSE_INTERFACE_DIAGNOSIS_NOT_CHANGED_POLICY_OR_MODEL_ADMISSION',
        original_remaining_distance_literal_m=remaining,original_stationary_position_band_m=position,
        original_stationary_heading_band_deg=heading,methods=rows,
        all_four_original_references_retained_NOT_P_only_tolerance_or_candidate_filter=True,
        future_planner_cost_or_goal_connection_amendment_NOT_adopted=not bundle.get('additional_common_pose_interface_predicate_is_NEW_planning_version',False),
        own_frozen_bundle=str(bundle_path),this_review_does_NOT_change_or_adopt_any_plan_controller_or_rule=True,
        no_fitted_response_new_model_steering_seed_or_physics_change=True,
        no_tracking_or_terminal_controller_or_guard_change=True,
        actual_outcomes_NOT_relabelled_or_isolated_causality_claimed=True,
        no_live_stage_or_timeline_access=True,new_PhysX_trial_count=0,use_for_final_statistics=False))
    print('Saved all-four terminal reference geometry review:',target)
    return target


def choose_terminal_connection_candidate(rows):
    """Prospective common pose-interface filter; original native cost unchanged.

    This chooses an offline hypothesis, never a measured best trial. The
    geometry predicate is not a stopping/steering/uncertainty certificate.
    """
    if not rows or len({row['candidate_id'] for row in rows})!=len(rows):
        raise ValueError('Complete distinct candidate identities required')
    admitted=[]
    for row in rows:
        if row['original_native_admitted'] is not True:continue
        if type(row.get('original_native_cost_s')) not in (int,float) or not math.isfinite(row['original_native_cost_s']):
            raise ValueError('Original finite native cost required; no fitted or replacement cost')
        geometry=row['reference_geometry']
        if geometry['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only']:
            admitted.append(row)
    if not admitted:return None
    return min(admitted,key=lambda row:(row['original_native_cost_s'],
        row['reference_geometry']['planned_endpoint']['goal_position_error_mm'],row['candidate_id']))


def common_terminal_margin_sensitivity(candidate):
    """Fixed all-four pose-margin grid on the SAME original 26 decisions.

    This reports every cell, including no-candidate outcomes. It does not
    choose a policy, change acceptance, estimate response bounds, or use any
    actual result to fit a margin. Costs and the original tie break stay fixed.
    """
    if (candidate.get('version')!='prospective_common_terminal_pose_interface_candidate_v1'
            or candidate.get('common_remaining_reference_distance_m')!=.025
            or candidate.get('common_pose_band_m')!=.030 or candidate.get('common_heading_band_deg')!=2.):
        raise ValueError('Own original 25mm interface and 30mm/2deg candidate archive required')
    methods=candidate['methods']
    if len(methods)!=4 or {m['method'] for m in methods}!=set(METHODS):
        raise ValueError('Every original method required, not a favorable subset')
    pools=[];table=[]
    for method in methods:
        rows=method['complete_original_candidate_decisions'];ids=[row['candidate_id'] for row in rows]
        if len(rows)!=26 or len(set(ids))!=26:raise ValueError('Exactly 26 distinct original candidate identities required')
        pools.append(set(ids));admitted=[]
        for row in rows:
            if type(row['original_native_admitted']) is not bool:raise ValueError('Exact native admission decision required')
            if not row['original_native_admitted']:
                if row['original_native_cost_s'] is not None or row['reference_geometry'] is not None:
                    raise ValueError('No replacement geometry or cost for rejected native candidates')
                continue
            cost=row['original_native_cost_s']
            if type(cost) not in (float,int) or not math.isfinite(cost) or cost<0:
                raise ValueError('Original finite nonnegative native cost required')
            for point in ('planned_endpoint','ideal_on_reference_handoff_pose'):
                values=[row['reference_geometry'][point][key] for key in ('goal_position_error_mm','goal_absolute_heading_error_deg')]
                if any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in values):
                    raise ValueError('Own finite pose-only errors required; no fitted response replacement')
            if row['reference_geometry']['remaining_reference_distance_m']!=.025:
                raise ValueError('Same original interface distance required')
            admitted.append(row)
        for position_mm in (10.,20.,30.):
            for heading_deg in (.5,1.,1.5,2.):
                eligible=[row for row in admitted
                    if row['reference_geometry']['ideal_on_reference_handoff_pose']['goal_position_error_mm']<=position_mm
                    and row['reference_geometry']['ideal_on_reference_handoff_pose']['goal_absolute_heading_error_deg']<=heading_deg]
                ordered=sorted(eligible,key=lambda row:(row['original_native_cost_s'],
                    row['reference_geometry']['planned_endpoint']['goal_position_error_mm'],row['candidate_id']))
                selected=ordered[0] if ordered else None
                if (position_mm,heading_deg)==(30.,2.) and selected!=method['candidate_selected_by_original_native_cost']:
                    raise ValueError('Sensitivity must exactly reproduce the original common-interface choice')
                table.append(dict(method=method['method'],pose_position_limit_mm=position_mm,pose_heading_limit_deg=heading_deg,
                    eligible_candidate_count=len(eligible),eligible_candidate_ids=[row['candidate_id'] for row in ordered],
                    selected_candidate_id=selected['candidate_id'] if selected else None,
                    original_native_cost_s=selected['original_native_cost_s'] if selected else None,
                    selected_ideal_handoff=selected['reference_geometry']['ideal_on_reference_handoff_pose'] if selected else None))
    if any(pool!=pools[0] for pool in pools):raise ValueError('All methods must retain the SAME complete original candidate pool')
    return table


def write_common_terminal_margin_sensitivity(candidate_path,output=None):
    """Append-only pose feasibility sensitivity, never a new execution plan."""
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    candidate_path=Path(candidate_path).resolve();candidate=json.loads(candidate_path.read_bytes())
    pins={**candidate['input_sha256'],str(candidate_path):file_sha(candidate_path)}
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Original candidate evidence changed')
    table=common_terminal_margin_sensitivity(candidate)
    definition=dict(version='ALL_FOUR_fixed_pose_margin_grid_on_original_26_candidates_v1',input_sha256=pins,
        metric_function_sha256=hashlib.sha256(inspect.getsource(common_terminal_margin_sensitivity).encode()).hexdigest(),
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_common_terminal_margin_sensitivity).encode()).hexdigest())
    output=(Path(output) if output is not None else candidate_path.parent/'offline_margin_sensitivity').resolve()
    target=output/'margin_sensitivity.json';csv_path=output/'all_four_margin_grid.csv'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (any(saved.get(k)!=v for k,v in definition.items()) or not csv_path.is_file()
                or file_sha(csv_path)!=saved['output_sha256'].get(str(csv_path))):
            raise ValueError('Existing margin sensitivity differs; no overwrite')
        return target
    output.mkdir(parents=True,exist_ok=True)
    columns=('method','pose_position_limit_mm','pose_heading_limit_deg','eligible_candidate_count',
        'selected_candidate_id','original_native_cost_s')
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({key:row[key] for key in columns} for row in table)
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Original candidate input changed during export')
    write_exclusive(target,dict(**definition,status='POSE_MARGIN_SENSITIVITY_NOT_SELECTED_POLICY_OR_RESPONSE_BOUND',
        cells=table,grid_cell_count=len(table),original_pool_size_per_method=26,
        same_fixed_position_heading_grid_for_every_method=True,all_no_candidate_cells_retained=True,
        unchanged_original_native_admission_cost_and_tie_break=True,
        original_30mm_2deg_candidate_selections_exactly_reproduced=True,
        measured_outcomes_NOT_inputs_to_this_geometric_sensitivity=True,
        post_development_analysis_NOT_pre_registered_or_independent_validation=True,
        geometric_margin_is_NOT_a_measured_braking_steering_uncertainty_or_fixed_hold_bound=True,
        no_new_policy_selected_or_executor_plan_acceptance_physics_changed=True,
        no_live_stage_or_timeline_access=True,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        output_sha256={str(csv_path):file_sha(csv_path)}))
    print('Saved all-four fixed geometric margin sensitivity:',target)
    return target


def write_terminal_connection_candidate(output=None):
    """Try a new common planning-interface screen without changing old plans.

    All four methods receive the same original 26 candidates, their OWN
    original propagator/cost/admission, and the same 25 mm reference sample
    and 30 mm/2 deg pose predicate. No measured response is used to choose a
    candidate or tune constants. Existing outcomes are diagnostic evidence,
    not an independent test set for this post-development amendment.
    """
    import copy
    import inspect
    import numpy as np
    import screen_hospital_h4_model_seed_candidates as native
    import hospital_h4_path_execution as execution
    from hospital_runtime_evidence_io import write_exclusive
    bundle_path=RESULT/'hospital_h4_execution_bundle/bundle_r01/execution_bundle.json'
    bundle=json.loads(bundle_path.read_bytes());data=json.loads(native.TARGET.read_bytes())
    native.previous.verify(data['input_sha256']);native.previous.verify(bundle['output_sha256'])
    setup,_,origin,resolution,model,pins=native.previous.load_inputs()
    with np.load(native.map_review.GRID) as grid:occupied=grid['occupied'].copy()
    checker=native.previous.HospitalEnvelopeChecker(occupied,origin,resolution,setup['start_xyyaw'])
    plans={json.loads(Path(path).read_bytes())['method']:json.loads(Path(path).read_bytes()) for path in bundle['plans']}
    if set(plans)!=set(METHODS) or len(data['candidate_pool'])!=26:
        raise ValueError('Same complete original four-method/26-candidate experiment required')
    original={row['method']:row for row in data['methods']}
    remaining=.025;position=execution.ACCEPTANCE['position_error_m'];heading=execution.ACCEPTANCE['heading_error_deg']
    source=_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_path_execution.py')
    # Bind to the original executed source literal, not a newly optimized threshold.
    import ast
    thresholds=[node.comparators[0].value for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='remaining'
        and len(node.ops)==len(node.comparators)==1 and isinstance(node.ops[0],ast.LtE)
        and isinstance(node.comparators[0],ast.Constant)]
    if thresholds!=[remaining] or (position,heading)!=(.030,2.):
        raise ValueError('Original handoff/pose-band definitions required, not a tuned tolerance')
    pins.update({**data['input_sha256'],str(native.TARGET):file_sha(native.TARGET),str(bundle_path):file_sha(bundle_path),
        **{str(Path(path)):file_sha(path) for path in bundle['plans']}})
    definition=dict(version='prospective_common_terminal_pose_interface_candidate_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_terminal_connection_candidate).encode()).hexdigest(),
        selector_function_sha256=hashlib.sha256(inspect.getsource(choose_terminal_connection_candidate).encode()).hexdigest(),
        reference_function_sha256=hashlib.sha256(inspect.getsource(terminal_reference_geometry).encode()).hexdigest())
    key=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else RESULT/'hospital_h4_terminal_connection_candidates'/key).resolve()
    target=output/'terminal_connection_candidate.json'
    if target.exists():
        old=json.loads(target.read_bytes())
        if any(old.get(k)!=v for k,v in definition.items()):raise ValueError('Existing connection candidate differs; no overwrite')
        return target
    methods=[]
    for method in METHODS:
        evaluations={row['candidate_id']:row for row in original[method]['native_candidate_evaluations']}
        if set(evaluations)!={item['candidate_id'] for item in data['candidate_pool']}:
            raise ValueError('No omitted original native candidate decision')
        rows=[];predictions={}
        for item in data['candidate_pool']:
            expected=evaluations[item['candidate_id']]
            row=dict(candidate_id=item['candidate_id'],original_native_admitted=expected['admitted_by_native_model'],
                original_native_cost_s=None,reference_geometry=None)
            if expected['admitted_by_native_model']:
                full=method==METHODS[-1]
                if full:
                    result=native.full_rollout(item['actions'],model,checker)
                    summary=native.summary_of(result,setup['local_goal_xyyaw'])
                    cost,states,speeds=result['cost_s'],result['states'],result['speeds']
                    clear=result['sampled_full_envelope_clear']
                else:
                    dynamics=native.previous.shared.METHODS[method][1]
                    cost,states,poses=native.previous.replay(item['actions'],dynamics)
                    distance,angle=native.previous.terminal_errors(states[-1],setup['local_goal_xyyaw'])
                    summary=dict(position_error_m=distance,heading_error_deg=math.degrees(angle),terminal_speed_m_s=None)
                    clear=not checker([(0.,0.,0.),*poses]);speeds=[native.previous.shared.SPEED]*len(states)
                if not clear or any(expected.get(name)!=value for name,value in summary.items()):
                    raise ValueError('Original native response/admission must reproduce exactly: '+method+' '+item['candidate_id'])
                candidate=copy.deepcopy(plans[method])
                candidate.update(selected_candidate_id=item['candidate_id'],
                    selected_actions=[{**action,'target_speed_m_s':speed} for action,speed in zip(item['actions'],speeds)],
                    selected_states=[native.previous.shared.state_record(state,setup['start_xyyaw'],full) for state in states])
                geometry=terminal_reference_geometry(candidate,remaining,position,heading)
                row.update(original_native_cost_s=cost,reference_geometry=geometry)
                predictions[item['candidate_id']]=dict(selected_actions=candidate['selected_actions'],
                    selected_states=candidate['selected_states'])
            rows.append(row)
        admitted=[row for row in rows if row['original_native_admitted']]
        before=min(admitted,key=lambda row:(row['original_native_cost_s'],row['reference_geometry']['planned_endpoint']['goal_position_error_mm'],row['candidate_id']))
        if (before['candidate_id']!=original[method]['selected_candidate_id']
                or before['original_native_cost_s']!=original[method]['native_cost_s']):
            raise ValueError('Unchanged original cost/selector must reproduce before adding the prospective filter')
        selected=choose_terminal_connection_candidate(rows)
        methods.append(dict(method=method,original_selected_candidate_id=before['candidate_id'],
            original_selected_reference_geometry=before['reference_geometry'],
            complete_original_candidate_decisions=rows,
            common_interface_candidate_count=sum(row['original_native_admitted'] and
                row['reference_geometry']['ideal_on_reference_handoff_pose']['inside_stationary_pose_band_only'] for row in rows),
            candidate_selected_by_original_native_cost=selected,
            selected_candidate_prediction=predictions[selected['candidate_id']] if selected else None,
            changed_from_original_selection=selected is not None and selected['candidate_id']!=before['candidate_id']))
        print('Offline connection candidate:',method,selected['candidate_id'] if selected else 'NONE',flush=True)
    native.previous.verify(pins)
    write_exclusive(target,dict(**definition,status='POST_DEVELOPMENT_OFFLINE_PLANNER_CANDIDATE_NOT_ADOPTED_OR_DRIVEN',
        common_remaining_reference_distance_m=remaining,common_pose_band_m=position,common_heading_band_deg=heading,
        methods=methods,same_original_pool_and_own_native_cost_for_all_four=True,
        original_native_selected_decisions_and_costs_exactly_reproduced=True,
        additional_common_pose_interface_predicate_is_a_NEW_planning_candidate=True,
        no_actual_response_or_outcome_used_in_candidate_selection=True,
        no_claim_this_post_development_amendment_is_pre_registered_or_held_out=True,
        pose_only_filter_NOT_tracking_braking_steering_uncertainty_or_fixed_hold_validation=True,
        unchanged_Newton_Euler_steering_speed_material_map_and_runtime_safety=True,
        original_plans_PASS_FAIL_HOLDs_and_runtime_sources_NOT_modified=True,
        no_controller_tolerance_gain_or_time_limit_change=True,
        no_new_live_stage_timeline_or_PhysX_access=True,new_PhysX_trial_count=0,
        adopted_by_any_live_runner=False,actual_motion_authorized=False,use_for_final_statistics=False))
    print('Saved prospective common terminal connection candidate:',target)
    return target


def write_terminal_connection_model_review(candidate,output=None):
    """Original nominal executor on changed plans; reuse identical old cases.

    This deliberately keeps the inherited 180 s nominal screen, not a new
    physical 300 s trial. Captured-wheel forces/caster stiction remain absent.
    The original B1 nominal PASS and actual FAIL must not be conflated.
    """
    import copy
    import inspect
    import numpy as np
    import screen_hospital_h4_common_executor as screen
    from hospital_runtime_evidence_io import write_exclusive
    candidate=Path(candidate).resolve();proposal=json.loads(candidate.read_bytes())
    if (proposal['version']!='prospective_common_terminal_pose_interface_candidate_v1'
            or proposal['adopted_by_any_live_runner'] or proposal['actual_motion_authorized']
            or proposal['new_PhysX_trial_count']!=0):
        raise ValueError('Explicit offline common connection candidate required, not an admitted trial')
    screen.bundle.cohort.verify(proposal['input_sha256'])
    original=json.loads(screen.TARGET.read_bytes());screen.bundle.cohort.verify(original['input_sha256'])
    screen.bundle.cohort.verify(original['output_sha256'])
    frozen=json.loads(screen.bundle.TARGET.read_bytes())
    plans={json.loads(Path(path).read_bytes())['method']:json.loads(Path(path).read_bytes()) for path in frozen['plans']}
    native=json.loads(screen.bundle.cohort.MODELS.read_bytes());design=screen.bundle.cohort.support.load_design()
    with np.load(screen.bundle.cohort.MAP) as grid:
        occupied,origin,resolution=grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m'])
    pins={**proposal['input_sha256'],**original['input_sha256'],**original['output_sha256'],
        str(candidate):file_sha(candidate),str(screen.TARGET):file_sha(screen.TARGET)}
    definition=dict(version='prospective_terminal_connection_original_nominal_executor_review_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_terminal_connection_model_review).encode()).hexdigest())
    key=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else candidate.parent/'nominal_review'/key).resolve()
    target=output/'nominal_connection_review.json'
    if target.exists():
        old=json.loads(target.read_bytes())
        if (any(old.get(k)!=v for k,v in definition.items())
                or any(not Path(path).is_file() or file_sha(path)!=sha for path,sha in old['output_sha256'].items())):
            raise ValueError('Existing nominal connection review changed; no overwrite')
        return target
    methods=proposal['methods']
    if len(methods)!=4 or {row['method'] for row in methods}!=set(METHODS):
        raise ValueError('Complete all-four candidate accounting required')
    output.mkdir(parents=True,exist_ok=True);cases=[];paths=[]
    for method in methods:
        selected=method['candidate_selected_by_original_native_cost']
        if selected is None:
            cases.append(dict(method=method['method'],model_status='NO_INTERFACE_CANDIDATE',new_nominal_case_executed=False))
            continue
        plan=copy.deepcopy(plans[method['method']]);prediction=method['selected_candidate_prediction']
        if not method['changed_from_original_selection']:
            expected=json.loads(json.dumps({key:plan[key] for key in ('selected_actions','selected_states')}))
            if prediction!=expected:raise ValueError('Unchanged candidate must retain exact original actions and states')
            retained=[row for row in original['cases'] if row['method']==method['method']]
            if sorted(row['integration_dt_s'] for row in retained)!=sorted(screen.TIMESTEPS_S):
                raise ValueError('Both original unmodified MODEL resolutions required')
            cases.extend({**row,'new_nominal_case_executed':False,'exact_original_case_reused':True} for row in retained)
            continue
        plan.update(**prediction,selected_candidate_id=selected['candidate_id'])
        screen.execution.validate_plan(plan)
        for dt in screen.TIMESTEPS_S:
            report,rows=screen.case(plan,native['rigid_body_parameter_bounds'],dt,occupied,origin,resolution,design)
            path=output/(method['method']+'_'+str(round(1000*dt))+'ms.csv')
            with path.open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            paths.append(path)
            cases.append(dict(**report,new_nominal_case_executed=True,exact_original_case_reused=False,
                selected_candidate_id=selected['candidate_id'],model_trace=str(path),model_trace_sha256=file_sha(path)))
            print('Prospective connection MODEL:',method['method'],dt,report['model_status'],
                'mm/deg:',1000*report['final_position_error_m'],report['final_heading_error_deg'],
                's:',report['simulation_duration_s'],'error:',report['error'],flush=True)
    screen.bundle.cohort.verify(pins)
    report=dict(**definition,status='NOMINAL_CONNECTION_CANDIDATE_REVIEW_NOT_PHYSICAL_VALIDATION',cases=cases,
        newly_executed_nominal_case_count=sum(row['new_nominal_case_executed'] for row in cases),
        exact_unchanged_original_case_reuse_count=sum(row.get('exact_original_case_reused',False) for row in cases),
        inherited_nominal_180s_limit_NOT_physical_common_300s=True,
        unchanged_original_10ms_and_20ms_steps_are_NOT_independent_repeats=True,
        physical_hold_force_caster_stiction_and_cooked_contacts_NOT_simulated=True,
        original_B1_nominal_PASS_is_NOT_its_actual_success=True,
        no_measured_response_refit_or_gain_model_controller_tolerance_change=True,
        selected_candidate_NOT_adopted_by_live_runner=True,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False,
        output_sha256={str(path):file_sha(path) for path in paths})
    write_exclusive(target,json.loads(json.dumps(report,allow_nan=False)))
    print('Saved original-executor nominal connection review:',target)
    return target


def write_terminal_connection_execution_bundle(candidate,nominal_review,output=None):
    """Freeze prospective all-four inputs, rechecking full reference support.

    This is a planner-version release, not an actual run or final comparison.
    Original bundles, native constants and common operative code stay intact.
    The builder is preserved in a small immutable source snapshot, avoiding
    a future dependency on an editable analysis/helper source path.
    """
    import copy
    import inspect
    import numpy as np
    import hospital_h4_subfloor_compact_path_launch as launch
    import screen_hospital_h4_common_executor as screen
    from hospital_geometry_tools import pose_collision
    from hospital_runtime_evidence_io import write_exclusive
    candidate=Path(candidate).resolve();nominal_review=Path(nominal_review).resolve()
    proposal=json.loads(candidate.read_bytes());nominal=json.loads(nominal_review.read_bytes())
    if (proposal['version']!='prospective_common_terminal_pose_interface_candidate_v1'
            or proposal['adopted_by_any_live_runner'] or proposal['actual_motion_authorized']
            or nominal['version']!='prospective_terminal_connection_original_nominal_executor_review_v1'
            or nominal['input_sha256'].get(str(candidate))!=file_sha(candidate)
            or nominal['actual_motion_authorized'] or nominal['new_PhysX_trial_count']!=0
            or len(nominal['cases'])!=8
            or any(row['model_status']!='MODEL_SCREEN_PASS' or not all(row['checks'].values()) for row in nominal['cases'])):
        raise ValueError('Exact all-four candidate and both original nominal resolutions required')
    launch.preparation.verify(proposal['input_sha256']);launch.preparation.verify(nominal['input_sha256'])
    launch.preparation.verify(nominal['output_sha256'])
    old,_,plans=launch.load_frozen_bundle();proposals={row['method']:row for row in proposal['methods']}
    if len(proposal['methods'])!=4 or set(proposals)!=set(plans):raise ValueError('All four own-native candidates required')
    payload=launch.fresh_runtime_source_bundle(include_visual=True)
    pins={**old['input_sha256'],**old['output_sha256'],**proposal['input_sha256'],
        **nominal['input_sha256'],**nominal['output_sha256'],str(launch.BUNDLE):file_sha(launch.BUNDLE),
        str(candidate):file_sha(candidate),str(nominal_review):file_sha(nominal_review)}
    definition=dict(version='H4_ALL_FOUR_COMMON_TERMINAL_INTERFACE_NATIVE_BUNDLE_V2',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_terminal_connection_execution_bundle).encode()).hexdigest(),
        builder_source_snapshot_sha256=hashlib.sha256(payload).hexdigest())
    key=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else candidate.parent/'execution_bundle'/key).resolve()
    target=output/'execution_bundle.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (any(saved.get(k)!=v for k,v in definition.items())
                or any(not Path(path).is_file() or file_sha(path)!=sha for path,sha in saved['output_sha256'].items())):
            raise ValueError('Existing prospective execution bundle changed; no overwrite')
        launch.load_terminal_connection_bundle(target)
        return target
    with np.load(launch.cohort.MAP) as grid:
        occupied,origin,resolution=grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m'])
    covers=launch.cohort.support_covers(launch.cohort.support.load_design())
    values=[]
    for method,original in plans.items():
        row=proposals[method];selected=row['candidate_selected_by_original_native_cost']
        if selected is None or not selected['original_native_admitted']:raise ValueError('No inferred native feasibility')
        plan=copy.deepcopy(original);plan.update(**row['selected_candidate_prediction'])
        prediction=selected['reference_geometry'];end=plan['selected_states'][-1]
        local_error=launch.execution.relative_pose((end['x_m'],end['y_m'],math.radians(end['yaw_deg'])),plan['goal_local_xyyaw'])
        plan.update(selected_candidate_id=selected['candidate_id'],generator_selector_version=definition['version'],
            selected_native_source=str(candidate),selected_native_source_sha256=file_sha(candidate),
            native_cost_s=selected['original_native_cost_s'],original_native_eligible_count=original['eligible_candidate_count'],
            eligible_candidate_count=row['common_interface_candidate_count'],
            action_count=len(plan['selected_actions']),path_length_m=.05*len(plan['selected_actions']),
            predicted_terminal_position_error_m=prediction['planned_endpoint']['goal_position_error_mm']/1000.,
            predicted_terminal_heading_error_deg=math.degrees(local_error[2]),
            predicted_terminal_speed_m_s=math.hypot(end['u_m_s'],end['v_m_s']) if method==METHODS[-1] else None,
            selector_scope='NEW common 25mm-before-end pose-interface predicate, then original own-native cost; no measured-outcome selection',
            original_spatial_actions_and_states_unchanged=not row['changed_from_original_selection'],
            prospective_terminal_connection_reference_geometry=prediction,
            additional_common_pose_interface_predicate_is_NEW_planning_version=True,
            original_plan=str(next(path for path in old['plans'] if json.loads(Path(path).read_bytes())['method']==method)),
            post_development_amendment_NOT_a_held_out_or_final_comparison=True,
            input_sha256=pins)
        launch.execution.validate_plan(plan);reference=launch.execution.PathReference(plan)
        count=0;missing=0.
        for a,b in zip(reference.poses,reference.poses[1:]):
            turn=launch.execution.wrap(b[2]-a[2])
            steps=max(1,math.ceil(math.dist(a[:2],b[:2])/.005),math.ceil(abs(turn)/math.radians(.25)))
            for index in range(steps+1):
                f=index/steps;pose=[a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1]),a[2]+f*turn]
                if pose_collision(occupied,pose,origin,resolution,clearance=.010):
                    raise ValueError('Prospective reference violates original expanded full-envelope raster')
                missing=max(missing,launch.cohort.support_area_missing(pose,covers));count+=1
        if missing>1e-9:raise ValueError('Prospective whole-body reference leaves original support')
        schedule=sum(.05/action['target_speed_m_s'] for action in plan['selected_actions'])
        plan.update(reference_center_arc_m=reference.arc[-1],nominal_longitudinal_schedule_duration_s=schedule,
            nominal_plus_90s_terminal_plus_5s_hold_s=schedule+launch.execution.CONTROLLER_CONFIG['terminal_timeout_s']+5.,
            inherited_180s_budget_does_NOT_guarantee_full_terminal_allowance=schedule+95.>180.,
            sampled_reference_geometry_and_TRUE_support=dict(samples=count,maximum_uncovered_area_m2=missing,
                clearance_m=.010,maximum_translation_spacing_m=.005,maximum_heading_spacing_deg=.25,
                continuous_collision_or_cooked_contact_certificate=False))
        values.append(plan)
    output.mkdir(parents=True,exist_ok=True);paths=[]
    for plan in values:
        path=output/(plan['method']+'.json');write_exclusive(path,launch.canonical(plan));paths.append(path)
    model_path=output/'nominal_model_screen.json'
    model=dict(version='H4_TERMINAL_CONNECTION_NOMINAL_DIAGNOSTIC_BINDING_V2',
        case_count=8,nominal_software_MODEL_PASS_count=8,cases=nominal['cases'],actual_drive_admission=False,
        original_B1_nominal_PASS_is_NOT_actual_success=True,
        input_sha256=pins,output_sha256=nominal['output_sha256'])
    write_exclusive(model_path,model);paths.append(model_path)
    archive=output/'builder_source_snapshot.zip'
    with archive.open('xb') as stream:stream.write(payload)
    paths.append(archive);launch.preparation.verify(pins)
    write_exclusive(target,dict(**definition,plans=[str(output/(plan['method']+'.json')) for plan in values],
        original_bundle=str(launch.BUNDLE),terminal_connection_candidate=str(candidate),
        nominal_connection_review=str(nominal_review),model_screen=str(model_path),
        additional_common_pose_interface_predicate_is_NEW_planning_version=True,
        original_v3_runtime_controls_safety_physics_map_support_and_task_unchanged=True,
        original_outcomes_NOT_relabelled_and_different_planner_versions_NOT_pooled=True,
        actual_motion_authorized=False,use_for_final_statistics=False,new_PhysX_trial_count=0,
        output_sha256={str(path):file_sha(path) for path in paths}))
    launch.load_terminal_connection_bundle(target)
    print('Saved separate all-four prospective execution bundle:',target)
    return target


def write_planner_version_development_contrast(original,prospective,output=None):
    """Explicit within-method development contrast, not a matched final cell.

    Only the disclosed planner/reference and its derived schedule contract
    may differ. Both input outcomes, including failure/partial traces, stay
    authoritative. Do not attribute a route/mode change to one isolated
    Newton-Euler term or turn completed versus failed clocks into a ranking.
    """
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    paths=[Path(original).resolve(),Path(prospective).resolve()]
    comparison_path=write_explicit_development_comparison(paths)
    comparison=json.loads(comparison_path.read_bytes());rows=comparison['actual_development_rows']
    raws=[json.loads(path.read_bytes()) for path in paths]
    if any(raw['method']!='P_dynamics_full_envelope' for raw in raws):
        raise ValueError('Explicit original/prospective P outcomes required, not cross-method relabelling')
    protocols=[json.loads(Path(raw['path_protocol']).read_bytes()) for raw in raws]
    old_plan,new_plan=[json.loads(Path(raw['plan_path']).read_bytes()) for raw in raws]
    if (protocols[0].get('terminal_connection_bundle') is not None
            or protocols[1].get('terminal_connection_bundle') is None
            or new_plan['generator_selector_version']!='H4_ALL_FOUR_COMMON_TERMINAL_INTERFACE_NATIVE_BUNDLE_V2'
            or new_plan['selected_candidate_id']==old_plan['selected_candidate_id']):
        raise ValueError('One original and one changed prospective terminal-connection reference required')
    allowed={'generator_selector_version','common_resource_contract'}
    differences={run:fields for run,fields in comparison['physical_control_or_domain_mismatches'].items() if set(fields)-allowed}
    if differences:raise ValueError('Unreviewed physical/control/domain mismatch, not a planner-only version contrast: '+str(differences))
    for raw in raws:
        if (raw['guards']['maximum_simulation_duration_s']!=300.
                or raw['guards']['maximum_wall_time_s']!=900.):
            raise ValueError('Same already approved common physical time budgets required')
    pins={**comparison['input_sha256'],str(comparison_path):file_sha(comparison_path)}
    definition=dict(version='original_vs_prospective_P_individual_development_contrast_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_planner_version_development_contrast).encode()).hexdigest())
    output=(Path(output) if output is not None else paths[1].parent/'offline_planner_version_contrast').resolve()
    target=output/'planner_version_contrast.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if any(saved.get(k)!=v for k,v in definition.items()):raise ValueError('Existing version contrast differs; no overwrite')
        return target
    complete=all(row['raw_verdict']=='PASS' and row['complete_fixed_hold'] for row in rows)
    metrics=('total_logged_route_clock_s','tracking_only_time_weighted_position_RMS_mm',
        'tracking_only_time_weighted_heading_RMS_deg','terminal_position_error_mm','terminal_pulse_count')
    delta={key:rows[1][key]-rows[0][key] for key in metrics} if complete else None
    records=[]
    for raw,plan,row in zip(raws,(old_plan,new_plan),rows):
        records.append(dict(**row,selected_candidate_id=plan['selected_candidate_id'],
            planned_modes=sorted({action['mode'] for action in plan['selected_actions']}),
            own_reference_geometry=terminal_reference_geometry(plan,.025,.030,2.),
            actual_handoff=raw.get('terminal_handoff'),last_recorded_state_NOT_a_completed_terminal=raw.get('last_recorded_state'),
            original_error=raw.get('error')))
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Version contrast input changed')
    write_exclusive(target,dict(**definition,status='PLANNER_VERSION_DEVELOPMENT_CONTRAST_NOT_FINAL_OR_ISOLATED_CAUSALITY',
        original_and_prospective=records,both_complete_PASS=complete,new_minus_original_if_both_complete=delta,
        expected_planner_and_derived_schedule_differences=comparison['physical_control_or_domain_mismatches'],
        same_saved_task_physics_common_control_and_actual_safety_limits=True,
        selected_reference_and_mode_change_are_NOT_one_isolated_force_load_effect=True,
        separate_cohort_or_source_seed_is_NOT_proof_of_equal_measured_initial_history=True,
        failed_partial_clocks_and_errors_NOT_ranked_as_completed_performance=True,
        original_outcomes_NOT_relabelled_or_pooled_into_final_comparison=True,
        no_live_stage_or_timeline_access=True,new_PhysX_trial_count=0,
        independent_paired_repeats_or_statistical_superiority_established=False,use_for_final_statistics=False))
    print('Saved explicit P planner-version development contrast:',target)
    return target


def initial_motion_reuse_evidence():
    """Replay ALL 135 legacy lateral traces without invoking a legacy writer.

    Exact legacy trace/jerk functions are loaded in private namespaces. Their
    original definitions are reused, not reimplemented under new metrics.
    Current byte snapshots do not invent historical freeze hashes, random
    independence, a publication status, or forward-mode controller validity.
    """
    import ast
    import runpy
    import statistics
    import numpy as np
    import scipy
    folder=_ijcas_path(WORKSPACE, 'studies/CICS26/scripts')
    jerk_path=folder/'analyze_wheelchair_cics26_trials.py'
    trace_path=folder/'build_patient_transport_cics26_paper_evidence.py'
    comparison_path=folder/'compare_patient_transport_cics26_lateral_fullmatrix.py'
    builder=folder/'build_patient_transport_final9_evidence.py'
    metric_path=_ijcas_path(WORKSPACE, 'studies/CICS26/paper_assets/patient_transport/final9/patient_transport_final9_metrics.csv')
    pins={str(path):file_sha(path) for path in (jerk_path,trace_path,comparison_path,builder,metric_path)}
    jerk=runpy.run_path(str(jerk_path))['_jerk']
    tree=ast.parse(trace_path.read_text())
    selected=[node for node in tree.body
        if isinstance(node,ast.FunctionDef) and node.name in ('read_csv','analyze_trace')
        or isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ('WINDOW_M','GRID') for target in node.targets)]
    scope=dict(__name__='_IJCAS_existing_initial_motion_metric_replay',__file__=str(trace_path),
        csv=csv,math=math,np=np,Path=Path,_jerk=jerk)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(trace_path),'exec'),scope)
    if scope['WINDOW_M']!=.70 or not np.array_equal(scope['GRID'],np.linspace(0.,.70,71)):
        raise ValueError('Original legacy 0.70m/71-point metric definition required')
    conditions=tuple(f'{mass}{suffix}' for mass in (240,265,290) for suffix in ('rearx0p10','center','frontx0p10'))
    methods=('B0','B1','B2','P','B3H')
    with metric_path.open(newline='') as stream:saved=list(csv.DictReader(stream))
    if len(saved)!=45 or {(row['condition'],row['method']) for row in saved}!={(c,m) for c in conditions for m in methods}:
        raise ValueError('Complete original 9-condition/five-method table required')
    saved={(row['condition'],row['method']):row for row in saved};cells=[];traces_by_sha=defaultdict(list);response=[];missing_metadata=[]
    for condition in conditions:
        mass=float(condition[:3]);com=-.10 if 'rear' in condition else .10 if 'front' in condition else 0.
        for method in methods:
            label=f'cics26_pt_{condition}_lateral_lowbed_frontpos_{method}_v5'
            directory=_ijcas_path(WORKSPACE, 'studies/CICS26/results/patient_transport_lab/natural_flip_pilot', label)
            summary=directory/'summary.csv';pins[str(summary)]=file_sha(summary)
            summaries=scope['read_csv'](summary)
            if len(summaries)!=3 or {row['repeat'] for row in summaries}!={'1','2','3'}:
                raise ValueError('Exactly three original recorded repeats required: '+label)
            for row in summaries:
                if (row['run_label']!=label or row['method']!=method or row['analysis_role']!='recorded'
                        or row['motion_mode']!='lateral_differential_steering_90deg'
                        or float(row['model_mass_kg_provisional'])!=mass
                        or float(row['reverse_target_distance_m'])!=.70):
                    raise ValueError('Own original lateral/load/CoM metadata required: '+label)
                if row.get('base_com_x_m') not in (None,''):
                    if float(row['base_com_x_m'])!=com:raise ValueError('Recorded CoM differs from its declared condition: '+label)
                else:
                    missing_metadata.append(dict(summary=str(summary),repeat=row['repeat'],field='base_com_x_m',
                        declared_condition=condition,actual_value=None,declared_condition_NOT_used_as_actual_value=True))
            values=[]
            for repeat in (1,2,3):
                trace=directory/f'trace_r{repeat:02d}.csv';sha=file_sha(trace);pins[str(trace)]=sha
                traces_by_sha[sha].append(str(trace));values.append(scope['analyze_trace'](trace))
            reached=sum(value['reached'] for value in values)
            if reached!=3:raise ValueError('Legacy raw replay contradicts the saved completed table; do not invent a replacement success')
            med=lambda key:statistics.median(value[key] for value in values)
            curves=[value['curve'] for value in values]
            record=dict(condition=condition,legacy_method=method,run_count=3,reached_count=reached,
                original_summary_flip_count=sum(row['flip_completed']=='True' for row in summaries),
                median_max_lateral_mm=med('max_lat_mm'),median_max_yaw_deg=med('max_yaw_deg'),
                median_peak_jerk_m_s3=med('peak_jerk'),median_travel_time_s=med('travel_time_s'),
                trajectory_spread_mm=1000.*float(np.sqrt(np.mean(np.var(curves,axis=0,ddof=1)))))
            expected=saved[condition,method]
            mapping={'flip_count':'original_summary_flip_count','reached_count':'reached_count',
                **{key:key for key in ('median_max_lateral_mm','median_max_yaw_deg','median_peak_jerk_m_s3',
                    'median_travel_time_s','trajectory_spread_mm')}}
            if any(not math.isfinite(float(record[key])) or not math.isclose(float(record[key]),float(expected[name]),rel_tol=1e-10,abs_tol=1e-10)
                    for name,key in mapping.items()):
                raise ValueError('Original saved metrics and exact legacy raw replay differ: '+label)
            cells.append(record)
            if method=='P':response.extend(summaries)
    aggregate=[]
    names=dict(B0='Immediate',B1='Fixed low-speed',B2='Acceleration ramp',P='Response-step + hold',B3H='Fixed micro-yaw + hold')
    keys=('median_max_lateral_mm','median_max_yaw_deg','median_peak_jerk_m_s3','median_travel_time_s','trajectory_spread_mm')
    for method in methods:
        subset=[row for row in cells if row['legacy_method']==method]
        aggregate.append(dict(legacy_method=method,label=names[method],
            **{key:statistics.median(row[key] for row in subset) for key in keys},
            original_summary_flip_count=sum(row['original_summary_flip_count'] for row in subset),
            reached_count=sum(row['reached_count'] for row in subset)))
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Legacy input changed during read-only replay')
    return dict(status='EXISTING_LATERAL_INITIAL_MOTION_REPLAY_NOT_NEW_HOSPITAL_PLANNING_TRIALS',
        condition_count=9,methods=list(methods),recorded_run_count=135,reached_count=sum(row['reached_count'] for row in cells),
        cells=cells,median_of_nine_condition_summaries=aggregate,
        original_response_detected_count=sum(row['adaptive_response_reached']=='True' for row in response),
        original_response_selected_rates_rad_s=sorted({float(row['adaptive_selected_rate_rad_s']) for row in response}),
        identical_trace_byte_groups=[paths for paths in traces_by_sha.values() if len(paths)>1],
        missing_legacy_metadata_retained_NOT_filled=missing_metadata,
        exact_existing_trace_and_jerk_definitions_reused_without_legacy_writer=True,
        legacy_saved_table_numerically_reproduced=True,
        original_trace_provenance='CURRENT_BYTE_SNAPSHOTS_NO_ORIGINAL_FREEZE_HASH_INVENTED',
        simulation_mass_CoM_sensitivity_NOT_measured_patient_uncertainty=True,
        legacy_method_P_is_NOT_the_current_planning_method_P=True,
        lateral_results_do_NOT_validate_longitudinal_hospital_response=True,
        CICS_runtime_integration=False,new_PhysX_trial_count=0,
        publication_metadata_and_permission_to_reuse_NOT_invented=True,
        numpy_version=np.__version__,scipy_version=scipy.__version__,input_sha256=pins)


def write_manuscript_development_assembly(development_table,version_contrast,output=None):
    """Build one native TeX working manuscript from explicit verified evidence.

    Not a submission-ready paper or a final comparison qualification. Prior
    lateral, original corridor, hospital development and planner amendments
    remain separate. Missing evidence is not replaced by model predictions.
    No legacy writer, experiment runner or live Isaac interface is invoked.
    """
    import inspect
    import os
    from hospital_runtime_evidence_io import write_exclusive
    development_table,version_contrast=map(lambda p:Path(p).resolve(),(development_table,version_contrast))
    development=json.loads(development_table.read_bytes());contrast=json.loads(version_contrast.read_bytes())
    if (development.get('status')!='MANUSCRIPT_ASSEMBLY_DEVELOPMENT_TABLE_NOT_FINAL_EVALUATION'
            or development.get('use_for_final_statistics') is not False
            or contrast.get('status')!='PLANNER_VERSION_DEVELOPMENT_CONTRAST_NOT_FINAL_OR_ISOLATED_CAUSALITY'
            or contrast.get('use_for_final_statistics') is not False):
        raise ValueError('Explicit development table and separately labelled P version contrast required')
    expected={(task,method) for task in ('H1','H2','H4') for method in METHODS}
    rows=development['rows'];versions=contrast['original_and_prospective']
    if len(rows)!=12 or {(row['task'],row['method']) for row in rows}!=expected:
        raise ValueError('All four original method outcomes, including FAILs, required for each supplied hospital task')
    original=next(row for row in rows if row['task']=='H4' and row['method']==METHODS[-1])
    if (len(versions)!=2 or original['run_id']!=versions[0]['run_id']
            or any(row['method']!=METHODS[-1] for row in versions)
            or versions[0]['selected_candidate_id']==versions[1]['selected_candidate_id']):
        raise ValueError('Original H4 P must remain the original comparison row; amended P is a separate development version')
    for record in (development,contrast):
        for section in ('input_sha256','output_sha256'):
            if any(file_sha(Path(path))!=sha for path,sha in record.get(section,{}).items()):
                raise ValueError('Explicit manuscript evidence or existing table output changed')
    pins={**development['input_sha256'],**development['output_sha256'],**contrast['input_sha256'],
        str(development_table):file_sha(development_table),str(version_contrast):file_sha(version_contrast)}
    anchor_path=CONFIG/'hospital_extension_scope_1029.json';scope_path=CONFIG/'hospital_simulation_scope_addendum_20261116.json'
    h3_path=RESULT/'hospital_h3b_prefix_response_observation/response_completion/closeout_r01/live_final_response_goal.json'
    anchor=json.loads(anchor_path.read_bytes());scope=json.loads(scope_path.read_bytes());h3=json.loads(h3_path.read_bytes())
    for path in (anchor_path,scope_path,h3_path,PROTOCOL_PATH):pins[str(path)]=file_sha(path)
    for name,sha in anchor['source_sha256'].items():
        if not Path(name).is_absolute():pins[str(_ijcas_path(WORKSPACE, name))]=sha
    pins.update(h3['input_sha256'])
    if (scope['real_robot_trials_required_for_current_completion_scope'] is not False
            or scope['CICS_runtime_integration'] is not False
            or h3['saved_body_and_twelve_joint_handoffs_verified']!=4
            or h3['independent_fresh_frozen_whole_sequence_repeats']!=0
            or (h3['original_whole_model_PASS_count'],h3['original_whole_model_case_count'])!=(2,8)):
        raise ValueError('Preserved simulation-only scope and exact segmented H3 accounting required')
    prior=initial_motion_reuse_evidence();pins.update(prior['input_sha256'])
    causal=json.loads((_ijcas_path(WORKSPACE, 'studies/IJCAS/results/rigid_body_edge_gate/operational_dynamics_causality_r01/operational_dynamics_causal_repeats.json')).read_bytes())
    robust=json.loads((_ijcas_path(WORKSPACE, 'studies/IJCAS/results/rigid_body_edge_gate/integrated_robustness_repeats.json')).read_bytes())
    original_conditions=len(robust['conditions']);original_runs=sum(len(cell['runs']) for cell in robust['conditions'])
    original_passes=sum(run['verdict']=='PASS' for cell in robust['conditions'] for run in cell['runs'])
    for name in ('steering_lattice.py','rigid_body_edge.py','hierarchical_dynamics_planner.py'):
        path=_ijcas_path(WORKSPACE, 'src/patient_initial_stability/patient_initial_stability', name);pins[str(path)]=file_sha(path)
    figure=Path(versions[1]['result']).parent/'offline_summary/measured_execution.png'
    if not figure.is_file():raise ValueError('Own measured amended-P figure required; no fabricated visual')
    figure_manifest=figure.parent/'figure_manifest.json';fm=json.loads(figure_manifest.read_bytes())
    if fm.get('output_sha256',{}).get(str(figure))!=file_sha(figure):raise ValueError('Own measured figure integrity required')
    pins.update({str(figure):file_sha(figure),str(figure_manifest):file_sha(figure_manifest),**fm['input_sha256']})
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Manuscript source evidence changed')
    definition=dict(version='IJCAS_native_TeX_manuscript_development_assembly_v1',input_sha256=pins,
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_manuscript_development_assembly).encode()).hexdigest(),
        initial_motion_replay_function_sha256=hashlib.sha256(inspect.getsource(initial_motion_reuse_evidence).encode()).hexdigest())
    key=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else RESULT/'paper_evidence_inventory/manuscript_assembly'/key).resolve()
    target=output/'manuscript_assembly.json';tex_path=output/'manuscript_development.tex'
    prior_path=output/'existing_initial_motion_replay.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (any(saved.get(k)!=v for k,v in definition.items()) or any(not Path(path).is_file()
                or file_sha(Path(path))!=sha for path,sha in saved['output_sha256'].items())):
            raise ValueError('Existing manuscript assembly differs; no overwrite')
        return target
    def number(value,digits=2):return '--' if value is None else f'{value:.{digits}f}'
    legacy=[]
    for row in prior['median_of_nine_condition_summaries']:
        label={'B0':r'$I_0$','B1':r'$I_1$','B2':r'$I_2$','P':r'$I_R$','B3H':r'$I_F$'}[row['legacy_method']]
        legacy.append(' & '.join((label,row['label'],number(row['median_max_lateral_mm'],3),
            number(row['median_max_yaw_deg'],3),number(row['median_peak_jerk_m_s3'],3),
            number(row['median_travel_time_s'],3),number(row['trajectory_spread_mm'],3),
            str(row['original_summary_flip_count'])+'/27'))+r' \\')
    hospital=[]
    for row in rows:
        hospital.append(' & '.join((row['task'],row['method'].split('_')[0],row['raw_verdict'],
            number(row['successful_route_clock_end_s']),number(row['completed_tracking_position_RMS_mm']),
            number(row['terminal_position_error_mm']),number(row['terminal_absolute_heading_error_deg'],3)))+r' \\')
    amended=[]
    for label,row in zip((r'$P_{\mathrm{v1}}$',r'$P_{\mathrm{v2}}$'),versions):
        complete=row['raw_verdict']=='PASS' and row['complete_fixed_hold']
        amended.append(' & '.join((label,row['raw_verdict'],number(row['total_logged_route_clock_s'] if complete else None),
            number(row['tracking_only_time_weighted_position_RMS_mm'] if complete else None),
            number(row['terminal_position_error_mm'] if complete else None),
            number(abs(row['terminal_heading_error_deg']) if complete else None,3),str(row['terminal_pulse_count'])))+r' \\')
    def raw_tex_path(path):return r'\detokenize{'+os.path.relpath(path,output)+'}'
    text=r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,geometry}
\geometry{margin=24mm}
\title{Dynamics- and Full-Envelope-Aware Multi-Steering Planning\\for Patient-Transport Robots in Hospital Simulation}
\author{Author information and verified prior-study citation pending}
\date{Working development manuscript -- not a submission-ready evaluation}
\begin{document}
\maketitle
\begin{abstract}
This working manuscript describes multi-steering path planning that carries
steering and body-response states, evaluates bounded Newton--Euler feasibility,
and checks a caster-inclusive body envelope in hospital simulation. Spatial
route candidates are evaluated using method-specific state propagation and
speed-consistent refinement. Prior lateral initial-motion experiments and
original corridor results are retained as distinct evidence. The current
hospital records establish development feasibility and expose planning/execution
trade-offs; they do not yet establish independently repeated superiority or a
calibrated response model. Final comparative conclusions are intentionally
reserved for a frozen, matched evaluation.
\end{abstract}

\section{Introduction and research scope}
Patient-transport navigation must account for body clearance and finite
steering transitions, not only the trajectory of a reference point. The
research question is whether carried steering/body state, bounded dynamic
feasibility and full-envelope planning improve declared hospital-task outcomes
against matched kinematic and transition-cost baselines. The implementation is
hierarchical candidate generation and dynamic refinement, not globally optimal
joint state-space search.

The study scope is Isaac Sim hospital validation. It does not claim clinical
safety, real-robot transfer, patient-specific identification or exact sensor
hardware emulation. Prior initial-motion control is included as a separately
identified study, not transplanted into the present longitudinal planner.

\section{Relationship to the existing initial-motion study}
The previous simulation study considered lateral differential departure with
the active steering axes near $90^\circ$. Nine declared load/CoM conditions
combine total masses of 240, 265 and 290 kg with rear, centre and front CoM
conditions. These are simulation sensitivity settings, not measured patient
distributions. Five methods were recorded three times per condition; all
135 saved traces reach the common 0.70 m analysis distance. The original
trace/jerk definitions were replayed without acquiring new data.

Table~\ref{tab:initial} reports medians across the nine condition summaries;
each condition summary uses three traces. Its symbols $I_0,I_1,I_2,I_R,I_F$
are deliberately distinct from the present planning methods $B0,B1,B2,P$.
\begin{table}[ht]
\centering\scriptsize
\caption{Existing lateral initial-motion evidence; reused, not new hospital
planning trials. Lateral/yaw values are maximum deviations in the common
distance window. Flip counts retain the original summary definition.}
\label{tab:initial}
\begin{tabular}{llrrrrrr}\toprule
ID & Existing method & Lateral (mm) & Yaw ($^\circ$) & Jerk (m/s$^3$) & Time (s) & Spread (mm) & Flip \\\midrule
@@INITIAL_ROWS@@
\bottomrule\end{tabular}
\end{table}

The fixed micro-yaw method and the response-step method show different
lateral/yaw/jerk trade-offs. The fixed low-speed arm has a different target
speed, so its time and jerk are not like-for-like equal-speed comparisons.
The response method detects a response in 27/27 records but selects the same
first rate, 0.0080078125 rad/s, throughout. These records do not demonstrate
load-dependent step escalation. Their validity is not extended to the present
forward/reverse hospital manoeuvres. Bibliographic status and any required
reuse attribution must be completed from the actual prior-study version;
no publication metadata is invented in this draft.

\section{Planning state and bounded dynamic feasibility}
\subsection{Carried state and steering modes}
The dynamic planning state contains planar pose, front/rear steering, motion
mode and direction, and body longitudinal/lateral velocity and yaw rate:
\begin{equation}
s=(x,y,\psi,\delta_f,\delta_r,m,d,u,v,r).
\end{equation}
With $L=l_f+l_r$, the geometric steering targets for commanded curvature
$\kappa_c$ are
\begin{equation}
(\delta_f^*,\delta_r^*)=\begin{cases}
(\arctan(L\kappa_c),0), & m=\text{front-only},\\
(0,-\arctan(L\kappa_c)), & m=\text{rear-only},\\
(\arctan(l_f\kappa_c),-\arctan(l_r\kappa_c)), & m=\text{counterphase}.
\end{cases}
\end{equation}
Source-bound response/steering-target corrections are retained in the versioned
model; they are not re-estimated from the evaluation run. Steering evolves
under a rate limiter, rather than an instantaneous mode-dependent reset:
\begin{equation}
\delta_{i,k+1}=\delta_{i,k}+\operatorname{clip}
(\bar\delta_i^*-\delta_{i,k},-\dot\delta_{\max}\Delta t,
\dot\delta_{\max}\Delta t).
\end{equation}
The implementation uses two half steering updates per integration substep.
The speed-consistent refinement carries $(u,v,r)$ and integrates until the
primitive's longitudinal travel is reached; fixed nominal time is not
silently substituted for distance at startup.

\subsection{Response propagation and Newton--Euler constraints}
Mode-dependent geometric targets provide
\begin{align}
r^*&=u^*\rho_m^\kappa(\tan\delta_f-\tan\delta_r)/L,\\
v^*&=u^*\rho_m^v(l_r\tan\delta_f+l_f\tan\delta_r)/L.
\end{align}
Bounded first-order acceleration responses move $(u,v,r)$ towards these
targets without overshoot. For example,
\begin{equation}
\dot u_c=\operatorname{clip}((u^*-u)/\tau_u,-a_{u,\max},a_{u,\max}).
\end{equation}
The mass-normalized body wrench is evaluated using
\begin{equation}
f_x=\dot u-rv,\qquad f_y=\dot v+ru,\qquad n_z=r_g^2\dot r.
\end{equation}
Here $f_x,f_y$ are force per mass and $n_z$ is yaw moment per mass. Bounded
CoM location/height, yaw radius, friction and caster-disturbance parameters
define a conservative two-support-line feasibility envelope. For example,
\begin{equation}
n_f=\frac{g l_r-h f_x}{L},\quad n_r=g-n_f,\quad
\mathrm{LTR}=\frac{2|c_y+h f_y/g|}{w}.
\end{equation}
Positive support, bounded LTR, friction utilization and actuator limits are
checked at the declared parameter corners. This is not a reconstruction of
individual six-wheel contact forces. A planning-performance improvement is
attributed to a force/load term only when its operational admission or
execution effect is separately demonstrated. Nonbinding safety constraints
remain part of the method without being renamed a causal performance gain.

\subsection{Full-envelope and stopping constraints}
The conservative caster-inclusive body rectangle is 1.630 m by 0.965 m.
For the declared clearance $c$, let
\begin{equation}
\mathcal B_c=\{p:|p_x|\leq1.630/2+c,\ |p_y|\leq0.965/2+c\}.
\end{equation}
At pose $q$, the footprint is $\mathcal F(q)=[x,y]^T+R(\psi)\mathcal B_c$.
Spatial edges and their stopping continuation are retained only when their
sampled full-envelope checks satisfy the declared occupancy/support contract.
Hospital development uses $c=10$ mm. Sampled raster clearance is not claimed
to be a continuous cooked-mesh or loaded-patient three-dimensional guarantee.

\section{Matched planning comparison and common execution}
The four planning methods are: $B0$, fixed counterphase kinematics; $B1$,
multimode kinematics with instantaneous steering transitions; $B2$, multimode
kinematics with transition-time cost; and $P$, carried-state dynamic
propagation and full-envelope feasibility. Shared spatial candidates and
budgets distinguish method effects from different search opportunities.
Each native cost/admission rule is retained; extra execution stabilisation is
not relabelled as a planning contribution.

Hospital ground-truth development uses a common acceleration-limited startup,
course tracking, measured braking/steering settlement and bounded terminal
closure. H4 has a 50 mm/s command cap and 8$^\circ$/s steering limit. The
reported successful H4 comparison cohort uses common 300 s simulation and
900 s wall limits. Acceptance requires the original 30 mm/2$^\circ$ pose
bands, speed at most 5 mm/s, zero unexpected contact and one fixed 5 s hold
evaluated over its final 3 s. No successful later window is selected.

Hospital source geometry, local support and the collision-policy version
are part of the experiment definition. The current H4 version uses local
floor-aligned support and disables four buried wall-base collision meshes
in the Session layer, identically for the listed methods. It is separate
from earlier original-environment pilots. Software commissioning issues are
not counted as algorithm-performance trials or scientific method limitations.

\section{Preserved evidence and hospital development results}
\subsection{Original corridor evidence}
The preserved original corridor contrast records @@ORIGINAL_PAIRS@@ paired
executions: the kinematic route succeeds in @@KINEMATIC_SUCCESS@@ and the
dynamic-aware route in @@DYNAMIC_SUCCESS@@. A separate fixed-route condition
study records @@ORIGINAL_PASSES@@/@@ORIGINAL_RUNS@@ PASS outcomes over
@@ORIGINAL_CONDITIONS@@ conditions. These remain original corridor evidence;
they are not relabelled as new hospital tasks or independent isolation of
every dynamic constraint. Existing corridor sensor/Nav2 success is likewise
not treated as hospital sensor validation.

\subsection{Explicit H1, H2 and H4 development outcomes}
\begin{table}[ht]
\centering\small
\caption{One explicitly supplied development outcome per method and task.
These are not independently paired final repetitions. Failed partial traces
are retained and are not ranked as completed time/error results.}
\label{tab:hospital}
\begin{tabular}{lllrrrr}\toprule
Task & Method & Result & End clock (s) & Tracking RMS (mm) & Goal (mm) & $|e_\psi|$ ($^\circ$) \\\midrule
@@HOSPITAL_ROWS@@
\bottomrule\end{tabular}
\end{table}
Tracking RMS excludes terminal correction. Time is the recorded route-clock
end, not wall time. H1/H2 share a ward and represent task variation, not two
independent hospital layouts. Their legacy trace provenance is a current-byte
snapshot; H4 traces are pinned in their executed protocols. The listed H4
$B0/B2$ CSV bytes are identical despite separate execution records: no
independent robustness sample is inferred from that identity. The H4 $B1$
outcome is a terminal steering-alignment timeout, not an omitted success.

The archived H3 response chain has five successful components, four verified
saved physical handoffs and an archived final-goal readback. It has zero
independent fresh frozen whole-sequence repeats. The original whole-sequence
model remains 2/8 PASS. Segmented physical completion is not promoted to a
frozen whole-sequence/model admission.

\subsection{Separate terminal-interface amendment}
The original common terminal transition occurs with 25 mm reference distance
remaining. In the original $P$ reference, even an ideal on-reference pose at
this point is 38.36 mm from the goal. A new, explicitly post-development
planning version applies the same 30 mm/2$^\circ$ pose-interface predicate
to all four methods' original 26 candidates while retaining their own costs.
The selected $P$ path changes from counterphase to a rear-only curve; the
other selected native paths remain unchanged. Its ideal interface distance
is 20.74 mm. This geometric predicate is not a braking or uncertainty bound.

\begin{table}[ht]
\centering\small
\caption{Original and amended $P$ development versions. Reference geometry
and steering mode differ; this is not an isolated force/load ablation or a
new matched four-method final comparison.}
\label{tab:amendment}
\begin{tabular}{llrrrrr}\toprule
Version & Result & End clock (s) & Tracking RMS (mm) & Goal (mm) & $|e_\psi|$ ($^\circ$) & Pulses \\\midrule
@@VERSION_ROWS@@
\bottomrule\end{tabular}
\end{table}
The amended run reduces completion time and terminal translation error and
requires no terminal pulse, while tracking RMS and terminal heading error
increase. It therefore does not establish universal metric dominance. Its
terminal-transition heading decomposes approximately as
$0.930^\circ$ planned residual plus $1.047^\circ$ tracking deviation. No
isolated caster/friction/load cause is inferred from this decomposition.
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{@@MEASURED_FIGURE@@}
\caption{Measured amended-$P$ hospital development run and its own frozen
reference. This figure is not a nominal-model prediction or a final repeated
comparison. Body outlines and obstacle AABBs are planar proxies.}
\end{figure}

\section{Evaluation still required before final comparative conclusions}
The current evidence supports development feasibility, not a submission-ready
claim of superiority. The remaining evaluation must freeze common planning
and execution versions, declare operational hospital tasks independently of
steering-policy outcomes, acquire fresh matched condition/history blocks and
retain all planning and execution failures. Repeated deterministic source
replays must not be counted as independent random robustness trials.

The declared study also requires mechanism evidence for carried state,
stopping and the full envelope; sensitivity/uncertainty assessment within the
claimed simulated domain; and hospital sensor/navigation evidence to the
extent claimed. H5/H6 task/support/driving and fresh frozen H3 whole-sequence
validation are not established by the present records. No fixed trial count
is asserted here as a journal rule or acceptance guarantee.

\section{Conclusion placeholder}
Final cross-environment effect sizes, uncertainty intervals and comparative
conclusions will be written from the frozen evaluation, without replacing
negative outcomes or selecting the best observed variant. The existing
initial-motion study, original corridor results and hospital development
records will retain their distinct provenance and validation scope.
\end{document}
'''
    replacements={'@@INITIAL_ROWS@@':'\n'.join(legacy),'@@HOSPITAL_ROWS@@':'\n'.join(hospital),
        '@@VERSION_ROWS@@':'\n'.join(amended),'@@MEASURED_FIGURE@@':raw_tex_path(figure),
        '@@ORIGINAL_PAIRS@@':str(causal['pair_count']),'@@KINEMATIC_SUCCESS@@':str(causal['kinematic_success_count']),
        '@@DYNAMIC_SUCCESS@@':str(causal['dynamics_aware_success_count']),'@@ORIGINAL_PASSES@@':str(original_passes),
        '@@ORIGINAL_RUNS@@':str(original_runs),'@@ORIGINAL_CONDITIONS@@':str(original_conditions)}
    for name,value in replacements.items():text=text.replace(name,value)
    if '@@' in text:raise ValueError('Unresolved manuscript data marker; no partial draft export')
    if any(file_sha(Path(path))!=sha for path,sha in pins.items()):raise ValueError('Manuscript input changed during assembly')
    output.mkdir(parents=True,exist_ok=True)
    with tex_path.open('x',encoding='utf-8') as stream:stream.write(text)
    write_exclusive(prior_path,prior)
    write_exclusive(target,dict(**definition,status='WORKING_NATIVE_TEX_MANUSCRIPT_NOT_SUBMISSION_READY_OR_FINAL_STATISTICS',
        explicit_original_development_table=str(development_table),explicit_separate_P_version_contrast=str(version_contrast),
        original_initial_motion_runs_replayed=135,legacy_replay_not_new_data=True,
        initial_motion_P_NOT_renamed_as_current_planning_P=True,
        original_hospital_FAILs_and_planner_versions_preserved=True,
        H3_archived_segmented_goal_NOT_current_live_retention_or_whole_sequence_success=True,
        setup_errors_NOT_presented_as_scientific_method_limitations=True,
        no_final_hospital_qualification_or_statistical_superiority_inferred=True,
        latex_compilation_verified=False,author_bibliography_and_prior_reuse_metadata_complete=False,
        final_hospital_evaluation_complete=False,submission_ready=False,
        real_robot_trials_required_for_this_scope=False,CICS_runtime_integration=False,
        no_live_stage_or_timeline_access=True,new_PhysX_trial_count=0,use_for_final_statistics=False,
        output_sha256={str(path):file_sha(path) for path in (tex_path,prior_path)}))
    print('Saved native working manuscript and evidence manifest:',target)
    return target


def write_korean_manuscript_review(output=None):
    """Editable Korean-first DOCX, not automatic translation/submission.

    The source is intentionally a normal UTF-8 text document, independently
    authored from the verified evidence and the user's two-page CICS paper.
    No external document service, translation, TeX engine or Isaac app runs.
    """
    import inspect
    import io
    import re
    import zipfile
    import xml.etree.ElementTree as ET
    from hospital_runtime_evidence_io import write_exclusive
    paper=_ijcas_path(WORKSPACE, 'studies/IJCAS/paper');source=paper/'korean_manuscript.txt'
    cics=Path('/home/jk/Documents/CICS_한양대학교_박정근.pdf')
    table=RESULT/'paper_evidence_inventory/manuscript_tables/hospital_development_6abbfc85692e6f0c.json'
    contrast=RESULT/'hospital_h4_experiments/cohort_09_full_task_r01/route/offline_planner_version_contrast/planner_version_contrast.json'
    h3=RESULT/'hospital_h3b_prefix_response_observation/response_completion/closeout_r01/live_final_response_goal.json'
    paths=(source,cics,table,contrast,h3,CONFIG/'hospital_extension_scope_1029.json',
        CONFIG/'hospital_simulation_scope_addendum_20261116.json',PROTOCOL_PATH,
        _ijcas_path(WORKSPACE, 'studies/CICS26/paper_assets/cics26_final_integrated/platform_proposed_summary.csv'),
        paper/'IJCAS.cls',paper/'picins.sty')
    pins={str(p):file_sha(p) for p in paths}
    if pins[str(cics)]!='749fd18fc7e52f3fd112e73821dd12d2152448acdbcff3ccdd8ad826bf9bccd3':
        raise ValueError('The actual user-provided two-page CICS reference changed; review required')
    for path in (table,contrast,h3):
        report=json.loads(path.read_bytes())
        for section in ('input_sha256','output_sha256'):
            for name,sha in report.get(section,{}).items():
                if file_sha(name)!=sha:raise ValueError('Korean review evidence changed: '+name)
                if name in pins and pins[name]!=sha:raise ValueError('Conflicting Korean review evidence pins')
                pins[name]=sha
    prior=initial_motion_reuse_evidence();pins.update(prior['input_sha256'])
    text=source.read_text(encoding='utf-8');lines=text.splitlines()
    if (not lines or not all(name in text for name in ('H1','H2','H3','H4','H5','H6','Nav2','센서','CICS','B0','B1','B2'))
            or '53개 최종 평가 셀' not in text or '[작성 예정]' not in text):
        raise ValueError('Full task/sensor scope and honest unfinished-results markers required')
    output=(Path(output) if output is not None else paper).resolve()
    target=output/'korean_review_manifest.json';doc=output/'korean_manuscript.docx'
    definition=dict(version='KOREAN_FIRST_EDITABLE_MANUSCRIPT_WITH_COMPLETE_HOSPITAL_SCOPE_v1',
        input_sha256=pins,exporter_function_sha256=hashlib.sha256(inspect.getsource(write_korean_manuscript_review).encode()).hexdigest())
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (any(saved.get(k)!=v for k,v in definition.items()) or any(not Path(p).is_file()
                or file_sha(p)!=sha for p,sha in saved['output_sha256'].items())):
            raise ValueError('Existing Korean review document changed; no overwrite of author edits')
        return target
    # Minimal portable OOXML; no optional python-docx installation is needed.
    ns='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    ET.register_namespace('w',ns)
    def node(parent,name,attrs=None):
        return ET.SubElement(parent,'{'+ns+'}'+name,{'{'+ns+'}'+k:str(v) for k,v in (attrs or {}).items()})
    document=ET.Element('{'+ns+'}document');body=node(document,'body')
    def paragraph(parent,value,style='Normal',bold=False):
        p=node(parent,'p');properties=node(p,'pPr');node(properties,'pStyle',{'val':style})
        if style=='Note':node(properties,'shd',{'fill':'F1F5F9'})
        r=node(p,'r')
        if bold:node(node(r,'rPr'),'b')
        t=node(r,'t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=value
        return p
    extracted={};index=0;table_number=0
    while index<len(lines):
        line=lines[index].strip()
        if not line:index+=1;continue
        if line.startswith('|'):
            table_lines=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                table_lines.append([cell.strip() for cell in lines[index].strip().strip('|').split('|')]);index+=1
            width=len(table_lines[0])
            if width!=4 or any(len(row)!=width for row in table_lines):raise ValueError('Four complete review-table columns required')
            tbl=node(body,'tbl');pr=node(tbl,'tblPr');node(pr,'tblW',{'w':0,'type':'auto'})
            node(pr,'tblLayout',{'type':'fixed'});borders=node(pr,'tblBorders')
            for side in ('top','left','bottom','right','insideH','insideV'):
                node(borders,side,{'val':'single','sz':4,'color':'B8C4CF'})
            grid=node(tbl,'tblGrid')
            for _ in range(width):node(grid,'gridCol',{'w':2350})
            for row_index,row in enumerate(table_lines):
                tr=node(tbl,'tr')
                if row_index==0:node(node(tr,'trPr'),'tblHeader')
                for value in row:
                    cell=node(tr,'tc');cp=node(cell,'tcPr');node(cp,'tcW',{'w':2350,'type':'dxa'})
                    if row_index==0:node(cp,'shd',{'fill':'DDEAF2'})
                    paragraph(cell,value,'TableText',bold=row_index==0)
            names=('comparison_methods.csv','hospital_scenarios.csv','figure_plan.csv')
            if table_number>=len(names):raise ValueError('Unexpected source table; review the document design')
            extracted[names[table_number]]=table_lines;table_number+=1
            continue
        style=('Title' if index==0 else 'Heading2' if re.match(r'^\d+\.\d+\s',line)
            else 'Heading1' if re.match(r'^(?:\d+\.\s|부록 [ABC]\.)',line) or line=='초록'
            else 'Note' if line.startswith('[') or line.startswith('한글 검토 원고') or line.startswith('작성 원칙:')
            else 'Normal')
        paragraph(body,line,style);index+=1
    if table_number!=3:raise ValueError('Methods, scenarios and Figure plan tables all required')
    section=node(body,'sectPr');node(section,'pgSz',{'w':11906,'h':16838})
    node(section,'pgMar',{'top':1134,'right':1134,'bottom':1134,'left':1134,'header':567,'footer':567,'gutter':0})
    styles=ET.Element('{'+ns+'}styles')
    for name,size,color,bold in (('Normal',21,'202A35',False),('Title',32,'164B63',True),
            ('Heading1',27,'164B63',True),('Heading2',23,'285C75',True),('Note',20,'546575',False),('TableText',18,'202A35',False)):
        st=node(styles,'style',{'type':'paragraph','styleId':name});node(st,'name',{'val':name})
        if name!='Normal':node(st,'basedOn',{'val':'Normal'})
        else:st.set('{'+ns+'}default','1')
        pp=node(st,'pPr');node(pp,'spacing',{'after':100,'line':320,'lineRule':'auto'})
        if name in ('Title','Heading1','Heading2'):node(pp,'keepNext')
        rp=node(st,'rPr');node(rp,'rFonts',{'ascii':'Noto Sans CJK KR','hAnsi':'Noto Sans CJK KR','eastAsia':'Noto Sans CJK KR'})
        node(rp,'sz',{'val':size});node(rp,'szCs',{'val':size});node(rp,'color',{'val':color})
        node(rp,'lang',{'val':'ko-KR','eastAsia':'ko-KR'})
        if bold:node(rp,'b')
    content_types='''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    root_rels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    doc_rels='''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    files={'[Content_Types].xml':content_types.encode(),'_rels/.rels':root_rels.encode(),
        'word/document.xml':ET.tostring(document,encoding='utf-8',xml_declaration=True),
        'word/styles.xml':ET.tostring(styles,encoding='utf-8',xml_declaration=True),'word/_rels/document.xml.rels':doc_rels.encode()}
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in files.items():
            ET.fromstring(value);info=zipfile.ZipInfo(name,date_time=(2026,11,17,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,value)
    output.mkdir(parents=True,exist_ok=True);artifacts={doc:payload.getvalue()}
    for name,rows in extracted.items():
        stream=io.StringIO(newline='');csv.writer(stream).writerows(rows);artifacts[output/name]=stream.getvalue().encode('utf-8-sig')
    if any(p.exists() for p in artifacts):raise ValueError('Unfinalized author-review output exists; no overwrite')
    if any(file_sha(p)!=sha for p,sha in pins.items()):raise ValueError('Korean review input changed during export')
    for path,value in artifacts.items():
        with path.open('xb') as stream:stream.write(value)
    write_exclusive(target,dict(**definition,status='KOREAN_AUTHOR_REVIEW_AND_FIGURE_PLANNING_NOT_TRANSLATED_OR_SUBMISSION_READY',
        CICS_reference_pages=2,existing_patient_initial_motion_runs_replayed=prior['recorded_run_count'],
        wheelchair_values_from_user_paper_and_existing_aggregate_NOT_part_of_135_replay=True,
        full_H1_to_H6_and_hospital_sensor_Nav2_scope_retained=True,
        reduced_repeat_53_cells_is_only_a_proposal_NOT_an_executed_protocol=True,
        unknown_development_or_static_trials_NOT_hidden_in_a_fixed_remaining_drive_count=True,
        generic_English_article_draft_NOT_current_author_review_source=True,
        translate_to_IJCAS_only_after_author_Korean_logic_results_figures_approval=True,
        submission_ready=False,new_PhysX_trial_count=0,actual_motion_authorized=False,
        output_sha256={str(p):hashlib.sha256(value).hexdigest() for p,value in artifacts.items()}))
    print('Saved editable Korean review document, full-scope method/scenario and Figure tables:',target)
    return target


def read_platform_joint_geometry(asset,robot,joint_roles):
    """Read only a local source USD, never an Isaac/Kit or live stage.

    Joint anchors are transformed from each connected body into base_link.
    Current source geometry is not a measured dynamic state or proof that
    an older runtime used these current file bytes.
    """
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    asset=Path(asset).resolve();before=file_sha(asset)
    stage=Usd.Stage.Open(str(asset))
    if stage is None or UsdGeom.GetStageMetersPerUnit(stage)!=1. or str(UsdGeom.GetStageUpAxis(stage))!='Z':
        raise ValueError('Readable metre-based Z-up local source USD required')
    root=stage.GetPrimAtPath(robot);base=stage.GetPrimAtPath(robot+'/base_link')
    if not root.IsValid() or not base.IsValid():raise ValueError('Exact source actor/base required; no replacement root')
    cache=UsdGeom.XformCache();inverse=cache.GetLocalToWorldTransform(base).GetInverse()
    joints={};bodies=[];errors=[]
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        errors.extend(str(error) for error in prim.GetPrimIndex().localErrors)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.RevoluteJoint):joints[prim.GetName()]=prim
    if errors or len(joints)!=12 or len(bodies)!=13 or set(joint_roles.values())!=set(joints):
        raise ValueError('Complete exact thirteen-body/twelve-joint source geometry required')
    records={}
    for role,name in joint_roles.items():
        joint=UsdPhysics.RevoluteJoint(joints[name])
        first,last=[list(map(str,relationship.GetTargets())) for relationship in (joint.GetBody0Rel(),joint.GetBody1Rel())]
        if (len(first)!=1 or len(last)!=1 or any(not p.startswith(robot+'/') for p in (*first,*last))):
            raise ValueError('One source body at each joint endpoint inside the declared actor required')
        positions=[]
        for body,point in ((first[0],joint.GetLocalPos0Attr().Get()),(last[0],joint.GetLocalPos1Attr().Get())):
            if point is None:raise ValueError('Source joint anchor missing')
            transform=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body))
            positions.append(list(map(float,inverse.Transform(transform.Transform(Gf.Vec3d(*point))))))
        if any(not math.isfinite(x) for point in positions for x in point):raise ValueError('Finite source joint anchors required')
        records[role]=dict(path=str(joints[name].GetPath()),body0=first[0],body1=last[0],axis=str(joint.GetAxisAttr().Get()),
            body0_anchor_in_base_m=positions[0],body1_anchor_in_base_m=positions[1],
            two_source_anchor_distance_m=math.dist(*positions))
    layers=[Path(layer.realPath).resolve() for layer in stage.GetUsedLayers() if layer.realPath]
    pins={str(path):file_sha(path) for path in layers}
    if file_sha(asset)!=before or pins.get(str(asset))!=before:raise ValueError('Source USD changed during read-only geometry extraction')
    return dict(source_asset=str(asset),source_actor=robot,units='metres_Z_up_base_link_coordinates',
        rigid_body_count=len(bodies),revolute_joint_count=len(joints),joints=records,
        maximum_two_source_anchor_distance_m=max(row['two_source_anchor_distance_m'] for row in records.values()),
        source_authored_pose_NOT_a_live_measured_endpoint=True,
        legacy_current_bytes_NOT_invented_as_a_historical_runtime_digest=True,input_sha256=pins)


def platform_geometry_comparison(legacy,current,nominal,bounds,measurements):
    """Keep source geometry, nominal steering and load-support bounds distinct."""
    keys={'body_length_m','body_width_m','wheelbase_m','front_length_m','rear_length_m',
        'mode_curvature_response_scale','mode_lateral_response_scale'}
    if set(nominal)!=keys:raise ValueError('Complete explicit preserved nominal steering parameters required')
    for key in ('body_length_m','body_width_m','wheelbase_m','front_length_m','rear_length_m'):
        if type(nominal[key]) not in (int,float) or not math.isfinite(nominal[key]) or nominal[key]<=0:
            raise ValueError('Positive finite nominal source geometry required')
    if not math.isclose(nominal['front_length_m']+nominal['rear_length_m'],nominal['wheelbase_m'],abs_tol=1e-12):
        raise ValueError('Nominal front/rear lengths must sum to its own wheelbase')
    def geometry(source):
        if source['rigid_body_count']!=13 or source['revolute_joint_count']!=12:
            raise ValueError('Complete thirteen-body/twelve-joint geometry required')
        rows=source['joints']
        names={'front_steering','rear_steering','front_rolling','rear_rolling',
            *[corner+'_'+role for corner in ('fl','fr','rl','rr') for role in ('swivel','rolling')]}
        if set(rows)!=names:raise ValueError('Every exact steering/rolling/caster source role required')
        anchors={name:rows[name]['body0_anchor_in_base_m'] for name in names}
        if (any(len(p)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p) for p in anchors.values())
                or any(rows[name]['axis']!='Z' for name in ('front_steering','rear_steering','fl_swivel','fr_swivel','rl_swivel','rr_swivel'))):
            raise ValueError('Finite source anchors and Z steering/swivel axes required')
        if not anchors['rear_steering'][0]<0<anchors['front_steering'][0]:raise ValueError('Source front/rear axes must straddle the body origin')
        return dict(active_steering_axes_in_base_m={axle:anchors[axle+'_steering'] for axle in ('front','rear')},
            active_pivot_span_m=anchors['front_steering'][0]-anchors['rear_steering'][0],
            passive_caster_swivel_positions_m={corner:anchors[corner+'_swivel'] for corner in ('fl','fr','rl','rr')},
            passive_caster_source_wheel_anchors_m={corner:anchors[corner+'_rolling'] for corner in ('fl','fr','rl','rr')},
            passive_caster_planar_pivot_to_wheel_distance_m={corner:math.dist(anchors[corner+'_swivel'][:2],anchors[corner+'_rolling'][:2]) for corner in ('fl','fr','rl','rr')},
            caster_pivot_track_front_rear_m={end:abs(anchors[a+'_swivel'][1]-anchors[b+'_swivel'][1])
                for end,a,b in (('front','fl','fr'),('rear','rl','rr'))})
    old,new=geometry(legacy),geometry(current)
    if any(type(bounds[k]) not in (int,float) or not math.isfinite(bounds[k]) for k in ('front_support_x_m','rear_support_x_m','track_width_m')):
        raise ValueError('Actual frozen finite load-support parameters required')
    # Geometric identities are conditional only; do NOT reinterpret an existing
    # identified response coefficient as a proven compensation for this ratio.
    ratio=nominal['wheelbase_m']/new['active_pivot_span_m']
    steering=[]
    for mode,front,rear in (('coordinated_counterphase',math.atan(nominal['front_length_m']*.5),-math.atan(nominal['rear_length_m']*.5)),
            ('front_only',math.atan(nominal['wheelbase_m']*.5),0.),('rear_only',0.,-math.atan(nominal['wheelbase_m']*.5))):
        steering.append(dict(mode=mode,illustrative_requested_curvature_1_m=.5,target_front_rad=front,target_rear_rad=rear,
            target_front_deg=math.degrees(front),target_rear_deg=math.degrees(rear),
            geometry_only_current_source_curvature_1_m=(math.tan(front)-math.tan(rear))/new['active_pivot_span_m'],
            not_measured_response_not_initial_joint_assignment=True))
    residuals=dict(front_support_minus_source_axis_m=bounds['front_support_x_m']-new['active_steering_axes_in_base_m']['front'][0],
        rear_support_minus_source_axis_m=bounds['rear_support_x_m']-new['active_steering_axes_in_base_m']['rear'][0],
        track_minus_source_front_caster_pivots_m=bounds['track_width_m']-new['caster_pivot_track_front_rear_m']['front'],
        track_minus_source_rear_caster_pivots_m=bounds['track_width_m']-new['caster_pivot_track_front_rear_m']['rear'])
    return dict(version='SOURCE_GEOMETRY_NOMINAL_STEERING_AND_FROZEN_LOAD_BOUNDS_SEPARATED_v1',
        status='SOURCE_AND_PARAMETER_PROVENANCE_REVIEW_NOT_BODY_RESPONSE_MODEL_VALIDATION',
        legacy_CAD_source=old,current_hospital_USD_source=new,preserved_nominal_steering=nominal,
        frozen_Newton_Euler_support_parameters={k:bounds[k] for k in ('front_support_x_m','rear_support_x_m','track_width_m')},
        frozen_support_vs_current_source_residuals_m=residuals,
        numeric_source_float_consistency_tolerance_m=1e-6,
        frozen_support_positions_match_current_source_to_float_precision=all(abs(x)<=1e-6 for x in residuals.values()),
        nominal_wheelbase_minus_current_source_span_m=nominal['wheelbase_m']-new['active_pivot_span_m'],
        nominal_command_to_source_geometry_only_curvature_ratio=ratio,illustrative_nominal_target_angles=steering,
        conditional_geometry_formula='kappa=(tan(delta_f)-tan(delta_r))/L_source, fixed body-longitudinal velocity, no slip/bias/lag/disturbance',
        geometry_ratio_is_NOT_a_response_fit_or_proven_cause_of_identified_coefficients=True,
        existing_measurement_record_unresolved=list(measurements['physical_geometry_evidence']['unresolved']),
        all_nominal_response_constants_runtime_results_and_existing_FAIL_HOLD_preserved=True,
        source_schematic_NOT_cooked_contact_full_3D_patient_or_hardware_safety_validation=True,
        body_response_validation_complete=False,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def draw_platform_geometry_figure(report,paths):
    """Korean Figure 2: actual source geometry and nominal COMMAND diagrams."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle,FancyArrowPatch
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    with plt.rc_context({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':9,'pdf.fonttype':42}):
        fig=plt.figure(figsize=(10.2,7.4),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1.4,1.])
        main=fig.add_subplot(grid[0,:]);new=report['current_hospital_USD_source'];nominal=report['preserved_nominal_steering']
        length,width=nominal['body_length_m'],nominal['body_width_m'];margin=.010
        main.add_patch(Rectangle((-length/2-margin,-width/2-margin),length+2*margin,width+2*margin,
            facecolor='#dceef1',edgecolor='#077e8d',linewidth=1.5,linestyle='--',label='외곽선 + 공통 여유 10 mm'))
        main.add_patch(Rectangle((-length/2,-width/2),length,width,facecolor='none',edgecolor='#374151',linewidth=1.2,label='계획 외곽선 1.630 × 0.965 m'))
        main.plot([-.77,.77],[0.,0.],color='#a3aab4',lw=6,zorder=2)
        for corner,p in new['passive_caster_swivel_positions_m'].items():
            wheel=new['passive_caster_source_wheel_anchors_m'][corner]
            main.plot([p[0],wheel[0]],[p[1],wheel[1]],color='#616d7b',lw=2)
            main.scatter([p[0]],[p[1]],s=35,facecolor='white',edgecolor='#616d7b',zorder=4,label='캐스터 피벗' if corner=='fl' else None)
            main.scatter([wheel[0]],[wheel[1]],s=16,color='#616d7b',zorder=4,label='캐스터 차축 앵커' if corner=='fl' else None)
            main.annotate(corner.upper(),p[:2],xytext=(0,12 if p[1]>0 else -18),textcoords='offset points',ha='center',fontsize=8)
        for axle,p in new['active_steering_axes_in_base_m'].items():
            main.scatter([p[0]],[p[1]],s=80,marker='s',color='#b94c39',zorder=5)
            main.text(p[0],.08,'전륜 능동축' if axle=='front' else '후륜 능동축',ha='center',color='#9e3e30',fontsize=9)
        main.scatter([0.],[0.],s=35,color='#111827',zorder=5);main.text(0.,.08,'차체 기준점',ha='center')
        main.annotate('',xy=(.26,-.18),xytext=(0.,-.18),arrowprops=dict(arrowstyle='->',color='#243247'))
        main.text(.28,-.18,'+x',va='center');main.annotate('',xy=(0.,-.04),xytext=(0.,-.18),arrowprops=dict(arrowstyle='->',color='#243247'))
        main.text(-.045,-.035,'+y',ha='right')
        lf=new['active_steering_axes_in_base_m']['front'][0];lr=new['active_steering_axes_in_base_m']['rear'][0]
        main.annotate('',xy=(lf,-.61),xytext=(lr,-.61),arrowprops=dict(arrowstyle='<->',color='#b94c39'))
        main.text(0.,-.65,'현재 USD 능동축 간격 1.330 m',ha='center',va='top',color='#9e3e30')
        main.set_title('(a) 현재 시뮬레이션 소스의 축·캐스터 위치와 계획 외곽선',fontsize=11,pad=9)
        main.set(xlim=(-1.5,1.5),ylim=(-.85,.85),aspect='equal',xlabel='차체 x (m)',ylabel='차체 y (m)')
        main.spines[['top','right']].set_visible(False);main.legend(loc='upper right',fontsize=8,ncol=2,framealpha=.95)
        names={'coordinated_counterphase':'역상 조향','front_only':'전륜 단독 조향','rear_only':'후륜 단독 조향'}
        colors=['#087c8b','#5661a7','#b86b27']
        for i,(row,color) in enumerate(zip(report['illustrative_nominal_target_angles'],colors)):
            ax=fig.add_subplot(grid[1,i]);a=nominal['front_length_m'];b=nominal['rear_length_m']
            ax.plot([-b,a],[0.,0.],color='#a3aab4',lw=4)
            for x,angle,label in ((a,row['target_front_rad'],'전'),(-b,row['target_rear_rad'],'후')):
                direction=(math.cos(angle),math.sin(angle));span=.21
                ax.plot([x-span*direction[0],x+span*direction[0]],[-span*direction[1],span*direction[1]],color=color,lw=6,solid_capstyle='round')
                ax.scatter([x],[0.],s=18,color='#111827',zorder=4)
                ax.add_patch(FancyArrowPatch((x,0.),(x+.32*direction[0],.32*direction[1]),arrowstyle='->',mutation_scale=10,color=color))
                ax.text(x,-.23,label+' '+f'{math.degrees(angle):+.1f}°',ha='center',fontsize=9)
            ax.axhline(0.,color='#94a3b8',lw=.7,ls=':',zorder=0)
            ax.text(0.,.42,names[row['mode']],ha='center',color=color,fontsize=11)
            ax.text(0.,-.47,'명목 L = 1.290 m\nκ 명령 = +0.50 /m',ha='center',fontsize=9)
            ax.set(xlim=(-1.,1.12),ylim=(-.74,.68),aspect='equal');ax.axis('off')
        fig.suptitle('플랫폼 형상과 세 조향 모드의 명령 기하',fontsize=14,color='#17384b')
        fig.supxlabel('(b) 명령각 설명도: 측정 응답·초기 캐스터각 지정·새 실험이 아님\n1.290 m 명목 조향 기하와 1.330 m 소스/하중 지지 기하는 구분하여 사용',fontsize=9)
        try:
            for path in paths:fig.savefig(path,dpi=220,metadata={'Creator':'IJCAS source-bound offline platform review'})
        finally:plt.close(fig)


def write_platform_geometry_review(output=None):
    """Source-bound platform Figure 2; do not overwrite an author's manuscript."""
    import inspect,io
    import patient_initial_stability.steering_lattice as lattice
    from hospital_runtime_evidence_io import write_exclusive
    legacy_roles={'front_steering':'front_steering_joint','rear_steering':'rear_steering_joint',
        'front_rolling':'front_wheel_joint','rear_rolling':'rear_wheel_joint',
        **{corner+'_'+role:'caster_'+corner+('_swivel_joint' if role=='swivel' else '_wheel_joint')
            for corner in ('fl','fr','rl','rr') for role in ('swivel','rolling')}}
    names={'fl':'front_left','fr':'front_right','rl':'rear_left','rr':'rear_right'}
    current_roles={'front_steering':'base_link_to_front_active_steering','rear_steering':'base_link_to_rear_active_steering',
        'front_rolling':'front_active_steering_to_drive_wheel','rear_rolling':'rear_active_steering_to_drive_wheel',
        **{corner+'_'+role:('base_link_to_'+name+'_caster' if role=='swivel' else name+'_caster_to_wheel')
            for corner,name in names.items() for role in ('swivel','rolling')}}
    legacy=read_platform_joint_geometry(_ijcas_path(WORKSPACE, 'src/patient_transport_description/urdf/patient_transport/patient_transport.usd'),'/patient_transport',legacy_roles)
    current=read_platform_joint_geometry(_ijcas_path(WORKSPACE, 'isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd'),'/World/patient_transport_lab_screening',current_roles)
    model_path=RESULT/'planning/hospital_h1_planning_r06/P_dynamics_full_envelope.json'
    retained_path=RESULT/'hospital_h4_experiments/cohort_09_full_task_r01/route/hospital_h4_cohort_09_full_task_r01_P_dynamics_full_envelope_r01.json'
    model=json.loads(model_path.read_bytes());retained=json.loads(retained_path.read_bytes())
    if (file_sha(model_path)!='1d3f6978a5bb3bb6085a2d8882fe3855f1811599197cbc4365a223d874b292ac'
            or retained['input_sha256'].get(str(model_path))!=file_sha(model_path)
            or retained['input_sha256'].get(current['source_asset'])!=current['input_sha256'][current['source_asset']]
            or retained['input_sha256'].get(str(Path(lattice.__file__).resolve()))!=file_sha(lattice.__file__)):
        raise ValueError('Exact preserved source/nominal/Newton-Euler bindings from actual H4 required')
    nominal=dict(body_length_m=lattice.BODY_LENGTH_M,body_width_m=lattice.BODY_WIDTH_M,wheelbase_m=lattice.WHEELBASE_M,
        front_length_m=lattice.FRONT_LENGTH_M,rear_length_m=lattice.REAR_LENGTH_M,
        mode_curvature_response_scale=dict(lattice.MODE_CURVATURE_RESPONSE_SCALE),
        mode_lateral_response_scale=dict(lattice.MODE_LATERAL_RESPONSE_SCALE))
    measurements=load_yaml(CONFIG/'patient_transport_sim_platform.yaml')
    report=platform_geometry_comparison(legacy,current,nominal,model['rigid_body_parameter_bounds'],measurements)
    sources=(model_path,retained_path,CONFIG/'patient_transport_sim_platform.yaml',
        _ijcas_path(WORKSPACE, 'src/patient_transport_description/config/patient_transport_lab_measurements.yaml'),
        _ijcas_path(WORKSPACE, 'config/platforms/patient_transport.yaml'),Path(lattice.__file__).resolve())
    pins={**legacy['input_sha256'],**current['input_sha256'],**{str(p):file_sha(p) for p in sources}}
    report.update(source_geometry_records=dict(legacy=legacy,current=current),input_sha256=pins,
        analysis_source={f.__name__:inspect.getsource(f) for f in
            (read_platform_joint_geometry,platform_geometry_comparison,draw_platform_geometry_figure,write_platform_geometry_review)})
    output=(Path(output) if output is not None else _ijcas_path(WORKSPACE, 'studies/IJCAS/reports/platform_geometry_review_new')).resolve()
    target=output/'platform_geometry_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes());expected={k:v for k,v in saved.items() if k!='output_sha256'}
        if expected!=report or any(not Path(p).is_file() or file_sha(p)!=sha for p,sha in saved['output_sha256'].items()):
            raise ValueError('Existing platform figure/review changed; no overwrite of author edits')
        return target
    outputs=[output/('figure_02_platform_geometry.'+suffix) for suffix in ('png','pdf','svg')]
    table=output/'source_geometry.csv';caption=output/'figure_02_caption_ko.txt'
    if any(p.exists() for p in (*outputs,table,caption)):raise ValueError('Unfinalized platform figure output exists; no overwrite')
    output.mkdir(parents=True,exist_ok=True);draw_platform_geometry_figure(report,outputs)
    stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(['항목','이전 CAD 소스','현재 병원 USD','명목 조향 기하','하중 모델','단위'])
    writer.writerow(['능동 조향축 / 전후 지지선 간격',report['legacy_CAD_source']['active_pivot_span_m'],report['current_hospital_USD_source']['active_pivot_span_m'],nominal['wheelbase_m'],model['rigid_body_parameter_bounds']['front_support_x_m']-model['rigid_body_parameter_bounds']['rear_support_x_m'],'m'])
    writer.writerow(['전방 캐스터 피벗 간격',report['legacy_CAD_source']['caster_pivot_track_front_rear_m']['front'],report['current_hospital_USD_source']['caster_pivot_track_front_rear_m']['front'],'해당 없음',model['rigid_body_parameter_bounds']['track_width_m'],'m'])
    for corner in names:
        writer.writerow([corner+' 캐스터 평면 피벗-차축 거리',report['legacy_CAD_source']['passive_caster_planar_pivot_to_wheel_distance_m'][corner],report['current_hospital_USD_source']['passive_caster_planar_pivot_to_wheel_distance_m'][corner],'해당 없음','개별 캐스터 힘 모델 아님','m'])
    with table.open('xb') as f:f.write(stream.getvalue().encode('utf-8-sig'))
    text=('그림 2. 플랫폼 형상과 세 조향 모드의 명령 기하. (a) 현재 병원 실험에 사용된 USD의 조인트 앵커를 차체 기준 좌표로 읽은 능동 조향축과 네 수동 캐스터 위치, 공통 계획 외곽선(1.630 × 0.965 m) 및 각 방향 10 mm 여유. 캐스터 기호는 소스의 작성 자세이며 새로운 초기각을 지정하거나 캐스터를 구동한 결과가 아니다. '
        '(b) 기존 명목 조향 기하(L = 1.290 m)에서 κ = +0.50 m⁻¹에 대한 역상·전륜 단독·후륜 단독 조향 목표각의 설명도. 현재 USD의 능동축 간격과 동결된 Newton–Euler 지지선 간격은 약 1.330 m이다. 목표각 설명도는 유한 조향 전환, 실제 응답, 하중·마찰 모델 또는 물리적 접촉의 검증 결과를 나타내지 않는다. '
        '두 기하 값을 구분하며, 기존 응답 계수가 이 차이를 보상한다는 인과 해석은 하지 않는다.\n')
    with caption.open('x',encoding='utf-8') as f:f.write(text)
    if any(file_sha(p)!=sha for p,sha in pins.items()):raise ValueError('Platform source changed during figure export')
    write_exclusive(target,dict(**report,output_sha256={str(p):file_sha(p) for p in (*outputs,table,caption)}))
    print('Saved source-bound Korean platform Figure 2 / provenance review:',target)
    print('Nominal steering L / current source L / load-support span:',nominal['wheelbase_m'],
        report['current_hospital_USD_source']['active_pivot_span_m'],model['rigid_body_parameter_bounds']['front_support_x_m']-model['rigid_body_parameter_bounds']['rear_support_x_m'])
    print('No live access, response fit, physics trial, manuscript overwrite or model validation.')
    return target


def unified_wheelbase_profile(geometry_review):
    """One forward 1.330m geometry, with legacy values as provenance ONLY.

    Re-express identified yaw gains at FIXED measured steering angles. This
    algebra is not re-identification, a physical fit, or validation of the
    changed command angles. Historical modules and source USD stay untouched.
    """
    nominal=geometry_review['preserved_nominal_steering']
    source=geometry_review['current_hospital_USD_source']
    axes=source['active_steering_axes_in_base_m'];front,rear=axes['front'][0],-axes['rear'][0]
    if (abs(front-.665)>1e-6 or abs(rear-.665)>1e-6
            or not math.isclose(front+rear,1.330,abs_tol=1e-6)
            or any(abs(axes[side][1])>1e-6 for side in ('front','rear'))):
        raise ValueError('Reviewed current symmetric USD axes at +/-0.665m required')
    old_length=nominal['wheelbase_m'];new_length=1.330
    if (not math.isclose(old_length,1.290,abs_tol=1e-12)
            or not math.isclose(nominal['front_length_m']/old_length,.5,abs_tol=1e-12)
            or not math.isclose(nominal['rear_length_m']/old_length,.5,abs_tol=1e-12)):
        raise ValueError('Only the recorded symmetric legacy geometry has this exact scalar transfer')
    modes={'coordinated_counterphase','front_only','rear_only'}
    for name in ('mode_curvature_response_scale','mode_lateral_response_scale'):
        if set(nominal[name])!=modes or any(not math.isfinite(v) or v<=0 for v in nominal[name].values()):
            raise ValueError('Complete finite identified response coefficients required')
    ratio=new_length/old_length
    return dict(version='HOSPITAL_UNIFIED_1330_GEOMETRY_OFFLINE_CANDIDATE_v1',
        status='PROPOSED_FORWARD_PROFILE_NOT_ADOPTED_OR_MOTION_VALIDATED',
        geometry=dict(wheelbase_m=new_length,front_length_m=.665,rear_length_m=.665,
            frame='base_link_NOT_CoM',definition='distance_between_active_steering_axes'),
        mode_curvature_response_scale={mode:value*ratio for mode,value in nominal['mode_curvature_response_scale'].items()},
        mode_lateral_response_scale=dict(nominal['mode_lateral_response_scale']),
        transfer=dict(yaw_gain_multiplier=ratio,
            formula='c_new=c_old*L_new/L_old at FIXED measured front/rear steering',
            lateral_transfer='unchanged only because both old and new axle fractions are exactly 1/2',
            old_geometry_and_gains_for_provenance_only=dict(nominal),
            algebraic_reparameterization_NOT_a_new_identification_or_physical_PASS=True,
            measured_steering_bias_response_times_and_force_bounds_NOT_refitted=True,
            new_commands_paths_and_terminal_conditions_REQUIRE_revalidation=True),
        physical_USD_NE_support_positions_CoM_mass_gains_and_limits_unchanged=True,
        new_live_profile_installed=False,actual_motion_authorized=False)


def unified_wheelbase_kernel(profile):
    """Private original Python kernels with explicit geometry; no monkeypatch.

    This is an offline diagnostic binding, NOT the live executor, ROS adapter,
    compiled native kernel, or a future motion-admission implementation.
    """
    from types import FunctionType,SimpleNamespace
    import patient_initial_stability.steering_lattice as original
    geometry=profile['geometry']
    if (profile.get('version')!='HOSPITAL_UNIFIED_1330_GEOMETRY_OFFLINE_CANDIDATE_v1'
            or geometry!=dict(wheelbase_m=1.330,front_length_m=.665,rear_length_m=.665,
                frame='base_link_NOT_CoM',definition='distance_between_active_steering_axes')):
        raise ValueError('One explicit reviewed forward geometry required')
    transfer=profile['transfer'];old=transfer['old_geometry_and_gains_for_provenance_only']
    modes=set(original.ALL_MODES)
    if (not math.isclose(old['wheelbase_m'],1.290,abs_tol=1e-12)
            or not math.isclose(old['front_length_m'],.645,abs_tol=1e-12)
            or not math.isclose(old['rear_length_m'],.645,abs_tol=1e-12)
            or profile['new_live_profile_installed'] is not False
            or profile['actual_motion_authorized'] is not False):
        raise ValueError('Offline symmetric basis transfer only, without live admission')
    for field in ('mode_curvature_response_scale','mode_lateral_response_scale'):
        if set(profile[field])!=modes or set(old[field])!=modes:
            raise ValueError('Complete original and transferred coefficient modes required')
        frozen=(original.MODE_CURVATURE_RESPONSE_SCALE if field=='mode_curvature_response_scale'
            else original.MODE_LATERAL_RESPONSE_SCALE)
        if old[field]!=frozen:raise ValueError('Original identified coefficients must remain unchanged')
        for mode in modes:
            expected=old[field][mode]*(1.330/1.290 if field=='mode_curvature_response_scale' else 1.)
            if not math.isfinite(profile[field][mode]) or not math.isclose(profile[field][mode],expected,rel_tol=1e-12):
                raise ValueError('Only exact coefficient basis transfer is supported')
    values=dict(WHEELBASE_M=geometry['wheelbase_m'],FRONT_LENGTH_M=geometry['front_length_m'],
        REAR_LENGTH_M=geometry['rear_length_m'],MODE_CURVATURE_RESPONSE_SCALE=dict(profile['mode_curvature_response_scale']),
        MODE_LATERAL_RESPONSE_SCALE=dict(profile['mode_lateral_response_scale']))
    def bind(function,**extra):
        if function.__closure__:raise ValueError('Closure-free original kernel required')
        result=FunctionType(function.__code__,dict(function.__globals__,**values,**extra),function.__name__,function.__defaults__)
        result.__kwdefaults__=function.__kwdefaults__;return result
    targets=bind(original.steering_targets)
    return SimpleNamespace(steering_targets=targets,propagate=bind(original.propagate,steering_targets=targets),
        global_legacy_source_and_module_NOT_modified=True)


def wheelbase_route_replay(plan,profile,*,legacy_h4_nominal_context=None):
    """Replay both parameterizations, never overwrite an old plan's states.

    NO map/support/resource/closed-loop check is performed here. The changed
    endpoints are a diagnostic requiring NEW planning, not an accepted route.
    """
    import patient_initial_stability.steering_lattice as original
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    if not plan.get('plan_success') or len(plan['selected_actions'])!=len(plan['selected_states']):
        raise ValueError('Own complete frozen plan required for geometry comparison')
    if not plan['selected_actions']:
        raise ValueError('Nonempty frozen route required')
    dynamics={'full':original.FULL_DYNAMICS,'kinematic':original.KINEMATIC_DYNAMICS,
        'transition_cost':original.TRANSITION_COST_DYNAMICS}[plan['planner_dynamics']]
    initial_keys=('initial_front_rad','initial_rear_rad','initial_steering_propagated')
    if all(key in plan for key in initial_keys):
        initial=tuple(plan[key] for key in initial_keys)
        bounds=plan.get('rigid_body_parameter_bounds')
        initial_basis='EXPLICIT_FROZEN_NOMINAL_PLAN_INITIAL_STATE_NOT_PHYSICAL_STATE'
    elif (not any(key in plan for key in initial_keys[:2])
            and plan.get('scenario')=='H4_ward_corridor_turn'
            and plan.get('generator_selector_version') in (
                'H4_shared_kinematic_and_full_model_SAS_control_count_search_v2',
                'H4_ALL_FOUR_COMMON_TERMINAL_INTERFACE_NATIVE_BUNDLE_V2')
            and legacy_h4_nominal_context is not None
            and plan.get('initial_steering_propagated') is (model_full:=plan['planner_dynamics']=='full')):
        # These exact original generators explicitly construct State(0,0,0,
        # counterphase,0,0,full); the old bundle omitted the two angle fields.
        # Require source-bound context and verify EVERY saved state below.
        initial=(0.,0.,model_full)
        bounds=legacy_h4_nominal_context['rigid_body_parameter_bounds']
        initial_basis='SOURCE_BOUND_ORIGINAL_H4_NOMINAL_ZERO_INITIALIZATION_NOT_PHYSICAL_STATE'
    else:
        return dict(status='UNREVIEWED_LEGACY_INITIAL_SCHEMA_NO_MIGRATION_CONCLUSION',
            actual_motion_authorized=False,existing_verdict_NOT_changed=True)
    model=(MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
        if plan['planner_dynamics']=='full' else None)
    def replay(propagate,*,check_saved):
        state=original.State(0.,0.,0.,'coordinated_counterphase',*initial)
        error=0.;infeasible=0;duration=0.
        for action,saved in zip(plan['selected_actions'],plan['selected_states']):
            state,_,elapsed=propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
                speed_m_s=plan['common_max_command_speed_m_s'],dynamics=dynamics,rigid_body_model=model,
                target_speed_m_s=action['target_speed_m_s'])
            infeasible+=int(not state.rigid_body_feasible);duration+=elapsed
            if check_saved:
                values=(state.x-saved['x_m'],state.y-saved['y_m'],original.wrap(state.yaw-math.radians(saved['yaw_deg'])),
                    state.front-saved['front_rad'],state.rear-saved['rear_rad'])
                if model:values+=tuple(getattr(state,key)-saved[field] for key,field in
                    (('longitudinal_velocity','u_m_s'),('lateral_velocity','v_m_s'),('yaw_rate','yaw_rate_rad_s')))
                error=max(error,*(abs(value) for value in values))
        return state,error,infeasible,duration
    old,error,old_bad,old_time=replay(original.propagate,check_saved=True)
    if error>1e-9:
        return dict(status='LEGACY_STATE_REPLAY_NOT_VERIFIED_NO_MIGRATION_CONCLUSION',
            maximum_legacy_state_difference=error,actual_motion_authorized=False)
    new,_,new_bad,new_time=replay(unified_wheelbase_kernel(profile).propagate,check_saved=False)
    goal=plan['goal_local_xyyaw']
    def metrics(state,duration,infeasible):
        position=math.hypot(state.x-goal[0],state.y-goal[1]);heading=abs(math.degrees(original.wrap(state.yaw-goal[2])))
        speed=math.hypot(state.longitudinal_velocity,state.lateral_velocity)
        return dict(terminal_position_error_m=position,terminal_heading_error_deg=heading,
            terminal_model_speed_m_s=speed if model else None,model_duration_s=duration,
            original_25mm_2deg_goal_test=position<=.025 and heading<=2.,infeasible_NE_edges=infeasible)
    return dict(status='BOTH_PARAMETERIZATIONS_REPLAYED_OFFLINE_ONLY',maximum_legacy_state_difference=error,
        initial_state_basis=initial_basis,
        legacy=metrics(old,old_time,old_bad),unified1330=metrics(new,new_time,new_bad),
        changed_endpoint_distance_m=math.hypot(new.x-old.x,new.y-old.y),
        changed_endpoint_heading_deg=abs(math.degrees(original.wrap(new.yaw-old.yaw))),
        collision_support_resource_and_closed_loop_execution_NOT_checked=True,
        actual_motion_authorized=False,existing_verdict_NOT_changed=True)


def write_wheelbase_migration_review(root=RESULT):
    """One versioned forward candidate and all22 frozen-plan impact records."""
    import patient_initial_stability.steering_lattice as original
    from hospital_runtime_evidence_io import write_exclusive
    root=Path(root).resolve();pins={}
    def read(path):
        path=Path(path).resolve();pins[str(path)]=file_sha(path);return json.loads(path.read_bytes())
    source=_ijcas_path(WORKSPACE, 'studies/IJCAS/reports/technical_evidence/figures/figure_02_platform_geometry/platform_geometry_review.json')
    review=read(source)
    asset=review['source_geometry_records']['current']['source_asset']
    if file_sha(asset)!=review['input_sha256'][asset]:raise ValueError('Reviewed source USD changed')
    pins[asset]=file_sha(asset);profile=unified_wheelbase_profile(review);kernel=unified_wheelbase_kernel(profile)
    before=(original.WHEELBASE_M,original.FRONT_LENGTH_M,original.REAR_LENGTH_M,dict(original.MODE_CURVATURE_RESPONSE_SCALE))
    angles=[];maximum_angle=0.;yaw_error=lateral_error=0.
    for mode in original.ALL_MODES:
        for curvature in original.CURVATURES:
            old=original.steering_targets(mode,curvature);new=kernel.steering_targets(mode,curvature)
            maximum_angle=max(maximum_angle,*(abs(value) for value in new))
            angles.append(dict(mode=mode,curvature_1_m=curvature,legacy_target_deg=list(map(math.degrees,old)),
                unified1330_target_deg=list(map(math.degrees,new))))
        for front in range(-45,46,5):
            for rear in range(-45,46,5):
                a,b=math.tan(math.radians(front)),math.tan(math.radians(rear))
                yaw_error=max(yaw_error,abs((a-b)/original.WHEELBASE_M*original.MODE_CURVATURE_RESPONSE_SCALE[mode]-
                    (a-b)/1.330*profile['mode_curvature_response_scale'][mode]))
                lateral_error=max(lateral_error,abs((original.REAR_LENGTH_M*a+original.FRONT_LENGTH_M*b)/original.WHEELBASE_M-
                    (.665*a+.665*b)/1.330))
    if yaw_error>1e-12 or lateral_error>1e-12 or maximum_angle>math.radians(45.):
        raise ValueError('Exact fixed-angle representation and original45deg steering limit required')
    h4_context=read(_ijcas_path(root, 'hospital_h4_shared_model_seed_screen/screen_r01/shared_model_seed_screen.json'))
    if h4_context['version']!='H4_shared_kinematic_and_full_model_SAS_control_count_search_v2':
        raise ValueError('Original H4 nominal initialization/model context required')
    for name in ('screen_hospital_h4_shared_spatial_candidates.py','screen_hospital_h4_model_seed_candidates.py'):
        path=_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts', name)
        if h4_context['input_sha256'].get(str(path))!=file_sha(path):
            raise ValueError('Original H4 nominal-initialization generator changed')
        pins[str(path)]=file_sha(path)
    rows=[]
    for task,method,path,planning_only in common_comparison_references(root):
        raw=read(path)
        if planning_only:
            rows.append(dict(task=task,method=method,role='PRIOR_BOUNDED_NO_PLAN_REQUIRES_NEW_GEOMETRY_SEARCH',
                original_status=raw['status'],new_plan_or_drive_inferred=False));continue
        packet=read(raw['protocol']) if raw.get('protocol') else {}
        plan_path=Path(raw.get('plan_path') or packet['original_plan']);plan=read(plan_path)
        expected=raw.get('plan_sha256') or packet.get('input_sha256',{}).get(str(plan_path.resolve()))
        if not expected or expected!=file_sha(plan_path):raise ValueError('Frozen original plan required')
        replay=wheelbase_route_replay(plan,profile,legacy_h4_nominal_context=h4_context if task=='H4' else None)
        rows.append(dict(task=task,method=method,original_result=str(path),original_verdict=raw['verdict'],
            original_plan=str(plan_path),diagnostic=replay))
        print('[WHEELBASE OFFLINE]',task,method,replay['status'],flush=True)
    if before!=(original.WHEELBASE_M,original.FRONT_LENGTH_M,original.REAR_LENGTH_M,dict(original.MODE_CURVATURE_RESPONSE_SCALE)):
        raise ValueError('Legacy namespace changed during private diagnostic')
    for path in (__file__,original.__file__,_ijcas_path(WORKSPACE, 'src/patient_initial_stability/patient_initial_stability/rigid_body_edge.py')):
        pins[str(Path(path).resolve())]=file_sha(path)
    for path,expected in pins.items():
        if file_sha(path)!=expected:raise ValueError('Diagnostic input changed: '+path)
    verified=[row for row in rows if row.get('diagnostic',{}).get('status')=='BOTH_PARAMETERIZATIONS_REPLAYED_OFFLINE_ONLY']
    requires_replanning=[dict(task=row['task'],method=row['method']) for row in verified
        if not row['diagnostic']['unified1330']['original_25mm_2deg_goal_test']]
    summary=dict(version='UNIFIED1330_FORWARD_MIGRATION_IMPACT_NOT_NEW_PHYSICAL_RESULTS_v1',
        status='OFFLINE_PROFILE_AND_FROZEN_CONTROL_REPROPAGATION_ONLY',profile=profile,angle_queries=angles,
        fixed_measured_angle_formula_transfer=dict(maximum_yaw_curvature_difference_1_m=yaw_error,
            maximum_lateral_ratio_difference=lateral_error,query_count=3*19*19,
            not_a_new_fit_or_validation_against_measured_body_motion=True),
        original_frozen_controls_replayed_without_replanning=rows,
        counts=dict(comparison_references=len(rows),verified_frozen_plan_replays=len(verified),
            prior_planning_only_references=sum('diagnostic' not in row for row in rows),
            frozen_controls_outside_original_model_pose_goal_after_change=len(requires_replanning)),
        unchanged_frozen_controls_requiring_new_planning=requires_replanning,
        remaining_model_pose_goal_matches_are_NOT_collision_or_execution_PASS=True,
        all_legacy_sources_results_USD_and_live_state_unchanged=True,
        new_full_native_plans_closed_loop_checks_and_live_admission_required=True,
        automatic_speed_policy_and_compiled_native_geometry_migration_NOT_completed_here=True,
        new_PhysX_trials=0,actual_motion_authorized=False,input_sha256=pins)
    key=hashlib.sha256(json.dumps(summary,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    target=_ijcas_path(root, 'paper_evidence_inventory/unified_wheelbase_review', key, 'migration_review.json')
    if target.exists():
        if json.loads(target.read_bytes())!=summary:raise ValueError('Existing migration review differs; no overwrite')
    else:write_exclusive(target,summary)
    print('Saved unified-wheelbase candidate and impact review:',target)
    return target


def unified_wheelbase_connections(profile,*,include_compiled=False):
    """Private OFFLINE command/timing/native bindings of the same geometry.

    No historical source/module is patched; no live controller is installed.
    Exact reviewed literal counts fail closed if a source changes. Native
    fallback must use the NEW Python kernel, never the historical geometry.
    """
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace
    import hospital_path_execution as direct
    import hospital_path_execution_v2 as terminal
    import hospital_timed_path_execution as timed
    import patient_initial_stability.steering_lattice as lattice
    kernel=unified_wheelbase_kernel(profile);bindings=[]
    def geometry_bind(function,expected):
        tree=ast.parse(textwrap.dedent(inspect.getsource(function)));counts={1.29:0,.645:0}
        class Geometry(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) is float and node.value in counts:
                    counts[node.value]+=1
                    return ast.copy_location(ast.Constant({1.29:1.330,.645:.665}[node.value]),node)
                return node
        tree=Geometry().visit(tree)
        if counts!=expected:raise ValueError('Unreviewed geometry literal anchors: '+function.__qualname__)
        namespace=dict(function.__globals__)
        exec(compile(ast.fix_missing_locations(tree),'<offline_unified1330_geometry>','exec'),namespace)
        bindings.append(dict(function=function.__qualname__,source=inspect.getfile(function),
            source_sha256=file_sha(inspect.getfile(function)),reviewed_literal_counts={str(k):v for k,v in counts.items()}))
        return namespace[tree.body[0].name]
    direct_targets=geometry_bind(direct.direct_joint_targets,{1.29:2,.645:2})
    steering_geometry=geometry_bind(terminal.steering_geometry,{1.29:2,.645:2})
    angles=geometry_bind(timed.steering_angles,{1.29:2,.645:2})
    leg=type('OfflineUnified1330SpatialLeg',(timed.SpatialLeg,),
        {'__init__':geometry_bind(timed.SpatialLeg.__init__,{1.29:0,.645:1})})
    timing=type('OfflineUnified1330LegTiming',(timed.LegTiming,),
        {'__init__':geometry_bind(timed.LegTiming.__init__,{1.29:1,.645:1})})
    compiled=None
    if include_compiled:
        import hospital_mission_planning as mission
        facade=SimpleNamespace(**dict(vars(lattice),WHEELBASE_M=1.330,FRONT_LENGTH_M=.665,REAR_LENGTH_M=.665,
            MODE_CURVATURE_RESPONSE_SCALE=dict(profile['mode_curvature_response_scale']),
            MODE_LATERAL_RESPONSE_SCALE=dict(profile['mode_lateral_response_scale']),
            steering_targets=kernel.steering_targets,propagate=kernel.propagate))
        factory=mission.compiled_original_NE_propagation
        tree=ast.parse(textwrap.dedent(inspect.getsource(factory)));replaced=[]
        class PrivateLattice(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                if (node.module=='patient_initial_stability' and len(node.names)==1
                        and node.names[0].name=='steering_lattice' and node.names[0].asname=='lattice'):
                    replaced.append(True)
                    return ast.copy_location(ast.Assign([ast.Name('lattice',ast.Store())],
                        ast.Name('_offline_unified_lattice',ast.Load())),node)
                return node
        tree=PrivateLattice().visit(tree)
        if len(replaced)!=1:raise ValueError('One original lattice-import anchor required')
        namespace=dict(factory.__globals__,_offline_unified_lattice=facade)
        exec(compile(ast.fix_missing_locations(tree),'<offline_unified1330_native_factory>','exec'),namespace)
        compiled=namespace[factory.__name__]()
        compiled.offline_profile_version=profile['version']
        bindings.append(dict(function=factory.__qualname__,source=inspect.getfile(factory),
            source_sha256=file_sha(inspect.getfile(factory)),private_lattice_import_substitutions=1))
    return SimpleNamespace(python=kernel,compiled=compiled,direct_joint_targets=direct_targets,
        steering_geometry=steering_geometry,steering_angles=angles,SpatialLeg=leg,LegTiming=timing,
        bindings=bindings,live_controller_installed=False,actual_motion_authorized=False)


def unified_mission_model_binding(profile,geometry,plan,*,simulation_budget_s):
    """Same common controller/NE screen with explicit1.330m geometry only.

    Private command, measured alignment AND plant bindings are required.
    Neither module globals nor an Isaac controller are installed. The model
    still has no captured-wheel/contact forces and uses software-only ACKs.
    """
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace
    import hospital_mission_execution as mission
    import hospital_mission_sequence as sequence
    import screen_hospital_h4_common_executor as screen
    connections=unified_wheelbase_connections(profile)
    if plan.get('geometry_profile')!=profile:
        raise ValueError('The exact unified profile must belong to this frozen plan')
    mission.validate_mission_plan(plan,geometry)
    CommonController=sequence.unified_geometry_motion_class(profile)
    # Reuse the runtime's EXACT pose/time/reverse/direction/SAT/support code.
    # Its physical contact/tilt prefix is deliberately NOT supplied with
    # invented measurements: the planar NE screen has no such channels.
    guard_type=sequence.lifecycle.MissionRuntimeGuard
    guard_tree=ast.parse(textwrap.dedent(inspect.getsource(guard_type._check)))
    body=guard_tree.body[0].body
    anchors=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='stamp']
    if len(anchors)!=1:raise ValueError('Exact original runtime resource prefix anchor required')
    guard_tree.body[0].body=body[anchors[0]:]
    guard_namespace=dict(guard_type._check.__globals__)
    exec(compile(ast.fix_missing_locations(guard_tree),'<original_model_pose_resource_guard_without_physical_prefix>','exec'),guard_namespace)
    ModelGuard=type('ModelPoseResourceGuard',(guard_type,),{'_check':guard_namespace['_check']})
    instances=[];model_grid={}
    class Controller(CommonController):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.model_resource_guard=(ModelGuard(geometry,model_grid['occupied'],model_grid['origin'],model_grid['resolution'],
                simulation_budget_s=simulation_budget_s) if model_grid else None)
            instances.append(self)
        def step(self,timestamp,pose,*args,**kwargs):
            info=super().step(timestamp,pose,*args,**kwargs)
            if not all(math.isfinite(float(v)) for v in (timestamp,*pose)):
                raise ValueError('Finite MODEL clock/pose required')
            if self.model_resource_guard is None:raise ValueError('Own model raster guard not bound')
            self.model_resource_guard(dict(time_s=timestamp,x_m=pose[0],y_m=pose[1],yaw_rad=pose[2]),info,{},[])
            return info
    # Original screen has exactly two L and two half-L literals. All force,
    # velocity integration, sampled SAT/support and acceptance code survives.
    tree=ast.parse(textwrap.dedent(inspect.getsource(screen.case)))
    counts={1.29:0,.645:0}
    class Geometry(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is float and node.value in counts:
                counts[node.value]+=1
                return ast.copy_location(ast.Constant({1.29:1.330,.645:.665}[node.value]),node)
            return node
    tree=Geometry().visit(tree)
    if counts!={1.29:2,.645:2}:raise ValueError('Unreviewed MODEL plant geometry anchors')
    if plan.get('measured_replan_start') is not None:
        initializers=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and len(n.targets)==1 and isinstance(n.targets[0],ast.Tuple)
            and [v.id for v in n.targets[0].elts if isinstance(v,ast.Name)]==['velocity','actual','commanded']]
        if len(initializers)!=1:raise ValueError('One original MODEL initial-state statement required')
        initializers[0].value=ast.parse("(np.asarray(plan['initial_u_v_r'],float).copy(), "
            "np.asarray([plan['initial_front_rad'],plan['initial_rear_rad']],float), "
            "tuple(plan['measured_replan_start']['captured_steering_targets_rad']))",mode='eval').body
        first_stamps=[n for n in ast.walk(tree) if isinstance(n,ast.If)
            and ast.dump(n.test,include_attributes=False)==ast.dump(
                ast.parse('controller.previous_time is None',mode='eval').body,include_attributes=False)]
        if len(first_stamps)!=1:raise ValueError('One original first MODEL control-stamp anchor required')
        # The first resumed control command has no elapsed control interval,
        # exactly like the real SDK bridge. Repeated float additions can put
        # its tick 1e-15s above literal2. A nonzero retained target then rounds
        # by one ULP and produces a fictitious >8deg/s quotient. Initialize
        # both stamps from that SAME tick; keep every later8deg/s check exact.
        first_stamps[0].body.append(ast.parse('last_command_time = now').body[0])
    namespace=dict(screen.case.__globals__,
        MODE_CURVATURE_RESPONSE_SCALE=dict(profile['mode_curvature_response_scale']),
        MODE_LATERAL_RESPONSE_SCALE=dict(profile['mode_lateral_response_scale']))
    exec(compile(ast.fix_missing_locations(tree),'<offline_unified1330_closed_loop_plant>','exec'),namespace)
    private_screen=SimpleNamespace(**dict(vars(screen),case=namespace['case'],
        execution=SimpleNamespace(**dict(vars(screen.execution),direct_joint_targets=connections.direct_joint_targets))))
    # Keep the original mission factory/ACK sequencing; substitute its ONE
    # imported screen, rather than mutating any historical module instance.
    factory=mission.mission_model_case_binding
    tree=ast.parse(textwrap.dedent(inspect.getsource(factory)));imports=[]
    class Screen(ast.NodeTransformer):
        def visit_Import(self,node):
            if len(node.names)==1 and node.names[0].name=='screen_hospital_h4_common_executor' and node.names[0].asname=='original':
                imports.append(True)
                return ast.copy_location(ast.Assign([ast.Name('original',ast.Store())],ast.Name('_unified_screen',ast.Load())),node)
            return node
    tree=Screen().visit(tree)
    if len(imports)!=1:raise ValueError('One original MODEL screen import required')
    namespace=dict(factory.__globals__,_unified_screen=private_screen,MissionTrackingController=Controller,
        mission_execution_contract=sequence.final_target_hold_motion_contract)
    exec(compile(ast.fix_missing_locations(tree),'<offline_unified1330_mission_factory>','exec'),namespace)
    run,execution,design=namespace[factory.__name__](geometry,plan,simulation_budget_s=simulation_budget_s)
    def checked_run(candidate,bounds,dt,occupied,origin,resolution,support):
        # Construct the own guard only when its exact map is supplied. The
        # original MODEL factory constructs Controller inside this call.
        model_grid.update(occupied=occupied,origin=origin,resolution=resolution)
        report,rows=run(candidate,bounds,dt,occupied,origin,resolution,support)
        report.update(geometry_profile=profile,
            original_NE_plant_integration_and_collision_support_checks_code_unchanged=False,
            original_plant_changed_ONLY_geometry_literals_and_transferred_response_basis=True,
            force_bounds_integration_SAT_support_guards_and_acceptance_unchanged=True,
            command_and_measured_alignment_and_plant_share_one_wheelbase=True,
            actual_directional_tracking_floor_forward_reverse_m_s=[.010,.020],
            original_case_geometry_literal_substitutions={str(k):v for k,v in counts.items()},
            original_common_controller_except_geometry_unchanged=True,
            original_runtime_pose_reverse_and_all_terminal_direction_commands_checked=True,
            MODEL_resource_guard=instances[0].model_resource_guard.summary(),
            physical_contact_and_tilt_prefix_NOT_fabricated_in_planar_model=True,
            new_PhysX_trials=0,current_live_executor_installed=False,actual_motion_authorized=False)
        return report,rows
    checked_run.connections=connections
    checked_run.model_guard_class=ModelGuard
    return checked_run,execution,design


def write_unified_mission_model_screen(evaluation_path,plan_path,geometry_path,map_path,*,output,simulation_budget_s):
    """Two prespecified nominal resolutions, append-only; no physical trial."""
    import numpy as np
    from hospital_runtime_evidence_io import write_exclusive
    from hospital_mission_lifecycle import freeze_workflow_sources,normalize_runtime_evidence
    evaluation_path,plan_path,geometry_path,map_path=map(lambda p:Path(p).resolve(),
        (evaluation_path,plan_path,geometry_path,map_path))
    evaluation=json.loads(evaluation_path.read_bytes());profile=evaluation['profile']
    source_plan=json.loads(plan_path.read_bytes());geometry=json.loads(geometry_path.read_bytes())
    if source_plan.get('geometry_profile')!=profile or source_plan.get('Isaac_execution_verified') is not False:
        raise ValueError('Explicit not-physically-validated unified plan required')
    plan=dict(source_plan,physical_execution_verified=False,
        integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True)
    if plan.get('geometry_manifest_sha256')!=file_sha(geometry_path):
        raise ValueError('Same recorded task geometry required')
    with np.load(map_path) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    if hashlib.sha256(occupied.tobytes()).hexdigest()!=geometry['planning_occupancy_digest']:
        raise ValueError('Exact recorded common map required')
    # Validate schema/bindings before reserving any generated evidence folder.
    unified_mission_model_binding(profile,geometry,plan,simulation_budget_s=simulation_budget_s)
    sources=[Path(__file__),_ijcas_path(WORKSPACE, 'src/patient_initial_stability/test/test_hospital_paper_protocol.py')]
    for name in ('hospital_mission_execution.py','hospital_mission_sequence.py','hospital_path_execution.py',
            'hospital_path_execution_v2.py','hospital_path_execution_v3.py','hospital_timed_path_execution.py',
            'screen_hospital_h4_common_executor.py','hospital_h4_path_execution.py','hospital_geometry_tools.py',
            'hospital_mission_geometry.py','hospital_mission_lifecycle.py','hospital_runtime_evidence_io.py'):
        sources.append(_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts', name))
    for name in ('rigid_body_edge.py','steering_lattice.py'):
        sources.append(_ijcas_path(WORKSPACE, 'src/patient_initial_stability/patient_initial_stability', name))
    from hospital_mission_lifecycle import host_source_files
    sources=list(dict.fromkeys([*sources,*host_source_files()]))
    pins={str(p):file_sha(p) for p in (evaluation_path,plan_path,geometry_path,map_path,*sources)}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    definition=dict(version='UNIFIED1330_COMMON_DIRECTIONAL_EXECUTOR_MODEL_SCREEN_v1',
        plan=str(plan_path),geometry=str(geometry_path),map=str(map_path),input_sha256=pins,
        integration_steps_s=[.020,.010],simulation_budget_s=simulation_budget_s,
        new_PhysX_trials=0,actual_motion_authorized=False,use_for_final_statistics=False,
        software_resolution_cases_NOT_independent_experiments=True,
        map_compatibility_check_NOT_fresh_current_map_planning_time=True,
        common_controller_and_acceptance_unchanged_except_geometry=True,
        schema_aliases_added_ONLY_to_derived_model_copy=['physical_execution_verified',
            'integrated_stage_portal_bypass_and_continuous_executor_validation_pending'])
    write_exclusive(output/'definition.json',definition)
    write_exclusive(output/'executor_plan.json',plan)
    freeze_workflow_sources(output/'sources.zip',{str(p):pins[str(p)] for p in sources})
    cases=[];files=[output/'definition.json',output/'executor_plan.json',output/'sources.zip']
    for dt in definition['integration_steps_s']:
        run,_,design=unified_mission_model_binding(profile,geometry,plan,simulation_budget_s=simulation_budget_s)
        report,rows=run(plan,plan['rigid_body_parameter_bounds'],dt,occupied,origin,resolution,design)
        trace=output/(str(round(dt*1000))+'ms.csv')
        with trace.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        report.update(trace=str(trace),trace_sha256=file_sha(trace))
        report_path=trace.with_suffix('.json');write_exclusive(report_path,normalize_runtime_evidence(report))
        files.extend((trace,report_path));cases.append(report)
        print('[UNIFIED EXECUTOR MODEL]',dt,report['model_status'],
            'mm/deg',1000*report['final_position_error_m'],report['final_heading_error_deg'],
            'sim_s',report['simulation_duration_s'],'error',report['error'],flush=True)
    for name,expected in pins.items():
        if file_sha(name)!=expected:raise ValueError('MODEL screen input changed: '+name)
    target=output/'model_screen.json'
    write_exclusive(target,normalize_runtime_evidence(dict(definition,cases=cases,
        output_sha256={str(p):file_sha(p) for p in files},
        both_nominal_resolutions_pass=all(c['model_status']=='MODEL_SCREEN_PASS' for c in cases))))
    print('[UNIFIED EXECUTOR MODEL] saved:',target,flush=True)
    return target


def verify_unified_wheelbase_connections(profile,*,parameter_bounds,include_compiled=True):
    """Primitive-domain SOFTWARE parity, not planning/PhysX validation."""
    import struct
    import numpy as np
    from dataclasses import fields
    import patient_initial_stability.steering_lattice as lattice
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    bindings=unified_wheelbase_connections(profile,include_compiled=include_compiled)
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**parameter_bounds))
    command_queries=native_queries=timing_queries=0
    def pack(result):
        state,poses,elapsed=result
        values=tuple(struct.pack('!d',getattr(state,f.name)) if type(getattr(state,f.name)) is float
            else getattr(state,f.name) for f in fields(lattice.State))
        return (values,b''.join(struct.pack('!3d',*pose) for pose in poses),struct.pack('!d',elapsed))
    for mode in lattice.ALL_MODES:
        for curvature in lattice.CURVATURES:
            expected=bindings.python.steering_targets(mode,curvature)
            if bindings.steering_geometry(mode,curvature)!=expected or not np.allclose(
                    bindings.steering_angles(curvature,mode),expected,rtol=0.,atol=1e-15):
                raise ValueError('Planner/command/vector geometry disagree')
            for direction in (-1,1):
                for speed in (.003,.010,.030,.050):
                    steer,wheels=bindings.direct_joint_targets(direction*speed,mode,curvature,expected,.1,expected)
                    if (steer!=expected or any(abs(w-direction*speed/(.074*math.cos(a)))>1e-14
                            for w,a in zip(wheels,expected))):raise ValueError('Shared command mapping changed beyond geometry')
                    command_queries+=1
                    if bindings.compiled is not None:
                        state=lattice.State(.123,-.321,.4,'coordinated_counterphase',.07,-.03,True,1,.02,.001,-.002)
                        kw=dict(speed_m_s=.05,rigid_body_model=model,target_speed_m_s=speed)
                        left=bindings.compiled(state,mode,curvature,direction,**kw)
                        right=bindings.python.propagate(state,mode,curvature,direction,**kw)
                        if pack(left)!=pack(right):raise ValueError('Compiled/Python candidate edge differs')
                        native_queries+=1
        for direction in (-1,1):
            # Source-independent synthetic smooth input; NOT a hospital route.
            x=np.linspace(0.,.2,101);k=.55*np.sin(2*math.pi*x/x[-1]);k[0]=k[-1]=0.
            pose=np.c_[direction*x,np.zeros((len(x),2))]
            leg=bindings.SpatialLeg(dict(mode=mode,direction=direction,poses_local_xyyaw=pose.tolist(),
                center_arc_m=x.tolist(),curvature_1_m=k.tolist(),next_cusp_local_xyyaw=pose[-1].tolist()))
            timing=bindings.LegTiming(leg)
            if not all(timing.checks.values()):raise ValueError('Shared nominal timing constraints failed')
            timing_queries+=1
    fallback_verified=False
    if bindings.compiled is not None:
        # Non-default response map is outside native domain. Fallback must
        # remain 1.330m, rather than quietly returning to the legacy model.
        state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True)
        kw=dict(speed_m_s=.05,rigid_body_model=model,target_speed_m_s=.03,
            response_scale_by_mode=profile['mode_curvature_response_scale'])
        if pack(bindings.compiled(state,'front_only',.5,1,**kw))!=pack(
                bindings.python.propagate(state,'front_only',.5,1,**kw)):
            raise ValueError('Candidate native fallback used different geometry')
        fallback_verified=True
    import hospital_sensor_navigation as sensor
    geometry=dict(front_axis_xy_m=[.665,0.],rear_axis_xy_m=[-.665,0.],
        front_rolling_offset_xy_m=[0.,0.],rear_rolling_offset_xy_m=[0.,0.],
        front_radius_m=.074,rear_radius_m=.074,front_drive_sign=1, rear_drive_sign=1,
        front_steering_zero_rad=0.,rear_steering_zero_rad=0.,front_steering_sign=1,rear_steering_sign=1)
    encoder_queries=0;encoder_error=0.
    for u in (-.05,-.03,.03,.05):
        for v in (-.003,0.,.003):
            for r in (-.02,0.,.02):
                angles=tuple(math.atan((v+x*r)/u) for x in (.665,-.665))
                rates=tuple(u/(.074*math.cos(angle)) for angle in angles)
                observed=sensor.centerline_encoder_twist(*rates,*angles,geometry=geometry)['body_twist_u_v_r']
                encoder_error=max(encoder_error,*(abs(a-b) for a,b in zip(observed,(u,v,r))))
                encoder_queries+=1
    if encoder_error>1e-12:raise ValueError('Explicit-source1.330m encoder inverse disagrees')
    inspect_source=str(Path(sensor.__file__).resolve())
    bindings.bindings.append(dict(function=sensor.centerline_encoder_twist.__qualname__,
        source=inspect_source,source_sha256=file_sha(inspect_source),
        explicit_encoder_geometry=geometry,synthetic_inputs_NOT_live_sensor_data=True))
    return dict(version='UNIFIED1330_OFFLINE_COMMAND_TIMING_NATIVE_CONNECTION_CHECK_v1',
        status='SOFTWARE_CONNECTION_VERIFIED_NOT_PLANNING_OR_PHYSX_PASS',command_queries=command_queries,
        native_primitive_queries=native_queries,synthetic_nominal_timing_queries=timing_queries,
        native_fallback_retains_new_geometry=fallback_verified,
        explicit_source_encoder_queries=encoder_queries,maximum_synthetic_encoder_twist_error=encoder_error,
        exact_native_state_pose_time_comparison=True if include_compiled else None,
        bindings=bindings.bindings,new_hospital_plans_generated=0,new_PhysX_trials=0,
        unchanged_rigid_body_parameter_bounds=dict(parameter_bounds),
        current_live_executor_installed=False,actual_motion_authorized=False)


def comparison_outcome(record, *, planning_only=False):
    """Separate raw process receipts from a paper's binary task outcome.

    Unknown/engineering HOLDs are NOT silently converted to method failures.
    A bounded no-plan result is NOT a proof of geometric infeasibility.
    """
    raw=record.get('verdict');error=str(record.get('error') or '')
    if planning_only:
        if record.get('plan_success') is not False:raise ValueError('Explicit failed planning record required')
        return dict(raw_status=record['status'],stage='PLANNING',outcome='NO_PLAN_WITHIN_DECLARED_SEARCH',
            mission_success=0,execution_success=None,execution_time_s=None,
            physical_infeasibility_proven=False)
    started=record.get('new_route_trials')==1 or record.get('path_execution_started') is True
    if raw=='PASS' and started:
        return dict(raw_status=raw,stage='EXECUTION',outcome='TASK_COMPLETED',mission_success=1,execution_success=1)
    known=('path tracking deviation','full-envelope','simulation cap','whole-clock cap',
        'whole-mission MODEL time cap','Declared900s wall cap',
        'v2 steering alignment timeout:','v2 terminal pulse stalled:',
        'Original maximum4command direction changes exceeded','Original reverse centre-polyline resource',
        'Alignment requires measured stop')
    if started and raw in ('HOLD','FAIL') and any(text in error for text in known):
        return dict(raw_status=raw,stage='EXECUTION',outcome='EXECUTION_ABORTED_AT_DECLARED_LIMIT',
            mission_success=0,execution_success=0,reason=error)
    if started and raw=='FAIL':
        return dict(raw_status=raw,stage='EXECUTION',outcome='REPORTED_FAILURE_CAUSE_REVIEW_REQUIRED',
            mission_success=None,execution_success=None,reason=error)
    return dict(raw_status=raw,stage='ENGINEERING_OR_UNRESOLVED',outcome='NOT_QUALIFIED_AS_BINARY_METHOD_OUTCOME',
        mission_success=None,execution_success=None,reason=error)


def common_comparison_references(root=RESULT):
    """Explicit reviewed development references, not a latest/best-run search."""
    root=Path(root);rows=[]
    suffixes=('b0_fixed','b1_kinematic','b2_transition','proposed')
    for task,folder in (('H1','path_execution_v3'),('H2','hospital_h2_path_execution_v3')):
        for method,suffix in zip(METHODS,suffixes):
            path=_ijcas_path(root, folder, 'cohort_01', 'hospital_' + task.lower() + '_stopalign_v3_' + suffix + '_path_r01.json')
            rows.append((task,method,path,False))
    h3={'P_dynamics_full_envelope':7,'B0_fixed_counterphase_ramp':8}
    for method in METHODS:
        if method in h3:
            path=_ijcas_path(root, 'hospital_h3_experiments', f'cohort_{h3[method]:02d}_held_static_r01', 'route', method, 'route_readback.json')
            rows.append(('H3',method,path,False))
        else:rows.append(('H3',method,_ijcas_path(root, 'planning/hospital_h3_native_planning_r03', method + '.json'),True))
    for task,cohorts in (('H4',(7,8,6,9)),('H5',(2,3,4,1))):
        for method,number in zip(METHODS,cohorts):
            cohort=(f'cohort_{number:02d}_full_task_r01' if task=='H4' else
                f'cohort_{number:02d}_'+('P' if method.startswith('P_') else method[:2])+'_development_r01')
            name='hospital_'+task.lower()+'_'+cohort+'_'+method+'_r01.json'
            rows.append((task,method,_ijcas_path(root, 'hospital_' + task.lower() + '_experiments', cohort, 'route', name),False))
    for method,number in zip(METHODS,(9,10,11,17)):
        rows.append(('H6',method,_ijcas_path(root, 'hospital_h6_experiments', f'cohort_{number:02d}_held_static_r01', 'route', method, 'route_readback.json'),False))
    if len(rows)!=24 or len({(task,method) for task,method,_,_ in rows})!=24:
        raise ValueError('Exactly six tasks times four explicitly named references required')
    return rows


def build_common_protocol_review(root=RESULT):
    """Offline next-step audit; no adopted protocol or automatic live trial."""
    root=Path(root).resolve();pins={};rows=[]
    def read(path):
        path=Path(path).resolve();payload=path.read_bytes();pins[str(path)]=hashlib.sha256(payload).hexdigest()
        return json.loads(payload)
    def policy_only(value):
        if isinstance(value,dict):return {k:policy_only(v) for k,v in value.items()
            if k not in ('simulation_budget_s','simulation_limit_s')}
        if isinstance(value,list):return [policy_only(v) for v in value]
        return value
    reference=_ijcas_path(root, 'hospital_h3_experiments/cohort_07_held_static_r01/route/P_dynamics_full_envelope/route_readback.json')
    current=read(reference)['contract']
    geometry=read(_ijcas_path(WORKSPACE, 'studies/IJCAS/reports/technical_evidence/figures/figure_02_platform_geometry/platform_geometry_review.json'))
    for role in ('legacy','current'):
        source=geometry['source_geometry_records'][role]['source_asset']
        if file_sha(source)!=geometry['input_sha256'][source]:raise ValueError('Reviewed geometry source changed')
        pins[source]=geometry['input_sha256'][source]
    dated=read(_ijcas_path(WORKSPACE, 'studies/IJCAS/results/rigid_body_edge_gate/ijcas_stage_rigid_body_audit.json'))
    dated_axes={}
    for body in dated['rigid_bodies']:
        for axle in ('front','rear'):
            if body['prim'].endswith('/'+axle+'_active_steering_link'):
                if body['center_of_mass_local_m'][0]!=0.:raise ValueError('Dated centre is not an axis-x witness')
                dated_axes[axle]=body['center_of_mass_base_m'][0]
    if set(dated_axes)!={'front','rear'}:raise ValueError('Both dated steering-link axes required')
    for task,method,path,planning_only in common_comparison_references(root):
        raw=read(path)
        if raw.get('method')!=method:raise ValueError('Explicit reference method identity mismatch: '+str(path))
        row=dict(task=task,method=method,result=str(path),outcome=comparison_outcome(raw,planning_only=planning_only),
            planning_only=planning_only,final_cell_admission=False)
        if planning_only:
            row.update(planning_time_s=raw['planning_time_s'],prior_search_profile_NOT_recomputed_for30mm_s=True,
                need_new_physical_trial_NOT_inferred=True)
        else:
            packet=read(raw['protocol']) if raw.get('protocol') else {}
            plan_path=Path(raw.get('plan_path') or packet['original_plan']);plan=read(plan_path)
            expected=raw.get('plan_sha256') or packet.get('input_sha256',{}).get(str(plan_path.resolve()))
            if not expected or expected!=file_sha(plan_path):raise ValueError('Own immutable plan digest required: '+str(path))
            if plan['method']!=method:raise ValueError('Plan/method identity mismatch')
            speeds=sorted({float(action['target_speed_m_s']) for action in plan['selected_actions']})
            if not speeds or any(not math.isfinite(v) or v<=0 for v in speeds):raise ValueError('Explicit finite selected target speeds required')
            poses=[plan['start_world_xyyaw']]+[state['world_xyyaw'] for state in plan['selected_states']]
            length=sum(math.dist(a[:2],b[:2]) for a,b in zip(poses,poses[1:]))
            changes=sum(a['direction']!=b['direction'] for a,b in zip(plan['selected_actions'],plan['selected_actions'][1:]))
            trace=Path(raw.get('trace') or path.with_suffix('.csv'));digest=file_sha(trace);pins[str(trace.resolve())]=digest
            if raw.get('trace_sha256') and raw['trace_sha256']!=digest:raise ValueError('Saved trace hash changed')
            same=bool(raw.get('contract')) and policy_only(raw['contract'])==policy_only(current)
            max_speed=max(speeds);speed_match=abs(max_speed-.030)<=1e-12
            row.update(plan=str(plan_path),selected_target_speeds_m_s=speeds,
                current_common_controller_contract_matches_ignoring_task_clock=same,
                candidate30mm_s_selected_target_profile_bound_matches=speed_match,
                candidate30mm_s_AND_current_controller_parameter_match=same and speed_match,
                parameter_match_is_NOT_final_protocol_admission_or_independent_repeat=True,
                controller_contract=raw.get('contract') or raw.get('common_controller'),
                existing_command_limits=raw.get('command_limits'),existing_acceptance=raw.get('acceptance'),
                simulation_budget_s=packet.get('simulation_budget_s',raw.get('guards',{}).get('maximum_simulation_duration_s')),
                maximum_wall_time_s=packet.get('maximum_wall_time_s',raw.get('guards',{}).get('maximum_wall_time_s')),
                trace=str(trace),trace_sha256=digest)
            row['candidate_nominal_clock_screen']=dict(reference_polyline_length_m=length,
                direction_changes=changes,constant30mm_s_reference_time_s=length/.030,
                reference_time_plus_prehold_and_final_hold_s=length/.030+7.,
                braking_steering_cusp_response_and_terminal_correction_NOT_included=True,
                NOT_a_physical_travel_time_bound_or_full_executor_model_PASS=True,
                new_native_NE_speed_profile_repropagation_and_collision_audit_still_required=True)
        rows.append(row)
    physical=[r for r in rows if not r['planning_only']]
    return dict(version='H1_H6_COMMON_EXECUTOR_AND_SPEED_COMPATIBILITY_REVIEW_v1',
        status='OFFLINE_PROTOCOL_CANDIDATE_NOT_ADOPTED_NOT_MOTION_AUTHORITY',references=rows,
        reference_common_controller_result=str(reference),
        wheelbase_origin_review=dict(
            legacy_CAD_axis_span_m=geometry['legacy_CAD_source']['active_pivot_span_m'],
            current_USD_axis_span_m=geometry['current_hospital_USD_source']['active_pivot_span_m'],
            retained_nominal_command_wheelbase_m=geometry['preserved_nominal_steering']['wheelbase_m'],
            frozen_NE_support_span_m=geometry['frozen_Newton_Euler_support_parameters']['front_support_x_m']-
                geometry['frozen_Newton_Euler_support_parameters']['rear_support_x_m'],
            prior_dated_USD_audit_utc=dated['generated_utc'],
            prior_dated_axis_span_witness_m=dated_axes['front']-dated_axes['rear'],
            exact_first_model_transition_timestamp_NOT_established=True,
            geometry_response_coefficients_NOT_reidentified_or_assumed_to_cancel_difference=True,
            current_asset_nominal_command_parameters_and_all_results_unchanged=True),
        primary_recommendation_NOT_user_adoption=dict(
            policy='COMMON50mm_s_CAP_AUTOMATIC_METHOD_SPEED_SELECTION',
            shared_speed_cap_m_s=.050,shared_candidate_target_speeds_m_s=[.003,.010,.030,.050],
            manual_P_only_speed_override_forbidden=True,
            common_base_metrics_limits_and_compute_budget_required=True,
            method_specific_transition_or_dynamics_cost_terms_must_be_declared=True,
            cached_guided_P_generation_time_NOT_primary_end_to_end_planning_time=True,
            fresh_automatic_selection_and_total_setup_seed_search_time_must_be_verified=True,
            additional_fixed30mm_s_experiment_suite_NOT_required=True,
            fixed30_candidate_mismatch_count_NOT_a_rerun_count_for_this50_option=True,
            no_new_motion_or_final_protocol_admission=True),
        candidate=dict(shared_cruise_target_m_s=.030,original_maximum_command_speed_m_s=.050,
            same_method_blind_braking_cusp_terminal_and_steering_policy_required=True,
            identical_paths_or_identical_speed_at_every_timestamp_NOT_required=True,
            task_specific_clock_limits_must_be_declared_and_identical_for_all_four_methods=True,
            clock_adequacy_under_new_speed_NOT_yet_verified=True,
            change_physical_geometry_or_nominal_steering_parameters=False,
            adopting_candidate_requires_frozen_planning_and_model_checks_before_new_motion=True),
        counts=dict(comparison_cells=24,explicit_physical_route_references=len(physical),planning_only_references=2,
            physical_references_matching_current_controller=sum(r['current_common_controller_contract_matches_ignoring_task_clock'] for r in physical),
            physical_references_matching_candidate30mm_s_AND_controller=sum(r['candidate30mm_s_AND_current_controller_parameter_match'] for r in physical),
            physical_references_not_matching_candidate30mm_s_AND_controller=sum(not r['candidate30mm_s_AND_current_controller_parameter_match'] for r in physical),
            existing_final_cells_admitted_by_this_audit=0),
        remaining_drive_count_NOT_determined_before_protocol_choice_and_replanning=True,
        parameter_mismatch_is_NOT_an_automatic_order_to_repeat_every_trial=True,
        all_original_PASS_FAIL_HOLDs_preserved=True,new_PhysX_trials=0,actual_motion_authorized=False,
        input_sha256=pins)


def write_common_protocol_review(root=RESULT):
    from hospital_runtime_evidence_io import write_exclusive
    report=build_common_protocol_review(root)
    report['analysis_source_sha256']={str(Path(__file__).resolve()):file_sha(__file__)}
    identity=hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    target=Path(root)/'paper_evidence_inventory/common_protocol_review'/identity/'compatibility_review.json'
    if target.exists():
        if json.loads(target.read_bytes())!=report:raise ValueError('Existing common review differs; no overwrite')
    else:write_exclusive(target,report)
    print('Common protocol compatibility:',report['counts']);print('Saved:',target)
    print('Candidate only. No Play/Stop/reset, adopted protocol or new trial.')
    return target


def paper_outcome_presentation(record,*,planning_only=False):
    """Cause-specific display; a raw HOLD is never itself a numeric score."""
    outcome=comparison_outcome(record,planning_only=planning_only)
    label={
        'TASK_COMPLETED':'SUCCESS',
        'NO_PLAN_WITHIN_DECLARED_SEARCH':'PLAN_FAIL_WITHIN_BUDGET',
        'EXECUTION_ABORTED_AT_DECLARED_LIMIT':'EXEC_FAIL',
        'REPORTED_FAILURE_CAUSE_REVIEW_REQUIRED':'PENDING_CAUSE_REVIEW',
        'NOT_QUALIFIED_AS_BINARY_METHOD_OUTCOME':'ERR_OR_BLOCKED_PENDING_REVIEW'}[outcome['outcome']]
    success=outcome['mission_success']
    return dict(**outcome,table_label=label,table_missing_value='—',
        graph_style=('completed_curve' if success==1 else 'x_at_abort_or_timeout_NO_extension'
            if success==0 else 'grey_hatched_NOT_zero_height'),
        task_completion_time_available=success==1,
        recorded_abort_or_planning_time_is_NOT_successful_completion_time=success!=1,
        failed_route_partial_metrics_must_be_labelled_at_abort=outcome['stage']=='EXECUTION' and success==0,
        algorithm_success_denominator_eligible=success is not None,
        engineering_exclusions_and_all_original_attempts_must_be_disclosed=True)


def unified_h3_planner_binding(profile,manifest,occupied,origin,resolution,*,reserve_m=.050,terminal_direction_change_reserve=0,
        initial_state=None):
    """Bind the existing whole-task search privately to ONE new geometry."""
    import ast,inspect,textwrap
    from types import FunctionType,SimpleNamespace
    import plan_hospital_h3_native as h3
    import patient_initial_stability.steering_lattice as lattice
    import patient_initial_stability.hierarchical_dynamics_planner as hierarchy
    import hospital_mission_planning as resource
    if type(terminal_direction_change_reserve) is not int or terminal_direction_change_reserve not in (0,2):
        raise ValueError('Explicit zero or two changes reserved from original runtime maximum4 required')
    private_resource=SimpleNamespace(**dict(vars(resource),MotionResourceLimits=lambda:resource.MotionResourceLimits(
        maximum_direction_changes=4-terminal_direction_change_reserve)))
    connections=unified_wheelbase_connections(profile,include_compiled=True)
    def clone(function,**values):
        result=FunctionType(function.__code__,dict(function.__globals__,**values),function.__name__,
            function.__defaults__,function.__closure__)
        result.__kwdefaults__=function.__kwdefaults__;return result
    private_lattice=SimpleNamespace(**dict(vars(lattice),
        plan=clone(lattice.plan,steering_targets=connections.python.steering_targets),
        propagate=connections.python.propagate,steering_targets=connections.python.steering_targets))
    private_hierarchy=SimpleNamespace(**dict(vars(hierarchy),
        refine_speed_consistent_lattice_route=clone(hierarchy.refine_speed_consistent_lattice_route,
            steering_targets=connections.python.steering_targets,propagate=connections.python.propagate)))
    source=h3.continuous_task_planner;tree=ast.parse(textwrap.dedent(inspect.getsource(source)))
    counts=dict(lattice=0,hierarchy=0,native=0)
    class Private(ast.NodeTransformer):
        def visit_Import(self,node):
            if (terminal_direction_change_reserve and len(node.names)==1
                    and node.names[0].name=='hospital_mission_planning' and node.names[0].asname=='binding'):
                counts['resource']=counts.get('resource',0)+1
                return ast.copy_location(ast.Assign([ast.Name('binding',ast.Store())],ast.Name('_unified_resource',ast.Load())),node)
            return node
        def visit_ImportFrom(self,node):
            if node.module=='patient_initial_stability' and len(node.names)==1:
                alias=node.names[0]
                if (alias.name,alias.asname) in (('steering_lattice','lattice'),('hierarchical_dynamics_planner','hierarchy')):
                    counts[alias.asname]+=1
                    return ast.copy_location(ast.Assign([ast.Name(alias.asname,ast.Store())],
                        ast.Name('_unified_'+alias.asname,ast.Load())),node)
            return node
        def visit_Call(self,node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                    and node.func.value.id=='binding' and node.func.attr=='compiled_original_NE_propagation'
                    and not node.args and not node.keywords):
                counts['native']+=1
                return ast.copy_location(ast.Name('_unified_native',ast.Load()),node)
            return self.generic_visit(node)
    tree=Private().visit(tree)
    expected=dict(lattice=1,hierarchy=1,native=1)
    if terminal_direction_change_reserve:expected['resource']=1
    if counts!=expected:raise ValueError('Whole-task geometry binding anchors changed')
    namespace=dict(source.__globals__,_unified_lattice=private_lattice,
        _unified_hierarchy=private_hierarchy,_unified_native=connections.compiled,_unified_resource=private_resource)
    exec(compile(ast.fix_missing_locations(tree),'<unified1330_whole_H3_binding>','exec'),namespace)
    method,parameters=namespace[source.__name__](manifest,occupied,origin,resolution,
        additional_execution_reserve_m=reserve_m,initial_state=initial_state)
    parameters.pop('original_H3_start_goal_map_and_NE_model_unchanged')
    parameters.update(canonical_geometry=dict(profile['geometry']),
        original_H3_start_goal_map_and_NE_parameter_bounds_unchanged=True,
        response_coefficients_algebraically_reexpressed_not_refitted=True,
        all_search_refinement_command_geometry_uses_unified_profile=True,
        private_geometry_binding_anchors=counts)
    if terminal_direction_change_reserve:
        parameters.update(maximum_direction_changes=4-terminal_direction_change_reserve,
            reserved_direction_changes_for_terminal=terminal_direction_change_reserve,
            original_TOTAL_runtime_direction_change_cap=4,physical_runtime_limit_NOT_increased=True,
            same_planning_reserve_for_all_methods=True)
    import audit_hospital_h1_plan as h1_audit
    import audit_hospital_h3_plan as h3_audit
    original=clone(h1_audit.audit,propagate=connections.python.propagate)
    independent=resource.additional_planning_reserve_audit(original,reserve_m=reserve_m)
    auditor=clone(h3_audit.audit,audit_actions=independent)
    return method,auditor,connections,parameters


def common_cruise_profile_caps(actions,levels=(.003,.010,.030,.050),*,forward_floor=.010,reverse_floor=.020):
    """Common method-blind cruise choices; low terminal levels are separate."""
    if (not actions or not levels or any(type(a.get('direction')) is not int or a['direction'] not in (-1,1) for a in actions)
            or any(not math.isfinite(v) or not 0.<v<=.050 for v in (*levels,forward_floor,reverse_floor))):
        raise ValueError('Explicit signed actions and common bounded speed policy required')
    floor=max(forward_floor if a['direction']>0 else reverse_floor for a in actions)
    result=tuple(sorted({float(v) for v in levels if v>=floor},reverse=True))
    if not result:raise ValueError('No common cruise candidate respects the declared directional floor')
    return result


def nominal_execution_clock(edge_times,simulation_budget_s):
    """Original NE edge times plus common2s prehold/5s hold, not a guarantee.

    Reject a speed candidate already over the task clock in its own nominal
    plan. Steering/ACK/braking/caster execution still needs the full model
    and actual guards; this never increases an accepted trial's budget.
    """
    from numbers import Real
    times=tuple(edge_times)
    if (not times or any(not isinstance(v,Real) or isinstance(v,bool) or not math.isfinite(v) or v<=0. for v in times)
            or simulation_budget_s not in (180.,240.,300.,600.)):
        raise ValueError('Finite positive native edge times and original declared task clock required')
    estimate=math.fsum(times)+7.
    return dict(nominal_edges_plus_fixed_prehold_and_hold_s=estimate,
        declared_simulation_budget_s=simulation_budget_s,admissible=estimate<=simulation_budget_s,
        NOT_a_physical_completion_time_bound_or_full_executor_model_PASS=True)


def unified_h3_guided_candidate(method,seed,manifest,occupied,origin,resolution,model,*,budget_s,execution_budget_s=None,
        initial_state=None,terminal_position_tolerance_m=.025):
    """Own fresh controls -> model-ranked common speeds -> strict NE beam.

    Same existing NE/SAT/resource/final predicates. Extra neutral cusp edges
    are real controls, never state resets. No speed is picked from an actual
    outcome. A first feasible plan is not a global optimum or drive PASS.
    """
    import time
    import hospital_mission_planning as resource
    import plan_hospital_h1 as original
    import patient_initial_stability.steering_lattice as lattice
    if type(budget_s) not in (int,float) or not math.isfinite(budget_s) or not 0.<budget_s<=180.:
        raise ValueError('Remaining positive finite common180s budget required')
    if execution_budget_s is not None and execution_budget_s not in (180.,240.,300.,600.):
        raise ValueError('Original task execution clock required; no extension')
    if (type(terminal_position_tolerance_m) not in (int,float)
            or not math.isfinite(terminal_position_tolerance_m) or not .005<=terminal_position_tolerance_m<=.025):
        raise ValueError('Only an explicitly stricter5-to25mm planning endpoint is allowed')
    begun=time.perf_counter();deadline=begun+budget_s
    row=dict(method=METHODS[-1],planner_dynamics='full',allowed_modes=list(lattice.ALL_MODES),
        plan_success=False,status='NO_FEASIBLE_GUIDED_NE_CANDIDATE_WITHIN_DECLARED_SEARCH',
        initial_front_rad=0.,initial_rear_rad=0.,initial_u_v_r=[0.,0.,0.],initial_steering_propagated=True,
        selected_actions=[],selected_states=[],refinement_attempts=[],
        planner_form='fresh_owned_spatial_seed_model_ranked_common_cruise_profiles_NE_guided_refinement',
        cached_other_trial_or_measured_state_used=False,actual_motion_authorized=False,use_for_final_statistics=False)
    if not seed or not seed['plan_success']:
        row.update(status='OWN_SPATIAL_SEED_NOT_FOUND_WITHIN_BUDGET',planning_time_s=time.perf_counter()-begun)
        return row
    actions,bridges=resource.neutral_direction_bridges(seed['selected_actions'])
    caps=common_cruise_profile_caps(actions);row['common_cruise_caps_m_s']=list(caps)
    row['explicit_neutral_bridges']=bridges
    base=method.__globals__['HospitalEnvelopeChecker']
    class Watch(base):
        def __init__(self,*args,**kwargs):
            self.notice=time.perf_counter()+15.;self.phase='profile_ranking'
            super().__init__(*args,**kwargs)
        def check_deadline(self):
            super().check_deadline();now=time.perf_counter()
            if now>=self.notice:
                print('[UNIFIED H3 P]',self.phase,'elapsed',round(now-begun,1),'s',flush=True);self.notice=now+15.
    checker=Watch(occupied,origin,resolution,manifest['start_xyyaw'],deadline=deadline)
    refine=method.__globals__['refine_speed_consistent_lattice_route'];State=refine.__globals__['State']
    propagate=refine.__globals__['propagate'];profiles=[];events=[]
    initial=(State(0.,0.,0.,'coordinated_counterphase',0.,0.,True) if initial_state is None else initial_state)
    if type(initial) is not State or initial.started is not True or initial.rigid_body_feasible is not True:
        raise ValueError('Exact complete feasible continued planner state required')
    if initial_state is not None:
        from dataclasses import asdict
        row.update(initial_front_rad=initial.front,initial_rear_rad=initial.rear,
            initial_u_v_r=[initial.longitudinal_velocity,initial.lateral_velocity,initial.yaw_rate],
            measured_complete_initial_state=asdict(initial),cached_other_trial_or_measured_state_used=True,
            measured_state_is_own_current_trial_NOT_external_seed=True)
    try:
        for cap in caps:
            state=initial
            violations=dict(force=0,resource=0,geometry=0,cusp=0)
            for index,action in enumerate(actions):
                checker.check_deadline();previous=state
                stopping=math.sqrt(.005**2+2*.020*resource.direction_remaining_arc(actions,index,.05))
                speed=max(v for v in (.003,.010,.030,.050) if v<=min(cap,stopping)+1e-12)
                state,poses,_=propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
                    speed_m_s=.05,dynamics=lattice.FULL_DYNAMICS,rigid_body_model=model,target_speed_m_s=speed)
                violations['force']+=int(not state.rigid_body_feasible)
                valid_resource=getattr(poses,'resource_feasible',True)
                violations['resource']+=int(not valid_resource)
                if valid_resource:violations['geometry']+=int(checker(poses))
                violations['cusp']+=int(not resource.stop_connected_edge(previous,state,actions,index))
            goal=manifest['local_goal_xyyaw'];pose_error=math.hypot(state.x-goal[0],state.y-goal[1])
            heading_error=abs(lattice.wrap(state.yaw-goal[2]))
            profiles.append(dict(cap_m_s=cap,rough_seed_violations=violations,
                rough_terminal_position_m=pose_error,rough_terminal_heading_rad=heading_error,
                rough_rollout_is_ranking_ONLY_not_accepted_or_infeasibility_proof=True,
                rank=[sum(violations.values()),100.*pose_error+60.*heading_error,-cap]))
        profiles.sort(key=lambda r:r['rank']);row['automatic_model_ranked_profiles']=profiles
        print('[UNIFIED H3 P] automatic model order:',[p['cap_m_s'] for p in profiles],flush=True)
        for profile in profiles:
            checker.check_deadline();cap=profile['cap_m_s'];checker.phase='NE_beam_cap_'+str(cap)
            print('[UNIFIED H3 P] strict guided refinement at',cap,'m/s',flush=True)
            result,diagnostics=resource.refine_with_NE_guided_beam(refine,
                'OWN_FRESH_H3_NEUTRAL_CONNECTED_SOURCE',actions,tuple(manifest['local_goal_xyyaw']),
                tuple(v for v in (.003,.010,.030,.050) if v<=cap),model,checker,.005,.020,
                step_m=.05,terminal_extension_edges=0,initial_front_rad=initial.front,initial_rear_rad=initial.rear,
                initial_mode=initial.mode,
                propagate_initial_steering=True,include_configuration_cost=True,maximum_iterations=24,
                source_dynamics=lattice.KINEMATIC_DYNAMICS,body_radius_m=.5*math.hypot(1.630,.965),
                beam_width=64,lookahead_edges=12,event_sink=events.append,mode_refinement=True,
                stop_at_direction_changes=True,initial_state=initial_state,
                goal_position_tolerance_m=terminal_position_tolerance_m)
            row['refinement_attempts'].append(dict(cap_m_s=cap,success=result is not None,diagnostics=diagnostics))
            if result is None:continue
            if execution_budget_s is not None:
                clock=nominal_execution_clock(result.edge_times_s,execution_budget_s)
                row['refinement_attempts'][-1]['nominal_execution_clock']=clock
                if not clock['admissible']:
                    row['refinement_attempts'][-1]['selection']='REJECTED_BY_ORIGINAL_TASK_CLOCK_TRY_NEXT_PROFILE_WITHIN_SAME_PLANNING_BUDGET'
                    continue
            final=result.states[-1];goal=manifest['local_goal_xyyaw']
            row.update(plan_success=True,status='STRICT_GUIDED_NE_PLAN_FOUND_EXECUTION_PENDING',
                selected_actions=[{**a,'target_speed_m_s':s} for a,s in zip(result.actions,result.speeds_m_s)],
                selected_states=[original.state_record(s,manifest['start_xyyaw'],True) for s in result.states],
                action_count=len(result.actions),path_length_m=.05*len(result.actions),search_cost_s=result.cost_s,
                predicted_edge_times_s=list(result.edge_times_s),selected_automatic_cruise_cap_m_s=cap,
                mode_switch_count=sum(a['mode']!=b['mode'] for a,b in zip(result.actions,result.actions[1:])),
                predicted_terminal_position_error_m=math.hypot(final.x-goal[0],final.y-goal[1]),
                predicted_terminal_heading_error_deg=math.degrees(lattice.wrap(final.yaw-goal[2])),
                predicted_terminal_speed_m_s=math.hypot(final.longitudinal_velocity,final.lateral_velocity),
                planning_endpoint_tolerance_m=terminal_position_tolerance_m,
                selection='FIRST_STRICT_FEASIBLE_IN_MODEL_RANKED_COMMON_PROFILES_NOT_GLOBAL_OPTIMUM')
            break
    except TimeoutError as error:
        row.update(status='PLANNING_TIMEOUT_NOT_PROVEN_INFEASIBLE',error=str(error))
    finally:
        row.update(planning_time_s=time.perf_counter()-begun,NE_events=events,collision_checker_statistics=checker.statistics)
    return row


UNIFIED_P_SEED_POLICY='OWN_MULTIMODE30_FIXED45_NE_EXECUTION_CLOCK_CONNECTED_v3'


def unified_P_owned_seed(method,manifest,occupied,origin,resolution,model,*,budget_s,node_limit,policy):
    """Fresh seed portfolio, not cached baseline reuse or a physical retry.

    One shared45s seed clock: multimode gets at most30s, then the original
    fixed-mode search gets only the unspent remainder. First success wins.
    The same rule applies to every task; NE refinement still pays the rest
    of the original total180s clock. Legacy exports keep their old policy.
    """
    import time
    if (policy not in ('FIXED_COUNTERPHASE_v1',UNIFIED_P_SEED_POLICY)
            or type(budget_s) not in (int,float) or not math.isfinite(budget_s) or not 0.<budget_s<=180.
            or type(node_limit) is not int or not 0<node_limit<=60000):
        raise ValueError('Explicit frozen seed policy and original finite planning budgets required')
    begun=time.perf_counter();total=min(45.,budget_s);attempts=[];seed=None
    families=(METHODS[0],) if policy=='FIXED_COUNTERPHASE_v1' else (METHODS[1],METHODS[0])
    for index,family in enumerate(families):
        remaining=total-(time.perf_counter()-begun)
        if remaining<=0.:break
        limit=min(30.,remaining) if len(families)==2 and index==0 else remaining
        seed=method(family,manifest,occupied,origin,resolution,model,limit,node_limit,{})
        attempts.append(dict(family=family,budget_s=limit,status=seed['status'],
            planning_time_s=seed['planning_time_s'],plan_success=seed['plan_success']))
        if seed['plan_success']:break
    if seed is None:raise RuntimeError('No seed search dispatched inside original positive budget')
    seed=dict(seed,owned_seed_policy=policy,owned_seed_attempts=attempts,
        owned_seed_total_time_s=time.perf_counter()-begun,
        no_cached_baseline_plan_or_measured_state_used=True)
    return seed


def write_unified_h3_planning_review(evaluation_path,*,methods=None,guided_recovery=False,recorded_geometry_path=None,terminal_direction_change_reserve=0,task_geometry_path=None,owned_seed_policy='FIXED_COUNTERPHASE_v1'):
    """Fresh bounded planning; no cached P seed, live command, or retry.

    This first integration is DEVELOPMENT. Command-floor retiming and full
    execution-model validation still precede admission to a matched drive.
    """
    import time
    from dataclasses import asdict
    import numpy as np
    from hospital_h3_turn_planning import load_setup,model_bounds,MODEL_PLAN
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    import patient_initial_stability.steering_lattice as lattice
    from hospital_runtime_evidence_io import write_exclusive
    evaluation_path=Path(evaluation_path).resolve();evaluation=json.loads(evaluation_path.read_bytes())
    if evaluation['version']!='UNIFIED1330_ONE_BLOCK_EVALUATION_CASE_ACCOUNTING_AND_PRESENTATION_v1':
        raise ValueError('Explicit unified evaluation review required')
    profile=evaluation['profile'];unified_wheelbase_kernel(profile)
    methods=tuple(methods or (METHODS[0],METHODS[-1],METHODS[1],METHODS[2]))
    if type(guided_recovery) is not bool or not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Distinct explicit methods required; no automatic retry')
    if (owned_seed_policy not in ('FIXED_COUNTERPHASE_v1',UNIFIED_P_SEED_POLICY)
            or owned_seed_policy==UNIFIED_P_SEED_POLICY and not guided_recovery):
        raise ValueError('Reviewed multimode seed portfolio requires original guided NE refinement')
    old_path=RESULT/'planning/hospital_h3_native_planning_r03/summary.json'
    old=json.loads(old_path.read_bytes());manifest_path=Path(old['geometry_manifest']);preview=Path(old['preview_receipt'])
    manifest,occupied,origin,resolution,manifest_sha=load_setup(manifest_path,preview)
    legacy_manifest_path=manifest_path;recorded_map=None
    if task_geometry_path is not None:
        if recorded_geometry_path is not None:raise ValueError('One explicit task geometry required')
        import hospital_mission_runtime as retained
        from hospital_mission_sequence import TASK_SCENARIOS
        manifest_path=Path(task_geometry_path).resolve();manifest=json.loads(manifest_path.read_bytes())
        current=retained.read(retained.GEOMETRY)
        if (manifest.get('scenario') not in TASK_SCENARIOS.values()
                or manifest.get('original_fixed_task_geometry') is None
                or any(manifest[k]!=current[k] for k in ('occupancy_grid','planning_occupancy_digest',
                    'same_three_authored_support_rectangles','full_body_rectangle_m','full_body_clearance_m'))):
            raise ValueError('Original fixed task and exact current map/support required')
        original=retained.read(manifest['original_fixed_task_geometry'])
        if any(manifest[k]!=original[k] for k in ('start_xyyaw','goal_xyyaw')):
            raise ValueError('Never choose a different start or goal to favour a method')
        recorded_map=Path(manifest['occupancy_grid']);manifest_sha=file_sha(manifest_path)
        with np.load(recorded_map) as grid:
            occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
        if hashlib.sha256(occupied.tobytes()).hexdigest()!=manifest['planning_occupancy_digest']:
            raise ValueError('Exact current map digest required')
    if recorded_geometry_path is not None:
        recorded_geometry_path=Path(recorded_geometry_path).resolve()
        recorded=json.loads(recorded_geometry_path.read_bytes())
        if (recorded.get('scenario')!=manifest['scenario'] or recorded.get('start_xyyaw')!=manifest['start_xyyaw']
                or recorded.get('goal_xyyaw')!=manifest['goal_xyyaw'] or recorded.get('local_goal_xyyaw')!=manifest['local_goal_xyyaw']
                or Path(recorded.get('original_H3_task_geometry','')).resolve()!=legacy_manifest_path.resolve()
                or recorded.get('full_body_rectangle_m')!=[1.630,.965] or recorded.get('full_body_clearance_m')!=.010
                or recorded.get('actual_motion_authorized') is not False or recorded.get('use_for_final_statistics') is not False):
            raise ValueError('Exact recorded same H3 task and unchanged body/limits required')
        recorded_map=Path(recorded['occupancy_grid']).resolve()
        with np.load(recorded_map) as grid:
            occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
        raster=recorded['raster']
        if (occupied.dtype!=np.dtype(bool) or list(occupied.shape)!=raster['shape_rows_columns']
                or not np.array_equal(origin,raster['origin_xy_m']) or resolution!=raster['resolution_m']
                or hashlib.sha256(occupied.tobytes()).hexdigest()!=recorded['planning_occupancy_digest']):
            raise ValueError('Exact recorded map frame/digest required')
        from hospital_geometry_tools import pose_collision
        if any(pose_collision(occupied,recorded[k],origin,resolution,.010) for k in ('start_xyyaw','goal_xyyaw')):
            raise ValueError('Original H3 endpoint footprint blocked in recorded map')
        manifest=dict(recorded,environment_condition=manifest['environment_condition'])
        manifest_path=recorded_geometry_path;manifest_sha=file_sha(manifest_path)
    policy=evaluation['recommended_common_policy'];budget=policy['proposed_total_planning_wall_cap_s']
    node_limit=policy['proposed_lattice_expansion_cap'];initialization=time.perf_counter()
    method,auditor,connections,parameters=unified_h3_planner_binding(profile,manifest,occupied,origin,resolution,
        terminal_direction_change_reserve=terminal_direction_change_reserve)
    if recorded_map is not None:
        # The historical preview is NOT a preview of this later full map.
        # Reuse independent Python action/NE/SAT replay directly, retaining
        # explicit recorded-task/frame checks above, not a forged preview.
        auditor=auditor.__globals__['audit_actions']
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**model_bounds()))
    # Common native warmup is measured separately, never paid only by P.
    connections.compiled(lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,True),
        'coordinated_counterphase',.0,1,speed_m_s=.05,rigid_body_model=model,target_speed_m_s=.03)
    initialization=time.perf_counter()-initialization
    parent=RESULT/'planning';number=1;task_key=manifest['scenario'].split('_')[0].lower()
    while (parent/f'hospital_{task_key}_unified_planning_r{number:02d}').exists():number+=1
    output=parent/f'hospital_{task_key}_unified_planning_r{number:02d}';output.mkdir()
    np.savez_compressed(output/'planning_occupancy.npz',occupied=occupied,origin_xy_m=origin,resolution_m=resolution)
    pins={str(p.resolve()):file_sha(p) for p in (evaluation_path,old_path,manifest_path,preview,MODEL_PLAN,
        Path(__file__),_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/plan_hospital_h3_native.py'),
        _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_mission_planning.py'),
        Path(lattice.__file__),_ijcas_path(WORKSPACE, 'src/patient_initial_stability/patient_initial_stability/hierarchical_dynamics_planner.py'))}
    if recorded_map is not None:
        pins.update({str(recorded_map):file_sha(recorded_map),str(legacy_manifest_path):file_sha(legacy_manifest_path)})
    for item in connections.bindings:pins[item['source']]=item['source_sha256']
    for name in ('plan_hospital_h1.py','hospital_native_envelope.py','hospital_exact_sat.c','hospital_planning_tools.py',
            'hospital_geometry_tools.py','audit_hospital_h1_plan.py','audit_hospital_h3_plan.py'):
        p=_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts', name);pins[str(p)]=file_sha(p)
    metadata=dict(schema_version=1,scenario=manifest['scenario'],environment_condition=manifest['environment_condition'],
        trial_kind='UNIFIED1330_FRESH_H3_DEVELOPMENT_PLANNING_NOT_PhysX_OR_FINAL_COMPARISON',
        geometry_profile=profile,geometry_manifest=str(manifest_path),geometry_manifest_sha256=manifest_sha,
        preview_receipt=str(preview),preview_receipt_sha256=file_sha(preview),
        historical_preview_is_NOT_a_new_full_map_or_live_state_confirmation=recorded_map is not None,
        fresh_planning_on_exact_recorded_common_map=recorded_map is not None,
        start_world_xyyaw=manifest['start_xyyaw'],goal_world_xyyaw=manifest['goal_xyyaw'],goal_local_xyyaw=manifest['local_goal_xyyaw'],
        planning_occupancy_digest=manifest['planning_occupancy_digest'],common_max_command_speed_m_s=.050,
        common_max_acceleration_m_s2=.020,planning_terminal_speed_tolerance_m_s=.005,full_envelope_clearance_m=.010,
        planning_additional_execution_reserve_m=.050,rigid_body_parameter_bounds=asdict(model.bounds),
        continuous_task_search_binding=parameters,common_initialization_time_s=initialization,
        common_initialization_excluded_from_method_clock_but_reported=True,
        own_total_planning_budget_s=budget,node_limit=node_limit,
        node_limit_scope='LATTICE_EXPANSIONS_ONLY_NE_BEAM_EDGE_EVALUATIONS_REPORTED_SEPARATELY',
        actual_executor_directional_speed_floor_retiming_STILL_pending=True,
        use_for_final_statistics=False,Isaac_execution_verified=False,Nav2_execution_verified=False,
        actual_motion_authorized=False,input_sha256=pins)
    write_exclusive(output/'definition.json',dict(**metadata,methods=list(methods),
        P_fresh_internal_seed_family=('B0_counterphase_same_original_whole_task_search'
            if owned_seed_policy=='FIXED_COUNTERPHASE_v1' else UNIFIED_P_SEED_POLICY),P_seed_budget_s=45.,
        P_owned_seed_policy=owned_seed_policy,
        guided_NE_recovery=guided_recovery,
        guided_policy=dict(beam_width=64,lookahead_edges=12,maximum_iterations=24,
            profiles_selected_by_same_model_NOT_actual_outcomes=True,
            common_directional_speed_floors_m_s=[.010,.020],no_P_only_manual_cruise_override=True) if guided_recovery else None,
        no_prior_saved_baseline_plan_or_measured_state_borrowed=True))
    from hospital_mission_lifecycle import freeze_workflow_sources
    freeze_workflow_sources(output/'sources.zip',pins)
    print('[UNIFIED H3] output:',output,flush=True);rows=[]
    for name in methods:
        print('[UNIFIED H3] START',name,'total budget',budget,'s',flush=True)
        began=time.perf_counter();seed=None;pool={}
        if name==METHODS[-1]:
            seed=unified_P_owned_seed(method,manifest,occupied,origin,resolution,model,
                budget_s=budget,node_limit=node_limit,policy=owned_seed_policy)
            if seed['plan_success']:pool[seed['method']]=seed['selected_actions']
            write_exclusive(output/'P_own_fresh_seed.json',{**metadata,**seed})
            print('[UNIFIED H3] P own fresh seed:',seed['status'],'elapsed',round(time.perf_counter()-began,3),'s',flush=True)
        remaining=budget-(time.perf_counter()-began)
        if remaining<=0:raise RuntimeError('Total fresh planning budget consumed before own method stage')
        from hospital_mission_sequence import TASK_BUDGETS
        execution_budget=(TASK_BUDGETS[manifest['scenario'].split('_')[0]]
            if owned_seed_policy==UNIFIED_P_SEED_POLICY else None)
        row=(unified_h3_guided_candidate(method,seed,manifest,occupied,origin,resolution,model,budget_s=remaining,
                execution_budget_s=execution_budget)
            if name==METHODS[-1] and guided_recovery else
            method(name,manifest,occupied,origin,resolution,model,remaining,node_limit,pool))
        if seed is not None:
            row['own_refinement_stage_time_s']=row['planning_time_s'];row['own_fresh_seed_time_s']=seed['owned_seed_total_time_s']
            row['own_selected_seed_search_time_s']=seed['planning_time_s']
            row['own_fresh_seed_success']=seed['plan_success'];row['no_cached_other_method_seed_used']=True
            row['owned_seed_policy']=owned_seed_policy;row['owned_seed_attempts']=seed['owned_seed_attempts']
        row.update(planning_time_s=time.perf_counter()-began,wall_clock_budget_s=budget)
        path=output/(name+'.json');write_exclusive(path,{**metadata,**row})
        checked=None
        if row['plan_success']:
            checked=auditor(path);write_exclusive(output/'offline_audits'/(name+'.json'),checked)
        summary={key:row.get(key) for key in ('method','plan_success','status','planning_time_s','action_count','path_length_m')}
        summary['independent_unified_Python_full_SAT_audit']=checked['verdict'] if checked else 'NO_PLAN_NO_AUDIT'
        rows.append(summary)
        print('[UNIFIED H3] RESULT',name,summary,flush=True)
        for source,expected in pins.items():
            if file_sha(source)!=expected:raise ValueError('Planning input changed: '+source)
    target=output/'summary.json';write_exclusive(target,dict(**metadata,methods=rows,
        status='FRESH_UNIFIED_PLANNING_OUTCOMES_RECORDED_EXECUTION_NOT_ADMITTED',all_attempts_retained=True,
        bounded_no_plan_is_NOT_physical_impossibility=True,new_PhysX_trials=0))
    print('[UNIFIED H3] Saved:',target,flush=True)
    return target


def write_unified_task_geometry(task):
    """Original declared endpoints on the already adopted, exact full environment."""
    import copy
    import numpy as np
    import hospital_mission_runtime as retained
    from hospital_mission_sequence import TASK_SCENARIOS
    from hospital_geometry_tools import pose_collision
    from hospital_h4_cohort_preparation_v4 import support_area_missing
    from hospital_path_execution import relative_pose
    paths=dict(H1=RESULT/'hospital_h1_geometry.json',
        H2=RESULT/'hospital_h2_setup/setup_r01/hospital_h2_geometry.json',
        H4=RESULT/'hospital_h4_experiments/cohort_09_full_task_r01/protocol.json',
        H5=RESULT/'hospital_h5_experiments/cohort_01_P_development_r01/protocol.json',H6=Path(retained.GEOMETRY))
    if task not in paths:raise ValueError('One explicit remaining original task required; H3 is already complete')
    original_path=paths[task];original=retained.read(original_path);g=copy.deepcopy(retained.read(retained.GEOMETRY))
    g.update(scenario=TASK_SCENARIOS[task],start_xyyaw=original['start_xyyaw'],goal_xyyaw=original['goal_xyyaw'],
        local_goal_xyyaw=list(relative_pose(original['goal_xyyaw'],original['start_xyyaw'])),
        original_fixed_task_geometry=str(original_path),original_fixed_task_geometry_sha256=file_sha(original_path),
        current_adopted_environment_retained=True,environment_condition='SAME_ADOPTED_H6_COMMON_ENVIRONMENT_NO_NEW_EDIT',
        actual_motion_authorized=False,use_for_final_statistics=False)
    with np.load(g['occupancy_grid']) as grid:
        if hashlib.sha256(grid['occupied'].tobytes()).hexdigest()!=g['planning_occupancy_digest']:raise ValueError('Map changed')
        for role in ('start_xyyaw','goal_xyyaw'):
            if (pose_collision(grid['occupied'],g[role],grid['origin_xy_m'],float(grid['resolution_m']),.010)
                    or support_area_missing(g[role],g['same_three_authored_support_rectangles'])>1e-9):
                raise ValueError('Original task endpoint blocked or unsupported; no silent pose change')
    digest=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest()[:16]
    target=RESULT/'paper_evidence_inventory/unified_task_geometry'/task/digest/'task_geometry.json'
    if target.exists():
        if retained.read(target)!=g:raise ValueError('Existing geometry changed')
    else:retained.evidence.write_exclusive(target,g)
    return target


def write_unified_evaluation_review(root=RESULT):
    """One-block case budget + connection verification, NOT live admission.

    24 new-version main decisions are not 24 guaranteed drives. The 22 old
    route references are not lost, but cannot silently become 1.330m runs.
    Five sensor cases and one explicit H5 replanning case are a separate
    proposed follow-up budget, not approved motion or a journal requirement.
    """
    from hospital_h3_turn_planning import model_bounds,MODEL_PLAN,MODEL_SHA
    from hospital_runtime_evidence_io import write_exclusive
    root=Path(root).resolve();migration_path=write_wheelbase_migration_review(root)
    migration=json.loads(migration_path.read_bytes());profile=migration['profile']
    if migration['counts']['verified_frozen_plan_replays']!=22:
        raise ValueError('All22 explicit old-plan replays required before new case accounting')
    connection=verify_unified_wheelbase_connections(profile,parameter_bounds=model_bounds())
    reference=_ijcas_path(root, 'hospital_h3_experiments/cohort_07_held_static_r01/route/P_dynamics_full_envelope/route_readback.json')
    reference_raw=json.loads(reference.read_bytes());contract=reference_raw['contract']
    cells=[];pins={str(migration_path):file_sha(migration_path),str(MODEL_PLAN):MODEL_SHA,
        str(reference):file_sha(reference),str(Path(__file__).resolve()):file_sha(__file__)}
    for task,method,path,planning_only in common_comparison_references(root):
        raw=json.loads(path.read_bytes());pins[str(path)]=file_sha(path)
        cells.append(dict(task=task,method=method,original_result=str(path),
            original_presentation=paper_outcome_presentation(raw,planning_only=planning_only),
            new_geometry_frozen_version_final_cell_completed=False,
            next_operation='NEW_COMMON_VERSION_PLANNING_THEN_FIRST_DRIVE_ONLY_IF_ADMITTED' if planning_only else
                'NEW_COMMON_VERSION_PLANNING_THEN_REVALIDATION_ONLY_IF_ADMITTED',
            legacy_physical_trial_was_run=not planning_only,
            old_verdict_NOT_relabelled=True,new_motion_authorized=False))
    for binding in connection['bindings']:pins[binding['source']]=binding['source_sha256']
    followups=[dict(case='S_'+task,scope='HOSPITAL_SENSOR_LOCALIZATION_NAV2',task=task,
        completed_by_this_review=False,new_motion_authorized=False) for task in ('H1','H2','H3','H4','H6')]
    followups.append(dict(case='H5_DECLARED_STOPPED_REPLANNING',scope='OBSERVE_STOP_REPLAN_RESUME_COMPLETE',
        static_cart_bypass_alone_does_NOT_complete_this_case=True,
        completed_by_this_review=False,new_motion_authorized=False))
    result=dict(version='UNIFIED1330_ONE_BLOCK_EVALUATION_CASE_ACCOUNTING_AND_PRESENTATION_v1',
        status='OFFLINE_REDUCED_DESIGN_AND_CONNECTION_REVIEW_NOT_LIVE_PROTOCOL',
        one_matched_execution_block_proposed=True,additional_second_block_NOT_mandatory_or_adopted=True,
        fifty_three_cell_proposal_NOT_an_approved_requirement=True,
        counts=dict(main_cells=24,new_geometry_final_cells_completed=0,main_cells_pending=24,
            old_physical_references_to_revalidate_if_new_plan_admitted=22,
            old_planning_only_cells_with_no_drive=2,maximum_main_drives_if_all_new_plans_admitted=24,
            frozen_old_control_sequences_failing_changed_model_pose_goal=migration['counts']['frozen_controls_outside_original_model_pose_goal_after_change'],
            separate_proposed_sensor_cases=5,separate_proposed_H5_replanning_cases=1,
            total_proposed_main_plus_followup_cases=30),
        remaining_actual_drive_count_NOT_determined_before_new_planning_and_preflight=True,
        compatibility_proof_may_reduce_reruns_but_none_admits_old1_290_trial_as_new1_330_trial_here=True,
        canonical_forward_geometry=profile['geometry'],profile=profile,connection_verification=connection,
        main_cells=cells,separate_proposed_followup_cases=followups,
        recommended_common_policy=dict(
            version='UNIFIED1330_COMMON50_AUTOMATIC_SPEED_ONE_BLOCK_CANDIDATE_v1',
            speed_cap_m_s=.050,candidate_target_speeds_m_s=[.003,.010,.030,.050],
            acceleration_m_s2=.020,steering_slew_deg_s=8.,curvature_cap_1_m=.7,
            physical_envelope_clearance_m=.010,body_rectangle_m=[1.630,.965],
            terminal_acceptance=dict(contract['acceptance']),
            common_controller_base_contract=dict(contract),
            actual_directional_speed_floors_and_phase_policy_must_be_replayed_in_execution_MODEL=True,
            low_speed_plan_overridden_by_faster_executor_NOT_an_execution_model_PASS=True,
            manual_P_only_speed_or_time_override_forbidden=True,
            task_simulation_caps_s=dict(H1=180.,H2=180.,H3=300.,H4=300.,H5=240.,H6=600.),
            task_wall_caps_s={task:900. for task in ('H1','H2','H3','H4','H5','H6')},
            task_caps_retained_from_prior_common_definitions_NOT_a_guarantee_of_adequacy=True,
            full_new_path_braking_cusp_final_hold_and_wall_time_adequacy_STILL_required=True,
            proposed_total_planning_wall_cap_s=180.,proposed_lattice_expansion_cap=60000,
            planning_budget_applies_to_all_methods_and_tasks_early_solution_does_NOT_wait_for_cap=True,
            finite_pool_generator_has_no_lattice_node_count_but_same_total_wall_cap=True,
            full_planning_deadline_enforcement_STILL_requires_new_planner_binding_validation=True,
            cached_seed_pool_setup_refinement_and_selection_time_must_be_reported=True,
            identical_random_seed_or_source_caster_joint_history_within_matched_block=True,
            same_source_model_physics_environment_map_start_goal_controller_for_all_four=True,
            common_task_variant_changes_and_collision_disabling_must_be_disclosed=True,
            H3_and_H6_must_be_continuous_no_midmission_restage_or_state_reset=True,
            planner_mechanism_differences_declared_NOT_identical_outputs_required=True,
            new_stage_requires_static_support_and_standstill_before_drive=True,
            no_automatic_retry_until_desired_PASS=True,new_motion_authorized=False),
        reporting_rules=dict(
            raw_PASS_FAIL_HOLD_preserved=True,HOLD_must_be_classified_by_cause_not_plotted_as_third_score=True,
            algorithmic_budget_no_plan_or_verified_runtime_guard_failure_is_mission_zero=True,
            engineering_error_is_not_algorithm_failure_or_success=True,
            missing_or_unexecuted_metrics_are_null_NOT_zero=True,
            success_completion_time_summary_must_include_success_count_and_failure_count=True,
            partial_error_clearance_or_time_not_ranked_as_full_mission_result=True,
            one_run_is_case_evidence_NOT_independent_repeatability_or_significance=True,
            all_attempts_and_engineering_exclusions_reported_separately=True),
        manuscript_rules=dict(
            prospective_platform_model_equations_parameters_and_figures_wheelbase_m=1.330,
            prospective_front_rear_axes_in_base_link_m=[.665,-.665],
            axes_are_NOT_CoM_support_arms=True,
            legacy1_290_results_and_existing_CICS_prior_work_must_retain_original_geometry=True,
            old_results_cannot_be_silently_rewritten_as_unified1330_results=True,
            algebraic_coefficient_basis_conversion_NOT_new_physical_identification=True,
            future_results_placeholders_not_filled_before_new_validation=True),
        new_PhysX_trials=0,current_live_executor_installed=False,actual_motion_authorized=False,input_sha256=pins)
    for path,expected in pins.items():
        if file_sha(path)!=expected:raise ValueError('Evaluation review input changed: '+path)
    key=hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    target=_ijcas_path(root, 'paper_evidence_inventory/unified_evaluation_review', key, 'evaluation_review.json')
    if target.exists():
        if json.loads(target.read_bytes())!=result:raise ValueError('Existing evaluation review differs; no overwrite')
    else:write_exclusive(target,result)
    print('One-block case accounting:',result['counts']);print('Saved unified evaluation review:',target)
    return target


def fixed_mode_candidate_permission_witness(source,manifest,occupied,origin,resolution):
    """Test a known B0 route against B1/B2's OWN declared planning models.

    This refutes 'no permissible geometric route' when a witness exists. It
    does NOT claim that either bounded search found this route, solve either
    method's search, replace its failure, or establish physical feasibility.
    P uses different propagation/hard constraints: no superset claim for P.
    """
    import patient_initial_stability.steering_lattice as lattice
    import plan_hospital_h1 as shared
    from hospital_geometry_tools import pose_collision
    from hospital_planning_tools import local_to_world
    if (source.get('method')!='B0_fixed_counterphase_ramp' or not source.get('plan_success')
            or not source.get('selected_actions') or len(source['selected_actions'])!=len(source['selected_states'])
            or source.get('planner_dynamics')!='kinematic' or source.get('initial_steering_propagated') is not False
            or source.get('initial_front_rad')!=0. or source.get('initial_rear_rad')!=0.):
        raise ValueError('Complete nominal B0 fixed-counterphase source candidate required')
    rows=[];goal=manifest['local_goal_xyyaw'];start=manifest['start_xyyaw']
    for method in ('B1_multimode_kinematic_ramp','B2_multimode_transition_cost_ramp'):
        modes,dynamics=shared.METHODS[method]
        if dynamics.rate_limited_steering or dynamics.apply_identified_response:
            raise ValueError('This witness only applies to the declared B1/B2 ideal propagation')
        state=lattice.State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
        cost=0.;collisions=samples=0;maximum_difference=0.;bounds_clear=True
        for action,saved in zip(source['selected_actions'],source['selected_states']):
            mode=action['mode'];curvature=action['curvature_1_m'];direction=action['direction'];speed=action['target_speed_m_s']
            if (mode!='coordinated_counterphase' or mode not in modes or curvature not in lattice.CURVATURES
                    or direction!=1 or type(speed) not in (int,float) or not math.isfinite(speed) or not 0.<speed<=shared.SPEED):
                raise ValueError('Source action must remain in original B0 and tested method action spaces')
            previous=state
            state,poses,edge=lattice.propagate(state,mode,curvature,direction,speed_m_s=shared.SPEED,
                dynamics=dynamics,rigid_body_model=None,target_speed_m_s=speed)
            switch=lattice.MODE_SWITCH_PENALTY_S if previous.started and mode!=previous.mode and dynamics.include_mode_switch_cost else 0.
            reversal=lattice.DIRECTION_SWITCH_PENALTY_S if previous.started and previous.direction!=0 and direction!=previous.direction else 0.
            front,rear=lattice.steering_targets(mode,curvature)
            transient=(lattice.STEERING_TRANSIENT_PENALTY_WEIGHT*max(abs(front-previous.front),abs(rear-previous.rear))/lattice.STEERING_RATE_RAD_S
                if previous.started and dynamics.include_steering_transient_cost else 0.)
            cost+=edge+switch+reversal+transient+.02*abs(curvature)+state.rigid_body_penalty_s
            bounds_clear=bounds_clear and -.3<=state.x<=goal[0]+.4
            maximum_difference=max(maximum_difference,abs(state.x-saved['x_m']),abs(state.y-saved['y_m']),
                abs(lattice.wrap(state.yaw-math.radians(saved['yaw_deg']))),abs(state.front-saved['front_rad']),abs(state.rear-saved['rear_rad']))
            for pose in poses:
                collisions+=bool(pose_collision(occupied,local_to_world(pose,start),origin,resolution,.010));samples+=1
        error=math.hypot(state.x-goal[0],state.y-goal[1]);heading=abs(lattice.wrap(state.yaw-goal[2]))
        checks=dict(all_actions_permitted_by_this_method=True,all_original_polygon_cell_envelopes_clear=collisions==0,
            original_search_x_bounds_satisfied=bool(bounds_clear),same_ideal_states_as_source=maximum_difference<=1e-10,
            terminal_position_within_25mm=error<=.025,terminal_heading_within_2deg=heading<=math.radians(2.),
            original_method_cost_is_finite=math.isfinite(cost))
        rows.append(dict(method=method,permitted_geometric_candidate_exists=all(checks.values()),checks=checks,
            original_method_candidate_cost_s=cost,action_count=len(source['selected_actions']),substep_count=samples,
            colliding_substeps=collisions,maximum_state_difference=maximum_difference,
            terminal_position_error_m=error,terminal_heading_error_deg=math.degrees(heading),
            witness_was_NOT_found_by_this_methods_recorded_search=True,physical_execution_feasibility_claimed=False))
    return dict(version='H5_known_B0_route_in_declared_B1_B2_action_spaces_DIAGNOSTIC_WITNESS_v1',
        status='PERMITTED_CANDIDATE_TEST_NOT_A_NEW_PLANNER_OR_PHYSICAL_SUCCESS',methods=rows,
        source_method='B0_fixed_counterphase_ramp',no_search_failure_relabelled_or_replaced=True,
        witness_NOT_final_B1_B2_comparison_plan=True,P_feasible_set_NOT_assumed_to_contain_ideal_baseline_sets=True,
        no_global_optimality_or_physical_feasibility_claim=True,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False)


def write_fixed_mode_candidate_permission_review(native_path,output=None):
    """Source-bound feasibility diagnosis, preserving all recorded failures."""
    import inspect
    import numpy as np
    import record_hospital_asset_measurement as tools
    from hospital_runtime_evidence_io import write_exclusive
    native_path=Path(native_path).resolve()
    if native_path.name!='planning_probe.json' or not native_path.is_file():raise ValueError('Completed original native probe required')
    native=json.loads(native_path.read_bytes());protocol=native['protocol']
    if (native.get('version')!='H5_SAME_GENERIC_LATTICE_ORIGINAL_P_REFINEMENT_NATIVE_EXACT_SAT_OFFLINE_v2'
            or native.get('status')!='ALL_FOUR_NATIVE_DEVELOPMENT_PROBES_RECORDED_NOT_PHYSICAL_VALIDATION'):
        raise ValueError('Exact retained original native development probe required')
    if tools.write_corridor_native_planning_probe(protocol['prior_probe'],native_path.parent)!=native_path:
        raise ValueError('Original complete native sources/map/model could not be verified')
    path=native_path.parent/'B0_fixed_counterphase_ramp.json';source=json.loads(path.read_bytes())
    manifest=json.loads(Path(protocol['common_geometry']).read_bytes())
    with np.load(protocol['common_grid']) as grid:
        report=fixed_mode_candidate_permission_witness(source,manifest,grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m']))
    report.update(original_native_probe=str(native_path),source_B0_candidate=str(path),
        actual_recorded_search_outcomes=[row for row in native['methods'] if row['method'] in ('B1_multimode_kinematic_ramp','B2_multimode_transition_cost_ramp')],
        input_sha256={**protocol['input_sha256'],**native['output_sha256'],str(native_path):file_sha(native_path)},
        analysis_source={f.__name__:inspect.getsource(f) for f in (fixed_mode_candidate_permission_witness,write_fixed_mode_candidate_permission_review)})
    key=hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else RESULT/'hospital_h5_geometry_candidates/feasibility_diagnosis'/key).resolve()
    target=output/'B1_B2_permitted_candidate_witness.json'
    for name,sha in report['input_sha256'].items():
        if file_sha(name)!=sha:raise ValueError('Original feasibility-witness input changed')
    if target.exists():
        try:saved=json.loads(target.read_bytes())
        except (ValueError,UnicodeError) as error:raise ValueError('Existing witness is edited/non-JSON; no overwrite') from error
        if saved!=report:raise ValueError('Existing witness differs; no overwrite')
        return target
    write_exclusive(target,report)
    print('Saved B1/B2 permitted-route witness (NOT a search or physical success):',target)
    print('permitted geometric candidates:',[(row['method'],row['permitted_geometric_candidate_exists']) for row in report['methods']])
    return target


def main():
    protocol = load_yaml(PROTOCOL_PATH)
    measurement_path = MEASUREMENT_PATH if MEASUREMENT_PATH.exists() else MEASUREMENT_TEMPLATE
    measurement = load_yaml(measurement_path)

    required_measurements = (
        "measurement_date",
        "measured_by",
        "location_or_asset_source",
        "bed.identifier",
        "bed.source_type",
        "bed.source_reference",
        "bed.overall_length_m",
        "bed.overall_width_m",
        "bed.collision_height_m",
        "bed.measurement_uncertainty_m",
        "doorway.identifier",
        "doorway.source_reference",
        "doorway.clear_width_m",
        "doorway.wall_thickness_m",
        "doorway.measurement_uncertainty_m",
        "ward_corridor.identifier",
        "ward_corridor.source_reference",
        "ward_corridor.clear_width_m",
        "ward_corridor.measurement_uncertainty_m",
        "mobile_obstacle.kind",
        "mobile_obstacle.identifier",
        "mobile_obstacle.source_reference",
        "mobile_obstacle.overall_length_m",
        "mobile_obstacle.overall_width_m",
        "mobile_obstacle.measurement_uncertainty_m",
        "scenario_poses.H1_bedside_departure.start_xyyaw",
        "scenario_poses.H1_bedside_departure.goal_xyyaw",
        "scenario_poses.H2_bedside_lateral_alignment.start_xyyaw",
        "scenario_poses.H2_bedside_lateral_alignment.goal_xyyaw",
        "scenario_poses.H3_doorway_immediate_turn.start_xyyaw",
        "scenario_poses.H3_doorway_immediate_turn.goal_xyyaw",
        "scenario_poses.H4_ward_corridor_turn.start_xyyaw",
        "scenario_poses.H4_ward_corridor_turn.goal_xyyaw",
        "scenario_poses.H5_obstacle_bypass_return.start_xyyaw",
        "scenario_poses.H5_obstacle_bypass_return.goal_xyyaw",
        "asset_and_collision_audit.hospital_stage_usd",
        "asset_and_collision_audit.bed_visual_prim",
        "asset_and_collision_audit.bed_collision_prim",
        "asset_and_collision_audit.visual_collision_bounds_match",
    )
    missing_measurements = [
        path for path in required_measurements if value_at(measurement, path) is None
    ]

    generation_guards = (
        "asset_and_collision_audit.visual_collision_bounds_match",
        "asset_and_collision_audit.cooked_collision_geometry_verified",
        "asset_and_collision_audit.complete_local_inventory_available",
        "asset_and_collision_audit.planner_geometry_generated_from_this_file",
        "asset_and_collision_audit.isaac_geometry_generated_from_this_file",
        "asset_and_collision_audit.nav2_map_generated_from_this_file",
        "freeze.geometry_frozen",
    )
    false_guards = [
        path for path in generation_guards if value_at(measurement, path) is not True
    ]

    protocol_checks = protocol_checks_for(protocol)

    evidence_paths = {
        "platform_measurement_source": _ijcas_path(WORKSPACE, 'src/patient_transport_description/config/patient_transport_lab_measurements.yaml'),
        "ijcas_platform_usd": _ijcas_path(WORKSPACE, 'isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd'),
        "corridor_pilot_plan": _ijcas_path(WORKSPACE, 'studies/IJCAS/results/rigid_body_edge_gate/corridor_90_hierarchical_rigid_body_envelope.json'),
        "initial_motion_evidence": _ijcas_path(WORKSPACE, 'studies/CICS26/docs/cics26_final_two_page_integrated_results.md'),
    }
    evidence_checks = {name: path.exists() for name, path in evidence_paths.items()}

    hardcoded_scripts = []
    generalization_guards = {
        "studies/IJCAS/scripts/run_integrated_validation.sh": (
            "IJCAS_SCENARIO=$(python3",
            "scenario:=\"$IJCAS_SCENARIO\"",
        ),
        "studies/IJCAS/scripts/analyze_integrated_rigid_body_envelope_execution.py": (
            '"--planner-result"',
            'scenario = planner.get("scenario")',
        ),
        "studies/IJCAS/scripts/plan_hierarchical_rigid_body_envelope_gate.py": (
            '"--scenario"',
            "scenario_name = args.scenario",
        ),
    }
    for relative, required_tokens in generalization_guards.items():
        path = _ijcas_path(WORKSPACE, relative)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if not all(token in text for token in required_tokens):
            hardcoded_scripts.append(relative)

    bedside_report = (
        _ijcas_path(WORKSPACE, 'studies/IJCAS/results/operational_task_gate/bedside_dynamic_repeat_gate.md')
    )
    bedside_status = "MISSING"
    if bedside_report.exists():
        text = bedside_report.read_text(encoding="utf-8")
        if "**VERDICT: PASS**" in text:
            bedside_status = "PASS"
        elif "**VERDICT: HOLD**" in text:
            bedside_status = "HOLD"
        elif "**VERDICT: FAIL**" in text:
            bedside_status = "FAIL"

    setup_ready = (
        not missing_measurements
        and not false_guards
        and all(protocol_checks.values())
        and all(evidence_checks.values())
        and not hardcoded_scripts
    )
    verdict = "PASS" if setup_ready else "HOLD"

    result = {
        "schema_version": 1,
        "verdict": verdict,
        "meaning": "ready_for_pilot_trials" if setup_ready else "setup_incomplete",
        "measurement_record": str(measurement_path),
        "protocol_checks": protocol_checks,
        "evidence_checks": evidence_checks,
        "missing_measurements": missing_measurements,
        "false_generation_or_freeze_guards": false_guards,
        "corridor_90_hardcoded_scripts": hardcoded_scripts,
        "legacy_bedside_repeat_gate": bedside_status,
        "claim_guard": (
            "Legacy corridor and provisional bedside results remain pilot evidence; "
            "they cannot be counted as final hospital-paper trials."
        ),
    }

    RESULT.mkdir(parents=True, exist_ok=True)
    json_path = RESULT / "hospital_paper_readiness.json"
    md_path = RESULT / "hospital_paper_readiness.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# IJCAS hospital-paper readiness audit",
        "",
        f"**VERDICT: {verdict}**",
        "",
        "The PASS condition means that hospital-environment pilot trials may start. "
        "It does not authorize final paper trials or a physical-platform claim.",
        "",
        "## Protocol",
        "",
    ]
    for name, passed in protocol_checks.items():
        lines.append(f"- {'PASS' if passed else 'HOLD'}: {name}")
    lines.extend(["", "## Existing evidence", ""])
    for name, passed in evidence_checks.items():
        lines.append(f"- {'FOUND' if passed else 'MISSING'}: {name}")
    lines.extend(
        [
            "",
            "## Blocking geometry fields",
            "",
            *(
                [f"- `{item}`" for item in missing_measurements]
                if missing_measurements
                else ["- none"]
            ),
            "",
            "## Generation/freeze guards not yet satisfied",
            "",
            *(
                [f"- `{item}`" for item in false_guards]
                if false_guards
                else ["- none"]
            ),
            "",
            "## Scenario-generalization work",
            "",
            *(
                [f"- hard-coded `corridor_90_turn`: `{item}`" for item in hardcoded_scripts]
                if hardcoded_scripts
                else ["- none"]
            ),
            "",
            "## Legacy evidence status",
            "",
            f"- bedside repeat gate: {bedside_status}",
            "- corridor and provisional bedside runs remain pilot evidence only.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== IJCAS HOSPITAL-PAPER READINESS AUDIT ===")
    print("protocol checks:", sum(protocol_checks.values()), "/", len(protocol_checks))
    print("existing evidence:", sum(evidence_checks.values()), "/", len(evidence_checks))
    print("missing measurement fields:", len(missing_measurements))
    print("unsatisfied generation/freeze guards:", len(false_guards))
    print("corridor-90 hard-coded scripts:", len(hardcoded_scripts))
    print("legacy bedside repeat gate:", bedside_status)
    print("VERDICT:", verdict)
    print("saved:", json_path)
    print("saved:", md_path)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-ledger',action='store_true',help='Read-only scope accounting; no legacy Markdown/setup audit overwrite')
    parser.add_argument('--full-scope-inventory',action='store_true',help='H1-H6 bounded saved-artifact inventory; preserve all route outcomes and separate engineering/MODEL/sensor scope, no live access')
    parser.add_argument('--common-protocol-review',action='store_true',help='Explicit 24-cell controller/speed compatibility and outcome-stage review; offline candidate only, no drive authority')
    parser.add_argument('--wheelbase-migration-review',action='store_true',help='Offline unified1.330m candidate, exact coefficient-basis transfer and frozen-plan impact; never change live inputs or old verdicts')
    parser.add_argument('--unified-evaluation-review',action='store_true',help='Offline1.330m command/timing/native parity, one-block24-cell accounting and cause-specific HOLD presentation; no live authority')
    parser.add_argument('--execution-figure',type=Path,help='Explicit actual H4 result; measured PNG/PDF without live access or legacy-figure overwrite')
    parser.add_argument('--figure-output',type=Path,help='Optional fresh derived-output directory; existing differing figures are never overwritten')
    parser.add_argument('--prediction-review',type=Path,help='Explicit actual H4 outcome versus BOTH previously pinned nominal MODEL resolutions; no fit/live access')
    parser.add_argument('--prediction-output',type=Path,help='Optional fresh derived prediction-review directory; never overwrite other outputs')
    parser.add_argument('--development-comparison',type=Path,nargs='+',help='Explicit intact actual H4 outcomes; no automatic best/latest or final-statistical qualification')
    parser.add_argument('--comparison-output',type=Path,help='Optional fresh output directory for an explicit development comparison')
    parser.add_argument('--comparison-figure',type=Path,nargs='+',help='Explicit unique matched completed H4 outcomes; actual-route PNG/PDF with no final-statistics claim')
    parser.add_argument('--comparison-figure-output',type=Path,help='Optional fresh directory for the matched development figure')
    parser.add_argument('--phase-response-review',type=Path,nargs='+',help='Explicit actual H4 outcomes; measured straight/curve and terminal event accounting, no fit/live access')
    parser.add_argument('--phase-output',type=Path,help='Optional fresh output directory for measured phase response diagnosis')
    parser.add_argument('--orientation-response-review',type=Path,nargs='+',help='Own-reference signed heading decomposition and frozen-segment measured steering response; no fit/live access')
    parser.add_argument('--orientation-output',type=Path,help='Optional fresh directory for measured orientation diagnosis')
    parser.add_argument('--development-table',type=Path,nargs='+',help='Explicit actual H1/H2/H4 development outcomes to provenance-tagged CSV/LaTeX; no final-statistical claim')
    parser.add_argument('--table-output',type=Path,help='Optional fresh directory for development manuscript table assets')
    parser.add_argument('--terminal-reference-review',type=Path,nargs='+',help='Actual matched H4 outcomes and ALL FOUR frozen reference/terminal pose interfaces; no new policy or motion')
    parser.add_argument('--terminal-reference-output',type=Path,help='Optional fresh output directory for terminal reference geometry review')
    parser.add_argument('--terminal-connection-candidate',action='store_true',help='Prospective ALL-FOUR original-pool terminal pose-interface planning candidate; no measured fit, adoption or motion')
    parser.add_argument('--terminal-connection-output',type=Path,help='Optional fresh directory for the prospective terminal connection candidate')
    parser.add_argument('--terminal-connection-model-review',type=Path,help='Explicit prospective connection candidate; original 10/20ms nominal executor on changed selections only, no PhysX')
    parser.add_argument('--connection-model-output',type=Path,help='Optional fresh directory for nominal connection diagnostic')
    parser.add_argument('--terminal-margin-sensitivity',type=Path,help='Fixed all-four pose-only margin grid from the same original candidate decisions; no policy adoption or motion')
    parser.add_argument('--margin-output',type=Path,help='Optional fresh geometric sensitivity directory')
    parser.add_argument('--terminal-connection-execution-bundle',type=Path,nargs=2,metavar=('CANDIDATE','NOMINAL_REVIEW'),help='Freeze separate all-four candidate plans for the unchanged runtime; no live action')
    parser.add_argument('--connection-bundle-output',type=Path,help='Optional fresh prospective execution-bundle directory')
    parser.add_argument('--planner-version-contrast',type=Path,nargs=2,metavar=('ORIGINAL','PROSPECTIVE'),help='Explicit actual original/prospective P outcomes; preserve failed partials and disclose changed planner references')
    parser.add_argument('--version-contrast-output',type=Path,help='Optional fresh output directory for individual P planner-version contrast')
    parser.add_argument('--manuscript-assembly',type=Path,nargs=2,metavar=('DEVELOPMENT_TABLE','P_VERSION_CONTRAST'),help='Native TeX working manuscript with replayed existing initial-motion evidence; not submission-ready or new trials')
    parser.add_argument('--manuscript-output',type=Path,help='Optional fresh native manuscript output directory')
    parser.add_argument('--korean-review',action='store_true',help='Korean-first editable DOCX and Figure plan from the user CICS paper; no translation or scope removal')
    parser.add_argument('--korean-output',type=Path,help='Optional fresh author-review output directory; never overwrite author edits')
    parser.add_argument('--platform-geometry-review',action='store_true',help='Native USD-only source/nominal/load-geometry provenance and Korean Figure 2; no Kit/live access or model change')
    parser.add_argument('--platform-geometry-output',type=Path,help='Optional fresh platform figure directory; never overwrite author edits')
    parser.add_argument('--fixed-mode-permission-review',type=Path,help='Completed H5 native probe: B0 candidate permission in B1/B2, diagnostic only; no failed search relabelled')
    parser.add_argument('--permission-output',type=Path,help='Optional fresh feasibility-diagnosis directory')
    args=parser.parse_args()
    if sum(bool(x) for x in (args.execution_ledger,args.full_scope_inventory,args.common_protocol_review,args.wheelbase_migration_review,args.unified_evaluation_review,args.execution_figure,args.prediction_review,args.development_comparison,args.comparison_figure,args.phase_response_review,args.orientation_response_review,args.development_table,args.terminal_reference_review,args.terminal_connection_candidate,args.terminal_connection_model_review,args.terminal_margin_sensitivity,args.terminal_connection_execution_bundle,args.planner_version_contrast,args.manuscript_assembly,args.korean_review,args.platform_geometry_review,args.fixed_mode_permission_review))>1:parser.error('Choose one explicit offline action')
    if args.figure_output and not args.execution_figure:parser.error('--figure-output requires --execution-figure')
    if args.prediction_output and not args.prediction_review:parser.error('--prediction-output requires --prediction-review')
    if args.comparison_output and not args.development_comparison:parser.error('--comparison-output requires --development-comparison')
    if args.comparison_figure_output and not args.comparison_figure:parser.error('--comparison-figure-output requires --comparison-figure')
    if args.phase_output and not args.phase_response_review:parser.error('--phase-output requires --phase-response-review')
    if args.orientation_output and not args.orientation_response_review:parser.error('--orientation-output requires --orientation-response-review')
    if args.table_output and not args.development_table:parser.error('--table-output requires --development-table')
    if args.terminal_reference_output and not args.terminal_reference_review:parser.error('--terminal-reference-output requires --terminal-reference-review')
    if args.terminal_connection_output and not args.terminal_connection_candidate:parser.error('--terminal-connection-output requires --terminal-connection-candidate')
    if args.connection_model_output and not args.terminal_connection_model_review:parser.error('--connection-model-output requires --terminal-connection-model-review')
    if args.margin_output and not args.terminal_margin_sensitivity:parser.error('--margin-output requires --terminal-margin-sensitivity')
    if args.connection_bundle_output and not args.terminal_connection_execution_bundle:parser.error('--connection-bundle-output requires --terminal-connection-execution-bundle')
    if args.version_contrast_output and not args.planner_version_contrast:parser.error('--version-contrast-output requires --planner-version-contrast')
    if args.manuscript_output and not args.manuscript_assembly:parser.error('--manuscript-output requires --manuscript-assembly')
    if args.korean_output and not args.korean_review:parser.error('--korean-output requires --korean-review')
    if args.platform_geometry_output and not args.platform_geometry_review:parser.error('--platform-geometry-output requires --platform-geometry-review')
    if args.permission_output and not args.fixed_mode_permission_review:parser.error('--permission-output requires --fixed-mode-permission-review')
    if args.fixed_mode_permission_review:write_fixed_mode_candidate_permission_review(args.fixed_mode_permission_review,args.permission_output)
    elif args.platform_geometry_review:write_platform_geometry_review(args.platform_geometry_output)
    elif args.korean_review:write_korean_manuscript_review(args.korean_output)
    elif args.manuscript_assembly:write_manuscript_development_assembly(*args.manuscript_assembly,args.manuscript_output)
    elif args.planner_version_contrast:write_planner_version_development_contrast(*args.planner_version_contrast,args.version_contrast_output)
    elif args.terminal_margin_sensitivity:write_common_terminal_margin_sensitivity(args.terminal_margin_sensitivity,args.margin_output)
    elif args.terminal_connection_execution_bundle:write_terminal_connection_execution_bundle(*args.terminal_connection_execution_bundle,args.connection_bundle_output)
    elif args.terminal_connection_model_review:write_terminal_connection_model_review(args.terminal_connection_model_review,args.connection_model_output)
    elif args.terminal_connection_candidate:write_terminal_connection_candidate(args.terminal_connection_output)
    elif args.comparison_figure:write_development_comparison_figure(args.comparison_figure,args.comparison_figure_output)
    elif args.terminal_reference_review:write_terminal_reference_review(args.terminal_reference_review,args.terminal_reference_output)
    elif args.development_table:write_development_manuscript_table(args.development_table,args.table_output)
    elif args.phase_response_review:write_phase_response_review(args.phase_response_review,args.phase_output)
    elif args.orientation_response_review:write_orientation_response_review(args.orientation_response_review,args.orientation_output)
    elif args.development_comparison:write_explicit_development_comparison(args.development_comparison,args.comparison_output)
    elif args.prediction_review:write_frozen_prediction_review(args.prediction_review,args.prediction_output)
    elif args.execution_figure:write_execution_figure(args.execution_figure,args.figure_output)
    elif args.wheelbase_migration_review:write_wheelbase_migration_review()
    elif args.unified_evaluation_review:write_unified_evaluation_review()
    elif args.common_protocol_review:write_common_protocol_review()
    elif args.full_scope_inventory:write_full_scope_inventory()
    elif args.execution_ledger:write_execution_ledger()
    else:main()

