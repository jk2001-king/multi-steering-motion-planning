"""Prepare H4 commissioning definitions offline; never schedule Kit/trials.

The default action preserves the already frozen static entry. The optional
initial-motion design is a separate fixed 0.65 m commissioning component,
not a planning-method result or permission to move the retained robot.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np

import hospital_h4_subfloor_compact_path_launch as launch


PRELIMINARY_SHORT_DESIGN = launch.DESIGN/'initial_motion_design.json'
INITIAL_DESIGN = launch.DESIGN/'initial_motion_design_0p65m.json'
# The inherited >=95% progress and <=30 mm endpoint bands must both make
# sense for this new commissioning task. L > .030/.05 = .60 m, then take
# the next ORIGINAL 5 cm edge: .65 m. No tolerance or controller is relaxed.
INITIAL_EDGES = math.ceil(launch.execution.ACCEPTANCE['position_error_m']/(1.-.95)/.05)+1
INITIAL_DISTANCE_M = round(.05*INITIAL_EDGES,10)


def initial_motion_reference(plans):
    """Fixed original straight prefix, with no data-dependent distance choice.

    All four frozen methods prescribe these exact same first 13 commands.
Use the original straight geometric reference, not P's model-predicted
lateral bias as a newly fitted commissioning goal. B0 is only the source
reference/schema here; this component must never be counted as a B0 trial.
"""
    methods = tuple(launch.execution.METHODS)
    if set(plans) != set(methods):
        raise ValueError('All four unchanged frozen H4 method references required')
    expected = dict(mode='coordinated_counterphase', curvature_1_m=0.,
        direction=1, target_speed_m_s=.05)
    for plan in plans.values():
        launch.execution.validate_plan(plan)
        if (len(plan['selected_actions']) < INITIAL_EDGES
                or plan['selected_actions'][:INITIAL_EDGES] != [expected]*INITIAL_EDGES
                or plan['start_world_xyyaw'] != plans[methods[0]]['start_world_xyyaw']):
            raise ValueError('All methods must retain the exact common forward straight prefix')
    original = plans[methods[0]]
    prefix = copy.deepcopy(original)
    prefix['selected_actions'] = copy.deepcopy(original['selected_actions'][:INITIAL_EDGES])
    prefix['selected_states'] = copy.deepcopy(original['selected_states'][:INITIAL_EDGES])
    origin = original['start_world_xyyaw']
    for i, state in enumerate(prefix['selected_states'], 1):
        expected_pose = [origin[0]+.05*i*math.cos(origin[2]),
            origin[1]+.05*i*math.sin(origin[2]), origin[2]]
        if any(abs(a-b) > 1e-10 for a,b in zip(state['world_xyyaw'], expected_pose)):
            raise ValueError('Original geometric commissioning prefix is not the exact fixed straight reference')
    # Do not carry full-route predicted errors, timing or candidate cost into
    # a new short component where those numbers have a different meaning.
    for key in tuple(prefix):
        if (key.startswith('predicted_') or key.startswith('nominal_')
                or key in ('native_cost_s','path_length_m','action_count','reference_center_arc_m',
                    'selected_candidate_id','native_candidate_evaluations','sampled_reference_geometry_and_TRUE_support')):
            prefix.pop(key)
    prefix.update(goal_world_xyyaw=prefix['selected_states'][-1]['world_xyyaw'],
        goal_local_xyyaw=list(launch.execution.relative_pose(prefix['selected_states'][-1]['world_xyyaw'],origin)),
        original_full_route_goal_world_xyyaw=original['goal_world_xyyaw'],
        original_full_route_candidate_id=original['selected_candidate_id'],
        diagnostic_only_prefix_NOT_a_new_planning_result=True,
        reference_schema_source_method_NOT_an_executed_method_trial=methods[0],
        actual_motion_authorized=False,use_for_final_statistics=False)
    launch.execution.validate_plan(prefix)
    reference=launch.execution.PathReference(prefix)
    if not math.isclose(reference.arc[-1],INITIAL_DISTANCE_M,rel_tol=0.,abs_tol=1e-10):
        raise ValueError('Commissioning reference must remain exactly 0.65 m')
    return prefix


def build_initial_motion_design():
    """Geometry plus original-controller MODEL screen only; no drive admission."""
    import screen_hospital_h4_common_executor as screen
    from hospital_h4_geometry_metrics import densify_poses
    from hospital_geometry_tools import pose_collision

    bundle, _, plans=launch.load_frozen_bundle()
    prefix=initial_motion_reference(plans)
    ref=launch.execution.PathReference(prefix)
    support=screen.bundle.cohort.support.load_design()
    covers=screen.bundle.cohort.support_covers(support)
    native=launch.preparation.read_small(screen.bundle.cohort.MODELS)
    with np.load(launch.cohort.MAP) as grid:
        occupied,origin,resolution=grid['occupied'],grid['origin_xy_m'],float(grid['resolution_m'])
    poses=densify_poses(ref.poses)
    reference_clear=all(not pose_collision(occupied,pose,origin,resolution,clearance=.010) for pose in poses)
    missing=max(screen.bundle.cohort.support_area_missing(pose,covers) for pose in poses)
    if not reference_clear or missing>1e-9:
        raise ValueError('Fixed commissioning prefix failed the unchanged full-envelope/support screen')
    cases=[]
    for dt in screen.TIMESTEPS_S:
        record,rows=screen.case(prefix,native['rigid_body_parameter_bounds'],dt,occupied,origin,resolution,support)
        record.update(trace_samples=len(rows),peak_command_speed_m_s=max(abs(r['command_speed_m_s']) for r in rows))
        cases.append(record)
    preliminary=launch.preparation.read_small(PRELIMINARY_SHORT_DESIGN)
    if (preliminary['distance_m']!=.20 or preliminary['status']!='HOLD_OFFLINE_COMPONENT_MODEL_SCREEN'
            or any({k for k,v in c['checks'].items() if not v}!={'at_least_95_percent_reference_progress'}
                for c in preliminary['original_common_executor_MODEL_cases'])):
        raise ValueError('Preserve the exact preliminary 20 cm progress-band mismatch, not a physical failure')
    files=(launch.BUNDLE,screen.bundle.cohort.MODELS,launch.cohort.MAP,screen.bundle.cohort.support.TARGET,
        Path(screen.__file__),Path(launch.execution.__file__),PRELIMINARY_SHORT_DESIGN)
    pins={**bundle['input_sha256'],**bundle['output_sha256'],
        **{str(path):launch.preparation.sha(path) for path in files}}
    launch.preparation.verify(pins)
    passed=all(case['model_status']=='MODEL_SCREEN_PASS' for case in cases)
    return launch.canonical(dict(version='H4_fixed_common_0p65m_initial_motion_design_v1',
        status='OFFLINE_COMPONENT_MODEL_SCREEN_PASSED_ACTUAL_STATIC_LIVE_ADAPTER_AND_CONSENT_PENDING' if passed
            else 'HOLD_OFFLINE_COMPONENT_MODEL_SCREEN',
        role='engineering_initial_motion_and_measured_stop_NOT_planning_method_comparison',
        distance_m=INITIAL_DISTANCE_M,original_common_command_edge_count=INITIAL_EDGES,
        length_derivation=dict(original_endpoint_band_m=launch.execution.ACCEPTANCE['position_error_m'],
            original_required_progress_fraction=.95,original_edge_length_m=.05,
            minimum_distance_for_30mm_band_and_95_percent_progress_m=.60,
            selected_next_original_edge_distance_m=INITIAL_DISTANCE_M),
        preliminary_0p20m_MODEL_HOLD_preserved=str(PRELIMINARY_SHORT_DESIGN),
        preliminary_is_NOT_an_actual_static_or_motion_failure=True,
        controller_tolerances_progress_requirement_and_model_NOT_retuned=True,
        no_actual_motion_data_or_successful_caster_entry_used_to_choose_length=True,
        diagnostic_reference=prefix,original_full_route_and_task_NOT_modified=True,
        command_limits=launch.execution.COMMAND_LIMITS,guards=launch.execution.GUARDS,
        acceptance=launch.execution.ACCEPTANCE,terminal_controller=launch.execution.CONTROLLER_CONFIG,
        initial_pre_hold_s=2.,fixed_terminal_hold_s=5.,fixed_tail_evaluation_s=3.,
        reference_geometry=dict(sample_count=len(poses),expanded_full_envelope_clear=reference_clear,
            expansion_each_side_m=.010,maximum_missing_authored_support_area_m2=missing,
            sample_spacing_m_deg=[.005,.25],cooked_collision_or_continuous_safety_verified=False),
        original_common_executor_MODEL_cases=cases,
        MODEL_resolution_cases_are_NOT_independent_validation_or_actual_trials=True,
        body_response_caster_traction_and_hold_force_NOT_validated_by_this_design=True,
        required_actual_static_result=str(launch.STATIC_OUTPUT/(launch.STATIC_RUN_ID+'.json')),
        actual_static_not_assumed_or_relabelled=True,
        required_before_any_motion=['own corrected actual static PASS/raw CSV independently replayed',
            'retained PAUSED 13 links and all 12 q/qd; complete unchanged protected environment',
            'shared runtime action implemented and tested without modifying the pending static input snapshot',
            'separate explicit consent for exactly ONE component'],
        approval_for_static_does_NOT_authorize_this_component=True,
        commissioning_run_changes_entry_history_and_cannot_seed_final_matched_repeats=True,
        new_trial_count=0,qualified_final_comparison_cells=0,
        actual_motion_authorized=False,actual_drive_admission=False,use_for_final_statistics=False,
        automatic_retry=False,CICS_runtime_integration=False,real_robot_required=False,
        analysis_source_sha256={str(Path(__file__)):launch.preparation.sha(Path(__file__))},
        input_sha256=pins))


def prepare_static_definition():
    launch.write_runtime_source_bundle()
    # Retrospective evidence stays separate from the immutable raw FAIL. No
    # adopted query criterion, physical result or live marker is overwritten.
    reviewed = launch.review_recorded_static_closeout()
    if launch.CLOSEOUT_REVIEW.exists():
        launch.load_recorded_closeout_review()
    else:
        launch.evidence.write_exclusive(launch.CLOSEOUT_REVIEW, reviewed)
    report = launch.prepare_retained_static_protocol()
    if launch.STATIC_DESIGN.exists():
        launch.load_retained_static_protocol()
    else:
        launch.evidence.write_exclusive(launch.STATIC_DESIGN, report)
    print('H4 current-actor identity diagnosis complete OFFLINE; raw FAIL unchanged')
    print('Next is ONE separately authorized corrected prospective STATIC trial, not a route')
    print('Same 1s settling + 2s recording, 26+4 queries, physical limits, material and gains')
    print('Short initial-motion validation and any whole route remain NOT admitted')
    print('saved:', launch.STATIC_DESIGN)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('static-support','initial-motion-design','initial-motion-admission','fresh-whole-path','first-route-after-static','continue-staged-workflow'),default='static-support')
    parser.add_argument('--original-fresh-protocol',type=Path)
    parser.add_argument('--cohort',default='cohort_04_full_task_r01')
    parser.add_argument('--method',choices=launch.execution.METHODS,default='P_dynamics_full_envelope')
    parser.add_argument('--previous-route',type=Path)
    parser.add_argument('--common-simulation-budget-s',type=float,choices=(300.,))
    parser.add_argument('--terminal-connection-bundle',type=Path,help='Separate verified all-four prospective planner bundle; no original-reference replacement or motion')
    args=parser.parse_args()
    if args.terminal_connection_bundle and args.action!='fresh-whole-path':
        parser.error('--terminal-connection-bundle requires a separate fresh-whole-path design')
    if args.action=='continue-staged-workflow':
        if args.original_fresh_protocol is None:raise ValueError('Exact original unused staged protocol required')
        report,baseline,payload=launch.prepare_staged_workflow_continuation(args.original_fresh_protocol)
        amendment=report['staged_boundary_continuation']
        output=Path(amendment['protocol']);archive=Path(amendment['executed_source_archive'])
        baseline_path=Path(amendment['amended_preparation'])
        output.parent.mkdir(parents=True,exist_ok=True)
        if baseline_path.exists():
            if launch.preparation.read_small(baseline_path)!=baseline:raise ValueError('Existing pre-static baseline differs; no overwrite')
        else:launch.write_runtime_evidence(baseline_path,baseline)
        if archive.exists():
            if archive.read_bytes()!=payload:raise ValueError('Existing remainder source archive differs; no overwrite')
        else:
            with archive.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
        launch.preparation.verify(report['input_sha256'])
        if output.exists():
            if launch.preparation.read_small(output)!=report:raise ValueError('Existing remainder definition differs; no overwrite')
        else:launch.write_runtime_evidence(output,report)
        launch.load_staged_workflow_continuation(output)
        print('Exact two Top aperture fields reviewed; original preparation/error/source archive unchanged.')
        print('OFFLINE ONLY: same stopped actor, unused static once -> conditional original B2 once.')
        print('NO Stop/restaging/display refresh/Play or physical trial performed.')
        print('saved:',output)
        return
    if args.action=='first-route-after-static':
        if args.original_fresh_protocol is None:
            raise ValueError('Explicit own executed fresh protocol required; no automatic cohort selection')
        report=launch.prepare_first_route_continuation(args.original_fresh_protocol)
        output=Path(report['protocol']);archive=Path(report['new_runtime_archive'])
        payload=launch.fresh_runtime_source_bundle(
            include_visual=report.get('runtime_archive_includes_visual_sources') is True)
        output.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():
            if archive.read_bytes()!=payload:raise ValueError('Continuation source archive differs; no overwrite')
        else:
            with archive.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
        launch.preparation.verify(report['input_sha256'])
        if output.exists():
            launch.load_first_route_continuation(output)
        else:launch.write_runtime_evidence(output,report)
        print('Own fresh static physical CSV + 26+4 queries replayed PASS; clocks reviewed separately.')
        print('Original static/workflow/source archive preserved. NO new static, Stop, Play or route.')
        print('Only the first not-yet-started frozen whole route remains; exact live continuity still required.')
        print('saved:',output)
        return
    if args.action=='fresh-whole-path':
        report=launch.prepare_fresh_trial_protocol(args.cohort,args.method,
            previous_route=args.previous_route,simulation_budget_s=args.common_simulation_budget_s,
            initialize_visuals_before_static=args.common_simulation_budget_s is not None,
            terminal_connection_bundle=args.terminal_connection_bundle)
        context=launch.fresh_trial_context(report)
        payload=launch.fresh_runtime_source_bundle(
            include_visual=(report.get('restore_new_actor_visual_cache_after_static_only') is True
                or report.get('initialize_new_actor_visuals_before_static') is True))
        context.output.mkdir(parents=True,exist_ok=True)
        if context.source_archive.exists():
            if context.source_archive.read_bytes()!=payload:raise ValueError('Fresh runtime already frozen differently; no overwrite')
        else:
            with context.source_archive.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
        launch.preparation.verify(report['input_sha256'])
        if context.protocol_path.exists():
            launch.load_fresh_trial_protocol(context.protocol_path)
        else:launch.write_runtime_evidence(context.protocol_path,report)
        print('Fresh single fixed-start whole-H4 design prepared OFFLINE; no Isaac operation.')
        print('saved:',context.protocol_path)
        return
    if args.action=='static-support':
        prepare_static_definition()
        return
    if args.action=='initial-motion-admission':
        report=launch.prepare_initial_motion_protocol()
        payload=launch.initial_motion_source_bundle()
        if launch.INITIAL_SOURCES.exists():
            if launch.INITIAL_SOURCES.read_bytes()!=payload:
                raise ValueError('Initial-motion runtime sources already frozen differently; no overwrite')
        else:
            with launch.INITIAL_SOURCES.open('xb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
        launch.preparation.verify(report['input_sha256'])
        if launch.INITIAL_PROTOCOL.exists():
            if launch.preparation.read_small(launch.INITIAL_PROTOCOL)!=report:
                raise ValueError('Initial-motion definition already frozen differently; no overwrite')
        else:
            launch.evidence.write_exclusive(launch.INITIAL_PROTOCOL,report)
        print('Actual corrected static raw CSV + 26+4 queries replayed PASS.')
        print('ONE fixed 0.65 m initial-motion protocol prepared OFFLINE; separate motion consent required.')
        print('No full route, final comparison, Play, Stop or reset performed.')
        print('saved:',launch.INITIAL_PROTOCOL)
        return
    if INITIAL_DESIGN.exists():
        record=json.loads(INITIAL_DESIGN.read_bytes())
        launch.preparation.verify(record['input_sha256'])
    else:
        record=build_initial_motion_design()
        launch.evidence.write_exclusive(INITIAL_DESIGN,record)
    print('H4 fixed 0.65 m initial-motion OFFLINE design:',record['status'])
    for case in record['original_common_executor_MODEL_cases']:
        print(case['integration_dt_s'],case['model_status'],'mm/deg:',
            1000*case['final_position_error_m'],case['final_heading_error_deg'],'error:',case['error'])
    print('Actual static, live continuity, adapter and separate motion consent still required. NO Play or trial.')
    print('saved:',INITIAL_DESIGN)


if __name__ == '__main__':
    main()
