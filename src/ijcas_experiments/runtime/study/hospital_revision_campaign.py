"""Predeclared reviewer experiment design and shared per-cell orchestration.

Import is pure. Protocol export, planning and SDK execution are separate
explicit entry points. All cells terminate on their own outcome; no trial is
replaced by a successful retry and no historical trial becomes a repetition.
"""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import hospital_controlled_comparison as planning
import hospital_controlled_runtime as runtime
import hospital_controlled_route_binding as binding

ROOT, GATE = planning.ROOT, planning.GATE
OUTPUT = GATE/'revision_campaign_20261121'
PROTOCOL = OUTPUT/'protocol.json'
METHODS = tuple(planning.shared.METHODS)


def definition():
    """Five prespecified draws per task, paired across four methods."""
    rng = np.random.default_rng(20261121)
    ready_root = GATE/'paper_evidence_inventory/revised_P_uniform_evaluation_r02/ready'
    rows, inputs, tasks = [], {}, {}
    for number in range(1,7):
        task = 'H'+str(number)
        ready_file = ready_root/(task+'.json')
        ready = runtime.read(ready_file)
        p = Path(ready['planning_sources'][planning.P])
        reference = runtime.read(p)
        geometry_path = Path(reference['geometry_manifest'])
        geometry = runtime.read(geometry_path)
        if (geometry['scenario'] != runtime.sequence.TASK_SCENARIOS[task]
                or geometry['planning_occupancy_digest'] != reference['planning_occupancy_digest']):
            raise ValueError('Original fixed task and common environment required')
        for path in (ready_file,p,geometry_path):
            inputs[str(path)] = runtime.sha(path)
        tasks[task] = dict(original_geometry=str(geometry_path),
            original_start_xyyaw=geometry['start_xyyaw'], goal_xyyaw=geometry['goal_xyyaw'],
            geometry_profile=reference['geometry_profile'],
            simulation_budget_s=runtime.sequence.TASK_BUDGETS[task],
            wall_budget_s=1500. if task=='H6' else 900.)
        # Independent, fixed-seed start-pose draws, not repeated deterministic
        # replays. Paired methods share each draw and retain a common goal.
        for repeat in range(5):
            dx,dy = rng.uniform(-.010,.010,2)
            dyaw = float(rng.uniform(-math.radians(.5),math.radians(.5)))
            x,y,yaw = geometry['start_xyyaw']
            start = [float(x+math.cos(yaw)*dx-math.sin(yaw)*dy),
                     float(y+math.sin(yaw)*dx+math.cos(yaw)*dy),float(yaw+dyaw)]
            shift = (repeat+number-1)%4
            order = METHODS[shift:]+METHODS[:shift]
            for order_index,method in enumerate(order):
                rows.append(dict(cell_id=f'{task}_{method}_s{repeat+1:02d}', task=task,
                    method=method, paired_condition_id=f'{task}_s{repeat+1:02d}',
                    start_xyyaw=start, body_pose_perturbation=[float(dx),float(dy),dyaw],
                    initial_active_steering_rule='unchanged nominal source seed; no steering perturbation claimed',
                    order_within_condition=order_index, goal_xyyaw=geometry['goal_xyyaw']))
    return dict(version='REVIEW_COMMON30_PAIRED_START_POSE_N5_v1',
        status='PREDECLARED_NOT_COMPLETED_RESULTS', authority=copy.deepcopy(runtime.APPROVAL),
        main_cells=rows, main_planned_cell_count=120, repeats_per_task_method=5,
        random_generator='numpy.default_rng(20261121)',
        perturbation_distribution=dict(body_x_m=[-.010,.010],body_y_m=[-.010,.010],
            heading_deg=[-.5,.5], independent_uniform_draws=True),
        common_speed_policy=asdict(planning.CommonSpeedPolicy()),
        same_rule_does_not_guarantee_identical_observed_speed=True,
        common_spatial_search_clock_m_s=.050, planning_wall_budget_s=180.,
        lattice_expansion_limit=60000, P_lookahead_edges_all_tasks=8,
        P_beam_width=64, P_local_iterations=24, tasks=tasks,
        ablation=dict(tasks=['H2','H3'],paired_with_same_five_main_conditions=True,
            additional_planned_cells=10,variant='P_no_entry_state_history',
            sole_predictive_change='zero front/rear steering and u/v/r at each primitive entry; preserve pose',
            same_coefficients_constraints_speed_rule_and_physical_controller=True),
        full_envelope_comparison=dict(scope='OFFLINE_PLANNING_AND_INDEPENDENT_FULL_ENVELOPE_AUDIT',
            tasks=['H2','H3'],same_five_conditions=True,
            variant='point-reference collision planning; independently rescore the complete envelope',
            known_colliding_paths_are_NOT_executed=True),
        force_load_review=dict(scope='OFFLINE_ACCEPTED_AND_REJECTED_CANDIDATE_ACTIVITY',
            no_motor_or_patient_safety_guarantee=True,
            reduce_claims_if_no_demonstrated_path_selection_effect=True),
        statistics=dict(unit='one separately constructed physical trial, not telemetry frames',
            include_all_planning_failures_and_execution_interruptions=True,
            completed_case_metrics_only_with_explicit_valid_n=True,
            retain_partial_metrics_separately=True,standard_deviation_ddof=1,
            report_pairwise_complete_subset_size=True,
            small_sample_descriptive_NOT_confirmatory_power=True),
        engineering_pair_excluded_from_main_repetitions=str(runtime.BASE),
        no_blind_physical_retry=True, no_replacement_until_success=True,
        maximum_new_drives_130_not_a_guaranteed_remaining_drive_count=True,
        main_live_dispatch_requires_finished_pilot_and_frozen_implementation=True,
        manuscript_edits=False, additional_real_robot_motion=False,
        original_evidence_inputs_sha256=inputs)


def export_protocol():
    value = runtime.evidence.json_value(definition())
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Predeclared protocol cannot be replaced after observing outcomes')
        return PROTOCOL
    runtime.evidence.write_exclusive(PROTOCOL,value)
    runtime.lc.freeze_workflow_sources(OUTPUT/'design_sources.zip',
        {str(Path(__file__).resolve()):runtime.sha(Path(__file__)),
         str(Path(planning.__file__).resolve()):runtime.sha(Path(planning.__file__))})
    return PROTOCOL


async def run_new_pilot_cell(method, *, prior_result, authorized=False):
    """Next fresh cell with the verified class-closure connector."""
    if authorized is not True:
        raise ValueError('Current explicit additional-experiment authority required')
    if method != planning.P:
        raise ValueError('This first continuation is the still-unstarted H2 P engineering cell')
    folder = runtime.BASE/'P_CS'
    if folder.exists():
        raise FileExistsError('Consumed pilot; no automatic retry')
    prior = runtime.read(prior_result)
    if prior.get('verdict') not in ('PASS','HOLD','FAIL') or not prior['checks'].get('original_gains_targets_restored'):
        raise ValueError('Preserved restored prior boundary required before a different method')
    original_plan = planning.OUTPUT/(method+'.json')
    model_path = planning.OUTPUT/'executor_model_screen'/method/'model_screen.json'
    model = runtime.read(model_path)
    if not model['both_nominal_resolutions_pass']:
        raise ValueError('Own model PASS required')
    runtime.retained.runtime.task.verify_pins(model['input_sha256'])
    runtime.retained.runtime.task.verify_pins(model['output_sha256'])
    checked = runtime.read(planning.OUTPUT/'offline_audits'/(method+'.json'))
    if checked['verdict'] != 'PASS' or checked['plan_sha256'] != runtime.sha(original_plan):
        raise ValueError('Own independent plan audit required')
    geometry_path = Path(runtime.read(original_plan)['geometry_manifest'])
    geometry = runtime.read(geometry_path)
    grid_path = planning.OUTPUT/'planning_occupancy.npz'
    folder.mkdir(parents=True,exist_ok=False)
    runtime.evidence.write_exclusive(folder/'cell_definition.json',dict(method=method,
        authorization=runtime.APPROVAL, plan=str(original_plan),model_screen=str(model_path),
        source_sha256={str(Path(__file__).resolve()):runtime.sha(Path(__file__)),
                       str(Path(binding.__file__).resolve()):runtime.sha(Path(binding.__file__))},
        use_for_final_statistics=False,independent_final_repetition=False,
        no_blind_retry=True,maximum_static_trials=1,maximum_route_trials=1))
    actor = '/World/patient_transport_controlled_h2_p_cs_pilot_r01'
    context = runtime.context_for(folder,actor,geometry)
    sdk = runtime.retained.runtime.live_sdk()
    try:
        await runtime.prepare_source(context,sdk,geometry,prior_result=prior_result)
        path,packet = runtime.static_packet(context,geometry_path,grid_path)
        with np.load(grid_path) as grid:
            args=dict(occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),
                      resolution=float(grid['resolution_m']))
        host = runtime.ControlledStaticHost(sdk,packet,path,context=context,geometry=geometry,
                                           **args,authorized=True)
        result = await runtime.lc.run_held_static_once(host,
            runtime.lc.HeldStaticController(geometry['start_xyyaw']),authorized=True)
        if result['verdict'] != 'PASS':
            return dict(static_verdict=result['verdict'],route_verdict='NOT_RUN_STATIC_NOT_PASS')
        unused,packet,plan = runtime.route_packet(context,original_plan,geometry_path,grid_path)
        packet = copy.deepcopy(packet)
        packet['input_sha256'].update({str(Path(p).resolve()):runtime.sha(p)
                                       for p in (__file__,binding.__file__)})
        path = folder/'route/current_scope_sdk_binding.json'
        runtime.evidence.write_exclusive(path,packet)
        runtime.lc.freeze_workflow_sources(folder/'route/current_scope_sources.zip',packet['input_sha256'])
        port = binding.ControlledRouteHost(sdk,packet,path,context=context,plan=plan,
                                          geometry=geometry,**args,authorized=True)
        controller = runtime.sequence.unified_geometry_lifecycle_controller(plan,geometry,
            profile=plan['geometry_profile'],simulation_budget_s=180.)
        result = await runtime.lc.run_lifecycle_once(port,controller,authorized=True,
            sdk_bridge_class=runtime.sequence.RetainFinalTargetsBridge)
        return dict(static_verdict='PASS',route_verdict=result['verdict'],result=str(port.result))
    except BaseException as failure:
        runtime.evidence.write_exclusive(folder/'workflow_error.json',dict(
            error=type(failure).__name__+': '+str(failure),no_automatic_retry=True))
        raise


if __name__ == '__main__':
    print(export_protocol())
