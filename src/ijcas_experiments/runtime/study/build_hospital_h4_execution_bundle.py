#!/usr/bin/env python3
"""Freeze all FOUR native H4 plans in a common, non-admitted executor format.

No new path, smoothing, response fit, parameter retuning or live state is used.
Both shared spatial-generation stages are charged to every method. Original
native costs remain incomparable across fidelity models and are not runtime.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_cohort_preparation_v4 as cohort
import hospital_h4_path_execution as execution
from hospital_geometry_tools import pose_collision


OUTPUT = cohort.GATE/'hospital_h4_execution_bundle/bundle_r01'
TARGET = OUTPUT/'execution_bundle.json'
PREVIEW = OUTPUT/'four_method_native_references.png'
KINEMATIC_POOL = cohort.GATE/'hospital_h4_shared_candidate_screen/screen_r01/shared_candidate_screen.json'
VERSION = 'H4_four_native_plans_same_original_v3_executor_schema_bundle_v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def plans():
    data = json.loads(cohort.MODELS.read_bytes())
    setup = json.loads(cohort.SETUP.read_bytes())
    complete = json.loads(cohort.MAP_REVIEW.read_bytes())
    previous = json.loads(KINEMATIC_POOL.read_bytes())
    preflight = json.loads(cohort.PREFLIGHT.read_bytes())
    for report in (data, setup, complete, previous, preflight):
        cohort.verify(report['input_sha256'])
    cohort.verify(setup['output_sha256'])
    if (tuple(r['method'] for r in data['methods']) != execution.METHODS
            or len(data['candidate_pool']) != 26
            or not data['shared_union_pool_fixed_before_any_final_method_selection']
            or not data['every_method_must_pass_its_OWN_native_model_endpoint_and_envelope']):
        raise ValueError('Exact original shared 26-candidate pool and all four native selectors required')
    with np.load(cohort.MAP) as grid:
        occupied, origin, resolution = grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m'])
        digest = hashlib.sha256(occupied.tobytes()).hexdigest()
    if digest != setup['candidate_occupancy_digest']:
        raise ValueError('Complete authored map differs from the original candidate raster')
    pool_sha = hashlib.sha256(canonical(data['candidate_pool'])).hexdigest()
    # The original screen stored generator duration separately from failed P
    # refinement. Do not charge only the later full-model seed addon.
    first_duration = previous['shared_spatial_generation_time_s']
    total_generation = first_duration+data['shared_generation_wall_time_s']
    sources = [Path(__file__), Path(execution.__file__),
        *[_ijcas_path(cohort.ROOT, 'studies/IJCAS/scripts', name) for name in execution.COMMON_SOURCE_FILES]]
    pins = {**data['input_sha256'], **setup['input_sha256'], **setup['output_sha256'], **preflight['input_sha256'],
        **complete['input_sha256'], **previous['input_sha256'],
        **{str(p): cohort.completion.sha(p) for p in
            (*sources, cohort.MODELS, cohort.SETUP, cohort.MAP_REVIEW, cohort.MAP, KINEMATIC_POOL,
                cohort.PREFLIGHT, cohort.support.TARGET, cohort.support.LIVE)}}
    design = cohort.support.load_design(); covers = cohort.support_covers(design)
    result = []
    for native in data['methods']:
        if not native['plan_success']:
            raise ValueError('No successful native model plan for '+native['method'])
        plan = copy.deepcopy(native)
        plan.update(scenario=execution.SCENARIO, generator_selector_version=data['version'],
            selected_native_source=str(cohort.MODELS), selected_native_source_sha256=cohort.completion.sha(cohort.MODELS),
            start_world_xyyaw=setup['start_xyyaw'], goal_world_xyyaw=setup['goal_xyyaw'], goal_local_xyyaw=setup['local_goal_xyyaw'],
            common_max_command_speed_m_s=.05, common_max_acceleration_m_s2=.02,
            full_envelope_clearance_m=.010, footprint=execution.FOOTPRINT,
            planning_occupancy_digest=digest, geometry_manifest=str(cohort.SETUP),
            geometry_manifest_sha256=cohort.completion.sha(cohort.SETUP),
            shared_candidate_pool_sha256=pool_sha, shared_candidate_count=26,
            shared_generation_kinematic_stage_wall_s=first_duration,
            shared_generation_full_model_addon_wall_s=data['shared_generation_wall_time_s'],
            total_shared_generation_wall_s_charged_to_this_method=total_generation,
            total_generation_plus_native_selection_wall_s=total_generation+native['selection_wall_time_s'],
            original_failed_P_refinement_not_hidden_or_used_as_successful_search_time=True,
            native_cost_s_is_NOT_common_executed_duration=True,
            full_state_response_model_independently_validated=False,
            native_MODEL_success_is_NOT_actual_support_or_drive_admission=True,
            original_H1_H2_planner_algorithm_unchanged_claimed=False,
            current_H3_whole_MODEL_2_of_8_unchanged=True,
            source_authored_caster_seed_policy='same original reference; no assigned or success-selected caster angle',
            shared_executor_version=execution.EXECUTION_VERSION, shared_controller_config=execution.CONTROLLER_CONFIG,
            shared_command_limits=execution.COMMAND_LIMITS, shared_guards=execution.GUARDS, shared_acceptance=execution.ACCEPTANCE,
            B0_fixed_means_PLANNED_modes_only_common_terminal_stabilizer_can_use_other_modes=True,
            terminal_stabilization_must_be_logged_separately_from_planning_decisions=True,
            original_spatial_actions_and_states_unchanged=True, actual_motion_authorized=False,
            CICS_input_used=False, sensor_or_Nav2_execution=False, use_for_final_statistics=False,
            input_sha256=pins)
        execution.validate_plan(plan)
        if plan['selected_actions'] != native['selected_actions'] or plan['selected_states'] != native['selected_states']:
            raise ValueError('The execution bundle must not modify/smooth/relabel a native trajectory')
        reference = execution.PathReference(plan)
        support_samples, maximum_missing = 0, 0.
        for a, b in zip(reference.poses, reference.poses[1:]):
            angle = execution.wrap(b[2]-a[2])
            count = max(1, math.ceil(math.dist(a[:2], b[:2])/.005), math.ceil(abs(angle)/math.radians(.25)))
            for i in range(count+1):
                t = i/count; pose = [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*angle]
                if pose_collision(occupied, pose, origin, resolution, clearance=.010):
                    raise ValueError('Native H4 execution-reference interpolation hits the authored map')
                maximum_missing = max(maximum_missing, cohort.support_area_missing(pose, covers)); support_samples += 1
        if maximum_missing > 1e-9:
            raise ValueError('A TRUE full-body native reference extends outside the fixed support')
        nominal = sum(.050/action['target_speed_m_s'] for action in plan['selected_actions'])
        plan.update(reference_center_arc_m=reference.arc[-1], nominal_longitudinal_schedule_duration_s=nominal,
            nominal_schedule_NOT_controller_feedback_or_actual_duration=True,
            nominal_plus_90s_terminal_plus_5s_hold_s=nominal+execution.CONTROLLER_CONFIG['terminal_timeout_s']+5.,
            inherited_180s_budget_does_NOT_guarantee_full_terminal_allowance=nominal+execution.CONTROLLER_CONFIG['terminal_timeout_s']+5. > execution.GUARDS['maximum_simulation_duration_s'],
            sampled_reference_geometry_and_TRUE_support=dict(samples=support_samples, maximum_uncovered_area_m2=maximum_missing,
                clearance_m=.010, maximum_translation_spacing_m=.005, maximum_heading_spacing_deg=.25,
                continuous_collision_or_cooked_contact_certificate=False))
        result.append(plan)
    return result, pins, setup, occupied, origin, resolution


def draw(values, occupied, origin, resolution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8, 9), layout='constrained')
    ax.imshow(occupied, origin='lower', cmap='Greys', extent=[origin[0], origin[0]+occupied.shape[1]*resolution,
        origin[1], origin[1]+occupied.shape[0]*resolution], vmin=0, vmax=1)
    for plan, color, style in zip(values, ('#0088bb', '#f58518', '#7f3c8d', '#198754'), ('-', '-', '--', '-')):
        poses = np.asarray([plan['start_world_xyyaw']]+[s['world_xyyaw'] for s in plan['selected_states']])
        ax.plot(poses[:, 0], poses[:, 1], color=color, linestyle=style, label=plan['method'], linewidth=1.6)
    for key, color, label in (('start_world_xyyaw', '#198754', 'Fixed start'), ('goal_world_xyyaw', '#d65f00', 'Unchanged goal')):
        pose = values[0][key]
        ax.add_patch(Polygon(cohort.full_footprint(pose), facecolor=color, alpha=.2, edgecolor=color))
        ax.annotate(label, pose[:2], xytext=(8, 12), textcoords='offset points', fontsize=9, color=color)
    ax.set(aspect='equal', xlim=(15, 22), ylim=(23, 33), xlabel='World X (m)', ylabel='World Y (m)',
        title='H4 frozen native MODEL references\nSame v3 executor binding; no physical drive admission')
    ax.legend(loc='upper right', fontsize=7)
    fig.savefig(PREVIEW, dpi=160); plt.close(fig)


def main():
    if TARGET.exists():
        old = json.loads(TARGET.read_bytes()); cohort.verify(old['input_sha256']); cohort.verify(old['output_sha256'])
        for path in old['plans']:
            execution.validate_plan(json.loads(Path(path).read_bytes()))
        print('Existing immutable H4 execution bundle verified:', TARGET)
        return
    values, pins, setup, occupied, origin, resolution = plans()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    paths = []
    for plan in values:
        path = OUTPUT/(plan['method']+'.json'); paths.append(path)
        with path.open('x') as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
    draw(values, occupied, origin, resolution)
    report = dict(version=VERSION, status='FOUR_NATIVE_REFERENCES_FROZEN_COMMON_EXECUTOR_MODEL_SCREEN_PENDING',
        plans=list(map(str, paths)), preview=str(PREVIEW), unchanged_original_task=True,
        all_four_methods_share_original_v3_tracking_terminal_and_acceptance=True,
        shared_candidate_pool_identical_for_all_selectors=True, generators_and_native_selectors_are_disclosed_NEW_H4_versions=True,
        existing_180s_simulation_900s_wall_resource_guards_not_changed=True,
        resource_allowance_warning='A 7.6m reference plus the full 90s terminal allowance does not fit the inherited 180s cap; no budget was silently raised',
        current_H3_whole_MODEL_PASS_count=2, current_H3_whole_MODEL_case_count=8,
        independent_fresh_frozen_whole_H3_repeats=0, current_live_robot_or_timeline_accessed=False,
        full_state_model_or_cooked_collision_validated=False, actual_motion_authorized=False,
        CICS_runtime_integration=False, real_robot_required=False, use_for_final_statistics=False,
        input_sha256=pins, output_sha256={str(p): cohort.completion.sha(p) for p in (*paths, PREVIEW)})
    with TARGET.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 FOUR NATIVE REFERENCES / UNCHANGED COMMON v3 EXECUTOR BUNDLE ===')
    for p in values:
        print(p['method'], p['selected_candidate_id'], 'length m:', p['path_length_m'],
            'nominal schedule s:', p['nominal_longitudinal_schedule_duration_s'])
    print('saved:', TARGET)
    print('No live scene or H3 model change; common-executor MODEL screen and actual static remain pending.')


if __name__ == '__main__':
    main()
