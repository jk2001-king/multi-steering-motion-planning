"""Read-only gate for all retained final-forward MODEL design variants.

No motion authority. Require actual SECOND-reverse raw PASS, original goal,
all 12 retained joints and all explored cases, including every prior HOLD.
"""

import json
from pathlib import Path

import numpy as np

import hospital_h3b_post_second_reverse as data
from hospital_h3b_timed_course_speed_profile import VERSION as DESIGN_VERSION, PLANNED_SPEED_CAP_M_S


VARIANTS = ('shared_timing_feedforward', 'existing_course_with_shared_timing',
    'inherited_complete_forward_executor', 'existing_course_shared_timing_planned_cap30')
inspect_second_reverse = data.inspect_second_reverse
validate_live = data.reverse.validate_live
sha, verify_files = data.sha, data.verify_files


def load_review(directory, info=None):
    directory = Path(directory).resolve()
    report = json.loads((directory/'final_forward_review.json').read_bytes())
    data.verify_sources(report);verify_files(report['input_sha256'])
    if info is None:info = inspect_second_reverse()[0]
    cases = report.get('cases', [])
    expected = {(v, d, r, s) for v in VARIANTS for d in data.STEPS for r in data.RESIDUALS for s in data.SCALES}
    actual = {(c['design_variant'], c['dt_s'], c['residual_case'], c['SYNTHETIC_body_target_scale']) for c in cases}
    if (report.get('version') != data.VERSION or report.get('PhysX_executed') is not False
            or report.get('automatic_motion_authorized') is not False or report.get('CICS_input_used') is not False
            or report.get('second_reverse_result') != info['second_reverse_result']
            or report.get('second_reverse_result_sha256') != info['second_reverse_result_sha256']
            or report.get('actual_start_world_xyyaw') != info['initial_world_xyyaw']
            or report.get('retained_all_joint_positions_rad') != info['retained_all_joint_positions_rad']
            or report.get('original_mission_goal_xyyaw') != info['mission_goal_xyyaw']
            or report.get('strict_map_digest') != info['strict_map_digest']
            or report.get('final_leg') != info['final_leg']
            or report.get('selected_design_variant') != VARIANTS[-1]
            or report.get('selected_design_version') != DESIGN_VERSION
            or report.get('planned_speed_cap_m_s') != PLANNED_SPEED_CAP_M_S
            or report.get('physical_speed_cap_m_s') != .050
            or report.get('synthetic_cases_used_for_design_and_selection_NOT_independent_validation') is not True
            or report.get('old_actual_PASS_FAIL_and_all_model_HOLDs_preserved') is not True
            or len(cases) != 32 or actual != expected):
        raise ValueError('Exact actual second-reverse PASS/frozen final goal and complete 32-case exploratory review required')
    verify_files({c['trace_file']: c['trace_sha256'] for c in cases})
    selected = [c for c in cases if c['design_variant'] == VARIANTS[-1]]
    gates = ('actual_second_reverse_swept_strict_10mm_reproduced_clear',
        'actual_start_strict_10mm_clear', 'frozen_final_forward_strict_10mm_clear')
    passed = all(all(c['checks'].values()) for c in selected) and all(report.get(k) is True for k in gates)
    if report.get('all_selected_final_forward_design_cases_pass') is not passed:
        raise ValueError('Selected model verdict must retain every failed design case')
    if any(c.get('planned_speed_cap_m_s') != PLANNED_SPEED_CAP_M_S
            or c.get('tracking_gains') != {'yaw_error_gain_1_m_rad': 1.5, 'cross_error_gain_1_m2': 2.}
            or c.get('same_existing_course_step_and_gains_no_override') is not True for c in selected):
        raise ValueError('Selected design must retain the disclosed 30mm/s profile and existing 1.5/2 gains')
    initial = data.reverse.relative_pose(info['initial_world_xyyaw'], info['mission_start_xyyaw'])
    if any(c.get('model_parameters') != data.model_bounds()
            or c.get('fixed_next_cusp_local_xyyaw') != info['final_leg']['exact_mission_goal_local_xyyaw']
            or not np.allclose(c['initial_actual_local_xyyaw'], initial, rtol=0., atol=1e-12)
            or c.get('PhysX_executed') is not False or c.get('automatic_motion_authorized') is not False for c in cases):
        raise ValueError('Every model case must retain actual start, original goal, same plant and MODEL-only scope')
    return report
