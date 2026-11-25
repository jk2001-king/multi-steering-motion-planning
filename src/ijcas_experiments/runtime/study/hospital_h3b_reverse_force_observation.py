"""Passive force-observer contract for ONE same-condition rear reverse.

Source locations and result cohort change; actuator/control functions, frozen
reference, parameters and safety gates do not. Solver generalized effort is
not an isolated actuator-torque or saturation measurement.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import ast
import json
from pathlib import Path

from hospital_h3b_reverse_force_prefix import WORKSPACE, GATE, COHORT, REFERENCE, review_reference
from hospital_h3_flush_support import sha, verify_files
from hospital_h3b_rolling_observation import telemetry_snapshot, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3b_rolling_observation import coverage as original_coverage


OUTPUT = COHORT/'instrumentation'


def function_contract():
    entries = _ijcas_path(WORKSPACE, 'isaac/scripts')
    pairs = [
        ('isaac_run_ijcas_hospital_h3b_rear_prefix_pilot.py', 'isaac_run_ijcas_hospital_h3b_reverse_force_prefix.py',
         ('physics_step', 'contacts', 'angles', 'verify_gains', 'set_hold', 'command', 'measure', 'environmental_guard', 'sample')),
        ('isaac_run_ijcas_hospital_h3b_rear_reverse_pilot.py', 'isaac_run_ijcas_hospital_h3b_reverse_force_pilot.py',
         ('numeric', 'physics_step', 'contacts', 'angles', 'verify_gains', 'measure', 'set_hold', 'command', 'sample')),
    ]
    matched = {}; sources = {}
    for old_name, new_name, names in pairs:
        paths = [entries/old_name, entries/new_name]
        functions = [{n.name: n for n in ast.walk(ast.parse(p.read_text()))
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in paths]
        for name in names:
            if ast.dump(functions[0][name], include_attributes=False) != ast.dump(functions[1][name], include_attributes=False):
                raise ValueError('Observer cohort changed a control/guard function: '+new_name+' '+name)
        matched[new_name] = list(names)
        sources.update({str(p): sha(p) for p in paths})
    return dict(control_and_guard_function_AST_equal=True, matched_functions=matched, input_sha256=sources)


def review(info):
    inputs = review_reference(); contract = function_contract(); baseline = json.loads(REFERENCE.read_bytes())
    first = info['first_record']; prep = first['physics_preparation_snapshot']
    for key in ('environment_condition', 'source_stage_sha256', 'robot_reference_sha256', 'start_xyyaw',
                'component_end_xyyaw', 'simulation_mass_kg', 'live_support_readback', 'support_floor', 'disabled_legacy_floor_colliders'):
        if prep[key] != baseline['physics_preparation_snapshot'][key]:
            raise ValueError('Unchanged-condition preparation differs: '+key)
    if (first['final_world_xyyaw'] != baseline['initial_saved_world_xyyaw']
            or info['retained_all_joint_positions_rad'] != baseline['expected_retained_joint_positions_rad']
            or info['suffix_candidate_sha256'] != baseline['suffix_candidate_sha256']
            or info['mission_start_xyyaw'] != baseline['original_mission_start_xyyaw']
            or info['mission_goal_xyyaw'] != baseline['mission_goal_world_xyyaw_UNCHANGED']):
        raise ValueError('Same actual start/all joints/caster history/reference/task required; never reset to force equivalence')
    return dict(reference_trial=str(REFERENCE), reference_trial_sha256=sha(REFERENCE),
                unchanged_condition_verified=True, original_verdict_preserved='FAIL',
                exact_same_saved_start_and_12_joint_positions=True,
                controller_contract=contract, input_sha256={**inputs, **contract['input_sha256'], **info['input_sha256']})


def coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors):
    result = original_coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors)
    if result['status'] == 'RECORDED_FOR_CAUSAL_REVIEW_NOT_A_ROLLING_OR_TRAVERSAL_PASS':
        result['status'] = 'RECORDED_FOR_REVERSE_FORCE_REVIEW_NOT_A_MOTION_PASS'
    result.update(original_motion_verdict_is_authoritative=True,
                  unique_caster_friction_or_drive_cause_isolated=False,
                  projected_joint_effort_is_NOT_isolated_motor_torque=True)
    return result
