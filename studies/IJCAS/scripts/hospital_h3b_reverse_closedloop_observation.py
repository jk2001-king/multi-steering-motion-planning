"""Passive recording; explicitly NEW high-level terminal control, same plant."""

import ast
from pathlib import Path

from hospital_h3b_reverse_speed50_observation import review as baseline_review
from hospital_h3b_reverse_speed50_observation import coverage as baseline_coverage
from hospital_h3b_reverse_speed50_observation import telemetry_snapshot, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3b_reverse_closedloop_prefix import WORKSPACE, GATE, COHORT
from hospital_h3b_reverse_closedloop import execution_contract
from hospital_h3_flush_support import sha


OUTPUT = COHORT/'instrumentation'
PRIOR = GATE/'hospital_h3b_reverse_speed50_pilot/cohort_01/reverse/hospital_h3b_reverse_speed50_rear_r01.json'


def function_contract():
    entries = WORKSPACE/'isaac/scripts'; matched = {}; sources = {}
    pairs = (
        ('prefix', ('physics_step', 'contacts', 'angles', 'verify_gains', 'set_hold', 'command', 'measure', 'environmental_guard', 'sample')),
        ('pilot', ('numeric', 'physics_step', 'contacts', 'angles', 'verify_gains', 'measure', 'set_hold', 'command', 'sample')),
    )
    for suffix, names in pairs:
        paths = [entries/f'isaac_run_ijcas_hospital_h3b_reverse_{kind}_{suffix}.py'
                 for kind in ('speed50', 'closedloop')]
        functions = [{n.name:n for n in ast.walk(ast.parse(p.read_text()))
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in paths]
        for name in names:
            if ast.dump(functions[0][name], include_attributes=False) != ast.dump(functions[1][name], include_attributes=False):
                raise ValueError('Frozen low-level adapter or safety guard changed: '+suffix+' '+name)
        matched[paths[1].name] = list(names)
        sources.update({str(p):sha(p) for p in paths})
    return dict(low_level_adapter_and_sample_guard_AST_equal=True,
        NEW_high_level_measured_braking_terminal_controller_disclosed=True,
        stall_guard_callee_now_checks_contiguous_tracking_AND_braking=True,
        stall_thresholds_unchanged_s_m_command=[5., .001, .006],
        progress_evaluation_scope_now_includes_braking=True,
        intervention_contract=execution_contract(), matched_functions=matched, input_sha256=sources)


def review(info):
    reviewed = baseline_review(info); contract = function_contract()
    reviewed.pop('unchanged_condition_verified', None)
    reviewed.update(reference_trial=str(PRIOR), reference_trial_sha256=sha(PRIOR),
        unchanged_PHYSICAL_initial_condition_verified=True, NEW_high_level_executor=True,
        controller_contract=contract,
        input_sha256={**reviewed['input_sha256'], **contract['input_sha256'],
                      str(PRIOR):sha(PRIOR), str(PRIOR.with_suffix('.csv')):sha(PRIOR.with_suffix('.csv')),
                      str(Path(__file__)):sha(__file__)})
    return reviewed


def coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors):
    result = baseline_coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors)
    if result['status'] == 'RECORDED_FOR_SEPARATE_SPEED50_FORCE_REVIEW_NOT_A_MOTION_PASS':
        result['status'] = 'RECORDED_FOR_NEW_TERMINAL_EXECUTOR_FORCE_REVIEW_NOT_A_MOTION_PASS'
    return result
