"""Passive telemetry for ONE separate 50 mm/s reverse-cap intervention."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import ast
from pathlib import Path

from hospital_h3b_reverse_force_observation import review as baseline_review
from hospital_h3b_reverse_force_observation import coverage as baseline_coverage
from hospital_h3b_reverse_force_observation import telemetry_snapshot, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3b_reverse_speed50_prefix import WORKSPACE, COHORT
from hospital_h3b_reverse_speed50 import verify_speed_only_change
from hospital_h3_flush_support import sha


OUTPUT = COHORT/'instrumentation'


def function_contract():
    verify_speed_only_change(); entries = _ijcas_path(WORKSPACE, 'isaac/scripts'); matched = {}; sources = {}
    pairs = (
        ('prefix', ('physics_step', 'contacts', 'angles', 'verify_gains', 'set_hold', 'command', 'measure', 'environmental_guard', 'sample')),
        ('pilot', ('numeric', 'physics_step', 'contacts', 'angles', 'verify_gains', 'measure', 'set_hold', 'command', 'sample')),
    )
    for suffix, names in pairs:
        paths = [entries/f'isaac_run_ijcas_hospital_h3b_reverse_{kind}_{suffix}.py'
                 for kind in ('force', 'speed50')]
        functions = [{n.name:n for n in ast.walk(ast.parse(p.read_text()))
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in paths]
        changes = 0
        for name in names:
            old, new = functions[0][name], functions[1][name]
            if suffix == 'pilot' and name == 'command':
                for node in ast.walk(new):
                    if isinstance(node, ast.Constant) and type(node.value) is float and node.value == .050:
                        node.value = .010; changes += 1
            if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
                raise ValueError('Only the explicit reverse command cap may differ: '+suffix+' '+name)
        if changes != (1 if suffix == 'pilot' else 0):
            raise ValueError('Unexpected speed-cap guard changes')
        matched[paths[1].name] = list(names)
        sources.update({str(p):sha(p) for p in paths})
    return dict(control_and_guard_function_AST_equal_EXCEPT_reverse_command_cap=True,
                intervention_contract=verify_speed_only_change(), matched_functions=matched, input_sha256=sources)


def review(info):
    baseline = baseline_review(info); contract = function_contract()
    baseline.update(controller_contract=contract,
                    separate_speed_cap_intervention=True,
                    input_sha256={**baseline['input_sha256'], **contract['input_sha256'],
                                  str(Path(__file__)):sha(__file__)})
    return baseline


def coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors):
    result = baseline_coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors)
    if result['status'] == 'RECORDED_FOR_REVERSE_FORCE_REVIEW_NOT_A_MOTION_PASS':
        result['status'] = 'RECORDED_FOR_SEPARATE_SPEED50_FORCE_REVIEW_NOT_A_MOTION_PASS'
    return result
