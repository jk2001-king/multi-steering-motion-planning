"""Observe the SAME next-forward executor after cohort_03's actual reverse.

No body/steering fit or planner/control change. Original model screens are
reused ONLY after complete prefix AND reverse CSV/endpoint/12-joint equality.
The measured scene is never reset to obtain equality. Whole-sequence HOLDs
remain blocking; this admits one already-PASS component observation only.
"""

import ast
import json
from pathlib import Path

import hospital_h3b_next_forward as prior
from hospital_h3b_next_forward import *
import hospital_h3b_reverse_response_observation_v3 as reverse_observation
import hospital_h3b_prefix_response_observation_v3 as prefix
from hospital_h3b_cp30_runtime_v2 import wheel_body_snapshot, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3b_cp30_runtime_v2 import coverage as wheel_coverage
from hospital_runtime_interface import audit_exports, probe_contact_classifier


OBSERVATION_VERSION = 'H3B_same_next_forward_six_wheel_response_observation_v1'
ROLE = 'SAME_NEXT_FORWARD_COMPONENT_PASSIVE_RESPONSE_OBSERVATION_NOT_VALIDATION'
BASE, COHORT = prefix.BASE, prefix.COHORT
REVERSE = reverse_observation.RESULT
REFERENCE = prior.OUTPUT / 'hospital_h3b_next_forward_r01.json'
OUTPUT, OBSERVER_OUTPUT = COHORT / 'next_forward', COHORT / 'next_forward_instrumentation'
RESULT = OUTPUT / 'hospital_h3b_next_response_r01.json'
LIVE = COHORT / 'next_forward_handoff/live_handoff.json'
PROTOCOL_PATH = BASE / 'next_forward_protocol_r01/next_response_design.json'
PREFIX_ANALYSIS = BASE / 'analyses/analysis_r02/actual_prefix_response_review.json'
REVERSE_ANALYSIS = BASE / 'reverse_analyses/analysis_r02/actual_reverse_response_review.json'
ENTRIES = WORKSPACE / 'isaac/scripts'
PRIMARY = ENTRIES / 'isaac_run_ijcas_hospital_h3b_next_response_primary.py'
OBSERVER = ENTRIES / 'isaac_run_ijcas_hospital_h3b_next_response_instrumented.py'
HANDOFF = ENTRIES / 'isaac_review_ijcas_hospital_h3b_next_response_handoff.py'
telemetry_snapshot = wheel_body_snapshot
validate_tensor = reverse_observation.validate_tensor
PRIMARY_CALLBACKS = ('numeric', 'physics_step', 'contacts', 'angles', 'verify_gains',
                     'measure', 'set_hold', 'command', 'sample')


def exact_history(current, old):
    """Fail closed; never choose/reset an endpoint to recover equivalence."""
    record, reference = (json.loads(p.read_bytes()) for p in (current, old))
    verify_sources(record); verify_files(record['frozen_input_sha256'])
    if (record.get('verdict') != 'PASS' or not record.get('checks')
            or not all(record['checks'].values()) or record.get('original_motor_gains_restored') is not True
            or record.get('PhysX_executed') is not True):
        raise ValueError('Authoritative actual PASS and restored gains required: ' + str(current))
    if current.with_suffix('.csv').read_bytes() != old.with_suffix('.csv').read_bytes():
        raise ValueError('Complete measured input/state history differs; do NOT reset or waive equality')
    rows = read_trace(current.with_suffix('.csv'))
    if len(rows) != record['sample_count'] or rows[-1] != record['last_recorded_state']:
        raise ValueError('Complete actual CSV/final twelve-joint snapshot required')
    for key in ('last_recorded_state', 'final_world_xyyaw', 'target_world_xyyaw',
                'mission_goal_world_xyyaw_UNCHANGED', 'physics_preparation_snapshot', 'strict_map_digest'):
        # Prefix uses its original spelling and has a fresh preparation ID;
        # its full physical equivalence is also enforced by reverse.load_profile.
        if key in record and key in reference and key != 'physics_preparation_snapshot' and record[key] != reference[key]:
            raise ValueError('Original actual model seed changed: ' + key)
    return record


def load_profile():
    info = prior.load_admitted_profile()
    actual_reverse = exact_history(REVERSE, Path(info['reverse_result']))
    actual_prefix = exact_history(prefix.RESULT, Path(info['first_result']))
    reverse_observation.load_reviewed_profile()  # original physical + full-prefix equivalence checks
    for path in (PREFIX_ANALYSIS, REVERSE_ANALYSIS):
        analysis = json.loads(path.read_bytes()); verify_sources(analysis); verify_files(analysis['input_sha256'])
        verify_files({c['trace_file']: c['trace_sha256'] for c in analysis['body_replays']})
        if (analysis['original_actual_verdict_unchanged'] != 'PASS' or analysis['telemetry_errors_unchanged']
                or analysis['whole_H3_drive_admission'] is not False
                or not analysis['coverage']['six_wheel_transforms_and_COM_and_origin_velocities_recorded']):
            raise ValueError('Complete actual-domain response audit required, not model validation')
    pins = {**info['input_sha256'], **actual_prefix['frozen_input_sha256'], **actual_reverse['frozen_input_sha256']}
    for path in (REVERSE, REVERSE.with_suffix('.csv'), prefix.RESULT, prefix.RESULT.with_suffix('.csv'),
                 PREFIX_ANALYSIS, REVERSE_ANALYSIS, Path(__file__)):
        pins[str(path)] = sha(path)
    if (actual_reverse['prior_prefix_result'] if 'prior_prefix_result' in actual_reverse else actual_reverse['first_result']) != str(prefix.RESULT):
        raise ValueError('Actual reverse must retain THIS actual prefix history')
    info.update(first_record=actual_prefix, preparation_record=actual_prefix,
        first_result=str(prefix.RESULT), first_result_sha256=sha(prefix.RESULT),
        reverse_record=actual_reverse, reverse_result=str(REVERSE), reverse_result_sha256=sha(REVERSE),
        expected_saved_last_state=actual_reverse['last_recorded_state'],
        initial_world_xyyaw=actual_reverse['final_world_xyyaw'],
        retained_all_joint_positions_rad=reverse_observation.joints_from_last(actual_reverse),
        exact_complete_prefix_AND_reverse_history_equivalence=True,
        original_four_next_forward_model_screens_reused_NOT_recomputed=True,
        whole_H3_drive_admission=False, new_body_or_steering_model_deployed=False, input_sha256=pins)
    return info


def coverage(*args):
    result = wheel_coverage(*args)
    result['status'] = ('ACTUAL_NEXT_FORWARD_SIX_WHEEL_RESPONSE_RECORDED_NOT_MODEL_VALIDATION'
        if result['status'] == 'CP30_PASSIVE_DATA_RECORDED_FOR_RESPONSE_REVIEW_NOT_MODEL_VALIDATION'
        else 'INCOMPLETE_ACTUAL_NEXT_FORWARD_RESPONSE_DATA')
    return result


def entry_texts():
    """Disclosed lineage/API/measurement-only edits; original async control stays identical."""
    original_primary = ENTRIES / 'isaac_run_ijcas_hospital_h3b_next_forward_pilot.py'
    original_observer = ENTRIES / 'isaac_run_ijcas_hospital_h3b_next_forward_instrumented.py'
    paths = dict(primary=original_primary, observer=original_observer)
    replacements = [
        (original_primary.name, PRIMARY.name), (original_observer.name, OBSERVER.name),
        ('hospital_h3b_next_forward_observation.py', Path(__file__).name),
        ('hospital_h3b_next_forward.py', Path(__file__).name),
        ('ijcas_hospital_h3b_next_forward_', 'ijcas_hospital_h3b_next_response_'),
        ('hospital_h3b_next_forward_r01', 'hospital_h3b_next_response_r01'),
        ('h3b_reverse_closedloop_pilot_finished_wait_for_review', 'h3b_reverse_response_pilot_finished_wait_for_review'),
        ('last_h3b_reverse_closedloop_pilot_gate', 'last_h3b_reverse_response_pilot_gate'),
        ('h3b_next_forward_pilot_finished_wait_for_review', 'h3b_next_response_pilot_finished_wait_for_review'),
        ('last_h3b_next_forward_pilot_gate', 'last_h3b_next_response_pilot_gate'),
    ]
    texts = {}
    for kind, path in paths.items():
        text = path.read_text()
        for old, new in replacements:
            text = text.replace(old, new)
        if kind == 'primary':
            needle = "h=loader(scripts/'hospital_h3b_next_response_observation.py');info=h['load_admitted_profile']()"
            text = text.replace(needle, "h=loader(scripts/'hospital_h3b_next_response_observation.py');h['validate_runtime_api']();info=h['load_admitted_profile']()")
            needle = '        robot=motor=drive_indices=old_kp=old_kd=old_limits=old_targets=None'
            text = text.replace(needle, "        report.update(same_original_next_forward_controller_AND_evaluator=True,\n"
                "            passive_six_wheel_response_observation_only=True,new_body_or_steering_model_deployed=False,\n"
                "            complete_prefix_AND_reverse_history_equivalence_verified=True,\n"
                "            existing_four_component_model_screens_reused_NOT_recomputed=True,\n"
                "            response_observation_protocol=str(h['PROTOCOL_PATH']),response_observation_protocol_sha256=h['sha'](h['PROTOCOL_PATH']))\n" + needle)
        else:
            text = text.replace('    import math\n', '    import math\n    import numpy as np\n')
            text = text.replace("rules = load(scripts/'hospital_h3b_next_response_observation.py'); info = rules['load_admitted_profile']()",
                "rules = load(scripts/'hospital_h3b_next_response_observation.py'); helper['validate_runtime_api'](); info = rules['load_admitted_profile']()")
            text = text.replace("    output = helper['OUTPUT']", "    output = helper['OBSERVER_OUTPUT']")
            # This is the exact already-used SIX-wheel retained tensor preflight.
            old_text = reverse_observation.OBSERVER.read_text()
            block = old_text.split('    def retained_tensor_check():', 1)[1].split('    patterns =', 1)[0]
            text = text.replace("    helper['telemetry_snapshot'](view, 0.)\n",
                "    helper['telemetry_snapshot'](view, 0.)\n    def retained_tensor_check():" + block)
            text = text.replace("    clock = {'time_s': 0., 'next_friction_s': 0.}",
                "    if retained_tensor_check()!=initial_tensor: raise RuntimeError('Retained body/joints/settings changed during observer SDK preflight; no motion launched')\n    clock = {'time_s': 0., 'next_friction_s': 0.}")
            text = text.replace("original_controller_verdict=original['verdict'], original_error=original.get('error'),",
                "original_controller_verdict=original['verdict'], original_controller_source_sha256=original['source_sha256'], original_error=original.get('error'),")
            text = text.replace("            reference_trial=reviewed['reference_trial'],", "            initial_retained_tensor_check=initial_tensor,six_wheel_transforms_COM_and_origin_velocities_recorded=True,\n            response_observation_protocol=str(helper['PROTOCOL_PATH']),response_observation_protocol_sha256=helper['sha'](helper['PROTOCOL_PATH']),\n            reference_trial=reviewed['reference_trial'],")
            text = text.replace("scripts/'hospital_h3b_next_response_observation.py', scripts/'hospital_h3b_rolling_observation.py',",
                "scripts/'hospital_h3b_next_response_observation.py', scripts/'hospital_h3b_cp30_runtime_v2.py', scripts/'hospital_h3b_rolling_observation.py',")
            text = text.replace('Passive 120Hz joint/body + 30Hz native wheel-floor friction + raw contacts attached BEFORE motion.',
                'Passive 120Hz body/twelve joints/SIX wheel poses+velocities and 30Hz native friction/raw contacts attached BEFORE motion.')
            text = text.replace('ONE next FORWARD only: fixed screened reference, same course step/gains and 50mm/s cap; no terminal reapproach or later suffix leg.',
                'SAME already-PASS next FORWARD only; retained actual reverse history, unchanged path/control/gains/guards. Passive response observation, NOT model validation.')
        texts[kind] = text
    return texts


def validate_runtime_api():
    exports = audit_exports(globals(), (PRIMARY, OBSERVER, HANDOFF))
    first = json.loads(prefix.RESULT.read_bytes()); prep = first['physics_preparation_snapshot']
    if classify_contact is not prior.classify_contact:
        raise ValueError('Original contact classifier identity required')
    return dict(exports=exports, **probe_contact_classifier(classify_contact, ROBOT_ROOT, WHEELS,
        prep['support_floor'], prep['disabled_legacy_floor_colliders']))


def function_contract():
    generated = entry_texts()
    for kind, path in (('primary', PRIMARY), ('observer', OBSERVER)):
        if ast.dump(ast.parse(path.read_text()), include_attributes=False) != ast.dump(ast.parse(generated[kind]), include_attributes=False):
            raise ValueError('Entry differs outside declared lineage/API/passive measurement edits')
    for old, new, names in ((ENTRIES/'isaac_run_ijcas_hospital_h3b_next_forward_pilot.py', PRIMARY, PRIMARY_CALLBACKS),
            (ENTRIES/'isaac_run_ijcas_hospital_h3b_next_forward_instrumented.py', OBSERVER, ('step', 'contacts', 'remember'))):
        functions = [{n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(p.read_text()))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in (old, new)]
        if any(functions[0][name] != functions[1][name] for name in names):
            raise ValueError('Original control/safety/passive callback changed')
    return dict(entire_primary_control_flow_equal_after_declared_noncontrol_edits=True,
        all_nine_primary_callbacks_AST_unchanged=True, passive_callbacks_AST_unchanged=True,
        original_controller_AND_evaluator_used_without_override=True,
        runtime_interface_and_classifier_probes=validate_runtime_api(),
        input_sha256={str(p): sha(p) for p in (PRIMARY, OBSERVER, HANDOFF,
            ENTRIES/'isaac_run_ijcas_hospital_h3b_next_forward_pilot.py',
            ENTRIES/'isaac_run_ijcas_hospital_h3b_next_forward_instrumented.py')})


def load_protocol():
    record = json.loads(PROTOCOL_PATH.read_bytes()); verify_sources(record); verify_files(record['input_sha256'])
    if (record.get('version') != OBSERVATION_VERSION or record.get('controller_contract') != function_contract()
            or record.get('status') != 'SAME_NEXT_FORWARD_OBSERVATION_FROZEN_LIVE_SDK_PENDING'
            or record.get('whole_H3_drive_admission') is not False or record.get('new_model_deployed') is not False):
        raise ValueError('Exact one-component response protocol required; whole-H3 remains HOLD')
    return record


def load_admitted_profile():
    info = load_profile(); protocol = load_protocol(); live = json.loads(LIVE.read_bytes()); verify_files(live['input_sha256'])
    if (live.get('status') != 'PAUSED_ACTUAL_REVERSE_PASS_NEXT_RESPONSE_HANDOFF_NO_MOTION'
            or live.get('reverse_result_sha256') != sha(REVERSE)
            or live.get('protocol_sha256') != sha(PROTOCOL_PATH)
            or live.get('physical_state_reset') is not False or live.get('new_PhysX_motion_executed') is not False
            or live.get('retained_tensor_checks', {}).get('post_cleanup_original_settings_verified') is not True):
        raise ValueError('Exact cohort_03 actual reverse-PASS live handoff required')
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    info.update(live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
        admission=str(PROTOCOL_PATH), admission_sha256=sha(PROTOCOL_PATH),
        input_sha256={**info['input_sha256'], **protocol['input_sha256'], **live['input_sha256'],
            str(PROTOCOL_PATH): sha(PROTOCOL_PATH), str(LIVE): sha(LIVE)})
    return info


def review(info):
    contract = function_contract()
    return dict(reference_trial=info['reverse_result'], reference_trial_sha256=info['reverse_result_sha256'],
        same_saved_reverse_PASS_body_and_12_joints=True, controller_contract=contract,
        same_existing_next_forward_controller=True, whole_H3_drive_admission=False,
        input_sha256={**info['input_sha256'], **contract['input_sha256'], str(Path(__file__)): sha(__file__)})
