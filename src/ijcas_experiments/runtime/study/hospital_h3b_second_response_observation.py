"""SAME timed front-only SECOND reverse, with passive six-wheel telemetry.

Bind to the actual cohort_03 prefix -> reverse -> forward history, never a
reset/selected endpoint. Keep the frozen shared-timing controller, all safety
limits and eight component design screens. Those reused screens are NOT an
independently validated body model or whole-H3 drive admission.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import ast
import json
from pathlib import Path

import hospital_h3b_timed_second_reverse as prior
from hospital_h3b_timed_second_reverse import *
import hospital_h3b_next_response_observation as next_observation
from hospital_h3b_cp30_runtime_v2 import wheel_body_snapshot, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3b_cp30_runtime_v2 import coverage as wheel_coverage
from hospital_runtime_interface import audit_exports, probe_contact_classifier


OBSERVATION_VERSION = 'H3B_same_timed_second_reverse_six_wheel_response_observation_v1'
ROLE = 'SAME_TIMED_SECOND_REVERSE_PASSIVE_RESPONSE_OBSERVATION_NOT_VALIDATION'
BASE, COHORT, prefix = next_observation.BASE, next_observation.COHORT, next_observation.prefix
FORWARD, REVERSE = next_observation.RESULT, next_observation.REVERSE
REFERENCE = prior.OUTPUT/'hospital_h3b_timed_second_reverse_r01.json'
OUTPUT, OBSERVER_OUTPUT = COHORT/'second_reverse', COHORT/'second_reverse_instrumentation'
RESULT = OUTPUT/'hospital_h3b_second_response_r01.json'
LIVE = COHORT/'second_reverse_handoff/live_handoff.json'
PROTOCOL_PATH = BASE/'second_reverse_protocol_r01/second_response_design.json'
FORWARD_ANALYSIS = BASE/'next_forward_analyses/analysis_r01/actual_next_forward_response_review.json'
ENTRIES = _ijcas_path(WORKSPACE, 'isaac/scripts')
PRIMARY = ENTRIES/'isaac_run_ijcas_hospital_h3b_second_response_primary.py'
OBSERVER = ENTRIES/'isaac_run_ijcas_hospital_h3b_second_response_instrumented.py'
HANDOFF = ENTRIES/'isaac_review_ijcas_hospital_h3b_second_response_handoff.py'
telemetry_snapshot = wheel_body_snapshot
validate_tensor = next_observation.validate_tensor
exact_history = next_observation.exact_history
PRIMARY_CALLBACKS = next_observation.PRIMARY_CALLBACKS


def load_profile():
    info = prior.load_admitted_profile()
    history = next_observation.load_admitted_profile()
    forward = exact_history(FORWARD, Path(info['forward_result']))
    if (forward.get('same_original_next_forward_controller_AND_evaluator') is not True
            or forward['prior_reverse_result'] != str(REVERSE)
            or forward['prior_reverse_result_sha256'] != sha(REVERSE)
            or forward['expected_retained_joint_positions_rad'] != history['retained_all_joint_positions_rad']
            or forward['initial_saved_world_xyyaw'] != history['initial_world_xyyaw']
            or forward['physics_preparation_snapshot'] != history['preparation_record']['physics_preparation_snapshot']):
        raise ValueError('Exact actual next-forward control, physical baseline and retained first-reverse history required')
    analysis = json.loads(FORWARD_ANALYSIS.read_bytes())
    verify_sources(analysis); verify_files(analysis['input_sha256'])
    verify_files({c['trace_file']: c['trace_sha256'] for c in analysis['body_replays']})
    if (analysis['original_actual_verdict_unchanged'] != 'PASS' or analysis['telemetry_errors_unchanged']
            or analysis['whole_H3_drive_admission'] is not False
            or analysis['primary_CSV_byte_identical_to_previous_actual_next_forward'] is not True
            or not analysis['coverage']['six_wheel_transforms_and_COM_and_origin_velocities_recorded']):
        raise ValueError('Complete actual next-forward response audit required, NOT a new model validation')
    reverse = history['reverse_record']
    total = reverse['measured_center_distance_m']
    changes = len(reverse['direction_gate_events']) + len(forward['direction_gate_events'])
    if total != info['previous_measured_reverse_total_m'] or changes != info['completed_direction_changes']:
        raise ValueError('Cumulative measured reverse/direction-change history changed; never waive resources')
    pins = {**info['input_sha256'], **history['input_sha256'], **forward['frozen_input_sha256'],
        **analysis['input_sha256'], str(FORWARD_ANALYSIS): sha(FORWARD_ANALYSIS),
        str(FORWARD): sha(FORWARD), str(FORWARD.with_suffix('.csv')): sha(FORWARD.with_suffix('.csv')),
        str(Path(__file__)): sha(__file__)}
    info.update(first_record=history['first_record'], first_result=history['first_result'],
        first_result_sha256=history['first_result_sha256'], reverse_record=reverse,
        reverse_result=str(REVERSE), reverse_result_sha256=sha(REVERSE),
        forward_record=forward, forward_result=str(FORWARD), forward_result_sha256=sha(FORWARD),
        preparation_record=forward, initial_world_xyyaw=forward['final_world_xyyaw'],
        expected_saved_last_state=forward['last_recorded_state'],
        retained_all_joint_positions_rad=next_observation.reverse_observation.joints_from_last(forward),
        previous_measured_reverse_total_m=total, completed_direction_changes=changes,
        exact_complete_prefix_reverse_AND_forward_history_equivalence=True,
        original_eight_timing_AND_eight_adapter_design_cases_reused_NOT_recomputed=True,
        new_body_or_steering_model_deployed=False, whole_H3_drive_admission=False, input_sha256=pins)
    return info


def coverage(*args):
    result = wheel_coverage(*args)
    result['status'] = ('ACTUAL_SECOND_REVERSE_SIX_WHEEL_RESPONSE_RECORDED_NOT_MODEL_VALIDATION'
        if result['status'] == 'CP30_PASSIVE_DATA_RECORDED_FOR_RESPONSE_REVIEW_NOT_MODEL_VALIDATION'
        else 'INCOMPLETE_ACTUAL_SECOND_REVERSE_RESPONSE_DATA')
    return result


def entry_texts():
    original_primary = ENTRIES/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_pilot.py'
    original_observer = ENTRIES/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_instrumented.py'
    replacements = [(original_primary.name, PRIMARY.name), (original_observer.name, OBSERVER.name),
        ('hospital_h3b_timed_second_reverse_observation.py', Path(__file__).name),
        ('hospital_h3b_timed_second_reverse.py', Path(__file__).name),
        ('ijcas_hospital_h3b_timed_second_reverse_', 'ijcas_hospital_h3b_second_response_'),
        ('hospital_h3b_timed_second_reverse_r01', 'hospital_h3b_second_response_r01'),
        ('h3b_next_forward_pilot_finished_wait_for_review', 'h3b_next_response_pilot_finished_wait_for_review'),
        ('last_h3b_next_forward_pilot_gate', 'last_h3b_next_response_pilot_gate'),
        ('h3b_timed_second_reverse_pilot_finished_wait_for_review', 'h3b_second_response_pilot_finished_wait_for_review'),
        ('last_h3b_timed_second_reverse_pilot_gate', 'last_h3b_second_response_pilot_gate')]
    texts = {}
    for kind, path in (('primary', original_primary), ('observer', original_observer)):
        text = path.read_text()
        for old, new in replacements: text = text.replace(old, new)
        if kind == 'primary':
            needle = "h=loader(scripts/'hospital_h3b_second_response_observation.py');info=h['load_admitted_profile']()"
            text = text.replace(needle, "h=loader(scripts/'hospital_h3b_second_response_observation.py');h['validate_runtime_api']();info=h['load_admitted_profile']()")
            needle = '        robot=motor=drive_indices=old_kp=old_kd=old_limits=old_targets=None'
            text = text.replace(needle, "        report.update(same_original_timed_second_reverse_controller_AND_evaluator=True,\n"
                "            passive_six_wheel_response_observation_only=True,new_body_or_steering_model_deployed=False,\n"
                "            complete_prefix_reverse_AND_forward_history_equivalence_verified=True,\n"
                "            eight_timing_AND_eight_adapter_component_design_cases_reused_NOT_recomputed=True,\n"
                "            response_observation_protocol=str(h['PROTOCOL_PATH']),response_observation_protocol_sha256=h['sha'](h['PROTOCOL_PATH']))\n"+needle)
        else:
            text = text.replace('    import math\n', '    import math\n    import numpy as np\n')
            text = text.replace("rules = load(scripts/'hospital_h3b_second_response_observation.py'); info = rules['load_admitted_profile']()",
                "rules = load(scripts/'hospital_h3b_second_response_observation.py'); helper['validate_runtime_api'](); info = rules['load_admitted_profile']()")
            text = text.replace("    output = helper['OUTPUT']", "    output = helper['OBSERVER_OUTPUT']")
            snapshot = next_observation.OBSERVER.read_text().split('    def retained_tensor_check():', 1)[1].split('    patterns =', 1)[0]
            text = text.replace("    helper['telemetry_snapshot'](view, 0.)\n",
                "    helper['telemetry_snapshot'](view, 0.)\n    def retained_tensor_check():"+snapshot)
            text = text.replace("    clock = {'time_s': 0., 'next_friction_s': 0.}",
                "    if retained_tensor_check()!=initial_tensor: raise RuntimeError('Retained body/joints/settings changed during observer SDK preflight; no motion launched')\n    clock = {'time_s': 0., 'next_friction_s': 0.}")
            text = text.replace("original_controller_verdict=original['verdict'], original_error=original.get('error'),",
                "original_controller_verdict=original['verdict'], original_controller_source_sha256=original['source_sha256'], original_error=original.get('error'),")
            text = text.replace("            reference_trial=reviewed['reference_trial'],",
                "            initial_retained_tensor_check=initial_tensor,six_wheel_transforms_COM_and_origin_velocities_recorded=True,\n"
                "            response_observation_protocol=str(helper['PROTOCOL_PATH']),response_observation_protocol_sha256=helper['sha'](helper['PROTOCOL_PATH']),\n"
                "            reference_trial=reviewed['reference_trial'],")
            text = text.replace("scripts/'hospital_h3b_second_response_observation.py', scripts/'hospital_h3b_rolling_observation.py',",
                "scripts/'hospital_h3b_second_response_observation.py', scripts/'hospital_h3b_cp30_runtime_v2.py', scripts/'hospital_h3b_rolling_observation.py',")
            text = text.replace('Passive 120Hz joint/body + 30Hz native wheel-floor friction + raw contacts attached BEFORE motion.',
                'Passive 120Hz body/twelve joints/SIX wheel poses+velocities and 30Hz native friction/raw contacts attached BEFORE motion.')
            text = text.replace('ONE frozen SECOND front-only reverse: screened shared timing/braking controller, same gains/guards; no later leg.',
                'SAME already-PASS SECOND front-only reverse; original path/timing/control/gains/guards and actual history. Passive response observation, NOT model validation.')
        texts[kind] = text
    return texts


def validate_runtime_api():
    exports = audit_exports(globals(), (PRIMARY, OBSERVER, HANDOFF))
    prep = json.loads(FORWARD.read_bytes())['physics_preparation_snapshot']
    if classify_contact is not prior.classify_contact: raise ValueError('Original classifier identity required')
    return dict(exports=exports, **probe_contact_classifier(classify_contact, ROBOT_ROOT, WHEELS,
        prep['support_floor'], prep['disabled_legacy_floor_colliders']))


def function_contract():
    generated = entry_texts()
    for kind, path in (('primary', PRIMARY), ('observer', OBSERVER)):
        if ast.dump(ast.parse(path.read_text()), include_attributes=False) != ast.dump(ast.parse(generated[kind]), include_attributes=False):
            raise ValueError('Entry differs beyond declared lineage/API/passive measurements')
    for old, new, names in ((ENTRIES/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_pilot.py', PRIMARY, PRIMARY_CALLBACKS),
            (ENTRIES/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_instrumented.py', OBSERVER, ('step', 'contacts', 'remember'))):
        maps = [{n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(p.read_text()))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in (old, new)]
        if any(maps[0][name] != maps[1][name] for name in names): raise ValueError('Control/guard/passive callback changed')
    return dict(entire_primary_control_flow_equal_after_declared_noncontrol_edits=True,
        all_nine_primary_callbacks_AST_unchanged=True, passive_callbacks_AST_unchanged=True,
        original_controller_AND_evaluator_used_without_override=True,
        runtime_interface_and_classifier_probes=validate_runtime_api(),
        input_sha256={str(p):sha(p) for p in (PRIMARY, OBSERVER, HANDOFF,
            ENTRIES/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_pilot.py',
            ENTRIES/'isaac_run_ijcas_hospital_h3b_timed_second_reverse_instrumented.py')})


def load_protocol():
    record = json.loads(PROTOCOL_PATH.read_bytes()); verify_sources(record); verify_files(record['input_sha256'])
    if (record.get('version') != OBSERVATION_VERSION or record.get('controller_contract') != function_contract()
            or record.get('status') != 'SAME_SECOND_REVERSE_OBSERVATION_FROZEN_LIVE_SDK_PENDING'
            or record.get('whole_H3_drive_admission') is not False or record.get('new_model_deployed') is not False):
        raise ValueError('Exact one-component response protocol required; whole-H3 HOLD remains')
    return record


def load_admitted_profile():
    info = load_profile(); protocol = load_protocol(); live = json.loads(LIVE.read_bytes()); verify_files(live['input_sha256'])
    if (live.get('status') != 'PAUSED_ACTUAL_FORWARD_PASS_SECOND_RESPONSE_HANDOFF_NO_MOTION'
            or live.get('forward_result_sha256') != sha(FORWARD) or live.get('protocol_sha256') != sha(PROTOCOL_PATH)
            or live.get('physical_state_reset') is not False or live.get('new_PhysX_motion_executed') is not False
            or live.get('retained_tensor_checks', {}).get('post_cleanup_original_settings_verified') is not True):
        raise ValueError('Exact retained actual cohort_03 NEXT-forward live handoff required')
    validate_live(info, live['live_world_xyyaw'], live['retained_all_joint_positions_rad'], live['live_root_velocity'])
    info.update(live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
        admission=str(PROTOCOL_PATH), admission_sha256=sha(PROTOCOL_PATH),
        input_sha256={**info['input_sha256'], **protocol['input_sha256'], **live['input_sha256'],
            str(PROTOCOL_PATH):sha(PROTOCOL_PATH), str(LIVE):sha(LIVE)})
    return info


def review(info):
    contract = function_contract()
    return dict(reference_trial=info['forward_result'], reference_trial_sha256=info['forward_result_sha256'],
        same_saved_actual_forward_PASS_body_and_12_joints=True, same_existing_timed_second_reverse_controller=True,
        controller_contract=contract, whole_H3_drive_admission=False,
        input_sha256={**info['input_sha256'], **contract['input_sha256'], str(Path(__file__)):sha(__file__)})
