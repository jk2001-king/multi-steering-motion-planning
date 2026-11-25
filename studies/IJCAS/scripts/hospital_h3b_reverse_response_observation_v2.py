"""Same measured-terminal first reverse after the actual CP50 observation.

No new control law, fit, path, gain, caster seed or full-H3 admission. Reuse
the complete original closedloop primary through declared lineage-only edits.
Add the already-tested passive SIX-wheel pose/velocity measurement to the
original 120Hz joint/body and 30Hz native-contact observer. Never reset a
different live state to the saved endpoint.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_prefix_response_observation_v2 as prefix
import hospital_h3b_reverse_closedloop_pilot as prior
from hospital_h3b_cp30_runtime_v2 import wheel_body_snapshot, friction_snapshot, FRICTION_PERIOD_S, coverage as wheel_coverage
from hospital_h3b_reverse_closedloop_prefix import validate_model
from hospital_h3_flush_support import sha, verify_files
from hospital_h3_turn_planning import WORKSPACE, GATE, verify_sources


OBSERVATION_VERSION = 'H3B_same_actual_first_reverse_six_wheel_response_observation_v2'
VERSION = prior.VERSION
ROLE = 'SAME_MEASURED_TERMINAL_FIRST_REVERSE_PASSIVE_RESPONSE_OBSERVATION_NOT_VALIDATION'
BASE, COHORT = prefix.BASE, prefix.COHORT
FIRST = prefix.RESULT
OUTPUT, OBSERVER_OUTPUT = COHORT/'reverse', COHORT/'reverse_instrumentation'
RESULT = OUTPUT/'hospital_h3b_reverse_response_r01.json'
ADMISSION = COHORT/'reverse_admission/closedloop_model_review.json'
LIVE = COHORT/'reverse_handoff/live_handoff.json'
PROTOCOL_PATH = BASE/'reverse_protocol_r02/reverse_response_observation_design.json'
REFERENCE = prior.OUTPUT/'hospital_h3b_reverse_closedloop_rear_r01.json'
PREFIX_ANALYSIS = BASE/'analyses/analysis_r01/actual_prefix_response_review.json'
ENTRIES = WORKSPACE/'isaac/scripts'
PRIMARY = ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_response_primary_v2.py'
OBSERVER = ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_response_instrumented_v2.py'
HANDOFF = ENTRIES/'isaac_review_ijcas_hospital_h3b_reverse_response_handoff_v2.py'
PREFIX_MARKER_STATUS = 'h3b_prefix_response_pilot_finished_fresh_setup_required'
# This legacy key is what the ACTUAL, frozen prefix entry writes. Bind its
# value to FIRST+SHA rather than renaming the live scene or old entry.
PREFIX_MARKER_GATE = 'last_h3b_reverse_force_prefix_pilot_gate'
POLICY, TRACKING, GUARDS = prior.POLICY, prior.TRACKING, prior.GUARDS
ROBOT_ROOT, WHEELS, BAD_CONTACTS = prior.ROBOT_ROOT, prior.WHEELS, prior.BAD_CONTACTS
StopAlignGate, FirstReverseReference = prior.StopAlignGate, prior.FirstReverseReference
FirstReverseProgress, arrival_checks = prior.FirstReverseProgress, prior.arrival_checks
direct_joint_targets, motion_stalled = prior.direct_joint_targets, prior.motion_stalled
evaluate_component, relative_pose = prior.evaluate_component, prior.relative_pose
read_trace, joints_from_last = prior.read_trace, prior.joints_from_last
telemetry_snapshot = wheel_body_snapshot


def load_profile():
    info = prior.load_profile()
    first = json.loads(FIRST.read_bytes())
    prefix.load_protocol(); verify_sources(first); verify_files(first['frozen_input_sha256'])
    if (first.get('trial_kind') != prefix.ROLE or first.get('verdict') != 'PASS'
            or not first.get('checks') or not all(first['checks'].values())
            or first.get('original_motor_gains_restored') is not True
            or first.get('PhysX_executed') is not True
            or first.get('passive_same_actual_domain_observation_NOT_new_model_validation') is not True
            or first.get('M1_M2_M3_models_NOT_used_in_commands') is not True):
        raise ValueError('Require the actual completed CP50 observation-prefix PASS, not an older scene')
    old = info['first_record']
    for key in ('last_recorded_state','final_world_xyyaw','start_world_xyyaw',
            'mission_goal_world_xyyaw_UNCHANGED','strict_flush_map_digest','target_world_xyyaw'):
        if first[key] != old[key]:
            raise ValueError('Actual prefix differs from the old fixed first-reverse seed: '+key+'; do NOT reset to match')
    if FIRST.with_suffix('.csv').read_bytes() != Path(info['first_result']).with_suffix('.csv').read_bytes():
        raise ValueError('Same complete source-to-prefix history required, not an endpoint-only success selection')
    for key in ('environment_condition','source_stage_sha256','robot_reference_sha256','start_xyyaw',
            'component_end_xyyaw','simulation_mass_kg','live_support_readback','support_floor','disabled_legacy_floor_colliders'):
        if first['physics_preparation_snapshot'][key] != old['physics_preparation_snapshot'][key]:
            raise ValueError('Original physical baseline changed: '+key)
    rows = read_trace(FIRST.with_suffix('.csv'))
    if len(rows) != first['sample_count'] or rows[-1] != first['last_recorded_state']:
        raise ValueError('Complete actual prefix CSV and final twelve-joint state required')
    static_path = Path(first['static_reference'])
    verify_files({static_path:first['static_reference_sha256'],static_path.with_suffix('.csv'):first['static_trace_sha256'],
        first['geometry_manifest']:first['geometry_manifest_sha256']})
    evaluated = prior.evaluate_prefix(rows,first['target_world_xyyaw'],read_trace(static_path.with_suffix('.csv')),
        first['physics_preparation_snapshot']['simulation_mass_kg'],first['doorway_reference_bound_m'],
        first['wheel_conservative_sphere_radii_m'],first['contact_report_counts_by_category'],
        first['contact_callback_errors'],first['fixed_hold_attempt_count'],first['direction_gate_events'])
    if evaluated['verdict'] != 'PASS' or any(first['checks'].get(k)!=v for k,v in evaluated['checks'].items()):
        raise ValueError('Actual raw prefix PASS must reproduce')
    analysis = json.loads(PREFIX_ANALYSIS.read_bytes()); verify_sources(analysis); verify_files(analysis['input_sha256'])
    observer_path = prefix.OBSERVER_OUTPUT/'instrumentation.json'
    observer = json.loads(observer_path.read_bytes()); verify_sources(observer)
    verify_files({str(prefix.OBSERVER_OUTPUT/n):v for n,v in observer['saved_file_sha256'].items()})
    if (analysis['original_actual_verdict_unchanged']!='PASS' or analysis['whole_H3_drive_admission'] is not False
            or observer['original_controller_result_sha256']!=sha(FIRST) or observer['telemetry_errors']
            or not all(observer['each_wheel_has_friction_anchor_observations'])):
        raise ValueError('Complete audited actual-prefix passive data required; no model success inferred')
    info.update(first_record=first,first_result=str(FIRST),first_result_sha256=sha(FIRST),
        initial_world_xyyaw=first['final_world_xyyaw'],expected_saved_last_state=first['last_recorded_state'],
        retained_all_joint_positions_rad=joints_from_last(first),
        input_sha256={**info['input_sha256'],**first['frozen_input_sha256'],**analysis['input_sha256'],
            str(FIRST):sha(FIRST),str(FIRST.with_suffix('.csv')):sha(FIRST.with_suffix('.csv')),
            str(PREFIX_ANALYSIS):sha(PREFIX_ANALYSIS),str(Path(__file__)):sha(__file__)},
        same_original_reverse_law_AND_complete_prefix_history=True,
        whole_H3_drive_admission=False,M1_M2_M3_models_deployed=False)
    return info


def validate_live(info,pose,joints,velocity):
    return prior.validate_live(info,pose,joints,velocity)


def validate_tensor(info,transform,names,positions,velocity,settings):
    transform,positions,velocity=(np.asarray(v,float) for v in (transform,positions,velocity))
    if (transform.shape!=(7,) or positions.shape!=(12,) or velocity.shape!=(6,) or len(set(names))!=12
            or not all(np.isfinite(v).all() for v in (transform,positions,velocity))):
        raise ValueError('Finite retained root/six velocities and all twelve joints required')
    if abs(float(np.linalg.norm(transform[3:]))-1.)>.001:
        raise ValueError('Invalid retained tensor quaternion')
    x,y,z,w=transform[3:];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    result=validate_live(info,[transform[0],transform[1],yaw],dict(zip(names,map(float,positions))),velocity)
    last=info['expected_saved_last_state']
    if abs(transform[2]-last['z_m'])>.001:
        raise ValueError('Retained prefix base height changed; no recovery or state assignment permitted')
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
    if max(abs(math.degrees(roll)-last['roll_deg']),abs(math.degrees(pitch)-last['pitch_deg']))>.1:
        raise ValueError('Retained prefix roll/pitch changed; no recovery permitted')
    baseline,_=prefix.load_receipt()
    if settings!=baseline['live_static_joint_settings']:
        raise ValueError('Post-cleanup gains/limits/zero velocity targets differ from the unchanged original baseline')
    return dict(result,base_height_difference_m=abs(float(transform[2])-last['z_m']),
        all_twelve_actual_positions_read_NOT_assigned=True,post_cleanup_original_settings_verified=True,
        state_equality_tolerances_unchanged=dict(position_m=.001,heading_deg=.1,all_twelve_direct_joint_rad=.001))


def load_reviewed_profile():
    info=load_profile();record=json.loads(ADMISSION.read_bytes());validate_model(record)
    if (record['first_result']!=str(FIRST) or record['first_result_sha256']!=sha(FIRST)
            or record['initial_actual_world_xyyaw']!=info['initial_world_xyyaw']
            or record['retained_all_joint_positions_rad']!=info['retained_all_joint_positions_rad']
            or record['suffix_candidate_sha256']!=info['suffix_candidate_sha256']
            or record['strict_map_digest']!=info['strict_map_digest']):
        raise ValueError('All eight SAME-first-reverse model design screens must bind to THIS actual prefix')
    info.update(admission=str(ADMISSION),admission_sha256=sha(ADMISSION),
        input_sha256={**info['input_sha256'],**record['input_sha256'],str(ADMISSION):sha(ADMISSION)})
    return info


def load_admitted_profile():
    info=load_reviewed_profile();protocol=load_protocol();live=json.loads(LIVE.read_bytes());verify_files(live['input_sha256'])
    if (live.get('status')!='PAUSED_FRESH_PREFIX_REAR_HANDOFF_REVIEWED_NO_MOTION'
            or live.get('physical_state_reset') is not False or live.get('rear_reverse_PhysX_executed') is not False
            or live['first_result_sha256']!=sha(FIRST) or live['model_admission_sha256']!=info['admission_sha256']
            or not live.get('retained_tensor_checks',{}).get('post_cleanup_original_settings_verified')):
        raise ValueError('THIS retained actual-prefix no-motion live handoff required; do not reset/reopen')
    validate_live(info,live['live_world_xyyaw'],live['retained_all_joint_positions_rad'],live['live_root_velocity'])
    info.update(live_receipt=str(LIVE),live_receipt_sha256=sha(LIVE),
        input_sha256={**info['input_sha256'],**protocol['input_sha256'],str(PROTOCOL_PATH):sha(PROTOCOL_PATH),
            **live['input_sha256'],str(LIVE):sha(LIVE)})
    return info


def coverage(*args):
    result=wheel_coverage(*args)
    result['status']=('ACTUAL_FIRST_REVERSE_SIX_WHEEL_RESPONSE_RECORDED_NOT_MODEL_VALIDATION'
        if result['status']=='CP30_PASSIVE_DATA_RECORDED_FOR_RESPONSE_REVIEW_NOT_MODEL_VALIDATION'
        else 'INCOMPLETE_ACTUAL_REVERSE_RESPONSE_DATA')
    return result


def entry_texts():
    paths=dict(primary=ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_pilot.py',
        observer=ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_instrumented.py',
        handoff=ENTRIES/'isaac_review_ijcas_hospital_h3b_reverse_closedloop_handoff.py')
    substitutions=[(paths['primary'].name,PRIMARY.name),(paths['observer'].name,OBSERVER.name),(paths['handoff'].name,HANDOFF.name),
        ('hospital_h3b_reverse_closedloop_observation.py','hospital_h3b_reverse_response_observation_v2.py'),
        ('hospital_h3b_reverse_closedloop_pilot.py','hospital_h3b_reverse_response_observation_v2.py'),
        ('ijcas_hospital_h3b_reverse_closedloop_','ijcas_hospital_h3b_reverse_response_'),
        ('h3b_reverse_closedloop_prefix_pilot_finished_fresh_setup_required',PREFIX_MARKER_STATUS),
        ('last_h3b_reverse_closedloop_prefix_pilot_gate',PREFIX_MARKER_GATE),
        ('h3b_reverse_closedloop_pilot_finished_wait_for_review','h3b_reverse_response_pilot_finished_wait_for_review'),
        ('last_h3b_reverse_closedloop_pilot_gate','last_h3b_reverse_response_pilot_gate'),
        ('hospital_h3b_reverse_closedloop_rear_r01','hospital_h3b_reverse_response_r01'),
        ('H3-B NEW MEASURED TERMINAL','H3-B SAME MEASURED TERMINAL RESPONSE OBSERVATION'),
        ('H3-B NEW MEASURED-TERMINAL','H3-B SAME MEASURED-TERMINAL RESPONSE OBSERVATION'),
        ('ONE_NEW_MEASURED_TERMINAL_REAR_REVERSE_WITH_PASSIVE_FORCES',ROLE)]
    result={}
    for kind,path in paths.items():
        text=path.read_text()
        for before,after in substitutions:text=text.replace(before,after)
        if kind=='primary':
            needle='        robot=motor=drive_indices=old_kp=old_kd=old_limits=old_targets=None'
            text=text.replace(needle,"        report.update(same_original_measured_terminal_controller_NOT_a_new_revision=True,\n"
                "            passive_six_wheel_response_observation_only=True,M1_M2_M3_models_NOT_used_in_commands=True,\n"
                "            response_observation_protocol=str(h['PROTOCOL_PATH']),response_observation_protocol_sha256=h['sha'](h['PROTOCOL_PATH']))\n"+needle)
        elif kind=='handoff':
            text=text.replace("h = load(scripts / 'hospital_h3b_reverse_response_observation_v2.py'); info = h['load_reviewed_profile']()",
                "h = load(scripts / 'hospital_h3b_reverse_response_observation_v2.py'); info = h['load_reviewed_profile'](); h['load_protocol']()")
            needle="    checks = h['validate_live'](info, pose, joints, velocity)"
            addition="""
    settings={}
    for key,getter in (('kp',view.get_dof_stiffnesses),('kd',view.get_dof_dampings),
            ('force_limit',view.get_dof_max_forces),('velocity_target_rad_s',view.get_dof_velocity_targets)):
        settings[key]=dict(zip(names,map(float,numeric(getter()))))
    tensor_checks=h['validate_tensor'](info,transform,names,q,velocity,settings)
"""
            text=text.replace(needle,needle+addition)
            text=text.replace("script_sha256=h['sha'](script), physical_state_reset=False,",
                "script_sha256=h['sha'](script), physical_state_reset=False,retained_tensor_checks=tensor_checks,")
        else:
            text=text.replace("    output = helper['OUTPUT']","    output = helper['OBSERVER_OUTPUT']")
            text=text.replace("    helper['telemetry_snapshot'](view, 0.)","""    helper['telemetry_snapshot'](view, 0.)
    def retained_tensor_check():
        names=list(view.shared_metatype.dof_names)
        def numeric(value):
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            elif hasattr(value,'numpy'):value=value.numpy()
            return np.asarray(value,dtype=float).reshape(-1)
        settings={}
        for key,getter in (('kp',view.get_dof_stiffnesses),('kd',view.get_dof_dampings),
                ('force_limit',view.get_dof_max_forces),('velocity_target_rad_s',view.get_dof_velocity_targets)):
            settings[key]=dict(zip(names,map(float,numeric(getter()))))
        return helper['validate_tensor'](info,numeric(view.get_root_transforms()),names,
            numeric(view.get_dof_positions()),numeric(view.get_root_velocities()),settings)
    initial_tensor=retained_tensor_check()""")
            text=text.replace('    import math\n','    import math\n    import numpy as np\n')
            text=text.replace("    clock = {'time_s': 0., 'next_friction_s': 0.}",
                "    if retained_tensor_check()!=initial_tensor: raise RuntimeError('Retained body/joints/settings changed during observer SDK preflight; no motion launched')\n    clock = {'time_s': 0., 'next_friction_s': 0.}")
            text=text.replace("original_controller_verdict=original['verdict'], original_error=original.get('error'),",
                "original_controller_verdict=original['verdict'],original_controller_source_sha256=original['source_sha256'],original_error=original.get('error'),")
            text=text.replace("            actual_prefix=info['first_result'], actual_prefix_sha256=info['first_result_sha256'],",
                "            actual_prefix=info['first_result'], actual_prefix_sha256=info['first_result_sha256'],\n"
                "            initial_retained_tensor_check=initial_tensor,six_wheel_transforms_COM_and_origin_velocities_recorded=True,\n"
                "            response_observation_protocol=str(helper['PROTOCOL_PATH']),response_observation_protocol_sha256=helper['sha'](helper['PROTOCOL_PATH']),")
            text=text.replace("    print('ONE NEW terminal executor: measured-speed/ramp-aware braking + arrival gate; at most one same-direction reapproach. Same path/50mm_s cap/gains/contact/caster conditions.')",
                "    print('SAME already-PASS measured-terminal rear reverse; retained actual CP50-prefix history. No control/model fit or physics change; one passive observation only.')")
            text=text.replace("    print('Passive 120Hz joint/body + 30Hz native wheel-floor friction + raw contacts attached BEFORE motion.')",
                "    print('Passive 120Hz body/twelve joints/SIX wheel poses+velocities and 30Hz native friction/raw contacts attached BEFORE motion.')")
        result[kind]=text
    return result


def function_contract():
    generated=entry_texts();paths=dict(primary=PRIMARY,observer=OBSERVER,handoff=HANDOFF)
    for key,path in paths.items():
        if ast.dump(ast.parse(path.read_text()),include_attributes=False)!=ast.dump(ast.parse(generated[key]),include_attributes=False):
            raise ValueError('Entry changed outside declared admission/provenance/passive-sensor edits: '+key)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):
                name=Path(node.value).name
                if name.startswith('isaac_') and not (ENTRIES/name).is_file():raise ValueError('Missing entry dependency: '+name)
    for old,new,names in ((ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_pilot.py',PRIMARY,
            ('numeric','physics_step','contacts','angles','verify_gains','measure','set_hold','command','sample')),
            (ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_instrumented.py',OBSERVER,('step','contacts','remember'))):
        trees=[{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text()))
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for p in (old,new)]
        if any(trees[0][name]!=trees[1][name] for name in names):raise ValueError('Control/safety or passive callback changed')
    assigned={n.targets[0].id for n in ast.walk(ast.parse(PRIMARY.read_text())) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    if 'ijcas_hospital_h3b_reverse_response_pilot_task' not in assigned:raise ValueError('Observer primary dispatch task mismatch')
    return dict(entire_entries_match_declared_noncontrol_transforms=True,all_nine_primary_callbacks_AST_unchanged=True,
        same_original_measured_terminal_controller_AND_evaluator=True,passive_callbacks_AST_unchanged=True,
        added_sensors='existing runtime-only SIX wheel body pose/COM+origin velocity snapshot, no actions',
        input_sha256={str(p):sha(p) for p in (*paths.values(),ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_pilot.py',
            ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_closedloop_instrumented.py',ENTRIES/'isaac_review_ijcas_hospital_h3b_reverse_closedloop_handoff.py')})


def load_protocol():
    record=json.loads(PROTOCOL_PATH.read_bytes());verify_sources(record);verify_files(record['input_sha256'])
    if (record.get('version')!=OBSERVATION_VERSION or record.get('controller_contract')!=function_contract()
            or record.get('status')!='SAME_FIRST_REVERSE_OBSERVATION_MODEL_DESIGN_ADMITTED_LIVE_SDK_PENDING'
            or record.get('whole_H3_drive_admission') is not False or record.get('new_model_deployed') is not False
            or record.get('first_result_sha256')!=sha(FIRST)):
        raise ValueError('Only the same actual-prefix first-reverse passive observation is admitted')
    return record


def review(info):
    protocol=load_protocol();reference=json.loads(REFERENCE.read_bytes());verify_sources(reference)
    if reference['verdict']!='PASS' or reference.get('original_motor_gains_restored') is not True:
        raise ValueError('Original same-executor rear-reverse PASS must remain authoritative')
    return dict(reference_trial=str(REFERENCE),reference_trial_sha256=sha(REFERENCE),controller_contract=function_contract(),
        same_original_executor_AND_physical_prefix_history=True,
        input_sha256={**info['input_sha256'],**protocol['input_sha256'],str(PROTOCOL_PATH):sha(PROTOCOL_PATH),
            str(REFERENCE):sha(REFERENCE),str(REFERENCE.with_suffix('.csv')):sha(REFERENCE.with_suffix('.csv'))})

