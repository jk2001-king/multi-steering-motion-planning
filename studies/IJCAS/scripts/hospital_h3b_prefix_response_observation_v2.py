"""Observe the ACTUAL H3 first-forward domain, not another small-S fit.

Same fixed CP50 course-feedback prefix, source initial state and physical
settings; passive 120Hz body/twelve-joint/six-wheel and 30Hz native contacts.
No M1/M2/M3 in control. No model acceptance, reverse or whole-H3 admission.
"""

import ast
import json
from pathlib import Path

import hospital_h3b_reverse_force_prefix as parent
import hospital_h3b_neutral_response as static_baseline
from hospital_h3b_cp30_runtime_v2 import wheel_body_snapshot, vector, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3_turn_planning import GATE, WORKSPACE, verify_sources
from hospital_h3_flush_support import sha, verify_files


VERSION = "H3B_same_actual_CP50_large_angle_prefix_passive_observation_v2"
CONDITION = parent.CONDITION
ROLE = "SAME_ACTUAL_H3_PREFIX_PASSIVE_RESPONSE_OBSERVATION_NOT_MODEL_VALIDATION"
CASE = "actual_first_forward_prefix_CP50_large_angle"
BASE = GATE/"hospital_h3b_prefix_response_observation"
COHORT = BASE/"cohort_02"
OUTPUT, OBSERVER_OUTPUT = COHORT/"prefix", COHORT/"instrumentation"
RESULT = OUTPUT/"hospital_h3b_prefix_response_r01.json"
PROTOCOL_PATH = BASE/"protocol_r02/actual_prefix_observation_design.json"
PREPARATION_PROTOCOL_PATH = PROTOCOL_PATH
REFERENCE = parent.OUTPUT/"hospital_h3b_reverse_force_prefix_r01.json"
READBACK = static_baseline.READBACK
STATE_TOLERANCES_UNCHANGED = static_baseline.STATE_TOLERANCES_UNCHANGED
ENTRIES = WORKSPACE/"isaac/scripts"
PREPARE = ENTRIES/"isaac_prepare_ijcas_hospital_h3b_prefix_response_observation_v2.py"
PRIMARY = ENTRIES/"isaac_run_ijcas_hospital_h3b_prefix_response_primary_v2.py"
OBSERVER = ENTRIES/"isaac_run_ijcas_hospital_h3b_prefix_response_instrumented_v2.py"
POLICY, READINESS, ACCEPTANCE = parent.POLICY, parent.READINESS, parent.ACCEPTANCE
BAD_CONTACTS, ROBOT_ROOT, WHEELS, GUARDS = parent.BAD_CONTACTS, parent.ROBOT_ROOT, parent.WHEELS, parent.GUARDS
TRACKING, COMMAND_LIMITS, TRACKING_GAINS = parent.TRACKING, parent.COMMAND_LIMITS, parent.TRACKING_GAINS
classify_contact, relative_pose = parent.classify_contact, parent.relative_pose
wheel_crossing_checks, motion_stalled = parent.wheel_crossing_checks, parent.motion_stalled
direct_joint_targets, component_goal = parent.direct_joint_targets, parent.component_goal
interpolation_table, reference_at_pose = parent.interpolation_table, parent.reference_at_pose
FirstLegFeedback, body_velocity = parent.FirstLegFeedback, parent.body_velocity
evaluate_readiness, load_profile = parent.evaluate_readiness, parent.load_profile
load_receipt, validate_handoff = static_baseline.load_receipt, static_baseline.validate_handoff
read_trace = static_baseline.read_trace


def coverage(*args):
    result=static_baseline.coverage(*args)
    result["status"]=("ACTUAL_CP50_LARGEANGLE_PREFIX_PASSIVE_DATA_RECORDED_NOT_MODEL_VALIDATION"
        if result["status"]=="NEUTRAL_CP30_IDENTIFICATION_DATA_RECORDED_NOT_MODEL_VALIDATION" else "INCOMPLETE_ACTUAL_PREFIX_RESPONSE_DATA")
    return result


def entry_texts():
    old_prepare=ENTRIES/"isaac_prepare_ijcas_hospital_h3b_reverse_force_prefix.py"
    old_primary=ENTRIES/"isaac_run_ijcas_hospital_h3b_reverse_force_prefix.py"
    replacements=[("hospital_h3b_reverse_force_prefix.py","hospital_h3b_prefix_response_observation_v2.py"),
        (old_prepare.name,PREPARE.name),(old_primary.name,PRIMARY.name),
        ("ijcas_hospital_h3b_reverse_force_prefix_","ijcas_hospital_h3b_prefix_response_"),
        ("H3B_reverse_force_prefix_preparation","H3B_prefix_response_preparation"),
        ("hospital_h3b_reverse_force_prefix_r01","hospital_h3b_prefix_response_r01"),
        ("FORCE-OBSERVATION SAME FIRST FORWARD PREFIX","ACTUAL-DOMAIN PASSIVE SAME FIRST FORWARD PREFIX"),
        ("FORCE-OBSERVATION IDENTICAL FIRST FORWARD PREFIX","ACTUAL-DOMAIN PASSIVE FIRST FORWARD PREFIX")]
    output={}
    for label,path in (("prepare",old_prepare),("primary",old_primary)):
        text=path.read_text()
        for before,after in replacements:text=text.replace(before,after)
        needle=("    h=load(scripts/'hospital_h3b_prefix_response_observation_v2.py')" if label=="prepare"
            else '    h=loader["load"](scripts/"hospital_h3b_prefix_response_observation_v2.py")')
        if text.count(needle)!=1:raise ValueError("Prefix observer admission insertion drift")
        addition="\n    h['load_protocol']()"
        if label=="primary":
            addition+='\n    if globals().pop("ijcas_hospital_h3b_prefix_response_observer_ready",None)!="PREFIX_RESPONSE_PREFLIGHT_COMPLETE":\n        raise RuntimeError("Use ONLY the dedicated actual-prefix passive-observer entry after actual fresh static PASS")'
        text=text.replace(needle,needle+addition)
        if label=="prepare":
            needle="        with target.open('x') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\\n')"
            if text.count(needle)!=1:raise ValueError("Prefix preparation provenance insertion drift")
            text=text.replace(needle,"        record.update(prefix_response_protocol=str(h['PROTOCOL_PATH']),prefix_response_protocol_sha256=support['sha'](h['PROTOCOL_PATH']))\n"+needle)
        else:
            needle='        robot=motor=drive_indices=old_kp=old_kd=old_limits=old_targets=None'
            if text.count(needle)!=1:raise ValueError("Prefix primary provenance insertion drift")
            text=text.replace(needle,'        report.update(prefix_response_protocol=str(h["PROTOCOL_PATH"]),prefix_response_protocol_sha256=support["sha"](h["PROTOCOL_PATH"]),\n            passive_same_actual_domain_observation_NOT_new_model_validation=True,M1_M2_M3_models_NOT_used_in_commands=True)\n'+needle)
        text=text.replace('"h3b_reverse_force_prefix_pilot_finished_fresh_setup_required"','"h3b_prefix_response_pilot_finished_fresh_setup_required"')
        # Correct metadata paths and primary dispatch name before freezing v2.
        # Original v1 was never executed; retain it and its protocol unchanged.
        text=text.replace("isaac_run_ijcas_hospital_h3b_prefix_response_observation_v2.py",PRIMARY.name)
        text=text.replace("ijcas_hospital_h3b_prefix_response_task","ijcas_hospital_h3b_prefix_response_pilot_task")
        output[label]=text
    # Use the already-working SAME-PHASE neutral observer SDK/serialization;
    # change ONLY preparation/role/dispatch lineage, never passive callbacks.
    text=static_baseline.OBSERVER.read_text()
    for before,after in [("hospital_h3b_neutral_response.py","hospital_h3b_prefix_response_observation_v2.py"),
        (static_baseline.PRIMARY.name,PRIMARY.name),(static_baseline.OBSERVER.name,OBSERVER.name),
        ("ijcas_hospital_h3b_neutral_observer_condition","ijcas_hospital_h3b_prefix_response_observer_condition"),
        ("H3B_neutral_straight_identification_observation","H3B_actual_prefix_response_observation"),
        ("ijcas_hospital_h3b_neutral_observer_ready","ijcas_hospital_h3b_prefix_response_observer_ready"),
        ("NEUTRAL_OBSERVER_PREFLIGHT_COMPLETE","PREFIX_RESPONSE_PREFLIGHT_COMPLETE"),
        ("ijcas_hospital_h3b_neutral_response_task","ijcas_hospital_h3b_prefix_response_pilot_task"),
        ("ijcas_hospital_h3b_neutral_observer_task","ijcas_hospital_h3b_prefix_response_observer_task"),
        ("NEUTRAL_STRAIGHT_CP30_IDENTIFICATION_NOT_VALIDATION",ROLE),
        ('marker.GetCustomDataByKey("H3B_mode_speed_preparation")','marker.GetCustomDataByKey("H3B_prefix_response_preparation")'),
        ('marker.GetCustomDataByKey("H3B_mode_speed_preparation_sha256")','marker.GetCustomDataByKey("H3B_prefix_response_preparation_sha256")'),
        ('preparation.get("case")!=helper["CASE"]','preparation.get("role")!=helper["ROLE"]'),
        ('preparation.get("neutral_identification_protocol_sha256")','preparation.get("prefix_response_protocol_sha256")'),
        ('globals()["ijcas_hospital_h3b_neutral_component_condition"]="H3B_neutral_straight_component_v1"',
         'globals()["ijcas_hospital_h3b_prefix_response_condition"]="H3B_rear_reverse_prefix_reproduction_v1"'),
        ("CP30 actual static","common actual static"),
        ("CP30 PRE-hold baseline","recorded common PRE-hold baseline, NOT historical prefix joints"),
        ("NEUTRAL CP30 IDENTIFICATION","ACTUAL CP50 LARGE-ANGLE PREFIX RESPONSE"),
        ("H3-B NEW neutral input: ONLY curvature -> zero; same 0.65m/30mm_s/ramp/gains/guards and prescribed caster seed. Identification, not model validation.",
         "H3-B SAME actual first-forward prefix: CP50/up to ~24deg, same fixed course law/gains/path/guards. Passive observation ONLY; no new fit/model admission.")]:text=text.replace(before,after)
    output["observer"]=text
    return output


def function_contract():
    generated=entry_texts();pins={}
    def funcs(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    for label,original,entry in (("prepare",ENTRIES/"isaac_prepare_ijcas_hospital_h3b_reverse_force_prefix.py",PREPARE),
            ("primary",ENTRIES/"isaac_run_ijcas_hospital_h3b_reverse_force_prefix.py",PRIMARY),("observer",static_baseline.OBSERVER,OBSERVER)):
        if ast.dump(ast.parse(entry.read_text()),include_attributes=False)!=ast.dump(ast.parse(generated[label]),include_attributes=False):
            raise ValueError("Actual prefix observer entry differs beyond explicit metadata/observer changes")
        if label!="prepare":
            names=("physics_step","contacts","angles","verify_gains","set_hold","command","measure","environmental_guard","sample") if label=="primary" else ("step","contacts","remember")
            a,b=funcs(original.read_text()),funcs(entry.read_text())
            if any(a[n]!=b[n] for n in names):raise ValueError("Original control/physical guard/observation callback changed")
        pins.update({str(p):sha(p) for p in (original,entry)})
    # All literal entry dependencies must exist, including provenance-only paths.
    for entry in (PREPARE,PRIMARY,OBSERVER):
        for node in ast.walk(ast.parse(entry.read_text())):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith(".py"):
                name=Path(node.value).name
                if name.startswith("isaac_") and not (ENTRIES/name).is_file():
                    raise ValueError("Missing entry dependency: "+name)
    assigned={n.targets[0].id for n in ast.walk(ast.parse(PRIMARY.read_text()))
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    if "ijcas_hospital_h3b_prefix_response_pilot_task" not in assigned:
        raise ValueError("Observer/primary task dispatch key differs")
    return dict(full_entries_match_declared_noncontrol_transformations=True,actual_prefix_control_safety_and_cleanup_callbacks_AST_unchanged=True,
        six_wheel_and_native_contact_passive_callbacks_AST_unchanged=True,input_sha256=pins)


def load_protocol():
    record=json.loads(PROTOCOL_PATH.read_bytes());verify_sources(record);verify_files(record["input_sha256"])
    if (record.get("version")!=VERSION or record.get("status")!="SAME_ACTUAL_PREFIX_PASSIVE_OBSERVATION_DESIGN_ADMITTED_FRESH_STATIC_SDK_PENDING"
            or record["controller_contract"]!=function_contract() or record["small_S_models_cover_whole_H3_legs"]!=0
            or record["M1_M2_M3_models_deployed"] is not False or record["whole_H3_drive_admission"] is not False):
        raise ValueError("Same actual-prefix observation only; original candidate/whole-route HOLDs retained")
    return record

