"""One NEW matched-length straight CP30 identification component.

Only the .65m input curvature changes to zero. No fitted model runs on the
robot; original ramp/brake, control, guards, hold/cleanup and passive SDK math
are retained. This is identification data, NOT validation of a new response
model or H3. Runtime dependencies are hospital_* only (private-loader safe).
"""

import ast
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_mode_speed_review as old
import hospital_h3b_cp30_handoff_v3 as handoff
from hospital_h3b_cp30_runtime_v2 import (wheel_body_snapshot, vector, friction_snapshot,
    FRICTION_PERIOD_S, coverage as old_coverage)
from hospital_h3b_reverse_paused_forces import validate_state
from hospital_h3_turn_planning import GATE, WORKSPACE, verify_sources
from hospital_h3_flush_support import sha, verify_files
from hospital_static_gate import evaluate_static
from hospital_h3b_mode_speed_evidence import read_trace


VERSION = "H3B_matched_length_straight_CP30_identification_v1"
ROLE = "MATCHED_LENGTH_STRAIGHT_RESPONSE_IDENTIFICATION_NOT_MODEL_VALIDATION"
CASE = "coordinated_counterphase_30mm_s"
DISTANCE = .650
KNOTS = tuple((x, 0.) for x, _ in old.KNOTS)
LEGS = ((1, DISTANCE),)
BASE = GATE / "hospital_h3b_neutral_response"
COHORT = BASE / "cohort_01"
OUTPUT, OBSERVER_OUTPUT = COHORT / CASE, COHORT / "instrumentation"
PROTOCOL_PATH = BASE / "protocol_r01/neutral_response_design.json"
PREPARATION_PROTOCOL_PATH = PROTOCOL_PATH
REFERENCE = handoff.REFERENCE
READBACK = handoff.READBACK
STATE_TOLERANCES_UNCHANGED = handoff.STATE_TOLERANCES_UNCHANGED
ENTRIES = WORKSPACE / "isaac/scripts"
PREPARE = ENTRIES / "isaac_prepare_ijcas_hospital_h3b_neutral_response.py"
PRIMARY = ENTRIES / "isaac_run_ijcas_hospital_h3b_neutral_response_primary.py"
OBSERVER = ENTRIES / "isaac_run_ijcas_hospital_h3b_neutral_response_instrumented.py"
RESULT = OUTPUT / "hospital_h3b_neutral_response_coordinated_counterphase_30mm_s_r01.json"
POLICY, READINESS, ACCEPTANCE = old.POLICY, old.READINESS, old.ACCEPTANCE
GUARDS, COMMAND_LIMITS, BAD_CONTACTS = old.GUARDS, old.COMMAND_LIMITS, old.BAD_CONTACTS
ROBOT_ROOT, WHEELS = old.ROBOT_ROOT, old.WHEELS
classify_contact, relative_pose = old.classify_contact, old.relative_pose
wheel_crossing_checks, motion_stalled = old.wheel_crossing_checks, old.motion_stalled
direct_joint_targets, steering_geometry = old.direct_joint_targets, old.steering_geometry
ForwardStopProfile = old.ForwardStopProfile
FIRST_CASE = CASE


def require_case(case):
    if case != CASE:
        raise ValueError("Only the separate straight CP30 identification case is admitted")
    return "coordinated_counterphase", .030


def curvature_at(progress, phase):
    if not math.isfinite(progress):
        raise ValueError("Finite fixed-start progress required")
    return 0.


def nominal_component(mode, step=.001):
    if mode != "coordinated_counterphase" or not 0 < step <= .005:
        raise ValueError("Only the fixed straight CP30 component is supported")
    return [[float(x), 0., 0.] for x in np.linspace(0., DISTANCE, math.ceil(DISTANCE/step)+1)]


def component_goal(start, mode):
    require_case(CASE)
    if mode != "coordinated_counterphase":
        raise ValueError("Unchanged counterphase configuration required")
    return start[0]+DISTANCE*math.cos(start[2]), start[1]+DISTANCE*math.sin(start[2]), start[2]


def evaluate_readiness(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, events, case):
    require_case(case)
    evaluated = old.evaluate_readiness(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, events, case)
    checks = dict(evaluated["checks"])
    # ONLY S-excitation obligations are replaced, never physical safety gates.
    for key in ("moderate_S_and_single_hold_completed", "observed_active_axle_both_steering_signs", "both_reference_curvature_signs"):
        del checks[key]
    checks.update(straight_and_single_hold_completed=True,
        prescribed_zero_reference_curvature=all(abs(r["requested_curvature_1_m"]) < 1e-12 for r in rows),
        prescribed_zero_steering_commands=all(max(abs(r["command_front_rad"]), abs(r["command_rear_rad"])) < 1e-6 for r in rows))
    complete = all(checks.values())
    return dict(evaluated, checks=checks, verdict="PASS" if complete else "FAIL", mode_speed_component_verified=complete,
                neutral_response_identification_motion_verified=complete)


def load_receipt():
    return handoff.load_receipt()


def validate_handoff(receipt, static_path, transform, names, positions, settings):
    static_path = Path(static_path)
    record = json.loads(static_path.read_bytes())
    verify_sources(record)
    rows = read_trace(static_path.with_suffix(".csv"))
    if (record.get("verdict") != "PASS" or not all(record["checks"].values())
            or evaluate_static(rows, record["start_xyyaw"], record["simulation_mass_kg"],
                record["contact_report_counts_by_category"], record["contact_callback_errors"])["verdict"] != "PASS"
            or static_path.with_suffix(".csv").read_bytes() != Path(receipt["actual_static_result"]).with_suffix(".csv").read_bytes()):
        raise ValueError("Fresh actual static PASS must match the recorded CP30 static baseline; no state reset or tolerance waiver")
    pose = receipt["live_world_xyyaw"]
    if set(names) != set(receipt["live_all_twelve_joint_positions_rad"]):
        raise ValueError("All twelve original joints required")
    expected = dict(x_m=pose[0], y_m=pose[1], yaw_rad=pose[2], **{
        "joint_"+name+"_rad": value for name, value in receipt["live_all_twelve_joint_positions_rad"].items()})
    try:
        result = validate_state(dict(last_recorded_state=expected), transform, names, positions)
    except ValueError as error:
        raise ValueError("Fresh neutral static body/twelve joints differ from the SAME-phase CP30 baseline; keep PAUSED, do not reset to match") from error
    if settings != receipt["live_static_joint_settings"]:
        raise ValueError("Static gains/limits/velocity targets differ from the frozen CP30 condition")
    return dict(result, reference_phase="MATCHED_ACTUAL_STATIC_BEFORE_HOLD_ENGAGEMENT",
        actual_static_result=str(static_path), actual_static_sha256=sha(static_path),
        CP30_same_phase_baseline=str(READBACK), CP30_same_phase_baseline_sha256=sha(READBACK),
        state_tolerances_unchanged=STATE_TOLERANCES_UNCHANGED,
        no_historical_post_Play_equality_and_no_state_assignment=True)


def coverage(*args):
    result = old_coverage(*args)
    complete = result["status"] == "CP30_PASSIVE_DATA_RECORDED_FOR_RESPONSE_REVIEW_NOT_MODEL_VALIDATION"
    result["status"] = "NEUTRAL_CP30_IDENTIFICATION_DATA_RECORDED_NOT_MODEL_VALIDATION" if complete else "INCOMPLETE_NEUTRAL_RESPONSE_DATA"
    return result


def entry_texts():
    """Explicit source transformations; all old files stay byte-preserved."""
    templates = {
        "prepare": ENTRIES/"isaac_prepare_ijcas_hospital_h3b_cp30_observation_v2.py",
        "primary": ENTRIES/"isaac_run_ijcas_hospital_h3b_cp30_observation_primary_v2.py",
        "observer": ENTRIES/"isaac_run_ijcas_hospital_h3b_cp30_instrumented_v3.py"}
    common = [
        ("hospital_h3b_cp30_runtime_v2.py", "hospital_h3b_neutral_response.py"),
        ("hospital_h3b_cp30_handoff_v3.py", "hospital_h3b_neutral_response.py"),
        ("hospital_h3b_mode_speed_review.py", "hospital_h3b_neutral_response.py"),
        (templates["prepare"].name, PREPARE.name), (templates["primary"].name, PRIMARY.name), (templates["observer"].name, OBSERVER.name),
        ("hospital_h3b_cp30_observation_preparation", "hospital_h3b_neutral_response_preparation"),
        ("hospital_h3b_cp30_observation/cohort_01", "hospital_h3b_neutral_response/cohort_01"),
        ("hospital_h3b_cp30_observation_", "hospital_h3b_neutral_response_"),
        ("ijcas_hospital_h3b_cp30_observation_primary_task", "ijcas_hospital_h3b_neutral_primary_task"),
        ("ijcas_hospital_h3b_cp30_observer_task", "ijcas_hospital_h3b_neutral_observer_task"),
        ("ijcas_hospital_h3b_mode_speed_component_task", "ijcas_hospital_h3b_neutral_response_task"),
        ("ijcas_hospital_h3b_mode_speed_condition", "ijcas_hospital_h3b_neutral_component_condition"),
        ("H3B_mode_speed_component_v1", "H3B_neutral_straight_component_v1"),
        ("ijcas_hospital_h3b_cp30_observer_condition", "ijcas_hospital_h3b_neutral_observer_condition"),
        ("H3B_one_same_CP30_friction_observation", "H3B_neutral_straight_identification_observation"),
        ("ijcas_hospital_h3b_cp30_observer_ready", "ijcas_hospital_h3b_neutral_observer_ready"),
        ("PASSIVE_OBSERVER_PREFLIGHT_COMPLETE", "NEUTRAL_OBSERVER_PREFLIGHT_COMPLETE"),
        ("ONE_SAME_CONDITION_CP30_WHEEL_BODY_CONTACT_OBSERVATION_NO_FIT", "NEUTRAL_STRAIGHT_CP30_IDENTIFICATION_NOT_VALIDATION"),
        ("same_condition_observation_protocol", "neutral_identification_protocol"),
        ('"original_controller_source_sha256":reference["source_sha256"]',
         '"comparison_CP30_controller_source_sha256":reference["source_sha256"]'),
        ('report["original_controller_verdict"]=original["verdict"]',
         'report["original_controller_verdict"]=original["verdict"]\n            report["original_controller_source_sha256"]=original["source_sha256"]'),
        ('"actual_prehold_static_readback":str(helper["READBACK"])',
         '"CP30_baseline_static_readback":str(helper["READBACK"])'),
        ('"actual_prehold_static_readback_sha256":support["sha"](helper["READBACK"])',
         '"CP30_baseline_static_readback_sha256":support["sha"](helper["READBACK"])'),
        ("H3B_moderate_S_existing_mode_speed_component", "H3B_matched_straight_response_identification"),
        ("NEW moderate-S", "NEW matched-length straight"), ("MODERATE-S", "STRAIGHT IDENTIFICATION"),
        ("NEW forward 0.65 m moderate S", "NEW forward 0.65 m STRAIGHT"),
        ("NEW 0.65 m / +/-0.25 curvature forward component", "NEW 0.65 m / ZERO curvature forward component"),
        ("H3-B ONE SAME CP30 observation: original S, speed, caster seed, gains, materials and all safety gates unchanged.",
         "H3-B NEW neutral input: ONLY curvature -> zero; same 0.65m/30mm_s/ramp/gains/guards and prescribed caster seed. Identification, not model validation."),
        ("historical_CP30_static_twelve_joint_equality_NOT_claimed", "CP30_actual_static_baseline_twelve_joint_match_required"),
        ("Historical post-Play first row is diagnostic ONLY; no pose/joint/caster reset or historical static joint equality claim.",
         "Fresh actual static matched to recorded CP30 PRE-hold baseline; no post-Play comparison, caster assignment or tolerance waiver."),
        ("same-condition CP30 reproduction", "matched-length straight CP30 identification"),
        ("SAME-CONDITION CP30 WHEEL/BODY/CONTACT", "NEUTRAL CP30 IDENTIFICATION WHEEL/BODY/CONTACT"),
        ("same-condition observation", "neutral identification"),
    ]
    result = {}
    for key, path in templates.items():
        text = path.read_text()
        for before, after in common:
            text = text.replace(before, after)
        if key == "prepare":
            text = text.replace("globals().pop('ijcas_hospital_h3b_mode_speed_review', '')", "globals().pop('ijcas_hospital_h3b_mode_speed_review', str(h['PROTOCOL_PATH']))")
        result[key] = text
    return result


def function_contract():
    expected = entry_texts()
    paths = dict(prepare=PREPARE, primary=PRIMARY, observer=OBSERVER)
    for key, path in paths.items():
        if ast.dump(ast.parse(path.read_text()), include_attributes=False) != ast.dump(ast.parse(expected[key]), include_attributes=False):
            raise ValueError("Entry changed beyond declared reference/cohort/admission/provenance substitutions: "+key)
    original = ENTRIES/"isaac_run_ijcas_hospital_h3b_cp30_observation_primary_v2.py"
    names = ("physics_step", "contacts", "angles", "verify_gains", "set_hold", "command", "measure", "environmental_guard", "sample")
    trees = [{n.name: n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in (original, PRIMARY)]
    for name in names:
        if ast.dump(trees[0][name], include_attributes=False) != ast.dump(trees[1][name], include_attributes=False):
            raise ValueError("Original control/guard callback changed: "+name)
    source = ENTRIES/"isaac_run_ijcas_hospital_h3b_cp30_instrumented_v3.py"
    callbacks = [{n.name: n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.FunctionDef)} for p in (source, OBSERVER)]
    for name in ("step", "contacts", "remember"):
        if ast.dump(callbacks[0][name], include_attributes=False) != ast.dump(callbacks[1][name], include_attributes=False):
            raise ValueError("Passive observation callback changed: "+name)
    return dict(entire_entry_flow_matches_explicit_substitution_contract=True,
        control_guard_hold_cleanup_callbacks_AST_unchanged=True, passive_recording_callbacks_AST_unchanged=True,
        profile_change="ONLY fixed curvature knots -> zero; same .65m/cap/ramp/static source prescription",
        input_sha256={str(p): sha(p) for p in (*paths.values(), original, source)})


def load_protocol():
    record = json.loads(PROTOCOL_PATH.read_bytes())
    verify_sources(record)
    verify_files(record["input_sha256"])
    if (record.get("version") != VERSION or record.get("role") != ROLE
            or record.get("status") != "NEUTRAL_IDENTIFICATION_COMPONENT_DESIGN_ADMITTED_FRESH_SOURCE_STATIC_SDK_PENDING"
            or record.get("controller_contract") != function_contract()
            or record.get("whole_H3_drive_admission") is not False or record.get("candidate_model_deployed") is not False
            or record.get("fixed_progress_curvature_knots") != [list(k) for k in KNOTS]
            or not record.get("component_design_cases")
            or any(not all(c["model_checks"].values()) for c in record["component_design_cases"])):
        raise ValueError("Frozen neutral IDENTIFICATION-only design is not admitted")
    verify_files({c["trace_file"]: c["trace_sha256"] for c in record["component_design_cases"]})
    handoff.load_receipt()
    return record
