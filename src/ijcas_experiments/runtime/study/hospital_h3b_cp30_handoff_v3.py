"""Same-phase static handoff for CP30; NOT relaxed state/motion acceptance.

V1/v2 and their failed pre-launch checks remain preserved. The current fresh
static readback supplies the actual PRE-engagement body and twelve joints;
the historical post-Play sample remains diagnostic only. No state is assigned
and no exact historical static twelve-joint match is claimed (not recorded).
The primary v2 controller/physics/guards and passive measurement math remain
unchanged. This correction does not authorize any whole-H3 replay.
"""

import ast
import json
from pathlib import Path

import hospital_h3b_cp30_runtime_v2 as base
from hospital_h3b_reverse_paused_forces import validate_state as original_validate_state
from hospital_h3b_cp30_prehold_phase_audit import compare
from hospital_static_gate import evaluate_static


VERSION = "H3B_CP30_same_phase_actual_static_handoff_v3"
CASE, BASE, COHORT = base.CASE, base.BASE, base.COHORT
REFERENCE, OUTPUT, OBSERVER_OUTPUT = base.REFERENCE, base.OUTPUT, base.OBSERVER_OUTPUT
PRIMARY, ENTRIES = base.PRIMARY, base.ENTRIES
OBSERVER = ENTRIES / "isaac_run_ijcas_hospital_h3b_cp30_instrumented_v3.py"
PREPARATION_PROTOCOL_PATH = base.PROTOCOL_PATH
PROTOCOL_PATH = BASE / "protocol_r03/current_static_handoff_admission.json"
READBACK = BASE / "prehold_diagnostics/readback_r01.json"
PREPARATION = base.GATE / "hospital_h3b_cp30_observation_preparation/prepare_r01.json"
sha, verify_files, verify_sources, read_trace = base.sha, base.verify_files, base.verify_sources, base.read_trace
vector, wheel_body_snapshot = base.vector, base.wheel_body_snapshot
friction_snapshot, FRICTION_PERIOD_S, coverage = base.friction_snapshot, base.FRICTION_PERIOD_S, base.coverage
function_contract = base.function_contract
STATE_TOLERANCES_UNCHANGED = dict(position_m=.001, heading_deg=.1, all_twelve_direct_joint_rad=.001)


def load_receipt():
    record = json.loads(READBACK.read_bytes())
    verify_sources(record)
    verify_files(record["input_sha256"])
    if (record.get("status") != "CP30_STATIC_VS_POST_ENGAGEMENT_PHASE_DIFFERENCES_RECORDED_NO_DRIVE"
            or record.get("stage_root_session_timeline_body_and_joints_unchanged") is not True
            or record.get("all_twelve_joint_positions_read_NOT_assigned") is not True
            or record.get("new_PhysX_executed") is not False
            or record.get("component_drive_admission") is not False
            or record.get("state_or_acceptance_threshold_modified") is not False
            or record.get("original_CP30_reference") != str(REFERENCE)
            or record.get("original_CP30_verdict_unchanged") != "PASS"):
        raise ValueError("Exact read-only actual PRE-engagement twelve-joint receipt required")
    static_path = Path(record["actual_static_result"])
    static = json.loads(static_path.read_bytes())
    verify_sources(static)
    rows = read_trace(static_path.with_suffix(".csv"))
    evaluated = evaluate_static(rows, static["start_xyyaw"], static["simulation_mass_kg"],
        static["contact_report_counts_by_category"], static["contact_callback_errors"])
    if static.get("verdict") != "PASS" or not all(static["checks"].values()) or evaluated["verdict"] != "PASS":
        raise ValueError("Current actual static PASS must reproduce, not a motion endpoint")
    reference = json.loads(REFERENCE.read_bytes())
    first = read_trace(REFERENCE.with_suffix(".csv"))[0]
    names = list(record["live_all_twelve_joint_positions_rad"])
    recomputed = compare(names, [record["live_all_twelve_joint_positions_rad"][n] for n in names],
        record["live_root_transform"], first, reference["observed_prehold_world_xyyaw"], rows[-1], .001)
    if any(record.get(k) != value for k, value in recomputed.items()):
        raise ValueError("Saved phase/body/per-joint differences do not reproduce")
    flagged = [r for r in recomputed["per_joint_differences"] if r["exceeds_old_equality_0p001rad"]]
    if (recomputed["old_wrong_phase_equality_would_pass"] is not False
            or len(flagged) != 4 or any(r["group"] != "wheel_rotation" or "caster_to_wheel" not in r["joint"] for r in flagged)
            or recomputed["new_static_BODY_matches_existing_paused_xy_tolerance"] is not True):
        raise ValueError("This addendum applies only to the recorded four wheel-spin cross-phase mismatch")
    old_static_csv = Path(reference["static_reference"]).with_suffix(".csv")
    if static_path.with_suffix(".csv").read_bytes() != old_static_csv.read_bytes():
        raise ValueError("Recorded fresh and original static observations differ")
    preparation = json.loads(PREPARATION.read_bytes())
    verify_sources(preparation)
    verify_files(preparation["input_sha256"])
    if (preparation.get("case") != CASE
            or preparation.get("same_condition_observation_protocol_sha256") != sha(PREPARATION_PROTOCOL_PATH)):
        raise ValueError("Preserve the existing v2 preparation/static lineage")
    prep = static["physics_preparation_snapshot"]
    verify_files({static["physics_preparation"]: static["physics_preparation_sha256"]})
    if json.loads(Path(static["physics_preparation"]).read_bytes()) != prep:
        raise ValueError("Static physics snapshot changed")
    for key in ("environment_condition", "source_stage_sha256", "robot_reference_sha256", "start_xyyaw", "component_end_xyyaw",
                "simulation_mass_kg", "live_support_readback", "support_floor", "disabled_legacy_floor_colliders"):
        if prep[key] != reference["physics_preparation_snapshot"][key]:
            raise ValueError("Original prescribed source/start/physics condition changed: " + key)
    return record, static


def validate_handoff(receipt, static_path, transform, names, positions, settings):
    """Same old tolerances, now applied to the actual SAME pre-hold phase."""
    if Path(static_path) != Path(receipt["actual_static_result"]):
        raise ValueError("The actual static receipt changed; keep PAUSED and review, never reset to match")
    if set(names) != set(receipt["live_all_twelve_joint_positions_rad"]):
        raise ValueError("Current twelve-joint set differs from the recorded actual static state")
    pose = receipt["live_world_xyyaw"]
    expected = dict(x_m=pose[0], y_m=pose[1], yaw_rad=pose[2], **{
        "joint_" + name + "_rad": value for name, value in receipt["live_all_twelve_joint_positions_rad"].items()})
    try:
        result = original_validate_state(dict(last_recorded_state=expected), transform, names, positions)
    except ValueError as error:
        raise ValueError("Current PRE-hold body/twelve joints differ from their SAME-phase readback; keep PAUSED, no reset/reopen/Play") from error
    if settings != receipt["live_static_joint_settings"]:
        raise ValueError("Current static gains/limits/velocity targets differ from the saved unchanged baseline")
    return dict(result, reference_phase="CURRENT_ACTUAL_STATIC_PASS_BEFORE_HOLD_ENGAGEMENT",
                reference_readback=str(READBACK), reference_readback_sha256=sha(READBACK),
                identical_old_state_equality_tolerances=STATE_TOLERANCES_UNCHANGED,
                historical_static_twelve_joint_equality_NOT_claimed=True)


def observer_contract():
    old = base.OBSERVER.read_text()
    new = OBSERVER.read_text()
    normalized = new.replace("hospital_h3b_cp30_handoff_v3.py", "hospital_h3b_cp30_runtime_v2.py")
    normalized = normalized.replace("isaac_run_ijcas_hospital_h3b_cp30_instrumented_v3.py", "isaac_run_ijcas_hospital_h3b_cp30_instrumented_v2.py")
    begin, end = "    # SAME_PHASE_HANDOFF_BEGIN\n", "    # SAME_PHASE_HANDOFF_END\n"
    left, separator, rest = normalized.partition(begin)
    if not separator or end not in rest:
        raise ValueError("Explicit same-phase handoff replacement block missing")
    old_begin = old.index('    state_rules=loader(scripts/"hospital_h3b_reverse_paused_forces.py")')
    old_end = old.index('    patterns=[rules["ROBOT_ROOT"]', old_begin)
    normalized = left + old[old_begin:old_end] + rest.split(end, 1)[1]
    normalized = normalized.replace('helper["PREPARATION_PROTOCOL_PATH"]', 'helper["PROTOCOL_PATH"]')
    normalized = normalized.replace('"initial_actual_static_body_and_twelve_joints_match":retained,',
                                    '"initial_saved_CP30_body_and_twelve_joints_match":retained,')
    for begin, end in (("    # SAME_PHASE_RECHECK_BEGIN\n", "    # SAME_PHASE_RECHECK_END\n"),
                       ("            # SAME_PHASE_PROVENANCE_BEGIN\n", "            # SAME_PHASE_PROVENANCE_END\n")):
        left, separator, rest = normalized.partition(begin)
        if not separator or end not in rest:
            raise ValueError("Explicit handoff recheck/provenance block missing")
        normalized = left + rest.split(end, 1)[1]
    if ast.dump(ast.parse(normalized), include_attributes=False) != ast.dump(ast.parse(old), include_attributes=False):
        raise ValueError("Observer changed beyond the disclosed handoff/metadata correction")
    return dict(passive_recording_callbacks_and_primary_dispatch_unchanged=True,
                primary_v2_entry_byte_unchanged=True, state_tolerances_unchanged=STATE_TOLERANCES_UNCHANGED,
                input_sha256={str(p): sha(p) for p in (base.OBSERVER, OBSERVER, PRIMARY)})


def load_protocol():
    parent = base.load_protocol()
    record = json.loads(PROTOCOL_PATH.read_bytes())
    verify_sources(record)
    verify_files(record["input_sha256"])
    if (record.get("version") != VERSION
            or record.get("status") != "SAME_CP30_CURRENT_STATIC_HANDOFF_ADMITTED_LIVE_AND_SDK_PENDING"
            or record.get("parent_protocol") != str(PREPARATION_PROTOCOL_PATH)
            or record.get("parent_protocol_sha256") != sha(PREPARATION_PROTOCOL_PATH)
            or record.get("protocol") != parent["protocol"]
            or record.get("component_design_cases") != parent["component_design_cases"]
            or record.get("controller_contract") != function_contract()
            or record.get("observer_contract") != observer_contract()
            or record.get("actual_static_readback") != str(READBACK)
            or record.get("actual_static_readback_sha256") != sha(READBACK)
            or record.get("state_equality_tolerances") != STATE_TOLERANCES_UNCHANGED
            or record.get("whole_H3_drive_admission") is not False):
        raise ValueError("Exact same-phase CP30 correction admission required")
    load_receipt()
    return record
