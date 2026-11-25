"""CP30 runtime-only import repair; no offline analysis imports or control changes.

V1 helpers/entries/protocol remain byte-preserved. This module loads only
hospital_* runtime siblings through the frozen private loader. No sys.path,
sys.meta_path, sys.modules or builtins mutation, model fit, or motion here.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import ast
import json
from pathlib import Path

import numpy as np

from hospital_h3_turn_planning import GATE, WORKSPACE, verify_sources
from hospital_h3_flush_support import sha, verify_files
from hospital_h3b_alignment_audit import telemetry_snapshot
from hospital_h3b_rolling_observation import friction_snapshot, FRICTION_PERIOD_S, coverage as old_coverage
from hospital_h3_instrumentation import vector
from hospital_h3b_mode_speed_evidence import read_trace


VERSION = "H3B_same_CP30_runtime_import_boundary_only_v2"
CASE = "coordinated_counterphase_30mm_s"
BASE = GATE / "hospital_h3b_cp30_observation"
COHORT = BASE / "cohort_01"
OUTPUT = COHORT / CASE
OBSERVER_OUTPUT = COHORT / "instrumentation"
BASE_PROTOCOL_PATH = BASE / "protocol_r01/prospective_protocol.json"
PROTOCOL_PATH = BASE / "protocol_r02/runtime_import_addendum.json"
REFERENCE = GATE / "hospital_h3b_mode_speed_components/cohort_01" / CASE / ("hospital_h3b_mode_speed_" + CASE + "_r01.json")
ENTRIES = _ijcas_path(WORKSPACE, 'isaac/scripts')
PREPARE = ENTRIES / "isaac_prepare_ijcas_hospital_h3b_cp30_observation_v2.py"
PRIMARY = ENTRIES / "isaac_run_ijcas_hospital_h3b_cp30_observation_primary_v2.py"
OBSERVER = ENTRIES / "isaac_run_ijcas_hospital_h3b_cp30_instrumented_v2.py"
WHEELS = ("front_active_drive_wheel_link", "rear_active_drive_wheel_link",
          "front_left_caster_wheel_link", "front_right_caster_wheel_link",
          "rear_left_caster_wheel_link", "rear_right_caster_wheel_link")


def function_contract():
    pairs = ((ENTRIES / "isaac_run_ijcas_hospital_h3b_mode_speed_component.py", PRIMARY),)
    names = ("physics_step", "contacts", "angles", "verify_gains", "set_hold", "command", "measure", "environmental_guard", "sample")
    inputs = {}
    for original, new in pairs:
        maps = [{n.name: n for n in ast.walk(ast.parse(p.read_text()))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} for p in (original, new)]
        for name in names:
            if ast.dump(maps[0][name], include_attributes=False) != ast.dump(maps[1][name], include_attributes=False):
                raise ValueError("CP30 observation changed a control/guard function: " + name)
        # Entire entry must be a disclosed output/provenance-only copy; this
        # also pins loops/control outside the named nested functions.
        text = new.read_text()
        original_text = original.read_text()
        normalized = text.replace("isaac_run_ijcas_hospital_h3b_cp30_observation_primary_v2.py", "isaac_run_ijcas_hospital_h3b_mode_speed_component.py")
        normalized = normalized.replace("isaac_prepare_ijcas_hospital_h3b_cp30_observation_v2.py", "isaac_prepare_ijcas_hospital_h3b_mode_speed_component.py")
        normalized = normalized.replace("hospital_h3b_cp30_observation/cohort_01", "hospital_h3b_mode_speed_components/cohort_01")
        normalized = normalized.replace("hospital_h3b_cp30_observation_", "hospital_h3b_mode_speed_")
        normalized = normalized.replace("ijcas_hospital_h3b_cp30_observation_primary_task", "ijcas_hospital_h3b_mode_speed_component_task")
        for begin, end in (("    # CP30_OBSERVATION_ADMISSION_BEGIN\n", "    # CP30_OBSERVATION_ADMISSION_END\n"),
                           ("        # CP30_OBSERVATION_PROVENANCE_BEGIN\n", "        # CP30_OBSERVATION_PROVENANCE_END\n")):
            left, separator, rest = normalized.partition(begin)
            if not separator or end not in rest:
                raise ValueError("Explicit observer admission/provenance block missing")
            normalized = left + rest.split(end, 1)[1]
        if ast.dump(ast.parse(normalized), include_attributes=False) != ast.dump(ast.parse(original_text), include_attributes=False):
            raise ValueError("New CP30 entry differs beyond declared cohort/admission/provenance")
        inputs.update({str(p): sha(p) for p in (original, new)})
    return dict(entire_control_flow_equal_after_declared_noncontrol_changes=True,
                control_guard_and_cleanup_functions_AST_equal=True, matched_functions=list(names), input_sha256=inputs)


def load_protocol():
    record = json.loads(PROTOCOL_PATH.read_bytes())
    original = json.loads(BASE_PROTOCOL_PATH.read_bytes())
    verify_sources(record)
    verify_sources(original)
    verify_files(record["input_sha256"])
    verify_files(original["input_sha256"])
    if (record.get("version") != VERSION
            or record.get("status") != "SAME_CP30_IMPORT_REPAIR_ADMITTED_FRESH_STATIC_AND_SDK_PENDING"
            or record.get("base_protocol") != str(BASE_PROTOCOL_PATH)
            or record.get("base_protocol_sha256") != sha(BASE_PROTOCOL_PATH)
            or record.get("protocol") != original["protocol"]
            or record.get("component_design_cases") != original["component_design_cases"]
            or original.get("status") != "ONE_SAME_CP30_OBSERVATION_DESIGN_ADMITTED_FRESH_STATIC_AND_SDK_PENDING"
            or original.get("whole_H3_drive_admission") is not False
            or record.get("whole_H3_drive_admission") is not False
            or record.get("controller_contract") != function_contract()
            or record.get("runtime_import_only_no_model_or_control_change") is not True):
        raise ValueError("Unchanged pre-motion CP30 design plus explicit import-only repair required")
    if not record["component_design_cases"] or not all(all(c["checks"].values()) for c in record["component_design_cases"]):
        raise ValueError("Original CP30 model/strict-envelope design checks must remain complete")
    return record


def origin_velocity(transform, velocity, local_com):
    transform, velocity, local_com = (np.asarray(v, float) for v in (transform, velocity, local_com))
    if (transform.shape != (7,) or velocity.shape != (6,) or local_com.shape != (3,)
            or not all(np.isfinite(v).all() for v in (transform, velocity, local_com))):
        raise ValueError("Finite measured link transform/velocity/COM required")
    q = transform[3:].copy()
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.) > 1e-3:
        raise ValueError("Invalid measured link quaternion")
    q /= norm
    xyz, w = q[:3], q[3]
    world_offset = local_com + 2 * np.cross(xyz, np.cross(xyz, local_com) + w * local_com)
    return velocity[:3] - np.cross(velocity[3:], world_offset)


def wheel_body_snapshot(view, time_s):
    row = telemetry_snapshot(view, time_s)
    names = list(view.shared_metatype.link_names)
    n = len(names)
    if view.count != 1 or len(set(names)) != n or any(name not in names for name in WHEELS):
        raise ValueError("One articulation and all six expected wheel links required")
    transforms = np.asarray(vector(view.get_link_transforms(), n * 7)).reshape(n, 7)
    velocities = np.asarray(vector(view.get_link_velocities(), n * 6)).reshape(n, 6)
    coms = np.asarray(vector(view.get_coms(), n * 7)).reshape(n, 7)
    for name in WHEELS:
        i = names.index(name)
        for key, value in zip(("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"), transforms[i]):
            row[name + "__" + key] = float(value)
        for key, value in zip(("com_vx_m_s", "com_vy_m_s", "com_vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"), velocities[i]):
            row[name + "__" + key] = float(value)
        for axis, value, center in zip("xyz", origin_velocity(transforms[i], velocities[i], coms[i, :3]), coms[i, :3]):
            row[name + "__origin_v" + axis + "_m_s"] = float(value)
            row[name + "__com_local_" + axis + "_m"] = float(center)
    return row


def coverage(joints, friction, anchors, contacts, primary, errors):
    result = old_coverage(joints, friction, anchors, contacts, primary, errors)
    wheels = bool(joints) and all(all(name + "__qw" in row and name + "__origin_vx_m_s" in row for name in WHEELS) for row in joints)
    complete = result["status"] == "RECORDED_FOR_CAUSAL_REVIEW_NOT_A_ROLLING_OR_TRAVERSAL_PASS" and wheels
    result.update(status="CP30_PASSIVE_DATA_RECORDED_FOR_RESPONSE_REVIEW_NOT_MODEL_VALIDATION" if complete else "INCOMPLETE_CP30_OBSERVATION_DO_NOT_INFER_CAUSE",
        six_wheel_transforms_and_COM_and_origin_velocities_recorded=wheels,
        original_motion_verdict_is_authoritative=True, model_validation_or_H3_success_implied=False)
    return result
