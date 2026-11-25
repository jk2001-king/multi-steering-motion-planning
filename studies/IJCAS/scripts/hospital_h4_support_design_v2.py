"""Serialization-only adapter; v1 geometry, tolerances and source pins retained.

JSON arrays become Python lists; v1's tuple-containing in-memory geometry did
not compare equal to its own JSON snapshot. Canonicalize BOTH sides through
JSON, retaining exact numbers/keys and every original scientific constraint.
This does not waive a geometric/state/material check or overwrite v1 evidence.
"""

import json
from pathlib import Path

import hospital_h4_support_design as original


ROOT, GATE, floor = original.ROOT, original.GATE, original.floor
PATCH_ROOT, PATCH_MESH = original.PATCH_ROOT, original.PATCH_MESH
AREA_TOLERANCE_M2 = original.AREA_TOLERANCE_M2
MAXIMUM_PLANAR_JOIN_WELD_M = original.MAXIMUM_PLANAR_JOIN_WELD_M
OUTPUT = GATE/'hospital_h4_support/design_r02'
TARGET = OUTPUT/'support_design.json'
LIVE = GATE/'hospital_h4_support/session_r02/live_support_preparation.json'
ENTRY = ROOT/'isaac/scripts/isaac_prepare_ijcas_hospital_h4_local_support_v2.py'
VERSION = original.VERSION+'_JSON_array_snapshot_v2'
area, ccw, rectangle = original.area, original.ccw, original.rectangle
clip_halfplane, subtract_convex, uncovered = original.clip_halfplane, original.subtract_convex, original.uncovered
floor_triangles, ear_triangles, solid_mesh = original.floor_triangles, original.ear_triangles, original.solid_mesh


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def build():
    h = floor.inventory.retained.completion
    first = json.loads(original.TARGET.read_bytes())
    h.verify_files(first['input_sha256']); h.verify_sources(first)
    if first != canonical(original.build()):
        raise ValueError('Original geometry/model/material/state/input snapshot differs beyond JSON array serialization')
    report = canonical(first)
    report.update(version=VERSION,
        serialization_only_revision_preserves_exact_v1_geometry_numbers_and_checks=True,
        v1_loader_tuple_list_equality_failure_is_NOT_a_physical_or_geometric_failure=True,
        original_design=str(original.TARGET), original_design_sha256=h.sha(original.TARGET),
        original_entry_never_admitted_or_executed_as_a_physical_test=True)
    sources = (Path(__file__), ENTRY)
    report['input_sha256'].update({str(path): h.sha(path) for path in (*sources, original.TARGET)})
    report['source_sha256'].update({path.name: h.sha(path) for path in sources})
    return report


def load_design():
    h = floor.inventory.retained.completion
    report = json.loads(TARGET.read_bytes()); h.verify_files(report['input_sha256']); h.verify_sources(report)
    if report != canonical(build()):
        raise ValueError('Local support design differs; no substitute geometry or tolerance waiver')
    return report
