"""Pure helpers for read-only review of an EXISTING hospital corridor corner.

No task start/goal, traversable width, occupancy policy, obstacle placement or
physical readiness is inferred from these bounds. Existing cart pose retained.
"""

import hashlib
import json
import math
from pathlib import Path


ROI = [15., 22., 23., 34.5]  # xmin,xmax,ymin,ymax metres; review area, not task bounds.
REVIEW_PATHS = (
    "/World/Hospital/Geo_M1_DoorWall8",
    "/World/Hospital/Geo_M1_DoorWall7",
    "/World/Hospital/Geo_W_WallDoorCostum_536",
    "/World/Hospital/Geo_M1_SideWall15",
    "/World/Hospital/SM_SupplyCart_02a7",
)
CART = REVIEW_PATHS[-1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intersects_xy(bound, roi=ROI):
    if bound is None:
        return False
    lo, hi = bound["minimum_m"], bound["maximum_m"]
    if (len(lo) != 3 or len(hi) != 3 or len(roi) != 4 or not all(math.isfinite(v) for v in (*lo, *hi, *roi))
            or any(a > b for a, b in zip(lo, hi)) or roi[0] >= roi[1] or roi[2] >= roi[3]):
        raise ValueError("Finite ordered review bounds required")
    return hi[0] >= roi[0] and lo[0] <= roi[1] and hi[1] >= roi[2] and lo[1] <= roi[3]


def validate_live_targets(assets, original_candidates):
    lookup = {row["path"]: row for row in assets}
    original = {row["path"]: row for row in original_candidates}
    if any(path not in lookup or path not in original for path in REVIEW_PATHS):
        raise ValueError("An original review wall/cart is missing; do not substitute a success-selected scene")
    for path in REVIEW_PATHS:
        actual = lookup[path]["world_aligned_visible_bound"]
        expected = original[path]["world_aligned_bound"]
        if actual is None or expected is None or any(abs(a - b) > 1e-5 for key in ("minimum_m", "maximum_m")
                for a, b in zip(actual[key], expected[key])):
            raise ValueError("Existing H4 candidate asset moved or changed: " + path)
    return {path: lookup[path] for path in REVIEW_PATHS}


def verify_continuity_hold(workspace):
    workspace = Path(workspace)
    folder = workspace / "studies/IJCAS/results/hospital_paper_gate/hospital_h3b_cusp_continuity/continuity_r01"
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    if (summary["status"] != "HOLD_NO_CONTINUOUS_CUSP_CONNECTION_IN_BOUNDED_SEARCH_NOT_GLOBAL_INFEASIBILITY"
            or len(summary["searches"]) != 2 or any(row["action_count"] for row in summary["searches"])
            or summary.get("PhysX_executed") is not False):
        raise ValueError("Expected actual recorded two-search HOLD before the agreed H4 geometry fallback")
    inputs = {str(summary_path): sha(summary_path)}
    for name, expected in summary["source_sha256"].items():
        path = workspace / "studies/IJCAS/scripts" / name
        if sha(path) != expected:
            raise ValueError("Recorded H3-B search source changed: " + name)
    for path, expected in summary["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Original H3-B evidence changed: " + path)
    for row in summary["searches"]:
        path = folder / (row["search_variant"] + ".json")
        if sha(path) != row["result_sha256"]:
            raise ValueError("Recorded continuity-search result changed")
        inputs[str(path)] = row["result_sha256"]
    return inputs
