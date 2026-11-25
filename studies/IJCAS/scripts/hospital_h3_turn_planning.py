"""Read-only H3 task/preview/model provenance checks for offline planning."""

import hashlib
import json
import math
from pathlib import Path
import runpy

import numpy as np


HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]
GATE=WORKSPACE/"studies/IJCAS/results/hospital_paper_gate"
SUPPORT=runpy.run_path(str(HERE/"hospital_h3_flush_support.py"))
GEO=runpy.run_path(str(HERE/"hospital_geometry_tools.py"))
DEFAULT_SETUP=GATE/"hospital_h3_turn_setup/setup_r01/hospital_h3_turn_geometry.json"
DEFAULT_PREVIEW=GATE/"hospital_h3_turn_previews/preview_r01/task_preview.json"
MODEL_PLAN=GATE/"planning/hospital_h1_planning_r06/P_dynamics_full_envelope.json"
MODEL_SHA="1d3f6978a5bb3bb6085a2d8882fe3855f1811599197cbc4365a223d874b292ac"


def verify_sources(record):
    for name,digest in record["source_sha256"].items():
        folder=WORKSPACE/("isaac/scripts" if name.startswith("isaac_") else "studies/IJCAS/scripts")
        if SUPPORT["sha"](folder/name)!=digest:
            raise ValueError("Recorded H3 source changed: "+name)


def load_setup(path=DEFAULT_SETUP,preview_path=DEFAULT_PREVIEW):
    path,preview_path=Path(path).resolve(),Path(preview_path).resolve()
    manifest=json.loads(path.read_bytes());preview=json.loads(preview_path.read_bytes())
    if (manifest.get("scenario")!="H3_doorway_immediate_turn" or manifest.get("environment_condition")!=SUPPORT["CONDITION"]
            or manifest.get("purpose")!="H3_FLUSH_DOOR_ENTRY_90DEG_RIGHT_TURN_TASK_CANDIDATE_NOT_STRAIGHT_COMPONENT"
            or not manifest.get("checks") or not all(manifest["checks"].values())
            or manifest.get("footprint")!={"length_m":1.63,"width_m":.965,"clearance_m":.01}):
        raise ValueError("Need the actual H3 turning TASK setup, not a straight component or changed envelope")
    if ((preview_path.parent/"preview_preparation_failure.json").exists()
            or preview.get("scenario")!=manifest["scenario"] or preview.get("environment_condition")!=manifest["environment_condition"]
            or preview.get("status")!="ACTUAL_STOPPED_TASK_PREVIEW_CREATED_USER_REVIEW_PENDING"
            or Path(preview["geometry_manifest"]).resolve()!=path
            or preview["geometry_manifest_sha256"]!=SUPPORT["sha"](path)
            or preview.get("robot_placed_at_original_task_start") is not True
            or preview.get("other_hospital_assets_preserved") is not True
            or preview.get("only_endpoint_markers_not_a_planned_route") is not True):
        raise ValueError("Actual stopped task preview/manifest mismatch or rolled-back preview")
    if preview["start_xyyaw"]!=manifest["start_xyyaw"] or preview["goal_xyyaw"]!=manifest["goal_xyyaw"]:
        raise ValueError("H3 preview and task frame/start/goal mismatch")
    if preview.get("PhysX_executed") is not False or preview.get("source_USD_saved") is not False:
        raise ValueError("Expected a genuine stopped preview, not execution data")
    SUPPORT["verify_files"](manifest["input_sha256"])
    verify_sources(manifest);verify_sources(preview)
    if SUPPORT["sha"](manifest["source_audit"])!=manifest["source_audit_sha256"]:
        raise ValueError("H3 task collision audit changed")
    source_path,source,geometry,_=SUPPORT["load_setup"](Path(manifest["source_geometry_manifest"]).parent,WORKSPACE)
    if (SUPPORT["sha"](source_path)!=manifest["source_geometry_manifest_sha256"]
            or Path(preview["live_component_geometry_manifest"]).resolve()!=source_path.resolve()
            or preview["source_stage_sha256"]!=geometry["source_stage_sha256"]
            or source["strict_map_digest"]!=manifest["planning_occupancy_digest"]):
        raise ValueError("H3 task raster/environment differs from the actual preview")
    with np.load(manifest["occupancy_grid"]) as grid:
        occupied,origin,resolution=grid["occupied"].copy(),grid["origin_xy_m"].copy(),float(grid["resolution_m"])
    if (occupied.dtype!=np.dtype(bool) or list(occupied.shape)!=manifest["raster"]["shape_rows_columns"]
            or not np.array_equal(origin,manifest["raster"]["origin_xy_m"]) or resolution!=manifest["raster"]["resolution_m"]
            or hashlib.sha256(occupied.tobytes()).hexdigest()!=manifest["planning_occupancy_digest"]):
        raise ValueError("H3 task raster/frame mismatch")
    sx,sy,yaw=manifest["start_xyyaw"];gx,gy,gyaw=manifest["goal_xyyaw"]
    local=[math.cos(yaw)*(gx-sx)+math.sin(yaw)*(gy-sy),-math.sin(yaw)*(gx-sx)+math.cos(yaw)*(gy-sy),
        math.atan2(math.sin(gyaw-yaw),math.cos(gyaw-yaw))]
    if not np.allclose(local,manifest["local_goal_xyyaw"],rtol=0,atol=1e-10) or abs(local[2]+math.pi/2)>1e-10:
        raise ValueError("H3 world/local goal does not describe the fixed clockwise quarter turn")
    if any(GEO["pose_collision"](occupied,manifest[key],origin,resolution,.01) for key in ("start_xyyaw","goal_xyyaw")):
        raise ValueError("H3 task endpoint full envelope is blocked")
    return manifest,occupied,origin,resolution,SUPPORT["sha"](path)


def model_bounds():
    if SUPPORT["sha"](MODEL_PLAN)!=MODEL_SHA:
        raise ValueError("The inherited H1/H2 model reference changed")
    return json.loads(MODEL_PLAN.read_bytes())["rigid_body_parameter_bounds"]
