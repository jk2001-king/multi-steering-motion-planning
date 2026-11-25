#!/usr/bin/env python3
"""Build a NEW strict raster for the live H3 flush condition. No PhysX execution."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

import numpy as np
from scipy.ndimage import binary_fill_holes


HERE=Path(__file__).resolve().parent
WORKSPACE=_ijcas_workspace();GATE=_ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
FLUSH=runpy.run_path(str(HERE/"hospital_h3_flush_geometry.py"))
GEO=runpy.run_path(str(HERE/"hospital_geometry_tools.py"))
ORIGIN=np.array([11.,13.5]);SHAPE=(710,1050);RESOLUTION=.01;SLAB=(.030,.450)


def project(audit):
    FLUSH["verify_variant"](audit)
    occupied=np.zeros(SHAPE,dtype=bool);included=[];low_terrain=[]
    maximum=ORIGIN+np.asarray(SHAPE[::-1])*RESOLUTION
    for asset in audit["local_assets"]:
        name=asset["path"].rsplit("/",1)[-1];mask=np.zeros(SHAPE,dtype=bool)
        for prim in asset["collision"]["primitives"]:
            bound=prim["world_aligned_bound"];lo,hi=bound["minimum_m"],bound["maximum_m"]
            if hi[0]<ORIGIN[0] or lo[0]>maximum[0] or hi[1]<ORIGIN[1] or lo[1]>maximum[1]:continue
            if hi[2]<SLAB[0]:
                if hi[2]-lo[2]>.005 and hi[2]>.005:
                    rows,cols=GEO["polygon_cells"]([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],ORIGIN,RESOLUTION,SHAPE)
                    mask[rows,cols]=True;low_terrain.append(prim["path"])
                continue
            if lo[2]>SLAB[1]:continue
            mesh=prim.get("authored_mesh")
            if mesh is None or mesh.get("physics_approximation")!="none":raise ValueError("Need actual authored mesh: "+prim["path"])
            if name.startswith(("SM_HospitalBed","SM_SupplyCart")):
                bb=asset["world_aligned_visible_bound"];a,b=bb["minimum_m"],bb["maximum_m"]
                rows,cols=GEO["polygon_cells"]([(a[0],a[1]),(b[0],a[1]),(b[0],b[1]),(a[0],b[1])],ORIGIN,RESOLUTION,SHAPE)
                mask[rows,cols]=True
            else:mask|=GEO["mesh_mask"](mesh,ORIGIN,RESOLUTION,SHAPE,*SLAB)
        mask=binary_fill_holes(mask)
        if mask.any():occupied|=mask;included.append(asset["path"])
    return occupied,included,low_terrain


def derive(report,audit,reference):
    if (report.get("condition")!=FLUSH["CONDITION"] or not report.get("checks")
            or not all(report["checks"].values()) or report.get("live_geometry_verified") is not True):
        raise ValueError("Need the live flush geometry report")
    occupied,included,terrain=project(audit)
    start,end=reference["start_xyyaw"],reference["component_end_xyyaw"]
    distance=reference["component_reference_distance_m"]
    poses=[[start[0]-f*distance,start[1],start[2]] for f in np.linspace(0,1,math.ceil(distance/.005)+1)]
    failures=[i for i,p in enumerate(poses) if GEO["pose_collision"](occupied,p,ORIGIN,RESOLUTION,.01)]
    original=reference["threshold_collision_paths"]
    checks={"explicit_flush_inventory_verified":FLUSH["verify_variant"](audit),
        "unchanged_component_start_end_and_clearance":reference["footprint"]=={"length_m":1.630,"width_m":.965,"clearance_m":.010},
        "strict_swept_envelope_clear_for_straight_COMPONENT":not failures,
        "new_raster_matches_only_selected_threshold_removed":hashlib.sha256(occupied.tobytes()).hexdigest()==reference["component_geometry_map_digest"],
        "other_low_terrain_stays_blocked":all(p not in original for p in terrain)}
    manifest={"schema_version":1,"scenario":"H3_doorway_immediate_turn","environment_condition":FLUSH["CONDITION"],
        "purpose":"FLUSH_STRAIGHT_COMPONENT_READINESS_NOT_H3_IMMEDIATE_TURN_MISSION",
        "status":"FLUSH_GEOMETRY_READY_PHYSICS_STATIC_PENDING" if all(checks.values()) else "HOLD_FLUSH_GEOMETRY",
        "source_stage":report["source_stage"],"door_condition":"fully_open_baseline_session_only",
        "start_xyyaw":start,"component_end_xyyaw":end,"component_reference_distance_m":distance,
        "footprint":reference["footprint"],"raster":{"origin_xy_m":ORIGIN.tolist(),"shape_rows_columns":list(SHAPE),"resolution_m":RESOLUTION},
        "strict_map_digest":hashlib.sha256(occupied.tobytes()).hexdigest(),"obstacle_height_slab_m":list(SLAB),
        "terrain_policy":"NO_TERRAIN_EXEMPTION_selected_threshold_physically_absent_only_in_this_flush_variant_other_low_terrain_blocked",
        "suppressed_threshold_root":FLUSH["THRESHOLD"],"original_threshold_collision_paths":original,
        "support_floor_top_z_m":FLUSH["SUPPORT_TOP_M"],"included_collision_assets":included,"remaining_blocked_low_terrain":terrain,
        "checks":checks,"geometry_sweep_collision_indices":failures,
        "component_map_may_be_used_for_path_planning_or_Nav2":False,
        "static_support_verified":False,"component_crossing_verified":False,"H3_task_start_goal_selected":False,
        "path_planning_executed":False,"PhysX_or_Nav2_executed":False,"use_for_final_statistics":False}
    return manifest,occupied


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--geometry-directory",type=Path,required=True)
    args=parser.parse_args();report_path=args.geometry_directory.resolve()/"flush_geometry_report.json"
    if (report_path.parent/"preparation_failure.json").exists():raise ValueError("The live preparation rolled back; do not reuse its intermediate export")
    raw=report_path.read_bytes();report=json.loads(raw)
    if report.get("source_USD_saved") is not False or report.get("PhysX_executed") is not False:
        raise ValueError("Expected explicit geometry-only flush preparation")
    for name,digest in report["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError("Flush evidence changed: "+name)
    for name,digest in report["source_sha256"].items():
        folder=_ijcas_path(WORKSPACE, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError("Flush preparation source changed: "+name)
    audit_path=Path(report["active_geometry_audit"]);audit=json.loads(audit_path.read_bytes())
    reference_path=GATE/"hospital_h3_terrain_setup/setup_r01/terrain_component_setup.json"
    reference=json.loads(reference_path.read_bytes());manifest,occupied=derive(report,audit,reference)
    if not all(manifest["checks"].values()):raise ValueError("Flush map/component preflight failed: "+str(manifest["checks"]))
    folder=GATE/"hospital_h3_flush_setup";folder.mkdir(parents=True,exist_ok=True);i=1
    while (folder/f"setup_r{i:02d}").exists():i+=1
    output=folder/f"setup_r{i:02d}";output.mkdir();grid=output/"flush_occupancy.npz"
    np.savez_compressed(grid,occupied=occupied,origin_xy_m=ORIGIN,resolution_m=RESOLUTION)
    manifest.update(flush_geometry_report=str(report_path),flush_geometry_sha256=hashlib.sha256(raw).hexdigest(),
        source_audit=str(audit_path),fully_open_geometry_report=report["fully_open_geometry_report"],
        fully_open_geometry_sha256=report["fully_open_geometry_sha256"],occupancy_grid=str(grid),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/"hospital_h3_flush_geometry.py",HERE/"hospital_geometry_tools.py")},
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (report_path,audit_path,reference_path,grid)})
    target=output/"flush_component_setup.json"
    with target.open("x",encoding="utf-8") as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write("\n")
    print("=== H3 FLUSH STRICT GEOMETRY SETUP ===");print("checks:",manifest["checks"]);print("saved:",target)
    print("No PhysX/static/crossing/planner/Nav2 result is implied.")


if __name__=="__main__":main()
