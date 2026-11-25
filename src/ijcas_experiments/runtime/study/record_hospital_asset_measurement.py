#!/usr/bin/env python3
"""Record an exported bed asset without inventing physical measurements."""

from __future__ import annotations

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import yaml


WORKSPACE = _ijcas_workspace()
CONFIG = _ijcas_path(WORKSPACE, 'studies/IJCAS/config')
GATE = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')


def validate_bound(bound):
    if not isinstance(bound, dict):
        raise ValueError("Missing asset bound")
    for key in ("minimum_m", "maximum_m", "size_m", "center_m"):
        values = bound.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"Invalid bound field: {key}")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError(f"Non-finite bound field: {key}")
    for axis in range(3):
        lo, hi = bound["minimum_m"][axis], bound["maximum_m"][axis]
        if hi <= lo:
            raise ValueError("Asset bound must have positive extent")
        if not math.isclose(hi - lo, bound["size_m"][axis], abs_tol=1e-8):
            raise ValueError("Asset extent does not match minimum/maximum")
        if not math.isclose((lo + hi) / 2, bound["center_m"][axis], abs_tol=1e-8):
            raise ValueError("Asset centre does not match minimum/maximum")


def bed_evidence(audit):
    if audit.get("kind") != "ijcas_hospital_asset_geometry_audit":
        raise ValueError("Not an IJCAS hospital-asset audit")
    version = audit.get("schema_version")
    if version not in (1, 2):
        raise ValueError("Unsupported hospital-asset audit schema")
    scale = audit.get("meters_per_unit")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid meters_per_unit")
    if version == 1 and scale != 1.0:
        raise ValueError("Legacy exporter did not convert non-metre stage units")
    if version == 2 and audit.get("coordinate_units") != "metres_converted_from_stage_units":
        raise ValueError("Missing metre-conversion evidence")
    if audit.get("up_axis") != "Z":
        raise ValueError("A Z-up hospital stage is required")
    bed = audit.get("targets", {}).get("bed", {})
    if bed.get("valid") is not True or not str(bed.get("path", "")).startswith("/World/Hospital/"):
        raise ValueError("Missing valid hospital bed target")
    visible = bed.get("world_aligned_visible_bound")
    collision = bed.get("collision", {})
    validate_bound(visible)
    validate_bound(collision.get("bound"))
    paths = collision.get("collision_prim_paths", [])
    if not paths or len(set(paths)) != len(paths) or collision.get("collision_prim_count") != len(paths):
        raise ValueError("Collision path/count mismatch")
    if any(not path.startswith(bed["path"] + "/") and path != bed["path"] for path in paths):
        raise ValueError("Collision prim lies outside the target bed")
    bounds_match = all(
        math.isclose(a, b, abs_tol=1e-6, rel_tol=0)
        for key in ("minimum_m", "maximum_m")
        for a, b in zip(visible[key], collision["bound"][key])
    )
    return bed, bounds_match


def cached_medical_obstacle_roots(inventory):
    """Group original cached colliders by name, not by planning performance.

    This is a historical authored-AABB inventory, not current collision,
    support, semantic completeness, intrinsic dimensions or drive evidence.
    """
    if inventory.get('version')!='H4_all_stage_USD_collision_inventory_and_cached_queries_from_retained_H3_v1':
        raise ValueError('Exact complete cached inventory schema required')
    summary=inventory['inventory_summary'];colliders=inventory['collision_prims']
    if (summary.get('authored_inventory_coverage_complete') is not True
            or not isinstance(colliders,list)
            or type(summary.get('collision_prim_count')) is not int
            or summary.get('collision_prim_count')!=len(colliders)
            or summary.get('errors')!=[]
            or summary.get('active_unloaded_payload_paths')!=[]):
        raise ValueError('Complete cached authored inventory required; no missing rows/payloads invented')
    groups={};seen=set()
    for row in colliders:
        path=row['path']
        if not isinstance(path,str) or not path.startswith('/') or '//' in path:
            raise ValueError('Original absolute collider path required')
        if path in seen:raise ValueError('Duplicate original collider path')
        seen.add(path);parts=path.split('/')
        if len(parts)<4 or parts[1:3]!=['World','Hospital']:continue
        name=parts[3].lower();family=None
        if 'bedsidetable' in name:continue
        for token,kind in (('supplycart','supply_cart'),('gascart','gas_cart'),('gurney','gurney'),
                ('hospitalbed','hospital_bed'),('wheelchair','wheelchair')):
            if token in name:family=kind;break
        if family is None:
            if name.startswith('sm_cart_'):family='cart_name_match'
            elif name.startswith('geo_m_bed_'):family='bed_proxy_name_match'
            else:continue
        if row['is_active'] is not True or row['collision_enabled_value'] is not True:continue
        bound=row.get('world_aligned_bound') or {};low=bound.get('minimum_m');high=bound.get('maximum_m')
        if (not isinstance(low,list) or not isinstance(high,list) or len(low)!=len(high) or len(low)!=3
                or any(type(x) not in (float,int) or not math.isfinite(x) for x in (*low,*high))
                or any(a>b for a,b in zip(low,high))):
            raise ValueError('Finite original medical-object bounds required; no replacement bounds')
        root='/'.join(parts[:4])
        group=groups.setdefault(root,dict(path=root,name_based_family=family,cached_collision_paths=[],
            minimum_m=list(low),maximum_m=list(high)))
        group['cached_collision_paths'].append(path)
        group['minimum_m']=[min(a,b) for a,b in zip(group['minimum_m'],low)]
        group['maximum_m']=[max(a,b) for a,b in zip(group['maximum_m'],high)]
    rows=[]
    for root,group in sorted(groups.items()):
        low,high=group['minimum_m'],group['maximum_m']
        rows.append(dict(**group,cached_enabled_collision_count=len(group['cached_collision_paths']),
            world_AABB_extent_m=[b-a for a,b in zip(low,high)],
            world_AABB_center_m=[(a+b)/2 for a,b in zip(low,high)]))
    return rows


def write_cached_obstacle_inventory(path,output=None):
    """Read-only future-H5 asset preparation; never edit measurement YAML."""
    import inspect
    from collections import Counter
    from hospital_runtime_evidence_io import write_exclusive
    path=Path(path).resolve();payload=path.read_bytes();inventory=json.loads(payload)
    rows=cached_medical_obstacle_roots(inventory)
    if not rows:raise ValueError('No named cached medical-object roots; no task obstacle invented')
    definition=dict(version='cached_original_hospital_obstacle_candidates_for_future_H5_v1',
        input_sha256={str(path):hashlib.sha256(payload).hexdigest()},
        metric_function_sha256=hashlib.sha256(inspect.getsource(cached_medical_obstacle_roots).encode()).hexdigest(),
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_cached_obstacle_inventory).encode()).hexdigest())
    output=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/cached_inventory_r01').resolve()
    target=output/'cached_original_obstacles.json';table=output/'cached_original_obstacles.csv'
    stream=io.StringIO(newline='')
    fields=['path','name_based_family','cached_enabled_collision_count',
        *[side+'_'+axis+'_m' for side in ('min','max') for axis in 'xyz']]
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in rows:
        writer.writerow(dict(path=row['path'],name_based_family=row['name_based_family'],
            cached_enabled_collision_count=row['cached_enabled_collision_count'],
            **{side+'_'+axis+'_m':row[key][i] for side,key in [('min','minimum_m'),('max','maximum_m')] for i,axis in enumerate('xyz')}))
    table_bytes=stream.getvalue().encode('utf-8')
    report=dict(**definition,status='HISTORICAL_AUTHORED_ASSET_CANDIDATES_NO_H5_TASK_OR_DRIVE_ADMISSION',
        original_inventory_collision_rows=len(inventory['collision_prims']),candidate_roots=rows,
        counts_by_name_based_family=dict(Counter(row['name_based_family'] for row in rows)),
        no_current_flag_or_transform_verification_claimed=True,
        name_patterns_NOT_a_complete_medical_asset_ontology=True,
        world_AABBs_NOT_oriented_dimensions_or_cooked_shapes=True,
        original_floor_surfaces_or_complete_task_support_NOT_verified=True,
        no_obstacle_task_pose_route_or_planning_map_selected=True,
        no_selection_using_any_method_PASS_FAIL_or_cost=True,
        no_live_stage_robot_timeline_or_measurements_YAML_change=True,
        new_PhysX_trial_count=0,actual_drive_admission=False,use_for_final_statistics=False,
        output_sha256={str(table):hashlib.sha256(table_bytes).hexdigest()})
    if path.read_bytes()!=payload:raise ValueError('Cached source changed during derived export')
    if target.exists():
        if (json.loads(target.read_bytes())!=report or not table.is_file()
                or table.read_bytes()!=table_bytes):
            raise ValueError('Existing cached obstacle inventory changed; no overwrite')
        return target
    output.mkdir(parents=True,exist_ok=True)
    with table.open('xb') as stream:
        stream.write(table_bytes)
    write_exclusive(target,report)
    print('Saved cached original medical-obstacle candidates:',target)
    return target


def compare_cached_medical_obstacle_readback(inventory,actual):
    """Compare named cached IDs/bounds, not cooked shapes or task feasibility."""
    cached=cached_medical_obstacle_roots(inventory)
    if not isinstance(actual,list) or not actual or len({row['path'] for row in actual})!=len(actual):
        raise ValueError('Distinct actual named medical-object rows required')
    current={row['path']:row for row in actual}
    if set(current)!={row['path'] for row in cached}:
        raise ValueError('Every original named candidate must be recorded; no missing/best-object selection')
    records=[]
    for old in cached:
        row=current[old['path']]
        if row.get('active') is not True or row.get('loaded') is not True or row.get('unloaded_payload_paths')!=[]:
            raise ValueError('Actual complete loaded candidate required; no reload or replacement')
        prims=row['enabled_collision_prims'];paths=[prim['path'] for prim in prims]
        if len(set(paths))!=len(paths) or any(not path.startswith(old['path']+'/') and path!=old['path'] for path in paths):
            raise ValueError('Unique collision identities inside the exact candidate root required')
        if not prims:raise ValueError('Actual enabled candidate collider bounds required')
        for prim in prims:
            bound=prim['world_aligned_bound'];lo,hi=bound['minimum_m'],bound['maximum_m']
            if (len(lo)!=3 or len(hi)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in (*lo,*hi))
                    or any(a>b for a,b in zip(lo,hi))):raise ValueError('Finite ordered actual collider bounds required')
        low=[min(prim['world_aligned_bound']['minimum_m'][i] for prim in prims) for i in range(3)]
        high=[max(prim['world_aligned_bound']['maximum_m'][i] for prim in prims) for i in range(3)]
        delta=max(abs(a-b) for key,values in [('minimum_m',low),('maximum_m',high)] for a,b in zip(old[key],values))
        records.append(dict(path=old['path'],name_based_family=old['name_based_family'],
            cached_enabled_collision_count=old['cached_enabled_collision_count'],actual_enabled_collision_count=len(prims),
            same_cached_enabled_collision_paths=set(paths)==set(old['cached_collision_paths']),
            maximum_cached_vs_actual_union_AABB_difference_m=delta,
            actual_union_AABB=dict(minimum_m=low,maximum_m=high),
            AABB_equality_is_NOT_intrinsic_shape_or_cooked_collision_verification=True))
    return dict(candidate_count=len(records),candidates=records,
        all_cached_enabled_collision_paths_retained=all(row['same_cached_enabled_collision_paths'] for row in records),
        maximum_cached_vs_actual_union_AABB_difference_m=max(row['maximum_cached_vs_actual_union_AABB_difference_m'] for row in records),
        historical_bounds_NOT_used_as_replacements_for_live_GETs=True,
        H5_task_support_start_goal_map_and_drive_admission=False)


async def review_medical_assets_retaining_completed_route(result):
    """GET-only future-H5 geometry review at an exact finished H4 boundary.

    Call inside the existing Isaac app only after its own route has ended.
    No timeline request, selection, camera/render edit, source save or trial.
    A source-bound body/12 joints/13 links and all 4997 protected environment
    rows are checked before and after the bounded asynchronous asset GETs.
    """
    import builtins
    import inspect
    import time
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Usd,UsdGeom,UsdPhysics
    import hospital_h4_subfloor_compact_path_launch as launch
    from hospital_runtime_evidence_io import timeline_state,write_exclusive
    result=Path(result).resolve();boundary=launch.completed_route_boundary(result)
    raw=launch.preparation.read_small(result)
    context=launch.load_archived_fresh_trial_protocol(launch.FRESH_TRIAL_ROOT/raw['execution_cohort']/'protocol.json')
    cache=GATE/'hospital_h4_geometry_review/complete_inventory_r01/live_collision_inventory.json'
    payload=cache.read_bytes();inventory=json.loads(payload);candidates=cached_medical_obstacle_roots(inventory)
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        task=getattr(builtins,name,None)
        if task is not None and not task.done():raise RuntimeError('An experiment/route/display task is active; no concurrent asset review')
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    expected=boundary['required_prior_timeline']
    if stage is None or expected['playing'] or expected['stopped'] or timeline_state(timeline)!=expected:
        raise ValueError('Exact completed PAUSED route required; no Play/Stop/reset/reopen recovery')
    if Path(stage.GetRootLayer().realPath).resolve()!=_ijcas_path(WORKSPACE, 'isaac/scenes/hospital_experiment.usda'):
        raise ValueError('Same original hospital source stage required, not a reopened replacement')
    if UsdGeom.GetStageMetersPerUnit(stage)!=1. or str(UsdGeom.GetStageUpAxis(stage))!='Z':
        raise ValueError('Original metre-based Z-up hospital required')
    view=SimulationManager.get_physics_sim_view()
    if view is None:raise ValueError('Existing completed-route tensor view required; no initialization or Play')
    tensor=view.create_articulation_view(context.cohort.ROBOT+'/base_link')
    if tensor.count!=1 or tensor.max_links!=13:raise ValueError('Exactly one retained thirteen-link actor required')
    def snapshot():return launch.normalize_runtime_evidence(launch.preparation.old.tensor_snapshot(tensor))
    before=snapshot();layers=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    if before!=boundary['required_prior_tensor']:
        raise ValueError('Actual body/all 12 joints/13 links differ from the finished route; no repair')
    began=time.monotonic()
    def unchanged():
        if (usd.get_stage()!=stage or timeline_state(timeline)!=expected or snapshot()!=before
                or stage.GetRootLayer().ExportToString()!=layers['Root']
                or stage.GetSessionLayer().ExportToString()!=layers['Session']):
            raise ValueError('Completed route or layers changed during GET-only review; no restore')
        if time.monotonic()-began>180.:raise TimeoutError('GET-only asset review exceeded wall budget; no retry')
    print('[H5 ASSETS GET] Retaining exact completed H4 body/12 joints/13 links; no timeline command.',flush=True)
    environment=await launch.fresh_environment_readback(context,stage,timeline,'H5_original_assets_GET_only')
    unchanged()
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
    actual=[]
    for index,row in enumerate(candidates):
        root=stage.GetPrimAtPath(row['path'])
        if not root.IsValid():raise ValueError('Original medical asset missing: '+row['path'])
        prims=[];unloaded=[]
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if prim.IsActive() and prim.HasPayload() and not prim.IsLoaded():unloaded.append(str(prim.GetPath()))
            if (not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):continue
            value=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            prims.append(dict(path=str(prim.GetPath()),type_name=prim.GetTypeName(),is_instance_proxy=prim.IsInstanceProxy(),
                world_aligned_bound=dict(minimum_m=list(map(float,value.GetMin())),maximum_m=list(map(float,value.GetMax())))))
        actual.append(dict(path=row['path'],active=root.IsActive(),loaded=root.IsLoaded(),
            unloaded_payload_paths=unloaded,enabled_collision_prims=prims))
        if (index+1)%10==0:print('[H5 ASSETS GET] Original named candidates=%d/%d; no motion' % (index+1,len(candidates)),flush=True)
        await omni.kit.app.get_app().next_update_async();unchanged()
    comparison=compare_cached_medical_obstacle_readback(inventory,actual)
    unchanged();launch.preparation.verify(boundary['input_sha256'])
    if cache.read_bytes()!=payload:raise ValueError('Historical source changed during readback; no write')
    report=dict(version='H5_original_named_medical_asset_GETs_at_exact_completed_H4_boundary_v1',
        status='CURRENT_AUTHORED_ASSETS_RECORDED_NO_H5_TASK_OR_DRIVE_ADMISSION',
        retained_result=str(result),original_route_verdict=raw['verdict'],route_verdict_NOT_relabelled=True,
        actual_timeline_before_and_after=expected,body_12_joints_13_links_gains_targets_exactly_retained=True,
        Root_Session_exactly_unchanged=True,all_4997_protected_environment_rows=environment,
        actual_assets=actual,cached_vs_actual= comparison,
        current_AABBs_NOT_cooked_shapes_or_physical_support_or_visible_pixels=True,
        no_H5_obstacle_task_start_goal_route_or_map_selected=True,
        no_Play_Stop_Pause_reset_reopen_save_selection_camera_renderer_or_physics_edit=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={**boundary['input_sha256'],str(cache):hashlib.sha256(payload).hexdigest()},
        analysis_source={function.__name__:inspect.getsource(function) for function in
            (cached_medical_obstacle_roots,compare_cached_medical_obstacle_readback,review_medical_assets_retaining_completed_route)})
    target=GATE/'hospital_h5_geometry_candidates/retained_route_readback'/raw['execution_cohort']/'live_assets.json'
    if target.exists():
        if json.loads(target.read_bytes())!=report:raise ValueError('Existing asset readback differs; no overwrite')
    else:write_exclusive(target,report)
    print('=== H5 ORIGINAL MEDICAL ASSETS / RETAINED H4 GET-ONLY REVIEW ===',flush=True)
    print('Original H4 verdict preserved / current named assets:',raw['verdict'],len(actual),flush=True)
    print('cached enabled identities retained / maximum AABB difference m:',comparison['all_cached_enabled_collision_paths_retained'],comparison['maximum_cached_vs_actual_union_AABB_difference_m'],flush=True)
    print('saved:',target,flush=True)
    print('No H5 start/goal, map, cooked support, static or drive admission. No new trial.',flush=True)
    return report


def schedule_medical_assets_review(result,task_proposal=None):
    """One observable GET-only task; repeated clicks never create concurrency."""
    import asyncio
    import builtins
    import traceback
    key='ijcas_h5_original_assets_get_task' if task_proposal is None else 'ijcas_h5_corridor_cached_queries_task'
    previous=getattr(builtins,key,None)
    if previous is not None:
        if not previous.done():
            print('[H5 ASSETS GET] Existing read-only review still running; no duplicate task.',flush=True)
        elif previous.cancelled():
            print('[H5 ASSETS GET] Prior review was cancelled; no automatic retry.',flush=True)
        elif previous.exception() is not None:
            print('[H5 ASSETS GET] Prior review failed; no retry. Original error follows.',flush=True)
            traceback.print_exception(type(previous.exception()),previous.exception(),previous.exception().__traceback__)
        else:
            print('[H5 ASSETS GET] Existing review already complete; no duplicate execution.',flush=True)
        return previous
    other='ijcas_h5_corridor_cached_queries_task' if task_proposal is None else 'ijcas_h5_original_assets_get_task'
    other_task=getattr(builtins,other,None)
    if other_task is not None and not other_task.done():
        raise RuntimeError('Another H5 GET-only review is active; no concurrent review')
    map_task=getattr(builtins,'ijcas_h5_current_corridor_map_get_task',None)
    if map_task is not None and not map_task.done():
        raise RuntimeError('A current-map GET is active; no concurrent asset/query review')
    corner_task=getattr(builtins,'ijcas_h5_corrected_goal_corner_get_task',None)
    if corner_task is not None and not corner_task.done():
        raise RuntimeError('A corrected-corner GET is active; no concurrent asset/query review')
    coroutine=(review_medical_assets_retaining_completed_route(result) if task_proposal is None
        else review_proposed_corridor_cached_queries(result,task_proposal))
    task=asyncio.ensure_future(coroutine)
    setattr(builtins,key,task)
    def finished(value):
        if value.cancelled():
            print('[H5 ASSETS GET] Cancelled; no automatic retry or scene command.',flush=True)
        elif value.exception() is not None:
            error=value.exception()
            print('[H5 ASSETS GET] Read-only review stopped; no state recovery or retry.',flush=True)
            traceback.print_exception(type(error),error,error.__traceback__)
    task.add_done_callback(finished)
    print('[H5 ASSETS GET] Scheduled '+('original medical-asset GETs' if task_proposal is None else 'proposed-task cached ray GETs')+
        ' only; no Play/Stop/Pause/reset.',flush=True)
    return task


def corridor_bypass_task_geometry(asset_report,floor_record,support_design):
    """Task-only H5 proposal from an existing cart and the original corridor.

    A southbound right-hand lane is one quarter of the original floor width
    from its west edge. Endpoint buffers are fixed 2.5m from the cart faces,
    not obtained from any steering trajectory, cost or measured outcome.
    No route is selected; AABB overlap is not a physical collision claim.
    """
    from hospital_h4_support_design import floor_triangles,rectangle,ccw,uncovered,area
    from hospital_geometry_tools import footprint_polygon
    if (asset_report.get('version')!='H5_original_named_medical_asset_GETs_at_exact_completed_H4_boundary_v1'
            or asset_report.get('body_12_joints_13_links_gains_targets_exactly_retained') is not True
            or asset_report.get('Root_Session_exactly_unchanged') is not True
            or asset_report.get('new_PhysX_trial_count')!=0
            or asset_report.get('actual_motion_authorized') is not False
            or asset_report.get('use_for_final_statistics') is not False
            or asset_report.get('all_4997_protected_environment_rows',{}).get('row_count')!=4997
            or asset_report.get('cached_vs_actual',{}).get('all_cached_enabled_collision_paths_retained') is not True
            or asset_report.get('cached_vs_actual',{}).get('maximum_cached_vs_actual_union_AABB_difference_m')!=0.):
        raise ValueError('Actual retained-route GET-only asset report required')
    selected='/World/Hospital/SM_SupplyCart_02a7'
    candidates=asset_report['cached_vs_actual']['candidates']
    matches=[row for row in candidates if row['path']==selected]
    if len(matches)!=1 or matches[0]['name_based_family']!='supply_cart':
        raise ValueError('The originally declared corridor cart is required; no favorable-object substitution')
    cart=matches[0]['actual_union_AABB'];lo,hi=cart['minimum_m'],cart['maximum_m']
    if (len(lo)!=3 or len(hi)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in (*lo,*hi))
            or any(a>=b for a,b in zip(lo,hi))):raise ValueError('Finite positive actual cart bounds required')
    roots=[root for root in floor_record['original_floor_roots'] if root['path']=='/World/Hospital/Geo_Floor_Costum18_24']
    if len(roots)!=1:raise ValueError('Exact original vertical-corridor floor required')
    triangles,_=floor_triangles(floor_record)
    points=[point for mesh in roots[0]['meshes'] for point in mesh['world_points_m']]
    west,east=min(p[0] for p in points),max(p[0] for p in points)
    if not west<lo[0]<hi[0]<east:raise ValueError('Original cart must remain inside the original corridor width')
    lane_x=west+.25*(east-west);buffer=2.5
    start=[lane_x,hi[1]+buffer,-math.pi/2];goal=[lane_x,lo[1]-buffer,-math.pi/2]
    footprints=[ccw(footprint_polygon(pose).tolist()) for pose in (start,goal)]
    bound=support_design['canonical_unchanged_support_bound']
    support=[rectangle(bound['minimum_m'][0],bound['maximum_m'][0],bound['minimum_m'][1],bound['maximum_m'][1]),
        *[ccw(poly) for poly in support_design['nonoverlapping_support_rectangles']]]
    support_missing=[sum(abs(area(poly)) for poly in uncovered([fp],support)) for fp in footprints]
    if any(value>1e-9 for value in support_missing):
        raise ValueError('The same existing authored support must cover both expanded endpoints; no floor extension')
    # At -90deg the straight swept rectangle is an exact planar rectangle.
    swept=rectangle(lane_x-.965/2-.010,lane_x+.965/2+.010,goal[1]-1.630/2-.010,start[1]+1.630/2+.010)
    cart_polygon=rectangle(lo[0],hi[0],lo[1],hi[1])
    from hospital_h4_support_design import clip_halfplane
    intersection=list(swept)
    for a,b in zip(cart_polygon,cart_polygon[1:]+cart_polygon[:1]):intersection=clip_halfplane(intersection,a,b)
    overlap=abs(area(intersection))
    if overlap<=0:raise ValueError('The declared lane does not intersect the mapped cart AABB; do not invent a bypass requirement')
    return dict(version='H5_existing_cart_southbound_right_hand_lane_TASK_PROPOSAL_v1',
        scenario='H5_obstacle_bypass_return',task='bypass_original_parked_cart_and_return_to_the_same_declared_lane',
        selected_original_cart=selected,actual_cart_union_AABB=cart,
        original_corridor_floor=roots[0]['path'],original_corridor_width_bounds_m=[west,east],
        lane_rule='Southbound right-hand lane at west + 0.25 * original floor width',
        endpoint_rule='Fixed 2.5m longitudinal buffers from the actual cart AABB faces; same lane and heading',
        fixed_longitudinal_buffer_m=buffer,start_xyyaw=start,goal_xyyaw=goal,
        footprint=dict(length_m=1.630,width_m=.965,clearance_m=.010),expanded_endpoint_footprints_xy=footprints,
        authored_existing_support_uncovered_endpoint_area_m2=support_missing,
        original_visual_floor_uncovered_endpoint_area_m2=[sum(abs(area(poly)) for poly in uncovered([fp],triangles)) for fp in footprints],
        same_previously_authorized_support_planar_join_NOT_a_new_floor_change=True,
        nominal_straight_lane_swept_rectangle_xy=swept,mapped_cart_AABB_overlap_area_m2=overlap,
        straight_AABB_overlap_is_NOT_measured_contact_or_cooked_geometry=True,
        task_defined_without_planner_steering_trajectory_cost_or_outcome=True,
        same_H4_hospital_corridor_different_task_NOT_an_independent_layout=True,
        static_parked_cart_bypass_is_NOT_online_replanning_or_dynamic_obstacle_validation=True,
        no_obstacle_pose_road_floor_material_or_live_robot_change=True,
        complete_task_map_support_queries_static_and_drive_admission=False,
        route_or_planner_candidate_selected=False,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def write_corridor_bypass_task_proposal(asset_path,output=None):
    """Offline geometry proposal/figure; no measurements YAML or stage write."""
    import inspect
    asset_path=Path(asset_path).resolve();asset=json.loads(asset_path.read_bytes())
    floor_path=GATE/'hospital_h4_geometry_review/floor_surfaces_r01/live_floor_surfaces.json'
    design_path=GATE/'hospital_h4_support/design_r02/support_design.json'
    protocol_path=_ijcas_path(WORKSPACE, 'studies/IJCAS/config/hospital_paper_validation_protocol.yaml')
    floor=json.loads(floor_path.read_bytes());design=json.loads(design_path.read_bytes())
    pins={**asset['input_sha256'],**floor['input_sha256'],**design['input_sha256'],
        **{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (asset_path,floor_path,design_path,protocol_path)}}
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in pins.items()):raise ValueError('Task proposal source changed')
    if yaml.safe_load(protocol_path.read_bytes())['scenarios']['H4_ward_corridor_turn']['original_parked_cart_candidate']!='/World/Hospital/SM_SupplyCart_02a7':
        raise ValueError('Originally declared cart identity required before any task proposal')
    geometry=corridor_bypass_task_geometry(asset,floor,design)
    definition=dict(input_sha256=pins,retained_asset_receipt=str(asset_path),
        geometry_function_sha256=hashlib.sha256(inspect.getsource(corridor_bypass_task_geometry).encode()).hexdigest(),
        exporter_function_sha256=hashlib.sha256(inspect.getsource(write_corridor_bypass_task_proposal).encode()).hexdigest(),
        analysis_source={f.__name__:inspect.getsource(f) for f in (corridor_bypass_task_geometry,write_corridor_bypass_task_proposal)})
    key=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/task_proposals'/key).resolve()
    target=output/'task_proposal.json';figure=output/'task_geometry.png'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (saved.get('status')!='PROPOSED_AUTHORED_TASK_GEOMETRY_NO_H5_ROUTE_OR_DRIVE_ADMISSION'
                or any(saved.get(k)!=v for k,v in {**json.loads(json.dumps(geometry)),**definition}.items()) or not figure.is_file()
                or hashlib.sha256(figure.read_bytes()).hexdigest()!=saved['output_sha256'].get(str(figure))):
            raise ValueError('Existing H5 proposal differs; no overwrite')
        return target
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':160}):
        fig,ax=plt.subplots(figsize=(6.,8.))
        try:
            bound=design['canonical_unchanged_support_bound'];lo,hi=bound['minimum_m'],bound['maximum_m']
            covers=[[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],*design['nonoverlapping_support_rectangles']]
            for index,cover in enumerate(covers):
                ax.add_patch(Polygon(cover,facecolor='#e9eee9',edgecolor='#bbc8bb',alpha=.65,label='Existing authored support' if index==0 else None))
            cart=geometry['actual_cart_union_AABB'];lo,hi=cart['minimum_m'],cart['maximum_m']
            ax.add_patch(Polygon([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]],facecolor='#cf9b5b',edgecolor='#865a29',label='Original cart union AABB'))
            for label,pose,fp,color in zip(('Proposed start','Proposed goal'),(geometry['start_xyyaw'],geometry['goal_xyyaw']),geometry['expanded_endpoint_footprints_xy'],('#087e91','#a74c76')):
                ax.add_patch(Polygon(fp,fill=False,edgecolor=color,lw=1.5))
                ax.scatter(pose[0],pose[1],color=color,marker='v',s=40,label=label)
            start,goal=geometry['start_xyyaw'],geometry['goal_xyyaw']
            ax.plot([start[0],goal[0]],[start[1],goal[1]],ls='--',color='#444444',label='Declared lane, NOT a planned route')
            west,east=geometry['original_corridor_width_bounds_m']
            ax.set(xlim=(west-.35,east+.35),ylim=(goal[1]-1.1,start[1]+1.1),xlabel='World x (m)',ylabel='World y (m)',
                title='H5 prospective task | original cart / same corridor\nNo asset moved; no route or motion admitted')
            ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower right')
            fig.text(.5,.02,'Planar source geometry only; no cooked support, H5 static,\nphysical collision, online replanning or drive validation.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0.,.07,1.,1.))
            with figure.open('xb') as stream:fig.savefig(stream,format='png',facecolor='white')
        finally:plt.close(fig)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in pins.items()):raise ValueError('Task proposal input changed during export')
    from hospital_runtime_evidence_io import write_exclusive
    write_exclusive(target,dict(**geometry,**definition,status='PROPOSED_AUTHORED_TASK_GEOMETRY_NO_H5_ROUTE_OR_DRIVE_ADMISSION',
        output_sha256={str(figure):hashlib.sha256(figure.read_bytes()).hexdigest()}))
    print('Saved H5 original-cart geometry proposal:',target)
    return target


def load_corridor_bypass_task_proposal(path):
    """Verify this task definition, not an H5 map or physical admission."""
    import inspect
    path=Path(path).resolve();task=json.loads(path.read_bytes())
    if (task.get('status')!='PROPOSED_AUTHORED_TASK_GEOMETRY_NO_H5_ROUTE_OR_DRIVE_ADMISSION'
            or task.get('geometry_function_sha256')!=hashlib.sha256(inspect.getsource(corridor_bypass_task_geometry).encode()).hexdigest()
            or task.get('exporter_function_sha256')!=hashlib.sha256(inspect.getsource(write_corridor_bypass_task_proposal).encode()).hexdigest()):
        raise ValueError('Exact reviewed task geometry/source required; no older definition silently adopted')
    for name,sha in {**task['input_sha256'],**task['output_sha256']}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Task proposal input/output changed: '+name)
    asset_path=Path(task['retained_asset_receipt']);asset=json.loads(asset_path.read_bytes())
    floor_path=GATE/'hospital_h4_geometry_review/floor_surfaces_r01/live_floor_surfaces.json'
    design_path=GATE/'hospital_h4_support/design_r02/support_design.json'
    protocol_path=_ijcas_path(WORKSPACE, 'studies/IJCAS/config/hospital_paper_validation_protocol.yaml')
    floor=json.loads(floor_path.read_bytes());design=json.loads(design_path.read_bytes())
    required={}
    for collection in (asset['input_sha256'],floor['input_sha256'],design['input_sha256'],
            {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (asset_path,floor_path,design_path,protocol_path)}):
        for name,sha in collection.items():
            if name in required and required[name]!=sha:raise ValueError('Conflicting task source pins')
            required[name]=sha
    if (task['input_sha256']!=required
            or set(task['output_sha256'])!={str(path.parent/'task_geometry.png')}
            or task.get('analysis_source')!={f.__name__:inspect.getsource(f) for f in
                (corridor_bypass_task_geometry,write_corridor_bypass_task_proposal)}):
        raise ValueError('Every original task source/figure/analysis binding required; no omitted pins')
    geometry=json.loads(json.dumps(corridor_bypass_task_geometry(asset,floor,design)))
    if any(task.get(key)!=value for key,value in geometry.items()):raise ValueError('Task geometry differs from its original inputs')
    return task,asset,design


def corridor_bypass_query_specifications(task):
    """Fixed endpoint/grid/control rays, not a selected successful route."""
    if (task.get('version')!='H5_existing_cart_southbound_right_hand_lane_TASK_PROPOSAL_v1'
            or task.get('actual_motion_authorized') is not False
            or task.get('complete_task_map_support_queries_static_and_drive_admission') is not False):
        raise ValueError('A task-only H5 proposal required; no admitted or substituted route')
    if (len(task['expanded_endpoint_footprints_xy'])!=2
            or any(len(task[name])!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in task[name])
                for name in ('start_xyyaw','goal_xyyaw'))):
        raise ValueError('Both complete finite declared endpoint poses and footprints required')
    specs=[]
    def add(name,xy,kind,z=.6,length=.8):
        if len(xy)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in xy):
            raise ValueError('Finite original query XY required')
        specs.append(dict(id=name,kind=kind,xy_m=list(xy),origin_z_m=z,length_m=length,direction=[0.,0.,-1.]))
    for label,pose,footprint in zip(('start','goal'),(task['start_xyyaw'],task['goal_xyyaw']),task['expanded_endpoint_footprints_xy']):
        if len(footprint)!=4:raise ValueError('Four original full-footprint corners required')
        add(label+'_center',pose[:2],'endpoint')
        for index,point in enumerate(footprint):add(label+'_corner_'+str(index),point,'endpoint')
    west,east=task['original_corridor_width_bounds_m']
    if not west<east:raise ValueError('Positive original corridor width required')
    for index in range(7):
        y=task['start_xyyaw'][1]+index*(task['goal_xyyaw'][1]-task['start_xyyaw'][1])/6
        for lane,fraction in enumerate((.25,.5,.75)):
            add('grid_%d_%d' % (index,lane),[west+fraction*(east-west),y],'corridor_grid')
    bounds=task['actual_cart_union_AABB']
    add('original_cart_control',[(a+b)/2 for a,b in zip(bounds['minimum_m'][:2],bounds['maximum_m'][:2])],
        'cart_control',z=2.,length=2.5)
    return specs


def summarize_corridor_cached_queries(specifications,rows,supports,top,limits,cart,actor):
    """Keep ALL cached hits. Classify samples only; never a static/drive PASS."""
    if (len(rows)!=len(specifications) or len({row['id'] for row in rows})!=len(rows)
            or not math.isfinite(top) or set(limits)!= {'height_tolerance_m','horizontal_tolerance_m','minimum_upward_normal_z'}):
        raise ValueError('All original query rows and unchanged native query limits required')
    diagnostics=[]
    for spec,row in zip(specifications,rows):
        if row['id']!=spec['id']:raise ValueError('Original ordered query identity required')
        floor=[];other=[];own=[];cart_hits=[]
        for index,hit in enumerate(row['all_hits']):
            values=(*hit['position_m'],*hit['normal'],hit['distance_m'])
            if (len(hit['position_m'])!=3 or len(hit['normal'])!=3
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in values)):
                raise ValueError('Finite raw hit evidence required, not coerced or invented values')
            p,n=hit['position_m'],hit['normal'];name=hit['collision']
            on_ray=(all(abs(a-b)<=limits['horizontal_tolerance_m'] for a,b in zip(p[:2],spec['xy_m']))
                and -limits['height_tolerance_m']<=hit['distance_m']<=spec['length_m']+limits['height_tolerance_m']
                and abs(spec['origin_z_m']-p[2]-hit['distance_m'])<=limits['height_tolerance_m'])
            if (name in supports and on_ray and abs(p[2]-top)<=limits['height_tolerance_m']
                    and n[2]>=limits['minimum_upward_normal_z']):floor.append(index)
            if name not in supports:other.append(index)
            if name==actor or name.startswith(actor+'/'):own.append(index)
            if name==cart or name.startswith(cart+'/'):cart_hits.append(index)
        diagnostics.append(dict(id=spec['id'],kind=spec['kind'],hit_count=len(row['all_hits']),
            matching_original_top_upward_support_hit_indices=floor,
            every_non_support_hit_indices=other,own_actor_hit_indices=own,original_cart_hit_indices=cart_hits,
            errors=row['errors']))
    return dict(query_count=len(rows),query_error_count=sum(len(row['errors']) for row in rows),
        sampled_original_top_upward_support_ray_count=sum(bool(d['matching_original_top_upward_support_hit_indices']) for d in diagnostics),
        original_cart_control_hit=bool(diagnostics[-1]['original_cart_hit_indices']),
        duplicate_XY_ray_count=len(specifications)-len({tuple(row['xy_m']) for row in specifications}),
        all_hits_retained_including_self_unknown_obstacles_and_below_floor=True,
        sample_presence_NOT_continuous_path_support_collision_or_static_admission=True,diagnostics=diagnostics)


async def review_proposed_corridor_cached_queries(result,task_proposal):
    """Read only existing PhysX cache at the completed H4 boundary; no cook."""
    import inspect
    import carb
    import omni.usd
    import omni.timeline
    import omni.physx
    from isaacsim.core.simulation_manager import SimulationManager
    import hospital_h4_subfloor_compact_path_launch as launch
    from hospital_runtime_evidence_io import timeline_state,write_exclusive
    task_path=Path(task_proposal).resolve();task_payload=task_path.read_bytes()
    task,asset,design=load_corridor_bypass_task_proposal(task_path)
    if json.loads(task_payload)!=task:raise ValueError('Task proposal changed while loading; no cached query')
    result=Path(result).resolve()
    if Path(asset['retained_result']).resolve()!=result:raise ValueError('Task must retain its own original H4 result')
    current=await review_medical_assets_retaining_completed_route(result)
    if current!=asset:raise ValueError('The exact original actual asset receipt must remain unchanged')
    raw=launch.preparation.read_small(result)
    context=launch.load_archived_fresh_trial_protocol(launch.FRESH_TRIAL_ROOT/raw['execution_cohort']/'protocol.json')
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    if stage is None:raise ValueError('Existing retained stage required; no reload')
    expected=asset['actual_timeline_before_and_after']
    view=SimulationManager.get_physics_sim_view()
    if view is None:raise ValueError('Existing physics cache required; no initialization or Play')
    tensor=view.create_articulation_view(context.cohort.ROBOT+'/base_link')
    if tensor.count!=1 or tensor.max_links!=13:raise ValueError('The exact retained thirteen-link actor required')
    layers=[stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()]
    def unchanged():
        if (usd.get_stage()!=stage or timeline_state(timeline)!=expected
                or launch.normalize_runtime_evidence(launch.preparation.old.tensor_snapshot(tensor))!=raw['actual_post_route_tensor_snapshot']
                or [stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()]!=layers):
            raise ValueError('Retained H4 boundary changed during cached GETs; no state repair or recovery')
    unchanged();specs=corridor_bypass_query_specifications(task);rows=[];interface_error=None
    try:query=omni.physx.get_physx_scene_query_interface()
    except Exception as error:query=None;interface_error=type(error).__name__+': '+str(error)
    print('[H5 CACHED TASK GET] Reading all 32 fixed rays; no new physics cooking or motion.',flush=True)
    for spec in specs:
        hits=[];errors=[]
        def receive(hit):
            try:
                value=dict(collision=str(hit.collision),position_m=list(map(float,hit.position)),
                    normal=list(map(float,hit.normal)),distance_m=float(hit.distance))
                if (len(value['position_m'])!=3 or len(value['normal'])!=3 or not value['collision']
                        or any(not math.isfinite(x) for x in (*value['position_m'],*value['normal'],value['distance_m']))):
                    raise ValueError('Malformed/nonfinite cached hit: '+repr(value))
                hits.append(value)
            except Exception as error:errors.append(type(error).__name__+': '+str(error))
            return True
        if query is None:errors.append(interface_error or 'Cached scene query interface unavailable')
        else:
            try:query.raycast_all(carb.Float3(*spec['xy_m'],spec['origin_z_m']),carb.Float3(*spec['direction']),spec['length_m'],receive)
            except Exception as error:errors.append(type(error).__name__+': '+str(error))
        hits.sort(key=lambda h:(h['distance_m'],h['collision'],h['position_m'],h['normal']))
        rows.append(dict(id=spec['id'],all_hits=hits,errors=errors));unchanged()
    limits=dict(context.stationary.QUERY_LIMITS);supports=list(context.stationary.SUPPORTS)
    summary=summarize_corridor_cached_queries(specs,rows,supports,float(design['top_height_m']),limits,
        task['selected_original_cart'],context.cohort.ROBOT)
    if task_path.read_bytes()!=task_payload:raise ValueError('Task proposal changed during cached GETs; no result admission')
    pins={**task['input_sha256'],**task['output_sha256'],str(task_path):hashlib.sha256(task_payload).hexdigest()}
    launch.preparation.verify(pins);unchanged()
    report=dict(version='H5_proposed_task_existing_cached_rays_no_motion_v1',
        status='CACHED_RAY_SAMPLES_RECORDED_NOT_H5_MAP_STATIC_OR_DRIVE_ADMISSION',
        retained_result=str(result),original_H4_verdict=raw['verdict'],task_proposal=str(task_path),
        actual_timeline_before_and_after=expected,body_12_joints_13_links_gains_targets_exactly_retained=True,
        Root_Session_exactly_unchanged=True,all_4997_protected_environment_rows=asset['all_4997_protected_environment_rows'],
        source_asset_receipt=str(task['retained_asset_receipt']),specifications=specs,raw_query_rows=rows,
        original_support_colliders=supports,original_top_height_m=float(design['top_height_m']),
        unchanged_original_query_limits=limits,summary=summary,
        cached_samples_NOT_continuous_collision_free_map_or_support_verification=True,
        H4_route_verdict_NOT_relabelled=True,H5_static_or_drive_admission=False,
        no_Play_Stop_Pause_reset_reopen_save_cook_camera_renderer_or_physics_edit=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,analysis_source={f.__name__:inspect.getsource(f) for f in
            (load_corridor_bypass_task_proposal,corridor_bypass_query_specifications,
             summarize_corridor_cached_queries,review_proposed_corridor_cached_queries)})
    key=hashlib.sha256(task_payload).hexdigest()[:16]
    target=GATE/'hospital_h5_geometry_candidates/cached_task_queries'/key/'live_queries.json'
    if target.exists():
        if json.loads(target.read_bytes())!=report:raise ValueError('Existing cached task readback differs; no overwrite')
    else:write_exclusive(target,report)
    print('=== H5 FIXED TASK / EXISTING CACHED QUERY READBACK (NO MOTION) ===',flush=True)
    print('rays / sampled original-top support / cart control / errors:',len(rows),
        summary['sampled_original_top_upward_support_ray_count'],summary['original_cart_control_hit'],summary['query_error_count'],flush=True)
    print('Exact H4 body / 12 joints / 13 links / Root / Session / timeline retained: True',flush=True)
    print('saved:',target,flush=True)
    print('Samples only: no complete H5 map, static/drive admission, retry or new trial.',flush=True)
    return report


def load_corridor_cached_query_receipt(path):
    """Recompute a saved GET transaction; support samples are not free space."""
    import inspect
    path=Path(path).resolve();receipt=json.loads(path.read_bytes())
    task,asset,design=load_corridor_bypass_task_proposal(receipt['task_proposal'])
    functions=(load_corridor_bypass_task_proposal,corridor_bypass_query_specifications,
        summarize_corridor_cached_queries,review_proposed_corridor_cached_queries)
    expected_pins={**task['input_sha256'],**task['output_sha256'],
        str(Path(receipt['task_proposal']).resolve()):hashlib.sha256(Path(receipt['task_proposal']).read_bytes()).hexdigest()}
    if (receipt.get('version')!='H5_proposed_task_existing_cached_rays_no_motion_v1'
            or receipt.get('status')!='CACHED_RAY_SAMPLES_RECORDED_NOT_H5_MAP_STATIC_OR_DRIVE_ADMISSION'
            or receipt.get('analysis_source')!={f.__name__:inspect.getsource(f) for f in functions}
            or receipt.get('input_sha256')!=expected_pins
            or receipt.get('retained_result')!=asset['retained_result']
            or receipt.get('source_asset_receipt')!=task['retained_asset_receipt']
            or receipt.get('original_H4_verdict')!=asset['original_route_verdict']
            or receipt.get('actual_timeline_before_and_after')!=asset['actual_timeline_before_and_after']
            or receipt.get('all_4997_protected_environment_rows')!=asset['all_4997_protected_environment_rows']
            or receipt.get('body_12_joints_13_links_gains_targets_exactly_retained') is not True
            or receipt.get('Root_Session_exactly_unchanged') is not True
            or receipt.get('no_Play_Stop_Pause_reset_reopen_save_cook_camera_renderer_or_physics_edit') is not True
            or receipt.get('H5_static_or_drive_admission') is not False
            or receipt.get('actual_motion_authorized') is not False
            or receipt.get('new_PhysX_trial_count')!=0 or receipt.get('use_for_final_statistics') is not False):
        raise ValueError('Exact unchanged original cached-ray transaction required')
    for name,sha in expected_pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Cached-ray input changed: '+name)
    specs=corridor_bypass_query_specifications(task)
    limits=dict(height_tolerance_m=.0005,horizontal_tolerance_m=.0005,minimum_upward_normal_z=.99)
    supports=['/World/PatientTransportLabGround/CollisionBox','/World/IJCAS_HospitalH4SupportR01/CollisionMesh']
    if (receipt.get('specifications')!=specs or receipt.get('unchanged_original_query_limits')!=limits
            or receipt.get('original_support_colliders')!=supports
            or receipt.get('original_top_height_m')!=design['top_height_m']):
        raise ValueError('Original ray geometry/supports/top/limits changed')
    summary=summarize_corridor_cached_queries(specs,receipt['raw_query_rows'],supports,
        design['top_height_m'],limits,task['selected_original_cart'],
        json.loads(Path(receipt['retained_result']).read_bytes())['physics_preparation_snapshot']['new_robot_root'])
    if receipt.get('summary')!=summary:raise ValueError('Saved cached-ray summary does not match every raw hit')
    return receipt,task,asset,design


def corridor_endpoint_geometry_review(task,receipt,inventory):
    """Separate floor hits from obstacle hits and propose, never assign, a goal.

    The old complete inventory is HISTORICAL outside the 45 medical roots.
    It diagnoses the observed shelf; a new current complete inventory is
    still required. A single fixed 150mm furniture reserve defines the
    proposed endpoint, before any method is planned or driven.
    """
    from hospital_h4_geometry_metrics import rectangle_aabb_distances
    cached_medical_obstacle_roots(inventory)  # validate full original inventory
    if (task.get('version')!='H5_existing_cart_southbound_right_hand_lane_TASK_PROPOSAL_v1'
            or task.get('footprint')!=dict(length_m=1.630,width_m=.965,clearance_m=.010)
            or task['start_xyyaw'][0]!=task['goal_xyyaw'][0]
            or task['start_xyyaw'][2]!=-math.pi/2 or task['goal_xyyaw'][2]!=-math.pi/2):
        raise ValueError('Exact original southbound same-lane task and full footprint required')
    if len(receipt['specifications'])!=len(receipt['raw_query_rows']):
        raise ValueError('Every original query row required')
    seen=set();colliders=[];non_support=[];supports=set(receipt['original_support_colliders'])
    for row in inventory['collision_prims']:
        path=row['path']
        if path in seen:raise ValueError('Duplicate historical collider identity')
        seen.add(path)
        if (not path.startswith('/World/Hospital/') or row['is_active'] is not True
                or row['collision_enabled_value'] is not True):continue
        bound=row.get('world_aligned_bound') or {};lo,hi=bound.get('minimum_m'),bound.get('maximum_m')
        if (not isinstance(lo,list) or not isinstance(hi,list) or len(lo)!=3 or len(hi)!=3
                or any(type(x) not in (int,float) or not math.isfinite(x) for x in (*lo,*hi))
                or any(a>b for a,b in zip(lo,hi))):raise ValueError('Finite complete historical collider bounds required')
        if hi[2]>=.006 and lo[2]<=.450:colliders.append(dict(path=path,world_aligned_bound=bound))
    def screen(pose):
        return [row for row in colliders if rectangle_aabb_distances([pose],row['world_aligned_bound'],clearance=.010)[0]==0.]
    for spec,row in zip(receipt['specifications'],receipt['raw_query_rows']):
        if spec['id']!=row['id']:raise ValueError('Original ordered query identities required')
        for hit in row['all_hits']:
            if hit['collision'] not in supports:
                non_support.append(dict(query_id=spec['id'],query_kind=spec['kind'],xy_m=spec['xy_m'],**hit))
    start_hits=screen(task['start_xyyaw']);goal_hits=screen(task['goal_xyyaw'])
    proposal=None;reserve=.150
    # Do not generalise this task edit to other collisions or choose a route.
    if (not start_hits and goal_hits and all(row['world_aligned_bound']['maximum_m'][1]
            < task['actual_cart_union_AABB']['minimum_m'][1] for row in goal_hits)):
        goal=list(task['goal_xyyaw'])
        goal[1]=max(row['world_aligned_bound']['maximum_m'][1] for row in goal_hits)+1.630/2+.010+reserve
        if task['goal_xyyaw'][1]<goal[1]<task['actual_cart_union_AABB']['minimum_m'][1]-1.630/2-.010:
            proposal=dict(start_xyyaw=list(task['start_xyyaw']),goal_xyyaw=goal,
                rule='Same start/lane/heading; end north of each original goal-blocking furniture AABB plus expanded half-length and fixed 150mm reserve',
                fixed_furniture_reserve_m=reserve,blocking_anchor_paths=[row['path'] for row in goal_hits],
                original_goal_shift_north_m=goal[1]-task['goal_xyyaw'][1],
                cart_south_face_to_new_goal_center_m=task['actual_cart_union_AABB']['minimum_m'][1]-goal[1],
                historical_AABB_goal_screen_overlaps=screen(goal),
                no_planner_cost_trajectory_method_or_drive_outcome_used=True,
                endpoint_proposal_NOT_task_freeze_or_map_static_drive_admission=True)
    return dict(version='H5_cached_floor_obstacle_channels_and_prospective_endpoint_review_v1',
        status='HOLD_ORIGINAL_PROPOSED_GOAL_OBSTACLE_SCREEN' if start_hits or goal_hits else 'ENDPOINT_CANDIDATE_SCREEN_ONLY_CURRENT_COMPLETE_MAP_PENDING',
        original_start_xyyaw=task['start_xyyaw'],original_goal_xyyaw=task['goal_xyyaw'],
        original_start_historical_AABB_overlaps=start_hits,original_goal_historical_AABB_overlaps=goal_hits,
        every_cached_non_support_hit=non_support,
        sampled_support_hit_count=receipt['summary']['sampled_original_top_upward_support_ray_count'],
        cached_query_error_count=receipt['summary']['query_error_count'],
        sampled_floor_presence_does_NOT_mean_obstacle_free=True,
        historical_inventory_NOT_current_complete_geometry_or_cooked_shape_certificate=True,
        original_goal_relabelled_or_live_robot_moved=False,
        prospective_common_endpoint_amendment=proposal,
        live_complete_ROI_inventory_and_new_endpoint_samples_required=True,
        H5_static_or_drive_admission=False,new_PhysX_trial_count=0,use_for_final_statistics=False)


def write_corridor_endpoint_review(path,output=None):
    """Offline source-bound endpoint review; retain the original task/GETs."""
    import inspect
    from hospital_runtime_evidence_io import write_exclusive
    path=Path(path).resolve();receipt,task,asset,_=load_corridor_cached_query_receipt(path)
    inventory_path=GATE/'hospital_h4_geometry_review/complete_inventory_r01/live_collision_inventory.json'
    inventory=json.loads(inventory_path.read_bytes())
    review=corridor_endpoint_geometry_review(task,receipt,inventory)
    pins={**receipt['input_sha256'],str(path):hashlib.sha256(path.read_bytes()).hexdigest(),
        str(inventory_path):hashlib.sha256(inventory_path.read_bytes()).hexdigest()}
    functions=(load_corridor_cached_query_receipt,corridor_endpoint_geometry_review,write_corridor_endpoint_review)
    review.update(cached_queries=str(path),original_task_proposal=receipt['task_proposal'],
        retained_result=receipt['retained_result'],input_sha256=pins,
        analysis_source={f.__name__:inspect.getsource(f) for f in functions})
    key=hashlib.sha256(json.dumps(review,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    target=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/endpoint_reviews'/key)/'endpoint_review.json'
    if target.exists():
        if json.loads(target.read_bytes())!=review:raise ValueError('Existing endpoint review differs; no overwrite')
    else:write_exclusive(target,review)
    return target


def load_corridor_endpoint_review(path):
    """Load only the exact offline endpoint proposal; it admits no motion."""
    import inspect
    path=Path(path).resolve();payload=path.read_bytes();review=json.loads(payload)
    receipt,task,asset,design=load_corridor_cached_query_receipt(review['cached_queries'])
    inventory_path=GATE/'hospital_h4_geometry_review/complete_inventory_r01/live_collision_inventory.json'
    expected=corridor_endpoint_geometry_review(task,receipt,json.loads(inventory_path.read_bytes()))
    expected.update(cached_queries=review['cached_queries'],original_task_proposal=receipt['task_proposal'],
        retained_result=receipt['retained_result'],
        input_sha256={**receipt['input_sha256'],
            str(Path(review['cached_queries']).resolve()):hashlib.sha256(Path(review['cached_queries']).read_bytes()).hexdigest(),
            str(inventory_path):hashlib.sha256(inventory_path.read_bytes()).hexdigest()},
        analysis_source={f.__name__:inspect.getsource(f) for f in
            (load_corridor_cached_query_receipt,corridor_endpoint_geometry_review,write_corridor_endpoint_review)})
    if review!=expected or path.read_bytes()!=payload:raise ValueError('Exact original endpoint review required; no implicit task amendment')
    proposal=review['prospective_common_endpoint_amendment']
    if proposal is None or proposal['historical_AABB_goal_screen_overlaps']:
        raise ValueError('No clear prospective common endpoint proposal; no map review scheduled')
    return review,receipt,task,asset,design


async def read_corridor_collision_inventory(stage,roi,actor,unchanged):
    """Bounded all-stage USD GETs, independent of visibility and the old ROI.

    AABB records only: never cooked-shape export, physics initialization or
    a reduced selection of medically named objects. Yield before large UI
    batches; keep errors and unloaded payloads rather than filling holes.
    """
    import time
    import omni.kit.app
    from pxr import Usd,UsdGeom,UsdPhysics
    if (len(roi)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in roi)
            or roi[0]>=roi[1] or roi[2]>=roi[3]):raise ValueError('Finite positive explicit ROI required')
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide],useExtentsHint=False,ignoreVisibility=True)
    records=[];unloaded=[];seen=set();errors=[];traversed=0
    began=last_yield=heartbeat=time.monotonic()
    predicate=Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    unchanged()
    for prim in Usd.PrimRange(stage.GetPseudoRoot(),predicate):
        traversed+=1;path=str(prim.GetPath())
        if prim.IsActive() and path!='/':
            for error in prim.GetPrimIndex().localErrors:
                message='USD composition error: '+str(error)
                if message not in errors:errors.append(message)
        if prim.IsActive() and prim.HasPayload() and not prim.IsLoaded():unloaded.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if path in seen:raise ValueError('Duplicate current collider identity')
            seen.add(path);flag=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            row=dict(path=path,type_name=prim.GetTypeName(),is_active=bool(prim.IsActive()),
                is_instance_proxy=bool(prim.IsInstanceProxy()),collision_enabled_value=flag)
            if not prim.IsActive():row['classification']='inactive'
            elif flag is False:row['classification']='disabled'
            elif path==actor or path.startswith(actor+'/'):row['classification']='robot_self_excluded'
            else:
                try:
                    value=cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    lo,hi=list(map(float,value.GetMin())),list(map(float,value.GetMax()))
                    if (any(not math.isfinite(x) for x in (*lo,*hi)) or any(a>b for a,b in zip(lo,hi))):
                        raise ValueError('Unavailable finite current collider bounds')
                    row['world_aligned_bound']=dict(minimum_m=lo,maximum_m=hi,size_m=[b-a for a,b in zip(lo,hi)])
                    outside=hi[0]<roi[0] or lo[0]>roi[1] or hi[1]<roi[2] or lo[1]>roi[3]
                    row['classification']='outside_ROI' if outside else 'enabled_in_ROI'
                except Exception as error:
                    row['classification']='bound_unavailable';errors.append(path+': '+type(error).__name__+': '+str(error))
            records.append(row)
        now=time.monotonic()
        if now-began>180.:raise TimeoutError('Bounded current collider GET exceeded 180s; no retry')
        if traversed%64==0 or now-last_yield>=.05:
            await omni.kit.app.get_app().next_update_async();unchanged();last_yield=time.monotonic()
        if now-heartbeat>=15.:
            print('[H5 MAP GET] all-stage prims=%d colliders=%d; same PAUSED state, no motion' % (traversed,len(records)),flush=True)
            heartbeat=now
    unchanged()
    from collections import Counter
    return dict(all_stage_prims_traversed=traversed,collision_prim_count=len(records),collision_prims=records,
        classification_counts=dict(Counter(row['classification'] for row in records)),
        active_unloaded_payload_paths=unloaded,errors=errors,
        authored_inventory_coverage_complete=bool(records) and not unloaded and not errors,
        geometry_basis='CURRENT_USD_AABBs_NOT_COOKED_SHAPES_OR_CONTINUOUS_SUPPORT',
        invisible_nonmedical_and_previously_outside_ROI_colliders_NOT_filtered=True)


async def review_corridor_task_map_retaining_route(result,endpoint_review):
    """Current complete H5 inventory and PROPOSED endpoint rays; no motion."""
    import inspect
    import carb
    import omni.kit.app
    import omni.usd
    import omni.timeline
    import omni.physx
    from isaacsim.core.simulation_manager import SimulationManager
    import hospital_h4_subfloor_compact_path_launch as launch
    from hospital_runtime_evidence_io import timeline_state,write_exclusive
    review_path=Path(endpoint_review).resolve();payload=review_path.read_bytes()
    review,prior,task,asset,design=load_corridor_endpoint_review(review_path)
    if review!=json.loads(payload) or str(Path(result).resolve())!=review['retained_result']:
        raise ValueError('Exact own endpoint proposal/result required')
    if await review_medical_assets_retaining_completed_route(result)!=asset:
        raise ValueError('Original retained-route asset receipt changed')
    raw=launch.preparation.read_small(result)
    context=launch.load_archived_fresh_trial_protocol(launch.FRESH_TRIAL_ROOT/raw['execution_cohort']/'protocol.json')
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    expected=asset['actual_timeline_before_and_after'];view=SimulationManager.get_physics_sim_view()
    if stage is None or view is None:raise ValueError('Existing retained USD/tensor cache required; no initialization')
    tensor=view.create_articulation_view(context.cohort.ROBOT+'/base_link')
    if tensor.count!=1 or tensor.max_links!=13:raise ValueError('Exact completed actor required')
    layers=[stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()]
    def unchanged():
        if (usd.get_stage()!=stage or timeline_state(timeline)!=expected
                or launch.normalize_runtime_evidence(launch.preparation.old.tensor_snapshot(tensor))!=raw['actual_post_route_tensor_snapshot']
                or [stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()]!=layers):
            raise ValueError('Retained H4 body/joints/links/layers/timeline changed; no repair or retry')
    proposal=review['prospective_common_endpoint_amendment'];candidate=dict(task)
    candidate.update(start_xyyaw=proposal['start_xyyaw'],goal_xyyaw=proposal['goal_xyyaw'])
    # This copy is ONLY a named query geometry. No live task/actor is authored.
    specs=corridor_bypass_query_specifications(candidate)
    west,east=task['original_corridor_width_bounds_m']
    roi=[west-.5,east+.5,task['goal_xyyaw'][1]-1.630/2-.010-.5,
        task['start_xyyaw'][1]+1.630/2+.010+.5]
    print('[H5 MAP GET] Complete all-stage collider GET + proposed endpoint samples; NO Play/Stop/reset.',flush=True)
    unchanged();inventory=await read_corridor_collision_inventory(stage,roi,context.cohort.ROBOT,unchanged)
    rows=[];interface_error=None
    try:query=omni.physx.get_physx_scene_query_interface()
    except Exception as error:query=None;interface_error=type(error).__name__+': '+str(error)
    for spec in specs:
        hits=[];errors=[]
        def receive(hit):
            try:
                value=dict(collision=str(hit.collision),position_m=list(map(float,hit.position)),
                    normal=list(map(float,hit.normal)),distance_m=float(hit.distance))
                if (not value['collision'] or len(value['position_m'])!=3 or len(value['normal'])!=3
                        or any(not math.isfinite(x) for x in (*value['position_m'],*value['normal'],value['distance_m']))):
                    raise ValueError('Malformed/nonfinite cached hit: '+repr(value))
                hits.append(value)
            except Exception as error:errors.append(type(error).__name__+': '+str(error))
            return True
        if query is None:errors.append(interface_error or 'Cached query unavailable')
        else:
            try:query.raycast_all(carb.Float3(*spec['xy_m'],spec['origin_z_m']),carb.Float3(*spec['direction']),spec['length_m'],receive)
            except Exception as error:errors.append(type(error).__name__+': '+str(error))
        hits.sort(key=lambda h:(h['distance_m'],h['collision'],h['position_m'],h['normal']))
        rows.append(dict(id=spec['id'],all_hits=hits,errors=errors));unchanged()
        await omni.kit.app.get_app().next_update_async();unchanged()
    summary=summarize_corridor_cached_queries(specs,rows,list(context.stationary.SUPPORTS),design['top_height_m'],
        dict(context.stationary.QUERY_LIMITS),task['selected_original_cart'],context.cohort.ROBOT)
    environment=await launch.fresh_environment_readback(context,stage,timeline,'H5_complete_map_GET_only')
    if environment!=asset['all_4997_protected_environment_rows']:raise ValueError('Protected environment fingerprint changed')
    pins={**review['input_sha256'],str(review_path):hashlib.sha256(payload).hexdigest()}
    launch.preparation.verify(pins);unchanged()
    if review_path.read_bytes()!=payload:raise ValueError('Endpoint review changed during GETs')
    report=dict(version='H5_current_all_stage_AABB_inventory_prospective_endpoint_cached_queries_v1',
        status='CURRENT_AUTHORED_MAP_INPUTS_RECORDED_NOT_H5_STATIC_OR_DRIVE_ADMISSION',
        retained_result=str(Path(result).resolve()),original_H4_verdict=raw['verdict'],endpoint_review=str(review_path),
        original_task_start_goal_preserved=[task['start_xyyaw'],task['goal_xyyaw']],
        prospective_common_endpoint_amendment=proposal,query_geometry_is_NOT_an_authored_task_or_robot_pose=True,
        review_roi_xy_m=roi,inventory=inventory,specifications=specs,raw_query_rows=rows,summary=summary,
        original_support_colliders=list(context.stationary.SUPPORTS),original_top_height_m=design['top_height_m'],
        unchanged_original_query_limits=dict(context.stationary.QUERY_LIMITS),
        actual_timeline_before_and_after=expected,body_12_joints_13_links_gains_targets_exactly_retained=True,
        Root_Session_exactly_unchanged=True,all_4997_protected_environment_rows=environment,
        no_Play_Stop_Pause_reset_reopen_save_cook_camera_renderer_or_physics_edit=True,
        H5_static_or_drive_admission=False,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,analysis_source={f.__name__:inspect.getsource(f) for f in
            (load_corridor_endpoint_review,read_corridor_collision_inventory,review_corridor_task_map_retaining_route)})
    target=GATE/'hospital_h5_geometry_candidates/current_task_map'/hashlib.sha256(payload).hexdigest()[:16]/'live_map_inputs.json'
    if target.exists():
        if json.loads(target.read_bytes())!=report:raise ValueError('Existing current map GETs differ; no overwrite')
    else:write_exclusive(target,report)
    print('=== H5 CURRENT COMPLETE INVENTORY / PROSPECTIVE ENDPOINT QUERY GETs ===',flush=True)
    print('colliders / complete / errors:',inventory['collision_prim_count'],inventory['authored_inventory_coverage_complete'],inventory['errors'],flush=True)
    print('sampled support / cart / query errors:',summary['sampled_original_top_upward_support_ray_count'],summary['original_cart_control_hit'],summary['query_error_count'],flush=True)
    print('Retained H4 body/12 joints/13 links/Root/Session/timeline: EXACTLY unchanged',flush=True)
    print('saved:',target,flush=True)
    print('Map inputs only; no H5 task freeze, static, drive, replan or final-statistics admission.',flush=True)
    return report


def schedule_corridor_map_review(result,endpoint_review):
    """One observable current-map GET; duplicate clicks do not retry."""
    import asyncio
    import builtins
    import traceback
    key='ijcas_h5_current_corridor_map_get_task';previous=getattr(builtins,key,None)
    if previous is not None:
        if not previous.done():print('[H5 MAP GET] Existing task RUNNING; no duplicate.',flush=True)
        elif previous.cancelled():print('[H5 MAP GET] Existing task CANCELLED; no automatic retry.',flush=True)
        elif previous.exception() is not None:
            error=previous.exception();traceback.print_exception(type(error),error,error.__traceback__)
            print('[H5 MAP GET] Existing task FAILED; no automatic retry.',flush=True)
        else:print('[H5 MAP GET] Existing task COMPLETED; saved GET result retained, no duplicate.',flush=True)
        return previous
    for other in ('ijcas_h5_original_assets_get_task','ijcas_h5_corridor_cached_queries_task',
            'ijcas_h5_corrected_goal_corner_get_task',
            'ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        task=getattr(builtins,other,None)
        if task is not None and not task.done():raise RuntimeError('Another task is active; no concurrent map GET')
    task=asyncio.ensure_future(review_corridor_task_map_retaining_route(result,endpoint_review));setattr(builtins,key,task)
    def finished(value):
        if value.cancelled():print('[H5 MAP GET] Cancelled; no retry or scene command.',flush=True)
        elif value.exception() is not None:
            error=value.exception();print('[H5 MAP GET] Read-only task failed; no recovery/retry.',flush=True)
            traceback.print_exception(type(error),error,error.__traceback__)
    task.add_done_callback(finished)
    print('[H5 MAP GET] Scheduled complete current inventory / proposed endpoint GETs ONLY; NO motion.',flush=True)
    return task


def corrected_corridor_endpoint_geometry(task,review):
    """Rebuild BOTH endpoint centres and corners after a task-only amendment.

    V1 map GETs moved the centre/grid but retained original task corners.
    Preserve those raw GETs; never count them as the four new goal corners.
    This geometry is for future common planning/preflight, not a live edit.
    """
    import copy
    from hospital_geometry_tools import footprint_polygon
    from hospital_h4_support_design import ccw
    proposal=review['prospective_common_endpoint_amendment']
    if (proposal is None or proposal['historical_AABB_goal_screen_overlaps']
            or proposal['start_xyyaw']!=task['start_xyyaw']):
        raise ValueError('Exact source-bound common endpoint proposal required')
    candidate=copy.deepcopy(task)
    candidate.update(start_xyyaw=proposal['start_xyyaw'],goal_xyyaw=proposal['goal_xyyaw'],
        expanded_endpoint_footprints_xy=[ccw(footprint_polygon(pose).tolist())
            for pose in (proposal['start_xyyaw'],proposal['goal_xyyaw'])])
    specs=corridor_bypass_query_specifications(candidate)
    return dict(version='H5_common_cart_bypass_endpoint_geometry_CORNER_RECOMPUTED_v2',
        start_xyyaw=candidate['start_xyyaw'],goal_xyyaw=candidate['goal_xyyaw'],footprint=candidate['footprint'],
        expanded_endpoint_footprints_xy=candidate['expanded_endpoint_footprints_xy'],
        corrected_endpoint_grid_control_query_specifications=specs,
        selected_original_cart=task['selected_original_cart'],original_cart_union_AABB=task['actual_cart_union_AABB'],
        original_lane_rule=task['lane_rule'],endpoint_amendment_rule=proposal['rule'],
        endpoint_rule_is_NOT_original_symmetric_2p5m_buffer=True,
        no_method_cost_trajectory_or_motion_outcome_used=True,
        actual_motion_authorized=False,task_frozen=False,use_for_final_statistics=False)


def load_current_corridor_map_receipt(path):
    """Verify all V1 map GETs and expose, not repair, their stale four corners."""
    import inspect,copy
    from collections import Counter
    path=Path(path).resolve();raw=json.loads(path.read_bytes())
    review,prior,task,asset,design=load_corridor_endpoint_review(raw['endpoint_review'])
    functions=(load_corridor_endpoint_review,read_corridor_collision_inventory,review_corridor_task_map_retaining_route)
    pins={**review['input_sha256'],str(Path(raw['endpoint_review']).resolve()):hashlib.sha256(Path(raw['endpoint_review']).read_bytes()).hexdigest()}
    if (raw.get('version')!='H5_current_all_stage_AABB_inventory_prospective_endpoint_cached_queries_v1'
            or raw.get('status')!='CURRENT_AUTHORED_MAP_INPUTS_RECORDED_NOT_H5_STATIC_OR_DRIVE_ADMISSION'
            or raw.get('analysis_source')!={f.__name__:inspect.getsource(f) for f in functions}
            or raw.get('input_sha256')!=pins or raw.get('retained_result')!=asset['retained_result']
            or raw.get('actual_timeline_before_and_after')!=asset['actual_timeline_before_and_after']
            or raw.get('all_4997_protected_environment_rows')!=asset['all_4997_protected_environment_rows']
            or raw.get('prospective_common_endpoint_amendment')!=review['prospective_common_endpoint_amendment']
            or raw.get('original_task_start_goal_preserved')!=[task['start_xyyaw'],task['goal_xyyaw']]
            or any(raw.get(k) is not True for k in ('body_12_joints_13_links_gains_targets_exactly_retained',
                'Root_Session_exactly_unchanged','no_Play_Stop_Pause_reset_reopen_save_cook_camera_renderer_or_physics_edit'))
            or raw.get('H5_static_or_drive_admission') is not False or raw.get('actual_motion_authorized') is not False
            or raw.get('new_PhysX_trial_count')!=0 or raw.get('use_for_final_statistics') is not False):
        raise ValueError('Exact original current-map GET transaction required; no reinterpretation as a trial')
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Current-map source changed: '+name)
    # Reproduce the ACTUAL old query definition, including its four old corners.
    mixed=copy.deepcopy(task);mixed.update(start_xyyaw=review['prospective_common_endpoint_amendment']['start_xyyaw'],
        goal_xyyaw=review['prospective_common_endpoint_amendment']['goal_xyyaw'])
    if raw['specifications']!=corridor_bypass_query_specifications(mixed):raise ValueError('Actual V1 ray coordinates changed')
    actor=json.loads(Path(raw['retained_result']).read_bytes())['physics_preparation_snapshot']['new_robot_root']
    if (raw['original_support_colliders']!=prior['original_support_colliders']
            or raw['original_top_height_m']!=design['top_height_m']
            or raw['unchanged_original_query_limits']!=prior['unchanged_original_query_limits']):
        raise ValueError('Native support/query references changed')
    summary=summarize_corridor_cached_queries(raw['specifications'],raw['raw_query_rows'],raw['original_support_colliders'],
        raw['original_top_height_m'],raw['unchanged_original_query_limits'],task['selected_original_cart'],actor)
    if raw['summary']!=summary:raise ValueError('Current-map query summary does not reproduce every raw hit')
    inventory=raw['inventory'];rows=inventory['collision_prims'];roi=raw['review_roi_xy_m']
    if (not inventory['authored_inventory_coverage_complete'] or inventory['errors'] or inventory['active_unloaded_payload_paths']
            or inventory['collision_prim_count']!=len(rows) or len({r['path'] for r in rows})!=len(rows)
            or inventory['classification_counts']!=dict(Counter(r['classification'] for r in rows))
            or inventory['all_stage_prims_traversed']<len(rows)):
        raise ValueError('Complete unique current all-stage inventory required')
    west,east=task['original_corridor_width_bounds_m']
    if roi!=[west-.5,east+.5,task['goal_xyyaw'][1]-1.630/2-.010-.5,task['start_xyyaw'][1]+1.630/2+.010+.5]:
        raise ValueError('Exact original task-derived current ROI required')
    for row in rows:
        name=row['path']
        if not row['is_active']:classification='inactive'
        elif row['collision_enabled_value'] is False:classification='disabled'
        elif name==actor or name.startswith(actor+'/'):classification='robot_self_excluded'
        else:
            bound=row.get('world_aligned_bound') or {};lo,hi=bound.get('minimum_m'),bound.get('maximum_m')
            if (not isinstance(lo,list) or not isinstance(hi,list) or len(lo)!=3 or len(hi)!=3
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in (*lo,*hi))
                    or any(a>b for a,b in zip(lo,hi))):raise ValueError('Every enabled current collider needs finite original bounds')
            classification='outside_ROI' if hi[0]<roi[0] or lo[0]>roi[1] or hi[1]<roi[2] or lo[1]>roi[3] else 'enabled_in_ROI'
        if row['classification']!=classification:raise ValueError('Current collider classification does not match its own flags/bounds/ROI')
    return raw,review,task,design


def build_corridor_common_grid(raw,review,task,design,resolution=.010):
    """All-current-collider grid + exact authored support, not a drive PASS."""
    import numpy as np
    from hospital_geometry_tools import polygon_cells,pose_collision
    from hospital_h4_support_design import rectangle,ccw,uncovered,area
    from hospital_h4_geometry_metrics import rectangle_aabb_distances
    if resolution!=.010:raise ValueError('Preserved common 10mm planning raster required')
    geometry=corrected_corridor_endpoint_geometry(task,review);roi=raw['review_roi_xy_m']
    origin=np.floor(np.array([roi[0],roi[2]])/resolution)*resolution
    shape=np.ceil((np.array([roi[1],roi[3]])-origin)/resolution).astype(int)[::-1]
    occupied=np.zeros(shape,dtype=bool);included=[];excluded=[]
    for row in raw['inventory']['collision_prims']:
        if row['classification']!='enabled_in_ROI':continue
        lo,hi=row['world_aligned_bound']['minimum_m'],row['world_aligned_bound']['maximum_m']
        if hi[2]<.006 or lo[2]>.450:
            excluded.append(dict(path=row['path'],reason='outside_preserved_6mm_to_450mm_obstacle_slab',bound=row['world_aligned_bound']));continue
        rr,cc=polygon_cells([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],origin,resolution,shape)
        occupied[rr,cc]=True;included.append(dict(path=row['path'],world_aligned_bound=row['world_aligned_bound'],occupied_cell_count=int(len(rr))))
    x0=origin[0]+np.arange(shape[1])*resolution;y0=origin[1]+np.arange(shape[0])*resolution
    x1,y1=x0+resolution,y0+resolution
    in_roi=(y0[:,None]>=roi[2])&(y1[:,None]<=roi[3])&(x0[None,:]>=roi[0])&(x1[None,:]<=roi[1])
    occupied[~in_roi]=True  # Unchecked slivers outside the recorded ROI are NOT free.
    b=design['canonical_unchanged_support_bound']
    covers=[rectangle(b['minimum_m'][0],b['maximum_m'][0],b['minimum_m'][1],b['maximum_m'][1]),
        *[ccw(p) for p in design['nonoverlapping_support_rectangles']]]
    # Non-overlapping axis-aligned authored rectangles; compute area, not just
    # cell-centre membership, including cells spanning the unchanged join.
    support_area=np.zeros(shape,dtype=float)
    for poly in covers:
        lo=np.min(poly,axis=0);hi=np.max(poly,axis=0)
        if any(x not in (lo[0],hi[0]) or y not in (lo[1],hi[1]) for x,y in poly):raise ValueError('Exact axis-aligned support rectangles required')
        dx=np.maximum(0.,np.minimum(x1,hi[0])-np.maximum(x0,lo[0]));dy=np.maximum(0.,np.minimum(y1,hi[1])-np.maximum(y0,lo[1]))
        support_area+=dy[:,None]*dx[None,:]
    supported=support_area>=resolution**2-1e-12
    conservative=occupied|~supported
    endpoint=[]
    for name,pose,fp in zip(('start','goal'),(geometry['start_xyyaw'],geometry['goal_xyyaw']),geometry['expanded_endpoint_footprints_xy']):
        missing=sum(abs(area(poly)) for poly in uncovered([fp],covers))
        overlaps=[row['path'] for row in included if rectangle_aabb_distances([pose],row['world_aligned_bound'],clearance=.010)[0]==0.]
        endpoint.append(dict(name=name,world_xyyaw=pose,source_AABB_overlaps=overlaps,
            authored_support_missing_full_footprint_area_m2=missing,
            common_raster_full_footprint_clear=not pose_collision(conservative,pose,origin,resolution)))
    corrected_specs=geometry['corrected_endpoint_grid_control_query_specifications'];actual={s['id']:s for s in raw['specifications']}
    mismatches=[dict(id=s['id'],actually_queried_xy_m=actual[s['id']]['xy_m'],required_prospective_xy_m=s['xy_m'])
        for s in corrected_specs if actual[s['id']]!=s]
    dx,dy=geometry['goal_xyyaw'][0]-geometry['start_xyyaw'][0],geometry['goal_xyyaw'][1]-geometry['start_xyyaw'][1]
    yaw=geometry['start_xyyaw'][2]
    report=dict(version='H5_CURRENT_COMPLETE_AABB_COMMON_GRID_CORRECTED_TASK_GEOMETRY_v2',
        status='COMMON_AUTHORED_GRID_TASK_CANDIDATE_OFFLINE_PLANNING_ONLY',geometry=geometry,
        start_xyyaw=geometry['start_xyyaw'],goal_xyyaw=geometry['goal_xyyaw'],
        local_goal_xyyaw=[math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy,0.],
        raster=dict(origin_xy_m=origin.tolist(),resolution_m=resolution,shape_rows_columns=shape.tolist()),
        obstacle_height_slab_m=[.006,.450],included_current_obstacle_AABBs=included,excluded_current_ROI_bounds_by_height=excluded,
        exact_same_authored_support_rectangles=covers,full_footprint_endpoints=endpoint,
        current_complete_inventory_collision_count=raw['inventory']['collision_prim_count'],
        current_complete_inventory_ROI_count=raw['inventory']['classification_counts']['enabled_in_ROI'],
        actual_v1_map_ray_geometry_mismatches=mismatches,
        original_map_GETs_preserved_NOT_relabelled_as_corrected_corner_measurements=True,
        corrected_goal_corner_live_queries_complete=not mismatches,
        corrected_goal_corner_queries_required_before_any_drive_admission=True,
        source_cell_coverage_numeric_area_precision_m2=1e-12,
        occupied_or_unsupported_cell_count=int(np.count_nonzero(conservative)),
        current_same_4997_row_environment=raw['all_4997_protected_environment_rows'],
        authored_AABB_grid_NOT_cooked_collision_or_continuous_physical_support_certificate=True,
        common_grid_does_NOT_select_any_method_or_trajectory=True,
        no_robot_obstacle_floor_material_control_gain_or_timeline_change=True,
        H5_static_or_drive_admission=False,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    return report,conservative,supported


def write_corridor_common_grid(path,output=None):
    """Offline current-data map export; no scene writes or physical replay."""
    import inspect
    import numpy as np
    from hospital_runtime_evidence_io import write_exclusive
    path=Path(path).resolve();raw,review,task,design=load_current_corridor_map_receipt(path)
    report,occupied,supported=build_corridor_common_grid(raw,review,task,design)
    pins={**raw['input_sha256'],str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    report.update(current_live_map_receipt=str(path),input_sha256=pins,
        analysis_source={f.__name__:inspect.getsource(f) for f in
            (corrected_corridor_endpoint_geometry,load_current_corridor_map_receipt,build_corridor_common_grid,write_corridor_common_grid)})
    report=json.loads(json.dumps(report,allow_nan=False))  # Freeze JSON-native representation once.
    key=hashlib.sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/common_map'/key).resolve()
    target=output/'common_map_candidate.json';grid=output/'common_occupancy.npz';figure=output/'common_task_map.png'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if ({k:v for k,v in saved.items() if k!='output_sha256'}!=report
                or any(not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in saved['output_sha256'].items())):
            raise ValueError('Existing common-map output differs; no overwrite')
        return target
    if grid.exists() or figure.exists():raise ValueError('Unfinalized common-map output exists; no overwrite')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(grid,occupied=occupied,supported_full_cells=supported,
        origin_xy_m=report['raster']['origin_xy_m'],resolution_m=report['raster']['resolution_m'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from hospital_geometry_tools import footprint_polygon
    fig,ax=plt.subplots(figsize=(6,9),layout='constrained');origin=report['raster']['origin_xy_m'];res=report['raster']['resolution_m']
    ax.imshow(occupied,origin='lower',cmap='Greys',vmin=0,vmax=1,
        extent=[origin[0],origin[0]+occupied.shape[1]*res,origin[1],origin[1]+occupied.shape[0]*res])
    for pose,color,label in ((report['start_xyyaw'],'#138a5d','Same start'),(report['goal_xyyaw'],'#157f9f','Proposed common goal'),(task['goal_xyyaw'],'#bd423b','Original goal: shelf overlap')):
        ax.add_patch(Polygon(footprint_polygon(pose),facecolor=color,edgecolor=color,alpha=.3,label=label))
        ax.arrow(*pose[:2],.38*math.cos(pose[2]),.38*math.sin(pose[2]),color=color,width=.025,length_includes_head=True)
    cart=task['actual_cart_union_AABB'];lo,hi=cart['minimum_m'],cart['maximum_m']
    ax.annotate('Original parked cart',((lo[0]+hi[0])/2,(lo[1]+hi[1])/2),xytext=(19.1,25.9),
        arrowprops=dict(arrowstyle='->'),fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax.set(aspect='equal',xlabel='Hospital world X (m)',ylabel='Hospital world Y (m)',
        title='H5 shared source map / task candidate\nAll current colliders + unchanged authored support')
    ax.legend(loc='upper right',fontsize=8)
    fig.text(.5,.01,'No route, model, static or drive PASS.\nFour corrected goal-corner cached rays remain unmeasured.',ha='center',fontsize=8)
    try:fig.savefig(figure,dpi=160)
    finally:plt.close(fig)
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Common-map input changed during export')
    write_exclusive(target,dict(**report,output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (grid,figure)}))
    print('Saved current-data H5 common grid / corrected task candidate:',target)
    print('Endpoint full-footprint raster checks:',report['full_footprint_endpoints'])
    print('Unmeasured corrected goal-corner rays:',len(report['actual_v1_map_ray_geometry_mismatches']))
    print('No new query, motion, method PASS, task freeze or drive admission.')
    return target


def write_corridor_planning_probe(map_path,output=None):
    """One bounded all-four OFFLINE search on the exact same corrected grid.

    Reuse the original primitive propagation, selectors, costs and model.
    Bypass only H1's task-specific three-segment offset connector: it cannot
    express a same-lane obstacle bypass. Generic weighted A* and P refinement
    are unchanged. This is declared development, not an unchanged-H1/final
    comparison or a runtime bundle; no query or physics command is issued.
    """
    import inspect
    import numpy as np
    from types import FunctionType
    from dataclasses import asdict
    import plan_hospital_h1 as shared
    import patient_initial_stability.steering_lattice as lattice
    import patient_initial_stability.rigid_body_edge as rigid
    import patient_initial_stability.hierarchical_dynamics_planner as refinement
    from hospital_runtime_evidence_io import write_exclusive
    map_path=Path(map_path).resolve();manifest=json.loads(map_path.read_bytes())
    # Rebuild from all actual data, including the explicit missing-corner flag.
    if write_corridor_common_grid(manifest['current_live_map_receipt'],map_path.parent)!=map_path:
        raise ValueError('Exact current-data common-map export required')
    if any(not row['common_raster_full_footprint_clear'] or row['source_AABB_overlaps']
            or row['authored_support_missing_full_footprint_area_m2']>1e-9 for row in manifest['full_footprint_endpoints']):
        raise ValueError('Blocked/unsupported common endpoint; no offline method search')
    grid_path=map_path.parent/'common_occupancy.npz'
    with np.load(grid_path) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    model_path=GATE/'planning/hospital_h1_planning_r06/P_dynamics_full_envelope.json'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!='1d3f6978a5bb3bb6085a2d8882fe3855f1811599197cbc4365a223d874b292ac':
        raise ValueError('Preserved original H1/H4 Newton-Euler parameter reference required')
    bounds=json.loads(model_path.read_bytes())['rigid_body_parameter_bounds']
    model=rigid.MassNormalizedRigidBodyEdgeModel(rigid.RigidBodyParameterBounds(**bounds))
    functions=(corrected_corridor_endpoint_geometry,load_current_corridor_map_receipt,build_corridor_common_grid,
        write_corridor_common_grid,write_corridor_planning_probe)
    sources=[map_path,grid_path,model_path,Path(shared.__file__),Path(lattice.__file__),Path(rigid.__file__),Path(refinement.__file__)]
    pins={**manifest['input_sha256'],**manifest['output_sha256'],
        **{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    protocol=dict(version='H5_CURRENT_MAP_GENERIC_LATTICE_AND_ORIGINAL_P_REFINEMENT_OFFLINE_PROBE_v1',
        scenario='H5_obstacle_bypass_return',common_geometry=str(map_path),common_grid=str(grid_path),
        start_world_xyyaw=manifest['start_xyyaw'],goal_world_xyyaw=manifest['goal_xyyaw'],
        common_per_method_CPU_wall_budget_s=45.,common_node_limit=20000,
        original_H1_specific_offset_connector_bypassed_for_all_methods=True,
        unchanged_generic_lattice_and_P_refinement_source=str(Path(shared.__file__).resolve()),
        primitive_mode_cost_dynamics_and_model_constants_unchanged=True,
        P_refines_successful_shared_baseline_candidates_NOT_a_global_joint_state_search=True,
        initial_state_scope='Nominal source-seed standstill planning; no actual release-state or drive admission',
        rigid_body_parameter_bounds=bounds,common_speed_cap_m_s=shared.SPEED,
        common_speed_levels_m_s=list(shared.SPEED_LEVELS),common_acceleration_m_s2=shared.DECELERATION,
        terminal_planning_speed_m_s=shared.TERMINAL_SPEED,
        corrected_goal_corner_live_queries_complete=manifest['corrected_goal_corner_live_queries_complete'],
        pending_query_or_static_does_NOT_block_offline_search_or_admit_motion=True,
        actual_motion_authorized=False,new_PhysX_trial_count=0,use_for_final_statistics=False,
        input_sha256=pins,analysis_source={f.__name__:inspect.getsource(f) for f in functions})
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/planning_probes'/key).resolve()
    target=output/'planning_probe.json';definition=output/'planning_probe_protocol.json'
    if target.exists():
        old=json.loads(target.read_bytes())
        if (old['protocol']!=protocol or not definition.is_file() or json.loads(definition.read_bytes())!=protocol
                or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in old['output_sha256'].items())):
            raise ValueError('Existing planning probe differs; no overwrite or rerun')
        return target
    if definition.exists():raise ValueError('Previous offline search reserved; inspect its records, no automatic second search')
    output.mkdir(parents=True,exist_ok=True);write_exclusive(definition,protocol)
    scope=dict(shared.plan_method.__globals__)
    scope['counterphase_connector']=lambda goal,checker:None
    run=FunctionType(shared.plan_method.__code__,scope,shared.plan_method.__name__,shared.plan_method.__defaults__,shared.plan_method.__closure__)
    pool={};records=[];artifacts=[definition]
    for method in shared.METHODS:
        print('[H5 OFFLINE PLAN] '+method+'; shared 45s/20000-node CPU bound; no Isaac/physics.',flush=True)
        row=run(method,manifest,occupied,origin,resolution,model,45.,20000,pool)
        row=json.loads(json.dumps(row,allow_nan=False))
        row.update(scenario='H5_obstacle_bypass_return',planning_role='OFFLINE_DEVELOPMENT_NOT_ACTUAL_DRIVE_OR_FINAL_COMPARISON',
            common_geometry=str(map_path),all_four_use_same_map_model_and_resource_contract=True,
            new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
        path=output/(method+'.json');write_exclusive(path,row);artifacts.append(path);records.append(row)
        if row['plan_success'] and method!='P_dynamics_full_envelope':pool[method]=row['selected_actions']
        print('[H5 OFFLINE PLAN] '+method+': '+row['status']+'; elapsed=%.2fs'%row['planning_time_s'],flush=True)
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Original common planning inputs changed')
    report=dict(version=protocol['version'],status='BOUNDED_ALL_FOUR_OFFLINE_PLANNING_PROBE_RECORDED',protocol=protocol,
        methods=[dict(method=row['method'],plan_success=row['plan_success'],status=row['status'],
            planning_time_s=row['planning_time_s'],action_count=row.get('action_count'),
            predicted_terminal_position_error_m=row.get('predicted_terminal_position_error_m'),
            predicted_terminal_heading_error_deg=row.get('predicted_terminal_heading_error_deg')) for row in records],
        no_global_infeasibility_or_actual_success_inferred=True,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False,
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    write_exclusive(target,report)
    print('Saved all-four bounded OFFLINE planning probe:',target,flush=True)
    return target


def load_corridor_planning_probe(path):
    """Read a completed original probe; never recreate or resume a search."""
    import plan_hospital_h1 as shared
    path=Path(path).resolve()
    if path.name!='planning_probe.json' or not path.is_file():
        raise ValueError('Existing completed original planning probe required; no automatic search')
    report=json.loads(path.read_bytes());protocol=report['protocol']
    if (report.get('status')!='BOUNDED_ALL_FOUR_OFFLINE_PLANNING_PROBE_RECORDED'
            or report.get('version')!='H5_CURRENT_MAP_GENERIC_LATTICE_AND_ORIGINAL_P_REFINEMENT_OFFLINE_PROBE_v1'
            or report.get('new_PhysX_trial_count')!=0 or report.get('actual_motion_authorized') is not False
            or report.get('use_for_final_statistics') is not False
            or not (path.parent/'planning_probe_protocol.json').is_file()):
        raise ValueError('Complete original offline probe required, not a physical outcome')
    # The existing exporter is strictly idempotent when this target exists.
    # It reconstructs every source/model/map pin and verifies all artifacts.
    if write_corridor_planning_probe(protocol['common_geometry'],path.parent)!=path:
        raise ValueError('Original planning protocol/map mismatch')
    rows=[json.loads((path.parent/(name+'.json')).read_bytes()) for name in shared.METHODS]
    keys=('method','plan_success','status','planning_time_s','action_count',
        'predicted_terminal_position_error_m','predicted_terminal_heading_error_deg')
    if report['methods']!=[{key:row.get(key) for key in keys} for row in rows]:
        raise ValueError('Original probe summary must reproduce all four raw method records')
    if any(row.get('actual_motion_authorized') is not False or row.get('new_PhysX_trial_count')!=0
            or row.get('use_for_final_statistics') is not False for row in rows):
        raise ValueError('Original development method records cannot be promoted')
    return report,rows


def verify_corridor_native_checker(manifest,occupied,origin,resolution,prior_rows):
    """Check the EXISTING exact native SAT against Python on actual H5 data.

    Includes every prior successful spatial edge, deterministic random poses,
    raster-boundary poses, and independent polygon/cell SAT samples. This is
    a numerical software check, not actual cooked collision/model validation.
    """
    import numpy as np
    from hospital_geometry_tools import pose_collision
    from hospital_planning_tools import HospitalEnvelopeChecker,local_to_world
    from hospital_native_envelope import NativeHospitalEnvelopeChecker
    from patient_initial_stability.steering_lattice import State,propagate,KINEMATIC_DYNAMICS
    start=manifest['start_xyyaw'];old=HospitalEnvelopeChecker(occupied,origin,resolution,start)
    native=NativeHospitalEnvelopeChecker(occupied,origin,resolution,start)
    upper=origin+np.asarray(occupied.shape[::-1])*resolution;rng=np.random.default_rng(501711)
    poses=np.column_stack((rng.uniform(origin[0]-.1,upper[0]+.1,1500),
        rng.uniform(origin[1]-.1,upper[1]+.1,1500),rng.uniform(-math.pi,math.pi,1500))).tolist()
    for delta in (-1e-9,-1e-12,0.,1e-12,1e-9):
        for yaw in (0.,math.pi/2,math.pi/4,-math.pi/2):
            a,b=1.630/2+.010,.965/2+.010
            ex=abs(math.cos(yaw))*a+abs(math.sin(yaw))*b
            ey=abs(math.sin(yaw))*a+abs(math.cos(yaw))*b
            poses.extend(((origin[0]+ex+delta,(origin[1]+upper[1])/2,yaw),
                (upper[0]-ex+delta,(origin[1]+upper[1])/2,yaw),
                ((origin[0]+upper[0])/2,origin[1]+ey+delta,yaw),
                ((origin[0]+upper[0])/2,upper[1]-ey+delta,yaw)))
    batch_count=0;route_substeps=0
    for row in prior_rows:
        if not row['plan_success'] or row['method']=='P_dynamics_full_envelope':continue
        state=State(0.,0.,0.,'coordinated_counterphase',0.,0.,False)
        for action in row['selected_actions']:
            state,local,_=propagate(state,action['mode'],action['curvature_1_m'],action['direction'],
                speed_m_s=.050,dynamics=KINEMATIC_DYNAMICS,target_speed_m_s=action['target_speed_m_s'])
            if old(local)!=native(local):raise ValueError('H5 native batch and original full-envelope SAT disagree')
            poses.extend(local_to_world(pose,start) for pose in local)
            batch_count+=1;route_substeps+=len(local)
    poses.extend((manifest['start_xyyaw'],manifest['goal_xyyaw']))
    independent=0
    for index,pose in enumerate(poses):
        expected=old.world_collision(pose)
        if native.world_collision(pose)!=expected:raise ValueError('H5 native and original finite-cell SAT disagree')
        if index%37==0 or index>=len(poses)-2:
            if pose_collision(occupied,pose,origin,resolution,.010)!=expected:
                raise ValueError('H5 independent polygon/cell SAT disagrees')
            independent+=1
    return dict(version='H5_existing_native_exact_SAT_actual_map_equivalence_v1',
        status='SOFTWARE_EQUIVALENCE_VERIFIED_NOT_COOKED_COLLISION_OR_MODEL_VALIDATION',
        deterministic_seed=501711,world_pose_count=len(poses),prior_route_batch_count=batch_count,
        prior_route_substep_count=route_substeps,independent_polygon_cell_checks=independent,
        disagreement_count=0,clearance_m=.010,unchanged_body_length_width_m=[1.630,.965],
        no_quantization_decimation_point_robot_or_reduced_envelope=True,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False)


def write_corridor_native_planning_probe(prior_path,output=None):
    """All-four development probe reusing the already-tested native exact SAT.

    No original source, search costs, node/time budget, map, model or physical
    state is changed. All successes are independently repropagated through
    original polygon/cell SAT. Preserve the prior timeout records separately.
    """
    import inspect,time
    import numpy as np
    from types import FunctionType
    import plan_hospital_h1 as shared
    import hospital_native_envelope as native
    import audit_hospital_h1_plan as independent
    import patient_initial_stability.rigid_body_edge as rigid
    import hospital_geometry_tools as geometry
    import hospital_planning_tools as collision
    from hospital_runtime_evidence_io import write_exclusive
    prior_path=Path(prior_path).resolve();prior,prior_rows=load_corridor_planning_probe(prior_path)
    previous=prior['protocol'];map_path=Path(previous['common_geometry']);manifest=json.loads(map_path.read_bytes())
    with np.load(previous['common_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    begun=time.perf_counter();build=native.initialize_native();warmup=time.perf_counter()-begun
    if build['fast_math_used'] or build['model_or_clearance_changed'] or '-ffp-contract=off' not in build['compile_flags']:
        raise ValueError('Exact existing native SAT without fast-math required')
    files=(prior_path,Path(native.__file__),native.SOURCE,Path(independent.__file__),Path(geometry.__file__),Path(collision.__file__))
    pins={**previous['input_sha256'],**prior['output_sha256'],
        **{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    protocol=dict(version='H5_SAME_GENERIC_LATTICE_ORIGINAL_P_REFINEMENT_NATIVE_EXACT_SAT_OFFLINE_v2',
        prior_probe=str(prior_path),common_geometry=str(map_path),common_grid=previous['common_grid'],
        start_world_xyyaw=manifest['start_xyyaw'],goal_world_xyyaw=manifest['goal_xyyaw'],
        common_per_method_CPU_wall_budget_s=45.,common_node_limit=20000,
        same_task_map_model_modes_primitives_costs_heuristic_and_budgets_as_original_probe=True,
        private_checker_binding_only_no_global_module_patch=True,
        original_H1_specific_offset_connector_bypassed_for_all_methods=True,
        native_build=build,native_compile_and_equivalence_checks_excluded_from_search_budget=True,
        P_reuses_only_successful_common_spatial_candidates=True,
        common_speed_cap_m_s=shared.SPEED,common_speed_levels_m_s=list(shared.SPEED_LEVELS),
        common_acceleration_m_s2=shared.DECELERATION,terminal_planning_speed_m_s=shared.TERMINAL_SPEED,
        rigid_body_parameter_bounds=previous['rigid_body_parameter_bounds'],
        corrected_goal_corner_live_queries_complete=manifest['corrected_goal_corner_live_queries_complete'],
        actual_motion_authorized=False,new_PhysX_trial_count=0,use_for_final_statistics=False,input_sha256=pins,
        analysis_source={f.__name__:inspect.getsource(f) for f in (load_corridor_planning_probe,
            verify_corridor_native_checker,write_corridor_native_planning_probe)})
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/native_planning_probes'/key).resolve()
    target=output/'planning_probe.json';definition=output/'planning_probe_protocol.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (saved['protocol']!=protocol or json.loads(definition.read_bytes())!=protocol
                or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in saved['output_sha256'].items())):
            raise ValueError('Existing native planning probe differs; no overwrite or automatic rerun')
        return target
    if definition.exists():raise ValueError('Native development search already reserved; inspect records, no automatic rerun')
    output.mkdir(parents=True,exist_ok=True);write_exclusive(definition,protocol);artifacts=[definition]
    print('[H5 NATIVE PLAN] Checking existing native SAT against original H5 full-envelope geometry; no Isaac.',flush=True)
    equivalence=verify_corridor_native_checker(manifest,occupied,origin,resolution,prior_rows)
    equivalence.update(native_initialization_time_s_excluded_from_method_budgets=warmup)
    check_path=output/'native_equivalence.json';write_exclusive(check_path,equivalence);artifacts.append(check_path)
    grid_path=output/'planning_occupancy.npz'
    np.savez_compressed(grid_path,occupied=occupied,origin_xy_m=origin,resolution_m=resolution);artifacts.append(grid_path)
    scope=dict(shared.plan_method.__globals__);scope['HospitalEnvelopeChecker']=native.NativeHospitalEnvelopeChecker
    scope['counterphase_connector']=lambda goal,checker:None
    run=FunctionType(shared.plan_method.__code__,scope,shared.plan_method.__name__,shared.plan_method.__defaults__,shared.plan_method.__closure__)
    model=rigid.MassNormalizedRigidBodyEdgeModel(rigid.RigidBodyParameterBounds(**previous['rigid_body_parameter_bounds']))
    records=[];pool={}
    for method in shared.METHODS:
        print('[H5 NATIVE PLAN] '+method+'; unchanged 45s/20000-node search bound.',flush=True)
        row=run(method,manifest,occupied,origin,resolution,model,45.,20000,pool)
        row=json.loads(json.dumps(row,allow_nan=False))
        row.update(scenario='H5_obstacle_bypass_return',common_geometry=str(map_path),
            planning_role='OFFLINE_NATIVE_IMPLEMENTATION_DEVELOPMENT_NOT_PHYSICAL_OR_FINAL_COMPARISON',
            planning_occupancy_digest=hashlib.sha256(occupied.tobytes()).hexdigest(),
            start_world_xyyaw=manifest['start_xyyaw'],goal_local_xyyaw=manifest['local_goal_xyyaw'],
            common_max_command_speed_m_s=shared.SPEED,full_envelope_clearance_m=.010,
            rigid_body_parameter_bounds=previous['rigid_body_parameter_bounds'],
            new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
        path=output/(method+'.json');write_exclusive(path,row);artifacts.append(path)
        audit=None
        if row['plan_success']:
            audit=independent.audit(path)
            audit.update(scenario='H5_obstacle_bypass_return',use_for_final_statistics=False,
                actual_motion_authorized=False,new_PhysX_trial_count=0)
            audit_path=output/(method+'_independent_audit.json');write_exclusive(audit_path,audit);artifacts.append(audit_path)
            if audit['verdict']=='PASS' and method!='P_dynamics_full_envelope':pool[method]=row['selected_actions']
        record={k:row.get(k) for k in ('method','plan_success','status','planning_time_s','action_count','path_length_m',
            'mode_switch_count','predicted_terminal_position_error_m','predicted_terminal_heading_error_deg')}
        record['independent_offline_audit']=audit['verdict'] if audit else 'NOT_RUN_NO_SUCCESSFUL_PLAN'
        records.append(record)
        print('[H5 NATIVE PLAN] '+method+': '+row['status']+'; %.2fs; audit='%row['planning_time_s']+record['independent_offline_audit'],flush=True)
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Original/native development source changed')
    report=dict(version=protocol['version'],status='ALL_FOUR_NATIVE_DEVELOPMENT_PROBES_RECORDED_NOT_PHYSICAL_VALIDATION',
        protocol=protocol,methods=records,prior_probe_preserved=True,
        no_global_infeasibility_or_universal_P_superiority_claim=True,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False,
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    write_exclusive(target,report)
    print('Saved all-four native-exact-SAT H5 OFFLINE probe:',target,flush=True)
    return target


async def review_corrected_corridor_goal_corners(map_path):
    """Only four missing goal-corner GETs plus the unchanged cart control.

    Do not rerun the inventory or old 32 rays, cook, start physics, or author
    any task/pose. Preserve the original V1 mixed-corner receipt verbatim.
    """
    import inspect
    import carb
    import omni.kit.app
    import omni.usd
    import omni.timeline
    import omni.physx
    from isaacsim.core.simulation_manager import SimulationManager
    import hospital_h4_subfloor_compact_path_launch as launch
    from hospital_runtime_evidence_io import timeline_state,write_exclusive
    map_path=Path(map_path).resolve();payload=map_path.read_bytes()
    original,review,task,design=load_current_corridor_map_receipt(map_path)
    geometry=corrected_corridor_endpoint_geometry(task,review)
    specs=[s for s in geometry['corrected_endpoint_grid_control_query_specifications']
        if s['id'].startswith('goal_corner_') or s['id']=='original_cart_control']
    if [s['id'] for s in specs]!=['goal_corner_'+str(i) for i in range(4)]+['original_cart_control']:
        raise ValueError('Exactly four recomputed goal corners and the original cart control required')
    previous={s['id']:s for s in original['specifications']}
    if any(s==previous[s['id']] for s in specs[:4]) or specs[-1]!=previous['original_cart_control']:
        raise ValueError('Must measure precisely the four missing coordinates; no substituted control')
    pins={**original['input_sha256'],str(map_path):hashlib.sha256(payload).hexdigest()}
    output=GATE/'hospital_h5_geometry_candidates/corrected_goal_corner_queries'/hashlib.sha256(payload).hexdigest()[:16]
    target=output/'live_goal_corners.json'
    if target.exists():raise ValueError('Corrected corner GETs already recorded; preserve the result, no overwrite')
    asset=json.loads(Path(task['retained_asset_receipt']).read_bytes())
    if await review_medical_assets_retaining_completed_route(original['retained_result'])!=asset:
        raise ValueError('Exact original retained H4 asset/state receipt required')
    raw=launch.preparation.read_small(original['retained_result'])
    context=launch.load_archived_fresh_trial_protocol(launch.FRESH_TRIAL_ROOT/raw['execution_cohort']/'protocol.json')
    usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
    expected=original['actual_timeline_before_and_after'];view=SimulationManager.get_physics_sim_view()
    if stage is None or view is None:raise ValueError('Existing stage and retained physics cache required; no initialization')
    tensor=view.create_articulation_view(context.cohort.ROBOT+'/base_link')
    if tensor.count!=1 or tensor.max_links!=13:raise ValueError('Exact own retained 13-link actor required')
    layers=[stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()]
    def unchanged():
        if (usd.get_stage()!=stage or timeline_state(timeline)!=expected
                or launch.normalize_runtime_evidence(launch.preparation.old.tensor_snapshot(tensor))!=raw['actual_post_route_tensor_snapshot']
                or [stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()]!=layers):
            raise ValueError('Retained H4 body/joints/links/layers/timeline changed; no recovery or query retry')
    unchanged();rows=[];interface_error=None
    try:query=omni.physx.get_physx_scene_query_interface()
    except Exception as error:query=None;interface_error=type(error).__name__+': '+str(error)
    print('[H5 GOAL CORNERS GET] Four corrected corners + ONE original cart control; existing cache ONLY, NO motion.',flush=True)
    for spec in specs:
        hits=[];errors=[]
        def receive(hit):
            try:
                value=dict(collision=str(hit.collision),position_m=list(map(float,hit.position)),
                    normal=list(map(float,hit.normal)),distance_m=float(hit.distance))
                if (not value['collision'] or len(value['position_m'])!=3 or len(value['normal'])!=3
                        or any(not math.isfinite(x) for x in (*value['position_m'],*value['normal'],value['distance_m']))):
                    raise ValueError('Malformed/nonfinite original cached hit')
                hits.append(value)
            except Exception as error:errors.append(type(error).__name__+': '+str(error))
            return True
        if query is None:errors.append(interface_error or 'Cached query unavailable')
        else:
            try:query.raycast_all(carb.Float3(*spec['xy_m'],spec['origin_z_m']),carb.Float3(*spec['direction']),spec['length_m'],receive)
            except Exception as error:errors.append(type(error).__name__+': '+str(error))
        hits.sort(key=lambda h:(h['distance_m'],h['collision'],h['position_m'],h['normal']))
        rows.append(dict(id=spec['id'],all_hits=hits,errors=errors));unchanged()
        await omni.kit.app.get_app().next_update_async();unchanged()
    summary=summarize_corridor_cached_queries(specs,rows,list(context.stationary.SUPPORTS),design['top_height_m'],
        dict(context.stationary.QUERY_LIMITS),task['selected_original_cart'],context.cohort.ROBOT)
    environment=await launch.fresh_environment_readback(context,stage,timeline,'H5_corrected_goal_corners_GET_only')
    if environment!=original['all_4997_protected_environment_rows']:raise ValueError('Original protected environment changed')
    launch.preparation.verify(pins);unchanged()
    if map_path.read_bytes()!=payload:raise ValueError('Original map receipt changed during GETs')
    corners=summary['diagnostics'][:4]
    checked=(summary['query_error_count']==0 and summary['original_cart_control_hit']
        and all(d['matching_original_top_upward_support_hit_indices'] and not d['every_non_support_hit_indices'] for d in corners))
    report=dict(version='H5_missing_four_corrected_goal_corner_cached_queries_and_original_cart_control_v1',
        status='CORRECTED_GOAL_CORNER_SAMPLES_VERIFIED_NOT_DRIVE_ADMISSION' if checked else 'CORRECTED_GOAL_QUERY_EVIDENCE_REQUIRES_REVIEW',
        original_current_map=str(map_path),retained_result=original['retained_result'],original_H4_verdict=raw['verdict'],
        corrected_goal_xyyaw=geometry['goal_xyyaw'],specifications=specs,raw_query_rows=rows,summary=summary,
        all_four_corrected_corner_support_samples_without_other_hits=bool(checked),
        original_32_rays_and_inventory_preserved_NOT_rerun_or_relabelled=True,
        original_support_colliders=list(context.stationary.SUPPORTS),original_top_height_m=design['top_height_m'],
        unchanged_original_query_limits=dict(context.stationary.QUERY_LIMITS),
        actual_timeline_before_and_after=expected,body_12_joints_13_links_gains_targets_exactly_retained=True,
        Root_Session_exactly_unchanged=True,all_4997_protected_environment_rows=environment,
        no_Play_Stop_Pause_reset_reopen_save_cook_camera_renderer_or_physics_edit=True,
        H5_static_or_drive_admission=False,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,analysis_source={f.__name__:inspect.getsource(f) for f in
            (load_current_corridor_map_receipt,corrected_corridor_endpoint_geometry,review_corrected_corridor_goal_corners)})
    write_exclusive(target,report)
    print('=== H5 FOUR CORRECTED GOAL CORNERS / EXISTING CART CONTROL GETs ===',flush=True)
    print('corner support samples:',sum(bool(d['matching_original_top_upward_support_hit_indices']) for d in corners),'/ 4; cart:',summary['original_cart_control_hit'],'; query errors:',summary['query_error_count'],flush=True)
    print('status:',report['status'],'; same H4 state/layers/timeline retained: True',flush=True)
    print('saved:',target,flush=True)
    print('Samples ONLY. No H5 static, drive, replan, model or final-statistics admission.',flush=True)
    return report


def schedule_corrected_corridor_goal_corners(map_path):
    """Observable single GET task; report DONE/running/error on duplicate clicks."""
    import asyncio,builtins,traceback
    key='ijcas_h5_corrected_goal_corner_get_task';previous=getattr(builtins,key,None)
    if previous is not None:
        if not previous.done():print('[H5 GOAL CORNERS GET] Existing task RUNNING; no duplicate.',flush=True)
        elif previous.cancelled():print('[H5 GOAL CORNERS GET] Existing task CANCELLED; no automatic retry.',flush=True)
        elif previous.exception() is not None:
            error=previous.exception();traceback.print_exception(type(error),error,error.__traceback__)
            print('[H5 GOAL CORNERS GET] Existing task FAILED; no automatic retry.',flush=True)
        else:print('[H5 GOAL CORNERS GET] Existing task COMPLETED:',previous.result()['status'],'; no duplicate.',flush=True)
        return previous
    for other in ('ijcas_h5_original_assets_get_task','ijcas_h5_corridor_cached_queries_task','ijcas_h5_current_corridor_map_get_task',
            'ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        value=getattr(builtins,other,None)
        if value is not None and not value.done():raise RuntimeError('Another live task is active; no concurrent corrected-corner GET')
    task=asyncio.ensure_future(review_corrected_corridor_goal_corners(map_path));setattr(builtins,key,task)
    def finished(value):
        if value.cancelled():print('[H5 GOAL CORNERS GET] Cancelled; no retry or scene command.',flush=True)
        elif value.exception() is not None:
            error=value.exception();traceback.print_exception(type(error),error,error.__traceback__)
            print('[H5 GOAL CORNERS GET] Read-only task failed; no recovery/retry.',flush=True)
    task.add_done_callback(finished)
    print('[H5 GOAL CORNERS GET] Scheduled four missing corners + existing cart control; NO Play/Stop/reset.',flush=True)
    return task


def write_corridor_guided_planning_probe(native_path,output=None):
    """Matched H5 development revision using EXISTING common map guidance.

    Preserve all earlier timeouts/node limits and every method's full action
    space. A relaxed point-distance field only orders search; exact rectangle
    SAT still decides all edges. This is not a narrowed fixed-mode fallback.
    """
    import inspect,time
    import numpy as np
    from types import FunctionType
    import plan_hospital_h1 as shared
    import hospital_h3_navigation_guidance as navigation
    import hospital_native_envelope as native
    import audit_hospital_h1_plan as independent
    import patient_initial_stability.rigid_body_edge as rigid
    from hospital_runtime_evidence_io import write_exclusive
    native_path=Path(native_path).resolve()
    if native_path.name!='planning_probe.json' or not native_path.is_file():
        raise ValueError('Completed existing native probe required; no automatic prior search')
    previous=json.loads(native_path.read_bytes());old=previous['protocol']
    if (previous.get('version')!='H5_SAME_GENERIC_LATTICE_ORIGINAL_P_REFINEMENT_NATIVE_EXACT_SAT_OFFLINE_v2'
            or previous.get('status')!='ALL_FOUR_NATIVE_DEVELOPMENT_PROBES_RECORDED_NOT_PHYSICAL_VALIDATION'
            or previous.get('actual_motion_authorized') is not False or previous.get('use_for_final_statistics') is not False
            or previous.get('new_PhysX_trial_count')!=0 or not (native_path.parent/'planning_probe_protocol.json').is_file()):
        raise ValueError('Original native development receipt required; no physical or final-statistics reinterpretation')
    if write_corridor_native_planning_probe(old['prior_probe'],native_path.parent)!=native_path:
        raise ValueError('Existing native receipt/model/map does not reproduce')
    map_path=Path(old['common_geometry']);manifest=json.loads(map_path.read_bytes())
    with np.load(old['common_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    begun=time.perf_counter()
    guidance=navigation.NavigationGuidance(occupied,origin,resolution,manifest['start_xyyaw'],manifest['goal_xyyaw'])
    preparation_s=time.perf_counter()-begun
    files=(native_path,Path(navigation.__file__))
    pins={**old['input_sha256'],**previous['output_sha256'],
        **{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    protocol=dict(version='H5_ALL_FOUR_EXISTING_MAP_GUIDANCE_NATIVE_EXACT_SAT_OFFLINE_v3',
        previous_native_probe=str(native_path),common_geometry=str(map_path),common_grid=old['common_grid'],
        start_world_xyyaw=manifest['start_xyyaw'],goal_world_xyyaw=manifest['goal_xyyaw'],
        common_per_method_CPU_wall_budget_s=45.,common_node_limit=20000,
        original_heuristic_unchanged=False,all_methods_share_same_existing_map_guidance=True,
        heuristic_revision=guidance.parameters,heuristic_field_NOT_an_edge_collision_or_support_certificate=True,
        no_mode_or_curvature_subset_fixed_route_fallback_or_goal_change=True,
        same_task_map_model_modes_primitives_costs_speed_and_budgets_as_native_v2=True,
        original_H1_specific_offset_connector_bypassed_for_all_methods=True,
        private_function_bindings_only_no_global_module_patch=True,
        guidance_preparation_and_native_warmup_excluded_from_search_budgets=True,
        common_speed_cap_m_s=shared.SPEED,common_speed_levels_m_s=list(shared.SPEED_LEVELS),
        common_acceleration_m_s2=shared.DECELERATION,terminal_planning_speed_m_s=shared.TERMINAL_SPEED,
        rigid_body_parameter_bounds=old['rigid_body_parameter_bounds'],native_build=old['native_build'],
        corrected_goal_corner_live_queries_complete=manifest['corrected_goal_corner_live_queries_complete'],
        actual_motion_authorized=False,new_PhysX_trial_count=0,use_for_final_statistics=False,input_sha256=pins,
        analysis_source={write_corridor_guided_planning_probe.__name__:inspect.getsource(write_corridor_guided_planning_probe)})
    key=hashlib.sha256(json.dumps(protocol,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_geometry_candidates/guided_planning_probes'/key).resolve()
    target=output/'planning_probe.json';definition=output/'planning_probe_protocol.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (saved['protocol']!=protocol or json.loads(definition.read_bytes())!=protocol
                or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in saved['output_sha256'].items())):
            raise ValueError('Existing guided probe differs; no overwrite or automatic rerun')
        return target
    if definition.exists():raise ValueError('Guided search already reserved; inspect records, no automatic rerun')
    output.mkdir(parents=True,exist_ok=True);write_exclusive(definition,protocol);artifacts=[definition]
    grid_path=output/'planning_occupancy.npz'
    np.savez_compressed(grid_path,occupied=occupied,origin_xy_m=origin,resolution_m=resolution);artifacts.append(grid_path)
    scope=dict(shared.plan_method.__globals__)
    scope.update(HospitalEnvelopeChecker=native.NativeHospitalEnvelopeChecker,pose_guidance=guidance,
        counterphase_connector=lambda goal,checker:None)
    run=FunctionType(shared.plan_method.__code__,scope,shared.plan_method.__name__,shared.plan_method.__defaults__,shared.plan_method.__closure__)
    model=rigid.MassNormalizedRigidBodyEdgeModel(rigid.RigidBodyParameterBounds(**old['rigid_body_parameter_bounds']))
    pool={};records=[]
    for method in shared.METHODS:
        print('[H5 GUIDED PLAN] '+method+'; full original action space, shared map guidance, 45s/20000 nodes.',flush=True)
        row=run(method,manifest,occupied,origin,resolution,model,45.,20000,pool)
        row=json.loads(json.dumps(row,allow_nan=False))
        row.update(scenario='H5_obstacle_bypass_return',common_geometry=str(map_path),
            planning_role='OFFLINE_COMMON_GUIDANCE_DEVELOPMENT_NOT_PHYSICAL_OR_FINAL_COMPARISON',
            original_heuristic_unchanged=False,all_methods_share_same_new_guidance=True,
            planning_occupancy_digest=hashlib.sha256(occupied.tobytes()).hexdigest(),
            start_world_xyyaw=manifest['start_xyyaw'],goal_local_xyyaw=manifest['local_goal_xyyaw'],
            common_max_command_speed_m_s=shared.SPEED,full_envelope_clearance_m=.010,
            rigid_body_parameter_bounds=old['rigid_body_parameter_bounds'],
            new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
        path=output/(method+'.json');write_exclusive(path,row);artifacts.append(path);audit=None
        if row['plan_success']:
            audit=independent.audit(path)
            audit.update(scenario='H5_obstacle_bypass_return',use_for_final_statistics=False,
                actual_motion_authorized=False,new_PhysX_trial_count=0)
            audit_path=output/(method+'_independent_audit.json');write_exclusive(audit_path,audit);artifacts.append(audit_path)
            if audit['verdict']=='PASS' and method!='P_dynamics_full_envelope':pool[method]=row['selected_actions']
        record={key:row.get(key) for key in ('method','plan_success','status','planning_time_s','action_count','path_length_m',
            'mode_switch_count','predicted_terminal_position_error_m','predicted_terminal_heading_error_deg')}
        record['independent_offline_audit']=audit['verdict'] if audit else 'NOT_RUN_NO_SUCCESSFUL_PLAN'
        records.append(record)
        print('[H5 GUIDED PLAN] '+method+': '+row['status']+'; %.2fs; audit='%row['planning_time_s']+record['independent_offline_audit'],flush=True)
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Common guided planning inputs changed')
    report=dict(version=protocol['version'],status='ALL_FOUR_COMMON_GUIDANCE_DEVELOPMENT_PROBES_RECORDED_NOT_PHYSICAL_VALIDATION',
        protocol=protocol,methods=records,guidance_preparation_s_excluded_from_method_budgets=preparation_s,
        previous_timeout_node_limit_and_success_records_preserved=True,
        no_global_optimality_infeasibility_or_universal_P_superiority_claim=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    write_exclusive(target,report)
    print('Saved all-four map-guided H5 OFFLINE probe:',target,flush=True)
    return target


def load_corrected_corridor_goal_corner_receipt(path):
    """Verify the actual five GETs; never convert samples into static admission."""
    import inspect
    path=Path(path).resolve();raw=json.loads(path.read_bytes())
    current_path=Path(raw['original_current_map']);current,review,task,design=load_current_corridor_map_receipt(current_path)
    geometry=corrected_corridor_endpoint_geometry(task,review)
    specs=[s for s in geometry['corrected_endpoint_grid_control_query_specifications']
        if s['id'].startswith('goal_corner_') or s['id']=='original_cart_control']
    pins={**current['input_sha256'],str(current_path):hashlib.sha256(current_path.read_bytes()).hexdigest()}
    if (raw.get('version')!='H5_missing_four_corrected_goal_corner_cached_queries_and_original_cart_control_v1'
            or raw.get('specifications')!=specs or raw.get('input_sha256')!=pins
            or raw.get('corrected_goal_xyyaw')!=geometry['goal_xyyaw']
            or raw.get('retained_result')!=current['retained_result']
            or raw.get('original_H4_verdict')!=current['original_H4_verdict']
            or raw.get('actual_timeline_before_and_after')!=current['actual_timeline_before_and_after']
            or raw.get('all_4997_protected_environment_rows')!=current['all_4997_protected_environment_rows']
            or raw.get('original_support_colliders')!=current['original_support_colliders']
            or raw.get('original_top_height_m')!=current['original_top_height_m']
            or raw.get('unchanged_original_query_limits')!=current['unchanged_original_query_limits']
            or raw.get('analysis_source')!={f.__name__:inspect.getsource(f) for f in
                (load_current_corridor_map_receipt,corrected_corridor_endpoint_geometry,review_corrected_corridor_goal_corners)}
            or any(raw.get(k) is not True for k in ('body_12_joints_13_links_gains_targets_exactly_retained',
                'Root_Session_exactly_unchanged','original_32_rays_and_inventory_preserved_NOT_rerun_or_relabelled',
                'no_Play_Stop_Pause_reset_reopen_save_cook_camera_renderer_or_physics_edit'))
            or raw.get('H5_static_or_drive_admission') is not False or raw.get('actual_motion_authorized') is not False
            or raw.get('use_for_final_statistics') is not False or raw.get('new_PhysX_trial_count')!=0):
        raise ValueError('Exact actual corrected-corner GET/state/source/limit transaction required')
    actor=json.loads(Path(raw['retained_result']).read_bytes())['physics_preparation_snapshot']['new_robot_root']
    summary=summarize_corridor_cached_queries(specs,raw['raw_query_rows'],current['original_support_colliders'],
        design['top_height_m'],current['unchanged_original_query_limits'],task['selected_original_cart'],actor)
    checked=(summary['query_error_count']==0 and summary['original_cart_control_hit']
        and all(d['matching_original_top_upward_support_hit_indices'] and not d['every_non_support_hit_indices']
            for d in summary['diagnostics'][:4]))
    if (raw.get('summary')!=summary or not checked
            or raw.get('all_four_corrected_corner_support_samples_without_other_hits') is not True
            or raw.get('status')!='CORRECTED_GOAL_CORNER_SAMPLES_VERIFIED_NOT_DRIVE_ADMISSION'):
        raise ValueError('Four actually measured clear support corners AND original cart control required; no waiver')
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Corrected corner source changed: '+name)
    return raw,geometry,current,design


def corridor_executor_plan_candidates(probe,source_rows,manifest,design,occupied,origin,resolution):
    """Bind all four unchanged native H5 references; screen actual line segments.

    True full-footprint support and original finite-cell SAT are sampled at
    5mm/0.25deg, not just native edge endpoints. No task, control, body model,
    caster seed, safety threshold or source trajectory is fitted or changed.
    """
    import copy
    from hospital_task_path_execution import task_execution_binding
    import hospital_h4_cohort_preparation_v4 as support
    from hospital_geometry_tools import pose_collision
    if (probe.get('version')!='H5_ALL_FOUR_EXISTING_MAP_GUIDANCE_NATIVE_EXACT_SAT_OFFLINE_v3'
            or probe.get('status')!='ALL_FOUR_COMMON_GUIDANCE_DEVELOPMENT_PROBES_RECORDED_NOT_PHYSICAL_VALIDATION'):
        raise ValueError('Explicit completed common-guidance native H5 development version required')
    initial=task_execution_binding('H5_obstacle_bypass_return')
    if (tuple(row['method'] for row in source_rows)!=initial.METHODS
            or any(not row['plan_success'] for row in source_rows)
            or any(row['independent_offline_audit']!='PASS' for row in probe['methods'])):
        raise ValueError('All FOUR exact complete and independently audited native plans required')
    if any(not row['selected_actions'] or any(type(a.get('target_speed_m_s')) not in (int,float)
            or not math.isfinite(a['target_speed_m_s']) or not 0.<a['target_speed_m_s']<=.05
            for a in row['selected_actions']) for row in source_rows):
        raise ValueError('Every original action needs a finite positive speed under the common cap')
    nominal={row['method']:sum(.050/a['target_speed_m_s'] for a in row['selected_actions']) for row in source_rows}
    required=max(nominal.values())+2.+initial.CONTROLLER_CONFIG['terminal_timeout_s']+5.
    duration=60.*math.ceil(required/60.)
    execution=task_execution_binding('H5_obstacle_bypass_return',simulation_budget_s=duration)
    resource=dict(version='H5_ALL_FOUR_NOMINAL_PLUS_COMMON_ALLOWANCES_PROSPECTIVE_RESOURCE_v1',
        methods=list(execution.METHODS),nominal_action_schedule_s=nominal,initial_pre_hold_s=2.,
        common_terminal_allocation_s=execution.CONTROLLER_CONFIG['terminal_timeout_s'],fixed_terminal_hold_s=5.,
        largest_nominal_plus_common_allocations_s=required,round_up_to_whole_minute_s=60.,
        maximum_simulation_duration_s=duration,maximum_wall_time_s=900.,
        common_all_four_NOT_P_specific=True,physical_completion_guarantee=False,actual_motion_authorized=False)
    covers=support.support_covers(design);plans=[];geometry=[]
    for source in source_rows:
        plan=copy.deepcopy(source)
        plan.update(goal_world_xyyaw=manifest['goal_xyyaw'],start_world_xyyaw=manifest['start_xyyaw'],
            goal_local_xyyaw=manifest['local_goal_xyyaw'],common_max_acceleration_m_s2=.020,
            footprint=dict(execution.FOOTPRINT),full_envelope_clearance_m=.010,
            native_MODEL_success_is_NOT_actual_support_or_drive_admission=True,
            shared_executor_version=execution.EXECUTION_VERSION,shared_controller_config=dict(execution.CONTROLLER_CONFIG),
            shared_command_limits=dict(execution.COMMAND_LIMITS),shared_guards=dict(execution.GUARDS),
            shared_acceptance=dict(execution.ACCEPTANCE),
            source_authored_caster_seed_policy='unchanged source-authored seed; no assigned/success-selected caster angle',
            full_state_response_model_independently_validated=False,
            original_spatial_actions_and_states_unchanged=True,actual_motion_authorized=False,use_for_final_statistics=False)
        execution.validate_plan(plan);reference=execution.PathReference(plan)
        count=collisions=0;maximum_missing=0.
        for a,b in zip(reference.poses,reference.poses[1:]):
            angle=execution.wrap(b[2]-a[2]);steps=max(1,math.ceil(math.dist(a[:2],b[:2])/.005),math.ceil(abs(angle)/math.radians(.25)))
            for index in range(steps+1):
                fraction=index/steps
                pose=(a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1]),a[2]+fraction*angle)
                collisions+=bool(pose_collision(occupied,pose,origin,resolution,clearance=.010))
                maximum_missing=max(maximum_missing,float(support.support_area_missing(pose,covers)));count+=1
        def pose_description(pose):
            error=execution.relative_pose(plan['goal_world_xyyaw'],pose)
            return dict(world_xyyaw=list(pose),goal_position_error_m=math.hypot(error[0],error[1]),
                goal_absolute_heading_error_deg=abs(math.degrees(error[2])),
                inside_original_stationary_pose_band=math.hypot(error[0],error[1])<=.030 and abs(math.degrees(error[2]))<=2.)
        _,handoff=reference.at(max(0.,reference.arc[-1]-.025))
        screening=dict(method=plan['method'],reference_center_arc_m=reference.arc[-1],samples=count,
            colliding_interpolated_reference_footprints=collisions,maximum_uncovered_support_area_m2=maximum_missing,
            checks=dict(all_sampled_reference_footprints_clear=collisions==0,all_full_footprint_authored_support_covered=maximum_missing<=1e-9),
            maximum_translation_spacing_m=.005,maximum_heading_spacing_deg=.25,
            continuous_collision_or_cooked_contact_certificate=False,
            planned_endpoint=pose_description(reference.poses[-1]),ideal_25mm_remaining_handoff=pose_description(handoff),
            pose_band_diagnostic_NOT_a_velocity_stop_hold_or_model_validation=True)
        plan['sampled_execution_reference_geometry']=screening
        plan['nominal_longitudinal_schedule_duration_s']=nominal[plan['method']]
        if plan['selected_actions']!=source['selected_actions'] or plan['selected_states']!=source['selected_states']:
            raise ValueError('Binding must not smooth, translate, refit or replace any native trajectory')
        plans.append(plan);geometry.append(screening)
    return plans,resource,geometry


def write_corridor_execution_bundle(probe_path,corners_path,output=None):
    """Task/geometry binding only; prospective MODEL and live trials separate."""
    import inspect
    import numpy as np
    import hospital_task_path_execution as binding
    import hospital_h4_path_execution as control
    import hospital_h4_cohort_preparation_v4 as support
    from hospital_runtime_evidence_io import write_exclusive
    probe_path=Path(probe_path).resolve();corners_path=Path(corners_path).resolve()
    if probe_path.name!='planning_probe.json' or not probe_path.is_file():raise ValueError('Completed guided planning probe required')
    probe=json.loads(probe_path.read_bytes());protocol=probe['protocol']
    if (probe.get('version')!='H5_ALL_FOUR_EXISTING_MAP_GUIDANCE_NATIVE_EXACT_SAT_OFFLINE_v3'
            or not (probe_path.parent/'planning_probe_protocol.json').is_file()
            or write_corridor_guided_planning_probe(protocol['previous_native_probe'],probe_path.parent)!=probe_path):
        raise ValueError('Exact original guided model/map/source version required; no automatic search')
    corners,corrected,current,design=load_corrected_corridor_goal_corner_receipt(corners_path)
    map_path=Path(protocol['common_geometry']);manifest=json.loads(map_path.read_bytes())
    if (manifest['current_live_map_receipt']!=corners['original_current_map']
            or manifest['geometry']!=corrected):raise ValueError('Own same-map corrected endpoint samples required')
    rows=[json.loads((probe_path.parent/(name+'.json')).read_bytes()) for name in control.METHODS]
    with np.load(protocol['common_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    values,resource,screening=corridor_executor_plan_candidates(probe,rows,manifest,design,occupied,origin,resolution)
    source_paths=[Path(binding.__file__),*[_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts', name) for name in control.COMMON_SOURCE_FILES]]
    pins={}
    for collection in (protocol['input_sha256'],probe['output_sha256'],corners['input_sha256'],
            {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (probe_path,corners_path,*source_paths)}):
        for name,sha in collection.items():
            if name in pins and pins[name]!=sha:raise ValueError('Conflicting frozen source pins')
            pins[name]=sha
    definition=dict(version='H5_ALL_FOUR_NATIVE_REFERENCE_COMMON_V3_TASK_BINDING_BUNDLE_v1',
        guided_native_probe=str(probe_path),corrected_goal_corner_receipt=str(corners_path),
        common_resource_proposal=resource,input_sha256=pins,
        analysis_source={f.__name__:inspect.getsource(f) for f in (load_corrected_corridor_goal_corner_receipt,
            corridor_executor_plan_candidates,write_corridor_execution_bundle)})
    key=hashlib.sha256(json.dumps(definition,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_execution_bundle'/key).resolve()
    target=output/'execution_bundle.json';geometry_path=output/'task_geometry.json';grid_path=output/'planning_occupancy.npz'
    preview=output/'four_method_references.png';paths=[output/(row['method']+'.json') for row in values]
    geometry=dict(scenario='H5_obstacle_bypass_return',start_xyyaw=manifest['start_xyyaw'],goal_xyyaw=manifest['goal_xyyaw'],
        local_goal_xyyaw=manifest['local_goal_xyyaw'],raster=manifest['raster'],
        candidate_occupancy_digest=hashlib.sha256(occupied.tobytes()).hexdigest(),
        original_current_map=manifest['current_live_map_receipt'],original_common_map=str(map_path),
        original_cart=corrected['selected_original_cart'],original_cart_union_AABB=corrected['original_cart_union_AABB'],
        corrected_goal_corner_receipt=str(corners_path),corrected_goal_corner_samples_verified=True,
        actual_H5_static_or_drive_admission=False,full_state_model_independently_validated=False,
        source_authored_floor_support_NOT_a_cooked_continuous_certificate=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    geometry_sha=hashlib.sha256((json.dumps(geometry,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
    generation=sum(row['planning_time_s'] for row in probe['methods'][:-1]);common=probe['guidance_preparation_s_excluded_from_method_budgets']
    accounting=[]
    for plan in values:
        plan.update(geometry_manifest=str(geometry_path),geometry_manifest_sha256=geometry_sha,
            selected_native_source=str(probe_path.parent/(plan['method']+'.json')),
            generator_selector_version=probe['version'],input_sha256=pins)
        own=plan['planning_time_s'];additional=generation if plan['method']=='P_dynamics_full_envelope' else 0.
        accounting.append(dict(method=plan['method'],own_recorded_search_or_refinement_stage_s=own,
            additional_baseline_candidate_generation_s=additional,common_map_guidance_preparation_s=common,
            measured_generation_plus_stage_s=own+additional,with_common_guidance_preparation_s=own+additional+common))
    report=dict(**definition,status='FOUR_TASK_BOUND_REFERENCES_NOMINAL_MODEL_SCREEN_PENDING' if all(all(s['checks'].values()) for s in screening)
        else 'HOLD_EXECUTION_REFERENCE_GEOMETRY_NO_DRIVE_ADMISSION',
        plans=list(map(str,paths)),geometry_manifest=str(geometry_path),planning_occupancy=str(grid_path),preview=str(preview),
        original_v3_tracking_terminal_hold_controls_and_physical_safety_bands_unchanged=True,
        all_native_actions_and_states_preserved=True,sampled_reference_geometry=screening,
        planning_computation_accounting=accounting,common_45s_was_stage_limit_NOT_equal_end_to_end_planning_budget=True,
        planning_resource_accounting_NOT_final_matched_evaluation=True,
        original_H4_and_H3_results_and_current_H4_boundary_preserved=True,
        proposed_common_PhysX_resource_is_NOT_authorized_by_this_bundle=True,
        MODEL_case_caster_contact_and_drive_validation_still_required=True,
        current_H3_whole_MODEL_2_of_8_unchanged=True,independent_fresh_whole_H3_repeats=0,
        current_live_robot_or_timeline_accessed=False,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False)
    # Freeze strict JSON types before any exclusive file creation.
    geometry=json.loads(json.dumps(geometry,allow_nan=False));values=json.loads(json.dumps(values,allow_nan=False))
    report=json.loads(json.dumps(report,allow_nan=False))
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Execution-bundle source changed')
    if target.exists():
        saved=json.loads(target.read_bytes())
        if ({k:v for k,v in saved.items() if k!='output_sha256'}!=report
                or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in saved['output_sha256'].items())):
            raise ValueError('Existing H5 execution bundle differs; no overwrite')
        return target
    if output.exists() and any(output.iterdir()):raise ValueError('Unfinalized H5 bundle directory retained; no overwrite')
    output.mkdir(parents=True,exist_ok=True);write_exclusive(geometry_path,geometry)
    if hashlib.sha256(geometry_path.read_bytes()).hexdigest()!=geometry_sha:raise ValueError('Strict task geometry serialization differs')
    with grid_path.open('xb') as stream:np.savez_compressed(stream,occupied=occupied,origin_xy_m=origin,resolution_m=resolution)
    for path,plan in zip(paths,values):write_exclusive(path,plan)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from hospital_geometry_tools import footprint_polygon
    fig,ax=plt.subplots(figsize=(7,9),layout='constrained')
    ax.imshow(occupied,origin='lower',cmap='Greys',vmin=0,vmax=1,
        extent=[origin[0],origin[0]+occupied.shape[1]*resolution,origin[1],origin[1]+occupied.shape[0]*resolution])
    for plan,color,style in zip(values,('#168aad','#e88b18','#8840a3','#148747'),('-',':','--','-')):
        poses=np.asarray([plan['start_world_xyyaw']]+[s['world_xyyaw'] for s in plan['selected_states']])
        ax.plot(poses[:,0],poses[:,1],color=color,ls=style,lw=1.5,label=plan['method'])
    for pose,color in ((manifest['start_xyyaw'],'#148747'),(manifest['goal_xyyaw'],'#e88b18')):
        ax.add_patch(Polygon(footprint_polygon(pose),facecolor=color,edgecolor=color,alpha=.22))
    ax.set(aspect='equal',xlabel='Hospital world X (m)',ylabel='Hospital world Y (m)',
        title='H5 four native references / unchanged common v3 control\nOFFLINE binding only; no static, MODEL or drive admission')
    ax.legend(fontsize=8,loc='upper right')
    try:
        with preview.open('xb') as stream:fig.savefig(stream,format='png',dpi=160)
    finally:plt.close(fig)
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Bundle source changed while exporting')
    write_exclusive(target,dict(**report,output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (geometry_path,grid_path,preview,*paths)}))
    print('Saved H5 native-reference/common-controller OFFLINE bundle:',target,flush=True)
    print('geometry checks:',[(s['method'],s['checks']) for s in screening],flush=True)
    print('PROPOSED common simulation/wall limits:',resource['maximum_simulation_duration_s'],resource['maximum_wall_time_s'],'; no live authorization',flush=True)
    return target


def screen_corridor_execution_bundle(bundle_path,output=None):
    """Original v3 closed-loop software MODEL at both 10/20ms, all methods.

    Rebind only the task/schema and prospective common duration. Original
    bias/rate/Newton-Euler parameters remain unchanged. No actual caster,
    wheel-hold force or cooked contact is simulated or validated here.
    """
    import inspect,csv
    import numpy as np
    from types import FunctionType
    import hospital_task_path_execution as binding
    import screen_hospital_h4_common_executor as original
    from hospital_runtime_evidence_io import write_exclusive
    bundle_path=Path(bundle_path).resolve()
    if bundle_path.name!='execution_bundle.json' or not bundle_path.is_file():raise ValueError('Completed explicit H5 execution-schema bundle required')
    bundle=json.loads(bundle_path.read_bytes())
    if (bundle.get('version')!='H5_ALL_FOUR_NATIVE_REFERENCE_COMMON_V3_TASK_BINDING_BUNDLE_v1'
            or write_corridor_execution_bundle(bundle['guided_native_probe'],bundle['corrected_goal_corner_receipt'],bundle_path.parent)!=bundle_path
            or bundle.get('status')!='FOUR_TASK_BOUND_REFERENCES_NOMINAL_MODEL_SCREEN_PENDING'):
        raise ValueError('All four unchanged and clear/supported task-bound references required before MODEL screen')
    native=json.loads(Path(bundle['guided_native_probe']).read_bytes())
    bounds=native['protocol']['rigid_body_parameter_bounds']
    geometry=json.loads(Path(bundle['geometry_manifest']).read_bytes())
    _,_,_,design=load_corrected_corridor_goal_corner_receipt(bundle['corrected_goal_corner_receipt'])
    execution=binding.task_execution_binding('H5_obstacle_bypass_return',
        simulation_budget_s=bundle['common_resource_proposal']['maximum_simulation_duration_s'])
    scope=dict(original.case.__globals__,execution=execution)
    run=FunctionType(original.case.__code__,scope,original.case.__name__,original.case.__defaults__,original.case.__closure__)
    files=(bundle_path,Path(original.__file__))
    pins={**bundle['input_sha256'],**bundle['output_sha256'],
        **{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    definition=dict(version='H5_ORIGINAL_COMMON_V3_CLOSED_LOOP_NOMINAL_MODEL_DIAGNOSTIC_v1',
        execution_bundle=str(bundle_path),common_resource_proposal=bundle['common_resource_proposal'],
        integration_steps_s=list(original.TIMESTEPS_S),rigid_body_parameter_bounds=bounds,
        original_MODEL_case_control_and_propagation_code_unchanged=True,
        only_task_schema_and_declared_duration_are_privately_bound=True,
        original_H4_source_or_globals_modified=False,actual_motion_authorized=False,
        use_for_final_statistics=False,new_PhysX_trial_count=0,input_sha256=pins,
        analysis_source={screen_corridor_execution_bundle.__name__:inspect.getsource(screen_corridor_execution_bundle)})
    key=hashlib.sha256(json.dumps(definition,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]
    output=(Path(output) if output is not None else GATE/'hospital_h5_common_executor_screen'/key).resolve()
    target=output/'common_executor_model_screen.json';reservation=output/'model_definition.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if (any(saved.get(k)!=v for k,v in definition.items())
                or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in saved['output_sha256'].items())):
            raise ValueError('Existing H5 MODEL screen differs; no overwrite or automatic rerun')
        return target
    if output.exists() and any(output.iterdir()):raise ValueError('Unfinalized H5 MODEL screen retained; no automatic rerun')
    output.mkdir(parents=True,exist_ok=True);write_exclusive(reservation,definition);artifacts=[reservation]
    cases=[]
    with np.load(bundle['planning_occupancy']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    for plan_path in map(Path,bundle['plans']):
        plan=json.loads(plan_path.read_bytes());execution.validate_plan(plan)
        for dt in original.TIMESTEPS_S:
            print('[H5 NOMINAL MODEL] '+plan['method']+'; dt=%.3fs; original controls/model, no Isaac.'%dt,flush=True)
            report,rows=run(plan,bounds,dt,occupied,origin,resolution,design)
            report.update(scenario=geometry['scenario'],plan_path=str(plan_path),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                actual_configured_MODEL_simulation_duration_limit_s=execution.GUARDS['maximum_simulation_duration_s'],
                inherited_error_message_may_name_the_original_180s_limit_NOT_this_binding=bool(report['error'] and '180s' in report['error']),
                actual_motion_authorized=False,new_PhysX_trial_count=0)
            trace=output/(plan['method']+'_%dms.csv'%round(1000*dt))
            with trace.open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            report.update(model_trace=str(trace),model_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
            cases.append(json.loads(json.dumps(report,allow_nan=False)));artifacts.append(trace)
            print('[H5 NOMINAL MODEL] '+plan['method']+' '+report['model_status']+'; goal=%.2fmm/%.3fdeg; error='%(
                1000*report['final_position_error_m'],report['final_heading_error_deg'])+str(report['error']),flush=True)
    for name,sha in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Original H5 MODEL screen source changed')
    report=dict(**definition,status='NOMINAL_SOFTWARE_MODEL_SCREEN_RECORDED_NOT_PHYSICAL_OR_MODEL_VALIDATION',
        cases=cases,case_count=len(cases),nominal_MODEL_PASS_count=sum(row['model_status']=='MODEL_SCREEN_PASS' for row in cases),
        physical_wheel_hold_force_caster_history_stiction_and_cooked_contact_NOT_simulated=True,
        two_software_timesteps_are_NOT_independent_trials_or_robustness=True,
        prior_whole_H3_MODEL_PASS_count=2,prior_whole_H3_MODEL_case_count=8,
        independent_fresh_whole_H3_repeats=0,no_live_scene_body_joint_or_timeline_access=True,
        no_fit_model_control_physics_or_acceptance_change=True,
        output_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    write_exclusive(target,report)
    print('Saved H5 original-controller NOMINAL MODEL diagnostic:',target,flush=True)
    print('MODEL passes:',report['nominal_MODEL_PASS_count'],'/',len(cases),'; no physical/static/drive admission.',flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=GATE / "hospital_asset_geometry_audit.json")
    parser.add_argument("--output", type=Path, default=CONFIG / "hospital_environment_measurements.yaml")
    parser.add_argument('--cached-obstacle-inventory',type=Path,help='Complete saved all-stage collision inventory; future-H5 authored assets ONLY, no YAML/scene changes')
    parser.add_argument('--inventory-output',type=Path,help='Fresh derived-output directory for cached original obstacles')
    parser.add_argument('--propose-corridor-bypass',type=Path,help='Actual H5 GET-only asset report; task/figure proposal only, not a route or live change')
    parser.add_argument('--task-output',type=Path,help='Optional fresh geometry task proposal directory')
    parser.add_argument('--current-corridor-map',type=Path,help='Actual complete current-map GET receipt; offline shared grid/task candidate only')
    parser.add_argument('--map-output',type=Path,help='Optional fresh common-grid directory; no overwrite')
    parser.add_argument('--corridor-planning-probe',type=Path,help='Exact current-data common map; ONE bounded all-four offline planning probe, no trial')
    parser.add_argument('--planning-output',type=Path,help='Optional fresh offline-probe directory; no automatic rerun')
    parser.add_argument('--corridor-native-planning-probe',type=Path,help='Completed original probe; same bounded all-four development search with existing native exact SAT')
    parser.add_argument('--corridor-guided-planning-probe',type=Path,help='Completed native probe; common existing map-guidance revision for all four methods, unchanged full action spaces')
    parser.add_argument('--corridor-execution-bundle',type=Path,nargs=2,metavar=('GUIDED_PROBE','ACTUAL_GOAL_CORNERS'),help='Bind FOUR unchanged H5 references to common v3 schema; no live or MODEL admission')
    parser.add_argument('--bundle-output',type=Path,help='Optional fresh H5 execution-schema bundle directory')
    parser.add_argument('--corridor-executor-model-screen',type=Path,help='Explicit complete H5 bundle; unchanged shared v3 nominal MODEL at 10/20ms, no actual trial')
    parser.add_argument('--model-output',type=Path,help='Optional fresh nominal MODEL-screen directory')
    args = parser.parse_args()
    if sum(bool(x) for x in (args.cached_obstacle_inventory,args.propose_corridor_bypass,args.current_corridor_map,args.corridor_planning_probe,args.corridor_native_planning_probe,args.corridor_guided_planning_probe,args.corridor_execution_bundle,args.corridor_executor_model_screen))>1:parser.error('Choose one offline asset action')
    if args.inventory_output and not args.cached_obstacle_inventory:parser.error('--inventory-output requires --cached-obstacle-inventory')
    if args.task_output and not args.propose_corridor_bypass:parser.error('--task-output requires --propose-corridor-bypass')
    if args.map_output and not args.current_corridor_map:parser.error('--map-output requires --current-corridor-map')
    if args.planning_output and not (args.corridor_planning_probe or args.corridor_native_planning_probe or args.corridor_guided_planning_probe):parser.error('--planning-output requires an offline planning action')
    if args.bundle_output and not args.corridor_execution_bundle:parser.error('--bundle-output requires --corridor-execution-bundle')
    if args.model_output and not args.corridor_executor_model_screen:parser.error('--model-output requires --corridor-executor-model-screen')
    if args.corridor_executor_model_screen:
        screen_corridor_execution_bundle(args.corridor_executor_model_screen,args.model_output)
        return
    if args.corridor_execution_bundle:
        write_corridor_execution_bundle(*args.corridor_execution_bundle,args.bundle_output)
        return
    if args.corridor_guided_planning_probe:
        write_corridor_guided_planning_probe(args.corridor_guided_planning_probe,args.planning_output)
        return
    if args.corridor_native_planning_probe:
        write_corridor_native_planning_probe(args.corridor_native_planning_probe,args.planning_output)
        return
    if args.corridor_planning_probe:
        write_corridor_planning_probe(args.corridor_planning_probe,args.planning_output)
        return
    if args.current_corridor_map:
        write_corridor_common_grid(args.current_corridor_map,args.map_output)
        return
    if args.propose_corridor_bypass:
        write_corridor_bypass_task_proposal(args.propose_corridor_bypass,args.task_output)
        return
    if args.cached_obstacle_inventory:
        write_cached_obstacle_inventory(args.cached_obstacle_inventory,args.inventory_output)
        return
    raw = args.audit.read_bytes()
    audit = json.loads(raw)
    bed, bounds_match = bed_evidence(audit)
    template = CONFIG / "hospital_environment_measurement_template.yaml"
    config = yaml.safe_load((args.output if args.output.exists() else template).read_text(encoding="utf-8"))
    if config["freeze"]["geometry_frozen"]:
        raise ValueError("Geometry is frozen; create a new measurement revision before updating it")
    config.update({
        "status": "asset_evidence_partial",
        "measurement_date": audit["created_utc"].split("T")[0],
        "measured_by": "automated_USD_asset_bounds_export",
        "location_or_asset_source": audit["stage_root_layer"],
    })
    bound = bed["world_aligned_visible_bound"]
    config["bed"].update({
        "identifier": bed["path"],
        "source_type": "audited_USD",
        "source_reference": str(args.audit.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "overall_length_m": max(bound["size_m"][:2]),
        "overall_width_m": min(bound["size_m"][:2]),
        "collision_height_m": bed["collision"]["bound"]["size_m"][2],
        "world_aligned_bound_m": bound,
        "aggregate_world_bounds_match": bounds_match,
        "dimension_basis": "world_AABB_for_current_authored_asset_pose",
        "dimension_limitation": "Recompute oriented dimensions if the bed pose changes; aggregate bounds do not verify cooked contact geometry.",
        "measurement_uncertainty_m": None,
        "uncertainty_status": "USD_numeric_precision_not_physical_measurement_uncertainty",
    })
    config["asset_and_collision_audit"].update({
        "hospital_stage_usd": audit["stage_root_layer"],
        "bed_visual_prim": bed["path"],
        "bed_collision_prim": bed["collision"]["collision_prim_paths"],
        "visual_collision_bounds_match": bounds_match,
        "cooked_collision_geometry_verified": False,
        "complete_local_inventory_available": audit.get("candidate_export_complete", False),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("=== IJCAS HOSPITAL ASSET MEASUREMENT RECORD ===")
    print("bed length/width/height m:", *(round(config["bed"][key], 6) for key in (
        "overall_length_m", "overall_width_m", "collision_height_m")))
    print("aggregate bounds match:", bounds_match)
    print("cooked collision verified: False")
    print("geometry frozen: False")
    print("saved:", args.output)


if __name__ == "__main__":
    main()
