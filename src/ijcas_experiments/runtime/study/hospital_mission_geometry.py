"""Offline integrated-mission map coverage from an existing ALL-stage receipt.

Never stitch old task PASSes, move their endpoints or turn the H5 ROI into a
global free-space claim. Reclassify every recorded enabled collider against
an explicit new analysis ROI; retain unknown/unsupported space as blocked.
Importing this module never edits a scene or starts a simulation. Offline
reviews remain separate from the explicit, user-approved prepare-only entry;
that entry can archive/Stop once and author the reviewed Session condition,
but never admits Play, static testing, a route or a retry.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from hospital_geometry_tools import polygon_cells,pose_collision,footprint_polygon
from hospital_h4_support_design import area,ccw,uncovered
import hospital_runtime_evidence_io as evidence


ROOT=_ijcas_workspace()
GATE=_ijcas_path(ROOT, 'studies/IJCAS/results/hospital_paper_gate')
VERSION='H6_ALL_STAGE_AUTHORED_COVERAGE_AND_EXISTING_TASK_CONNECTION_REVIEW_v1'
RESOLUTION_M=.010
SLAB_M=(.006,.450)
MAX_CELLS=4_000_000


def vector(value,length,name):
    result=np.asarray(value,float)
    if result.shape!=(length,) or not np.isfinite(result).all():raise ValueError('Finite '+name+' required')
    return result


def authored_slab_transverse_gap(meshes, *, center_y_m, lower_z_m=.006, upper_z_m=.450):
    """Conservative clear Y band of named structural triangles over ALL X.

    This projects triangle surfaces clipped to the entire declared height
    slab, not one favorable X/height ray. An empty band is only a property of
    these structural meshes: doors, furniture, full-body traversal and cooked
    PhysX support are NOT certified. No source mesh or raster is edited.
    """
    from hospital_geometry_tools import clip_height
    if (not meshes or not all(math.isfinite(v) for v in (center_y_m,lower_z_m,upper_z_m))
            or not lower_z_m<upper_z_m):
        raise ValueError('Named nonempty structural meshes and finite slab/center required')
    intervals=[];paths=set()
    for mesh in meshes:
        path=mesh.get('path');points=np.asarray(mesh['world_points_m'],float)
        counts=np.asarray(mesh['face_vertex_counts']);indices=np.asarray(mesh['face_vertex_indices'])
        if (not isinstance(path,str) or not path or path in paths or points.ndim!=2
                or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all()
                or counts.ndim!=1 or indices.ndim!=1 or indices.dtype.kind not in 'iu'
                or not len(counts) or not np.all(counts==3) or len(indices)!=3*len(counts)
                or np.any(indices<0) or np.any(indices>=len(points))):
            raise ValueError('Unique complete finite named triangle meshes required')
        paths.add(path)
        for triangle in points[indices.reshape(-1,3)]:
            polygon=clip_height(triangle,lower_z_m,upper_z_m)
            if polygon:
                values=np.asarray(polygon)[:,1]
                intervals.append([float(values.min()),float(values.max())])
    merged=[]
    for left,right in sorted(intervals):
        if merged and left<=merged[-1][1]+1e-12:merged[-1][1]=max(merged[-1][1],right)
        else:merged.append([left,right])
    gaps=[[a[1],b[0]] for a,b in zip(merged,merged[1:]) if a[1]<center_y_m<b[0]]
    if len(gaps)!=1:raise ValueError('Exactly one finite central structural aperture required')
    return dict(clear_Y_interval_m=gaps[0],width_m=gaps[0][1]-gaps[0][0],
        projected_occupied_Y_intervals_m=merged,source_paths=sorted(paths),height_slab_m=[lower_z_m,upper_z_m],
        all_X_and_whole_slab_projection=True,other_objects_or_body_cooked_traversal_NOT_certified=True)


def south_widening_vertices(points, *, original_gap_y_m, lower_anchor_y_m, geometric_width_m):
    """Explicit NEW architectural vertex deformation, never a map-cell erase.

    Keep X/Z, the upper jamb and vertices outside the anchored wall span.
    Move the lower aperture boundary south; linearly distribute the change
    over the lower panel and aperture vertices. This defines new vertex
    geometry, not an exact continuum deformation of triangle interiors.
    Caller must retain topology, check face orientation and rebuild every
    collider contribution. NOT original-source rigid geometry or live setup.
    """
    original=vector(original_gap_y_m,2,'original structural gap');values=np.asarray(points,float)
    left,right=map(float,original);new_left=right-geometric_width_m
    if (values.ndim!=2 or values.shape[1]!=3 or not len(values) or not np.isfinite(values).all()
            or not all(math.isfinite(v) for v in (lower_anchor_y_m,geometric_width_m))
            or not lower_anchor_y_m<new_left<left<right):
        raise ValueError('Finite vertices and strictly wider positive anchored aperture required')
    result=values.copy();y=values[:,1]
    lower=(y>lower_anchor_y_m)&(y<left);opening=(y>=left)&(y<right)
    result[lower,1]=lower_anchor_y_m+(y[lower]-lower_anchor_y_m)*(new_left-lower_anchor_y_m)/(left-lower_anchor_y_m)
    result[opening,1]=new_left+(y[opening]-left)*(right-new_left)/(right-left)
    return result,dict(original_gap_Y_m=[left,right],new_gap_Y_m=[new_left,right],
        lower_anchor_y_m=lower_anchor_y_m,geometric_width_m=geometric_width_m,
        X_Z_and_upper_jamb_and_outside_anchor_unchanged=True,
        NEW_architectural_vertex_geometry_NOT_rigid_source_preservation=True,
        no_raster_or_source_USD_or_Session_edit=True)


def disjoint_support_rectangles(polygons):
    """Actual polygon-union subtraction; overlapping area cannot fill holes."""
    pieces=[]
    for polygon in polygons:
        points=np.asarray(polygon,float)
        if points.shape!=(4,2) or not np.isfinite(points).all():raise ValueError('Four actual finite support corners required')
        low,high=points.min(axis=0),points.max(axis=0)
        corners={(float(x),float(y)) for x,y in points}
        expected={(float(x),float(y)) for x in (low[0],high[0]) for y in (low[1],high[1])}
        if corners!=expected or np.any(high<=low):raise ValueError('Nondegenerate axis-aligned authored rectangles required')
        ordered=ccw([(float(x),float(y)) for x,y in points])
        if abs(area(ordered)-(high[0]-low[0])*(high[1]-low[1]))>1e-10:
            raise ValueError('Support corners must be in perimeter order, never a crossed polygon')
        pieces.extend(uncovered([ordered],pieces))
    if not pieces:raise ValueError('Actual nonempty authored support union required')
    result=[]
    for piece in pieces:
        p=np.asarray(piece,float);low,high=p.min(axis=0),p.max(axis=0)
        if abs(abs(area(piece))-(high[0]-low[0])*(high[1]-low[1]))>1e-10:
            raise ValueError('Support union fragment is not an exact rectangle')
        result.append((low,high))
    return result


def build_authored_grid(inventory,retained_actor,roi_xy,support_rectangles):
    """Use ALL current recorded flags/bounds, including the old outside_ROI."""
    roi=vector(roi_xy,4,'explicit analysis ROI');low=roi[[0,2]];high=roi[[1,3]]
    if np.any(high<=low) or not isinstance(retained_actor,str) or not retained_actor.startswith('/World/'):
        raise ValueError('Positive explicit ROI and exact historical retained actor required')
    origin=np.floor(low/RESOLUTION_M)*RESOLUTION_M
    shape_xy=np.ceil((high-origin)/RESOLUTION_M).astype(int)
    if np.any(shape_xy<=0) or int(shape_xy[0])*int(shape_xy[1])>MAX_CELLS:
        raise ValueError('Bounded complete analysis raster required; no silent coarsening')
    rows=inventory['collision_prims']
    if (inventory.get('authored_inventory_coverage_complete') is not True or inventory.get('errors')!=[]
            or inventory.get('active_unloaded_payload_paths')!=[] or not rows
            or inventory['collision_prim_count']!=len(rows) or len({r['path'] for r in rows})!=len(rows)):
        raise ValueError('Complete unique ALL-stage source inventory required')
    from collections import Counter
    if inventory['classification_counts']!=dict(Counter(row['classification'] for row in rows)):
        raise ValueError('Every original collider classification/count must be preserved')
    shape=tuple(map(int,shape_xy[::-1]));obstacles=np.zeros(shape,bool)
    included=[];excluded=[]
    for row in rows:
        path=row['path'];active=row['is_active'];enabled=row['collision_enabled_value'];classification=row['classification']
        if (not isinstance(path,str) or not path.startswith('/World/') or type(active) is not bool
                or type(enabled) is not bool):raise ValueError('Exact original collider flags/identity required')
        if not active:
            if classification!='inactive':raise ValueError('Inactive classification mismatch')
            excluded.append(dict(path=path,reason='original_inactive'));continue
        if not enabled:
            if classification!='disabled':raise ValueError('Disabled classification mismatch')
            excluded.append(dict(path=path,reason='original_disabled_no_historical_bound_invented'));continue
        if path==retained_actor or path.startswith(retained_actor+'/'):
            if classification!='robot_self_excluded':raise ValueError('Exact own-actor classification mismatch')
            excluded.append(dict(path=path,reason='exact_original_retained_robot_not_environment'));continue
        if classification not in ('outside_ROI','enabled_in_ROI'):
            raise ValueError('Every enabled environment collider needs a verified recorded bound')
        bound=row['world_aligned_bound'];lo=vector(bound['minimum_m'],3,'collider minimum');hi=vector(bound['maximum_m'],3,'collider maximum')
        if np.any(lo>hi):raise ValueError('Inverted collider bound')
        if 'size_m' in bound and not np.allclose(vector(bound['size_m'],3,'collider extent'),hi-lo,rtol=0.,atol=1e-10):
            raise ValueError('Original collider bound extent mismatch')
        if hi[2]<SLAB_M[0] or lo[2]>SLAB_M[1]:
            excluded.append(dict(path=path,reason='outside_UNCHANGED_robot_height_slab'));continue
        if np.any(hi[:2]<low) or np.any(lo[:2]>high):
            excluded.append(dict(path=path,reason='outside_NEW_analysis_ROI'));continue
        rr,cc=polygon_cells([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],origin,RESOLUTION_M,shape)
        obstacles[rr,cc]=True
        included.append(dict(path=path,original_classification=classification,
            world_aligned_bound=bound,occupied_cell_count=int(len(rr))))
    x0=origin[0]+np.arange(shape[1])*RESOLUTION_M;y0=origin[1]+np.arange(shape[0])*RESOLUTION_M
    x1=x0+RESOLUTION_M;y1=y0+RESOLUTION_M
    inside=(x0[None,:]>=low[0])&(x1[None,:]<=high[0])&(y0[:,None]>=low[1])&(y1[:,None]<=high[1])
    support_area=np.zeros(shape,float)
    partitions=disjoint_support_rectangles(support_rectangles)
    for lo,hi in partitions:
        dx=np.maximum(0.,np.minimum(x1,hi[0])-np.maximum(x0,lo[0]))
        dy=np.maximum(0.,np.minimum(y1,hi[1])-np.maximum(y0,lo[1]))
        support_area+=dy[:,None]*dx[None,:]
    supported=support_area>=RESOLUTION_M**2-1e-12
    # Unmeasured outside-ROI slivers and cells not fully covered by the
    # existing authored support union are NEVER silently declared free.
    blocked=obstacles|~supported|~inside
    report=dict(version=VERSION,source_collider_count=len(rows),analysis_roi_xy_m=roi.tolist(),
        raster=dict(origin_xy_m=origin.tolist(),resolution_m=RESOLUTION_M,shape_rows_columns=list(shape)),
        original_height_slab_m=list(SLAB_M),included_obstacles=included,excluded_colliders=excluded,
        old_outside_ROI_rows_now_included=sum(row['original_classification']=='outside_ROI' for row in included),
        actual_original_support_rectangles=support_rectangles,support_union_partition_count=len(partitions),
        fully_supported_cell_count=int(supported.sum()),blocked_cell_count=int(blocked.sum()),
        support_cell_area_arithmetic_tolerance_m2=1e-12,
        original_flags_bounds_ROI_and_support_NOT_modified=True,
        authored_AABB_coverage_NOT_cooked_shapes_or_continuous_PhysX_support=True,
        H6_start_goal_path_control_or_motion_admitted=False)
    return evidence.json_value(report),blocked,supported,obstacles


def task_anchor_review(anchors,grid,blocked,supports):
    """Controls are existing task poses, NOT an H6 mission waypoint sequence."""
    if not anchors or len({a['id'] for a in anchors})!=len(anchors):raise ValueError('Unique explicitly sourced pose controls required')
    origin=np.asarray(grid['raster']['origin_xy_m']);resolution=grid['raster']['resolution_m'];rows=[]
    components,count=label(~blocked,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    for item in anchors:
        pose=vector(item['world_xyyaw'],3,'original task anchor');polygon=footprint_polygon(pose).tolist()
        missing=float(sum(abs(area(piece)) for piece in uncovered([polygon],supports)))
        index=np.floor((pose[:2]-origin)/resolution).astype(int)
        inside=bool(0<=index[0]<blocked.shape[1] and 0<=index[1]<blocked.shape[0])
        component=int(components[index[1],index[0]]) if inside else 0
        rows.append(dict(**item,full_original_10mm_footprint_clear_in_source_raster=not pose_collision(blocked,pose,origin,resolution),
            unsupported_full_footprint_area_m2=missing,point_center_component_id=component,
            point_center_free_in_source_raster=component!=0,source_pose_NOT_changed=True))
    # All 45 unordered pose pairs, not just geometrically favorable links.
    connections=[]
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            dx=right['world_xyyaw'][0]-left['world_xyyaw'][0];dy=right['world_xyyaw'][1]-left['world_xyyaw'][1]
            d=right['world_xyyaw'][2]-left['world_xyyaw'][2]
            connections.append(dict(first=left['id'],second=right['id'],XY_gap_m=math.hypot(dx,dy),
                body_yaw_gap_deg=math.degrees(math.atan2(math.sin(d),math.cos(d))),
                same_POINT_center_component=left['point_center_component_id']!=0
                    and left['point_center_component_id']==right['point_center_component_id'],
                point_connectivity_is_NOT_a_full_body_steering_or_dynamics_route=True))
    return dict(anchors=rows,all_unordered_anchor_pairs=connections,point_center_component_count=int(count),
        all_source_controls_examined_not_success_selected=True,
        controls_NOT_a_frozen_H6_start_goal_or_path=True,
        point_disconnection_NOT_global_continuous_geometric_infeasibility=True,
        body_velocity_steering_caster_and_hold_states_NOT_injected=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def existing_task_controls(h5_protocol,author_manifest):
    """Read original fixed task coordinates with their saved provenance pins."""
    p=json.loads(Path(h5_protocol).read_bytes());author=json.loads(Path(author_manifest).read_bytes())
    digest=hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='definition_sha256'},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (p.get('scenario')!='H5_obstacle_bypass_return' or p.get('definition_sha256')!=digest
            or p.get('actual_motion_authorized') is not False or p.get('use_for_final_statistics') is not False):
        raise ValueError('Exact offline frozen H5 definition required; not an invented mission task')
    paths={
        'H1':GATE/'planning/hospital_h1_planning_r06/P_dynamics_full_envelope.json',
        'H2':GATE/'planning/hospital_h2_planning_r01/P_dynamics_full_envelope.json',
        'H3':GATE/'hospital_h3b_segmented_completion/audit_r01/frozen_replay_inputs.json',
        'H4':GATE/'hospital_h4_experiments/cohort_09_full_task_r01/protocol.json',
        'H5':Path(h5_protocol).resolve()}
    pins=dict(p['input_sha256']);anchors=[]
    for name,path in paths.items():
        expected=pins.get(str(path)) or author['input_sha256'].get(str(path))
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if (name!='H5' and expected!=actual):raise ValueError('Original task provenance missing/changed: '+name)
        if str(path) in pins and pins[str(path)]!=actual:raise ValueError('Conflicting task input pin: '+name)
        pins[str(path)]=actual;record=json.loads(path.read_bytes())
        keys=('start_world_xyyaw','goal_world_xyyaw') if name in ('H1','H2') else (
            ('original_start_xyyaw','original_goal_xyyaw') if name=='H3' else ('start_xyyaw','goal_xyyaw'))
        for endpoint,key in zip(('start','goal'),keys):
            pose=vector(record[key],3,'source task control')
            anchors.append(dict(id=name+'_'+endpoint,scenario=name,endpoint=endpoint,world_xyyaw=pose.tolist(),
                source=str(path),source_sha256=actual,source_field=key,
                role='EXISTING_TASK_COORDINATE_CONTROL_NOT_A_NEW_MISSION_WAYPOINT'))
    for path,value in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=value:raise ValueError('Original task evidence changed: '+path)
    pins[str(Path(author_manifest).resolve())]=hashlib.sha256(Path(author_manifest).read_bytes()).hexdigest()
    return anchors,pins


def doorway_condition_review(h3_manifest,h3_audit,inventory,*,door_root):
    """Compare recorded environment conditions, not an inferred live state."""
    threshold=h3_manifest['suppressed_threshold_root']
    doors=[asset for asset in h3_audit['local_assets'] if asset['path']==door_root]
    if len(doors)!=1:raise ValueError('Exact explicitly sourced original H3 door asset required')
    door=doors[0]
    if (h3_manifest['environment_condition']!='H3_flush_fully_open_continuous_floor'
            or h3_manifest['door_condition']!='fully_open_baseline_session_only'):
        raise ValueError('Exact previously declared H3 flush/open condition required')
    threshold_rows=[row for row in inventory['collision_prims'] if row['path']==threshold or row['path'].startswith(threshold+'/')]
    door_rows=[row for row in inventory['collision_prims'] if row['path']==door['path'] or row['path'].startswith(door['path']+'/')]
    active_threshold=[row['path'] for row in threshold_rows if row['is_active'] and row['collision_enabled_value'] is True]
    active_door=[row for row in door_rows if row['is_active'] and row['collision_enabled_value'] is True]
    if not threshold_rows or not active_door:raise ValueError('Actual recorded threshold and door identities required; no historical bounds invented')
    expected=door['collision']['bound']
    low=np.min([row['world_aligned_bound']['minimum_m'] for row in active_door],axis=0)
    high=np.max([row['world_aligned_bound']['maximum_m'] for row in active_door],axis=0)
    error=max(float(np.max(np.abs(low-vector(expected['minimum_m'],3,'original H3 door minimum')))),
        float(np.max(np.abs(high-vector(expected['maximum_m'],3,'original H3 door maximum')))))
    return dict(original_H3_environment_condition=h3_manifest['environment_condition'],
        original_H3_door_condition=h3_manifest['door_condition'],original_H3_suppressed_threshold_root=threshold,
        threshold_rows_in_newer_saved_inventory=threshold_rows,currently_recorded_enabled_threshold_paths=active_threshold,
        same_H3_threshold_suppression_recorded=not active_threshold,
        original_H3_fully_open_door_collision_bound=expected,newer_recorded_door_collision_union_AABB=dict(minimum_m=low.tolist(),maximum_m=high.tolist()),
        door_bound_comparison_is_collision_union_NOT_an_angle_or_cooked_shape_measurement=True,
        maximum_door_AABB_difference_m=error,door_AABB_matches_recorded_H3_within_1um=error<=1e-6,
        physical_clearance_or_causal_obstacle_failure_NOT_inferred_from_AABB=True,
        no_door_threshold_flag_or_scene_change_performed=True,
        same_H3_environment_NOT_assumed_for_H6=True)


def pinned_H3_geometry(pins,anchors):
    """Use the recorded lineage, never the newest similarly named folder."""
    controls={item['id']:item['world_xyyaw'] for item in anchors}
    found=[]
    for filename,expected in pins.items():
        path=Path(filename)
        if path.name!='hospital_h3_turn_geometry.json' or path.parent.parent.name!='hospital_h3_turn_setup':continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Pinned original H3 geometry changed')
        record=json.loads(path.read_bytes())
        if (record['start_xyyaw']==controls['H3_start'] and record['goal_xyyaw']==controls['H3_goal']):
            found.append((path,record))
    if len(found)!=1:raise ValueError('Exactly one matching ORIGINAL pinned H3 task geometry required')
    return found[0]


def transfer_bed_coverage_bounds(assets,anchors,supports,selected_roots):
    """Full recorded bed boxes, not hand-picked free goal poses or new floor."""
    disjoint_support_rectangles(supports)
    if (not assets or len({a['path'] for a in assets})!=len(assets)
            or len(selected_roots)!=2 or len(set(selected_roots))!=2 or not anchors
            or len({a['id'] for a in anchors})!=len(anchors)):
        raise ValueError('Complete unique asset roster, two distinct selected beds and original controls required')
    points=[vector(a['world_xyyaw'],3,'unchanged task control')[:2] for a in anchors]
    beds=[];found=set()
    for asset in assets:
        if asset['name_based_family']!='hospital_bed':continue
        path=asset['path']
        if not path.startswith('/World/Hospital/') or path.count('/')!=3:
            raise ValueError('Exact original Hospital bed identity required')
        bound=asset['recorded_full_collision_union_bound']
        lo=vector(bound['minimum_m'],3,'bed minimum');hi=vector(bound['maximum_m'],3,'bed maximum')
        if np.any(hi<=lo):raise ValueError('Nondegenerate full bed bounds required')
        polygon=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]
        missing=float(sum(abs(area(p)) for p in uncovered([polygon],supports)))
        selected=path in selected_roots
        if selected:points.extend([lo[:2],hi[:2]]);found.add(path)
        beds.append(dict(path=path,recorded_full_collision_union_bound=bound,
            selected_by_user_for_OFFLINE_candidate=selected,
            full_asset_XY_box_unsupported_authored_area_m2=missing,
            full_asset_XY_box_inside_recorded_support_union=missing<=1e-12,
            bed_box_support_NOT_robot_pose_or_cooked_support_or_accessibility=True))
    if found!=set(selected_roots):raise ValueError('Every selected bed must exist in the complete original bed roster')
    margin=math.hypot(1.630/2+.010,.965/2+.010)+.500
    points=np.asarray(points);lo=points.min(axis=0)-margin;hi=points.max(axis=0)+margin
    origin=np.floor(lo/RESOLUTION_M)*RESOLUTION_M
    shape=np.ceil((hi-origin)/RESOLUTION_M).astype(int)
    if int(shape[0])*int(shape[1])>MAX_CELLS:
        raise ValueError('Selected full coverage exceeds raster budget; no coarsening or bed omission')
    return dict(ordered_user_selected_bed_roots=list(selected_roots),all_named_bed_roots=beds,
        all_original_controls_retained=len(anchors),fixed_ROI_margin_m=margin,
        analysis_roi_xy_m=[float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1])],
        raster_shape_rows_columns=[int(shape[1]),int(shape[0])],
        selected_bed_support_NOT_H6_start_goal_or_route_adoption=True,
        new_support_or_scene_edit=False,actual_motion_authorized=False)


def bed_box_port_candidates(bound):
    """Six predeclared headings at four box sides; no map/planner feedback.

    Reuse H1/H2's existing150mm reserve AFTER the unchanged10mm full-body
    clearance. Box-side labels do not identify clinical head/foot docking.
    """
    lo=vector(bound['minimum_m'],3,'bed minimum');hi=vector(bound['maximum_m'],3,'bed maximum')
    if np.any(hi<=lo):raise ValueError('Full finite nondegenerate bed bounds required')
    x,y=(lo[:2]+hi[:2])/2.;a,b=1.630/2+.010,.965/2+.010;reserve=.150
    specs=[('bbox_north_heading0',[x,hi[1]+b+reserve,0.]),
           ('bbox_north_headingPi',[x,hi[1]+b+reserve,math.pi]),
           ('bbox_south_heading0',[x,lo[1]-b-reserve,0.]),
           ('bbox_south_headingPi',[x,lo[1]-b-reserve,math.pi]),
           ('bbox_west_heading0',[lo[0]-a-reserve,y,0.]),
           ('bbox_east_headingPi',[hi[0]+a+reserve,y,math.pi])]
    return [dict(id=name,world_xyyaw=[float(v) for v in pose],reserve_after_full_body_clearance_m=reserve,
        role='FIXED_BOX_SIDE_DIAGNOSTIC_NOT_CLINICAL_DOCKING_OR_ADOPTED_H6_GOAL') for name,pose in specs]


def bed_port_geometry_review(ports,blocked,obstacles,origin,resolution,roi,supports):
    """Separate known obstacle overlap, missing ROI and authored support."""
    if blocked.shape!=obstacles.shape or blocked.dtype!=bool or obstacles.dtype!=bool:
        raise ValueError('Same explicit boolean occupancy and obstacle-only masks required')
    origin=vector(origin,2,'map origin');roi=vector(roi,4,'coverage ROI');result=[]
    for port in ports:
        pose=vector(port['world_xyyaw'],3,'fixed port pose');polygon=footprint_polygon(pose)
        inside=bool(np.all(polygon>=roi[[0,2]]) and np.all(polygon<=roi[[1,3]]))
        rr,cc=polygon_cells(polygon,origin,resolution,blocked.shape)
        known_overlap=bool(np.any(obstacles[rr,cc]))
        missing=float(sum(abs(area(p)) for p in uncovered([polygon.tolist()],supports)))
        result.append(dict(**port,full_footprint_inside_analysis_ROI=inside,
            known_obstacle_cell_overlap=known_overlap,unsupported_full_footprint_area_m2=missing,
            full_rectangle_clear_in_recorded_map=inside and not pose_collision(blocked,pose,origin,resolution),
            outside_ROI_is_unknown_NOT_proven_physical_collision=not inside,
            pose_geometry_NOT_steering_route_cooked_support_or_clinical_alignment=True))
    return result


def export_transfer_port_review(medical_review_path,transfer_candidate_path):
    """All ports retained; no favorable final-goal selection or live action."""
    import io,os,zipfile
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    medical_path=Path(medical_review_path).resolve();candidate_path=Path(transfer_candidate_path).resolve()
    medical=json.loads(medical_path.read_bytes());candidate=json.loads(candidate_path.read_bytes())
    coverage_path=Path(medical['contract']['source_coverage']);coverage=json.loads(coverage_path.read_bytes())
    if (coverage['contract'].get('user_selected_transfer_candidate')!=str(candidate_path)
            or coverage['input_sha256'].get(str(candidate_path))!=sha(candidate_path)
            or candidate['actual_motion_authorized'] is not False):
        raise ValueError('Own explicit bed-pair coverage candidate required')
    pins={**medical['input_sha256'],**medical['output_sha256'],str(medical_path):sha(medical_path)}
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Original port-review source changed: '+name)
    beds={a['path']:a for a in medical['all_hospital_bed_roots_retained']}
    target=candidate['ordered_bed_roots'][1]
    if target not in beds:raise ValueError('Explicit selected target bed must remain in complete roster')
    ports=bed_box_port_candidates(beds[target]['recorded_full_collision_union_bound'])
    # Independently reproduce BOTH original service-side placement controls;
    # do not infer a new reserve or fit a favorable destination offset.
    reproduction=[]
    for control,bed,port in [('H1_start','SM_HospitalBed_02d11','bbox_north_heading0'),
                             ('H2_goal','SM_HospitalBed_02d12','bbox_south_headingPi')]:
        source=next(a for a in coverage['contract']['anchors'] if a['id']==control)
        generated=next(a for a in bed_box_port_candidates(beds['/World/Hospital/'+bed]['recorded_full_collision_union_bound']) if a['id']==port)
        error=float(np.max(np.abs(np.asarray(source['world_xyyaw'])-generated['world_xyyaw'])))
        if error>1e-12:raise ValueError('Existing H1/H2 service reserve rule does not reproduce')
        reproduction.append(dict(control=control,source_world_xyyaw=source['world_xyyaw'],
            formula_world_xyyaw=generated['world_xyyaw'],maximum_absolute_error=error))
    grid=next(Path(p) for p in medical['output_sha256'] if p.endswith('medical_keepout_grid.npz'))
    with np.load(grid) as a:
        rows=bed_port_geometry_review(ports,a['occupied'],a['obstacle_mesh_and_medical_keepouts'],
            a['origin_xy_m'],float(a['resolution_m']),coverage['coverage_grid']['analysis_roi_xy_m'],
            coverage['contract']['same_three_source_support_rectangles'])
    files=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for p in files:
            info=zipfile.ZipInfo(str(p.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())
    payload=buffer.getvalue()
    contract=dict(version='H6_PREDECLARED_ALL_BOX_SIDE_TARGET_PORT_GEOMETRY_ONLY_v1',
        source_medical_review=str(medical_path),source_transfer_candidate=str(candidate_path),target_bed=target,
        all_six_headings_at_four_positions_retained=True,source_H1_H2_reserve_m=.150,
        recorded_current_environment_NOT_the_future_fully_open_condition=True,
        no_goal_selected_from_method_outcomes=True,source_archive_sha256=hashlib.sha256(payload).hexdigest(),input_sha256=pins)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/target_ports'/key;target_path=folder/'target_port_review.json'
    if target_path.exists():
        saved=json.loads(target_path.read_bytes())
        if saved['contract']!=contract:raise ValueError('Original target-port contract changed')
        for p,h in saved['output_sha256'].items():
            if sha(p)!=h:raise ValueError('Original port output changed')
        return target_path
    if folder.exists():raise ValueError('Partial target-port review preserved; no overwrite')
    folder.mkdir(parents=True)
    with (folder/'executed_sources.zip').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    report=dict(contract=contract,source_service_placement_reproduction=reproduction,all_target_ports=rows,
        status='ALL_PREDECLARED_PORTS_REVIEWED_NOT_H6_GOAL_SELECTION_OR_ROUTE_ADMISSION',
        bedside_parallel_and_box_end_approaches_are_distinct_task_definitions=True,
        missing_ROI_NOT_reported_as_physical_infeasibility=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,output_sha256={str(p):sha(p) for p in (folder/'executed_sources.zip',folder/'software_contract.json')})
    evidence.write_exclusive(target_path,report)
    return target_path


def export_coverage_review(h5_protocol,author_manifest,common_map,*,transfer_candidate=None):
    """Append-only reusable coverage export, not a mission planner/live entry."""
    import io,os,zipfile
    import record_hospital_asset_measurement as source
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    common_path=Path(common_map).resolve();common=json.loads(common_path.read_bytes())
    raw,_,_,_=source.load_current_corridor_map_receipt(common['current_live_map_receipt'])
    anchors,pins=existing_task_controls(h5_protocol,author_manifest)
    if pins.get(str(common_path))!=sha(common_path):raise ValueError('Original frozen common-map source required')
    retained_actor=json.loads(Path(raw['retained_result']).read_bytes())['physics_preparation_snapshot']['new_robot_root']
    h1_plan=json.loads(Path(anchors[0]['source']).read_bytes());h1_manifest=Path(h1_plan['geometry_manifest'])
    if sha(h1_manifest)!=h1_plan['geometry_manifest_sha256']:raise ValueError('Original H1 geometry changed')
    h1=json.loads(h1_manifest.read_bytes())
    h3_path,h3=pinned_H3_geometry(pins,anchors)
    audit_path=Path(h3['source_audit']);audit=json.loads(audit_path.read_bytes())
    if (pins.get(str(h3_path))!=sha(h3_path) or sha(audit_path)!=h3['source_audit_sha256']):
        raise ValueError('Original H3 flush/open geometry provenance changed')
    pins.update({str(h1_manifest):sha(h1_manifest),str(audit_path):sha(audit_path),str(common_path):sha(common_path)})
    margin=math.hypot(1.630/2+.010,.965/2+.010)+.500
    points=np.asarray([item['world_xyyaw'][:2] for item in anchors]);lo=points.min(axis=0)-margin;hi=points.max(axis=0)+margin
    roi=[float(lo[0]),float(hi[0]),float(lo[1]),float(hi[1])]
    transfer={}
    if transfer_candidate is not None:
        candidate_path=Path(transfer_candidate).resolve();candidate=json.loads(candidate_path.read_bytes())
        if (candidate['actual_motion_authorized'] is not False or candidate['robot_start_xyyaw'] is not None
                or candidate['robot_goal_xyyaw'] is not None
                or candidate['status']!='USER_SELECTED_BED_PAIR_FOR_OFFLINE_REVIEW_NOT_FROZEN_ROBOT_POSES_OR_MOTION'):
            raise ValueError('Explicit offline bed-pair candidate without invented robot poses required')
        for filename,value in candidate['input_sha256'].items():
            path=(_ijcas_path(ROOT, filename)).resolve()
            if sha(path)!=value:raise ValueError('Selected bed/environment source changed: '+str(path))
            pins[str(path)]=value
        medical_path=(_ijcas_path(ROOT, candidate['original_medical_review'])).resolve()
        if str(medical_path) not in pins:raise ValueError('Original full medical roster must be pinned')
        medical=json.loads(medical_path.read_bytes())
        for filename,value in {**medical['input_sha256'],**medical['output_sha256']}.items():
            if filename in pins and pins[filename]!=value:raise ValueError('Conflicting original source provenance')
            if sha(filename)!=value:raise ValueError('Original medical evidence changed: '+filename)
            pins[filename]=value
        if medical['contract']['original_environment_digest']!=raw['all_4997_protected_environment_rows']:
            raise ValueError('Same original inventory/environment required for selected beds')
        beds=transfer_bed_coverage_bounds(medical['medical_asset_policy']['assets'],anchors,
            common['exact_same_authored_support_rectangles'],candidate['ordered_bed_roots'])
        roi=beds['analysis_roi_xy_m'];pins[str(candidate_path)]=sha(candidate_path)
        transfer=dict(user_selected_transfer_candidate=str(candidate_path),selected_bed_coverage=beds,
            coverage_rule_includes_all_ten_controls_AND_both_full_selected_bed_bounds=True,
            recorded_current_door_threshold_flags_NOT_replaced_by_adopted_future_condition=True)
    buffer=io.BytesIO();files=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            member=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));member.external_attr=0o100644<<16
            archive.writestr(member,path.read_bytes())
    payload=buffer.getvalue()
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    for filename in (Path(source.__file__).resolve(),Path(__import__('hospital_geometry_tools').__file__).resolve(),
            Path(__import__('hospital_h4_support_design').__file__).resolve(),font):pins[str(filename)]=sha(filename)
    contract=dict(version=VERSION,role='H6_OFFLINE_MAP_COVERAGE_AND_CONDITION_COMPATIBILITY_NOT_A_FROZEN_MISSION',
        original_common_map=str(common_path),all_stage_receipt=common['current_live_map_receipt'],
        anchors=anchors,ROI_rule=('union of ALL ten original controls AND BOTH full user-selected bed bounds + original margin'
            if transfer else 'union of ALL ten original task-pose controls + circumscribed full footprint radius + fixed 0.500m'),
        fixed_ROI_margin_m=margin,analysis_roi_xy_m=roi,retained_actor_at_recording=retained_actor,
        same_three_source_support_rectangles=common['exact_same_authored_support_rectangles'],
        original_height_slab_m=list(SLAB_M),resolution_m=RESOLUTION_M,maximum_cells=MAX_CELLS,
        executed_source_archive_sha256=hashlib.sha256(payload).hexdigest(),source_members_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        H6_task_start_goal_or_waypoints_frozen=False,body_state_or_reference_endpoint_stitching=False,
        source_4997_row_environment=raw['all_4997_protected_environment_rows'],
        native_cooking_or_support_or_drive_or_model_validation=False,new_trial_count=0,actual_motion_authorized=False,
        source_USD_or_Session_or_live_timeline_modified=False,use_for_final_statistics=False,input_sha256=pins,**transfer)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/coverage'/key;target=folder/'coverage_review.json'
    if target.exists():
        report=json.loads(target.read_bytes())
        if report['contract']!=contract or (folder/'executed_sources.zip').read_bytes()!=payload:
            raise ValueError('Existing coverage definition changed; no overwrite')
        for path,value in {**report['input_sha256'],**report['output_sha256']}.items():
            if sha(path)!=value:raise ValueError('Existing coverage evidence changed: '+path)
        return target
    if folder.exists():raise ValueError('Partial coverage output preserved; no automatic restart/overwrite')
    folder.mkdir(parents=True,exist_ok=False)
    with (folder/'executed_sources.zip').open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    grid,blocked,supported,obstacles=build_authored_grid(raw['inventory'],retained_actor,roi,common['exact_same_authored_support_rectangles'])
    controls=task_anchor_review(anchors,grid,blocked,common['exact_same_authored_support_rectangles'])
    conditions=doorway_condition_review(h3,audit,raw['inventory'],door_root=h1['door_state']['path'])
    raster=folder/'authored_coverage_grid.npz'
    np.savez_compressed(raster,occupied=blocked,obstacle_AABBs=obstacles,supported_full_cells=supported,
        origin_xy_m=grid['raster']['origin_xy_m'],resolution_m=RESOLUTION_M)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon
    prop=FontProperties(fname=str(font))
    image=np.where(~supported,2,np.where(obstacles,1,0));origin=np.asarray(grid['raster']['origin_xy_m'])
    figure,ax=plt.subplots(figsize=(8,11),layout='constrained')
    ax.imshow(image,origin='lower',cmap=ListedColormap(['#ffffff','#47515c','#dce9f3']),vmin=0,vmax=2,
        extent=[origin[0],origin[0]+blocked.shape[1]*RESOLUTION_M,origin[1],origin[1]+blocked.shape[0]*RESOLUTION_M])
    for i,item in enumerate(controls['anchors']):
        pose=item['world_xyyaw'];color='#16744b' if item['full_original_10mm_footprint_clear_in_source_raster'] else '#bd3138'
        ax.add_patch(Polygon(footprint_polygon(pose),facecolor='none',edgecolor=color,linewidth=.8,alpha=.65))
        ax.plot(pose[0],pose[1],'o',color=color,markersize=3)
        ax.annotate(item['id'],pose[:2],xytext=((7 if i%2 else -60),(8 if i%2 else -13)),
            textcoords='offset points',fontsize=8,color=color)
        ax.arrow(pose[0],pose[1],.28*math.cos(pose[2]),.28*math.sin(pose[2]),color=color,width=.01,length_includes_head=True)
    ax.set(aspect='equal',xlabel='Hospital world X (m)',ylabel='Hospital world Y (m)')
    ax.set_title('H6 통합 지도 범위 검토 — 경로·주행 검증 아님',fontproperties=prop)
    ax.text(.02,.02,'회색: 원래 충돌체 경계상자\n청색: 기록된 지지면이 셀 전체를 덮지 못하는 영역\n표식: 기존 과제 좌표, H6 경유점 아님',
        transform=ax.transAxes,fontproperties=prop,fontsize=9,bbox=dict(facecolor='white',alpha=.92,edgecolor='none'))
    preview=folder/'authored_coverage_review.png'
    try:figure.savefig(preview,dpi=150)
    finally:plt.close(figure)
    outputs={str(path):sha(path) for path in (raster,preview,folder/'executed_sources.zip',folder/'software_contract.json')}
    for path,value in pins.items():
        if sha(path)!=value:raise ValueError('Source changed during offline coverage analysis: '+path)
    report=dict(contract=contract,coverage_grid=grid,source_control_review=controls,H3_environment_comparison=conditions,
        status='AUTHORED_COVERAGE_PREPARED_NOT_A_CONNECTED_FULL_BODY_MISSION_OR_DRIVE_ADMISSION',
        all_ten_source_footprints_clear=all(row['full_original_10mm_footprint_clear_in_source_raster'] for row in controls['anchors']),
        all_ten_source_full_footprints_have_recorded_authored_support=all(row['unsupported_full_footprint_area_m2']<=1e-12 for row in controls['anchors']),
        point_center_components_do_NOT_establish_actual_door_closed_or_global_infeasibility=True,
        must_review_mesh_apertures_and_condition_compatibility_before_H6_planning=True,
        next_gate='Refine actual doorway collision mesh geometry and explicitly freeze one common mission environment; never waive a bound, teleport, stitch PASSes or relabel H6 as complete.',
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,input_sha256=pins,output_sha256=outputs)
    evidence.write_exclusive(target,report)
    return target


MESH_REVIEW_VERSION='H6_RECORDED_STATIC_MESH_PROJECTION_FIXED_BLOCKERS_RETAINED_v2'


def semantic_array(value,name):
    """Decode the existing lossless typed archive codec, not flattened guesses."""
    import hospital_compact_usd_fingerprint as compact
    decoded=compact._numeric_json(value)
    if decoded is None:raise ValueError('Exact numeric USD archive container required: '+name)
    descriptor,flat=decoded;shape=tuple(int(size) for _,size in descriptor)
    if flat is None:return np.empty(shape,dtype=float)
    result=flat.reshape(shape)
    if not np.isfinite(result).all():raise ValueError('Finite archived USD array required: '+name)
    return result


def read_authenticated_mesh_rows(manifest_path, selected_paths, *, expected_manifest_sha256,
        allowed_prim_types=('Mesh',)):
    """OFFLINE selective decoding of immutable source rows, not a full rehash.

    Use the original archive's exact section/row boundaries, but decode only
    explicitly requested Hospital meshes. Every decoded row must match its
    authenticated compact semantic digest. Scan all row names for duplicate
    or missing identities. Never load the 1.68GB document into Python/Kit.
    Stat equality is NOT reported as a newly verified full-file SHA.
    """
    import mmap,re,sys
    import hospital_compact_usd_fingerprint as compact
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('Selective source archive decoding is OFFLINE ONLY')
    if (allowed_prim_types not in (('Mesh',),('Mesh','Material'))
            or not isinstance(selected_paths,(list,tuple)) or not 1<=len(selected_paths)<=64
            or any(not isinstance(p,str) or not p.startswith('/World/Hospital/') for p in selected_paths)
            or len(set(selected_paths))!=len(selected_paths)):
        raise ValueError('One to64 unique explicit Hospital source mesh paths required')
    manifest_path=Path(manifest_path).resolve();raw=manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:
        raise ValueError('Authenticated original compact manifest required')
    manifest=json.loads(raw);expected=manifest['compact_environment_rows']
    if (manifest['codec']!=compact.FORMAT or manifest['prim_row_count']!=len(expected)
            or not set(selected_paths)<=set(expected)):
        raise ValueError('Complete original compact roster and exact selected identities required')
    archive=Path(manifest['original_archive']);before=archive.stat()
    if (before.st_size!=manifest['original_archive_bytes'] or before.st_mtime_ns!=manifest['original_archive_mtime_ns']):
        raise ValueError('Original archive metadata changed; no silent source substitution')
    selected=set(selected_paths);rows={};seen=set();decoded_bytes=0
    start_marker=b'  "protected_environment_before_Stop": {'
    end_marker=b'\n  },\n  "environment_after_actual_static":'
    pattern=re.compile(rb'^    "((?:[^"\\\r\n]|\\.)*)": \{',re.MULTILINE)
    with archive.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
        start=data.find(start_marker);end=data.find(end_marker,start)
        if start<0 or end<0:raise ValueError('Exact original archive section required')
        matches=list(pattern.finditer(data,start,end));decoder=json.JSONDecoder()
        for i,match in enumerate(matches):
            name=json.loads(b'"'+match.group(1)+b'"')
            if name in seen:raise ValueError('Duplicate original archived prim: '+name)
            seen.add(name)
            if name not in selected:continue
            limit=matches[i+1].start() if i+1<len(matches) else end
            decoded_bytes+=limit-match.start()
            if decoded_bytes>32*1024*1024:raise ValueError('Selected source rows exceed32MiB offline decoding budget')
            piece=data[match.start():limit].decode('utf-8').strip().rstrip(',')
            key,index=decoder.raw_decode(piece);index+=1
            while index<len(piece) and piece[index].isspace():index+=1
            value,consumed=decoder.raw_decode(piece,index)
            if key!=name or piece[consumed:].strip():raise ValueError('Exact selected archived row boundary required')
            if (value.get('type') not in allowed_prim_types or not value.get('active')
                    or compact.archived_row(name,value,manifest['approved_disabled_paths'])!=expected[name]):
                raise ValueError('Selected source mesh semantic fingerprint changed: '+name)
            rows[name]=value
    after=archive.stat()
    if ((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            or seen!=set(expected) or set(rows)!=selected):
        raise ValueError('Archive identity/stat/complete roster or selected source rows changed')
    return rows,dict(original_manifest=str(manifest_path),manifest_sha256=expected_manifest_sha256,
        original_archive=str(archive),historical_full_archive_sha256=manifest['original_archive_sha256'],
        new_full_archive_rehash_performed=False,complete_row_name_roster_verified=len(seen),
        selected_semantically_authenticated_meshes=sum(row['type']=='Mesh' for row in rows.values()),
        selected_semantically_authenticated_materials=sum(row['type']=='Material' for row in rows.values()),
        selected_decoded_bytes=decoded_bytes,
        selected_row_sha256={p:expected[p]['row_sha256'] for p in selected_paths},
        archive_metadata_unchanged_during_read=True,no_Kit_or_source_scene_access=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False)


def source_mesh_composition_metadata(stage, expected_rows):
    """GETs of existing source values, instances and shading metadata only.

    Authenticate each complete collider/material-value row with the original
    compact codec. The extra interpolation/instance/primvar metadata is a NEW
    observation, not invented from point counts. Never load, deinstance,
    author, cook, select, frame, refresh or start a simulation here.
    """
    from pxr import Sdf,UsdGeom,UsdShade
    import hospital_compact_usd_fingerprint as compact
    if (not isinstance(expected_rows,dict) or not 1<=len(expected_rows)<=64
            or any(not isinstance(p,str) or not p.startswith('/World/Hospital/')
                or not Sdf.Path(p).IsPrimPath() for p in expected_rows)):
        raise ValueError('Bounded unique original Hospital mesh rows required')
    cache=UsdGeom.XformCache();records=[]
    for path,row in expected_rows.items():
        prim=stage.GetPrimAtPath(path);mesh=UsdGeom.Mesh(prim)
        if not mesh or not prim.IsActive() or not prim.IsLoaded() or row.get('type')!='Mesh':
            raise ValueError('Existing active loaded original source mesh required: '+path)
        attributes={a.GetName():dict(default_sha256=compact.native_digest(a.Get()),
            samples=[[t,compact.native_digest(a.Get(t))] for t in a.GetTimeSamples()])
            for a in prim.GetAttributes()}
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        content=dict(active=bool(prim.IsActive()),type=prim.GetTypeName(),attributes=attributes,
            relationships={r.GetName():list(map(str,r.GetTargets())) for r in prim.GetRelationships()},
            world_sha256=compact.native_digest(cache.GetLocalToWorldTransform(prim)),
            bound_physics_material=str(material.GetPath()) if material else None)
        digest=hashlib.sha256(compact._json_bytes(content)).hexdigest()
        if digest!=compact.archived_row(path,row,[])['row_sha256']:
            raise ValueError('Original source mesh value/world/material fingerprint changed: '+path)
        ancestors=[];current=prim
        while current and not current.IsPseudoRoot():
            if len(ancestors)>=16:raise ValueError('Source instance ancestry exceeds16prim GET budget')
            prototype=current.GetPrototype() if current.IsInstance() else None
            source=current.GetPrimInPrototype() if current.IsInstanceProxy() else None
            ancestors.append(dict(path=str(current.GetPath()),type_name=current.GetTypeName(),
                active=bool(current.IsActive()),loaded=bool(current.IsLoaded()),
                instanceable=bool(current.IsInstanceable()),instance=bool(current.IsInstance()),
                instance_proxy=bool(current.IsInstanceProxy()),
                prototype_path=str(prototype.GetPath()) if prototype else None,
                corresponding_prim_in_prototype=str(source.GetPath()) if source else None,
                prim_stack=[dict(layer=s.layer.identifier,path=str(s.path),
                    specifier=str(s.specifier)) for s in current.GetPrimStack()]))
            current=current.GetParent()
        primvars=[]
        for pv in UsdGeom.PrimvarsAPI(prim).FindPrimvarsWithInheritance():
            if len(primvars)>=64:raise ValueError('Inherited primvars exceed64item source GET budget')
            attr=pv.GetAttr();indices=pv.GetIndicesAttr()
            primvars.append(dict(name=pv.GetPrimvarName(),attribute_path=str(attr.GetPath()),
                type_name=str(pv.GetTypeName()),interpolation=str(pv.GetInterpolation()),
                element_size=int(pv.GetElementSize()),indexed=bool(pv.IsIndexed()),
                value_sha256=compact.native_digest(attr.Get()),time_samples=list(map(float,attr.GetTimeSamples())),
                indices_sha256=compact.native_digest(indices.Get()) if indices else None))
        normals=mesh.GetNormalsAttr();normal_value=normals.Get()
        records.append(dict(path=path,original_complete_value_world_material_row_sha256=digest,
            exact_original_values_world_topology_normals_extent_and_material_retained=True,
            points_type=str(mesh.GetPointsAttr().GetTypeName()),
            point_count=len(mesh.GetPointsAttr().Get()),face_count=len(mesh.GetFaceVertexCountsAttr().Get()),
            normals_type=str(normals.GetTypeName()),normal_count=len(normal_value) if normal_value is not None else 0,
            normals_interpolation=str(mesh.GetNormalsInterpolation()),
            normals_authored_interpolation_metadata=normals.GetMetadata('interpolation'),
            normals_value_sha256=compact.native_digest(normal_value),
            orientation=str(mesh.GetOrientationAttr().Get()),subdivision_scheme=str(mesh.GetSubdivisionSchemeAttr().Get()),
            mesh_and_ancestor_composition=ancestors,inherited_primvars=primvars,
            whole_stage_joint_search_and_cooked_shapes_NOT_queried=True))
    return records


def checked_source_mesh_metadata_GET(stage, timeline, current_stage, snapshot,
        expected_timeline, expected_tensor, expected_rows):
    """Synchronous bounded readback; callbacks provide existing GET handles.

    Stable existing PAUSED B2 state is required before AND after reading.
    No event-loop yields or mutating timeline/physics/stage operation.
    """
    state=evidence.timeline_state(timeline)
    if state!=expected_timeline or state['playing'] or state['stopped']:
        raise ValueError('Exact saved PAUSED boundary required; no recovery or timeline command')
    if current_stage()!=stage:raise ValueError('Existing stage identity changed before source GET')
    layers=lambda:dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    before=layers()
    if sum(len(s.encode()) for s in before.values())>2*1024*1024:
        raise ValueError('Existing layer text exceeds2MiB GET budget')
    first=snapshot()
    if first!=expected_tensor:raise ValueError('Saved body/all12joints/all13links/gains/targets changed; no recovery')
    records=source_mesh_composition_metadata(stage,expected_rows)
    last=snapshot();after=layers();final_time=evidence.timeline_state(timeline)
    checks=dict(same_stage=current_stage()==stage,Root_Session_exactly_retained=before==after,
        actual_PAUSED_timeline_exactly_retained=state==final_time,
        body_all12joints_all13links_gains_targets_exactly_retained=first==last==expected_tensor,
        all_selected_source_value_world_material_fingerprints_exact=True)
    return dict(status='SOURCE_MESH_COMPOSITION_AND_RETAINED_PAUSED_STATE_VERIFIED_GET_ONLY'
        if all(checks.values()) else 'HOLD_SOURCE_GET_BOUNDARY_CHANGED_NO_RECOVERY',checks=checks,
        actual_timeline=final_time,actual_tensor_before=first,actual_tensor_after=last,meshes=records,
        layer_sha256_before={k:hashlib.sha256(v.encode()).hexdigest() for k,v in before.items()},
        layer_sha256_after={k:hashlib.sha256(v.encode()).hexdigest() for k,v in after.items()},
        whole4997_environment_rows_and_global_joint_graph_NOT_requeried=True,
        no_scene_open_load_author_deinstance_cook_display_or_timeline_command=True,
        source_metadata_NOT_environment_static_drive_or_sensor_admission=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)


def run_source_mesh_metadata_GET_once(contract_path, *, authorized=False):
    """User-operated entry for the shared GET reader, never a motion runner."""
    import builtins,runpy,time
    from datetime import datetime,timezone
    if authorized is not True:raise RuntimeError('Explicit GET-only invocation required')
    path=Path(contract_path).resolve()
    if path.stat().st_size>4*1024*1024:raise ValueError('Bounded reviewed GET contract required')
    contract=json.loads(path.read_bytes());target=Path(contract['GET_target'])
    if (contract.get('version')!='H6_ORIGINAL_MESH_COMPOSITION_GET_ONLY_v1'
            or contract.get('actual_motion_authorized') is not False
            or contract.get('new_PhysX_trial_count')!=0 or target.parent!=path.parent):
        raise ValueError('Explicit reviewed same-folder metadata GET definition required')
    if target.exists():
        print('[H6 SOURCE GET] Already recorded; NOT a new live readback:',target,flush=True)
        return str(target)
    for name in ('ijcas_hospital_latest_task_experiment','ijcas_h4_latest_experiment_task',
                 'ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():raise RuntimeError('Another research task is active; no competing GET')
    for name,digest in contract['input_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Reviewed source GET input changed: '+name)
    raw=json.loads(Path(contract['required_B2_result']).read_bytes())
    selected=json.loads(Path(contract['authenticated_selected_source_rows']).read_bytes())['original_mesh_rows']
    if (set(selected)!=set(contract['selected_paths'])
            or raw['physics_preparation_snapshot']['new_robot_root']!=contract['required_B2_actor']
            or raw['actual_final_timeline']!=contract['required_actual_saved_timeline']
            or len(raw['actual_post_route_tensor_snapshot']['names'])!=12
            or len(raw['actual_post_route_tensor_snapshot']['link_names'])!=13):
        raise ValueError('Exact selected source roster and saved12joint/13link B2 boundary required')
    loader=runpy.run_path(str(_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py')))['load']
    shared=loader(_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_subfloor_compact_path_launch.py'))
    normalize=shared['normalize_runtime_evidence'];snapshotter=shared['preparation'].old.tensor_snapshot
    started=time.monotonic()
    report=dict(status='HOLD_SOURCE_GET_NOT_COMPLETED_NO_RECOVERY',error=None,
        current_UTC=datetime.now(timezone.utc).isoformat(),source_contract=str(path),
        input_sha256=contract['input_sha256'],new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    try:
        import omni.usd
        import omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager
        usd=omni.usd.get_context();stage=usd.get_stage();timeline=omni.timeline.get_timeline_interface()
        if stage is None:raise ValueError('Existing stage required; no open/reload')
        if evidence.timeline_state(timeline)!=raw['actual_final_timeline']:
            raise ValueError('Current timeline differs from saved PAUSED B2; no recovery')
        view=SimulationManager.get_physics_sim_view()
        if view is None:raise ValueError('Existing physics view required; never Play/initialize to create it')
        tensor=view.create_articulation_view(contract['required_B2_actor']+'/base_link')
        snapshot=lambda:normalize(snapshotter(tensor))
        report.update(checked_source_mesh_metadata_GET(stage,timeline,usd.get_stage,snapshot,
            raw['actual_final_timeline'],raw['actual_post_route_tensor_snapshot'],selected))
    except Exception as error:
        report.update(status='HOLD_SOURCE_GET_REQUIRES_REVIEW_NO_RECOVERY',error=type(error).__name__+': '+str(error))
    report['wall_time_s']=float(time.monotonic()-started)
    evidence.write_exclusive(target,normalize(report))
    print('=== H6 ORIGINAL SOURCE INSTANCE / NORMAL / PRIMVAR METADATA GET ONLY ===',flush=True)
    print('status:',report['status'],'error:',report.get('error'),flush=True)
    print('retained B2 body/all12joints/all13links/layers/timeline:',report.get('checks'),flush=True)
    for mesh in report.get('meshes',[]):
        print(mesh['path'],'normals:',mesh['normals_interpolation'],mesh['normal_count'],
            'instanceable ancestry:',[p['path'] for p in mesh['mesh_and_ancestor_composition'] if p['instanceable']],flush=True)
    print('saved:',target,flush=True)
    print('No Play/Stop/Pause/reset, source edit, load, deinstance, camera/renderer, cooking, staging, static or drive.',flush=True)
    return str(target)


def export_source_mesh_metadata_GET(native_review_path, retained_result_path, prior_GET_contract_path,
        *, expected_prior_GET_sha256):
    """Offline append-only contract/launcher for the shared source GET entry.

    Historical source bytes remain in their archives. Only this extensible
    geometry helper has a newly pinned revision; all other prior runtime
    Python dependencies must still match the authenticated previous reader.
    No new standalone Python script, Kit connection or trial is created.
    """
    import io,os,sys,zipfile
    if any(n in sys.modules for n in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('GET preparation is OFFLINE ONLY; do not prepare in Script Editor')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    native_path=Path(native_review_path).resolve();native=json.loads(native_path.read_bytes())
    prior_path=Path(prior_GET_contract_path).resolve()
    if sha(prior_path)!=expected_prior_GET_sha256:raise ValueError('Authenticated prior GET dependency lineage required')
    prior=json.loads(prior_path.read_bytes());result_path=Path(retained_result_path).resolve();raw=json.loads(result_path.read_bytes())
    if (prior['required_B2_result']!=str(result_path) or sha(result_path)!=prior['input_sha256'][str(result_path)]
            or native.get('actual_motion_authorized') is not False
            or native.get('all_eight_native_masks_equal_planning') is not True):
        raise ValueError('Own verified native geometry and original retained B2 result required')
    for p,h in native['output_sha256'].items():
        if sha(p)!=h:raise ValueError('Native source geometry output changed: '+p)
    for p,h in raw['input_sha256'].items():
        if sha(p)!=h:raise ValueError('Protected completed B2 evidence changed: '+p)
    selected_path=native_path.parent/'authenticated_selected_source_rows.json'
    selected=json.loads(selected_path.read_bytes());paths=list(selected['original_mesh_rows'])
    if (len(paths)!=8 or set(paths)!={r['path'] for r in native['meshes']}
            or selected['receipt']!=native['source_receipt']):
        raise ValueError('Exact authenticated eight source mesh identities required')
    source_path=Path(__file__).resolve();dependencies={}
    for name,h in prior['input_sha256'].items():
        if not name.endswith('.py'):continue
        actual=sha(name)
        if Path(name).resolve()!=source_path and actual!=h:
            raise ValueError('Previously frozen GET runtime dependency changed: '+name)
        dependencies[name]=actual
    for name in ('hospital_private_loader.py','hospital_geometry_tools.py','hospital_h4_support_design.py',
                 'hospital_runtime_evidence_io.py','hospital_compact_usd_fingerprint.py','hospital_h4_root_difference_readback.py'):
        p=_ijcas_path(ROOT, 'studies/IJCAS/scripts', name);dependencies[str(p)]=sha(p)
    dependencies[str(source_path)]=sha(source_path)
    folder=native_path.parent/'source_metadata_GET_r01';target=folder/'live_source_metadata.json'
    contract_path=folder/'GET_contract.json'
    contract=dict(version='H6_ORIGINAL_MESH_COMPOSITION_GET_ONLY_v1',
        role='ORIGINAL_SOURCE_METADATA_AND_SAME_RETAINED_B2_GET_NOT_H6_AUTHORING_OR_DRIVE',
        required_B2_result=str(result_path),required_B2_actor=raw['physics_preparation_snapshot']['new_robot_root'],
        required_actual_saved_timeline=raw['actual_final_timeline'],
        authenticated_selected_source_rows=str(selected_path),selected_paths=paths,GET_target=str(target),
        input_sha256={**dependencies,**{str(p):sha(p) for p in (native_path,selected_path,result_path,prior_path)}},
        completed_B2_input_pins_verified_offline=len(raw['input_sha256']),
        whole875_historical_files_NOT_rehashed_in_GUI_GET=True,
        original_large_archive_NOT_read_in_GUI=True,no_actor_or_physics_initialization=True,
        existing_physics_view_read_handle_only=True,no_event_loop_yield_or_timeline_command=True,
        source_normal_interpolation_instance_ancestry_and_inherited_primvar_GETs=True,
        no_source_USD_Kit_environment_static_drive_or_final_statistics_admission=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    if folder.exists():raise ValueError('Source metadata GET preparation already exists; no overwrite or automatic new revision')
    folder.mkdir()
    evidence.write_exclusive(contract_path,contract)
    command=(f'''"""H6 source metadata GET ONLY; use the existing shared reader."""\nif globals().get("ijcas_h6_source_metadata_GET_only_authorized") is not True:\n    raise RuntimeError("Explicit GET-only invocation required")\nfrom pathlib import Path\nimport hashlib,json,runpy\n_contract=Path({str(contract_path)!r})\nif hashlib.sha256(_contract.read_bytes()).hexdigest()!={sha(contract_path)!r}:\n    raise ValueError("Reviewed source GET contract changed")\n_definition=json.loads(_contract.read_bytes())\nfor _name,_digest in _definition["input_sha256"].items():\n    if hashlib.sha256(Path(_name).read_bytes()).hexdigest()!=_digest:\n        raise ValueError("Reviewed GET source changed: "+_name)\n_load=runpy.run_path({str(_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py'))!r})["load"]\n_reader=_load({str(source_path)!r})\n_reader["run_source_mesh_metadata_GET_once"](_contract,authorized=True)\n''')
    compile(command,str(folder/'reviewed_GET_command.txt'),'exec')
    with (folder/'reviewed_GET_command.txt').open('xb') as f:f.write(command.encode());f.flush();os.fsync(f.fileno())
    buffer=io.BytesIO();sources=[Path(n) for n in dependencies]
    sources.append(_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py'))
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as saved:
        for p in sources:saved.writestr(str(p.relative_to(ROOT)),p.read_bytes())
    with (folder/'executed_GET_sources.zip').open('xb') as f:f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
    return contract_path


def mesh_session_override_plan(metadata, original_rows, geometry_arrays):
    """Plan exact Session opinions from an observed, completed source GET.

    Pure data validation; not a live authoring entry point or permission.
    Preserve the source local transforms. A changed leaf frame is represented
    by one additional matrix op, not by replacing the original scale/ops.
    """
    import hospital_compact_usd_fingerprint as compact
    if (metadata.get('status')!='SOURCE_MESH_COMPOSITION_AND_RETAINED_PAUSED_STATE_VERIFIED_GET_ONLY'
            or metadata.get('error') is not None or not metadata.get('checks')
            or not all(value is True for value in metadata['checks'].values())):
        raise ValueError('Successful actual original-source metadata GET required')
    records=metadata.get('meshes',[]);paths=[r.get('path') for r in records]
    if (not 1<=len(paths)<=64 or len(set(paths))!=len(paths)
            or set(paths)!=set(original_rows) or set(paths)!=set(geometry_arrays)):
        raise ValueError('Exact bounded source/metadata/geometry identity set required')
    roots=set();items=[]
    for record in records:
        path=record['path'];row=original_rows[path];arrays=geometry_arrays[path]
        if (not isinstance(path,str) or not path.startswith('/World/Hospital/')
                or record['original_complete_value_world_material_row_sha256']!=
                    compact.archived_row(path,row,[])['row_sha256']
                or not record['exact_original_values_world_topology_normals_extent_and_material_retained']
                or record['normals_interpolation']!='vertex'
                or record['point_count']!=record['normal_count']
                or record['subdivision_scheme']!='none' or record['orientation']!='rightHanded'):
            raise ValueError('Exact static vertex-normal original mesh required: '+str(path))
        ancestors=record['mesh_and_ancestor_composition']
        selected=[a['path'] for a in ancestors if a['instanceable'] and a['instance']]
        if (not ancestors or ancestors[0]['path']!=path or not ancestors[0]['instance_proxy']
                or len(selected)!=1 or selected[0].count('/')!=3
                or not selected[0].startswith('/World/Hospital/')
                or not path.startswith(selected[0]+'/')):
            raise ValueError('One exact direct Hospital instance root per selected mesh required')
        roots.add(selected[0]);required=('local_points','local_normals','local_extent','counts','indices',
            'world_matrix','rounded_world_points_m')
        if any(name not in arrays for name in required):raise ValueError('Complete native geometry recipe required')
        # Validate native local arrays directly. Inverting already-rounded
        # world points again is not bitwise idempotent near zero and must not
        # introduce a second quantization into the reviewed source recipe.
        checked,local_check=mesh_local_geometry_recipe(row,np.asarray(arrays['local_points'],float),
            target_world_matrix=np.eye(4))
        for name in ('local_points','local_normals','local_extent','counts','indices'):
            value=np.asarray(arrays[name]);expected=np.asarray(local_check[name])
            if value.dtype!=expected.dtype or not np.array_equal(value,expected):
                raise ValueError('Geometry recipe is not the exact checked native representation: '+name)
        matrix=np.asarray(arrays['world_matrix']);world=np.asarray(arrays['rounded_world_points_m'])
        if (matrix.dtype!=np.dtype('float64') or world.dtype!=np.dtype('float64')
                or matrix.shape!=(4,4) or not np.isfinite(matrix).all()
                or not np.array_equal(matrix[:3,3],np.zeros(3)) or matrix[3,3]!=1.
                or np.linalg.cond(matrix[:3,:3])>1e10
                or not np.array_equal((np.c_[np.asarray(arrays['local_points'],float),
                    np.ones(len(arrays['local_points']))]@matrix)[:,:3],world)):
            raise ValueError('Exact affine native points-to-world reconstruction required')
        canonical=arrays
        if len(canonical['local_points'])!=record['point_count']:
            raise ValueError('Point count differs from actually observed source')
        source_world=semantic_array(row['world_matrix'],'source world')
        frame_changed=not np.array_equal(source_world,canonical['world_matrix'])
        opinions=['points','normals','extent']
        if frame_changed:opinions+=['xformOp:transform:h6_layout','xformOpOrder']
        items.append(dict(path=path,instance_root=selected[0],
            original_row_sha256=record['original_complete_value_world_material_row_sha256'],
            source_world_matrix=source_world.tolist(),target_world_matrix=canonical['world_matrix'].tolist(),
            point_count=record['point_count'],normal_interpolation='vertex',
            additional_leaf_matrix_op=frame_changed,allowed_property_opinions=opinions,
            native_array_sha256={name:hashlib.sha256(np.ascontiguousarray(canonical[name]).tobytes()).hexdigest()
                for name in required},
            observed_primvars=record['inherited_primvars']))
    return dict(version='H6_SELECTED_INSTANCE_SESSION_GEOMETRY_OPINIONS_OFFLINE_v1',
        roots_to_deinstance=sorted(roots),meshes=items,
        only_points_normals_extent_and_explicit_leaf_matrix_op=True,
        never_replace_original_xform_ops_or_edit_shared_prototypes=True,
        collision_material_UV_and_topology_changes_authorized=False,
        new_actual_scene_or_timeline_action_authorized=False,
        actual_source_metadata_is_historical_NOT_new_current_live_readback=True)


def _offline_anonymous_USD_only(stage):
    import sys
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('Anonymous Session composition review is OFFLINE ONLY; not an Isaac command')
    if (not stage or not stage.GetRootLayer().anonymous or not stage.GetSessionLayer().anonymous
            or any(not layer.anonymous for layer in stage.GetUsedLayers())):
        raise ValueError('Only anonymous local fixture layers; no original USD or external asset opening')


def _composed_fixture_snapshot(stage):
    """Complete bounded fixture values, not prototype-identity comparisons."""
    from pxr import Usd,UsdGeom,UsdShade
    import hospital_compact_usd_fingerprint as compact
    result={};cache=UsdGeom.XformCache()
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if len(result)>=4096:raise ValueError('Anonymous fixture exceeds4096-prim review budget')
        attrs={}
        for attr in prim.GetAttributes():
            attrs[attr.GetName()]=dict(default=compact.native_digest(attr.Get()),
                samples=[[float(t),compact.native_digest(attr.Get(t))] for t in attr.GetTimeSamples()],
                metadata=compact.native_digest({k:v for k,v in attr.GetAllMetadata().items()
                    if k not in ('default','timeSamples')}),connections=list(map(str,attr.GetConnections())))
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        result[str(prim.GetPath())]=dict(type=prim.GetTypeName(),active=bool(prim.IsActive()),
            loaded=bool(prim.IsLoaded()),metadata=compact.native_digest({k:v for k,v in prim.GetAllMetadata().items()
                if k!='instanceable'}),instanceable=bool(prim.IsInstanceable()),attributes=attrs,
            relationships={rel.GetName():dict(targets=list(map(str,rel.GetTargets())),
                metadata=compact.native_digest(rel.GetAllMetadata())) for rel in prim.GetRelationships()},
            world_matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).tolist(),
            physics_material=str(material.GetPath()) if material else None)
    return result


def preview_anonymous_mesh_session_override(stage, plan, geometry_arrays):
    """Author ONLY a NEW scratch Session and check its complete field scope.

    The supplied anonymous fixture, its Root/Session, referenced source
    layers, and all sibling instances remain unchanged. No runtime exports.
    This is not the future live writer and cannot operate inside Kit.
    """
    import copy
    _offline_anonymous_USD_only(stage)
    from pxr import Usd,UsdGeom,Sdf,Gf,Vt
    import hospital_h4_root_difference_readback as fields
    if (plan.get('version')!='H6_SELECTED_INSTANCE_SESSION_GEOMETRY_OPINIONS_OFFLINE_v1'
            or plan.get('new_actual_scene_or_timeline_action_authorized') is not False):
        raise ValueError('Explicit offline Session recipe required')
    items=plan['meshes'];paths=[item['path'] for item in items];roots=plan['roots_to_deinstance']
    if (not 1<=len(paths)<=64 or set(paths)!=set(geometry_arrays) or len(paths)!=len(set(paths))
            or not roots or set(roots)!={item['instance_root'] for item in items}):
        raise ValueError('Exact selected fixture roots and geometry required')
    source_layers={layer.identifier:(layer,layer.ExportToString()) for layer in stage.GetUsedLayers()}
    source_before=_composed_fixture_snapshot(stage)
    root=Sdf.Layer.CreateAnonymous('H6_OFFLINE_source_COPY.usda')
    session=Sdf.Layer.CreateAnonymous('H6_OFFLINE_Session_OPINIONS.usda')
    if (not root.ImportFromString(stage.GetRootLayer().ExportToString())
            or not session.ImportFromString(stage.GetSessionLayer().ExportToString())):
        raise ValueError('Anonymous fixture copies failed')
    scratch=Usd.Stage.Open(root,session);before=_composed_fixture_snapshot(scratch)
    if before!=source_before:raise ValueError('Scratch composed fixture differs before any override')
    for item in items:
        prim=scratch.GetPrimAtPath(item['path']);arrays=geometry_arrays[item['path']]
        if (not prim or not prim.IsInstanceProxy() or prim.GetTypeName()!='Mesh'
                or not np.allclose(before[item['path']]['world_matrix'],item['source_world_matrix'],rtol=0.,atol=1e-11)):
            raise ValueError('Source proxy/world frame mismatch before Session opinions')
        for name,digest in item['native_array_sha256'].items():
            if hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()!=digest:
                raise ValueError('Native arrays changed after plan creation: '+name)
        if set(item['allowed_property_opinions'])!={'points','normals','extent',*(
                ('xformOp:transform:h6_layout','xformOpOrder') if item['additional_leaf_matrix_op'] else ())}:
            raise ValueError('Recipe contains extra property permissions')
    with Usd.EditContext(scratch,session):
        for path in roots:
            prim=scratch.GetPrimAtPath(path)
            if (not path.startswith('/World/Hospital/') or path.count('/')!=3
                    or not prim or not prim.IsInstance() or not prim.IsInstanceable()
                    or not prim.SetInstanceable(False)):
                raise ValueError('Exact selected instance root required; no parent/prototype edit')
        for item in items:
            path=item['path'];prim=scratch.GetPrimAtPath(path);mesh=UsdGeom.Mesh(prim);arrays=geometry_arrays[path]
            if not mesh or prim.IsInstanceProxy() or mesh.GetNormalsInterpolation()!='vertex':
                raise ValueError('Selected source mesh is not an editable vertex-normal scratch prim')
            for attr,name in ((mesh.GetPointsAttr(),'local_points'),(mesh.GetNormalsAttr(),'local_normals'),
                    (mesh.GetExtentAttr(),'local_extent')):
                if not attr.Set(Vt.Vec3fArray.FromNumpy(arrays[name])):raise ValueError('Scratch array opinion failed')
            if item['additional_leaf_matrix_op']:
                xform=UsdGeom.Xformable(prim);name='xformOp:transform:h6_layout'
                if prim.HasAttribute(name):raise ValueError('Reserved leaf matrix op already exists')
                old=np.asarray(xform.GetLocalTransformation())
                parent=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()))
                target=np.asarray(item['target_world_matrix'])
                delta=target@np.linalg.inv(parent)@np.linalg.inv(old)
                xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble,'h6_layout').Set(Gf.Matrix4d(delta.tolist()))
    after=_composed_fixture_snapshot(scratch)
    if set(before)!=set(after):raise ValueError('Composed prim identities changed; no hidden additions/removals')
    normalized=copy.deepcopy(after);changed_arrays=0;changed_frames=0
    for path in roots:normalized[path]['instanceable']=before[path]['instanceable']
    for item in items:
        path=item['path'];arrays=geometry_arrays[path];mesh=UsdGeom.Mesh(scratch.GetPrimAtPath(path))
        for attr,name in ((mesh.GetPointsAttr(),'local_points'),(mesh.GetNormalsAttr(),'local_normals'),
                (mesh.GetExtentAttr(),'local_extent'),(mesh.GetFaceVertexCountsAttr(),'counts'),
                (mesh.GetFaceVertexIndicesAttr(),'indices')):
            if not np.array_equal(np.asarray(attr.Get()),arrays[name]):raise ValueError('Native readback differs: '+name)
        if not np.allclose(after[path]['world_matrix'],item['target_world_matrix'],rtol=0.,atol=1e-11):
            raise ValueError('Additional matrix op did not compose the reviewed target frame')
        for name in ('points','normals','extent'):
            normalized[path]['attributes'][name]['default']=before[path]['attributes'][name]['default'];changed_arrays+=1
        if item['additional_leaf_matrix_op']:
            del normalized[path]['attributes']['xformOp:transform:h6_layout']
            normalized[path]['attributes']['xformOpOrder']['default']=before[path]['attributes']['xformOpOrder']['default']
            normalized[path]['world_matrix']=before[path]['world_matrix'];changed_frames+=1
        if mesh.GetNormalsInterpolation()!='vertex':raise ValueError('Normals interpolation changed')
    if normalized!=before:
        changed=[path for path in before if normalized[path]!=before[path]]
        raise ValueError('Unreviewed composed values/material/UV/topology/world change: '+str(changed[:8]))
    if root.ExportToString()!=stage.GetRootLayer().ExportToString():raise ValueError('Scratch Root changed')
    if (_composed_fixture_snapshot(stage)!=source_before or any(layer.ExportToString()!=text
            for layer,text in source_layers.values())):
        raise ValueError('Original anonymous source/Session/referenced asset changed')
    # Validate the COMPLETE scratch Session, not merely selected composed values.
    before_fields=fields.layer_snapshot(stage.GetSessionLayer().ExportToString())
    final_fields=fields.layer_snapshot(session.ExportToString());new_specs=set(final_fields)-set(before_fields)
    property_permissions={item['path']+'.'+name for item in items for name in item['allowed_property_opinions']}
    for path,spec in final_fields.items():
        if path in before_fields and spec==before_fields[path]:continue
        if spec['spec_type']=='AttributeSpec':
            if path not in property_permissions or set(spec['info'])-{'custom','default','typeName','variability'}:
                raise ValueError('Extra authored Session property/metadata: '+path)
        elif spec['spec_type']=='PrimSpec':
            if (path not in roots and any(k!='specifier' for k in spec['info'])):
                raise ValueError('Unexpected Session ancestor opinion: '+path)
            if path in roots and set(spec['info'])-{'specifier','instanceable'}:
                raise ValueError('Extra selected-root Session metadata')
        else:raise ValueError('Unreviewed Session spec type: '+path)
    if set(before_fields)-set(final_fields):raise ValueError('Existing Session specs removed')
    report=dict(status='ANONYMOUS_SELECTED_INSTANCE_SESSION_OPINIONS_VERIFIED_NOT_LIVE_AUTHORING',
        composed_prim_count=len(before),selected_instance_roots=len(roots),selected_meshes=len(items),
        geometry_array_opinions=changed_arrays,additional_leaf_matrix_ops=changed_frames,
        source_Root_Session_and_all_referenced_anonymous_layers_unchanged=True,
        complete_unrelated_composed_values_and_selected_UV_material_topology_physics_retained=True,
        exact_checked_native_arrays_and_target_world_frames=True,
        complete_Session_opinion_scope_checked=True,Session_new_spec_count=len(new_specs),
        whole_original_Hospital_or_shader_network_NOT_reconstructed=True,
        actual_scene_cooking_support_static_or_drive_NOT_performed=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    return report,session.ExportToString(),scratch


def preview_anonymous_common_environment(stage, geometry_plan, geometry_arrays, condition):
    """Compose the remaining recorded condition on a NEW anonymous copy.

    This is deliberately not a Kit writer. The first door changes only its
    frame (not points/normals); two explicitly named thresholds change only
    collision flags; parked assets retain their original ordered root ops.
    No whole-Hospital or cooked-support claim follows from this fixture.
    """
    import copy
    _offline_anonymous_USD_only(stage)
    from pxr import Usd,UsdGeom,Gf
    import hospital_h4_root_difference_readback as fields
    if (condition.get('version')!='H6_COMMON_ENVIRONMENT_ANONYMOUS_COMPOSITION_v1'
            or condition.get('actual_motion_authorized') is not False):
        raise ValueError('Explicit offline common-environment recipe required')
    first=condition['first_door'];disabled=condition['disabled_thresholds'];rigid=condition['parked_assets']
    if (len(disabled)!=2 or len(set(disabled))!=2 or len(rigid)!=2
            or len({a['root'] for a in rigid})!=2
            or first['path'] in {i['path'] for i in geometry_plan['meshes']}
            or first['root'] in geometry_plan['roots_to_deinstance']):
        raise ValueError('Disjoint first door, two thresholds and two parked roots required')
    source_layers={layer.identifier:(layer,layer.ExportToString()) for layer in stage.GetUsedLayers()}
    report,_,scratch=preview_anonymous_mesh_session_override(stage,geometry_plan,geometry_arrays)
    before=_composed_fixture_snapshot(scratch);session=scratch.GetSessionLayer()
    fields_before=fields.layer_snapshot(session.ExportToString())
    permissions={};deinstance=set();world_targets={}
    def root_of(path):
        if not isinstance(path,str) or not path.startswith('/World/Hospital/') or path.count('/')<4:
            raise ValueError('Exact selected Hospital child path required')
        return '/'.join(path.split('/')[:4])
    def allow(path,names):permissions.setdefault(path,set()).update(names)
    first_root=root_of(first['path'])
    if first_root!=first['root']:raise ValueError('First-door instance ancestry differs')
    if not np.allclose(before[first['path']]['world_matrix'],first['source_world_matrix'],rtol=0.,atol=1e-11):
        raise ValueError('First-door original frame differs')
    target=np.asarray(first['target_world_matrix'],float);source=np.asarray(first['source_world_matrix'],float)
    if (target.shape!=(4,4) or not np.isfinite(target).all() or not np.array_equal(target[:3,3],[0.,0.,0.])
            or target[3,3]!=1. or np.linalg.cond(target[:3,:3])>1e10):
        raise ValueError('Finite affine first-door frame required')
    # A rigid world placement must not stretch the source leaf or change Z.
    delta=np.linalg.inv(source)@target
    if (not np.allclose(delta[:3,:3]@delta[:3,:3].T,np.eye(3),rtol=0.,atol=1e-10)
            or not np.isclose(np.linalg.det(delta[:3,:3]),1.,rtol=0.,atol=1e-10)
            or not np.allclose(delta[:,2],[0.,0.,1.,0.],rtol=0.,atol=1e-10)):
        raise ValueError('First-door placement must be rigid, upright and Z-retaining')
    leaf=scratch.GetPrimAtPath(first['path'])
    if not leaf or not leaf.IsInstanceProxy() or leaf.GetTypeName()!='Mesh':
        raise ValueError('Original first-door mesh instance proxy required')
    # A mesh-level op must not silently carry unreviewed children with it.
    if any(p.startswith(first['path']+'/') for p in before):
        raise ValueError('Unreviewed descendant under first-door leaf')
    deinstance.add(first_root);allow(first['path'],{'xformOp:transform:h6_common','xformOpOrder'})
    for path in disabled:
        root=root_of(path);prim=scratch.GetPrimAtPath(path)
        if (not prim or prim.GetTypeName()!='Mesh'
                or prim.GetAttribute('physics:collisionEnabled').Get() is not True):
            raise ValueError('Only explicit original enabled threshold meshes may be suppressed')
        if prim.IsInstanceProxy():deinstance.add(root)
        allow(path,{'physics:collisionEnabled'})
    with Usd.EditContext(scratch,session):
        for path in sorted(deinstance):
            root=scratch.GetPrimAtPath(path)
            if not root.IsInstance() or not root.SetInstanceable(False):
                raise ValueError('Exact additional instance root required')
        prim=scratch.GetPrimAtPath(first['path']);xform=UsdGeom.Xformable(prim)
        if prim.HasAttribute('xformOp:transform:h6_common'):raise ValueError('Reserved first-door op already exists')
        local=np.asarray(xform.GetLocalTransformation())
        parent=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent()))
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble,'h6_common').Set(
            Gf.Matrix4d((target@np.linalg.inv(parent)@np.linalg.inv(local)).tolist()))
        world_targets[first['path']]=target
        for path in disabled:
            if not scratch.GetPrimAtPath(path).GetAttribute('physics:collisionEnabled').Set(False):
                raise ValueError('Anonymous threshold flag failed')
        for asset in rigid:
            root=asset['root'];prim=scratch.GetPrimAtPath(root);ops=asset['source_root_ordered_ops']
            changes=asset['exact_prospective_Session_default_overrides']
            if (root_of(root+'/child')!=root or not prim or prim.IsInstanceProxy()
                    or not changes or set(changes)-{'xformOp:translate','xformOp:orient'}
                    or not np.allclose(before[root]['world_matrix'],asset['source_root_world_matrix'],rtol=0.,atol=1e-11)):
                raise ValueError('Exact recorded parked-root pose/limited original ops required')
            actual_ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
            if [o.GetOpName() for o in actual_ops]!=[o['name'] for o in ops]:
                raise ValueError('Parked asset op order changed')
            for op,record in zip(actual_ops,ops):
                if (record['inverse'] or record['time_samples'] or op.GetAttr().GetTimeSamples()
                        or fields.semantic_value(op.Get())!=record['value']):
                    raise ValueError('Parked asset source op value/metadata differs')
            for name,value in changes.items():
                if not prim.HasAttribute(name):raise ValueError('No new parked-root ops may be created')
                if name=='xformOp:translate' and value.get('type')=='Vec3d':
                    native=Gf.Vec3d(*map(float,value['values']))
                elif name=='xformOp:orient' and value.get('type')=='Quatd':
                    native=Gf.Quatd(float(value['real']),Gf.Vec3d(*map(float,value['imaginary']['values'])))
                    if not np.isclose(native.GetLength(),1.,rtol=0.,atol=1e-6):
                        raise ValueError('Finite normalized parked-root orientation required')
                else:raise ValueError('Unreviewed parked-root native type')
                if not prim.GetAttribute(name).Set(native):raise ValueError('Anonymous parked-root default failed')
            allow(root,set(changes))
            actual=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            expected=np.asarray(asset['proposed_root_world_matrix'])
            if not np.allclose(actual,expected,rtol=0.,atol=1e-11):
                raise ValueError('Parked-root placement differs from recorded rigid recipe')
            delta=np.linalg.inv(np.asarray(before[root]['world_matrix']))@expected
            for path,row in before.items():
                if path==root or path.startswith(root+'/'):world_targets[path]=np.asarray(row['world_matrix'])@delta
    after=_composed_fixture_snapshot(scratch);normalized=copy.deepcopy(after)
    if set(before)!=set(after):raise ValueError('Common condition changed composed prim identities')
    for path in deinstance:normalized[path]['instanceable']=before[path]['instanceable']
    for path,expected in world_targets.items():
        if not np.allclose(after[path]['world_matrix'],expected,rtol=0.,atol=1e-11):
            raise ValueError('Common condition world frame differs: '+path)
        normalized[path]['world_matrix']=before[path]['world_matrix']
    for path,names in permissions.items():
        for name in names:
            if name not in before[path]['attributes']:del normalized[path]['attributes'][name]
            else:normalized[path]['attributes'][name]['default']=before[path]['attributes'][name]['default']
    if normalized!=before:raise ValueError('Unreviewed common-condition attribute, UV, material or frame change')
    fields_after=fields.layer_snapshot(session.ExportToString())
    for path,spec in fields_after.items():
        if path in fields_before and fields_before[path]==spec:continue
        if spec['spec_type']=='AttributeSpec':
            owner,name=path.rsplit('.',1)
            if (name not in permissions.get(owner,set())
                    or set(spec['info'])-{'custom','default','typeName','variability'}):
                raise ValueError('Unexpected common-environment Session property: '+path)
        elif spec['spec_type']=='PrimSpec':
            allowed={'specifier','instanceable'} if path in deinstance else {'specifier'}
            if set(spec['info'])-allowed:raise ValueError('Unexpected common-environment Session metadata')
        else:raise ValueError('Unexpected common-environment Session spec')
    if set(fields_before)-set(fields_after):raise ValueError('Existing geometry Session opinions removed')
    if any(layer.ExportToString()!=text for layer,text in source_layers.values()):
        raise ValueError('Input anonymous source layers changed during complete composition')
    destination_counts={key:report.pop(key) for key in ('selected_instance_roots','selected_meshes',
        'geometry_array_opinions','additional_leaf_matrix_ops','Session_new_spec_count')}
    initial_fields=fields.layer_snapshot(stage.GetSessionLayer().ExportToString())
    report.update(status='ANONYMOUS_COMBINED_H6_CONDITION_COMPOSED_NOT_LIVE_ENVIRONMENT_ADMISSION',
        destination_geometry_phase_counts=destination_counts,
        total_selected_instance_roots=len(set(geometry_plan['roots_to_deinstance'])|deinstance),
        total_Session_new_spec_count=len(set(fields_after)-set(initial_fields)),
        first_door_original_points_normals_UV_and_Z_retained=True,
        threshold_collision_flags_disabled_exactly=list(disabled),
        parked_root_original_order_scale_and_geometry_retained=True,
        no_second_destination_root_rotation_applied=True,
        additional_selected_instance_roots=sorted(deinstance),
        all_composed_fields_checked_outside_explicit_condition_defaults=True,
        actual_Hospital_shader_network_cooked_support_static_and_drive_NOT_verified=True)
    return report,session.ExportToString(),scratch


def reconstructed_anonymous_mesh_instances(plan, original_rows):
    """Offline composition fixture using authenticated geometry and observed UVs.

    Reconstruct only the selected meshes' values/frames/binding identities.
    The shader networks and the rest of the Hospital are NOT reconstructed.
    Additional unselected instances share every fixture source asset to test
    that the Session opinions do not modify a prototype or sibling instance.
    """
    import sys
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('Source instance reconstruction is OFFLINE ONLY')
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf,Vt
    import hospital_compact_usd_fingerprint as compact
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/Hospital')
    assets=[];maximum_frame_error=0.;checked_attributes=0
    def decode(value):
        if value is None or type(value) in (str,bool,int,float):return value
        kind=value.get('type')
        if kind=='TokenArray':return Vt.TokenArray(value['values'])
        numeric=semantic_array(value,'fixture '+str(kind))
        if kind=='Vec3f':return Gf.Vec3f(*map(float,numeric))
        if kind in ('Vec3fArray','Vec2fArray','FloatArray'):
            return getattr(Vt,kind).FromNumpy(numeric.astype(np.float32))
        if kind=='IntArray':return Vt.IntArray.FromNumpy(numeric.astype(np.int32))
        raise ValueError('Unreviewed source fixture value type: '+str(kind))
    for index,root_path in enumerate(plan['roots_to_deinstance']):
        asset=Usd.Stage.CreateInMemory();asset_root=UsdGeom.Xform.Define(asset,'/Asset')
        asset.SetDefaultPrim(asset_root.GetPrim());assets.append(asset);instance_matrix=None
        selected=[item for item in plan['meshes'] if item['instance_root']==root_path]
        for item in selected:
            path=item['path'];row=original_rows[path];relative='/Asset'+path[len(root_path):]
            mesh=UsdGeom.Mesh.Define(asset,relative);prim=mesh.GetPrim()
            pvs={v['name']:v for v in item['observed_primvars']}
            for name,attr_row in row['attributes'].items():
                if attr_row['time_samples']:raise ValueError('Static source fixture only; time samples not invented')
                attr=prim.GetAttribute(name);value=attr_row['value']
                if not attr:
                    if name.startswith('primvars:'):
                        pv=pvs.get(name.split(':',1)[1])
                        if not pv or pv['indexed'] or pv['attribute_path']!=path+'.'+name:
                            raise ValueError('Exact observed local nonindexed primvar required for source reconstruction')
                        attr=prim.CreateAttribute(name,Sdf.ValueTypeNames.Find(pv['type_name']),custom=False)
                    else:
                        types={'physics:collisionEnabled':Sdf.ValueTypeNames.Bool,
                            'physics:approximation':Sdf.ValueTypeNames.Token,
                            'xformOp:scale':Sdf.ValueTypeNames.Float3,'xformOpOrder':Sdf.ValueTypeNames.TokenArray}
                        if name not in types:raise ValueError('Unknown non-schema source fixture attribute: '+name)
                        attr=prim.CreateAttribute(name,types[name],custom=False)
                if value is not None and not attr.Set(decode(value)):raise ValueError('Fixture source value failed')
                if compact.native_digest(attr.Get())!=compact.semantic_json_digest(value):
                    raise ValueError('Reconstructed fixture source value differs: '+path+'.'+name)
                checked_attributes+=1
            mesh.SetNormalsInterpolation(item['normal_interpolation'])
            for name,pv in pvs.items():
                attr=prim.GetAttribute('primvars:'+name)
                if not attr:raise ValueError('Missing original primvar value; no inherited value invented')
                variable=UsdGeom.Primvar(attr);variable.SetInterpolation(pv['interpolation'])
                variable.SetElementSize(pv['element_size'])
                if (compact.native_digest(variable.Get())!=pv['value_sha256']
                        or bool(variable.IsIndexed())!=pv['indexed']):
                    raise ValueError('Reconstructed source UV/color differs from actual GET')
            UsdShade.MaterialBindingAPI.Apply(prim)
            for name,targets in row['relationships'].items():
                mapped=[]
                for target in targets:
                    if not target.startswith(root_path+'/'):
                        raise ValueError('Fixture will not invent an external binding target')
                    mapped.append('/Asset'+target[len(root_path):])
                    if name.startswith('material:binding'):UsdShade.Material.Define(asset,mapped[-1])
                relationship=prim.GetRelationship(name)
                if not relationship:relationship=prim.CreateRelationship(name,custom=False)
                relationship.SetTargets(mapped)
            local=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            root_matrix=np.linalg.inv(local)@np.asarray(item['source_world_matrix'])
            if instance_matrix is not None and not np.allclose(instance_matrix,root_matrix,rtol=0.,atol=1e-11):
                raise ValueError('Shared instance sections require incompatible source ancestor frames')
            instance_matrix=root_matrix
        for path,offset in ((root_path,0.),('/World/Hospital/UnselectedControl'+str(index).zfill(2),10.)):
            instance=UsdGeom.Xform.Define(stage,path)
            instance.GetPrim().GetReferences().AddReference(asset.GetRootLayer().identifier)
            matrix=instance_matrix.copy();matrix[3,0]+=offset
            instance.AddTransformOp().Set(Gf.Matrix4d(matrix.tolist()));instance.GetPrim().SetInstanceable(True)
        cache=UsdGeom.XformCache()
        for item in selected:
            path=item['path'];prim=stage.GetPrimAtPath(path)
            error=float(np.max(np.abs(np.asarray(cache.GetLocalToWorldTransform(prim))-item['source_world_matrix'])))
            maximum_frame_error=max(maximum_frame_error,error)
            if error>1e-11 or not prim.IsInstanceProxy():raise ValueError('Source fixture frame/proxy composition failed')
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if (str(material.GetPath()) if material else None)!=original_rows[path]['bound_physics_material']:
                raise ValueError('Original material binding identity did not reconstruct')
    UsdGeom.Xform.Define(stage,'/World/UnrelatedControl').AddTranslateOp().Set(Gf.Vec3d(1.,2.,3.))
    return stage,assets,dict(selected_source_attribute_values_verified=checked_attributes,
        maximum_source_fixture_frame_error=maximum_frame_error,
        source_assets_and_sibling_controls_each=len(assets),
        original_shader_network_and_complete_Hospital_NOT_reconstructed=True,
        actual_Hospital_USD_or_Kit_NOT_opened=True)


def export_mesh_session_override_review(native_review_path, metadata_result_path, *, output=None):
    """Append-only anonymous-USD test of the actually observed instance layout."""
    import time,zipfile,sys,os
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('Session export review is OFFLINE ONLY; not an Isaac command')
    from pxr import Usd,UsdGeom,Sdf
    from hospital_geometry_tools import mesh_mask
    started=time.monotonic();native_review_path=Path(native_review_path).resolve()
    metadata_result_path=Path(metadata_result_path).resolve();folder=native_review_path.parent
    native=json.loads(native_review_path.read_bytes())
    metadata=json.loads(metadata_result_path.read_bytes());verified_path=metadata_result_path.with_name('actual_GET_verification.json')
    verified=json.loads(verified_path.read_bytes())
    if verified['source_result_sha256']!=hashlib.sha256(metadata_result_path.read_bytes()).hexdigest():
        raise ValueError('Independently verified actual source metadata bytes required')
    selected_path=folder/'authenticated_selected_source_rows.json';selected=json.loads(selected_path.read_bytes())
    arrays_path=folder/'native_local_geometry_arrays.npz';values=np.load(arrays_path,allow_pickle=False)
    arrays={item['path']:{key:values[item['array_prefix']+'_'+key].copy() for key in
        ('local_points','local_normals','local_extent','counts','indices','world_matrix','rounded_world_points_m')}
        for item in native['meshes']}
    plan=mesh_session_override_plan(metadata,selected['original_mesh_rows'],arrays)
    if len(plan['meshes'])!=8 or len(plan['roots_to_deinstance'])!=7:
        raise ValueError('H6 adopted original8mesh/7instance-root scope required')
    inputs=[native_review_path,arrays_path,selected_path,metadata_result_path,verified_path,
        metadata_result_path.with_name('GET_contract.json'),metadata_result_path.with_name('executed_GET_sources.zip'),
        Path(native['source_geometry']),Path(__file__),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_geometry_tools.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_compact_usd_fingerprint.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_root_difference_readback.py')]
    pins={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    output=Path(output).resolve() if output is not None else folder/'session_instance_composition_r01'
    target=output/'session_override_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if saved['input_sha256']!=pins or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h
                for p,h in saved['output_sha256'].items()):
            raise ValueError('Existing offline Session review changed; no overwrite or duplicate execution')
        return target
    # Completed GET used an earlier shared-reader version. Verify that frozen
    # source from its archive; never rewrite its original receipt or reissue it.
    archive_path=metadata_result_path.with_name('executed_GET_sources.zip')
    if pins[str(archive_path)]!=verified['executed_GET_sources_sha256']:
        raise ValueError('Completed actual GET source archive changed')
    with zipfile.ZipFile(archive_path) as archive:
        for name,digest in metadata['input_sha256'].items():
            historical=archive.read(str(Path(name).relative_to(ROOT))) if Path(name)==Path(__file__) else Path(name).read_bytes()
            if hashlib.sha256(historical).hexdigest()!=digest:raise ValueError('Completed GET dependency changed: '+name)
    stage,assets,fixture=reconstructed_anonymous_mesh_instances(plan,selected['original_mesh_rows'])
    report,session_text,scratch=preview_anonymous_mesh_session_override(stage,plan,arrays)
    layer=Sdf.Layer.CreateAnonymous('H6_OFFLINE_PATCH_REIMPORT.usda')
    if not layer.ImportFromString(session_text):raise ValueError('Portable Session opinions did not reimport')
    replay=Usd.Stage.Open(stage.GetRootLayer(),layer)
    if _composed_fixture_snapshot(replay)!=_composed_fixture_snapshot(scratch):
        raise ValueError('Exported Session opinion replay differs from checked scratch composition')
    geometry=json.loads(Path(native['source_geometry']).read_bytes());raster=geometry['raster']
    masks=[];cache=UsdGeom.XformCache()
    for item in native['meshes']:
        path=item['path'];mesh=UsdGeom.Mesh(replay.GetPrimAtPath(path));matrix=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()))
        local=np.asarray(mesh.GetPointsAttr().Get(),float);world=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
        old=arrays[path]['rounded_world_points_m'];error=float(np.max(np.abs(world-old)))
        if error>1e-11:raise ValueError('Composed exported recipe differs from the exact reviewed world geometry')
        expected=values[item['array_prefix']+'_planning_mask']
        mask=mesh_mask(dict(world_points_m=world,face_vertex_counts=mesh.GetFaceVertexCountsAttr().Get(),
            face_vertex_indices=mesh.GetFaceVertexIndicesAttr().Get()),raster['origin_xy_m'],raster['resolution_m'],
            expected.shape,*geometry['obstacle_height_slab_m'])
        if not np.array_equal(mask,expected):raise ValueError('Composed Session geometry raster differs from adopted planning mask')
        masks.append(dict(path=path,world_error_m=error,composed_mask_identical_to_adopted_planning=True))
    output.mkdir(parents=True,exist_ok=True);patch_path=output/'anonymous_Session_opinions.usda'
    evidence.write_exclusive(output/'session_opinion_plan.json',plan)
    with patch_path.open('x',encoding='utf-8') as stream:
        stream.write(session_text);stream.flush();os.fsync(stream.fileno())
    archive_path=output/'executed_offline_sources.zip'
    with zipfile.ZipFile(archive_path,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in inputs:
            if path.suffix=='.py':archive.write(path,str(path.relative_to(ROOT)))
    outputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (output/'session_opinion_plan.json',patch_path,archive_path)}
    report.update(fixture=fixture,composed_mesh_raster_checks=masks,
        source_metadata_result=str(metadata_result_path),source_metadata_result_sha256=pins[str(metadata_result_path)],
        historical_GET_frozen_reader_NOT_overwritten_or_rerun=True,
        exported_Session_reimport_composes_identically=True,
        all8_composed_mesh_masks_equal_adopted_planning=True,
        native_SDK='.'.join(map(str,Usd.GetVersion())),elapsed_s=time.monotonic()-started,
        input_sha256=pins,output_sha256=outputs)
    evidence.write_exclusive(target,report)
    return target


def export_common_environment_session_review(native_review_path, *, output):
    """Verify the combined adopted H6 condition without opening a real USD.

    Reuse the completed destination-eight patch and disjoint actual two-mesh
    GET. Replay parked-root defaults from their earlier actual-frame review;
    never repeat a live GET, apply the old destination rotation twice, or
    replace an existing physical trial. No runtime command is exported.
    """
    import copy,os,sys,time,zipfile
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('Common-environment export is OFFLINE ONLY')
    from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Gf,Vt
    from hospital_geometry_tools import mesh_mask
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inputs=set();started=time.monotonic()
    def read(path):
        path=Path(path).resolve();inputs.add(path);return json.loads(path.read_bytes())
    native_path=Path(native_review_path).resolve();folder=native_path.parent;native=read(native_path)
    geometry_path=Path(native['source_geometry']);geometry=read(geometry_path)
    config_path=_ijcas_path(ROOT, 'studies/IJCAS/config/hospital_h6_common_environment.json');common=read(config_path)
    for name,digest in common['immutable_source_sha256'].items():
        path=_ijcas_path(ROOT, name)
        if sha(path)!=digest:raise ValueError('Adopted common-condition source changed: '+name)
        inputs.add(path)
    destination=folder/'session_instance_composition_r02';prior=read(destination/'session_override_review.json')
    for name,digest in prior['output_sha256'].items():
        if sha(name)!=digest:raise ValueError('Completed destination Session artifact changed')
        inputs.add(Path(name))
    plan=read(destination/'session_opinion_plan.json');selected=read(folder/'authenticated_selected_source_rows.json')
    original_rows=selected['original_mesh_rows'];fixture_plan=copy.deepcopy(plan)
    values_path=folder/'native_local_geometry_arrays.npz';inputs.add(values_path)
    with np.load(values_path,allow_pickle=False) as values:
        arrays={item['path']:{key:values[item['array_prefix']+'_'+key].copy() for key in
            ('local_points','local_normals','local_extent','counts','indices','world_matrix','rounded_world_points_m')}
            for item in native['meshes']}
    remaining=folder/'remaining_common_environment_source_GET_r01'
    meta_path=remaining/'live_source_metadata.json';metadata=read(meta_path)
    proof=read(remaining/'actual_GET_verification.json');contract=read(remaining/'GET_contract.json')
    archive=remaining/'executed_GET_sources.zip';inputs.add(archive)
    if (sha(meta_path)!=proof['source_result_sha256'] or sha(archive)!=proof['executed_GET_sources_sha256']
            or sha(remaining/'GET_contract.json')!=proof['contract_sha256']
            or metadata['input_sha256']!=contract['input_sha256']
            or metadata['error'] is not None or not all(v is True for v in metadata['checks'].values())
            or metadata['actual_tensor_before']!=metadata['actual_tensor_after']):
        raise ValueError('Independently verified remaining actual source GET required')
    # The completed GET's executed helper bytes, not a subsequently extended
    # offline helper, are its historical code provenance.
    with zipfile.ZipFile(archive) as frozen:
        for name,digest in contract['input_sha256'].items():
            path=Path(name);relative=str(path.relative_to(ROOT))
            raw=frozen.read(relative) if relative in frozen.namelist() else path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Completed GET frozen input changed: '+name)
    extra=read(remaining/'authenticated_selected_source_rows.json')['original_mesh_rows']
    if set(extra)&set(original_rows) or len(extra)!=2:raise ValueError('Exactly two disjoint original meshes required')
    original_rows={**original_rows,**extra}
    for item in metadata['meshes']:
        roots=[a['path'] for a in item['mesh_and_ancestor_composition'] if a['instanceable'] and a['instance']]
        if len(roots)!=1 or item['normals_interpolation']!='vertex':
            raise ValueError('Observed remaining instance root and vertex-normal metadata required')
        fixture_plan['meshes'].append(dict(path=item['path'],instance_root=roots[0],
            source_world_matrix=semantic_array(extra[item['path']]['world_matrix'],'source world').tolist(),
            normal_interpolation='vertex',observed_primvars=item['inherited_primvars']))
        fixture_plan['roots_to_deinstance'].append(roots[0])
    fixture_plan['roots_to_deinstance']=sorted(set(fixture_plan['roots_to_deinstance']))
    stage,assets,fixture=reconstructed_anonymous_mesh_instances(fixture_plan,original_rows)
    rigid_path=next(Path(p) for p in geometry['input_sha256'] if p.endswith('actual_frames_rigid_override_review.json'))
    if sha(rigid_path)!=geometry['input_sha256'][str(rigid_path)]:raise ValueError('Recorded asset-frame recipe changed')
    rigid=read(rigid_path);parked=[copy.deepcopy(a) for a in rigid['proposed_selected_assets'] if a['root'] in
        ('/World/Hospital/SM_SupplyCart_01e14','/World/Hospital/SM_Chair_01a14')]
    if len(parked)!=2:raise ValueError('Both adopted parked medical assets required')
    for asset in parked:
        asset['source_root_world_matrix']=(np.asarray(asset['source_root_local_matrix'])@
            np.asarray(asset['source_parent_world_matrix'])).tolist()
    manifest=native['source_receipt']['original_manifest'];inputs.add(Path(manifest))
    parked_rows,receipt=read_authenticated_mesh_rows(manifest,[a['collider'] for a in parked],
        expected_manifest_sha256=native['source_receipt']['manifest_sha256'])
    # Reconstruct recorded root ops and geometry for rigid-placement tests.
    # Cart/chair normal interpolation/primvar metadata was NOT observed in
    # the new GET; do not invent it or claim those shader networks complete.
    parked_geometry={}
    for index,asset in enumerate(parked):
        if not np.array_equal(asset['source_parent_world_matrix'],np.eye(4)):
            raise ValueError('This original parked-root fixture requires its verified identity parent')
        root=UsdGeom.Xform.Define(stage,asset['root'])
        for record in asset['source_root_ordered_ops']:
            value=record['value'];name=record['name']
            if record['inverse'] or record['time_samples']:raise ValueError('Original static root ops required')
            if name=='xformOp:translate':root.AddTranslateOp().Set(Gf.Vec3d(*map(float,value['values'])))
            elif name=='xformOp:orient':root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(float(value['real']),Gf.Vec3d(*map(float,value['imaginary']['values']))))
            elif name=='xformOp:scale':root.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float,value['values'])))
            else:raise ValueError('Unknown parked-root op')
        row=parked_rows[asset['collider']];attrs=row['attributes'];mesh=UsdGeom.Mesh.Define(stage,asset['collider'])
        for attr,name,kind in ((mesh.GetPointsAttr(),'points',Vt.Vec3fArray),
                (mesh.GetFaceVertexCountsAttr(),'faceVertexCounts',Vt.IntArray),
                (mesh.GetFaceVertexIndicesAttr(),'faceVertexIndices',Vt.IntArray)):
            value=semantic_array(attrs[name]['value'],name)
            attr.Set(kind.FromNumpy(value.astype(np.float32 if name=='points' else np.int32)))
        for op in asset['source_child_ordered_ops']:
            if op['name']!='xformOp:scale' or op['inverse'] or op['time_samples']:
                raise ValueError('Exact original child-unit scale required')
            UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3f(*map(float,op['value']['values'])))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        world=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
        if not np.allclose(world,semantic_array(row['world_matrix'],'parked world'),rtol=0.,atol=1e-11):
            raise ValueError('Original parked source frame did not reconstruct')
        local=np.asarray(mesh.GetPointsAttr().Get(),float)
        source_world=(np.c_[local,np.ones(len(local))]@world)[:,:3]
        transform=np.asarray(asset['proposed_world_rigid_transform_row_vector'])
        parked_geometry[asset['collider']]=(np.c_[source_world,np.ones(len(local))]@transform)[:,:3]
    historical_path=next(_ijcas_path(ROOT, p) for p in common['immutable_source_sha256'] if p.endswith('offline_condition_proposal.json'))
    historical=read(historical_path);spec=historical['original_door_source_review']['original_hinge_specification']
    first=common['definition']['door'];source=semantic_array(original_rows[first['collider']]['world_matrix'],'first-door world')
    rotation=np.asarray(spec['rotation_matrix']);pivot=np.asarray(spec['hinge_world_origin_m'])
    rigid_world=np.eye(4);rigid_world[:3,:3]=rotation.T;rigid_world[3,:3]=pivot-pivot@rotation.T
    first_target=source@rigid_world
    variant=read(geometry['task_definition'])
    task_path=Path(variant['original_task_definition'])
    if sha(task_path)!=variant['original_task_definition_sha256']:raise ValueError('Adopted base task changed')
    task=read(task_path);thresholds=task['common_environment']['destination_flush_threshold']['collision_paths']
    condition=dict(version='H6_COMMON_ENVIRONMENT_ANONYMOUS_COMPOSITION_v1',actual_motion_authorized=False,
        first_door=dict(root=first['root'],path=first['collider'],source_world_matrix=source.tolist(),
            target_world_matrix=first_target.tolist(),original_recorded_H3_specification=spec),
        disabled_thresholds=common['definition']['threshold']['collision_paths']+thresholds,
        parked_assets=parked)
    report,text,scratch=preview_anonymous_common_environment(stage,plan,arrays,condition)
    layer=Sdf.Layer.CreateAnonymous('H6_COMMON_REPLAY.usda')
    if not layer.ImportFromString(text):raise ValueError('Combined Session export did not reimport')
    replay=Usd.Stage.Open(stage.GetRootLayer(),layer)
    if _composed_fixture_snapshot(replay)!=_composed_fixture_snapshot(scratch):raise ValueError('Combined Session replay differs')
    grid_path=Path(geometry['occupancy_grid']);inputs.add(grid_path)
    if sha(grid_path)!=geometry['output_sha256'][str(grid_path)]:raise ValueError('Adopted planning grid changed')
    with np.load(grid_path,allow_pickle=False) as grid:
        expected=grid['occupied'].copy();recomposed=grid['unchanged_other188_colliders'].copy()
        for key in ('unchanged_fixed_support_ROI','unchanged_medical_keepouts','unchanged_chair_bay'):
            recomposed|=grid[key]
        raster=geometry['raster'];checks=[];cache=UsdGeom.XformCache()
        for path in [i['path'] for i in plan['meshes']]+list(extra)+list(parked_geometry):
            mesh=UsdGeom.Mesh(replay.GetPrimAtPath(path));local=np.asarray(mesh.GetPointsAttr().Get(),float)
            matrix=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()));world=(np.c_[local,np.ones(len(local))]@matrix)[:,:3]
            target=(arrays[path]['rounded_world_points_m'] if path in arrays else parked_geometry.get(path))
            if path==first['collider']:
                local_source=semantic_array(extra[path]['attributes']['points']['value'],'source points')
                target=(np.c_[local_source,np.ones(len(local_source))]@first_target)[:,:3]
            elif path in extra:
                local_source=semantic_array(extra[path]['attributes']['points']['value'],'source points')
                target=(np.c_[local_source,np.ones(len(local_source))]@semantic_array(extra[path]['world_matrix'],'source world'))[:,:3]
            error=float(np.max(np.abs(world-target)))
            if error>1e-11:raise ValueError('Composed common mesh differs from recorded recipe: '+path)
            disabled=path in condition['disabled_thresholds']
            if not disabled:
                recomposed|=mesh_mask(dict(world_points_m=world,face_vertex_counts=mesh.GetFaceVertexCountsAttr().Get(),
                    face_vertex_indices=mesh.GetFaceVertexIndicesAttr().Get()),raster['origin_xy_m'],raster['resolution_m'],
                    expected.shape,*geometry['obstacle_height_slab_m'])
            checks.append(dict(path=path,world_recipe_error_m=error,collision_suppressed=disabled))
        if not np.array_equal(recomposed,expected):
            raise ValueError('Complete composed common condition differs from adopted planning map: '+str(int(np.count_nonzero(recomposed!=expected))))
    inputs.update((Path(__file__),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')))
    pins={str(path):sha(path) for path in sorted(inputs)}
    output=Path(output).resolve()
    if output.exists():raise ValueError('Existing combined review preserved; no overwrite')
    output.mkdir(parents=True);patch=output/'anonymous_common_Session_opinions.usda'
    with patch.open('x',encoding='utf-8') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(output/'common_condition_recipe.json',condition)
    frozen_path=output/'executed_offline_sources.zip'
    with zipfile.ZipFile(frozen_path,'x',zipfile.ZIP_DEFLATED) as frozen:
        for path in inputs:
            if path.suffix=='.py':frozen.write(path,str(path.relative_to(ROOT)))
    report.update(source10_fixture=fixture,parked2_geometry_and_frame_fixture_ONLY_not_shader_or_unobserved_metadata=True,
        selectively_authenticated_parked_rows=receipt,combined_meshes_verified=checks,
        complete_composed_12_contributions_plus_retained188_and_fixed_keepouts_bitwise_equal_adopted_grid=True,
        exported_combined_Session_reimport_identical=True,native_SDK='.'.join(map(str,Usd.GetVersion())),
        elapsed_s=time.monotonic()-started,input_sha256=pins,
        output_sha256={str(path):sha(path) for path in (patch,output/'common_condition_recipe.json',frozen_path)})
    target=output/'common_environment_review.json';evidence.write_exclusive(target,report)
    return target


PREPARATION_VERSION='H6_REVIEWED_COMMON_ENVIRONMENT_ONE_STOP_SOURCE_DISPLAY_ONLY_v1'


def common_environment_descriptor(stage,path):
    """Compact complete values for ONE named collider/material, no authoring."""
    from pxr import UsdGeom,UsdShade
    import hospital_compact_usd_fingerprint as compact
    prim=stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive() or not prim.IsLoaded():raise ValueError('Original active environment prim missing: '+path)
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    return dict(active=bool(prim.IsActive()),type=prim.GetTypeName(),
        attributes={a.GetName():dict(default_sha256=compact.native_digest(a.Get()),
            samples=[[t,compact.native_digest(a.Get(t))] for t in a.GetTimeSamples()]) for a in prim.GetAttributes()},
        relationships={r.GetName():list(map(str,r.GetTargets())) for r in prim.GetRelationships()},
        world_matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).tolist(),
        bound_physics_material=str(material.GetPath()) if material else None)


def verify_common_environment_descriptors(stage,expected):
    """Exact value/identity checks; only native affine arithmetic has 1e-11m tolerance.

    This numerical matrix composition tolerance is NOT a collision, support,
    tracking or terminal-band waiver. The observed final whole-row digest is
    separately frozen, so later trials must retain their actual boundary.
    """
    for path,target in expected.items():
        actual=common_environment_descriptor(stage,path)
        matrix=actual.pop('world_matrix');goal=target['world_matrix']
        if (actual!={k:v for k,v in target.items() if k!='world_matrix'}
                or not np.allclose(matrix,goal,rtol=0.,atol=1e-11)):
            raise ValueError('H6 environment values/frame differ from explicit recipe: '+path)
    return True


def checked_common_Session_patch(text):
    """Bounded property-only patch; no references, whole-layer import or broad defaults."""
    from pxr import Sdf
    if not isinstance(text,str) or len(text.encode())>512*1024:raise ValueError('Bounded reviewed Session opinions required')
    layer=Sdf.Layer.CreateAnonymous('H6_CHECKED_PROPERTY_OPINIONS.usda')
    if not layer.ImportFromString(text):raise ValueError('Invalid reviewed Session opinion syntax')
    paths=[];layer.Traverse('/',lambda path:paths.append(path));attributes={};roots=[]
    names={'points','normals','extent','xformOp:transform:h6_layout','xformOp:transform:h6_common',
        'xformOpOrder','physics:collisionEnabled','xformOp:orient','xformOp:translate'}
    for path in paths:
        if path==Sdf.Path.absoluteRootPath:continue
        spec=layer.GetObjectAtPath(path)
        if not str(path).startswith('/World/Hospital') or not (str(path)=='/World/Hospital' or str(path).startswith('/World/Hospital/')):
            if str(path)=='/World' and set(spec.ListInfoKeys())=={'specifier'}:continue
            raise ValueError('Environment patch escapes exact Hospital subtree')
        if isinstance(spec,Sdf.AttributeSpec):
            if (path.name not in names or set(spec.ListInfoKeys())-{'custom','default','typeName','variability'}
                    or not spec.HasInfo('default') or spec.custom):raise ValueError('Unreviewed Session property or metadata')
            if path.name=='physics:collisionEnabled' and spec.default is not False:raise ValueError('Only exact threshold disable flags allowed')
            attributes[str(path)]=spec
        elif isinstance(spec,Sdf.PrimSpec):
            if set(spec.ListInfoKeys())-{'specifier','instanceable'} or spec.specifier!=Sdf.SpecifierOver:
                raise ValueError('Only property ancestor overs/selected instance flags allowed')
            if spec.HasInfo('instanceable'):
                if str(path).count('/')!=3 or spec.instanceable is not False:raise ValueError('Only exact direct selected roots may be deinstanced')
                roots.append(str(path))
        else:raise ValueError('No relationships, variants, references or composition edits in patch')
    if len(attributes)!=33 or len(roots)!=9 or len(paths)-1!=64:
        raise ValueError('Exact independently verified33 properties/9 roots/64 specs required')
    return layer,attributes,sorted(roots)


def apply_common_Session_patch(stage,text,expected_before,expected_after,*,authorized=False):
    """Explicit caller-approved Session properties ONLY. No timeline operation.

    The offline/actual orchestration owns authority. This primitive performs
    the same bounded data transaction on anonymous fixtures during tests.
    On failure it leaves exact evidence; no rollback, replacement or retry.
    """
    if authorized is not True:raise ValueError('Explicit preparation-only environment consent required before SDK access')
    from pxr import Usd
    import hospital_h4_root_difference_readback as fields
    layer,attributes,roots=checked_common_Session_patch(text)
    verify_common_environment_descriptors(stage,expected_before)
    before=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    for path in roots:
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsInstance() or not prim.IsInstanceable():raise ValueError('Exact original instance root required: '+path)
    for path,spec in attributes.items():
        prim=stage.GetPrimAtPath(spec.path.GetPrimPath())
        if not prim:raise ValueError('Original selected mesh/root missing')
        if path.endswith(('.xformOp:transform:h6_layout','.xformOp:transform:h6_common')) and prim.HasAttribute(spec.name):
            raise ValueError('Environment already applied/reserved; no second application')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for path in roots:
            if not stage.GetPrimAtPath(path).SetInstanceable(False):raise RuntimeError('Selected Session instance opinion failed')
        for path,spec in attributes.items():
            prim=stage.GetPrimAtPath(spec.path.GetPrimPath());attr=prim.GetAttribute(spec.name)
            if not attr:attr=prim.CreateAttribute(spec.name,spec.typeName,custom=False,variability=spec.variability)
            if str(attr.GetTypeName())!=str(spec.typeName) or not attr.Set(spec.default):
                raise RuntimeError('Exact typed Session default failed: '+path)
    verify_common_environment_descriptors(stage,expected_after)
    after=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    if before['Root']!=after['Root']:raise RuntimeError('Original Root changed during Session-only environment transaction')
    changes=layer_field_differences_for_common(before['Session'],after['Session'])
    before_specs,after_specs=map(fields.layer_snapshot,(before['Session'],after['Session']))
    for path in set(before_specs)|set(after_specs):
        a,b=before_specs.get(path),after_specs.get(path)
        if a==b:continue
        if path in attributes:
            if not b or (a and {k:v for k,v in a['info'].items() if k!='default'}!=
                    {k:v for k,v in b['info'].items() if k!='default'}):raise RuntimeError('Existing property metadata changed')
        elif path in roots:
            if not b or set(b['info'])-({'specifier','instanceable'}|set((a or {}).get('info',{}))):
                raise RuntimeError('Extra instance-root opinion')
            for key,value in (a or {}).get('info',{}).items():
                if key!='instanceable' and b['info'].get(key)!=value:raise RuntimeError('Root metadata changed')
        elif a is None and b and b['spec_type']=='PrimSpec' and set(b['info'])=={'specifier'}:
                if not any(p.startswith(path+'/') or p.startswith(path+'.') for p in attributes):raise RuntimeError('Unrelated ancestor over created')
        else:raise RuntimeError('Unrelated Session spec changed: '+path)
    return dict(exact_Session_changes=changes,Root_unchanged=True,property_opinions=33,selected_instance_roots=9,
        source_values_and_target_native_frames_verified=True,no_timeline_or_cooking_command=True)


def layer_field_differences_for_common(before,after):
    import hospital_h4_subfloor_compact_path_launch as shared
    return shared.layer_field_differences(before,after)


def source_composition_signature(stage,observed):
    """Current metadata recheck at preparation boundary, NOT a duplicate GET trial."""
    from pxr import UsdGeom
    import hospital_compact_usd_fingerprint as compact
    if len(observed)!=10 or len({r['path'] for r in observed})!=10:
        raise ValueError('Exactly ten independently observed original source meshes required')
    for record in observed:
        prim=stage.GetPrimAtPath(record['path']);mesh=UsdGeom.Mesh(prim)
        if (not mesh or mesh.GetNormalsInterpolation()!=record['normals_interpolation']
                or mesh.GetNormalsAttr().GetMetadata('interpolation')!=record['normals_authored_interpolation_metadata']):
            raise ValueError('Observed original source-normal metadata changed')
        current=prim
        for ancestor in record['mesh_and_ancestor_composition']:
            values=dict(path=str(current.GetPath()),type_name=current.GetTypeName(),active=bool(current.IsActive()),
                loaded=bool(current.IsLoaded()),instanceable=bool(current.IsInstanceable()),instance=bool(current.IsInstance()),
                instance_proxy=bool(current.IsInstanceProxy()))
            if any(values[k]!=ancestor[k] for k in values):raise ValueError('Observed selected source composition changed')
            current=current.GetParent()
        variables=[]
        for pv in UsdGeom.PrimvarsAPI(prim).FindPrimvarsWithInheritance():
            attr=pv.GetAttr();indices=pv.GetIndicesAttr()
            variables.append(dict(name=pv.GetPrimvarName(),attribute_path=str(attr.GetPath()),type_name=str(pv.GetTypeName()),
                interpolation=str(pv.GetInterpolation()),element_size=int(pv.GetElementSize()),indexed=bool(pv.IsIndexed()),
                value_sha256=compact.native_digest(attr.Get()),time_samples=list(map(float,attr.GetTimeSamples())),
                indices_sha256=compact.native_digest(indices.Get()) if indices else None))
        if variables!=record['inherited_primvars']:raise ValueError('Observed source primvar metadata/value changed')
    return True


def common_preparation_context(protocol,path):
    """Configure the ORIGINAL constructor/archive/Stop/display code; no static binding."""
    from types import SimpleNamespace
    import hospital_h4_subfloor_compact_path_launch as shared
    p=SimpleNamespace(**vars(shared.preparation));robot=protocol['new_robot_root']
    p.ROBOT,p.OLD_ROBOT,p.MARKER,p.COHORT=robot,protocol['retired_robot_root'],protocol['marker'],protocol['cohort']
    p.LIVE=Path(protocol['output_directory'])/'preparation.json'
    s=SimpleNamespace(**vars(shared.stationary))
    s.actor_configuration=shared._bound(shared.stationary.actor_configuration,ROBOT=robot,preparation=p)
    s.critical_environment=shared._bound(shared.stationary.critical_environment,ROBOT=robot,preparation=p)
    return SimpleNamespace(protocol=protocol,protocol_path=Path(path),output=p.LIVE.parent,preparation=p,
        cohort=SimpleNamespace(ROBOT=robot,MARKER=protocol['marker']),stationary=s)


def export_common_preparation_protocol(common_review_path,prior_result,*,cohort='cohort_01_common_preparation_r01'):
    """OFFLINE prepare-only definition. Approval is NOT Play/static/drive consent."""
    import copy,sys,zipfile,os,re
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):raise RuntimeError('Protocol export is OFFLINE ONLY')
    from pxr import Usd,Sdf,Gf,Plug,PhysxSchema
    import hospital_task_boundary as boundary
    import hospital_compact_usd_fingerprint as compact
    import hospital_h4_root_difference_readback as fields
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    common_review_path=Path(common_review_path).resolve();folder=common_review_path.parent
    review=json.loads(common_review_path.read_bytes());proof_path=folder/'independent_verification.json'
    proof=json.loads(proof_path.read_bytes());recipe_path=folder/'common_condition_recipe.json'
    if (proof['source_report_sha256']!=sha(common_review_path) or not proof['composed_12_plus_original188_fixed_support_ROI_and_medical_keepouts_bitwise_equal_adopted_grid']):
        raise ValueError('Complete independently verified common Session geometry required')
    for name,digest in review['output_sha256'].items():
        if sha(name)!=digest:raise ValueError('Reviewed complete environment artifact changed')
    if re.fullmatch(r'cohort_[0-9]{2}_common_preparation_r[0-9]{2}',cohort) is None:raise ValueError('Explicit common preparation-only cohort required')
    prior=boundary.completed_task_boundary(prior_result)
    patch_path=folder/'anonymous_common_Session_opinions.usda';patch,attributes,roots=checked_common_Session_patch(patch_path.read_text())
    recipe=json.loads(recipe_path.read_bytes());native_folder=folder.parent
    native=json.loads((native_folder/'native_local_geometry_review.json').read_bytes())
    rows={};metadata=[];extra_inputs=[]
    for child in ('source_metadata_GET_r01','remaining_common_environment_source_GET_r01'):
        base=native_folder/child;meta=json.loads((base/'live_source_metadata.json').read_bytes())
        verified=json.loads((base/'actual_GET_verification.json').read_bytes())
        if sha(base/'live_source_metadata.json')!=verified['source_result_sha256'] or meta['actual_tensor_after']!=prior['required_prior_tensor']:
            raise ValueError('Actual source metadata must belong to this same retained B2 boundary')
        source_contract=json.loads((base/'GET_contract.json').read_bytes())
        selected_path=Path(source_contract['authenticated_selected_source_rows'])
        metadata.extend(meta['meshes']);rows.update(json.loads(selected_path.read_bytes())['original_mesh_rows'])
        extra_inputs.extend([selected_path,*[base/n for n in ('live_source_metadata.json','actual_GET_verification.json','GET_contract.json')]])
    manifest_path=Path(native['source_receipt']['original_manifest']);manifest=json.loads(manifest_path.read_bytes())
    parked=recipe['parked_assets'];moved=[path for path in manifest['compact_environment_rows'] if any(
        path==a['root'] or path.startswith(a['root']+'/') for a in parked)]
    more,selected_receipt=read_authenticated_mesh_rows(manifest_path,moved,
        expected_manifest_sha256=native['source_receipt']['manifest_sha256'],allowed_prim_types=('Mesh','Material'))
    rows.update(more);before={};after={}
    for path,row in rows.items():
        item=dict(active=row['active'],type=row['type'],attributes={name:dict(default_sha256=compact.semantic_json_digest(a['value']),
            samples=[[t,compact.semantic_json_digest(v)] for t,v in a['time_samples']]) for name,a in row['attributes'].items()},
            relationships=row['relationships'],world_matrix=semantic_array(row['world_matrix'],'source world').tolist(),
            bound_physics_material=row['bound_physics_material'])
        before[path]=item;target=copy.deepcopy(item)
        for attribute,spec in attributes.items():
            if str(spec.path.GetPrimPath())==path:
                target['attributes'][spec.name]=dict(default_sha256=compact.native_digest(spec.default),samples=[])
        for name in ('xformOp:transform:h6_layout','xformOp:transform:h6_common'):
            spec=attributes.get(path+'.'+name)
            if spec:target['world_matrix']=np.asarray(spec.default*Gf.Matrix4d(item['world_matrix'])).tolist()
        for asset in parked:
            if path==asset['root'] or path.startswith(asset['root']+'/'):
                delta=np.linalg.inv(asset['source_root_world_matrix'])@np.asarray(asset['proposed_root_world_matrix'])
                target['world_matrix']=(np.asarray(item['world_matrix'])@delta).tolist()
        after[path]=target
    target=GATE/'hospital_h6_experiments'/cohort;robot='/World/patient_transport_h6_'+cohort;marker='/World/IJCAS_HospitalH6_'+cohort
    variant_path=Path(native['source_geometry']);geometry=json.loads(variant_path.read_bytes())
    definition_path=Path(geometry['task_definition']);definition=json.loads(definition_path.read_bytes())
    env_contract=dict(version=PREPARATION_VERSION,original_environment_manifest=str(manifest_path),
        selected_original_rows_receipt=selected_receipt,expected_before=before,expected_after=after,
        observed_source10_metadata=metadata,world_affine_numeric_tolerance_m_NOT_safety_waiver=1e-11,
        exact_properties=sorted(attributes),exact_instance_roots=roots,approved_thresholds=recipe['disabled_thresholds'])
    protocol=dict(version=PREPARATION_VERSION,scenario='H6_integrated_transfer_mission',cohort=cohort,
        method='COMMON_ENVIRONMENT_PREPARATION_ONLY',output_directory=str(target),new_robot_root=robot,marker=marker,
        required_prior_result=prior['prior_result'],prior_verdict_preserved=prior['prior_verdict'],
        **{k:prior[k] for k in ('retired_robot_root','required_prior_tensor','required_prior_timeline',
            'retiring_actor_fixed_configuration','expected_nominal_motor_baseline','protected_environment_sha256')},
        start_xyyaw=definition['start_xyyaw'],goal_xyyaw=definition['goal_xyyaw'],
        common_environment_review=str(common_review_path),common_environment_patch=str(patch_path),
        environment_contract=str(target/'environment_contract.json'),
        preparation_only=True,maximum_Stop_calls=1,maximum_static_trials=0,maximum_route_trials=0,
        Stop_wall_budget_s=15.,Stop_update_budget=240,compact_readback_wall_budget_s=180.,
        initialize_new_actor_visuals_before_static=True,automatic_retry=False,actual_motion_authorized=False,
        use_for_final_statistics=False,camera_only_explicit_start_width_m=6.,
        user_approval=dict(question_item_id='call_ba8fa3bd952c45c79580ef4cdcfeef1d:0',
            answer_ko='코드 검증 후 H6 환경·새 코호트 준비 승인',
            excludes=['Play','static','route','reset','automatic_retry']),
        status='OFFLINE_APPROVED_PREPARATION_DESIGN_NOT_ACTUAL_STAGING_OR_PHYSICAL_ADMISSION')
    context=common_preparation_context(protocol,target/'preparation_protocol.json')
    import hospital_h4_subfloor_compact_path_launch as shared
    schema_info=Path(PhysxSchema.__file__).resolve().parents[2]/'plugins/PhysxSchema/resources/plugInfo.json'
    Plug.Registry().RegisterPlugins(str(schema_info))  # OFFLINE schema registration, not physics initialization.
    if not Usd.SchemaRegistry().IsAppliedAPISchema('PhysxRigidBodyAPI'):
        raise RuntimeError('Standalone USD needs the installed PhysxSchema/resources PXR_PLUGINPATH_NAME at process start; no Kit or physics launch required')
    reference=shared.fresh_source_reference(context,True)
    files=[common_review_path,proof_path,recipe_path,patch_path,manifest_path,variant_path,definition_path,
        Path(prior['prior_result']),*extra_inputs,schema_info,Path(__file__),_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py'),
        _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_camera_view.py'),_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h4_visual_instance_refresh.py'),
        _ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    # Preserve the entire inherited code/evidence chain, plus current private
    # dependencies not part of B2's older input map. No old file is rewritten.
    dependencies={str(p):sha(p) for p in files}
    for name in ('hospital_task_experiment.py','hospital_task_runtime.py','hospital_task_boundary.py'):
        path=_ijcas_path(ROOT, 'studies/IJCAS/scripts', name);dependencies[str(path)]=sha(path)
    protocol['input_sha256']={**prior['input_sha256'],**dependencies}
    protocol['definition_sha256']=shared.digest(protocol)
    if target.exists():raise ValueError('Existing H6 cohort/definition retained; no overwrite')
    target.mkdir(parents=True)
    evidence.write_exclusive(target/'environment_contract.json',env_contract)
    evidence.write_exclusive(target/'anonymous_source_reference.json',reference)
    protocol['input_sha256'].update({str(target/n):sha(target/n) for n in ('environment_contract.json','anonymous_source_reference.json')})
    protocol.pop('definition_sha256');protocol['definition_sha256']=shared.digest(protocol)
    evidence.write_exclusive(target/'preparation_protocol.json',protocol)
    archive=target/'preparation_sources.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as frozen:
        for name in sorted(protocol['input_sha256']):
            path=Path(name)
            if path.suffix=='.py' and path.is_relative_to(ROOT):frozen.write(path,str(path.relative_to(ROOT)))
    return target/'preparation_protocol.json'


def load_common_preparation_protocol(path):
    """Small reviewed prepare-only contract; no stage, SDK or output mutation."""
    import hospital_h4_subfloor_compact_path_launch as shared
    path=Path(path).resolve()
    if path.stat().st_size>4*1024*1024:raise ValueError('Bounded preparation protocol required')
    p=json.loads(path.read_bytes())
    expected=GATE/'hospital_h6_experiments'/p['cohort']
    if (p.get('version')!=PREPARATION_VERSION or path!=expected/'preparation_protocol.json'
            or p.get('preparation_only') is not True or p.get('actual_motion_authorized') is not False
            or (p.get('maximum_Stop_calls'),p.get('maximum_static_trials'),p.get('maximum_route_trials'))!=(1,0,0)
            or p.get('automatic_retry') is not False or p.get('method')!='COMMON_ENVIRONMENT_PREPARATION_ONLY'
            or p.get('output_directory')!=str(expected)
            or p.get('new_robot_root')!='/World/patient_transport_h6_'+p['cohort']
            or p.get('marker')!='/World/IJCAS_HospitalH6_'+p['cohort']
            or p.get('definition_sha256')!=shared.digest({k:v for k,v in p.items() if k!='definition_sha256'})):
        raise ValueError('Exact one-Stop/no-Play preparation contract required')
    return common_preparation_context(p,path)


def common_preparation_operations(context,sdk):
    """Bind the existing archive/ONE Stop/source/display function code unchanged."""
    import time
    import hospital_h4_subfloor_compact_path_launch as shared
    import hospital_task_boundary as boundary
    from pxr import Usd,Sdf
    p=context.protocol;environment=json.loads(Path(p['environment_contract']).read_bytes())
    usd=sdk.omni.usd.get_context();timeline=sdk.omni.timeline.get_timeline_interface()
    stage=usd.get_stage();memory={};stopped=dict(playing=False,stopped=True,current_time_s=0.)
    def write(path,value):return evidence.write_exclusive(path,shared.normalize_runtime_evidence(value))
    def source_configuration():return shared.fresh_source_for_live_actor(context,stage)
    def own_boundary(state):return boundary.BoundaryMonitor(context,sdk,expected_state=state)
    def scan_selected_roots():
        from pxr import UsdPhysics
        roots=environment['exact_instance_roots']+[a['root'] for a in json.loads(
            Path(p['common_environment_patch']).with_name('common_condition_recipe.json').read_bytes())['parked_assets']]
        for root in roots:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root),Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
                    raise ValueError('Selected environment is not a static joint-free asset: '+str(prim.GetPath()))
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):continue
            for rel in ('physics:body0','physics:body1'):
                relationship=prim.GetRelationship(rel)
                for target in relationship.GetTargets() if relationship else []:
                    if any(target.HasPrefix(Sdf.Path(root)) for root in roots):
                        raise ValueError('An external joint references a selected environment asset')
    async def rows(phase,*,after):
        monitor=own_boundary(stopped if after else p['required_prior_timeline']);monitor.check()
        began=heartbeat=time.monotonic();actual={};prep=context.preparation
        for path,row in prep.compact.stage_rows(stage,prep.SELECTED,prep.old.original):
            if path in actual:raise ValueError('Duplicate protected environment row')
            actual[path]=row
            if len(actual)%16==0:
                monitor.check();await sdk.omni.kit.app.get_app().next_update_async();monitor.check()
            if time.monotonic()-heartbeat>=15.:
                print('[H6 PREPARATION] %s rows=%d/4997; NO Play' % (phase,len(actual)),flush=True);heartbeat=time.monotonic()
            if time.monotonic()-began>p['compact_readback_wall_budget_s']:raise TimeoutError('Whole environment readback budget; no retry')
        monitor.check();manifest=prep.load_manifest(verify_inputs=False);original=manifest['compact_environment_rows']
        if set(actual)!=set(original) or len(actual)!=4997:raise ValueError('Exact4997 collider/material/scene roster required')
        if not after:
            digest=prep.compact.compare_rows(original,actual,prep.SELECTED,after_flags=True)
            if digest!=p['protected_environment_sha256']:raise ValueError('Original B2 protected environment changed')
        else:
            for path,first in original.items():
                if path in environment['expected_after']:continue
                target=dict(first)
                if path in prep.SELECTED:target['collision_enabled']=False
                if actual[path]!=target:raise ValueError('Unrelated environment row changed: '+path)
            verify_common_environment_descriptors(stage,environment['expected_after'])
            digest=hashlib.sha256(prep.compact._json_bytes(dict(format=prep.compact.FORMAT,rows=actual))).hexdigest()
            if 'post_digest' in memory and digest!=memory['post_digest']:raise ValueError('Actual prepared environment changed between staging phases')
            memory['post_digest']=digest
        result=dict(row_count=4997,sha256=digest)
        memory[phase+'_layers']=monitor.texts()
        if after and not (context.output/'actual_common_environment_rows.json').exists():
            write(context.output/'actual_common_environment_rows.json',dict(phase=phase,codec=prep.compact.FORMAT,
                compact_rows=actual,sha256=digest,expected_changed_rows=sorted(environment['expected_after']),
                matrix_composition_tolerance_m_NOT_safety_waiver=1e-11))
        return result
    async def environment_readback(own,supplied_stage,supplied_timeline,phase):
        if own is not context or supplied_stage!=stage:raise ValueError('Own H6 preparation context/stage required')
        if phase=='before_archive':
            verify_common_environment_descriptors(stage,environment['expected_before'])
            source_composition_signature(stage,environment['observed_source10_metadata']);scan_selected_roots()
            return await rows(phase,after=False)
        if evidence.timeline_state(supplied_timeline)!=stopped:raise ValueError('Confirmed STOPPED zero-time required')
        if phase=='completed_Stop':
            verify_common_environment_descriptors(stage,environment['expected_before'])
            source_composition_signature(stage,environment['observed_source10_metadata']);scan_selected_roots()
            write(context.output/'environment_application_reserved.json',dict(maximum_application_count=1,
                actual_timeline=stopped,environment_patch_sha256=hashlib.sha256(Path(p['common_environment_patch']).read_bytes()).hexdigest(),
                no_Play_static_or_drive_authorized=True))
            applied=apply_common_Session_patch(stage,Path(p['common_environment_patch']).read_text(),
                environment['expected_before'],environment['expected_after'],authorized=True)
            write(context.output/'environment_application.json',applied)
            return await rows(phase,after=True)
        if phase=='before_renderer_initialization':
            # The constructor may only create the new robot/marker and retire
            # B2. This typed delta proof avoids a third long environment scan
            # inside the exact visual-registration snapshot.
            reference=memory['completed_Stop_layers'];current=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
            boundary.staging_scope_review(context,reference,current)
            verify_common_environment_descriptors(stage,environment['expected_after'])
            return dict(row_count=4997,sha256=memory['post_digest'])
        if phase=='new_source_staged':return await rows(phase,after=True)
        raise ValueError('Unknown H6 preparation environment phase: '+phase)
    def stop_fields(before,after,robot):
        if before==memory['before_archive_layers']['Root']:
            return shared.review_fresh_Stop_motion_fields(before,after,robot)
        if before!=memory['before_archive_layers']['Session']:raise ValueError('Unknown pre-Stop layer boundary')
        differences=shared.layer_field_differences(before,after);camera=[]
        for row in differences:
            if Sdf.Path(row['path']).GetPrimPath().HasPrefix(Sdf.Path(boundary.CAMERA)):
                boundary._validate_display_row(row);camera.append(row)
        if not camera:return shared.review_fresh_Stop_motion_fields(before,after,robot)
        def no_camera(text):
            layer=Sdf.Layer.CreateAnonymous('H6_STOP_CLASSIFY.usda')
            # Historical Session text may use #sdf rather than #usda.
            if text.startswith('#sdf '):layer=Sdf.Layer.CreateAnonymous('H6_STOP_CLASSIFY.sdf')
            if not layer.ImportFromString(text):raise ValueError('Stop Session classifier cannot parse typed copy')
            if layer.GetPrimAtPath(boundary.CAMERA):
                edit=Sdf.BatchNamespaceEdit();edit.Add(boundary.CAMERA,Sdf.Path.emptyPath)
                if not layer.Apply(edit):raise ValueError('Cannot classify camera-only Stop delta')
            return layer.ExportToString()
        motion=shared.review_fresh_Stop_motion_fields(no_camera(before),no_camera(after),robot)
        return dict(retiring_actor_motion_fields=motion,exact_Top_XY_aperture_only_differences=camera)
    original_display=shared._bound(shared.initialize_fresh_staged_display,preparation=context.preparation,
        fresh_source_for_live_actor=shared.fresh_source_for_live_actor,fresh_environment_readback=environment_readback)
    async def display(own):
        if own is not context:raise ValueError('Own first-display actor required')
        monitor=own_boundary(stopped);last=monitor.texts();stable=0
        for _ in range(60):
            monitor.check();await sdk.omni.kit.app.get_app().next_update_async();monitor.check();current=monitor.texts()
            stable=stable+1 if current==last else 0;last=current
            if stable>=3:break
        else:raise TimeoutError('Top display defaults did not settle before exact initialization')
        return await original_display(context)
    stage_operation=shared._bound(shared.prepare_fresh_live_trial,preparation=context.preparation,
        fresh_source_for_live_actor=shared.fresh_source_for_live_actor,fresh_environment_readback=environment_readback,
        retiring_actor_configuration=shared.retiring_actor_configuration,initialize_fresh_staged_display=display,
        review_fresh_Stop_motion_fields=stop_fields,
        print=lambda *a,**kw:print(*[v.replace('[H4 FRESH]','[H6 PREPARATION]').replace(
            'all original environment rows unchanged','all approved H6 environment rows verified') if isinstance(v,str) else v for v in a],**kw))
    return stage_operation


async def run_common_preparation_once(path,*,authorized=False):
    """Approved archive/ONE Stop/Session environment/source/display ONLY."""
    if authorized is not True:raise ValueError('Separate prepare-only consent required before SDK access')
    import asyncio,builtins,time
    context=load_common_preparation_protocol(path);p=context.protocol;output=context.output
    reserve=output/'dispatch_reserved.json'
    if reserve.exists():
        print('[H6 PREPARATION] Already dispatched; no new action or retry. Inspect existing preparation/error receipts.',flush=True)
        return dict(status='PRIOR_DISPATCH_RETAINED_NO_NEW_ACTION')
    for name in ('ijcas_hospital_latest_task_experiment','ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done() and previous is not asyncio.current_task():raise ValueError('Another research task is active')
    evidence.write_exclusive(reserve,dict(protocol=str(context.protocol_path),explicit_preparation_only_consent=True,
        maximum_Stop_calls=1,maximum_static_trials=0,maximum_route_trials=0,automatic_retry=False))
    phase='current_PAUSED_preflight'
    try:
        import hospital_task_runtime as runtime
        import hospital_task_boundary as boundary
        import hospital_h4_camera_view as camera
        import hospital_h4_subfloor_compact_path_launch as shared
        sdk=runtime.live_sdk();stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
        monitor=boundary.BoundaryMonitor(context,sdk,expected_state=p['required_prior_timeline'])
        view=sdk.SimulationManager.get_physics_sim_view()
        if view is None:raise ValueError('Existing retained B2 physical view required; no initialization')
        tensor=view.create_articulation_view(p['retired_robot_root']+'/base_link')
        def unchanged():
            monitor.check()
            if shared.normalize_runtime_evidence(context.preparation.old.tensor_snapshot(tensor))!=p['required_prior_tensor']:
                raise ValueError('Retained B2 body/all12joints/all13links/gains/targets changed')
        unchanged();began=time.monotonic()
        for index,(name,digest) in enumerate(p['input_sha256'].items(),1):
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Reviewed preparation dependency changed: '+name)
            if index%16==0:
                unchanged();await sdk.omni.kit.app.get_app().next_update_async();unchanged()
            if time.monotonic()-began>180.:raise TimeoutError('Bounded input verification exceeded')
        unchanged()
        env=json.loads(Path(p['environment_contract']).read_bytes())
        verify_common_environment_descriptors(stage,env['expected_before']);source_composition_signature(stage,env['observed_source10_metadata'])
        if shared.retiring_actor_configuration(context,stage)!=p['retiring_actor_fixed_configuration']:
            raise ValueError('Prior B2 fixed configuration changed')
        for root in (p['new_robot_root'],p['marker']):
            if stage.GetPrimAtPath(root):raise ValueError('New actor/marker already exists; never replace it')
        phase='camera_only_before_archive'
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport is None or str(viewport.camera_path)!=camera.CAMERA:raise ValueError('Existing Top viewport required; no camera/renderer substitution')
        view_record=camera.focus_top_xy_only(stage,[*p['start_xyyaw'][:2],0.],viewport.resolution,p['camera_only_explicit_start_width_m'])
        unchanged();evidence.write_exclusive(output/'view_preparation.json',view_record)
        phase='archive_ONE_Stop_common_environment_source_display'
        result=await common_preparation_operations(context,sdk)(context)
        if (evidence.timeline_state(timeline)!=dict(playing=False,stopped=True,current_time_s=0.)
                or result['Play_or_static_or_drive_executed'] is not False):raise ValueError('Prepare-only STOPPED closeout required')
        summary=dict(status='H6_COMMON_ENVIRONMENT_NEW_SOURCE_DISPLAY_PREPARED_STATIC_AND_DRIVE_PENDING',
            new_robot_root=p['new_robot_root'],timeline=evidence.timeline_state(timeline),
            actual_environment_sha256=result['protected_environment_sha256'],Stop_calls=1,
            static_trials=0,route_trials=0,automatic_retry=False,use_for_final_statistics=False,
            source_reference=result['source_reference'],actual_visible_pixels_require_user_observation=True)
        evidence.write_exclusive(output/'prepare_only_closeout.json',summary)
        print('=== H6 COMMON ENVIRONMENT / NEW FIXED-START SOURCE PREPARED ===',flush=True)
        print('actor:',p['new_robot_root'],'; 265kg /13 bodies /12 source joints',flush=True)
        print('STOPPED. ONE Stop; NO Play/static/drive/reset/retry. saved:',output/'prepare_only_closeout.json',flush=True)
        return summary
    except BaseException as error:
        diagnostic=dict(phase=phase,error=type(error).__name__+': '+str(error),
            no_automatic_retry_or_rollback=True,no_static_or_drive_admission=True,
            original_reports_preserved=True,setup_error_NOT_a_scientific_method_limitation=True)
        if hasattr(error,'boundary_transaction'):diagnostic['exact_failed_boundary_transaction']=error.boundary_transaction
        if 'stage' in locals() and stage is not None:
            try:
                texts=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
                diagnostic['actual_current_layer_sha256']={k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()}
                if sum(len(v.encode()) for v in texts.values())<=2*1024*1024:diagnostic['actual_current_layer_texts']=texts
                diagnostic['actual_current_timeline']=evidence.timeline_state(timeline)
            except Exception as read_error:diagnostic['readback_error']=type(read_error).__name__+': '+str(read_error)
        try:evidence.write_exclusive(output/'preparation_error.json',diagnostic)
        except Exception as secondary:print('[H6 PREPARATION] Error receipt also failed:',str(secondary),flush=True)
        raise


def schedule_common_preparation(path,*,authorized=False):
    if authorized is not True:raise ValueError('Explicit prepare-only consent required before dispatch')
    import asyncio,builtins,traceback
    context=load_common_preparation_protocol(path)
    if (context.output/'dispatch_reserved.json').exists():
        print('[H6 PREPARATION] Existing one-time dispatch retained; no duplicate or retry.',flush=True);return None
    for name in ('ijcas_hospital_latest_task_experiment','ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():raise ValueError('Another research task is active; no competing preparation')
    def done(future):
        if future.cancelled():print('[H6 PREPARATION] Cancelled; preserve current state. No retry.',flush=True)
        elif future.exception() is not None:
            error=future.exception();traceback.print_exception(type(error),error,error.__traceback__)
            print('[H6 PREPARATION] Aborted; no additional Stop/Play/reset/retry. See exact error receipt.',flush=True)
    future=asyncio.ensure_future(run_common_preparation_once(path,authorized=True));future.add_done_callback(done)
    builtins.ijcas_hospital_latest_task_experiment=future
    print('[H6 PREPARATION] ONE approved archive/Stop -> common Session environment -> NEW source/FIRST display scheduled. NO Play/static/route.',flush=True)
    return future


def mesh_local_geometry_recipe(row, world_points_m, *, target_world_matrix=None):
    """NEW anonymous-USD geometry candidate, never a live authoring command.

    Explicitly round points to the source point3f representation, recompute
    local extent and area-weighted vertex normals using the retained original
    triangle topology. This proposes vertex-normal interpolation; it does NOT
    infer source interpolation/instancing metadata absent from the archive.
    A caller must reproject the rounded world mesh into its original map.
    """
    attrs=row['attributes']
    if (row.get('type')!='Mesh' or attrs['points']['value'].get('type')!='Vec3fArray'
            or attrs.get('orientation',{}).get('value')!='rightHanded'
            or attrs.get('subdivisionScheme',{}).get('value')!='none'
            or any(attrs[n]['time_samples'] for n in ('points','faceVertexCounts','faceVertexIndices'))):
        raise ValueError('Explicit static point3f right-handed non-subdivided source mesh required')
    source=semantic_array(attrs['points']['value'],'points')
    counts=semantic_array(attrs['faceVertexCounts']['value'],'counts')
    indices=semantic_array(attrs['faceVertexIndices']['value'],'indices')
    matrix=semantic_array(row['world_matrix'],'world_matrix') if target_world_matrix is None else np.asarray(target_world_matrix,float)
    world=np.asarray(world_points_m,float)
    if (world.shape!=source.shape or world.ndim!=2 or world.shape[1]!=3 or not len(world)
            or not np.isfinite(world).all() or counts.ndim!=1 or not len(counts) or not np.all(counts==3)
            or indices.ndim!=1 or indices.dtype.kind not in 'iu' or len(indices)!=3*len(counts)
            or np.any(indices<0) or np.any(indices>=len(source)) or matrix.shape!=(4,4)
            or not np.isfinite(matrix).all() or not np.allclose(matrix[:3,3],0.,rtol=0.,atol=1e-12)
            or abs(matrix[3,3]-1.)>1e-12 or np.linalg.cond(matrix[:3,:3])>1e10):
        raise ValueError('Same finite source triangle topology and invertible affine row transform required')
    inverse=np.linalg.inv(matrix)
    local64=(np.c_[world,np.ones(len(world))]@inverse)[:,:3]
    with np.errstate(over='ignore',invalid='ignore'):local=local64.astype(np.float32)
    if not np.isfinite(local).all():raise ValueError('Changed points do not fit native USD point3f')
    triangles=indices.reshape(-1,3);old=source[triangles];new=local.astype(float)[triangles]
    old_cross=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
    cross=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
    nondegenerate=np.linalg.norm(old_cross,axis=1)>1e-12
    if (np.any(np.linalg.norm(cross[nondegenerate],axis=1)<=1e-12)
            or np.any(np.sum(cross[nondegenerate]*old_cross[nondegenerate],axis=1)<=0)):
        raise ValueError('Changed native triangles collapse or reverse original local orientation')
    normals=np.zeros_like(local,dtype=float)
    for corner in range(3):np.add.at(normals,triangles[:,corner],cross)
    lengths=np.linalg.norm(normals,axis=1);missing=lengths<=1e-12
    used=np.zeros(len(local),bool);used[indices]=True
    if np.any(missing&used):raise ValueError('Referenced native vertex has undefined accumulated normal')
    if np.any(missing):
        original=semantic_array(attrs['normals']['value'],'normals')
        if original.shape!=normals.shape:raise ValueError('Unused source vertices need explicit finite vertex normals')
        normals[missing]=original[missing];lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths<=1e-12):raise ValueError('Zero source or changed vertex normal')
    normals=(normals/lengths[:,None]).astype(np.float32)
    extent=np.asarray([local.min(axis=0),local.max(axis=0)],dtype=np.float32)
    reconstructed=(np.c_[local.astype(float),np.ones(len(local))]@matrix)[:,:3]
    report=dict(point_count=len(local),triangle_count=len(counts),source_point_type='point3f[]',
        original_topology_and_local_orientation_retained=True,
        maximum_world_point_roundtrip_error_m=float(np.max(np.abs(reconstructed-world))),
        normals_policy='NEW_area_weighted_vertex_normals_with_original_index_splits',
        proposed_normals_interpolation='vertex',source_interpolation_and_instance_metadata_NOT_inferred=True,
        actual_source_USD_or_Kit_or_physics_NOT_modified=True,new_PhysX_trial_count=0,actual_motion_authorized=False)
    return report,dict(local_points=local,local_normals=normals,local_extent=extent,
        counts=counts.astype(np.int32),indices=indices.astype(np.int32),world_matrix=matrix.copy(),
        rounded_world_points_m=reconstructed)


def archived_mesh_geometry(row,source_bound):
    """World geometry from one authenticated current-equivalent archived prim.

    Unknown/nontriangle/approximated shapes retain their AABB in projection.
    Malformed data is an error, never empty/free space. No fan triangulation,
    active flag change, primitive move, unit fit or cooked-shape claim.
    """
    if row['type']!='Mesh':return dict(projection='RETAIN_ORIGINAL_AABB',reason='NOT_A_MESH'),{}
    attrs=row['attributes']
    for name in ('points','faceVertexCounts','faceVertexIndices','physics:approximation'):
        if name in attrs and attrs[name]['time_samples']:
            return dict(projection='RETAIN_ORIGINAL_AABB',reason='TIME_SAMPLED_GEOMETRY_OR_POLICY'),{}
    points=semantic_array(attrs['points']['value'],'points')
    counts=semantic_array(attrs['faceVertexCounts']['value'],'faceVertexCounts')
    indices=semantic_array(attrs['faceVertexIndices']['value'],'faceVertexIndices')
    matrix=semantic_array(row['world_matrix'],'world_matrix')
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)==0 or matrix.shape!=(4,4)
            or counts.ndim!=1 or indices.ndim!=1 or counts.dtype.kind not in 'iu' or indices.dtype.kind not in 'iu'
            or np.any(counts<3) or int(counts.sum())!=len(indices) or np.any(indices<0) or np.any(indices>=len(points))
            or not np.allclose(matrix[:3,3],0.,rtol=0.,atol=1e-12) or abs(matrix[3,3]-1.)>1e-12):
        raise ValueError('Complete static mesh topology and affine USD row-vector transform required')
    world=np.c_[points,np.ones(len(points))]@matrix
    world=world[:,:3]
    low=vector(source_bound['minimum_m'],3,'recorded minimum');high=vector(source_bound['maximum_m'],3,'recorded maximum')
    if np.any(world.min(axis=0)<low-1e-6) or np.any(world.max(axis=0)>high+1e-6):
        raise ValueError('Archived transformed mesh exceeds actual recorded collider bound')
    approximation=attrs.get('physics:approximation',{}).get('value')
    holes=semantic_array(attrs['holeIndices']['value'],'holeIndices') if 'holeIndices' in attrs else np.empty(0)
    reason='STATIC_TRIANGLES_APPROXIMATION_NONE' if approximation=='none' and np.all(counts==3) and not holes.size else (
        'MESH_APPROXIMATION_UNKNOWN_OR_NOT_NONE' if approximation!='none' else
        'AUTHORED_NONTRIANGLE_OR_HOLE_TOPOLOGY_NOT_SILENTLY_RETRIANGULATED')
    return dict(projection='AUTHORED_STATIC_TRIANGLE_SLAB' if reason=='STATIC_TRIANGLES_APPROXIMATION_NONE' else 'RETAIN_ORIGINAL_AABB',
        reason=reason,recorded_physics_approximation=approximation,point_count=len(points),face_count=len(counts),
        current_bound_contains_all_transformed_points_with_numeric_1um=True,
        approximate_bounds_or_authored_mesh_NOT_cooked_PhysX_validation=True),dict(points=world,counts=counts,indices=indices)


def fixed_cell_blockers(grid,supported):
    """Independent support/ROI blockers must survive obstacle replacement.

    Removing an obstacle from a UNION cannot recover which overlapping cells
    were also unsupported or outside the measured ROI. Reconstruct these
    fixed masks from their own original sources, never from union subtraction.
    """
    raster=grid['raster'];shape=tuple(raster['shape_rows_columns'])
    origin=vector(raster['origin_xy_m'],2,'original fixed-mask origin')
    roi=vector(grid['analysis_roi_xy_m'],4,'original fixed-mask ROI');resolution=raster['resolution_m']
    if (not isinstance(supported,np.ndarray) or supported.dtype!=np.dtype(bool) or supported.ndim!=2
            or supported.shape!=shape or not supported.size or supported.size>MAX_CELLS
            or type(resolution) not in (int,float) or not math.isfinite(resolution) or resolution<=0
            or roi[0]>=roi[1] or roi[2]>=roi[3]):
        raise ValueError('Same bounded supported-cell mask and original finite ROI/resolution required')
    x0=origin[0]+np.arange(shape[1])*resolution;y0=origin[1]+np.arange(shape[0])*resolution
    inside=(x0[None,:]>=roi[0])&(x0[None,:]+resolution<=roi[1])&(
        y0[:,None]>=roi[2])&(y0[:,None]+resolution<=roi[3])
    return ~supported|~inside


def refine_recorded_mesh_grid(coverage,bundle,array_store):
    """One source-compatible grid; unknown geometry NEVER makes a free hole."""
    from hospital_geometry_tools import mesh_mask
    grid=coverage['coverage_grid'];raster=grid['raster'];shape=tuple(raster['shape_rows_columns'])
    origin=np.asarray(raster['origin_xy_m']);included=grid['included_obstacles']
    records=bundle['primitives']
    if (len(records)!=len(included) or len({r['path'] for r in records})!=len(records)
            or {r['path'] for r in records}!={r['path'] for r in included}):
        raise ValueError('Every original included obstacle needs an explicit geometry/policy record')
    lookup={r['path']:r for r in records};obstacles=np.zeros(shape,bool);methods=[]
    for item in included:
        record=lookup[item['path']]
        if record['source_world_aligned_bound']!=item['world_aligned_bound']:
            raise ValueError('Source collider bound or mission-frame geometry changed')
        mode=record['geometry']['projection'];lo=item['world_aligned_bound']['minimum_m'];hi=item['world_aligned_bound']['maximum_m']
        if mode=='AUTHORED_STATIC_TRIANGLE_SLAB':
            if record['geometry'].get('recorded_physics_approximation')!='none':
                raise ValueError('Triangle projection cannot waive an unknown/convex collision approximation')
            prefix=record['array_prefix']
            mesh=dict(world_points_m=np.asarray(array_store[prefix+'_points']),
                face_vertex_counts=np.asarray(array_store[prefix+'_counts']).tolist(),
                face_vertex_indices=np.asarray(array_store[prefix+'_indices']))
            points=mesh['world_points_m']
            if (points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all()
                    or np.any(points.min(axis=0)<np.asarray(lo)-1e-6) or np.any(points.max(axis=0)>np.asarray(hi)+1e-6)):
                raise ValueError('Archived geometry arrays no longer reproduce the recorded collider bound')
            mask=mesh_mask(mesh,origin,raster['resolution_m'],shape,*grid['original_height_slab_m'])
            # Do not truncate changed geometry to hide new occupied cells.
            rr,cc=polygon_cells([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],origin,raster['resolution_m'],shape)
            bound_mask=np.zeros(shape,bool);bound_mask[rr,cc]=True
            if np.any(mask&~bound_mask):raise ValueError('Projected mesh leaves original bound cells; no clipping repair')
            obstacles|=mask
        elif mode=='RETAIN_ORIGINAL_AABB':
            rr,cc=polygon_cells([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],origin,raster['resolution_m'],shape)
            obstacles[rr,cc]=True
        else:raise ValueError('Unknown authored obstacle representation; no free-space substitution')
        methods.append(dict(path=item['path'],projection=mode,reason=record['geometry']['reason']))
    # Preserve the exact existing support/raster extent. Triangle refinement
    # only changes obstacle representation, never floor coverage or flags.
    source=coverage['output_sha256'];filename=next(Path(p) for p in source if p.endswith('authored_coverage_grid.npz'))
    with np.load(filename) as original:
        supported=original['supported_full_cells'].copy();old_obstacles=original['obstacle_AABBs'].copy()
        old_blocked=original['occupied'].copy()
    fixed=fixed_cell_blockers(grid,supported)
    if (old_obstacles.dtype!=np.dtype(bool) or old_blocked.dtype!=np.dtype(bool)
            or old_obstacles.shape!=shape or old_blocked.shape!=shape
            or not np.array_equal(old_blocked,old_obstacles|fixed)):
        raise ValueError('Original complete support/ROI/obstacle union must reproduce before refinement')
    if np.any(obstacles&~old_obstacles):raise ValueError('Refinement added cells outside unchanged source AABBs')
    blocked=obstacles|fixed
    return dict(version=MESH_REVIEW_VERSION,primitive_policies=methods,
        original_obstacle_count=len(included),triangle_projected_primitives=sum(r['projection']=='AUTHORED_STATIC_TRIANGLE_SLAB' for r in methods),
        retained_AABB_primitives=sum(r['projection']=='RETAIN_ORIGINAL_AABB' for r in methods),
        original_obstacle_cells=int(old_obstacles.sum()),refined_obstacle_cells=int(obstacles.sum()),
        removed_AABB_only_cells=int(np.count_nonzero(old_obstacles&~obstacles)),
        fixed_support_or_ROI_blocker_cells=int(fixed.sum()),
        fixed_blockers_recovered_vs_legacy_union_subtraction=int(np.count_nonzero(fixed&~(obstacles|(old_blocked&~old_obstacles)))),
        fixed_mask_reconstructed_from_original_support_and_ROI_not_union_subtraction=True,
        exact_original_support_and_ROI_and_flags_preserved=True,unknown_or_missing_geometry_NOT_free=True,
        no_H6_task_route_cooked_support_or_drive_admission=True),blocked,supported,obstacles


def export_archived_mesh_bundle(coverage_path,archive_manifest_path):
    """OFFLINE process only: reuse the original bounded per-prim JSON reader.

    The 1.68GB historical archive is hashed in chunks, memory-mapped read-only
    and decoded one prim at a time. NEVER call this in Kit/Script Editor.
    Exact compact rows prove geometry/policy equivalence to the saved H5
    environment; four previously approved flags are only accounted for in a
    copied fingerprint dictionary, never changed in a USD scene.
    """
    import copy,io,os,sys,time,zipfile
    import hospital_compact_usd_fingerprint as compact
    if any(name in sys.modules for name in ('omni.kit.app','omni.physx','isaacsim')):
        raise RuntimeError('Large-archive mesh extraction is OFFLINE ONLY; no Kit or live adapter')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    coverage_path=Path(coverage_path).resolve();manifest_path=Path(archive_manifest_path).resolve()
    coverage=json.loads(coverage_path.read_bytes());manifest=json.loads(manifest_path.read_bytes())
    for path,value in {**coverage['input_sha256'],**coverage['output_sha256']}.items():
        if sha(path)!=value:raise ValueError('Original authored coverage evidence changed: '+path)
    if coverage['input_sha256'].get(str(manifest_path))!=sha(manifest_path):
        raise ValueError('Original full-archive compact manifest must be in the verified source lineage')
    expected=manifest['compact_environment_rows'];after=copy.deepcopy(expected)
    for name in manifest['approved_disabled_paths']:after[name]['collision_enabled']=False
    fingerprint=compact.compare_rows(expected,after,manifest['approved_disabled_paths'],True)
    reference=coverage['contract']['source_4997_row_environment']
    if (manifest['codec']!=compact.FORMAT or len(after)!=reference['row_count'] or fingerprint!=reference['sha256']):
        raise ValueError('Entire saved current environment is not the archived geometry plus exact approved flags')
    archive=Path(manifest['original_archive']);before=archive.stat()
    if (before.st_size!=manifest['original_archive_bytes'] or before.st_mtime_ns!=manifest['original_archive_mtime_ns']):
        raise ValueError('Original immutable archive metadata changed; no source substitution')
    buffer=io.BytesIO();sources=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as saved:
        for path in sources:
            member=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));member.external_attr=0o100644<<16
            saved.writestr(member,path.read_bytes())
    source_bytes=buffer.getvalue()
    contract=dict(version=MESH_REVIEW_VERSION,role='RECORDED_USD_GEOMETRY_EXTRACTION_NOT_NATIVE_COOKING_OR_MOTION',
        source_coverage=str(coverage_path),source_coverage_sha256=sha(coverage_path),
        original_compact_manifest=str(manifest_path),original_compact_manifest_sha256=sha(manifest_path),
        full_original_archive=str(archive),full_original_archive_expected_sha256=manifest['original_archive_sha256'],
        full_original_archive_bytes=before.st_size,all_4997_row_fingerprint=fingerprint,
        original_flags_accounted_for_ONLY_in_copied_dictionary=True,
        selected_paths=[r['path'] for r in coverage['coverage_grid']['included_obstacles']],
        source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),source_members_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        no_original_source_scene_or_report_edit=True,new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/mesh_refinement'/key;target=folder/'mesh_geometry_bundle.json'
    if target.exists():
        result=json.loads(target.read_bytes())
        if result['contract']!=contract or (folder/'executed_sources.zip').read_bytes()!=source_bytes:
            raise ValueError('Existing mesh extraction source differs; no overwrite')
        for path,value in {**result['input_sha256'],**result['output_sha256']}.items():
            if sha(path)!=value:raise ValueError('Original mesh bundle changed: '+path)
        return target
    if folder.exists():raise ValueError('Partial offline extraction retained; inspect its process, no automatic restart')
    print('[H6 MESH OFFLINE] hashing immutable archive in bounded chunks; no Kit/SDK command',flush=True)
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=contract['full_original_archive_expected_sha256']:
        raise ValueError('Full original archive SHA differs; no geometry extraction')
    folder.mkdir(parents=True,exist_ok=False)
    with (folder/'executed_sources.zip').open('xb') as stream:stream.write(source_bytes);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    requested={row['path']:row for row in coverage['coverage_grid']['included_obstacles']}
    primitives=[];arrays={};seen=set();started=heartbeat=time.monotonic();largest=0
    for name,row,byte_count in compact.iter_archived_rows(archive):
        if name in seen:raise ValueError('Duplicate original archived prim')
        seen.add(name);largest=max(largest,byte_count)
        if name in requested:
            if compact.archived_row(name,row,manifest['approved_disabled_paths'])!=expected[name]:
                raise ValueError('Full archived geometry/policy row differs from the current-equivalent fingerprint: '+name)
            geometry,values=archived_mesh_geometry(row,requested[name]['world_aligned_bound'])
            prefix='mesh%03d'%len(primitives)
            arrays.update({prefix+'_'+field:value for field,value in values.items()})
            primitives.append(dict(path=name,array_prefix=prefix,geometry=geometry,
                source_world_aligned_bound=requested[name]['world_aligned_bound'],
                full_archived_prim_row_sha256=expected[name]['row_sha256']))
        if time.monotonic()-heartbeat>=15.:
            print('[H6 MESH OFFLINE] original rows=%d selected=%d/%d elapsed=%.1fs; no live action'%(
                len(seen),len(primitives),len(requested),time.monotonic()-started),flush=True);heartbeat=time.monotonic()
    after_stat=archive.stat()
    if (len(seen)!=manifest['prim_row_count'] or set(seen)!=set(expected)
            or {r['path'] for r in primitives}!=set(requested)
            or (before.st_size,before.st_mtime_ns)!=(after_stat.st_size,after_stat.st_mtime_ns)):
        raise ValueError('Complete immutable archive/selected geometry coverage was not verified')
    numeric=folder/'recorded_world_mesh_arrays.npz';np.savez_compressed(numeric,**arrays)
    result=dict(contract=contract,primitives=primitives,selected_geometry_count=len(primitives),
        all_original_archive_rows_seen=len(seen),largest_one_decoded_row_bytes=largest,
        full_original_archive_sha256_verified=digest.hexdigest(),
        extraction_elapsed_s=time.monotonic()-started,
        geometry_is_recorded_USD_NOT_new_live_readback_or_cooked_PhysX=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256={**coverage['input_sha256'],**coverage['output_sha256'],str(coverage_path):sha(coverage_path),
            str(manifest_path):sha(manifest_path),str(Path(compact.__file__).resolve()):sha(Path(compact.__file__).resolve())},
        output_sha256={str(path):sha(path) for path in (numeric,folder/'executed_sources.zip',folder/'software_contract.json')})
    evidence.write_exclusive(target,result)
    print('[H6 MESH OFFLINE] complete original rows=%d; geometry=%d; source untouched'%(len(seen),len(primitives)),flush=True)
    return target


def export_mesh_projection(coverage_path,bundle_path):
    """Same source environment, support/slab/ROI; only map representation."""
    import io,os,zipfile
    coverage_path=Path(coverage_path).resolve();bundle_path=Path(bundle_path).resolve()
    coverage=json.loads(coverage_path.read_bytes());bundle=json.loads(bundle_path.read_bytes())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    pins={**bundle['input_sha256'],**bundle['output_sha256'],str(bundle_path):sha(bundle_path)}
    for path,value in pins.items():
        if sha(path)!=value:raise ValueError('Recorded mesh/source evidence changed: '+path)
    if (bundle['contract']['source_coverage']!=str(coverage_path)
            or bundle['contract']['source_coverage_sha256']!=sha(coverage_path)):
        raise ValueError('Exact existing coverage/mesh-bundle lineage required')
    folder=bundle_path.parent;target=folder/'mesh_projection_review.json';history=None
    if target.exists() and json.loads(target.read_bytes())['version']!=MESH_REVIEW_VERSION:
        # Preserve the original report/grid and reuse its authenticated mesh
        # arrays. A projector correction must not re-extract the 1.6GB archive
        # or silently return the historical implementation's raster.
        previous=json.loads(target.read_bytes())
        history=dict(path=str(target),sha256=sha(target),version=previous['version'])
        for name,value in {**previous.get('output_sha256',{}),str(target):sha(target)}.items():
            if name in pins and pins[name]!=value:raise ValueError('Historical projection pin conflict')
            if sha(name)!=value:raise ValueError('Historical projection changed; no corrected export')
            pins[name]=value
        folder=folder/'corrected_fixed_blockers_v2';target=folder/'mesh_projection_review.json'
    if target.exists():
        report=json.loads(target.read_bytes())
        if report['input_sha256']!=pins:raise ValueError('Existing projection input changed; no overwrite')
        for path,value in report['output_sha256'].items():
            if sha(path)!=value:raise ValueError('Existing projection output changed; no overwrite')
        return target
    numeric=next(Path(path) for path in bundle['output_sha256'] if path.endswith('.npz'))
    with np.load(numeric) as arrays:
        summary,blocked,supported,obstacles=refine_recorded_mesh_grid(coverage,bundle,arrays)
    controls=task_anchor_review(coverage['contract']['anchors'],coverage['coverage_grid'],blocked,
        coverage['contract']['same_three_source_support_rectangles'])
    raster=folder/'mesh_refined_grid.npz';preview=folder/'mesh_refinement_review.png'
    if raster.exists() or preview.exists():raise ValueError('Incomplete projection preserved; no automatic overwrite/restart')
    folder.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for path in (Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')):
            info=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,path.read_bytes())
    archive=folder/'projection_sources.zip'
    with archive.open('xb') as f:f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
    np.savez_compressed(raster,occupied=blocked,supported_full_cells=supported,obstacle_mesh_projection=obstacles,
        fixed_support_and_ROI_blockers=fixed_cell_blockers(coverage['coverage_grid'],supported),
        origin_xy_m=coverage['coverage_grid']['raster']['origin_xy_m'],resolution_m=RESOLUTION_M)
    old_grid=next(Path(p) for p in coverage['output_sha256'] if p.endswith('authored_coverage_grid.npz'))
    with np.load(old_grid) as old:old_obstacles=old['obstacle_AABBs'].copy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Polygon
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    origin=np.asarray(coverage['coverage_grid']['raster']['origin_xy_m'])
    extent=[origin[0],origin[0]+blocked.shape[1]*RESOLUTION_M,origin[1],origin[1]+blocked.shape[0]*RESOLUTION_M]
    fixed=fixed_cell_blockers(coverage['coverage_grid'],supported)
    image=np.where(fixed,2,np.where(obstacles,1,0))
    difference=np.where(fixed,3,np.where(obstacles,1,np.where(old_obstacles,2,0)))
    fig,(ax,zoom)=plt.subplots(1,2,figsize=(12,9),layout='constrained',gridspec_kw={'width_ratios':[1.1,1.]})
    ax.imshow(image,origin='lower',extent=extent,cmap=ListedColormap(['#ffffff','#47515c','#dce9f3']),vmin=0,vmax=2)
    offsets={'H1_goal':(-80,28),'H2_start':(-78,-24),'H3_goal':(12,4)}
    for i,item in enumerate(controls['anchors']):
        pose=item['world_xyyaw'];color='#16744b' if item['full_original_10mm_footprint_clear_in_source_raster'] else '#bd3138'
        ax.add_patch(Polygon(footprint_polygon(pose),fill=False,edgecolor=color,linewidth=.6,alpha=.6))
        ax.plot(*pose[:2],'o',color=color,markersize=2)
        ax.annotate(item['id'],pose[:2],xytext=offsets.get(item['id'],((5 if i%2 else -55),(6 if i%2 else -14))),
            textcoords='offset points',fontsize=7,color=color,arrowprops=dict(arrowstyle='-',lw=.4,color=color))
    ax.set(aspect='equal',xlabel='World X (m)',ylabel='World Y (m)')
    ax.set_title('동일 환경의 삼각형 투영\n기록된 문·문턱·지지면 유지',fontproperties=font)
    zoom.imshow(difference,origin='lower',extent=extent,cmap=ListedColormap(['#ffffff','#47515c','#f3d38c','#dce9f3']),vmin=0,vmax=3)
    zoom.set(xlim=(16.4,17.7),ylim=(14.9,16.8),aspect='equal',xlabel='World X (m)',ylabel='World Y (m)')
    zoom.set_title('원래 H3 문 주변의 표현 차이\n황색: AABB에만 있던 셀',fontproperties=font)
    zoom.text(.02,.02,'문 개방이나 문턱 제거를 실행한 결과가 아님\n주행 가능성·실제 접촉·H6 성공 검증 아님',transform=zoom.transAxes,
        fontproperties=font,fontsize=9,bbox=dict(facecolor='white',alpha=.92,edgecolor='none'))
    try:fig.savefig(preview,dpi=150)
    finally:plt.close(fig)
    for path,value in pins.items():
        if sha(path)!=value:raise ValueError('Source changed during mesh projection: '+path)
    report=dict(version=MESH_REVIEW_VERSION,status='AUTHORED_STATIC_MESH_PROJECTION_RECORDED_NO_H6_ROUTE_OR_COOKED_DRIVE_ADMISSION',
        source_coverage=str(coverage_path),mesh_bundle=str(bundle_path),projection_summary=summary,
        before_source_control_review=coverage['source_control_review'],after_source_control_review=controls,
        original_H3_environment_difference_NOT_repaired_or_waived=coverage['H3_environment_comparison'],
        source_task_poses_and_all_45_pair_definitions_unchanged=True,
        positive_POINT_connectivity_is_NOT_a_full_footprint_steering_force_or_continuous_mission=True,
        original_H4_H5_AABB_planning_and_results_NOT_relabelled=True,
        original_1029_NE_response_and_control_gains_and_acceptance_NOT_changed=True,
        recorded_meshes_NOT_new_live_geometry_or_cooked_PhysX_shapes=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        historical_projection_NOT_overwritten_or_relabelled=True,
        historical_projection_preserved=history,
        input_sha256=pins,output_sha256={str(path):sha(path) for path in (raster,preview,archive)})
    evidence.write_exclusive(target,report)
    return target


MEDICAL_KEEPOUT_VERSION='H6_RECORDED_MEDICAL_ASSET_UNION_KEEPOUT_NOT_UNDER_FURNITURE_v1'


def medical_asset_keepouts(inventory,asset_receipt,grid):
    """Retain the entire recorded medical-asset envelope, including leg gaps.

    Triangle projections describe collision geometry, not permission to drive
    underneath a bed/cart. Use the existing, independently captured 45-root
    roster, verifying EVERY enabled collider against the ALL-stage inventory.
    The conservative union box is NOT a newly measured visible silhouette or
    a cooked shape. No new clearance, scene edit or success-selected omission.
    """
    from collections import Counter
    rows=inventory['collision_prims']
    if (inventory.get('authored_inventory_coverage_complete') is not True or inventory.get('errors')!=[]
            or inventory.get('active_unloaded_payload_paths')!=[] or not rows
            or len(rows)!=inventory['collision_prim_count'] or len({r['path'] for r in rows})!=len(rows)
            or dict(Counter(r['classification'] for r in rows))!=inventory['classification_counts']):
        raise ValueError('Complete unique recorded ALL-stage inventory required')
    assets=asset_receipt['actual_assets'];comparison=asset_receipt['cached_vs_actual'];families=comparison['candidates']
    if (not assets or len({r['path'] for r in assets})!=len(assets)
            or len(families)!=len(assets) or len({r['path'] for r in families})!=len(families)
            or {r['path'] for r in assets}!={r['path'] for r in families}
            or comparison['candidate_count']!=len(assets)
            or comparison['all_cached_enabled_collision_paths_retained'] is not True
            or comparison['maximum_cached_vs_actual_union_AABB_difference_m']!=0.):
        raise ValueError('Original complete medical roster and exact captured comparison required')
    roots=[r['path'] for r in assets]
    if any(not root.startswith('/World/Hospital/') or root.count('/')!=3 for root in roots):
        raise ValueError('Exact original Hospital asset roots required, never arbitrary map exclusions')
    families={r['path']:r for r in families}
    raster=grid['raster'];shape=tuple(raster['shape_rows_columns']);origin=vector(raster['origin_xy_m'],2,'raster origin')
    if (len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape)
            or shape[0]*shape[1]>MAX_CELLS or raster['resolution_m']!=RESOLUTION_M
            or grid['original_height_slab_m']!=list(SLAB_M)):
        raise ValueError('Same bounded source raster and original obstacle slab required')
    roi=vector(grid['analysis_roi_xy_m'],4,'source ROI');low=roi[[0,2]];high=roi[[1,3]]
    mask=np.zeros(shape,bool);records=[]
    allowed={'bed_proxy_name_match','cart_name_match','gas_cart','gurney','hospital_bed','supply_cart','wheelchair'}
    for asset in assets:
        root=asset['path'];family=families[root]
        if (asset['active'] is not True or asset['loaded'] is not True or asset['unloaded_payload_paths']!=[]
                or family['name_based_family'] not in allowed or family['same_cached_enabled_collision_paths'] is not True
                or family['maximum_cached_vs_actual_union_AABB_difference_m']!=0.):
            raise ValueError('Original loaded medical asset/family readback required: '+root)
        current=[]
        for row in rows:
            if not row['path'].startswith(root+'/'):continue
            if type(row['is_active']) is not bool or type(row['collision_enabled_value']) is not bool:
                raise ValueError('Exact captured active/collision flags required')
            if row['is_active'] and row['collision_enabled_value']:
                if row['classification'] not in ('outside_ROI','enabled_in_ROI'):
                    raise ValueError('Medical asset cannot be excluded as robot or disabled')
                current.append(row)
        saved=asset['enabled_collision_prims']
        if (not current or len({r['path'] for r in saved})!=len(saved)
                or {r['path'] for r in current}!={r['path'] for r in saved}
                or family['actual_enabled_collision_count']!=len(current)
                or family['cached_enabled_collision_count']!=len(current)):
            raise ValueError('Every enabled medical collider must match both recorded sources: '+root)
        by_path={r['path']:r for r in saved};minimum=[];maximum=[]
        for row in current:
            bound=row['world_aligned_bound'];other=by_path[row['path']]['world_aligned_bound']
            lo=vector(bound['minimum_m'],3,'medical primitive minimum');hi=vector(bound['maximum_m'],3,'medical primitive maximum')
            if (np.any(hi<=lo) or not np.array_equal(lo,vector(other['minimum_m'],3,'second minimum'))
                    or not np.array_equal(hi,vector(other['maximum_m'],3,'second maximum'))):
                raise ValueError('Recorded medical bounds differ; never fill using stale/cached substitutes')
            minimum.append(lo);maximum.append(hi)
        lo=np.min(minimum,axis=0);hi=np.max(maximum,axis=0);bound=family['actual_union_AABB']
        if (not np.array_equal(lo,vector(bound['minimum_m'],3,'captured asset minimum'))
                or not np.array_equal(hi,vector(bound['maximum_m'],3,'captured asset maximum'))):
            raise ValueError('All-section asset union must equal the actual captured union')
        intersects=not (hi[2]<SLAB_M[0] or lo[2]>SLAB_M[1] or np.any(hi[:2]<low) or np.any(lo[:2]>high))
        if intersects:
            rr,cc=polygon_cells([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],origin,RESOLUTION_M,shape)
            mask[rr,cc]=True
        records.append(dict(path=root,name_based_family=family['name_based_family'],
            enabled_collider_paths=sorted(r['path'] for r in current),
            recorded_full_collision_union_bound=dict(minimum_m=lo.tolist(),maximum_m=hi.tolist()),
            kept_as_whole_asset_in_analysis_ROI=intersects,
            union_box_NOT_a_new_visual_silhouette_or_cooked_shape=True))
    return dict(version=MEDICAL_KEEPOUT_VERSION,all_original_medical_roots_verified=len(records),
        in_ROI_full_asset_keepouts=sum(r['kept_as_whole_asset_in_analysis_ROI'] for r in records),assets=records,
        collision_mesh_leg_gaps_NOT_permission_to_drive_under_furniture=True,
        same_original_10mm_robot_footprint_clearance_NO_added_asset_inflation=True,
        same_map_policy_required_for_all_comparison_methods=True,
        no_asset_collider_flag_pose_material_support_or_scene_change=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False),mask


def export_medical_keepout_review(coverage_path,projection_path,asset_receipt_path):
    """Offline map-policy review; never rewrite an earlier H4/H5/H6 result."""
    import io,os,zipfile
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    coverage_path=Path(coverage_path).resolve();projection_path=Path(projection_path).resolve()
    asset_path=Path(asset_receipt_path).resolve()
    coverage=json.loads(coverage_path.read_bytes());projection=json.loads(projection_path.read_bytes());assets=json.loads(asset_path.read_bytes())
    pins={}
    for path,record in ((coverage_path,coverage),(projection_path,projection),(asset_path,assets)):
        for name,value in {**record['input_sha256'],**record.get('output_sha256',{}),str(path):sha(path)}.items():
            if name in pins and pins[name]!=value:raise ValueError('Conflicting original evidence pin')
            pins[name]=value
    for path,value in pins.items():
        if sha(path)!=value:raise ValueError('Original saved source changed: '+path)
    if (projection['source_coverage']!=str(coverage_path)
            or assets['all_4997_protected_environment_rows']!=coverage['contract']['source_4997_row_environment']):
        raise ValueError('Same recorded environment/mesh-projection lineage required')
    raw_path=Path(coverage['contract']['all_stage_receipt'])
    if pins.get(str(raw_path))!=sha(raw_path):raise ValueError('Original complete-inventory receipt must be pinned')
    raw=json.loads(raw_path.read_bytes())
    if (raw['all_4997_protected_environment_rows']!=assets['all_4997_protected_environment_rows']
            or raw['retained_result']!=assets['retained_result']):raise ValueError('Same actual retained environment required')
    report,keepout=medical_asset_keepouts(raw['inventory'],assets,coverage['coverage_grid'])
    raster=next(Path(p) for p in projection['output_sha256'] if p.endswith('mesh_refined_grid.npz'))
    with np.load(raster) as arrays:
        old_blocked=arrays['occupied'].copy();supported=arrays['supported_full_cells'].copy()
        old_obstacles=arrays['obstacle_mesh_projection'].copy()
    obstacles=old_obstacles|keepout;blocked=old_blocked|keepout
    controls=task_anchor_review(coverage['contract']['anchors'],coverage['coverage_grid'],blocked,
        coverage['contract']['same_three_source_support_rectangles'])
    source_files=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in source_files:
            member=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));member.external_attr=0o100644<<16
            archive.writestr(member,path.read_bytes())
    source_bytes=buffer.getvalue()
    contract=dict(version=MEDICAL_KEEPOUT_VERSION,source_coverage=str(coverage_path),source_projection=str(projection_path),
        original_asset_receipt=str(asset_path),original_environment_digest=assets['all_4997_protected_environment_rows'],
        source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),input_sha256=pins,
        no_new_scene_geometry_map_adoption_or_motion_authority=True)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/medical_keepouts'/key;target=folder/'medical_keepout_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if saved['contract']!=contract:raise ValueError('Prior medical-keepout contract changed; no overwrite')
        for path,value in saved['output_sha256'].items():
            if sha(path)!=value:raise ValueError('Prior medical-keepout output changed; no overwrite')
        return target
    if folder.exists():raise ValueError('Incomplete medical review retained; inspect rather than automatically overwrite')
    folder.mkdir(parents=True,exist_ok=False)
    with (folder/'executed_sources.zip').open('xb') as stream:stream.write(source_bytes);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    result_grid=folder/'medical_keepout_grid.npz'
    np.savez_compressed(result_grid,occupied=blocked,supported_full_cells=supported,
        medical_asset_keepouts=keepout,obstacle_mesh_and_medical_keepouts=obstacles,
        origin_xy_m=coverage['coverage_grid']['raster']['origin_xy_m'],resolution_m=RESOLUTION_M)
    for path,value in pins.items():
        if sha(path)!=value:raise ValueError('Source changed during medical-keepout review: '+path)
    result=dict(contract=contract,medical_asset_policy=report,
        newly_blocked_furniture_gap_cells=int(np.count_nonzero(keepout&~old_obstacles)),
        obstacle_cells_before=int(old_obstacles.sum()),obstacle_cells_after=int(obstacles.sum()),
        source_control_review=controls,
        all_hospital_bed_roots_retained=[r for r in report['assets'] if r['name_based_family']=='hospital_bed'],
        H6_departure_arrival_room_or_goal_NOT_selected_from_method_outcomes=True,
        earlier_projection_is_geometry_ONLY_not_permission_to_drive_under_furniture=True,
        prior_H4_H5_actual_trials_and_maps_NOT_relabelled=True,
        status='MEDICAL_ENVELOPE_MAP_POLICY_REVIEWED_H6_TASK_AND_FULL_BODY_ROUTE_PENDING',
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,output_sha256={str(p):sha(p) for p in (folder/'executed_sources.zip',folder/'software_contract.json',result_grid)})
    evidence.write_exclusive(target,result)
    return target


BODY_SWEEP_VERSION='H6_CARDINAL_RECTANGLE_TRANSLATION_AND_CLEARED_DISK_ROTATION_GEOMETRY_ONLY_v1'


def cardinal_body_sweep_masks(blocked,resolution):
    """Sufficient GEOMETRIC connectivity, not steering/dynamics feasibility.

    Rectangle translation is limited to axis-aligned orientations and four
    neighboring cell centers. For those translations the continuous swept
    rectangle is the union of the two overlapping endpoint rectangles. A yaw
    transition is permitted only where its entire circumscribed disk is clear.
    No in-place rotation or lateral translation is admitted as a robot command.
    """
    from scipy.ndimage import maximum_filter,distance_transform_edt
    if (not isinstance(blocked,np.ndarray) or blocked.dtype!=np.dtype(bool)
            or blocked.ndim!=2 or not blocked.size or blocked.size>MAX_CELLS
            or type(resolution) not in (int,float) or not math.isfinite(resolution) or resolution<=0.):
        raise ValueError('Bounded original boolean raster and finite positive resolution required')
    # Exactly footprint_polygon defaults; no physical/map safety change.
    half_length,half_width=1.630/2+.010,.965/2+.010
    if resolution>2*half_width:
        raise ValueError('Translation step must not exceed the shorter rectangle side')
    offsets=np.floor((np.array([half_length,half_width])+resolution/2+1e-12)/resolution).astype(int)
    sizes=2*offsets+1
    # Outside the measured raster is occupied, never an artificial shortcut.
    free_x=~maximum_filter(blocked,size=(int(sizes[1]),int(sizes[0])),mode='constant',cval=True)
    free_y=~maximum_filter(blocked,size=(int(sizes[0]),int(sizes[1])),mode='constant',cval=True)
    distance=distance_transform_edt(np.pad(~blocked,1,constant_values=False),sampling=resolution)[1:-1,1:-1]
    radius=math.hypot(half_length,half_width)
    # Distances are to occupied CELL CENTERS. The half-cell diagonal bounds
    # all points in each occupied square. This is conservative discretization
    # certification, not a changed physical clearance or actual grid inflation.
    rotatable=(distance>radius+resolution/math.sqrt(2)+1e-12)&free_x&free_y
    return dict(free_x=free_x,free_y=free_y,rotatable=rotatable,
        rectangle_half_length_m=half_length,rectangle_half_width_m=half_width,
        rotation_disk_radius_m=radius,occupied_cell_center_certificate_padding_m=resolution/math.sqrt(2),
        four_neighbor_translation_step_m=float(resolution),same_10mm_footprint_clearance=True,
        geometry_can_translate_sideways_or_rotate_in_place_NOT_a_robot_control_permission=True)


def _cardinal_orientation(yaw):
    if not math.isfinite(yaw):raise ValueError('Finite source heading required')
    heading=math.atan2(math.sin(yaw),math.cos(yaw))
    options=[(abs(math.atan2(math.sin(heading-k*math.pi/2),math.cos(heading-k*math.pi/2))),k%2)
             for k in range(-2,3)]
    difference,orientation=min(options)
    if difference>1e-12:
        raise ValueError('This restricted geometry screen does not snap a non-cardinal source heading')
    return orientation


def optimistic_inscribed_disk_review(anchors,blocked,origin,resolution):
    """A necessary-condition OUTER approximation, within this raster only.

    Any oriented rectangle contains its width/2+clearance disk. For each grid
    cell, D(center)+half_cell_diagonal bounds ABOVE the nearest occupied-cell
    center distance anywhere in that cell (the distance function is 1-Lipschitz).
    Keep the WHOLE cell whenever that optimistic bound reaches the disk radius.
    Use EIGHT neighbors, including corner transitions. If even this larger
    center-space is disconnected, the rectangular body cannot connect those
    poses within this full-cell occupancy policy. A positive result is only a
    necessary condition, and no result proves global/physical infeasibility.
    """
    from scipy.ndimage import distance_transform_edt
    # Shared validation, no altered footprint or silently coarsened map.
    cardinal_body_sweep_masks(blocked,resolution)
    origin=vector(origin,2,'original raster origin')
    d=distance_transform_edt(np.pad(~blocked,1,constant_values=False),sampling=resolution)[1:-1,1:-1]
    radius=.965/2+.010
    optimistic=d+resolution/math.sqrt(2)>=radius-1e-12
    components,count=label(optimistic,structure=np.ones((3,3),int))
    rows=[]
    for anchor in anchors:
        pose=vector(anchor['world_xyyaw'],3,'original necessary-condition control')
        x,y=np.floor((pose[:2]-origin)/resolution).astype(int)
        inside=0<=y<blocked.shape[0] and 0<=x<blocked.shape[1]
        rows.append(dict(id=anchor['id'],world_xyyaw=pose.tolist(),
            optimistic_inscribed_disk_component=int(components[y,x]) if inside else 0))
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            connected=a['optimistic_inscribed_disk_component']!=0 and (
                a['optimistic_inscribed_disk_component']==b['optimistic_inscribed_disk_component'])
            pairs.append(dict(first=a['id'],second=b['id'],necessary_optimistic_disk_connection=connected,
                disconnected_implies_no_continuous_full_rectangle_center_path_IN_THIS_RASTER_POLICY=not connected,
                NOT_global_physical_or_multisteering_planner_infeasibility=True))
    return dict(inscribed_disk_radius_m=radius,optimistic_cell_distance_upper_bound_padding_m=resolution/math.sqrt(2),
        eight_neighbor_cell_connections=True,component_count=int(count),anchors=rows,all_unordered_anchor_pairs=pairs,
        point_obstacle_centers_and_optimistic_cells_are_a_relaxation_NOT_an_inflated_blocked_map=True,
        conclusions_limited_to_recorded_full_cell_occupancy_unknown_support_and_analysis_ROI=True),dict(
            optimistic_disk_cells=optimistic,optimistic_disk_components=components)


def translation_sweep_free(blocked,first,last,origin,resolution):
    """Exact convex swept polygon for unchanged body yaw on a straight line."""
    from scipy.spatial import ConvexHull
    first=vector(first,3,'source sweep pose');last=vector(last,3,'target sweep pose')
    if first[2]!=last[2]:raise ValueError('Straight translation sweep may not alter source yaw')
    points=np.vstack([footprint_polygon(first),footprint_polygon(last)])
    polygon=points[ConvexHull(points).vertices]
    high=origin+np.array(blocked.shape[::-1])*resolution
    if np.any(polygon<origin) or np.any(polygon>high):return False
    rows,columns=polygon_cells(polygon,origin,resolution,blocked.shape)
    return bool(len(rows) and not blocked[rows,columns].any())


def cardinal_body_connection_review(anchors,blocked,origin,resolution):
    """All pairs of unchanged original controls; no H6 waypoint selection."""
    if (not anchors or len({a['id'] for a in anchors})!=len(anchors)
            or any(not isinstance(a['id'],str) or not a['id'] for a in anchors)):
        raise ValueError('All explicitly sourced unique original controls required')
    origin=vector(origin,2,'raster origin')
    masks=cardinal_body_sweep_masks(blocked,resolution)
    structure=np.array([[0,1,0],[1,1,1],[0,1,0]])
    lx,nx=label(masks['free_x'],structure=structure);ly,ny=label(masks['free_y'],structure=structure)
    parent=list(range(nx+ny+1))
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    rotation_pairs=np.unique(np.column_stack([lx[masks['rotatable']],ly[masks['rotatable']]]),axis=0)
    if any(a==0 or b==0 for a,b in rotation_pairs):raise ValueError('Rotation must retain both endpoint rectangles')
    for a,b in rotation_pairs:
        a,b=find(int(a)),find(nx+int(b));parent[max(a,b)]=min(a,b)
    lut=np.array([find(i) for i in range(len(parent))],np.int32)
    cx=lut[lx];cy=np.where(ly>0,lut[ly+nx],0)
    rows=[]
    for anchor in anchors:
        pose=vector(anchor['world_xyyaw'],3,'unchanged original task control')
        orientation=_cardinal_orientation(float(pose[2]));components=cx if orientation==0 else cy
        center=np.floor((pose[:2]-origin)/resolution).astype(int)
        source_clear=not pose_collision(blocked,pose,origin,resolution)
        connectors=[]
        # Every nearby candidate is examined, not a favorable single choice.
        for iy in range(int(center[1])-1,int(center[1])+2):
            for ix in range(int(center[0])-1,int(center[0])+2):
                if not (0<=iy<blocked.shape[0] and 0<=ix<blocked.shape[1]):continue
                component=int(components[iy,ix])
                if not source_clear or component==0:continue
                target=[float(origin[0]+(ix+.5)*resolution),float(origin[1]+(iy+.5)*resolution),float(pose[2])]
                if (pose_collision(blocked,target,origin,resolution)
                        or not translation_sweep_free(blocked,pose,target,origin,resolution)):continue
                connectors.append(dict(cell_row_column=[iy,ix],unchanged_yaw_target_pose=target,
                    sufficient_body_sweep_component=component,continuous_straight_connector_verified=True))
        reviewed=dict(anchor)
        reviewed.update(source_pose_NOT_changed=True,source_full_footprint_clear=source_clear,
            cardinal_rectangle_orientation=orientation,verified_local_connectors=connectors,
            sufficient_body_sweep_components=sorted({c['sufficient_body_sweep_component'] for c in connectors}))
        rows.append(reviewed)
    pairs=[]
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            common=sorted(set(left['sufficient_body_sweep_components'])&set(right['sufficient_body_sweep_components']))
            pairs.append(dict(first=left['id'],second=right['id'],sufficient_GEOMETRIC_body_sweep_connection=bool(common),
                common_geometry_components=common,
                negative_result_NOT_full_orientation_continuous_geometric_infeasibility=True,
                positive_result_NOT_multisteering_force_caster_control_or_PhysX_validation=True))
    report=dict(version=BODY_SWEEP_VERSION,anchors=rows,all_unordered_anchor_pairs=pairs,
        original_map_resolution_m=float(resolution),fixed_orientation_component_counts=[int(nx),int(ny)],
        cleared_disk_rotation_cell_count=int(masks['rotatable'].sum()),
        rotation_orientation_component_joins=rotation_pairs.tolist(),
        mask_policy={k:v for k,v in masks.items() if not isinstance(v,np.ndarray)},
        all_source_controls_and_pairs_examined_not_success_selected=True,
        restricted_axis_aligned_holonomic_geometric_certificate_NOT_a_robot_route=True,
        continuous_steering_and_dynamic_feasibility_pending=True,
        H6_room_start_goal_map_or_motion_NOT_adopted=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False)
    return evidence.json_value(report),dict(free_x=masks['free_x'],free_y=masks['free_y'],
        rotatable=masks['rotatable'],geometry_components_x=cx,geometry_components_y=cy)


def export_body_sweep_review(medical_review_path):
    """Preserve the original H5 map/results; no H6 task or scene is adopted."""
    import io,os,zipfile
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    path=Path(medical_review_path).resolve();original=json.loads(path.read_bytes())
    pins={**original['input_sha256'],**original['output_sha256'],str(path):sha(path)}
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Original medical map evidence changed: '+name)
    raster=next(Path(name) for name in original['output_sha256'] if name.endswith('medical_keepout_grid.npz'))
    with np.load(raster) as a:
        blocked=a['occupied'].copy();origin=a['origin_xy_m'].copy();resolution=float(a['resolution_m'])
    if resolution!=RESOLUTION_M:raise ValueError('Original10mm raster required, no coarsening')
    anchors=original['source_control_review']['anchors']
    sufficient,arrays=cardinal_body_connection_review(anchors,blocked,origin,resolution)
    necessary,outer=optimistic_inscribed_disk_review(anchors,blocked,origin,resolution)
    for positive,outer_pair in zip(sufficient['all_unordered_anchor_pairs'],necessary['all_unordered_anchor_pairs']):
        if (positive['first'],positive['second'])!=(outer_pair['first'],outer_pair['second']):
            raise ValueError('All original pair identities/order must remain exact')
        if positive['sufficient_GEOMETRIC_body_sweep_connection'] and not outer_pair['necessary_optimistic_disk_connection']:
            raise ValueError('Contradictory sufficient/necessary geometry certificates')
    sources=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for item in sources:
            info=zipfile.ZipInfo(str(item.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,item.read_bytes())
    source_bytes=buffer.getvalue()
    contract=dict(version=BODY_SWEEP_VERSION,original_medical_review=str(path),input_sha256=pins,
        same_environment=original['contract']['original_environment_digest'],source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),
        all_ten_original_task_controls_and_45_pairs=True,no_new_H6_task_map_or_motion_adoption=True)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/body_sweeps'/key;target=folder/'body_sweep_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if saved['contract']!=contract:raise ValueError('Prior body-sweep review changed')
        for name,value in saved['output_sha256'].items():
            if sha(name)!=value:raise ValueError('Prior geometry output changed: '+name)
        return target
    if folder.exists():raise ValueError('Incomplete prior geometry export retained; do not overwrite')
    folder.mkdir(parents=True)
    with (folder/'executed_sources.zip').open('xb') as stream:stream.write(source_bytes);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    grid=folder/'body_sweep_certificates.npz'
    np.savez_compressed(grid,**arrays,**outer,origin_xy_m=origin,resolution_m=resolution)
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Source changed during body-sweep export: '+name)
    result=dict(contract=contract,sufficient_body_geometry=sufficient,necessary_outer_approximation=necessary,
        positive_sufficient_pair_count=sum(p['sufficient_GEOMETRIC_body_sweep_connection'] for p in sufficient['all_unordered_anchor_pairs']),
        negative_necessary_pair_count=sum(not p['necessary_optimistic_disk_connection'] for p in necessary['all_unordered_anchor_pairs']),
        status='BOUNDED_RECORDED_RASTER_FULL_RECTANGLE_GEOMETRY_ONLY_NOT_H6_TASK_OR_ROBOT_ROUTE',
        isolated_door_threshold_or_actual_collision_cause_NOT_proved=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,output_sha256={str(name):sha(name) for name in (folder/'executed_sources.zip',folder/'software_contract.json',grid)})
    evidence.write_exclusive(target,result)
    return target


def recorded_obstacle_partition_masks(coverage,bundle,array_store,root_groups):
    """Diagnostic copies only; project every original collider exactly once.

    Never subtract a target mask from an already-unioned raster: that would
    incorrectly erase overlapping obstacles. Instead partition ORIGINAL
    collider identities, project each disjoint partition and retain their OR.
    This helper cannot modify a stage, flags, support or a planning result.
    """
    if (not isinstance(root_groups,dict) or not root_groups or 'retained' in root_groups
            or any(not isinstance(k,str) or not k for k in root_groups)
            or any(not isinstance(v,str) or not v.startswith('/World/Hospital/') or v.endswith('/')
                   for v in root_groups.values()) or len(set(root_groups.values()))!=len(root_groups)):
        raise ValueError('Distinct explicitly declared original Hospital roots required')
    included=coverage['coverage_grid']['included_obstacles'];records=bundle['primitives']
    if ({r['path'] for r in records}!={r['path'] for r in included}
            or len(records)!=len(included) or len({r['path'] for r in records})!=len(records)):
        raise ValueError('All original unique collider identities required before partitioning')
    assignments={key:[] for key in ('retained',*root_groups)}
    for row in included:
        matches=[key for key,root in root_groups.items() if row['path']==root or row['path'].startswith(root+'/')]
        if len(matches)>1:raise ValueError('Overlapping diagnostic roots cannot count a collider twice')
        assignments[matches[0] if matches else 'retained'].append(row['path'])
    if any(not assignments[key] for key in root_groups):
        raise ValueError('Each predeclared diagnostic root must exist in original included colliders')
    partitions={};reports={}
    for key,paths in assignments.items():
        chosen=set(paths)
        part=dict(coverage,coverage_grid=dict(coverage['coverage_grid'],
            included_obstacles=[row for row in included if row['path'] in chosen]))
        selected=dict(bundle,primitives=[row for row in records if row['path'] in chosen])
        report,_,_,obstacles=refine_recorded_mesh_grid(part,selected,array_store)
        partitions[key]=obstacles;reports[key]=dict(original_collider_paths=paths,projection_review=report)
    return dict(groups=root_groups,partitions=reports,all_original_colliders_accounted_for_once=len(included),
        whole_union_subtraction_NOT_used=True,scene_flags_materials_support_and_task_NOT_modified=True),partitions


def diagnostic_two_root_factorial(anchors,partitions,base_blocked,baseline,origin,resolution):
    """Predeclared 2x2 COPY-raster attribution, NOT four experiment scenarios."""
    if set(partitions)!={'retained','door','threshold'}:
        raise ValueError('Exactly predeclared door and threshold partitions required')
    arrays=[baseline,base_blocked,*partitions.values()]
    if any(not isinstance(a,np.ndarray) or a.dtype!=np.dtype(bool) or a.shape!=baseline.shape for a in arrays):
        raise ValueError('Exact same original boolean raster shape required')
    reproduced=base_blocked|partitions['retained']|partitions['door']|partitions['threshold']
    if not np.array_equal(reproduced,baseline):
        raise ValueError('Unmodified four-part OR must reproduce original medical raster exactly')
    results=[];grids={}
    for keep_door,keep_threshold in ((True,True),(True,False),(False,True),(False,False)):
        case='door_%d_threshold_%d'%(keep_door,keep_threshold)
        blocked=base_blocked|partitions['retained']
        if keep_door:blocked=blocked|partitions['door']
        if keep_threshold:blocked=blocked|partitions['threshold']
        sufficient,_=cardinal_body_connection_review(anchors,blocked,origin,resolution)
        necessary,_=optimistic_inscribed_disk_review(anchors,blocked,origin,resolution)
        if len(sufficient['all_unordered_anchor_pairs'])!=len(necessary['all_unordered_anchor_pairs']):
            raise ValueError('Original pair coverage/order cannot change across diagnostics')
        for s,n in zip(sufficient['all_unordered_anchor_pairs'],necessary['all_unordered_anchor_pairs']):
            if (s['first'],s['second'])!=(n['first'],n['second']):raise ValueError('Original pair identity changed')
            if s['sufficient_GEOMETRIC_body_sweep_connection'] and not n['necessary_optimistic_disk_connection']:
                raise ValueError('Contradictory sufficient and necessary copy-raster certificates')
        results.append(dict(case=case,original_door_contribution_retained_in_COPY=keep_door,
            original_threshold_contribution_retained_in_COPY=keep_threshold,
            blocked_cells=int(blocked.sum()),removed_COPY_cells=int(np.count_nonzero(baseline&~blocked)),
            all_unchanged_anchor_poses=[dict(id=a['id'],world_xyyaw=a['world_xyyaw']) for a in anchors],
            sufficient_geometric_pair_count=sum(p['sufficient_GEOMETRIC_body_sweep_connection'] for p in sufficient['all_unordered_anchor_pairs']),
            negative_necessary_pair_count=sum(not p['necessary_optimistic_disk_connection'] for p in necessary['all_unordered_anchor_pairs']),
            sufficient_body_geometry=sufficient,necessary_outer_approximation=necessary,
            removed_door_COPY_is_NOT_a_physically_open_door_or_adopted_setup=True,
            every_comparison_is_geometry_ONLY_no_actual_method_trial_or_safety_admission=True))
        grids[case]=blocked
    return dict(cases=results,original_baseline_reproduced_exactly=True,
        all_other_colliders_medical_keepouts_support_unknown_space_and_resolution_preserved=True,
        root_choice_must_come_from_prior_H3_H1_conditions_NOT_P_outcomes=True,
        environment_adoption_or_actual_door_threshold_cause_NOT_proved=True,
        H6_control_and_task_NOT_frozen=True,new_PhysX_trial_count=0,actual_motion_authorized=False),grids


def export_root_attribution_review(body_review_path):
    """Record a fixed source-root attribution, never select a new environment."""
    import io,os,zipfile
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    body_path=Path(body_review_path).resolve();body=json.loads(body_path.read_bytes())
    medical_path=Path(body['contract']['original_medical_review']);medical=json.loads(medical_path.read_bytes())
    coverage_path=Path(medical['contract']['source_coverage']);coverage=json.loads(coverage_path.read_bytes())
    projection_path=Path(medical['contract']['source_projection']);projection=json.loads(projection_path.read_bytes())
    bundle_path=Path(projection['mesh_bundle']);bundle=json.loads(bundle_path.read_bytes())
    pins={}
    for path,record in ((body_path,body),(medical_path,medical),(coverage_path,coverage),
                        (projection_path,projection),(bundle_path,bundle)):
        for name,value in {**record['input_sha256'],**record.get('output_sha256',{}),str(path):sha(path)}.items():
            if name in pins and pins[name]!=value:raise ValueError('Conflicting geometry source pin: '+name)
            pins[name]=value
    h1_source=Path(next(a['source'] for a in coverage['contract']['anchors'] if a['id']=='H1_start'))
    h1=json.loads(h1_source.read_bytes());h1_geometry=Path(h1['geometry_manifest'])
    if pins.get(str(h1_source))!=sha(h1_source) or pins.get(str(h1_geometry))!=sha(h1_geometry):
        raise ValueError('Exact prior H1 door definition must predate diagnostic outcomes')
    h1_manifest=json.loads(h1_geometry.read_bytes())
    groups=dict(door=h1_manifest['door_state']['path'],
        threshold=coverage['H3_environment_comparison']['original_H3_suppressed_threshold_root'])
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Original geometry input changed: '+name)
    mesh_arrays=next(Path(p) for p in bundle['output_sha256'] if p.endswith('recorded_world_mesh_arrays.npz'))
    with np.load(mesh_arrays) as source:
        partition_review,partitions=recorded_obstacle_partition_masks(coverage,bundle,source,groups)
    original_grid=next(Path(p) for p in medical['output_sha256'] if p.endswith('medical_keepout_grid.npz'))
    refined_grid=next(Path(p) for p in projection['output_sha256'] if p.endswith('mesh_refined_grid.npz'))
    with np.load(original_grid) as a:
        baseline=a['occupied'].copy();keepouts=a['medical_asset_keepouts'].copy()
        origin=a['origin_xy_m'].copy();resolution=float(a['resolution_m'])
    with np.load(refined_grid) as a:
        # Keep furniture even where it overlaps an original door/mesh mask.
        base=fixed_cell_blockers(coverage['coverage_grid'],a['supported_full_cells'])|keepouts
    anchors=medical['source_control_review']['anchors']
    review,grids=diagnostic_two_root_factorial(anchors,partitions,base,baseline,origin,resolution)
    first=review['cases'][0]
    if (first['sufficient_body_geometry']!=body['sufficient_body_geometry']
            or first['necessary_outer_approximation']!=body['necessary_outer_approximation']):
        raise ValueError('Unmodified baseline must reproduce the previous full-body review exactly')
    sources=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for path in sources:
            info=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,path.read_bytes())
    source_bytes=buffer.getvalue()
    contract=dict(version='H6_PREDECLARED_H1_DOOR_H3_THRESHOLD_COPY_RASTER_ATTRIBUTION_v1',
        source_body_review=str(body_path),source_roots=groups,root_source_H1_geometry=str(h1_geometry),
        root_source_H3_condition_in_coverage=str(coverage_path),input_sha256=pins,
        source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),
        no_planning_map_replacement_task_adoption_or_actual_obstacle_flag_change=True,
        no_inference_that_removing_door_is_recreating_original_H3_open_door=True)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/root_attribution'/key;target=folder/'diagnostic_root_attribution.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if saved['contract']!=contract:raise ValueError('Existing diagnostic contract changed')
        for name,value in saved['output_sha256'].items():
            if sha(name)!=value:raise ValueError('Prior copy-only diagnostic output changed: '+name)
        return target
    if folder.exists():raise ValueError('Incomplete existing diagnostic retained, no overwrite')
    folder.mkdir(parents=True)
    with (folder/'executed_sources.zip').open('xb') as stream:stream.write(source_bytes);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    grid=folder/'DIAGNOSTIC_COPIES_NOT_PLANNING_MAPS.npz'
    np.savez_compressed(grid,**grids,**{k+'_contribution':v for k,v in partitions.items()},base_blocked=base,
        origin_xy_m=origin,resolution_m=resolution)
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Source changed during diagnostic export: '+name)
    result=dict(contract=contract,original_collider_partition_review=partition_review,copy_only_factorial=review,
        status='MAP_CONTRIBUTION_ATTRIBUTION_ONLY_NOT_FOUR_TRIALS_OPEN_DOOR_RECONSTRUCTION_OR_H6_ADMISSION',
        conclusions_do_NOT_identify_actual_contact_force_or_isolated_physical_cause=True,
        original_H3_H4_H5_results_maps_and_all_physical_flags_preserved=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,output_sha256={str(p):sha(p) for p in (folder/'executed_sources.zip',folder/'software_contract.json',grid)})
    evidence.write_exclusive(target,result)
    return target


def historical_H3_open_door_mesh(current_record,array_store,historical_leaf,*,recorded_specification=None):
    """Use ORIGINAL recorded H3 vertices, not a deleted or AABB-fitted door.

    The historical hinge recipe is independently checked against every vertex.
    Its angle is predeclared by the old source helper, never selected by path
    outcomes. This is only a retrospective source-geometry check, not an SDK
    instruction or proof of today's actual door state.
    """
    import hospital_h3_door_state as door
    root=door.ASSETS['leaf'];collision=historical_leaf['collision']
    if (historical_leaf['path']!=root or collision['collision_prim_count']!=1
            or len(collision['primitives'])!=1 or not current_record['path'].startswith(root+'/')
            or current_record['geometry']['projection']!='AUTHORED_STATIC_TRIANGLE_SLAB'
            or current_record['geometry']['recorded_physics_approximation']!='none'):
        raise ValueError('Exact original single static door mesh identities/approximation required')
    primitive=collision['primitives'][0];mesh=primitive['authored_mesh']
    if primitive['path']!=current_record['path'] or collision['collision_prim_paths']!=[primitive['path']]:
        raise ValueError('Historical and recorded current door collider identities must remain equal')
    prefix=current_record['array_prefix']
    points=np.asarray(array_store[prefix+'_points']);counts=np.asarray(array_store[prefix+'_counts'])
    indices=np.asarray(array_store[prefix+'_indices']);target=np.asarray(mesh['world_points_m'],float)
    if (points.ndim!=2 or points.shape[1]!=3 or not len(points) or target.shape!=points.shape
            or not np.isfinite(points).all() or not np.isfinite(target).all()
            or counts.ndim!=1 or indices.ndim!=1
            or not np.issubdtype(counts.dtype,np.integer) or not np.issubdtype(indices.dtype,np.integer)
            or not np.array_equal(counts,mesh['face_vertex_counts'])
            or not np.array_equal(indices,mesh['face_vertex_indices'])
            or not np.all(counts==3) or len(indices)!=3*len(counts)
            or np.any(indices<0) or np.any(indices>=len(points)) or mesh['physics_approximation']!='none'):
        raise ValueError('Same finite indexed triangle topology/approximation required; no mesh substitution')
    specification=door.opening_specification(door.OBSERVED_ORIGIN_M,door.OBSERVED_AXES,
        door.OBSERVED_LOCAL_MIN,door.OBSERVED_LOCAL_MAX)
    if recorded_specification is not None and specification!=recorded_specification:
        raise ValueError('Original recorded hinge specification changed; never select a new angle')
    predicted=np.asarray(door.rotate_world_points(points,specification))
    residual=float(np.max(np.abs(predicted-target)))
    if residual>2e-5:raise ValueError('Original H3 door vertices do not reproduce the predeclared hinge rotation')
    lo=vector(primitive['world_aligned_bound']['minimum_m'],3,'historical door minimum')
    hi=vector(primitive['world_aligned_bound']['maximum_m'],3,'historical door maximum')
    if np.any(lo>hi) or np.any(target.min(axis=0)<lo-1e-6) or np.any(target.max(axis=0)>hi+1e-6):
        raise ValueError('Historical vertices leave the recorded door bound; no clipping or refit')
    report=dict(root=root,collider=primitive['path'],vertex_count=len(points),triangle_count=len(counts),
        identical_source_topology_and_collision_approximation=True,original_hinge_specification=specification,
        maximum_vertex_recipe_residual_m=residual,historical_geometry_consistency_tolerance_m=2e-5,
        exact_historical_world_vertices_used_NOT_fitted_or_AABB_rotated=True,
        new_door_angle_NOT_selected_by_path_or_method_outcomes=True,
        door_remains_an_obstacle_in_the_OFFLINE_candidate=True,
        no_current_scene_pose_read_or_changed=True)
    return report,mesh


def export_historical_H3_condition_candidate(attribution_path):
    """A source-faithful condition PROPOSAL, not a new map/task adoption."""
    import io,os,zipfile
    from hospital_geometry_tools import mesh_mask
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    attribution_path=Path(attribution_path).resolve();attribution=json.loads(attribution_path.read_bytes())
    body_path=Path(attribution['contract']['source_body_review']);body=json.loads(body_path.read_bytes())
    medical_path=Path(body['contract']['original_medical_review']);medical=json.loads(medical_path.read_bytes())
    coverage_path=Path(medical['contract']['source_coverage']);coverage=json.loads(coverage_path.read_bytes())
    projection_path=Path(medical['contract']['source_projection']);projection=json.loads(projection_path.read_bytes())
    bundle_path=Path(projection['mesh_bundle']);bundle=json.loads(bundle_path.read_bytes())
    # Select the actual original task lineage by BOTH original endpoints. Do
    # not select a later similarly named setup/flush receipt from a file glob.
    h3_path,h3=pinned_H3_geometry(coverage['input_sha256'],coverage['contract']['anchors'])
    historical_path=Path(h3['source_audit']);historical=json.loads(historical_path.read_bytes())
    if sha(historical_path)!=h3['source_audit_sha256']:
        raise ValueError('Original actual H3 source audit changed')
    setup_path=Path(h3['source_geometry_manifest'])
    if sha(setup_path)!=h3['source_geometry_manifest_sha256']:
        raise ValueError('Original H3 task-to-flush lineage changed')
    setup=json.loads(setup_path.read_bytes());open_path=Path(setup['fully_open_geometry_report'])
    if sha(open_path)!=setup['fully_open_geometry_sha256']:
        raise ValueError('Original fully-open geometry receipt changed')
    opened=json.loads(open_path.read_bytes())
    door_helper=_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h3_door_state.py')
    original_audit=_ijcas_path(ROOT, 'studies/IJCAS/scripts/audit_hospital_h3_doorway.py')
    if (sha(door_helper)!=opened['source_sha256']['hospital_h3_door_state.py']
            or sha(original_audit)!=opened['audit_script_sha256']
            or opened['door_state']!='fully_open_baseline_session_only'):
        raise ValueError('Exact historical hinge helper/asset definition required; no new geometry fit')
    roots=attribution['contract']['source_roots']
    leaves=[a for a in historical['local_assets'] if a['path']==roots['door']]
    current=[a for a in bundle['primitives'] if a['path']==roots['door'] or a['path'].startswith(roots['door']+'/')]
    if (len(leaves)!=1 or len(current)!=1 or h3['suppressed_threshold_root']!=roots['threshold']
            or h3['environment_condition']!='H3_flush_fully_open_continuous_floor'
            or h3['door_condition']!='fully_open_baseline_session_only'):
        raise ValueError('Exact original H3 open-door/flush condition and identities required')
    pins={**attribution['input_sha256'],**attribution['output_sha256'],str(attribution_path):sha(attribution_path),
        str(h3_path):sha(h3_path),str(historical_path):sha(historical_path)}
    for extra in (setup_path,open_path,door_helper,original_audit):
        if str(extra) in pins and pins[str(extra)]!=sha(extra):raise ValueError('Conflicting original door provenance')
        pins[str(extra)]=sha(extra)
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Original common-condition source changed: '+name)
    source_arrays=next(Path(name) for name in bundle['output_sha256'] if name.endswith('recorded_world_mesh_arrays.npz'))
    with np.load(source_arrays) as a:
        source_review,door_mesh=historical_H3_open_door_mesh(current[0],a,leaves[0],
            recorded_specification=opened['door_state_specification'])
    copies=next(Path(name) for name in attribution['output_sha256'] if name.endswith('DIAGNOSTIC_COPIES_NOT_PLANNING_MAPS.npz'))
    with np.load(copies) as a:
        base=a['base_blocked'].copy();retained=a['retained_contribution'].copy()
        origin=a['origin_xy_m'].copy();resolution=float(a['resolution_m'])
    if resolution!=RESOLUTION_M:raise ValueError('Original10mm raster required')
    # Door is retained at its EXACT historical world vertices. The original
    # H3 threshold policy is merely represented in this unused candidate.
    historical_door=mesh_mask(door_mesh,origin,resolution,base.shape,*SLAB_M)
    candidate=base|retained|historical_door
    if not historical_door.any():raise ValueError('The recorded original door must not disappear')
    sufficient,arrays=cardinal_body_connection_review(medical['source_control_review']['anchors'],candidate,origin,resolution)
    necessary,outer=optimistic_inscribed_disk_review(medical['source_control_review']['anchors'],candidate,origin,resolution)
    buffer=io.BytesIO();sources=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for path in sources:
            info=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,path.read_bytes())
    source_bytes=buffer.getvalue()
    contract=dict(version='H6_EXACT_RECORDED_H3_DOOR_AND_FLUSH_POLICY_OFFLINE_CONDITION_PROPOSAL_v1',
        original_task_lineage=str(h3_path),original_H3_door_audit=str(historical_path),
        original_recorded_open_door_specification_source=str(open_path),
        current_other_environment_source=str(attribution_path),source_roots=roots,input_sha256=pins,
        source_archive_sha256=hashlib.sha256(source_bytes).hexdigest(),
        all_other_current_obstacles_support_and_medical_envelopes_retained=True,
        entire_original_H3_scene_OR_physical_state_NOT_restored=True,
        current_map_scene_or_H6_task_adoption=False,actual_motion_authorized=False)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/historical_condition_candidate'/key;target=folder/'offline_condition_proposal.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if saved['contract']!=contract:raise ValueError('Historical-condition proposal changed')
        for name,value in saved['output_sha256'].items():
            if sha(name)!=value:raise ValueError('Prior offline proposal output changed: '+name)
        return target
    if folder.exists():raise ValueError('Incomplete prior proposal preserved; do not overwrite')
    folder.mkdir(parents=True)
    with (folder/'executed_sources.zip').open('xb') as stream:stream.write(source_bytes);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    grid=folder/'UNADOPTED_CONDITION_GEOMETRY.npz'
    np.savez_compressed(grid,occupied=candidate,historical_door=historical_door,base_blocked=base,retained_other_obstacles=retained,
        **arrays,**outer,origin_xy_m=origin,resolution_m=resolution)
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Source changed during condition proposal export: '+name)
    result=dict(contract=contract,original_door_source_review=source_review,sufficient_body_geometry=sufficient,
        necessary_outer_approximation=necessary,
        source_choice_is_original_H3_condition_NOT_a_favorable_new_door_angle=True,
        status='SOURCE_FAITHFUL_DOOR_AND_THRESHOLD_POLICY_PROPOSAL_NOT_ENVIRONMENT_OR_MISSION_ADOPTION',
        exact_whole_H3_environment_restoration_NOT_claimed=True,
        H6_room_goal_steering_dynamics_cooked_support_sensor_and_drive_validation_pending=True,
        original_actual_trials_and_current_H5_workflow_unchanged=True,
        new_PhysX_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        input_sha256=pins,output_sha256={str(p):sha(p) for p in (folder/'executed_sources.zip',folder/'software_contract.json',grid)})
    evidence.write_exclusive(target,result)
    return target


def compose_recorded_common_condition(base,partitions,historical_door,baseline):
    """Replace only authenticated source contributions, never a union mask.

    An unrelated obstacle or medical keepout overlapping the original leaf or
    threshold MUST survive. No additional room door/threshold is opened.
    """
    if set(partitions)!={'retained','door','threshold'}:
        raise ValueError('Exact original door/threshold/retained partition required')
    arrays=[base,baseline,historical_door,*partitions.values()]
    if (any(not isinstance(a,np.ndarray) or a.dtype!=np.dtype(bool) or a.ndim!=2 for a in arrays)
            or any(a.shape!=baseline.shape for a in arrays) or not baseline.size or baseline.size>MAX_CELLS):
        raise ValueError('Same bounded boolean condition raster required')
    if not np.array_equal(base|partitions['retained']|partitions['door']|partitions['threshold'],baseline):
        raise ValueError('Every original contribution must reproduce the full SOURCE baseline')
    if not historical_door.any() or not partitions['door'].any() or not partitions['threshold'].any():
        raise ValueError('Both original contributions and the retained historical door must be present')
    return base|partitions['retained']|historical_door


def export_selected_transfer_condition_review(port_review_path,condition_path):
    """Selected beds, ALL fixed port candidates and the adopted definition.

    Applies only the already adopted historical H3 leaf and flush policy to
    the expanded OFFLINE source geometry. It does not choose a robot goal,
    open destination-room doors, change H5 or admit planning/physics/motion.
    """
    import io,os,zipfile
    from hospital_geometry_tools import mesh_mask
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    port_path=Path(port_review_path).resolve();port=json.loads(port_path.read_bytes())
    condition_path=Path(condition_path).resolve();condition=json.loads(condition_path.read_bytes())
    medical_path=Path(port['contract']['source_medical_review']);medical=json.loads(medical_path.read_bytes())
    coverage_path=Path(medical['contract']['source_coverage']);coverage=json.loads(coverage_path.read_bytes())
    projection_path=Path(medical['contract']['source_projection']);projection=json.loads(projection_path.read_bytes())
    bundle_path=Path(projection['mesh_bundle']);bundle=json.loads(bundle_path.read_bytes())
    candidate_path=Path(port['contract']['source_transfer_candidate']);candidate=json.loads(candidate_path.read_bytes())
    if (condition['status']!='USER_ADOPTED_CONDITION_DEFINITION_ONLY_NOT_LIVE_SETUP_OR_TASK_ADMISSION'
            or condition['user_decision']['source']!='actual user reply to request_user_input_async'
            or condition['execution_authority']['actual_motion_authorized'] is not False
            or candidate['robot_start_xyyaw'] is not None or candidate['robot_goal_xyyaw'] is not None
            or candidate['actual_motion_authorized'] is not False
            or (_ijcas_path(ROOT, candidate['common_environment_definition'])).resolve()!=condition_path
            or port['contract']['target_bed']!=candidate['ordered_bed_roots'][1]
            or coverage['contract']['user_selected_transfer_candidate']!=str(candidate_path)):
        raise ValueError('Own selected-bed coverage and explicit definition-only adoption required')
    definition=condition['definition']
    groups=dict(door=definition['door']['root'],threshold=definition['threshold']['root'])
    if (definition['other_environment']['same_recorded_grid_resolution_m']!=RESOLUTION_M
            or definition['other_environment']['same_recorded_obstacle_height_slab_m']!=list(SLAB_M)
            or definition['other_environment']['same_full_body_clearance_m']!=.010
            or dict(row_count=definition['other_environment']['protected_environment_source_row_count'],
                    sha256=definition['other_environment']['protected_environment_source_sha256'])!=medical['contract']['original_environment_digest']):
        raise ValueError('Same environment digest/raster/slab/full-body clearance required')
    pins={}
    for path,record in ((port_path,port),(medical_path,medical),(coverage_path,coverage),
                        (projection_path,projection),(bundle_path,bundle),(candidate_path,candidate)):
        for name,value in {**record['input_sha256'],**record.get('output_sha256',{}),str(path):sha(path)}.items():
            name=str((_ijcas_path(ROOT, name)).resolve()) if not Path(name).is_absolute() else name
            if name in pins and pins[name]!=value:raise ValueError('Conflicting selected-transfer source: '+name)
            pins[name]=value
    pins[str(condition_path)]=sha(condition_path)
    for name,value in condition['immutable_source_sha256'].items():
        path=str((_ijcas_path(ROOT, name)).resolve())
        if path in pins and pins[path]!=value:raise ValueError('Historical adopted source pin conflict')
        pins[path]=value
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Selected transfer condition source changed: '+name)
    # Reuse the ORIGINAL historical lineage already authenticated by adoption.
    old_path=next(Path(p) for p in pins if p.endswith('/offline_condition_proposal.json')
                  and '/historical_condition_candidate/' in p)
    old=json.loads(old_path.read_bytes())
    if old['contract']['source_roots']!=groups:raise ValueError('Adopted exact historical asset identities required')
    h3_path,h3=pinned_H3_geometry(coverage['input_sha256'],coverage['contract']['anchors'])
    if str(h3_path)!=old['contract']['original_task_lineage']:
        raise ValueError('Expanded coverage must retain the original H3 task lineage')
    historical_path=Path(old['contract']['original_H3_door_audit']);historical=json.loads(historical_path.read_bytes())
    opened_path=Path(old['contract']['original_recorded_open_door_specification_source'])
    opened=json.loads(opened_path.read_bytes())
    for path,value in ((_ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_h3_door_state.py'),opened['source_sha256']['hospital_h3_door_state.py']),
                       (_ijcas_path(ROOT, 'studies/IJCAS/scripts/audit_hospital_h3_doorway.py'),opened['audit_script_sha256'])):
        if sha(path)!=value:raise ValueError('Original adopted hinge/audit source changed')
        if str(path) in pins and pins[str(path)]!=value:raise ValueError('Historical hinge/audit pin conflict')
        pins[str(path)]=value
    leaves=[row for row in historical['local_assets'] if row['path']==groups['door']]
    current=[row for row in bundle['primitives'] if row['path']==definition['door']['collider']]
    if len(leaves)!=1 or len(current)!=1:raise ValueError('Same single historical/current door collider required')
    source_arrays=next(Path(p) for p in bundle['output_sha256'] if p.endswith('recorded_world_mesh_arrays.npz'))
    with np.load(source_arrays) as source:
        partition_review,partitions=recorded_obstacle_partition_masks(coverage,bundle,source,groups)
        door_review,door_mesh=historical_H3_open_door_mesh(current[0],source,leaves[0],
            recorded_specification=opened['door_state_specification'])
    baseline_path=next(Path(p) for p in medical['output_sha256'] if p.endswith('medical_keepout_grid.npz'))
    refined_path=next(Path(p) for p in projection['output_sha256'] if p.endswith('mesh_refined_grid.npz'))
    with np.load(baseline_path) as a:
        baseline=a['occupied'].copy();keepout=a['medical_asset_keepouts'].copy()
        supported=a['supported_full_cells'].copy();origin=a['origin_xy_m'].copy();resolution=float(a['resolution_m'])
    with np.load(refined_path) as a:
        base=fixed_cell_blockers(coverage['coverage_grid'],a['supported_full_cells'])|keepout
    if resolution!=RESOLUTION_M:raise ValueError('No condition-specific coarsening')
    historical_door=mesh_mask(door_mesh,origin,resolution,base.shape,*SLAB_M)
    candidate_grid=compose_recorded_common_condition(base,partitions,historical_door,baseline)
    anchors=[dict(row) for row in coverage['contract']['anchors']]
    anchors.extend(dict(row,role='PREDECLARED_TARGET_PORT_NOT_AN_ADOPTED_ROBOT_GOAL') for row in port['all_target_ports'])
    if len(anchors)!=16:raise ValueError('All ten unchanged source controls plus all six fixed ports required')
    sufficient,arrays=cardinal_body_connection_review(anchors,candidate_grid,origin,resolution)
    necessary,outer=optimistic_inscribed_disk_review(anchors,candidate_grid,origin,resolution)
    for s,n in zip(sufficient['all_unordered_anchor_pairs'],necessary['all_unordered_anchor_pairs']):
        if (s['first'],s['second'])!=(n['first'],n['second']):raise ValueError('Original complete pair order changed')
        if s['sufficient_GEOMETRIC_body_sweep_connection'] and not n['necessary_optimistic_disk_connection']:
            raise ValueError('Sufficient/necessary geometry contradiction')
    obstacle_only=keepout|partitions['retained']|historical_door
    ports=bed_port_geometry_review([{k:row[k] for k in ('id','world_xyyaw','reserve_after_full_body_clearance_m','role')}
        for row in port['all_target_ports']],candidate_grid,obstacle_only,origin,resolution,
        coverage['coverage_grid']['analysis_roi_xy_m'],coverage['contract']['same_three_source_support_rectangles'])
    sources=[Path(__file__).resolve(),_ijcas_path(ROOT, 'src/patient_initial_stability/test/test_hospital_mission_geometry.py')]
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:
        for path in sources:
            info=zipfile.ZipInfo(str(path.relative_to(ROOT)),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16
            z.writestr(info,path.read_bytes())
    payload=buffer.getvalue()
    contract=dict(version='H6_SELECTED_BED_PAIR_ADOPTED_H3_DEFINITION_ALL_FIXED_PORTS_OFFLINE_v1',
        original_port_review=str(port_path),common_condition_definition=str(condition_path),source_coverage=str(coverage_path),
        source_medical_review=str(medical_path),source_projection=str(projection_path),source_mesh_bundle=str(bundle_path),
        all_original_controls_and_ports_retained=True,source_roots=groups,
        all_other_current_colliders_retained=len(partition_review['partitions']['retained']['original_collider_paths']),
        earlier_condition_definition_ROI_retained_count=definition['other_environment']['all_other_recorded_ROI_collider_contributions_retained'],
        analysis_coverage_expansion_NOT_a_different_environment_policy=True,
        destination_room_doors_thresholds_NOT_assumed_open=True,actual_motion_authorized=False,
        source_archive_sha256=hashlib.sha256(payload).hexdigest(),input_sha256=pins)
    key=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
    folder=GATE/'hospital_h6_geometry_review/selected_transfer_condition'/key;target=folder/'condition_connectivity_review.json'
    if target.exists():
        saved=json.loads(target.read_bytes())
        if saved['contract']!=contract:raise ValueError('Existing selected condition contract changed')
        for p,h in saved['output_sha256'].items():
            if sha(p)!=h:raise ValueError('Original selected condition output changed')
        return target
    if folder.exists():raise ValueError('Partial prior condition review preserved; no overwrite')
    folder.mkdir(parents=True)
    with (folder/'executed_sources.zip').open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    evidence.write_exclusive(folder/'software_contract.json',contract)
    grid=folder/'ADOPTED_DEFINITION_OFFLINE_GEOMETRY_NOT_LIVE.npz'
    np.savez_compressed(grid,occupied=candidate_grid,source_baseline=baseline,base_blocked=base,
        retained_other_obstacles=partitions['retained'],original_door=partitions['door'],original_threshold=partitions['threshold'],
        historical_door=historical_door,medical_keepouts=keepout,supported_full_cells=supported,
        **arrays,**outer,origin_xy_m=origin,resolution_m=resolution)
    report=dict(contract=contract,original_collider_partition_review=partition_review,original_door_source_review=door_review,
        all_target_ports=ports,sufficient_body_geometry=sufficient,necessary_outer_approximation=necessary,
        source_obstacle_count=len(bundle['primitives']),complete_anchor_pair_count=len(sufficient['all_unordered_anchor_pairs']),
        positive_sufficient_pair_count=sum(p['sufficient_GEOMETRIC_body_sweep_connection'] for p in sufficient['all_unordered_anchor_pairs']),
        negative_necessary_pair_count=sum(not p['necessary_optimistic_disk_connection'] for p in necessary['all_unordered_anchor_pairs']),
        no_H6_robot_goal_selected=True,no_steering_dynamics_clinical_alignment_or_cooked_support_proof=True,
        definition_adopted_NOT_mission_map_or_live_setup_or_driving_admission=True,
        current_H5_and_historical_H3_trials_unchanged=True,new_PhysX_trial_count=0,
        actual_motion_authorized=False,use_for_final_statistics=False,input_sha256=pins,
        output_sha256={str(p):sha(p) for p in (folder/'executed_sources.zip',folder/'software_contract.json',grid)})
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during selected condition review: '+p)
    evidence.write_exclusive(target,report)
    return target
