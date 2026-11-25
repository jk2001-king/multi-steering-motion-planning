"""USD camera/material and EXISTING Fabric-cache GETs; no renderer repair.

Missing caches are missing evidence, not an instruction to populate/restart
them. No camera, setting, visibility, transform or physics writes are used.
"""

import math

import numpy as np

import hospital_h4_robot_visibility as display


def camera_depth_readback(stage, viewport, geometry):
    from pxr import Gf, UsdGeom

    if viewport is None:
        return dict(available=False)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(str(viewport.camera_path)))
    if not camera:
        return dict(available=False, error='Active viewport USD camera unavailable')
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
    inverse = matrix.GetInverse()
    clip = camera.GetClippingRangeAttr().Get()
    bound = geometry.get('world_visible_bound')
    data = dict(available=True, camera_path=str(viewport.camera_path),
        projection=str(camera.GetProjectionAttr().Get()), camera_position_m=list(matrix.ExtractTranslation()),
        camera_forward_world=list(matrix.TransformDir(Gf.Vec3d(0,0,-1))),
        near_m=float(clip[0]),far_m=float(clip[1]))
    if bound is not None:
        depths = [-float(inverse.Transform(Gf.Vec3d(x,y,z))[2])
            for x in (bound['minimum_m'][0],bound['maximum_m'][0])
            for y in (bound['minimum_m'][1],bound['maximum_m'][1])
            for z in (bound['minimum_m'][2],bound['maximum_m'][2])]
        if not np.isfinite(depths).all():
            return dict(data, depth_valid=False)
        data.update(depth_valid=True, actor_AABB_nearest_depth_m=min(depths),actor_AABB_farthest_depth_m=max(depths),
            whole_actor_AABB_clipped_by_near_or_far=bool(max(depths)<clip[0] or min(depths)>clip[1]),
            actor_AABB_depth_overlaps_camera_clip_range=bool(max(depths)>=clip[0] and min(depths)<=clip[1]),
            AABB_overlap_does_NOT_prove_visible_pixels=True)
    return data


def display_setting_readback(settings, viewport):
    keys = ['/app/useFabricSceneDelegate','/physics/updateToUsd','/physics/updateVelocitiesToUsd',
        '/physics/fabricEnabled','/persistent/app/hydra/displayPurpose/guide',
        '/persistent/app/hydra/displayPurpose/proxy','/persistent/app/hydra/displayPurpose/render',
        '/persistent/app/viewport/displayOptions','/rtx/rendermode']
    if viewport is not None:
        keys += ['/persistent/app/viewport/'+str(viewport.id)+'/'+part+'/visible'
            for part in ('scene/meshes','guide/selection')]
    return {key:settings.get(key) for key in keys}


def material_and_occluder_readback(stage, geometry):
    from pxr import Usd, UsdGeom, UsdShade

    rows = []
    for row in geometry.get('geometry',[]):
        if row['purpose'] not in ('default','render','proxy'):
            continue
        p = stage.GetPrimAtPath(row['path'])
        opacity = UsdGeom.PrimvarsAPI(p).FindPrimvarWithInheritance('displayOpacity')
        values = list(map(float,opacity.ComputeFlattened())) if opacity and opacity.HasValue() else None
        material,_ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        rows.append(dict(path=row['path'],display_opacity=values,
            all_authored_display_opacity_zero=bool(values is not None and values and all(v==0. for v in values)),
            bound_render_material=str(material.GetPath()) if material else None,
            shader_opacity_or_final_rendered_alpha_NOT_inferred=True))
    # This exact original roof's AABB was found over the fixed start in the
    # saved full inventory. Its visibility is GET-only; never hide the roof.
    path='/World/Hospital/Geo_Roof_Costum8_25'
    p=stage.GetPrimAtPath(path)
    roof=dict(path=path,exists=bool(p.IsValid()),active=bool(p.IsValid() and p.IsActive()))
    if p.IsValid() and p.IsActive() and UsdGeom.Imageable(p):
        roof['computed_visibility']=str(UsdGeom.Imageable(p).ComputeVisibility())
        roof['computed_purpose']=str(UsdGeom.Imageable(p).ComputePurpose())
        box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],
            useExtentsHint=False,ignoreVisibility=True).ComputeWorldBound(p).ComputeAlignedRange()
        if not box.IsEmpty():
            roof['world_AABB']=dict(minimum_m=list(box.GetMin()),maximum_m=list(box.GetMax()))
    return dict(actor_render_materials=rows,original_roof_over_fixed_start=roof,
        exact_triangle_occlusion_or_material_transparency_NOT_proven=True)


def inspect_cache_rows(stage, rt_stage, paths, path_factory=str):
    """Read cached values as-is. In particular never updateWorldXforms()."""
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    rows=[]
    for path in paths:
        usd=stage.GetPrimAtPath(path)
        expected=np.asarray(cache.GetLocalToWorldTransform(usd),dtype=float)
        # USDRT GetPrimAtPath can lazily populate absent prims from USD.
        # HasPrimAtPath queries Fabric directly; never Get a missing prim.
        existing=bool(rt_stage.HasPrimAtPath(path_factory(path),False))
        row=dict(path=path,existing_Fabric_prim=existing,USD_world_position_m=expected[3,:3].tolist())
        if existing:
            p=rt_stage.GetPrimAtPath(path_factory(path))
            if not p.IsValid():
                raise ValueError('Previously existing Fabric prim became unavailable')
            existing_attrs={str(a.GetName()):a for a in p.GetAttributes()}
            values={}
            for name in ('omni:fabric:worldMatrix','omni:fabric:localMatrix','_worldPosition','_worldOrientation',
                    '_worldScale','_localMatrix','visibility','purpose'):
                if name not in existing_attrs:
                    continue
                a=existing_attrs[name]
                value=a.Get()
                if value is None or isinstance(value,(str,bool,int,float)):
                    values[name]=value
                elif name=='_worldOrientation':
                    # Keep the runtime quaternion representation without assuming
                    # XYZW vs WXYZ ordering or changing it.
                    values[name]=str(value)
                else:
                    arr=np.asarray(value,dtype=float)
                    if not np.isfinite(arr).all():
                        raise ValueError('Non-finite cached render attribute: '+path+'.'+name)
                    values[name]=arr.tolist()
            row['cached_attributes']=values
            matrix=values.get('omni:fabric:worldMatrix')
            if matrix is not None:
                array=np.asarray(matrix,dtype=float)
                if array.shape != (4,4):
                    raise ValueError('Unexpected cached worldMatrix shape')
                row['cached_worldMatrix_max_abs_delta_from_USD']=float(np.max(np.abs(array-expected)))
                row['cached_world_position_delta_m']=float(np.linalg.norm(array[3,:3]-expected[3,:3]))
                row['cached_matrix_may_be_stale_and_active_render_path_still_requires_review']=True
            position=values.get('_worldPosition')
            if position is not None:
                array=np.asarray(position,dtype=float)
                if array.shape != (3,):
                    raise ValueError('Unexpected cached legacy world-position shape')
                row['legacy_world_position_delta_m']=float(np.linalg.norm(array-expected[3,:3]))
        rows.append(row)
    return rows


def fabric_readback(stage, geometry):
    from pxr import Usd, UsdPhysics, UsdUtils

    try:
        from usdrt import Usd as RtUsd, Sdf as RtSdf
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if not RtUsd.Stage.SimStageWithHistoryExists(stage_id):
            return dict(existing_sim_Fabric_stage=False,cache_created_or_populated=False,
                unavailable_is_NOT_a_reason_to_Play=True)
        rt=RtUsd.Stage.Attach(stage_id)
        paths=sorted({r['path'] for r in geometry.get('geometry',[]) if r['purpose'] in ('default','render','proxy')}
            | {str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(display.ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)})
        return dict(existing_sim_Fabric_stage=True,cache_created_or_populated=False,
            rows=inspect_cache_rows(stage,rt,paths,RtSdf.Path),
            missing_prims_never_queried_or_populated=True,
            cached_attributes_are_GET_only_NOT_a_physics_state_readback=True)
    except Exception as error:
        return dict(optional_cache_error=type(error).__name__+': '+str(error),cache_created_or_populated=False,
            unavailable_is_NOT_a_reason_to_Play=True)
