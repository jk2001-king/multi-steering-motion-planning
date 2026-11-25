"""Read-only display diagnosis of the actual compact-staged H4 actor.

This is not staging, physics admission or a visibility/pose repair. Selection
is the only UI action; camera framing is deliberately left to the viewport.
"""

import math

import hospital_h4_subfloor_remaining_staging as preparation


ROBOT = preparation.ROBOT


def geometry_readback(stage):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(ROBOT)
    result = dict(robot_root=ROBOT, exists=bool(prim.IsValid()))
    if not prim.IsValid():
        return dict(result, status='EXPECTED_NEW_ACTOR_ABSENT_NO_RECREATE_OR_PLAY')
    result.update(active=prim.IsActive(), loaded=prim.IsLoaded())
    if not prim.IsActive():
        return dict(result, status='EXPECTED_NEW_ACTOR_INACTIVE_NO_ACTIVATION_OR_PLAY')
    ancestors = []
    ancestor = prim
    while ancestor and not ancestor.IsPseudoRoot():
        imageable = UsdGeom.Imageable(ancestor)
        if imageable:
            a = imageable.GetVisibilityAttr()
            ancestors.append(dict(path=str(ancestor.GetPath()), visibility=str(a.Get()),
                computed_visibility=str(imageable.ComputeVisibility()),
                authored_visibility_layers=[s.layer.identifier for s in a.GetPropertyStack()]))
        ancestor = ancestor.GetParent()
    rows = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Gprim):
            continue
        im = UsdGeom.Imageable(p)
        rows.append(dict(path=str(p.GetPath()), type=p.GetTypeName(),
            visibility=str(im.ComputeVisibility()), purpose=str(im.ComputePurpose())))
    render_rows = [r for r in rows if r['purpose'] in ('default', 'render', 'proxy')]
    visible = [r for r in render_rows if r['visibility'] != 'invisible']
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False, ignoreVisibility=False).ComputeWorldBound(prim).ComputeAlignedRange()
    # BBoxCache evaluates the queried subtree and may still return a bound
    # beneath an invisible ancestor. ComputeVisibility above covers ancestors.
    bound = None if bbox.IsEmpty() or not visible else dict(minimum_m=list(bbox.GetMin()), maximum_m=list(bbox.GetMax()))
    base = stage.GetPrimAtPath(ROBOT+'/base_link')
    position = None if not base.IsValid() else list(UsdGeom.XformCache().GetLocalToWorldTransform(base).ExtractTranslation())
    status = ('USD_RENDER_GEOMETRY_PRESENT_CAMERA_FILTER_OR_OCCLUSION_NOT_YET_ISOLATED'
        if visible and bound else 'USD_RENDER_GEOMETRY_HIDDEN_OR_EMPTY_NO_AUTOMATIC_EDIT')
    return dict(result, status=status, ancestors=ancestors, geometry=rows,
        render_geometry_count=len(render_rows), visible_render_geometry_count=len(visible),
        world_visible_bound=bound, actual_base_position_m=position,
        pixel_visibility_or_renderer_occlusion_proven=False)


def viewport_readback(viewport, geometry):
    """Project the eight AABB corners; never infer actual unoccluded pixels."""
    if viewport is None:
        return dict(available=False)
    from pxr import Gf

    data = dict(available=True, camera_path=str(viewport.camera_path),
        resolution_px=list(viewport.resolution))
    bound = geometry.get('world_visible_bound')
    if not bound:
        return data
    projected = [viewport.world_to_ndc.Transform(Gf.Vec3d(x, y, z))
        for x in (bound['minimum_m'][0], bound['maximum_m'][0])
        for y in (bound['minimum_m'][1], bound['maximum_m'][1])
        for z in (bound['minimum_m'][2], bound['maximum_m'][2])]
    if not all(math.isfinite(float(v)) for p in projected for v in p):
        return dict(data, projection_valid=False)
    minimum = [min(float(p[i]) for p in projected) for i in (0, 1)]
    maximum = [max(float(p[i]) for p in projected) for i in (0, 1)]
    outside = any(maximum[i] < -1 or minimum[i] > 1 for i in (0, 1))
    return dict(data, projection_valid=True, projected_AABB_xy_min=minimum,
        projected_AABB_xy_max=maximum, projected_AABB_outside_viewport_xy=outside,
        projection_is_NOT_a_depth_clipping_or_occlusion_test=True)


def verify_actual_preparation():
    protocol = preparation.load_protocol()
    report = preparation.read_small(preparation.LIVE)
    preparation.verify(report['input_sha256'])
    if (preparation.FAILURE.exists()
            or report['status'] != 'REVIEWED_COMPLETED_STOP_COMPACT_SUBFLOOR_COHORT_STAGED_STATIC_PENDING'
            or report['new_robot_root'] != ROBOT
            or report['all_original_protected_row_count'] != 4997
            or report['all_rows_match_original_except_exact_four_flags'] is not True):
        raise ValueError('Actual compact preparation receipt required; never recreate or Play')
    return protocol, report
