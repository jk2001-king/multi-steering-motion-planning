"""Bounded visual-only USD resync; restore EXACT original Session contents.

Only source `visuals` instance metadata is touched. Collision, rigid-body,
mass, joint and articulation APIs are forbidden anywhere in these subtrees.
No layer-wide import/rollback: remove only our 13 newly authored empty overs.
Caller must keep the exact non-playing timeline boundary while processing
renderer UI frames. A completed PAUSED trial must never be Stopped for this.
"""

import hashlib
import json

import hospital_h4_instance_display_review as instances


def masked_session(layer,paths):
    from pxr import Sdf
    copy=Sdf.Layer.CreateAnonymous('H4_VISUAL_METADATA_COMPARE_ONLY.usda')
    copy.TransferContent(layer)
    edits=Sdf.BatchNamespaceEdit()
    for path in paths:
        if copy.GetPrimAtPath(path):edits.Add(path,Sdf.Path.emptyPath)
    if not copy.Apply(edits):raise RuntimeError('Cannot compare typed Session copy')
    return copy.ExportToString()


def geometry_fingerprint(stage,paths):
    from pxr import Usd,UsdGeom,UsdShade
    digest=instances.render.display.preparation.compact.native_digest
    values={}
    for path in paths:
        for p in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Gprim):continue
            material,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            values[str(p.GetPath())]=dict(type=p.GetTypeName(),
                attributes={a.GetName():dict(default=digest(a.Get()),samples=[(t,digest(a.Get(t))) for t in a.GetTimeSamples()])
                    for a in p.GetAttributes()},
                relationships={r.GetName():list(map(str,r.GetTargets())) for r in p.GetRelationships()},
                material=str(material.GetPath()) if material else None,
                visible=str(UsdGeom.Imageable(p).ComputeVisibility()),purpose=str(UsdGeom.Imageable(p).ComputePurpose()))
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest(),len(values)


def capture(stage,paths,actor=None):
    from pxr import Usd,UsdGeom,UsdPhysics
    actor=instances.render.display.ROBOT if actor is None else actor
    paths=sorted(paths)
    if len(paths)!=13 or len(set(paths))!=13:
        raise ValueError('Exactly the actual thirteen visual instance roots required')
    session=stage.GetSessionLayer()
    for path in paths:
        p=stage.GetPrimAtPath(path)
        if (not path.startswith(actor+'/') or not path.endswith('/visuals')
                or not p.IsValid() or not p.IsInstance() or not p.IsA(UsdGeom.Xform)
                or session.GetPrimAtPath(path) is not None
                or session.GetPrimAtPath(str(p.GetParent().GetPath())) is None):
            raise ValueError('Original inherited visual-only instance with no local override required: '+path)
        for child in Usd.PrimRange(p,Usd.TraverseInstanceProxies()):
            if (child.HasAPI(UsdPhysics.CollisionAPI) or child.HasAPI(UsdPhysics.RigidBodyAPI)
                    or child.HasAPI(UsdPhysics.MassAPI) or child.HasAPI(UsdPhysics.ArticulationRootAPI)
                    or child.IsA(UsdPhysics.Joint)):
                raise ValueError('A physical prim exists below visuals; do NOT deinstance it: '+str(child.GetPath()))
    fingerprint,count=geometry_fingerprint(stage,paths)
    if count!=22:raise ValueError('Exactly the source twenty-two visual primitives required')
    return dict(paths=paths,root_text=stage.GetRootLayer().ExportToString(),
        session_text=session.ExportToString(),masked_session_text=masked_session(session,paths),
        visual_geometry_sha256=fingerprint,visual_primitive_count=count)


def display_modules_for_actor(actor):
    """Private read-only bindings for the declared actor, not a module mutation."""
    from types import FunctionType,SimpleNamespace
    def bind(function,**values):
        return FunctionType(function.__code__,dict(function.__globals__,**values),
            function.__name__,function.__defaults__,function.__closure__)
    display=SimpleNamespace(**vars(instances.render.display));display.ROBOT=actor
    display.geometry_readback=bind(display.geometry_readback,ROBOT=actor)
    render=SimpleNamespace(**vars(instances.render));render.display=display
    render.fabric_readback=bind(render.fabric_readback,display=display)
    readers=SimpleNamespace(**vars(instances));readers.render=render
    readers.usd_instances=bind(readers.usd_instances,render=render)
    return readers


def begin(stage,transaction):
    from pxr import Usd
    if (stage.GetRootLayer().ExportToString()!=transaction['root_text']
            or stage.GetSessionLayer().ExportToString()!=transaction['session_text']):
        raise RuntimeError('Stage changed before visual metadata refresh; no action')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for path in transaction['paths']:
            if not stage.GetPrimAtPath(path).SetInstanceable(False):
                raise RuntimeError('Visual metadata authoring failed: '+path)
    verify_transient(stage,transaction)


def verify_transient(stage,transaction):
    if (stage.GetRootLayer().ExportToString()!=transaction['root_text']
            or masked_session(stage.GetSessionLayer(),transaction['paths'])!=transaction['masked_session_text']
            or geometry_fingerprint(stage,transaction['paths'])!=(transaction['visual_geometry_sha256'],22)):
        raise RuntimeError('Non-visual-metadata change detected; no broad rollback permitted')


def restore(stage,transaction):
    """Remove only exact newly authored metadata overs, never restore a layer."""
    from pxr import Sdf
    session=stage.GetSessionLayer();edits=Sdf.BatchNamespaceEdit()
    for path in transaction['paths']:
        spec=session.GetPrimAtPath(path)
        if spec is None:continue  # begin may have stopped before this path.
        if (set(spec.ListInfoKeys())!={'specifier','instanceable'}
                or spec.GetInfo('specifier')!=Sdf.SpecifierOver
                or spec.GetInfo('instanceable') is not False or len(spec.nameChildren) or len(spec.properties)):
            raise RuntimeError('Visual override contains unexpected/user fields; refusing to remove: '+path)
        edits.Add(path,Sdf.Path.emptyPath)
    if not session.Apply(edits):raise RuntimeError('Cannot remove only our exact visual metadata overs')
    if (stage.GetRootLayer().ExportToString()!=transaction['root_text']
            or session.ExportToString()!=transaction['session_text']
            or geometry_fingerprint(stage,transaction['paths'])!=(transaction['visual_geometry_sha256'],22)
            or not all(stage.GetPrimAtPath(p).IsInstance() for p in transaction['paths'])):
        raise RuntimeError('Exact original source/Session restoration not proven; no automatic repair or trial')
    return dict(original_Root_and_Session_text_exactly_restored=True,
        source_visual_geometry_material_visibility_purpose_and_all_attributes_unchanged=True,
        all_thirteen_visual_roots_original_instancing_restored=True,
        visual_primitive_count=22,actual_static_or_drive_admission=False)


async def initialize_stopped_actor(stage,actor,*,state,next_frame,physical_readback,viewport,settings,budget_s=15.):
    """Register the NEW actor before static preflight, never after a PASS.

    Only thirteen render-only instanceable overrides are briefly authored and
    exactly removed. No physics/timeline/renderer setting/camera is changed.
    Cache registration is checked, not assumed to prove visible pixels.
    """
    import time
    expected=dict(playing=False,stopped=True,current_time_s=0.)
    if state()!=expected:raise ValueError('New-actor display initialization requires STOPPED zero time')
    readers=display_modules_for_actor(actor);render=readers.render
    geometry=render.display.geometry_readback(stage)
    if geometry.get('visible_render_geometry_count')!=22:
        raise ValueError('All twenty-two source render shapes must exist before registration')
    topology=readers.usd_instances(stage,geometry)
    targets=[row['instance_path'] for row in topology['instance_roots']]
    transaction=capture(stage,targets,actor=actor)
    actual=physical_readback();preferences=render.display_setting_readback(settings,viewport)
    projection=render.display.viewport_readback(viewport,geometry)
    depth=render.camera_depth_readback(stage,viewport,geometry)
    before=readers.existing_instance_cache(stage,topology)
    began=False;restored=False;begun=time.monotonic();frames=0
    async def frame():
        nonlocal frames
        await next_frame()
        frames+=1
        if state()!=expected or physical_readback()!=actual or time.monotonic()-begun>budget_s:
            raise RuntimeError('STOPPED source state or bounded visual-initialization frames changed')
    try:
        began=True;begin(stage,transaction)
        await frame();await frame()
        verify_transient(stage,transaction)
        restoration=restore(stage,transaction);restored=True
        await frame();await frame()
        def ready(cache):
            rows=cache.get('instance_root_rows',[])
            positions=[row.get('cached_world_position_delta_m',row.get('legacy_world_position_delta_m')) for row in rows]
            return (cache.get('existing_sim_Fabric_stage') is True and len(rows)==13
                and {row['path'] for row in rows}==set(targets)
                and all(row.get('existing_Fabric_prim') is True for row in rows)
                and all(value is not None and isinstance(value,(float,int)) and 0<=value<=.001 for value in positions))
        after=readers.existing_instance_cache(stage,topology)
        # A slow renderer may finish registration on a later UI frame. Wait
        # boundedly for the SAME transaction; never retoggle/retry physics.
        while not ready(after) and frames<60:
            await frame()
            after=readers.existing_instance_cache(stage,topology)
        if not ready(after):
            raise RuntimeError('Thirteen visual roots/1mm cached positions not registered; no static/drive admission')
        if (stage.GetRootLayer().ExportToString()!=transaction['root_text']
                or stage.GetSessionLayer().ExportToString()!=transaction['session_text']
                or geometry!=render.display.geometry_readback(stage)
                or preferences!=render.display_setting_readback(settings,viewport)
                or projection!=render.display.viewport_readback(viewport,geometry)
                or depth!=render.camera_depth_readback(stage,viewport,geometry)):
            raise RuntimeError('Exact layers/source geometry/settings/camera preservation not proven')
        return dict(restoration,status='NEW_ACTOR_RENDER_ROOTS_REGISTERED_BEFORE_STATIC_PIXELS_REQUIRE_OBSERVATION',
            actor=actor,timeline_before_and_after=expected,
            source_actor_before_and_after=actual,existing_visual_cache_before=before,existing_visual_cache_after=after,
            registered_visual_root_count=13,visible_USD_shape_count=22,
            renderer_UI_frame_count=frames,single_visual_metadata_transaction_count=1,
            existing_1mm_cache_position_readback_threshold_m=.001,
            camera_projection=projection,camera_depth=depth,settings_before_and_after=preferences,
            actual_visible_pixels_confirmed=False,any_timeline_or_physics_or_camera_command=False,
            actual_static_or_drive_admission=False,new_physical_trial_count=0)
    except BaseException:
        if began and not restored:restore(stage,transaction)
        raise
