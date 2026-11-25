"""Camera-only Session view adjustment, NOT any actor/physics repair."""

import hashlib
import math


CAMERA='/OmniverseKit_Top'


def without_top_camera(source_layer):
    """Compare typed anonymous copies, never serialize/reparse the live layer.

    Kit Session exports can use '#sdf 1.4.32', not '#usda'. Forcing that text
    through a .usda parser fails AFTER a camera edit. TransferContent avoids
    the file-format assumption and never writes into the source/live layer.
    """
    from pxr import Sdf
    if not isinstance(source_layer,Sdf.Layer):
        raise TypeError('Pass the actual Sdf.Layer, not its format-dependent exported text')
    layer=Sdf.Layer.CreateAnonymous('H4_VIEW_COMPARE_ONLY.usda')
    layer.TransferContent(source_layer)
    if layer.GetPrimAtPath(CAMERA):
        edits=Sdf.BatchNamespaceEdit();edits.Add(CAMERA,Sdf.Path.emptyPath)
        if not layer.Apply(edits):
            raise ValueError('Cannot compare exact camera-only Session scope')
    return layer.ExportToString()


def read_camera(stage):
    from pxr import UsdGeom
    cam=UsdGeom.Camera(stage.GetPrimAtPath(CAMERA))
    if not cam:
        raise ValueError('Existing built-in Top camera required; never create a substitute camera')
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(cam.GetPrim())
    return dict(path=CAMERA,projection=str(cam.GetProjectionAttr().Get()),
        world_matrix=[[float(matrix[i][j]) for j in range(4)] for i in range(4)],
        position_m=list(matrix.ExtractTranslation()),clipping_range_m=list(cam.GetClippingRangeAttr().Get()),
        horizontal_aperture=float(cam.GetHorizontalApertureAttr().Get()),
        vertical_aperture=float(cam.GetVerticalApertureAttr().Get()))


def set_top_clip_only(stage,clipping_range):
    """Restore a recorded clipping range without changing any camera pose.

    Orthographic Frame Selected can move Top back to about 50 km. A far plane
    of 100 m is incompatible with that view. Do not repeat the 8 m pose edit.
    """
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    near,far=map(float,clipping_range)
    if not all(math.isfinite(v) for v in (near,far)) or not 0.<near<far:
        raise ValueError('A finite positive ordered recorded clipping range is required')
    before=read_camera(stage);prim=stage.GetPrimAtPath(CAMERA)
    if (before['projection']!='orthographic' or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('Existing nonphysical orthographic Top camera required')
    root_text=stage.GetRootLayer().ExportToString()
    protected=without_top_camera(stage.GetSessionLayer())
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(near,far))
    after=read_camera(stage)
    if (stage.GetRootLayer().ExportToString()!=root_text
            or without_top_camera(stage.GetSessionLayer())!=protected
            or {k:v for k,v in before.items() if k!='clipping_range_m'}!=
                {k:v for k,v in after.items() if k!='clipping_range_m'}):
        raise RuntimeError('Unexpected camera-pose or non-camera change; no recovery/reset attempted')
    return dict(before=before,after=after,Root_layer_unchanged=True,
        Session_delta_limited_to_existing_Top_camera=True,camera_pose_unchanged=True)


def focus_top_xy_only(stage,target_xyz,resolution,width_m=6.):
    """Frame explicit retained coordinates, NOT Frame Selected or actor extents.

    Retain camera Z/orientation and the restored clipping range. Only the
    camera XY and orthographic apertures change. No new camera is created.
    """
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    xyz=list(map(float,target_xyz));pixels=list(map(float,resolution))
    if (len(xyz)!=3 or len(pixels)!=2 or not all(math.isfinite(v) for v in (*xyz,*pixels,width_m))
            or min(pixels)<=0. or width_m<=0. or UsdGeom.GetStageMetersPerUnit(stage)!=1.):
        raise ValueError('Finite retained XYZ, positive viewport/width and original metre-scaled stage required')
    before=read_camera(stage);prim=stage.GetPrimAtPath(CAMERA)
    if (before['projection']!='orthographic' or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('Existing nonphysical orthographic Top camera required')
    world=Gf.Matrix4d(*[v for row in before['world_matrix'] for v in row])
    forward=world.TransformDir(Gf.Vec3d(0,0,-1))
    depth=before['position_m'][2]-xyz[2]
    if (abs(forward[0])>1e-6 or abs(forward[1])>1e-6 or forward[2]>-.999999
            or not before['clipping_range_m'][0]<depth<before['clipping_range_m'][1]):
        raise ValueError('Downward Top camera and already-correct clipping required; no height/clip guessing')
    root=stage.GetRootLayer().ExportToString();protected=without_top_camera(stage.GetSessionLayer())
    world.SetTranslateOnly(Gf.Vec3d(xyz[0],xyz[1],before['position_m'][2]))
    local=world*UsdGeom.XformCache().GetParentToWorldTransform(prim).GetInverse()
    height_m=width_m*pixels[1]/pixels[0]
    cam=UsdGeom.Camera(prim)
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(local)
        cam.GetHorizontalApertureAttr().Set(width_m/Gf.Camera.APERTURE_UNIT)
        cam.GetVerticalApertureAttr().Set(height_m/Gf.Camera.APERTURE_UNIT)
    after=read_camera(stage)
    window=cam.GetCamera().frustum.GetWindow()
    actual_window=list(window.GetSize())
    if (stage.GetRootLayer().ExportToString()!=root
            or protected!=without_top_camera(stage.GetSessionLayer())
            or after['position_m'][2]!=before['position_m'][2]
            or after['clipping_range_m']!=before['clipping_range_m']
            or after['world_matrix'][:3]!=before['world_matrix'][:3]
            or max(abs(after['position_m'][i]-xyz[i]) for i in (0,1))>1e-8
            or max(abs(a-b) for a,b in zip(actual_window,(width_m,height_m)))>1e-5):
        raise RuntimeError('Unexpected non-camera or view-window change; no automatic repair')
    return dict(before=before,after=after,explicit_target_xyz_m=xyz,
        orthographic_window_width_height_m=actual_window,
        camera_height_orientation_and_clipping_unchanged=True,
        Root_layer_unchanged=True,Session_delta_limited_to_existing_Top_camera=True,
        actor_extents_and_Frame_Selected_NOT_used=True)


def adjust_top_camera_only(stage,base_z):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    before=read_camera(stage)
    prim=stage.GetPrimAtPath(CAMERA)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('The view camera must not be a physical actor')
    if before['projection']!='orthographic' or not math.isfinite(base_z):
        raise ValueError('Existing orthographic Top view and finite retained base height required')
    root_text=stage.GetRootLayer().ExportToString()
    session_text=stage.GetSessionLayer().ExportToString()
    non_camera_before=without_top_camera(stage.GetSessionLayer())
    cache=UsdGeom.XformCache()
    world=Gf.Matrix4d(*[v for row in before['world_matrix'] for v in row])
    forward=world.TransformDir(Gf.Vec3d(0,0,-1))
    if abs(forward[0])>1e-6 or abs(forward[1])>1e-6 or forward[2]>-.999999:
        raise ValueError('Only the existing downward-looking Top view may be adjusted')
    position=world.ExtractTranslation()
    world.SetTranslateOnly(Gf.Vec3d(position[0],position[1],base_z+8.))
    local=world*cache.GetParentToWorldTransform(prim).GetInverse()
    cam=UsdGeom.Camera(prim)
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(local)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(.1,100.))
        coi=prim.GetAttribute('omni:kit:centerOfInterest')
        if coi and coi.Get() is not None:
            target=world.GetInverse().Transform(Gf.Vec3d(position[0],position[1],base_z+.2))
            if str(coi.GetTypeName())=='double3':
                coi.Set(Gf.Vec3d(*target))
            elif str(coi.GetTypeName())=='float3':
                coi.Set(Gf.Vec3f(*target))
    after=read_camera(stage)
    new_session=stage.GetSessionLayer().ExportToString()
    if (stage.GetRootLayer().ExportToString()!=root_text
            or non_camera_before!=without_top_camera(stage.GetSessionLayer())
            or after['position_m'][:2]!=before['position_m'][:2]
            or after['horizontal_aperture']!=before['horizontal_aperture']
            or after['vertical_aperture']!=before['vertical_aperture']
            or abs(after['position_m'][2]-(base_z+8.))>1e-8):
        raise RuntimeError('Unexpected non-camera change; no rollback/reset/physics action performed')
    return dict(before=before,after=after,Root_layer_unchanged=True,
        Session_delta_limited_to_existing_Top_camera=True,
        original_Root_text_sha256=hashlib.sha256(root_text.encode()).hexdigest(),
        original_Session_text_sha256=hashlib.sha256(session_text.encode()).hexdigest(),
        adjusted_Session_text_sha256=hashlib.sha256(new_session.encode()).hexdigest(),
        camera_only_NOT_a_robot_pose_change=True,
        cause_or_actual_visual_recovery_NOT_yet_proven=True)
