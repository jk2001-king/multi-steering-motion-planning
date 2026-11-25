"""Read original H4 floor surfaces while retaining actual H3 FINAL PAUSED."""


def _review_h4_original_floor_surfaces():
    import itertools
    import json
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_floor_review.py')
    print('H4 original-floor readback: no Play, reset, collider or material change.', flush=True)
    prepared = r['load_preflight']()
    h = r['inventory'].retained.completion
    info, _, manifest, geometry, inventory, _ = h.load_closeout(h.DIRECTORY)
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Retained actual H3 FINAL must remain PAUSED; no reset/reopen/Play/Stop')
    if Path(stage.GetRootLayer().realPath).resolve() != r['inventory'].retained.SOURCE.resolve():
        raise RuntimeError('Original retained hospital root required')
    before_layers = (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_final_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') != info['final_forward_result']):
        raise RuntimeError('Actual cohort_03 FINAL PASS required, not a reset or failed scene')
    prep = info['final_forward_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    support_values = support['support_readback'](stage, manifest)
    if support_values != prep['live_support_readback']:
        raise RuntimeError('Retained H3 support/material/settings changed')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Existing tensor view unavailable; do not initialize or press Play')
    view = sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count != 1:
        raise RuntimeError('Exactly one retained articulation required')

    def numeric(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            value = value.numpy()
        result = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(result).all():
            raise RuntimeError('Finite retained tensors required')
        return result.tolist()

    def snapshot():
        names = list(view.shared_metatype.dof_names)
        settings = {}
        for key, getter in (('kp', view.get_dof_stiffnesses), ('kd', view.get_dof_dampings),
                ('force_limit', view.get_dof_max_forces), ('velocity_target_rad_s', view.get_dof_velocity_targets)):
            settings[key] = dict(zip(names, numeric(getter())))
        return dict(names=names, transform=numeric(view.get_root_transforms()), positions=numeric(view.get_dof_positions()),
            velocity=numeric(view.get_root_velocities()), settings=settings)

    initial = snapshot()
    checks = h.validate_tensor(info, initial['transform'], initial['names'], initial['positions'], initial['velocity'], initial['settings'])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False, ignoreVisibility=True)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    roots = []
    for path in r['FLOORS']:
        prim = stage.GetPrimAtPath(path)
        old = prepared['original_floor_roots'][path]
        if not prim.IsValid() or not prim.IsActive():
            raise RuntimeError('Original corridor floor missing/inactive: '+path)
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if (not np.allclose(list(bounds.GetMin()), old['world_aligned_visible_bound']['minimum_m'], rtol=0, atol=2e-5)
                or not np.allclose(list(bounds.GetMax()), old['world_aligned_visible_bound']['maximum_m'], rtol=0, atol=2e-5)
                or str(UsdGeom.Imageable(prim).ComputeVisibility()) != old['computed_visibility']):
            raise RuntimeError('Original floor moved or visibility changed: '+path)
        meshes = []; collision_flags = []
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        for child in Usd.PrimRange(prim, predicate):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                enabled = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False and child.IsActive()
                collision_flags.append(dict(path=str(child.GetPath()), collision_enabled=bool(enabled)))
                if enabled:
                    raise RuntimeError('Previously visual-only floor collider became enabled: '+str(child.GetPath()))
            if not child.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(child)
            matrix = transforms.GetLocalToWorldTransform(child)
            material, _ = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial('physics')
            row = dict(path=str(child.GetPath()), is_active=bool(child.IsActive()),
                computed_visibility=str(UsdGeom.Imageable(child).ComputeVisibility()),
                world_points_m=[list(map(float, matrix.Transform(Gf.Vec3d(*point)))) for point in mesh.GetPointsAttr().Get()],
                face_vertex_counts=list(map(int, mesh.GetFaceVertexCountsAttr().Get())),
                face_vertex_indices=list(map(int, mesh.GetFaceVertexIndicesAttr().Get())),
                double_sided=mesh.GetDoubleSidedAttr().Get(),
                resolved_physics_material_path=str(material.GetPath()) if material else None)
            row['flat_mesh_summary'] = r['mesh_summary'](row)
            meshes.append(row)
        if not meshes:
            raise RuntimeError('No actual authored floor mesh found: '+path)
        roots.append(dict(path=path, meshes=meshes, original_collision_flags=collision_flags,
            root_AABB_does_NOT_replace_surface_topology=True))
    ground = stage.GetPrimAtPath(support['GROUND'])
    cube = UsdGeom.Cube(ground)
    half = float(cube.GetSizeAttr().Get())/2
    matrix = transforms.GetLocalToWorldTransform(ground)
    corners = [list(map(float, matrix.Transform(Gf.Vec3d(*point)))) for point in itertools.product((-half, half), repeat=3)]
    material, _ = UsdShade.MaterialBindingAPI(ground).ComputeBoundMaterial('physics')
    if (initial != snapshot() or timeline.is_playing() or timeline.is_stopped()
            or before_layers != (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
            or h.sha(r['inventory'].retained.SOURCE) != r['inventory'].retained.SOURCE_SHA):
        raise RuntimeError('Retained H3 body/joints/settings/layers/source/timeline changed; no restoration attempted')
    h.verify_files(prepared['input_sha256'])
    report = dict(version=r['VERSION'], status='ORIGINAL_VISUAL_FLOOR_TOPOLOGY_RECORDED_SUPPORT_SETUP_NOT_APPLIED',
        original_floor_roots=roots, canonical_support_world_corners_m=corners,
        canonical_support_resolved_material=str(material.GetPath()) if material else None,
        unchanged_H3_support_readback=support_values, retained_H3_tensor_checks=checks,
        H3_body_12_joints_settings_root_session_source_and_timeline_unchanged=True,
        visual_floors_remain_without_enabled_collision=True,
        no_floor_extension_collider_or_material_assignment_performed=True,
        no_static_or_drive_admission=True, new_PhysX_executed=False,
        CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**prepared['input_sha256'], str(r['PREFLIGHT']): h.sha(r['PREFLIGHT'])})
    r['TARGET'].parent.mkdir(parents=True, exist_ok=True)
    with r['TARGET'].open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 ORIGINAL CORRIDOR FLOOR SURFACES / CANONICAL SUPPORT READBACK (NO MOTION) ===')
    print('H3 body mm/deg / joints rad:', 1000*checks['position_difference_m'], checks['yaw_difference_deg'], checks['maximum_joint_difference_rad'])
    for row in roots:
        print(row['path'], 'mesh count:', len(row['meshes']), 'polygon area SUM (not union) m2:',
            sum(mesh['flat_mesh_summary']['projected_polygon_area_sum_m2'] for mesh in row['meshes']))
    print('canonical support material:', report['canonical_support_resolved_material'])
    print('Original floor surfaces recorded; NO floor/collider/material/robot/timeline change.')
    print('saved:', r['TARGET'])
    print('No H4 static/drive admission. Do not reset/reopen/Play/Stop/save or run old controllers.')


_review_h4_original_floor_surfaces()
