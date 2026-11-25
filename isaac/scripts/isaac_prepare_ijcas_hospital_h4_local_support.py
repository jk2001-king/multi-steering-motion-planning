"""Author ONLY an audited local H4 support Session subtree; no robot motion."""


def _prepare_h4_local_support():
    import json
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_support_design.py')
    print('H4 authorized Session support preparation: no Play, reset, robot or old-floor change.', flush=True)
    design = r['load_design']()
    if r['LIVE'].exists():
        raise RuntimeError('Preparation receipt already exists; no automatic second authoring or overwrite')
    h = r['floor'].inventory.retained.completion
    info, _, manifest, geometry, inventory, _ = h.load_closeout(h.DIRECTORY)
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Actual retained H3 FINAL must remain PAUSED; no reopen/reset/Play/Stop')
    if Path(stage.GetRootLayer().realPath).resolve() != r['floor'].inventory.retained.SOURCE.resolve():
        raise RuntimeError('Original retained hospital source required')
    if stage.GetPrimAtPath(r['PATCH_ROOT']).IsValid():
        raise RuntimeError('Exact new H4 support subtree already exists; never overwrite it')
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_final_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') != info['final_forward_result']):
        raise RuntimeError('Actual cohort_03 final endpoint required, not an old/reset/failed state')
    prep = info['final_forward_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Canonical H3 support/material/gains changed')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Existing tensor view unavailable; do not initialize or press Play')
    view = sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count != 1:
        raise RuntimeError('Exactly one retained H3 articulation required')

    def numeric(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            value = value.numpy()
        values = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise RuntimeError('Finite retained body/joint tensors required')
        return values.tolist()

    def snapshot():
        names = list(view.shared_metatype.dof_names)
        settings = {}
        for key, getter in (('kp', view.get_dof_stiffnesses), ('kd', view.get_dof_dampings),
                ('force_limit', view.get_dof_max_forces), ('velocity_target_rad_s', view.get_dof_velocity_targets)):
            settings[key] = dict(zip(names, numeric(getter())))
        return dict(names=names, transform=numeric(view.get_root_transforms()), positions=numeric(view.get_dof_positions()),
            velocity=numeric(view.get_root_velocities()), settings=settings)

    initial = snapshot()
    body_checks = h.validate_tensor(info, initial['transform'], initial['names'], initial['positions'], initial['velocity'], initial['settings'])
    original_floor_record = json.loads(r['floor'].TARGET.read_bytes())
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    for root in original_floor_record['original_floor_roots']:
        for saved in root['meshes']:
            prim = stage.GetPrimAtPath(saved['path'])
            if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
                raise RuntimeError('Original visual floor mesh missing or changed: '+saved['path'])
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                raise RuntimeError('Original visual-only floor collider unexpectedly enabled')
            mesh = UsdGeom.Mesh(prim); matrix = transforms.GetLocalToWorldTransform(prim)
            points = [list(map(float, matrix.Transform(Gf.Vec3d(*point)))) for point in mesh.GetPointsAttr().Get()]
            if (not np.allclose(points, saved['world_points_m'], rtol=0, atol=2e-5)
                    or list(mesh.GetFaceVertexCountsAttr().Get()) != saved['face_vertex_counts']
                    or list(mesh.GetFaceVertexIndicesAttr().Get()) != saved['face_vertex_indices']
                    or str(UsdGeom.Imageable(prim).ComputeVisibility()) != saved['computed_visibility']):
                raise RuntimeError('Original visible floor topology/transform/visibility changed')
    session = stage.GetSessionLayer()
    old_session = session.ExportToString()
    old_root = stage.GetRootLayer().ExportToString()
    material = UsdShade.Material.Get(stage, design['canonical_support_material'])
    if not material:
        raise RuntimeError('Original canonical material is unavailable; do not create/substitute material')
    created = False
    try:
        with Usd.EditContext(stage, session):
            parent = UsdGeom.Xform.Define(stage, r['PATCH_ROOT'])
            created = True
            anchor = [design['patch_planar_polygon'][0][0],
                design['canonical_unchanged_support_bound']['maximum_m'][1], design['top_height_m']]
            parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*anchor))
            parent.GetPrim().SetCustomDataByKey('condition', 'H4_local_same_top_material_support_session_only_no_drive')
            parent.GetPrim().SetCustomDataByKey('support_design_sha256', h.sha(r['TARGET']))
            parent.GetPrim().SetCustomDataByKey('support_design_path', str(r['TARGET']))
            mesh = UsdGeom.Mesh.Define(stage, r['PATCH_MESH'])
            local = [Gf.Vec3f(*(point[i]-anchor[i] for i in range(3))) for point in design['patch_world_mesh']['world_points_m']]
            mesh.CreatePointsAttr().Set(Vt.Vec3fArray(local))
            mesh.CreateFaceVertexCountsAttr().Set(design['patch_world_mesh']['face_vertex_counts'])
            mesh.CreateFaceVertexIndicesAttr().Set(design['patch_world_mesh']['face_vertex_indices'])
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr().Set(True)
            mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set('none')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
        # Verify only the NEW subtree was authored: remove it from an isolated
        # in-memory layer COPY, never from the live stage, then compare text.
        scratch = Sdf.Layer.CreateAnonymous('h4_support_delta_validation')
        if not scratch.ImportFromString(session.ExportToString()):
            raise RuntimeError('Cannot verify isolated Session layer delta')
        edit = Sdf.BatchNamespaceEdit(); edit.Add(r['PATCH_ROOT'], Sdf.Path.emptyPath)
        if not scratch.Apply(edit) or scratch.ExportToString() != old_session:
            raise RuntimeError('Session delta exceeds the one authorized new H4 subtree')
        live_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(r['PATCH_MESH']))
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(live_mesh.GetPrim())
        actual_points = [list(map(float, matrix.Transform(Gf.Vec3d(*point)))) for point in live_mesh.GetPointsAttr().Get()]
        if not np.allclose(actual_points, design['patch_world_mesh']['world_points_m'], rtol=0, atol=2e-5):
            raise RuntimeError('Authored support does not match the frozen world geometry')
        count = len(design['patch_planar_polygon'])
        actual_top = [tuple(point[:2]) for point in actual_points[:count]]
        triangles = [[actual_top[index] for index in face] for face in r['ear_triangles'](actual_top)]
        original_triangles, _ = r['floor_triangles'](original_floor_record)
        residual = r['uncovered'](triangles, [*original_triangles, design['explicit_join_strip']])
        if sum(abs(r['area'](poly)) for poly in residual) > r['AREA_TOLERANCE_M2']:
            raise RuntimeError('Float-authored support extends outside audited original floor/seam topology')
        bound, _ = UsdShade.MaterialBindingAPI(live_mesh.GetPrim()).ComputeBoundMaterial('physics')
        if (not bound or str(bound.GetPath()) != design['canonical_support_material']
                or UsdPhysics.CollisionAPI(live_mesh.GetPrim()).GetCollisionEnabledAttr().Get() is not True
                or str(UsdPhysics.MeshCollisionAPI(live_mesh.GetPrim()).GetApproximationAttr().Get()) != 'none'
                or live_mesh.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)):
            raise RuntimeError('Actual static support collision/material differs from design')
        if (initial != snapshot() or timeline.is_playing() or timeline.is_stopped()
                or stage.GetRootLayer().ExportToString() != old_root
                or h.sha(r['floor'].inventory.retained.SOURCE) != r['floor'].inventory.retained.SOURCE_SHA
                or support['support_readback'](stage, manifest) != prep['live_support_readback']):
            raise RuntimeError('Existing H3 body/joints/gains/support/root/source/timeline changed; no reset permitted')
        h.verify_files(design['input_sha256'])
        report = dict(version=r['VERSION'], status='H4_LOCAL_SUPPORT_AUTHORED_SESSION_ONLY_COOKED_STATIC_AND_DRIVE_PENDING',
            support_design=str(r['TARGET']), support_design_sha256=h.sha(r['TARGET']),
            only_new_session_subtree=r['PATCH_ROOT'], isolated_layer_copy_delta_check=True,
            actual_world_points_m=actual_points, actual_physics_material=str(bound.GetPath()),
            actual_top_union_subtraction_uncovered_area_m2=sum(abs(r['area'](poly)) for poly in residual),
            actual_closed_static_triangle_mesh_approximation='none',
            retained_H3_tensor_checks=body_checks,
            existing_H3_body_12_joints_gains_support_root_source_and_timeline_unchanged=True,
            original_floor_wall_cart_stair_door_threshold_attributes_unchanged=True,
            actual_cooked_queries_continuous_contact_and_static_NOT_validated=True,
            no_Play_Stop_reset_reopen_or_save_performed=True, new_PhysX_executed=False,
            H4_robot_staging_or_driving_admitted=False, CICS_runtime_integration=False,
            use_for_final_statistics=False,
            input_sha256={**design['input_sha256'], str(r['TARGET']): h.sha(r['TARGET'])})
        r['LIVE'].parent.mkdir(parents=True, exist_ok=True)
        with r['LIVE'].open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    except Exception:
        # Recover ONLY this script's newly created subtree, never a preexisting
        # asset/robot/source. No state reset, gain restoration or retry.
        if created:
            rollback = Sdf.BatchNamespaceEdit(); rollback.Add(r['PATCH_ROOT'], Sdf.Path.emptyPath)
            if not session.Apply(rollback) or session.ExportToString() != old_session:
                print('H4 support preparation HOLD: own new subtree rollback incomplete; do not Play/save.', flush=True)
            else:
                print('Removed only this attempt\'s new H4 Session subtree; original Session restored, no retry.', flush=True)
        raise
    print('=== H4 LOCAL ORIGINAL-FLOOR SUPPORT PREPARED (SESSION ONLY, NO MOTION) ===')
    print('new subtree:', r['PATCH_ROOT'])
    print('support area m2 / top m:', design['patch_world_mesh']['support_area_m2'], design['top_height_m'])
    print('same canonical material:', report['actual_physics_material'])
    print('stair/floor-topology checks: verified; no old asset/collider flag changed')
    print('H3 body mm/deg / joints rad:', 1000*body_checks['position_difference_m'], body_checks['yaw_difference_deg'], body_checks['maximum_joint_difference_rad'])
    print('saved:', r['LIVE'])
    print('Keep PAUSED. No H4 robot staging, cooked/static/drive admission, reset/reopen/Play/Stop/save or old controller.')


_prepare_h4_local_support()
