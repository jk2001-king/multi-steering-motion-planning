"""Get-only H4 geometry review; retain the actual PAUSED H3 FINAL-response PASS."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _review_retained_h4_geometry():
    import hashlib
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    workspace = _ijcas_workspace()
    scripts = _ijcas_path(workspace, 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_retained_review.py')
    preflight = r['load_preflight']()
    h = r['completion']
    info, _, manifest, geometry, inventory, _ = h.load_closeout(h.DIRECTORY)
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    timeline = omni.timeline.get_timeline_interface()
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Require the same actual FINAL-response PASS PAUSED; do not reopen/reset/Play/Stop')
    if Path(stage.GetRootLayer().realPath).resolve() != r['SOURCE'].resolve():
        raise RuntimeError('The retained original hospital root is required')
    if (not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.)
            or str(UsdGeom.GetStageUpAxis(stage)) != 'Z'):
        raise RuntimeError('Expected original metre-based Z-up stage')
    before_layers = (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_final_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') != info['final_forward_result']):
        raise RuntimeError('The actual cohort_03 FINAL response state is required, not a reset scene')
    prep = info['final_forward_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Original support/contact settings changed; no correction attempted')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Existing tensor view unavailable; no initialize/Play/reset is permitted')
    view = sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count != 1:
        raise RuntimeError('Exactly one retained articulation required')

    def numeric(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            value = value.numpy()
        value = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(value).all():
            raise RuntimeError('Finite retained tensor required')
        return value.tolist()

    def snapshot():
        names = list(view.shared_metatype.dof_names)
        settings = {}
        for key, getter in (('kp', view.get_dof_stiffnesses), ('kd', view.get_dof_dampings),
                ('force_limit', view.get_dof_max_forces), ('velocity_target_rad_s', view.get_dof_velocity_targets)):
            settings[key] = dict(zip(names, numeric(getter())))
        return dict(names=names, transform=numeric(view.get_root_transforms()), positions=numeric(view.get_dof_positions()),
            velocity=numeric(view.get_root_velocities()), settings=settings)

    initial = snapshot()
    checks = h.validate_tensor(info, initial['transform'], initial['names'], initial['positions'],
        initial['velocity'], initial['settings'])
    hospital = stage.GetPrimAtPath('/World/Hospital')
    if not hospital.IsValid():
        raise RuntimeError('Existing hospital root is missing')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=False)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())

    def bound(prim):
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = list(map(float, value.GetMin())), list(map(float, value.GetMax()))
        if not all(math.isfinite(v) for v in (*lo, *hi)) or any(a > b for a, b in zip(lo, hi)):
            return None
        return dict(minimum_m=lo, maximum_m=hi, size_m=[b-a for a, b in zip(lo, hi)])

    def collision(prim):
        values = []
        for item in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if (not item.IsActive() or not item.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(item).GetCollisionEnabledAttr().Get() is False):
                continue
            value = dict(path=str(item.GetPath()), type_name=item.GetTypeName(), world_aligned_bound=bound(item))
            if item.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(item)
                transform = transforms.GetLocalToWorldTransform(item)
                value['authored_mesh'] = dict(world_points_m=[list(map(float, transform.Transform(Gf.Vec3d(*point))))
                        for point in mesh.GetPointsAttr().Get()],
                    face_vertex_counts=list(map(int, mesh.GetFaceVertexCountsAttr().Get())),
                    face_vertex_indices=list(map(int, mesh.GetFaceVertexIndicesAttr().Get())),
                    meaning='authored_mesh_NOT_cooked_PhysX_shape')
            values.append(value)
        return dict(enabled_collision_prim_count=len(values), primitives=values)

    assets = []
    roots = list(hospital.GetChildren())
    for prim in roots:
        if not prim.IsActive() or not prim.IsA(UsdGeom.Imageable):
            continue
        visible = bound(prim)
        if r['geometry'].intersects_xy(visible):
            assets.append(dict(path=str(prim.GetPath()),
                computed_visibility=str(UsdGeom.Imageable(prim).ComputeVisibility()),
                world_aligned_visible_bound=visible, collision=collision(prim)))
    original = json.loads(r['AUDIT'].read_bytes())
    selected = r['geometry'].validate_live_targets(assets, original['candidate_paths'])
    if (initial != snapshot() or timeline.is_playing() or timeline.is_stopped()
            or before_layers != (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
            or h.sha(r['SOURCE']) != r['SOURCE_SHA']):
        raise RuntimeError('Read-only body/joints/gains/root/session/source/timeline invariant failed')
    h.verify_files(preflight['input_sha256'])
    receipt = dict(version=r['VERSION'], status='RETAINED_H3_FINAL_H4_AUTHORED_GEOMETRY_RECORDED_TASK_SETUP_PENDING',
        retained_final_response_result=info['final_forward_result'], retained_tensor_checks=checks,
        retained_all_twelve_joint_positions_rad=dict(zip(initial['names'], initial['positions'])),
        root_session_source_timeline_robot_joints_and_settings_unchanged=True,
        review_region_xy_m=r['geometry'].ROI, all_top_level_roots_examined=len(roots),
        selected_original_assets=selected, assets=sorted(assets, key=lambda row: row['path']),
        authored_meshes_are_NOT_cooked_PhysX_collision_validation=True,
        complete_collision_coverage_for_future_map_proven=False,
        no_H4_start_goal_or_occupancy_map_selected=True, new_obstacle_placed=False,
        H3_abandoned_or_global_infeasibility_claimed=False,
        H4_or_H3_motion_admitted=False, physical_state_reset=False, new_PhysX_executed=False,
        source_USD_saved=False, CICS_runtime_integration=False, use_for_final_statistics=False,
        preflight=str(r['PREFLIGHT']), preflight_sha256=h.sha(r['PREFLIGHT']),
        input_sha256={**preflight['input_sha256'], str(r['PREFLIGHT']): h.sha(r['PREFLIGHT'])})
    r['TARGET'].parent.mkdir(parents=True, exist_ok=True)
    with r['TARGET'].open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    context.get_selection().set_selected_prim_paths(list(r['geometry'].REVIEW_PATHS), True)
    print('=== H4 AUTHORED CORRIDOR / CART GEOMETRY REVIEW WITH H3 FINAL RETAINED (NO MOTION) ===')
    print('H3 FINAL body mm/deg:', 1000*checks['position_difference_m'], checks['yaw_difference_deg'])
    print('all twelve joints retained; max delta rad:', checks['maximum_joint_difference_rad'])
    print('original selected wall/cart assets:', len(selected), '; local asset roots:', len(assets))
    print('saved:', r['TARGET'])
    print('F then Top View to inspect the existing corridor/cart; no obstacle or task pose was moved.')
    print('H4 authored geometry only; cooked collisions, strict task map, support and driving are NOT validated.')
    print('H3 FINAL remains PAUSED and unchanged; do not reopen/reset/Play/save or use the old STOPPED H4 inspector.')


_review_retained_h4_geometry()
