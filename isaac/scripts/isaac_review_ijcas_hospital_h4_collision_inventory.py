"""All-stage collision inventory and optional cached queries. NO motion/writes."""


def _review_h4_complete_collision_inventory():
    import json
    import math
    from pathlib import Path
    import runpy
    import time

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    workspace = Path('/home/jk/Research/IJCAS_ws')
    scripts = workspace/'studies/IJCAS/scripts'
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_collision_inventory.py')
    print('H4 get-only inventory: checking frozen receipts; no Play or robot command.', flush=True)
    prepared = r['load_preflight']()
    h = r['retained'].completion
    info, _, manifest, geometry, inventory, _ = h.load_closeout(h.DIRECTORY)
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('The same H3 FINAL-response PASS must remain PAUSED; no reset/reopen/Play/Stop')
    if Path(stage.GetRootLayer().realPath).resolve() != r['retained'].SOURCE.resolve():
        raise RuntimeError('Original retained hospital root required')
    before_layers = (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_final_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') != info['final_forward_result']):
        raise RuntimeError('Actual cohort_03 final endpoint required, not a reset or failed scene')
    prep = info['final_forward_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Retained support/contact settings changed')
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
    retained_checks = h.validate_tensor(info, initial['transform'], initial['names'], initial['positions'],
        initial['velocity'], initial['settings'])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
        useExtentsHint=False, ignoreVisibility=True)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = []; unloaded = []; errors = []; traversed = 0
    heartbeat = time.monotonic()

    def bound(prim):
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = list(map(float, aligned.GetMin())), list(map(float, aligned.GetMax()))
        if not all(math.isfinite(value) for value in (*lo, *hi)) or any(a > b for a, b in zip(lo, hi)):
            raise ValueError('Unavailable finite collider bounds: '+str(prim.GetPath()))
        return dict(minimum_m=lo, maximum_m=hi, size_m=[b-a for a, b in zip(lo, hi)])

    def rigid_ancestors(prim):
        values = []
        current = prim
        while current and current.GetPath() != stage.GetPseudoRoot().GetPath():
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(current)
                values.append(dict(path=str(current.GetPath()),
                    rigid_body_enabled=body.GetRigidBodyEnabledAttr().Get(),
                    kinematic_enabled=body.GetKinematicEnabledAttr().Get()))
            current = current.GetParent()
        return values

    # All active/inactive/loaded collision prims, including instance proxies.
    # Never first choose a visual root or filter by render visibility.
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        traversed += 1
        if prim.IsActive() and prim.HasPayload() and not prim.IsLoaded():
            unloaded.append(str(prim.GetPath()))
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        record = dict(path=path, type_name=prim.GetTypeName(), is_active=bool(prim.IsActive()),
            is_instance_proxy=bool(prim.IsInstanceProxy()), collision_enabled_value=enabled)
        if not prim.IsActive():
            record['classification'] = 'inactive'
        elif enabled is False:
            record['classification'] = 'disabled'
        elif path == support['ROBOT'] or path.startswith(support['ROBOT']+'/'):
            record['classification'] = 'robot_self_excluded'
        else:
            try:
                value = bound(prim)
                record['world_aligned_bound'] = value
                if not r['intersects_roi'](value, prepared['review_roi_xy_m']):
                    record['classification'] = 'outside_ROI'
                else:
                    record['classification'] = 'enabled_in_ROI'
                    record['rigid_body_ancestors'] = rigid_ancestors(prim)
                    record['computed_visibility_not_used_to_filter'] = str(UsdGeom.Imageable(prim).ComputeVisibility()) if prim.IsA(UsdGeom.Imageable) else None
                    record['physics_attributes'] = {}
                    for name in ('physics:approximation', 'physxCollision:contactOffset', 'physxCollision:restOffset'):
                        attribute = prim.GetAttribute(name)
                        value = attribute.Get() if attribute else None
                        record['physics_attributes'][name] = value if isinstance(value, (type(None), bool, int, float, str)) else str(value)
                    if prim.IsA(UsdGeom.Mesh):
                        mesh = UsdGeom.Mesh(prim)
                        matrix = transforms.GetLocalToWorldTransform(prim)
                        record['authored_mesh'] = dict(
                            world_points_m=[list(map(float, matrix.Transform(Gf.Vec3d(*point)))) for point in mesh.GetPointsAttr().Get()],
                            face_vertex_counts=list(map(int, mesh.GetFaceVertexCountsAttr().Get())),
                            face_vertex_indices=list(map(int, mesh.GetFaceVertexIndicesAttr().Get())),
                            meaning='authored_geometry_NOT_a_cooked_shape_export')
            except Exception as error:
                record['classification'] = 'bound_unavailable'
                errors.append(path+': '+type(error).__name__+': '+str(error))
        records.append(record)
        if time.monotonic()-heartbeat >= 15.:
            print('H4 get-only inventory:', traversed, 'prims /', len(records), 'collision prims; timeline still PAUSED.', flush=True)
            heartbeat = time.monotonic()
    summary = r['inventory_summary'](records, unloaded, errors)

    queries = dict(available=False, positive_control_hit=False, control=None, endpoint_rays=[], endpoint_boxes=[], errors=[])
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface
        query = get_physx_scene_query_interface()

        def hit_dict(hit):
            value = dict(hit=bool(hit['hit']))
            if value['hit']:
                value.update(collision=str(hit['collision']), rigid_body=str(hit['rigidBody']),
                    position_m=list(map(float, hit['position'])), normal=list(map(float, hit['normal'])),
                    distance_m=float(hit['distance']))
            return value

        # Positive control on the existing cart, never on a newly added shape.
        cart = '/World/Hospital/SM_SupplyCart_02a7'
        candidates = []
        for row in records:
            if row['path'].startswith(cart+'/') and row.get('authored_mesh'):
                mesh = row['authored_mesh']; points = np.asarray(mesh['world_points_m'])
                offset = 0
                for count in mesh['face_vertex_counts']:
                    indices = mesh['face_vertex_indices'][offset:offset+count]; offset += count
                    if count >= 3:
                        candidates.append(np.mean(points[indices], axis=0))
        if not candidates:
            raise RuntimeError('No authored cart surface for cached-query positive control')
        point = max(candidates, key=lambda value: value[2])
        control = hit_dict(query.raycast_closest(carb.Float3(float(point[0]), float(point[1]), float(point[2]+.3)),
            carb.Float3(0., 0., -1.), float(point[2]+.5)))
        queries['control'] = control
        queries['positive_control_hit'] = control['hit'] and (control['collision'] == cart or control['collision'].startswith(cart+'/'))
        for sample in prepared['prescribed_endpoint_ray_samples']:
            x, y = sample['xy_m']
            hit = hit_dict(query.raycast_closest(carb.Float3(x, y, .6), carb.Float3(0., 0., -1.), .8))
            queries['endpoint_rays'].append({**sample, **hit})
        for name, pose in (('start', prepared['candidate_start_xyyaw']), ('goal', prepared['candidate_goal_xyyaw'])):
            contacts = []

            def callback(hit):
                contacts.append(dict(collision=str(getattr(hit, 'collision', 'UNAVAILABLE')),
                    rigid_body=str(getattr(hit, 'rigid_body', 'UNAVAILABLE'))))
                return True

            lower, upper = prepared['query_height_slab_m']
            width = prepared['footprint']['width_m']; length = prepared['footprint']['length_m']
            query.overlap_box(carb.Float3(length/2+.01, width/2+.01, (upper-lower)/2),
                carb.Float3(pose[0], pose[1], (upper+lower)/2),
                carb.Float4(0., 0., math.sin(pose[2]/2), math.cos(pose[2]/2)), callback, False)
            queries['endpoint_boxes'].append(dict(task_pose=name, contacts=contacts,
                no_overlap_reported=len(contacts) == 0))
        queries['available'] = True
    except Exception as error:
        queries['errors'].append(type(error).__name__+': '+str(error))
    queries['query_observations_are_NOT_static_support_or_drive_admission'] = True
    queries['no_initialization_or_Play_was_used'] = True

    if (initial != snapshot() or timeline.is_playing() or timeline.is_stopped()
            or before_layers != (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
            or h.sha(r['retained'].SOURCE) != r['retained'].SOURCE_SHA):
        raise RuntimeError('Body/joints/gains/root/session/source/timeline changed; no restoration or retry attempted')
    h.verify_files(prepared['input_sha256'])
    report = dict(version=r['VERSION'],
        status='COMPLETE_AUTHORED_COLLIDER_INVENTORY_RECORDED_MAP_AND_PHYSICS_ADMISSION_PENDING' if summary['authored_inventory_coverage_complete'] else 'HOLD_AUTHORED_COLLIDER_COVERAGE_INCOMPLETE',
        all_stage_prims_traversed=traversed, inventory_summary=summary, collision_prims=records,
        sampled_cached_PhysX_queries=queries, retained_H3_tensor_checks=retained_checks,
        original_H3_body_12_joints_settings_root_session_source_and_timeline_unchanged=True,
        candidate_task=str(r['SETUP']), review_roi_xy_m=prepared['review_roi_xy_m'],
        all_assets_including_invisible_and_non_hospital_colliders_considered=True,
        cooked_shape_geometry_exported=False, continuous_support_or_swept_collision_proven=False,
        H4_static_or_motion_admitted=False, new_PhysX_executed=False,
        real_robot_validation_required=False, CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**prepared['input_sha256'], str(r['PREFLIGHT']): h.sha(r['PREFLIGHT'])})
    r['TARGET'].parent.mkdir(parents=True, exist_ok=True)
    with r['TARGET'].open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H4 COMPLETE USD COLLIDER INVENTORY / CACHED QUERY REVIEW (NO MOTION) ===')
    print('H3 body mm/deg / joints rad:', 1000*retained_checks['position_difference_m'], retained_checks['yaw_difference_deg'], retained_checks['maximum_joint_difference_rad'])
    print('all-stage prims / collision prims:', traversed, summary['collision_prim_count'])
    print('collision classifications:', summary['classification_counts'])
    print('authored inventory complete:', summary['authored_inventory_coverage_complete'])
    print('cached query available / cart control hit:', queries['available'], queries['positive_control_hit'])
    print('endpoint ray hits:', sum(hit['hit'] for hit in queries['endpoint_rays']), '/', len(queries['endpoint_rays']))
    print('endpoint box overlap counts:', [len(box['contacts']) for box in queries['endpoint_boxes']])
    print('query errors:', queries['errors'])
    print('saved:', r['TARGET'])
    print('No H4 drive/static admission. H3 final remains PAUSED; no reset/reopen/Play/Stop/save or old controller.')


_review_h4_complete_collision_inventory()
