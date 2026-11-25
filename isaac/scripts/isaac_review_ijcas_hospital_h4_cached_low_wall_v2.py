"""GET ONLY import repair v2; same frozen queries and no timeline/physics writes."""


def _review_h4_cached_low_wall():
    import json
    from pathlib import Path
    import runpy

    import carb
    import numpy as np
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_cached_low_wall_review_v2.py')
    if r['LIVE'].exists():
        print('GET-only low-wall review already recorded; NOT rerun:', r['LIVE'])
        return
    protocol = r['load_protocol']()
    s = r['static']; c = s.cohort
    raw = json.loads(s.RESULT.read_bytes()); c.verify(raw['input_sha256'])
    stage, timeline = omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Actual reopened-static PASS must remain PAUSED; no Play/initialize to recover it')
    marker = stage.GetPrimAtPath(c.MARKER)
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'static_support_passed'
            or marker.GetCustomDataByKey('static_gate_result') != str(s.RESULT)
            or marker.GetCustomDataByKey('static_gate_sha256') != r['sha'](s.RESULT)):
        raise RuntimeError('Exact actual NEW static PASS marker required, not an old actor or receipt')
    root, session = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    before = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    environment = s.validate_environment(stage)
    if environment != raw['environment_after_static']:
        raise RuntimeError('Original floor/material/scene/Hospital collider baseline changed')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Current retained tensor view unavailable; do NOT Play or initialize')
    tensor = sim.create_articulation_view(c.ROBOT+'/base_link')
    if tensor.count != 1 or tensor.max_links != 13:
        raise RuntimeError('One retained thirteen-link H4 actor required')

    def numeric(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            value = value.numpy()
        return np.asarray(value, float).reshape(-1).tolist()

    def snapshot():
        return dict(transform=numeric(tensor.get_root_transforms()), names=list(tensor.shared_metatype.dof_names),
            positions=numeric(tensor.get_dof_positions()), velocity=numeric(tensor.get_root_velocities()),
            link_transforms=numeric(tensor.get_link_transforms()),
            gains=dict(kp=numeric(tensor.get_dof_stiffnesses()), kd=numeric(tensor.get_dof_dampings()),
                max_efforts=numeric(tensor.get_dof_max_forces())))

    actual = snapshot()
    continuity = r['continuity'].validate_tensor(raw, actual['transform'], actual['names'], actual['positions'], actual['velocity'], actual['gains'])
    links = s.passive_final_link_readback(tensor)
    expected_links = raw['passive_post_static_link_readback']
    if links['link_names'] != expected_links['link_names']:
        raise RuntimeError('Actual thirteen-link correspondence changed')
    a = np.asarray(links['all_thirteen_link_world_transforms_xyzw'], float)
    b = np.asarray(expected_links['all_thirteen_link_world_transforms_xyzw'], float)
    link_delta = float(np.max(np.linalg.norm(a[:, :3]-b[:, :3], axis=1)))
    qa, qb = a[:, 3:]/np.linalg.norm(a[:, 3:], axis=1)[:, None], b[:, 3:]/np.linalg.norm(b[:, 3:], axis=1)[:, None]
    link_angle = float(np.max(np.degrees(2*np.arccos(np.clip(np.abs(np.sum(qa*qb, axis=1)), 0., 1.)))))
    if (link_delta > protocol['original_guard_tolerances']['body_position_m']
            or link_angle > protocol['original_guard_tolerances']['yaw_deg']):
        raise RuntimeError('Actual links differ from this own static PASS; no recovery/reset')
    query = omni.physx.get_physx_scene_query_interface()
    observations = []
    for spec in protocol['sampled_rays']:
        hits, errors = [], []

        def receive(hit):
            try:
                hits.append(dict(collision=str(hit.collision), position_m=list(map(float, hit.position)),
                    normal=list(map(float, hit.normal)), distance_m=float(hit.distance)))
                return True
            except Exception as error:
                errors.append(type(error).__name__+': '+str(error))
                return False

        try:
            query.raycast_all(carb.Float3(*spec['origin_m']), carb.Float3(*spec['direction']), spec['distance_m'], receive)
        except Exception as error:
            errors.append(type(error).__name__+': '+str(error))
        observations.append(dict(id=spec['id'], all_hits=sorted(hits, key=lambda h: h['distance_m']), errors=errors))
    result = r['evaluate'](protocol, observations)
    after = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    result['checks']['Root_Session_timeline_body_joints_links_and_gains_unchanged'] = (
        root == stage.GetRootLayer().ExportToString() and session == stage.GetSessionLayer().ExportToString()
        and before == after and actual == snapshot() and s.validate_environment(stage) == environment)
    r['verify'](protocol['input_sha256'])
    result = dict(result, version=r['VERSION'], status='CACHED_FLOOR_AND_ORIGINAL_LOW_WALL_VERIFIED_NO_DRIVE' if all(result['checks'].values())
        else 'HOLD_CACHED_LOW_WALL_QUERY_OR_CONTINUITY_INCOMPLETE',
        actual_new_static_result=str(s.RESULT), actual_new_static_raw_verdict='PASS', live_continuity=continuity,
        maximum_saved_live_link_position_difference_m=link_delta,
        maximum_saved_live_link_orientation_difference_deg=link_angle, current_link_readback=links,
        actual_cached_queries=observations, actual_drive_admission=False, use_for_final_statistics=False,
        any_Play_Stop_Pause_reset_update_reopen_save_or_physics_initialization=False,
        floor_height_material_map_obstacle_flags_control_or_contact_policy_changed=False,
        actual_contact_offset_old_force_or_old_penetration_measured=False,
        CICS_runtime_integration=False,
        input_sha256={**protocol['input_sha256'], str(r['PROTOCOL']): r['sha'](r['PROTOCOL'])})
    r['write_exclusive'](r['LIVE'], result)
    print('=== H4 RETAINED NEW STATIC / CACHED LOW WALL AND FLOOR REVIEW (GET ONLY) ===')
    print('actual static stays PASS; old static/P route FAIL preserved')
    print('saved/live body mm/deg / joints rad:', 1000*continuity['position_difference_m'], continuity['heading_difference_deg'], continuity['maximum_joint_difference_rad'])
    print('all thirteen links retained; max position delta mm:', 1000*link_delta)
    for row in result['query_diagnostics']:
        print(row['id'], 'original surface / floor:', row['expected_original_collider_surface_recorded'], row['original_top_support_recorded'])
    print('failed checks:', [k for k, value in result['checks'].items() if not value])
    print('STATUS:', result['status']); print('saved:', r['LIVE'])
    print('No Play, map/support/contact-policy change, retry or drive admission. Contact margin/penetration not measured.')


_review_h4_cached_low_wall()
