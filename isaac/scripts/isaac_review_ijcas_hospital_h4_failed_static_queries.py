"""GET ONLY: retained actual static FAIL and SAME cached ray samples.

No Play/Stop/Pause, updates, physics initialization, seed assignment or edits.
Even a consistent diagnostic does NOT change the original FAIL or admit drive.
"""


def _review_failed_static_queries():
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
    r = load(scripts/'hospital_h4_static_query_review.py')
    review = r['load_review']()
    if not all(review['checks'].values()):
        raise RuntimeError('Offline diagnosis incomplete; no substitute or recovery allowed')
    if r['LIVE'].exists():
        raise RuntimeError('This single diagnostic is already recorded; no overwrite')
    launch, c, stationary = r['launch'], r['cohort'], r['static']
    raw = json.loads(stationary.RESULT.read_bytes())
    c.verify(raw['input_sha256'])
    stage, timeline = omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Actual failed H4 static must remain PAUSED; no Play to recover data')
    marker = stage.GetPrimAtPath(c.MARKER)
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'static_support_failed'
            or marker.GetCustomDataByKey('static_gate_result') != str(stationary.RESULT)
            or marker.GetCustomDataByKey('static_gate_sha256') != c.completion.sha(stationary.RESULT)):
        raise RuntimeError('Exact original actual static FAIL marker required')
    root_before, session_before = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    before = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    environment = stationary.validate_environment(stage)
    if environment != raw['environment_after_static']:
        raise RuntimeError('Original environment differs from the failed static receipt')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Retained tensor view unavailable; do NOT Play or initialize it')
    tensor = sim.create_articulation_view(c.ROBOT+'/base_link')
    if tensor.count != 1 or tensor.max_links != 13:
        raise RuntimeError('One retained thirteen-body H4 actor required')

    def numeric(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            value = value.numpy()
        return np.asarray(value, float).reshape(-1)

    def snapshot():
        return dict(transform=numeric(tensor.get_root_transforms()).tolist(),
            names=list(tensor.shared_metatype.dof_names), positions=numeric(tensor.get_dof_positions()).tolist(),
            velocity=numeric(tensor.get_root_velocities()).tolist(),
            gains=dict(kp=numeric(tensor.get_dof_stiffnesses()).tolist(), kd=numeric(tensor.get_dof_dampings()).tolist(),
                max_efforts=numeric(tensor.get_dof_max_forces()).tolist()))

    actual = snapshot()
    continuity = launch.validate_tensor(raw, actual['transform'], actual['names'], actual['positions'],
        actual['velocity'], actual['gains'])
    protocol = stationary.load_protocol()
    query = omni.physx.get_physx_scene_query_interface()
    queries = []
    for spec in protocol['sampled_cooked_support_queries']:
        hits, errors, own = [], [], []

        def receive(hit):
            try:
                row = dict(collision=str(hit.collision), position_m=list(map(float, hit.position)),
                    normal=list(map(float, hit.normal)), distance_m=float(hit.distance))
                (own if row['collision'].startswith(c.ROBOT+'/') else hits).append(row)
                return True
            except Exception as error:
                errors.append(type(error).__name__+': '+str(error))
                return False

        try:
            query.raycast_all(carb.Float3(*spec['xy_m'], .6), carb.Float3(0., 0., -1.), .8, receive)
        except Exception as error:
            errors.append(type(error).__name__+': '+str(error))
        hits.sort(key=lambda hit: hit['distance_m'])
        first = dict(hit=True, **hits[0]) if hits else dict(hit=False)
        queries.append(dict(id=spec['id'], xy_m=spec['xy_m'], **first, errors=errors,
            all_nonself_hits=hits, own_robot_occlusion_hits=own))
    inventory_path = c.support.floor.inventory.TARGET
    inventory = json.loads(inventory_path.read_bytes())
    with np.load(launch.cohort.MAP) as grid:
        diagnosed = r['diagnose_queries'](protocol['sampled_cooked_support_queries'], queries, protocol['support_top_m'],
            inventory['collision_prims'], grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m']))
    failed = [row for row in diagnosed if not row['original_closest_hit_check']]
    checks = dict(actual_failed_static_body_and_twelve_joint_continuity=True,
        all_original_query_transactions_available=all(row['all_hit_transaction_complete'] for row in diagnosed),
        all_original_top_supports_recorded=all(row['support_at_original_height_recorded'] for row in diagnosed),
        every_occluder_stays_an_original_mapped_blocked_obstacle=all(
            row['every_non_support_hit_is_original_mapped_blocked_obstacle'] for row in diagnosed),
        same_failed_auxiliary_point_and_obstacle_retained=len(failed) == 1
            and failed[0]['id'] == review['failed_query_id']
            and failed[0]['original_collision'] == review['original_existing_occluding_obstacle'])
    after = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
    checks['Root_Session_timeline_body_joints_gains_unchanged'] = (before == after
        and root_before == stage.GetRootLayer().ExportToString() and session_before == stage.GetSessionLayer().ExportToString()
        and actual == snapshot())
    pins = {**review['input_sha256'], str(r['TARGET']): c.completion.sha(r['TARGET']),
        str(stationary.RESULT): c.completion.sha(stationary.RESULT)}
    c.verify(pins)
    # Runtime motor effort limits may be +inf. Do not invent finite values.
    for name in ('max_efforts',):
        actual['gains'][name] = [value if np.isfinite(value) else 'positive_infinity' for value in actual['gains'][name]]
    result = dict(version=r['VERSION'],
        status='LIVE_SAME_FAILED_STATIC_QUERY_DIAGNOSIS_VERIFIED_NO_ADMISSION' if all(checks.values()) else
            'HOLD_LIVE_QUERY_DIAGNOSIS_INCOMPLETE', checks=checks, actual_static_verdict='FAIL',
        actual_static_result=str(stationary.RESULT), static_relabelled_or_criteria_changed=False,
        live_static_continuity=continuity, actual_state=actual,
        actual_cached_queries=queries, query_diagnostics=diagnosed,
        original_obstacle_never_removed_filtered_or_exempted=True,
        revised_query_contract_NOT_applied_or_authorized=True,
        additional_Play_Stop_Pause_reset_update_reopen_or_save=False,
        actual_drive_admission=False, CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256=pins)
    c.save_exclusive(r['LIVE'], result)
    print('=== H4 ACTUAL FAILED STATIC / CACHED RAY OCCLUSION REVIEW (NO MOTION) ===')
    print('original static verdict: FAIL; preserved, NOT relabelled')
    print('saved/live mm/deg:', 1000*continuity['position_difference_m'], continuity['heading_difference_deg'])
    print('all twelve joints retained; max delta rad:', continuity['maximum_joint_difference_rad'])
    print('original-top support hits:', sum(row['support_at_original_height_recorded'] for row in diagnosed), '/', len(diagnosed))
    print('same mapped obstacle / auxiliary point:', checks['same_failed_auxiliary_point_and_obstacle_retained'])
    print('STATUS:', result['status']); print('saved:', r['LIVE'])
    print('No retry, Play, obstacle change, new static PASS or drive admission. Separate query-contract decision pending.')


_review_failed_static_queries()
