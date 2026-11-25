"""Read the SAME paused fresh-prefix state; no init/reset/Play or USD edit."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _review_reverse_closedloop_handoff():
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    workspace = _ijcas_workspace()
    scripts = _ijcas_path(workspace, 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts / 'hospital_private_loader.py'))['load']
    h = load(scripts / 'hospital_h3b_reverse_response_observation_v2.py'); info = h['load_reviewed_profile'](); h['load_protocol']()
    support = runpy.run_path(str(scripts / 'hospital_h3_flush_support.py'))
    timeline = omni.timeline.get_timeline_interface(); stage = omni.usd.get_context().get_stage()
    if timeline.is_playing() or timeline.is_stopped() or stage is None:
        raise RuntimeError('Keep the SAME fresh-prefix PASS PAUSED; no Stop/reopen/reset/Play')
    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task, 'done') and not task.done():
            raise RuntimeError('A hospital task is still running')
    root_before = stage.GetRootLayer().ExportToString(); session_before = stage.GetSessionLayer().ExportToString()
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_prefix_response_pilot_finished_fresh_setup_required'
            or marker.GetCustomDataByKey('last_h3b_reverse_force_prefix_pilot_gate') != info['first_result']):
        raise RuntimeError('Not the SAME freshly reproduced first PASS; old first/stalled scenes are not accepted')
    first = info['first_record']; prep = first['physics_preparation_snapshot']
    _, manifest, geometry, inventory = support['load_setup'](Path(first['geometry_manifest']).parent, workspace)
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Support/contact/motor baseline changed')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None: raise RuntimeError('Existing tensor view unavailable; do not initialize or reset it')
    view = sim.create_articulation_view(support['ROBOT'] + '/base_link')
    if view.count != 1: raise RuntimeError('Need exactly one retained live articulation')
    def numeric(value):
        if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'): value = value.numpy()
        result = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(result).all(): raise RuntimeError('Nonfinite tensor readback')
        return result
    names = list(view.shared_metatype.dof_names)
    q = numeric(view.get_dof_positions()); transform = numeric(view.get_root_transforms())
    velocity = numeric(view.get_root_velocities())
    if len(names) != 12 or len(q) != 12 or len(transform) != 7 or len(velocity) != 6:
        raise RuntimeError('Unexpected live tensor shape; no changes attempted')
    x, y, z, w = transform[3:]
    yaw = math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
    pose = [float(transform[0]), float(transform[1]), yaw]; joints = dict(zip(names, map(float, q)))
    checks = h['validate_live'](info, pose, joints, velocity)
    settings={}
    for key,getter in (('kp',view.get_dof_stiffnesses),('kd',view.get_dof_dampings),
            ('force_limit',view.get_dof_max_forces),('velocity_target_rad_s',view.get_dof_velocity_targets)):
        settings[key]=dict(zip(names,map(float,numeric(getter()))))
    tensor_checks=h['validate_tensor'](info,transform,names,q,velocity,settings)

    if (timeline.is_playing() or timeline.is_stopped() or stage.GetRootLayer().ExportToString() != root_before
            or stage.GetSessionLayer().ExportToString() != session_before):
        raise RuntimeError('Timeline/root/session changed during read-only handoff review')
    h['verify_files'](info['input_sha256'])
    target = h['LIVE']; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): raise FileExistsError('One live handoff receipt per cohort; do not stack/rewrite')
    script = _ijcas_path(workspace, 'isaac/scripts/isaac_review_ijcas_hospital_h3b_reverse_response_handoff_v2.py')
    report = dict(status='PAUSED_FRESH_PREFIX_REAR_HANDOFF_REVIEWED_NO_MOTION',
        first_result=info['first_result'], first_result_sha256=info['first_result_sha256'],
        model_admission=info['admission'], model_admission_sha256=info['admission_sha256'],
        live_world_xyyaw=pose, live_root_velocity=velocity.tolist(), retained_all_joint_positions_rad=joints,
        **checks, input_sha256={**info['input_sha256'], str(script): h['sha'](script)},
        script_sha256=h['sha'](script), physical_state_reset=False,retained_tensor_checks=tensor_checks,
        source_USD_and_session_unchanged=True, rear_reverse_PhysX_executed=False,
        full_H3B_turning_path_verified=False, automatic_motion_launched=False)
    with target.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print('=== H3-B SAME MEASURED TERMINAL RESPONSE OBSERVATION FRESH PREFIX LIVE HANDOFF REVIEW (NO MOTION) ===')
    print('saved/live delta mm/deg:', round(checks['position_difference_m']*1000, 6), round(checks['yaw_difference_deg'], 6))
    print('all 12 joints retained; max delta rad:', checks['maximum_joint_difference_rad'])
    print('rear-only reference length mm:', round(info['nominal_reference_reverse_m']*1000, 3))
    print('HANDOFF REVIEW: PASS; NOT rear motion or full H3 success')
    print('saved:', target)
    print('Keep PAUSED. Run ONLY dedicated isaac_run_ijcas_hospital_h3b_reverse_response_instrumented_v2.py; no reset/reopen/manual Play/old controller.')


_review_reverse_closedloop_handoff()

