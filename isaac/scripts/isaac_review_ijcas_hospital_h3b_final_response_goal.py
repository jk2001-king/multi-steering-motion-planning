"""Read-only closeout at the actual cohort_03 FINAL-response PASS endpoint."""


def _review_final_response_goal():
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    workspace = Path('/home/jk/Research/IJCAS_ws')
    scripts = workspace/'studies/IJCAS/scripts'
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h = load(scripts/'hospital_h3b_response_completion.py')
    if 'IJCAS_H3B_RESPONSE_CLOSEOUT_DIR' not in globals():
        raise ValueError('Set IJCAS_H3B_RESPONSE_CLOSEOUT_DIR to the explicit audited cohort_03 directory')
    info, audit, manifest, geometry, inventory, report = h['load_closeout'](globals()['IJCAS_H3B_RESPONSE_CLOSEOUT_DIR'])
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Require the SAME actual FINAL-response PASS PAUSED; no reset/reopen/Play/Stop')
    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task, 'done') and not task.done():
            raise RuntimeError('A hospital motion or observer task is still running')
    before_layers = (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_final_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') != info['final_forward_result']):
        raise RuntimeError('Require the actual cohort_03 FINAL-response endpoint, not the old final/failed/reset scene')
    prep = info['final_forward_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Original support/contact/geometry changed; no correction attempted')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Existing physics tensor view unavailable; no initialization/reset permitted')
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
            raise RuntimeError('Nonfinite retained tensor')
        return result

    def snapshot():
        names = list(view.shared_metatype.dof_names)
        settings = {}
        for key, getter in (('kp', view.get_dof_stiffnesses), ('kd', view.get_dof_dampings),
                ('force_limit', view.get_dof_max_forces), ('velocity_target_rad_s', view.get_dof_velocity_targets)):
            settings[key] = dict(zip(names, map(float, numeric(getter()))))
        return dict(names=names, transform=numeric(view.get_root_transforms()).tolist(),
            positions=numeric(view.get_dof_positions()).tolist(), velocity=numeric(view.get_root_velocities()).tolist(),
            settings=settings)

    initial = snapshot()
    checks = h['validate_tensor'](info, initial['transform'], initial['names'], initial['positions'],
        initial['velocity'], initial['settings'])
    x, y, z, w = initial['transform'][3:]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    pose = initial['transform'][:2]+[yaw]
    joints = dict(zip(initial['names'], initial['positions']))
    terminal = h['final'].arrival_checks(pose, info['mission_goal_xyyaw'], math.hypot(*initial['velocity'][:2]),
        (joints['base_link_to_front_active_steering'], joints['base_link_to_rear_active_steering']))
    if not all(terminal.values()):
        raise RuntimeError('Unchanged original goal/stop/neutral acceptance not retained; no recovery')
    if (initial != snapshot() or timeline.is_playing() or timeline.is_stopped() or before_layers !=
            (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())):
        raise RuntimeError('Body/joints/settings/root/session/timeline changed during read-only review')
    inputs = {**report['input_sha256'], str(h['TARGET']): h['sha'](h['TARGET']),
        str(h['SUMMARY']): h['sha'](h['SUMMARY']), str(h['SCRIPT']): h['sha'](h['SCRIPT'])}
    h['verify_files'](inputs)
    receipt = dict(version=h['VERSION'], status='ACTUAL_COHORT03_FINAL_GOAL_FIVE_RESPONSE_OBSERVATIONS_CLOSED_NO_MOTION',
        final_forward_result=info['final_forward_result'], final_forward_result_sha256=info['final_forward_result_sha256'],
        offline_completion_audit=str(h['TARGET']), offline_completion_audit_sha256=h['sha'](h['TARGET']),
        live_world_xyyaw=pose, live_root_velocity=initial['velocity'], retained_all_joint_positions_rad=joints,
        retained_tensor_checks=checks, original_final_goal_checks=terminal, raw_final_forward_audit=audit,
        saved_body_and_twelve_joint_handoffs_verified=report['chain_audit']['saved_body_and_12_joint_handoffs_verified'],
        original_mission_goal_xyyaw=info['mission_goal_xyyaw'], root_session_timeline_body_joints_settings_unchanged=True,
        original_whole_model_PASS_count=2, original_whole_model_case_count=8,
        all_six_original_whole_model_HOLDs_preserved=True, independent_fresh_frozen_whole_sequence_repeats=0,
        physical_state_reset=False, new_PhysX_motion_executed=False, automatic_motion_authorized=False,
        whole_H3_drive_admission=False, CICS_input_used=False, use_for_final_statistics=False, input_sha256=inputs)
    with h['LIVE'].open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print('=== H3-B ACTUAL FINAL RESPONSE GOAL / FIVE-COMPONENT LIVE CLOSEOUT (NO MOTION) ===')
    print('raw FINAL response:', audit['original_verdict_unchanged'])
    print('saved/live mm/deg:', 1000*checks['position_difference_m'], checks['yaw_difference_deg'])
    print('all twelve joints retained; max delta rad:', checks['maximum_joint_difference_rad'])
    print('ORIGINAL FINAL GOAL RETAINED:', all(terminal.values()))
    print('four saved physical handoffs / all five six-wheel response data sets: VERIFIED')
    print('Whole MODEL stays 2/8; independent fresh frozen whole-sequence repeats: 0')
    print('saved:', h['LIVE'])
    print('Next is offline body-response model review, NOT another leg or an admitted whole-route drive.')
    print('No reset/reopen/Play/save/CICS, control change or automatic retry.')


_review_final_response_goal()
