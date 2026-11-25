"""Actual SECOND-reverse PASS continuity and final MODEL review; NO motion."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _review_h3b_post_second_reverse():
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    workspace = _ijcas_workspace();scripts = _ijcas_path(workspace, 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h = load(scripts/'hospital_h3b_final_forward_review_gate.py')
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    if 'IJCAS_H3B_FINAL_FORWARD_REVIEW_DIR' not in globals():
        raise ValueError('Set IJCAS_H3B_FINAL_FORWARD_REVIEW_DIR to the explicitly reviewed offline directory')
    directory = Path(globals()['IJCAS_H3B_FINAL_FORWARD_REVIEW_DIR'])
    info, audit, manifest, geometry, inventory = h['inspect_second_reverse']()
    report = h['load_review'](directory, info)
    timeline = omni.timeline.get_timeline_interface();stage = omni.usd.get_context().get_stage()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Keep the actual SECOND-reverse PASS PAUSED; no reset/reopen/Stop/Play')
    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task, 'done') and not task.done():
            raise RuntimeError('Another hospital task is still running')
    before_root = stage.GetRootLayer().ExportToString();before_session = stage.GetSessionLayer().ExportToString()
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_timed_second_reverse_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_timed_second_reverse_pilot_gate') != info['second_reverse_result']):
        raise RuntimeError('Need the SAME actual SECOND-reverse PASS scene, not any older first-reverse/forward endpoint')
    prep = info['second_reverse_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Original support/contact/motor baseline changed; no correction attempted')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:raise RuntimeError('Existing tensor view unavailable; do not initialize/reset to create one')
    view = sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count != 1:raise RuntimeError('Exactly one retained articulation required')

    def numeric(value):
        if hasattr(value, 'detach'):value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):value = value.numpy()
        array = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(array).all():raise RuntimeError('Nonfinite live tensor')
        return array

    names = list(view.shared_metatype.dof_names);q = numeric(view.get_dof_positions())
    transform = numeric(view.get_root_transforms());velocity = numeric(view.get_root_velocities())
    if len(names) != 12 or len(q) != 12 or len(transform) != 7 or len(velocity) != 6:
        raise RuntimeError('Unexpected retained body/joint tensor shape')
    x, y, z, w = transform[3:];yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    pose = [float(transform[0]), float(transform[1]), yaw];joints = dict(zip(names, map(float, q)))
    checks = h['validate_live'](info, pose, joints, velocity)
    if (timeline.is_playing() or timeline.is_stopped() or stage.GetRootLayer().ExportToString() != before_root
            or stage.GetSessionLayer().ExportToString() != before_session):
        raise RuntimeError('Timeline/root/session changed during read-only review')
    h['verify_files'](report['input_sha256'])
    script = _ijcas_path(workspace, 'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_second_reverse.py')
    gate = scripts/'hospital_h3b_final_forward_review_gate.py'
    receipt = dict(status='ACTUAL_SECOND_REVERSE_PASS_CONTINUITY_FINAL_FORWARD_ADAPTER_PENDING_NO_MOTION',
        second_reverse_result=info['second_reverse_result'],second_reverse_result_sha256=info['second_reverse_result_sha256'],
        offline_review=str(directory/'final_forward_review.json'),offline_review_sha256=h['sha'](directory/'final_forward_review.json'),
        live_world_xyyaw=pose,live_root_velocity=velocity.tolist(),retained_all_joint_positions_rad=joints,
        raw_second_reverse_audit=audit,**checks,script_sha256=h['sha'](script),
        all_selected_final_forward_design_cases_pass=report['all_selected_final_forward_design_cases_pass'],
        selected_planned_speed_cap_m_s=report['planned_speed_cap_m_s'],
        original_mission_goal_xyyaw=info['mission_goal_xyyaw'],
        input_sha256={**report['input_sha256'],str(directory/'final_forward_review.json'):h['sha'](directory/'final_forward_review.json'),
            str(script):h['sha'](script),str(gate):h['sha'](gate)},
        root_session_and_timeline_unchanged=True,physical_state_reset=False,new_PhysX_motion_executed=False,
        automatic_motion_authorized=False,final_forward_executor_admitted=False,
        full_H3B_turning_path_verified=False,CICS_input_used=False)
    path = directory/'live_post_second_reverse.json'
    encoded = json.dumps(receipt, indent=2, allow_nan=False)
    with path.open('x') as stream:stream.write(encoded)
    print('=== H3-B ACTUAL SECOND REVERSE PASS LIVE REVIEW (NO MOTION) ===')
    print('raw second-reverse verdict:', audit['original_verdict_unchanged'])
    print('saved/live mm/deg:', 1000*checks['position_difference_m'], checks['yaw_difference_deg'])
    print('all 12 joints retained; max delta rad:', checks['maximum_joint_difference_rad'])
    print('LIVE CONTINUITY: PASS; NOT final-forward drive admission')
    print('FINAL FORWARD MODEL:', report['status'])
    print('selected planned speed cap mm/s:', 1000*report['planned_speed_cap_m_s']);print('saved:', path)
    print('Keep PAUSED. No reset/reopen/Play/save or old controller. Dedicated FINAL-forward adapter admission is still pending.')


_review_h3b_post_second_reverse()
