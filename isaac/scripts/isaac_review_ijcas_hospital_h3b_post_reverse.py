"""Read-only current reverse-PASS state review. No Play/reset/scene edit."""


def _review_h3b_post_reverse():
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
    h = load(scripts/'review_hospital_h3b_post_reverse.py')
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    if 'IJCAS_H3B_POST_REVERSE_REVIEW_DIR' not in globals():
        raise ValueError('Set IJCAS_H3B_POST_REVERSE_REVIEW_DIR to the explicitly reviewed offline output directory')
    review_dir = Path(globals()['IJCAS_H3B_POST_REVERSE_REVIEW_DIR'])
    report = h['load_review'](review_dir)
    info, audit, manifest, geometry, inventory = h['inspect_reverse']()
    timeline = omni.timeline.get_timeline_interface(); stage = omni.usd.get_context().get_stage()
    if timeline.is_playing() or timeline.is_stopped() or stage is None:
        raise RuntimeError('Keep the SAME measured-terminal reverse PASS PAUSED; do not Stop/reopen/Play/reset')
    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task,'done') and not task.done():
            raise RuntimeError('A hospital task is still running')
    root_before = stage.GetRootLayer().ExportToString(); session_before = stage.GetSessionLayer().ExportToString()
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_reverse_closedloop_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_reverse_closedloop_pilot_gate') != info['reverse_result']):
        raise RuntimeError('Wrong live scene: actual NEW measured-terminal reverse-PASS marker required')
    prep = info['reverse_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage,manifest,geometry,inventory,set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage,manifest) != prep['live_support_readback']:
        raise RuntimeError('Support/contact/motor baseline changed')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None: raise RuntimeError('Existing physics view unavailable; do not initialize/reset it')
    view = sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count != 1: raise RuntimeError('Exactly one existing retained articulation required')

    def numeric(value):
        if hasattr(value,'detach'): value = value.detach().cpu().numpy()
        elif hasattr(value,'numpy'): value = value.numpy()
        data = np.asarray(value,dtype=float).reshape(-1)
        if not np.isfinite(data).all(): raise RuntimeError('Nonfinite live tensor')
        return data

    names = list(view.shared_metatype.dof_names)
    q = numeric(view.get_dof_positions()); transform = numeric(view.get_root_transforms())
    velocity = numeric(view.get_root_velocities())
    if len(names)!=12 or len(q)!=12 or len(transform)!=7 or len(velocity)!=6:
        raise RuntimeError('Unexpected live tensor shape')
    x,y,z,w = transform[3:]; yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    pose = [float(transform[0]),float(transform[1]),yaw]; joints = dict(zip(names,map(float,q)))
    checks = h['reverse'].validate_live(info,pose,joints,velocity)
    if (timeline.is_playing() or timeline.is_stopped() or stage.GetRootLayer().ExportToString()!=root_before
            or stage.GetSessionLayer().ExportToString()!=session_before):
        raise RuntimeError('Timeline/root/session changed during read-only review')
    h['reverse'].verify_files(report['input_sha256'])
    script = workspace/'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_reverse.py'
    output = review_dir/'live_post_reverse.json'
    if output.exists(): raise FileExistsError('Preserve existing live receipt; no overwrite')
    receipt = dict(status='ACTUAL_REVERSE_PASS_PAUSED_CONTINUITY_RECORDED_NOT_DRIVE_AUTHORITY',
        reverse_result=info['reverse_result'],reverse_result_sha256=info['reverse_result_sha256'],
        offline_review=str(review_dir/'post_reverse_review.json'),
        offline_review_sha256=h['reverse'].sha(review_dir/'post_reverse_review.json'),
        live_world_xyyaw=pose,live_root_velocity=velocity.tolist(),retained_all_joint_positions_rad=joints,
        raw_reverse_audit=audit,**checks,script_sha256=h['reverse'].sha(script),
        input_sha256={**report['input_sha256'],str(review_dir/'post_reverse_review.json'):h['reverse'].sha(review_dir/'post_reverse_review.json')},
        all_next_forward_model_cases_pass=report['all_next_forward_model_cases_pass'],
        root_session_and_timeline_unchanged=True,physical_state_reset=False,
        new_PhysX_motion_executed=False,automatic_motion_authorized=False,full_H3B_turning_path_verified=False)
    with output.open('x') as stream: json.dump(receipt,stream,indent=2,allow_nan=False)
    print('=== H3-B ACTUAL FIRST REVERSE PASS LIVE REVIEW (NO MOTION) ===')
    print('raw measured-terminal reverse verdict:',audit['original_verdict_unchanged'])
    print('saved/live mm/deg:',1000*checks['position_difference_m'],checks['yaw_difference_deg'])
    print('all 12 joints retained; max delta rad:',checks['maximum_joint_difference_rad'])
    print('next forward MODEL screen:',report['status'])
    print('saved:',output)
    print('Keep PAUSED. No old controller, reset/reopen/manual Play/save. Dedicated NEXT-forward executor admission still pending.')


_review_h3b_post_reverse()
