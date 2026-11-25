"""Read-only actual forward-PASS continuity, even when second reverse is HOLD."""


def _review_h3b_post_forward():
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    workspace=Path('/home/jk/Research/IJCAS_ws');scripts=workspace/'studies/IJCAS/scripts'
    load=runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h=load(scripts/'hospital_h3b_post_forward.py')
    support=runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    if 'IJCAS_H3B_POST_FORWARD_REVIEW_DIR' not in globals():
        raise ValueError('Set IJCAS_H3B_POST_FORWARD_REVIEW_DIR to the explicitly reviewed offline output directory')
    directory=Path(globals()['IJCAS_H3B_POST_FORWARD_REVIEW_DIR'])
    report=h['load_review'](directory);info,audit,manifest,geometry,inventory=h['inspect_forward']()
    timeline=omni.timeline.get_timeline_interface();stage=omni.usd.get_context().get_stage()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Keep the actual NEXT-forward PASS PAUSED; no reopen/Stop/Play/reset')
    for name,task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task,'done') and not task.done():
            raise RuntimeError('Another hospital task is still running')
    root_before=stage.GetRootLayer().ExportToString();session_before=stage.GetSessionLayer().ExportToString()
    marker=stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status')!='h3b_next_forward_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_next_forward_pilot_gate')!=info['forward_result']):
        raise RuntimeError('Need the SAME actual next-forward PASS scene, not any earlier prefix/reverse endpoint')
    prep=info['forward_record']['physics_preparation_snapshot']
    support['live_geometry_check'](stage,manifest,geometry,inventory,set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage,manifest)!=prep['live_support_readback']:
        raise RuntimeError('Support/contact/motor baseline differs; no changes attempted')
    sim=SimulationManager.get_physics_sim_view()
    if sim is None:raise RuntimeError('Existing tensor view unavailable; never initialize/reset to create one')
    view=sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count!=1:raise RuntimeError('Exactly one retained live articulation required')

    def numeric(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        elif hasattr(value,'numpy'):value=value.numpy()
        data=np.asarray(value,dtype=float).reshape(-1)
        if not np.isfinite(data).all():raise RuntimeError('Nonfinite live tensor')
        return data

    names=list(view.shared_metatype.dof_names);q=numeric(view.get_dof_positions())
    transform=numeric(view.get_root_transforms());velocity=numeric(view.get_root_velocities())
    if len(names)!=12 or len(q)!=12 or len(transform)!=7 or len(velocity)!=6:
        raise RuntimeError('Unexpected retained tensor shape')
    x,y,z,w=transform[3:];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    pose=[float(transform[0]),float(transform[1]),yaw];joints=dict(zip(names,map(float,q)))
    checks=h['forward'].validate_live(info,pose,joints,velocity)
    if (timeline.is_playing() or timeline.is_stopped() or stage.GetRootLayer().ExportToString()!=root_before
            or stage.GetSessionLayer().ExportToString()!=session_before):
        raise RuntimeError('Timeline/root/session changed during read-only review')
    h['verify_files'](report['input_sha256'])
    path=directory/'live_post_forward.json'
    if path.exists():raise FileExistsError('Preserve the previous live review; do not overwrite')
    script=workspace/'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_forward.py'
    receipt=dict(status='ACTUAL_NEXT_FORWARD_PASS_PAUSED_CONTINUITY_RECORDED_NOT_DRIVE_AUTHORITY',
        forward_result=info['forward_result'],forward_result_sha256=info['forward_result_sha256'],
        offline_review=str(directory/'post_forward_review.json'),offline_review_sha256=h['sha'](directory/'post_forward_review.json'),
        live_world_xyyaw=pose,live_root_velocity=velocity.tolist(),retained_all_joint_positions_rad=joints,
        raw_forward_audit=audit,**checks,script_sha256=h['sha'](script),
        input_sha256={**report['input_sha256'],str(directory/'post_forward_review.json'):h['sha'](directory/'post_forward_review.json')},
        root_session_and_timeline_unchanged=True,physical_state_reset=False,new_PhysX_motion_executed=False,
        all_second_reverse_model_cases_pass=report['all_second_reverse_model_cases_pass'],
        automatic_motion_authorized=False,full_H3B_turning_path_verified=False)
    with path.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print('=== H3-B ACTUAL NEXT FORWARD PASS LIVE REVIEW (NO MOTION) ===')
    print('raw next-forward verdict:',audit['original_verdict_unchanged'])
    print('saved/live mm/deg:',1000*checks['position_difference_m'],checks['yaw_difference_deg'])
    print('all 12 joints retained; max delta rad:',checks['maximum_joint_difference_rad'])
    print('LIVE CONTINUITY: PASS; NOT second-reverse drive admission')
    print('SECOND REVERSE MODEL:',report['status']);print('saved:',path)
    print('Keep PAUSED. No reset/reopen/Play/save or old controller. Reference-margin review required before any second reverse.')


_review_h3b_post_forward()
