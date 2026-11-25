"""Read-only dedicated handoff for ONE screened timed SECOND reverse; no motion."""


def _review_h3b_timed_second_reverse():
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
    h=load(scripts/'hospital_h3b_timed_second_reverse.py')
    support=runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    info=h['load_prepared_profile']();audit=info['raw_forward_audit']
    _,manifest,geometry,inventory=support['load_setup'](Path(info['preparation_record']['geometry_manifest']).parent,workspace)
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
    checks=h['validate_live'](info,pose,joints,velocity)
    if (timeline.is_playing() or timeline.is_stopped() or stage.GetRootLayer().ExportToString()!=root_before
            or stage.GetSessionLayer().ExportToString()!=session_before):
        raise RuntimeError('Timeline/root/session changed during read-only review')
    h['verify_files'](info['input_sha256'])
    path=h['LIVE'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve the previous live review; do not overwrite')
    script=workspace/'isaac/scripts/isaac_review_ijcas_hospital_h3b_timed_second_reverse.py'
    receipt=dict(status='TIMED_SECOND_REVERSE_ACTUAL_FORWARD_PASS_CONTINUITY_RECORDED_NO_MOTION',
        forward_result=info['forward_result'],forward_result_sha256=info['forward_result_sha256'],
        executor_admission=info['admission'],executor_admission_sha256=info['admission_sha256'],
        model_review=info['model_review'],model_review_sha256=info['model_review_sha256'],
        live_world_xyyaw=pose,live_root_velocity=velocity.tolist(),retained_all_joint_positions_rad=joints,
        raw_forward_audit=audit,**checks,script_sha256=h['sha'](script),
        input_sha256={**info['input_sha256'],str(script):h['sha'](script)},
        root_session_and_timeline_unchanged=True,physical_state_reset=False,new_PhysX_motion_executed=False,
        all_timed_executor_design_cases_pass=True,CICS_input_used=False,
        automatic_motion_authorized=False,full_H3B_turning_path_verified=False)
    with path.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print('=== H3-B TIMED SECOND REVERSE LIVE HANDOFF REVIEW (NO MOTION) ===')
    print('raw next-forward verdict:',audit['original_verdict_unchanged'])
    print('saved/live mm/deg:',1000*checks['position_difference_m'],checks['yaw_difference_deg'])
    print('all 12 joints retained; max delta rad:',checks['maximum_joint_difference_rad'])
    print('LIVE CONTINUITY: PASS; frozen timed SECOND reverse only; still no motion')
    print('all EIGHT timed model design cases: PASS; reference length mm:',round(1000*info['reference_length_m'],3));print('saved:',path)
    print('Keep PAUSED. Next ONLY: isaac_run_ijcas_hospital_h3b_timed_second_reverse_instrumented.py. No reset/reopen/manual Play/save or old controller.')


_review_h3b_timed_second_reverse()
