"""Read-only handoff at cohort_03's actual NEXT-forward PASS, no actions."""


def _review_second_response_handoff():
    import json
    import math
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    workspace = Path('/home/jk/Research/IJCAS_ws'); scripts = workspace/'studies/IJCAS/scripts'
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h = load(scripts/'hospital_h3b_second_response_observation.py')
    info = h['load_profile'](); protocol = h['load_protocol'](); h['validate_runtime_api']()
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    timeline = omni.timeline.get_timeline_interface(); stage = omni.usd.get_context().get_stage()
    if timeline.is_playing() or timeline.is_stopped() or stage is None:
        raise RuntimeError('Require retained actual cohort_03 NEXT-forward PASS PAUSED; no reset/reopen/Play')
    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task, 'done') and not task.done():
            raise RuntimeError('A hospital task is still running')
    before = (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_next_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_next_response_pilot_gate') != info['forward_result']):
        raise RuntimeError('Not the actual completed cohort_03 reverse-PASS scene; no substitution')
    first = info['preparation_record']; prep = first['physics_preparation_snapshot']
    _, manifest, geometry, inventory = support['load_setup'](Path(first['geometry_manifest']).parent, workspace)
    support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Original support/contact/motor baseline changed')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None: raise RuntimeError('Existing tensor view unavailable; no initialization/reset permitted')
    view = sim.create_articulation_view(support['ROBOT']+'/base_link')
    if view.count != 1: raise RuntimeError('Exactly one retained articulation required')
    def numeric(value):
        if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'): value = value.numpy()
        result = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(result).all(): raise RuntimeError('Nonfinite retained tensor')
        return result
    names = list(view.shared_metatype.dof_names); q = numeric(view.get_dof_positions())
    transform = numeric(view.get_root_transforms()); velocity = numeric(view.get_root_velocities())
    settings = {}
    for key, getter in (('kp', view.get_dof_stiffnesses), ('kd', view.get_dof_dampings),
            ('force_limit', view.get_dof_max_forces), ('velocity_target_rad_s', view.get_dof_velocity_targets)):
        settings[key] = dict(zip(names, map(float, numeric(getter()))))
    checks = h['validate_tensor'](info, transform, names, q, velocity, settings)
    x, y, z, w = transform[3:]; yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    pose = [float(transform[0]), float(transform[1]), yaw]; joints = dict(zip(names, map(float, q)))
    if (timeline.is_playing() or timeline.is_stopped() or before !=
            (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())):
        raise RuntimeError('Read-only review changed scene/timeline')
    pins = {**info['input_sha256'], **protocol['input_sha256'], str(h['PROTOCOL_PATH']): h['sha'](h['PROTOCOL_PATH'])}
    h['verify_files'](pins)
    target = h['LIVE']; target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='PAUSED_ACTUAL_FORWARD_PASS_SECOND_RESPONSE_HANDOFF_NO_MOTION',
        forward_result=info['forward_result'], forward_result_sha256=info['forward_result_sha256'],
        protocol=str(h['PROTOCOL_PATH']), protocol_sha256=h['sha'](h['PROTOCOL_PATH']),
        live_world_xyyaw=pose, live_root_velocity=velocity.tolist(), retained_all_joint_positions_rad=joints,
        retained_tensor_checks=checks, physical_state_reset=False, new_PhysX_motion_executed=False,
        source_USD_and_session_unchanged=True, automatic_motion_launched=False, whole_H3_drive_admission=False,
        input_sha256=pins)
    with target.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print('=== H3-B SAME SECOND-REVERSE RESPONSE LIVE HANDOFF REVIEW (NO MOTION) ===')
    print('saved/live mm/deg:', checks['position_difference_m']*1000, checks['yaw_difference_deg'])
    print('all twelve joints retained; max delta rad:', checks['maximum_joint_difference_rad'])
    print('reference length mm:', info['reference_length_m']*1000)
    print('HANDOFF REVIEW: PASS; same component ONLY, NOT whole-H3/model validation')
    print('saved:', target)
    print('Next ONLY: isaac_run_ijcas_hospital_h3b_second_response_instrumented.py; no reset/reopen/manual Play/save.')


_review_second_response_handoff()

