"""Archive ACTUAL H3 FINAL before authorized Stop; stage a fresh H4 reference.

No Play/static run, driving, old-floor change, caster assignment or source save.
After Stop the old physical continuation is over; an archive is evidence, not
permission or a promise to reconstruct that physical/contact state.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _prepare_h4_fresh_cohort():
    import json
    import os
    from pathlib import Path
    import runpy

    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    scripts = _ijcas_path(_ijcas_workspace(), 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_cohort_preparation_v4.py')
    h, d = r['completion'], r['support']
    preflight = r['load_preflight']()
    if any(path.exists() for path in (r['ARCHIVE'], r['LIVE'], r['FAILURE'])):
        raise RuntimeError('This cohort already has an archive/result/failure; no overwrite, retry or recovery')
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or timeline.is_playing() or timeline.is_stopped():
        raise RuntimeError('Exact retained H3 FINAL must still be PAUSED before archival')
    if Path(stage.GetRootLayer().realPath).resolve() != r['SOURCE'].resolve():
        raise RuntimeError('Original hospital source required; do not reopen/substitute a stage')
    for task in globals().values():
        if hasattr(task, 'done') and hasattr(task, 'cancel') and not task.done():
            raise RuntimeError('An asynchronous task is still running; do not retire the articulation')
    for path in (r['ROBOT'], r['MARKER']):
        if stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError('A new H4 cohort prim already exists; no overwrite or automatic retry')
    info, _, manifest, geometry, inventory, closed = h.load_closeout(h.DIRECTORY)
    old_support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    marker = stage.GetPrimAtPath(old_support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_final_response_pilot_finished_wait_for_review'
            or marker.GetCustomDataByKey('last_h3b_final_response_pilot_gate') != info['final_forward_result']):
        raise RuntimeError('Exact actual cohort_03 final endpoint marker required')
    prep = info['final_forward_record']['physics_preparation_snapshot']
    old_support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
    if old_support['support_readback'](stage, manifest) != prep['live_support_readback']:
        raise RuntimeError('Original H3 material/ground/robot/scene baseline changed')
    if any(p.GetTypeName() == 'OmniGraph' for p in stage.Traverse()):
        raise RuntimeError('An active OmniGraph could command the new robot; preparation refused')
    sim = SimulationManager.get_physics_sim_view()
    if sim is None:
        raise RuntimeError('Retained actual tensor view unavailable; never Play to obtain it')
    view = sim.create_articulation_view(r['OLD_ROBOT']+'/base_link')
    if view.count != 1:
        raise RuntimeError('Exactly one retained H3 articulation required')

    def array(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        elif hasattr(value, 'numpy'):
            value = value.numpy()
        result = np.asarray(value, dtype=float)
        if not np.isfinite(result).all():
            raise RuntimeError('Finite actual-state archive values required')
        return result

    names = list(view.shared_metatype.dof_names)
    links = list(view.shared_metatype.link_names)
    if len(names) != 12 or len(links) != 13 or view.max_links != 13:
        raise RuntimeError('Actual full thirteen-body/twelve-joint archive unavailable')

    def snapshot():
        settings = {}
        for key, getter in (('kp', view.get_dof_stiffnesses), ('kd', view.get_dof_dampings),
                ('force_limit', view.get_dof_max_forces), ('velocity_target_rad_s', view.get_dof_velocity_targets)):
            settings[key] = dict(zip(names, array(getter()).reshape(-1).tolist()))
        return dict(names=names, link_names=links, transform=array(view.get_root_transforms()).reshape(-1).tolist(),
            positions=array(view.get_dof_positions()).reshape(-1).tolist(),
            velocity=array(view.get_root_velocities()).reshape(-1).tolist(), settings=settings,
            joint_velocities_rad_s=array(view.get_dof_velocities()).reshape(-1).tolist(),
            joint_position_targets_rad=array(view.get_dof_position_targets()).reshape(-1).tolist(),
            all_thirteen_link_world_transforms_xyzw=array(view.get_link_transforms()).reshape(13, 7).tolist(),
            all_thirteen_link_velocities_at_CoM=array(view.get_link_velocities()).reshape(13, 6).tolist())

    actual = snapshot()
    state_checks = h.validate_tensor(info, actual['transform'], names, actual['positions'], actual['velocity'], actual['settings'])
    session = stage.GetSessionLayer()
    prior_session, prior_root = session.ExportToString(), stage.GetRootLayer().ExportToString()
    design = d.load_design()

    def verify_support():
        return r['verify_actual_support'](stage)

    verify_support()

    def save_exclusive(path, report):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())

    # A real write/read-back archive is mandatory BEFORE the first timeline mutation.
    r['ARCHIVE'].mkdir(parents=True, exist_ok=False)
    archive = dict(version=r['VERSION'], status='ACTUAL_H3_FINAL_ARCHIVED_BEFORE_AUTHORIZED_STOP',
        timeline_state='PAUSED', timeline_current_time_s=float(timeline.get_current_time()),
        actual_state=actual, actual_final_body_twelve_joint_checks=state_checks,
        final_response_result=info['final_forward_result'], final_response_result_sha256=h.sha(info['final_forward_result']),
        original_goal_world_xyyaw=info['mission_goal_xyyaw'],
        all_five_raw_component_verdicts=closed['chain_audit']['all_five_raw_component_verdicts'],
        prior_whole_H3_MODEL_PASS_count=2, prior_whole_H3_MODEL_case_count=8,
        independent_fresh_frozen_whole_H3_repeats=0,
        source_stage=str(r['SOURCE']), source_sha256=r['SOURCE_SHA'],
        session_layer_text_before_Stop=prior_session, root_layer_text_before_Stop=prior_root,
        physical_contact_solver_state_is_NOT_serialized_or_reconstructible_from_this_archive=True,
        no_automatic_restore_replay_or_H3_success_upgrade=True,
        H4_support_receipt=str(d.LIVE), H4_support_receipt_sha256=h.sha(d.LIVE),
        input_sha256={**preflight['input_sha256'], str(r['PREFLIGHT']): h.sha(r['PREFLIGHT'])})
    archive_path = r['ARCHIVE']/'actual_H3_final_state.json'
    save_exclusive(archive_path, archive)
    if (json.loads(archive_path.read_bytes()) != archive or snapshot() != actual
            or timeline.is_playing() or timeline.is_stopped() or session.ExportToString() != prior_session):
        raise RuntimeError('Durable archive/live continuity verification failed BEFORE Stop; no staging allowed')
    print('H3 FINAL actual body/12 joints/13 link states + Session snapshot archived BEFORE authorized Stop.', flush=True)
    stopped = False
    try:
        with Usd.EditContext(stage, session):
            # Explicitly authorized fresh-cohort boundary, not recovery or a hidden retry.
            timeline.stop()
            stopped = True
            if not timeline.is_stopped():
                raise RuntimeError('Authorized Stop did not complete; no new robot may be staged')
            stage.GetPrimAtPath(r['OLD_ROBOT']).SetActive(False)
            readback = r['author_new_robot'](stage, preflight['start_xyyaw'])
            cohort = UsdGeom.Xform.Define(stage, r['MARKER']).GetPrim()
            cohort.SetCustomDataByKey('status', 'fresh_H4_prepared_static_pending')
            cohort.SetCustomDataByKey('robot_root', r['ROBOT'])
            cohort.SetCustomDataByKey('archived_H3_actual_state', str(archive_path))
            cohort.SetCustomDataByKey('cohort_preflight', str(r['PREFLIGHT']))
            cohort.SetCustomDataByKey('cohort_preflight_sha256', h.sha(r['PREFLIGHT']))
            cohort.SetCustomDataByKey('physics_preparation_result', str(r['LIVE']))
        verify_support()
        old_support['live_geometry_check'](stage, manifest, geometry, inventory, set(prep['disabled_legacy_floor_colliders']))
        if (not timeline.is_stopped() or stage.GetPrimAtPath(r['OLD_ROBOT']).IsActive()
                or stage.GetRootLayer().ExportToString() != prior_root
                or h.sha(r['SOURCE']) != r['SOURCE_SHA']
                or r['protected_session_text'](prior_session) != r['protected_session_text'](session.ExportToString())):
            raise RuntimeError('A protected hospital/support/source/Session/timeline setting changed')
        h.verify_files(preflight['input_sha256'])
        report = dict(version=r['VERSION'], status='H4_FRESH_SOURCE_ROBOT_STAGED_STOPPED_STATIC_PENDING',
            new_robot_root=r['ROBOT'], retired_H3_robot_root=r['OLD_ROBOT'], retired_H3_inactive=True,
            archived_actual_H3_state=str(archive_path), archived_actual_H3_state_sha256=h.sha(archive_path),
            archive_durably_verified_before_Stop=True,
            original_H3_live_physical_continuation_ended_NOT_restored=True,
            live_new_robot_readback=readback, actual_H4_static_support_validated=False,
            task_candidate=str(r['SETUP']), start_xyyaw=preflight['start_xyyaw'], goal_xyyaw=preflight['goal_xyyaw'],
            support_colliders=preflight['supports'], support_top_m=preflight['support_top_m'],
            old_ground_local_patch_hospital_obstacles_scene_and_materials_unchanged=True,
            Session_delta_limited_to_authorized_old_new_robot_and_marker=True,
            fresh_source_seed_recorded_NOT_assigned_or_angle_selected=True,
            old_H3_PASS_FAIL_HOLD_evidence_preserved=True, original_source_USD_saved=False,
            timeline='STOPPED', Play_or_static_or_drive_executed=False, actual_drive_admission=False,
            CICS_runtime_integration=False, use_for_final_statistics=False,
            input_sha256={**archive['input_sha256'], str(archive_path): h.sha(archive_path)})
        save_exclusive(r['LIVE'], report)
    except Exception as error:
        # No automatic reset/restore/retry: Stop invalidates the old solver state.
        # The durable H3 archive remains, and the new actor is inhibited on failure.
        if stopped:
            with Usd.EditContext(stage, session):
                new = stage.GetPrimAtPath(r['ROBOT'])
                if new.IsValid():
                    new.SetActive(False)
        save_exclusive(r['FAILURE'], dict(status='HOLD_AFTER_AUTHORIZED_COHORT_BOUNDARY_NO_PLAY_OR_RECOVERY',
            error=type(error).__name__+': '+str(error), authorized_Stop_performed=stopped,
            actual_H3_archive=str(archive_path), automatic_retry_or_restore=False))
        raise
    omni.usd.get_context().get_selection().set_selected_prim_paths([r['ROBOT']], True)
    print('=== H4 FRESH COHORT STATIC PREPARATION (NO PLAY / NO DRIVING) ===')
    print('H3 final actual state archived:', archive_path)
    print('new H4 robot / fixed start:', r['ROBOT'], preflight['start_xyyaw'])
    print('mass / bodies / joints:', readback['simulation_mass_kg'], len(readback['rigid_bodies']), len(readback['revolute_joint_names']))
    print('source-authored caster/joint seed retained; no angle selection or assignment')
    print('original floor + local H4 support, materials, scene and hospital obstacles unchanged')
    print('saved:', r['LIVE'])
    print('Timeline STOPPED. H4 STATIC CONTACT GATE PENDING; no manual Play/save or old controller.')


_prepare_h4_fresh_cohort()
