"""ONE visual-only resync/restore transaction; never Play/Stop/reopen/reset.

An explicit fresh-static protocol selects that actor's own PAUSED boundary.
Absent that configuration the original compact STOPPED operation is retained.
"""

if globals().get('ijcas_hospital_h4_visual_instance_refresh_authorized') is not True:
    raise RuntimeError('Explicit VISUAL INSTANCE REFRESH ONLY flag required; no action taken')


async def _refresh_h4_visual_instances_once():
    from pathlib import Path
    import runpy
    import time
    import carb.settings
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Usd

    scripts=Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load=runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    refresh=load(scripts/'hospital_h4_visual_instance_refresh.py')
    instances=refresh['instances'];render=instances.render;display=render.display;prep=display.preparation
    folder=prep.GATE/'hospital_h4_robot_visibility/visual_refresh_once_r01'
    reserve=folder/'single_visual_refresh_reserved.json'
    output=folder/'visual_refresh_readback.json';failure=folder/'visual_refresh_failure.json'
    if any(p.exists() for p in (reserve,output,failure)):
        raise RuntimeError('This single visual refresh is already reserved/recorded; no retry or overwrite')
    protocol,actual=display.verify_actual_preparation()
    context=omni.usd.get_context();stage=context.get_stage()
    timeline=omni.timeline.get_timeline_interface();viewport=get_active_viewport()
    def stopped():
        return (context.get_stage()==stage and not timeline.is_playing() and timeline.is_stopped()
            and float(timeline.get_current_time())==0.)
    if stage is None or viewport is None or not stopped():
        raise RuntimeError('Exact retained STOPPED stage required; no automatic Stop/reset')
    validate=prep.old.constructor_for(prep.ROBOT).__globals__['validate_new_robot']
    asset=Usd.Stage.Open(str(prep.ASSET));before=validate(stage,asset,protocol['original_start_xyyaw'])
    if before!=actual['live_new_robot_readback']:
        raise RuntimeError('Retained actual source body/joints/gains/material differs; no repair')
    geometry=display.geometry_readback(stage);topology=instances.usd_instances(stage,geometry)
    paths=[r['instance_path'] for r in topology['instance_roots']]
    transaction=refresh['capture'](stage,paths)
    settings=carb.settings.get_settings();settings_before=render.display_setting_readback(settings,viewport)
    body_geometry=dict(geometry,geometry=[])
    body_cache_before=render.fabric_readback(stage,body_geometry)
    if (body_cache_before.get('optional_cache_error')
            or body_cache_before.get('existing_sim_Fabric_stage') is not True
            or len(body_cache_before.get('rows',[]))!=13
            or not all(row.get('existing_Fabric_prim') is True for row in body_cache_before.get('rows',[]))):
        raise RuntimeError('Existing physical-body cache comparison unavailable; no visual edit')
    prep.write_exclusive(reserve,dict(status='ONE_VISUAL_REFRESH_RESERVED_NOT_A_PHYSICAL_ATTEMPT',
        target_paths=paths,twenty_two_visuals_no_physical_APIs_verified=True,
        actual_static_or_drive_admission=False))
    began=False;restored=False;restore_result=None;begun=time.monotonic()
    before_cache=instances.existing_instance_cache(stage,topology)
    try:
        print('[H4 VISUAL REFRESH] Exact 13 visuals / 22 shapes checked; body/joints retained. NO Play.',flush=True)
        began=True
        refresh['begin'](stage,transaction)
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()  # renderer UI only while STOPPED
            if not stopped() or time.monotonic()-begun>15.:
                raise RuntimeError('Stopped-state/UI budget violated; no retry or timeline action')
        refresh['verify_transient'](stage,transaction)
        if validate(stage,asset,protocol['original_start_xyyaw'])!=before:
            raise RuntimeError('Actor changed during visual-only UI refresh; no pose recovery')
        print('[H4 VISUAL REFRESH] Restoring exact original visual instancing and Session contents...',flush=True)
        restore_result=refresh['restore'](stage,transaction);restored=True
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()
            if not stopped() or time.monotonic()-begun>15.:
                raise RuntimeError('Stopped-state/UI budget violated after visual restoration')
        if (stage.GetRootLayer().ExportToString()!=transaction['root_text']
                or stage.GetSessionLayer().ExportToString()!=transaction['session_text']
                or validate(stage,asset,protocol['original_start_xyyaw'])!=before
                or settings_before!=render.display_setting_readback(settings,viewport)
                or body_cache_before!=render.fabric_readback(stage,body_geometry)):
            raise RuntimeError('Exact USD/actor/settings/physical-body cache preservation not proven')
        result=dict(restore_result,
            status='ONE_VISUAL_INSTANCE_REFRESH_AND_EXACT_RESTORATION_COMPLETE_PIXELS_PENDING',
            existing_visual_cache_before=before_cache,
            existing_visual_cache_after=instances.existing_instance_cache(stage,topology),
            actual_source_body_twelve_joints_gains_and_material_unchanged=True,
            existing_physical_body_cache_exactly_unchanged=True,
            no_claim_to_serialize_or_restore_solver_or_contact_state=True,
            renderer_physics_and_view_settings_unchanged=True,
            actual_visible_recovery_confirmed=False,
            any_Play_Stop_reset_reopen_restage_or_physical_parameter_change=False,
            use_for_final_statistics=False,wall_s=time.monotonic()-begun)
        prep.write_exclusive(output,result)
        print('=== H4 ONE VISUAL REFRESH / EXACT ORIGINAL SESSION RESTORED ===',flush=True)
        print('Root / Session / actor / physical-body cache unchanged:',True,flush=True)
        print('saved:',output,flush=True)
        print('Inspect pixels WITHOUT F. No static/drive admission or automatic trial.',flush=True)
    except BaseException as error:
        if began and not restored:
            try:
                restore_result=refresh['restore'](stage,transaction);restored=True
            except Exception as cleanup_error:
                restore_result=dict(cleanup_error=type(cleanup_error).__name__+': '+str(cleanup_error))
        report=dict(status='VISUAL_REFRESH_ERROR_NOT_A_PHYSX_OR_CONTROLLER_RESULT',
            error=type(error).__name__+': '+str(error),original_visual_Session_restore_verified=restored,
            restore_readback=restore_result,any_automatic_retry_or_physical_recovery=False,
            actual_static_or_drive_admission=False)
        if not failure.exists():prep.write_exclusive(failure,report)
        print('[H4 VISUAL REFRESH] FAILED:',report['error'],'; original visual Session restored:',restored,flush=True)
        raise


async def _refresh_h4_fresh_static_visuals(protocol_path,workflow_owner=None):
    """Reuse the visual transaction; do not stage or repeat an actual static."""
    import builtins
    from pathlib import Path
    import runpy
    import time
    import carb.settings
    import omni.kit.app
    import omni.physics.tensors
    import omni.timeline
    import omni.usd
    from omni.kit.viewport.utility import get_active_viewport

    scripts=Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load=runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    launch=load(scripts/'hospital_h4_subfloor_compact_path_launch.py')
    refresh=load(scripts/'hospital_h4_visual_instance_refresh.py')
    fresh=launch['load_fresh_trial_protocol'](protocol_path)
    prep=fresh.preparation
    raw=prep.read_small(fresh.stationary.RESULT)
    prep.verify(raw['input_sha256'])
    if (raw['verdict']!='PASS' or not raw['checks']
            or any(v is not True for v in raw['checks'].values())
            or raw['run_id']!=fresh.static_run_id or raw['cohort']!=fresh.protocol['cohort']
            or raw['trace_sha256']!=prep.sha(fresh.stationary.TRACE)
            or raw['static_protocol_sha256']!=prep.sha(fresh.protocol_path)
            or raw['actual_final_timeline']!=raw['pause_confirmation']['after']
            or raw.get('error') or raw.get('postprocessing_error') or raw.get('pause_error')):
        raise RuntimeError('Own completed fresh static PASS/raw evidence required; no display edit')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            if not (name=='ijcas_h4_latest_experiment_task' and previous is workflow_owner
                    and fresh.protocol.get('restore_new_actor_visual_cache_after_static_only') is True):
                raise RuntimeError('Experiment or route still active; no concurrent display edit')
    folder=fresh.output/'display_refresh'
    reserve=folder/'single_visual_refresh_reserved.json'
    output=folder/'visual_refresh_readback.json';failure=folder/'visual_refresh_failure.json'
    if any(path.exists() for path in (reserve,output,failure)):
        raise RuntimeError('This actor visual refresh is already reserved/recorded; no repeat')
    usd=omni.usd.get_context();stage=usd.get_stage()
    timeline=omni.timeline.get_timeline_interface();viewport=get_active_viewport()
    expected=raw['actual_final_timeline']
    def retained():
        return (usd.get_stage()==stage and launch['evidence'].timeline_state(timeline)==expected
            and expected['playing'] is False and expected['stopped'] is False)
    if stage is None or viewport is None or not retained():
        raise RuntimeError('Exact own completed PAUSED static required; no Stop/reset/Play')
    # Attaching read-only tensor views does not initialize or advance PhysX.
    simulation=omni.physics.tensors.create_simulation_view('numpy')
    tensor=simulation.create_articulation_view(fresh.cohort.ROBOT+'/base_link')
    def actual():
        return launch['normalize_runtime_evidence'](prep.old.tensor_snapshot(tensor))
    before=actual()
    if before!=raw['actual_post_static_tensor_snapshot']:
        raise RuntimeError('Current 13-link/12-joint state differs from own static; no display/pose recovery')
    source=launch['fresh_source_for_live_actor'](fresh,stage)
    readers=refresh['display_modules_for_actor'](fresh.cohort.ROBOT)
    render=readers.render;display=render.display
    geometry=display.geometry_readback(stage)
    topology=readers.usd_instances(stage,geometry)
    paths=[row['instance_path'] for row in topology['instance_roots']]
    projection=display.viewport_readback(viewport,geometry)
    depth=render.camera_depth_readback(stage,viewport,geometry)
    if (geometry.get('visible_render_geometry_count')!=22
            or projection.get('projected_AABB_outside_viewport_xy') is not False
            or depth.get('whole_actor_AABB_clipped_by_near_or_far') is not False):
        raise RuntimeError('Actor hidden/outside camera/clipped; visual metadata refresh is not an appropriate repair: '
            +repr(dict(geometry=geometry,projection=projection,depth=depth)))
    transaction=refresh['capture'](stage,paths,actor=fresh.cohort.ROBOT)
    settings=carb.settings.get_settings()
    settings_before=render.display_setting_readback(settings,viewport)
    body_geometry=dict(geometry,geometry=[])
    body_cache_before=render.fabric_readback(stage,body_geometry)
    if (body_cache_before.get('optional_cache_error')
            or body_cache_before.get('existing_sim_Fabric_stage') is not True
            or len(body_cache_before.get('rows',[]))!=13
            or any(row.get('existing_Fabric_prim') is not True for row in body_cache_before.get('rows',[]))):
        raise RuntimeError('Existing 13-body cache cannot be compared; no visual edit')
    cache_before=readers.existing_instance_cache(stage,topology)
    write=launch['write_runtime_evidence']
    write(reserve,dict(status='ONE_FRESH_ACTOR_VISUAL_REFRESH_RESERVED_NOT_A_PHYSICAL_TRIAL',
        actor=fresh.cohort.ROBOT,static_result=str(fresh.stationary.RESULT),
        static_result_sha256=prep.sha(fresh.stationary.RESULT),target_paths=paths,
        own_completed_static_timeline=expected,source_instance_metadata_only=True,
        actual_static_or_drive_admission=False))
    began=False;restored=False;begun=time.monotonic()
    print('[H4 FRESH DISPLAY] Own PAUSED static / all 13 links and 12 joints EXACTLY retained.',flush=True)
    print('[H4 FRESH DISPLAY] Actual base XYZ m:',before['transform'][:3],'; refreshing 13 visual instances ONLY.',flush=True)
    async def frame():
        await omni.kit.app.get_app().next_update_async()
        if not retained() or time.monotonic()-begun>15. or actual()!=before:
            raise RuntimeError('Exact paused physical-state/UI budget changed; no physical recovery')
    try:
        began=True;refresh['begin'](stage,transaction)
        await frame();await frame()
        refresh['verify_transient'](stage,transaction)
        restoration=refresh['restore'](stage,transaction);restored=True
        await frame();await frame()
        if (stage.GetRootLayer().ExportToString()!=transaction['root_text']
                or stage.GetSessionLayer().ExportToString()!=transaction['session_text']
                or source!=launch['fresh_source_for_live_actor'](fresh,stage)
                or body_cache_before!=render.fabric_readback(stage,body_geometry)
                or settings_before!=render.display_setting_readback(settings,viewport)
                or geometry!=display.geometry_readback(stage)
                or projection!=display.viewport_readback(viewport,geometry)
                or depth!=render.camera_depth_readback(stage,viewport,geometry)):
            raise RuntimeError('Exact source/layers/physics/cache/settings/camera preservation not proven')
        write(output,dict(restoration,
            status='FRESH_PAUSED_STATIC_VISUAL_REFRESH_RESTORED_PIXELS_PENDING',actor=fresh.cohort.ROBOT,
            actual_tensor_before_and_after=before,actual_timeline_before_and_after=expected,
            geometry=geometry,camera_projection=projection,camera_depth=depth,
            existing_visual_cache_before=cache_before,
            existing_visual_cache_after=readers.existing_instance_cache(stage,topology),
            physical_state_source_parameters_and_body_cache_exactly_unchanged=True,
            actual_visible_recovery_confirmed=False,
            any_Play_Stop_reset_reopen_restage_or_physical_parameter_change=False,
            new_static_or_route_trial_count=0,actual_static_or_drive_admission=False,
            input_sha256={str(path):prep.sha(path) for path in
                (fresh.protocol_path,fresh.stationary.RESULT,fresh.stationary.TRACE,
                 Path(__file__),scripts/'hospital_h4_visual_instance_refresh.py')},
            use_for_final_statistics=False))
        print('=== H4 FRESH ROBOT DISPLAY REFRESH / EXACT STATE RETAINED ===',flush=True)
        print('Root / Session / body / all 12 joints / 13 links / camera unchanged: True',flush=True)
        print('saved:',output,flush=True)
        print('Inspect pixels WITHOUT F. No new static, route, Stop or Play.',flush=True)
    except BaseException as error:
        cleanup=None
        if began and not restored:
            try:
                refresh['restore'](stage,transaction);restored=True
            except Exception as cleanup_error:
                cleanup=type(cleanup_error).__name__+': '+str(cleanup_error)
        if not failure.exists():
            write(failure,dict(status='FRESH_VISUAL_REFRESH_ERROR_NOT_A_PHYSICAL_TRIAL',
                error=type(error).__name__+': '+str(error),original_visual_Session_restore_verified=restored,
                cleanup_error=cleanup,actual_static_or_drive_admission=False,automatic_retry=False))
        raise


def _h4_visual_refresh_done(task):
    import traceback
    try:task.result()
    except BaseException:
        print('[H4 VISUAL REFRESH] Task ended with an error; no silent background failure.',flush=True)
        traceback.print_exc()


import asyncio
import builtins
previous=getattr(builtins,'ijcas_h4_latest_visual_refresh_task',None)
if previous is not None and not previous.done():
    raise RuntimeError('A visual refresh is already active; no concurrent refresh')
fresh_protocol=globals().get('ijcas_hospital_h4_fresh_static_protocol')
if fresh_protocol is not None and not isinstance(fresh_protocol,str):
    raise RuntimeError('An explicit fresh static protocol path string is required')
workflow_owner=globals().get('ijcas_hospital_h4_visual_refresh_workflow_owner')
if workflow_owner is not None:
    if (fresh_protocol is None or workflow_owner is not asyncio.current_task()
            or workflow_owner is not getattr(builtins,'ijcas_h4_latest_experiment_task',None)):
        raise RuntimeError('Only the current approved workflow may await its own visual-only child')
operation=(_refresh_h4_visual_instances_once() if fresh_protocol is None
    else _refresh_h4_fresh_static_visuals(fresh_protocol,workflow_owner=workflow_owner))
ijcas_hospital_h4_visual_refresh_task=asyncio.ensure_future(operation)
builtins.ijcas_h4_latest_visual_refresh_task=ijcas_hospital_h4_visual_refresh_task
ijcas_hospital_h4_visual_refresh_task.add_done_callback(_h4_visual_refresh_done)
print('[H4 VISUAL REFRESH] Scheduled ONE visual-only refresh and exact restoration; NO motion.',flush=True)
