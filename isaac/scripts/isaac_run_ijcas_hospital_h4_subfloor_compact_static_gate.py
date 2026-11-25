"""Compact-staged subfloor static trial; SEPARATE Play consent required.

Original zero-command controls/physical limits and 26 adopted support rays.
Four extra cooked identity-absence probes; no obstacle contact exception.
No actual preparation may be synthesized, replayed, reset or retried.
"""

if globals().get('ijcas_hospital_h4_subfloor_compact_static_authorized') is not True:
    raise RuntimeError('Separate explicit authorization for this ONE subfloor static Play is required; preparation approval is insufficient')


async def _run_h4_compact_static(attempt):
    import asyncio
    from collections import Counter
    import csv
    import json
    import math
    import os
    from pathlib import Path
    import runpy
    import time

    import carb
    import numpy as np
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.api.sensors import RigidContactView
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import PhysicsSchemaTools, Usd, UsdUtils

    if globals().get('ijcas_hospital_h4_subfloor_compact_static_authorized') is not True:
        raise RuntimeError('Explicit user authorization for this ONE automatic H4 static Play is required')
    scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    r = load(scripts/'hospital_h4_subfloor_compact_static_gate.py')
    c = r['cohort']
    protocol = r['load_protocol']()
    prep = r['load_actual_preparation']()
    stage, timeline = omni.usd.get_context().get_stage(), omni.timeline.get_timeline_interface()
    if stage is None or not timeline.is_stopped():
        raise RuntimeError('H4 must be STOPPED after successful actual subfloor-variant staging; no reset/reopen/Play')
    marker = stage.GetPrimAtPath(c.MARKER)
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'subfloor_environment_staged_STATIC_PENDING_NO_PLAY'
            or marker.GetCustomDataByKey('robot_root') != c.ROBOT
            or marker.GetCustomDataByKey('physics_preparation_result') != str(c.LIVE)
            or marker.GetCustomDataByKey('preparation_protocol') != str(c.PREFLIGHT)
            or marker.GetCustomDataByKey('preparation_protocol_sha256') != c.completion.sha(c.PREFLIGHT)):
        raise RuntimeError('Exact actual subfloor-cohort marker/preparation required; old controllers are not allowed')
    environment = r['validate_stopped_handoff'](stage, timeline, protocol, prep)
    source_stage = Usd.Stage.Open(str(c.ASSET))
    readback = c.validate_new_robot(stage, source_stage, protocol['start_xyyaw'])
    if readback != prep['live_new_robot_readback']:
        raise RuntimeError('Actual H4 robot pose/source seed/settings differ from its own PRE-Play preparation')
    if any(path.exists() for path in (r['RESERVATION'], r['RESULT'], r['TRACE'], r['PREFLIGHT_HOLD'])):
        raise RuntimeError('This H4 cohort already reserved or ran its single static attempt; no overwrite/retry')
    c.write_exclusive(r['RESERVATION'],dict(run_id=r['RUN_ID'], user_authorized=True,
        protocol_sha256=c.completion.sha(r['PROTOCOL']), actual_preparation_sha256=c.completion.sha(c.LIVE),
        actual_static_result_not_yet_available=True, no_automatic_retry=True))
    attempt['owns_reservation'] = True
    manifest = c.load_manifest(verify_inputs=False)
    context = omni.usd.get_context()

    async def compact_environment_readback(phase):
        # Only run while STOPPED before Play, or PAUSED after observation.
        expected_state = (bool(timeline.is_playing()), bool(timeline.is_stopped()), float(timeline.get_current_time()))
        if expected_state[0]:
            raise RuntimeError('Never perform full environment hashing during the physics observation window')
        layers = (stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString())
        began_hash = heartbeat_hash = yielded_hash = time.monotonic()
        result = {}

        def unchanged():
            if (context.get_stage() != stage
                    or (bool(timeline.is_playing()),bool(timeline.is_stopped()),float(timeline.get_current_time())) != expected_state
                    or (stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString()) != layers):
                raise RuntimeError('Stage/timeline changed during nonmoving compact environment readback')
            if time.monotonic()-began_hash > protocol['compact_readback_wall_budget_s']:
                raise TimeoutError('Nonmoving compact environment readback budget exceeded; no automatic retry')

        for path, row in c.compact.stage_rows(stage,c.SELECTED,c.old.original):
            if path in result:
                raise RuntimeError('Duplicate protected environment prim identity')
            result[path] = row
            wall = time.monotonic()
            if wall-heartbeat_hash >= protocol['compact_readback_heartbeat_s']:
                print('[H4 COMPACT STATIC READBACK] phase=%s rows=%d/%d wall=%.1fs; NO motion' %
                    (phase,len(result),protocol['all_archived_environment_rows_required'],wall-began_hash),flush=True)
                heartbeat_hash = wall
            if (len(result)%protocol['compact_readback_UI_yield_max_rows'] == 0
                    or wall-yielded_hash >= protocol['compact_readback_UI_yield_wall_interval_s']):
                unchanged()
                await omni.kit.app.get_app().next_update_async()
                unchanged()
                yielded_hash = time.monotonic()
        unchanged()
        fingerprint = c.compact.compare_rows(manifest['compact_environment_rows'],result,c.SELECTED,after_flags=True)
        if fingerprint != prep['protected_environment_sha256'] or len(result) != protocol['all_archived_environment_rows_required']:
            raise RuntimeError('Actual compact-staged full environment changed; no geometry/flag/material waiver')
        return dict(phase=phase,all_original_row_count=len(result),protected_environment_sha256=fingerprint,
            only_exact_four_approved_collision_flags_differ_from_original_archive=True,
            timeline_and_Root_Session_unchanged_during_readback=True,
            whole_archive_redecoded_or_hashed_inside_Kit=False,wall_time_s=time.monotonic()-began_hash)

    before_compact = await compact_environment_readback('STOPPED_before_separate_static_Play')
    r['validate_stopped_handoff'](stage,timeline,protocol,prep)
    base = c.ROBOT+'/base_link'
    steering = ('base_link_to_front_active_steering', 'base_link_to_rear_active_steering')
    drives = ('front_active_steering_to_drive_wheel', 'rear_active_steering_to_drive_wheel')
    clock, counts, pairs, callback_errors = {'time_s': 0.}, Counter(), Counter(), []

    def physics_step(dt):
        value = float(dt)
        if not math.isfinite(value) or value <= 0.:
            callback_errors.append('Invalid physics step duration')
        else:
            clock['time_s'] += value

    def contacts(headers, data):
        del data
        try:
            for header in headers:
                pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
                category = r['classify_contact'](pair, environment['disabled_legacy_colliders'])
                if category not in r['KNOWN_CATEGORIES']:
                    raise ValueError('Unknown contact classification: '+str(category))
                if category != 'unrelated':
                    counts[category] += 1; pairs[(pair, category)] += 1
        except Exception as error:
            callback_errors.append(type(error).__name__+': '+str(error))

    def angles(quaternion):
        w, x, y, z = map(float, quaternion)
        return (math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x))))),
            math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))))

    def sampled_queries():
        query = omni.physx.get_physx_scene_query_interface()
        result = []
        for spec in protocol['sampled_cooked_support_queries']:
            hits, errors, self_hits = [], [], []

            def receive(hit):
                try:
                    record = dict(collision=str(hit.collision), position_m=list(map(float, hit.position)),
                        normal=list(map(float, hit.normal)), distance_m=float(hit.distance))
                    # Filter ONLY the new robot's own ray occlusion. Old robot,
                    # unknown supports and every hospital collider stay visible.
                    if record['collision'].startswith(c.ROBOT+'/'):
                        self_hits.append(record)
                    else:
                        hits.append(record)
                    return True
                except Exception as error:
                    errors.append(type(error).__name__+': '+str(error))
                    return False

            try:
                query.raycast_all(carb.Float3(*spec['xy_m'], .6), carb.Float3(0., 0., -1.), .8, receive)
            except Exception as error:
                errors.append(type(error).__name__+': '+str(error))
            hits.sort(key=lambda hit: hit['distance_m'])
            closest = dict(hit=False) if not hits else dict(hit=True, **hits[0])
            result.append(dict(id=spec['id'], xy_m=spec['xy_m'], **closest, errors=errors,
                all_nonself_hits=hits, own_robot_occlusion_hits=self_hits))
        return result

    def sampled_subfloor_queries():
        query = omni.physx.get_physx_scene_query_interface()
        result = []
        for spec in protocol['additional_disabled_subfloor_probes']:
            hits, errors, self_hits = [], [], []

            def receive(hit):
                try:
                    record = dict(collision=str(hit.collision), position_m=list(map(float, hit.position)),
                        normal=list(map(float, hit.normal)), distance_m=float(hit.distance))
                    # ONLY own actor occlusion is separated. Disabled meshes
                    # and every other identity remain in the raw hit records.
                    if record['collision'].startswith(c.ROBOT+'/'):
                        self_hits.append(record)
                    else:
                        hits.append(record)
                    return True
                except Exception as error:
                    errors.append(type(error).__name__+': '+str(error))
                    return False

            try:
                query.raycast_all(carb.Float3(*spec['origin_m']), carb.Float3(*spec['direction']), spec['distance_m'], receive)
            except Exception as error:
                errors.append(type(error).__name__+': '+str(error))
            result.append(dict(id=spec['id'], all_nonself_hits=sorted(hits, key=lambda h: h['distance_m']),
                own_robot_occlusion_hits=self_hits, errors=errors))
        return result

    async def run():
        report = dict(version=r['VERSION'], cohort=c.COHORT, scenario='H4_ward_corridor_turn', run_id=r['RUN_ID'],
            trial_kind='H4_compact_staged_subfloor_same_static_plus_four_cooked_identity_probes_only',
            verdict='FAIL', checks={'stationary_observation_completed': False},
            geometry_manifest=str(c.SETUP), geometry_manifest_sha256=c.completion.sha(c.SETUP),
            physics_preparation=str(c.LIVE), physics_preparation_sha256=c.completion.sha(c.LIVE),
            physics_preparation_snapshot=prep, static_protocol=str(r['PROTOCOL']),
            static_protocol_sha256=c.completion.sha(r['PROTOCOL']), start_xyyaw=protocol['start_xyyaw'],
            simulation_mass_kg=265., settle_s=r['SETTLE_S'], sampling_s=r['RECORD_S'],
            support_colliders=list(r['SUPPORTS']), source_robot_seed_checked_before_Play=True,
            measured_post_Play_joint_angles_are_NOT_the_authored_pre_Play_seed=True,
            initial_motion_executed=False, planned_path_executed=False, CICS_input_used=False, Nav2_executed=False,
            caster_joint_state_reset_during_run=False, material_modified_during_run=False,
            motor_gains_modified_by_test=False, source_stage_saved=False, use_for_final_statistics=False,
            full_H4_route_or_continuous_cooked_support_validated=False, actual_drive_admission=False,
            query_definition_frozen_before_new_trial=True, known_old_contact_route_still_NOT_admitted=True,
            exactly_one_reserved_attempt_no_automatic_retry=True,
            compact_environment_before_Play=before_compact,
            full_environment_hash_checks_OUTSIDE_physics_recording_window=True,
            input_sha256={**protocol['input_sha256'], **prep['input_sha256'],
                str(r['PROTOCOL']): c.completion.sha(r['PROTOCOL']), str(c.LIVE): c.completion.sha(c.LIVE)})
        rows, views, queries, subfloor_queries = [], [], [], []
        robot = motor = steer_indices = drive_indices = old_kp = old_kd = old_efforts = None
        physics_sub = contact_sub = None
        wall_start, heartbeat = time.monotonic(), time.monotonic()

        def same_gains():
            kp, kd = motor.get_gains()
            return bool(np.array_equal(np.asarray(kp), old_kp) and np.array_equal(np.asarray(kd), old_kd)
                and np.array_equal(np.asarray(motor.get_max_efforts()), old_efforts))

        def zero_command():
            robot.apply_action(ArticulationAction(joint_positions=np.zeros(2, dtype=np.float32), joint_indices=steer_indices))
            robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_indices))

        try:
            physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(physics_step)
            contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
            for label, name in r['WHEELS'].items():
                views.append(RigidContactView(prim_paths_expr=c.ROBOT+'/'+name, filter_paths_expr=[],
                    name=r['RUN_ID']+'_'+label, prepare_contact_sensors=False, disable_stablization=False))
            app = omni.kit.app.get_app()
            attempt['static_Play_requested'] = True
            timeline.play()
            for _ in range(5):
                await app.next_update_async()
            if not timeline.is_playing():
                raise RuntimeError('H4 timeline changed during static initialization')
            robot = SingleArticulation(base); robot.initialize()
            names = list(robot.dof_names)
            indices = {name: i for i, name in enumerate(names)}
            if len(names) != 12 or any(name not in indices for name in (*steering, *drives)):
                raise RuntimeError('Full twelve-joint new H4 articulation and active joint mapping required')
            steer_indices = np.array([indices[n] for n in steering], dtype=np.int32)
            drive_indices = np.array([indices[n] for n in drives], dtype=np.int32)
            motor = robot.get_articulation_controller()
            kp, kd = motor.get_gains()
            old_kp, old_kd, old_efforts = np.asarray(kp).copy(), np.asarray(kd).copy(), np.asarray(motor.get_max_efforts()).copy()
            if np.max(np.abs(old_kp[drive_indices])) > 1e-6:
                raise RuntimeError('Original velocity-only drive gains required')
            if (not np.isfinite(old_kp).all() or not np.isfinite(old_kd).all()
                    or np.isnan(old_efforts).any() or np.isneginf(old_efforts).any()):
                raise RuntimeError('Invalid runtime motor baseline')
            report['runtime_motor_baseline'] = dict(joint_names=names, kp=old_kp.tolist(), kd=old_kd.tolist(),
                max_efforts=[float(v) if np.isfinite(v) else 'positive_infinity' for v in old_efforts])
            for view in views:
                view.initialize()
            sim = omni.physx.get_physx_simulation_interface()
            stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            encoded = PhysicsSchemaTools.sdfPathToInt(base)
            # Same initialization/1s settling/2s recording convention as H1/H2.
            began = previous = clock['time_s']
            report['initialization_simulation_duration_s'] = began
            next_sample = r['SETTLE_S']
            while clock['time_s']-began < r['SETTLE_S']+r['RECORD_S']:
                zero_command(); sim.wake_up(stage_id, encoded)
                await app.next_update_async()
                if not timeline.is_playing():
                    raise RuntimeError('H4 timeline changed during static observation')
                wall = time.monotonic()
                if wall-wall_start > r['GUARDS']['maximum_wall_time_s']:
                    raise TimeoutError('H4 static 120s wall-time guard')
                now = clock['time_s']
                if now-previous > r['GUARDS']['maximum_control_gap_s']:
                    raise TimeoutError('H4 static 0.25s control-gap guard')
                previous = now
                position, quaternion = robot.get_world_pose()
                position, quaternion = np.asarray(position, float), np.asarray(quaternion, float)
                if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                    raise RuntimeError('Non-finite actual static pose')
                if np.linalg.norm(position[:2]-np.asarray(protocol['start_xyyaw'][:2])) > r['GUARDS']['maximum_xy_displacement_m']:
                    raise RuntimeError('H4 zero-command drift exceeded 30mm safety guard')
                roll, pitch, yaw = angles(quaternion)
                if max(abs(roll), abs(pitch)) > r['GUARDS']['maximum_abs_roll_pitch_deg']:
                    raise RuntimeError('H4 static tilt exceeded 2deg safety guard')
                if callback_errors or any(counts[key] for key in r['BAD_CONTACTS']):
                    raise RuntimeError('Unexpected obstacle/nonwheel/legacy/unclassified external contact')
                if not same_gains():
                    raise RuntimeError('Original motor gains/damping/limits changed during static test')
                elapsed = now-began
                if wall-heartbeat >= 15.:
                    print('[H4 STATIC] wall=%.1fs sim=%.3fs samples=%d contacts=%s' %
                        (wall-wall_start, elapsed, len(rows), dict(counts)), flush=True)
                    heartbeat = wall
                if elapsed < next_sample:
                    continue
                forces = [np.asarray(view.get_net_contact_forces(dt=1./r['PHYSICS_HZ']), float).reshape(-1, 3)[0] for view in views]
                if not np.isfinite(forces).all():
                    raise RuntimeError('Non-finite actual wheel support force')
                row = dict(time_s=elapsed, x_m=float(position[0]), y_m=float(position[1]), z_m=float(position[2]),
                    roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
                row.update({label+'_fz_n': float(force[2]) for label, force in zip(r['WHEELS'], forces)})
                rows.append(row); next_sample += 1./60.
            zero_command()
            report['checks']['stationary_observation_completed'] = True
        except (Exception, asyncio.CancelledError) as error:
            report['error'] = type(error).__name__+': '+str(error)
        finally:
            try:
                if robot is not None and drive_indices is not None and timeline.is_playing():
                    zero_command()
            except Exception as error:
                report['checks']['final_zero_command_sent'] = False
                report['zero_command_error'] = str(error)
            if timeline.is_playing():
                timeline.pause()
            physics_sub = contact_sub = None
            try:
                if timeline.is_stopped():
                    raise RuntimeError('Timeline was stopped/reset; final static state no longer retained')
                if motor is None or old_kp is None:
                    raise RuntimeError('Actual runtime motor readback unavailable')
                report['checks']['original_motor_gains_and_limits_unchanged'] = same_gains()
                position, quaternion = robot.get_world_pose()
                roll, pitch, yaw = angles(quaternion)
                joint_positions = np.asarray(robot.get_joint_positions(), float)
                joint_velocities = np.asarray(robot.get_joint_velocities(), float)
                if (not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                        or not np.isfinite(joint_positions).all() or not np.isfinite(joint_velocities).all()):
                    raise RuntimeError('Non-finite actual post-static body/twelve-joint readback')
                report['final_actual_world_xyyaw'] = [float(position[0]), float(position[1]), math.radians(yaw)]
                report['final_actual_joint_names'] = list(robot.dof_names)
                report['final_actual_twelve_joint_positions_rad'] = joint_positions.tolist()
                report['final_actual_twelve_joint_velocities_rad_s'] = joint_velocities.tolist()
                report['environment_after_static'] = r['critical_environment'](stage,prep)
                report['compact_environment_after_Pause'] = await compact_environment_readback('PAUSED_after_static_observation')
                report['checks']['all_4997_original_environment_rows_unchanged_except_approved_four_flags'] = True
                report['checks']['nominal_actor_fixed_parameters_unchanged'] = r['actor_configuration'](stage) == protocol['nominal_actor_fixed_configuration']
                report['checks']['original_support_material_scene_and_hospital_colliders_unchanged'] = report['environment_after_static'] == environment
                c.verify(report['input_sha256'])
                report['checks']['all_original_input_evidence_unchanged'] = True
                queries = sampled_queries()
                subfloor_queries = sampled_subfloor_queries()
                probe_evaluation = r['evaluate_subfloor_queries'](protocol, queries, subfloor_queries)
                report['checks'].update(probe_evaluation['checks'])
                report['auxiliary_probe_diagnostics'] = probe_evaluation['auxiliary_probe_diagnostics']
                evaluated = r['evaluate'](rows, protocol['start_xyyaw'], counts, callback_errors,
                    protocol['sampled_cooked_support_queries'], queries, protocol['support_top_m'])
                r['merge_evaluation'](report, evaluated)
            except Exception as error:
                report['checks']['final_readback_and_evaluation_completed'] = False
                report['evaluation_error'] = type(error).__name__+': '+str(error)
            report['checks']['contact_callback_monitor_healthy'] = not callback_errors
            report['checks']['only_expected_external_support'] = not any(counts[key] for key in r['BAD_CONTACTS'])
            report['checks']['only_known_contact_categories_recorded'] = not (set(counts)-r['KNOWN_CATEGORIES'])
            if report.get('error') or not all(report['checks'].values()):
                report['verdict'] = 'FAIL'
            if not attempt.get('static_Play_requested'):
                # A sensor/subscription setup error before Play is not a
                # physical FAIL. Preserve it without manufacturing a trial.
                report['verdict'] = 'HOLD'
            report['automatic_static_Play_requested'] = bool(attempt.get('static_Play_requested'))
            report['actual_physics_callback_duration_s'] = clock['time_s']
            report['physical_outcome_not_inferred_from_reservation_or_setup_only'] = True
            report['actual_final_timeline'] = dict(playing=bool(timeline.is_playing()),
                stopped=bool(timeline.is_stopped()),current_time_s=float(timeline.get_current_time()))
            report.update(contact_report_counts_by_category=dict(counts), contact_callback_errors=callback_errors,
                contact_pairs_by_category=[dict(pair=pair, category=category, callback_count=count)
                    for (pair, category), count in sorted(pairs.items())],
                actual_sampled_cooked_support_queries=queries,
                actual_disabled_subfloor_probes=subfloor_queries,
                approved_disabled_subfloor_paths=protocol['approved_disabled_subfloor_paths'],
                prior_environment_outcomes_NOT_pooled_or_relabelled=True,
                wall_time_s=time.monotonic()-wall_start)
            # Optional current-state GETs after Pause. Never advance physics,
            # infer old contact forces or replace the static core verdict.
            try:
                from isaacsim.core.simulation_manager import SimulationManager
                view = SimulationManager.get_physics_sim_view()
                if view is None:
                    raise RuntimeError('Current tensor view unavailable; no Play to create it')
                report['passive_post_static_link_readback'] = r['passive_final_link_readback'](
                    view.create_articulation_view(base))
                report['passive_post_static_link_readback_errors'] = []
            except Exception as error:
                report['passive_post_static_link_readback'] = None
                report['passive_post_static_link_readback_errors'] = [type(error).__name__+': '+str(error)]
            if rows:
                with r['TRACE'].open('x', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                    stream.flush(); os.fsync(stream.fileno())
                report['trace_sha256'] = c.completion.sha(r['TRACE'])
            c.write_exclusive(r['RESULT'],report)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                marker.SetCustomDataByKey('status', {'PASS':'static_support_passed',
                    'FAIL':'static_support_failed','HOLD':'static_support_not_started_HOLD'}[report['verdict']])
                marker.SetCustomDataByKey('static_gate_result', str(r['RESULT']))
                marker.SetCustomDataByKey('static_gate_sha256', c.completion.sha(r['RESULT']))
            print('=== H4 SUBFLOOR VARIANT STATIC / DUAL-QUERY SUPPORT AND FOUR IDENTITY PROBES ===')
            print('run:', r['RUN_ID'])
            for label, force in report.get('mean_wheel_vertical_loads_n', {}).items():
                print('%s: %.2f N' % (label, force))
            print('contact categories:', dict(counts))
            if 'maximum_xy_drift_m' in report:
                print('maximum XY drift mm:', round(1000*report['maximum_xy_drift_m'], 3))
                print('vertical force / weight:', round(report['mean_vertical_force_to_weight_ratio'], 4))
            print('sampled cooked support hits:', sum(row['hit'] for row in queries), '/', len(protocol['sampled_cooked_support_queries']))
            print('additional disabled-identity probe transactions:', len(subfloor_queries), '/ 4')
            print('failed checks:', [key for key, ok in report['checks'].items() if not ok])
            if report.get('error') or report.get('evaluation_error'):
                print('error:', report.get('error') or report.get('evaluation_error'))
            print('VERDICT:', report['verdict']); print('saved:', r['RESULT'])
            print('Actual timeline:',report['actual_final_timeline'])
            print('H4 START static scope ONLY; pre-Play HOLD is NOT a physical trial. No path/whole-route/Nav2/CICS result.')
            print('Do not Play/save/reset or use old controllers. No automatic retry.')

    print('H4 COMPACT zero-command static support admitted:', r['RUN_ID'])
    print('Automatic Play -> 1s settling -> 2s recording -> Pause. ZERO forward/CICS/Nav2; do not touch timeline controls.')
    await run()


async def _dispatch_h4_compact_static():
    # A pre-Play failure is a HOLD, not a fabricated physical FAIL/PASS.
    attempt = {}
    try:
        await _run_h4_compact_static(attempt)
    except (Exception, asyncio.CancelledError) as error:
        import runpy
        from pathlib import Path
        scripts = Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
        r = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load'](scripts/'hospital_h4_subfloor_compact_static_gate.py')
        if (attempt.get('owns_reservation') and r['RESERVATION'].exists()
                and not r['RESULT'].exists() and not r['PREFLIGHT_HOLD'].exists()):
            requested=bool(attempt.get('static_Play_requested'))
            r['preparation'].write_exclusive(r['PREFLIGHT_HOLD'],dict(
                status=('COMPACT_STATIC_POST_LAUNCH_RECORDING_INCOMPLETE_NO_VERDICT' if requested else
                    'RESERVED_COMPACT_STATIC_PREFLIGHT_HOLD_NO_PHYSICAL_RESULT'),
                static_Play_requested=requested,
                physical_outcome_NOT_inferred_from_missing_result=True,
                error=type(error).__name__+': '+str(error),actual_static_or_drive_admission=False,
                automatic_retry_or_recovery=False,old_PASS_FAIL_HOLDs_preserved=True))
        raise


import asyncio
ijcas_hospital_h4_subfloor_compact_static_task = asyncio.ensure_future(_dispatch_h4_compact_static())
print('H4 separate COMPACT static preflight scheduled; no Play before all actual checks. No Stop/reset/retry.')
