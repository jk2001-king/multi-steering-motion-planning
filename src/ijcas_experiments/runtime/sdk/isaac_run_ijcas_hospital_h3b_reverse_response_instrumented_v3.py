"""ONE NEW measured-braking/terminal rear-reverse attempt with passive SDK recording.

Require the NEW closedloop-cohort prefix PASS and read-only handoff, not the old
stalled endpoint. No observer action/init/reset/gain/material/sensor-API edits.
The authoritative primary motion controller is separate from observer coverage.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


def _schedule_reverse_closedloop_observation():
    import asyncio
    import csv
    import json
    import math
    import numpy as np
    from pathlib import Path
    import runpy

    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import PhysicsSchemaTools

    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task, 'done') and not task.done():
            raise RuntimeError('A hospital trial is running')
    workspace = _ijcas_workspace(); scripts = _ijcas_path(workspace, 'studies/IJCAS/scripts')
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    helper = load(scripts/'hospital_h3b_reverse_response_observation_v3.py')
    rules = load(scripts/'hospital_h3b_reverse_response_observation_v3.py'); helper['validate_runtime_api'](); info = rules['load_admitted_profile']()
    reviewed = helper['review'](info)
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    contact_helper = runpy.run_path(str(scripts/'hospital_h3_instrumentation.py'))
    timeline = omni.timeline.get_timeline_interface(); stage = omni.usd.get_context().get_stage()
    if timeline.is_stopped() or timeline.is_playing() or stage is None:
        raise RuntimeError('Keep the SAME NEW closedloop-cohort prefix PASS PAUSED; no reset/reopen/Play')
    marker = stage.GetPrimAtPath(support['COMPONENT'])
    if (not marker.IsValid() or marker.GetCustomDataByKey('status') != 'h3b_prefix_response_pilot_finished_fresh_setup_required'
            or marker.GetCustomDataByKey('last_h3b_reverse_force_prefix_pilot_gate') != info['first_result']):
        raise RuntimeError('Require the newly reproduced closedloop-cohort prefix, not either old FAIL scene')
    output = helper['OBSERVER_OUTPUT']
    if output.exists() or rules['OUTPUT'].exists():
        raise RuntimeError('One force-observation attempt per cohort; preserve outcome before review')
    first = info['first_record']; prep = first['physics_preparation_snapshot']
    before_root = stage.GetRootLayer().ExportToString(); before_session = stage.GetSessionLayer().ExportToString()
    sim = SimulationManager.get_physics_sim_view()
    if sim is None: raise RuntimeError('Existing tensor view unavailable; observer will not initialize or reset')
    view = sim.create_articulation_view(rules['ROBOT_ROOT']+'/base_link')
    if view.count != 1: raise RuntimeError('Exactly one existing articulation required')
    helper['telemetry_snapshot'](view, 0.)
    def retained_tensor_check():
        names=list(view.shared_metatype.dof_names)
        def numeric(value):
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            elif hasattr(value,'numpy'):value=value.numpy()
            return np.asarray(value,dtype=float).reshape(-1)
        settings={}
        for key,getter in (('kp',view.get_dof_stiffnesses),('kd',view.get_dof_dampings),
                ('force_limit',view.get_dof_max_forces),('velocity_target_rad_s',view.get_dof_velocity_targets)):
            settings[key]=dict(zip(names,map(float,numeric(getter()))))
        return helper['validate_tensor'](info,numeric(view.get_root_transforms()),names,
            numeric(view.get_dof_positions()),numeric(view.get_root_velocities()),settings)
    initial_tensor=retained_tensor_check()
    patterns = [rules['ROBOT_ROOT']+'/'+n for n in rules['WHEELS']]
    friction_view = sim.create_rigid_contact_view(patterns, [[prep['support_floor']] for _ in patterns], max_contact_data_count=256)
    sensor_paths = [str(p) for p in friction_view.sensor_paths]
    if friction_view.sensor_count != 6 or friction_view.filter_count != 1 or len(sensor_paths) != 6:
        raise RuntimeError('Native six-wheel canonical-floor friction recording is unsupported')
    matched = []
    for path in sensor_paths:
        matches = [p for p in patterns if path == p or path.startswith(p+'/')]
        if len(matches) != 1: raise RuntimeError('Native sensor does not identify one expected wheel')
        matched.extend(matches)
    if len(set(matched)) != 6 or set(matched) != set(patterns): raise RuntimeError('Native sensor coverage is not one-to-one')
    # Zero cached anchors while PAUSED are allowed here; all six wheels must
    # actually produce data DURING motion for observer coverage to be complete.
    helper['friction_snapshot'](friction_view, 0., 1./120., sensor_paths)
    if (stage.GetRootLayer().ExportToString() != before_root or stage.GetSessionLayer().ExportToString() != before_session
            or timeline.is_stopped() or timeline.is_playing()):
        raise RuntimeError('Observer preflight changed scene/timeline; primary NOT launched')
    if retained_tensor_check()!=initial_tensor: raise RuntimeError('Retained body/joints/settings changed during observer SDK preflight; no motion launched')
    clock = {'time_s': 0., 'next_friction_s': 0.}
    joint_rows = []; friction_rows = []; anchors = []; raw_contacts = []; errors = []
    def remember(error):
        message = f'{type(error).__name__}: {error}'
        if message not in errors and len(errors) < 100: errors.append(message)
    def step(dt):
        try:
            dt = float(dt)
            if not math.isfinite(dt) or not 0. < dt <= .025: raise ValueError('Invalid observer timestep')
            clock['time_s'] += dt
            if len(joint_rows) >= 12000: raise RuntimeError('Observer joint capacity reached; primary policy unchanged')
            joint_rows.append(helper['telemetry_snapshot'](view, clock['time_s']))
            if clock['time_s']+1e-8 >= clock['next_friction_s']:
                clock['next_friction_s'] += helper['FRICTION_PERIOD_S']
                row, points = helper['friction_snapshot'](friction_view, clock['time_s'], dt, sensor_paths)
                if len(anchors)+len(points) > 100000: raise RuntimeError('Observer anchor capacity reached; primary unchanged')
                friction_rows.append(row); anchors.extend(points)
        except Exception as error: remember(error)
    def contacts(headers, data):
        try:
            points = contact_helper['contact_points'](headers, data, clock['time_s'], PhysicsSchemaTools.intToSdfPath,
                lambda pair: rules['classify_contact'](pair, prep['support_floor'], prep['disabled_legacy_floor_colliders']))
            if len(raw_contacts)+len(points) > 500000: raise RuntimeError('Observer contact capacity reached; primary unchanged')
            raw_contacts.extend(points)
        except Exception as error: remember(error)
    physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(step)
    contact_sub = None; primary = None
    try:
        contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
        globals()['ijcas_hospital_h3b_reverse_response_observer_ready'] = 'PASSIVE_OBSERVER_PREFLIGHT_COMPLETE'
        entry = _ijcas_path(workspace, 'isaac/scripts/isaac_run_ijcas_hospital_h3b_reverse_response_primary_v3.py')
        exec(compile(entry.read_text(encoding='utf-8'), str(entry), 'exec'), globals())
        primary = globals()['ijcas_hospital_h3b_reverse_response_pilot_task']
        state = primary.get_coro().cr_frame.f_locals
        run_id = state['run_id']; primary_path = Path(state['summary_path'])
        output.mkdir(parents=True, exist_ok=False)
    except Exception:
        # No await/yield: a scheduled primary has not issued action/Play yet.
        if primary is not None and not primary.done(): primary.cancel()
        physics_sub = contact_sub = None
        raise
    async def finish():
        nonlocal physics_sub, contact_sub
        report = dict(run_id=run_id, role='SAME_MEASURED_TERMINAL_FIRST_REVERSE_PASSIVE_RESPONSE_OBSERVATION_NOT_VALIDATION', status='INCOMPLETE',
            reference_trial=reviewed['reference_trial'], reference_trial_sha256=reviewed['reference_trial_sha256'],
            reference_input_sha256=reviewed['input_sha256'], control_function_contract=reviewed['controller_contract'],
            actual_prefix=info['first_result'], actual_prefix_sha256=info['first_result_sha256'],
            initial_retained_tensor_check=initial_tensor,six_wheel_transforms_COM_and_origin_velocities_recorded=True,
            response_observation_protocol=str(helper['PROTOCOL_PATH']),response_observation_protocol_sha256=helper['sha'](helper['PROTOCOL_PATH']),
            original_controller_result=str(primary_path),
            control_or_physics_parameters_modified_by_observer=False, stop_contact_envelope_and_terminal_acceptance_unchanged=True, new_progress_scope_includes_braking=True,
            disclosed_high_level_executor_changed=True, reverse_speed_cap_m_s=.050,
            joint_observer_nominal_hz=120., friction_observer_nominal_hz=30., native_friction_buffer_capacity=256,
            native_sensor_paths_in_recorded_wheel_index_order=sensor_paths, native_filter=prep['support_floor'],
            forces_are_SDK_last_step_impulses_divided_by_physics_dt_NOT_observer_period=True,
            SDK_world_vector_sign_convention_retained_not_assumed_motor_or_traction_sign=True,
            projected_joint_effort_is_NOT_isolated_actuator_motor_torque=True,
            callback_timestamp_uncertainty_s=1./120., callback_and_controller_samples_NOT_substep_synchronized=True,
            SDK_root_velocity_at_COM_and_converted_link_origin_velocity_both_recorded=True,
            old_original_FAIL_preserved=True, full_H3B_turning_path_verified=False, use_for_final_statistics=False,
            source_sha256={p.name: support['sha'](p) for p in (
                _ijcas_path(workspace, 'isaac/scripts/isaac_run_ijcas_hospital_h3b_reverse_response_instrumented_v3.py'),
                scripts/'hospital_h3b_reverse_response_observation_v3.py', scripts/'hospital_h3b_rolling_observation.py',
                scripts/'hospital_h3_instrumentation.py', scripts/'hospital_h3b_alignment_audit.py', scripts/'hospital_private_loader.py')})
        original = {}
        try:
            app = omni.kit.app.get_app()
            while not primary.done(): await app.next_update_async()
            if primary.cancelled() or primary.exception() is not None: raise RuntimeError('Primary ended abnormally; retain its authoritative receipt')
            original = json.loads(primary_path.read_bytes())
            report.update(original_controller_verdict=original['verdict'],original_controller_source_sha256=original['source_sha256'],original_error=original.get('error'),
                          original_controller_result_sha256=support['sha'](primary_path),
                          original_motor_gains_restored=original.get('original_motor_gains_restored', False))
            if helper['function_contract']() != reviewed['controller_contract']:
                raise RuntimeError('Control/guard functions changed during observation')
            support['verify_files'](reviewed['input_sha256'])
        except (Exception, asyncio.CancelledError) as error: remember(error)
        finally:
            physics_sub = contact_sub = None
            for name, rows in (('joint_body_solver_trace.csv', joint_rows), ('wheel_friction_summary.csv', friction_rows),
                               ('friction_anchors.csv', anchors), ('raw_contact_points.csv', raw_contacts)):
                if rows:
                    with (output/name).open('x', newline='', encoding='utf-8') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            report.update(helper['coverage'](joint_rows, friction_rows, anchors, raw_contacts, original, errors))
            report['telemetry_errors'] = errors
            report['saved_file_sha256'] = {p.name: support['sha'](p) for p in output.glob('*.csv')}
            with (output/'instrumentation.json').open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
            print('=== H3-B SAME MEASURED-TERMINAL RESPONSE OBSERVATION REAR-REVERSE FORCE INSTRUMENTATION ===')
            print('original controller verdict:', report.get('original_controller_verdict', 'NOT_AVAILABLE'))
            print('joint / friction samples / anchors:', len(joint_rows), len(friction_rows), len(anchors))
            print('each wheel has friction data:', report['each_wheel_has_friction_anchor_observations'])
            print('telemetry errors:', errors)
            print('STATUS:', report['status']); print('saved:', output/'instrumentation.json')
            print('Keep PAUSED. Motion PASS/FAIL remains authoritative; observer completion is NOT reverse success or an isolated cause.')
    print('SAME already-PASS measured-terminal rear reverse; retained actual CP50-prefix history. No control/model fit or physics change; one passive observation only.')
    print('Passive 120Hz body/twelve joints/SIX wheel poses+velocities and 30Hz native friction/raw contacts attached BEFORE motion.')
    print('Another FAIL is valid. No automatic retry or later suffix leg; do not touch timeline controls.')
    return asyncio.ensure_future(finish())


ijcas_hospital_h3b_reverse_response_observer_task = _schedule_reverse_closedloop_observation()

