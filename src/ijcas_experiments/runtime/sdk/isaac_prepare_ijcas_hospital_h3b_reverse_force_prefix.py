"""Fresh source -> reproduce the SAME fixed forward prefix in a new cohort.

No caster-angle assignment or paused-state recovery. Review the actual new
endpoint before any rear-only reverse component is authorized.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace


ijcas_hospital_h3b_reverse_force_prefix_condition='H3B_rear_reverse_prefix_reproduction_v1'

def _prepare_rear_prefix_pilot():
    import json
    from pathlib import Path
    import runpy

    import omni.timeline
    import omni.usd
    from pxr import Usd

    if globals().pop('ijcas_hospital_h3b_reverse_force_prefix_condition',None)!='H3B_rear_reverse_prefix_reproduction_v1':
        raise RuntimeError('Explicit H3B_rear_reverse_prefix_reproduction_v1 selection required')
    workspace=_ijcas_workspace()
    scripts,entries=_ijcas_path(workspace, 'studies/IJCAS/scripts'),_ijcas_path(workspace, 'isaac/scripts')
    load=runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h=load(scripts/'hospital_h3b_reverse_force_prefix.py')
    support=runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    info=h['load_profile']()
    h['interpolation_table'](info)
    stage=omni.usd.get_context().get_stage()
    if stage is None or not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError('Reopen ORIGINAL hospital source STOPPED, not a previous moving/paused scene')
    if stage.GetPrimAtPath(support['COMPONENT']).IsValid():
        raise RuntimeError('Old component/session is still present. Reopen the original hospital USD without saving session changes; Stop alone is insufficient')
    for name,task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task,'done') and not task.done():
            raise RuntimeError('Another hospital trial is running')
    output=h['OUTPUT']
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('One prospective first-leg attempt per cohort; preserve its outcome before review')
    folder=output.parent/'preparations';folder.mkdir(parents=True,exist_ok=True)
    number=1
    while any((folder/f'prepare_r{number:02d}{s}').exists() for s in ('.json','_failure.json')):number+=1
    target=folder/f'prepare_r{number:02d}.json'
    session=stage.GetSessionLayer();before=session.ExportToString()
    try:
        # Geometry/support ONLY: no reversal command or readiness motion here.
        globals()['ijcas_hospital_h3b_prepare_condition']='H3B_common_straight_reversal'
        entry=entries/'isaac_prepare_ijcas_hospital_h3b_reverse_readiness.py'
        exec(compile(entry.read_text(encoding='utf-8'),str(entry),'exec'),globals())
        marker=stage.GetPrimAtPath(support['COMPONENT'])
        base_path=Path(marker.GetCustomDataByKey('H3B_readiness_preparation'))
        base=json.loads(base_path.read_bytes())
        if (base['mission_start_xyyaw_unchanged']!=info['mission_start_xyyaw']
                or base['mission_goal_xyyaw_unchanged']!=info['mission_goal_xyyaw']):
            raise RuntimeError('The original H3 mission start/goal changed')
        record={'condition':h['CONDITION'],'role':h['ROLE'],
            'fresh_prefix_reproduction_for_rear_reverse_review':True,
            'rear_reverse_execution_authorized':False,'original_first_PASS_and_reverse_FAIL_preserved':True,
            'actual_new_endpoint_requires_review_before_reverse':True,
            'status':'RESPONSE_LINK_FIRST_SPATIAL_LEG_PREPARED_STATIC_AND_PhysX_PENDING',
            'base_preparation_reference':str(base_path),'base_preparation_sha256':support['sha'](base_path),
            'model_profile':info['profile'],'model_profile_sha256':info['profile_sha256'],
            'input_sha256':info['input_sha256'],'mode':info['mode'],'duration_s':info['duration_s'],
            'command_limits':h['COMMAND_LIMITS'],'tracking':h['TRACKING'],
            'straight_S_absolute_30mm_2deg_corridor_not_applicable_to_curved_task':True,
            'existing_H1_reference_tracking_80mm_10deg_guard':True,
            'terminal_acceptance_remains_30mm_2deg':True,
            'component_fixed_model_stop_world_xyyaw':h['component_goal'](info['mission_start_xyyaw'],info),
            'first_stop_model_local_xyyaw':info['first_stop_model_local_xyyaw'],
            'mission_start_xyyaw_unchanged':info['mission_start_xyyaw'],
            'mission_goal_xyyaw_unchanged':info['mission_goal_xyyaw'],
            'component_target_is_NOT_full_mission_goal':True,'strict_map_digest':info['strict_map_digest'],
            'caster_state_assignment_or_success_angle_filter':False,
            'source_authored_initial_state_not_same_natural_entry_history':True,
            'existing_H1_spatial_feedback_used':False,'existing_H1_numeric_gains_and_state_machine_used':True,
            'shared_course_feedback_revision_used':True,'original_34mm_first_leg_FAIL_preserved':True,
            'future_comparison_requires_common_executor_NEW_cohort':True,'terminal_pulse_feedback_used':False,
            'tracking_gains':h['TRACKING_GAINS'],'reference_kind':info['reference_kind'],
            'frozen_response_used_OFFLINE_only_not_actuator_inversion':True,
            'legacy_model_fields_represent_NEW_nominal_spatial_cusp_not_model_trace':True,
            'synthetic_stress_checks_all_pass':info['synthetic_stress_checks_all_pass'],'reverse_execution_authorized':False,
            'prior_failures_preserved':True,'PhysX_executed':False,'full_H3B_turning_path_verified':False,
            'source_USD_saved':False,'use_for_final_statistics':False,
            'source_sha256':{p.name:support['sha'](p) for p in (
                entries/'isaac_prepare_ijcas_hospital_h3b_reverse_force_prefix.py',
                entries/'isaac_run_ijcas_hospital_h3b_reverse_force_prefix.py',
                scripts/'hospital_h3b_reverse_force_prefix.py',scripts/'hospital_h3b_course_tracking.py',
                scripts/'hospital_h3b_response_first_leg_pilot.py',scripts/'hospital_h3b_response_link.py',
                scripts/'hospital_h3b_first_leg_feedback_review.py',scripts/'hospital_path_execution.py',scripts/'hospital_private_loader.py')}}
        with target.open('x') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
        with Usd.EditContext(stage,session):
            marker.SetCustomDataByKey('H3B_reverse_force_prefix_preparation',str(target))
            marker.SetCustomDataByKey('H3B_reverse_force_prefix_preparation_sha256',support['sha'](target))
    except Exception as error:
        session.ImportFromString(before)
        with target.with_name(target.stem+'_failure.json').open('x') as stream:
            json.dump({'status':'FAILED_SESSION_ROLLED_BACK_NO_PLAY','error':str(error)},stream)
        raise
    print('=== H3-B FORCE-OBSERVATION IDENTICAL FIRST FORWARD PREFIX PREPARED ===')
    print('saved:',target)
    print('UNCHANGED course-feedback forward executor, H1 1.5/2 gains and fixed path/cusp. New cohort only; original PASS/FAIL preserved.')
    print('Nominal model motion duration (informational, not replay clock) s:',round(info['duration_s'],3),'mode:',info['mode'])
    print('No reverse/full H3 authorization. Fixed NEW nominal first-cusp target, not a predicted trace or recaptured actual goal.')
    print('Curved task: existing H1 reference tracking 80 mm / 10 deg; terminal 30 mm / 2 deg, strict envelope/contact/stop guards retained.')
    print('Timeline STOPPED. Next run isaac_run_ijcas_hospital_h3_flush_static_gate.py; require actual PASS and keep PAUSED.')
    print('Source-authored caster state retained. No CICS, motion, manual Play or source USD save.')


_prepare_rear_prefix_pilot()



print('FRESH PREFIX ONLY: prepare -> static PASS -> identical course-prefix replay -> PAUSE. No rear reverse or robot teleport.')
