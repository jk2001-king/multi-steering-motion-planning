"""Fresh source -> same H3 flush support, separate moderate-S component only."""


def _prepare_mode_speed_component():
    import json
    from pathlib import Path
    import runpy

    import omni.timeline
    import omni.usd
    from pxr import Usd

    if globals().pop('ijcas_hospital_h3b_mode_speed_condition', None) != 'H3B_mode_speed_component_v1':
        raise RuntimeError('Explicit H3B_mode_speed_component_v1 selection required')
    workspace = Path('/home/jk/Research/IJCAS_ws')
    scripts, entries = workspace/'studies/IJCAS/scripts', workspace/'isaac/scripts'
    load = runpy.run_path(str(scripts/'hospital_private_loader.py'))['load']
    h = load(scripts/'hospital_h3b_mode_speed_review.py')
    support = runpy.run_path(str(scripts/'hospital_h3_flush_support.py'))
    case = globals().pop('ijcas_hospital_h3b_mode_speed_case', h['FIRST_CASE'])
    mode, speed = h['require_case'](case)
    observation = load(scripts/'hospital_h3b_cp30_runtime_v2.py')
    protocol = observation['load_protocol']()
    if case != observation['CASE']:
        raise RuntimeError('This separate observation admits ONLY the frozen CP30 case')
    review_path = Path(globals().pop('ijcas_hospital_h3b_mode_speed_review', ''))
    review = json.loads(review_path.read_bytes())
    if review.get('version') != h['VERSION'] or (review_path.parent/'failure.json').exists():
        raise RuntimeError('Use a completed NEW offline mode/speed review, not an old/failed artifact')
    support['verify_files'](review['input_sha256'])
    support['verify_files']({scripts/k:v for k,v in review['source_sha256'].items()})
    selected = next(r for r in review['cases'] if r['case'] == case)
    if not all(selected['model_checks'].values()) or selected['PhysX_executed'] is not False:
        raise RuntimeError('Selected component did not pass offline admission, or is mislabeled physical evidence')
    support['verify_files']({selected['trace_file']:selected['trace_sha256']})
    stage = omni.usd.get_context().get_stage()
    if stage is None or not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError('Reopen ORIGINAL source STOPPED; do not continue the failed natural-entry scene')
    for name, task in list(globals().items()):
        if name.startswith('ijcas_hospital_') and name.endswith('_task') and hasattr(task,'done') and not task.done():
            raise RuntimeError('Another hospital trial is running')
    output = workspace/'studies/IJCAS/results/hospital_paper_gate/hospital_h3b_cp30_observation/cohort_01'/case
    if output.exists() and any(output.glob('*')):
        raise RuntimeError('One attempt per predeclared case/cohort; preserve its outcome and review before retry')
    folder = workspace/'studies/IJCAS/results/hospital_paper_gate/hospital_h3b_cp30_observation_preparation'
    folder.mkdir(parents=True,exist_ok=True)
    number=1
    while any((folder/f'prepare_r{number:02d}{s}').exists() for s in ('.json','_failure.json')): number+=1
    target = folder/f'prepare_r{number:02d}.json'
    session = stage.GetSessionLayer(); before = session.ExportToString()
    try:
        # Reuse ONLY the unchanged geometry/support wrapper, not its motion gate.
        globals()['ijcas_hospital_h3b_prepare_condition']='H3B_common_straight_reversal'
        entry=entries/'isaac_prepare_ijcas_hospital_h3b_reverse_readiness.py'
        exec(compile(entry.read_text(encoding='utf-8'),str(entry),'exec'),globals())
        marker=stage.GetPrimAtPath(support['COMPONENT'])
        base_path=Path(marker.GetCustomDataByKey('H3B_readiness_preparation'))
        base=json.loads(base_path.read_bytes())
        if (base['mission_start_xyyaw_unchanged']!=review['fixed_mission_start_xyyaw']
                or base['mission_goal_xyyaw_unchanged']!=review['fixed_mission_goal_xyyaw']):
            raise RuntimeError('The original H3 task changed')
        record={'version':h['VERSION'],'role':h['ROLE'],'case':case,'mode':mode,'speed_limit_m_s':speed,
            'status':'PREPARED_NEW_COMPONENT_STATIC_AND_MOTION_PENDING',
            'base_preparation_reference':str(base_path),'base_preparation_sha256':support['sha'](base_path),
            'offline_review':str(review_path),'offline_review_sha256':support['sha'](review_path),
            'input_sha256':{**review['input_sha256'],str(review_path):support['sha'](review_path),
                selected['trace_file']:selected['trace_sha256']},
            'fixed_progress_curvature_knots':h['KNOTS'],'command_limits':h['COMMAND_LIMITS'],
            'component_nominal_goal_world_xyyaw':h['component_goal'](review['fixed_mission_start_xyyaw'],mode),
            'mission_start_xyyaw_unchanged':review['fixed_mission_start_xyyaw'],
            'mission_goal_xyyaw_unchanged':review['fixed_mission_goal_xyyaw'],
            'strict_map_digest':review['strict_map_digest'],
            'caster_state_assignment_or_success_angle_filter':False,
            'source_authored_initial_state_not_same_natural_entry_history':True,
            'profile_and_mode_speed_changes_not_isolated_causal_test':True,
            'prior_failures_preserved':True,'PhysX_executed':False,'full_H3B_turning_path_verified':False,
            'source_USD_saved':False,'use_for_final_statistics':False,
            'same_condition_observation_protocol':str(observation['PROTOCOL_PATH']),
            'same_condition_observation_protocol_sha256':support['sha'](observation['PROTOCOL_PATH']),
            'separate_passive_observation_NOT_original_speed_case_or_H3_comparison':True,
            'source_sha256':{p.name:support['sha'](p) for p in (entries/'isaac_prepare_ijcas_hospital_h3b_cp30_observation_v2.py',
                entries/'isaac_run_ijcas_hospital_h3b_cp30_observation_primary_v2.py',scripts/'hospital_h3b_mode_speed_review.py',
                scripts/'hospital_private_loader.py')}}
        with target.open('x') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
        with Usd.EditContext(stage,session):
            marker.SetCustomDataByKey('H3B_mode_speed_preparation',str(target))
            marker.SetCustomDataByKey('H3B_mode_speed_preparation_sha256',support['sha'](target))
    except Exception as error:
        session.ImportFromString(before)
        with target.with_name(target.stem+'_failure.json').open('x') as stream:
            json.dump({'status':'FAILED_SESSION_ROLLED_BACK_NO_PLAY','error':str(error)},stream)
        raise
    print('=== H3-B NEW MODERATE-S MODE/SPEED COMPONENT PREPARED ===')
    print('case:',case,'saved:',target)
    print('Unchanged H3 task/start, flush map, contact/motor settings and safety thresholds; source-authored caster state retained.')
    print('NEW 0.65 m / +/-0.25 curvature forward component, NOT the old S, natural-entry continuation or saved route.')
    print('Timeline STOPPED. Next: isaac_run_ijcas_hospital_h3_flush_static_gate.py; require PASS and keep PAUSED.')
    print('No CICS, caster reset, motion or full H3 validation yet. Do not save or manually Play.')


_prepare_mode_speed_component()
