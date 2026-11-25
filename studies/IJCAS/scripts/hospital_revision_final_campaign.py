"""Approved n3 matched block, using shared planners and guarded SDK hosts.

No physical loop, model, collision predicate, or success criterion is copied.
Scientific design and the legacy host's all-cell adapter protocol are separate.
Every declared negative stays in its denominator; no in-block retuning/retry.
"""
import argparse
import asyncio
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import hospital_revision_final_design as design
import hospital_revision_joint_followup as predecessor
import hospital_revision_live as live
import hospital_revision_metrics as metrics
import hospital_revision_beam_goal_rank as search
import hospital_revision_seed_order_review as seed_order
import hospital_revision_envelope as envelope
import hospital_revision_transition_envelope as temporal

runtime = design.original.runtime
control = design.control
planning = live.planning
initial = design.initial
joints = predecessor.joints
OUTPUT = design.original.GATE/'revision_final_n3_20261122'
PROTOCOL = OUTPUT/'protocol.json'
EXECUTION_PROTOCOL = OUTPUT/'execution_protocol.json'
AUTHORITY = '2026-11-22 사용자: 일단 3조건으로 쭉 진행해보자 그러면 쭉 가자'


def execution_cells(value):
    ablated = {c['parent_main_cell_id']: c for c in value['history_ablation_cells']}
    result = []
    for cell in value['main_cells']:
        result.append(dict(copy.deepcopy(cell), experiment_role='MAIN', predictive_history_policy='carry'))
        if cell['cell_id'] in ablated:
            result.append(dict(copy.deepcopy(ablated[cell['cell_id']]),
                experiment_role='HISTORY_ABLATION', predictive_history_policy='steady_both'))
    return result


def implementation_sources():
    files = {Path(p) for p in runtime.read(predecessor.PROTOCOL)['source_sha256']}
    files.update(Path(p) for p in runtime.lc.host_source_files())
    files.update(Path(m.__file__) for m in (design, live, planning, metrics, search,
        search.terminal, search.parent, seed_order, seed_order.base, predecessor,
        predecessor.artifacts, envelope, temporal, predecessor.joints.initial,
        predecessor.joints.serialization, predecessor.joints.serialization.follow,
        predecessor.joints.serialization.placement, predecessor.joints.serialization.placement.live))
    files.update((Path(__file__), Path(planning.mission.__file__),
        Path(predecessor.control.base.shared.__file__), Path(predecessor.control.base.__file__),
        Path(seed_order.base.followup.__file__), Path(seed_order.base.followup.mechanisms.__file__)))
    files.update(design.original.ROOT/'src/patient_initial_stability/patient_initial_stability'/name
        for name in ('steering_lattice.py','hierarchical_dynamics_planner.py','rigid_body_edge.py'))
    files.update(Path(m.__file__) for m in tuple(sys.modules.values())
        if getattr(m,'__file__',None) and Path(m.__file__).is_file()
        and Path(m.__file__).resolve().is_relative_to(design.original.ROOT)
        and Path(m.__file__).suffix=='.py')
    return sorted({p.resolve() for p in files})


def definition():
    value = design.definition(3)
    value.update(version='APPROVED_FINAL_N3_COMMON_V7_UNIFORM_NATIVE_P_v1',
        status='APPROVED_PREDECLARED_NOT_COMPLETED_RESULTS', user_authority=AUTHORITY,
        automatic_dispatch=True, counts_NOT_approved_or_guaranteed_drive_counts=False,
        cell_counts_are_declared_attempts_NOT_guaranteed_drives_or_successes=True,
        all_main_and_ablation_attempts=78, old_results_never_replaced=True,
        source_static_policy='original held support screen; V7 transient hold in same mandatory route prehold',
        route_order='H1 through H6; rotated four-method order within each paired draw; ablation follows its P control',
        planning_timing_policy='one offline worker; entire offline block while admitted live workflow is complete and PAUSED; no parallel planner/model jobs',
        no_other_heavy_jobs_by_this_runner=True,
        common_controller_contract=control.contract(value['tasks']['H6']['geometry_profile'],
            simulation_budget_s=600.,wall_budget_s=1500.),
        prior_development_block=str(predecessor.OUTPUT),
        prior_nine_P_development_PASS_required_before_live=True,
        clean_restored_execution_noncompletion_continues_for_ALL_methods_including_P=True,
        technical_or_unrestored_boundary_stops_without_automatic_repair=True,
        model_predicted_HOLD_is_diagnostic_NOT_changed_to_PASS=True,
        initial_joint_variation_is_source_sensitivity_actual_bound_and_route_entry_angles_recorded=True)
    return runtime.evidence.json_value(value)


def export_protocol():
    value = definition()
    adapter = dict(copy.deepcopy(value), main_comparison_cells=copy.deepcopy(value['main_cells']),
        main_cells=execution_cells(value),
        legacy_main_cells_key_is_ALL_78_execution_cells_NOT_78_main_comparisons=True,
        scientific_protocol=str(PROTOCOL))
    for path, raw in ((PROTOCOL,value),(EXECUTION_PROTOCOL,adapter)):
        if path.exists():
            if runtime.read(path) != raw:
                raise ValueError('Final predeclared definition cannot change after export')
        else:
            runtime.evidence.write_exclusive(path,raw)
    pins = {str(p):runtime.sha(p) for p in implementation_sources()}
    target=OUTPUT/'sources.json'
    if target.exists():
        if runtime.read(target)!=pins: raise ValueError('Final implementation is frozen')
    else:
        runtime.evidence.write_exclusive(target,pins)
        runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',pins)
    return PROTOCOL


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'sources.json'))
    runtime.retained.runtime.task.verify_pins(runtime.read(PROTOCOL)['original_evidence_inputs_sha256'])


def campaign_namespace():
    return SimpleNamespace(**dict(vars(design.original), OUTPUT=OUTPUT, PROTOCOL=EXECUTION_PROTOCOL))


def folder_for(name):
    if name not in {c['cell_id'] for c in runtime.read(EXECUTION_PROTOCOL)['main_cells']}:
        raise ValueError('Exactly one approved final cell required')
    return OUTPUT/'planning'/name


def source_binding(cell):
    return runtime.bind(initial.cell_bindings, OUTPUT=OUTPUT, PROTOCOL=EXECUTION_PROTOCOL,
        private_source_context=joints.serialization.follow.initial_context,
        InitialConditionStaticHost=joints.InitializedJointStaticHost,
        __file__=__file__)(cell)


def build_assets():
    export_protocol(); verify_sources()
    for cell in runtime.read(PROTOCOL)['main_cells']:
        source_binding(cell)
    paths=sorted((OUTPUT/'source_conditions').glob('*.json'))
    if len(paths)!=18: raise ValueError('Exactly eighteen paired combined source conditions required')
    value=dict(conditions=18,shared_by_four_methods_and_corresponding_ablation=True,
        source_receipts_sha256={str(p):runtime.sha(p) for p in paths}, actual_motion_trials=0)
    target=OUTPUT/'source_assets.json'
    if target.exists():
        if runtime.read(target)!=value: raise ValueError('Final source overlays changed')
    else: runtime.evidence.write_exclusive(target,value)
    return value


def seed_binding():
    # Exact B1(10s), B0(20s), B2(remainder) schedule from the validated candidate.
    return runtime.bind(seed_order.labelled_seed, PROTOCOL=EXECUTION_PROTOCOL)


def plan_cell(name, *, envelope_variant=None):
    verify_sources()
    cell=next(c for c in runtime.read(EXECUTION_PROTOCOL)['main_cells'] if c['cell_id']==name)
    policy=cell['predictive_history_policy']
    if envelope_variant is not None and (cell['experiment_role']!='MAIN'
            or cell['method']!=design.original.planning.P or cell['task'] not in ('H2','H3')):
        raise ValueError('Only the six declared full-history envelope sensitivity inputs')
    output=OUTPUT/'planning' if envelope_variant is None else OUTPUT/'offline_envelope'/envelope_variant
    base=seed_order.base.followup.planner_bindings(policy)
    def bindings(*a,**k):
        method,audit,connections,parameters,activity=base(*a,**k)
        if envelope_variant:
            checker=method.__globals__['HospitalEnvelopeChecker']
            cls=envelope.point_checker(checker) if envelope_variant=='point' else temporal.endpoint_checker(checker)
            method=runtime.bind(method,HospitalEnvelopeChecker=cls)
            parameters=dict(parameters,offline_envelope_variant=envelope_variant,
                independently_checked_full_sampled_rectangle_unchanged=True)
        return method,audit,connections,parameters,activity
    original_write=runtime.evidence.write_exclusive
    def labelled_write(path,raw):
        if Path(path).name in ('definition.json',cell['method']+'.json'):
            raw=dict(raw,legacy_zero_reset_entry_history_ablation=raw.get('entry_history_ablation',False),
                entry_history_ablation=policy!='carry',predictive_history_policy=policy,
                experiment_role=cell['experiment_role'], final_scientific_protocol=str(PROTOCOL),
                final_controller_version=control.VERSION)
        return original_write(path,raw)
    private_runtime=SimpleNamespace(**dict(vars(runtime),evidence=SimpleNamespace(
        **dict(vars(runtime.evidence),write_exclusive=labelled_write))))
    shared=SimpleNamespace(**dict(vars(planning.shared),unified_P_owned_seed=seed_binding()))
    call=runtime.bind(planning.run_cell,campaign=campaign_namespace(),shared=shared,
        runtime=private_runtime, OUTPUT=output,planner_bindings=bindings,
        guided_binding=search.guided_binding,__file__=__file__)
    result=call(name,lookahead_edges=4)
    audit=runtime.read(output/name/'independent_plan_audit.json') if result['plan_success'] else None
    result=dict(result,predictive_history_policy=policy,experiment_role=cell['experiment_role'],
        final_scientific_protocol=str(PROTOCOL),new_PhysX_trials=0,
        independently_valid_reference=bool(audit and audit['verdict']=='PASS'))
    if envelope_variant:
        result.update(offline_envelope_variant=envelope_variant,
            full_envelope_colliding_samples=audit.get('colliding_samples') if audit else None,
            unsafe_paths_NOT_physically_executed=True)
    runtime.evidence.write_exclusive(output/name/'final_planning_receipt.json',result)
    return result


def model_cell(name):
    folder=folder_for(name)
    result=runtime.read(folder/'final_planning_receipt.json')
    if not result['independently_valid_reference']: return None
    shared=predecessor.shared
    namespace=SimpleNamespace(**dict(vars(shared),planning_namespace=lambda:SimpleNamespace(
        **dict(vars(planning),OUTPUT=shared.ReferenceFolders({name:folder})))))
    scope=dict(control=control,OUTPUT=OUTPUT,PROTOCOL=EXECUTION_PROTOCOL,
        PLAN_FOLDERS={name:folder},__file__=__file__)
    binding=runtime.bind(shared.model_binding,**scope)
    call=runtime.bind(shared.model_screen,**dict(scope,
        planning_namespace=namespace.planning_namespace,model_binding=binding,
        export_protocol=lambda:EXECUTION_PROTOCOL,verify_sources=verify_sources))
    value=call(name)
    predecessor.artifacts.publish(folder,OUTPUT/'model'/name/'model_screen.json',controller_version=control.VERSION)
    return value


def predecessor_ready():
    path=predecessor.OUTPUT/'completed.json'
    if not path.exists(): return False
    value=runtime.read(path)
    ps=[r for r in value['results'] if '_P_' in r['cell_id']]
    return (len(ps)==9 and all(r['actual_verdict']=='PASS' for r in ps)
        and predecessor.control.base.shared.execution.restored_boundary(runtime.read(value['last_actual_report'])))


def offline():
    export_protocol(); verify_sources()
    if not predecessor_ready(): raise ValueError('Prior actual bounded block must finish safely first')
    build_assets()
    runtime.evidence.write_exclusive(OUTPUT/'offline_reservation.json',dict(
        main_and_history_cells=78,offline_envelope_cells=12,one_planning_attempt_each=True,
        no_parallel_planners_or_live_motion=True,actual_motion_trials=0))
    rows=[]
    for cell in runtime.read(EXECUTION_PROTOCOL)['main_cells']:
        name=cell['cell_id'];value=plan_cell(name);model=model_cell(name)
        row=dict(value,model_report=str(OUTPUT/'model'/name/'model_screen.json') if model else None,
            both_nominal_resolutions_pass=model['both_nominal_resolutions_pass'] if model else None)
        runtime.evidence.write_exclusive(OUTPUT/'offline_outcomes'/(name+'.json'),row);rows.append(row)
        print('[FINAL N3 OFFLINE]',len(rows),'/78',name,value['plan_success'],flush=True)
    extra=[]
    for cell in runtime.read(PROTOCOL)['main_cells']:
        if cell['method']==design.original.planning.P and cell['task'] in ('H2','H3'):
            for variant in ('point','endpoints'):
                extra.append(plan_cell(cell['cell_id'],envelope_variant=variant))
    value=dict(status='FINAL_N3_OFFLINE_FINISHED',rows=rows,offline_envelope=extra,
        model_HOLD_is_diagnostic_for_ALL=True,actual_motion_trials=0)
    runtime.evidence.write_exclusive(OUTPUT/'offline_completed.json',value)
    return value


def policy_file():
    value=dict(version='FINAL_N3_COMMON_DIAGNOSTIC_MODELS_v1',
        user_authority='뭐가 가장 베스트의 방향성인지 판단하고 진행해',
        current_explicit_n3_authority=AUTHORITY,protocol_sha256=runtime.sha(EXECUTION_PROTOCOL),
        model_is_prediction_not_actual_failure=True,independent_full_envelope_audit_required=True,
        actual_static_PASS_required=True,unchanged_actual_controller_limits_and_success_criteria=True,
        applies_to_all_methods_and_history_ablation=True,no_physical_retry=True)
    path=OUTPUT/'diagnostic_model_policy.json'
    if path.exists():
        if runtime.read(path)!=value: raise ValueError('Final admission policy frozen')
    else: runtime.evidence.write_exclusive(path,value)
    return path


def bound_cell(cell):
    common=control.validation_cell()
    old=common.__globals__['runtime'];prepared=source_binding(cell).__globals__['runtime']
    def context(folder,actor,geometry):
        return prepared.context_for(folder,actor+'_final_n3_v7',geometry)
    private=SimpleNamespace(**dict(vars(old),context_for=context,
        prepare_source=prepared.prepare_source,ControlledStaticHost=joints.InitializedJointStaticHost))
    parent=common.__globals__['PrimaryRouteHost']
    class FinalHost(parent):
        subscribe=joints.route_subscriber()
        check_post_observation_layers=joints.InitializedJointRouteHost.check_post_observation_layers
        def save(self,report,rows):
            report=dict(report,use_for_final_statistics=True,development_trial_NOT_final_repetition=False,
                final_fixed_version_trial=True,controller_candidate_version=control.VERSION,
                predictive_history_policy=cell['predictive_history_policy'],
                final_scientific_protocol=str(PROTOCOL),final_experiment_role=cell['experiment_role'])
            return live.PrimaryRouteHost.save(self,report,rows)
    packet0=common.__globals__['task_route_packet']
    def packet(*a,**k):
        path,value,plan=packet0(*a,**k);value=copy.deepcopy(value)
        value['input_sha256'].update(runtime.read(OUTPUT/'sources.json'))
        value['input_sha256'].update({str(p):runtime.sha(p) for p in (PROTOCOL,EXECUTION_PROTOCOL)})
        return path,value,plan
    return runtime.bind(common,runtime=private,campaign=campaign_namespace(),
        planning=SimpleNamespace(**dict(vars(planning),OUTPUT=OUTPUT/'planning')),
        checked_model_policy=runtime.bind(live.checked_model_policy,campaign=campaign_namespace()),
        PrimaryRouteHost=FinalHost,task_route_packet=packet,__file__=__file__)


def safe_restored_boundary(raw):
    if not predecessor.control.base.shared.execution.restored_boundary(raw): return False
    checks=raw.get('checks',{})
    return all(checks.get(k) is True for k in (
        'all4997_environment_rows_and145_source_settings_and_physics_unchanged',
        'original_all12joint_gains_targets_and_limits_restored','all_frozen_inputs_unchanged',
        'no_unreviewed_USD_edits_or_callback_errors','contacts_healthy_through_actual_Pause'))


async def run(parent,*,authorized=False):
    if authorized is not True: raise ValueError('Explicit user-approved final n3 authority required')
    import builtins
    previous=await asyncio.shield(parent)
    if getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not parent:
        raise ValueError('Exact latest owned predecessor required')
    if not predecessor_ready() or not (OUTPUT/'offline_completed.json').exists():
        raise ValueError('Completed predecessor and complete frozen offline block required')
    verify_sources();last=Path(previous['last_actual_report'])
    if not safe_restored_boundary(runtime.read(last)): raise ValueError('Complete actual boundary required')
    runtime.evidence.write_exclusive(OUTPUT/'execution_reservation.json',dict(
        authority=AUTHORITY,maximum_conditional_cells=78,prior_result=str(last),
        protocol_sha256=runtime.sha(PROTOCOL),automatic_retry=False))
    task=asyncio.current_task();builtins.ijcas_hospital_latest_task_experiment=task
    builtins.ijcas_revision_primary_task=task
    results=[]
    try:
        for cell in runtime.read(EXECUTION_PROTOCOL)['main_cells']:
            verify_sources();name=cell['cell_id'];folder=folder_for(name)
            planned=runtime.read(folder/'final_planning_receipt.json')
            if not planned['independently_valid_reference']:
                outcome=dict(cell_id=name,status='PLANNING_FAILURE_NO_PHYSICAL_RUN' if not planned['plan_success']
                    else 'INDEPENDENT_REFERENCE_AUDIT_REJECTED',actual_route_started=False,
                    actual_verdict='PLAN_FAILURE' if not planned['plan_success'] else 'REFERENCE_REJECTED',
                    planning_summary=str(folder/'summary.json'),metrics=None)
            else:
                print('[FINAL N3 ACTUAL START]',name,flush=True)
                outcome=await bound_cell(cell)(name,prior_result=last,authorized=True,model_policy_path=policy_file())
                last=Path(outcome['report']);raw=runtime.read(last)
                if not safe_restored_boundary(raw):
                    raise ValueError('Technical/actual boundary unresolved; no automatic recovery or next cell')
                outcome=dict(outcome,metrics=metrics.metrics(last) if outcome['actual_route_started'] else None)
            outcome.update(experiment_role=cell['experiment_role'],controller_version=control.VERSION,
                predictive_history_policy=cell['predictive_history_policy'],final_fixed_version_trial=True,
                attempt_replaced=False)
            runtime.evidence.write_exclusive(OUTPUT/'outcomes'/(name+'.json'),outcome);results.append(outcome)
            print('[FINAL N3 ACTUAL RESULT]',name,outcome['actual_verdict'],flush=True)
            await asyncio.sleep(0)
        value=dict(status='FINAL_N3_ALL_DECLARED_CELLS_FINISHED',results=results,
            last_actual_report=str(last),all_negative_results_preserved=True,automatic_retry=False)
        runtime.evidence.write_exclusive(OUTPUT/'completed.json',value)
        return value
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT/'stopped.json',dict(error=type(error).__name__+': '+str(error),
            results=results,last_actual_report=str(last),automatic_retry=False))
        raise


def dispatch():
    verify_sources()
    if not (OUTPUT/'offline_completed.json').exists(): raise ValueError('Complete offline block required')
    regression=runtime.read(OUTPUT/'regression.json')
    if regression['exit_status'] or regression['failed'] or regression['errors'] or regression['skipped']:
        raise ValueError('Complete current-source regression must pass before actual dispatch')
    runtime.retained.runtime.task.verify_pins(regression['implementation_sources_sha256'])
    source=("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "_n3_parent=getattr(builtins,'ijcas_revision_targeted_joint_v7',None)\n"
        "assert _n3_parent is not None and _n3_parent.done() and not _n3_parent.exception(),'Exact completed predecessor required'\n"
        "assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is _n3_parent,'Intervening motion'\n"
        "assert not hasattr(builtins,'ijcas_revision_final_n3'),'Final block already reserved'\n"
        f"_n3_path=Path({str(Path(__file__).resolve())!r})\n"
        "_n3_module=types.ModuleType('hospital_revision_final_campaign');_n3_module.__file__=str(_n3_path)\n"
        "sys.modules[_n3_module.__name__]=_n3_module\n"
        "exec(compile(_n3_path.read_bytes(),str(_n3_path),'exec'),_n3_module.__dict__)\n"
        "_n3_job=asyncio.ensure_future(_n3_module.run(_n3_parent,authorized=True))\n"
        "builtins.ijcas_revision_final_n3=_n3_job\n"
        "print({'final_n3_main72_ablation6_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch.json',dict(source=source,
        protocol_sha256=runtime.sha(PROTOCOL),authority=AUTHORITY,automatic_retry=False))
    return predecessor.control.base.shared.runner.remote(source)


def describe(values):
    values=[float(v) for v in values if v is not None]
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
        sample_SD=statistics.stdev(values) if len(values)>1 else None,ddof=1)


def results_snapshot():
    protocol=runtime.read(PROTOCOL);rows=[]
    for cell in runtime.read(EXECUTION_PROTOCOL)['main_cells']:
        name=cell['cell_id'];planned_path=folder_for(name)/'summary.json'
        planned=runtime.read(planned_path) if planned_path.exists() else {}
        outcome_path=OUTPUT/'outcomes'/(name+'.json')
        outcome=runtime.read(outcome_path) if outcome_path.exists() else {}
        measured=outcome.get('metrics') or {}
        report=runtime.read(outcome['report']) if outcome.get('report') else {}
        completed=bool(outcome.get('actual_verdict')=='PASS' and outcome.get('actual_route_started')
            and measured.get('stationary_hold_and_all_final_checks_satisfied'))
        rows.append(dict(cell_id=name,task=cell['task'],method=cell['method'],
            experiment_role=cell['experiment_role'],paired_condition_id=cell['paired_condition_id'],
            predictive_history_policy=cell['predictive_history_policy'],
            accounted=bool(outcome),outcome=outcome.get('actual_verdict','PENDING'),
            actual_route_started=bool(outcome.get('actual_route_started')),completed=completed,
            plan_success=planned.get('plan_success'),planning_time_s=planned.get('planning_time_s'),
            tracking_position_RMS_mm=measured.get('tracking_position_RMS_mm'),
            tracking_yaw_RMS_deg=measured.get('tracking_yaw_RMS_deg'),
            travel_sim_s=measured.get('controller_elapsed_simulation_s'),
            actual_distance_m=measured.get('actual_distance_m'),
            final_position_error_mm=measured.get('final_position_error_mm'),
            final_yaw_error_deg=measured.get('final_yaw_error_deg'),
            actual_error=report.get('error'),failed_checks=[k for k,v in report.get('checks',{}).items() if not v],
            partial_metrics_NOT_ranked_as_completed_missions=not completed,
            report=outcome.get('report'),trace=measured.get('trace'),
            outcome_path=str(outcome_path) if outcome else None,planning_summary=str(planned_path)))
    groups=[]
    for task in protocol['tasks']:
        for role,method in [('MAIN',m) for m in design.original.METHODS]+(
                [('HISTORY_ABLATION',design.original.planning.P)] if task in ('H2','H3') else []):
            subset=[r for r in rows if (r['task'],r['experiment_role'],r['method'])==(task,role,method)]
            passed=[r for r in subset if r['completed']]
            groups.append(dict(task=task,experiment_role=role,method=method,declared_n=len(subset),
                accounted_n=sum(r['accounted'] for r in subset),completed_n=len(passed),
                completion_fraction_all_declared=len(passed)/len(subset),
                planning_time_all_declared_attempts_s=describe([r['planning_time_s'] for r in subset]),
                completed_tracking_position_RMS_mm=describe([r['tracking_position_RMS_mm'] for r in passed]),
                completed_tracking_yaw_RMS_deg=describe([r['tracking_yaw_RMS_deg'] for r in passed]),
                completed_travel_sim_s=describe([r['travel_sim_s'] for r in passed])))
    return dict(protocol=str(PROTOCOL),controller_version=control.VERSION,rows=rows,groups=groups,
        accounted_cells=sum(r['accounted'] for r in rows),declared_cells=78,
        status='COMPLETE' if all(r['accounted'] for r in rows) else 'IN_PROGRESS',
        no_old_development_successes_pooled=True,completion_denominator_includes_all_declared_failures=True,
        completed_RMS_conditional_on_success_with_explicit_n=True,
        not_a_universal_P_success_or_statistical_significance_claim=True)


def export_results():
    value=results_snapshot()
    if value['status']!='COMPLETE': raise ValueError('Not all seventy-eight cells accounted; no final result pack')
    folder=OUTPUT/'analysis'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    runtime.evidence.write_exclusive(folder/'results.json',value)
    for filename,rows in (('individual.csv',value['rows']),('summary.csv',value['groups'])):
        flattened=[{k:json_value(v) for k,v in row.items()} for row in rows]
        with (folder/filename).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    return folder


def json_value(value):
    import json
    return json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value


def regression():
    previous=runtime.read(sorted((predecessor.OUTPUT.parent/'regression_current_v7_joint_reporting').glob('*/regression.json'))[-1])
    command=previous['command'][:-1]+[
        'src/patient_initial_stability/test/test_hospital_revision_final_campaign.py',
        '--junitxml='+str(OUTPUT/'regression_junit.xml')]
    begun=datetime.now(timezone.utc).isoformat()
    process=subprocess.run(command,text=True,capture_output=True,cwd=design.original.ROOT,timeout=180.)
    xml=ET.parse(OUTPUT/'regression_junit.xml').getroot().findall('testsuite')
    totals={k:sum(int(s.get(k,'0')) for s in xml) for k in ('tests','failures','errors','skipped')}
    value=dict(command=command,started_UTC=begun,exit_status=process.returncode,tests=totals['tests'],
        failed=totals['failures'],errors=totals['errors'],skipped=totals['skipped'],
        stdout=process.stdout,stderr=process.stderr,
        implementation_sources_sha256=runtime.read(OUTPUT/'sources.json'),
        unit_tests_NOT_physical_success_proof=True)
    runtime.evidence.write_exclusive(OUTPUT/'regression.json',value)
    if process.returncode or any(totals[k] for k in ('failures','errors','skipped')):
        raise ValueError('Frozen current-source regression not fully successful')
    return value


def current_get(*,final=False):
    key='ijcas_revision_final_n3' if final else 'ijcas_revision_targeted_joint_v7'
    source=("import builtins,json,omni.timeline\n"
        f"_n3_get_job=getattr(builtins,{key!r},None)\n"
        "_n3_get_timeline=omni.timeline.get_timeline_interface()\n"
        "print(json.dumps({'job_exists':_n3_get_job is not None,"
        "'latest_owned_job_done':bool(_n3_get_job and _n3_get_job.done()),"
        "'job_is_latest':getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is _n3_get_job,"
        "'current_job_error':str(_n3_get_job.exception()) if _n3_get_job and _n3_get_job.done() and _n3_get_job.exception() else None,"
        "'playing':_n3_get_timeline.is_playing(),'stopped':_n3_get_timeline.is_stopped(),"
        "'current_time_s':_n3_get_timeline.get_current_time()}))\n")
    import json
    result=json.loads(predecessor.control.base.shared.runner.remote(source)['output'])
    if (not result['job_exists'] or not result['latest_owned_job_done'] or not result['job_is_latest']
            or result['current_job_error'] is not None or result['playing'] or result['stopped']):
        raise ValueError('Exact completed actual PAUSED workflow required: '+str(result))
    return result


def all_steps():
    """One external offline worker; actual motion remains in one owned Kit task."""
    export_protocol();build_assets();regression()
    print('[FINAL N3] Sources, eighteen shared initial conditions and regression frozen.',flush=True)
    while not predecessor_ready():
        if (predecessor.OUTPUT/'stopped.json').exists() or (predecessor.OUTPUT/'workflow_error.json').exists():
            raise ValueError('Prior bounded development block stopped; no automatic final actuation')
        print('[FINAL N3] Waiting for the existing bounded workflow; no concurrent actuation.',flush=True)
        time.sleep(30.)
    before=current_get()
    last=runtime.read(predecessor.OUTPUT/'completed.json')['last_actual_report']
    if abs(before['current_time_s']-runtime.read(last)['actual_final_timeline']['current_time_s'])>1e-6:
        raise ValueError('Current Pause differs from actual final predecessor snapshot')
    runtime.evidence.write_exclusive(OUTPUT/'before_offline_live_GET.json',before)
    offline();print(dispatch(),flush=True)
    previous_n=-1
    while not (OUTPUT/'completed.json').exists():
        if (OUTPUT/'stopped.json').exists():
            raise ValueError('Final block stopped; retain all outcomes and inspect without automatic retry')
        value=results_snapshot()
        if value['accounted_cells']!=previous_n:
            previous_n=value['accounted_cells']
            print('[FINAL N3 PROGRESS]',previous_n,'/78',flush=True)
            for group in value['groups']:
                if group['accounted_n']==group['declared_n']:
                    print('[FINAL N3 GROUP]',group,flush=True)
        time.sleep(30.)
    result=runtime.read(OUTPUT/'completed.json');time.sleep(.5)
    final=current_get(final=True)
    if abs(final['current_time_s']-runtime.read(result['last_actual_report'])['actual_final_timeline']['current_time_s'])>1e-6:
        raise ValueError('Final actual timeline differs from last saved report')
    final['final_report_sha256']=runtime.sha(result['last_actual_report'])
    runtime.evidence.write_exclusive(OUTPUT/'final_live_GET.json',final)
    folder=export_results()
    runtime.evidence.write_exclusive(OUTPUT/'all_steps_completed.json',dict(
        status='FINAL_N3_EXECUTION_AND_REPORTING_FINISHED',analysis_folder=str(folder),
        final_live_GET=str(OUTPUT/'final_live_GET.json'),all_failures_preserved=True))
    print('[FINAL N3 FINISHED]',folder,flush=True)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('definition','assets','offline','dispatch','all','results'))
    args=parser.parse_args()
    print(export_protocol() if args.action=='definition' else build_assets() if args.action=='assets'
        else offline() if args.action=='offline' else all_steps() if args.action=='all'
        else export_results() if args.action=='results' else dispatch(),flush=True)
