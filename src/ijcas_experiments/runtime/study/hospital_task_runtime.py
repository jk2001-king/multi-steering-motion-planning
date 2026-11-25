"""Shared H5 single-workflow adapters; no implicit live trial.

Reuse the exact original zero-command observation, force/contact checks,
clock handling and confirmed Pause. Only task identities and the H5 all-hit
query contract differ. The stable entry requires separate workflow consent.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import builtins
import ast
import asyncio
from collections import Counter,deque
import copy
import csv
import dis
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from types import CodeType,FunctionType,SimpleNamespace
import zipfile

import numpy as np

import hospital_task_experiment as task
from hospital_drive_hold import drive_hold_gain_arrays
from hospital_geometry_tools import pose_collision
from hospital_stop_analysis import analyze_stop_trace


VERSION='H5_ORIGINAL_STATIC_RUNTIME_WITH_EXPLICIT_TASK_QUERIES_v1'
WORKFLOW_VERSION='H5_ONE_ARCHIVED_SOURCE_STATIC_CONDITIONAL_COMMON_V3_ROUTE_v1'
ROUTE_FUNCTIONS=('numeric','_h4_path_step','_h4_path_contacts','_h4_path_angles','compact_environment_readback','_run_h4_path')
ENTRY=_ijcas_path(task.ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_task_experiment.py')
CAMERA_HELPER=_ijcas_path(task.ROOT, 'studies/IJCAS/scripts/hospital_h4_camera_view.py')


def finite_hit(hit):
    result=dict(collision=str(hit.collision),position_m=list(map(float,hit.position)),
        normal=list(map(float,hit.normal)),distance_m=float(hit.distance))
    if (len(result['position_m'])!=3 or len(result['normal'])!=3
            or not all(math.isfinite(v) for v in (*result['position_m'],*result['normal'],result['distance_m']))):
        raise ValueError('Malformed/nonfinite scene-query hit; no fabricated finite value')
    return result


def invalid_hit_diagnostic(hit):
    """Keep invalid native evidence as text; never emit NaN into strict JSON."""
    result={}
    for name in ('collision','position','normal','distance'):
        try:result[name+'_repr']=repr(getattr(hit,name))
        except Exception as error:result[name+'_read_error']=type(error).__name__+': '+str(error)
    return result


def load_preparation_design(path):
    """No live GETs; only exact own proposal, archived code and anonymous USD."""
    path=Path(path).resolve();p=json.loads(path.read_bytes());context=task.task_context(p)
    if path!=context.protocol_path or p!=task.build_protocol(p['frozen_bundle'],p['nominal_model_screen'],
            p['required_prior_result'],cohort=p['cohort'],method=p['method']):
        raise ValueError('Exact completed own H5 source preparation definition required')
    archive=Path(p['runtime_source_archive'])
    if task.sha(archive)!=p['runtime_source_archive_sha256'] or archive.read_bytes()!=task.runtime_source_archive():
        raise ValueError('The exact preparation helper version must remain archived and active')
    source_path=context.output/'anonymous_source_readback_v2.json';source=json.loads(source_path.read_bytes())
    checks=task.source_reference_checks(p,source['anonymous_source_without_SDK_namespace'],source['anonymous_source_with_unvalued_SDK_namespace'])
    if (source['version']!='H5_ANONYMOUS_AUTHORED_FIELDS_AND_COMPOSED_CONFIGURATION_REVIEW_v2'
            or source['task_protocol']!=str(path) or source['task_protocol_sha256']!=task.sha(path)
            or source['checks']!=checks or not all(checks.values())
            or source['status']!='ANONYMOUS_USD_SOURCE_CHECKS_COMPLETE_NOT_LIVE_STAGING_STATIC_OR_DRIVE_ADMISSION'
            or source['live_stage_robot_or_timeline_accessed'] is not False
            or source['actual_motion_authorized'] is not False or source['new_PhysX_trial_count']!=0):
        raise ValueError('Full authored-field/composed-value anonymous source review required; not the original HOLD')
    task.verify_pins(source['input_sha256'])
    return p,source,source_path


def static_runtime_definition(preparation_path):
    """Prospective runtime configuration; does not reserve or execute anything."""
    p,source,source_path=load_preparation_design(preparation_path)
    design_path=task.shared.STATIC_DESIGN
    original=json.loads(design_path.read_bytes());task.verify_pins(original['input_sha256'])
    if len(original['additional_disabled_subfloor_probes'])!=4:
        raise ValueError('All four original disabled-identity diagnostic queries must remain')
    context=task.task_context(p);result=copy.deepcopy(p)
    result.update(static_runtime_version=VERSION,preparation_definition=str(Path(preparation_path).resolve()),
        anonymous_source_review=str(source_path),
        source_actor_node_paths=list(source['anonymous_source_with_unvalued_SDK_namespace']['fixed_configuration_rows']),
        additional_disabled_subfloor_probes=original['additional_disabled_subfloor_probes'],
        auxiliary_probe_known_nonself_colliders=original['auxiliary_probe_known_nonself_colliders'],
        unchanged_original_static_limits=dict(task.shared.stationary.LIMITS),
        unchanged_original_static_guards=dict(task.shared.stationary.GUARDS),
        exact_original_zero_command_static_function_code_reused=True,
        original_four_absence_probes_remain_environment_diagnostics_NOT_H5_drive_validation=True,
        static_input_source_archive_preserved=True,actual_motion_authorized=False,
        status='OFFLINE_H5_STATIC_RUNTIME_BINDING_NOT_A_LIVE_ENTRY_OR_WORKFLOW_AUTHORIZATION',
        input_sha256=task.merge_pins(p['input_sha256'],
            {str(path):task.sha(path) for path in (Path(preparation_path).resolve(),source_path,
                Path(p['runtime_source_archive']),design_path,Path(__file__).resolve())}))
    if result['approved_disabled_subfloor_paths']!=original['approved_disabled_subfloor_paths']:
        raise ValueError('Do not silently add disabled collision identities')
    if result['supports']!=original['supports']:
        raise ValueError('Exactly the two original support identities must remain')
    if len(result['source_actor_node_paths'])!=145:raise ValueError('All 145 exact source actor nodes required')
    task.verify_pins(result['input_sha256'])
    return task.evidence.json_value(result)


def collect_task_queries(protocol,query,vec3):
    """GET-only 32 original-coordinate rays; every hit survives serialization.

    The self/nonself partitions are compatibility fields for the unchanged
    static loop. The H5 evaluator always consumes ALL hits and validates the
    actual source-node identities rather than trusting the prefix partition.
    """
    result=[];actor=protocol['new_robot_root']
    for spec in protocol['sampled_cooked_support_queries']:
        hits=[];errors=[];invalid=[]
        def receive(hit):
            try:
                hits.append(finite_hit(hit))
            except Exception as error:
                errors.append(type(error).__name__+': '+str(error));invalid.append(invalid_hit_diagnostic(hit))
            return True
        try:
            query.raycast_all(vec3(*spec['xy_m'],spec['origin_z_m']),vec3(*spec['direction']),spec['length_m'],receive)
        except Exception as error:errors.append(type(error).__name__+': '+str(error))
        belongs=lambda hit:hit['collision']==actor or hit['collision'].startswith(actor+'/')
        result.append(dict(id=spec['id'],all_hits=hits,errors=errors,
            invalid_raw_hit_diagnostics=invalid,
            own_robot_occlusion_hits=[hit for hit in hits if belongs(hit)],
            all_nonself_hits=[hit for hit in hits if not belongs(hit)]))
    return result


def support_evaluation(protocol,observations):
    raw=[dict(id=row['id'],all_hits=row['all_hits'],errors=row['errors']) for row in observations]
    result=task.static_query_evaluation(protocol,raw,actor_node_paths=protocol['source_actor_node_paths'])
    for row in observations:
        actor=protocol['new_robot_root']
        own=[h for h in row['all_hits'] if h['collision']==actor or h['collision'].startswith(actor+'/')]
        nonself=[h for h in row['all_hits'] if h not in own]
        if own!=row['own_robot_occlusion_hits'] or nonself!=row['all_nonself_hits']:
            raise ValueError('Raw H5 hits and compatibility partitions disagree; no omitted hit')
    return result


def subfloor_evaluation(protocol,support,observations):
    original=task.shared.stationary.evaluate_subfloor_queries(protocol,support,observations)
    return dict(original,inherited_component_scope_text=original['scope'],
        scope='FOUR_UNCHANGED_IDENTITY_ABSENCE_PROBES_WITH_EXPLICIT_H5_32_ALL_HIT_SUPPORT_QUERIES',
        original_evaluator_code_unchanged=True,support_query_count=len(support))


def query_functions(scope):
    """Only read-only collectors; the original static loop calls after Pause."""
    query=scope['omni'].physx.get_physx_scene_query_interface()
    def support():return collect_task_queries(scope['protocol'],query,scope['carb'].Float3)
    _,original_probes=task.shared.completed_static.frozen_query_functions(scope)
    def probes():
        rows=original_probes()
        for row in rows:
            invalid=[]
            for key in ('all_nonself_hits','own_robot_occlusion_hits'):
                valid=[]
                for hit in row[key]:
                    native=SimpleNamespace(collision=hit['collision'],position=hit['position_m'],normal=hit['normal'],distance=hit['distance_m'])
                    try:finite_hit(native);valid.append(hit)
                    except Exception as error:
                        invalid.append(dict(original_partition=key,**invalid_hit_diagnostic(native)))
                        row['errors'].append(type(error).__name__+': '+str(error))
                row[key]=valid
            row['invalid_raw_hit_diagnostics']=invalid
        return rows
    probes.__wrapped__=original_probes
    return support,probes


def task_log(*args,**kwargs):
    """Change only the component's progress heading, never recorded evidence."""
    if args and isinstance(args[0],str):
        first=args[0]
        if first.startswith('[H4 STATIC]'):first='[H5 STATIC]'+first[len('[H4 STATIC]'):]
        elif first=='=== H4 FRESH FIXED-START STATIC ===':first='=== H5 FRESH FIXED-START STATIC ==='
        elif first.startswith('[H4 WATCH '):first='[H5 WATCH '+first[len('[H4 WATCH '):]
        elif first.startswith('H4 path phase:'):first='H5 path phase:'+first[len('H4 path phase:'):]
        elif first=='=== IJCAS HOSPITAL H4 FROZEN PATH EXECUTION GATE ===':first='=== H5 FROZEN PATH EXECUTION GATE ==='
        args=(first,*args[1:])
    builtins.print(*args,**kwargs)


def static_context(protocol,path):
    """Explicit paths and query bindings, not an executed trial."""
    path=Path(path).resolve();context=task.task_context(protocol)
    if (protocol.get('static_runtime_version')!=VERSION or path!=context.output/'runtime_protocol.json'
            or len(protocol.get('source_actor_node_paths',[]))!=145):
        raise ValueError('Own exact prospective static-runtime definition required')
    context.protocol_path=path;context.stationary.PROTOCOL=path
    context.stationary.evaluate_subfloor_queries=subfloor_evaluation
    context.support_evaluator=support_evaluation
    helpers=task.preparation_functions(context)
    completed=SimpleNamespace(**vars(task.shared.completed_static));completed.frozen_query_functions=query_functions
    static=task.shared._bound(task.shared.run_retained_static,preparation=context.preparation,cohort=context.cohort,
        completed_static=completed,STATIC_OUTPUT=context.stationary.RESULT.parent,STATIC_RUN_ID=context.static_run_id,
        STATIC_DESIGN=path,fresh_source_for_live_actor=helpers.source_for_live_actor,
        evaluate_support_queries_for_actor=support_evaluation,print=task_log)
    replay=task.shared._bound(task.shared.load_fresh_static_inputs,preparation=context.preparation,
        completed_static=completed)
    context.operations=SimpleNamespace(stage=helpers.stage_after_archive,static=static,replay_static=replay)
    return context


async def run_static_once(context,*,authorized=False):
    """One library operation, requiring explicit workflow/static consent."""
    if authorized is not True:raise ValueError('Separate explicit single-static consent required before SDK imports')
    # The original function enforces this cohort's own STOPPED/PAUSED source,
    # layer, environment, reservation and gain state before calling Play.
    return await context.operations.static(context)


def reporting_plan(protocol):
    """Fill ONE missing task-report field, never modify a reference or search."""
    bundle=json.loads(Path(protocol['frozen_bundle']).read_bytes())
    paths={json.loads(Path(p).read_bytes())['method']:Path(p) for p in bundle['plans']}
    proposed=json.loads(paths['P_dynamics_full_envelope'].read_bytes());pool=proposed['spatial_candidates']
    baselines=task.shared.execution.METHODS[:3]
    if set(pool)!=set(baselines):raise ValueError('Exact three original spatial candidates required')
    for method in baselines:
        if pool[method]!=json.loads(paths[method].read_bytes())['selected_actions']:
            raise ValueError('Candidate pool differs from the actual saved native baseline actions')
    digest=task.shared.digest(pool)
    plan=json.loads(paths[protocol['method']].read_bytes())
    if 'shared_candidate_pool_sha256' in plan:raise ValueError('Unexpected pre-existing pool reporting field; review instead of overwriting')
    plan['shared_candidate_pool_sha256']=digest
    context=task.task_context(protocol);context.execution.validate_plan(plan)
    metadata=dict(field='shared_candidate_pool_sha256',value=digest,
        definition='canonical JSON of actual P spatial_candidates; reporting reference, NOT equal planner inputs or cost',
        original_native_plan_path=str(paths[protocol['method']]),original_native_plan_sha256=task.sha(paths[protocol['method']]),
        source_P_plan=str(paths['P_dynamics_full_envelope']),source_P_plan_sha256=task.sha(paths['P_dynamics_full_envelope']),
        baseline_candidate_names=list(baselines),all_candidates_evaluated=proposed['all_spatial_candidates_evaluated'],
        selected_spatial_source=proposed['selected_spatial_source_route_id'],
        measured_planning_computation=bundle['planning_computation_accounting'],
        source_actions_states_limits_and_saved_files_unchanged=True)
    return plan,metadata,bundle


def route_function_nodes():
    """Read only the named definitions; NEVER execute the old entry's body."""
    source=task.shared.ENTRY.read_text();tree=ast.parse(source)
    found={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ROUTE_FUNCTIONS}
    if set(found)!=set(ROUTE_FUNCTIONS) or any(n.decorator_list or n.args.defaults or n.args.kw_defaults for n in found.values()):
        raise ValueError('Exact non-decorated frozen route definitions required')
    return [found[name] for name in ROUTE_FUNCTIONS],source


def route_source_definition():
    nodes,source=route_function_nodes()
    return dict(entry=str(task.shared.ENTRY),entry_sha256=task.sha(task.shared.ENTRY),
        functions={n.name:hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in nodes},
        extracted_definitions_only_no_old_top_level_import_initialization_scheduling_or_authorization=True,
        full_module_compiled_without_executing_its_top_level=True)


def compile_route_functions(scope):
    # Python 3.11 uses whole-module import information when selecting call
    # opcodes. Compiling isolated AST definitions is semantically similar but
    # not byte-for-byte identical (e.g. np.asarray LOAD_ATTR/LOAD_METHOD).
    # Compile the untouched WHOLE source without executing ANY module body,
    # then bind only its reviewed function code objects to the private scope.
    _,source=route_function_nodes();namespace=dict(scope,__file__=str(task.shared.ENTRY),
        _ijcas_path=_ijcas_path,_ijcas_workspace=_ijcas_workspace)
    module=compile(source,str(task.shared.ENTRY),'exec',dont_inherit=True)
    codes={code.co_name:code for code in module.co_consts if isinstance(code,CodeType) and code.co_name in ROUTE_FUNCTIONS}
    if set(codes)!=set(ROUTE_FUNCTIONS) or any(code.co_freevars for code in codes.values()):
        raise ValueError('Exact top-level original route code objects without closures required')
    for name in ROUTE_FUNCTIONS:namespace[name]=FunctionType(codes[name],namespace,name)
    required=set()
    def globals_in(code):
        for instruction in dis.get_instructions(code):
            if instruction.opname=='LOAD_GLOBAL':required.add(instruction.argval)
        for value in code.co_consts:
            if isinstance(value,CodeType):globals_in(value)
    for name in ROUTE_FUNCTIONS:globals_in(namespace[name].__code__)
    missing=required-set(namespace)-set(vars(builtins))
    if missing:raise ValueError('Missing original route global bindings BEFORE any reservation: '+repr(sorted(missing)))
    return namespace


def task_report_writer(context,metadata):
    """Bind only labels in a NEW actual H5 report; never relabel saved trials."""
    expected=context.output/'route'/(context.route_run_id+'.json')
    def write(path,report):
        if (Path(path)!=expected or report.get('scenario')!=task.SCENARIO
                or report.get('run_id')!=context.route_run_id or report.get('method')!=context.protocol['method']
                or report.get('physics_preparation_snapshot',{}).get('new_robot_root')!=context.cohort.ROBOT
                or report.get('start_world_xyyaw')!=context.protocol['start_xyyaw']
                or report.get('goal_world_xyyaw')!=context.protocol['goal_xyyaw']
                or report.get('trial_kind')!='H4_frozen_path_PhysX_development_ground_truth_feedback_NOT_sensor_navigation'
                or report.get('execution_role')!='fresh_fixed_start_whole_H4_development_pilot'
                or report.get('use_for_final_statistics') is not False):
            raise ValueError('Only this NEW H5 measurement report may receive task metadata; old outcomes are immutable')
        changed={}
        for key,new in (('trial_kind','H5_frozen_path_PhysX_development_ground_truth_feedback_NOT_sensor_navigation'),
                ('execution_role','fresh_fixed_start_whole_H5_development_pilot')):
            changed[key]=dict(inherited_component_label=report[key],task_label=new);report[key]=new
        for old,new in (('selected_frozen_H4_route_PhysX_verified','selected_frozen_H5_route_PhysX_verified'),
                ('full_H4_task_completed','full_H5_task_completed')):
            if old not in report:raise ValueError('Missing original completion metadata; never invent an outcome')
            report[new]=report.pop(old);changed[old]=dict(task_field=new,numeric_or_boolean_value_unchanged=True)
        report.update(task_only_component_metadata_binding=changed,runtime_plan_reporting_addition=metadata,
            new_H5_task_report_NOT_a_relabelled_old_trial=True,
            common_motion_stop_scope='shared_H1_actuator_hold_component_benchmark_NOT_H5_moving_stop_or_path_validation',
            H5_32_support_and_four_absence_probe_static_replayed=report.pop('prospective_26_support_and_four_absence_probe_static_replayed'),
            inherited_marker_property_name='last_h4_path_gate; component bookkeeping name only')
        task.shared.write_runtime_evidence(path,report)
    return write


def live_sdk():
    """Import the already running app's SDK; never create a SimulationApp."""
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.api.sensors import RigidContactView
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import PhysicsSchemaTools,Usd,UsdUtils
    return SimpleNamespace(omni=omni,RigidContactView=RigidContactView,SingleArticulation=SingleArticulation,
        SimulationManager=SimulationManager,ArticulationAction=ArticulationAction,
        PhysicsSchemaTools=PhysicsSchemaTools,Usd=Usd,UsdUtils=UsdUtils)


def actual_route_inputs(context):
    prep,static,replay=context.operations.replay_static(context)
    plan,metadata,bundle=reporting_plan(context.protocol)
    plan_path=Path(metadata['original_native_plan_path']);manifest_path=Path(plan['geometry_manifest'])
    manifest=json.loads(manifest_path.read_bytes())
    if (task.sha(manifest_path)!=plan['geometry_manifest_sha256'] or manifest['scenario']!=task.SCENARIO
            or manifest['start_xyyaw']!=plan['start_world_xyyaw'] or manifest['goal_xyyaw']!=plan['goal_world_xyyaw']):
        raise ValueError('Exact own H5 task/goal manifest required; never translate a reference')
    with np.load(context.cohort.MAP) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    if (occupied.dtype!=np.dtype(bool) or hashlib.sha256(occupied.tobytes()).hexdigest()!=plan['planning_occupancy_digest']
            or plan['planning_occupancy_digest']!=manifest['candidate_occupancy_digest']
            or list(occupied.shape)!=manifest['raster']['shape_rows_columns']
            or not np.array_equal(origin,manifest['raster']['origin_xy_m']) or resolution!=manifest['raster']['resolution_m']):
        raise ValueError('Complete original H5 raster required before any route reservation')
    if context.execution.GUARDS!=context.protocol['guards']:
        raise ValueError('Do not alter shared prospective duration or safety guards')
    return SimpleNamespace(prep=prep,static=static,replay=replay,plan=plan,plan_path=plan_path,
        metadata=metadata,bundle=bundle,manifest_path=manifest_path,manifest=manifest,
        occupied=occupied,origin=origin,resolution=resolution)


def prepare_route_scope(context,inputs,sdk):
    """GET-only preflight and complete frozen-kernel bindings, not a command."""
    c,s=context.cohort,context.stationary;p=context.protocol;plan=inputs.plan
    timeline=sdk.omni.timeline.get_timeline_interface();stage=sdk.omni.usd.get_context().get_stage()
    if stage is None or task.evidence.timeline_state(timeline)!=inputs.static['actual_final_timeline']:
        raise ValueError('Exact own H5 static PAUSED boundary required before route setup')
    if timeline.is_playing() or timeline.is_stopped():raise ValueError('Own physical static must remain PAUSED')
    setup=stage.GetPrimAtPath(c.MARKER)
    if not setup.IsValid() or setup.GetCustomDataByKey('robot_root')!=c.ROBOT:
        raise ValueError('Own fresh H5 marker and actor required')
    environment=s.critical_environment(stage,inputs.prep)
    if environment!=inputs.static['environment_after_static']:
        raise ValueError('Actual H5 physical settings differ from the completed static')
    view=sdk.SimulationManager.get_physics_sim_view()
    if view is None:raise ValueError('Existing completed-static physics view required; never Play to create it')
    tensor=view.create_articulation_view(c.ROBOT+'/base_link')
    if tensor.count!=1 or tensor.max_links!=13:raise ValueError('One exact thirteen-link completed-static actor required')
    validate=task.shared._bound(task.shared.validate_complete_tensor,preparation=context.preparation)
    def live_static_check():return validate(inputs.static,tensor)
    handoff=live_static_check()
    canonical='/World/patient_transport_h5_nominal_recipe'
    source_context=SimpleNamespace(protocol=p,cohort=SimpleNamespace(ROBOT=canonical),stationary=SimpleNamespace(
        actor_configuration=task.shared._bound(task.shared.stationary.actor_configuration,ROBOT=canonical)))
    source=task.preparation_functions(context).source_reference(source_context,True)
    physics=task.shared._bound(task.shared.physics_baseline_digest,cohort=c)(inputs.prep,environment,
        inputs.static['runtime_motor_baseline'],source_settings=source['fixed_configuration_rows'])
    output=context.output/'route';run_id=context.route_run_id
    launch=dict(PROTOCOL=context.protocol_path,RESERVATION=output/'single_route_reserved.json',
        validate_environment=lambda stage,prep,protocol:s.critical_environment(stage,prep),
        layer_field_differences=task.shared.layer_field_differences,
        write_runtime_evidence=task_report_writer(context,inputs.metadata))
    scope=dict(asyncio=asyncio,Counter=Counter,deque=deque,csv=csv,hashlib=hashlib,json=json,math=math,os=os,
        Path=Path,time=time,np=np,**vars(sdk),WORKSPACE=task.ROOT,HELPERS=_ijcas_path(task.ROOT, 'studies/IJCAS/scripts'),
        execution_helpers=vars(context.execution),Controller=context.execution.TrackingController,
        direct_joint_targets=context.execution.direct_joint_targets,relative_pose=context.execution.relative_pose,
        hold_gains=drive_hold_gain_arrays,stop_analysis=analyze_stop_trace,pose_collision=pose_collision,
        evidence=task.evidence,c=c,cohort=p['cohort'],stationary=s,protocol=p,static_protocol=p,
        preparation=inputs.prep,static=inputs.static,plan=plan,launch=launch,
        initial_motion=False,fresh_protocol_path=str(context.protocol_path),trial_context=context,
        PLAN_PATH=inputs.plan_path,plan_raw=inputs.plan_path.read_bytes(),plan_sha=task.sha(inputs.plan_path),
        static_path=s.RESULT,STATIC_TRACE=s.TRACE,static_raw=s.RESULT.read_bytes(),
        component_path=task.shared.BENCHMARK,component_raw=task.shared.BENCHMARK.read_bytes(),
        manifest_path=inputs.manifest_path,manifest_raw=inputs.manifest_path.read_bytes(),
        manifest_sha=task.sha(inputs.manifest_path),manifest=inputs.manifest,
        environment=environment,physics_sha=physics,timeline=timeline,stage=stage,setup=setup,
        ROOT=c.ROBOT,BASE=c.ROBOT+'/base_link',
        STEERING=('base_link_to_front_active_steering','base_link_to_rear_active_steering'),
        DRIVES=('front_active_steering_to_drive_wheel','rear_active_steering_to_drive_wheel'),
        WHEELS=c.WHEELS,BAD_CONTACTS=s.BAD_CONTACTS,
        supports=c.support_covers(c.support.load_design()),tensor=tensor,live_static_check=live_static_check,live_handoff=handoff,
        occupied=inputs.occupied,map_origin=inputs.origin,resolution=inputs.resolution,
        OUTPUT=output,run_id=run_id,summary_path=output/(run_id+'.json'),trace_path=output/(run_id+'.csv'),
        runtime_guards=dict(context.execution.GUARDS),h4_clock_state={'time_s':0.},
        counts=Counter(),pairs=Counter(),callback_errors=[],print=task_log)
    return compile_route_functions(scope)


async def run_route_once(context,*,authorized=False):
    if authorized is not True:raise ValueError('Separate explicit consent for ONE fixed H5 route required')
    inputs=actual_route_inputs(context);scope=prepare_route_scope(context,inputs,live_sdk())
    output=scope['OUTPUT'];reserve=scope['launch']['RESERVATION']
    if reserve.exists() or any(output.glob('*.json')) or any(output.glob('*.csv')):
        raise ValueError('This own static already reserved/recorded a route; no retry or second method')
    output.mkdir(parents=True,exist_ok=True)
    task.evidence.write_exclusive(reserve,dict(method=context.protocol['method'],run_id=context.route_run_id,
        actual_static_sha256=task.sha(context.stationary.RESULT),protocol_sha256=task.sha(context.protocol_path),
        explicit_user_authorization=True,no_motion_yet=True,no_automatic_retry=True))
    await scope['_run_h4_path']()
    return json.loads(scope['summary_path'].read_bytes())


def workflow_source_archive():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        for path in (Path(__file__).resolve(),ENTRY):
            item=zipfile.ZipInfo(str(path.relative_to(task.ROOT)),date_time=(1980,1,1,0,0,0));item.external_attr=0o100644<<16
            archive.writestr(item,path.read_bytes())
    return buffer.getvalue()


def workflow_definition(preparation_path):
    p=static_runtime_definition(preparation_path);context=task.task_context(p)
    _,metadata,_=reporting_plan(p);payload=workflow_source_archive()
    lo=min(p['start_xyyaw'][1],p['goal_xyyaw'][1])-.825
    hi=max(p['start_xyyaw'][1],p['goal_xyyaw'][1])+.825
    p.update(workflow_version=WORKFLOW_VERSION,route_kernel=route_source_definition(),route_reporting_metadata=metadata,
        workflow_runtime_sources=str(context.output/'workflow_runtime_sources.zip'),
        workflow_runtime_sources_sha256=hashlib.sha256(payload).hexdigest(),
        camera_preparation=dict(helper=str(CAMERA_HELPER),source_sha256=task.sha(CAMERA_HELPER),
            operation='existing_Top_XY_and_apertures_only_BEFORE_archiving_layers_NO_Frame_Selected',
            target_xyz_m=[(p['start_xyyaw'][0]+p['goal_xyyaw'][0])/2.,(lo+hi)/2.,0.],
            minimum_window_height_m=hi-lo+1.,retain_height_orientation_clipping_and_renderer=True,
            actual_visible_pixels_NOT_assumed=True),
        source_geometry_sha256=task.sha(Path(json.loads(Path(p['frozen_bundle']).read_bytes())['geometry_manifest'])),
        sequence=['verify_actual_prior_PAUSED_state','camera_only_before_layer_archive','archive_and_ONE_confirmed_Stop',
            'same_source_new_actor_and_visual_registration','ONE_original_static','if_PASS_ONE_same_selected_method_route',
            'ONE_5s_hold_and_confirmed_Pause_gain_restore'],
        maximum_Stop_calls=1,maximum_static_trials=1,maximum_route_trials=1,
        explicit_separate_user_workflow_authorization_required=True,
        original_source_and_task_and_all_safety_limits_unchanged=True,
        status='OFFLINE_SINGLE_H5_WORKFLOW_READY_FOR_REVIEW_NOT_MOTION_AUTHORIZATION')
    p['input_sha256'].pop(str(Path(__file__).resolve()))
    p['input_sha256']=task.merge_pins(p['input_sha256'],
        {str(task.shared.ENTRY):task.sha(task.shared.ENTRY),str(CAMERA_HELPER):task.sha(CAMERA_HELPER),
            p['workflow_runtime_sources']:p['workflow_runtime_sources_sha256']})
    p['definition_sha256']=task.shared.digest(p)
    return task.evidence.json_value(p)


def export_workflow(preparation_path):
    p=workflow_definition(preparation_path);context=task.task_context(p);path=context.output/'runtime_protocol.json'
    archive=Path(p['workflow_runtime_sources']);payload=workflow_source_archive()
    if path.exists():
        if json.loads(path.read_bytes())!=p or archive.read_bytes()!=payload:raise ValueError('Frozen workflow differs; no overwrite')
        return path
    if archive.exists() or (context.output/'workflow_dispatch_reserved.json').exists():
        raise ValueError('Existing partial/dispatched workflow retained; no overwrite or automatic retry')
    with archive.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    task.evidence.write_exclusive(path,p)
    return path


def load_workflow(path):
    """Small source/config checks only; no expensive planning/MODEL reruns in Kit."""
    path=Path(path).resolve();p=json.loads(path.read_bytes());context=task.task_context(p)
    if (path!=context.output/'runtime_protocol.json' or p.get('workflow_version')!=WORKFLOW_VERSION
            or p.get('definition_sha256')!=task.shared.digest({k:v for k,v in p.items() if k!='definition_sha256'})
            or p.get('actual_motion_authorized') is not False or p.get('use_for_final_statistics') is not False
            or p.get('automatic_retry') is not False
            or (p.get('maximum_Stop_calls'),p.get('maximum_static_trials'),p.get('maximum_route_trials'))!=(1,1,1)
            or p['route_kernel']!=route_source_definition()
            or Path(p['workflow_runtime_sources']).read_bytes()!=workflow_source_archive()):
        raise ValueError('Exact frozen reviewed H5 workflow source/identity/single-trial contract required')
    prep=json.loads(Path(p['preparation_definition']).read_bytes())
    if any(p.get(k)!=v for k,v in prep.items() if k not in ('status','input_sha256')):
        raise ValueError('Original H5 preparation/task/physical settings changed')
    if (p['command_limits']!=context.execution.COMMAND_LIMITS or p['guards']!=context.execution.GUARDS
            or p['acceptance']!=context.execution.ACCEPTANCE
            or p['terminal_controller_config']!=context.execution.CONTROLLER_CONFIG):
        raise ValueError('Shared control, resource and acceptance bands must remain exact')
    plan,metadata,_=reporting_plan(p)
    if metadata!=p['route_reporting_metadata']:raise ValueError('Actual native references/pool reporting metadata changed')
    return static_context(p,path)


async def verify_workflow_inputs(context,sdk):
    stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface();p=context.protocol
    if stage is None or task.evidence.timeline_state(timeline)!=p['required_prior_timeline']:
        raise ValueError('Actual saved H4 boundary must still exist; no reopen/reset/Play to recreate it')
    view=sdk.SimulationManager.get_physics_sim_view()
    if view is None:raise ValueError('Existing actual prior tensor required')
    tensor=view.create_articulation_view(p['retired_robot_root']+'/base_link')
    if tensor.count!=1 or tensor.max_links!=13:raise ValueError('Exact actual prior thirteen-link actor required')
    layers=(stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString())
    def unchanged(*,camera_delta=False):
        if (sdk.omni.usd.get_context().get_stage()!=stage or task.evidence.timeline_state(timeline)!=p['required_prior_timeline']
                or task.shared.normalize_runtime_evidence(context.preparation.old.tensor_snapshot(tensor))!=p['required_prior_tensor']
                or (not camera_delta and (stage.GetRootLayer().ExportToString(),stage.GetSessionLayer().ExportToString())!=layers)):
            raise ValueError('Prior body/all joints/links/targets/gains/layers changed before archive; no recovery')
    unchanged();began=heartbeat=time.monotonic()
    for i,(name,value) in enumerate(p['input_sha256'].items(),1):
        if task.sha(name)!=value:raise ValueError('Frozen workflow input changed: '+name)
        if i%16==0:
            unchanged();await sdk.omni.kit.app.get_app().next_update_async();unchanged()
        if time.monotonic()-heartbeat>=15.:
            task_log('[H5 PREFLIGHT] frozen input pins verified:',i,'/',len(p['input_sha256']));heartbeat=time.monotonic()
        if time.monotonic()-began>180.:raise TimeoutError('Frozen input preflight wall limit; no motion or retry')
    unchanged()
    return stage,timeline,unchanged


def prepare_task_view(context,sdk,stage,timeline,unchanged):
    """One camera-only change, BEFORE the exact Stop/static layer baselines."""
    from omni.kit.viewport.utility import get_active_viewport
    view=task.load(CAMERA_HELPER);viewport=get_active_viewport();p=context.protocol
    if viewport is None or str(viewport.camera_path)!=view['CAMERA']:
        raise ValueError('Existing Top viewport required before archive; no renderer/camera substitution')
    unchanged();noncamera=view['without_top_camera'](stage.GetSessionLayer());root=stage.GetRootLayer().ExportToString()
    width=p['camera_preparation']['minimum_window_height_m']*float(viewport.resolution[0])/float(viewport.resolution[1])
    result=view['focus_top_xy_only'](stage,p['camera_preparation']['target_xyz_m'],viewport.resolution,width)
    unchanged(camera_delta=True)
    if stage.GetRootLayer().ExportToString()!=root or view['without_top_camera'](stage.GetSessionLayer())!=noncamera:
        raise ValueError('Non-camera field changed during task view preparation')
    result.update(actual_visible_pixels_NOT_assumed=True,all_prior_body_joint_link_targets_and_gains_retained=True,
        actual_timeline=task.evidence.timeline_state(timeline),no_Play_Stop_reset_or_renderer_change=True)
    task.shared.write_runtime_evidence(context.output/'view_preparation.json',result)


async def run_workflow_once(path,*,authorized=False):
    if authorized is not True:raise ValueError('Separate explicit approval for ONE H5 archive/Stop/static/conditional-route workflow required')
    context=load_workflow(path);reserve=context.output/'workflow_dispatch_reserved.json'
    if reserve.exists():raise ValueError('This workflow already dispatched; no retry or replacement')
    task.evidence.write_exclusive(reserve,dict(protocol=str(context.protocol_path),sha256=task.sha(context.protocol_path),
        explicit_single_workflow_consent=True,no_motion_yet=True,no_automatic_retry=True))
    phase='prior_PAUSED_input_preflight'
    try:
        sdk=live_sdk();stage,timeline,unchanged=await verify_workflow_inputs(context,sdk)
        phase='camera_only_BEFORE_archive';prepare_task_view(context,sdk,stage,timeline,unchanged)
        phase='archive_ONE_Stop_new_source_and_display';await context.operations.stage(context)
        phase='ONE_original_static';static=await run_static_once(context,authorized=True)
        if static['verdict']!='PASS':
            task_log('[H5 WORKFLOW] Static verdict:',static['verdict'],'; route NOT started. No retry.')
            return dict(status='STATIC_NOT_PASS_NO_ROUTE',static=static)
        context.operations.replay_static(context)
        phase='ONE_fixed_native_route';route=await run_route_once(context,authorized=True)
        task_log('[H5 WORKFLOW] Final actual route verdict:',route['verdict'],'; no additional leg or retry.')
        return dict(status='SINGLE_H5_DEVELOPMENT_WORKFLOW_FINISHED',static_verdict=static['verdict'],route_verdict=route['verdict'])
    except BaseException as error:
        diagnostic=dict(phase=phase,error=type(error).__name__+': '+str(error),
            no_automatic_retry_or_state_restore=True,existing_reports_remain_authoritative=True,
            setup_or_recording_error_NOT_a_scientific_method_limitation=True)
        try:task.evidence.write_exclusive(context.output/'workflow_error.json',diagnostic)
        except Exception as secondary:task_log('[H5 WORKFLOW] Diagnostic writer also failed:',type(secondary).__name__,str(secondary))
        raise
