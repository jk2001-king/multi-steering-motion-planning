"""Shared task experiment preparation; no implicit live experiment entry.

H5 binds its own geometry, queries and actor to the existing frozen source
constructor, complete environment readback, Stop archive and visual setup.
No controller is copied. The static/route launcher is intentionally NOT
implemented here yet; an exported design grants no live authority.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import hospital_h4_subfloor_compact_path_launch as shared
import hospital_runtime_evidence_io as evidence
from hospital_private_loader import load
from hospital_task_path_execution import task_execution_binding


ROOT, GATE = shared.ROOT, shared.GATE
SCENARIO = 'H5_obstacle_bypass_return'
VERSION = 'H5_EXPLICIT_TASK_SOURCE_AND_STATIC_QUERY_PREPARATION_DESIGN_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_pins(*collections):
    result={}
    for collection in collections:
        for name,value in collection.items():
            if name in result and result[name]!=value:
                raise ValueError('Conflicting immutable dependency: '+name)
            result[name]=value
    return result


def verify_pins(pins):
    for name,value in pins.items():
        if sha(name)!=value:raise ValueError('Task source or recorded evidence changed: '+name)


def asset_tools():
    # Works in embedded Kit without changing sys.path or global import state.
    return SimpleNamespace(**load(_ijcas_path(ROOT, 'studies/IJCAS/scripts/record_hospital_asset_measurement.py')))


def task_context(protocol):
    """Only explicit new H5 identities; original helper globals stay intact."""
    import re
    name=protocol['cohort'];method=protocol['method']
    if (protocol.get('version')!=VERSION or protocol.get('scenario')!=SCENARIO
            or re.fullmatch(r'cohort_[0-9]{2}_[A-Za-z0-9_]+_r[0-9]{2}',name) is None
            or method not in shared.execution.METHODS):
        raise ValueError('Explicit supported task/cohort/method required; no H3/H6 fallback')
    target=GATE/'hospital_h5_experiments'/name
    robot='/World/patient_transport_h5_'+name;marker='/World/IJCAS_HospitalH5_'+name
    if (protocol['new_robot_root']!=robot or protocol['marker']!=marker
            or Path(protocol['output_directory'])!=target
            or protocol['retired_robot_root']==robot):
        raise ValueError('Exact own fresh output/actor identities required')
    p=SimpleNamespace(**vars(shared.preparation))
    p.ROBOT,p.MARKER,p.OLD_ROBOT,p.COHORT=robot,marker,protocol['retired_robot_root'],name
    p.LIVE=target/'preparation.json'
    c=SimpleNamespace(**vars(shared.cohort))
    c.ROBOT,c.MARKER,c.COHORT,c.LIVE=robot,marker,name,p.LIVE
    c.MAP=Path(protocol['planning_occupancy'])
    s=SimpleNamespace(**vars(shared.stationary))
    s.ROBOT,s.MARKER,s.preparation=robot,marker,p
    s.actor_configuration=shared._bound(shared.stationary.actor_configuration,ROBOT=robot,preparation=p)
    s.critical_environment=shared._bound(shared.stationary.critical_environment,ROBOT=robot,preparation=p)
    s.classify_contact=shared._bound(shared.stationary.classify_contact,ROBOT=robot,cohort=c)
    s.RESULT=target/'static'/'hospital_h5_static_r01.json';s.TRACE=s.RESULT.with_suffix('.csv')
    s.PROTOCOL=target/'protocol_v2.json'
    return SimpleNamespace(protocol=protocol,output=target,preparation=p,cohort=c,stationary=s,
        static_run_id=s.RESULT.stem,route_run_id='hospital_h5_'+name+'_'+method+'_r01',
        protocol_path=s.PROTOCOL,execution=task_execution_binding(SCENARIO,
            simulation_budget_s=protocol['common_resource_proposal']['maximum_simulation_duration_s']))


def preparation_functions(context):
    """Reuse exact frozen operations by configuration, never by relabelling data.

    These callables are returned, not called or scheduled. Their bytecode is
    unchanged. Stage/Stop consent must be checked by a future complete entry;
    preparation alone must never admit a static or route trial.
    """
    p=context.preparation
    reference=shared._bound(shared.fresh_source_reference,preparation=p)
    actor=shared._bound(shared.fresh_source_for_live_actor,preparation=p,fresh_source_reference=reference)
    environment=shared._bound(shared.fresh_environment_readback,preparation=p)
    retiring=shared._bound(shared.retiring_actor_configuration,stationary=context.stationary)
    display=shared._bound(shared.initialize_fresh_staged_display,preparation=p,
        fresh_source_for_live_actor=actor,fresh_environment_readback=environment)
    stage=shared._bound(shared.prepare_fresh_live_trial,preparation=p,
        fresh_source_for_live_actor=actor,fresh_environment_readback=environment,
        retiring_actor_configuration=retiring,initialize_fresh_staged_display=display)
    return SimpleNamespace(source_reference=reference,source_for_live_actor=actor,
        environment_readback=environment,initialize_staged_display=display,stage_after_archive=stage)


def static_query_evaluation(protocol,observations,*,actor_node_paths):
    """Two channels: original-top floor detection and mapped nonself hits.

    Keep every all-hit row, including self and control hits. Start/goal expanded
    footprint corners must have no nonself obstacle. Grid/control obstacle hits
    remain mapped obstacles, never support or a collision-policy exemption.
    Unknown/self-prefix-only hits, off-ray values and disabled identities fail.
    """
    api=asset_tools();specs=protocol['sampled_cooked_support_queries']
    limits=protocol['additional_sampled_query_limits'];actor=protocol['new_robot_root']
    supports=protocol['supports'];top=protocol['support_top_m']
    if limits!=shared.stationary.QUERY_LIMITS or set(supports)!=set(shared.stationary.SUPPORTS):
        raise ValueError('Original support identities and unchanged query tolerances required')
    summary=api.summarize_corridor_cached_queries(specs,observations,supports,top,limits,
        protocol['selected_original_cart'],actor)
    inventory=protocol['query_known_enabled_colliders']
    own=set(actor_node_paths)
    if (not own or any(not p.startswith(actor+'/') and p!=actor for p in own)
            or len(own)!=len(actor_node_paths)):
        raise ValueError('Exact source-verified actor node paths, not a blanket namespace exemption, required')
    valid=known=clear=True;details=[]
    disabled=set(protocol['approved_disabled_subfloor_paths'])|set(protocol['original_disabled_collider_paths'])
    for spec,row,diagnostic in zip(specs,observations,summary['diagnostics']):
        row_valid=row_known=row_clear=True;nonself=[]
        for hit in row['all_hits']:
            path=hit['collision'];xyz=hit['position_m'];distance=hit['distance_m']
            on_ray=(all(abs(a-b)<=limits['horizontal_tolerance_m'] for a,b in zip(xyz[:2],spec['xy_m']))
                and -limits['height_tolerance_m']<=distance<=spec['length_m']+limits['height_tolerance_m']
                and abs(spec['origin_z_m']-xyz[2]-distance)<=limits['height_tolerance_m'])
            row_valid=bool(row_valid and on_ray and path not in disabled)
            if path in own:continue
            if path==actor or path.startswith(actor+'/'):
                row_known=False;continue
            if path not in supports:
                nonself.append(path)
                bound=inventory.get(path)
                row_known=bool(row_known and bound is not None)
                if bound is not None:
                    row_valid=bool(row_valid and all(lo-limits['horizontal_tolerance_m']<=v<=hi+limits['horizontal_tolerance_m']
                        for v,lo,hi in zip(xyz,bound['minimum_m'],bound['maximum_m'])))
                if spec['kind']=='endpoint':row_clear=False
        valid=bool(valid and row_valid);known=bool(known and row_known);clear=bool(clear and row_clear)
        details.append(dict(id=spec['id'],raw_ray_transaction_valid=row_valid,
            exact_self_support_or_known_enabled_original_obstacles_only=row_known,
            expanded_endpoint_sample_clear=row_clear,all_nonself_obstacle_identities=nonself,
            floor_detected=bool(diagnostic['matching_original_top_upward_support_hit_indices'])))
    checks=dict(all_prescribed_query_rows_and_raw_geometry_valid=bool(valid and summary['query_error_count']==0),
        original_top_floor_detected_at_every_prescribed_sample=all(d['floor_detected'] for d in details),
        only_exact_actor_support_and_mapped_enabled_obstacle_identities=known,
        no_nonself_obstacle_at_expanded_start_or_goal_samples=clear,
        unchanged_original_cart_control_hit=summary['original_cart_control_hit'])
    return dict(version='H5_PROSPECTIVE_TWO_CHANNEL_ALL_HIT_STATIC_QUERY_CONTRACT_v1',
        checks=checks,summary=summary,diagnostics=details,
        sampled_floor_AND_obstacle_identity_contract_only=True,
        original_obstacles_NOT_reclassified_as_support_or_contact_exemptions=True,
        whole_path_cooked_support_or_static_or_drive_admission=False)


def original_preparation_history(output):
    """Keep the first OFFLINE audit and its exact executed source, unmodified.

    Only this helper evolved. The original source SHA must be reproduced from
    its archived bytes; every other old input still has to match its real file.
    This is explicit historical-source resolution, not a general pin waiver.
    """
    output=Path(output);paths=[output/name for name in
        ('protocol.json','anonymous_source_readback.json','preparation_source_v1.zip')]
    if not any(path.exists() for path in paths):return None
    if not all(path.is_file() for path in paths):raise ValueError('Incomplete original OFFLINE audit history; no overwrite')
    protocol,review=map(lambda p:json.loads(p.read_bytes()),paths[:2]);member=str(Path(__file__).relative_to(ROOT))
    with zipfile.ZipFile(paths[2]) as archive:
        if archive.namelist()!=[member]:raise ValueError('Exact single original preparation source archive required')
        source_sha=hashlib.sha256(archive.read(member)).hexdigest()
    for pins in (protocol['input_sha256'],review['input_sha256']):
        for path,value in pins.items():
            actual=source_sha if Path(path)==Path(__file__).resolve() else sha(path)
            if actual!=value:raise ValueError('Historical OFFLINE evidence/source changed: '+path)
    if (protocol['version']!='H5_EXPLICIT_TASK_SOURCE_AND_STATIC_QUERY_PREPARATION_DESIGN_v1'
            or review['status']!='HOLD_ANONYMOUS_SOURCE_CONFIGURATION_NO_LIVE_ADMISSION'
            or [k for k,v in review['checks'].items() if not v]!=['SDK_unvalued_namespace_changes_only_actor_metadata']
            or review['anonymous_source_without_SDK_namespace']['fixed_configuration_rows']!=
                review['anonymous_source_with_unvalued_SDK_namespace']['fixed_configuration_rows']
            or review['live_stage_robot_or_timeline_accessed'] is not False
            or review['new_PhysX_trial_count']!=0):
        raise ValueError('Exact prior anonymous-only unchanged-configuration HOLD required; no relabelling')
    return dict(original_protocol=str(paths[0]),original_HOLD=str(paths[1]),original_source_archive=str(paths[2]),
        original_source_member=member,original_source_sha256=source_sha,
        original_receipts_NOT_rewritten_or_relabelled=True,new_physical_trial_count=0,
        input_sha256={str(path):sha(path) for path in paths})


def runtime_source_archive():
    """Deterministic small helper snapshot, not another copied controller."""
    buffer=io.BytesIO();member=str(Path(__file__).relative_to(ROOT))
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as archive:
        item=zipfile.ZipInfo(member,date_time=(1980,1,1,0,0,0));item.external_attr=0o100644<<16
        archive.writestr(item,Path(__file__).read_bytes())
    return buffer.getvalue()


def build_protocol(bundle_path,model_path,prior_result,*,cohort,method):
    """Derive a new H5 proposal from actual H4 closeout, not a live readback."""
    api=asset_tools();bundle_path=Path(bundle_path).resolve();model_path=Path(model_path).resolve()
    if (bundle_path.name!='execution_bundle.json' or model_path.name!='common_executor_model_screen.json'
            or not bundle_path.is_file() or not model_path.is_file()):
        raise ValueError('Already completed exact bundle/MODEL receipt paths required; no implicit rerun')
    bundle=json.loads(bundle_path.read_bytes());model=json.loads(model_path.read_bytes())
    if (model.get('execution_bundle')!=str(bundle_path) or model.get('case_count')!=8
            or model.get('actual_motion_authorized') is not False or model.get('use_for_final_statistics') is not False):
        raise ValueError('Own all-eight recorded software observations required, not an admission')
    if (api.write_corridor_execution_bundle(bundle['guided_native_probe'],bundle['corrected_goal_corner_receipt'],bundle_path.parent)!=bundle_path
            or api.screen_corridor_execution_bundle(bundle_path,model_path.parent)!=model_path
            or bundle['status']!='FOUR_TASK_BOUND_REFERENCES_NOMINAL_MODEL_SCREEN_PENDING'
            or model['status']!='NOMINAL_SOFTWARE_MODEL_SCREEN_RECORDED_NOT_PHYSICAL_OR_MODEL_VALIDATION'):
        raise ValueError('Exact complete task bundle and all-eight original MODEL observations required; no automatic search/replay')
    corners,geometry,current,design=api.load_corrected_corridor_goal_corner_receipt(bundle['corrected_goal_corner_receipt'])
    prior=shared.completed_route_boundary(prior_result)
    if prior['prior_result']!=corners['retained_result']:
        raise ValueError('Initial H5 preparation must archive its own actually retained H4 source result')
    plans={json.loads(Path(path).read_bytes())['method']:Path(path) for path in bundle['plans']}
    if method not in plans:raise ValueError('One of the unchanged FOUR methods must be explicit')
    selected=json.loads(plans[method].read_bytes());cases=[c for c in model['cases'] if c['method']==method]
    if len(cases)!=2 or {c['integration_dt_s'] for c in cases}!={.01,.02}:
        raise ValueError('Both original nominal observations required; HOLDs cannot be omitted')
    rows=current['inventory']['collision_prims']
    known={r['path']:r['world_aligned_bound'] for r in rows if r['classification']=='enabled_in_ROI'}
    target=GATE/'hospital_h5_experiments'/cohort
    history=original_preparation_history(target)
    pins=merge_pins(bundle['input_sha256'],bundle['output_sha256'],model['input_sha256'],model['output_sha256'],prior['input_sha256'],
        history['input_sha256'] if history else {},
        {str(path):sha(path) for path in (bundle_path,model_path,Path(shared.__file__),
            _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py'))})
    p=dict(version=VERSION,scenario=SCENARIO,cohort=cohort,method=method,output_directory=str(target),
        new_robot_root='/World/patient_transport_h5_'+cohort,marker='/World/IJCAS_HospitalH5_'+cohort,
        **{key:prior[key] for key in ('retired_robot_root','required_prior_tensor','required_prior_timeline',
            'retiring_actor_fixed_configuration','expected_nominal_motor_baseline','protected_environment_sha256')},
        required_prior_result=prior['prior_result'],saved_prior_boundary=prior,
        frozen_bundle=str(bundle_path),nominal_model_screen=str(model_path),selected_plan=str(plans[method]),
        planning_occupancy=bundle['planning_occupancy'],start_xyyaw=selected['start_world_xyyaw'],goal_xyyaw=selected['goal_world_xyyaw'],
        common_resource_proposal=bundle['common_resource_proposal'],
        command_limits=selected['shared_command_limits'],guards=selected['shared_guards'],acceptance=selected['shared_acceptance'],
        terminal_controller_config=selected['shared_controller_config'],
        environment_variant='four_exact_original_subfloor_collision_flags_disabled_Session_only',
        approved_disabled_subfloor_paths=list(shared.preparation.SELECTED),
        original_disabled_collider_paths=[r['path'] for r in rows if r['classification']=='disabled'],
        supports=current['original_support_colliders'],support_top_m=current['original_top_height_m'],
        additional_sampled_query_limits=current['unchanged_original_query_limits'],
        sampled_cooked_support_queries=geometry['corrected_endpoint_grid_control_query_specifications'],
        selected_original_cart=geometry['selected_original_cart'],query_known_enabled_colliders=known,
        original_32_ray_and_separate_four_corner_receipts_preserved=True,
        selected_method_MODEL_statuses=[c['model_status'] for c in cases],
        nominal_MODEL_is_NOT_actual_motion_or_independent_model_validation=True,
        MODEL_HOLDs_retained_NOT_suppressed_or_structural_path_absence=True,
        trial_role='fresh_source_fixed_start_zero_command_static_NOT_path_performance',
        automatic_Stop_count_limit=1,Stop_wall_budget_s=15.,Stop_update_budget=240,
        compact_readback_wall_budget_s=180.,compact_readback_heartbeat_s=15.,
        compact_readback_UI_yield_max_rows=16,compact_readback_UI_yield_wall_interval_s=.05,
        all_archived_environment_rows_required=4997,static_settle_s=1.,static_record_s=2.,
        initialize_new_actor_visuals_before_static=True,
        source_authored_seed_no_angle_selection=True,maximum_static_attempts=1,maximum_route_attempts=1,
        physical_condition='nominal_265kg_canonical_friction_nominal_CoM',
        body_mass_kg=265.,body_count=13,revolute_joint_count=12,
        automatic_retry=False,no_CICS_Nav2_or_sensor_navigation=True,
        new_live_trial_count=0,actual_motion_authorized=False,use_for_final_statistics=False,
        live_stage_or_timeline_accessed=False,
        original_OFFLINE_preparation_and_HOLD_history=history,
        runtime_source_archive=str(target/'preparation_source_v2.zip'),
        runtime_source_archive_sha256=hashlib.sha256(runtime_source_archive()).hexdigest(),
        status='OFFLINE_PREPARATION_DESIGN_COMPLETE_LIVE_STATIC_ROUTE_ENTRY_NOT_IMPLEMENTED_OR_AUTHORIZED',
        input_sha256=pins)
    context=task_context(p)
    if (context.execution.COMMAND_LIMITS!=p['command_limits'] or context.execution.GUARDS!=p['guards']
            or context.execution.ACCEPTANCE!=p['acceptance'] or context.execution.CONTROLLER_CONFIG!=p['terminal_controller_config']):
        raise ValueError('H5 adapter must preserve exact shared limits/control and prospective common duration')
    verify_pins(pins)
    return evidence.json_value(p)


def export_protocol(*args,**kwargs):
    p=build_protocol(*args,**kwargs);context=task_context(p);target=context.protocol_path
    archive=Path(p['runtime_source_archive']);payload=runtime_source_archive()
    if target.exists():
        if json.loads(target.read_bytes())!=p or archive.read_bytes()!=payload:
            raise ValueError('Existing task design/source differs; no overwrite')
        return target
    history=p['original_OFFLINE_preparation_and_HOLD_history']
    allowed={Path(history[key]) for key in ('original_protocol','original_HOLD','original_source_archive')} if history else set()
    if context.output.exists() and set(context.output.iterdir())-allowed:
        raise ValueError('Existing/unfinalized task directory retained; no overwrite')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    evidence.write_exclusive(target,p)
    return target


def anonymous_source_readback(context,namespace_present):
    """Record authored Sdf fields AND composed values in one installed SDK.

    Adding an unvalued property spec need not change a composed value or its
    hash when the schema already supplies that property. Do not infer authored
    metadata from a hash difference. These are anonymous stages, never Kit's.
    """
    from pxr import Usd,UsdShade
    helper=shared.completed_static.stationary.reference
    stage=Usd.Stage.CreateInMemory('H5_ANONYMOUS_SOURCE_FIELDS_NOT_A_LIVE_TRIAL.usda')
    UsdShade.Material.Define(stage,context.preparation.old.constructor.MATERIAL)
    made=context.preparation.old.constructor_for(context.cohort.ROBOT)(stage,context.protocol['start_xyyaw'])
    if namespace_present:
        shared._bound(helper.add_metadata_to_anonymous_layer,ROBOT=context.cohort.ROBOT)(stage.GetRootLayer())
    attribute=stage.GetPrimAtPath(context.cohort.ROBOT).GetAttribute('isaac:namespace')
    return dict(source_constructor_readback=made,fixed_configuration_rows=context.stationary.actor_configuration(stage),
        anonymous_authored_layer_snapshot=helper.fields.snapshot(stage.GetRootLayer()),
        namespace_resolved_value=helper.fields.semantic(attribute.Get()) if attribute else None,
        namespace_unvalued_declaration_in_reference=namespace_present,
        created_in_anonymous_noncontext_stage_only=True)


def expected_namespace_spec():
    return dict(spec_type='AttributeSpec',info=dict(custom=False,displayName='Namespace',
        documentation='Namespace of the prim in Isaac Sim',typeName='string',
        variability=dict(type='Variability',value='Sdf.VariabilityVarying')))


def source_reference_checks(protocol,plain,namespaced):
    """Anonymous USD source checks, not a live pose/support/drive evaluation."""
    made=plain['source_constructor_readback'];robot=protocol['new_robot_root']
    bodies=made['rigid_bodies'];names=made['revolute_joint_names'];seed=made['actual_source_authored_joint_seed']
    first,last=plain['fixed_configuration_rows'],namespaced['fixed_configuration_rows']
    property_path=robot+'.isaac:namespace'
    a,b=plain['anonymous_authored_layer_snapshot'],namespaced['anonymous_authored_layer_snapshot']
    only_expected_metadata=(property_path not in a and b.get(property_path)==expected_namespace_spec()
        and a=={path:row for path,row in b.items() if path!=property_path})
    pose=made['start_xyyaw'];expected=protocol['start_xyyaw']
    relative=shared.execution.wrap(pose[2]-expected[2])
    checks=dict(anonymous_reference_only=plain['created_in_anonymous_noncontext_stage_only'] is True
            and namespaced['created_in_anonymous_noncontext_stage_only'] is True,
        exact_proposed_start_and_source_height=math.dist(pose[:2],expected[:2])<=1e-10
            and abs(relative)<=1e-10 and made['base_z_m']==0.,
        thirteen_original_positive_mass_bodies=len(bodies)==13 and len({b['path'] for b in bodies})==13
            and all(b['path'].startswith(robot+'/') and math.isfinite(b['mass_kg']) and b['mass_kg']>0. for b in bodies),
        original_total_mass_265kg=made['simulation_mass_kg']==265. and sum(b['mass_kg'] for b in bodies)==265.,
        twelve_original_revolute_joints_and_seed=len(names)==12 and len(set(names))==12
            and set(names)==set(protocol['expected_nominal_motor_baseline']['joint_names']) and set(seed)==set(names),
        unchanged_source_link_poses_and_all_joint_relationships=made['all_link_relative_poses_match_source'] is True
            and made['all_joint_body_relationships_remapped_to_new_robot'] is True,
        six_original_wheel_capsules=len(made['six_wheel_capsule_paths'])==6
            and len(set(made['six_wheel_capsule_paths']))==6
            and all(p.startswith(robot+'/') for p in made['six_wheel_capsule_paths']),
        all_145_source_configuration_rows_retained=len(first)==len(last)==145 and set(first)==set(last),
        SDK_unvalued_namespace_changes_only_actor_metadata=set(first)==set(last)
            and set(p for p in first if first[p]!=last[p])<={robot} and only_expected_metadata
            and plain['namespace_unvalued_declaration_in_reference'] is False
            and namespaced['namespace_unvalued_declaration_in_reference'] is True,
        source_constructor_readback_identical_before_after_namespace=made==namespaced['source_constructor_readback'],
        all_original_source_checks_explicitly_not_a_support_trial=made['static_support_or_drive_validated'] is False)
    return checks


def export_anonymous_source_reference(protocol_path):
    """Use installed USD only, never SimulationApp, Kit, physics or live USD."""
    path=Path(protocol_path).resolve();p=json.loads(path.read_bytes())
    expected=build_protocol(p['frozen_bundle'],p['nominal_model_screen'],p['required_prior_result'],cohort=p['cohort'],method=p['method'])
    context=task_context(p)
    if path!=context.protocol_path or p!=expected or sha(p['runtime_source_archive'])!=p['runtime_source_archive_sha256']:
        raise ValueError('Exact unchanged own offline task design required; no repair or adoption')
    plain=anonymous_source_readback(context,False);namespaced=anonymous_source_readback(context,True)
    checks=source_reference_checks(p,plain,namespaced)
    report=dict(version='H5_ANONYMOUS_AUTHORED_FIELDS_AND_COMPOSED_CONFIGURATION_REVIEW_v2',
        task_protocol=str(path),task_protocol_sha256=sha(path),checks=checks,
        status='ANONYMOUS_USD_SOURCE_CHECKS_COMPLETE_NOT_LIVE_STAGING_STATIC_OR_DRIVE_ADMISSION'
            if all(checks.values()) else 'HOLD_ANONYMOUS_SOURCE_CONFIGURATION_NO_LIVE_ADMISSION',
        anonymous_source_without_SDK_namespace=plain,anonymous_source_with_unvalued_SDK_namespace=namespaced,
        live_stage_robot_or_timeline_accessed=False,actual_motion_authorized=False,new_PhysX_trial_count=0,
        original_H4_state_or_archived_H3_state_not_restored_or_modified=True,
        cooked_support_contact_caster_response_and_rendered_pixels_NOT_verified=True,
        original_OFFLINE_HOLD_preserved=p['original_OFFLINE_preparation_and_HOLD_history'],
        input_sha256=merge_pins(p['input_sha256'],{str(path):sha(path),p['runtime_source_archive']:sha(p['runtime_source_archive'])}))
    verify_pins(report['input_sha256']);target=context.output/'anonymous_source_readback_v2.json'
    if target.exists():
        if json.loads(target.read_bytes())!=report:raise ValueError('Original source audit differs; no overwrite')
        return target
    evidence.write_exclusive(target,report)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--prior-result',type=Path,required=True)
    parser.add_argument('--cohort',required=True)
    parser.add_argument('--method',required=True,choices=shared.execution.METHODS)
    parser.add_argument('--source-reference',action='store_true',help='Also inspect anonymous USD using the installed SDK only; never start Kit/physics')
    args=parser.parse_args()
    target=export_protocol(args.bundle,args.model,args.prior_result,cohort=args.cohort,method=args.method)
    print('Saved H5 preparation/static-query OFFLINE design:',target,flush=True)
    if args.source_reference:
        source=export_anonymous_source_reference(target)
        print('Saved installed-SDK anonymous source review:',source,flush=True)
    print('No live entry, Stop, staging, Play, static, route, retry or physical admission.',flush=True)


if __name__=='__main__':main()
