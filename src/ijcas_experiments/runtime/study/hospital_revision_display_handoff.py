"""Finish the unstarted B1 static after an exact, nonphysical display review.

Never normalize a physical subtree or change the monitoring policy. Adopt
one verified current camera/lighting baseline on the SAME untouched actor,
then run the original strict static and route hosts. Retain the old pre-Play
abort and completed B0 negative. No Stop, constructor, camera or motor command
is sent by this handoff. All subsequent cells use the original fixed host.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import hospital_revision_session_resume as previous

experiment=previous.experiment
runtime=previous.runtime
OUTPUT=experiment.OUTPUT/'display_handoff_r01'
CELL='H1_B1_multimode_kinematic_ramp_f01'
KEY='ijcas_revision_n3_display_handoff_r01'
GET=previous.OUTPUT/'B1_static_boundary_GET_r01.json'
LIGHT=Path('/home/jk/miniconda3/envs/isaac_env/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.viewport.menubar.lighting-107.3.1+107.3.0/data/usd/Default.usda')
AUTHORITY='사용자: 구조적 문제 직접 해결, 아이작 다시 켜서 이어서 진행; 미시작 정적·동일 고정 비교만'


def finite_vector(value,kind,size):
    return (isinstance(value,dict) and value.get('type')==kind and len(value.get('values',[]))==size
        and all(type(x) in (int,float) and math.isfinite(x) for x in value['values']))


def validate_display_row(label,row):
    p,f,b,a=(row[k] for k in ('path','field','before','after'))
    if label=='Root' and p=='/' and f=='customLayerData':
        old,new=copy.deepcopy(b),copy.deepcopy(a)
        if old['cameraSettings'].pop('boundCamera')!='/OmniverseKit_Persp' or new['cameraSettings'].pop('boundCamera')!='/OmniverseKit_Top' or old!=new:
            raise ValueError('Only the observed bound viewport camera may differ in layer metadata')
        return
    render='/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0'
    if label=='Root' and p==render+'.camera' and f=='targetPaths':
        expected=lambda camera:dict(type='PathListOp',value=f'SdfPathListOp(Explicit Items: [{camera}])')
        if b!=expected('/OmniverseKit_Persp') or a!=expected('/OmniverseKit_Top'):raise ValueError('Exact reviewed RenderProduct camera relation required')
        return
    if label!='Session':raise ValueError('Unreviewed Root display or physical field')
    if p==render+'.viewPickingId' and f=='default' and type(b) is int and type(a) is int:return
    camera_properties={
        '/OmniverseKit_Persp.omni:kit:centerOfInterest','/OmniverseKit_Persp.xformOp:translate',
        '/OmniverseKit_Top.omni:kit:centerOfInterest','/OmniverseKit_Top.xformOp:translate'}
    if p in camera_properties and f=='default' and all(finite_vector(v,'Vec3d',3) for v in (b,a)):return
    if p in ('/OmniverseKit_Top.horizontalAperture','/OmniverseKit_Top.verticalAperture') and f=='default' and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (b,a)):return
    if b!={'field_is_not_authored':True}:raise ValueError('Only the observed new nonphysical light opinions')
    prims=('/Environment','/Environment/defaultLight','/OmniKit_Viewport_LightRig')
    if p in prims and ((f=='__spec_type__' and a=='PrimSpec') or (f=='specifier' and a==dict(type='Specifier',value='Sdf.SpecifierOver'))):return
    if p=='/OmniKit_Viewport_LightRig':
        if f in ('hide_in_stage_window','no_delete') and a is True:return
        if f=='references' and a==dict(type='ReferenceListOp',value=f'SdfReferenceListOp(Explicit Items: [SdfReference({LIGHT}, , SdfLayerOffset(0, 1), {{}})])'):return
    properties={
        '/Environment/defaultLight.visibility':('token','invisible','Sdf.VariabilityVarying'),
        '/OmniKit_Viewport_LightRig.xformOpOrder':('token[]',dict(type='TokenArray',values=['xformOp:transform']),'Sdf.VariabilityUniform'),
        '/OmniKit_Viewport_LightRig.xformOp:transform':('matrix4d',None,'Sdf.VariabilityVarying')}
    if p in properties:
        kind,default,variability=properties[p]
        if f=='__spec_type__' and a=='AttributeSpec' or f=='custom' and a is False or f=='typeName' and a==kind or f=='variability' and a==dict(type='Variability',value=variability):return
        if f=='default':
            if kind!='matrix4d' and a==default:return
            if kind=='matrix4d' and isinstance(a,dict) and a.get('type')=='Matrix4d' and len(a.get('values',[]))==4 and all(finite_vector(v,'Vec4d',4) for v in a['values']):return
    raise ValueError('Unreviewed display/physical delta: '+p+':'+f)


def definition():
    block=experiment.campaign();block.verify_sources();previous.verify()
    folder=block.OUTPUT/'live'/CELL
    failed=runtime.read(folder/'static/held_static_readback.json')
    if (failed['error']!='ValueError: Unreviewed field or spec metadata: /Environment:__spec_type__'
            or failed['sample_count']!=0 or failed['new_static_trials']!=0 or failed['new_route_trials']!=0
            or failed['first_Play_transition']['requested'] or failed['first_Play_transition']['observed']
            or failed['actual_final_timeline']!=dict(playing=False,stopped=True,current_time_s=0.)):
        raise ValueError('Only the exact unstarted zero-Play B1 static may continue')
    expected=json.loads(runtime.read(GET)['result']['output'])
    for label,rows in expected['diff'].items():
        for row in rows:validate_display_row(label,row)
    paths=(GET,folder/'preparation.json',folder/'prepared_layer_texts.json',
        folder/'display_initialization/renderer_initialization.json',folder/'static/held_static_readback.json',
        block.OUTPUT/'stopped.json',LIGHT,Path(__file__))
    return dict(cell_id=CELL,user_authority=AUTHORITY,
        old_prePlay_abort_retained=True,same_actor_first_actual_static_and_method_route=True,
        no_Stop_restaging_display_refresh_camera_or_motor_command_in_handoff=True,
        controller_model_physics_goals_speed_and_acceptance_unchanged=True,
        ground_truth_comparison_only_NOT_a_claim_of_sensor_image_invariance=True,
        expected_display_differences=expected['diff'],
        input_sha256={str(p.resolve()):runtime.sha(p) for p in paths})


def verify():
    experiment.campaign().verify_sources();previous.verify()
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'definition.json')['input_sha256'])


async def adopt_same_source(context,sdk,geometry,*,prior_result=None,preview=False):
    from pxr import Sdf,Usd,UsdGeom,UsdPhysics,UsdRender
    stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    if runtime.evidence.timeline_state(timeline)!=dict(playing=False,stopped=True,current_time_s=0.):
        raise ValueError('Same untouched zero-time source required; no reset')
    block=experiment.campaign();old=block.OUTPUT/'live'/CELL;prep=runtime.read(old/'preparation.json')
    if context.cohort.ROBOT!=prep['new_robot_root']:raise ValueError('The same existing source actor is mandatory')
    before=runtime.read(old/'prepared_layer_texts.json')
    current=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    deltas={k:runtime.shared.layer_field_differences(before[k],current[k]) for k in before}
    if deltas!=runtime.read(OUTPUT/'definition.json')['expected_display_differences']:
        raise ValueError('Current layers differ from the exact reviewed GET; no additional waiver')
    for label,rows in deltas.items():
        for row in rows:validate_display_row(label,row)
    # Composed physical identity is checked, not just path spelling.
    display_roots=('/Environment','/OmniKit_Viewport_LightRig','/OmniverseKit_Persp','/OmniverseKit_Top')
    display=[]
    for prim in stage.Traverse():
        if not any(str(prim.GetPath())==p or str(prim.GetPath()).startswith(p+'/') for p in display_roots):continue
        schemas=list(prim.GetAppliedSchemas())
        if prim.GetTypeName() not in ('Xform','Scope','DistantLight','DomeLight','Camera') or any('Physics' in s or 'Collision' in s or 'RigidBody' in s or 'Scripting' in s for s in schemas):
            raise ValueError('Reviewed display hierarchy must contain only cameras/lights, no physics/geometry/scripts')
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Physical display subtree is not admissible')
        display.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),schemas=schemas))
    layer=Sdf.Layer.CreateAnonymous('.usda');layer.ImportFromString(current['Session'])
    refs=list(layer.GetPrimAtPath('/OmniKit_Viewport_LightRig').referenceList.GetAddedOrExplicitItems())
    if len(refs)!=1 or refs[0].assetPath!=str(LIGHT) or str(refs[0].primPath)!='':raise ValueError('Exact installed light-only reference required')
    for camera in ('/OmniverseKit_Persp','/OmniverseKit_Top'):
        if not stage.GetPrimAtPath(camera).IsA(UsdGeom.Camera):raise ValueError('Actual typed viewport camera required')
    render=stage.GetPrimAtPath('/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0')
    if not render.IsA(UsdRender.Product):raise ValueError('Actual viewport RenderProduct required')
    bodies=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies)!=13 or any(not p.startswith(context.cohort.ROBOT+'/') for p in bodies):raise ValueError('Same13 source bodies only')
    if context.stationary.actor_configuration(stage)!=prep['source_reference']['fixed_configuration_rows']:
        raise ValueError('Original all145 physical/source settings must remain exact')
    if context.stationary.critical_environment(stage,prep)!=prep['critical_environment']:
        raise ValueError('Original physics/material/support settings must remain exact')
    source=Usd.Stage.Open(str(context.preparation.ASSET))
    constructor=context.preparation.old.constructor_for(context.cohort.ROBOT)
    validate=constructor.__globals__['validate_new_robot']
    if validate(stage,source,geometry['start_xyyaw'])!=prep['live_new_robot_readback']:
        raise ValueError('Same complete source constructor readback required')
    receipt=dict(actual_display_hierarchy=display,exact_reviewed_differences=deltas,
        same13_bodies_all145_settings_and_source_readback=True,no_live_layer_or_motor_edit=True,
        old_prePlay_abort=str(old/'static/held_static_readback.json'))
    if preview:return receipt
    runtime.evidence.write_exclusive(context.output/'display_handoff.json',receipt)
    prep=copy.deepcopy(prep)
    prep.update(Stop_calls=0,reviewed_display_handoff=str(context.output/'display_handoff.json'),
        prior_boundary=str(context.output/'prior_boundary.json'))
    prep.update({k+'_sha256':hashlib.sha256(v.encode()).hexdigest() for k,v in current.items()})
    runtime.evidence.write_exclusive(context.output/'prior_boundary.json',dict(
        timeline=dict(playing=False,stopped=True,current_time_s=0.),same_prepared_actor=True,
        no_physics_started_in_aborted_static=True,original_preparation=str(old)))
    runtime.evidence.write_exclusive(context.preparation.LIVE,prep)
    runtime.evidence.write_exclusive(context.output/'prepared_layer_texts.json',current)
    runtime.evidence.write_exclusive(context.output/'display_initialization/renderer_initialization.json',
        runtime.read(old/'display_initialization/renderer_initialization.json'))
    return prep


def first_binding(block,cell):
    call=block.bound_cell(cell);old=call.__globals__['runtime']
    private=SimpleNamespace(**dict(vars(old),prepare_source=adopt_same_source))
    campaign=SimpleNamespace(**dict(vars(call.__globals__['campaign']),OUTPUT=OUTPUT/'same_actor'))
    return runtime.bind(call,runtime=private,campaign=campaign,__file__=__file__)


async def run(parent,*,authorized=False):
    if authorized is not True:raise ValueError('Current autonomous continuation authority required')
    import builtins
    verify();block=experiment.campaign()
    if (not parent.done() or parent.exception() is None
            or getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not parent):
        raise ValueError('Exact stopped prior owner required')
    cells=runtime.read(block.EXECUTION_PROTOCOL)['main_cells'];existing=[];pending=[]
    for cell in cells:
        path=block.OUTPUT/'outcomes'/(cell['cell_id']+'.json')
        (existing if path.exists() else pending).append(runtime.read(path) if path.exists() else cell)
    if len(existing)!=1 or existing[0]['cell_id']!=previous.FIRST or pending[0]['cell_id']!=CELL:
        raise ValueError('Only the exact first B1 pre-Play boundary may resume')
    task=asyncio.current_task();builtins.ijcas_hospital_latest_task_experiment=task;builtins.ijcas_revision_primary_task=task
    last=Path(existing[-1]['report']);results=list(existing)
    runtime.evidence.write_exclusive(OUTPUT/'execution_reservation.json',dict(
        remaining_declared_cells=len(pending),old_actual_negative_preserved=True,
        old_static_Play_calls=0,no_automatic_retry=True,user_authority=AUTHORITY))
    try:
        for index,cell in enumerate(pending):
            verify();name=cell['cell_id'];folder=block.folder_for(name)
            planned=runtime.read(folder/'final_planning_receipt.json')
            if not planned['independently_valid_reference']:
                value=dict(cell_id=name,status='PLANNING_FAILURE_NO_PHYSICAL_RUN' if not planned['plan_success'] else 'INDEPENDENT_REFERENCE_AUDIT_REJECTED',
                    actual_route_started=False,actual_verdict='PLAN_FAILURE' if not planned['plan_success'] else 'REFERENCE_REJECTED',
                    planning_summary=str(folder/'summary.json'),metrics=None)
            else:
                print('[N3 HANDOFF ACTUAL START]',name,flush=True)
                call=first_binding(block,cell) if index==0 else runtime.bind(block.bound_cell(cell),__file__=__file__)
                value=await call(name,prior_result=last,authorized=True,model_policy_path=block.policy_file())
                last=Path(value['report'])
                if not block.safe_restored_boundary(runtime.read(last)):raise ValueError('Unresolved actual boundary; stop without automatic retry')
                value=dict(value,metrics=block.metrics.metrics(last) if value['actual_route_started'] else None)
            value.update(experiment_role=cell['experiment_role'],controller_version=block.control.VERSION,
                predictive_history_policy=cell['predictive_history_policy'],final_fixed_version_trial=True,
                attempt_replaced=False,reviewed_unstarted_display_handoff=str(OUTPUT/'definition.json') if index==0 else None)
            runtime.evidence.write_exclusive(block.OUTPUT/'outcomes'/(name+'.json'),value);results.append(value)
            print('[N3 HANDOFF RESULT]',name,value['actual_verdict'],flush=True)
            await asyncio.sleep(0)
        runtime.evidence.write_exclusive(block.OUTPUT/'completed.json',dict(
            status='FINAL_N3_ALL_DECLARED_CELLS_FINISHED',results=results,last_actual_report=str(last),
            all_negative_results_preserved=True,automatic_retry=False,reviewed_historical_stop=str(block.OUTPUT/'stopped.json')))
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT/'stopped.json',dict(error=type(error).__name__+': '+str(error),results=results,last_actual_report=str(last),no_automatic_retry=True))
        raise


def dispatch():
    value=definition();runtime.evidence.write_exclusive(OUTPUT/'definition.json',value)
    runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',value['input_sha256'])
    source=("import importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        "import hospital_revision_display_handoff as _handoff\n"
        f"assert not hasattr(builtins,{KEY!r}),'Already reserved'\n"
        f"builtins.{KEY}=asyncio.ensure_future(_handoff.run(builtins.{previous.KEY},authorized=True))\n"
        "print({'same_B1_source_unstarted_static_then_conditional_route_scheduled':True,'Stop_or_restage':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch.json',dict(source=source,user_authority=AUTHORITY))
    return experiment.original.predecessor.control.base.shared.runner.remote(source)


def all_steps():
    print(dispatch(),flush=True);block=experiment.campaign();n=-1
    while not (block.OUTPUT/'completed.json').exists():
        if (OUTPUT/'stopped.json').exists():raise ValueError('Display continuation stopped; inspect without blind retry')
        snapshot=block.results_snapshot()
        if snapshot['accounted_cells']!=n:
            n=snapshot['accounted_cells'];print('[N3 CURRENT PROGRESS]',n,'/78',flush=True)
            if n:
                row=next(r for r in reversed(snapshot['rows']) if r['accounted'])
                print('[N3 LATEST RESULT]',row,flush=True)
        time.sleep(30.)
    result=block.export_results()
    import hospital_revision_final_comparison_analysis as analysis
    detail=analysis.export()
    runtime.evidence.write_exclusive(block.OUTPUT/'all_steps_completed.json',dict(
        status='FINAL_N3_EXECUTION_AND_REPORTING_FINISHED',analysis_folder=str(result),
        detailed_feedback_analysis=str(detail),all_failures_preserved=True,reviewed_display_handoff=str(OUTPUT/'definition.json')))
    print('[N3 FINISHED]',result,detail,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('all','dispatch','definition'))
    action=parser.parse_args().action
    if action=='all':all_steps()
    elif action=='dispatch':print(dispatch())
    else:print(definition())
