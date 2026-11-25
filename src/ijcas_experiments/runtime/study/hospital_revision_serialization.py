"""Typed float32 angular writeback proof and one pre-route setup recovery.

The recorded-state proof already recognizes exact radians -> vector3f degrees
serialization. Apply that SAME rule to FIRST writeback, retaining every
physical check and every numerical residual limit. Old sources stay frozen.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
import ast
import asyncio
import copy
import inspect
import math
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import hospital_revision_placement as placement
import hospital_revision_execution as execution
import hospital_revision_initial_conditions as initial
import hospital_revision_followthrough as follow

runtime=placement.runtime
campaign=placement.campaign
CELL='H4_B1_multimode_kinematic_ramp_s02'
ROOT=campaign.OUTPUT/'typed_writeback_requalification'
QUEUE=campaign.OUTPUT/'execution_queue_v4_typed_writeback'
POLICY=campaign.OUTPUT/'typed_writeback_resolution_policy.json'


def exact_degree_value(value,sdk_value,type_name):
    """Same exact double-rounded OR native-float multiply rule as recorded proof."""
    value=np.asarray(value,float);sdk_value=np.asarray(sdk_value,float)
    if value.shape!=(3,) or sdk_value.shape!=(3,) or not np.isfinite([value,sdk_value]).all():
        raise ValueError('Complete finite three-component angular vectors required')
    rounded=np.degrees(sdk_value).astype(np.float32).astype(float)
    native=sdk_value.astype(np.float32)
    native_multiply=(native*np.float32(180./math.pi)).astype(float)
    exact_native=np.array_equal(native.astype(float),sdk_value) and np.array_equal(value,native_multiply)
    if str(type_name)=='vector3f' and (np.array_equal(value,rounded) or exact_native):
        return sdk_value.copy()
    return np.radians(value)


def first_writeback_binding():
    original=runtime.retained.prepared_source_writeback_review
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    before=copy.deepcopy(tree)
    found=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.If) and ast.unparse(node.test)=="field == 'physics:angularVelocity'"
                and len(node.body)==1 and ast.unparse(node.body[0])=='value = np.radians(value)'):
            found.append(node)
    if len(found)!=1:raise ValueError('One original first-writeback unit conversion required')
    original_body=copy.deepcopy(found[0].body)
    found[0].body=ast.parse("value = _exact_degree_value(value, velocities[i,part], prim.GetAttribute(field).GetTypeName())").body
    undo=copy.deepcopy(tree)
    node=next(n for n in ast.walk(undo) if isinstance(n,ast.If)
        and ast.unparse(n.test)=="field == 'physics:angularVelocity'")
    node.body=original_body
    if ast.dump(undo)!=ast.dump(before):
        raise ValueError('Every layer/state/metadata/geometry/tolerance check must stay identical')
    scope=dict(original.__globals__,_exact_degree_value=exact_degree_value)
    exec(compile(ast.fix_missing_locations(tree),'<first_writeback_same_exact_recorded_float_rule>','exec'),scope)
    checked=scope[original.__name__]
    def review(*args,**kwargs):
        result=checked(*args,**kwargs)
        result['angular_unit_encoding_policy']='EXACT_VECTOR3F_DEGREES_FROM_SDK_RADIANS; SAME_EXISTING_RECORDED_WRITEBACK_RULE'
        result['physical_or_numerical_residual_limits_increased']=False
        return result
    return review


FIRST_REVIEW=first_writeback_binding()


def retained_proxy(context):
    helpers=getattr(context,'initial_condition_source_helpers',context.placement_source_helpers)
    return SimpleNamespace(**dict(vars(runtime.retained),shared=helpers,
        prepared_source_writeback_review=FIRST_REVIEW))


def checked_guard(self,row,info):
    return runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.guard,
        retained=retained_proxy(self.context))(self,row,info)


def checked_post_layers(self):
    return runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.check_post_observation_layers,
        retained=retained_proxy(self.context))(self)


class SerializationStaticHost(placement.PlacementStaticHost):
    guard=checked_guard
    check_post_observation_layers=checked_post_layers


class SerializationInitialHost(initial.InitialConditionStaticHost):
    guard=checked_guard
    check_post_observation_layers=checked_post_layers


def fixed_context(folder,actor,geometry):
    if Path(folder).is_relative_to(ROOT):actor+='_typed_setup02'
    return placement.context_for(folder,actor,geometry)


def fixed_runtime():
    return SimpleNamespace(**dict(vars(runtime),context_for=fixed_context,
        prepare_source=placement.prepare_source,ControlledStaticHost=SerializationStaticHost))


def export_policy():
    old=campaign.OUTPUT/'live'/CELL
    value=dict(version='FIRST_WRITEBACK_EXACT_TYPED_FLOAT32_v1',
        source_static=str(old/'static/held_static_readback.json'),
        source_static_sha256=runtime.sha(old/'static/held_static_readback.json'),
        cell_id=CELL,qualified_output=str(_ijcas_path(ROOT, 'live', CELL)),
        original_static_outcome_preserved_NOT_promoted=True,
        original_route_attempts=0,maximum_additional_setup_static_trials=1,
        maximum_FIRST_method_route_trials=1,no_retry_of_any_started_route=True,
        only_exact_serialization_proven_pre_actuation_infrastructure_failure_eligible=True,
        rule_not_specific_to_P_or_a_favorable_outcome=True,
        unchanged_model_controller_physical_limits_speed_clocks_and_success_criteria=True,
        user_authority=runtime.APPROVAL,source_sha256=runtime.sha(__file__))
    if POLICY.exists():
        if runtime.read(POLICY)!=value:raise ValueError('Preserve the declared setup-only resolution policy')
    else:runtime.evidence.write_exclusive(POLICY,value)
    return value


def resolved_outcome(path):
    path=Path(path)
    original=runtime.read(path)
    if path==campaign.OUTPUT/'live'/CELL/'trial_outcome.json':
        if original.get('actual_route_started') or original['status']!='STATIC_NOT_PASS_NO_ROUTE':
            raise ValueError('Never replace a started method route or a different failure')
        policy=runtime.read(POLICY)
        if runtime.sha(policy['source_static'])!=policy['source_static_sha256']:
            raise ValueError('Original failed static bytes changed')
        value=runtime.read(Path(policy['qualified_output'])/'trial_outcome.json')
        return dict(value,technical_setup_incident=policy['source_static'],
            setup_only_resolution_policy=str(POLICY),original_method_route_replaced=False)
    return original


def queue_runtime():
    return SimpleNamespace(**dict(vars(runtime),read=resolved_outcome))


async def prove_current_setup_boundary():
    """GET only: all13/12, exact typed state writeback, no motor write occurred."""
    old=campaign.OUTPUT/'live'/CELL
    raw=runtime.read(old/'static/held_static_readback.json')
    if (raw['new_route_trials']!=0 or raw['sample_count']!=0
            or not raw['checks'].get('no_motor_write_restoration_required')
            or not raw['checks'].get('actual_Pause_confirmed')
            or not raw.get('error','').startswith('ValueError: First writeback disagrees')):
        raise ValueError('Only the diagnosed pre-actuation encoding abort may qualify')
    sdk=runtime.retained.runtime.live_sdk()
    stage=sdk.omni.usd.get_context().get_stage();timeline=sdk.omni.timeline.get_timeline_interface()
    state=lambda:runtime.evidence.timeline_state(timeline)
    saved=raw['actual_final_timeline']
    if state()!=saved or saved['playing'] or saved['stopped']:
        raise ValueError('Same stable actual Pause required; no automatic timeline command')
    packet=runtime.read(raw['protocol']);prep=runtime.read(raw['preparation_record'])
    view=sdk.SimulationManager.get_physics_sim_view().create_articulation_view(packet['actor']+'/base_link')
    snapshot=runtime.shared.preparation.old.tensor_snapshot(view)
    if snapshot!=raw['actual_final_readback']['tensor']:
        raise ValueError('Exact saved source13link/12joint tensor changed')
    baseline=prep['expected_nominal_motor_baseline']
    if (snapshot['names']!=baseline['joint_names'] or len(snapshot['link_names'])!=13
            or any(snapshot['gains'][k]!=baseline[k] for k in ('kp','kd','max_efforts'))
            or snapshot['joint_position_targets_rad']!=[0.]*12
            or snapshot['joint_velocity_targets_rad_s']!=[0.]*12):
        raise ValueError('Complete original all12 motor settings and zero targets required')
    before=runtime.read(old/'prepared_layer_texts.json')
    after=dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString())
    types=raw['preflight']['prePlay_source_state_types']
    proof=FIRST_REVIEW(stage,before,after,snapshot,packet['actor'],types)
    def stable():
        if sdk.omni.usd.get_context().get_stage()!=stage or state()!=saved:
            raise ValueError('GET stage/Pause boundary changed')
    expected=runtime.read(runtime.retained.H6/'actual_common_environment_rows.json')
    await runtime.bootstrap.verify_rows(stage,expected['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async,unchanged=stable)
    if runtime.shared.preparation.old.tensor_snapshot(view)!=snapshot:
        raise ValueError('Actual tensor changed during GET-only environment proof')
    value=dict(status='GET_ONLY_TYPED_SETUP_ABORT_ARCHIVE_BOUNDARY_NOT_STATIC_PASS_OR_ROUTE_ADMISSION',
        actual_final_timeline=saved,actual_final_readback=dict(tensor=snapshot),
        checks=dict(original_gains_targets_restored=True),
        original_static_verdict_unchanged='HOLD',original_static=str(old/'static/held_static_readback.json'),
        first_writeback_numeric_proof=proof,all4997_protected_rows_verified=True,
        no_live_stage_timeline_gain_target_or_actor_write=True)
    path=_ijcas_path(ROOT, 'current_setup_boundary_GET.json')
    runtime.evidence.write_exclusive(path,value)
    return path


def initial_cell_binding(cell):
    return runtime.bind(initial.cell_bindings,private_source_context=follow.initial_context,
        InitialConditionStaticHost=SerializationInitialHost,__file__=__file__)(cell)


async def run_all(*,authorized=False):
    if authorized is not True:raise ValueError('Current reviewer-experiment authority required')
    export_policy()
    runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'reservation.json'),dict(
        physical_route_retry=False,maximum_current_setup_requalification=1,
        source_sha256=runtime.sha(__file__),authority=runtime.APPROVAL))
    runtime.lc.freeze_workflow_sources(_ijcas_path(ROOT, 'sources.zip'),{str(Path(p).resolve()):runtime.sha(p)
        for p in (__file__,placement.__file__,execution.__file__,initial.__file__,follow.__file__)})
    try:
        boundary=await prove_current_setup_boundary()
        current_campaign=SimpleNamespace(**dict(vars(campaign),OUTPUT=ROOT))
        run=runtime.bind(placement.live.run_cell,runtime=fixed_runtime(),
            campaign=current_campaign,__file__=__file__)
        outcome=await run(CELL,prior_result=boundary,authorized=True,model_policy_path=execution.POLICY)
        result_path=Path(outcome['report'])
        if not execution.restored_boundary(runtime.read(result_path)):
            raise ValueError('Requalified static/route actual Pause and restoration not verified')
        if outcome.get('actual_route_started'):
            runtime.evidence.write_exclusive(result_path.parent/'controlled_actual_metrics.json',execution.metrics.metrics(result_path))
        print('[REVIEW SETUP QUALIFICATION RESULT]',outcome,flush=True)
        live_proxy=SimpleNamespace(**dict(vars(placement.live),run_cell=runtime.bind(
            placement.live.run_cell,runtime=fixed_runtime(),__file__=__file__)))
        main=await runtime.bind(execution.run,live=live_proxy,runtime=queue_runtime(),QUEUE=QUEUE)(
            prior_result=result_path,authorized=True)
        if main['status']!='FINITE_QUEUE_FINISHED':raise ValueError('Main queue did not finish')
        joint=await runtime.bind(initial.run,cell_bindings=initial_cell_binding,__file__=__file__)(
            prior_result=Path(main['last_actual_report']),authorized=True)
        runtime.evidence.write_exclusive(initial.OUTPUT/'after_primary_result.json',joint)
        result=dict(status='ALL_DECLARED_REVISION_PHYSICAL_BLOCKS_FINISHED',
            main=main,initial_conditions=joint,all_negative_outcomes_preserved=True)
        runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'all_blocks_completed.json'),result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(_ijcas_path(ROOT, 'stopped.json'),dict(
            error=type(error).__name__+': '+str(error),no_automatic_recovery_or_retry=True))
        raise


def dispatch():
    from hospital_revision_runner import remote
    path=Path(__file__).resolve()
    command=("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "old=getattr(builtins,'ijcas_hospital_latest_task_experiment',None)\n"
        "assert old is None or old.done(),'Another actual experiment is active'\n"
        f"p=Path({str(path)!r})\n"
        "m=types.ModuleType('hospital_revision_serialization');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.run_all(authorized=True))\n"
        "builtins.ijcas_hospital_latest_task_experiment=job\nbuiltins.ijcas_revision_primary_task=job\n"
        "print({'typed_setup_qualification_and_remaining_queue_scheduled':True,'started_method_routes_retried':False})\n")
    runtime.evidence.write_exclusive(campaign.OUTPUT/'typed_writeback_dispatch.json',dict(
        exact_command=command,source_sha256=runtime.sha(path),preserved_stopped_queue=str(placement.CONTINUATION)))
    return remote(command)


if __name__=='__main__':
    print(dispatch())
