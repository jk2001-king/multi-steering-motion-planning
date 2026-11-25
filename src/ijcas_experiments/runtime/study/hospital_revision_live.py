"""Predeclared primary-trial provenance over the unchanged physical hosts.

Private AST bindings change only current authority and primary-vs-engineering
labels. Undo must recover the complete original AST. Physical observations,
controller, commands, limits, checks, cleanup and verdict logic stay intact.
"""
import ast
import copy
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import numpy as np
import hospital_revision_campaign as campaign
import hospital_revision_planning as planning

runtime=campaign.runtime
binding=campaign.binding


def primary_initializer():
    original=runtime.lc.RetainedMissionSDKHost.__init__
    source=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=copy.deepcopy(source)
    body=tree.body[0].body
    found=[i for i,n in enumerate(body) if isinstance(n,ast.ImportFrom)
        and n.module=='hospital_mission_sequence'
        and 'checked_route_lifecycle_contract' in [a.name for a in n.names]]
    if len(found)!=1:
        raise ValueError('One original authority import required')
    index=found[0]+1
    body[index:index]=ast.parse('checked_route_lifecycle_contract = _controlled_contract\n'
                               'REMAINING_NO_NAV2_APPROVAL = _controlled_approval').body
    changed=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and len(node.ops)==1 and isinstance(node.ops[0],ast.IsNot)
                and isinstance(node.left,ast.Call) and isinstance(node.left.func,ast.Attribute)
                and isinstance(node.left.func.value,ast.Name) and node.left.func.value.id=='protocol'
                and node.left.func.attr=='get' and len(node.left.args)==1
                and isinstance(node.left.args[0],ast.Constant) and node.left.args[0].value=='use_for_final_statistics'
                and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)
                and node.comparators[0].value is False):
            node.comparators[0].value=True
            changed.append(node)
    if len(changed)!=1:
        raise ValueError('One original role-label check required')
    changed[0].comparators[0].value=False
    undo=copy.deepcopy(tree)
    del undo.body[0].body[index:index+2]
    if ast.dump(undo,include_attributes=False)!=ast.dump(source,include_attributes=False):
        raise ValueError('Primary binding must preserve every physical host statement')
    changed[0].comparators[0].value=True
    wrapper=ast.ClassDef(name='_PrimaryInitializer',bases=[],keywords=[],body=tree.body,decorator_list=[])
    namespace=dict(original.__globals__,_controlled_contract=runtime.controlled_contract,
                   _controlled_approval=copy.deepcopy(runtime.APPROVAL))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
                 '<primary_role_and_authority_only>','exec'),namespace)
    candidate=namespace['_PrimaryInitializer'].__dict__['__init__']
    if candidate.__code__.co_freevars!=original.__code__.co_freevars:
        raise ValueError('Exact original super-class closure required')
    result=FunctionType(candidate.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=original.__kwdefaults__
    return result


def primary_observer():
    original=runtime.lc.run_lifecycle_once
    source=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=copy.deepcopy(source)
    changed=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.keyword) and node.arg in ('use_for_final_statistics','actual_motion_authorized')
                and isinstance(node.value,ast.Constant) and node.value.value is False):
            changed.append(node)
            node.value.value=True
    if sorted(n.arg for n in changed)!=['actual_motion_authorized','use_for_final_statistics']:
        raise ValueError('Two original observation provenance fields required')
    for n in changed:n.value.value=False
    if ast.dump(tree,include_attributes=False)!=ast.dump(source,include_attributes=False):
        raise ValueError('No controller/guard/decision/cleanup edits in primary observer')
    for n in changed:n.value.value=True
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<primary_observation_labels_only>','exec'),namespace)
    return namespace[original.__name__]


def primary_saver():
    original=runtime.lc.RetainedMissionSDKHost.save
    source=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=copy.deepcopy(source)
    changed=[n for n in ast.walk(tree) if isinstance(n,ast.keyword)
             and n.arg=='ground_truth_development_NOT_sensor_Nav2_or_final_repetition']
    if len(changed)!=1:
        raise ValueError('One engineering-only provenance label required')
    old=changed[0].arg
    changed[0].arg='ground_truth_primary_predeclared_NOT_sensor_Nav2'
    undo=copy.deepcopy(tree)
    for n in ast.walk(undo):
        if isinstance(n,ast.keyword) and n.arg=='ground_truth_primary_predeclared_NOT_sensor_Nav2':n.arg=old
    if ast.dump(undo,include_attributes=False)!=ast.dump(source,include_attributes=False):
        raise ValueError('No result/trace/verdict change permitted')
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<primary_result_role_label_only>','exec'),namespace)
    return namespace[original.__name__]


class PrimaryRouteHost(runtime.lc.RetainedMissionSDKHost):
    __init__=primary_initializer()
    _save_with_primary_label=primary_saver()

    def save(self,report,rows):
        report=dict(report,predeclared_campaign=dict(
            protocol=self.protocol['primary_protocol'],protocol_sha256=self.protocol['primary_protocol_sha256'],
            cell=self.protocol['primary_cell'],
            fresh_predeclared_trial_NOT_relabelled_engineering_or_deterministic_replay=True))
        return self._save_with_primary_label(report,rows)


def task_route_packet(context, original_plan, geometry_path, grid_path, *, simulation_budget_s, wall_budget_s):
    """Reuse the exact packet exporter with predeclared common task clocks."""
    original=runtime.route_packet
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    counts=dict(sim=0,wall=0,contract_wall=0)
    for node in ast.walk(tree):
        if isinstance(node,ast.keyword) and node.arg=='simulation_budget_s' and isinstance(node.value,ast.Constant) and node.value.value==180.:
            node.value.value=float(simulation_budget_s)
            counts['sim']+=1
        if isinstance(node,ast.keyword) and node.arg=='maximum_wall_time_s' and isinstance(node.value,ast.Constant) and node.value.value==900.:
            node.value.value=float(wall_budget_s)
            counts['wall']+=1
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and node.func.attr=='unified_geometry_lifecycle_contract'):
            node.keywords.append(ast.keyword(arg='wall_budget_s',value=ast.Constant(float(wall_budget_s))))
            counts['contract_wall']+=1
    if counts!=dict(sim=2,wall=1,contract_wall=1):
        raise ValueError('Original packet-clock anchors changed')
    def contract(packet):
        if packet['scenario']=='H6_integrated_transfer_mission':
            packet['common_wall_budget_review']=copy.deepcopy(runtime.sequence.H6_COMMON_WALL_REVIEW)
        return runtime.controlled_contract(packet)
    namespace=dict(original.__globals__,controlled_contract=contract)
    exec(compile(ast.fix_missing_locations(tree),'<predeclared_common_task_clock_packet>','exec'),namespace)
    return namespace[original.__name__](context,original_plan,geometry_path,grid_path)


def checked_model_policy(path):
    """Explicit prospective admission policy, never a fabricated model PASS."""
    value=runtime.read(path)
    required=dict(model_is_prediction_not_actual_failure=True,
        independent_full_envelope_audit_required=True,actual_static_PASS_required=True,
        unchanged_actual_controller_limits_and_success_criteria=True,
        applies_to_all_methods_and_history_ablation=True,no_physical_retry=True)
    if any(value.get(k) is not v for k,v in required.items()):
        raise ValueError('Complete common diagnostic-model policy required')
    if value.get('user_authority')!='뭐가 가장 베스트의 방향성인지 판단하고 진행해':
        raise ValueError('Current delegated admission decision required')
    if value.get('protocol_sha256')!=runtime.sha(campaign.PROTOCOL):
        raise ValueError('Same frozen primary design required')
    return value


async def run_cell(cell_id, *, prior_result, authorized=False,
                   model_policy_path=None, no_entry_history=False):
    if authorized is not True:
        raise ValueError('Current explicit additional-experiment authority required')
    protocol=runtime.read(campaign.PROTOCOL)
    matches=[r for r in protocol['main_cells'] if r['cell_id']==cell_id]
    if len(matches)!=1:
        raise ValueError('One unconsumed predeclared main cell required')
    cell=copy.deepcopy(matches[0])
    if no_entry_history and (cell['method']!=campaign.planning.P or cell['task'] not in ('H2','H3')):
        raise ValueError('Only predeclared H2/H3 P history ablations are admitted')
    label=cell_id+('_no_entry_history' if no_entry_history else '')
    if no_entry_history:
        cell.update(ablation_variant='P_no_entry_state_history',parent_main_cell_id=cell_id,
                    cell_id=label)
    policy=checked_model_policy(model_policy_path) if model_policy_path is not None else None
    output=campaign.OUTPUT/'live'/label
    if output.exists():
        raise FileExistsError('Consumed primary cell; no retry or replacement')
    planned=planning.OUTPUT/label
    summary=runtime.read(planned/'summary.json')
    if not summary['plan_success']:
        return dict(cell_id=cell_id,status='PLANNING_FAILURE_NO_PHYSICAL_RUN',planning_summary=str(planned/'summary.json'))
    model_path=planned/'executor_model_screen/model_screen.json'
    model=runtime.read(model_path)
    if not model['both_nominal_resolutions_pass'] and policy is None:
        return dict(cell_id=cell_id,status='MODEL_ADMISSION_REJECTED_NO_PHYSICAL_RUN',model_screen=str(model_path))
    runtime.retained.runtime.task.verify_pins(model['input_sha256'])
    runtime.retained.runtime.task.verify_pins(model['output_sha256'])
    original_plan=Path(summary['plan'])
    independent=runtime.read(planned/'independent_plan_audit.json')
    if independent['verdict']!='PASS' or independent['plan_sha256']!=runtime.sha(original_plan):
        raise ValueError('Own independent plan PASS required')
    geometry_path=planned/'task_geometry.json'
    grid_path=planned/'planning_occupancy.npz'
    geometry=runtime.read(geometry_path)
    if geometry['start_xyyaw']!=cell['start_xyyaw'] or geometry['goal_xyyaw']!=cell['goal_xyyaw']:
        raise ValueError('Exact prospectively drawn start and common goal required')
    output.mkdir(parents=True,exist_ok=False)
    runtime.evidence.write_exclusive(output/'primary_trial_definition.json',dict(
        protocol=str(campaign.PROTOCOL),protocol_sha256=runtime.sha(campaign.PROTOCOL),cell=cell,
        role=('PREDECLARED_HISTORY_ABLATION' if no_entry_history else 'PREDECLARED_MAIN_NEW_PHYSICAL_TRIAL_NOT_ENGINEERING'),
        model_admission_policy=policy,model_screen=str(model_path),
        model_prediction_both_resolutions_pass=model['both_nominal_resolutions_pass'],
        authority=runtime.APPROVAL,old_engineering_trials_excluded=True,
        source_sha256={str(Path(__file__).resolve()):runtime.sha(Path(__file__))},
        maximum_static_trials=1,maximum_route_trials=1,no_automatic_retry=True))
    actor='/World/patient_transport_revision_'+label.replace('_dynamics_full_envelope','').lower()
    context=runtime.context_for(output,actor,geometry)
    sdk=runtime.retained.runtime.live_sdk()
    try:
        await runtime.prepare_source(context,sdk,geometry,prior_result=prior_result)
        unused,packet=runtime.static_packet(context,geometry_path,grid_path)
        packet=copy.deepcopy(packet)
        packet['run_id']='hospital_revision_'+cell_id+'_static'
        packet['input_sha256'][str(Path(__file__).resolve())]=runtime.sha(Path(__file__))
        if model_policy_path is not None:
            packet['input_sha256'][str(Path(model_policy_path).resolve())]=runtime.sha(model_policy_path)
        static_path=output/'static/current_primary_support_protocol.json'
        runtime.evidence.write_exclusive(static_path,packet)
        with np.load(grid_path) as grid:
            args=dict(occupied=grid['occupied'].copy(),origin=grid['origin_xy_m'].copy(),
                      resolution=float(grid['resolution_m']))
        host=runtime.ControlledStaticHost(sdk,packet,static_path,context=context,geometry=geometry,
                                          **args,authorized=True)
        static=await runtime.lc.run_held_static_once(host,
            runtime.lc.HeldStaticController(geometry['start_xyyaw']),authorized=True)
        if static['verdict']!='PASS':
            outcome=dict(cell_id=label,status='STATIC_NOT_PASS_NO_ROUTE',actual_route_started=False,
                         actual_verdict=static['verdict'],report=str(context.stationary.RESULT),attempt_replaced=False)
            runtime.evidence.write_exclusive(output/'trial_outcome.json',outcome)
            return outcome
        task=protocol['tasks'][cell['task']]
        sim,wall=task['simulation_budget_s'],task['wall_budget_s']
        unused,packet,plan=task_route_packet(context,original_plan,geometry_path,grid_path,
                                            simulation_budget_s=sim,wall_budget_s=wall)
        packet=copy.deepcopy(packet)
        packet.update(use_for_final_statistics=True,primary_protocol=str(campaign.PROTOCOL),
            primary_protocol_sha256=runtime.sha(campaign.PROTOCOL),primary_cell=cell,
            simulation_budget_s=sim,maximum_wall_time_s=wall,
            common_lifecycle_contract=runtime.sequence.unified_geometry_lifecycle_contract(
                plan['geometry_profile'],simulation_budget_s=sim,wall_budget_s=wall))
        if cell['task']=='H6':packet['common_wall_budget_review']=copy.deepcopy(runtime.sequence.H6_COMMON_WALL_REVIEW)
        packet['input_sha256'][str(Path(__file__).resolve())]=runtime.sha(Path(__file__))
        packet['input_sha256'][str(campaign.PROTOCOL)]=runtime.sha(campaign.PROTOCOL)
        if model_policy_path is not None:
            packet['input_sha256'][str(Path(model_policy_path).resolve())]=runtime.sha(model_policy_path)
        path=output/'route/current_primary_sdk_binding.json'
        runtime.evidence.write_exclusive(path,packet)
        runtime.lc.freeze_workflow_sources(output/'route/primary_sources.zip',packet['input_sha256'])
        port=PrimaryRouteHost(sdk,packet,path,context=context,plan=plan,geometry=geometry,
                              **args,authorized=True)
        controller=runtime.sequence.unified_geometry_lifecycle_controller(plan,geometry,
            profile=plan['geometry_profile'],simulation_budget_s=sim,wall_budget_s=wall)
        report=await primary_observer()(port,controller,authorized=True,
            sdk_bridge_class=runtime.sequence.RetainFinalTargetsBridge)
        observed_motion=report.get('new_route_trials')==1
        state='COMPLETED' if report['verdict']=='PASS' else (
            'EXECUTION_INTERRUPTED' if observed_motion else 'INFRASTRUCTURE_ABORT_BEFORE_ROUTE')
        outcome=dict(cell_id=label,status=state,
            actual_verdict=report['verdict'],report=str(port.result),
            actual_route_started=observed_motion,
            paired_condition_id=cell['paired_condition_id'],attempt_replaced=False)
        runtime.evidence.write_exclusive(output/'trial_outcome.json',outcome)
        return outcome
    except BaseException as failure:
        runtime.evidence.write_exclusive(output/'infrastructure_error.json',dict(
            error=type(failure).__name__+': '+str(failure),no_automatic_retry=True,cell_id=cell_id))
        raise
