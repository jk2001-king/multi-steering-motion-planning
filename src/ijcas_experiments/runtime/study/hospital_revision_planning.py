"""Shared bounded planning for the predeclared reviewer campaign.

All methods use one30mm/s execution schedule and original50mm/s spatial
search normalization. P and its entry-history ablation share lookahead8.
No Kit or scene operation; planning failures remain failures of this attempt.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
import argparse
import ast
import copy
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import hospital_revision_campaign as campaign

runtime=campaign.runtime
shared=campaign.planning.shared
mission=campaign.planning.mission
OUTPUT=campaign.OUTPUT/'planning'


def propagation_binding(original, *, no_entry_history, activity):
    """Ablate only five incoming predictive states; never a physical state."""
    def propagate(state,*args,**kwargs):
        if no_entry_history and kwargs.get('rigid_body_model') is not None:
            state=replace(state,front=0.,rear=0.,longitudinal_velocity=0.,
                          lateral_velocity=0.,yaw_rate=0.,started=True)
        result=original(state,*args,**kwargs)
        if kwargs.get('rigid_body_model') is not None:
            last=result[0]
            activity['model_primitive_calls']+=1
            activity['infeasible_result_count']+=int(not last.rigid_body_feasible)
            activity['maximum_friction_utilization']=max(activity['maximum_friction_utilization'],last.friction_utilization)
            activity['maximum_abs_ltr']=max(activity['maximum_abs_ltr'],last.abs_ltr)
            activity['friction_limit_exceedance_count']+=int(last.friction_utilization>.8)
            activity['LTR_limit_exceedance_count']+=int(last.abs_ltr>.8)
        return result
    propagate.__dict__.update(original.__dict__)
    return propagate


def guided_binding(*, lookahead_edges=8):
    if type(lookahead_edges) is not int or lookahead_edges not in (4,8):
        raise ValueError('Explicit uniform lookahead4 or original8 required')
    original=shared.unified_h3_guided_candidate
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    found=[]
    class Resource(ast.NodeTransformer):
        def visit_Import(self,node):
            if len(node.names)==1 and node.names[0].name=='hospital_mission_planning' and node.names[0].asname=='resource':
                found.append(True)
                return ast.copy_location(ast.Assign([ast.Name('resource',ast.Store())],
                    ast.Name('_campaign_resource',ast.Load())),node)
            return node
    tree=Resource().visit(tree)
    if len(found)!=1:
        raise ValueError('One original resource import required')
    def refine(*args,**kwargs):
        if kwargs.get('lookahead_edges')!=12:
            raise ValueError('Original lookahead anchor changed')
        return mission.refine_with_NE_guided_beam(*args,**dict(kwargs,lookahead_edges=lookahead_edges))
    resource=SimpleNamespace(**dict(vars(mission),refine_with_NE_guided_beam=refine))
    namespace=dict(original.__globals__,_campaign_resource=resource,
                   common_cruise_profile_caps=campaign.planning.common_caps)
    exec(compile(ast.fix_missing_locations(tree),'<common_all_task_lookahead8>','exec'),namespace)
    return namespace[original.__name__]


def planner_bindings(profile,geometry,occupied,origin,resolution,*,no_entry_history=False):
    activity=dict(model_primitive_calls=0,infeasible_result_count=0,
                  maximum_friction_utilization=0.,maximum_abs_ltr=0.,
                  friction_limit_exceedance_count=0,LTR_limit_exceedance_count=0)
    def connections(*args,**kwargs):
        original=shared.unified_wheelbase_connections(*args,**kwargs)
        result=SimpleNamespace(**vars(original))
        result.python=SimpleNamespace(**vars(original.python))
        result.python.propagate=propagation_binding(original.python.propagate,
            no_entry_history=no_entry_history,activity=activity)
        if original.compiled is not None:
            result.compiled=propagation_binding(original.compiled,
                no_entry_history=no_entry_history,activity=activity)
        return result
    factory=runtime.bind(shared.unified_h3_planner_binding,unified_wheelbase_connections=connections)
    method,auditor,conn,parameters=factory(profile,geometry,occupied,origin,resolution,
        reserve_m=.050,terminal_direction_change_reserve=2)
    method=runtime.bind(method,SPEED=.050)
    return method,auditor.__globals__['audit_actions'],conn,parameters,activity


def perturbed_geometry(original,cell):
    value=copy.deepcopy(original)
    start=list(cell['start_xyyaw'])
    goal=list(cell['goal_xyyaw'])
    dx,dy=goal[0]-start[0],goal[1]-start[1]
    angle=math.atan2(math.sin(goal[2]-start[2]),math.cos(goal[2]-start[2]))
    value.update(start_xyyaw=start,goal_xyyaw=goal,
        local_goal_xyyaw=[math.cos(start[2])*dx+math.sin(start[2])*dy,
                        -math.sin(start[2])*dx+math.cos(start[2])*dy,angle],
        predeclared_condition_id=cell['paired_condition_id'],
        original_fixed_task_geometry=cell['original_geometry'],
        perturbation_is_NOT_a_measured_state_or_terminal_reset=True,
        actual_motion_authorized=False,use_for_final_statistics=False)
    return value


def run_cell(cell_id, *, no_entry_history=False, lookahead_edges=8):
    protocol=runtime.read(campaign.PROTOCOL)
    matches=[c for c in protocol['main_cells'] if c['cell_id']==cell_id]
    if len(matches)!=1:
        raise ValueError('One predeclared campaign cell required')
    cell=dict(matches[0])
    if no_entry_history and (cell['method']!=campaign.planning.P or cell['task'] not in ('H2','H3')):
        raise ValueError('The history ablation is predeclared only for H2/H3 P')
    label=cell_id+('_no_entry_history' if no_entry_history else '')
    output=OUTPUT/label
    if output.exists():
        raise FileExistsError('Consumed planning attempt; no retry: '+str(output))
    for p,digest in protocol['original_evidence_inputs_sha256'].items():
        if runtime.sha(p)!=digest:
            raise ValueError('Original task evidence changed: '+p)
    task=protocol['tasks'][cell['task']]
    ready_path=campaign.GATE/'paper_evidence_inventory/revised_P_uniform_evaluation_r02/ready'/(cell['task']+'.json')
    source_plan=Path(runtime.read(ready_path)['planning_sources'][campaign.planning.P])
    reference=runtime.read(source_plan)
    cell['original_geometry']=task['original_geometry']
    geometry=perturbed_geometry(runtime.read(task['original_geometry']),cell)
    with np.load(geometry['occupancy_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel,RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import State
    model=MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**reference['rigid_body_parameter_bounds']))
    begun=time.perf_counter()
    method,auditor,connections,parameters,activity=planner_bindings(reference['geometry_profile'],geometry,
        occupied,origin,resolution,no_entry_history=no_entry_history)
    connections.compiled(State(0.,0.,0.,'coordinated_counterphase',0.,0.,True),
        'coordinated_counterphase',0.,1,speed_m_s=.05,rigid_body_model=model,target_speed_m_s=.03)
    initialization=time.perf_counter()-begun
    for k in activity:
        activity[k]=0 if isinstance(activity[k],int) else 0.
    output.mkdir(parents=True,exist_ok=False)
    geometry_path=output/'task_geometry.json'
    runtime.evidence.write_exclusive(geometry_path,geometry)
    np.savez_compressed(output/'planning_occupancy.npz',occupied=occupied,origin_xy_m=origin,resolution_m=resolution)
    fields=('environment_condition','geometry_profile','rigid_body_parameter_bounds')
    metadata={k:reference[k] for k in fields}
    metadata.update(schema_version=1,scenario=geometry['scenario'],
        geometry_manifest=str(geometry_path),geometry_manifest_sha256=runtime.sha(geometry_path),
        start_world_xyyaw=geometry['start_xyyaw'],goal_world_xyyaw=geometry['goal_xyyaw'],
        goal_local_xyyaw=geometry['local_goal_xyyaw'],planning_occupancy_digest=geometry['planning_occupancy_digest'],
        common_max_command_speed_m_s=.050,common_max_acceleration_m_s2=.020,
        full_envelope_clearance_m=.010,planning_additional_execution_reserve_m=.050,
        common_speed_policy=protocol['common_speed_policy'],geometry_profile_source=str(source_plan),
        continuous_task_search_binding=parameters,common_initialization_time_s=initialization,
        actual_motion_authorized=False,use_for_final_statistics=False,Isaac_execution_verified=False,
        physical_execution_verified=False,Nav2_execution_verified=False,
        integrated_stage_portal_bypass_and_continuous_executor_validation_pending=True,
        fixed_P_lookahead_edges_all_tasks=lookahead_edges,entry_history_ablation=no_entry_history,
        main_cell=cell,protocol=str(campaign.PROTOCOL),protocol_sha256=runtime.sha(campaign.PROTOCOL))
    sources=list(dict.fromkeys([Path(__file__),Path(campaign.__file__),Path(campaign.planning.__file__),
        Path(shared.__file__),Path(mission.__file__),*runtime.lc.host_source_files(),
        _ijcas_path(ROOT, 'src/patient_initial_stability/patient_initial_stability/steering_lattice.py'),
        _ijcas_path(ROOT, 'src/patient_initial_stability/patient_initial_stability/hierarchical_dynamics_planner.py'),
        _ijcas_path(ROOT, 'src/patient_initial_stability/patient_initial_stability/rigid_body_edge.py')]))
    pins={str(p.resolve()):runtime.sha(p) for p in [campaign.PROTOCOL,source_plan,geometry_path,*sources]}
    metadata['input_sha256']=pins
    runtime.evidence.write_exclusive(output/'definition.json',metadata)
    runtime.lc.freeze_workflow_sources(output/'sources.zip',{str(p.resolve()):pins[str(p.resolve())] for p in sources})
    print('[REVIEW PLAN] START',label,flush=True)
    begin=time.perf_counter()
    name=cell['method']
    if name==campaign.planning.P:
        seed=shared.unified_P_owned_seed(method,geometry,occupied,origin,resolution,model,
            budget_s=180.,node_limit=60000,policy=shared.UNIFIED_P_SEED_POLICY)
        runtime.evidence.write_exclusive(output/'own_seed.json',seed)
        remaining=180.-(time.perf_counter()-begin)
        row=guided_binding(lookahead_edges=lookahead_edges)(method,seed,geometry,occupied,origin,resolution,model,
            budget_s=remaining,execution_budget_s=task['simulation_budget_s']) if remaining>0 else dict(
                method=name,plan_success=False,status='PLANNING_BUDGET_EXHAUSTED',selected_actions=[],selected_states=[])
        row.update(owned_seed_attempts=seed['owned_seed_attempts'],owned_seed_total_time_s=seed['owned_seed_total_time_s'])
    else:
        row=method(name,geometry,occupied,origin,resolution,model,180.,60000,{})
        if row['plan_success']:
            speeds=campaign.planning.CommonSpeedPolicy().profile(row['selected_actions'])
            row['selected_actions']=[dict(a,target_speed_m_s=s) for a,s in zip(row['selected_actions'],speeds)]
    row.update(planning_time_s=time.perf_counter()-begin,
        planning_candidate_model_activity=copy.deepcopy(activity))
    if row['plan_success']:
        row['common_speed_rule_check']=campaign.planning.validate_profile(row)
    plan_path=output/(name+'.json')
    runtime.evidence.write_exclusive(plan_path,dict(metadata,**row))
    audit=None
    if row['plan_success']:
        audit=auditor(plan_path)
        runtime.evidence.write_exclusive(output/'independent_plan_audit.json',audit)
    result=dict(cell_id=label,method=name,variant='P_no_entry_state_history' if no_entry_history else name,
        plan=str(plan_path),plan_success=row['plan_success'],status=row['status'],
        planning_time_s=row['planning_time_s'],
        independent_model_consistency=audit['verdict'] if audit else 'NOT_RUN_NO_PLAN',
        physical_trials=0,physical_execution_NOT_claimed=True,
        model_activity=row['planning_candidate_model_activity'])
    runtime.evidence.write_exclusive(output/'summary.json',result)
    print('[REVIEW PLAN] RESULT',json.dumps(result),flush=True)
    return result


ROOT=campaign.ROOT
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell_id')
    parser.add_argument('--no-entry-history',action='store_true')
    args=parser.parse_args()
    run_cell(args.cell_id,no_entry_history=args.no_entry_history)
