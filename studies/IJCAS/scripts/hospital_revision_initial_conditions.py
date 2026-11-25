"""Prospective steering-initial-condition supplement to the frozen pose block.

Five independent paired joint-state draws in every H1--H6 task, B2 and P.
The original n=5 pose repetitions and all negatives remain separate. Frozen
paths are reused (no new planning-time repetition is claimed). Only a NEW
source's initial JointStateAPI values differ; there is never a paused/moving
articulation reset. Original geometry, material, mass and motor settings stay.
"""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import hospital_revision_campaign as campaign
import hospital_revision_live as live

runtime=campaign.runtime
OUTPUT=campaign.OUTPUT/'steering_initial_supplement'
PROTOCOL=OUTPUT/'protocol.json'
CASTERS=tuple('base_link_to_'+end+'_'+side+'_caster'
              for end in ('front','rear') for side in ('left','right'))
ACTIVE=('base_link_to_front_active_steering','base_link_to_rear_active_steering')


def definition():
    original=runtime.read(campaign.PROTOCOL)
    rng=np.random.default_rng(2026112102)
    rows=[]
    for task in ('H2','H1','H3','H4','H5','H6'):
        for repeat in range(1,6):
            paired=task+f'_s{repeat:02d}'
            angles=dict(zip((*ACTIVE,*CASTERS),map(float,
                np.r_[rng.uniform(-.5,.5,2),rng.uniform(-20.,20.,4)])))
            selected=[c for c in original['main_cells'] if c['paired_condition_id']==paired
                      and c['method'] in (campaign.planning.B2,campaign.planning.P)]
            if len(selected)!=2:raise ValueError('Own paired B2/P main conditions required')
            for source in selected:
                cell=copy.deepcopy(source)
                cell.update(parent_main_cell_id=source['cell_id'],
                    cell_id=source['cell_id']+'_joint_initial',
                    paired_condition_id=paired+'_joint_initial',
                    initial_joint_angles_deg=angles,initial_joint_velocities_deg_s=0.,
                    initial_active_steering_rule='prespecified new-source JointStateAPI; measured before/after static',
                    source_start_pose_NOT_new_draw=True)
                rows.append(cell)
    return dict(version='PAIRED_NEW_SOURCE_STEERING_SENSITIVITY_N5_v1',
        status='PROSPECTIVE_NOT_COMPLETED_RESULTS',main_cells=rows,
        tasks=copy.deepcopy(original['tasks']),parent_protocol=str(campaign.PROTOCOL),
        parent_protocol_sha256=runtime.sha(campaign.PROTOCOL),
        authority='교수님 피드백 모든 부분에 기반하여 계속 진행하도록 한 사용자 요청',
        random_generator='numpy.default_rng(2026112102)',planned_cells=60,
        methods=['B2','P'],tasks_covered=['H1','H2','H3','H4','H5','H6'],
        active_steering_initial_range_deg=[-.5,.5],passive_swivel_initial_range_deg=[-20.,20.],
        active_steering_within_original_nominal1deg_release_tolerance=True,
        existing_actual_static_and_pre_hold_may_change_angles_and_are_NOT_bypassed=True,
        record_actual_first_physics_and_route_entry_angles=True,
        steering_sensitivity_separate_from_five_pose_only_repetitions=True,
        old_negative_trials_NOT_removed_replaced_or_relabelled=True,
        saved_main_paths_reused=True,no_new_independent_planning_time_measurement_claimed=True,
        shared_controller_speed_physics_success_limits_unchanged=True,
        force_or_safety_constraints_NOT_relaxed=True,
        no_actual_robot_or_sensor_Nav2_trials=True,
        maximum_static_trials_per_cell=1,maximum_route_trials_per_cell=1,
        no_automatic_retry=True)


def export_protocol():
    value=definition()
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL)!=value:raise ValueError('Joint-condition supplement already frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL,value)
        runtime.lc.freeze_workflow_sources(OUTPUT/'design_sources.zip',
            {str(Path(__file__).resolve()):runtime.sha(Path(__file__))})
    return PROTOCOL


def seed_asset(path,angles_deg):
    """New reference overlay only; original source bytes are never modified."""
    from pxr import Usd, PhysxSchema
    constructor=runtime.shared.preparation.old.constructor
    original=Path(constructor.ASSET).resolve()
    before=runtime.sha(original)
    if set(angles_deg)!=set((*ACTIVE,*CASTERS)) or any(not math.isfinite(v) for v in angles_deg.values()):
        raise ValueError('Exactly six finite predeclared steering angles required')
    if any(abs(angles_deg[n])>.5 for n in ACTIVE) or any(abs(angles_deg[n])>20. for n in CASTERS):
        raise ValueError('Prespecified steering support exceeded')
    path=Path(path).resolve()
    if path.exists():raise FileExistsError('Do not overwrite a source condition')
    path.parent.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.CreateNew(str(path))
    root=stage.DefinePrim(constructor.OLD_ROBOT,'Xform')
    if not root.GetReferences().AddReference(str(original),constructor.OLD_ROBOT):
        raise ValueError('Unchanged robot source reference required')
    for name,angle in angles_deg.items():
        prim=stage.GetPrimAtPath(constructor.OLD_ROBOT+'/joints/'+name)
        if not prim:raise ValueError('Missing original revolute joint '+name)
        state=PhysxSchema.JointStateAPI.Apply(prim,'angular')
        state.CreatePositionAttr().Set(float(angle))
        state.CreateVelocityAttr().Set(0.)
    stage.GetRootLayer().Save()
    if runtime.sha(original)!=before:raise ValueError('Original robot asset changed')
    return dict(asset=str(path),sha256=runtime.sha(path),original=str(original),original_sha256=before,
        initial_angles_deg=dict(angles_deg),changes='six JointStateAPI initial positions and zero velocities only',
        new_source_not_paused_actor_reset=True,solver_first_step_initialization_NOT_an_executed_motion_trial=True)


def private_source_context(context,source_asset):
    """Context-local constructor/reference; no shared module monkeypatches."""
    p=context.preparation
    old=SimpleNamespace(**vars(p.old))
    original=old.constructor
    source=SimpleNamespace(**dict(vars(original),ASSET=Path(source_asset).resolve()))
    namespace=dict(original.author_new_robot.__globals__,ASSET=source.ASSET)
    # The complete original validator/constructor is retained; only the
    # reference asset now carries the declared initial-condition overlay.
    for name in ('validate_new_robot','author_new_robot'):
        function=getattr(original,name)
        from types import FunctionType
        namespace[name]=FunctionType(function.__code__,namespace,name,function.__defaults__,function.__closure__)
        setattr(source,name,namespace[name])
    old.constructor=source
    old.constructor_for=runtime.bind(p.old.constructor_for,constructor=source)
    p.old=old
    p.ASSET=source.ASSET
    shared=SimpleNamespace(**dict(vars(runtime.shared),preparation=p))
    shared.fresh_source_reference=runtime.bind(runtime.shared.fresh_source_reference,preparation=p)
    shared.fresh_source_for_live_actor=runtime.bind(runtime.shared.fresh_source_for_live_actor,
        fresh_source_reference=shared.fresh_source_reference)
    context.initial_condition_source_helpers=shared
    return context


async def prepare_source(context,sdk,geometry,*,prior_result=None):
    return await runtime.bind(runtime.prepare_source,shared=context.initial_condition_source_helpers)(
        context,sdk,geometry,prior_result=prior_result)


class InitialConditionStaticHost(runtime.ControlledStaticHost):
    async def preflight(self):
        proxy=SimpleNamespace(**dict(vars(runtime.retained),shared=self.context.initial_condition_source_helpers))
        return await runtime.bind(runtime.lc.PreparedHeldStaticSDKHost.preflight,retained=proxy)(self)

    async def bind_after_play(self):
        result=await super().bind_after_play()
        snapshot=runtime.shared.preparation.old.tensor_snapshot(self.tensor)
        prescribed=self.context.initial_joint_condition
        observed={name:math.degrees(snapshot['positions'][snapshot['names'].index(name)])
                  for name in (*ACTIVE,*CASTERS)}
        errors={name:math.degrees(math.atan2(math.sin(math.radians(observed[name]-value)),
                     math.cos(math.radians(observed[name]-value)))) for name,value in prescribed.items()}
        receipt=dict(time_s=self.timeline.get_current_time_s() if hasattr(self.timeline,'get_current_time_s')
                     else self.timeline.get_current_time(),
            prescribed_new_source_angles_deg=prescribed,first_bound_physics_angles_deg=observed,
            wrapped_difference_deg=errors,complete_tensor_snapshot=snapshot,
            measurement_is_first_BOUND_sample_NOT_guaranteed_first_solver_step=True,
            active_servo_target_remains_neutral_and_is_NOT_disabled=True,
            original_static_and_route_entry_release_limits_unchanged=True,
            passive_initialization_verified=all(abs(errors[n])<=2.5 for n in CASTERS))
        runtime.evidence.write_exclusive(self.output/'initial_joint_initialization_readback.json',receipt)
        if not receipt['passive_initialization_verified']:
            raise ValueError('Declared new-source caster initialization not verified; no claimed steering sensitivity result')
        return result


def cell_bindings(cell):
    """Use one source overlay for the pair, with the original plan/audit files."""
    source_path=OUTPUT/'source_conditions'/(cell['paired_condition_id']+'.usda')
    receipt=source_path.with_suffix('.json')
    if not receipt.exists():
        runtime.evidence.write_exclusive(receipt,seed_asset(source_path,cell['initial_joint_angles_deg']))
    else:
        value=runtime.read(receipt)
        if value['initial_angles_deg']!=cell['initial_joint_angles_deg'] or value['sha256']!=runtime.sha(source_path):
            raise ValueError('Same paired source seed and immutable asset required')
    def context_for(folder,actor,geometry):
        context=private_source_context(runtime.context_for(folder,actor,geometry),source_path)
        context.initial_joint_condition=copy.deepcopy(cell['initial_joint_angles_deg'])
        return context
    private_runtime=SimpleNamespace(**dict(vars(runtime),context_for=context_for,
        prepare_source=prepare_source,ControlledStaticHost=InitialConditionStaticHost))
    private_campaign=SimpleNamespace(**dict(vars(campaign),OUTPUT=OUTPUT,PROTOCOL=PROTOCOL))
    class OriginalPlanFolder:
        def __truediv__(self,label):
            if label!=cell['cell_id']:raise ValueError('One declared initial-condition cell required')
            return live.planning.OUTPUT/cell['parent_main_cell_id']
    private_planning=SimpleNamespace(**dict(vars(live.planning),OUTPUT=OriginalPlanFolder()))
    policy=runtime.bind(live.checked_model_policy,campaign=private_campaign)
    run=runtime.bind(live.run_cell,runtime=private_runtime,campaign=private_campaign,
        planning=private_planning,checked_model_policy=policy,__file__=__file__)
    return run


def build_source_assets():
    export_protocol()
    for cell in runtime.read(PROTOCOL)['main_cells']:
        cell_bindings(cell)
    return dict(protocol=str(PROTOCOL),planned_cells=60,
        source_conditions=len(list((OUTPUT/'source_conditions').glob('*.json'))),actual_PhysX_trials=0)


async def run(*,prior_result,authorized=False):
    import hospital_revision_execution as execution
    import hospital_revision_metrics as metrics
    if authorized is not True:raise ValueError('Current reviewer supplemental-experiment authority required')
    protocol=runtime.read(PROTOCOL)
    output=OUTPUT/'execution'
    output.mkdir(parents=True,exist_ok=False)
    policy_path=OUTPUT/'diagnostic_model_admission.json'
    policy=copy.deepcopy(runtime.read(execution.POLICY))
    policy.update(protocol=str(PROTOCOL),protocol_sha256=runtime.sha(PROTOCOL))
    runtime.evidence.write_exclusive(policy_path,policy)
    runtime.evidence.write_exclusive(output/'reservation.json',dict(protocol=str(PROTOCOL),
        protocol_sha256=runtime.sha(PROTOCOL),actual_maximum_cells=60,prior_result=str(prior_result),
        source_sha256={str(Path(__file__).resolve()):runtime.sha(Path(__file__))},
        one_static_and_conditional_one_route_per_cell=True,no_automatic_retry=True))
    runtime.lc.freeze_workflow_sources(output/'source.zip',
        {str(Path(__file__).resolve()):runtime.sha(Path(__file__))})
    last=Path(prior_result)
    if not execution.restored_boundary(runtime.read(last)):
        raise ValueError('Actual preserved/restored prior result required')
    outcomes=[]
    try:
        for cell in protocol['main_cells']:
            original=cell['parent_main_cell_id']
            reason,unused=execution.eligibility(dict(cell,cell_id=original),False)
            if reason!='ELIGIBLE_FOR_OWN_FRESH_STATIC':
                outcome=dict(cell_id=cell['cell_id'],status=reason,actual_route_started=False,
                    reused_parent_planning_result=original,no_new_planning_time_measurement=True)
            else:
                print('[REVIEW JOINT INITIAL START]',cell['cell_id'],flush=True)
                bound=cell_bindings(cell)
                outcome=await bound(cell['cell_id'],prior_result=last,authorized=True,model_policy_path=policy_path)
                report_path=Path(outcome['report']);report=runtime.read(report_path)
                if not execution.restored_boundary(report):
                    raise ValueError('Supplement actual Pause/restoration missing')
                last=report_path
                if outcome.get('actual_route_started'):
                    try:
                        runtime.evidence.write_exclusive(report_path.parent/'controlled_actual_metrics.json',
                            metrics.metrics(report_path))
                    except Exception as error:
                        runtime.evidence.write_exclusive(report_path.parent/'metrics_unavailable.json',dict(error=str(error)))
                initial=report_path.parent.parent/'static/initial_joint_initialization_readback.json'
                if not initial.exists() or not runtime.read(initial)['passive_initialization_verified']:
                    raise ValueError('Declared initial condition not verified; supplement stops without retry')
            outcomes.append(outcome)
            runtime.evidence.write_exclusive(output/'outcomes'/(cell['cell_id']+'.json'),outcome)
            print('[REVIEW JOINT INITIAL RESULT]',outcome,flush=True)
        result=dict(status='FINITE_JOINT_INITIAL_SUPPLEMENT_FINISHED',outcomes=outcomes,last_actual_report=str(last))
        runtime.evidence.write_exclusive(output/'completed.json',result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(output/'stopped.json',dict(
            error=type(error).__name__+': '+str(error),last_actual_report=str(last),
            completed_cells=len(outcomes),no_automatic_recovery_or_retry=True))
        raise


async def after_primary(parent):
    """One declared next block, only after the existing queue safely finishes."""
    import asyncio
    import builtins
    import hospital_revision_execution as execution
    receipt=OUTPUT/'after_primary_result.json'
    try:
        # Shield protects the already-running main campaign from accidental
        # cancellation of this waiting companion; no second task may actuate.
        result=await asyncio.shield(parent)
        if result.get('status')!='FINITE_QUEUE_FINISHED':
            raise ValueError('Primary queue did not finish normally; no supplementary actuation')
        if getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is not parent:
            raise ValueError('Another operation replaced the admitted primary task')
        last=Path(result['last_actual_report'])
        if not execution.restored_boundary(runtime.read(last)):
            raise ValueError('Primary final actual Pause/restoration incomplete')
        current=asyncio.current_task()
        builtins.ijcas_hospital_latest_task_experiment=current
        builtins.ijcas_revision_primary_task=current
        result=await run(prior_result=last,authorized=True)
        runtime.evidence.write_exclusive(receipt,result)
        return result
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT/'after_primary_blocked.json',dict(
            error=type(error).__name__+': '+str(error),no_new_automatic_recovery_or_retry=True))
        raise


def dispatch_after_primary():
    from hospital_revision_runner import remote
    if not PROTOCOL.exists() or len(list((OUTPUT/'source_conditions').glob('*.json')))!=30:
        raise ValueError('Frozen definition and all thirty validated paired source overlays required')
    path=Path(__file__).resolve()
    source=("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "parent=getattr(builtins,'ijcas_revision_primary_task',None)\n"
        "assert parent is not None, 'Own primary task required'\n"
        "assert not hasattr(builtins,'ijcas_revision_joint_supplement_waiter'), 'Supplement already reserved'\n"
        f"p=Path({str(path)!r})\n"
        "m=types.ModuleType('hospital_revision_initial_conditions');m.__file__=str(p);sys.modules[m.__name__]=m\n"
        "exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "builtins.ijcas_revision_joint_supplement_waiter=asyncio.ensure_future(m.after_primary(parent))\n"
        "print({'supplement_waiting_for_primary':True,'maximum_cells':60,'concurrent_motion':False,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch_after_primary.json',dict(
        exact_command=source,protocol_sha256=runtime.sha(PROTOCOL),entry_source_sha256=runtime.sha(path)))
    return remote(source)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',action='store_true')
    parser.add_argument('--after-primary',action='store_true')
    args=parser.parse_args()
    print(build_source_assets() if args.assets else dispatch_after_primary() if args.after_primary else export_protocol())
