"""Retire a completed timeout without retrying or changing its verdict.

The original route has already zeroed, paused, and restored all motors. Its
postflight calls the expired mission guard before its environment GET, which
leaves the queue correctly blocked. A separate PAUSED, non-actuating proof
checks the missing boundary. Shared frozen per-cell execution is then reused
for only the unconsumed cells. No controller or experiment input is changed.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import hospital_revision_final_seed30 as experiment
import hospital_revision_display_handoff_native as previous

runtime = experiment.runtime
OUTPUT = experiment.OUTPUT / 'timeout_boundary_resume_r01'
KEY = 'ijcas_revision_n3_timeout_boundary_resume_r01'
AUTHORITY = '2026-11-22 사용자: 모든 조건 완료까지 계속 진행; 중간 구조 문제 직접 해결'
INITIAL = 'H3_P_dynamics_full_envelope_f03_steady_both'


def check_timeout_report(raw, packet):
    """Strictly diagnostic: never fabricate successful postflight/completion."""
    checks = raw.get('checks', {})
    if (raw.get('verdict') != 'HOLD' or raw.get('new_route_trials') != 1
            or raw.get('error') != 'TimeoutError: Declared whole-clock cap includes pre-hold/ACKs/terminal observation; no extension'
            or raw.get('postflight_error') != 'TimeoutError: Declared whole-mission physics budget'
            or any(checks.get(k) is not True for k in (
                'final_zero_target_sent', 'actual_Pause_confirmed',
                'original_gains_targets_restored', 'contacts_healthy_through_actual_Pause'))
            or checks.get('whole_route_and_fixed_hold_completed') is not False
            or checks.get('complete_postflight_verified') is not False
            or raw.get('callback_errors') or raw.get('notice_counts', {}).get('unreviewed_paths') != 0
            or not raw.get('actual_final_readback', {}).get('stable_PAUSED_tensor')):
        raise ValueError('Only the cleanly paused/restored exact deadline postflight case is eligible')
    final = raw['actual_final_readback']
    state = raw['actual_final_timeline']
    if state['playing'] or state['stopped'] or final['timeline'] != state:
        raise ValueError('Confirmed same PAUSED deadline boundary required')
    tensor = final['tensor']
    original = packet['required_tensor_snapshot']
    if (len(tensor['names']) != 12 or len(tensor['link_names']) != 13
            or any(tensor[k] != original[k] for k in (
                'names', 'link_names', 'gains', 'joint_position_targets_rad',
                'joint_velocity_targets_rad_s'))):
        raise ValueError('All original motors, limits, targets and source identities required')
    rejection = raw.get('runtime_guard', {}).get('rejected', {})
    if rejection.get('error') != 'TimeoutError: Declared whole-mission physics budget':
        raise ValueError('No physical/collision guard rejection may be waived')
    return True


async def prove_paused_boundary(report_path, cell):
    """GET-only: same timeline/layers/13-link/12-joint tensor plus 4997 rows."""
    from pxr import UsdPhysics
    report_path = Path(report_path).resolve()
    raw = runtime.read(report_path)
    packet = runtime.read(raw['protocol'])
    check_timeout_report(raw, packet)
    if packet['primary_cell']['cell_id'] != cell['cell_id']:
        raise ValueError('Exact declared cell, not an arbitrary old timeout')
    runtime.retained.runtime.task.verify_pins(packet['input_sha256'])
    if runtime.sha(raw['protocol']) != raw['protocol_sha256'] or runtime.sha(raw['trace']) != raw['trace_sha256']:
        raise ValueError('Immutable original protocol and trace required')
    sdk = runtime.retained.runtime.live_sdk()
    stage = sdk.omni.usd.get_context().get_stage()
    timeline = sdk.omni.timeline.get_timeline_interface()
    state = lambda: runtime.evidence.timeline_state(timeline)
    texts = lambda: dict(Root=stage.GetRootLayer().ExportToString(),
                         Session=stage.GetSessionLayer().ExportToString())
    saved = raw['actual_final_timeline']
    if state() != saved:
        raise ValueError('Actual retained Pause/clock changed; no restoration command')
    current = texts()
    display = runtime.retained.split_viewport_renderer_defaults(stage,
        raw['actual_final_readback']['layer_texts'], current)
    if display['normalized_after'] != raw['actual_final_readback']['layer_texts']:
        raise ValueError('Only the existing typed viewport-default proof is permitted')
    actor = packet['actor']
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 13 or any(not p.startswith(actor + '/') for p in bodies):
        raise ValueError('Exactly the retained source actor required')
    view = sdk.SimulationManager.get_physics_sim_view().create_articulation_view(actor + '/base_link')
    snapshot = runtime.shared.preparation.old.tensor_snapshot(view)
    if snapshot != raw['actual_final_readback']['tensor']:
        raise ValueError('Retained actual full tensor changed')
    prep = runtime.read(raw['preparation_record'])
    context = runtime.context_for(report_path.parent.parent, actor, runtime.read(packet['geometry_file']))
    static = runtime.read(packet['original_static_record'])

    def stable():
        if sdk.omni.usd.get_context().get_stage() != stage or state() != saved or texts() != current:
            raise ValueError('PAUSED GET stage/layers/timeline changed')

    expected = runtime.read(runtime.retained.H6 / 'actual_common_environment_rows.json')
    digest = await runtime.bootstrap.verify_rows(stage, expected['compact_rows'],
        next_frame=sdk.omni.kit.app.get_app().next_update_async, unchanged=stable)
    stable()
    if (digest != prep['protected_environment_sha256']
            or context.stationary.actor_configuration(stage) != prep['source_reference']['fixed_configuration_rows']
            or context.stationary.critical_environment(stage, prep) != static['environment_after_static']
            or runtime.shared.preparation.old.tensor_snapshot(view) != snapshot):
        raise ValueError('Complete environment/source/physics/tensor proof failed')
    receipt = dict(status='TIMEOUT_RETIRED_WITH_SEPARATE_PAUSED_GET_NOT_SUCCESS',
        original_report=str(report_path), original_report_sha256=runtime.sha(report_path),
        original_HOLD_and_timeout_and_failed_postflight_NOT_changed=True,
        original_trace_sha256=raw['trace_sha256'], cell_id=cell['cell_id'],
        actual_final_timeline=saved, actual_final_readback=dict(tensor=snapshot),
        all4997_rows_and145_source_settings_and_critical_physics_verified_NOW=True,
        protected_environment_sha256=digest, original_all12_motor_settings_targets_limits_verified=True,
        exact_saved_tensor_and_layers_retained=True, viewport_default_review=display['exact_renderer_default_differences'],
        no_Play_Stop_Pause_reset_camera_gain_target_or_scene_command=True,
        no_time_limit_extension_no_retry_no_extra_repetition=True,
        checks=dict(original_gains_targets_restored=True),
        input_sha256={str(report_path):runtime.sha(report_path),str(Path(raw['protocol'])):runtime.sha(raw['protocol'])})
    target = OUTPUT / 'paused_boundary_GET' / (cell['cell_id'] + '.json')
    runtime.evidence.write_exclusive(target, receipt)
    return target


async def run(parent, *, authorized=False):
    import builtins
    if authorized is not True or not parent.done() or parent.exception() is None:
        raise ValueError('Stopped exact predecessor and current continuation authority required')
    if getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not parent:
        raise ValueError('Another task owns the simulator')
    block = experiment.campaign()
    block.verify_sources()
    definition = runtime.read(OUTPUT / 'definition.json')
    runtime.retained.runtime.task.verify_pins(definition['input_sha256'])
    cells = runtime.read(block.EXECUTION_PROTOCOL)['main_cells']
    stopped = runtime.read(previous.OUTPUT / 'stopped.json')
    initial = next(c for c in cells if c['cell_id'] == INITIAL)
    last = Path(stopped['last_actual_report'])
    if last != block.OUTPUT / 'live' / INITIAL / 'route/route_readback.json':
        raise ValueError('Only the diagnosed consumed H3 ablation may initiate continuation')
    if len(list((block.OUTPUT / 'outcomes').glob('*.json'))) != 41:
        raise ValueError('Expected 41 completed outcomes plus exactly one consumed timeout')
    await prove_paused_boundary(last, initial)
    task = asyncio.current_task()
    builtins.ijcas_hospital_latest_task_experiment = task
    builtins.ijcas_revision_primary_task = task
    results = []
    runtime.evidence.write_exclusive(OUTPUT / 'execution_reservation.json', dict(
        authority=AUTHORITY, no_consumed_route_retried=True, fixed_protocol=str(block.PROTOCOL),
        original_timeout_retained=True, controller_search_speed_physics_and_acceptance_unchanged=True))
    try:
        for cell in cells:
            name = cell['cell_id']
            target = block.OUTPUT / 'outcomes' / (name + '.json')
            if target.exists():
                results.append(runtime.read(target))
                continue
            block.verify_sources()
            if name == INITIAL:
                value = runtime.read(last.parent.parent / 'trial_outcome.json')
                value = dict(value, metrics=block.metrics.metrics(last),
                    separate_timeout_boundary_GET=str(OUTPUT/'paused_boundary_GET'/(name+'.json')))
            else:
                folder = block.folder_for(name)
                planned = runtime.read(folder / 'final_planning_receipt.json')
                if not planned['independently_valid_reference']:
                    value = dict(cell_id=name,
                        status='PLANNING_FAILURE_NO_PHYSICAL_RUN' if not planned['plan_success'] else 'INDEPENDENT_REFERENCE_AUDIT_REJECTED',
                        actual_route_started=False, actual_verdict='PLAN_FAILURE' if not planned['plan_success'] else 'REFERENCE_REJECTED',
                        planning_summary=str(folder/'summary.json'), metrics=None)
                else:
                    print('[N3 REMAINING ACTUAL START]', name, flush=True)
                    value = await block.bound_cell(cell)(name, prior_result=last,
                        authorized=True, model_policy_path=block.policy_file())
                    last = Path(value['report'])
                    raw = runtime.read(last)
                    if not block.safe_restored_boundary(raw):
                        proof = await prove_paused_boundary(last, cell)
                        value = dict(value, separate_timeout_boundary_GET=str(proof))
                    value = dict(value, metrics=block.metrics.metrics(last) if value['actual_route_started'] else None)
            value.update(experiment_role=cell['experiment_role'],controller_version=block.control.VERSION,
                predictive_history_policy=cell['predictive_history_policy'],final_fixed_version_trial=True,attempt_replaced=False)
            runtime.evidence.write_exclusive(target, value)
            results.append(value)
            print('[N3 REMAINING RESULT]', name, value['actual_verdict'], flush=True)
            await asyncio.sleep(0)
        runtime.evidence.write_exclusive(block.OUTPUT/'completed.json', dict(
            status='FINAL_N3_ALL_DECLARED_CELLS_FINISHED', results=results,last_actual_report=str(last),
            all_negative_results_preserved=True,automatic_retry=False,
            separate_deadline_boundary_proofs=str(OUTPUT/'paused_boundary_GET')))
    except BaseException as error:
        runtime.evidence.write_exclusive(OUTPUT/'stopped.json',dict(error=type(error).__name__+': '+str(error),
            last_actual_report=str(last),accounted_cells=len(results),automatic_retry=False))
        raise


def dispatch():
    block=experiment.campaign();block.verify_sources()
    files=[Path(__file__).resolve(),previous.OUTPUT/'stopped.json',previous.OUTPUT/'definition.json']
    definition=dict(authority=AUTHORITY,original_timeout_retained=True,maximum_routes_per_unconsumed_cell=1,
        no_controller_model_physics_speed_or_acceptance_changes=True,
        input_sha256={str(p):runtime.sha(p) for p in files})
    runtime.evidence.write_exclusive(OUTPUT/'definition.json',definition)
    runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip',definition['input_sha256'])
    source=("import importlib,asyncio,builtins\nimportlib.invalidate_caches()\n"
        "import hospital_revision_timeout_resume as _remaining\n"
        f"assert not hasattr(builtins,{KEY!r}),'Already reserved'\n"
        f"builtins.{KEY}=asyncio.ensure_future(_remaining.run(builtins.{previous.KEY},authorized=True))\n"
        "print({'remaining_queue_scheduled':True,'consumed_timeout_NOT_retried':True})\n")
    runtime.evidence.write_exclusive(OUTPUT/'dispatch.json',dict(source=source,authority=AUTHORITY))
    return experiment.original.predecessor.control.base.shared.runner.remote(source)


def monitor():
    block=experiment.campaign();n=-1
    while not (block.OUTPUT/'completed.json').exists():
        if (OUTPUT/'stopped.json').exists():raise ValueError('Unresolved continuation boundary; inspect, never blind retry')
        snapshot=block.results_snapshot()
        if snapshot['accounted_cells']!=n:
            n=snapshot['accounted_cells'];print('[N3 REMAINING PROGRESS]',n,'/78',flush=True)
        time.sleep(30.)
    analysis=block.export_results()
    import hospital_revision_final_comparison_analysis as detailed
    detail=detailed.export()
    runtime.evidence.write_exclusive(block.OUTPUT/'all_steps_completed.json',dict(
        status='FINAL_N3_ALL78_ACCOUNTED_AND_READONLY_ANALYSIS_EXPORTED',analysis_folder=str(analysis),
        detailed_feedback_analysis=str(detail),current_native_owner=KEY,
        final_independent_live_GET_still_required=True))
    print('[N3 REMAINING COMPLETE]',analysis,detail,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['dispatch','monitor']);args=parser.parse_args()
    if args.action=='dispatch':print(dispatch(),flush=True)
    else:monitor()
