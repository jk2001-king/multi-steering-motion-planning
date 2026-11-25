"""Stable H4 experiment entry. Configuration selects an admitted action.

No old controller is selected implicitly; unsupported actions fail before
Kit imports or scheduling. Each trial needs explicit, action-specific consent.
"""

action=globals().get('ijcas_hospital_h4_action')
if action not in ('static_support','initial_motion','fresh_whole_path','first_route_after_static','review_prepath_hold','initialize_staged_display','review_staged_boundary','continue_staged_workflow'):
    raise RuntimeError('Only explicit admitted experiment actions or review_prepath_hold GETs are implemented')
if action=='static_support' and globals().get('ijcas_hospital_h4_single_static_authorized') is not True:
    raise RuntimeError('Explicit consent for ONE corrected retained-state static trial is required')
if action=='initial_motion' and globals().get('ijcas_hospital_h4_single_initial_motion_authorized') is not True:
    raise RuntimeError('Separate explicit consent for ONE initial-motion component is required; static consent is insufficient')
if action=='fresh_whole_path' and globals().get('ijcas_hospital_h4_single_fresh_whole_path_authorized') is not True:
    raise RuntimeError('Separate explicit consent for ONE fresh preparation/static/whole-path workflow is required')
if action=='first_route_after_static' and globals().get('ijcas_hospital_h4_first_route_continuation_authorized') is not True:
    raise RuntimeError('Explicit continuation of the already approved, not-yet-started first route required')
if action=='initialize_staged_display' and globals().get('ijcas_hospital_h4_visual_initialization_authorized') is not True:
    raise RuntimeError('Explicit stopped-actor DISPLAY ONLY authorization required; no action')
if action=='continue_staged_workflow' and globals().get('ijcas_hospital_h4_staged_remainder_authorized') is not True:
    raise RuntimeError('Explicit original unused static/conditional-route remainder consent required')

import asyncio
import builtins
from pathlib import Path
import runpy
import traceback

scripts=Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
scope=runpy.run_path(str(scripts/'hospital_private_loader.py'))['load'](scripts/'hospital_h4_subfloor_compact_path_launch.py')


def _show_h4_experiment_error(task):
    if task.cancelled():
        print('[H4 EXPERIMENT] Cancelled; no automatic retry.',flush=True)
        return
    error=task.exception()
    if error is not None:
        traceback.print_exception(type(error),error,error.__traceback__)
        print('[H4 EXPERIMENT] Aborted; do not rerun or manipulate timeline.',flush=True)


if action=='continue_staged_workflow':
    protocol=globals().get('ijcas_hospital_h4_trial_protocol')
    if not isinstance(protocol,str):raise RuntimeError('Exact reviewed staged continuation protocol required')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            raise RuntimeError('An experiment/route/display task is still active; no competing workflow')
    scope['load_staged_workflow_continuation'](protocol)
    ijcas_hospital_h4_experiment_task=asyncio.ensure_future(scope['run_staged_workflow_continuation'](protocol))
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    print('[H4 REMAINDER] Same stopped actor -> ONE original static -> conditional ONE B2 route. NO Stop/restage/refresh/retry.',flush=True)
elif action=='review_staged_boundary':
    protocol=globals().get('ijcas_hospital_h4_trial_protocol')
    if not isinstance(protocol,str):raise RuntimeError('Exact own archived stopped-actor protocol required')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            raise RuntimeError('An experiment/route/display task is still active; no concurrent boundary review')
    ijcas_hospital_h4_experiment_task=asyncio.ensure_future(scope['review_recorded_staged_boundary'](protocol))
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    print('[H4 STAGED REVIEW] GET-only exact original/current layer comparison. NO timeline or display command.',flush=True)
elif action=='initialize_staged_display':
    protocol=globals().get('ijcas_hospital_h4_trial_protocol')
    if not isinstance(protocol,str):raise RuntimeError('Exact archived staged-actor protocol required')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            raise RuntimeError('An experiment/route/display task is still active; no competing renderer operation')
    ijcas_hospital_h4_experiment_task=asyncio.ensure_future(scope['initialize_recorded_staged_display'](protocol))
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    print('[H4 DISPLAY INIT] Own stopped actor -> checked render registration. NO Play/Stop/static/route.',flush=True)
elif action=='review_prepath_hold':
    protocol=globals().get('ijcas_hospital_h4_trial_protocol')
    if not isinstance(protocol,str):raise RuntimeError('Explicit own archived protocol required for GET-only review')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            raise RuntimeError('Experiment/route/display still active; no concurrent review')
    ijcas_hospital_h4_experiment_task=asyncio.ensure_future(scope['review_fresh_prepath_hold'](protocol))
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    print('[H4 GET REVIEW] Scheduled read-only retained HOLD/field review. NO timeline, control or scene command.',flush=True)
elif action=='first_route_after_static':
    path=globals().get('ijcas_hospital_h4_route_continuation')
    if not isinstance(path,str):raise RuntimeError('Explicit reviewed continuation protocol required')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            raise RuntimeError('An experiment/route/display task is still active; no competing operation')
    reviewed=scope['load_first_route_continuation'](path)
    dispatched=runpy.run_path(str(scope['ENTRY']),init_globals={
        'ijcas_hospital_h4_subfloor_compact_path_authorized':True,
        'ijcas_hospital_h4_fresh_trial_protocol':str(reviewed.protocol_path),
        'ijcas_hospital_h4_route_continuation':path,
        'ijcas_hospital_h4_method':reviewed.protocol['method']},run_name='__main__')
    ijcas_hospital_h4_experiment_task=dispatched['ijcas_hospital_h4_path_task']
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    builtins.ijcas_h4_latest_path_task=ijcas_hospital_h4_experiment_task
    print('[H4 EXPERIMENT] Same fresh static -> FIRST whole route only. No Stop, repeated static or reset.',flush=True)
elif action=='fresh_whole_path':
    protocol=globals().get('ijcas_hospital_h4_trial_protocol')
    if not isinstance(protocol,str):raise RuntimeError('Explicit frozen fresh-trial protocol path required')
    for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task'):
        previous=getattr(builtins,name,None)
        if previous is not None and not previous.done():
            raise RuntimeError('A prior experiment/route/display task is still active; no competing workflow')
    # Validate the complete offline artifact before even scheduling live GETs.
    reviewed=scope['load_fresh_trial_protocol'](protocol)
    if (reviewed.protocol.get('terminal_connection_bundle') is not None
            and globals().get('ijcas_hospital_h4_terminal_connection_candidate_authorized') is not True):
        raise RuntimeError('Separate explicit consent for ONE new terminal-connection planner candidate is required; original-plan consent is insufficient')
    ijcas_hospital_h4_experiment_task=asyncio.ensure_future(scope['run_fresh_whole_trial'](protocol))
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    print('[H4 EXPERIMENT] ONE approved fresh workflow scheduled: archive/Stop -> source start -> static -> conditional whole route. No retry.',flush=True)
elif action=='static_support':
    ijcas_hospital_h4_experiment_task=asyncio.ensure_future(scope['run_retained_static']())
    ijcas_hospital_h4_experiment_task.add_done_callback(_show_h4_experiment_error)
    print('[H4 EXPERIMENT] ONE static preflight scheduled; NO Play before retained-state/environment checks.',flush=True)
else:
    # SAME control loop, guards, captured-wheel hold and evidence pipeline as
    # the whole-route adapter. Only its admitted input/role/output bindings
    # differ. No new copied controller or automatic subsequent action.
    dispatched=runpy.run_path(str(scope['ENTRY']),init_globals={
        'ijcas_hospital_h4_initial_motion_authorized':True},run_name='__main__')
    ijcas_hospital_h4_experiment_task=dispatched['ijcas_hospital_h4_path_task']

# Keep the latest task reachable if postprocessing fails. A manual GC sweep
# must never be needed to retain the already completed physical evidence.
builtins.ijcas_h4_latest_experiment_task=ijcas_hospital_h4_experiment_task
