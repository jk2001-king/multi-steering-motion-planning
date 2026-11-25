"""Stable configured hospital workflow entry; no default motion or retry."""

if globals().get('ijcas_hospital_task_workflow_authorized') is not True:
    raise RuntimeError('Separate explicit consent for ONE complete H5 workflow required; no SDK/action imported')
protocol=globals().get('ijcas_hospital_task_protocol')
if not isinstance(protocol,str):raise RuntimeError('Exact frozen reviewed workflow protocol path required')

import asyncio
import builtins
from pathlib import Path
import runpy
import traceback

for name in ('ijcas_h4_latest_experiment_task','ijcas_h4_latest_path_task','ijcas_h4_latest_visual_refresh_task',
        'ijcas_hospital_latest_task_experiment'):
    previous=getattr(builtins,name,None)
    if previous is not None and not previous.done():raise RuntimeError('A research task is still active; no competing workflow')

folder=Path('/home/jk/Research/IJCAS_ws/studies/IJCAS/scripts')
scope=runpy.run_path(str(folder/'hospital_private_loader.py'))['load'](folder/'hospital_task_runtime.py')

def _show_result(task):
    if task.cancelled():
        print('[H5 WORKFLOW] Cancelled; preserve current state, no automatic retry.',flush=True)
    elif task.exception() is not None:
        error=task.exception();traceback.print_exception(type(error),error,error.__traceback__)
        print('[H5 WORKFLOW] Aborted; existing reports preserved. Do not rerun or manipulate timeline.',flush=True)

ijcas_hospital_task_experiment=asyncio.ensure_future(scope['run_workflow_once'](protocol,authorized=True))
ijcas_hospital_task_experiment.add_done_callback(_show_result)
builtins.ijcas_hospital_latest_task_experiment=ijcas_hospital_task_experiment
print('[H5 WORKFLOW] ONE approved workflow scheduled: preflight/view -> archive/ONE Stop -> new source/display -> ONE static -> conditional ONE route. No retry.',flush=True)
