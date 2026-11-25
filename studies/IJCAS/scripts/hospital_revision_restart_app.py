"""User-requested fresh Isaac app launch only; never Play or a trial."""
from pathlib import Path
import os
import socket
import subprocess

import hospital_revision_final_seed30 as experiment

runtime=experiment.runtime
OUTPUT=experiment.OUTPUT/'process_restart_r01'
ISAAC=Path('/home/jk/miniconda3/envs/isaac_env/bin/isaacsim')


def launch():
    experiment.campaign().verify_sources()
    try:
        with socket.create_connection(('127.0.0.1',8226),timeout=1.):
            raise ValueError('Existing Isaac execution port; do not launch a duplicate app')
    except (ConnectionRefusedError,TimeoutError,OSError) as error:
        if isinstance(error,ValueError):raise
    receipt=OUTPUT/'app_launch.json'
    if receipt.exists():raise FileExistsError('Consumed app launch; inspect instead of repeating')
    if not ISAAC.is_file():raise FileNotFoundError(ISAAC)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    command=[str(ISAAC),'isaacsim.exp.full','--enable','isaacsim.code_editor.vscode']
    with (OUTPUT/'isaac_stdout.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=experiment.original.design.original.ROOT,
            env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    value=dict(pid=process.pid,command=command,DISPLAY=os.environ.get('DISPLAY'),
        user_authority='2026-11-22 아이작 꺼짐 다시 켜서 이어서 진행해',
        original_source_and_results_not_modified=True,
        no_scene_or_timeline_or_actuation_command=True,
        launcher_sha256=runtime.sha(__file__))
    runtime.evidence.write_exclusive(receipt,value)
    return value


if __name__=='__main__':print(launch(),flush=True)
