"""Standard ROS executables; motion is never implied by a readiness check."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess

from ijcas_paths import ROOT, load_archived_json


def check_main():
    parser = argparse.ArgumentParser(description='Offline final experiment readiness; no simulator or motion')
    parser.add_argument('--full', action='store_true', help='Audit all preserved original file hashes')
    args = parser.parse_args()
    from .reproduce import BLOCK, SELECTED, SENSOR_DATA
    paths = [BLOCK/'protocol.json',BLOCK/'execution_protocol.json',
        SELECTED/'FINAL_EXPERIMENT_CLOSEOUT.json',SELECTED/'selection_manifest.json',
        SENSOR_DATA/'fig06_metrics.json',_ijcas_path(ROOT, 'isaac/scenes/hospital_experiment.usda')]
    missing = [str(p) for p in paths if not p.is_file()]
    errors = []
    receipt = _ijcas_path(ROOT, 'migration/commissioning_latest.json')
    commissioning = load_archived_json(receipt) if receipt.is_file() else None
    for directory in (_ijcas_path(ROOT, 'studies/IJCAS/scripts'),_ijcas_path(ROOT, 'isaac/scripts')):
        for path in directory.glob('*.py'):
            try:ast.parse(path.read_text())
            except SyntaxError as error:errors.append(str(error))
    result = dict(workspace=str(ROOT),missing=missing,syntax_errors=errors,
        archived_result_reproduction_ready=not (missing or errors),
        selectable_experiment_entry='ros2 launch ijcas_experiments experiment.launch.py',
        selection_and_offline_planning_supported=True,
        fresh_sensor_execution_supported=True,
        local_commissioning_receipt=commissioning,
        live_reproduction_note='Only cases explicitly listed in a saved commissioning receipt are physically checked. This readiness command starts no simulator and grants no motion authority.',
        simulator_started=False,new_trials=0)
    if args.full:
        from verify_workspace import verify
        result['layout_verification'] = verify(full=True)
    print(json.dumps(result,indent=2))
    return int(bool(missing or errors or args.full and not result['layout_verification']['PASS']))


def results_main():
    parser = argparse.ArgumentParser(description='Recompute fixed results from archived logs; no new trials')
    parser.add_argument('--output',default=None)
    args = parser.parse_args()
    from .reproduce import export
    folder, counts = export(Path(args.output) if args.output else None)
    print(json.dumps(dict(output=str(folder),completion_counts=counts,new_trials=0),indent=2))


def sensor_main():
    parser = argparse.ArgumentParser(description='Original measured-feedback ROS/AMCL/Nav2 bridge')
    parser.add_argument('configuration',type=Path)
    parser.add_argument('--inspect-only',action='store_true',help='Validate syntax and show configuration; do not start ROS nodes')
    args = parser.parse_args()
    value = json.loads(args.configuration.read_text())
    if args.inspect_only:
        print(json.dumps(value,indent=2));return
    # No launch defaults or stored questionnaires are promoted to motion consent.
    from hospital_sensor_ros_bridge import run_bridge
    run_bridge(str(args.configuration.resolve()))


def isaac_main():
    parser = argparse.ArgumentParser(description='Start Isaac Sim without Script Editor commands or automatic motion')
    parser.add_argument('--executable',default=os.environ.get('ISAAC_SIM_EXECUTABLE','isaacsim'))
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    executable = shutil.which(args.executable)
    if executable is None and Path(args.executable).is_file():
        executable = str(Path(args.executable).resolve())
    if executable is None:
        parser.error('Isaac executable not found; pass --executable /path/to/isaacsim or ISAAC_SIM_EXECUTABLE')
    command = [executable,'isaacsim.exp.full']+(['--no-window'] if args.headless else [])
    if args.dry_run:
        print(json.dumps(dict(command=command,simulator_started=False,automatic_motion=False),indent=2));return
    return subprocess.call(command,cwd=ROOT)
