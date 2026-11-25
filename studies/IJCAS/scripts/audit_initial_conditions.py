"""Post-block source/observed initialization audit; no simulator calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ijcas_paths import load_archived_json, resolve_archived_path

HERE = Path(__file__).resolve().parent
BLOCK = HERE.parent / 'results/hospital_paper_gate/revision_final_n3_seed30_v2_20261122'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation(row, names):
    if row is None:
        return None
    return dict(time_s=float(row['time_s']), phase=row['phase'],
        pose_xyyaw=[float(row[k]) for k in ('x_m', 'y_m', 'yaw_rad')],
        command_speed_m_s=float(row['command_speed_m_s']),
        joint_angles_deg={name: math.degrees(float(row['joint_' + name + '_rad']))
            for name in names if 'joint_' + name + '_rad' in row})


def audit():
    protocol = load_archived_json(BLOCK / 'execution_protocol.json')
    cells = protocol['main_cells']
    if len(cells) != 78 or not (BLOCK / 'completed.json').is_file():
        raise ValueError('All78 declared outcomes must finish before the final initialization audit')
    rows = []
    for cell in cells:
        outcome_path = BLOCK / 'outcomes' / (cell['cell_id'] + '.json')
        value = load_archived_json(outcome_path)
        row = dict(cell_id=cell['cell_id'], paired_condition_id=cell['paired_condition_id'],
            prescribed_source_angles_deg=cell['initial_joint_angles_deg'],
            prescribed_start_xyyaw=cell['start_xyyaw'],
            model_initial_active_steering_deg=[0., 0.],
            actual_outcome=value['actual_verdict'], input_sha256={str(outcome_path): sha(outcome_path)})
        report_path = resolve_archived_path(value['report']) if value.get('report') else None
        if report_path is None:
            row.update(status='NOT_MEASURED_NO_ACTUAL_TRIAL', first_bound=None,
                first_route_sample=None, first_path_tracking_sample=None, first_nonzero_drive_command_sample=None)
            rows.append(row)
            continue
        report = load_archived_json(report_path)
        initial_path = report_path.parent.parent / 'static/initial_joint_initialization_readback.json'
        if initial_path.is_file():
            initial = json.loads(initial_path.read_text())
            row['first_bound'] = {k: initial[k] for k in (
                'time_s', 'prescribed_new_source_angles_deg', 'first_bound_physics_angles_deg',
                'wrapped_difference_deg', 'passive_initialization_verified')}
            row['input_sha256'][str(initial_path)] = sha(initial_path)
        else:
            row['first_bound'] = None
        first = tracking = moving = None
        trace = resolve_archived_path(report['trace']) if report.get('trace') and value.get('actual_route_started') else None
        if trace and trace.is_file():
            if sha(trace) != report['trace_sha256']:
                raise ValueError('Actual trace digest mismatch: ' + str(trace))
            with trace.open(newline='') as stream:
                for sample in csv.DictReader(stream):
                    if first is None:
                        first = sample
                    if tracking is None and sample['phase'] == 'path_tracking':
                        tracking = sample
                    if moving is None and abs(float(sample['command_speed_m_s'])) > 1e-9:
                        moving = sample
                    if first is not None and tracking is not None and moving is not None:
                        break
            row['input_sha256'][str(trace)] = sha(trace)
        names = tuple(cell['initial_joint_angles_deg'])
        row.update(status='OBSERVED_RECORDS_EXTRACTED',
            first_route_sample=observation(first, names),
            first_path_tracking_sample=observation(tracking, names),
            first_nonzero_drive_command_sample=observation(moving, names),
            observed_missing_values_are_NOT_prescribed_or_invented_values=True)
        if row['first_route_sample'] is not None:
            pose = row['first_route_sample']['pose_xyyaw']; start = cell['start_xyyaw']
            row['first_route_vs_prescribed_start'] = dict(
                position_difference_m=math.hypot(pose[0]-start[0], pose[1]-start[1]),
                wrapped_yaw_difference_deg=math.degrees(math.atan2(
                    math.sin(pose[2]-start[2]), math.cos(pose[2]-start[2]))))
        rows.append(row)
    return dict(protocol=str(BLOCK / 'execution_protocol.json'),
        protocol_sha256=sha(BLOCK / 'execution_protocol.json'), rows=rows,
        note='Prescribed, first-BOUND, tracking-entry and first nonzero-command states are distinct; physical preparation may change angles. This is not an additional experiment.',
        new_experiments=0, simulator_commands=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE.parent/'reports/initialization_audit.json')
    args = parser.parse_args()
    result = audit()
    path = args.output
    if path.exists():
        raise FileExistsError('Preserve the previous audit; choose a separately versioned output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
