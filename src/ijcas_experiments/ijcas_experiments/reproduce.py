"""Recompute the published selection without a simulator or result replacement."""
import csv
import gzip
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import FunctionType, SimpleNamespace

from ijcas_paths import ROOT, load_archived_json, resolve_archived_path

GATE = ROOT/'studies/IJCAS/results/hospital_paper_gate'
BLOCK = GATE/'revision_final_n3_seed30_v2_20261122'
SELECTED = BLOCK/'technical_camera_replacement_r01'
SENSOR_DATA = ROOT/'studies/IJCAS/reports/technical_evidence/figures/figures_02_06_source'


def sha(path):
    value = hashlib.sha256()
    with resolve_archived_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def describe(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
        sample_SD=statistics.stdev(values) if len(values)>1 else None, ddof=1)


def same_number(actual, expected, label):
    if ((actual is None) != (expected is None) or actual is not None and
            not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError('Recomputed value differs from the archived selection: '+label)


def main_results():
    closeout = load_archived_json(SELECTED/'FINAL_EXPERIMENT_CLOSEOUT.json')
    for path, expected in closeout['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Final closeout input digest mismatch: '+path)
    final = load_archived_json(SELECTED/'analysis_completed.json')
    saved = load_archived_json(Path(final['analysis_folder'])/'results.json')
    rows = saved['rows']
    if len(rows) != 78 or len({r['cell_id'] for r in rows}) != 78:
        raise ValueError('All72 main and six history outcomes are required')
    selection = load_archived_json(final['selection_manifest'])
    replacements = {r['cell_id']:r for r in selection['replacements']}
    if set(replacements) != {'H6_B1_multimode_kinematic_ramp_f02'}:
        raise ValueError('Only the approved camera interruption replacement is permitted')
    # The original metric implementation and reference interpolation are reused.
    # Only file I/O resolves archived absolute paths into this workspace.
    import hospital_revision_metrics as implementation
    runtime = SimpleNamespace(read=load_archived_json, sha=sha)
    metric = FunctionType(implementation.metrics.__code__,
        dict(implementation.metrics.__globals__,runtime=runtime),
        implementation.metrics.__name__,implementation.metrics.__defaults__,
        implementation.metrics.__closure__)
    pins = {}
    checked = 0
    for row in rows:
        outcome_path = BLOCK/'outcomes'/(row['cell_id']+'.json')
        if row['cell_id'] in replacements:
            replacement = replacements[row['cell_id']]
            for field in ('original_outcome','selected_outcome'):
                path = replacement['original_technical_outcome' if field=='original_outcome'
                    else 'selected_replacement_outcome']
                if sha(path) != replacement[field+'_sha256']:
                    raise ValueError('Technical replacement provenance changed')
            outcome_path = Path(replacement['selected_replacement_outcome'])
        outcome = load_archived_json(outcome_path)
        pins[str(outcome_path)] = sha(outcome_path)
        if not row['accounted'] or outcome.get('actual_verdict') != row['outcome']:
            raise ValueError('Missing or relabelled outcome: '+row['cell_id'])
        planning = load_archived_json(row['planning_summary'])
        same_number(row['planning_time_s'],planning['planning_time_s'],row['cell_id']+'/planning_time')
        if outcome.get('report'):
            report = load_archived_json(outcome['report'])
            recorded = outcome.get('metrics')
            if recorded:
                actual = metric(resolve_archived_path(outcome['report']))
                for name in ('sample_count','tracking_sample_count','tracking_duration_s',
                        'tracking_position_RMS_mm','tracking_yaw_RMS_deg',
                        'maximum_tracking_position_error_mm','maximum_absolute_tracking_yaw_error_deg',
                        'final_position_error_mm','final_yaw_error_deg','actual_distance_m',
                        'controller_elapsed_simulation_s'):
                    same_number(actual[name],recorded[name],row['cell_id']+'/'+name)
                for key, source in (('tracking_position_RMS_mm','tracking_position_RMS_mm'),
                        ('tracking_yaw_RMS_deg','tracking_yaw_RMS_deg'),
                        ('travel_sim_s','controller_elapsed_simulation_s'),
                        ('actual_distance_m','actual_distance_m'),
                        ('final_position_error_mm','final_position_error_mm'),
                        ('final_yaw_error_deg','final_yaw_error_deg')):
                    same_number(row[key],actual[source],row['cell_id']+'/'+key)
                    row[key] = actual[source]
                pins[actual['report']] = actual['report_sha256']
                pins[actual['trace']] = actual['trace_sha256']
                checked += 1
            completed = bool(outcome.get('actual_route_started') and outcome['actual_verdict']=='PASS'
                and report['checks'] and all(report['checks'].values()))
            if row['completed'] != completed:
                raise ValueError('Completion differs from the original full criterion')
        elif row['completed']:
            raise ValueError('No physical report for a claimed completed mission')
    groups = []
    for old in saved['groups']:
        group = dict(old)
        members = [r for r in rows if (r['task'],r['experiment_role'],r['method']) ==
            (old['task'],old['experiment_role'],old['method'])]
        passed = [r for r in members if r['completed']]
        if len(members) != 3 or len(passed) != old['completed_n']:
            raise ValueError('The original three-condition denominator must be retained')
        for name, field, subset in (
                ('planning_time_all_declared_attempts_s','planning_time_s',members),
                ('completed_tracking_position_RMS_mm','tracking_position_RMS_mm',passed),
                ('completed_tracking_yaw_RMS_deg','tracking_yaw_RMS_deg',passed),
                ('completed_travel_sim_s','travel_sim_s',passed)):
            value = describe([r[field] for r in subset])
            for statistic in ('mean','sample_SD'):
                same_number(value[statistic],old[name][statistic],name+'/'+statistic)
            group[name] = value
        groups.append(group)
    counts = {method:dict(completed=sum(r['completed'] for r in rows
        if r['experiment_role']=='MAIN' and r['method']==method),declared=18)
        for method in sorted({r['method'] for r in rows})}
    for method, value in counts.items():
        if value['completed'] != closeout['method_completion_counts'][method]['completed_n']:
            raise ValueError('Completion count differs from the fixed result')
    return dict(rows=rows,groups=groups,counts=counts,recomputed_physical_reports=checked,
        input_sha256=pins, protocol=load_archived_json(BLOCK/'protocol.json'),
        retained_negative_outcomes=True, approved_camera_replacement=selection,
        statistics='sample SD, ddof=1; RMS/travel conditional on completion with explicit n',
        planning_times='recorded wall-clock measurements, NOT a new timing experiment',
        RMS_interval='adjacent path_tracking samples only, trapezoidal squared-error integral',
        reference='original MissionPathReference.at(logged actual path_progress_m)',
        simulator_started=False,new_trials=0)


def sensor_results():
    result = {}
    expected = load_archived_json(SENSOR_DATA/'fig06_metrics.json')
    for task in ('H5','H6'):
        row = {}
        for kind, suffix, position, yaw, frequency in (
                ('localization','exact_matched_20Hz','localization_position_mm','localization_yaw_deg',20),
                ('actual_tracking','direct_60Hz','actual_tracking_position_mm','actual_tracking_yaw_deg',60)):
            path = SENSOR_DATA/(f'fig06_{task}_{suffix}.csv')
            with path.open(newline='') as stream:
                samples = list(csv.DictReader(stream))
            duration = sum(float(b['time_s'])-float(a['time_s']) for a,b in zip(samples,samples[1:])
                if a['phase']==b['phase']=='path_tracking')
            tracked = [s for s in samples if s['phase']=='path_tracking']
            if duration <= 0 or not tracked:
                raise ValueError('Missing path_tracking interval in '+str(path))
            for signal, column in (('position_RMS_mm',position),('yaw_RMS_deg',yaw)):
                integral = sum((float(b['time_s'])-float(a['time_s']))*
                    (float(a[column])**2+float(b[column])**2)/2
                    for a,b in zip(samples,samples[1:]) if a['phase']==b['phase']=='path_tracking')
                row[kind+'_'+signal] = math.sqrt(integral/duration)
            same_number(row[kind+'_position_RMS_mm'],expected[task][kind+'_position_RMS_mm'],task+'/'+kind)
            row[kind+'_samples'] = len(tracked)
            row[kind+'_duration_s'] = duration
            row[kind+'_source'] = str(path)
            row[kind+'_source_sha256'] = sha(path)
            row[kind+'_nominal_frequency_Hz'] = frequency
        result[task] = row
    return result


def raw_sensor_results(task, route=None, plan_path=None, output=None):
    """Recompute sensor scores from raw control receipts and physics trace.

    Exact integer stamps only: the held 20 Hz estimate in a 60 Hz physics
    row is NOT treated as a fresh localization sample. No interpolation or
    result-dependent reference registration is performed.
    """
    from .sensor_runner import sensor_selection
    from hospital_mission_execution import MissionPathReference
    spec = sensor_selection(task)
    route = Path(route) if route else Path(spec['route_template']).parent
    report_path = route / 'route_readback.json'
    report = load_archived_json(report_path)
    protocol = load_archived_json(route / 'protocol.json')
    plan_path = Path(plan_path) if plan_path else (
        route / 'frozen_plan.json' if (route / 'frozen_plan.json').exists() else Path(spec['frozen_plan']))
    plan = load_archived_json(plan_path)
    reference = MissionPathReference(plan)
    trace = route / 'physics_trace.csv'
    receipts = route / 'actual_control_receipts.jsonl.gz'
    if sha(receipts) != report['actual_control_receipts']['sha256']:
        raise ValueError('Actual control receipt archive digest changed')
    with trace.open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    if len(raw) != report['sample_count'] or not raw:
        raise ValueError('Complete original simulator trace required')
    direct, truth, distance, previous = [], {}, 0., None
    def wrapped_degrees(value):
        return math.degrees(math.atan2(math.sin(value), math.cos(value)))
    for row in raw:
        stamp = round(float(row['actual_timeline_s']) * 1e9)
        if stamp in truth:
            raise ValueError('Duplicate simulator stamp; no ambiguous pose match')
        truth[stamp] = row
        pose = [float(row[k]) for k in ('x_m', 'y_m', 'yaw_rad')]
        _, target = reference.at(float(row['path_progress_m']))
        sample = dict(time_s=float(row['time_s']), actual_timeline_s=float(row['actual_timeline_s']),
            phase=row['phase'], GT_x_m=pose[0], GT_y_m=pose[1], GT_yaw_rad=pose[2],
            reference_x_m=target[0], reference_y_m=target[1], reference_yaw_rad=target[2],
            actual_tracking_position_mm=1000*math.hypot(pose[0]-target[0], pose[1]-target[1]),
            actual_tracking_yaw_deg=wrapped_degrees(pose[2]-target[2]))
        if previous is not None:
            if sample['time_s'] <= previous['time_s']:
                raise ValueError('Strictly increasing simulator time required')
            distance += math.hypot(pose[0]-previous['GT_x_m'], pose[1]-previous['GT_y_m'])
        direct.append(sample)
        previous = sample
    matched = []
    with gzip.open(receipts, 'rt') as stream:
        for line in stream:
            receipt = json.loads(line)
            stamp = receipt['stamp_ns']
            feedback = receipt['proposal']['matched_sensor_feedback']
            if stamp != feedback['stamp_ns'] or stamp not in truth:
                raise ValueError('Exact same-time simulator pose required for each 20 Hz receipt')
            row = truth[stamp]
            estimated = feedback['estimated_map_xyyaw']
            pose = [float(row[k]) for k in ('x_m', 'y_m', 'yaw_rad')]
            matched.append(dict(time_s=float(row['time_s']), stamp_ns=stamp, phase=row['phase'],
                GT_x_m=pose[0], GT_y_m=pose[1], GT_yaw_rad=pose[2],
                estimated_x_m=estimated[0], estimated_y_m=estimated[1], estimated_yaw_rad=estimated[2],
                localization_position_mm=1000*math.hypot(estimated[0]-pose[0], estimated[1]-pose[1]),
                localization_yaw_deg=wrapped_degrees(estimated[2]-pose[2])))
    if len(matched) != report['actual_control_receipts']['records']:
        raise ValueError('Every actual 20 Hz receipt must be scored')
    def metrics(rows, prefix):
        tracked = [r for r in rows if r['phase'] == 'path_tracking']
        duration = 0.
        integrals = dict(position=0., yaw=0.)
        for a, b in zip(rows, rows[1:]):
            if b['time_s'] <= a['time_s']:
                raise ValueError('Strictly increasing scored timestamps required')
            if a['phase'] == b['phase'] == 'path_tracking':
                dt = b['time_s'] - a['time_s']
                duration += dt
                for name, unit in (('position', 'mm'), ('yaw', 'deg')):
                    key = prefix + '_' + name + '_' + unit
                    integrals[name] += dt * (a[key]**2 + b[key]**2) / 2
        result = dict(samples=len(tracked), duration_s=duration)
        for name, unit in (('position', 'mm'), ('yaw', 'deg')):
            key = prefix + '_' + name + '_' + unit
            result[name + '_RMS_' + unit] = math.sqrt(integrals[name]/duration) if duration > 0 else None
            result['maximum_absolute_' + name + '_' + unit] = max(
                (abs(r[key]) for r in tracked), default=None)
        return result
    last, goal = direct[-1], plan['goal_world_xyyaw']
    result = dict(task=task, actual_verdict=report['verdict'],
        all_physical_and_ROS_completion_checks_passed=(report['verdict'] == 'PASS'
            and bool(report['checks']) and all(report['checks'].values())),
        completion_checks=report['checks'],
        localization=metrics(matched, 'localization'), actual_tracking=metrics(direct, 'actual_tracking'),
        localization_frequency_Hz=20, actual_tracking_frequency_Hz=60,
        final_position_error_mm=1000*math.hypot(last['GT_x_m']-goal[0], last['GT_y_m']-goal[1]),
        final_yaw_error_deg=wrapped_degrees(last['GT_yaw_rad']-goal[2]), actual_distance_m=distance,
        travel_simulation_s=report['actual_final_timeline']['current_time_s']-
            protocol['required_PAUSED_timeline']['current_time_s'],
        evaluation_interval='path_tracking; adjacent included samples only',
        time_weighting='trapezoidal integral of squared error divided by included duration',
        angle_difference='atan2(sin(yaw difference),cos(yaw difference))',
        reference='MissionPathReference.at(logged actual simulator path_progress_m)',
        localization_match='exact integer ns; no nearest match/interpolation/held estimate scoring',
        new_trials=0, paper_results_replaced=False,
        input_sha256={str(p):sha(p) for p in (report_path, route/'protocol.json', plan_path, trace, receipts)})
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / 'localization_exact_20Hz.csv', matched)
        write_csv(output / 'actual_tracking_60Hz.csv', direct)
        with (output / 'metrics.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
            for k,v in row.items()} for row in rows)


def export(output=None):
    folder = Path(output).resolve() if output else ROOT/'results/reproduced'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    protected = ROOT/'studies'
    if folder.is_relative_to(protected) or folder.exists():
        raise FileExistsError('Use a new output directory outside the archived study')
    main = main_results()
    sensors = sensor_results()
    raw_sensors = {task:raw_sensor_results(task) for task in ('H5', 'H6')}
    for task, row in sensors.items():
        for kind in ('localization', 'actual_tracking'):
            for signal in ('position_RMS_mm', 'yaw_RMS_deg'):
                same_number(raw_sensors[task][kind][signal], row[kind+'_'+signal],
                    task+'/raw_'+kind+'_'+signal)
    folder.mkdir(parents=True,exist_ok=False)
    write_csv(folder/'individual.csv',main['rows'])
    write_csv(folder/'summary.csv',main['groups'])
    write_csv(folder/'history_ablation.csv',[r for r in main['groups'] if r['experiment_role']=='HISTORY_ABLATION'])
    details = load_archived_json(load_archived_json(SELECTED/'analysis_completed.json')['detailed_feedback_analysis'])
    for row in details['force_load_constraints']:
        if sha(row['plan']) != row['plan_sha256']:
            raise ValueError('Constraint input plan digest changed')
        plan = load_archived_json(row['plan'])
        states = plan.get('selected_states',[])
        for output_name, field in (('maximum_selected_friction_utilization','friction_utilization'),
                ('maximum_selected_abs_LTR','abs_ltr')):
            maximum = max((s[field] for s in states),default=None)
            same_number(maximum,row[output_name],row['cell_id']+'/'+output_name)
            row[output_name] = maximum
    write_csv(folder/'constraints.csv',details['force_load_constraints'])
    write_csv(folder/'offline_envelope.csv',details['offline_envelope_counterfactuals'])
    (folder/'sensor_results.json').write_text(json.dumps(sensors,indent=2)+'\n')
    (folder/'sensor_raw_results.json').write_text(json.dumps(raw_sensors,indent=2)+'\n')
    (folder/'results.json').write_text(json.dumps(main,indent=2)+'\n')
    outputs = {p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    (folder/'completed.json').write_text(json.dumps(dict(
        status='ARCHIVED_RESULTS_RECOMPUTED_NOT_NEW_EXPERIMENTS',
        all78_selected_conditions_checked=True,
        physical_reports_recomputed=main['recomputed_physical_reports'],
        output_sha256=outputs,simulator_started=False,new_trials=0),indent=2)+'\n')
    return folder, main['counts']
