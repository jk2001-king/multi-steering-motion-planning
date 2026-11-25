"""Append-only metrics from actual guarded traces; never run an experiment."""
import csv
import json
import math
from pathlib import Path

import hospital_controlled_runtime as runtime
from hospital_mission_execution import MissionPathReference


def metrics(report_path):
    report_path=Path(report_path).resolve()
    report=runtime.read(report_path)
    packet=runtime.read(report['protocol'])
    trace=Path(report['trace'])
    if runtime.sha(trace)!=report['trace_sha256']:
        raise ValueError('Actual trace digest changed')
    plan_path=report_path.parent/packet['plan_filename']
    plan=runtime.read(plan_path)
    reference=MissionPathReference(plan)
    with trace.open(newline='') as stream:
        raw=list(csv.DictReader(stream))
    rows=[]
    distance=0.
    previous=None
    for raw_row in raw:
        row={k:float(raw_row[k]) for k in ('time_s','x_m','y_m','yaw_rad',
            'path_progress_m','command_speed_m_s','vx_physx_m_s','vy_physx_m_s')}
        row['phase']=raw_row['phase']
        _,pose=reference.at(row['path_progress_m'])
        row['position_error_m']=math.hypot(row['x_m']-pose[0],row['y_m']-pose[1])
        row['yaw_error_rad']=math.atan2(math.sin(row['yaw_rad']-pose[2]),math.cos(row['yaw_rad']-pose[2]))
        row['observed_body_u_m_s']=math.cos(row['yaw_rad'])*row['vx_physx_m_s']+math.sin(row['yaw_rad'])*row['vy_physx_m_s']
        row['observed_planar_speed_m_s']=math.hypot(row['vx_physx_m_s'],row['vy_physx_m_s'])
        if previous is not None:
            if row['time_s']<=previous['time_s']:
                raise ValueError('Strictly increasing actual timestamps required')
            distance+=math.hypot(row['x_m']-previous['x_m'],row['y_m']-previous['y_m'])
        previous=row
        rows.append(row)
    integral=dict(position_error_m=0.,yaw_error_rad=0.)
    means=dict(command_speed_m_s=0.,observed_body_u_m_s=0.,observed_planar_speed_m_s=0.)
    duration=0.
    for a,b in zip(rows,rows[1:]):
        if a['phase']!= 'path_tracking' or b['phase']!='path_tracking':
            continue
        dt=b['time_s']-a['time_s']
        duration+=dt
        for k in integral:
            integral[k]+=dt*(a[k]**2+b[k]**2)/2
        for k in means:
            means[k]+=dt*(abs(a[k])+abs(b[k]))/2
    tracking=[r for r in rows if r['phase']=='path_tracking']
    last=rows[-1]
    goal=plan['goal_world_xyyaw']
    return dict(report=str(report_path),report_sha256=runtime.sha(report_path),
        trace=str(trace),trace_sha256=runtime.sha(trace),plan=str(plan_path),
        actual_verdict=report['verdict'],sample_count=len(rows),tracking_sample_count=len(tracking),
        tracking_duration_s=duration,
        tracking_position_RMS_mm=1000*math.sqrt(integral['position_error_m']/duration) if duration>0 else None,
        tracking_yaw_RMS_deg=math.degrees(math.sqrt(integral['yaw_error_rad']/duration)) if duration>0 else None,
        maximum_tracking_position_error_mm=1000*max(r['position_error_m'] for r in tracking) if tracking else None,
        maximum_absolute_tracking_yaw_error_deg=math.degrees(max(abs(r['yaw_error_rad']) for r in tracking)) if tracking else None,
        tracking_time_weighted_mean_absolute_speed_mm_s={k:1000*v/duration if duration>0 else None for k,v in means.items()},
        tracking_maximum_command_mm_s=1000*max(abs(r['command_speed_m_s']) for r in tracking) if tracking else None,
        tracking_metric_status='MEASURED' if duration>0 else 'NOT_MEASURED_NO_PATH_TRACKING_INTERVAL',
        final_position_error_mm=1000*math.hypot(last['x_m']-goal[0],last['y_m']-goal[1]),
        final_yaw_error_deg=math.degrees(math.atan2(math.sin(last['yaw_rad']-goal[2]),math.cos(last['yaw_rad']-goal[2]))),
        actual_distance_m=distance,logged_interval_duration_s=last['time_s']-rows[0]['time_s'],
        controller_elapsed_simulation_s=report['actual_final_timeline']['current_time_s']-
            packet['required_PAUSED_timeline']['current_time_s'],
        scope=('actual simulator/base_link tracking; newly predeclared primary trial' if
            report.get('ground_truth_primary_predeclared_NOT_sensor_Nav2') is True and
            report.get('use_for_final_statistics') is True else
            'actual simulator/base_link tracking; engineering pilot NOT final repetitions'),
        primary_cell=report.get('predeclared_campaign',{}).get('cell'),
        stationary_hold_and_all_final_checks_satisfied=report['verdict']=='PASS' and all(report['checks'].values()),
        reference_correspondence='same MissionPathReference.at(logged actual controller path_progress_m)',
        time_weighting='trapezoidal squared errors, adjacent path_tracking samples only',
        angles='atan2(sin(delta_yaw),cos(delta_yaw))',
        excludes_pre_hold_terminal_correction_and_final_stationary_hold_from_RMS=True,
        velocity_point='PhysX root CoM velocity, rotated into body axes; not base_link point velocity')


def export(report_path,output=None):
    report_path=Path(report_path).resolve()
    value=metrics(report_path)
    target=Path(output).resolve() if output is not None else report_path.parent/'controlled_actual_metrics.json'
    runtime.evidence.write_exclusive(target,value)
    print(json.dumps(value,indent=2))
    return target


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    export(args.report,args.output)
