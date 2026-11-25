"""Read-only post-hoc constant-command sensitivity, not a replacement metric.

Retain native adjacent samples: never integrate across excluded waiting or
ramp intervals. This does not assert identical observed speed or identify
an isolated component effect. Main whole-tracking RMS and all failures stay.
"""
import csv
import math
from pathlib import Path

import hospital_revision_metrics as original


def summarize(rows, cruise_m_s=.030):
    def selected(r):
        return r['phase']=='path_tracking' and abs(abs(r['command_speed_m_s'])-cruise_m_s)<=1e-6
    duration=position=yaw=velocity=0.
    for a,b in zip(rows, rows[1:]):
        if not (selected(a) and selected(b)): continue
        dt=b['time_s']-a['time_s']
        if dt<=0: raise ValueError('Strictly increasing native timestamps required')
        duration+=dt
        position+=dt*(a['position_error_m']**2+b['position_error_m']**2)/2
        yaw+=dt*(a['yaw_error_rad']**2+b['yaw_error_rad']**2)/2
        velocity+=dt*(a['observed_planar_speed_m_s']+b['observed_planar_speed_m_s'])/2
    return dict(duration_s=duration,selected_samples=sum(map(selected,rows)),
        position_RMS_mm=1000*math.sqrt(position/duration) if duration else None,
        yaw_RMS_deg=math.degrees(math.sqrt(yaw/duration)) if duration else None,
        observed_planar_speed_mean_mm_s=1000*velocity/duration if duration else None,
        common_command_magnitude_mm_s=cruise_m_s*1000,
        selection=f'path_tracking AND abs(abs(command_speed)-{cruise_m_s:.6f})<=1e-6 m/s at BOTH native interval endpoints',
        posthoc_sensitivity_NOT_primary_RMS_or_isolated_causal_effect=True,
        excluded_gaps_never_bridged=True,does_NOT_require_identical_observed_speed=True)


def measure(report_path):
    path=Path(report_path);runtime=original.runtime
    report=runtime.read(path);packet=runtime.read(report['protocol'])
    if report['verdict']!='PASS' or not all(report['checks'].values()):
        raise ValueError('Only completed full missions; no partial/full-task ranking')
    if runtime.sha(report['trace'])!=report['trace_sha256']: raise ValueError('Original trace changed')
    reference=original.MissionPathReference(runtime.read(path.parent/packet['plan_filename']))
    rows=[]
    with Path(report['trace']).open(newline='') as stream:
        for r in csv.DictReader(stream):
            _,pose=reference.at(float(r['path_progress_m']))
            yaw=float(r['yaw_rad'])-pose[2]
            rows.append(dict(time_s=float(r['time_s']),phase=r['phase'],command_speed_m_s=float(r['command_speed_m_s']),
                position_error_m=math.hypot(float(r['x_m'])-pose[0],float(r['y_m'])-pose[1]),
                yaw_error_rad=math.atan2(math.sin(yaw),math.cos(yaw)),
                observed_planar_speed_m_s=math.hypot(float(r['vx_physx_m_s']),float(r['vy_physx_m_s']))))
    return dict(summarize(rows),report=str(path),report_sha256=runtime.sha(path),
        trace=report['trace'],trace_sha256=report['trace_sha256'],
        source_sha256=runtime.sha(Path(__file__)),reference_point='base_link',
        reference_correspondence='same MissionPathReference.at(actual path_progress_m) as primary RMS')
