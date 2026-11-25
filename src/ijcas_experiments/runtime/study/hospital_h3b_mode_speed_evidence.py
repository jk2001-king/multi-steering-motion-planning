"""Reproduce four measured component outcomes; never infer a universal domain."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from hospital_h3_turn_planning import GATE, SUPPORT, verify_sources, load_setup
from hospital_h3b_mode_speed_review import ROLE, VERSION, evaluate_readiness, motion_stalled, curvature_at
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from hospital_static_gate import evaluate_static


CASES = ('front_only_50mm_s', 'coordinated_counterphase_50mm_s',
         'coordinated_counterphase_10mm_s', 'front_only_10mm_s')


def read_trace(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = []
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in ('phase', 'command_mode'):
                    row[key] = value
                elif key == 'position_hold_enabled':
                    if value not in ('True', 'False'):
                        raise ValueError('Invalid recorded hold state')
                    row[key] = value == 'True'
                else:
                    row[key] = float(value)
                    if not math.isfinite(row[key]):
                        raise ValueError('Nonfinite recorded telemetry')
            rows.append(row)
    if not rows or any(b['time_s'] <= a['time_s'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Finite monotone actual trace required')
    return rows


def inspect():
    manifest, occupied, origin, resolution, _ = load_setup()
    checker = NativeHospitalEnvelopeChecker(occupied, origin, resolution, manifest['start_xyyaw'])
    inputs, records, metrics = {}, {}, []
    for case in CASES:
        path = GATE/'hospital_h3b_mode_speed_components/cohort_01'/case/f'hospital_h3b_mode_speed_{case}_r01.json'
        record = json.loads(path.read_bytes()); verify_sources(record)
        if (record['trial_kind'] != ROLE or record['component_version'] != VERSION
                or record['component_case'] != case or not record['PhysX_executed']
                or not record['original_motor_gains_restored'] or not record['checks']
                or record['full_H3B_turning_path_verified'] or record['CICS_input_used']):
            raise ValueError('Expected completed-cleanup measured component, not whole route/CICS evidence')
        prep = json.loads(Path(record['component_preparation_reference']).read_bytes())
        verify_sources(prep); SUPPORT['verify_files'](prep['input_sha256'])
        inputs.update(prep['input_sha256'])
        for key,digest in (('component_preparation_reference','component_preparation_sha256'),
                ('static_reference','static_reference_sha256'),('geometry_manifest','geometry_manifest_sha256'),
                ('offline_model_review','offline_model_review_sha256')):
            SUPPORT['verify_files']({record[key]:record[digest]}); inputs[record[key]]=record[digest]
        static_path = Path(record['static_reference'])
        static = json.loads(static_path.read_bytes()); verify_sources(static)
        SUPPORT['verify_files']({static_path.with_suffix('.csv'):record['static_trace_sha256']})
        static_rows = read_trace(static_path.with_suffix('.csv'))
        if evaluate_static(static_rows,static['start_xyyaw'],static['simulation_mass_kg'],
                static['contact_report_counts_by_category'],static['contact_callback_errors'])['verdict']!='PASS':
            raise ValueError('Raw static PASS does not reproduce')
        rows = read_trace(path.with_suffix('.csv'))
        if len(rows)!=record['sample_count'] or rows[-1]!=record['last_recorded_state']:
            raise ValueError('Actual trace/final state mismatch')
        physics = record['physics_preparation_snapshot']; verify_sources(physics)
        SUPPORT['verify_files']({physics['source_stage']:physics['source_stage_sha256'],
            physics['robot_reference']:physics['robot_reference_sha256'],
            static['physics_preparation']:static['physics_preparation_sha256']})
        if checker._batch(np.array([[r['x_m'],r['y_m'],r['yaw_rad']] for r in rows]),(0.,0.,0.)):
            raise ValueError('Actual saved component envelope is blocked')
        errors = []
        for a,b in zip(rows,rows[1:]):
            if abs(b['requested_curvature_1_m']-curvature_at(a['longitudinal_m'],b['phase']))>1e-10:
                raise ValueError('Fixed-progress reference commands changed')
            errors.extend(abs(b[f'drive_{axle}_actual_velocity_target_rad_s']-
                b['command_speed_m_s']/(.074*math.cos(a[axle+'_rad']))) for axle in ('front','rear'))
        if max(errors,default=0.)>1e-6:
            raise ValueError('Applied drive targets do not reproduce')
        if record['verdict']=='PASS':
            evaluated = evaluate_readiness(rows,record['target_world_xyyaw'],static_rows,physics['simulation_mass_kg'],
                record['doorway_reference_bound_m'],record['wheel_conservative_sphere_radii_m'],
                record['contact_report_counts_by_category'],record['contact_callback_errors'],
                record['fixed_hold_attempt_count'],record['direction_gate_events'],case)
            for key,value in evaluated.items():
                if key=='checks':
                    if any(record['checks'].get(k)!=v for k,v in value.items()): raise ValueError('Checks differ')
                elif record.get(key)!=value: raise ValueError('Raw PASS differs: '+key)
            if not all(record['checks'].values()) or any(motion_stalled(rows[:i]) for i in range(1,len(rows)+1)):
                raise ValueError('A purported PASS violated a guard')
        elif (case!='coordinated_counterphase_10mm_s'
                or record.get('error')!='RuntimeError: Five-second measured forward-progress stall'
                or record['fixed_hold_attempt_count']!=0 or not motion_stalled(rows)
                or any(motion_stalled(rows[:i]) for i in range(1,len(rows)))):
            raise ValueError('The measured first-stall FAIL does not reproduce')
        if record['contact_callback_errors'] or any(v for k,v in record['contact_report_counts_by_category'].items() if k!='wheel_support'):
            raise ValueError('Unexpected contacts/callback failure')
        inputs.update({str(p):SUPPORT['sha'](p) for p in (path,path.with_suffix('.csv'),static_path.with_suffix('.csv'))})
        records[case] = record
        metrics.append({'case':case,'verdict':record['verdict'],'result':str(path),
            'terminal_position_error_m':record.get('terminal_position_error_m'),
            'terminal_heading_error_deg':record.get('terminal_heading_error_deg'),
            'last_progress_m':rows[-1]['longitudinal_m'],
            'initial_caster_joint_deg':{k:math.degrees(v) for k,v in record['initial_measured_caster_joint_angles_rad'].items()},
            'maximum_lateral_m':max(abs(r['lateral_m']) for r in rows),
            'maximum_heading_deg':max(abs(r['yaw_error_deg']) for r in rows),
            'original_gains_restored':True})
    reference = records[CASES[0]]
    common = ('fixed_progress_curvature_knots','start_world_xyyaw','readiness','guards','acceptance',
              'source_sha256','strict_flush_map_digest','initial_measured_caster_joint_angles_rad','static_trace_sha256')
    if any(record[k]!=reference[k] for record in records.values() for k in common):
        raise ValueError('Mixed source/start/profile/guard/initial-state comparison')
    if any(record['physics_preparation_snapshot']['live_support_readback']!=reference['physics_preparation_snapshot']['live_support_readback']
           for record in records.values()): raise ValueError('Physics/contact/motor baselines differ')
    return {'status':'FOUR_MEASURED_COMPONENT_OUTCOMES_REPRODUCED_NOT_WHOLE_ROUTE_DOMAIN',
        'cases':metrics,'input_sha256':inputs,'same_profile_initial_state_and_guards_verified':True,
        'low_speed_forward_mode_difference_observed_once':True,
        'mode_specific_reference_trajectory_not_identical_world_path':True,
        'reverse_turning_or_large_curvature_not_verified':True,
        'statistical_superiority_or_universal_speed_threshold_established':False,
        'validation_data_used_to_retune_rigid_body_model':False,
        'full_H3B_turning_path_verified':False,'new_PhysX_trial_performed':False,'use_for_final_statistics':False}
