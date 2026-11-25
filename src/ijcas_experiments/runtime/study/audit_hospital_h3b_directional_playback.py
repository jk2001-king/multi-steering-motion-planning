#!/usr/bin/env python3
"""Read-only fixed-reference model diagnosis; cannot authorize Isaac driving."""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_hospital_h3b_mode_aware_seed import inspect as inspect_seed
from hospital_h3b_directional_playback import VERSION,playback
from hospital_h3_turn_planning import GATE,WORKSPACE,MODEL_PLAN,SUPPORT,load_setup,model_bounds
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from hospital_geometry_tools import pose_collision
from hospital_planning_tools import local_to_world


def collision_diagnostic(rows,checker,occupied,origin,resolution,start):
    poses=np.array([[r['x_m'],r['y_m'],r['yaw_rad']] for r in rows])
    blocked=checker(poses)
    if not blocked:
        if any(pose_collision(occupied,local_to_world(p,start),origin,resolution,.010) for p in poses):
            raise ValueError('Native/original sampled-envelope collision disagreement')
        return {'sampled_model_envelope_clear':True,'first_predicted_clearance_violation':None,
            'independent_original_polygon_SAT_run':True,'predicted_violation_is_NOT_a_PhysX_contact':True}
    low,high=0,len(poses)-1
    while low<high:
        middle=(low+high)//2
        if checker(poses[:middle+1]):high=middle
        else:low=middle+1
    if not pose_collision(occupied,local_to_world(poses[low],start),origin,resolution,.010):
        raise ValueError('First model clearance violation does not reproduce with original polygon SAT')
    row=rows[low]
    return {'sampled_model_envelope_clear':False,'first_predicted_clearance_violation':{
        'sample_index':low,'time_s':row['time_s'],'phase':row['phase'],'leg_index':row['leg_index'],
        'local_xyyaw':poses[low].tolist(),'world_xyyaw':local_to_world(poses[low],start),
        'command_mode':row['command_mode'],'command_speed_m_s':row['command_speed_m_s'],
        'reference_position_error_m':row['reference_position_error_m']},
        'independent_original_polygon_SAT_run':True,'predicted_violation_is_NOT_a_PhysX_contact':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,default=GATE/'hospital_h3b_mode_aware_seed/seed_r01/audits/mode_aware_audit_r01.json')
    args=parser.parse_args();audit_path=args.audit.resolve()
    saved=json.loads(audit_path.read_bytes())
    audit=inspect_seed(audit_path.parent.parent)
    if saved!=audit:raise ValueError('Selected independent spatial audit changed or does not reproduce')
    manifest,occupied,origin,resolution,_=load_setup(audit['geometry_manifest'],audit['preview_receipt'])
    checker=NativeHospitalEnvelopeChecker(occupied,origin,resolution,manifest['start_xyyaw'])
    bounds=model_bounds()
    base=GATE/'hospital_h3b_directional_model_playback';base.mkdir(parents=True,exist_ok=True)
    number=1
    while (output:=base/f'playback_r{number:02d}').exists():number+=1
    output.mkdir()
    # Both integration resolutions are predeclared; never select the more
    # favourable one or replace a measured outcome with a model result.
    dts=(.020,.010)
    inputs={**audit['input_sha256'],str(audit_path):SUPPORT['sha'](audit_path),str(MODEL_PLAN):SUPPORT['sha'](MODEL_PLAN)}
    for relative in ('src/patient_initial_stability/patient_initial_stability/rigid_body_edge.py',
            'src/patient_initial_stability/patient_initial_stability/steering_lattice.py',
            'studies/IJCAS/scripts/hospital_path_execution.py','studies/IJCAS/scripts/hospital_exact_sat.c',
            'studies/IJCAS/scripts/hospital_geometry_tools.py','studies/IJCAS/scripts/hospital_native_envelope.py'):
        path=_ijcas_path(WORKSPACE, relative);inputs[str(path)]=SUPPORT['sha'](path)
    report={'version':VERSION,'role':'OPEN_LOOP_MODEL_DIAGNOSTIC_NOT_PHYSICAL_EXECUTION_GATE',
        'spatial_audit':str(audit_path),'spatial_audit_sha256':SUPPORT['sha'](audit_path),
        'selected_spatial_family':audit['selected_family'],'geometry_and_nominal_timing_verdict':'PASS',
        'fixed_start_xyyaw':manifest['start_xyyaw'],'fixed_goal_xyyaw':manifest['goal_xyyaw'],
        'strict_map_digest':manifest['planning_occupancy_digest'],'rigid_body_parameter_bounds':bounds,
        'predeclared_integration_steps_s':dts,'input_sha256':inputs,
        'source_sha256':{p.name:SUPPORT['sha'](p) for p in (Path(__file__),Path(__file__).with_name('hospital_h3b_directional_playback.py'))},
        'model_initial_state_assumption':'original task pose; zero body velocity and active steering; NOT current live robot state',
        'known_low_speed_counterphase_model_false_positive_not_erased':True,
        'physical_reverse_or_large_curvature_or_whole_route_verified':False,
        'captured_wheel_hold_and_release_are_not_simulated_or_measured':True,
        'no_model_retuning_or_control_feedback_added':True,'no_pose_velocity_or_joint_resets_at_cusps':True,
        'PhysX_executed':False,'full_dynamics_plan_certified':False,'full_H3B_turning_path_verified':False,
        'CICS_input_used':False,'Nav2_executed':False,'use_for_final_statistics':False,
        'model_violation_is_NOT_measured_PhysX_failure_or_global_infeasibility':True}
    print('=== H3-B DIRECTIONAL FIXED-REFERENCE MODEL DIAGNOSTIC ===',flush=True)
    print('output:',output,flush=True)
    cases=[]
    try:
        for dt in dts:
            diagnostic,rows=playback(audit['geometry_audit'],audit['nominal_timing'],bounds,manifest['local_goal_xyyaw'],dt)
            collision=collision_diagnostic(rows,checker,occupied,origin,resolution,manifest['start_xyyaw'])
            diagnostic['sampled_envelope_diagnostic']=collision
            diagnostic['model_checks']['sampled_model_full_envelope_clear']=collision['sampled_model_envelope_clear']
            diagnostic['model_playback_verdict']='PASS' if all(diagnostic['model_checks'].values()) else 'FAIL'
            trace=output/f'model_dt_{round(dt*1000):02d}ms.csv'
            with trace.open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            diagnostic.update(trace_file=str(trace),trace_sha256=SUPPORT['sha'](trace),sample_count=len(rows))
            cases.append(diagnostic)
            print('dt s:',dt,'MODEL ONLY:',diagnostic['model_playback_verdict'],
                'terminal mm/deg:',1000*diagnostic['model_terminal_position_error_m'],diagnostic['model_terminal_heading_error_deg'],flush=True)
            print('  force-load limits:',diagnostic['model_checks']['inherited_force_load_limits'],
                'sampled envelope clear:',collision['sampled_model_envelope_clear'],flush=True)
            print('  failed MODEL checks:',[k for k,v in diagnostic['model_checks'].items() if not v],flush=True)
        coarse,fine=cases
        error=float(np.linalg.norm(np.array(coarse['final_model_local_xyyaw'][:2])-fine['final_model_local_xyyaw'][:2]))
        yaw=abs(coarse['final_model_local_xyyaw'][2]-fine['final_model_local_xyyaw'][2])
        report.update(cases=cases,integration_resolution_comparison={'terminal_position_difference_m':error,
            'terminal_heading_difference_deg':float(np.degrees(yaw)),'neither_resolution_used_to_relabel_measured_outcomes':True},
            status='HOLD_FIXED_REFERENCE_MODEL_DEVIATION_AND_UNVALIDATED_EXECUTION_DOMAIN' if any(c['model_playback_verdict']=='FAIL' for c in cases)
                else 'HOLD_MODEL_ONLY_REVERSE_LARGE_CURVATURE_AND_ACTUATOR_HANDSHAKE_UNVERIFIED')
        SUPPORT['verify_files'](inputs)
        path=output/'diagnostic.json'
        with path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        print('STATUS:',report['status'],flush=True);print('saved:',path,flush=True)
        print('No Isaac/ROS motion, model retuning or source USD change. Keep STOPPED. Do not execute this CSV as a plan.',flush=True)
    except Exception as error:
        with (output/'failure.json').open('x') as stream:json.dump({'status':'MODEL_DIAGNOSTIC_ERROR_NO_PHYSICAL_TRIAL','error':f'{type(error).__name__}: {error}'},stream)
        raise


if __name__=='__main__':main()
