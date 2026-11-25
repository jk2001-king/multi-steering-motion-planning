#!/usr/bin/env python3
"""Fixed reserve candidate: signed dynamics + spatial-reference first-leg review.

NO PhysX replay/permission, retuning, new gains, changed map or caster reset.
First-leg feedback follows the NEW SPATIAL PLAN, not the old predicted model
trace. Fixed reference source/target changes are explicit, not a like-for-like
comparison with the failed feedforward trial or a full-route controller proof.
"""

import csv
import json
from pathlib import Path

import numpy as np

from hospital_h3_turn_planning import GATE,WORKSPACE,SUPPORT,verify_sources,load_setup,model_bounds
from hospital_h3b_directional_playback import playback
from hospital_h3b_first_leg_feedback_review import model_probe,BIAS_CASES,INITIAL_CASES,TRACKING_GAINS
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from audit_hospital_h3b_directional_playback import collision_diagnostic


CANDIDATE=GATE/'hospital_h3b_clearance_reserve_seed/seed_r01/reserve_candidate.json'
VERSION='H3B_reserve_fixed_candidate_directional_and_first_leg_feedback_model_review_v1'


def load_candidate(path=CANDIDATE):
    path=Path(path).resolve();record=json.loads(path.read_bytes());verify_sources(record)
    SUPPORT['verify_files'](record['input_sha256'])
    if (record.get('status')!='RESERVE_GEOMETRY_AND_NOMINAL_TIMING_ADMITTED_HOLD_DYNAMIC_AND_EXECUTION_REVIEW'
            or record.get('PhysX_executed') is not False or record.get('automatic_PhysX_execution_authorized') is not False
            or record.get('nominal_search_clearance_m')!=.03 or record.get('nominal_curvature_cap_1_m')!=.6
            or not all(record['geometry_audit']['checks'].values()) or not all(record['nominal_timing']['checks'].values())):
        raise ValueError('Expected the actual admitted spatial reserve candidate, NOT a dynamics/PhysX certificate')
    manifest,occupied,origin,resolution,_=load_setup()
    if (record['unchanged_start_xyyaw']!=manifest['start_xyyaw'] or record['unchanged_goal_xyyaw']!=manifest['goal_xyyaw']
            or record['unchanged_strict_map_digest']!=manifest['planning_occupancy_digest']):
        raise ValueError('Original H3 task/map changed')
    return record,manifest,occupied,origin,resolution


def spatial_first_leg_info(record):
    geometry,timing=record['geometry_audit'],record['nominal_timing']
    stop=timing['cusp_sample_indices'][0]
    if (any(d!=1 for d in geometry['interval_directions'][:stop])
            or any(m!='coordinated_counterphase' for m in geometry['interval_modes'][:stop])
            or abs(geometry['sample_curvature_1_m'][stop])>1e-10):
        raise ValueError('Expected a straight-cusp-ended counterphase FIRST FORWARD leg')
    points=geometry['poses_local_xyyaw']
    rows=[dict(leg_index=0,command_mode='coordinated_counterphase',
        command_speed_m_s=timing['sample_unsigned_reference_speed_m_s'][i],
        requested_curvature_1_m=geometry['sample_curvature_1_m'][i],
        x_m=points[i][0],y_m=points[i][1],yaw_rad=points[i][2]) for i in range(stop+1)]
    return {'reference_kind':'NEW_FROZEN_SPATIAL_PLAN_FIRST_LEG_NOT_MODEL_OR_MEASURED_TRACE',
        'legacy_model_prefix_rows_key_is_schema_adapter_ONLY':True,
        'model_prefix_rows':rows,'first_stop_model_local_xyyaw':list(points[stop]),
        'fixed_component_goal_is_new_nominal_cusp_not_old_predicted_stop':True,
        'original_full_mission_goal_unchanged':record['unchanged_goal_xyyaw']}


def save_trace(output,name,result,rows):
    path=output/(name+'.csv')
    if rows:
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result.update(trace_file=str(path),trace_sha256=SUPPORT['sha'](path),sample_count=len(rows))


def main():
    candidate,manifest,occupied,origin,resolution=load_candidate();info=spatial_first_leg_info(candidate)
    base=GATE/'hospital_h3b_reserve_execution_reviews';base.mkdir(parents=True,exist_ok=True);number=1
    while (output:=base/f'review_r{number:02d}').exists():number+=1
    output.mkdir()
    failed=GATE/'hospital_h3b_first_leg_model_pilot/cohort_01/hospital_h3b_first_forward_leg_r01.json'
    header={'version':VERSION,'role':'MODEL_ONLY_FIXED_RESERVE_CANDIDATE_REVIEW_NOT_PhysX_DRIVE_AUTHORIZATION',
        'candidate':str(CANDIDATE),'candidate_sha256':SUPPORT['sha'](CANDIDATE),
        'unchanged_mission_start_xyyaw':manifest['start_xyyaw'],'unchanged_mission_goal_xyyaw':manifest['goal_xyyaw'],
        'strict_map_digest':manifest['planning_occupancy_digest'],'model_parameters':model_bounds(),
        'predeclared_dt_s':[.02,.01],'predeclared_synthetic_bias_cases_deg':BIAS_CASES,
        'predeclared_initial_pose_velocity_cases':INITIAL_CASES,'tracking_gains':TRACKING_GAINS,
        'first_leg_reference_kind':info['reference_kind'],
        'first_leg_fixed_nominal_cusp_local_xyyaw':info['first_stop_model_local_xyyaw'],
        'first_leg_target_differs_from_old_feedforward_target_explicitly':True,
        'strict_clearance_m':.010,'additional_nominal_reserve_diagnostic_m':.020,
        'synthetic_cases_are_not_identified_uncertainty_bounds_or_statistics':True,
        'full_reference_is_feedforward_diagnostic_NOT_a_whole_route_feedback_controller':True,
        'no_model_gain_map_goal_caster_or_old_outcome_change':True,
        'PhysX_executed':False,'full_H3B_turning_path_verified':False,'reverse_turning_verified':False,
        'automatic_PhysX_execution_authorized':False,'use_for_final_statistics':False,
        'input_sha256':{**candidate['input_sha256'],**{str(p):SUPPORT['sha'](p) for p in (CANDIDATE,failed,failed.with_suffix('.csv'))}},
        'source_sha256':{p.name:SUPPORT['sha'](p) for p in (Path(__file__),Path(__file__).with_name('hospital_h3b_directional_playback.py'),
            Path(__file__).with_name('hospital_h3b_first_leg_feedback_review.py'),Path(__file__).with_name('hospital_path_execution.py'))}}
    with (output/'contract.json').open('x') as stream:json.dump(header,stream,indent=2,allow_nan=False);stream.write('\n')
    print('=== H3-B RESERVE CANDIDATE EXECUTION MODEL REVIEW (NO MOTION) ===',flush=True)
    print('output:',output,flush=True)
    strict=NativeHospitalEnvelopeChecker(occupied,origin,resolution,manifest['start_xyyaw'])
    reserve=NativeHospitalEnvelopeChecker(occupied,origin,resolution,manifest['start_xyyaw'],clearance=.03)
    whole=[];first=[]
    try:
        for dt in (.02,.01):
            result,rows=playback(candidate['geometry_audit'],candidate['nominal_timing'],model_bounds(),manifest['local_goal_xyyaw'],dt)
            clearance=collision_diagnostic(rows,strict,occupied,origin,resolution,manifest['start_xyyaw'])
            result['sampled_envelope_diagnostic']=clearance
            result['model_checks']['sampled_strict_model_envelope_clear']=clearance['sampled_model_envelope_clear']
            result['reserve30mm_model_clear']=not reserve([[r[k] for k in ('x_m','y_m','yaw_rad')] for r in rows])
            result['model_case_admitted']=all(result['model_checks'].values())
            save_trace(output,f'whole_fixed_reference_{round(dt*1000)}ms',result,rows);whole.append(result)
            print(f"WHOLE MODEL {round(dt*1000)}ms: {'PASS' if result['model_case_admitted'] else 'HOLD'} "
                f"terminal={1000*result['model_terminal_position_error_m']:.2f}mm/{result['model_terminal_heading_error_deg']:.3f}deg "
                f"strict_clear={clearance['sampled_model_envelope_clear']}",flush=True)
            for bias in BIAS_CASES:
                for initial in INITIAL_CASES:
                    result,rows=model_probe(info,dt,bias,initial)
                    clearance=collision_diagnostic(rows,strict,occupied,origin,resolution,manifest['start_xyyaw'])
                    result['sampled_envelope_diagnostic']=clearance
                    extra_clear=not reserve([[r[k] for k in ('x_m','y_m','yaw_rad')] for r in rows])
                    result['model_checks'].update(sampled_strict_model_envelope_clear=clearance['sampled_model_envelope_clear'],
                        additional20mm_rectangular_model_margin_clear=extra_clear)
                    result['model_case_admitted']=all(result['model_checks'].values())
                    save_trace(output,f'first_{bias}_{initial}_{round(dt*1000)}ms',result,rows);first.append(result)
                    print(f"FIRST MODEL {round(dt*1000)}ms {bias}/{initial}: {'PASS' if result['model_case_admitted'] else 'HOLD'} "
                        f"terminal={1000*result['terminal_position_error_m']:.1f}mm/{result['terminal_heading_error_deg']:.2f}deg "
                        f"strict_clear={clearance['sampled_model_envelope_clear']} reserve_clear={extra_clear}",flush=True)
        admitted=all(r['model_case_admitted'] for r in whole+first)
        report={**header,'whole_fixed_reference_model_cases':whole,'first_leg_feedback_model_cases':first,
            'all_predeclared_model_cases_admitted':admitted,
            'status':'MODEL_CASES_ADMITTED_STILL_NOT_PhysX_CERTIFIED' if admitted
                else 'HOLD_RESERVE_GEOMETRY_NOT_SUFFICIENT_FOR_FROZEN_MODEL_EXECUTION_DOMAIN',
            'not_global_H3_impossibility_and_not_a_new_measured_failure':True,
            'next_step':'MEASURED_STEERING_AND_BODY_RESPONSE_MODEL_CONSISTENCY_AUDIT_BEFORE_MORE_RETRIES'}
        SUPPORT['verify_files'](header['input_sha256'])
        target=output/'execution_review.json'
        with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        print('FIRST MODEL cases admitted:',sum(r['model_case_admitted'] for r in first),'/',len(first),flush=True)
        print('STATUS:',report['status'],flush=True);print('saved:',target,flush=True)
        print('Keep PAUSED. No new trial, parameter fit, guard relaxation, CICS/Nav2 or full-route/reverse PhysX claim.',flush=True)
    except Exception as error:
        with (output/'failure.json').open('x') as stream:
            json.dump({'status':'OFFLINE_REVIEW_ERROR_NO_PhysX','error':f'{type(error).__name__}: {error}'},stream)
        raise


if __name__=='__main__':main()
