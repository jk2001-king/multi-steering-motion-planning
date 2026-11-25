#!/usr/bin/env python3
"""Freeze an OFFLINE adapter contract; no Isaac imports or robot command."""

import json
from pathlib import Path

import numpy as np

import hospital_h3b_timed_second_reverse as h


def prepare_adapter_model():
    if h.ADAPTER_MODEL.exists():
        h.load_adapter_reviewed_profile()
        return
    from hospital_h3b_timed_second_reverse_model import model_case
    from audit_hospital_h3b_directional_playback import collision_diagnostic
    from review_hospital_h3b_reserve_execution import save_trace
    info = h.load_profile()
    _, _, manifest, _, _ = h.previous.inspect_forward()
    with np.load(manifest['occupancy_grid']) as grid:
        occupied=grid['occupied'].copy();origin=grid['origin_xy_m'].copy();resolution=float(grid['resolution_m'])
    checker=h.previous.NativeHospitalEnvelopeChecker(occupied,origin,resolution,info['mission_start_xyyaw'])
    contract=h.controller_and_adapter_contract()
    header=dict(version=h.VERSION, role='EXACT_TIMED_SECOND_REVERSE_ADAPTER_MODEL_WITH_COMMAND_NEUTRAL_HANDOFF',
        forward_result=info['forward_result'],forward_result_sha256=info['forward_result_sha256'],
        actual_start_world_xyyaw=info['initial_world_xyyaw'],retained_all_joint_positions_rad=info['retained_all_joint_positions_rad'],
        original_mission_goal_xyyaw=info['mission_goal_xyyaw'],strict_map_digest=info['strict_map_digest'],
        fixed_second_reverse=info['leg'],controller_contract=contract,
        exact_zero_command_AND_actual_neutral_before_ONE_captured_hold=True,
        reused_synthetic_cases_for_design_NOT_independent_validation=True,
        caster_stiction_and_actual_captured_hold_feedback_NOT_modelled=True,
        PhysX_executed=False,automatic_motion_authorized=False,CICS_input_used=False,use_for_final_statistics=False,
        input_sha256={**info['input_sha256'],**contract['input_sha256']},
        source_sha256={p.name:h.sha(p) for p in (Path(__file__),Path(h.__file__),Path(__file__).with_name('hospital_h3b_timed_second_reverse_model.py'))})
    h.ADAPTER_MODEL.parent.mkdir(parents=True,exist_ok=True)
    trace_dir=h.ADAPTER_MODEL.parent/'model_traces'
    trace_dir.mkdir(exist_ok=False)
    with (trace_dir/'contract.json').open('x') as stream:json.dump(header,stream,indent=2,allow_nan=False)
    cases=[]
    for scale in h.previous.SCALES:
        for residual in h.previous.RESIDUALS:
            for dt in h.previous.STEPS:
                result,trace=model_case(info,dt,residual,scale)
                diag=collision_diagnostic(trace,checker,occupied,origin,resolution,info['mission_start_xyyaw'])
                result['sampled_envelope_diagnostic']=diag
                result['checks']['strict_10mm_model_envelope_clear']=diag['sampled_model_envelope_clear']
                name=f"scale{round(scale*100)}_{residual}_{round(dt*1000)}ms"
                save_trace(trace_dir,name,result,trace);cases.append(result)
                print('EXACT ADAPTER MODEL',scale,residual,dt,'PASS' if all(result['checks'].values()) else 'HOLD',
                    'mm/deg:',round(1000*result['terminal_position_error_m'],3),round(result['terminal_heading_error_deg'],6),flush=True)
    passed=all(all(c['checks'].values()) for c in cases)
    report=dict(header,cases=cases,all_adapter_model_cases_pass=passed,
        status='EXACT_SECOND_REVERSE_ADAPTER_MODEL_PASS_PhysX_PENDING' if passed else 'HOLD_EXACT_SECOND_REVERSE_ADAPTER_MODEL')
    h.verify_files(header['input_sha256'])
    encoded=json.dumps(report,indent=2,allow_nan=False)
    with h.ADAPTER_MODEL.open('x') as stream:stream.write(encoded)
    if not passed:raise RuntimeError('Adapter model HOLD preserved; no runtime admission prepared')


def main():
    prepare_adapter_model()
    info = h.load_adapter_reviewed_profile()
    record = h.admission_payload(info)
    if h.ADMISSION.exists():
        if json.loads(h.ADMISSION.read_bytes()) != record:
            raise FileExistsError('Existing contract differs; preserve it, do not overwrite or bypass')
        print('Exact existing adapter contract verified; no additional trial authorized.')
    else:
        h.ADMISSION.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, indent=2, allow_nan=False)
        with h.ADMISSION.open('x') as stream:
            stream.write(encoded)
    h.verify_files(record['input_sha256'])
    print('=== H3-B SHARED TIMING SECOND-REVERSE ADAPTER PREPARED (NO MOTION) ===')
    print('raw next-forward PASS retained; all EIGHT offline design cases verified')
    print('reference mode / direction / length mm:', info['leg']['mode'], -1, round(1000*info['reference_length_m'], 3))
    print('limits unchanged: 50mm/s, 8deg/s, 30mm/2deg, strict envelope +10mm')
    print('saved:', h.ADMISSION)
    print('Keep PAUSED. Next: dedicated read-only live handoff review; no reset/Play/save or old controller.')


if __name__ == '__main__':
    main()
