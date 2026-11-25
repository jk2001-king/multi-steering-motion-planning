#!/usr/bin/env python3
"""Reproduce the saved mode-aware spatial candidate; no physical commands."""

import argparse
import json
from pathlib import Path

from hospital_h3_turn_planning import GATE, WORKSPACE, SUPPORT, load_setup
from hospital_h3b_mode_aware_seed import ModeSeedModel, audit_geometry, nominal_timing, FAMILIES
from hospital_h3b_mode_speed_evidence import inspect as inspect_components


STATUS='MODE_AWARE_GEOMETRY_NOMINAL_TIMING_REPRODUCED_HOLD_DYNAMICS_AND_REVERSE'


def inspect(directory):
    directory=Path(directory).resolve()
    summary_path=directory/'summary.json';contract_path=directory/'contract.json'
    summary=json.loads(summary_path.read_bytes());contract=json.loads(contract_path.read_bytes())
    if any(summary.get(k)!=v for k,v in contract.items()):raise ValueError('Summary/contract mismatch')
    if (summary.get('selected_family') not in FAMILIES or summary.get('new_PhysX_trial_performed') is not False
            or summary.get('dynamic_feasibility_certified') is not False
            or summary.get('P_full_dynamics_plan_selected') is not False or summary.get('use_for_final_statistics') is not False):
        raise ValueError('Need a geometry/nominal-timing candidate, not dynamics or execution evidence')
    SUPPORT['verify_files'](summary['input_sha256'])
    SUPPORT['verify_files']({WORKSPACE/'studies/IJCAS/scripts'/k:v for k,v in summary['source_sha256'].items()})
    measured=inspect_components()
    if measured!=summary['measured_component_review']:raise ValueError('Measured component review changed')
    manifest,occupied,origin,resolution,digest=load_setup(summary['geometry_manifest'],summary['preview_receipt'])
    if (digest!=summary['geometry_manifest_sha256'] or summary['start_xyyaw']!=manifest['start_xyyaw']
            or summary['goal_xyyaw']!=manifest['goal_xyyaw']
            or summary['planning_occupancy_digest']!=manifest['planning_occupancy_digest']):
        raise ValueError('Original H3 task/map/frame changed')
    row=next(r for r in summary['results'] if r['family']==summary['selected_family'])
    path=directory/(summary['selected_family']+'.json')
    if Path(row['result_file'])!=path:raise ValueError('Unexpected candidate file')
    SUPPORT['verify_files']({path:row['result_sha256']})
    result=json.loads(path.read_bytes())
    if result['contract_sha256']!=SUPPORT['sha'](contract_path):raise ValueError('Candidate contract changed')
    actions=json.loads(Path(summary['warm_start_seed']).read_bytes())['actions']
    model=ModeSeedModel(actions,summary['selected_family'])
    geometry=audit_geometry(model,result['parameters'],manifest,occupied,origin,resolution)
    timing=nominal_timing(geometry)
    if (geometry!=result['geometry_audit'] or geometry!=summary['geometry_audit']
            or geometry['verdict']!='PASS' or not all(geometry['checks'].values())
            or timing!=result['nominal_timing'] or timing!=summary['nominal_timing'] or not all(timing['checks'].values())):
        raise ValueError('Saved geometry/nominal timing does not reproduce')
    return {'status':STATUS,'role':'MODE_AWARE_SPATIAL_SEED_ONLY_NOT_FOUR_METHOD_OR_FULL_DYNAMICS_SELECTION',
        'selected_family':summary['selected_family'],'geometry_manifest':summary['geometry_manifest'],
        'preview_receipt':summary['preview_receipt'],'start_xyyaw':manifest['start_xyyaw'],'goal_xyyaw':manifest['goal_xyyaw'],
        'planning_occupancy_digest':manifest['planning_occupancy_digest'],
        'geometry_audit':geometry,'nominal_timing':timing,'measured_component_review':measured,
        'input_sha256':{**summary['input_sha256'],**{str(p):SUPPORT['sha'](p) for p in (summary_path,contract_path,path)}},
        'source_sha256':{**summary['source_sha256'],Path(__file__).name:SUPPORT['sha'](__file__)},
        'unverified_execution_scope':summary['unverified_execution_scope'],
        'environment_goal_clearance_or_motor_parameters_changed':False,'prior_FAILs_and_HOLDs_preserved':True,
        'dynamic_feasibility_certified':False,'P_full_dynamics_plan_selected':False,
        'PhysX_executed':False,'use_for_final_statistics':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=GATE/'hospital_h3b_mode_aware_seed/seed_r01')
    args=parser.parse_args();report=inspect(args.directory)
    folder=args.directory.resolve()/'audits';folder.mkdir(exist_ok=True)
    number=1
    while (path:=folder/f'mode_aware_audit_r{number:02d}.json').exists():number+=1
    with path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    g=report['geometry_audit'];t=report['nominal_timing']
    print('=== H3-B MODE-AWARE SPATIAL + NOMINAL TIMING AUDIT ===')
    print('four measured component outcomes: reproduced; spatial family:',report['selected_family'])
    print('geometry:',g['verdict'],'independent SAT samples:',g['sample_count'],'leg modes:',g['leg_modes'])
    print('centerline length/reverse m:',g['total_path_length_m'],g['total_reverse_m'])
    print('nominal motion+cusp dwell s:',t['reference_motion_plus_nominal_cusp_dwell_s'])
    print('STATUS:',report['status']);print('saved:',path)
    print('Preview only. No reverse-turning, whole-route dynamics, PhysX, CICS or Nav2 PASS. Do not manually Play.')


if __name__=='__main__':main()
