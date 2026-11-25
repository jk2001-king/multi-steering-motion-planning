#!/usr/bin/env python3
"""One bounded spatial reserve-candidate review; NO motion/drive permission.

Same H3 map/task/resources/existing modes. Search a stricter NOMINAL 30 mm
rectangular clearance and |k|<=0.60, leaving the original physical 10 mm and
|k|<=0.70 guards unchanged. These reserves are design choices, not an identified
uncertainty bound or proof that feedback can use them. No gain/model fitting.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from hospital_h3b_mode_aware_seed import ModeSeedModel,audit_geometry,nominal_timing
from hospital_h3b_warm_start import Objective
from hospital_h3_turn_planning import GATE,WORKSPACE,SUPPORT,load_setup,verify_sources
from hospital_geometry_tools import pose_collision
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from hospital_planning_tools import local_to_world


NOMINAL_CLEARANCE=.030
NOMINAL_CURVATURE_CAP=.60
VERSION='H3B_spatial_nominal_clearance_and_curvature_reserve_review_v1'


def reserve_bounds(model):
    lower,upper=model.bounds();n=len(model.free)
    lower[:n]=np.maximum(lower[:n],-NOMINAL_CURVATURE_CAP)
    upper[:n]=np.minimum(upper[:n],NOMINAL_CURVATURE_CAP)
    return lower,upper


def reserve_objective(model,manifest,occupied,origin,resolution,deadline,watch):
    objective=Objective(model,manifest,occupied,origin,resolution,deadline,watch)
    objective.points=np.array([(x,y) for x in np.linspace(-.815-NOMINAL_CLEARANCE,.815+NOMINAL_CLEARANCE,7)
        for y in np.linspace(-.4825-NOMINAL_CLEARANCE,.4825+NOMINAL_CLEARANCE,5)])
    return objective


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-s',type=float,default=45.)
    args=parser.parse_args()
    if not math.isfinite(args.budget_s) or not 0<args.budget_s<=60.:parser.error('Bounded candidate search requires 0 < budget <=60 s')
    seed_directory=GATE/'hospital_h3b_mode_aware_seed/seed_r01'
    summary_path=seed_directory/'summary.json';summary=json.loads(summary_path.read_bytes());verify_sources(summary)
    SUPPORT['verify_files'](summary['input_sha256'])
    candidate_path=seed_directory/(summary['selected_family']+'.json');candidate=json.loads(candidate_path.read_bytes())
    seed_path=Path(summary['warm_start_seed']);seed=json.loads(seed_path.read_bytes())
    model=ModeSeedModel(seed['actions'],summary['selected_family'],candidate['parameters'])
    manifest,occupied,origin,resolution,_=load_setup(summary['geometry_manifest'],summary['preview_receipt'])
    base=GATE/'hospital_h3b_clearance_reserve_seed';base.mkdir(parents=True,exist_ok=True);number=1
    while (output:=base/f'seed_r{number:02d}').exists():number+=1
    output.mkdir()
    header={'version':VERSION,'role':'STRONGER_NOMINAL_SPATIAL_RESERVE_CANDIDATE_NOT_A_DYNAMICS_OR_PhysX_PASS',
        'family':model.family,'unchanged_start_xyyaw':manifest['start_xyyaw'],'unchanged_goal_xyyaw':manifest['goal_xyyaw'],
        'unchanged_strict_map_digest':manifest['planning_occupancy_digest'],
        'nominal_search_clearance_m':NOMINAL_CLEARANCE,'original_PhysX_clearance_guard_m':.010,
        'nominal_curvature_cap_1_m':NOMINAL_CURVATURE_CAP,'original_physical_curvature_cap_1_m':.70,
        'maximum_reverse_episode_m':.5,'maximum_total_reverse_m':1.,'maximum_cusps':4,
        'budget_s':args.budget_s,'max_function_evaluations':160,
        'new_candidate_generation_design_not_unchanged_four_method_comparison':True,
        'reserves_are_NOT_identified_uncertainty_bounds_or_closed_loop_robustness_proof':True,
        'gains_model_map_assets_goal_casters_and_original_failures_changed':False,
        'PhysX_executed':False,'directional_dynamics_verified':False,'feedback_execution_verified':False,
        'P_full_dynamics_plan_selected':False,'use_for_final_statistics':False,
        'input_sha256':{**summary['input_sha256'],**{str(p):SUPPORT['sha'](p) for p in (summary_path,candidate_path,seed_path)}},
        'source_sha256':{p.name:SUPPORT['sha'](p) for p in (Path(__file__),
            Path(__file__).with_name('hospital_h3b_mode_aware_seed.py'),Path(__file__).with_name('hospital_h3b_warm_start.py'))}}
    with (output/'contract.json').open('x') as stream:json.dump(header,stream,indent=2,allow_nan=False);stream.write('\n')
    print('=== H3-B NOMINAL CLEARANCE / CURVATURE RESERVE CANDIDATE (NO MOTION) ===',flush=True)
    print('output:',output,flush=True)
    print('Same task/map/resources. Nominal search clearance 30 mm / curvature cap .60; original physical guards unchanged.',flush=True)
    started=time.perf_counter()
    def watch(row):
        print(f"[SPATIAL ONLY] elapsed={row['elapsed_s']:.1f}s calls={row['calls']} endpoint={1000*row['position_m']:.1f}mm/{row['heading_deg']:.2f}deg",flush=True)
    objective=reserve_objective(model,manifest,occupied,origin,resolution,started+args.budget_s,watch)
    lower,upper=reserve_bounds(model);initial=np.clip(model.initial(),lower,upper)
    status='OPTIMIZER_COMPLETED_NOT_A_CERTIFICATE';optimizer=None
    try:
        with threadpool_limits(limits=1):
            optimizer=least_squares(lambda z:objective.evaluate(z)[0],initial,jac=lambda z:objective.evaluate(z)[1],
                bounds=(lower,upper),method='trf',tr_solver='lsmr',x_scale='jac',max_nfev=160,
                ftol=1e-7,xtol=1e-7,gtol=1e-7)
    except TimeoutError:status='BOUNDED_SEARCH_TIME_EXHAUSTED_NOT_GLOBAL_INFEASIBILITY'
    z=objective.best[1] if objective.best is not None else initial
    geometry=audit_geometry(model,z,manifest,occupied,origin,resolution)
    poses=np.array(geometry['poses_local_xyyaw'])
    checker=NativeHospitalEnvelopeChecker(occupied,origin,resolution,manifest['start_xyyaw'],clearance=NOMINAL_CLEARANCE)
    blocked=checker(poses);independent=False
    if not blocked:
        independent=True
        if any(pose_collision(occupied,local_to_world(p,manifest['start_xyyaw']),origin,resolution,NOMINAL_CLEARANCE) for p in poses):
            raise ValueError('Independent original polygon SAT disagrees with native reserve check')
    geometry['checks'].update(stricter_nominal_30mm_full_envelope_clear=not blocked and independent,
        nominal_curvature_headroom_reserved=max(abs(k) for k in geometry['curvature_knots_1_m'])<=NOMINAL_CURVATURE_CAP+1e-9)
    geometry['verdict']='PASS' if all(geometry['checks'].values()) else 'FAIL'
    timing=nominal_timing(geometry) if geometry['verdict']=='PASS' else None
    admitted=geometry['verdict']=='PASS' and timing is not None and all(timing['checks'].values())
    report={**header,'optimizer_status':status,'optimizer_message':None if optimizer is None else str(optimizer.message),
        'elapsed_s':time.perf_counter()-started,'objective_calls':objective.calls,'parameters':z.tolist(),
        'geometry_audit':geometry,'nominal_timing':timing,
        'status':'RESERVE_GEOMETRY_AND_NOMINAL_TIMING_ADMITTED_HOLD_DYNAMIC_AND_EXECUTION_REVIEW' if admitted
            else 'HOLD_NO_RESERVE_CANDIDATE_IN_BOUNDED_LOCAL_SEARCH_NOT_GLOBAL_H3_INFEASIBILITY',
        'automatic_PhysX_execution_authorized':False}
    SUPPORT['verify_files'](header['input_sha256'])
    target=output/'reserve_candidate.json'
    with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print('geometry:',geometry['verdict'],'terminal mm/deg:',1000*geometry['terminal_position_error_m'],geometry['terminal_heading_error_deg'],flush=True)
    print('failed geometry checks:',[k for k,v in geometry['checks'].items() if not v],flush=True)
    print('STATUS:',report['status'],flush=True);print('saved:',target,flush=True)
    print('Geometry only. Do not Play/retry or execute this candidate with old controllers. Full H3/reverse/feedback/PhysX unverified.',flush=True)


if __name__=='__main__':main()
