"""Measured next-forward endpoint -> NEW remaining two-leg spatial candidate.

No robot move/reset, controller/gain fit or old-reference outcome relabelling.
The intermediate cusp is a NEW planning variable; the original final mission
goal, map, modes and physical limits stay fixed. Reused synthetic screens are
design/selection checks, NOT independent robustness validation.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import copy
import json
import math
from pathlib import Path

import numpy as np

import hospital_h3b_post_forward as previous
from hospital_h3b_measured_suffix import MeasuredSuffixModel
from hospital_h3b_measured_suffix_refinement import DenseSuffixObjective
from hospital_h3b_mode_aware_seed import audit_geometry, nominal_timing
from hospital_h3_turn_planning import verify_sources
from hospital_geometry_tools import pose_collision
from hospital_planning_tools import local_to_world


WORKSPACE,GATE=previous.WORKSPACE,previous.GATE
BASE=GATE/'hospital_h3b_final_suffix_replanning'
OLD_REVIEW=previous.BASE/'review_r01'
LIVE=OLD_REVIEW/'live_post_forward.json'
VERSION='H3B_actual_forward_PASS_remaining_two_leg_reserve_replan_v1'
FAMILY='counterphase_forward_front_only_reverse'
NOMINAL_CLEARANCE=.030
NOMINAL_CURVATURE=.60
NOMINAL_REVERSE_ARC_CAP=.450
FORWARD_LENGTH_BOX=(.50,1.10)
sha,verify_files=previous.sha,previous.verify_files


def load_start():
    info,audit,manifest,geometry,inventory=previous.inspect_forward()
    old=previous.load_review(OLD_REVIEW);live=json.loads(LIVE.read_bytes())
    verify_files(live['input_sha256'])
    if (live.get('status')!='ACTUAL_NEXT_FORWARD_PASS_PAUSED_CONTINUITY_RECORDED_NOT_DRIVE_AUTHORITY'
            or live.get('forward_result_sha256')!=info['forward_result_sha256']
            or live.get('offline_review')!=str(OLD_REVIEW/'post_forward_review.json')
            or live.get('offline_review_sha256')!=sha(OLD_REVIEW/'post_forward_review.json')
            or live.get('physical_state_reset') is not False or live.get('new_PhysX_motion_executed') is not False
            or live.get('automatic_motion_authorized') is not False
            or live.get('root_session_and_timeline_unchanged') is not True
            or live.get('all_second_reverse_model_cases_pass')!=old['all_second_reverse_model_cases_pass']):
        raise ValueError('Exact read-only actual next-forward PASS continuity receipt required')
    entry=_ijcas_path(WORKSPACE, 'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_forward.py')
    verify_files({entry:live['script_sha256']})
    previous.forward.validate_live(info,live['live_world_xyyaw'],live['retained_all_joint_positions_rad'],live['live_root_velocity'])
    info.update(live_receipt=str(LIVE),live_receipt_sha256=sha(LIVE),old_model_review=old,
        input_sha256={**info['input_sha256'],**old['input_sha256'],**live['input_sha256'],
            str(LIVE):sha(LIVE),str(OLD_REVIEW/'post_forward_review.json'):sha(OLD_REVIEW/'post_forward_review.json'),
            **{c['trace_file']:c['trace_sha256'] for c in old['cases']}})
    planning=dict(manifest,start_xyyaw=info['mission_start_xyyaw'],goal_xyyaw=info['mission_goal_xyyaw'],
        local_goal_xyyaw=list(previous.forward.relative_pose(info['mission_goal_xyyaw'],info['mission_start_xyyaw'])))
    return info,audit,planning,geometry,inventory


def build_model(candidate,actual_initial):
    g=candidate['geometry_audit']
    previous.extract_second_reverse(candidate)  # verify the unchanged source family
    counts=g['leg_control_counts'];cut=sum(counts[:2])
    knots=np.asarray(g['curvature_knots_1_m'][cut:],float)
    directions=[d for n,d in zip(counts[2:],g['leg_directions'][2:]) for _ in range(n)]
    actions=[dict(mode='coordinated_counterphase',direction=d,curvature_1_m=float((knots[i]+knots[i+1])/2))
        for i,d in enumerate(directions)]
    model=MeasuredSuffixModel(actions,FAMILY,actual_initial)
    model.initial_parameters=np.r_[knots[model.free],g['leg_longitudinal_distances_m'][2:]]
    if list(model.leg_directions)!=[-1,1] or model.leg_modes!=['front_only','coordinated_counterphase']:
        raise ValueError('Only the two remaining existing-mode legs may be replanned')
    return model


def search_bounds(model):
    if list(model.leg_directions)!=[-1,1]:raise ValueError('Expected exactly reverse then forward')
    # Bound actual center arc via sqrt(1+(arm*k)^2), not the shorter body-u
    # longitudinal parameter. This is a stricter design bound, not a new rule.
    maximum_reverse=NOMINAL_REVERSE_ARC_CAP/math.sqrt(1+(.645*NOMINAL_CURVATURE)**2)
    return (np.r_[np.full(len(model.free),-NOMINAL_CURVATURE),.05,FORWARD_LENGTH_BOX[0]],
            np.r_[np.full(len(model.free),NOMINAL_CURVATURE),maximum_reverse,FORWARD_LENGTH_BOX[1]])


def geometry_review(model,z,info,manifest,occupied,origin,resolution):
    g=audit_geometry(model,z,manifest,occupied,origin,resolution)
    checker=previous.NativeHospitalEnvelopeChecker(occupied,origin,resolution,info['mission_start_xyyaw'],clearance=NOMINAL_CLEARANCE)
    blocked=checker(g['poses_local_xyyaw']);independent=False
    if not blocked:
        independent=True
        if any(pose_collision(occupied,local_to_world(p,info['mission_start_xyyaw']),origin,resolution,NOMINAL_CLEARANCE)
                for p in g['poses_local_xyyaw']):
            raise ValueError('Independent original SAT disagrees with stricter nominal envelope')
    initial=previous.forward.relative_pose(info['initial_world_xyyaw'],info['mission_start_xyyaw'])
    g['checks'].update(actual_planning_start_retained_NOT_pose_reset=bool(np.allclose(g['poses_local_xyyaw'][0],initial,rtol=0,atol=1e-12)),
        original_final_mission_goal_unchanged=manifest['goal_xyyaw']==info['mission_goal_xyyaw'],
        nominal_30mm_full_envelope_clear=not blocked and independent,
        nominal_curvature_under_point60=max(abs(k) for k in g['curvature_knots_1_m'])<=NOMINAL_CURVATURE+1e-9,
        nominal_reverse_center_arc_under_450mm=g['leg_lengths_m'][0]<=NOMINAL_REVERSE_ARC_CAP+1e-9,
        tighter_nominal_final_position_under_2mm=g['terminal_position_error_m']<=.002,
        tighter_nominal_final_heading_under_point2deg=g['terminal_heading_error_deg']<=.2,
        original_cumulative_total_reverse_under_1m=info['previous_measured_reverse_total_m']+g['total_reverse_m']<=1.,
        original_cumulative_direction_changes_under_four=info['completed_direction_changes']+2<=4)
    g['checks']={k:bool(v) for k,v in g['checks'].items()}
    g['verdict']='PASS' if all(g['checks'].values()) else 'FAIL'
    return g


def reverse_profile(geometry):
    direction=np.array(geometry['interval_directions']);cusps=np.flatnonzero(np.diff(direction))+1
    if len(cusps)!=1 or geometry['leg_directions']!=[-1,1] or geometry['leg_modes']!=['front_only','coordinated_counterphase']:
        raise ValueError('NEW two-leg geometry required, not the old four-leg reference')
    end=int(cusps[0]);arc=np.array(geometry['travelled_arc_m'][:end+1],float);arc-=arc[0]
    profile=dict(original_suffix_leg_index=0,original_sample_indices_inclusive=[0,end],
        mode='front_only',direction=-1,poses_local_xyyaw=geometry['poses_local_xyyaw'][:end+1],
        center_arc_m=arc.tolist(),curvature_1_m=geometry['sample_curvature_1_m'][:end+1],
        next_cusp_local_xyyaw=geometry['poses_local_xyyaw'][end],reference_length_m=float(arc[-1]))
    previous.FirstReverseReference(profile)
    return profile


def design_model_cases(info,geometry):
    """Same eight previously used cases; design checks, not new validation."""
    adapted=dict(info,second_reverse=reverse_profile(geometry))
    for scale in previous.SCALES:
        for residual in previous.RESIDUALS:
            for dt in previous.STEPS:
                result,trace=previous.model_case(adapted,dt,residual,scale)
                result.update(NEW_replanned_intermediate_cusp=True,whole_final_forward_NOT_modelled=True,
                    reused_sensitivity_cases_are_design_selection_NOT_independent_validation=True)
                yield result,trace


def load_candidate(directory):
    report=json.loads((Path(directory)/'final_suffix_candidate.json').read_bytes())
    verify_sources(report);verify_files(report['input_sha256'])
    if (report['version']!=VERSION or report['PhysX_executed'] is not False
            or report['automatic_motion_authorized'] is not False or report['forward_result_sha256']!=sha(previous.RESULT)):
        raise ValueError('Exact new remaining-suffix candidate required')
    return report
