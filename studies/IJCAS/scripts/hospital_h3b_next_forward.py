"""One next FORWARD component from the retained measured-reverse PASS.

This is an explicit physical adapter of the already screened next-leg model
prototype. No new planner, target, gain, caster assignment, reverse command or
terminal correction. Preserve all preceding actual outcomes and state history.
"""

import ast
import json
import math
from pathlib import Path

from hospital_private_loader import load
from hospital_path_execution import direct_joint_targets, relative_pose, wrap
from hospital_h3b_policy import StopAlignGate, POLICY
from hospital_h3b_reverse_closedloop import arrival_checks
from hospital_h3_flush_crossing import ROBOT_ROOT, WHEELS, BAD_CONTACTS, classify_contact, evaluate_passage
from hospital_h3_turn_planning import verify_sources, model_bounds


HERE = Path(__file__).resolve().parent
# Kit's embedded importer does not reliably resolve bare review_* siblings.
# Load this frozen dependency privately, without altering sys.path/modules.
REVIEW = load(HERE/'review_hospital_h3b_post_reverse.py')
reverse = REVIEW['reverse']
WORKSPACE = reverse.WORKSPACE
GATE = reverse.GATE
VERSION = 'H3B_retained_reverse_PASS_next_counterphase_forward_v1'
ROLE = 'ONE_NEXT_FORWARD_COMPONENT_FROM_ACTUAL_REVERSE_PASS_NOT_FULL_SUFFIX'
REVIEW_DIR = GATE/'hospital_h3b_post_reverse_reviews/review_r03'
LIVE = REVIEW_DIR/'live_post_reverse.json'
COHORT = GATE/'hospital_h3b_next_forward_pilot/cohort_01'
ADMISSION = COHORT/'admission/executor_admission.json'
OUTPUT = COHORT/'forward'
read_trace, sha, verify_files = reverse.read_trace, reverse.sha, reverse.verify_files
validate_live = reverse.validate_live
body_velocity = REVIEW['body_velocity']
NextForwardModelController = REVIEW['NextForwardModelController']
TRACKING_GAINS = REVIEW['TRACKING_GAINS']


def execution_contract(length):
    return dict(version=VERSION, direction=1, mode='coordinated_counterphase',
        nominal_reference_length_m=length, forward_speed_cap_m_s=.050,
        acceleration_m_s2=.020, curvature_limit_1_m=.7, steering_command_slew_deg_s=8.,
        tracking_limits_m_deg=[.080,10.], terminal_limits_m_deg=[.030,2.],
        stopped_speed_m_s=.005, actual_neutral_deg=2.5,
        fixed_hold_s=5., fixed_tail_s=3., maximum_control_gap_s=.25,
        maximum_simulation_s=90., maximum_wall_s=300., braking_timeout_s=8.,
        measured_stop_hold_neutral_release_direction=1,
        # Supplemental forward-component travelled-distance guard. The 0.5m
        # policy is a REVERSE resource bound, not a forward distance limit.
        maximum_measured_center_distance_m=length+2*.080,
        no_terminal_reapproach_or_pulse=True, no_reverse_or_automatic_next_leg=True,
        next_leg_reference_adapter_is_NEW_not_unchanged_four_method_comparison=True,
        inherited_course_step_and_1p5_2_gains_unchanged=True,
        physical_model_gain_friction_caster_map_or_fixed_goal_change=False,
        model_screen_is_NOT_PhysX_contact_or_caster_response_certificate=True)


def load_profile():
    info, audit, _, _, _ = REVIEW['inspect_reverse']()
    model = REVIEW['load_review'](REVIEW_DIR)
    if not model['all_next_forward_model_cases_pass'] or any(not all(c['checks'].values()) for c in model['cases']):
        raise ValueError('All FOUR frozen next-forward model cases must pass before this component')
    live = json.loads(LIVE.read_bytes()); verify_files(live['input_sha256'])
    expected = dict(status='ACTUAL_REVERSE_PASS_PAUSED_CONTINUITY_RECORDED_NOT_DRIVE_AUTHORITY',
        reverse_result=info['reverse_result'], reverse_result_sha256=info['reverse_result_sha256'],
        offline_review=str(REVIEW_DIR/'post_reverse_review.json'),
        offline_review_sha256=sha(REVIEW_DIR/'post_reverse_review.json'),
        physical_state_reset=False, new_PhysX_motion_executed=False,
        automatic_motion_authorized=False, all_next_forward_model_cases_pass=True,
        root_session_and_timeline_unchanged=True)
    if any(live.get(k) != v for k,v in expected.items()):
        raise ValueError('Need the exact actual reverse-PASS read-only receipt, not the prefix/failed scene')
    script = WORKSPACE/'isaac/scripts/isaac_review_ijcas_hospital_h3b_post_reverse.py'
    verify_files({script:live['script_sha256']})
    validate_live(info,live['live_world_xyyaw'],live['retained_all_joint_positions_rad'],live['live_root_velocity'])
    leg = model['next_leg']
    if (model['actual_start_world_xyyaw'] != info['initial_world_xyyaw']
            or model['retained_all_joint_positions_rad'] != info['retained_all_joint_positions_rad']
            or any(c['model_parameters'] != model_bounds() or c['tracking_gains'] != TRACKING_GAINS
                   or c['fixed_goal_local_xyyaw'] != leg['fixed_goal_local_xyyaw'] for c in model['cases'])):
        raise ValueError('Frozen model/start/12 joints/gains/fixed next cusp changed')
    info.update(leg=leg, next_cusp_local_xyyaw=leg['fixed_goal_local_xyyaw'],
        preparation_record=info['first_record'], raw_reverse_audit=audit,
        reference_length_m=leg['length_m'],
        live_receipt=str(LIVE), live_receipt_sha256=sha(LIVE),
        model_review=str(REVIEW_DIR/'post_reverse_review.json'), model_review_sha256=sha(REVIEW_DIR/'post_reverse_review.json'),
        intervention_contract=execution_contract(leg['length_m']),
        input_sha256={**info['input_sha256'], **model['input_sha256'], **live['input_sha256'],
            **{c['trace_file']:c['trace_sha256'] for c in model['cases']},
            str(REVIEW_DIR/'post_reverse_review.json'):sha(REVIEW_DIR/'post_reverse_review.json'),
            str(LIVE):sha(LIVE), str(Path(__file__)):sha(__file__)})
    return info


def ForwardReference(info):
    return REVIEW['NextForwardReference'](info['leg'])


def ForwardProgress(info, started):
    # EXACT same class/step used by all four frozen model screens. Only the
    # physical state observations come from the actual adapter at runtime.
    return NextForwardModelController(info['leg'], started)


def controller_and_adapter_contract():
    paths = [WORKSPACE/'isaac/scripts'/name for name in (
        'isaac_run_ijcas_hospital_h3b_reverse_closedloop_pilot.py',
        'isaac_run_ijcas_hospital_h3b_next_forward_pilot.py')]
    funcs = [{n.name:n for n in ast.walk(ast.parse(p.read_text()))
              if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for p in paths]
    unchanged = ('numeric','physics_step','contacts','angles','verify_gains','measure','set_hold')
    for name in unchanged:
        if ast.dump(funcs[0][name],include_attributes=False) != ast.dump(funcs[1][name],include_attributes=False):
            raise ValueError('Shared low-level measurement/hold/plant adapter changed: '+name)
    if 'step' in REVIEW['NextForwardModelController'].__dict__:
        raise ValueError('The next-forward course step law must stay inherited unchanged')
    sources = [*paths, Path(__file__), HERE/'review_hospital_h3b_post_reverse.py',
        HERE/'hospital_h3b_course_tracking.py', HERE/'hospital_h3b_first_leg_feedback_review.py',
        HERE/'hospital_path_execution.py', HERE/'hospital_h3b_next_forward_observation.py',
        HERE/'review_hospital_h3b_next_forward_executor.py',
        HERE/'hospital_h3b_policy.py', HERE/'hospital_drive_hold.py',
        WORKSPACE/'isaac/scripts/isaac_run_ijcas_hospital_h3b_next_forward_instrumented.py']
    return dict(unchanged_AST_functions=list(unchanged),
        frozen_next_forward_model_controller_used_without_step_override=True,
        command_direction_mode_and_reference_adapter_change_EXPLICIT=True,
        reverse_0p5m_guard_NOT_relaxed_applies_only_to_reverse=True,
        supplemental_forward_distance_guard= 'reference_length + 2*0.080 m',
        input_sha256={str(p):sha(p) for p in sources})


def admission_payload(info):
    contract = controller_and_adapter_contract()
    return dict(version=VERSION, status='ONE_NEXT_FORWARD_EXECUTOR_ADMITTED_PhysX_PENDING',
        reverse_result=info['reverse_result'], reverse_result_sha256=info['reverse_result_sha256'],
        live_receipt=info['live_receipt'], live_receipt_sha256=info['live_receipt_sha256'],
        model_review=info['model_review'], model_review_sha256=info['model_review_sha256'],
        actual_start_world_xyyaw=info['initial_world_xyyaw'],
        retained_all_joint_positions_rad=info['retained_all_joint_positions_rad'],
        fixed_goal_local_xyyaw=info['next_cusp_local_xyyaw'], original_mission_goal_xyyaw=info['mission_goal_xyyaw'],
        strict_map_digest=info['strict_map_digest'], execution_contract=info['intervention_contract'],
        controller_contract=contract, input_sha256={**info['input_sha256'], **contract['input_sha256']},
        PhysX_executed=False, automatic_motion_launched=False, full_H3B_turning_path_verified=False,
        first_forward_and_first_reverse_original_PASS_preserved=True, previous_FAILs_preserved=True,
        use_for_final_statistics=False)


def load_admitted_profile():
    info = load_profile(); report = json.loads(ADMISSION.read_bytes()); verify_files(report['input_sha256'])
    if report != admission_payload(info):
        raise ValueError('Dedicated exact next-forward executor admission required; do not launch old controllers')
    info.update(admission=str(ADMISSION), admission_sha256=sha(ADMISSION),
        input_sha256={**report['input_sha256'],str(ADMISSION):sha(ADMISSION)})
    return info


def motion_stalled(rows):
    if not rows: return False
    end = rows[-1]['time_s']; tail = [r for r in rows if r['time_s'] >= end-5.]
    return (len(tail)>1 and tail[-1]['time_s']-tail[0]['time_s'] >= 4.95
        and all(r['phase'] in ('forward_motion','forward_braking') and not r['position_hold_enabled']
                and r['command_speed_m_s'] >= .006 for r in tail)
        and max(r['reference_progress_m'] for r in tail)-min(r['reference_progress_m'] for r in tail)<.001)


def evaluate_component(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events,reference_length):
    numeric=('command_time_s','command_speed_m_s','command_front_rad','command_rear_rad',
        'front_rad','rear_rad','reference_progress_m','reference_position_error_m','reference_heading_error_deg')
    if any(not math.isfinite(float(r[k])) for r in rows for k in numeric):
        raise ValueError('Finite actual command/reference observations required')
    common = evaluate_passage(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count)
    excluded = {'all_six_wheels_started_east_and_finished_west','ordinary_positive_ramp_within_30mm_s','passage_and_hold_completed'}
    checks = {k:v for k,v in common['checks'].items() if k not in excluded}
    hold = next((r for r in rows if r['phase']=='terminal_hold'),None)
    measured = lambda r:max(r['pose_filtered_speed_m_s'], math.hypot(r['vx_physx_m_s'],r['vy_physx_m_s']))
    gate = gate_events[0] if len(gate_events)==1 else {}
    max_slew = max_acceleration = 0.
    for a,b in zip(rows,rows[1:]):
        dt=b['command_time_s']-a['command_time_s']
        ds=abs(b['command_speed_m_s']-a['command_speed_m_s'])
        da=max(abs(b['command_'+axle+'_rad']-a['command_'+axle+'_rad']) for axle in ('front','rear'))
        if dt<0 or (dt==0 and max(ds,da)>1e-12): raise ValueError('Invalid command clock/change')
        max_acceleration=max(max_acceleration,ds/dt if dt else 0.)
        max_slew=max(max_slew,da/dt if dt else 0.)
    distance=sum(math.hypot(b['x_m']-a['x_m'],b['y_m']-a['y_m']) for a,b in zip(rows,rows[1:]))
    order={'pre_hold':0,'entry_gate':1,'forward_motion':2,'forward_braking':3,'terminal_hold':4}
    phases=[order.get(r['phase'],-1) for r in rows]
    checks.update(next_forward_and_single_hold_completed=True,
        one_contiguous_forward_episode_and_terminal_hold=all(p>=0 for p in phases)
            and all(b>=a for a,b in zip(phases,phases[1:])),
        only_counterphase_forward_or_zero_commands=all(r['command_mode']=='coordinated_counterphase'
            and 0<=r['command_speed_m_s']<=.050+1e-9 for r in rows),
        one_measured_forward_stop_hold_neutral_release=len(gate_events)==1 and gate.get('direction')==1
            and gate.get('captured_hold_acknowledged') is True and gate.get('hold_release_acknowledged') is True
            and gate.get('measured_speed_m_s',1.)<=.005 and gate.get('steering_error_deg',180.)<=2.5,
        arrived_before_ONE_hold=hold is not None and all(arrival_checks(
            [hold[k] for k in ('x_m','y_m','yaw_rad')],goal,measured(hold),(hold['front_rad'],hold['rear_rad'])).values()),
        reference_progress_reached_fixed_end_neighbourhood=rows[-1]['reference_progress_m']>=reference_length-.050,
        no_terminal_motion_restart=not any(a['phase']=='forward_braking' and b['phase']=='forward_motion' for a,b in zip(rows,rows[1:])),
        command_acceleration_under_20mm_s2=max_acceleration<=.020+1e-7,
        command_slew_under_8deg_s=max_slew<=math.radians(8.)+1e-5,
        fixed_reference_tracking_under_80mm_10deg=all(0<=r['reference_position_error_m']<=.080
            and abs(r['reference_heading_error_deg'])<=10. for r in rows),
        no_hold_steering_or_speed_command=all(abs(r['command_speed_m_s'])<1e-7
            and max(abs(r['command_'+a+'_rad']) for a in ('front','rear'))<1e-7 for r in rows if r['position_hold_enabled']),
        held_actual_steering_neutral=all(max(abs(r[a+'_rad']) for a in ('front','rear'))<=math.radians(2.5)
            for r in rows if r['position_hold_enabled']),
        supplemental_forward_distance_guard=distance<=reference_length+2*.080)
    return dict(checks={k:bool(v) for k,v in checks.items()},verdict='PASS' if all(checks.values()) else 'FAIL',
        **{k:common[k] for k in ('terminal_position_error_m','terminal_heading_error_deg','final_world_xyyaw','fixed_terminal_hold_analysis','final_force_weight_ratio')},
        maximum_observed_command_slew_deg_s=math.degrees(max_slew),maximum_observed_command_acceleration_m_s2=max_acceleration,
        next_forward_component_verified=all(checks.values()),
        full_H3B_turning_path_verified=False,full_suffix_execution_verified=False,
        next_front_only_reverse_verified=False,Nav2_executed=False,CICS_input_used=False,use_for_final_statistics=False)
