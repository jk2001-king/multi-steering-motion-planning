"""Read-only controller replay and prospective cusp-deadband impact audit.

No simulator, target, scene, gain or actual outcome is changed. A different
handoff predicate is OBSERVED but never applied during these archived replays.
Exact reproduction is mandatory before a trace can be called unaffected.
"""
import csv
import json
import math
from pathlib import Path
from types import FunctionType

import hospital_mission_sequence as sequence
import hospital_revision_followup as followup

runtime=followup.runtime
OUTPUT=followup.OUTPUT/'cusp_deadband_review'
MIN_TARGET_M_S=.003
BRAKING_OFFSET_M=.010
ACCELERATION_M_S2=.020
MAX_CONTROL_GAP_S=.25
# Derived from the unchanged endpoint speed limiter and maximum allowed
# sample gap, not from an increased final goal/stop tolerance.
STOPPED_HANDOFF_RESERVE_M=(BRAKING_OFFSET_M+MIN_TARGET_M_S**2/(2*ACCELERATION_M_S2)
                         +MIN_TARGET_M_S*MAX_CONTROL_GAP_S)


def stopped_handoff_candidate(original,controller,pose,measured_speed,remaining):
    if original(controller,pose,measured_speed,remaining):
        return True
    if controller.mission_final_leg:
        return False
    return (measured_speed<=.005 and remaining<=STOPPED_HANDOFF_RESERVE_M
        and math.hypot(pose[0]-controller.reference.poses[-1][0],
                       pose[1]-controller.reference.poses[-1][1])<=.05)


def replay(report_path):
    report_path=Path(report_path)
    raw=runtime.read(report_path);packet=runtime.read(raw['protocol'])
    if runtime.sha(raw['trace'])!=raw['trace_sha256'] or runtime.sha(raw['protocol'])!=raw['protocol_sha256']:
        raise ValueError('Archived trace or protocol digest changed')
    for name in ('hospital_mission_sequence.py','hospital_mission_execution.py','hospital_mission_lifecycle.py',
                 'hospital_path_execution.py','hospital_path_execution_v2.py','hospital_path_execution_v3.py'):
        source=Path(__file__).with_name(name).resolve()
        if packet['input_sha256'].get(str(source))!=runtime.sha(source):
            raise ValueError('Actual control source no longer matches archived trial: '+name)
    plan=runtime.read(report_path.parent/packet['plan_filename'])
    geometry=runtime.read(packet['geometry_file'])
    controller=sequence.unified_geometry_lifecycle_controller(plan,geometry,profile=plan['geometry_profile'],
        simulation_budget_s=packet['simulation_budget_s'],wall_budget_s=packet['maximum_wall_time_s'])
    original=controller.motion.tracking_step
    handoff=original.__globals__['_mission_handoff'];predicate_differences=[];stamp=[0.]
    def observe(tracker,pose,speed,remaining):
        old=handoff(tracker,pose,speed,remaining)
        new=stopped_handoff_candidate(handoff,tracker,pose,speed,remaining)
        if new!=old:
            predicate_differences.append(dict(time_s=stamp[0],remaining_m=remaining,
                measured_speed_m_s=speed,command_before_step_m_s=tracker.speed,
                old=old,candidate=new,interior_leg_index=controller.motion.leg_index))
        return old  # NEVER apply the candidate to an archived trajectory.
    clone=FunctionType(original.__code__,dict(original.__globals__,_mission_handoff=observe),
        original.__name__,original.__defaults__,original.__closure__)
    clone.__kwdefaults__=original.__kwdefaults__
    controller.motion.tracking_step=clone
    snapshot=packet['required_tensor_snapshot']
    x,y,z,qx,qy,qz,qw=snapshot['transform']
    yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    steer=[snapshot['positions'][snapshot['names'].index('base_link_to_'+a+'_active_steering')]
           for a in ('front','rear')]
    initial=controller.step(0.,[x,y,yaw],math.hypot(*snapshot['velocity'][:2]),steer,
        drive_hold_confirmed=False)
    prior_held=initial['drive_hold_requested']
    count=0;max_speed_difference=0.;max_progress_difference=0.;first_mismatch=None
    for row in csv.DictReader(Path(raw['trace']).open()):
        stamp[0]=float(row['time_s'])
        speed=max(math.hypot(float(row['vx_physx_m_s']),float(row['vy_physx_m_s'])),
                  float(row['pose_filtered_speed_m_s']))
        try:
            info=controller.step(stamp[0],[float(row[k]) for k in ('x_m','y_m','yaw_rad')],
                speed,[float(row[k]) for k in ('front_rad','rear_rad')],
                drive_hold_confirmed=bool(prior_held))
        except Exception as error:
            first_mismatch=dict(time_s=stamp[0],error=type(error).__name__+': '+str(error))
            break
        front_kp,rear_kp=(float(row['drive_'+a+'_actual_kp']) for a in ('front','rear'))
        if (front_kp>0)!=(rear_kp>0):
            first_mismatch=dict(time_s=stamp[0],error='Inconsistent recorded drive hold gains')
            break
        # Inputs precede same-stamp commands. Gains do not change in PhysX:
        # previous commanded held/released gains determine the next input ACK.
        prior_held=front_kp>0
        ds=abs(info['speed_m_s']-float(row['command_speed_m_s']))
        dp=abs(info['progress_m']-float(row['path_progress_m']))
        max_speed_difference=max(max_speed_difference,ds)
        max_progress_difference=max(max_progress_difference,dp);count+=1
        if ds>1e-9 or dp>1e-8 or info['phase']!=row['phase']:
            first_mismatch=dict(time_s=stamp[0],speed_difference_m_s=ds,
                progress_difference_m=dp,replayed_phase=info['phase'],recorded_phase=row['phase'])
            break
    if count==0 and first_mismatch is None:
        first_mismatch=dict(error='NOT_MEASURED_NO_RECORDED_CONTROL_SAMPLES')
    verified=first_mismatch is None and count>0
    return dict(report=str(report_path),report_sha256=runtime.sha(report_path),
        trace=raw['trace'],trace_sha256=raw['trace_sha256'],original_verdict=raw['verdict'],
        original_controller_replay_verified=verified,verified_samples=count,
        maximum_command_difference_m_s=max_speed_difference,maximum_progress_difference_m=max_progress_difference,
        first_replay_mismatch=first_mismatch,
        candidate_predicate_difference_count=len(predicate_differences),
        first_candidate_difference=predicate_differences[0] if predicate_differences else None,
        candidate_control_revalidation_needed=(bool(predicate_differences) if verified else None),
        unaffected_in_archived_controller_inputs=(verified and not predicate_differences),
        no_counterfactual_physical_success_or_RMS_claim=True)


def audit():
    ledger=runtime.read(followup.LEDGER)
    rows=ledger['records']+ledger['steering_initial_supplement']['records']
    records=[]
    for row in rows:
        if not row['actual_route_started']:
            continue
        target=OUTPUT/'individual'/(row['cell_id']+'.json')
        try:
            value=replay(row['actual_report'])
        except Exception as error:
            value=dict(report=row['actual_report'],original_controller_replay_verified=False,
                verified_samples=0,candidate_predicate_difference_count=0,
                first_replay_mismatch=dict(error=type(error).__name__+': '+str(error)),
                candidate_control_revalidation_needed=None,unaffected_in_archived_controller_inputs=False)
        value.update(cell_id=row['cell_id'],task=row['task'],method=row['method'])
        runtime.evidence.write_exclusive(target,value);records.append(value)
        print('[CUSP REPLAY]',row['cell_id'],value['original_controller_replay_verified'],
            value['candidate_predicate_difference_count'],flush=True)
    result=dict(version='ARCHIVED_CONTROLLER_EXACT_REPLAY_STOPPED_CUSP_CANDIDATE_v1',
        candidate_reserve_m=STOPPED_HANDOFF_RESERVE_M,
        derived_from=dict(minimum_braking_target_m_s=MIN_TARGET_M_S,offset_m=BRAKING_OFFSET_M,
            acceleration_m_s2=ACCELERATION_M_S2,original_max_control_gap_s=MAX_CONTROL_GAP_S),
        additional_handoff_requires_original_measured_stopped_5mm_s=True,
        final_leg_goal_hold_collision_and_actual_physical_limits_unchanged=True,
        candidate_applied_to_actual_or_replayed_controller=False,records=records,
        old_recorded_outcomes_not_changed=True,
        unverified_replays_NOT_called_unaffected=True,
        no_blanket_claim_that_all_old_experiments_need_rerun=True)
    runtime.evidence.write_exclusive(OUTPUT/'summary.json',result)
    return result


if __name__=='__main__':
    print('Reviewed',len(audit()['records']),'archived routes; no motion')
