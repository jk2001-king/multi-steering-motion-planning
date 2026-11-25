"""Common stopped-at-actual-goal handoff; no success criterion is relaxed.

A planned endpoint may lie beyond the actual goal within its25mm planning
tolerance. A stationary robot already inside the actual30mm/2deg goal can
therefore remain outside the nominal25mm reference-end handoff indefinitely.
Add only a final-leg handoff into the ORIGINAL braking/capture/fixed-hold
procedure. Passing the complete actual5s hold is still mandatory.
"""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_cusp_candidate as parent

runtime=parent.runtime
VERSION='COMMON1330_STOPPED_ACTUAL_GOAL_HANDOFF_v2'
OUTPUT=parent.review.followup.OUTPUT/'actual_goal_handoff_v2_review'


def stopped_at_actual_goal(motion,tracker,virtual_pose,speed,remaining):
    if not tracker.mission_final_leg:return False
    length=float(motion.reference.arc[-1])
    if (length<=0. or motion.reference.progress<.95*length or remaining>.050
        or speed>.005 or speed<0.):return False
    actual_yaw=virtual_pose[2]-(math.pi if motion.leg['direction']<0 else 0.)
    goal=motion.goal
    return (math.hypot(virtual_pose[0]-goal[0],virtual_pose[1]-goal[1])<=.030
        and abs(math.atan2(math.sin(actual_yaw-goal[2]),math.cos(actual_yaw-goal[2])))<=math.radians(2.))


def bind_goal_handoff(motion):
    original=motion.tracking_step
    if (getattr(original,'cusp_reserve_version',None)!=parent.VERSION
        or getattr(original,'actual_goal_handoff_version',None)):
        raise ValueError('Exact once-bound common interior-cusp controller required')
    prior=original.__globals__['_mission_handoff']
    motion.actual_goal_handoff_events=[]
    def handoff(tracker,pose,speed,remaining):
        old=prior(tracker,pose,speed,remaining)
        new=old or stopped_at_actual_goal(motion,tracker,pose,speed,remaining)
        if new and not old:
            motion.actual_goal_handoff_events.append(dict(remaining_reference_m=remaining,
                measured_speed_m_s=speed,actual_goal_xyyaw=list(motion.goal),
                virtual_observation_xyyaw=list(pose),direction=motion.leg['direction'],
                begins_original_terminal_braking_and_fixed_hold_NOT_completion=True))
        return new
    bound=runtime.bind(original,_mission_handoff=handoff)
    bound.__dict__.update(original.__dict__)
    bound.actual_goal_handoff_version=VERSION
    motion.tracking_step=bound
    return motion


def motion_class(profile):
    original=parent.motion_class(profile)
    class CommonActualGoalController(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);bind_goal_handoff(self)
    return CommonActualGoalController


def contract(profile,*,simulation_budget_s,wall_budget_s=900.):
    old=parent.contract(profile,simulation_budget_s=simulation_budget_s,wall_budget_s=wall_budget_s)
    value=copy.deepcopy(old)
    value.update(version=VERSION,parent_controller_version=old['version'],
        additional_change='prior interior stopped-cusp correction plus stopped inside actual goal final handoff',
        final_leg_handoff_goal_hold_collision_and_actuation_limits_unchanged=False,
        actual_goal_position_heading_speed_and_all_fixed_hold_collision_actuation_limits_unchanged=True,
        final_handoff_max_remaining_reference_m=.050,final_handoff_minimum_total_progress_fraction=.95,
        final_handoff_actual_position_m=.030,final_handoff_actual_yaw_deg=2.,
        final_handoff_measured_speed_m_s=.005,
        no_success_without_original_braking_ACK_capture_fixed5s_hold_and_postflight=True)
    return value


def controller(plan,geometry,*,profile,simulation_budget_s,start_time_s=0.,wall_budget_s=900.):
    value=parent.controller(plan,geometry,profile=profile,simulation_budget_s=simulation_budget_s,
        start_time_s=start_time_s,wall_budget_s=wall_budget_s)
    bind_goal_handoff(value.motion)
    value.contract=contract(profile,simulation_budget_s=simulation_budget_s,wall_budget_s=wall_budget_s)
    return value


def private_sequence():
    return SimpleNamespace(**dict(vars(parent.sequence),unified_geometry_motion_class=motion_class,
        unified_geometry_lifecycle_controller=controller,unified_geometry_lifecycle_contract=contract))


def model_binding(*args,**kwargs):
    run,execution,design=runtime.bind(parent.model_binding,private_sequence=private_sequence,VERSION=VERSION)(*args,**kwargs)
    def measured(*a,**k):
        value,rows=run(*a,**k)
        value.update(actual_goal_handoff_version=VERSION,
            actual30mm_2deg_5mm_s_and_fixed_hold_criteria_NOT_changed=True,
            nominal_final_reference_handoff_changed_explicitly=True)
        return value,rows
    return measured,execution,design


def replay(report_path):
    report_path=Path(report_path);raw=runtime.read(report_path);holder=[]
    version=raw.get('controller_candidate_version')
    if version not in (None,parent.VERSION):raise ValueError('Unreviewed actual controller version')
    factory=parent.controller if version==parent.VERSION else parent.sequence.unified_geometry_lifecycle_controller
    def make(*args,**kwargs):
        value=factory(*args,**kwargs);holder.append(value);return value
    def compare(old,tracker,pose,speed,remaining):
        return old(tracker,pose,speed,remaining) or stopped_at_actual_goal(holder[0].motion,tracker,pose,speed,remaining)
    sequence=SimpleNamespace(**dict(vars(parent.sequence),unified_geometry_lifecycle_controller=make))
    value=runtime.bind(parent.review.replay,sequence=sequence,stopped_handoff_candidate=compare)(report_path)
    value.update(candidate_version=VERSION,replayed_actual_version=version or 'ORIGINAL_COMMON30',
        changed_predicate_is_final_actual_goal_handoff_only=True,
        old_interior_cusp_variant_retained_when_replaying_its_actual_trace=True)
    return value


def audit():
    paths={Path(r['report']) for r in runtime.read(parent.review.OUTPUT/'summary.json')['records']}
    paths.update(parent.review.followup.OUTPUT.rglob('route/route_readback.json'))
    records=[]
    for path in sorted(paths):
        raw=runtime.read(path)
        if raw.get('new_route_trials')!=1:continue
        value=replay(path);records.append(value)
        runtime.evidence.write_exclusive(OUTPUT/'individual'/(runtime.sha(path)[:20]+'.json'),value)
    value=dict(version=VERSION,actual_trials=0,actual_candidate_never_applied_in_replay=True,
        verified_n=sum(r['original_controller_replay_verified'] for r in records),
        records_n=len(records),changed_n=sum(bool(r['candidate_predicate_difference_count']) for r in records),
        unverified_n=sum(not r['original_controller_replay_verified'] for r in records),records=records,
        final_success_limits_and_fixed_hold_criteria_unchanged=True,
        no_counterfactual_physical_success_or_RMS_claim=True)
    runtime.evidence.write_exclusive(OUTPUT/'summary.json',value)
    return value


if __name__=='__main__':
    value=audit();print({k:v for k,v in value.items() if k!='records'})
