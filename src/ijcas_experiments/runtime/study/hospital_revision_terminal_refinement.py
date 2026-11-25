"""Bounded two-coordinate escape from a strict-goal local minimum.

OFFLINE only. No new curvature, motion mode, primitive, goal tolerance, speed,
model, body margin or controller is introduced. A valid result still needs
the existing independent full-path audit. The caller owns the original wall
deadline; every proposed suffix is checked under that same deadline.
"""
import inspect
import itertools
import math

from patient_initial_stability.hierarchical_dynamics_planner import DynamicRouteResult
from patient_initial_stability.steering_lattice import ALL_MODES
import hospital_mission_planning as mission


def bounded_pairs(actions,evaluate,*,modes,curvatures,check_deadline,
                  window=12,shortlist=24,goal_aware_ranking=False):
    """Evaluate single changes, then pairs on distinct primitive indices.

    Singles need not immediately improve the score; this is what lets a
    coupled change escape coordinate descent. Known invalid trajectories are
    never returned. This finite search is not an infeasibility proof.
    """
    if (type(window) is not int or not 1<=window<=12 or type(shortlist) is not int
            or not 2<=shortlist<=24 or not actions):
        raise ValueError('Explicit bounded terminal neighbourhood required')
    proposals=[];log=dict(single_evaluations=0,pair_evaluations=0,accepted=False)
    for index in range(max(0,len(actions)-window),len(actions)):
        for mode,curvature in itertools.product(modes,curvatures):
            if (mode,curvature)==(actions[index]['mode'],actions[index]['curvature_1_m']):
                continue
            check_deadline()
            candidate=[dict(a) for a in actions]
            candidate[index].update(mode=mode,curvature_1_m=curvature)
            result=evaluate(candidate,index)
            log['single_evaluations']+=1
            if result is None:
                continue
            if result['goal_accepted']:
                log.update(accepted=True,changes=[(index,mode,curvature)])
                return result,log
            rank=result['constraint_rank'] if goal_aware_ranking else result['score']
            proposals.append((rank,index,mode,curvature))
    # Preserve alternatives on several indices; one index must not consume
    # the entire shortlist simply because its tiny changes rank first.
    ranked=sorted(proposals)
    kept=[];per_index={}
    for item in ranked:
        if per_index.get(item[1],0)>=3:
            continue
        kept.append(item);per_index[item[1]]=per_index.get(item[1],0)+1
        if len(kept)>=shortlist:
            break
    for a,b in itertools.combinations(kept,2):
        if a[1]==b[1]:
            continue
        check_deadline()
        candidate=[dict(c) for c in actions]
        for change in (a,b):
            candidate[change[1]].update(mode=change[2],curvature_1_m=change[3])
        result=evaluate(candidate,min(a[1],b[1]))
        log['pair_evaluations']+=1
        if result is not None and result['goal_accepted']:
            log.update(accepted=True,changes=[a[1:],b[1:]])
            return result,log
    return None,log


def paired_binding(original_refine,*,window=12,shortlist=24,goal_aware_ranking=False):
    """Reuse the full original refiner; fallback ONLY after strict failure."""
    signature=inspect.signature(original_refine)
    ns=original_refine.__globals__
    def refine(*args,**kwargs):
        result,diagnostics=original_refine(*args,**kwargs)
        if result is not None or 'terminal_position_error_m' not in diagnostics:
            return result,diagnostics
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        if v['terminal_extension_edges']!=0 or v['propagate_initial_steering'] is not True:
            raise ValueError('No terminal extension or uninitialized state in paired fallback')
        actions=[dict(mode=a['mode'],curvature_1_m=float(a['curvature_1_m']),
                      direction=int(a['direction'])) for a in v['actions']]
        for change in diagnostics['refinement_changes']:
            if len(change)!=3:
                raise ValueError('Exact mode/curvature refinement history required')
            i,mode,curvature=change
            actions[i].update(mode=mode,curvature_1_m=curvature)
        speeds=[]
        for i in range(len(actions)):
            cap=math.sqrt(v['terminal_speed_tolerance_m_s']**2+
                2*v['terminal_deceleration_m_s2']*mission.direction_remaining_arc(actions,i,v['step_m']))
            speeds.append(max(s for s in v['speed_levels_m_s'] if s<=cap+1e-12))
        initial=ns['State'](0.,0.,0.,v['initial_mode'],v['initial_front_rad'],v['initial_rear_rad'],True,0)
        checker=v['collision_checker'];propagate=ns['propagate'];base=None
        counts=dict(rigid_body=0,resource=0,stop=0,cusp=0,geometry=0)
        def evaluate(candidate,changed_index=0):
            checker.check_deadline()
            if base is not None and changed_index:
                states=list(base['states'][:changed_index]);times=list(base['times'][:changed_index])
                costs=list(base['costs'][:changed_index]);state=states[-1]
            else:
                states=[];times=[];costs=[];state=initial;changed_index=0
            for i in range(changed_index,len(candidate)):
                checker.check_deadline();a=candidate[i];previous=state
                state,poses,dt=propagate(state,a['mode'],a['curvature_1_m'],a['direction'],
                    step_m=v['step_m'],speed_m_s=max(v['speed_levels_m_s']),dynamics=v['dynamics'],
                    rigid_body_model=v['rigid_body_model'],target_speed_m_s=speeds[i])
                remaining=mission.direction_remaining_arc(candidate,i,v['step_m'])
                reason=None
                if not state.rigid_body_feasible:reason='rigid_body'
                elif not getattr(poses,'resource_feasible',True):reason='resource'
                elif math.hypot(state.longitudinal_velocity,state.lateral_velocity)>math.sqrt(
                        v['terminal_speed_tolerance_m_s']**2+2*v['terminal_deceleration_m_s2']*remaining)+1e-12:reason='stop'
                elif not mission.stop_connected_edge(previous,state,candidate,i):reason='cusp'
                elif checker(poses):reason='geometry'
                if reason:
                    counts[reason]+=1
                    return None
                cost=dt+state.rigid_body_penalty_s
                if v['include_configuration_cost'] and previous.started:
                    if v['dynamics'].include_mode_switch_cost and a['mode']!=previous.mode:
                        cost+=ns['MODE_SWITCH_PENALTY_S']
                    if v['dynamics'].include_steering_transient_cost:
                        front,rear=ns['steering_targets'](a['mode'],a['curvature_1_m'])
                        cost+=ns['STEERING_TRANSIENT_PENALTY_WEIGHT']*max(
                            abs(front-previous.front),abs(rear-previous.rear))/ns['STEERING_RATE_RAD_S']
                states.append(state);times.append(dt);costs.append(cost)
            ep=math.hypot(state.x-v['goal'][0],state.y-v['goal'][1])
            ey=abs(math.atan2(math.sin(state.yaw-v['goal'][2]),math.cos(state.yaw-v['goal'][2])))
            speed=math.hypot(state.longitudinal_velocity,state.lateral_velocity)
            return dict(states=states,times=times,costs=costs,actions=candidate,
                position_error=ep,heading_error=ey,terminal_speed=speed,
                score=100*ep+v['heading_objective_weight']*ey,
                constraint_rank=(max(ep/v['goal_position_tolerance_m'],ey/v['goal_heading_tolerance_rad'],
                    speed/v['terminal_speed_tolerance_m_s']),
                    ep/v['goal_position_tolerance_m']+ey/v['goal_heading_tolerance_rad'],
                    100*ep+v['heading_objective_weight']*ey),
                goal_accepted=ep<=v['goal_position_tolerance_m'] and ey<=v['goal_heading_tolerance_rad']
                    and speed<=v['terminal_speed_tolerance_m_s'])
        base=evaluate(actions)
        if base is None:
            raise ValueError('Reconstructed original final candidate failed original predicates')
        if (abs(base['position_error']-diagnostics['terminal_position_error_m'])>1e-9
                or abs(math.degrees(base['heading_error'])-diagnostics['terminal_heading_error_deg'])>1e-7):
            raise ValueError('Original refined candidate not exactly reconstructed')
        accepted,log=bounded_pairs(actions,evaluate,modes=ALL_MODES,
            curvatures=ns['CURVATURES'],check_deadline=checker.check_deadline,window=window,shortlist=shortlist,
            goal_aware_ranking=goal_aware_ranking)
        diagnostics=dict(diagnostics,paired_terminal_refinement=dict(log,rejections=counts,
            base_position_error_m=base['position_error'],base_heading_error_deg=math.degrees(base['heading_error']),
            same_goal_constraints_speed_model_and_candidate_family=True))
        if accepted is None:
            return None,diagnostics
        diagnostics.update(terminal_position_error_m=accepted['position_error'],
            terminal_heading_error_deg=math.degrees(accepted['heading_error']),
            terminal_speed_m_s=accepted['terminal_speed'])
        return DynamicRouteResult(v['route_id'],tuple(accepted['states']),tuple(accepted['actions']),
            tuple(speeds),tuple(accepted['times']),math.fsum(accepted['costs']),
            tuple(sorted(diagnostics.items()))),diagnostics
    refine.mode_refinement_binding=dict(original_refine.mode_refinement_binding,
        additional_pair_search_only_after_strict_failure=True,terminal_window=window,shortlist=shortlist,
        proposal_ranking=('normalized_strict_goal_violation_then_original_goal_score' if goal_aware_ranking
            else 'original_weighted_goal_score'),
        no_new_primitive_or_goal_tolerance_or_physical_control=True)
    return refine
