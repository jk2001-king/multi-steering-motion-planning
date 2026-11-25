"""Goal-conditioned forward-only kinematic connectivity, NOT a dynamics claim.

Backward graph expansion inverts the existing *forward* SE(2) primitives. It
does not introduce reverse driving. Every candidate is replayed from the exact
original start with direction=+1 and independently checked by polygon SAT.
Instantaneous-steering modes have no geometric history: pose-only merging is
an explicit bounded existence-search revision, not an optimal cost comparison.
"""

from dataclasses import dataclass
import heapq
import math
import time

import numpy as np

from patient_initial_stability.steering_lattice import (
    ALL_MODES, CURVATURES, State, propagate, KINEMATIC_DYNAMICS, wrap,
)
from hospital_geometry_tools import pose_collision
from hospital_native_envelope import NativeHospitalEnvelopeChecker
from hospital_planning_tools import local_to_world


@dataclass(frozen=True)
class Primitive:
    mode: str
    curvature: float
    poses: np.ndarray


def primitives(modes=ALL_MODES):
    result=[]
    for mode in modes:
        for curvature in CURVATURES:
            if curvature==0. and mode!=modes[0]:continue  # geometrically identical straight edges
            state=State(0.,0.,0.,mode,0.,0.,False)
            _,poses,_=propagate(state,mode,curvature,1,speed_m_s=.05,dynamics=KINEMATIC_DYNAMICS)
            result.append(Primitive(mode,curvature,np.asarray(poses)))
    return tuple(result)


def compose(poses,reference):
    poses=np.asarray(poses);c,s=math.cos(reference[2]),math.sin(reference[2])
    return np.column_stack((reference[0]+c*poses[:,0]-s*poses[:,1],
        reference[1]+s*poses[:,0]+c*poses[:,1],reference[2]+poses[:,2]))


def predecessor(current,primitive):
    end=primitive.poses[-1];yaw=current[2]-end[2];c,s=math.cos(yaw),math.sin(yaw)
    return (current[0]-c*end[0]+s*end[1],current[1]-s*end[0]-c*end[1],yaw)


def in_bounds(poses,goal):
    poses=np.asarray(poses)
    half_y=abs(np.sin(poses[:,2]))*1.63/2+abs(np.cos(poses[:,2]))*.965/2
    return bool(np.all(poses[:,0]>=-.3) and np.all(poses[:,0]<=goal[0]+.4)
        and np.all(poses[:,1]-half_y>=-1.99) and np.all(poses[:,1]+half_y<=1.99))


def key(pose):
    return (round(pose[0]/.025),round(pose[1]/.025),round(wrap(pose[2])/math.radians(1.)))


def distance(pose,target):
    return math.hypot(pose[0]-target[0],pose[1]-target[1])


def independent_replay(actions,manifest,occupied,origin,resolution):
    state=State(0.,0.,0.,"coordinated_counterphase",0.,0.,False)
    collisions=samples=0;states=[];all_poses=[];goal=manifest["local_goal_xyyaw"]
    for action in actions:
        state,poses,_=propagate(state,action["mode"],action["curvature_1_m"],action["direction"],
            speed_m_s=.05,dynamics=KINEMATIC_DYNAMICS)
        states.append({"x_m":state.x,"y_m":state.y,"yaw_rad":state.yaw})
        all_poses.extend(poses)
        for pose in poses:
            samples+=1
            collisions+=bool(pose_collision(occupied,local_to_world(pose,manifest["start_xyyaw"]),origin,resolution,.01))
    position=math.hypot(state.x-goal[0],state.y-goal[1]);heading=abs(wrap(state.yaw-goal[2]))
    checks={"all_forward_existing_curvatures_and_modes":bool(actions) and all(a["direction"]==1 and a["mode"] in ALL_MODES
        and a["curvature_1_m"] in CURVATURES for a in actions),"original_polygon_SAT_no_collisions":collisions==0,
        "unchanged_search_bounds":bool(all_poses) and in_bounds(all_poses,goal),
        "original_25mm_position_tolerance":position<=.025,"original_2deg_heading_tolerance":heading<=math.radians(2.)}
    return {"verdict":"PASS" if all(checks.values()) else "FAIL","checks":checks,"colliding_samples":collisions,
        "samples":samples,"terminal_position_error_m":position,"terminal_heading_error_deg":math.degrees(heading),
        "kinematic_states":states,"dynamic_feasibility_certified":False,"PhysX_executed":False}


def search(manifest,occupied,origin,resolution,modes=ALL_MODES,budget_s=30.,node_limit=80000,backward=True):
    begun=time.perf_counter();goal=tuple(manifest["local_goal_xyyaw"]);start=(0.,0.,0.)
    checker=NativeHospitalEnvelopeChecker(occupied,origin,resolution,manifest["start_xyyaw"],deadline=begun+budget_s)
    root,target=(goal,start) if backward else (start,goal);templates=primitives(modes)
    def heuristic(pose):
        return max(distance(pose,target)/.05,abs(wrap(pose[2]-target[2]))/(.7*.05))
    nodes=[(root,None,None,0.)];queue=[(1.5*heuristic(root),0)];best={key(root):0.}
    counts={"expanded":0,"generated":0,"collision":0,"bounds":0,"dominated":0,"candidate_replay_failed":0}
    nearest=root;nearest_score=heuristic(root);actions=[];validation=None;status="NODE_LIMIT_OR_FRONTIER_EXHAUSTED_NOT_GLOBAL_INFEASIBILITY"
    try:
        while queue and counts["expanded"]<node_limit:
            checker.check_deadline();_,index=heapq.heappop(queue);pose,parent,action,cost=nodes[index]
            if cost>best.get(key(pose),math.inf)+1e-12:continue
            counts["expanded"]+=1;score=heuristic(pose)
            if score<nearest_score:nearest_score=score;nearest=pose
            if distance(pose,target)<=.025 and abs(wrap(pose[2]-target[2]))<=math.radians(2.):
                chain=[];i=index
                while nodes[i][1] is not None:
                    chain.append(nodes[i][2]);i=nodes[i][1]
                if not backward:chain.reverse()
                candidate=[{"mode":p.mode,"curvature_1_m":p.curvature,"direction":1} for p in chain]
                checker.check_deadline()
                checked=independent_replay(candidate,manifest,occupied,origin,resolution)
                if checked["verdict"]=="PASS":
                    actions=candidate;validation=checked;status="FORWARD_KINEMATIC_CONNECTION_INDEPENDENTLY_AUDITED_NOT_DYNAMICS_OR_PhysX";break
                counts["candidate_replay_failed"]+=1
            for primitive in templates:
                counts["generated"]+=1
                if backward:
                    new=predecessor(pose,primitive);edge=compose(primitive.poses,new)
                else:
                    edge=compose(primitive.poses,pose);new=tuple(edge[-1])
                new_cost=cost+1.+.02*abs(primitive.curvature)
                state_key=key(new)
                if new_cost>=best.get(state_key,math.inf)-1e-12:
                    counts["dominated"]+=1;continue
                if not in_bounds(edge,goal):counts["bounds"]+=1;continue
                if checker(edge):counts["collision"]+=1;continue
                best[state_key]=new_cost;nodes.append((new,index,primitive,new_cost))
                heapq.heappush(queue,(new_cost+1.5*heuristic(new),len(nodes)-1))
    except TimeoutError:status="CONNECTIVITY_TIME_BUDGET_EXHAUSTED_NOT_GLOBAL_INFEASIBILITY"
    return {"status":status,"actions":actions,"independent_kinematic_audit":validation,
        "budget_s":budget_s,"node_limit":node_limit,"elapsed_s":time.perf_counter()-begun,"allowed_modes":list(modes),
        "graph_expansion":"inverse_forward_SE2_primitives_from_goal_NOT_reverse_driving" if backward else "forward_from_original_start",
        "pose_only_merging_explicit_search_revision":True,"heuristic_weight":1.5,"globally_optimal_or_complete":False,
        "counters":counts,"stored_nodes":len(nodes),"queue_entries":len(queue),"best_combined_pose_sample":list(nearest),
        "best_combined_sample_position_error_m":distance(nearest,target),
        "best_combined_sample_heading_error_deg":math.degrees(wrap(nearest[2]-target[2])),
        "checker_statistics":checker.statistics,"global_infeasibility_proven":False,"PhysX_executed":False}
