"""Common map-aware SEARCH GUIDANCE only, never a collision certificate.

The inherited goal-bearing heuristic steers toward the goal before the doorway.
This separate revision guides all four methods toward the same map-derived
door passage. Actual edges retain original propagation and exact full SAT.
No steering mode, goal, footprint, speed, force bound or search bound changes.
"""

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from hospital_native_envelope import cloned_plan_method
from plan_hospital_h1 import pose_guidance,SPEED


class NavigationGuidance:
    def __init__(self,occupied,origin,resolution,start,goal_world,stride=4):
        self.origin=np.asarray(origin,dtype=float)
        self.resolution=float(resolution)*stride
        self.start=tuple(start);self.goal=tuple(goal_world)
        self.lookahead_m=1.630/2
        self.c,self.s=math.cos(start[2]),math.sin(start[2])
        rows=math.ceil(occupied.shape[0]/stride);cols=math.ceil(occupied.shape[1]/stride)
        padded=np.ones((rows*stride,cols*stride),dtype=bool)
        padded[:occupied.shape[0],:occupied.shape[1]]=occupied
        blocked=padded.reshape(rows,stride,cols,stride).any(axis=(1,3))
        # This relaxed center-space field is ONLY a non-admissible heuristic.
        # Its inflation/rounding is never allowed to approve/reject a true edge.
        free=distance_transform_edt(~blocked)*self.resolution>(.965/2+.010+self.resolution*math.sqrt(2)/2)
        self.distance=np.full((rows,cols),np.inf,dtype=float)
        successor=np.full((rows,cols,2),-1,dtype=np.int32)
        gx,gy=np.floor((np.asarray(self.goal[:2])-self.origin)/self.resolution).astype(int)
        if not (0<=gy<rows and 0<=gx<cols and free[gy,gx]):
            raise ValueError("Goal has no relaxed guidance cell; keep original SAT/task, do not edit obstacles")
        self.distance[gy,gx]=0.;queue=[(0.,int(gy),int(gx))]
        neighbors=tuple((dy,dx,math.hypot(dx,dy)*self.resolution) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy)
        while queue:
            cost,y,x=heapq.heappop(queue)
            if cost!=self.distance[y,x]:continue
            for dy,dx,step in neighbors:
                yy,xx=y+dy,x+dx
                if not (0<=yy<rows and 0<=xx<cols and free[yy,xx]):continue
                if dx and dy and not (free[y,xx] and free[yy,x]):continue
                new=cost+step
                if new<self.distance[yy,xx]-1e-12:
                    self.distance[yy,xx]=new;successor[yy,xx]=(y,x)
                    heapq.heappush(queue,(new,yy,xx))
        self.targets=np.full((rows,cols,2),np.nan,dtype=float)
        for y,x in zip(*np.nonzero(np.isfinite(self.distance))):
            yy,xx=int(y),int(x);travel=0.
            while travel<self.lookahead_m and successor[yy,xx,0]>=0:
                ny,nx=map(int,successor[yy,xx]);travel+=math.hypot(nx-xx,ny-yy)*self.resolution
                yy,xx=ny,nx
            self.targets[y,x]=self.goal[:2] if successor[yy,xx,0]<0 else self.origin+(np.asarray([xx,yy])+.5)*self.resolution
        self.parameters={"kind":"common_map_distance_and_lookahead_pose_guidance_non_admissible",
            "coarse_stride":stride,"guidance_resolution_m":self.resolution,"lookahead_m":self.lookahead_m,
            "relaxed_center_radius_m":.965/2+.010,"never_used_as_collision_checker":True,
            "fallback_to_original_guidance_when_field_unreachable":True,
            "global_optimality_claimed":False,"same_guidance_for_all_methods":True}

    def __call__(self,state,goal):
        wx=self.start[0]+self.c*state.x-self.s*state.y
        wy=self.start[1]+self.s*state.x+self.c*state.y
        x,y=np.floor((np.asarray([wx,wy])-self.origin)/self.resolution).astype(int)
        if not (0<=y<self.distance.shape[0] and 0<=x<self.distance.shape[1]) or not math.isfinite(self.distance[y,x]):
            return pose_guidance(state,goal)
        distance=math.hypot(state.x-goal[0],state.y-goal[1])
        if distance<=.025:return pose_guidance(state,goal)
        tx,ty=self.targets[y,x]
        bearing=math.atan2(ty-wy,tx-wx)-self.start[2]
        wrap=lambda value:math.atan2(math.sin(value),math.cos(value))
        turn=abs(wrap(bearing-state.yaw))+abs(wrap(goal[2]-bearing))
        return float(self.distance[y,x])/SPEED+.4*turn/(.70*SPEED)


def guided_plan_method(guidance):
    method=cloned_plan_method()
    method.__globals__["pose_guidance"]=guidance
    return method
