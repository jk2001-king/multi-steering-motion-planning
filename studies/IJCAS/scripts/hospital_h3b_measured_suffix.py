"""Measured-state SUFFIX geometry model, not an actuator/robot pose reset.

The original mission frame, goal, map and spatial search bounds are unchanged.
Only the planning initial condition is the saved measured first-cusp pose.
Existing mode-aware integration/Jacobians are rigidly composed into that SAME
mission frame. Actual steering/caster/speed are retained as required later
dynamic/physical inputs; nominal zero curvature is NOT an actual angle reset.
"""

import math

import numpy as np

from hospital_h3b_mode_aware_seed import ModeSeedModel


VERSION='H3B_measured_first_cusp_suffix_candidate_v1'


class MeasuredSuffixModel(ModeSeedModel):
    def __init__(self,actions,family,start_in_original_mission_frame):
        super().__init__(actions,family)
        self.measured_start=np.asarray(start_in_original_mission_frame,float).copy()
        if self.measured_start.shape!=(3,) or not np.isfinite(self.measured_start).all():
            raise ValueError('Finite measured planning initial pose in the ORIGINAL mission frame required')
        yaw=self.measured_start[2]
        self.start_rotation=np.array(((math.cos(yaw),-math.sin(yaw)),(math.sin(yaw),math.cos(yaw))))

    def integrate(self,z,substeps=2,jacobian=False,independent=False):
        poses,derivative,distance,curvature,controls=super().integrate(z,substeps,jacobian,independent)
        poses[:,:2]=poses[:,:2]@self.start_rotation.T+self.measured_start[:2]
        poses[:,2]+=self.measured_start[2]
        if derivative is not None:
            derivative[:,:2,:]=np.einsum('ij,njp->nip',self.start_rotation,derivative[:,:2,:])
        return poses,derivative,distance,curvature,controls


def from_saved_candidate(candidate,measured_start):
    """Drop completed controls only; preserve all four remaining signed legs."""
    geometry=candidate['geometry_audit']
    counts=geometry['leg_control_counts'];directions=geometry['leg_directions']
    if len(counts)!=5 or directions!=[1,-1,1,-1,1]:
        raise ValueError('Expected the original five-leg candidate after exactly ONE completed forward leg')
    first=counts[0];knots=np.asarray(geometry['curvature_knots_1_m'],float)[first:]
    actions=[];i=0
    for count,direction in zip(counts[1:],directions[1:]):
        for _ in range(count):
            actions.append(dict(mode='coordinated_counterphase',curvature_1_m=float((knots[i]+knots[i+1])/2),direction=direction));i+=1
    model=MeasuredSuffixModel(actions,geometry['family'],measured_start)
    model.initial_parameters=np.r_[knots[model.free],geometry['leg_longitudinal_distances_m'][1:]]
    model.unpack(model.initial_parameters)
    if list(model.leg_directions)!=[-1,1,-1,1] or len(model.cusps)!=3:
        raise ValueError('Remaining direction resources changed')
    return model
