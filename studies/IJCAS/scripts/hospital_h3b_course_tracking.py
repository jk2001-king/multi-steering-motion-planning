"""Shared-executor CANDIDATE, not a change to IJCAS planning or an old PASS.

Keep reference, 1.5/2 gains, speed/braking/stop/neutral/terminal limits fixed.
For this counterphase FORWARD component the reference course equals its body
yaw (nominal beta=0). Use measured body course while actually moving, rather
than assuming lateral velocity is zero. No bias fit from the failed endpoint.
Every comparison method would need this SAME new executor in a NEW cohort.
"""

import math

from hospital_h3b_first_leg_feedback_review import FirstLegFeedback, TRACKING_GAINS
from hospital_path_execution import wrap, relative_pose


VERSION = 'H3B_shared_course_feedback_candidate_v1'


def body_velocity(world_xy_velocity, yaw):
    if len(world_xy_velocity)!=2 or not all(math.isfinite(v) for v in (*world_xy_velocity,yaw)):
        raise ValueError('Finite actual world velocity/yaw required')
    c,s=math.cos(yaw),math.sin(yaw)
    return c*world_xy_velocity[0]+s*world_xy_velocity[1], -s*world_xy_velocity[0]+c*world_xy_velocity[1]


class CourseFirstLegFeedback(FirstLegFeedback):
    """Exact old state machine/gains; disclose yaw->course tracking change."""
    def step(self,timestamp,pose,measured_speed,measured_steer,body_xy_velocity):
        if len(body_xy_velocity)!=2 or not all(math.isfinite(v) for v in body_xy_velocity):
            raise ValueError('Finite actual body-frame velocity required')
        # Old method still performs ALL pose/body-yaw guards, state transitions,
        # original braking, measured-stop and neutral checks before this change.
        result=super().step(timestamp,pose,measured_speed,measured_steer)
        u,v=body_xy_velocity
        # Same existing 5 mm/s stopped threshold; course is undefined at rest.
        beta=math.atan2(v,u) if u>.005 else 0.
        if self.phase=='tracking':
            index, reference=self.reference.at(self.reference.progress)
            _,cross_error,_=relative_pose(reference,pose)
            course_error=wrap(reference[2]-pose[2]-beta)
            result['curvature_1_m']=self.curvature=max(-.70,min(.70,
                self.reference.actions[index]['curvature_1_m']
                +TRACKING_GAINS['yaw_error_gain_1_m_rad']*course_error
                +TRACKING_GAINS['cross_error_gain_1_m2']*cross_error))
        result.update(measured_course_minus_body_yaw_deg=math.degrees(beta),
            disclosed_shared_executor_course_feedback=True)
        self.last_result=dict(result)
        return result
