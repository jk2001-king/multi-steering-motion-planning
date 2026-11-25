"""NEW composition: shared timing + the EXISTING forward course equation.

Only counterphase FORWARD is supported. This is not a new gain fit, reverse
feedback transplant, planner decision, CICS coupling or old-controller result.
The 1.5/2 law and >5mm/s course-definition threshold are inherited; the new
timing/braking state machine and dense reference adaptation are disclosed.
"""

import math

from hospital_timed_path_execution import TimedLegController
from hospital_h3b_first_leg_feedback_review import TRACKING_GAINS
from hospital_path_execution import relative_pose, wrap


VERSION = 'H3B_shared_timing_existing_counterphase_forward_course_composition_v1'


class TimedCourseForward(TimedLegController):
    def __init__(self, profile, started):
        if profile.get('mode') != 'coordinated_counterphase' or profile.get('direction') != 1:
            raise ValueError('Existing forward course equation is NOT a reverse or single-axle controller')
        super().__init__(profile, started)

    def step(self, timestamp, pose, measured_speed, actual_steering, body_xy_velocity):
        if len(body_xy_velocity) != 2 or not all(math.isfinite(v) for v in body_xy_velocity):
            raise ValueError('Finite measured body-frame course velocity required')
        result = super().step(timestamp, pose, measured_speed, actual_steering)
        u, v = body_xy_velocity
        beta = math.atan2(v, u) if u > .005 else 0.
        reference = result['reference_local_xyyaw']
        _, cross_error, _ = relative_pose(reference, pose)
        course_error = wrap(reference[2] - pose[2] - beta)
        nominal = result['curvature_1_m']
        if self.phase == 'tracking' and not self.finished:
            result['curvature_1_m'] = max(-.70, min(.70, nominal
                + TRACKING_GAINS['yaw_error_gain_1_m_rad'] * course_error
                + TRACKING_GAINS['cross_error_gain_1_m2'] * cross_error))
        result.update(nominal_reference_curvature_1_m=nominal,
            measured_course_minus_body_yaw_deg=math.degrees(beta),
            disclosed_existing_forward_course_equation_with_NEW_timing=True)
        return result
