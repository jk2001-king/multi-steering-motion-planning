"""One explicit final-forward DESIGN condition: planned cap 30mm/s.

Physical cap remains 50mm/s, acceleration 20mm/s², steering slew 8deg/s.
No step-law or gain override: retain the course/timing composition exactly.
This candidate was informed by reused synthetic design cases, NOT an
independently validated improvement or proof that slower avoids caster stall.
"""

import numpy as np

from hospital_h3b_timed_course_forward import TimedCourseForward, TRACKING_GAINS


VERSION = 'H3B_final_forward_existing_course_shared_timing_planned_cap30_v1'
PLANNED_SPEED_CAP_M_S = .030


class CappedTimedCourseForward(TimedCourseForward):
    def __init__(self, profile, started):
        super().__init__(profile, started)
        timing = self.timing
        timing.speed = np.minimum(timing.speed, PLANNED_SPEED_CAP_M_S)
        dx = np.diff(timing.leg.longitudinal)
        timing.acceleration = np.diff(timing.speed ** 2) / (2. * dx)
        timing.rate = timing.slope * np.maximum(timing.speed[:-1], timing.speed[1:])
        timing.time = np.r_[0., np.cumsum(2. * dx / (timing.speed[:-1] + timing.speed[1:]))]
        if (max(abs(timing.acceleration)) > .020 + 1e-12 or np.degrees(max(timing.rate)) > 8. + 1e-10
                or max(timing.speed) > PLANNED_SPEED_CAP_M_S + 1e-12):
            raise ValueError('Stricter planned speed profile violated unchanged physical bounds')
