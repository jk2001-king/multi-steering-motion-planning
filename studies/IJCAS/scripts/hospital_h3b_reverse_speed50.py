"""Separate, explicitly selected 50 mm/s rear-reverse speed-cap intervention.

Only the reverse tracking speed literal changes. The historical 10 mm/s
controller, its policy and all of its PASS/FAIL receipts remain unchanged.
This module alone does NOT authorize a simulator trial.
"""

import ast
import inspect
import math
import textwrap

from hospital_h3b_measured_reverse import FirstReverseProgress as OriginalReverseProgress
from hospital_h3b_measured_reverse import ramp, wrap


SPEED_CAP_M_S = .050
VERSION = 'H3B_rear_reverse_speed_cap_50mm_s_separate_intervention_v1'


class FirstReverseProgress(OriginalReverseProgress):
    def step(self,timestamp,pose,measured_speed,actual_steering):
        if (not math.isfinite(timestamp) or not math.isfinite(measured_speed) or measured_speed<0
                or len(actual_steering)!=2 or not all(math.isfinite(v) for v in actual_steering)):
            raise ValueError('Finite actual speed/steering required')
        dt=timestamp-self.previous_time
        if not 0<=dt<=.25:raise RuntimeError('Existing 0.25s control-gap guard')
        self.previous_time=timestamp;ref,error,k=self.reference.project(pose)
        heading=math.degrees(wrap(pose[2]-ref[2]));remaining=self.reference.arc[-1]-self.reference.progress
        if error>.080 or abs(heading)>10.:raise RuntimeError('Existing curved-reference 80mm/10deg guard')
        if self.phase=='tracking' and remaining<=.005+self.speed*self.speed/(2*.020):
            self.phase='braking';self.braking_started=timestamp
        if self.phase=='braking' and timestamp-self.braking_started>8.:
            raise TimeoutError('Existing 8s measured braking/neutral guard')
        self.speed=ramp(self.speed,-.050 if self.phase=='tracking' else 0.,dt,.020)
        if self.phase=='braking':k=0.
        ready=(self.phase=='braking' and abs(self.speed)<1e-7 and measured_speed<=.005
            and max(abs(v) for v in actual_steering)<=math.radians(2.5))
        return dict(speed_m_s=self.speed,curvature_1_m=k,ready=ready,phase=self.phase,
            progress_m=self.reference.progress,remaining_m=remaining,path_error_m=error,
            path_heading_error_deg=heading,reference_local_xyyaw=ref.tolist())


def verify_speed_only_change():
    old = ast.parse(textwrap.dedent(inspect.getsource(OriginalReverseProgress.step)))
    new = ast.parse(textwrap.dedent(inspect.getsource(FirstReverseProgress.step)))
    changed = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Constant) and type(node.value) is float and node.value == .050:
            node.value = .010; changed += 1
    if changed != 1 or ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
        raise ValueError('Separate probe must change ONLY the 10 -> 50 mm/s tracking-speed literal')
    return dict(only_reverse_tracking_speed_literal_changed=True,
                original_cap_m_s=.010, separate_probe_cap_m_s=SPEED_CAP_M_S,
                braking_acceleration_m_s2=.020, stop_m_s=.005,
                tracking_m_deg=[.080, 10.], steering_slew_deg_s=8.,
                terminal_m_deg=[.030, 2.], strict_clearance_m=.010,
                original_10mm_s_policy_or_results_modified=False)
