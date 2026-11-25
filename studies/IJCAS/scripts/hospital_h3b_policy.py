"""H3-B development contract; explicitly separate from forward-only H3-A.

User authorized bounded reverse on the same hospital geometry and task poses.
Limits are provisional simulation-development choices, not clinical/hardware
identification. Never relax them automatically after an unsuccessful attempt.
"""

import math

from hospital_path_execution_v3 import steering_geometry


POLICY={"version":"H3B_bounded_reverse_stop_align_v1","forward_speed_limit_m_s":.05,
    "reverse_speed_limit_m_s":.01,"acceleration_m_s2":.02,"curvature_limit_1_m":.7,
    "maximum_reverse_episode_m":.50,"maximum_total_reverse_m":1.,"maximum_direction_changes":4,
    "step_m":.05,"clearance_m":.01,"stopped_speed_m_s":.005,"stationary_settle_s":.50,
    "alignment_tolerance_deg":2.5,"alignment_dwell_s":.25,"braking_timeout_s":8.,
    "alignment_timeout_s":16.,"maximum_control_gap_s":.25,"scope":"provisional_simulation_development_not_clinical_limit"}


def route_metrics(actions):
    total=episode=maximum=0.;changes=0;previous=0
    for action in actions:
        direction=action["direction"]
        if direction not in (-1,1):raise ValueError("Direction must be +1 or -1")
        changes+=bool(previous and direction!=previous)
        if direction<0:
            total+=.05;episode=episode+.05 if previous<0 else .05;maximum=max(maximum,episode)
        else:episode=0.
        previous=direction
    return {"reverse_distance_m":total,"maximum_reverse_episode_m":maximum,"direction_changes":changes}


def validate_route_resources(actions):
    metrics=route_metrics(actions)
    if (metrics["reverse_distance_m"]>POLICY["maximum_total_reverse_m"]+1e-9
            or metrics["maximum_reverse_episode_m"]>POLICY["maximum_reverse_episode_m"]+1e-9
            or metrics["direction_changes"]>POLICY["maximum_direction_changes"]):
        raise ValueError("H3-B bounded-reverse resource limit exceeded")
    return metrics


class StopAlignGate:
    """Measured actuator handshake, not a presumed stop or pose reset.

    ``hold_active`` must report verified actuator state from the adapter. Ready
    is returned only AFTER hold was acknowledged and then released. This pure
    state machine alone is not evidence of physical braking/alignment success.
    """

    def __init__(self,direction,mode,curvature,timestamp):
        if direction not in (-1,1) or not math.isfinite(timestamp) or not math.isfinite(curvature) or abs(curvature)>.7:
            raise ValueError("Invalid H3-B gate request")
        self.target_steer=steering_geometry(mode,curvature)
        self.direction,self.mode,self.curvature=direction,mode,curvature
        self.stage="braking";self.started=self.stage_started=timestamp
        self.aligned_since=None;self.saw_hold=False;self.previous_time=None

    def step(self,timestamp,command_speed,measured_speed,measured_steer,hold_active):
        if (len(measured_steer)!=2 or measured_speed<0 or not all(math.isfinite(v) for v in
                (timestamp,command_speed,measured_speed,*measured_steer))):raise ValueError("Invalid measured cusp state")
        if self.previous_time is not None and not 0<=timestamp-self.previous_time<=.25:
            raise RuntimeError("H3-B gate control-gap guard")
        self.previous_time=timestamp
        stopped=abs(command_speed)<1e-7 and measured_speed<=POLICY["stopped_speed_m_s"]
        if self.stage=="braking":
            if timestamp-self.stage_started>POLICY["braking_timeout_s"]:raise TimeoutError("H3-B braking timeout")
            if stopped:self.stage="settling";self.stage_started=timestamp
        elif self.stage=="settling":
            if not stopped or not hold_active:self.stage_started=timestamp
            else:
                self.saw_hold=True
                if timestamp-self.stage_started>=POLICY["stationary_settle_s"]:
                    self.stage="alignment";self.stage_started=timestamp
            if timestamp-self.started>POLICY["braking_timeout_s"]+POLICY["alignment_timeout_s"]:
                raise TimeoutError("H3-B hold acknowledgement/settling timeout")
        elif self.stage=="alignment":
            if not hold_active or not stopped:raise RuntimeError("H3-B lost captured hold during steering alignment")
            if timestamp-self.stage_started>POLICY["alignment_timeout_s"]:raise TimeoutError("H3-B steering alignment timeout")
            error=max(abs(a-b) for a,b in zip(measured_steer,self.target_steer))
            if error<=math.radians(POLICY["alignment_tolerance_deg"]):
                if self.aligned_since is None:self.aligned_since=timestamp
                if timestamp-self.aligned_since>=POLICY["alignment_dwell_s"]:
                    self.stage="release_wait";self.stage_started=timestamp
            else:self.aligned_since=None
        elif self.stage=="release_wait":
            if not stopped:raise RuntimeError("H3-B motion occurred before hold release acknowledgement")
            if not hold_active and self.saw_hold:self.stage="ready"
            elif timestamp-self.stage_started>2.:raise TimeoutError("H3-B hold release acknowledgement timeout")
        elif self.stage!="ready":raise RuntimeError("Unknown H3-B direction gate stage")
        return {"stage":self.stage,"drive_hold_requested":self.stage in ("settling","alignment"),
            "ready":self.stage=="ready","direction":self.direction,"mode":self.mode,"curvature_1_m":self.curvature,
            "target_steer_rad":list(self.target_steer),"nonzero_speed_permitted":self.stage=="ready"}
