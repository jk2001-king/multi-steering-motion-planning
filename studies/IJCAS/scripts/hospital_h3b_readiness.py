"""Short straight signed-motion readiness, NOT a doorway/turning path verdict.

Reuse unchanged flat-support/final-hold checks. Door passage and forward-only
checks do not apply to this separate outside-door reversal component. Nonzero
steering alignment and a full planned reversal path remain unverified here.
"""

import math
from pathlib import Path
import runpy

from hospital_h3b_policy import POLICY,StopAlignGate


_old=runpy.run_path(str(Path(__file__).with_name("hospital_h3_flush_crossing.py")))
for _name in ("ROBOT_ROOT","WHEELS","READINESS","ACCEPTANCE","BAD_CONTACTS","classify_contact","relative_pose","wheel_crossing_checks"):
    globals()[_name]=_old[_name]
GUARDS={**_old["GUARDS"],"maximum_simulation_duration_s":90.,"maximum_wall_time_s":300.}
COMMAND_LIMITS={"forward_speed_m_s":.010,"reverse_speed_m_s":.010,"acceleration_m_s2":.020,"steering_target_rad":0.}
LEGS=((1,.100),(-1,.000),(1,.050))


class SignedStopProfile:
    def __init__(self,direction,target,started):
        if direction not in (-1,1) or not all(math.isfinite(x) for x in (target,started)):raise ValueError("Invalid readiness leg")
        self.direction,self.target,self.previous_time=direction,target,started
        self.command=0.;self.braking=False

    def step(self,timestamp,progress,measured_speed):
        if not all(math.isfinite(v) for v in (timestamp,progress,measured_speed)) or measured_speed<0:raise ValueError("Invalid measured readiness state")
        dt=timestamp-self.previous_time
        if not 0<=dt<=.25:raise RuntimeError("H3-B readiness control-gap guard")
        self.previous_time=timestamp
        remaining=self.direction*(self.target-progress)
        if remaining<=.005+self.command*self.command/(2*.020):self.braking=True
        target=0. if self.braking else self.direction*.010
        self.command+=max(-.020*dt,min(.020*dt,target-self.command))
        ready=self.braking and abs(self.command)<1e-7 and measured_speed<=.005
        return self.command,ready


def motion_stalled(rows):
    if not rows or not rows[-1]["phase"].endswith("_motion"):return False
    last=rows[-1];phase=last["phase"];tail=[r for r in rows if r["phase"]==phase and r["time_s"]>=last["time_s"]-5.]
    return (len(tail)>1 and tail[-1]["time_s"]-tail[0]["time_s"]>=4.95
        and all(abs(r["command_speed_m_s"])>=.006 for r in tail)
        and max(r["longitudinal_m"] for r in tail)-min(r["longitudinal_m"] for r in tail)<.001)


def evaluate_readiness(rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count,gate_events):
    # Pure shared support/hold mathematics, not reuse/relabel of a measured result.
    base=_old["evaluate_passage"](rows,goal,static_rows,mass_kg,bound,radii,counts,errors,hold_count)
    excluded={"all_six_wheels_started_east_and_finished_west","ordinary_positive_ramp_within_30mm_s","passage_and_hold_completed"}
    checks={k:v for k,v in base["checks"].items() if k not in excluded}
    motion=[]
    for index,(direction,target) in enumerate(LEGS):
        segment=[r for r in rows if r["phase"].startswith(f"leg_{index}_")]
        travelled=direction*(segment[-1]["longitudinal_m"]-segment[0]["longitudinal_m"]) if len(segment)>1 else 0.
        motion.append({"leg_index":index,"direction":direction,"signed_travel_m":travelled,
            "last_longitudinal_m":segment[-1]["longitudinal_m"] if segment else None,"fixed_target_m":target})
    checks.update({"readiness_and_one_hold_completed":True,"three_measured_motion_legs":all(m["signed_travel_m"]>=.030 for m in motion),
        "both_direction_gates_measured":len(gate_events)==2 and [g["direction"] for g in gate_events]==[-1,1],
        "each_gate_measured_stop_hold_alignment_and_release":len(gate_events)==2 and all(g["captured_hold_acknowledged"] and g["hold_release_acknowledged"]
            and abs(g["command_speed_m_s"])<1e-7 and g["measured_speed_m_s"]<=.005 and g["steering_error_deg"]<=2.5 for g in gate_events),
        "all_signed_commands_within_10mm_s":all(abs(r["command_speed_m_s"])<=.010+1e-9 for r in rows),
        "held_commands_are_zero":all(abs(r["command_speed_m_s"])<1e-7 for r in rows if r["position_hold_enabled"]),
        "all_cusp_samples_have_zero_command":all(abs(r["command_speed_m_s"])<1e-7 for r in rows if r["phase"].startswith("cusp_"))})
    return {"checks":{k:bool(v) for k,v in checks.items()},"verdict":"PASS" if all(checks.values()) else "FAIL",
        **{key:base[key] for key in ("terminal_position_error_m","terminal_heading_error_deg","final_world_xyyaw","fixed_terminal_hold_analysis","final_force_weight_ratio")},
        "measured_signed_motion_legs":motion,"nonzero_steering_alignment_physically_verified":False,
        "full_H3B_turning_path_verified":False,"full_dynamics_plan_certified":False}
