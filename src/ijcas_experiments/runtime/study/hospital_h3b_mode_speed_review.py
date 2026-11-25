"""Separate, rate-consistent forward component; never a whole-H3 certificate.

The old 0.22 m S probe and its measured failures remain immutable. This NEW
0.65 m moderate-curvature S is shared by all nine mode/speed cases. Changing
the profile as well as the mode means a future PASS cannot isolate a cause.
No caster state is assigned. Offline model admission is not physical success:
the inherited model has no explicit caster swivel/stiction state.
"""

import math

import numpy as np

from hospital_h3b_rolling_readiness import (POLICY, READINESS, ACCEPTANCE,
    BAD_CONTACTS, ROBOT_ROOT, WHEELS, GUARDS, classify_contact, relative_pose,
    wheel_crossing_checks, motion_stalled, direct_joint_targets)
from hospital_h3_flush_crossing import evaluate_passage
from hospital_path_execution_v3 import steering_geometry


VERSION = "H3B_existing_mode_speed_forward_component_v1"
ROLE = "MODERATE_S_MODE_SPEED_COMPONENT_NOT_OLD_S_OR_SAVED_ROUTE_REPLAY"
MODES = ("coordinated_counterphase", "front_only", "rear_only")
SPEEDS = (.010, .030, .050)
DISTANCE = .650
KNOTS = ((0., 0.), (.080, 0.), (.200, .250), (.320, 0.),
         (.440, -.250), (.560, 0.), (DISTANCE, 0.))
LEGS = ((1, DISTANCE),)
FIRST_CASE = "front_only_50mm_s"
COMMAND_LIMITS = {"forward_speed_m_s": .050, "reverse_speed_m_s": .010,
    "acceleration_m_s2": .020, "steering_command_slew_deg_s": 8.,
    "curvature_limit_1_m": .7}


def cases():
    return {f"{mode}_{round(speed*1000)}mm_s": (mode, speed)
            for mode in MODES for speed in SPEEDS}


def require_case(case):
    if case not in cases():
        raise ValueError("Select one of the nine predeclared mode/speed cases")
    return cases()[case]


def curvature_at(progress, phase):
    if not math.isfinite(progress):
        raise ValueError("Finite fixed-start progress required")
    return float(np.interp(progress, *zip(*KNOTS))) if phase.startswith("leg_0_") else 0.


def spatial_rate_bound(mode, knots=KNOTS):
    """Exact worst |d(delta)/dx| for each linear-curvature interval."""
    if mode not in MODES or len(knots) < 2:
        raise ValueError("Known mode and a finite increasing profile required")
    arm = .645 if mode == "coordinated_counterphase" else 1.29
    bounds = []
    for (x0, k0), (x1, k1) in zip(knots, knots[1:]):
        if not all(math.isfinite(v) for v in (x0, k0, x1, k1)) or x1 <= x0 or max(abs(k0), abs(k1)) > .7:
            raise ValueError("Finite increasing x and unchanged curvature bounds required")
        minimum_k = 0. if k0*k1 <= 0. else min(abs(k0), abs(k1))
        bounds.append(arm*abs((k1-k0)/(x1-x0))/(1.+(arm*minimum_k)**2))
    return max(bounds)


def nominal_component(mode, step=.001):
    """Ideal geometry with mode-specific body lateral velocity, NOT x=yaw arc."""
    if mode not in MODES or not math.isfinite(step) or not 0 < step <= .005:
        raise ValueError("Known mode and dense samples required")
    # At body longitudinal speed u, r/u=k and v/u=+/-0.645*k.
    lateral_arm = {"coordinated_counterphase": 0., "front_only": .645,
                   "rear_only": -.645}[mode]
    pose = np.zeros(3)
    result = [pose.tolist()]
    def derivative(x, state):
        k = float(np.interp(x, *zip(*KNOTS)))
        ratio = lateral_arm*k
        c, s = math.cos(state[2]), math.sin(state[2])
        denominator = c-ratio*s
        if denominator <= 0:
            raise ValueError("Component is not monotone in fixed longitudinal x")
        return np.array([1., (s+ratio*c)/denominator, k/denominator])
    for (left, _), (right, _) in zip(KNOTS, KNOTS[1:]):
        n = math.ceil((right-left)/step)
        dx = (right-left)/n
        for j in range(n):
            x = left+j*dx
            a = derivative(x, pose)
            b = derivative(x+dx/2, pose+dx*a/2)
            c = derivative(x+dx/2, pose+dx*b/2)
            d = derivative(x+dx, pose+dx*c)
            pose += dx*(a+2*b+2*c+d)/6
            result.append(pose.tolist())
    return result


def component_goal(start, mode):
    x, y, yaw = nominal_component(mode)[-1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return start[0]+c*x-s*y, start[1]+s*x+c*y, start[2]+yaw


class ForwardStopProfile:
    """Same ramp/brake law, selectable EXISTING forward cap, no reversal."""
    def __init__(self, direction, target, started, speed_limit):
        if direction != 1 or speed_limit not in SPEEDS or not all(math.isfinite(v) for v in (target, started)):
            raise ValueError("Predeclared forward case required")
        self.target, self.previous_time, self.speed_limit = target, started, speed_limit
        self.command = 0.
        self.braking = False

    def step(self, timestamp, progress, measured_speed):
        if not all(math.isfinite(v) for v in (timestamp, progress, measured_speed)) or measured_speed < 0:
            raise ValueError("Finite measured state required")
        dt = timestamp-self.previous_time
        if not 0 <= dt <= .25:
            raise RuntimeError("Existing 0.25 s control-gap guard")
        self.previous_time = timestamp
        if self.target-progress <= .005+self.command*self.command/(2*.020):
            self.braking = True
        target = 0. if self.braking else self.speed_limit
        self.command += max(-.020*dt, min(.020*dt, target-self.command))
        return self.command, self.braking and abs(self.command) < 1e-7 and measured_speed <= .005


def simulate(case, bounds, dt=.01):
    """Inherited identified response + Newton-Euler; NOT caster-stiction proof."""
    from patient_initial_stability.rigid_body_edge import MassNormalizedRigidBodyEdgeModel, RigidBodyParameterBounds
    from patient_initial_stability.steering_lattice import (MODE_CURVATURE_RESPONSE_SCALE,
        MODE_LATERAL_RESPONSE_SCALE, STEERING_TRACKING_BIAS_RAD)
    mode, speed = require_case(case)
    if not math.isfinite(dt) or not 0 < dt <= .02:
        raise ValueError("Fine fixed model integration required")
    model = MassNormalizedRigidBodyEdgeModel(RigidBodyParameterBounds(**bounds))
    profile = ForwardStopProfile(1, DISTANCE, 0., speed)
    pose = np.zeros(3)
    velocity = np.zeros(3)
    actual = np.zeros(2)
    commanded = (0., 0.)
    rows = [dict(time_s=0., x_m=0., y_m=0., yaw_rad=0., u_m_s=0., v_m_s=0., r_rad_s=0.,
                 command_speed_m_s=0., front_rad=0., rear_rad=0., command_front_rad=0., command_rear_rad=0.)]
    feasible = True
    max_friction = max_ltr = max_actuator = max_tracking = max_command_rate = 0.
    complete = False
    for index in range(1, math.ceil(GUARDS["maximum_simulation_duration_s"]/dt)+1):
        now = index*dt
        command_speed, complete = profile.step(now, pose[0], math.hypot(*velocity[:2]))
        if complete:
            break
        k = curvature_at(float(pose[0]), "leg_0_braking" if profile.braking else "leg_0_motion")
        old_command = commanded
        commanded, _ = direct_joint_targets(command_speed, mode, k, commanded, dt, actual)
        max_command_rate = max(max_command_rate, *(abs(a-b)/dt for a,b in zip(commanded, old_command)))
        target = np.array(commanded)+[STEERING_TRACKING_BIAS_RAD[a] for a in ("front", "rear")]
        middle = actual+np.clip(target-actual, -math.radians(8.)*dt/2, math.radians(8.)*dt/2)
        actual = middle+np.clip(target-middle, -math.radians(8.)*dt/2, math.radians(8.)*dt/2)
        tf, tr = np.tan(middle)
        achieved_k = (tf-tr)/1.29*MODE_CURVATURE_RESPONSE_SCALE[mode]
        ratio = (.645*tf+.645*tr)/1.29*MODE_LATERAL_RESPONSE_SCALE[mode]
        step = model.step(*velocity, command_speed, command_speed*ratio, command_speed*achieved_k, dt)
        new_velocity = np.array([step.longitudinal_velocity_m_s, step.lateral_velocity_m_s, step.yaw_rate_rad_s])
        u, v, r = (velocity+new_velocity)/2
        middle_yaw = pose[2]+r*dt/2
        pose += dt*np.array([u*math.cos(middle_yaw)-v*math.sin(middle_yaw),
                             u*math.sin(middle_yaw)+v*math.cos(middle_yaw), r])
        velocity = new_velocity
        feasible &= bool(step.feasible)
        max_friction = max(max_friction, step.friction_utilization)
        max_ltr = max(max_ltr, step.abs_ltr)
        max_actuator = max(max_actuator, step.actuator_utilization)
        max_tracking = max(max_tracking, *(abs(a-b) for a,b in zip(commanded, steering_geometry(mode, k))))
        rows.append(dict(time_s=now, x_m=float(pose[0]), y_m=float(pose[1]), yaw_rad=float(pose[2]),
            u_m_s=float(velocity[0]), v_m_s=float(velocity[1]), r_rad_s=float(velocity[2]), command_speed_m_s=command_speed,
            front_rad=float(actual[0]), rear_rad=float(actual[1]), command_front_rad=commanded[0], command_rear_rad=commanded[1]))
    goal = nominal_component(mode)[-1]
    error = math.hypot(pose[0]-goal[0], pose[1]-goal[1])
    checks = dict(model_completed_before_existing_timeout=complete,
        inherited_force_load_limits=feasible,
        same_30mm_lateral_corridor=max(abs(row["y_m"]) for row in rows) <= .030,
        same_2deg_heading_corridor=max(abs(row["yaw_rad"]) for row in rows) <= math.radians(2.),
        same_5mm_s_stop=math.hypot(*velocity[:2]) <= .005,
        actual_neutral_before_capture=max(abs(actual)) <= math.radians(2.5),
        model_terminal_within_30mm=error <= .030,
        commanded_slew_within_8deg_s=max_command_rate <= math.radians(8.)+1e-10,
        nominal_spatial_slew_compatible=spatial_rate_bound(mode)*speed <= math.radians(8.)+1e-10)
    return dict(case=case, mode=mode, speed_limit_m_s=speed, model_checks={k:bool(v) for k,v in checks.items()},
        model_motion_duration_s=rows[-1]["time_s"], final_local_xyyaw=pose.tolist(),
        model_terminal_error_m=error, maximum_command_tracking_error_deg=math.degrees(max_tracking),
        maximum_command_slew_deg_s=math.degrees(max_command_rate),
        maximum_friction_utilization=max_friction, maximum_abs_ltr=max_ltr, maximum_actuator_utilization=max_actuator,
        maximum_lateral_m=max(abs(row["y_m"]) for row in rows),
        maximum_heading_deg=math.degrees(max(abs(row["yaw_rad"]) for row in rows)),
        nominal_required_slew_deg_s=math.degrees(spatial_rate_bound(mode)*speed),
        model_has_no_explicit_caster_swivel_or_stiction_state=True,
        PhysX_executed=False, full_dynamics_plan_certified=False, full_H3B_turning_path_verified=False), rows


def evaluate_readiness(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count, gate_events, case):
    mode, speed = require_case(case)
    if any(not math.isfinite(float(r[k])) for r in rows for k in ('front_rad','rear_rad','command_front_rad',
            'command_rear_rad','command_time_s','requested_curvature_1_m','longitudinal_m')):
        raise ValueError('Finite measured and commanded steering/progress required')
    common = evaluate_passage(rows, goal, static_rows, mass_kg, bound, radii, counts, errors, hold_count)
    excluded = {"all_six_wheels_started_east_and_finished_west", "ordinary_positive_ramp_within_30mm_s", "passage_and_hold_completed"}
    checks = {k:v for k,v in common["checks"].items() if k not in excluded}
    active = ("front", "rear") if mode == "coordinated_counterphase" else (mode.split("_")[0],)
    moving = [r for r in rows if r["phase"].startswith("leg_0_") and r["command_speed_m_s"] > .003]
    slew = []
    for a,b in zip(rows,rows[1:]):
        dt = b["command_time_s"]-a["command_time_s"]
        if dt < 0:
            raise ValueError("Command clock went backwards")
        for axle in ("front", "rear"):
            delta = abs(b["command_"+axle+"_rad"]-a["command_"+axle+"_rad"])
            if dt == 0 and delta > 1e-12:
                raise ValueError("Steering command changed without elapsed time")
            slew.append(delta/dt if dt else 0.)
    checks.update(moderate_S_and_single_hold_completed=True,
        selected_mode_retained=all(r.get('command_mode')==mode for r in rows),
        unchanged_component_scope_no_cusps=not gate_events and all(not r["phase"].startswith(("leg_1_", "cusp_")) for r in rows),
        observed_active_axle_both_steering_signs=bool(moving) and all(min(r[a+"_rad"] for r in moving)<-math.radians(5.)
            and max(r[a+"_rad"] for r in moving)>math.radians(5.) for a in active),
        both_reference_curvature_signs=bool(moving) and min(r["requested_curvature_1_m"] for r in moving)<=-.24
            and max(r["requested_curvature_1_m"] for r in moving)>=.24,
        selected_forward_speed_cap=all(0<=r["command_speed_m_s"]<=speed+1e-9 for r in rows),
        commands_within_8deg_s=max(slew,default=0.)<=math.radians(8.)+1e-5,
        all_hold_commands_zero=all(abs(r["command_speed_m_s"])<1e-7 for r in rows if r["position_hold_enabled"]),
        neutral_before_drive_hold=all(max(abs(r["front_rad"]),abs(r["rear_rad"]),abs(r["command_front_rad"]),
            abs(r["command_rear_rad"]))<=math.radians(2.5) for r in rows if r["position_hold_enabled"]))
    return {"checks":{k:bool(v) for k,v in checks.items()}, "verdict":"PASS" if all(checks.values()) else "FAIL",
        **{k:common[k] for k in ("terminal_position_error_m", "terminal_heading_error_deg", "final_world_xyyaw",
            "fixed_terminal_hold_analysis", "final_force_weight_ratio")},
        "maximum_observed_command_slew_deg_s":math.degrees(max(slew,default=0.)),
        "mode_speed_component_verified":all(checks.values()), "old_S_probe_verified":False,
        "full_H3B_turning_path_verified":False, "full_dynamics_plan_certified":False}
