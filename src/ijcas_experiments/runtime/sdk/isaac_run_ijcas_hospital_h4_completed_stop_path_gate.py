"""ONE user-authorized H4 frozen-route development pilot, not final statistics.

Requires actual completed-Stop staging, replayed static PASS and PAUSED completed-static
body/twelve-joint/motor continuity. The old v3 tracking/terminal control and
180s/900s limits stay unchanged. Only task-root/contact/support bindings differ.
No implicit launch, repeated method, pose/caster reset, CICS or source save.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import asyncio
from collections import Counter, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import runpy
import time

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.types import ArticulationAction
from pxr import PhysicsSchemaTools, Usd, UsdUtils

if globals().get("ijcas_hospital_h4_path_authorized") is not True:
    raise RuntimeError("Explicit user authorization for ONE H4 route trial is required")
for _name, _task in tuple(globals().items()):
    if (_name.startswith("ijcas_hospital_") and _name.endswith("_task")
            and hasattr(_task, "done") and not _task.done()):
        raise RuntimeError("A hospital task is active; no replacement, retry or competing controller")
method = globals().get("ijcas_hospital_h4_method")
WORKSPACE = _ijcas_workspace()
HELPERS = _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts')
load = runpy.run_path(str(HELPERS/"hospital_private_loader.py"))["load"]
launch = load(HELPERS/"hospital_h4_completed_stop_path_launch.py")
c, stationary = launch["cohort"], launch["stationary"]
inputs = launch["load_actual_inputs"](method)
protocol, plan = inputs["protocol"], inputs["plan"]
preparation, static = inputs["preparation"], inputs["static"]
execution_helpers = vars(launch["execution"])
Controller = execution_helpers["H4TrackingController"]
selection = execution_helpers["method_settings"](method)
cohort = "cohort_01_completed_stop_continuation"
PLAN_PATH = next(Path(path) for path in inputs["bundle"]["plans"]
    if json.loads(Path(path).read_bytes())["method"] == method)
plan_raw = PLAN_PATH.read_bytes()
plan_sha = hashlib.sha256(plan_raw).hexdigest()
direct_joint_targets, relative_pose = execution_helpers["direct_joint_targets"], execution_helpers["relative_pose"]
hold_gains = load(HELPERS/"hospital_drive_hold.py")["drive_hold_gain_arrays"]
stop_analysis = load(HELPERS/"hospital_stop_analysis.py")["analyze_stop_trace"]
pose_collision = load(HELPERS/"hospital_geometry_tools.py")["pose_collision"]
static_path = stationary.RESULT
static_raw = static_path.read_bytes()
component_path = launch["BENCHMARK"]
component_raw = component_path.read_bytes()
manifest_path = Path(plan["geometry_manifest"])
manifest_raw = manifest_path.read_bytes()
manifest_sha, manifest = hashlib.sha256(manifest_raw).hexdigest(), json.loads(manifest_raw)
if (manifest_sha != plan["geometry_manifest_sha256"] or manifest["scenario"] != plan["scenario"]
        or manifest["start_xyyaw"] != plan["start_world_xyyaw"] or manifest["goal_xyyaw"] != plan["goal_world_xyyaw"]):
    raise RuntimeError("Exact frozen H4 task/goal/geometry required; no plan translation")
timeline, stage = omni.timeline.get_timeline_interface(), omni.usd.get_context().get_stage()
if stage is None or timeline.is_playing() or timeline.is_stopped():
    raise RuntimeError("The completed actual H4 static PASS must remain PAUSED")
setup = stage.GetPrimAtPath(c.MARKER)
if (not setup.IsValid() or setup.GetCustomDataByKey("status") != "static_support_passed"
        or setup.GetCustomDataByKey("robot_root") != c.ROBOT
        or setup.GetCustomDataByKey("static_gate_result") != str(static_path)
        or setup.GetCustomDataByKey("static_gate_sha256") != c.completion.sha(static_path)):
    raise RuntimeError("Exact unused completed H4 static marker/receipt required; old H3/H2 controllers prohibited")
environment = stationary.validate_environment(stage)
if environment != static["environment_after_static"]:
    raise RuntimeError("Live H4 physical configuration differs from completed static")
physics_sha = launch["physics_baseline_digest"](preparation, environment, static["runtime_motor_baseline"])
ROOT, BASE = c.ROBOT, c.ROBOT+"/base_link"
STEERING = ("base_link_to_front_active_steering", "base_link_to_rear_active_steering")
DRIVES = ("front_active_steering_to_drive_wheel", "rear_active_steering_to_drive_wheel")
WHEELS, BAD_CONTACTS = c.WHEELS, stationary.BAD_CONTACTS
supports = c.support_covers(c.support.load_design())
sim_view = SimulationManager.get_physics_sim_view()
if sim_view is None:
    raise RuntimeError("Retained H4 tensor view unavailable; never Play to obtain it")
tensor = sim_view.create_articulation_view(BASE)
if tensor.count != 1 or tensor.max_links != 13:
    raise RuntimeError("Exactly one completed-static H4 articulation with thirteen bodies required")

def numeric(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float).reshape(-1)

def live_static_check():
    names = list(tensor.shared_metatype.dof_names)
    gains = dict(kp=numeric(tensor.get_dof_stiffnesses()), kd=numeric(tensor.get_dof_dampings()),
        max_efforts=numeric(tensor.get_dof_max_forces()))
    return launch["validate_tensor"](static, numeric(tensor.get_root_transforms()), names,
        numeric(tensor.get_dof_positions()), numeric(tensor.get_root_velocities()), gains)

live_handoff = live_static_check()
with np.load(c.MAP) as data:
    occupied, map_origin, resolution = data["occupied"].copy(), data["origin_xy_m"].copy(), float(data["resolution_m"])
if (occupied.dtype != np.dtype(bool) or hashlib.sha256(occupied.tobytes()).hexdigest() != plan["planning_occupancy_digest"]
        or plan["planning_occupancy_digest"] != manifest["candidate_occupancy_digest"]
        or list(occupied.shape) != manifest["raster"]["shape_rows_columns"]
        or not np.array_equal(map_origin, manifest["raster"]["origin_xy_m"]) or resolution != manifest["raster"]["resolution_m"]):
    raise RuntimeError("Complete authored strict H4 raster differs from the frozen bundle")
OUTPUT = launch["OUTPUT"]
run_id = selection["run_prefix"]+"_r01"
summary_path, trace_path = (OUTPUT/(run_id+ext) for ext in (".json", ".csv"))
if launch["RESERVATION"].exists() or any(OUTPUT.glob("*.json")) or any(OUTPUT.glob("*.csv")):
    raise RuntimeError("This completed static state already reserved/ran one route; no automatic retry/second method")
OUTPUT.mkdir(parents=True, exist_ok=True)
with launch["RESERVATION"].open("x", encoding="utf-8") as stream:
    json.dump(dict(method=method, run_id=run_id, actual_static_sha256=c.completion.sha(static_path),
        protocol_sha256=c.completion.sha(launch["PROTOCOL"]), explicit_user_authorization=True,
        no_motion_yet=True, no_automatic_retry=True), stream, indent=2, allow_nan=False)

h4_clock_state = {"time_s": 0.}
counts, pairs, callback_errors = Counter(), Counter(), []


def _h4_path_step(dt):
    value = float(dt)
    if not math.isfinite(value) or value <= 0.:
        callback_errors.append("Invalid physics step duration")
    else:
        h4_clock_state["time_s"] += value


def _h4_path_contacts(headers, data):
    del data
    try:
        for header in headers:
            pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
            category = stationary.classify_contact(pair, environment["disabled_legacy_colliders"])
            if category not in stationary.KNOWN_CATEGORIES:
                raise RuntimeError("Unknown H4 contact classification: "+str(category))
            if category != "unrelated":
                counts[category] += 1
                pairs[(pair, category)] += 1
    except Exception as error:
        callback_errors.append(str(error))


def _h4_path_angles(quaternion):
    w, x, y, z = map(float, quaternion)
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        math.asin(max(-1., min(1., 2*(w*y-z*x)))), math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


async def _run_h4_path():
    source_hashes = {}
    for name in execution_helpers["COMMON_SOURCE_FILES"]:
        path = _ijcas_path(WORKSPACE, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts', name)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"schema_version": 1, "run_id": run_id, "scenario": plan["scenario"], "method": plan["method"], "verdict": "FAIL",
        "trial_kind": "H4_frozen_path_PhysX_development_ground_truth_feedback_NOT_sensor_navigation",
        "execution_version": execution_helpers["EXECUTION_VERSION"], "execution_role": "single_frozen_route_development_pilot",
        "execution_cohort": cohort, "plan_path": str(PLAN_PATH), "plan_sha256": plan_sha,
        "geometry_manifest_sha256": manifest_sha, "planning_occupancy_digest": plan["planning_occupancy_digest"],
        "physics_preparation_snapshot": preparation, "static_gate_reference": str(static_path),
        "static_gate_sha256": hashlib.sha256(static_raw).hexdigest(), "common_motion_stop_reference": str(component_path),
        "common_motion_stop_sha256": hashlib.sha256(component_raw).hexdigest(),
        "common_motion_stop_scope": "shared_H1_actuator_hold_component_benchmark_NOT_H4_moving_stop_or_path_validation",
        "start_world_xyyaw": plan["start_world_xyyaw"], "goal_world_xyyaw": plan["goal_world_xyyaw"],
        "planned_action_count": len(plan["selected_actions"]), "planned_nominal_distance_m": .05*len(plan["selected_actions"]),
        "planner_selected_modes": sorted({a["mode"] for a in plan["selected_actions"]}),
        "feedback_source": "Isaac_ground_truth_pose_and_joints_NOT_sensor_localization", "online_replanning": False,
        "Nav2_executed": False, "sensor_closed_loop_verified": False, "CICS_input_used": False,
        "selected_frozen_H4_route_PhysX_verified": False, "path_execution_started": False, "path_executed": False,
        "caster_state_reset_during_run": False, "body_pose_reset_during_run": False, "material_modified_during_run": False,
        "source_stage_saved": False, "use_for_final_statistics": False, "paired_random_seed_or_equal_caster_state_verified": False,
        "common_controller": execution_helpers["CONTROLLER_CONFIG"]["version"],
        "terminal_controller_config": dict(execution_helpers["CONTROLLER_CONFIG"]),
        "command_limits": dict(execution_helpers["COMMAND_LIMITS"]), "guards": dict(execution_helpers["GUARDS"]),
        "acceptance": dict(execution_helpers["ACCEPTANCE"]), "baseline_fixed_mode_scope": "planned_path_only_common_terminal_may_switch_modes",
        "source_sha256": source_hashes, "fixed_hold_attempt_count": 0,
        "path_protocol": str(launch["PROTOCOL"]), "path_protocol_sha256": c.completion.sha(launch["PROTOCOL"]),
        "input_sha256": {**protocol["input_sha256"], **static["input_sha256"],
            str(launch["PROTOCOL"]): c.completion.sha(launch["PROTOCOL"]),
            str(static_path): c.completion.sha(static_path), str(stationary.TRACE): c.completion.sha(stationary.TRACE),
            str(c.LIVE): c.completion.sha(c.LIVE)},
        "actual_static_replayed_PASS": True, "completed_static_live_handoff": live_handoff,
        "physics_baseline_sha256": physics_sha, "physical_condition": "nominal_265kg_canonical_friction_nominal_CoM",
        "shared_candidate_pool_sha256": plan["shared_candidate_pool_sha256"],
        "generator_selector_version": plan["generator_selector_version"],
        "source_authored_caster_seed_policy": plan["source_authored_caster_seed_policy"],
        "all_four_method_comparison_completed": False, "whole_H3_replay_or_robustness_verified": False,
        "baseline_launch_did_NOT_require_proposed_physical_PASS": True,
        "CICS_runtime_integration": False, "real_robot_validation_required": False}

    for path in (Path(__file__), HELPERS/"hospital_h4_completed_stop_path_launch.py", HELPERS/"hospital_h4_completed_stop_static_gate.py",
            HELPERS/"hospital_private_loader.py"):
        source_hashes[path.name] = c.completion.sha(path)
    controller = Controller(plan)
    rows, views = [], []
    robot = motor = drive_idx = old_kp = old_kd = old_targets = old_limits = None
    phase, previous_pose, previous_time = "pre_hold", None, None
    steer_command, hold_targets = (0., 0.), None
    history = deque()
    physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_h4_path_step)
    contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(_h4_path_contacts)
    app, wall_start, last_watch = omni.kit.app.get_app(), time.perf_counter(), -float("inf")
    try:
        robot = SingleArticulation(BASE)
        robot.initialize()
        indices = {name: i for i, name in enumerate(robot.dof_names)}
        if any(name not in indices for name in (*STEERING, *DRIVES)):
            raise RuntimeError("Incomplete active-joint mapping")
        steer_idx = np.array([indices[name] for name in STEERING], dtype=np.int32)
        drive_idx = np.array([indices[name] for name in DRIVES], dtype=np.int32)
        motor = robot.get_articulation_controller()
        kp, kd = motor.get_gains()
        old_kp, old_kd = np.asarray(kp).copy(), np.asarray(kd).copy()
        old_limits = np.asarray(motor.get_max_efforts()).copy()
        observed_settings = execution_helpers["motor_settings_record"](old_kp, old_kd, old_limits)
        observed_settings["joint_names"] = list(robot.dof_names)
        report["observed_original_motor_settings"] = observed_settings
        if observed_settings != static["runtime_motor_baseline"]:
            raise RuntimeError("Actual motor baseline differs from this completed static")
        if np.max(np.abs(old_kp[drive_idx])) > 1e-6:
            raise RuntimeError("Original velocity-only drive gains are not restored")
        old_targets = np.asarray(robot.get_applied_action().joint_positions)[drive_idx].copy()
        if not np.isfinite(old_targets).all():
            raise RuntimeError("Initial drive position targets are non-finite")
        for index, name in enumerate(WHEELS):
            view = RigidContactView(prim_paths_expr=ROOT+"/"+name, filter_paths_expr=[], name=run_id+"_"+str(index),
                prepare_contact_sensors=False, disable_stablization=False)
            view.initialize()
            views.append(view)
        sim = omni.physx.get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        encoded_base = PhysicsSchemaTools.sdfPathToInt(BASE)

        def measure():
            position, quaternion = robot.get_world_pose()
            q, qd = np.asarray(robot.get_joint_positions(), dtype=float), np.asarray(robot.get_joint_velocities(), dtype=float)
            linear = np.asarray(robot.get_linear_velocity(), dtype=float)
            if not all(np.isfinite(value).all() for value in (position, quaternion, q, qd, linear)):
                raise RuntimeError("Non-finite H4 physical state")
            roll, pitch, yaw = _h4_path_angles(quaternion)
            return (float(position[0]), float(position[1]), yaw), position, q, qd, linear, roll, pitch

        def verify_motor_settings(expected_kp):
            actual_kp, actual_kd = motor.get_gains()
            if not (np.allclose(actual_kp, expected_kp, rtol=1e-6, atol=1e-6) and np.array_equal(actual_kd, old_kd)
                    and np.array_equal(np.asarray(motor.get_max_efforts()), old_limits)):
                raise RuntimeError("Motor gain/damping/torque-limit guard failed")

        def set_hold(enabled):
            nonlocal hold_targets
            if enabled:
                hold_targets = np.asarray(robot.get_joint_positions(), dtype=float)[drive_idx].copy()
                if not np.isfinite(hold_targets).all():
                    raise RuntimeError("Non-finite measured drive hold targets")
                robot.apply_action(ArticulationAction(joint_positions=hold_targets.astype(np.float32),
                    joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_idx))
                new_kp, new_kd = hold_gains(old_kp, old_kd, drive_idx)
            else:
                new_kp, new_kd = old_kp.copy(), old_kd.copy()
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                motor.set_gains(kps=new_kp, kds=new_kd, save_to_usd=False)
            verify_motor_settings(new_kp)

        def command(steer, wheels, hold):
            robot.apply_action(ArticulationAction(joint_positions=np.asarray(steer, dtype=np.float32), joint_indices=steer_idx))
            robot.apply_action(ArticulationAction(joint_positions=hold_targets.astype(np.float32) if hold else None,
                joint_velocities=np.asarray(wheels, dtype=np.float32), joint_indices=drive_idx))

        def guard(pose, linear, roll, pitch):
            if callback_errors or any(counts[key] for key in BAD_CONTACTS):
                raise RuntimeError("Unexpected obstacle/support contact or contact-monitor error")
            if max(abs(roll), abs(pitch)) > math.radians(2.) or math.hypot(linear[0], linear[1]) > .080:
                raise RuntimeError("Body tilt/speed safety guard")
            last = previous_pose if previous_pose is not None else pose
            turn = execution_helpers["wrap"](pose[2]-last[2])
            steps = max(1, math.ceil(math.hypot(pose[0]-last[0], pose[1]-last[1])/.005), math.ceil(abs(turn)/math.radians(.25)))
            for index in range(1, steps+1):
                fraction = index/steps
                check = (last[0]+fraction*(pose[0]-last[0]), last[1]+fraction*(pose[1]-last[1]), last[2]+fraction*turn)
                if pose_collision(occupied, check, map_origin, resolution, plan["full_envelope_clearance_m"]):
                    raise RuntimeError("Actual swept full-envelope raster guard")
                if c.support_area_missing(check, supports) > 1e-9:
                    raise RuntimeError("Actual swept full-body authored support-union guard")
            if controller.phase != "path_tracking" and math.hypot(pose[0]-controller.goal[0], pose[1]-controller.goal[1]) > .12:
                raise RuntimeError("Terminal controller left the 120 mm goal neighbourhood")

        async def sample(steer, wheels, hold, info, phase_start):
            nonlocal previous_pose, previous_time, last_watch
            command(steer, wheels, hold)
            while True:
                sim.wake_up(stage_id, encoded_base)
                await app.next_update_async()
                if not timeline.is_playing():
                    raise RuntimeError("Timeline controls changed during H4 execution")
                if time.perf_counter()-wall_start > 900.:
                    raise TimeoutError("H4 pilot exceeded 900 wall-clock seconds")
                now = h4_clock_state["time_s"]
                if previous_time is None or now > previous_time:
                    break
            if now > 180. or (previous_time is not None and now-previous_time > .25):
                raise TimeoutError("H4 simulated-duration/control-gap guard")
            pose, position, q, qd, linear, roll, pitch = measure()
            guard(pose, linear, roll, pitch)
            verify_motor_settings(hold_gains(old_kp, old_kd, drive_idx)[0] if hold else old_kp)
            applied = robot.get_applied_action()
            velocities, positions = np.asarray(applied.joint_velocities)[drive_idx], np.asarray(applied.joint_positions)
            if (not np.allclose(velocities, wheels, rtol=0, atol=1e-6)
                    or not np.allclose(positions[steer_idx], steer, rtol=0, atol=1e-6)
                    or (hold and not np.allclose(positions[drive_idx], hold_targets, rtol=0, atol=1e-6))):
                raise RuntimeError("Actual steering/drive targets differ from commands")
            history.append((now, pose))
            while len(history) > 2 and history[1][0] <= now-.20:
                history.popleft()
            dt = now-history[0][0]
            speed = math.hypot(pose[0]-history[0][1][0], pose[1]-history[0][1][1])/dt if dt > 0 else math.hypot(linear[0], linear[1])
            row = {"time_s": now, "phase": phase, "phase_time_s": now-phase_start, "x_m": pose[0], "y_m": pose[1],
                "z_m": float(position[2]), "yaw_rad": pose[2], "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
                "vx_physx_m_s": float(linear[0]), "vy_physx_m_s": float(linear[1]), "vz_physx_m_s": float(linear[2]),
                "pose_filtered_speed_m_s": speed, "command_speed_m_s": info.get("speed_m_s", 0.),
                "command_mode": info.get("mode", "pre_hold"), "command_curvature_1_m": info.get("curvature_1_m", 0.),
                "action_index": info.get("action_index", -1), "path_progress_m": info.get("progress_m", 0.),
                "path_error_m": info.get("path_error_m", 0.), "path_heading_error_deg": info.get("path_heading_error_deg", 0.),
                "goal_position_error_m": math.hypot(pose[0]-controller.goal[0], pose[1]-controller.goal[1]),
                "goal_heading_error_deg": math.degrees(execution_helpers["wrap"](controller.goal[2]-pose[2])),
                "front_rad": float(q[steer_idx[0]]), "rear_rad": float(q[steer_idx[1]]),
                "terminal_stage": info.get("terminal_stage", "pre_hold"), "terminal_pulse_count": info.get("pulse_count", 0),
                "target_front_rad": float(steer[0]), "target_rear_rad": float(steer[1]), "position_hold_enabled": hold}
            for index, axle in enumerate(("front", "rear")):
                row["drive_"+axle+"_rad"] = float(q[drive_idx[index]])
                row["drive_"+axle+"_velocity_rad_s"] = float(qd[drive_idx[index]])
                row["drive_"+axle+"_actual_velocity_target_rad_s"] = float(velocities[index])
            for name, index in indices.items():
                row["joint_"+name+"_rad"] = float(q[index])
            for index, view in enumerate(views):
                value = float(np.asarray(view.get_net_contact_forces(dt=1./120.)).reshape(-1, 3)[0, 2])
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite measured wheel support force")
                row[f"wheel_{index}_fz_n"] = value
            rows.append(row)
            previous_pose, previous_time = pose, now
            if time.perf_counter()-last_watch >= 15.:
                print(f"[H4 WATCH {run_id}] wall={time.perf_counter()-wall_start:.1f}s sim={now:.1f}s "
                    f"phase={phase}/{row['terminal_stage']}; goal={1000*row['goal_position_error_m']:.2f}mm/"
                    f"{row['goal_heading_error_deg']:.3f}deg; command/observed={1000*row['command_speed_m_s']:.2f}/{1000*speed:.2f}mm/s")
                last_watch = time.perf_counter()
            return row, pose, q, linear

        pose, _, q, _, linear, roll, pitch = measure()
        offset = relative_pose(pose, plan["start_world_xyyaw"])
        if math.hypot(offset[0], offset[1]) > .010 or abs(offset[2]) > math.radians(.5):
            raise RuntimeError("Robot is outside 10 mm / 0.5 deg of ORIGINAL H4 start; never translate the plan")
        if max(abs(float(q[i])) for i in steer_idx) > math.radians(1.):
            raise RuntimeError("Actual release steering differs from the nominal zero-angle plan")
        guard(pose, linear, roll, pitch)
        report["observed_prehold_world_xyyaw"] = list(pose)
        if stationary.validate_environment(stage) != environment:
            raise RuntimeError("Physical configuration changed before Play")
        live_static_check()
        set_hold(True)
        command((0., 0.), (0., 0.), True)
        timeline.play()
        pre_start = h4_clock_state["time_s"]
        print("H4 path phase: 2 s measured-wheel pre-hold at ORIGINAL task start")
        while h4_clock_state["time_s"]-pre_start < 2.:
            row, pose, q, linear = await sample((0., 0.), (0., 0.), True, {}, pre_start)
        offset = relative_pose(pose, plan["start_world_xyyaw"])
        if (math.hypot(offset[0], offset[1]) > .010 or abs(offset[2]) > math.radians(.5)
                or row["pose_filtered_speed_m_s"] > .005 or math.hypot(linear[0], linear[1]) > .005):
            raise RuntimeError("H4 pre-hold release pose/speed is outside fixed plan tolerance")
        report.update(release_world_xyyaw=list(pose), release_offset_body_xyyaw=list(offset),
            release_physx_speed_m_s=math.hypot(linear[0], linear[1]))
        set_hold(False)
        report["path_execution_started"] = True
        motion_start = command_time = h4_clock_state["time_s"]
        controller.previous_time = motion_start
        motion_hold, printed_phase = False, None
        while True:
            measured_speed = max(math.hypot(linear[0], linear[1]), row["pose_filtered_speed_m_s"])
            info = controller.step(h4_clock_state["time_s"], pose, measured_speed, tuple(float(q[i]) for i in steer_idx))
            phase = info["phase"]
            label = (phase, info["terminal_stage"])
            if label != printed_phase:
                print("H4 path phase:", phase, "/", info["terminal_stage"])
                printed_phase = label
            if info["hold_ready"]:
                break
            requested_hold = bool(info["drive_hold_requested"])
            if requested_hold and (abs(info["speed_m_s"]) > 1e-7 or (not motion_hold and measured_speed > .005)):
                raise RuntimeError("Active-wheel hold requested before commanded/measured stop")
            if requested_hold != motion_hold:
                set_hold(requested_hold)
                motion_hold = requested_hold
            dt = h4_clock_state["time_s"]-command_time
            steer_command, wheels = direct_joint_targets(info["speed_m_s"], info["mode"], info["curvature_1_m"],
                steer_command, dt, tuple(float(q[i]) for i in steer_idx))
            command_time = h4_clock_state["time_s"]
            row, pose, q, linear = await sample(steer_command, wheels, motion_hold, info, motion_start)
        entry = controller.hold_entry_candidate or {}
        if (not entry or entry["position_error_m"] > .030 or abs(entry["heading_error_deg"]) > 2.
                or measured_speed > .005 or not motion_hold or controller.terminal_stage != "final_hold_ready"):
            raise RuntimeError("H4 final hold entry is outside unchanged stopped acceptance")
        report.update(fixed_hold_entry_candidate=entry, fixed_hold_attempt_count=1,
            terminal_hold_capture_physx_speed_m_s=math.hypot(linear[0], linear[1]))
        set_hold(True)
        phase, hold_start = "terminal_hold", h4_clock_state["time_s"]
        print("H4 path phase: ONE 5 s terminal hold; fixed final 3 s evaluation, not a PASS yet")
        while h4_clock_state["time_s"]-hold_start < 5.:
            row, pose, q, linear = await sample(steer_command, (0., 0.), True, info, hold_start)
        tail = [r for r in rows if r["phase"] == "terminal_hold" and r["phase_time_s"] >= 2.]
        analysis = stop_analysis(tail)
        position_error = math.hypot(pose[0]-controller.goal[0], pose[1]-controller.goal[1])
        heading_error = math.degrees(execution_helpers["wrap"](pose[2]-controller.goal[2]))
        balance = float(np.mean([sum(r[f"wheel_{i}_fz_n"] for i in range(6)) for r in tail])/(preparation["simulation_mass_kg"]*9.81))
        checks = {"path_execution_and_hold_completed": True,
            "at_least_95_percent_frozen_path_progress": controller.reference.progress >= .95*controller.reference.arc[-1],
            "terminal_position_error_under_30mm": position_error <= .030,
            "terminal_heading_error_under_2deg": abs(heading_error) <= 2.,
            "fixed_tail_complete_3s": analysis["duration_s"] >= 2.9,
            "fixed_tail_position_under_30mm": all(r["goal_position_error_m"] <= .030 for r in tail),
            "fixed_tail_heading_under_2deg": all(abs(r["goal_heading_error_deg"]) <= 2. for r in tail),
            "fixed_tail_pose_standstill": analysis["development_standstill_check"] == "PASS",
            "fixed_tail_physx_speed_under_5mm_s": all(math.hypot(r["vx_physx_m_s"], r["vy_physx_m_s"]) <= .005 for r in tail),
            "fixed_tail_filtered_speed_under_5mm_s": all(r["pose_filtered_speed_m_s"] <= .005 for r in tail),
            "all_six_wheels_mean_load_over_1N": all(np.mean([r[f"wheel_{i}_fz_n"] for r in tail]) > 1. for i in range(6)),
            "vertical_force_weight_balance": .8 <= balance <= 1.2,
            "only_expected_external_support": counts["wheel_support"] > 0 and not any(counts[k] for k in BAD_CONTACTS),
            "contact_callback_monitor_healthy": not callback_errors,
            "plan_file_unchanged": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == plan_sha,
            "single_fixed_hold_attempt_no_best_window_selection": report["fixed_hold_attempt_count"] == 1}
        report.update(checks={key: bool(value) for key, value in checks.items()}, verdict="PASS" if all(checks.values()) else "FAIL",
            path_executed=True, terminal_position_error_m=position_error, terminal_heading_error_deg=heading_error,
            terminal_hold_analysis=analysis, terminal_vertical_force_weight_ratio=balance, final_world_xyyaw=list(pose),
            final_physx_speed_m_s=math.hypot(linear[0], linear[1]), motion_and_terminal_correction_time_s=hold_start-motion_start)
    except (Exception, asyncio.CancelledError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report.setdefault("checks", {})["path_execution_and_hold_completed"] = False
        report["verdict"] = "FAIL"
    finally:
        if robot is not None and drive_idx is not None:
            try:
                robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_idx))
            except Exception as error:
                report["zero_command_error"], report["verdict"] = str(error), "FAIL"
                report.setdefault("checks", {})["final_zero_command_sent"] = False
        if timeline.is_playing():
            timeline.pause()
        if motor is not None and old_kp is not None:
            try:
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    motor.set_gains(kps=old_kp, kds=old_kd, save_to_usd=False)
                restored_kp, restored_kd = motor.get_gains()
                if not (np.allclose(restored_kp, old_kp, rtol=1e-6, atol=1e-6)
                        and np.array_equal(restored_kd, old_kd)
                        and np.array_equal(np.asarray(motor.get_max_efforts()), old_limits)):
                    raise RuntimeError("Original motor gain/damping/torque-limit restoration failed")
                if old_targets is not None:
                    robot.apply_action(ArticulationAction(joint_positions=old_targets.astype(np.float32),
                        joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_idx))
                report["original_motor_gains_restored"] = True
            except Exception as error:
                report["restore_error"], report["verdict"] = str(error), "FAIL"
                report["original_motor_gains_restored"] = False
        report.setdefault("checks", {})["original_motor_gains_restored"] = report.get("original_motor_gains_restored", False)
        if callback_errors or any(counts[k] for k in BAD_CONTACTS):
            report["checks"]["only_expected_external_support"] = False
        try:
            if timeline.is_stopped():
                raise RuntimeError("Timeline was stopped; actual physical continuation was lost")
            report["checks"]["physical_environment_unchanged"] = stationary.validate_environment(stage) == environment
            c.verify(report["input_sha256"])
            report["checks"]["all_frozen_input_evidence_unchanged"] = True
            final_pose, _, final_q, final_qd, _, _, _ = measure()
            report["final_actual_world_xyyaw"] = list(final_pose)
            report["final_actual_joint_names"] = list(robot.dof_names)
            report["final_actual_twelve_joint_positions_rad"] = final_q.tolist()
            report["final_actual_twelve_joint_velocities_rad_s"] = final_qd.tolist()
        except Exception as error:
            report["checks"]["final_actual_state_and_input_integrity_verified"] = False
            report["final_readback_error"] = str(error)
        if not all(report["checks"].values()):
            report["verdict"] = "FAIL"
        physics_sub = contact_sub = None
        report.update(selected_frozen_H4_route_PhysX_verified=report["verdict"] == "PASS", sample_count=len(rows),
            terminal_pulse_count=controller.pulse_count, terminal_handoff=controller.handoff,
            terminal_execution_mode_switches=controller.terminal_switches, path_progress_m=controller.reference.progress,
            reference_polyline_length_m=controller.reference.arc[-1], contact_report_counts_by_category=dict(counts),
            contact_monitor_errors=callback_errors, wall_time_s=time.perf_counter()-wall_start,
            contact_pairs_by_category=[{"pair": pair, "category": category, "callback_count": count}
                for (pair, category), count in sorted(pairs.items())])
        report["measured_path_length_m"] = sum(math.hypot(b["x_m"]-a["x_m"], b["y_m"]-a["y_m"])
            for a, b in zip(rows, rows[1:]) if a["phase"] != "pre_hold" and b["phase"] != "pre_hold")
        if rows:
            report["last_recorded_state"] = {key: rows[-1][key] for key in ("time_s", "phase", "terminal_stage", "x_m", "y_m",
                "yaw_rad", "goal_position_error_m", "goal_heading_error_deg", "command_speed_m_s")}
            with trace_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        if rows:
            report["trace_sha256"] = c.completion.sha(trace_path)
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            setup.SetCustomDataByKey("last_h4_path_gate", str(summary_path))
            setup.SetCustomDataByKey("status", "path_gate_finished_fresh_start_required")
        print("=== IJCAS HOSPITAL H4 FROZEN PATH EXECUTION GATE ===")
        print("run/method:", run_id, "/", plan["method"])
        print("contact categories:", dict(counts))
        print("path progress/reference polyline m:", round(controller.reference.progress, 4), round(controller.reference.arc[-1], 4))
        if "terminal_position_error_m" in report:
            print("terminal error mm/deg:", round(1000*report["terminal_position_error_m"], 3), round(report["terminal_heading_error_deg"], 6))
            print("hold-tail regression speed mm/s:", round(1000*report["terminal_hold_analysis"]["pose_regression_speed_m_s"], 3))
        print("original motor gains restored:", report.get("original_motor_gains_restored", False))
        print("failed checks:", [key for key, value in report["checks"].items() if not value])
        if report.get("error"):
            print("error:", report["error"])
        print("VERDICT:", report["verdict"])
        print("saved:", summary_path)
        print("Timeline PAUSED; see gain-restoration status above. Ground-truth development trial, NOT sensor/Nav2 or final statistics.")
        print("Do not save the source USD or press Play again. Fresh source/start/static PASS required for another run.")


ijcas_hospital_h4_path_task = asyncio.ensure_future(_run_h4_path())
print("H4 frozen v3 path pilot scheduled:", run_id, "/", plan["method"])
print("Automatic 2 s pre-hold -> ordinary ramp/path tracking -> bounded v3 terminal closure -> 5 s hold -> Pause/restore gains.")
print("15-second read-only heartbeat included. Do not touch timeline controls. No CICS input, sensor localization or Nav2.")

