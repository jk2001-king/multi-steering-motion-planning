"""One selected frozen H1 path PhysX pilot, not a ROS/Nav2 or paper trial.

Run PAUSED immediately after a fresh H1 static PASS at the original start.
Automatic measured-wheel pre-hold -> ordinary ramp/path feedback -> bounded
terminal closure -> five-second hold -> Pause and restore original gains.
Only active-drive hold gains are temporarily changed. Never reset a body,
caster joint or material during the run. Existing results are never replaced.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

# Check before replacing shared Script Editor globals used by a live trial.
for _h1guard_name in ("ijcas_hospital_h1_static_task", "ijcas_hospital_h1_initial_task", "ijcas_hospital_h1_zero_task",
        "ijcas_hospital_h1_motion_stop_task", "ijcas_hospital_h1_path_task"):
    _h1guard_task = globals().get(_h1guard_name)
    if _h1guard_task is not None and not _h1guard_task.done():
        raise RuntimeError("Another hospital task is running; do not replace its Script Editor globals")

import asyncio
from collections import Counter, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import time

import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from isaacsim.core.api.sensors import RigidContactView
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import PhysicsSchemaTools, Usd, UsdPhysics, UsdUtils


WORKSPACE = _ijcas_workspace()
RESULTS = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
HELPERS = _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts')
execution_version = globals().pop("ijcas_hospital_h1_controller_version", "legacy_v1")
if execution_version not in ("legacy_v1", "stop_align_v2", "stop_align_v3"):
    raise ValueError("Unknown H1 execution controller version")
staged_terminal = execution_version != "legacy_v1"
readiness_only = globals().pop("ijcas_hospital_h1_readiness_only", False)
if not isinstance(readiness_only, bool):
    raise ValueError("readiness_only must be an explicit Python True/False")
execution_cohort = (globals().pop("ijcas_hospital_h1_execution_cohort", "cohort_01")
    if execution_version == "stop_align_v3" else globals().pop("ijcas_hospital_h1_v2_cohort", "cohort_01"))
if not isinstance(execution_cohort, str) or re.fullmatch(r"cohort_[0-9]{2,}", execution_cohort) is None:
    raise ValueError("Use an explicit cohort_NN execution cohort name")
execution_file = {"legacy_v1": "hospital_path_execution.py", "stop_align_v2": "hospital_path_execution_v2.py",
    "stop_align_v3": "hospital_path_execution_v3.py"}[execution_version]
execution_helpers = runpy.run_path(str(HELPERS / execution_file))
method_helpers = runpy.run_path(str(HELPERS / "hospital_path_methods.py"))
# One-shot selection avoids an old B0/B1/B2 editor global silently changing
# a later legacy proposed-only invocation. Explicitly set for each new run.
selection = method_helpers["method_settings"](globals().pop("ijcas_hospital_h1_method", "P_dynamics_full_envelope"))
Controller = execution_helpers["H1TrackingController"]
direct_joint_targets = execution_helpers["direct_joint_targets"]
relative_pose = execution_helpers["relative_pose"]
hold_gains = runpy.run_path(str(HELPERS / "hospital_drive_hold.py"))["drive_hold_gain_arrays"]
stop_analysis = runpy.run_path(str(HELPERS / "hospital_stop_analysis.py"))["analyze_stop_trace"]
classify_contact = runpy.run_path(str(HELPERS / "hospital_contact_rules.py"))["classify_contact"]
pose_collision = runpy.run_path(str(HELPERS / "hospital_geometry_tools.py"))["pose_collision"]
ROOT = "/World/patient_transport_lab_screening"
BASE = ROOT+"/base_link"
STEERING = ("base_link_to_front_active_steering", "base_link_to_rear_active_steering")
DRIVES = ("front_active_steering_to_drive_wheel", "rear_active_steering_to_drive_wheel")
WHEELS = ("front_active_drive_wheel_link", "rear_active_drive_wheel_link", "front_left_caster_wheel_link",
    "front_right_caster_wheel_link", "rear_left_caster_wheel_link", "rear_right_caster_wheel_link")
BAD_CONTACTS = ("hospital_obstacle", "unexpected_legacy_support", "nonwheel_support", "unclassified_external")
timeline = omni.timeline.get_timeline_interface()
if timeline.is_playing() or timeline.is_stopped():
    raise RuntimeError("Run the fresh H1 static gate first; leave the resulting timeline PAUSED")
for name in ("ijcas_hospital_h1_static_task", "ijcas_hospital_h1_initial_task", "ijcas_hospital_h1_zero_task",
        "ijcas_hospital_h1_motion_stop_task", "ijcas_hospital_h1_path_task"):
    task = globals().get(name)
    if task is not None and not task.done():
        raise RuntimeError("Another hospital task is running")
directory_value = globals().get("ijcas_hospital_h1_planning_directory")
if not isinstance(directory_value, str) or not directory_value:
    raise RuntimeError("Set ijcas_hospital_h1_planning_directory to the audited offline plan folder")
DIRECTORY = Path(directory_value).resolve()
PLAN_PATH = DIRECTORY / (selection["method"]+".json")
plan = json.loads(PLAN_PATH.read_bytes())
execution_helpers["validate_plan"](plan)
plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
if plan.get("method") != selection["method"] or plan.get("planner_dynamics") != selection["planner_dynamics"]:
    raise RuntimeError("Selected method and stored plan disagree")
if selection["method"] == "P_dynamics_full_envelope" and not plan.get("initial_steering_propagated"):
    raise RuntimeError("The proposed plan must propagate its initial steering state")
audits = [json.loads(path.read_bytes()) for path in sorted((DIRECTORY / "offline_audits").glob("audit_r*.json"))]
if not any(a.get("verdict") == "PASS" and a.get("plan_sha256") == plan_sha
        and a.get("method", "P_dynamics_full_envelope") == selection["method"] and a.get("checks")
        and all(a["checks"].values()) for a in audits):
    raise RuntimeError("An independent offline audit PASS for this exact plan is required")
manifest_path = Path(plan["geometry_manifest"])
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
manifest = json.loads(manifest_path.read_bytes())
if manifest_sha != plan["geometry_manifest_sha256"]:
    raise RuntimeError("Plan geometry is stale")
if hashlib.sha256(Path(manifest["source_audit"]).read_bytes()).hexdigest() != manifest["source_sha256"]:
    raise RuntimeError("Hospital source audit changed")
if not (np.allclose(plan["start_world_xyyaw"], manifest["start_xyyaw"], atol=1e-10, rtol=0)
        and np.allclose(plan["goal_world_xyyaw"], manifest["goal_xyyaw"], atol=1e-10, rtol=0)):
    raise RuntimeError("Plan start/goal do not match the original hospital task")
stage = omni.usd.get_context().get_stage()
if stage is None or Path(stage.GetRootLayer().realPath).resolve() != Path(manifest["source_stage"]).resolve():
    raise RuntimeError("Open the audited hospital source stage")
setup = stage.GetPrimAtPath("/World/IJCAS_HospitalSetupPreview")
if not setup.IsValid() or setup.GetCustomDataByKey("setup_status") != "stationary_gate_passed":
    raise RuntimeError("Prepare the original H1 pose/physics and run a fresh stationary support PASS")
static_path = Path(setup.GetCustomDataByKey("static_gate_result"))
static = json.loads(static_path.read_bytes())
if (static.get("verdict") != "PASS" or static.get("schema_version") != 2
        or static["geometry_manifest_sha256"] != manifest_sha
        or setup.GetCustomDataByKey("physics_manifest_sha256") != manifest_sha):
    raise RuntimeError("The stationary support evidence is missing/stale")
preparation = static["physics_preparation_snapshot"]
component_path = RESULTS / "motion_stop/hospital_h1_ramp_stop_hold_r01.json"
component = json.loads(component_path.read_bytes())
if (component.get("verdict") != "PASS" or not component.get("original_motor_gains_restored")
        or component["geometry_manifest_sha256"] != manifest_sha):
    raise RuntimeError("The preceding common motion/stop component PASS is required")
if any(prim.GetTypeName() == "OmniGraph" for prim in stage.Traverse()):
    raise RuntimeError("Disable competing OmniGraphs before direct motor control")
for path in preparation["disabled_legacy_floor_colliders"]:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI):
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            raise RuntimeError("A previously disabled legacy floor collider is active: "+path)
support = stage.GetPrimAtPath(preparation["support_floor"])
if not support.IsValid() or UsdPhysics.CollisionAPI(support).GetCollisionEnabledAttr().Get() is False:
    raise RuntimeError("Canonical support floor is not enabled")
with np.load(DIRECTORY / "planning_occupancy.npz") as archive:
    occupied, map_origin, resolution = archive["occupied"].copy(), archive["origin_xy_m"].copy(), float(archive["resolution_m"])
if (hashlib.sha256(occupied.tobytes()).hexdigest() != plan["planning_occupancy_digest"]
        or occupied.shape != tuple(manifest["raster"]["shape_rows_columns"])
        or not np.allclose(map_origin, manifest["raster"]["origin_xy_m"], atol=1e-10, rtol=0)
        or not math.isclose(resolution, manifest["raster"]["resolution_m"])):
    raise RuntimeError("Planning raster differs from the audited H1 geometry")
output_folder = {"legacy_v1": "path_execution", "stop_align_v2": "path_execution_v2", "stop_align_v3": "path_execution_v3"}[execution_version]
OUTPUT = RESULTS / output_folder
if staged_terminal:
    OUTPUT = OUTPUT / ("readiness" if readiness_only else execution_cohort)
OUTPUT.mkdir(parents=True, exist_ok=True)
execution_prefix = selection["run_prefix"]
reference_prefix = method_helpers["method_settings"]("P_dynamics_full_envelope")["run_prefix"]
if staged_terminal:
    version_prefix = "hospital_h1_stopalign_"+execution_version.rsplit("_", 1)[1]+"_"
    execution_prefix = execution_prefix.replace("hospital_h1_", version_prefix, 1)
    reference_prefix = reference_prefix.replace("hospital_h1_", version_prefix, 1)
number = 1
while any((OUTPUT / f"{execution_prefix}_r{number:02d}{ext}").exists() for ext in (".json", ".csv")):
    number += 1
run_id = f"{execution_prefix}_r{number:02d}"
summary_path, trace_path = (OUTPUT / (run_id+ext) for ext in (".json", ".csv"))
clock_state = {"time_s": 0.}
counts, pairs, callback_errors = Counter(), Counter(), []


def _step(dt):
    clock_state["time_s"] += float(dt)


def _contacts(headers, data):
    del data
    try:
        for header in headers:
            pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
            category = classify_contact(pair, {preparation["support_floor"]}, set(preparation["disabled_legacy_floor_colliders"]))
            if category != "unrelated":
                counts[category] += 1
                pairs[(pair, category)] += 1
    except Exception as error:
        callback_errors.append(str(error))


def _angles(quaternion):
    w, x, y, z = map(float, quaternion)
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        math.asin(max(-1., min(1., 2*(w*y-z*x)))), math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


async def _run_h1_path():
    report = {"schema_version": 1, "run_id": run_id, "scenario": "H1_bedside_departure", "verdict": "FAIL",
        "trial_kind": "development_frozen_hospital_path_PhysX_execution_ground_truth_feedback",
        "plan_path": str(PLAN_PATH), "plan_sha256": plan_sha, "method": plan["method"],
        "execution_version": execution_version,
        "execution_role": "readiness_only_not_matched_comparison" if readiness_only else "four_method_development_pilot",
        "execution_cohort": "readiness" if readiness_only else execution_cohort if staged_terminal else "legacy_v1",
        "geometry_manifest_sha256": manifest_sha, "physics_preparation_snapshot": preparation,
        "static_gate_reference": str(static_path), "static_gate_sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
        "common_motion_stop_reference": str(component_path),
        "common_motion_stop_sha256": hashlib.sha256(component_path.read_bytes()).hexdigest(), "source_stage_saved": False,
        "start_world_xyyaw": plan["start_world_xyyaw"], "goal_world_xyyaw": plan["goal_world_xyyaw"],
        "planned_action_count": len(plan["selected_actions"]), "planned_nominal_distance_m": .05*len(plan["selected_actions"]),
        "planner_selected_modes": sorted({action["mode"] for action in plan["selected_actions"]}),
        "baseline_fixed_mode_scope": "planning path only; common bounded terminal controller may switch modes",
        "feedback_source": "Isaac_ground_truth_pose_and_joints_NOT_sensor_localization",
        "online_replanning": False, "Nav2_executed": False, "sensor_closed_loop_verified": False,
        "full_H1_PhysX_execution_verified": False, "path_executed": False,
        "path_execution_started": False,
        "cics_or_micro_yaw_input": False, "caster_state_reset_during_run": False, "material_modified_during_run": False,
        "body_pose_reset_during_run": False, "use_for_final_statistics": False,
        "common_controller": "bounded_path_yaw_lateral_feedback_and_bounded_2d_mode_switching_terminal",
        "command_limits": {"speed_m_s": .05, "acceleration_m_s2": .02, "steering_rate_deg_s": 8., "curvature_1_m": .70},
        "guards": {"path_error_m": .080, "path_heading_error_deg": 10., "tilt_deg": 2., "body_speed_m_s": .080,
            "maximum_control_gap_s": .25, "maximum_simulation_duration_s": 180., "maximum_wall_time_s": 900.},
        "acceptance": {"position_error_m": .030, "heading_error_deg": 2., "terminal_speed_m_s": .005,
            "hold_regression_speed_m_s": .001, "hold_drift_m": .003, "hold_observation_s": 5., "fixed_hold_tail_s": 3.},
        "terminal_capture_inner_band": {"position_m": .015, "heading_deg": 1.5},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            _ijcas_path(WORKSPACE, 'isaac/scripts/isaac_run_ijcas_hospital_h1_path_gate.py'), HELPERS / "hospital_path_execution.py",
            HELPERS / "hospital_drive_hold.py", HELPERS / "hospital_contact_rules.py", HELPERS / "hospital_geometry_tools.py")}}
    if staged_terminal:
        report["common_controller"] = execution_helpers["CONTROLLER_CONFIG"]["version"]
        report["terminal_controller_config"] = dict(execution_helpers["CONTROLLER_CONFIG"])
        report["source_sha256"][execution_file] = hashlib.sha256((HELPERS / execution_file).read_bytes()).hexdigest()
    if execution_version == "stop_align_v3":
        dependency = HELPERS / "hospital_path_execution_v2.py"
        report["source_sha256"][dependency.name] = hashlib.sha256(dependency.read_bytes()).hexdigest()
        report["fixed_hold_attempt_count"] = 0
    controller = Controller(plan)
    rows, views = [], []
    robot = motor = drive_idx = old_kp = old_kd = old_targets = None
    phase, previous_pose, previous_time = "pre_hold", None, None
    steer_command, hold_targets = (0., 0.), None
    physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_step)
    contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(_contacts)
    app, wall_start = omni.kit.app.get_app(), time.perf_counter()
    history = deque()
    try:
        if selection["method"] != "P_dynamics_full_envelope" and not readiness_only:
            reference_path = OUTPUT / (reference_prefix+"_r01.json")
            reference = json.loads(reference_path.read_bytes())
            mismatches = method_helpers["comparison_mismatches"](reference, report)
            if reference.get("verdict") != "PASS" or mismatches:
                raise RuntimeError("Comparison controller/geometry/physics differs from proposed r01: "+str(mismatches))
            report["matched_proposed_reference"] = str(reference_path)
            report["matched_proposed_reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            report["nominal_conditions_and_common_controller_match"] = True
            report["paired_random_seed_or_equal_caster_state_verified"] = False
        robot = SingleArticulation(BASE)
        robot.initialize()
        indices = {name: i for i, name in enumerate(robot.dof_names)}
        if any(name not in indices for name in (*STEERING, *DRIVES)):
            raise RuntimeError("Active motor joint mapping is incomplete")
        steer_idx = np.array([indices[name] for name in STEERING], dtype=np.int32)
        drive_idx = np.array([indices[name] for name in DRIVES], dtype=np.int32)
        motor = robot.get_articulation_controller()
        kp, kd = motor.get_gains()
        old_kp, old_kd = np.asarray(kp).copy(), np.asarray(kd).copy()
        old_limits = np.asarray(motor.get_max_efforts()).copy()
        if np.max(np.abs(old_kp[drive_idx])) > 1e-6:
            raise RuntimeError("Original velocity-only drive gains were not restored")
        old_targets = np.asarray(robot.get_applied_action().joint_positions)[drive_idx].copy()
        if not np.isfinite(old_targets).all():
            raise RuntimeError("Initial wheel position targets are not finite")
        for i, name in enumerate(WHEELS):
            view = RigidContactView(prim_paths_expr=ROOT+"/"+name, filter_paths_expr=[], name=run_id+"_"+str(i),
                prepare_contact_sensors=False, disable_stablization=False)
            view.initialize()
            views.append(view)
        sim = omni.physx.get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        encoded_base = PhysicsSchemaTools.sdfPathToInt(BASE)

        def measure():
            position, quaternion = robot.get_world_pose()
            q = np.asarray(robot.get_joint_positions(), dtype=float)
            qd = np.asarray(robot.get_joint_velocities(), dtype=float)
            linear = np.asarray(robot.get_linear_velocity(), dtype=float)
            if not all(np.isfinite(v).all() for v in (position, quaternion, q, qd, linear)):
                raise RuntimeError("Non-finite measured physical state")
            roll, pitch, yaw = _angles(quaternion)
            return (float(position[0]), float(position[1]), yaw), position, q, qd, linear, roll, pitch

        def verify_motor_settings(expected_kp):
            actual_kp, actual_kd = motor.get_gains()
            if not (np.allclose(actual_kp, expected_kp, rtol=1e-6, atol=1e-6)
                    and np.array_equal(actual_kd, old_kd)
                    and np.array_equal(np.asarray(motor.get_max_efforts()), old_limits)):
                raise RuntimeError("Motor gain/damping/torque-limit guard failed")

        def set_hold(enabled):
            nonlocal hold_targets
            if enabled:
                hold_targets = np.asarray(robot.get_joint_positions(), dtype=float)[drive_idx].copy()
                if not np.isfinite(hold_targets).all():
                    raise RuntimeError("Cannot capture finite wheel hold targets")
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
                raise RuntimeError("Body tilt/speed safety guard triggered")
            last = previous_pose if previous_pose is not None else pose
            turn = execution_helpers["wrap"](pose[2]-last[2])
            steps = max(1, math.ceil(math.hypot(pose[0]-last[0], pose[1]-last[1])/.005), math.ceil(abs(turn)/math.radians(.25)))
            for n in range(1, steps+1):
                fraction = n/steps
                check = (last[0]+fraction*(pose[0]-last[0]), last[1]+fraction*(pose[1]-last[1]), last[2]+fraction*turn)
                if pose_collision(occupied, check, map_origin, resolution, plan["full_envelope_clearance_m"]):
                    raise RuntimeError("Actual swept full-envelope raster guard triggered")
            if controller.phase != "path_tracking" and math.hypot(pose[0]-controller.goal[0], pose[1]-controller.goal[1]) > .12:
                raise RuntimeError("Terminal controller left the 120 mm goal neighbourhood")

        async def sample(steer, wheels, hold, info, phase_start):
            nonlocal previous_pose, previous_time
            command(steer, wheels, hold)
            while True:
                sim.wake_up(stage_id, encoded_base)
                await app.next_update_async()
                if not timeline.is_playing():
                    raise RuntimeError("Timeline controls changed during the H1 pilot")
                if time.perf_counter()-wall_start > 900.:
                    raise TimeoutError("H1 pilot exceeded 900 wall-clock seconds")
                now = clock_state["time_s"]
                if previous_time is None or now > previous_time:
                    break
            if now > 180. or (previous_time is not None and now-previous_time > .25):
                raise TimeoutError("H1 simulated-duration/control-gap safety guard")
            pose, position, q, qd, linear, roll, pitch = measure()
            guard(pose, linear, roll, pitch)
            verify_motor_settings(hold_gains(old_kp, old_kd, drive_idx)[0] if hold else old_kp)
            applied = robot.get_applied_action()
            actual_velocities = np.asarray(applied.joint_velocities)[drive_idx]
            actual_positions = np.asarray(applied.joint_positions)
            if not np.allclose(actual_velocities, wheels, rtol=0, atol=1e-6):
                raise RuntimeError("Actual wheel velocity targets differ from commands")
            if not np.allclose(actual_positions[steer_idx], steer, rtol=0, atol=1e-6):
                raise RuntimeError("Actual steering targets differ from commands")
            if hold and not np.allclose(actual_positions[drive_idx], hold_targets, rtol=0, atol=1e-6):
                raise RuntimeError("Captured wheel hold targets changed")
            history.append((now, pose))
            while len(history) > 2 and history[1][0] <= now-.20:
                history.popleft()
            dt = now-history[0][0]
            pose_speed = math.hypot(pose[0]-history[0][1][0], pose[1]-history[0][1][1])/dt if dt > 0 else math.hypot(linear[0], linear[1])
            row = {"time_s": now, "phase": phase, "phase_time_s": now-phase_start,
                "x_m": pose[0], "y_m": pose[1], "z_m": float(position[2]), "yaw_rad": pose[2],
                "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
                "vx_physx_m_s": float(linear[0]), "vy_physx_m_s": float(linear[1]), "vz_physx_m_s": float(linear[2]),
                "pose_filtered_speed_m_s": pose_speed, "command_speed_m_s": info.get("speed_m_s", 0.),
                "command_mode": info.get("mode", "pre_hold"), "command_curvature_1_m": info.get("curvature_1_m", 0.),
                "action_index": info.get("action_index", -1), "path_progress_m": info.get("progress_m", 0.),
                "path_error_m": info.get("path_error_m", 0.), "path_heading_error_deg": info.get("path_heading_error_deg", 0.),
                "goal_position_error_m": math.hypot(pose[0]-controller.goal[0], pose[1]-controller.goal[1]),
                "goal_heading_error_deg": math.degrees(execution_helpers["wrap"](controller.goal[2]-pose[2])),
                "front_rad": float(q[steer_idx[0]]), "rear_rad": float(q[steer_idx[1]]),
                "terminal_stage": info.get("terminal_stage", "legacy_v1"), "terminal_pulse_count": info.get("pulse_count", 0),
                "target_front_rad": float(steer[0]), "target_rear_rad": float(steer[1]), "position_hold_enabled": hold}
            for i, axle in enumerate(("front", "rear")):
                row["drive_"+axle+"_rad"] = float(q[drive_idx[i]])
                row["drive_"+axle+"_velocity_rad_s"] = float(qd[drive_idx[i]])
                row["drive_"+axle+"_actual_velocity_target_rad_s"] = float(actual_velocities[i])
            for name, i in indices.items():
                row["joint_"+name+"_rad"] = float(q[i])
            for i, view in enumerate(views):
                value = float(np.asarray(view.get_net_contact_forces(dt=1./120.)).reshape(-1, 3)[0, 2])
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite wheel support force")
                row[f"wheel_{i}_fz_n"] = value
            rows.append(row)
            previous_pose, previous_time = pose, now
            return row, pose, q, linear

        pose, _, q, _, linear, roll, pitch = measure()
        offset = relative_pose(pose, plan["start_world_xyyaw"])
        if math.hypot(offset[0], offset[1]) > .010 or abs(offset[2]) > math.radians(.5):
            raise RuntimeError("Robot is not within 10 mm / 0.5 degrees of the ORIGINAL H1 start; do not translate the plan")
        if max(abs(float(q[i])) for i in steer_idx) > math.radians(1.):
            raise RuntimeError("Steering joints do not match the nominal zero-angle plan release state")
        guard(pose, linear, roll, pitch)
        report["observed_prehold_world_xyyaw"] = list(pose)
        set_hold(True)
        command((0., 0.), (0., 0.), True)
        timeline.play()
        pre_start = clock_state["time_s"]
        print("H1 path phase: 2 s measured-wheel pre-hold at the original task start")
        while clock_state["time_s"]-pre_start < 2.:
            row, pose, q, linear = await sample((0., 0.), (0., 0.), True, {}, pre_start)
        offset = relative_pose(pose, plan["start_world_xyyaw"])
        if (math.hypot(offset[0], offset[1]) > .010 or abs(offset[2]) > math.radians(.5)
                or row["pose_filtered_speed_m_s"] > .005 or math.hypot(linear[0], linear[1]) > .005):
            raise RuntimeError("Pre-hold release pose/speed is outside the fixed plan tolerance")
        report["release_world_xyyaw"] = list(pose)
        report["release_offset_body_xyyaw"] = list(offset)
        report["release_physx_speed_m_s"] = math.hypot(linear[0], linear[1])
        set_hold(False)
        report["path_execution_started"] = True
        motion_start, command_time = clock_state["time_s"], clock_state["time_s"]
        controller.previous_time = motion_start
        printed_phase = None
        motion_hold = False
        while True:
            if staged_terminal:
                measured_speed = max(math.hypot(linear[0], linear[1]), row["pose_filtered_speed_m_s"])
                info = controller.step(clock_state["time_s"], pose, measured_speed, tuple(float(q[i]) for i in steer_idx))
            else:
                info = controller.step(clock_state["time_s"], pose, math.hypot(linear[0], linear[1]))
            phase = info["phase"]
            phase_label = (phase, info.get("terminal_stage", "legacy_v1"))
            if printed_phase != phase_label:
                print("H1 path phase:", phase, "/ "+info["terminal_stage"] if "terminal_stage" in info else "")
                printed_phase = phase_label
            if info["hold_ready"]:
                break
            requested_hold = bool(info.get("drive_hold_requested", False))
            if requested_hold and abs(info["speed_m_s"]) > 1e-7:
                raise RuntimeError("Nonzero wheel speed requested while active-wheel position hold is enabled")
            if requested_hold and not motion_hold and measured_speed > .005:
                raise RuntimeError("Staged controller requested active-wheel hold before measured zero-speed engagement")
            if requested_hold != motion_hold:
                set_hold(requested_hold)
                motion_hold = requested_hold
            dt = clock_state["time_s"]-command_time
            steer_command, wheels = direct_joint_targets(info["speed_m_s"], info["mode"], info["curvature_1_m"],
                steer_command, dt, tuple(float(q[i]) for i in steer_idx))
            command_time = clock_state["time_s"]
            row, pose, q, linear = await sample(steer_command, wheels, motion_hold, info, motion_start)
        report["terminal_hold_capture_physx_speed_m_s"] = math.hypot(linear[0], linear[1])
        report["terminal_handoff"] = controller.handoff
        if staged_terminal:
            report["terminal_pulse_count"] = controller.pulse_count
        if execution_version == "stop_align_v3":
            report["fixed_hold_entry_candidate"] = controller.hold_entry_candidate
            entry = controller.hold_entry_candidate or {}
            if (not entry or entry["position_error_m"] > .030 or abs(entry["heading_error_deg"]) > 2.
                    or measured_speed > .005 or not motion_hold
                    or controller.terminal_stage != "final_hold_ready"):
                raise RuntimeError("v3 fixed hold entry is outside the unchanged stopped acceptance region")
            report["fixed_hold_entry_measured_speed_m_s"] = measured_speed
            report["fixed_hold_attempt_count"] = 1
            print("H1 stopped inside existing 30 mm / 2 deg acceptance; starting ONE fixed hold test, not a PASS yet")
        set_hold(True)
        # Keep the final steering targets; do not straighten loaded wheels at
        # rest and introduce an unmodelled body displacement during evaluation.
        phase, hold_start = "terminal_hold", clock_state["time_s"]
        print("H1 path phase: 5 s terminal hold; fixed final 3 s evaluation")
        while clock_state["time_s"]-hold_start < 5.:
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
            "fixed_tail_position_under_30mm": all(r["goal_position_error_m"] <= .030 for r in tail),
            "fixed_tail_heading_under_2deg": all(abs(r["goal_heading_error_deg"]) <= 2. for r in tail),
            "fixed_tail_pose_standstill": analysis["development_standstill_check"] == "PASS",
            "fixed_tail_physx_speed_under_5mm_s": all(math.hypot(r["vx_physx_m_s"], r["vy_physx_m_s"]) <= .005 for r in tail),
            "fixed_tail_filtered_speed_under_5mm_s": all(r["pose_filtered_speed_m_s"] <= .005 for r in tail),
            "all_six_wheels_mean_load_over_1N": all(np.mean([r[f"wheel_{i}_fz_n"] for r in tail]) > 1. for i in range(6)),
            "vertical_force_weight_balance": .8 <= balance <= 1.2,
            "only_expected_external_support": counts["wheel_support"] > 0 and not any(counts[key] for key in BAD_CONTACTS),
            "contact_callback_monitor_healthy": not callback_errors,
            "plan_file_unchanged": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() == plan_sha}
        if execution_version == "stop_align_v3":
            checks["single_fixed_hold_attempt_no_best_window_selection"] = report["fixed_hold_attempt_count"] == 1
        report.update({"checks": {key: bool(value) for key, value in checks.items()},
            "verdict": "PASS" if all(checks.values()) else "FAIL", "path_executed": True,
            "terminal_position_error_m": position_error, "terminal_heading_error_deg": heading_error,
            "terminal_hold_analysis": analysis, "terminal_vertical_force_weight_ratio": balance,
            "final_world_xyyaw": list(pose), "final_physx_speed_m_s": math.hypot(linear[0], linear[1]),
            "motion_and_terminal_correction_time_s": hold_start-motion_start})
    except (Exception, asyncio.CancelledError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report.setdefault("checks", {})["path_execution_and_hold_completed"] = False
    finally:
        if robot is not None and drive_idx is not None:
            try:
                robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_idx))
            except Exception as error:
                report["zero_command_error"], report["verdict"] = str(error), "FAIL"
        # Pause before restoring velocity-only gains, so no unobserved creep
        # interval is introduced after the measured hold window.
        if timeline.is_playing():
            timeline.pause()
        if motor is not None and old_kp is not None:
            try:
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    motor.set_gains(kps=old_kp, kds=old_kd, save_to_usd=False)
                restored_kp, restored_kd = motor.get_gains()
                if not (np.allclose(restored_kp, old_kp) and np.array_equal(restored_kd, old_kd)
                        and np.array_equal(np.asarray(motor.get_max_efforts()), old_limits)):
                    raise RuntimeError("Original gain/damping/torque-limit restoration failed")
                if old_targets is not None:
                    robot.apply_action(ArticulationAction(joint_positions=old_targets.astype(np.float32),
                        joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_idx))
                report["original_motor_gains_restored"] = True
            except Exception as error:
                report["restore_error"], report["verdict"] = str(error), "FAIL"
                report["original_motor_gains_restored"] = False
        report["checks"] = {**report.get("checks", {}), "original_motor_gains_restored": report.get("original_motor_gains_restored", False)}
        if callback_errors or any(counts[key] for key in BAD_CONTACTS):
            report["checks"]["only_expected_external_support"] = False
        if not all(report["checks"].values()):
            report["verdict"] = "FAIL"
        report["full_H1_PhysX_execution_verified"] = report["verdict"] == "PASS"
        if readiness_only:
            report["four_method_comparison_verified"] = False
        if staged_terminal:
            report["terminal_pulse_count"] = controller.pulse_count
        physics_sub = contact_sub = None
        report.update({"sample_count": len(rows), "contact_report_counts_by_category": dict(counts),
            "contact_monitor_errors": callback_errors,
            "contact_pairs_by_category": [{"pair": pair, "category": category, "callback_count": count}
                for (pair, category), count in sorted(pairs.items())],
            "path_progress_m": controller.reference.progress, "reference_polyline_length_m": controller.reference.arc[-1],
            "terminal_execution_mode_switches": controller.terminal_switches,
            "terminal_handoff": controller.handoff, "wall_time_s": time.perf_counter()-wall_start})
        report["measured_path_length_m"] = sum(math.hypot(last["x_m"]-first["x_m"], last["y_m"]-first["y_m"])
            for first, last in zip(rows, rows[1:]) if first["phase"] != "pre_hold" and last["phase"] != "pre_hold")
        if rows:
            report["last_recorded_state"] = {key: rows[-1][key] for key in ("time_s", "phase", "terminal_stage",
                "x_m", "y_m", "yaw_rad", "goal_position_error_m", "goal_heading_error_deg", "command_speed_m_s")}
        if rows:
            with trace_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            setup.SetCustomDataByKey("last_h1_path_gate", str(summary_path))
            setup.SetCustomDataByKey("setup_status", "path_gate_finished_fresh_start_required")
        print("=== IJCAS HOSPITAL H1 FROZEN PATH EXECUTION GATE ===")
        print("run:", run_id)
        print("execution version / role:", execution_version, report["execution_role"])
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
        print("Timeline paused. Ground-truth PhysX development pilot only; no sensor/Nav2 or final comparison claim.")
        print("Do not save the source USD. Do not press Play again after restored velocity-mode gains.")


ijcas_hospital_h1_path_task = asyncio.ensure_future(_run_h1_path())
print("H1 frozen path pilot scheduled:", run_id, "/", selection["method"])
print("Automatic 2 s pre-hold -> ordinary ramp/path tracking -> bounded terminal closure -> 5 s hold -> Pause and restore gains.")
print("Keep timeline controls untouched. No CICS input, sensor localization or Nav2 is used.")
