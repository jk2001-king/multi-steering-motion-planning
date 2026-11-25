"""One H2 zero-command support gate. Auto Play -> 1 s settle -> 2 s -> Pause.

Run STOPPED immediately after H2 physics preparation. No forward/CICS/path or
Nav2 command and no gain/material/caster reset. Keep timeline controls alone.
Results are unique H2 files, never H1 evidence or a hospital-driving PASS.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

for _h2static_name in ("ijcas_hospital_h1_static_task", "ijcas_hospital_h1_initial_task", "ijcas_hospital_h1_zero_task",
        "ijcas_hospital_h1_motion_stop_task", "ijcas_hospital_h1_path_task", "ijcas_hospital_h2_static_task",
        "ijcas_hospital_h2_path_task"):
    _h2static_task = globals().get(_h2static_name)
    if _h2static_task is not None and not _h2static_task.done():
        raise RuntimeError("Another hospital task is running; H2 static launch refused")

import asyncio
from collections import Counter
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
from isaacsim.core.utils.types import ArticulationAction
from pxr import PhysicsSchemaTools, Usd, UsdPhysics, UsdUtils


WORKSPACE = _ijcas_workspace()
RESULTS = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
ROOT = "/World/patient_transport_lab_screening"
BASE = ROOT+"/base_link"
PREVIEW = "/World/IJCAS_HospitalH2SetupPreview"
WHEEL_BODIES = {"active_front": "front_active_drive_wheel_link", "active_rear": "rear_active_drive_wheel_link",
    "caster_fl": "front_left_caster_wheel_link", "caster_fr": "front_right_caster_wheel_link",
    "caster_rl": "rear_left_caster_wheel_link", "caster_rr": "rear_right_caster_wheel_link"}
STEERING = ("base_link_to_front_active_steering", "base_link_to_rear_active_steering")
DRIVES = ("front_active_steering_to_drive_wheel", "rear_active_steering_to_drive_wheel")
SETTLE_S, SAMPLE_S, PHYSICS_DT = 1., 2., 1./120.
classify_contact = runpy.run_path(str(_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_contact_rules.py')))["classify_contact"]
static_rules = runpy.run_path(str(_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_static_gate.py')))
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_stopped():
    raise RuntimeError("H2 static gate requires STOPPED timeline after H2 physics preparation")
stage = omni.usd.get_context().get_stage()
setup = stage.GetPrimAtPath(PREVIEW) if stage is not None else None
if (setup is None or not setup.IsValid() or setup.GetCustomDataByKey("setup_status") != "physics_prepared_stationary_gate_pending"
        or setup.GetCustomDataByKey("physics_preparation_schema") != 3):
    raise RuntimeError("Run the H2 physics preparation first; do not use an H1 preparation file")
PREPARED = Path(setup.GetCustomDataByKey("physics_preparation_result") or "")
preparation_raw = PREPARED.read_bytes()
preparation = json.loads(preparation_raw)
if (preparation.get("schema_version") != 3 or preparation.get("scenario") != "H2_bedside_lateral_alignment"
        or preparation.get("all_audited_hospital_floor_colliders_disabled") is not True
        or hashlib.sha256(preparation_raw).hexdigest() != setup.GetCustomDataByKey("physics_preparation_sha256")):
    raise RuntimeError("H2 physics preparation is invalid/stale")
manifest_path = Path(preparation["geometry_manifest"])
manifest_raw = manifest_path.read_bytes()
manifest = json.loads(manifest_raw)
if (hashlib.sha256(manifest_raw).hexdigest() != preparation["geometry_manifest_sha256"]
        or setup.GetCustomDataByKey("physics_manifest_sha256") != preparation["geometry_manifest_sha256"]
        or Path(stage.GetRootLayer().realPath).resolve() != Path(manifest["source_stage"]).resolve()):
    raise RuntimeError("H2 stage/geometry changed after physics preparation")
for name, expected in manifest["input_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("H2 source evidence changed: "+name)
for name, expected in preparation["source_sha256"].items():
    source = _ijcas_path(WORKSPACE, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts', name)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("H2 physics preparation code changed; prepare again: "+name)
if any(prim.GetTypeName() == "OmniGraph" for prim in stage.Traverse()):
    raise RuntimeError("Competing action graph is still active")
support = stage.GetPrimAtPath(preparation["support_floor"])
if not support.IsValid() or UsdPhysics.CollisionAPI(support).GetCollisionEnabledAttr().Get() is False:
    raise RuntimeError("Canonical support floor is not enabled")
for path in preparation["disabled_legacy_floor_colliders"]:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
        raise RuntimeError("Duplicate support collider is active: "+path)
OUTPUT = RESULTS / "hospital_h2_static_contact"
OUTPUT.mkdir(parents=True, exist_ok=True)
index = 1
while any((OUTPUT / f"hospital_h2_static_r{index:02d}{extension}").exists() for extension in (".json", ".csv")):
    index += 1
run_id = f"hospital_h2_static_r{index:02d}"
summary_path, trace_path = (OUTPUT / (run_id+extension) for extension in (".json", ".csv"))
h2_clock_state = {"time_s": 0.}
h2_counts, h2_pairs, h2_callback_errors = Counter(), Counter(), []
expected_supports = {preparation["support_floor"]}
disabled_legacy_supports = set(preparation["disabled_legacy_floor_colliders"])


def _h2_static_step(dt):
    h2_clock_state["time_s"] += float(dt)


def _h2_static_contacts(headers, data):
    del data
    try:
        for header in headers:
            pair = tuple(sorted((str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)))))
            category = classify_contact(pair, expected_supports, disabled_legacy_supports)
            if category != "unrelated":
                h2_counts[category] += 1
                h2_pairs[(pair, category)] += 1
    except Exception as error:
        h2_callback_errors.append(str(error))


def _h2_static_rpy(quaternion):
    w, x, y, z = map(float, quaternion)
    return (math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
        math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x))))),
        math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


async def _run_h2_static():
    report = {"schema_version": 3, "run_id": run_id, "scenario": manifest["scenario"],
        "trial_kind": "H2_zero_command_static_contact_NOT_path_execution", "verdict": "FAIL",
        "checks": {"stationary_observation_completed": False}, "geometry_manifest": str(manifest_path),
        "geometry_manifest_sha256": preparation["geometry_manifest_sha256"], "physics_preparation": str(PREPARED),
        "physics_preparation_sha256": hashlib.sha256(preparation_raw).hexdigest(), "physics_preparation_snapshot": preparation,
        "simulation_mass_kg": preparation["simulation_mass_kg"], "start_xyyaw": manifest["start_xyyaw"],
        "settle_s": SETTLE_S, "sampling_s": SAMPLE_S, "initial_motion_executed": False, "planned_path_executed": False,
        "Nav2_executed": False, "CICS_input_used": False, "caster_joint_state_reset_during_run": False,
        "material_modified_during_run": False, "motor_gains_modified_by_test": False, "use_for_final_statistics": False,
        "source_stage_saved": False, "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ("isaac_run_ijcas_hospital_h2_static_gate.py", _ijcas_path(WORKSPACE, 'isaac/scripts/isaac_run_ijcas_hospital_h2_static_gate.py')),
            ("hospital_static_gate.py", _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_static_gate.py')),
            ("hospital_contact_rules.py", _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_contact_rules.py')))}}
    rows, views = [], []
    robot = motor = drive_indices = old_kp = old_kd = old_limits = None
    physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(_h2_static_step)
    contact_sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(_h2_static_contacts)
    app, wall_start = omni.kit.app.get_app(), time.perf_counter()
    try:
        for label, name in WHEEL_BODIES.items():
            views.append(RigidContactView(prim_paths_expr=ROOT+"/"+name, filter_paths_expr=[], name=run_id+"_"+label,
                prepare_contact_sensors=False, disable_stablization=False))
        timeline.play()
        for _ in range(5):
            await app.next_update_async()
        if not timeline.is_playing():
            raise RuntimeError("Timeline controls changed during H2 initialization")
        robot = SingleArticulation(BASE)
        robot.initialize()
        indices = {name: i for i, name in enumerate(robot.dof_names)}
        if any(name not in indices for name in (*STEERING, *DRIVES)):
            raise RuntimeError("Missing active steering/drive joint mapping")
        steer_indices = np.array([indices[name] for name in STEERING], dtype=np.int32)
        drive_indices = np.array([indices[name] for name in DRIVES], dtype=np.int32)
        motor = robot.get_articulation_controller()
        kp, kd = motor.get_gains()
        old_kp, old_kd = np.asarray(kp).copy(), np.asarray(kd).copy()
        old_limits = np.asarray(motor.get_max_efforts()).copy()
        if np.max(np.abs(old_kp[drive_indices])) > 1e-6:
            raise RuntimeError("Drive gains are not the original velocity-only baseline")

        def zero_command():
            robot.apply_action(ArticulationAction(joint_positions=np.zeros(2, dtype=np.float32), joint_indices=steer_indices))
            robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_indices))

        for view in views:
            view.initialize()
        sim = omni.physx.get_physx_simulation_interface()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        encoded_base = PhysicsSchemaTools.sdfPathToInt(BASE)
        trial_start = previous_time = h2_clock_state["time_s"]
        next_sample = SETTLE_S
        while h2_clock_state["time_s"]-trial_start < SETTLE_S+SAMPLE_S:
            zero_command()
            sim.wake_up(stage_id, encoded_base)
            await app.next_update_async()
            if not timeline.is_playing():
                raise RuntimeError("Timeline controls changed during the H2 static gate")
            if time.perf_counter()-wall_start > 120.:
                raise TimeoutError("H2 stationary trial exceeded 120 wall-clock seconds")
            now = h2_clock_state["time_s"]
            if now-previous_time > .25:
                raise TimeoutError("H2 static control gap exceeded 0.25 simulated seconds")
            previous_time = now
            position, quaternion = robot.get_world_pose()
            position = np.asarray(position, dtype=float)
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                raise RuntimeError("Non-finite H2 pose")
            if np.linalg.norm(position[:2]-np.asarray(manifest["start_xyyaw"][:2])) > .030:
                raise RuntimeError("H2 zero-command drift exceeded the 30 mm safety bound")
            if h2_callback_errors or any(h2_counts[key] for key in static_rules["BAD_CONTACTS"]):
                raise RuntimeError("H2 unexpected obstacle/support contact or callback error")
            actual_kp, actual_kd = motor.get_gains()
            if not (np.array_equal(actual_kp, old_kp) and np.array_equal(actual_kd, old_kd)
                    and np.array_equal(np.asarray(motor.get_max_efforts()), old_limits)):
                raise RuntimeError("H2 motor gains/damping/torque limits changed during the static test")
            roll, pitch, yaw = _h2_static_rpy(quaternion)
            if max(abs(roll), abs(pitch)) > 2.:
                raise RuntimeError("H2 static tilt exceeded the two-degree safety bound")
            elapsed = now-trial_start
            if elapsed < next_sample:
                continue
            forces = [np.asarray(view.get_net_contact_forces(dt=PHYSICS_DT), dtype=float).reshape(-1, 3)[0] for view in views]
            if not np.isfinite(forces).all():
                raise RuntimeError("Non-finite measured H2 support force")
            row = {"time_s": elapsed, "x_m": float(position[0]), "y_m": float(position[1]), "z_m": float(position[2]),
                "roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw}
            row.update({label+"_fz_n": float(force[2]) for label, force in zip(WHEEL_BODIES, forces)})
            rows.append(row)
            next_sample += 1./60.
        zero_command()
        report.update(static_rules["evaluate_static"](rows, manifest["start_xyyaw"], preparation["simulation_mass_kg"],
            h2_counts, h2_callback_errors))
        report["checks"]["stationary_observation_completed"] = True
    except (Exception, asyncio.CancelledError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["verdict"] = "FAIL"
    finally:
        if robot is not None and drive_indices is not None and timeline.is_playing():
            try:
                robot.apply_action(ArticulationAction(joint_velocities=np.zeros(2, dtype=np.float32), joint_indices=drive_indices))
            except Exception as error:
                report["zero_command_error"], report["verdict"] = str(error), "FAIL"
                report["checks"]["final_zero_command_sent"] = False
        if timeline.is_playing():
            timeline.pause()
        if motor is not None and old_kp is not None:
            actual_kp, actual_kd = motor.get_gains()
            report["checks"]["original_motor_gains_and_limits_unchanged"] = bool(
                np.array_equal(actual_kp, old_kp) and np.array_equal(actual_kd, old_kd)
                and np.array_equal(np.asarray(motor.get_max_efforts()), old_limits))
        report["checks"]["contact_callback_monitor_healthy"] = not h2_callback_errors
        report["checks"]["only_expected_external_support"] = not any(h2_counts[key] for key in static_rules["BAD_CONTACTS"])
        if report.get("error") or not all(report["checks"].values()):
            report["verdict"] = "FAIL"
        physics_sub = contact_sub = None
        report.update({"contact_report_counts_by_category": dict(h2_counts), "contact_callback_errors": h2_callback_errors,
            "contact_pairs_by_category": [{"pair": pair, "category": category, "callback_count": count}
                for (pair, category), count in sorted(h2_pairs.items())], "wall_time_s": time.perf_counter()-wall_start})
        if rows:
            with trace_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            setup.SetCustomDataByKey("setup_status", "stationary_gate_passed" if report["verdict"] == "PASS" else "stationary_gate_failed")
            setup.SetCustomDataByKey("static_gate_result", str(summary_path))
        print("=== IJCAS HOSPITAL H2 STATIC CONTACT GATE ===")
        print("run:", run_id)
        for label, load in report.get("mean_wheel_vertical_loads_n", {}).items():
            print(f"{label}: {load:.2f} N")
        for label, key in (("expected wheel-floor callbacks", "wheel_support"), ("hospital obstacle callbacks", "hospital_obstacle"),
                ("unexpected legacy-floor callbacks", "unexpected_legacy_support"), ("non-wheel floor callbacks", "nonwheel_support"),
                ("unclassified external callbacks", "unclassified_external")):
            print(label+":", h2_counts[key])
        if "maximum_xy_drift_m" in report:
            print("maximum XY drift mm:", round(1000*report["maximum_xy_drift_m"], 3))
            print("base Z span mm:", round(1000*report["base_z_span_m"], 3))
            print("maximum roll/pitch deg:", round(report["maximum_abs_roll_pitch_deg"], 3))
            print("vertical force / weight:", round(report["mean_vertical_force_to_weight_ratio"], 4))
        print("failed checks:", [key for key, value in report["checks"].items() if not value])
        if report.get("error"):
            print("error:", report["error"])
        print("VERDICT:", report["verdict"])
        print("saved:", summary_path)
        print("Timeline PAUSED. H2 path execution, sensor/Nav2 and CICS have NOT run. Do not save the source USD.")


ijcas_hospital_h2_static_task = asyncio.ensure_future(_run_h2_static())
print("Hospital H2 zero-command static test scheduled:", run_id)
print("Automatic Play -> 1 s settling -> 2 s sampling -> Pause. Do not move the robot or press timeline controls.")
