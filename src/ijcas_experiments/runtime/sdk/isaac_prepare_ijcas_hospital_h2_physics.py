"""H2 nominal physics baseline, Session Layer only. Does NOT start physics.

Keep the validated H1 mass/contact/joint settings; only task placement differs.
Use the exact H2 preview manifest and preserve raised thresholds and furniture.
No passive-caster state reset or source USD save. Output is a new revision.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

for _h2guard_name in ("ijcas_hospital_h1_static_task", "ijcas_hospital_h1_initial_task", "ijcas_hospital_h1_zero_task",
        "ijcas_hospital_h1_motion_stop_task", "ijcas_hospital_h1_path_task", "ijcas_hospital_h2_static_task",
        "ijcas_hospital_h2_path_task"):
    _h2guard_task = globals().get(_h2guard_name)
    if _h2guard_task is not None and not _h2guard_task.done():
        raise RuntimeError("A hospital trial is running; H2 physics preparation refused")

import hashlib
import json
import math
from pathlib import Path
import runpy

import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


WORKSPACE = _ijcas_workspace()
RESULTS = _ijcas_path(WORKSPACE, 'studies/IJCAS/results/hospital_paper_gate')
SOURCE_USD = _ijcas_path(WORKSPACE, 'isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd')
ROOT = "/World/patient_transport_lab_screening"
GROUND_ROOT = "/World/PatientTransportLabGround"
GROUND = GROUND_ROOT+"/CollisionBox"
MATERIAL = GROUND_ROOT+"/WheelGroundContact"
PREVIEW = "/World/IJCAS_HospitalH2SetupPreview"
WHEELS = ("front_active_drive_wheel_link", "rear_active_drive_wheel_link", "front_left_caster_wheel_link",
    "front_right_caster_wheel_link", "rear_left_caster_wheel_link", "rear_right_caster_wheel_link")
JOINT_SETTINGS = {"base_link_to_front_active_steering": (30., 5., 250.),
    "base_link_to_rear_active_steering": (30., 5., 250.), "front_active_steering_to_drive_wheel": (0., 50., 500.),
    "rear_active_steering_to_drive_wheel": (0., 50., 500.)}
floor_rule = runpy.run_path(str(_ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_contact_rules.py')))["exported_floor_paths"]
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_stopped():
    raise RuntimeError("H2 physics requires STOPPED timeline after the original H2 preview")
stage = omni.usd.get_context().get_stage()
setup = stage.GetPrimAtPath(PREVIEW) if stage is not None else None
if setup is None or not setup.IsValid():
    raise RuntimeError("Run the original H2 preview first, not H1 preparation")
manifest_path = Path(setup.GetCustomDataByKey("geometry_manifest") or "")
manifest_raw = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
manifest = json.loads(manifest_raw)
if (manifest.get("scenario") != "H2_bedside_lateral_alignment" or not all(manifest["checks"].values())
        or setup.GetCustomDataByKey("geometry_manifest_sha256") != manifest_sha):
    raise RuntimeError("H2 preview/setup evidence is stale")
if Path(stage.GetRootLayer().realPath).resolve() != Path(manifest["source_stage"]).resolve():
    raise RuntimeError("Open the audited hospital source stage")
if not math.isclose(float(UsdGeom.GetStageMetersPerUnit(stage)), 1.) or str(UsdGeom.GetStageUpAxis(stage)) != "Z":
    raise RuntimeError("H2 physics requires a metre-based Z-up hospital stage")
for name, expected in manifest["input_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("H2 source evidence changed: "+name)
audit = json.loads(Path(manifest["source_audit"]).read_bytes())
floor_paths = floor_rule(audit)
if not floor_paths:
    raise RuntimeError("No authoritative hospital floor collider paths")
base = stage.GetPrimAtPath(ROOT+"/base_link")
if not base.IsValid():
    raise RuntimeError("The current IJCAS robot is missing")
matrix = UsdGeom.XformCache().GetLocalToWorldTransform(base)
position, forward = matrix.ExtractTranslation(), matrix.TransformDir(Gf.Vec3d(1., 0., 0.))
start = manifest["start_xyyaw"]
yaw_error = math.atan2(forward[1], forward[0])-start[2]
if (math.hypot(position[0]-start[0], position[1]-start[1]) > .005
        or abs(math.atan2(math.sin(yaw_error), math.cos(yaw_error))) > math.radians(.5)):
    raise RuntimeError("Robot differs from the original H2 start; reopen the source and run H2 preview again")
scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
if len(scenes) != 1:
    raise RuntimeError("Expected exactly one PhysicsScene")
graphs = [prim for prim in stage.Traverse() if prim.GetTypeName() == "OmniGraph"]
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    useExtentsHint=False)
for name, expected in manifest["audited_bed_bounds"].items():
    prim = stage.GetPrimAtPath(name)
    if not prim.IsValid():
        raise RuntimeError("Missing audited bed: "+name)
    bound = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
    if any(not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-5)
            for actuals, values in ((bound.GetMin(), expected["minimum_m"]), (bound.GetMax(), expected["maximum_m"]))
            for actual, value in zip(actuals, values)):
        raise RuntimeError("Bed geometry was moved after H2 setup: "+name)
output_folder = RESULTS / "hospital_h2_physics_preparation"
output_folder.mkdir(parents=True, exist_ok=True)
index = 1
while (output_folder / f"hospital_h2_physics_r{index:02d}.json").exists():
    index += 1
OUTPUT = output_folder / f"hospital_h2_physics_r{index:02d}.json"
session = stage.GetSessionLayer()
session_before = session.ExportToString()
try:
    with Usd.EditContext(stage, session):
        disabled_graphs = []
        for prim in graphs:
            disabled_graphs.append(str(prim.GetPath()))
            prim.SetActive(False)
        disabled_floors, deinstanced_floor_roots = [], []
        for path in floor_paths:
            for prefix in Sdf.Path(path).GetPrefixes():
                ancestor = stage.GetPrimAtPath(prefix)
                if ancestor.IsValid() and ancestor.IsInstance():
                    deinstanced_floor_roots.append(str(prefix))
                    ancestor.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError("Audited floor collider is not editable: "+path)
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                raise RuntimeError("Floor collision override failed: "+path)
            disabled_floors.append(path)
        for prim in list(stage.Traverse()):
            path = str(prim.GetPath())
            if path.startswith(ROOT+"/") or path.startswith(GROUND_ROOT+"/") or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = UsdPhysics.CollisionAPI(prim)
            if api.GetCollisionEnabledAttr().Get() is False:
                continue
            try:
                bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                lo, hi = bounds.GetMin(), bounds.GetMax()
                floor_like = abs(float(hi[2])-.001) <= .005 and float(hi[2]-lo[2]) < .010
                named_plane = ("ground" in path.lower() or "floor" in path.lower()) and (
                    "collisionplane" in path.lower() or prim.GetTypeName() == "Plane")
            except Exception:
                floor_like, named_plane = False, "collisionplane" in path.lower()
            if floor_like or named_plane:
                api.CreateCollisionEnabledAttr(False).Set(False)
                disabled_floors.append(path)
        legacy = "/World/Hospital/GroundPlane/CollisionPlane"
        legacy_prim = stage.GetPrimAtPath(legacy)
        if legacy_prim.IsValid() and legacy_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(legacy_prim).CreateCollisionEnabledAttr(False).Set(False)
            disabled_floors.append(legacy)
        ground_root = stage.GetPrimAtPath(GROUND_ROOT)
        if not ground_root.IsValid():
            ground_root = stage.DefinePrim(GROUND_ROOT, "Xform")
            ground_root.GetReferences().AddReference(str(SOURCE_USD), GROUND_ROOT)
        if not stage.GetPrimAtPath(GROUND).IsValid() or not stage.GetPrimAtPath(MATERIAL).IsValid():
            raise RuntimeError("Validated IJCAS support floor/material branch is missing")
        transform = UsdGeom.Xformable(ground_root)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp(opSuffix="hospitalH2").Set(Gf.Vec3d(start[0], start[1], .001))
        ground = stage.GetPrimAtPath(GROUND)
        UsdPhysics.CollisionAPI.Apply(ground).CreateCollisionEnabledAttr(True).Set(True)
        UsdGeom.Imageable(ground).MakeInvisible()
        material = UsdShade.Material(stage.GetPrimAtPath(MATERIAL))
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(.8).Set(.8)
        physics_material.CreateDynamicFrictionAttr(.6).Set(.6)
        physics_material.CreateRestitutionAttr(0.).Set(0.)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateCompliantContactAccelerationSpringAttr(False).Set(False)
        physx_material.CreateCompliantContactStiffnessAttr(100000.).Set(100000.)
        physx_material.CreateCompliantContactDampingAttr(1500.).Set(1500.)
        physx_material.CreateFrictionCombineModeAttr("max").Set("max")
        bound_wheels = []
        for name in WHEELS:
            body = stage.GetPrimAtPath(ROOT+"/"+name)
            if not body.IsValid():
                raise RuntimeError("Missing wheel: "+name)
            colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
            if len(colliders) != 1 or colliders[0].GetTypeName() != "Capsule":
                raise RuntimeError("Unexpected wheel collision shape: "+name)
            UsdShade.MaterialBindingAPI.Apply(body).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
            bound_wheels.append(str(body.GetPath()))
        UsdShade.MaterialBindingAPI.Apply(ground).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")
        mass = UsdPhysics.MassAPI.Apply(base)
        previous_mass, inertia = mass.GetMassAttr().Get(), mass.GetDiagonalInertiaAttr().Get()
        if previous_mass is None or previous_mass <= 0 or inertia is None:
            raise RuntimeError("Base mass/inertia is missing")
        mass.CreateMassAttr(245.).Set(245.)
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v)*245./float(previous_mass) for v in inertia)))
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0., 0., .270))
        bodies = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                if value is None or value <= 0:
                    raise RuntimeError("Missing positive body mass: "+str(prim.GetPath()))
                bodies.append({"path": str(prim.GetPath()), "mass_kg": float(value)})
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False).Set(False)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.).Set(0.)
        total_mass = sum(row["mass_kg"] for row in bodies)
        if len(bodies) != 13 or abs(total_mass-265.) > .01:
            raise RuntimeError("Expected 265 kg / 13 bodies")
        for name, (kp, kd, force) in JOINT_SETTINGS.items():
            prim = stage.GetPrimAtPath(ROOT+"/joints/"+name)
            if not prim.IsValid() or not prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError("Missing active joint: "+name)
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                raise RuntimeError("Missing angular joint drive: "+name)
            drive.CreateTypeAttr("force").Set("force")
            drive.CreateStiffnessAttr(kp).Set(kp)
            drive.CreateDampingAttr(kd).Set(kd)
            drive.CreateMaxForceAttr(force).Set(force)
            drive.CreateTargetPositionAttr(0.).Set(0.)
            drive.CreateTargetVelocityAttr(0.).Set(0.)
        scene = UsdPhysics.Scene(scenes[0])
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0., 0., -1.))
        scene.CreateGravityMagnitudeAttr(9.81).Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(scenes[0]).CreateTimeStepsPerSecondAttr(120).Set(120)
        base.CreateAttribute("experiment:ijcasRobustnessCondition", Sdf.ValueTypeNames.String).Set("nominal_265_center")
        report = {"schema_version": 3, "scenario": manifest["scenario"], "status": "prepared_not_executed",
            "source_stage": stage.GetRootLayer().identifier, "geometry_manifest": str(manifest_path),
            "geometry_manifest_sha256": manifest_sha, "source_audit_sha256": manifest["source_sha256"],
            "robot_reference": str(SOURCE_USD), "robot_base": ROOT+"/base_link", "start_xyyaw": start,
            "condition": "nominal_265_center", "simulation_mass_kg": total_mass, "physics_hz": 120, "gravity_m_s2": 9.81,
            "static_friction": .8, "dynamic_friction": .6, "contact_stiffness_n_m": 100000., "contact_damping_n_s_m": 1500.,
            "joint_settings_USD_angular_units": JOINT_SETTINGS, "support_floor": GROUND, "support_floor_top_z_m": .001,
            "bound_wheel_bodies": bound_wheels, "rigid_bodies": bodies,
            "disabled_legacy_floor_colliders": sorted(set(disabled_floors)),
            "deinstanced_floor_roots": sorted(set(deinstanced_floor_roots)),
            "audited_hospital_floor_collider_count": len(floor_paths),
            "all_audited_hospital_floor_colliders_disabled": all(
                UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get() is False for path in floor_paths),
            "disabled_action_graphs": disabled_graphs, "raised_door_thresholds": "retained; traversal not validated",
            "mass_and_material_status": "same_H1_provisional_simulation_parameters_not_physical_identification",
            "session_only": True, "ros_execution_enabled": False, "caster_joint_state_reset": False,
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
                ("isaac_prepare_ijcas_hospital_h2_physics.py", _ijcas_path(WORKSPACE, 'isaac/scripts/isaac_prepare_ijcas_hospital_h2_physics.py')),
                ("hospital_contact_rules.py", _ijcas_path(WORKSPACE, 'studies/IJCAS/scripts/hospital_contact_rules.py')))}}
        if not report["all_audited_hospital_floor_colliders_disabled"]:
            raise RuntimeError("An audited floor collider was not disabled")
        with OUTPUT.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        setup.SetCustomDataByKey("setup_status", "physics_prepared_stationary_gate_pending")
        setup.SetCustomDataByKey("physics_manifest_sha256", manifest_sha)
        setup.SetCustomDataByKey("physics_preparation_schema", 3)
        setup.SetCustomDataByKey("physics_preparation_result", str(OUTPUT))
        setup.SetCustomDataByKey("physics_preparation_sha256", hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
except Exception:
    session.ImportFromString(session_before)
    raise
print("=== IJCAS HOSPITAL H2 PHYSICS PREPARED ===")
print("mass/bodies:", round(total_mass, 3), "kg /", len(bodies))
print("friction static/dynamic: 0.8 / 0.6; physics: 120 Hz; unchanged H1 contact parameters")
print("audited floor colliders disabled:", len(floor_paths), "/", len(floor_paths))
print("disabled legacy floor colliders:", len(set(disabled_floors)))
print("disabled competing action graphs:", len(disabled_graphs))
print("saved:", OUTPUT)
print("Session Layer only. Timeline stays stopped. Next run isaac_run_ijcas_hospital_h2_static_gate.py.")
