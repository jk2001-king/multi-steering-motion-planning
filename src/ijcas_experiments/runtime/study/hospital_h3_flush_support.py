"""Evidence and support checks for H3_flush; not a PhysX result generator.

The old retained-threshold procedures and observations remain byte-unchanged.
Functions needing USD import it locally; geometry/evidence tests run without Kit.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy


CONDITION = "H3_flush_fully_open_continuous_floor"
THRESHOLD = "/World/Hospital/Geo_M_DoorFloor20"
ENVIRONMENT = "/World/IJCAS_HospitalH3FlushEnvironment"
COMPONENT = "/World/IJCAS_HospitalH3FlushComponent"
ROBOT = "/World/patient_transport_lab_screening"
GROUND_ROOT = "/World/PatientTransportLabGround"
GROUND = GROUND_ROOT + "/CollisionBox"
MATERIAL = GROUND_ROOT + "/WheelGroundContact"
WHEELS = (
    "front_active_drive_wheel_link", "rear_active_drive_wheel_link",
    "front_left_caster_wheel_link", "front_right_caster_wheel_link",
    "rear_left_caster_wheel_link", "rear_right_caster_wheel_link",
)
JOINT_SETTINGS = {
    "base_link_to_front_active_steering": (30., 5., 250.),
    "base_link_to_rear_active_steering": (30., 5., 250.),
    "front_active_steering_to_drive_wheel": (0., 50., 500.),
    "rear_active_steering_to_drive_wheel": (0., 50., 500.),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError("Recorded H3_flush evidence changed: " + str(path))


def load_setup(directory, workspace):
    """Verify real input lineage; never turn a fixture/model flag into evidence."""
    workspace = Path(workspace)
    path = Path(directory) / "flush_component_setup.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("environment_condition") != CONDITION
            or manifest.get("scenario") != "H3_doorway_immediate_turn"
            or manifest.get("purpose") != "FLUSH_STRAIGHT_COMPONENT_READINESS_NOT_H3_IMMEDIATE_TURN_MISSION"
            or manifest.get("component_map_may_be_used_for_path_planning_or_Nav2") is not False
            or not manifest.get("checks") or not all(manifest["checks"].values())
            or manifest.get("suppressed_threshold_root") != THRESHOLD):
        raise ValueError("Need the actual explicit H3_flush component geometry manifest")
    if manifest.get("footprint") != {"length_m": 1.63, "width_m": .965, "clearance_m": .01}:
        raise ValueError("The common footprint/clearance changed")
    verify_files(manifest["input_sha256"])
    report_path = Path(manifest["flush_geometry_report"])
    if (report_path.parent / "preparation_failure.json").exists():
        raise ValueError("The flush geometry preparation rolled back")
    report = json.loads(report_path.read_bytes())
    if (sha(report_path) != manifest["flush_geometry_sha256"]
            or report.get("condition") != CONDITION or report.get("live_geometry_verified") is not True
            or not report.get("checks") or not all(report["checks"].values())
            or report.get("source_USD_saved") is not False or report.get("PhysX_executed") is not False):
        raise ValueError("Invalid live geometry-only flush report")
    verify_files(report["input_sha256"])
    verify_files(report["original_threshold_results_preserved"])
    if len(report["original_threshold_results_preserved"]) < 2:
        raise ValueError("Original threshold FAIL records must be preserved")
    if sha(report["source_stage"]) != report["source_stage_sha256"]:
        raise ValueError("The hospital source USD changed")
    fully_open_path = Path(manifest["fully_open_geometry_report"])
    if sha(fully_open_path) != manifest["fully_open_geometry_sha256"]:
        raise ValueError("The fully-open door report changed")
    fully_open = json.loads(fully_open_path.read_bytes())
    for record in (manifest, report, fully_open):
        for name, expected in record["source_sha256"].items():
            folder = _ijcas_path(workspace, 'isaac/scripts' if name.startswith('isaac_') else 'studies/IJCAS/scripts')
            if sha(folder / name) != expected:
                raise ValueError("A recorded source changed: " + name)
    audit = json.loads(Path(manifest["source_audit"]).read_bytes())
    runpy.run_path(str(_ijcas_path(workspace, 'studies/IJCAS/scripts/hospital_h3_flush_geometry.py')))["verify_variant"](audit)
    return path, manifest, report, audit


def floor_coverage(world_corners, start, end, footprint):
    """Analytic coverage of the full straight component by ONE flat solid box.

    This checks authored support geometry, NOT dynamic contact or traversal.
    Reject rotated/sloped/gapped/noncanonical support rather than using its AABB.
    """
    if len(world_corners) != 8 or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in world_corners):
        raise ValueError("Need eight finite live cube corners")
    lo = [min(p[i] for p in world_corners) for i in range(3)]
    hi = [max(p[i] for p in world_corners) for i in range(3)]
    expected = list(itertools.product(*[(lo[i], hi[i]) for i in range(3)]))
    if any(not any(max(abs(p[i]-q[i]) for i in range(3)) <= 2e-5 for p in world_corners) for q in expected):
        raise ValueError("Support must be an axis-aligned flat box, not a rotated/sloped AABB")
    if (abs(hi[2]-.001) > 2e-5 or any(abs((hi[i]-lo[i])-size) > 2e-5 for i, size in enumerate((20.,20.,.1)))):
        raise ValueError("The common 20 x 20 x 0.1 m floor/top baseline changed")
    if (len(start) != 3 or len(end) != 3 or not all(math.isfinite(v) for v in (*start,*end))
            or abs(start[1]-end[1]) > 1e-8 or abs(start[2]-math.pi) > 1e-8 or abs(end[2]-math.pi) > 1e-8
            or start[0] <= end[0]):
        raise ValueError("Expected the frozen straight westward COMPONENT, not a turning mission")
    if footprint != {"length_m":1.63, "width_m":.965, "clearance_m":.01}:
        raise ValueError("Common full-envelope dimensions changed")
    hx = footprint["length_m"]/2 + footprint["clearance_m"]
    hy = footprint["width_m"]/2 + footprint["clearance_m"]
    required_lo = [min(start[0],end[0])-hx, start[1]-hy]
    required_hi = [max(start[0],end[0])+hx, start[1]+hy]
    if any(required_lo[i] < lo[i]+.001 or required_hi[i] > hi[i]-.001 for i in range(2)):
        raise ValueError("The solid support does not cover the entire expanded component corridor")
    return {"authored_single_solid_cube_covers_whole_component":True,
        "support_bounds_m":{"minimum_m":lo,"maximum_m":hi},
        "required_component_xy_bounds_m":{"minimum_m":required_lo,"maximum_m":required_hi},
        "flat_support_top_z_m":hi[2],"PhysX_contact_along_component_verified":False}


def live_geometry_check(stage, manifest, report, audit, disabled_floors=()):
    """Read-only live checks, allowing ONLY the enumerated flat duplicate floors."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    if Path(stage.GetRootLayer().realPath).resolve() != Path(manifest["source_stage"]).resolve():
        raise ValueError("Wrong live hospital source stage")
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) > 1e-9 or str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        raise ValueError("Expected metre-based Z-up stage")
    marker = stage.GetPrimAtPath(ENVIRONMENT)
    if (not marker.IsValid() or marker.GetCustomDataByKey("condition") != CONDITION
            or marker.GetCustomDataByKey("geometry_report_sha256") != manifest["flush_geometry_sha256"]
            or marker.GetCustomDataByKey("setup_manifest_sha256") != sha(Path(marker.GetCustomDataByKey("setup_manifest")))):
        raise ValueError("Live flush environment marker/evidence mismatch")
    door = stage.GetPrimAtPath("/World/IJCAS_HospitalH3DoorState")
    if (not door.IsValid() or door.GetCustomDataByKey("condition") != manifest["door_condition"]
            or door.GetCustomDataByKey("geometry_audit_sha256") != manifest["fully_open_geometry_sha256"]):
        raise ValueError("Live fully-open door does not match this flush setup")
    threshold = stage.GetPrimAtPath(THRESHOLD)
    if not threshold.IsValid() or threshold.IsActive():
        raise ValueError("Selected threshold must remain visually AND physically inactive")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
    for asset in audit["local_assets"]:
        prim = stage.GetPrimAtPath(asset["path"])
        if not prim.IsValid() or not prim.IsActive():
            raise ValueError("Another hospital asset is missing/inactive: " + asset["path"])
        bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if any(not np.allclose(actual,asset["world_aligned_visible_bound"][key],rtol=0,atol=2e-5)
                for key,actual in (("minimum_m",tuple(bounds.GetMin())),("maximum_m",tuple(bounds.GetMax())))):
            raise ValueError("Another hospital asset moved: " + asset["path"])
        if str(UsdGeom.Imageable(prim).ComputeVisibility()) != asset["computed_visibility"]:
            raise ValueError("Another hospital asset visibility changed: " + asset["path"])
        for path in asset["collision"]["collision_prim_paths"]:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError("Missing hospital collider: " + path)
            disabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
            if disabled != (path in disabled_floors):
                raise ValueError("Unexpected hospital collider flag: " + path)
    verify_files(report["original_threshold_results_preserved"])
    if sha(report["source_stage"]) != report["source_stage_sha256"]:
        raise ValueError("Source USD changed")
    return True


def support_readback(stage, manifest):
    """Read actual USD values, not a report of requested constants."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cube_prim = stage.GetPrimAtPath(GROUND)
    if (not cube_prim.IsA(UsdGeom.Cube) or cube_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or UsdPhysics.CollisionAPI(cube_prim).GetCollisionEnabledAttr().Get() is False):
        raise ValueError("Need one enabled static canonical Cube support")
    cube = UsdGeom.Cube(cube_prim)
    half = float(cube.GetSizeAttr().Get()) / 2
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(cube_prim)
    corners = [list(matrix.Transform(Gf.Vec3d(*p))) for p in itertools.product((-half,half),repeat=3)]
    coverage = floor_coverage(corners,manifest["start_xyyaw"],manifest["component_end_xyyaw"],manifest["footprint"])
    material = stage.GetPrimAtPath(MATERIAL)
    physical = UsdPhysics.MaterialAPI(material)
    physx = PhysxSchema.PhysxMaterialAPI(material)
    values = {"static_friction":physical.GetStaticFrictionAttr().Get(),"dynamic_friction":physical.GetDynamicFrictionAttr().Get(),
        "restitution":physical.GetRestitutionAttr().Get(),"contact_stiffness_n_m":physx.GetCompliantContactStiffnessAttr().Get(),
        "contact_damping_n_s_m":physx.GetCompliantContactDampingAttr().Get(),
        "acceleration_spring":physx.GetCompliantContactAccelerationSpringAttr().Get(),
        "friction_combine":str(physx.GetFrictionCombineModeAttr().Get())}
    expected = {"static_friction":.8,"dynamic_friction":.6,"restitution":0.,"contact_stiffness_n_m":100000.,
        "contact_damping_n_s_m":1500.,"acceleration_spring":False,"friction_combine":"max"}
    def same(actual, wanted):
        if isinstance(wanted, (bool, str)):
            return actual == wanted
        return actual is not None and math.isclose(actual,wanted,rel_tol=1e-6,abs_tol=1e-6)
    if not all(same(values[key],value) for key,value in expected.items()):
        raise ValueError("H1/H2 contact baseline mismatch: " + str(values))
    joints = {}
    for name, expected in JOINT_SETTINGS.items():
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(ROBOT+"/joints/"+name),"angular")
        actual = (drive.GetStiffnessAttr().Get(),drive.GetDampingAttr().Get(),drive.GetMaxForceAttr().Get())
        if actual != expected or str(drive.GetTypeAttr().Get()) != "force":
            raise ValueError("H1/H2 joint baseline mismatch: " + name)
        if drive.GetTargetPositionAttr().Get() != 0. or drive.GetTargetVelocityAttr().Get() != 0.:
            raise ValueError("Nonzero authored joint target: " + name)
        joints[name] = list(actual)
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError("Expected one PhysicsScene")
    scene = UsdPhysics.Scene(scenes[0])
    hz = PhysxSchema.PhysxSceneAPI(scenes[0]).GetTimeStepsPerSecondAttr().Get()
    if (hz != 120 or not same(scene.GetGravityMagnitudeAttr().Get(),9.81)
            or tuple(scene.GetGravityDirectionAttr().Get()) != (0.,0.,-1.)):
        raise ValueError("Physics frequency/gravity baseline changed")
    # Bindings must resolve to the canonical material for floor + all six wheels.
    from pxr import UsdShade
    for path in (GROUND,*(ROBOT+"/"+name for name in WHEELS)):
        bound, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial("physics")
        if not bound or str(bound.GetPath()) != MATERIAL:
            raise ValueError("Unexpected resolved wheel/support material: " + path)
    from pxr import Usd
    bodies = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mass is None or not math.isfinite(mass) or mass <= 0:
                raise ValueError("Missing positive body mass")
            if PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get() is True:
                raise ValueError("Robot body gravity disabled")
            bodies.append({"path":str(prim.GetPath()),"mass_kg":float(mass)})
    if len(bodies) != 13 or abs(sum(row["mass_kg"] for row in bodies)-265.) > .01:
        raise ValueError("Expected the common 265 kg / 13-body baseline")
    return {"support_coverage":coverage,"material_readback":values,
        "joint_settings_USD_angular_units":joints,"physics_hz":int(hz),"gravity_m_s2":9.81,
        "rigid_bodies":bodies,"simulation_mass_kg":sum(row["mass_kg"] for row in bodies)}
