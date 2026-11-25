"""Authorized H3 archival -> fresh H4 staging, never static/drive admission.

The old articulation is retired only AFTER a durable actual-state archive.
A separate reference supplies the unselected source-authored joint seed. USD
composition can be tested without starting Kit or a physics timeline.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_support_design_v2 as support
from hospital_geometry_tools import footprint_polygon, pose_collision


ROOT, GATE = support.ROOT, support.GATE
completion = support.floor.inventory.retained.completion
OLD_ROBOT = '/World/patient_transport_lab_screening'
ROBOT = '/World/patient_transport_h4_cohort_01'
MARKER = '/World/IJCAS_HospitalH4Cohort01'
ASSET = _ijcas_path(ROOT, 'isaac/assets/patient_transport_ijcas/patient_transport_ijcas.usd')
ASSET_SHA = 'ecca5698c07f70a39d242054573a1c3df7454ceff2df3c44af4ff821fa3eb74a'
SETUP = support.floor.inventory.SETUP
MAP_REVIEW = GATE/'hospital_h4_complete_map_support/review_r01/complete_map_support_review.json'
MAP = MAP_REVIEW.parent/'complete_authored_AABB_occupancy.npz'
MODELS = GATE/'hospital_h4_shared_model_seed_screen/screen_r01/shared_model_seed_screen.json'
OUTPUT = GATE/'hospital_h4_cohort_preparation/cohort_01'
PREFLIGHT = OUTPUT/'preflight/cohort_preflight.json'
ARCHIVE = OUTPUT/'archived_H3_before_Stop'
LIVE = OUTPUT/'live_h4_preparation.json'
FAILURE = OUTPUT/'preparation_failure.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_prepare_ijcas_hospital_h4_fresh_cohort.py')
SOURCE = support.floor.inventory.retained.SOURCE
SOURCE_SHA = support.floor.inventory.retained.SOURCE_SHA
VERSION = 'H4_authorized_H3_archive_then_fresh_source_reference_staging_v1'
MATERIAL = '/World/PatientTransportLabGround/WheelGroundContact'
GROUND = '/World/PatientTransportLabGround/CollisionBox'
WHEELS = ('front_active_drive_wheel_link', 'rear_active_drive_wheel_link',
    'front_left_caster_wheel_link', 'front_right_caster_wheel_link',
    'rear_left_caster_wheel_link', 'rear_right_caster_wheel_link')
JOINT_SETTINGS = {'base_link_to_front_active_steering': (30., 5., 250.),
    'base_link_to_rear_active_steering': (30., 5., 250.),
    'front_active_steering_to_drive_wheel': (0., 50., 500.),
    'rear_active_steering_to_drive_wheel': (0., 50., 500.)}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def verify(pins):
    completion.verify_files(pins)


def support_covers(design):
    bound = design['canonical_unchanged_support_bound']
    return [support.rectangle(bound['minimum_m'][0], bound['maximum_m'][0],
        bound['minimum_m'][1], bound['maximum_m'][1]), *design['nonoverlapping_support_rectangles']]


def support_area_missing(pose, covers):
    return sum(abs(support.area(p)) for p in support.uncovered([footprint_polygon(pose, .010)], covers))


def sampled_support_screen(models, design):
    """5 mm / 0.25 degree samples, NOT an analytic continuous certificate."""
    covers = support_covers(design)
    results = []
    for method in models['methods']:
        poses = [s['world_xyyaw'] for s in method['selected_states']]
        if not method['plan_success'] or len(poses) < 2:
            raise ValueError('All four selected native MODEL paths are required')
        maximum = support_area_missing(poses[-1], covers)
        count = 1
        for a, b in zip(poses, poses[1:]):
            angle = math.atan2(math.sin(b[2]-a[2]), math.cos(b[2]-a[2]))
            n = max(1, math.ceil(math.hypot(b[0]-a[0], b[1]-a[1])/.005),
                math.ceil(abs(angle)/math.radians(.25)))
            for i in range(n):
                t = i/n
                pose = [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), a[2]+t*angle]
                maximum = max(maximum, support_area_missing(pose, covers))
                count += 1
        results.append(dict(method=method['method'], pose_samples=count,
            maximum_uncovered_area_m2=maximum, sampled_full_footprint_support_covered=maximum <= 1e-9))
    return results


def prepare():
    design = support.load_design()
    actual = json.loads(support.LIVE.read_bytes())
    setup = json.loads(SETUP.read_bytes())
    map_review = json.loads(MAP_REVIEW.read_bytes())
    models = json.loads(MODELS.read_bytes())
    for value in (actual, setup, map_review, models):
        verify(value['input_sha256'])
        if 'output_sha256' in value:
            verify(value['output_sha256'])
    if (completion.sha(ASSET) != ASSET_SHA or completion.sha(SOURCE) != SOURCE_SHA
            or actual['existing_H3_body_12_joints_gains_support_root_source_and_timeline_unchanged'] is not True
            or actual['only_new_session_subtree'] != support.PATCH_ROOT
            or actual['actual_top_union_subtraction_uncovered_area_m2'] > 1e-9
            or actual['actual_physics_material'] != MATERIAL
            or setup['task_pose_selection_uses_method_performance_or_measured_H3_endpoint'] is not False
            or setup['footprint'] != dict(length_m=1.630, width_m=.965, clearance_m=.010)):
        raise ValueError('Unchanged original assets, actual H4 support and independent task rule required')
    with np.load(MAP) as grid:
        occupied, origin, resolution = grid['occupied'], grid['origin_xy_m'], float(grid['resolution_m'])
        for key in ('start_xyyaw', 'goal_xyyaw'):
            if pose_collision(occupied, setup[key], origin, resolution, .010):
                raise ValueError('H4 endpoint full-envelope collision: '+key)
            if support_area_missing(setup[key], support_covers(design)) > 1e-9:
                raise ValueError('H4 endpoint full support is unavailable: '+key)
    native_support = sampled_support_screen(models, design)
    if len(native_support) != 4 or not all(r['sampled_full_footprint_support_covered'] for r in native_support):
        raise ValueError('Selected native MODEL path extends outside the authored support')
    sources = (Path(__file__), ENTRY, _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_geometry_tools.py'))
    inputs = {**design['input_sha256'], **actual['input_sha256'], **setup['input_sha256'],
        **setup['output_sha256'], **map_review['input_sha256'], **models['input_sha256'],
        **{str(path): completion.sha(path) for path in
            (*sources, ASSET, SOURCE, SETUP, support.TARGET, support.LIVE, MAP_REVIEW, MAP, MODELS)}}
    verify(inputs)
    return canonical(dict(version=VERSION, status='H4_NEW_COHORT_PREPARED_STATIC_AND_DRIVE_NOT_ADMITTED',
        user_authorization='Archive actual H3 FINAL first, then Stop and stage a NEW H4 cohort; no Play or driving',
        task_candidate=str(SETUP), start_xyyaw=setup['start_xyyaw'], goal_xyyaw=setup['goal_xyyaw'],
        new_robot_root=ROBOT, retired_H3_robot_root=OLD_ROBOT, robot_asset=str(ASSET),
        robot_asset_sha256=ASSET_SHA, original_source_reference_prim=OLD_ROBOT,
        prescribed_initial_state='source-authored reference only; no caster selection or joint-state assignments',
        all_twelve_joint_seed_and_link_relative_transforms_must_match_source=True,
        fresh_static_support_contact_test_required_before_any_driving=True,
        only_after_durable_H3_actual_state_archive_may_timeline_Stop=True,
        current_H3_physical_continuation_ends_after_archival_and_Stop=True,
        archive_is_evidence_NOT_an_admitted_live_state_restore_recipe=True,
        old_H3_PASS_FAIL_HOLDs_and_original_source_remain_immutable=True,
        allowed_Session_changes=[OLD_ROBOT+' (retired/inactive; Stop may restore USD poses)', ROBOT, MARKER],
        old_floor_patch_walls_cart_stairs_materials_scene_unchanged=True,
        supports=[GROUND, support.PATCH_MESH], support_top_m=design['top_height_m'],
        simulation_mass_kg=265., rigid_body_count=13, physics_hz=120,
        active_joint_settings_USD_angular_units=JOINT_SETTINGS,
        selected_native_model_path_sampled_support=native_support,
        sampled_support_check_spacing_m=.005, sampled_support_heading_spacing_deg=.25,
        continuous_collision_or_cooked_equivalence_proven=False,
        prior_whole_H3_MODEL_PASS_count=2, prior_whole_H3_MODEL_case_count=8,
        independent_fresh_frozen_whole_H3_repeats=0,
        automatic_Play_or_static_trial_authorized=False, actual_drive_admission=False,
        CICS_runtime_integration=False, real_robot_validation_required=False,
        use_for_final_statistics=False, new_PhysX_executed=False,
        source_sha256={p.name: completion.sha(p) for p in sources}, input_sha256=inputs))


def load_preflight():
    report = json.loads(PREFLIGHT.read_bytes())
    verify(report['input_sha256'])
    if report != prepare():
        raise ValueError('H4 cohort preflight changed; no substitute task, source or settings')
    return report


def joint_seed(stage, root):
    """Read complete authored state attributes; absence is preserved as absence."""
    from pxr import Usd, UsdPhysics
    result = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            result[prim.GetName()] = {a.GetName(): a.Get() for a in prim.GetAttributes()
                if a.GetName().startswith('state:')}
    if len(result) != 12:
        raise ValueError('Expected twelve source-authored revolute joints')
    return result


def validate_new_robot(stage, source_stage, start):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    robot = stage.GetPrimAtPath(ROBOT)
    if not robot.IsActive() or joint_seed(stage, ROBOT) != joint_seed(source_stage, OLD_ROBOT):
        raise ValueError('Fresh source joint seed changed; do not assign/reset caster or joint positions')
    cache, source_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    base_matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/base_link'))
    position = base_matrix.ExtractTranslation()
    forward = base_matrix.TransformDir((1., 0., 0.))
    actual = [position[0], position[1], math.atan2(forward[1], forward[0])]
    if max(abs(a-b) for a, b in zip(actual, start)) > 2e-5:
        raise ValueError('Fresh H4 placement differs from fixed geometry-derived start')
    source_base_inverse = source_cache.GetLocalToWorldTransform(source_stage.GetPrimAtPath(OLD_ROBOT+'/base_link')).GetInverse()
    base_inverse = base_matrix.GetInverse()
    bodies, joints, colliders = [], [], []
    for prim in Usd.PrimRange(robot):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            source_prim = source_stage.GetPrimAtPath(OLD_ROBOT+path[len(ROBOT):])
            a = cache.GetLocalToWorldTransform(prim)*base_inverse
            b = source_cache.GetLocalToWorldTransform(source_prim)*source_base_inverse
            if not np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=2e-5):
                raise ValueError('A source link/caster relative pose changed: '+path)
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mass is None or not math.isfinite(mass) or mass <= 0:
                raise ValueError('Positive authored mass required: '+path)
            bodies.append(dict(path=path, mass_kg=float(mass)))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            for relation in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel()):
                targets = relation.GetTargets()
                if len(targets) != 1 or not str(targets[0]).startswith(ROBOT+'/'):
                    raise ValueError('Reference joint relationship was not remapped to the NEW robot')
            joints.append(prim.GetName())
        if prim.GetTypeName() == 'OmniGraph':
            raise ValueError('No new runtime graph/controller is allowed')
    if len(bodies) != 13 or len(joints) != 12 or abs(sum(b['mass_kg'] for b in bodies)-265.) > .01:
        raise ValueError('The 265 kg / 13-body / twelve-joint baseline changed')
    for name in WHEELS:
        body = stage.GetPrimAtPath(ROBOT+'/'+name)
        shapes = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        if len(shapes) != 1 or not shapes[0].IsA(UsdGeom.Capsule):
            raise ValueError('Original single Capsule wheel geometry required')
        source_shape = source_stage.GetPrimAtPath(OLD_ROBOT+str(shapes[0].GetPath())[len(ROBOT):])
        capsule, original = UsdGeom.Capsule(shapes[0]), UsdGeom.Capsule(source_shape)
        if (capsule.GetRadiusAttr().Get() != original.GetRadiusAttr().Get()
                or capsule.GetHeightAttr().Get() != original.GetHeightAttr().Get()
                or capsule.GetAxisAttr().Get() != original.GetAxisAttr().Get()):
            raise ValueError('Source wheel collider dimensions/axis changed')
        material, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial('physics')
        if not material or str(material.GetPath()) != MATERIAL:
            raise ValueError('Original canonical wheel material required')
        colliders.append(str(shapes[0].GetPath()))
    for name, expected in JOINT_SETTINGS.items():
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(ROBOT+'/joints/'+name), 'angular')
        actual_settings = (drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get())
        if (actual_settings != expected or drive.GetTypeAttr().Get() != 'force'
                or drive.GetTargetPositionAttr().Get() != 0. or drive.GetTargetVelocityAttr().Get() != 0.):
            raise ValueError('Original active joint settings/neutral targets changed')
    return dict(start_xyyaw=list(actual), base_z_m=float(position[2]), rigid_bodies=bodies,
        simulation_mass_kg=sum(b['mass_kg'] for b in bodies), revolute_joint_names=sorted(joints),
        actual_source_authored_joint_seed=joint_seed(stage, ROBOT),
        all_link_relative_poses_match_source=True, six_wheel_capsule_paths=colliders,
        all_joint_body_relationships_remapped_to_new_robot=True,
        static_support_or_drive_validated=False)


def author_new_robot(stage, start):
    """Session caller only; creates one NEW reference, never changes the seed."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    if stage.GetPrimAtPath(ROBOT).IsValid():
        raise ValueError('New H4 robot root already exists; no overwrite or retry')
    source = Usd.Stage.Open(str(ASSET))
    robot = stage.DefinePrim(ROBOT, 'Xform')
    if not robot.GetReferences().AddReference(str(ASSET), OLD_ROBOT):
        raise ValueError('Cannot compose unchanged robot reference')
    base = stage.GetPrimAtPath(ROBOT+'/base_link')
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(base)
    p = matrix.ExtractTranslation(); forward = matrix.TransformDir(Gf.Vec3d(1., 0., 0.))
    yaw = start[2]-math.atan2(forward[1], forward[0])
    xform = UsdGeom.Xformable(robot)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, 'hospitalH4Fresh').Set(Gf.Vec3d(
        start[0]-(math.cos(yaw)*p[0]-math.sin(yaw)*p[1]),
        start[1]-(math.sin(yaw)*p[0]+math.cos(yaw)*p[1]), 0.))
    xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, 'hospitalH4Fresh').Set(math.degrees(yaw))
    material = UsdShade.Material.Get(stage, MATERIAL)
    if not material:
        raise ValueError('Canonical material must already exist; never create a substitute')
    mass = UsdPhysics.MassAPI.Apply(base)
    prior_mass, inertia = mass.GetMassAttr().Get(), mass.GetDiagonalInertiaAttr().Get()
    if prior_mass is None or prior_mass <= 0 or inertia is None:
        raise ValueError('Original measured mass/inertia unavailable')
    mass.CreateMassAttr().Set(245.)
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v)*245./prior_mass for v in inertia)))
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0., 0., .270))
    for prim in Usd.PrimRange(robot):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(False)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
    for name in WHEELS:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(ROBOT+'/'+name)).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    for name, (kp, kd, effort) in JOINT_SETTINGS.items():
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(ROBOT+'/joints/'+name), 'angular')
        if not drive:
            raise ValueError('Source angular drive missing: '+name)
        drive.CreateTypeAttr().Set('force')
        drive.CreateStiffnessAttr().Set(kp); drive.CreateDampingAttr().Set(kd)
        drive.CreateMaxForceAttr().Set(effort)
        drive.CreateTargetPositionAttr().Set(0.); drive.CreateTargetVelocityAttr().Set(0.)
    return validate_new_robot(stage, source, start)


def protected_session_text(text):
    """Ignore ONLY the authorized old/new robot roots and new cohort marker."""
    from pxr import Sdf
    copy = Sdf.Layer.CreateAnonymous('H4_protected_session_delta')
    if not copy.ImportFromString(text):
        raise ValueError('Cannot inspect isolated Session layer copy')
    for path in (OLD_ROBOT, ROBOT, MARKER):
        if copy.GetPrimAtPath(path):
            edit = Sdf.BatchNamespaceEdit(); edit.Add(path, Sdf.Path.emptyPath)
            if not copy.Apply(edit):
                raise ValueError('Cannot compare authorized Session change scope')
    return copy.ExportToString()
