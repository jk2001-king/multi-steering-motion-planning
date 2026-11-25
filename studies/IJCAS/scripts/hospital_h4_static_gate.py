"""H4-only stationary support contract and strict, root-specific contacts.

The old static evaluator is imported unchanged. The new reference root and
two exact support paths are explicit, never a blanket floor/obstacle exemption.
An offline protocol is NOT permission or proof that the actual robot is staged.
"""

import json
import math
from pathlib import Path

import numpy as np

import hospital_h4_cohort_preparation_v4 as cohort
import hospital_static_gate as original


ROOT, GATE, ROBOT, MARKER = cohort.ROOT, cohort.GATE, cohort.ROBOT, cohort.MARKER
VERSION = 'H4_fresh_source_static_original_limits_plus_sampled_cooked_support_queries_v1'
OUTPUT = GATE/'hospital_h4_static_contact/cohort_01'
RUN_ID = 'hospital_h4_static_r01'
RESULT, TRACE = OUTPUT/(RUN_ID+'.json'), OUTPUT/(RUN_ID+'.csv')
RESERVATION = OUTPUT/'single_attempt_reserved.json'
PROTOCOL = GATE/'hospital_h4_static_admission/design_r01/static_protocol.json'
ENTRY = ROOT/'isaac/scripts/isaac_run_ijcas_hospital_h4_static_gate.py'
SETTLE_S, RECORD_S, PHYSICS_HZ = 1., 2., 120
GUARDS = dict(maximum_control_gap_s=.25, maximum_wall_time_s=120.,
    maximum_xy_displacement_m=.030, maximum_abs_roll_pitch_deg=2.)
LIMITS = dict(original.LIMITS)
QUERY_LIMITS = dict(height_tolerance_m=.0005, horizontal_tolerance_m=.0005, minimum_upward_normal_z=.99)
BAD_CONTACTS = original.BAD_CONTACTS
WHEELS = dict(zip(original.WHEEL_LABELS, cohort.WHEELS))
SUPPORTS = (cohort.GROUND, cohort.support.PATCH_MESH)
KNOWN_CATEGORIES = frozenset(('unrelated', 'robot_internal', 'wheel_support', *BAD_CONTACTS))


def classify_contact(pair, disabled_legacy_supports=()):
    """Keep old semantics, but do NOT alias retired H3 paths into the new robot."""
    if len(pair) != 2 or any(not isinstance(p, str) or not p.startswith('/') for p in pair):
        return 'unclassified_external'
    belongs = [p.startswith(ROBOT+'/') for p in pair]
    if not any(belongs):
        return 'unrelated'
    if all(belongs):
        return 'robot_internal'
    robot, other = pair if belongs[0] else pair[::-1]
    wheel = any(robot == ROBOT+'/'+name or robot.startswith(ROBOT+'/'+name+'/') for name in cohort.WHEELS)
    if other in SUPPORTS:
        return 'wheel_support' if wheel else 'nonwheel_support'
    if other in disabled_legacy_supports:
        return 'unexpected_legacy_support'
    return 'hospital_obstacle' if other.startswith('/World/Hospital/') else 'unclassified_external'


def floor_query_samples(setup, support_receipt):
    """Ten endpoint points plus 16 interior samples of the four top triangles."""
    rows = []
    a, b = setup['footprint']['length_m']/2, setup['footprint']['width_m']/2
    for key in ('start_xyyaw', 'goal_xyyaw'):
        x, y, yaw = setup[key]
        for label, u, v in (('center', 0., 0.), ('front_left', a, b), ('front_right', a, -b),
                ('rear_left', -a, b), ('rear_right', -a, -b)):
            rows.append(dict(id=key+'/'+label, xy_m=[x+math.cos(yaw)*u-math.sin(yaw)*v,
                y+math.sin(yaw)*u+math.cos(yaw)*v], allowed_supports=list(SUPPORTS)))
    top = [p[:2] for p in support_receipt['actual_world_points_m'][:6]]
    triangles = cohort.support.ear_triangles(top)
    if len(triangles) != 4:
        raise ValueError('Exactly four actual H4 top triangles are required')
    weights = ((1/3, 1/3, 1/3), (.6, .2, .2), (.2, .6, .2), (.2, .2, .6))
    for index, face in enumerate(triangles):
        for number, barycentric in enumerate(weights):
            xy = [sum(weight*top[vertex][axis] for weight, vertex in zip(barycentric, face)) for axis in range(2)]
            rows.append(dict(id=f'patch_triangle_{index}/interior_{number}', xy_m=xy,
                allowed_supports=[cohort.support.PATCH_MESH]))
    return rows


def query_checks(specification, observed, top_height):
    """Reject missing/duplicate samples, unexpected supports, wrong heights/normals."""
    expected = {r['id']: r for r in specification}
    unique = len(observed) == len(expected) and len({r.get('id') for r in observed}) == len(expected)
    complete = unique and {r.get('id') for r in observed} == set(expected)
    valid = complete
    for row in observed:
        spec = expected.get(row.get('id'))
        if spec is None or row.get('errors') or not row.get('hit'):
            valid = False
            continue
        position, normal = row.get('position_m', []), row.get('normal', [])
        if (row.get('collision') not in spec['allowed_supports'] or len(position) != 3 or len(normal) != 3
                or not all(math.isfinite(float(v)) for v in (*position, *normal))
                or not np.allclose(position[:2], spec['xy_m'], rtol=0, atol=QUERY_LIMITS['horizontal_tolerance_m'])
                or abs(position[2]-top_height) > QUERY_LIMITS['height_tolerance_m']
                or normal[2] < QUERY_LIMITS['minimum_upward_normal_z']):
            valid = False
    return dict(all_prescribed_cooked_support_query_samples_recorded=bool(complete),
        all_sampled_cooked_support_hits_at_original_top_and_upward_normal=bool(valid))


def evaluate(rows, start, counts, errors, query_specification, observed_queries, top_height):
    result = original.evaluate_static(rows, start, 265., counts, errors)
    result['checks'].update(query_checks(query_specification, observed_queries, top_height))
    result['checks']['only_known_contact_categories_recorded'] = not (set(counts)-KNOWN_CATEGORIES)
    result['verdict'] = 'PASS' if all(result['checks'].values()) else 'FAIL'
    return result


def merge_evaluation(report, evaluated):
    """A later numeric PASS may NEVER erase an earlier safety/cleanup failure."""
    checks = dict(report['checks'])
    for key, value in evaluated['checks'].items():
        checks[key] = bool(value) and checks.get(key, True)
    report.update({key: value for key, value in evaluated.items() if key not in ('checks', 'verdict')})
    report['checks'] = checks
    report['verdict'] = 'PASS' if not report.get('error') and all(checks.values()) else 'FAIL'


def prepare_protocol():
    # Frozen input validation remains usable AFTER staging: never rerun the
    # fresh-cohort builder, which correctly refuses an already archived cohort.
    preflight = json.loads(cohort.PREFLIGHT.read_bytes())
    cohort.verify(preflight['input_sha256'])
    if (preflight['version'] != cohort.VERSION or preflight['footprint'] != cohort.FOOTPRINT
            or preflight['actual_expanded_footprint_area_m2'] != 1.650*.985
            or preflight['earlier_v1_v2_support_checks_invalidated_due_to_positional_length_argument'] is not True):
        raise ValueError('Corrected full-body/original-precision v4 preflight is required')
    setup = json.loads(cohort.SETUP.read_bytes())
    support = json.loads(cohort.support.LIVE.read_bytes())
    query_samples = floor_query_samples(setup, support)
    sources = (Path(__file__), ENTRY, Path(original.__file__), Path(cohort.__file__),
        ROOT/'studies/IJCAS/scripts/hospital_private_loader.py')
    return cohort.canonical(dict(version=VERSION, status='OFFLINE_STATIC_PROTOCOL_ONLY_ACTUAL_STAGING_REQUIRED',
        run_id=RUN_ID, new_robot_root=ROBOT, required_actual_preparation=str(cohort.LIVE),
        required_actual_archived_H3=str(cohort.ARCHIVE/'actual_H3_final_state.json'),
        start_xyyaw=setup['start_xyyaw'], goal_xyyaw=setup['goal_xyyaw'],
        settle_s=SETTLE_S, record_s=RECORD_S, physics_hz=PHYSICS_HZ,
        unchanged_original_static_limits=LIMITS, guards=GUARDS, supports=list(SUPPORTS),
        additional_sampled_query_limits=QUERY_LIMITS,
        sampled_cooked_support_queries=query_samples,
        support_top_m=preflight['support_top_m'],
        automatic_Play_1s_settling_2s_recording_then_Pause_only_after_user_launch=True,
        explicit_user_static_authorization_required=True,
        no_forward_yaw_CICS_Nav2_or_gain_material_caster_change=True,
        exactly_one_attempt_no_automatic_retry=True, actual_live_staging_verified=False,
        static_PASS_would_mean_start_support_AND_sampled_cooked_floor_hits_only=True,
        static_PASS_is_NOT_whole_path_contact_or_upper_body_collision_certification=True,
        H4_drive_admission=False, final_comparison_statistics=False, new_PhysX_executed=False,
        input_sha256={**preflight['input_sha256'], str(cohort.PREFLIGHT): cohort.completion.sha(cohort.PREFLIGHT),
            **{str(p): cohort.completion.sha(p) for p in sources}}))


def load_protocol():
    report = json.loads(PROTOCOL.read_bytes()); cohort.verify(report['input_sha256'])
    if report != prepare_protocol():
        raise ValueError('H4 static protocol or original limits changed; no live trial permitted')
    return report


def load_actual_preparation():
    """An offline flag/fixture must never be promoted to an actual admission."""
    if not cohort.LIVE.is_file() or cohort.FAILURE.exists():
        raise ValueError('Actual successful H4 v4 staging receipt is required; do not Play/reset to create it')
    prep = json.loads(cohort.LIVE.read_bytes()); cohort.verify(prep['input_sha256'])
    preflight = json.loads(cohort.PREFLIGHT.read_bytes()); cohort.verify(preflight['input_sha256'])
    required = dict(version=cohort.VERSION, status='H4_FRESH_SOURCE_ROBOT_STAGED_STOPPED_STATIC_PENDING',
        new_robot_root=ROBOT, retired_H3_robot_root=cohort.OLD_ROBOT, retired_H3_inactive=True,
        archive_durably_verified_before_Stop=True, original_H3_live_physical_continuation_ended_NOT_restored=True,
        task_candidate=str(cohort.SETUP), start_xyyaw=preflight['start_xyyaw'], goal_xyyaw=preflight['goal_xyyaw'],
        support_colliders=list(SUPPORTS), support_top_m=preflight['support_top_m'],
        old_ground_local_patch_hospital_obstacles_scene_and_materials_unchanged=True,
        Session_delta_limited_to_authorized_old_new_robot_and_marker=True,
        fresh_source_seed_recorded_NOT_assigned_or_angle_selected=True,
        old_H3_PASS_FAIL_HOLD_evidence_preserved=True, original_source_USD_saved=False,
        timeline='STOPPED', Play_or_static_or_drive_executed=False, actual_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False)
    if any(prep.get(key) != value for key, value in required.items()):
        raise ValueError('Exact actual corrected v4 staging, not a relabelled/failed/old cohort, is required')
    path = cohort.ARCHIVE/'actual_H3_final_state.json'
    if (prep['archived_actual_H3_state'] != str(path)
            or prep['archived_actual_H3_state_sha256'] != cohort.completion.sha(path)):
        raise ValueError('Actual H3 archive changed or was not retained')
    archive = json.loads(path.read_bytes()); cohort.verify(archive['input_sha256'])
    state = archive['actual_state']
    if (archive['status'] != 'ACTUAL_H3_FINAL_ARCHIVED_BEFORE_AUTHORIZED_STOP'
            or len(state['names']) != 12 or len(state['link_names']) != 13
            or len(state['all_thirteen_link_world_transforms_xyzw']) != 13
            or archive['prior_whole_H3_MODEL_PASS_count'] != 2
            or archive['prior_whole_H3_MODEL_case_count'] != 8
            or archive['independent_fresh_frozen_whole_H3_repeats'] != 0):
        raise ValueError('Original actual H3 full-state archive and scope are required')
    return prep


def validate_environment(stage):
    """Read-only original collider, support, material, scene and source checks."""
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    if (Path(stage.GetRootLayer().realPath).resolve() != cohort.SOURCE.resolve()
            or cohort.completion.sha(cohort.SOURCE) != cohort.SOURCE_SHA):
        raise ValueError('Original hospital source changed or was reopened/replaced')
    if any(p.GetTypeName() == 'OmniGraph' for p in stage.Traverse()):
        raise ValueError('A competing OmniGraph is active')
    old = stage.GetPrimAtPath(cohort.OLD_ROBOT)
    if not old.IsValid() or old.IsActive():
        raise ValueError('Old H3 actor must remain retired/inactive, never another active collider')
    design = cohort.support.load_design()
    receipt = json.loads(cohort.support.LIVE.read_bytes())
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(cohort.support.PATCH_MESH))
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    points = [list(map(float, matrix.Transform(Gf.Vec3d(*p)))) for p in mesh.GetPointsAttr().Get()]
    if (points != receipt['actual_world_points_m']
            or list(mesh.GetFaceVertexCountsAttr().Get()) != design['patch_world_mesh']['face_vertex_counts']
            or list(mesh.GetFaceVertexIndicesAttr().Get()) != design['patch_world_mesh']['face_vertex_indices']
            or str(UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get()) != 'none'):
        raise ValueError('Actual H4 support topology/height/shape changed')
    for path in SUPPORTS:
        prim = stage.GetPrimAtPath(path)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
        if (not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
                or prim.HasAPI(UsdPhysics.RigidBodyAPI) or not material or str(material.GetPath()) != cohort.MATERIAL):
            raise ValueError('Only the two audited static supports and canonical material are allowed')
    material = stage.GetPrimAtPath(cohort.MATERIAL)
    api, physx = UsdPhysics.MaterialAPI(material), PhysxSchema.PhysxMaterialAPI(material)
    actual = dict(static_friction=api.GetStaticFrictionAttr().Get(), dynamic_friction=api.GetDynamicFrictionAttr().Get(),
        restitution=api.GetRestitutionAttr().Get(), contact_stiffness_n_m=physx.GetCompliantContactStiffnessAttr().Get(),
        contact_damping_n_s_m=physx.GetCompliantContactDampingAttr().Get(),
        acceleration_spring=physx.GetCompliantContactAccelerationSpringAttr().Get(),
        friction_combine=str(physx.GetFrictionCombineModeAttr().Get()))
    expected = dict(static_friction=.8, dynamic_friction=.6, restitution=0., contact_stiffness_n_m=100000.,
        contact_damping_n_s_m=1500., acceleration_spring=False, friction_combine='max')
    for key, value in expected.items():
        if (actual[key] != value if isinstance(value, (bool, str)) else
                actual[key] is None or not math.isclose(actual[key], value, rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError('Original H1/H2 canonical material setting changed: '+key)
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('One original PhysicsScene required')
    scene = UsdPhysics.Scene(scenes[0])
    if (PhysxSchema.PhysxSceneAPI(scenes[0]).GetTimeStepsPerSecondAttr().Get() != PHYSICS_HZ
            or tuple(scene.GetGravityDirectionAttr().Get()) != (0., 0., -1.)
            or not math.isclose(scene.GetGravityMagnitudeAttr().Get(), 9.81, abs_tol=1e-6)):
        raise ValueError('Original gravity or 120 Hz physics changed')
    bodies = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if (mass is None or not math.isfinite(mass) or mass <= 0.
                    or PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get() is True):
                raise ValueError('Original positive masses and enabled robot gravity are required')
            bodies.append(dict(path=str(prim.GetPath()), mass_kg=float(mass)))
    if len(bodies) != 13 or abs(sum(p['mass_kg'] for p in bodies)-265.) > .01:
        raise ValueError('Original 265 kg / thirteen-body baseline changed')
    base_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(ROBOT+'/base_link'))
    if (base_mass.GetMassAttr().Get() != 245.
            or not np.allclose(base_mass.GetCenterOfMassAttr().Get(), (0., 0., .270), rtol=0, atol=1e-6)):
        raise ValueError('Original base mass or nominal CoM changed')
    for name, expected_settings in cohort.JOINT_SETTINGS.items():
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(ROBOT+'/joints/'+name), 'angular')
        if ((drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get()) != expected_settings
                or drive.GetTypeAttr().Get() != 'force'
                or drive.GetTargetPositionAttr().Get() != 0. or drive.GetTargetVelocityAttr().Get() != 0.):
            raise ValueError('Original authored neutral motor settings changed')
    for name in cohort.WHEELS:
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ROBOT+'/'+name)).ComputeBoundMaterial('physics')
        if not material or str(material.GetPath()) != cohort.MATERIAL:
            raise ValueError('Original wheel-floor material binding changed')
    inventory = json.loads(cohort.support.floor.inventory.TARGET.read_bytes())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)
    disabled = []
    checked = 0
    for row in inventory['collision_prims']:
        if row['classification'] not in ('enabled_in_ROI', 'disabled'):
            continue
        prim = stage.GetPrimAtPath(row['path'])
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('An original hospital/support collider disappeared')
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if row['classification'] == 'disabled':
            if enabled is not False:
                raise ValueError('Duplicate/legacy collider became enabled')
            disabled.append(row['path']); continue
        if not prim.IsActive() or enabled is False:
            raise ValueError('An original enabled collider was disabled')
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if (not np.allclose(bound.GetMin(), row['world_aligned_bound']['minimum_m'], rtol=0, atol=2e-5)
                or not np.allclose(bound.GetMax(), row['world_aligned_bound']['maximum_m'], rtol=0, atol=2e-5)):
            raise ValueError('An original wall/cart/terrain/support collider moved: '+row['path'])
        checked += 1
    return dict(unchanged_original_ROI_colliders_checked=checked, disabled_legacy_colliders=disabled,
        material_readback=actual, physics_hz=PHYSICS_HZ, rigid_bodies=bodies,
        simulation_mass_kg=sum(p['mass_kg'] for p in bodies),
        cooked_shape_equivalence_or_whole_route_support_claimed=False)
