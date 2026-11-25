"""Fresh H4 Session reconstruction after application exit, NO physical replay.

Recreate the previously audited nominal support/material/source-authored actor
at the unchanged original start. Reuse immutable constructors and environment
readers, not archived dynamic transforms or solver/contact state. Old files and
the contact-aborted route remain unchanged; no static/drive authority is added.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import hospital_h4_restart_review_v2 as restart
import hospital_h4_cohort_preparation_v4 as original
import hospital_h4_static_gate as stationary
import hospital_h4_root_difference_readback as fields


ROOT, GATE = original.ROOT, original.GATE
SOURCE, SOURCE_SHA, ASSET, ASSET_SHA = original.SOURCE, original.SOURCE_SHA, original.ASSET, original.ASSET_SHA
# The USD namespace is recreated in a NEW application/Session, not continued.
# Result/cohort identity is deliberately different, despite the same prim path.
ROBOT, MARKER, OLD_ROBOT = original.ROBOT, original.MARKER, original.OLD_ROBOT
GROUND, MATERIAL, WHEELS = original.GROUND, original.MATERIAL, original.WHEELS
COHORT = 'cohort_02_reopened_r01'
OUTPUT = GATE/'hospital_h4_cohort_preparation'/COHORT
PREFLIGHT = OUTPUT/'preparation_protocol.json'
LIVE, FAILURE, RESERVATION = (OUTPUT/name for name in ('live_h4_preparation.json', 'preparation_failure.json', 'single_staging_reserved.json'))
ENTRY = ROOT/'isaac/scripts/isaac_prepare_ijcas_hospital_h4_reopened_cohort.py'
VERSION = 'H4_reopened_source_fresh_nominal_Session_cohort_02_preparation_only_v1'
sha, verify = restart.sha, restart.verify
write_exclusive = restart.write_exclusive
GROUND_ROOT = '/World/PatientTransportLabGround'
SOURCE_ROBOT = '/World/patient_transport'


def load_source_readback():
    report = json.loads(restart.RESTART.read_bytes()); verify(report['input_sha256'])
    required = ('original_source_file_unchanged', 'new_session_not_an_actual_state_restore',
        'every_recorded_enabled_Hospital_ROI_collider_available',
        'every_recorded_enabled_Hospital_ROI_bound_unchanged', 'contact_collider_authored_mesh_matches_recorded',
        'all_ten_previously_disabled_collider_flags_recorded_without_invented_old_bounds',
        'Root_Session_and_timeline_unchanged_by_readback')
    if (report['version'] != restart.VERSION
            or report['status'] != 'FRESH_SOURCE_AUTHORED_READBACK_COMPLETE_SETUP_PENDING'
            or set(report['checks']) != set(required) or not all(v is True for v in report['checks'].values())
            or report['timeline'] != dict(playing=False, stopped=True, current_time_s=0.)
            or report['actual_static_or_drive_admission'] is not False
            or report['old_solver_contact_cache_or_physical_state_restored'] is not False
            or report['existing_graph_paths']):
        raise ValueError('Exact successful actual v2 fresh-source readback required')
    if len(report['existing_rigid_body_paths']) != 13 or any(
            not p.startswith(SOURCE_ROBOT+'/') for p in report['existing_rigid_body_paths']):
        raise ValueError('Only the thirteen-body source robot may be replaced by the new nominal H4 actor')
    return report


def prepare_protocol():
    source = load_source_readback()
    design = original.support.load_design()
    old_path = json.loads(restart.original.RESULT.read_bytes())
    if old_path['verdict'] != 'FAIL' or sha(SOURCE) != SOURCE_SHA or sha(ASSET) != ASSET_SHA:
        raise ValueError('Preserve the old H4 FAIL and exact immutable source/robot asset')
    setup = json.loads(original.SETUP.read_bytes())
    disabled = sorted(r['path'] for r in source['collision_comparisons'] if r['recorded_classification'] == 'disabled')
    if len(disabled) != 10 or sorted(source['previously_disabled_colliders_currently_enabled']) != disabled:
        raise ValueError('Exactly the ten original duplicate floor flags must be bound explicitly')
    source_files = (Path(__file__), ENTRY, Path(original.__file__), Path(stationary.__file__), Path(fields.__file__),
        ROOT/'studies/IJCAS/scripts/hospital_h4_cohort_preparation.py')
    return dict(version=VERSION, cohort=COHORT, status='FRESH_NOMINAL_SESSION_SETUP_PREPARED_NOT_STATIC_OR_DRIVE',
        actual_source_readback=str(restart.RESTART), original_start_xyyaw=setup['start_xyyaw'],
        unchanged_goal_xyyaw=setup['goal_xyyaw'], source_robot_to_retire=SOURCE_ROBOT,
        new_robot_root=ROBOT, reused_USD_namespace_is_NOT_same_actor_or_physical_continuation=True,
        empty_inactive_old_H3_root_is_namespace_placeholder_NOT_H3_state_restore=True,
        reauthor_same_canonical_ground_and_same_H4_local_support=True,
        support_design=str(original.support.TARGET), support_top_m=design['top_height_m'],
        disabled_duplicate_floor_paths=disabled,
        wall_cart_stair_and_contact_collider_flags_geometry_offsets_unchanged=True,
        robot_mass_kg=265., physics_hz=120, source_authored_twelve_joint_seed_NOT_assigned=True,
        original_motor_material_and_support_settings_unchanged=True,
        old_contact_FAIL_and_static_FAIL_and_H3_archives_preserved=True,
        source_USD_session_snapshot_or_saved_dynamic_state_imported=False,
        actual_new_static_trial_required_before_motion=True, actual_static_or_drive_admission=False,
        known_contact_map_height_gap_remains_unresolved_for_drive=True,
        Play_Stop_Pause_reset_or_physics_initialization_permitted=False,
        CICS_runtime_integration=False, real_robot_required=False, use_for_final_statistics=False,
        input_sha256={**source['input_sha256'],
            **{str(p): sha(p) for p in (*source_files, restart.RESTART, original.support.TARGET, original.support.LIVE, original.SETUP)}})


def load_protocol():
    protocol = json.loads(PREFLIGHT.read_bytes()); verify(protocol['input_sha256'])
    if protocol != prepare_protocol():
        raise ValueError('Fresh H4 reconstruction protocol changed')
    return protocol


def session_delta(before, after, floor_paths, deinstanced_paths, scene_path):
    """Field-level guard outside newly constructed, separately validated roots."""
    from pxr import Sdf
    a, b = fields.layer_snapshot(before), fields.layer_snapshot(after)
    new_roots = (ROBOT, MARKER, OLD_ROBOT, GROUND_ROOT, original.support.PATCH_ROOT)
    properties = {p+'.physics:collisionEnabled' for p in floor_paths} | {
        scene_path+'.physics:gravityDirection', scene_path+'.physics:gravityMagnitude',
        scene_path+'.physxScene:timeStepsPerSecond'}
    parents = set()
    for path in (*new_roots, *properties, SOURCE_ROBOT, *deinstanced_paths):
        parents.update(str(p) for p in Sdf.Path(path).GetPrimPath().GetPrefixes())
    changes = []
    for path in sorted(set(a)|set(b)):
        if a.get(path) == b.get(path):
            continue
        p = Sdf.Path(path)
        if any(p.GetPrimPath().HasPrefix(Sdf.Path(root)) for root in new_roots):
            kind = 'new_validated_nominal_subtree'
        elif path in properties:
            kind = 'exact_floor_collision_flag_or_original_scene_setting'
        else:
            first, last = a.get(path, {}).get('info', {}), b.get(path, {}).get('info', {})
            changed = {key for key in set(first)|set(last) if first.get(key) != last.get(key)}
            allowed = {'specifier'} if path in parents else set()
            if path == SOURCE_ROBOT:
                allowed.add('active')
            if path in deinstanced_paths:
                allowed.add('instanceable')
            if not changed <= allowed or path not in parents:
                raise ValueError('Session delta exceeds explicit preparation scope: '+path)
            kind = 'necessary_ancestor_over_or_exact_retire_deinstance_metadata'
        changes.append(dict(path=path, change_kind=kind))
    return changes


def author_session(stage, protocol):
    """Caller must already bind a fresh STOPPED source and Session EditTarget."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError('Nominal setup must be authored ONLY in Session Layer')
    if any(stage.GetPrimAtPath(path).IsValid() for path in (ROBOT, MARKER, GROUND_ROOT, original.support.PATCH_ROOT)):
        raise ValueError('Fresh reconstructed namespace already exists; no overwrite/retry')
    old = stage.GetPrimAtPath(SOURCE_ROBOT)
    if not old.IsValid() or not old.IsActive():
        raise ValueError('Original source robot is not the recorded active actor')
    old.SetActive(False)
    # This is deliberately an EMPTY inactive namespace, not a revived H3 actor.
    stage.DefinePrim(OLD_ROBOT, 'Xform').SetActive(False)
    deinstanced = []
    for path in protocol['disabled_duplicate_floor_paths']:
        for prefix in Sdf.Path(path).GetPrefixes():
            prim = stage.GetPrimAtPath(prefix)
            if prim.IsValid() and prim.IsInstance():
                prim.SetInstanceable(False); deinstanced.append(str(prefix))
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('An exact duplicate-floor collider is not editable: '+path)
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    design = original.support.load_design()
    root = stage.DefinePrim(GROUND_ROOT, 'Xform')
    if not root.GetReferences().AddReference(str(ASSET), GROUND_ROOT):
        raise ValueError('Canonical ground reference failed')
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    local_box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False,
        ignoreVisibility=True).ComputeWorldBound(stage.GetPrimAtPath(GROUND)).ComputeAlignedRange()
    target = design['canonical_unchanged_support_bound']
    translation = [(target['minimum_m'][i]+target['maximum_m'][i]-local_box.GetMin()[i]-local_box.GetMax()[i])/2
        for i in range(3)]
    xf = UsdGeom.Xformable(root); xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, 'reopenedH4Nominal').Set(Gf.Vec3d(*translation))
    ground = stage.GetPrimAtPath(GROUND)
    UsdPhysics.CollisionAPI.Apply(ground).CreateCollisionEnabledAttr().Set(True)
    UsdGeom.Imageable(ground).MakeInvisible()
    material = UsdShade.Material.Get(stage, MATERIAL)
    if not material:
        raise ValueError('Original ground asset material branch is missing')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(.8); api.CreateDynamicFrictionAttr().Set(.6); api.CreateRestitutionAttr().Set(0.)
    physics_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physics_material.CreateCompliantContactAccelerationSpringAttr().Set(False)
    physics_material.CreateCompliantContactStiffnessAttr().Set(100000.)
    physics_material.CreateCompliantContactDampingAttr().Set(1500.)
    physics_material.CreateFrictionCombineModeAttr().Set('max')
    UsdShade.MaterialBindingAPI.Apply(ground).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    anchor = [design['patch_planar_polygon'][0][0], target['maximum_m'][1], design['top_height_m']]
    patch = UsdGeom.Xform.Define(stage, original.support.PATCH_ROOT)
    patch.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*anchor))
    mesh = UsdGeom.Mesh.Define(stage, original.support.PATCH_MESH)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*(point[i]-anchor[i] for i in range(3)))
        for point in design['patch_world_mesh']['world_points_m']]))
    mesh.CreateFaceVertexCountsAttr().Set(design['patch_world_mesh']['face_vertex_counts'])
    mesh.CreateFaceVertexIndicesAttr().Set(design['patch_world_mesh']['face_vertex_indices'])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr().Set(True); mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set('none')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    robot_readback = original.author_new_robot(stage, protocol['original_start_xyyaw'])
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('Exactly one original PhysicsScene required')
    scene = UsdPhysics.Scene(scenes[0])
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0., 0., -1.)); scene.CreateGravityMagnitudeAttr().Set(9.81)
    if not scenes[0].HasAPI(PhysxSchema.PhysxSceneAPI):
        raise ValueError('The original source PhysxSceneAPI must already exist')
    PhysxSchema.PhysxSceneAPI(scenes[0]).GetTimeStepsPerSecondAttr().Set(120)
    marker = UsdGeom.Xform.Define(stage, MARKER).GetPrim()
    for key, value in dict(status='reopened_H4_prepared_static_pending', cohort=COHORT, robot_root=ROBOT,
            preparation_protocol=str(PREFLIGHT), preparation_protocol_sha256=sha(PREFLIGHT),
            physics_preparation_result=str(LIVE), old_namespace_recreated_NOT_same_physical_actor=True).items():
        marker.SetCustomDataByKey(key, value)
    environment = stationary.validate_environment(stage)
    original.verify_actual_support(stage)
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=False,
        ignoreVisibility=True).ComputeWorldBound(ground).ComputeAlignedRange()
    if not all(np.allclose(actual, target[key], rtol=0, atol=1e-12) for actual, key in (
            (bound.GetMin(), 'minimum_m'), (bound.GetMax(), 'maximum_m'))):
        raise ValueError('Rebuilt canonical box bounds differ from the original nominal support')
    active_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(active_bodies) != 13 or any(not p.startswith(ROBOT+'/') for p in active_bodies):
        raise ValueError('Only the thirteen-body new H4 robot may be active')
    return dict(live_new_robot_readback=robot_readback, environment_readback=environment,
        deinstanced_floor_roots=sorted(set(deinstanced)), physics_scene=str(scenes[0].GetPath()),
        support_colliders=[GROUND, original.support.PATCH_MESH], support_top_m=design['top_height_m'],
        simulation_mass_kg=265., all_active_rigid_body_paths=active_bodies,
        source_authored_joint_seed_recorded_NOT_assigned=True,
        canonical_ground_world_bound=dict(minimum_m=list(map(float, bound.GetMin())), maximum_m=list(map(float, bound.GetMax()))))


def main():
    protocol = prepare_protocol()
    if PREFLIGHT.exists():
        load_protocol()
    else:
        write_exclusive(PREFLIGHT, protocol)
    print('=== H4 REOPENED NOMINAL SESSION PREPARATION READY (OFFLINE) ===')
    print('new result cohort:', COHORT)
    print('same source/reference/mass/material/gains/support/start/goal; NO old dynamic-state restoration')
    print('saved:', PREFLIGHT)
    print('Setup ONLY. No static or drive trial admitted; old low-collider route still HOLD.')


if __name__ == '__main__':
    main()
