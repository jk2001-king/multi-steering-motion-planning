"""Approved NEW environment preparation only; never a static/drive admission.

Archive the actual reopened-static state before ONE deferred Stop. Reuse the
unchanged nominal constructor at a new root; disable precisely four approved
subfloor meshes in Session. Old results and original source stay immutable.
Live imports deliberately exclude offline planners and fitting scripts.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import hashlib
import json
import math
import os
from pathlib import Path
from types import FunctionType

import numpy as np

import hospital_h4_cached_low_wall_review_v2 as cached
import hospital_h4_cohort_preparation as constructor
import hospital_h4_root_difference_readback as fields


static = cached.static
original = static.preparation.original
ROOT, GATE = cached.ROOT, cached.GATE
SOURCE, SOURCE_SHA, ASSET, ASSET_SHA = original.SOURCE, original.SOURCE_SHA, original.ASSET, original.ASSET_SHA
OLD_ROBOT, OLD_MARKER = static.ROBOT, static.MARKER
ROBOT = '/World/patient_transport_h4_subfloor_cohort_01'
MARKER = '/World/IJCAS_HospitalH4SubfloorCohort01'
COHORT = 'cohort_03_subfloor_r01'
VERSION = 'H4_approved_common_subfloor_four_mesh_Session_variant_archive_Stop_staging_only_v1'
PROPOSAL = GATE/'hospital_h4_subfloor_wallbase_policy/proposal_r01/unadopted_environment_policy.json'
OUTPUT = GATE/'hospital_h4_cohort_preparation'/COHORT
PROTOCOL, ARCHIVE, BOUNDARY, LIVE, FAILURE, RESERVATION = (OUTPUT/name for name in (
    'preflight_r02/preparation_protocol.json', 'actual_prior_H4_static_state_before_Stop.json',
    'actual_Stop_boundary.json', 'live_h4_preparation.json', 'preparation_failure.json', 'single_staging_reserved.json'))
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_prepare_ijcas_hospital_h4_subfloor_cohort.py')
SELECTED = (
    '/World/Hospital/Geo_W_WallBase1_6/Geo_W_WallBase1_2',
    '/World/Hospital/Geo_W_WallBase32/Geo_W_WallBase/Geo_W_WallBase',
    '/World/Hospital/Geo_W_WallBase33/Geo_W_WallBase/Geo_W_WallBase',
    '/World/Hospital/Geo_W_WallBase34/Geo_W_WallBase/Geo_W_WallBase')
sha, verify = cached.sha, cached.verify


def write_exclusive(path, report):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    if json.loads(path.read_bytes()) != report:
        raise RuntimeError('Durable exclusive JSON verification failed: '+str(path))


def prepare_protocol():
    proposal = json.loads(PROPOSAL.read_bytes()); verify(proposal['input_sha256'])
    low = json.loads(cached.LIVE.read_bytes()); verify(low['input_sha256'])
    raw = json.loads(static.RESULT.read_bytes()); verify(raw['input_sha256'])
    if (tuple(r['path'] for r in proposal['proposed_meshes']) != SELECTED
            or proposal['original_planning_map_6_to_450mm_occupancy_changed_cells'] != 0
            or not proposal['all_sampled_authored_geometry_screens_clear']
            or low['status'] != 'CACHED_FLOOR_AND_ORIGINAL_LOW_WALL_VERIFIED_NO_DRIVE'
            or not all(low['checks'].values()) or raw['verdict'] != 'PASS'
            or not all(raw['checks'].values()) or sha(static.TRACE) != raw['trace_sha256']
            or sha(SOURCE) != SOURCE_SHA or sha(ASSET) != ASSET_SHA):
        raise ValueError('Exact approved proposal, retained actual static PASS and immutable assets required')
    setup = json.loads(original.SETUP.read_bytes())
    paths = (Path(__file__), ENTRY, PROPOSAL, cached.LIVE, static.RESULT, static.TRACE,
        Path(constructor.__file__), Path(fields.__file__), _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py'))
    return dict(version=VERSION, cohort=COHORT,
        status='USER_APPROVED_NEW_ENVIRONMENT_ARCHIVE_ONE_STOP_AND_STAGING_ONLY',
        adopted_policy_origin=str(PROPOSAL), original_unadopted_proposal_preserved=True,
        approved_collision_enabled_false_paths=list(SELECTED), old_robot_root=OLD_ROBOT,
        new_robot_root=ROBOT, new_marker=MARKER,
        original_start_xyyaw=setup['start_xyyaw'], unchanged_goal_xyyaw=setup['goal_xyyaw'],
        required_actual_static=str(static.RESULT), required_actual_GET_review=str(cached.LIVE),
        archive_all_thirteen_link_poses_velocities_twelve_joints_and_Root_Session_before_Stop=True,
        automatic_Stop_count_limit=1, wait_for_deferred_Stop_before_staging=True,
        Stop_wall_budget_s=15., Stop_update_budget=240,
        necessary_instanceable_metadata_only_on_actual_selected_mesh_ancestors=True,
        every_other_Hospital_collider_geometry_flag_material_and_support_unchanged=True,
        all_four_methods_require_same_new_environment_fingerprint=True,
        old_original_environment_outcomes_must_not_be_pooled_or_relabelled=True,
        original_planning_map_changed_cells=0, simulation_mass_kg=265.,
        source_authored_joint_seed_NOT_assigned_or_angle_selected=True,
        no_runtime_CICS_or_model_fit_control_gain_material_or_safety_change=True,
        fresh_static_support_pending=True, Play_static_trial_or_drive_authorized=False,
        automatic_retry_or_solver_state_restore_authorized=False, actual_drive_admission=False,
        use_for_final_statistics=False,
        input_sha256={**proposal['input_sha256'], **low['input_sha256'], **raw['input_sha256'],
            **{str(p): sha(p) for p in paths}})


def load_protocol():
    protocol = json.loads(PROTOCOL.read_bytes()); verify(protocol['input_sha256'])
    if protocol != prepare_protocol():
        raise ValueError('Exact approved preparation-only protocol changed')
    return protocol


def constructor_for(root):
    """Same immutable constructor bytecode; change only its target namespace."""
    scope = dict(constructor.author_new_robot.__globals__, ROBOT=root)
    for name in ('validate_new_robot', 'author_new_robot'):
        function = getattr(constructor, name)
        scope[name] = FunctionType(function.__code__, scope, name, function.__defaults__, function.__closure__)
    return scope['author_new_robot']


def numeric(value, shape, allow_positive_infinity=False):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    result = np.asarray(value, float).reshape(shape)
    if np.isnan(result).any() or np.isneginf(result).any() or (not allow_positive_infinity and not np.isfinite(result).all()):
        raise ValueError('Complete finite measured state required; only passive force limits may be +inf')
    return result.tolist()


def tensor_snapshot(view):
    names, links = list(view.shared_metatype.dof_names), list(view.shared_metatype.link_names)
    if view.count != 1 or view.max_links != 13 or len(names) != len(set(names)) or len(names) != 12:
        raise ValueError('One actual twelve-joint/thirteen-link articulation required')
    if len(links) != 13 or len(set(links)) != 13:
        raise ValueError('Complete thirteen-link correspondence required')
    limits = numeric(view.get_dof_max_forces(), (12,), True)
    return dict(names=names, link_names=links,
        transform=numeric(view.get_root_transforms(), (7,)), velocity=numeric(view.get_root_velocities(), (6,)),
        positions=numeric(view.get_dof_positions(), (12,)), joint_velocities_rad_s=numeric(view.get_dof_velocities(), (12,)),
        joint_position_targets_rad=numeric(view.get_dof_position_targets(), (12,)),
        joint_velocity_targets_rad_s=numeric(view.get_dof_velocity_targets(), (12,)),
        gains=dict(kp=numeric(view.get_dof_stiffnesses(), (12,)), kd=numeric(view.get_dof_dampings(), (12,)),
            max_efforts=['positive_infinity' if math.isinf(v) else v for v in limits]),
        all_thirteen_link_world_transforms_xyzw=numeric(view.get_link_transforms(), (13, 7)),
        all_thirteen_link_velocities_at_CoM=numeric(view.get_link_velocities(), (13, 6)))


def validate_actual_state(raw, actual):
    gains = {k: [math.inf if v == 'positive_infinity' else v for v in row] for k, row in actual['gains'].items()}
    continuity = cached.continuity.validate_tensor(raw, actual['transform'], actual['names'], actual['positions'], actual['velocity'], gains)
    previous = raw['passive_post_static_link_readback']
    a, b = np.asarray(actual['all_thirteen_link_world_transforms_xyzw']), np.asarray(previous['all_thirteen_link_world_transforms_xyzw'])
    if actual['link_names'] != previous['link_names'] or b.shape != (13, 7):
        raise ValueError('Actual own-static thirteen-link correspondence changed')
    norms = np.linalg.norm(a[:, 3:], axis=1)
    if not np.allclose(norms, 1., rtol=0, atol=1e-4):
        raise ValueError('Normalized measured link quaternions required')
    qa, qb = a[:, 3:]/norms[:, None], b[:, 3:]/np.linalg.norm(b[:, 3:], axis=1)[:, None]
    position = float(np.max(np.linalg.norm(a[:, :3]-b[:, :3], axis=1)))
    angle = float(np.max(np.degrees(2*np.arccos(np.clip(np.abs(np.sum(qa*qb, axis=1)), 0., 1.)))))
    if position > .001 or angle > .1:
        raise ValueError('Actual links differ from own completed static; no reset/recovery')
    return dict(continuity, maximum_link_position_difference_m=position, maximum_link_orientation_difference_deg=angle)


def environment_snapshot(stage):
    """All Hospital collider attributes/world geometry, supports and materials.

    Instance/prototype metadata is deliberately NOT physics. It is separately
    field-audited. No physical attribute (including offsets) is filtered out.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    roots = ('/World/Hospital', '/World/PatientTransportLabGround', original.support.PATCH_ROOT)
    result = {}
    cache = UsdGeom.XformCache()
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise ValueError('Protected environment root missing: '+root)
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not (p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdShade.Material)):
                continue
            attrs = {a.GetName(): dict(value=fields.semantic_value(a.Get()),
                time_samples=[(t, fields.semantic_value(a.Get(t))) for t in a.GetTimeSamples()]) for a in p.GetAttributes()}
            relationships = {r.GetName(): list(map(str, r.GetTargets())) for r in p.GetRelationships()}
            bound_material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
            result[str(p.GetPath())] = dict(active=bool(p.IsActive()), type=p.GetTypeName(),
                attributes=attrs, relationships=relationships,
                world_matrix=fields.semantic_value(cache.GetLocalToWorldTransform(p)),
                bound_physics_material=str(bound_material.GetPath()) if bound_material else None)
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('Exactly one unchanged PhysicsScene required')
    result['__unchanged_nominal_robot_and_scene__'] = dict(asset_sha256=sha(ASSET),
        simulation_mass_kg=265., source_authored_seed=True,
        original_active_joint_settings=constructor.JOINT_SETTINGS,
        physics_scene_path=str(scenes[0].GetPath()),
        physics_scene_attributes={a.GetName(): fields.semantic_value(a.Get()) for a in scenes[0].GetAttributes()})
    return json.loads(json.dumps(result, allow_nan=False))


def check_environment_delta(before, after):
    expected = json.loads(json.dumps(before))
    for path in SELECTED:
        row = expected.get(path)
        if row is None or row['attributes']['physics:collisionEnabled']['value'] is not True:
            raise ValueError('All four approved original meshes must still be enabled before preparation')
        row['attributes']['physics:collisionEnabled']['value'] = False
    if expected != after:
        changed = [p for p in sorted(set(expected)|set(after)) if expected.get(p) != after.get(p)]
        raise ValueError('Protected environment changed beyond four exact collision flags: '+str(changed))
    return hashlib.sha256(json.dumps(after, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_stop_layers(root_before, root_after, session_before, session_after):
    """Fail closed on unknown Stop writes; never restore/import a live layer.

    Root must remain identical. Session may only release per-link runtime
    defaults below the retiring actor. No gains, material, collider or scene
    changes qualify as runtime cleanup; all changed fields are returned.
    """
    from pxr import Sdf
    if root_before != root_after:
        raise ValueError('Root changed at Stop boundary; exact field review required before staging')
    a, b = fields.layer_snapshot(session_before), fields.layer_snapshot(session_after)
    changes = []
    for path in sorted(set(a)|set(b)):
        if a.get(path) == b.get(path):
            continue
        p = Sdf.Path(path)
        first, last = a.get(path, {}).get('info', {}), b.get(path, {}).get('info', {})
        keys = {k for k in set(first)|set(last) if first.get(k) != last.get(k)}
        if (not p.IsPropertyPath() or not p.GetPrimPath().HasPrefix(Sdf.Path(OLD_ROBOT))
                or p.name not in fields.RUNTIME_FIELDS or not keys <= {'default'}
                or a.get(path, {}).get('spec_type') != b.get(path, {}).get('spec_type')):
            raise ValueError('Unreviewed Session Stop-boundary field: '+path)
        changes.append(dict(path=path, changed_fields=sorted(keys), before=first.get('default'), after=last.get('default')))
    return changes


def check_staging_layers(before, after, deinstanced):
    from pxr import Sdf
    a, b = fields.layer_snapshot(before), fields.layer_snapshot(after)
    properties = {p+'.physics:collisionEnabled' for p in SELECTED}
    parents = {str(q) for path in (*SELECTED, ROBOT, MARKER) for q in Sdf.Path(path).GetPrefixes()}
    changes = []
    for path in sorted(set(a)|set(b)):
        if a.get(path) == b.get(path):
            continue
        p = Sdf.Path(path)
        if any(p.GetPrimPath().HasPrefix(Sdf.Path(root)) for root in (ROBOT, MARKER)):
            kind = 'new_validated_actor_or_marker'
        elif path in properties:
            if b.get(path, {}).get('info', {}).get('default') is not False:
                raise ValueError('Only collisionEnabled=False is approved')
            kind = 'exact_approved_subfloor_collision_flag'
        else:
            first, last = a.get(path, {}).get('info', {}), b.get(path, {}).get('info', {})
            keys = {k for k in set(first)|set(last) if first.get(k) != last.get(k)}
            allowed = {'specifier'} if path in parents else set()
            if path == OLD_ROBOT:
                allowed.add('active')
                if last.get('active') is not False:
                    raise ValueError('Retired actual actor must be inactive')
            if path in deinstanced:
                allowed.add('instanceable')
                if last.get('instanceable') is not False:
                    raise ValueError('Necessary ancestor deinstancing must be explicit false')
            if not keys <= allowed or not allowed:
                raise ValueError('Unapproved Session staging field: '+path)
            kind = 'exact_retirement_or_recorded_instance_ancestor_metadata'
        changes.append(dict(path=path, change_kind=kind))
    return changes


def author_session(stage, protocol):
    from pxr import Sdf, UsdGeom, UsdPhysics
    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError('New environment and actor must be authored ONLY in Session')
    if any(stage.GetPrimAtPath(p).IsValid() for p in (ROBOT, MARKER)):
        raise ValueError('New cohort already exists; no overwrite/retry')
    old = stage.GetPrimAtPath(OLD_ROBOT)
    if not old.IsValid() or not old.IsActive():
        raise ValueError('Actual prior actor must remain present until authorized retirement')
    deinstanced = []
    for path in SELECTED:
        for prefix in Sdf.Path(path).GetPrefixes():
            prim = stage.GetPrimAtPath(prefix)
            if prim.IsValid() and prim.IsInstance():
                prim.SetInstanceable(False); deinstanced.append(str(prefix))
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('An approved subfloor mesh is not editable: '+path)
        collision = UsdPhysics.CollisionAPI(prim)
        if collision.GetCollisionEnabledAttr().Get() is not True:
            raise ValueError('Approved original collider was already disabled: '+path)
        collision.CreateCollisionEnabledAttr().Set(False)
    old.SetActive(False)
    readback = constructor_for(ROBOT)(stage, protocol['original_start_xyyaw'])
    original.verify_actual_support(stage)
    active_bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(active_bodies) != 13 or any(not p.startswith(ROBOT+'/') for p in active_bodies):
        raise ValueError('Only the NEW thirteen-body nominal actor may remain active')
    marker = UsdGeom.Xform.Define(stage, MARKER).GetPrim()
    for key, value in dict(status='subfloor_environment_staged_STATIC_PENDING_NO_PLAY', cohort=COHORT,
            robot_root=ROBOT, prior_actual_archive=str(ARCHIVE), preparation_protocol=str(PROTOCOL),
            preparation_protocol_sha256=sha(PROTOCOL), physics_preparation_result=str(LIVE)).items():
        marker.SetCustomDataByKey(key, value)
    return dict(live_new_robot_readback=readback, deinstanced_mesh_ancestors=sorted(set(deinstanced)),
        all_active_rigid_body_paths=active_bodies, disabled_subfloor_meshes=list(SELECTED))


def main():
    if PROTOCOL.exists():
        load_protocol()
    else:
        write_exclusive(PROTOCOL, prepare_protocol())
    print('Approved four-mesh Session variant: archive -> ONE completed Stop -> nominal new actor ONLY.')
    print('saved:', PROTOCOL)
    print('No Play/static trial/drive/retry. Existing FAIL/HOLD/PASS files preserved.')


if __name__ == '__main__':
    main()
