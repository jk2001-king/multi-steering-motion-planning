"""Prospective static admission for the COMPACT completed-Stop preparation.

Software preparation is NOT a trial or authority to Play. Reuse the unchanged
zero-command control, physical thresholds, 26 support rays and four diagnostic
identity probes. Never reopen/decode/hash the original giant archive in Kit.
"""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import hospital_h4_subfloor_remaining_staging as preparation
import hospital_h4_subfloor_static_gate as prior


ROOT, GATE, ROBOT, MARKER = preparation.ROOT, preparation.GATE, preparation.ROBOT, preparation.MARKER
cohort = SimpleNamespace(**vars(preparation))
cohort.PREFLIGHT = preparation.PROTOCOL
cohort.SETUP = preparation.old.original.SETUP
cohort.completion = preparation.old.original.completion
cohort.validate_new_robot = preparation.old.constructor_for(ROBOT).__globals__['validate_new_robot']
VERSION = 'H4_compact_completed_Stop_preparation_same_static_and_26_plus_four_probes_v1'
OUTPUT = GATE/'hospital_h4_static_contact'/preparation.COHORT/'compact_completed_stop_r01'
RUN_ID = 'hospital_h4_subfloor_compact_static_r01'
RESULT, TRACE, RESERVATION = (OUTPUT/n for n in (RUN_ID+'.json', RUN_ID+'.csv', 'single_attempt_reserved.json'))
PREFLIGHT_HOLD = OUTPUT/'preflight_hold.json'
DESIGN = GATE/'hospital_h4_static_admission'/preparation.COHORT/'compact_completed_stop_r01'
PROTOCOL, ACTOR_RECIPE = DESIGN/'static_protocol.json', DESIGN/'source_actor_fixed_configuration.json'
ENTRY = ROOT/'isaac/scripts/isaac_run_ijcas_hospital_h4_subfloor_compact_static_gate.py'
SETTLE_S, RECORD_S, PHYSICS_HZ = prior.SETTLE_S, prior.RECORD_S, prior.PHYSICS_HZ
GUARDS, LIMITS, QUERY_LIMITS = prior.GUARDS, prior.LIMITS, prior.QUERY_LIMITS
BAD_CONTACTS, WHEELS, SUPPORTS, KNOWN_CATEGORIES = prior.BAD_CONTACTS, prior.WHEELS, prior.SUPPORTS, prior.KNOWN_CATEGORIES
classify_contact, evaluate, evaluate_subfloor_queries = prior.classify_contact, prior.evaluate, prior.evaluate_subfloor_queries
merge_evaluation, passive_final_link_readback = prior.merge_evaluation, prior.passive_final_link_readback


def actor_configuration(stage):
    """All actor attribute defaults/samples and relationships, minus motion state.

    Source-body relative poses/seed are separately verified BEFORE Play by the
    immutable constructor validator. Scale/collider offsets/gains/friction/
    masses/inertia are never exempted. No physics commands or views are used.
    """
    from pxr import Usd, UsdPhysics, UsdShade
    root = stage.GetPrimAtPath(ROBOT)
    if not root.IsValid() or not root.IsActive():
        raise ValueError('Exact new source-authored actor required')
    rows = {}
    body_motion = {'physics:velocity', 'physics:angularVelocity', 'xformOpOrder',
        'xformOp:translate', 'xformOp:orient', 'xformOp:transform'}
    joint_motion = {'state:angular:physics:position', 'state:angular:physics:velocity'}
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        attrs = {}
        for a in p.GetAttributes():
            name = a.GetName()
            if ((p.HasAPI(UsdPhysics.RigidBodyAPI) and name in body_motion)
                    or (p.IsA(UsdPhysics.RevoluteJoint) and name in joint_motion)):
                continue
            attrs[name] = dict(value_sha256=preparation.compact.native_digest(a.Get()),
                samples=[[t, preparation.compact.native_digest(a.Get(t))] for t in a.GetTimeSamples()])
        material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
        data = dict(type=p.GetTypeName(), applied_schemas=list(p.GetAppliedSchemas()),
            active=p.IsActive(), instanceable=p.IsInstanceable(), attributes=attrs,
            relationships={r.GetName():list(map(str,r.GetTargets())) for r in p.GetRelationships()},
            bound_physics_material=str(material.GetPath()) if material else None)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            data['fixed_nonmotion_xform_op_order']=[str(op) for op in (p.GetAttribute('xformOpOrder').Get() or [])
                if str(op) not in body_motion and str(op)!='!resetXformStack!']
        rows[str(p.GetPath())] = hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return rows


def build_actor_recipe():
    """An isolated anonymous USD constructor fixture; no Kit/PhysX or trial."""
    from pxr import Usd, UsdShade
    protocol = preparation.load_protocol()
    stage = Usd.Stage.CreateInMemory('H4_SOURCE_RECIPE_ONLY_NOT_ACTUAL_STAGING.usda')
    UsdShade.Material.Define(stage, preparation.old.constructor.MATERIAL)
    readback = preparation.old.constructor_for(ROBOT)(stage, protocol['original_start_xyyaw'])
    return dict(version=VERSION, status='ISOLATED_SOURCE_RECIPE_SOFTWARE_ONLY_NOT_ACTUAL_STAGING_OR_STATIC',
        immutable_constructor=str(Path(preparation.old.constructor.__file__)),
        source_asset_sha256=preparation.ASSET_SHA, source_constructor_readback=readback,
        fixed_configuration_rows=actor_configuration(stage),
        motion_state_exclusions_are_NOT_parameter_or_collider_exemptions=True,
        source_seed_and_relative_poses_require_separate_actual_PRE_Play_check=True,
        actual_physical_success=False, actual_drive_admission=False,
        input_sha256={str(p):preparation.sha(p) for p in (Path(__file__), Path(preparation.compact.__file__),
            Path(preparation.old.constructor.__file__), preparation.ASSET, preparation.PROTOCOL)})


def _protocol_from_records(prep, baseline, recipe):
    inventory_path = preparation.old.original.support.floor.inventory.TARGET
    inventory = preparation.read_small(inventory_path) if inventory_path.stat().st_size <= preparation.MAX_SMALL_JSON_BYTES else json.loads(inventory_path.read_bytes())
    # The frozen existing inventory is modest and pre-existing; never the giant
    # protected attribute snapshot. No ray, path or pose is reselected here.
    probes = prior.absence_probe_specs(inventory)
    known = sorted({r['path'] for r in inventory['collision_prims']
        if r['classification'] in ('enabled_in_ROI','outside_ROI')
        and r.get('collision_enabled_value') is not False and r['path'] not in preparation.SELECTED} | set(SUPPORTS))
    files = (Path(__file__), ENTRY, ACTOR_RECIPE, preparation.PROTOCOL, preparation.MANIFEST,
        Path(prior.__file__), prior.previous.PROTOCOL, inventory_path)
    return dict(version=VERSION, run_id=RUN_ID, cohort=preparation.COHORT,
        status='PROSPECTIVE_COMPACT_STATIC_DESIGN_ONLY_ACTUAL_PREPARATION_AND_SEPARATE_PLAY_APPROVAL_REQUIRED',
        required_actual_preparation=str(preparation.LIVE), preparation_protocol=str(preparation.PROTOCOL),
        new_robot_root=ROBOT, start_xyyaw=baseline['start_xyyaw'], goal_xyyaw=baseline['goal_xyyaw'],
        settle_s=SETTLE_S, record_s=RECORD_S, physics_hz=PHYSICS_HZ,
        unchanged_original_static_limits=LIMITS, guards=GUARDS, supports=list(SUPPORTS),
        additional_sampled_query_limits=QUERY_LIMITS, support_top_m=baseline['support_top_m'],
        sampled_cooked_support_queries=baseline['sampled_cooked_support_queries'],
        approved_disabled_subfloor_paths=list(preparation.SELECTED), additional_disabled_subfloor_probes=probes,
        auxiliary_probe_known_nonself_colliders=known,
        same_original_zero_command_control_physical_limits_and_26_support_queries=True,
        four_added_negative_identity_probes_do_NOT_filter_any_disabled_mesh_hits=True,
        source_recipe=str(ACTOR_RECIPE), nominal_actor_fixed_configuration=recipe['fixed_configuration_rows'],
        compact_codec=preparation.compact.FORMAT, all_archived_environment_rows_required=prep['all_original_protected_prim_rows_required'],
        whole_environment_compact_hash_checks_are_OUTSIDE_the_fixed_physics_recording_window=True,
        compact_readback_UI_yield_max_rows=16, compact_readback_UI_yield_wall_interval_s=.05,
        compact_readback_heartbeat_s=15., compact_readback_wall_budget_s=180.,
        large_original_archive_never_read_redecoded_or_hashed_inside_Kit=True,
        source_authored_seed_caster_angles_gains_mass_material_support_and_safety_unchanged=True,
        no_new_Stop_reset_recovery_or_retry=True, exactly_one_new_static_attempt=True,
        completed_staging_approval_does_NOT_authorize_static_Play=True,
        explicit_separate_single_static_authorization_required=True,
        automatic_Play_1s_settling_2s_recording_then_Pause_only_after_separate_user_launch=True,
        old_H3_H4_PASS_FAIL_HOLDs_preserved_NOT_pooled_or_relabelled=True,
        all_four_methods_require_same_new_environment_fingerprint=True,
        original_1029_planning_dynamics_full_envelope_and_multi_steering_contribution_unchanged=True,
        actual_static_trial_executed=False, actual_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**prep['input_sha256'], **baseline['input_sha256'], **recipe['input_sha256'],
            **{str(p):preparation.sha(p) for p in files}})


def prepare_protocol():
    prep = preparation.load_protocol(); baseline = prior.previous.load_protocol()
    recipe = preparation.read_small(ACTOR_RECIPE); preparation.verify(recipe['input_sha256'])
    if (recipe['status'] != 'ISOLATED_SOURCE_RECIPE_SOFTWARE_ONLY_NOT_ACTUAL_STAGING_OR_STATIC'
            or recipe['source_asset_sha256'] != preparation.ASSET_SHA or recipe['actual_physical_success'] is not False):
        raise ValueError('Isolated nominal source recipe required, never an invented physical result')
    return _protocol_from_records(prep, baseline, recipe)


def load_protocol():
    protocol = preparation.read_small(PROTOCOL); preparation.verify(protocol['input_sha256'])
    prep = preparation.read_small(preparation.PROTOCOL)
    baseline = preparation.read_small(prior.previous.PROTOCOL)
    recipe = preparation.read_small(ACTOR_RECIPE)
    if protocol != _protocol_from_records(prep, baseline, recipe):
        raise ValueError('Exact compact static design changed; no Play')
    if str(preparation.old.ARCHIVE) in protocol['input_sha256']:
        raise ValueError('Never include the giant archive in live hashes')
    return protocol


def validate_preparation_record(report, snapshots, protocol):
    required = dict(version=preparation.VERSION, cohort=preparation.COHORT,
        status='REVIEWED_COMPLETED_STOP_COMPACT_SUBFLOOR_COHORT_STAGED_STATIC_PENDING',
        new_robot_root=ROBOT, retired_robot_root=preparation.OLD_ROBOT,
        preparation_protocol=str(preparation.PROTOCOL), disabled_subfloor_meshes=list(preparation.SELECTED),
        start_xyyaw=protocol['start_xyyaw'], unchanged_goal_xyyaw=protocol['goal_xyyaw'],
        original_preparation_FAILURE_and_HOLD_preserved=True,
        original_archived_body_twelve_joints_thirteen_links_preserved_NOT_restored=True,
        codec=preparation.compact.FORMAT, actual_compact_readbacks=str(preparation.SNAPSHOTS),
        all_original_protected_row_count=protocol['all_archived_environment_rows_required'],
        all_rows_match_original_except_exact_four_flags=True,
        all_four_methods_require_same_new_environment_fingerprint=True,
        original_Root_and_source_USD_file_unchanged=True,
        timeline=dict(playing=False,stopped=True,current_time_s=0.),
        any_additional_Stop_Play_pause_reset_reopen_save_physics_initialization_or_trial=False,
        old_physical_continuation_ended_at_PRIOR_Stop_NOT_restored=True,
        large_original_archive_redecoded_or_hashed_inside_Kit=False,
        fresh_static_support_or_drive_validated=False, actual_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False)
    if any(report.get(k) != v for k,v in required.items()):
        raise ValueError('Actual successful COMPACT completed-Stop staging receipt required')
    manifest = preparation.load_manifest(verify_inputs=False)
    for key, after in (('before',False),('after',True)):
        actual = snapshots[key]
        digest = preparation.compact.compare_rows(manifest['compact_environment_rows'],actual,preparation.SELECTED,after)
        if snapshots[key+'_sha256'] != digest:
            raise ValueError('Actual compact preparation readbacks do not reproduce')
    if (snapshots['after_sha256'] != report['protected_environment_sha256']
            or snapshots['row_count'] != protocol['all_archived_environment_rows_required']
            or snapshots['every_original_protected_row_verified'] is not True
            or report['original_archive_full_sha256_verified_OFFLINE'] != manifest['original_archive_sha256']):
        raise ValueError('Actual preparation lacks the full original environment fingerprint')
    bodies=report['live_new_robot_readback']['rigid_bodies']; paths=report['all_active_rigid_body_paths']
    if (len(paths)!=13 or len(set(paths))!=13 or {b['path'] for b in bodies}!=set(paths)
            or any(not p.startswith(ROBOT+'/') for p in paths)
            or report['live_new_robot_readback']['simulation_mass_kg']!=265.
            or len(report['live_new_robot_readback']['revolute_joint_names'])!=12):
        raise ValueError('Complete nominal thirteen-body/twelve-joint readback required')


def load_actual_preparation():
    if (not all(p.is_file() for p in (preparation.LIVE,preparation.SNAPSHOTS,preparation.RESERVATION))
            or preparation.FAILURE.exists()):
        raise ValueError('Actual successful compact remaining-staging receipt required; never Play/reset to create it')
    report=preparation.read_small(preparation.LIVE); preparation.verify(report['input_sha256'])
    reserved=preparation.read_small(preparation.RESERVATION)
    if (reserved.get('version')!=preparation.VERSION or reserved.get('previously_approved_remaining_staging_only') is not True
            or reserved.get('new_Stop_or_Play_or_trial_authorized') is not False
            or reserved.get('protocol_sha256')!=preparation.sha(preparation.PROTOCOL)):
        raise ValueError('Exact preparation-only reservation required, not a static authorization')
    snapshots=preparation.read_small(preparation.SNAPSHOTS)
    validate_preparation_record(report,snapshots,load_protocol())
    return report


def critical_environment(stage, report):
    from pxr import PhysxSchema, UsdPhysics, UsdShade
    old=preparation.old
    if (Path(stage.GetRootLayer().realPath).resolve()!=preparation.SOURCE.resolve()
            or preparation.sha(preparation.SOURCE)!=preparation.SOURCE_SHA
            or any(p.GetTypeName()=='OmniGraph' for p in stage.Traverse())):
        raise ValueError('Unchanged original source without competing graphs required')
    for path in (preparation.OLD_ROBOT,old.original.OLD_ROBOT,'/World/patient_transport'):
        p=stage.GetPrimAtPath(path)
        if not p.IsValid() or p.IsActive():
            raise ValueError('All prior actors must remain retired/inactive')
    bodies=[]
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            mass=UsdPhysics.MassAPI(p).GetMassAttr().Get()
            if (not str(p.GetPath()).startswith(ROBOT+'/') or mass is None or not math.isfinite(mass) or mass<=0.
                    or PhysxSchema.PhysxRigidBodyAPI(p).GetDisableGravityAttr().Get() is True):
                raise ValueError('Only the original positive-mass/gravity thirteen-body new actor is allowed')
            bodies.append(dict(path=str(p.GetPath()),mass_kg=float(mass)))
    if bodies!=report['live_new_robot_readback']['rigid_bodies'] or len(bodies)!=13:
        raise ValueError('An individual nominal mass or active body changed')
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('One original physics scene required')
    scene = UsdPhysics.Scene(scenes[0])
    if (PhysxSchema.PhysxSceneAPI(scenes[0]).GetTimeStepsPerSecondAttr().Get() != PHYSICS_HZ
            or tuple(scene.GetGravityDirectionAttr().Get()) != (0., 0., -1.)
            or not math.isclose(scene.GetGravityMagnitudeAttr().Get(),9.81,abs_tol=1e-6)):
        raise ValueError('Original 120 Hz / gravity settings changed')
    old.original.verify_actual_support(stage)
    for name in WHEELS.values():
        material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ROBOT+'/'+name)).ComputeBoundMaterial('physics')
        if not material or str(material.GetPath())!=old.constructor.MATERIAL:
            raise ValueError('Canonical wheel-floor material binding changed')
    inventory=json.loads(old.original.support.floor.inventory.TARGET.read_bytes())
    return dict(simulation_mass_kg=sum(b['mass_kg'] for b in bodies),rigid_bodies=bodies,physics_hz=PHYSICS_HZ,
        protected_environment_sha256=report['protected_environment_sha256'],
        disabled_legacy_colliders=sorted(r['path'] for r in inventory['collision_prims'] if r['classification']=='disabled'),
        approved_disabled_subfloor_colliders=list(preparation.SELECTED),
        cooked_shape_equivalence_or_whole_route_support_claimed=False)


def validate_stopped_handoff(stage, timeline, protocol, report):
    state = dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
        current_time_s=float(timeline.get_current_time()))
    prep = preparation.read_small(preparation.PROTOCOL)
    if (state != dict(playing=False,stopped=True,current_time_s=0.)
            or hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest() != prep['protected_Root_text_sha256']
            or protocol['new_robot_root'] != ROBOT):
        raise ValueError('Actual compact-staged STOPPED Root/actor boundary changed; no recovery/Play')
    if actor_configuration(stage) != protocol['nominal_actor_fixed_configuration']:
        raise ValueError('Nominal actor fixed parameters differ from immutable source recipe')
    return critical_environment(stage,report)


def main():
    if ACTOR_RECIPE.exists():
        recipe=preparation.read_small(ACTOR_RECIPE);preparation.verify(recipe['input_sha256'])
    else:
        recipe=build_actor_recipe();preparation.write_exclusive(ACTOR_RECIPE,recipe)
    if PROTOCOL.exists():
        load_protocol()
    else:
        preparation.write_exclusive(PROTOCOL,prepare_protocol())
    print('COMPACT static design prepared OFFLINE: same physical controls/26 rays + four identity probes.')
    print('source recipe rows:',len(recipe['fixed_configuration_rows']))
    print('saved:',PROTOCOL)
    print('Actual staging and separate static Play approval still required. No new trial or drive admission.')


if __name__=='__main__':
    main()
