"""Prospective static gate for the approved FOUR-mesh environment variant.

This prepares software, not Play authority or an actual success. Preserve the
original zero-command trial, limits and 26 adopted support rays. Add four
identity-specific cooked-query probes; no obstacle contact gets an exemption.
Old environment trials must never be pooled with this new cohort.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import hashlib
import json
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

import hospital_h4_subfloor_preparation as preparation
import hospital_h4_reopened_static_gate as previous


ROOT, GATE, ROBOT, MARKER = preparation.ROOT, preparation.GATE, preparation.ROBOT, preparation.MARKER
cohort = SimpleNamespace(**vars(preparation))
cohort.PREFLIGHT = preparation.PROTOCOL
cohort.SETUP = preparation.original.SETUP
cohort.completion = preparation.original.completion
cohort.validate_new_robot = preparation.constructor_for(ROBOT).__globals__['validate_new_robot']
VERSION = 'H4_subfloor_cohort_03_same_static_control_26_support_plus_four_absence_probes_v1'
OUTPUT = GATE/'hospital_h4_static_contact'/preparation.COHORT
RUN_ID = 'hospital_h4_subfloor_static_r01'
RESULT, TRACE, RESERVATION = (OUTPUT/name for name in (RUN_ID+'.json', RUN_ID+'.csv', 'single_attempt_reserved.json'))
PROTOCOL = GATE/'hospital_h4_static_admission'/preparation.COHORT/'static_protocol.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_run_ijcas_hospital_h4_subfloor_static_gate.py')
SETTLE_S, RECORD_S, PHYSICS_HZ = previous.SETTLE_S, previous.RECORD_S, previous.PHYSICS_HZ
GUARDS, LIMITS, QUERY_LIMITS = previous.GUARDS, previous.LIMITS, previous.QUERY_LIMITS
BAD_CONTACTS, WHEELS, SUPPORTS, KNOWN_CATEGORIES = previous.BAD_CONTACTS, previous.WHEELS, previous.SUPPORTS, previous.KNOWN_CATEGORIES
merge_evaluation, passive_final_link_readback = previous.merge_evaluation, previous.passive_final_link_readback
_classifier = previous.classify_contact
classify_contact = FunctionType(_classifier.__code__, dict(_classifier.__globals__, ROBOT=ROBOT),
    'classify_contact', _classifier.__defaults__, _classifier.__closure__)


def absence_probe_specs(inventory):
    """Four ray/triangle intersections fixed from the ORIGINAL saved meshes."""
    lookup = {r['path']: r for r in inventory['collision_prims']}
    result = []
    for path in preparation.SELECTED:
        row = lookup[path]; mesh = row['authored_mesh']
        points = np.asarray(mesh['world_points_m'], float)
        candidates = []; offset = 0
        for count in mesh['face_vertex_counts']:
            face = mesh['face_vertex_indices'][offset:offset+count]; offset += count
            for i in range(1, count-1):
                tri = points[[face[0], face[i], face[i+1]]]
                normal = np.cross(tri[1]-tri[0], tri[2]-tri[0]); norm = np.linalg.norm(normal)
                if norm > 0. and abs(normal[2]/norm) >= .99:
                    candidates.append((float(tri[:, 2].mean()), tri.mean(axis=0)))
        if not candidates:
            raise ValueError('Original subfloor mesh lacks a verified near-horizontal triangle: '+path)
        z, target = max(candidates, key=lambda r: r[0])
        if abs(z-row['world_aligned_bound']['maximum_m'][2]) > .000020:
            raise ValueError('Probe does not intersect original mesh top within the unchanged bound uncertainty')
        result.append(dict(id='disabled_subfloor/'+str(len(result)), original_collider=path,
            original_triangle_target_xyz_m=target.tolist(), origin_m=[float(target[0]), float(target[1]), z+.5],
            direction=[0., 0., -1.], distance_m=.8,
            criterion='transaction complete; approved disabled identities absent; do NOT filter them out',
            floor_presence_at_this_auxiliary_outside_ROI_probe_not_claimed=True))
    return result


def prepare_protocol():
    prep = preparation.load_protocol(); baseline = previous.load_protocol()
    inventory_path = preparation.original.support.floor.inventory.TARGET
    inventory = json.loads(inventory_path.read_bytes())
    probes = absence_probe_specs(inventory)
    known = sorted({r['path'] for r in inventory['collision_prims']
        if r['classification'] in ('enabled_in_ROI', 'outside_ROI')
        and r.get('collision_enabled_value') is not False and r['path'] not in preparation.SELECTED} | set(SUPPORTS))
    paths = (Path(__file__), ENTRY, previous.PROTOCOL, preparation.PROTOCOL, inventory_path)
    return dict(version=VERSION, run_id=RUN_ID, cohort=preparation.COHORT,
        status='OFFLINE_PROTOCOL_ONLY_ACTUAL_STAGING_AND_SEPARATE_STATIC_PLAY_APPROVAL_PENDING',
        required_actual_preparation=str(preparation.LIVE), preparation_protocol=str(preparation.PROTOCOL),
        new_robot_root=ROBOT, start_xyyaw=baseline['start_xyyaw'], goal_xyyaw=baseline['goal_xyyaw'],
        settle_s=SETTLE_S, record_s=RECORD_S, physics_hz=PHYSICS_HZ,
        unchanged_original_static_limits=LIMITS, guards=GUARDS, supports=list(SUPPORTS),
        additional_sampled_query_limits=QUERY_LIMITS, support_top_m=baseline['support_top_m'],
        sampled_cooked_support_queries=baseline['sampled_cooked_support_queries'],
        same_original_26_queries_adopted_dual_channel_rule_and_all_tolerances=True,
        original_physical_evaluator_and_zero_command_controller_unchanged=True,
        approved_disabled_subfloor_paths=list(preparation.SELECTED),
        additional_disabled_subfloor_probes=probes, auxiliary_probe_known_nonself_colliders=known,
        disabled_identity_absence_required_in_all_26_plus_four_ray_records=True,
        absence_probes_are_NOT_continuous_cooked_collision_or_body_response_validation=True,
        contact_policy='all actual obstacle/nonwheel/legacy/unknown contacts FAIL; no subfloor exception',
        archive_Stop_and_original_static_and_path_FAILs_remain_immutable=True,
        no_pooling_old_and_new_environment_trials=True, same_new_environment_required_for_all_four_methods=True,
        original_1029_multi_steering_dynamics_and_full_envelope_planning_unchanged=True,
        new_environment_setup_approval_does_NOT_authorize_this_Play_or_trial=True,
        explicit_separate_single_static_authorization_required=True, exactly_one_attempt_no_retry=True,
        automatic_Play_1s_settling_2s_recording_then_Pause_only_after_separate_user_launch=True,
        actual_staging_verified=False, actual_static_trial_executed=False, actual_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256={**prep['input_sha256'], **baseline['input_sha256'],
            **{str(p): preparation.sha(p) for p in paths}})


def load_protocol():
    protocol = json.loads(PROTOCOL.read_bytes()); preparation.verify(protocol['input_sha256'])
    if protocol != prepare_protocol():
        raise ValueError('Exact prospective subfloor static protocol changed; no Play')
    return protocol


def validate_preparation_records(report, archive, boundary):
    """Pure record checks; never creates a receipt or substitutes for live USD."""
    protocol = preparation.load_protocol()
    required = dict(version=preparation.VERSION, cohort=preparation.COHORT,
        status='NEW_SUBFLOOR_ENVIRONMENT_COHORT_STAGED_STOPPED_STATIC_PENDING',
        new_robot_root=ROBOT, retired_robot_root=preparation.OLD_ROBOT,
        actual_prior_static_archive=str(preparation.ARCHIVE),
        preparation_protocol=str(preparation.PROTOCOL),
        archive_durably_verified_before_Stop=True, Stop_called_once_and_deferred_completion_verified=True,
        old_physical_continuation_ended_NOT_restored=True,
        timeline=dict(playing=False, stopped=True, current_time_s=0.),
        disabled_subfloor_meshes=list(preparation.SELECTED),
        start_xyyaw=protocol['original_start_xyyaw'], unchanged_goal_xyyaw=protocol['unchanged_goal_xyyaw'],
        same_environment_fingerprint_required_for_all_four_methods=True,
        original_root_and_source_USD_file_unchanged=True,
        old_environment_outcomes_preserved_NOT_pooled_or_relabelled=True,
        original_planning_map_changed_cells=0, fresh_static_support_validated=False,
        Play_static_or_drive_executed=False, actual_drive_admission=False,
        caster_state_assigned_or_selected=False, CICS_runtime_integration=False, use_for_final_statistics=False)
    if any(report.get(k) != v for k, v in required.items()):
        raise ValueError('Exact successful NEW environment preparation-only receipt required')
    if (archive.get('status') != 'ACTUAL_REOPENED_H4_STATIC_PASS_ARCHIVED_BEFORE_AUTHORIZED_STOP'
            or archive.get('version') != preparation.VERSION
            or archive.get('cohort') != preparation.COHORT
            or archive.get('actual_static_result') != str(preparation.static.RESULT)
            or archive.get('actual_static_sha256') != preparation.sha(preparation.static.RESULT)
            or archive.get('source_USD_file_sha256') != preparation.SOURCE_SHA
            or archive.get('physical_solver_contact_state_NOT_serialized_or_restoreable') is not True
            or archive['timeline_before_Stop']['playing'] or archive['timeline_before_Stop']['stopped']
            or boundary.get('Stop_called_once') is not True
            or boundary.get('version') != preparation.VERSION
            or boundary.get('actual_timeline') != required['timeline']
            or boundary.get('no_physical_state_restore_or_Play') is not True
            or not 2 <= boundary.get('awaited_update_count', 0) <= protocol['Stop_update_budget']+1
            or boundary['actual_root_layer_text'] != archive['root_layer_text_before_Stop']):
        raise ValueError('Actual pre-Stop archive and completed Stop boundary required')
    raw = json.loads(preparation.static.RESULT.read_bytes())
    preparation.validate_actual_state(raw, archive['actual_state'])
    active = report['all_active_rigid_body_paths']; bodies = report['live_new_robot_readback']['rigid_bodies']
    if (len(active) != 13 or len(set(active)) != 13 or any(not p.startswith(ROBOT+'/') for p in active)
            or {r['path'] for r in bodies} != set(active)
            or report['live_new_robot_readback']['simulation_mass_kg'] != 265.
            or len(report['live_new_robot_readback']['revolute_joint_names']) != 12
            or report['live_new_robot_readback']['all_link_relative_poses_match_source'] is not True
            or report['live_new_robot_readback']['static_support_or_drive_validated'] is not False
            or len(report.get('protected_environment_sha256', '')) != 64):
        raise ValueError('Complete nominal NEW thirteen-body/twelve-joint preparation readback required')
    delta = preparation.check_stop_layers(archive['root_layer_text_before_Stop'], boundary['actual_root_layer_text'],
        archive['session_layer_text_before_Stop'], boundary['actual_session_layer_text'])
    if delta != report['Stop_Session_runtime_field_deltas']:
        raise ValueError('Recorded Stop boundary field changes do not reproduce')


def load_actual_preparation():
    if (not all(p.is_file() for p in (preparation.LIVE, preparation.ARCHIVE, preparation.BOUNDARY, preparation.RESERVATION))
            or preparation.FAILURE.exists()):
        raise ValueError('Actual successful subfloor preparation/archive/Stop receipt required; do NOT Play to create it')
    reserved = json.loads(preparation.RESERVATION.read_bytes())
    required = dict(version=preparation.VERSION, cohort=preparation.COHORT,
        user_authorized_archive_one_Stop_and_new_Session_cohort=True,
        Play_static_trial_drive_or_retry_authorized=False,
        protocol_sha256=preparation.sha(preparation.PROTOCOL))
    if any(reserved.get(k) != v for k, v in required.items()):
        raise ValueError('Actual preparation reservation must bind its exact preparation-only authority and protocol')
    report, archive, boundary = [json.loads(p.read_bytes()) for p in (preparation.LIVE, preparation.ARCHIVE, preparation.BOUNDARY)]
    preparation.verify(report['input_sha256']); preparation.verify(archive['input_sha256'])
    if (report['actual_prior_static_archive_sha256'] != preparation.sha(preparation.ARCHIVE)
            or boundary['actual_prior_archive_sha256'] != preparation.sha(preparation.ARCHIVE)
            or boundary['actual_prior_archive'] != str(preparation.ARCHIVE)):
        raise ValueError('Durable actual prior-state archive hash differs')
    validate_preparation_records(report, archive, boundary)
    return report


def read_environment(stage, report, archived_environment):
    """No exemptions: all static geometry/flags/offsets and scene stay frozen."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade
    if (Path(stage.GetRootLayer().realPath).resolve() != preparation.SOURCE.resolve()
            or preparation.sha(preparation.SOURCE) != preparation.SOURCE_SHA
            or preparation.sha(preparation.ASSET) != preparation.ASSET_SHA
            or any(p.GetTypeName() == 'OmniGraph' for p in stage.Traverse())):
        raise ValueError('Unchanged hospital/robot source without competing graphs required')
    fingerprint = preparation.check_environment_delta(archived_environment, preparation.environment_snapshot(stage))
    if fingerprint != report['protected_environment_sha256']:
        raise ValueError('New environment fingerprint differs from its actual preparation')
    for path in (preparation.OLD_ROBOT, preparation.original.OLD_ROBOT, '/World/patient_transport'):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.IsActive():
            raise ValueError('Every prior actor must remain retired/inactive')
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if (not str(prim.GetPath()).startswith(ROBOT+'/') or mass is None or not math.isfinite(mass)
                or mass <= 0. or PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get() is True):
            raise ValueError('Only the original positive-mass/gravity NEW robot may be active')
        bodies.append(dict(path=str(prim.GetPath()), mass_kg=float(mass)))
    if (bodies != report['live_new_robot_readback']['rigid_bodies'] or len(bodies) != 13
            or abs(sum(b['mass_kg'] for b in bodies)-265.) > .01):
        raise ValueError('Individual nominal masses or thirteen-body correspondence changed')
    base = UsdPhysics.MassAPI(stage.GetPrimAtPath(ROBOT+'/base_link'))
    if base.GetMassAttr().Get() != 245. or not np.allclose(base.GetCenterOfMassAttr().Get(), (0., 0., .270), rtol=0, atol=1e-6):
        raise ValueError('Original base mass/CoM changed')
    for name, settings in preparation.constructor.JOINT_SETTINGS.items():
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(ROBOT+'/joints/'+name), 'angular')
        if ((drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get(), drive.GetMaxForceAttr().Get()) != settings
                or drive.GetTypeAttr().Get() != 'force' or drive.GetTargetPositionAttr().Get() != 0.
                or drive.GetTargetVelocityAttr().Get() != 0.):
            raise ValueError('Original motor settings/neutral targets changed')
    for name in WHEELS.values():
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(ROBOT+'/'+name)).ComputeBoundMaterial('physics')
        if not material or str(material.GetPath()) != preparation.constructor.MATERIAL:
            raise ValueError('Canonical wheel material binding changed')
    preparation.original.verify_actual_support(stage)
    inventory = json.loads(preparation.original.support.floor.inventory.TARGET.read_bytes())
    return dict(protected_environment_sha256=fingerprint, simulation_mass_kg=sum(b['mass_kg'] for b in bodies),
        rigid_bodies=bodies, physics_hz=PHYSICS_HZ,
        disabled_legacy_colliders=sorted(r['path'] for r in inventory['collision_prims'] if r['classification'] == 'disabled'),
        approved_disabled_subfloor_colliders=list(preparation.SELECTED),
        material_readback=json.loads(preparation.static.RESULT.read_bytes())['environment_after_static']['material_readback'],
        cooked_shape_equivalence_or_whole_route_support_claimed=False)


def validate_environment(stage):
    report = load_actual_preparation()
    archive = json.loads(preparation.ARCHIVE.read_bytes())
    return read_environment(stage, report, archive['protected_environment_before_Stop'])


def validate_stopped_handoff(stage, timeline, protocol, report):
    state = dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()), current_time_s=float(timeline.get_current_time()))
    archive = json.loads(preparation.ARCHIVE.read_bytes())
    if (state != dict(playing=False, stopped=True, current_time_s=0.)
            or stage.GetRootLayer().ExportToString() != archive['root_layer_text_before_Stop']
            or protocol['new_robot_root'] != ROBOT):
        raise ValueError('Actual STOPPED post-preparation Root/actor handoff changed; no recovery/Play')
    return read_environment(stage, report, archive['protected_environment_before_Stop'])


def evaluate(rows, start, counts, errors, specifications, observations, top):
    return previous.evaluate(rows, start, counts, errors, specifications, observations, top)


def evaluate_subfloor_queries(protocol, support_observations, observations):
    specs = protocol['additional_disabled_subfloor_probes']
    complete = (len(observations) == len(specs)
        and [r.get('id') for r in observations] == [s['id'] for s in specs])
    valid = complete; absent = True; rows = []
    known = set(protocol['auxiliary_probe_known_nonself_colliders'])
    disabled = set(protocol['approved_disabled_subfloor_paths'])
    for spec, row in zip(specs, observations):
        ok = row.get('id') == spec['id'] and not row.get('errors') and isinstance(row.get('all_nonself_hits'), list)
        hits = row.get('all_nonself_hits', [])
        for hit in hits:
            xyz, normal = hit.get('position_m', []), hit.get('normal', [])
            distance = hit.get('distance_m', float('nan'))
            if (hit.get('collision') not in known or len(xyz) != 3 or len(normal) != 3
                    or not np.isfinite([*xyz, *normal, distance]).all()
                    or not 0. <= distance <= spec['distance_m']+.0005
                    or not np.allclose(xyz[:2], spec['origin_m'][:2], rtol=0, atol=.0005)):
                ok = False
            if hit.get('collision') in disabled:
                absent = False
        valid = bool(valid and ok)
        rows.append(dict(id=spec['id'], complete_valid_query_transaction=bool(ok),
            any_approved_disabled_identity_returned=any(h.get('collision') in disabled for h in hits),
            nonself_hit_count=len(hits)))
    for row in support_observations:
        if any(hit.get('collision') in disabled for hit in row.get('all_nonself_hits', [])):
            absent = False
    return dict(checks=dict(all_four_subfloor_probe_transactions_complete=bool(valid),
        approved_disabled_identities_absent_in_support_and_probe_rays=bool(complete and absent)),
        auxiliary_probe_diagnostics=rows,
        scope='FOUR_IDENTITY_ABSENCE_PROBES_WITH_26_POSITIVE_SUPPORT_CONTROLS_NOT_CONTINUOUS_COLLISION_PROOF')


def main():
    if PROTOCOL.exists():
        load_protocol()
    else:
        preparation.write_exclusive(PROTOCOL, prepare_protocol())
    print('Subfloor-variant static protocol prepared OFFLINE; actual staging and separate Play approval pending.')
    print('Same physical controls/limits and 26 rays; four additional disabled-identity probes, no hit filtering.')
    print('saved:', PROTOCOL)


if __name__ == '__main__':
    main()
