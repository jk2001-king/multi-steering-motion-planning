"""Finish the UNEXECUTED staging after the one already authorized Stop.

Exact reviewed Root cleanup only, not a general runtime-field waiver. The old
failure/HOLD/archive remain immutable; no additional Stop or H3 restoration.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import hashlib
import json
import os
from pathlib import Path

import hospital_h4_root_difference_readback as difference


prior = difference.prior
original = prior.cohort
ROOT, GATE = original.ROOT, original.GATE
OLD_ROBOT, ROBOT, MARKER = original.OLD_ROBOT, original.ROBOT, original.MARKER
SOURCE, SOURCE_SHA, ASSET, ASSET_SHA = original.SOURCE, original.SOURCE_SHA, original.ASSET, original.ASSET_SHA
SETUP, ARCHIVE = original.SETUP, original.ARCHIVE
MATERIAL, GROUND, WHEELS, JOINT_SETTINGS = original.MATERIAL, original.GROUND, original.WHEELS, original.JOINT_SETTINGS
support, completion = original.support, original.completion
canonical, verify = original.canonical, original.verify
author_new_robot, validate_new_robot = original.author_new_robot, original.validate_new_robot
protected_session_text = original.protected_session_text
VERSION = 'H4_completed_authorized_Stop_remaining_staging_exact_review_v1'
OUTPUT = original.OUTPUT/'completed_stop_continuation_r01'
PREFLIGHT = OUTPUT/'continuation_protocol.json'
LIVE, FAILURE, RESERVATION = (OUTPUT/name for name in (
    'live_h4_preparation.json', 'preparation_failure.json', 'single_staging_reserved.json'))
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_complete_ijcas_hospital_h4_staging_after_stop.py')
LAUNCH = _ijcas_path(ROOT, 'isaac/scripts/isaac_complete_ijcas_hospital_h4_staging_and_static.py')
ROOT_EXCEPTION = 'root_layer_matches_actual_pre_Stop_archive'
MISSING = {'field_is_not_authored': True}


def source_steering_defaults():
    """Read immutable asset defaults, never write a joint state in live USD."""
    from pxr import Usd
    if completion.sha(ASSET) != ASSET_SHA:
        raise ValueError('Original robot asset bytes changed')
    stage = Usd.Stage.Open(str(ASSET))
    result = {}
    for end in ('front', 'rear'):
        prim = stage.GetPrimAtPath(OLD_ROBOT+'/joints/base_link_to_'+end+'_active_steering')
        for field in ('position', 'velocity'):
            name = 'state:angular:physics:'+field
            attr = prim.GetAttribute(name)
            if not attr or not attr.HasAuthoredValueOpinion() or attr.Get() != 0.:
                raise ValueError('Four source-authored steering state defaults must be exactly zero')
            result[str(attr.GetPath())] = float(attr.Get())
    return result


def review_exact_changes(changes, links, defaults):
    """Require the exact 53 removals + four defaults; reject ANY other field."""
    if len(links) != 13 or len(set(links)) != 13 or len(defaults) != 4:
        raise ValueError('Exact thirteen-link archive/four source defaults required')
    removed = {OLD_ROBOT+'/'+link+'.'+name for link in links for name in
        ('physics:velocity', 'physics:angularVelocity', 'xformOp:translate', 'xformOp:orient')}
    removed.add(OLD_ROBOT+'/base_link.xformOpOrder')
    fields = ('__spec_type__', 'custom', 'default', 'typeName', 'variability')
    expected = {(path, key) for path in removed for key in fields} | {(path, 'default') for path in defaults}
    actual = [(row['path'], row['field']) for row in changes]
    if len(actual) != 269 or len(set(actual)) != 269 or set(actual) != expected:
        raise ValueError('Root diff is not exactly the reviewed 269 fields; no actor-subtree waiver')
    for row in changes:
        if row['automatically_waived'] is not False or row['before'] == row['current']:
            raise ValueError('A changed field was waived or has no actual change')
        if row['path'] in removed:
            if row['current'] != MISSING or row['unchanged_disk_source'] != MISSING:
                raise ValueError('Only the recorded runtime property REMOVALS are admitted')
        elif row['current'] != defaults[row['path']] or row['unchanged_disk_source'] != MISSING:
            raise ValueError('Only the four exact immutable source steering defaults are admitted')
    return dict(exact_reviewed_changed_fields=269, removed_old_H3_runtime_properties=53,
        removed_property_info_fields=265, source_default_steering_state_fields=4,
        unknown_changes_or_subtree_exemptions=0, original_failure_and_root_equality_HOLD_preserved=True,
        H3_solver_or_contact_state_restoration_claimed=False)


def prepare_protocol():
    failure, archive, preflight = prior.load_context()
    observed = json.loads(difference.RESULT.read_bytes()); verify(observed['input_sha256'])
    checks = observed['current_boundary_checks']
    if (observed['actual_timeline'] != dict(playing=False, stopped=True, current_time_s=0.)
            or checks.get(ROOT_EXCEPTION) is not False
            or not all(value is True for key, value in checks.items() if key != ROOT_EXCEPTION)
            or observed['source_USD_file_sha256'] != SOURCE_SHA
            or observed['actual_H4_staging_or_static_or_drive_admission'] is not False
            or observed['timeline_root_session_and_source_unchanged_by_inspection'] is not True):
        raise ValueError('Only the recorded STOPPED, unexecuted staging boundary is in scope')
    root, session = observed['actual_current_root_layer_text'], observed['actual_current_session_layer_text']
    if (hashlib.sha256(root.encode()).hexdigest() != observed['actual_current_root_text_sha256']
            or hashlib.sha256(session.encode()).hexdigest() != observed['actual_current_session_text_sha256']
            or session != archive['session_layer_text_before_Stop']):
        raise ValueError('Captured Root or exact protected Session text is inconsistent')
    recomputed = difference.compare_layers(archive['root_layer_text_before_Stop'], root, SOURCE.read_text())
    if recomputed != observed['root_difference']:
        raise ValueError('Saved Root field diff does not reproduce from the actual captured layers')
    defaults = source_steering_defaults()
    review = review_exact_changes(recomputed['changed_fields'], archive['actual_state']['link_names'], defaults)
    if any(path.exists() for path in (original.LIVE, *legacy_static_paths())):
        raise ValueError('An original H4 preparation/static attempt appeared; no continuation')
    sources = (Path(__file__), ENTRY, LAUNCH, _ijcas_path(ROOT, 'studies/IJCAS/scripts/hospital_private_loader.py'))
    pins = {**observed['input_sha256'], **preflight['input_sha256'],
        **{str(path): completion.sha(path) for path in (*sources, difference.RESULT, original.FAILURE,
            ARCHIVE/'actual_H3_final_state.json', original.PREFLIGHT, ASSET, SOURCE)}}
    verify(pins)
    return canonical(dict(version=VERSION, status='OFFLINE_EXACT_BOUNDARY_REVIEW_REMAINING_STAGING_ONLY',
        reviewed_actual_root_difference=str(difference.RESULT), exact_cleanup_review=review,
        source_authored_steering_state_defaults=defaults,
        reviewed_current_root_text_sha256=observed['actual_current_root_text_sha256'],
        reviewed_current_session_text_sha256=observed['actual_current_session_text_sha256'],
        original_failed_preparation=str(original.FAILURE), original_root_equality_HOLD=str(prior.RESULT),
        original_pre_Stop_actual_H3_archive=str(ARCHIVE/'actual_H3_final_state.json'),
        start_xyyaw=preflight['start_xyyaw'], goal_xyyaw=preflight['goal_xyyaw'],
        supports=preflight['supports'], support_top_m=preflight['support_top_m'],
        new_robot_root=ROBOT, old_robot_root=OLD_ROBOT, fresh_reference_author=original.author_new_robot.__module__,
        no_additional_Stop_Play_reset_reopen_save_or_restore=True,
        conditional_single_static_authorization_remains_separate=True,
        old_failure_and_HOLD_NOT_relabelled_as_success=True, H3_whole_MODEL_PASS=2, H3_whole_MODEL_total=8,
        independent_fresh_whole_H3_repeats=0, H4_staged=False, H4_static_validated=False,
        actual_drive_admission=False, CICS_runtime_integration=False, use_for_final_statistics=False,
        input_sha256=pins))


def legacy_static_paths():
    output = GATE/'hospital_h4_static_contact/cohort_01'
    return tuple(output/name for name in ('single_attempt_reserved.json', 'hospital_h4_static_r01.json', 'hospital_h4_static_r01.csv'))


def load_protocol():
    report = json.loads(PREFLIGHT.read_bytes()); verify(report['input_sha256'])
    if report != prepare_protocol():
        raise ValueError('Frozen completed-Stop continuation protocol changed')
    return report


def validate_live_boundary(stage, timeline, protocol):
    """Read only before any reservation/Session edit; never initialize physics."""
    if (stage is None or timeline.is_playing() or not timeline.is_stopped()
            or float(timeline.get_current_time()) != 0.
            or Path(stage.GetRootLayer().realPath).resolve() != SOURCE.resolve()):
        raise ValueError('Exact original STOPPED source at zero is required; no timeline recovery')
    capture = json.loads(difference.RESULT.read_bytes())
    if (stage.GetRootLayer().ExportToString() != capture['actual_current_root_layer_text']
            or stage.GetSessionLayer().ExportToString() != capture['actual_current_session_layer_text']):
        raise ValueError('Root/Session changed SINCE the reviewed Stop capture; no additional edit waiver')
    current = prior.inspect(stage, timeline)
    if current['checks'] != capture['current_boundary_checks']:
        raise ValueError('Live boundary no longer matches the actual reviewed boundary')
    verify(protocol['input_sha256'])
    return current


def save_exclusive(path, report):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def stage_remaining(stage, timeline):
    """Authorized remaining Session staging only, NOT an H4 physical retry."""
    from pxr import Usd, UsdGeom
    protocol = load_protocol()
    if any(path.exists() for path in (LIVE, FAILURE, RESERVATION)):
        raise ValueError('Continuation already reserved/succeeded/failed; no overwrite or automatic retry')
    validate_live_boundary(stage, timeline, protocol)
    session = stage.GetSessionLayer()
    root_before, session_before = stage.GetRootLayer().ExportToString(), session.ExportToString()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_exclusive(RESERVATION, dict(version=VERSION, protocol_sha256=completion.sha(PREFLIGHT),
        no_additional_Stop_or_Play=True, remaining_staging_only=True))
    try:
        with Usd.EditContext(stage, session):
            stage.GetPrimAtPath(OLD_ROBOT).SetActive(False)
            readback = author_new_robot(stage, protocol['start_xyyaw'])
            marker = UsdGeom.Xform.Define(stage, MARKER).GetPrim()
            for key, value in dict(status='fresh_H4_prepared_static_pending', robot_root=ROBOT,
                    archived_H3_actual_state=str(ARCHIVE/'actual_H3_final_state.json'),
                    cohort_preflight=str(PREFLIGHT), cohort_preflight_sha256=completion.sha(PREFLIGHT),
                    physics_preparation_result=str(LIVE), completed_Stop_continuation_version=VERSION).items():
                marker.SetCustomDataByKey(key, value)
        # Import the SAME original static environment reader only after retiring
        # the old actor. This is USD readback, not cooked/static certification.
        import hospital_h4_static_gate as static
        environment = static.validate_environment(stage)
        original.verify_actual_support(stage)
        if (timeline.is_playing() or not timeline.is_stopped() or float(timeline.get_current_time()) != 0.
                or root_before != stage.GetRootLayer().ExportToString()
                or protected_session_text(session_before) != protected_session_text(session.ExportToString())):
            raise ValueError('Protected Root/Session/timeline changed during remaining staging')
        verify(protocol['input_sha256'])
        report = dict(version=VERSION, status='H4_COMPLETED_STOP_REMAINING_STAGING_SUCCEEDED_STATIC_PENDING',
            new_robot_root=ROBOT, retired_H3_robot_root=OLD_ROBOT, retired_H3_inactive=True,
            archived_actual_H3_state=str(ARCHIVE/'actual_H3_final_state.json'),
            archived_actual_H3_state_sha256=completion.sha(ARCHIVE/'actual_H3_final_state.json'),
            original_failed_preparation=str(original.FAILURE), original_failure_and_HOLD_preserved=True,
            archive_durably_verified_before_original_Stop=True, additional_Stop_Play_reset_or_restore=False,
            original_H3_live_physical_continuation_ended_NOT_restored=True,
            live_new_robot_readback=readback, environment_readback=environment,
            task_candidate=str(SETUP), start_xyyaw=protocol['start_xyyaw'], goal_xyyaw=protocol['goal_xyyaw'],
            support_colliders=protocol['supports'], support_top_m=protocol['support_top_m'],
            only_authorized_old_new_robot_and_marker_Session_changes=True,
            post_Stop_Root_unchanged=True, source_USD_saved=False, timeline='STOPPED',
            fresh_source_seed_recorded_NOT_assigned_or_angle_selected=True,
            actual_H4_static_support_validated=False, Play_or_static_or_drive_executed=False,
            actual_drive_admission=False, CICS_runtime_integration=False, use_for_final_statistics=False,
            input_sha256={**protocol['input_sha256'], str(PREFLIGHT): completion.sha(PREFLIGHT)})
        save_exclusive(LIVE, report)
    except Exception as error:
        # Preserve all previous evidence; inhibit only our newly created actor.
        with Usd.EditContext(stage, session):
            actor = stage.GetPrimAtPath(ROBOT)
            if actor.IsValid():
                actor.SetActive(False)
        save_exclusive(FAILURE, dict(version=VERSION, status='HOLD_REMAINING_STAGING_NO_PLAY_NO_RECOVERY',
            error=type(error).__name__+': '+str(error), additional_Stop_or_Play=False,
            original_archive_and_failure_preserved=True, automatic_retry_or_restore=False))
        raise
    return report


def load_actual_preparation():
    if not LIVE.is_file() or FAILURE.exists() or not RESERVATION.is_file():
        raise ValueError('Actual successful completed-Stop remaining staging receipt required')
    protocol = load_protocol()
    report = json.loads(LIVE.read_bytes()); verify(report['input_sha256'])
    required = dict(version=VERSION, status='H4_COMPLETED_STOP_REMAINING_STAGING_SUCCEEDED_STATIC_PENDING',
        new_robot_root=ROBOT, retired_H3_robot_root=OLD_ROBOT, retired_H3_inactive=True,
        original_failed_preparation=str(original.FAILURE), original_failure_and_HOLD_preserved=True,
        archive_durably_verified_before_original_Stop=True, additional_Stop_Play_reset_or_restore=False,
        original_H3_live_physical_continuation_ended_NOT_restored=True,
        task_candidate=str(SETUP), start_xyyaw=protocol['start_xyyaw'], goal_xyyaw=protocol['goal_xyyaw'],
        support_colliders=protocol['supports'], support_top_m=protocol['support_top_m'],
        only_authorized_old_new_robot_and_marker_Session_changes=True,
        post_Stop_Root_unchanged=True, source_USD_saved=False, timeline='STOPPED',
        fresh_source_seed_recorded_NOT_assigned_or_angle_selected=True,
        actual_H4_static_support_validated=False, Play_or_static_or_drive_executed=False,
        actual_drive_admission=False, CICS_runtime_integration=False, use_for_final_statistics=False)
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError('Exact new actual staging receipt required; old failure cannot be relabelled')
    archive = ARCHIVE/'actual_H3_final_state.json'
    if report.get('archived_actual_H3_state') != str(archive) or report.get('archived_actual_H3_state_sha256') != completion.sha(archive):
        raise ValueError('Actual original pre-Stop archive identity changed')
    return report
