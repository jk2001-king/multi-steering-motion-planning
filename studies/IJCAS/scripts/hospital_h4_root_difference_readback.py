"""Get-only Root-layer field differences after the already completed Stop.

No field difference is waived. Current Root and Session text are captured for
offline review; OLD root-equality HOLD, failed preparation and H3 archive stay
immutable. Runtime-looking changes are labels, never drive/staging admission.
"""

import difflib
import hashlib
import json
import math
from pathlib import Path

import hospital_h4_stop_boundary_readback as prior


cohort = prior.cohort
OUTPUT = cohort.GATE/'hospital_h4_cohort_preparation/root_difference_readback_r01'
RESULT = OUTPUT/'live_root_difference.json'
ENTRY = cohort.ROOT/'isaac/scripts/isaac_review_ijcas_hospital_h4_root_difference.py'
RUNTIME_FIELDS = frozenset(('physics:velocity', 'physics:angularVelocity', 'xformOp:translate',
    'xformOp:orient', 'xformOp:scale', 'xformOpOrder',
    'state:angular:physics:position', 'state:angular:physics:velocity'))


def semantic_value(value):
    """Deterministic diagnostic rendering, without float rounding/tolerances."""
    if isinstance(value, float) and not math.isfinite(value):
        return dict(type='float', value=str(value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): semantic_value(value[key]) for key in sorted(value, key=str)}
    if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
        return dict(type=type(value).__name__, real=semantic_value(value.GetReal()),
            imaginary=semantic_value(value.GetImaginary()))
    try:
        values = list(value)
    except TypeError:
        # Sdf listOps and paths expose stable semantic str, not object repr.
        return dict(type=type(value).__name__, value=str(value))
    return dict(type=type(value).__name__, values=[semantic_value(v) for v in values])


def layer_snapshot(text):
    from pxr import Sdf
    suffix = '.usda' if text.lstrip().startswith('#usda') else '.sdf'
    layer = Sdf.Layer.CreateAnonymous('h4_readonly_layer_comparison'+suffix)
    if not layer.ImportFromString(text):
        raise ValueError('Cannot parse isolated layer COPY; never import into the live layer')
    paths = set()
    layer.Traverse(Sdf.Path.absoluteRootPath, lambda path: paths.add(str(path)))
    paths.add('/')
    result = {}
    for path in sorted(paths):
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            # Traverse also reports relationship/connection target paths. They
            # have no independent Spec; the owning property's exact listOp is
            # recorded in targetPaths/connectionPaths below, never discarded.
            if Sdf.Path(path).IsTargetPath():
                continue
            raise ValueError('Layer spec is missing at traversed path: '+path)
        result[path] = dict(spec_type=type(spec).__name__,
            info={str(key): semantic_value(spec.GetInfo(key)) for key in sorted(spec.ListInfoKeys())})
    return result


def compare_layers(before_text, current_text, disk_text):
    from pxr import Sdf
    before, current, disk = map(layer_snapshot, (before_text, current_text, disk_text))
    missing = {'field_is_not_authored': True}
    changes = []
    for path in sorted(set(before)|set(current)):
        first, last, original = before.get(path, {}), current.get(path, {}), disk.get(path, {})
        a, b, d = first.get('info', {}), last.get('info', {}), original.get('info', {})
        keys = sorted(set(a)|set(b))
        if first.get('spec_type') != last.get('spec_type'):
            keys = ['__spec_type__', *keys]
        for key in keys:
            old = first.get('spec_type', missing) if key == '__spec_type__' else a.get(key, missing)
            new = last.get('spec_type', missing) if key == '__spec_type__' else b.get(key, missing)
            source = original.get('spec_type', missing) if key == '__spec_type__' else d.get(key, missing)
            if old == new:
                continue
            sdf_path = Sdf.Path(path)
            old_actor = sdf_path.GetPrimPath().HasPrefix(Sdf.Path(cohort.OLD_ROBOT))
            runtime_property = old_actor and sdf_path.IsPropertyPath() and sdf_path.name in RUNTIME_FIELDS
            label = 'OLD_actor_runtime_property_candidate' if runtime_property else (
                'OLD_actor_other_field_REVIEW_REQUIRED' if old_actor else 'OUTSIDE_OLD_actor_REVIEW_REQUIRED')
            changes.append(dict(path=path, field=key, before=old, current=new, unchanged_disk_source=source,
                current_matches_unchanged_disk_authored_field=new == source,
                diagnostic_classification=label, automatically_waived=False))
    return dict(raw_root_text_equal=before_text == current_text,
        parsed_layer_field_values_equal=not changes, changed_field_count=len(changes), changed_fields=changes,
        unified_root_text_diff='\n'.join(difflib.unified_diff(before_text.splitlines(), current_text.splitlines(),
            fromfile='actual_archived_pre_Stop_Root', tofile='actual_current_STOPPED_Root', n=3)),
        all_changes_require_review_NO_tolerance_or_actor_subtree_exemption=True,
        diagnostic_comparison_does_NOT_resolve_original_HOLD=True)


def inspect(stage, timeline):
    prior_result = json.loads(prior.RESULT.read_bytes()); cohort.verify(prior_result['input_sha256'])
    _, archived, _ = prior.load_context()
    if stage is None:
        raise ValueError('Existing live stage required; never reopen the source')
    root_text, session_text = stage.GetRootLayer().ExportToString(), stage.GetSessionLayer().ExportToString()
    source = cohort.SOURCE.read_text()
    root_before_path, session_identifier = str(stage.GetRootLayer().realPath), stage.GetSessionLayer().identifier
    state_before = dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
        current_time_s=float(timeline.get_current_time()))
    observed = prior.inspect(stage, timeline)
    difference = compare_layers(archived['root_layer_text_before_Stop'], root_text, source)
    state_after = dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
        current_time_s=float(timeline.get_current_time()))
    if (root_text != stage.GetRootLayer().ExportToString() or session_text != stage.GetSessionLayer().ExportToString()
            or state_before != state_after or cohort.completion.sha(cohort.SOURCE) != cohort.SOURCE_SHA):
        raise ValueError('Live source/Root/Session/timeline changed during the get-only inspection; no continuation')
    pins = {**observed['input_sha256'], str(prior.RESULT): cohort.completion.sha(prior.RESULT),
        str(Path(__file__)): cohort.completion.sha(Path(__file__)), str(ENTRY): cohort.completion.sha(ENTRY)}
    cohort.verify(pins)
    return dict(version='H4_completed_Stop_actual_Root_field_diff_capture_GET_ONLY_v1',
        status='ACTUAL_ROOT_DIFF_CAPTURED_FOR_OFFLINE_REVIEW_NOT_STAGING_OR_STATIC_ADMISSION',
        actual_timeline=state_before, current_boundary_checks=observed['checks'],
        previous_readback_verdict=prior_result['status'], previous_readback_immutable=True,
        original_preparation_failure_immutable=True, actual_H3_archive_immutable=True,
        actual_live_root_identifier=root_before_path, actual_live_session_identifier=session_identifier,
        actual_current_root_layer_text=root_text, actual_current_session_layer_text=session_text,
        actual_current_root_text_sha256=hashlib.sha256(root_text.encode()).hexdigest(),
        actual_current_session_text_sha256=hashlib.sha256(session_text.encode()).hexdigest(),
        root_difference=difference, source_USD_file_sha256=cohort.completion.sha(cohort.SOURCE),
        timeline_root_session_and_source_unchanged_by_inspection=True,
        actual_H4_staging_or_static_or_drive_admission=False,
        additional_Stop_Play_pause_reset_reopen_save_or_restore=False,
        no_physics_initialization_tensor_reconstruction_or_caster_assignment=True,
        CICS_runtime_integration=False, use_for_final_statistics=False, input_sha256=pins)
