"""Serialization-only Session-copy adapter for either USD or Sdf text cookies.

The v1 offline artifact stays intact and its entry has never been run live.
No geometry, initial seed, gain, acceptance or allowed mutation is relaxed.
"""

import json
from pathlib import Path

import hospital_h4_cohort_preparation as original


ROOT, GATE, support, completion = original.ROOT, original.GATE, original.support, original.completion
OLD_ROBOT, ROBOT, MARKER = original.OLD_ROBOT, original.ROBOT, original.MARKER
ASSET, ASSET_SHA = original.ASSET, original.ASSET_SHA
SETUP, MAP_REVIEW, MAP, MODELS = original.SETUP, original.MAP_REVIEW, original.MAP, original.MODELS
OUTPUT, ARCHIVE, LIVE, FAILURE = original.OUTPUT, original.ARCHIVE, original.LIVE, original.FAILURE
PREFLIGHT = OUTPUT/'preflight_v2/cohort_preflight.json'
ENTRY = ROOT/'isaac/scripts/isaac_prepare_ijcas_hospital_h4_fresh_cohort_v2.py'
SOURCE, SOURCE_SHA = original.SOURCE, original.SOURCE_SHA
VERSION = original.VERSION+'_USD_Sdf_text_cookie_v2'
MATERIAL, GROUND, WHEELS, JOINT_SETTINGS = original.MATERIAL, original.GROUND, original.WHEELS, original.JOINT_SETTINGS
canonical, verify = original.canonical, original.verify
support_covers, support_area_missing = original.support_covers, original.support_area_missing
sampled_support_screen = original.sampled_support_screen
joint_seed, validate_new_robot, author_new_robot = original.joint_seed, original.validate_new_robot, original.author_new_robot


def protected_session_text(text):
    """Select matching parser, then perform the EXACT same strict scope check."""
    if text.startswith('#usda '):
        suffix = '.usda'
    elif text.startswith('#sdf '):
        suffix = '.sdf'
    else:
        raise ValueError('Unknown Session text format; no skipped delta validation')
    from pxr import Sdf
    copy = Sdf.Layer.CreateAnonymous('H4_protected_session_delta'+suffix)
    if not copy.ImportFromString(text):
        raise ValueError('Cannot inspect isolated Session layer copy')
    for path in (OLD_ROBOT, ROBOT, MARKER):
        if copy.GetPrimAtPath(path):
            edit = Sdf.BatchNamespaceEdit(); edit.Add(path, Sdf.Path.emptyPath)
            if not copy.Apply(edit):
                raise ValueError('Cannot compare authorized Session change scope')
    return copy.ExportToString()


def prepare():
    report = original.prepare()
    report.update(version=VERSION,
        serialization_only_adapter_for_both_USD_and_Sdf_Session_text=True,
        scientific_constraints_and_authorized_mutation_scope_unchanged=True,
        original_v1_offline_preflight_retained=str(original.PREFLIGHT),
        original_v1_entry_never_executed_live=True)
    paths = (Path(__file__), ENTRY, original.PREFLIGHT)
    report['input_sha256'].update({str(p): completion.sha(p) for p in paths})
    report['source_sha256'].update({p.name: completion.sha(p) for p in (Path(__file__), ENTRY)})
    verify(report['input_sha256'])
    return canonical(report)


def load_preflight():
    report = json.loads(PREFLIGHT.read_bytes())
    verify(report['input_sha256'])
    if report != prepare():
        raise ValueError('H4 cohort preflight changed; no substitute task, source or settings')
    return report
