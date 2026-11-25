"""Restore TRUE full-body support screening; no physical/scientific waiver.

v1/v2 passed .010 as footprint_polygon's positional LENGTH, not clearance.
Those offline support screens are superseded, never physical results. All
geometry/model/gain/seed inputs remain unchanged. New checks explicitly bind
1.630 x .965 m and .010 m clearance and verify the resulting area and sides.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
import math
from pathlib import Path

import hospital_h4_cohort_preparation_v2 as previous
from hospital_geometry_tools import footprint_polygon


ROOT, GATE, support, completion = previous.ROOT, previous.GATE, previous.support, previous.completion
OLD_ROBOT, ROBOT, MARKER = previous.OLD_ROBOT, previous.ROBOT, previous.MARKER
ASSET, ASSET_SHA = previous.ASSET, previous.ASSET_SHA
SETUP, MAP_REVIEW, MAP, MODELS = previous.SETUP, previous.MAP_REVIEW, previous.MAP, previous.MODELS
OUTPUT, ARCHIVE, LIVE, FAILURE = previous.OUTPUT, previous.ARCHIVE, previous.LIVE, previous.FAILURE
PREFLIGHT = OUTPUT/'preflight_v3/cohort_preflight.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_prepare_ijcas_hospital_h4_fresh_cohort_v3.py')
SOURCE, SOURCE_SHA = previous.SOURCE, previous.SOURCE_SHA
VERSION = previous.VERSION+'_true_full_footprint_named_arguments_v3'
MATERIAL, GROUND, WHEELS, JOINT_SETTINGS = previous.MATERIAL, previous.GROUND, previous.WHEELS, previous.JOINT_SETTINGS
canonical, verify, support_covers = previous.canonical, previous.verify, previous.support_covers
joint_seed, validate_new_robot, author_new_robot = previous.joint_seed, previous.validate_new_robot, previous.author_new_robot
protected_session_text = previous.protected_session_text
FOOTPRINT = dict(length_m=1.630, width_m=.965, clearance_m=.010)


def full_footprint(pose):
    polygon = [tuple(map(float, p)) for p in footprint_polygon(pose,
        length=FOOTPRINT['length_m'], width=FOOTPRINT['width_m'], clearance=FOOTPRINT['clearance_m'])]
    expected = sorted([FOOTPRINT['length_m']+2*FOOTPRINT['clearance_m'],
        FOOTPRINT['width_m']+2*FOOTPRINT['clearance_m']]*2)
    sides = sorted(math.dist(a, b) for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    if (any(abs(a-b) > 1e-10 for a, b in zip(sides, expected))
            or abs(abs(support.area(polygon))-expected[0]*expected[-1]) > 1e-9):
        raise ValueError('Full 1.630 x .965 m plus 10 mm footprint dimensions were not preserved')
    return polygon


def support_area_missing(pose, covers):
    return sum(abs(support.area(p)) for p in support.uncovered([full_footprint(pose)], covers))


def sampled_support_screen(models, design):
    covers = support_covers(design)
    setup = json.loads(SETUP.read_bytes())
    results = []
    for method in models['methods']:
        poses = [setup['start_xyyaw']]+[s['world_xyyaw'] for s in method['selected_states']]
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
                maximum = max(maximum, support_area_missing(pose, covers)); count += 1
        results.append(dict(method=method['method'], pose_samples=count,
            maximum_uncovered_area_m2=maximum, sampled_full_footprint_support_covered=maximum <= 1e-9))
    return results


def prepare():
    if any(path.exists() for path in (ARCHIVE, LIVE, FAILURE)):
        raise ValueError('An earlier live cohort boundary already exists; this correction cannot reset or retry it')
    report = previous.prepare()
    design = support.load_design()
    setup = json.loads(SETUP.read_bytes())
    models = json.loads(MODELS.read_bytes())
    endpoint_missing = {key: support_area_missing(setup[key], support_covers(design))
        for key in ('start_xyyaw', 'goal_xyyaw')}
    screen = sampled_support_screen(models, design)
    if (setup['footprint'] != FOOTPRINT or any(value > 1e-9 for value in endpoint_missing.values())
            or len(screen) != 4 or not all(r['sampled_full_footprint_support_covered'] for r in screen)):
        raise ValueError('Correct FULL-envelope support screen failed; no staging/parameter/geometry waiver')
    report.update(version=VERSION, footprint=FOOTPRINT,
        actual_expanded_footprint_length_m=1.650, actual_expanded_footprint_width_m=.985,
        actual_expanded_footprint_area_m2=1.650*.985,
        true_full_footprint_endpoint_uncovered_area_m2=endpoint_missing,
        selected_native_model_path_sampled_support=screen,
        original_native_model_paths_start_goal_support_and_all_settings_unchanged=True,
        earlier_v1_v2_support_checks_invalidated_due_to_positional_length_argument=True,
        earlier_v1_v2_sources_and_receipts_retained_NOT_physical_or_comparison_evidence=True,
        no_prior_cohort_archival_or_staging_executed_at_offline_correction=True)
    paths = (Path(__file__), ENTRY, previous.PREFLIGHT)
    report['input_sha256'].update({str(p): completion.sha(p) for p in paths})
    report['source_sha256'].update({p.name: completion.sha(p) for p in (Path(__file__), ENTRY)})
    verify(report['input_sha256'])
    return canonical(report)


def load_preflight():
    report = json.loads(PREFLIGHT.read_bytes()); verify(report['input_sha256'])
    if report != prepare():
        raise ValueError('Corrected full-footprint H4 preflight changed; no substitute task or constraints')
    return report
