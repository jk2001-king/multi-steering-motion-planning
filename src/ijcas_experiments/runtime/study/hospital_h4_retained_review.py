"""H4 asset review from the retained H3 FINAL PASS, NOT a change of task.

The old H4 inspector required reopening a STOPPED source. This separate entry
preserves the actual cohort_03 PAUSED endpoint and reads existing H4 assets.
It cannot set a task pose, map, collider, physics state or drive admission.
"""

from ijcas_paths import layout_path as _ijcas_path, workspace_root as _ijcas_workspace

import json
from pathlib import Path

import hospital_h3b_response_completion as completion
import hospital_h4_geometry_review as geometry


ROOT = completion.WORKSPACE
BASE = completion.final.GATE/'hospital_h4_geometry_review'
PREFLIGHT = BASE/'retained_preparation_r01/retained_geometry_preflight.json'
TARGET = BASE/'retained_final_r01/live_h4_geometry.json'
ENTRY = _ijcas_path(ROOT, 'isaac/scripts/isaac_review_ijcas_hospital_h4_retained_geometry.py')
AUDIT = completion.final.GATE/'hospital_asset_geometry_audit_v02.json'
SOURCE = _ijcas_path(ROOT, 'isaac/scenes/hospital_experiment.usda')
SOURCE_SHA = '9130af19a583e157b48137ed6ada13ac8540d1f26583f218bd3ed903906988a5'
VERSION = 'H4_authored_corridor_cart_review_from_retained_H3_final_v1'


def prepare():
    info, _, _, _, _, closed = completion.load_closeout(completion.DIRECTORY)
    live = json.loads(completion.LIVE.read_bytes())
    completion.verify_files(live['input_sha256'])
    if (live.get('status') != 'ACTUAL_COHORT03_FINAL_GOAL_FIVE_RESPONSE_OBSERVATIONS_CLOSED_NO_MOTION'
            or live['final_forward_result_sha256'] != info['final_forward_result_sha256']
            or live['root_session_timeline_body_joints_settings_unchanged'] is not True
            or live['saved_body_and_twelve_joint_handoffs_verified'] != 4
            or not all(live['original_final_goal_checks'].values())):
        raise ValueError('Exact actual final-response closeout must remain authoritative')
    if completion.sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Original hospital source changed; no substitute scene')
    original = json.loads(AUDIT.read_bytes())
    selected = [row for row in original['candidate_paths'] if row['path'] in geometry.REVIEW_PATHS]
    if len(selected) != len(geometry.REVIEW_PATHS):
        raise ValueError('Every original H4 wall/cart candidate is required')
    pins = {**closed['input_sha256'], **live['input_sha256'],
        **{str(path): completion.sha(path) for path in (completion.TARGET, completion.LIVE,
            AUDIT, SOURCE, ENTRY, Path(__file__), Path(geometry.__file__))}}
    completion.verify_files(pins)
    return dict(version=VERSION, status='READ_ONLY_H4_ASSET_REVIEW_PREPARED_CURRENT_H3_FINAL_MUST_BE_RETAINED',
        original_final_response_result=info['final_forward_result'],
        expected_current_actual_world_xyyaw=info['initial_world_xyyaw'],
        expected_all_twelve_joint_positions_rad=info['retained_all_joint_positions_rad'],
        original_H3_goal_and_all_PASS_FAIL_HOLDs_remain_unchanged=True,
        H3_is_NOT_abandoned_or_declared_impossible=True,
        review_region_xy_m=geometry.ROI, selected_original_asset_paths=list(geometry.REVIEW_PATHS),
        visual_and_authored_collision_inventory_only_NOT_cooked_PhysX_or_free_width_proof=True,
        current_flush_session_and_original_H1_H2_maps_are_NOT_assumed_identical=True,
        root_session_robot_joints_gains_collision_and_timeline_writes_permitted=False,
        manual_Play_Stop_reset_reopen_or_save_required=False,
        H4_start_goal_map_or_motion_prepared=False, physical_drive_admission=False,
        CICS_runtime_integration=False, use_for_final_statistics=False, live_receipt=str(TARGET),
        input_sha256=pins, source_sha256={path.name: completion.sha(path) for path in
            (Path(__file__), ENTRY, Path(geometry.__file__))})


def load_preflight():
    report = json.loads(PREFLIGHT.read_bytes())
    completion.verify_files(report['input_sha256'])
    completion.verify_sources(report)
    if report != prepare():
        raise ValueError('Prepared retained-state review changed; do not reset or use the old STOPPED inspector')
    return report
