"""Passive force recording for ONE NEW timed second reverse; no actuation."""

from pathlib import Path

from hospital_h3b_rolling_observation import telemetry_snapshot, friction_snapshot, FRICTION_PERIOD_S
from hospital_h3b_rolling_observation import coverage as baseline_coverage
from hospital_h3b_timed_second_reverse import COHORT, controller_and_adapter_contract, sha


OUTPUT = COHORT/'instrumentation'
function_contract = controller_and_adapter_contract


def review(info):
    contract = function_contract()
    return dict(reference_trial=info['forward_result'], reference_trial_sha256=info['forward_result_sha256'],
        same_saved_actual_forward_PASS_body_and_12_joints=True,
        NEW_timed_second_reverse_NOT_a_planner_or_CICS_comparison=True,
        controller_contract=contract,
        input_sha256={**info['input_sha256'], **contract['input_sha256'], str(Path(__file__)): sha(__file__)})


def coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors):
    result = baseline_coverage(joint_rows, friction_rows, anchors, raw_contacts, primary, errors)
    if result['status'] == 'RECORDED_FOR_CAUSAL_REVIEW_NOT_A_ROLLING_OR_TRAVERSAL_PASS':
        result['status'] = 'RECORDED_FOR_TIMED_SECOND_REVERSE_REVIEW_NOT_A_MOTION_PASS'
    result.update(original_motion_verdict_is_authoritative=True,
        unique_caster_friction_or_drive_cause_isolated=False,
        projected_joint_effort_is_NOT_isolated_motor_torque=True)
    return result
