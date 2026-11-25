"""Same single zero-command trial; reviewed Root metadata + same-SDK source.

No physics settings, controller, rays, criteria or historical outcomes change.
The v1 task was rejected before reservation/Play. This is NOT an automatic
retry; the user must explicitly launch this separately identified entry.
"""

import hashlib
from pathlib import Path

import hospital_h4_subfloor_compact_static_gate as previous
import hospital_h4_namespace_reference as reference


preparation,cohort=previous.preparation,previous.cohort
ROOT,GATE,ROBOT,MARKER=previous.ROOT,previous.GATE,previous.ROBOT,previous.MARKER
VERSION='H4_compact_static_exact_namespace_Root_binding_same_SDK_source_reference_v2'
DESIGN=GATE/'hospital_h4_static_admission'/preparation.COHORT/'namespace_review_r01'
PROTOCOL=DESIGN/'static_protocol_v2.json'
ENTRY=ROOT/'isaac/scripts/isaac_run_ijcas_hospital_h4_subfloor_compact_static_gate_v2.py'
OUTPUT=GATE/'hospital_h4_static_contact'/preparation.COHORT/'namespace_review_r01'
RUN_ID='hospital_h4_subfloor_compact_static_v2_r01'
RESULT,TRACE,RESERVATION,PREFLIGHT_HOLD=(OUTPUT/n for n in (RUN_ID+'.json',RUN_ID+'.csv','single_attempt_reserved.json','preflight_hold.json'))
DIAGNOSTICS=DESIGN/'preflight_diagnostics'
SETTLE_S,RECORD_S,PHYSICS_HZ=previous.SETTLE_S,previous.RECORD_S,previous.PHYSICS_HZ
GUARDS,LIMITS,QUERY_LIMITS=previous.GUARDS,previous.LIMITS,previous.QUERY_LIMITS
BAD_CONTACTS,WHEELS,SUPPORTS,KNOWN_CATEGORIES=previous.BAD_CONTACTS,previous.WHEELS,previous.SUPPORTS,previous.KNOWN_CATEGORIES
classify_contact,evaluate,evaluate_subfloor_queries=previous.classify_contact,previous.evaluate,previous.evaluate_subfloor_queries
merge_evaluation,passive_final_link_readback=previous.merge_evaluation,previous.passive_final_link_readback
actor_configuration,critical_environment=previous.actor_configuration,previous.critical_environment
load_actual_preparation=previous.load_actual_preparation


def prepare_protocol():
    base=previous.load_protocol();review=reference.load_review()
    # Replace an environment-dependent opaque actor hash by an explicit policy
    # requiring a complete same-registry source reconstruction at runtime.
    base={k:v for k,v in base.items() if k not in ('nominal_actor_fixed_configuration','source_recipe')}
    files=(Path(__file__),ENTRY,Path(reference.__file__),reference.REVIEW,previous.PROTOCOL,
        ROOT/'studies/IJCAS/scripts/prepare_hospital_h4_subfloor_compact_static_gate_v2.py')
    return dict(base,version=VERSION,run_id=RUN_ID,
        status='SAME_UNSTARTED_STATIC_TRIAL_EXACT_REVIEWED_METADATA_AND_SAME_SDK_REFERENCE_V2',
        reviewed_Root_sha256=review['exact_actual_Root_sha256'],namespace_metadata_review=str(reference.REVIEW),
        actor_reference_policy='ALL actor attributes under the SAME USD/schema registry and immutable source constructor; no added exclusions',
        full_fixed_actor_configuration_from_same_SDK_source_required=True,
        original_motion_exclusions_unchanged_no_physical_parameter_waiver=True,
        v1_preflight_exception_preserved_NOT_a_physical_FAIL_or_a_new_PASS=True,
        original_v1_trial_files_must_all_remain_absent=True,
        previous_single_static_approval_does_NOT_automatically_schedule_this_version=True,
        input_sha256={**base['input_sha256'],**review['input_sha256'],**{str(path):preparation.sha(path) for path in files}})


def load_protocol():
    frozen=preparation.read_small(PROTOCOL);preparation.verify(frozen['input_sha256'])
    if frozen!=prepare_protocol():
        raise ValueError('Exact v2 static reference policy changed; no Play')
    source=reference.source_reference(frozen['start_xyyaw'])
    return dict(frozen,nominal_actor_fixed_configuration=source['fixed_configuration_rows'],
        runtime_anonymous_source_reference=source)


def validate_stopped_handoff(stage,timeline,protocol,report):
    if any(path.exists() for path in (previous.RESERVATION,previous.PREFLIGHT_HOLD,previous.RESULT,previous.TRACE)):
        raise ValueError('v1 actually reserved or began a trial; do not bypass its single-attempt boundary')
    state=dict(playing=bool(timeline.is_playing()),stopped=bool(timeline.is_stopped()),current_time_s=float(timeline.get_current_time()))
    if (state!=dict(playing=False,stopped=True,current_time_s=0.)
            or hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest()!=protocol['reviewed_Root_sha256']
            or protocol['new_robot_root']!=ROBOT):
        raise ValueError('State or EXACT reviewed Root changed; no normalization/recovery/Play')
    actual=actor_configuration(stage);expected=protocol['nominal_actor_fixed_configuration']
    if actual!=expected:
        changed=[k for k in sorted(set(actual)|set(expected)) if actual.get(k)!=expected.get(k)]
        raise ValueError('Same-SDK source actor configuration mismatch at: '+str(changed))
    if protocol['runtime_anonymous_source_reference']['source_constructor_readback']!=report['live_new_robot_readback']:
        raise ValueError('Anonymous same-SDK source reference does not reproduce the immutable constructor readback')
    return critical_environment(stage,report)
