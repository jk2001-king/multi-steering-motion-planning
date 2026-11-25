"""Repair the missing classifier export; NEW FRESH prefix only, never retry v2."""

from types import FunctionType

import hospital_h3b_reverse_response_observation_v2 as previous
from hospital_h3b_reverse_response_observation_v2 import *
import hospital_h3b_prefix_response_observation_v3 as prefix
from hospital_runtime_interface import audit_exports,probe_contact_classifier


OBSERVATION_VERSION='H3B_same_actual_first_reverse_six_wheel_response_observation_v3'
classify_contact=prior.classify_contact
BASE,COHORT=prefix.BASE,prefix.COHORT
FIRST=prefix.RESULT
OUTPUT,OBSERVER_OUTPUT=COHORT/'reverse',COHORT/'reverse_instrumentation'
RESULT=OUTPUT/'hospital_h3b_reverse_response_r01.json'
LIVE=COHORT/'reverse_handoff/live_handoff.json'
# No new model is invented: these eight design cases may be reused ONLY
# after a COMPLETE fresh source-to-prefix CSV and final twelve-joint match.
ADMISSION=previous.ADMISSION
PROTOCOL_PATH=BASE/'reverse_protocol_r03/reverse_response_observation_design.json'
PRIMARY=ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_response_primary_v3.py'
OBSERVER=ENTRIES/'isaac_run_ijcas_hospital_h3b_reverse_response_instrumented_v3.py'
HANDOFF=ENTRIES/'isaac_review_ijcas_hospital_h3b_reverse_response_handoff_v3.py'


def load_profile():
    # Same original admission code, copied namespace ONLY for declared fresh
    # record/cohort references. Original module/function globals stay intact.
    fn=previous.load_profile
    return FunctionType(fn.__code__,dict(fn.__globals__,FIRST=FIRST,prefix=prefix,
        PREFIX_ANALYSIS=previous.PREFIX_ANALYSIS,__file__=__file__),fn.__name__)()


def load_reviewed_profile():
    info=load_profile();record=json.loads(ADMISSION.read_bytes());validate_model(record)
    if (record['first_result']!=str(previous.FIRST) or record['first_result_sha256']!=sha(previous.FIRST)
            or record['initial_actual_world_xyyaw']!=info['initial_world_xyyaw']
            or record['retained_all_joint_positions_rad']!=info['retained_all_joint_positions_rad']
            or record['suffix_candidate_sha256']!=info['suffix_candidate_sha256']
            or record['strict_map_digest']!=info['strict_map_digest']):
        raise ValueError('Model reuse requires EXACT fresh complete-prefix/body/twelve-joint equivalence; never reset to match')
    info.update(admission=str(ADMISSION),admission_sha256=sha(ADMISSION),
        exact_seed_component_model_screens_reused_without_recomputation=True,
        input_sha256={**info['input_sha256'],**record['input_sha256'],str(ADMISSION):sha(ADMISSION)})
    return info


def load_admitted_profile():
    fn=previous.load_admitted_profile
    return FunctionType(fn.__code__,dict(fn.__globals__,load_reviewed_profile=load_reviewed_profile,
        load_protocol=load_protocol,LIVE=LIVE,FIRST=FIRST,PROTOCOL_PATH=PROTOCOL_PATH),fn.__name__)()


def entry_texts():
    replacements=[(previous.PRIMARY.name,PRIMARY.name),(previous.OBSERVER.name,OBSERVER.name),(previous.HANDOFF.name,HANDOFF.name),
        ('hospital_h3b_reverse_response_observation_v2.py','hospital_h3b_reverse_response_observation_v3.py')]
    result={}
    for kind,text in previous.entry_texts().items():
        for before,after in replacements:text=text.replace(before,after)
        if kind=='primary':
            needle="h=loader(scripts/'hospital_h3b_reverse_response_observation_v3.py');info=h['load_admitted_profile']()"
            text=text.replace(needle,"h=loader(scripts/'hospital_h3b_reverse_response_observation_v3.py');h['validate_runtime_api']();info=h['load_admitted_profile']()")
            text=text.replace('fresh_model_screens_recomputed_from_NEW_actual_prefix=True,',
                'fresh_model_screens_recomputed_from_NEW_actual_prefix=False,\n            exact_complete_fresh_prefix_equivalence_for_existing_eight_model_screens=True,')
        elif kind=='observer':
            needle="rules = load(scripts/'hospital_h3b_reverse_response_observation_v3.py'); info = rules['load_admitted_profile']()"
            text=text.replace(needle,"rules = load(scripts/'hospital_h3b_reverse_response_observation_v3.py'); helper['validate_runtime_api'](); info = rules['load_admitted_profile']()")
        else:
            text=text.replace("h['load_protocol']()","h['load_protocol']();h['validate_runtime_api']()")
        result[kind]=text
    return result


def validate_runtime_api():
    exports=audit_exports(globals(),(PRIMARY,OBSERVER,HANDOFF))
    first=json.loads(previous.FIRST.read_bytes());prep=first['physics_preparation_snapshot']
    if classify_contact is not prior.classify_contact:raise ValueError('Original classifier identity changed')
    return dict(exports=exports,**probe_contact_classifier(classify_contact,ROBOT_ROOT,WHEELS,
        prep['support_floor'],prep['disabled_legacy_floor_colliders']))


def function_contract():
    generated=entry_texts();paths=dict(primary=PRIMARY,observer=OBSERVER,handoff=HANDOFF)
    for kind,path in paths.items():
        if ast.dump(ast.parse(path.read_text()),include_attributes=False)!=ast.dump(ast.parse(generated[kind]),include_attributes=False):
            raise ValueError('Callback repair entry changed beyond declared export/fresh-cohort/admission/provenance changes')
    for old,new,names in ((previous.PRIMARY,PRIMARY,('numeric','physics_step','contacts','angles','verify_gains','measure','set_hold','command','sample')),
            (previous.OBSERVER,OBSERVER,('step','contacts','remember'))):
        trees=[{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text()))
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for p in (old,new)]
        if any(trees[0][name]!=trees[1][name] for name in names):raise ValueError('Existing control/safety/passive event code changed')
    return dict(all_nine_primary_callbacks_AST_unchanged=True,same_original_measured_terminal_controller_AND_evaluator=True,
        passive_callbacks_AST_unchanged=True,missing_export_repaired_as_original_classifier_alias=True,
        runtime_interface_and_classifier_probes=validate_runtime_api(),
        input_sha256={str(p):sha(p) for p in (*paths.values(),previous.PRIMARY,previous.OBSERVER,previous.HANDOFF)})


def load_protocol():
    record=json.loads(PROTOCOL_PATH.read_bytes());verify_sources(record);verify_files(record['input_sha256'])
    if (record.get('version')!=OBSERVATION_VERSION or record.get('controller_contract')!=function_contract()
            or record.get('status')!='CLASSIFIER_EXPORT_REPAIRED_FRESH_PREFIX_AND_LIVE_SDK_PENDING'
            or record.get('fresh_prefix_required_no_failed_state_retry') is not True
            or record.get('whole_H3_drive_admission') is not False or record.get('new_model_deployed') is not False):
        raise ValueError('Only a fresh-prefix identical reverse observation after explicit callback repair')
    return record


def review(info):
    fn=previous.review
    return FunctionType(fn.__code__,dict(fn.__globals__,load_protocol=load_protocol,
        function_contract=function_contract,PROTOCOL_PATH=PROTOCOL_PATH),fn.__name__)(info)
