"""FRESH cohort after v2 reverse callback error; SAME forward observation."""

import hospital_h3b_prefix_response_observation_v2 as previous
from hospital_h3b_prefix_response_observation_v2 import *
from hospital_runtime_interface import audit_exports,probe_contact_classifier


VERSION='H3B_same_actual_CP50_large_angle_prefix_passive_observation_v3'
COHORT=BASE/'cohort_03'
OUTPUT,OBSERVER_OUTPUT=COHORT/'prefix',COHORT/'instrumentation'
RESULT=OUTPUT/'hospital_h3b_prefix_response_r01.json'
PROTOCOL_PATH=BASE/'protocol_r03/actual_prefix_observation_design.json'
PREPARATION_PROTOCOL_PATH=PROTOCOL_PATH
PREPARE=ENTRIES/'isaac_prepare_ijcas_hospital_h3b_prefix_response_observation_v3.py'
PRIMARY=ENTRIES/'isaac_run_ijcas_hospital_h3b_prefix_response_primary_v3.py'
OBSERVER=ENTRIES/'isaac_run_ijcas_hospital_h3b_prefix_response_instrumented_v3.py'


def entry_texts():
    replacements=[(previous.PREPARE.name,PREPARE.name),(previous.PRIMARY.name,PRIMARY.name),(previous.OBSERVER.name,OBSERVER.name),
        ('hospital_h3b_prefix_response_observation_v2.py','hospital_h3b_prefix_response_observation_v3.py')]
    result={}
    for kind,text in previous.entry_texts().items():
        for before,after in replacements:text=text.replace(before,after)
        # Pure export/classifier audit precedes scheduling/actions and is also
        # checked from the frozen protocol. No callback counters are touched.
        if kind in ('primary','prepare'):
            text=text.replace("    h['load_protocol']()","    h['load_protocol'](); h['validate_runtime_api']()")
        else:
            text=text.replace('    reviewed=helper["load_protocol"]()',
                '    reviewed=helper["load_protocol"](); helper["validate_runtime_api"]()')
        result[kind]=text
    return result


def validate_runtime_api():
    exports=audit_exports(globals(),(PREPARE,PRIMARY,OBSERVER))
    first=json.loads(REFERENCE.read_bytes());prep=first['physics_preparation_snapshot']
    return dict(exports=exports,**probe_contact_classifier(classify_contact,ROBOT_ROOT,WHEELS,
        prep['support_floor'],prep['disabled_legacy_floor_colliders']))


def function_contract():
    expected=entry_texts();paths=dict(prepare=PREPARE,primary=PRIMARY,observer=OBSERVER)
    for kind,path in paths.items():
        if ast.dump(ast.parse(path.read_text()),include_attributes=False)!=ast.dump(ast.parse(expected[kind]),include_attributes=False):
            raise ValueError('Fresh-prefix entry differs beyond declared cohort/admission/export preflight changes')
    for old,new,names in ((previous.PRIMARY,PRIMARY,('physics_step','contacts','angles','verify_gains','set_hold','command','measure','environmental_guard','sample')),
            (previous.OBSERVER,OBSERVER,('step','contacts','remember'))):
        trees=[{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text()))
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} for p in (old,new)]
        if any(trees[0][name]!=trees[1][name] for name in names):raise ValueError('Original prefix control/safety/passive callback changed')
    return dict(actual_prefix_control_safety_and_cleanup_callbacks_AST_unchanged=True,
        passive_callbacks_AST_unchanged=True,runtime_interface_and_classifier_probes=validate_runtime_api(),
        input_sha256={str(p):sha(p) for p in (*paths.values(),previous.PRIMARY,previous.OBSERVER)})


def load_protocol():
    record=json.loads(PROTOCOL_PATH.read_bytes());verify_sources(record);verify_files(record['input_sha256'])
    if (record.get('version')!=VERSION or record.get('controller_contract')!=function_contract()
            or record.get('status')!='FRESH_IDENTICAL_PREFIX_AFTER_CALLBACK_FAILURE_STATIC_SDK_PENDING'
            or record.get('same_fixed_prefix_control_and_physics') is not True
            or record.get('whole_H3_drive_admission') is not False):
        raise ValueError('Only one fresh identical prefix, no failed-endpoint recovery or whole-route admission')
    return record
