"""Exact unvalued Namespace metadata binding, never a live-stage repair.

Anonymous copies only. Do not remove attributes or ignore actor subtrees.
The full actor configuration is compared under the SAME USD/schema registry
as its source reference, rather than trusting a cross-process opaque hash.
"""

import hashlib
from pathlib import Path

import hospital_h4_compact_static_boundary_review as fields
import hospital_h4_subfloor_compact_static_gate as prior


preparation=prior.preparation
ROOT, GATE, ROBOT=preparation.ROOT,preparation.GATE,preparation.ROBOT
CAPTURE=GATE/'hospital_h4_static_task_diagnosis/Root_boundary_20261116T130053340884Z.json'
TASK_CAPTURE=GATE/'hospital_h4_static_task_diagnosis/task_readback_20261116T125549741624Z.json'
WIDGET=Path('/home/jk/miniconda3/envs/isaac_env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.gui.property/isaacsim/gui/property/namespace.py')
NAME='isaac:namespace'
REVIEW=GATE/'hospital_h4_static_admission'/preparation.COHORT/'namespace_review_r01/namespace_source_review.json'
VERSION='H4_exact_unvalued_namespace_declaration_and_same_SDK_source_reference_v1'


def add_metadata_to_anonymous_layer(layer):
    """Build one explicit expected spec; refuse a real/source layer."""
    from pxr import Sdf
    if not layer.anonymous:
        raise ValueError('Only an isolated anonymous reference layer may be authored')
    owner=Sdf.CreatePrimInLayer(layer,ROBOT)
    if layer.GetPropertyAtPath(ROBOT+'.'+NAME):
        raise ValueError('Do not overwrite or normalize an existing namespace property')
    attr=Sdf.AttributeSpec(owner,NAME,Sdf.ValueTypeNames.String,Sdf.VariabilityVarying,False)
    attr.SetInfo('displayName','Namespace')
    attr.SetInfo('documentation','Namespace of the prim in Isaac Sim')
    return attr


def verify_exact_Root_addition(historical_text,current_layer):
    expected=fields.read_historical(historical_text)
    if expected.GetPrimAtPath(ROBOT):
        raise ValueError('Historical Root must not already contain this new actor spec')
    add_metadata_to_anonymous_layer(expected)
    if fields.snapshot(expected)!=fields.snapshot(current_layer):
        raise ValueError('Root differs beyond the EXACT eight unvalued Namespace declaration fields')
    return fields.compare(historical_text,current_layer)


def source_reference(start):
    """Pure USD in-memory source reference; NEVER Kit/context/PhysX APIs."""
    from pxr import Sdf,Usd,UsdShade
    stage=Usd.Stage.CreateInMemory('H4_ANONYMOUS_SDK_REFERENCE_NOT_LIVE_STAGING.usda')
    UsdShade.Material.Define(stage,preparation.old.constructor.MATERIAL)
    readback=preparation.old.constructor_for(ROBOT)(stage,start)
    add_metadata_to_anonymous_layer(stage.GetRootLayer())
    rows=prior.actor_configuration(stage)
    if len(rows)!=145:
        raise ValueError('Immutable source must retain all 145 actor nodes')
    # Do not assume an absent authored default means the SDK fallback is None.
    # Both reference and live actor are evaluated in the same schema registry.
    namespace=stage.GetPrimAtPath(ROBOT).GetAttribute(NAME)
    if namespace.GetTimeSamples() or namespace.GetConnections():
        raise ValueError('Unexpected namespace samples/connections in source reference')
    return dict(version=VERSION,USD_version=list(Usd.GetVersion()),source_constructor_readback=readback,
        fixed_configuration_rows=rows,namespace_resolved_value=fields.semantic(namespace.Get()),
        root_namespace_authored_spec=fields.snapshot(stage.GetRootLayer())[ROBOT+'.'+NAME],
        created_in_anonymous_noncontext_stage_only=True,
        any_live_actor_stage_or_physics_mutation=False,actual_static_or_drive_admission=False)


def captured_actor_stage(capture):
    """Copy only this actor; do not open Hospital network references."""
    from pxr import Sdf,Usd,UsdShade
    stage=Usd.Stage.CreateInMemory('H4_SAVED_ACTOR_RECONSTRUCTION_NOT_ACTUAL_STATE.usda')
    stage.DefinePrim('/World','Xform')
    UsdShade.Material.Define(stage,preparation.old.constructor.MATERIAL)
    for text,target in ((capture['actual_current_Root_text'],stage.GetRootLayer()),
            (capture['actual_current_Session_text'],stage.GetSessionLayer())):
        source=fields.read_historical(text)
        Sdf.CreatePrimInLayer(target,'/World')
        if source.GetPrimAtPath(ROBOT) and not Sdf.CopySpec(source,ROBOT,target,ROBOT):
            raise ValueError('Cannot reconstruct exact saved actor in anonymous copy')
    return stage


def build_review():
    p=preparation
    capture=p.read_small(CAPTURE);p.verify(capture['input_sha256'])
    task=p.read_small(TASK_CAPTURE)
    timeline=dict(playing=False,stopped=True,current_time_s=0.)
    if (capture['actual_timeline']!=timeline or capture['new_robot_root']!=ROBOT
            or capture['Root_Session_timeline_unchanged_by_readback'] is not True
            or capture['nominal_source_pose_seed_twelve_joints_and_parameters_match_preparation'] is not True
            or capture['actor_readback_errors'] or any(capture['static_attempt_files'].values())
            or task['after']['timeline']!=timeline
            or len(task['task_observation']['tasks'])!=1
            or task['task_observation']['tasks'][0]['state']!='DONE_WITH_EXCEPTION'):
        raise ValueError('Exact actual pre-Play rejection and own source-state capture required')
    old=p.read_small(p.old.BOUNDARY)['actual_root_layer_text']
    current=fields.read_historical(capture['actual_current_Root_text'])
    diff=verify_exact_Root_addition(old,current)
    if diff!=capture['boundary_difference'] or diff['changed_field_count']!=8:
        raise ValueError('Saved full Root field diff does not independently reproduce')
    if hashlib.sha256(capture['actual_current_Root_text'].encode()).hexdigest()!=capture['current_Root_sha256']:
        raise ValueError('Actual current Root capture hash differs')
    source=source_reference(prior.load_protocol()['start_xyyaw'])
    reconstructed=prior.actor_configuration(captured_actor_stage(capture))
    if reconstructed!=source['fixed_configuration_rows'] or len(reconstructed)!=145:
        raise ValueError('Saved actor differs from exact source reference beyond the reviewed metadata')
    widget=WIDGET.read_text()
    if ('SetDisplayName("Namespace")' not in widget
            or 'SetDocumentation("Namespace of the prim in Isaac Sim")' not in widget):
        raise ValueError('Installed widget definition no longer matches the exact observed metadata')
    files=(Path(__file__),Path(fields.__file__),CAPTURE,TASK_CAPTURE,WIDGET,prior.PROTOCOL,p.PROTOCOL,p.old.BOUNDARY,p.LIVE)
    return dict(version=VERSION,status='EXACT_NAMESPACE_METADATA_REVIEWED_SAME_SDK_LIVE_SOURCE_COMPARISON_REQUIRED',
        exact_actual_Root_sha256=capture['current_Root_sha256'],exact_Root_field_addition_count=8,
        complete_Root_field_diff=diff,installed_widget_metadata_definition_matches=True,
        actual_widget_event_or_fallback_causality_independently_traced=False,
        actual_live_v1_opaque_actor_check_was_False_and_is_NOT_relabelled=True,
        saved_actor_reconstruction_equals_source_under_identical_offline_USD_registry=True,
        source_actor_node_count=145,offline_anonymous_source_reference=source,
        original_preflight_rejection_preserved_and_no_physical_static_trial_had_started=True,
        only_reference_bookkeeping_changes_NO_physical_parameter_or_tolerance_change=True,
        no_live_Root_or_Session_edit_or_reset_required=True,
        actual_static_or_drive_admission=False,
        input_sha256={**capture['input_sha256'],**prior.load_protocol()['input_sha256'],
            **{str(path):p.sha(path) for path in files}})


def load_review():
    report=preparation.read_small(REVIEW);preparation.verify(report['input_sha256'])
    if (report['version']!=VERSION or report['status']!='EXACT_NAMESPACE_METADATA_REVIEWED_SAME_SDK_LIVE_SOURCE_COMPARISON_REQUIRED'
            or report['exact_Root_field_addition_count']!=8 or report['source_actor_node_count']!=145
            or report['actual_static_or_drive_admission'] is not False):
        raise ValueError('Exact bounded Namespace review required; never a physical admission')
    return report
