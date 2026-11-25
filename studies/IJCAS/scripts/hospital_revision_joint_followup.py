"""Targeted paired development of previously uncompleted joint conditions.

Reuse the original draws and source overlays, initialized-joint USD proofs,
the common execution candidate, and the shared fresh-source/static/route
runner. No physical loop is cloned. Original n5 statistics remain intact.
Only failed P joint conditions and their paired B2 controls are selected.
"""
import asyncio
import copy
import csv
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_terminal_geometry as control
import hospital_revision_accepted_reference_validation as shared
import hospital_revision_joint_writeback as joints
import hospital_revision_model_artifacts as artifacts

runtime = control.runtime
OUTPUT = control.OUTPUT.parent/'targeted_joint_conditions_common_v7'
PROTOCOL = OUTPUT/'protocol.json'
UNIFORM = control.OUTPUT.parent/'uniform_B1_first_native_P_compatibility_v1'
UNIFORM_MODELS = control.OUTPUT.parent/'uniform_B1_first_native_v7_model_screen_v1'
LEDGER = control.parent.parent.base.shared.campaign.OUTPUT/'final_evidence/20261121_233144_359073/joint_initial_individual_results.csv'


def definition():
    original = runtime.read(joints.initial.PROTOCOL)
    with LEDGER.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    failed = {r['paired_condition_id'] for r in rows
        if r['method'] == joints.campaign.planning.P and r['completed'] != 'True'}
    cells = [copy.deepcopy(c) for pair in dict.fromkeys(c['paired_condition_id']
        for c in original['main_cells'] if c['paired_condition_id'] in failed)
        for method in (joints.campaign.planning.P, joints.campaign.planning.B2)
        for c in original['main_cells'] if c['paired_condition_id'] == pair and c['method'] == method]
    if len(cells) != 2*len(failed) or len(failed) != 9:
        raise ValueError('Exactly nine archived P noncompletions and nine paired B2 controls')
    return dict(version='TARGETED_ORIGINAL_JOINT_CONDITION_PAIRED_COMMON_V7_DEVELOPMENT_v1',
        main_cells=cells, tasks=original['tasks'],
        common_speed_policy=runtime.read(joints.campaign.PROTOCOL)['common_speed_policy'],
        controller_version=control.VERSION, old_joint_ledger=str(LEDGER), old_joint_ledger_sha256=runtime.sha(LEDGER),
        original_joint_protocol=str(joints.initial.PROTOCOL), original_joint_protocol_sha256=runtime.sha(joints.initial.PROTOCOL),
        only_nine_original_failed_P_conditions_and_their_B2_controls=True,
        original_draws_and_immutable_source_overlays_reused=True,
        uniform_new_P_plans_vs_original_B2_own_plans=True,
        same_common_controller_speed_braking_hold_PD_and_actual_acceptance=True,
        original60_joint_results_and120_main_results_NOT_replaced=True,
        seen_condition_development_NOT_final_independent_repetitions=True,
        planning_times_reused_NOT_new_planning_repetitions=True,
        nominal_model_does_NOT_simulate_initial_passive_caster_contact_states=True,
        independent_full_envelope_audit_and_actual_own_static_PASS_required=True,
        all_failed_or_infrastructure_attempts_preserved=True,
        no_automatic_physical_retry=True, one_static_and_conditional_one_route_per_new_actor=True,
        stop_on_any_P_noncompletion_or_unrestored_boundary=True,
        completed_safe_B2_noncompletion_may_continue_as_negative_comparison=True,
        real_platform_photo_deferred_by_author=True, manuscript_or_sensor_changes=False)


def folders():
    return {c['cell_id']: ((UNIFORM/'planning') if c['method'] == joints.campaign.planning.P
        else control.parent.parent.base.shared.planning.OUTPUT)/c['parent_main_cell_id']
        for c in definition()['main_cells']}


def export_protocol():
    value = definition(); inputs = {}
    for folder in folders().values():
        summary = runtime.read(folder/'summary.json')
        paths = [folder/'summary.json']
        if summary['plan_success']:
            plan = Path(summary['plan']); audit = runtime.read(folder/'independent_plan_audit.json')
            if audit['verdict'] != 'PASS' or audit['plan_sha256'] != runtime.sha(plan):
                raise ValueError('Own unchanged full-envelope audit required')
            paths.extend((plan, folder/'independent_plan_audit.json', folder/'task_geometry.json', folder/'planning_occupancy.npz'))
        inputs.update({str(p.resolve()): runtime.sha(p) for p in paths})
    value['reference_inputs_sha256'] = inputs
    files = (Path(__file__), Path(control.__file__), Path(shared.__file__), Path(joints.__file__),
        Path(joints.initial.__file__), Path(joints.serialization.__file__), Path(joints.serialization.follow.__file__),
        Path(artifacts.__file__), *control.extra_source_paths, Path(control.base.__file__),
        Path(control.base.goal.__file__), Path(control.base.shared.__file__))
    pins = {str(p.resolve()): runtime.sha(p) for p in files}
    value['source_sha256'] = pins
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Targeted original-joint follow-up definition is frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        runtime.evidence.write_exclusive(OUTPUT/'sources.json', pins)
        runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip', pins)
    return PROTOCOL


def verify_sources():
    raw = runtime.read(PROTOCOL)
    runtime.retained.runtime.task.verify_pins(raw['source_sha256'])
    runtime.retained.runtime.task.verify_pins(raw['reference_inputs_sha256'])


def planning_namespace():
    return SimpleNamespace(**dict(vars(control.base.shared.planning), OUTPUT=shared.ReferenceFolders(folders())))


def scoped_control():
    return SimpleNamespace(**dict(vars(control), extra_source_paths=(
        *control.extra_source_paths, Path(shared.__file__), Path(__file__), Path(joints.__file__))))


def model_screen(name):
    cell = next(c for c in definition()['main_cells'] if c['cell_id'] == name)
    folder = folders()[name]
    if not runtime.read(folder/'summary.json')['plan_success']:
        return dict(cell_id=name, status='PARENT_B2_PLANNING_FAILURE_PRESERVED_NO_PHYSICAL_RUN')
    target = OUTPUT/'model'/name/'model_screen.json'
    if cell['method'] == joints.campaign.planning.P:
        source = UNIFORM_MODELS/'model'/cell['parent_main_cell_id']/'model_screen.json'
        raw = runtime.read(source)
        if raw['controller_version'] != control.VERSION or raw['plan_sha256'] != runtime.sha(raw['plan']):
            raise ValueError('Same exact controller and nominal model plan required for artifact reuse')
        for row in raw['cases']:
            if runtime.sha(row['trace']) != row['trace_sha256']:
                raise ValueError('Existing nominal model trace changed')
        value = dict(raw, reused_nominal_report=str(source), reused_nominal_report_sha256=runtime.sha(source),
            new_model_runs=0, initial_caster_contact_effects_NOT_predicted=True,
            original_actual_joint_conditions_still_require_new_static_and_route=True)
        if target.exists():
            if runtime.read(target) != value: raise ValueError('Frozen reused model artifact changed')
        else: runtime.evidence.write_exclusive(target, value)
    else:
        scope = dict(control=scoped_control(), OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
            PLAN_FOLDERS=folders(), __file__=__file__)
        value = runtime.bind(shared.model_screen, **dict(scope, planning_namespace=planning_namespace,
            export_protocol=export_protocol, verify_sources=verify_sources,
            model_binding=runtime.bind(shared.model_binding, **scope)))(name)
    packet = folder/'executor_model_screen/model_screen.json'
    # Original B2 references already own their original diagnostic-model
    # packet. Its checks stay unchanged; the V7 result remains separately
    # available above. New P references require an exact owned-model packet.
    if not packet.exists():
        artifacts.publish(folder, target, controller_version=control.VERSION)
    return value


def bound_cell(cell):
    common = control.validation_cell()
    initialized = joints.cell_bindings(cell)
    old = common.__globals__['runtime']; init = initialized.__globals__['runtime']
    def context_for(folder, actor, geometry):
        return init.context_for(folder, actor+'_common_terminal20_joint_v7', geometry)
    private = SimpleNamespace(**dict(vars(old), context_for=context_for,
        prepare_source=init.prepare_source, ControlledStaticHost=joints.InitializedJointStaticHost))
    parent = common.__globals__['PrimaryRouteHost']
    class CommonInitializedHost(parent):
        subscribe = joints.route_subscriber()
        check_post_observation_layers = joints.InitializedJointRouteHost.check_post_observation_layers
    campaign = SimpleNamespace(**dict(vars(control.base.shared.campaign), OUTPUT=OUTPUT, PROTOCOL=PROTOCOL))
    checked = runtime.bind(control.base.shared.live.checked_model_policy, campaign=campaign)
    original_packet = common.__globals__['task_route_packet']
    def packet(*args, **kwargs):
        path, value, plan = original_packet(*args, **kwargs)
        value = copy.deepcopy(value)
        value['input_sha256'].update(runtime.read(PROTOCOL)['source_sha256'])
        value['input_sha256'][str(PROTOCOL)] = runtime.sha(PROTOCOL)
        return path, value, plan
    return runtime.bind(common, runtime=private, campaign=campaign, planning=planning_namespace(),
        checked_model_policy=checked, PrimaryRouteHost=CommonInitializedHost, task_route_packet=packet)


def policy_file():
    return runtime.bind(control.base.shared.policy_file, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
        candidate=control.candidate())()


async def run(previous, *, authorized=False):
    if authorized is not True: raise ValueError('Current autonomous scoped experiment authority required')
    import builtins
    prior = await asyncio.shield(previous)
    if getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not previous:
        raise ValueError('Exact latest completed task required; no parallel physics trials')
    passed = {r['cell_id'] for r in prior.get('results', []) if r.get('actual_verdict') == 'PASS'}
    if not {'H4_P_dynamics_full_envelope_s04','H3_P_dynamics_full_envelope_s01'} <= passed:
        raise ValueError('Both main-condition P verifications must pass before joint follow-up')
    last = Path(prior['last_actual_report']); verify_sources(); results = []
    if not control.base.shared.execution.restored_boundary(runtime.read(last)):
        raise ValueError('Actual saved Pause and original all-joint restoration required')
    current = asyncio.current_task(); builtins.ijcas_hospital_latest_task_experiment = current
    builtins.ijcas_revision_primary_task = current
    for cell in runtime.read(PROTOCOL)['main_cells']:
        name = cell['cell_id']; folder = folders()[name]
        verify_sources()
        if not runtime.read(folder/'summary.json')['plan_success']:
            outcome = dict(cell_id=name,status='PLANNING_FAILURE_NO_PHYSICAL_RUN',actual_route_started=False,
                actual_verdict='PLAN_FAILURE',original_B2_planning_failure_not_relabelled=True)
        else:
            model = runtime.read(OUTPUT/'model'/name/'model_screen.json')
            if cell['method'] == joints.campaign.planning.P and not model['both_nominal_resolutions_pass']:
                raise ValueError('Own P model artifacts must both pass before actual motion')
            outcome = await bound_cell(cell)(name,prior_result=last,authorized=True,model_policy_path=policy_file())
            last = Path(outcome['report'])
            if not control.base.shared.execution.restored_boundary(runtime.read(last)):
                raise ValueError('Actual Pause/restoration missing; stop queue')
            verify_sources()
            outcome = dict(outcome,metrics=control.base.shared.metrics.metrics(last)
                if outcome['actual_route_started'] else None)
        outcome.update(controller_version=control.VERSION,known_condition_development_NOT_final_repetition=True)
        runtime.evidence.write_exclusive(OUTPUT/'outcomes'/(name+'.json'),outcome); results.append(outcome)
        print('[TARGETED JOINT RESULT]',name,outcome['actual_verdict'],flush=True)
        if cell['method'] == joints.campaign.planning.P and outcome['actual_verdict'] != 'PASS':
            value = dict(status='P_NONCOMPLETION_REQUIRES_DIAGNOSIS_NO_AUTO_RETRY',results=results,last_actual_report=str(last))
            runtime.evidence.write_exclusive(OUTPUT/'stopped.json',value); return value
    value = dict(status='TARGETED_JOINT_PAIRED_DEVELOPMENT_FINISHED',results=results,last_actual_report=str(last),
        original_n5_statistics_never_replaced=True,automatic_retry=False)
    runtime.evidence.write_exclusive(OUTPUT/'completed.json',value); return value


if __name__ == '__main__':
    export_protocol(); verify_sources()
    for cell in definition()['main_cells']:
        print(cell['cell_id'], model_screen(cell['cell_id']).get('both_nominal_resolutions_pass'), flush=True)
