"""Offline ablation of within-primitive full-envelope sampling, not body size.

Keep the real rectangle, 60mm padding, map, response propagation, history,
force/load and search settings. Compare checking all propagated poses with
checking only the first and last sampled poses of each primitive. The real
independent all-sample audit is never weakened and no path is driven here.
"""
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_planning as planning

runtime = planning.runtime
campaign = planning.campaign
OUTPUT = campaign.GATE / 'revision_followup_20261122/transition_envelope_sampling_v1'
PROTOCOL = OUTPUT / 'protocol.json'


def endpoint_checker(base):
    class EndpointSampledFullBody(base):
        def __call__(self, poses):
            values = list(poses)
            return super().__call__(values if len(values) <= 2 else [values[0], values[-1]])
    return EndpointSampledFullBody


def export_protocol():
    parent = runtime.read(campaign.PROTOCOL)
    value = dict(version='WITHIN_PRIMITIVE_FULL_ENVELOPE_SAMPLING_ABLATION_v1',
        main_cells=[c for c in parent['main_cells'] if c['method'] == campaign.planning.P
            and c['task'] in ('H2', 'H3')],
        tasks=parent['tasks'], common_speed_policy=parent['common_speed_policy'],
        original_evidence_inputs_sha256=parent['original_evidence_inputs_sha256'],
        paired_full_sampling_control=str(campaign.PROTOCOL),
        paired_full_sampling_control_sha256=runtime.sha(campaign.PROTOCOL),
        original_seed_policy_lookahead8_parameters_and_prediction_retained=True,
        only_first_last_sample_checked_during_primitive_search=True,
        body_rectangle_m=[1.630, .965], per_side_planning_padding_m=.060,
        independent_all_sample_rectangle_audit_unchanged=True,
        full_envelope_point_reference_ablation_is_separate=True,
        physical_trials=0, unsafe_paths_NOT_executed=True,
        all_negative_and_null_effect_findings_retained=True,
        no_claim_of_continuous_time_3D_collision_guarantee=True)
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Envelope sampling ablation already frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        sources = (Path(__file__), Path(planning.__file__))
        runtime.lc.freeze_workflow_sources(OUTPUT / 'sources.zip',
            {str(p.resolve()): runtime.sha(p) for p in sources})
    return PROTOCOL


def run_cell(cell_id):
    export_protocol()
    def bindings(*args, **kwargs):
        method, audit, connections, parameters, activity = planning.planner_bindings(*args, **kwargs)
        base = method.__globals__['HospitalEnvelopeChecker']
        method = runtime.bind(method, HospitalEnvelopeChecker=endpoint_checker(base))
        return method, audit, connections, parameters, activity
    private = SimpleNamespace(**dict(vars(campaign), PROTOCOL=PROTOCOL, OUTPUT=OUTPUT))
    run = runtime.bind(planning.run_cell, campaign=private, OUTPUT=OUTPUT/'planning',
        planner_bindings=bindings, __file__=__file__)
    result = run(cell_id, lookahead_edges=8)
    audit_path = OUTPUT/'planning'/cell_id/'independent_plan_audit.json'
    audit = runtime.read(audit_path) if audit_path.exists() else None
    value = dict(result, variant='P_FIRST_LAST_POSES_FULL_RECTANGLE',
        full_sampling_independent_audit=audit['verdict'] if audit else 'NOT_RUN_NO_PLAN',
        full_sampling_colliding_samples=audit.get('colliding_samples') if audit else None,
        physical_trials=0, not_an_actual_collision_or_success=True)
    runtime.evidence.write_exclusive(OUTPUT/'results'/(cell_id+'.json'), value)
    return value


def run():
    export_protocol()
    runtime.evidence.write_exclusive(OUTPUT/'reservation.json', dict(
        attempts=10, physical_trials=0, no_automatic_retry=True))
    results = []
    for cell in runtime.read(PROTOCOL)['main_cells']:
        results.append(run_cell(cell['cell_id']))
    runtime.evidence.write_exclusive(OUTPUT/'completed.json', dict(results=results, physical_trials=0))
    return results


if __name__ == '__main__':
    run()
