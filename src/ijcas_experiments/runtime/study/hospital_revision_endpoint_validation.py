"""Finite shared-V5 H4 paired test, then the accepted native H3 reference.

The previous H4 timeout stays intact. Reuse every shared source preparation,
static screen, route monitor, Pause and gain restoration operation. This
module only declares checked inputs, source pins, sequence and admission.
"""
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_accepted_reference_validation as shared
import hospital_revision_endpoint_handoff as endpoint

runtime = shared.runtime
OUTPUT = endpoint.OUTPUT
PROTOCOL = endpoint.PROTOCOL
P = endpoint.parent.base.shared.campaign.planning.P
B2 = endpoint.parent.base.shared.campaign.planning.B2
H4_IDS = ('H4_'+P+'_s04', 'H4_'+B2+'_s04')
PLAN_FOLDERS = {name: endpoint.parent.base.shared.planning.OUTPUT/name for name in H4_IDS}
PLAN_FOLDERS['H3_'+P+'_s01'] = shared.PLAN_FOLDERS['H3_'+P+'_s01']
PRIOR_REPORT = endpoint.parent.OUTPUT/'live'/H4_IDS[0]/'route/route_readback.json'
PRIOR_SHA256 = '41d6e0881beef89be899c22e24ad1a39b357277ee6a6fc493e6de1e3ba6296b8'
control = SimpleNamespace(**dict(vars(endpoint),
    extra_source_paths=(*endpoint.extra_source_paths, Path(shared.__file__))))


def bind(function, **extra):
    scope = dict(control=control, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL, PLAN_FOLDERS=PLAN_FOLDERS,
        PREDECESSOR_POLICY='exact V4 H4s04 timeout preserved; actual Pause and original gains/targets restored',
        __file__=__file__)
    scope.update(extra)
    return runtime.bind(function, **scope)


def export_protocol():
    return bind(shared.export_protocol)()


def verify_sources():
    return bind(shared.verify_sources)()


def planning_namespace():
    return bind(shared.planning_namespace)()


def model_binding(*args, **kwargs):
    return bind(shared.model_binding)(*args, **kwargs)


def model_screen(name):
    return bind(shared.model_screen, planning_namespace=planning_namespace,
        export_protocol=export_protocol, verify_sources=verify_sources,
        model_binding=model_binding)(name)


def bound_cell():
    return bind(shared.bound_cell, planning_namespace=planning_namespace)()


def policy_file():
    return bind(shared.policy_file)()


def predecessor_admitted(previous):
    return (Path(previous['last_actual_report']) == PRIOR_REPORT
        and runtime.sha(PRIOR_REPORT) == PRIOR_SHA256
        and runtime.read(PRIOR_REPORT)['verdict'] == 'HOLD')


def review():
    export_protocol(); verify_sources()
    target = OUTPUT/'archived_handoff_review.json'
    if target.exists():
        return runtime.read(target)
    rows = []
    for path in sorted((endpoint.parent.OUTPUT/'outcomes').glob('*.json')):
        outcome = runtime.read(path)
        value = endpoint.replay(outcome['report'])
        value.update(cell_id=outcome['cell_id'])
        rows.append(value)
        print('[REFERENCE END REPLAY]', value['cell_id'],
            value['original_controller_replay_verified'], value['candidate_predicate_difference_count'], flush=True)
    value = dict(records=rows, all_original_replays_exact=all(r['original_controller_replay_verified'] for r in rows),
        passed_recorded_trials_unaffected=all(r['candidate_predicate_difference_count'] == 0
            for r in rows if r['original_verdict'] == 'PASS'),
        failed_H4_connection_identified=any(r['cell_id'] == H4_IDS[0]
            and r['candidate_predicate_difference_count'] > 0 for r in rows),
        candidate_NOT_applied_to_archived_observations=True,
        no_counterfactual_physical_success_or_RMS_claim=True, actual_motion_trials=0)
    runtime.evidence.write_exclusive(target, value)
    return value


async def run(parent, *, authorized=False):
    return await bind(shared.after_current, predecessor_admitted=predecessor_admitted,
        verify_sources=verify_sources, policy_file=policy_file, bound_cell=bound_cell)(
            parent, authorized=authorized)


def dispatch():
    export_protocol(); verify_sources()
    checked = runtime.read(OUTPUT/'archived_handoff_review.json')
    if not all(checked[k] for k in ('all_original_replays_exact',
            'passed_recorded_trials_unaffected', 'failed_H4_connection_identified')):
        raise ValueError('Exact source-matched replay and scoped handoff impact review required')
    for name in PLAN_FOLDERS:
        screen = runtime.read(OUTPUT/'model'/name/'model_screen.json')
        if '_P_' in name and not screen['both_nominal_resolutions_pass']:
            raise ValueError('Own two-resolution P model screens required')
    source = ("import sys,types,asyncio,builtins\nfrom pathlib import Path\n"
        "parent=getattr(builtins,'ijcas_revision_hold_damping_v4_policy_checked',None)\n"
        "assert parent is not None and parent.done() and not parent.exception(),'Exact completed V4 predecessor required'\n"
        "assert getattr(builtins,'ijcas_hospital_latest_task_experiment',None) is parent,'Intervening motion'\n"
        "assert not hasattr(builtins,'ijcas_revision_stopped_endpoint_v5'),'Already reserved'\n"
        f"p=Path({str(Path(__file__).resolve())!r})\n"
        "m=types.ModuleType('hospital_revision_endpoint_validation');m.__file__=str(p)\n"
        "sys.modules[m.__name__]=m\nexec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\n"
        "job=asyncio.ensure_future(m.run(parent,authorized=True))\n"
        "builtins.ijcas_revision_stopped_endpoint_v5=job\n"
        "print({'common_stopped_endpoint_H4_pair_then_H3_scheduled':True,'automatic_retry':False})\n")
    runtime.evidence.write_exclusive(OUTPUT/'reservation.json', dict(source=source,
        maximum_routes=3, one_static_per_fresh_source=True, automatic_retry=False))
    return endpoint.parent.base.shared.runner.remote(source)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('review', 'models', 'dispatch'))
    args = parser.parse_args()
    if args.action == 'review':
        print(review())
    elif args.action == 'models':
        for name in PLAN_FOLDERS:
            model_screen(name)
    else:
        print(dispatch())
