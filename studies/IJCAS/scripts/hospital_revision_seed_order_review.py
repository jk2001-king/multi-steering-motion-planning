"""Small offline regression check for a geometry-seed ordering hypothesis.

Keep the original successful B1 family first, but cap its first slice at10s
so B0 still has20s within the same45s seed budget. This may preserve easy
original paths without starving the harder B0 fallback. Same rule for every
task; no cached path, moved goal, different model or extra wall budget.
"""
from pathlib import Path
from types import SimpleNamespace

import hospital_revision_uniform_P_compatibility as base

runtime = base.runtime
OUTPUT = base.OUTPUT.parent/'B1_first_seed_order_regression_v1'
PROTOCOL = OUTPUT/'protocol.json'
CELL_IDS = ('H2_P_dynamics_full_envelope_s05', 'H3_P_dynamics_full_envelope_s01')


def export_protocol():
    parent = runtime.read(base.PROTOCOL)
    campaign = base.followup.campaign
    value = dict(parent, version='COMMON_B1_10_B0_20_B2_REMAINDER45_SEED_REGRESSION_v1',
        main_cells=[c for name in CELL_IDS for c in parent['main_cells'] if c['cell_id'] == name],
        P_seed_portfolio=[dict(method=m, maximum_s=s) for m, s in
            ((campaign.METHODS[1], 10.), (campaign.METHODS[0], 20.), (campaign.planning.B2, 45.))],
        sole_additional_change='owned spatial seed family order; same45s seed/180s total',
        task_specific_seed_policy_NOT_used=True,
        previous_uniform_v3_regression_retained=True, actual_motion_trials=0,
        final_statistics=False, independent_new_repetitions=0)
    if len(value['main_cells']) != len(CELL_IDS):
        raise ValueError('Exactly the explicit declared condition list required')
    if PROTOCOL.exists():
        if runtime.read(PROTOCOL) != value:
            raise ValueError('Seed-order regression definition frozen')
    else:
        runtime.evidence.write_exclusive(PROTOCOL, value)
        files = (Path(__file__), Path(base.__file__), Path(base.search.__file__),
            Path(base.followup.__file__), Path(base.planning.__file__))
        pins = {str(p.resolve()): runtime.sha(p) for p in files}
        runtime.evidence.write_exclusive(OUTPUT/'sources.json', pins)
        runtime.lc.freeze_workflow_sources(OUTPUT/'sources.zip', pins)
    return PROTOCOL


def verify_sources():
    runtime.retained.runtime.task.verify_pins(runtime.read(OUTPUT/'sources.json'))


def labelled_seed(*args, **kwargs):
    value = runtime.bind(base.followup.seed_portfolio, PROTOCOL=PROTOCOL)(*args, **kwargs)
    # The shared historical function's fixed text label does not determine
    # behavior: it reads the actual order from PROTOCOL. Preserve that old
    # text for provenance and give the real new order an unambiguous label.
    return dict(value, inherited_seed_policy_text_NOT_actual_order=value.get('owned_seed_policy'),
        owned_seed_policy='COMMON_B1_10_B0_20_B2_REMAINDER45_v1')


def run():
    export_protocol(); verify_sources()
    if (OUTPUT/'reserved.json').exists():
        raise ValueError('Already consumed offline candidate; no automatic retry')
    runtime.evidence.write_exclusive(OUTPUT/'reserved.json', dict(
        planning_attempts=len(CELL_IDS), actual_motion_trials=0))
    results = []
    for name in CELL_IDS:
        value = runtime.bind(base.run_cell, OUTPUT=OUTPUT, PROTOCOL=PROTOCOL,
            verify_sources=verify_sources, __file__=__file__)(name)
        results.append(value)
        print('[SEED ORDER SCREEN]', len(results), '/', len(CELL_IDS), name,
            value['plan_success'], flush=True)
    value = dict(status='SEED_ORDER_REGRESSION_SCREEN_FINISHED', results=results,
        actual_motion_trials=0, final_statistics=False)
    runtime.evidence.write_exclusive(OUTPUT/'completed.json', value)
    return value


def all_compatibility():
    """Same validated seed order, prospectively frozen for all30 conditions."""
    names = tuple(c['cell_id'] for c in runtime.read(base.PROTOCOL)['main_cells'])
    if len(names) != 30:
        raise ValueError('Original six tasks times five P conditions required')
    root = base.OUTPUT.parent/'uniform_B1_first_native_P_compatibility_v1'
    protocol = root/'protocol.json'
    verify = runtime.bind(verify_sources, OUTPUT=root)
    export = runtime.bind(export_protocol, OUTPUT=root, PROTOCOL=protocol, CELL_IDS=names)
    followup = SimpleNamespace(**dict(vars(base.followup),
        seed_portfolio=runtime.bind(labelled_seed, PROTOCOL=protocol)))
    private_base = SimpleNamespace(**dict(vars(base),
        run_cell=runtime.bind(base.run_cell, followup=followup)))
    return runtime.bind(run, OUTPUT=root, PROTOCOL=protocol, CELL_IDS=names,
        export_protocol=export, verify_sources=verify, base=private_base)()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='new finite all30 compatibility screen')
    args = parser.parse_args()
    all_compatibility() if args.all else run()
