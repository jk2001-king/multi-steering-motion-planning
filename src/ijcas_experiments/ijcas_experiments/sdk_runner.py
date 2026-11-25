"""Fresh SDK process adapter. Never imported by check/plan/analyze."""
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import sys
import traceback


def bind_relocated_support(context):
    """Bind byte-verified frozen support data to unchanged live readback.

    Do not regenerate the historical design using obsolete development
    inputs. The original mesh/material/collision checks and full 4997-row
    preview remain mandatory. Original evidence bytes are never edited.
    """
    from hospital_controlled_runtime import bind
    from ijcas_paths import ROOT
    from ijcas_experiments.workflow import sha
    prep = context.preparation
    original = prep.old.original
    check = original.verify_actual_support
    readback = check.__globals__['support_readback_checks']
    support = readback.__globals__['support']
    manifest = {r['path']: r['sha256_original'] for r in
        (json.loads(line) for line in (ROOT / 'migration/sha256_manifest.jsonl').read_text().splitlines())}
    for path in (support.TARGET, support.LIVE):
        if sha(path) != manifest[str(path.relative_to(ROOT))]:
            raise ValueError('Original frozen support artifact changed: ' + str(path))
    def frozen_design():
        if sha(support.TARGET) != manifest[str(support.TARGET.relative_to(ROOT))]:
            raise ValueError('Frozen support design changed during execution')
        return json.loads(support.TARGET.read_text())
    private_support = SimpleNamespace(**dict(vars(support), load_design=frozen_design))
    private_readback = bind(readback, support=private_support)
    prep.old = SimpleNamespace(**dict(vars(prep.old), original=SimpleNamespace(
        **dict(vars(original), verify_actual_support=bind(check, support_readback_checks=private_readback)))))
    return context


def configure_paths():
    root = Path(__file__).resolve().parents[3]
    for folder in (root / 'tools', root / 'studies/IJCAS/scripts', root / 'isaac/scripts',
                   root / 'src/patient_initial_stability', root / 'src/ijcas_experiments'):
        sys.path.insert(0, str(folder))
    return root


def app_configuration(spec):
    """Retain the GUI experiment's timeline mode in the Python experience.

    Isaac5.0 full.kit sets this to true; base.python inherits false. The
    latter advances the timeline by float32 loop dt rather than the original
    fixed timeline ticks, changing threshold crossings and sensor scheduling.
    This restores the application clock, not a controller/safety threshold.
    """
    return dict(headless=spec['headless'],
        extra_args=['--/app/player/useFixedTimeStepping=true'])


def current_authority(spec, output):
    if spec.get('stage') != 'run' or spec.get('allow_motion') is not True:
        raise ValueError('A current explicit run/allow_motion=true selection is required')
    if spec.get('feedback') not in ('ground_truth', 'amcl_encoder'):
        raise ValueError('One explicit feedback backend required; never fall back')
    if spec['feedback'] == 'amcl_encoder' and (spec.get('scenario') not in ('H5', 'H6')
            or spec.get('method') != 'P' or not spec.get('sensor_case')):
        raise ValueError('One separate registered H5/H6 P sensor case required')
    protocol = output / 'protocol.json'
    if protocol.is_file():
        raw = json.loads(protocol.read_text())
        if raw.get('current_motion_authorized') is not True or raw['main_cells'] != [spec['cell']]:
            raise ValueError('Selection must match its one prospectively frozen motion protocol')
    return dict(approval_source='CURRENT_EXPLICIT_CLI_SELECTION',
        selection_file=str(output / 'selection.json'), cell_id=spec['cell']['cell_id'],
        scope='ONE_NEW_SOURCE_ONE_STATIC_CONDITIONAL_ONE_ROUTE_NO_RETRY',
        original_paper_denominator_unchanged=True)


def build_live_binding(spec, output):
    """Bind the existing final hosts, V7 controller, static gate and restoration."""
    import hospital_revision_final_campaign as final
    import hospital_revision_live as live
    from hospital_controlled_runtime import bind
    from ijcas_experiments.workflow import sha
    authority = current_authority(spec, output)
    protocol = output / 'protocol.json'
    cell = spec['cell']
    source = bind(final.source_binding, OUTPUT=output, PROTOCOL=protocol,
                  EXECUTION_PROTOCOL=protocol)
    campaign = SimpleNamespace(**dict(vars(final.design.original), OUTPUT=output, PROTOCOL=protocol))
    factory = bind(final.bound_cell, OUTPUT=output, PROTOCOL=protocol,
                   EXECUTION_PROTOCOL=protocol, source_binding=source,
                   campaign_namespace=lambda: campaign)
    call = factory(cell)
    old = call.__globals__['runtime']
    old_context = old.context_for
    def selected_context(*args):
        return bind_relocated_support(old_context(*args))
    def check(value):
        if value != authority:
            raise ValueError('Current exact selected run authority required; archived approvals are not accepted')
        return dict(authority)
    contract = bind(old.controlled_contract, APPROVAL=authority)
    private = SimpleNamespace(**dict(vars(old), APPROVAL=authority, context_for=selected_context,
        checked_approval=check, controlled_contract=contract))
    private.static_packet = bind(old.static_packet, APPROVAL=authority, checked_approval=check)
    private.route_packet = bind(old.route_packet, APPROVAL=authority, controlled_contract=contract)
    parent_static = old.ControlledStaticHost
    class SelectedStaticHost(parent_static):
        __init__ = bind(old.lc.PreparedHeldStaticSDKHost.__init__, checked_fresh_held_approval=check)
    private.ControlledStaticHost = SelectedStaticHost
    parent_route = call.__globals__['PrimaryRouteHost']
    class SelectedRouteHost(parent_route):
        __init__ = bind(live.primary_initializer, runtime=private)()
        def save(self, report, rows):
            return live.PrimaryRouteHost.save(self, dict(report, standalone_selected_run=True,
                use_for_final_statistics=False, final_fixed_version_trial=False,
                development_trial_NOT_final_repetition=True,
                original_paper_result_replacement=False, fresh_selection=str(output / 'selection.json')), rows)
    base_packet = bind(live.task_route_packet, runtime=private)
    def packet(*args, **kwargs):
        path, value, plan = base_packet(*args, **kwargs)
        value = dict(value)
        value['input_sha256'].update(json.loads((output / 'sources.json').read_text()))
        value['input_sha256'][str(protocol)] = sha(protocol)
        return path, value, plan
    def checked_policy(path):
        value = json.loads(Path(path).read_text())
        if value != dict(authority=authority, protocol_sha256=sha(protocol),
            model_is_prediction_not_actual_failure=True, independent_full_envelope_audit_required=True,
            actual_static_PASS_required=True, unchanged_actual_controller_limits_and_success_criteria=True,
            no_physical_retry=True):
            raise ValueError('Exact prospective diagnostic policy required')
        return value
    return bind(call, runtime=private, PrimaryRouteHost=SelectedRouteHost,
        task_route_packet=packet, checked_model_policy=checked_policy), authority


async def prepare_environment(spec, output):
    import omni.usd
    import omni.timeline
    import omni.kit.app
    import hospital_session_bootstrap as bootstrap
    import hospital_h4_cohort_preparation as constructor
    from hospital_controlled_runtime import bind
    from ijcas_paths import load_archived_json, resolve_archived_path
    context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing() or not timeline.is_stopped() or timeline.get_current_time() != 0.:
        raise ValueError('Only a fresh untouched STOPPED process may prepare the environment')
    if Path(spec['scene']).resolve() != constructor.SOURCE.resolve():
        raise ValueError('Only the registered and validated hospital source is supported')
    success, error = await context.open_stage_async(spec['scene'])
    if not success:
        raise RuntimeError('Cannot open selected source: ' + str(error))
    for unused in range(30):
        await omni.kit.app.get_app().next_update_async()
    original_build = bind(bootstrap.build_static_patch, read=load_archived_json)
    def build():
        # USD layer strings are not JSON path fields. Relocate only asset
        # identifiers in the private patch, never the archived layer bytes.
        from pxr import UsdUtils
        from ijcas_experiments.workflow import write, sha
        patch, pins = original_build()
        changes = {}
        def relocate(asset):
            if not Path(asset).is_absolute():
                return asset
            mapped = str(resolve_archived_path(asset))
            if mapped != asset:
                if not Path(mapped).is_file():
                    raise FileNotFoundError('Migrated static asset missing: ' + mapped)
                changes[asset] = dict(path=mapped, sha256=sha(mapped))
                pins[mapped] = sha(mapped)
            return mapped
        UsdUtils.ModifyAssetPaths(patch, relocate)
        write(output / 'static_asset_relocation.json', dict(
            original_records_unchanged=True, assets=changes,
            complete_4997_row_preview_and_live_validation_required=True))
        return patch, pins
    prepare = bind(bootstrap.prepare_environment_once, OUTPUT=output / 'environment',
                   build_static_patch=build)
    await prepare(authorized=True)


async def run_selected(spec, output):
    import builtins
    from ijcas_experiments.workflow import sha, write, private_planner
    import hospital_revision_final_campaign as final
    from hospital_controlled_runtime import bind
    current_authority(spec, output)
    if spec['feedback'] == 'amcl_encoder':
        from ijcas_experiments.sensor_runner import run
        result = await run(spec, output)
        write(output / 'execution_result.json', result)
        return result
    verify = private_planner(spec, output)[1]
    verify()
    if getattr(builtins, 'ijcas_hospital_latest_task_experiment', None) is not None:
        raise ValueError('Fresh isolated process only; never adopt another experiment')
    # Calculate the existing two-resolution diagnostic model in this SDK,
    # before any environment change, Play or motor access.
    model = bind(final.model_cell, OUTPUT=output, PROTOCOL=output / 'protocol.json',
        EXECUTION_PROTOCOL=output / 'protocol.json', verify_sources=verify,
        folder_for=lambda name: output / 'planning' / name)
    model(spec['cell']['cell_id'])
    verify()
    await prepare_environment(spec, output)
    job = asyncio.current_task()
    builtins.ijcas_hospital_latest_task_experiment = job
    builtins.ijcas_revision_primary_task = job
    call, authority = build_live_binding(spec, output)
    policy = output / 'diagnostic_model_policy.json'
    write(policy, dict(authority=authority, protocol_sha256=sha(output / 'protocol.json'),
        model_is_prediction_not_actual_failure=True, independent_full_envelope_audit_required=True,
        actual_static_PASS_required=True, unchanged_actual_controller_limits_and_success_criteria=True,
        no_physical_retry=True))
    result = await call(spec['cell']['cell_id'], prior_result=None, authorized=True, model_policy_path=policy)
    verify()
    result = dict(result, standalone_selected_run=True, original_results_unchanged=True,
                  automatic_retry=False, simulator_started=True)
    if result.get('actual_route_started'):
        result['metrics'] = final.metrics.metrics(Path(result['report']))
        result['metrics']['scope'] = 'standalone simulator commissioning; excluded from fixed paper statistics'
    write(output / 'execution_result.json', result)
    return result


def main():
    configure_paths()
    from ijcas_experiments.workflow import write
    if len(sys.argv) != 3:
        raise ValueError('Exactly selection.json and one reserved output directory required')
    selection_file, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if selection_file != output / 'selection.json':
        raise ValueError('One own reserved selection required')
    spec = json.loads(selection_file.read_text())
    current_authority(spec, output)
    if (output / 'sdk_reserved.json').exists():
        raise FileExistsError('Consumed SDK attempt; no automatic retry')
    write(output / 'sdk_reserved.json', dict(selection_file=str(selection_file),
        maximum_static_trials=1, maximum_route_trials=1, automatic_retry=False))
    # Do not let Kit interpret the selection/output paths as application args.
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp
    app = SimulationApp(app_configuration(spec))
    failed = False
    try:
        import carb
        import isaacsim
        full_app = Path(isaacsim.__file__).resolve().parent / 'apps/isaacsim.exp.full.kit'
        from ijcas_experiments.workflow import sha
        fixed_clock = carb.settings.get_settings().get('/app/player/useFixedTimeStepping')
        if fixed_clock is not True:
            raise ValueError('Original full-application fixed timeline stepping required before any trial')
        write(output / 'simulation_clock_configuration.json', dict(
            setting='/app/player/useFixedTimeStepping', requested=True, actual=fixed_clock,
            original_application_definition=str(full_app), source_sha256=sha(full_app),
            reason='Restore Isaac5.0 full GUI timeline mode, not base.python default',
            physics_dt_and_controller_thresholds_unchanged=True,
            no_timestamp_rounding_rebasing_or_recorded_pose_replay=True))
        job = asyncio.ensure_future(run_selected(spec, output))
        while not job.done() and app.is_running():
            app.update()
        if not job.done():
            job.cancel()
            while not job.done():
                app.update()
            raise RuntimeError('Application closed during a bounded trial; no retry')
        job.result()
    except BaseException as error:
        failed = True
        write(output / 'sdk_error.json', dict(error=type(error).__name__ + ': ' + str(error),
            traceback=traceback.format_exc(),
            automatic_retry=False, physical_completion_NOT_claimed=True))
    finally:
        app.close()
    # Some Kit shutdown paths terminate with status zero. The parent also
    # requires a result receipt and checks sdk_error.json independently.
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
