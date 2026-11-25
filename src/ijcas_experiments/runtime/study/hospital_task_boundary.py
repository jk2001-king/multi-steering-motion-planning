"""Reviewed, one-shot completion of an interrupted FIRST task staging.

No controller copy or change to historical pinned sources. A confirmed Stop
is consumed, never issued again. Only existing Top camera XY/aperture defaults
may differ in the nonplaying preparation review. Every other native field is
exact. Static/route loops and their strict recording boundaries stay original.
Importing this module performs no SDK action and creates no experiment files.
"""

import builtins
import hashlib
import json
import math
from pathlib import Path
import time

import hospital_task_runtime as runtime


VERSION = 'REVIEWED_COMPLETED_STOP_FIRST_STAGING_v1'
CAMERA = '/OmniverseKit_Top'
STOPPED = dict(playing=False, stopped=True, current_time_s=0.)
ACTIVE_TASKS = ('ijcas_hospital_latest_task_experiment', 'ijcas_h4_latest_experiment_task',
                'ijcas_h4_latest_path_task', 'ijcas_h4_latest_visual_refresh_task')
task = runtime.task
shared = task.shared


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _matrix(value):
    if not isinstance(value, dict) or set(value) != {'type', 'values'} or value['type'] != 'Matrix4d':
        raise ValueError('Exact native Matrix4d camera default required')
    rows = value['values']
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError('Four camera matrix rows required')
    result = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'type', 'values'} or row['type'] != 'Vec4d'
                or not isinstance(row['values'], list) or len(row['values']) != 4
                or not all(_number(v) for v in row['values'])):
            raise ValueError('Finite exact native camera matrix rows required')
        result.append(row['values'])
    return result


def _validate_display_row(row):
    path = row['path']
    if row['field'] != 'default':
        raise ValueError('Unreviewed field or spec metadata: ' + path + ':' + row['field'])
    if path in (CAMERA + '.horizontalAperture', CAMERA + '.verticalAperture'):
        if not all(_number(row[k]) and row[k] > 0 for k in ('before', 'after')):
            raise ValueError('Existing finite positive camera apertures required')
    elif path == CAMERA + '.xformOp:transform':
        first, last = _matrix(row['before']), _matrix(row['after'])
        if first[:3] != last[:3] or first[3][2:] != last[3][2:]:
            raise ValueError('Only camera XY translation may change; not orientation/Z/affine terms')
    else:
        raise ValueError('Non-display or unreviewed camera field changed: ' + path)


def display_only_differences(before, after):
    """Classify actual native deltas, not a broad ignored camera subtree."""
    differences = shared.layer_field_differences(before, after)
    for row in differences:
        _validate_display_row(row)
    return differences


def layer_review(before, after):
    """Root exact; Session only the explicitly classified display defaults."""
    root = shared.layer_field_differences(before['Root'], after['Root'])
    if root:
        raise ValueError('Root native fields changed: ' + repr(root[:8]))
    return dict(Root=root, Session=display_only_differences(before['Session'], after['Session']))


def staging_scope_review(context, before, after):
    """Do not adopt unrelated edits while intentionally creating the actor."""
    from pxr import Sdf
    if shared.layer_field_differences(before['Root'], after['Root']):
        raise ValueError('Original Root changed during Session-only first staging')
    changes = shared.layer_field_differences(before['Session'], after['Session'])
    display = []
    old = context.protocol['retired_robot_root']
    roots = [Sdf.Path(context.cohort.ROBOT), Sdf.Path(context.cohort.MARKER)]
    for row in changes:
        path = Sdf.Path(row['path'])
        if any(path.GetPrimPath().HasPrefix(root) for root in roots):
            continue  # Every new actor setting is independently source-checked.
        if row['path'] == old and row['field'] == 'active' and row['after'] is False:
            continue
        display.append(row)
    # Use the exact same narrow classifier on the remaining native records.
    for row in display:
        if not row['path'].startswith(CAMERA+'.'):
            raise ValueError('Unrelated Session edit during first staging: '+repr(row))
    # Camera defaults are the only remaining paths, but values/metadata must
    # still pass the classifier, without ignoring an entire camera subtree.
    for row in display:
        _validate_display_row(row)
    return dict(Session_changed_field_count=len(changes), display_differences=display,
                all_other_changes_confined_to_declared_first_staging=True)


def _read(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Bounded continuation receipt exceeded: ' + str(path))
    return json.loads(path.read_bytes())


def _no_attempt(context):
    if context.preparation.LIVE.exists():
        raise ValueError('First preparation already exists; never restage it')
    for name in ('static', 'route', 'display_initialization'):
        folder = context.output / name
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('An actor/display/physical attempt already exists; no retry: ' + name)


class BoundaryChanged(ValueError):
    def __init__(self, message, transaction):
        super().__init__(message)
        self.boundary_transaction = transaction


def continuation_definition(protocol_path, query_path):
    """Offline validation of saved evidence, NOT current scene admission."""
    context = runtime.load_workflow(protocol_path)
    p, folder = context.protocol, context.output
    query_path = Path(query_path).resolve()
    if query_path != folder / 'interrupted_Stop_review/live_boundary.json':
        raise ValueError('Exact own actual GET receipt required')
    files = {name: folder / name for name in ('workflow_dispatch_reserved.json', 'workflow_reserved.json',
        'actual_prior_state_before_Stop.json', 'actual_Stop_boundary.json', 'workflow_error.json')}
    dispatch, reserve, archive, stop, error = (_read(path) for path in files.values())
    query = _read(query_path)
    task.verify_pins(p['input_sha256'])
    task.verify_pins(query['input_sha256'])
    if (dispatch['protocol'] != str(context.protocol_path) or dispatch['sha256'] != task.sha(context.protocol_path)
            or dispatch['explicit_single_workflow_consent'] is not True
            or reserve['protocol_sha256'] != task.sha(context.protocol_path)
            or reserve['cohort'] != p['cohort'] or reserve['method'] != p['method']
            or reserve['no_automatic_retry'] is not True
            or archive['actual_state'] != p['required_prior_tensor']
            or archive['timeline'] != p['required_prior_timeline']
            or archive['protected_environment'] != dict(row_count=4997, sha256=p['protected_environment_sha256'])
            or archive['prior_result'] != p['required_prior_result']
            or archive['prior_result_sha256'] != task.sha(p['required_prior_result'])
            or stop['Stop_called_once'] is not True or stop['actual_timeline'] != STOPPED
            or error['phase'] != 'archive_ONE_Stop_new_source_and_display'
            or error['error'] != 'RuntimeError: Stage/layers/nonplaying state changed during fresh preflight'
            or error['no_automatic_retry_or_state_restore'] is not True):
        raise ValueError('Original archive, single consumed Stop and exact interruption required')
    if p.get('initialize_new_actor_visuals_before_static') is not True:
        raise ValueError('Original pre-static display initialization must remain enabled')
    reproduced = {name: shared.review_fresh_Stop_motion_fields(archive[name+'_text'], stop[name+'_text'],
        p['retired_robot_root']) for name in ('Root', 'Session')}
    if reproduced != stop['exact_motion_field_differences']:
        raise ValueError('Original recorded Stop field review does not reproduce')
    if (query['status'] != 'INTERRUPTED_H5_STOP_CURRENT_FIELD_DIFF_GET_ONLY_NO_ADMISSION'
            or query['actual_timeline'] != STOPPED or query['original_recorded_Stop_timeline'] != STOPPED
            or query['Root_Session_stage_timeline_unchanged_by_this_query'] is not True
            or query['actor_identity_existence'] != dict(retired_robot_root=True, new_robot_root=False, marker=False)
            or query['actual_motion_authorized'] is not False or query['new_PhysX_trial_count'] != 0):
        raise ValueError('Stable own STOPPED actor-absent GET required; not a physical result')
    before = {name: stop[name+'_text'] for name in ('Root', 'Session')}
    after = {name: query['actual_current_'+name+'_text'] for name in ('Root', 'Session')}
    deltas = layer_review(before, after)
    if deltas != query['exact_current_vs_recorded_Stop_field_differences']:
        raise ValueError('Actual GET native field evidence does not reproduce')
    _no_attempt(context)
    pins = task.merge_pins(p['input_sha256'], {str(path): task.sha(path)
        for path in (*files.values(), query_path, context.protocol_path,
                     Path(runtime.__file__).resolve(), Path(__file__).resolve())})
    result = dict(version=VERSION, original_protocol=str(context.protocol_path), actual_GET=str(query_path),
        cohort=p['cohort'], method=p['method'], reviewed_display_differences=deltas,
        maximum_additional_Stop_calls=0, maximum_new_static_trials=1, maximum_new_route_trials=1,
        first_actor_staging_only=True, automatic_retry=False, actual_motion_authorized=False,
        use_for_final_statistics=False, historical_failed_transaction_cause_NOT_claimed=True,
        input_sha256=pins)
    result['definition_sha256'] = shared.digest(result)
    return result


def export_continuation(protocol_path, query_path):
    result = continuation_definition(protocol_path, query_path)
    target = Path(protocol_path).parent / 'completed_Stop_continuation/contract.json'
    task.evidence.write_exclusive(target, result)
    return target


class BoundaryMonitor:
    """Read-only preparation guard. Failed exact transactions remain available."""

    def __init__(self, context, sdk, *, expected_state=None):
        self.context, self.sdk = context, sdk
        self.usd = sdk.omni.usd.get_context()
        self.stage = self.usd.get_stage()
        self.timeline = sdk.omni.timeline.get_timeline_interface()
        self.expected_state = dict(STOPPED if expected_state is None else expected_state)
        if (set(self.expected_state) != set(STOPPED) or self.expected_state['playing'] is not False
                or type(self.expected_state['stopped']) is not bool
                or not _number(self.expected_state['current_time_s'])
                or self.expected_state['current_time_s'] < 0
                or self.stage is None or task.evidence.timeline_state(self.timeline) != self.expected_state):
            raise ValueError('Existing exact nonplaying boundary required; no commands issued')
        self.initial = self.texts()
        self.last_transaction = None
        self._last_texts = None
        self._last_review = None

    def texts(self):
        result = dict(Root=self.stage.GetRootLayer().ExportToString(),
                      Session=self.stage.GetSessionLayer().ExportToString())
        if sum(len(v.encode()) for v in result.values()) > 2 * 1024 * 1024:
            raise ValueError('Preparation layer-text budget exceeded')
        return result

    def check(self):
        current = self.texts()
        state = task.evidence.timeline_state(self.timeline)
        same_stage = self.usd.get_stage() == self.stage
        # Unchanged layer bytes require no repeated Sdf parsing. The full
        # native comparison is performed on EVERY newly observed text pair.
        if same_stage and state == self.expected_state and current == self._last_texts:
            return self._last_review
        differences = {k: shared.layer_field_differences(self.initial[k], current[k])
                       if self.initial[k] != current[k] else [] for k in ('Root', 'Session')}
        transaction = dict(stage_same=same_stage, actual_timeline=state,
                           exact_native_field_differences=differences)
        self.last_transaction = transaction
        if not transaction['stage_same'] or state != self.expected_state:
            raise BoundaryChanged('Stage/nonplaying boundary changed; exact transaction retained', transaction)
        try:
            result = layer_review(self.initial, current) if any(differences.values()) else differences
        except ValueError as error:
            raise BoundaryChanged(str(error), transaction) from error
        self._last_texts, self._last_review = current, result
        return result


async def checked_environment(context, monitor, phase):
    """All original 4997 rows; camera deltas logged separately, never physics."""
    prep = context.preparation
    began = heartbeat = time.monotonic()
    rows = {}
    monitor.check()
    for path, row in prep.compact.stage_rows(monitor.stage, prep.SELECTED, prep.old.original):
        if path in rows:
            raise ValueError('Duplicate protected environment row')
        rows[path] = row
        if time.monotonic() - began > context.protocol['compact_readback_wall_budget_s']:
            raise TimeoutError('Preparation environment budget exceeded; no retry')
        if len(rows) % 16 == 0:
            monitor.check()
            await monitor.sdk.omni.kit.app.get_app().next_update_async()
            monitor.check()
        if time.monotonic() - heartbeat >= 15.:
            print('[H5 PREPARATION] '+phase, len(rows), '/4997; nonplaying, no motion', flush=True)
            heartbeat = time.monotonic()
    monitor.check()
    manifest = prep.load_manifest(verify_inputs=False)
    fingerprint = prep.compact.compare_rows(manifest['compact_environment_rows'], rows, prep.SELECTED, after_flags=True)
    if len(rows) != 4997 or fingerprint != context.protocol['protected_environment_sha256']:
        raise ValueError('Full original environment differs beyond the four approved flags')
    return dict(row_count=len(rows), sha256=fingerprint)


def checked_display_initializer(context, sdk, helpers):
    """Bind the nested original reader too; do not mutate frozen globals.

    Display registration itself still requires exact layer restoration. Unlike
    the earlier preparation review, its internal snapshot must therefore be
    byte-exact: even a known camera change is reported, not silently accepted
    only to fail the legacy final restoration check. This is instrumentation,
    NOT approval of an unreviewed field or recovery of an existing actor.
    """
    class ExactDisplayBoundary(BoundaryMonitor):
        def check(self):
            result = super().check()
            if any(result.values()):
                raise BoundaryChanged('Display snapshot changed; exact transaction retained',
                                      self.last_transaction)
            return result

    monitor = ExactDisplayBoundary(context, sdk)

    async def environment(own_context, stage, timeline, phase):
        # Native SDK getters may return different Python wrappers for the same
        # underlying stage/interface. Do not mistake Python object identity
        # for a native scene change. The original frozen display callable owns
        # the timeline getter; both readbacks must independently be STOPPED.
        state = task.evidence.timeline_state(timeline)
        if (own_context is not context or stage != monitor.stage
                or state != STOPPED or phase != 'before_renderer_initialization'):
            raise BoundaryChanged('Unexpected nested display-readback identity or phase',
                dict(actual_phase=phase, context_same=own_context is context,
                     stage_same=stage == monitor.stage, supplied_timeline=state))
        return await checked_environment(context, monitor, phase)

    # The original display bytecode, source/physical checks and visual-only
    # restoration remain unchanged. Only its private reader binding changes.
    return shared._bound(helpers.initialize_staged_display,
                         fresh_environment_readback=environment)


def staged_abort_frames(future):
    """Read an existing terminal task only; never search GC or recreate SDKs."""
    if future is None or not future.done() or future.cancelled():
        raise ValueError('Original terminal H5 staging task required; no SDK/action')
    error = future.exception()
    if type(error) is not RuntimeError or str(error) != 'Stage/layers/nonplaying state changed during fresh preflight':
        raise ValueError('Exact original nested renderer-preflight interruption required')
    frames, trace = {}, error.__traceback__
    names = {'finish_first_staging', 'initialize_fresh_staged_display', 'fresh_environment_readback'}
    while trace is not None:
        name = trace.tb_frame.f_code.co_name
        if name in names:
            if name in frames:raise ValueError('Ambiguous original staging traceback')
            frames[name] = trace.tb_frame.f_locals
        trace = trace.tb_next
    if set(frames) != names:raise ValueError('Original staging/display/preflight frames missing')
    return frames


async def run_first_staged_display_once(future, *, authorized=False):
    """Complete ONLY a never-attempted display registration of the SAME actor.

    No constructor, pose, camera, timeline, static or route operation. The
    old failed workflow stays terminal. A reviewed camera-only preparation
    boundary may settle, but the actual renderer transaction stays exact.
    """
    if authorized is not True:raise ValueError('Explicit same-actor display-only consent required')
    frames = staged_abort_frames(future)
    staging, display, preflight = (frames[name] for name in
        ('finish_first_staging', 'initialize_fresh_staged_display', 'fresh_environment_readback'))
    context, sdk = staging['context'], staging['sdk']
    if (context.cohort.ROBOT != '/World/patient_transport_h5_cohort_01_P_development_r01'
            or display['context'] is not context or preflight['context'] is not context):
        raise ValueError('Exact original H5 actor/context required; no replacement')
    folder = context.output/'completed_Stop_continuation'
    reserve, output = folder/'single_staged_display_completion_reserved.json', folder/'staged_display_completion.json'
    if reserve.exists() or output.exists():raise ValueError('Display completion already dispatched; no retry')
    _no_attempt(context)
    query_path = folder/'staged_abort_live_get.json'
    query = _read(query_path)
    if (query['status'] != 'H5_STAGED_RENDERER_PREFLIGHT_ABORT_GET_ONLY_NO_ADMISSION'
            or query['actor'] != context.cohort.ROBOT or not query['actor_exists'] or not query['actor_active']
            or query['source_physical_configuration_unchanged'] is not True
            or query['current_physical_readback_error'] is not None
            or query['renderer_initialization_reservation_exists'] is not False
            or query['static_or_route_folder_exists'] is not False
            or query['actual_timeline'] != STOPPED):
        raise ValueError('Own actual source-unchanged STOPPED GET before any display/static/route required')
    layer_review(query['expected_preflight_layer_text'], query['current_layer_text'])
    task.verify_pins(context.protocol['input_sha256'])
    monitor = BoundaryMonitor(context, sdk)
    if monitor.stage != preflight['stage'] or layer_review(query['current_layer_text'], monitor.initial)['Root']:
        raise ValueError('Original GET stage/Root changed')
    if display['physical']() != display['before']:
        raise ValueError('Actual source physical settings changed; no display recovery')
    task.evidence.write_exclusive(reserve, dict(same_existing_actor=context.cohort.ROBOT,
        actual_GET_sha256=task.sha(query_path), no_Stop_Play_reset_restaging_static_route=True,
        no_automatic_retry=True, actual_motion_authorized=False))
    try:
        # Drain existing UI work under the narrow field classifier BEFORE the
        # original renderer captures its exact restoration baseline. Never
        # restore/adopt camera Z/orientation, physics or an unknown field.
        previous, stable, began = monitor.texts(), 0, time.monotonic()
        for _ in range(60):
            monitor.check()
            await sdk.omni.kit.app.get_app().next_update_async()
            monitor.check()
            current = monitor.texts()
            stable = stable+1 if current == previous else 0
            previous = current
            if stable >= 3:break
            if time.monotonic()-began > 3.:raise TimeoutError('Display preparation did not settle; no retry')
        else:raise TimeoutError('Display preparation update bound exceeded; no retry')
        protected = await checked_environment(context, monitor, 'same_actor_before_FIRST_display')
        if display['physical']() != display['before']:
            raise ValueError('Source physical settings changed during display preparation')
        helpers = task.preparation_functions(context)
        result = await checked_display_initializer(context, sdk, helpers)(context)
        if display['physical']() != display['before'] or task.evidence.timeline_state(monitor.timeline) != STOPPED:
            raise ValueError('Same source/STOPPED state not retained after display registration')
        report = dict(status='SAME_H5_ACTOR_FIRST_DISPLAY_REGISTERED_NOT_STATIC_OR_DRIVE_ADMISSION',
            actor=context.cohort.ROBOT, protected_environment=protected,
            reviewed_preparation_camera_differences=layer_review(monitor.initial, monitor.texts()),
            original_staging_error_preserved=True, source_physical_settings_retained=True,
            original_constructor_NOT_called=True, actual_timeline=STOPPED,
            render_registration=result, actual_visible_pixels_confirmed=False,
            new_PhysX_trial_count=0, actual_motion_authorized=False, use_for_final_statistics=False,
            input_sha256={str(query_path):task.sha(query_path),str(Path(__file__).resolve()):task.sha(__file__)})
        task.evidence.write_exclusive(output, report)
        print('[H5 DISPLAY] SAME actor display registration complete; NO Play/Stop/static/route.',flush=True)
        print('[H5 DISPLAY] Inspect pixels WITHOUT F. saved:', output,flush=True)
        return report['status']
    except BaseException as error:
        task.evidence.write_exclusive(folder/'staged_display_completion_error.json',dict(
            error=type(error).__name__+': '+str(error),
            exact_failed_boundary_transaction=getattr(error,'boundary_transaction',None),
            no_automatic_retry=True, no_static_or_route=True, actual_motion_authorized=False))
        raise


async def finish_first_staging(context, sdk, contract, monitor):
    """Only the never-executed suffix: no Stop, reset, prior-state restoration."""
    from pxr import Usd, UsdGeom, Sdf
    p, stage, timeline = context.protocol, monitor.stage, monitor.timeline
    helpers = task.preparation_functions(context)
    before = await checked_environment(context, monitor, 'before_first_staging')
    retiring = shared._bound(shared.retiring_actor_configuration, stationary=context.stationary)
    if retiring(context, stage) != p['retiring_actor_fixed_configuration']:
        raise ValueError('All retiring-actor fixed source settings must remain exact')
    if any(stage.GetPrimAtPath(p[k]).IsValid() for k in ('new_robot_root', 'marker')):
        raise ValueError('New actor or marker already exists; no replacement')
    _no_attempt(context)
    monitor.check()
    if p.get('initialize_new_actor_visuals_before_static') is not True:
        raise ValueError('Original pre-static display initialization must remain enabled')
    shared.write_runtime_evidence(context.output/'completed_Stop_continuation/before_first_staging.json',
        dict(protected_environment=before, current_timeline=STOPPED,
            reviewed_current_display_differences=layer_review(
                {k:_read(context.output/'actual_Stop_boundary.json')[k+'_text'] for k in ('Root','Session')}, monitor.texts()),
            no_additional_Stop_or_Play=True, original_contract_sha256=contract['definition_sha256']))
    staging_before = monitor.texts()
    # These are the ORIGINAL source constructor/namespace/marker operations.
    # Display registration is completed before the static layer baseline.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        stage.GetPrimAtPath(p['retired_robot_root']).SetActive(False)
        made = context.preparation.old.constructor_for(context.cohort.ROBOT)(stage, p['start_xyyaw'])
        marker = UsdGeom.Xform.Define(stage, context.cohort.MARKER).GetPrim()
        for key, value in dict(robot_root=context.cohort.ROBOT, status='fresh_static_pending_NO_PLAY', cohort=p['cohort']).items():
            marker.SetCustomDataByKey(key, value)
        namespace = stage.GetPrimAtPath(context.cohort.ROBOT).CreateAttribute(
            'isaac:namespace', Sdf.ValueTypeNames.String, custom=False, variability=Sdf.VariabilityVarying)
        namespace.SetDisplayName('Namespace')
        namespace.SetDocumentation('Namespace of the prim in Isaac Sim')
        # Bind the INSIDE reader as well as the outside preparation guard.
        # The original exact physical/layer restoration checks still apply.
        await checked_display_initializer(context, sdk, helpers)(context)
    # No camera-only edit is made here. Original strict static/route boundaries
    # remain in force once the own first preparation baseline is recorded.
    staging_scope = staging_scope_review(context, staging_before, monitor.texts())
    return await record_source_preparation(context, sdk, made, contract['definition_sha256'], staging_scope)


async def record_source_preparation(context, sdk, made, continuation_id, staging_scope):
    """Record an already-created exact source actor. No constructor/scene edit."""
    p = context.protocol
    if context.preparation.LIVE.exists():raise ValueError('Preparation already recorded; no overwrite')
    staged = BoundaryMonitor(context, sdk)
    stage, timeline = staged.stage, staged.timeline
    protected = await checked_environment(context, staged, 'first_source_staged')
    helpers = task.preparation_functions(context)
    reference = helpers.source_for_live_actor(context, stage)
    if reference['source_constructor_readback'] != made:
        raise ValueError('Already-staged constructor state differs from exact nominal source')
    archive, stop = (context.output/name for name in ('actual_prior_state_before_Stop.json', 'actual_Stop_boundary.json'))
    prep = dict(cohort=p['cohort'], new_robot_root=context.cohort.ROBOT, retired_robot_root=p['retired_robot_root'],
        start_xyyaw=p['start_xyyaw'], simulation_mass_kg=265., live_new_robot_readback=made,
        source_constructor_readback=made, protected_environment_sha256=protected['sha256'], source_reference=reference,
        expected_nominal_motor_baseline=p['expected_nominal_motor_baseline'], timeline=task.evidence.timeline_state(timeline),
        old_physical_continuity_NOT_restored=True, archive=str(archive), archive_sha256=task.sha(archive),
        Stop_boundary=str(stop), Stop_boundary_sha256=task.sha(stop), Play_or_static_or_drive_executed=False,
        completed_Stop_continuation=continuation_id, additional_Stop_calls=0,
        exact_first_staging_scope_review=staging_scope)
    prep['critical_environment'] = context.stationary.critical_environment(stage, prep)
    prep.update(Root_sha256=hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest(),
                Session_sha256=hashlib.sha256(stage.GetSessionLayer().ExportToString().encode()).hexdigest())
    shared.write_runtime_evidence(context.preparation.LIVE, prep)
    return prep


def displayed_trials_definition(protocol_path):
    """Offline immutable evidence for remaining FIRST static/conditional route."""
    import zipfile
    context = runtime.load_workflow(protocol_path)
    folder = context.output/'completed_Stop_continuation'
    display_path, renderer_path = folder/'staged_display_completion.json', context.output/'display_initialization/renderer_initialization.json'
    display, renderer = _read(display_path), _read(renderer_path)
    if (display['status'] != 'SAME_H5_ACTOR_FIRST_DISPLAY_REGISTERED_NOT_STATIC_OR_DRIVE_ADMISSION'
            or display['actor'] != context.cohort.ROBOT or display['actual_timeline'] != STOPPED
            or display['source_physical_settings_retained'] is not True
            or display['original_constructor_NOT_called'] is not True
            or display['new_PhysX_trial_count'] != 0 or display['actual_motion_authorized'] is not False
            or display['render_registration'] != renderer
            or renderer['registered_visual_root_count'] != 13 or renderer['visible_USD_shape_count'] != 22
            or renderer['original_Root_and_Session_text_exactly_restored'] is not True
            or renderer['timeline_before_and_after'] != STOPPED
            or display['protected_environment'] != dict(row_count=4997,sha256=context.protocol['protected_environment_sha256'])):
        raise ValueError('Own completed same-source FIRST display registration required')
    source_path = Path(__file__).resolve()
    archived = folder/'executed_display_completion_sources.zip'
    with zipfile.ZipFile(archived) as archive:
        payload = archive.read(str(source_path.relative_to(task.ROOT)))
    if hashlib.sha256(payload).hexdigest() != display['input_sha256'][str(source_path)]:
        raise ValueError('Original actually executed display source must remain archived, not repinned')
    for name,value in display['input_sha256'].items():
        if name != str(source_path) and task.sha(name) != value:raise ValueError('Actual display input changed: '+name)
    task.verify_pins(renderer['input_sha256'])
    task.verify_pins(context.protocol['input_sha256'])
    if context.preparation.LIVE.exists():raise ValueError('Preparation already recorded; no second completion')
    for name in ('static','route'):
        path=context.output/name
        if path.exists() and any(path.iterdir()):raise ValueError('A physical trial already began; no retry')
    files=(context.protocol_path,display_path,renderer_path,archived,source_path,
        folder/'staged_abort_live_get.json',context.output/'actual_prior_state_before_Stop.json',
        context.output/'actual_Stop_boundary.json',Path(runtime.__file__).resolve())
    pins=task.merge_pins(context.protocol['input_sha256'],{str(p):task.sha(p) for p in files})
    result=dict(version='SAME_DISPLAYED_H5_FIRST_STATIC_AND_CONDITIONAL_ROUTE_v1',
        original_protocol=str(context.protocol_path),actor=context.cohort.ROBOT,
        maximum_additional_Stop_calls=0,maximum_constructor_calls=0,maximum_display_refresh_calls=0,
        maximum_new_static_trials=1,maximum_new_route_trials=1,automatic_retry=False,
        original_static_route_control_queries_guards_and_resource_limits_unchanged=True,
        actual_motion_authorized=False,use_for_final_statistics=False,input_sha256=pins)
    result['definition_sha256']=shared.digest(result)
    return result


async def run_displayed_trials_once(contract_path, future, *, authorized=False):
    """Same existing source actor -> original static -> conditional original P."""
    if authorized is not True:raise ValueError('Explicit same-actor FIRST static/route consent required')
    contract_path=Path(contract_path).resolve();contract=_read(contract_path)
    frames=staged_abort_frames(future)
    staging,display=frames['finish_first_staging'],frames['initialize_fresh_staged_display']
    context,sdk=staging['context'],staging['sdk']
    folder=context.output/'displayed_trials_continuation'
    reserve=folder/'single_dispatch_reserved.json'
    if contract_path!=folder/'contract.json' or reserve.exists():raise ValueError('Own unused displayed continuation required; no retry')
    expected=displayed_trials_definition(context.protocol_path)
    if contract!=expected:raise ValueError('Same displayed source/protocol/evidence changed')
    task.verify_pins(contract['input_sha256'])
    task.evidence.write_exclusive(reserve,dict(explicit_same_actor_static_route_consent=True,
        contract_sha256=task.sha(contract_path),additional_Stop_calls=0,constructor_calls=0,
        display_refresh_calls=0,automatic_retry=False))
    phase='complete_ALREADY_staged_source_record'
    try:
        monitor=BoundaryMonitor(context,sdk)
        query=_read(context.output/'completed_Stop_continuation/staged_abort_live_get.json')
        layer_review(query['current_layer_text'],monitor.initial)
        if display['physical']()!=display['before']:raise ValueError('Displayed source physical settings changed')
        scope=staging_scope_review(context,staging['staging_before'],monitor.initial)
        await record_source_preparation(context,sdk,staging['made'],contract['definition_sha256'],scope)
        phase='ONE_FIRST_original_static'
        static=await runtime.run_static_once(context,authorized=True)
        result=dict(status='STATIC_NOT_PASS_NO_ROUTE',static_verdict=static['verdict'],route_verdict=None,
            additional_Stop_calls=0,constructor_calls=0,display_refresh_calls=0,automatic_retry=False)
        if static['verdict']=='PASS':
            context.operations.replay_static(context)
            phase='ONE_FIRST_original_P_route'
            route=await runtime.run_route_once(context,authorized=True)
            result.update(status='FIRST_H5_STATIC_AND_CONDITIONAL_P_WORKFLOW_FINISHED',route_verdict=route['verdict'])
        task.evidence.write_exclusive(folder/'completion.json',result)
        print('[H5 FIRST TRIALS] Completed:',result,flush=True)
        return result
    except BaseException as error:
        task.evidence.write_exclusive(folder/'error.json',dict(phase=phase,error=type(error).__name__+': '+str(error),
            exact_failed_boundary_transaction=getattr(error,'boundary_transaction',None),
            original_archive_Stop_errors_and_display_preserved=True,automatic_retry=False))
        raise


async def run_remaining_once(contract_path, *, authorized=False):
    """Explicitly admitted first staging/static/conditional route, NEVER retry."""
    if authorized is not True:
        raise ValueError('Explicit consent for remaining first staging/static/conditional route required')
    contract_path = Path(contract_path).resolve()
    contract = _read(contract_path)
    context = runtime.load_workflow(contract['original_protocol'])
    if contract_path != context.output/'completed_Stop_continuation/contract.json':
        raise ValueError('Exact own immutable continuation contract required')
    reserve = contract_path.parent/'single_remaining_dispatch_reserved.json'
    if reserve.exists():
        raise ValueError('Remaining workflow already dispatched; no automatic retry or second call')
    for name in ACTIVE_TASKS:
        previous = getattr(builtins, name, None)
        if previous is not None and not previous.done():
            # The dispatcher registers its OWN task only after scheduling; do
            # not misclassify that owner as a competing historical operation.
            import asyncio
            if previous is not asyncio.current_task():
                raise ValueError('Another research task is active; no competing workflow')
    expected = continuation_definition(context.protocol_path, contract['actual_GET'])
    if contract != expected:
        raise ValueError('Reviewed actual receipts/source/contract changed; no recovery')
    task.verify_pins(contract['input_sha256'])
    task.evidence.write_exclusive(reserve, dict(explicit_remaining_workflow_consent=True,
        contract_sha256=task.sha(contract_path), maximum_additional_Stop_calls=0, automatic_retry=False))
    phase, monitor = 'current_STOPPED_preflight', None
    try:
        sdk = runtime.live_sdk()
        monitor = BoundaryMonitor(context, sdk)
        stop = _read(context.output/'actual_Stop_boundary.json')
        layer_review({k:stop[k+'_text'] for k in ('Root','Session')}, monitor.initial)
        phase = 'never_executed_first_staging'
        await finish_first_staging(context, sdk, contract, monitor)
        phase = 'ONE_original_static'
        static = await runtime.run_static_once(context, authorized=True)
        if static['verdict'] != 'PASS':
            return dict(status='STATIC_NOT_PASS_NO_ROUTE', static_verdict=static['verdict'])
        context.operations.replay_static(context)
        phase = 'ONE_original_fixed_route'
        route = await runtime.run_route_once(context, authorized=True)
        result = dict(status='SINGLE_REMAINING_H5_DEVELOPMENT_WORKFLOW_FINISHED',
            static_verdict=static['verdict'], route_verdict=route['verdict'], additional_Stop_calls=0)
        task.evidence.write_exclusive(contract_path.parent/'completion.json', result)
        print('[H5 REMAINING] Completed; static/route:', static['verdict'], route['verdict'], '; no retry.', flush=True)
        return result
    except BaseException as error:
        diagnostic = dict(phase=phase, error=type(error).__name__+': '+str(error),
            original_archive_Stop_error_preserved=True, additional_Stop_calls=0, automatic_retry=False,
            actual_failed_boundary_transaction=getattr(error,'boundary_transaction',None))
        task.evidence.write_exclusive(contract_path.parent/'remaining_error.json', diagnostic)
        raise


MATCHED_VERSION = 'H5_MATCHED_FRESH_METHOD_FROM_COMPLETED_TASK_BOUNDARY_v1'


def load_task_workflow(path, *, ancestry=()):
    """Load original or matched task definitions; never open the hospital."""
    path = Path(path).resolve()
    if str(path) in ancestry or len(ancestry) >= 64:
        raise ValueError('Cyclic/overlong task evidence lineage; no SDK access')
    protocol = _read(path)
    if protocol.get('matched_workflow_version') != MATCHED_VERSION:
        return runtime.load_workflow(path)
    expected = matched_workflow_definition(protocol['matched_template_protocol'],
        protocol['required_prior_result'], cohort=protocol['cohort'], method=protocol['method'],
        ancestry=(*ancestry, str(path)))
    if (protocol != expected or path != Path(protocol['output_directory'])/'runtime_protocol.json'
            or Path(protocol['matched_source_archive']).read_bytes() != matched_source_archive()):
        raise ValueError('Matched task source/definition changed; no repair or waiver')
    task.verify_pins(protocol['input_sha256'])
    return runtime.static_context(protocol, path)


def completed_task_boundary(path, *, ancestry=()):
    """Full saved H5 route closeout, not a live-state or success assumption.

    FAIL is retained as FAIL and can supply a later archive boundary only if
    Pause, gains, environment and complete tensor integrity all passed.
    """
    import csv
    import numpy as np
    path = Path(path).resolve()
    raw = _read(path)
    context = load_task_workflow(raw['path_protocol'], ancestry=ancestry)
    if (path != context.output/'route'/(context.route_run_id+'.json')
            or raw['method'] != context.protocol['method'] or raw['scenario'] != task.SCENARIO
            or raw['execution_role'] != 'fresh_fixed_start_whole_H5_development_pilot'
            or raw['trial_kind'] != 'H5_frozen_path_PhysX_development_ground_truth_feedback_NOT_sensor_navigation'
            or raw['verdict'] not in ('PASS','FAIL') or raw['path_execution_started'] is not True
            or raw['automatic_route_Play_requested'] is not True or raw['use_for_final_statistics'] is not False):
        raise ValueError('Own actually executed H5 route required, not reservation/setup/other task')
    required = ('actual_route_Pause_confirmed','original_motor_gains_restored',
                'physical_environment_unchanged','all_frozen_input_evidence_unchanged')
    state, pause = raw['actual_final_timeline'], raw['pause_confirmation']
    if (any(raw['checks'].get(key) is not True for key in required)
            or state != pause['after'] or state['playing'] is not False or state['stopped'] is not False
            or not _number(state['current_time_s']) or state['current_time_s'] <= 0
            or raw['original_motor_gains_restored'] is not True
            or raw['trace_sha256'] != task.sha(path.with_suffix('.csv'))):
        raise ValueError('Complete actual PAUSED/gain-restored route boundary required')
    task.verify_pins(raw['input_sha256'])
    prep, static, replay = context.operations.replay_static(context)
    if (raw['static_gate_reference'] != str(context.stationary.RESULT)
            or raw['static_gate_sha256'] != task.sha(context.stationary.RESULT)
            or raw['physics_preparation_snapshot'] != prep):
        raise ValueError('Route must retain its own actual static, not another method PASS')
    for key in ('compact_environment_before_route_Play','compact_environment_after_route_Pause'):
        if (raw[key]['all_original_row_count'] != 4997
                or raw[key]['protected_environment_sha256'] != context.protocol['protected_environment_sha256']):
            raise ValueError('Both whole protected-environment records required')
    actual, baseline = raw['actual_post_route_tensor_snapshot'], static['runtime_motor_baseline']
    for key, shape in (('transform',(7,)),('velocity',(6,)),('positions',(12,)),
            ('joint_velocities_rad_s',(12,)),('joint_position_targets_rad',(12,)),
            ('joint_velocity_targets_rad_s',(12,)),('all_thirteen_link_world_transforms_xyzw',(13,7)),
            ('all_thirteen_link_velocities_at_CoM',(13,6))):
        values = np.asarray(actual[key], float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError('Finite complete saved route tensor required: '+key)
    if (baseline != raw['observed_original_motor_settings'] or actual['names'] != baseline['joint_names']
            or actual['gains'] != {k:baseline[k] for k in ('kp','kd','max_efforts')}
            or len(actual['link_names']) != 13 or len(set(actual['link_names'])) != 13
            or actual['link_names'] != static['actual_post_static_tensor_snapshot']['link_names']
            or actual['positions'] != raw['final_actual_twelve_joint_positions_rad']
            or actual['joint_velocities_rad_s'] != raw['final_actual_twelve_joint_velocities_rad_s']):
        raise ValueError('Saved route joint/link/gain correspondence failed')
    if (not np.allclose(np.linalg.norm(np.asarray(actual['all_thirteen_link_world_transforms_xyzw'])[:,3:],axis=1),
                       1.,rtol=0,atol=1e-4)
            or not math.isclose(float(np.linalg.norm(actual['transform'][3:])),1.,rel_tol=0,abs_tol=1e-4)):
        raise ValueError('Normalized measured root and link quaternions required')
    with path.with_suffix('.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row['time_s']) for row in rows])
    if (len(rows) != raw['sample_count'] or len(rows) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not any(row['phase']=='path_tracking' for row in rows)):
        raise ValueError('Complete actual monotonic H5 trace required')
    shared.initial_motion_trace_replay(raw, rows)  # Original fixed-tail evaluator; FAIL never upgraded.
    pins = task.merge_pins(raw['input_sha256'], {str(p):task.sha(p) for p in
        (path, path.with_suffix('.csv'), context.protocol_path, context.stationary.RESULT, context.stationary.TRACE)})
    return task.evidence.json_value(dict(prior_result=str(path), prior_verdict=raw['verdict'],
        retired_robot_root=context.cohort.ROBOT, required_prior_tensor=actual, required_prior_timeline=state,
        retiring_actor_fixed_configuration=static['runtime_anonymous_source_reference']['fixed_configuration_rows'],
        expected_nominal_motor_baseline=baseline,
        protected_environment_sha256=context.protocol['protected_environment_sha256'],
        prior_result_NOT_relabelled=True, live_state_NOT_accessed_or_restored=True,
        no_new_experiment_or_Stop_authority=True, actual_motion_authorized=False, input_sha256=pins))


def matched_source_archive():
    import io
    import zipfile
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        path = Path(__file__).resolve()
        member = zipfile.ZipInfo(str(path.relative_to(task.ROOT)), date_time=(1980,1,1,0,0,0))
        member.external_attr = 0o100644 << 16
        archive.writestr(member, path.read_bytes())
    return output.getvalue()


def matched_workflow_definition(template_path, prior_result, *, cohort, method, ancestry=()):
    """Same frozen FOUR-method H5 bundle, new explicit method/cohort only.

    No planning/MODEL rerun, goal/limit tuning, live state read or motion.
    The original preparation/runtime files stay byte-exact. The shared
    runtime receives its original code objects with new task identities.
    """
    import copy
    template_path = Path(template_path).resolve()
    original = runtime.load_workflow(template_path)
    task.verify_pins(original.protocol['input_sha256'])
    prior = completed_task_boundary(prior_result, ancestry=ancestry)
    if (prior['protected_environment_sha256'] != original.protocol['protected_environment_sha256']
            or prior['expected_nominal_motor_baseline'] != original.protocol['expected_nominal_motor_baseline']):
        raise ValueError('Matched environment and all-twelve motor settings must remain identical')
    p = copy.deepcopy(original.protocol)
    target = task.GATE/'hospital_h5_experiments'/cohort
    old_robot = p['new_robot_root']
    p.update(cohort=cohort, method=method, output_directory=str(target),
        new_robot_root='/World/patient_transport_h5_'+cohort, marker='/World/IJCAS_HospitalH5_'+cohort,
        **{k:prior[k] for k in ('retired_robot_root','required_prior_tensor','required_prior_timeline',
            'retiring_actor_fixed_configuration','expected_nominal_motor_baseline','protected_environment_sha256')},
        required_prior_result=prior['prior_result'], saved_prior_boundary=prior,
        matched_workflow_version=MATCHED_VERSION, matched_template_protocol=str(template_path),
        matched_source_archive=str(target/'matched_workflow_sources.zip'),
        matched_source_archive_sha256=hashlib.sha256(matched_source_archive()).hexdigest(),
        preparation_definition=str(target/'runtime_protocol.json'),
        anonymous_source_review=str(target/'anonymous_source_readback.json'),
        camera_preparation=None,
        sequence=['verify_actual_prior_PAUSED_state','archive_and_ONE_confirmed_Stop',
            'same_source_new_actor_and_FIRST_visual_registration','ONE_original_static',
            'if_own_raw_static_PASS_ONE_selected_method_route','ONE_5s_hold_confirmed_Pause_gain_restore'],
        status='OFFLINE_MATCHED_DEVELOPMENT_DEFINITION_NOT_LIVE_OR_FINAL_PROTOCOL_ADMISSION')
    # This validates the cohort syntax, exact output/actor identity and method
    # BEFORE accepting arbitrary paths. No files or stage are created.
    task.task_context(p)
    p['source_actor_node_paths'] = [p['new_robot_root']+node[len(old_robot):]
        if node==old_robot or node.startswith(old_robot+'/') else node for node in p['source_actor_node_paths']]
    if any(node != p['new_robot_root'] and not node.startswith(p['new_robot_root']+'/') for node in p['source_actor_node_paths']):
        raise ValueError('All145 source node identities must belong to the same new actor')
    plan, metadata, _ = runtime.reporting_plan(p)
    if (plan['start_world_xyyaw'] != original.protocol['start_xyyaw']
            or plan['goal_world_xyyaw'] != original.protocol['goal_xyyaw']
            or plan['planning_occupancy_digest'] != runtime.reporting_plan(original.protocol)[0]['planning_occupancy_digest']):
        raise ValueError('Matched methods must share unchanged task endpoints and complete map')
    model = _read(p['nominal_model_screen'])
    cases = [row for row in model['cases'] if row['method']==method]
    if len(cases)!=2 or {row['integration_dt_s'] for row in cases}!={.01,.02}:
        raise ValueError('Both original MODEL observations, including HOLDs, must remain visible')
    p.update(selected_plan=metadata['original_native_plan_path'], route_reporting_metadata=metadata,
             selected_method_MODEL_statuses=[row['model_status'] for row in cases])
    p['input_sha256'] = task.merge_pins(original.protocol['input_sha256'], prior['input_sha256'],
        {str(template_path):task.sha(template_path), str(Path(__file__).resolve()):task.sha(__file__),
         p['matched_source_archive']:p['matched_source_archive_sha256']})
    p.pop('definition_sha256', None)
    p['definition_sha256'] = shared.digest(p)
    return task.evidence.json_value(p)


def export_matched_workflow(template_path, prior_result, *, cohort, method):
    """Append-only offline definition. This exports NO executable command."""
    protocol = matched_workflow_definition(template_path, prior_result, cohort=cohort, method=method)
    target = Path(protocol['output_directory'])/'runtime_protocol.json'
    archive = Path(protocol['matched_source_archive'])
    if target.exists():
        if _read(target) != protocol or archive.read_bytes() != matched_source_archive():
            raise ValueError('Existing matched definition differs; no overwrite')
        return target
    if target.parent.exists():raise ValueError('Partial/existing matched cohort preserved; no overwrite')
    target.parent.mkdir(parents=True)
    with archive.open('xb') as stream:
        stream.write(matched_source_archive())
        stream.flush()
        import os
        os.fsync(stream.fileno())
    task.evidence.write_exclusive(target, protocol)
    return target


def matched_stage_operation(context, sdk):
    """Original archive/ONE Stop/source constructor with reviewed UI guards.

    Exact renderer restoration and static/route boundaries remain unchanged.
    No viewport focus/renderer preference or existing actor pose is edited.
    """
    helpers = task.preparation_functions(context)

    async def environment(own, stage, timeline, phase):
        monitor = BoundaryMonitor(own, sdk, expected_state=task.evidence.timeline_state(timeline))
        if own is not context or stage != monitor.stage:
            raise ValueError('Own stage/context required during matched source preparation')
        return await checked_environment(own, monitor, phase)

    async def display(own):
        if own is not context:raise ValueError('Own new actor required before FIRST display registration')
        monitor = BoundaryMonitor(own, sdk)
        previous, stable, began = monitor.texts(), 0, time.monotonic()
        for _ in range(60):
            monitor.check()
            await sdk.omni.kit.app.get_app().next_update_async()
            monitor.check()
            current = monitor.texts()
            stable = stable+1 if current==previous else 0
            previous = current
            if stable>=3:break
            if time.monotonic()-began>3.:raise TimeoutError('New-actor display preparation did not settle')
        else:raise TimeoutError('New-actor display preparation update bound exceeded')
        return await checked_display_initializer(own, sdk, helpers)(own)

    return shared._bound(helpers.stage_after_archive, fresh_environment_readback=environment,
                         initialize_fresh_staged_display=display, print=runtime.task_log)


async def run_matched_workflow_once(path, *, authorized=False):
    """One explicitly approved NEW cohort, never replay a completed attempt."""
    if authorized is not True:
        raise ValueError('Separate explicit ONE matched-method archive/Stop/static/route consent required')
    context = load_task_workflow(path)
    if context.protocol.get('matched_workflow_version') != MATCHED_VERSION:
        raise ValueError('Own new matched definition required, not the old dispatched workflow')
    reserve = context.output/'workflow_dispatch_reserved.json'
    if reserve.exists():raise ValueError('Matched workflow already dispatched; no retry')
    for name in ACTIVE_TASKS:
        previous = getattr(builtins, name, None)
        if previous is not None and not previous.done():
            import asyncio
            if previous is not asyncio.current_task():raise ValueError('Another research task is active')
    _no_attempt(context)
    task.evidence.write_exclusive(reserve, dict(protocol=str(context.protocol_path),
        protocol_sha256=task.sha(context.protocol_path), method=context.protocol['method'],
        explicit_single_matched_workflow_consent=True, maximum_Stop_calls=1,
        maximum_static_trials=1, maximum_route_trials=1, automatic_retry=False))
    phase = 'actual_prior_PAUSED_preflight'
    try:
        sdk = runtime.live_sdk()
        await runtime.verify_workflow_inputs(context, sdk)
        phase = 'archive_ONE_Stop_source_and_FIRST_display'
        await matched_stage_operation(context, sdk)(context)
        usd = sdk.omni.usd.get_context()
        stop = _read(context.output/'actual_Stop_boundary.json')
        stage = usd.get_stage()
        staging_scope_review(context, {k:stop[k+'_text'] for k in ('Root','Session')},
            dict(Root=stage.GetRootLayer().ExportToString(),Session=stage.GetSessionLayer().ExportToString()))
        phase = 'ONE_own_original_static'
        static = await runtime.run_static_once(context, authorized=True)
        result = dict(status='STATIC_NOT_PASS_NO_ROUTE', method=context.protocol['method'],
            static_verdict=static['verdict'], route_verdict=None, automatic_retry=False)
        if static['verdict']=='PASS':
            context.operations.replay_static(context)
            phase = 'ONE_own_selected_method_route'
            route = await runtime.run_route_once(context, authorized=True)
            result.update(status='ONE_MATCHED_DEVELOPMENT_WORKFLOW_FINISHED',route_verdict=route['verdict'])
        task.evidence.write_exclusive(context.output/'completion.json', result)
        print('[H5 MATCHED] Completed:', result, '; no retry or next method.', flush=True)
        return result
    except BaseException as error:
        task.evidence.write_exclusive(context.output/'workflow_error.json',dict(phase=phase,
            error=type(error).__name__+': '+str(error), automatic_retry=False,
            exact_failed_boundary_transaction=getattr(error,'boundary_transaction',None),
            prior_results_and_current_failed_attempt_preserved=True))
        raise
