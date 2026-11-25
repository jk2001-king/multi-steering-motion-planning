"""Reconstruct ONLY the recorded static hospital environment in a fresh app.

No timeline, actor, renderer, controller, or source-file write. First compose
and verify every protected row in an isolated USD stage; only then copy the
bounded static opinions into the new live Session. This is not admission for
static testing or driving and does not restore any historical dynamic state.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / 'studies/IJCAS/results/hospital_paper_gate'
RECORDED = GATE / 'hospital_h6_experiments/cohort_01_common_preparation_r01'
OUTPUT = GATE / 'controlled_comparison_20261120/environment_bootstrap_r01'
STATIC_ROOTS = ('/World/Hospital', '/World/PatientTransportLabGround',
                '/World/IJCAS_HospitalH4SupportR01', '/physicsScene', '/FlatGrid')


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_static(source, target):
    """Copy only named static subtrees, never World/actors/cameras/Render."""
    from pxr import Sdf
    copied = []
    for path in STATIC_ROOTS:
        if source.GetPrimAtPath(path) is None:
            continue
        parent = Sdf.Path(path).GetParentPath()
        if parent != Sdf.Path.absoluteRootPath:
            Sdf.CreatePrimInLayer(target, str(parent))
        if not Sdf.CopySpec(source, path, target, path):
            raise RuntimeError('Static opinion copy failed: ' + path)
        copied.append(path)
    return copied


def build_static_patch():
    from pxr import Sdf, UsdUtils
    import hospital_mission_geometry as geometry
    packet = read(RECORDED / 'preparation_protocol.json')
    recorded = read(RECORDED / 'actual_Stop_boundary.json')
    patch_path = Path(packet['common_environment_patch'])
    paths = [RECORDED / name for name in ('preparation_protocol.json',
        'actual_Stop_boundary.json', 'actual_common_environment_rows.json')]
    paths.extend((patch_path, Path(__file__)))
    for path in paths[:-1]:
        expected = packet['input_sha256'].get(str(path))
        if expected is not None and sha(path) != expected:
            raise ValueError('Recorded environment dependency changed: ' + str(path))
    # Strong Session opinions retain the old source-authored static values;
    # the already reviewed common H6 patch is strongest. No dynamic subtree
    # from either historical layer is copied.
    layers = []
    for name in ('Root_text', 'Session_text'):
        suffix = '.usda' if recorded[name].lstrip().startswith('#usda') else '.sdf'
        source = Sdf.Layer.CreateAnonymous('historical_static_source' + suffix)
        if not source.ImportFromString(recorded[name]):
            raise ValueError('Cannot read recorded layer')
        selected = Sdf.Layer.CreateAnonymous('selected_static_opinions.usda')
        copy_static(source, selected)
        layers.append(selected)
    UsdUtils.StitchLayers(layers[1], layers[0])
    common, _, _ = geometry.checked_common_Session_patch(patch_path.read_text())
    UsdUtils.StitchLayers(common, layers[1])
    return common, {str(p.resolve()): sha(p) for p in paths}


async def verify_rows(stage, expected, *, next_frame, unchanged):
    import hospital_h4_subfloor_compact_path_launch as shared
    prep = shared.preparation
    actual = {}
    for path, row in prep.compact.stage_rows(stage, prep.SELECTED, prep.old.original):
        if path in actual:
            raise ValueError('Duplicate protected environment row')
        actual[path] = row
        if len(actual) % 32 == 0:
            unchanged()
            await next_frame()
            unchanged()
    if set(actual) != set(expected) or len(actual) != 4997:
        raise ValueError('Complete recorded 4997-row roster required')
    differences = [p for p in expected if actual[p] != expected[p]]
    if differences:
        raise ValueError('Static environment differs: ' + repr(differences[:12]))
    return hashlib.sha256(prep.compact._json_bytes(
        dict(format=prep.compact.FORMAT, rows=actual))).hexdigest()


async def prepare_environment_once(*, authorized=False):
    if authorized is not True:
        raise ValueError('Fresh application environment preparation authority required')
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Sdf, Usd, UsdPhysics
    from hospital_runtime_evidence_io import write_exclusive
    import hospital_h4_cohort_preparation as constructor
    if OUTPUT.exists():
        raise FileExistsError('Consumed bootstrap; no automatic retry: ' + str(OUTPUT))
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    timeline = omni.timeline.get_timeline_interface()
    source = str(constructor.SOURCE)
    if (not stage or stage.GetRootLayer().identifier != source
            or sha(source) != constructor.SOURCE_SHA):
        raise ValueError('Exact original hospital source required')
    state = lambda: (timeline.is_playing(), timeline.is_stopped(), timeline.get_current_time())
    if state() != (False, True, 0.):
        raise ValueError('Fresh untouched STOPPED source required; no reset')
    bodies = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 13 or any(not p.startswith('/World/patient_transport/') for p in bodies):
        raise ValueError('Fresh original source actor only; never replace another live experiment')
    session = stage.GetSessionLayer()
    if any(session.GetPrimAtPath(p) for p in STATIC_ROOTS):
        raise ValueError('Do not replace existing live Session environment opinions')
    root_text = stage.GetRootLayer().ExportToString()
    session_text = session.ExportToString()
    patch, pins = build_static_patch()
    expected_receipt = read(RECORDED / 'actual_common_environment_rows.json')
    expected = expected_receipt['compact_rows']
    def unchanged():
        if (ctx.get_stage() != stage or state() != (False, True, 0.)
                or stage.GetRootLayer().ExportToString() != root_text):
            raise RuntimeError('Fresh source/timeline changed during preparation')
    # Preview is not attached to the live context. Reuse the live immutable
    # Root so its existing relative references retain their resolution anchor.
    preview = Usd.Stage.Open(stage.GetRootLayer(), patch)
    if not preview:
        raise ValueError('Cannot compose isolated static preview')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_exclusive(OUTPUT / 'reserved.json', dict(
        kind='STATIC_ENVIRONMENT_RECONSTRUCTION_ONLY', input_sha256=pins,
        old_dynamic_state_restored=False, Play_Stop_static_route_calls=0,
        automatic_retry=False))
    next_frame = omni.kit.app.get_app().next_update_async
    try:
        preview_digest = await verify_rows(preview, expected, next_frame=next_frame, unchanged=unchanged)
        if preview_digest != expected_receipt['sha256']:
            raise ValueError('Preview digest does not match recorded common environment')
        if session.ExportToString() != session_text:
            raise RuntimeError('Live Session changed during isolated preview; do not apply')
        for path, value in pins.items():
            if sha(path) != value:
                raise RuntimeError('Bootstrap input changed during preview')
        unchanged()
        copied = copy_static(patch, session)
        digest = await verify_rows(stage, expected, next_frame=next_frame, unchanged=unchanged)
        if digest != preview_digest:
            raise RuntimeError('Actual common environment differs from isolated preview')
        write_exclusive(OUTPUT / 'environment_prepared.json', dict(
            status='FRESH_COMMON_ENVIRONMENT_VERIFIED_NOT_STATIC_OR_DRIVE_ADMISSION',
            input_sha256=pins, source_file_unchanged=sha(source) == constructor.SOURCE_SHA,
            original_live_Root_text_unchanged=True, Session_static_roots=copied,
            protected_environment_rows=4997, protected_environment_sha256=digest,
            matches_recorded_common_environment=True, source_actor_untouched=bodies,
            old_dynamic_state_restored=False, Play_Stop_static_route_calls=0,
            timeline=dict(playing=False, stopped=True, current_time_s=0.),
            automatic_retry=False, use_for_final_statistics=False))
        print('[FRESH ENVIRONMENT] all4997 common rows verified; STOPPED; no static/route.', flush=True)
    except Exception as error:
        write_exclusive(OUTPUT / 'error.json', dict(error=type(error).__name__ + ': ' + str(error),
            no_automatic_retry=True, actual_timeline=list(state()), no_PhysX_trial_run=True))
        raise

