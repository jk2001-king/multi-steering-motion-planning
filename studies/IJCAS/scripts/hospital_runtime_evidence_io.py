"""Bounded evidence serialization and deferred-Pause confirmation.

No Play, Stop, reset, stage edit, physics initialization or implicit retry.
Existing frozen writers/runtimes are deliberately not modified.
"""

import json
import math
import os
from pathlib import Path
import time


MAX_BYTES = 4 * 1024 * 1024


def json_value(value):
    """Only explicit JSON types; tuple/list conversion is representation only.

    Reject non-string keys rather than silently conflating 1 and '1'. Reject
    NaN/Inf and unsupported objects BEFORE exclusively creating any file.
    """
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Non-finite JSON evidence')
        return value
    if type(value) in (list, tuple):
        return [json_value(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError('Evidence keys must be strings; no coercion')
        return {key: json_value(item) for key, item in value.items()}
    raise TypeError('Unsupported evidence type: ' + type(value).__name__)


def write_exclusive(path, value):
    normalized = json_value(value)
    payload = (json.dumps(normalized, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if len(payload) > MAX_BYTES:
        raise ValueError('Evidence exceeds explicit 4 MiB budget')
    if json.loads(payload) != normalized:
        raise RuntimeError('JSON pre-creation verification failed')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # Persist the new directory entry, not merely the open file contents.
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload or json.loads(path.read_bytes()) != normalized:
        raise RuntimeError('Durable evidence bytes failed verification; file preserved: ' + str(path))
    return normalized


def timeline_state(timeline):
    return dict(playing=bool(timeline.is_playing()), stopped=bool(timeline.is_stopped()),
        current_time_s=float(timeline.get_current_time()))


async def confirm_pause(timeline, next_update, *, check_context=lambda: None,
        max_updates=120, wall_budget_s=5.):
    """Complete ONLY a Pause, then verify a stable PAUSED UI boundary.

    A request is not an observation. UI yields are needed for a deferred
    timeline transition. No environment hashing or query is performed until
    the requested transition AND a subsequent stable boundary are observed.
    """
    before = timeline_state(timeline)
    if before['stopped']:
        raise RuntimeError('Already STOPPED; no Play/reset to recreate a paused trial')
    requested = before['playing']
    if requested:
        timeline.pause()
    began, updates = time.monotonic(), 0
    while timeline.is_playing():
        if updates >= max_updates or time.monotonic() - began >= wall_budget_s:
            raise TimeoutError('Pause completion unconfirmed; no queries/cleanup/admission')
        check_context()
        await next_update()
        check_context()
        updates += 1
        if timeline.is_stopped():
            raise RuntimeError('Unexpected Stop/reset during Pause completion')
    paused = timeline_state(timeline)
    if paused['stopped'] or not math.isfinite(paused['current_time_s']):
        raise RuntimeError('A finite PAUSED state is required')
    check_context()
    await next_update()
    check_context()
    updates += 1
    if timeline_state(timeline) != paused:
        raise RuntimeError('Timeline changed after Pause; no state-restoration claim')
    return dict(before=before, after=paused, pause_requested_by_this_call=requested,
        updates_to_observe_stable_pause=updates, any_Play_Stop_reset_or_trial=False)
