"""Lossless compact fingerprints for the existing USD semantic snapshot.

No Kit, physics, stage mutation or large JSON expansion. Numeric USD buffers
are hashed in bulk after lossless promotion to the scalar types represented
by the old JSON. Archived JSON can be decoded one prim at a time OFFLINE.
The resulting per-prim digests can be compared to live USD without loading
the multi-gigabyte archive into the Script Editor process.
"""

import hashlib
import json
import math
import mmap
import re
import struct

import numpy as np

import hospital_h4_root_difference_readback as fields


FORMAT = 'USD_semantic_property_tree_numeric_bulk_sha256_v1'


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _numeric_json(value):
    """Describe every homogeneous typed container, not only its flat values."""
    if not isinstance(value, dict) or set(value) != {'type', 'values'} or not isinstance(value['values'], list):
        return None
    nodes = [value]; descriptor = []
    while nodes and isinstance(nodes[0], dict):
        first = nodes[0]
        if set(first) != {'type', 'values'} or not isinstance(first['values'], list):
            return None
        name, size = first['type'], len(first['values'])
        if any(not isinstance(n, dict) or set(n) != {'type', 'values'} or n['type'] != name
                or not isinstance(n['values'], list) or len(n['values']) != size for n in nodes):
            return None
        descriptor.append([name, size])
        nodes = [v for n in nodes for v in n['values']]
    if not nodes:
        return descriptor, None
    if not all(type(n) in (bool, int, float) for n in nodes):
        return None
    array = np.asarray(nodes)
    if array.dtype.kind not in 'biuf' or (array.dtype.kind == 'f' and not np.isfinite(array).all()):
        return None
    return descriptor, array


def _numeric_native(value):
    if isinstance(value, (str, bytes, dict, bool, int, float)) or hasattr(value, 'GetReal'):
        return None
    try:
        if len(value) == 0:
            return [[type(value).__name__, 0]], None
        array = np.asarray(value)
        if array.ndim == 0 or array.dtype.kind not in 'biuf':
            return None
        if array.dtype.kind == 'f' and not np.isfinite(array).all():
            return None
        descriptor = []; current = value
        while not isinstance(current, (bool, int, float, np.number)):
            count = len(current); descriptor.append([type(current).__name__, count])
            if not count:
                return descriptor, None
            current = current[0]
        return descriptor, array
    except (TypeError, ValueError, IndexError):
        return None


def _digest_numeric(descriptor, array):
    digest = hashlib.sha256(b'numeric-semantic-tree\0'+_json_bytes(descriptor))
    if array is None:
        digest.update(b'empty')
    else:
        if array.dtype.kind == 'f':
            dtype = '<f8'
        elif array.dtype.kind == 'b':
            dtype = '|b1'
        elif array.dtype.kind == 'u' and array.size and int(array.max()) > np.iinfo(np.int64).max:
            dtype = '<u8'
        else:
            dtype = '<i8'
        normalized = np.ascontiguousarray(array, dtype=dtype).reshape(-1)
        digest.update(dtype.encode()+b'\0')
        digest.update(memoryview(normalized).cast('B'))
    return digest.hexdigest()


def semantic_json_digest(value):
    numeric = _numeric_json(value)
    if numeric is not None:
        return _digest_numeric(*numeric)
    if isinstance(value, dict):
        payload = ['dict', [[str(k), semantic_json_digest(value[k])] for k in sorted(value)]]
    elif isinstance(value, list):
        payload = ['list', [semantic_json_digest(v) for v in value]]
    elif isinstance(value, float):
        # Preserve signed zero and every bit that the original JSON retained.
        return hashlib.sha256(b'float\0'+struct.pack('<d', value)).hexdigest()
    else:
        payload = [type(value).__name__, value]
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def native_digest(value):
    numeric = _numeric_native(value)
    if numeric is not None:
        return _digest_numeric(*numeric)
    if isinstance(value, dict):
        return semantic_json_digest(fields.semantic_value(value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return semantic_json_digest(fields.semantic_value(value))
    if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
        return semantic_json_digest(fields.semantic_value(value))
    try:
        length = len(value)
        children = [native_digest(v) for v in value]
    except TypeError:
        return semantic_json_digest(fields.semantic_value(value))
    if length != len(children):
        raise ValueError('A semantic container changed during fingerprinting')
    # fields.semantic_value wraps iterable objects as {type, values}.
    values_digest = hashlib.sha256(_json_bytes(['list', children])).hexdigest()
    payload = ['dict', [['type', semantic_json_digest(type(value).__name__)], ['values', values_digest]]]
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def archived_row(path, value, selected):
    if path == '__unchanged_nominal_robot_and_scene__':
        return dict(row_sha256=semantic_json_digest(value), collision_enabled=None)
    attributes = {}
    for name, data in sorted(value['attributes'].items()):
        default = True if path in selected and name == 'physics:collisionEnabled' else data['value']
        attributes[name] = dict(default_sha256=semantic_json_digest(default),
            samples=[[t, semantic_json_digest(v)] for t, v in data['time_samples']])
    content = dict(active=value['active'], type=value['type'], attributes=attributes,
        relationships=value['relationships'], world_sha256=semantic_json_digest(value['world_matrix']),
        bound_physics_material=value['bound_physics_material'])
    flag = value['attributes'].get('physics:collisionEnabled', {}).get('value')
    return dict(row_sha256=hashlib.sha256(_json_bytes(content)).hexdigest(), collision_enabled=flag)


def _cache_key(attribute):
    """Only numeric authored arrays; never cache remapped paths/tokens/assets."""
    name = str(attribute.GetTypeName())
    if not name.endswith('[]') or not re.fullmatch(r'(?:bool|u?char|u?int(?:64)?|half\d?|float\d?|double\d?|(?:point|normal|vector|color|texCoord)\d[fdh]|matrix\dd)\[\]', name):
        return None
    if attribute.GetNumTimeSamples():
        return None
    stack = attribute.GetPropertyStack()
    return (name, tuple((s.layer.identifier, str(s.path)) for s in stack)) if stack else None


def stage_rows(stage, selected, original):
    """Yield compact rows, allowing the caller to update the UI between rows."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    cache = UsdGeom.XformCache(); numeric_cache = {}
    for root in ('/World/Hospital', '/World/PatientTransportLabGround', original.support.PATCH_ROOT):
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise ValueError('Protected environment root missing: '+root)
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not (p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdShade.Material)):
                continue
            path = str(p.GetPath()); attributes = {}; flag = None
            for a in sorted(p.GetAttributes(), key=lambda a: a.GetName()):
                name = a.GetName()
                if name == 'physics:collisionEnabled':
                    flag = a.Get()
                key = _cache_key(a)
                if key is not None and key in numeric_cache:
                    default = numeric_cache[key]
                else:
                    default = native_digest(True if path in selected and name == 'physics:collisionEnabled' else a.Get())
                    if key is not None:
                        numeric_cache[key] = default
                attributes[name] = dict(default_sha256=default,
                    samples=[[t, native_digest(a.Get(t))] for t in a.GetTimeSamples()])
            material, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')
            content = dict(active=bool(p.IsActive()), type=p.GetTypeName(), attributes=attributes,
                relationships={r.GetName(): list(map(str, r.GetTargets())) for r in p.GetRelationships()},
                world_sha256=native_digest(cache.GetLocalToWorldTransform(p)),
                bound_physics_material=str(material.GetPath()) if material else None)
            yield path, dict(row_sha256=hashlib.sha256(_json_bytes(content)).hexdigest(), collision_enabled=flag)
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('Exactly one unchanged PhysicsScene required')
    metadata = dict(asset_sha256=original.completion.sha(original.ASSET), simulation_mass_kg=265.,
        source_authored_seed=True, original_active_joint_settings=original.JOINT_SETTINGS,
        physics_scene_path=str(scenes[0].GetPath()),
        physics_scene_attributes={a.GetName(): fields.semantic_value(a.Get()) for a in scenes[0].GetAttributes()})
    # The old JSON roundtrip converted tuple settings to plain JSON lists.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    yield '__unchanged_nominal_robot_and_scene__', dict(row_sha256=semantic_json_digest(metadata), collision_enabled=None)


def compare_rows(expected, actual, selected, after_flags):
    if set(expected) != set(actual):
        raise ValueError('Protected collider/material roster changed')
    for path, first in expected.items():
        target = dict(first)
        if path in selected:
            if first['collision_enabled'] is not True:
                raise ValueError('All four approved original flags must be enabled in the archive')
            target['collision_enabled'] = False if after_flags else True
        if actual[path] != target:
            raise ValueError('Protected environment fingerprint/flag changed: '+path)
    return hashlib.sha256(_json_bytes(dict(format=FORMAT, rows=actual))).hexdigest()


def iter_archived_rows(path):
    """Read the immutable, indented original archive one prim at a time.

    mmap does not create a full Python byte/string copy. The largest original
    prim row is decoded independently; no full-archive json.loads/read_bytes.
    """
    start_marker = b'  "protected_environment_before_Stop": {'
    end_marker = b'\n  },\n  "environment_after_actual_static":'
    pattern = re.compile(rb'^    "((?:[^"\\\r\n]|\\.)*)": \{', re.MULTILINE)
    with open(path, 'rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        start = data.find(start_marker)
        end = data.find(end_marker, start)
        if start < 0 or end < 0:
            raise ValueError('Exact original protected-environment archive section required')
        matches = list(pattern.finditer(data, start, end))
        if not matches:
            raise ValueError('Original archive has no protected prim rows')
        decoder = json.JSONDecoder()
        for i, match in enumerate(matches):
            limit = matches[i+1].start() if i+1 < len(matches) else end
            piece = data[match.start():limit].decode('utf-8').strip().rstrip(',')
            key, index = decoder.raw_decode(piece)
            index += 1  # colon
            while piece[index].isspace():
                index += 1
            value, consumed = decoder.raw_decode(piece, index)
            if piece[consumed:].strip():
                raise ValueError('Unexpected archived row suffix')
            yield key, value, limit-match.start()
