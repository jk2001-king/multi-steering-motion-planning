"""Explicit portable paths for code, immutable evidence, and new run outputs.

Historical JSON/source ZIP bytes are never edited. No old workspace, symlink
compatibility root, global monkey patch, or simulator action is required.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import sys


def workspace_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src/ijcas_experiments/package.xml').is_file():
            return parent
    value = os.environ.get('IJCAS_WORKSPACE_ROOT', '')
    if value and (Path(value) / 'src/ijcas_experiments/package.xml').is_file():
        return Path(value).resolve()
    raise RuntimeError('Source workspace not found; source install/setup.bash')


ROOT = workspace_root()
PACKAGE = ROOT / 'src/ijcas_experiments'
LEGACY_ROOT = Path('/home/jk/Research/assistive_initial_motion_ws')
ARCHIVED_RELOCATION_ROOT = Path('/home/jk/Research/IJCAS_ws')
MAPPINGS = (
    ('tools/ijcas_paths.py', 'src/ijcas_experiments/ijcas_paths.py'),
    ('studies/IJCAS/scripts', 'src/ijcas_experiments/runtime/study'),
    ('studies/IJCAS/config', 'src/ijcas_experiments/config/study'),
    ('isaac/scripts', 'src/ijcas_experiments/runtime/sdk'),
    ('isaac', 'src/ijcas_experiments/assets/isaac'),
    ('tools', 'src/ijcas_experiments/tools'),
    ('config', 'src/ijcas_experiments/config'),
    ('docs', 'src/ijcas_experiments/docs'),
    ('migration', 'archive/original/migration'),
    ('data', 'archive/original/data'),
    ('studies', 'archive/original/studies'),
    ('external', 'archive/original/external'),
    ('isaac_model', 'archive/original/isaac_model'),
    ('results', 'log/ijcas'),
)


def _mapped(relative):
    relative = Path(relative)
    for old, new in MAPPINGS:
        if relative.is_relative_to(old):
            return ROOT / new / relative.relative_to(old)
    return ROOT / relative


def layout_path(base, *parts):
    """Resolve a legacy workspace path expression; unrelated test roots stay local."""
    value = Path(base).joinpath(*parts)
    for origin in (ROOT, LEGACY_ROOT, ARCHIVED_RELOCATION_ROOT):
        if value.is_relative_to(origin):
            return _mapped(value.relative_to(origin))
    return value


@lru_cache(maxsize=1)
def _technical_data_moves():
    receipt = ROOT / 'archive/original/migration/publication_cleanup.json'
    return json.loads(receipt.read_text()).get('technical_data_moves', {}) if receipt.is_file() else {}


def resolve_archived_path(value):
    path = Path(value)
    for base in (ROOT, LEGACY_ROOT, ARCHIVED_RELOCATION_ROOT):
        if not path.is_relative_to(base):
            continue
        relative = path.relative_to(base)
        relative = Path(_technical_data_moves().get(relative.as_posix(), relative))
        # Existing old run IDs are evidence, not new log destinations.
        if relative.is_relative_to('results'):
            saved = ROOT / 'archive/original' / relative
            if saved.exists():
                return saved
        return _mapped(relative)
    return path


def resolve_evidence_path(value):
    path = Path(value)
    for base in (ROOT, LEGACY_ROOT, ARCHIVED_RELOCATION_ROOT):
        if path.is_relative_to(base):
            relative = Path(legacy_relative_path(resolve_archived_path(path)))
            preserved = ROOT / 'archive/original/migration/original_sources' / relative
            return preserved if preserved.is_file() else resolve_archived_path(path)
    return path


def legacy_relative_path(path):
    """Inverse name for historical manifests / ZIP members, never a disk fallback."""
    path = Path(path)
    if not path.is_relative_to(ROOT):
        return str(path)
    relative = path.relative_to(ROOT)
    for old, new in MAPPINGS:
        if relative.is_relative_to(new):
            return str(Path(old) / relative.relative_to(new))
    if relative.is_relative_to('archive/original'):
        return str(relative.relative_to('archive/original'))
    return str(relative)


def archive_member_path(path, names):
    path = Path(path)
    candidates = [path.relative_to(ROOT).as_posix(), legacy_relative_path(path)]
    matches = list(dict.fromkeys(name for name in candidates if name in names))
    if len(matches) != 1:
        raise ValueError('Exactly one original/current bounded source member required: ' + str(path))
    return matches[0]


@lru_cache(maxsize=1)
def layout_manifest():
    file = ROOT / 'archive/layout_20261126/source_manifest.json'
    return json.loads(file.read_text()) if file.is_file() else {}


def original_layout_source(path):
    """Return a pre-layout source view only after verifying both full-file hashes.

    The migration manifest is local evidence, not a control/safety allowlist.
    Its entries were produced by reversible path-only AST substitutions.
    """
    path = Path(path).resolve()
    relative = str(path.relative_to(ROOT))
    entry = layout_manifest().get(relative)
    if entry is None:
        return path.read_bytes()
    current = path.read_bytes()
    if hashlib.sha256(current).hexdigest() != entry['active_sha256']:
        raise ValueError('Active source differs from reviewed layout transform: ' + str(path))
    saved = ROOT / 'archive/original' / entry['original_path']
    value = saved.read_bytes()
    if hashlib.sha256(value).hexdigest() != entry['original_sha256']:
        raise ValueError('Original layout source bytes changed: ' + str(saved))
    return value


def relocate_json_view(value):
    if isinstance(value, str):
        if any(value == str(base) or value.startswith(str(base) + '/')
                for base in (ROOT, LEGACY_ROOT, ARCHIVED_RELOCATION_ROOT)):
            return str(resolve_archived_path(value))
        return value
    if isinstance(value, list):
        return [relocate_json_view(item) for item in value]
    if isinstance(value, dict):
        return {relocate_json_view(key): relocate_json_view(item) for key, item in value.items()}
    return value


def load_archived_json(path):
    return relocate_json_view(json.loads(resolve_archived_path(path).read_text()))


if __name__ == '__main__':
    for value in sys.argv[1:]:
        print(resolve_archived_path(value))
