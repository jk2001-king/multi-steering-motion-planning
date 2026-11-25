"""Resolve archived path strings without editing original experiment evidence.

This is explicit and opt-in: no monkey patch, old-workspace fallback or simulator
command. Hashes must be calculated on the original bytes, not the returned view.
"""
from pathlib import Path
from functools import lru_cache
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path('/home/jk/Research/assistive_initial_motion_ws')
ARCHIVED_RELOCATION_ROOT = Path('/home/jk/Research/IJCAS_ws')


@lru_cache(maxsize=1)
def _technical_data_moves():
    receipt = ROOT/'migration/publication_cleanup.json'
    return json.loads(receipt.read_text()).get('technical_data_moves', {}) if receipt.is_file() else {}


def resolve_archived_path(value):
    path = Path(value)
    for base in (LEGACY_ROOT, ARCHIVED_RELOCATION_ROOT, ROOT):
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError:
            continue
        return ROOT / _technical_data_moves().get(relative, relative)
    return path


def resolve_evidence_path(value):
    """Find original bytes for a historically pinned source hash, if migrated."""
    path = Path(value)
    try:
        relative = path.relative_to(LEGACY_ROOT)
    except ValueError:
        return path
    preserved = ROOT / 'migration/original_sources' / relative
    return preserved if preserved.is_file() else resolve_archived_path(path)


def relocate_json_view(value):
    if isinstance(value, str):
        if any(value == str(base) or value.startswith(str(base) + '/')
                for base in (LEGACY_ROOT, ARCHIVED_RELOCATION_ROOT, ROOT)):
            return str(resolve_archived_path(value))
        return value
    if isinstance(value, list):
        return [relocate_json_view(item) for item in value]
    if isinstance(value, dict):
        return {relocate_json_view(key): relocate_json_view(item)
                for key, item in value.items()}
    return value


def load_archived_json(path):
    return relocate_json_view(json.loads(resolve_archived_path(path).read_text()))


if __name__ == '__main__':
    for value in sys.argv[1:]:
        print(resolve_archived_path(value))
