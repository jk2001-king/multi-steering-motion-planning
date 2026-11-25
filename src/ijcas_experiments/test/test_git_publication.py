"""Publication may extend Git history, but not silently bless source changes."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'tools'))
from verify_workspace import expected_active_digest


def test_unmodified_source_keeps_original_expected_digest():
    assert expected_active_digest('runtime/control.py', {'active_sha256': 'old'}, {}) == 'old'


def test_revision_requires_exact_previous_digest():
    relative = 'src/ijcas_experiments/tools/verify_workspace.py'
    with pytest.raises(ValueError, match='Unreviewed'):
        expected_active_digest(relative, {'active_sha256': 'old'}, {
            relative: {'previous_active_sha256': 'other', 'active_sha256': 'new'}})


def test_publication_receipt_cannot_allow_control_kernel_changes():
    relative = 'runtime/control.py'
    with pytest.raises(ValueError, match='Unreviewed'):
        expected_active_digest(relative, {'active_sha256': 'old'}, {
            relative: {'previous_active_sha256': 'old', 'active_sha256': 'new'}})


def test_reviewed_verifier_receipt_matches_exact_source():
    revisions = json.loads((PACKAGE / 'config/publication_source_revisions.json').read_text())
    assert list(revisions) == ['src/ijcas_experiments/tools/verify_workspace.py']
    entry = next(iter(revisions.values()))
    assert hashlib.sha256((PACKAGE / 'tools/verify_workspace.py').read_bytes()).hexdigest() == entry['active_sha256']
