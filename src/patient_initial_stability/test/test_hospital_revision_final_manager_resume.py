from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'studies/IJCAS/scripts'))
import hospital_revision_final_manager_resume as module


def test_only_exact_offline_guard_is_an_expected_negative():
    assert module.expected_offline_rejection(ValueError(module.GUARD_ERROR))
    assert not module.expected_offline_rejection(RuntimeError(module.GUARD_ERROR))
    assert not module.expected_offline_rejection(ValueError('Unknown transform/pose/physics change'))


def test_negative_keeps_rejection_and_does_not_invent_time_or_collision(tmp_path,monkeypatch):
    monkeypatch.setattr(module.base,'OUTPUT',tmp_path)
    name,variant=module.CONSUMED
    folder=tmp_path/'offline_envelope'/variant/name;folder.mkdir(parents=True)
    (folder/'definition.json').write_text(json.dumps(dict(input_sha256={})))
    (folder/'own_seed.json').write_text(json.dumps(dict(plan_success=True,owned_seed_total_time_s=24.)))
    raw=module.negative_receipt(name,variant,ValueError(module.GUARD_ERROR),already_observed=True)
    assert not raw['plan_success'] and not raw['independently_valid_reference']
    assert raw['planning_time_s'] is None and raw['full_envelope_colliding_samples'] is None
    assert raw['new_PhysX_trials']==0 and raw['guard_NOT_waived_or_changed']
    assert raw['consumed_attempt_NOT_repeated'] and raw['original_already_observed_rejection']
    with pytest.raises(FileExistsError):
        module.negative_receipt(name,variant,ValueError(module.GUARD_ERROR))


def test_recovery_is_for_offline_negative_not_main_or_physics_error(tmp_path,monkeypatch):
    monkeypatch.setattr(module.base,'OUTPUT',tmp_path)
    with pytest.raises(ValueError,match='physics'):
        module.negative_receipt('any','point',ValueError('physics mismatch'))
    with pytest.raises(ValueError,match='Own spatial'):
        module.negative_receipt('any','main',ValueError(module.GUARD_ERROR))


def test_existing_main_and_first_drive_are_never_replayed(tmp_path,monkeypatch):
    monkeypatch.setattr(module.base,'OUTPUT',tmp_path)
    monkeypatch.setattr(module.base,'verify_sources',lambda:None)
    (tmp_path/'dispatch.json').write_text('{}')
    with pytest.raises(ValueError,match='unstarted FIRST'):module.finish_offline()
