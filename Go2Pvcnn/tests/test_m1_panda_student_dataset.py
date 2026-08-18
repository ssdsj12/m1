import dataclasses
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.tasks.m1_panda_student_dataset import (
    DaggerRecord,
    VersionedDaggerReplay,
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "asset_sha": "zero-clearance-sha",
        "teacher_commit": "teacher-commit",
        "observation_dim": 100,
        "history_length": 10,
        "action_dim": 23,
        "control_dt": 0.005,
        "action_scales": {
            "leg_position_rad": 0.25,
            "wheel_velocity_radps": 8.0,
            "arm_position_rad": 0.2,
        },
        "dagger_stage": "mix-50",
    }


def _record(*, env_id: int, episode_id: int, hard: bool, step: int = 0) -> DaggerRecord:
    return DaggerRecord(
        env_id=env_id,
        episode_id=episode_id,
        step=step,
        history=torch.full((10, 100), float(env_id)),
        teacher_action=torch.ones(23),
        executed_action=torch.zeros(23),
        wrench_target=torch.arange(6, dtype=torch.float32),
        safety_target=float(hard),
        hard=hard,
        metadata={"phase": "accelerate", "takeover": hard},
    )


def test_replay_keeps_environment_episode_identity_and_hard_samples(tmp_path):
    replay = VersionedDaggerReplay(capacity=4, hard_fraction=0.5, seed=5)
    replay.extend([
        _record(env_id=i % 2, episode_id=10 + i // 2, hard=i >= 2, step=i)
        for i in range(6)
    ])
    assert len(replay) == 4
    assert sum(record.hard for record in replay.records) >= 2
    path = tmp_path / "shard-00000.pt"
    replay.save(path, _manifest())
    loaded = VersionedDaggerReplay.load(path, expected_manifest=_manifest())
    assert [(r.env_id, r.episode_id) for r in loaded.records] == [
        (r.env_id, r.episode_id) for r in replay.records
    ]


def test_replay_validates_records_and_clones_tensor_storage():
    replay = VersionedDaggerReplay(capacity=3, hard_fraction=1 / 3, seed=1)
    record = _record(env_id=0, episode_id=2, hard=False)
    replay.add(record)
    record.history.fill_(99)
    assert replay.records[0].history[0, 0].item() == 0.0
    with pytest.raises(ValueError, match="history"):
        replay.add(dataclasses.replace(record, history=torch.zeros(9, 100)))
    with pytest.raises(ValueError, match="finite"):
        replay.add(dataclasses.replace(record, teacher_action=torch.full((23,), torch.nan)))


def test_weighted_sampling_favors_hard_samples_reproducibly():
    replay = VersionedDaggerReplay(capacity=10, hard_fraction=0.5, seed=17)
    replay.extend([
        *[_record(env_id=0, episode_id=i, hard=False) for i in range(5)],
        *[_record(env_id=1, episode_id=100 + i, hard=True) for i in range(5)],
    ])
    first = replay.sample(200, hard_weight=4.0)
    second = VersionedDaggerReplay(capacity=10, hard_fraction=0.5, seed=17)
    second.extend(replay.records)
    replayed = second.sample(200, hard_weight=4.0)
    assert [r.episode_id for r in first] == [r.episode_id for r in replayed]
    assert sum(record.hard for record in first) > 120


def test_atomic_save_leaves_complete_canonical_manifest_and_no_temp(tmp_path):
    replay = VersionedDaggerReplay(capacity=2, hard_fraction=0.5, seed=3)
    replay.extend([
        _record(env_id=0, episode_id=1, hard=False),
        _record(env_id=1, episode_id=2, hard=True),
    ])
    path = tmp_path / "shard.pt"
    replay.save(path, _manifest())
    manifest_path = Path(f"{path}.manifest.json")
    assert path.is_file() and manifest_path.is_file()
    assert json.loads(manifest_path.read_text()) == _manifest()
    assert list(tmp_path.glob("*.tmp")) == []
    text = manifest_path.read_text()
    assert text == json.dumps(_manifest(), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("asset_sha", "old-10mm-sha"),
        ("teacher_commit", "wrong"),
        ("observation_dim", 99),
        ("history_length", 9),
        ("action_dim", 16),
        ("control_dt", 0.01),
        ("action_scales", {"leg_position_rad": 1.0}),
        ("dagger_stage", "wrong-stage"),
    ],
)
def test_load_rejects_incompatible_manifest(tmp_path, field, value):
    replay = VersionedDaggerReplay(capacity=2, hard_fraction=0.5, seed=3)
    replay.add(_record(env_id=0, episode_id=1, hard=True))
    path = tmp_path / "shard.pt"
    replay.save(path, _manifest())
    expected = dict(_manifest())
    expected[field] = value
    with pytest.raises(ValueError, match=field):
        VersionedDaggerReplay.load(path, expected_manifest=expected)


def test_load_rejects_missing_or_partial_manifest(tmp_path):
    path = tmp_path / "shard.pt"
    torch.save({"records": []}, path)
    with pytest.raises(FileNotFoundError, match="manifest"):
        VersionedDaggerReplay.load(path, expected_manifest=_manifest())
    Path(f"{path}.manifest.json").write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match="asset_sha"):
        VersionedDaggerReplay.load(path, expected_manifest=_manifest())
