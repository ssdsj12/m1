from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from go2_pvcnn.training.m1_panda_arm_mpc_residual_bridge import (
    BRIDGE_LEGACY_SAMPLE_COUNT,
    migrate_legacy_u100_checkpoint,
    validate_bridge_parent,
    validate_counted_checkpoint,
)
from go2_pvcnn.training.m1_panda_arm_mpc_residual_lineage import (
    ResidualSourcePaths,
    sha256_file,
    source_lineage,
)


def _write_sources(tmp_path: Path) -> ResidualSourcePaths:
    paths = ResidualSourcePaths(
        asset=tmp_path / "robot.usd",
        config=tmp_path / "train_cfg.py",
        reward=tmp_path / "reward.py",
        runtime=tmp_path / "wrapper.py",
    )
    for name, path in (
        ("asset", paths.asset),
        ("config", paths.config),
        ("reward", paths.reward),
        ("runtime", paths.runtime),
    ):
        path.write_text(name, encoding="utf-8")
    return paths


def _legacy_checkpoint(path: Path) -> dict[str, object]:
    payload = {
        "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(100)}}},
        "obs_norm_state_dict": {
            "_mean": torch.tensor([[1.0, 2.0]]),
            "_var": torch.tensor([[3.0, 4.0]]),
            "_std": torch.tensor([[5.0, 6.0]]),
        },
        "critic_obs_norm_state_dict": {
            "_mean": torch.tensor([[7.0, 8.0]]),
            "_var": torch.tensor([[9.0, 10.0]]),
            "_std": torch.tensor([[11.0, 12.0]]),
        },
        "iter": 99,
    }
    torch.save(payload, path)
    return payload


def _manifest_fixture(tmp_path: Path):
    paths = _write_sources(tmp_path)
    lineage = source_lineage(paths)
    pilot = tmp_path / "pilot_manifest.json"
    pilot.write_text(
        json.dumps(
            {
                **lineage,
                "schema_version": 2,
                "stage": "pilot",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": False,
                "pilot_accepted": True,
                "requested_iterations": 10,
                "completed_iterations": 10,
                "optimizer_summaries": [{"update": value} for value in range(1, 11)],
                "pilot_decision": {"accepted": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    candidates = []
    for update in (0, 25, 50, 75, 100):
        checkpoint = tmp_path / f"candidate_u{update:03d}.pt"
        _legacy_checkpoint(checkpoint)
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "completed_updates": update,
            }
        )
    short = tmp_path / "short_manifest.json"
    short.write_text(
        json.dumps(
            {
                **lineage,
                "schema_version": 2,
                "stage": "short",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": True,
                "requested_iterations": 100,
                "completed_iterations": 100,
                "pilot_manifest": str(pilot),
                "pilot_manifest_sha256": sha256_file(pilot),
                "candidate_checkpoints": candidates,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return paths, pilot, short, candidates


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    else:
        assert left == right


def test_bridge_parent_and_legacy_count_migration_are_exact(tmp_path):
    paths, _, short, _ = _manifest_fixture(tmp_path)
    parent = validate_bridge_parent(short, paths)

    assert parent.completed_updates == 100
    assert parent.sample_count == BRIDGE_LEGACY_SAMPLE_COUNT == 204800
    assert parent.checkpoint_sha256 == sha256_file(parent.checkpoint)

    original = torch.load(parent.checkpoint, map_location="cpu", weights_only=False)
    migrated = tmp_path / "migrated_u100.pt"
    migrated_sha = migrate_legacy_u100_checkpoint(
        parent.checkpoint,
        migrated,
        expected_parent_sha256=parent.checkpoint_sha256,
        sample_count=parent.sample_count,
    )
    assert migrated_sha == sha256_file(migrated)
    assert validate_counted_checkpoint(migrated, expected_count=204800) == (
        204800,
        204800,
    )

    migrated_payload = torch.load(migrated, map_location="cpu", weights_only=False)
    for key, original_value in original.items():
        migrated_value = migrated_payload[key]
        if key in ("obs_norm_state_dict", "critic_obs_norm_state_dict"):
            migrated_value = dict(migrated_value)
            assert migrated_value.pop("_count").dtype == torch.int64
        _assert_nested_equal(original_value, migrated_value)


@pytest.mark.parametrize("updates", ((0, 25, 50, 75), (0, 25, 50, 75, 99, 100)))
def test_bridge_parent_rejects_wrong_candidate_set(tmp_path, updates):
    paths, _, short, candidates = _manifest_fixture(tmp_path)
    document = json.loads(short.read_text(encoding="utf-8"))
    document["candidate_checkpoints"] = [
        value for value in candidates if value["completed_updates"] in updates
    ]
    if 99 in updates:
        extra = copy.deepcopy(candidates[-1])
        extra["completed_updates"] = 99
        document["candidate_checkpoints"].append(extra)
    short.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate"):
        validate_bridge_parent(short, paths)


def test_bridge_parent_rejects_wrong_u100_hash(tmp_path):
    paths, _, short, _ = _manifest_fixture(tmp_path)
    document = json.loads(short.read_text(encoding="utf-8"))
    document["candidate_checkpoints"][-1]["checkpoint_sha256"] = "0" * 64
    short.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        validate_bridge_parent(short, paths)


@pytest.mark.parametrize("missing", ("obs_norm_state_dict", "critic_obs_norm_state_dict"))
def test_bridge_parent_rejects_missing_normalizer_dictionary(tmp_path, missing):
    paths, _, short, candidates = _manifest_fixture(tmp_path)
    checkpoint = Path(candidates[-1]["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop(missing)
    torch.save(payload, checkpoint)
    document = json.loads(short.read_text(encoding="utf-8"))
    document["candidate_checkpoints"][-1]["checkpoint_sha256"] = sha256_file(checkpoint)
    short.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="normalizer"):
        validate_bridge_parent(short, paths)


def test_bridge_parent_rejects_preexisting_count(tmp_path):
    paths, _, short, candidates = _manifest_fixture(tmp_path)
    checkpoint = Path(candidates[-1]["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["obs_norm_state_dict"]["_count"] = torch.tensor(204800, dtype=torch.int64)
    torch.save(payload, checkpoint)
    document = json.loads(short.read_text(encoding="utf-8"))
    document["candidate_checkpoints"][-1]["checkpoint_sha256"] = sha256_file(checkpoint)
    short.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        validate_bridge_parent(short, paths)


def test_migration_rejects_wrong_sha_or_existing_target(tmp_path):
    source = tmp_path / "legacy.pt"
    _legacy_checkpoint(source)
    target = tmp_path / "migrated.pt"
    with pytest.raises(ValueError, match="SHA"):
        migrate_legacy_u100_checkpoint(
            source,
            target,
            expected_parent_sha256="0" * 64,
            sample_count=204800,
        )
    target.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        migrate_legacy_u100_checkpoint(
            source,
            target,
            expected_parent_sha256=sha256_file(source),
            sample_count=204800,
        )
