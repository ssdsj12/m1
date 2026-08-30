from __future__ import annotations

import hashlib
import json

import pytest

from go2_pvcnn.training.m1_panda_arm_mpc_residual_lineage import (
    ResidualSourcePaths,
    pilot_schema_sha256,
    sha256_file,
    source_lineage,
    validate_source_lineage,
)


def _paths(tmp_path):
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


def test_source_lineage_uses_canonical_reward_runtime_bundle(tmp_path):
    paths = _paths(tmp_path)

    lineage = source_lineage(paths)
    payload = json.dumps(
        {
            "reward_sha256": lineage["reward_sha256"],
            "runtime_sha256": lineage["runtime_sha256"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert lineage["asset_path"] == str(paths.asset.resolve())
    assert lineage["runtime_sha256"] == sha256_file(paths.runtime)
    assert lineage["reward_runtime_bundle_sha256"] == hashlib.sha256(
        payload
    ).hexdigest()
    assert lineage["pilot_schema_sha256"] == pilot_schema_sha256()
    validate_source_lineage(lineage, paths)


def test_source_lineage_fails_closed_on_runtime_or_bundle_drift(tmp_path):
    paths = _paths(tmp_path)
    lineage = source_lineage(paths)

    paths.runtime.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime SHA"):
        validate_source_lineage(lineage, paths)

    current = source_lineage(paths)
    current["reward_runtime_bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reward-runtime bundle SHA"):
        validate_source_lineage(current, paths)


@pytest.mark.parametrize(
    "missing",
    (
        "asset_sha256",
        "config_sha256",
        "reward_sha256",
        "runtime_sha256",
        "reward_runtime_bundle_sha256",
        "pilot_schema_sha256",
    ),
)
def test_source_lineage_rejects_missing_hash_fields(tmp_path, missing):
    paths = _paths(tmp_path)
    lineage = source_lineage(paths)
    lineage.pop(missing)

    with pytest.raises(ValueError, match="missing"):
        validate_source_lineage(lineage, paths)


def test_pilot_schema_hash_is_stable_lowercase_sha256():
    digest = pilot_schema_sha256()

    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")
    assert digest == pilot_schema_sha256()
