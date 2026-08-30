"""Fail-closed source lineage for M1+Panda residual PPO artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


PILOT_SCHEMA = {
    "schema_version": 1,
    "updates": 10,
    "hard_failure_count_max": 0,
    "mpc_feasible_rate_min": 0.99,
    "qp_feasible_rate": 1.0,
    "four_contact_rate": 1.0,
    "saturation_fraction_max_exclusive": 0.01,
    "kl_abort_count_max": 3,
    "median_completed_mini_batches_min": 6,
    "median_value_loss_max_exclusive": 100.0,
    "action_std_min": 0.005,
    "action_std_max": 0.02,
}


@dataclass(frozen=True)
class ResidualSourcePaths:
    asset: Path
    config: Path
    reward: Path
    runtime: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pilot_schema_sha256() -> str:
    return _canonical_sha256(PILOT_SCHEMA)


def reward_runtime_bundle_sha256(
    reward_path: str | Path, runtime_path: str | Path
) -> str:
    return _canonical_sha256(
        {
            "reward_sha256": sha256_file(reward_path),
            "runtime_sha256": sha256_file(runtime_path),
        }
    )


def source_lineage(paths: ResidualSourcePaths) -> dict[str, str]:
    resolved = {
        label: Path(getattr(paths, label)).expanduser().resolve()
        for label in ("asset", "config", "reward", "runtime")
    }
    for path in resolved.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {
        f"{label}_sha256": sha256_file(path)
        for label, path in resolved.items()
    }
    bundle = reward_runtime_bundle_sha256(
        resolved["reward"], resolved["runtime"]
    )
    return {
        **{f"{label}_path": str(path) for label, path in resolved.items()},
        **hashes,
        "reward_runtime_bundle_sha256": bundle,
        "pilot_schema_sha256": pilot_schema_sha256(),
    }


def validate_source_lineage(
    document: Mapping[str, object], paths: ResidualSourcePaths
) -> None:
    expected = source_lineage(paths)
    labels = {
        "asset_path": "asset path",
        "config_path": "config path",
        "reward_path": "reward path",
        "runtime_path": "runtime path",
        "asset_sha256": "asset SHA",
        "config_sha256": "config SHA",
        "reward_sha256": "reward SHA",
        "runtime_sha256": "runtime SHA",
        "reward_runtime_bundle_sha256": "reward-runtime bundle SHA",
        "pilot_schema_sha256": "pilot schema SHA",
    }
    for key, label in labels.items():
        if key not in document:
            raise ValueError(f"source lineage is missing {key}")
        value = document[key]
        if not isinstance(value, str) or value != expected[key]:
            raise ValueError(f"{label} does not match current source lineage")


__all__ = [
    "PILOT_SCHEMA",
    "ResidualSourcePaths",
    "pilot_schema_sha256",
    "reward_runtime_bundle_sha256",
    "sha256_file",
    "source_lineage",
    "validate_source_lineage",
]
