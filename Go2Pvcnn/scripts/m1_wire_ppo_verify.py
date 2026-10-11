#!/usr/bin/env python3
"""Evaluate a SHA-pinned Stage-A/Stage-B wire PPO checkpoint on GPU0."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = ROOT / "rsl_rl"
TRAIN_SCRIPT = ROOT / "scripts" / "m1_wire_ppo_train.py"
DEFAULT_CONFIG = ROOT / "config" / "m1_wire_ppo_train.yaml"
DEFAULT_OBJECT_CATALOG = ROOT / "config" / "m1_object_catalog.json"
DEFAULT_OBJECT_ASSETS_ROOT = ROOT / "assets" / "m1_objects"
for import_path in (ROOT, RSL_RL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


RUNTIME_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# Immutable formal Stage-B target limits.  Runtime aggregate verification uses
# the config reconstructed from the training manifest; this fallback exists
# only for the standalone pure-CPU helpers and is deliberately explicit rather
# than reading mutable ``WirePpoRewardCfg`` defaults.
FORMAL_ACTION_TARGET_LIMITS = {
    "action_maximum_centerline_error_m": 0.03,
    "action_maximum_tangent_error": 0.25,
    "action_maximum_deformation_error_m": 0.02,
    "action_maximum_synchronization_error_m_s": 0.05,
}


class _FormalActionTargetCfg:
    action_maximum_centerline_error_m = FORMAL_ACTION_TARGET_LIMITS[
        "action_maximum_centerline_error_m"
    ]
    action_maximum_tangent_error = FORMAL_ACTION_TARGET_LIMITS[
        "action_maximum_tangent_error"
    ]
    action_maximum_deformation_error_m = FORMAL_ACTION_TARGET_LIMITS[
        "action_maximum_deformation_error_m"
    ]
    action_maximum_synchronization_error_m_s = FORMAL_ACTION_TARGET_LIMITS[
        "action_maximum_synchronization_error_m_s"
    ]


FORMAL_ACTION_TARGET_CFG = _FormalActionTargetCfg()


def configure_runtime_environment() -> None:
    for name, value in RUNTIME_ENVIRONMENT.items():
        os.environ[name] = value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _manifest_sha(document: dict[str, object]) -> str:
    body = deepcopy(document)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_sha_manifest(path: Path, body: dict[str, object]) -> dict[str, object]:
    document = deepcopy(body)
    document.pop("manifest_sha256", None)
    document["manifest_sha256"] = _manifest_sha(document)
    _atomic_bytes(Path(path), _canonical_json(document))
    return document


def build_arg_parser(*, include_app_launcher_args: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("approval", "action"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--object-id", default="wire_000")
    parser.add_argument("--object-catalog", type=Path, default=DEFAULT_OBJECT_CATALOG)
    parser.add_argument("--object-assets-root", type=Path, default=DEFAULT_OBJECT_ASSETS_ROOT)
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(device="cuda:0", headless=True)
    else:
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument(
            "--headless", action=argparse.BooleanOptionalAction, default=True
        )
    return parser


def validate_cli_contract(args: argparse.Namespace) -> None:
    if args.stage not in {"approval", "action"}:
        raise ValueError("--stage must be approval or action")
    if not isinstance(args.episodes, int) or isinstance(args.episodes, bool) or args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_envs != 1:
        raise ValueError("--num-envs must be 1 for deterministic verification")
    if args.device != "cuda:0":
        raise ValueError("--device must be cuda:0")


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _forces(row: dict[str, object], side: str) -> tuple[float, ...]:
    value = row.get(f"contact_forces_{side}")
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(_finite(force, label=f"{side} contact force") for force in value)
    except ValueError:
        return ()


def _physical_gates(row: dict[str, object], *, stage: str) -> bool:
    from scripts import m1_wire_ppo_train as train

    normalized = dict(row)
    normalized["contact_forces_left"] = _forces(row, "left")
    normalized["contact_forces_right"] = _forces(row, "right")
    return train._formal_success_episode(normalized, expected_stage=stage)


def _action_target_gate(
    row: dict[str, object], *, reward_cfg: object
) -> bool:
    """Replay the Stage-B target with the pinned training reward config.

    The verifier must never silently instantiate ``WirePpoRewardCfg()`` here:
    that would make an evaluation artifact depend on whatever defaults happen
    to be checked out at verification time.  Callers reconstruct ``reward_cfg``
    from the SHA-pinned training manifest/config (or an explicitly supplied
    formal config) before invoking this gate.
    """
    target = row.get("action_target")
    if not isinstance(target, dict) or target.get("reached") is not True:
        return False
    try:
        return (
            _finite(target.get("wire_centerline_error"), label="wire centerline error")
            <= float(reward_cfg.action_maximum_centerline_error_m)
            and _finite(target.get("tangent_error"), label="tangent error")
            <= float(reward_cfg.action_maximum_tangent_error)
            and _finite(target.get("deformation_error"), label="deformation error")
            <= float(reward_cfg.action_maximum_deformation_error_m)
            and _finite(target.get("synchronization_error"), label="synchronization error")
            <= float(reward_cfg.action_maximum_synchronization_error_m_s)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def successful_episode_rows(
    rows: Iterable[dict[str, object]], *, stage: str, reward_cfg: object | None = None
) -> list[dict[str, object]]:
    if stage not in {"approval", "action"}:
        raise ValueError("stage must be approval or action")
    if stage == "action" and reward_cfg is None:
        reward_cfg = FORMAL_ACTION_TARGET_CFG
    selected: list[dict[str, object]] = []
    for row in rows:
        if row.get("success") is not True or not _physical_gates(row, stage=stage):
            continue
        if stage == "action" and not _action_target_gate(row, reward_cfg=reward_cfg):
            continue
        selected.append(row)
    return selected


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _load_sha_manifest(path: Path, *, label: str) -> dict[str, object]:
    document = _load_json_object(path, label=label)
    declared = document.get("manifest_sha256")
    if not isinstance(declared, str) or declared != _manifest_sha(document):
        raise ValueError(f"{label} SHA-256 mismatch")
    return document


def _load_training_ledger(
    document: dict[str, object], manifest_path: Path, *, stage: str
) -> list[dict[str, object]]:
    relative = document.get("episodes_jsonl")
    declared_sha = document.get("episodes_jsonl_sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint manifest has no episode ledger")
    ledger = (manifest_path.parent / relative).resolve()
    if ledger.parent != manifest_path.parent.resolve() or not ledger.is_file():
        raise ValueError("checkpoint episode ledger must be a file inside the run directory")
    if not isinstance(declared_sha, str) or declared_sha != sha256_file(ledger):
        raise ValueError("episode ledger SHA-256 mismatch")
    rows: list[dict[str, object]] = []
    counts = {"accepted": 0, "rejected": 0}
    reasons: Counter[str] = Counter()
    for index, line in enumerate(ledger.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"episode ledger row {index} is invalid JSON") from error
        if not isinstance(row, dict) or row.get("episode_index") != index:
            raise ValueError("episode ledger indices must be contiguous")
        reason = row.get("terminal_reason")
        success = row.get("success")
        if (
            row.get("stage") != stage
            or not isinstance(success, bool)
            or not isinstance(reason, str)
            or not reason
            or not isinstance(row.get("terminated"), bool)
            or not isinstance(row.get("truncated"), bool)
            or row["terminated"] == row["truncated"]
        ):
            raise ValueError("episode ledger terminal evidence is malformed")
        if success:
            if reason != "accepted" or not _physical_gates(row, stage=stage):
                raise ValueError("accepted episode does not meet formal physical gates")
            counts["accepted"] += 1
        else:
            if reason == "accepted":
                raise ValueError("rejected episode cannot use accepted terminal reason")
            counts["rejected"] += 1
        reasons[reason] += 1
        rows.append(row)
    if document.get("acceptance_counts") != counts:
        raise ValueError("checkpoint acceptance counts do not match episode ledger")
    if document.get("terminal_reason_counts") != dict(sorted(reasons.items())):
        raise ValueError("checkpoint terminal reasons do not match episode ledger")
    return rows


def _validate_approval_lineage(
    lineage: object, *, config_path: Path
) -> dict[str, object]:
    from scripts import m1_wire_ppo_train as train

    if not isinstance(lineage, dict):
        raise ValueError("action checkpoint has no approval lineage")
    checkpoint_value = lineage.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError("action checkpoint approval lineage is malformed")
    replayed = train.validate_approval_resume(
        Path(checkpoint_value), config_path=Path(config_path).resolve()
    )
    if lineage != replayed:
        raise ValueError("action checkpoint approval lineage does not replay exactly")
    return replayed


def validate_evaluation_checkpoint(
    checkpoint: Path, *, stage: str, config_path: Path = DEFAULT_CONFIG
) -> dict[str, object]:
    from scripts import m1_wire_ppo_train as train

    if stage not in {"approval", "action"}:
        raise ValueError("stage must be approval or action")
    checkpoint = Path(checkpoint).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    manifest_path = checkpoint.parent / "training.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_path}")
    document = _load_sha_manifest(manifest_path, label="training manifest")
    if (
        document.get("schema_version") != 1
        or document.get("status") != "completed"
        or document.get("stage") != stage
        or document.get("action_dim") != 15
    ):
        raise ValueError("checkpoint training manifest is incomplete or incompatible")
    train._require_compatible_sha(
        document,
        label="source",
        path_key="source",
        sha_key="source_sha256",
        expected_path=TRAIN_SCRIPT,
    )
    train._require_compatible_sha(
        document,
        label="config",
        path_key="config",
        sha_key="config_sha256",
        expected_path=config_path,
    )
    rows = _load_training_ledger(document, manifest_path, stage=stage)
    best = document.get("best_checkpoint")
    if not isinstance(best, dict):
        raise ValueError("checkpoint manifest has no best checkpoint")
    relative = best.get("path")
    declared_sha = best.get("sha256")
    if not isinstance(relative, str) or not isinstance(declared_sha, str):
        raise ValueError("best checkpoint record is malformed")
    resolved = (manifest_path.parent / relative).resolve()
    actual_sha = sha256_file(checkpoint)
    if resolved != checkpoint or declared_sha != actual_sha:
        raise ValueError("checkpoint path or SHA-256 does not match best checkpoint")
    counts = document.get("acceptance_counts")
    if (
        not isinstance(counts, dict)
        or not isinstance(counts.get("accepted"), int)
        or counts["accepted"] < 1
        or not any(row.get("success") is True for row in rows)
    ):
        raise ValueError("checkpoint has no accepted training episode")
    approval_lineage = None
    if stage == "action":
        approval_lineage = _validate_approval_lineage(
            document.get("approval_lineage"), config_path=config_path
        )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "training_manifest": str(manifest_path.resolve()),
        "training_manifest_sha256": sha256_file(manifest_path),
        "training_episodes_jsonl_sha256": str(document["episodes_jsonl_sha256"]),
        "approval_lineage": approval_lineage,
    }


def _target_record(info: dict[str, object]) -> dict[str, object]:
    target = info.get("action_target")
    if not isinstance(target, dict):
        return {
            "reached": False,
            "wire_centerline_error": None,
            "tangent_error": None,
            "deformation_error": None,
            "synchronization_error": None,
            "force_balance_error": None,
        }
    result: dict[str, object] = {"reached": target.get("reached") is True}
    for name in (
        "wire_centerline_error",
        "tangent_error",
        "deformation_error",
        "synchronization_error",
        "force_balance_error",
    ):
        value = target.get(name)
        result[name] = None if value is None else _finite(value, label=name)
    return result


def build_episode_record(
    info: dict[str, object], *, episode_index: int, stage: str, checkpoint_sha256: str,
    reward_cfg: object | None = None,
) -> dict[str, object]:
    from scripts import m1_wire_ppo_train as train

    if not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    if _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError("checkpoint_sha256 must be lowercase SHA-256")
    row = deepcopy(info)
    row["stage"] = stage
    row["contact_forces_left"] = list(_forces(row, "left"))
    row["contact_forces_right"] = list(_forces(row, "right"))
    row["action_target"] = _target_record(row)
    contacts_left = row.get("contacts_left")
    contacts_right = row.get("contacts_right")
    bilateral = bool(
        isinstance(contacts_left, int)
        and not isinstance(contacts_left, bool)
        and isinstance(contacts_right, int)
        and not isinstance(contacts_right, bool)
        and contacts_left >= 2
        and contacts_right >= 2
        and len(row["contact_forces_left"]) >= contacts_left
        and len(row["contact_forces_right"]) >= contacts_right
        and all(force > 0.2 for force in row["contact_forces_left"][:contacts_left])
        and all(force > 0.2 for force in row["contact_forces_right"][:contacts_right])
    )
    try:
        lift_reached = _finite(row.get("lift_height"), label="lift height") >= 0.05
        hold_reached = _finite(row.get("hold_time"), label="hold time") >= 1.0
    except ValueError:
        lift_reached = hold_reached = False
    requested_success = row.get("success") is True
    if stage == "action" and reward_cfg is None:
        reward_cfg = FORMAL_ACTION_TARGET_CFG
    row["success"] = bool(
        requested_success
        and _physical_gates(row, stage=stage)
        and (stage == "approval" or _action_target_gate(row, reward_cfg=reward_cfg))
    )
    reason = "accepted" if row["success"] else str(
        row.get("failure_reason")
        or ("truncated" if row.get("truncated") else "formal_acceptance_rejected")
    )
    return {
        "schema_version": 1,
        "episode_index": episode_index,
        "stage": stage,
        "seed": row.get("seed"),
        "checkpoint_sha256": checkpoint_sha256,
        "success": row["success"],
        "terminal_reason": reason,
        "terminated": row.get("terminated") is True,
        "truncated": row.get("truncated") is True,
        "safety_failure": row.get("safety_failure") is True,
        "bilateral_contact": bilateral,
        "contacts_left": contacts_left,
        "contacts_right": contacts_right,
        "contact_forces_left": row["contact_forces_left"],
        "contact_forces_right": row["contact_forces_right"],
        "lift_reached": lift_reached,
        "lift_height": row.get("lift_height"),
        "hold_reached": hold_reached,
        "hold_time": row.get("hold_time"),
        "slip": row.get("slip"),
        "action_target": row["action_target"],
        "reward": row.get("reward"),
        "phase": row.get("phase"),
        "diagnostics": train._diagnostics_summary(row),
    }


def write_episode_ledger(path: Path, rows: Iterable[dict[str, object]]) -> None:
    payload = b"".join(_canonical_json(row) for row in rows)
    _atomic_bytes(Path(path), payload)


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _resolve_pinned_file(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _pinned_training_context(
    checkpoint_lineage: dict[str, object], *, checkpoint: Path, stage: str
) -> tuple[dict[str, object], object]:
    """Validate a checkpoint lineage and reconstruct its reward thresholds.

    This intentionally follows the manifest's absolute config path, verifies
    its on-disk SHA, and then constructs the exact reward dataclass used by
    training.  Nothing in this path falls back to current Python defaults.
    """
    from scripts import m1_wire_ppo_train as train

    expected_keys = {
        "checkpoint", "checkpoint_sha256", "training_manifest",
        "training_manifest_sha256", "training_episodes_jsonl_sha256",
        "approval_lineage",
    }
    if not isinstance(checkpoint_lineage, dict) or set(checkpoint_lineage) != expected_keys:
        raise ValueError("checkpoint_lineage structure is incomplete or has unexpected fields")
    checkpoint = Path(checkpoint).resolve()
    if checkpoint_lineage.get("checkpoint") != str(checkpoint):
        raise ValueError("checkpoint lineage path does not match aggregate checkpoint")
    checkpoint_sha = _require_sha(checkpoint_lineage.get("checkpoint_sha256"), label="checkpoint lineage SHA-256")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint lineage SHA-256 does not match checkpoint")
    manifest_path = _resolve_pinned_file(
        checkpoint_lineage.get("training_manifest"), label="training manifest"
    )
    manifest_sha = _require_sha(
        checkpoint_lineage.get("training_manifest_sha256"),
        label="training manifest SHA-256",
    )
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("training manifest SHA-256 does not match its path")
    training = _load_sha_manifest(manifest_path, label="training manifest")
    if (
        training.get("schema_version") != 1
        or training.get("status") != "completed"
        or training.get("stage") != stage
        or training.get("action_dim") != 15
    ):
        raise ValueError("training manifest is incomplete or incompatible with aggregate stage")
    train._require_compatible_sha(
        training,
        label="source",
        path_key="source",
        sha_key="source_sha256",
        expected_path=TRAIN_SCRIPT,
    )
    config_path = _resolve_pinned_file(training.get("config"), label="training config")
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("training config path is not the formal wire PPO config")
    train._require_compatible_sha(
        training,
        label="config",
        path_key="config",
        sha_key="config_sha256",
        expected_path=config_path,
    )
    best = training.get("best_checkpoint")
    if not isinstance(best, dict):
        raise ValueError("training manifest has no best checkpoint")
    best_path = best.get("path")
    best_sha = _require_sha(best.get("sha256"), label="training best checkpoint SHA-256")
    if not isinstance(best_path, str) or (manifest_path.parent / best_path).resolve() != checkpoint:
        raise ValueError("training best checkpoint path does not match aggregate checkpoint")
    if best_sha != checkpoint_sha:
        raise ValueError("training best checkpoint SHA-256 does not match aggregate checkpoint")
    training_ledger = _load_training_ledger(training, manifest_path, stage=stage)
    training_ledger_sha = _require_sha(
        training.get("episodes_jsonl_sha256"), label="training episode ledger SHA-256"
    )
    if training_ledger_sha != checkpoint_lineage.get("training_episodes_jsonl_sha256"):
        raise ValueError("checkpoint lineage episode ledger SHA-256 mismatch")
    if not any(row.get("success") is True for row in training_ledger):
        raise ValueError("training manifest has no accepted episode")
    config = train.load_train_config(config_path)
    reward_cfg = train._reward_cfg(config, stage)
    if stage == "action":
        for field, expected in FORMAL_ACTION_TARGET_LIMITS.items():
            try:
                actual = float(getattr(reward_cfg, field))
            except (AttributeError, TypeError, ValueError):
                raise ValueError(f"action target threshold {field} is not pinned") from None
            if actual != float(expected):
                raise ValueError(
                    f"action target threshold {field} does not match formal wire PPO config"
                )
    if stage == "action":
        expected_approval = checkpoint_lineage.get("approval_lineage")
        if not isinstance(expected_approval, dict):
            raise ValueError("action checkpoint lineage has no approval lineage")
        if training.get("approval_lineage") != expected_approval:
            raise ValueError("action training manifest approval lineage does not match checkpoint lineage")
        replayed = train.validate_approval_resume(checkpoint_lineage["approval_lineage"]["checkpoint"], config_path=config_path)
        if expected_approval != replayed:
            raise ValueError("action checkpoint approval lineage does not replay exactly")
    elif checkpoint_lineage.get("approval_lineage") is not None:
        raise ValueError("approval checkpoint lineage must not contain approval lineage")
    return training, reward_cfg


def publish_aggregate_manifest(
    path: Path,
    *,
    stage: str,
    seed: int,
    requested_episodes: int,
    checkpoint: Path,
    checkpoint_lineage: dict[str, object],
    ledger: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    _, reward_cfg = _pinned_training_context(
        checkpoint_lineage, checkpoint=checkpoint, stage=stage
    )
    accepted = len(successful_episode_rows(rows, stage=stage, reward_cfg=reward_cfg))
    body = {
        "schema_version": 1,
        "status": "completed",
        "stage": stage,
        "seed": int(seed),
        "requested_episodes": int(requested_episodes),
        "evaluated_episodes": len(rows),
        "accepted_episodes": accepted,
        "rejected_episodes": len(rows) - accepted,
        "formal_passed": accepted > 0,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_lineage": deepcopy(checkpoint_lineage),
        "episodes_jsonl": Path(ledger).name,
        "episodes_jsonl_sha256": sha256_file(ledger),
        "episode_outcomes": {
            "bilateral_contact": sum(row.get("bilateral_contact") is True for row in rows),
            "lift": sum(row.get("lift_reached") is True for row in rows),
            "hold": sum(row.get("hold_reached") is True for row in rows),
            "action_target": sum(
                stage == "action"
                and isinstance(row.get("action_target"), dict)
                and row["action_target"].get("reached") is True
                for row in rows
            ),
        },
        "formal_thresholds": {
            "minimum_contacts_per_hand": 2,
            "contact_force_strictly_greater_than_n": 0.2,
            "minimum_lift_m": 0.05,
            "minimum_hold_s": 1.0,
            "maximum_slip_m": 0.02,
        },
        "source": str(THIS_FILE),
        "source_sha256": sha256_file(THIS_FILE),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return write_sha_manifest(path, body)


def verify_aggregate_manifest(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    document = _load_sha_manifest(path, label="aggregate manifest")
    relative = document.get("episodes_jsonl")
    declared_sha = document.get("episodes_jsonl_sha256")
    if not isinstance(relative, str):
        raise ValueError("aggregate manifest has no episode ledger")
    ledger = (path.parent / relative).resolve()
    if ledger.parent != path.parent or not ledger.is_file():
        raise ValueError("aggregate episode ledger must be inside its output directory")
    if not isinstance(declared_sha, str) or sha256_file(ledger) != declared_sha:
        raise ValueError("episode ledger SHA-256 mismatch")
    checkpoint = document.get("checkpoint")
    checkpoint_sha = document.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint, str)
        or not Path(checkpoint).is_file()
        or not isinstance(checkpoint_sha, str)
        or sha256_file(Path(checkpoint)) != checkpoint_sha
    ):
        raise ValueError("checkpoint SHA-256 mismatch")
    checkpoint_path = Path(checkpoint).resolve()
    stage = document.get("stage")
    requested = document.get("requested_episodes")
    if (
        stage not in {"approval", "action"}
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= 0
    ):
        raise ValueError("aggregate stage or requested episode count is invalid")
    training_manifest, reward_cfg = _pinned_training_context(
        document.get("checkpoint_lineage"), checkpoint=checkpoint_path, stage=stage
    )
    rows: list[dict[str, object]] = []
    for index, line in enumerate(ledger.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"aggregate episode row {index} is invalid JSON") from error
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != 1
            or row.get("episode_index") != index
            or row.get("stage") != stage
            or row.get("checkpoint_sha256") != checkpoint_sha
            or not isinstance(row.get("success"), bool)
        ):
            raise ValueError("aggregate episode ledger is malformed")
        rows.append(row)
    accepted = len(successful_episode_rows(rows, stage=stage, reward_cfg=reward_cfg))
    outcomes = {
        "bilateral_contact": sum(row.get("bilateral_contact") is True for row in rows),
        "lift": sum(row.get("lift_reached") is True for row in rows),
        "hold": sum(row.get("hold_reached") is True for row in rows),
        "action_target": sum(
            stage == "action"
            and isinstance(row.get("action_target"), dict)
            and row["action_target"].get("reached") is True
            for row in rows
        ),
    }
    if (
        len(rows) != requested
        or document.get("evaluated_episodes") != len(rows)
        or document.get("accepted_episodes") != accepted
        or document.get("rejected_episodes") != len(rows) - accepted
        or document.get("formal_passed") is not (accepted > 0)
        or document.get("episode_outcomes") != outcomes
    ):
        raise ValueError("aggregate counts do not match episode ledger")
    return document


def _state_target(state: object, *, stage: str, reward_cfg: object) -> dict[str, object]:
    from go2_pvcnn.tasks.m1_wire_ppo_curriculum import action_episode

    return {
        "reached": bool(stage == "action" and action_episode(state, reward_cfg)),
        "wire_centerline_error": float(state.wire_centerline_error),
        "tangent_error": float(state.tangent_error),
        "deformation_error": float(state.deformation_error),
        "synchronization_error": float(state.synchronization_error),
        "force_balance_error": float(state.force_balance_error),
    }


class _EvaluationRecorder:
    """Satisfy the RSL-RL adapter without creating an unpinned second ledger."""

    def __init__(self) -> None:
        self.acceptance_counts = {"accepted": 0, "rejected": 0}

    def record(self, _info: dict[str, object]) -> None:
        raise RuntimeError("evaluation steps must run through M1WirePpoEnv directly")


def main() -> int:
    configure_runtime_environment()
    args = build_arg_parser(include_app_launcher_args=True).parse_args()
    validate_cli_contract(args)
    config_path = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_lineage = validate_evaluation_checkpoint(
        checkpoint, stage=args.stage, config_path=config_path
    )
    output_dir = (
        ROOT / "artifacts" / f"m1_wire_ppo_{args.stage}_verify_seed{args.seed}"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    app = None
    physical_env = None
    wrapped_env = None
    rows: list[dict[str, object]] = []
    try:
        import torch
        from isaaclab.app import AppLauncher

        from scripts import m1_wire_ppo_train as train

        train.configure_torch_threads(torch)
        train._seed_everything(args.seed, torch)
        app = AppLauncher(args).app
        import gymnasium as gym
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper import (
            M1DualPandaO6BimanualWrapper,
        )
        from go2_pvcnn.tasks.m1_wire_ppo_curriculum import WirePpoStage
        from go2_pvcnn.tasks.m1_wire_ppo_env import M1WirePpoEnv
        from rsl_rl.runners import OnPolicyRunner
        from scripts import m1_dual_panda_o6_bimanual_play as play

        config = train.load_train_config(config_path)
        catalog, instances = play._prepare_object_scene(
            object_id=args.object_id,
            object_catalog=args.object_catalog,
            object_assets_root=args.object_assets_root,
        )
        cfg = play.build_play_config(
            seed=args.seed,
            object_id=args.object_id,
            catalog=catalog,
            object_instances=instances,
        )
        physical_env = gym.make(config["task_id"], cfg=cfg)
        wrapper = M1DualPandaO6BimanualWrapper(
            physical_env, mode="teacher", object_id=args.object_id
        )
        wrapper.runtime.wire_object_id = args.object_id
        wrapper.runtime.wire_target_prim_path = cfg.target_contact_prim_path.replace(
            "{ENV_REGEX_NS}", "/World/envs/env_0"
        )
        play._configure_reduced_wire_runtime(wrapper)
        reward_cfg = train._reward_cfg(config, args.stage)
        ppo_env = M1WirePpoEnv(
            wrapper,
            stage=WirePpoStage(args.stage),
            reward_cfg=reward_cfg,
            max_episode_steps=int(config["max_episode_steps"]),
            device=args.device,
        )
        wrapped_env = train.WireRslRlEnv(
            ppo_env, _EvaluationRecorder(), seed=args.seed, torch_module=torch
        )
        runner = OnPolicyRunner(
            wrapped_env, deepcopy(config["runner"]), log_dir=None, device=args.device
        )
        runner.load(str(checkpoint), load_optimizer=False, keep_std=True)
        policy = runner.get_inference_policy(device=args.device)
        checkpoint_sha = sha256_file(checkpoint)

        with torch.inference_mode():
            for episode_index in range(args.episodes):
                observation, _ = ppo_env.reset(seed=args.seed + episode_index)
                reward_sum = 0.0
                while True:
                    action = policy(observation.reshape(1, -1))[0]
                    observation, reward, terminated, truncated, info = ppo_env.step(action)
                    reward_sum += float(reward)
                    if terminated or truncated:
                        state = ppo_env._previous_state
                        terminal = dict(info)
                        terminal.update(
                            {
                                "seed": args.seed + episode_index,
                                "reward": reward_sum,
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                                "contact_forces_left": list(state.contact_forces_left),
                                "contact_forces_right": list(state.contact_forces_right),
                                "safety_failure": bool(state.safety_failure),
                                "action_target": _state_target(
                                    state, stage=args.stage, reward_cfg=reward_cfg
                                ),
                            }
                        )
                        rows.append(
                            build_episode_record(
                                terminal,
                                episode_index=episode_index,
                                stage=args.stage,
                                checkpoint_sha256=checkpoint_sha,
                                reward_cfg=reward_cfg,
                            )
                        )
                        break

        ledger = output_dir / "episodes.jsonl"
        write_episode_ledger(ledger, rows)
        manifest_path = output_dir / "aggregate.manifest.json"
        publish_aggregate_manifest(
            manifest_path,
            stage=args.stage,
            seed=args.seed,
            requested_episodes=args.episodes,
            checkpoint=checkpoint,
            checkpoint_lineage=checkpoint_lineage,
            ledger=ledger,
            rows=rows,
        )
        verify_aggregate_manifest(manifest_path)
        return 0
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        elif physical_env is not None:
            physical_env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
