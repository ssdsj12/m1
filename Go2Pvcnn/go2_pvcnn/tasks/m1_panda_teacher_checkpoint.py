"""Strict checkpoint and manifest contracts for M1 + Panda Teacher stages."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from go2_pvcnn.tasks.m1_panda_teacher import (
    M1PandaDisturbanceCfg,
    stage_disturbance_cfg,
)
from go2_pvcnn.tasks.m1_residual_action import M1ResidualActionComposerCfg


MANIFEST_FILENAME = "run_manifest.json"
TEACHER_SCHEMA_VERSION = 1
TEACHER_OBSERVATION_DIM = 60
TEACHER_ACTION_DIM = 16
TEACHER_HIDDEN_DIMS = (256, 128)


def _require_file(path: str | os.PathLike[str], *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return resolved


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of an existing checkpoint file."""
    resolved = _require_file(path, label="checkpoint")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_manifest(
    path: str | os.PathLike[str], payload: dict[str, object]
) -> None:
    """Atomically publish a JSON manifest in its target directory."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_manifest_for_checkpoint(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Load the run manifest adjacent to an existing checkpoint."""
    checkpoint = _require_file(checkpoint_path, label="checkpoint")
    manifest_path = checkpoint.parent / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {MANIFEST_FILENAME}: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return payload


def module_sha256(module: torch.nn.Module) -> str:
    """Hash a module state deterministically across devices."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state entry {name!r} must be a torch.Tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    return payload


def _expected_state_shapes(
    observation_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, int],
) -> dict[str, tuple[int, ...]]:
    first, second = hidden_dims
    return {
        "actor.0.weight": (first, observation_dim),
        "actor.0.bias": (first,),
        "actor.2.weight": (second, first),
        "actor.2.bias": (second,),
        "actor.4.weight": (action_dim, second),
        "actor.4.bias": (action_dim,),
        "critic.0.weight": (first, observation_dim),
        "critic.0.bias": (first,),
        "critic.2.weight": (second, first),
        "critic.2.bias": (second,),
        "critic.4.weight": (1, second),
        "critic.4.bias": (1,),
        "std": (action_dim,),
    }


def validate_teacher_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_stage: str,
    expected_observation_dim: int,
    expected_action_dim: int,
    expected_actor_hidden_dims: tuple[int, int],
    expected_base_sha256: str | None = None,
    require_optimizer: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate manifest metadata and actual RSL ActorCritic tensor shapes."""
    checkpoint_path = _require_file(path, label="checkpoint")
    manifest = load_manifest_for_checkpoint(checkpoint_path)
    expected_manifest = {
        "schema_version": TEACHER_SCHEMA_VERSION,
        "stage": expected_stage,
        "observation_dim": expected_observation_dim,
        "action_dim": expected_action_dim,
        "actor_hidden_dims": list(expected_actor_hidden_dims),
    }
    for field, expected in expected_manifest.items():
        actual = manifest.get(field)
        if actual != expected:
            raise ValueError(
                f"manifest {field} mismatch: expected {expected!r}, got {actual!r}"
            )
    if expected_base_sha256 is not None:
        actual_base_sha256 = manifest.get("base_checkpoint_sha256")
        if actual_base_sha256 != expected_base_sha256:
            raise ValueError(
                "manifest base_checkpoint_sha256 mismatch: "
                f"expected {expected_base_sha256!r}, got {actual_base_sha256!r}"
            )

    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint model_state_dict must be a dictionary")
    if require_optimizer and not isinstance(
        checkpoint.get("optimizer_state_dict"), dict
    ):
        raise ValueError(
            "checkpoint optimizer_state_dict must be present for resume validation"
        )

    if (
        len(expected_actor_hidden_dims) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in expected_actor_hidden_dims
        )
    ):
        raise ValueError("expected_actor_hidden_dims must contain two positive integers")
    expected_shapes = _expected_state_shapes(
        expected_observation_dim,
        expected_action_dim,
        expected_actor_hidden_dims,
    )
    for name, expected_shape in expected_shapes.items():
        value = state_dict.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"model_state_dict is missing tensor {name}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"model_state_dict {name} shape mismatch: "
                f"expected {expected_shape}, got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"model_state_dict {name} contains non-finite values")
    return checkpoint, manifest


def load_frozen_teacher_actor(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device,
    policy_cfg: dict[str, object],
) -> torch.nn.Module:
    """Strictly load and freeze the exact 60-to-16 A0 ActorCritic."""
    if not isinstance(policy_cfg, dict):
        raise TypeError("policy_cfg must be a dictionary")
    required_policy = {
        "class_name": "ActorCritic",
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
        "activation": "elu",
    }
    for field, expected in required_policy.items():
        actual = policy_cfg.get(field)
        if actual != expected:
            raise ValueError(
                f"policy_cfg {field} mismatch: expected {expected!r}, got {actual!r}"
            )

    checkpoint, _ = validate_teacher_checkpoint(
        path,
        expected_stage="A0",
        expected_observation_dim=TEACHER_OBSERVATION_DIM,
        expected_action_dim=TEACHER_ACTION_DIM,
        expected_actor_hidden_dims=TEACHER_HIDDEN_DIMS,
    )
    from rsl_rl.modules import ActorCritic

    constructor_cfg = dict(policy_cfg)
    constructor_cfg.pop("class_name", None)
    constructor_cfg.pop("noise_std_type", None)
    constructor_cfg.pop("state_dependent_std", None)
    actor = ActorCritic(
        TEACHER_OBSERVATION_DIM,
        TEACHER_OBSERVATION_DIM,
        TEACHER_ACTION_DIM,
        **constructor_cfg,
    ).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor


def _json_compatible_dataclass(value: object) -> dict[str, object]:
    payload = asdict(value)
    return json.loads(json.dumps(payload))


def build_run_manifest(
    *,
    stage: str,
    task_id: str,
    seed: int,
    composer_cfg: M1ResidualActionComposerCfg,
    disturbance_cfg: M1PandaDisturbanceCfg,
    base_checkpoint: str | os.PathLike[str] | None = None,
    frozen_actor: torch.nn.Module | None = None,
    resume_checkpoint: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Build the complete JSON-compatible start manifest for one run."""
    expected_task_id = f"Isaac-M1-Panda-Teacher-{stage}-v0"
    if stage not in {"A0", "A1"}:
        raise ValueError(f"stage must be 'A0' or 'A1', got {stage!r}")
    if task_id != expected_task_id:
        raise ValueError(
            f"task_id mismatch for {stage}: expected {expected_task_id!r}, got {task_id!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(composer_cfg, M1ResidualActionComposerCfg):
        raise TypeError("composer_cfg must be an M1ResidualActionComposerCfg")
    if not isinstance(disturbance_cfg, M1PandaDisturbanceCfg):
        raise TypeError("disturbance_cfg must be an M1PandaDisturbanceCfg")
    if disturbance_cfg != stage_disturbance_cfg(stage):
        raise ValueError(f"disturbance_cfg does not match the approved {stage} defaults")

    base_path: Path | None = None
    base_hash: str | None = None
    frozen_hash: str | None = None
    if stage == "A0":
        if base_checkpoint is not None or frozen_actor is not None:
            raise ValueError("A0 does not accept a base checkpoint or frozen actor")
    else:
        if base_checkpoint is None or frozen_actor is None:
            raise ValueError("A1 requires a base checkpoint and frozen actor")
        base_path = _require_file(base_checkpoint, label="base checkpoint")
        base_hash = file_sha256(base_path)
        frozen_hash = module_sha256(frozen_actor)

    resume_path: Path | None = None
    if resume_checkpoint is not None:
        resume_path = _require_file(resume_checkpoint, label="resume checkpoint")

    return {
        "schema_version": TEACHER_SCHEMA_VERSION,
        "stage": stage,
        "task_id": task_id,
        "observation_dim": TEACHER_OBSERVATION_DIM,
        "action_dim": TEACHER_ACTION_DIM,
        "actor_hidden_dims": list(TEACHER_HIDDEN_DIMS),
        "seed": seed,
        "composer": _json_compatible_dataclass(composer_cfg),
        "disturbance": _json_compatible_dataclass(disturbance_cfg),
        "checkpoint_pattern": "model_<iteration>.pt",
        "base_checkpoint": str(base_path) if base_path is not None else None,
        "base_checkpoint_sha256": base_hash,
        "frozen_actor_initial_sha256": frozen_hash,
        "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        "status": "running",
    }


__all__ = [
    "MANIFEST_FILENAME",
    "TEACHER_SCHEMA_VERSION",
    "TEACHER_OBSERVATION_DIM",
    "TEACHER_ACTION_DIM",
    "TEACHER_HIDDEN_DIMS",
    "atomic_write_manifest",
    "build_run_manifest",
    "file_sha256",
    "load_frozen_teacher_actor",
    "load_manifest_for_checkpoint",
    "module_sha256",
    "validate_teacher_checkpoint",
]
