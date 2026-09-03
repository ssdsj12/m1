"""Train and serialize the deterministic M1 bimanual z16 action model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn

from go2_pvcnn.control.m1_bimanual_coordination.latent_contracts import (
    ACTION_DIM,
    FEATURE_ORDER,
    HORIZON,
    LATENT_DIM,
    MODEL_FORMAT_VERSION,
    STATE_DIM,
    TASK_FEATURE_DIM,
    LatentArtifactMetadata,
    LatentNormalizer,
)
from go2_pvcnn.control.m1_bimanual_coordination.latent_model import LatentActionModel


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[object, ...]
    validation: tuple[object, ...]
    test: tuple[object, ...]


def deterministic_split(groups: Sequence[object], *, seed: int) -> DatasetSplit:
    unique = list(dict.fromkeys(groups))
    generator = random.Random(int(seed))
    generator.shuffle(unique)
    train_end = int(0.8 * len(unique))
    validation_end = train_end + int(0.1 * len(unique))
    return DatasetSplit(
        train=tuple(unique[:train_end]),
        validation=tuple(unique[train_end:validation_end]),
        test=tuple(unique[validation_end:]),
    )


def composite_loss(
    batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    target_action = batch["teacher_action"]
    target_task = batch["teacher_task"]
    decoded_action = outputs["decoded_action"]
    decoded_task = outputs["decoded_task"]
    latent = outputs["latent"]
    body_action = outputs["body_action"]
    losses = {
        "action": nn.functional.mse_loss(decoded_action, target_action),
        "box_effect": nn.functional.mse_loss(decoded_task[:, :, :12], target_task[:, :, :12]),
        "palm_effect": nn.functional.mse_loss(decoded_task[:, :, 12:36], target_task[:, :, 12:36]),
        "contact": nn.functional.mse_loss(decoded_task[:, :, 36:48], target_task[:, :, 36:48]),
        "base": nn.functional.mse_loss(body_action[:, :12], target_action[:, 0, :12]),
        "effort_smoothness": torch.mean((decoded_action[:, 1:] - decoded_action[:, :-1]) ** 2),
        "latent_smoothness": torch.mean((latent[1:] - latent[:-1]) ** 2) if latent.shape[0] > 1 else torch.zeros((), dtype=latent.dtype, device=latent.device),
        "safety": torch.mean(body_action[:, 12:16] ** 2)
        + torch.mean(torch.relu(torch.abs(body_action) - 1.0) ** 2),
    }
    weights = {
        "action": 1.0,
        "box_effect": 2.0,
        "palm_effect": 2.0,
        "contact": 1.0,
        "base": 2.0,
        "effort_smoothness": 0.1,
        "latent_smoothness": 0.05,
        "safety": 5.0,
    }
    losses["total"] = sum(weights[name] * losses[name] for name in weights)
    return losses


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_dataset(output_dir: Path, seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    count = 100
    state = generator.normal(0.0, 0.25, (count, STATE_DIM)).astype(np.float32)
    teacher_action = generator.normal(0.0, 0.1, (count, HORIZON, ACTION_DIM)).astype(np.float32)
    teacher_action[:, :, 12:16] = 0.0
    teacher_task = generator.normal(0.0, 0.1, (count, HORIZON, TASK_FEATURE_DIM)).astype(np.float32)
    seeds = np.repeat(np.arange(10, dtype=np.int64), 10)
    trials = np.tile(np.arange(10, dtype=np.int64), 10)
    data = {
        "state": state,
        "teacher_action": teacher_action,
        "teacher_task": teacher_task,
        "phase": np.zeros((count, 1), dtype=np.float32),
        "seed": seeds,
        "trial": trials,
        "action_order": np.asarray([f"joint_{index}" for index in range(ACTION_DIM)]),
    }
    np.savez_compressed(output_dir / "teacher_success.npz", **data)
    np.savez_compressed(
        output_dir / "teacher_failures.npz",
        state=state[:4],
        reason=np.asarray(["synthetic_failure"] * 4),
        seed=seeds[:4],
        trial=trials[:4],
    )
    return data


def _load_dataset(dataset_dir: Path) -> dict[str, np.ndarray]:
    path = dataset_dir / "teacher_success.npz"
    if not path.is_file():
        raise FileNotFoundError(f"teacher dataset not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _indices(data: dict[str, np.ndarray], groups: tuple[object, ...]) -> np.ndarray:
    allowed = set(groups)
    return np.asarray(
        [
            index
            for index, key in enumerate(zip(data["seed"].tolist(), data["trial"].tolist()))
            if key in allowed
        ],
        dtype=np.int64,
    )


def _batch(data: dict[str, np.ndarray], indices: np.ndarray, normalizer: LatentNormalizer) -> dict[str, torch.Tensor]:
    state = torch.from_numpy(data["state"][indices]).to(dtype=torch.float32)
    return {
        "state": normalizer.normalize(state),
        "teacher_action": torch.from_numpy(data["teacher_action"][indices]).to(dtype=torch.float32),
        "teacher_task": torch.from_numpy(data["teacher_task"][indices]).to(dtype=torch.float32),
        "phase": torch.from_numpy(data["phase"][indices]).to(dtype=torch.float32),
    }


def _forward(model: LatentActionModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    latent = model.encode(batch["state"], batch["teacher_action"], batch["teacher_task"])
    return {
        "latent": latent,
        "decoded_action": model.decode_trajectory(batch["state"], latent),
        "decoded_task": model.decode_task_trajectory(batch["state"], latent),
        "body_action": model.body_action(
            batch["state"],
            latent,
            batch["phase"],
            torch.zeros((batch["state"].shape[0], ACTION_DIM), dtype=torch.float32),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.synthetic_smoke:
        data = _synthetic_dataset(args.output_dir, args.seed)
        dataset_path = args.output_dir / "teacher_success.npz"
    else:
        if args.dataset_dir is None:
            parser.error("--dataset-dir is required without --synthetic-smoke")
        data = _load_dataset(args.dataset_dir)
        dataset_path = args.dataset_dir / "teacher_success.npz"
        failure_source = args.dataset_dir / "teacher_failures.npz"
        if failure_source.is_file():
            with np.load(failure_source, allow_pickle=False) as archive:
                np.savez_compressed(
                    args.output_dir / "teacher_failures.npz",
                    **{name: archive[name] for name in archive.files},
                )
        with np.load(dataset_path, allow_pickle=False) as archive:
            np.savez_compressed(
                args.output_dir / "teacher_success.npz",
                **{name: archive[name] for name in archive.files},
            )
    groups = list(zip(data["seed"].tolist(), data["trial"].tolist()))
    split = deterministic_split(groups, seed=args.seed)
    split_json = {name: [list(value) for value in getattr(split, name)] for name in ("train", "validation", "test")}
    (args.output_dir / "split.json").write_text(json.dumps(split_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    train_indices = _indices(data, split.train)
    validation_indices = _indices(data, split.validation)
    test_indices = _indices(data, split.test)
    train_state = torch.from_numpy(data["state"][train_indices]).to(dtype=torch.float32)
    scale = train_state.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    normalizer = LatentNormalizer(mean=train_state.mean(dim=0), scale=scale)
    normalization_path = args.output_dir / "normalization.pt"
    torch.save({"mean": normalizer.mean, "scale": normalizer.scale}, normalization_path)
    batches = {
        "train": _batch(data, train_indices, normalizer),
        "validation": _batch(data, validation_indices, normalizer),
        "test": _batch(data, test_indices, normalizer),
    }
    model = LatentActionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for _epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = composite_loss(batches["train"], _forward(model, batches["train"]))
        losses["total"].backward()
        optimizer.step()
    model.eval()
    metrics: dict[str, float] = {}
    with torch.no_grad():
        for name, batch in batches.items():
            losses = composite_loss(batch, _forward(model, batch))
            metrics[f"{name}_total"] = float(losses["total"].item())
            metrics[f"{name}_action"] = float(losses["action"].item())
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("training produced non-finite metrics")
    model_path = args.output_dir / "latent_action_model.pt"
    torch.save(model.state_dict(), model_path)
    action_order = tuple(str(value) for value in data["action_order"].tolist())
    metadata = LatentArtifactMetadata(
        format_version=MODEL_FORMAT_VERSION,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        horizon=HORIZON,
        task_feature_dim=TASK_FEATURE_DIM,
        latent_dim=LATENT_DIM,
        action_order=action_order,
        feature_order=FEATURE_ORDER,
        normalization_sha256=_sha256(normalization_path),
        dataset_sha256=_sha256(dataset_path),
        training_seed=args.seed,
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
