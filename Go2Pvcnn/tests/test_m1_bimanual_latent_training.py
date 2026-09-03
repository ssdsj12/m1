from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.m1_dual_panda_o6_train_latent import (
    composite_loss,
    deterministic_split,
)


def test_split_is_deterministic_and_group_disjoint() -> None:
    groups = [(seed, trial) for seed in range(10) for trial in range(10)]

    first = deterministic_split(groups, seed=42)
    second = deterministic_split(groups, seed=42)

    assert first == second
    assert len(first.train) == 80
    assert len(first.validation) == 10
    assert len(first.test) == 10
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)


def test_composite_loss_contains_every_design_term() -> None:
    batch = {
        "teacher_action": torch.zeros(2, 25, 43),
        "teacher_task": torch.zeros(2, 25, 50),
    }
    outputs = {
        "decoded_action": torch.ones(2, 25, 43),
        "decoded_task": torch.ones(2, 25, 50),
        "body_action": torch.ones(2, 43),
        "latent": torch.ones(2, 16),
    }

    losses = composite_loss(batch, outputs)

    assert set(losses) == {
        "action",
        "box_effect",
        "palm_effect",
        "contact",
        "base",
        "effort_smoothness",
        "latent_smoothness",
        "safety",
        "total",
    }
    assert torch.isfinite(losses["total"])


def test_synthetic_smoke_writes_versioned_artifact(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/m1_dual_panda_o6_train_latent.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--synthetic-smoke",
            "--epochs",
            "1",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in (
        "teacher_success.npz",
        "teacher_failures.npz",
        "split.json",
        "normalization.pt",
        "latent_action_model.pt",
        "metadata.json",
        "metrics.json",
    ):
        assert (tmp_path / name).is_file()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metadata["latent_dim"] == 16
    assert np.isfinite(metrics["validation_total"])

