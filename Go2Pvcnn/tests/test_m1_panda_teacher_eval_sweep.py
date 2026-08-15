from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "m1_panda_teacher_eval_sweep.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_teacher_eval_sweep_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(checkpoint, seed, *, timeout, contact, bad, reward):
    total = timeout + contact + bad
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": f"sha-{Path(checkpoint).name}",
        "base_checkpoint_sha256": "base-sha",
        "frozen_actor_sha256": "frozen-sha",
        "seed": seed,
        "num_envs": 64,
        "steps": 2000,
        "curriculum_scale": 1.0,
        "axis_abs_wrench_seen": [19.5, 19.5, 19.5, 4.8, 4.8, 4.8],
        "termination_counts": {
            "time_out": timeout,
            "base_contact": contact,
            "bad_orientation": bad,
        },
        "termination_rates": {
            "time_out": timeout / total,
            "base_contact": contact / total,
            "bad_orientation": bad / total,
        },
        "reward_sum": reward * 64 * 2000,
        "reward_count": 64 * 2000,
        "mean_reward": reward,
        "finite": True,
    }


def test_rank_completed_rows_selects_lexicographic_winner(tmp_path):
    module = _load_script()
    rows = []
    for checkpoint, timeout, contact, bad, reward in (
        ("/models/model_2700.pt", 90, 5, 5, 3.0),
        ("/models/model_4500.pt", 80, 10, 10, 5.0),
    ):
        for seed in (42, 43, 44):
            path = tmp_path / f"{Path(checkpoint).stem}-{seed}.json"
            path.write_text(
                json.dumps(
                    _summary(
                        checkpoint,
                        seed,
                        timeout=timeout,
                        contact=contact,
                        bad=bad,
                        reward=reward,
                    )
                ),
                encoding="utf-8",
            )
            rows.append(path)

    result = module.rank_completed_rows(rows)

    assert result["winner"]["checkpoint"] == "/models/model_2700.pt"
    assert result["candidates"][0]["rank_key"] >= result["candidates"][1][
        "rank_key"
    ]


def test_validate_sweep_inputs_rejects_duplicates_missing_seeds_and_existing_dir(
    tmp_path,
):
    module = _load_script()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")

    with pytest.raises(ValueError, match="unique"):
        module.validate_sweep_inputs(
            [checkpoint, checkpoint], [42, 43, 44], tmp_path / "new"
        )
    with pytest.raises(ValueError, match="seeds"):
        module.validate_sweep_inputs(
            [checkpoint], [42, 43], tmp_path / "new"
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        module.validate_sweep_inputs(
            [checkpoint], [42, 43, 44], existing
        )


def test_run_play_child_rejects_nonzero_exit_and_missing_row(tmp_path, monkeypatch):
    module = _load_script()
    row = tmp_path / "row.json"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 3, stdout="out", stderr="err"
        ),
    )
    with pytest.raises(RuntimeError, match="exit 3"):
        module.run_play_child(["python", "play.py"], row)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="out", stderr="err"
        ),
    )
    with pytest.raises(FileNotFoundError, match="row"):
        module.run_play_child(["python", "play.py"], row)
