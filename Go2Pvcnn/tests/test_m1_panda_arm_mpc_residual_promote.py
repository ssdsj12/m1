from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from go2_pvcnn.training.m1_panda_arm_mpc_residual_guard import ResidualEvalMetrics


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_arm_mpc_residual_promote.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("residual_promote_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(**overrides):
    values = dict(
        hard_failure_count=0,
        mpc_feasible_rate=1.0,
        qp_feasible_rate=1.0,
        four_contact_rate=1.0,
        roll_pitch_rms=0.001,
        base_height_rms=0.0002,
        ee_position_error=0.01,
        ee_orientation_error=0.04,
        wrench_error=80.0,
        slip=0.002,
        intervention_ratio=0.0,
        saturation_fraction=(0.0,) * 8,
    )
    values.update(overrides)
    return ResidualEvalMetrics(**values)


def _safe_run(tmp_path, module):
    asset = tmp_path / "robot.usd"
    config = tmp_path / "train_cfg.py"
    reward = tmp_path / "reward.py"
    for path, payload in ((asset, b"asset"), (config, b"config"), (reward, b"reward")):
        path.write_bytes(payload)
    candidates = []
    for updates in (0, 25, 50, 75, 100):
        checkpoint = tmp_path / f"candidate_u{updates:03d}.pt"
        checkpoint.write_bytes(f"candidate-{updates}".encode())
        candidates.append(
            {
                "completed_updates": updates,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": module.sha256_file(checkpoint),
            }
        )
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "short",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": True,
                "asset_path": str(asset),
                "asset_sha256": module.sha256_file(asset),
                "config_path": str(config),
                "config_sha256": module.sha256_file(config),
                "reward_path": str(reward),
                "reward_sha256": module.sha256_file(reward),
                "candidate_checkpoints": candidates,
            }
        ),
        encoding="utf-8",
    )
    return manifest


class RecordingWorkerRunner:
    def __init__(self, module, *, checkpoint_sha=None, improve=False):
        self.module = module
        self.checkpoint_sha = checkpoint_sha
        self.improve = improve
        self.zero_pair_calls = 0
        self.candidate_calls = 0

    def __call__(self, *, mode, seed, output, checkpoint, device, headless):
        baseline = _metrics()
        if mode == "zero-pair":
            self.zero_pair_calls += 1
            candidate = _metrics()
            checkpoint_value = None
            checkpoint_sha = None
        else:
            self.candidate_calls += 1
            candidate = _metrics(
                ee_position_error=0.009 if self.improve else 0.01
            )
            checkpoint_value = str(checkpoint)
            checkpoint_sha = (
                self.checkpoint_sha
                if self.checkpoint_sha is not None
                else self.module.sha256_file(checkpoint)
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "mode": mode,
                    "seed": seed,
                    "steps": 4000,
                    "checkpoint": checkpoint_value,
                    "checkpoint_sha256": checkpoint_sha,
                    "baseline": asdict(baseline),
                    "candidate": asdict(candidate),
                }
            ),
            encoding="utf-8",
        )


def test_driver_launches_nine_calibration_and_fifteen_candidate_workers(tmp_path):
    module = _load_script()
    runner = RecordingWorkerRunner(module)

    module.run_promotion(_safe_run(tmp_path, module), worker_runner=runner)

    assert runner.zero_pair_calls == 9
    assert runner.candidate_calls == 15


def test_driver_fails_closed_on_worker_checkpoint_sha_mismatch(tmp_path):
    module = _load_script()
    runner = RecordingWorkerRunner(module, checkpoint_sha="0" * 64)

    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        module.run_promotion(_safe_run(tmp_path, module), worker_runner=runner)


def test_equivalent_candidates_never_publish_best(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)

    result = module.run_promotion(
        manifest_path, worker_runner=RecordingWorkerRunner(module)
    )

    assert not result["accepted"]
    assert not (tmp_path / "model_best.pt").exists()


def test_improved_candidate_publishes_sha_identical_best(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)

    result = module.run_promotion(
        manifest_path,
        worker_runner=RecordingWorkerRunner(module, improve=True),
    )

    assert result["accepted"]
    assert result["best_completed_updates"] == 0
    assert module.sha256_file(tmp_path / "model_best.pt") == result[
        "best_checkpoint_sha256"
    ]
