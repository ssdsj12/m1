from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

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
    runtime = tmp_path / "runtime.py"
    for path, payload in (
        (asset, b"asset"),
        (config, b"config"),
        (reward, b"reward"),
        (runtime, b"runtime"),
    ):
        path.write_bytes(payload)
    lineage = module.source_lineage(
        module.ResidualSourcePaths(asset, config, reward, runtime)
    )
    pilot = tmp_path / "pilot_manifest.json"
    pilot.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "pilot",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": False,
                "pilot_accepted": True,
                "requested_iterations": 10,
                "completed_iterations": 10,
                "optimizer_summaries": [
                    {"update": update} for update in range(1, 11)
                ],
                "pilot_decision": {"accepted": True},
                **lineage,
            }
        ),
        encoding="utf-8",
    )
    candidates = []
    for updates in (0, 25, 50, 75, 100):
        checkpoint = tmp_path / f"candidate_u{updates:03d}.pt"
        torch.save(
            {
                "model_state_dict": {"candidate": torch.tensor(updates)},
                "optimizer_state_dict": {},
                "obs_norm_state_dict": {"_mean": torch.zeros(1, 103)},
                "critic_obs_norm_state_dict": {"_mean": torch.zeros(1, 103)},
                "iter": updates,
            },
            checkpoint,
        )
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
                "schema_version": 2,
                "stage": "short",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": True,
                "requested_iterations": 100,
                "completed_iterations": 100,
                "pilot_manifest": str(pilot),
                "pilot_manifest_sha256": module.sha256_file(pilot),
                **lineage,
                "candidate_checkpoints": candidates,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _safe_bridge_run(tmp_path, module):
    short = _safe_run(tmp_path, module)
    short_document = json.loads(short.read_text(encoding="utf-8"))
    candidates = []
    for updates in (100, 150, 200, 250, 300):
        checkpoint = tmp_path / f"bridge_candidate_u{updates:03d}.pt"
        count = 204800 + (updates - 100) * 2048
        torch.save(
            {
                "model_state_dict": {"candidate": torch.tensor(updates)},
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(updates)}}},
                "obs_norm_state_dict": {
                    "_mean": torch.zeros(1, 103),
                    "_count": torch.tensor(count, dtype=torch.int64),
                },
                "critic_obs_norm_state_dict": {
                    "_mean": torch.zeros(1, 103),
                    "_count": torch.tensor(count, dtype=torch.int64),
                },
                "iter": updates - 1,
            },
            checkpoint,
        )
        candidates.append(
            {
                "completed_updates": updates,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": module.sha256_file(checkpoint),
            }
        )
    parent = short_document["candidate_checkpoints"][-1]
    bridge = tmp_path / "bridge_manifest.json"
    bridge.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "stage": "bridge",
                "status": "safe_complete",
                "accepted": False,
                "promotion_required": True,
                "requested_iterations": 200,
                "completed_iterations": 200,
                "starting_updates": 100,
                "requested_additional_updates": 200,
                "target_total_updates": 300,
                "completed_total_updates": 300,
                "starting_normalizer_count": 204800,
                "short_manifest": str(short),
                "short_manifest_sha256": module.sha256_file(short),
                "pilot_manifest": short_document["pilot_manifest"],
                "pilot_manifest_sha256": short_document["pilot_manifest_sha256"],
                "parent_checkpoint": parent["checkpoint"],
                "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                "migrated_checkpoint": candidates[0]["checkpoint"],
                "migrated_checkpoint_sha256": candidates[0]["checkpoint_sha256"],
                **{
                    key: value
                    for key, value in short_document.items()
                    if key.endswith("_path")
                    or key.endswith("_sha256")
                    and key
                    in {
                        "asset_sha256",
                        "config_sha256",
                        "reward_sha256",
                        "runtime_sha256",
                        "reward_runtime_bundle_sha256",
                        "pilot_schema_sha256",
                    }
                },
                "candidate_checkpoints": candidates,
            }
        ),
        encoding="utf-8",
    )
    return bridge


class RecordingWorkerRunner:
    def __init__(
        self,
        module,
        *,
        checkpoint_sha=None,
        improve=False,
        lineage_override=None,
    ):
        self.module = module
        self.checkpoint_sha = checkpoint_sha
        self.improve = improve
        self.lineage_override = lineage_override
        self.zero_pair_calls = 0
        self.candidate_calls = 0

    def __call__(
        self,
        *,
        mode,
        seed,
        output,
        checkpoint,
        device,
        headless,
        source_lineage,
    ):
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
                    **(
                        source_lineage
                        if self.lineage_override is None
                        else {**source_lineage, **self.lineage_override}
                    ),
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


def test_bridge_driver_keeps_worker_count_and_emits_schema_v3_lineage(tmp_path):
    module = _load_script()
    bridge = _safe_bridge_run(tmp_path, module)
    runner = RecordingWorkerRunner(module)

    result = module.run_promotion(bridge_manifest=bridge, worker_runner=runner)

    assert runner.zero_pair_calls == 9
    assert runner.candidate_calls == 15
    assert result["schema_version"] == 3
    assert result["bridge_manifest"] == str(bridge.resolve())
    assert result["bridge_manifest_sha256"] == module.sha256_file(bridge)
    assert [value["completed_updates"] for value in result["candidates"]] == [
        100,
        150,
        200,
        250,
        300,
    ]


@pytest.mark.parametrize("failure", ("missing_count", "mismatched_count"))
def test_bridge_promotion_rejects_invalid_normalizer_counts(tmp_path, failure):
    module = _load_script()
    bridge = _safe_bridge_run(tmp_path, module)
    document = json.loads(bridge.read_text(encoding="utf-8"))
    candidate = document["candidate_checkpoints"][1]
    checkpoint = Path(candidate["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if failure == "missing_count":
        payload["obs_norm_state_dict"].pop("_count")
    else:
        payload["critic_obs_norm_state_dict"]["_count"] += 1
    torch.save(payload, checkpoint)
    candidate["checkpoint_sha256"] = module.sha256_file(checkpoint)
    bridge.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="count"):
        module.run_promotion(
            bridge_manifest=bridge, worker_runner=RecordingWorkerRunner(module)
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("status", "safety_stopped", "safe-complete"),
        ("candidate_set", None, "candidate"),
        ("short_manifest_sha256", "0" * 64, "short manifest SHA"),
        ("parent_checkpoint_sha256", "0" * 64, "parent_checkpoint_sha256"),
    ),
)
def test_bridge_promotion_rejects_invalid_bridge_lineage(
    tmp_path, field, value, match
):
    module = _load_script()
    bridge = _safe_bridge_run(tmp_path, module)
    document = json.loads(bridge.read_text(encoding="utf-8"))
    if field == "candidate_set":
        document["candidate_checkpoints"].pop()
    else:
        document[field] = value
    bridge.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        module.run_promotion(
            bridge_manifest=bridge, worker_runner=RecordingWorkerRunner(module)
        )


def test_bridge_cli_rejects_schema_v2_short_manifest(tmp_path):
    module = _load_script()
    short = _safe_run(tmp_path, module)
    with pytest.raises(ValueError, match="schema-v3"):
        module.run_promotion(
            bridge_manifest=short, worker_runner=RecordingWorkerRunner(module)
        )


def test_driver_fails_closed_on_worker_checkpoint_sha_mismatch(tmp_path):
    module = _load_script()
    runner = RecordingWorkerRunner(module, checkpoint_sha="0" * 64)

    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        module.run_promotion(_safe_run(tmp_path, module), worker_runner=runner)


def test_driver_fails_closed_on_worker_runtime_lineage_mismatch(tmp_path):
    module = _load_script()
    runner = RecordingWorkerRunner(
        module, lineage_override={"runtime_sha256": "0" * 64}
    )

    with pytest.raises(RuntimeError, match="runtime SHA"):
        module.run_promotion(_safe_run(tmp_path, module), worker_runner=runner)


def test_promotion_rejects_incomplete_short_or_rejected_pilot(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["completed_iterations"] = 99
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="100"):
        module.run_promotion(manifest_path, worker_runner=RecordingWorkerRunner(module))

    document["completed_iterations"] = 100
    pilot = Path(document["pilot_manifest"])
    pilot_document = json.loads(pilot.read_text(encoding="utf-8"))
    pilot_document["pilot_accepted"] = False
    pilot.write_text(json.dumps(pilot_document), encoding="utf-8")
    document["pilot_manifest_sha256"] = module.sha256_file(pilot)
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot"):
        module.run_promotion(manifest_path, worker_runner=RecordingWorkerRunner(module))


def test_promotion_rejects_candidate_without_normalizer_state(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = document["candidate_checkpoints"][0]
    checkpoint = Path(candidate["checkpoint"])
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop("obs_norm_state_dict")
    torch.save(payload, checkpoint)
    candidate["checkpoint_sha256"] = module.sha256_file(checkpoint)
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="normalizer"):
        module.run_promotion(manifest_path, worker_runner=RecordingWorkerRunner(module))


def test_promotion_rejects_reused_worker_output(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)
    reused = tmp_path / "noise_calibration/seed_42_pair_0.json"
    reused.parent.mkdir(parents=True)
    reused.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="worker output"):
        module.run_promotion(manifest_path, worker_runner=RecordingWorkerRunner(module))


def test_resume_reuses_valid_complete_workers_and_retries_failed_worker(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage = module.source_lineage(
        module.ResidualSourcePaths(
            *(
                Path(manifest[f"{label}_path"])
                for label in ("asset", "config", "reward", "runtime")
            )
        )
    )
    seeder = RecordingWorkerRunner(module)
    for seed in module.SEEDS:
        for pair_index in module.PAIR_INDICES:
            seeder(
                mode="zero-pair",
                seed=seed,
                output=tmp_path
                / "noise_calibration"
                / f"seed_{seed}_pair_{pair_index}.json",
                checkpoint=None,
                device="cuda:0",
                headless=True,
                source_lineage=lineage,
            )
    checkpoint = Path(manifest["candidate_checkpoints"][0]["checkpoint"])
    complete = tmp_path / "candidate_eval/candidate_u000/seed_42.json"
    seeder(
        mode="candidate",
        seed=42,
        output=complete,
        checkpoint=checkpoint,
        device="cuda:0",
        headless=True,
        source_lineage=lineage,
    )
    failed = tmp_path / "candidate_eval/candidate_u000/seed_43.json"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "failed",
                "mode": "candidate",
                "seed": 43,
                "steps": module.EVAL_STEPS,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": module.sha256_file(checkpoint),
                **lineage,
            }
        ),
        encoding="utf-8",
    )
    runner = RecordingWorkerRunner(module)

    module.run_promotion(manifest_path, worker_runner=runner, resume=True)

    assert runner.zero_pair_calls == 0
    assert runner.candidate_calls == 14
    assert json.loads(failed.read_text(encoding="utf-8"))["status"] == "complete"


def test_resume_fails_closed_on_retryable_worker_identity_mismatch(tmp_path):
    module = _load_script()
    manifest_path = _safe_run(tmp_path, module)
    failed = tmp_path / "noise_calibration/seed_42_pair_0.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(
        json.dumps(
            {
                "status": "failed",
                "mode": "zero-pair",
                "seed": 44,
                "steps": module.EVAL_STEPS,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="identity"):
        module.run_promotion(
            manifest_path,
            worker_runner=RecordingWorkerRunner(module),
            resume=True,
        )


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
    assert len(result["runtime_sha256"]) == 64
    assert len(result["reward_runtime_bundle_sha256"]) == 64
    assert result["best_completed_updates"] == 0
    assert module.sha256_file(tmp_path / "model_best.pt") == result[
        "best_checkpoint_sha256"
    ]
