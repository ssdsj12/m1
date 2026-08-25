from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from go2_pvcnn.tasks.m1_panda_folded_load_training_guard import EpisodeRecord, sha256_file


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/m1_panda_folded_load_train.py"
EVAL = ROOT / "scripts/m1_panda_folded_load_eval.py"
RUNNER = ROOT / "rsl_rl/rsl_rl/runners/on_policy_runner.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, *, stage="L0-C0", accepted=True) -> Path:
    checkpoint = tmp_path / "model_final.pt"
    checkpoint.write_bytes(b"parent-policy")
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "accepted": accepted,
                "final_checkpoint": str(checkpoint),
                "final_checkpoint_sha256": sha256_file(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    return path


def _episode(command, *, env_id, timeout=True, contact=False, orientation=False,
             vx_error=.01, wz_error=.02, stationary_vx=0.0, stationary_wz=0.0):
    steps = 100
    return EpisodeRecord(
        command=command,
        steps=steps,
        time_out=timeout,
        base_contact=contact,
        bad_orientation=orientation,
        vx_error_sq_sum=steps * vx_error**2,
        wz_error_sq_sum=steps * wz_error**2,
        stationary_abs_vx_sum=steps * stationary_vx,
        stationary_abs_wz_sum=steps * stationary_wz,
        env_id=env_id,
    )


def _balanced_records():
    records = []
    for env_id in range(64):
        group = env_id % 5
        command = (
            (0.05, 0.0, 0.0),
            (-0.05, 0.0, 0.0),
            (0.0, 0.0, 0.15),
            (0.0, 0.0, -0.15),
            (0.0, 0.0, 0.0),
        )[group]
        records.append(_episode(command, env_id=env_id))
    return records


def test_l0_is_fresh_and_non_l0_requires_accepted_immediate_parent(tmp_path):
    train = _load(TRAIN, "folded_train_lineage")
    assert train.validate_parent("L0-C0", None) is None
    with pytest.raises(ValueError, match="fresh"):
        train.validate_parent("L0-C0", tmp_path / "old.json")
    with pytest.raises(ValueError, match="requires"):
        train.validate_parent("L1-C1", None)

    rejected = _manifest(tmp_path, accepted=False)
    with pytest.raises(ValueError, match="accepted=true"):
        train.validate_parent("L1-C1", rejected)

    wrong = _manifest(tmp_path, stage="L1-C1")
    with pytest.raises(ValueError, match="immediate parent"):
        train.validate_parent("L1-C1", wrong)


def test_parent_checkpoint_sha_and_empty_run_directory_are_enforced(tmp_path):
    train = _load(TRAIN, "folded_train_sha")
    manifest = _manifest(tmp_path)
    lineage = train.validate_parent("L1-C1", manifest)
    assert lineage.stage == "L0-C0"
    assert lineage.checkpoint_sha256 == sha256_file(lineage.checkpoint)
    lineage.checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        train.validate_parent("L1-C1", manifest)

    run_dir = tmp_path / "new-run"
    assert train.prepare_empty_run_dir(run_dir) == run_dir.resolve()
    (run_dir / "occupied").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        train.prepare_empty_run_dir(run_dir)


def test_fixed_evaluation_uses_same_physical_gates_and_all_64_environments():
    evaluate = _load(EVAL, "folded_eval_gates")
    report = evaluate.evaluate_records("L0-C0", 42, _balanced_records())
    assert report["passed"] is True
    assert report["completed_episodes"] == 64
    assert report["seed"] == 42
    assert dict(report["bucket_counts"])["forward"] >= 8

    duplicate = _balanced_records()
    duplicate[-1] = _episode((0.05, 0.0, 0.0), env_id=0)
    with pytest.raises(ValueError, match="exactly one episode"):
        evaluate.evaluate_records("L0-C0", 42, duplicate)
    bad = _balanced_records()
    bad[0] = _episode((0.05, 0.0, 0.0), env_id=0, contact=True)
    assert evaluate.evaluate_records("L0-C0", 42, bad)["passed"] is False


def test_scripts_and_runner_contain_exact_operational_contracts():
    train = TRAIN.read_text(encoding="utf-8")
    evaluate = EVAL.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'TASK_ID = "Isaac-M1-Panda-Folded-Load-v0"' in train
    assert "M1PandaFoldedLoadEnvWrapper" in train
    assert "configure_folded_load_stage" in train
    assert "--run_dir" in train and "--parent_manifest" in train
    assert "load_optimizer=False" in train and "clip_std(max=0.01)" in train
    assert "apply_external_force_torque" not in train
    assert "--num_envs" in evaluate and "default=64" in evaluate
    assert "EVALUATION_SEEDS = (42, 43, 44)" in evaluate
    assert "balanced_eval_commands" in evaluate
    assert "act_inference" in evaluate or "get_inference_policy" in evaluate
    assert "apply_external_force_torque" not in evaluate
    for scalar in (
        '"Loss/kl_max"',
        '"Loss/kl_aborted"',
        '"Loss/completed_mini_batches"',
        '"Loss/grad_norm"',
        '"Policy/active_action_std_min"',
        '"Policy/active_action_std_max"',
    ):
        assert scalar in runner
