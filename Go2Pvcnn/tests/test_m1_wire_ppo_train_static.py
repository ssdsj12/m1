from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m1_wire_ppo_train.py"
CONFIG = ROOT / "config" / "m1_wire_ppo_train.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("m1_wire_ppo_train", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approval_run(tmp_path: Path, module):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"approved checkpoint")
    body = {
        "schema_version": 1,
        "status": "completed",
        "stage": "approval",
        "seed": 42,
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "action_dim": 15,
        "acceptance_counts": {"accepted": 1, "rejected": 3},
        "terminal_reason_counts": {"accepted": 1, "timeout": 3},
        "best_checkpoint": {"path": "best.pt", "sha256": _sha(checkpoint)},
    }
    body["manifest_sha256"] = module.manifest_body_sha256(body)
    (tmp_path / "training.manifest.json").write_text(
        json.dumps(body), encoding="utf-8"
    )
    return checkpoint


def test_cli_exposes_exact_two_stage_contract_and_validates_positive_values():
    module = _load_module()
    parser = module.build_arg_parser(include_app_launcher_args=False)
    args = parser.parse_args(
        [
            "--stage", "approval",
            "--num-envs", "1",
            "--max-iterations", "3",
            "--seed", "7",
            "--device", "cuda:0",
            "--run-dir", "/tmp/wire-ppo",
        ]
    )
    module.validate_cli_contract(args)
    assert args.stage == "approval"
    assert args.resume_checkpoint is None
    assert args.num_envs == 1
    assert args.max_iterations == 3

    with pytest.raises(ValueError, match="positive"):
        module.validate_cli_contract(SimpleNamespace(**{**vars(args), "max_iterations": 0}))
    with pytest.raises(ValueError, match="cuda:0"):
        module.validate_cli_contract(SimpleNamespace(**{**vars(args), "device": "cpu"}))


def test_launcher_parser_does_not_register_device_twice():
    module = _load_module()
    parser = module.build_arg_parser(include_app_launcher_args=True)
    args = parser.parse_args(["--stage", "approval", "--run-dir", "/tmp/wire-ppo"])
    assert args.device == "cuda:0"


def test_action_requires_verified_approval_checkpoint_and_manifest(tmp_path):
    module = _load_module()
    parser = module.build_arg_parser(include_app_launcher_args=False)
    missing = parser.parse_args(
        ["--stage", "action", "--run-dir", str(tmp_path / "action")]
    )
    with pytest.raises(ValueError, match="--resume-checkpoint"):
        module.validate_cli_contract(missing)

    checkpoint = _approval_run(tmp_path / "approval", module)
    args = parser.parse_args(
        [
            "--stage", "action",
            "--resume-checkpoint", str(checkpoint),
            "--run-dir", str(tmp_path / "action"),
        ]
    )
    module.validate_cli_contract(args)
    lineage = module.validate_approval_resume(checkpoint)
    assert lineage["checkpoint_sha256"] == _sha(checkpoint)
    assert lineage["manifest_sha256"] == _sha(checkpoint.parent / "training.manifest.json")

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        module.validate_approval_resume(checkpoint)


def test_action_rejects_unaccepted_or_nonapproval_manifest(tmp_path):
    module = _load_module()
    checkpoint = _approval_run(tmp_path, module)
    manifest_path = tmp_path / "training.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stage"] = "action"
    manifest["manifest_sha256"] = module.manifest_body_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Stage A approval"):
        module.validate_approval_resume(checkpoint)

    checkpoint = _approval_run(tmp_path, module)
    manifest = json.loads(manifest_path.read_text())
    manifest["acceptance_counts"]["accepted"] = 0
    manifest["manifest_sha256"] = module.manifest_body_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="accepted episode"):
        module.validate_approval_resume(checkpoint)


def test_runtime_environment_pins_gpu0_and_one_cpu_helper_thread(monkeypatch):
    module = _load_module()
    for name in module.RUNTIME_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    module.configure_runtime_environment()
    assert module.RUNTIME_ENVIRONMENT == {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in module.RUNTIME_ENVIRONMENT.items():
        assert module.os.environ[name] == value


def test_config_and_source_wire_rsl_rl_to_exact_15d_stage_environment():
    module = _load_module()
    config = module.load_train_config(CONFIG)
    assert config["action_dim"] == 15
    assert config["runner"]["policy"]["class_name"] == "ActorCritic"
    assert config["runner"]["algorithm"]["class_name"] == "PPO"
    assert set(config["rewards"]) == {"approval", "action"}

    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "OnPolicyRunner",
        "M1WirePpoEnv",
        "WirePpoRewardCfg",
        "WIRE_PPO_ACTION_DIM",
        "runner.learn(",
        "runner.load(",
    ):
        assert token in source


def test_best_checkpoint_is_published_atomically(tmp_path):
    module = _load_module()

    class FakeRunner:
        def save(self, path):
            Path(path).write_bytes(b"checkpoint")

    target = tmp_path / "best.pt"
    digest = module.atomic_publish_checkpoint(FakeRunner(), target)
    assert target.read_bytes() == b"checkpoint"
    assert digest == _sha(target)
    assert not list(tmp_path.glob(".best.pt.*.tmp"))


def test_episode_jsonl_and_manifest_include_reproducibility_and_outcomes(tmp_path):
    module = _load_module()
    recorder = module.EpisodeRecorder(tmp_path / "episodes.jsonl")
    assert recorder.path.is_file()
    assert recorder.path.read_bytes() == b""
    recorder.record({"success": True, "failure_reason": None, "reward": 2.0})
    recorder.record({"success": False, "failure_reason": "safety_projection", "reward": -1.0})
    assert recorder.acceptance_counts == {"accepted": 1, "rejected": 1}
    assert recorder.terminal_reason_counts == {"accepted": 1, "safety_projection": 1}
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in rows] == [0, 1]

    manifest = module.build_training_manifest(
        stage="approval",
        seed=42,
        num_envs=1,
        max_iterations=2,
        device="cuda:0",
        run_dir=tmp_path,
        config_path=CONFIG,
    )
    assert manifest["stage"] == "approval"
    assert manifest["seed"] == 42
    assert manifest["action_dim"] == 15
    assert manifest["source_sha256"] == _sha(SCRIPT)
    assert manifest["config_sha256"] == _sha(CONFIG)
    assert manifest["acceptance_counts"] == {"accepted": 0, "rejected": 0}
    assert manifest["terminal_reason_counts"] == {}
    assert manifest["episodes_jsonl"] == "episodes.jsonl"


def test_manifest_atomic_writer_embeds_verifiable_body_sha(tmp_path):
    module = _load_module()
    target = tmp_path / "training.manifest.json"
    module.atomic_write_manifest(target, {"schema_version": 1, "stage": "approval"})
    document = json.loads(target.read_text())
    assert document["manifest_sha256"] == module.manifest_body_sha256(document)
    assert not list(tmp_path.glob(".training.manifest.json.*.tmp"))
