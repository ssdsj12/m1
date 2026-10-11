from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
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
    episodes = tmp_path / "episodes.jsonl"
    accepted = {
        "episode_index": 0,
        "terminal_reason": "accepted",
        "success": True,
        "contacts_left": 2,
        "contacts_right": 2,
        "contact_forces_left": [0.21, 0.3],
        "contact_forces_right": [0.4, 0.5],
        "lift_height": 0.05,
        "hold_time": 1.0,
        "slip": 0.02,
    }
    episodes.write_text(json.dumps(accepted) + "\n", encoding="utf-8")
    body = {
        "schema_version": 1,
        "status": "completed",
        "stage": "approval",
        "seed": 42,
        "source": str(module.THIS_FILE),
        "source_sha256": _sha(module.THIS_FILE),
        "config": str(CONFIG),
        "config_sha256": _sha(CONFIG),
        "action_dim": 15,
        "episodes_jsonl": "episodes.jsonl",
        "episodes_jsonl_sha256": _sha(episodes),
        "acceptance_counts": {"accepted": 1, "rejected": 0},
        "terminal_reason_counts": {"accepted": 1},
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
    with pytest.raises(ValueError, match="acceptance counts"):
        module.validate_approval_resume(checkpoint)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run, body: (run / "episodes.jsonl").unlink(), "episode ledger"),
        (lambda run, body: body.update(episodes_jsonl_sha256="0" * 64), "episode ledger SHA-256"),
        (lambda run, body: body["acceptance_counts"].update(accepted=2), "acceptance counts"),
        (lambda run, body: body.update(source_sha256="not-a-sha"), "source SHA-256"),
        (lambda run, body: body.update(config_sha256="not-a-sha"), "config SHA-256"),
        (lambda run, body: body.update(source_sha256="0" * 64), "source is incompatible"),
        (lambda run, body: body.update(config_sha256="0" * 64), "config is incompatible"),
    ],
)
def test_action_rejects_incomplete_or_incompatible_approval_evidence(tmp_path, mutation, message):
    module = _load_module()
    checkpoint = _approval_run(tmp_path, module)
    manifest_path = tmp_path / "training.manifest.json"
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(tmp_path, body)
    body["manifest_sha256"] = module.manifest_body_sha256(body)
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError), match=message):
        module.validate_approval_resume(checkpoint)


def test_action_rejects_accepted_row_that_does_not_meet_fixed_physical_gate(tmp_path):
    module = _load_module()
    checkpoint = _approval_run(tmp_path, module)
    episodes = tmp_path / "episodes.jsonl"
    row = json.loads(episodes.read_text(encoding="utf-8"))
    row["contact_forces_left"][0] = 0.2
    episodes.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "training.manifest.json"
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body["episodes_jsonl_sha256"] = _sha(episodes)
    body["manifest_sha256"] = module.manifest_body_sha256(body)
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="physical acceptance"):
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


def test_torch_runtime_is_single_threaded_and_repeated_calls_are_safe():
    torch = pytest.importorskip("torch")
    module = _load_module()
    module.configure_torch_threads(torch)
    module.configure_torch_threads(torch)
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1


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


def test_best_checkpoint_is_policy_that_generated_new_accepted_rollout(tmp_path):
    module = _load_module()
    recorder = module.EpisodeRecorder(tmp_path / "episodes.jsonl")

    class StatefulRunner:
        state = b"rollout-policy"

        def save(self, path):
            Path(path).write_bytes(self.state)

    runner = StatefulRunner()
    tracker = module.RolloutBestCheckpointTracker(runner, recorder, tmp_path)
    tracker.capture_rollout_policy()
    recorder.record({
        "success": True,
        "contacts_left": 2,
        "contacts_right": 2,
        "contact_forces_left": [0.21, 0.22],
        "contact_forces_right": [0.3, 0.4],
        "lift_height": 0.05,
        "hold_time": 1.0,
        "slip": 0.02,
    })
    runner.state = b"updated-policy"
    published = tracker.after_update()
    assert (tmp_path / "best.pt").read_bytes() == b"rollout-policy"
    assert published == {"path": "best.pt", "sha256": _sha(tmp_path / "best.pt")}
    assert tracker.rollout_checkpoint.read_bytes() == b"updated-policy"


def test_episode_jsonl_and_manifest_include_reproducibility_and_outcomes(tmp_path):
    module = _load_module()
    recorder = module.EpisodeRecorder(tmp_path / "episodes.jsonl")
    assert recorder.path.is_file()
    assert recorder.path.read_bytes() == b""
    recorder.record({
        "success": True,
        "failure_reason": None,
        "reward": 2.0,
        "contacts_left": 2,
        "contacts_right": 2,
        "contact_forces_left": [0.21, 0.22],
        "contact_forces_right": [0.3, 0.4],
        "lift_height": 0.05,
        "hold_time": 1.0,
        "slip": 0.02,
    })
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


def test_formal_acceptance_is_fixed_and_does_not_follow_reward_shaping_config(tmp_path):
    module = _load_module()
    config = deepcopy(module.load_train_config(CONFIG))
    config["rewards"]["approval"].update({
        "minimum_contacts_per_hand": 1,
        "contact_force_threshold": 0.01,
        "minimum_lift_m": 0.001,
        "minimum_hold_s": 0.01,
        "maximum_slip_m": 1.0,
    })
    assert module.validate_reward_config(config) is None
    recorder = module.EpisodeRecorder(tmp_path / "episodes.jsonl")
    row = recorder.record({
        "success": True,
        "contacts_left": 1,
        "contacts_right": 1,
        "contact_forces_left": [1.0],
        "contact_forces_right": [1.0],
        "lift_height": 1.0,
        "hold_time": 2.0,
        "slip": 0.0,
    })
    assert row["success"] is False
    assert recorder.acceptance_counts == {"accepted": 0, "rejected": 1}


def test_recorder_materializes_contact_forces_from_production_diagnostics(tmp_path):
    module = _load_module()
    recorder = module.EpisodeRecorder(tmp_path / "episodes.jsonl")
    row = recorder.record({
        "success": True,
        "contacts_left": 2,
        "contacts_right": 2,
        "lift_height": 0.05,
        "hold_time": 1.0,
        "slip": 0.02,
        "diagnostics": {
            "left_contact_mask": [True, False, True, False, False],
            "right_contact_mask": [False, True, False, True, False],
            "left_fingertip_forces_b": [
                [0.21, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.22, 0.0],
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            ],
            "right_fingertip_forces_b": [
                [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0],
                [0.0, 0.4, 0.0], [0.0, 0.0, 0.0],
            ],
        },
    })
    assert row["success"] is True
    assert row["contact_forces_left"] == pytest.approx([0.21, 0.22])
    assert row["contact_forces_right"] == pytest.approx([0.3, 0.4])


def test_wire_adapter_exposes_unwrapped_and_real_runner_completes_one_cpu_iteration(tmp_path):
    torch = pytest.importorskip("torch")
    module = _load_module()
    from rsl_rl.runners import OnPolicyRunner

    class TinyWireEnv:
        max_episode_steps = 2
        device = torch.device("cpu")
        stage = SimpleNamespace(value="approval")

        def __init__(self):
            self.steps = 0

        @property
        def unwrapped(self):
            return self

        def reset(self, seed=None):
            self.steps = 0
            return torch.zeros(6), {}

        def step(self, action):
            self.steps += 1
            truncated = self.steps == self.max_episode_steps
            return torch.full((6,), float(self.steps)), 0.1, False, truncated, {
                "success": False,
                "failure_reason": "truncated" if truncated else None,
            }

        def close(self):
            pass

    recorder = module.EpisodeRecorder(tmp_path / "episodes.jsonl")
    physical = TinyWireEnv()
    env = module.WireRslRlEnv(physical, recorder, seed=42, torch_module=torch)
    assert env.unwrapped is physical
    cfg = deepcopy(module.load_train_config(CONFIG)["runner"])
    cfg.update(num_steps_per_env=2, save_interval=100, empirical_normalization=False)
    cfg["policy"].update(actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=0.1)
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, schedule="fixed")
    runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
    result = runner.learn(num_learning_iterations=1)
    assert result.completed_iterations == 1
    assert recorder.acceptance_counts == {"accepted": 0, "rejected": 1}


def test_truncated_transition_is_exposed_as_rsl_rl_timeout(tmp_path):
    torch = pytest.importorskip("torch")
    module = _load_module()

    class TruncatingEnv:
        max_episode_steps = 1
        device = torch.device("cpu")
        stage = SimpleNamespace(value="approval")

        def reset(self, seed=None):
            return torch.zeros(3), {}

        def step(self, action):
            return torch.ones(3), 0.0, False, True, {"success": False}

        def close(self):
            pass

    env = module.WireRslRlEnv(
        TruncatingEnv(), module.EpisodeRecorder(tmp_path / "episodes.jsonl"),
        seed=0, torch_module=torch,
    )
    _, _, dones, infos = env.step(torch.zeros(1, 15))
    assert dones.tolist() == [True]
    assert infos["time_outs"].dtype is torch.bool
    assert infos["time_outs"].tolist() == [True]


def test_manifest_atomic_writer_embeds_verifiable_body_sha(tmp_path):
    module = _load_module()
    target = tmp_path / "training.manifest.json"
    module.atomic_write_manifest(target, {"schema_version": 1, "stage": "approval"})
    document = json.loads(target.read_text())
    assert document["manifest_sha256"] == module.manifest_body_sha256(document)
    assert not list(tmp_path.glob(".training.manifest.json.*.tmp"))
