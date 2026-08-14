from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "m1_panda_teacher_train.py"
SMOKE_PATH = ROOT / "scripts" / "m1_panda_teacher_smoke.py"
RUNBOOK_PATH = (
    ROOT.parent
    / "docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_teacher_train_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_teacher_smoke_under_test", SMOKE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_teacher_train_cfg_has_exact_small_mlp_ppo_contract_and_is_independent():
    from agent.m1_panda_teacher_train_cfg import get_m1_panda_teacher_train_cfg

    left = get_m1_panda_teacher_train_cfg()
    right = get_m1_panda_teacher_train_cfg()

    assert left["num_steps_per_env"] == 24
    assert left["save_interval"] == 100
    assert left["empirical_normalization"] is False
    assert left["policy"]["class_name"] == "ActorCritic"
    assert left["policy"]["actor_hidden_dims"] == [256, 128]
    assert left["policy"]["critic_hidden_dims"] == [256, 128]
    assert left["policy"]["init_noise_std"] == 0.01
    assert left["algorithm"]["class_name"] == "PPO"
    assert left["algorithm"]["num_learning_epochs"] == 5
    assert left["algorithm"]["num_mini_batches"] == 4

    left["policy"]["actor_hidden_dims"].append(64)
    left["algorithm"]["num_learning_epochs"] = 99
    assert right["policy"]["actor_hidden_dims"] == [256, 128]
    assert right["algorithm"]["num_learning_epochs"] == 5


def test_agent_package_exports_teacher_train_cfg_factory():
    import agent

    assert callable(agent.get_m1_panda_teacher_train_cfg)


def test_training_script_exposes_exact_stage_task_mapping_without_isaac_import():
    module = _load_script()

    assert module.TASK_IDS == {
        "A0": "Isaac-M1-Panda-Teacher-A0-v0",
        "A1": "Isaac-M1-Panda-Teacher-A1-v0",
    }


@pytest.mark.parametrize(
    ("stage", "base_checkpoint", "message"),
    [
        ("A0", "/tmp/base.pt", "does not accept"),
        ("A1", None, "requires"),
    ],
)
def test_cli_contract_rejects_invalid_stage_base_checkpoint_combinations(
    stage, base_checkpoint, message
):
    module = _load_script()
    args = SimpleNamespace(
        stage=stage,
        base_checkpoint=base_checkpoint,
        max_iterations=1,
        num_envs=1,
        save_interval=1,
        num_steps_per_env=1,
        learning_epochs=1,
        num_mini_batches=1,
        resume_checkpoint=None,
        run_name=None,
    )

    with pytest.raises(ValueError, match=message):
        module.validate_cli_contract(args)


@pytest.mark.parametrize(
    "field",
    [
        "max_iterations",
        "num_envs",
        "save_interval",
        "num_steps_per_env",
        "learning_epochs",
        "num_mini_batches",
    ],
)
def test_cli_contract_rejects_nonpositive_training_overrides(field):
    module = _load_script()
    args = SimpleNamespace(
        stage="A0",
        base_checkpoint=None,
        max_iterations=1,
        num_envs=1,
        save_interval=1,
        num_steps_per_env=1,
        learning_epochs=1,
        num_mini_batches=1,
        resume_checkpoint=None,
        run_name=None,
    )
    setattr(args, field, 0)

    with pytest.raises(ValueError, match=field):
        module.validate_cli_contract(args)


def test_cli_contract_forbids_run_name_when_resuming():
    module = _load_script()
    args = SimpleNamespace(
        stage="A0",
        base_checkpoint=None,
        max_iterations=1,
        num_envs=1,
        save_interval=1,
        num_steps_per_env=1,
        learning_epochs=1,
        num_mini_batches=1,
        resume_checkpoint="/tmp/model_1.pt",
        run_name="different-run",
    )

    with pytest.raises(ValueError, match="run_name"):
        module.validate_cli_contract(args)


def test_build_log_dir_is_stage_scoped_and_refuses_existing_directory(tmp_path):
    module = _load_script()

    created = module.build_log_dir(tmp_path, "A1", "named-run")

    assert created == tmp_path / "a1" / "named-run"
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        module.build_log_dir(tmp_path, "A1", "named-run")


def test_resume_reuses_checkpoint_parent_as_log_directory(tmp_path):
    module = _load_script()
    run_dir = tmp_path / "a0" / "existing"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model_3.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        resume_checkpoint=checkpoint,
        log_root=tmp_path,
        stage="A0",
        run_name=None,
    )

    assert module.resolve_log_dir(args) == run_dir.resolve()


def test_resume_advances_runner_to_iteration_after_loaded_checkpoint():
    module = _load_script()
    runner = SimpleNamespace(current_learning_iteration=7)

    module.advance_runner_after_resume(runner)

    assert runner.current_learning_iteration == 8


def test_runtime_snapshot_requires_exact_dimensions_and_nonzero_finite_wrench():
    module = _load_script()
    wrapper = SimpleNamespace(
        num_actions=16,
        max_abs_wrench_seen=1.0,
        get_observations=lambda: (torch.zeros(1, 60), {}),
    )

    assert module.runtime_contract_snapshot(wrapper) == {
        "policy_observation_dim": 60,
        "action_dim": 16,
        "max_abs_wrench_b_seen": 1.0,
    }

    wrapper.max_abs_wrench_seen = 0.0
    with pytest.raises(RuntimeError, match="nonzero"):
        module.runtime_contract_snapshot(wrapper)


def test_training_script_declares_all_required_cli_flags():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    required = {
        "--stage",
        "--num_envs",
        "--seed",
        "--max_iterations",
        "--run_name",
        "--log-root",
        "--base-checkpoint",
        "--resume-checkpoint",
        "--reset-optimizer",
        "--save-interval",
        "--num-steps-per-env",
        "--learning-epochs",
        "--num-mini-batches",
    }

    assert all(flag in source for flag in required)
    assert "AppLauncher.add_app_launcher_args(parser)" in source


def test_training_flow_orders_strict_checkpoint_manifest_and_cleanup_boundaries():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert source.index("load_frozen_teacher_actor(") < source.index(
        "M1PandaTeacherEnvWrapper("
    )
    assert source.index("validate_teacher_checkpoint(") < source.index(
        "runner.load("
    )
    assert source.index("atomic_write_manifest(") < source.index("runner.learn(")
    assert source.index("runner.learn(") < source.index(
        "wrapper.assert_frozen_actor_unchanged()"
    )
    assert source.index("env.close()") < source.index("simulation_app.close()")
    assert "status\"] = \"failed\"" in source
    assert "status\"] = \"completed\"" in source
    assert "deepcopy(train_cfg)" in source


def test_training_flow_resolves_string_environment_cfg_with_isaaclab_parser():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "from isaaclab_tasks.utils import parse_env_cfg" in source
    assert "parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)" in source
    assert "env_cfg_entry = gym.spec" not in source


def test_smoke_driver_declares_exact_four_stage_sequence_and_subprocess_contract():
    module = _load_smoke()
    source = SMOKE_PATH.read_text(encoding="utf-8")

    assert module.STAGE_SEQUENCE == (
        "a0_initial",
        "a0_resume",
        "a1_initial",
        "a1_resume",
    )
    assert (
        "subprocess.run(command, check=False, timeout=600, text=True, "
        "capture_output=True)"
    ) in source


def test_smoke_driver_discovers_latest_checkpoint_by_numeric_suffix(tmp_path):
    module = _load_smoke()
    (tmp_path / "model_2.pt").write_bytes(b"2")
    (tmp_path / "model_10.pt").write_bytes(b"10")
    (tmp_path / "model_latest.pt").write_bytes(b"ignored")

    assert module.latest_checkpoint(tmp_path).name == "model_10.pt"
    assert module.checkpoint_iteration(tmp_path / "model_10.pt") == 10


def test_smoke_driver_preserves_partial_output_when_child_times_out(
    tmp_path, monkeypatch
):
    module = _load_smoke()

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0], timeout=600, output="partial-out", stderr="partial-err"
        )

    monkeypatch.setattr(module.subprocess, "run", _timeout)

    with pytest.raises(RuntimeError, match="timed out"):
        module._run_child("timeout_stage", ["python", "train.py"], tmp_path)

    assert (tmp_path / "timeout_stage.stdout.log").read_text() == "partial-out"
    assert (tmp_path / "timeout_stage.stderr.log").read_text() == "partial-err"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": "failed"}, "status"),
        ({"stage": "A1"}, "stage"),
        ({"base_checkpoint_sha256": "wrong"}, "base_checkpoint_sha256"),
        ({"frozen_actor_final_sha256": "changed"}, "frozen actor"),
    ],
)
def test_smoke_manifest_validation_rejects_contract_drift(tmp_path, updates, message):
    module = _load_smoke()
    manifest = {
        "status": "completed",
        "stage": "A0",
        "base_checkpoint_sha256": "base",
        "frozen_actor_initial_sha256": "frozen",
        "frozen_actor_final_sha256": "frozen",
    }
    manifest.update(updates)

    with pytest.raises(RuntimeError, match=message):
        module.validate_completed_manifest(
            manifest,
            expected_stage="A0",
            expected_base_sha256="base",
            require_frozen_hash=True,
        )


def test_teacher_runbook_contains_complete_formal_resume_and_monitoring_commands():
    source = RUNBOOK_PATH.read_text(encoding="utf-8")

    assert source.count("scripts/m1_panda_teacher_train.py") >= 4
    assert "--stage A0 --num_envs 64 --max_iterations 3000" in source
    assert "--stage A0 --resume-checkpoint /ABS/A0/model_N.pt" in source
    assert "--stage A1 --base-checkpoint /ABS/A0/model_N.pt" in source
    assert "--resume-checkpoint /ABS/A1/model_M.pt" in source
    assert "scripts/m1_panda_teacher_smoke.py" in source
    assert "tensorboard --logdir" in source
    assert "run_manifest.json" in source
    assert "runtime_contract.max_abs_wrench_b_seen > 0" in source
    assert "sm_120" in source and "sm_90" in source
    assert "Ctrl+C" in source
