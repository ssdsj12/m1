from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "m1_panda_teacher_play.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_teacher_play_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTerminationManager:
    def __init__(self, **terms):
        self._terms = terms
        self.time_outs = terms.get("time_out")

    def get_term(self, name):
        if name not in self._terms:
            raise KeyError(name)
        return self._terms[name]


@pytest.mark.parametrize(
    ("stage", "base_checkpoint", "message"),
    [
        ("A0", Path("base.pt"), "does not accept"),
        ("A1", None, "requires"),
    ],
)
def test_play_cli_rejects_invalid_base_checkpoint_combinations(
    stage, base_checkpoint, message
):
    module = _load_script()
    args = SimpleNamespace(
        stage=stage,
        checkpoint=Path("model.pt"),
        base_checkpoint=base_checkpoint,
        num_envs=1,
        steps=0,
        stats_interval=100,
    )

    with pytest.raises(ValueError, match=message):
        module.validate_cli_contract(args)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"checkpoint": None}, "checkpoint"),
        ({"num_envs": 0}, "num_envs"),
        ({"steps": -1}, "steps"),
        ({"stats_interval": 0}, "stats_interval"),
    ],
)
def test_play_cli_rejects_missing_or_out_of_range_values(updates, message):
    module = _load_script()
    args = SimpleNamespace(
        stage="A0",
        checkpoint=Path("model.pt"),
        base_checkpoint=None,
        num_envs=1,
        steps=0,
        stats_interval=100,
    )
    for name, value in updates.items():
        setattr(args, name, value)

    with pytest.raises(ValueError, match=message):
        module.validate_cli_contract(args)


def test_full_scale_play_requires_a1_disturbance_finite_steps_and_summary(tmp_path):
    module = _load_script()
    args = SimpleNamespace(
        stage="A1",
        checkpoint=Path("model.pt"),
        base_checkpoint=Path("base.pt"),
        num_envs=64,
        steps=2000,
        stats_interval=100,
        disable_disturbance=False,
        full_scale_disturbance=True,
        summary_json=tmp_path / "row.json",
    )

    module.validate_cli_contract(args)

    args.disable_disturbance = True
    with pytest.raises(ValueError, match="full-scale"):
        module.validate_cli_contract(args)
    args.disable_disturbance = False
    args.stage = "A0"
    args.base_checkpoint = None
    with pytest.raises(ValueError, match="A1"):
        module.validate_cli_contract(args)
    args.stage = "A1"
    args.base_checkpoint = Path("base.pt")
    args.steps = 0
    with pytest.raises(ValueError, match="positive steps"):
        module.validate_cli_contract(args)
    args.steps = 2000
    args.summary_json = None
    with pytest.raises(ValueError, match="summary-json"):
        module.validate_cli_contract(args)


def test_update_reset_counts_preserves_unavailable_terms():
    module = _load_script()
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=None))
    counts = {"bad_orientation": None, "base_contact": None, "time_out": None}

    module.update_reset_counts(env, counts)

    assert counts == {
        "bad_orientation": None,
        "base_contact": None,
        "time_out": None,
    }


def test_update_reset_counts_accumulates_available_terms():
    module = _load_script()
    manager = _FakeTerminationManager(
        bad_orientation=torch.tensor([True, False]),
        base_contact=torch.tensor([False, True]),
        time_out=torch.tensor([False, False]),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(termination_manager=manager, device="cpu")
    )
    counts = {"bad_orientation": None, "base_contact": None, "time_out": None}

    module.update_reset_counts(env, counts)

    assert counts == {"bad_orientation": 1, "base_contact": 1, "time_out": 0}


def test_format_play_stats_reports_wrench_axes_and_unavailable_terms():
    module = _load_script()

    line = module.format_play_stats(
        step=100,
        mean_reward=1.25,
        done_count=3,
        wrench_b=torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]]),
        max_abs_wrench_seen=6.0,
        reset_counts={
            "bad_orientation": 1,
            "base_contact": None,
            "time_out": 2,
        },
    )

    assert "step=100" in line
    assert "mean_reward=1.250000" in line
    assert "done=3" in line
    assert "wrench_axis_abs_max=[1.000,2.000,3.000,4.000,5.000,6.000]" in line
    assert "max_abs_wrench_seen=6.000" in line
    assert "bad_orientation=1" in line
    assert "base_contact=unavailable" in line
    assert "time_out=2" in line


@pytest.mark.parametrize(
    "wrench",
    [torch.zeros(1, 5), torch.full((1, 6), float("nan"))],
)
def test_format_play_stats_rejects_invalid_wrench(wrench):
    module = _load_script()

    with pytest.raises(ValueError, match="wrench"):
        module.format_play_stats(
            step=1,
            mean_reward=0.0,
            done_count=0,
            wrench_b=wrench,
            max_abs_wrench_seen=0.0,
            reset_counts={name: None for name in module.TERMINATION_NAMES},
        )


def test_play_script_declares_strict_cli_and_inference_lifecycle():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    required_flags = {
        "--stage",
        "--checkpoint",
        "--base-checkpoint",
        "--num-envs",
        "--seed",
        "--steps",
        "--stats-interval",
        "--disable-disturbance",
        "--full-scale-disturbance",
        "--summary-json",
    }

    assert all(flag in source for flag in required_flags)
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert source.index("validate_teacher_checkpoint(") < source.index("runner.load(")
    assert "require_optimizer=False" in source
    assert "load_optimizer=False" in source
    assert "keep_std=True" in source
    assert "torch.inference_mode()" in source
    assert "simulation_app.is_running()" in source
    assert "disturbance_enabled=not args.disable_disturbance" in source
    assert "initial_curriculum_step=initial_curriculum_step" in source
    assert "atomic_write_manifest(args.summary_json" in source
    assert "wrapper.assert_frozen_actor_unchanged()" in source
    assert source.index("env.close()") < source.index("simulation_app.close()")
    assert "runner.learn(" not in source
    assert "build_run_manifest(" not in source


def test_play_script_reuses_exact_teacher_task_and_wrapper_boundaries():
    module = _load_script()
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert module.TASK_IDS == {
        "A0": "Isaac-M1-Panda-Teacher-A0-v0",
        "A1": "Isaac-M1-Panda-Teacher-A1-v0",
    }
    assert "from isaaclab_tasks.utils import parse_env_cfg" in source
    assert "M1PandaTeacherEnvWrapper(" in source
    assert "load_frozen_teacher_actor(" in source
    assert "file_sha256(args.base_checkpoint)" in source
    assert "M1RslRlEnvWrapper" not in source
