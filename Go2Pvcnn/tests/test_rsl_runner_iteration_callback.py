from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest
import torch

from rsl_rl.runners import on_policy_runner
from rsl_rl.runners.on_policy_runner import (
    IterationSummary,
    LearnResult,
    freeze_environment_metrics,
    freeze_episode_metrics,
    policy_active_std_range,
)


def test_freeze_episode_metrics_detaches_sorts_and_rejects_nonfinite() -> None:
    source = torch.tensor([1.0, 0.0], requires_grad=True)

    frozen = freeze_episode_metrics(
        [
            {"Termination/time_out": source},
            {"Reward/base_target": 2.5},
        ]
    )

    assert frozen == (
        ("Reward/base_target", (2.5,)),
        ("Termination/time_out", (1.0, 0.0)),
    )
    source.data.fill_(9.0)
    assert frozen[1][1] == (1.0, 0.0)

    with pytest.raises(ValueError, match="finite"):
        freeze_episode_metrics(
            [{"Reward/base_target": torch.tensor([float("nan")])}]
        )


def test_freeze_environment_metrics_copies_sorts_and_rejects_nonfinite() -> None:
    source = {"wrench_scale": torch.tensor(0.5), "reset_scale": 0.25}

    frozen = freeze_environment_metrics(source)

    assert frozen == (("reset_scale", 0.25), ("wrench_scale", 0.5))
    source["reset_scale"] = 1.0
    assert frozen[0] == ("reset_scale", 0.25)

    with pytest.raises(ValueError, match="finite"):
        freeze_environment_metrics({"wrench_scale": float("inf")})


def test_iteration_summary_and_learn_result_are_frozen() -> None:
    summary = IterationSummary(3, 4096, (1.0,), (), 1.0e-4, 0.01, ())
    result = LearnResult(completed_iterations=4, stop_reason="patience")

    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.iteration = 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.stop_reason = "changed"


def test_callback_is_optional_and_default_save_loop_is_preserved() -> None:
    signature = inspect.signature(on_policy_runner.OnPolicyRunner.learn)
    source = Path(on_policy_runner.__file__).read_text(encoding="utf-8")

    assert signature.parameters["iteration_callback"].default is None
    assert signature.return_annotation in {"LearnResult", LearnResult}
    assert "if it % self.save_interval == 0" in source
    assert "return LearnResult(" in source
    assert source.index("if new_ids.numel() > 0:") < source.index(
        'ep_infos.append(infos["log"])'
    )


def test_active_std_range_ignores_inactive_coordinates_and_rejects_nonfinite():
    actor = torch.nn.Module()
    actor.std = torch.nn.Parameter(torch.tensor((0.01, 0.02, 100.0)))
    actor.register_buffer("active_action_mask", torch.tensor((True, True, False)))

    minimum, maximum = policy_active_std_range(actor)

    assert minimum.item() == pytest.approx(0.01)
    assert maximum.item() == pytest.approx(0.02)
    actor.std.data[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        policy_active_std_range(actor)
