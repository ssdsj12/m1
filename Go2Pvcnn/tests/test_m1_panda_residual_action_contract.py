from dataclasses import dataclass

import pytest
import torch

from go2_pvcnn.tasks.m1_panda_residual_wbc_wrapper import (
    M1PandaResidualWbcController,
)


@dataclass(frozen=True)
class _FakeCommand:
    effort: torch.Tensor


class _FakeTeacher:
    def __init__(self, index):
        self.index = index
        self.calls = []
        self.resets = []

    def reset(self, state, *, seed):
        self.resets.append((state, seed))

    def step(self, state, **kwargs):
        self.calls.append((state, kwargs))
        return _FakeCommand(
            effort=torch.full((23,), float(self.index), dtype=torch.float64)
        )


def _controller(count=2):
    teachers = [_FakeTeacher(index) for index in range(count)]
    return M1PandaResidualWbcController(
        teachers, device="cpu", dtype=torch.float64, base_seed=10
    ), teachers


def _limits(count=2):
    return torch.tensor([[[-1.0, 1.0]] * 12] * count, dtype=torch.float64)


def test_controller_maps_batched_8d_input_to_one_23d_teacher_command_per_env():
    controller, teachers = _controller()
    action = torch.zeros(2, 8, dtype=torch.float64)
    before = action.clone()

    result = controller.step(
        states=("state0", "state1"),
        normalized_residual=action,
        measured_mount_wrench_b=torch.zeros(2, 6, dtype=torch.float64),
        leg_soft_limits=_limits(),
    )

    assert len(result.teacher_commands) == 2
    assert all(command.effort.shape == (23,) for command in result.teacher_commands)
    assert torch.equal(action, before)
    assert result.applied_residual.physical.shape == (2, 8)
    assert result.correction_wrench_b.shape == (2, 6)
    assert teachers[0].calls[0][1]["residual_command"].physical.shape == (8,)
    assert teachers[0].calls[0][1]["leg_soft_limits"].shape == (12, 2)


def test_controller_mount_feedback_has_expected_opposition_sign():
    controller, _ = _controller(1)

    result = controller.step(
        states=("state",),
        normalized_residual=torch.zeros(1, 8, dtype=torch.float64),
        measured_mount_wrench_b=torch.tensor(
            [[10.0, -10.0, 2.0, 4.0, -4.0, 1.0]], dtype=torch.float64
        ),
        leg_soft_limits=_limits(1),
    )

    assert torch.allclose(
        result.correction_wrench_b,
        torch.tensor(
            [[-1.5, 1.5, -0.3, -0.4, 0.4, -0.1]], dtype=torch.float64
        ),
    )
    assert torch.equal(
        result.applied_residual.wrench_b, result.correction_wrench_b
    )


def test_controller_selective_reset_clears_only_selected_residual_state():
    controller, teachers = _controller(3)
    controller.step(
        states=(0, 1, 2),
        normalized_residual=torch.ones(3, 8, dtype=torch.float64),
        measured_mount_wrench_b=torch.zeros(3, 6, dtype=torch.float64),
        leg_soft_limits=_limits(3),
    )
    before = controller.previous_physical

    controller.reset(torch.tensor([1]), states=(None, "reset1", None))

    after = controller.previous_physical
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[1], torch.zeros(8, dtype=torch.float64))
    assert torch.equal(after[2], before[2])
    assert teachers[0].resets == []
    assert teachers[1].resets == [("reset1", 11)]
    assert teachers[2].resets == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("normalized_residual", torch.zeros(2, 7, dtype=torch.float64), "shape"),
        ("measured_mount_wrench_b", torch.zeros(2, 5, dtype=torch.float64), "shape"),
        ("leg_soft_limits", torch.zeros(2, 12, 3, dtype=torch.float64), "leg_soft_limits"),
    ),
)
def test_controller_rejects_boundary_shape_mismatch(field, value, match):
    controller, _ = _controller()
    kwargs = {
        "states": (0, 1),
        "normalized_residual": torch.zeros(2, 8, dtype=torch.float64),
        "measured_mount_wrench_b": torch.zeros(2, 6, dtype=torch.float64),
        "leg_soft_limits": _limits(),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        controller.step(**kwargs)


def test_controller_rejects_shared_teacher_instances():
    teacher = _FakeTeacher(0)
    with pytest.raises(ValueError, match="distinct"):
        M1PandaResidualWbcController(
            [teacher, teacher], device="cpu", dtype=torch.float64
        )


def test_controller_module_does_not_depend_on_ppo_or_arm_mpc():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "go2_pvcnn/tasks/m1_panda_residual_wbc_wrapper.py"
    ).read_text()
    assert "rsl_rl" not in source
    assert "arm_mpc" not in source
