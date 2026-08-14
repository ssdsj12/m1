from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from go2_pvcnn.tasks.m1_panda_teacher import (
    M1PandaDisturbanceCfg,
    M1PandaDisturbanceScheduler,
    base_wrench_to_body_local,
    clear_external_wrench,
    stage_disturbance_cfg,
)


def test_stage_defaults_are_exact():
    a0 = stage_disturbance_cfg("A0")
    assert a0.force_limit_n == (10.0, 10.0, 10.0)
    assert a0.torque_limit_nm == (2.0, 2.0, 2.0)
    assert (a0.hold_time_min_s, a0.hold_time_max_s) == (1.0, 2.0)
    assert a0.mode_probabilities == (1.0, 0.0, 0.0)
    assert a0.curriculum_start_scale == pytest.approx(0.25)
    assert a0.curriculum_steps == 50_000

    a1 = stage_disturbance_cfg("A1")
    assert a1.force_limit_n == (20.0, 20.0, 20.0)
    assert a1.torque_limit_nm == (5.0, 5.0, 5.0)
    assert (a1.hold_time_min_s, a1.hold_time_max_s) == (0.25, 1.0)
    assert a1.mode_probabilities == (0.50, 0.30, 0.20)
    assert a1.pulse_on_fraction == pytest.approx(0.20)
    assert a1.curriculum_steps == 75_000


@pytest.mark.parametrize("stage", ["", "a0", "A2", None, 0])
def test_unknown_stage_is_rejected(stage):
    with pytest.raises(ValueError, match="stage must be 'A0' or 'A1'"):
        stage_disturbance_cfg(stage)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("force_limit_n", (1.0, 2.0), "force_limit_n"),
        ("force_limit_n", (1.0, 0.0, 2.0), "force_limit_n"),
        ("force_limit_n", (1.0, math.inf, 2.0), "force_limit_n"),
        ("torque_limit_nm", (1.0, -1.0, 2.0), "torque_limit_nm"),
        ("hold_time_min_s", 0.0, "hold times"),
        ("hold_time_max_s", 0.5, "hold times"),
        ("curriculum_start_scale", 0.0, "curriculum_start_scale"),
        ("curriculum_start_scale", 1.1, "curriculum_start_scale"),
        ("curriculum_steps", 0, "curriculum_steps"),
        ("curriculum_steps", True, "curriculum_steps"),
        ("mode_probabilities", (0.5, 0.5, 0.5), "mode_probabilities"),
        ("mode_probabilities", (1.0, -0.1, 0.1), "mode_probabilities"),
        ("pulse_on_fraction", 0.0, "pulse_on_fraction"),
        ("pulse_on_fraction", 1.1, "pulse_on_fraction"),
    ],
)
def test_invalid_configuration_is_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(stage_disturbance_cfg("A0"), **{field: value})


def test_a0_scheduler_is_seeded_bounded_and_independent():
    cfg = stage_disturbance_cfg("A0")
    left = M1PandaDisturbanceScheduler(cfg, 8, "cpu", 0.02, seed=7)
    right = M1PandaDisturbanceScheduler(cfg, 8, "cpu", 0.02, seed=7)

    first = left.advance()

    assert torch.equal(first, right.advance())
    assert first.shape == (8, 6)
    assert torch.all(first[:, :3].abs() <= 2.5 + 1.0e-6)
    assert torch.all(first[:, 3:].abs() <= 0.5 + 1.0e-6)
    assert left.remaining_steps.unique().numel() > 1
    assert left.curriculum_scale == pytest.approx(0.25 + 0.75 / 50_000)


def test_scheduler_duration_uses_inclusive_ceil_step_bounds():
    cfg = replace(
        stage_disturbance_cfg("A0"),
        hold_time_min_s=0.021,
        hold_time_max_s=0.061,
    )
    scheduler = M1PandaDisturbanceScheduler(cfg, 128, "cpu", 0.02, seed=11)

    scheduler.advance()

    assert int(scheduler.duration_steps.min()) == 2
    assert int(scheduler.duration_steps.max()) == 4
    assert torch.equal(scheduler.remaining_steps, scheduler.duration_steps - 1)


def _single_mode_scheduler(mode_probabilities, *, duration_steps=5, pulse_fraction=0.2):
    step_dt = 0.02
    cfg = replace(
        stage_disturbance_cfg("A1"),
        hold_time_min_s=duration_steps * step_dt,
        hold_time_max_s=duration_steps * step_dt,
        mode_probabilities=mode_probabilities,
        pulse_on_fraction=pulse_fraction,
    )
    return M1PandaDisturbanceScheduler(cfg, 2, "cpu", step_dt, seed=19)


def test_hold_mode_keeps_sampled_target_for_the_segment():
    scheduler = _single_mode_scheduler((1.0, 0.0, 0.0), duration_steps=3)

    first = scheduler.advance()
    target = scheduler.target_wrench_b

    assert torch.equal(first, target)
    assert torch.equal(scheduler.advance(), target)
    assert torch.equal(scheduler.advance(), target)


def test_ramp_mode_interpolates_from_previous_value_to_target():
    scheduler = _single_mode_scheduler((0.0, 1.0, 0.0), duration_steps=2)

    halfway = scheduler.advance()
    target = scheduler.target_wrench_b

    assert torch.allclose(halfway, target * 0.5)
    assert torch.allclose(scheduler.advance(), target)


def test_pulse_mode_is_on_only_for_configured_leading_fraction():
    scheduler = _single_mode_scheduler(
        (0.0, 0.0, 1.0), duration_steps=5, pulse_fraction=0.2
    )

    first = scheduler.advance()
    target = scheduler.target_wrench_b
    later = torch.stack([scheduler.advance() for _ in range(4)])

    assert torch.equal(first, target)
    assert torch.equal(later, torch.zeros_like(later))


def test_curriculum_reaches_full_scale_and_stays_there():
    cfg = replace(
        stage_disturbance_cfg("A0"),
        hold_time_min_s=0.02,
        hold_time_max_s=0.02,
        curriculum_steps=2,
    )
    scheduler = M1PandaDisturbanceScheduler(cfg, 1, "cpu", 0.02, seed=1)

    assert scheduler.curriculum_scale == pytest.approx(0.25)
    scheduler.advance()
    assert scheduler.curriculum_scale == pytest.approx(0.625)
    scheduler.advance()
    assert scheduler.curriculum_scale == pytest.approx(1.0)
    scheduler.advance()
    assert scheduler.curriculum_scale == pytest.approx(1.0)


def test_reset_clears_only_selected_environments_without_rewinding_curriculum():
    scheduler = M1PandaDisturbanceScheduler(
        stage_disturbance_cfg("A1"), 4, "cpu", 0.02, seed=3
    )
    scheduler.advance()
    before = scheduler.current_wrench_b
    scale_before = scheduler.curriculum_scale

    scheduler.reset([1, 3])

    assert torch.equal(scheduler.current_wrench_b[[1, 3]], torch.zeros(2, 6))
    assert torch.equal(scheduler.current_wrench_b[[0, 2]], before[[0, 2]])
    assert torch.equal(
        scheduler.remaining_steps[[1, 3]], torch.zeros(2, dtype=torch.long)
    )
    assert scheduler.curriculum_scale == scale_before


def test_diagnostic_properties_return_clones():
    scheduler = M1PandaDisturbanceScheduler(
        stage_disturbance_cfg("A0"), 2, "cpu", 0.02, seed=2
    )
    scheduler.advance()

    current = scheduler.current_wrench_b
    duration = scheduler.duration_steps
    current.zero_()
    duration.zero_()

    assert not torch.equal(scheduler.current_wrench_b, current)
    assert not torch.equal(scheduler.duration_steps, duration)


@pytest.mark.parametrize("num_envs", [0, -1, True, 1.5])
def test_scheduler_rejects_invalid_num_envs(num_envs):
    with pytest.raises(ValueError, match="num_envs"):
        M1PandaDisturbanceScheduler(
            stage_disturbance_cfg("A0"), num_envs, "cpu", 0.02, seed=0
        )


@pytest.mark.parametrize("step_dt", [0.0, -0.1, math.inf, True])
def test_scheduler_rejects_invalid_step_dt(step_dt):
    with pytest.raises(ValueError, match="step_dt"):
        M1PandaDisturbanceScheduler(
            stage_disturbance_cfg("A0"), 1, "cpu", step_dt, seed=0
        )


@pytest.mark.parametrize(
    "env_ids", [[-1], [4], [True], [1.5], torch.tensor([[1]])]
)
def test_reset_rejects_invalid_indices_without_mutating_state(env_ids):
    scheduler = M1PandaDisturbanceScheduler(
        stage_disturbance_cfg("A0"), 4, "cpu", 0.02, seed=0
    )
    scheduler.advance()
    before = scheduler.current_wrench_b

    with pytest.raises((TypeError, ValueError, IndexError)):
        scheduler.reset(env_ids)

    assert torch.equal(scheduler.current_wrench_b, before)


def test_base_frame_wrench_is_converted_to_rotated_hand_local_axes():
    half = 2.0**-0.5
    base_quat_w = torch.tensor([[half, 0.0, 0.0, half]])
    hand_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    force_b = torch.tensor([[20.0, 0.0, 0.0]])
    torque_b = torch.tensor([[0.0, 5.0, 0.0]])

    force_h, torque_h = base_wrench_to_body_local(
        force_b, torque_b, base_quat_w, hand_quat_w
    )

    assert torch.allclose(force_h, torch.tensor([[0.0, -20.0, 0.0]]), atol=1e-5)
    assert torch.allclose(torque_h, torch.tensor([[5.0, 0.0, 0.0]]), atol=1e-5)


def test_wrench_conversion_supports_batches_and_preserves_inputs():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    force = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    torque = -force
    force_before = force.clone()
    torque_before = torque.clone()

    force_local, torque_local = base_wrench_to_body_local(
        force, torque, identity, identity
    )

    assert torch.equal(force_local, force)
    assert torch.equal(torque_local, torque)
    assert torch.equal(force, force_before)
    assert torch.equal(torque, torque_before)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("force_b", torch.zeros(1, 4), ValueError),
        ("torque_b", torch.zeros(2, 3), ValueError),
        ("base_quat_w", torch.zeros(1, 3), ValueError),
        ("body_quat_w", torch.tensor([[math.nan, 0.0, 0.0, 0.0]]), ValueError),
        ("force_b", torch.zeros(1, 3, dtype=torch.float64), TypeError),
    ],
)
def test_wrench_conversion_rejects_invalid_inputs(field, value, error):
    inputs = {
        "force_b": torch.zeros(1, 3),
        "torque_b": torch.zeros(1, 3),
        "base_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "body_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    inputs[field] = value

    with pytest.raises(error):
        base_wrench_to_body_local(**inputs)


def test_clear_external_wrench_uses_empty_tensor_contract():
    class Robot:
        device = "cpu"
        has_external_wrench = True

        def __init__(self):
            self.calls = 0

        def set_external_force_and_torque(self, forces, torques):
            self.calls += 1
            assert tuple(forces.shape) == (0, 3)
            assert tuple(torques.shape) == (0, 3)
            self.has_external_wrench = False

    robot = Robot()
    clear_external_wrench(robot)
    assert robot.calls == 1
    assert robot.has_external_wrench is False


def test_clear_external_wrench_accepts_only_known_isaaclab_shape_bug():
    class Robot:
        device = "cpu"
        has_external_wrench = True

        def set_external_force_and_torque(self, forces, torques):
            self.has_external_wrench = False
            raise RuntimeError(
                "shape mismatch: value tensor of shape [0] cannot be broadcast "
                "to indexing result of shape [29, 3]"
            )

    clear_external_wrench(Robot())


def test_clear_external_wrench_rejects_unrelated_runtime_error():
    class Robot:
        device = "cpu"
        has_external_wrench = False

        def set_external_force_and_torque(self, forces, torques):
            raise RuntimeError("unrelated actuator buffer failure")

    with pytest.raises(RuntimeError, match="unrelated actuator buffer"):
        clear_external_wrench(Robot())
