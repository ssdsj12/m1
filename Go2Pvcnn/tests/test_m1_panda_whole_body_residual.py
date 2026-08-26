import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    MountWrenchFeedback,
    MountWrenchFeedbackCfg,
    RESIDUAL_NAMES,
    WholeBodyResidualCfg,
    WholeBodyResidualComposer,
)


def test_residual_configuration_freezes_channel_order_and_limits():
    cfg = WholeBodyResidualCfg()

    assert RESIDUAL_NAMES == (
        "Fx",
        "Fy",
        "Fz",
        "Mx",
        "My",
        "Mz",
        "delta_height",
        "delta_stance",
    )
    assert cfg.physical_limits == (30.0, 30.0, 50.0, 15.0, 15.0, 8.0, 0.04, 0.08)
    assert cfg.slew_fraction_per_step == pytest.approx(0.05)


def test_residual_channel_scaling_and_slew_reach_exact_physical_limits():
    composer = WholeBodyResidualComposer(2, "cpu", torch.float64)
    normalized = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    command, diagnostics = composer.step(normalized)

    assert torch.allclose(
        command.physical[0],
        torch.tensor(
            [1.5, -1.5, 2.5, -0.75, 0.75, -0.4, 0.002, -0.004],
            dtype=torch.float64,
        ),
    )
    assert diagnostics.slew_saturated[0].all()
    assert not diagnostics.amplitude_saturated.any()
    for _ in range(19):
        command, diagnostics = composer.step(normalized)
    expected = torch.tensor(
        [30.0, -30.0, 50.0, -15.0, 15.0, -8.0, 0.04, -0.08],
        dtype=torch.float64,
    )
    assert torch.allclose(command.physical[0], expected)
    assert torch.equal(command.wrench_b, command.physical[:, :6])
    assert torch.equal(command.delta_height, command.physical[:, 6])
    assert torch.equal(command.delta_stance, command.physical[:, 7])


def test_residual_amplitude_clip_and_diagnostics_are_read_only_clones():
    composer = WholeBodyResidualComposer(1, "cpu", torch.float32)

    command, diagnostics = composer.step(torch.full((1, 8), 2.0))

    assert diagnostics.amplitude_saturated.all()
    command.physical.zero_()
    diagnostics.normalized_clipped.zero_()
    next_command, _ = composer.step(torch.ones(1, 8))
    assert torch.all(next_command.physical > 0.0)


def test_residual_reset_is_selective():
    composer = WholeBodyResidualComposer(3, "cpu", torch.float64)
    for _ in range(4):
        composer.step(torch.ones(3, 8, dtype=torch.float64))
    before = composer.previous_physical

    composer.reset(torch.tensor([1], dtype=torch.long))

    after = composer.previous_physical
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[1], torch.zeros(8, dtype=torch.float64))
    assert torch.equal(after[2], before[2])


def test_residual_rejects_bad_shape_dtype_device_and_nonfinite_atomically():
    composer = WholeBodyResidualComposer(2, "cpu", torch.float64)
    composer.step(torch.ones(2, 8, dtype=torch.float64))
    before = composer.previous_physical

    with pytest.raises(ValueError, match="shape"):
        composer.step(torch.ones(2, 7, dtype=torch.float64))
    with pytest.raises(TypeError, match="dtype"):
        composer.step(torch.ones(2, 8, dtype=torch.float32))
    invalid = torch.ones(2, 8, dtype=torch.float64)
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        composer.step(invalid)

    assert torch.equal(composer.previous_physical, before)


def test_mount_feedback_initializes_filter_without_zero_state_impulse():
    feedback = MountWrenchFeedback(2, "cpu", torch.float64)
    measured = torch.tensor(
        [[10.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0, 4.0]],
        dtype=torch.float64,
    )

    output = feedback.update(measured, torch.zeros_like(measured))

    expected = torch.zeros_like(measured)
    expected[:, :3] = -0.15 * measured[:, :3]
    expected[:, 3:] = -0.10 * measured[:, 3:]
    assert torch.allclose(output, expected)
    assert torch.equal(feedback.filtered_wrench, measured)


def test_mount_feedback_filters_then_clips_combined_command():
    feedback = MountWrenchFeedback(
        1,
        "cpu",
        torch.float64,
        MountWrenchFeedbackCfg(filter_alpha=0.25),
    )
    feedback.update(torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 6, dtype=torch.float64))

    output = feedback.update(
        torch.tensor([[100.0, -100.0, 100.0, 100.0, -100.0, 100.0]], dtype=torch.float64),
        torch.tensor([[30.0, -30.0, 50.0, 15.0, -15.0, 8.0]], dtype=torch.float64),
    )

    assert torch.equal(feedback.filtered_wrench, torch.full((1, 6), 25.0, dtype=torch.float64).mul(
        torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0, 1.0]], dtype=torch.float64)
    ))
    assert torch.all(output.abs() <= torch.tensor([[30.0, 30.0, 50.0, 15.0, 15.0, 8.0]], dtype=torch.float64))


def test_mount_feedback_zero_gain_and_zero_residual_are_exact_zero():
    feedback = MountWrenchFeedback(
        1,
        "cpu",
        torch.float64,
        MountWrenchFeedbackCfg(force_gain=0.0, moment_gain=0.0),
    )

    output = feedback.update(torch.ones(1, 6, dtype=torch.float64), torch.zeros(1, 6, dtype=torch.float64))

    assert torch.equal(output, torch.zeros(1, 6, dtype=torch.float64))


def test_mount_feedback_bias_warmup_and_selective_reset():
    feedback = MountWrenchFeedback(
        2,
        "cpu",
        torch.float64,
        MountWrenchFeedbackCfg(bias_warmup_samples=2),
    )
    measurement = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    feedback.update(measurement, torch.zeros_like(measurement))
    output = feedback.update(measurement, torch.zeros_like(measurement))
    assert torch.allclose(output, torch.zeros_like(output))

    feedback.reset(torch.tensor([1], dtype=torch.long))

    assert feedback.initialized.tolist() == [True, False]
    assert feedback.bias_sample_count.tolist() == [2, 0]


def test_mount_feedback_rejects_nonfinite_without_mutating_state():
    feedback = MountWrenchFeedback(1, "cpu", torch.float64)
    zeros = torch.zeros(1, 6, dtype=torch.float64)
    feedback.update(zeros, zeros)
    before = feedback.filtered_wrench
    invalid = zeros.clone()
    invalid[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        feedback.update(invalid, zeros)

    assert torch.equal(feedback.filtered_wrench, before)
