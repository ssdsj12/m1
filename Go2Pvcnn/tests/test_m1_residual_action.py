import pytest
import torch

from go2_pvcnn.tasks.m1_residual_action import (
    M1ResidualActionComposer,
    M1ResidualActionComposerCfg,
)


def test_default_configuration_and_state_contract():
    cfg = M1ResidualActionComposerCfg()
    composer = M1ResidualActionComposer(cfg, num_envs=3, device="cpu")

    assert cfg.leg_action_scale == pytest.approx(0.25)
    assert cfg.wheel_action_scale == pytest.approx(8.0)
    assert cfg.leg_residual_limit_rad == pytest.approx(0.05)
    assert cfg.wheel_residual_limit_rad_s == pytest.approx(1.0)
    assert cfg.leg_slew_limit_rad_per_step == pytest.approx(0.01)
    assert cfg.wheel_slew_limit_rad_s_per_step == pytest.approx(0.2)
    assert composer.physical_residual.shape == (3, 16)
    assert composer.physical_residual.dtype == torch.float32
    assert not composer.physical_residual.any()
    assert not composer.amplitude_clipped.any()
    assert not composer.slew_clipped.any()


def test_diagnostic_properties_are_clones():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )

    leaked_physical = composer.physical_residual
    leaked_amplitude = composer.amplitude_clipped
    leaked_slew = composer.slew_clipped
    leaked_physical.fill_(7.0)
    leaked_amplitude.fill_(True)
    leaked_slew.fill_(True)

    assert not composer.physical_residual.any()
    assert not composer.amplitude_clipped.any()
    assert not composer.slew_clipped.any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"leg_action_scale": 0.0}, "leg_action_scale must be finite and > 0"),
        ({"wheel_action_scale": -1.0}, "wheel_action_scale must be finite and > 0"),
        ({"leg_residual_limit_rad": -0.1}, "leg_residual_limit_rad must be finite and >= 0"),
        (
            {"wheel_residual_limit_rad_s": float("inf")},
            "wheel_residual_limit_rad_s must be finite and >= 0",
        ),
        (
            {"leg_slew_limit_rad_per_step": float("nan")},
            "leg_slew_limit_rad_per_step must be finite and >= 0",
        ),
        (
            {"wheel_slew_limit_rad_s_per_step": -0.1},
            "wheel_slew_limit_rad_s_per_step must be finite and >= 0",
        ),
    ],
)
def test_configuration_rejects_invalid_numbers(kwargs, message):
    with pytest.raises(ValueError, match=message):
        M1ResidualActionComposerCfg(**kwargs)


@pytest.mark.parametrize("num_envs", [0, -1, True, 1.5])
def test_constructor_rejects_invalid_num_envs(num_envs):
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        M1ResidualActionComposer(
            M1ResidualActionComposerCfg(), num_envs=num_envs, device="cpu"
        )


def test_constructor_rejects_non_floating_dtype():
    with pytest.raises(TypeError, match="dtype must be a floating torch.dtype"):
        M1ResidualActionComposer(
            M1ResidualActionComposerCfg(),
            num_envs=1,
            device="cpu",
            dtype=torch.int64,
        )


def test_zero_residual_preserves_base_action_after_initialization():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )
    base = torch.randn(2, 16)

    result = composer.compose(base, torch.zeros_like(base))

    assert torch.equal(result, base)


def test_maps_leg_and_wheel_physical_residuals_to_action_space():
    cfg = M1ResidualActionComposerCfg(
        leg_slew_limit_rad_per_step=1.0,
        wheel_slew_limit_rad_s_per_step=10.0,
    )
    composer = M1ResidualActionComposer(cfg, num_envs=1, device="cpu")
    base = torch.zeros(1, 16)
    residual = torch.full((1, 16), 0.5)

    result = composer.compose(base, residual)

    assert torch.allclose(result[:, :12], torch.full((1, 12), 0.1))
    assert torch.allclose(result[:, 12:], torch.full((1, 4), 0.0625))
    assert torch.allclose(
        composer.physical_residual[:, :12], torch.full((1, 12), 0.025)
    )
    assert torch.allclose(
        composer.physical_residual[:, 12:], torch.full((1, 4), 0.5)
    )


def test_clips_normalized_amplitude_and_reports_mask():
    cfg = M1ResidualActionComposerCfg(
        leg_slew_limit_rad_per_step=1.0,
        wheel_slew_limit_rad_s_per_step=10.0,
    )
    composer = M1ResidualActionComposer(cfg, num_envs=1, device="cpu")
    residual = torch.tensor([[2.0] + [0.0] * 11 + [-3.0] + [0.0] * 3])

    composer.compose(torch.zeros_like(residual), residual)

    assert composer.physical_residual[0, 0] == pytest.approx(0.05)
    assert composer.physical_residual[0, 12] == pytest.approx(-1.0)
    assert composer.amplitude_clipped[0, 0]
    assert composer.amplitude_clipped[0, 12]
    assert composer.amplitude_clipped.sum().item() == 2


def test_slew_limits_positive_and_negative_transitions():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=1, device="cpu"
    )
    base = torch.zeros(1, 16)

    first = composer.compose(base, torch.ones_like(base))
    second = composer.compose(base, -torch.ones_like(base))

    assert torch.allclose(first[:, :12], torch.full((1, 12), 0.04))
    assert torch.allclose(first[:, 12:], torch.full((1, 4), 0.025))
    assert torch.allclose(second, torch.zeros_like(second))
    assert composer.slew_clipped.all()


def test_current_step_keeps_gradients_and_history_is_detached():
    cfg = M1ResidualActionComposerCfg(
        leg_slew_limit_rad_per_step=1.0,
        wheel_slew_limit_rad_s_per_step=10.0,
    )
    composer = M1ResidualActionComposer(cfg, num_envs=1, device="cpu")
    base = torch.zeros(1, 16, requires_grad=True)
    residual = torch.full((1, 16), 0.5, requires_grad=True)

    result = composer.compose(base, residual)
    result.sum().backward()

    assert torch.equal(base.grad, torch.ones_like(base))
    assert torch.allclose(residual.grad[:, :12], torch.full((1, 12), 0.2))
    assert torch.allclose(residual.grad[:, 12:], torch.full((1, 4), 0.125))
    assert not composer.physical_residual.requires_grad


def _snapshot(composer):
    return (
        composer.physical_residual,
        composer.amplitude_clipped,
        composer.slew_clipped,
    )


def test_selective_reset_clears_only_requested_environments():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=3, device="cpu"
    )
    action = torch.ones(3, 16)
    composer.compose(torch.zeros_like(action), action)

    composer.reset(torch.tensor([1], dtype=torch.int64))

    assert composer.physical_residual[0].abs().sum() > 0
    assert not composer.physical_residual[1].any()
    assert composer.physical_residual[2].abs().sum() > 0
    assert not composer.amplitude_clipped[1].any()
    assert not composer.slew_clipped[1].any()


def test_full_reset_clears_all_state_and_diagnostics():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )
    action = torch.full((2, 16), 2.0)
    composer.compose(torch.zeros_like(action), action)

    composer.reset()

    assert not composer.physical_residual.any()
    assert not composer.amplitude_clipped.any()
    assert not composer.slew_clipped.any()


@pytest.mark.parametrize(
    ("env_ids", "error", "message"),
    [
        (torch.tensor([True, False]), TypeError, "env_ids must contain integers"),
        (
            torch.tensor([[0]], dtype=torch.int64),
            ValueError,
            "env_ids must be one-dimensional",
        ),
        ([0, 1.5], TypeError, "env_ids must contain integers"),
        ([-1], IndexError, "env_ids contains out-of-range index -1"),
        ([2], IndexError, "env_ids contains out-of-range index 2"),
    ],
)
def test_invalid_reset_is_atomic(env_ids, error, message):
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=2, device="cpu"
    )
    action = torch.ones(2, 16)
    composer.compose(torch.zeros_like(action), action)
    before = _snapshot(composer)

    with pytest.raises(error, match=message):
        composer.reset(env_ids)

    after = _snapshot(composer)
    assert all(
        torch.equal(left, right) for left, right in zip(before, after, strict=True)
    )


@pytest.mark.parametrize(
    ("base", "residual", "error", "message"),
    [
        (
            torch.zeros(1, 15),
            torch.zeros(1, 16),
            ValueError,
            "base_action must have shape",
        ),
        (
            torch.zeros(1, 16),
            torch.zeros(1, 15),
            ValueError,
            "normalized_residual must have shape",
        ),
        (
            torch.zeros(1, 16, dtype=torch.float64),
            torch.zeros(1, 16),
            TypeError,
            "base_action must have dtype",
        ),
        (
            torch.zeros(1, 16),
            torch.zeros(1, 16, dtype=torch.float64),
            TypeError,
            "normalized_residual must have dtype",
        ),
        (
            torch.zeros(1, 16, device="meta"),
            torch.zeros(1, 16),
            ValueError,
            "base_action must be on device",
        ),
        (
            torch.zeros(1, 16),
            torch.zeros(1, 16, device="meta"),
            ValueError,
            "normalized_residual must be on device",
        ),
        (
            torch.full((1, 16), float("nan")),
            torch.zeros(1, 16),
            ValueError,
            "base_action must contain only finite values",
        ),
        (
            torch.zeros(1, 16),
            torch.full((1, 16), float("inf")),
            ValueError,
            "normalized_residual must contain only finite values",
        ),
    ],
)
def test_invalid_compose_is_atomic(base, residual, error, message):
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=1, device="cpu"
    )
    valid = torch.ones(1, 16)
    composer.compose(torch.zeros_like(valid), valid)
    before = _snapshot(composer)

    with pytest.raises(error, match=message):
        composer.compose(base, residual)

    after = _snapshot(composer)
    assert all(
        torch.equal(left, right) for left, right in zip(before, after, strict=True)
    )


def test_non_tensor_compose_input_has_clear_error():
    composer = M1ResidualActionComposer(
        M1ResidualActionComposerCfg(), num_envs=1, device="cpu"
    )
    with pytest.raises(TypeError, match="base_action must be a torch.Tensor"):
        composer.compose([[0.0] * 16], torch.zeros(1, 16))
