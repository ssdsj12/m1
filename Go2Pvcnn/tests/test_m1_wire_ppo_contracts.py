import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.wire_ppo_contracts import (
    WIRE_PPO_ACTION_DIM,
    WireLatentCommand,
    WirePpoObservationSpec,
    decode_wire_latent,
    validate_wire_ppo_transition,
)


def test_wire_action_contract_is_exactly_fifteen_normalized_values():
    assert WIRE_PPO_ACTION_DIM == 15
    action = torch.linspace(-1.0, 1.0, WIRE_PPO_ACTION_DIM)
    command = decode_wire_latent(action, device="cpu", dtype=torch.float32)
    assert isinstance(command, WireLatentCommand)
    assert command.as_vector.shape == (WIRE_PPO_ACTION_DIM,)
    assert command.as_vector.dtype == torch.float32
    assert command.as_vector.device.type == "cpu"
    assert torch.equal(command.as_vector, action)


def test_decode_is_deterministic_and_respects_requested_device_and_dtype():
    action = torch.arange(WIRE_PPO_ACTION_DIM, dtype=torch.float64) / 7.0 - 1.0
    first = decode_wire_latent(action, device=torch.device("cpu"), dtype=torch.float32)
    second = decode_wire_latent(action, device="cpu", dtype=torch.float32)
    assert torch.equal(first.as_vector, second.as_vector)
    assert torch.equal(first.palm_delta, first.as_vector[:6])
    assert torch.equal(first.hand_delta, first.as_vector[6:12])
    assert torch.equal(first.phase_force, first.as_vector[12:15])


@pytest.mark.parametrize(
    "action",
    [
        torch.zeros(14),
        torch.zeros(16),
        torch.tensor([float("nan")] + [0.0] * 14),
        torch.tensor([float("inf")] + [0.0] * 14),
        torch.full((15,), 1.01),
        torch.full((15,), -1.01),
        torch.zeros(15, dtype=torch.int64),
    ],
)
def test_decode_rejects_malformed_action_before_decoding(action):
    with pytest.raises((TypeError, ValueError)):
        decode_wire_latent(action, device="cpu", dtype=torch.float32)


def test_observation_spec_validates_normalized_shape_dtype_device_and_finite_values():
    spec = WirePpoObservationSpec()
    observation = torch.zeros(spec.observation_dim, dtype=torch.float32)
    spec.validate(observation)
    with pytest.raises(ValueError):
        spec.validate(torch.zeros(spec.observation_dim + 1))
    with pytest.raises(ValueError):
        spec.validate(torch.full((spec.observation_dim,), float("nan")))
    with pytest.raises(ValueError):
        spec.validate(torch.full((spec.observation_dim,), 1.01))
    with pytest.raises(TypeError):
        spec.validate(torch.zeros(spec.observation_dim, dtype=torch.int64))


def test_transition_validation_accepts_valid_cpu_transition():
    spec = WirePpoObservationSpec()
    observation = torch.zeros(spec.observation_dim, dtype=torch.float32)
    action = torch.zeros(WIRE_PPO_ACTION_DIM, dtype=torch.float32)
    validate_wire_ppo_transition(observation, action, torch.tensor(0.25), False)
    validate_wire_ppo_transition(observation, action, 0.25, True)


@pytest.mark.parametrize(
    "observation, action, reward, done",
    [
        (torch.zeros(1), torch.zeros(15), torch.tensor(0.0), False),
        (torch.full((WirePpoObservationSpec().observation_dim,), float("inf")), torch.zeros(15), 0.0, False),
        (torch.zeros(WirePpoObservationSpec().observation_dim), torch.full((15,), 1.1), 0.0, False),
        (torch.zeros(WirePpoObservationSpec().observation_dim), torch.zeros(15), torch.tensor(float("nan")), False),
        (torch.zeros(WirePpoObservationSpec().observation_dim), torch.zeros(15), torch.zeros(2), False),
        (torch.zeros(WirePpoObservationSpec().observation_dim), torch.zeros(15), 0.0, 1),
    ],
)
def test_transition_rejects_invalid_inputs_atomically(observation, action, reward, done):
    with pytest.raises((TypeError, ValueError)):
        validate_wire_ppo_transition(observation, action, reward, done)
