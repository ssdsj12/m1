import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.arm_mpc import (
    ARM_DOF,
    ARM_TASK_DOF,
    ArmMpcCfg,
    ArmMpcInput,
    condense_arm_dynamics,
    rollout_linearized_arm,
)


DTYPE = torch.float64


def _input(**overrides) -> ArmMpcInput:
    horizon = 20
    values = {
        "q": torch.zeros(ARM_DOF, dtype=DTYPE),
        "qd": torch.zeros(ARM_DOF, dtype=DTYPE),
        "ee_pose_b": torch.zeros(ARM_TASK_DOF, dtype=DTYPE),
        "ee_twist_b": torch.zeros(ARM_TASK_DOF, dtype=DTYPE),
        "target_pose_b": torch.zeros((horizon, ARM_TASK_DOF), dtype=DTYPE),
        "target_twist_b": torch.zeros((horizon, ARM_TASK_DOF), dtype=DTYPE),
        "jacobian_b": torch.zeros((ARM_TASK_DOF, ARM_DOF), dtype=DTYPE),
        "arm_mass_matrix": torch.eye(ARM_DOF, dtype=DTYPE),
        "arm_bias": torch.zeros(ARM_DOF, dtype=DTYPE),
        "base_arm_coupling": torch.zeros((ARM_TASK_DOF, ARM_DOF), dtype=DTYPE),
        "q_min": -torch.ones(ARM_DOF, dtype=DTYPE),
        "q_max": torch.ones(ARM_DOF, dtype=DTYPE),
        "qd_max": 2.0 * torch.ones(ARM_DOF, dtype=DTYPE),
        "qdd_max": 4.0 * torch.ones(ARM_DOF, dtype=DTYPE),
        "effort_max": 20.0 * torch.ones(ARM_DOF, dtype=DTYPE),
    }
    values.update(overrides)
    return ArmMpcInput(**values)


def test_default_cfg_freezes_50_hz_twenty_node_horizon():
    cfg = ArmMpcCfg()

    assert cfg.dt == pytest.approx(0.02)
    assert cfg.horizon_steps == 20
    assert cfg.horizon_seconds == pytest.approx(0.4)


def test_input_accepts_only_exact_cpu_float64_finite_contract():
    sample = _input()

    assert sample.q.dtype == DTYPE
    assert sample.q.device.type == "cpu"

    with pytest.raises(TypeError, match="q must have dtype"):
        _input(q=torch.zeros(ARM_DOF, dtype=torch.float32))
    with pytest.raises(ValueError, match="jacobian_b must have shape"):
        _input(jacobian_b=torch.zeros((ARM_DOF, ARM_TASK_DOF), dtype=DTYPE))
    with pytest.raises(ValueError, match="target_pose_b must have shape"):
        _input(target_pose_b=torch.zeros((19, ARM_TASK_DOF), dtype=DTYPE))
    with pytest.raises(ValueError, match="arm_mass_matrix must contain only finite"):
        matrix = torch.eye(ARM_DOF, dtype=DTYPE)
        matrix[0, 0] = torch.nan
        _input(arm_mass_matrix=matrix)


def test_input_rejects_inconsistent_position_and_positive_limit_contracts():
    with pytest.raises(ValueError, match="q_min must be strictly below q_max"):
        _input(q_min=torch.ones(ARM_DOF, dtype=DTYPE))
    with pytest.raises(ValueError, match="qd_max must be strictly positive"):
        _input(qd_max=torch.zeros(ARM_DOF, dtype=DTYPE))
    with pytest.raises(ValueError, match="qdd_max must be strictly positive"):
        _input(qdd_max=torch.zeros(ARM_DOF, dtype=DTYPE))
    with pytest.raises(ValueError, match="effort_max must be strictly positive"):
        _input(effort_max=torch.zeros(ARM_DOF, dtype=DTYPE))


def test_rollout_matches_constant_acceleration_discretization():
    q = torch.arange(ARM_DOF, dtype=DTYPE) * 0.1
    qd = torch.arange(ARM_DOF, dtype=DTYPE) * -0.05
    qdd = torch.full((2, ARM_DOF), 0.25, dtype=DTYPE)
    jacobian = torch.arange(ARM_TASK_DOF * ARM_DOF, dtype=DTYPE).reshape(
        ARM_TASK_DOF, ARM_DOF
    ) / 100.0

    rollout = rollout_linearized_arm(q, qd, qdd, jacobian, dt=0.02)

    q1 = q + 0.02 * qd + 0.5 * 0.02**2 * qdd[0]
    qd1 = qd + 0.02 * qdd[0]
    q2 = q1 + 0.02 * qd1 + 0.5 * 0.02**2 * qdd[1]
    qd2 = qd1 + 0.02 * qdd[1]
    assert torch.allclose(rollout.q, torch.stack((q1, q2)))
    assert torch.allclose(rollout.qd, torch.stack((qd1, qd2)))
    assert torch.allclose(rollout.pose_delta_b[0], jacobian @ (q1 - q))
    assert torch.allclose(rollout.pose_delta_b[1], jacobian @ (q2 - q))
    assert torch.allclose(rollout.ee_twist_b, torch.stack((jacobian @ qd1, jacobian @ qd2)))


def test_zero_acceleration_preserves_constant_velocity():
    q = torch.linspace(-0.3, 0.3, ARM_DOF, dtype=DTYPE)
    qd = torch.linspace(0.1, 0.7, ARM_DOF, dtype=DTYPE)
    qdd = torch.zeros((20, ARM_DOF), dtype=DTYPE)
    jacobian = torch.eye(ARM_TASK_DOF, ARM_DOF, dtype=DTYPE)

    rollout = rollout_linearized_arm(q, qd, qdd, jacobian, dt=0.02)

    expected_qd = qd.expand(20, -1)
    elapsed = 0.02 * torch.arange(1, 21, dtype=DTYPE).unsqueeze(1)
    expected_q = q + elapsed * qd
    assert torch.equal(rollout.qd, expected_qd)
    assert torch.allclose(rollout.q, expected_q)


def test_condensed_affine_maps_match_iterative_rollout():
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(ARM_DOF, generator=generator, dtype=DTYPE)
    qd = torch.randn(ARM_DOF, generator=generator, dtype=DTYPE)
    qdd = torch.randn((5, ARM_DOF), generator=generator, dtype=DTYPE)
    jacobian = torch.randn(
        (ARM_TASK_DOF, ARM_DOF), generator=generator, dtype=DTYPE
    )

    condensed = condense_arm_dynamics(q, qd, horizon_steps=5, dt=0.02)
    rollout = rollout_linearized_arm(q, qd, qdd, jacobian, dt=0.02)

    flat_qdd = qdd.reshape(-1)
    assert torch.allclose(
        condensed.q_offset + condensed.q_from_qdd @ flat_qdd,
        rollout.q.reshape(-1),
    )
    assert torch.allclose(
        condensed.qd_offset + condensed.qd_from_qdd @ flat_qdd,
        rollout.qd.reshape(-1),
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"q": torch.zeros(6, dtype=DTYPE)}, "q must have shape"),
        ({"qdd": torch.zeros((20, 6), dtype=DTYPE)}, "qdd must have shape"),
        ({"jacobian_b": torch.zeros((7, 6), dtype=DTYPE)}, "jacobian_b must have shape"),
        ({"dt": 0.0}, "dt must be finite and positive"),
    ],
)
def test_rollout_rejects_malformed_inputs(kwargs, error):
    values = {
        "q": torch.zeros(ARM_DOF, dtype=DTYPE),
        "qd": torch.zeros(ARM_DOF, dtype=DTYPE),
        "qdd": torch.zeros((20, ARM_DOF), dtype=DTYPE),
        "jacobian_b": torch.zeros((ARM_TASK_DOF, ARM_DOF), dtype=DTYPE),
        "dt": 0.02,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=error):
        rollout_linearized_arm(**values)
