import torch

from go2_pvcnn.control.m1_panda_coordination.joint_torque_wrench import (
    wrench_from_joint_torque,
)


def test_joint_torque_wrench_recovers_full_rank_wrench():
    jacobian = torch.cat((torch.eye(6, dtype=torch.float64), torch.zeros(6, 1, dtype=torch.float64)), dim=1)
    wrench = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64)
    torque = jacobian.T @ wrench
    assert torch.allclose(wrench_from_joint_torque(jacobian, torque, damping=0.0), wrench)


def test_joint_torque_wrench_rejects_wrong_shapes():
    with torch.no_grad():
        try:
            wrench_from_joint_torque(torch.zeros(6, 6, dtype=torch.float64), torch.zeros(7, dtype=torch.float64))
        except ValueError:
            pass
        else:
            raise AssertionError("expected shape validation")
