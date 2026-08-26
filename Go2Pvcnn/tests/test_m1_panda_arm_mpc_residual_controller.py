import torch

from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    MountWrenchFeedbackCfg,
)
from go2_pvcnn.tasks.m1_panda_residual_wbc_wrapper import (
    M1PandaResidualWbcController,
)


class _Teacher:
    def __init__(self):
        self.kwargs = None

    def step(self, state, **kwargs):
        self.kwargs = kwargs
        return state


def test_controller_combines_negative_mpc_feedforward_feedback_and_rl_before_teacher():
    teacher = _Teacher()
    controller = M1PandaResidualWbcController(
        [teacher],
        device="cpu",
        dtype=torch.float64,
        feedback_cfg=MountWrenchFeedbackCfg(force_gain=0.0, moment_gain=0.0),
    )
    predicted = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]], dtype=torch.float64)

    result = controller.step(
        states=("state",),
        normalized_residual=torch.ones((1, 8), dtype=torch.float64),
        measured_mount_wrench_b=torch.zeros((1, 6), dtype=torch.float64),
        predicted_mount_wrench_b=predicted,
        leg_soft_limits=torch.tensor([[[-1.0, 1.0]] * 12], dtype=torch.float64),
        teacher_kwargs=({"arm_reference": "mpc-reference"},),
    )

    # First composer step is limited to 5% of each frozen physical limit.
    expected_rl = torch.tensor(
        [[1.5, 1.5, 2.5, 0.75, 0.75, 0.4]], dtype=torch.float64
    )
    assert torch.equal(result.correction_wrench_b, -predicted + expected_rl)
    assert torch.equal(
        teacher.kwargs["residual_command"].wrench_b,
        (-predicted + expected_rl)[0],
    )
    assert teacher.kwargs["arm_reference"] == "mpc-reference"
    assert torch.equal(result.predicted_mount_wrench_b, predicted)


def test_omitted_mpc_feedforward_is_exactly_legacy_zero_prediction():
    kwargs = dict(
        states=("state",),
        normalized_residual=torch.zeros((1, 8), dtype=torch.float64),
        measured_mount_wrench_b=torch.zeros((1, 6), dtype=torch.float64),
        leg_soft_limits=torch.tensor([[[-1.0, 1.0]] * 12], dtype=torch.float64),
    )
    old = M1PandaResidualWbcController(
        [_Teacher()], device="cpu", dtype=torch.float64
    ).step(**kwargs)
    explicit = M1PandaResidualWbcController(
        [_Teacher()], device="cpu", dtype=torch.float64
    ).step(
        **kwargs,
        predicted_mount_wrench_b=torch.zeros((1, 6), dtype=torch.float64),
    )

    assert torch.equal(old.applied_residual.physical, explicit.applied_residual.physical)
    assert torch.equal(old.correction_wrench_b, explicit.correction_wrench_b)
