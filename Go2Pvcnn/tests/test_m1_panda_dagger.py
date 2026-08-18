import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.dagger import (
    DaggerStageCfg,
    StudentLossCfg,
    select_dagger_action,
    student_dagger_loss,
)
from go2_pvcnn.control.m1_panda_coordination.student_model import StudentOutput


def _output(batch: int = 8) -> StudentOutput:
    raw_action = torch.linspace(-1.5, 1.5, batch * 23).reshape(batch, 23)
    return StudentOutput(
        wrench_hat=torch.full((batch, 6), 0.25),
        latent=torch.zeros(batch, 32),
        safety_logit=torch.linspace(-1.0, 1.0, batch).unsqueeze(-1),
        raw_action=raw_action,
        action=torch.tanh(raw_action),
    )


def test_safety_override_always_executes_teacher():
    selected = select_dagger_action(
        student=torch.zeros(4, 23),
        teacher=torch.ones(4, 23),
        safe_to_execute_student=torch.tensor([True, False, True, False]),
        cfg=DaggerStageCfg(name="mix-50", teacher_probability=0.5, seed=7),
        rollout_step=3,
    )
    torch.testing.assert_close(
        selected.executed[~selected.safe_to_execute_student], torch.ones(2, 23)
    )
    assert selected.teacher_executed.tolist()[1::2] == [True, True]


def test_fixed_seed_and_rollout_step_are_reproducible():
    args = dict(
        student=torch.zeros(64, 23),
        teacher=torch.ones(64, 23),
        safe_to_execute_student=torch.ones(64, dtype=torch.bool),
        cfg=DaggerStageCfg("mix", 0.25, 11),
        rollout_step=19,
    )
    first = select_dagger_action(**args)
    second = select_dagger_action(**args)
    assert torch.equal(first.executed, second.executed)
    assert torch.equal(first.teacher_executed, second.teacher_executed)


def test_probability_extremes_respect_safety_override():
    student = torch.zeros(3, 23)
    teacher = torch.ones(3, 23)
    safe = torch.tensor([True, False, True])
    all_teacher = select_dagger_action(
        student, teacher, safe, DaggerStageCfg("teacher", 1.0, 1), 0
    )
    assert all_teacher.teacher_executed.tolist() == [True, True, True]
    all_student = select_dagger_action(
        student, teacher, safe, DaggerStageCfg("student", 0.0, 1), 0
    )
    assert all_student.teacher_executed.tolist() == [False, True, False]
    torch.testing.assert_close(all_student.executed[0], student[0])
    torch.testing.assert_close(all_student.executed[1], teacher[1])


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_stage_rejects_invalid_teacher_probability(probability):
    with pytest.raises(ValueError, match="teacher_probability"):
        DaggerStageCfg("bad", probability, 3)


def test_selection_rejects_invalid_actions_shapes_and_mask():
    student = torch.zeros(2, 23)
    teacher = torch.ones(2, 23)
    cfg = DaggerStageCfg("mix", 0.5, 1)
    bad_student = student.clone()
    bad_student[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        select_dagger_action(bad_student, teacher, torch.ones(2, dtype=torch.bool), cfg, 0)
    with pytest.raises(ValueError, match="same shape"):
        select_dagger_action(student, teacher[:1], torch.ones(2, dtype=torch.bool), cfg, 0)
    with pytest.raises(TypeError, match="boolean"):
        select_dagger_action(student, teacher, torch.ones(2), cfg, 0)


def test_dagger_loss_contains_all_s1_terms_and_propagates_gradients():
    output = _output()
    for tensor in (output.raw_action, output.action, output.wrench_hat, output.safety_logit):
        tensor.requires_grad_(True)
    losses = student_dagger_loss(
        output=output,
        target_action=torch.zeros(8, 23),
        target_wrench=torch.zeros(8, 6),
        target_safety=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32),
        hard_mask=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool),
        previous_action=torch.zeros(8, 23),
        cfg=StudentLossCfg(),
    )
    assert set(losses) == {"total", "action", "wrench", "safety", "slew", "saturation"}
    assert all(value.ndim == 0 and bool(torch.isfinite(value)) for value in losses.values())
    losses["total"].backward()
    assert output.action.grad is not None
    assert output.wrench_hat.grad is not None
    assert output.safety_logit.grad is not None
    assert output.raw_action.grad is not None


def test_hard_samples_receive_the_configured_multiplier():
    output = StudentOutput(
        wrench_hat=torch.zeros(2, 6),
        latent=torch.zeros(2, 32),
        safety_logit=torch.zeros(2, 1),
        raw_action=torch.zeros(2, 23),
        action=torch.zeros(2, 23),
    )
    target = torch.zeros(2, 23)
    target[0] = 1.0
    target[1] = 1.0
    base = student_dagger_loss(
        output, target, torch.zeros(2, 6), torch.zeros(2),
        torch.zeros(2, dtype=torch.bool), torch.zeros(2, 23), StudentLossCfg()
    )
    hard = student_dagger_loss(
        output, target, torch.zeros(2, 6), torch.zeros(2),
        torch.ones(2, dtype=torch.bool), torch.zeros(2, 23),
        StudentLossCfg(hard_sample_multiplier=2.0),
    )
    assert hard["action"] == pytest.approx(2.0 * base["action"].item())


def test_dagger_loss_rejects_nonfinite_or_wrong_contracts():
    output = _output(2)
    with pytest.raises(ValueError, match="target_action"):
        student_dagger_loss(
            output, torch.zeros(2, 22), torch.zeros(2, 6), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.zeros(2, 23), StudentLossCfg()
        )
    wrench = torch.zeros(2, 6)
    wrench[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        student_dagger_loss(
            output, torch.zeros(2, 23), wrench, torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.zeros(2, 23), StudentLossCfg()
        )
