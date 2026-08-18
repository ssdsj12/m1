import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.student_model import (
    M1PandaStudent,
    StudentHistoryBuffer,
    StudentNetworkCfg,
)


def test_history_reset_only_clears_selected_environments():
    history = StudentHistoryBuffer(3, "cpu")
    history.append(torch.ones(3, 100))
    history.reset(torch.tensor([1]))
    assert torch.count_nonzero(history.value[0]) > 0
    assert torch.count_nonzero(history.value[1]) == 0
    assert torch.count_nonzero(history.value[2]) > 0


def test_history_rolls_explicit_frames_and_returns_a_clone():
    history = StudentHistoryBuffer(2, "cpu", dtype=torch.float64)
    for value in range(12):
        history.append(torch.full((2, 100), float(value), dtype=torch.float64))
    assert history.value.shape == (2, 10, 100)
    assert history.value[0, 0, 0].item() == pytest.approx(2.0)
    assert history.value[0, -1, 0].item() == pytest.approx(11.0)
    exposed = history.value
    exposed.zero_()
    assert torch.count_nonzero(history.value) > 0


def test_student_network_outputs_finite_wrench_latent_and_action():
    model = M1PandaStudent(StudentNetworkCfg())
    out = model(torch.zeros(4, 10, 100))
    assert out.wrench_hat.shape == (4, 6)
    assert out.latent.shape == (4, 32)
    assert out.safety_logit.shape == (4, 1)
    assert out.raw_action.shape == (4, 23)
    assert out.action.shape == (4, 23)
    assert all(
        bool(torch.isfinite(value).all())
        for value in (
            out.wrench_hat,
            out.latent,
            out.safety_logit,
            out.raw_action,
            out.action,
        )
    )
    assert bool(torch.all(out.action.abs() <= 1.0))


@pytest.mark.parametrize("shape", [(4, 9, 100), (4, 10, 99), (4, 100)])
def test_student_network_rejects_wrong_history_shape(shape):
    model = M1PandaStudent(StudentNetworkCfg())
    with pytest.raises(ValueError, match="history"):
        model(torch.zeros(shape))


def test_history_and_network_reject_nonfinite_and_dtype_mismatch():
    history = StudentHistoryBuffer(2, "cpu", dtype=torch.float64)
    with pytest.raises(TypeError, match="dtype"):
        history.append(torch.zeros(2, 100, dtype=torch.float32))
    bad = torch.zeros(2, 100, dtype=torch.float64)
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        history.append(bad)

    model = M1PandaStudent(StudentNetworkCfg())
    bad_history = torch.zeros(2, 10, 100)
    bad_history[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        model(bad_history)


def test_student_network_gradients_reach_gru_estimator_and_actor():
    model = M1PandaStudent(StudentNetworkCfg())
    output = model(torch.randn(3, 10, 100))
    loss = (
        output.action.square().mean()
        + output.wrench_hat.square().mean()
        + output.safety_logit.square().mean()
    )
    loss.backward()
    for name in ("gru.weight_ih_l0", "estimator_head.weight", "actor.0.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_student_state_dict_round_trip_is_exact():
    torch.manual_seed(17)
    first = M1PandaStudent(StudentNetworkCfg())
    history = torch.randn(2, 10, 100)
    expected = first(history)
    second = M1PandaStudent(StudentNetworkCfg())
    second.load_state_dict(first.state_dict(), strict=True)
    actual = second(history)
    torch.testing.assert_close(actual.wrench_hat, expected.wrench_hat)
    torch.testing.assert_close(actual.latent, expected.latent)
    torch.testing.assert_close(actual.safety_logit, expected.safety_logit)
    torch.testing.assert_close(actual.raw_action, expected.raw_action)
    torch.testing.assert_close(actual.action, expected.action)
