import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.student_contracts import (
    STUDENT_ACTION_DIM,
    STUDENT_HISTORY_LENGTH,
    STUDENT_OBSERVATION_DIM,
    StudentActionScaleCfg,
    StudentNominalCommand,
    apply_student_residual,
    teacher_residual_label,
)


def _nominal(*, dtype=torch.float32):
    return StudentNominalCommand(
        position=torch.zeros(2, 23, dtype=dtype),
        velocity=torch.zeros(2, 23, dtype=dtype),
    )


def test_student_dimensions_are_frozen():
    assert STUDENT_OBSERVATION_DIM == 100
    assert STUDENT_HISTORY_LENGTH == 10
    assert STUDENT_ACTION_DIM == 23


def test_teacher_label_reconstructs_safe_teacher_targets_without_slew_clipping():
    nominal = _nominal()
    q_des = nominal.position.clone()
    q_des[:, :12] += 0.01
    q_des[:, 16:] -= 0.005
    qd_des = nominal.velocity.clone()
    qd_des[:, 12:16] += 0.25

    label = teacher_residual_label(
        q_des, qd_des, nominal, StudentActionScaleCfg()
    )
    reconstructed = apply_student_residual(
        label,
        nominal,
        StudentActionScaleCfg(),
        previous_action=torch.zeros_like(label),
    )

    torch.testing.assert_close(reconstructed.position, q_des)
    torch.testing.assert_close(reconstructed.velocity[:, 12:16], qd_des[:, 12:16])
    assert not bool(reconstructed.saturated.any())


def test_student_residual_clips_amplitude_and_group_specific_slew():
    action = torch.full((2, 23), 2.0)
    command = apply_student_residual(
        action,
        _nominal(),
        StudentActionScaleCfg(),
        previous_action=torch.zeros_like(action),
    )

    torch.testing.assert_close(command.position[:, :12], torch.full((2, 12), 0.02))
    torch.testing.assert_close(command.velocity[:, 12:16], torch.full((2, 4), 0.5))
    torch.testing.assert_close(command.position[:, 16:], torch.full((2, 7), 0.01))
    torch.testing.assert_close(
        command.normalized_action[:, :12], torch.full((2, 12), 0.08)
    )
    torch.testing.assert_close(
        command.normalized_action[:, 12:16], torch.full((2, 4), 0.0625)
    )
    torch.testing.assert_close(
        command.normalized_action[:, 16:], torch.full((2, 7), 0.05)
    )
    assert command.saturated.dtype == torch.bool
    assert bool(command.saturated.all())


@pytest.mark.parametrize("field", ["position", "velocity"])
def test_nominal_rejects_nonfinite_or_wrong_width(field):
    values = {
        "position": torch.zeros(2, 23),
        "velocity": torch.zeros(2, 23),
    }
    values[field] = torch.zeros(2, 22)
    with pytest.raises(ValueError, match="shape"):
        teacher_residual_label(
            torch.zeros(2, 23),
            torch.zeros(2, 23),
            StudentNominalCommand(**values),
            StudentActionScaleCfg(),
        )

    values[field] = torch.zeros(2, 23)
    values[field][0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        teacher_residual_label(
            torch.zeros(2, 23),
            torch.zeros(2, 23),
            StudentNominalCommand(**values),
            StudentActionScaleCfg(),
        )


def test_action_boundary_rejects_dtype_device_and_nonfinite_mismatch():
    nominal = _nominal(dtype=torch.float64)
    with pytest.raises(TypeError, match="dtype"):
        apply_student_residual(
            torch.zeros(2, 23, dtype=torch.float32),
            nominal,
            StudentActionScaleCfg(),
            previous_action=torch.zeros(2, 23, dtype=torch.float64),
        )
    bad = torch.zeros(2, 23, dtype=torch.float64)
    bad[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        apply_student_residual(
            bad,
            nominal,
            StudentActionScaleCfg(),
            previous_action=torch.zeros_like(bad),
        )
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            apply_student_residual(
                torch.zeros(2, 23, dtype=torch.float64, device="cuda:0"),
                nominal,
                StudentActionScaleCfg(),
                previous_action=torch.zeros(2, 23, dtype=torch.float64),
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leg_position_rad": 0.0},
        {"wheel_velocity_radps": float("inf")},
        {"arm_slew_per_step": -0.1},
    ],
)
def test_action_scale_cfg_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        StudentActionScaleCfg(**kwargs)
