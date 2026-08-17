import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.constraints import (
    compute_velocity_bounds,
)
from go2_pvcnn.control.m1_panda_coordination.motion_distribution import (
    MotionDistributionCfg,
    distribute_motion,
)


def _coordination_inputs(dtype=torch.float64):
    jacobian = torch.zeros(6, 10, dtype=dtype)
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    jacobian[5, 2] = 1.0
    jacobian[:, 3:9] = torch.eye(6, dtype=dtype)
    return {
        "coordinated_jacobian": jacobian,
        "pose_error": torch.zeros(6, dtype=dtype),
        "desired_twist": torch.zeros(6, dtype=dtype),
        "q": torch.zeros(10, dtype=dtype),
        "qd": torch.zeros(10, dtype=dtype),
        "q_min": torch.full((10,), -10.0, dtype=dtype),
        "q_max": torch.full((10,), 10.0, dtype=dtype),
        "v_max": torch.full((10,), 10.0, dtype=dtype),
        "a_max": torch.full((10,), 1000.0, dtype=dtype),
        "manipulability_gradient": torch.zeros(10, dtype=dtype),
        "sigma_min": torch.tensor(1.0, dtype=dtype),
        "dt": 0.02,
    }


def test_velocity_bounds_intersect_position_velocity_and_acceleration_limits():
    q = torch.tensor([0.9, 0.0, 0.0], dtype=torch.float64)
    qd = torch.tensor([0.0, 0.4, -0.4], dtype=torch.float64)
    q_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
    q_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    v_max = torch.tensor([10.0, 0.3, 10.0], dtype=torch.float64)
    a_max = torch.tensor([100.0, 100.0, 2.0], dtype=torch.float64)

    lower, upper = compute_velocity_bounds(
        q, qd, q_min, q_max, v_max, a_max, dt=0.02
    )

    assert torch.equal(lower, torch.tensor([-2.0, -0.3, -0.44], dtype=torch.float64))
    assert torch.allclose(
        upper, torch.tensor([2.0, 0.3, -0.36], dtype=torch.float64)
    )


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_velocity_bounds_reject_invalid_dt(dt):
    values = torch.zeros(2)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        compute_velocity_bounds(
            values,
            values,
            -torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            dt=dt,
        )


def test_velocity_bounds_reject_empty_intersection():
    with pytest.raises(ValueError, match="velocity bounds are infeasible at indices: 0"):
        compute_velocity_bounds(
            torch.tensor([2.0]),
            torch.tensor([0.0]),
            torch.tensor([-1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            dt=0.1,
        )


def test_velocity_bounds_reject_shape_and_finite_mismatch():
    with pytest.raises(ValueError, match="q and qd shape must match"):
        compute_velocity_bounds(
            torch.zeros(2),
            torch.zeros(3),
            -torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            dt=0.1,
        )
    with pytest.raises(ValueError, match="v_max must contain only finite values"):
        compute_velocity_bounds(
            torch.zeros(2),
            torch.zeros(2),
            -torch.ones(2),
            torch.ones(2),
            torch.tensor([1.0, float("nan")]),
            torch.ones(2),
            dt=0.1,
        )


def test_motion_distribution_default_configuration_is_frozen():
    cfg = MotionDistributionCfg()

    assert cfg.pose_gain == pytest.approx(10.0)
    assert cfg.damping == pytest.approx(1.0e-4)
    assert cfg.singularity_threshold == pytest.approx(0.1)
    assert cfg.null_gain == pytest.approx(5.0)
    assert cfg.null_damping == pytest.approx(0.5)
    assert cfg.max_saturation_passes == 10


def test_arm_only_solution_is_selected_when_feasible():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 0.4
    inputs["desired_twist"][2] = -0.2

    result = distribute_motion(**inputs)

    assert not result.base_active.item()
    assert torch.count_nonzero(result.qd_coord[:3]) == 0
    assert torch.allclose(
        inputs["coordinated_jacobian"] @ result.qd_coord,
        inputs["desired_twist"],
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert result.phi.item() == pytest.approx(1.0)


def test_pose_error_feedback_participates_in_p1_target():
    inputs = _coordination_inputs()
    inputs["pose_error"][1] = 0.02

    result = distribute_motion(**inputs)

    achieved = inputs["coordinated_jacobian"] @ result.qd_coord
    assert achieved[1].item() == pytest.approx(0.2, abs=1.0e-7)


def test_saturated_arm_joint_is_frozen_and_motion_redistributes_to_base():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 1.0
    inputs["v_max"][3] = 0.2

    result = distribute_motion(**inputs)

    assert result.base_active.item()
    assert result.saturated[3].item()
    assert result.qd_coord[3].item() == pytest.approx(0.2)
    assert result.qd_coord[0].item() == pytest.approx(0.8, abs=1.0e-6)
    assert torch.allclose(
        inputs["coordinated_jacobian"] @ result.qd_coord,
        inputs["desired_twist"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_rank_loss_activates_base_to_complete_the_task():
    inputs = _coordination_inputs()
    inputs["coordinated_jacobian"][5, 8] = 0.0
    inputs["desired_twist"][5] = 0.3

    result = distribute_motion(**inputs)

    assert result.base_active.item()
    assert result.qd_coord[2].item() == pytest.approx(0.3, abs=1.0e-6)


def test_singularity_threshold_activates_base_even_when_arm_solution_exists():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 0.2
    inputs["sigma_min"] = torch.tensor(0.05, dtype=torch.float64)

    result = distribute_motion(**inputs)

    assert result.base_active.item()
    assert result.sigma_min.item() == pytest.approx(0.05)


def test_base_activates_when_acceleration_bounds_make_zero_base_velocity_unreachable():
    inputs = _coordination_inputs()
    inputs["qd"][0] = 1.0
    inputs["a_max"][0] = 1.0

    result = distribute_motion(**inputs)

    assert result.base_active.item()
    assert result.qd_coord[0].item() >= 0.98 - 1.0e-9


def test_manipulability_gradient_moves_only_in_p1_null_space():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 0.2
    inputs["manipulability_gradient"][9] = 1.0

    result = distribute_motion(**inputs)

    assert result.qd_coord[9].item() > 0.0
    assert torch.allclose(
        inputs["coordinated_jacobian"] @ result.qd_coord,
        inputs["desired_twist"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_null_space_psi_scales_before_end_effector_phi():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 0.2
    inputs["manipulability_gradient"][9] = 1.0
    inputs["q"][9] = inputs["q_max"][9]

    result = distribute_motion(**inputs)

    assert result.psi.item() == pytest.approx(0.0)
    assert result.phi.item() == pytest.approx(1.0)
    assert torch.allclose(
        inputs["coordinated_jacobian"] @ result.qd_coord,
        inputs["desired_twist"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_end_effector_phi_reduces_only_after_null_space_is_disabled():
    inputs = _coordination_inputs()
    inputs["coordinated_jacobian"].zero_()
    inputs["coordinated_jacobian"][0, 0] = 1.0
    inputs["coordinated_jacobian"][0, 3] = 1.0
    inputs["desired_twist"][0] = 1.0
    inputs["v_max"].fill_(0.2)

    result = distribute_motion(**inputs)

    assert result.base_active.item()
    assert result.psi.item() == pytest.approx(0.0)
    assert result.phi.item() == pytest.approx(0.4, abs=2.0e-3)
    achieved = inputs["coordinated_jacobian"] @ result.qd_coord
    assert achieved[0].item() == pytest.approx(result.phi.item(), abs=2.0e-3)


def test_outputs_are_finite_within_bounds_and_preserve_batch_dimensions():
    first = _coordination_inputs(dtype=torch.float32)
    jacobian = torch.stack((first["coordinated_jacobian"],) * 2)
    zeros6 = torch.zeros(2, 6)
    zeros10 = torch.zeros(2, 10)
    result = distribute_motion(
        coordinated_jacobian=jacobian,
        pose_error=zeros6,
        desired_twist=torch.tensor(
            [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.2, 0.0, 0.0, 0.0, 0.0]]
        ),
        q=zeros10,
        qd=zeros10,
        q_min=torch.full((2, 10), -1.0),
        q_max=torch.full((2, 10), 1.0),
        v_max=torch.full((2, 10), 0.5),
        a_max=torch.full((2, 10), 100.0),
        manipulability_gradient=zeros10,
        sigma_min=torch.ones(2),
        dt=0.02,
    )
    lower, upper = compute_velocity_bounds(
        zeros10,
        zeros10,
        torch.full((2, 10), -1.0),
        torch.full((2, 10), 1.0),
        torch.full((2, 10), 0.5),
        torch.full((2, 10), 100.0),
        dt=0.02,
    )

    assert result.qd_coord.shape == (2, 10)
    assert result.base_active.shape == (2,)
    assert result.phi.shape == (2,)
    assert result.psi.shape == (2,)
    assert result.saturated.shape == (2, 10)
    assert torch.isfinite(result.qd_coord).all()
    assert torch.all(result.qd_coord >= lower)
    assert torch.all(result.qd_coord <= upper)


def test_motion_distribution_rejects_non_finite_input():
    inputs = _coordination_inputs()
    inputs["desired_twist"][2] = float("nan")

    with pytest.raises(ValueError, match="desired_twist must contain only finite values"):
        distribute_motion(**inputs)
