import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "go2_pvcnn/mdp/m1_panda_coordinated.py"
SPEC = importlib.util.spec_from_file_location("m1_panda_coordinated_mdp_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
coordinated_base_target_error_b = MODULE.coordinated_base_target_error_b
coordinated_base_tracking_reward = MODULE.coordinated_base_tracking_reward
coordinated_base_velocity_tracking_reward = MODULE.coordinated_base_velocity_tracking_reward
coordinated_desired_twist_b = MODULE.coordinated_desired_twist_b
coordinated_ee_pose_error_b = MODULE.coordinated_ee_pose_error_b
coordinated_ee_tracking_reward = MODULE.coordinated_ee_tracking_reward
coordinated_folded_arm_error = MODULE.coordinated_folded_arm_error
coordinated_wheel_contact = MODULE.coordinated_wheel_contact


def _env(batch: int = 2):
    body_pos = torch.zeros(batch, 3, 3)
    body_pos[:, 1] = torch.tensor([0.0, 0.0, 0.6])
    body_pos[:, 2] = torch.tensor([0.30, 0.0, 1.10])
    if batch > 1:
        body_pos[1, :, 0] += 10.0
    body_quat = torch.zeros(batch, 3, 4)
    body_quat[..., 0] = 1.0
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.0, 0.0, 0.6], [10.0, 0.0, 0.6]])[:batch],
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
            root_lin_vel_b=torch.zeros(batch, 3),
            root_ang_vel_b=torch.zeros(batch, 3),
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            joint_pos=torch.zeros(batch, 7),
        )
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=torch.zeros(batch, 4, 3))
    )
    return SimpleNamespace(
        num_envs=batch,
        device=torch.device("cpu"),
        scene={"robot": robot, "contact_forces": sensor},
        scene_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])[:batch],
        cfg=SimpleNamespace(
            mission_target_base_pose=(0.5, 0.0, 0.0),
            mission_ee_target_offset_b=(0.05, 0.0, 0.0),
            mission_folded_arm_target=(0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0),
            mission_arrival_position_tolerance_m=0.08,
            mission_arrival_yaw_tolerance_rad=0.10,
            mission_balance_target_height_m=0.6115,
            mission_base_linear_speed_limit_mps=0.25,
            mission_base_yaw_rate_limit_rad_s=0.50,
        ),
    )


def test_base_target_error_is_body_frame_and_environment_relative():
    env = _env()
    error = coordinated_base_target_error_b(env)
    assert error.shape == (2, 3)
    torch.testing.assert_close(error, torch.tensor([[0.5, 0.0, 0.0]]).repeat(2, 1))
    env.scene["robot"].data.root_pos_w[:, 0] += 0.5
    torch.testing.assert_close(coordinated_base_target_error_b(env), torch.zeros(2, 3))


def test_ee_error_freezes_reset_pose_and_adds_reachable_offset():
    env = _env()
    error = coordinated_ee_pose_error_b(env, base_body_id=1, hand_body_id=2)
    assert error.shape == (2, 6)
    torch.testing.assert_close(error[:, :3], torch.tensor([[0.05, 0.0, 0.0]]).repeat(2, 1))
    torch.testing.assert_close(error[:, 3:], torch.zeros(2, 3))
    env.scene["robot"].data.body_pos_w[:, 2, 0] += 0.05
    torch.testing.assert_close(
        coordinated_ee_pose_error_b(env, base_body_id=1, hand_body_id=2),
        torch.zeros(2, 6),
        atol=1e-6,
        rtol=0.0,
    )


def test_desired_twist_and_wheel_contact_have_frozen_widths():
    env = _env()
    desired = coordinated_desired_twist_b(env)
    assert desired.shape == (2, 6)
    torch.testing.assert_close(
        desired, torch.tensor([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    )
    sensor = env.scene["contact_forces"]
    sensor.data.net_forces_w[0, [0, 2], 2] = 2.0
    contact = coordinated_wheel_contact(env, sensor_body_ids=(0, 1, 2, 3))
    assert contact.shape == (2, 4)
    assert contact[0].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_rewards_gate_folded_arm_before_arrival_and_ee_after_arrival():
    env = _env()
    base_reward = coordinated_base_tracking_reward(env)
    assert torch.all(base_reward > 0.0)
    folded = coordinated_folded_arm_error(env, arm_joint_ids=tuple(range(7)))
    assert torch.all(folded > 0.0)
    ee_before = coordinated_ee_tracking_reward(env, base_body_id=1, hand_body_id=2)
    assert torch.equal(ee_before, torch.zeros(2))

    env.scene["robot"].data.root_pos_w[:, 0] += 0.5
    assert torch.equal(
        coordinated_folded_arm_error(env, arm_joint_ids=tuple(range(7))),
        torch.zeros(2),
    )
    ee_after = coordinated_ee_tracking_reward(env, base_body_id=1, hand_body_id=2)
    assert torch.all(ee_after > 0.0)


def test_base_tracking_reward_is_gated_by_height_and_tilt_balance():
    env = _env(batch=1)
    upright = coordinated_base_tracking_reward(env)
    env.scene["robot"].data.root_pos_w[:, 2] = 0.35
    lowered = coordinated_base_tracking_reward(env)
    env.scene["robot"].data.root_pos_w[:, 2] = 0.60
    env.scene["robot"].data.root_quat_w[:] = torch.tensor(
        [[0.70710678, 0.70710678, 0.0, 0.0]]
    )
    tilted = coordinated_base_tracking_reward(env)
    assert torch.all(upright > 10.0 * lowered)
    assert torch.all(upright > 10.0 * tilted)


def test_base_velocity_reward_prefers_safe_target_directed_motion():
    env = _env(batch=1)
    stationary = coordinated_base_velocity_tracking_reward(env)
    env.scene["robot"].data.root_lin_vel_b[:, 0] = 0.25
    matching = coordinated_base_velocity_tracking_reward(env)
    assert torch.all(matching > 10.0 * stationary)
    env.scene["robot"].data.root_pos_w[:, 2] = 0.35
    unsafe = coordinated_base_velocity_tracking_reward(env)
    assert torch.all(matching > 10.0 * unsafe)


def test_nonfinite_state_is_rejected_at_observation_boundary():
    env = _env()
    env.scene["robot"].data.root_pos_w[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        coordinated_base_target_error_b(env)
