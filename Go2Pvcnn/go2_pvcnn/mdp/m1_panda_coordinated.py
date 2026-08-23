"""Batched coordinated PPO targets, observations, and reward primitives."""

from __future__ import annotations

import math

import torch


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating torch.Tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def _rotate_world_to_body_xy(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (cosine * vector[:, 0] + sine * vector[:, 1], -sine * vector[:, 0] + cosine * vector[:, 1]),
        dim=-1,
    )


def _quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[:, 1:] = -result[:, 1:]
    return result / quaternion.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = first.unbind(dim=-1)
    bw, bx, by, bz = second.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[:, :1]), vector), dim=-1)
    return _quat_multiply(
        _quat_multiply(_quat_inverse(quaternion), pure), quaternion
    )[:, 1:]


def _axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    normalized = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    normalized = torch.where(normalized[:, :1] < 0.0, -normalized, normalized)
    vector = normalized[:, 1:]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, normalized[:, :1].clamp_min(1.0e-12))
    scale = torch.where(norm > 1.0e-8, angle / norm, 2.0 * torch.ones_like(norm))
    return vector * scale


def _scene_origins(env, reference: torch.Tensor) -> torch.Tensor:
    origins = getattr(env, "scene_origins", None)
    if origins is None:
        origins = env.scene.env_origins
    return torch.as_tensor(origins, device=reference.device, dtype=reference.dtype)


def coordinated_base_target_error_b(env) -> torch.Tensor:
    robot = env.scene["robot"]
    root_position = _require_finite("root_pos_w", robot.data.root_pos_w)
    root_quaternion = _require_finite("root_quat_w", robot.data.root_quat_w)
    target = root_position.new_tensor(env.cfg.mission_target_base_pose)
    target_xy = _scene_origins(env, root_position)[:, :2] + target[:2]
    yaw = _yaw_from_quaternion(root_quaternion)
    xy_error = _rotate_world_to_body_xy(target_xy - root_position[:, :2], yaw)
    yaw_error = torch.remainder(target[2] - yaw + torch.pi, 2.0 * torch.pi) - torch.pi
    result = torch.cat((xy_error, yaw_error.unsqueeze(-1)), dim=-1)
    return _require_finite("base_target_error_b", result)


def _current_ee_pose_b(env, base_body_id: int, hand_body_id: int):
    robot = env.scene["robot"]
    positions = _require_finite("body_pos_w", robot.data.body_pos_w)
    quaternions = _require_finite("body_quat_w", robot.data.body_quat_w)
    base_position = positions[:, base_body_id]
    base_quaternion = quaternions[:, base_body_id]
    hand_position = positions[:, hand_body_id]
    hand_quaternion = quaternions[:, hand_body_id]
    position_b = _quat_rotate_inverse(base_quaternion, hand_position - base_position)
    quaternion_b = _quat_multiply(_quat_inverse(base_quaternion), hand_quaternion)
    return position_b, quaternion_b


def _pair_body_ids(asset_cfg, base_body_id, hand_body_id) -> tuple[int, int]:
    if asset_cfg is not None:
        ids = tuple(int(value) for value in asset_cfg.body_ids)
        if len(ids) != 2:
            raise ValueError("asset_cfg must resolve exactly base and hand bodies")
        return ids
    if base_body_id is None or hand_body_id is None:
        raise ValueError("base_body_id and hand_body_id are required without asset_cfg")
    return int(base_body_id), int(hand_body_id)


def coordinated_ee_pose_error_b(
    env, *, asset_cfg=None, base_body_id: int | None = None, hand_body_id: int | None = None
) -> torch.Tensor:
    base_body_id, hand_body_id = _pair_body_ids(
        asset_cfg, base_body_id, hand_body_id
    )
    position, quaternion = _current_ee_pose_b(env, base_body_id, hand_body_id)
    cache_name = "_m1_panda_coordinated_initial_ee_pose_b"
    initial = getattr(env, cache_name, None)
    if initial is None or initial.shape[0] != position.shape[0]:
        initial = torch.cat((position, quaternion), dim=-1).detach().clone()
        setattr(env, cache_name, initial)
    offset = position.new_tensor(env.cfg.mission_ee_target_offset_b)
    position_error = initial[:, :3] + offset - position
    orientation_error = _axis_angle(
        _quat_multiply(initial[:, 3:], _quat_inverse(quaternion))
    )
    result = torch.cat((position_error, orientation_error), dim=-1)
    return _require_finite("ee_pose_error_b", result)


def coordinated_desired_twist_b(env) -> torch.Tensor:
    error = coordinated_base_target_error_b(env)
    desired = error.new_zeros((env.num_envs, 6))
    linear_limit = float(env.cfg.mission_base_linear_speed_limit_mps)
    yaw_limit = float(env.cfg.mission_base_yaw_rate_limit_rad_s)
    linear = error[:, :2]
    linear_norm = torch.linalg.vector_norm(linear, dim=-1, keepdim=True)
    linear_scale = torch.clamp(linear_limit / linear_norm.clamp_min(1.0e-8), max=1.0)
    desired[:, :2] = linear * linear_scale
    desired[:, 5] = torch.clamp(error[:, 2], -yaw_limit, yaw_limit)
    desired = torch.where(_arrived(env).unsqueeze(-1), torch.zeros_like(desired), desired)
    return _require_finite("desired_twist_b", desired)


def coordinated_wheel_contact(
    env,
    *,
    sensor_cfg=None,
    sensor_body_ids: tuple[int, int, int, int] | None = None,
    threshold_n: float = 1.0,
) -> torch.Tensor:
    if not math.isfinite(threshold_n) or threshold_n <= 0.0:
        raise ValueError("threshold_n must be finite and positive")
    forces = _require_finite(
        "wheel_forces", env.scene["contact_forces"].data.net_forces_w
    )
    if sensor_cfg is not None:
        sensor_body_ids = tuple(int(value) for value in sensor_cfg.body_ids)
    if sensor_body_ids is None or len(sensor_body_ids) != 4:
        raise ValueError("wheel contact requires exactly four sensor body ids")
    ids = torch.tensor(sensor_body_ids, device=forces.device, dtype=torch.long)
    selected = forces.index_select(1, ids)
    return (torch.linalg.vector_norm(selected, dim=-1) > threshold_n).to(torch.float32)


def _arrived(env) -> torch.Tensor:
    error = coordinated_base_target_error_b(env)
    return (
        torch.linalg.vector_norm(error[:, :2], dim=-1)
        <= float(env.cfg.mission_arrival_position_tolerance_m)
    ) & (error[:, 2].abs() <= float(env.cfg.mission_arrival_yaw_tolerance_rad))


def _balance_score(env, *, height_scale_m: float, tilt_scale_rad: float) -> torch.Tensor:
    robot = env.scene["robot"]
    root_position = _require_finite("root_pos_w", robot.data.root_pos_w)
    root_quaternion = _require_finite("root_quat_w", robot.data.root_quat_w)
    height = root_position[:, 2] - _scene_origins(env, root_position)[:, 2]
    target_height = float(env.cfg.mission_balance_target_height_m)
    gravity_w = torch.zeros_like(root_position)
    gravity_w[:, 2] = -1.0
    projected_gravity = _quat_rotate_inverse(root_quaternion, gravity_w)
    tilt = torch.linalg.vector_norm(projected_gravity[:, :2], dim=-1)
    return torch.exp(
        -(height - target_height).square() / height_scale_m**2
        -tilt.square() / tilt_scale_rad**2
    )


def coordinated_base_tracking_reward(
    env,
    *,
    position_scale_m: float = 0.35,
    yaw_scale_rad: float = 0.5,
    height_scale_m: float = 0.08,
    tilt_scale_rad: float = 0.35,
) -> torch.Tensor:
    error = coordinated_base_target_error_b(env)
    return torch.exp(
        -torch.linalg.vector_norm(error[:, :2], dim=-1).square() / position_scale_m**2
        -error[:, 2].square() / yaw_scale_rad**2
    ) * _balance_score(
        env, height_scale_m=height_scale_m, tilt_scale_rad=tilt_scale_rad
    )


def coordinated_base_velocity_tracking_reward(
    env,
    *,
    height_scale_m: float = 0.08,
    tilt_scale_rad: float = 0.35,
) -> torch.Tensor:
    desired = coordinated_desired_twist_b(env)
    robot = env.scene["robot"]
    linear_velocity = _require_finite("root_lin_vel_b", robot.data.root_lin_vel_b)
    angular_velocity = _require_finite("root_ang_vel_b", robot.data.root_ang_vel_b)
    linear_limit = float(env.cfg.mission_base_linear_speed_limit_mps)
    yaw_limit = float(env.cfg.mission_base_yaw_rate_limit_rad_s)
    normalized_error = (
        (linear_velocity[:, :2] - desired[:, :2]).square().sum(dim=-1)
        / linear_limit**2
        + (angular_velocity[:, 2] - desired[:, 5]).square() / yaw_limit**2
    )
    score = 1.0 - normalized_error.clamp(max=2.0)
    score = score * _balance_score(
        env, height_scale_m=height_scale_m, tilt_scale_rad=tilt_scale_rad
    )
    return torch.where(_arrived(env), torch.zeros_like(score), score)


def coordinated_folded_arm_error(
    env, *, asset_cfg=None, arm_joint_ids: tuple[int, ...] | None = None
) -> torch.Tensor:
    joint_position = _require_finite("joint_pos", env.scene["robot"].data.joint_pos)
    if asset_cfg is not None:
        arm_joint_ids = tuple(int(value) for value in asset_cfg.joint_ids)
    if arm_joint_ids is None or len(arm_joint_ids) != 7:
        raise ValueError("folded-arm error requires exactly seven arm joints")
    ids = torch.tensor(arm_joint_ids, device=joint_position.device, dtype=torch.long)
    arm = joint_position.index_select(1, ids)
    target = arm.new_tensor(env.cfg.mission_folded_arm_target)
    error = (arm - target).square().mean(dim=-1)
    return torch.where(_arrived(env), torch.zeros_like(error), error)


def coordinated_ee_tracking_reward(
    env,
    *,
    asset_cfg=None,
    base_body_id: int | None = None,
    hand_body_id: int | None = None,
    scale_m: float = 0.10,
) -> torch.Tensor:
    error = coordinated_ee_pose_error_b(
        env,
        asset_cfg=asset_cfg,
        base_body_id=base_body_id,
        hand_body_id=hand_body_id,
    )
    score = torch.exp(
        -torch.linalg.vector_norm(error[:, :3], dim=-1).square() / scale_m**2
        -0.25 * torch.linalg.vector_norm(error[:, 3:], dim=-1).square()
    )
    return torch.where(_arrived(env), score, torch.zeros_like(score))


__all__ = [
    "coordinated_base_target_error_b",
    "coordinated_base_tracking_reward",
    "coordinated_base_velocity_tracking_reward",
    "coordinated_desired_twist_b",
    "coordinated_ee_pose_error_b",
    "coordinated_ee_tracking_reward",
    "coordinated_folded_arm_error",
    "coordinated_wheel_contact",
]
