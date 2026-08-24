"""RSL-RL wrapper for the combined coordinated 23-effort task."""

from __future__ import annotations

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

import go2_pvcnn.mdp as mdp
from go2_pvcnn.tasks.m1_panda_coordinated_disturbance import (
    CoordinatedDisturbanceCfg,
    CoordinatedDisturbanceScheduler,
    base_wrench_to_body_local,
)


class M1PandaCoordinatedEnvWrapper(VecEnv):
    def __init__(
        self,
        env,
        *,
        training_randomization: bool = False,
        seed: int = 0,
    ):
        if not isinstance(training_randomization, bool):
            raise TypeError("training_randomization must be a bool")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.env = env
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.num_actions = int(env.action_manager.total_action_dim)
        if self.num_actions != 23:
            raise ValueError(f"coordinated env must expose 23 actions, got {self.num_actions}")
        self.env.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype="float32")
        self._training_randomization = training_randomization
        self._robot = env.scene["robot"] if training_randomization else None
        self._base_body_id = None
        self._hand_body_id = None
        self._disturbance = None
        self._root_pose_deviation_max = torch.zeros((), device=self.device)
        self._root_velocity_deviation_max = torch.zeros((), device=self.device)
        self._joint_position_deviation_max = torch.zeros((), device=self.device)
        self._joint_velocity_deviation_max = torch.zeros((), device=self.device)
        if training_randomization:
            self._base_body_id = self._exact_body_id("BASE_LINK")
            self._hand_body_id = self._exact_body_id("panda_hand")
            if self._base_body_id == self._hand_body_id:
                raise RuntimeError("BASE_LINK and panda_hand must be distinct bodies")
            step_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
            self._disturbance = CoordinatedDisturbanceScheduler(
                CoordinatedDisturbanceCfg(),
                self.num_envs,
                self.device,
                step_dt,
                seed=seed,
            )
            self._observe_reset_deviation(
                torch.arange(self.num_envs, device=self.device)
            )

    def _exact_body_id(self, name: str) -> int:
        ids, names = self._robot.find_bodies(name, preserve_order=True)
        if len(ids) != 1 or names != [name]:
            raise RuntimeError(
                f"expected exactly one body named {name!r}, got "
                f"ids={ids!r}, names={names!r}"
            )
        return int(ids[0])

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.env.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.env.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.unwrapped.episode_length_buf = value

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def _format(self, obs_dict):
        if not isinstance(obs_dict, dict) or "policy" not in obs_dict:
            raise RuntimeError("coordinated env must return policy observations")
        obs = obs_dict["policy"]
        if obs.ndim != 2 or obs.shape[0] != self.num_envs or not torch.isfinite(obs).all():
            raise RuntimeError("coordinated policy observations must be finite batched tensors")
        if obs.dtype != torch.float32 or obs.device != self.device:
            raise RuntimeError("coordinated observations must be float32 on env device")
        return obs, {"observations": {"critic": obs}}

    def get_observations(self):
        return self._format(self.env.observation_manager.compute())

    def _nominal_wheel_actions(self):
        desired_twist = mdp.coordinated_desired_twist_b(self.env.unwrapped)
        cfg = self.env.unwrapped.cfg
        wheel_action = (
            desired_twist[:, 0]
            * float(cfg.mission_wheel_damping_nm_per_rad_s)
            / float(cfg.mission_wheel_radius_m)
            / float(cfg.mission_wheel_action_scale_nm)
        )
        nominal = torch.zeros(
            self.num_envs, 23, dtype=desired_twist.dtype, device=self.device
        )
        nominal[:, 12:16] = wheel_action.unsqueeze(-1)
        return nominal

    def _mission_arrived(self):
        error = mdp.coordinated_base_target_error_b(self.env.unwrapped)
        cfg = self.env.unwrapped.cfg
        return (
            torch.linalg.vector_norm(error[:, :2], dim=-1)
            <= float(cfg.mission_arrival_position_tolerance_m)
        ) & (error[:, 2].abs() <= float(cfg.mission_arrival_yaw_tolerance_rad))

    def reset(self):
        obs, _ = self.env.reset()
        if self._disturbance is not None:
            self._disturbance.reset()
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._observe_reset_deviation(env_ids)
            self._apply_training_wrench(self._disturbance.current_wrench_b)
        return self._format(obs)

    def _apply_training_wrench(self, wrench_b: torch.Tensor) -> None:
        if (
            not isinstance(wrench_b, torch.Tensor)
            or tuple(wrench_b.shape) != (self.num_envs, 6)
            or wrench_b.device != self.device
            or not bool(torch.isfinite(wrench_b).all())
        ):
            raise RuntimeError(
                "training wrench must be finite with shape (num_envs, 6) "
                "on the environment device"
            )
        force_h, torque_h = base_wrench_to_body_local(
            wrench_b[:, :3],
            wrench_b[:, 3:],
            self._robot.data.body_quat_w[:, self._base_body_id],
            self._robot.data.body_quat_w[:, self._hand_body_id],
        )
        self._robot.set_external_force_and_torque(
            force_h.unsqueeze(1),
            torque_h.unsqueeze(1),
            body_ids=[self._hand_body_id],
        )

    def _observe_reset_deviation(self, env_ids: torch.Tensor) -> None:
        if self._robot is None or env_ids.numel() == 0:
            return
        data = self._robot.data
        ids = env_ids.to(device=self.device, dtype=torch.long)
        default_root = data.default_root_state[ids]
        current_root = data.root_state_w[ids]
        expected_position = default_root[:, :3] + self.env.scene.env_origins[ids]
        position_deviation = (current_root[:, :3] - expected_position).abs().max()
        quaternion_dot = torch.sum(
            current_root[:, 3:7] * default_root[:, 3:7], dim=-1
        ).abs()
        orientation_deviation = 2.0 * torch.acos(
            torch.clamp(quaternion_dot, 0.0, 1.0)
        ).max()
        root_pose_deviation = torch.maximum(
            position_deviation, orientation_deviation
        )
        root_velocity_deviation = (
            current_root[:, 7:13] - default_root[:, 7:13]
        ).abs().max()
        joint_position_deviation = (
            data.joint_pos[ids] - data.default_joint_pos[ids]
        ).abs().max()
        joint_velocity_deviation = (
            data.joint_vel[ids] - data.default_joint_vel[ids]
        ).abs().max()
        values = (
            root_pose_deviation,
            root_velocity_deviation,
            joint_position_deviation,
            joint_velocity_deviation,
        )
        if not all(bool(torch.isfinite(value)) for value in values):
            raise RuntimeError("reset deviation diagnostics must be finite")
        self._root_pose_deviation_max = torch.maximum(
            self._root_pose_deviation_max, root_pose_deviation
        )
        self._root_velocity_deviation_max = torch.maximum(
            self._root_velocity_deviation_max, root_velocity_deviation
        )
        self._joint_position_deviation_max = torch.maximum(
            self._joint_position_deviation_max, joint_position_deviation
        )
        self._joint_velocity_deviation_max = torch.maximum(
            self._joint_velocity_deviation_max, joint_velocity_deviation
        )

    def _attach_guard_episode_metrics(
        self, extras: dict, dones: torch.Tensor
    ) -> None:
        completed = int(torch.count_nonzero(dones).item())
        if completed == 0:
            return
        log = extras.get("log")
        if not isinstance(log, dict):
            raise RuntimeError("training reset must provide an Isaac Lab log dictionary")
        sources = {
            "Termination/time_out": ("Episode_Termination/time_out", True),
            "Termination/base_contact": (
                "Episode_Termination/base_contact",
                True,
            ),
            "Termination/bad_orientation": (
                "Episode_Termination/bad_orientation",
                True,
            ),
            "Reward/base_target": ("Episode_Reward/base_target", False),
            "Reward/ee_tracking": ("Episode_Reward/ee_tracking", False),
        }
        for target, (source, is_count) in sources.items():
            if source not in log:
                raise RuntimeError(f"training reset log is missing {source!r}")
            scalar = torch.as_tensor(log[source], device=self.device).reshape(-1)
            if scalar.numel() != 1 or not bool(torch.isfinite(scalar).all()):
                raise RuntimeError(f"training reset metric {source!r} must be finite")
            value = scalar.item() / completed if is_count else scalar.item()
            log[target] = torch.full(
                (completed,), float(value), device=self.device, dtype=torch.float32
            )

    def step(self, actions):
        if tuple(actions.shape) != (self.num_envs, 23):
            raise ValueError("coordinated actions must have shape (num_envs, 23)")
        actions = actions.clone()
        arrived = self._mission_arrived()
        actions[arrived, 12:16] = 0.0
        actions = actions + self._nominal_wheel_actions()
        actions = torch.clamp(actions, -1.0, 1.0)
        if self._disturbance is not None:
            self._apply_training_wrench(self._disturbance.advance())
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        obs, obs_extras = self._format(obs)
        extras = dict(extras or {})
        extras.setdefault("observations", {}).update(obs_extras["observations"])
        # The local RSL-RL runner still consumes the legacy VecEnv four-tuple.
        # Preserve both Gymnasium termination causes for downstream logging.
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        dones = terminated | truncated
        if self._disturbance is not None:
            self._attach_guard_episode_metrics(extras, dones)
        if self._disturbance is not None and bool(dones.any()):
            done_ids = dones.nonzero(as_tuple=False).flatten()
            self._disturbance.reset(done_ids)
            self._observe_reset_deviation(done_ids)
        return obs, rewards, dones, extras

    @property
    def current_wrench_b(self) -> torch.Tensor:
        if self._disturbance is None:
            return torch.zeros(self.num_envs, 6, device=self.device)
        return self._disturbance.current_wrench_b

    def get_training_diagnostics(self) -> dict[str, float]:
        wrench = self.current_wrench_b
        force_norm = torch.linalg.vector_norm(wrench[:, :3], dim=-1)
        torque_norm = torch.linalg.vector_norm(wrench[:, 3:], dim=-1)
        nonzero_ratio = (wrench.abs().sum(dim=-1) > 0).to(torch.float32).mean()
        values = {
            "curriculum_scale": (
                float(self._disturbance.curriculum_scale)
                if self._disturbance is not None
                else 0.0
            ),
            "force_norm_max": float(force_norm.max().item()),
            "torque_norm_max": float(torque_norm.max().item()),
            "nonzero_wrench_ratio": float(nonzero_ratio.item()),
            "root_pose_deviation_max": float(
                self._root_pose_deviation_max.item()
            ),
            "root_velocity_deviation_max": float(
                self._root_velocity_deviation_max.item()
            ),
            "joint_position_deviation_max": float(
                self._joint_position_deviation_max.item()
            ),
            "joint_velocity_deviation_max": float(
                self._joint_velocity_deviation_max.item()
            ),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise RuntimeError("training diagnostics must be finite")
        return values


__all__ = ["M1PandaCoordinatedEnvWrapper"]
