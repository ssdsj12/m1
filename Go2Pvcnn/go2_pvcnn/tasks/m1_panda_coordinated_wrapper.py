"""RSL-RL wrapper for the combined coordinated 23-effort task."""

from __future__ import annotations

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

import go2_pvcnn.mdp as mdp


class M1PandaCoordinatedEnvWrapper(VecEnv):
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.num_actions = int(env.action_manager.total_action_dim)
        if self.num_actions != 23:
            raise ValueError(f"coordinated env must expose 23 actions, got {self.num_actions}")
        self.env.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype="float32")

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
        return self._format(obs)

    def step(self, actions):
        if tuple(actions.shape) != (self.num_envs, 23):
            raise ValueError("coordinated actions must have shape (num_envs, 23)")
        actions = actions.clone()
        arrived = self._mission_arrived()
        actions[arrived, 12:16] = 0.0
        actions = actions + self._nominal_wheel_actions()
        actions = torch.clamp(actions, -1.0, 1.0)
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        obs, obs_extras = self._format(obs)
        extras = dict(extras or {})
        extras.setdefault("observations", {}).update(obs_extras["observations"])
        # The local RSL-RL runner still consumes the legacy VecEnv four-tuple.
        # Preserve both Gymnasium termination causes for downstream logging.
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        return obs, rewards, terminated | truncated, extras


__all__ = ["M1PandaCoordinatedEnvWrapper"]
