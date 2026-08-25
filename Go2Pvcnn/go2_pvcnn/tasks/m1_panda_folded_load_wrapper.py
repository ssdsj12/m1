"""RSL-RL boundary for folded-load locomotion without Panda actions or wrench."""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

import go2_pvcnn.mdp as mdp
from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import (
    classify_command_buckets,
    sample_episode_commands,
    stage_spec,
)


_ARM_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))


class M1PandaFoldedLoadEnvWrapper(VecEnv):
    """Own episode commands, metrics, and the exact-zero inactive boundary."""

    def __init__(self, env, *, stage: str, seed: int = 0):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.stage = stage_spec(stage)
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.num_actions = int(env.action_manager.total_action_dim)
        if self.num_actions != 23:
            raise ValueError(f"folded-load env must expose 23 actions, got {self.num_actions}")
        self.env.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype="float32")
        self._robot = env.scene["robot"]
        arm_ids, arm_names = self._robot.find_joints(
            list(_ARM_NAMES), preserve_order=True
        )
        if tuple(arm_names) != _ARM_NAMES or len(set(map(int, arm_ids))) != 7:
            raise RuntimeError("folded-load wrapper requires seven canonical Panda joints")
        self._arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
        self._seed = seed
        self._command_generation = 0
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_family = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.env.unwrapped.folded_load_commands = self.commands
        self._vx_error_sq = torch.zeros(self.num_envs, device=self.device)
        self._wz_error_sq = torch.zeros(self.num_envs, device=self.device)
        self._stationary_abs_vx = torch.zeros(self.num_envs, device=self.device)
        self._stationary_abs_wz = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._inactive_action_max = torch.zeros((), device=self.device)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

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
            raise RuntimeError("folded-load env must return policy observations")
        obs = obs_dict["policy"]
        if tuple(obs.shape) != (self.num_envs, 103) or not bool(torch.isfinite(obs).all()):
            raise RuntimeError("folded-load policy observations must be finite [num_envs, 103]")
        return obs, {"observations": {"critic": obs}}

    def get_observations(self):
        return self._format(self.env.observation_manager.compute())

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        batch = sample_episode_commands(
            int(env_ids.numel()),
            self.stage,
            seed=self._seed + self._command_generation,
            device=self.device,
        )
        self._command_generation += 1
        self.commands[env_ids] = batch.twist
        self.command_family[env_ids] = batch.family

    def reset(self):
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        obs, _ = self.env.reset()
        return self._format(obs)

    def _accumulate_tracking(self) -> None:
        linear = self._robot.data.root_lin_vel_b
        angular = self._robot.data.root_ang_vel_b
        if not bool(torch.isfinite(linear).all() and torch.isfinite(angular).all()):
            raise RuntimeError("folded-load body velocity must remain finite")
        self._vx_error_sq += (linear[:, 0] - self.commands[:, 0]).square()
        self._wz_error_sq += (angular[:, 2] - self.commands[:, 2]).square()
        stationary = self.commands[:, 0].eq(0.0) & self.commands[:, 2].eq(0.0)
        self._stationary_abs_vx += torch.where(
            stationary, linear[:, 0].abs(), torch.zeros_like(linear[:, 0])
        )
        self._stationary_abs_wz += torch.where(
            stationary, angular[:, 2].abs(), torch.zeros_like(angular[:, 2])
        )
        self._episode_steps += 1

    def _termination_term(self, name: str) -> torch.Tensor:
        value = self.env.unwrapped.termination_manager.get_term(name)
        value = torch.as_tensor(value, device=self.device, dtype=torch.bool)
        if value.shape != (self.num_envs,):
            raise RuntimeError(f"termination term {name!r} must have shape [num_envs]")
        return value

    def _completed_metrics(
        self, done_ids: torch.Tensor, truncated: torch.Tensor
    ) -> dict[str, object]:
        steps = self._episode_steps[done_ids].clamp_min(1)
        command = self.commands[done_ids].clone()
        return {
            "env_id": done_ids.clone(),
            "command": command,
            "family": self.command_family[done_ids].clone(),
            "bucket": classify_command_buckets(command),
            "steps": steps.clone(),
            "vx_error_sq_sum": self._vx_error_sq[done_ids].clone(),
            "wz_error_sq_sum": self._wz_error_sq[done_ids].clone(),
            "stationary_abs_vx_sum": self._stationary_abs_vx[done_ids].clone(),
            "stationary_abs_wz_sum": self._stationary_abs_wz[done_ids].clone(),
            "time_out": truncated[done_ids].clone(),
            "base_contact": self._termination_term("base_contact")[done_ids].clone(),
            "bad_orientation": self._termination_term("bad_orientation")[done_ids].clone(),
        }

    def step(self, actions):
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (
            self.num_envs,
            23,
        ):
            raise ValueError("folded-load actions must have shape (num_envs, 23)")
        if actions.device != self.device or not bool(torch.isfinite(actions).all()):
            raise ValueError("folded-load actions must be finite on the env device")
        self._accumulate_tracking()
        safe_actions = actions.clone()
        safe_actions[:, 16:23] = 0.0
        if not bool(safe_actions[:, 16:23].eq(0.0).all()):
            raise RuntimeError("inactive Panda action leakage")
        self._inactive_action_max = torch.maximum(
            self._inactive_action_max, safe_actions[:, 16:23].abs().max()
        )
        obs, rewards, terminated, truncated, extras = self.env.step(safe_actions)
        dones = terminated | truncated
        extras = dict(extras or {})
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        if bool(dones.any()):
            done_ids = dones.nonzero(as_tuple=False).flatten()
            extras["episode_metrics"] = self._completed_metrics(done_ids, truncated)
            for buffer in (
                self._vx_error_sq,
                self._wz_error_sq,
                self._stationary_abs_vx,
                self._stationary_abs_wz,
                self._episode_steps,
            ):
                buffer[done_ids] = 0
            self._resample_commands(done_ids)
            obs = self.env.observation_manager.compute()
        obs, obs_extras = self._format(obs)
        extras.setdefault("observations", {}).update(obs_extras["observations"])
        return obs, rewards, dones, extras

    def get_training_diagnostics(self) -> dict[str, float]:
        arm = self._robot.data.joint_pos.index_select(1, self._arm_ids)
        target = self._robot.data.default_joint_pos.index_select(1, self._arm_ids)
        fold_error = (arm - target).abs().max()
        torque = getattr(self._robot.data, "applied_torque", None)
        if isinstance(torque, torch.Tensor):
            arm_torque = torque.index_select(1, self._arm_ids).abs()
            limits = arm_torque.new_tensor((87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0))
            effort_utilization = (arm_torque / limits).max()
        else:
            effort_utilization = fold_error.new_zeros(())
        limits = getattr(self._robot.data, "soft_joint_pos_limits", None)
        if isinstance(limits, torch.Tensor):
            arm_limits = limits.index_select(1, self._arm_ids)
            joint_limit_proximity = torch.minimum(
                arm - arm_limits[..., 0], arm_limits[..., 1] - arm
            ).min()
        else:
            joint_limit_proximity = fold_error.new_tensor(float("inf"))
        mount_wrench_norm = fold_error.new_zeros(())
        if hasattr(self._robot, "root_physx_view"):
            wrench = mdp.m1_panda_mount_wrench_b(
                self.env.unwrapped, SimpleNamespace(name="robot")
            )
            mount_wrench_norm = torch.linalg.vector_norm(wrench, dim=-1).max()
        values = {
            "inactive_action_max": float(self._inactive_action_max.item()),
            "fold_error_max": float(fold_error.item()),
            "effort_utilization_max": float(effort_utilization.item()),
            "joint_limit_proximity_min": float(joint_limit_proximity.item()),
            "mount_wrench_norm_max": float(mount_wrench_norm.item()),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise RuntimeError("folded-load diagnostics must be finite")
        return values


__all__ = ["M1PandaFoldedLoadEnvWrapper"]
