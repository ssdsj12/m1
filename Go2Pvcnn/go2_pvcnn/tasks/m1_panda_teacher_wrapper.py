"""RSL-RL wrapper for M1 + Panda privileged Teacher balance stages."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

from go2_pvcnn.tasks.m1_panda_teacher import (
    M1PandaDisturbanceCfg,
    M1PandaDisturbanceScheduler,
    base_wrench_to_body_local,
    stage_disturbance_cfg,
)
from go2_pvcnn.tasks.m1_panda_teacher_checkpoint import module_sha256
from go2_pvcnn.tasks.m1_residual_action import (
    M1_ACTION_DIM,
    M1ResidualActionComposer,
    M1ResidualActionComposerCfg,
)


TEACHER_OBSERVATION_DIM = 60


class M1PandaTeacherEnvWrapper(VecEnv):
    """Apply Teacher disturbances and bounded residual composition around an env."""

    def __init__(
        self,
        env,
        *,
        stage: str,
        base_actor=None,
        disturbance_cfg: M1PandaDisturbanceCfg | None = None,
        seed: int = 0,
    ) -> None:
        if stage not in {"A0", "A1"}:
            raise ValueError(f"stage must be 'A0' or 'A1', got {stage!r}")
        if getattr(env.cfg, "teacher_stage", None) != stage:
            raise ValueError(
                f"wrapper stage {stage!r} does not match env cfg stage "
                f"{getattr(env.cfg, 'teacher_stage', None)!r}"
            )
        if stage == "A0" and base_actor is not None:
            raise ValueError("A0 does not accept a base_actor")
        if stage == "A1":
            if not isinstance(base_actor, torch.nn.Module):
                raise TypeError("A1 requires a base_actor torch.nn.Module")
            if base_actor.training:
                raise ValueError("A1 base_actor must be in eval mode")
            if any(parameter.requires_grad for parameter in base_actor.parameters()):
                raise ValueError("A1 base_actor parameters must be frozen")
            if not callable(getattr(base_actor, "act_inference", None)):
                raise TypeError("A1 base_actor must define act_inference(obs)")

        self.env = env
        self.stage = stage
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        if env.action_manager.total_action_dim != M1_ACTION_DIM:
            raise ValueError(
                f"Teacher env must expose {M1_ACTION_DIM} actions, got "
                f"{env.action_manager.total_action_dim}"
            )
        self.num_actions = M1_ACTION_DIM
        self.env.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(M1_ACTION_DIM,),
            dtype=env.action_space.dtype,
        )

        self._robot = env.scene["robot"]
        self._base_body_id = self._exact_body_id("BASE_LINK")
        self._hand_body_id = self._exact_body_id("panda_hand")
        if self._base_body_id == self._hand_body_id:
            raise RuntimeError("BASE_LINK and panda_hand must resolve to distinct bodies")

        approved_cfg = stage_disturbance_cfg(stage)
        env_disturbance_cfg = M1PandaDisturbanceCfg(
            force_limit_n=tuple(env.cfg.teacher_force_limit_n),
            torque_limit_nm=tuple(env.cfg.teacher_torque_limit_nm),
            hold_time_min_s=float(env.cfg.teacher_hold_time_s[0]),
            hold_time_max_s=float(env.cfg.teacher_hold_time_s[1]),
            curriculum_start_scale=float(env.cfg.teacher_curriculum_start_scale),
            curriculum_steps=int(env.cfg.teacher_curriculum_steps),
            mode_probabilities=tuple(env.cfg.teacher_mode_probabilities),
            pulse_on_fraction=float(env.cfg.teacher_pulse_on_fraction),
        )
        if env_disturbance_cfg != approved_cfg:
            raise ValueError(
                f"env disturbance cfg does not match approved {stage} defaults"
            )
        if disturbance_cfg is not None and disturbance_cfg != approved_cfg:
            raise ValueError(
                f"disturbance_cfg does not match approved {stage} defaults"
            )
        selected_disturbance_cfg = disturbance_cfg or approved_cfg
        step_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        self._disturbance = M1PandaDisturbanceScheduler(
            selected_disturbance_cfg,
            self.num_envs,
            self.device,
            step_dt,
            seed=seed,
        )
        self._residual_composer = M1ResidualActionComposer(
            M1ResidualActionComposerCfg(), self.num_envs, self.device
        )
        self._base_actor = base_actor
        self._base_composer = None
        self._frozen_actor_hash = None
        if stage == "A1":
            actor_devices = {
                value.device
                for value in (*base_actor.parameters(), *base_actor.buffers())
            }
            if any(device != self.device for device in actor_devices):
                raise ValueError(
                    f"A1 base_actor must be on device {self.device}, got "
                    f"{sorted(map(str, actor_devices))}"
                )
            self._base_composer = M1ResidualActionComposer(
                M1ResidualActionComposerCfg(), self.num_envs, self.device
            )
            self._frozen_actor_hash = module_sha256(base_actor)
        shape = (self.num_envs, M1_ACTION_DIM)
        self._last_final_action = torch.zeros(shape, device=self.device)
        self._last_trainable_residual = torch.zeros(shape, device=self.device)
        self.env.m1_teacher_trainable_residual = torch.zeros(
            shape, device=self.device
        )
        self.env.m1_teacher_previous_trainable_residual = torch.zeros(
            shape, device=self.device
        )
        self._latest_observation = torch.zeros(
            self.num_envs, TEACHER_OBSERVATION_DIM, device=self.device
        )
        self._max_abs_wrench_seen = torch.zeros((), device=self.device)
        self.reset()

    def _exact_body_id(self, name: str) -> int:
        ids, names = self._robot.find_bodies(name, preserve_order=True)
        if len(ids) != 1 or names != [name]:
            raise RuntimeError(
                f"expected exactly one body named {name!r}, got ids={ids!r}, names={names!r}"
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

    def _format_observations(self, obs_dict) -> tuple[torch.Tensor, dict]:
        if not isinstance(obs_dict, dict) or "policy" not in obs_dict:
            raise RuntimeError("Teacher environment must return a policy observation group")
        obs = obs_dict["policy"]
        expected_shape = (self.num_envs, TEACHER_OBSERVATION_DIM)
        if not isinstance(obs, torch.Tensor) or tuple(obs.shape) != expected_shape:
            actual_shape = tuple(obs.shape) if isinstance(obs, torch.Tensor) else None
            raise RuntimeError(
                f"policy observation must have shape {expected_shape}, got {actual_shape}"
            )
        if obs.device != self.device or obs.dtype != torch.float32:
            raise RuntimeError(
                f"policy observation must use float32 on {self.device}, got "
                f"{obs.dtype} on {obs.device}"
            )
        if not bool(torch.isfinite(obs).all()):
            raise RuntimeError("policy observation must contain only finite values")
        self._latest_observation = obs.detach().clone()
        return obs, {"observations": {"critic": obs}}

    def get_observations(self):
        return self._format_observations(self.env.observation_manager.compute())

    def reset(self):
        obs_dict, _ = self.env.reset()
        self._reset_state(None)
        self._apply_wrench(self._disturbance.current_wrench_b)
        return self._format_observations(obs_dict)

    def _reset_state(self, env_ids: torch.Tensor | Sequence[int] | None) -> None:
        if self._base_composer is not None:
            self._base_composer.reset(env_ids)
        self._residual_composer.reset(env_ids)
        self._disturbance.reset(env_ids)
        if env_ids is None:
            normalized_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            normalized_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            normalized_ids = torch.tensor(
                list(env_ids), device=self.device, dtype=torch.long
            )
        self._last_final_action[normalized_ids] = 0
        self._last_trainable_residual[normalized_ids] = 0
        self.env.m1_teacher_trainable_residual[normalized_ids] = 0
        self.env.m1_teacher_previous_trainable_residual[normalized_ids] = 0

    def _validate_action_tensor(self, name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected_shape = (self.num_envs, M1_ACTION_DIM)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if value.device != self.device or value.dtype != torch.float32:
            raise ValueError(
                f"{name} must use float32 on {self.device}, got "
                f"{value.dtype} on {value.device}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")

    def _apply_wrench(self, wrench_b: torch.Tensor) -> None:
        if tuple(wrench_b.shape) != (self.num_envs, 6) or not bool(
            torch.isfinite(wrench_b).all()
        ):
            raise RuntimeError("scheduled wrench must be finite with shape (num_envs, 6)")
        self._max_abs_wrench_seen = torch.maximum(
            self._max_abs_wrench_seen, wrench_b.abs().max()
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

    def step(self, actions: torch.Tensor):
        self._validate_action_tensor("actions", actions)
        if self.stage == "A1":
            with torch.no_grad():
                base_residual = self._base_actor.act_inference(
                    self._latest_observation
                )
            self._validate_action_tensor("base_actor output", base_residual)
            base_action = self._base_composer.compose(
                torch.zeros_like(actions), base_residual
            )
        else:
            base_action = torch.zeros_like(actions)
        self.env.m1_teacher_previous_trainable_residual.copy_(
            self.env.m1_teacher_trainable_residual
        )
        clipped = actions.clamp(-1.0, 1.0)
        self.env.m1_teacher_trainable_residual.copy_(clipped)
        self._last_trainable_residual.copy_(clipped)
        final_action = self._residual_composer.compose(base_action, actions)
        self._last_final_action.copy_(final_action.detach())
        wrench_b = self._disturbance.advance()
        self._apply_wrench(wrench_b)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(final_action)
        expected_vector_shape = (self.num_envs,)
        if (
            not isinstance(rewards, torch.Tensor)
            or tuple(rewards.shape) != expected_vector_shape
            or not bool(torch.isfinite(rewards).all())
        ):
            raise RuntimeError("rewards must be a finite torch.Tensor")
        for name, value in (("terminated", terminated), ("truncated", truncated)):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected_vector_shape
                or value.dtype != torch.bool
            ):
                raise RuntimeError(
                    f"{name} must be a bool tensor with shape {expected_vector_shape}"
                )
        if not isinstance(extras, dict):
            raise RuntimeError("extras must be a dictionary")
        dones = terminated | truncated
        if bool(dones.any()):
            done_ids = dones.nonzero(as_tuple=False).flatten()
            self._reset_state(done_ids)
            self._apply_wrench(self._disturbance.current_wrench_b)
        extras["time_outs"] = truncated
        obs, obs_extras = self._format_observations(obs_dict)
        extras.setdefault("observations", {}).update(obs_extras["observations"])
        return obs, rewards, dones, extras

    @property
    def last_final_action(self) -> torch.Tensor:
        return self._last_final_action.clone()

    @property
    def last_trainable_residual(self) -> torch.Tensor:
        return self._last_trainable_residual.clone()

    @property
    def current_wrench_b(self) -> torch.Tensor:
        return self._disturbance.current_wrench_b

    @property
    def max_abs_wrench_seen(self) -> float:
        return float(self._max_abs_wrench_seen.item())

    @property
    def residual_composer(self) -> M1ResidualActionComposer:
        return self._residual_composer

    @property
    def base_composer(self) -> M1ResidualActionComposer:
        if self._base_composer is None:
            raise RuntimeError("A0 does not have a base composer")
        return self._base_composer

    @property
    def frozen_actor_hash(self) -> str | None:
        return self._frozen_actor_hash

    def assert_frozen_actor_unchanged(self) -> None:
        if self._base_actor is None or self._frozen_actor_hash is None:
            return
        current_hash = module_sha256(self._base_actor)
        if current_hash != self._frozen_actor_hash:
            raise RuntimeError(
                "frozen base actor changed: initial "
                f"{self._frozen_actor_hash}, current {current_hash}"
            )
