"""Public 8D/103D VecEnv boundary over a private 23-effort runtime."""

from __future__ import annotations

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

from go2_pvcnn.control.m1_panda_coordination.arm_mpc import LinearizedArmMpc
from go2_pvcnn.control.m1_panda_coordination.residual_observation import (
    ResidualObservationParts,
    build_residual_observation,
)
from go2_pvcnn.control.m1_panda_coordination.runtime_adapter import (
    PhysxTeacherAdapter,
    build_teacher_gains,
)
from go2_pvcnn.control.m1_panda_coordination.safety import SafetyState
from go2_pvcnn.control.m1_panda_coordination.teacher import (
    ArmReference,
    M1PandaWbcTeacher,
)
from go2_pvcnn.tasks.m1_panda_residual_wbc_wrapper import (
    M1PandaResidualWbcController,
)
from go2_pvcnn.tasks.mdp.m1_panda_arm_mpc_residual import (
    ResidualRewardSignals,
    SmallEeTrajectory,
    compute_residual_reward,
)


class M1PandaArmMpcResidualEnvWrapper(VecEnv):
    def __init__(self, env, *, runtime=None, seed: int = 0, trajectory_scale: float = 1.0):
        if int(env.action_manager.total_action_dim) != 23:
            raise ValueError("private Arm-MPC environment must expose 23 efforts")
        if runtime is None:
            runtime = M1PandaArmMpcResidualRuntime(
                env, seed=seed, trajectory_scale=trajectory_scale
            )
        if int(runtime.num_envs) != int(env.num_envs):
            raise ValueError("runtime num_envs must match environment")
        self.env = env
        self.runtime = runtime
        self.num_envs = int(env.num_envs)
        self.num_actions = 8
        self.device = torch.device(env.device)
        self.max_episode_length = env.max_episode_length
        self.env.action_space = gym.spaces.Box(-1.0, 1.0, (8,), dtype="float32")
        self._physics_step = 0

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
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return gym.spaces.Box(-float("inf"), float("inf"), (103,), dtype="float32")

    def _observations(self):
        observation = self.runtime.observations()
        if not isinstance(observation, torch.Tensor) or tuple(observation.shape) != (
            self.num_envs,
            103,
        ):
            raise RuntimeError("runtime observations must have shape (num_envs, 103)")
        if not observation.is_floating_point() or not torch.isfinite(observation).all().item():
            raise RuntimeError("runtime observations must be finite floating tensors")
        observation = observation.to(device=self.device, dtype=torch.float32)
        return observation, {"observations": {"critic": observation}}

    def get_observations(self):
        return self._observations()

    def reset(self):
        self.env.reset()
        self._physics_step = 0
        self.runtime.reset()
        return self._observations()

    def step(self, actions):
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (
            self.num_envs,
            8,
        ):
            raise ValueError("public residual actions must have shape (num_envs, 8)")
        if not actions.is_floating_point() or not torch.isfinite(actions).all().item():
            raise ValueError("public residual actions must be finite floating tensors")
        effort, reward, metrics = self.runtime.compute(actions, self._physics_step)
        if not isinstance(effort, torch.Tensor) or effort.shape != (self.num_envs, 23):
            raise RuntimeError("runtime effort must have shape (num_envs, 23)")
        if not torch.isfinite(effort).all().item():
            raise RuntimeError("runtime effort must be finite")
        _, _, terminated, truncated, extras = self.env.step(
            effort.to(device=self.device, dtype=torch.float32)
        )
        refresh = getattr(self.runtime, "refresh", None)
        if callable(refresh):
            refresh(self._physics_step + 1)
        dones = terminated | truncated
        if bool(dones.any().item()):
            self.runtime.reset(dones.nonzero(as_tuple=False).flatten().cpu())
        self._physics_step += 1
        observation, observation_extras = self._observations()
        extras = dict(extras or {})
        extras.setdefault("observations", {}).update(observation_extras["observations"])
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        extras["environment_metrics"] = dict(metrics)
        return observation, reward.to(device=self.device, dtype=torch.float32), dones, extras


class M1PandaArmMpcResidualRuntime:
    """Per-environment 50 Hz MPC and 200 Hz residual-WBC orchestration."""

    def __init__(
        self,
        env,
        *,
        seed: int,
        trajectory_scale: float,
        adapter_factory=None,
        teacher_factory=None,
        planner_factory=None,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self.env = env
        self.num_envs = int(env.num_envs)
        self.seed = seed
        self.trajectory_scale = float(trajectory_scale)
        self._adapter_factory = adapter_factory or (
            lambda owner, env_id: PhysxTeacherAdapter(owner, env_index=env_id)
        )
        self._teacher_factory = teacher_factory or self._default_teacher
        self._planner_factory = planner_factory or LinearizedArmMpc
        self.adapters = [
            self._adapter_factory(env, env_id) for env_id in range(self.num_envs)
        ]
        self.planners = [None] * self.num_envs
        self.trajectories = [
            SmallEeTrajectory(seed=seed + env_id, scale=self.trajectory_scale)
            for env_id in range(self.num_envs)
        ]
        self.states = []
        self.base_contacts = [0] * self.num_envs
        self.teachers = []
        self.controller = None
        self._centers = []
        self._solutions = [None] * self.num_envs
        self._targets = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._previous_normalized = torch.zeros((self.num_envs, 8), dtype=torch.float64)
        self._last_metrics: dict[str, float] = {}
        self._last_measured = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._last_predicted = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._last_effort = torch.zeros((self.num_envs, 23), dtype=torch.float64)
        self._last_qp_feasible = torch.zeros(self.num_envs, dtype=torch.bool)

    @staticmethod
    def _default_teacher(state, env_id):
        kp, kd = build_teacher_gains()
        return M1PandaWbcTeacher(
            kp=kp,
            kd=kd,
            effort_limit=state.wbc_input.effort_limit,
            safe_arm_target=state.controlled_q[-7:],
        )

    def refresh(self, physics_step: int) -> None:
        built = [adapter.build_state(physics_step) for adapter in self.adapters]
        self.states = [value[0] for value in built]
        self.base_contacts = [int(value[1]) for value in built]

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            ids = list(range(self.num_envs))
        else:
            ids = [int(value) for value in torch.as_tensor(env_ids).reshape(-1)]
        self.refresh(0)
        if self.controller is None:
            self.teachers = [
                self._teacher_factory(state, env_id)
                for env_id, state in enumerate(self.states)
            ]
            self.controller = M1PandaResidualWbcController(
                self.teachers, device="cpu", dtype=torch.float64, base_seed=self.seed
            )
            ids = list(range(self.num_envs))
        for env_id in ids:
            self.planners[env_id] = self._planner_factory()
            self.trajectories[env_id] = SmallEeTrajectory(
                seed=self.seed + env_id, scale=self.trajectory_scale
            )
            if len(self._centers) != self.num_envs:
                self._centers = [state.ee_pose.clone() for state in self.states]
            else:
                self._centers[env_id] = self.states[env_id].ee_pose.clone()
            self._solutions[env_id] = None
            self._previous_normalized[env_id].zero_()
        assert self.controller is not None
        self.controller.reset(ids, states=self.states)
        self._update_targets(0.0)

    def _update_targets(self, time_s: float) -> None:
        for env_id in range(self.num_envs):
            self._targets[env_id] = self.trajectories[env_id].sample(
                self._centers[env_id], time_s
            )

    def _replan(self, physics_step: int) -> None:
        for env_id, (adapter, state) in enumerate(zip(self.adapters, self.states)):
            horizon_times = [
                state.time_s + (index + 1) * 0.02 for index in range(20)
            ]
            target_pose = torch.stack(
                [
                    self.trajectories[env_id].sample(
                        self._centers[env_id], time_s
                    )
                    for time_s in horizon_times
                ]
            )
            target_twist = torch.zeros((20, 6), dtype=torch.float64)
            mpc_input = adapter.build_arm_mpc_input(
                state, target_pose, target_twist
            )
            planner = self.planners[env_id]
            if planner is None:
                raise RuntimeError("planner must be initialized by reset")
            self._solutions[env_id] = planner.plan(mpc_input)
            self._targets[env_id] = target_pose[0]

    def compute(self, actions: torch.Tensor, physics_step: int):
        if self.controller is None or len(self.states) != self.num_envs:
            raise RuntimeError("runtime must be reset before compute")
        normalized = torch.as_tensor(actions).detach().to(
            device="cpu", dtype=torch.float64
        ).clone()
        if normalized.shape != (self.num_envs, 8) or not torch.isfinite(normalized).all().item():
            raise ValueError("actions must be finite with shape (num_envs, 8)")
        if physics_step % 4 == 0 or any(value is None for value in self._solutions):
            self._replan(physics_step)
        solutions = self._solutions
        assert all(value is not None for value in solutions)
        predicted = torch.stack(
            [value.predicted_dynamic_mount_wrench_b for value in solutions]
        )
        references = tuple(
            ArmReference(q_ref=value.q_ref, qd_ref=value.qd_ref)
            for value in solutions
        )
        measured = torch.stack(
            [adapter.read_mount_wrench_b() for adapter in self.adapters]
        )
        limits = torch.stack(
            [adapter.leg_soft_limits() for adapter in self.adapters]
        )
        step = self.controller.step(
            states=self.states,
            normalized_residual=normalized,
            measured_mount_wrench_b=measured,
            predicted_mount_wrench_b=predicted,
            leg_soft_limits=limits,
            teacher_kwargs=tuple(
                {"arm_reference": reference} for reference in references
            ),
        )
        commands = step.teacher_commands
        effort = torch.stack([command.effort for command in commands])
        self._last_measured = measured.clone()
        self._last_predicted = predicted.clone()
        self._last_effort = effort.clone()
        self._last_qp_feasible = torch.tensor(
            [bool(command.qp_result.success) for command in commands], dtype=torch.bool
        )
        joint_margins = []
        for state in self.states:
            q = state.coord_q[-7:]
            joint_margins.append(
                torch.minimum(q - state.coord_q_min[-7:], state.coord_q_max[-7:] - q)
            )
        joint_margin = torch.stack(joint_margins)
        ee_error = self._targets - torch.stack([state.ee_pose for state in self.states])
        signals = ResidualRewardSignals(
            roll=torch.tensor([state.roll for state in self.states], dtype=torch.float64),
            pitch=torch.tensor([state.pitch for state in self.states], dtype=torch.float64),
            base_height_error=torch.zeros(self.num_envs, dtype=torch.float64),
            support_margin=torch.tensor(
                [0.02 * state.wheel_contact_count for state in self.states], dtype=torch.float64
            ),
            wheel_contact_count=torch.tensor(
                [state.wheel_contact_count for state in self.states], dtype=torch.float64
            ),
            joint_margin=joint_margin.min(dim=1).values,
            hard_failure=torch.tensor(
                [
                    float(self.base_contacts[index] or commands[index].terminate)
                    for index in range(self.num_envs)
                ], dtype=torch.float64,
            ),
            ee_position_error=torch.linalg.vector_norm(ee_error[:, :3], dim=1),
            ee_orientation_error=torch.linalg.vector_norm(ee_error[:, 3:], dim=1),
            wrench_error=torch.linalg.vector_norm(measured - predicted, dim=1),
            wheel_slip=torch.tensor(
                [state.max_lateral_slip for state in self.states], dtype=torch.float64
            ),
            residual=normalized,
            previous_residual=self._previous_normalized,
            intervention=torch.tensor(
                [
                    float(
                        command.safety_state.name != "TRACK"
                        or solutions[index].diagnostics.fallback_used
                    )
                    for index, command in enumerate(commands)
                ], dtype=torch.float64,
            ),
        )
        reward = compute_residual_reward(signals).total
        self._previous_normalized = normalized.clone()
        feasible = sum(bool(value.diagnostics.feasible) for value in solutions)
        qp = sum(bool(command.qp_result.success) for command in commands)
        self._last_metrics = {
            "mpc_replanned": float(physics_step % 4 == 0),
            "mpc_feasible_rate": feasible / self.num_envs,
            "qp_feasible_rate": qp / self.num_envs,
            "four_contact_rate": sum(
                state.wheel_contact_count == 4 for state in self.states
            ) / self.num_envs,
        }
        return effort, reward, dict(self._last_metrics)

    def diagnostics_snapshot(self) -> dict[str, object]:
        if any(value is None for value in self._solutions):
            raise RuntimeError("diagnostics require at least one completed MPC plan")
        solutions = self._solutions
        return {
            "measured_mount_wrench_b": self._last_measured.clone(),
            "predicted_mount_wrench_b": self._last_predicted.clone(),
            "target_pose": self._targets.clone(),
            "effort": self._last_effort.clone(),
            "mpc_fallback": torch.tensor(
                [value.diagnostics.fallback_used for value in solutions],
                dtype=torch.bool,
            ),
            "mpc_feasible": torch.tensor(
                [value.diagnostics.feasible for value in solutions],
                dtype=torch.bool,
            ),
            "qp_feasible": self._last_qp_feasible.clone(),
        }

    def observations(self) -> torch.Tensor:
        if self.controller is None or len(self.states) != self.num_envs:
            raise RuntimeError("runtime must be reset before observations")
        result = []
        filtered = self.controller.filtered_mount_wrench_b
        for env_id, state in enumerate(self.states):
            q = state.coord_q[-7:]
            margin = torch.minimum(
                q - state.coord_q_min[-7:], state.coord_q_max[-7:] - q
            )
            ee_error = self._targets[env_id] - state.ee_pose
            m1 = torch.cat(
                (
                    state.controlled_q,
                    state.controlled_qd,
                    state.coord_q[:3],
                    state.coord_qd[:3],
                    torch.tensor(
                        (
                            state.roll, state.pitch, float(state.wheel_contact_count),
                            state.max_lateral_slip, float(state.signals_finite),
                            state.time_s, float(self.base_contacts[env_id]),
                        ), dtype=torch.float64,
                    ),
                )
            )
            arm = torch.cat((q, state.coord_qd[-7:], ee_error))
            parts = ResidualObservationParts(
                m1_state=m1,
                arm_state=arm,
                filtered_mount_wrench=filtered[env_id],
                task_state=ee_error,
                sigma_min=state.sigma_min.reshape(1),
                joint_limit_margin_min=margin.min().reshape(1),
                joint_limit_margin_mean=margin.mean().reshape(1),
                support_margin=torch.tensor(
                    [0.02 * state.wheel_contact_count], dtype=torch.float64
                ),
                previous_residual=self._previous_normalized[env_id],
            )
            result.append(build_residual_observation(parts).flat)
        return torch.stack(result).to(dtype=torch.float32)


__all__ = ["M1PandaArmMpcResidualEnvWrapper", "M1PandaArmMpcResidualRuntime"]
