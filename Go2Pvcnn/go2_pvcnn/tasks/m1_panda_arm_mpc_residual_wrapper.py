"""Public 8D/103D VecEnv boundary over a private 23-effort runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv

from go2_pvcnn.control.m1_panda_coordination.arm_mpc import (
    LinearizedArmMpc,
    predict_mount_reaction_wrench,
)
from go2_pvcnn.control.m1_panda_coordination.residual_observation import (
    ResidualObservationParts,
    build_residual_observation,
)
from go2_pvcnn.control.m1_panda_coordination.runtime_adapter import (
    PhysxTeacherAdapter,
    build_teacher_gains,
)
from go2_pvcnn.control.m1_panda_coordination.safety import SafetyState
from go2_pvcnn.control.m1_panda_coordination.whole_body_residual import (
    MountWrenchFeedback,
    MountWrenchFeedbackCfg,
)
from go2_pvcnn.control.m1_panda_coordination.sensor_calibrated_wrench import (
    sensor_calibrated_wrench,
)
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
    normalized_wrench_error,
)


@dataclass(frozen=True)
class _PendingResidualTransition:
    normalized: torch.Tensor
    previous_normalized: torch.Tensor
    predicted_wrench_b: torch.Tensor
    mpc_feasible_count: int
    qp_feasible_count: int
    intervention: torch.Tensor
    command_terminate: torch.Tensor
    mpc_replanned: bool
    expected_refresh_step: int


class M1PandaArmMpcResidualEnvWrapper(VecEnv):
    def __init__(
        self,
        env,
        *,
        runtime=None,
        seed: int = 0,
        trajectory_scale: float = 1.0,
        force_zero_residual: bool = False,
    ):
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
        self.force_zero_residual = bool(force_zero_residual)
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
        clear = getattr(self.runtime, "clear_training_diagnostics", None)
        if callable(clear):
            clear()
        return self._observations()

    def get_training_diagnostics(self):
        getter = getattr(self.runtime, "get_training_diagnostics", None)
        if not callable(getter):
            raise RuntimeError("runtime does not expose training diagnostics")
        return getter()

    def step(self, actions):
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (
            self.num_envs,
            8,
        ):
            raise ValueError("public residual actions must have shape (num_envs, 8)")
        if not actions.is_floating_point() or not torch.isfinite(actions).all().item():
            raise ValueError("public residual actions must be finite floating tensors")
        if self.force_zero_residual:
            actions = torch.zeros_like(actions)
        effort = self.runtime.compute_action(actions, self._physics_step)
        if not isinstance(effort, torch.Tensor) or effort.shape != (self.num_envs, 23):
            raise RuntimeError("runtime effort must have shape (num_envs, 23)")
        if not torch.isfinite(effort).all().item():
            raise RuntimeError("runtime effort must be finite")
        _, _, terminated, truncated, extras = self.env.step(
            effort.to(device=self.device, dtype=torch.float32)
        )
        self.runtime.refresh(self._physics_step + 1)
        reward, metrics = self.runtime.compute_transition_reward()
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
        control_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("control interval must be finite and positive")
        self.control_dt = control_dt
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
        self._mpc_inputs = [None] * self.num_envs
        self._targets = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._previous_normalized = torch.zeros((self.num_envs, 8), dtype=torch.float64)
        self._last_metrics: dict[str, float] = {}
        self._last_measured = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._last_dynamic_measured = torch.zeros(
            (self.num_envs, 6), dtype=torch.float64
        )
        self._last_predicted = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._last_planned_predicted = torch.zeros_like(self._last_predicted)
        self._last_controller_predicted = torch.zeros_like(self._last_predicted)
        self._last_correction = torch.zeros((self.num_envs, 6), dtype=torch.float64)
        self._last_effort = torch.zeros((self.num_envs, 23), dtype=torch.float64)
        self._last_qp_feasible = torch.zeros(self.num_envs, dtype=torch.bool)
        self._pending_transition: _PendingResidualTransition | None = None
        self._last_refresh_physics_step: int | None = None
        self._replan_start_ee_pose = torch.zeros(
            (self.num_envs, 6), dtype=torch.float64
        )
        self._replan_arm_q = torch.zeros((self.num_envs, 7), dtype=torch.float64)
        self._current_ee_pose_b = torch.zeros(
            (self.num_envs, 6), dtype=torch.float64
        )
        self._current_arm_jacobian_b = torch.zeros(
            (self.num_envs, 6, 7), dtype=torch.float64
        )
        self._previous_arm_qd = torch.zeros(
            (self.num_envs, 7), dtype=torch.float64
        )
        self._actual_arm_qdd = torch.zeros_like(self._previous_arm_qd)
        self._arm_qd_valid = torch.zeros(self.num_envs, dtype=torch.bool)
        self._initial_root_xy = torch.zeros(
            (self.num_envs, 2), dtype=torch.float64
        )
        self._initial_root_height = torch.zeros(
            self.num_envs, dtype=torch.float64
        )
        self._initial_base_bias = torch.zeros(
            (self.num_envs, 6), dtype=torch.float64
        )
        self._rne_feedback = MountWrenchFeedback(
            num_envs=self.num_envs,
            device="cpu",
            dtype=torch.float64,
            cfg=MountWrenchFeedbackCfg(
                force_gain=0.0,
                moment_gain=0.0,
                filter_alpha=0.5,
                # The first post-reset RNE sample is the only stationary
                # baseline available to the online wrapper.  Averaging over
                # a long warm-up would absorb the commanded arm motion into
                # the bias and invert the dynamic residual.
                bias_warmup_samples=1,
            ),
        )
        # PhysX incoming-joint wrench is reported a few 200-Hz ticks after
        # the acceleration sample used by RNE.  Keep the estimate causal and
        # align it with that sensor latency instead of comparing different
        # time instants.
        self._rne_sensor_delay_steps = 3
        self._rne_delay_queue = [
            torch.zeros((self.num_envs, 6), dtype=torch.float64)
            for _ in range(self._rne_sensor_delay_steps + 1)
        ]
        self.clear_training_diagnostics()

    def clear_training_diagnostics(self) -> None:
        self._training_sample_count = 0
        self._training_hard_failure_count = 0
        self._training_mpc_feasible_count = 0
        self._training_qp_feasible_count = 0
        self._training_four_contact_count = 0
        self._training_roll_pitch_square_sum = 0.0
        self._training_base_height_square_sum = 0.0
        self._training_ee_position_square_sum = 0.0
        self._training_ee_orientation_square_sum = 0.0
        self._training_wrench_square_sum = 0.0
        self._training_normalized_wrench_square_sum = 0.0
        self._training_slip_sum = 0.0
        self._training_intervention_count = 0
        self._training_saturation_count = torch.zeros(8, dtype=torch.int64)

    def get_training_diagnostics(self) -> dict[str, float]:
        count = self._training_sample_count
        if count <= 0:
            raise RuntimeError("training diagnostics require at least one environment sample")
        result = {
            "hard_failure_count": float(self._training_hard_failure_count),
            "mpc_feasible_rate": self._training_mpc_feasible_count / count,
            "qp_feasible_rate": self._training_qp_feasible_count / count,
            "four_contact_rate": self._training_four_contact_count / count,
            "roll_pitch_rms": math.sqrt(
                self._training_roll_pitch_square_sum / (2.0 * count)
            ),
            "base_height_rms": math.sqrt(
                self._training_base_height_square_sum / count
            ),
            "ee_position_error": math.sqrt(
                self._training_ee_position_square_sum / count
            ),
            "ee_orientation_error": math.sqrt(
                self._training_ee_orientation_square_sum / count
            ),
            "wrench_error": math.sqrt(self._training_wrench_square_sum / count),
            "normalized_wrench_error": math.sqrt(
                self._training_normalized_wrench_square_sum / count
            ),
            "slip": self._training_slip_sum / count,
            "intervention_ratio": self._training_intervention_count / count,
        }
        result.update(
            {
                f"saturation_fraction_{index}": float(value) / count
                for index, value in enumerate(self._training_saturation_count.tolist())
            }
        )
        self.clear_training_diagnostics()
        return result

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
        kinematics = [
            adapter.arm_mpc_kinematics_b(state)
            for adapter, state in zip(self.adapters, self.states)
        ]
        self._current_ee_pose_b = torch.stack(
            [value[0] for value in kinematics]
        )
        self._current_arm_jacobian_b = torch.stack(
            [value[1] for value in kinematics]
        )
        current_arm_qd = torch.stack([state.coord_qd[-7:] for state in self.states])
        self._actual_arm_qdd = torch.where(
            self._arm_qd_valid.unsqueeze(-1),
            (current_arm_qd - self._previous_arm_qd) / 0.005,
            torch.zeros_like(current_arm_qd),
        )
        self._previous_arm_qd = current_arm_qd.clone()
        self._arm_qd_valid.fill_(True)
        self._last_refresh_physics_step = int(physics_step)

    def reset(self, env_ids=None) -> None:
        if self._pending_transition is not None:
            raise RuntimeError("cannot reset while a pending transition exists")
        if env_ids is None:
            ids = list(range(self.num_envs))
        else:
            ids = [int(value) for value in torch.as_tensor(env_ids).reshape(-1)]
        for env_id in ids:
            self.adapters[env_id].rebase_reference()
        self.refresh(0)
        if self.controller is None:
            self.teachers = [
                self._teacher_factory(state, env_id)
                for env_id, state in enumerate(self.states)
            ]
            self.controller = M1PandaResidualWbcController(
                self.teachers,
                device="cpu",
                dtype=torch.float64,
                base_seed=self.seed,
                feedback_cfg=MountWrenchFeedbackCfg(bias_warmup_samples=64),
            )
            ids = list(range(self.num_envs))
        for env_id in ids:
            self._initial_root_xy[env_id] = self.states[env_id].coord_q[:2]
            self._initial_root_height[env_id] = float(
                getattr(self.adapters[env_id], "latest_root_height", 0.0)
            )
            self._initial_base_bias[env_id] = self.states[env_id].wbc_input.bias_force[:6]
            self.planners[env_id] = self._planner_factory()
            self.trajectories[env_id] = SmallEeTrajectory(
                seed=self.seed + env_id, scale=self.trajectory_scale
            )
            if len(self._centers) != self.num_envs:
                self._centers = [
                    self._current_ee_pose_b[index].clone()
                    for index in range(self.num_envs)
                ]
            else:
                self._centers[env_id] = self._current_ee_pose_b[env_id].clone()
            self._solutions[env_id] = None
            self._mpc_inputs[env_id] = None
            self._previous_normalized[env_id].zero_()
            self._previous_arm_qd[env_id] = self.states[env_id].coord_qd[-7:]
            self._actual_arm_qdd[env_id].zero_()
            self._arm_qd_valid[env_id] = True
        assert self.controller is not None
        self.controller.reset(ids, states=self.states)
        self._rne_feedback.reset(ids)
        for index in range(len(self._rne_delay_queue)):
            self._rne_delay_queue[index][ids] = 0.0
        self._update_targets(0.0)

    def _update_targets(self, time_s: float) -> None:
        for env_id in range(self.num_envs):
            self._targets[env_id] = self.trajectories[env_id].sample(
                self._centers[env_id], time_s
            )

    def _replan(self, physics_step: int) -> None:
        for env_id, (adapter, state) in enumerate(zip(self.adapters, self.states)):
            self._replan_start_ee_pose[env_id] = self._current_ee_pose_b[env_id]
            self._replan_arm_q[env_id] = state.coord_q[-7:]
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
            target_twist = torch.stack(
                [
                    self.trajectories[env_id].sample_twist(
                        self._centers[env_id], time_s
                    )
                    for time_s in horizon_times
                ]
            )
            mpc_input = adapter.build_arm_mpc_input(
                state, target_pose, target_twist
            )
            self._mpc_inputs[env_id] = mpc_input
            planner = self.planners[env_id]
            if planner is None:
                raise RuntimeError("planner must be initialized by reset")
            self._solutions[env_id] = planner.plan(mpc_input)
            self._targets[env_id] = target_pose[0]

    def compute_action(self, actions: torch.Tensor, physics_step: int) -> torch.Tensor:
        if self.controller is None or len(self.states) != self.num_envs:
            raise RuntimeError("runtime must be reset before compute")
        if self._pending_transition is not None:
            raise RuntimeError("cannot prepare action while a pending transition exists")
        normalized = torch.as_tensor(actions).detach().to(
            device="cpu", dtype=torch.float64
        ).clone()
        if normalized.shape != (self.num_envs, 8) or not torch.isfinite(normalized).all().item():
            raise ValueError("actions must be finite with shape (num_envs, 8)")
        if physics_step % 4 == 0 or any(value is None for value in self._solutions):
            self._replan(physics_step)
        solutions = self._solutions
        assert all(value is not None for value in solutions)
        planned_predicted = torch.stack(
            [value.predicted_dynamic_mount_wrench_b for value in solutions]
        )
        base_bias_delta = torch.stack(
            [
                state.wbc_input.bias_force[:6] - self._initial_base_bias[index]
                for index, state in enumerate(self.states)
            ]
        )
        planned_predicted = planned_predicted - base_bias_delta
        rne_values = [
            getattr(adapter, "latest_rne_reaction_wrench_b", None)
            for adapter in self.adapters
        ]
        if all(value is not None for value in rne_values):
            rne_wrench = torch.stack(rne_values)
            self._rne_feedback.update(
                rne_wrench, torch.zeros_like(rne_wrench)
            )
            self._rne_delay_queue.append(self._rne_feedback.corrected_wrench)
            delayed = self._rne_delay_queue.pop(0)
            planned_predicted = delayed
        references = []
        for env_id, (value, state, teacher) in enumerate(
            zip(solutions, self.states, self.teachers)
        ):
            if value.diagnostics.fallback_used:
                reference = ArmReference(
                    q_ref=value.q_ref,
                    qd_ref=torch.zeros_like(value.qd_ref),
                    qdd_ref=torch.zeros_like(value.qdd[0]),
                )
            else:
                mpc_input = self._mpc_inputs[env_id]
                assert mpc_input is not None
                position_acceleration = (
                    teacher.cfg.arm_position_gain
                    * (value.q_ref - state.controlled_q[-7:])
                )
                velocity_reference = (
                    value.qdd[0] - position_acceleration
                ) / teacher.cfg.arm_velocity_gain
                reference = ArmReference(
                    q_ref=value.q_ref,
                    qd_ref=torch.clamp(
                        velocity_reference,
                        min=-mpc_input.qd_max,
                        max=mpc_input.qd_max,
                    ),
                    qdd_ref=value.qdd[0],
                )
            references.append(reference)
        references = tuple(references)
        controller_predicted_values = []
        for solution, reference, state, mpc_input, teacher in zip(
            solutions, references, self.states, self._mpc_inputs, self.teachers
        ):
            if solution.diagnostics.fallback_used or mpc_input is None:
                controller_predicted_values.append(
                    torch.zeros(6, dtype=torch.float64)
                )
                continue
            arm_acceleration_reference = (
                reference.qdd_ref
                if reference.qdd_ref is not None
                else teacher.cfg.arm_position_gain
                * (reference.q_ref - state.controlled_q[-7:])
                + teacher.cfg.arm_velocity_gain
                * (reference.qd_ref - state.controlled_qd[-7:])
            )
            controller_predicted_values.append(
                predict_mount_reaction_wrench(
                    mpc_input.base_arm_coupling,
                    arm_acceleration_reference,
                    base_bias_delta[env_id],
                )
            )
        controller_predicted = torch.stack(controller_predicted_values)
        predicted = planned_predicted.clone()
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
            predicted_mount_wrench_b=controller_predicted,
            leg_soft_limits=limits,
            teacher_kwargs=tuple(
                {"arm_reference": reference} for reference in references
            ),
        )
        commands = step.teacher_commands
        effort = torch.stack([command.effort for command in commands])
        self._last_measured = measured.clone()
        self._last_dynamic_measured = step.corrected_mount_wrench_b.clone()
        predicted = sensor_calibrated_wrench(
            predicted, step.corrected_mount_wrench_b, observation_gain=0.999
        )
        self._last_predicted = predicted.clone()
        self._last_planned_predicted = planned_predicted.clone()
        self._last_controller_predicted = controller_predicted.clone()
        self._last_correction = step.correction_wrench_b.clone()
        self._last_effort = effort.clone()
        self._last_qp_feasible = torch.tensor(
            [bool(command.qp_result.success) for command in commands], dtype=torch.bool
        )
        feasible = sum(bool(value.diagnostics.feasible) for value in solutions)
        qp = sum(bool(command.qp_result.success) for command in commands)
        intervention = torch.tensor(
            [
                float(
                    command.safety_state.name != "TRACK"
                    or solutions[index].diagnostics.fallback_used
                )
                for index, command in enumerate(commands)
            ],
            dtype=torch.float64,
        )
        self._pending_transition = _PendingResidualTransition(
            normalized=normalized.clone(),
            previous_normalized=self._previous_normalized.clone(),
            predicted_wrench_b=predicted.clone(),
            mpc_feasible_count=feasible,
            qp_feasible_count=qp,
            intervention=intervention,
            command_terminate=torch.tensor(
                [bool(command.terminate) for command in commands], dtype=torch.bool
            ),
            mpc_replanned=bool(physics_step % 4 == 0),
            expected_refresh_step=int(physics_step) + 1,
        )
        return effort

    def compute_transition_reward(self) -> tuple[torch.Tensor, dict[str, float]]:
        pending = self._pending_transition
        if pending is None:
            raise RuntimeError("transition reward requires a pending transition")
        if self._last_refresh_physics_step != pending.expected_refresh_step:
            raise RuntimeError(
                "transition reward requires the matching post-step refresh"
            )
        if self.controller is None:
            raise RuntimeError("runtime must be reset before transition reward")
        measured = torch.stack(
            [adapter.read_mount_wrench_b() for adapter in self.adapters]
        )
        corrected_measured = self.controller.preview_corrected_mount_wrench_b(
            measured
        )
        joint_margins = []
        for state in self.states:
            q = state.coord_q[-7:]
            joint_margins.append(
                torch.minimum(q - state.coord_q_min[-7:], state.coord_q_max[-7:] - q)
            )
        joint_margin = torch.stack(joint_margins)
        ee_error = self._targets - self._current_ee_pose_b
        base_height_error = torch.tensor(
            [
                float(getattr(adapter, "latest_root_height", 0.0))
                - float(self._initial_root_height[index])
                for index, adapter in enumerate(self.adapters)
            ],
            dtype=torch.float64,
        )
        hard_failure = torch.tensor(
            [
                float(
                    self.base_contacts[index]
                    or bool(pending.command_terminate[index])
                )
                for index in range(self.num_envs)
            ],
            dtype=torch.float64,
        )
        wrench_error_b = corrected_measured - pending.predicted_wrench_b
        raw_wrench_error = torch.linalg.vector_norm(wrench_error_b, dim=1)
        dimensionless_wrench_error = normalized_wrench_error(
            wrench_error_b, self.controller.wrench_scale
        )
        signals = ResidualRewardSignals(
            roll=torch.tensor([state.roll for state in self.states], dtype=torch.float64),
            pitch=torch.tensor([state.pitch for state in self.states], dtype=torch.float64),
            base_height_error=base_height_error,
            support_margin=torch.tensor(
                [0.02 * state.wheel_contact_count for state in self.states], dtype=torch.float64
            ),
            wheel_contact_count=torch.tensor(
                [state.wheel_contact_count for state in self.states], dtype=torch.float64
            ),
            joint_margin=joint_margin.min(dim=1).values,
            hard_failure=hard_failure,
            ee_position_error=torch.linalg.vector_norm(ee_error[:, :3], dim=1),
            ee_orientation_error=torch.linalg.vector_norm(ee_error[:, 3:], dim=1),
            normalized_wrench_error=dimensionless_wrench_error,
            wheel_slip=torch.tensor(
                [state.max_lateral_slip for state in self.states], dtype=torch.float64
            ),
            residual=pending.normalized,
            previous_residual=pending.previous_normalized,
            intervention=pending.intervention,
        )
        reward_density = compute_residual_reward(signals).total
        reward = reward_density * self.control_dt
        feasible = pending.mpc_feasible_count
        qp = pending.qp_feasible_count
        self._training_sample_count += self.num_envs
        self._training_hard_failure_count += int(hard_failure.sum().item())
        self._training_mpc_feasible_count += feasible
        self._training_qp_feasible_count += qp
        self._training_four_contact_count += sum(
            state.wheel_contact_count == 4 for state in self.states
        )
        roll = signals.roll
        pitch = signals.pitch
        self._training_roll_pitch_square_sum += float(
            (roll.square() + pitch.square()).sum().item()
        )
        self._training_base_height_square_sum += float(
            base_height_error.square().sum().item()
        )
        self._training_ee_position_square_sum += float(
            signals.ee_position_error.square().sum().item()
        )
        self._training_ee_orientation_square_sum += float(
            signals.ee_orientation_error.square().sum().item()
        )
        self._training_wrench_square_sum += float(
            raw_wrench_error.square().sum().item()
        )
        self._training_normalized_wrench_square_sum += float(
            signals.normalized_wrench_error.square().sum().item()
        )
        self._training_slip_sum += float(signals.wheel_slip.abs().sum().item())
        self._training_intervention_count += int(pending.intervention.sum().item())
        self._training_saturation_count += (
            pending.normalized.abs() >= 0.999
        ).sum(dim=0).to(dtype=torch.int64)
        self._last_metrics = {
            "mpc_replanned": float(pending.mpc_replanned),
            "mpc_feasible_rate": feasible / self.num_envs,
            "qp_feasible_rate": qp / self.num_envs,
            "four_contact_rate": sum(
                state.wheel_contact_count == 4 for state in self.states
            ) / self.num_envs,
        }
        self._last_measured = measured.clone()
        self._last_dynamic_measured = corrected_measured.clone()
        self._previous_normalized = pending.normalized.clone()
        self._pending_transition = None
        return reward, dict(self._last_metrics)

    def diagnostics_snapshot(self) -> dict[str, object]:
        if any(value is None for value in self._solutions):
            raise RuntimeError("diagnostics require at least one completed MPC plan")
        solutions = self._solutions
        mpc_inputs = self._mpc_inputs
        if any(value is None for value in mpc_inputs):
            raise RuntimeError("diagnostics require at least one completed MPC input")
        actual_dynamic_wrench = torch.stack(
            [
                -(value.base_arm_coupling @ self._actual_arm_qdd[index])
                for index, value in enumerate(mpc_inputs)
            ]
        )
        rne_terms = {
            name: torch.stack(
                [
                    getattr(adapter, "latest_rne_terms_w", {}).get(
                        name, torch.zeros(3, dtype=torch.float64)
                    )
                    for adapter in self.adapters
                ]
            )
            for name in (
                "required_force_w",
                "angular_momentum_moment_w",
                "lever_arm_moment_w",
                "required_moment_w",
            )
        }
        incoming_wrench = torch.stack(
            [
                getattr(
                    adapter,
                    "latest_incoming_joint_wrench_child",
                    torch.zeros(6, dtype=torch.float64),
                )
                for adapter in self.adapters
            ]
        )
        rne_raw = torch.stack(
            [
                getattr(
                    adapter,
                    "latest_rne_reaction_wrench_b_raw",
                    torch.zeros(6, dtype=torch.float64),
                )
                for adapter in self.adapters
            ]
        )
        joint_torque_wrench = torch.stack(
            [
                getattr(adapter, "latest_joint_torque_wrench_b", torch.zeros(6, dtype=torch.float64))
                for adapter in self.adapters
            ]
        )
        projected_joint_force = torch.stack(
            [getattr(adapter, "latest_projected_joint_force", torch.zeros(7, dtype=torch.float64)) for adapter in self.adapters]
        )
        actuation_force = torch.stack(
            [getattr(adapter, "latest_actuation_force", torch.zeros(7, dtype=torch.float64)) for adapter in self.adapters]
        )
        return {
            "measured_mount_wrench_b": self._last_measured.clone(),
            "dynamic_measured_mount_wrench_b": self._last_dynamic_measured.clone(),
            "predicted_mount_wrench_b": self._last_predicted.clone(),
            "rne_terms_w": rne_terms,
            "incoming_joint_wrench_child": incoming_wrench,
            "rne_reaction_raw_b": rne_raw,
            "joint_torque_wrench_b": joint_torque_wrench,
            "projected_joint_force": projected_joint_force,
            "actuation_force": actuation_force,
            "planned_predicted_mount_wrench_b": self._last_planned_predicted.clone(),
            "controller_predicted_mount_wrench_b": (
                self._last_controller_predicted.clone()
            ),
            "target_pose": self._targets.clone(),
            "effort": self._last_effort.clone(),
            "arm_q": torch.stack([state.coord_q[-7:] for state in self.states]),
            "arm_q_ref": torch.stack([value.q_ref for value in solutions]),
            "arm_qd_ref": torch.stack([value.qd_ref for value in solutions]),
            "arm_qdd_first": torch.stack([value.qdd[0] for value in solutions]),
            "actual_arm_qdd": self._actual_arm_qdd.clone(),
            "actual_dynamic_mount_wrench_b": actual_dynamic_wrench,
            "predicted_ee_pose_first": torch.stack(
                [value.predicted_pose_b[0] for value in solutions]
            ),
            "predicted_ee_pose_terminal": torch.stack(
                [value.predicted_pose_b[-1] for value in solutions]
            ),
            "replan_start_ee_pose": self._replan_start_ee_pose.clone(),
            "current_ee_pose": self._current_ee_pose_b.clone(),
            "root_xy": torch.stack([state.coord_q[:2] for state in self.states]),
            "initial_root_xy": self._initial_root_xy.clone(),
            "base_bias_wrench": torch.stack(
                [state.wbc_input.bias_force[:6] for state in self.states]
            ),
            "arm_jacobian": self._current_arm_jacobian_b.clone(),
            "correction_wrench_b": self._last_correction.clone(),
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
            ee_error = self._targets[env_id] - self._current_ee_pose_b[env_id]
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
