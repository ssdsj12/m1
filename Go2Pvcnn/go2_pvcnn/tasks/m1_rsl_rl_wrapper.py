"""RSL-RL VecEnv wrapper for the M1 smoke task."""

from __future__ import annotations

import gymnasium as gym
import torch
import isaaclab.utils.math as math_utils

from rsl_rl.env import VecEnv

from go2_pvcnn.tasks.m1_curriculum import (
    build_wave_reference_actions,
    build_semantic_spatial_wave_reference,
    build_spatial_axle_wheel_targets,
    build_temporal_axle_wheel_targets,
    build_lateral_steering_correction,
    semantic_obstacle_ahead_mask,
    update_wave_encounter_phase,
    smooth_wave_reference_actions,
    update_clearance_drive_release,
    wheel_x_from_fixed_obstacle,
    update_sequential_wheel_crossing_reference,
    sequential_active_leg_mask,
    sequential_leg_residual_scale,
    compose_sequential_leg_actions,
    build_sequential_phase_observation,
    blend_policy_wave_gate,
    apply_fixed_course_gate_safety_window,
    update_axle_pair_crossing_reference,
    build_stabilized_task_space_wheel_actions,
    merge_task_space_support_with_jointspace_active,
    resolve_m1_leg_joint_ids_by_wheel,
)


def _mask_wave_gate_after_root_x(
    wave_gate: torch.Tensor,
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    disable_after_x: float | None,
) -> torch.Tensor:
    if disable_after_x is None:
        return wave_gate
    root_x = root_pos_w[:, 0] - env_origins[:, 0]
    return wave_gate & (root_x <= float(disable_after_x))


def _gate_wave_leg_actions(
    leg_actions: torch.Tensor,
    wave_gate: torch.Tensor | None,
    action_limit: float | None,
) -> torch.Tensor:
    """Lock the default stance outside wave and optionally limit active wave actions."""
    prepared = leg_actions
    if wave_gate is not None:
        prepared = prepared * wave_gate.to(leg_actions.dtype).unsqueeze(-1)
    if action_limit is not None:
        prepared = torch.clamp(prepared, -float(action_limit), float(action_limit))
    return prepared


def _gate_from_spatial_reference(
    obstacle_active: torch.Tensor, spatial_reference: torch.Tensor
) -> torch.Tensor:
    """Release policy legs only while a front/rear crossing window is active."""
    active = torch.as_tensor(
        obstacle_active, dtype=torch.bool, device=spatial_reference.device
    )
    reference_active = torch.abs(spatial_reference).amax(dim=1) > 1.0e-6
    return active & reference_active


def _combine_wave_reference_actions(
    leg_actions: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    return leg_actions + reference


def _slowest_forward_wheel_velocity(actual_wheel_velocity: torch.Tensor) -> torch.Tensor:
    """Use the slowest wheel as reference without spreading reverse motion."""
    return actual_wheel_velocity.amin(dim=1, keepdim=True).clamp_min(0.0)


def _wheel_forward_velocity(
    actual_wheel_velocity: torch.Tensor,
    wheel_signs,
) -> torch.Tensor:
    """Map raw joint-axis velocity into the common robot-forward direction."""
    if wheel_signs is None:
        return actual_wheel_velocity
    signs = torch.as_tensor(
        wheel_signs,
        dtype=actual_wheel_velocity.dtype,
        device=actual_wheel_velocity.device,
    ).reshape(1, 4)
    return actual_wheel_velocity * signs


def _update_wheel_sync_integral(
    *, previous: torch.Tensor, error: torch.Tensor, step_dt: float, limit: float
) -> torch.Tensor:
    """Accumulate wheel speed error until the environment resets it."""
    return torch.clamp(
        previous + error * float(step_dt),
        -float(limit),
        float(limit),
    )


class M1RslRlEnvWrapper(VecEnv):
    """Flatten the M1 smoke `policy` observation group for RSL-RL."""

    def __init__(self, env, clip_actions: float | None = 1.0):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self._wheel_joint_ids = None
        self._wheel_body_ids = None
        self._leg_joint_ids = None
        self._task_space_nominal_wheel_pos_b = None
        self._task_space_nominal_base_height = None
        self._task_space_balanced_wheel_x_b = None
        self._wheel_sync_integral = None
        self._wave_elapsed_s = None
        self._wave_obstacle_active = None
        self._smoothed_wave_reference = None
        self._clearance_drive_axle = None
        self._clearance_drive_released = None
        self._clearance_phase_elapsed_s = None
        self._sequential_crossing_phase = None
        self._sequential_crossing_phase_steps = None
        self._sequential_drive_allowed = None
        if hasattr(env, "action_manager"):
            self.num_actions = env.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(env.single_action_space)
        if clip_actions is not None:
            self.env.action_space = gym.spaces.Box(
                low=-clip_actions,
                high=clip_actions,
                shape=(self.num_actions,),
                dtype=env.action_space.dtype,
            )
        self.env.reset()

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
        policy_groups = [obs_dict["policy"]]
        policy_groups.extend(
            value
            for name, value in sorted(obs_dict.items())
            if name.startswith("policy_")
        )
        obs = torch.cat(
            [group.reshape(group.shape[0], -1) for group in policy_groups],
            dim=-1,
        )
        if bool(getattr(self.cfg, "wave_policy_phase_observation", False)):
            obs = torch.cat((obs, self._phase_observation(obs)), dim=1)
        return obs, {"observations": {"critic": obs}}

    def _phase_observation(self, reference: torch.Tensor) -> torch.Tensor:
        if self._sequential_crossing_phase is None:
            phase = torch.full(
                (reference.shape[0],), -1, dtype=torch.long, device=reference.device
            )
            phase_steps = torch.zeros_like(phase)
        else:
            phase = self._sequential_crossing_phase
            phase_steps = self._sequential_crossing_phase_steps
        return build_sequential_phase_observation(
            phase=phase,
            phase_steps=phase_steps,
            progress_steps=int(
                getattr(self.cfg, "wave_policy_phase_progress_steps", 50)
            ),
        ).to(dtype=reference.dtype, device=reference.device)

    def get_observations(self):
        obs_dict = self.env.unwrapped.observation_manager.compute()
        return self._format_observations(obs_dict)

    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._format_observations(obs_dict)

    def _build_stabilized_task_space_reference(
        self,
        *,
        robot,
        actions: torch.Tensor,
        wheel_pos_w: torch.Tensor,
        crossing_obstacle_x: torch.Tensor,
        axle_pair_mode: bool,
    ) -> torch.Tensor:
        if self._leg_joint_ids is None:
            self._leg_joint_ids = torch.as_tensor(
                resolve_m1_leg_joint_ids_by_wheel(robot.joint_names),
                dtype=torch.long,
                device=actions.device,
            )
        leg_ids_by_wheel = self._leg_joint_ids
        leg_joint_ids = leg_ids_by_wheel.reshape(-1)
        jacobians = robot.root_physx_view.get_jacobians()
        root_rotation_b_w = math_utils.matrix_from_quat(
            math_utils.quat_inv(robot.data.root_quat_w)
        )
        wheel_pos_b = torch.einsum(
            "nij,nkj->nki",
            root_rotation_b_w,
            wheel_pos_w - robot.data.root_pos_w.unsqueeze(1),
        )
        if self._task_space_nominal_wheel_pos_b is None:
            self._task_space_nominal_wheel_pos_b = wheel_pos_b.detach().clone()
            self._task_space_nominal_base_height = (
                robot.data.root_pos_w[:, 2].detach().clone()
            )
        inactive = self._sequential_crossing_phase < 0
        self._task_space_nominal_wheel_pos_b[inactive] = wheel_pos_b[inactive].detach()
        self._task_space_nominal_base_height[inactive] = (
            robot.data.root_pos_w[inactive, 2].detach()
        )
        base_height_error = torch.clamp(
            (
                self._task_space_nominal_base_height
                - robot.data.root_pos_w[:, 2]
            )
            * float(getattr(self.cfg, "wave_task_space_base_height_gain", 1.0)),
            min=-float(getattr(self.cfg, "wave_task_space_base_height_max_m", 0.10)),
            max=float(getattr(self.cfg, "wave_task_space_base_height_max_m", 0.10)),
        )
        stabilized_nominal_wheel_pos_b = self._task_space_nominal_wheel_pos_b.clone()
        stabilized_nominal_wheel_pos_b[..., 2] -= base_height_error.unsqueeze(-1)
        if self._task_space_balanced_wheel_x_b is None:
            self._task_space_balanced_wheel_x_b = wheel_pos_b[..., 0].detach().clone()
        self._task_space_balanced_wheel_x_b[inactive] = (
            wheel_pos_b[inactive, :, 0].detach()
        )
        if axle_pair_mode:
            balance_steps = int(
                getattr(self.cfg, "wave_task_space_balance_steps", 20)
            )
            balance_reached = self._sequential_crossing_phase_steps == balance_steps
            self._task_space_balanced_wheel_x_b[balance_reached] = (
                wheel_pos_b[balance_reached, :, 0].detach()
            )

        joint_column_offset = 0 if robot.is_fixed_base else 6
        wheel_xyz_jacobians = []
        for wheel_index in range(4):
            body_index = int(self._wheel_body_ids[wheel_index].item())
            if robot.is_fixed_base:
                body_index -= 1
            joint_columns = leg_ids_by_wheel[wheel_index] + joint_column_offset
            position_jacobian_w = jacobians[:, body_index, :3, joint_columns]
            wheel_xyz_jacobians.append(
                torch.bmm(root_rotation_b_w, position_jacobian_w)
            )
        wheel_xyz_jacobians = torch.stack(wheel_xyz_jacobians, dim=1)
        leg_joint_pos = robot.data.joint_pos.index_select(
            1, leg_joint_ids
        ).reshape(actions.shape[0], 4, 3)
        default_leg_joint_pos = robot.data.default_joint_pos.index_select(
            1, leg_joint_ids
        ).reshape(actions.shape[0], 4, 3)
        obstacle_delta_w = torch.zeros_like(robot.data.root_pos_w)
        obstacle_delta_w[:, 0] = crossing_obstacle_x
        obstacle_delta_w[:, 1] = (
            self.env.scene.env_origins[:, 1] - robot.data.root_pos_w[:, 1]
        )
        obstacle_x_b = torch.bmm(
            root_rotation_b_w, obstacle_delta_w.unsqueeze(-1)
        ).squeeze(-1)[:, 0]
        reference = build_stabilized_task_space_wheel_actions(
            phase=self._sequential_crossing_phase,
            phase_steps=self._sequential_crossing_phase_steps,
            wheel_pos_b=wheel_pos_b,
            nominal_wheel_pos_b=stabilized_nominal_wheel_pos_b,
            wheel_x_obstacle_b=obstacle_x_b,
            leg_joint_pos=leg_joint_pos,
            default_leg_joint_pos=default_leg_joint_pos,
            wheel_xyz_jacobians=wheel_xyz_jacobians,
            lift_delta=float(getattr(self.cfg, "wave_task_space_lift_delta_m", 0.10)),
            past_bar_x=float(getattr(self.cfg, "wave_sequential_past_bar_x_m", 0.13)),
            action_scale=float(getattr(self.cfg, "wave_task_space_action_scale", 0.80)),
            damping=float(getattr(self.cfg, "wave_task_space_ik_damping", 0.05)),
            max_joint_step=float(
                getattr(self.cfg, "wave_task_space_max_joint_step", 0.15)
            ),
            lateral_body_shift=float(
                getattr(self.cfg, "wave_task_space_lateral_body_shift_m", 0.05)
            ),
            swing_with_body=True,
            axle_pair_mode=axle_pair_mode,
            pair_support_extension=float(
                getattr(
                    self.cfg,
                    "wave_task_space_pair_support_extension_m",
                    0.08,
                )
            ),
            pair_body_shift_x=float(
                getattr(self.cfg, "wave_task_space_pair_body_shift_x_m", 0.15)
            ),
            balanced_wheel_x_b=self._task_space_balanced_wheel_x_b,
            balance_steps=int(
                getattr(self.cfg, "wave_task_space_balance_steps", 20)
            ),
            lift_ramp_steps=int(
                getattr(self.cfg, "wave_task_space_lift_ramp_steps", 10)
            ),
        )
        self.env.unwrapped.m1_task_space_ik_joint_ids = leg_ids_by_wheel.detach()
        self.env.unwrapped.m1_task_space_ik_jacobians = wheel_xyz_jacobians.detach()
        self.env.unwrapped.m1_task_space_ik_full_jacobians = jacobians.detach()
        self.env.unwrapped.m1_task_space_ik_actions = reference.detach()
        return reference

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        self.env.unwrapped.m1_raw_policy_actions = actions.detach()
        raw_actions = actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        if bool(getattr(self.cfg, "roll_equal_wheel_actions", False)):
            prepared = torch.zeros_like(actions)
            raw = torch.clamp(actions[:, 12:13], -1.0, 1.0)
            # The simplified cylindrical wheel colliders use positive velocity for robot-forward (+X).
            wheel_action = 0.40 + 0.05 * raw
            wheel_targets = wheel_action.expand(-1, 4).clone()
            if bool(getattr(self.cfg, "roll_sync_actual_wheel_velocity", False)):
                robot = self.env.scene["robot"]
                if self._wheel_joint_ids is None:
                    wheel_joint_ids, _ = robot.find_joints(".*_FOOT_JOINT", preserve_order=True)
                    self._wheel_joint_ids = torch.as_tensor(
                        wheel_joint_ids, dtype=torch.long, device=actions.device
                    )
                actual_wheel_velocity = robot.data.joint_vel.index_select(1, self._wheel_joint_ids)
                forward_wheel_velocity = _wheel_forward_velocity(
                    actual_wheel_velocity,
                    getattr(self.cfg, "wave_wheel_action_signs", None),
                )
                wheel_speed_error = wheel_targets - forward_wheel_velocity
                if self._wheel_sync_integral is None:
                    self._wheel_sync_integral = torch.zeros_like(wheel_speed_error)
                step_dt = float(self.cfg.sim.dt * self.cfg.decimation)
                integral_limit = float(
                    getattr(self.cfg, "roll_wheel_sync_integral_limit", 0.50)
                )
                self._wheel_sync_integral = torch.clamp(
                    self._wheel_sync_integral + wheel_speed_error * step_dt,
                    -integral_limit,
                    integral_limit,
                )
                correction = (
                    float(getattr(self.cfg, "roll_wheel_sync_gain", 0.50)) * wheel_speed_error
                    + float(getattr(self.cfg, "roll_wheel_sync_integral_gain", 1.0))
                    * self._wheel_sync_integral
                )
                correction_limit = float(
                    getattr(self.cfg, "roll_wheel_sync_max_correction", 0.50)
                )
                wheel_targets += torch.clamp(
                    correction, -correction_limit, correction_limit
                )
                actual_mean = actual_wheel_velocity.mean(dim=1, keepdim=True)
                equalize_correction = float(
                    getattr(self.cfg, "roll_wheel_equalize_gain", 0.0)
                ) * (actual_mean - actual_wheel_velocity)
                equalize_limit = float(
                    getattr(self.cfg, "roll_wheel_equalize_max_correction", 0.50)
                )
                wheel_targets += torch.clamp(
                    equalize_correction, -equalize_limit, equalize_limit
                )
            prepared[:, 12:16] = wheel_targets
            return prepared

        if bool(getattr(self.cfg, "wave_fixed_forward_wheels", False)):
            prepared = torch.zeros_like(actions)
            leg_limit = getattr(self.cfg, "wave_leg_action_limit", 0.15)
            residual_limit = getattr(self.cfg, "wave_policy_leg_residual_limit", leg_limit)
            policy_actions = (
                raw_actions
                if bool(getattr(self.cfg, "wave_unclipped_policy_legs", False))
                else actions
            )
            leg_actions = policy_actions[:, :12]
            if residual_limit is not None:
                leg_actions = torch.clamp(
                    leg_actions, -float(residual_limit), float(residual_limit)
                )
            step_dt = float(self.cfg.sim.dt * self.cfg.decimation)
            wave_gate = None
            reference = None
            spatial_obstacle_x = None
            if bool(getattr(self.cfg, "wave_semantic_obstacle_gating", False)):
                robot = self.env.scene["robot"]
                scanner = self.env.scene[
                    str(getattr(self.cfg, "wave_semantic_scanner_name", "semantic_height_scanner"))
                ]
                if bool(getattr(self.cfg, "wave_spatial_reference", False)):
                    reference, obstacle_active, spatial_obstacle_x = build_semantic_spatial_wave_reference(
                        root_pos_w=robot.data.root_pos_w,
                        root_quat_w=robot.data.root_quat_w,
                        ray_hits_w=scanner.data.ray_hits_w,
                        semantic_map=scanner.data.semantic_map,
                        amplitude=float(getattr(self.cfg, "wave_reference_raw_amplitude", 0.04)),
                        knee_ratio=float(getattr(self.cfg, "wave_reference_knee_ratio", 1.5)),
                        rear_amplitude_scale=float(
                            getattr(self.cfg, "wave_rear_amplitude_scale", 1.0)
                        ),
                        front_overlap_scale=float(
                            getattr(self.cfg, "wave_front_overlap_scale", 1.0)
                        ),
                        front_support_ratio=float(
                            getattr(self.cfg, "wave_front_support_ratio", 0.0)
                        ),
                        rear_support_ratio=float(
                            getattr(self.cfg, "wave_rear_support_ratio", 0.0)
                        ),
                        front_lift_window=tuple(
                            getattr(
                                self.cfg,
                                "wave_front_lift_window",
                                (0.80, 0.65, 0.05, -0.10),
                            )
                        ),
                        rear_lift_window=tuple(
                            getattr(
                                self.cfg,
                                "wave_rear_lift_window",
                                (0.15, 0.0, -0.40, -0.60),
                            )
                        ),
                        corridor_half_width_m=float(
                            getattr(self.cfg, "wave_semantic_gate_half_width", 0.25)
                        ),
                    )
                    wave_gate = obstacle_active
                    if bool(
                        getattr(self.cfg, "wave_gate_from_spatial_reference", False)
                    ):
                        wave_gate = _gate_from_spatial_reference(
                            obstacle_active, reference
                        )
                    self.env.unwrapped.m1_wave_spatial_obstacle_x = (
                        spatial_obstacle_x.detach()
                    )
                else:
                    obstacle_active = semantic_obstacle_ahead_mask(
                        root_pos_w=robot.data.root_pos_w,
                        root_quat_w=robot.data.root_quat_w,
                        ray_hits_w=scanner.data.ray_hits_w,
                        semantic_map=scanner.data.semantic_map,
                        approach_min_m=float(getattr(self.cfg, "wave_semantic_gate_min_x", -0.35)),
                        approach_max_m=float(getattr(self.cfg, "wave_semantic_gate_max_x", 0.80)),
                        corridor_half_width_m=float(
                            getattr(self.cfg, "wave_semantic_gate_half_width", 0.25)
                        ),
                    )
                if reference is None and bool(getattr(self.cfg, "wave_reset_phase_on_obstacle", False)):
                    if self._wave_elapsed_s is None or self._wave_elapsed_s.shape != obstacle_active.shape:
                        self._wave_elapsed_s = torch.full_like(
                            obstacle_active, -1.0, dtype=actions.dtype
                        )
                        self._wave_obstacle_active = torch.zeros_like(obstacle_active)
                    frequency = float(getattr(self.cfg, "wave_reference_frequency", 0.5))
                    minimum_duration = float(
                        getattr(self.cfg, "wave_minimum_gate_duration_s", 1.0 / frequency)
                    )
                    self._wave_elapsed_s, self._wave_obstacle_active, wave_gate = (
                        update_wave_encounter_phase(
                            obstacle_active=obstacle_active,
                            previous_active=self._wave_obstacle_active,
                            elapsed_s=self._wave_elapsed_s,
                            step_dt=step_dt,
                            minimum_duration_s=minimum_duration,
                            maximum_duration_s=getattr(
                                self.cfg, "wave_single_cycle_duration_s", None
                            ),
                        )
                    )
                elif reference is None:
                    wave_gate = obstacle_active
            if wave_gate is not None and bool(
                getattr(self.cfg, "wave_gate_from_policy_action", False)
            ):
                oracle_wave_gate = wave_gate.clone()
                gate_index = int(
                    getattr(self.cfg, "wave_policy_gate_action_index", 15)
                )
                wave_gate, policy_gate_score = blend_policy_wave_gate(
                    oracle_gate=oracle_wave_gate,
                    policy_score=raw_actions[:, gate_index],
                    policy_weight=float(
                        getattr(self.cfg, "wave_policy_gate_weight", 1.0)
                    ),
                    threshold=float(
                        getattr(self.cfg, "wave_policy_gate_threshold", 0.0)
                    ),
                )
                minimum_root_x = getattr(
                    self.cfg, "wave_policy_gate_minimum_root_x_m", None
                )
                fallback_root_x = getattr(
                    self.cfg, "wave_policy_gate_fallback_root_x_m", None
                )
                if minimum_root_x is not None and fallback_root_x is not None:
                    root_local_x = (
                        robot.data.root_pos_w[:, 0]
                        - self.env.scene.env_origins[:, 0]
                    )
                    wave_gate, gate_fallback = (
                        apply_fixed_course_gate_safety_window(
                            policy_gate=wave_gate,
                            oracle_gate=oracle_wave_gate,
                            root_local_x=root_local_x,
                            minimum_root_x=float(minimum_root_x),
                            fallback_root_x=float(fallback_root_x),
                        )
                    )
                    self.env.unwrapped.m1_wave_policy_gate_fallback = (
                        gate_fallback.detach()
                    )
                self.env.unwrapped.m1_wave_gate_target = oracle_wave_gate.detach()
                self.env.unwrapped.m1_wave_policy_gate_score = (
                    policy_gate_score.detach()
                )
            leg_wave_gate = wave_gate
            if (
                (
                    bool(getattr(self.cfg, "wave_sequential_crossing_reference", False))
                    or bool(getattr(self.cfg, "wave_axle_pair_crossing_reference", False))
                )
                and wave_gate is not None
                and spatial_obstacle_x is not None
            ):
                robot = self.env.scene["robot"]
                if self._wheel_body_ids is None:
                    wheel_body_ids, _ = robot.find_bodies(
                        ".*_FOOT_LINK", preserve_order=True
                    )
                    self._wheel_body_ids = torch.as_tensor(
                        wheel_body_ids, dtype=torch.long, device=actions.device
                    )
                wheel_pos_w = robot.data.body_pos_w.index_select(1, self._wheel_body_ids)
                wheel_heights = wheel_pos_w[..., 2]
                crossing_obstacle_x = spatial_obstacle_x
                fixed_obstacle_center_x = getattr(
                    self.cfg, "wave_fixed_obstacle_center_x_m", None
                )
                if fixed_obstacle_center_x is not None:
                    root_local_x = (
                        robot.data.root_pos_w[:, 0]
                        - self.env.scene.env_origins[:, 0]
                    )
                    crossing_obstacle_x = (
                        float(fixed_obstacle_center_x) - root_local_x
                    )
                wheel_x_from_obstacle = wheel_x_from_fixed_obstacle(
                    wheel_pos_w=wheel_pos_w,
                    root_pos_w=robot.data.root_pos_w,
                    obstacle_x_from_root=crossing_obstacle_x,
                )
                if (
                    self._sequential_crossing_phase is None
                    or self._sequential_crossing_phase.shape != wave_gate.shape
                ):
                    self._sequential_crossing_phase = torch.full(
                        wave_gate.shape, -1, dtype=torch.long, device=actions.device
                    )
                    self._sequential_crossing_phase_steps = torch.zeros_like(
                        self._sequential_crossing_phase
                    )
                common_kwargs = dict(
                    obstacle_x=crossing_obstacle_x,
                    wave_gate=wave_gate,
                    wheel_x_from_obstacle=wheel_x_from_obstacle,
                    wheel_heights=wheel_heights,
                    previous_phase=self._sequential_crossing_phase,
                    previous_phase_steps=self._sequential_crossing_phase_steps,
                    required_height=float(getattr(self.cfg, "wave_axle_clearance_height_m", 0.16)),
                    past_bar_x=float(getattr(self.cfg, "wave_sequential_past_bar_x_m", 0.13)),
                )
                if bool(getattr(self.cfg, "wave_axle_pair_crossing_reference", False)):
                    (
                        self._sequential_crossing_phase,
                        self._sequential_crossing_phase_steps,
                        sequential_reference,
                        self._sequential_drive_allowed,
                        active_leg_mask,
                    ) = update_axle_pair_crossing_reference(
                        **common_kwargs,
                        ramp_steps=int(getattr(self.cfg, "wave_axle_pair_ramp_steps", 5)),
                        swing_steps=int(getattr(self.cfg, "wave_sequential_swing_steps", 50)),
                        restore_steps=int(
                            getattr(self.cfg, "wave_axle_pair_restore_steps", 20)
                        ),
                        support_steps=int(
                            getattr(self.cfg, "wave_axle_pair_support_steps", 20)
                        ),
                        curriculum_swing_timeout_steps=getattr(
                            self.cfg, "wave_pair_curriculum_swing_timeout_steps", None
                        ),
                        front_start_x=float(
                            getattr(self.cfg, "wave_axle_pair_front_start_x_m", -0.25)
                        ),
                        rear_start_obstacle_x=float(
                            getattr(self.cfg, "wave_sequential_rear_start_x", 0.05)
                        ),
                    )
                    leg_wave_gate = active_leg_mask.any(dim=1)
                    joint_space_reference = sequential_reference
                    if bool(getattr(self.cfg, "wave_task_space_ik", False)):
                        sequential_reference = (
                            self._build_stabilized_task_space_reference(
                                robot=robot,
                                actions=actions,
                                wheel_pos_w=wheel_pos_w,
                                crossing_obstacle_x=crossing_obstacle_x,
                                axle_pair_mode=True,
                            )
                        )
                    active_leg_scale = torch.zeros_like(
                        active_leg_mask, dtype=leg_actions.dtype
                    )
                else:
                    (
                        self._sequential_crossing_phase,
                        self._sequential_crossing_phase_steps,
                        sequential_reference,
                        self._sequential_drive_allowed,
                        leg_wave_gate,
                    ) = update_sequential_wheel_crossing_reference(
                        **common_kwargs,
                        swing_steps=int(getattr(self.cfg, "wave_sequential_swing_steps", 50)),
                        min_lift_steps=int(getattr(self.cfg, "wave_sequential_min_lift_steps", 5)),
                        ramp_steps=int(getattr(self.cfg, "wave_sequential_ramp_steps", 10)),
                        restore_steps=int(getattr(self.cfg, "wave_sequential_restore_steps", 20)),
                        clearance_target_height=float(
                            getattr(self.cfg, "wave_sequential_clearance_target_height_m", 0.20)
                        ),
                        support_extension=float(
                            getattr(self.cfg, "wave_sequential_support_extension", 0.0)
                        ),
                        opposite_abduction=float(
                            getattr(self.cfg, "wave_sequential_opposite_abduction", 0.0)
                        ),
                        balance_steps=int(
                            getattr(self.cfg, "wave_sequential_balance_steps", 0)
                        ),
                        front_hip_action=float(
                            getattr(self.cfg, "wave_sequential_front_hip_action", -1.50)
                        ),
                        front_knee_action=float(
                            getattr(self.cfg, "wave_sequential_front_knee_action", -0.80)
                        ),
                        rear_hip_action=float(
                            getattr(self.cfg, "wave_sequential_rear_hip_action", 1.50)
                        ),
                        rear_knee_action=float(
                            getattr(self.cfg, "wave_sequential_rear_knee_action", 0.80)
                        ),
                        right_track_only=bool(
                            getattr(self.cfg, "wave_right_track_only", False)
                        ),
                        keep_drive_during_wave=bool(
                            getattr(self.cfg, "wave_sequential_keep_drive_during_wave", False)
                        ),
                        front_start_x=float(
                            getattr(self.cfg, "wave_sequential_front_start_x_m", -0.20)
                        ),
                        front_restore_obstacle_x=float(
                            getattr(self.cfg, "wave_sequential_front_restore_x", 0.30)
                        ),
                        rear_start_obstacle_x=float(
                            getattr(self.cfg, "wave_sequential_rear_start_x", 0.05)
                        ),
                        rear_restore_obstacle_x=float(
                            getattr(self.cfg, "wave_sequential_rear_restore_x", -0.55)
                        ),
                    )
                    joint_space_reference = sequential_reference
                    if bool(getattr(self.cfg, "wave_task_space_ik", False)):
                        if self._leg_joint_ids is None:
                            self._leg_joint_ids = torch.as_tensor(
                                resolve_m1_leg_joint_ids_by_wheel(robot.joint_names),
                                dtype=torch.long,
                                device=actions.device,
                            )
                        leg_ids_by_wheel = self._leg_joint_ids
                        leg_joint_ids = leg_ids_by_wheel.reshape(-1)
                        jacobians = robot.root_physx_view.get_jacobians()
                        root_rotation_b_w = math_utils.matrix_from_quat(
                            math_utils.quat_inv(robot.data.root_quat_w)
                        )
                        wheel_pos_from_root_w = (
                            wheel_pos_w - robot.data.root_pos_w.unsqueeze(1)
                        )
                        wheel_pos_b = torch.einsum(
                            "nij,nkj->nki",
                            root_rotation_b_w,
                            wheel_pos_from_root_w,
                        )
                        if self._task_space_nominal_wheel_pos_b is None:
                            self._task_space_nominal_wheel_pos_b = wheel_pos_b.detach().clone()
                            self._task_space_nominal_base_height = (
                                robot.data.root_pos_w[:, 2].detach().clone()
                            )
                        inactive = self._sequential_crossing_phase < 0
                        self._task_space_nominal_wheel_pos_b[inactive] = (
                            wheel_pos_b[inactive].detach()
                        )
                        self._task_space_nominal_base_height[inactive] = (
                            robot.data.root_pos_w[inactive, 2].detach()
                        )
                        base_height_error = torch.clamp(
                            (
                                self._task_space_nominal_base_height
                                - robot.data.root_pos_w[:, 2]
                            )
                            * float(
                                getattr(self.cfg, "wave_task_space_base_height_gain", 1.0)
                            ),
                            min=-float(
                                getattr(self.cfg, "wave_task_space_base_height_max_m", 0.10)
                            ),
                            max=float(
                                getattr(self.cfg, "wave_task_space_base_height_max_m", 0.10)
                            ),
                        )
                        stabilized_nominal_wheel_z = (
                            self._task_space_nominal_wheel_pos_b[..., 2]
                            - base_height_error.unsqueeze(-1)
                        )
                        joint_column_offset = 0 if robot.is_fixed_base else 6
                        wheel_xyz_jacobians = []
                        for wheel_index in range(4):
                            body_index = int(self._wheel_body_ids[wheel_index].item())
                            if robot.is_fixed_base:
                                body_index -= 1
                            joint_columns = (
                                leg_ids_by_wheel[wheel_index] + joint_column_offset
                            )
                            position_jacobian_w = jacobians[
                                :, body_index, :3, joint_columns
                            ]
                            position_jacobian_b = torch.bmm(
                                root_rotation_b_w, position_jacobian_w
                            )
                            wheel_xyz_jacobians.append(position_jacobian_b)
                        wheel_xyz_jacobians = torch.stack(
                            wheel_xyz_jacobians, dim=1
                        )
                        leg_joint_pos = robot.data.joint_pos.index_select(
                            1, leg_joint_ids
                        ).reshape(actions.shape[0], 4, 3)
                        default_leg_joint_pos = robot.data.default_joint_pos.index_select(
                            1, leg_joint_ids
                        ).reshape(actions.shape[0], 4, 3)
                        obstacle_delta_w = torch.zeros_like(robot.data.root_pos_w)
                        obstacle_delta_w[:, 0] = crossing_obstacle_x
                        obstacle_delta_w[:, 1] = (
                            self.env.scene.env_origins[:, 1]
                            - robot.data.root_pos_w[:, 1]
                        )
                        obstacle_x_b = torch.bmm(
                            root_rotation_b_w, obstacle_delta_w.unsqueeze(-1)
                        ).squeeze(-1)[:, 0]
                        stabilized_nominal_wheel_pos_b = (
                            self._task_space_nominal_wheel_pos_b.clone()
                        )
                        stabilized_nominal_wheel_pos_b[..., 2] = (
                            stabilized_nominal_wheel_z
                        )
                        root_lateral_offset = (
                            robot.data.root_pos_w[:, 1]
                            - self.env.scene.env_origins[:, 1]
                        )
                        task_space_reference = build_stabilized_task_space_wheel_actions(
                            phase=self._sequential_crossing_phase,
                            phase_steps=self._sequential_crossing_phase_steps,
                            wheel_pos_b=wheel_pos_b,
                            nominal_wheel_pos_b=stabilized_nominal_wheel_pos_b,
                            wheel_x_obstacle_b=obstacle_x_b,
                            leg_joint_pos=leg_joint_pos,
                            default_leg_joint_pos=default_leg_joint_pos,
                            wheel_xyz_jacobians=wheel_xyz_jacobians,
                            lift_delta=float(
                                getattr(self.cfg, "wave_task_space_lift_delta_m", 0.10)
                            ),
                            rear_lift_delta=float(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_rear_lift_delta_m",
                                    getattr(self.cfg, "wave_task_space_lift_delta_m", 0.10),
                                )
                            ),
                            past_bar_x=float(
                                getattr(self.cfg, "wave_sequential_past_bar_x_m", 0.13)
                            ),
                            action_scale=float(
                                getattr(self.cfg, "wave_task_space_action_scale", 0.80)
                            ),
                            damping=float(
                                getattr(self.cfg, "wave_task_space_ik_damping", 0.05)
                            ),
                            max_joint_step=float(
                                getattr(self.cfg, "wave_task_space_max_joint_step", 0.15)
                            ),
                            lateral_body_shift=float(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_lateral_body_shift_m",
                                    0.05,
                                )
                            ),
                            swing_with_body=bool(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_swing_with_body",
                                    True,
                                )
                            ),
                            balance_steps=int(
                                getattr(self.cfg, "wave_task_space_balance_steps", 20)
                            ),
                            lift_ramp_steps=int(
                                getattr(self.cfg, "wave_task_space_lift_ramp_steps", 10)
                            ),
                            rear_restore_steps=int(
                                getattr(self.cfg, "wave_sequential_restore_steps", 20)
                            ),
                            rear_restore_forward_offset=float(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_rear_restore_forward_offset_m",
                                    0.0,
                                )
                            ),
                            base_lateral_offset=root_lateral_offset,
                            lateral_recovery_gain=float(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_lateral_recovery_gain",
                                    0.0,
                                )
                            ),
                            lateral_recovery_max=float(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_lateral_recovery_max_m",
                                    0.0,
                                )
                            ),
                            stabilize_supports=bool(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_stabilize_supports",
                                    False,
                                )
                            ),
                            swing_ramp_steps=int(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_swing_ramp_steps",
                                    1,
                                )
                            ),
                            longitudinal_body_shift=float(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_longitudinal_body_shift_m",
                                    0.0,
                                )
                            ),
                            balance_supports=bool(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_balance_supports",
                                    False,
                                )
                            ),
                            active_swing_xz_only=bool(
                                getattr(
                                    self.cfg,
                                    "wave_task_space_active_swing_xz_only",
                                    False,
                                )
                            ),
                        )
                        if bool(
                            getattr(self.cfg, "wave_task_space_support_only", False)
                        ):
                            sequential_reference = (
                                merge_task_space_support_with_jointspace_active(
                                    task_space_reference=task_space_reference,
                                    joint_space_reference=joint_space_reference,
                                    phase=self._sequential_crossing_phase,
                                )
                            )
                        else:
                            sequential_reference = task_space_reference
                        task_space_active_phase = (
                            (
                                (self._sequential_crossing_phase >= 0)
                                & (self._sequential_crossing_phase <= 4)
                            )
                            | (
                                (self._sequential_crossing_phase >= 6)
                                & (self._sequential_crossing_phase <= 10)
                            )
                        )
                        if bool(
                            getattr(self.cfg, "wave_task_space_stop_during_wave", True)
                        ):
                            self._sequential_drive_allowed &= ~task_space_active_phase
                        self.env.unwrapped.m1_task_space_ik_joint_ids = (
                            leg_ids_by_wheel.detach()
                        )
                        self.env.unwrapped.m1_task_space_ik_jacobians = (
                            wheel_xyz_jacobians.detach()
                        )
                        self.env.unwrapped.m1_task_space_ik_full_jacobians = (
                            jacobians.detach()
                        )
                        self.env.unwrapped.m1_task_space_ik_actions = (
                            task_space_reference.detach()
                        )
                    active_leg_scale = sequential_leg_residual_scale(
                        self._sequential_crossing_phase,
                        support_scale=float(
                            getattr(
                                self.cfg,
                                "wave_sequential_support_residual_scale",
                                0.75,
                            )
                        ),
                        crossing_scale=float(
                            getattr(
                                self.cfg,
                                "wave_sequential_crossing_residual_scale",
                                0.25,
                            )
                        ),
                        support_abduction_scale=float(
                            getattr(
                                self.cfg,
                                "wave_sequential_support_abduction_residual_scale",
                                0.15,
                            )
                        ),
                    )
                reference = sequential_reference
                leg_actions = compose_sequential_leg_actions(
                    policy_actions=leg_actions,
                    teacher_actions=sequential_reference,
                    residual_scale=active_leg_scale.to(leg_actions.dtype),
                    policy_control=bool(
                        getattr(self.cfg, "wave_sequential_policy_control", False)
                    ),
                    policy_weight=float(
                        getattr(self.cfg, "wave_sequential_policy_weight", 1.0)
                    ),
                )
                self.env.unwrapped.m1_wave_reference_actions = (
                    sequential_reference.detach()
                )
                self.env.unwrapped.m1_sequential_crossing_phase = (
                    self._sequential_crossing_phase.detach()
                )
                self.env.unwrapped.m1_sequential_crossing_phase_steps = (
                    self._sequential_crossing_phase_steps.detach()
                )
                self.env.unwrapped.m1_sequential_wheel_x_from_obstacle = (
                    wheel_x_from_obstacle.detach()
                )
                self.env.unwrapped.m1_sequential_wheel_heights = (
                    wheel_heights.detach()
                )
            if bool(getattr(self.cfg, "wave_reference_actions", False)):
                if reference is None:
                    if self._wave_elapsed_s is not None and bool(
                        getattr(self.cfg, "wave_reset_phase_on_obstacle", False)
                    ):
                        reference_time = torch.clamp(self._wave_elapsed_s, min=0.0)
                    else:
                        reference_time = self.episode_length_buf.to(actions.dtype) * step_dt
                    constant_phase = getattr(
                        self.cfg, "wave_reference_constant_phase_s", None
                    )
                    if constant_phase is None:
                        reference_time = reference_time + float(
                            getattr(self.cfg, "wave_reference_time_offset_s", 0.0)
                        )
                    else:
                        reference_time = torch.full_like(
                            reference_time, float(constant_phase)
                        )
                    reference = build_wave_reference_actions(
                        episode_time_s=reference_time,
                        amplitude=float(getattr(self.cfg, "wave_reference_raw_amplitude", 0.04)),
                        knee_ratio=float(getattr(self.cfg, "wave_reference_knee_ratio", 1.5)),
                        frequency=float(getattr(self.cfg, "wave_reference_frequency", 0.5)),
                        rear_amplitude_scale=float(
                            getattr(self.cfg, "wave_rear_amplitude_scale", 1.0)
                        ),
                        front_support_ratio=float(
                            getattr(self.cfg, "wave_front_support_ratio", 0.0)
                        ),
                        rear_support_ratio=float(
                            getattr(self.cfg, "wave_rear_support_ratio", 0.0)
                        ),
                        pulse_ramp_s=getattr(
                            self.cfg, "wave_reference_pulse_ramp_s", None
                        ),
                        pulse_hold_s=getattr(
                            self.cfg, "wave_reference_pulse_hold_s", None
                        ),
                    )
                smoothing_alpha = getattr(
                    self.cfg, "wave_reference_smoothing_alpha", None
                )
                if smoothing_alpha is not None:
                    if (
                        self._smoothed_wave_reference is None
                        or self._smoothed_wave_reference.shape != reference.shape
                    ):
                        self._smoothed_wave_reference = torch.zeros_like(reference)
                    reference = smooth_wave_reference_actions(
                        previous=self._smoothed_wave_reference,
                        target=reference,
                        active=(
                            wave_gate
                            if wave_gate is not None
                            else torch.ones(
                                reference.shape[0], dtype=torch.bool, device=reference.device
                            )
                        ),
                        alpha=float(smoothing_alpha),
                    )
                    self._smoothed_wave_reference = reference.detach()
                self.env.unwrapped.m1_wave_reference_actions = reference.detach()
                leg_actions = _combine_wave_reference_actions(leg_actions, reference)
            if wave_gate is not None:
                disable_after_x = getattr(self.cfg, "wave_disable_obstacle_after_root_x", None)
                if disable_after_x is not None:
                    robot = self.env.scene["robot"]
                    wave_gate = _mask_wave_gate_after_root_x(
                        wave_gate=wave_gate,
                        root_pos_w=robot.data.root_pos_w,
                        env_origins=self.env.scene.env_origins,
                        disable_after_x=float(disable_after_x),
                    )
            if bool(getattr(self.cfg, "wave_left_right_symmetric", False)):
                symmetric_pairs = ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))
                for left_index, right_index in symmetric_pairs:
                    pair_action = 0.5 * (
                        leg_actions[:, left_index : left_index + 1]
                        + leg_actions[:, right_index : right_index + 1]
                    )
                    leg_actions[:, left_index : left_index + 1] = pair_action
                    leg_actions[:, right_index : right_index + 1] = pair_action
            if bool(getattr(self.cfg, "wave_lock_abduction", False)):
                leg_actions[:, (0, 3, 6, 9)] = 0.0
            leg_actions = _gate_wave_leg_actions(leg_actions, leg_wave_gate, leg_limit)
            self.env.unwrapped.m1_wave_gate = (
                leg_wave_gate.detach()
                if leg_wave_gate is not None
                else torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
            )
            self.env.unwrapped.m1_prepared_leg_actions = leg_actions.detach()
            prepared[:, :12] = leg_actions
            raw = torch.clamp(actions[:, 12:13], -1.0, 1.0)
            residual_scale = float(getattr(self.cfg, "wave_wheel_residual_scale", 0.02))
            front_action = float(getattr(self.cfg, "wave_front_wheel_action", 0.40)) + residual_scale * raw
            rear_action = float(getattr(self.cfg, "wave_rear_wheel_action", 0.40)) + residual_scale * raw
            obstacle_wheel_action = getattr(self.cfg, "wave_obstacle_wheel_action", None)
            if obstacle_wheel_action is not None and wave_gate is not None:
                obstacle_front_action = (
                    float(getattr(self.cfg, "wave_obstacle_front_wheel_action", obstacle_wheel_action))
                    + residual_scale * raw
                )
                obstacle_rear_action = (
                    float(getattr(self.cfg, "wave_obstacle_rear_wheel_action", obstacle_wheel_action))
                    + residual_scale * raw
                )
                active_column = wave_gate.unsqueeze(-1)
                front_action = torch.where(active_column, obstacle_front_action, front_action)
                rear_action = torch.where(active_column, obstacle_rear_action, rear_action)
            obstacle_boost = float(getattr(self.cfg, "wave_obstacle_wheel_boost", 0.0))
            if obstacle_boost > 0.0:
                robot = self.env.scene["robot"]
                root_x = robot.data.root_pos_w[:, 0] - self.env.scene.env_origins[:, 0]
                boost_active = (
                    (root_x >= float(getattr(self.cfg, "wave_obstacle_boost_start_x", 0.05)))
                    & (root_x <= float(getattr(self.cfg, "wave_obstacle_boost_end_x", 0.40)))
                ).to(dtype=actions.dtype).unsqueeze(-1)
                front_action = front_action + obstacle_boost * boost_active
                rear_action = rear_action + obstacle_boost * boost_active
            wheel_targets = torch.cat(
                (front_action.expand(-1, 2), rear_action.expand(-1, 2)), dim=1
            )
            swing_wheel_action = getattr(
                self.cfg, "wave_sequential_swing_wheel_action", None
            )
            if (
                swing_wheel_action is not None
                and self._sequential_crossing_phase is not None
            ):
                swing_phase = (
                    (self._sequential_crossing_phase == 1)
                    | (self._sequential_crossing_phase == 3)
                    | (self._sequential_crossing_phase == 7)
                    | (self._sequential_crossing_phase == 9)
                )
                swing_targets = (
                    float(swing_wheel_action) + residual_scale * raw
                ).expand(-1, 4)
                wheel_targets = torch.where(
                    swing_phase.unsqueeze(-1), swing_targets, wheel_targets
                )
            phase_wheel_assist = float(
                getattr(self.cfg, "wave_phase_wheel_assist", 0.0)
            )
            if phase_wheel_assist > 0.0 and wave_gate is not None:
                phase_base_action = float(
                    getattr(self.cfg, "wave_obstacle_front_wheel_action", 0.40)
                )
                if spatial_obstacle_x is not None:
                    phase_targets = build_spatial_axle_wheel_targets(
                        obstacle_x=spatial_obstacle_x,
                        active=wave_gate,
                        base_action=phase_base_action,
                        assist_action=phase_wheel_assist,
                    )
                elif self._wave_elapsed_s is not None:
                    phase_targets = build_temporal_axle_wheel_targets(
                        episode_time_s=torch.clamp(self._wave_elapsed_s, min=0.0),
                        active=wave_gate,
                        frequency=float(
                            getattr(self.cfg, "wave_reference_frequency", 0.5)
                        ),
                        base_action=phase_base_action,
                        assist_action=phase_wheel_assist,
                    )
                else:
                    phase_targets = wheel_targets
                phase_targets = phase_targets + residual_scale * raw
                wheel_targets = torch.where(
                    wave_gate.unsqueeze(-1), phase_targets, wheel_targets
                )
            if (
                bool(getattr(self.cfg, "wave_hold_wheels_until_axle_clear", False))
                and wave_gate is not None
                and spatial_obstacle_x is not None
            ):
                robot = self.env.scene["robot"]
                if self._wheel_body_ids is None:
                    wheel_body_ids, _ = robot.find_bodies(
                        ".*_FOOT_LINK", preserve_order=True
                    )
                    self._wheel_body_ids = torch.as_tensor(
                        wheel_body_ids, dtype=torch.long, device=actions.device
                    )
                wheel_heights = robot.data.body_pos_w.index_select(
                    1, self._wheel_body_ids
                )[..., 2]
                if (
                    self._clearance_drive_axle is None
                    or self._clearance_drive_axle.shape != wave_gate.shape
                ):
                    self._clearance_drive_axle = torch.full(
                        wave_gate.shape, -1, dtype=torch.long, device=actions.device
                    )
                    self._clearance_drive_released = torch.zeros_like(wave_gate)
                    self._clearance_phase_elapsed_s = torch.zeros(
                        wave_gate.shape, dtype=actions.dtype, device=actions.device
                    )
                axle_switch_x = float(
                    getattr(self.cfg, "wave_axle_switch_obstacle_x", 0.05)
                )
                candidate_axle = torch.where(
                    wave_gate,
                    torch.where(
                        spatial_obstacle_x >= axle_switch_x,
                        torch.zeros_like(self._clearance_drive_axle),
                        torch.ones_like(self._clearance_drive_axle),
                    ),
                    torch.full_like(self._clearance_drive_axle, -1),
                )
                phase_changed = candidate_axle != self._clearance_drive_axle
                self._clearance_phase_elapsed_s = torch.where(
                    phase_changed,
                    torch.zeros_like(self._clearance_phase_elapsed_s),
                    self._clearance_phase_elapsed_s + step_dt,
                )
                clearance_armed = self._clearance_phase_elapsed_s >= float(
                    getattr(self.cfg, "wave_clearance_minimum_hold_s", 0.15)
                )
                armed_wheel_heights = torch.where(
                    clearance_armed.unsqueeze(-1),
                    wheel_heights,
                    torch.full_like(wheel_heights, -torch.inf),
                )
                (
                    self._clearance_drive_axle,
                    self._clearance_drive_released,
                    drive_allowed,
                ) = update_clearance_drive_release(
                    obstacle_x=spatial_obstacle_x,
                    wave_gate=wave_gate,
                    wheel_heights=armed_wheel_heights,
                    previous_axle=self._clearance_drive_axle,
                    previous_released=self._clearance_drive_released,
                    required_height=float(
                        getattr(self.cfg, "wave_axle_clearance_height_m", 0.16)
                    ),
                    axle_switch_x=axle_switch_x,
                )
                wheel_targets = torch.where(
                    drive_allowed.unsqueeze(-1), wheel_targets, torch.zeros_like(wheel_targets)
                )
                self.env.unwrapped.m1_wave_drive_allowed = drive_allowed.detach()
                self.env.unwrapped.m1_wave_drive_axle = self._clearance_drive_axle.detach()
                self.env.unwrapped.m1_wave_clearance_armed = clearance_armed.detach()
            if bool(getattr(self.cfg, "wave_sync_actual_wheel_velocity", False)):
                robot = self.env.scene["robot"]
                if self._wheel_joint_ids is None:
                    wheel_joint_ids, _ = robot.find_joints(".*_FOOT_JOINT", preserve_order=True)
                    self._wheel_joint_ids = torch.as_tensor(
                        wheel_joint_ids, dtype=torch.long, device=actions.device
                    )
                actual_wheel_velocity = robot.data.joint_vel.index_select(1, self._wheel_joint_ids)
                forward_wheel_velocity = _wheel_forward_velocity(
                    actual_wheel_velocity,
                    getattr(self.cfg, "wave_wheel_action_signs", None),
                )
                wheel_speed_error = wheel_targets - forward_wheel_velocity
                if self._wheel_sync_integral is None:
                    self._wheel_sync_integral = torch.zeros_like(wheel_speed_error)
                step_dt = float(self.cfg.sim.dt * self.cfg.decimation)
                integral_limit = float(
                    getattr(self.cfg, "wave_wheel_sync_integral_limit", 0.50)
                )
                self._wheel_sync_integral = _update_wheel_sync_integral(
                    previous=self._wheel_sync_integral,
                    error=wheel_speed_error,
                    step_dt=step_dt,
                    limit=integral_limit,
                )
                correction = (
                    float(getattr(self.cfg, "wave_wheel_sync_gain", 0.50)) * wheel_speed_error
                    + float(getattr(self.cfg, "wave_wheel_sync_integral_gain", 1.0))
                    * self._wheel_sync_integral
                )
                correction_limit = float(
                    getattr(self.cfg, "wave_wheel_sync_max_correction", 0.20)
                )
                if wave_gate is not None:
                    obstacle_limit = float(
                        getattr(self.cfg, "wave_obstacle_sync_max_correction", correction_limit)
                    )
                    correction_limit = torch.where(
                        wave_gate.unsqueeze(-1),
                        torch.full_like(wheel_speed_error, obstacle_limit),
                        torch.full_like(wheel_speed_error, correction_limit),
                    )
                wheel_targets = wheel_targets + torch.clamp(
                    correction, -correction_limit, correction_limit
                )
                equalize_gain = float(getattr(self.cfg, "wave_wheel_equalize_gain", 0.0))
                if equalize_gain > 0.0:
                    if bool(getattr(self.cfg, "wave_wheel_equalize_to_slowest", False)):
                        velocity_reference = _slowest_forward_wheel_velocity(
                            forward_wheel_velocity
                        )
                    else:
                        velocity_reference = forward_wheel_velocity.mean(dim=1, keepdim=True)
                    equalize_correction = equalize_gain * (
                        velocity_reference - forward_wheel_velocity
                    )
                    equalize_limit = float(
                        getattr(self.cfg, "wave_wheel_equalize_max_correction", 0.50)
                    )
                    wheel_targets = wheel_targets + torch.clamp(
                        equalize_correction, -equalize_limit, equalize_limit
                    )
            rear_velocity_feedforward = float(
                getattr(self.cfg, "wave_rear_wheel_velocity_feedforward", 0.0)
            )
            if rear_velocity_feedforward != 0.0:
                wheel_targets = wheel_targets.clone()
                wheel_targets[:, 2:4] += rear_velocity_feedforward
            lateral_steering_gain = float(
                getattr(self.cfg, "wave_lateral_steering_gain", 0.0)
            )
            yaw_damping_gain = float(
                getattr(self.cfg, "wave_yaw_damping_gain", 0.0)
            )
            if lateral_steering_gain > 0.0 or yaw_damping_gain > 0.0:
                robot = self.env.scene["robot"]
                root_local_y = (
                    robot.data.root_pos_w[:, 1] - self.env.scene.env_origins[:, 1]
                )
                wheel_targets = wheel_targets + build_lateral_steering_correction(
                    lateral_y=root_local_y,
                    yaw_rate=robot.data.root_ang_vel_b[:, 2],
                    lateral_gain=lateral_steering_gain,
                    yaw_damping_gain=yaw_damping_gain,
                    max_correction=float(
                        getattr(self.cfg, "wave_steering_max_correction", 0.5)
                    ),
                )
            if bool(getattr(self.cfg, "wave_lock_left_right_wheel_targets", False)):
                front_pair = wheel_targets[:, :2].mean(dim=1, keepdim=True)
                rear_pair = wheel_targets[:, 2:4].mean(dim=1, keepdim=True)
                wheel_targets = torch.cat(
                    (front_pair.expand(-1, 2), rear_pair.expand(-1, 2)), dim=1
                )
            if bool(getattr(self.cfg, "wave_lock_all_wheel_targets", False)):
                wheel_target = wheel_targets.mean(dim=1, keepdim=True)
                wheel_targets = wheel_target.expand(-1, 4)
            sequential_drive_allowed = getattr(
                self, "_sequential_drive_allowed", None
            )
            if sequential_drive_allowed is not None:
                wheel_targets = torch.where(
                    sequential_drive_allowed.unsqueeze(-1),
                    wheel_targets,
                    torch.zeros_like(wheel_targets),
                )
            if bool(getattr(self.cfg, "wave_forward_only_wheels", False)):
                wheel_targets = torch.clamp(wheel_targets, min=0.0)
            wheel_signs = getattr(self.cfg, "wave_wheel_action_signs", None)
            if wheel_signs is not None:
                signs = torch.tensor(
                    wheel_signs, dtype=wheel_targets.dtype, device=wheel_targets.device
                ).reshape(1, 4)
                wheel_targets = wheel_targets * signs
            prepared[:, 12:16] = wheel_targets
            return prepared

        return actions

    def step(self, actions):
        actions = self._prepare_actions(actions)
        obs_dict, rewards, dones, truncated, extras = self.env.step(actions)
        dones = dones | truncated
        if self._wheel_sync_integral is not None and bool(dones.any()):
            self._wheel_sync_integral[dones] = 0.0
        if self._wave_elapsed_s is not None and bool(dones.any()):
            self._wave_elapsed_s[dones] = -1.0
            self._wave_obstacle_active[dones] = False
        if self._clearance_drive_axle is not None and bool(dones.any()):
            self._clearance_drive_axle[dones] = -1
            self._clearance_drive_released[dones] = False
            self._clearance_phase_elapsed_s[dones] = 0.0
        if self._sequential_crossing_phase is not None and bool(dones.any()):
            self._sequential_crossing_phase[dones] = -1
            self._sequential_crossing_phase_steps[dones] = 0
            self._sequential_drive_allowed[dones] = True
        extras["time_outs"] = truncated
        obs, obs_extras = self._format_observations(obs_dict)
        extras.setdefault("observations", {}).update(obs_extras["observations"])
        return obs, rewards, dones, extras
