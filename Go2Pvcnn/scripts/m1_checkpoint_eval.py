#!/usr/bin/env python3
"""Deterministically evaluate and optionally promote an M1 roll checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
PVCNN_ROOT = GO2PVCNN_ROOT.parent / "pvcnn"
for path in (GO2PVCNN_ROOT, GO2PVCNN_ROOT / "rsl_rl", PVCNN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from go2_pvcnn.pvcnn_runtime import configure_pvcnn_cuda

configure_pvcnn_cuda(GO2PVCNN_ROOT.parent)

import torch


def _parse_wave_window(value: str) -> tuple[float, float, float, float]:
    values = tuple(float(part.strip()) for part in value.split(","))
    if len(values) != 4:
        raise ValueError("wave windows require start,full_start,full_end,end")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate an M1 roll checkpoint against the Stage 1 gate.")
    parser.add_argument("--task", default="Isaac-M1-Roll-v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--perception-checkpoint", type=Path, default=None)
    parser.add_argument("--num_envs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--clip-actions", dest="clip_actions", type=float, default=1.0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--promote", type=Path, default=None)
    parser.add_argument("--obstacle-threshold", type=float, default=None)
    parser.add_argument("--lock-legs", action="store_true")
    parser.add_argument("--wheel-sync-gain", type=float, default=None)
    parser.add_argument("--wheel-sync-max-correction", type=float, default=None)
    parser.add_argument("--wheel-action", type=float, default=None)
    parser.add_argument("--semantic-crossing", action="store_true")
    parser.add_argument("--disable-crossing-reset", action="store_true")
    parser.add_argument("--enable-wave-reference-actions", action="store_true")
    parser.add_argument("--spatial-wave-reference", action="store_true")
    parser.add_argument("--disable-left-right-symmetry", action="store_true")
    parser.add_argument("--front-wave-window", type=str, default=None)
    parser.add_argument("--rear-wave-window", type=str, default=None)
    parser.add_argument("--temporal-wave-reference", action="store_true")
    parser.add_argument("--dynamic-wave-phase", action="store_true")
    parser.add_argument("--wave-reference-frequency", type=float, default=None)
    parser.add_argument("--wave-pulse-ramp", type=float, default=None)
    parser.add_argument("--wave-pulse-hold", type=float, default=None)
    parser.add_argument("--wave-cycle-duration", type=float, default=None)
    parser.add_argument("--wave-balance-steps", type=int, default=None)
    parser.add_argument("--wave-support-extension", type=float, default=None)
    parser.add_argument("--wave-opposite-abduction", type=float, default=None)
    parser.add_argument("--wave-front-start-x", type=float, default=None)
    parser.add_argument("--wave-front-hip-action", type=float, default=None)
    parser.add_argument("--wave-front-knee-action", type=float, default=None)
    parser.add_argument("--wave-rear-hip-action", type=float, default=None)
    parser.add_argument("--wave-rear-knee-action", type=float, default=None)
    parser.add_argument("--min-crossing-rate", type=float, default=0.50)
    parser.add_argument("--wave-reference-amplitude", type=float, default=None)
    parser.add_argument("--wave-reference-knee-ratio", type=float, default=None)
    parser.add_argument("--front-support-ratio", type=float, default=None)
    parser.add_argument("--rear-support-ratio", type=float, default=None)
    parser.add_argument("--rear-amplitude-scale", type=float, default=None)
    parser.add_argument("--obstacle-wheel-action", type=float, default=None)
    parser.add_argument("--obstacle-front-wheel-action", type=float, default=None)
    parser.add_argument("--obstacle-rear-wheel-action", type=float, default=None)
    parser.add_argument("--wheel-equalize-gain", type=float, default=None)
    parser.add_argument("--wheel-equalize-max-correction", type=float, default=None)
    parser.add_argument("--phase-wheel-assist", type=float, default=None)
    parser.add_argument("--disable-semantic-gating", action="store_true")
    parser.add_argument("--policy-gate-threshold", type=float, default=None)
    parser.add_argument("--fixed-leg-actions", type=str, default=None)
    parser.add_argument("--front-pair-lift-swing-probe", action="store_true")
    parser.add_argument("--single-wheel-lift-swing-probe", type=int, default=None)
    parser.add_argument("--sequential-front-wheel-probe", action="store_true")
    parser.add_argument("--probe-support-extension", type=str, default=None)
    parser.add_argument("--probe-opposite-abduction", type=str, default=None)
    parser.add_argument("--probe-lift-scale", type=str, default=None)
    parser.add_argument("--leg-action-scale", type=float, default=None)
    parser.add_argument("--preserve-leg-action-order", action="store_true")
    parser.add_argument("--wave-leg-action-limit", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _termination_term(env, name: str) -> torch.Tensor:
    manager = env.unwrapped.termination_manager
    try:
        value = manager.get_term(name)
    except Exception:
        value = None
    if value is None:
        return torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    return torch.as_tensor(value, dtype=torch.bool, device=env.unwrapped.device).reshape(-1)


def _configured_orientation_limits(env_cfg) -> tuple[float, float]:
    params = env_cfg.terminations.bad_orientation.params
    normal = float(params.get("normal_limit_angle", params.get("limit_angle", 0.45)))
    wave = float(params.get("wave_limit_angle", normal))
    return normal, wave


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    exit_code = 1
    try:
        import gymnasium as gym

        from agent import get_m1_train_cfg
        import go2_pvcnn.tasks  # noqa: F401
        from go2_pvcnn.tasks.m1_curriculum import (
            evaluate_obstacle_gate,
            evaluate_roll_gate,
            required_axle_lift_passed,
            update_semantic_crossing_tracker,
            update_wheel_obstacle_clearance,
        )
        from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper
        from go2_pvcnn.tasks.m1_pvcnn_perception import M1PvcnnRslRlEnvWrapper
        from extension.convention import extract_roll_pitch_batch
        from rsl_rl.runners import OnPolicyRunner

        env_cfg_entry = gym.spec(args.task).kwargs["env_cfg_entry_point"]
        env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device
        if args.enable_wave_reference_actions:
            env_cfg.wave_reference_actions = True
        if args.spatial_wave_reference:
            env_cfg.wave_spatial_reference = True
        if args.disable_left_right_symmetry:
            env_cfg.wave_left_right_symmetric = False
        if args.front_wave_window is not None:
            env_cfg.wave_front_lift_window = _parse_wave_window(args.front_wave_window)
        if args.rear_wave_window is not None:
            env_cfg.wave_rear_lift_window = _parse_wave_window(args.rear_wave_window)
        if args.temporal_wave_reference:
            env_cfg.wave_spatial_reference = False
        if args.dynamic_wave_phase:
            env_cfg.wave_reference_constant_phase_s = None
        if args.wave_reference_frequency is not None:
            env_cfg.wave_reference_frequency = args.wave_reference_frequency
        if args.wave_pulse_ramp is not None:
            env_cfg.wave_reference_pulse_ramp_s = args.wave_pulse_ramp
        if args.wave_pulse_hold is not None:
            env_cfg.wave_reference_pulse_hold_s = args.wave_pulse_hold
        if args.wave_cycle_duration is not None:
            env_cfg.wave_single_cycle_duration_s = args.wave_cycle_duration
            env_cfg.wave_minimum_gate_duration_s = args.wave_cycle_duration
        if args.wave_balance_steps is not None:
            env_cfg.wave_sequential_balance_steps = args.wave_balance_steps
        if args.wave_support_extension is not None:
            env_cfg.wave_sequential_support_extension = args.wave_support_extension
        if args.wave_opposite_abduction is not None:
            env_cfg.wave_sequential_opposite_abduction = args.wave_opposite_abduction
        if args.wave_front_start_x is not None:
            env_cfg.wave_sequential_front_start_x_m = args.wave_front_start_x
        if args.wave_front_hip_action is not None:
            env_cfg.wave_sequential_front_hip_action = args.wave_front_hip_action
        if args.wave_front_knee_action is not None:
            env_cfg.wave_sequential_front_knee_action = args.wave_front_knee_action
        if args.wave_rear_hip_action is not None:
            env_cfg.wave_sequential_rear_hip_action = args.wave_rear_hip_action
        if args.wave_rear_knee_action is not None:
            env_cfg.wave_sequential_rear_knee_action = args.wave_rear_knee_action
        if args.lock_legs:
            env_cfg.wave_leg_action_limit = 0.0
            env_cfg.wave_reference_actions = False
        if args.wheel_sync_gain is not None:
            env_cfg.wave_wheel_sync_gain = args.wheel_sync_gain
        if args.wheel_sync_max_correction is not None:
            env_cfg.wave_wheel_sync_max_correction = args.wheel_sync_max_correction
        if args.wheel_action is not None:
            env_cfg.wave_front_wheel_action = args.wheel_action
            env_cfg.wave_rear_wheel_action = args.wheel_action
        if args.wave_reference_amplitude is not None:
            env_cfg.wave_reference_raw_amplitude = args.wave_reference_amplitude
        if args.wave_reference_knee_ratio is not None:
            env_cfg.wave_reference_knee_ratio = args.wave_reference_knee_ratio
        if args.front_support_ratio is not None:
            env_cfg.wave_front_support_ratio = args.front_support_ratio
        if args.rear_support_ratio is not None:
            env_cfg.wave_rear_support_ratio = args.rear_support_ratio
        if args.rear_amplitude_scale is not None:
            env_cfg.wave_rear_amplitude_scale = args.rear_amplitude_scale
        if args.obstacle_wheel_action is not None:
            env_cfg.wave_obstacle_wheel_action = args.obstacle_wheel_action
        if args.obstacle_front_wheel_action is not None:
            env_cfg.wave_obstacle_front_wheel_action = args.obstacle_front_wheel_action
        if args.obstacle_rear_wheel_action is not None:
            env_cfg.wave_obstacle_rear_wheel_action = args.obstacle_rear_wheel_action
        if args.wheel_equalize_gain is not None:
            env_cfg.wave_wheel_equalize_gain = args.wheel_equalize_gain
        if args.wheel_equalize_max_correction is not None:
            env_cfg.wave_wheel_equalize_max_correction = args.wheel_equalize_max_correction
        if args.phase_wheel_assist is not None:
            env_cfg.wave_phase_wheel_assist = args.phase_wheel_assist
        if args.disable_semantic_gating:
            env_cfg.wave_semantic_obstacle_gating = False
        if args.policy_gate_threshold is not None:
            env_cfg.wave_policy_gate_threshold = args.policy_gate_threshold
        if args.leg_action_scale is not None:
            env_cfg.actions.leg_pos.scale = args.leg_action_scale
        if args.preserve_leg_action_order:
            env_cfg.actions.leg_pos.preserve_order = True
        if args.wave_leg_action_limit is not None:
            env_cfg.wave_leg_action_limit = args.wave_leg_action_limit
        if args.disable_crossing_reset and hasattr(env_cfg.terminations, "crossing_success"):
            env_cfg.terminations.crossing_success = None

        env = gym.make(args.task, cfg=env_cfg)
        pvcnn_model = None
        if args.perception_checkpoint is not None:
            from models.s3dis.pvcnn import PVCNN

            perception = torch.load(args.perception_checkpoint, map_location="cpu", weights_only=False)
            pvcnn_model = PVCNN(
                num_classes=3,
                extra_feature_channels=0,
                width_multiplier=float(perception["width_multiplier"]),
            ).to(env_cfg.sim.device)
            pvcnn_model.load_state_dict(perception["pvcnn_state_dict"])
            wrapped_env = M1PvcnnRslRlEnvWrapper(
                env.unwrapped, pvcnn_model, clip_actions=args.clip_actions
            )
        else:
            wrapped_env = M1RslRlEnvWrapper(env.unwrapped, clip_actions=args.clip_actions)
        runner = OnPolicyRunner(wrapped_env, get_m1_train_cfg(), log_dir=None, device=env_cfg.sim.device)
        if pvcnn_model is not None:
            runner.alg.pvcnn_model = pvcnn_model
        runner.load(args.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env_cfg.sim.device)
        obs, _ = wrapped_env.get_observations()

        robot = env.unwrapped.scene["robot"]
        leg_action_term = env.unwrapped.action_manager.get_term("leg_pos")
        leg_action_joint_order = list(getattr(leg_action_term, "_joint_names", ()))
        leg_joint_ids = getattr(leg_action_term, "_joint_ids", slice(0, 12))
        contact_sensor = env.unwrapped.scene["contact_forces"]
        wheel_joint_ids, _ = robot.find_joints(".*_FOOT_JOINT", preserve_order=True)
        wheel_joint_order = [robot.joint_names[index] for index in wheel_joint_ids]
        wheel_direction_signs = torch.as_tensor(
            getattr(env_cfg, "wave_wheel_action_signs", None) or (1.0, 1.0, 1.0, 1.0),
            dtype=robot.data.joint_vel.dtype,
            device=env.unwrapped.device,
        ).reshape(1, 4)
        wheel_body_ids, wheel_body_names = robot.find_bodies(".*_FOOT_LINK", preserve_order=True)
        contact_body_ids, _ = contact_sensor.find_bodies(wheel_body_names, preserve_order=True)
        root_start = robot.data.root_pos_w.clone()
        root_at_done = root_start.clone()
        done_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        success_seen = torch.zeros_like(done_seen)
        timeout_seen = torch.zeros_like(done_seen)
        bad_orientation_seen = torch.zeros_like(done_seen)
        max_tilt = torch.zeros(args.num_envs, device=env.unwrapped.device)
        max_dx = torch.zeros(args.num_envs, device=env.unwrapped.device)
        leg_action_limit_cfg = getattr(env_cfg, "wave_leg_action_limit", 1.0)
        if leg_action_limit_cfg is None:
            leg_action_limit = float("inf")
        else:
            leg_action_limit = float(leg_action_limit_cfg)
        leg_abs_sum = 0.0
        leg_action_count = 0
        max_abs_leg_action = 0.0
        saturated_leg_actions = 0
        leg_action_sum_by_index = torch.zeros(12, device=env.unwrapped.device)
        leg_action_rows = 0
        wave_prepared_leg_abs_sum = 0.0
        wave_prepared_leg_count = 0
        wave_prepared_leg_max_abs = 0.0
        wave_prepared_leg_max_abs_by_index = torch.zeros(
            12, device=env.unwrapped.device
        )
        wave_prepared_leg_sum_by_index = torch.zeros(
            12, device=env.unwrapped.device
        )
        wave_prepared_leg_rows = 0
        wave_first_prepared_leg_action = None
        wave_max_prepared_leg_action_delta = 0.0
        previous_prepared_leg_actions = None
        nonwave_prepared_leg_count = 0
        nonwave_prepared_leg_max_abs = 0.0
        wave_joint_posture_error_max = 0.0
        nonwave_joint_posture_error_max = 0.0
        fixed_leg_actions = None
        if args.fixed_leg_actions is not None:
            fixed_rows = [
                [float(value.strip()) for value in row.split(",")]
                for row in args.fixed_leg_actions.split(";")
            ]
            if any(len(row) != 12 for row in fixed_rows):
                raise ValueError("--fixed-leg-actions requires 12 values per semicolon-separated row")
            if len(fixed_rows) not in (1, args.num_envs):
                raise ValueError("--fixed-leg-actions requires one row or one row per environment")
            fixed_leg_actions = torch.tensor(
                fixed_rows, dtype=torch.float32, device=env.unwrapped.device
            )
            if fixed_leg_actions.shape[0] == 1:
                fixed_leg_actions = fixed_leg_actions.expand(args.num_envs, -1)
        probe_support_extension = None
        if args.probe_support_extension is not None:
            values = [
                float(value.strip())
                for value in args.probe_support_extension.split(",")
                if value.strip()
            ]
            if len(values) not in (1, args.num_envs):
                raise ValueError(
                    "--probe-support-extension requires one value or one per environment"
                )
            probe_support_extension = torch.tensor(
                values, dtype=torch.float32, device=env.unwrapped.device
            )
            if probe_support_extension.numel() == 1:
                probe_support_extension = probe_support_extension.expand(args.num_envs)
        probe_opposite_abduction = None
        if args.probe_opposite_abduction is not None:
            values = [
                float(value.strip())
                for value in args.probe_opposite_abduction.split(",")
                if value.strip()
            ]
            if len(values) not in (1, args.num_envs):
                raise ValueError(
                    "--probe-opposite-abduction requires one value or one per environment"
                )
            probe_opposite_abduction = torch.tensor(
                values, dtype=torch.float32, device=env.unwrapped.device
            )
            if probe_opposite_abduction.numel() == 1:
                probe_opposite_abduction = probe_opposite_abduction.expand(args.num_envs)
        probe_lift_scale = torch.ones(args.num_envs, device=env.unwrapped.device)
        if args.probe_lift_scale is not None:
            values = [
                float(value.strip())
                for value in args.probe_lift_scale.split(",")
                if value.strip()
            ]
            if len(values) not in (1, args.num_envs):
                raise ValueError(
                    "--probe-lift-scale requires one value or one per environment"
                )
            probe_lift_scale = torch.tensor(
                values, dtype=torch.float32, device=env.unwrapped.device
            )
            if probe_lift_scale.numel() == 1:
                probe_lift_scale = probe_lift_scale.expand(args.num_envs)
        wheel_target_sum = torch.zeros(4, device=env.unwrapped.device)
        wheel_velocity_sum = torch.zeros(4, device=env.unwrapped.device)
        wheel_torque_abs_sum = torch.zeros(4, device=env.unwrapped.device)
        wheel_torque_abs_max = torch.zeros(4, device=env.unwrapped.device)
        wheel_height_sum = torch.zeros(4, device=env.unwrapped.device)
        wheel_height_max = torch.full((4,), -torch.inf, device=env.unwrapped.device)
        wheel_height_max_by_env = torch.full(
            (args.num_envs, 4), -torch.inf, device=env.unwrapped.device
        )
        wheel_relative_x_min = torch.full_like(wheel_height_max_by_env, torch.inf)
        wheel_relative_x_max = torch.full_like(wheel_height_max_by_env, -torch.inf)
        wheel_relative_z_max = torch.full_like(wheel_height_max_by_env, -torch.inf)
        wheel_contact_sum = torch.zeros(4, device=env.unwrapped.device)
        wheel_sample_rows = 0
        wheel_prelift_seen = torch.zeros(
            (args.num_envs, 4), dtype=torch.bool, device=env.unwrapped.device
        )
        wheel_clearance_required = torch.zeros_like(wheel_prelift_seen)
        first_wheel_prelift_root_x = torch.full(
            (args.num_envs, 4), torch.nan, device=env.unwrapped.device
        )
        wheel_overbar_clear_seen = torch.zeros_like(wheel_prelift_seen)
        obstacle_center_xy = getattr(
            env_cfg.scene.terrain,
            "semantic_course_mandatory_small_xy",
            (0.65, 0.0),
        )
        obstacle_center_x = float(obstacle_center_xy[0])
        obstacle_center_y = float(obstacle_center_xy[1])
        obstacle_size = getattr(
            env_cfg.scene.terrain, "semantic_course_cuboid_size_overrides", {}
        ).get("small", (0.06, 0.60, 0.06))
        obstacle_size_x = float(obstacle_size[0])
        obstacle_size_y = float(obstacle_size[1])
        obstacle_height = float(obstacle_size[2])
        wheel_radius = float(getattr(env_cfg, "acceptance_wheel_radius_m", 0.095))
        clearance_margin = float(
            getattr(env_cfg, "acceptance_wheel_clearance_margin_m", 0.005)
        )
        clearance_contact_force_limit = float(
            getattr(env_cfg, "acceptance_clearance_contact_force_limit_n", 1.0)
        )
        required_wheel_clearance_height = 0.0
        max_precontact_wheel_height = torch.full(
            (args.num_envs, 4), -torch.inf, device=env.unwrapped.device
        )
        max_overbar_wheel_height = torch.full_like(
            max_precontact_wheel_height, -torch.inf
        )
        min_overbar_wheel_contact_force = torch.full_like(
            max_precontact_wheel_height, torch.inf
        )
        axle_overbar_action_seen = torch.zeros(
            (args.num_envs, 2), dtype=torch.bool, device=env.unwrapped.device
        )
        prepared_leg_action_at_axle_overbar = torch.zeros(
            (args.num_envs, 2, 12), device=env.unwrapped.device
        )
        obstacle_x_at_axle_overbar = torch.full(
            (args.num_envs, 2), torch.nan, device=env.unwrapped.device
        )
        crossbar_collision_seen = torch.zeros(
            args.num_envs, dtype=torch.bool, device=env.unwrapped.device
        )
        crossbar_collision_mask = torch.zeros(
            (args.num_envs, 4), dtype=torch.bool, device=env.unwrapped.device
        )
        crossbar_collision_wheel_pos = torch.full(
            (args.num_envs, 4, 3), torch.nan, device=env.unwrapped.device
        )
        crossbar_collision_wheel_force = torch.full(
            (args.num_envs, 4), torch.nan, device=env.unwrapped.device
        )
        crossbar_collision_phase = torch.full(
            (args.num_envs,), -99, dtype=torch.long, device=env.unwrapped.device
        )
        crossbar_collision_phase_steps = torch.full_like(
            crossbar_collision_phase, -1
        )
        crossbar_collision_wheel_x_from_obstacle = torch.full(
            (args.num_envs, 4), torch.nan, device=env.unwrapped.device
        )
        pitch_sum = torch.zeros(args.num_envs, device=env.unwrapped.device)
        base_height_sum = torch.zeros(args.num_envs, device=env.unwrapped.device)
        base_height_samples = torch.zeros(args.num_envs, device=env.unwrapped.device)
        max_abs_base_height_error = torch.zeros(args.num_envs, device=env.unwrapped.device)
        final_base_height = torch.zeros(args.num_envs, device=env.unwrapped.device)
        front_rear_height_delta_sum = torch.zeros(args.num_envs, device=env.unwrapped.device)
        max_abs_pitch = torch.zeros(args.num_envs, device=env.unwrapped.device)
        final_pitch = torch.zeros(args.num_envs, device=env.unwrapped.device)
        final_front_rear_height_delta = torch.zeros(args.num_envs, device=env.unwrapped.device)
        crossing_candidate_xy = torch.zeros((args.num_envs, 2), device=env.unwrapped.device)
        crossing_candidate_heading = torch.zeros_like(crossing_candidate_xy)
        crossing_candidate_valid = torch.zeros(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        semantic_crossed = torch.zeros_like(crossing_candidate_valid)
        front_pair_lift_swing_phase = torch.zeros(
            args.num_envs, dtype=torch.long, device=env.unwrapped.device
        )
        front_pair_probe_phase_steps = torch.zeros_like(front_pair_lift_swing_phase)
        max_sequential_crossing_phase = torch.full_like(
            front_pair_lift_swing_phase, -1
        )
        phase_entry_seen = torch.zeros(
            (args.num_envs, 6), dtype=torch.bool, device=env.unwrapped.device
        )
        phase_entry_root_pos = torch.full(
            (args.num_envs, 6, 3), torch.nan, device=env.unwrapped.device
        )
        phase_entry_wheel_pos = torch.full(
            (args.num_envs, 6, 4, 3), torch.nan, device=env.unwrapped.device
        )
        phase_entry_wheel_velocity = torch.full(
            (args.num_envs, 6, 4), torch.nan, device=env.unwrapped.device
        )
        phase_entry_leg_joint_pos = torch.full(
            (args.num_envs, 6, 12), torch.nan, device=env.unwrapped.device
        )
        phase_entry_leg_joint_target = torch.full_like(
            phase_entry_leg_joint_pos, torch.nan
        )
        first_sequential_phase_root_x = torch.full(
            (args.num_envs,), torch.nan, device=env.unwrapped.device
        )
        first_sequential_phase_wheel_x = torch.full(
            (args.num_envs, 4), torch.nan, device=env.unwrapped.device
        )
        policy_gate_score_min = torch.full(
            (args.num_envs,), torch.inf, device=env.unwrapped.device
        )
        policy_gate_score_max = torch.full_like(policy_gate_score_min, -torch.inf)
        policy_gate_positive_score_min = torch.full_like(policy_gate_score_min, torch.inf)
        policy_gate_positive_score_max = torch.full_like(policy_gate_score_min, -torch.inf)
        policy_gate_true_positive = torch.zeros_like(policy_gate_score_min)
        policy_gate_positive_samples = torch.zeros_like(policy_gate_score_min)
        policy_gate_true_negative = torch.zeros_like(policy_gate_score_min)
        policy_gate_negative_samples = torch.zeros_like(policy_gate_score_min)

        for _ in range(args.steps):
            evaluation_active = ~(done_seen | success_seen)
            root_before_step = robot.data.root_pos_w.clone()
            with torch.inference_mode():
                actions = policy(obs)
                if args.sequential_front_wheel_probe:
                    wheel_pos_local_now = (
                        robot.data.body_pos_w[:, wheel_body_ids]
                        - env.unwrapped.scene.env_origins.unsqueeze(1)
                    )
                    phase = front_pair_lift_swing_phase
                    phase_steps = front_pair_probe_phase_steps
                    first_clear = wheel_pos_local_now[:, 0, 2] >= 0.16
                    first_past = (
                        (wheel_pos_local_now[:, 0, 0] >= 0.775)
                        & first_clear
                    )
                    second_clear = wheel_pos_local_now[:, 1, 2] >= 0.16
                    second_past = (
                        (wheel_pos_local_now[:, 1, 0] >= 0.775)
                        & second_clear
                    )
                    advance = (
                        ((phase == 0) & (phase_steps >= 5) & first_clear)
                        | ((phase == 1) & first_past)
                        | ((phase == 2) & (phase_steps >= 5) & second_clear)
                        | ((phase == 3) & second_past)
                    )
                    phase = torch.where(advance, phase + 1, phase)
                    phase_steps = torch.where(
                        advance, torch.zeros_like(phase_steps), phase_steps
                    )
                    front_pair_lift_swing_phase = phase
                    front_pair_probe_phase_steps = phase_steps
                    blend = torch.clamp(
                        phase_steps.to(actions.dtype) / 50.0, 0.0, 1.0
                    )
                    actions[:, :12] = 0.0
                    for wheel_index, lift_phase, swing_phase in ((0, 0, 1), (1, 2, 3)):
                        hip_index = wheel_index * 3 + 1
                        knee_index = wheel_index * 3 + 2
                        hold = phase > swing_phase
                        lifting = phase == lift_phase
                        swinging = phase == swing_phase
                        actions[hold, hip_index] = -2.0
                        actions[hold, knee_index] = -1.0
                        actions[lifting, hip_index] = 1.0
                        actions[lifting, knee_index] = -2.0
                        actions[swinging, hip_index] = 1.0 - 3.0 * blend[swinging]
                        actions[swinging, knee_index] = -2.0 + blend[swinging]
                    front_pair_probe_phase_steps += 1
                elif (
                    args.front_pair_lift_swing_probe
                    or args.single_wheel_lift_swing_probe is not None
                ):
                    if args.single_wheel_lift_swing_probe is not None:
                        probe_wheel = int(args.single_wheel_lift_swing_probe)
                        if probe_wheel not in (0, 1, 2, 3):
                            raise ValueError(
                                "--single-wheel-lift-swing-probe requires wheel index 0 through 3"
                            )
                        front_clear = (
                            robot.data.body_pos_w[:, wheel_body_ids[probe_wheel], 2]
                            >= 0.16
                        )
                        leg_pairs = ((probe_wheel * 3 + 1, probe_wheel * 3 + 2),)
                        probe_is_front = probe_wheel < 2
                    else:
                        front_clear = (
                            robot.data.body_pos_w[:, wheel_body_ids[:2], 2] >= 0.16
                        ).all(dim=1)
                        leg_pairs = ((1, 2), (4, 5))
                        probe_is_front = True
                    start_swing = (
                        (front_pair_lift_swing_phase == 0)
                        & (front_pair_probe_phase_steps >= 5)
                        & front_clear
                    )
                    front_pair_lift_swing_phase[start_swing] = 1
                    front_pair_probe_phase_steps = torch.where(
                        start_swing,
                        torch.zeros_like(front_pair_probe_phase_steps),
                        front_pair_probe_phase_steps,
                    )
                    actions[:, :12] = 0.0
                    lifting = front_pair_lift_swing_phase == 0
                    swinging = front_pair_lift_swing_phase == 1
                    swing_blend = torch.clamp(
                        front_pair_probe_phase_steps.to(actions.dtype) / 50.0,
                        0.0,
                        1.0,
                    )
                    for hip_index, knee_index in leg_pairs:
                        if probe_is_front:
                            actions[lifting, hip_index] = probe_lift_scale[lifting]
                            actions[lifting, knee_index] = -2.0 * probe_lift_scale[lifting]
                            actions[swinging, hip_index] = (
                                probe_lift_scale[swinging]
                                + (-2.0 - probe_lift_scale[swinging])
                                * swing_blend[swinging]
                            )
                            actions[swinging, knee_index] = (
                                -2.0 * probe_lift_scale[swinging]
                                + (-1.0 + 2.0 * probe_lift_scale[swinging])
                                * swing_blend[swinging]
                            )
                        else:
                            actions[lifting, hip_index] = -1.0
                            actions[lifting, knee_index] = 2.0
                            actions[swinging, hip_index] = (
                                -1.0 - swing_blend[swinging]
                            )
                            actions[swinging, knee_index] = (
                                2.0 - 3.0 * swing_blend[swinging]
                            )
                    if (
                        args.single_wheel_lift_swing_probe is not None
                        and probe_support_extension is not None
                    ):
                        selected_wheel = int(args.single_wheel_lift_swing_probe)
                        for support_wheel in range(4):
                            if support_wheel == selected_wheel:
                                continue
                            support_knee = support_wheel * 3 + 2
                            direction = 1.0 if support_wheel < 2 else -1.0
                            actions[:, support_knee] += (
                                direction * probe_support_extension
                            )
                    if (
                        args.single_wheel_lift_swing_probe is not None
                        and probe_opposite_abduction is not None
                    ):
                        selected_wheel = int(args.single_wheel_lift_swing_probe)
                        opposite_side = (1, 3) if selected_wheel in (0, 2) else (0, 2)
                        for support_wheel in opposite_side:
                            actions[:, support_wheel * 3] += probe_opposite_abduction
                    front_pair_probe_phase_steps += 1
                if fixed_leg_actions is not None:
                    actions[:, :12] = fixed_leg_actions
                leg_abs = torch.abs(actions[:, :12])
                leg_abs_sum += float(leg_abs.sum().item())
                leg_action_count += int(leg_abs.numel())
                max_abs_leg_action = max(max_abs_leg_action, float(leg_abs.max().item()))
                saturated_leg_actions += int((leg_abs >= 0.99 * leg_action_limit).sum().item())
                leg_action_sum_by_index += actions[:, :12].sum(dim=0)
                leg_action_rows += int(actions.shape[0])
                obs, _, dones, extras = wrapped_env.step(actions)
            gate_target = getattr(env.unwrapped, "m1_wave_gate_target", None)
            if gate_target is not None:
                gate_target = torch.as_tensor(
                    gate_target, dtype=torch.bool, device=env.unwrapped.device
                )
                gate_index = int(getattr(env_cfg, "wave_policy_gate_action_index", 15))
                gate_score = actions[:, gate_index]
                gate_prediction = gate_score > float(
                    getattr(env_cfg, "wave_policy_gate_threshold", 0.0)
                )
                active_gate_score = torch.where(
                    evaluation_active, gate_score, torch.full_like(gate_score, torch.inf)
                )
                policy_gate_score_min = torch.minimum(
                    policy_gate_score_min, active_gate_score
                )
                policy_gate_score_max = torch.maximum(
                    policy_gate_score_max,
                    torch.where(
                        evaluation_active,
                        gate_score,
                        torch.full_like(gate_score, -torch.inf),
                    ),
                )
                positive = evaluation_active & gate_target
                negative = evaluation_active & ~gate_target
                policy_gate_positive_score_min = torch.minimum(
                    policy_gate_positive_score_min,
                    torch.where(positive, gate_score, torch.full_like(gate_score, torch.inf)),
                )
                policy_gate_positive_score_max = torch.maximum(
                    policy_gate_positive_score_max,
                    torch.where(positive, gate_score, torch.full_like(gate_score, -torch.inf)),
                )
                policy_gate_true_positive += (positive & gate_prediction).to(torch.float32)
                policy_gate_positive_samples += positive.to(torch.float32)
                policy_gate_true_negative += (negative & ~gate_prediction).to(torch.float32)
                policy_gate_negative_samples += negative.to(torch.float32)
            latest_collision_mask = getattr(
                env.unwrapped, "m1_crossbar_collision_mask", None
            )
            if latest_collision_mask is not None:
                latest_collision_mask = torch.as_tensor(
                    latest_collision_mask,
                    dtype=torch.bool,
                    device=env.unwrapped.device,
                )
                capture_collision = latest_collision_mask.any(dim=1) & ~crossbar_collision_seen
                if bool(capture_collision.any()):
                    latest_collision_pos = torch.as_tensor(
                        env.unwrapped.m1_crossbar_collision_wheel_pos_local,
                        device=env.unwrapped.device,
                    )
                    latest_collision_force = torch.as_tensor(
                        env.unwrapped.m1_crossbar_collision_wheel_force,
                        device=env.unwrapped.device,
                    )
                    crossbar_collision_mask[capture_collision] = latest_collision_mask[
                        capture_collision
                    ]
                    crossbar_collision_wheel_pos[capture_collision] = latest_collision_pos[
                        capture_collision
                    ]
                    crossbar_collision_wheel_force[capture_collision] = latest_collision_force[
                        capture_collision
                    ]
                    latest_phase = getattr(
                        env.unwrapped, "m1_sequential_crossing_phase", None
                    )
                    latest_phase_steps = getattr(
                        env.unwrapped, "m1_sequential_crossing_phase_steps", None
                    )
                    latest_wheel_x = getattr(
                        env.unwrapped, "m1_sequential_wheel_x_from_obstacle", None
                    )
                    if latest_phase is not None:
                        crossbar_collision_phase[capture_collision] = torch.as_tensor(
                            latest_phase,
                            dtype=torch.long,
                            device=env.unwrapped.device,
                        )[capture_collision]
                    if latest_phase_steps is not None:
                        crossbar_collision_phase_steps[capture_collision] = torch.as_tensor(
                            latest_phase_steps,
                            dtype=torch.long,
                            device=env.unwrapped.device,
                        )[capture_collision]
                    if latest_wheel_x is not None:
                        crossbar_collision_wheel_x_from_obstacle[capture_collision] = (
                            torch.as_tensor(
                                latest_wheel_x,
                                device=env.unwrapped.device,
                            )[capture_collision]
                        )
                    crossbar_collision_seen |= capture_collision
            sequential_phase_now = getattr(
                env.unwrapped, "m1_sequential_crossing_phase", None
            )
            if sequential_phase_now is not None:
                sequential_phase_now = torch.as_tensor(
                    sequential_phase_now,
                    dtype=torch.long,
                    device=env.unwrapped.device,
                )
                max_sequential_crossing_phase = torch.maximum(
                    max_sequential_crossing_phase,
                    torch.where(
                        evaluation_active,
                        sequential_phase_now,
                        max_sequential_crossing_phase,
                    ),
                )
                root_pos_local_now = (
                    robot.data.root_pos_w - env.unwrapped.scene.env_origins
                )
                wheel_pos_local_now = (
                    robot.data.body_pos_w[:, wheel_body_ids]
                    - env.unwrapped.scene.env_origins.unsqueeze(1)
                )
                for phase_index in range(6):
                    entering = (
                        (sequential_phase_now == phase_index)
                        & ~phase_entry_seen[:, phase_index]
                    )
                    if bool(entering.any()):
                        phase_entry_seen[entering, phase_index] = True
                        phase_entry_root_pos[entering, phase_index] = (
                            root_pos_local_now[entering]
                        )
                        phase_entry_wheel_pos[entering, phase_index] = (
                            wheel_pos_local_now[entering]
                        )
                        phase_entry_wheel_velocity[entering, phase_index] = (
                            robot.data.joint_vel[entering][:, wheel_joint_ids]
                            * wheel_direction_signs
                        )
                        phase_entry_leg_joint_pos[entering, phase_index] = (
                            robot.data.joint_pos[entering][:, leg_joint_ids]
                        )
                        phase_entry_leg_joint_target[entering, phase_index] = (
                            robot.data.joint_pos_target[entering][:, leg_joint_ids]
                        )
                first_phase = (
                    evaluation_active
                    & (sequential_phase_now >= 0)
                    & torch.isnan(first_sequential_phase_root_x)
                )
                if bool(first_phase.any()):
                    first_sequential_phase_root_x[first_phase] = (
                        robot.data.root_pos_w[first_phase, 0]
                        - env.unwrapped.scene.env_origins[first_phase, 0]
                    )
                    first_sequential_phase_wheel_x[first_phase] = (
                        robot.data.body_pos_w[first_phase][:, wheel_body_ids, 0]
                        - env.unwrapped.scene.env_origins[first_phase, None, 0]
                    )
            wave_gate = torch.as_tensor(
                getattr(env.unwrapped, "m1_wave_gate", torch.ones(args.num_envs)),
                dtype=torch.bool,
                device=env.unwrapped.device,
            ).reshape(-1)
            prepared_leg_actions = torch.as_tensor(
                getattr(env.unwrapped, "m1_prepared_leg_actions", actions[:, :12]),
                device=env.unwrapped.device,
            )
            prepared_leg_abs = torch.abs(prepared_leg_actions)
            if previous_prepared_leg_actions is not None and bool(wave_gate.any()):
                prepared_delta = torch.abs(
                    prepared_leg_actions - previous_prepared_leg_actions
                )
                wave_max_prepared_leg_action_delta = max(
                    wave_max_prepared_leg_action_delta,
                    float(prepared_delta[wave_gate].max().item()),
                )
            previous_prepared_leg_actions = prepared_leg_actions.detach().clone()
            joint_posture_error = torch.abs(
                robot.data.joint_pos[:, leg_joint_ids]
                - robot.data.default_joint_pos[:, leg_joint_ids]
            )
            if bool(wave_gate.any()):
                wave_prepared = prepared_leg_abs[wave_gate]
                wave_prepared_signed = prepared_leg_actions[wave_gate]
                if wave_first_prepared_leg_action is None:
                    wave_first_prepared_leg_action = wave_prepared_signed[0].detach().clone()
                wave_prepared_leg_sum_by_index += wave_prepared_signed.sum(dim=0)
                wave_prepared_leg_rows += int(wave_prepared_signed.shape[0])
                wave_prepared_leg_abs_sum += float(wave_prepared.sum().item())
                wave_prepared_leg_count += int(wave_prepared.numel())
                wave_prepared_leg_max_abs = max(
                    wave_prepared_leg_max_abs, float(wave_prepared.max().item())
                )
                wave_prepared_leg_max_abs_by_index = torch.maximum(
                    wave_prepared_leg_max_abs_by_index,
                    wave_prepared.amax(dim=0),
                )
                wave_joint_posture_error_max = max(
                    wave_joint_posture_error_max,
                    float(joint_posture_error[wave_gate].max().item()),
                )
            nonwave_gate = ~wave_gate
            if bool(nonwave_gate.any()):
                nonwave_prepared = prepared_leg_abs[nonwave_gate]
                nonwave_prepared_leg_count += int(nonwave_prepared.numel())
                nonwave_prepared_leg_max_abs = max(
                    nonwave_prepared_leg_max_abs, float(nonwave_prepared.max().item())
                )
                nonwave_joint_posture_error_max = max(
                    nonwave_joint_posture_error_max,
                    float(joint_posture_error[nonwave_gate].max().item()),
                )
            wheel_target_sum += (
                robot.data.joint_vel_target[:, wheel_joint_ids] * wheel_direction_signs
            ).sum(dim=0)
            wheel_velocity_sum += (
                robot.data.joint_vel[:, wheel_joint_ids] * wheel_direction_signs
            ).sum(dim=0)
            wheel_torque_abs = robot.data.applied_torque[:, wheel_joint_ids].abs()
            wheel_torque_abs_sum += wheel_torque_abs.sum(dim=0)
            wheel_torque_abs_max = torch.maximum(
                wheel_torque_abs_max, wheel_torque_abs.max(dim=0).values
            )
            wheel_height_sum += robot.data.body_pos_w[:, wheel_body_ids, 2].sum(dim=0)
            wheel_height_max = torch.maximum(
                wheel_height_max, robot.data.body_pos_w[:, wheel_body_ids, 2].max(dim=0).values
            )
            wheel_height_max_by_env = torch.maximum(
                wheel_height_max_by_env, robot.data.body_pos_w[:, wheel_body_ids, 2]
            )
            wheel_force = torch.linalg.vector_norm(
                contact_sensor.data.net_forces_w[:, contact_body_ids, :], dim=-1
            )
            wheel_contact_sum += (wheel_force > 1.0).to(dtype=torch.float32).sum(dim=0)
            wheel_sample_rows += args.num_envs
            active_rows = evaluation_active.unsqueeze(1)
            wheel_pos_local = (
                robot.data.body_pos_w[:, wheel_body_ids]
                - env.unwrapped.scene.env_origins.unsqueeze(1)
            )
            wheel_relative_to_base = (
                robot.data.body_pos_w[:, wheel_body_ids]
                - robot.data.root_pos_w.unsqueeze(1)
            )
            wheel_relative_x_min = torch.minimum(
                wheel_relative_x_min,
                torch.where(
                    active_rows,
                    wheel_relative_to_base[..., 0],
                    torch.full_like(wheel_relative_to_base[..., 0], torch.inf),
                ),
            )
            wheel_relative_x_max = torch.maximum(
                wheel_relative_x_max,
                torch.where(
                    active_rows,
                    wheel_relative_to_base[..., 0],
                    torch.full_like(wheel_relative_to_base[..., 0], -torch.inf),
                ),
            )
            wheel_relative_z_max = torch.maximum(
                wheel_relative_z_max,
                torch.where(
                    active_rows,
                    wheel_relative_to_base[..., 2],
                    torch.full_like(wheel_relative_to_base[..., 2], -torch.inf),
                ),
            )
            next_prelift, next_overbar_clear, required_wheel_clearance_height = (
                update_wheel_obstacle_clearance(
                    wheel_pos_local=wheel_pos_local,
                    wheel_contact_force=wheel_force,
                    prelift_seen=wheel_prelift_seen,
                    overbar_clear_seen=wheel_overbar_clear_seen,
                    obstacle_center_x=obstacle_center_x,
                    obstacle_size_x=obstacle_size_x,
                    obstacle_height=obstacle_height,
                    wheel_radius=wheel_radius,
                    clearance_margin=clearance_margin,
                    contact_force_limit=clearance_contact_force_limit,
                    obstacle_center_y=obstacle_center_y,
                    obstacle_size_y=obstacle_size_y,
                )
            )
            new_prelift = active_rows & next_prelift & ~wheel_prelift_seen
            if bool(new_prelift.any()):
                root_x_now = (
                    robot.data.root_pos_w[:, 0] - env.unwrapped.scene.env_origins[:, 0]
                ).unsqueeze(1).expand(-1, 4)
                first_wheel_prelift_root_x = torch.where(
                    new_prelift, root_x_now, first_wheel_prelift_root_x
                )
            wheel_prelift_seen = torch.where(
                active_rows, next_prelift, wheel_prelift_seen
            )
            wheel_overbar_clear_seen = torch.where(
                active_rows, next_overbar_clear, wheel_overbar_clear_seen
            )
            half_obstacle_size_x = 0.5 * obstacle_size_x
            lateral_swept_region = (
                torch.abs(wheel_pos_local[..., 1] - obstacle_center_y)
                <= 0.5 * obstacle_size_y + wheel_radius
            )
            wheel_clearance_required |= active_rows & lateral_swept_region
            precontact_region = wheel_pos_local[..., 0] <= (
                obstacle_center_x - half_obstacle_size_x - wheel_radius
            )
            overbar_region = (
                torch.abs(wheel_pos_local[..., 0] - obstacle_center_x)
                <= half_obstacle_size_x
            )
            precontact_region &= lateral_swept_region
            overbar_region &= lateral_swept_region
            precontact_active = precontact_region & active_rows
            overbar_active = overbar_region & active_rows
            max_precontact_wheel_height = torch.maximum(
                max_precontact_wheel_height,
                torch.where(
                    precontact_active,
                    wheel_pos_local[..., 2],
                    torch.full_like(wheel_pos_local[..., 2], -torch.inf),
                ),
            )
            max_overbar_wheel_height = torch.maximum(
                max_overbar_wheel_height,
                torch.where(
                    overbar_active,
                    wheel_pos_local[..., 2],
                    torch.full_like(wheel_pos_local[..., 2], -torch.inf),
                ),
            )
            min_overbar_wheel_contact_force = torch.minimum(
                min_overbar_wheel_contact_force,
                torch.where(
                    overbar_active,
                    wheel_force,
                    torch.full_like(wheel_force, torch.inf),
                ),
            )
            axle_overbar_now = torch.stack(
                (overbar_active[:, :2].any(dim=1), overbar_active[:, 2:].any(dim=1)),
                dim=1,
            )
            new_axle_overbar = axle_overbar_now & ~axle_overbar_action_seen
            spatial_obstacle_x = getattr(
                env.unwrapped, "m1_wave_spatial_obstacle_x", None
            )
            for axle_index in range(2):
                capture = new_axle_overbar[:, axle_index]
                prepared_leg_action_at_axle_overbar[capture, axle_index] = (
                    prepared_leg_actions[capture]
                )
                if spatial_obstacle_x is not None:
                    obstacle_x_at_axle_overbar[capture, axle_index] = (
                        spatial_obstacle_x[capture]
                    )
            axle_overbar_action_seen |= axle_overbar_now
            _, pitch = extract_roll_pitch_batch(robot.data.root_quat_w)
            root_pos = root_before_step - env.unwrapped.scene.env_origins
            base_height = root_pos[:, 2]
            base_height_target = float(getattr(env_cfg, "base_height_target", 0.55))
            recovery_start_x = float(getattr(env_cfg, "base_height_recovery_start_x", 0.70))
            recovery_mask = (root_pos[:, 0] >= recovery_start_x) & evaluation_active
            base_height_error = torch.abs(base_height - base_height_target)
            base_height_sum += base_height * recovery_mask.to(base_height.dtype)
            base_height_samples += recovery_mask.to(base_height.dtype)
            max_abs_base_height_error = torch.maximum(
                max_abs_base_height_error,
                base_height_error * recovery_mask.to(base_height.dtype),
            )
            final_base_height = torch.where(
                evaluation_active, base_height, final_base_height
            )
            wheel_heights = robot.data.body_pos_w[:, wheel_body_ids, 2]
            front_rear_height_delta = wheel_heights[:, :2].mean(dim=1) - wheel_heights[:, 2:].mean(dim=1)
            pitch_sum += pitch
            front_rear_height_delta_sum += front_rear_height_delta
            max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(pitch))
            final_pitch = pitch
            final_front_rear_height_delta = front_rear_height_delta
            if args.semantic_crossing:
                scanner = env.unwrapped.scene["semantic_height_scanner"]
                crossing_candidate_xy, crossing_candidate_heading, crossing_candidate_valid, semantic_crossed = (
                    update_semantic_crossing_tracker(
                        root_pos_w=robot.data.root_pos_w,
                        root_quat_w=robot.data.root_quat_w,
                        ray_hits_w=scanner.data.ray_hits_w,
                        semantic_map=scanner.data.semantic_map,
                        candidate_xy=crossing_candidate_xy,
                        candidate_heading=crossing_candidate_heading,
                        candidate_valid=crossing_candidate_valid,
                        crossed=semantic_crossed,
                    )
                )
            projected_g = robot.data.projected_gravity_b
            tilt = torch.acos(torch.clamp(torch.abs(projected_g[:, 2]), -1.0, 1.0))
            max_tilt = torch.maximum(
                max_tilt,
                torch.where(evaluation_active, tilt, torch.zeros_like(tilt)),
            )
            step_dx = robot.data.root_pos_w[:, 0] - root_start[:, 0]
            max_dx = torch.maximum(
                max_dx,
                torch.where(evaluation_active, step_dx, max_dx),
            )
            crossing_success = torch.zeros_like(success_seen)
            if args.obstacle_threshold is not None:
                crossing_success = max_dx >= float(args.obstacle_threshold)
                if args.semantic_crossing:
                    crossing_success &= semantic_crossed
            new_success = crossing_success & evaluation_active
            root_at_done[new_success] = robot.data.root_pos_w[new_success]
            success_seen |= new_success
            new_done = dones & ~done_seen & ~success_seen
            root_at_done[new_done] = root_before_step[new_done]
            timeout_seen |= new_done & torch.as_tensor(extras["time_outs"], device=done_seen.device).bool()
            bad_orientation_seen |= new_done & _termination_term(env, "bad_orientation")
            done_seen |= dones & ~success_seen
            if bool((done_seen | success_seen).all()):
                break

        final_root = robot.data.root_pos_w.clone()
        eval_root = torch.where(
            (done_seen | success_seen).unsqueeze(-1), root_at_done, final_root
        )
        final_root_local = eval_root - env.unwrapped.scene.env_origins
        final_wheel_pos_local = (
            robot.data.body_pos_w[:, wheel_body_ids]
            - env.unwrapped.scene.env_origins.unsqueeze(1)
        )
        final_leg_joint_pos = robot.data.joint_pos[:, leg_joint_ids]
        final_leg_joint_target = robot.data.joint_pos_target[:, leg_joint_ids]
        final_leg_applied_torque = robot.data.applied_torque[:, leg_joint_ids]
        delta = eval_root - root_start
        normal_tilt_limit, wave_tilt_limit = _configured_orientation_limits(env_cfg)
        if args.obstacle_threshold is not None:
            report = evaluate_obstacle_gate(
                episodes=args.num_envs,
                timeout_episodes=int(timeout_seen.sum().item()),
                bad_orientation_episodes=int(bad_orientation_seen.sum().item()),
                crossing_episodes=int((max_dx >= args.obstacle_threshold).sum().item()),
                mean_max_dx=float(max_dx.mean().item()),
                mean_dy=float(delta[:, 1].mean().item()),
                max_tilt_rad=float(max_tilt.max().item()),
                max_tilt_limit_rad=wave_tilt_limit,
            )
        else:
            report = evaluate_roll_gate(
                episodes=args.num_envs,
                timeout_episodes=int(timeout_seen.sum().item()),
                bad_orientation_episodes=int(bad_orientation_seen.sum().item()),
                mean_dx=float(delta[:, 0].mean().item()),
                mean_dy=float(delta[:, 1].mean().item()),
                max_tilt_rad=float(max_tilt.max().item()),
            )
        report.update(
            {
                "normal_tilt_limit_rad": normal_tilt_limit,
                "wave_tilt_limit_rad": wave_tilt_limit,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "perception_checkpoint": (
                    str(args.perception_checkpoint.resolve())
                    if args.perception_checkpoint is not None
                    else None
                ),
                "task": args.task,
                "seed": args.seed,
            }
        )
        ik_joint_ids = getattr(env.unwrapped, "m1_task_space_ik_joint_ids", None)
        ik_jacobians = getattr(env.unwrapped, "m1_task_space_ik_jacobians", None)
        ik_full_jacobians = getattr(
            env.unwrapped, "m1_task_space_ik_full_jacobians", None
        )
        ik_actions = getattr(env.unwrapped, "m1_task_space_ik_actions", None)
        if ik_joint_ids is not None:
            report["task_space_ik_joint_ids"] = ik_joint_ids.detach().cpu().tolist()
        if ik_jacobians is not None:
            report["task_space_ik_jacobian_env0"] = (
                ik_jacobians[0].detach().cpu().tolist()
            )
        if ik_full_jacobians is not None:
            report["task_space_ik_full_jacobian_env0"] = (
                ik_full_jacobians[0].detach().cpu().tolist()
            )
        if ik_actions is not None:
            report["task_space_ik_action_env0"] = ik_actions[0].detach().cpu().tolist()
        mean_wheel_velocity = wheel_velocity_sum / max(wheel_sample_rows, 1)
        wheel_contact_rate = wheel_contact_sum / max(wheel_sample_rows, 1)
        max_mean_wheel_velocity_spread = float(
            (mean_wheel_velocity.max() - mean_wheel_velocity.min()).item()
        )
        wheel_velocity_sync_ok = max_mean_wheel_velocity_spread <= 0.08
        all_wheels_contact_ok = bool((wheel_contact_rate >= 0.95).all().item())
        wheel_contact_requirement_applied = not args.semantic_crossing
        wheel_sync_passed = bool(
            wheel_velocity_sync_ok
            and (all_wheels_contact_ok or not wheel_contact_requirement_applied)
        )
        mean_base_height = base_height_sum / torch.clamp(base_height_samples, min=1.0)
        base_height_tolerance = float(getattr(env_cfg, "base_height_recovery_tolerance", 0.06))
        has_recovery_samples = base_height_samples > 0
        base_height_recovery_ok = bool(
            has_recovery_samples.all().item()
            and (torch.abs(mean_base_height - base_height_target) <= base_height_tolerance).all().item()
            and (torch.abs(final_base_height - base_height_target) <= base_height_tolerance).all().item()
        )
        roll_sync_enabled = bool(
            getattr(env_cfg, "roll_equal_wheel_actions", False)
        ) and bool(getattr(env_cfg, "roll_sync_actual_wheel_velocity", False))
        if bool(getattr(env_cfg, "wave_sync_actual_wheel_velocity", False)) or roll_sync_enabled:
            report["passed"] = bool(report["passed"] and wheel_sync_passed)
        semantic_candidate_episodes = int(crossing_candidate_valid.sum().item())
        semantic_crossing_episodes = int((semantic_crossed & crossing_candidate_valid).sum().item())
        semantic_candidate_rate = semantic_candidate_episodes / max(args.num_envs, 1)
        semantic_crossing_rate = semantic_crossing_episodes / max(semantic_candidate_episodes, 1)
        semantic_crossing_passed = bool(
            semantic_candidate_rate >= 0.50
            and semantic_crossing_rate >= float(args.min_crossing_rate)
        )
        wave_mean_abs_prepared_leg_action = (
            wave_prepared_leg_abs_sum / max(wave_prepared_leg_count, 1)
        )
        nonwave_stance_command_ok = bool(
            nonwave_prepared_leg_count > 0 and nonwave_prepared_leg_max_abs <= 1.0e-6
        )
        wave_motion_passed = bool(
            wave_prepared_leg_count > 0
            and wave_mean_abs_prepared_leg_action >= 0.02
            and wave_joint_posture_error_max >= 0.05
        )
        if args.semantic_crossing:
            report["passed"] = bool(report["passed"] and semantic_crossing_passed)
            report["passed"] = bool(report["passed"] and base_height_recovery_ok)
        if bool(getattr(env_cfg, "wave_unclipped_policy_legs", False)):
            report["passed"] = bool(report["passed"] and nonwave_stance_command_ok)
            report["passed"] = bool(report["passed"] and wave_motion_passed)
        termination_tilt_limit = wave_tilt_limit
        configured_tilt_limit = float(
            report.get("max_tilt_limit_rad", termination_tilt_limit)
        )
        acceptance_max_tilt = float(
            getattr(env_cfg, "acceptance_max_tilt_rad", configured_tilt_limit)
        )
        report["max_tilt_limit_rad"] = configured_tilt_limit
        report["max_tilt_limit_rad"] = min(
            float(report["max_tilt_limit_rad"]), acceptance_max_tilt
        )
        report["tilt_ok"] = bool(report["max_tilt_rad"] <= report["max_tilt_limit_rad"])
        wave_action_delta_limit = float(
            getattr(env_cfg, "wave_max_action_delta_acceptance", float("inf"))
        )
        wave_action_smooth_passed = bool(
            wave_max_prepared_leg_action_delta <= wave_action_delta_limit
        )
        min_front_wheel_height = float(
            getattr(env_cfg, "acceptance_min_front_wheel_height_m", -float("inf"))
        )
        min_rear_wheel_height = float(
            getattr(env_cfg, "acceptance_min_rear_wheel_height_m", -float("inf"))
        )
        front_pair_lift_passed, rear_pair_lift_passed = required_axle_lift_passed(
            wheel_height_max=wheel_height_max,
            wheel_clearance_required=wheel_clearance_required,
            min_front_height=min_front_wheel_height,
            min_rear_height=min_rear_wheel_height,
        )
        active_wheel_clearance_by_env = wheel_prelift_seen & wheel_overbar_clear_seen
        active_wheel_clearance_passed = bool(
            wheel_clearance_required.any(dim=1).all().item()
            and (
                active_wheel_clearance_by_env | ~wheel_clearance_required
            ).all().item()
        )
        require_active_wheel_clearance = bool(
            getattr(env_cfg, "acceptance_require_active_wheel_clearance", True)
        )
        if args.semantic_crossing and require_active_wheel_clearance:
            report["passed"] = bool(report["passed"] and active_wheel_clearance_passed)
        report["passed"] = bool(
            report["passed"]
            and report["tilt_ok"]
            and wave_action_smooth_passed
            and front_pair_lift_passed
            and rear_pair_lift_passed
        )
        report.update(
            {
                "mean_abs_leg_action": leg_abs_sum / max(leg_action_count, 1),
                "max_abs_leg_action": max_abs_leg_action,
                "leg_action_saturation_rate": saturated_leg_actions / max(leg_action_count, 1),
                "mean_leg_action_by_index": (
                    leg_action_sum_by_index / max(leg_action_rows, 1)
                ).detach().cpu().tolist(),
                "wheel_order": list(wheel_body_names),
                "wheel_joint_order": wheel_joint_order,
                "mean_wheel_target_by_index": (
                    wheel_target_sum / max(wheel_sample_rows, 1)
                ).detach().cpu().tolist(),
                "mean_wheel_velocity_by_index": (
                    mean_wheel_velocity
                ).detach().cpu().tolist(),
                "mean_abs_wheel_torque_by_index": (
                    wheel_torque_abs_sum / max(wheel_sample_rows, 1)
                ).detach().cpu().tolist(),
                "max_abs_wheel_torque_by_index": (
                    wheel_torque_abs_max.detach().cpu().tolist()
                ),
                "mean_wheel_height_by_index": (
                    wheel_height_sum / max(wheel_sample_rows, 1)
                ).detach().cpu().tolist(),
                "mean_pitch_rad": (
                    pitch_sum / max(args.steps, 1)
                ).detach().cpu().tolist(),
                "final_pitch_rad_by_env": final_pitch.detach().cpu().tolist(),
                "mean_base_height_m": mean_base_height.detach().cpu().tolist(),
                "final_base_height_m_by_env": final_base_height.detach().cpu().tolist(),
                "final_root_pos_m_by_env": final_root_local.detach().cpu().tolist(),
                "final_wheel_pos_m_by_env": final_wheel_pos_local.detach().cpu().tolist(),
                "final_leg_joint_pos_rad_by_env": (
                    final_leg_joint_pos.detach().cpu().tolist()
                ),
                "final_leg_joint_target_rad_by_env": (
                    final_leg_joint_target.detach().cpu().tolist()
                ),
                "final_leg_applied_torque_nm_by_env": (
                    final_leg_applied_torque.detach().cpu().tolist()
                ),
                "max_abs_base_height_error_m_by_env": (
                    max_abs_base_height_error.detach().cpu().tolist()
                ),
                "base_height_recovery_ok": base_height_recovery_ok,
                "max_abs_pitch_rad_by_env": max_abs_pitch.detach().cpu().tolist(),
                "mean_front_rear_wheel_height_delta": (
                    front_rear_height_delta_sum / max(args.steps, 1)
                ).detach().cpu().tolist(),
                "final_front_rear_wheel_height_delta_by_env": (
                    final_front_rear_height_delta.detach().cpu().tolist()
                ),
                "max_wheel_height_by_index": wheel_height_max.detach().cpu().tolist(),
                "max_wheel_height_by_env": wheel_height_max_by_env.detach().cpu().tolist(),
                "wheel_relative_x_min_m_by_env": (
                    wheel_relative_x_min.detach().cpu().tolist()
                ),
                "wheel_relative_x_max_m_by_env": (
                    wheel_relative_x_max.detach().cpu().tolist()
                ),
                "wheel_relative_z_max_m_by_env": (
                    wheel_relative_z_max.detach().cpu().tolist()
                ),
                "max_tilt_by_env": max_tilt.detach().cpu().tolist(),
                "wheel_contact_rate_by_index": (
                    wheel_contact_rate
                ).detach().cpu().tolist(),
                "max_mean_wheel_velocity_spread": max_mean_wheel_velocity_spread,
                "wheel_velocity_sync_ok": wheel_velocity_sync_ok,
                "all_wheels_contact_ok": all_wheels_contact_ok,
                "wheel_contact_requirement_applied": wheel_contact_requirement_applied,
                "wheel_sync_passed": wheel_sync_passed,
                "semantic_candidate_episodes": semantic_candidate_episodes,
                "semantic_candidate_rate": semantic_candidate_rate,
                "semantic_crossing_episodes": semantic_crossing_episodes,
                "semantic_crossing_rate": semantic_crossing_rate,
                "semantic_crossing_passed": semantic_crossing_passed,
                "policy_gate_score_min_by_env": policy_gate_score_min.detach().cpu().tolist(),
                "policy_gate_score_max_by_env": policy_gate_score_max.detach().cpu().tolist(),
                "policy_gate_positive_score_min_by_env": (
                    policy_gate_positive_score_min.detach().cpu().tolist()
                ),
                "policy_gate_positive_score_max_by_env": (
                    policy_gate_positive_score_max.detach().cpu().tolist()
                ),
                "policy_gate_true_positive_rate_by_env": (
                    policy_gate_true_positive
                    / torch.clamp(policy_gate_positive_samples, min=1.0)
                ).detach().cpu().tolist(),
                "policy_gate_true_negative_rate_by_env": (
                    policy_gate_true_negative
                    / torch.clamp(policy_gate_negative_samples, min=1.0)
                ).detach().cpu().tolist(),
                "leg_action_joint_order": leg_action_joint_order,
                "wave_sample_count": wave_prepared_leg_count,
                "wave_mean_abs_prepared_leg_action": wave_mean_abs_prepared_leg_action,
                "wave_max_abs_prepared_leg_action": wave_prepared_leg_max_abs,
                "wave_max_abs_prepared_leg_action_by_index": (
                    wave_prepared_leg_max_abs_by_index.detach().cpu().tolist()
                ),
                "wave_first_prepared_leg_action": (
                    wave_first_prepared_leg_action.detach().cpu().tolist()
                    if wave_first_prepared_leg_action is not None
                    else None
                ),
                "wave_mean_prepared_leg_action_by_index": (
                    wave_prepared_leg_sum_by_index / max(wave_prepared_leg_rows, 1)
                ).detach().cpu().tolist(),
                "wave_max_prepared_leg_action_delta": wave_max_prepared_leg_action_delta,
                "wave_action_delta_limit": wave_action_delta_limit,
                "wave_action_smooth_passed": wave_action_smooth_passed,
                "min_front_wheel_height_m": min_front_wheel_height,
                "min_rear_wheel_height_m": min_rear_wheel_height,
                "front_pair_lift_passed": front_pair_lift_passed,
                "rear_pair_lift_passed": rear_pair_lift_passed,
                "wheel_prelift_seen_by_env": wheel_prelift_seen.detach().cpu().tolist(),
                "first_wheel_prelift_root_x_m_by_env": (
                    first_wheel_prelift_root_x.detach().cpu().tolist()
                ),
                "wheel_overbar_clear_seen_by_env": (
                    wheel_overbar_clear_seen.detach().cpu().tolist()
                ),
                "active_wheel_clearance_by_env": (
                    active_wheel_clearance_by_env.detach().cpu().tolist()
                ),
                "wheel_clearance_required_by_env": (
                    wheel_clearance_required.detach().cpu().tolist()
                ),
                "active_wheel_clearance_passed": active_wheel_clearance_passed,
                "active_wheel_clearance_required": require_active_wheel_clearance,
                "required_wheel_center_height_m": required_wheel_clearance_height,
                "clearance_contact_force_limit_n": clearance_contact_force_limit,
                "max_precontact_wheel_height_by_env": (
                    max_precontact_wheel_height.detach().cpu().tolist()
                ),
                "max_overbar_wheel_height_by_env": (
                    max_overbar_wheel_height.detach().cpu().tolist()
                ),
                "min_overbar_wheel_contact_force_by_env": (
                    min_overbar_wheel_contact_force.detach().cpu().tolist()
                ),
                "prepared_leg_action_at_axle_overbar_by_env": (
                    prepared_leg_action_at_axle_overbar.detach().cpu().tolist()
                ),
                "spatial_obstacle_x_at_axle_overbar_by_env": (
                    obstacle_x_at_axle_overbar.detach().cpu().tolist()
                ),
                "crossbar_collision_mask_by_env": (
                    crossbar_collision_mask.detach().cpu().tolist()
                ),
                "crossbar_collision_wheel_pos_m_by_env": (
                    crossbar_collision_wheel_pos.detach().cpu().tolist()
                ),
                "crossbar_collision_wheel_force_n_by_env": (
                    crossbar_collision_wheel_force.detach().cpu().tolist()
                ),
                "crossbar_collision_phase_by_env": (
                    crossbar_collision_phase.detach().cpu().tolist()
                ),
                "crossbar_collision_phase_steps_by_env": (
                    crossbar_collision_phase_steps.detach().cpu().tolist()
                ),
                "crossbar_collision_wheel_x_from_obstacle_by_env": (
                    crossbar_collision_wheel_x_from_obstacle.detach().cpu().tolist()
                ),
                "nonwave_sample_count": nonwave_prepared_leg_count,
                "nonwave_max_abs_prepared_leg_action": nonwave_prepared_leg_max_abs,
                "wave_max_joint_posture_error_rad": wave_joint_posture_error_max,
                "nonwave_max_joint_posture_error_rad": nonwave_joint_posture_error_max,
                "nonwave_stance_command_ok": nonwave_stance_command_ok,
                "wave_motion_passed": wave_motion_passed,
                "front_pair_lift_swing_phase_by_env": (
                    front_pair_lift_swing_phase.detach().cpu().tolist()
                ),
                "sequential_crossing_phase_by_env": (
                    torch.as_tensor(
                        getattr(
                            env.unwrapped,
                            "m1_sequential_crossing_phase",
                            torch.full(
                                (args.num_envs,),
                                -1,
                                device=env.unwrapped.device,
                            ),
                        )
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ),
                "max_sequential_crossing_phase_by_env": (
                    max_sequential_crossing_phase.detach().cpu().tolist()
                ),
                "first_sequential_phase_root_x_m_by_env": (
                    first_sequential_phase_root_x.detach().cpu().tolist()
                ),
                "first_sequential_phase_wheel_x_m_by_env": (
                    first_sequential_phase_wheel_x.detach().cpu().tolist()
                ),
                "phase_entry_root_pos_m_by_env": (
                    phase_entry_root_pos.detach().cpu().tolist()
                ),
                "phase_entry_wheel_pos_m_by_env": (
                    phase_entry_wheel_pos.detach().cpu().tolist()
                ),
                "phase_entry_wheel_velocity_by_env": (
                    phase_entry_wheel_velocity.detach().cpu().tolist()
                ),
                "phase_entry_leg_joint_pos_rad_by_env": (
                    phase_entry_leg_joint_pos.detach().cpu().tolist()
                ),
                "phase_entry_leg_joint_target_rad_by_env": (
                    phase_entry_leg_joint_target.detach().cpu().tolist()
                ),
            }
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if report["passed"] and args.promote:
            args.promote.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.checkpoint, args.promote)
            report["promoted_to"] = str(args.promote.resolve())
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)
        env.close()
        exit_code = 0 if report["passed"] else 2
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
