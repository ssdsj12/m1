from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
M1_TRAIN_CFG = REPO_ROOT / "agent" / "m1_train_cfg.py"
M1_WRAPPER = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_rsl_rl_wrapper.py"
M1_TRAIN = REPO_ROOT / "scripts" / "m1_train.py"
M1_PLAY = REPO_ROOT / "scripts" / "m1_play.py"
M1_STAGE05_RUNNER = REPO_ROOT / "scripts" / "run_m1_roll_stage05_forward_long.sh"
M1_STABILITY_PROBE = REPO_ROOT / "scripts" / "m1_stability_probe.py"
M1_CHECKPOINT_EVAL = REPO_ROOT / "scripts" / "m1_checkpoint_eval.py"
M1_WAVE_DISTILL = REPO_ROOT / "scripts" / "m1_wave_distill.py"
M1_ROLL_CFG = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_roll_env_cfg.py"


def test_m1_train_cfg_declares_plain_mlp_ppo():
    source = M1_TRAIN_CFG.read_text()

    assert "def get_m1_train_cfg" in source
    assert '"class_name": "ActorCritic"' in source
    assert '"class_name": "PPO"' in source
    assert '"num_steps_per_env": 24' in source
    assert '"empirical_normalization": False' in source
    assert '"entropy_coef": 0.0' in source
    assert '"init_noise_std": 0.01' in source


def test_m1_rsl_rl_wrapper_flattens_policy_group():
    source = M1_WRAPPER.read_text()

    assert "class M1RslRlEnvWrapper(VecEnv)" in source
    assert 'obs_dict["policy"]' in source
    assert '"observations": {"critic": obs}' in source
    assert "extras[\"time_outs\"] = truncated" in source
    assert "torch.clamp(actions" in source
    assert "roll_equal_wheel_actions" in source
    assert "prepared[:, 12:16]" in source
    assert "wheel_action = 0.40 + 0.05 * raw" in source


def test_m1_roll_stage_synchronizes_actual_wheel_velocity():
    wrapper = M1_WRAPPER.read_text()
    cfg = M1_ROLL_CFG.read_text()
    evaluator = M1_CHECKPOINT_EVAL.read_text()

    assert "roll_sync_actual_wheel_velocity" in cfg
    assert "roll_wheel_equalize_gain" in cfg
    assert 'getattr(self.cfg, "roll_sync_actual_wheel_velocity", False)' in wrapper
    assert 'getattr(env_cfg, "roll_sync_actual_wheel_velocity", False)' in evaluator
    assert 'getattr(env_cfg, "roll_equal_wheel_actions", False)' in evaluator


def test_m1_train_script_uses_m1_task_and_runner():
    source = M1_TRAIN.read_text()

    assert "Isaac-M1-Walk-v0" in source
    assert "get_m1_train_cfg" in source
    assert "M1RslRlEnvWrapper" in source
    assert "OnPolicyRunner" in source
    assert "logs/m1_walk" in source
    assert "runner.learn" in source
    assert "--clip-actions" in source
    assert "clip_actions=args.clip_actions" in source
    assert 'parser.add_argument("--init-noise-std", type=float, default=None)' in source
    assert 'train_cfg["policy"]["init_noise_std"] = args.init_noise_std' in source
    assert "runner.alg.actor_critic.std.data.fill_(args.init_noise_std)" in source


def test_m1_play_script_supports_open_loop_and_checkpoint_policy():
    source = M1_PLAY.read_text()

    assert "Isaac-M1-Walk-v0" in source
    assert "--checkpoint" in source
    assert "build_m1_smoke_action" in source
    assert "M1RslRlEnvWrapper" in source
    assert "runner.load" in source
    assert "get_inference_policy" in source
    assert "env.step(actions)" in source
    assert "--clip-actions" in source
    assert 'parser.add_argument("--lock-legs", action="store_true")' in source
    assert 'parser.add_argument("--disable-crossing-reset", action="store_true")' in source
    assert 'parser.add_argument("--enable-wave-reference-actions", action="store_true")' in source
    assert "env_cfg.wave_reference_actions = True" in source
    assert 'parser.add_argument("--wave-leg-action-limit", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-reference-amplitude", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-reference-knee-ratio", type=float, default=None)' in source
    assert 'parser.add_argument("--rear-support-ratio", type=float, default=None)' in source
    assert 'parser.add_argument("--obstacle-wheel-action", type=float, default=None)' in source
    assert "env_cfg.wave_leg_action_limit = 0.0" in source
    assert "env_cfg.wave_leg_action_limit = args.wave_leg_action_limit" in source
    assert "env_cfg.wave_reference_raw_amplitude = args.wave_reference_amplitude" in source
    assert "env_cfg.wave_reference_knee_ratio = args.wave_reference_knee_ratio" in source
    assert "env_cfg.wave_rear_support_ratio = args.rear_support_ratio" in source
    assert "env_cfg.wave_obstacle_wheel_action = args.obstacle_wheel_action" in source
    assert "env_cfg.wave_reference_actions = False" in source
    assert "env_cfg.terminations.crossing_success = None" in source
    assert "env_cfg.wave_front_wheel_action = args.rolling_wheel_velocity" in source
    assert "env_cfg.wave_rear_wheel_action = args.rolling_wheel_velocity" in source


def test_m1_stage05_long_runner_uses_forward_roll_task():
    source = M1_STAGE05_RUNNER.read_text()

    assert "Isaac-M1-Roll-v0" in source
    assert "--max_iterations 3000" in source
    assert "m1_roll_stage05_forward_long" in source
    assert "--clip-actions 1.0" in source


def test_m1_stability_probe_reports_max_forward_progress():
    source = M1_STABILITY_PROBE.read_text()

    assert "max_root_x" in source
    assert '"mean_max_dx"' in source
    assert "mean_max_dx" in source
    assert '"--wheel-effort-limit"' in source
    assert '"--wheel-damping"' in source
    assert '"--explicit-wheel-actuator"' in source
    assert '"--episode-length"' in source
    assert '"--wheel-pulse-actions"' in source
    assert '"--leg-stiffness"' in source
    assert "DCMotorCfg(" in source
    assert 'actuators["wheels"].effort_limit_sim' in source
    assert 'actuators["wheels"].damping' in source
    assert '"mean_wheel_pos"' in source
    assert '"mean_wheel_vel"' in source


def test_m1_checkpoint_eval_reports_per_wheel_motion_and_contact():
    source = M1_CHECKPOINT_EVAL.read_text()

    assert '"mean_wheel_target_by_index"' in source
    assert '"mean_wheel_velocity_by_index"' in source
    assert '"mean_wheel_height_by_index"' in source
    assert '"mean_front_rear_wheel_height_delta"' in source
    assert '"mean_pitch_rad"' in source
    assert '"final_pitch_rad_by_env"' in source
    assert '"mean_base_height_m"' in source
    assert '"final_base_height_m_by_env"' in source
    assert '"max_abs_base_height_error_m_by_env"' in source
    assert '"base_height_recovery_ok"' in source
    assert '"wheel_contact_rate_by_index"' in source
    assert '"wheel_sync_passed"' in source
    assert 'parser.add_argument("--wheel-action", type=float, default=None)' in source
    assert "env_cfg.wave_front_wheel_action = args.wheel_action" in source
    assert 'report["passed"] = bool(report["passed"] and wheel_sync_passed)' in source
    assert "wheel_contact_requirement_applied = not args.semantic_crossing" in source
    assert '"wheel_contact_requirement_applied"' in source
    assert "_configured_orientation_limits(env_cfg)" in source
    assert '"normal_tilt_limit_rad"' in source
    assert '"wave_tilt_limit_rad"' in source
    assert '"--semantic-crossing"' in source
    assert 'parser.add_argument("--disable-crossing-reset", action="store_true")' in source
    assert 'parser.add_argument("--enable-wave-reference-actions", action="store_true")' in source
    assert "env_cfg.wave_reference_actions = True" in source
    assert 'parser.add_argument("--wave-reference-knee-ratio", type=float, default=None)' in source
    assert 'parser.add_argument("--rear-support-ratio", type=float, default=None)' in source
    assert "env_cfg.wave_reference_knee_ratio = args.wave_reference_knee_ratio" in source
    assert "env_cfg.wave_rear_support_ratio = args.rear_support_ratio" in source
    assert "leg_action_limit_cfg is None" in source
    assert "leg_action_limit = float(\"inf\")" in source
    assert "env_cfg.terminations.crossing_success = None" in source
    assert '"semantic_crossing_passed"' in source
    assert "success_seen" in source
    assert "new_success" in source
    assert "done_seen | success_seen" in source
    assert '"max_wheel_height_by_index"' in source
    assert '"wheel_joint_order"' in source
    assert '"mean_abs_wheel_torque_by_index"' in source
    assert '"max_abs_wheel_torque_by_index"' in source
    assert '"final_root_pos_m_by_env"' in source
    assert '"final_wheel_pos_m_by_env"' in source
    assert '"final_leg_joint_pos_rad_by_env"' in source
    assert 'parser.add_argument("--obstacle-front-wheel-action", type=float, default=None)' in source
    assert 'parser.add_argument("--obstacle-rear-wheel-action", type=float, default=None)' in source
    assert 'parser.add_argument("--wheel-equalize-gain", type=float, default=None)' in source
    assert 'parser.add_argument("--wheel-equalize-max-correction", type=float, default=None)' in source
    assert 'parser.add_argument("--phase-wheel-assist", type=float, default=None)' in source
    assert "env_cfg.wave_phase_wheel_assist = args.phase_wheel_assist" in source
    assert "env_cfg.wave_obstacle_front_wheel_action = args.obstacle_front_wheel_action" in source
    assert "env_cfg.wave_obstacle_rear_wheel_action = args.obstacle_rear_wheel_action" in source


def test_m1_checkpoint_eval_can_override_sequential_prebalance_for_physics_probes():
    source = M1_CHECKPOINT_EVAL.read_text()

    assert 'parser.add_argument("--wave-balance-steps", type=int, default=None)' in source
    assert 'parser.add_argument("--wave-support-extension", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-opposite-abduction", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-front-start-x", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-front-hip-action", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-front-knee-action", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-rear-hip-action", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-rear-knee-action", type=float, default=None)' in source
    assert "env_cfg.wave_sequential_balance_steps = args.wave_balance_steps" in source
    assert "env_cfg.wave_sequential_support_extension = args.wave_support_extension" in source
    assert "env_cfg.wave_sequential_opposite_abduction = args.wave_opposite_abduction" in source
    assert "env_cfg.wave_sequential_front_start_x_m = args.wave_front_start_x" in source
    assert "env_cfg.wave_sequential_front_hip_action = args.wave_front_hip_action" in source
    assert "env_cfg.wave_sequential_front_knee_action = args.wave_front_knee_action" in source
    assert "env_cfg.wave_sequential_rear_hip_action = args.wave_rear_hip_action" in source
    assert "env_cfg.wave_sequential_rear_knee_action = args.wave_rear_knee_action" in source


def test_m1_checkpoint_eval_distinguishes_wave_execution_from_non_wave_stance():
    source = M1_CHECKPOINT_EVAL.read_text()

    assert 'getattr(env.unwrapped, "m1_wave_gate"' in source
    assert 'getattr(env.unwrapped, "m1_prepared_leg_actions"' in source


def test_m1_checkpoint_eval_requires_contact_free_wheel_clearance():
    source = M1_CHECKPOINT_EVAL.read_text()

    assert "update_wheel_obstacle_clearance" in source
    assert '"wheel_prelift_seen_by_env"' in source
    assert '"wheel_overbar_clear_seen_by_env"' in source
    assert '"active_wheel_clearance_passed"' in source
    assert "report[\"passed\"] and active_wheel_clearance_passed" in source
    assert '"max_precontact_wheel_height_by_env"' in source
    assert '"max_overbar_wheel_height_by_env"' in source
    assert '"min_overbar_wheel_contact_force_by_env"' in source
    assert 'parser.add_argument("--spatial-wave-reference", action="store_true")' in source
    assert "env_cfg.wave_spatial_reference = True" in source
    assert 'parser.add_argument("--disable-left-right-symmetry", action="store_true")' in source
    assert "env_cfg.wave_left_right_symmetric = False" in source
    assert '"final_leg_joint_target_rad_by_env"' in source
    assert '"final_leg_applied_torque_nm_by_env"' in source
    assert '"crossbar_collision_mask_by_env"' in source
    assert '"crossbar_collision_wheel_pos_m_by_env"' in source
    assert '"crossbar_collision_wheel_force_n_by_env"' in source
    assert '"crossbar_collision_phase_by_env"' in source
    assert '"crossbar_collision_phase_steps_by_env"' in source
    assert '"crossbar_collision_wheel_x_from_obstacle_by_env"' in source
    assert 'parser.add_argument("--front-wave-window", type=str, default=None)' in source
    assert 'parser.add_argument("--rear-wave-window", type=str, default=None)' in source
    assert 'parser.add_argument("--front-pair-lift-swing-probe", action="store_true")' in source
    assert 'parser.add_argument("--single-wheel-lift-swing-probe", type=int, default=None)' in source
    assert 'parser.add_argument("--sequential-front-wheel-probe", action="store_true")' in source
    assert "front_pair_lift_swing_phase" in source
    assert "env_cfg.wave_front_lift_window = _parse_wave_window" in source
    assert "env_cfg.wave_rear_lift_window = _parse_wave_window" in source
    assert 'parser.add_argument("--temporal-wave-reference", action="store_true")' in source
    assert 'parser.add_argument("--dynamic-wave-phase", action="store_true")' in source
    assert 'parser.add_argument("--wave-reference-frequency", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-pulse-ramp", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-pulse-hold", type=float, default=None)' in source
    assert 'parser.add_argument("--wave-cycle-duration", type=float, default=None)' in source
    assert "env_cfg.wave_spatial_reference = False" in source
    assert "env_cfg.wave_reference_constant_phase_s = None" in source
    assert '"nonwave_max_abs_prepared_leg_action"' in source
    assert '"wave_mean_abs_prepared_leg_action"' in source
    assert '"wave_max_joint_posture_error_rad"' in source
    assert '"nonwave_stance_command_ok"' in source
    assert '"wave_motion_passed"' in source
    assert 'getattr(env_cfg, "wave_unclipped_policy_legs", False)' in source
    assert 'report["passed"] = bool(report["passed"] and wave_motion_passed)' in source
    assert '"wave_max_prepared_leg_action_delta"' in source
    assert '"wave_max_abs_prepared_leg_action_by_index"' in source
    assert '"wave_first_prepared_leg_action"' in source
    assert '"wave_mean_prepared_leg_action_by_index"' in source
    assert '"wave_action_smooth_passed"' in source
    assert 'getattr(env_cfg, "acceptance_max_tilt_rad"' in source
    assert '"front_pair_lift_passed"' in source
    assert '"rear_pair_lift_passed"' in source
    assert 'getattr(env_cfg, "acceptance_min_front_wheel_height_m"' in source
    assert 'getattr(env_cfg, "acceptance_min_rear_wheel_height_m"' in source
    assert '"wheel_relative_x_min_m_by_env"' in source
    assert '"wheel_relative_x_max_m_by_env"' in source
    assert '"wheel_relative_z_max_m_by_env"' in source


def test_m1_checkpoint_eval_has_a_tilt_limit_for_roll_gate_reports():
    source = M1_CHECKPOINT_EVAL.read_text()

    assert 'report.get("max_tilt_limit_rad", termination_tilt_limit)' in source


def test_m1_checkpoint_eval_can_sweep_learned_wave_gate_threshold():
    source = M1_CHECKPOINT_EVAL.read_text()

    assert 'parser.add_argument("--policy-gate-threshold"' in source
    assert "env_cfg.wave_policy_gate_threshold = args.policy_gate_threshold" in source


def test_m1_wave_distillation_uses_dagger_rollout_and_soft_temporal_losses():
    source = M1_WAVE_DISTILL.read_text()

    assert 'default="Isaac-M1-Pvcnn-Crossing-60mm-Guided-Fixed-v0"' in source
    assert "scheduled_student_rollout_weight" in source
    assert 'parser.add_argument("--student-rollout-final-weight"' in source
    assert 'parser.add_argument("--smoothness-weight"' in source
    assert 'parser.add_argument("--overshoot-weight"' in source
    assert "F.smooth_l1_loss" in source
    assert "temporal_loss" in source
    assert "overshoot_loss" in source
    assert "active_wave = torch.linalg.vector_norm(teacher_actions, dim=1) > 1.0e-6" in source
    assert "prediction[active_wave]" in source
    assert "teacher_actions[active_wave]" in source
    assert "env_cfg.wave_sequential_policy_control = not args.hierarchical_gate" in source
    assert "env_cfg.wave_sequential_policy_weight = student_weight" in source
    assert 'parser.add_argument("--teacher-forcing-fraction"' in source
    assert "expand_checkpoint_observations" in source
    assert "optimizer = torch.optim.Adam(actor_critic.actor.parameters()" in source
    assert 'parser.add_argument("--wheel-preservation-weight"' in source
    assert "wheel_preservation_loss" in source
    assert 'parser.add_argument("--hierarchical-gate"' in source
    assert "env_cfg.wave_gate_from_policy_action = bool(args.hierarchical_gate)" in source
    assert "binary_cross_entropy_with_logits" in source
    assert "m1_wave_gate_target" in source
    assert 'parser.add_argument("--nonwave-weight"' in source
    assert "nonwave_loss" in source
