from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG_FILE = ROOT / "go2_pvcnn" / "tasks" / "m1_pvcnn_small_obstacle_env_cfg.py"
REGISTER_FILE = ROOT / "go2_pvcnn" / "tasks" / "register_m1_envs.py"
SEMANTIC_CLEARANCE_FILE = ROOT / "extension" / "mdp" / "semantic_body_part_clearance.py"
FRONT_OBSTACLE_PLAY = ROOT / "scripts" / "run_m1_front_obstacle_play.sh"
WRAPPER_FILE = ROOT / "go2_pvcnn" / "tasks" / "m1_rsl_rl_wrapper.py"


def test_m1_pvcnn_small_obstacle_cfg_reuses_original_semantic_environment() -> None:
    source = CFG_FILE.read_text()

    assert "TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg" in source
    assert "class M1PvcnnFlatSmallAvoidanceEnvCfg(" in source
    assert "scene: M1PvcnnFlatSmallSceneCfg" in source
    assert "M1_CFG.replace" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK"' in source
    assert "actions: M1PvcnnCrossingActionsCfg" in source
    assert "SemanticCourseTerrainImporter" not in source
    assert "make_obstacle_cfg" not in source
    assert "obstacle =" not in source


def test_m1_pvcnn_small_obstacle_cfg_uses_m1_body_contract() -> None:
    source = CFG_FILE.read_text()

    assert "M1_BASE_BODY_NAME" in source
    assert "M1_FOOT_BODY_NAMES" in source
    assert "M1_LEG_JOINT_NAMES" in source
    assert "M1_WHEEL_JOINT_NAMES" in source
    assert "SEMANTIC_CONTACT_BODY_NAMES" not in source
    assert "wave_fixed_forward_wheels: bool = True" in source
    assert "wave_leg_action_limit: float = 0.60" in source
    assert "wave_reference_raw_amplitude: float = 0.55" in source
    assert "wheel_velocity_sync = _m1_wheel_velocity_sync_term()" in source
    assert "wheel_action_match = None" in source
    assert "def _m1_semantic_body_clearance_term()" in source
    assert '"foot_sphere_radius_m": 0.096' in source
    assert 'body_names=list(M1_FOOT_BODY_NAMES)' in source
    assert "def _m1_semantic_wheel_over_term()" in source
    assert "def _m1_semantic_pair_lift_reward(" in source
    assert "spatial_wheel_lift_score(" in source
    assert "wheel_crossbar_collision_mask" in source
    assert "def _m1_wheel_crossbar_contact(" in source
    assert "self.terminations.crossbar_contact = DoneTerm(" in source
    assert "semantic_pair_lift = RewTerm(" in source
    assert "crossing_success = RewTerm(" in source
    assert "crossing_progress = RewTerm(" in source
    assert "raw_leg_action_l2 = RewTerm(" in source
    assert "class M1PvcnnCrossingActionsCfg(M1RollActionsCfg)" in source
    assert "scale=0.80" in source
    assert "preserve_order=True" in source
    assert "wave_semantic_obstacle_gating: bool = True" in source
    assert "wave_spatial_reference: bool = True" in source
    assert "wave_reference_raw_amplitude: float = 0.55" in source
    assert "wave_sync_actual_wheel_velocity: bool = True" in source
    assert "wave_obstacle_wheel_action: float = 0.25" in source
    assert "wave_obstacle_sync_max_correction: float = 0.10" in source
    assert "wave_wheel_sync_gain: float = 0.50" in source
    assert "wave_wheel_sync_max_correction: float = 0.50" in source
    assert "class M1PvcnnCrossing60mmEnvCfg" in source
    assert "wave_leg_action_limit: float = 0.08" in source
    assert "wave_policy_leg_residual_limit: float = 0.0" in source
    assert "wave_obstacle_wheel_action: float = 0.50" in source
    assert "wave_obstacle_front_wheel_action: float = 0.20" in source
    assert "wave_obstacle_rear_wheel_action: float = 0.95" in source
    assert "wave_disable_obstacle_after_root_x: float = 1.15" in source
    assert 'semantic_course_scale_profile_overrides = {"small": (0.60, 0.06)}' in source
    assert "semantic_course_cuboid_size_overrides" in source
    assert '"small": (0.06, 0.60, 0.06)' in source
    assert "wave_obstacle_sync_max_correction: float = 0.30" in source
    assert "wave_wheel_equalize_gain: float = 2.0" in source
    assert "wave_wheel_equalize_max_correction: float = 0.50" in source
    assert "wave_lock_left_right_wheel_targets: bool = True" in source
    assert "wave_lock_all_wheel_targets: bool = True" in source
    assert "wave_wheel_action_signs: tuple[float, float, float, float] | None = None" in source
    assert "self.rewards.wheel_velocity_sync = _m1_wheel_velocity_sync_term(weight=-80.0)" in source
    assert "base_height_recovery = RewTerm(" in source
    assert "joint_posture_recovery = RewTerm(" in source
    assert '"target_height": 0.55' in source
    assert 'self.rewards.base_height_recovery.params["recovery_start_x"] = (' in source
    assert 'self.rewards.joint_posture_recovery.params["recovery_start_x"] = (' in source
    assert 'self.rewards.crossing_success.params["threshold"] = 1.45' in source
    assert 'params={"threshold": 1.50, "asset_cfg": SceneEntityCfg("robot")}' in source
    assert "PolicyElevationSemanticMapCfg" in source
    assert "terrain_generator.num_cols = 2" in source
    assert "semantic_course_mandatory_small_xy = (0.40, 0.0)" in source
    assert "self.episode_length_s = 120.0" in source
    assert 'self.events.reset_base.params["pose_range"] = {' in source
    assert '"yaw": (0.0, 0.0)' in source
    assert 'self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)' in source
    assert "self.events.push_robot = None" in source


def test_m1_crossing_demo_has_fixed_front_obstacle_and_play_registration() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()

    assert "class M1PvcnnCrossing60mmPlayEnvCfg(M1PvcnnCrossing60mmEnvCfg)" in cfg
    assert "self.scene.terrain.terrain_generator.num_rows = 1" in cfg
    assert "self.scene.terrain.terrain_generator.num_cols = 1" in cfg
    assert "self.scene.terrain.semantic_course_mandatory_small_xy = (0.65, 0.0)" in cfg
    assert "self.curriculum.terrain_levels = None" in cfg
    assert "self.observations.policy.enable_corruption = False" in cfg
    assert "self.observations.policy_elevation_semantic_map.enable_corruption = False" in cfg
    assert "M1PvcnnCrossing60mmPlayEnvCfg" in registry
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Play-v0"' in registry


def test_contact_free_crossing_tasks_gate_policy_to_axle_windows() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()
    wrapper = WRAPPER_FILE.read_text()
    contact_free_cfg = cfg.split(
        "class M1PvcnnCrossing60mmContactFreeTrainEnvCfg", 1
    )[1].split("@configclass", 1)[0]

    assert "class M1PvcnnCrossing60mmContactFreeTrainEnvCfg(" in cfg
    assert "class M1PvcnnCrossing60mmContactFreePlayEnvCfg(" in cfg
    assert "wave_reference_actions: bool = False" in cfg
    assert "wave_gate_from_spatial_reference: bool = False" in cfg
    assert "wave_left_right_symmetric: bool = False" in contact_free_cfg
    assert "wave_lock_abduction: bool = False" in contact_free_cfg
    assert "wave_hold_wheels_until_axle_clear: bool = False" in cfg
    assert "wave_sequential_crossing_reference: bool = True" in cfg
    assert "wave_axle_pair_crossing_reference: bool = False" in cfg
    assert "wave_axle_pair_ramp_steps: int = 10" in cfg
    assert "wave_axle_pair_support_steps: int = 20" in cfg
    assert "wave_axle_pair_front_start_x_m: float = -0.28" in cfg
    assert "wave_sequential_swing_steps: int = 50" in cfg
    assert "wave_sequential_past_bar_x_m: float = 0.15" in cfg
    assert "wave_sequential_balance_steps: int = 0" in contact_free_cfg
    assert "wave_sequential_support_extension: float = 0.0" in contact_free_cfg
    assert "wave_sequential_opposite_abduction: float = -0.10" in contact_free_cfg
    assert "wave_sequential_front_hip_action: float = -0.30" in contact_free_cfg
    assert "wave_sequential_front_knee_action: float = -0.60" in contact_free_cfg
    assert "wave_sequential_rear_hip_action: float = 0.30" in contact_free_cfg
    assert "wave_sequential_rear_knee_action: float = 0.60" in contact_free_cfg
    assert "wave_task_space_rear_restore_forward_offset_m: float = 0.12" in contact_free_cfg
    assert "wave_sequential_support_residual_scale: float = 0.0" in contact_free_cfg
    assert "wave_sequential_crossing_residual_scale: float = 0.0" in contact_free_cfg
    assert "wave_sequential_support_abduction_residual_scale: float = 0.0" in contact_free_cfg
    assert "wave_sequential_front_start_x_m: float = -0.45" in contact_free_cfg
    assert "wave_fixed_obstacle_center_x_m: float | None = 0.85" in cfg
    assert "self.scene.terrain.semantic_course_mandatory_small_xy = (0.85, -0.20)" in contact_free_cfg
    assert 'crossbar_params["obstacle_center_x"] = 0.85' in contact_free_cfg
    assert 'getattr(self.cfg, "wave_sequential_front_start_x_m", -0.20)' in wrapper
    assert '"wave_fixed_obstacle_center_x_m", None' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_support_extension", 0.0)' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_opposite_abduction", 0.0)' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_balance_steps", 0)' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_front_hip_action", -1.50)' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_front_knee_action", -0.80)' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_rear_hip_action", 1.50)' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_rear_knee_action", 0.80)' in wrapper
    assert '"wave_sequential_crossing_residual_scale"' in wrapper
    assert 'getattr(self.cfg, "wave_sequential_past_bar_x_m", 0.13)' in wrapper
    assert "wave_clearance_minimum_hold_s: float = 0.15" in cfg
    assert "wave_axle_clearance_height_m: float = 0.16" in cfg
    assert "wave_axle_switch_obstacle_x: float = 0.05" in cfg
    assert "wave_front_lift_window: tuple[float, float, float, float] = (0.80, 0.70, 0.15, 0.05)" in cfg
    assert "wave_obstacle_front_wheel_action: float = 6.0" in contact_free_cfg
    assert "wave_obstacle_rear_wheel_action: float = 6.0" in contact_free_cfg
    assert "wave_sequential_swing_wheel_action: float = 7.0" in contact_free_cfg
    assert '"wave_sequential_swing_wheel_action", None' in wrapper
    assert "wave_rear_wheel_velocity_feedforward: float = 0.40" in contact_free_cfg
    assert '"wave_rear_wheel_velocity_feedforward", 0.0' in wrapper
    assert "wave_wheel_equalize_gain: float = 3.0" in contact_free_cfg
    assert "wave_wheel_equalize_max_correction: float = 1.0" in contact_free_cfg
    assert "wave_lateral_steering_gain: float = 0.0" in cfg
    assert "wave_yaw_damping_gain: float = 0.0" in cfg
    assert "update_clearance_drive_release" in wrapper
    assert "update_sequential_wheel_crossing_reference" in wrapper
    assert "sequential_leg_residual_scale" in wrapper
    assert "m1_sequential_crossing_phase" in wrapper
    assert "m1_sequential_wheel_x_from_obstacle" in wrapper
    assert "sequential_wheel_crossing_progress_score" in cfg
    assert "self.rewards.sequential_crossing_progress = RewTerm(" in cfg
    pair_cfg = cfg.split(
        "class M1PvcnnCrossing60mmPairCurriculumEnvCfg", 1
    )[1].split("@configclass", 1)[0]
    assert "self.rewards.crossing_progress = None" in pair_cfg
    assert "func=_m1_strict_sequential_crossing_success" in pair_cfg
    assert "self.terminations.crossbar_contact = None" in pair_cfg
    assert "self.rewards.lateral_position_l2 = RewTerm(" in cfg
    assert "func=_m1_phase_aware_minimum_base_height" in cfg
    assert '"normal_minimum_height": 0.35' in cfg
    assert '"wave_minimum_height": 0.30' in cfg
    assert "func=_m1_phase_aware_bad_orientation" in cfg
    assert cfg.count("| (phase == 2)") == 2
    assert '"normal_limit_angle": 0.60' in cfg
    assert '"wave_limit_angle": 0.90' in cfg
    assert "m1_wave_drive_allowed" in wrapper
    assert "m1_crossbar_collision_mask" in cfg
    assert "m1_crossbar_collision_wheel_pos_local" in cfg
    assert "m1_crossbar_collision_wheel_force" in cfg
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Train-v0"' in registry
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Play-v0"' in registry


def test_pair_curriculum_penalizes_contact_but_requires_full_sequence() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()
    pair_cfg = cfg.split(
        "class M1PvcnnCrossing60mmPairCurriculumEnvCfg", 1
    )[1].split("@configclass", 1)[0]

    assert "class M1PvcnnCrossing60mmPairCurriculumEnvCfg(" in cfg
    assert "self.terminations.crossbar_contact = None" in pair_cfg
    assert "self.rewards.crossbar_contact_penalty = RewTerm(" in pair_cfg
    assert "weight=-10.0" in pair_cfg
    assert "self.rewards.sequential_crossing_progress.weight = 200.0" in pair_cfg
    assert "func=_m1_strict_sequential_crossing_success" in pair_cfg
    assert "self.wave_axle_clearance_height_m = 0.16" in cfg
    assert "self.wave_sequential_past_bar_x_m = 0.15" in cfg
    assert "self.wave_pair_curriculum_swing_timeout_steps = None" in cfg
    assert "self.terminations.minimum_base_height = None" not in pair_cfg
    assert 'bad_orientation_params["wave_limit_angle"] = 0.65' in pair_cfg
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Pair-Curriculum-v0"' in registry


def test_sequential_progress_reward_uses_potential_delta() -> None:
    cfg = CFG_FILE.read_text()
    reward_start = cfg.index("def _m1_sequential_crossing_progress(env):")
    reward_end = cfg.index("\ndef _m1_lateral_position_l2", reward_start)
    reward_impl = cfg[reward_start:reward_end]

    assert "previous = getattr(env.unwrapped, \"m1_crossing_progress_potential\", None)" in reward_impl
    assert "delta, next_potential = progress_potential_delta(" in reward_impl
    assert "return delta" in reward_impl
    assert "return score" not in reward_impl


def test_sequential_teacher_locks_policy_leg_residuals() -> None:
    cfg = CFG_FILE.read_text()

    assert "wave_sequential_support_residual_scale: float = 0.0" in cfg
    assert "wave_sequential_crossing_residual_scale: float = 0.0" in cfg
    assert "wave_sequential_support_abduction_residual_scale: float = 0.0" in cfg


def test_contact_free_crossing_uses_right_track_sequential_wave() -> None:
    cfg = CFG_FILE.read_text()
    wrapper = WRAPPER_FILE.read_text()
    contact_free_cfg = cfg.split(
        "class M1PvcnnCrossing60mmContactFreeTrainEnvCfg", 1
    )[1].split("@configclass", 1)[0]

    assert "wave_sequential_crossing_reference: bool = True" in cfg
    assert "wave_axle_pair_crossing_reference: bool = False" in cfg
    assert "wave_right_track_only: bool = True" in cfg

    assert "wave_task_space_ik: bool = True" in contact_free_cfg
    assert "wave_task_space_support_only: bool = True" in contact_free_cfg
    assert "wave_task_space_active_swing_xz_only: bool = True" in contact_free_cfg
    assert "wave_task_space_lift_delta_m: float = 0.22\n" in contact_free_cfg
    assert "wave_task_space_rear_lift_delta_m: float = 0.20\n" in contact_free_cfg
    assert "wave_task_space_max_joint_step: float = 0.25" in cfg
    assert "wave_wheel_action_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)" in cfg
    assert "root_physx_view.get_jacobians()" in wrapper
    assert "self._task_space_nominal_wheel_pos_b" in wrapper
    assert "self._task_space_nominal_base_height" in wrapper
    assert "wave_task_space_stop_during_wave: bool = False" in cfg
    assert "task_space_active_phase" in wrapper
    assert "build_stabilized_task_space_wheel_actions(" in wrapper
    assert "merge_task_space_support_with_jointspace_active(" in wrapper
    assert "joint_space_reference=joint_space_reference" in wrapper
    assert "wave_task_space_balance_steps: int = 30\n" in contact_free_cfg
    assert "self.scene.terrain.semantic_course_mandatory_small_xy = (0.85, -0.20)" in cfg
    assert '"small": (0.06, 0.16, 0.06)' in cfg
    assert "self.semantic_obstacle_curriculum.non_plane_counts = zero_counts" in cfg
    assert "self.scene.terrain.semantic_obstacle_curriculum = (" in cfg
    assert "wave_task_space_lateral_body_shift_m: float = 0.06\n" in contact_free_cfg
    assert "wave_task_space_longitudinal_body_shift_m: float = 0.04\n" in contact_free_cfg
    assert "wave_task_space_balance_supports: bool = True\n" in contact_free_cfg
    assert "wave_sequential_front_start_x_m: float = -0.45\n" in contact_free_cfg
    assert "wave_sequential_rear_start_x: float = -0.10\n" in contact_free_cfg
    assert "wave_obstacle_front_wheel_action: float = 6.0\n" in contact_free_cfg
    assert "wave_obstacle_rear_wheel_action: float = 6.0\n" in contact_free_cfg
    assert "wave_sequential_swing_wheel_action: float = 7.0\n" in contact_free_cfg
    assert "wave_lock_all_wheel_targets: bool = False\n" in contact_free_cfg
    assert "wave_task_space_lateral_recovery_gain: float = 0.0\n" in contact_free_cfg
    assert "wave_task_space_lateral_recovery_max_m: float = 0.0\n" in contact_free_cfg
    assert "base_lateral_offset=root_lateral_offset" in wrapper
    assert "lateral_recovery_gain=float(" in wrapper
    assert "wave_task_space_swing_with_body: bool = True\n" in contact_free_cfg
    assert "wave_sequential_keep_drive_during_wave: bool = True\n" in contact_free_cfg
    assert 'getattr(self.cfg, "wave_sequential_keep_drive_during_wave", False)' in wrapper
    assert "swing_with_body=bool(" in wrapper
    assert "wave_task_space_stabilize_supports: bool = True\n" in contact_free_cfg
    assert "stabilize_supports=bool(" in wrapper
    assert "wave_task_space_swing_ramp_steps: int = 20\n" in contact_free_cfg
    assert "swing_ramp_steps=int(" in wrapper
    assert "longitudinal_body_shift=float(" in wrapper
    assert "balance_supports=bool(" in wrapper
    assert "wave_sequential_support_residual_scale: float = 0.0\n" in contact_free_cfg
    assert "wave_sequential_crossing_residual_scale: float = 0.0\n" in contact_free_cfg
    assert "wave_sequential_support_abduction_residual_scale: float = 0.0\n" in contact_free_cfg


def test_autonomous_policy_crossing_replaces_teacher_but_keeps_wave_gate() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()

    assert "class M1PvcnnCrossing60mmPolicyPlayEnvCfg" in cfg
    assert "wave_sequential_policy_control: bool = False" in cfg
    assert "wave_gate_from_policy_action: bool = True" in cfg
    assert "wave_policy_gate_action_index: int = 15" in cfg
    assert "wave_policy_gate_threshold: float = -0.03" in cfg
    assert "wave_policy_gate_minimum_root_x_m: float = 0.120" in cfg
    assert "wave_policy_gate_fallback_root_x_m: float = 0.125" in cfg
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Policy-Play-v0"' in registry


def test_m1_crossbar_termination_uses_small_obstacle_filtered_contacts() -> None:
    cfg = CFG_FILE.read_text()
    sensor = (
        ROOT
        / "go2_pvcnn"
        / "sensor"
        / "semantic_contacter"
        / "semantic_global_contact_sensor.py"
    ).read_text()

    assert "class M1SemanticGlobalContactSensor" in sensor
    assert '"FAR_FOOT_LINK"' in sensor
    assert "semantic_contact_small = _m1_semantic_global_contact_sensor" in cfg
    assert '"contact_sensor_cfg": SceneEntityCfg(' in cfg
    assert '"semantic_contact_small",' in cfg


def test_m1_front_obstacle_play_script_uses_accepted_pvcnn_pair() -> None:
    source = FRONT_OBSTACLE_PLAY.read_text()

    assert "run_m1_contactfree_policy_play.sh" in source


def test_m1_wrapper_flattens_and_joins_policy_observation_groups() -> None:
    import torch

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    wrapper = object.__new__(M1RslRlEnvWrapper)
    obs, extras = wrapper._format_observations(
        {
            "policy": torch.ones((2, 3)),
            "policy_elevation_semantic_map": 2.0 * torch.ones((2, 2, 4, 4)),
        }
    )

    assert obs.shape == (2, 35)
    torch.testing.assert_close(obs[:, :3], torch.ones((2, 3)))
    torch.testing.assert_close(obs[:, 3:], 2.0 * torch.ones((2, 32)))
    torch.testing.assert_close(extras["observations"]["critic"], obs)


def test_m1_wrapper_wheel_sync_correction_slows_fast_wheels_and_boosts_slow_wheels() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.tensor([[0.60, 0.60, 0.20, 0.20]])),
        find_joints=lambda *args, **kwargs: ([0, 1, 2, 3], []),
    )
    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.0,
        wave_reference_actions=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=True,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.40,
        wave_rear_wheel_action=0.40,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=True,
        wave_wheel_sync_gain=0.50,
        wave_wheel_sync_integral_gain=1.0,
        wave_wheel_sync_integral_limit=0.50,
        wave_wheel_sync_max_correction=0.20,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={"robot": robot},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.zeros(1, dtype=torch.long)),
    )
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.zeros((1, 16)))

    assert prepared[0, 12] < 0.40
    assert prepared[0, 13] < 0.40
    assert prepared[0, 14] > 0.40
    assert prepared[0, 15] > 0.40


def test_wave_leg_gate_locks_stance_outside_wave_without_clipping_inside() -> None:
    import torch

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import _gate_wave_leg_actions

    actions = torch.tensor([[2.5, -1.7], [2.5, -1.7]])
    gate = torch.tensor([False, True])

    prepared = _gate_wave_leg_actions(actions, gate, action_limit=None)

    torch.testing.assert_close(prepared[0], torch.zeros(2))
    torch.testing.assert_close(prepared[1], actions[1])


def test_spatial_reference_gate_releases_legs_only_in_crossing_windows() -> None:
    import torch

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import _gate_from_spatial_reference

    gate = _gate_from_spatial_reference(
        obstacle_active=torch.tensor([True, True, False]),
        spatial_reference=torch.tensor(
            [
                [0.0] * 12,
                [0.0, 0.8, -1.6] + [0.0] * 9,
                [0.0, 0.8, -1.6] + [0.0] * 9,
            ]
        ),
    )

    assert gate.tolist() == [False, True, False]


def test_precomputed_spatial_wave_reference_is_added_to_policy_actions() -> None:
    import torch

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import _combine_wave_reference_actions

    policy_actions = torch.tensor([[0.1, -0.2]])
    spatial_reference = torch.tensor([[0.4, 0.3]])

    combined = _combine_wave_reference_actions(policy_actions, spatial_reference)

    torch.testing.assert_close(combined, torch.tensor([[0.5, 0.1]]))


def test_m1_unlocked_crossing_course_releases_policy_only_during_wave() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()

    assert "class M1PvcnnCrossing60mmUnlockedEnvCfg(M1PvcnnCrossing60mmEnvCfg)" in cfg
    assert "wave_leg_action_limit: float | None = None" in cfg
    assert "wave_policy_leg_residual_limit: float | None = None" in cfg
    assert "wave_reference_actions: bool = False" in cfg
    assert "wave_unclipped_policy_legs: bool = True" in cfg
    assert "class M1PvcnnUnlockedCrossingActionsCfg(M1RollActionsCfg)" in cfg
    assert "clip=None" in cfg
    assert "wheel_vel = mdp.JointVelocityActionCfg(" in cfg
    assert "joint_names=list(M1_WHEEL_JOINT_NAMES)" in cfg
    assert "def _m1_prepared_leg_action_l2(env)" in cfg
    assert "self.rewards.raw_leg_action_l2 = RewTerm(" in cfg
    assert "self.rewards.semantic_pair_lift.weight = 20.0" in cfg
    assert "func=_m1_prepared_leg_action_l2, weight=-2.0" in cfg
    assert "wave_obstacle_front_wheel_action: float = 0.50" in cfg
    assert "wave_obstacle_rear_wheel_action: float = 0.50" in cfg
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Unlocked-v0"' in registry


def test_m1_guided_crossing_course_adds_wave_teacher_without_action_limits() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()

    assert "class M1PvcnnCrossing60mmGuidedEnvCfg(M1PvcnnCrossing60mmUnlockedEnvCfg)" in cfg
    assert "wave_reference_actions: bool = True" in cfg
    assert "wave_reference_raw_amplitude: float = 0.80" in cfg
    assert "wave_reference_knee_ratio: float = -2.0" in cfg
    assert "wave_reference_smoothing_alpha: float = 1.0" in cfg
    assert "wave_reference_pulse_ramp_s: float = 0.02" in cfg
    assert "wave_reference_pulse_hold_s: float = 1.96" in cfg
    assert "wave_reference_time_offset_s: float = 0.02" in cfg
    assert "wave_reference_constant_phase_s: float = 0.5" in cfg
    assert "wave_spatial_reference: bool = False" in cfg
    assert "wave_single_cycle_duration_s: float = 2.0" in cfg
    assert "wave_rear_amplitude_scale: float = 0.0" in cfg
    assert "wave_front_support_ratio: float = 0.0" in cfg
    assert "acceptance_max_tilt_rad: float = 0.45" in cfg
    assert "acceptance_min_front_wheel_height_m: float = 0.13" in cfg
    assert "acceptance_min_rear_wheel_height_m: float = 0.14" in cfg
    assert "wave_max_action_delta_acceptance: float = 2.0" in cfg
    assert "wave_rear_support_ratio: float = 0.50" in cfg
    assert "wave_obstacle_front_wheel_action: float = 6.4" in cfg
    assert "wave_obstacle_rear_wheel_action: float = 8.0" in cfg
    assert "wave_phase_wheel_assist: float = 0.0" in cfg
    assert "wave_lock_left_right_wheel_targets: bool = False" in cfg
    assert "wave_lateral_steering_gain: float = 2.0" in cfg
    assert "wave_yaw_damping_gain: float = 0.5" in cfg
    assert "wave_steering_max_correction: float = 0.5" in cfg
    assert "wave_wheel_equalize_max_correction: float = 0.80" in cfg
    assert "func=_m1_raw_leg_action_l2, weight=-2.0" in cfg
    wrapper = (ROOT / "go2_pvcnn" / "tasks" / "m1_rsl_rl_wrapper.py").read_text()
    assert "m1_wave_reference_actions" in wrapper
    assert "build_spatial_axle_wheel_targets" in wrapper
    assert "build_temporal_axle_wheel_targets" in wrapper
    assert "build_lateral_steering_correction" in wrapper
    assert "spatial_obstacle_x" in wrapper
    assert "wave_single_cycle_duration_s" in wrapper
    assert "wave_reference_time_offset_s" in wrapper
    assert "wave_reference_constant_phase_s" in wrapper
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Guided-v0"' in registry
    assert "base_height_target: float = 0.57" in cfg
    assert 'self.rewards.base_height_recovery.params["target_height"] = self.base_height_target' in cfg


def test_m1_guided_fixed_course_matches_distilled_play_geometry() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()

    assert "class M1PvcnnCrossing60mmGuidedFixedEnvCfg(" in cfg
    assert "M1PvcnnCrossing60mmGuidedEnvCfg" in cfg
    assert 'self.experiment_name = "m1_pvcnn_crossing_60mm_guided_fixed"' in cfg
    assert "self.scene.terrain.terrain_generator.num_rows = 1" in cfg
    assert "self.scene.terrain.terrain_generator.num_cols = 1" in cfg
    assert "self.scene.terrain.semantic_course_mandatory_small_xy = (0.65, 0.0)" in cfg
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Guided-Fixed-v0"' in registry


def test_m1_distilled_play_course_is_fixed_and_has_no_teacher() -> None:
    cfg = CFG_FILE.read_text()
    registry = REGISTER_FILE.read_text()

    assert "class M1PvcnnCrossing60mmDistilledPlayEnvCfg(" in cfg
    assert "M1PvcnnCrossing60mmGuidedEnvCfg" in cfg
    assert "wave_reference_actions: bool = False" in cfg
    assert "self.scene.terrain.terrain_generator.num_rows = 1" in cfg
    assert "self.scene.terrain.terrain_generator.num_cols = 1" in cfg
    assert "self.episode_length_s = 20.0" in cfg
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-Distilled-Play-v0"' in registry


def test_m1_wrapper_can_equalize_fast_wheels_down_to_slowest_actual_velocity() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.tensor([[0.65, 0.65, 0.40, 0.40]])),
        find_joints=lambda *args, **kwargs: ([0, 1, 2, 3], []),
    )
    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.0,
        wave_reference_actions=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=True,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.50,
        wave_rear_wheel_action=0.50,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=True,
        wave_wheel_sync_gain=0.0,
        wave_wheel_sync_integral_gain=0.0,
        wave_wheel_sync_integral_limit=0.50,
        wave_wheel_sync_max_correction=0.0,
        wave_wheel_equalize_gain=2.0,
        wave_wheel_equalize_max_correction=0.50,
        wave_wheel_equalize_to_slowest=True,
        wave_forward_only_wheels=True,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={"robot": robot},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.zeros(1, dtype=torch.long)),
    )
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.zeros((1, 16)))

    assert prepared[0, 12] < 0.50
    assert prepared[0, 13] < 0.50
    assert torch.isclose(prepared[0, 14], torch.tensor(0.50))
    assert torch.isclose(prepared[0, 15], torch.tensor(0.50))
    assert (prepared[0, 12:16] >= 0.0).all()
    assert torch.isclose(prepared[0, 12], prepared[0, 13])
    assert torch.isclose(prepared[0, 14], prepared[0, 15])


def test_slowest_wheel_reference_never_propagates_reverse_velocity() -> None:
    import torch
    import go2_pvcnn.tasks.m1_rsl_rl_wrapper as wrapper_module

    assert hasattr(wrapper_module, "_slowest_forward_wheel_velocity")
    reference = wrapper_module._slowest_forward_wheel_velocity(
        torch.tensor([[1.0, 0.8, -0.4, -0.7], [1.0, 0.8, 0.4, 0.6]])
    )
    torch.testing.assert_close(reference, torch.tensor([[0.0], [0.4]]))


def test_wheel_sync_integral_accumulates_during_obstacle_wave() -> None:
    import torch
    import go2_pvcnn.tasks.m1_rsl_rl_wrapper as wrapper_module

    assert hasattr(wrapper_module, "_update_wheel_sync_integral")
    integral = wrapper_module._update_wheel_sync_integral(
        previous=torch.zeros((1, 4)),
        error=torch.ones((1, 4)),
        step_dt=0.02,
        limit=0.50,
    )
    integral = wrapper_module._update_wheel_sync_integral(
        previous=integral,
        error=torch.ones((1, 4)),
        step_dt=0.02,
        limit=0.50,
    )

    torch.testing.assert_close(integral, torch.full((1, 4), 0.04))


def test_m1_wrapper_can_lock_left_right_wheel_targets_after_velocity_corrections() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.tensor([[0.20, 0.80, 0.10, 0.90]])),
        find_joints=lambda *args, **kwargs: ([0, 1, 2, 3], []),
    )
    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.0,
        wave_reference_actions=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=True,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.50,
        wave_rear_wheel_action=0.50,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=True,
        wave_wheel_sync_gain=0.50,
        wave_wheel_sync_integral_gain=0.0,
        wave_wheel_sync_integral_limit=0.50,
        wave_wheel_sync_max_correction=0.50,
        wave_wheel_equalize_gain=2.0,
        wave_wheel_equalize_max_correction=0.50,
        wave_lock_left_right_wheel_targets=True,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={"robot": robot},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.zeros(1, dtype=torch.long)),
    )
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.zeros((1, 16)))

    assert torch.isclose(prepared[0, 12], prepared[0, 13])
    assert torch.isclose(prepared[0, 14], prepared[0, 15])


def test_m1_wrapper_can_lock_all_wheel_targets_after_velocity_corrections() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    robot = SimpleNamespace(
        data=SimpleNamespace(joint_vel=torch.tensor([[0.20, 0.80, 0.10, 0.90]])),
        find_joints=lambda *args, **kwargs: ([0, 1, 2, 3], []),
    )
    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.0,
        wave_reference_actions=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=True,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.30,
        wave_rear_wheel_action=0.70,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=True,
        wave_wheel_sync_gain=0.50,
        wave_wheel_sync_integral_gain=0.0,
        wave_wheel_sync_integral_limit=0.50,
        wave_wheel_sync_max_correction=0.50,
        wave_wheel_equalize_gain=2.0,
        wave_wheel_equalize_max_correction=0.50,
        wave_lock_left_right_wheel_targets=True,
        wave_lock_all_wheel_targets=True,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={"robot": robot},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.zeros(1, dtype=torch.long)),
    )
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.zeros((1, 16)))

    torch.testing.assert_close(prepared[0, 12:16], prepared[0, 12].expand(4))


def test_m1_wrapper_applies_wheel_action_signs_after_target_locking() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.0,
        wave_reference_actions=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=True,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.50,
        wave_rear_wheel_action=0.50,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=False,
        wave_lock_all_wheel_targets=True,
        wave_wheel_action_signs=(1.0, -1.0, 1.0, -1.0),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.zeros(1, dtype=torch.long)),
    )
    wrapper._wave_elapsed_s = None
    wrapper._wave_obstacle_active = None
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.zeros((1, 16)))

    torch.testing.assert_close(prepared[0, 12:16], torch.tensor([0.50, -0.50, 0.50, -0.50]))


def test_wheel_velocity_feedback_is_normalized_to_robot_forward_direction() -> None:
    import torch

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import _wheel_forward_velocity

    normalized = _wheel_forward_velocity(
        torch.tensor([[0.30, 0.30, -0.30, -0.30]]),
        (1.0, 1.0, -1.0, -1.0),
    )

    torch.testing.assert_close(normalized, torch.full((1, 4), 0.30))


def test_m1_wrapper_clamps_final_wave_reference_leg_actions() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.10,
        wave_semantic_obstacle_gating=False,
        wave_reference_actions=True,
        wave_reference_raw_amplitude=0.50,
        wave_reference_knee_ratio=1.0,
        wave_reference_frequency=12.5,
        wave_reset_phase_on_obstacle=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=False,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.40,
        wave_rear_wheel_action=0.40,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=False,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.ones(1, dtype=torch.long)),
    )
    wrapper._wave_elapsed_s = None
    wrapper._wave_obstacle_active = None
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.zeros((1, 16)))

    assert float(prepared[:, :12].abs().max()) <= 0.100001


def test_m1_wrapper_can_lock_policy_leg_residuals_while_keeping_wheels_active() -> None:
    import torch
    from types import SimpleNamespace

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import M1RslRlEnvWrapper

    wrapper = object.__new__(M1RslRlEnvWrapper)
    wrapper.clip_actions = 1.0
    cfg = SimpleNamespace(
        wave_fixed_forward_wheels=True,
        wave_leg_action_limit=0.10,
        wave_policy_leg_residual_limit=0.0,
        wave_semantic_obstacle_gating=False,
        wave_reference_actions=False,
        wave_left_right_symmetric=False,
        wave_lock_abduction=False,
        wave_wheel_residual_scale=0.0,
        wave_front_wheel_action=0.50,
        wave_rear_wheel_action=0.50,
        wave_obstacle_wheel_boost=0.0,
        wave_sync_actual_wheel_velocity=False,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    wrapper.env = SimpleNamespace(
        scene={},
        unwrapped=SimpleNamespace(cfg=cfg, episode_length_buf=torch.zeros(1, dtype=torch.long)),
    )
    wrapper._wave_elapsed_s = None
    wrapper._wave_obstacle_active = None
    wrapper._wheel_joint_ids = None
    wrapper._wheel_sync_integral = None

    prepared = wrapper._prepare_actions(torch.ones((1, 16)))

    torch.testing.assert_close(prepared[:, :12], torch.zeros((1, 12)))
    torch.testing.assert_close(prepared[:, 12:16], 0.50 * torch.ones((1, 4)))


def test_wave_gate_can_be_disabled_after_crossing_finish_line() -> None:
    import torch

    from go2_pvcnn.tasks.m1_rsl_rl_wrapper import _mask_wave_gate_after_root_x

    gate = torch.tensor([True, True, True])
    root_pos_w = torch.tensor([[0.50, 0.0, 0.0], [0.71, 0.0, 0.0], [1.20, 0.0, 0.0]])
    env_origins = torch.tensor([[0.00, 0.0, 0.0], [0.00, 0.0, 0.0], [0.60, 0.0, 0.0]])

    masked = _mask_wave_gate_after_root_x(gate, root_pos_w, env_origins, disable_after_x=0.70)

    assert masked.tolist() == [True, False, True]


def test_m1_pvcnn_small_obstacle_train_and_play_tasks_are_registered() -> None:
    source = REGISTER_FILE.read_text()

    assert "M1PvcnnFlatSmallAvoidanceEnvCfg" in source
    assert "M1PvcnnFlatSmallAvoidanceEnvCfg_PLAY" in source
    assert 'id="Isaac-M1-Pvcnn-Flat-Small-Avoidance-v0"' in source
    assert 'id="Isaac-M1-Pvcnn-Flat-Small-Avoidance-Play-v0"' in source
    assert 'id="Isaac-M1-Pvcnn-Crossing-60mm-v0"' in source
    assert 'id="Isaac-M1-Pvcnn-Crossing-100mm-v0"' in source


def test_semantic_body_queries_support_m1_link_names() -> None:
    source = SEMANTIC_CLEARANCE_FILE.read_text()

    assert '".*_foot", ".*_FOOT_LINK"' in source
    assert '".*_calf", ".*_KNEE_LINK"' in source
    assert '".*_thigh", ".*_HIP_LINK"' in source
