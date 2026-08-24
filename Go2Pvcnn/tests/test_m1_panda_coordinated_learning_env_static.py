from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py"
WRAPPER = ROOT / "go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py"


def test_policy_observation_contract_is_ordered_and_exactly_103_wide():
    source = CFG.read_text()
    ordered_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "controlled_joint_pos",
        "controlled_joint_vel",
        "base_target_error_b",
        "ee_pose_error_b",
        "desired_twist_b",
        "wheel_contact",
        "mount_wrench_b",
        "previous_action",
    )
    positions = [source.index(f"        {term} = ObsTerm") for term in ordered_terms]
    assert positions == sorted(positions)
    assert "COORDINATED_POLICY_OBSERVATION_DIM = 103" in source
    assert "COORDINATED_POLICY_OBSERVATION_WIDTHS" in source
    assert "M1_PANDA_WBC_CONTROLLED_JOINT_NAMES" in source
    assert source.count("preserve_order=True") >= 2


def test_learning_rewards_cover_mission_balance_slip_rate_and_effort():
    source = CFG.read_text()
    for reward in (
        "base_target",
        "base_velocity_target",
        "folded_arm",
        "ee_tracking",
        "alive",
        "termination_penalty",
        "base_height",
        "base_linear_velocity",
        "base_angular_velocity",
        "flat_orientation_l2",
        "feet_slide",
        "action_l2",
        "action_rate",
        "joint_torques",
    ):
        assert f"    {reward} = RewTerm" in source
    assert "mission_arrival_position_tolerance_m" in source
    assert "mission_arrival_yaw_tolerance_rad" in source
    assert "mission_ee_target_offset_b" in source
    assert "func=isaac_mdp.is_terminated, weight=-10000.0" in source


def test_wrapper_clamps_the_23_actions_before_stepping_physics():
    source = WRAPPER.read_text()
    phase_mask = source.index("actions[arrived, 12:16] = 0.0")
    nominal = source.index("actions = actions + self._nominal_wheel_actions()")
    clamp = source.index("actions = torch.clamp(actions, -1.0, 1.0)")
    step = source.index("self.env.step(actions)")
    assert phase_mask < nominal < clamp < step
    assert "mission_arrival_position_tolerance_m" in source
    assert "mission_arrival_yaw_tolerance_rad" in source
    assert "mdp.coordinated_desired_twist_b(self.env.unwrapped)" in source
    assert "mission_wheel_radius_m" in source
    assert "mission_wheel_damping_nm_per_rad_s" in source
    assert "mission_wheel_action_scale_nm" in source


def test_residual_action_scales_match_leg_wheel_and_arm_authority():
    source = CFG.read_text()
    assert "class M1PandaCoordinatedActionsCfg" in source
    for term, names, scale in (
        ("leg_effort", "M1_LEG_JOINT_NAMES", "5.0"),
        ("wheel_effort", "M1_WHEEL_JOINT_NAMES", "50.0"),
        ("arm_effort", "PANDA_ARM_JOINT_NAMES", "2.0"),
    ):
        start = source.index(f"    {term} = isaac_mdp.JointEffortActionCfg(")
        block = source[start : source.index("    )", start) + 5]
        assert f"joint_names=list({names})" in block
        assert f"scale={scale}" in block
        assert "preserve_order=True" in block
    assert "mission_base_linear_speed_limit_mps: float = 0.10" in source


def test_domain_randomization_is_explicit_and_deterministic_by_default():
    source = CFG.read_text()
    assert "class M1PandaCoordinatedEventsCfg" in source
    assert "func=mdp.reset_coordinated_joints_by_offset" in source
    assert "configure_coordinated_training_domain_randomization" in source
    assert '"leg_position_range": (0.0, 0.0)' in source
    assert '"arm_position_range": (0.0, 0.0)' in source
