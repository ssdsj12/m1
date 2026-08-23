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
        "folded_arm",
        "ee_tracking",
        "alive",
        "termination_penalty",
        "base_height",
        "base_linear_velocity",
        "base_angular_velocity",
        "flat_orientation_l2",
        "feet_slide",
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
    clamp = source.index("actions = torch.clamp(actions, -1.0, 1.0)")
    step = source.index("self.env.step(actions)")
    assert clamp < step
