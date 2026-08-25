from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py"
REGISTRY = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"
MDP_INIT = ROOT / "go2_pvcnn/mdp/__init__.py"
ASSET = ROOT / "go2_pvcnn/assets/m1_panda.py"


def test_folded_load_task_has_isolated_id_and_103_by_23_200hz_contract():
    source = CFG.read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    assert 'id="Isaac-M1-Panda-Folded-Load-v0"' in registry
    assert "m1_panda_folded_load_env_cfg:M1PandaFoldedLoadEnvCfg" in registry
    assert "FOLDED_LOAD_POLICY_OBSERVATION_DIM = 103" in source
    assert "combined_action_dim: int = 23" in source
    assert "self.decimation = 1" in source
    assert "self.sim.dt = 0.005" in source
    assert "self.scene.robot = M1_PANDA_FOLDED_LOAD_CFG.replace" in source


def test_folded_load_pd_override_is_task_local_and_preserves_global_asset():
    source = CFG.read_text(encoding="utf-8")
    asset = ASSET.read_text(encoding="utf-8")
    assert "M1_PANDA_FOLDED_LOAD_CFG = M1_PANDA_CFG.copy()" in source
    assert '"panda_shoulder": M1_PANDA_CFG.actuators["panda_shoulder"].replace(' in source
    assert "stiffness=120.0" in source
    assert "damping=8.0" in source
    assert "self.scene.robot = M1_PANDA_FOLDED_LOAD_CFG.replace" in source
    assert "stiffness=80.0" in asset
    assert "damping=4.0" in asset


def test_observation_order_and_widths_preserve_checkpoint_boundary():
    source = CFG.read_text(encoding="utf-8")
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
    assert "FOLDED_LOAD_POLICY_OBSERVATION_WIDTHS" in source
    assert "func=mdp.folded_load_desired_twist_b" in source


def test_actions_preserve_12_leg_4_wheel_7_arm_order_and_arm_pd_payload():
    source = CFG.read_text(encoding="utf-8")
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
    asset = ASSET.read_text(encoding="utf-8")
    for value in ("-0.569", "-2.810", "3.037", "0.741"):
        assert value in asset
    assert '"panda_finger_joint.*": 0.04' in asset
    assert "effort_limit=87.0" in asset
    assert "effort_limit=12.0" in asset
    assert "stiffness=80.0" in asset
    assert "damping=4.0" in asset


def test_reward_set_and_weights_are_exact_and_exclude_old_mission_objectives():
    source = CFG.read_text(encoding="utf-8")
    rewards = source[source.index("class M1PandaFoldedLoadRewardsCfg") : source.index("class M1PandaFoldedLoadEventsCfg")]
    expected = {
        "track_vx": "2.0",
        "track_wz": "1.0",
        "lateral_velocity": "-0.5",
        "alive": "1.0",
        "base_height": "-12.0",
        "base_linear_velocity": "-1.0",
        "base_angular_velocity": "-0.1",
        "flat_orientation_l2": "-2.0",
        "feet_slide": "-0.1",
        "active_action_l2": "-0.02",
        "active_action_rate": "-0.01",
        "joint_torques": "-1.0e-5",
        "termination_penalty": "-10000.0",
    }
    for name, weight in expected.items():
        start = rewards.index(f"    {name} = RewTerm(")
        block = rewards[start : rewards.index("    )", start) + 5]
        assert f"weight={weight}" in block
    for removed in ("base_target =", "ee_tracking =", "folded_arm ="):
        assert removed not in rewards
    assert 'params={"target_height": 0.6115}' in rewards


def test_default_events_are_deterministic_and_have_no_external_wrench_event():
    source = CFG.read_text(encoding="utf-8")
    assert "M1PandaFoldedLoadEventsCfg" in source
    assert '"leg_position_range": (0.0, 0.0)' in source
    assert '"arm_position_range"' not in source
    assert '"static_friction_range": (1.0, 1.0)' in source
    assert "apply_external_force_torque" not in source


def test_mdp_exports_folded_load_terms_without_changing_legacy_exports():
    source = MDP_INIT.read_text(encoding="utf-8")
    assert "from .m1_panda_folded_load import *" in source
    assert "from .m1_panda_coordinated import *" in source
