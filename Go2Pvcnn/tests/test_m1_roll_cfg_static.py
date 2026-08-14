from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROLL_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_roll_env_cfg.py"
REGISTER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "register_m1_envs.py"
REWARDS_FILE = REPO_ROOT / "go2_pvcnn" / "mdp" / "rewards.py"


def test_m1_roll_cfg_trains_wheels_and_locks_legs():
    source = ROLL_FILE.read_text()

    assert "class M1RollEnvCfg(M1SmokeEnvCfg)" in source
    assert "base_velocity = mdp.UniformLevelVelocityCommandCfg" in source
    assert "lin_vel_x=(0.02, 0.04)" in source
    assert "leg_pos = mdp.JointPositionActionCfg" in source
    assert "joint_names=list(M1_LEG_JOINT_NAMES)" in source
    assert "scale=0.05" in source
    assert "wheel_vel = mdp.JointVelocityActionCfg" in source
    assert "joint_names=list(M1_WHEEL_JOINT_NAMES)" in source
    assert "scale=1.0" in source
    assert "roll_equal_wheel_actions: bool = True" in source
    assert "track_lin_vel_xy = RewTerm" in source
    assert '"target_height": 0.55' in source
    assert "lateral_velocity = RewTerm" in source
    assert "wheel_action_match = RewTerm" in source
    assert "self.terminations.base_contact = None" in source


def test_m1_roll_gym_id_is_registered():
    source = REGISTER_FILE.read_text()

    assert "M1RollEnvCfg" in source
    assert 'id="Isaac-M1-Roll-v0"' in source
    assert "Isaac-M1-Roll-v0" in source


def test_lateral_velocity_reward_helper_exists():
    source = REWARDS_FILE.read_text()

    assert "def track_lin_vel_y_l2" in source
    assert "asset.data.root_lin_vel_b[:, 1]" in source
