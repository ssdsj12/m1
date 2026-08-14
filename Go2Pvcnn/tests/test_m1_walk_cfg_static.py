from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WALK_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_walk_env_cfg.py"
REGISTER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "register_m1_envs.py"


def test_m1_walk_cfg_declares_trainable_velocity_tracking_task():
    source = WALK_FILE.read_text()

    assert "class M1WalkEnvCfg(M1SmokeEnvCfg)" in source
    assert "class M1WalkCommandsCfg" in source
    assert "base_velocity = mdp.UniformLevelVelocityCommandCfg" in source
    assert "rel_standing_envs=0.5" in source
    assert "lin_vel_x=(0.0, 0.08)" in source
    assert "velocity_commands = ObsTerm" in source
    assert "track_lin_vel_xy = RewTerm" in source
    assert "track_ang_vel_z = RewTerm" in source
    assert "base_height = RewTerm" in source
    assert '"target_height": 0.60' in source
    assert "front_rear_joint_speed = RewTerm" in source
    assert "front_rear_action = RewTerm" in source
    assert "paired_joint_speed_mismatch" in source
    assert "paired_action_mismatch" in source
    assert "weight=-0.20" in source
    assert "feet_air_time = RewTerm" in source
    assert "air_time_variance = RewTerm" in source
    assert "wheel_vel = mdp.JointVelocityActionCfg" in source
    assert "scale=0.0" in source
    assert "clip={\".*\": (0.0, 0.0)}" in source
    assert "self.terminations.base_contact = None" in source


def test_m1_walk_gym_id_is_registered():
    source = REGISTER_FILE.read_text()

    assert "M1WalkEnvCfg" in source
    assert 'id="Isaac-M1-Walk-v0"' in source
    assert '"env_cfg_entry_point": "go2_pvcnn.tasks.m1_walk_env_cfg:M1WalkEnvCfg"' in source
