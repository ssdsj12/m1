from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WAVE_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_wave_env_cfg.py"
WRAPPER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_rsl_rl_wrapper.py"
REGISTER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "register_m1_envs.py"


def test_wave_flat_preserves_roll_contract_and_releases_bounded_legs():
    source = WAVE_FILE.read_text()

    assert "class M1WaveFlatEnvCfg(M1RollEnvCfg)" in source
    assert "roll_equal_wheel_actions: bool = False" in source
    assert "wave_fixed_forward_wheels: bool = True" in source
    assert "wave_leg_action_limit: float = 0.10" in source
    assert "wave_reference_actions: bool = True" in source
    assert "wave_reference_raw_amplitude: float = 0.10" in source
    assert "wave_left_right_symmetric: bool = True" in source
    assert "wave_lock_abduction: bool = True" in source
    assert "wave_front_wheel_action: float = 0.40" in source
    assert "wave_rear_wheel_action: float = 0.40" in source
    assert "self.episode_length_s = 30.0" in source
    assert "class M1WaveFlatRewardsCfg(M1RollRewardsCfg)" in source
    assert "forward_velocity = RewTerm" in source
    assert "lin_vel_x = (0.03, 0.05)" in source
    assert "termination_penalty = RewTerm" in source
    assert 'params["limit_angle"] = 0.35' in source


def test_wrapper_keeps_wave_wheels_equal_and_negative():
    source = WRAPPER_FILE.read_text()

    assert "wave_fixed_forward_wheels" in source
    assert "wave_leg_action_limit" in source
    assert "prepared[:, :12]" in source
    assert "prepared[:, 12:16] = wheel_action.expand(-1, 4)" in source
    assert "wave_front_wheel_action" in source
    assert "wave_rear_wheel_action" in source
    assert "wave_obstacle_wheel_boost" in source
    assert "build_wave_reference_actions" in source
    assert "wave_left_right_symmetric" in source
    assert "wave_lock_abduction" in source
    assert "leg_actions[:, (0, 3, 6, 9)] = 0.0" in source
    assert "symmetric_pairs" in source


def test_wave_flat_task_is_registered():
    source = REGISTER_FILE.read_text()

    assert "M1WaveFlatEnvCfg" in source
    assert 'id="Isaac-M1-Wave-Flat-v0"' in source
