from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "m1_smoke_env_cfg.py"
REGISTER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "register_envs.py"
M1_REGISTER_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "register_m1_envs.py"
TASKS_INIT_FILE = REPO_ROOT / "go2_pvcnn" / "tasks" / "__init__.py"


def test_m1_smoke_cfg_declares_no_mpc_16_action_contract():
    source = TASK_FILE.read_text()

    assert "class M1SmokeEnvCfg(ManagerBasedRLEnvCfg)" in source
    assert "M1_CFG.replace" in source
    assert "M1_LEG_JOINT_NAMES" in source
    assert "M1_WHEEL_JOINT_NAMES" in source
    assert "leg_pos = mdp.JointPositionActionCfg" in source
    assert "joint_names=list(M1_LEG_JOINT_NAMES)" in source
    assert "wheel_vel = mdp.JointVelocityActionCfg" in source
    assert "joint_names=list(M1_WHEEL_JOINT_NAMES)" in source
    assert "body_names=list(M1_FOOT_BODY_NAMES)" in source
    assert "body_names=M1_BASE_BODY_NAME" in source
    assert "control_mode: str = M1_ROLLING_MODE" in source
    assert "available_control_modes: tuple[str, str] = (M1_ROLLING_MODE, M1_WAVE_MODE)" in source
    assert "rolling_wheel_velocity: float = 0.5" in source
    assert "wave_wheel_velocity: float = 1.5" in source
    assert "wave_amplitude: float = 0.0" in source
    assert "wave_frequency: float = 1.0" in source
    assert "wave_phase_offsets: tuple[float, float, float, float]" in source
    assert 'planner_backend: str = "none"' in source
    assert "use_batched_reference_trajectory: bool = False" in source
    assert "planner_owned_reference_cache: bool = False" in source
    assert "terrain_type=\"plane\"" in source


def test_m1_smoke_gym_id_is_registered():
    source = M1_REGISTER_FILE.read_text()

    assert "M1SmokeEnvCfg" in source
    assert "M1WalkEnvCfg" in source
    assert 'id="Isaac-M1-Smoke-v0"' in source
    assert 'id="Isaac-M1-Walk-v0"' in source
    assert '"env_cfg_entry_point": "go2_pvcnn.tasks.m1_smoke_env_cfg:M1SmokeEnvCfg"' in source
    assert '"env_cfg_entry_point": "go2_pvcnn.tasks.m1_walk_env_cfg:M1WalkEnvCfg"' in source


def test_m1_task_package_imports_m1_registration_by_default():
    source = TASKS_INIT_FILE.read_text()
    legacy_source = REGISTER_FILE.read_text()

    assert "register_m1_envs" in source
    assert "register_envs" not in source
    assert "Isaac-M1-Smoke-v0" not in legacy_source
    assert "Isaac-M1-Walk-v0" not in legacy_source
