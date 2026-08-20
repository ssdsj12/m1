from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py"
REG = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"
SCRIPT = ROOT / "scripts/m1_panda_coordinated_play.py"


def test_combined_config_preserves_asset_and_action_wrench_contracts():
    source = CFG.read_text()
    assert "class M1PandaCoordinatedEnvCfg" in source
    assert "M1PandaWbcRollTeacherEnvCfg" in source
    assert "M1_PANDA_CFG" in source
    assert "mission_target_base_pose" in source
    assert "mission_ee_target_pose" in source
    assert "mount_wrench_b" in source
    assert "23" in source


def test_combined_task_and_cli_are_registered():
    assert 'id="Isaac-M1-Panda-Coordinated-v0"' in REG.read_text()
    script = SCRIPT.read_text()
    for flag in ("--num_envs", "--max_steps", "--target-base-pose", "--ee-target-pose"):
        assert flag in script
    for key in ("phase_counts", "base_assist_count", "ee_error"):
        assert key in script
