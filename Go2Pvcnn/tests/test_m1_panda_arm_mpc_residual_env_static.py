from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "go2_pvcnn/tasks/m1_panda_arm_mpc_residual_env_cfg.py"
REGISTER = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"
TASK_ID = "Isaac-M1-Panda-ArmMpc-Residual-v0"


def test_new_gym_id_uses_dedicated_stationary_cfg_without_changing_old_id():
    source = REGISTER.read_text()
    assert f'id="{TASK_ID}"' in source
    block = source.split(f'id="{TASK_ID}"', 1)[1].split("gym.register(", 1)[0]
    assert "m1_panda_arm_mpc_residual_env_cfg:M1PandaArmMpcResidualEnvCfg" in block
    old = source.split('id="Isaac-M1-Panda-Residual-Wbc-v0"', 1)[1].split("gym.register(", 1)[0]
    assert "m1_panda_wbc_roll_teacher_env_cfg:M1PandaWbcRollTeacherEnvCfg" in old


def test_dedicated_cfg_freezes_200_hz_private_effort_bridge_and_no_disturbance():
    source = CFG.read_text()
    assert "class M1PandaArmMpcResidualEnvCfg(M1PandaWbcTeacherEnvCfg)" in source
    assert "self.sim.dt = 0.005" in source
    assert "self.decimation = 1" in source
    assert "private_action_dim = 23" in source
    assert "public_action_dim = 8" in source
    assert "observation_dim = 103" in source
    assert "external_wrench" not in source
    assert "domain_random" not in source
    assert "payload" not in source.lower()
