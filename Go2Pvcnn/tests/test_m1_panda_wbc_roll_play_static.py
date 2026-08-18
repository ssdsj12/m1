import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py"
REGISTRY = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"


def test_c1a_env_inherits_c0_effort_contract_and_has_runtime_margin():
    source = ENV_CFG.read_text()

    assert (
        "class M1PandaWbcRollTeacherEnvCfg(M1PandaWbcTeacherEnvCfg)"
        in source
    )
    assert "self.decimation = 1" in source
    assert "self.episode_length_s = 30.0" in source
    assert "self.sim.dt = 0.005" in source


def test_c1a_has_one_independent_gym_registration():
    source = REGISTRY.read_text()
    tree = ast.parse(source)
    registrations = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "gym.register"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if "id" in keywords and ast.literal_eval(keywords["id"]) == (
            "Isaac-M1-Panda-Wbc-Teacher-C1a-v0"
        ):
            registrations.append(keywords)

    assert len(registrations) == 1
    kwargs = ast.literal_eval(registrations[0]["kwargs"])
    assert kwargs["env_cfg_entry_point"] == (
        "go2_pvcnn.tasks.m1_panda_wbc_roll_teacher_env_cfg:"
        "M1PandaWbcRollTeacherEnvCfg"
    )
    assert kwargs["rsl_rl_cfg_entry_point"] is None
