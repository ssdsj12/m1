import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_FILE = ROOT / "go2_pvcnn/assets/m1_panda.py"
ENV_FILE = ROOT / "go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py"
REGISTRY_FILE = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"
LEGACY_TEACHER_FILE = ROOT / "go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py"
LEGACY_PLAY_FILE = ROOT / "scripts/m1_panda_teacher_play.py"

CONTROLLED_JOINTS = (
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT",
    "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment(node, name):
    for statement in node.body:
        target = statement.target if isinstance(statement, ast.AnnAssign) else (
            statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1 else None
        )
        if isinstance(target, ast.Name) and target.id == name:
            return statement.value
    raise AssertionError(f"missing {node.name}.{name}")


def _keywords(call):
    return {keyword.arg: keyword.value for keyword in call.keywords}


def test_wbc_asset_is_an_isolated_copy_with_only_controlled_actuators_zeroed():
    source = ASSET_FILE.read_text()
    tree = ast.parse(source)
    assert "M1_PANDA_WBC_CFG = M1_PANDA_CFG.copy()" in source
    assert "M1_PANDA_WBC_CFG.init_state.pos = (0.0, 0.0, 0.6115)" in source
    assert "solver_position_iteration_count=16" in source
    assert "solver_velocity_iteration_count=4" in source
    assert source.index("M1_PANDA_CFG.actuators.update(") < source.index("M1_PANDA_WBC_CFG =")
    assert ast.literal_eval(
        next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "M1_PANDA_WBC_CONTROLLED_JOINT_NAMES"
        )
    ) == CONTROLLED_JOINTS

    for actuator in ("legs", "wheels", "panda_shoulder", "panda_forearm"):
        token = f'M1_PANDA_CFG.actuators["{actuator}"].replace(stiffness=0.0, damping=0.0)'
        assert token in source
    assert 'M1_PANDA_CFG.actuators["panda_hand"].copy()' in source
    assert '"panda_finger_joint.*": 0.04' in source
    assert "stiffness=2000.0" in source and "damping=100.0" in source


def test_wbc_environment_exposes_one_exact_ordered_effort_action_and_c0_timing():
    source = ENV_FILE.read_text()
    tree = ast.parse(source)
    actions = _class(tree, "M1PandaWbcTeacherActionsCfg")
    assignments = [
        statement for statement in actions.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
    ]
    assert len(assignments) == 1
    call = _assignment(actions, "joint_effort")
    assert ast.unparse(call.func) == "isaac_mdp.JointEffortActionCfg"
    keywords = _keywords(call)
    assert ast.literal_eval(keywords["asset_name"]) == "robot"
    assert ast.unparse(keywords["joint_names"]) == "list(M1_PANDA_WBC_CONTROLLED_JOINT_NAMES)"
    assert ast.literal_eval(keywords["scale"]) == 1.0
    assert ast.literal_eval(keywords["preserve_order"]) is True

    env = _class(tree, "M1PandaWbcTeacherEnvCfg")
    post = next(node for node in env.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    post_source = ast.unparse(post)
    for token in (
        "self.decimation = 1",
        "self.episode_length_s = 20.0",
        "self.sim.dt = 0.005",
        "self.sim.render_interval = 4",
        "self.sim.physx.enable_external_forces_every_iteration = True",
        "self.scene.contact_forces.update_period = self.sim.dt",
    ):
        assert token in post_source
    assert "rsl_rl" not in source.lower()


def test_wbc_registration_is_lazy_and_does_not_change_legacy_teacher_sources():
    source = REGISTRY_FILE.read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("go2_pvcnn.tasks")
        for node in tree.body
    )
    registration = next(
        node.value for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "gym.register"
        and ast.literal_eval(_keywords(node.value)["id"]) == "Isaac-M1-Panda-Wbc-Teacher-C0-v0"
    )
    kwargs = ast.literal_eval(_keywords(registration)["kwargs"])
    assert kwargs == {
        "env_cfg_entry_point": "go2_pvcnn.tasks.m1_panda_wbc_teacher_env_cfg:M1PandaWbcTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": None,
    }
    assert _sha256(LEGACY_TEACHER_FILE) == "b17f5861ffbee473850cc3be4e22cb0b3ea67e9e273c507d0115006d228e13e6"
    assert _sha256(LEGACY_PLAY_FILE) == "c12bfbd742411ead1448dd6abb0cc1b0fe01a16768c919d3b96b03697e12bdfc"
