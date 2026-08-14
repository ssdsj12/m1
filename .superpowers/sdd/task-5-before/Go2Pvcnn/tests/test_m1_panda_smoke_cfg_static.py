import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET_FILE = ROOT / "go2_pvcnn/assets/m1_panda.py"
ENV_FILE = ROOT / "go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py"
REGISTRY_FILE = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"

EXPECTED_REGISTRATIONS = [
    ("Isaac-M1-Pvcnn-Crossing-60mm-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Play-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmPlayEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Unlocked-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmUnlockedEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Guided-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmGuidedEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Guided-Fixed-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmGuidedFixedEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Distilled-Play-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmDistilledPlayEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Train-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmContactFreeTrainEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Play-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmContactFreePlayEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Pair-Curriculum-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmPairCurriculumEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-60mm-Policy-Play-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing60mmPolicyPlayEnvCfg"),
    ("Isaac-M1-Pvcnn-Crossing-100mm-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnCrossing100mmEnvCfg"),
    ("Isaac-M1-Pvcnn-Flat-Small-Avoidance-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnFlatSmallAvoidanceEnvCfg"),
    ("Isaac-M1-Pvcnn-Flat-Small-Avoidance-Play-v0", "go2_pvcnn.tasks.m1_pvcnn_small_obstacle_env_cfg:M1PvcnnFlatSmallAvoidanceEnvCfg_PLAY"),
    ("Isaac-M1-Smoke-v0", "go2_pvcnn.tasks.m1_smoke_env_cfg:M1SmokeEnvCfg"),
    ("Isaac-M1-Panda-Smoke-v0", "go2_pvcnn.tasks.m1_panda_smoke_env_cfg:M1PandaSmokeEnvCfg"),
    ("Isaac-M1-Small-Obstacle-5mm-v0", "go2_pvcnn.tasks.m1_small_obstacle_env_cfg:M1SmallObstacle5mmEnvCfg"),
    ("Isaac-M1-Small-Obstacle-10mm-v0", "go2_pvcnn.tasks.m1_small_obstacle_env_cfg:M1SmallObstacle10mmEnvCfg"),
    ("Isaac-M1-Walk-v0", "go2_pvcnn.tasks.m1_walk_env_cfg:M1WalkEnvCfg"),
    ("Isaac-M1-Roll-v0", "go2_pvcnn.tasks.m1_roll_env_cfg:M1RollEnvCfg"),
    ("Isaac-M1-Wave-Flat-v0", "go2_pvcnn.tasks.m1_wave_env_cfg:M1WaveFlatEnvCfg"),
    ("Isaac-M1-Small-Obstacle-v0", "go2_pvcnn.tasks.m1_small_obstacle_env_cfg:M1SmallObstacleEnvCfg"),
]


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignments(body):
    return [node for node in body if isinstance(node, (ast.Assign, ast.AnnAssign))]


def _assignment_name(node):
    target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _assignment_value(node):
    return node.value


def _keywords(call):
    return {keyword.arg: keyword.value for keyword in call.keywords}


def _update_dict(tree, receiver):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "update" and ast.unparse(node.func.value) == receiver:
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Dict)
            return node.args[0]
    raise AssertionError(f"missing {receiver}.update(...)")


def _assert_action_contract(source):
    tree = ast.parse(source)
    actions = _class(tree, "M1PandaSmokeActionsCfg")
    terms = [
        node
        for node in _assignments(actions.body)
        if isinstance(_assignment_value(node), ast.Call)
    ]
    assert [_assignment_name(node) for node in terms] == ["leg_pos", "wheel_vel"]

    expected = {
        "leg_pos": ("mdp.JointPositionActionCfg", "list(M1_LEG_JOINT_NAMES)", 0.25, True, {".*": (-100.0, 100.0)}),
        "wheel_vel": ("mdp.JointVelocityActionCfg", "list(M1_WHEEL_JOINT_NAMES)", 8.0, True, {".*": (-8.0, 8.0)}),
    }
    for node in terms:
        name = _assignment_name(node)
        call = _assignment_value(node)
        keywords = _keywords(call)
        assert ast.unparse(call.func) == expected[name][0]
        assert ast.literal_eval(keywords["asset_name"]) == "robot"
        assert ast.unparse(keywords["joint_names"]) == expected[name][1]
        assert ast.literal_eval(keywords["scale"]) == expected[name][2]
        assert ast.literal_eval(keywords["use_default_offset"]) is expected[name][3]
        assert ast.literal_eval(keywords["clip"]) == expected[name][4]


def _assert_registry_contract(source):
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("go2_pvcnn.tasks")
        for node in tree.body
    )
    registrations = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "gym.register"
    ]
    assert len(registrations) == len(EXPECTED_REGISTRATIONS)
    for call, (expected_id, expected_cfg) in zip(registrations, EXPECTED_REGISTRATIONS, strict=True):
        assert [keyword.arg for keyword in call.keywords] == [
            "id",
            "entry_point",
            "disable_env_checker",
            "kwargs",
        ]
        keywords = _keywords(call)
        assert ast.literal_eval(keywords["id"]) == expected_id
        assert ast.literal_eval(keywords["entry_point"]) == "isaaclab.envs:ManagerBasedRLEnv"
        assert ast.literal_eval(keywords["disable_env_checker"]) is True
        kwargs = ast.literal_eval(keywords["kwargs"])
        assert list(kwargs) == ["env_cfg_entry_point", "rsl_rl_cfg_entry_point"]
        assert kwargs["env_cfg_entry_point"] == expected_cfg
        assert kwargs["rsl_rl_cfg_entry_point"] is None


def test_combined_cfg_has_exact_home_pose_and_hold_actuators():
    tree = ast.parse(ASSET_FILE.read_text())
    simple = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"M1_PANDA_BASE_BODY_NAME", "M1_PANDA_MOUNT_BODY_NAME", "M1_PANDA_DOF_COUNT"}
    }
    assert simple == {
        "M1_PANDA_BASE_BODY_NAME": "BASE_LINK",
        "M1_PANDA_MOUNT_BODY_NAME": "panda_link0",
        "M1_PANDA_DOF_COUNT": 25,
    }

    home = ast.literal_eval(_update_dict(tree, "M1_PANDA_CFG.init_state.joint_pos"))
    assert home == {
        "panda_joint1": 0.0,
        "panda_joint2": -0.569,
        "panda_joint3": 0.0,
        "panda_joint4": -2.810,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }

    actuator_dict = _update_dict(tree, "M1_PANDA_CFG.actuators")
    actuator_names = [ast.literal_eval(key) for key in actuator_dict.keys]
    assert actuator_names == ["panda_shoulder", "panda_forearm", "panda_hand"]
    expected = {
        "panda_shoulder": (["panda_joint[1-4]"], 87.0, 2.175, 80.0, 4.0),
        "panda_forearm": (["panda_joint[5-7]"], 12.0, 2.61, 80.0, 4.0),
        "panda_hand": (["panda_finger_joint.*"], 200.0, 0.2, 2000.0, 100.0),
    }
    for name, call in zip(actuator_names, actuator_dict.values, strict=True):
        assert ast.unparse(call.func) == "ImplicitActuatorCfg"
        keywords = _keywords(call)
        actual = tuple(
            ast.literal_eval(keywords[key])
            for key in ("joint_names_expr", "effort_limit", "velocity_limit", "stiffness", "damping")
        )
        assert actual == expected[name]


def test_action_terms_have_exact_order_types_and_parameters():
    _assert_action_contract(ENV_FILE.read_text())


def test_action_contract_rejects_an_extra_panda_term():
    mutated = ENV_FILE.read_text().replace(
        "    wheel_vel = mdp.JointVelocityActionCfg(",
        "    panda_arm = mdp.JointPositionActionCfg(asset_name=\"robot\")\n"
        "    wheel_vel = mdp.JointVelocityActionCfg(",
        1,
    )
    with pytest.raises(AssertionError):
        _assert_action_contract(mutated)


def test_each_joint_observation_has_exactly_the_m1_selector():
    tree = ast.parse(ENV_FILE.read_text())
    observations = _class(tree, "M1PandaSmokeObservationsCfg")
    policy = next(node for node in observations.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    obs_terms = {
        _assignment_name(node): _assignment_value(node)
        for node in _assignments(policy.body)
        if isinstance(_assignment_value(node), ast.Call) and ast.unparse(_assignment_value(node).func) == "ObsTerm"
    }
    joint_terms = {
        name: call
        for name, call in obs_terms.items()
        if ast.unparse(_keywords(call)["func"]) in {"isaac_mdp.joint_pos_rel", "isaac_mdp.joint_vel_rel"}
    }
    assert list(joint_terms) == ["joint_pos", "joint_vel"]
    assert {ast.unparse(_keywords(call)["func"]) for call in joint_terms.values()} == {
        "isaac_mdp.joint_pos_rel",
        "isaac_mdp.joint_vel_rel",
    }
    for call in joint_terms.values():
        params = _keywords(call)["params"]
        assert isinstance(params, ast.Dict) and len(params.keys) == 1
        assert ast.literal_eval(params.keys[0]) == "asset_cfg"
        selector = params.values[0]
        assert ast.unparse(selector.func) == "SceneEntityCfg"
        assert ast.literal_eval(selector.args[0]) == "robot"
        assert ast.unparse(_keywords(selector)["joint_names"]) == "list(M1_JOINT_NAMES)"

    all_joint_functions = [
        ast.unparse(_keywords(call)["func"])
        for call in obs_terms.values()
        if "joint" in ast.unparse(_keywords(call)["func"])
    ]
    assert all_joint_functions == ["isaac_mdp.joint_pos_rel", "isaac_mdp.joint_vel_rel"]
    assert ast.unparse(_keywords(obs_terms["actions"])["func"]) == "isaac_mdp.last_action"


def test_combined_cfg_copies_before_mutation_and_keeps_task4_scope():
    asset = ASSET_FILE.read_text()
    env = ENV_FILE.read_text()
    assert asset.index("M1_PANDA_CFG = M1_CFG.copy()") < asset.index("M1_PANDA_CFG.spawn")
    assert "usd_path=M1_PANDA_USD_PATH" in asset
    assert Path(ROOT / "assets/m1_panda/m1_panda.usd").is_file()
    assert "wrench" not in env.lower()


def test_registry_is_lazy_and_preserves_all_m1_gym_ids():
    _assert_registry_contract(REGISTRY_FILE.read_text())


@pytest.mark.parametrize(
    "old,new",
    [
        ("go2_pvcnn.tasks.m1_smoke_env_cfg:M1SmokeEnvCfg", "go2_pvcnn.tasks.wrong:WrongCfg"),
        ("isaaclab.envs:ManagerBasedRLEnv", "wrong.module:WrongEnv"),
        ("disable_env_checker=True", "disable_env_checker=False"),
        ('"rsl_rl_cfg_entry_point": None}', '"rsl_rl_cfg_entry_point": None, "extra": 1}'),
        ('"rsl_rl_cfg_entry_point": None}', '"rsl_rl_cfg_entry_point": "wrong.module:Cfg"}'),
    ],
    ids=["old-target", "manager", "checker", "extra-kwarg", "non-none-rsl"],
)
def test_registry_contract_rejects_each_mapping_drift(old, new):
    source = REGISTRY_FILE.read_text()
    assert old in source
    mutated = source.replace(old, new, 1)
    with pytest.raises(AssertionError):
        _assert_registry_contract(mutated)
