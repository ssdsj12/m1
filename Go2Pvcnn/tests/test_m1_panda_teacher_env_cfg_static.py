from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
REWARD_FILE = ROOT / "go2_pvcnn/mdp/m1_panda_teacher_rewards.py"
ENV_FILE = ROOT / "go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py"
SMOKE_ENV_FILE = ROOT / "go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py"
MDP_INIT_FILE = ROOT / "go2_pvcnn/mdp/__init__.py"


def _load_rewards():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_teacher_rewards_under_test", REWARD_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rewards = _load_rewards()


class _Scene(dict):
    def __init__(self, robot, env_origins):
        super().__init__(robot=robot)
        self.env_origins = env_origins


def _fake_env():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor(
                [[3.0, 4.0, 0.6], [0.0, -2.0, 0.6]], dtype=torch.float32
            ),
            joint_vel=torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                dtype=torch.float32,
            ),
            applied_torque=torch.tensor(
                [[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]],
                dtype=torch.float32,
            ),
        )
    )
    env = SimpleNamespace(
        num_envs=2,
        scene=_Scene(
            robot,
            torch.tensor([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]),
        ),
        m1_teacher_trainable_residual=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        m1_teacher_previous_trainable_residual=torch.tensor(
            [[0.5, 1.0], [2.0, 2.0]], dtype=torch.float32
        ),
    )
    return env


def _asset_cfg(joint_ids=slice(None)):
    return SimpleNamespace(name="robot", joint_ids=joint_ids)


def test_base_xy_drift_is_relative_to_each_environment_origin():
    env = _fake_env()

    result = rewards.base_xy_drift_l2(env, _asset_cfg())

    assert torch.equal(result, torch.tensor([13.0, 2.0]))


def test_selected_joint_velocity_and_torque_use_only_requested_ids():
    env = _fake_env()
    selector = _asset_cfg([1, 3])

    velocity = rewards.selected_joint_velocity_l2(env, selector)
    torque = rewards.selected_joint_torques_l2(env, selector)

    assert torch.equal(velocity, torch.tensor([20.0, 10.0]))
    assert torch.equal(torque, torch.tensor([5.0, 2.5]))


def test_teacher_residual_terms_consume_wrapper_published_state():
    env = _fake_env()

    amplitude = rewards.teacher_residual_l2(env)
    rate = rewards.teacher_residual_rate_l2(env)

    assert torch.equal(amplitude, torch.tensor([5.0, 25.0]))
    assert torch.equal(rate, torch.tensor([1.25, 5.0]))


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("m1_teacher_trainable_residual", None, "trainable residual"),
        (
            "m1_teacher_trainable_residual",
            torch.zeros(1, 2),
            "shape",
        ),
        (
            "m1_teacher_trainable_residual",
            torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]),
            "finite",
        ),
    ],
)
def test_teacher_residual_terms_reject_missing_or_invalid_state(
    attribute, value, message
):
    env = _fake_env()
    if value is None:
        delattr(env, attribute)
    else:
        setattr(env, attribute, value)

    with pytest.raises(RuntimeError, match=message):
        rewards.teacher_residual_l2(env)
    with pytest.raises(RuntimeError, match=message):
        rewards.teacher_residual_rate_l2(env)


def test_teacher_residual_rate_rejects_previous_shape_mismatch():
    env = _fake_env()
    env.m1_teacher_previous_trainable_residual = torch.zeros(2, 3)

    with pytest.raises(RuntimeError, match="same shape"):
        rewards.teacher_residual_rate_l2(env)


def _class(tree, name):
    return next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _class_assignments(node):
    result = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                result[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            result[statement.target.id] = statement.value
    return result


def _keywords(call):
    return {keyword.arg: keyword.value for keyword in call.keywords}


def test_teacher_cfg_has_shared_a0_a1_inheritance_and_exact_stage_fields():
    tree = ast.parse(ENV_FILE.read_text())
    base = _class(tree, "M1PandaTeacherBaseEnvCfg")
    a0 = _class(tree, "M1PandaTeacherA0EnvCfg")
    a1 = _class(tree, "M1PandaTeacherA1EnvCfg")

    assert [ast.unparse(item) for item in base.bases] == ["M1PandaSmokeEnvCfg"]
    assert [ast.unparse(item) for item in a0.bases] == ["M1PandaTeacherBaseEnvCfg"]
    assert [ast.unparse(item) for item in a1.bases] == ["M1PandaTeacherBaseEnvCfg"]

    a0_fields = {
        name: ast.literal_eval(value)
        for name, value in _class_assignments(a0).items()
        if name.startswith("teacher_")
    }
    a1_fields = {
        name: ast.literal_eval(value)
        for name, value in _class_assignments(a1).items()
        if name.startswith("teacher_")
    }
    assert a0_fields == {
        "teacher_stage": "A0",
        "teacher_force_limit_n": (10.0, 10.0, 10.0),
        "teacher_torque_limit_nm": (2.0, 2.0, 2.0),
        "teacher_hold_time_s": (1.0, 2.0),
        "teacher_curriculum_start_scale": 0.25,
        "teacher_curriculum_steps": 50_000,
        "teacher_mode_probabilities": (1.0, 0.0, 0.0),
        "teacher_pulse_on_fraction": 0.20,
    }
    assert a1_fields == {
        "teacher_stage": "A1",
        "teacher_force_limit_n": (20.0, 20.0, 20.0),
        "teacher_torque_limit_nm": (5.0, 5.0, 5.0),
        "teacher_hold_time_s": (0.25, 1.0),
        "teacher_curriculum_start_scale": 0.25,
        "teacher_curriculum_steps": 75_000,
        "teacher_mode_probabilities": (0.50, 0.30, 0.20),
        "teacher_pulse_on_fraction": 0.20,
    }


def test_teacher_cfg_inherits_exact_60_observation_and_16_action_contract():
    source = ENV_FILE.read_text()
    smoke = SMOKE_ENV_FILE.read_text()

    assert "M1PandaSmokeEnvCfg" in source
    assert "observations:" not in source
    assert "actions:" not in source
    assert "class M1PandaSmokeObservationsCfg" in smoke
    for term in (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "mount_wrench_b",
    ):
        assert f"        {term} = ObsTerm" in smoke
    assert "class M1PandaSmokeActionsCfg" in smoke
    assert "    leg_pos = mdp.JointPositionActionCfg" in smoke
    assert "    wheel_vel = mdp.JointVelocityActionCfg" in smoke
    assert "Panda joints stay outside policy observations" in smoke


def test_teacher_reward_terms_have_exact_functions_weights_and_selectors():
    tree = ast.parse(ENV_FILE.read_text())
    reward_class = _class(tree, "M1PandaTeacherRewardsCfg")
    terms = _class_assignments(reward_class)
    assert list(terms) == [
        "alive",
        "base_height",
        "base_linear_velocity",
        "base_angular_velocity",
        "flat_orientation_l2",
        "base_xy_drift",
        "wheel_speed",
        "residual",
        "residual_rate",
        "joint_torques",
        "feet_slide",
    ]

    expected = {
        "alive": ("isaac_mdp.is_alive", 2.0),
        "base_height": ("mdp.base_height_l2", -12.0),
        "base_linear_velocity": ("mdp.lin_vel_z_l2", -2.0),
        "base_angular_velocity": ("mdp.ang_vel_xy_l2", -0.15),
        "flat_orientation_l2": ("mdp.flat_orientation_l2", -8.0),
        "base_xy_drift": ("mdp.base_xy_drift_l2", -1.0),
        "wheel_speed": ("mdp.selected_joint_velocity_l2", -0.01),
        "residual": ("mdp.teacher_residual_l2", -0.02),
        "residual_rate": ("mdp.teacher_residual_rate_l2", -0.01),
        "joint_torques": ("mdp.selected_joint_torques_l2", -5.0e-5),
        "feet_slide": ("mdp.feet_slide", -0.20),
    }
    for name, call in terms.items():
        assert ast.unparse(call.func) == "RewTerm"
        keywords = _keywords(call)
        assert ast.unparse(keywords["func"]) == expected[name][0]
        assert ast.literal_eval(keywords["weight"]) == expected[name][1]

    height_params = _keywords(terms["base_height"])["params"]
    assert ast.literal_eval(height_params) == {"target_height": 0.60}
    drift_params = ast.unparse(_keywords(terms["base_xy_drift"])["params"])
    wheel_params = ast.unparse(_keywords(terms["wheel_speed"])["params"])
    torque_params = ast.unparse(_keywords(terms["joint_torques"])["params"])
    feet_params = ast.unparse(_keywords(terms["feet_slide"])["params"])
    assert drift_params == "{'asset_cfg': SceneEntityCfg('robot')}"
    assert "joint_names=list(M1_WHEEL_JOINT_NAMES)" in wheel_params
    assert "joint_names=list(M1_JOINT_NAMES)" in torque_params
    assert feet_params.count("body_names=list(M1_FOOT_BODY_NAMES)") == 2


def test_teacher_cfg_keeps_smoke_termination_and_fixed_panda_boundaries():
    source = ENV_FILE.read_text()
    smoke = SMOKE_ENV_FILE.read_text()

    assert "terminations:" not in source
    assert "time_out = DoneTerm" in (ROOT / "go2_pvcnn/tasks/m1_smoke_env_cfg.py").read_text()
    assert "base_contact = DoneTerm" in (ROOT / "go2_pvcnn/tasks/m1_smoke_env_cfg.py").read_text()
    assert "bad_orientation = DoneTerm" in (ROOT / "go2_pvcnn/tasks/m1_smoke_env_cfg.py").read_text()
    assert "panda" not in "\n".join(
        line for line in smoke.splitlines() if "ActionCfg" in line or "action" in line.lower()
    ).lower().replace("m1panda", "")


def test_teacher_reward_helpers_are_exported_from_mdp_namespace():
    source = MDP_INIT_FILE.read_text()
    assert "from .m1_panda_teacher_rewards import *" in source
