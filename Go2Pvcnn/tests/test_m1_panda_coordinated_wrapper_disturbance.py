from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py"


def _load_wrapper(monkeypatch):
    import go2_pvcnn
    import sys
    import types

    mdp = types.ModuleType("go2_pvcnn.mdp")
    mdp.coordinated_desired_twist_b = lambda env: torch.zeros(env.num_envs, 6)
    mdp.coordinated_base_target_error_b = lambda env: torch.full(
        (env.num_envs, 3), 10.0
    )
    monkeypatch.setitem(sys.modules, "go2_pvcnn.mdp", mdp)
    monkeypatch.setattr(go2_pvcnn, "mdp", mdp, raising=False)
    spec = importlib.util.spec_from_file_location("coordinated_wrapper_under_test", WRAPPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.M1PandaCoordinatedEnvWrapper


class FakeRobot:
    def __init__(self, call_log: list[str]) -> None:
        self.call_log = call_log
        self.external_force_calls = []
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(2, 2, 1)
        self.data = SimpleNamespace(
            body_quat_w=identity,
            root_state_w=torch.tensor(
                [
                    [0.01, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.02, 0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            default_root_state=torch.tensor(
                [[0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ).repeat(2, 1),
            joint_pos=torch.tensor([[0.01, 0.0], [0.0, 0.02]]),
            default_joint_pos=torch.zeros(2, 2),
            joint_vel=torch.tensor([[0.03, 0.0], [0.0, 0.04]]),
            default_joint_vel=torch.zeros(2, 2),
        )

    def find_bodies(self, name, preserve_order=False):
        mapping = {"BASE_LINK": 0, "panda_hand": 1}
        return [mapping[name]], [name]

    def set_external_force_and_torque(self, force, torque, *, body_ids):
        self.call_log.append("external_force")
        self.external_force_calls.append((force.clone(), torque.clone(), list(body_ids)))


class FakeObservationManager:
    def __init__(self, env) -> None:
        self.env = env

    def compute(self):
        return {"policy": torch.zeros(self.env.num_envs, 103)}


class FakeScene(dict):
    def __init__(self, robot) -> None:
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(2, 3)


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = "cpu"
        self.max_episode_length = 100
        self.action_manager = SimpleNamespace(total_action_dim=23)
        self.action_space = SimpleNamespace(dtype="float32")
        self.observation_space = SimpleNamespace()
        self.call_log: list[str] = []
        self.robot = FakeRobot(self.call_log)
        self.scene = FakeScene(self.robot)
        self.cfg = SimpleNamespace(
            sim=SimpleNamespace(dt=0.005),
            decimation=1,
            mission_wheel_damping_nm_per_rad_s=30.0,
            mission_wheel_radius_m=0.095,
            mission_wheel_action_scale_nm=50.0,
            mission_arrival_position_tolerance_m=0.08,
            mission_arrival_yaw_tolerance_rad=0.10,
        )
        self.unwrapped = self
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.observation_manager = FakeObservationManager(self)

    def reset(self):
        return self.observation_manager.compute(), {}

    def step(self, actions):
        self.call_log.append("env_step")
        log = {
            "Episode_Termination/time_out": 0,
            "Episode_Termination/base_contact": 1,
            "Episode_Termination/bad_orientation": 0,
            "Episode_Reward/base_target": torch.tensor(2.0),
            "Episode_Reward/ee_tracking": torch.tensor(1.0),
        }
        return (
            self.observation_manager.compute(),
            torch.ones(2),
            torch.tensor([False, True]),
            torch.tensor([False, False]),
            {"log": log},
        )


def test_enabled_wrapper_applies_hand_wrench_before_step_and_resets_done(
    monkeypatch,
) -> None:
    wrapper_cls = _load_wrapper(monkeypatch)
    env = FakeEnv()
    wrapper = wrapper_cls(env, training_randomization=True, seed=7)

    wrapper.step(torch.zeros(2, 23))

    assert env.call_log[:2] == ["external_force", "env_step"]
    force, torque, body_ids = env.robot.external_force_calls[0]
    assert force.shape == torque.shape == (2, 1, 3)
    assert body_ids == [1]
    assert torch.equal(wrapper.current_wrench_b[1], torch.zeros(6))
    assert not torch.equal(wrapper.current_wrench_b[0], torch.zeros(6))


def test_wrapper_expands_reset_aggregates_to_completed_episode_metrics(
    monkeypatch,
) -> None:
    wrapper_cls = _load_wrapper(monkeypatch)
    wrapper = wrapper_cls(FakeEnv(), training_randomization=True, seed=7)

    _, _, dones, extras = wrapper.step(torch.zeros(2, 23))

    assert int(dones.sum()) == 1
    assert torch.equal(
        extras["log"]["Termination/base_contact"], torch.ones(1)
    )
    assert torch.equal(extras["log"]["Reward/base_target"], torch.full((1,), 2.0))


def test_default_wrapper_does_not_create_or_advance_disturbance(monkeypatch) -> None:
    wrapper_cls = _load_wrapper(monkeypatch)
    env = FakeEnv()
    wrapper = wrapper_cls(env)

    wrapper.step(torch.zeros(2, 23))

    assert wrapper._disturbance is None
    assert env.robot.external_force_calls == []
    assert env.call_log == ["env_step"]


def test_nonfinite_wrench_fails_before_robot_call(monkeypatch) -> None:
    wrapper_cls = _load_wrapper(monkeypatch)
    env = FakeEnv()
    wrapper = wrapper_cls(env, training_randomization=True, seed=3)

    with pytest.raises(RuntimeError, match="finite"):
        wrapper._apply_training_wrench(torch.full((2, 6), float("nan")))
    assert env.robot.external_force_calls == []


def test_training_diagnostics_are_finite_python_floats(monkeypatch) -> None:
    wrapper_cls = _load_wrapper(monkeypatch)
    wrapper = wrapper_cls(FakeEnv(), training_randomization=True, seed=5)
    wrapper.step(torch.zeros(2, 23))

    diagnostics = wrapper.get_training_diagnostics()

    assert {
        "curriculum_scale",
        "force_norm_max",
        "torque_norm_max",
        "nonzero_wrench_ratio",
        "root_pose_deviation_max",
        "root_velocity_deviation_max",
        "joint_position_deviation_max",
        "joint_velocity_deviation_max",
    } <= diagnostics.keys()
    assert all(type(value) is float for value in diagnostics.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())


def test_constructor_does_not_label_pre_reset_physics_state_as_reset_dr(
    monkeypatch,
) -> None:
    wrapper_cls = _load_wrapper(monkeypatch)
    wrapper = wrapper_cls(FakeEnv(), training_randomization=True, seed=5)

    diagnostics = wrapper.get_training_diagnostics()

    assert diagnostics["root_pose_deviation_max"] == 0.0
    assert diagnostics["root_velocity_deviation_max"] == 0.0
    assert diagnostics["joint_position_deviation_max"] == 0.0
    assert diagnostics["joint_velocity_deviation_max"] == 0.0
