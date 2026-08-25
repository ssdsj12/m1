from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch

ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
WRAPPER = Path(__file__).resolve().parents[1] / "go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py"


@pytest.fixture
def wrapper_cls(monkeypatch):
    import go2_pvcnn

    mdp = types.ModuleType("go2_pvcnn.mdp")
    mdp.m1_panda_mount_wrench_b = lambda env, asset_cfg: torch.zeros(env.num_envs, 6)
    monkeypatch.setitem(sys.modules, "go2_pvcnn.mdp", mdp)
    monkeypatch.setattr(go2_pvcnn, "mdp", mdp, raising=False)
    spec = importlib.util.spec_from_file_location("folded_load_wrapper_under_test", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.M1PandaFoldedLoadEnvWrapper


class _ObservationManager:
    def __init__(self, env):
        self.env = env

    def compute(self):
        return {"policy": torch.zeros(self.env.num_envs, 103)}


class _TerminationManager:
    def __init__(self, env):
        self.env = env

    def get_term(self, name):
        return self.env.termination_terms[name]


class _Robot:
    def __init__(self, num_envs):
        joint_pos = torch.zeros(num_envs, 25)
        joint_pos[:, 16:23] = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        self.device = "cpu"
        self.joint_names = [f"m1_{i}" for i in range(16)] + list(ARM_NAMES) + ["finger1", "finger2"]
        self.data = SimpleNamespace(
            root_lin_vel_b=torch.zeros(num_envs, 3),
            root_ang_vel_b=torch.zeros(num_envs, 3),
            joint_pos=joint_pos,
            default_joint_pos=joint_pos.clone(),
            applied_torque=torch.zeros(num_envs, 25),
            soft_joint_pos_limits=torch.stack(
                (torch.full_like(joint_pos, -4.0), torch.full_like(joint_pos, 4.0)), dim=-1
            ),
        )

    def find_joints(self, names, preserve_order=False):
        return [self.joint_names.index(name) for name in names], list(names)


class _Env:
    def __init__(self):
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 100
        self.action_manager = SimpleNamespace(total_action_dim=23)
        self.action_space = SimpleNamespace()
        self.observation_space = SimpleNamespace()
        self.robot = _Robot(self.num_envs)
        self.scene = {"robot": self.robot}
        self.unwrapped = self
        self.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.005), decimation=1)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.observation_manager = _ObservationManager(self)
        self.termination_manager = _TerminationManager(self)
        self.termination_terms = {
            "base_contact": torch.zeros(self.num_envs, dtype=torch.bool),
            "bad_orientation": torch.zeros(self.num_envs, dtype=torch.bool),
        }
        self.last_action = None
        self.next_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.next_truncated = torch.zeros(self.num_envs, dtype=torch.bool)

    def reset(self):
        return self.observation_manager.compute(), {}

    def step(self, actions):
        self.last_action = actions.clone()
        return (
            self.observation_manager.compute(),
            torch.ones(self.num_envs),
            self.next_terminated.clone(),
            self.next_truncated.clone(),
            {"log": {}},
        )


def test_wrapper_zeroes_arm_before_environment_step_and_never_needs_wrench(wrapper_cls):
    env = _Env()
    wrapper = wrapper_cls(env, stage="L0-C0", seed=3)
    actions = torch.ones(env.num_envs, 23)

    wrapper.step(actions)

    assert env.last_action[:, :16].eq(1.0).all()
    assert env.last_action[:, 16:].eq(0.0).all()
    assert actions.eq(1.0).all()
    assert wrapper.get_training_diagnostics()["inactive_action_max"] == 0.0


def test_wrapper_rejects_bad_or_nonfinite_actions_before_step(wrapper_cls):
    wrapper = wrapper_cls(_Env(), stage="L0-C0", seed=3)
    with pytest.raises(ValueError, match=r"\(num_envs, 23\)"):
        wrapper.step(torch.zeros(4, 22))
    bad = torch.zeros(4, 23); bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        wrapper.step(bad)


def test_done_resamples_only_selected_commands_and_reports_exact_buckets(wrapper_cls):
    env = _Env()
    wrapper = wrapper_cls(env, stage="L1-C2", seed=9)
    before = wrapper.commands.clone()
    env.next_terminated[1] = True
    env.next_truncated[3] = True
    env.termination_terms["base_contact"][1] = True

    _, _, dones, extras = wrapper.step(torch.zeros(4, 23))

    assert dones.tolist() == [False, True, False, True]
    torch.testing.assert_close(wrapper.commands[[0, 2]], before[[0, 2]])
    assert not torch.equal(wrapper.commands[[1, 3]], before[[1, 3]])
    metrics = extras["episode_metrics"]
    assert metrics["env_id"].tolist() == [1, 3]
    assert metrics["base_contact"].tolist() == [True, False]
    assert metrics["time_out"].tolist() == [False, True]
    assert metrics["command"].shape == (2, 3)
    assert set(metrics["bucket"].keys()) == {"stationary", "forward", "reverse", "left", "right"}


def test_reset_samples_commands_before_observation_and_diagnostics_are_finite(wrapper_cls):
    env = _Env()
    wrapper = wrapper_cls(env, stage="L2-D3", seed=4)
    obs, extras = wrapper.reset()
    assert obs.shape == (4, 103)
    assert extras["observations"]["critic"].shape == (4, 103)
    assert env.folded_load_commands.data_ptr() == wrapper.commands.data_ptr()
    diagnostics = wrapper.get_training_diagnostics()
    assert diagnostics["fold_error_max"] == pytest.approx(0.0)
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())


def test_completed_records_are_lossless_and_drained_once(wrapper_cls):
    env = _Env()
    wrapper = wrapper_cls(env, stage="L1-C2", seed=9)
    wrapper.set_evaluation_commands(
        torch.tensor(
            [
                [0.10, 0.0, 0.0],
                [-0.10, 0.0, 0.0],
                [0.0, 0.0, 0.30],
                [0.0, 0.0, -0.30],
            ]
        )
    )
    env.robot.data.root_lin_vel_b[1, 0] = -0.08
    env.next_truncated[1] = True

    wrapper.step(torch.zeros(4, 23))

    records = wrapper.drain_completed_episode_records()
    assert len(records) == 1
    assert records[0].env_id == 1
    assert records[0].command == pytest.approx((-0.10, 0.0, 0.0))
    assert records[0].steps == 1
    assert records[0].time_out is True
    assert records[0].vx_error_sq_sum == pytest.approx(0.02**2)
    assert wrapper.drain_completed_episode_records() == ()


def test_fixed_evaluation_commands_require_exact_finite_shape(wrapper_cls):
    wrapper = wrapper_cls(_Env(), stage="L0-C0", seed=3)
    commands = torch.zeros(4, 3)
    commands[:, 0] = torch.tensor((0.01, -0.01, 0.0, 0.0))
    wrapper.set_evaluation_commands(commands)
    torch.testing.assert_close(wrapper.commands, commands)

    with pytest.raises(ValueError, match=r"\(num_envs, 3\)"):
        wrapper.set_evaluation_commands(torch.zeros(3, 3))
    commands[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        wrapper.set_evaluation_commands(commands)
