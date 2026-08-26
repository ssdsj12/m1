import gymnasium as gym
import pytest
import torch

from go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper import (
    M1PandaArmMpcResidualEnvWrapper,
    M1PandaArmMpcResidualRuntime,
)


class _ActionManager:
    total_action_dim = 23


class _FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.max_episode_length = 100
        self.action_manager = _ActionManager()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,))
        self.last_effort = None
        self.unwrapped = self
        self.cfg = object()
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)

    def reset(self):
        return {"policy": torch.zeros(2, 1)}, {}

    def step(self, effort):
        self.last_effort = effort.clone()
        terminated = torch.tensor([False, True])
        truncated = torch.zeros(2, dtype=torch.bool)
        return {"policy": torch.zeros(2, 1)}, torch.full((2,), -99.0), terminated, truncated, {"log": {}}


class _FakeRuntime:
    def __init__(self):
        self.num_envs = 2
        self.steps = []
        self.reset_ids = []

    def reset(self, env_ids=None):
        self.reset_ids.append(None if env_ids is None else env_ids.clone())

    def observations(self):
        return torch.zeros(2, 103)

    def compute(self, actions, physics_step):
        self.steps.append((physics_step, actions.clone(), physics_step % 4 == 0))
        effort = torch.full((2, 23), float(physics_step + 1))
        reward = torch.tensor([1.0, 2.0])
        return effort, reward, {"mpc_replanned": float(physics_step % 4 == 0)}


def test_wrapper_exposes_only_8_actions_and_exact_103_observations():
    env = _FakeEnv()
    runtime = _FakeRuntime()
    wrapper = M1PandaArmMpcResidualEnvWrapper(env, runtime=runtime)

    observations, extras = wrapper.get_observations()

    assert wrapper.num_actions == 8
    assert wrapper.action_space.shape == (8,)
    assert env.action_manager.total_action_dim == 23
    assert observations.shape == (2, 103)
    assert torch.equal(extras["observations"]["critic"], observations)


def test_wrapper_runs_four_step_mpc_cadence_and_writes_private_23d_effort():
    env = _FakeEnv()
    runtime = _FakeRuntime()
    wrapper = M1PandaArmMpcResidualEnvWrapper(env, runtime=runtime)
    wrapper.reset()

    for _ in range(5):
        obs, reward, dones, extras = wrapper.step(torch.zeros(2, 8))

    assert [entry[2] for entry in runtime.steps] == [True, False, False, False, True]
    assert env.last_effort.shape == (2, 23)
    assert reward.tolist() == [1.0, 2.0]
    assert obs.shape == (2, 103)
    assert dones.tolist() == [False, True]
    assert extras["environment_metrics"]["mpc_replanned"] == 1.0
    assert torch.equal(runtime.reset_ids[-1], torch.tensor([1]))


def test_wrapper_rejects_wrong_or_nonfinite_public_action_atomically():
    env = _FakeEnv()
    runtime = _FakeRuntime()
    wrapper = M1PandaArmMpcResidualEnvWrapper(env, runtime=runtime)

    with pytest.raises(ValueError, match="shape"):
        wrapper.step(torch.zeros(2, 23))
    actions = torch.zeros(2, 8)
    actions[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        wrapper.step(actions)
    assert runtime.steps == []


class _PhysicalEnv:
    num_envs = 1
    device = "cpu"


class _PhysicalAdapter:
    def __init__(self):
        self.state = _physical_state()

    def build_state(self, physics_step):
        self.state.physics_step = physics_step
        self.state.time_s = physics_step * 0.005
        return self.state, 0

    def build_arm_mpc_input(self, state, target_pose, target_twist):
        return (state, target_pose, target_twist)

    def read_mount_wrench_b(self):
        return torch.zeros(6, dtype=torch.float64)

    def leg_soft_limits(self):
        return torch.tensor([[-1.0, 1.0]] * 12, dtype=torch.float64)


class _PhysicalPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, sample):
        self.calls += 1
        target = sample[1]
        diagnostics = type("Diagnostics", (), {"feasible": True, "fallback_used": False})()
        return type(
            "Solution",
            (),
            {
                "q_ref": torch.zeros(7, dtype=torch.float64),
                "qd_ref": torch.zeros(7, dtype=torch.float64),
                "predicted_dynamic_mount_wrench_b": torch.zeros(6, dtype=torch.float64),
                "predicted_pose_b": target,
                "diagnostics": diagnostics,
            },
        )()


class _PhysicalTeacher:
    def reset(self, state, *, seed):
        self.seed = seed

    def step(self, state, **kwargs):
        qp = type("Qp", (), {"success": True})()
        safety = type("Safety", (), {"name": "TRACK"})()
        return type(
            "Command",
            (),
            {
                "effort": torch.zeros(23, dtype=torch.float64),
                "qp_result": qp,
                "safety_state": safety,
                "terminate": False,
            },
        )()


def _physical_state():
    wbc = type("Wbc", (), {"effort_limit": torch.ones(23, dtype=torch.float64)})()
    state = type("State", (), {})()
    state.physics_step = 0
    state.time_s = 0.0
    state.ee_pose = torch.zeros(6, dtype=torch.float64)
    state.coord_q = torch.zeros(10, dtype=torch.float64)
    state.coord_qd = torch.zeros(10, dtype=torch.float64)
    state.coord_q_min = -torch.ones(10, dtype=torch.float64)
    state.coord_q_max = torch.ones(10, dtype=torch.float64)
    state.sigma_min = torch.tensor(1.0, dtype=torch.float64)
    state.controlled_q = torch.zeros(23, dtype=torch.float64)
    state.controlled_qd = torch.zeros(23, dtype=torch.float64)
    state.roll = 0.0
    state.pitch = 0.0
    state.wheel_contact_count = 4
    state.max_lateral_slip = 0.0
    state.signals_finite = True
    state.wbc_input = wbc
    return state


def test_physical_runtime_replans_every_four_steps_and_builds_exact_public_contract():
    adapters = []
    planners = []

    def adapter_factory(env, env_id):
        adapter = _PhysicalAdapter()
        adapters.append(adapter)
        return adapter

    def planner_factory():
        planner = _PhysicalPlanner()
        planners.append(planner)
        return planner

    runtime = M1PandaArmMpcResidualRuntime(
        _PhysicalEnv(),
        seed=42,
        trajectory_scale=0.1,
        adapter_factory=adapter_factory,
        teacher_factory=lambda state, env_id: _PhysicalTeacher(),
        planner_factory=planner_factory,
    )
    runtime.reset()

    for step in range(5):
        effort, reward, metrics = runtime.compute(torch.zeros(1, 8), step)
        runtime.refresh(step + 1)

    assert planners[0].calls == 2
    assert effort.shape == (1, 23)
    assert reward.shape == (1,)
    assert runtime.observations().shape == (1, 103)
    assert metrics["mpc_feasible_rate"] == 1.0
    snapshot = runtime.diagnostics_snapshot()
    assert snapshot["measured_mount_wrench_b"].shape == (1, 6)
    assert snapshot["predicted_mount_wrench_b"].shape == (1, 6)
    assert snapshot["target_pose"].shape == (1, 6)
    assert snapshot["effort"].shape == (1, 23)
    assert snapshot["mpc_fallback"].tolist() == [False]
