import gymnasium as gym
import pytest
import torch
from types import SimpleNamespace

import go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper as residual_wrapper
from go2_pvcnn.tasks.m1_panda_arm_mpc_residual_wrapper import (
    M1PandaArmMpcResidualEnvWrapper,
    M1PandaArmMpcResidualRuntime,
)


class _ActionManager:
    total_action_dim = 23


class _FakeEnv:
    def __init__(self, events=None):
        self.num_envs = 2
        self.device = "cpu"
        self.max_episode_length = 100
        self.action_manager = _ActionManager()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,))
        self.last_effort = None
        self.unwrapped = self
        self.cfg = object()
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.events = [] if events is None else events

    def reset(self):
        return {"policy": torch.zeros(2, 1)}, {}

    def step(self, effort):
        self.events.append("env.step")
        self.last_effort = effort.clone()
        terminated = torch.tensor([False, True])
        truncated = torch.zeros(2, dtype=torch.bool)
        return {"policy": torch.zeros(2, 1)}, torch.full((2,), -99.0), terminated, truncated, {"log": {}}


class _FakeRuntime:
    def __init__(self, events=None):
        self.num_envs = 2
        self.steps = []
        self.reset_ids = []
        self.events = [] if events is None else events
        self._reward = torch.tensor([1.0, 2.0])
        self._metrics = {}

    def reset(self, env_ids=None):
        self.reset_ids.append(None if env_ids is None else env_ids.clone())
        if env_ids is not None:
            ids = torch.as_tensor(env_ids).reshape(-1).tolist()
            self.events.append(f"reset:{ids}")

    def observations(self):
        return torch.zeros(2, 103)

    def compute_action(self, actions, physics_step):
        self.events.append(f"compute_action:{physics_step}")
        self.steps.append((physics_step, actions.clone(), physics_step % 4 == 0))
        effort = torch.full((2, 23), float(physics_step + 1))
        self._metrics = {"mpc_replanned": float(physics_step % 4 == 0)}
        return effort

    def refresh(self, physics_step):
        self.events.append(f"refresh:{physics_step}")

    def compute_transition_reward(self):
        self.events.append("compute_transition_reward")
        return self._reward.clone(), dict(self._metrics)

    def get_training_diagnostics(self):
        return {"samples": 10.0}


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


def test_wrapper_finalizes_reward_after_physics_and_before_done_reset():
    events = []
    env = _FakeEnv(events=events)
    runtime = _FakeRuntime(events=events)
    wrapper = M1PandaArmMpcResidualEnvWrapper(env, runtime=runtime)
    wrapper.reset()
    events.clear()

    _, reward, dones, _ = wrapper.step(torch.zeros(2, 8))

    assert events == [
        "compute_action:0",
        "env.step",
        "refresh:1",
        "compute_transition_reward",
        "reset:[1]",
    ]
    assert reward.tolist() == [1.0, 2.0]
    assert dones.tolist() == [False, True]


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


def test_zero_stage_forces_exact_zero_residual_before_runtime():
    env = _FakeEnv()
    runtime = _FakeRuntime()
    wrapper = M1PandaArmMpcResidualEnvWrapper(
        env, runtime=runtime, force_zero_residual=True
    )
    wrapper.reset()

    wrapper.step(torch.ones((2, 8)))

    assert torch.equal(runtime.steps[-1][1], torch.zeros((2, 8)))


def test_wrapper_exposes_runtime_training_diagnostics():
    wrapper = M1PandaArmMpcResidualEnvWrapper(_FakeEnv(), runtime=_FakeRuntime())

    assert wrapper.get_training_diagnostics() == {"samples": 10.0}


class _PhysicalEnv:
    num_envs = 1
    device = "cpu"

    def __init__(self, *, dt=0.0025, decimation=2):
        self.cfg = SimpleNamespace(
            sim=SimpleNamespace(dt=dt),
            decimation=decimation,
        )


class _PhysicalAdapter:
    def __init__(self):
        self.state = _physical_state()
        self.rebase_count = 0
        self.pose = torch.tensor(
            [0.25, -0.1, 0.5, 0.0, 0.0, 0.0], dtype=torch.float64
        )
        self.mount_wrench = torch.zeros(6, dtype=torch.float64)

    def rebase_reference(self):
        self.rebase_count += 1

    def build_state(self, physics_step):
        self.state.physics_step = physics_step
        self.state.time_s = physics_step * 0.005
        return self.state, 0

    def build_arm_mpc_input(self, state, target_pose, target_twist):
        self.last_target_twist = target_twist.clone()
        return type(
            "Input",
            (),
            {
                "state": state,
                "target_pose_b": target_pose,
                "target_twist_b": target_twist,
                "qd": torch.zeros(7, dtype=torch.float64),
                "qd_max": torch.full((7,), 2.5, dtype=torch.float64),
                "base_arm_coupling": torch.eye(6, 7, dtype=torch.float64),
            },
        )()

    def arm_mpc_kinematics_b(self, state):
        del state
        jacobian = torch.zeros((6, 7), dtype=torch.float64)
        return self.pose.clone(), jacobian

    def read_mount_wrench_b(self):
        return self.mount_wrench.clone()

    def leg_soft_limits(self):
        return torch.tensor([[-1.0, 1.0]] * 12, dtype=torch.float64)


class _PhysicalPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, sample):
        self.calls += 1
        target = sample.target_pose_b
        diagnostics = type("Diagnostics", (), {"feasible": True, "fallback_used": False})()
        return type(
            "Solution",
            (),
            {
                "q_ref": torch.full((7,), 0.0002, dtype=torch.float64),
                "qd_ref": torch.full((7,), 0.02, dtype=torch.float64),
                "qdd": torch.ones((20, 7), dtype=torch.float64),
                "predicted_dynamic_mount_wrench_b": torch.full(
                    (6,), 9.0, dtype=torch.float64
                ),
                "predicted_pose_b": target,
                "diagnostics": diagnostics,
            },
        )()


class _PhysicalTeacher:
    def __init__(self):
        self.arm_references = []
        self.cfg = type(
            "Cfg", (), {"arm_position_gain": 20.0, "arm_velocity_gain": 5.0}
        )()

    def reset(self, state, *, seed):
        self.seed = seed

    def step(self, state, **kwargs):
        self.last_arm_reference = kwargs["arm_reference"]
        self.arm_references.append(self.last_arm_reference)
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
    wbc = type(
        "Wbc",
        (),
        {
            "effort_limit": torch.ones(23, dtype=torch.float64),
            "bias_force": torch.zeros(31, dtype=torch.float64),
        },
    )()
    state = type("State", (), {})()
    state.physics_step = 0
    state.time_s = 0.0
    state.ee_pose = torch.zeros(6, dtype=torch.float64)
    state.coordinated_jacobian = torch.zeros((6, 10), dtype=torch.float64)
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


def _make_physical_runtime(*, dt=0.0025, decimation=2):
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
        _PhysicalEnv(dt=dt, decimation=decimation),
        seed=42,
        trajectory_scale=0.1,
        adapter_factory=adapter_factory,
        teacher_factory=lambda state, env_id: _PhysicalTeacher(),
        planner_factory=planner_factory,
    )
    return runtime, adapters, planners


def test_runtime_resolves_exact_200_hz_control_dt():
    runtime, _, _ = _make_physical_runtime()

    assert runtime.control_dt == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("dt", "decimation"),
    ((0.0, 2), (float("nan"), 2), (0.0025, 0)),
)
def test_runtime_rejects_invalid_control_interval(dt, decimation):
    with pytest.raises(ValueError, match="control interval"):
        _make_physical_runtime(dt=dt, decimation=decimation)


def test_runtime_scales_only_ppo_reward_by_control_dt(monkeypatch):
    reward_density = torch.tensor([7.0], dtype=torch.float64)
    monkeypatch.setattr(
        residual_wrapper,
        "compute_residual_reward",
        lambda signals: SimpleNamespace(total=reward_density.clone()),
    )
    runtime, adapters, _ = _make_physical_runtime()
    runtime.reset()
    runtime.compute_action(torch.zeros(1, 8), physics_step=0)
    adapters[0].mount_wrench[0] = 30.0
    corrected_measurement = runtime.controller.preview_corrected_mount_wrench_b(
        adapters[0].mount_wrench[None]
    )
    expected_wrench_error = torch.linalg.vector_norm(
        corrected_measurement - runtime._pending_transition.predicted_wrench_b,
        dim=1,
    ).item()
    runtime.refresh(1)

    reward, _ = runtime.compute_transition_reward()
    diagnostics = runtime.get_training_diagnostics()

    torch.testing.assert_close(reward, reward_density * 0.005)
    assert diagnostics["wrench_error"] == pytest.approx(expected_wrench_error)
    assert diagnostics["wrench_error"] < 30.0


def test_runtime_requires_matching_post_step_refresh_before_transition_reward():
    runtime, _, _ = _make_physical_runtime()
    runtime.reset()

    effort = runtime.compute_action(torch.zeros(1, 8), physics_step=0)

    assert effort.shape == (1, 23)
    with pytest.raises(RuntimeError, match="post-step refresh"):
        runtime.compute_transition_reward()
    with pytest.raises(RuntimeError, match="pending transition"):
        runtime.compute_action(torch.zeros(1, 8), physics_step=0)


def test_runtime_advances_residual_history_only_after_reward_finalization():
    runtime, _, _ = _make_physical_runtime()
    runtime.reset()
    action = torch.full((1, 8), 0.25)

    runtime.compute_action(action, physics_step=0)

    assert torch.equal(
        runtime._previous_normalized,
        torch.zeros((1, 8), dtype=torch.float64),
    )
    runtime.refresh(1)
    reward, metrics = runtime.compute_transition_reward()
    torch.testing.assert_close(
        runtime._previous_normalized, action.to(dtype=torch.float64)
    )
    assert reward.shape == (1,)
    assert metrics["mpc_feasible_rate"] == 1.0


def test_runtime_reward_uses_post_transition_tilt_ee_pose_and_wrench():
    nominal, nominal_adapters, _ = _make_physical_runtime()
    nominal.reset()
    nominal.compute_action(torch.zeros(1, 8), physics_step=0)
    nominal.refresh(1)
    nominal_reward, _ = nominal.compute_transition_reward()

    changed, changed_adapters, _ = _make_physical_runtime()
    changed.reset()
    changed.compute_action(torch.zeros(1, 8), physics_step=0)
    changed_adapters[0].state.roll = 0.15
    changed_adapters[0].pose[0] += 0.03
    changed_adapters[0].mount_wrench[0] = 30.0
    changed.refresh(1)
    changed_reward, _ = changed.compute_transition_reward()

    assert changed_reward.item() < nominal_reward.item()
    torch.testing.assert_close(
        changed._last_measured,
        torch.tensor([[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    )


def test_runtime_rejects_reset_until_pending_transition_is_finalized():
    runtime, _, _ = _make_physical_runtime()
    runtime.reset()
    runtime.compute_action(torch.zeros(1, 8), physics_step=0)

    with pytest.raises(RuntimeError, match="pending transition"):
        runtime.reset()


def test_physical_runtime_replans_every_four_steps_and_builds_exact_public_contract():
    runtime, adapters, planners = _make_physical_runtime()
    runtime.reset()

    assert adapters[0].rebase_count == 1
    torch.testing.assert_close(
        runtime._centers[0],
        torch.tensor([0.25, -0.1, 0.5, 0.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert runtime.controller._feedback.cfg.bias_warmup_samples == 64
    torch.testing.assert_close(
        runtime.controller.wrench_scale,
        torch.tensor(
            [30.0, 30.0, 50.0, 15.0, 15.0, 8.0], dtype=torch.float64
        ),
    )
    assert runtime._rne_feedback.cfg.bias_warmup_samples == 1
    assert runtime._rne_feedback.cfg.filter_alpha == 0.5
    assert runtime._rne_sensor_delay_steps == 3

    for step in range(5):
        effort = runtime.compute_action(torch.zeros(1, 8), step)
        runtime.refresh(step + 1)
        reward, metrics = runtime.compute_transition_reward()

    assert planners[0].calls == 2
    references = runtime.teachers[0].arm_references[:4]
    for reference in references:
        assert torch.equal(
            reference.q_ref, torch.full((7,), 0.0002, dtype=torch.float64)
        )
        assert torch.equal(
            reference.qd_ref, torch.full((7,), 0.1992, dtype=torch.float64)
        )
    assert torch.any(adapters[0].last_target_twist != 0.0)
    assert effort.shape == (1, 23)
    assert reward.shape == (1,)
    assert runtime.observations().shape == (1, 103)
    assert metrics["mpc_feasible_rate"] == 1.0
    snapshot = runtime.diagnostics_snapshot()
    assert snapshot["measured_mount_wrench_b"].shape == (1, 6)
    assert snapshot["dynamic_measured_mount_wrench_b"].shape == (1, 6)
    assert snapshot["predicted_mount_wrench_b"].shape == (1, 6)
    torch.testing.assert_close(
        snapshot["predicted_mount_wrench_b"],
        torch.full((1, 6), 0.009, dtype=torch.float64),
    )
    assert snapshot["planned_predicted_mount_wrench_b"].tolist() == [[9.0] * 6]
    torch.testing.assert_close(
        snapshot["controller_predicted_mount_wrench_b"],
        -torch.ones((1, 6), dtype=torch.float64),
    )
    assert snapshot["target_pose"].shape == (1, 6)
    assert snapshot["effort"].shape == (1, 23)
    assert snapshot["arm_q"].shape == (1, 7)
    assert snapshot["arm_q_ref"].shape == (1, 7)
    assert snapshot["arm_qd_ref"].shape == (1, 7)
    assert snapshot["arm_qdd_first"].shape == (1, 7)
    assert snapshot["actual_arm_qdd"].shape == (1, 7)
    assert snapshot["actual_dynamic_mount_wrench_b"].shape == (1, 6)
    assert snapshot["predicted_ee_pose_first"].shape == (1, 6)
    assert snapshot["predicted_ee_pose_terminal"].shape == (1, 6)
    assert snapshot["replan_start_ee_pose"].shape == (1, 6)
    assert snapshot["current_ee_pose"].shape == (1, 6)
    assert snapshot["arm_jacobian"].shape == (1, 6, 7)
    assert snapshot["root_xy"].shape == (1, 2)
    assert snapshot["initial_root_xy"].shape == (1, 2)
    assert snapshot["base_bias_wrench"].shape == (1, 6)
    assert snapshot["correction_wrench_b"].shape == (1, 6)
    assert snapshot["mpc_fallback"].tolist() == [False]
    training = runtime.get_training_diagnostics()
    required = {
        "hard_failure_count",
        "mpc_feasible_rate",
        "qp_feasible_rate",
        "four_contact_rate",
        "roll_pitch_rms",
        "base_height_rms",
        "ee_position_error",
        "ee_orientation_error",
        "wrench_error",
        "normalized_wrench_error",
        "slip",
        "intervention_ratio",
        *(f"saturation_fraction_{index}" for index in range(8)),
    }
    assert required <= set(training)
    assert training["mpc_feasible_rate"] == 1.0
    assert training["qp_feasible_rate"] == 1.0
    assert training["four_contact_rate"] == 1.0
