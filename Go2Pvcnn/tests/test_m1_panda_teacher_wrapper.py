from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from go2_pvcnn.tasks.m1_panda_teacher_wrapper import M1PandaTeacherEnvWrapper


class _FakeRobot:
    def __init__(self, num_envs, events):
        self.device = torch.device("cpu")
        self._events = events
        self._body_names = ["BASE_LINK", "panda_hand"]
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.data = SimpleNamespace(
            body_quat_w=identity.reshape(1, 1, 4).repeat(num_envs, 2, 1)
        )
        self.external_force = None
        self.external_torque = None
        self.external_wrench_calls = 0

    def find_bodies(self, expression, preserve_order=True):
        matches = [
            (index, name)
            for index, name in enumerate(self._body_names)
            if name == expression
        ]
        return [item[0] for item in matches], [item[1] for item in matches]

    def set_external_force_and_torque(self, forces, torques, body_ids=None):
        assert body_ids == [1]
        self.external_force = forces.detach().clone()
        self.external_torque = torques.detach().clone()
        self.external_wrench_calls += 1
        self._events.append("wrench")


class _FakeScene(dict):
    def __init__(self, robot, num_envs):
        super().__init__(robot=robot)
        self.env_origins = torch.zeros(num_envs, 3)


class _FakeObservationManager:
    def __init__(self, env):
        self.env = env

    def compute(self):
        return {"policy": self.env.policy_observation.clone()}


class _FakeEnv:
    def __init__(self, num_envs=2, stage="A0"):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.max_episode_length = 500
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.events = []
        self.robot = _FakeRobot(num_envs, self.events)
        self.scene = _FakeScene(self.robot, num_envs)
        self.action_manager = SimpleNamespace(total_action_dim=16)
        self.policy_observation = torch.arange(
            num_envs * 60, dtype=torch.float32
        ).reshape(num_envs, 60)
        self.observation_manager = _FakeObservationManager(self)
        self.cfg = SimpleNamespace(
            teacher_stage=stage,
            teacher_force_limit_n=(10.0, 10.0, 10.0)
            if stage == "A0"
            else (20.0, 20.0, 20.0),
            teacher_torque_limit_nm=(2.0, 2.0, 2.0)
            if stage == "A0"
            else (5.0, 5.0, 5.0),
            teacher_hold_time_s=(1.0, 2.0) if stage == "A0" else (0.25, 1.0),
            teacher_curriculum_start_scale=0.25,
            teacher_curriculum_steps=50_000 if stage == "A0" else 75_000,
            teacher_mode_probabilities=(1.0, 0.0, 0.0)
            if stage == "A0"
            else (0.50, 0.30, 0.20),
            teacher_pulse_on_fraction=0.20,
            sim=SimpleNamespace(dt=0.005),
            decimation=4,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=float
        )
        self.action_space = self.single_action_space
        self.observation_space = gym.spaces.Dict({})
        self.last_action = None
        self.next_terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.next_truncated = torch.zeros(num_envs, dtype=torch.bool)
        self.next_reward = torch.ones(num_envs)
        self.reset_count = 0

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.reset_count += 1
        return {"policy": self.policy_observation.clone()}, {}

    def step(self, action):
        self.events.append("step")
        self.last_action = action.detach().clone()
        self.policy_observation = self.policy_observation + 1.0
        return (
            {"policy": self.policy_observation.clone()},
            self.next_reward.clone(),
            self.next_terminated.clone(),
            self.next_truncated.clone(),
            {},
        )


class _FakeFrozenActor(torch.nn.Module):
    def __init__(self, *, output_shape=(2, 16), fill_value=0.5):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_shape = output_shape
        self.fill_value = fill_value
        self.last_observation = None
        self.eval()

    def act_inference(self, observation):
        self.last_observation = observation.detach().clone()
        return torch.full(
            self.output_shape,
            self.fill_value,
            device=observation.device,
            dtype=observation.dtype,
        )


def test_a0_wrapper_exposes_exact_rsl_observation_and_action_contract():
    env = _FakeEnv(num_envs=2, stage="A0")

    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=5)
    obs, extras = wrapper.get_observations()

    assert wrapper.num_envs == 2
    assert wrapper.num_actions == 16
    assert wrapper.device == torch.device("cpu")
    assert obs.shape == (2, 60)
    assert torch.equal(extras["observations"]["critic"], obs)
    assert env.reset_count == 1


def test_a0_step_composes_from_zero_and_applies_wrench_before_physics():
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=5)
    env.events.clear()
    raw = torch.full((2, 16), 0.5)

    next_obs, reward, done, extras = wrapper.step(raw)

    assert env.events == ["wrench", "step"]
    assert env.robot.external_wrench_calls == 2
    assert env.robot.external_force.shape == (2, 1, 3)
    assert env.robot.external_torque.shape == (2, 1, 3)
    assert torch.count_nonzero(env.robot.external_force) > 0
    assert wrapper.max_abs_wrench_seen > 0.0
    assert torch.allclose(env.last_action[:, :12], torch.full((2, 12), 0.04))
    assert torch.allclose(env.last_action[:, 12:], torch.full((2, 4), 0.025))
    assert torch.equal(wrapper.last_final_action, env.last_action)
    assert torch.equal(wrapper.last_trainable_residual, raw)
    assert torch.equal(env.m1_teacher_trainable_residual, raw)
    assert torch.equal(
        env.m1_teacher_previous_trainable_residual, torch.zeros_like(raw)
    )
    assert next_obs.shape == (2, 60)
    assert torch.equal(reward, torch.ones(2))
    assert not done.any()
    assert torch.equal(extras["observations"]["critic"], next_obs)
    assert torch.equal(extras["time_outs"], torch.zeros(2, dtype=torch.bool))


def test_done_resets_only_selected_wrapper_and_wrench_state():
    env = _FakeEnv(num_envs=3, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=9)
    raw = torch.full((3, 16), 0.5)
    wrapper.step(raw)
    env.next_terminated = torch.tensor([False, True, False])

    _, _, done, _ = wrapper.step(raw)

    assert torch.equal(done, torch.tensor([False, True, False]))
    physical = wrapper.residual_composer.physical_residual
    assert torch.allclose(physical[[0, 2], :12], torch.full((2, 12), 0.02))
    assert torch.equal(physical[1], torch.zeros(16))
    assert torch.equal(wrapper.current_wrench_b[1], torch.zeros(6))
    assert torch.count_nonzero(wrapper.current_wrench_b[[0, 2]]) > 0
    assert torch.equal(env.m1_teacher_trainable_residual[1], torch.zeros(16))
    assert torch.equal(
        env.m1_teacher_previous_trainable_residual[1], torch.zeros(16)
    )
    assert torch.equal(wrapper.last_final_action[1], torch.zeros(16))
    assert torch.equal(env.robot.external_force[1], torch.zeros(1, 3))
    assert torch.equal(env.robot.external_torque[1], torch.zeros(1, 3))


def test_explicit_reset_clears_all_state_and_external_wrench():
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=4)
    wrapper.step(torch.ones(2, 16))
    calls_before = env.robot.external_wrench_calls

    obs, extras = wrapper.reset()

    assert env.robot.external_wrench_calls == calls_before + 1
    assert torch.equal(wrapper.residual_composer.physical_residual, torch.zeros(2, 16))
    assert torch.equal(wrapper.current_wrench_b, torch.zeros(2, 6))
    assert torch.equal(wrapper.last_final_action, torch.zeros(2, 16))
    assert torch.equal(wrapper.last_trainable_residual, torch.zeros(2, 16))
    assert torch.equal(env.robot.external_force, torch.zeros(2, 1, 3))
    assert torch.equal(env.robot.external_torque, torch.zeros(2, 1, 3))
    assert obs.shape == (2, 60)
    assert torch.equal(extras["observations"]["critic"], obs)


@pytest.mark.parametrize(
    ("actions", "error", "message"),
    [
        (torch.zeros(2, 15), ValueError, "shape"),
        (torch.zeros(2, 16, dtype=torch.float64), ValueError, "float32"),
        (
            torch.cat(
                (torch.full((1, 1), float("nan")), torch.zeros(1, 15)), dim=1
            ).repeat(2, 1),
            ValueError,
            "finite",
        ),
    ],
)
def test_invalid_action_fails_before_mutating_wrapper_state(actions, error, message):
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=6)
    physical_before = wrapper.residual_composer.physical_residual
    wrench_before = wrapper.current_wrench_b
    published_before = env.m1_teacher_trainable_residual.clone()
    calls_before = env.robot.external_wrench_calls

    with pytest.raises(error, match=message):
        wrapper.step(actions)

    assert torch.equal(wrapper.residual_composer.physical_residual, physical_before)
    assert torch.equal(wrapper.current_wrench_b, wrench_before)
    assert torch.equal(env.m1_teacher_trainable_residual, published_before)
    assert env.robot.external_wrench_calls == calls_before
    assert env.last_action is None


def test_nonfinite_reward_and_returned_observation_fail_loudly():
    reward_env = _FakeEnv(num_envs=2, stage="A0")
    reward_wrapper = M1PandaTeacherEnvWrapper(reward_env, stage="A0", seed=1)
    reward_env.next_reward[0] = float("nan")
    with pytest.raises(RuntimeError, match="rewards.*finite"):
        reward_wrapper.step(torch.zeros(2, 16))

    obs_env = _FakeEnv(num_envs=2, stage="A0")
    obs_wrapper = M1PandaTeacherEnvWrapper(obs_env, stage="A0", seed=1)
    obs_env.policy_observation[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="observation.*finite"):
        obs_wrapper.step(torch.zeros(2, 16))


def test_constructor_rejects_wrong_observation_action_stage_and_bodies():
    observation_env = _FakeEnv(num_envs=2, stage="A0")
    observation_env.policy_observation = torch.zeros(2, 59)
    with pytest.raises(RuntimeError, match="observation.*shape"):
        M1PandaTeacherEnvWrapper(observation_env, stage="A0")

    action_env = _FakeEnv(num_envs=2, stage="A0")
    action_env.action_manager.total_action_dim = 15
    with pytest.raises(ValueError, match="16 actions"):
        M1PandaTeacherEnvWrapper(action_env, stage="A0")

    stage_env = _FakeEnv(num_envs=2, stage="A0")
    with pytest.raises(ValueError, match="does not match"):
        M1PandaTeacherEnvWrapper(stage_env, stage="A1", base_actor=object())

    body_env = _FakeEnv(num_envs=2, stage="A0")
    body_env.robot._body_names = ["BASE_LINK", "BASE_LINK", "panda_hand"]
    with pytest.raises(RuntimeError, match="exactly one body"):
        M1PandaTeacherEnvWrapper(body_env, stage="A0")


def test_constructor_rejects_env_disturbance_contract_drift():
    env = _FakeEnv(num_envs=2, stage="A0")
    env.cfg.teacher_force_limit_n = (99.0, 10.0, 10.0)

    with pytest.raises(ValueError, match="does not match approved"):
        M1PandaTeacherEnvWrapper(env, stage="A0")


def test_wrapper_diagnostics_return_clones():
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=0)
    wrapper.step(torch.full((2, 16), 0.5))
    final = wrapper.last_final_action
    residual = wrapper.last_trainable_residual
    wrench = wrapper.current_wrench_b

    final.zero_()
    residual.zero_()
    wrench.zero_()

    assert torch.count_nonzero(wrapper.last_final_action) > 0
    assert torch.count_nonzero(wrapper.last_trainable_residual) > 0
    assert torch.count_nonzero(wrapper.current_wrench_b) > 0


def test_a1_composes_frozen_base_actor_then_trainable_residual():
    env = _FakeEnv(num_envs=2, stage="A1")
    frozen = _FakeFrozenActor()
    wrapper = M1PandaTeacherEnvWrapper(
        env, stage="A1", base_actor=frozen, seed=7
    )
    cached = wrapper.get_observations()[0].clone()

    wrapper.step(torch.full((2, 16), -0.25))

    assert torch.equal(frozen.last_observation, cached)
    assert torch.allclose(
        wrapper.base_composer.physical_residual[:, :12],
        torch.full((2, 12), 0.01),
    )
    assert torch.allclose(
        wrapper.base_composer.physical_residual[:, 12:],
        torch.full((2, 4), 0.2),
    )
    assert torch.allclose(
        wrapper.residual_composer.physical_residual[:, :12],
        torch.full((2, 12), -0.01),
    )
    assert torch.allclose(
        wrapper.residual_composer.physical_residual[:, 12:],
        torch.full((2, 4), -0.2),
    )
    assert torch.equal(wrapper.last_final_action, torch.zeros(2, 16))


@pytest.mark.parametrize(
    ("actor_factory", "message"),
    [
        (lambda: None, "requires a base_actor"),
        (lambda: _FakeFrozenActor().train(), "eval mode"),
        (
            lambda: _FakeFrozenActor().requires_grad_(True),
            "parameters must be frozen",
        ),
    ],
)
def test_a1_constructor_rejects_invalid_base_actor(actor_factory, message):
    env = _FakeEnv(num_envs=2, stage="A1")

    with pytest.raises((TypeError, ValueError), match=message):
        M1PandaTeacherEnvWrapper(
            env, stage="A1", base_actor=actor_factory()
        )


@pytest.mark.parametrize(
    ("actor", "message"),
    [
        (_FakeFrozenActor(output_shape=(2, 15)), "shape"),
        (_FakeFrozenActor(fill_value=float("nan")), "finite"),
    ],
)
def test_a1_rejects_invalid_base_actor_output_before_mutating_state(actor, message):
    env = _FakeEnv(num_envs=2, stage="A1")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A1", base_actor=actor)
    calls_before = env.robot.external_wrench_calls

    with pytest.raises(ValueError, match=message):
        wrapper.step(torch.zeros(2, 16))

    assert torch.equal(wrapper.base_composer.physical_residual, torch.zeros(2, 16))
    assert torch.equal(wrapper.residual_composer.physical_residual, torch.zeros(2, 16))
    assert env.robot.external_wrench_calls == calls_before
    assert env.last_action is None


def test_a1_done_resets_both_composers_only_for_selected_environments():
    env = _FakeEnv(num_envs=3, stage="A1")
    frozen = _FakeFrozenActor(output_shape=(3, 16))
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A1", base_actor=frozen)
    wrapper.step(torch.full((3, 16), -0.25))
    env.next_terminated = torch.tensor([False, True, False])

    wrapper.step(torch.full((3, 16), -0.25))

    assert torch.equal(wrapper.base_composer.physical_residual[1], torch.zeros(16))
    assert torch.equal(wrapper.residual_composer.physical_residual[1], torch.zeros(16))
    assert torch.allclose(
        wrapper.base_composer.physical_residual[[0, 2], :12],
        torch.full((2, 12), 0.02),
    )
    assert torch.allclose(
        wrapper.residual_composer.physical_residual[[0, 2], :12],
        torch.full((2, 12), -0.0125),
    )


def test_a1_frozen_actor_hash_is_stable_and_detects_parameter_drift():
    env = _FakeEnv(num_envs=2, stage="A1")
    frozen = _FakeFrozenActor()
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A1", base_actor=frozen)
    initial_hash = wrapper.frozen_actor_hash

    wrapper.step(torch.zeros(2, 16))
    wrapper.assert_frozen_actor_unchanged()
    assert wrapper.frozen_actor_hash == initial_hash

    with torch.no_grad():
        frozen.anchor.add_(1.0)

    with pytest.raises(RuntimeError, match=f"{initial_hash}.*current"):
        wrapper.assert_frozen_actor_unchanged()
