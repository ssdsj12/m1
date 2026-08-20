from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.batched_rolling_teacher import (
    BatchedRollingTeacherBank,
)
from go2_pvcnn.control.m1_panda_coordination.contracts import (
    M1_LEG_JOINT_NAMES,
    M1_WHEEL_JOINT_NAMES,
    PANDA_ARM_JOINT_NAMES,
    PANDA_FINGER_JOINT_NAMES,
)
from go2_pvcnn.control.m1_panda_coordination.runtime_adapter import (
    PhysxTeacherAdapter,
)


WHEELS = (
    "FAR_FOOT_LINK",
    "FBL_FOOT_LINK",
    "RAR_FOOT_LINK",
    "RBL_FOOT_LINK",
)


class _FakeMath:
    @staticmethod
    def euler_xyz_from_quat(quaternion):
        batch = quaternion.shape[0]
        return tuple(torch.zeros(batch) for _ in range(3))


class _FakeContactSensor:
    def __init__(self, body_names, num_envs):
        self.body_names = body_names
        self.data = SimpleNamespace(
            net_forces_w=torch.zeros(num_envs, len(body_names), 3)
        )

    def find_bodies(self, names, preserve_order=True):
        del preserve_order
        if isinstance(names, str):
            names = [names]
        ids = [self.body_names.index(name) for name in names]
        return ids, [self.body_names[index] for index in ids]


def _fake_env(num_envs: int):
    joint_names = list(
        M1_LEG_JOINT_NAMES
        + M1_WHEEL_JOINT_NAMES
        + PANDA_ARM_JOINT_NAMES
        + PANDA_FINGER_JOINT_NAMES
    )
    body_names = ["BASE_LINK", *WHEELS, "panda_link0", "panda_hand"]
    root_pos = torch.stack(
        [torch.tensor([10.0 * index, index + 0.5, 1.0]) for index in range(num_envs)]
    )
    joint_pos = torch.stack(
        [torch.arange(25, dtype=torch.float32) + 100.0 * index for index in range(num_envs)]
    )
    data = SimpleNamespace(
        root_pos_w=root_pos,
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
        body_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(
            num_envs, len(body_names), 1
        ),
        joint_pos=joint_pos,
        body_pos_w=torch.zeros(num_envs, len(body_names), 3),
        root_lin_vel_w=torch.stack(
            [torch.full((3,), float(index)) for index in range(num_envs)]
        ),
        root_ang_vel_w=torch.stack(
            [torch.full((3,), float(index + 10)) for index in range(num_envs)]
        ),
        joint_vel=torch.stack(
            [torch.full((25,), float(index + 20)) for index in range(num_envs)]
        ),
    )
    robot = SimpleNamespace(
        num_instances=num_envs,
        joint_names=joint_names,
        body_names=body_names,
        data=data,
        device=torch.device("cpu"),
    )
    effort_names = list(
        M1_LEG_JOINT_NAMES + M1_WHEEL_JOINT_NAMES + PANDA_ARM_JOINT_NAMES
    )
    action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(_joint_names=effort_names)
    )
    return SimpleNamespace(
        scene={
            "robot": robot,
            "contact_forces": _FakeContactSensor(body_names, num_envs),
        },
        action_manager=action_manager,
        math_utils=_FakeMath,
    )


def test_adapter_selects_only_requested_environment():
    env = _fake_env(3)
    adapter = PhysxTeacherAdapter(env, env_index=2)
    torch.testing.assert_close(
        adapter._initial_root_pos, torch.tensor([20.0, 2.5, 1.0], dtype=torch.float64)
    )
    expected_velocity = torch.cat(
        (
            torch.full((3,), 2.0),
            torch.full((3,), 12.0),
            torch.full((25,), 22.0),
        )
    ).to(torch.float64)
    torch.testing.assert_close(adapter._generalized_velocity(), expected_velocity)


def test_adapter_rejects_out_of_range_environment_index():
    with pytest.raises(IndexError, match="env_index"):
        PhysxTeacherAdapter(_fake_env(2), env_index=2)


class _FakeTeacher:
    def __init__(self, index: int):
        self.index = index
        self.schedule = SimpleNamespace(phase=0)
        self.trajectory = SimpleNamespace(samples=[])
        self.motion_distributor = SimpleNamespace(calls=0)
        self.qp_backend = SimpleNamespace(warm_start=torch.tensor([float(index)]))
        self.safety = SimpleNamespace(state=SimpleNamespace(name="TRACK"))
        self.settling_center = torch.tensor([float(index)])
        self.history = []
        self.first_failure = SimpleNamespace(value=None)
        self.reset_calls = []

    def reset(self, state, *, seed: int):
        self.reset_calls.append((state, seed))
        self.schedule.phase = 0
        self.safety.state = SimpleNamespace(name="TRACK")
        self.qp_backend.warm_start.zero_()

    def step(self, state, *, mission_sample):
        self.schedule.phase += 1
        self.history.append(state)
        self.qp_backend.warm_start += 1
        return SimpleNamespace(
            effort=torch.tensor([float(self.index)]),
            state=state,
            mission=mission_sample,
        )


def _bank(num_envs: int = 2):
    teachers = [_FakeTeacher(index) for index in range(num_envs)]
    adapters = [SimpleNamespace(env_index=index) for index in range(num_envs)]
    return BatchedRollingTeacherBank(teachers, adapters, base_seed=41)


def test_reset_and_warm_start_are_not_shared_between_teachers():
    bank = _bank()
    bank.step(["state-0", "state-1"], ["mission-0", "mission-1"])
    qp0 = bank.teachers[0].qp_backend.warm_start.clone()
    phase0 = bank.teachers[0].schedule.phase
    history0 = list(bank.teachers[0].history)
    first_failure0 = bank.teachers[0].first_failure
    bank.reset(torch.tensor([1]), ["reset-0", "reset-1"])
    torch.testing.assert_close(bank.teachers[0].qp_backend.warm_start, qp0)
    assert bank.teachers[0].schedule.phase == phase0
    assert bank.teachers[0].history == history0
    assert bank.teachers[0].first_failure is first_failure0
    assert bank.teachers[1].safety.state.name == "TRACK"
    assert bank.teachers[1].reset_calls == [("reset-1", 42)]


def test_bank_steps_each_teacher_with_its_matching_mission():
    bank = _bank(3)
    commands = bank.step(
        ["state-0", "state-1", "state-2"],
        ["mission-0", "mission-1", "mission-2"],
    )
    assert [command.state for command in commands] == ["state-0", "state-1", "state-2"]
    assert [command.mission for command in commands] == [
        "mission-0",
        "mission-1",
        "mission-2",
    ]


def test_bank_rejects_shared_mutable_controller_subobjects():
    first, second = _FakeTeacher(0), _FakeTeacher(1)
    second.qp_backend = first.qp_backend
    with pytest.raises(ValueError, match="qp_backend"):
        BatchedRollingTeacherBank(
            [first, second],
            [SimpleNamespace(env_index=0), SimpleNamespace(env_index=1)],
        )


def test_bank_requires_one_distinct_adapter_per_teacher():
    teacher = _FakeTeacher(0)
    adapter = SimpleNamespace(env_index=0)
    with pytest.raises(ValueError, match="adapter"):
        BatchedRollingTeacherBank([teacher], [])
    with pytest.raises(ValueError, match="distinct"):
        BatchedRollingTeacherBank([_FakeTeacher(0), _FakeTeacher(1)], [adapter, adapter])
