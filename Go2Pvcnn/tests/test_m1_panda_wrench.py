from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE = Path(__file__).resolve().parents[1] / "go2_pvcnn/mdp/m1_panda_wrench.py"


def _quat_rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[..., 1:]
    return (
        vector * (2.0 * quat[..., :1].square() - 1.0)
        - 2.0 * quat[..., :1] * torch.linalg.cross(quat_vector, vector, dim=-1)
        + 2.0 * quat_vector * torch.sum(quat_vector * vector, dim=-1, keepdim=True)
    )


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_vector = quat[..., 1:]
    return (
        vector * (2.0 * quat[..., :1].square() - 1.0)
        + 2.0 * quat[..., :1] * torch.linalg.cross(quat_vector, vector, dim=-1)
        + 2.0 * quat_vector * torch.sum(quat_vector * vector, dim=-1, keepdim=True)
    )


@pytest.fixture
def wrench_module(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    utils = types.ModuleType("isaaclab.utils")
    math_utils = types.ModuleType("isaaclab.utils.math")
    math_utils.quat_rotate = _quat_rotate
    math_utils.quat_rotate_inverse = _quat_rotate_inverse
    utils.math = math_utils
    isaaclab.utils = utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)

    spec = importlib.util.spec_from_file_location("m1_panda_wrench_under_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identity_frame_keeps_force_and_shifts_moment_to_base_origin(wrench_module):
    result = wrench_module.shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 10.0, 0.0]]),
        torque_w=torch.tensor([[0.0, 0.0, 2.0]]),
        sensor_pos_w=torch.tensor([[1.0, 0.0, 0.0]]),
        base_pos_w=torch.zeros(1, 3),
        base_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )

    assert torch.allclose(result, torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 12.0]]))


def test_base_yaw_rotates_world_force_into_base_frame(wrench_module):
    half = 2.0**-0.5
    result = wrench_module.shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 1.0, 0.0]]),
        torque_w=torch.zeros(1, 3),
        sensor_pos_w=torch.zeros(1, 3),
        base_pos_w=torch.zeros(1, 3),
        base_quat_w=torch.tensor([[half, 0.0, 0.0, half]]),
    )

    assert torch.allclose(result[:, :3], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_batch_combines_nonzero_lever_arm_and_base_rotation(wrench_module):
    half = 2.0**-0.5
    result = wrench_module.shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]]),
        torque_w=torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]),
        sensor_pos_w=torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        base_pos_w=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        base_quat_w=torch.tensor([[half, 0.0, 0.0, half], [0.0, 0.0, 0.0, 1.0]]),
    )

    expected = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 5.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 3.0]])
    assert result.shape == (2, 6)
    assert torch.allclose(result, expected, atol=1e-6)


class _FakePhysxView:
    def __init__(self, incoming: torch.Tensor):
        self.incoming = incoming
        self.calls = 0

    def get_link_incoming_joint_force(self):
        self.calls += 1
        return self.incoming


class _FakeRobot:
    def __init__(self, responses, incoming, body_pos_w, body_quat_w):
        self.responses = responses
        self.find_calls = []
        self.root_physx_view = _FakePhysxView(incoming)
        self.data = SimpleNamespace(body_pos_w=body_pos_w, body_quat_w=body_quat_w)

    def find_bodies(self, name, preserve_order=False):
        self.find_calls.append((name, preserve_order))
        return self.responses[name]


def test_adapter_rotates_raw_joint_wrench_once_then_shifts_about_base(wrench_module):
    half = 2.0**-0.5
    incoming = torch.zeros(1, 2, 6)
    incoming[:, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    body_pos_w = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    body_quat_w = torch.tensor([[[half, 0.0, 0.0, half], [half, 0.0, 0.0, half]]])
    robot = _FakeRobot(
        responses={"mount": ([1], ["mount"]), "base": ([0], ["base"])},
        incoming=incoming,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[1], body_names=["conflicting_selector"])

    result = wrench_module.m1_panda_mount_wrench_b(
        SimpleNamespace(scene={"robot": robot}), asset_cfg, mount_body_name="mount", base_body_name="base"
    )

    assert torch.allclose(result, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 3.0]]), atol=1e-6)
    assert robot.find_calls == [("mount", True), ("base", True)]
    assert robot.root_physx_view.calls == 1


def test_wrench_module_exports_only_the_two_public_functions():
    source = MODULE.read_text()
    init_source = (MODULE.parent / "__init__.py").read_text()
    assert '__all__ = ["shift_rotate_wrench_to_base", "m1_panda_mount_wrench_b"]' in source
    assert "from .m1_panda_wrench import m1_panda_mount_wrench_b, shift_rotate_wrench_to_base" in init_source
    assert "from .m1_panda_wrench import *" not in init_source


@pytest.mark.parametrize(
    "responses",
    [
        {"mount": ([], []), "base": ([0], ["base"])},
        {"mount": ([1, 2], ["mount", "mount"]), "base": ([0], ["base"])},
        {"mount": ([1, 2], ["mount"]), "base": ([0], ["base"])},
        {"mount": ([1], ["mount"]), "base": ([], [])},
        {"mount": ([1], ["mount"]), "base": ([0, 2], ["base", "base"])},
    ],
    ids=["missing-mount", "duplicate-mount", "duplicate-mount-ids", "missing-base", "duplicate-base"],
)
def test_adapter_rejects_missing_or_duplicate_explicit_body(wrench_module, responses):
    robot = _FakeRobot(
        responses=responses,
        incoming=torch.zeros(1, 3, 6),
        body_pos_w=torch.zeros(1, 3, 3),
        body_quat_w=torch.zeros(1, 3, 4),
    )

    with pytest.raises(RuntimeError, match="Expected one mount/base body"):
        wrench_module.m1_panda_mount_wrench_b(
            SimpleNamespace(scene={"robot": robot}),
            SimpleNamespace(name="robot"),
            mount_body_name="mount",
            base_body_name="base",
        )
