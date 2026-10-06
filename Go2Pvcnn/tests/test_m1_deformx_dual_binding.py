from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from go2_pvcnn.control.m1_bimanual_coordination import deformx_adapter
from go2_pvcnn.control.m1_bimanual_coordination.deformx_adapter import DeformXDualWireBackend
from go2_pvcnn.control.m1_bimanual_coordination.dlo_wire_goal import DloCenterlineState


@dataclass(frozen=True)
class _FakeConfig:
    base_radius: float = 0.005
    base_length: float | None = None


@dataclass
class _FakeFrame:
    position: np.ndarray
    director: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray


@dataclass
class _FakeRodInitialState:
    start: np.ndarray
    direction: np.ndarray
    normal: np.ndarray


class _FakeEngine:
    instances: list["_FakeEngine"] = []

    def __init__(self, config, *, rod_initial_state, frame_initial_state, second_frame_initial_state):
        self.config = config
        self.rod_initial_state = rod_initial_state
        self.frame_initial_state = frame_initial_state
        self.second_frame_initial_state = second_frame_initial_state
        self.__class__.instances.append(self)

    def update_dual_frame_state(self, left, right):
        self.last_frames = (left, right)
        return SimpleNamespace(
            left_linear_impulse=np.zeros(3),
            left_angular_impulse=np.zeros(3),
            right_linear_impulse=np.zeros(3),
            right_angular_impulse=np.zeros(3),
            elapsed_time=0.01,
        )

    def snapshot(self):
        return SimpleNamespace(
            rod_position=np.array([[0.0, 0.1], [0.0, 0.0], [0.0, 0.0]], dtype=float)
        )


@pytest.fixture(autouse=True)
def fake_deformx(monkeypatch):
    _FakeEngine.instances.clear()
    monkeypatch.setattr(
        deformx_adapter,
        "_load_deformx",
        lambda: (_FakeEngine, _FakeConfig, _FakeFrame, _FakeRodInitialState),
    )


@pytest.fixture
def state():
    return DloCenterlineState(
        np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.10, 0.0, 0.0]]),
        radius_m=0.005,
    )


def _frame(position):
    return {
        "position": np.asarray(position, dtype=float),
        "director": np.eye(3),
    }


def test_bind_is_explicit_once_and_step_uses_bound_engine(state):
    backend = DeformXDualWireBackend.from_config(base_radius=0.005)
    left = _frame(state.positions[0])
    right = _frame(state.positions[-1])

    assert not backend.is_bound
    with pytest.raises(RuntimeError, match="bind"):
        backend.step(left_frame=left, right_frame=right)

    backend.bind(state, left, right, max_endpoint_error_m=0.005)

    assert backend.is_bound
    assert backend.bind_count == 1
    assert len(_FakeEngine.instances) == 1
    assert _FakeEngine.instances[0].rod_initial_state.start == pytest.approx(state.positions[0])
    backend.step(left_frame=left, right_frame=right)


def test_bind_rejects_nonfinite_frames(state):
    backend = DeformXDualWireBackend.from_config(base_radius=0.005)

    with pytest.raises(ValueError, match="finite"):
        backend.bind(state, _frame([np.nan, 0.0, 0.0]), _frame(state.positions[-1]))

    assert not backend.is_bound
    assert backend.bind_count == 0
    assert not _FakeEngine.instances


def test_bind_rejects_endpoint_mismatch_and_reversed_centerline(state):
    backend = DeformXDualWireBackend.from_config(base_radius=0.005)

    with pytest.raises(ValueError, match="endpoint"):
        backend.bind(state, _frame([0.006, 0.0, 0.0]), _frame(state.positions[-1]))

    with pytest.raises(ValueError, match="endpoint"):
        backend.bind(state, _frame(state.positions[-1]), _frame(state.positions[0]))

    reversed_state = DloCenterlineState(state.positions[::-1], radius_m=state.radius_m)
    with pytest.raises(ValueError, match="endpoint"):
        backend.bind(reversed_state, _frame(state.positions[0]), _frame(state.positions[-1]))

    assert not backend.is_bound
    assert backend.bind_count == 0
    assert not _FakeEngine.instances


@pytest.mark.parametrize("max_endpoint_error_m", [0.0, -0.001, np.nan, np.inf, "invalid"])
def test_bind_rejects_invalid_endpoint_tolerance(state, max_endpoint_error_m):
    backend = DeformXDualWireBackend.from_config(base_radius=0.005)

    with pytest.raises(ValueError, match="max_endpoint_error_m"):
        backend.bind(
            state,
            _frame(state.positions[0]),
            _frame(state.positions[-1]),
            max_endpoint_error_m=max_endpoint_error_m,
        )

    assert not backend.is_bound
    assert backend.bind_count == 0
    assert not _FakeEngine.instances


def test_bind_rejects_degenerate_centerline_before_engine_construction():
    backend = DeformXDualWireBackend.from_config(base_radius=0.005)
    degenerate = object.__new__(DloCenterlineState)
    object.__setattr__(degenerate, "positions", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    object.__setattr__(degenerate, "radius_m", 0.005)

    with pytest.raises(ValueError, match="centerline"):
        backend.bind(degenerate, _frame([0.0, 0.0, 0.0]), _frame([0.0, 0.0, 0.0]))

    assert not backend.is_bound
    assert backend.bind_count == 0
    assert not _FakeEngine.instances


def test_bind_rejects_second_bind_and_reset_clears_bind_state(state):
    backend = DeformXDualWireBackend.from_config(base_radius=0.005)
    left = _frame(state.positions[0])
    right = _frame(state.positions[-1])
    backend.bind(state, left, right)

    with pytest.raises(RuntimeError, match="already bound"):
        backend.bind(state, left, right)

    backend.reset()

    assert not backend.is_bound
    assert backend.bind_count == 0
    assert backend._frame_state_type is None
