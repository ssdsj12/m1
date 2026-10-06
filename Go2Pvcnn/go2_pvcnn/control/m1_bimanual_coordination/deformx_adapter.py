"""Optional DeformX Cosserat bridge for the M1 wire controller.

The import is lazy so CPU unit tests and non-wire tasks do not require the
DeformX checkout.  Isaac-facing code calls :meth:`step` once per coarse
control tick; DeformX advances its fine internal rod steps and returns the
integrated frame impulse.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import sys
from typing import Any

import numpy as np

from .dlo_wire_goal import DloCenterlineState


def _deformx_root() -> str:
    root = os.environ.get("DEFORMX_ROOT")
    if not root:
        raise RuntimeError("DEFORMX_ROOT must point to the DeformX checkout")
    return root


def _load_deformx() -> tuple[Any, Any, Any, Any]:
    root = _deformx_root()
    mesh_root = os.environ.get("DEFORMX_ELASTICA_ROOT", os.path.join(root, "PyElastica-Mesh"))
    for path in (root, mesh_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from co_sim.engine import CoSimEngine
        from co_sim.models import CoSimConfig, FrameState
        from co_sim.models import RodInitialState
    except ImportError as exc:
        raise RuntimeError(
            "DeformX backend unavailable; install its dependencies and set "
            "DEFORMX_ROOT/DEFORMX_ELASTICA_ROOT"
        ) from exc
    return CoSimEngine, CoSimConfig, FrameState, RodInitialState


@dataclass(frozen=True)
class DloStepResult:
    state: DloCenterlineState
    linear_impulse: np.ndarray
    angular_impulse: np.ndarray
    elapsed_time: float


@dataclass(frozen=True)
class DualDloStepResult:
    state: DloCenterlineState
    left_linear_impulse: np.ndarray
    left_angular_impulse: np.ndarray
    right_linear_impulse: np.ndarray
    right_angular_impulse: np.ndarray
    elapsed_time: float


class DeformXWireBackend:
    """Thin adapter translating M1 frame commands to DeformX."""

    def __init__(self, config: Any | None = None):
        Engine, Config, FrameState, _ = _load_deformx()
        self._frame_state_type = FrameState
        self._engine = Engine(Config() if config is None else config)

    @classmethod
    def from_config(cls, **kwargs: Any) -> "DeformXWireBackend":
        _, Config, _, _ = _load_deformx()
        return cls(Config(**kwargs))

    def step(
        self,
        position: np.ndarray,
        director: np.ndarray,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
    ) -> DloStepResult:
        zeros = np.zeros(3, dtype=np.float64)
        frame = self._frame_state_type(
            position=np.asarray(position, dtype=np.float64),
            director=np.asarray(director, dtype=np.float64),
            velocity=zeros if velocity is None else np.asarray(velocity, dtype=np.float64),
            acceleration=zeros if acceleration is None else np.asarray(acceleration, dtype=np.float64),
            omega=zeros if omega is None else np.asarray(omega, dtype=np.float64),
            alpha=zeros if alpha is None else np.asarray(alpha, dtype=np.float64),
        )
        impulse = self._engine.update_frame_state(frame)
        snapshot = self._engine.snapshot()
        centerline = np.asarray(snapshot.rod_position, dtype=np.float64).T
        radius = float(getattr(self._engine.config, "base_radius", 0.0))
        state = DloCenterlineState(centerline, radius_m=radius)
        return DloStepResult(
            state=state,
            linear_impulse=np.asarray(impulse.linear_impulse, dtype=np.float64),
            angular_impulse=np.asarray(impulse.angular_impulse, dtype=np.float64),
            elapsed_time=float(impulse.elapsed_time),
        )


class DeformXDualWireBackend:
    """Lazy dual-endpoint adapter around one DeformX Cosserat rod."""

    def __init__(self, config: Any):
        self.config = config
        self._engine = None
        self._frame_state_type = None

    @classmethod
    def from_config(cls, **kwargs: Any) -> "DeformXDualWireBackend":
        _, Config, _, _ = _load_deformx()
        return cls(Config(**kwargs))

    def _frame(self, values: dict[str, np.ndarray]):
        if self._frame_state_type is None:
            raise RuntimeError("dual backend frame type is not initialized")
        zeros = np.zeros(3, dtype=np.float64)
        return self._frame_state_type(
            position=np.asarray(values["position"], dtype=np.float64),
            director=np.asarray(values["director"], dtype=np.float64),
            velocity=np.asarray(values.get("velocity", zeros), dtype=np.float64),
            acceleration=np.asarray(values.get("acceleration", zeros), dtype=np.float64),
            omega=np.asarray(values.get("omega", zeros), dtype=np.float64),
            alpha=np.asarray(values.get("alpha", zeros), dtype=np.float64),
        )

    def _initialize(self, left_values: dict[str, np.ndarray], right_values: dict[str, np.ndarray]) -> None:
        Engine, _, FrameState, RodInitialState = _load_deformx()
        self._frame_state_type = FrameState
        left = self._frame(left_values)
        right = self._frame(right_values)
        delta = right.position - left.position
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length <= 1.0e-4:
            raise ValueError("dual DeformX endpoints must be distinct and finite")
        direction = delta / length
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(direction, normal))) > 0.9:
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        normal -= float(np.dot(normal, direction)) * direction
        normal /= np.linalg.norm(normal)
        config = replace(self.config, base_length=length)
        rod_initial = RodInitialState(
            start=left.position.copy(),
            direction=direction,
            normal=normal,
        )
        self._engine = Engine(
            config,
            rod_initial_state=rod_initial,
            frame_initial_state=left,
            second_frame_initial_state=right,
        )
        self.config = config

    def step(
        self,
        left_frame: dict[str, np.ndarray],
        right_frame: dict[str, np.ndarray],
    ) -> DualDloStepResult:
        if self._engine is None:
            self._initialize(left_frame, right_frame)
        left = self._frame(left_frame)
        right = self._frame(right_frame)
        impulse = self._engine.update_dual_frame_state(left, right)
        snapshot = self._engine.snapshot()
        state = DloCenterlineState(
            np.asarray(snapshot.rod_position, dtype=np.float64).T,
            radius_m=float(self.config.base_radius),
        )
        return DualDloStepResult(
            state=state,
            left_linear_impulse=np.asarray(impulse.left_linear_impulse, dtype=np.float64),
            left_angular_impulse=np.asarray(impulse.left_angular_impulse, dtype=np.float64),
            right_linear_impulse=np.asarray(impulse.right_linear_impulse, dtype=np.float64),
            right_angular_impulse=np.asarray(impulse.right_angular_impulse, dtype=np.float64),
            elapsed_time=float(impulse.elapsed_time),
        )


__all__ = [
    "DeformXDualWireBackend",
    "DeformXWireBackend",
    "DloStepResult",
    "DualDloStepResult",
]
