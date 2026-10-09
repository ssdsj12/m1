"""Pure, bounded reduced-Cosserat predictor for the M1 wire adapter.

The predictor intentionally has no Isaac, DeformX, or torch dependency.  It
uses the measured endpoint boundary frames as the authoritative geometry and
keeps a small sampled centerline for MPC and safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from .dlo_wire_goal import DloCenterlineState


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and nonzero")
    return value / norm


@dataclass(frozen=True)
class ReducedCosseratState:
    """Immutable reduced wire state at one controller tick."""

    positions: np.ndarray
    tangents: np.ndarray
    endpoint_positions: np.ndarray
    endpoint_velocities: np.ndarray
    radius_m: float
    rest_length_m: float
    endpoint_separation_m: float
    axial_stretch: float
    bending_proxy: float
    left_normal_force_n: float
    right_normal_force_n: float
    left_tangential_force_n: float
    right_tangential_force_n: float
    left_slip_m: float
    right_slip_m: float
    center_height_m: float
    timestamp_s: float
    finite: bool
    friction_feasible: bool
    slip_feasible: bool
    feasible: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        positions = _array(self.positions, (np.asarray(self.positions).shape[0], 3), "positions")
        if positions.shape[0] < 2:
            raise ValueError("positions must contain at least two samples")
        tangents = _array(self.tangents, positions.shape, "tangents")
        endpoints = _array(self.endpoint_positions, (2, 3), "endpoint_positions")
        velocities = _array(self.endpoint_velocities, (2, 3), "endpoint_velocities")
        for name, value in (
            ("radius_m", self.radius_m),
            ("rest_length_m", self.rest_length_m),
            ("endpoint_separation_m", self.endpoint_separation_m),
            ("axial_stretch", self.axial_stretch),
            ("bending_proxy", self.bending_proxy),
            ("left_normal_force_n", self.left_normal_force_n),
            ("right_normal_force_n", self.right_normal_force_n),
            ("left_tangential_force_n", self.left_tangential_force_n),
            ("right_tangential_force_n", self.right_tangential_force_n),
            ("left_slip_m", self.left_slip_m),
            ("right_slip_m", self.right_slip_m),
            ("center_height_m", self.center_height_m),
            ("timestamp_s", self.timestamp_s),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.radius_m <= 0.0 or self.rest_length_m <= 0.0:
            raise ValueError("radius_m and rest_length_m must be positive")
        object.__setattr__(self, "positions", _readonly(positions))
        object.__setattr__(self, "tangents", _readonly(tangents))
        object.__setattr__(self, "endpoint_positions", _readonly(endpoints))
        object.__setattr__(self, "endpoint_velocities", _readonly(velocities))

    @property
    def centerline_positions(self) -> np.ndarray:
        """Descriptive alias used by MPC adapters."""
        return self.positions

    @property
    def endpoint_slip_m(self) -> np.ndarray:
        return _readonly((self.left_slip_m, self.right_slip_m))


@dataclass(frozen=True)
class ReducedCosseratPrediction:
    """A short-horizon batch of reduced states."""

    positions: np.ndarray
    tangents: np.ndarray
    endpoint_positions: np.ndarray
    endpoint_velocities: np.ndarray
    endpoint_separation_m: np.ndarray
    axial_stretch: np.ndarray
    bending_proxy: np.ndarray
    left_slip_m: np.ndarray
    right_slip_m: np.ndarray
    friction_feasible: np.ndarray
    slip_feasible: np.ndarray
    feasible: np.ndarray
    failure_reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError("positions must have shape (horizon, samples, 3)")
        horizon, samples, _ = positions.shape
        tangents = np.asarray(self.tangents, dtype=np.float64)
        endpoints = np.asarray(self.endpoint_positions, dtype=np.float64)
        velocities = np.asarray(self.endpoint_velocities, dtype=np.float64)
        if tangents.shape != positions.shape or endpoints.shape != (horizon, 2, 3) or velocities.shape != (horizon, 2, 3):
            raise ValueError("prediction geometry has inconsistent shapes")
        for name, value in (("positions", positions), ("tangents", tangents),
                            ("endpoint_positions", endpoints), ("endpoint_velocities", velocities)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        scalar_names = ("endpoint_separation_m", "axial_stretch", "bending_proxy", "left_slip_m", "right_slip_m")
        for name in scalar_names:
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (horizon,):
                raise ValueError(f"{name} must have shape (horizon,)")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, _readonly(value))
        for name in ("friction_feasible", "slip_feasible", "feasible"):
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != (horizon,):
                raise ValueError(f"{name} must have shape (horizon,)")
            object.__setattr__(self, name, _readonly(value, dtype=bool))
        object.__setattr__(self, "positions", _readonly(positions))
        object.__setattr__(self, "tangents", _readonly(tangents))
        object.__setattr__(self, "endpoint_positions", _readonly(endpoints))
        object.__setattr__(self, "endpoint_velocities", _readonly(velocities))
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        if len(self.failure_reasons) != horizon or samples < 2:
            raise ValueError("failure_reasons must match prediction horizon")

    @property
    def centerline_positions(self) -> np.ndarray:
        return self.positions

    @property
    def endpoint_slip_m(self) -> np.ndarray:
        return np.stack((self.left_slip_m, self.right_slip_m), axis=1)

class ReducedCosseratPredictor:
    """Deterministic endpoint-and-centerline predictor with fail-closed checks."""

    def __init__(
        self,
        *,
        effective_mass_kg: float = 0.05,
        damping: float = 1.0,
        friction_coefficient: float = 0.5,
        max_axial_stretch: float = 0.25,
        max_bending_proxy: float = 20.0,
        max_slip_m: float = 0.02,
        normal_force_min_n: float = 0.0,
    ) -> None:
        values = {
            "effective_mass_kg": effective_mass_kg,
            "damping": damping,
            "friction_coefficient": friction_coefficient,
            "max_axial_stretch": max_axial_stretch,
            "max_bending_proxy": max_bending_proxy,
            "max_slip_m": max_slip_m,
            "normal_force_min_n": normal_force_min_n,
        }
        self._validate_limits(values)
        self.effective_mass_kg = float(effective_mass_kg)
        self.damping = float(damping)
        self.friction_coefficient = float(friction_coefficient)
        self.max_axial_stretch = float(max_axial_stretch)
        self.max_bending_proxy = float(max_bending_proxy)
        self.max_slip_m = float(max_slip_m)
        self.normal_force_min_n = float(normal_force_min_n)
        self._dt = None
        self._rest_length = None
        self._radius = None
        self._state: ReducedCosseratState | None = None

    @staticmethod
    def _validate_limits(values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name == "effective_mass_kg" and value == 0):
                raise ValueError("predictor limits must be nonnegative and mass positive")

    @property
    def state(self) -> ReducedCosseratState:
        if self._state is None:
            raise RuntimeError("predictor must be reset before reading state")
        return self._state

    def reset(
        self,
        centerline: DloCenterlineState | np.ndarray,
        radius_m: float | None = None,
        material: Mapping[str, Any] | Any | None = None,
        dt: float = 0.01,
    ) -> ReducedCosseratState:
        if isinstance(centerline, DloCenterlineState):
            source = centerline
            radius = source.radius_m if radius_m is None else radius_m
        else:
            if radius_m is None:
                raise ValueError("radius_m is required for an array centerline")
            source = DloCenterlineState(np.asarray(centerline, dtype=np.float64), float(radius_m))
            radius = radius_m
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not math.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        self._apply_material(material)
        positions = np.asarray(source.positions, dtype=np.float64)
        self._state = None
        self._dt = float(dt)
        self._radius = float(radius)
        self._rest_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        tangent = self._tangents(positions)
        self._state = self._make_state(
            positions, tangent, np.stack((positions[0], positions[-1])), np.zeros((2, 3)),
            np.zeros((2, 3)), (np.array([0., 0., 1.]), np.array([0., 0., 1.])), 0.0,
        )
        return self._state

    def _apply_material(self, material: Mapping[str, Any] | Any | None) -> None:
        if material is None:
            return
        def get(name: str, default: Any) -> Any:
            return material.get(name, default) if isinstance(material, Mapping) else getattr(material, name, default)
        values = {
            "effective_mass_kg": get("effective_mass_kg", get("mass_kg", self.effective_mass_kg)),
            **{name: get(name, getattr(self, name)) for name in (
                "damping", "friction_coefficient", "max_axial_stretch",
                "max_bending_proxy", "max_slip_m", "normal_force_min_n",
            )},
        }
        self._validate_limits(values)
        for name, value in values.items():
            setattr(self, name, float(value))

    @staticmethod
    def _tangents(positions: np.ndarray) -> np.ndarray:
        delta = np.gradient(positions, axis=0)
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        return delta / np.maximum(norms, 1.0e-12)

    @staticmethod
    def _boundary(value: Any, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if isinstance(value, Mapping):
            pose = value.get("pose", value.get("position"))
            twist = value.get("twist", value.get("velocity", np.zeros(3)))
            normal = value.get("normal")
        elif hasattr(value, "pose") or hasattr(value, "position"):
            pose = getattr(value, "pose", getattr(value, "position", None))
            twist = getattr(value, "twist", getattr(value, "velocity", np.zeros(3)))
            normal = getattr(value, "normal", None)
        elif isinstance(value, (tuple, list)) and len(value) >= 2 and np.asarray(value[0]).shape in ((3,), (6,), (7,)):
            pose, twist = value[0], value[1]
            normal = value[2] if len(value) > 2 else None
        else:
            pose, twist, normal = value, np.zeros(3), None
        if pose is None:
            raise ValueError(f"{name} pose is required")
        pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose_arr.size not in (3, 6, 7) or not np.isfinite(pose_arr).all():
            raise ValueError(f"{name} pose must be finite with 3, 6, or 7 values")
        twist_arr = np.asarray(twist, dtype=np.float64).reshape(-1)
        if twist_arr.size not in (3, 6) or not np.isfinite(twist_arr).all():
            raise ValueError(f"{name} twist must be finite with 3 or 6 values")
        normal_arr = None if normal is None else _unit(_array(normal, (3,), f"{name} normal"), f"{name} normal")
        return pose_arr[:3].copy(), twist_arr[:3].copy(), normal_arr

    @staticmethod
    def _wrench_pair(value: Any) -> tuple[tuple[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]:
        if isinstance(value, Mapping):
            values = (value.get("left", value.get("left_wrench")), value.get("right", value.get("right_wrench")))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            values = (value[0], value[1])
        else:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape in ((2, 6), (2, 3)):
                values = (arr[0], arr[1])
            else:
                raise ValueError("contact_wrench must contain left and right six-dimensional wrenches")
        result = []
        for index, item in enumerate(values):
            normal = None
            wrench = item
            if isinstance(item, Mapping):
                wrench = item.get("wrench", item.get("force"))
                normal = item.get("normal")
            if wrench is None:
                raise ValueError("contact wrench is required for both endpoints")
            vector = np.asarray(wrench, dtype=np.float64).reshape(-1)
            if vector.size not in (3, 6) or not np.isfinite(vector).all():
                raise ValueError(f"contact_wrench[{index}] must be finite with 3 or 6 values")
            normal_arr = None if normal is None else _unit(_array(normal, (3,), f"contact_wrench[{index}] normal"), "contact normal")
            result.append((vector[:3].copy(), normal_arr))
        return result[0], result[1]

    def _make_state(self, positions, tangents, endpoints, velocities, wrenches, normals, timestamp) -> ReducedCosseratState:
        separation = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        stretch = separation / self._rest_length - 1.0
        angles = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0))
        segment = self._rest_length / max(len(positions) - 1, 1)
        bending = float(np.max(angles) / max(segment, 1.0e-12)) if angles.size else 0.0
        forces = []
        tangential = []
        for wrench, normal in zip(wrenches, normals):
            n = np.array([0.0, 0.0, 1.0]) if normal is None else normal
            normal_force = max(0.0, float(np.dot(wrench, n)))
            tangent_force = float(np.linalg.norm(wrench - float(np.dot(wrench, n)) * n))
            forces.append(normal_force)
            tangential.append(tangent_force)
        previous = self._state.endpoint_positions if self._state is not None else endpoints
        slips = [float(np.linalg.norm(endpoints[i] - previous[i])) for i in range(2)]
        friction_ok = all(n >= self.normal_force_min_n and t <= self.friction_coefficient * n + 1.0e-12 for n, t in zip(forces, tangential))
        slip_ok = all(s <= self.max_slip_m + 1.0e-12 for s in slips)
        stretch_ok = abs(stretch) <= self.max_axial_stretch + 1.0e-12
        bending_ok = bending <= self.max_bending_proxy + 1.0e-12
        reasons = []
        if not stretch_ok: reasons.append("axial_stretch")
        if not bending_ok: reasons.append("bending_proxy")
        if not friction_ok: reasons.append("friction_cone")
        if not slip_ok: reasons.append("slip")
        return ReducedCosseratState(
            positions=positions, tangents=tangents, endpoint_positions=endpoints,
            endpoint_velocities=velocities, radius_m=self._radius, rest_length_m=self._rest_length,
            endpoint_separation_m=separation, axial_stretch=stretch, bending_proxy=bending,
            left_normal_force_n=forces[0], right_normal_force_n=forces[1],
            left_tangential_force_n=tangential[0], right_tangential_force_n=tangential[1],
            left_slip_m=slips[0], right_slip_m=slips[1], center_height_m=float(np.mean(positions[:, 2])),
            timestamp_s=float(timestamp), finite=True, friction_feasible=friction_ok,
            slip_feasible=slip_ok, feasible=stretch_ok and bending_ok and friction_ok and slip_ok,
            failure_reason=None if not reasons else ";".join(reasons),
        )

    def step(self, left_boundary: Any, right_boundary: Any, contact_wrench: Any, dt: float | None = None) -> ReducedCosseratState:
        if self._state is None or self._dt is None:
            raise RuntimeError("predictor must be reset before step")
        left, right = self._boundary(left_boundary, "left_boundary"), self._boundary(right_boundary, "right_boundary")
        wrench_pair = self._wrench_pair(contact_wrench)
        step_dt = self._dt if dt is None else float(dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        endpoints = np.stack((left[0], right[0]))
        measured = np.stack((left[1], right[1]))
        forces = np.stack((wrench_pair[0][0], wrench_pair[1][0]))
        previous_velocity = self._state.endpoint_velocities
        velocities = measured + step_dt * (forces / self.effective_mass_kg - self.damping * previous_velocity)
        alpha = np.linspace(0.0, 1.0, self._state.positions.shape[0])[:, None]
        positions = endpoints[0] * (1.0 - alpha) + endpoints[1] * alpha
        tangents = self._tangents(positions)
        normals = (
            left[2] if left[2] is not None else wrench_pair[0][1],
            right[2] if right[2] is not None else wrench_pair[1][1],
        )
        candidate = self._make_state(positions, tangents, endpoints, velocities, forces, normals, self._state.timestamp_s + step_dt)
        if candidate.feasible:
            self._state = candidate
        return candidate

    def predict(self, horizon: int, boundary_sequence: Sequence[Any], right_boundaries: Sequence[Any] | None = None, wrench_sequence: Sequence[Any] | None = None) -> ReducedCosseratPrediction:
        if self._state is None:
            raise RuntimeError("predictor must be reset before predict")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if wrench_sequence is None:
            wrench_sequence = right_boundaries
            right_boundaries = None
        if wrench_sequence is None:
            raise ValueError("wrench_sequence is required")
        if right_boundaries is None:
            pairs = []
            for item in boundary_sequence:
                if isinstance(item, Mapping):
                    pairs.append((item["left"], item["right"]))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    pairs.append((item[0], item[1]))
                else:
                    raise ValueError("boundary_sequence must contain left/right pairs")
            left_boundaries = [pair[0] for pair in pairs]
            right_boundaries = [pair[1] for pair in pairs]
        else:
            left_boundaries = boundary_sequence
        if len(left_boundaries) == 0 or len(right_boundaries) == 0 or len(wrench_sequence) == 0:
            raise ValueError("prediction sequences must not be empty")
        accepted_state = self._state
        simulated = accepted_state
        rows = []
        try:
            for index in range(horizon):
                left = left_boundaries[min(index, len(left_boundaries) - 1)]
                right = right_boundaries[min(index, len(right_boundaries) - 1)]
                wrench = wrench_sequence[min(index, len(wrench_sequence) - 1)]
                # Step through a private predictor state, then restore the accepted state.
                self._state = simulated
                candidate = self.step(left, right, wrench)
                rows.append(candidate)
                if not candidate.feasible:
                    # An invalid transition cannot seed any later horizon row.
                    rows.extend([candidate] * (horizon - len(rows)))
                    break
                simulated = candidate
        finally:
            self._state = accepted_state
        return ReducedCosseratPrediction(
            positions=np.stack([row.positions for row in rows]),
            tangents=np.stack([row.tangents for row in rows]),
            endpoint_positions=np.stack([row.endpoint_positions for row in rows]),
            endpoint_velocities=np.stack([row.endpoint_velocities for row in rows]),
            endpoint_separation_m=np.asarray([row.endpoint_separation_m for row in rows]),
            axial_stretch=np.asarray([row.axial_stretch for row in rows]),
            bending_proxy=np.asarray([row.bending_proxy for row in rows]),
            left_slip_m=np.asarray([row.left_slip_m for row in rows]),
            right_slip_m=np.asarray([row.right_slip_m for row in rows]),
            friction_feasible=np.asarray([row.friction_feasible for row in rows]),
            slip_feasible=np.asarray([row.slip_feasible for row in rows]),
            feasible=np.asarray([row.feasible for row in rows]),
            failure_reasons=tuple(row.failure_reason for row in rows),
        )


__all__ = ["ReducedCosseratState", "ReducedCosseratPrediction", "ReducedCosseratPredictor"]
