"""Transactional reduced wire prediction behind the optional DeformX boundary."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import torch

from .contracts import BimanualSnapshot
from .deformx_adapter import DeformXDualWireBackend, endpoint_wrenches_from_impulses
from .dlo_frame_provider import dual_palm_frames
from .dlo_wire_goal import DloCenterlineState
from .reduced_cosserat import ReducedCosseratPredictor, ReducedCosseratPrediction, ReducedCosseratState


@dataclass(frozen=True)
class ReducedWireUpdate:
    state: ReducedCosseratState
    prediction: ReducedCosseratPrediction | None
    endpoint_targets: torch.Tensor
    endpoint_velocities: torch.Tensor
    endpoint_wrenches: torch.Tensor
    lift_velocity: torch.Tensor
    failure_reason: str | None


class ReducedWireRuntime:
    """Update in the common M1 base frame; publish only feasible full horizons.

    ``initial_state`` and all wrenches are base-frame quantities. A configured
    backend binds once on first update; the caller owns contact-latch eligibility.
    A grasp-frame provider can be supplied for palm-local pinch attachments.
    Backend impulses are added as signed reaction estimates to measured endpoint
    wrenches. Rejection preserves reduced state, but cannot undo an engine step.
    """

    def __init__(
        self, initial_state: DloCenterlineState, *,
        backend: DeformXDualWireBackend | None = None,
        predictor: ReducedCosseratPredictor | None = None,
        horizon_steps: int = 25, frame_provider: Callable = dual_palm_frames,
        material: Any = None, max_force_n: float = 50.0, max_moment_nm: float = 5.0,
    ) -> None:
        if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, int) or horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        for limit in (max_force_n, max_moment_nm):
            if not np.isfinite(limit) or limit <= 0:
                raise ValueError("wrench limits must be finite and positive")
        self.horizon_steps = horizon_steps
        self.backend = backend
        self.frame_provider = frame_provider
        self.max_force_n = float(max_force_n)
        self.max_moment_nm = float(max_moment_nm)
        self._initial_state = initial_state
        self._material = material
        self._predictor = copy(predictor) if predictor is not None else ReducedCosseratPredictor()
        self._predictor.reset(initial_state, material=material)

    @property
    def state(self) -> ReducedCosseratState:
        return self._predictor.state

    def reset(self) -> None:
        candidate = copy(self._predictor)
        candidate.reset(self._initial_state, material=self._material)
        if self.backend is not None:
            self.backend.reset()
        self._predictor = candidate

    def _result(self, snapshot, state, prediction, wrenches, reason=None):
        template = snapshot.left_arm.palm_pose_b
        def tensor(value):
            return torch.tensor(np.array(value, copy=True), dtype=template.dtype, device=template.device)
        if reason is not None:
            targets = np.repeat(state.endpoint_positions[None], self.horizon_steps, axis=0)
            velocities = np.zeros_like(targets)
            lift = np.zeros(self.horizon_steps)
        else:
            targets = prediction.endpoint_positions
            velocities = prediction.endpoint_velocities
            # Both endpoints must support the same upward speed.
            lift = np.maximum(0.0, np.min(velocities[:, :, 2], axis=1))
        return ReducedWireUpdate(state, prediction, tensor(targets), tensor(velocities),
                                 tensor(wrenches), tensor(lift), reason)

    def update(self, snapshot: BimanualSnapshot, endpoint_wrenches: Any, dt: float) -> ReducedWireUpdate:
        if not isinstance(snapshot, BimanualSnapshot):
            raise TypeError("snapshot must be BimanualSnapshot")
        zeros = np.zeros((2, 6))
        def reject(reason):
            return self._result(snapshot, self.state, None, zeros, reason)
        try:
            dt = float(dt)
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("invalid dt")
        except (TypeError, ValueError, OverflowError):
            return reject("invalid_dt")
        try:
            if isinstance(endpoint_wrenches, torch.Tensor):
                endpoint_wrenches = endpoint_wrenches.detach().cpu().numpy()
            pairs = ReducedCosseratPredictor._wrench_pair(endpoint_wrenches)
            wrenches = np.stack([np.concatenate((force, moment)) for force, _, moment in pairs])
        except (TypeError, ValueError, OverflowError):
            return reject("invalid_endpoint_wrench")
        try:
            frames = self.frame_provider(snapshot)
            boundaries = [dict(frames[f"{side}_frame"]) for side in ("left", "right")]
            for boundary, (_, normal, _) in zip(boundaries, pairs):
                if normal is not None:
                    boundary["normal"] = normal
                ReducedCosseratPredictor._boundary(boundary, "endpoint")
        except (TypeError, ValueError, KeyError, AttributeError):
            return reject("invalid_endpoint_frame")
        backend_state = None
        if self.backend is not None:
            try:
                if not self.backend.is_bound:
                    self.backend.bind(self._initial_state, **frames)
                step = self.backend.step(**frames)
            except Exception:
                return reject("backend_update_failed")
            try:
                wrenches += endpoint_wrenches_from_impulses(
                    step, max_force_n=self.max_force_n, max_moment_nm=self.max_moment_nm)
                backend_state = step.state
                if not isinstance(backend_state, DloCenterlineState):
                    raise ValueError("invalid backend geometry")
                endpoints = np.stack([boundary["position"] for boundary in boundaries])
                if np.max(np.linalg.norm(backend_state.positions[[0, -1]] - endpoints, axis=1)) > .005:
                    raise ValueError("backend endpoint mismatch")
            except (TypeError, ValueError, AttributeError, OverflowError):
                return reject("invalid_backend_output")
        candidate = copy(self._predictor)
        # predict() uses its stored reset interval; keep it on the current tick.
        candidate._dt = dt
        try:
            state = candidate.step(*boundaries, wrenches, dt)
            if not state.feasible:
                return reject(f"infeasible_state:{state.failure_reason}")
            if backend_state is not None:
                geometry = ReducedCosseratPredictor().reset(backend_state)
                bending = max(state.bending_proxy, geometry.bending_proxy)
                if bending > candidate.max_bending_proxy:
                    return reject("infeasible_state:bending_limit")
                state = replace(state, positions=geometry.positions, tangents=geometry.tangents,
                                bending_proxy=bending, center_height_m=geometry.center_height_m)
                candidate._state = state
            sequence = []
            for index in range(self.horizon_steps):
                sequence.append(tuple({**boundary,
                    "position": state.endpoint_positions[side] + (index + 1) * dt * state.endpoint_velocities[side],
                    "velocity": state.endpoint_velocities[side],
                } for side, boundary in enumerate(boundaries)))
            prediction = candidate.predict(self.horizon_steps, sequence, wrench_sequence=[wrenches])
            # Revalidate at the implementation boundary before publishing any
            # tensors; another predictor implementation may violate its contract.
            if not isinstance(prediction, ReducedCosseratPrediction):
                return reject("invalid_prediction")
            prediction = replace(prediction)
            if prediction.endpoint_positions.shape[0] != self.horizon_steps:
                return reject("invalid_prediction")
            if not bool(prediction.feasible.all()):
                reason = next(reason for reason in prediction.failure_reasons if reason is not None)
                return reject(f"infeasible_prediction:{reason}")
            result = self._result(snapshot, state, prediction, wrenches)
        except (TypeError, ValueError, RuntimeError, OverflowError, StopIteration):
            return reject("invalid_prediction")
        self._predictor = candidate
        return result


__all__ = ["ReducedWireRuntime", "ReducedWireUpdate"]
