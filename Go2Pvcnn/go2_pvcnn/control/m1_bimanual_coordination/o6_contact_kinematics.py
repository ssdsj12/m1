"""O6 active/mimic fingertip kinematics and deterministic precontact closure."""

from __future__ import annotations

import math

import torch

from .contracts import BimanualPhase


GENERALIZED_DOF = 59
ACTIVE_HAND_DOF = 6
FINGERTIP_COUNT = 5
FINGER_ACTIVE_COLUMNS = ((0, 1), (2,), (3,), (4,), (5,))


def fold_o6_fingertip_jacobians(
    full_spatial_jacobians: torch.Tensor,
    active_generalized_ids: tuple[int, ...],
    mimic_specs: tuple[tuple[int, int, float], ...],
) -> torch.Tensor:
    """Fold passive mimic motion into six active fingertip linear Jacobian columns."""

    if not isinstance(full_spatial_jacobians, torch.Tensor):
        raise TypeError("full_spatial_jacobians must be a torch.Tensor")
    if full_spatial_jacobians.dtype != torch.float64 or full_spatial_jacobians.device.type != "cpu":
        raise TypeError("full_spatial_jacobians must be a CPU float64 tensor")
    if tuple(full_spatial_jacobians.shape) != (FINGERTIP_COUNT, 6, GENERALIZED_DOF):
        raise ValueError("full_spatial_jacobians must have shape (5, 6, 59)")
    if len(active_generalized_ids) != ACTIVE_HAND_DOF:
        raise ValueError("active_generalized_ids must contain six columns")
    if any(index < 0 or index >= GENERALIZED_DOF for index in active_generalized_ids):
        raise ValueError("active generalized column is out of range")
    linear = full_spatial_jacobians[:, :3, list(active_generalized_ids)].clone()
    for mimic_generalized_id, master_column, multiplier in mimic_specs:
        if mimic_generalized_id < 0 or mimic_generalized_id >= GENERALIZED_DOF:
            raise ValueError("mimic generalized column is out of range")
        if master_column < 0 or master_column >= ACTIVE_HAND_DOF:
            raise ValueError("mimic master column is out of range")
        if not math.isfinite(float(multiplier)):
            raise ValueError("mimic multiplier must be finite")
        linear[:, :, master_column] += (
            float(multiplier)
            * full_spatial_jacobians[:, :3, mimic_generalized_id]
        )
    return linear.reshape(15, ACTIVE_HAND_DOF)


class PrecontactHandController:
    """Hold an open approach pose, then close each uncontacted digit safely."""

    def __init__(
        self,
        *,
        close_rate: float = 0.35,
        preload_dt: float = 0.04,
        open_q: torch.Tensor | None = None,
        preload_q: torch.Tensor | None = None,
    ) -> None:
        if not math.isfinite(close_rate) or close_rate <= 0.0:
            raise ValueError("close_rate must be finite and positive")
        if not math.isfinite(preload_dt) or preload_dt <= 0.0:
            raise ValueError("preload_dt must be finite and positive")
        self.close_rate = float(close_rate)
        self.preload_dt = float(preload_dt)
        self.open_q = (
            torch.tensor([0.10, 0.15, 0.10, 0.10, 0.10, 0.10], dtype=torch.float64)
            if open_q is None
            else open_q.detach().to(device="cpu", dtype=torch.float64).clone()
        )
        self.preload_q = (
            torch.tensor([0.42, 0.55, 0.90, 0.90, 0.90, 0.90], dtype=torch.float64)
            if preload_q is None
            else preload_q.detach().to(device="cpu", dtype=torch.float64).clone()
        )
        if self.open_q.shape != (6,) or self.preload_q.shape != (6,):
            raise ValueError("open_q and preload_q must have shape (6,)")

    def reference(
        self,
        q: torch.Tensor,
        contact_mask: torch.Tensor,
        phase: BimanualPhase,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != torch.float64 or q.device.type != "cpu" or q.shape != (6,):
            raise ValueError("q must be a CPU float64 tensor with shape (6,)")
        if contact_mask.dtype != torch.bool or contact_mask.device.type != "cpu" or contact_mask.shape != (5,):
            raise ValueError("contact_mask must be a CPU bool tensor with shape (5,)")
        if phase is BimanualPhase.APPROACH:
            return self.open_q.clone(), torch.zeros(6, dtype=torch.float64)
        if phase is not BimanualPhase.PRELOAD:
            return q.clone(), torch.zeros(6, dtype=torch.float64)
        target = self.preload_q.clone()
        rate = torch.clamp(
            (target - q) / self.preload_dt,
            min=-self.close_rate,
            max=self.close_rate,
        )
        for finger, active_columns in enumerate(FINGER_ACTIVE_COLUMNS):
            if bool(contact_mask[finger]):
                for column in active_columns:
                    target[column] = q[column]
                    rate[column] = 0.0
        return target, rate


__all__ = ["PrecontactHandController", "fold_o6_fingertip_jacobians"]
