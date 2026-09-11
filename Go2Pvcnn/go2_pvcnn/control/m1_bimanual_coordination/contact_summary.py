"""Canonical Box-filtered contact facts for the dual O6 hands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class ContactSummary:
    side: str
    selected_names: tuple[str, ...]
    selected_indices: tuple[int, ...]
    filtered_forces_w: torch.Tensor
    raw_forces_w: torch.Tensor
    contact_mask: torch.Tensor
    filtered_force_max_n: float
    raw_force_max_n: float
    consistency_reason: str | None


def summarize_contacts(
    *,
    side: str,
    candidate_names: Sequence[Sequence[str]],
    filtered_forces_w: torch.Tensor,
    raw_forces_w: torch.Tensor,
    threshold_n: float = 0.2,
) -> ContactSummary:
    """Select one body per digit and derive Box contact from filtered force."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if (
        isinstance(threshold_n, bool)
        or not math.isfinite(float(threshold_n))
        or float(threshold_n) <= 0.0
    ):
        raise ValueError("threshold_n must be finite and positive")
    groups = tuple(tuple(group) for group in candidate_names)
    if not groups or any(not group for group in groups):
        raise ValueError("candidate_names must contain non-empty groups")
    filtered = torch.as_tensor(filtered_forces_w, dtype=torch.float64)
    raw = torch.as_tensor(raw_forces_w, dtype=torch.float64)
    if filtered.ndim != 2 or filtered.shape[1] != 3:
        raise ValueError("filtered_forces_w must have shape [body, 3]")
    if raw.shape != filtered.shape:
        raise ValueError("raw_forces_w must have shape [body, 3]")
    if not bool(torch.isfinite(filtered).all()) or not bool(torch.isfinite(raw).all()):
        raise ValueError("contact forces must be finite")
    if sum(len(group) for group in groups) != filtered.shape[0]:
        raise ValueError("candidate_names and force body dimensions differ")

    selected_indices: list[int] = []
    selected_names: list[str] = []
    offset = 0
    for group in groups:
        stop = offset + len(group)
        local_norms = torch.linalg.vector_norm(filtered[offset:stop], dim=1)
        local_index = int(torch.argmax(local_norms).item())
        selected_indices.append(offset + local_index)
        selected_names.append(group[local_index])
        offset = stop

    selected_filtered = filtered[selected_indices].clone()
    selected_raw = raw[selected_indices].clone()
    filtered_norms = torch.linalg.vector_norm(selected_filtered, dim=1)
    raw_norms = torch.linalg.vector_norm(selected_raw, dim=1)
    contact_mask = filtered_norms > float(threshold_n)
    filtered_max = float(filtered_norms.max().item())
    raw_max = float(raw_norms.max().item())
    consistency_reason = None
    if filtered_max <= float(threshold_n) < raw_max:
        consistency_reason = "raw_contact_without_box_contact"
    return ContactSummary(
        side=side,
        selected_names=tuple(selected_names),
        selected_indices=tuple(selected_indices),
        filtered_forces_w=selected_filtered,
        raw_forces_w=selected_raw,
        contact_mask=contact_mask.clone(),
        filtered_force_max_n=filtered_max,
        raw_force_max_n=raw_max,
        consistency_reason=consistency_reason,
    )
