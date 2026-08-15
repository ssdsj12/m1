"""Pure full-scale evaluation helpers for M1 + Panda Teacher checkpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


TERMINATION_NAMES = ("bad_orientation", "base_contact", "time_out")
EXPECTED_EVALUATION_SEEDS = {42, 43, 44}
MIN_FORCE_AXIS_N = 19.0
MIN_TORQUE_AXIS_NM = 4.75


def _finite_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


class TeacherEvaluationAccumulator:
    """Accumulate rewards and termination events across all evaluation steps."""

    def __init__(self, *, num_envs: int) -> None:
        if (
            not isinstance(num_envs, int)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.reward_sum = 0.0
        self.reward_count = 0
        self.termination_counts = {name: 0 for name in TERMINATION_NAMES}

    def update(
        self,
        *,
        rewards: torch.Tensor,
        termination_counts: dict[str, int],
    ) -> None:
        if (
            not isinstance(rewards, torch.Tensor)
            or tuple(rewards.shape) != (self.num_envs,)
            or not bool(torch.isfinite(rewards).all())
        ):
            raise ValueError(
                "rewards must be finite with shape (num_envs,)"
            )
        if not isinstance(termination_counts, dict):
            raise TypeError("termination_counts must be a dictionary")
        if set(termination_counts) != set(TERMINATION_NAMES):
            raise ValueError("termination_counts must contain all exact terms")
        self.reward_sum += float(rewards.sum().item())
        self.reward_count += int(rewards.numel())
        for name in TERMINATION_NAMES:
            value = termination_counts[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"termination count {name} must be nonnegative"
                )
            self.termination_counts[name] += value

    def finalize(
        self,
        *,
        checkpoint: str,
        checkpoint_sha256: str,
        base_checkpoint_sha256: str,
        seed: int,
        steps: int,
        curriculum_scale: float,
        axis_abs_wrench_seen: Sequence[float],
        frozen_actor_sha256: str,
    ) -> dict[str, object]:
        if self.reward_count == 0:
            raise RuntimeError("evaluation must contain at least one reward")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        axes = [
            _finite_number(value, label=f"axis_abs_wrench_seen[{index}]")
            for index, value in enumerate(axis_abs_wrench_seen)
        ]
        if len(axes) != 6:
            raise ValueError("axis_abs_wrench_seen must contain six values")
        total = sum(self.termination_counts.values())
        rates = {
            name: (count / total if total else 0.0)
            for name, count in self.termination_counts.items()
        }
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": str(checkpoint_sha256),
            "base_checkpoint_sha256": str(base_checkpoint_sha256),
            "frozen_actor_sha256": str(frozen_actor_sha256),
            "seed": seed,
            "steps": steps,
            "num_envs": self.num_envs,
            "curriculum_scale": _finite_number(
                curriculum_scale, label="curriculum_scale"
            ),
            "axis_abs_wrench_seen": axes,
            "reward_sum": self.reward_sum,
            "reward_count": self.reward_count,
            "mean_reward": self.reward_sum / self.reward_count,
            "termination_counts": dict(self.termination_counts),
            "termination_rates": rates,
            "finite": True,
        }


def validate_full_scale_summary(summary: dict[str, object]) -> None:
    """Reject evaluation rows that do not prove the full-scale contract."""
    if not isinstance(summary, dict):
        raise TypeError("summary must be a dictionary")
    if summary.get("finite") is not True:
        raise ValueError("summary finite must be true")
    scale = _finite_number(
        summary.get("curriculum_scale"), label="curriculum_scale"
    )
    if scale != 1.0:
        raise ValueError("curriculum_scale must be exactly 1.0")
    axes_value = summary.get("axis_abs_wrench_seen")
    if not isinstance(axes_value, (list, tuple)) or len(axes_value) != 6:
        raise ValueError("axis_abs_wrench_seen must contain six values")
    axes = [
        _finite_number(value, label=f"axis_abs_wrench_seen[{index}]")
        for index, value in enumerate(axes_value)
    ]
    for index, value in enumerate(axes[:3]):
        if value < MIN_FORCE_AXIS_N:
            raise ValueError(
                f"force axis {index} maximum {value} is below {MIN_FORCE_AXIS_N}"
            )
    for index, value in enumerate(axes[3:]):
        if value < MIN_TORQUE_AXIS_NM:
            raise ValueError(
                f"torque axis {index} maximum {value} is below "
                f"{MIN_TORQUE_AXIS_NM}"
            )
    for field in (
        "checkpoint_sha256",
        "base_checkpoint_sha256",
        "frozen_actor_sha256",
    ):
        value = summary.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a nonempty string")


def aggregate_candidate_summaries(
    summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Aggregate the exact three-seed evidence for one candidate checkpoint."""
    if not isinstance(summaries, (list, tuple)) or not summaries:
        raise ValueError("summaries must be a nonempty sequence")
    for summary in summaries:
        validate_full_scale_summary(summary)
    checkpoints = {str(summary.get("checkpoint")) for summary in summaries}
    checkpoint_hashes = {
        str(summary.get("checkpoint_sha256")) for summary in summaries
    }
    if len(checkpoints) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("summaries must describe one checkpoint")
    seeds = {summary.get("seed") for summary in summaries}
    if seeds != EXPECTED_EVALUATION_SEEDS or len(summaries) != 3:
        raise ValueError("summaries must contain exact seeds 42, 43, and 44")

    counts = {name: 0 for name in TERMINATION_NAMES}
    reward_sum = 0.0
    reward_count = 0
    for summary in summaries:
        row_counts = summary.get("termination_counts")
        if not isinstance(row_counts, dict) or set(row_counts) != set(
            TERMINATION_NAMES
        ):
            raise ValueError("termination_counts contract mismatch")
        for name in TERMINATION_NAMES:
            value = row_counts[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"termination count {name} is invalid")
            counts[name] += value
        row_reward_sum = _finite_number(
            summary.get("reward_sum"), label="reward_sum"
        )
        row_reward_count = summary.get("reward_count")
        if (
            not isinstance(row_reward_count, int)
            or isinstance(row_reward_count, bool)
            or row_reward_count <= 0
        ):
            raise ValueError("reward_count must be a positive integer")
        reward_sum += row_reward_sum
        reward_count += row_reward_count

    total = sum(counts.values())
    if total <= 0:
        raise ValueError("candidate must contain at least one termination event")
    timeout_rate = counts["time_out"] / total
    contact_rate = counts["base_contact"] / total
    orientation_rate = counts["bad_orientation"] / total
    mean_reward = reward_sum / reward_count
    return {
        "checkpoint": next(iter(checkpoints)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "seeds": sorted(EXPECTED_EVALUATION_SEEDS),
        "termination_counts": counts,
        "timeout_survival_rate": timeout_rate,
        "base_contact_rate": contact_rate,
        "bad_orientation_rate": orientation_rate,
        "reward_sum": reward_sum,
        "reward_count": reward_count,
        "mean_reward": mean_reward,
        "accepted": (
            timeout_rate >= 0.80
            and contact_rate <= 0.10
            and orientation_rate <= 0.10
        ),
        "rank_key": [
            timeout_rate,
            -contact_rate,
            -orientation_rate,
            mean_reward,
        ],
    }


__all__ = [
    "EXPECTED_EVALUATION_SEEDS",
    "TERMINATION_NAMES",
    "TeacherEvaluationAccumulator",
    "aggregate_candidate_summaries",
    "validate_full_scale_summary",
]
