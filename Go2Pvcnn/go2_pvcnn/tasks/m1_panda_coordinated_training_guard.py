"""Pure stability selection and atomic rollback for coordinated M1 + Panda PPO."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from rsl_rl.runners.on_policy_runner import IterationSummary


_METRIC_KEYS = (
    "Termination/time_out",
    "Termination/base_contact",
    "Termination/bad_orientation",
    "Reward/base_target",
    "Reward/ee_tracking",
)


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash an existing regular checkpoint file."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GuardSnapshot:
    iteration: int
    timesteps: int
    completed_episodes: int
    time_out_rate: float
    base_contact_rate: float
    bad_orientation_rate: float
    base_target: float
    ee_tracking: float
    mean_reward: float
    learning_rate: float
    kl_mean: float
    environment_metrics: tuple[tuple[str, float], ...]
    rank: tuple[float, float, float, float, int]
    eligible: bool


@dataclass(frozen=True)
class GuardDecision:
    snapshot: GuardSnapshot | None
    save_best: bool
    stop_reason: str | None


class TrainingGuard:
    """Track rolling episode metrics and select stable checkpoints without I/O."""

    def __init__(
        self,
        *,
        minimum_completed_episodes: int = 100,
        patience_updates: int = 50,
        catastrophe_patience_updates: int = 25,
        max_iterations: int = 600,
    ) -> None:
        if minimum_completed_episodes != 100:
            raise ValueError("minimum_completed_episodes must be exactly 100")
        for name, value in (
            ("patience_updates", patience_updates),
            ("catastrophe_patience_updates", catastrophe_patience_updates),
            ("max_iterations", max_iterations),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.minimum_completed_episodes = minimum_completed_episodes
        self.patience_updates = patience_updates
        self.catastrophe_patience_updates = catastrophe_patience_updates
        self.max_iterations = max_iterations
        self._rewards: deque[float] = deque(maxlen=minimum_completed_episodes)
        self._metrics = {
            key: deque(maxlen=minimum_completed_episodes) for key in _METRIC_KEYS
        }
        self._current_snapshot: GuardSnapshot | None = None
        self.diagnostic_best: GuardSnapshot | None = None
        self.eligible_best: GuardSnapshot | None = None
        self.patience_without_improvement = 0
        self.catastrophe_updates = 0
        self.updates_observed = 0

    @property
    def accepted(self) -> bool:
        return self.eligible_best is not None

    def _append_completed_episodes(self, summary: IterationSummary) -> bool:
        rewards = tuple(
            _finite(value, label="completed reward")
            for value in summary.completed_rewards
        )
        metrics = dict(summary.episode_metrics)
        if not rewards:
            if any(metrics.get(key, ()) for key in _METRIC_KEYS):
                raise ValueError("episode metrics require matching completed rewards")
            return False

        for key in _METRIC_KEYS:
            if key not in metrics:
                raise ValueError(f"missing required episode metric {key!r}")
            values = tuple(
                _finite(value, label=f"episode metric {key!r}")
                for value in metrics[key]
            )
            if len(values) != len(rewards):
                raise ValueError(
                    f"episode metric {key!r} must match completed rewards"
                )
            self._metrics[key].extend(values)
        self._rewards.extend(rewards)
        return True

    def _make_snapshot(self, summary: IterationSummary) -> GuardSnapshot | None:
        buffers = (self._rewards, *self._metrics.values())
        if any(len(values) < self.minimum_completed_episodes for values in buffers):
            return None

        def mean(values) -> float:
            return _finite(sum(values) / len(values), label="rolling metric")

        time_out_rate = mean(self._metrics["Termination/time_out"])
        base_contact_rate = mean(self._metrics["Termination/base_contact"])
        bad_orientation_rate = mean(
            self._metrics["Termination/bad_orientation"]
        )
        base_target = mean(self._metrics["Reward/base_target"])
        ee_tracking = mean(self._metrics["Reward/ee_tracking"])
        mean_reward = mean(self._rewards)
        learning_rate = _finite(summary.learning_rate, label="learning rate")
        kl_mean = _finite(summary.kl_mean, label="KL mean")
        environment_metrics = tuple(
            (str(key), _finite(value, label=f"environment metric {key!r}"))
            for key, value in summary.environment_metrics
        )
        rank = (
            base_contact_rate + bad_orientation_rate,
            -time_out_rate,
            -(base_target + ee_tracking),
            -mean_reward,
            int(summary.iteration),
        )
        eligible = (
            time_out_rate >= 0.90
            and base_contact_rate <= 0.05
            and bad_orientation_rate <= 0.05
        )
        return GuardSnapshot(
            iteration=int(summary.iteration),
            timesteps=int(summary.timesteps),
            completed_episodes=self.minimum_completed_episodes,
            time_out_rate=time_out_rate,
            base_contact_rate=base_contact_rate,
            bad_orientation_rate=bad_orientation_rate,
            base_target=base_target,
            ee_tracking=ee_tracking,
            mean_reward=mean_reward,
            learning_rate=learning_rate,
            kl_mean=kl_mean,
            environment_metrics=environment_metrics,
            rank=rank,
            eligible=eligible,
        )

    def observe(self, summary: IterationSummary) -> GuardDecision:
        """Observe one iteration and return a deterministic checkpoint decision."""
        self.updates_observed += 1
        had_completed_episode = self._append_completed_episodes(summary)
        if had_completed_episode:
            self._current_snapshot = self._make_snapshot(summary)
        snapshot = self._current_snapshot
        save_best = False
        improved_eligible = False

        if had_completed_episode and snapshot is not None:
            if (
                self.diagnostic_best is None
                or snapshot.rank < self.diagnostic_best.rank
            ):
                self.diagnostic_best = snapshot

            if snapshot.eligible:
                if self.eligible_best is None or snapshot.rank < self.eligible_best.rank:
                    self.eligible_best = snapshot
                    improved_eligible = True
                    save_best = True
            elif self.eligible_best is None and self.diagnostic_best is snapshot:
                save_best = True

        if self.eligible_best is not None:
            if improved_eligible:
                self.patience_without_improvement = 0
            else:
                self.patience_without_improvement += 1

            hard_failure = (
                snapshot.base_contact_rate + snapshot.bad_orientation_rate
                if snapshot is not None
                else 0.0
            )
            if hard_failure > 0.20:
                self.catastrophe_updates += 1
            else:
                self.catastrophe_updates = 0

        stop_reason = None
        if self.catastrophe_updates >= self.catastrophe_patience_updates:
            stop_reason = "catastrophe"
        elif self.patience_without_improvement >= self.patience_updates:
            stop_reason = "eligible_patience"
        elif self.updates_observed >= self.max_iterations:
            stop_reason = "max_iterations"

        return GuardDecision(
            snapshot=snapshot,
            save_best=save_best,
            stop_reason=stop_reason,
        )


def _atomic_runner_save(runner, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(name)
        runner.save(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write(target: Path, payload: dict[str, object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class AtomicCheckpointController:
    """Bridge pure guard decisions to atomically published checkpoint artifacts."""

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        guard: TrainingGuard | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.guard = guard if guard is not None else TrainingGuard()
        self.best_checkpoint = self.run_dir / "model_best.pt"
        self.best_metadata = self.run_dir / "best_checkpoint.json"
        self.final_checkpoint = self.run_dir / "model_final.pt"

    def on_iteration(self, runner, summary: IterationSummary) -> str | None:
        decision = self.guard.observe(summary)
        if decision.save_best:
            if decision.snapshot is None:
                raise RuntimeError("save-best decision requires a guard snapshot")
            _atomic_runner_save(runner, self.best_checkpoint)
            checkpoint_sha256 = sha256_file(self.best_checkpoint)
            metadata = asdict(decision.snapshot)
            metadata["environment_metrics"] = dict(
                decision.snapshot.environment_metrics
            )
            metadata["checkpoint"] = str(self.best_checkpoint)
            metadata["checkpoint_sha256"] = checkpoint_sha256
            _atomic_json_write(self.best_metadata, metadata)
        return decision.stop_reason

    def finalize(self, runner, stop_reason: str) -> dict[str, object]:
        selected = self.guard.eligible_best or self.guard.diagnostic_best
        if selected is None:
            _atomic_runner_save(runner, self.final_checkpoint)
            return {
                "status": "completed_without_100_episode_candidate",
                "stop_reason": str(stop_reason),
                "best_iteration": None,
                "rollback_source": None,
                "rollback_source_sha256": None,
                "final_checkpoint": str(self.final_checkpoint),
                "final_checkpoint_sha256": sha256_file(self.final_checkpoint),
                "accepted": False,
            }
        if not self.best_checkpoint.is_file():
            raise RuntimeError("cannot finalize without a valid best checkpoint")

        rollback_source_sha256 = sha256_file(self.best_checkpoint)
        runner.load(
            self.best_checkpoint,
            load_optimizer=False,
            keep_std=True,
        )
        _atomic_runner_save(runner, self.final_checkpoint)
        final_checkpoint_sha256 = sha256_file(self.final_checkpoint)
        accepted = self.guard.eligible_best is not None
        return {
            "status": "accepted" if accepted else "completed_without_eligible_best",
            "stop_reason": str(stop_reason),
            "best_iteration": selected.iteration,
            "rollback_source": str(self.best_checkpoint),
            "rollback_source_sha256": rollback_source_sha256,
            "final_checkpoint": str(self.final_checkpoint),
            "final_checkpoint_sha256": final_checkpoint_sha256,
            "accepted": accepted,
        }


__all__ = [
    "AtomicCheckpointController",
    "GuardDecision",
    "GuardSnapshot",
    "TrainingGuard",
    "sha256_file",
]
