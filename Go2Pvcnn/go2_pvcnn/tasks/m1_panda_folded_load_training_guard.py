"""Pure eligibility, catastrophe, and atomic acceptance for folded-load PPO."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from go2_pvcnn.tasks.m1_panda_folded_load_curriculum import StageSpec


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def sha256_file(path: str | os.PathLike[str]) -> str:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodeRecord:
    command: tuple[float, float, float]
    steps: int
    time_out: bool
    base_contact: bool
    bad_orientation: bool
    vx_error_sq_sum: float
    wz_error_sq_sum: float
    stationary_abs_vx_sum: float
    stationary_abs_wz_sum: float
    env_id: int = -1


@dataclass(frozen=True)
class EligibilitySnapshot:
    iteration: int
    completed_episodes: int
    time_out_rate: float
    base_contact_rate: float
    bad_orientation_rate: float
    vx_rmse: float
    wz_rmse: float
    stationary_abs_vx: float
    stationary_abs_wz: float
    bucket_counts: tuple[tuple[str, int], ...]
    rank: tuple[float, float, float, int]
    eligible: bool


@dataclass(frozen=True)
class GuardDecision:
    stop: bool
    eligible: bool
    save_best: bool
    reason: str | None
    snapshot: EligibilitySnapshot | None


class FoldedLoadTrainingGuard:
    """Evaluate exact rolling windows while enforcing always-on stop rules."""

    def __init__(self, stage: StageSpec):
        self.stage = stage
        self._episodes: deque[EpisodeRecord] = deque(
            maxlen=stage.completed_episode_window
        )
        self.eligible_best: EligibilitySnapshot | None = None
        self.diagnostic_best: EligibilitySnapshot | None = None
        self.high_failure_updates = 0
        self.medium_failure_updates = 0
        self.patience_without_improvement = 0
        self.updates_observed = 0

    @staticmethod
    def _validated(record: EpisodeRecord) -> EpisodeRecord:
        if not isinstance(record, EpisodeRecord):
            raise TypeError("episodes must contain EpisodeRecord values")
        if not isinstance(record.steps, int) or isinstance(record.steps, bool) or record.steps <= 0:
            raise ValueError("episode steps must be a positive integer")
        if len(record.command) != 3:
            raise ValueError("episode command must contain three values")
        for index, value in enumerate(record.command):
            _finite(value, f"command[{index}]")
        for name in (
            "vx_error_sq_sum",
            "wz_error_sq_sum",
            "stationary_abs_vx_sum",
            "stationary_abs_wz_sum",
        ):
            if _finite(getattr(record, name), name) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        return record

    @staticmethod
    def _hard_failure(record: EpisodeRecord) -> bool:
        return bool(record.base_contact or record.bad_orientation)

    @staticmethod
    def _bucket(records: list[EpisodeRecord], name: str) -> list[EpisodeRecord]:
        if name == "forward":
            return [record for record in records if record.command[0] > 0.0]
        if name == "reverse":
            return [record for record in records if record.command[0] < 0.0]
        if name == "left":
            return [record for record in records if record.command[2] > 0.0]
        if name == "right":
            return [record for record in records if record.command[2] < 0.0]
        if name == "stationary":
            return [
                record for record in records
                if record.command[0] == 0.0 and record.command[2] == 0.0
            ]
        raise KeyError(name)

    @staticmethod
    def _rmse(records: list[EpisodeRecord], field: str) -> float:
        steps = sum(record.steps for record in records)
        if steps <= 0:
            return math.inf
        return math.sqrt(sum(getattr(record, field) for record in records) / steps)

    def _snapshot(self, iteration: int) -> EligibilitySnapshot | None:
        if len(self._episodes) < self.stage.completed_episode_window:
            return None
        records = list(self._episodes)
        count = len(records)
        timeout = sum(record.time_out for record in records) / count
        contact = sum(record.base_contact for record in records) / count
        orientation = sum(record.bad_orientation for record in records) / count
        vx_rmse = self._rmse(records, "vx_error_sq_sum")
        wz_rmse = self._rmse(records, "wz_error_sq_sum")
        buckets = {
            name: self._bucket(records, name)
            for name in ("forward", "reverse", "left", "right")
        }
        directional_pass = True
        for name, values in buckets.items():
            directional_pass &= len(values) >= 25
            if values:
                directional_pass &= (
                    sum(record.base_contact for record in values) / len(values)
                    <= 0.02
                )
                directional_pass &= (
                    sum(record.bad_orientation for record in values) / len(values)
                    <= 0.02
                )
                field = "vx_error_sq_sum" if name in ("forward", "reverse") else "wz_error_sq_sum"
                limit = 0.04 if name in ("forward", "reverse") else 0.12
                directional_pass &= self._rmse(values, field) <= limit
        stationary = self._bucket(records, "stationary")
        stationary_steps = sum(record.steps for record in stationary)
        stationary_vx = (
            sum(record.stationary_abs_vx_sum for record in stationary) / stationary_steps
            if stationary_steps else 0.0
        )
        stationary_wz = (
            sum(record.stationary_abs_wz_sum for record in stationary) / stationary_steps
            if stationary_steps else 0.0
        )
        eligible = bool(
            timeout >= 0.95
            and contact <= 0.02
            and orientation <= 0.02
            and vx_rmse <= 0.04
            and wz_rmse <= 0.12
            and directional_pass
            and stationary_vx <= 0.03
            and stationary_wz <= 0.08
        )
        rank = (
            contact + orientation,
            vx_rmse / 0.04 + wz_rmse / 0.12,
            -timeout,
            int(iteration),
        )
        return EligibilitySnapshot(
            iteration=int(iteration),
            completed_episodes=count,
            time_out_rate=timeout,
            base_contact_rate=contact,
            bad_orientation_rate=orientation,
            vx_rmse=vx_rmse,
            wz_rmse=wz_rmse,
            stationary_abs_vx=stationary_vx,
            stationary_abs_wz=stationary_wz,
            bucket_counts=tuple(sorted((name, len(values)) for name, values in buckets.items())),
            rank=rank,
            eligible=eligible,
        )

    def update(
        self,
        iteration: int,
        episodes: Iterable[EpisodeRecord],
        *,
        finite: bool = True,
        inactive_action_max: float = 0.0,
        fold_hard_failure: bool = False,
    ) -> GuardDecision:
        self.updates_observed += 1
        if not finite:
            return GuardDecision(True, False, False, "nonfinite", None)
        inactive_action_max = _finite(inactive_action_max, "inactive_action_max")
        if inactive_action_max != 0.0:
            return GuardDecision(True, False, False, "inactive_action_leak", None)
        if fold_hard_failure:
            return GuardDecision(True, False, False, "fold_hard_failure", None)

        current = [self._validated(record) for record in episodes]
        if current:
            failure_rate = sum(self._hard_failure(record) for record in current) / len(current)
            self.high_failure_updates = self.high_failure_updates + 1 if failure_rate > 0.50 else 0
            self.medium_failure_updates = self.medium_failure_updates + 1 if failure_rate > 0.20 else 0
            self._episodes.extend(current)
        snapshot = self._snapshot(iteration)
        save_best = False
        improved = False
        if snapshot is not None:
            if self.diagnostic_best is None or snapshot.rank < self.diagnostic_best.rank:
                self.diagnostic_best = snapshot
            if snapshot.eligible and (
                self.eligible_best is None or snapshot.rank < self.eligible_best.rank
            ):
                self.eligible_best = snapshot
                save_best = True
                improved = True
        if self.eligible_best is not None:
            self.patience_without_improvement = 0 if improved else self.patience_without_improvement + 1

        reason = None
        if self.high_failure_updates >= 2:
            reason = "hard_failure_rate_gt_0.50_for_2_updates"
        elif self.medium_failure_updates >= 5:
            reason = "hard_failure_rate_gt_0.20_for_5_updates"
        elif self.patience_without_improvement >= 50:
            reason = "eligible_patience_50_updates"
        elif self.updates_observed >= 600:
            reason = "max_iterations_600"
        return GuardDecision(
            stop=reason is not None,
            eligible=bool(snapshot is not None and snapshot.eligible),
            save_best=save_best,
            reason=reason,
            snapshot=snapshot,
        )


def _atomic_json(target: Path, payload: dict) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return target
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AtomicStageArtifacts:
    """Publish evaluation reports and the accepted checkpoint atomically."""

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_evaluation(self, seed: int, payload: dict) -> Path:
        if seed not in (42, 43, 44):
            raise ValueError("evaluation seed must be 42, 43, or 44")
        document = dict(payload)
        if document.get("seed") != seed or not isinstance(document.get("passed"), bool):
            raise ValueError("evaluation payload must contain matching seed and boolean passed")
        return _atomic_json(self.run_dir / f"evaluation_seed_{seed}.json", document)

    def finalize_evaluations(
        self, best_checkpoint: str | os.PathLike[str], reports: Iterable[Path]
    ) -> dict[str, object]:
        best = Path(best_checkpoint)
        if not best.is_file():
            raise FileNotFoundError(best)
        documents = [json.loads(Path(path).read_text(encoding="utf-8")) for path in reports]
        seeds = sorted(document.get("seed") for document in documents)
        accepted = seeds == [42, 43, 44] and all(
            document.get("passed") is True for document in documents
        )
        decision = {
            "accepted": accepted,
            "seeds": seeds,
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": sha256_file(best),
            "final_checkpoint": None,
            "final_checkpoint_sha256": None,
        }
        if accepted:
            final = self.run_dir / "model_final.pt"
            _atomic_copy(best, final)
            decision["final_checkpoint"] = str(final)
            decision["final_checkpoint_sha256"] = sha256_file(final)
        _atomic_json(self.run_dir / "evaluation_aggregate.json", decision)
        return decision


__all__ = [
    "AtomicStageArtifacts",
    "EligibilitySnapshot",
    "EpisodeRecord",
    "FoldedLoadTrainingGuard",
    "GuardDecision",
    "sha256_file",
]
