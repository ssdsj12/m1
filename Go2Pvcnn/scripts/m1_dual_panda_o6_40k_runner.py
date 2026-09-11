"""Exact env-step ledger and four-GPU launcher for the bimanual lift cycle."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - runner executes on Linux
    fcntl = None


DEFAULT_PHASE_BUDGETS = {
    "contact_preload": 8_000,
    "lift_hold": 12_000,
    "lower_release": 12_000,
    "robustness": 8_000,
}


class RunLock:
    """Hold one non-blocking process lock for the complete budget run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stream = None

    def __enter__(self) -> "RunLock":
        if fcntl is None:
            raise RuntimeError("budget runner locking requires Linux fcntl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError("another budget runner is already active") from error
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._stream is not None
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None


def select_idle_gpus(
    rows: list[dict[str, int]], *, count: int = 4
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    eligible = [
        row
        for row in rows
        if int(row["memory_used_mb"]) <= 2_048
        and int(row["utilization_percent"]) <= 10
    ]
    eligible.sort(
        key=lambda row: (
            int(row["memory_used_mb"]),
            int(row["utilization_percent"]),
            int(row["index"]),
        )
    )
    if len(eligible) < count:
        raise RuntimeError(
            f"need {count} idle GPUs but found {len(eligible)} under resource limits"
        )
    return tuple(int(row["index"]) for row in eligible[:count])


def discover_idle_gpus(*, count: int = 4) -> tuple[int, ...]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, memory, utilization = (int(part.strip()) for part in line.split(","))
        rows.append(
            {
                "index": index,
                "memory_used_mb": memory,
                "utilization_percent": utilization,
            }
        )
    return select_idle_gpus(rows, count=count)


def wait_for_idle_gpus(
    *,
    count: int = 4,
    poll_seconds: float = 30.0,
    discover_fn: Callable[..., tuple[int, ...]] = discover_idle_gpus,
) -> tuple[int, ...]:
    if poll_seconds < 0.0:
        raise ValueError("poll_seconds must be non-negative")
    while True:
        try:
            return discover_fn(count=count)
        except RuntimeError as error:
            print(f"waiting for {count} idle GPUs: {error}", file=sys.stderr, flush=True)
            time.sleep(poll_seconds)


@dataclass(frozen=True)
class StepReservation:
    reservation_id: str
    phase: str
    worker: str
    lane_count: int
    physics_steps: int

    @property
    def env_steps(self) -> int:
        return self.lane_count * self.physics_steps


def _artifact_stem(gpu: int, reservation: StepReservation) -> str:
    return f"gpu{int(gpu)}-{reservation.reservation_id[:12]}"


class StepLedger:
    def __init__(self, *, total_env_steps: int, phase_budgets: dict[str, int]) -> None:
        if total_env_steps <= 0 or sum(phase_budgets.values()) != total_env_steps:
            raise ValueError("phase budgets must be positive and sum to total_env_steps")
        if not phase_budgets or any(value <= 0 for value in phase_budgets.values()):
            raise ValueError("every phase budget must be positive")
        self.total_env_steps = int(total_env_steps)
        self.phase_budgets = {str(key): int(value) for key, value in phase_budgets.items()}
        self._phase_consumed = {key: 0 for key in self.phase_budgets}
        self._reservations: dict[str, StepReservation] = {}
        self._commits: list[dict[str, object]] = []

    @property
    def consumed_env_steps(self) -> int:
        return sum(self._phase_consumed.values())

    @property
    def reserved_env_steps(self) -> int:
        return sum(item.env_steps for item in self._reservations.values())

    def phase_consumed(self, phase: str) -> int:
        return self._phase_consumed[phase]

    def phase_remaining(self, phase: str) -> int:
        reserved = sum(
            item.env_steps for item in self._reservations.values() if item.phase == phase
        )
        return self.phase_budgets[phase] - self._phase_consumed[phase] - reserved

    def reserve(
        self, *, phase: str, worker: str, lane_count: int, physics_steps: int
    ) -> StepReservation:
        if phase not in self.phase_budgets:
            raise KeyError(phase)
        if lane_count <= 0 or physics_steps <= 0:
            raise ValueError("lane_count and physics_steps must be positive")
        env_steps = lane_count * physics_steps
        if env_steps > self.phase_remaining(phase):
            raise ValueError("reservation exceeds remaining phase budget")
        reservation = StepReservation(
            reservation_id=uuid.uuid4().hex,
            phase=phase,
            worker=str(worker),
            lane_count=int(lane_count),
            physics_steps=int(physics_steps),
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def allocate_phase(
        self,
        phase: str,
        *,
        worker_lanes: dict[str, int],
        env_steps: int | None = None,
    ) -> tuple[StepReservation, ...]:
        if not worker_lanes:
            raise ValueError("worker_lanes must not be empty")
        remaining = self.phase_remaining(phase)
        requested = remaining if env_steps is None else int(env_steps)
        if requested <= 0 or requested > remaining:
            raise ValueError("env_steps must be positive and within the remaining phase budget")
        lane_values = tuple(int(value) for value in worker_lanes.values())
        divisor = math.gcd(*lane_values)
        if requested % divisor:
            raise ValueError("phase budget cannot be represented by worker lane counts")
        if len(set(lane_values)) != 1:
            raise ValueError("adaptive run must choose one common lane count across workers")
        lane_count = lane_values[0]
        total_physics_steps, remainder = divmod(requested, lane_count)
        if remainder:
            raise ValueError("phase budget is not divisible by lane count")
        workers = tuple(worker_lanes)
        base, extra = divmod(total_physics_steps, len(workers))
        result = []
        for index, worker in enumerate(workers):
            physics_steps = base + int(index < extra)
            if physics_steps:
                result.append(
                    self.reserve(
                        phase=phase,
                        worker=worker,
                        lane_count=lane_count,
                        physics_steps=physics_steps,
                    )
                )
        return tuple(result)

    def commit(self, reservation_id: str, *, executed_physics_steps: int) -> None:
        if reservation_id not in self._reservations:
            raise KeyError(reservation_id)
        reservation = self._reservations.pop(reservation_id)
        if executed_physics_steps < 0 or executed_physics_steps > reservation.physics_steps:
            self._reservations[reservation_id] = reservation
            raise ValueError("executed physics steps exceed reservation")
        executed_env_steps = int(executed_physics_steps) * reservation.lane_count
        self._phase_consumed[reservation.phase] += executed_env_steps
        self._commits.append(
            {
                **asdict(reservation),
                "executed_physics_steps": int(executed_physics_steps),
                "executed_env_steps": executed_env_steps,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "budget_env_steps": self.total_env_steps,
            "consumed_env_steps": self.consumed_env_steps,
            "remaining_env_steps": self.total_env_steps - self.consumed_env_steps,
            "phase_budgets": self.phase_budgets,
            "phase_consumed": self._phase_consumed,
            "reserved_env_steps": self.reserved_env_steps,
            "reservations": [asdict(item) for item in self._reservations.values()],
            "commits": list(self._commits),
            "budget_compliant": self.consumed_env_steps <= self.total_env_steps,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "StepLedger":
        data = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls(
            total_env_steps=int(data["budget_env_steps"]),
            phase_budgets={
                str(key): int(value)
                for key, value in data["phase_budgets"].items()
            },
        )
        for phase, value in data.get("phase_consumed", {}).items():
            ledger._phase_consumed[str(phase)] = int(value)
        ledger._reservations = {
            item["reservation_id"]: StepReservation(
                reservation_id=str(item["reservation_id"]),
                phase=str(item["phase"]),
                worker=str(item["worker"]),
                lane_count=int(item["lane_count"]),
                physics_steps=int(item["physics_steps"]),
            )
            for item in data.get("reservations", [])
        }
        ledger._commits = list(data.get("commits", []))
        return ledger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(DEFAULT_PHASE_BUDGETS), required=True)
    parser.add_argument("--physical-gpus", nargs="+", type=int)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--budget-env-steps", type=int)
    parser.add_argument("--wait-for-idle-seconds", type=float)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("tests/artifacts/t500_lift_40k_20260908/run-state.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tests/artifacts/t500_lift_40k_20260908"),
    )
    return parser


def _worker_command(
    reservation: StepReservation,
    *,
    report: Path,
    progress: Path,
    seed: int,
) -> list[str]:
    controlled_steps = reservation.physics_steps - 1
    if controlled_steps <= 0:
        raise ValueError("reservation must include reset plus at least one controlled step")
    return [
        sys.executable,
        "scripts/m1_dual_panda_o6_bimanual_probe.py",
        "--headless",
        "--device",
        "cuda:0",
        "--num-envs",
        str(reservation.lane_count),
        "--steps",
        str(controlled_steps),
        "--seed",
        str(seed),
        "--report",
        str(report),
        "--progress",
        str(progress),
    ]


def executed_physics_steps_from_artifacts(
    *,
    report_path: Path,
    progress_path: Path,
    expected_lanes: int,
    reserved_physics_steps: int,
) -> int:
    """Recover the exact completed physics count, preferring the final report."""

    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if int(payload["num_envs"]) != expected_lanes:
            raise RuntimeError("worker lane count differs from reservation")
        trials = payload.get("trials", [])
        if len(trials) == 1:
            executed = int(trials[0]["steps"]) + int(
                payload.get("reset_physics_steps_per_trial", 1)
            )
        else:
            executed = 0
    elif progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if int(payload["num_envs"]) != expected_lanes:
            raise RuntimeError("worker progress lane count differs from reservation")
        executed = int(payload["executed_physics_steps"])
    else:
        executed = 0
    if executed < 0 or executed > reserved_physics_steps:
        raise RuntimeError("worker artifact count exceeds reservation")
    return executed


def main() -> int:
    args = _parser().parse_args()
    lock_path = args.state.with_suffix(args.state.suffix + ".lock")
    with RunLock(lock_path):
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    ledger = StepLedger.load(args.state) if args.state.exists() else StepLedger(
        total_env_steps=40_000, phase_budgets=DEFAULT_PHASE_BUDGETS
    )
    physical_gpus = (
        tuple(args.physical_gpus)
        if args.physical_gpus is not None
        else (
            discover_idle_gpus(count=4)
            if args.wait_for_idle_seconds is None
            else wait_for_idle_gpus(
                count=4, poll_seconds=float(args.wait_for_idle_seconds)
            )
        )
    )
    if not 1 <= len(physical_gpus) <= 4:
        raise ValueError("physical-gpus must contain between one and four indices")
    if len(set(physical_gpus)) != len(physical_gpus) or any(
        gpu < 0 for gpu in physical_gpus
    ):
        raise ValueError("physical-gpus must contain unique non-negative indices")
    worker_lanes = {f"gpu{gpu}": int(args.num_envs) for gpu in physical_gpus}
    reservations = ledger.allocate_phase(
        args.phase,
        worker_lanes=worker_lanes,
        env_steps=args.budget_env_steps,
    )
    ledger.save(args.state)
    phase_root = args.output_root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, (gpu, reservation) in enumerate(
        zip(physical_gpus, reservations, strict=True)
    ):
        artifact_stem = _artifact_stem(gpu, reservation)
        report = phase_root / f"{artifact_stem}-report.json"
        progress = phase_root / f"{artifact_stem}-progress.json"
        report.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
        stdout = (phase_root / f"{artifact_stem}.stdout.log").open("w", encoding="utf-8")
        stderr = (phase_root / f"{artifact_stem}.stderr.log").open("w", encoding="utf-8")
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        }
        process = subprocess.Popen(
            _worker_command(
                reservation,
                report=report,
                progress=progress,
                seed=int(args.seed_base) + index,
            ),
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        jobs.append((reservation, report, progress, stdout, stderr, process))

    failures = 0
    for reservation, report, progress, stdout, stderr, process in jobs:
        return_code = process.wait()
        stdout.close()
        stderr.close()
        executed_physics_steps = executed_physics_steps_from_artifacts(
            report_path=report,
            progress_path=progress,
            expected_lanes=reservation.lane_count,
            reserved_physics_steps=reservation.physics_steps,
        )
        ledger.commit(
            reservation.reservation_id,
            executed_physics_steps=executed_physics_steps,
        )
        ledger.save(args.state)
        if return_code not in {0, 1} or executed_physics_steps == 0:
            failures += 1
    output = ledger.to_dict()
    output["selected_physical_gpus"] = list(physical_gpus)
    print(json.dumps(output, indent=2, sort_keys=True))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
