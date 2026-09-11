"""Pure exact-budget ledger and candidate selector for the T500 10k sweep."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


TOTAL_STEPS = 10_000


@dataclass
class StepLedger:
    budget_total: int = TOTAL_STEPS
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def consumed(self) -> int:
        return sum(int(entry["executed_steps"]) for entry in self.entries)

    @property
    def remaining(self) -> int:
        return self.budget_total - self.consumed

    def record(self, phase: str, requested_steps: int, executed_steps: int, **metadata: Any) -> None:
        if not phase:
            raise ValueError("phase must be non-empty")
        if any(entry["phase"] == phase for entry in self.entries):
            raise ValueError(f"phase already recorded: {phase}")
        if requested_steps < 0 or executed_steps < 0 or executed_steps > requested_steps:
            raise ValueError("invalid requested/executed step count")
        if executed_steps > self.remaining:
            raise ValueError("simulation step budget exceeded")
        self.entries.append(
            {
                "phase": phase,
                "requested_steps": requested_steps,
                "executed_steps": executed_steps,
                **metadata,
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "budget_total": self.budget_total,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "budget_compliant": self.remaining >= 0,
            "overrun_steps": max(0, -self.remaining),
            "entries": self.entries,
        }
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "StepLedger":
        payload = json.loads(path.read_text(encoding="utf-8"))
        ledger = cls(budget_total=int(payload["budget_total"]))
        ledger.entries = [dict(entry) for entry in payload["entries"]]
        return ledger


def select_candidate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    survivors: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for report in reports:
        trial = report["trials"][0]
        rates = [*trial["arm_mpc_feasible_rates"], *trial["hand_mpc_feasible_rates"]]
        safe = (
            trial["nonfinite_count"] == 0
            and trial["reset_count"] == 0
            and trial["hard_failure_count"] == 0
            and trial["limit_violation_count"] == 0
            and trial["first_safety_failure"] is None
            and all(math.isfinite(float(value)) for value in rates)
        )
        if not safe:
            continue
        key = (
            -min(float(value) for value in rates),
            float(report["metadata"]["tracking_angular_rate_max_rad_s"]),
            float(trial["hold_orientation_error_rad"]),
            float(trial["relative_palm_slip_m"]),
        )
        survivors.append((key, report))
    if not survivors:
        raise RuntimeError("no safe candidate survived")
    return min(survivors, key=lambda item: item[0])[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--record", nargs=3, metavar=("PHASE", "REQUESTED", "EXECUTED"))
    args = parser.parse_args()
    ledger = StepLedger.load(args.state) if args.state.exists() else StepLedger()
    if args.record:
        phase, requested, executed = args.record
        ledger.record(phase, int(requested), int(executed))
        ledger.save(args.state)
    print(json.dumps({"consumed": ledger.consumed, "remaining": ledger.remaining}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
