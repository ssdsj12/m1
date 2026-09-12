"""Evaluate a prior-off/prior-on report without changing a controller or artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PriorComparisonAcceptance:
    accepted: bool
    reason: str


def accept_prior_comparison(metrics: Mapping[str, object]) -> PriorComparisonAcceptance:
    """Apply the pure comparison gate; every safety/task regression is a rejection."""

    required = {
        "nll_improvement_fraction", "prior_off_jerk_p95", "prior_on_jerk_p95",
        "prior_off_task_success", "prior_on_task_success", "prior_off_safety_rejections",
        "prior_on_safety_rejections",
    }
    if set(metrics) != required or any(type(metrics[key]) not in (int, float) or not math.isfinite(float(metrics[key])) for key in required):
        return PriorComparisonAcceptance(False, "invalid_metrics")
    if float(metrics["nll_improvement_fraction"]) < 0.10:
        return PriorComparisonAcceptance(False, "nll_improvement_gate")
    if float(metrics["prior_on_jerk_p95"]) > float(metrics["prior_off_jerk_p95"]):
        return PriorComparisonAcceptance(False, "jerk_regression")
    if float(metrics["prior_on_task_success"]) < float(metrics["prior_off_task_success"]):
        return PriorComparisonAcceptance(False, "task_success_regression")
    if float(metrics["prior_on_safety_rejections"]) > float(metrics["prior_off_safety_rejections"]):
        return PriorComparisonAcceptance(False, "safety_regression")
    return PriorComparisonAcceptance(True, "accepted")


def _report(path: str) -> dict[str, object]:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError("report must be a regular JSON file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("report must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline fingertip-prior comparison gate.")
    parser.add_argument("--prior-off-report", required=True)
    parser.add_argument("--prior-on-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    off, on = _report(args.prior_off_report), _report(args.prior_on_report)
    metrics = {
        "nll_improvement_fraction": (float(off["executed_fingertip_nll"]) - float(on["executed_fingertip_nll"])) / float(off["executed_fingertip_nll"]),
        "prior_off_jerk_p95": off["jerk_p95"], "prior_on_jerk_p95": on["jerk_p95"],
        "prior_off_task_success": off["task_success"], "prior_on_task_success": on["task_success"],
        "prior_off_safety_rejections": off["safety_rejections"], "prior_on_safety_rejections": on["safety_rejections"],
    }
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("comparison output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "acceptance": asdict(accept_prior_comparison(metrics))}, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
