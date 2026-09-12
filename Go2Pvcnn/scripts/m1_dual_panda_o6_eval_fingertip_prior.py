"""Evaluate a prior-off/prior-on report without changing a controller or artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PriorComparisonAcceptance:
    accepted: bool
    reason: str

_P = {"evaluation_manifest_sha256", "scenario_definition_sha256", "trial_set_sha256", "safety_definition_sha256", "controller_contract_sha256", "prior_mode", "prior_config_sha256", "student_artifact_sha256"}
_SHA = re.compile(r"[0-9a-f]{64}")


def accept_prior_comparison(metrics: Mapping[str, object]) -> PriorComparisonAcceptance:
    """Apply the pure comparison gate; every safety/task regression is a rejection."""

    required = {
        "comparison_format_version", "provenance_format_version", "trial_count",
        "nll_improvement_fraction", "prior_off_jerk_p95", "prior_on_jerk_p95",
        "prior_off_task_success", "prior_on_task_success", "prior_off_safety_rejections",
        "prior_on_safety_rejections", "prior_off_provenance", "prior_on_provenance",
    }
    numeric = required - {"comparison_format_version", "provenance_format_version", "trial_count", "prior_off_provenance", "prior_on_provenance"}
    if set(metrics) != required or metrics.get("comparison_format_version") != 1 or metrics.get("provenance_format_version") != 1 or type(metrics.get("trial_count")) is not int or metrics["trial_count"] <= 0 or any(type(metrics[key]) not in (int, float) or not math.isfinite(float(metrics[key])) for key in numeric):
        return PriorComparisonAcceptance(False, "invalid_metrics")
    if any(float(metrics[key]) < 0.0 for key in ("prior_off_jerk_p95", "prior_on_jerk_p95", "prior_off_safety_rejections", "prior_on_safety_rejections")) or any(not 0.0 <= float(metrics[key]) <= 1.0 for key in ("prior_off_task_success", "prior_on_task_success")):
        return PriorComparisonAcceptance(False, "invalid_metrics")
    off_p, on_p = metrics["prior_off_provenance"], metrics["prior_on_provenance"]
    shared = _P - {"prior_mode", "prior_config_sha256", "student_artifact_sha256"}
    if type(off_p) is not dict or type(on_p) is not dict or set(off_p) != _P or set(on_p) != _P or any(type(p[key]) is not str or _SHA.fullmatch(p[key]) is None for p in (off_p, on_p) for key in shared):
        return PriorComparisonAcceptance(False, "invalid_metrics")
    if off_p["prior_mode"] != "disabled" or off_p["prior_config_sha256"] != "0" * 64 or off_p["student_artifact_sha256"] != "0" * 64 or on_p["prior_mode"] != "enabled" or any(type(on_p[key]) is not str or _SHA.fullmatch(on_p[key]) is None or on_p[key] == "0" * 64 for key in ("prior_config_sha256", "student_artifact_sha256")):
        return PriorComparisonAcceptance(False, "invalid_metrics")
    if any(off_p[key] != on_p[key] for key in shared):
        return PriorComparisonAcceptance(False, "provenance_mismatch")
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
    required = {"report_format_version", "executed_fingertip_nll", "jerk_p95", "task_success", "safety_rejections", "trial_count", "provenance"}
    if set(value) != required or value.get("report_format_version") != 1 or type(value.get("trial_count")) is not int or value["trial_count"] <= 0:
        raise ValueError("report schema is invalid")
    for key in ("executed_fingertip_nll", "jerk_p95", "task_success", "safety_rejections"):
        if type(value[key]) not in (int, float) or not math.isfinite(float(value[key])):
            raise ValueError("report values are invalid")
    if value["executed_fingertip_nll"] <= 0.0 or value["jerk_p95"] < 0.0 or value["safety_rejections"] < 0.0 or not 0.0 <= value["task_success"] <= 1.0:
        raise ValueError("report domains are invalid")
    if type(value["provenance"]) is not dict or set(value["provenance"]) != _P or any(type(x) is not str or _SHA.fullmatch(x) is None for x in value["provenance"].values()):
        raise ValueError("report provenance is invalid")
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
    if off["trial_count"] != on["trial_count"]:
        raise ValueError("comparison reports must have matching trial counts")
    metrics = {
        "comparison_format_version": 1, "provenance_format_version": 1,
        "trial_count": off["trial_count"],
        "prior_off_provenance": off["provenance"], "prior_on_provenance": on["provenance"],
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
