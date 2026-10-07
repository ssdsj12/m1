#!/usr/bin/env python3
"""Validate a staged ``wire_000`` contact report without Isaac dependencies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED = (
    "SCENE_TRACKED",
    "APPROACH_ENDPOINTS",
    "CONTACT_LATCHING",
    "COSSERAT_BOUND",
)

REQUIRED_FIELDS = frozenset(
    {
        "object_id",
        "target_prim_path",
        "state",
        "transition_reason",
        "centerline_midpoint_b",
        "tangent_b",
        "radius_m",
        "left_station_b",
        "right_station_b",
        "left_pinch_error_m",
        "right_pinch_error_m",
        "left_contact_mask",
        "right_contact_mask",
        "left_contact_ticks",
        "right_contact_ticks",
        "bind_count",
        "deformx_elapsed_time",
        "wbc_feasible",
        "wire_goal_committed",
    }
)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _ensure_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _ensure_finite(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains a non-JSON value")


def _record(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = REQUIRED_FIELDS.difference(value)
    if missing:
        raise ValueError(f"{path} is missing required fields: {sorted(missing)}")
    return value


def _nonnegative_number(record: dict[str, Any], name: str, path: str) -> float:
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{path}.{name} must be finite and non-negative")
    return normalized


def _nonnegative_integer(record: dict[str, Any], name: str, path: str) -> int:
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}.{name} must be a non-negative integer")
    return value


def validate_report(report: object) -> None:
    """Raise ``ValueError`` unless *report* proves the staged wire contract."""

    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    _ensure_finite(report)
    if report.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    raw_transitions = report.get("transitions")
    if not isinstance(raw_transitions, list) or not raw_transitions:
        raise ValueError("transitions must be a non-empty array")
    transitions = [
        _record(value, f"transitions[{index}]")
        for index, value in enumerate(raw_transitions)
    ]
    final = _record(report.get("final"), "final")
    records = [*transitions, final]

    for index, record in enumerate(records):
        path = "final" if index == len(transitions) else f"transitions[{index}]"
        if record["object_id"] != "wire_000":
            raise ValueError(f"{path}.object_id must be 'wire_000'")
        target_prim_path = record["target_prim_path"]
        if not isinstance(target_prim_path, str) or not target_prim_path.endswith(
            "/wire_000"
        ):
            raise ValueError(f"{path}.target_prim_path must end with '/wire_000'")
        _nonnegative_integer(record, "left_contact_ticks", path)
        _nonnegative_integer(record, "right_contact_ticks", path)
        bind_count = _nonnegative_integer(record, "bind_count", path)
        if bind_count > 1:
            raise ValueError(f"{path}.bind_count must never exceed 1")
        _nonnegative_number(record, "left_pinch_error_m", path)
        _nonnegative_number(record, "right_pinch_error_m", path)
        _nonnegative_number(record, "deformx_elapsed_time", path)

    states = [record["state"] for record in transitions]
    expected_index = 0
    for state in states:
        if expected_index < len(EXPECTED) and state == EXPECTED[expected_index]:
            expected_index += 1
    if expected_index != len(EXPECTED):
        raise ValueError(
            "transitions do not contain the required ordered state subsequence: "
            + " -> ".join(EXPECTED)
        )

    bound_seen = False
    for index, record in enumerate(transitions):
        if record["state"] == "COSSERAT_BOUND":
            bound_seen = True
        if not bound_seen:
            if _nonnegative_number(
                record, "deformx_elapsed_time", f"transitions[{index}]"
            ) != 0.0:
                raise ValueError("pre-bind DeformX elapsed time must be zero")
            if record["bind_count"] != 0:
                raise ValueError("pre-bind bind_count must be zero")
        elif record["bind_count"] != 1:
            raise ValueError("bind_count must equal 1 from COSSERAT_BOUND onward")

    if final["left_contact_ticks"] < 5 or final["right_contact_ticks"] < 5:
        raise ValueError("final contact_ticks must both be at least 5")
    if final["bind_count"] != 1:
        raise ValueError("final bind_count must equal 1")
    if (
        float(final["left_pinch_error_m"]) > 0.005
        or float(final["right_pinch_error_m"]) > 0.005
    ):
        raise ValueError("final pinch errors must both be at most 0.005 m")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with args.report.open("r", encoding="utf-8") as stream:
            report = json.load(stream, parse_constant=_reject_nonfinite)
        validate_report(report)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: staged wire contact report is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
