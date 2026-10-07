from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.wire_coupling import (
    WireCouplingDiagnostics,
)
from scripts.m1_dual_panda_o6_bimanual_play import (
    _WireReportRecorder,
    _write_wire_report_atomic,
)
from tests.test_m1_bimanual_runtime import _staged_runtime, _wire_tick


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_m1_wire_contact_report.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("wire_contact_report_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostics(
    state: str,
    *,
    transition_reason: str | None = None,
    left_contact_ticks: int = 0,
    right_contact_ticks: int = 0,
    bind_count: int = 0,
    deformx_elapsed_time: float = 0.0,
) -> WireCouplingDiagnostics:
    return WireCouplingDiagnostics(
        object_id="wire_000",
        target_prim_path="{ENV_REGEX_NS}/wire_000",
        state=state,
        transition_reason=transition_reason or f"entered_{state.lower()}",
        centerline_midpoint_b=(0.5, 0.0, 1.2),
        tangent_b=(1.0, 0.0, 0.0),
        radius_m=0.004,
        left_station_b=(0.476, 0.0, 1.2),
        right_station_b=(0.524, 0.0, 1.2),
        left_pinch_error_m=0.001,
        right_pinch_error_m=0.002,
        left_contact_mask=(True, True, False, False, False),
        right_contact_mask=(True, True, False, False, False),
        left_contact_ticks=left_contact_ticks,
        right_contact_ticks=right_contact_ticks,
        bind_count=bind_count,
        deformx_elapsed_time=deformx_elapsed_time,
        wbc_feasible=True,
        wire_goal_committed=True,
    )


def _valid_report() -> dict[str, object]:
    transitions = [
        _diagnostics("SCENE_TRACKED").to_json(),
        _diagnostics("APPROACH_ENDPOINTS").to_json(),
        _diagnostics(
            "CONTACT_LATCHING",
            left_contact_ticks=1,
            right_contact_ticks=1,
        ).to_json(),
        _diagnostics(
            "COSSERAT_BOUND",
            left_contact_ticks=5,
            right_contact_ticks=5,
            bind_count=1,
        ).to_json(),
    ]
    return {"schema_version": 1, "transitions": transitions, "final": transitions[-1]}


def test_diagnostics_are_frozen_and_json_safe():
    diagnostics = _diagnostics(
        "COSSERAT_BOUND",
        left_contact_ticks=5,
        right_contact_ticks=5,
        bind_count=1,
    )

    with pytest.raises(FrozenInstanceError):
        diagnostics.bind_count = 2

    payload = diagnostics.to_json()
    required = {
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
    assert required <= payload.keys()
    assert isinstance(payload["centerline_midpoint_b"], list)
    assert isinstance(payload["left_contact_mask"], list)
    json.dumps(payload, allow_nan=False)


def test_diagnostics_reject_nonfinite_values_before_json_publication():
    with pytest.raises(ValueError, match="finite"):
        replace(_diagnostics("SCENE_TRACKED"), radius_m=float("nan"))


def test_runtime_publishes_immutable_complete_diagnostics_without_prebind_elapsed_time():
    runtime, backend = _staged_runtime()

    _wire_tick(runtime, 1, left=False, right=False)

    diagnostics = runtime.latest_wire_diagnostics
    assert isinstance(diagnostics, WireCouplingDiagnostics)
    assert diagnostics.state == "APPROACH_ENDPOINTS"
    assert diagnostics.deformx_elapsed_time == 0.0
    assert backend.bind_calls == backend.step_calls == 0
    json.dumps(diagnostics.to_json(), allow_nan=False)


def test_report_recorder_keeps_only_state_transitions_and_latest_final():
    recorder = _WireReportRecorder()
    recorder.record(_diagnostics("SCENE_TRACKED"))
    recorder.record(_diagnostics("SCENE_TRACKED", transition_reason="state_unchanged"))
    recorder.record(_diagnostics("APPROACH_ENDPOINTS"))
    latest = _diagnostics("APPROACH_ENDPOINTS", transition_reason="state_unchanged")
    recorder.record(latest)

    report = recorder.to_json()

    assert [item["state"] for item in report["transitions"]] == [
        "SCENE_TRACKED",
        "APPROACH_ENDPOINTS",
    ]
    assert report["final"] == latest.to_json()
    json.dumps(report, allow_nan=False)


def test_wire_report_is_replaced_atomically(tmp_path: Path):
    destination = tmp_path / "nested" / "wire.json"
    destination.parent.mkdir()
    destination.write_text('{"old": true}\n', encoding="utf-8")

    report = _valid_report()
    _write_wire_report_atomic(destination, report)

    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert list(destination.parent.iterdir()) == [destination]


def test_validator_accepts_complete_ordered_wire_report():
    verifier = _load_verifier()

    assert verifier.validate_report(_valid_report()) is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["final"].__setitem__("object_id", "box_000"),
            "object_id",
        ),
        (
            lambda report: report["final"].__setitem__(
                "target_prim_path", "{ENV_REGEX_NS}/Box"
            ),
            "target_prim_path",
        ),
        (
            lambda report: report["transitions"].pop(1),
            "ordered",
        ),
        (
            lambda report: report["final"].__setitem__("radius_m", float("nan")),
            "finite",
        ),
        (
            lambda report: report["final"].__setitem__("bind_count", 2),
            "bind_count",
        ),
        (
            lambda report: (
                report.__setitem__("final", dict(report["final"])),
                report["transitions"][-1].__setitem__("bind_count", 0),
            ),
            "bind_count",
        ),
        (
            lambda report: report["final"].__setitem__(
                "left_pinch_error_m", 0.0050001
            ),
            "pinch",
        ),
        (
            lambda report: report["final"].__setitem__(
                "right_pinch_error_m", 0.0050001
            ),
            "pinch",
        ),
        (
            lambda report: report["final"].__setitem__("left_contact_ticks", 4),
            "contact_ticks",
        ),
        (
            lambda report: report["final"].__setitem__("right_contact_ticks", 4),
            "contact_ticks",
        ),
        (
            lambda report: report["transitions"][1].__setitem__(
                "deformx_elapsed_time", 1.0e-6
            ),
            "pre-bind",
        ),
    ],
)
def test_validator_rejects_invalid_wire_reports(mutate, message):
    verifier = _load_verifier()
    report = _valid_report()
    mutate(report)

    with pytest.raises(ValueError, match=message):
        verifier.validate_report(report)


def test_validator_rejects_wrong_object_or_prim_in_transition_history():
    verifier = _load_verifier()
    report = _valid_report()
    report["transitions"][0]["object_id"] = "wire_001"

    with pytest.raises(ValueError, match="object_id"):
        verifier.validate_report(report)
