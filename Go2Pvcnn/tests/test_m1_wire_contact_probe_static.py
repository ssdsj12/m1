from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.wire_contact_acceptance import (
    WireAcceptanceConfig,
    WireAcceptancePhase,
    WireAcceptanceTracker,
)
from scripts import m1_dual_panda_o6_bimanual_play as play


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_wire_contact_probe.py"


def _probe():
    assert PROBE.is_file(), "missing deterministic wire probe entrypoint"
    return importlib.import_module("scripts.m1_wire_contact_probe")


def _state(**changes):
    return replace(WireAcceptanceTracker(WireAcceptanceConfig()).state, **changes)


def _diagnostics(**changes):
    row = {
        "frame": "m1_base",
        "mission_phase": "PRELOAD",
        "mission_phase_elapsed_s": 0.1,
        "left_valid_contacts": 0,
        "right_valid_contacts": 0,
        "left_contact_mask": [False] * 5,
        "right_contact_mask": [False] * 5,
        "left_fingertip_distance_m": [0.0] * 5,
        "right_fingertip_distance_m": [0.0] * 5,
        "left_fingertip_forces_b": [[0.0] * 3 for _ in range(5)],
        "right_fingertip_forces_b": [[0.0] * 3 for _ in range(5)],
        "left_contact_normals_b": [[0.0] * 3 for _ in range(5)],
        "right_contact_normals_b": [[0.0] * 3 for _ in range(5)],
        "left_slip_m": 0.0,
        "right_slip_m": 0.0,
        "left_palm_error_m": 0.0,
        "right_palm_error_m": 0.0,
        "wire_center_height_m": 1.2,
        "feasibility": {key: True for key in ("object", "arm", "left_hand", "right_hand", "wbc", "safety")},
        "fallback_reasons": [],
        "success": True,  # Only the acceptance state may authorize success.
    }
    row.update(changes)
    return row


def _recorder():
    assert hasattr(play, "_WireContactReportRecorder"), "missing sampled contact recorder"
    recorder = play._WireContactReportRecorder(object_id="wire_000", seed=42)
    recorder.reset(wire_center_height_m=1.2, control_dt_s=0.005)
    return recorder


def test_probe_requires_explicit_object_and_report_with_fixed_seed():
    parser = _probe()._parser()
    for argv in ([], ["--object-id", "wire_000"], ["--wire-report", "wire.jsonl"]):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(argv)
        assert error.value.code == 2
    args = parser.parse_args(["--object-id", "wire_000", "--wire-report", "wire.jsonl"])
    assert (args.seed, args.max_steps) == (42, 4000)
    assert args.wire_contact_probe
    for value in ("0", "-1"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--object-id", "wire_000", "--wire-report", "wire.jsonl", "--max-steps", value])


def test_probe_delegates_startup_and_binding_to_play(monkeypatch):
    probe = _probe()
    calls = []
    monkeypatch.setattr(play, "main", lambda parser=None: calls.append(parser) or 0)
    assert probe.main() == 0
    assert len(calls) == 1
    assert calls[0].parse_args(["--object-id", "wire_000", "--wire-report", "wire.jsonl"]).wire_contact_probe
    source = PROBE.read_text(encoding="utf-8")
    assert "AppLauncher(" not in source
    assert "gym.make(" not in source
    assert "PreparedO6FingertipPriors.from_artifact(" not in source


def test_probe_default_asset_root_resolves_the_shipped_wire_catalog():
    args = _probe()._parser().parse_args(["--object-id", "wire_000", "--wire-report", "wire.jsonl"])
    assert args.object_assets_root == ROOT / "assets/m1_objects"
    catalog, instances = play._prepare_object_scene(
        object_id=args.object_id, object_catalog=args.object_catalog,
        object_assets_root=args.object_assets_root,
    )
    assert instances[0].object_class == "wire"
    assert catalog.resolve("wire").object_class == "wire"


def test_shared_launcher_parser_accepts_headless_before_launch(monkeypatch):
    from isaaclab.app import AppLauncher

    parser = _probe()._parser()
    monkeypatch.setattr(sys, "argv", [str(PROBE), "--object-id", "wire_000", "--wire-report", "wire.jsonl"])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--object-id", "wire_000", "--wire-report", "wire.jsonl", "--headless"])
    assert args.headless is True


def test_samples_preserve_force_measurements_and_never_infer_contact_from_distance():
    recorder = _recorder()
    diagnostics = _diagnostics()
    recorder.record(step=1, diagnostics=diagnostics, acceptance_state=_state())
    diagnostics["left_contact_mask"][0] = True
    final = recorder.finish("max_steps")
    assert final["left_valid_contacts"] == final["right_valid_contacts"] == 0
    assert not any(final["left_contact_mask"])
    assert final["left_fingertip_distance_m"] == [0.0] * 5
    assert final["left_fingertip_forces_b"] == [[0.0] * 3 for _ in range(5)]
    assert final["success"] is False
    assert final["terminal_reason"] == "max_steps"
    assert final["elapsed_simulation_time_s"] == pytest.approx(0.005)
    assert final["object_id"] == "wire_000" and final["seed"] == 42
    assert [row["record_type"] for row in recorder.records] == ["sample", "final"]


def test_final_includes_lift_hold_contacts_and_exact_tracker_reason():
    recorder = _recorder()
    state = _state(phase=WireAcceptancePhase.SUCCEEDED, bilateral_contact_elapsed_s=0.5,
                   lift_elapsed_s=1.2, hold_elapsed_s=1.0, lift_permitted=True, success=True)
    recorder.record(step=2, diagnostics=_diagnostics(left_valid_contacts=2, right_valid_contacts=3,
                    wire_center_height_m=1.26), acceptance_state=state)
    final = recorder.finish("accepted")
    assert final["success"] is True
    assert final["lift_height_m"] == pytest.approx(0.06)
    assert final["hold_elapsed_s"] == 1.0
    assert final["bilateral_contact_elapsed_s"] == 0.5
    assert final["max_left_valid_contacts"] == 2
    assert final["max_right_valid_contacts"] == 3
    assert final["max_lift_height_m"] == pytest.approx(0.06)
    assert final["max_hold_elapsed_s"] == 1.0
    assert final["terminal_reason"] == "accepted"
    assert recorder.finish("max_steps") == final
    assert len(recorder.records) == 2


def test_nonfinite_measurements_are_explicit_nulls_in_strict_json():
    recorder = _recorder()
    diagnostics = _diagnostics(left_fingertip_forces_b=[[float("nan"), 0.0, 0.0]] * 5,
                               right_slip_m=float("inf"))
    recorder.record(step=1, diagnostics=diagnostics,
                    acceptance_state=_state(phase=WireAcceptancePhase.TERMINATED,
                                            terminal_reason="nonfinite_observation"))
    final = recorder.finish("nonfinite_observation")
    json.dumps(recorder.records, allow_nan=False)
    assert final["left_fingertip_forces_b"][0][0] is None
    assert final["right_slip_m"] is None
    assert "left_fingertip_forces_b[0][0]" in final["nonfinite_fields"]
    assert "right_slip_m" in final["nonfinite_fields"]
    assert final["terminal_reason"] == "nonfinite_observation"
    assert final["success"] is False


def test_atomic_jsonl_replacement_and_failed_replace_preserve_old_report(tmp_path, monkeypatch):
    assert hasattr(play, "_write_wire_contact_report_atomic"), "missing atomic JSONL writer"
    destination = tmp_path / "nested" / "wire.jsonl"
    recorder = _recorder()
    recorder.record(step=1, diagnostics=_diagnostics(), acceptance_state=_state())
    recorder.finish("max_steps")
    play._write_wire_contact_report_atomic(destination, recorder.records)
    original = destination.read_bytes()
    assert [json.loads(line) for line in original.splitlines()] == recorder.records
    assert list(destination.parent.iterdir()) == [destination]
    def fail_replace(source, target):
        assert Path(source).parent == destination.parent
        assert Path(target) == destination
        assert destination.read_bytes() == original
        raise OSError("replace failure")
    monkeypatch.setattr(play.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        play._write_wire_contact_report_atomic(destination, [{"changed": True}])
    assert destination.read_bytes() == original
    assert list(destination.parent.iterdir()) == [destination]


class _Application:
    def __init__(self):
        self.running = True
        self.updates = 0

    def is_running(self):
        return self.running

    def update(self):
        self.updates += 1
        self.running = False


def _wrapper(state=None, fail=False):
    wrapper = SimpleNamespace(
        runtime=SimpleNamespace(mission=SimpleNamespace(phase=SimpleNamespace(name="PRELOAD")),
                                latest_wire_diagnostics=None),
        env=SimpleNamespace(unwrapped=SimpleNamespace(physics_dt=0.005)),
        wire_acceptance_state=state,
        wire_acceptance_diagnostics=_diagnostics(),
        steps=0,
    )
    wrapper.reset = lambda *, seed: SimpleNamespace(box=SimpleNamespace(pose_b=[0., 0., 1.2, 0., 0., 0.]))
    def step():
        wrapper.steps += 1
        if fail:
            raise RuntimeError("physics failed")
    wrapper.step = step
    return wrapper


@pytest.mark.parametrize("state, reason", [
    (_state(phase=WireAcceptancePhase.TERMINATED, terminal_reason="safety_rejected"), "safety_rejected"),
    (_state(phase=WireAcceptancePhase.SUCCEEDED, success=True, hold_elapsed_s=1.0), "accepted"),
    (_state(), "max_steps"),
])
def test_episode_records_every_control_interval_and_stops_on_acceptance(state, reason):
    assert hasattr(play, "_play_episode"), "missing acceptance-aware episode loop"
    wrapper = _wrapper(state)
    recorder = _recorder()
    args = SimpleNamespace(seed=42, max_steps=3, diagnostics=False)
    observed = play._play_episode(wrapper, _Application(), args, play._WireReportRecorder(), recorder)
    assert observed == reason
    expected_steps = 3 if reason == "max_steps" else 1
    assert wrapper.steps == expected_steps
    assert len(recorder.records) == expected_steps


def test_interactive_terminal_pauses_physics_and_keeps_window_open():
    assert hasattr(play, "_keep_play_open"), "missing interactive safe-terminal retention"
    app = _Application()
    events = []
    wrapper = _wrapper()
    wrapper.env.unwrapped.sim = SimpleNamespace(pause=lambda: events.append("pause"))
    play._keep_play_open(wrapper, app, headless=False)
    assert events == ["pause"] and app.updates == 1 and wrapper.steps == 0
    app = _Application()
    events.clear()
    play._keep_play_open(wrapper, app, headless=True)
    assert events == [] and app.updates == 0 and wrapper.steps == 0


def test_startup_error_can_be_reported_without_fabricated_measurements():
    assert hasattr(play, "_WireContactReportRecorder"), "missing sampled contact recorder"
    recorder = play._WireContactReportRecorder(object_id="wire_000", seed=42)
    final = recorder.finish("runtime_error:RuntimeError")
    assert final["success"] is False
    assert final["step"] == 0
    assert final["wire_center_height_m"] is None
    assert final["lift_height_m"] is None
    assert final["terminal_reason"] == "runtime_error:RuntimeError"
    json.dumps(final, allow_nan=False)


def _stub_isaac_startup(monkeypatch, tmp_path, *, failure=None, object_class="wire"):
    # Only the Isaac/Gym boundary is substituted; parsing, loop, serialization,
    # publication, and terminal handling run through the shipped Play code.
    launcher_module = ModuleType("isaaclab.app")

    app = _Application()
    wrapper = _wrapper(_state(phase=WireAcceptancePhase.TERMINATED,
                              terminal_reason="safety_rejected"), fail=failure == "step")
    launches = []
    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true")

        def __init__(self, args):
            launches.append(args)
            if failure == "startup":
                raise RuntimeError("launcher failed")
            self.app = app
    launcher_module.AppLauncher = Launcher
    monkeypatch.setitem(sys.modules, "isaaclab.app", launcher_module)
    tasks = ModuleType("go2_pvcnn.tasks")
    tasks.__path__ = []
    module = ModuleType("go2_pvcnn.tasks.m1_dual_panda_o6_bimanual_wrapper")
    module.M1DualPandaO6BimanualWrapper = lambda *args, **kwargs: nullcontext(wrapper)
    monkeypatch.setitem(sys.modules, tasks.__name__, tasks)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    gym = ModuleType("gymnasium")
    gym.make = lambda *args, **kwargs: wrapper.env
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setattr(play, "_prepare_object_scene", lambda **kwargs:
                        (object(), (SimpleNamespace(object_id="wire_000", object_class=object_class),)))
    monkeypatch.setattr(play, "build_play_config", lambda **kwargs:
                        SimpleNamespace(target_contact_prim_path="{ENV_REGEX_NS}/Wire"))
    destination = tmp_path / "result.jsonl"
    monkeypatch.setattr(sys, "argv", [str(PROBE), "--object-id", "wire_000", "--wire-report",
                                     str(destination), "--max-steps", "3", "--headless"])
    parser = _probe()._parser()
    parser.set_defaults(backend_selection={})
    return parser, destination, wrapper, launches


def test_shared_startup_publishes_headless_final_json_and_stops_safely(monkeypatch, tmp_path, capsys):
    parser, destination, wrapper, _ = _stub_isaac_startup(monkeypatch, tmp_path)
    assert play._run_with_fingertip_prior(parser, None) == 0
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["record_type"] for row in rows] == ["sample", "final"]
    assert rows[-1]["terminal_reason"] == "safety_rejected"
    assert rows[-1]["success"] is False
    assert wrapper.steps == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == rows[-1]


@pytest.mark.parametrize("failure", ["startup", "step"])
def test_shared_startup_preserves_machine_readable_runtime_failure(monkeypatch, tmp_path, capsys, failure):
    parser, destination, _, _ = _stub_isaac_startup(monkeypatch, tmp_path, failure=failure)
    with pytest.raises(RuntimeError):
        play._run_with_fingertip_prior(parser, None)
    assert destination.is_file(), "runtime exception lost the machine-readable report"
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert rows[-1]["terminal_reason"] == "runtime_error:RuntimeError"
    assert rows[-1]["success"] is False
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == rows[-1]


def test_probe_refuses_nonwire_target_before_launch(monkeypatch, tmp_path):
    parser, _, _, launches = _stub_isaac_startup(monkeypatch, tmp_path, object_class="cube")
    with pytest.raises(SystemExit) as error:
        play._run_with_fingertip_prior(parser, None)
    assert error.value.code == 2
    assert launches == []


def test_asset_preflight_failure_preserves_final_json(monkeypatch, tmp_path):
    parser, destination, _, launches = _stub_isaac_startup(monkeypatch, tmp_path)
    def missing_asset(**kwargs):
        raise FileNotFoundError("wire USD missing")
    monkeypatch.setattr(play, "_prepare_object_scene", missing_asset)
    with pytest.raises(SystemExit) as error:
        play._run_with_fingertip_prior(parser, None)
    assert error.value.code == 2
    assert destination.is_file(), "asset preflight lost the failure report"
    final = json.loads(destination.read_text())
    assert final["record_type"] == "final"
    assert final["success"] is False
    assert final["terminal_reason"] == "preflight_error:FileNotFoundError"
    assert launches == []
