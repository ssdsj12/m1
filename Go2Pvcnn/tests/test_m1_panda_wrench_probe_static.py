from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_panda_wrench_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("m1_panda_wrench_probe_under_test", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_covers_all_six_axes_and_clears_external_wrench():
    source = PROBE.read_text()
    for case in ("force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"):
        assert f'"{case}"' in source
    assert "clear_external_wrench as _clear_external_wrench" in source
    assert "base_wrench_to_body_local as _base_wrench_to_body_local" in source
    assert 'TASK_ID = "Isaac-M1-Panda-Smoke-v0"' in source


def test_probe_contract_is_one_env_exact_windows_and_strict_failures():
    source = PROBE.read_text()
    assert "env_cfg.scene.num_envs = 1" in source
    assert "SETTLE_STEPS = 100" in source
    assert "BASELINE_STEPS = 50" in source
    assert "TRANSITION_STEPS = 10" in source
    assert "SAMPLE_STEPS = 50" in source
    assert "M1_PANDA_DOF_COUNT" in source
    assert "terminated" in source and "truncated" in source
    assert "torch.isfinite" in source
    assert "raise RuntimeError" in source


def test_artifact_and_failure_status_are_committed_before_kit_close():
    source = PROBE.read_text()
    close_index = source.index("simulation_app.close()")
    assert source.index("_write_jsonl_atomic(args.output, rows)") < close_index
    assert "os.replace(temporary_path, output)" in source
    assert source.index("os._exit(1)") < close_index


def test_empty_clear_uses_an_empty_body_selector_for_installed_isaaclab():
    source = PROBE.read_text()
    helper = ROOT / "go2_pvcnn/tasks/m1_panda_teacher.py"
    assert "torch.zeros(0, 3, device=robot.device)" in helper.read_text()
    assert "clear_external_wrench as _clear_external_wrench" in source


def test_empty_clear_accepts_only_the_known_isaaclab_empty_assignment_bug():
    probe = _load_probe()

    class Robot:
        device = "cpu"
        has_external_wrench = True

        def set_external_force_and_torque(self, forces, torques):
            assert tuple(forces.shape) == (0, 3)
            assert tuple(torques.shape) == (0, 3)
            self.has_external_wrench = False
            raise RuntimeError(
                "shape mismatch: value tensor of shape [0] cannot be broadcast to indexing result of shape [29, 3]"
            )

    robot = Robot()
    probe._clear_external_wrench(robot)
    assert robot.has_external_wrench is False


def test_empty_clear_rejects_an_unrelated_shape_mismatch():
    probe = _load_probe()

    class Robot:
        device = "cpu"
        has_external_wrench = False

        def set_external_force_and_torque(self, forces, torques):
            raise RuntimeError("shape mismatch in an unrelated actuator buffer")

    with pytest.raises(RuntimeError, match="unrelated actuator buffer"):
        probe._clear_external_wrench(Robot())


def test_case_table_is_the_verbatim_base_frame_contract():
    probe = _load_probe()
    assert probe.CASES == {
        "force_x": ([20.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        "force_y": ([0.0, 20.0, 0.0], [0.0, 0.0, 0.0]),
        "force_z": ([0.0, 0.0, 20.0], [0.0, 0.0, 0.0]),
        "torque_x": ([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
        "torque_y": ([0.0, 0.0, 0.0], [0.0, 5.0, 0.0]),
        "torque_z": ([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]),
    }


def test_base_frame_wrench_is_transformed_into_rotated_hand_local_axes_each_step():
    probe = _load_probe()
    half = 2.0**-0.5
    # Base is +90 deg about world Z; hand is +180 deg about world Z.
    base_quat_w = torch.tensor([[half, 0.0, 0.0, half]])
    hand_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    force_b = torch.tensor([[20.0, 0.0, 0.0]])
    torque_b = torch.tensor([[0.0, 5.0, 0.0]])

    force_h, torque_h = probe._base_wrench_to_body_local(
        force_b, torque_b, base_quat_w, hand_quat_w
    )

    assert torch.allclose(force_h, torch.tensor([[0.0, -20.0, 0.0]]), atol=1e-5)
    assert torch.allclose(torque_h, torch.tensor([[5.0, 0.0, 0.0]]), atol=1e-5)


def test_channel_check_accepts_fraction_boundary_and_rejects_point_eighty_eight():
    probe = _load_probe()
    baseline = torch.zeros(6)
    samples = torch.full((50, 6), -5.0)
    samples[-5:, 0] = 1.0
    passed = probe._evaluate_channel(samples, baseline, channel=0, applied_magnitude=20.0, expected_sign=-1)
    assert passed["stable_sign"] is True
    assert passed["sign_count"] == 45
    assert passed["sign_fraction"] == 0.9
    assert passed["mean_expected_sign"] is True
    assert passed["magnitude_ratio"] == pytest.approx(0.22)
    assert passed["pass"] is True

    samples[-6:, 0] = 1.0
    unstable = probe._evaluate_channel(samples, baseline, channel=0, applied_magnitude=20.0, expected_sign=-1)
    assert unstable["stable_sign"] is False
    assert unstable["sign_count"] == 44
    assert unstable["sign_fraction"] == 0.88
    assert unstable["pass"] is False


def test_channel_check_requires_mean_expected_sign_and_strict_twenty_percent_ratio():
    probe = _load_probe()
    baseline = torch.zeros(6)
    wrong_mean = torch.zeros(50, 6)
    wrong_mean[:45, 0] = -1.0
    wrong_mean[-5:, 0] = 20.0
    wrong = probe._evaluate_channel(
        wrong_mean, baseline, channel=0, applied_magnitude=1.0, expected_sign=-1
    )
    assert wrong["sign_fraction"] == 0.9
    assert wrong["mean_expected_sign"] is False
    assert wrong["magnitude_ratio"] > 0.2
    assert wrong["pass"] is False

    exactly_gate = torch.full((50, 6), -4.0)
    boundary = probe._evaluate_channel(
        exactly_gate, baseline, channel=0, applied_magnitude=20.0, expected_sign=-1
    )
    assert boundary["magnitude_ratio"] == 0.2
    assert boundary["pass"] is False


def test_case_rows_record_transition_sample_and_sign_schema():
    source = PROBE.read_text()
    for key in (
        '"transition_steps"',
        '"sample_steps"',
        '"sign_count"',
        '"sign_fraction"',
        '"measured_mean"',
        '"baseline_subtracted"',
        '"magnitude_ratio"',
        '"pass"',
    ):
        assert key in source


def test_unexpected_reset_error_preserves_termination_diagnostics():
    probe = _load_probe()
    with pytest.raises(RuntimeError, match="base_contact.*true"):
        probe._check_no_reset(
            torch.tensor([True]),
            torch.tensor([False]),
            "force_z",
            details={"base_contact": "true"},
        )


def test_independent_case_windows_reset_and_do_not_reuse_baselines():
    probe = _load_probe()
    events = []
    baseline_value = 0

    def clear():
        events.append("clear")

    def reset():
        events.append("reset")
        return {"policy": torch.zeros(1, 1)}, {}

    def validate():
        events.append("validate")

    def collect_clear(steps, label):
        nonlocal baseline_value
        events.append((steps, label))
        if steps == probe.BASELINE_STEPS:
            baseline_value += 1
            return torch.full((steps, 6), float(baseline_value))
        return torch.zeros(steps, 6)

    first = probe._prepare_independent_window(
        label="force_x", clear=clear, reset=reset, validate=validate, collect_clear=collect_clear
    )
    second = probe._prepare_independent_window(
        label="force_y", clear=clear, reset=reset, validate=validate, collect_clear=collect_clear
    )

    assert events.count("reset") == 2
    assert events.count("validate") == 2
    assert torch.equal(first[1], torch.ones(6))
    assert torch.equal(second[1], torch.full((6,), 2.0))
    assert first[1].data_ptr() != second[1].data_ptr()


def test_independent_window_rejects_invalid_reset_result():
    probe = _load_probe()
    with pytest.raises(RuntimeError, match="reset failed"):
        probe._prepare_independent_window(
            label="force_z",
            clear=lambda: None,
            reset=lambda: None,
            validate=lambda: None,
            collect_clear=lambda steps, label: torch.zeros(steps, 6),
        )


def test_runtime_uses_independent_window_for_settle_and_every_case():
    source = PROBE.read_text()
    assert 'label="settle"' in source
    assert "_prepare_independent_window(" in source
    assert "for case_index" in source


def test_jsonl_artifact_is_written_via_atomic_replace(tmp_path):
    probe = _load_probe()
    output = tmp_path / "probe.jsonl"
    rows = [{"case": "settle", "pass": True}, {"case": "force_x", "pass": True}]

    probe._write_jsonl_atomic(output, rows)

    assert [__import__("json").loads(line) for line in output.read_text().splitlines()] == rows
    assert list(tmp_path.glob(".*probe.jsonl.*.tmp")) == []
