import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_panda_arm_mpc_probe.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase5_probe_freezes_gpu_gate_defaults_and_all_required_metrics():
    source = PROBE.read_text()
    assert "PROJECT_ROOT" in source
    assert "sys.path.insert(0, str(PROJECT_ROOT))" in source
    module = _load(PROBE, "m1_panda_arm_mpc_probe_test")
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args([])
    assert args.task == "Isaac-M1-Panda-ArmMpc-Residual-v0"
    assert args.device == "cuda:0"
    assert args.num_envs == 1
    assert args.steps == 4000
    assert args.seeds == [42, 43, 44]
    assert args.history_json is None
    summary = module.Phase5Summary(seed=42, requested_steps=1)
    required = {
        "finite", "mpc_feasible_rate", "qp_feasible_rate",
        "min_wheel_contact_count", "base_contacts", "joint_limit_violations",
        "reset_count", "max_abs_roll_rad", "max_abs_pitch_rad",
        "max_ee_position_error_m", "max_ee_orientation_error_rad",
        "force_direction_cosine", "moment_direction_cosine",
        "eligible_force_samples", "eligible_moment_samples", "accepted",
        "max_arm_reference_error_rad", "max_arm_qd_ref_abs_rad_s",
        "max_arm_qdd_first_abs_rad_s2", "max_correction_wrench_norm",
        "final_arm_q", "final_arm_q_ref", "final_measured_mount_wrench_b",
        "final_predicted_mount_wrench_b", "final_correction_wrench_b",
        "final_target_pose", "final_current_ee_pose", "final_predicted_ee_pose_first",
        "final_replan_start_ee_pose", "final_arm_qd_ref",
        "final_predicted_ee_pose_terminal",
        "max_root_xy_displacement_m", "final_root_xy", "initial_root_xy",
        "actual_force_direction_cosine", "actual_moment_direction_cosine",
        "max_actual_arm_qdd_abs_rad_s2", "final_actual_dynamic_mount_wrench_b",
    }
    assert required <= set(summary.to_dict())


def test_phase5_gate_rejects_missing_wrench_samples_and_accepts_exact_safe_boundary():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_gate_test")
    summary = module.Phase5Summary(seed=42, requested_steps=1)
    summary.steps = 1
    summary.mpc_feasible_count = 1
    summary.qp_feasible_count = 1
    summary.exit_reason = "steps_complete"
    assert not module.phase5_gates_pass(summary)
    summary.eligible_force_samples = 1
    summary.eligible_moment_samples = 1
    summary.force_direction_cosine_sum = 0.8
    summary.moment_direction_cosine_sum = 0.8
    assert module.phase5_gates_pass(summary)


def test_phase5_wrench_direction_uses_motion_increment_over_fixed_seed_baseline():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_delta_test")
    active = {
        "dynamic_measured_mount_wrench_b": torch.tensor([[3.0, 4.0, 0.0, 0.0, 0.5, 0.0]]),
        "predicted_mount_wrench_b": torch.tensor([[2.0, 2.0, 0.0, 0.0, 0.4, 0.0]]),
    }
    baseline = {
        "dynamic_measured_mount_wrench_b": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.2, 0.0]]),
        "predicted_mount_wrench_b": torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.1, 0.0]]),
    }

    measured, predicted = module._motion_wrench_increment(active, baseline)

    assert torch.equal(measured, torch.tensor([2.0, 3.0, 0.0, 0.0, 0.3, 0.0]))
    assert torch.equal(predicted, torch.tensor([1.5, 1.5, 0.0, 0.0, 0.3, 0.0]))


def test_phase5_lag_scan_finds_delayed_measured_wrench_without_changing_gate():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_lag_test")
    predicted = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.3, 0.0],
            [0.0, 3.0, 0.0, -0.3, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0, -0.3, 0.0],
            [0.0, -3.0, 0.0, 0.3, 0.0, 0.0],
        ]
    )
    measured = torch.cat((torch.zeros(2, 6), predicted[:-2]), dim=0)

    result = module._lagged_direction_cosines(measured, predicted, max_lag=3)

    assert result["best_force_lag_steps"] == 2
    assert result["best_moment_lag_steps"] == 2
    assert result["best_force_direction_cosine"] == pytest.approx(1.0)
    assert result["best_moment_direction_cosine"] == pytest.approx(1.0)


def test_phase5_bias_candidates_add_and_subtract_synchronized_bias_delta():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_bias_test")
    active = {"base_bias_wrench": torch.tensor([[2.0, 3.0, 4.0, 0.2, 0.3, 0.4]])}
    baseline = {"base_bias_wrench": torch.tensor([[1.0, 1.0, 1.0, 0.1, 0.1, 0.1]])}
    predicted = torch.ones(6)

    plus, minus = module._bias_augmented_predictions(active, baseline, predicted)

    torch.testing.assert_close(
        plus, torch.tensor([2.0, 3.0, 4.0, 1.1, 1.2, 1.3])
    )
    torch.testing.assert_close(
        minus, torch.tensor([0.0, -1.0, -2.0, 0.9, 0.8, 0.7])
    )


def test_phase5_raw_motion_increment_uses_synchronized_sensor_baseline():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_raw_test")
    active = {"measured_mount_wrench_b": torch.tensor([[3.0] * 6])}
    baseline = {"measured_mount_wrench_b": torch.tensor([[1.0] * 6])}

    result = module._raw_motion_wrench_increment(active, baseline)

    torch.testing.assert_close(result, torch.tensor([2.0] * 6))
