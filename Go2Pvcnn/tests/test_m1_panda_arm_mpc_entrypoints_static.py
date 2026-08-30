import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_panda_arm_mpc_probe.py"
EVAL = ROOT / "scripts/m1_panda_arm_mpc_residual_eval.py"
PLAY = ROOT / "scripts/m1_panda_arm_mpc_residual_play.py"
RUNTIME_ADAPTER = (
    ROOT
    / "go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py"
)


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


def test_phase5_wrench_direction_uses_matched_motion_increments():
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


def test_phase5_baseline_reuses_previous_snapshot_after_final_timeout():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_timeout_test")

    class _Runtime:
        def diagnostics_snapshot(self):
            raise AssertionError("post-timeout diagnostics must not be read")

    previous = {
        name: torch.full((1, 6), float(index))
        for index, name in enumerate(module.BASELINE_SNAPSHOT_FIELDS)
    }

    result = module._baseline_snapshot_after_step(
        _Runtime(), terminal=True, final_step=True, previous=previous
    )

    assert result is not previous
    for name in module.BASELINE_SNAPSHOT_FIELDS:
        torch.testing.assert_close(result[name], previous[name])
        assert result[name] is not previous[name]


def test_phase5_probe_extends_episode_beyond_requested_sample_count():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_horizon_test")

    class _Sim:
        dt = 0.0025

    class _Cfg:
        sim = _Sim()
        decimation = 2
        episode_length_s = 20.0

    cfg = _Cfg()
    module._extend_episode_horizon(cfg, steps=4000)

    assert cfg.episode_length_s > 20.0


def test_runtime_adapter_exports_complete_panda_link_dynamics_snapshot():
    source = RUNTIME_ADAPTER.read_text()

    for field in (
        "PandaLinkDynamicsState(",
        "body_link_pos_w",
        "body_link_quat_w",
        "body_com_pos_w",
        "body_com_quat_w",
        "get_masses()",
        "get_inertias()",
        "body_com_lin_vel_w",
        "body_com_ang_vel_w",
        "body_com_lin_acc_w",
        "body_com_ang_acc_w",
        "latest_rne_reaction_wrench_b",
    ):
        assert field in source


def test_phase6_eval_freezes_checkpoint_seed_and_zero_baseline_contract():
    module = _load(EVAL, "m1_panda_arm_mpc_residual_eval_test")
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args(
        ["--mode", "zero-pair", "--seed", "42"]
    )

    assert args.device == "cuda:0"
    assert args.seed == 42
    assert not hasattr(args, "seeds")
    assert args.mode == "zero-pair"
    assert args.checkpoint is None
    assert args.steps == 4000
    assert args.num_envs == 1
    source = EVAL.read_text(encoding="utf-8")
    assert '"baseline"' in source
    assert '"candidate"' in source
    assert "atomic_write_json" in source
    assert "source_lineage(" in source


def test_phase6_eval_resets_runtime_before_runner_reads_observations():
    source = EVAL.read_text(encoding="utf-8")
    wrapper_position = source.index("wrapper = M1PandaArmMpcResidualEnvWrapper(")
    reset_position = source.index("wrapper.reset()", wrapper_position)
    runner_position = source.index("runner = OnPolicyRunner(", wrapper_position)
    assert wrapper_position < reset_position < runner_position


def test_phase6_eval_and_play_use_canonical_normalized_inference_policy():
    eval_source = EVAL.read_text(encoding="utf-8")
    play_source = PLAY.read_text(encoding="utf-8")

    for source in (eval_source, play_source):
        assert "get_inference_policy(device=args.device)" in source
        assert "policy = runner.alg.actor_critic" not in source
        assert "policy.act_inference(observations)" not in source


def test_phase6_play_disables_manager_window_before_environment_creation():
    module = _load(PLAY, "m1_panda_arm_mpc_residual_play_test")
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args(
        ["--checkpoint", "/tmp/model.pt"]
    )
    assert args.device == "cuda:0"
    assert args.seed == 42
    assert args.num_envs == 1
    source = PLAY.read_text(encoding="utf-8")
    viewer_position = source.index("env_cfg.viewer")
    window_position = source.index("env_cfg.ui_window_class_type = None")
    make_position = source.index("gym.make(")
    assert viewer_position < make_position
    assert window_position < make_position
    assert "render_mode=\"human\"" in source
    assert "viewport_camera_controller" in source
