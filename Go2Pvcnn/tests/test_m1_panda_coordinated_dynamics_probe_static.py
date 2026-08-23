from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_coordinated_dynamics_probe.py"


def test_probe_declares_two_long_horizon_modes_and_gpu_runtime_contract():
    source = SCRIPT.read_text()
    assert "PROJECT_ROOT" in source
    assert "sys.path.insert(0, str(PROJECT_ROOT))" in source
    assert "Isaac-M1-Panda-Coordinated-v0" in source
    for token in (
        'choices=("hold", "controlled")',
        'default=8',
        'default=2000',
        'AppLauncher.add_app_launcher_args',
        'env.step(effort)',
    ):
        assert token in source


def test_probe_records_mount_stability_and_failure_metrics():
    source = SCRIPT.read_text()
    for metric in (
        "max_mount_position_drift_m",
        "max_mount_orientation_drift_rad",
        "max_mount_force_n",
        "max_mount_torque_nm",
        "non_finite_count",
        "reset_count",
        "base_contact_count",
        "bad_orientation_count",
        "joint_limit_violation_count",
        "max_abs_effort_nm",
        "hard_gates_passed",
    ):
        assert metric in source
    assert "atomic_write_summary" in source
    assert "M1_PANDA_BASE_BODY_NAME" in source
    assert "M1_PANDA_MOUNT_BODY_NAME" in source


def test_probe_uses_hold_impedance_and_bounded_controlled_arm_motion():
    source = SCRIPT.read_text()
    assert "build_teacher_gains" in source
    assert "controlled_arm_amplitude_rad" in source
    assert "0.05" in source
    assert "torch.clamp" in source
    assert "joint_effort_limits" in source
