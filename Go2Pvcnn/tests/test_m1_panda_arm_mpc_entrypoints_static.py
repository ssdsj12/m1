import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_panda_arm_mpc_probe.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase5_probe_freezes_gpu_gate_defaults_and_all_required_metrics():
    module = _load(PROBE, "m1_panda_arm_mpc_probe_test")
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args([])
    assert args.task == "Isaac-M1-Panda-ArmMpc-Residual-v0"
    assert args.device == "cuda:0"
    assert args.num_envs == 1
    assert args.steps == 4000
    assert args.seeds == [42, 43, 44]
    summary = module.Phase5Summary(seed=42, requested_steps=1)
    required = {
        "finite", "mpc_feasible_rate", "qp_feasible_rate",
        "min_wheel_contact_count", "base_contacts", "joint_limit_violations",
        "reset_count", "max_abs_roll_rad", "max_abs_pitch_rad",
        "max_ee_position_error_m", "max_ee_orientation_error_rad",
        "force_direction_cosine", "moment_direction_cosine",
        "eligible_force_samples", "eligible_moment_samples", "accepted",
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
