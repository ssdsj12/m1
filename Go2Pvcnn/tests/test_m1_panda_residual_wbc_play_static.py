import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_residual_wbc_play.py"
REGISTER = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"
TASK_ID = "Isaac-M1-Panda-Residual-Wbc-v0"
REQUIRED_SUMMARY_FIELDS = {
    "seed",
    "steps",
    "finite",
    "qp_feasible_rate",
    "min_wheel_contact_count",
    "base_contacts",
    "max_abs_roll_rad",
    "max_abs_pitch_rad",
    "joint_limit_violations",
    "reset_count",
    "max_abs_normalized_residual",
    "max_abs_physical_residual",
    "max_abs_filtered_mount_wrench",
    "max_abs_correction_wrench",
    "safety_state_counts",
    "exit_reason",
}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_residual_wbc_play_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_residual_gym_id_reuses_accepted_c1a_environment_cfg():
    source = REGISTER.read_text()
    assert f'id="{TASK_ID}"' in source
    block = source.split(f'id="{TASK_ID}"', 1)[1].split("gym.register(", 1)[0]
    assert "m1_panda_wbc_roll_teacher_env_cfg:M1PandaWbcRollTeacherEnvCfg" in block
    assert '"rsl_rl_cfg_entry_point": None' in block


def test_play_cli_defaults_to_zero_8d_residual_and_one_environment():
    module = _load_script()
    parser = module.build_arg_parser(include_app_launcher_args=False)
    args = parser.parse_args([])

    assert module.TASK_ID == TASK_ID
    assert args.task == TASK_ID
    assert args.residual_axis == -1
    assert args.residual_value == pytest.approx(0.0)
    assert args.warmup_steps == 64
    assert args.steps == 256
    assert args.num_envs == 1
    action = module.build_normalized_residual(args, "cpu", torch.float64)
    assert torch.equal(action, torch.zeros(1, 8, dtype=torch.float64))


@pytest.mark.parametrize(("axis", "value"), ((0, 0.1), (7, -0.25)))
def test_play_builds_only_the_selected_residual_axis(axis, value):
    module = _load_script()
    parser = module.build_arg_parser(include_app_launcher_args=False)
    args = parser.parse_args(
        ["--residual-axis", str(axis), "--residual-value", str(value)]
    )
    module.validate_args(args)

    action = module.build_normalized_residual(args, "cpu", torch.float64)

    expected = torch.zeros(1, 8, dtype=torch.float64)
    expected[0, axis] = value
    assert torch.equal(action, expected)


@pytest.mark.parametrize(
    "arguments",
    (
        ["--residual-axis", "8"],
        ["--residual-axis", "0", "--residual-value", "1.1"],
        ["--steps", "0"],
        ["--warmup-steps", "-1"],
        ["--num-envs", "2"],
    ),
)
def test_play_rejects_invalid_manual_probe_contract(arguments):
    module = _load_script()
    args = module.build_arg_parser(include_app_launcher_args=False).parse_args(arguments)
    with pytest.raises(ValueError):
        module.validate_args(args)


def test_play_has_no_learning_path_and_serializes_required_diagnostics():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    assert ".learn(" not in source
    assert "OnPolicyRunner" not in source
    assert "M1PandaResidualWbcController(" in source
    assert "adapter.read_mount_wrench_b()" in source
    assert "adapter.leg_soft_limits()" in source
    summary = _load_script().ResidualSmokeSummary(seed=42, requested_steps=2)
    assert REQUIRED_SUMMARY_FIELDS <= set(summary.to_dict())
    app_launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("AppLauncher")
    ]
    assert len(app_launches) == 1
