import ast
import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m1_panda_wbc_play.py"
SUMMARY_FIELDS = {
    "seed",
    "steps",
    "finite",
    "qp_feasible_count",
    "qp_feasible_rate",
    "max_ee_position_error_m",
    "min_singular_value",
    "max_abs_roll_rad",
    "max_abs_pitch_rad",
    "max_lateral_slip_mps",
    "joint_limit_violations",
    "base_contacts",
    "self_collisions",
    "safety_state_counts",
    "reset_count",
    "exit_reason",
}


def _source():
    return SCRIPT.read_text()


def _load_script():
    spec = importlib.util.spec_from_file_location("m1_panda_wbc_play_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_is_deterministic_c0_only_and_supports_unlimited_steps():
    source = _source()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    options = {
        ast.literal_eval(call.args[0]): {kw.arg: kw.value for kw in call.keywords}
        for call in calls if call.args and isinstance(call.args[0], ast.Constant)
    }
    for option in ("--steps", "--seed", "--summary-json", "--disable-target-motion", "--headless", "--device"):
        assert option in source
    assert ast.literal_eval(options["--steps"]["default"]) == 0
    assert ast.literal_eval(options["--num-envs"]["default"]) == 1
    assert "args.num_envs != 1" in source
    assert 'TASK_ID = "Isaac-M1-Panda-Wbc-Teacher-C0-v0"' in source


def test_adapter_reads_live_physx_dynamics_and_explicitly_combines_bias_force():
    source = _source()
    for call in (
        "root_view.get_generalized_mass_matrices()",
        "root_view.get_coriolis_and_centrifugal_forces()",
        "root_view.get_generalized_gravity_forces()",
        "root_view.get_jacobians()",
    ):
        assert call in source
    assert "bias_force = coriolis_force + gravity_force" in source
    assert "WbcJointMap.resolve(robot.joint_names)" in source
    assert "joint_map.controlled + 6" in source
    assert "dtype=torch.float64" in source


def test_runtime_builds_teacher_state_applies_effort_and_steps_once():
    source = _source()
    tree = ast.parse(source)
    assert "TeacherState(" in source
    assert "StandingWbcInput(" in source
    assert "teacher.step(state)" in source
    assert "command.effort" in source
    env_steps = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "env.step"
    ]
    assert len(env_steps) == 1
    assert ast.unparse(env_steps[0].args[0]) == "effort_action"


def test_diagnostics_and_atomic_summary_cover_all_acceptance_fields():
    source = _source()
    for token in (
        "ee_error",
        "sigma_min",
        "qp_feasible",
        "roll",
        "pitch",
        "wheel_contacts",
        "lateral_slip",
        "safety",
        "reset_cause",
    ):
        assert token in source
    tree = ast.parse(source)
    summary = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "C0Summary")
    method = next(node for node in summary.body if isinstance(node, ast.FunctionDef) and node.name == "to_dict")
    rendered = ast.unparse(method)
    assert SUMMARY_FIELDS <= {field for field in SUMMARY_FIELDS if repr(field) in rendered}
    assert "temporary_path.write_text" in source
    assert "os.replace(temporary_path, path)" in source


def test_play_has_no_learning_checkpoint_or_manifest_path():
    source = _source()
    banned = (
        "OnPolicyRunner",
        "rsl_rl",
        "load_checkpoint",
        ".learn(",
        "optimizer.step",
        "atomic_write_manifest",
    )
    for token in banned:
        assert token not in source


def test_teacher_gains_use_reference_qp_float64_contract():
    module = _load_script()
    kp, kd = module.build_teacher_gains()
    assert kp.shape == kd.shape == (23,)
    assert kp.dtype == kd.dtype == torch.float64


def test_bias_reader_upgrades_legacy_joint_only_physx_forces_to_floating_base():
    module = _load_script()

    class RootView:
        def __init__(self):
            self.calls = []

        def get_coriolis_and_centrifugal_forces(self):
            self.calls.append("legacy_coriolis")
            return torch.ones(1, 25)

        def get_generalized_gravity_forces(self):
            self.calls.append("legacy_gravity")
            return torch.full((1, 25), 2.0)

        def get_coriolis_and_centrifugal_compensation_forces(self):
            self.calls.append("full_coriolis")
            return torch.full((1, 31), 3.0)

        def get_gravity_compensation_forces(self):
            self.calls.append("full_gravity")
            return torch.full((1, 31), 4.0)

    root_view = RootView()
    result = module.read_generalized_bias_force(root_view, generalized_dof=31)
    assert result.shape == (31,)
    assert result.dtype == torch.float64
    assert torch.equal(result, torch.full((31,), 7.0, dtype=torch.float64))
    assert root_view.calls == [
        "legacy_coriolis",
        "legacy_gravity",
        "full_coriolis",
        "full_gravity",
    ]
