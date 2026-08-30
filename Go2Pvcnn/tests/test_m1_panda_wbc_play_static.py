import ast
import importlib.util
from pathlib import Path
import sys

import pytest
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
    "safety_reason_counts",
    "base_activation_count",
    "first_base_activation_step",
    "first_singularity_crossing_step",
    "max_arm_target_step_rad",
    "arm_snap_count",
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


def test_c0_adapter_keeps_accepted_radius_default_but_allows_c1a_injection():
    source = _source()

    assert "WHEEL_RADIUS_M = 0.0959" in source
    assert "wheel_radius_m: float = WHEEL_RADIUS_M" in source
    assert "self.wheel_radius_m = float(wheel_radius_m)" in source
    assert "build_wheel_contact_jacobian(" in source
    for field in (
        "latest_root_xy_yaw",
        "latest_root_vxy_yawrate",
        "latest_generalized_velocity",
        "latest_contact_jacobian",
        "latest_wheel_velocity",
    ):
        assert field in source


def test_shared_runtime_adapter_exposes_canonical_mount_wrench_and_leg_limits():
    source = (
        ROOT
        / "go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py"
    ).read_text()

    assert "def read_mount_wrench_b(" in source
    assert "shift_rotate_wrench_to_base(" in source
    assert "get_link_incoming_joint_force()" in source
    assert "def leg_soft_limits(" in source
    assert "self.joint_map.legs" in source


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
    assert "position_amplitude=0.005" in source
    assert "orientation_amplitude=0.01" in source


def test_runtime_recenters_after_settling_before_counting_mission_steps():
    source = _source()
    assert "SETTLE_STEPS = 100" in source
    assert "if not settled and physics_step == SETTLE_STEPS:" in source
    assert "teacher.reset(state, seed=args.seed)" in source
    assert "if settled:" in source
    assert "_update_summary(summary, state, command, base_contact, adapter)" in source
    assert "mission_step += 1" in source
    assert "mission_step < args.steps" in source


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
        "safety_reason",
        "base_active",
        "reset_cause",
    ):
        assert token in source
    assert "target_rotation" in source
    assert "actual_rotation" in source
    assert "predicted_twist" in source
    assert "measured_twist" in source
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
    assert torch.equal(kp[:12], torch.full((12,), 120.0, dtype=torch.float64))
    assert torch.equal(kd[:12], torch.full((12,), 20.0, dtype=torch.float64))
    assert torch.equal(kp[12:16], torch.zeros(4, dtype=torch.float64))
    assert torch.equal(kd[12:16], torch.full((4,), 2.0, dtype=torch.float64))
    assert torch.equal(kp[16:], torch.full((7,), 80.0, dtype=torch.float64))
    assert torch.equal(
        kd[16:],
        torch.tensor([4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0], dtype=torch.float64),
    )
    assert "torch.full((7,), 100.0, dtype=torch.float64)" in _source()


def test_relative_hand_orientation_uses_current_times_inverse_reference():
    module = _load_script()

    class FakeMath:
        @staticmethod
        def quat_inv(reference):
            return reference + 10.0

        @staticmethod
        def quat_mul(current, inverse_reference):
            assert torch.equal(current, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
            assert torch.equal(inverse_reference, torch.tensor([[15.0, 16.0, 17.0, 18.0]]))
            return torch.tensor([[9.0, 8.0, 7.0, 6.0]])

        @staticmethod
        def axis_angle_from_quat(relative):
            assert torch.equal(relative, torch.tensor([[9.0, 8.0, 7.0, 6.0]]))
            return torch.tensor([[0.1, 0.2, 0.3]])

    result = module.relative_axis_angle(
        FakeMath,
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
    )
    assert torch.equal(result, torch.tensor([[0.1, 0.2, 0.3]]))


def test_adapter_captures_initial_hand_quaternion_for_relative_pose():
    source = _source()
    assert "self._initial_hand_quat" in source
    assert "relative_axis_angle(" in source
    assert "self._initial_arm_q" in source
    assert "self._initial_arm_q - arm_q" in source


def test_body_jacobian_uses_direct_body_id_when_physx_includes_root_link():
    module = _load_script()
    jacobians = torch.arange(1 * 4 * 6 * 3).reshape(1, 4, 6, 3)
    selected = module.PhysxTeacherAdapter._body_jacobian(
        jacobians, body_id=2, body_count=4
    )
    assert torch.equal(selected, jacobians[0, 2])


def test_body_jacobian_offsets_body_id_when_physx_omits_root_link():
    module = _load_script()
    jacobians = torch.arange(1 * 3 * 6 * 3).reshape(1, 3, 6, 3)
    selected = module.PhysxTeacherAdapter._body_jacobian(
        jacobians, body_id=2, body_count=4
    )
    assert torch.equal(selected, jacobians[0, 1])


def test_bias_reader_prefers_full_physx_51_compensation_without_deprecated_calls():
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
        "full_coriolis",
        "full_gravity",
    ]


def test_wheel_contact_jacobian_uses_bottom_point_not_wheel_center():
    module = _load_script()
    body_jacobian = torch.zeros(6, 1, dtype=torch.float64)
    body_jacobian[4, 0] = 1.0  # unit angular velocity about world y

    point_jacobian = module.contact_point_linear_jacobian(
        body_jacobian, torch.tensor([0.0, 0.0, -0.0959], dtype=torch.float64)
    )

    assert point_jacobian.shape == (3, 1)
    assert point_jacobian[:, 0].tolist() == pytest.approx([-0.0959, 0.0, 0.0])
