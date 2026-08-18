import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / "go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py"
REGISTRY = ROOT / "go2_pvcnn/tasks/register_m1_envs.py"
SCRIPT = ROOT / "scripts/m1_panda_wbc_roll_play.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "m1_panda_wbc_roll_play_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_c1a_env_inherits_c0_effort_contract_and_has_runtime_margin():
    source = ENV_CFG.read_text()

    assert (
        "class M1PandaWbcRollTeacherEnvCfg(M1PandaWbcTeacherEnvCfg)"
        in source
    )
    assert "self.decimation = 1" in source
    assert "self.episode_length_s = 30.0" in source
    assert "self.sim.dt = 0.005" in source


def test_c1a_has_one_independent_gym_registration():
    source = REGISTRY.read_text()
    tree = ast.parse(source)
    registrations = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func) == "gym.register"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        if "id" in keywords and ast.literal_eval(keywords["id"]) == (
            "Isaac-M1-Panda-Wbc-Teacher-C1a-v0"
        ):
            registrations.append(keywords)

    assert len(registrations) == 1
    kwargs = ast.literal_eval(registrations[0]["kwargs"])
    assert kwargs["env_cfg_entry_point"] == (
        "go2_pvcnn.tasks.m1_panda_wbc_roll_teacher_env_cfg:"
        "M1PandaWbcRollTeacherEnvCfg"
    )
    assert kwargs["rsl_rl_cfg_entry_point"] is None


def test_c1a_script_has_independent_task_and_fixed_mission_contract():
    source = SCRIPT.read_text()

    assert 'TASK_ID = "Isaac-M1-Panda-Wbc-Teacher-C1a-v0"' in source
    assert "MISSION_STEPS = 4000" in source
    assert "PHASE_STEPS = 800" in source
    assert "WHEEL_RADIUS_M = 0.095" in source
    assert "SETTLE_STEPS = 100" in source
    assert "--disable-target-motion" in source
    assert "--summary-json" in source


def test_c1a_summary_exposes_rolling_balance_and_qp_gates():
    module = _load_script()
    payload = module.C1aSummary(seed=42, requested_steps=4000).to_dict()

    required = {
        "phase_counts",
        "completed_phase_count",
        "vx_rmse_mps",
        "forward_displacement_m",
        "reverse_displacement_m",
        "stop_settle_time_s",
        "max_rolling_residual_mps",
        "max_lateral_slip_mps",
        "min_wheel_contact_count",
        "max_wheel_velocity_spread_radps",
        "wheel_effort_saturation_count",
        "wheel_direction_mismatch_count",
        "qp_feasible_rate",
        "max_qp_equality_residual",
        "max_qp_inequality_violation",
        "safety_state_counts",
        "exit_reason",
        "hard_gates_passed",
    }
    assert required <= payload.keys()


def test_c1a_validate_args_rejects_invalid_shape_and_step_ranges():
    module = _load_script()

    good = type("Args", (), {"steps": 4000, "num_envs": 1, "stats_interval": 100})()
    module.validate_args(good)
    for values, message in (
        ({"steps": -1, "num_envs": 1, "stats_interval": 100}, "--steps"),
        ({"steps": 4001, "num_envs": 1, "stats_interval": 100}, "--steps"),
        ({"steps": 8, "num_envs": 2, "stats_interval": 100}, "one environment"),
        ({"steps": 8, "num_envs": 1, "stats_interval": 0}, "--stats-interval"),
    ):
        args = type("Args", (), values)()
        with pytest.raises(ValueError, match=message):
            module.validate_args(args)


def test_formal_hard_gates_reject_good_tracking_with_bad_balance():
    module = _load_script()
    summary = module.C1aSummary(seed=42, requested_steps=4000)
    summary.steps = 4000
    summary.phase_counts = {str(index): 800 for index in range(5)}
    summary.qp_feasible_count = 4000
    summary.track_scale_count = 4000
    summary.min_wheel_contact_count = 4
    summary.stop_settle_time_s = 0.5
    summary.forward_displacement_m = 0.2
    summary.reverse_displacement_m = -0.1
    summary.exit_reason = "steps_complete"

    assert module.formal_hard_gates_pass(summary)
    summary.max_abs_roll_rad = torch.deg2rad(torch.tensor(11.0)).item()
    assert not module.formal_hard_gates_pass(summary)


def test_runtime_wires_rolling_teacher_adapter_and_one_effort_step():
    source = SCRIPT.read_text()
    tree = ast.parse(source)

    for token in (
        "M1PandaRollingWbcTeacher(",
        "RollingTeacherState(",
        "rolling_contact_metrics(",
        "wheel_radius_m=WHEEL_RADIUS_M",
        "teacher.step(state)",
        "command.effort",
        "mission_step += 1",
    ):
        assert token in source
    env_steps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "env.step"
    ]
    assert len(env_steps) == 1
    assert ast.unparse(env_steps[0].args[0]) == "effort_action"
