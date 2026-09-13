from __future__ import annotations


def test_probe_exposes_orientation_contact_order():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'scripts/m1_dual_panda_o6_bimanual_probe.py').read_text()
    for key in ('right_palm_orientation_mpc', 'first_right_palm_base_contact_orientation',
                'first_right_selected_fingertip_contact_orientation', 'qp_feasible', 'qp_iterations'):
        assert key in source

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_dual_panda_o6_bimanual_probe.py"
PLAY = ROOT / "scripts/m1_dual_panda_o6_bimanual_play.py"
WRAPPER = ROOT / "go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py"
COLLECT = ROOT / "scripts/m1_dual_panda_o6_collect_teacher.py"
TRAIN_LATENT = ROOT / "scripts/m1_dual_panda_o6_train_latent.py"


def _load_acceptance_functions():
    source = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.FunctionDef)
        and node.name in {"trial_passes", "aggregate_acceptance"}
    ]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PROBE), "exec"), namespace)
    return namespace["trial_passes"], namespace["aggregate_acceptance"]


def _passing_trial(seed: int, index: int) -> dict[str, object]:
    return {
        "seed": seed,
        "trial_index": index,
        "lift_height_m": 0.10,
        "hold_duration_s": 3.0,
        "hold_position_error_m": 0.02,
        "hold_orientation_error_rad": 0.10,
        "relative_palm_slip_m": 0.005,
        "object_mpc_feasible_rate": 0.98,
        "arm_mpc_feasible_rates": [0.99, 0.99],
        "hand_mpc_feasible_rates": [0.99, 0.99],
        "wbc_qp_feasible_rate": 1.0,
        "max_abs_roll_rad": 0.17453292519943295,
        "max_abs_pitch_rad": 0.17453292519943295,
        "hard_failure_count": 0,
        "released_supported": True,
        "box_dropped": False,
    }


def test_acceptance_requires_all_30_trials_and_every_hard_gate():
    trial_passes, aggregate_acceptance = _load_acceptance_functions()
    trials = [_passing_trial(seed, index) for seed in (42, 43, 44) for index in range(10)]
    assert all(trial_passes(row) for row in trials)
    aggregate = aggregate_acceptance(trials, seeds=(42, 43, 44), trials_per_seed=10)
    assert aggregate["accepted"] is True
    assert aggregate["passing_trial_count"] == 30
    trials[7]["box_dropped"] = True
    assert aggregate_acceptance(trials, seeds=(42, 43, 44), trials_per_seed=10)["accepted"] is False


def test_acceptance_rejects_missing_or_duplicated_trials():
    _, aggregate_acceptance = _load_acceptance_functions()
    trials = [_passing_trial(seed, index) for seed in (42, 43, 44) for index in range(10)]
    assert not aggregate_acceptance(trials[:-1], seeds=(42, 43, 44), trials_per_seed=10)["accepted"]
    trials[-1] = _passing_trial(42, 0)
    assert not aggregate_acceptance(trials, seeds=(42, 43, 44), trials_per_seed=10)["accepted"]


def test_play_has_no_checkpoint_or_training_surface_and_uses_same_wrapper():
    source = PLAY.read_text(encoding="utf-8")
    assert "--checkpoint" not in source
    assert ".learn(" not in source
    assert "M1DualPandaO6BimanualWrapper" in source
    assert "Isaac-M1-DualPanda-O6-Bimanual-Lift-v0" in source


def test_probe_records_reproducibility_and_all_diagnostic_groups():
    source = PROBE.read_text(encoding="utf-8")
    for token in (
        "--seeds",
        "--trials-per-seed",
        "--report",
        "--jsonl",
        "--manifest",
        "phase_dwell_times_s",
        "fallback_counts",
        "max_contact_forces_n",
        "max_o6_body_contact_forces_n",
        "max_o6_body_contact_links",
        "contact_timing_steps",
        "max_consecutive_bilateral_contact_steps",
        "final_contact_latched_hand_q",
        "fingertip_jacobian_norms",
        "teacher_feasible_rate",
        "teacher_dynamics_residual_max",
        "teacher_contact_residual_max",
        "teacher_fallback_reason_counts",
        "command_fallback_reason_counts",
        "safety_feasible_rate",
        "teacher_action_max_abs",
        "first_limit_violation",
        "first_arm_mpc_failure",
        "first_object_mpc_failure",
        "first_safety_failure",
        "max_joint_limit_violation_rad",
        "min_palm_target_rotation_errors_rad",
        "final_palm_target_rotation_errors_rad",
        "min_fingertip_box_distances_m",
        "final_fingertip_box_distances_m",
        "final_fingertip_box_offsets_m",
        "final_left_hand_q",
        "final_right_hand_q",
        "collision_count",
        "limit_violation_count",
        "reset_count",
        "nonfinite_count",
        "initial_velocity_max",
        "initial_fingertip_palm_offsets_m",
        "initial_fingertip_box_ray_projections_m",
        "first_o6_body_contact_events",
        "max_o6_body_contact_events",
        "startup_terminal_count",
        "asset_sha256",
        "source_sha256",
        "git_ref",
        "isaac_version",
        "command",
        "_atomic_jsonl",
        "_artifact_manifest",
        "trials_jsonl",
        "aggregate_report",
    ):
        assert token in source




def test_probe_and_play_delegate_physical_reset_to_common_wrapper():
    probe = PROBE.read_text(encoding="utf-8")
    play = PLAY.read_text(encoding="utf-8")
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert "def _reset_physical_scene" not in probe
    assert "wrapper.reset(seed=" in probe
    assert "wrapper.reset(seed=" in play
    assert "env.reset(seed=" not in probe
    assert "env.reset(seed=" not in play
    assert "def reset(self, *, seed: int) -> BimanualSnapshot:" in wrapper
    assert "self.startup_complete = True" in wrapper
    assert "raw.sim.step(render=False)" in wrapper


def test_latent_data_entrypoints_separate_failures_and_freeze_training_split():
    collector = COLLECT.read_text(encoding="utf-8")
    trainer = TRAIN_LATENT.read_text(encoding="utf-8")

    assert "teacher_success.npz" in collector
    assert "teacher_failures.npz" in collector
    assert "wrapper.reset(seed=" in collector
    assert "last_teacher_solution" in collector
    assert "deterministic_split" in trainer
    assert "normalization.pt" in trainer
    assert "latent_action_model.pt" in trainer
    assert "metadata.json" in trainer


def test_wrapper_integrates_full_teacher_and_latent_modes_without_59_actions():
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert 'mode: str = "teacher"' in wrapper
    assert 'if mode not in {"teacher", "latent"}' in wrapper
    assert "build_teacher_input(" in wrapper
    assert "self.last_teacher_solution" in wrapper
    assert "LatentRuntime.from_artifact(" in wrapper
    assert "teacher_solution.action_trajectory[0]" in wrapper
    assert "self.safety.project(self._safety_input(" in wrapper
    assert "self.last_safety_result" in wrapper
    assert "projected_effort[31:43] = 0.0" in wrapper
    assert "if self.teacher.fixed_base" in wrapper
    assert "candidate = self._baseline_command.effort" in wrapper
    assert "torch.zeros(59" not in wrapper


def test_probe_selects_artifact_but_play_uses_only_canonical_environment_path():
    probe = PROBE.read_text(encoding="utf-8")
    play = PLAY.read_text(encoding="utf-8")

    assert '"--mode", choices=("teacher", "latent")' in probe
    assert '"--latent-artifact"' in probe
    assert "mode=args.mode" in probe
    assert '"--mode", choices=("teacher", "latent")' in play
    assert "mode=args.mode" in play
    assert "--latent-artifact" not in play


def test_play_and_probe_prior_are_opt_in_and_passed_to_wrapper():
    for source in (PLAY.read_text(encoding="utf-8"), PROBE.read_text(encoding="utf-8")):
        assert '"--fingertip-prior-artifact"' in source
        assert "default=None" in source
        assert "fingertip_prior_artifact=args.fingertip_prior_artifact" in source


def test_entrypoint_help_keeps_isaac_and_prior_loading_out_of_the_import_boundary():
    for path in (PLAY, PROBE):
        code = f'''import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("entrypoint_under_test", {str(path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.argv = [{str(path)!r}, "--help"]
try:
    module.main()
except SystemExit:
    pass
print(json.dumps({{"isaac": any(name.startswith("isaaclab") for name in sys.modules), "prior": any("expert_fingertip_prior" in name for name in sys.modules)}}))
'''
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        assert "--fingertip-prior-artifact" in result.stdout
        assert json.loads(result.stdout.splitlines()[-1]) == {"isaac": False, "prior": False}


def test_invalid_prior_artifact_is_rejected_before_isaac_launcher_or_gym(tmp_path):
    for path in (PLAY, PROBE):
        result = subprocess.run(
            [sys.executable, str(path), "--fingertip-prior-artifact", str(tmp_path / "missing")],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        assert result.returncode != 0
        assert "student artifact root must be a regular directory" in result.stderr
        assert "No module named 'isaaclab'" not in result.stderr
