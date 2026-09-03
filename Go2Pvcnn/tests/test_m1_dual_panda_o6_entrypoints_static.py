from __future__ import annotations

import ast
from pathlib import Path


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
        "phase_dwell_times_s",
        "fallback_counts",
        "max_contact_forces_n",
        "fingertip_jacobian_norms",
        "collision_count",
        "limit_violation_count",
        "reset_count",
        "nonfinite_count",
        "initial_velocity_max",
        "startup_terminal_count",
        "asset_sha256",
        "source_sha256",
        "git_ref",
        "isaac_version",
        "command",
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
