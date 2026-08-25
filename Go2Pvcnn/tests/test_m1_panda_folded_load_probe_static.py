from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/m1_panda_folded_load_probe.py"
RUNBOOK = ROOT.parent / "docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md"


def _load_probe():
    spec = importlib.util.spec_from_file_location("folded_probe_under_test", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _good_diagnostics():
    return {
        "inactive_action_max": 0.0,
        "fold_error_max": 0.01,
        "effort_utilization_max": 0.5,
        "joint_limit_proximity_min": 0.2,
        "mount_wrench_norm_max": 10.0,
        "base_contact_rate": 0.0,
        "bad_orientation_rate": 0.0,
        "hard_failure_rate": 0.0,
    }


def test_probe_gate_requires_finite_exact_mask_fold_effort_margin_and_mount_response():
    probe = _load_probe()
    report = probe.evaluate_probe(
        _good_diagnostics(), finite_state=True, physics_steps=4
    )
    assert report["passed"] is True
    assert all(report["checks"].values())

    mutations = (
        ("inactive_action_max", 1e-12),
        ("fold_error_max", 0.351),
        ("effort_utilization_max", 1.001),
        ("joint_limit_proximity_min", 0.01),
        ("mount_wrench_norm_max", 0.0),
    )
    for key, value in mutations:
        diagnostics = _good_diagnostics(); diagnostics[key] = value
        assert probe.evaluate_probe(
            diagnostics, finite_state=True, physics_steps=1
        )["passed"] is False
    assert probe.evaluate_probe(
        _good_diagnostics(), finite_state=False, physics_steps=1
    )["passed"] is False
    assert probe.evaluate_probe(
        _good_diagnostics(), finite_state=True, physics_steps=0
    )["passed"] is False


def test_probe_source_uses_eight_envs_zero_actions_and_nonzero_failure_exit():
    source = PROBE.read_text(encoding="utf-8")
    assert 'TASK_ID = "Isaac-M1-Panda-Folded-Load-v0"' in source
    assert 'default=8' in source and 'args.num_envs != 8' in source
    assert "torch.zeros((8, 23)" in source
    assert "wrapper.step" in source
    assert '"inactive_action_exact_zero"' in source
    assert '"fold_error_within_limit"' in source
    assert '"effort_within_asset_limit"' in source
    assert '"mount_response_present"' in source
    assert "return 0 if report[\"passed\"] else 2" in source
    assert "apply_external_force_torque" not in source


def test_probe_entrypoint_preserves_failure_status_after_simulation_shutdown():
    source = PROBE.read_text(encoding="utf-8")
    assert "exit_code = main()" in source
    assert "sys.stdout.flush()" in source
    assert "sys.stderr.flush()" in source
    assert "os._exit(exit_code)" in source


def test_runbook_contains_gpu0_probe_smokes_and_long_curriculum_commands():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "cd /home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn" in text
    launcher = "TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p"
    assert launcher in text
    assert "scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 16 --device cuda:0" in text
    assert "scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 256 --device cuda:0" in text
    assert "probe-pd120-j4m2650-8x16.json" in text
    assert "probe-pd120-j4m2650-8x256.json" in text
    assert "panda_joint4=-2.650 rad" in text
    assert "--stage L0-C0 --num_envs 8 --max_iterations 1 --device cuda:0 --run_dir logs/m1_panda_folded_load/smoke-pd120-8x1" in text
    assert "--stage L0-C0 --num_envs 64 --max_iterations 10 --device cuda:0 --run_dir logs/m1_panda_folded_load/smoke-pd120-64x10" in text
    assert "--start_stage L0-C0 --num_envs 4096 --device cuda:0 --experiment_root logs/m1_panda_folded_load/foundation-v1" in text
    assert "smoke" in text.lower() and "accepted" in text.lower()
