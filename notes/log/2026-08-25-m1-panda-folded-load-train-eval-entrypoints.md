# 2026-08-25 M1 + Panda folded-load train/eval entrypoints

## Purpose

Record Task 7 TDD evidence for one-stage guarded training, lossless completed-episode transfer, strict parent lineage, runner stability diagnostics, and fixed three-seed evaluation.

## Stage

T400.10b / folded-load curriculum Task 7.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Command And Procedure

The RED run produced `6 failed, 4 passed`: both scripts and the wrapper record/evaluation APIs were absent. After implementation:

```bash
cd Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_scripts.py \
  tests/test_m1_panda_folded_load_training_guard.py \
  tests/test_m1_panda_folded_load_wrapper.py \
  tests/test_m1_panda_folded_load_curriculum.py \
  tests/test_m1_panda_folded_load_ppo.py \
  tests/test_rsl_rl_active_action_mask.py \
  tests/test_rsl_runner_iteration_callback.py \
  tests/test_rsl_ppo_adaptive_schedule.py \
  tests/test_m1_panda_coordinated_training_guard.py \
  tests/test_m1_panda_teacher_noise_std.py
```

Result: `72 passed in 1.77s`. `py_compile` and `git diff --check` also exit `0`.

## Key Metrics And Contracts

- L0-C0 rejects every parent/resume path; every later stage requires the immediate parent's `accepted=true` manifest, final checkpoint, and matching SHA-256.
- A run directory must be empty. Manifest records PID, full command, asset SHA, parent manifest/checkpoint SHA, full PPO config, and `[1]*16+[0]*7`.
- Parent load skips optimizer state, preserves actor/critic weights, clamps inherited policy std to `0.01`, and resets the stage iteration counter.
- Wrapper drains each completed `EpisodeRecord` exactly once with its original environment ID and unflattened three-axis command.
- TensorBoard adds mean/max KL, KL abort, completed minibatches, gradient norm, active-only std min/max, inactive action, fold, effort, joint-limit, and hard-failure diagnostics.
- Evaluation is fixed to 64 environments, deterministic policy means, balanced commands, and seeds `42/43/44`; only three passing atomic reports publish byte-identical `model_final.pt` and set acceptance.
- No external-force/torque API is present in either new entrypoint.

## Boundary

This is CPU pure/static verification. Isaac application startup, real combined-articulation stepping, TensorBoard emission in a live run, one full 64-env evaluation, and GPU0 training remain unverified until Tasks 9–10. Task 8 curriculum orchestration is still open.

## Git Refs

- Baseline Ref: `8ab909b`
- Candidate Ref: pending Task 7 commit
- Current Work Ref: `codex/m1-panda-ppo-stability`
- Key Files:
  - `Go2Pvcnn/scripts/m1_panda_folded_load_train.py`
  - `Go2Pvcnn/scripts/m1_panda_folded_load_eval.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py`
  - `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - `Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py`
