# 2026-08-25 M1 + Panda folded-load curriculum contracts

## Purpose

Record Task 1 RED/GREEN evidence for the new folded-load locomotion foundation.

## Stage

T400.10b / folded-load curriculum Task 1.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [approved design](../../docs/superpowers/specs/2026-08-25-m1-panda-folded-load-locomotion-curriculum-design.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Command And Procedure

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_curriculum.py
```

The RED run failed at collection because the new pure curriculum module did not exist. The minimal implementation was then added and the same command was rerun.

## Input Conditions

- Worktree: `codex/m1-panda-ppo-stability`
- Baseline ref: `84d4b4d`
- No Isaac application, GPU, checkpoint, asset, or long-running process was used.
- The rejected coordinated policy was not loaded.

## Key Metrics

- Focused result: `7 passed in 0.78s`.
- Stage chain: all eight stages and parent links checked.
- Command levels: C0-C4 exact `vx/wz` limits checked.
- DR: deterministic L0/L1 and exact D1/D2/D3 root/leg/friction ranges checked.
- Protected reset fields: root Z, wheel position, Panda position/velocity, and restitution remain zero.
- Commands: seeded repeatability, family proportions, zero lateral speed, stage bounds, episode families, and deterministic balanced 64-env evaluation table checked.

## Result

Pass for the pure contract layer. This does not verify ActorCritic masking, PPO, Isaac reset writes, dynamic Panda loading, locomotion behavior, or stage acceptance.

## Follow-Up

Implement Task 2 active-action masking in the vendored ActorCritic with exact inactive output, probability, gradient, and checkpoint tests.

## Git Refs

- Baseline Ref: `84d4b4d`
- Candidate Ref: pending Task 1 commit
- Key Files:
  - `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py`
  - `Go2Pvcnn/tests/test_m1_panda_folded_load_curriculum.py`
