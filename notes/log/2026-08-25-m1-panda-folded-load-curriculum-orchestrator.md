# 2026-08-25 M1 + Panda folded-load curriculum orchestrator

## Purpose

Record Task 8 TDD evidence for strict stage sequencing, accepted SHA lineage, subprocess train/eval execution, and stop-on-failure rollback.

## Stage

T400.10b / folded-load curriculum Task 8.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Verification

The valid RED was `5 failed` because the curriculum script did not exist. GREEN and related verification:

```bash
cd Go2Pvcnn
python -m py_compile scripts/m1_panda_folded_load_curriculum.py
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_orchestrator.py \
  tests/test_m1_panda_folded_load_scripts.py \
  tests/test_m1_panda_folded_load_training_guard.py
```

Result: `17 passed in 0.86s`; diff check exits `0`.

## Key Contracts

- Executes only `L0-C0 -> L1-C1 -> L1-C2 -> L1-C3 -> L1-C4 -> L2-D1 -> L2-D2 -> L2-D3`.
- Each stage launches one training process, then fixed evaluations 42, 43, and 44.
- Process return code is insufficient: acceptance is revalidated from L0 through the current stage.
- Each link verifies accepted state, immediate parent stage/path, parent manifest SHA, parent final checkpoint path/SHA, and current final checkpoint SHA.
- Starting from a non-L0 stage requires a complete accepted prefix.
- First failure atomically writes `curriculum_state.json`, keeps the prior final checkpoint as rollback, and never creates the next stage.
- There is no difficulty fallback or continue-after-failure branch.

## Boundary

CPU fake-executor/static verification only. Real subprocess Isaac launches and GPU resource behavior remain for Tasks 9–10.

## Git Refs

- Baseline Ref: `162b5e2`
- Candidate Ref: pending Task 8 commit
- Current Work Ref: `codex/m1-panda-ppo-stability`
- Key Files:
  - `Go2Pvcnn/scripts/m1_panda_folded_load_curriculum.py`
  - `Go2Pvcnn/tests/test_m1_panda_folded_load_orchestrator.py`
