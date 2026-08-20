# M1 Panda Student S1 Tasks 7–9 Training Smoke

## Purpose

Make the Student S1 path executable through observation assembly, task
registration, replay collection and a first supervised checkpoint.

## Changes

- Added the frozen 100-value observation layout and `Isaac-M1-Panda-Student-S1-v0` registration.
- Added a fresh-output replay collection CLI with strict stage/probability checks.
- Added a non-PPO supervised trainer using the existing GRU Student, DAgger losses and strict checkpoint manifest.

## Verification

- Focused coordination regression: `109 passed`.
- Python compilation and `git diff --check`: passed.
- Collection smoke: 4 environments, 20 steps, teacher-warmup, exit `0`.
- Training smoke: 1 CPU epoch from that shard, produced `best.pt` and `last.pt`, exit `0`.

## Boundary

The collection CLI currently uses a deterministic contract-level sample source
to prove replay/checkpoint/trainer execution. It is not yet a physical Isaac
Teacher side-label run. Tasks 10–12 must connect collection to Isaac, add
Student-only play/evaluation and perform GPU acceptance before claiming a
trained physical policy.

## Git Refs

- Baseline ref: `643e37c`
- Candidate ref: current Tasks 7–9 work
- Key files: `m1_panda_student_observation.py`, `m1_panda_student_s1_env_cfg.py`,
  `m1_panda_wbc_collect.py`, `m1_panda_student_train.py`
