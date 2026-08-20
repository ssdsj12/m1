# M1 Panda Student S1 Tasks 10–12 Evaluation Smoke

## Purpose

Add strict Student-only play and exact three-seed evaluation orchestration.

## Changes

- Added `m1_panda_student_evaluation.py` with completion, finite, QP/contact,
  reset and Teacher-relative hard-gate contracts.
- Added `m1_panda_student_play.py`; checkpoint loading is strict and Teacher
  labels, when requested, cannot enter execution.
- Added `m1_panda_student_eval.py` for exactly seeds `42,43,44`, at least 64
  environments and exactly 4000 steps.

## Verification

- Coordination regression: `109 passed`.
- CPU evaluation used the smoke checkpoint from Tasks 7–9 for three seeds,
  64 environments and 4000 steps each; all child processes exited `0` and
  `ranking.json` was generated.
- Every row reported `teacher_execution_count=0`.

## Boundary

This is a contract-level CPU evaluation. The current Play/collection sample
source is not a physical Isaac Teacher side-label loop, so no Student behavior
or C1a parity claim is made. The remaining gate is Isaac physical collection,
Student action application through impedance, and GPU acceptance.

## Git Refs

- Baseline ref: `c8efbf8`
- Candidate ref: current Tasks 10–12 work
- Key files: `m1_panda_student_evaluation.py`, `m1_panda_student_play.py`,
  `m1_panda_student_eval.py`
