# M1 + Panda Phase 6 Guarded Learning-Rate Design

## Status

Approved by the user on 2026-08-31 as corrective option 1.

## Problem

The fresh seed-42 Phase 6 pilot at
`Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3`
finished safely but was rejected.  All ten updates preserved the physical
contract (`hard_failure_count=0`, MPC/QP/contact rates equal to `1.0`), and
the median value loss fell to `6.2402`, but every update reported a KL abort
and the median completed mini-batch count was only `4`.

The optimizer summaries identify the cause.  The configured learning rate
started at `1e-5`, but per-mini-batch adaptive increases raised it to
`2.25e-5`, `5.0625e-5`, and then the `1e-4` maximum in the first three
updates.  The adaptive PPO implementation only decreases the rate above
`2 * desired_kl = 0.020`, while the task aborts the update above
`kl_abort_threshold = 0.015`.  Consequently, KL values in `(0.015, 0.020]`
abort training without activating the decrease branch, while earlier low-KL
mini-batches continue to activate the increase branch.

## Selected Design

Apply a task-local upper bound to the Phase 6 residual PPO configuration:

- keep `learning_rate=1e-5`;
- change `max_learning_rate` from `1e-4` to `1e-5`;
- keep `min_learning_rate=1e-6`, so adaptive downward correction remains
  available;
- keep `schedule="adaptive"`, `desired_kl=0.01`, and
  `kl_abort_threshold=0.015` unchanged;
- do not alter the shared PPO adaptive-learning-rate implementation;
- do not relax pilot, short, promotion, or long acceptance gates.

This change prevents unvalidated upward amplification on this sensitive
residual task while preserving fail-closed KL protection and downward
adaptation.  It does not affect folded-load, coordinated, teacher, or other
PPO tasks.

## Files and Boundaries

- Modify
  `Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py` only for the
  task-local maximum learning rate.
- Strengthen
  `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py` so the
  contract requires both initial and maximum learning rates to equal
  `1e-5`, with the minimum remaining `1e-6`.
- Record the rejected v3 pilot, root-cause evidence, corrective commit, and
  fresh-pilot result in
  `notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md`.

## Verification and Promotion Rules

1. A focused static test must fail before the configuration change and pass
   afterward.
2. The relevant PPO/config/pilot/lineage CPU suite and `compileall` must pass.
3. A new seed-42 ten-update pilot must use a fresh output directory and
   lineage hashes.  The v3 pilot may remain as immutable failure evidence.
4. The fresh pilot must report `accepted=true`, at most three KL aborts,
   median completed mini-batches at least six, median value loss below 100,
   and all existing physical/std/finite gates.
5. A pilot failure stops promotion.  No manual manifest editing, checkpoint
   copying, or gate bypass is allowed.
6. Only an accepted pilot may launch the 100-update short run.  Only an
   accepted three-seed promotion manifest may launch the 3000-update long
   run.

## Rejected Alternatives

- Changing the global PPO scheduler was rejected because it would alter all
  training tasks and is unnecessary for the observed task-local failure.
- Increasing the KL abort threshold or weakening the pilot abort/batch gates
  was rejected because it would hide optimizer instability instead of
  removing its cause.
- Changing `desired_kl` was rejected because the approved Phase 6 contract
  fixes it at `0.01`.
