# M1 + Panda Phase 6 Bridge and Normalizer Continuity Design

## 1. Context and authoritative evidence

The corrected Phase 6 v6 pipeline completed its accepted ten-update pilot,
safe 100-update short run, nine zero-pair calibration workers, and fifteen
candidate workers for seeds 42, 43, and 44.  Every fixed-condition worker ran
for 4000 steps and completed successfully, but the atomic promotion manifest
reported `status=rejected` and `accepted=false`.  Therefore the 3000-update
long run correctly did not start.

The v6 promotion decisions were:

- u000, u025, and u100: `aggregate_equivalent`;
- u050 and u075: `seed_42_wrench_regression`;
- no hard failure, MPC/QP infeasibility, contact loss, intervention, or
  residual saturation occurred.

The learned u100 deterministic normalized action remained very small.  A real
GPU0 seed-42 rollout measured per-channel RMS between approximately `0.00092`
and `0.00323`, with maxima below `0.00872`.  Under the existing physical
limits this is only a small correction to an already strong zero-residual WBC
baseline.  One hundred updates therefore did not produce an improvement large
enough to exceed the calibrated PhysX noise tolerances.

The continuation path also exposes a state-continuity defect: the local
`EmpiricalNormalization` module stores mean, variance, and standard deviation
in its state dictionary, but its sample count is a plain Python integer.
Loading a checkpoint for additional training restores the statistics but not
the count, so the first new batch can almost completely overwrite the restored
statistics.  A bridge or long run must not continue until this count is saved
and restored fail closed.

## 2. Selected approach

Add a protected `bridge` stage that continues the accepted physical state of
the completed v6 short from update 100 through update 300.  Preserve all
existing physical limits, reward coefficients, KL limits, learning-rate
bounds, fixed-condition seeds, 4000-step worker length, noise calibration, and
promotion tolerances.

The bridge fixes training-state continuity, publishes five formal candidates
at completed updates 100, 150, 200, 250, and 300, and then runs the same
process-isolated promotion procedure.  Only an atomic promotion manifest with
`accepted=true` may authorize the existing 3000-update long stage.

This is preferred over changing the reward because the v6 reward and safety
path are physically stable and fully exercised.  It is also preferred over
accepting an equivalent candidate because that would weaken the explicit
fail-closed requirement instead of demonstrating measurable improvement.

## 3. Normalizer training-state contract

Actor and critic empirical normalizers must persist all state required for an
exact training continuation:

- running mean;
- running variance;
- running standard deviation;
- processed sample count.

The count must be a non-negative integer represented in the checkpoint state
and restored on load.  Loading a checkpoint with inconsistent, negative,
non-integral, or non-finite normalizer state fails before environment training
continues.

Existing v6 u100 predates the count field.  The bridge may migrate this one
parent checkpoint only by deriving the exact count from immutable lineage:

```text
completed_updates * num_steps_per_env * num_envs
= 100 * 256 * 8
= 204800 samples
```

The migration is allowed only when the parent is the hash-matched u100
candidate named in the safe-complete v6 short manifest and its actor/critic
mean, variance, and standard deviation dictionaries are present and non-empty.
All new bridge and long checkpoints must contain an explicit count; no further
inference fallback is permitted.

Loading for inference keeps the normalizers in evaluation mode and never
changes the count.  Loading for bridge or long training restores the count
before the first normalized observation is processed.

## 4. Bridge lineage and execution contract

`bridge` is a distinct guarded stage, not a renamed long run and not an
extension that mutates the completed short directory.

It consumes:

- the v6 safe-complete short manifest;
- its hash-matched u100 checkpoint;
- matching asset, task configuration, reward, runtime, pilot schema, and
  source-bundle hashes.

It restores:

- model parameters and trainable policy standard deviation;
- optimizer state and adaptive learning rate;
- actor and critic empirical normalizer state, including count;
- completed-update offset 100.

The bridge runs exactly 200 additional PPO updates on GPU0 with eight
environments.  The public completed-update sequence therefore ends at 300,
while the runner may use a local zero-based loop internally.  Every update
retains the current online safety controller:

- zero hard failures;
- MPC feasible rate at least 0.99;
- QP feasible rate exactly 1.0;
- four-wheel contact rate exactly 1.0;
- every residual saturation fraction below 0.01;
- finite optimizer, KL, value-loss, action-standard-deviation, reward, and
  physical diagnostics.

Any violation stops the bridge, records an atomic safety-stopped manifest, and
blocks promotion.  A safe bridge publishes candidates at total completed
updates 100, 150, 200, 250, and 300.  The validated parent u100 is atomically
copied into the bridge directory as its u100 candidate, and the copy must have
the identical SHA-256 hash.

## 5. Promotion contract

The bridge promotion driver keeps the existing fixed-condition protocol:

1. Run nine fresh zero-vs-zero calibration workers: three pairs for each of
   seeds 42, 43, and 44.
2. Run fifteen candidate workers: five bridge candidates for each of the same
   three seeds.
3. Run every worker in an isolated Isaac Sim process for exactly 4000 steps on
   GPU0.
4. Validate status, step count, seed set, checkpoint hash, source hash,
   normalizer dictionaries, and promotion lineage before using a worker result.
5. Calibrate tolerances exactly as before and retain all hard gates,
   lexicographic rank metrics, wrench-regression checks, and slip-regression
   checks unchanged.

The result is one atomic schema-v3 promotion manifest.  Reusing a
complete worker is allowed only through the existing hash-validated resume
contract.  Failed or starting workers may be retried; mismatched evidence is
never reused.

If no candidate is accepted, `best_checkpoint` remains null and no long
directory or process may be created.  The rejection remains evidence for the
next architecture review; thresholds are not weakened in place.

## 6. Conditional long-run contract

The 3000-update long run starts only when the bridge promotion manifest has
`accepted=true` and its selected checkpoint hash matches the file on disk.
Before launch, the selected checkpoint must contain explicit actor and critic
normalizer counts in addition to their statistic tensors.

Long restores the selected policy, policy standard deviation, normalizers,
counts, optimizer state, and adaptive learning rate.  Starting a fresh
optimizer is forbidden because it would break the bridge training-state
continuity that this design establishes.

The long run remains on GPU0 with eight environments for exactly 3000 updates.
It keeps the same online safety controller, learning-rate ceiling, KL abort,
checkpoint protection, atomic manifest, and fail-closed stop behavior.  A
running process is progress only.  Completion requires an inspected manifest
showing safe completion of all 3000 requested updates and valid final/best
checkpoint lineage.

## 7. Interfaces and artifacts

The implementation may extend the existing residual train and promotion
entrypoints, but must keep their current pilot, short, promotion, and long
contracts compatible.

New artifacts use fresh immutable roots, for example:

```text
logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7/
logs/m1_panda_arm_mpc_residual/long_s42_normalizer_continuity_v7/
```

The bridge run manifest records parent short path/hash, parent checkpoint
path/hash, parent completed updates, starting normalizer counts, completed
bridge updates, total completed updates, candidate paths/hashes, stop reason,
and every existing source-lineage field.

The promotion manifest records the bridge manifest path/hash and the total
completed-update value for each candidate.  The long manifest records both
the promotion and selected-checkpoint lineage.

## 8. Testing and acceptance

CPU tests must prove:

- empirical normalizer count round-trips through state dictionaries;
- inference never mutates the restored count;
- training resumes from the restored count instead of overwriting statistics;
- the one-time u100 migration derives exactly 204800 and rejects any lineage
  mismatch;
- bridge stage limits and candidate update labels are exactly
  100/150/200/250/300;
- bridge safety stops block promotion;
- promotion accepts bridge manifests and rejects malformed or hash-mismatched
  lineage;
- long refuses `accepted=false`, missing counts, and mismatched checkpoint
  hashes;
- existing pilot, short, promotion, and long tests remain green.

GPU acceptance requires, in order:

1. a safe-complete 200-update bridge ending at total update 300;
2. five valid normalized bridge candidates;
3. nine complete noise workers and fifteen complete candidate workers;
4. a parsed promotion manifest with `accepted=true`;
5. only then, a live and continuously monitored 3000-update long run;
6. final proof of safe 3000/3000 completion before Phase 6 is marked complete.

No reward coefficient, physical residual limit, promotion tolerance, required
seed, fixed worker length, or accepted-only long gate changes in this design.
