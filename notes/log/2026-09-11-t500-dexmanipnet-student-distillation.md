# T500 DexManipNet Compact Student Distillation

## Purpose

Record Task 7's offline compact student, strict artifact boundary, and synthetic-only CPU evidence.
No archive, shard, ensemble checkpoint, student weight, Isaac runtime, MPC, or Hand QP artifact was
added to the repository.

## Stage And Related Todo

T500.5 / Task 7; see the [T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Contract

- Real distillation re-verifies the Task 5 aggregate before every split load and accepts only a
  canonical-self-hash-verified, `production_deployable=true` Task 6 ensemble. Every selected
  checkpoint is SHA-verified before a `weights_only=True` load, then checked against the exact
  member seed, aggregate pin, model architecture, state-dict keys, shapes, dtypes, and finiteness.
- Fixed-seed teacher mixture samples are scored under the student; no component-to-component
  alignment is used. The ensemble remains a member-average of frozen four-component distributions,
  rather than constructing an invalid eight-plus-component `MixtureDistribution`.
- The staged atomic artifact contains exactly `metadata.json`, `student.pt`, `metrics.json`, and
  `latency.json`. The loader rejects missing/extra/symlink files, metadata self-hash tampering,
  weight SHA drift before Torch deserialization, schema/pin/provenance drift, and incompatible
  state dictionaries. It loads only with `weights_only=True` and returns an eval-only model.
- Real export requires NLL delta at most `0.05 nat/dim`, student endpoint RMSE no more than `5%`
  above teacher endpoint RMSE, both zero-baseline improvements at least `10%`, repeated weight SHA,
  and measured CPU p99 below `2 ms` after 100 warm-ups and 1000 measures. Synthetic provenance is
  permanently `production_approved=false`, regardless of measured values.

## TDD And Verification

The initial focused RED failed at collection because `artifact.py` did not exist. The first GREEN
then exposed a real synthetic integration defect: flattening Task 6 ensemble members would violate
the frozen four-component `MixtureDistribution` contract. A focused RED test was added for
member/component aggregation; the fix retains per-member four-component density evaluation.

Focused final verification:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_distill.py \
  tests/test_m1_bimanual_expert_prior_artifact.py
```

Result: `8 passed in 1.67s`.

The required synthetic smoke used a unique `/tmp/t500-student-smoke.*` directory, two epochs, and
a compact `16,16` smoke architecture. It reloaded the produced artifact successfully; its finite
metrics included NLL delta `-0.00520067 nat/dim`, endpoint improvement `2.174%`, first-step
improvement `1.509%`, and CPU p99 `0.109395 ms`. It explicitly emitted
`production_approved=false`; these values are smoke evidence only and fail the real-data
zero-baseline gates independently.

Tasks 1--7 regression result: `125 passed in 13.16s`. Both new CLI `--help` commands and
`py_compile` for the three Task 7 sources exited `0`.

## Git Refs

- Baseline Ref: `f770402388b55e24b293d0ae899cff404085e0d3`
- Candidate Ref: Task 7 worktree state based on `f770402`
- Key Files: `artifact.py`, `m1_dual_panda_o6_distill_fingertip_prior.py`,
  `m1_dual_panda_o6_eval_fingertip_prior.py`, and the two Task 7 tests.

## Follow-up

No real external data or deployable ensemble is available locally, so the real production gate is
unverified. Task 8 may consume only a real gate-approved student artifact; the synthetic smoke
artifact remains non-production by contract.

## Review Hardening

Post-Task-7 review made the metadata self-hash mandatory and added raw byte-count/SHA bindings for
both reports. The loader recomputes approval from pinned metric/latency gates, synthetic provenance
forces false, and metadata records fixed temporal-loss coefficients plus a latency-excluding
reproducibility fingerprint. Distillation independently retrains with the same seed before it can
approve a real artifact. The comparison evaluator now requires versioned, finite, domain-valid
reports with matching positive trial counts. RED covered absent self-hash, report-only approval
tampering, missing bindings, temporal contribution, and nonsensical comparison values; focused
final result was `13 passed`, Tasks 1--7 were `130 passed in 13.04s`, and a two-epoch synthetic
repeat reloaded with p99 `0.113534 ms` and `production_approved=false`.

## Loader Ordering And Comparison Provenance Correction

Report JSON is now schema/pin/SHA/byte-count/approval-validated after mandatory metadata and
weight SHA checks but before model construction or `torch.load`; a spy RED test proves report
tampering produces zero deserialization calls. Prior-off/on comparison now requires equal canonical
evaluation-manifest, scenario, ordered trial-set, safety-definition, and controller-contract SHA
provenance; only the prior artifact/config is allowed to differ outside this comparability object.

Metric/prior-identity hardening rejects bool numeric metrics and inconsistent derived values, records
the deterministic repeat flag, and requires disabled sentinels versus enabled artifact/config SHA
pins. Focused `16 passed`; Tasks 1--7 `133 passed in 13.26s`.
