# M1 + Panda Folded-Load Directional Evaluation Diagnostics Design

**Date:** 2026-08-25
**Status:** user-approved interactive design; written specification pending review
**Scope:** L0-C0 fixed-evaluation observability only

## 1. Problem

The fresh folded-load L0-C0 run stopped normally at update 74 after `eligible_patience_50_updates`. Its update-23 `model_best.pt` then failed all three deterministic fixed evaluations even though every serialized global gate passed:

- timeout `1.0`;
- base contact and bad orientation `0.0`;
- overall `vx_rmse=0.0354079 <= 0.04`;
- overall `wz_rmse=0.1060669 <= 0.12`;
- stationary drift below both limits;
- 16 episodes in each forward, reverse, left, and right bucket.

The only remaining failure source is the internal `directional_pass` conjunction. Existing reports do not serialize per-direction RMSE or pass state, so the retained artifacts cannot identify which direction failed.

## 2. Selected Approach

Extend fixed-evaluation reports without changing training, control, commands, rewards, thresholds, checkpoint selection, or acceptance semantics.

For each direction, serialize one nested record under `directional_metrics`:

```json
{
  "forward": {
    "episode_count": 16,
    "tracking_metric": "vx_rmse",
    "tracking_rmse": 0.0,
    "tracking_limit": 0.04,
    "base_contact_rate": 0.0,
    "bad_orientation_rate": 0.0,
    "passed": true
  }
}
```

The same schema applies to `reverse`, `left`, and `right`. Forward and reverse use `vx_rmse` with limit `0.04`; left and right use `wz_rmse` with limit `0.12`.

The top-level report additionally serializes `directional_pass`. Top-level `passed` continues to use the exact existing conjunction, now consuming the same per-direction booleans used for diagnostics. This prevents the diagnostic fields and acceptance decision from drifting apart.

## 3. Data Flow

`evaluate_records(stage, seed, episodes)` remains the only metric computation boundary:

1. validate exactly 64 unique environment records;
2. partition records into forward, reverse, left, and right buckets;
3. compute each bucket's count, tracking RMSE, contact rate, orientation rate, limit, and pass state;
4. compute the unchanged global and stationary metrics;
5. derive top-level `directional_pass` from all four bucket pass states;
6. derive top-level `passed` from the unchanged global, directional, and stationary gates;
7. atomically write the seed report through `AtomicStageArtifacts`.

No raw trajectories or per-step tensors are persisted.

## 4. Diagnostic Re-Evaluation

Re-use the rejected run's immutable checkpoint:

```text
Go2Pvcnn/logs/m1_panda_folded_load/foundation-v1/L0-C0/model_best.pt
SHA-256: f231009992ae07ae3de2560cfadb4d812fdb6cd38c8fa6deca7d4b2b8466ae8e
```

Do not overwrite the original `evaluation_seed_42/43/44.json` or `evaluation_aggregate.json`. Copy the eligible manifest and checkpoint into a new diagnostic run directory, preserving the checkpoint SHA, then run fixed seeds 42, 43, and 44 there. The diagnostic directory may never become a parent stage or resume source, even if the recomputed reports expose a passing direction set.

The purpose is attribution only: identify whether forward, reverse, left, or right tracking caused rejection.

## 5. Error Handling

- Empty directional buckets remain a hard failure and report `episode_count=0`, `tracking_rmse=null`, contact/orientation rates `null`, and `passed=false`; NaN and Infinity are never emitted.
- Each required direction must contain at least eight episodes, as before.
- Non-finite RMSE or rates raise before writing a success report.
- Seed reports remain atomic.
- A diagnostic re-evaluation failure cannot mutate the original run manifest, original reports, checkpoint, or curriculum state.
- The existing PhysX `Panda/root_joint` warning remains recorded and is unrelated to this observability change.

## 6. Verification

Pure tests must prove:

1. a valid balanced evaluation reports four passing directional records;
2. forward-only and reverse-only tracking failures identify the correct bucket and reject top-level `passed`;
3. left-only and right-only tracking failures identify the correct bucket and reject top-level `passed`;
4. contact and orientation failures are reflected in the corresponding directional record;
5. global thresholds, minimum direction counts, stationary gates, and top-level acceptance behavior remain unchanged;
6. serialized numeric fields are finite; empty-bucket unavailable values serialize as JSON `null` rather than NaN or Infinity.

Static/CPU regression covers the existing folded-load train/eval/orchestrator contracts. GPU0 then re-evaluates only the preserved `model_best.pt` for seeds 42/43/44 in the isolated diagnostic directory.

## 7. Non-Goals

This work does not resume training, alter patience, change reward weights, rebalance command sampling, loosen thresholds, promote L0-C0, create `model_final.pt`, advance to L1-C1, or claim locomotion convergence. Any behavior change requires a new diagnosis-driven design after the failing direction is known.
