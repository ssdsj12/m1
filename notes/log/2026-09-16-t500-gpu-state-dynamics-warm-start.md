# T500 GPU state, reduced dynamics and warm-start verification

## Purpose / stage / todo

Task3 in [approved GPU plan](../../docs/superpowers/plans/2026-09-15-t500-gpu-bimanual-rti-latent.md),
related [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md). Baseline `e280b8d`,
candidate `1d12e53`; independent review spec compliant / quality Approved, no findings.

## Changes and input invariants

Persistent CUDA float32 `[B,111]` state mirror with exact int64 timestamps;
explicit CPU dataclass upload boundary. Full59 generalized inertia and12 wheel
constraints condensed through71x71 KKT to43effort affine acceleration/contact maps.
Authoritative incoming actuation matrix determines generalized reconstruction,
not59 independently controlled commands. Finite, symmetry/SPD, rank and singular
checks reject rows without poisoning other rows or retained maps. Rank policy is
conservative float32 numerical rank; no physical safety threshold changed.

Separate accepted/shifted `[B,25,43]` actions and `[B,26,D]` states: shift terminal
hold and overwrite node0 from measurement. Environment/layout/phase generation
tokens and resets invalidate only affected rows, permanently until new acceptance.
Returned buffers are borrowed storage: consumers must check current validity.

## TDD and actual GPU0 checks

Missing-module RED exit2; additional invariant RED4 failures, default-dtype RED1
failure. Implementer twice GREEN82 tests on actual RTX5070 CUDA0, no skipped tests.
Parent fresh command from worktree root:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$PWD/Go2Pvcnn" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_state.py Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_dynamics.py Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_warm_start.py Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py Go2Pvcnn/tests/test_m1_bimanual_reduced_dynamics.py Go2Pvcnn/tests/test_m1_bimanual_full_action_teacher.py Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py
```

Exit0: **82 passed in2.31s**. Ten warm-up and100 resident updates:

- state21 storages stable, allocated/reserved delta0;
- dynamics38 pointers stable, allocated/reserved delta0;
- warm-start16 pointers stable, allocated/reserved delta0.

Dispatch guards reject newly backed tensor storage and host scalar extraction;
resident loops pass. CPUfloat64/GPUfloat32 map/reconstruction parity uses named
comparison atol/rtol3e-5, including coupled SPD passive/mimic inertia and effort
limit fixtures. No Torch default-dtype leakage.

## Conclusion / follow-up / refs

Infrastructure Task3 complete with independent review Approved.
No production prior, RTI planner, CUDA graph, online timing, Isaac physics or
formal30-trial acceptance claim. SVD/Cholesky cost is unbenchmarked and library
scratch/synchronization not proven graph-safe. Next: eager structured RTI/QP and
parallel line search, then scan/Triton, graph and full acceptance gates.

- Last Feature / Verified Commit: `1d12e53` (focused scope).
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.
- Key Files: private `gpu_rti/state_adapter.py`, `reduced_dynamics.py`, `warm_start.py`
  and their three focused tests under `Go2Pvcnn`.
