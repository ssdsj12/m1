# T500 DexManipNet Frozen O6 Prior Runtime

## Purpose

Record Task 8's runtime-only adapter. No external data, source-hand assets, weights, training,
Isaac, or Hand-MPC integration was added.

## Stage And Related Todo

T500.5 / Task 8; see the [T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Contract

- `FrozenO6FingertipPrior.from_artifact()` goes through the strict artifact loader and construction
  rejects a student unless `production_approved` is exactly `true`.
- Runtime input is strict CPU float64 O6 geometry: `(5,3)` fingertip positions, `(15,6)` folded
  Jacobian, six-axis qd, five-bit CPU contact mask, phase, and baseline six-axis qd. It packs the
  frozen `15 + 15 + 5 + 7` float32 order only.
- In `no_grad`, eval, and a per-adapter lock, component selection uses bounded diagonal precision
  and baseline tip velocity with the logit term. The first future node is returned as finite CPU
  float64 mean/precision; contacted rows are exactly zero precision.
- No target is persisted. Safe phases, invalid input, invalid/nonfinite output, invalid
  probabilities/precision, exceptions, and measured timeout return an explicit disabled result.

## TDD And Verification

RED was observed as the expected missing `expert_fingertip_prior.runtime` module collection error.
The focused test then covered frozen packing, deterministic selection, log-std/precision bounds,
safe/nonfinite failure, contacted-row zeroing, timeout discard, input mismatch, and non-production
construction rejection.

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py \
  tests/test_m1_bimanual_expert_prior_model.py \
  tests/test_m1_bimanual_expert_prior_train_static.py \
  tests/test_m1_bimanual_expert_prior_distill.py \
  tests/test_m1_bimanual_expert_prior_artifact.py \
  tests/test_m1_bimanual_expert_prior_runtime.py
```

Result: `139 passed in 13.15s`.

## Review Hardening

The runtime import graph is now isolated even transitively: artifact hashing uses a local stdlib
reader rather than the fetch/extraction module, and a fresh-process graph test rejects all offline
data modules plus HF/HDF5/Isaac imports. Production construction accepts no caller-supplied model
or `LoadedStudent`; only `from_artifact()` can create an adapter after strict loading. The private
test seam is explicitly named `_test_only_prior`.

Inference now runs in one persistent daemon fork worker with a deep-copied private model. A queue
deadline returns immediately on a hung call, terminates and permanently poisons that worker, and
subsequent queries bypass it. The worker catches every model-side `BaseException`; RuntimeError,
IndexError, and AssertionError tests all return finite `prior_exception` diagnostics. Config now
requires a finite, strictly positive logit weight. Focused result: `13 passed in 1.91s`; Tasks 1–8:
`147 passed in 14.65s`.

The runtime source scan found no forbidden direct import; a fresh-process import found no
transitive `huggingface_hub`, `h5py`, or `isaacgym` module. `py_compile` for the adapter/test
completed successfully. Repository-wide `git diff --check` remains blocked by the pre-existing
unrelated `.superpowers/sdd/task-6-brief.md` EOF whitespace warning; the Task 8 scoped diff is
clean.

## Git Refs

- Baseline Ref: `85534992bb5fa50a896a48d70f40f852850e30c0`
- Candidate Ref: Task 8 runtime worktree change
- Key Files: `expert_fingertip_prior/runtime.py`,
  `test_m1_bimanual_expert_prior_runtime.py`

## Follow-up

Task 9 must consume this target only after its unchanged baseline Hand QP succeeds, and must retain
the same-cycle baseline on every prior/second-QP failure. No real production artifact or physical
acceptance evidence is available locally.
