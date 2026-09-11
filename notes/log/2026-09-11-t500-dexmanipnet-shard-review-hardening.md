# T500 DexManipNet Shard Review Hardening

## Purpose

Close the Task 5 review findings at the offline conversion trust boundary: verify the exact local
archives and clean pinned ManipTrans source immediately before conversion, make collision normals
trustworthy, derive target-rate relative normal speed analytically, and bind every artifact file to
the aggregate manifest.

## Stage

T500.5 / Task 5 review hardening. This only changes offline DexManipNet fetch/conversion,
preprocessing, deterministic storage, and their CPU tests; it does not change training, Isaac, or
any MPC/runtime behavior.

## Related Todo

[T500.5 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Review Contract Closed

- Conversion now calls the shared Task 2 no-optional-lock Git checks before it examines extracted
  data. It recomputes both archive byte counts and SHA-256 values, requires the stored download
  manifest to exactly equal those facts, and rejects missing/mutated/symlink archives plus any
  tracked, untracked, ignored, submodule, commit, or tree drift in ManipTrans before an output
  directory is created.
- The aggregate body binds those recomputed archive `{name,size_bytes,sha256}` records and the
  verified ManipTrans `{commit,tree}` facts. The prior archive/source manifest digests are retained
  as evidence locators.
- Watertight, consistently wound inward collision meshes are deterministically inverted to
  positive-volume outward meshes. Inconsistent winding is rejected, so closest-face normals are
  never silently trusted.
- Object-relative fingertip points are resampled at the target rate with one `CubicSpline`; the
  same spline's analytic first derivative is projected onto target closest-face normals. No 60 Hz
  finite difference is used for normal speed.
- `audit.jsonl` SHA-256 and byte count are included in the canonical aggregate body. The verifier
  recomputes the aggregate body hash, audit hash/count, every declared shard hash, and rejects
  missing, extra, unsafe, or unsorted shard paths.

## Verification

Focused review regression from `Go2Pvcnn`:

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py
```

Result: `28 passed in 2.48s`.

The focused fixtures cover mutable/missing archives and source checkout rejection before output,
inward normalization/inconsistent-winding rejection, a cubic moving-object analytic normal-speed
oracle, and independent audit, shard, and aggregate mutation rejection.

Task 1–5 CPU regression:

```bash
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py
```

Result: `108 passed in 3.60s`.

Additional gates: both fetch and conversion `--help` exit `0`; `py_compile` over modified
production modules, CLIs, and focused tests exits `0`; `git diff --check` exits `0`.

## Input Conditions And Limits

- Baseline Ref: `51d52527574e48aaaaeb5c1ee12df8aa9ad5d62c`
- Candidate Ref: `3ccea5d` (`fix: harden DexManipNet shard provenance`)
- Tests use temporary Git repositories, archive payloads, synthetic HDF5 rollouts, and box/OBJ
  meshes. Neither production archive was downloaded or scanned.
- No production object-mesh acceptance rate, production shard SHA, training result, Isaac result,
  GPU result, or physical acceptance claim follows from this CPU-only evidence.

## Conclusion And Follow-up

Task 5's review boundary is closed at the synthetic offline contract level. The next functional
work remains Task 6's offline expert ensemble, consuming only hash-verified aggregates and shards.

## Git Refs

- Last Feature Commit: `3ccea5d` (`fix: harden DexManipNet shard provenance`)
- Last Verified Commit: `3ccea5d`
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`
- Key Files: `download.py`, `preprocess.py`, `storage.py`, fetch/conversion CLIs, and Task 5 tests.
