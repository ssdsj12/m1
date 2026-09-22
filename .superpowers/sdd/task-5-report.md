# Task 5 Report: RialTo object catalog and GPU0 smoke verification

## Result

The offline verifier and evidence contract are implemented. The RialTo assets
are not materialized in this checkout, so the catalog verification is
explicitly `preparation_required`; it does not claim an object or GPU0 pass.

Implemented files:

- `Go2Pvcnn/scripts/verify_m1_rialto_object_catalog.py`
- `Go2Pvcnn/scripts/verify_m1_dual_panda_o6_asset.py` (optional atomic
  `--report-path`; existing gates and stdout remain unchanged)
- `Go2Pvcnn/tests/test_m1_rialto_object_verification.py`
- `Go2Pvcnn/artifacts/m1_rialto_object_catalog/catalog_verification.json`
- `Go2Pvcnn/artifacts/m1_rialto_object_catalog/aggregate.manifest.json`

The verifier checks the six expected source records and catalog classes,
source/resolved SHA-256 values, repository-relative paths, USD dependency
closure for materialized USDs, physical/collision metadata, deterministic
candidate IDs, and explicit target/obstacle partitioning. It never imports or
calls the network-enabled preparation path. GPU0 is started only after all
offline prerequisites pass; contact initialization is false unless the real
scene smoke passes.

## Fresh evidence

Command:

```text
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_rialto_object_catalog.py --offline-only
```

Result: exit `1`, `verification_status=preparation_required`,
`hard_gates_passed=false`, `offline.status=preparation_required`,
`gpu_smoke.status=preparation_required`, and 15 missing/unmaterialized source,
resolved-USD, or resolved-SHA prerequisites. The preparation command is
recorded in the JSON report and requires explicit `--allow-network`.

Evidence aggregate SHA-256:

```text
f795db7748bb2c57573b18880c597b324a823fe1d19109fdd1c5face34508bbf
```

## Tests and compatibility gates

```text
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_rialto_object_verification.py
5 passed in 0.84s

PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_rialto_object_verification.py \
  tests/test_m1_object_catalog.py tests/test_m1_object_scene.py \
  tests/test_m1_object_target_contract.py \
  tests/test_m1_dual_panda_o6_asset_static.py
48 passed in 0.95s

PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  scripts/verify_m1_rialto_object_catalog.py \
  scripts/verify_m1_dual_panda_o6_asset.py \
  tests/test_m1_rialto_object_verification.py
exit 0
```

The unchanged legacy asset verifier was also run headlessly on GPU0 through
Isaac Lab with 2,000 physics steps and `--report-path`. Its JSON report has
`hard_gates_passed=true`, `physics_steps=2000`, `measured_physical_dof_count=53`,
and `nonfinite_count=0`. The Isaac run emitted the pre-existing OmniPBR
dependency warnings allowed by the legacy verifier.

## Limitations

- The six RialTo source files and converted cup/cube/cylinder USDs are absent;
  no runtime download or substitute asset was used.
- Consequently the real USD dependency checks for those files and the required
  GPU0 scene smoke with bottle/cup/cube/cylinder contact initialization remain
  blocked on explicit asset preparation.
- The generated evidence records the blocked state rather than a pass. After
  preparation, rerun the verifier without `--offline-only`; it will inspect
  every local hash/dependency and then launch the headless GPU0 smoke.

## Review Fix Round

The independent review findings are addressed in the working tree:

- `inspect_usd` now accepts an explicit pinned `asset_root` and classifies every
  local USD reference by canonical containment. Existing absolute references
  outside the root, `..` traversal references, network/resolver references, and
  symlink targets that escape the root fail the dependency gate. Absolute
  references that resolve inside the root remain valid.
- The preparation path passes the destination root into USD inspection before
  publishing converted outputs. The offline catalog verifier passes its pinned
  asset root into the same inspection contract and records
  `outside_root_dependencies` in per-class evidence.
- GPU smoke object and contact lookups use `InteractiveScene` mapping access
  (`scene[object_id]` / `scene[name]`) with explicit `KeyError` handling. Missing
  entries now produce a failed smoke result with diagnostic names instead of
  attribute-based lookup.

Review-fix regression:

```text
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_rialto_object_assets.py \
  Go2Pvcnn/tests/test_m1_rialto_object_verification.py
25 passed in 1.08s

PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_rialto_object_assets.py \
  Go2Pvcnn/tests/test_m1_rialto_object_verification.py \
  Go2Pvcnn/tests/test_m1_object_catalog.py \
  Go2Pvcnn/tests/test_m1_object_scene.py \
  Go2Pvcnn/tests/test_m1_object_target_contract.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_asset_static.py
68 passed in 1.17s
```

The new tests cover absolute and traversal escapes, symlink escapes, safe
absolute in-root references, and verifier propagation of an out-of-root
dependency. The checkout still lacks the pinned RialTo files and generated
USDs, so the honest status remains `preparation_required`; no GPU0 catalog
smoke pass is claimed.

The unchanged legacy verifier was rerun on GPU0 with the existing dual-Panda/O6
asset and 2,000 physics steps:

```text
hard_gates_passed=true
physics_steps=2000
measured_physical_dof_count=53
nonfinite_count=0
unexpected_contact_count=0
unexpected_reset_count=0
```

The run retained the verifier's existing `OmniPBR.mdl` built-in resolver
warning classification and did not change the legacy verifier contract.

## Residual Review Fixes

- `convert_usdz_or_glb` now accepts the declared complete `asset_root`, and
  `prepare_assets` passes its destination root through the conversion call.
  Converted layers may therefore reference legitimate sibling files such as
  `../sources/dep.usd` while absolute paths, traversal, symlink escapes, and
  other references outside the pinned root remain rejected.
- Dependency classification now detects any RFC-style URI scheme before a
  colon (`resolver:foo.usd`, including schemes with `+`, `-`, or `.`), while
  preserving local Windows drive-path handling.
- Regression coverage includes the converted-USD sibling-source case and
  explicit rejection of `resolver:foo.usd`.

Residual-fix verification:

```text
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_rialto_object_assets.py \
  tests/test_m1_rialto_object_verification.py
27 passed in 1.17s

PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_rialto_object_assets.py \
  tests/test_m1_rialto_object_verification.py \
  tests/test_m1_object_catalog.py tests/test_m1_object_scene.py \
  tests/test_m1_object_target_contract.py \
  tests/test_m1_dual_panda_o6_asset_static.py
70 passed in 1.21s
```

`compileall` and a scoped `git diff --check` passed for the changed files.
The broader worktree diff check still reports a pre-existing blank line at EOF
in unrelated `.superpowers/sdd/task-6-brief.md`; it was not modified.
