# Task 2 Report: Validated M1 Object Catalog

## Status

DONE_WITH_CONCERNS

Feature commit: `a31cc8a` (`feat: add validated M1 object catalog`)
Review-fix commit: this report is included in
`fix: harden object catalog preparation and digests`.

## Scope Implemented

- Added `Go2Pvcnn/config/m1_object_catalog.json` with the six required classes:
  `bottle`, `cup`, `bowl`, `book`, `cube`, and `cylinder`.
- Added `object_catalog.py` with frozen `ObjectClassRecord`,
  `ObjectInstance`, and `ObjectCatalog` contracts.
- Added strict catalog loading:
  - schema/version and required-field validation;
  - only relative POSIX USD paths;
  - repository asset-root containment and symlink rejection;
  - supported `.usd`, `.usda`, and `.usdc` suffixes;
  - regular-file existence checks;
  - separate `source_sha256` provenance and `resolved_sha256` materialized-USD
    digests;
  - byte-for-byte verification against `resolved_sha256` only;
  - finite positive mass/scale and non-empty collision/grasp profiles.
- Added `AssetPreparationRequiredError` for missing assets and for converted
  records whose generated USD digest is not yet recorded. Its message names the
  missing asset and the explicit `m1_rialto_object_assets.py prepare` command.
- Converted catalog records no longer compare generated USD bytes to the source
  USDZ/GLB digest. The committed checkout leaves their `resolved_sha256` unset
  until the preparation command materializes and records those outputs.
- Added deterministic instance validation:
  - class resolution before simulation;
  - unique, path-safe instance IDs;
  - stable lexicographic `object_id` ordering;
  - immutable tuple output.
- Added explicit `load_catalog(None, asset_root)` legacy mode. It returns an
  empty catalog with `uses_legacy_box=True`, allowing Task 3 to preserve the
  existing `/Box` scene without inventing a catalog asset.
- Added the `cylinder` `geometry_note` for the `poly.glb` candidate and exposed
  catalog types through the bimanual coordination package.

## Files

- `Go2Pvcnn/config/m1_object_catalog.json`
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_catalog.py`
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- `Go2Pvcnn/tests/test_m1_object_catalog.py`

## TDD Evidence

### RED

The focused test was written before the production module and failed at
collection as expected:

```text
PYTHONPATH=$PWD/Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_object_catalog.py

==================================== ERRORS ====================================
_______________ ERROR collecting tests/test_m1_object_catalog.py _______________
...
ModuleNotFoundError: No module named
'go2_pvcnn.control.m1_bimanual_coordination.object_catalog'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

Exit status: `2`.

### GREEN

Focused catalog tests after review remediation:

```text
PYTHONPATH=$PWD/Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_object_catalog.py
........                                                                 [100%]
8 passed in 0.88s
```

Related RialTo asset and bimanual contract regression:

```text
PYTHONPATH=$PWD/Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_object_catalog.py \
  Go2Pvcnn/tests/test_m1_rialto_object_assets.py \
  Go2Pvcnn/tests/test_m1_bimanual_expert_prior_contracts.py
..........................................                               [100%]
42 passed in 1.10s
```

Compilation and whitespace checks:

```text
PYTHONPATH=$PWD/Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_catalog.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_object_catalog.py
git diff --check -- Go2Pvcnn/config/m1_object_catalog.json \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_catalog.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_object_catalog.py \
  .superpowers/sdd/task-2-report.md
```

Both commands exited `0` with no output.

## Review Fixes

- `ObjectClassRecord` now carries `source_sha256` and `resolved_sha256`; the
  legacy `sha256` property is only a read alias for the resolved digest.
- A materialized synthetic catalog test uses distinct source/resolved hashes
  and proves that only the generated USD digest is checked.
- The committed catalog's missing-source checkout now fails with the dedicated
  preparation-required error, including the asset path and preparation command.
- The converted `cup`, `cube`, and `cylinder` records intentionally keep
  `resolved_sha256: null` until their generated USD outputs are materialized.

## Concerns / Unverified Boundary

- Task 1 deliberately does not commit the downloaded RialTo sources or
  generated USDs. The strict resolver therefore rejects the committed catalog
  in this checkout until the explicit preparation command materializes the
  files and the catalog records generated USD SHA-256 values. No runtime asset
  claim is made by this task.
- `poly.glb` is retained as a cylinder candidate only with an explicit
  geometry-inspection note; its actual converted geometry was not available in
  this checkout.
- No Isaac Sim startup or GPU0 scene smoke was run; scene wiring is Task 3.
- Existing unrelated worktree and Graphify changes were not staged or modified.
