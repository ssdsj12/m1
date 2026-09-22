# Task 1 Report: Pin RialTo object assets

## Status

DONE_WITH_CONCERNS

## Implementation

Commit `009a1e72187b51fbf8d7361f5233fe3185ad257b` (`feat: pin RialTo object assets`) adds:

- `Go2Pvcnn/assets/m1_objects/rialto/source_manifest.json`, pinned to RialToAssets revision `bda6e4106986d39f40fe542fd74678bb616f1c41`, with explicit raw GitHub URLs, source SHA-256 values, and `sources/<name>` resolved paths for `bottle_fixed.usd`, `coffeecup.usdz`, `bowlnrack2.usd`, `book_fixed.usd`, `box.glb`, and `poly.glb`.
- `Go2Pvcnn/scripts/m1_rialto_object_assets.py`, providing `sha256_file`, manifest validation, offline-by-default `fetch_sources`, deterministic `<source-stem>.usd` conversion output, and USD inspection (Pixar USD when available, with a dependency/prim/bounds text fallback for USDA).
- `Go2Pvcnn/assets/m1_objects/rialto/README.md`, documenting explicit network opt-in (`prepare --allow-network`) and the offline runtime contract.
- `Go2Pvcnn/tests/test_m1_rialto_object_assets.py`, covering manifest schema, source SHA recording, network opt-in, unsupported extensions, deterministic names, and inspection output.

## TDD evidence

RED was observed before implementation:

```text
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_rialto_object_assets.py
6 failed in 0.15s
```

The failures were the expected missing manifest/module failures. After implementation, the focused test is green:

```text
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_rialto_object_assets.py
......                                                                   [100%]
6 passed in 0.02s
```

Additional verification:

```text
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m compileall -q Go2Pvcnn/scripts/m1_rialto_object_assets.py Go2Pvcnn/tests/test_m1_rialto_object_assets.py
```

The CLI help smoke exited 0. Running `prepare` without `--allow-network` raised the intended explicit-opt-in `RuntimeError` before any network request.

## Source SHA-256 evidence

The following digests were independently downloaded from the pinned raw URLs and matched against the committed manifest:

```text
bottle_fixed.usd  91a7fe2ba5255739ffcbc54dba3d03f03bd9c1aacf5e48cf74827776885accff
coffeecup.usdz    e8dbb6447e4b57bca837a09c78fa9ee7af816b86b0adfe962036758423c650f9
bowlnrack2.usd    e45c6c75dc153d624fdf13a86af4a2333e4087039ea22ecc6f11cd2d97a2a25d
book_fixed.usd    76c3a49dc82ed0b6efa4497c1d3da0d06962e935a27be2d5645b5bf37e40d959
box.glb           39c9f12cc10e0f9158bb8b69453d980394596f9b92a47b48436d1e98661a2c41
poly.glb          0b7b45a70bd6ea18e98b5ff2894207aee73278b6aad637106a32d3631a115c5a
```

The deterministic converter test fixture output (`#usda 1.0\n`) has SHA-256 `28f84f705dab5dc6d5baaee8ee94e59b93b71bcd708c048c71066e7a8956e2d1`.

## Concerns

No Pixar USD (`usdcat`) or GLB converter (`gltf2usd`) is installed in the current `go2` environment, so real USDZ/GLB conversion and generated-asset SHA recording could not be executed here. The implementation fails clearly when those tools are absent; the six source files are intentionally not committed and must be materialized only by the explicit preparation command. No runtime network access or Graphify output was added or changed by this commit.
