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

## Review Fixes

Follow-up commit: `fix: harden RialTo asset preparation`.

The review findings are addressed in the follow-up fix:

- `inspect_usd` detects the `PXR-USDC` binary signature before any text decoding when Pixar USD/`pxr` is unavailable, and raises an explicit error that binary data will not be decoded as text.
- Converted USD output is inspected before it is accepted. Missing local `@dependency@` files fail preparation with the dependency names in the error. Direct USD inputs are validated by the same path.
- GLB conversion no longer assumes an unverified `gltf2usd` positional CLI contract. It fails clearly with `no supported GLB converter contract is configured` until a verified converter is added.
- Successful preparation writes `prepared_manifest.json` and CLI output metadata containing source SHA-256, USD SHA-256, generated USD SHA-256 for converted inputs, and inspection results.
- Regression coverage now includes a binary USDC fixture, an actual missing-file USD dependency, matching/stale local source hashes, unresolved converter output, explicit GLB refusal, and generated USD metadata.

Fresh verification:

```text
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_rialto_object_assets.py
............                                                             [100%]
12 passed in 0.03s

/home/xk/miniconda3/envs/go2/bin/python -m compileall -q Go2Pvcnn/scripts/m1_rialto_object_assets.py Go2Pvcnn/tests/test_m1_rialto_object_assets.py
# exit 0, no output

PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python Go2Pvcnn/scripts/m1_rialto_object_assets.py --help
usage: m1_rialto_object_assets.py [-h] {prepare,inspect} ...
...
# exit 0

git diff --check -- Go2Pvcnn/assets/m1_objects/rialto/README.md Go2Pvcnn/scripts/m1_rialto_object_assets.py Go2Pvcnn/tests/test_m1_rialto_object_assets.py
# exit 0, no output
```

`command -v usdcat` and `command -v gltf2usd` produced no paths in this environment, so no real converter-backed generated asset was promoted. The generated-SHA path is covered with a deterministic local converter fixture.

## P1 GLB Conversion Follow-Up

Follow-up commit: `fix: convert RialTo GLB assets with trimesh`.

The remaining GLB conversion gap is resolved without adding an external
converter dependency:

- Valid `.glb` files are loaded with the installed `trimesh` library using
  `process=False`, all scene transforms are baked, mesh instances are sorted
  deterministically, and the geometry is concatenated into one mesh.
- The combined mesh is emitted as self-contained ASCII USDA containing
  `points`, `faceVertexCounts`, `faceVertexIndices`, `extent`, and
  `subdivisionScheme = "none"`.
- Missing `trimesh`, invalid GLB headers, malformed GLB payloads, meshless GLBs,
  non-finite geometry, and incomplete mesh data fail with explicit
  `RuntimeError` messages before conversion output is accepted.
- USDZ conversion still requires `usdcat`, and converted output continues
  through the existing unresolved-dependency validator.

The focused regression now generates a small two-mesh GLB fixture, verifies
byte-identical output across two conversions, checks the generated USDA SHA
`d560385cce0956598b656f2139e8ea4cd15ed98b184133dbb97163cbccc33d55`, inspects
combined bounds `[[-1.0, -2.0, -3.0], [3.5, 2.0, 3.0]]`, and rejects a malformed
GLB without publishing `malformed.usd`.

## P2 Atomic Conversion Follow-Up

Follow-up commit: `fix: make RialTo conversion publication atomic`.

`convert_usdz_or_glb` now writes converter output to a unique temporary sibling
with the final USD suffix, validates that temporary output and its dependencies,
then atomically replaces the destination with `Path.replace`. Converter or
validation failures clean up the temporary file and preserve any previously
validated destination unchanged. The regression test exercises a malformed
reconversion over an existing output and verifies both byte preservation and
absence of temporary siblings.
