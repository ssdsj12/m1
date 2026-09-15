# T500 OakInkV2 geometry acquisition

## Purpose and stage

Unblock the dual-source corpus qualification in GPU RTI Task 1. User confirmed retaining both FAVOR and OakInkV2; no FAVOR-only waiver. Related [T500 branch](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md) and [qualification rejection](2026-09-15-t500-run-e-expert-artifact.md).

## Official source and frozen inputs

The [OakInk2 website](https://oakink.net/v2/) links the public HF dataset `kelvin34501/OakInk-v2`. Its official toolkit identifies `object_raw` and `object_repair` as object assets. ManipTrans requires COACD meshes and corresponding object URDFs; downloading the motion archives alone does not provide them.

Repository revision: `21705616140d726607027e70d58b7837f442ffd8`. API reports ungated access. Downloaded with installed `huggingface_hub.hf_hub_download`, explicit dataset repository/revision/filename, local directory `Go2Pvcnn/data/external/oakink_v2_geometry/upstream`.

| file | bytes verified | upstream LFS SHA-256 independently matched |
| --- | ---: | --- |
| object_raw.tar | 15605760 | `40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2` |
| object_repair.tar | 139714560 | `0c470e60bc229fa04ede1c33ef4994fb05c0e57cb0e2bab3f0adab0d9c18dab0` |

No full dataset/video download. Available disk before acquisition: 452 GiB.

## Verification procedure and result

1. Query HF dataset/tree metadata, record revision and LFS sizes/hashes.
2. Download exactly the two filenames at the frozen revision.
3. Recompute file SHA-256 with hashlib and assert size and upstream LFS hash equality. Both commands exited zero and printed VERIFIED.
4. Read tar member listings without extraction: raw has 221 members; repair has 417 members.
5. Read every local OakInkV2 `sequences/*/seq_info.json`, recursively collect `ObjURDF/align_ds/` references, and compare object IDs with raw tar `object_raw/align_ds/` members. Found 98 unique URDF references / 98 object IDs; raw covers 98 / 98, missing IDs none. This verifies ID coverage, not geometry validity or exact scan/frame compatibility.

## Approved independent overlay (current execution)

User selected an independent external-SHA manifest/resolver. Design and standalone resolver plan are committed at `b0dd830`; see [overlay design](../../docs/superpowers/specs/2026-09-15-t500-oakink-geometry-overlay-design.md) and [resolver plan](../../docs/superpowers/plans/2026-09-15-t500-oakink-geometry-resolver.md). The resolver implementation is delegated with TDD and a task-scoped review gate. No accepted geometry/corpus claim follows from this interface alone.

Standalone candidate `e81da2a` independently reran `101 passed in 1.07s` (overlay + existing DexManipNet regression). Task review identified one Important missing enforcement of recipe tool/version/arguments; fixed at `bcf43d8` with RED10 failures → GREEN111 passed. Independent re-review approved with no remaining findings. The delegated fixer hit a service usage limit, so parent applied this focused TDD fix. Review cannot alone prove unchanged external archive/run_e identity; candidate diff is limited to the new module/tests, and these external files are not written by the resolver.

Source audit/load integration plan `43f093c`, implementation `a7346f2`: independent review Approved, parent reran **125 passed in1.05s**. None/default and FAVOR behavior stay unchanged; only missing safe OakInk references can be resolved, never unsafe or symlink source paths. No CLI/aggregate change is included in this source-only task. Two Minor fixture coverage suggestions (additional unsafe-parent/path cases and explicit audit/None equality) are recorded in the ledger for whole-branch review.

Fresh complete expert-prior CPU regression at `a7346f2`: from Go2Pvcnn, `PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_expert_prior_*.py -q` → **415 passed in49.43s**, exit0, no warnings. Notes-only scoped whitespace check passed. This suite does not establish Isaac/GPU runtime or real corpus qualification.

Two independent mug CoACD processes completed with **exact byte equality**, both OBJ SHA `1e912267de4c7fa5b512e5340dcb92917f6207f3ddb4e0d4e710996956a9e15f`,763374 bytes,16 components, all watertight/consistent winding/positive volume. Unchanged `_outward_collision_mesh` gate passed for both. Max raw/generated bound difference is `0.0006288487300586676` m; no rescale/registration added. This establishes repeatability for this object on this pinned environment only, not all98 objects or cross-platform determinism. Raw input equals the official archive member byte-for-byte, SHA `fb5fd1c69b9dcb38732b08b9b0937b526bc9bfb6190737d843dcf85a3aa2f07b`,75254 bytes. Stage remains external probe data, not a published production geometry bundle.

Real mug probe manifest → externally SHA-bound resolver → zero-origin/unit-scale URDF → unchanged `load_object_collision_mesh` chain passed. Probe-only manifest SHA `88155498d4d2e13ad60d5717a5ce9e2fc9131e3986757d5080f24cee67fd971f`; object geometry SHA `3bc002782ec77136c8fb5de874da8f21fb03e53dd2be4103c18050851d303ee4`. These are **not** production pins.

Attempting a real compatible mug sequence found none: all7 mug sides are passive under interaction mode (`rh/lh_main`5, `lh/rh_main`2), so no gate is relaxed. Enumerating compatible sides yields1292 sides/90 unique geometry references, compared with98 total references. Started a separate eligible `O02@0018@00001/scan` probe for `9a05b@3_bih`,142 frames, both main hands. Output staging `Go2Pvcnn/data/external/oakink_v2_geometry/eligible-probe-SyYndB`, session39279. It is still running; real source acceptance/conversion windows have not yet been claimed.

Generation environment measured: Linux 6.8.0-124 x86_64/glibc2.35; numpy1.26.0, trimesh4.5.1, CoACD1.0.5, termcolor2.5.0. Native `lib_coacd.so` SHA `46b24d42a86e17f1ab68971a57464a8204b3d2bc3337668dc446892c6e41e6b4`; fixed recipe script SHA `d0166ef25faa216e6af935b75c02a3a42c08aea83aa1ccabbe5113bc8d007f83`. These identities describe the probe environment, not proof of deterministic outputs.

Read-only exact-stem probe found raw coverage 98/98, but only 42/98 unprocessed raw meshes are watertight; repair is not an equivalent substitute (bounds differ by up to 7.65 mm). Preserve raw frame and use the fixed ManipTrans decomposition recipe. Isolated `oakink_v2_geometry/venv` contains CoACD 1.0.5 and termcolor 2.5.0; go2 dependencies remain unchanged. C10001/mug probes run one at a time with OMP/BLAS thread limits and fixed seed/recipe. Measured memory reached roughly10GiB transiently; no bulk parallel launch. Determinism and full converter compatibility have not yet been verified.

## Boundary and follow-up

The archives are downloaded and SHA-verified; only individual referenced probe meshes have been staged/generated. A complete independently pinned geometry bundle, CLI/aggregate provenance and new dual-source conversion have not yet been published. Per-object frame/surface approximation, geometry checks and output identity still need evidence for the remaining objects. Preserve run_e and original archives as immutable evidence. Do not use the old FAVOR-only aggregate pin for any new corpus. Training/distillation/Isaac/GPU RTI remain unverified and unstarted.

Next: safely stage object meshes, validate exact reference coverage and frame/scale, generate deterministically pinned COACD/URDF assets following the frozen ManipTrans source, then re-run conversion into a new output directory and requalify both sources.

## Git refs

- Baseline Ref: `a87acff676b17fa6eafd7be3cb63569d0ceeb85a`.
- Candidate Ref: standalone resolver `bcf43d8`, source-only integration `a7346f2`; current notes record these reviewed candidates.
- Key Files: this log, notes entrypoints; binary upstream files stay outside Git under external assets.
