# T500 OakInkV2 Geometry Overlay Design

## Scope and status

Supplemental prerequisite for GPU RTI Task 1. User approved official geometry acquisition and retention of the dual-source qualification gate. This document proposes the controlled converter integration; it is not a production qualification claim.

## Frozen inputs

- Official dataset: `kelvin34501/OakInk-v2`, revision `21705616140d726607027e70d58b7837f442ffd8`.
- `object_raw.tar`: 15605760 bytes, SHA `40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2`.
- `object_repair.tar`: 139714560 bytes, SHA `0c470e60bc229fa04ede1c33ef4994fb05c0e57cb0e2bab3f0adab0d9c18dab0`, diagnostic only; not a substitute for raw coordinates.
- ManipTrans recipe: existing fixed source commit `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`, CoACD `1.0.5`, seed 1, max hulls 32, threshold 0.07, preprocessing auto/resolution 50, surface resolution 2000, MCTS nodes 20/iterations 2000/depth 5, PCA off, merge on, decimation/extrusion off.
- All 98 original URDF reference stems have corresponding raw mesh stems. This is filename coverage, not validated geometry readiness.

## Architecture

1. Independently SHA-validate upstream archives. Extract only regular, allowlisted referenced meshes into fresh staging; reject traversal, links, duplicate names and undeclared outputs.
2. Generate convex meshes without an additional rescale, registration, centering or rotation. Preserve source coordinates and pin tool/native-library/dependency/platform/thread identities and every effective recipe argument.
3. Validate finite triangular geometry, positive volume, consistent outward winding, enclosed components, extent/frame preservation and actual converter compatibility. Verify two independent decompositions before claiming deterministic generation. Reject failed objects; never replace them by a box or single convex hull solely to pass the gate.
4. Publish immutable geometry output and canonical manifest atomically. Manifest binds upstream member hashes, original reference paths, generated URDF and mesh hashes, zero origins/unit scale, recipe and measured checks. External manifest SHA is supplied by the caller, not trusted from a self-declaration.
5. Add optional converter geometry root/manifest/external-SHA configuration, required as a complete triple. Validate before sequence audit. A single verified resolver is used by both audit and rollout load. It handles only declared OakInkV2 references; default/FAVOR resolution remains unchanged and unknown/unsafe/drifted references fail closed.
6. Add independent geometry provenance to new aggregate evidence without repurposing original archive/source hashes. Keep accepted-row `object_geometry_sha256`. Convert into a fresh output, preserve run_e, requalify both sources and approve the new corpus pin before training.

## Safety and public boundaries

No symlink is introduced into the converter's trusted root. No pinned ManipTrans checkout, input archive, run_e or public T400/T500 contract is mutated. Default conversion behavior remains identical. Missing or mismatched geometry identity fails before conversion output publication. No geometry approximation changes contact/safety thresholds.

## Verification

CPU fixture tests cover metadata/hash drift, absolute/traversal/symlink/duplicate paths, incomplete CLI triple, allowlist mismatch, missing mesh, wrong origin/scale, source isolation, identical resolver identity for audit/load and exact prior-off behavior. Real mesh probes verify frame/scale/volume and reproducibility with fixed dependencies. Two complete conversions require consistent aggregate identities and positive accepted windows from FAVOR and OakInkV2. Production training, distilled prior, GPU RTI, Isaac smoke and formal 30-trial gates remain separate downstream deliverables.

## Resource policy

Geometry generation remains CPU-only in a dedicated environment. Limit concurrent CoACD processes based on measured memory, not CPU count. The first mug probe uses one thread and already consumes several GiB, so do not launch 98 jobs in parallel. Record timing and peak memory; preserve failure diagnostics for resume.
