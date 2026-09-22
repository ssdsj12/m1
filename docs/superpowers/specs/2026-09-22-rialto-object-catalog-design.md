# RialTo Real-USD Object Catalog for M1/O6

## Goal

Register real RialTo object assets for O6 bimanual manipulation while preserving
the existing single `/Box` scenario and its public contracts. The first catalog
contains bottle, cup, bowl, book, cube, and cylinder-compatible entries, with
stable per-instance `object_id` values.

## Scope

In scope:

- Pin RialTo source URLs, repository revision, and SHA-256 values.
- Store source and converted assets under `Go2Pvcnn/assets/m1_objects/rialto/`.
- Convert `coffeecup.usdz` to USD for the cup entry.
- Convert `box.glb` to USD for the cube entry.
- Convert and inspect `poly.glb` for the cylinder entry; if its geometry is not
  cylinder-like, record it as a generic polygon asset rather than mislabel it.
- Add a JSON object catalog and a resolver that validates paths and hashes.
- Add deterministic multi-instance scene specifications with unique object IDs.
- Expose target object and obstacle object IDs to the O6 scene/MPC adapter.
- Keep the existing `/Box` default path unchanged.

Out of scope:

- Contact-rich screw-cap MPC, tactile sensing, or VLM integration.
- Replacing the existing M1/O6 robot assets.
- Treating expert trajectory data as object geometry.

## Pinned asset inputs

The initial source set is:

| Class | RialTo source | Expected use |
|---|---|---|
| bottle | `objects/bottle_fixed.usd` | cylindrical grasp target |
| cup | `objects/coffeecup.usdz` | converted USD cup |
| bowl | `objects/bowlnrack2.usd` | open container |
| book | `objects/book_fixed.usd` | thin box-like target |
| cube | `objects/box.glb` | converted box target |
| cylinder candidate | `objects/poly.glb` | converted and geometry-checked |

Every downloaded source and generated USD is recorded with SHA-256. No runtime
network access is permitted.

## Catalog and resolver

`config/m1_object_catalog.json` maps `object_class` to an immutable resolved
asset, physical defaults, scale, collision profile, and grasp profile. A
resolver loads only paths inside the repository asset root, verifies the
declared SHA-256, and returns a typed asset record. Missing files, hash
mismatches, unsupported extensions, and duplicate IDs fail before simulation
startup.

Instances are separate from classes. A scene list may contain entries such as
`bottle_000`, `bottle_001`, and `cup_000`; IDs must be unique and stable under
deterministic ordering.

## Scene and MPC integration

The scene adapter consumes a list of instance specifications. The selected
`target_object_id` is exposed to the task/MPC layer as the manipulated object;
all other dynamic objects are exposed as obstacles. If no catalog scene is
provided, the current `/Box` setup remains the default for backward
compatibility.

The first implementation supports explicit target selection. Distance and
graspability ranking remain a separate future policy and must not silently
override an explicit ID.

## Verification

- Unit tests validate catalog schema, SHA checks, duplicate-ID rejection,
  deterministic ordering, and legacy Box fallback.
- Asset checks open each USD and verify dependencies resolve within the pinned
  asset root.
- Isaac GPU0 smoke loads a multi-object scene, confirms object IDs, target vs
  obstacle roles, workspace placement, and contact sensor initialization.
- Existing O6 asset, MPC, and expert-prior tests must remain green.

