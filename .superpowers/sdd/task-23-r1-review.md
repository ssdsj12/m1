# Task 2→3 Fix Round 1 Independent Review

## Spec Compliance

Fix Round 1 closes all three Important findings from the initial review.

- Exact root path is now enforced, not inferred: both builder and verifier require the authored root list to equal exactly `["/M1Panda/BASE_LINK"]` (`Go2Pvcnn/scripts/build_m1_panda_asset.py:60-67`, `Go2Pvcnn/scripts/verify_m1_panda_asset.py:141-144`).
- The complete mount contract is enforced: fixed type, exact `body0` and `body1`, effective `jointEnabled is True`, and `excludeFromArticulation is False` (`Go2Pvcnn/scripts/build_m1_panda_asset.py:69-81`, `Go2Pvcnn/scripts/verify_m1_panda_asset.py:145-168`).
- Tests now execute falsifiable predicates and real Sdf/PXR behavior rather than relying only on source tokens (`Go2Pvcnn/tests/test_m1_panda_asset_static.py:167-191`, `Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py:59-88`).

The authoritative Round 1 evidence also retains the previously accepted Task 2/3 results: 25 DOFs, required bodies unique, no remote/outside/unallowlisted unresolved dependencies, formal and relocated CPU verifier exit 0, reset/write/step/update complete, deterministic generated hashes, and no Task 4+ implementation.

## Strengths

- Builder validation runs at both relevant semantic boundaries. `_validate_stage_contract` first checks the fully composed live stage after removing the Panda root APIs but before relative references temporarily become unresolved (`Go2Pvcnn/scripts/build_m1_panda_asset.py:98-107`). After the root layer is exported, `_validate_serialized_asset` opens `combined_usd` by its file path and validates the independently reopened file-backed stage (`Go2Pvcnn/scripts/build_m1_panda_asset.py:110-113,176-178`). Because the build stage was created with an anonymous root layer, this reopen is not merely rechecking the same anonymous layer.
- Exact root enforcement is shared in meaning across builder and verifier. Wrong sole root, no root, and multiple roots all fail the executable verifier predicate test (`Go2Pvcnn/tests/test_m1_panda_asset_static.py:176-181`).
- Mount validation uses composed USD relationship targets and effective schema values, which is the correct layer-stack semantics for runtime acceptance. Each individual wrong input—body0, body1, disabled joint, or excluded joint—is proven to produce an error (`Go2Pvcnn/tests/test_m1_panda_asset_static.py:184-191`).
- Verifier JSON and validation share the same actual values. `mount_body0_targets`, `mount_body1_targets`, `mount_joint_enabled`, and `mount_joint_exclude_from_articulation` are read once from the opened stage, passed directly to `_mount_joint_contract_errors`, and returned unchanged in JSON (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:146-168,243-246`). `articulation_roots` is likewise computed once, validated, and returned (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:141-144,241`). This avoids a validation/reporting split-brain path.
- The cleanup behavior test is genuinely falsifiable. It constructs real Sdf reference/payload list-ops with all three targeted bad objects plus unrelated arcs, calls the production cleanup helper extracted from the builder, and requires all unrelated arcs to survive (`Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py:26-74`). Clearing whole lists or removing the wrong objects would fail.
- The PXR runner independently opens the formal exported asset and checks exact root plus every mount field (`Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py:76-88`). Combined with the recorded production build's post-Export gate and relocated runtime verifier, this provides behavior evidence at source-helper, serialized-USD, and runtime levels.
- Exact `OmniPBR.mdl` classification is now behavior-tested with counterexamples `Other.mdl`, `textures/OmniPBR.mdl`, and `/M1Panda`; broad suffix/path matching would fail (`Go2Pvcnn/tests/test_m1_panda_asset_static.py:167-173`).
- The Round 1 report correctly narrows its claims: one-step Task 3 CPU smoke only, not long-horizon holding/orientation stability or strict independence from Isaac Sim's MDL library. Checksums and `Git Ref: unavailable` remain aligned with the plan.

## Issues

### Critical

None.

### Important

None. All three Important findings from the initial review are closed.

### Minor

No new Round 1 Minor defects were introduced. The previously accepted boundaries remain unchanged:

- `actuators={}` plus one physics step is a narrow Task 3 topology/no-snap smoke, not proof of long-horizon Panda holding or orientation stability.
- Exact `OmniPBR.mdl` allowance establishes Isaac-Sim-offline compatibility, not strict project-owned material-library closure.
- `os._exit` is appropriate only while the verifier remains a terminal subprocess tool.

The PXR runner directly behavior-tests cleanup and the formal serialized asset, while the builder's malformed-stage rejection is established by readable predicates and the production gate rather than a separate negative in-memory stage fixture. Adding such a fixture would improve future regression isolation, but it is not a current Task 2/3 correctness gap and does not reopen the prior token-only Important finding.

## Assessment

**Approved.** Round 1 implements and exercises the exact M1 root invariant, the complete fixed-mount relationship/enabled/in-articulation contract, and meaningful falsifiable behavior coverage. The builder's serialized reopen gate operates on the exported file-backed composition, and verifier validation and JSON reporting use the same observed values. No Critical or Important issues remain in current Task 2/3 scope.
