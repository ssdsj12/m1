# Task 2→3 Independent Review

## Spec Compliance

The repaired generated asset is substantially compliant with the current Task 2/3 contract according to the authoritative evidence package: it reopens with one authored articulation root at `/M1Panda/BASE_LINK`, exposes 25 DOFs, has unique `BASE_LINK`, `panda_link0`, and `panda_hand` bodies, retains the contracted fixed-joint path, resolves all real USD dependencies inside the copied asset root, reports no remote dependencies, and completes initialization/reset/write/step/update on CPU. The complete-tree relocation run is the relevant evidence that the root references are genuinely relocatable rather than merely resolvable on the build host.

The builder remains within Task 2/3 scope: Panda is imported with position drives, `RobotAssembler(single_robot=True)` is used, the Panda source root joint remains disabled, and no IK, OSC, training, task registration, or existing M1/Go2 behavior was added or changed. `Git Ref: unavailable` and the two-file generated checksum checkpoint match the plan's stated contract.

The implementation is not fully spec-enforcing, however. The verifier can return success for a single authored articulation root at the wrong path, and neither builder nor verifier fully validates the fixed joint's enabled state and two connected bodies. Those are current Task 2/3 topology requirements, not future Task 4+ work.

## Strengths

- The Sdf list-op cleanup is precise. `RemoveItemEdits` is applied only to the three known bogus `_refresh_asset` objects, then only the two known absolute project references are replaced with relative references; it does not clear either list wholesale or delete unrelated payload/reference arcs (`Go2Pvcnn/scripts/build_m1_panda_asset.py:53-60`). An exact object with asset path `/M1Panda` or `/M1Panda/Panda` is not a legitimate project file arc in this layer, so removing all edits for those exact objects is appropriate.
- Removing both `UsdPhysics.ArticulationRootAPI` and `PhysxArticulationAPI` from the attach root is the right serialization repair for Isaac Lab 2.1, which discovers roots by `HasAPI`, while retaining the enabled assembler fixed joint in the base articulation. The authoritative reopened runtime result of one root and 25 DOFs supports the intended PhysX single-articulation semantics (`Go2Pvcnn/scripts/build_m1_panda_asset.py:62-64`).
- The two legal combined-layer references are authored as `m1_floating.usda` and `panda/panda.usd`, and the reported full-tree relocation run resolved the same eight real USD layers under the new root. This is stronger evidence than checking source text alone (`Go2Pvcnn/scripts/build_m1_panda_asset.py:57-60`).
- Async initialization is guarded correctly for this Isaac Lab version: `robot.is_initialized` is checked immediately after reset, and the flag is set only after `_initialize_impl()` returns successfully. The legal Franka home pose also closes the previously observed `panda_joint4=0` false-positive path (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:141-153`).
- Failure propagation is reliable in the exercised process model: validation errors select exit 1, outer exceptions also select exit 1, JSON is flushed, and `os._exit(exit_code)` avoids the observed hanging Kit teardown. This intentionally sacrifices teardown diagnostics, but it does not by itself create a success false positive in the checked path (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:206-229`).
- The `OmniPBR.mdl` exception is an exact allowlist, not a suffix or wildcard. Every other unresolved asset remains fatal. The reports and notes correctly state the boundary: this proves Isaac-Sim-offline operation with a compatible built-in MDL library, not strict independence from Isaac Sim's material library (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:19,70-77`).
- Notes and checksums are internally consistent. The early checksum failure in the review package came from the wrong working directory; the final authoritative `AUTHORITATIVE_CHECKSUM_CONTEXT=Go2Pvcnn` section records both entries successful. The manifest contains exactly the two files required by Task 2 and does not checksum itself (`Go2Pvcnn/assets/m1_panda/generated_files.sha256:1-2`).

## Issues

### Critical

None.

### Important

1. The verifier enforces only that there is one authored articulation root, not that the sole root is the required M1 root `/M1Panda/BASE_LINK` (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:112-118`). A malformed asset with one root at another prim can pass this static gate; 25 DOFs do not prove the authored root is at the specified M1 body. This is a current Task 2/3 requirement. Require `articulation_roots == ["/M1Panda/BASE_LINK"]`, and add the same post-cleanup/reopen invariant to the builder or a behavioral asset test.

2. The fixed-mount acceptance does not fully enforce "Panda `panda_link0` fixed to M1 `BASE_LINK`." The verifier checks only prim validity and type (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:119-123`); it does not check `physics:body0`, `physics:body1`, `physics:jointEnabled`, or `physics:excludeFromArticulation`. The builder checks type and exclusion but omits `jointEnabled` and both body relationships (`Go2Pvcnn/scripts/build_m1_panda_asset.py:119-125`), despite the fix report claiming enabled state is guarded. Add exact relationship-target checks, require effective `jointEnabled is True`, and require `excludeFromArticulation is False` in both serialized reopen validation and verifier output. This is current Task 2/3 topology correctness, not Task 4+ control work.

3. The static suite mainly proves that source tokens exist, not that the cleanup, dependency classification, exit selection, or topology predicates behave correctly (`Go2Pvcnn/tests/test_m1_panda_asset_static.py:27-138`). For example, the tests still pass if a required call is unreachable, if the root-path condition remains too weak, or if mount relationship validation is absent. The recorded formal and relocated runtime runs provide good evidence for today's generated file, but regressions in the verifier/builder logic would not be caught reliably. Add behavioral tests over Sdf list-ops and dependency classification, plus a reopened generated-asset assertion for exact root and fixed-joint relationships. These tests can remain Task 2/3-focused and need not introduce Task 4 environment/config behavior.

### Minor

1. `actuators={}` is acceptable for a narrow topology probe, and the imported Panda USD does author position drives, but the verifier does not demonstrate that Panda joints remain held: it observes only one physics step and only the translational delta between `BASE_LINK` and `panda_link0` (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:147-176`). The one-step `4.75e-05 m` gate is enough to satisfy the plan's present reset/write/one-step/update smoke and to show no immediate mount snap despite the disabled-root warning. It is not evidence of long-horizon stability, mount orientation stability, finite state over time, or Panda joint holding. Treat those as explicit Task 4+ smoke criteria; do not upgrade the current result to a long-term stability claim.

2. Allowing bare `OmniPBR.mdl` solely by exact asset text cannot prove that every occurrence semantically comes from an intended material shader, because `ComputeAllDependencies` does not retain that provenance here (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:70-77`). This is reasonable for the declared Isaac-Sim-offline boundary and does not allow other missing names, but a future strict project-owned deployment must replace/vendor the full material closure rather than broaden this allowlist.

3. `os._exit` intentionally bypasses `simulation_app.close()` and all Python cleanup (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:227-229`). Flushing stdout makes the current CLI status/JSON contract reliable, but this script should remain a terminal subprocess tool; importing or composing `main()` in a longer-lived process would be unsafe. This is a code-quality boundary, not a current CLI acceptance failure.

## Assessment

**Not approved.** The current generated asset and authoritative runtime evidence look correct, and there are no Critical defects. Approval is withheld because the automated Task 2/3 acceptance does not enforce two explicit topology invariants: the exact M1 articulation-root path and the full fixed-joint body/enabled/in-articulation semantics. Closing those Important gaps and replacing the corresponding token-only checks with behavioral assertions should be sufficient for re-review; long-horizon holding/stability and strict MDL self-containment belong to later scope unless the project strengthens the present contract.
