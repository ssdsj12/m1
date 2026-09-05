# Right O6 Mount Orientation Correction

## Scope

Correct the right O6 hand orientation in the project-owned combined
`M1DualPandaO6` USD so that its grasping side faces the object between the two
arms. The change is limited to asset assembly, asset verification, and physical
acceptance diagnostics.

This work does not add a training entry point, contact MPC, perception, or a
Residual policy. It does not change the T400/T500 public contracts, the 43-active-
DOF order, or the 53-physical-DOF topology.

## Root cause and invariant

The supplied left and right O6 assets are chiral mirrors, but the current
combined-asset builder mounts both hand base frames to their Panda wrist frames
with the identity quaternion. Runtime evidence shows the right palm base reaches
the box before a selected right digit body, preventing bilateral digit contact.

The corrected assembly uses these frozen wrist-to-hand rotations in WXYZ order:

- left O6: `(1, 0, 0, 0)`;
- right O6: `(0, 1, 0, 0)`, a 180-degree rotation about local X.

The right hand prim's authored rest transform and the right fixed joint frame
must describe the same pose. This prevents PhysX from snapping the child body to
the joint frame on the first simulation step.

## Asset construction

`build_m1_dual_panda_o6_asset.py` will own a side-indexed hand-mount quaternion
constant. `assemble_o6` will use it both when composing the hand prim transform
from the Panda wrist transform and when authoring the fixed-joint local frame.
The left path remains byte-for-behavior equivalent to the current identity
mount.

Rebuilding the combined asset will regenerate:

- `assets/m1_dual_panda_o6/m1_dual_panda_o6.usd`;
- `assets/m1_dual_panda_o6/asset_manifest.json`.

The manifest will record both hand-mount quaternions so downstream evidence is
SHA-pinned to the actual calibration.

## Verification and failure handling

Verification proceeds from cheapest to most physical:

1. A static unit test requires side-specific mount quaternions and rejects a
   shared identity hand mount.
2. Builder reopen validation checks the fixed-joint frame and authored hand
   transform agree for each side.
3. The asset verifier checks topology, 43/53 DOF counts, dependency closure, and
   fixed-mount drift exactly as before, plus the recorded hand calibration.
4. A three-step GPU0 geometry probe compares right digit positions with the
   right palm and confirms that at least one selected digit projects toward the
   object-facing `+Y` direction without a startup reset or non-finite state.
5. The normal GPU0 smoke must pass.
6. A longer PRELOAD probe must report a right digit link as the box contact body,
   no `right_hand_base_link` maximum contact, no joint-limit violation, and
   sustained bilateral digit contact before GRASP/LIFT work continues.

Any failed stage stops the sequence. The previous USD remains available in Git;
the builder must not publish a manifest unless serialized reopen validation
passes.

## Acceptance criteria

- Right mount quaternion is exactly `(0, 1, 0, 0)` and left remains identity.
- Combined asset retains one articulation root, 43 active DOFs, and 53 physical
  DOFs.
- No startup termination, reset, non-finite state, or hard joint-limit event.
- GPU0 smoke remains passing.
- The right box-contact maximum is attributed to a right digit proximal/distal
  body rather than `right_hand_base_link`.
- Existing CPU and pure-QP validation layers remain green.
