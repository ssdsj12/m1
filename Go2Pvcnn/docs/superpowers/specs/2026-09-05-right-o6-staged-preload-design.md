# Right O6 Staged PRELOAD Design

## Context

The serialized right O6 local-Z half-turn `(0, 0, 0, 1)` passes the asset
contract and keeps every reset-time fingertip projection toward the box
positive. A fixed-seed 1600-step GPU0 probe remains physically unacceptable:
the right hand reaches a minimum fingertip-to-box distance of `9.47 mm`, but
`right_hand_base_link` contacts the box at step 670 before any selected right
fingertip. That event coincides with approximately 86% of both the configured
right inward travel and forward travel, after which the right palm tracking
error grows to `51.9 mm` and the WBC reports 30 QP fallbacks.

This evidence isolates the remaining defect to the diagonal right-palm
PRELOAD path. It does not justify another mount rotation, collision filtering,
or redefining palm contact as fingertip contact.

## Decision

Use a two-stage right-palm PRELOAD trajectory:

1. Move only inward along base-frame `+Y` until the configured right inward
   travel is complete.
2. Hold that inward displacement and then move forward along base-frame `+X`
   until the configured forward travel is complete.

Keep the left-palm trajectory unchanged because it already produces distal
finger contact. Keep the current right O6 mount, hand closing controller,
contact definitions, box geometry, WBC interface, public tensor contracts,
and all T400/T500 contracts unchanged.

Alternatives rejected:

- Reducing forward travel while retaining a diagonal path leaves the right
  fingertips outside the box's front boundary.
- Ignoring or filtering palm-base contact weakens physical fidelity and would
  hide the observed collision instead of correcting the approach path.

## Controller Behavior

`BimanualObjectMpc` remains the owner of stateful PRELOAD palm targets. Its
existing per-side inward travel and contact latches remain unchanged.

Add a right-only forward-progress state measured in metres. At each 25 Hz
object-MPC update:

- Before finger preclosure, neither inward nor forward progress changes.
- While the right hand has no selected fingertip contact, advance right inward
  travel at `preload_palm_inward_speed_m_s`, capped by
  `right_preload_inward_limit_m`.
- Forward progress remains exactly zero until right inward travel reaches that
  cap.
- Once inward travel is complete, advance right forward progress at the same
  deterministic speed, capped by `preload_forward_limit_m`.
- On the first selected right fingertip contact, latch both progress values and
  stop advancing the palm target.
- On APPROACH or controller reset, clear the new progress state together with
  the existing PRELOAD state.

The right target is therefore:

```text
preload_entry_pose + [right_forward_progress, right_inward_travel, 0, 0, 0, 0]
```

The left target retains its current proportional inward/forward rule. This
limits the behavioral change to the side and path that failed GPU0 evidence.

## Data Flow and Contracts

The data flow remains:

```text
object MPC palm target -> dual arm MPC -> full-action teacher/WBC
                       -> arm effort command -> Isaac articulation
fingertip contact mask --------------------------------^ (PRELOAD latch)
```

No action dimension, joint ordering, solver input shape, frequency, dtype, or
device contract changes. The staged trajectory changes only the right palm
pose values emitted during PRELOAD.

## Failure Handling

Existing atomic last-safe fallback behavior remains authoritative. Invalid
inputs, infeasible object/arm/hand QPs, and infeasible WBC results must not
overwrite their respective most recent safe states. Palm-base contact remains
observable in diagnostics and is not accepted as a fingertip contact.

If the staged trajectory still causes `right_hand_base_link` to contact first,
or if it fails to create a selected right proximal/distal contact, stop further
mount or trajectory tuning and retain the diagnostic artifact for a new design
review.

## Verification

Implementation follows red-green TDD.

Pure CPU tests must prove:

- right forward progress is zero during incomplete inward travel;
- right forward progress begins only after full inward travel;
- progress is deterministic and capped;
- selected right fingertip contact freezes the staged target;
- APPROACH clears all staged state;
- left PRELOAD targets retain their prior behavior;
- fallback and last-safe behavior remain unchanged.

Then run the existing CPU unit and pure-QP layers. Only after they pass, run:

1. a 200-step GPU0 smoke for startup, finiteness, limits, resets, frequencies,
   and control direction;
2. the fixed-seed 1600-step PRELOAD probe.

The physical acceptance gate requires all of the following:

- no hard failure, unexpected reset, nonfinite value, or hard-limit event;
- first/maximum right O6 contact belongs to a right proximal or distal digit,
  not `right_hand_base_link`;
- selected right fingertip contact count is positive;
- at least 20 consecutive bilateral-contact steps;
- object, arm, hand, and WBC feasibility remain within the existing verifier
  contracts.

The formal 30-trial GPU0 layer remains blocked until this fixed-seed physical
gate passes. No training entrypoint, contact MPC, perception, or Residual work
is included.
