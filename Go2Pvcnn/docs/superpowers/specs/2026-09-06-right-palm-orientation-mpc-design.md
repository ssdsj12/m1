# Right Palm Single-Axis Orientation MPC Design

## Context

The fixed-seed staged-PRELOAD experiment proved that translation sequencing is
not sufficient. It delayed the first right O6 body contact from step 670 to
step 994, but the first and maximum contacts remained on
`right_hand_base_link`; selected right fingertip contact and bilateral contact
both remained zero. At first base-link contact, the nearest digit reference was
still outside the box, and the final right palm orientation tracking error was
approximately `0.219 rad`.

The current `BimanualObjectMpc` copies each measured palm orientation into every
target node. Consequently, the object MPC plans box pose, palm translation, and
wrench allocation but never chooses a palm orientation that makes a digit lead
the palm housing. The existing downstream path already accepts 6D palm poses:
`ObjectMpcSolution` routes `(25, 6)` trajectories through `DualArmMpcCoordinator`
and the full-action teacher/WBC. The new behavior therefore belongs inside the
object-MPC hierarchy and does not require a public contract change.

The existing Graphify index predates the current `m1_bimanual_coordination`
implementation and returned unrelated locomotion nodes for this path. It was
recorded as a dead end; this design uses the current source and GPU0 artifacts
as authority.

## Decision

Add a right-palm orientation subplanner that selects one palm-local principal
axis from geometry at episode entry and plans a bounded scalar angle trajectory
online at 25 Hz. Candidate geometry search supplies the target angle; a convex
one-dimensional horizon QP supplies a smooth, rate-limited trajectory.

The internal orientation basis has shape `(K, 3)`. This task runs with `K=1`.
The solver and SO(3) composition helpers accept any positive `K`, so a future
three-axis implementation can use `K=3` without changing the 6D palm-pose
output or downstream contracts. This task does not activate three-axis control.

Rejected alternatives:

- A purely linearized contact-margin QP is sensitive to changes in which digit
  has maximum support and can change gradient discontinuously near a corner.
- A fixed calibrated angle is stable but does not adapt to measured fingertip
  and box geometry and is not online geometry planning.
- Collision filtering or accepting palm-base contact would hide the physical
  ordering defect.

## Component Boundary

Create a focused pure-CPU module,
`go2_pvcnn/control/m1_bimanual_coordination/palm_orientation_mpc.py`. It owns:

- quaternion, matrix, rotation-vector, and exponential-map composition needed
  for orientation candidates;
- deterministic selection of one palm-local principal axis;
- fingertip-versus-housing support-margin evaluation;
- the scalar finite-horizon QP, last-safe state, contact latch, and reset.

`object_mpc.py` remains responsible for box dynamics, palm translation, and
wrench allocation. It calls the orientation subplanner and writes the returned
rotation-vector horizon into `right_palm_pose[:, 3:]`. No orientation-planner
state is added to `ObjectMpcSolution`.

## Geometry Model

All runtime geometry is expressed in the M1 base frame. At episode entry:

1. Convert the measured right-palm rotation vector to a rotation matrix
   `R_entry`.
2. Transform the five measured palm-to-fingertip offsets into the current palm
   frame. Refresh these local offsets every update so finger closure is reflected
   in the candidate geometry.
3. Define the approach ray as the normalized vector from the right palm origin
   to the nearest point on the oriented box. Compute that point by transforming
   the palm into the box frame, clamping it to the fixed task half extents
   `(0.10, 0.09, 0.05) m`, and transforming back. Reject a degenerate or
   nonfinite ray.
4. Represent the collision housing by a conservative, calibrated set of local
   support points derived from the serialized `right_hand_base_link` collision
   geometry. The verifier pins these points' source asset SHA and checks that
   they enclose the collision AABB; runtime code does not parse USD.

For a candidate local rotation `Exp(axis * theta)`, transform both the latest
measured fingertip offsets and housing support points into the base frame.
Define:

```text
digit_support(theta)   = max_i ray · transformed_fingertip_i
housing_support(theta) = max_j ray · transformed_housing_point_j
lead_margin(theta)     = digit_support(theta) - housing_support(theta)
```

The selected leading digit index uses lowest-index tie breaking. Candidate
angles form a fixed symmetric grid including zero. Axis selection evaluates
the three palm-local canonical axes in X, Y, Z order and chooses the
axis/candidate pair with maximum finite lead margin, then minimum absolute
angle, then axis order. The chosen axis is frozen for the episode; only its
scalar angle is replanned. This gives deterministic single-axis behavior while
leaving the basis representation extensible.

The box half extents and calibrated housing support points are configuration
data for this fixed first task, not learned parameters. Housing points may be
changed only after rebuilding or independently verifying the asset and updating
the pinned asset SHA.

## Scalar Orientation MPC

The orientation state is the scalar angle relative to `R_entry`. Every object
MPC update searches the bounded candidate grid using the latest measured palm,
fingertip, and box geometry. The best candidate becomes `theta_ref`.

The QP decision is `theta[0:H]`, with `H=25` and `dt=0.04 s`. Its cost contains:

- squared tracking error to `theta_ref`;
- first-difference slew cost;
- second-difference smoothness cost;
- a small angle regularizer preferring the entry orientation when geometry is
  indifferent.

Hard constraints enforce:

- `abs(theta) <= theta_max`;
- `abs(theta[k] - theta[k-1]) <= angular_rate_max * dt`;
- the first difference is measured from the current committed angle;
- all values are finite CPU `float64` tensors.

The defaults are conservative and remain internal configuration: a symmetric
candidate interval no wider than `±0.35 rad`, at least 29 grid samples including
zero, and an angular-rate limit no greater than `0.35 rad/s`. Exact defaults are
frozen by unit tests before GPU tuning. Tuning those limits after the first
physical run requires new evidence rather than an unrecorded constant change.

For each horizon node, compose orientation on SO(3):

```text
R_target[k] = R_entry @ Exp(axis_local * theta[k])
```

Convert `R_target[k]` back to the canonical base-frame rotation-vector contract.
Do not add rotation-vector components directly. The QP does not add variables
to the existing object wrench/platform QP; it is a subordinate planner committed
atomically with the containing object solution.

## Phase and Contact Behavior

- On reset, clear the chosen axis, entry orientation, committed scalar angle,
  contact latch, and last-safe orientation trajectory.
- During APPROACH, choose the episode axis and smoothly rotate toward the
  geometry-optimal angle while the existing palm translation remains unchanged.
- During PRELOAD, re-evaluate the angle target from current geometry and retain
  all previously approved inward-first/forward-second translation behavior.
- At the first selected right fingertip contact, latch the current orientation
  trajectory endpoint and hold it through transient contact loss.
- During GRASP, LIFT, HOLD, and RELEASE, hold the last contact-safe orientation;
  this task does not introduce contact MPC.
- Left-palm orientation behavior is unchanged.

## Failure and Atomicity

Invalid shape, dtype, device, nonfinite geometry, degenerate approach ray,
invalid basis, infeasible QP, or nonfinite SO(3) conversion rejects the complete
new orientation proposal. The planner returns a cloned last-safe trajectory; on
the first-cycle failure it returns the measured right orientation repeated over
the horizon.

`BimanualObjectMpc` commits a new solution only when both its existing object
plan and the right orientation plan are feasible. An orientation failure marks
the object solution infeasible so `DualArmMpcCoordinator` performs its existing
synchronized two-arm fallback. A rejected proposal must not mutate the chosen
axis, committed angle, contact latch, or last-safe state.

The PRELOAD wrapper continues to use the full 6D error with the real right palm
Jacobian. No direct joint-space orientation correction, collision exception, or
second controller is added.

## Diagnostics

Expose read-only orientation diagnostics through the existing probe without
changing public action or solver contracts:

- selected local axis and basis dimension;
- candidate/reference/committed angle;
- candidate lead margin and leading digit index;
- predicted housing and fingertip support;
- QP feasibility, iterations, and fallback reason;
- first right palm-base and selected fingertip contact events with the planned
  angle at those events.

These fields make it possible to distinguish a geometry-model error from arm
tracking failure or contact-sensor failure.

## Verification

Implementation uses red-green TDD.

CPU unit tests cover:

- deterministic axis and candidate tie breaking;
- correct SO(3) composition and sign-equivalent quaternion behavior;
- positive lead-margin preference over a zero or negative candidate;
- exact `(25, 6)` output, CPU `float64`, continuity, angle/rate bounds;
- `K=1` active operation and construction of a `K=3` basis without activating
  three-axis runtime control;
- APPROACH selection, PRELOAD replanning, contact latch, reset, fallback, and
  atomic rejection without last-safe-state pollution;
- unchanged left orientation and unchanged T400/T500 public contracts.

Pure-QP regression covers feasibility, bound satisfaction, deterministic repeat
solves, infeasible input fallback, and no mutation of the last-safe trajectory.
Existing Arm/Hand/WBC regression suites must remain green.

GPU0 verification proceeds only after CPU and QP gates pass:

1. Three-step geometry smoke confirms the chosen angle increases predicted
   right fingertip lead and rotates in the same direction in Isaac.
2. A 200-step smoke confirms coordinates, table clearance, minimum spacing,
   frequencies, control direction, finiteness, limits, and resets.
3. The fixed-seed 1600-step PRELOAD gate requires the first/maximum right O6
   contact to belong to a right proximal or distal digit rather than
   `right_hand_base_link`, positive selected right fingertip contact, at least
   20 consecutive bilateral-contact steps, and no hard failure, unexpected
   reset, nonfinite value, or hard-limit event.
4. Only after that gate passes, run all 30 formal GPU0 trials and emit per-trial
   JSONL plus a source-Git-SHA/asset-SHA-pinned aggregate manifest.

No training entrypoint, contact MPC, perception, Residual implementation, box
geometry change, collision filtering, or T400/T500 public-contract change is in
scope.
