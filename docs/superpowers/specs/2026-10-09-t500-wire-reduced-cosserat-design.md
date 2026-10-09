# T500 Wire Reduced-Cosserat Online Dynamics Design

## Goal

Add a low-dimensional online wire dynamics model so the existing dual-Panda,
dual-O6 controller can perform a physically validated wire sequence:

```text
approach -> bilateral contact -> close -> clamp -> lift -> hold
```

The first implementation must remain compatible with the current 43-channel
upper-body command contract and must not start formal 15D latent training until
accepted successful episodes exist.

## Scope

This stage adds a reduced Cosserat representation, not a full rod discretization.
The model keeps the two endpoints and a bounded number of centerline samples.
It consumes endpoint pose, twist, contact force, and contact normal estimates
from Isaac/DeformX and produces a short-horizon prediction for object MPC.

Out of scope: VLM task selection, perception training, residual learning,
lower-body locomotion, generalization to other objects, and formal latent
training.

## State and Interface

The immutable wire state contains:

- endpoint positions and velocities in the common M1 base frame;
- sampled centerline positions and tangents;
- radius, axial stretch, and a bounded bending proxy;
- left/right endpoint contact validity, normal force, and tangential force;
- endpoint slip and center height;
- simulation timestamp and finite-value diagnostics.

The dynamics adapter exposes:

```text
reset(centerline, radius, material, dt)
step(left_boundary, right_boundary, contact_wrench, dt) -> state
predict(horizon, boundary_sequence, wrench_sequence) -> state_sequence
```

The adapter must be deterministic for identical inputs, preserve tensor dtype
and device at the controller boundary, and reject non-finite or malformed
states without mutating the last accepted state.

## Reduced Dynamics

The online model uses:

1. endpoint translational dynamics with effective mass and damping;
2. axial stretch from endpoint separation relative to rest length;
3. tangent transport along the sampled centerline;
4. a bounded bending proxy driven by tangent change and endpoint moments;
5. endpoint friction/slip checks using normal force and a configurable friction
   coefficient.

The model is intentionally conservative. If a predicted contact force, stretch,
bending proxy, or slip leaves its configured validity region, the prediction is
marked infeasible and the object MPC must hold or request re-clamping rather
than continue lifting.

## MPC Coupling

Each control cycle is:

```text
Isaac/DeformX snapshot
  -> reduced wire state update
  -> short-horizon wire prediction
  -> endpoint pose, velocity, wrench and lift references
  -> object MPC
  -> left/right Arm MPC
  -> left/right Hand MPC
  -> WBC and safety projection
  -> atomic 43D command
```

The object MPC adds soft costs for endpoint tracking, centerline bending,
stretch, and asymmetric force allocation. It adds hard constraints for finite
state, endpoint separation, normal force lower bound after preload, friction
cone, slip bound, and synchronized lift velocity. The Arm MPC safety boundary
also performs a next-horizon recoverability check so it cannot accept a command
that makes the next arm horizon infeasible.

## Mission and Failure Handling

`APPROACH` tracks endpoint pre-grasp poses. `PRELOAD` closes O6 fingers while
reading target-filtered contacts. `GRASP` requires two valid contacts per hand
and latches contacted digits. `LIFT` enables positive center-height motion only
while bilateral contact, friction, slip, and Arm MPC recoverability remain
valid. `HOLD` requires stable height and endpoint attachment.

Contact loss, excessive slip, invalid wrench, infeasible wire prediction,
infeasible Arm MPC/WBC, or safety rejection cancels upward motion and returns
to re-clamping or terminates with a machine-readable reason. Failed commands
never overwrite the most recent safe command.

## Acceptance and Verification

CPU tests cover state validation, deterministic stepping, stretch/bending
limits, friction/slip checks, prediction shapes, fallback, and atomic state
rejection. Pure solver tests cover object, arm, hand, WBC, and safety behavior
with the new endpoint constraints.

GPU0 smoke must demonstrate:

- both hands reach their respective wire ends;
- each hand obtains at least two real wire fingertip contacts;
- contact force exceeds `0.2 N` for `0.5 s`;
- the wire rises at least `0.05 m`;
- endpoint slip stays at or below `0.02 m`;
- the wire remains lifted for `1.0 s`;
- no Arm MPC, WBC, or safety rejection occurs.

Only accepted episodes may be written to the future teacher-success dataset.
Formal 15D latent retraining starts only after the accepted episode quota in the
existing readiness design is met.

## Compatibility

The implementation must preserve the existing public contracts and existing
rigid-object catalog behavior. The reduced model is an adapter behind the wire
object-MPC boundary; replacing it later with a full DeformX Cosserat backend
must not require changing the Panda/O6 command or diagnostics contracts.
