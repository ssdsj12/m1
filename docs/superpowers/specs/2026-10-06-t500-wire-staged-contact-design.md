# T500 Wire Staged Contact and DeformX Binding Design

## 1. Goal

Make the two Panda–O6 arms approach the wire that actually exists in the
Isaac scene, establish target-filtered bilateral fingertip contact, and only
then bind the measured grasp span to the DeformX Cosserat backend. The first
deliverable ends when both hands can reproducibly contact `wire_000` and the
internal Cosserat model begins from a continuous state.

This is the selected staged approach. It fixes the current mismatch in which
the runtime initializes the rod between the palms before contact while the
Isaac contact sensors still filter only `/Box`.

## 2. Scope

### In scope

- Register an explicit, reachable default pose for `wire_000`.
- Derive a pre-contact centerline from the selected scene object's measured
  pose and catalog geometry.
- Plan distinct left and right palm/fingertip targets along that centerline.
- Read contact forces filtered against the selected object, including
  `wire_000`, instead of treating `/Box` as the only valid target.
- Require stable bilateral contact before initializing and stepping the dual
  DeformX backend.
- Preserve the existing atomic WBC acceptance boundary and safe fallback.
- Expose deterministic diagnostics for scene tracking, contact latching, and
  DeformX binding.

### Out of scope

- Rendering a deforming Cosserat mesh in Isaac Sim.
- Replacing the rigid wire USD proxy with a PhysX-deformable representation.
- Arbitrary fingertip-to-interior-node coupling, sliding contact, knot tying,
  cable routing, lifting, or task-policy training.
- VLM commands, perception-network training, Residual control, or changes to
  the 43-dimensional action/public T400–T500 contracts.

The rigid USD remains the collision and perception object in this stage.
After binding, DeformX is an internal predictive model used by MPC; it does
not teleport or visually replace the USD. This distinction prevents two
physics systems from simultaneously owning the same scene body.

## 3. Coordinate and Geometry Contract

All planning geometry is expressed in the robot base frame and uses
`torch.float64` on CPU at the Isaac adapter boundary.

For a wire catalog entry with length `L`, radius `r`, local longitudinal axis
`a_local`, measured object position `p_b`, and measured object rotation
`R_b_object`, define:

```text
t_b = normalize(R_b_object @ a_local)
c_b = p_b
s_left  = c_b - 0.30 * L * t_b
s_right = c_b + 0.30 * L * t_b
```

The two grasp stations are therefore separated by `0.60 * L`. They are not
the visual mesh extrema: each station leaves `0.20 * L` of material beyond
the grasp on its side. For the current 0.08 m proxy, the stations are at
`±0.024 m` from the center and the predicted held span is 0.048 m.

The catalog must provide finite positive dimensions and an explicit local
longitudinal axis. The first version uses local `+X` for `wire_000`. It must
not infer the axis from object ID or palm locations. The play launcher
registers `wire` at the reachable default position `(0.65, 0.0, 1.20)`; its
orientation is supplied by the scene pose.

The measured scene centerline is sampled between the two grasp stations. Its
point order is always left station to right station, independent of camera
view or current hand position.

## 4. Runtime State Machine

Wire coupling has an explicit episode-local state machine:

1. `SCENE_TRACKED`
   - Read `snapshot.box.pose_b` for the selected `wire_000` scene object.
   - Build the measured grasp-span centerline.
   - DeformX is uninitialized and returns no impulses.
2. `APPROACH_ENDPOINTS`
   - The existing object MPC remains responsible for the object trajectory
     and WBC feasibility.
   - The wire trajectory planner overrides only the two palm references and
     O6 thumb/index soft targets.
   - The palms approach separate grasp stations with bounded per-tick motion.
3. `CONTACT_LATCHING`
   - A side latches only when its target-filtered thumb/index contact meets
     the configured force threshold for 5 consecutive control ticks.
   - A latched side holds its accepted target while the other side continues
     approaching. Raw contact without target-filtered contact never latches.
4. `COSSERAT_BOUND`
   - Enter only after both sides are latched.
   - Initialize a DeformX rod representing the held span from the measured
     scene centerline, not from the historical palm-to-palm segment.
   - Reject the transition when either palm-to-station error exceeds 0.005 m
     or any input is non-finite.
   - Once accepted, the existing dual palm frames drive the two Cosserat
     endpoints and DeformX predictions feed the existing wire trajectory
     planner.
5. `SAFE_HOLD`
   - Enter on post-bind contact dropout, DeformX failure, non-finite geometry,
     infeasible WBC, or rejected binding.
   - Keep the last atomically accepted command and publish the failure reason.
   - Do not silently reinitialize the rod during the same episode.

`reset()` clears all contact counters, latches, bind state, cached centerline,
DeformX integration state, and pending/accepted wire goals. A new episode may
bind again from its new measured scene pose.

## 5. Components and Responsibilities

### Scene wire geometry provider

A focused provider converts the target object's measured pose plus catalog
metadata into a deterministic centerline and two grasp stations. It has no
knowledge of arms, contacts, MPC phases, or DeformX imports.

### Target contact sensor configuration

The environment configuration creates the existing 18 per-link filtered
contact sensors using the selected target scene prim path. For the default
box this remains `{ENV_REGEX_NS}/Box`; for this task it is
`{ENV_REGEX_NS}/wire_000`. Sensor names become target-neutral internally so
the wrapper cannot accidentally read stale `*_box_contact` entries.

The target scene ID is resolved before the environment is constructed. The
configuration rejects an ID that is missing from the active object catalog;
it never falls back to `/Box` after the user selected another object.

### Contact summary and latch

`summarize_contacts` consumes raw fingertip force and selected-target filtered
force. Its public hand-state shape is unchanged. The consistency reason is
renamed conceptually from `raw_contact_without_box_contact` to
`raw_contact_without_target_contact`, while legacy serialized diagnostics may
be accepted during migration.

A separate bilateral latch owns the 5-tick debounce and one-sided hold logic.
It depends only on the two hand contact summaries and has deterministic reset
semantics.

### Staged DeformX backend

The backend receives an explicit `bind(initial_centerline, left_frame,
right_frame)` call. `step()` before a successful bind is invalid and cannot
implicitly construct a rod. Binding initializes the rod at the scene-derived
left/right grasp stations, checks continuity, and records one immutable bind
event for diagnostics.

The initial implementation models only the span held between the two grasp
stations. The two free outer tails remain part of the rigid USD proxy but are
not included in the Cosserat prediction. Full-rod interior-node attachment is
a later extension and does not alter this interface.

### Wire trajectory planner

Before binding, the planner consumes the scene centerline. After binding, it
consumes the DeformX centerline. In both cases it produces the same
`DloWireTrajectoryPlan`, keeping object MPC, arm MPC, hand MPC, WBC, and the
atomic accepted-goal publication boundary unchanged.

## 6. Data Flow

```text
selected object ID + catalog metadata
                  |
Isaac target pose + target-filtered fingertip forces
                  |
        scene wire geometry provider
                  |
    pre-contact centerline + grasp stations
                  |
 object MPC -> bounded dual-palm/O6 targets -> Arm/Hand MPC -> WBC
                  |
       bilateral 5-tick contact latch
                  |
      continuity-checked DeformX bind
                  |
 Cosserat centerline/impulses -> existing wire trajectory path
                  |
          atomic accepted command/goal
```

The object pose used here is the live scene-backed `snapshot.box.pose_b`, not
a catalog nominal pose. Catalog data supplies shape semantics only.

## 7. Safety and Failure Handling

- Missing pose, axis, dimensions, or non-finite values: remain in safe hold;
  never initialize DeformX.
- One-sided contact: freeze the latched side at its last accepted target and
  allow only the unlatched side to advance.
- Contact chatter: require 5 consecutive ticks; any miss before binding
  resets only that side's counter.
- Excess bind mismatch: reject binding, report both errors, and retain the
  last safe command.
- Contact loss after binding: enter `SAFE_HOLD`; no automatic rebind in the
  episode.
- DeformX exception or non-finite output: discard the candidate wire goal and
  preserve the last atomically accepted WBC command.
- WBC infeasibility: do not commit contact latch progression, bind state, or
  `latest_wire_goal` from that candidate tick.
- CPU-heavy verification and preprocessing run with
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.

## 8. Diagnostics

Every diagnostic record includes:

- selected object and resolved target prim path;
- coupling state and transition reason;
- measured centerline midpoint, tangent, radius, and grasp stations;
- left/right station error;
- raw and target-filtered contact masks and debounce counters;
- bind accepted/rejected flag and initial continuity error;
- DeformX elapsed time and impulse norms after binding;
- WBC feasibility and whether the wire goal was atomically committed.

The play window remains open on safe termination under the existing play
contract. Diagnostics must not change solver behavior.

## 9. Verification

### CPU unit tests

- Transform local `+X` by known object rotations and verify centerline point
  order, midpoint, length, dtype, and finite checks.
- Verify the 0.08 m wire produces stations at `center ± 0.024 m`.
- Verify left and right palm targets are distinct and bounded per tick.
- Verify selected target IDs generate target-specific sensor prim paths and
  that an unknown ID is rejected.
- Verify raw-only contact does not latch, 4 filtered ticks do not latch, and
  the fifth consecutive tick does.
- Verify unilateral contact holds only the latched side.
- Verify DeformX cannot step before bind, initializes from the measured
  centerline, rejects mismatch above 0.005 m, and resets deterministically.
- Verify rejected WBC candidates do not mutate latch state, bind state,
  backend state, last safe command, or `latest_wire_goal`.
- Verify post-bind dropout enters `SAFE_HOLD` without reinitialization.

### Pure controller regression

- Existing object, arm, hand, and WBC feasible/limit/determinism tests remain
  green for `box_000` and non-wire catalog objects.
- Repeated fixed inputs produce byte-equivalent state transitions and
  centerline outputs.
- Invalid inputs do not contaminate the last accepted command.

### Isaac GPU0 smoke

1. A short fixed-seed smoke confirms the measured centerline midpoint matches
   the live `wire_000` pose, the target filter resolves to `wire_000`, and no
   DeformX engine is created before contact.
2. A longer fixed-seed smoke must demonstrate, in order,
   `SCENE_TRACKED -> APPROACH_ENDPOINTS -> CONTACT_LATCHING -> COSSERAT_BOUND`.
3. The smoke fails on NaN/Inf, wrong target prim, same-station targeting,
   implicit pre-contact binding, missing bilateral contact, continuity error
   above 0.005 m, or traceback.

Formal lift or deformable visual motion is not a success criterion for this
stage and must not be claimed from these tests.

## 10. Acceptance Criteria

The stage is complete only when all of the following are evidenced:

1. The scene-derived centerline midpoint is within 0.002 m of the live wire
   object position.
2. The two hands receive separate station targets at `±0.30 * L`.
3. Contact with `wire_000`, not `/Box`, produces valid left and right filtered
   contact masks.
4. Each side is stable for 5 consecutive ticks before latching.
5. DeformX remains uninitialized before bilateral latch and binds exactly once
   afterward with at most 0.005 m endpoint mismatch.
6. The bind transition does not cause a target discontinuity above 0.005 m.
7. Atomic rejection and reset tests pass without state leakage.
8. Existing CPU regression tests pass with the thread cap, and the GPU0 smoke
   records the full staged transition without NaN, traceback, or unsafe
   termination.

## 11. Follow-on Work

After this stage is verified, the next independent design can synchronize a
rendered deformable wire with the full Cosserat rod, attach palms to arbitrary
interior rod nodes, add controlled sliding/regrasp, and then plan manipulation
tasks such as lifting, routing, and insertion. Those additions require a
separate collision/ownership contract and are deliberately excluded here.
