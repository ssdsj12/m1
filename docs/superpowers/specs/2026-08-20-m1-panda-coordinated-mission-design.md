# M1 + Panda Coordinated Mission Design

## Goal

Train and execute one coordinated M1 + Panda articulation for a two-stage mission:

1. M1 carries a folded Panda to a target planar pose `(x, y, yaw)`.
2. Panda unfolds as needed and tracks a commanded end-effector pose or trajectory.

During stage 2, M1 may move slowly and within bounded limits to release Panda
null-space capacity. End-effector tracking and safety have priority over base
motion.

Student S1 remains a later deployment consumer. The first implementation is a
deterministic privileged WBC Teacher so each coordination decision is observable
and testable.

## Existing boundaries

- Use the combined M1 + Panda articulation from `M1_PANDA_CFG`.
- Preserve the 16-channel M1 base action boundary and the 7-channel Panda arm
  boundary used by the existing Teacher/WBC code; the combined execution action
  is 23 channels.
- Reuse the existing rolling mission, Panda safety supervisor, kinematics and
  Panda-first motion distribution instead of creating a parallel controller.
- Do not treat the old `Isaac-M1-Walk-v0` as this task: it is a M1-only asset.

## Mission state machine

The mission has explicit reset-safe states:

`FOLD_AND_NAVIGATE -> ARRIVE_HOLD -> UNFOLD_AND_TRACK -> COORDINATED_TRACK`

- `FOLD_AND_NAVIGATE`: interpolate Panda toward a validated folded target;
  command M1 toward the planar target pose. Panda remains the protected payload.
- `ARRIVE_HOLD`: require bounded position/yaw error and a short settled window;
  hold M1 and verify the folded arm is finite and inside limits.
- `UNFOLD_AND_TRACK`: interpolate from folded target to the initial arm target;
  M1 remains stationary unless safety requires a bounded correction.
- `COORDINATED_TRACK`: track the end-effector pose/trajectory. M1 planar motion
  is enabled only when the arm coordination metrics request it.

Every state transition is deterministic, monotonic, resettable per environment,
and recorded in diagnostics. Invalid or non-finite targets enter the existing
safety fallback rather than advancing the mission.

## Coordination law

At each control update, compute Panda task-space error, arm joint-limit margin,
manipulability/singularity indicators, and the required planar base twist.

1. Solve Panda motion first for the requested end-effector twist.
2. Measure residual task error and arm saturation.
3. If the arm margin or manipulability falls below configured hysteresis
   thresholds, activate the M1 planar null-space direction that improves the
   arm's feasible task-space solution.
4. Bound base velocity, acceleration, displacement from the arrived pose, and
   yaw correction. Apply the existing safety supervisor and action slew limits.
5. Re-solve Panda with the selected base contribution and emit one safe 23-D
   joint target/action.

The base contribution is accepted only when it improves a lexicographic safety
   objective: no joint-limit violation, no singularity escalation, bounded task
   error, then minimum base motion. A zero-space activation metric and the
   before/after arm margin are logged for every update.

## Training and rollout surfaces

The first runtime target is a combined Teacher environment with privileged
state, deterministic folded/track phases, and the existing WBC update rates.
Training data must include both ordinary tracking and deliberate near-limit
episodes so the null-space trigger is exercised. The old M1-only locomotion
checkpoint is not a compatible initialization for this combined mission unless
it is explicitly adapted through the combined Teacher boundary.

Student S1 is not changed in this phase. Once the Teacher passes the runtime
gates, its 23-D safe action and coordination diagnostics become the labels and
observations for a later Student update.

## Safety and failure behavior

- Reject non-finite target pose, trajectory, Jacobian, or action tensors.
- Keep Panda joint limits, M1 contact/balance limits, and action slew limits hard.
- On invalid tracking or worsening singularity, hold/retract Panda and reduce
  M1 motion before terminating.
- On terminal safety state, freeze the last finite safe target and latch until
  explicit reset.
- Never infer successful coordination from a single frame; use a settled window.

## Acceptance gates

Local contracts must cover state transitions, reset isolation, fold/unfold target
interpolation, null-space trigger hysteresis, lexicographic base selection,
limits, non-finite fallback, and 23-D action reconstruction.

The first Isaac smoke must demonstrate, for multiple environments:

- combined articulation creation with M1 + Panda;
- navigation reaches the target pose and enters `ARRIVE_HOLD`;
- Panda fold height is lower than its unfolded reference;
- end-effector tracking remains finite through `UNFOLD_AND_TRACK`;
- at least one deliberate near-limit case activates bounded M1 assistance;
- no unexpected reset, base contact, joint-limit violation, or action snap;
- diagnostics report phase, end-effector error, arm margin, base assistance,
  and safety state.

Behavioral thresholds will be fixed in the implementation plan after measuring
the existing C0/C1a runtime scales; they must not be silently borrowed from the
M1-only Walk task.

## Out of scope

- Grasping, force-controlled manipulation, payload identification, and real-robot
  maximum-load testing.
- Replacing the existing WBC with end-to-end PPO before the deterministic Teacher
  gates pass.
- Updating Student S1 or its evaluation contract in this design phase.
