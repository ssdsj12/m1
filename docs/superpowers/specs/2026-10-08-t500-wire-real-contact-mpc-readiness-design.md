# T500 Wire Real-Contact MPC Readiness Design

## Goal

Prepare the dual-Panda, dual-O6 system for formal MPC training by completing one deterministic real-contact wire manipulation pipeline. The accepted episode must approach the wire, establish bilateral fingertip contact, close and hold both hands, lift the wire, and hold it stably. Only complete accepted episodes may become formal teacher-success data.

This scope covers the `wire` object only. Bottle, cup, cylinder, cube, and other rigid-object tasks remain unchanged.

## Control Architecture

The controller uses the existing staged mission:

1. `APPROACH`: both palms track pre-grasp poses at the two wire ends.
2. `PRELOAD`: O6 fingers close progressively while the runtime reads real target-filtered contacts.
3. `GRASP`: each hand has at least two valid fingertip contacts; contacted finger joints are latched and clamp force is maintained.
4. `LIFT`: both Panda arms lift synchronously while contact count, slip, and safety remain valid.
5. `HOLD`: the wire reaches the target height and remains stable for the acceptance interval.

Only PhysX/DeformX target contacts count. Distance-only proximity must never be promoted to contact. A contacted digit is latched while uncontacted digits continue closing. `PRELOAD` may transition to `GRASP` only after both hands satisfy the bilateral contact requirement. Contact loss during lift pauses upward motion and returns control to re-clamping rather than forcing continued lift.

## Coordinate and Command Data Flow

Each control tick performs the following operations:

1. Read both palms, ten fingertips, and the two wire endpoints in world coordinates.
2. Convert all poses, twists, forces, and Jacobians into the common M1 base frame.
3. The wire/object MPC produces left and right palm pose targets and a wire lift trajectory.
4. The two arm MPCs convert palm tracking errors into Panda joint commands.
5. The two hand MPCs use the real O6 fingertip Jacobians to generate per-digit closing commands.
6. Contact sensors retain only collisions between the selected wire and O6 fingertip bodies, excluding self-contact, ground contact, and unrelated robot bodies.
7. Contacted finger joints are latched with low-rate clamping; uncontacted fingers continue closing.
8. WBC and the safety projection validate joint position, velocity, effort, inter-arm clearance, and finite values before atomically applying the 43-dimensional command.

The 15-dimensional latent remains an upper-level action representation. Execution continues through the full 43-dimensional upper-body command interface.

## Diagnostics

Every sampled control tick records:

- per-fingertip distance to the selected wire;
- real contact force and contact normal;
- valid contact count for each hand;
- current mission phase and time in phase;
- left and right palm tracking error;
- wire center height and endpoint slip relative to each palm;
- MPC, hand MPC, WBC, and safety feasibility;
- atomic rejection and fallback reasons.

These fields must distinguish reach failure, collision-geometry failure, incorrect closing direction, loss of clamp, and downstream safety rejection.

## Deterministic Acceptance Criteria

An episode is successful only if all conditions hold:

- each hand has at least two selected-wire fingertip contacts;
- every counted contact force is greater than `0.2 N`;
- bilateral contact is continuous for at least `0.5 s` before lift;
- wire center height rises at least `0.05 m` from its reset height;
- the lifted wire is held for at least `1.0 s`;
- endpoint slip relative to its corresponding palm remains at or below `0.02 m` during grasp and lift;
- no joint-limit violation, non-finite value, or atomic safety rejection occurs.

The thresholds use simulation time, not wall-clock time, so headless and rendered runs have identical semantics.

## Dataset and Training Contract

Only episodes satisfying every acceptance criterion enter `teacher_success.npz`. Incomplete or rejected episodes enter `teacher_failures.npz` with a specific terminal reason. Partial feasible windows may remain eligible only for explicitly labeled warm-start training; they must not be counted as complete expert trajectories.

Formal 15D latent training begins after at least 30 successful episodes are collected. Dataset splitting is deterministic and group-disjoint by `(seed, trial)`. The trained artifact includes its model, normalization, dataset SHA-256, metadata, metrics, split manifest, and provenance. MPC Play requires the pinned metadata and refuses incompatible or modified artifacts.

## Verification Order

1. CPU unit tests validate transforms, contact filtering, per-digit latching, stage timing, acceptance thresholds, failure reasons, shape, dtype, and deterministic reset.
2. Pure solver regression validates object, arm, hand, WBC, and safety feasibility, limit handling, repeatability, fallback, and atomic rejection.
3. GPU0 smoke runs one fixed wire episode and verifies coordinates, palm approach, contact direction, bilateral contact, clamp, lift, and diagnostics.
4. GPU0 formal validation runs the accepted episode set and produces per-episode JSONL plus a SHA-pinned aggregate manifest.
5. Successful trajectories are collected, the 15D latent is retrained, and the resulting artifact is loaded by MPC Play under strict contract validation.

## Failure Handling

- Loss of one hand's contact during `LIFT` cancels upward motion and requests re-clamping.
- Persistent contact loss, excessive slip, infeasible MPC/WBC, or a safety rejection terminates the episode with an explicit reason.
- Failed commands never overwrite the most recent safe command.
- Play remains open after a safe terminal state when running interactively; headless collection exits with a machine-readable result.

## Non-Goals

This stage does not add VLM task selection, perception training, Residual learning, contact-rich generalization to other objects, or lower-body M1 locomotion. It does not use kinematic attachment as evidence of real grasp success.
