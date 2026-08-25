# M1 + Panda Folded-Load Locomotion Curriculum Design

**Date:** 2026-08-25
**Status:** user-approved interactive design; written-spec review pending
**Scope:** foundation locomotion only; no Panda motion, external wrench, grasping, payload manipulation, or hardware claim

## 1. Motivation

The rejected fresh coordinated run completed 600 updates without an eligible checkpoint. Its diagnostic best at iteration 32 had timeout `0.67328125`, base-contact `0.0390625`, and bad-orientation `0.0`; later training degraded to predominantly base-contact termination. Three defects made that run unsuitable as the next foundation:

1. hand-wrench difficulty rose on a step clock instead of a performance gate;
2. `20 N/5 Nm` was enforced per axis, allowing vector norms near `34.6 N/8.66 Nm`;
3. catastrophe counting was disabled until an eligible best existed.

The replacement foundation first teaches M1 to carry the real folded Panda mass while moving. It prioritizes slow, auditable progress over wall-clock speed and never resumes from the rejected `accepted=false` policy.

## 2. Fixed Contracts and Non-Goals

The combined USD, single-articulation topology, 200 Hz control frequency, asset SHA-256, 103-wide observation boundary, and 23-wide actor output remain unchanged. The 23 actor coordinates preserve canonical order:

- 12 M1 leg efforts;
- 4 M1 wheel efforts;
- 7 Panda arm coordinates.

Only the first 16 coordinates are active in this curriculum. The Panda coordinates are inactive in action sampling and PPO optimization, not merely overwritten after sampling.

This curriculum does not include:

- random force or torque on `panda_hand`;
- commanded Panda arm motion;
- object assets, object perception, gripper actions, grasp rewards, or grasp evaluation;
- Student distillation or real-hardware validation.

## 3. Stage Architecture

Use separate, non-overwriting runs in this order:

```text
L0-C0 -> L1-C1 -> L1-C2 -> L1-C3 -> L1-C4 -> L2-D1 -> L2-D2 -> L2-D3
```

`L0` establishes low-speed folded-load locomotion. `L1` expands command magnitude without external disturbance or domain randomization. `L2` holds the final command range and progressively introduces initial-state and friction randomization. These names deliberately avoid collision with the existing Teacher A0/A1 stages.

Every stage owns an atomic manifest, ordinary checkpoints, eligible best metadata, final checkpoint, evaluation report, and stop reason. Only an `accepted=true` final checkpoint may initialize the next stage. A diagnostic best is never a promotion source.

The first `L0-C0` run starts from a fresh zero-action actor. Later stages load actor and critic weights from the immediately preceding accepted stage, reset optimizer state, start LR at `1e-5`, and enter with policy std no greater than `0.01`.

## 4. Panda Folded-Load Controller

Panda remains a dynamic part of the combined articulation. Its mass, inertia, gravity, joint reaction, and mount reaction affect M1 normally. It is not converted into a fixed visual payload.

The arm uses the existing implicit-actuator PD contract:

- joints 1-4: stiffness `80`, damping `4`, effort limit `87 Nm`;
- joints 5-7: stiffness `80`, damping `4`, effort limit `12 Nm`.

The fold target is the combined asset default:

```text
(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
```

The finger target remains the asset default `0.04`; fingers stay outside the PPO action contract. L0-L2 apply no Panda joint reset offsets. Diagnostics record maximum fold error, PD effort utilization, joint-limit proximity, and mount wrench.

## 5. Active-Action Mask

The policy still has 23 output rows so later arm activation does not change checkpoint tensor shapes. The active mask is:

```text
[1] * 16 + [0] * 7
```

Inactive Panda coordinates must:

- be exactly zero at the environment action boundary;
- contribute nothing to log probability or entropy;
- contribute no actor-gradient or std-gradient;
- keep their final actor rows and bias exactly zero across checkpoints.

Mask leakage, nonzero inactive output, or nonzero inactive-row gradient is a hard error. This prevents ignored arm samples from injecting PPO variance and prevents unsafe latent arm outputs from accumulating before a later unlock.

## 6. Command Distribution and Levels

The locomotion command contains forward body velocity `vx` and yaw rate `wz`; commanded lateral velocity is always zero. Every stage samples all directions rather than learning them sequentially:

- 20% stationary hold;
- 25% straight forward or reverse;
- 20% in-place left or right turn;
- 35% combined translation and turn.

Nonzero signs are balanced within each family. Commands remain constant for one episode so completed-episode attribution is exact.

| Level | `vx` range | `wz` range |
| --- | ---: | ---: |
| C0 | `[-0.05, 0.05] m/s` | `[-0.15, 0.15] rad/s` |
| C1 | `[-0.08, 0.08] m/s` | `[-0.25, 0.25] rad/s` |
| C2 | `[-0.12, 0.12] m/s` | `[-0.35, 0.35] rad/s` |
| C3 | `[-0.16, 0.16] m/s` | `[-0.48, 0.48] rad/s` |
| C4 | `[-0.20, 0.20] m/s` | `[-0.60, 0.60] rad/s` |

The policy observation keeps width 103. The existing desired-twist slot carries the sampled command. Base-position and EE-error slots remain finite for checkpoint compatibility but do not define reward or curriculum progression in L0-L2.

## 7. Domain-Randomization Levels

L0 and L1 use deterministic resets, friction `1.0`, and zero external wrench. After C4 acceptance, L2 applies one DR level per independent run:

| Level | Root pose and velocity | Leg position | Friction |
| --- | --- | ---: | ---: |
| D1 | XY `+/-0.005 m`; roll/pitch/yaw `+/-0.01 rad`; linear `+/-0.01 m/s`; angular `+/-0.02 rad/s` | `+/-0.005 rad` | `[0.95, 1.05]` |
| D2 | XY `+/-0.01 m`; roll/pitch `+/-0.015 rad`; yaw `+/-0.025 rad`; linear `+/-0.025 m/s`; angular `+/-0.05 rad/s` | `+/-0.01 rad` | `[0.90, 1.10]` |
| D3 | XY `+/-0.02 m`; roll/pitch `+/-0.03 rad`; yaw `+/-0.05 rad`; linear `+/-0.05 m/s`; angular `+/-0.10 rad/s` | `+/-0.02 rad` | `[0.80, 1.20]` |

Root Z, restitution, wheel position, Panda position, and Panda velocity offsets remain zero. Every sampled value is seeded, finite, selected-environment isolated, and checked against its configured range.

## 8. PPO Stability Contract

Each stage uses:

- 256 rollout steps per environment (`1.28 s` at 200 Hz);
- `gamma=0.9995`, `lambda=0.995`;
- 2 learning epochs and 4 mini-batches;
- initial LR `1e-5`, bounded to `[1e-6, 1e-4]`;
- desired KL `0.01`;
- immediate remaining-mini-batch abort when update KL exceeds `0.015`;
- initial active-action std `0.005`, bounded to `[0.005, 0.02]`;
- gradient-norm limit `0.5`;
- zero actor output initialization for L0-C0.

KL abort is local to one PPO update and is separately logged from stage stopping. Adaptive LR still responds within its bounds on subsequent updates. All LR, KL, early-abort, std, gradient, and active-mask diagnostics must be finite TensorBoard scalars.

## 9. Reward Contract

L0-L2 remove base-position tracking, end-effector tracking, and learned folded-arm objectives. Panda folding belongs to the PD controller and diagnostics. The locomotion reward is:

- body-X command tracking, weight `2.0`, exponential error scale `0.05 m/s`;
- yaw-rate command tracking, weight `1.0`, exponential error scale `0.15 rad/s`;
- lateral body velocity squared, weight `-0.5`;
- alive, weight `1.0`;
- base-height squared error to `0.6115 m`, weight `-12.0`;
- vertical velocity squared, weight `-1.0`;
- roll/pitch angular velocity squared, weight `-0.1`;
- flat-orientation squared error, weight `-2.0`;
- wheel/foot slide, weight `-0.1`;
- active action L2, weight `-0.02`;
- active action-rate L2, weight `-0.01`;
- selected joint-torque L2, weight `-1e-5`;
- non-timeout termination, weight `-10000.0` before the environment-step `dt` scale.

Every reward term operates only on finite tensors. Action regularizers use only the 16 active coordinates. Safety remains an independent acceptance gate; reward improvement can never override contact or orientation failure.

## 10. Promotion and Evaluation

Command levels require at least the latest 200 completed episodes. DR levels require at least the latest 400 completed episodes. A candidate is eligible only if the shared window satisfies:

- timeout rate `>=0.95`;
- base-contact rate `<=0.02`;
- bad-orientation rate `<=0.02`;
- body-X tracking RMSE `<=0.04 m/s`;
- yaw-rate tracking RMSE `<=0.12 rad/s`.

The window must contain at least 25 completed episodes for each non-stationary directional bucket: forward, reverse, left turn, and right turn. Each bucket independently obeys the same contact/orientation limits and its relevant tracking RMSE. Stationary episodes additionally require absolute body-X speed `<=0.03 m/s` and absolute yaw rate `<=0.08 rad/s`.

After training eligibility, run one full-episode evaluation with 64 environments for each seed `42`, `43`, and `44`. The fixed evaluation command table balances all command families and directions. All three evaluations must pass the same gates before manifest `accepted=true` and automatic promotion.

## 11. Always-On Stop and Rollback

Catastrophe logic is independent of whether an eligible checkpoint has ever existed:

- any non-finite observation, action, reward, optimizer value, or diagnostic: stop immediately;
- active-action mask leakage or Panda fold-control hard failure: stop immediately;
- hard-failure rate above `0.50` for 2 consecutive updates: stop;
- hard-failure rate above `0.20` for 5 consecutive updates: stop;
- after an eligible best exists, 50 updates without eligible-rank improvement: normal early stop;
- maximum stage updates: 600.

`hard_failure = base_contact OR bad_orientation` per completed episode. On failure, the stage writes a failed manifest and final diagnostic artifact but cannot promote it. The orchestration process restores the previous stage's accepted checkpoint and stops; it does not silently lower difficulty and continue inside the same run. A human or a later approved automation may relaunch the failed level after diagnosis.

## 12. Atomic Artifacts and Recovery

Every stage refuses a nonempty run directory and atomically writes:

- `run_manifest.json` with schema, parent checkpoint SHA, command/DR level, active mask, PPO contract, PID, stop reason, and acceptance state;
- ordinary numeric checkpoints;
- `model_best.pt` and `best_checkpoint.json` only for an eligible training candidate;
- three seed-specific evaluation JSON files and an aggregate evaluation decision;
- `model_final.pt`, identical by SHA to the accepted rollback source when accepted.

The rejected 2026-08-24 run remains evidence only. None of its actor, critic, optimizer, std, or diagnostic-best state initializes L0-C0.

## 13. Verification Sequence

Implementation follows RED to GREEN for every behavior change, then runs:

1. pure command sampler, action-mask, PPO KL-abort, guard, rank, and manifest tests;
2. static 103/23/200 Hz/asset and default-wrapper regressions;
3. 8-environment real GPU PD/load probe with zero external-wrench calls;
4. 8 environments x 1 PPO update wiring smoke;
5. 64-environment short train proving exact episode attribution, always-on catastrophe, and checkpoint hashes;
6. one guarded long run for only the current level;
7. seed 42/43/44 full-episode physical evaluation;
8. atomic promotion to the next level only after aggregate acceptance.

The GPU PD/load probe checks fold error, effort limits, mount response, finite state, inactive action zero, and absence of arm/body snap. No short smoke is relabeled as locomotion acceptance.

## 14. Completion Boundary

This design is complete when L2-D3 has an accepted three-seed checkpoint and all intermediate manifests form a valid SHA lineage. That outcome establishes simulated folded-load locomotion over the specified command and base-DR ranges. Panda motion, hand wrench, grasping, payload manipulation, six-axis sensor fusion, Student estimation, and real deployment each require later separately approved designs.
