# M1 + Panda Folded-Load PD Retune Design

**Date:** 2026-08-25  
**Status:** user-approved design choice; implementation pending written-spec review  
**Scope:** folded-load locomotion task only

## 1. Problem Evidence

The approved folded-load controller originally reused Panda arm gains `Kp=80`, `Kd=4`. Explicitly writing the fold position and zero-velocity targets fixed the missing-target failure, reducing the 256-step fold error from `3.0231 rad` to `0.2486 rad`. The task still failed its independent safety guard because joint4 reached a joint-limit proximity of `-0.1369 rad`.

A separate 256-step, eight-environment, all-zero-action GPU0 probe reproduced the same condition without locomotion policy motion:

- inactive Panda action maximum: `0.0`;
- fold error maximum: `0.2620 rad`;
- joint-limit proximity minimum: `-0.1503 rad`;
- effort utilization maximum: `1.0`;
- state tensors: finite.

Joint4 has target `-2.810 rad` and lower limit `-3.0718 rad`, leaving `0.2618 rad` geometric margin. The observed gravity deflection approximately exhausts that margin. This is a folded-load PD holding problem, not PPO action leakage, command sampling, or guard misclassification.

## 2. Selected Controller Contract

Use fixed gains throughout stationary and moving folded-load stages:

| Joints | Stiffness `Kp` | Damping `Kd` | Effort limit |
| --- | ---: | ---: | ---: |
| `panda_joint1–4` | `120` | `8` | unchanged `87 Nm` |
| `panda_joint5–7` | unchanged `80` | unchanged `4` | unchanged `12 Nm` |

The controller continues to write the approved position target before every physics step:

```text
(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
```

and writes zero arm velocity targets. Panda remains a dynamic part of the combined articulation; its mass, inertia, gravity, joint reaction, and mount reaction continue to affect M1.

There is no stationary/moving gain switch. Binary gain switching would introduce torque discontinuities and non-stationary training dynamics, while reducing gains during movement would weaken the controller under the larger inertial disturbance.

## 3. Isolation Boundary

Do not modify the global `M1_PANDA_CFG`, combined USD, or asset SHA. Create a folded-load-only copy of the articulation configuration and replace only its `panda_shoulder` actuator gains. The following paths retain their existing controller behavior:

- Teacher A0/A1;
- prioritized WBC C0/C1a;
- coordinated and Student tasks;
- existing smoke/play tasks.

The folded-load boundary remains 103 observations, 23 actor outputs, 200 Hz, and active mask `[1]*16+[0]*7`. Panda policy coordinates remain exact zero at sampling, optimization, wrapper, and environment boundaries.

## 4. Failure Handling

The retune does not relax any safety gate. Training must still stop immediately for:

- non-finite state or optimizer values;
- nonzero inactive Panda action;
- fold error above `0.35 rad`;
- effort utilization above `1.0`;
- joint-limit proximity at or below `0.01 rad`.

The existing `Panda/root_joint` disjointed-body-transform PhysX warning and mount-wrench transient remain recorded evidence. The retune cannot be described as providing mechanical or real-hardware safety margin.

## 5. Verification Sequence

Implementation is accepted only in this order:

1. CPU/static tests prove the folded-load-only `120/8` override and legacy `M1_PANDA_CFG` `80/4` non-regression.
2. GPU0, 8 environments, 16 zero-action steps: all physical probe checks pass.
3. GPU0, 8 environments, 256 zero-action steps: fold, effort, and joint-limit gates pass over the full PPO horizon.
4. GPU0 8×1 training smoke: clean completion, finite PPO diagnostics, no `fold_hard_failure`, `accepted=false`.
5. GPU0 64×10 stability smoke: clean completion, bounded diagnostics, `accepted=false`.
6. Only after all five gates may the fresh L0-C0 curriculum launch.

Failed smoke artifacts remain diagnostic-only and can never initialize another stage. No threshold is weakened to obtain a pass.

## 6. Non-Goals

This change does not add Panda motion, gain scheduling, gravity compensation, external wrench, grasping, Student transfer, sensor fusion, payload certification, or hardware deployment. Those require separate design and verification.
