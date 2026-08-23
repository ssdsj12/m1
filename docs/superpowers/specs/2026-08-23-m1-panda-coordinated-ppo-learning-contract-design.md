# M1 + Panda Coordinated PPO Learning Contract Design

## Goal

Turn the already executable `Isaac-M1-Panda-Coordinated-v0` PPO prerequisite into a learnable coordinated task before spending GPU time on a long run.

## Frozen boundaries

- Keep the accepted zero-clearance combined asset and 25-DOF single articulation.
- Keep policy action width `23`, applied as bounded residual effort to 12 leg, 4 wheel and 7 Panda arm joints; fingers remain position-held.
- Keep mount wrench order `[Fx,Fy,Fz,Tx,Ty,Tz]` in `BASE_LINK` frame.
- Keep Student S1, grasping, payload and real hardware out of scope.
- Keep physics `dt=0.005 s`, decimation `1`, flat ground and GPU0.

## Observation contract

The coordinated Teacher policy receives 103 float32 values:

1. root linear velocity in body frame `3`;
2. root angular velocity `3`;
3. projected gravity `3`;
4. controlled joint position relative to default `23`;
5. controlled joint velocity `23`;
6. planar base target error in body/yaw frame `3`;
7. Panda hand pose error in `BASE_LINK` frame `6`;
8. desired task twist `6`;
9. binary four-wheel contact `4`;
10. mount wrench `6`;
11. previous normalized action `23`.

Every term must be finite, batched and reset-safe. Base targets are relative to `env_origins`. The EE target is a small base-frame offset from the reset hand pose, so it is reachable and independent of environment placement.

## Action and mission contract

The actor output is clamped to `[-1,1]` before the existing 23-effort action manager. Implicit actuator posture remains the nominal controller; PPO learns bounded residual effort. During navigation, the task command exposes the planar base error and rewards a folded Panda. After the base is inside `0.08 m` and `0.10 rad`, the EE tracking reward activates and bounded base displacement remains permitted.

## Reward and safety contract

Positive terms: alive, exponential planar target tracking, yaw tracking, post-arrival EE position/orientation tracking. Penalties: pre-arrival folded-arm error, vertical/base angular velocity, tilt, wheel/foot slip, action rate, residual effort and mount-wrench clipping diagnostic. Existing base-contact and bad-orientation terminations remain hard. No reward may depend on privileged WBC/Teacher action.

## Gates

1. Static/pure tests freeze width `103`, target frames, finite behavior, phase gating, action clamp and reward signs.
2. GPU0 8-env one-iteration smoke must expose 103 observations, 23 actions, finite loss and a checkpoint.
3. GPU0 64-env 100-iteration sanity must show finite optimization, nonzero base-target progress and no catastrophic reset growth.
4. Only then start an isolated 64-env long run. Save every 100 iterations and monitor reward, episode length, base/EE errors, reset causes, action saturation and mount wrench.

## Failure handling

Do not resume an incompatible 67-observation checkpoint. On NaN, shape drift, repeated base-contact growth or mount-wrench normalization overflow, stop the run and retain the latest finite checkpoint and manifest. The disabled Panda `root_joint` warning is informational after the passed two 8×2000 dynamics gates unless measured mount drift regresses.
