# M1 + Panda Coordinated Mission Smoke

## Scope

Implemented the approved two-stage coordinated mission: folded Panda during
M1 navigation, then end-effector tracking with bounded M1 base assistance when
arm null-space margin is low. The existing six-axis base-frame mount wrench
contract remains unchanged.

## Verification

- Pure mission/assist/adapter/static suite: `57 passed`.
- Python compilation: mission, base-assist, adapter, combined config and CLI
  all passed.
- CPU mission smoke (`--max_steps 2000`):
  `FOLD_AND_NAVIGATE=1819`, `ARRIVE_HOLD=1`, `UNFOLD_AND_TRACK=1`,
  `COORDINATED_TRACK=179`, `base_assist_count=179`, `ee_error=0.0`.
- Isaac Lab startup used `/home/xk/miniconda3/envs/go2/bin/python`, CUDA 0,
  combined task `Isaac-M1-Panda-Coordinated-v0`, and 1-2 environments.
  Scene creation reached the combined action manager with `joint_effort` shape
  `23`, confirming the M1+Panda action boundary.

## Dynamic Gate Status

Partial, with short-horizon re-verification. Isaac startup emitted the existing PhysX warning:
`CreateJoint - found a joint with disjointed body transforms ... Panda/root_joint`
and warned that the simulation may snap objects together. PXR verification
confirms this Panda `root_joint` is explicitly disabled; the active mount is
`AssemblerFixedJoint`, with one articulation root and exact mount-plane
contract. A 2-env, 20-step combined reset/step re-verification stayed finite,
with max mount-relative displacement `0.000370 m`. Long-horizon stability is
still open and must be independently verified before claiming full runtime
acceptance.

Student S1 was not modified. The six-dimensional `mount_wrench_b` signal stays
`[Fx, Fy, Fz, Tx, Ty, Tz]` in the base frame and retains its existing layout.
