# Task 5 Report

Status: FIX_ROUND_1_AWAITING_REVIEW

Git Ref: unavailable

## Changes

- Added `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py`: pure transform and environment adapter.
- Exported both functions from `go2_pvcnn.mdp`.
- Wired exact mount/base params into the isolated smoke policy observation.
- Added Kit-free tensor/adapter tests and exact smoke AST coverage; the 12 leg-position + 4 wheel-velocity action contract is unchanged.
- Updated T400 dashboard/branch/progress and an independent log/index entry.

## Frame And Sign Contract

The raw PhysX incoming wrench is parent-on-child, expressed in the incoming joint frame and about the joint origin. The formal asset proves the child-side joint pose is `localPos1=(0,0,0)`, `localRot1=(1,0,0,0)`, so it coincides with the `panda_link0` actor frame/origin. The adapter first uses the mount world quaternion to rotate raw `[Fx,Fy,Fz,Mx,My,Mz]` into world. The unchanged pure transform then computes `M_BASE_w = M_mount_w + (p_mount_w - p_BASE_w) x F_w` and inverse-rotates force/moment into `BASE_LINK`. Output remains unnormalized `[F,M]` about the base actor origin.

The adapter uses `asset_cfg.name` only and resolves the explicit mount/base names independently with `preserve_order=True`; exact returned names and exactly one ID are required. Python integer indexing preserves the incoming tensor device.

## RED / GREEN And Verification

- Initial RED: `1 failed, 11 passed, 8 errors in 1.79s`, exit `1` (missing module/term only).
- First GREEN: `20 passed in 0.92s`, exit `0`.
- Unique-ID self-review RED: `1 failed, 4 passed in 0.77s`, then fixed with explicit ID cardinality.
- Planned regression: `25 passed in 0.84s`, exit `0`.
- Task 4 expanded regression: `41 passed in 0.86s`, exit `0`.
- Related `py_compile`: exit `0`.
- Bounded loco/source real Isaac `quat_rotate_inverse` and production pure-function calls: exit `0`; lever+yaw output approximately `[2,0,0,0,0,5]`.
- Bounded loco bootstrap + headless AppLauncher public export/real cfg/exact params/actions assertions: exit `0`.

## Self-review And Boundary

Tests cover batch lever+rotation, sign/order, adapter slices/poses/quaternion/output, explicit-name priority, `preserve_order`, and missing/duplicate bodies. Smoke adds no noise, normalization, Student/Teacher, IK, or OSC; Task 6 is untouched. No physical smoke env create/step or known-load live PhysX sign calibration was performed.

## Fix Round 1

- Root cause: the initial adapter trusted IsaacLab's high-level “world frame” docstring, but the installed vendor `TestLinkIncomingJointForce` behavior test constructs torque about each incoming joint origin, inverse-rotates world force and torque into the joint frame, and compares those values directly with `get_link_incoming_joint_force()`.
- Read-only bounded AppLauncher+PXR evidence for `/M1Panda/Panda/panda_link0/AssemblerFixedJoint`: body0 `/M1Panda/BASE_LINK`, body1 `/M1Panda/Panda/panda_link0`, localPos0 `(0,0,0.1389991939)`, localRot0 identity, localPos1 zero, localRot1 identity.
- Hypothesis RED: rotated base/mount with raw joint-frame `+X`, raw `Mz=2`, and nonzero world lever. Old output was approximately `[0,-1,0,0,0,2]`; required output `[1,0,0,0,0,3]`. Together with export/asset-contract checks: `4 failed, 22 passed`.
- Fix: rotate raw force and torque once from mount/joint frame into world with `quat_rotate`, then call the unchanged world-frame pure helper using mount actor position as the joint origin. Builder, verifier, and PXR behavior now enforce child local zero/identity. Public exports are explicit and module `__all__` contains only the two functions.
- Focused GREEN: `26 passed in 0.88s`. Final planned four-file suite: `26 passed in 0.81s`; all five Task 5/Task 4 related files: `42 passed in 0.81s`; Task 2–4 focused without wrench: `32 passed in 0.04s`; pycompile exit `0`.
- Asset evidence: rebuild exit `0`; 2/2 generated checksums unchanged/pass; PXR behavior exit `0`; CPU verifier exit `0`, 25 DOF, one physics step, child local pose recorded, `validation_errors=[]`.
- Real Isaac evidence: raw joint→world→base production sequence exit `0`, output approximately `[1,0,0,0,0,3]`; headless AppLauncher public export/cfg/action check exit `0`.
- Remaining Task 6 boundary: calibrate live combined-asset numerical sign/magnitude and sensor-facing convention. Parent-on-child naming is retained. Task 5 is awaiting independent re-review, not claimed PASS.
