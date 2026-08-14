# Task 5 Fix Round 1 Independent Review

## Spec Compliance

Fix Round 1 complies with the Task 5 wrench contract and closes the prior Critical, both Important issues, and the Minor export issue.

The production data path is now consistent end to end:

```text
raw parent-on-child wrench
  (incoming joint frame, about incoming joint origin)
    -- quat_rotate(panda_link0 actor quaternion) -->
world-frame wrench about panda_link0 actor origin
    -- M + (p_mount - p_base) x F -->
world-frame wrench about BASE_LINK actor origin
    -- quat_rotate_inverse(BASE_LINK actor quaternion) -->
[Fx, Fy, Fz, Mx, My, Mz] in BASE_LINK frame
```

This conversion is valid because the formal fixed-joint contract now requires body1 to be `panda_link0`, `localPos1=(0,0,0)`, and `localRot1=(1,0,0,0)`. Thus the child-side joint origin and axes coincide with the child actor origin and axes. IsaacLab's `(w,x,y,z)` `body_quat_w` maps actor-local vectors to world with `quat_rotate`; using `quat_rotate_inverse` only after the world-frame moment shift maps the result into `BASE_LINK` coordinates. There is no duplicate or reversed rotation.

The builder checks this contract before export and after reopening the serialized asset (`Go2Pvcnn/scripts/build_m1_panda_asset.py:62-97,107-123,127-130,192-194`). The runtime verifier independently inspects and reports the same child-side pose (`Go2Pvcnn/scripts/verify_m1_panda_asset.py:20-25,86-106,157-190,264-275`), and the PXR behavior check reads the formal USD and asserts body targets plus exact child pose (`Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py:76-92`). This is sufficient to use the mount actor pose in the adapter; Task 6 does not need to rediscover the already-locked frame contract.

The adapter continues to preserve the required explicit unique mount/base resolution, incoming body index, actor poses, base quaternion, batch dimension, device, and dtype. Python integer body indices introduce no CPU tensor. `quat_rotate`, `torch.linalg.cross`, `quat_rotate_inverse`, and `torch.cat` operate directly on the same simulation tensors; no normalization, clipping, casting, or host transfer was added.

The smoke observation contract remains exact, concatenated, and corruption-disabled; the 12 leg-position plus 4 wheel-velocity actions are unchanged, and no Student/Teacher/IK/OSC or Task 6 behavior was added.

## Strengths

- The adapter documents the raw parent-on-child, joint-frame/about-joint-origin convention and implements the necessary joint-to-world rotation explicitly (`Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py:30-65`).
- Both force and torque are rotated with the mount actor quaternion before the pure helper, preserving their common joint-frame basis (`m1_panda_wrench.py:55-64`).
- The pure helper remains a clear, independently tested world-frame spatial-wrench transform with the required positive lever shift and output ordering (`m1_panda_wrench.py:11-27`).
- Asset invariants are enforced at generation, serialized reopen, independent verification, and direct PXR inspection rather than being recorded only in prose.
- The new adapter-level test combines a 90-degree joint/base rotation, raw joint-frame force and torque, and a nonzero world lever (`Go2Pvcnn/tests/test_m1_panda_wrench.py:116-136`). Its expected `[1,0,0,0,0,3]` is correct: raw `+X` becomes world `+Y`, the lever contributes `+1 Mz`, raw `+2 Mz` remains `+2 Mz`, and base inverse rotation returns force to `+X`. The old implementation would produce approximately `[0,-1,0,0,0,2]`, so the test genuinely falsifies the reviewed defect.
- The report and log now state the joint-frame/joint-origin API behavior, the asset child-frame equivalence, the corrected conversion, and the narrow remaining Task 6 boundary accurately.
- Explicit imports plus module `__all__` close the prior wildcard namespace leak (`Go2Pvcnn/go2_pvcnn/mdp/__init__.py:9`; `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py:8`).

## Issues

### Critical

None.

### Important

None.

### Minor

None.

## Assessment

**Approved.** The previous frame/reference-point defect is fixed rather than deferred. The child-side USD joint pose is now a checked production invariant, so using `panda_link0` actor quaternion and position for the raw joint-frame wrench is justified. The adapter-level regression directly distinguishes the corrected implementation from the old one, documentation is aligned, and public exports are intentional.

Task 6 remains responsible only for live known-load sign/magnitude calibration and the sensor-facing sign convention on the combined simulation asset. That calibration does not reopen the now-established Task 5 coordinate-frame and moment-point contract.
