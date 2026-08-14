# Task 5 Independent Review

## Spec Compliance

Task 5 **does not currently comply** with the required wrench contract.

The pure helper implements the requested algebra for an input that is already a world-frame wrench about `panda_link0` actor origin:

```text
M_base_origin_w = M_sensor_w + (p_sensor_w - p_base_w) x F_w
F_b = R_wb F_w
M_b = R_wb M_base_origin_w
```

and returns unnormalized `[Fx, Fy, Fz, Mx, My, Mz]`. However, the adapter's premise that `get_link_incoming_joint_force()` supplies that world-frame/actor-origin input is contradicted by the installed Isaac Sim PhysX tensor behavior tests used with local IsaacLab 2.1. Those tests construct expected forces and torques in world coordinates, explicitly inverse-rotate both into each **joint frame**, and compare those joint-frame values directly with `get_link_incoming_joint_force()`. Their torque lever arms are measured about the corresponding incoming joint origins, not the child actor origins or COMs. Consequently, the adapter treats a joint-frame wrench as world-frame and rotates it again by the base world quaternion.

The local IsaacLab 2.1 high-level `body_incoming_wrench` docstring says “simulation world frame” (`IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:176-185`), while the installed lower-level PhysX behavior test establishes joint-frame behavior (`.../omni.physics.tensors.tests/.../tests.py:3040-3153`). The implementation followed the former statement without resolving this contradiction. Exact `AssemblerFixedJoint` joint-frame pose relative to `panda_link0` actor frame was not established by the reviewed Python source/API documentation; that exact asset relationship and live numerical behavior remain **Cannot verify — Task 6 must measure them**. This uncertainty cannot justify treating the raw wrench as world-frame.

Sign evidence is better: the same vendor behavior test's static force balance is consistent with the incoming wrench being the parent-on-child joint reaction. For example, the fixed child has a net external `+5 Y` load and the expected fixed-joint incoming force is `-5 Y`. Current production naming says “incoming mount wrench” and does not incorrectly claim the reverse direction. Task 6 should still calibrate the sign on the actual combined asset and document the sensor-facing convention.

Other binding points are satisfied: explicit unique resolution of `panda_link0` and `BASE_LINK`, Python integer indexing, actor poses and `(w,x,y,z)` base quaternion retrieval, exact smoke `ObsTerm` params, concatenation enabled, corruption disabled, no wrench normalization/noise/clipping, unchanged 12 leg plus 4 wheel actions, and no Student/Teacher/IK/OSC additions.

## Strengths

- `shift_rotate_wrench_to_base` is compact and implements the requested positive lever-arm shift and `[F,M]` ordering correctly for its declared inputs (`Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py:9-25`).
- Adapter lookup deliberately ignores resolved `SceneEntityCfg.body_ids` and uniquely resolves both explicit names with `preserve_order=True`; it checks both ID and name cardinality (`m1_panda_wrench.py:35-47`). This avoids confusing the manager selector for adapter-owned mount/base indices.
- Live adapter indexing uses Python integers, so it does not introduce CPU index tensors into CUDA tensors. The selected wrench, positions, quaternion, cross product, rotations, and concatenation naturally retain the live tensor device and dtype.
- Quaternion order and inverse rotation usage are correct for IsaacLab actor poses: `body_quat_w` is documented as `(w,x,y,z)`, and `quat_rotate_inverse` maps world vectors into the actor frame.
- Smoke wiring has the exact three params required and adds the wrench after the preserved existing observations (`Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py:40-69`). The group disables corruption and concatenates terms; the wrench term itself has no noise or processing options.
- Static smoke tests inspect the full observation term order and exact wrench params, then re-run the full 12+4 action contract assertion (`Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py:235-265`).
- Pure transform tests include identity shift, yaw, and a batched nonzero lever-plus-rotation case with hard-coded expected results (`Go2Pvcnn/tests/test_m1_panda_wrench.py:44-81`). Adapter tests verify exact slices, body poses/quaternion, explicit-name priority over conflicting `SceneEntityCfg` fields, single backend call, and missing/duplicate cardinality (`test_m1_panda_wrench.py:84-171`).
- Adding the module import after existing observations creates no evident circular dependency: the new module imports only Torch and IsaacLab math. The reported real package import exit `0` is consistent with that reading.

## Issues

### Critical

1. **Raw PhysX wrench is interpreted in the wrong frame and at the wrong assumed point.** `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py:48-54`

   `get_link_incoming_joint_force()` is sliced and immediately passed as `force_w`/`torque_w`. But the installed PhysX behavior test explicitly transforms world expectations into incoming joint frames before comparison (`/home/xk/miniconda3/envs/go2/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors.tests/omni/physicstensorstests/scripts/tests.py:3115-3125,3145-3153`). The same test's moment magnitudes use distances from incoming joint origins. Thus the adapter does not satisfy its declared input contract. Even if the mount joint orientation is identity relative to the base, rotating a joint/base-frame force by `quat_rotate_inverse(base_quat_w, ...)` again makes the output vary incorrectly with base world attitude. The implementation must first establish the mount joint world pose and transform the raw joint-frame wrench into the declared world-frame/about-sensor-point representation (or derive an equivalent direct joint-to-base spatial transform), then shift to the `BASE_LINK` actor origin. The exact fixed-joint pose and runtime convention are Task 6 `Cannot verify` items, not assumptions.

### Important

1. **Tests cannot falsify the adapter's actual API-frame mistake.** `Go2Pvcnn/tests/test_m1_panda_wrench.py:84-143`

   The fake backend labels arbitrary tensor channels as `incoming`, while the transform is monkeypatched to a sentinel. This proves indexing and argument plumbing only. The pure tests separately assume their inputs are already world-frame/about-sensor-origin. No test models a rotated incoming joint frame, a joint origin distinct from child actor origin, or the raw API's parent-on-child convention. The batched rotation-plus-lever test is useful for the helper but cannot validate the adapter-to-PhysX boundary. Add a falsifiable adapter-level joint-frame/lever case based on the verified API convention, then retain Task 6 as the real combined-asset calibration.

2. **Task report and log overstate what source inspection established.** `.superpowers/sdd/task-5-report.md:14-18`; `notes/log/2026-08-14-m1-panda-base-frame-mount-wrench.md:15-26`

   They assert that the incoming wrench is world-frame and that its torque can be shifted from the mount actor origin. The high-level IsaacLab comment says world-frame, but the lower-level vendor behavior test contradicts it and demonstrates joint-frame values. These artifacts should not record Task 5 as PASS until the production conversion is corrected and the unresolved exact mount joint-point relation is explicitly assigned to Task 6.

### Minor

1. **Wildcard export also leaks imported module names.** `Go2Pvcnn/go2_pvcnn/mdp/__init__.py:9`; `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py:5-6`

   Because `m1_panda_wrench.py` has no `__all__`, `from .m1_panda_wrench import *` also places `torch` and `math_utils` in the package namespace. This is not a circular-import failure and does not break the current smoke import, but defining `__all__` (or importing the two functions explicitly) would keep the public API intentional.

## Assessment

**Changes required / FAIL.** The mathematical helper, smoke wiring, action preservation, body lookup, quaternion convention, tensor device/dtype behavior, and most static tests are sound. The central adapter boundary is not: local lower-level source evidence shows the raw incoming joint wrench is expressed in the incoming joint frame and its torque is associated with the joint origin, while production treats it as a world-frame wrench about the child actor origin. This is a correctness defect in Task 5, not merely a deferred Task 6 calibration. Task 6 remains necessary to verify the exact combined-asset joint pose, live numerical sign/magnitude, and sensor-facing sign convention after the conversion is fixed.
