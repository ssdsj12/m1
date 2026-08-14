# M1 Minimal Smoke Adaptation Design

## Goal

Create an isolated copy of the Go2Pvcnn project under `/home/xk/coding/M1` and add a minimal M1 IsaacLab configuration that can load the M1 USD articulation, reset, and step without touching the existing Go2 semantic MPC training path.

## Scope

- Use `/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd` as the M1 USD entrypoint.
- Add a new `M1_CFG` asset config with M1 body and joint naming.
- Add a new `m1_smoke` task and Gym registration.
- Keep Go2 configs intact for reference.
- Do not wire M1 into the existing Go2 MPC planner yet.

## M1 Contracts

- Base body: `BASE_LINK`.
- Foot bodies: `FAR_FOOT_LINK`, `FBL_FOOT_LINK`, `RAR_FOOT_LINK`, `RBL_FOOT_LINK`.
- Controlled joints: all 16 M1 revolute joints:
  `FAR/FBL/RAR/RBL_{ABAD,HIP,KNEE,FOOT}_JOINT`.
- Initial smoke action dimension is 16.
- Initial smoke command/reward is intentionally simple and only validates robot loading and stepping.

## Non-Goals

- Reusing old Go2 checkpoints.
- Adapting Go2 MPC kinematics, foot order, or semantic reference rewards.
- Final locomotion reward tuning.

## Verification

- Static tests verify `M1_CFG` path, regexes, joint names, and new Gym id registration.
- A real IsaacLab smoke command should instantiate the M1 smoke environment with one env and run a small number of steps.
