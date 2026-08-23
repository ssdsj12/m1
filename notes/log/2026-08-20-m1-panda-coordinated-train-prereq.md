# M1 + Panda Coordinated Teacher Training Prerequisite

## Scope

Completed the combined 23-effort PPO runner prerequisite after A1 Teacher
resume. The A1 checkpoint is accepted as lineage/init metadata only; its
60-observation/16-action actor is not shape-loaded into the 67-observation,
23-action coordinated policy.

## Verification

- Static contract test: `1 passed`.
- Python compilation: wrapper and training entrypoint passed.
- Isaac GPU0 smoke: 2 environments, 1 PPO iteration, exit `0`.
- Combined task: `Isaac-M1-Panda-Coordinated-v0`.
- Policy observation: `67` values, including unchanged 6D
  `mount_wrench_b = [Fx, Fy, Fz, Tx, Ty, Tz]`.
- Action manager and actor output: `23` joint-effort values.
- PPO output: `48` timesteps, value loss `2.2907`, surrogate loss `0.2314`.
- Checkpoint: `Go2Pvcnn/logs/m1_panda_coordinated/m1_panda_coordinated_prereq_smoke3/model_0.pt`.

## Boundary

The PPO runner prerequisite is pass. This is not a dynamics acceptance claim:
Isaac still emits the existing `Panda/root_joint` disjointed-body-transform
snap warning. Asset/PXR re-verification confirms `root_joint` is explicitly
disabled and the active `AssemblerFixedJoint` is the only M1/Panda mount;
the serialized mount plane, single articulation root, 25 DOF and no-NaN
contracts pass. A 2-environment combined run for 20 physics steps stayed
finite with maximum mount-relative displacement `0.000370 m`. This is a
short-horizon re-verification, not a long-horizon dynamics acceptance.
Student S1 was not started or modified.
