# Right palm orientation boundary diagnosis

- Parent: [T500.4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
- Stage: Object MPC → dual Arm MPC / full-action teacher.
- Baseline Ref / Candidate Ref: `80954d2` plus existing working-tree integration;
  this investigation adds read-only probe diagnostics only.
- Key file: [probe](../../Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py).
- Procedure: from `Go2Pvcnn`, run the existing probe using the go2 environment,
  `--headless --device cuda:0 --steps 350 --seed 0`.
- Evidence: [350-step JSON](../../Go2Pvcnn/tests/artifacts/m1_dual_panda_o6_orientation_boundary_350.json).
- Static entrypoint tests: 9 passed; simulation command exited 0. This is a
  successful diagnostic reproduction, not successful physical acceptance.

## Findings

The first arm infeasibility (184), first left pinky limit event (270), and
first safety rejection (329) exactly match the preceding 1600-step run.

The planner's future-node geometric rates remain at or below 0.35 rad/s.
However, measured-pose-to-first-target displacement divided by 0.04 s grows
from 0.35 at step 0 to 2.535 at step 160 and 7.290 rad/s at step 184. This
quantity is an implied correction rate, not a measured angular velocity.
The existing bound is relative to the committed command, not measured tracking.

At step 184 the norm of the difference between direct rotation-vector
subtraction and `Log(R_target R_measured.T)` is 0.372009 rad. Current palm
rotation vector is approximately (1.237889, 1.983522, 0.331056), far from the
identity where a simple coordinate subtraction approximates a spatial rotation.
The dual-arm resampler and teacher difference rotation-vector coordinates,
while the arm QP uses a spatial geometric Jacobian. This proves a representation
mismatch, but does not prove that it alone causes all subsequent failures.

## Next experiment

First correct the T500 adapter's orientation interpolation, spatial error and
angular feedforward consistently, preserving public T400/T500 dataclass fields.
Run the same 350-step sample before altering any geometry or angle bounds.
Then evaluate a tracking-aware first-node bound and downstream feasibility
feedback if the corrected adapter still allows the reference to outrun tracking.
The 1600-step physical gate and formal 30-trial acceptance remain unpassed.
