# M1 + Panda A1 Teacher Resume: 500 Iterations

The combined M1 + Panda A1 Teacher was resumed from `model_9902.pt` with its
lineage-matched A0 base `model_2999.pt`, using 64 environments on GPU0.

- Completed iterations: `9903..10402`
- Final checkpoint: `Go2Pvcnn/logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt`
- Final mean reward: `6.20001220703125`
- Final mean episode length: `454.29`
- Timeout termination: `0.7376302481`
- Base-contact termination: `0.1529947966`
- Bad-orientation termination: `0.109375`

The A1 foundation continuation exited successfully. The next ordered stage is
the new coordinated mission Teacher; its PPO runner must still be implemented
and connected to the combined Isaac state/action adapter before training it.
