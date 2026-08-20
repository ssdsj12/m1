# M1 + Panda A1 Resume Training Block

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A1 --num_envs 64 --seed 42 --max_iterations 100 \
  --resume-checkpoint Go2Pvcnn/logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_9900.pt \
  --base-checkpoint Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --headless --device cuda:0
```

## Result

Pass. The strict A1 checkpoint lineage accepted the matching A0 base checkpoint
and trained iterations `9901` through `10000` with 64 combined M1 + Panda
environments. The run exited `0` and wrote `model_9902.pt` in the existing A1
run directory.

- Final mean reward: `6.280526638031006`
- Final base-contact termination: `0.265625`
- Final bad-orientation termination: `0.0625`
- Maximum observed mount wrench magnitude: `19.999818801879883`
- Policy observation: `60`; M1 action: `16`; mount wrench: `6`

This is a resumed A1 force-aware Teacher foundation block. It does not yet
train the new two-stage folded-navigation/end-effector coordinated mission;
that mission still needs its Isaac state adapter and PPO runner after the
combined-articulation snap warning is resolved or explicitly accepted.
