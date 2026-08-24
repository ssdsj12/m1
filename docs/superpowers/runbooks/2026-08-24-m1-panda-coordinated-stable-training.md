# M1 + Panda Coordinated Stable PPO Runbook

## Scope

This runbook operates the fresh 103-observation/23-action coordinated normal-control policy at 200 Hz. It covers randomized balance under Panda-hand wrench and joint/root/material domain randomization. It does not establish grasping, payload, sensor-overload, or real-hardware acceptance.

Run commands from `/home/xk/coding/M1/Go2Pvcnn` with `/home/xk/miniconda3/envs/go2/bin/python`; GPU commands use `CUDA_VISIBLE_DEVICES=0 --device cuda:0`.

## Deterministic Coordination Play

This command exercises the deterministic coordinated mission adapter, not a learned PPO checkpoint:

```bash
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_coordinated_play.py \
  --num_envs 1 --max_steps 200 --seed 42 --device cpu
```

## GPU Randomization and Physics Probe

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_coordinated_randomization_probe.py \
  --num_envs 8 --steps 256 --seed 42 --device cuda:0 --headless \
  --output tests/artifacts/m1_panda_coordinated_randomization_probe.json
```

Proceed only when the process exits 0 and `hard_gates_passed` is true. This checks seeded reset equality, cross-environment diversity, selected-reset isolation, exact state/material/wrench bounds, finite simulation, and a nonzero mount response to a `panda_hand` wrench.

## One-Update Wiring Smoke

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_coordinated_train.py \
  --num_envs 8 --max_iterations 1 --seed 42 \
  --run_name coordinated_stability_wiring_8x1_manual \
  --init-a1-checkpoint logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless
```

The A1 file is lineage/provenance only; actor, critic and optimizer start fresh. A one-update run normally has fewer than 100 completed episodes, so `status=completed_without_100_episode_candidate` and `accepted=false` are correct.

## Guarded Short Train

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_coordinated_train.py \
  --num_envs 64 --max_iterations 50 --seed 42 \
  --run_name coordinated_stability_guard_64x50_manual \
  --init-a1-checkpoint logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless
```

This validates `model_best.pt`, `best_checkpoint.json`, `model_final.pt`, manifest hashes and rollback behavior. It is not a convergence test; keep `accepted=false` if any eligibility gate fails.

## Production 64 × 600 Train

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_coordinated_train.py \
  --num_envs 64 --max_iterations 600 --seed 42 \
  --run_name coordinated_stable_fresh_s42_64x600_20260824 \
  --init-a1-checkpoint logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless
```

Never reuse a nonempty run directory. The guard uses the latest 100 completed episodes, eligibility `timeout>=0.90`, `base_contact<=0.05`, `bad_orientation<=0.05`, catastrophe hard-failure `>0.20` for 25 updates, eligible patience 50, and at most 600 updates.

## TensorBoard

```bash
/home/xk/miniconda3/envs/go2/bin/tensorboard \
  --logdir logs/m1_panda_coordinated --port 6006
```

Inspect `Loss/kl`, `Loss/learning_rate`, `Loss/lr_adjustment`, `Policy/mean_noise_std`, `DomainRandomization/curriculum_scale`, force/torque norms, nonzero wrench ratio, and reset deviation metrics. LR must stay in `[1e-6,3e-4]`; physical std must stay in `[0.005,0.05]`.

## Stop, Rollback, and Recovery

- Stop manually with `SIGINT` only when necessary; inspect `run_manifest.json` before using outputs.
- `accepted=true` means an eligible 100-episode best existed and `model_final.pt` was restored from it. It is not a grasping or hardware claim.
- `completed_without_eligible_best` means the final checkpoint is the diagnostic best but failed at least one behavior gate.
- `completed_without_100_episode_candidate` means too few completed episodes existed to rank a candidate.
- Verify SHA fields before play/evaluation:

```bash
sha256sum RUN_DIR/model_best.pt RUN_DIR/model_final.pt
/home/xk/miniconda3/envs/go2/bin/python -m json.tool RUN_DIR/run_manifest.json
/home/xk/miniconda3/envs/go2/bin/python -m json.tool RUN_DIR/best_checkpoint.json
```

Do not resume this fresh 600-update experiment with an old actor/optimizer. If rerunning after infrastructure failure, choose a new empty run name and start from zero again.

## Old long-v4 Checkpoint Audit

The pruning tool defaults to dry-run and accepts only the exact long-v4 directory:

```bash
/home/xk/miniconda3/envs/go2/bin/python \
  scripts/prune_m1_panda_coordinated_checkpoints.py \
  --run-dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823 \
  --keep-through 3500
```

`--apply` is destructive. It writes `checkpoint_pruning.json` in `planned` state before unlinking and `completed` only after the postcondition passes; deleted checkpoints require an external backup for recovery.
