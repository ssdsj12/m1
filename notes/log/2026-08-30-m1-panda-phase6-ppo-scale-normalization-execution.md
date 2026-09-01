# 2026-08-30 M1 + Panda Phase 6 PPO Scale/Normalization Execution

## Frozen implementation baseline

- Repository: `/home/xk/coding/M1`
- Branch: `main`
- Commit: `3a8dd51cf1b4d948c65a1c5191623440e93d6778`
- Device contract: `CUDA_VISIBLE_DEVICES=0`, Isaac/RSL device `cuda:0`
- Python: `/home/xk/miniconda3/envs/go2/bin/python`
- Focused CPU verification: `111 passed in 1.70s`
- Compile verification: `python -m compileall -q agent go2_pvcnn scripts rsl_rl/rsl_rl` exited `0`
- Whitespace verification: `git diff --check` emitted no output

## Phase 5 prerequisite

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 42 --headless \
  --summary-json Go2Pvcnn/logs/m1_panda_arm_mpc_probe/phase5_scale_norm_v3_s42.json
```

Authoritative result: `accepted=true`, steps `4000/4000`, MPC/QP rates
`1.0/1.0`, minimum wheel contacts `4`, base contacts `0`, joint-limit
violations `0`, resets `0`, maximum EE position error
`0.007537173326784423 m`, force/moment direction cosine
`0.9999999974317467 / 0.9999970430690538`.

## Fresh v3 artifact roots

- Pilot: `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3`
- Short: `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_scale_norm_v3`
- Long: `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_scale_norm_v3`

## Current status

Phase 5 prerequisite accepted. The corrected v4, v5, and v6 pilot/short runs
all completed safely.  Each short completed the full nine zero-pair plus
fifteen candidate fixed-condition workers, but every atomic promotion manifest
reported `accepted=false`.  No 3000-update long directory or process was
started.  The approved next design is a protected update-100-to-300 bridge with
exact empirical-normalizer count continuity, followed by the unchanged
24-worker promotion gate.

## Rejected v3 pilot and root cause

- Manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3/run_manifest.json`
- Result: `status=safe_complete`, `pilot_accepted=false`, ten of ten KL
  aborts, median completed mini-batches `4.0`, median value loss `6.2402`.
- Physical contract: zero hard failures and exact MPC/QP/four-contact rates
  of `1.0` throughout.
- Root cause: per-mini-batch low-KL updates raised the learning rate from
  `1e-5` to `1e-4`, while KL in `(0.015, 0.020]` crossed the hard abort
  threshold without crossing the shared scheduler's `2 * desired_kl`
  decrease threshold.
- Corrective design/plan commits: `5f96aae`, `6793159`, `0a4aa57`.
- Corrective implementation commit: `1b87602`; the residual task now keeps
  `max_learning_rate=learning_rate=1e-5` while retaining downward adaptation
  to `min_learning_rate=1e-6` and unchanged KL gates.

## Accepted guarded-learning-rate v4 pilot

Command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage pilot \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4 \
  --num_envs 8 --max_iterations 10 --seed 42 \
  --device cuda:0 --headless
```

- Exit: `0`.
- Manifest SHA-256:
  `019de32ec81a284a8bc539fc0d56e4463830b03885b113a01fe76b1c23693c0f`.
- Result: `status=safe_complete`, `completed_iterations=10`,
  `pilot_accepted=true`, `accepted=false`, `promotion_required=false`, and no
  candidate checkpoint.
- Pilot decision: no rejection reasons, KL abort count `0`, median completed
  mini-batches `8.0`, median value loss `6.396700620651245`.
- Optimizer bounds observed: maximum learning rate
  `9.999999747378752e-06`; active action standard deviation remained within
  `[0.009899962693452835, 0.01020970568060875]`.
- Every update reported zero hard failures and exact MPC/QP/four-contact
  rates of `1.0`.

## Safe-complete guarded-learning-rate v4 short

Command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage short \
  --pilot_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4/run_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4 \
  --num_envs 8 --max_iterations 100 --seed 42 \
  --device cuda:0 --headless
```

- Exit: `0`; elapsed training time approximately `2316.71 s`.
- Manifest SHA-256:
  `4d7b22c909f3daaff73f050ba70bca89ba0da6e19d18492428fdf65238775d56`.
- Result: `status=safe_complete`, `completed_iterations=100`,
  `stop_reason=requested_iterations_complete`, `accepted=false`, and
  `promotion_required=true`.
- Pilot parent SHA-256 exactly matches the accepted v4 pilot.
- Candidate completed updates are exactly `0/25/50/75/100`; all five file
  SHA-256 values match the manifest and all contain non-empty
  `obs_norm_state_dict` and `critic_obs_norm_state_dict` mappings.
- Online episode summaries retained length `4000`, time-out termination
  `1.0`, base-contact termination `0.0`, bad-orientation termination `0.0`,
  and mean reward approximately `101.98` at completion.

## Rejected guarded-learning-rate v4 promotion

- Promotion manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4/promotion_manifest.json`
- Manifest SHA-256:
  `c648bd8a515c08afd6b0de91af52c8641de41980d87fa880287a90f89664a2fe`.
- Evidence: nine of nine zero-pair workers and fifteen of fifteen candidate
  workers completed for exactly 4000 steps over seeds 42/43/44.
- Result: `status=rejected`, `accepted=false`, `best_checkpoint=null`.
- u000 and u100 were aggregate-equivalent.  u025/u050/u075 were rejected by
  the seed-42 wrench-regression gate.  No long run started.

The physical error path still compared the sensor's static loaded mount wrench
against a predicted dynamic reaction.  Commit `f632d41` changed the reward and
formal wrench diagnostic to compare the post-step bias-corrected dynamic
sensor wrench against the causal prediction.  A 4000-step seed-42 zero-pair
probe reduced the reported wrench RMS from approximately `94.94` to `18.98`
without changing the hard-safety outcome.

## Safe v5 runs and rejected dynamic-wrench promotion

- Pilot manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_dynamic_wrench_v5/run_manifest.json`
- Pilot manifest SHA-256:
  `5bd8c66eedbdb720b8ee193344ece7c6d02ebb06457d09d36bd10dac45e4f351`.
- Pilot result: ten of ten updates, `status=safe_complete`,
  `pilot_accepted=true`, no KL abort, and no physical hard failure.
- Short manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_dynamic_wrench_v5/run_manifest.json`
- Short manifest SHA-256:
  `60df429c45612bce3e34c0d948ad6477118e66317fb09b8c68d7fa0c79674072`.
- Short result: 100/100 updates, `status=safe_complete`, exactly five
  normalized candidates, and no online safety stop.
- Promotion manifest SHA-256:
  `1412a618ea3fdb1ad2b1cbb6c7adcb006047b31ee5b55d03c819628be078b881`.
- Promotion evidence: 9/9 calibration and 15/15 candidate workers complete;
  result `status=rejected`, `accepted=false`.
- u000 was aggregate-equivalent.  u025/u050/u075/u100 exceeded the unchanged
  seed-42 wrench regression tolerance by approximately
  `0.106/0.215/0.159/0.225`.  No long run started.

Runtime inspection then found that the reward consumed the corrected dynamic
wrench while observation indices 79:85 still exposed the raw filtered wrench
with its static bias.  Commit `4ba5324` aligned that observation slice to the
same corrected dynamic wrench without changing the 103D schema.

## Safe v6 runs and rejected corrected-observation promotion

- Pilot manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_corrected_obs_v6/run_manifest.json`
- Pilot manifest SHA-256:
  `f2237042fc03014836d40f0c86732fcd8ab2487206ae67f77e79d3bcf2049ac6`.
- Pilot result: ten of ten updates, `status=safe_complete`,
  `pilot_accepted=true`, all eight mini-batches completed per update, no KL
  abort, and zero hard failures.
- Short manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_corrected_obs_v6/run_manifest.json`
- Short manifest SHA-256:
  `4d827de3e76e0fb8321543ab41dce7061a9f76eb1744f6db5bb2f7602e742e50`.
- Short result: 100/100 updates, `status=safe_complete`,
  `stop_reason=requested_iterations_complete`, and exactly u000/u025/u050/
  u075/u100 with actor and critic normalizer dictionaries.
- Promotion manifest:
  `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_corrected_obs_v6/promotion_manifest.json`
- Promotion manifest SHA-256:
  `7b2a0c800d33c35a76414d1c6aa2b9785f6466e5e24a56f5f4c544d3527ae25b`.
- Promotion evidence: every one of the nine calibration and fifteen candidate
  workers completed for 4000 steps with exact seed and hash lineage.
- Result: `status=rejected`, `accepted=false`, `best_checkpoint=null`.
- u000/u025/u100 were aggregate-equivalent.  u050/u075 were rejected by the
  seed-42 wrench-regression gate with aggregate wrench deltas of approximately
  `+0.187/+0.162`; the tolerance remained `0.1`.
- u100 showed small favorable aggregate changes in roll/pitch RMS,
  end-effector position, and end-effector orientation, but none exceeded the
  calibrated rank tolerances.  No long run started.

## Post-v6 architecture diagnosis

A GPU0 seed-42 inference rollout of u100 confirmed that residual actions were
executed but remained small.  Per-channel normalized RMS values were:

```text
[0.002947, 0.001761, 0.000960, 0.002381,
 0.001184, 0.003001, 0.000924, 0.003225]
```

All normalized maxima stayed below `0.00872`.  These values map to corrections
too small to produce a tolerance-decisive improvement over the already strong
zero-residual controller in only 100 PPO updates.

Checkpoint inspection also established that empirical-normalizer mean,
variance, and standard deviation are persisted, but the processed sample count
is a plain integer and is not in the state dictionary.  A naive continuation
from u100 would therefore overwrite restored statistics on its first training
batch.  The approved bridge design first makes normalizer count an exact,
fail-closed part of checkpoint state, migrates the hash-matched v6 u100 count
as `100 * 256 * 8 = 204800`, then continues through total update 300 before
rerunning the unchanged promotion protocol.

The written design is committed at `5217989`:
`docs/superpowers/specs/2026-09-01-m1-panda-phase6-bridge-normalizer-continuity-design.md`.

## Bridge/normalizer continuity implementation preflight

The approved single-agent TDD implementation is now complete through the CPU
preflight.  No bridge GPU process, promotion worker, or long run had been
started at the time of this entry.

- Persistent empirical-normalizer count and first-observation normalization:
  commit `3c4a931`.
- Fail-closed v6 u100 lineage and atomic legacy-count migration: commit
  `4b324d1`.
- Guarded total-update 100-to-300 bridge stage: commit `b5a9cf4`.
- Schema-v3 bridge promotion and counted optimizer-preserving long lineage:
  commit `48832be`.
- Complete corrected residual/runner CPU suite: `134 passed in 1.89s`.
  The implementation plan's obsolete
  `test_m1_panda_arm_mpc_residual_runtime.py` name was replaced by the present
  `test_m1_panda_arm_mpc_residual_wrapper.py` and
  `test_m1_panda_arm_mpc_runtime.py` files.
- `python -m compileall -q agent go2_pvcnn scripts rsl_rl/rsl_rl`: exit `0`.
- `git diff --check`: no whitespace errors.

Immutable v6 input evidence:

- short manifest SHA-256:
  `4d827de3e76e0fb8321543ab41dce7061a9f76eb1744f6db5bb2f7602e742e50`;
- legacy u100 checkpoint SHA-256:
  `6a0bf84c007d1291218ed0573fc6fa44100dc7b5fe6051aefa44538375962b73`;
- migrated count contract: actor and critic both start at `204800` samples;
- source SHA-256 values remain asset `643fd061...`, config `4aab4e8e...`,
  reward `34576caf...`, and runtime `b5f8f17f...`.

Frozen GPU0 bridge command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage bridge \
  --short_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_corrected_obs_v6/run_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7 \
  --num_envs 8 --max_iterations 200 --seed 42 \
  --device cuda:0 --headless
```

Conditional promotion command after a verified safe-complete bridge:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_promote.py \
  --bridge_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7/run_manifest.json \
  --device cuda:0 --headless
```

The long command remains locked behind a parsed schema-v3
`promotion_manifest.json` with `accepted=true`, a SHA-valid bridge candidate,
equal scalar-int64 normalizer counts, and non-empty optimizer state.  Its fresh
root is
`Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_normalizer_continuity_v7`.

## Safe-complete v7 normalizer-continuity bridge

The frozen bridge command exited normally after approximately `1:13:28.63`.
The schema-v3 manifest SHA-256 is
`7de719d285bc66c2985be2685ddfda6f63ff8a48d19c51f7c8cdbb369901f07f`.
It records `status=safe_complete`, `200/200` additional updates, total update
`300`, `accepted=false`, `promotion_required=true`, and
`stop_reason=requested_iterations_complete`.

The immutable legacy parent remains
`6a0bf84c007d1291218ed0573fc6fa44100dc7b5fe6051aefa44538375962b73`;
its counted migrated u100 is
`c0e7752490f7a7ac2a7d3c29087bc0171872f350b14f5c382b76d2ee3e020b47`.
All five candidates contain finite model and optimizer state plus equal
scalar-int64 actor/critic normalizer counts:

| Total update | Checkpoint SHA-256 | Normalizer count |
| ---: | --- | ---: |
| 100 | `c0e7752490f7a7ac2a7d3c29087bc0171872f350b14f5c382b76d2ee3e020b47` | 204800 |
| 150 | `24126c85f09ad0a617a1778657034bddb914a9c888c26f11c712e6e23cc5b241` | 307208 |
| 200 | `6d290624db71c2e93c45791a77bba19a31e2d07055c7e57f22b99e8a8bb7ce60` | 409608 |
| 250 | `34d1ebbddaece4554e6cd9e3b10ea2d1a1091e42eac2ed49da53e90fd16a80c5` | 512008 |
| 300 | `1530badb0be9f339a732d2fab568848f98ffb814312cce0a5d16bb4cac7bc4d4` | 614408 |

The count sequence proves one normalized initial eight-environment
observation followed by exactly `256 * 8` samples per bridge update.  The
full bridge audit completed with `AUDIT_OK`; promotion is now unlocked, while
the 3000-update long run remains locked pending `accepted=true`.

## Rejected v7 promotion gate

The unchanged schema-v3 promotion gate completed all 24 isolated GPU0
workers: nine zero-pair noise-calibration workers and fifteen candidate
workers.  Every worker records `status=complete`, exactly `4000` steps, and
the required seed, checkpoint, source, and hash lineage.  The final promotion
manifest SHA-256 is
`9ad03f53a18a33740249e70b6def654a31b6b6a58986c0271969f47cb6796986`.

The fail-closed result is `status=rejected`, `accepted=false`, with
`best_checkpoint=null`:

| Total update | Decision |
| ---: | --- |
| 100 | `aggregate_equivalent` |
| 150 | `aggregate_equivalent` |
| 200 | `seed_42_wrench_regression` |
| 250 | `seed_42_rank_regression` |
| 300 | `seed_42_rank_regression` |

The final independent audit returned `PROMOTION_AUDIT_OK`.  It rehashed the
bridge manifest, all five candidate checkpoints, and the current asset,
configuration, reward, and runtime sources; it also checked the exact 9+15
worker set, equal scalar-int64 normalizer counts, and non-empty optimizer
state.  Because no candidate passed the unchanged promotion tolerances, the
fresh long-run root was not created and no 3000-update process was started.
