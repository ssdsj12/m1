# M1 + Panda Folded-Load L0-C0 Stop Diagnosis

## Purpose

Determine why the fresh 4096-env folded-load curriculum stopped shortly after launch.

## Stage And Todo

- Stage: T400.10b Task 10, L0-C0
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Launch evidence: [long launch](2026-08-25-m1-panda-folded-load-long-launch.md)

## Evidence

No folded-load curriculum/train/eval process remains, and GPU0 returned to baseline memory. There is no OOM, non-finite error, inactive-action leak, fold hard failure, or process traceback.

The L0-C0 training manifest records:

- `status=rejected` after evaluation;
- `completed_iterations=74` of requested `600`;
- `stop_reason=eligible_patience_50_updates`;
- `training_eligible=true`;
- `model_best.pt` exists with SHA `f2310099...`;
- no `model_final.pt` because fixed evaluation did not accept the candidate.

The best snapshot was update 23:

- timeout `1.0`;
- contact/orientation `0/0`;
- training-window `vx_rmse=0.0223452`;
- training-window `wz_rmse=0.0621072`;
- stationary `vx/wz=0.0010086/0.0007720`;
- all four directional bucket counts exceed 25.

No better eligible rank appeared for the next 50 guard updates, so update 74 intentionally triggered early stop and automatic evaluation.

All seeds 42/43/44 produced the same deterministic fixed-evaluation summary:

- timeout `1.0`;
- contact/orientation `0/0`;
- overall `vx_rmse=0.0354079 <= 0.04`;
- overall `wz_rmse=0.1060669 <= 0.12`;
- stationary `vx/wz=0.0008395/0.0002198`;
- forward/reverse/left/right counts `16/16/16/16`;
- `passed=false`.

Every reported global gate passes. The original reports did not serialize per-direction RMSE, so the exact failure was initially only identifiable as `directional_pass=false`.

The curriculum state therefore correctly records:

```text
status=stopped
stopped_stage=L0-C0
completed_stages=[]
rollback_checkpoint=null
reason=ValueError: stage L0-C0 must contain accepted=true
```

## Conclusion

This was a designed early stop followed by a failed fixed-direction acceptance test, not a crash. Restarting the same experiment blindly would discard the preserved rejection evidence and is not justified.

## Directional Re-Evaluation

Commit `23864b8` added JSON-safe per-direction metrics without changing any reward, threshold, training configuration, or acceptance decision. Commit `b03645c` added a `diagnostic_only` artifact path that can never publish `model_final.pt` and is explicitly rejected by curriculum parent validation.

The preserved checkpoint was copied, with identical SHA, into:

```text
Go2Pvcnn/logs/m1_panda_folded_load/foundation-v1/L0-C0-directional-diagnostic-v1/model_best.pt
sha256=f231009992ae07ae3de2560cfadb4d812fdb6cd38c8fa6deca7d4b2b8466ae8e
```

GPU 0 fixed evaluation was re-run for seeds 42, 43, and 44. All three produced identical results:

| Direction | Metric | Observed | Limit | Pass |
|---|---:|---:|---:|---:|
| forward | `vx_rmse` | `0.0496233165` | `0.04` | false |
| reverse | `vx_rmse` | `0.0504485387` | `0.04` | false |
| left | `wz_rmse` | `0.1498265562` | `0.12` | false |
| right | `wz_rmse` | `0.1501748692` | `0.12` | false |

Each direction contained 16 episodes and had zero base-contact and bad-orientation events. The global values remained `vx_rmse=0.0354079389 <= 0.04` and `wz_rmse=0.1060668891 <= 0.12`; averaging stationary and directional episodes therefore hid the four individual tracking failures.

The exact rejection cause is now established: **all four directional tracking buckets fail**, not only one direction. This is a tracking-quality problem under non-zero commands; it is not a stability/contact failure and not an evaluator crash.

Isolation checks after all three evaluations:

- diagnostic aggregate: `accepted=false`, `reports_passed=false`, `diagnostic_only=true`;
- diagnostic manifest: `status=diagnostic_complete`, `final_checkpoint=null`;
- diagnostic `model_final.pt`: absent;
- original L0-C0 manifest: still `status=rejected`, `accepted=false`, with no diagnostic marker;
- original checkpoint SHA: unchanged.

No retraining or curriculum restart was launched during this diagnosis.

## Recommended Follow-Up

Keep the existing acceptance thresholds. Redesign the first-stage command/tracking curriculum so forward, reverse, left, and right tracking improve independently, then train in a new run directory from iteration 0. Do not use this diagnostic directory as a parent or resume source.

## Git Refs

- Candidate code: `9314467`
- Launch record: `179e8f0`
- Directional report code: `23864b8`
- Diagnostic isolation code: `b03645c`
- Current Work Ref: `codex/m1-panda-ppo-stability`
