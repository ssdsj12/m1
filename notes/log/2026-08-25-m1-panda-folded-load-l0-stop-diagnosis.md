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

Every reported global gate passes. By elimination against `evaluate_records`, the failed condition is inside `directional_pass`: at least one of forward/reverse has bucket `vx_rmse > 0.04`, or at least one of left/right has bucket `wz_rmse > 0.12`. The current evaluation artifact does not serialize per-direction RMSE, so the exact failing direction is not observable from retained output.

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

## Follow-Up

First add per-direction RMSE/pass fields to evaluation reports and re-run the existing `model_best.pt` evaluation. Only after identifying the failed direction should training behavior, command curriculum, reward, or patience be redesigned. Do not weaken the existing thresholds merely to promote L0-C0.

## Git Refs

- Candidate code: `9314467`
- Launch record: `179e8f0`
- Current Work Ref: `codex/m1-panda-ppo-stability`
