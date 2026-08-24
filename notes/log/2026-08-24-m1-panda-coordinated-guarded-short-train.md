# 2026-08-24 M1 + Panda coordinated guarded short train

## Purpose

验证 fresh coordinated PPO 的 GPU0 runner、100-episode guard、best/JSON/final 原子 checkpoint 链和 TensorBoard 诊断连线；50 updates 不作为收敛验收。

## Runs

- Wiring smoke: `coordinated_stability_wiring_8x1_final_20260824`，8 envs × 1 update，seed 42，GPU0 headless。
- Guarded short train: `coordinated_stability_guard_64x50_20260824`，64 envs × 50 updates，seed 42，GPU0 headless。
- 两次均从 fresh zero-action actor 开始；A1 `model_10402.pt` 仅记录 provenance，未加载 policy/optimizer。

## Wiring Smoke

- exit `0`；`model_0.pt`、`model_final.pt`、schema-2 manifest 均存在，final SHA 与 manifest 一致。
- status `completed_without_100_episode_candidate`，`accepted=false`，没有把不足 100 episode 的结果误标为 eligible best。
- TensorBoard: sampled reset flag `1`，joint position/velocity reset max `0.029028/0.049825`；KL `126.103775`，LR 被有界 adaptive schedule 降至 `1e-6`，physical std `0.009973`。

## 64 × 50 Guard Result

- exit `0`；普通 checkpoints、`model_best.pt`、`best_checkpoint.json`、`model_final.pt` 和 manifest 均生成。
- stop reason `max_updates`；diagnostic best iteration `32`。
- rolling timeout/contact/orientation `0.605156/0.047344/0.0`，mean reward `63.7126`；未满足 timeout `>=0.90`，所以 `accepted=false`。
- best checkpoint SHA、best JSON、final rollback SHA 和 manifest 字段一致。
- 最后记录 curriculum `0.3304`、force/torque norm max `10.6193/2.5100`、nonzero wrench ratio `0.546875`、KL `0.010585`、LR `7.59375e-6`、std `0.010195`。

## Result

checkpoint guard、回退链和训练诊断连线通过短训基础设施验收。该 fresh policy 在 50 updates 内没有通过行为门，因此不得称为收敛、稳定控制、抓取或实机能力；下一阶段是带相同 guard 的 64×600 长训监控。
