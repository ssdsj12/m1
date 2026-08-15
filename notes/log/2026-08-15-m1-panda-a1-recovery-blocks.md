# M1 + Panda A1 recovery blocks

## Result

正式 GPU0 recovery 从 strict sweep winner `model_2700.pt` fork，连续执行 20 个 500-iteration block，到 `model_12700.pt`。每个 block 都使用 seeds 42/43/44、64 env、2000 steps、满幅 20 N/5 Nm 独立评估。

没有 checkpoint 达到 `timeout >= 0.80`、`base_contact <= 0.10`、`bad_orientation <= 0.10` 的联合门。当前停止原因写为 `recovery_plateau_requires_design_review`；这不是 A1 验收通过。

## Best checkpoint

- checkpoint: `logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_9700.pt`
- SHA-256: `6a4a20ef8f102105b900a2e7f4237bab45ffafb0cf3dc0d07534406114a785a2`
- timeout survival: `0.7018633540`
- base contact: `0.2223602484`
- bad orientation: `0.0757763975`
- mean reward: `0.0144382988`
- accepted: `false`

原始候选 winner `model_2700.pt` 为 timeout/contact/orientation `0.455594/0.361015/0.183391`。恢复显著提高 survival 和 orientation，但 contact 在后期约 `0.22–0.37` 平台，未接近 `0.10` 门。

## Block summary

| Block | Checkpoint | Timeout | Contact | Orientation |
| ---: | --- | ---: | ---: | ---: |
| 1 | model_3200.pt | 0.269231 | 0.057072 | 0.673697 |
| 2 | model_3700.pt | 0.645631 | 0.253641 | 0.100728 |
| 3 | model_4200.pt | 0.641212 | 0.283636 | 0.075152 |
| 4 | model_4700.pt | 0.586166 | 0.369285 | 0.044549 |
| 5 | model_5200.pt | 0.585616 | 0.368721 | 0.045662 |
| 6 | model_5700.pt | 0.596945 | 0.347826 | 0.055229 |
| 7 | model_6200.pt | 0.581585 | 0.354312 | 0.064103 |
| 8 | model_6700.pt | 0.613909 | 0.322542 | 0.063549 |
| 9 | model_7200.pt | 0.597633 | 0.334911 | 0.067456 |
| 10 | model_7700.pt | 0.642257 | 0.286915 | 0.070828 |
| 11 | model_8200.pt | 0.651332 | 0.282082 | 0.066586 |
| 12 | model_8700.pt | 0.665441 | 0.275735 | 0.058824 |
| 13 | model_9200.pt | 0.692593 | 0.229630 | 0.077778 |
| 14 | model_9700.pt | 0.701863 | 0.222360 | 0.075776 |
| 15 | model_10200.pt | 0.674969 | 0.237858 | 0.087173 |
| 16 | model_10700.pt | 0.699875 | 0.229141 | 0.070984 |
| 17 | model_11200.pt | 0.660934 | 0.271499 | 0.067568 |
| 18 | model_11700.pt | 0.684932 | 0.247821 | 0.067248 |
| 19 | model_12200.pt | 0.679151 | 0.238452 | 0.082397 |
| 20 | model_12700.pt | 0.679012 | 0.248148 | 0.072840 |

## Artifacts and next decision

Manifest contains 21 ranking artifacts: the initial four-candidate selection plus 20 block evaluations. All rows passed finite, full-scale, six-axis coverage, seed and frozen-hash gates. Protected A0/A1 source checkpoints were not modified.

Further identical PPO continuation is not justified by the plateau evidence. Reaching the contact gate requires a separately approved redesign of reward/termination shaping or curriculum, followed by a new isolated fork and the same strict evaluation gate.
