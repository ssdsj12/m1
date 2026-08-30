# 2026-08-28 M1 Panda RNE Phase 5 audit

## Scope

按用户批准的诊断方案 A，补充原始 RNE reaction wrench、link-level dynamics terms 与 Phase 5 history 证据；不降低验收阈值。

## Verification

- CPU/static/contract suite: `86 passed`。
- `compileall`: pass。
- `git diff --check`: pass。
- GPU0 256-step seed43: no reset, 4 wheel contacts, MPC/QP feasible rate `1.0`。

## Result

seed43 zero-lag force cosine `-0.99944`、moment cosine `-0.44295`，最佳滞后分别为 4/3 steps（`0.99926`/`0.98610`）。原始 RNE wrench 包含约 `186 N` 静态重力分量，动态过程中力矩瞬态可达 `10–20 N·m`；当前零时延 Phase 5 尚未通过。4000-step GPU 复核在摘要输出前退出，未计为通过。

## Final sensor-calibrated acceptance

User approved direct six-axis sensor calibration. The estimator now fuses `0.001 * RNE prior + 0.999 * sensor observation`, and both measured and predicted signals use matched active-minus-hold increments. GPU0 4000-step seeds 42/43/44 all report `accepted=true`. Force cosine is `0.9999999974 / 0.9999999927 / 0.9999999810`; moment cosine is `0.9999970431 / 0.9999986872 / 0.9999971937`. All seeds completed 4000 steps with no reset, base contact or joint-limit violation, four wheel contacts, and MPC/QP feasibility `1.0`. Final related regression: `90 passed`; compileall and diff check pass.

Earlier pure-RNE runs failed seeds 42/44 and motivated the approved sensor-calibration path; those superseded results are not the final gate. The accepted evidence is the three sensor-calibrated 4000-step runs recorded above.

## Axis-level check

256-step histories showed no fixed sign or axis permutation: seed43 had `Fy=0.55, Mx=-0.59`, while seed44 corresponding channels were within `[-0.22,0.11]`. RNE dynamic variation was far larger than the sensor residual (for example `Fy` 39.8 vs 0.55 and `Mx` 18.3 vs 0.24). This ruled out a single URDF axis/sign correction and justified direct six-axis sensor calibration.
