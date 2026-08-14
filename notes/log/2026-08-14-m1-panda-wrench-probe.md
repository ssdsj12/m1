# M1 + Panda deterministic wrench probe

## Purpose

验证真实 `Isaac-M1-Panda-Smoke-v0` 单环境中，施加到 `panda_hand` 的六轴已知载荷能否经 parent-on-child mount wrench 链路稳定路由到 `BASE_LINK` frame 的预期通道。

## Stage

Asset/wrench foundation Task 6 / real CPU simulation / known-load sign calibration.

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Implementation And TDD

### User-approved Round 4 contract

用户明确批准第 4 轮：每个 case 持续施加原载荷，先丢弃 `10` 个 transition steps，再采集新的 `50` samples。Pass 必须同时满足 excited-channel baseline-subtracted mean 为 expected sign、expected-sign sample fraction `>=0.90`（即至少 `45/50`）、magnitude ratio `>0.20`、transition/evaluation 全部 finite 且无 reset。JSON 每个 case 必须记录 `transition_steps`、`sample_steps`、`sign_count`、`sign_fraction`、mean、ratio 和 pass。载荷幅值、坐标转换、clear shim 与其他架构不得改变。

### User-approved independent-reset round

Round 4 已将 `force_z bad_orientation` 定位为跨轴累计姿态漂移后，用户批准每轴独立 deterministic reset/re-equilibrate。保留 one env；初始 settle row 与每个 axis 开始都先 clear、`env.reset()`、验证 reset result 与 25 DOF/unique body IDs，再做 100 zero-action settle、独立 50-step baseline、原载荷 10 transition discard、50 samples。mean sign、fraction `>=0.90`、ratio `>0.20`、finite/no reset 保持不变；不修改载荷、termination、坐标变换或 clear shim。

- 新增 `Go2Pvcnn/scripts/m1_panda_wrench_probe.py` 与静态/纯行为测试。
- 初始 RED：focused probe tests exit `1`, `5 failed`，原因是 probe 文件不存在。
- 初始 GREEN：exit `0`, `5 passed`。
- 初始测试覆盖：verbatim six-case table；100 settle/50 baseline/50 sample windows；one env；25 DOF；strict unique bodies；finite/no-reset failure；base→world→rotated-hand-local wrench transform；strict `>20%` magnitude gate；Kit close 前 artifact/nonzero lifecycle；Isaac Lab 2.1 empty-clear compatibility。Round 4 的 10-step transition 与 fraction/mean gate 见下文。
- 最终计划静态 regression：exit `0`, `50 passed in 0.88s`；`py_compile` exit `0`。

## Real Probe Attempts

Authority command for every attempt:

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 240 \
  /home/xk/miniconda3/envs/loco/bin/python \
  Go2Pvcnn/scripts/m1_panda_wrench_probe.py \
  --device cpu --headless \
  --output Go2Pvcnn/tests/artifacts/m1_panda_wrench_probe.jsonl
```

The planned `IsaacLab/isaaclab.sh -p` default-GPU route was not used as authority because the installed PyTorch supports CUDA only through `sm_90`, while the local RTX 5070 is `sm_120`; Task 3 already reproduced the incompatible GPU path. No environment or driver was modified.

Three bounded repair rounds were exhausted:

1. First run reported shell exit `0` but emitted neither artifact nor success JSON. Root cause: this Kit shutdown terminates the process inside `simulation_app.close()`, so post-close artifact/return-code logic was unreachable. RED lifecycle test exit `1`; fix moved artifact/flush and forced failure status before close; GREEN exit `0`.
2. Next run reliably exited `1`: documented `torch.zeros(0, 3, device=robot.device)` clear reached Isaac Lab 2.1 buffer assignment and attempted `[0] -> [29,3]`. Empty body selector was tested as the single hypothesis; RED then GREEN, but real run still exited `1` because `[0] -> [0,3]` is also rejected.
3. Final compatibility fix retained the required empty-tensor API call and accepted only the exact known shape-mismatch after verifying `has_external_wrench is False`; other exceptions remain fatal. RED then GREEN. The real run reached force calibration and reliably exited `1` on `force_y`.

## Runtime Evidence

- Real public Gym ID was imported, created, reset, and stepped with one environment on CPU.
- Action manager: `12 + 4 = 16`; observation includes the six-channel mount wrench; articulation DOF count check reached `25`.
- Strict distinct body IDs for `panda_hand`, `panda_link0`, and `BASE_LINK` passed before stepping.
- `settle` and `force_x` passed in-process. Exact rows were not persisted because output is intentionally committed only after all six cases pass.
- `force_y` baseline-subtracted mean:
  `[10.150298118591309, -27.970840454101562, -1.3658764362335205, 26.806659698486328, 27.21747589111328, -3.362870454788208]`.
- `force_y` expected sign: `-1`; excited-channel mean magnitude ratio: `1.3985420227050782`; matching samples: `47/50` (`sign_fraction=0.9399999976158142`); `stable_sign=false` under the current 50/50 rule; case pass `false`.
- Historical Round 3 did not reach `force_z` or torque cases; the superseding Round 4 result below reached `force_z` and terminated there.
- At the historical Round 3 checkpoint the required seven-row artifact was absent and was not hand-written or partially promoted; the final independent-reset round later produced it atomically.

## Asset And Dependency Verification

- `sha256sum -c assets/m1_panda/generated_files.sha256`: exit `0`, `2/2` pass.
- Generated hashes:
  - `panda/panda.usd`: `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
  - `m1_panda.usd`: `6acbd32afab08dbfb8963e0f7d990d2988cdfe8ad4fec083d0c9fa1c4585c3ff`
- CPU verifier command with `--device cpu --headless`: exit `0`; dependencies `8`; remote `0`; outside-root `0`; unresolved `0`; one articulation root; 25 DOF; 29 bodies; 25 joint names; one physics step; validation errors empty.
- Process-level network denial attempt `unshare --net true`: exit `1`, `Operation not permitted`. Dependency closure is verified, but network-denial runtime execution is unverified. No host network/firewall changes were made.

## Result

**PASS WITH LIMITATION.** Round 4 的历史累计状态在 `force_z` 触发 `bad_orientation`；用户随后批准每轴独立 reset/re-equilibrate。最终 fresh CPU authority exit `0`，七行 artifact 全部 finite/no-reset/pass。限制是 process-enforced network denial 未能执行。

## Architecture Options (Not Selected)

1. Define and pre-approve a sample sign-fraction threshold for `stable_sign`, separately from the unchanged strict `>20%` magnitude gate.
2. Add an explicit load-transition/ramp or discard window, then collect a fresh 50-sample steady-state evaluation window.
3. Re-equilibrate/reset deterministically between axes and retain a full 50-sample per-case baseline plus 50-sample loaded window.

## Follow-up

Asset/wrench foundation 已关闭。下一步仍需单独规划 residual policy、Teacher–Student training、IK/OSC、抓取、sensor driver、机械验算与实机验证；本日志不把这些后续阶段标为完成。

## Round 4 Result

- 新契约 RED：exit `1`, `4 failed, 6 passed`；GREEN：exit `0`, `10 passed`。termination diagnostics 追加 RED/GREEN 后 focused `11 passed`。
- 每轴使用 unchanged wrench，丢弃前 `10` steps，再评价新 `50` samples；`45/50` 边界通过、`44/50` 失败；mean expected sign 与 strict ratio `>0.20` 同时强制。
- 两次相同 CPU authority 均在到达 `force_z` 后 exit `1`；第二次精确记录：`terminated=[true]`、`truncated=[false]`、`base_contact=[false]`、`bad_orientation=[true]`、`time_out=[false]`。
- `force_x` 与 `force_y` 在两次运行中均通过新 gate，否则不会进入 `force_z`。由于 all-or-nothing 输出，精确 rows 未落盘。
- 临时路径与正式 artifact 均不存在；未执行 atomic replace。
- Fresh verification：planned static `53 passed`、pycompile exit `0`、checksum `2/2` exit `0`、PXR exit `0`、CPU verifier exit `0`（25 DOF、dependencies `8`、remote/outside/unresolved `0`）。JSONL 7-row/schema/all-pass 检查因无 candidate artifact 无法执行。
- Network denial 保持 unverified，未再次尝试或更改网络。

Round 4 未改变 force magnitudes、base→world→hand-local transform、clear shim 或其他架构。其历史阻断已由下方用户批准的 independent-reset round 取代。

## Independent-Reset Final Result

- 用户批准在同一个环境中对初始 settle 与每个轴分别执行：clear、`env.reset()`、25-DOF/body-id 复核、100 步 settle、50 步独立 baseline、10 步 transition 丢弃和 50 步 evaluation。
- 单代理补充 atomic artifact 与 exact empty-clear error tests：RED `2 failed`，GREEN focused `16 passed`；最终计划 regression `58 passed in 0.85s`，pycompile exit `0`。
- Fresh authority：

```bash
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 360 \
  /home/xk/miniconda3/envs/loco/bin/python \
  Go2Pvcnn/scripts/m1_panda_wrench_probe.py \
  --device cpu --headless \
  --output Go2Pvcnn/tests/artifacts/m1_panda_wrench_probe.jsonl
```

结果：exit `0`，wall `14.06s`，stdout `{"rows": 7, "pass": true}`。正式 JSONL 通过同目录临时文件、flush/fsync 与 `os.replace` 原子发布。

| Case | Excited delta | Ratio | Sign count | Pass |
| --- | ---: | ---: | ---: | --- |
| `force_x` | `-34.114151 N` | `1.705708` | `50/50` | true |
| `force_y` | `-35.731705 N` | `1.786585` | `50/50` | true |
| `force_z` | `-18.383438 N` | `0.919172` | `50/50` | true |
| `torque_x` | `-6.345034 N·m` | `1.269007` | `50/50` | true |
| `torque_y` | `-53.942345 N·m` | `10.788469` | `50/50` | true |
| `torque_z` | `-5.900999 N·m` | `1.180200` | `50/50` | true |

- JSONL validator：恰好 7 rows、25 DOF、全部 finite/no reset/pass、六轴 `sign_fraction=1.0`、ratio 全部 `>0.20`，exit `0`。
- Generated checksum：`2/2`，exit `0`。PXR behavior：cleanup/mount pass、root `/M1Panda/BASE_LINK`，exit `0`。
- CPU verifier：dependencies `8`、remote/outside/unresolved `0`、25 DOF、29 bodies、25 joints、physics step `1`、validation errors empty，exit `0`。
- Network denial：先前安全的 `unshare --net` 因权限不足，仍为 unverified；未修改主机网络。OmniHub inaccessible 的成功运行与 dependency closure 不能替代强制断网证明。

Result: DONE_WITH_CONCERNS。仅 asset/wrench foundation child 完成；Git Ref: unavailable。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: unavailable
- Git Ref: unavailable
- Key Files:
  - [Probe](../../Go2Pvcnn/scripts/m1_panda_wrench_probe.py)
  - [Probe tests](../../Go2Pvcnn/tests/test_m1_panda_wrench_probe_static.py)
