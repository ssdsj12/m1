# M1 + Panda Phase 5–6 Arm MPC + 8D Residual PPO Implementation Plan

> **Execution:** Use `superpowers:executing-plans` with one agent. Implement every task RED → GREEN → focused regression → commit.

**Goal:** Add a deterministic 50 Hz Panda arm MPC and a fresh 8D residual PPO path for stationary M1 balance plus small six-axis end-effector motion, without changing legacy 23D tasks.

**Architecture:** The public RL boundary is `103 observations -> 8 normalized residual actions`. An environment adapter turns those actions into physical wrench/height/stance corrections, combines them with MPC feedforward and measured-wrench feedback, safety-projects the request, and runs the existing 200 Hz WBC/QP to emit the internal 23 actuator efforts. MPC replans every four physics steps and holds its last safe reference between replans.

**Tech stack:** Python 3.11, PyTorch float64 control / float32 PPO, Isaac Sim 5.1, Isaac Lab ManagerBasedRLEnv, vendored RSL-RL, pytest, GPU0.

## Frozen contracts

- Work in `/home/xk/coding/M1` on the current branch; do not stage the two `graphify-out/cache/last_query_stamp` files.
- Preserve all old Gym IDs, wrappers, policies and checkpoint shapes.
- New public task: `Isaac-M1-Panda-ArmMpc-Residual-v0`, observation `(103,)`, action `(8,)`; its private articulation-effort bridge remains `(23,)`.
- Physics/WBC `200 Hz`; MPC `50 Hz`, `dt=0.02`, 20 nodes, `0.4 s` horizon.
- Canonical wrench is base frame, base origin, order `[Fx,Fy,Fz,Mx,My,Mz]`.
- `W_cmd = -W_arm_mpc + W_feedback + W_rl`; safety projection occurs before WBC/QP.
- First task excludes rolling, payload, grasp, pushes, broad domain randomization, Student and real deployment.
- Long training is blocked until Phase 5, zero-residual and short-train gates pass.

## Standard commands

```bash
cd /home/xk/coding/M1
export PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/xk/miniconda3/envs/go2/bin/python
```

GPU programs must run directly with `$PY` and `--device cuda:0`; do not use `go2pvcnn.sh`.

---

### Task 1: Arm MPC contracts and condensed linear model

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc.py`

- [ ] Write tests for strict float64/finite/device/shape validation, default `dt/horizon`, and the discrete model:

```python
q1 = q + 0.02 * qd + 0.5 * 0.02**2 * qdd
qd1 = qd + 0.02 * qdd
twist1 = jacobian @ qd1
pose_delta1 = jacobian @ (q1 - q)
```

Test that a zero acceleration sequence preserves constant velocity, and malformed `q`, `J`, horizon targets or limits raise `ValueError` before solving.

- [ ] Run RED:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc.py -q
```

- [ ] Implement immutable `ArmMpcCfg`, `ArmMpcInput`, `ArmMpcSolution`, `ArmMpcDiagnostics`, and pure helpers `rollout_linearized_arm()` / `condense_arm_dynamics()`. Required shapes are `q/qd (7,)`, `J (6,7)`, target pose/twist `(20,6)`, dynamics blocks `(7,7)` and `(6,7)`, limits `(7,)`.

- [ ] Run GREEN plus existing QP-contract tests, then commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc.py Go2Pvcnn/tests/test_m1_panda_qp_backend.py -q
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py Go2Pvcnn/tests/test_m1_panda_arm_mpc.py
git commit -m "feat: add linearized arm MPC contracts"
```

### Task 2: Deterministic arm MPC QP, limits and fallback

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc.py`

- [ ] Add failing tests for a 140-variable acceleration QP (`20*7`) with pose/twist tracking, acceleration and slew penalties, rest-posture regularization, and linear position/velocity/acceleration constraints. Assert all predicted samples obey soft limits.
- [ ] Add sign test `predicted_dynamic_mount_wrench_b = base_arm_coupling @ qdd[0]`, a static-motion test returning zero dynamic wrench, deterministic repeatability, infeasible fallback to the previous safe reference, and first-cycle fallback to current `q`/zero `qd`/zero wrench.
- [ ] Run RED, then implement `LinearizedArmMpc.plan(input)`. Reuse `DenseQpProblem` and `solve_reference_qp`; never add a second solver. Store a cloned last-safe solution only after full finite/constraint validation. Diagnostics must distinguish `feasible`, `fallback_used`, `fallback_reason`, iterations, saturation, EE error, `sigma_min` and joint margins.
- [ ] Run GREEN and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc.py Go2Pvcnn/tests/test_m1_panda_qp_backend.py -q
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py Go2Pvcnn/tests/test_m1_panda_arm_mpc.py
git commit -m "feat: solve constrained Panda arm MPC"
```

### Task 3: Runtime extraction and canonical wrench integration

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_runtime.py`

- [ ] Test exact generalized-coordinate slicing: arm indices are `6 + panda_joint_ids`, arm mass is `M[arm,arm]`, base-arm coupling is `M[:6,arm]`, and arm bias is `(coriolis+gravity)[arm]`. Test target horizon construction, float64 CPU copies, and canonical wrench metadata.
- [ ] Implement `PhysxTeacherAdapter.build_arm_mpc_input(target_pose_horizon, target_twist_horizon)` using the same state snapshot as `build_state`; reject stale/nonfinite data atomically.
- [ ] Run RED/GREEN and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc_runtime.py Go2Pvcnn/tests/test_m1_panda_residual_wbc.py -q
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_runtime.py
git commit -m "feat: expose arm MPC runtime dynamics"
```

### Task 4: MPC reference and feedforward through the existing WBC chain

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/m1_panda_residual_wbc_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_controller.py`

- [ ] Write tests proving: an optional `(q_ref,qd_ref)` overrides only Panda reference generation; omitting it is bitwise-equivalent to the old path; the composer receives `-predicted_dynamic_mount_wrench + feedback + rl`; RL correction is safety-projected before QP; MPC and residual fallbacks are separate diagnostics.
- [ ] Extend `M1PandaWbcTeacher.step` with an optional immutable `ArmReference`. Extend the residual controller step with optional `predicted_mount_wrench_b` and `arm_reference`, defaulting to zero/None so old callers do not change.
- [ ] Run GREEN plus Phase 1–4 controller regression and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_controller.py Go2Pvcnn/tests/test_m1_panda_residual_wbc.py Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py -q
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/m1_panda_residual_wbc_wrapper.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_controller.py
git commit -m "feat: feed arm MPC references into residual WBC"
```

### Task 5: Exact grouped 103-to-8 actor critic

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/residual_actor_critic.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_residual_actor_critic.py`
- Create: `Go2Pvcnn/tests/test_rsl_policy_class_resolution.py`

- [ ] Test frozen slices `59/20/6/(6+4+8)`, encoder outputs `128/64/32/32`, fusion `256->128->8`, independent critic parameters, `(N,8)` action and `(N,1)` value, deterministic inference, finite validation and strict state-dict reload.
- [ ] Implement `ResidualActorCritic` with the complete RSL interface (`act`, `act_inference`, `evaluate`, `get_actions_log_prob`, `update_distribution`, `std`, `entropy`, `clip_std`). Keep normalized actions at the policy boundary.
- [ ] Test then add generic `resolve_policy_class(name)` to the runner: legacy bare names resolve from current globals; dotted names use `importlib`. Configure no application-specific import in vendored RSL.
- [ ] Run regression and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_residual_actor_critic.py Go2Pvcnn/tests/test_rsl_policy_class_resolution.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py -q
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/residual_actor_critic.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/test_m1_panda_residual_actor_critic.py Go2Pvcnn/tests/test_rsl_policy_class_resolution.py
git commit -m "feat: add grouped 8D residual actor critic"
```

### Task 6: Stability-first rewards, curriculum and checkpoint guard

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py`
- Create: `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_guard.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py`

- [ ] Test `g_safe in [0,1]`, task reward suppression near instability, hard-failure dominance, EE/wrench/slip tracking, residual magnitude/rate/intervention penalties, and curriculum restricted to small stationary EE trajectories.
- [ ] Test stability-first lexicographic comparison, rolling-window eligibility, atomic best manifest, `accepted=false` without an eligible best, rollback after sustained regression, and saturation hard gate `<0.01` per channel.
- [ ] Implement pure tensor reward/curriculum functions and a serializable guard. Do not import Isaac in these modules.
- [ ] Run and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py -q
git add Go2Pvcnn/go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_guard.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py
git commit -m "feat: add residual stability reward and guard"
```

### Task 7: Dedicated 8D/103D environment boundary

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_env_cfg.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_env_static.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py`

- [ ] Static-test the new ID, 200 Hz physics, private 23-effort action bridge, disabled old task defaults, stationary command ranges and no payload/push/randomization.
- [ ] Unit-test a fake underlying env: wrapper public `num_actions=8`, observation exactly 103 via the frozen builder, four-step MPC cadence, held reference between replans, one independent planner/controller per environment, 8D normalized validation, 23D internal effort, reset-state clearing and diagnostic aggregation.
- [ ] Implement the wrapper as the RSL VecEnv contract. It must never expose the private 23D command as the policy action. If per-env CPU QP throughput cannot meet the short gate, report/block instead of silently bypassing MPC/WBC.
- [ ] Run and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_env_static.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py Go2Pvcnn/tests/test_m1_panda_residual_observation.py -q
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_env_static.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py
git commit -m "feat: add 8D arm MPC residual environment"
```

### Task 8: PPO config and guarded training entrypoint

**Files:**
- Create: `Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py`
- Modify: `Go2Pvcnn/agent/__init__.py`
- Create: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`

- [ ] Test rollout `256`, adaptive KL `0.01`, bounded LR/std, dotted actor class, fresh initialization, max `3000`, short-gate default, immutable manifest fields and refusal to launch long mode without an accepted short-run manifest.
- [ ] Implement CLI `--stage {zero,short,long}`, `--device cuda:0`, `--num_envs`, `--max_iterations`, `--seed`, `--headless`. `long` must validate asset/config hashes and the short-run accepted manifest. Wire iteration callbacks to guard, atomic best checkpoint and rollback; never load legacy 23D models.
- [ ] Run and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py Go2Pvcnn/tests/test_rsl_ppo_adaptive_schedule.py -q
git add Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py Go2Pvcnn/agent/__init__.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py
git commit -m "feat: add guarded residual PPO training"
```

### Task 9: Phase 5 probe, evaluation and GUI Play

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py`
- Create: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py`
- Create: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py`

- [ ] Static-test CLI/device/seed/checkpoint handling and that Play sets `env_cfg.viewer`/manager window disablement before `gym.make`, preventing the delayed `viewport_camera_controller` callback while retaining viewport rendering.
- [ ] Implement probe summaries for all exact Phase 5 gates, including conditional wrench-direction samples, fixed-seed zero-residual baseline deltas and explicit eligible sample counts. Evaluation must compare trained vs zero residual lexicographically and write `accepted=false` on any hard-gate failure.
- [ ] Run static tests and commit:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py -q
git add Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py
git commit -m "feat: add arm MPC probe eval and play"
```

### Task 10: CPU regression and GPU0 Phase 5/6 gates

**Files:**
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_gpu_smoke.py`
- Create: `notes/log/2026-08-26-m1-panda-arm-mpc-phase5-gate.md`
- Create: `notes/log/2026-08-26-m1-panda-residual-ppo-phase6-gate.md`

- [ ] Run the focused CPU suite, then the complete CPU suite:

```bash
$PY -m pytest Go2Pvcnn/tests/test_m1_panda_arm_mpc.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_runtime.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_controller.py Go2Pvcnn/tests/test_m1_panda_residual_actor_critic.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_env_static.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py -q
$PY -m pytest Go2Pvcnn/tests -q
```

- [ ] Run GPU0 Phase 5 zero-residual probe with one env and fixed seeds. Record exact command, commit, seed, steps, feasible/contact/reset/tilt/EE/wrench/baseline results. Do not weaken thresholds to obtain a pass:

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py --device cuda:0 --num_envs 1 --steps 4000 --seeds 42 43 44 --headless
```

- [ ] If Phase 5 passes, run Stage 0 and bounded Stage 1 short training/evaluation. Start at 8 envs; benchmark and reduce rather than bypass exact QPs if necessary:

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py --stage zero --device cuda:0 --num_envs 8 --max_iterations 10 --seed 42 --headless
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py --stage short --device cuda:0 --num_envs 8 --max_iterations 100 --seed 42 --headless
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py --device cuda:0 --seeds 42 43 44 --headless
```

- [ ] Commit only test code and evidence logs, never generated checkpoints:

```bash
git add Go2Pvcnn/tests/test_m1_panda_arm_mpc_gpu_smoke.py notes/log/2026-08-26-m1-panda-arm-mpc-phase5-gate.md notes/log/2026-08-26-m1-panda-residual-ppo-phase6-gate.md
git commit -m "test: verify arm MPC residual GPU gates"
```

### Task 11: Runbook, notes and conditional long launch

**Files:**
- Create: `docs/superpowers/runbooks/2026-08-26-m1-panda-arm-mpc-residual-training.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/index.md`
- Modify: `notes/log/index.md`

- [ ] Document exact probe/train/eval/play commands, artifact locations, manifest schema, stop/rollback behavior, GUI limitation and non-goals.
- [ ] Update T400 Phase 5/6 status using only verified evidence. Mark long training `blocked` if the accepted short manifest is absent.
- [ ] Only after an accepted short manifest, launch up to 3000 updates on GPU0:

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py --stage long --device cuda:0 --num_envs 8 --max_iterations 3000 --seed 42 --headless
```

- [ ] Run final compile/tests, inspect diff/status, and commit docs:

```bash
$PY -m compileall -q Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination Go2Pvcnn/go2_pvcnn/tasks Go2Pvcnn/go2_pvcnn/training Go2Pvcnn/agent Go2Pvcnn/scripts
$PY -m pytest Go2Pvcnn/tests -q
git diff --check
git status --short
git add docs/superpowers/runbooks/2026-08-26-m1-panda-arm-mpc-residual-training.md notes/todo/T400-m1-panda-force-aware-teacher-student.md notes/index.md notes/log/index.md
git commit -m "docs: add arm MPC residual training runbook"
```

## Completion rule

Implementation is not “complete” merely because files, checkpoints or a training process exist. Completion requires fresh CPU regression evidence, GPU0 Phase 5 physical gates, zero-residual equivalence, guarded short training, fixed-seed comparison, an accepted atomic best manifest, rollback verification and the runbook. A 3000-update launch remains conditional and must never be reported as successful training before evaluation accepts it.
