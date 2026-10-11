# PPO Wire 操作策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 在 DeformX reduced Cosserat wire 环境中训练 15D latent PPO，先获得接触/夹持/抬升 approval checkpoint，再训练 wire 操作动作策略。

**Architecture:** 新增一个 PPO 专用 wire 任务包装器，读取现有双 Panda+双 O6 snapshot 和 reduced-wire runtime，输出固定归一化观测并接收 15D latent。Stage A/B 使用同一个环境接口和不同 reward/curriculum 配置；MPC 只执行安全投影、限位、碰撞约束和原子 fallback。训练脚本生成 checkpoint、逐 episode JSONL 和 SHA-pinned manifest。

**Tech Stack:** Isaac Lab/Gymnasium、PyTorch、RSL-RL PPO、DeformX reduced Cosserat runtime、Python 3.11。

## Global Constraints

- PPO action dimension is exactly `15` and must not expose the full joint vector.
- Acceptance requires each hand >=2 valid contacts, force >`0.2 N`, lift >=`0.05 m`, hold >=`1 s`, slip <=`0.02 m`.
- Stage B cannot start unless a Stage A approval checkpoint and manifest pass validation.
- MPC remains safety projection/fallback only; no new contact MPC, VLM, perception, or Residual training.
- GPU training uses `CUDA_VISIBLE_DEVICES=0`; CPU helper threads are capped at one.
- Failed or non-finite episodes never enter the successful trajectory set.
- Preserve T400/T500 public contracts and all unrelated dirty/untracked files.

---

### Task 1: Define PPO wire observation/action contracts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_ppo_contracts.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_ppo_contracts.py`

**Interfaces:**
- `WIRE_PPO_ACTION_DIM: int = 15`
- `WirePpoObservationSpec.observation_dim: int`
- `decode_wire_latent(action: torch.Tensor, *, device, dtype) -> WireLatentCommand`
- `validate_wire_ppo_transition(observation, action, reward, done) -> None`

- [ ] Write failing tests for exact action shape `(15,)`, finite bounds `[-1,1]`, deterministic decode, observation shape/dtype/device, and atomic rejection of NaN/Inf.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=Go2Pvcnn python -m pytest -q Go2Pvcnn/tests/test_m1_wire_ppo_contracts.py`; expect collection/definition failures.
- [ ] Implement immutable dataclasses for normalized observation and decoded latent fields; reject malformed tensors before changing controller state.
- [ ] Run the focused suite and assert all tests pass.
- [ ] Commit: `feat: add wire PPO contracts`.

### Task 2: Add Stage A/B reward and curriculum

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_wire_ppo_curriculum.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_ppo_curriculum.py`

**Interfaces:**
- `WirePpoStage` enum with `APPROVAL` and `ACTION`
- `WirePpoRewardCfg`
- `compute_wire_ppo_reward(previous, current, stage, cfg) -> RewardBreakdown`
- `approval_episode(current) -> bool`

- [ ] Write failing tests for reward terms, stage gating, exact acceptance thresholds, and rejection of non-finite/failed diagnostics.
- [ ] Run the focused tests and verify failure before implementation.
- [ ] Implement curriculum progression `APPROACH -> CONTACT -> GRASP -> LIFT -> HOLD`; Stage B reward adds centerline, tangent, deformation, synchronization, force-balance, slip, and latent-smoothness terms.
- [ ] Ensure approval is true only for 2 contacts/hand, force >`0.2 N`, lift >=`0.05 m`, hold >=`1 s`, slip <=`0.02 m`, and no safety failure.
- [ ] Run focused tests and commit `feat: add staged wire PPO curriculum`.

### Task 3: Integrate 15D latent policy with bimanual wire wrapper

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_wire_ppo_env.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_ppo_env.py`

**Interfaces:**
- `M1WirePpoEnv.reset(seed=None) -> tuple[torch.Tensor, dict]`
- `M1WirePpoEnv.step(action: torch.Tensor) -> tuple[obs, reward, terminated, truncated, info]`
- `M1WirePpoEnv.stage: WirePpoStage`

- [ ] Write failing tests for latent decode to dual palm/O6 targets, MPC safety projection fallback, observation update, episode termination, and no state mutation on invalid actions.
- [ ] Run tests to verify failure.
- [ ] Implement adapter using existing snapshot, `DloWireTrajectoryPlanner`, reduced-wire runtime, and wrapper safety projection; keep contact gate unchanged and fix fingertip geometry mapping from commit `42c0909` in the integration path.
- [ ] Add deterministic reset and per-episode diagnostics (`stage`, `success`, `failure_reason`, contact/lift/hold/slip metrics).
- [ ] Run CPU env tests and existing wire runtime/contact suites; commit `feat: add PPO wire environment`.

### Task 4: Implement two-stage GPU PPO training entrypoint

**Files:**
- Create: `Go2Pvcnn/scripts/m1_wire_ppo_train.py`
- Create: `Go2Pvcnn/config/m1_wire_ppo_train.yaml`
- Test: `Go2Pvcnn/tests/test_m1_wire_ppo_train_static.py`

**Interfaces:**
- CLI `--stage {approval,action}`
- CLI `--resume-checkpoint PATH` required for `--stage action`
- CLI `--num-envs`, `--max-iterations`, `--seed`, `--device`, `--run-dir`

- [ ] Write failing static tests for stage parsing, approval checkpoint requirement, GPU0 selection, one-thread environment variables, and manifest fields.
- [ ] Run tests to verify failure.
- [ ] Implement RSL-RL PPO runner with 15D action space, stage-specific reward config, checkpoint validation, deterministic seed, and atomic best-checkpoint publication.
- [ ] Emit `episodes.jsonl`, `training.manifest.json`, source/config SHA256, stage, seed, acceptance counts, and terminal reason counts.
- [ ] Run static tests and commit `feat: add staged wire PPO training entrypoint`.

### Task 5: Validate approval then action training on GPU0

**Files:**
- Create: `Go2Pvcnn/scripts/m1_wire_ppo_verify.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_ppo_verify_static.py`
- Artifact: `Go2Pvcnn/artifacts/m1_wire_ppo_*`

**Interfaces:**
- `m1_wire_ppo_verify.py --stage approval|action --checkpoint PATH --seed 42 --episodes N`

- [ ] Write failing tests for manifest SHA verification, successful-episode filtering, and refusal to launch action stage without approval.
- [ ] Implement CPU manifest checks and GPU0 smoke runner with single-thread CPU limits.
- [ ] Run a short Stage A smoke; only after an approval checkpoint passes, run Stage B smoke.
- [ ] Produce per-episode JSONL and SHA-pinned aggregate manifest; record whether bilateral contact, lift, and hold were reached.
- [ ] Commit `test: add wire PPO verification artifacts` only after verification output exists.

