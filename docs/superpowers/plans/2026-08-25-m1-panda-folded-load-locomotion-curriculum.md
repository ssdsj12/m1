# M1 + Panda Folded-Load Locomotion Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a guarded L0-C0 through L2-D3 PPO curriculum that teaches M1 to carry a dynamically simulated, PD-folded Panda while moving forward, backward, and turning without arm actions or external wrench.

**Architecture:** Add a new folded-load task alongside the existing coordinated task so rejected coordinated policies and wrench logic cannot leak into the foundation. Extend RSL-RL with a generic active-action mask and per-update KL abort, then connect focused command, reset, reward, metrics, guard, training, evaluation, and atomic orchestration modules. Preserve the combined USD, 103 observations, 23 actor outputs, 200 Hz control, and legacy tasks.

**Tech Stack:** Python 3, PyTorch, Isaac Lab manager-based environments, RSL-RL PPO, Gymnasium, pytest, JSON/SHA-256 artifacts, TensorBoard.

## Global Constraints

- Execute this plan with one agent using `superpowers:executing-plans`; the user explicitly prohibited subagents.
- Preserve the combined USD, single-articulation topology, asset SHA-256, 103-wide observations, 23-wide actor output, and 200 Hz control.
- Canonical action order is 12 leg efforts, 4 wheel efforts, and 7 Panda coordinates; only indices `0:16` are active.
- Panda remains dynamic and uses the existing fold pose `(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)` and implicit PD limits from the asset.
- L0/L1 have no external wrench and no domain randomization; L2 randomizes only the approved root, leg, and friction fields.
- Never initialize L0-C0 from any previous coordinated run. Only an immediately preceding `accepted=true` stage may initialize a later stage.
- Use test-driven development, preserve unrelated dirty-worktree files, and commit each task separately.

## File Structure

- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py`: pure stage, command, DR, bucket, and lineage contracts.
- `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_folded_load.py`: Isaac tensors for commands, reward terms, reset randomization, and physical diagnostics.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py`: isolated manager-based task configuration and 103/23 boundary.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py`: command scheduling, inactive-action enforcement, episode attribution, and fold diagnostics.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py`: rolling eligibility, catastrophe logic, rank, and atomic artifact state.
- `Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py`: exact stability PPO defaults.
- `Go2Pvcnn/scripts/m1_panda_folded_load_{probe,train,eval,curriculum}.py`: physical probe, one-stage training, fixed evaluation, and stage orchestration.
- `Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py`: reusable active-action probability boundary.
- `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`: reusable remaining-minibatch KL abort.

---

### Task 1: Pure curriculum, command, and DR contracts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_curriculum.py`

**Interfaces:**
- Produces: `StageSpec`, `ResetRanges`, `CommandBatch`, `STAGE_ORDER`, `stage_spec(name)`, `sample_episode_commands(...)`, `balanced_eval_commands(...)`, and `classify_command_buckets(...)`.
- Consumes: only Python dataclasses and PyTorch; no Isaac imports, so unit tests run outside the simulator.

- [ ] **Step 1: Write failing stage-table tests**

```python
def test_stage_order_and_ranges_are_exact():
    assert STAGE_ORDER == ("L0-C0", "L1-C1", "L1-C2", "L1-C3", "L1-C4", "L2-D1", "L2-D2", "L2-D3")
    assert stage_spec("L0-C0").vx_limit == 0.05
    assert stage_spec("L1-C4").wz_limit == 0.60
    assert stage_spec("L2-D3").reset.friction == (0.80, 1.20)
    assert stage_spec("L2-D3").reset.root_z == (0.0, 0.0)
    assert stage_spec("L2-D3").reset.panda_position == (0.0, 0.0)
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_curriculum.py`

Expected: collection fails because `m1_panda_folded_load_curriculum` does not exist.

- [ ] **Step 3: Implement immutable stage and reset tables**

```python
@dataclass(frozen=True)
class ResetRanges:
    root_xy: float = 0.0
    root_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_linear_velocity: float = 0.0
    root_angular_velocity: float = 0.0
    leg_position: float = 0.0
    friction: tuple[float, float] = (1.0, 1.0)
    root_z: tuple[float, float] = (0.0, 0.0)
    wheel_position: tuple[float, float] = (0.0, 0.0)
    panda_position: tuple[float, float] = (0.0, 0.0)
    panda_velocity: tuple[float, float] = (0.0, 0.0)
    restitution: tuple[float, float] = (0.0, 0.0)

@dataclass(frozen=True)
class StageSpec:
    name: str
    parent: str | None
    vx_limit: float
    wz_limit: float
    completed_episode_window: int
    reset: ResetRanges
```

Define the eight exact stages from the design, with L2 retaining C4 command limits and windows of 200 for command levels and 400 for DR levels.

- [ ] **Step 4: Add deterministic command-distribution tests**

```python
def test_command_sampler_has_exact_families_and_balanced_signs():
    batch = sample_episode_commands(10_000, stage_spec("L0-C0"), seed=7)
    counts = torch.bincount(batch.family, minlength=4).float() / 10_000
    torch.testing.assert_close(counts, torch.tensor([.20, .25, .20, .35]), atol=.015, rtol=0)
    assert batch.twist[:, 1].eq(0).all()
    assert batch.twist[:, 0].abs().max() <= .05
    assert batch.twist[:, 2].abs().max() <= .15

def test_fixed_eval_table_balances_required_buckets():
    commands = balanced_eval_commands(64, stage_spec("L2-D3"))
    buckets = classify_command_buckets(commands)
    for name in ("forward", "reverse", "left", "right"):
        assert buckets[name].sum() >= 8
```

Implement categorical allocation with probabilities `(.20, .25, .20, .35)`, nonzero magnitudes sampled in `(0, limit]`, balanced signs, and a deterministic 64-row evaluation table.

- [ ] **Step 5: Run tests and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_curriculum.py`

Expected: all tests pass.

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py Go2Pvcnn/tests/test_m1_panda_folded_load_curriculum.py
git commit -m "feat: add folded-load curriculum contracts"
```

### Task 2: Active-action mask in ActorCritic

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py`
- Create: `Go2Pvcnn/tests/test_rsl_rl_active_action_mask.py`

**Interfaces:**
- Consumes: policy config key `active_action_mask: list[int] | None`.
- Produces: registered boolean buffer `active_action_mask`, exact-zero inactive actions and means, masked `entropy`, and masked `get_actions_log_prob(actions)`.

- [ ] **Step 1: Write RED tests for probability and gradients**

```python
def test_inactive_actions_probability_and_gradients_are_zero():
    model = ActorCritic(5, 5, 4, actor_hidden_dims=[8], critic_hidden_dims=[8],
                        active_action_mask=[1, 1, 0, 0])
    obs = torch.randn(6, 5)
    actions = model.act(obs)
    assert actions[:, 2:].eq(0).all()
    assert model.action_mean[:, 2:].eq(0).all()
    loss = -(model.get_actions_log_prob(actions).mean() + .01 * model.entropy.mean())
    loss.backward()
    final = [m for m in model.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert final.weight.grad[2:].eq(0).all()
    assert final.bias.grad[2:].eq(0).all()
    assert model.std.grad[2:].eq(0).all()
```

Also test invalid mask length, a zero-active mask, checkpoint round-trip, and legacy construction without a mask.

- [ ] **Step 2: Run the test and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_rsl_rl_active_action_mask.py`

Expected: `ActorCritic.__init__()` rejects `active_action_mask`.

- [ ] **Step 3: Implement the mask at the distribution boundary**

```python
mask = torch.ones(num_actions, dtype=torch.bool) if active_action_mask is None else torch.as_tensor(active_action_mask, dtype=torch.bool)
if mask.shape != (num_actions,) or not bool(mask.any()):
    raise ValueError("active_action_mask must have num_actions entries and at least one active action")
self.register_buffer("active_action_mask", mask)

def update_distribution(self, observations):
    raw_mean = self.actor(observations)
    mean = raw_mean * self.active_action_mask.to(raw_mean.dtype)
    self.distribution = Normal(mean, mean * 0.0 + self.std)

def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample() * self.active_action_mask.to(observations.dtype)

def get_actions_log_prob(self, actions):
    terms = self.distribution.log_prob(actions)
    return terms[:, self.active_action_mask].sum(dim=-1)
```

Mask the entropy sum in the same way. During zero initialization, zero the last actor layer so inactive rows and active outputs start exactly zero.

- [ ] **Step 4: Run focused and legacy module tests**

Run: `cd Go2Pvcnn && pytest -q tests/test_rsl_rl_active_action_mask.py tests/test_m1_panda_coordinated_train_cfg.py`

Expected: all tests pass and legacy no-mask behavior remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py Go2Pvcnn/tests/test_rsl_rl_active_action_mask.py
git commit -m "feat: mask inactive PPO action dimensions"
```

### Task 3: Bounded PPO configuration and remaining-minibatch KL abort

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`
- Create: `Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py`
- Modify: `Go2Pvcnn/agent/__init__.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_ppo.py`

**Interfaces:**
- Consumes: `kl_abort_threshold: float | None` in algorithm config.
- Produces: `PPO.last_kl`, `PPO.last_kl_aborted`, `PPO.last_completed_mini_batches`, and `get_m1_panda_folded_load_train_cfg()`.

- [ ] **Step 1: Write exact configuration and abort tests**

```python
def test_folded_load_ppo_contract():
    cfg = get_m1_panda_folded_load_train_cfg()
    assert cfg["num_steps_per_env"] == 256
    assert cfg["algorithm"]["num_learning_epochs"] == 2
    assert cfg["algorithm"]["num_mini_batches"] == 4
    assert cfg["algorithm"]["learning_rate"] == 1e-5
    assert cfg["algorithm"]["min_learning_rate"] == 1e-6
    assert cfg["algorithm"]["max_learning_rate"] == 1e-4
    assert cfg["algorithm"]["desired_kl"] == .01
    assert cfg["algorithm"]["kl_abort_threshold"] == .015
    assert cfg["algorithm"]["max_grad_norm"] == .5
    assert cfg["policy"]["active_action_mask"] == [1] * 16 + [0] * 7
```

Use a synthetic storage generator whose second minibatch has KL `.02`; assert only the first optimizer step occurs and `last_kl_aborted is True`.

- [ ] **Step 2: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_ppo.py`

Expected: missing config getter and PPO abort fields.

- [ ] **Step 3: Implement safe abort semantics**

In `PPO.update()`, reset counters before the loops; after computing finite mean KL and before the optimizer step, abort all remaining minibatches when it exceeds the configured threshold:

```python
self.last_kl = float(kl_mean.item())
if self.kl_abort_threshold is not None and self.last_kl > self.kl_abort_threshold:
    self.last_kl_aborted = True
    abort_update = True
    break
```

Do not roll back already completed minibatches. Preserve adaptive LR for the next update and raise immediately on non-finite KL, loss, gradient norm, LR, or std.

- [ ] **Step 4: Implement the exact folded-load config**

Set `gamma=.9995`, `lam=.995`, 2 epochs, 4 minibatches, LR bounds, std `.005` with bounds `.005-.02`, mask `[1]*16+[0]*7`, zero actor initialization, and no optimizer resume flag for L0.

- [ ] **Step 5: Run tests and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_ppo.py tests/test_rsl_rl_active_action_mask.py`

Expected: all pass.

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py Go2Pvcnn/agent/__init__.py Go2Pvcnn/tests/test_m1_panda_folded_load_ppo.py
git commit -m "feat: add guarded folded-load PPO contract"
```

### Task 4: Folded-load MDP, reward, and task registration

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_folded_load.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/__init__.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_mdp.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_env_static.py`

**Interfaces:**
- Consumes: `env.folded_load_commands` shaped `[num_envs, 3]` and selected stage reset ranges.
- Produces: task `Isaac-M1-Panda-Folded-Load-v0`, 103 observations, 23 actions, termination-before-dt reward, and finite physical metrics.

- [ ] **Step 1: Write RED tests for reward formulas and static contracts**

```python
def test_tracking_rewards_have_exact_scales():
    vx_error = torch.tensor([0.0, .05])
    wz_error = torch.tensor([0.0, .15])
    torch.testing.assert_close(track_vx_error(vx_error), torch.exp(-vx_error.square() / .05**2))
    torch.testing.assert_close(track_wz_error(wz_error), torch.exp(-wz_error.square() / .15**2))

def test_action_cost_uses_only_first_16_coordinates():
    actions = torch.zeros(2, 23); actions[:, 16:] = 100
    assert active_action_l2(actions).eq(0).all()
```

Static tests assert task ID, `decimation=1`, `sim.dt=.005`, 103 observations, 23 actions, zero wrench terms, exact Panda fold defaults, and exact reward weights.

- [ ] **Step 2: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_mdp.py tests/test_m1_panda_folded_load_env_static.py`

Expected: new modules and task ID are absent.

- [ ] **Step 3: Implement pure tensor reward helpers**

Implement X tracking `exp(-(vx-vx_cmd)^2/.05^2)`, yaw tracking `exp(-(wz-wz_cmd)^2/.15^2)`, `vy^2`, height error to `.6115`, Z velocity, XY angular velocity, projected-gravity orientation, slide, first-16 action/action-rate L2, selected torque L2, and non-timeout termination.

- [ ] **Step 4: Build the isolated environment config**

Reuse `M1_PANDA_CFG` and compatible observation slots, but configure only these weights: `2, 1, -.5, 1, -12, -1, -.1, -2, -.1, -.02, -.01, -1e-5, -10000`. Ensure the termination value is supplied before manager reward multiplication by `.005`. Remove base-position, EE, learned-fold, and external-wrench reward/event terms.

- [ ] **Step 5: Register and verify**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_mdp.py tests/test_m1_panda_folded_load_env_static.py tests/test_m1_panda_coordinated_env_static.py`

Expected: all new and legacy static tests pass.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/mdp/m1_panda_folded_load.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/__init__.py Go2Pvcnn/tests/test_m1_panda_folded_load_mdp.py Go2Pvcnn/tests/test_m1_panda_folded_load_env_static.py
git commit -m "feat: add folded-load locomotion task"
```

### Task 5: Wrapper command lifecycle, real reset DR, and diagnostics

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_randomization.py`

**Interfaces:**
- Consumes: `StageSpec`, 23 policy actions, Isaac selected `env_ids`, articulation state, and contact sensors.
- Produces: exact-zero Panda action boundary, one command per episode, `episode_metrics` with command buckets and squared tracking errors, and `folded_load_diagnostics`.

- [ ] **Step 1: Write RED boundary and attribution tests**

```python
def test_wrapper_zeroes_arm_and_never_calls_wrench(mock_env):
    wrapper = M1PandaFoldedLoadEnvWrapper(mock_env, stage="L0-C0", seed=3)
    action = torch.ones(mock_env.num_envs, 23)
    wrapper.step(action)
    assert mock_env.last_action[:, 16:].eq(0).all()
    assert mock_env.wrench_call_count == 0

def test_command_changes_only_for_reset_env_ids(mock_env):
    wrapper = M1PandaFoldedLoadEnvWrapper(mock_env, stage="L1-C2", seed=3)
    before = wrapper.commands.clone()
    wrapper.reset_idx(torch.tensor([1, 4]))
    assert wrapper.commands[[0, 2, 3, 5]].equal(before[[0, 2, 3, 5]])
```

- [ ] **Step 2: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_wrapper.py tests/test_m1_panda_folded_load_randomization.py`

Expected: wrapper does not exist.

- [ ] **Step 3: Implement the action and command boundary**

Clone incoming actions, set `[:, 16:23]=0`, assert exact zero and finite before forwarding. Sample commands with a device-local seeded generator only at reset, copy them into the desired-twist observation slot, and maintain per-episode sums for `vx_error_sq`, `wz_error_sq`, absolute stationary velocities, timeout, base contact, and bad orientation.

- [ ] **Step 4: Implement selected-environment reset DR**

For D1-D3, sample root XY/RPY/linear/angular velocity, leg position, and friction from the exact `StageSpec.reset` ranges. Convert RPY perturbations to quaternions, leave root Z, wheels, Panda positions/velocities, and restitution unchanged, and write only `env_ids`. Assert finite/range/isolation properties before each write.

- [ ] **Step 5: Implement folded-load hard diagnostics**

Expose maximum arm fold error, effort utilization, joint-limit proximity, mount wrench norm, inactive action maximum, and finite-state flags. Define hard fold failure with explicit asset limits (joint-limit margin `<=.01 rad`, effort utilization `>1.0`, fold error `>.35 rad`, or non-finite state), with values serialized in each episode record.

- [ ] **Step 6: Verify and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_wrapper.py tests/test_m1_panda_folded_load_randomization.py`

Expected: all pass, including D1/D2/D3 exact range and selected-env isolation tests.

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py Go2Pvcnn/tests/test_m1_panda_folded_load_wrapper.py Go2Pvcnn/tests/test_m1_panda_folded_load_randomization.py
git commit -m "feat: add folded-load wrapper and reset DR"
```

### Task 6: Always-on guard, eligibility, and atomic artifacts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py`

**Interfaces:**
- Consumes: completed episode records, PPO diagnostics, checkpoint path/SHA, stage spec, and iteration.
- Produces: `GuardDecision(stop, eligible, reason, rank)`, atomic manifests, eligible-best metadata, and evaluation decisions.

- [ ] **Step 1: Write RED catastrophe tests independent of eligibility**

```python
def test_catastrophe_stops_before_any_eligible_best(tmp_path):
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"), tmp_path)
    for iteration in (1, 2):
        decision = guard.update(iteration, episodes_with_hard_failure_rate(.51), finite=True)
    assert decision.stop and decision.reason == "hard_failure_rate_gt_0.50_for_2_updates"
    assert guard.eligible_best is None

def test_nonfinite_and_mask_leak_stop_immediately(tmp_path):
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"), tmp_path)
    assert guard.update(1, [], finite=False).stop
    assert guard.update(2, [], finite=True, inactive_action_max=1e-9).stop
```

- [ ] **Step 2: Add exact rolling-window and per-bucket tests**

Create 200/400-record fixtures and assert timeout `>=.95`, contact/orientation `<=.02`, VX RMSE `<=.04`, yaw RMSE `<=.12`, at least 25 forward/reverse/left/right records, per-bucket gates, and stationary `|vx|<=.03`, `|wz|<=.08`. Assert boundary equality passes and one-epsilon violations fail.

- [ ] **Step 3: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_training_guard.py`

Expected: guard module does not exist.

- [ ] **Step 4: Implement rank, stop counters, and atomic writes**

Use temporary sibling files plus `os.replace()` for JSON and checkpoints. Rank eligible candidates lexicographically by lower hard-failure rate, lower normalized tracking error, then higher timeout rate. Stop for any nonfinite/mask/fold failure immediately, `>.50` for 2 updates, `>.20` for 5, 50 updates without eligible-rank improvement, or 600 total updates.

- [ ] **Step 5: Implement acceptance semantics**

Write `model_best.pt` only for eligible training candidates. Set `accepted=true` only after seed 42/43/44 aggregate evaluation passes. Make `model_final.pt` SHA-identical to the accepted source. A failed run writes diagnostics and `accepted=false` but never a promotion checkpoint.

- [ ] **Step 6: Verify and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_training_guard.py`

Expected: all guard, atomicity, SHA, and rejected-promotion tests pass.

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py
git commit -m "feat: guard folded-load stage promotion"
```

### Task 7: One-stage train and fixed three-seed evaluation entrypoints

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_folded_load_train.py`
- Create: `Go2Pvcnn/scripts/m1_panda_folded_load_eval.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`

**Interfaces:**
- Consumes: `--stage`, an empty `--run_dir`, and for non-L0 an accepted parent manifest/checkpoint.
- Produces: one guarded training run, TensorBoard stability scalars, three fixed evaluation reports, and aggregate acceptance.

- [ ] **Step 1: Write RED CLI and lineage tests**

```python
def test_l0_rejects_resume_and_non_l0_requires_accepted_parent(tmp_path):
    assert validate_parent("L0-C0", None) is None
    with pytest.raises(ValueError, match="fresh"):
        validate_parent("L0-C0", tmp_path / "old.pt")
    rejected = write_manifest(tmp_path, accepted=False)
    with pytest.raises(ValueError, match="accepted=true"):
        validate_parent("L1-C1", rejected)
```

Static tests require task ID, GPU argument, run-directory refusal, atomic manifest, 64 eval envs, seeds `(42,43,44)`, balanced fixed commands, and no wrench symbols.

- [ ] **Step 2: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_scripts.py`

Expected: scripts do not exist.

- [ ] **Step 3: Implement one-stage training**

Launch Isaac before task imports, validate a new empty run directory, build the stage-specific env/wrapper/config, load only actor/critic from an accepted parent, reset optimizer, clamp inherited std to `.01`, register the guard callback, and save PID/config/active mask/asset SHA/parent SHA in `run_manifest.json`.

- [ ] **Step 4: Log stability diagnostics**

Extend `OnPolicyRunner` iteration summaries without changing legacy defaults. Log finite scalars for LR, mean/max KL, KL abort, completed minibatches, active std min/max, grad norm, inactive-action max, fold error, effort utilization, and hard-failure rates.

- [ ] **Step 5: Implement fixed evaluation**

For each seed 42/43/44, run 64 environments for one full episode using `balanced_eval_commands`, deterministic policy means, the same physical gates, and atomic `evaluation_seed_<seed>.json`. Aggregate only when all three pass, then call the guard's acceptance finalizer.

- [ ] **Step 6: Verify and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_scripts.py tests/test_m1_panda_folded_load_training_guard.py`

Expected: all pass.

```bash
git add Go2Pvcnn/scripts/m1_panda_folded_load_train.py Go2Pvcnn/scripts/m1_panda_folded_load_eval.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py
git commit -m "feat: add guarded folded-load train and eval"
```

### Task 8: Atomic stage orchestrator and rollback boundary

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_folded_load_curriculum.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_orchestrator.py`

**Interfaces:**
- Consumes: experiment root, current stage, train/eval scripts, and accepted manifests.
- Produces: sequential stage processes, verified SHA lineage, atomic promotion state, and stop-on-failure rollback pointer.

- [ ] **Step 1: Write RED orchestration tests**

```python
def test_rejected_stage_stops_and_keeps_previous_accepted_checkpoint(tmp_path, fake_runner):
    state = run_curriculum(tmp_path, start="L1-C2", runner=fake_runner.reject("L1-C2"))
    assert state.stopped_stage == "L1-C2"
    assert state.rollback_stage == "L1-C1"
    assert not (tmp_path / "L1-C3").exists()

def test_parent_sha_must_match_previous_final(tmp_path):
    with pytest.raises(ValueError, match="SHA"):
        validate_lineage(tampered_stage_manifests(tmp_path))
```

- [ ] **Step 2: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_orchestrator.py`

Expected: orchestrator does not exist.

- [ ] **Step 3: Implement explicit process sequencing**

Run only the current stage. On eligible training, invoke all three evaluations. Promote only an accepted manifest whose `model_final.pt` and parent SHAs verify. If train/eval fails, atomically write orchestration state pointing to the prior accepted stage and exit nonzero; never reduce difficulty or continue.

- [ ] **Step 4: Verify and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_orchestrator.py tests/test_m1_panda_folded_load_scripts.py`

Expected: all pass.

```bash
git add Go2Pvcnn/scripts/m1_panda_folded_load_curriculum.py Go2Pvcnn/tests/test_m1_panda_folded_load_orchestrator.py
git commit -m "feat: orchestrate folded-load curriculum stages"
```

### Task 9: Real GPU probe, smoke ladder, and runbook

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_folded_load_probe.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_probe_static.py`
- Create: `docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md`

**Interfaces:**
- Consumes: GPU device and folded-load task.
- Produces: atomic probe JSON and exact commands for probe, smoke, current-stage long train, evaluation, and orchestration.

- [ ] **Step 1: Write RED probe static tests**

Require 8 environments, zero actions, no wrench API, at least one physics step, exact inactive-action assertion, fold error, effort limits, mount response, finite-state checks, and nonzero process exit when any check fails.

- [ ] **Step 2: Run and confirm RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_probe_static.py`

Expected: probe script does not exist.

- [ ] **Step 3: Implement probe and runbook**

The runbook must contain these commands, always on GPU 0:

```bash
cd /home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/m1_panda_folded_load_probe.py --num_envs 8 --device cuda:0
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/m1_panda_folded_load_train.py --stage L0-C0 --num_envs 8 --max_iterations 1 --device cuda:0 --run_dir logs/m1_panda_folded_load/smoke-8x1
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/m1_panda_folded_load_train.py --stage L0-C0 --num_envs 64 --max_iterations 10 --device cuda:0 --run_dir logs/m1_panda_folded_load/smoke-64
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/m1_panda_folded_load_curriculum.py --start_stage L0-C0 --num_envs 4096 --device cuda:0 --experiment_root logs/m1_panda_folded_load/foundation-v1
```

State that smoke runs cannot be accepted as locomotion stages.

- [ ] **Step 4: Run the complete CPU suite**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_*.py tests/test_rsl_rl_active_action_mask.py`

Expected: all pass.

- [ ] **Step 5: Run GPU verification in order**

Run the probe, 8x1 smoke, then 64x10 smoke from the runbook. Expected: exit code 0, finite diagnostics, exact-zero inactive actions, no wrench calls, valid atomic manifests, and no acceptance marker for smoke directories.

- [ ] **Step 6: Commit verified operational files**

```bash
git add Go2Pvcnn/scripts/m1_panda_folded_load_probe.py Go2Pvcnn/tests/test_m1_panda_folded_load_probe_static.py docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md
git commit -m "docs: add folded-load verification runbook"
```

### Task 10: Start the guarded L0-C0 long run

**Files:**
- Runtime artifact: `Go2Pvcnn/logs/m1_panda_folded_load/foundation-v1/L0-C0/`
- Modify after launch: `docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md`

**Interfaces:**
- Consumes: the fully verified code and empty experiment root.
- Produces: a durable GPU-0 process, PID/manifest, and monitoring commands; later stages are started only by the orchestrator after acceptance.

- [ ] **Step 1: Verify launch preconditions**

Run: `git status --short && test ! -e Go2Pvcnn/logs/m1_panda_folded_load/foundation-v1/L0-C0`

Expected: only intentional documentation/runtime changes are shown and the L0-C0 directory is absent.

- [ ] **Step 2: Start the long orchestrated run**

Run the GPU-0 curriculum command from Task 9 under the repository's existing durable-session mechanism. Do not pass a resume checkpoint.

- [ ] **Step 3: Verify process and first update**

Check the recorded PID, `run_manifest.json`, TensorBoard event, and first numeric checkpoint/diagnostic. Expected: stage `L0-C0`, `accepted=false`, parent SHA null, active mask `16/23`, finite PPO scalars, and zero inactive actions.

- [ ] **Step 4: Record the exact monitor and stop commands**

Add the process ID, log path, TensorBoard command, manifest inspection command, and graceful-stop command to the runbook. Do not claim training completion until a three-seed accepted manifest exists.

- [ ] **Step 5: Commit only the runbook update**

```bash
git add docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md
git commit -m "docs: record folded-load long-run launch"
```

## Final Verification Gate

- Run `cd Go2Pvcnn && pytest -q tests/test_m1_panda_folded_load_*.py tests/test_rsl_rl_active_action_mask.py` and require zero failures.
- Run legacy coordinated static/config tests and require no regression.
- Run `git diff --check` and inspect every changed file.
- Confirm the GPU probe, 8x1 smoke, and 64x10 smoke artifacts are finite and rejected as acceptance evidence.
- Confirm L0 starts fresh and every later manifest parent SHA matches the immediately preceding accepted `model_final.pt`.
- Completion is only an accepted L2-D3 aggregate evaluation for seeds 42, 43, and 44; no claim is made about Panda motion, wrench, grasping, Student transfer, or hardware.
