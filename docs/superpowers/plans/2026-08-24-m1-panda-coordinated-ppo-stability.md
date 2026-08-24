# M1 + Panda Coordinated PPO Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh-from-zero, 200 Hz coordinated M1 + Panda PPO training path that survives late-training instability through longer rollouts, bounded adaptive optimization, real Panda-hand wrench/reset domain randomization, and automatic best-checkpoint rollback.

**Architecture:** Keep the accepted 103-observation/23-action single-articulation task intact. Add a dedicated coordinated PPO config, a generic immutable runner iteration callback, a pure M1/Panda training guard plus atomic checkpoint controller, and training-only domain-randomization wiring around the existing environment; keep default Play and every unrelated runner caller unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.7, Isaac Sim/Isaac Lab 5.1, Gymnasium, vendored RSL-RL PPO, TensorBoard, pytest, GPU0.

## Global Constraints

- Execute with one agent using `superpowers:executing-plans`; the user explicitly prohibited subagents.
- Work only in `/home/xk/coding/M1`; never stage or overwrite the pre-existing `graphify-out/` changes or `Go2Pvcnn/assets/m1_panda/m1.zip`.
- Preserve Gym ID `Isaac-M1-Panda-Coordinated-v0`, exact observation width `103`, action width/order `23 = 12 legs + 4 wheels + 7 Panda`, `sim.dt=0.005`, `decimation=1`, and asset SHA-256 `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`.
- New production runs are fresh policies: never load old actor, critic, optimizer, or `model_3500.pt`; the A1 checkpoint remains provenance-only.
- Training defaults are exactly `256 / gamma 0.9995 / lambda 0.995`, adaptive KL target `0.01`, LR `[1e-6, 3e-4]` starting at `1e-4`, and physical action std `[0.005, 0.05]` starting at `0.01`.
- Production guard defaults are 100 completed episodes, eligible gates `timeout>=0.90`, `base_contact<=0.05`, `bad_orientation<=0.05`, catastrophe `hard_failure>0.20` for 25 updates, patience 50 updates, maximum 600 updates.
- Training wrench acts only on `panda_hand`, reaches `20 N/5 Nm`, starts at scale `0.10`, reaches full scale in 50,000 per-environment 200 Hz steps, and is disabled by default outside the training entrypoint.
- Reset DR uses root x/y `±0.02 m`, roll/pitch `±0.03 rad`, yaw `±0.05 rad`, linear velocity `±0.05 m/s`, angular velocity `±0.10 rad/s`, leg position `±0.02 rad`, Panda position `±0.03 rad`, controlled velocity `±0.05 rad/s`, friction `[0.8,1.2]`, restitution `0`.
- Every behavior change follows RED → minimal GREEN → focused regression → commit. Every distinct verification pass gets a `notes/log/` record before completion.

---

### Task 1: Dedicated 200 Hz coordinated PPO configuration

**Files:**
- Create: `Go2Pvcnn/agent/m1_panda_coordinated_train_cfg.py`
- Modify: `Go2Pvcnn/agent/__init__.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_train_cfg.py`

**Interfaces:**
- Produces: `get_m1_panda_coordinated_train_cfg() -> dict` returning a new dictionary on every call.
- Consumed by: Task 7 training entrypoint; no A0/A1 configuration changes.

- [x] **Step 1: Write the failing exact-contract tests**

```python
from agent import get_m1_panda_coordinated_train_cfg


def test_coordinated_cfg_freezes_200_hz_time_horizon_and_adaptive_ppo():
    cfg = get_m1_panda_coordinated_train_cfg()
    assert cfg["num_steps_per_env"] == 256
    assert cfg["save_interval"] == 25
    assert cfg["algorithm"] == {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-4,
        "min_learning_rate": 1.0e-6,
        "max_learning_rate": 3.0e-4,
        "clip_param": 0.2,
        "gamma": 0.9995,
        "lam": 0.995,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.0,
        "clip_min_std": 0.005,
        "clip_max_std": 0.05,
        "max_grad_norm": 1.0,
        "use_clipped_value_loss": True,
        "schedule": "adaptive",
        "desired_kl": 0.01,
    }
    assert cfg["policy"]["init_noise_std"] == 0.01
    assert cfg["policy"]["noise_std_type"] == "scalar"


def test_coordinated_cfg_returns_independent_objects():
    left = get_m1_panda_coordinated_train_cfg()
    left["algorithm"]["gamma"] = 0.0
    assert get_m1_panda_coordinated_train_cfg()["algorithm"]["gamma"] == 0.9995
```

- [x] **Step 2: Run RED**

Run:
```bash
cd /home/xk/coding/M1
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_coordinated_train_cfg.py -q
```
Expected: FAIL because the module/export does not exist.

- [x] **Step 3: Implement the dedicated config**

```python
def get_m1_panda_coordinated_train_cfg() -> dict:
    return {
        "num_steps_per_env": 256,
        "save_interval": 25,
        "empirical_normalization": False,
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": 5,
            "num_mini_batches": 4, "learning_rate": 1.0e-4,
            "min_learning_rate": 1.0e-6, "max_learning_rate": 3.0e-4,
            "clip_param": 0.2, "gamma": 0.9995, "lam": 0.995,
            "value_loss_coef": 1.0, "entropy_coef": 0.0,
            "clip_min_std": 0.005, "clip_max_std": 0.05,
            "max_grad_norm": 1.0, "use_clipped_value_loss": True,
            "schedule": "adaptive", "desired_kl": 0.01,
        },
        "policy": {
            "class_name": "ActorCritic", "init_noise_std": 0.01,
            "noise_std_type": "scalar", "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128], "activation": "elu",
        },
    }
```

- [x] **Step 4: Run GREEN and compile**

Run the Step 2 command, then:
```bash
/home/xk/miniconda3/envs/go2/bin/python -m py_compile Go2Pvcnn/agent/m1_panda_coordinated_train_cfg.py Go2Pvcnn/agent/__init__.py
```
Expected: all tests pass; compile exit `0`.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/agent/m1_panda_coordinated_train_cfg.py Go2Pvcnn/agent/__init__.py Go2Pvcnn/tests/test_m1_panda_coordinated_train_cfg.py
git commit -m "feat: add stable coordinated PPO config"
```

### Task 2: Bounded adaptive KL/LR and physical std diagnostics

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Create: `Go2Pvcnn/tests/test_rsl_ppo_adaptive_schedule.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py`

**Interfaces:**
- `PPO._adapt_learning_rate(kl_mean: float) -> str` returns `"decrease"`, `"increase"`, or `"hold"`.
- `PPO.last_kl_mean: float` and `PPO.last_lr_adjustment: str` are generic diagnostics.
- `clip_min_std` and new `clip_max_std` are interpreted in physical standard-deviation units.

- [x] **Step 1: Write failing unit tests without rollout simulation**

```python
def test_adaptive_lr_obeys_configured_bounds(tiny_ppo):
    tiny_ppo.learning_rate = 1.0e-6
    assert tiny_ppo._adapt_learning_rate(0.03) == "hold"
    assert tiny_ppo.learning_rate == pytest.approx(1.0e-6)
    tiny_ppo.learning_rate = 3.0e-4
    assert tiny_ppo._adapt_learning_rate(0.001) == "hold"
    assert tiny_ppo.learning_rate == pytest.approx(3.0e-4)


def test_adaptive_lr_moves_toward_desired_kl(tiny_ppo):
    assert tiny_ppo._adapt_learning_rate(0.03) == "decrease"
    assert tiny_ppo.learning_rate == pytest.approx(1.0e-4 / 1.5)
    assert tiny_ppo._adapt_learning_rate(0.001) == "increase"


def test_policy_std_is_clamped_in_physical_units(tiny_ppo):
    tiny_ppo.actor_critic.std.data[:] = torch.tensor([0.001, 0.2])
    tiny_ppo._clamp_policy_std()
    assert torch.equal(tiny_ppo.actor_critic.std, torch.tensor([0.005, 0.05]))
```

The `tiny_ppo` fixture constructs a two-action scalar-mode `ActorCritic`, then passes it to `PPO` with `desired_kl=0.01`, `learning_rate=1e-4`, `min_learning_rate=1e-6`, `max_learning_rate=3e-4`, `clip_min_std=0.005`, and `clip_max_std=0.05`.

- [x] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_rsl_ppo_adaptive_schedule.py Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py -q
```
Expected: FAIL on missing LR bounds, max std, and diagnostics.

- [x] **Step 3: Implement bounded adaptation and diagnostics**

Add constructor fields and helpers:
```python
self.min_learning_rate = float(min_learning_rate)
self.max_learning_rate = float(max_learning_rate)
self.clip_max_std = clip_max_std
self.last_kl_mean = 0.0
self.last_lr_adjustment = "hold"

def _adapt_learning_rate(self, kl_mean: float) -> str:
    old = self.learning_rate
    if kl_mean > self.desired_kl * 2.0:
        self.learning_rate = max(self.min_learning_rate, old / 1.5)
    elif 0.0 < kl_mean < self.desired_kl / 2.0:
        self.learning_rate = min(self.max_learning_rate, old * 1.5)
    adjustment = "decrease" if self.learning_rate < old else "increase" if self.learning_rate > old else "hold"
    for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
    self.last_kl_mean = float(kl_mean)
    self.last_lr_adjustment = adjustment
    return adjustment

def _clamp_policy_std(self) -> None:
    if hasattr(self.actor_critic, "clip_std"):
        self.actor_critic.clip_std(min=self.clip_min_std, max=self.clip_max_std)
```

Replace the hard-coded `1e-5/1e-2` branch inside `update()` with `_adapt_learning_rate(float(kl_mean))`, call `_clamp_policy_std()` after optimizer updates, and log `Loss/kl` plus an integer `Loss/lr_adjustment` (`-1/0/+1`) in the runner.

- [x] **Step 4: Run GREEN and generic runner regression**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_rsl_ppo_adaptive_schedule.py Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py -q
```
Expected: all pass; legacy scalar/log std and checkpoint behavior remain green.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/algorithms/ppo.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/test_rsl_ppo_adaptive_schedule.py Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py
git commit -m "feat: bound adaptive PPO optimization"
```

### Task 3: Generic immutable runner iteration callback

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Create: `Go2Pvcnn/tests/test_rsl_runner_iteration_callback.py`

**Interfaces:**
- Produces frozen `IterationSummary(iteration, timesteps, completed_rewards, episode_metrics, learning_rate, kl_mean, environment_metrics)`.
- `OnPolicyRunner.learn(num_learning_iterations, init_at_random_ep_len=False, iteration_callback=None) -> LearnResult`; callback returns `None` to continue or a non-empty stop-reason string to stop.
- Optional `env.get_training_diagnostics() -> Mapping[str, float]` is copied to immutable scalar pairs; absent method yields an empty mapping.

- [ ] **Step 1: Write failing pure helper and source-compatibility tests**

```python
def test_freeze_episode_metrics_detaches_and_rejects_nonfinite():
    frozen = freeze_episode_metrics([{"Termination/time_out": torch.tensor([1.0, 0.0])}])
    assert frozen == (("Termination/time_out", (1.0, 0.0)),)
    with pytest.raises(ValueError, match="finite"):
        freeze_episode_metrics([{"Reward/base_target": torch.tensor([float("nan")])}])


def test_iteration_summary_is_frozen():
    summary = IterationSummary(3, 4096, (1.0,), (), 1e-4, 0.01, ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.iteration = 4


def test_callback_is_optional_and_default_save_loop_is_preserved():
    source = Path(on_policy_runner.__file__).read_text()
    assert "iteration_callback=None" in source
    assert "if it % self.save_interval == 0" in source
```

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_rsl_runner_iteration_callback.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py -q
```
Expected: FAIL because summary/result/callback APIs do not exist.

- [ ] **Step 3: Add the minimal generic API**

```python
@dataclass(frozen=True)
class IterationSummary:
    iteration: int
    timesteps: int
    completed_rewards: Sequence[float]
    episode_metrics: Sequence[tuple[str, Sequence[float]]]
    learning_rate: float
    kl_mean: float
    environment_metrics: Sequence[tuple[str, float]]

@dataclass(frozen=True)
class LearnResult:
    completed_iterations: int
    stop_reason: str

def freeze_episode_metrics(ep_infos) -> Sequence[tuple[str, Sequence[float]]]:
    values: dict[str, list[float]] = {}
    for info in ep_infos:
        for key, raw in info.items():
            tensor = torch.as_tensor(raw).detach().reshape(-1).cpu()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"episode metric {key!r} must be finite")
            values.setdefault(key, []).extend(map(float, tensor.tolist()))
    return tuple((key, tuple(values[key])) for key in sorted(values))
```

Accumulate `completed_rewards_this_iteration` at each `dones` event; after log/periodic-save construct the summary, invoke the callback, break on its returned reason, always save the last ordinary checkpoint, and return `LearnResult`. Existing callers that ignore the return value remain valid.

When a writer exists, add `Loss/kl`, `Loss/lr_adjustment`, and every finite environment diagnostic under `DomainRandomization/<key>` at the current iteration. Do not introduce M1/Panda metric names into the generic runner.

- [ ] **Step 4: Run GREEN plus representative runner callers**

Run the Step 2 command and static tests for teacher/coordinated train entrypoints. Expected: all pass and no caller requires a callback.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/test_rsl_runner_iteration_callback.py
git commit -m "feat: expose runner iteration summaries"
```

### Task 4: Pure best-checkpoint guard and atomic rollback controller

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_training_guard.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_training_guard.py`

**Interfaces:**
- `GuardSnapshot` stores the 100-episode rolling metrics and lexicographic rank.
- `TrainingGuard.observe(summary: IterationSummary) -> GuardDecision` is pure with respect to disk.
- `AtomicCheckpointController.on_iteration(runner, summary) -> str | None` performs atomic `model_best.pt`/JSON writes and returns the stop reason.
- `AtomicCheckpointController.finalize(runner, stop_reason) -> dict[str, object]` reloads eligible best when present, otherwise diagnostic best, using `load_optimizer=False, keep_std=True`; it atomically writes `model_final.pt` and returns manifest fields.

- [ ] **Step 1: Write the failing guard matrix**

Use this complete helper and matrix; each call supplies 100 values per required metric:
```python
def summary(it, timeout=1.0, contact=0.0, orientation=0.0, base=2.0, ee=1.0, reward=100.0, count=100):
    metrics = {
        "Termination/time_out": (timeout,) * count,
        "Termination/base_contact": (contact,) * count,
        "Termination/bad_orientation": (orientation,) * count,
        "Reward/base_target": (base,) * count,
        "Reward/ee_tracking": (ee,) * count,
    }
    return IterationSummary(it, it * 16384, (reward,) * count, tuple(metrics.items()), 1e-4, 0.01, ())


def test_guard_waits_for_100_completed_episodes():
    decision = TrainingGuard().observe(summary(1, count=99))
    assert decision.snapshot is None and not decision.save_best


def test_rank_minimizes_hard_failure_before_task_score_and_keeps_earlier_ties():
    guard = TrainingGuard()
    first = guard.observe(summary(1, contact=0.01, base=2.0))
    worse_safety = guard.observe(summary(2, contact=0.02, base=100.0))
    equal = guard.observe(summary(3, contact=0.01, base=2.0))
    assert first.save_best
    assert not worse_safety.save_best and not equal.save_best
    assert guard.eligible_best.iteration == 1


def test_nonfinite_metric_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        TrainingGuard().observe(summary(1, reward=float("nan")))


def test_ineligible_diagnostic_best_is_not_accepted():
    guard = TrainingGuard()
    guard.observe(summary(1, timeout=0.5, contact=0.1))
    assert guard.diagnostic_best is not None
    assert guard.eligible_best is None and not guard.accepted


def test_patience_catastrophe_recovery_and_cap():
    patience = TrainingGuard()
    patience.observe(summary(1))
    assert [patience.observe(summary(it)).stop_reason for it in range(2, 52)][-1] == "eligible_patience"
    catastrophe = TrainingGuard()
    catastrophe.observe(summary(1))
    for it in range(2, 26):
        assert catastrophe.observe(summary(it, contact=0.21)).stop_reason is None
    catastrophe.observe(summary(26, contact=0.0))
    assert catastrophe.catastrophe_updates == 0
    assert [catastrophe.observe(summary(it, contact=0.21)).stop_reason for it in range(27, 52)][-1] == "catastrophe"
    assert TrainingGuard(max_iterations=1).observe(summary(1)).stop_reason == "max_iterations"


def test_atomic_controller_hashes_best_and_rolls_back_final(tmp_path, fake_runner):
    controller = AtomicCheckpointController(tmp_path, TrainingGuard(max_iterations=1))
    assert controller.on_iteration(fake_runner, summary(1)) == "max_iterations"
    fields = controller.finalize(fake_runner, "max_iterations")
    assert fields["accepted"] is True
    assert Path(fields["final_checkpoint"]).name == "model_final.pt"
    assert fields["rollback_source_sha256"] == sha256_file(tmp_path / "model_best.pt")
    assert fields["final_checkpoint_sha256"] == sha256_file(tmp_path / "model_final.pt")
```

The `fake_runner` fixture owns a one-parameter `torch.nn.Linear`, implements `save(path)` with `torch.save({"model_state_dict": module.state_dict(), "iter": current_iteration}, path)`, and implements `load(path, load_optimizer, keep_std)` by restoring that state and recording both flags. This makes the rollback assertion exercise real checkpoint bytes rather than mocked hashes.

Use exact metric keys `Termination/time_out`, `Termination/base_contact`, `Termination/bad_orientation`, `Reward/base_target`, and `Reward/ee_tracking`; construct summaries with 100 scalar samples per key so eligibility is unambiguous.

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_coordinated_training_guard.py -q
```
Expected: FAIL because guard/controller do not exist.

- [ ] **Step 3: Implement exact ranking and stop state**

```python
rank = (
    snapshot.base_contact_rate + snapshot.bad_orientation_rate,
    -snapshot.time_out_rate,
    -(snapshot.base_target + snapshot.ee_tracking),
    -snapshot.mean_reward,
    snapshot.iteration,
)
eligible = (
    snapshot.completed_episodes >= 100
    and snapshot.time_out_rate >= 0.90
    and snapshot.base_contact_rate <= 0.05
    and snapshot.bad_orientation_rate <= 0.05
)
```

Maintain separate `diagnostic_best` and `eligible_best`; once eligible exists, only a better eligible candidate may replace `model_best.pt`. Evaluate catastrophe and patience after observing/saving the current update. Validate every scalar with `math.isfinite` before ranking.

Maintain one `deque(maxlen=100)` for completed rewards and for each required episode metric. A candidate is formed only when all six deques contain 100 samples; its rates/scores are means over those same most-recent 100 completed episodes. An update with no completed episode re-evaluates the unchanged window only for patience/catastrophe counting and cannot manufacture a new best.

- [ ] **Step 4: Implement atomic checkpoint files**

Use same-directory temporary paths and `os.replace`; write JSON with `allow_nan=False`. `best_checkpoint.json` contains iteration, timesteps, metric fields, LR, KL, DR diagnostics, SHA and `eligible`. `finalize()` uses eligible best if one exists, otherwise diagnostic best; only the former yields `accepted=true`. It returns `status`, `stop_reason`, `best_iteration`, `rollback_source_sha256`, `final_checkpoint_sha256`, and `accepted`.

- [ ] **Step 5: Run GREEN and compile**

Run the Step 2 command plus:
```bash
/home/xk/miniconda3/envs/go2/bin/python -m py_compile Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_training_guard.py
```
Expected: all pass; compile exit `0`.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_training_guard.py Go2Pvcnn/tests/test_m1_panda_coordinated_training_guard.py
git commit -m "feat: guard coordinated PPO checkpoints"
```

### Task 5: Training-only reset and friction domain randomization

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/events.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_domain_randomization.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_coordinated_learning_env_static.py`

**Interfaces:**
- `reset_coordinated_joints_by_offset(env, env_ids, leg_position_range, arm_position_range, velocity_range, asset_cfg)` performs one atomic reset: legs and Panda receive their separate additive ranges, wheels/other joints remain at defaults, and all values are clamped to soft limits.
- `configure_coordinated_training_domain_randomization(cfg, enabled: bool) -> None` enables exact approved ranges for train and restores deterministic ranges for Play/probes.
- Environment defaults remain deterministic until the train entrypoint explicitly calls the helper.

- [ ] **Step 1: Write RED tests for selected-joint isolation and exact ranges**

```python
def test_coordinated_joint_reset_uses_separate_ranges_and_keeps_wheels_default(fake_env):
    reset_coordinated_joints_by_offset(
        fake_env, torch.tensor([0]), (-0.02, 0.02), (-0.03, 0.03),
        (-0.05, 0.05), SceneEntityCfg("robot"),
    )
    written = fake_env.robot.last_written_state
    assert torch.all(written.leg_offset.abs() <= 0.02)
    assert torch.all(written.arm_offset.abs() <= 0.03)
    assert torch.equal(written.wheel_position, fake_env.robot.default_wheel_position)
    assert torch.all(written.controlled_velocity.abs() <= 0.05)


def test_training_dr_helper_sets_exact_ranges():
    cfg = M1PandaCoordinatedEnvCfg()
    configure_coordinated_training_domain_randomization(cfg, True)
    assert cfg.events.reset_base.params["pose_range"]["x"] == (-0.02, 0.02)
    assert cfg.events.reset_robot_joints.params["leg_position_range"] == (-0.02, 0.02)
    assert cfg.events.reset_robot_joints.params["arm_position_range"] == (-0.03, 0.03)
    assert cfg.events.physics_material.params["static_friction_range"] == (0.8, 1.2)
```

Also assert roll/pitch/yaw, all six velocity ranges, restitution `(0.0, 0.0)`, 64 friction buckets, and `configure_coordinated_training_domain_randomization(cfg, False)` restores all-zero/default reset ranges.

- [ ] **Step 2: Run RED**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_coordinated_domain_randomization.py Go2Pvcnn/tests/test_m1_panda_coordinated_learning_env_static.py -q
```
Expected: FAIL on missing event/helper.

- [ ] **Step 3: Implement the atomic coordinated-joint event**

Resolve exact leg, wheel, and Panda IDs from the canonical name constants, clone the full default state once, sample the two additive position ranges plus the controlled velocity range, clamp against soft position/velocity limits, then call exactly one:
```python
asset.write_joint_state_to_sim(
    joint_pos, joint_vel, env_ids=env_ids
)
```
Reject missing/duplicate canonical joints and non-finite ranges before writing.

- [ ] **Step 4: Add deterministic event defaults and the explicit training helper**

Define `M1PandaCoordinatedEventsCfg` with one overridden `reset_robot_joints` event using the atomic helper. The helper mutates only the coordinated cfg instance and sets every approved range; it must not change module-level shared config objects or other tasks.

- [ ] **Step 5: Run GREEN, compile, and coordinated cfg regression**

Run the Step 2 command, existing coordinated env tests, and `py_compile` for both modified modules. Expected: all pass; default cfg remains deterministic.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/mdp/events.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py Go2Pvcnn/tests/test_m1_panda_coordinated_domain_randomization.py Go2Pvcnn/tests/test_m1_panda_coordinated_learning_env_static.py
git commit -m "feat: randomize coordinated reset state"
```

### Task 6: Seeded Panda-hand wrench curriculum and wrapper diagnostics

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_disturbance.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_disturbance.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_wrapper_disturbance.py`

**Interfaces:**
- `CoordinatedDisturbanceCfg()` freezes `20 N/5 Nm`, `0.25–1.0 s`, scale `0.10→1.0/50_000`, probabilities `0.50/0.30/0.20`, pulse duty `0.20`.
- `CoordinatedDisturbanceScheduler.advance() -> Tensor[num_envs,6]`; reset is selective and never rewinds curriculum.
- Wrapper constructor gains keyword-only `training_randomization: bool = False, seed: int = 0`.
- `wrapper.get_training_diagnostics() -> dict[str, float]` exposes curriculum scale, force/torque norms, nonzero ratio, and observed root/joint reset deviation extrema.

- [ ] **Step 1: Write scheduler RED tests**

```python
def fixed_mode_scheduler(mode, duration_steps, seed):
    probabilities = {
        CONTINUOUS_MODE: (1.0, 0.0, 0.0),
        PULSE_MODE: (0.0, 1.0, 0.0),
        INTERMITTENT_MODE: (0.0, 0.0, 1.0),
    }[mode]
    cfg = replace(
        CoordinatedDisturbanceCfg(),
        hold_time_min_s=duration_steps * 0.005,
        hold_time_max_s=duration_steps * 0.005,
        mode_probabilities=probabilities,
    )
    return CoordinatedDisturbanceScheduler(cfg, 1, "cpu", 0.005, seed=seed)


def test_same_seed_reproduces_and_envs_are_independent():
    left = CoordinatedDisturbanceScheduler(CoordinatedDisturbanceCfg(), 8, "cpu", 0.005, seed=7)
    right = CoordinatedDisturbanceScheduler(CoordinatedDisturbanceCfg(), 8, "cpu", 0.005, seed=7)
    wrench = left.advance()
    assert torch.equal(wrench, right.advance())
    assert torch.unique(wrench, dim=0).shape[0] > 1
    assert torch.all(wrench[:, :3].abs() <= 2.0 + 1e-6)
    assert torch.all(wrench[:, 3:].abs() <= 0.5 + 1e-6)


@pytest.mark.parametrize(
    ("mode", "expected_nonzero_steps"),
    [(CONTINUOUS_MODE, 200), (PULSE_MODE, 40), (INTERMITTENT_MODE, 40)],
)
def test_mode_envelopes_have_exact_duty(mode, expected_nonzero_steps):
    scheduler = fixed_mode_scheduler(mode=mode, duration_steps=200, seed=9)
    values = torch.stack([scheduler.advance()[0] for _ in range(200)])
    assert int((values.abs().sum(dim=1) > 0).sum()) == expected_nonzero_steps


def test_curriculum_reaches_full_scale_and_selective_reset_keeps_progress():
    scheduler = CoordinatedDisturbanceScheduler(CoordinatedDisturbanceCfg(curriculum_steps=2), 4, "cpu", 0.005, seed=3)
    scheduler.advance(); before = scheduler.current_wrench_b.clone()
    scheduler.reset([1, 3]); scheduler.advance()
    assert scheduler.curriculum_scale == pytest.approx(1.0)
    assert torch.equal(scheduler.current_wrench_b[[0, 2]], before[[0, 2]])
    assert torch.equal(scheduler.current_wrench_b[[1, 3]], torch.zeros(2, 6)) is False


def test_nonfinite_wrench_fails_before_robot_call(fake_wrapper):
    with pytest.raises(RuntimeError, match="finite"):
        fake_wrapper._apply_training_wrench(torch.full((2, 6), float("nan")))
    assert fake_wrapper.robot.external_force_calls == []
```

Intermittent semantics are fixed here: a `0.25 s` period with its first `20%` on and remaining `80%` off, repeated until the sampled segment ends. This avoids 200 Hz Bernoulli chatter while preserving the approved intermittent mode and duty.

- [ ] **Step 2: Write wrapper RED tests**

Use fake robot/env objects and assert call order is `_apply_training_wrench` before `env.step`, exact `body_ids=[panda_hand_id]`, default-disabled wrapper never advances the scheduler, done IDs reset selectively, and diagnostics contain only finite Python floats.

- [ ] **Step 3: Run RED**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_coordinated_disturbance.py Go2Pvcnn/tests/test_m1_panda_coordinated_wrapper_disturbance.py -q
```
Expected: FAIL because scheduler/wrapper API does not exist.

- [ ] **Step 4: Implement the pure scheduler**

Use one device-local `torch.Generator` seeded from the CLI seed, per-env target/duration/mode/elapsed tensors, and the existing `base_wrench_to_body_local` conversion. `advance()` increments one control-step counter shared by synchronously stepped envs; this is exactly each environment's 200 Hz step count.

- [ ] **Step 5: Wire the wrapper without changing default behavior**

When enabled, resolve exactly one `BASE_LINK` and `panda_hand`, validate finite `[N,6]`, convert the base-frame wrench to hand-local axes, and call `set_external_force_and_torque(force_h.unsqueeze(1), torque_h.unsqueeze(1), body_ids=[hand_id])` before physics. When disabled, do not instantiate/advance the scheduler and clear no unrelated caller state. On done, reset only done scheduler rows after the env auto-reset.

- [ ] **Step 6: Run GREEN and wrapper regressions**

Run Step 3 plus existing coordinated wrapper, Teacher disturbance, and learning-env tests. Expected: all pass; default wrapper action sequence remains phase mask → nominal action → clamp → physics.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_disturbance.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py Go2Pvcnn/tests/test_m1_panda_coordinated_disturbance.py Go2Pvcnn/tests/test_m1_panda_coordinated_wrapper_disturbance.py
git commit -m "feat: disturb coordinated Panda hand"
```

### Task 7: Stable training entrypoint, manifest, and automatic final rollback

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_coordinated_train.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_coordinated_train_static.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_train_manifest.py`

**Interfaces:**
- Training defaults to 64 envs, seed 42, 600 maximum updates, GPU selected by caller, enabled training DR, and the Task 1 config.
- `initialize_fresh_zero_action_policy(runner)` zeros the actor output layer but leaves scalar std trainable.
- Manifest schema 2 records exact PPO/DR/guard contract and final best→rollback SHA chain.

- [ ] **Step 1: Rewrite static tests as the new exact contract**

Assert source imports `get_m1_panda_coordinated_train_cfg`, calls `configure_coordinated_training_domain_randomization(cfg, True)`, constructs `M1PandaCoordinatedEnvWrapper(env, training_randomization=True, seed=args.seed)`, never sets schedule to fixed, never calls `requires_grad_(False)`, and uses `AtomicCheckpointController` as runner callback.

- [ ] **Step 2: Write manifest-state unit tests**

Extract a pure `build_manifest_contract(args, asset_path, init_checkpoint, train_cfg)` helper and assert exact fields:
```python
assert manifest["schema_version"] == 2
assert manifest["ppo"]["num_steps_per_env"] == 256
assert manifest["ppo"]["gamma"] == 0.9995
assert manifest["ppo"]["learning_rate_bounds"] == [1e-6, 3e-4]
assert manifest["domain_randomization"]["target_body"] == "panda_hand"
assert manifest["guard"]["minimum_completed_episodes"] == 100
assert manifest["fresh_policy"] is True
```

- [ ] **Step 3: Run RED**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_coordinated_train_static.py Go2Pvcnn/tests/test_m1_panda_coordinated_train_manifest.py -q
```
Expected: FAIL on fixed schedule/frozen std/old config and missing controller contract.

- [ ] **Step 4: Implement the new train path**

Keep `--init-a1-checkpoint` provenance-only and required. Set `--max_iterations` default `600`; reject values above 600. Construct the Task 1 config without post-hoc PPO overrides. Update zero initialization to:
```python
torch.nn.init.zeros_(output_layer.weight)
torch.nn.init.zeros_(output_layer.bias)
if not runner.alg.actor_critic.noise_parameter.requires_grad:
    raise RuntimeError("coordinated policy std must remain trainable")
```

Call:
```python
learn_result = runner.learn(
    num_learning_iterations=args.max_iterations,
    init_at_random_ep_len=True,
    iteration_callback=controller.on_iteration,
)
final_fields = controller.finalize(runner, learn_result.stop_reason)
manifest.update(final_fields)
atomic_write_json(manifest_path, manifest)
```

On exceptions, preserve failed manifest state and any valid prior best; never overwrite it with a failed final.

- [ ] **Step 5: Run GREEN and full relevant local regression**

Run Step 3 plus Tasks 1–6 tests, existing coordinated tests, Teacher std/disturbance tests, and runner checkpoint tests. Compile all changed Python modules. Expected: all pass; no CUDA process required.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/scripts/m1_panda_coordinated_train.py Go2Pvcnn/tests/test_m1_panda_coordinated_train_static.py Go2Pvcnn/tests/test_m1_panda_coordinated_train_manifest.py
git commit -m "feat: train coordinated PPO with rollback"
```

### Task 8: GPU0 randomization/physics probe and guarded short train

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_coordinated_randomization_probe.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_randomization_probe_static.py`
- Create: `notes/log/2026-08-24-m1-panda-coordinated-randomization-probe.md`
- Create: `notes/log/2026-08-24-m1-panda-coordinated-guarded-short-train.md`

**Interfaces:**
- Probe emits one JSON summary with seeded reset diversity, bounds, selective-reset isolation, applied hand wrench, mount-wrench response, and finite/reset/contact counts.
- Short train uses 64 envs and 50 updates; it tests the 100-episode best/final chain, not convergence.

- [ ] **Step 1: Write probe static RED**

Assert AppLauncher import order, `training_randomization=True`, two same-seed comparison batches, selected-env reset comparison, `panda_hand`, mount wrench, finite checks, and atomic JSON output fields.

- [ ] **Step 2: Run RED, implement the probe, then local GREEN**

The probe must fail nonzero unless same-seed tensors match, different envs are non-identical, every reset/wrench bound holds, unselected rows remain unchanged across selected reset, and a nonzero hand wrench produces a finite nonzero mount response after settling.

- [ ] **Step 3: Run the real GPU0 probe**

Run from `Go2Pvcnn`:
```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_coordinated_randomization_probe.py \
  --num_envs 8 --steps 256 --seed 42 --device cuda:0 --headless \
  --output tests/artifacts/m1_panda_coordinated_randomization_probe.json
```
Expected: exit `0`; all hard gates true. Record exact metrics in the probe log before proceeding.

- [ ] **Step 4: Run a one-update GPU0 wiring smoke**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_coordinated_train.py \
  --num_envs 8 --max_iterations 1 --seed 42 \
  --run_name coordinated_stability_wiring_8x1_20260824 \
  --init-a1-checkpoint logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless
```
Expected: finite PPO, LR/KL/DR TensorBoard tags, ordinary `model_0.pt`, manifest completed without falsely claiming an eligible best.

- [ ] **Step 5: Run the 64-env×50 guarded short train**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_coordinated_train.py \
  --num_envs 64 --max_iterations 50 --seed 42 \
  --run_name coordinated_stability_guard_64x50_20260824 \
  --init-a1-checkpoint logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless
```
Expected: exit `0`; `model_best.pt`, `best_checkpoint.json`, `model_final.pt`, and manifest SHA chain all exist and agree. `accepted` may be false if the fresh 50-update policy misses gates; infrastructure correctness must not be relabeled as convergence.

- [ ] **Step 6: Record both verification logs and commit**

```bash
git add Go2Pvcnn/scripts/m1_panda_coordinated_randomization_probe.py Go2Pvcnn/tests/test_m1_panda_coordinated_randomization_probe_static.py notes/log/2026-08-24-m1-panda-coordinated-randomization-probe.md notes/log/2026-08-24-m1-panda-coordinated-guarded-short-train.md
git commit -m "test: verify coordinated PPO stability wiring"
```

### Task 9: Audited old-checkpoint pruning, notes alignment, and GPU0 long-run launch

**Files:**
- Create: `Go2Pvcnn/scripts/prune_m1_panda_coordinated_checkpoints.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_checkpoint_pruning.py`
- Create: `docs/superpowers/runbooks/2026-08-24-m1-panda-coordinated-stable-training.md`
- Create: `notes/log/2026-08-24-m1-panda-coordinated-checkpoint-pruning.md`
- Create: `notes/log/2026-08-24-m1-panda-coordinated-stable-long-launch.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/human/human-05-ppo-and-runner.md`
- Modify: `notes/ai/ai-05-ppo-and-runner.md`

**Interfaces:**
- Pruner accepts only the exact long-v4 resolved directory, defaults to dry-run, requires `--apply` to unlink, and atomically writes `checkpoint_pruning.json` before deletion.
- Runbook gives deterministic Play/probe, short train, long train, TensorBoard, stop/rollback inspection, and recovery commands.

- [ ] **Step 1: Write pruning RED tests in a temporary directory**

```python
def test_plan_keeps_3500_and_selects_only_numeric_models_above_it(fake_run):
    plan = build_pruning_plan(fake_run, keep_through=3500, expected_run_dir=fake_run)
    assert [item.path.name for item in plan.delete] == ["model_3600.pt", "model_4999.pt"]
    assert (fake_run / "model_3500.pt").is_file()


def test_plan_rejects_symlinks_and_paths_outside_exact_run(fake_run, tmp_path):
    (fake_run / "model_3600.pt").unlink()
    (fake_run / "model_3600.pt").symlink_to(tmp_path / "outside.pt")
    with pytest.raises(ValueError, match="symlink"):
        build_pruning_plan(fake_run, keep_through=3500, expected_run_dir=fake_run)
    with pytest.raises(ValueError, match="exact long-v4"):
        build_pruning_plan(tmp_path, keep_through=3500, expected_run_dir=fake_run)


def test_dry_run_does_not_unlink_and_apply_writes_hashes(fake_run):
    plan = build_pruning_plan(fake_run, keep_through=3500, expected_run_dir=fake_run)
    execute_pruning(plan, apply=False)
    assert (fake_run / "model_3600.pt").is_file()
    execute_pruning(plan, apply=True)
    audit = json.loads((fake_run / "checkpoint_pruning.json").read_text())
    assert audit["status"] == "completed"
    assert [item["name"] for item in audit["deleted"]] == ["model_3600.pt", "model_4999.pt"]
    assert all(len(item["sha256"]) == 64 for item in audit["deleted"])
    assert not (fake_run / "model_3600.pt").exists()


def test_postcondition_rejects_remaining_checkpoint_above_limit(fake_run):
    plan = build_pruning_plan(fake_run, keep_through=3500, expected_run_dir=fake_run)
    with pytest.raises(RuntimeError, match="remaining"):
        verify_pruning_postcondition(plan.run_dir, 3500, ignored_names={"model_3600.pt"})
```

The audit schema records original manifest SHA, upper bound 3500, kept `model_3500.pt` SHA, and every selected filename/SHA. Use atomic two-phase states `planned` then `completed`, so an interrupted unlink cannot masquerade as successful cleanup. It must never edit the original `run_manifest.json`.

`build_pruning_plan(run_dir, keep_through, expected_run_dir=EXPECTED_LONG_V4_RUN_DIR)` resolves and compares both directories before enumerating files. The CLI never exposes `expected_run_dir`; only tests inject their temporary fixture path.

- [ ] **Step 2: Run RED, implement, and run GREEN**

Run:
```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_coordinated_checkpoint_pruning.py -q
```
Expected after implementation: all pass.

- [ ] **Step 3: Dry-run and verify the real exact list**

```bash
/home/xk/miniconda3/envs/go2/bin/python Go2Pvcnn/scripts/prune_m1_panda_coordinated_checkpoints.py \
  --run-dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823 \
  --keep-through 3500
```
Expected selected files: `model_3600.pt` through `model_4900.pt`, plus `model_4999.pt`; exactly 15 files.

- [ ] **Step 4: Apply deletion and verify audit/postcondition**

Repeat Step 3 with `--apply`, then run:
```bash
find Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823 -maxdepth 1 -name 'model_*.pt' -printf '%f\n' | sort -V | tail
sha256sum Go2Pvcnn/logs/m1_panda_coordinated/coordinated_teacher_long_v4_64x5000_20260823/model_3500.pt
```
Expected: no checkpoint number above 3500; audit JSON lists 15 pre-delete hashes. Record that deleted files are recoverable only from external backup.

- [ ] **Step 5: Run final local verification and diff audit**

Run the complete focused suite from Tasks 1–9, all existing `test_m1_panda_coordinated*.py`, Teacher disturbance/std tests, runner tests, `py_compile`, `git diff --check`, and a source scan proving fixed schedule/frozen std are absent from the coordinated train path. Create one final verification log if this is a distinct pass.

- [ ] **Step 6: Write runbook and align project memory**

Document exact input/method/output contracts, TensorBoard tags, stop reasons, `accepted` meaning, and that this stage is coordinated normal control under disturbance—not grasping or hardware validation. Update T400.10a from implementation to long-run monitoring and set Git refs to the actual verified commits.

- [ ] **Step 7: Launch the isolated fresh GPU0 64×600 run**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_coordinated_train.py \
  --num_envs 64 --max_iterations 600 --seed 42 \
  --run_name coordinated_stable_fresh_s42_64x600_20260824 \
  --init-a1-checkpoint logs/m1_panda_teacher/a1/a1_force_balance_recovery_gpu0_20260815/model_10402.pt \
  --device cuda:0 --headless \
  > logs/m1_panda_coordinated/coordinated_stable_fresh_s42_64x600_20260824.stdout.log 2>&1
```

Run it in a persistent terminal/session, record PID and first healthy update in the launch log, and monitor without declaring convergence. The controller may stop before 600 only through the approved catastrophe/patience gates and must leave `model_final.pt` equal to the eligible best rollback source.

- [ ] **Step 8: Commit documentation and operational tooling**

```bash
git add Go2Pvcnn/scripts/prune_m1_panda_coordinated_checkpoints.py Go2Pvcnn/tests/test_m1_panda_coordinated_checkpoint_pruning.py docs/superpowers/runbooks/2026-08-24-m1-panda-coordinated-stable-training.md notes/log notes/todo.md notes/todo/T400-m1-panda-force-aware-teacher-student.md notes/human/human-05-ppo-and-runner.md notes/ai/ai-05-ppo-and-runner.md
git commit -m "train: launch guarded coordinated PPO run"
```

## Final Acceptance Checklist

- [ ] Exact 103/23/200 Hz/asset contracts unchanged.
- [ ] Fresh actor output is zero and scalar std remains trainable/clamped to `[0.005,0.05]`.
- [ ] Adaptive KL changes LR only within `[1e-6,3e-4]`; TensorBoard records LR, KL, adjustment.
- [ ] Default runner callers and deterministic coordinated wrapper behavior regress cleanly.
- [ ] Training-only reset/friction/wrench randomization is seeded, bounded, independent, and physically reaches mount wrench through `panda_hand`.
- [ ] Best selection, eligibility, catastrophe, patience, max-update, atomic hashes, and final rollback pass pure tests and GPU short-train artifact checks.
- [ ] Exactly the old checkpoints above 3500 are removed with a valid pre-delete SHA audit; `model_3500.pt` and original manifest remain.
- [ ] GPU0 long run is launched in a fresh directory with its PID/command recorded, but no convergence/grasping/real-hardware claim is made before behavioral acceptance.
- [ ] Todo, branch memory, verification logs, runner human/AI notes, runbook, and Git refs are aligned.
