# M1 + Panda Phase 6 Bridge and Normalizer Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the safe-complete Phase 6 v6 residual policy from total update 100 through update 300 with exact optimizer and empirical-normalizer continuity, rerun the unchanged 24-worker promotion gate, and start the 3000-update long run only when promotion records `accepted=true`.

**Architecture:** Persist empirical-normalizer sample counts in vendored RSL-RL and normalize the first observation of every learning call.  Add a pure bridge-lineage/migration module, then extend the existing train and promotion entrypoints with a schema-v3 bridge path while leaving pilot, short, physical limits, reward, and promotion tolerances unchanged.  GPU execution uses fresh bridge/long roots and the existing process-isolated seeds 42/43/44 gate.

**Tech Stack:** Python 3.11, PyTorch, pytest, vendored RSL-RL PPO, Isaac Lab/Isaac Sim 5.1, CUDA GPU0, JSON/SHA-256 manifests, Git.

## Global Constraints

- Work directly on `/home/xk/coding/M1` branch `main`; use one agent only.
- Do not stage or overwrite `graphify-out/cache/last_query_stamp` or unrelated user changes.
- Preserve the completed v6 pilot, short, worker JSON, checkpoints, and rejected promotion as immutable evidence.
- Keep reward coefficients, physical residual limits, learning-rate/KL bounds, seeds 42/43/44, 4000 evaluation steps, noise calibration, rank metrics, wrench/slip regression checks, and promotion tolerances unchanged.
- The bridge adds exactly 200 updates after the completed v6 u100 parent and publishes total-update candidates 100/150/200/250/300.
- Restore model, trainable standard deviation, optimizer, adaptive learning rate, actor normalizer, critic normalizer, and their sample counts before bridge collection.
- Migrate the one legacy v6 u100 count as exactly `100 * 256 * 8 = 204800`; reject every lineage mismatch.
- Every newly written bridge/promoted/long checkpoint must contain explicit actor and critic normalizer counts.
- Run Isaac/RSL only with `CUDA_VISIBLE_DEVICES=0`, logical device `cuda:0`, and `/home/xk/miniconda3/envs/go2/bin/python`.
- Never create or start the long run unless the parsed schema-v3 promotion manifest has `accepted=true` and a hash-valid counted checkpoint.
- Do not claim Phase 6 completion until a final long manifest proves safe completion of all 3000/3000 requested updates.

---

### Task 1: Persist Empirical-Normalizer Count and Normalize the First Rollout Observation

**Files:**
- Create: `Go2Pvcnn/tests/test_rsl_empirical_normalization_continuity.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/normalizer.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`

**Interfaces:**
- Consumes: `EmpiricalNormalization(shape, eps=1e-2, until=None)` and `OnPolicyRunner.learn(...)`.
- Produces: `EmpiricalNormalization.count -> int`, a persistent scalar-int64 `_count` state entry, exact load/save round trips, and normalized initial actor/critic observations before the first action.

- [ ] **Step 1: Write failing normalizer count tests**

Create the focused test file with these contracts:

```python
from __future__ import annotations

import torch

from rsl_rl.modules.normalizer import EmpiricalNormalization


def test_empirical_normalizer_count_round_trips_in_state_dict():
    normalizer = EmpiricalNormalization([2])
    normalizer.train()
    normalizer(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    state = normalizer.state_dict()
    assert state["_count"].dtype == torch.int64
    assert state["_count"].shape == torch.Size([])
    assert state["_count"].item() == 3

    restored = EmpiricalNormalization([2])
    restored.load_state_dict(state)
    assert restored.count == 3
    torch.testing.assert_close(restored.mean, normalizer.mean)
    torch.testing.assert_close(restored.std, normalizer.std)


def test_eval_forward_does_not_mutate_restored_count():
    normalizer = EmpiricalNormalization([2])
    normalizer.load_state_dict(
        {
            "_mean": torch.tensor([[2.0, 3.0]]),
            "_var": torch.tensor([[4.0, 9.0]]),
            "_std": torch.tensor([[2.0, 3.0]]),
            "_count": torch.tensor(204800, dtype=torch.int64),
        }
    )
    normalizer.eval()
    normalizer(torch.tensor([[4.0, 6.0]]))
    assert normalizer.count == 204800


def test_training_continues_from_restored_count_without_overwrite():
    normalizer = EmpiricalNormalization([1])
    normalizer.load_state_dict(
        {
            "_mean": torch.tensor([[10.0]]),
            "_var": torch.tensor([[4.0]]),
            "_std": torch.tensor([[2.0]]),
            "_count": torch.tensor(100, dtype=torch.int64),
        }
    )
    normalizer.train()
    normalizer(torch.tensor([[20.0]]))
    assert normalizer.count == 101
    assert normalizer.mean.item() == pytest.approx((1000.0 + 20.0) / 101.0)
```

Import `pytest` in the test file for the final approximation assertion.

- [ ] **Step 2: Run the count tests and verify RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_rsl_empirical_normalization_continuity.py
```

Expected: collection or assertions fail because `_count` is absent and `count` is not persistent.

- [ ] **Step 3: Implement the persistent count**

Replace the plain integer with a scalar buffer and use integer arithmetic for the update weight:

```python
self.register_buffer("_count", torch.zeros((), dtype=torch.int64))

@property
def count(self) -> int:
    return int(self._count.item())

@torch.jit.unused
def update(self, x):
    previous_count = self.count
    if self.until is not None and previous_count >= self.until:
        return
    count_x = int(x.shape[0])
    next_count = previous_count + count_x
    rate = count_x / next_count
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    delta_mean = mean_x - self._mean
    self._mean += rate * delta_mean
    self._var += rate * (
        var_x - self._var + delta_mean * (mean_x - self._mean)
    )
    self._std = torch.sqrt(self._var)
    self._count.fill_(next_count)
```

Do not silently accept a missing `_count` through a custom load hook; the bridge migration in Task 2 is the only legacy exception.

- [ ] **Step 4: Add a static test for first-observation normalization**

Append a source-level assertion that `learn()` enters train mode and normalizes the initial actor and critic observations before the main loop:

```python
def test_runner_normalizes_initial_observations_before_first_action():
    source = (
        Path(__file__).resolve().parents[1]
        / "rsl_rl/rsl_rl/runners/on_policy_runner.py"
    ).read_text(encoding="utf-8")
    train_mode = source.index("self.train_mode()")
    loop = source.index("for it in range(start_iter, tot_iter)")
    actor_normalization = source.index("obs = self.obs_normalizer(obs)", train_mode)
    critic_normalization = source.index(
        "critic_obs = self.critic_obs_normalizer(critic_obs)", train_mode
    )
    assert train_mode < actor_normalization < loop
    assert train_mode < critic_normalization < loop
```

- [ ] **Step 5: Run the new static test and verify RED**

Run the same focused file.  Expected: the count tests pass after Step 3, while the initial-observation source assertion fails because normalization currently begins inside the rollout loop.

- [ ] **Step 6: Normalize initial observations before collection**

Immediately after `self.train_mode()` in `OnPolicyRunner.learn`, add:

```python
obs = self.obs_normalizer(obs)
critic_obs = self.critic_obs_normalizer(critic_obs)
```

Keep the existing per-step normalization unchanged.

- [ ] **Step 7: Run focused and runner regression tests**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_rsl_empirical_normalization_continuity.py \
  tests/test_rsl_runner_iteration_callback.py \
  tests/test_rsl_ppo_adaptive_schedule.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
cd /home/xk/coding/M1
git add \
  Go2Pvcnn/rsl_rl/rsl_rl/modules/normalizer.py \
  Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  Go2Pvcnn/tests/test_rsl_empirical_normalization_continuity.py
git commit -m "fix: persist residual normalizer progress"
```

### Task 2: Add Fail-Closed Bridge Parent Validation and Legacy u100 Migration

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_bridge.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_bridge.py`

**Interfaces:**
- Consumes: a v6 schema-v2 safe-complete short manifest and its hash-recorded u100 checkpoint.
- Produces: `BridgeParent`, `validate_bridge_parent(...)`, `migrate_legacy_u100_checkpoint(...)`, and `validate_counted_checkpoint(...)`.

- [ ] **Step 1: Write failing bridge contract tests**

Create tests that build temporary asset/config/reward/runtime files, a valid accepted pilot manifest, a safe-complete short manifest, and a legacy checkpoint with both normalizer dictionaries but no `_count`.  Freeze these public assertions:

```python
assert parent.completed_updates == 100
assert parent.sample_count == 204800
assert parent.checkpoint_sha256 == sha256_file(parent.checkpoint)

migrated_sha = migrate_legacy_u100_checkpoint(
    parent.checkpoint,
    migrated,
    expected_parent_sha256=parent.checkpoint_sha256,
    sample_count=parent.sample_count,
)
assert migrated_sha == sha256_file(migrated)
actor_count, critic_count = validate_counted_checkpoint(
    migrated, expected_count=204800
)
assert (actor_count, critic_count) == (204800, 204800)
```

Also compare every original model/optimizer/normalizer tensor except `_count`, and add rejection cases for wrong parent SHA, update set other than `{0,25,50,75,100}`, missing normalizer dictionaries, pre-existing `_count`, wrong u100 hash, and non-empty migration target.

- [ ] **Step 2: Run bridge tests and verify RED**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_bridge.py
```

Expected: import fails because the bridge module does not exist.

- [ ] **Step 3: Implement immutable bridge constants and lineage type**

Use these exact definitions:

```python
BRIDGE_PARENT_UPDATES = 100
BRIDGE_ADDITIONAL_UPDATES = 200
BRIDGE_TOTAL_UPDATES = 300
BRIDGE_CANDIDATE_UPDATES = (100, 150, 200, 250, 300)
BRIDGE_PARENT_CANDIDATES = (0, 25, 50, 75, 100)
BRIDGE_LEGACY_SAMPLE_COUNT = 100 * 256 * 8

@dataclass(frozen=True)
class BridgeParent:
    short_manifest: Path
    short_manifest_sha256: str
    pilot_manifest: Path
    pilot_manifest_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    completed_updates: int
    sample_count: int
```

`validate_bridge_parent(manifest_path, source_paths)` must require schema 2, stage short, safe completion, exactly 100 requested/completed iterations, accepted false, promotion required true, matching current source lineage, accepted pilot lineage, exact candidate set, and a hash-valid legacy u100 checkpoint without counts.

- [ ] **Step 4: Implement atomic legacy migration and counted validation**

`migrate_legacy_u100_checkpoint` loads on CPU, verifies the expected SHA, clones both normalizer mappings, inserts scalar-int64 `_count` tensors, writes through a same-directory temporary file plus `os.replace`, and refuses an existing target.  `validate_counted_checkpoint` requires both counts to be scalar int64, non-negative, equal to one another, and equal to `expected_count` when supplied.

- [ ] **Step 5: Run bridge and lineage tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_bridge.py \
  tests/test_m1_panda_arm_mpc_residual_lineage.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
cd /home/xk/coding/M1
git add \
  Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_bridge.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_bridge.py
git commit -m "feat: validate phase6 bridge lineage"
```

### Task 3: Extend the Residual Trainer with the Protected Bridge Stage

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`

**Interfaces:**
- Consumes: `BridgeParent`, the migrated counted u100 checkpoint, and the existing safety callback.
- Produces: `--stage bridge --short_manifest PATH`, a schema-v3 bridge manifest, and candidates at total updates 100/150/200/250/300.

- [ ] **Step 1: Write failing bridge CLI and controller tests**

Add assertions for:

```python
assert module.resolve_max_iterations("bridge", None) == 200
assert module.resolve_max_iterations("bridge", 200) == 200
with pytest.raises(ValueError, match="exactly 200"):
    module.resolve_max_iterations("bridge", 199)

args = module.build_arg_parser(include_app_launcher_args=False).parse_args(
    ["--stage", "bridge", "--short_manifest", "short.json"]
)
assert args.stage == "bridge"
assert args.short_manifest == Path("short.json")
```

Create a fake runner whose `save()` writes distinct bytes.  Initialize the generalized safety controller with `starting_updates=100` and candidate updates `(100,150,200,250,300)`, register an existing u100 path, then feed summaries at zero-based bridge loop positions corresponding to totals 150/200/250/300.  Assert exact filenames and no u101/u299 candidates.

- [ ] **Step 2: Run train static tests and verify RED**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_train_static.py
```

Expected: bridge stage/parser/controller assertions fail.

- [ ] **Step 3: Add bridge arguments and exact stage limit**

Use:

```python
STAGE_LIMITS = {
    "zero": 10,
    "pilot": 10,
    "short": 100,
    "bridge": 200,
    "long": 3000,
}
```

Require exactly 200 for bridge just as pilot/short require their exact limits. Add `--short_manifest` and reject it for every non-bridge stage.  Continue to accept `--promotion_manifest` only for long.

- [ ] **Step 4: Generalize candidate accounting without changing short behavior**

Give `ResidualTrainingSafetyController` constructor arguments `starting_updates: int = 0` and `candidate_updates: tuple[int, ...] = (0,25,50,75,100)`.  Compute total completed updates as `int(summary.iteration) + 1`; bridge explicitly sets `runner.current_learning_iteration = 100` before `learn()`, so callback totals remain 101 through 300.  Add `register_initial_candidate(100, migrated_path)` and preserve `prime()` for fresh short.

- [ ] **Step 5: Wire bridge validation, migration, and exact resume state**

Before creating the bridge run directory, call `validate_bridge_parent`.  After constructing the runner:

```python
migrated = run_dir / "candidate_u100.pt"
migrate_legacy_u100_checkpoint(
    bridge_parent.checkpoint,
    migrated,
    expected_parent_sha256=bridge_parent.checkpoint_sha256,
    sample_count=bridge_parent.sample_count,
)
runner.load(str(migrated), load_optimizer=True, keep_std=True)
runner.current_learning_iteration = 100
```

Validate both restored normalizer counts before calling `learn()`.  The bridge manifest must record schema 3, original short path/hash, original parent checkpoint path/hash, migrated u100 path/hash, starting count 204800, starting updates 100, requested additional updates 200, and target total updates 300.

- [ ] **Step 6: Enforce safe bridge completion**

Require 200 completed bridge iterations, final total 300, and exactly the five bridge candidate updates before writing `status=safe_complete`.  A callback stop writes `status=safety_stopped`, `accepted=false`, `promotion_required=false`; only safe completion sets `promotion_required=true`.

- [ ] **Step 7: Run train/bridge/guard tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_train_static.py \
  tests/test_m1_panda_arm_mpc_residual_bridge.py \
  tests/test_m1_panda_arm_mpc_residual_guard.py \
  tests/test_m1_panda_arm_mpc_residual_pilot.py
```

Expected: all selected tests pass and legacy pilot/short behavior remains unchanged.

- [ ] **Step 8: Commit Task 3**

```bash
cd /home/xk/coding/M1
git add \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py
git commit -m "feat: add guarded phase6 bridge stage"
```

### Task 4: Add Schema-v3 Bridge Promotion and Counted Long Validation

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`

**Interfaces:**
- Consumes: a safe-complete schema-v3 bridge manifest and five counted candidates.
- Produces: `--bridge_manifest`, a schema-v3 promotion manifest, hash-identical `model_best.pt`, and an accepted-only long lineage that restores optimizer plus normalizer counts.

- [ ] **Step 1: Write failing bridge-promotion fixture and assertions**

Add a `_safe_bridge_run` fixture containing total candidate updates `(100,150,200,250,300)`, explicit `_count=204800` or later counts, original short/pilot lineage, and schema 3.  Assert that the driver still launches exactly nine calibration plus fifteen candidate workers and emits:

```python
assert result["schema_version"] == 3
assert result["bridge_manifest"] == str(bridge_manifest.resolve())
assert result["bridge_manifest_sha256"] == module.sha256_file(bridge_manifest)
assert [x["completed_updates"] for x in result["candidates"]] == [
    100, 150, 200, 250, 300
]
```

Add rejection tests for a missing `_count`, mismatched actor/critic counts, wrong bridge parent SHA, wrong candidate set, safety-stopped bridge, and schema-2 short passed through `--bridge_manifest`.

- [ ] **Step 2: Run promotion tests and verify RED**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_promote.py
```

Expected: bridge fixture/CLI/schema assertions fail.

- [ ] **Step 3: Generalize promotion input while preserving short compatibility**

Keep the existing `--short_manifest` path for historical runs and add a mutually exclusive `--bridge_manifest`.  Split manifest loading into short and bridge validators that both return `CandidateRecord` values.  The bridge validator requires schema 3, safe 200/200 additional updates, starting/total 100/300, promotion required true, exact source and parent hashes, candidates 100/150/200/250/300, and counted checkpoints.

- [ ] **Step 4: Emit schema-v3 bridge promotion lineage**

For bridge input, write `bridge_manifest`, `bridge_manifest_sha256`, `short_manifest`, `short_manifest_sha256`, and pilot lineage into the atomic result.  Keep calibration, comparison, selection, resume identity, and engineering floors byte-for-byte unchanged.  Continue publishing `model_best.pt` by atomic copy and verify its SHA.

- [ ] **Step 5: Write failing long-lineage tests**

Build an accepted schema-v3 promotion fixture.  Assert `validate_promotion_manifest` returns bridge, short, and checkpoint lineage.  Add failures for `accepted=false`, missing bridge manifest, bridge SHA mismatch, missing/mismatched `_count`, selected checkpoint not recorded among bridge candidates, and optimizer state absence.

- [ ] **Step 6: Implement counted long lineage and optimizer restore**

Extend `PromotionLineage` with bridge path/hash and require schema-v3 bridge ancestry.  Validate the selected checkpoint using `validate_counted_checkpoint` and require a non-empty `optimizer_state_dict`.  In long setup use:

```python
runner.load(
    str(promotion_lineage.checkpoint),
    load_optimizer=True,
    keep_std=True,
)
runner.current_learning_iteration = 0
```

Verify normalizer counts again before `learn()`.  Preserve the long limit of 3000 and all online safety callbacks.

- [ ] **Step 7: Run promotion, lineage, and static regressions**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_promote.py \
  tests/test_m1_panda_arm_mpc_residual_promotion.py \
  tests/test_m1_panda_arm_mpc_residual_lineage.py \
  tests/test_m1_panda_arm_mpc_residual_train_static.py
```

Expected: all selected tests pass; promotion thresholds and comparisons remain unchanged.

- [ ] **Step 8: Commit Task 4**

```bash
cd /home/xk/coding/M1
git add \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py
git commit -m "feat: promote counted phase6 bridge candidates"
```

### Task 5: Run Complete CPU Verification and Freeze the GPU Commands

**Files:**
- Modify: `notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: Tasks 1–4 implementation and tests.
- Produces: a clean CPU/compile/static gate and an authoritative v7 command ledger.

- [ ] **Step 1: Run the complete residual and runner test set**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_rsl_empirical_normalization_continuity.py \
  tests/test_rsl_runner_iteration_callback.py \
  tests/test_rsl_ppo_adaptive_schedule.py \
  tests/test_m1_panda_residual_actor_critic.py \
  tests/test_m1_panda_arm_mpc_residual_reward.py \
  tests/test_m1_panda_arm_mpc_residual_runtime.py \
  tests/test_m1_panda_arm_mpc_residual_guard.py \
  tests/test_m1_panda_arm_mpc_residual_pilot.py \
  tests/test_m1_panda_arm_mpc_residual_lineage.py \
  tests/test_m1_panda_arm_mpc_residual_bridge.py \
  tests/test_m1_panda_arm_mpc_residual_train_static.py \
  tests/test_m1_panda_arm_mpc_residual_promotion.py \
  tests/test_m1_panda_arm_mpc_residual_promote.py
```

Expected: all selected tests pass with no skipped bridge contract.

- [ ] **Step 2: Run compile and worktree checks**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m compileall -q \
  agent go2_pvcnn scripts rsl_rl/rsl_rl
cd /home/xk/coding/M1
git diff --check
git status --short
```

Expected: compile exits 0; diff check emits no output; status contains only intentional evidence/plan edits plus the unrelated graphify stamp.

- [ ] **Step 3: Record exact verification and launch commands**

Append test counts, commit hashes, source hashes, immutable v6 parent SHA, fresh v7 roots, and the exact Task 6/7 commands to the execution log.  Update the index and T400 current state without claiming GPU acceptance.

- [ ] **Step 4: Commit the implementation evidence preflight**

```bash
cd /home/xk/coding/M1
git add \
  notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md \
  notes/log/index.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: record phase6 bridge preflight"
```

### Task 6: Execute and Verify the GPU0 Update-100-to-300 Bridge

**Files:**
- Runtime input: `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_corrected_obs_v6/run_manifest.json`
- Runtime output: `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7/`
- Modify: `notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md`

**Interfaces:**
- Consumes: immutable v6 u100 plus schema-v3 migration/bridge code.
- Produces: a safe-complete bridge manifest and five counted candidates.

- [ ] **Step 1: Prove the fresh root and GPU0 availability**

```bash
cd /home/xk/coding/M1
test ! -e Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Expected: the root does not exist and GPU0 has no conflicting training process.

- [ ] **Step 2: Launch the exact bridge**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage bridge \
  --short_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_corrected_obs_v6/run_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7 \
  --num_envs 8 --max_iterations 200 --seed 42 \
  --device cuda:0 --headless
```

Expected: GPU0 remains owned by the bridge process; no hard failure, non-finite diagnostic, KL safety stop, QP failure, or lost contact occurs.

- [ ] **Step 3: Continuously monitor without interpreting liveness as acceptance**

Poll process liveness, GPU0 memory, atomic manifest state, TensorBoard update count, learning rate, KL mean/max/abort, completed mini-batches, gradient norm, action std, hard failures, MPC/QP/contact rates, wrench error, and candidate publication.  Report only state changes or errors; a running process is progress, not completion.

- [ ] **Step 4: Parse the bridge manifest and checkpoints**

Require schema 3, `status=safe_complete`, additional iterations 200/200, starting/total updates 100/300, accepted false, promotion required true, exact parent hashes, and candidate updates 100/150/200/250/300.  Recompute every candidate SHA and load each checkpoint to require model, optimizer, actor/critic normalizer mappings, scalar-int64 equal counts, and finite tensors.

- [ ] **Step 5: Record bridge evidence**

Append command, elapsed time, exit status, manifest SHA, parent/migration hashes, initial/final counts, candidate hashes, optimizer ranges, and safety metrics.  If the bridge safety-stops or fails, do not run promotion; return to systematic diagnosis.

### Task 7: Run the 24-Worker Promotion and Conditionally Start/Monitor Long

**Files:**
- Runtime input: `Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7/run_manifest.json`
- Runtime output: bridge `noise_calibration/`, `candidate_eval/`, `promotion_manifest.json`, and conditional `long_s42_normalizer_continuity_v7/`
- Modify: `notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: a verified safe bridge and unchanged physical promotion rules.
- Produces: 24 complete fixed-condition workers, an atomic promotion decision, and only on acceptance a monitored 3000-update long run.

- [ ] **Step 1: Launch the exact bridge promotion**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_promote.py \
  --bridge_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7/run_manifest.json \
  --device cuda:0 --headless
```

Expected: nine calibration and fifteen candidate workers execute in isolated processes.  Drain process output frequently enough to prevent PTY blocking and report completed-worker milestones.

- [ ] **Step 2: Verify all fixed-condition evidence**

Require exactly 9 calibration and 15 candidate JSON files, all `status=complete`, all `steps=4000`, exact seeds 42/43/44, exact checkpoint/source/bridge lineage, and no missing or duplicate candidate.  If a transient worker fails, inspect it first and use `--resume` only when the existing identity contract validates all complete and retryable artifacts.

- [ ] **Step 3: Enforce the atomic promotion branch**

Parse `promotion_manifest.json`.  If `accepted=false`, verify `best_checkpoint=null` and that the v7 long root does not exist; record exact candidate reasons and stop without weakening any threshold.  If `accepted=true`, verify the published best SHA, candidate membership, explicit equal normalizer counts, non-empty optimizer state, and every bridge/short/pilot/source hash before continuing.

- [ ] **Step 4: Launch long only on exact acceptance**

Run only when Step 3 proves acceptance:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage long \
  --promotion_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/bridge_s42_normalizer_continuity_v7/promotion_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_normalizer_continuity_v7 \
  --num_envs 8 --max_iterations 3000 --seed 42 \
  --device cuda:0 --headless
```

- [ ] **Step 5: Continuously monitor the 3000-update long run**

Monitor process/GPU liveness, completed updates, manifest status, checkpoint hashes, restored and increasing normalizer counts, optimizer/LR continuity, KL and minibatch aborts, value loss, action std, hard failure, MPC/QP/contact, wrench, slip, EE, and saturation diagnostics.  Do not stop merely because early learning metrics are poor; stop only through the approved online safety controller or a real runtime failure.

- [ ] **Step 6: Perform the final completion audit**

Require a safe-complete long manifest with exactly 3000/3000 requested updates, accepted promotion lineage, valid final/best checkpoint SHA, counted normalizers, finite optimizer diagnostics, zero hard safety violations, and no missing required artifacts.  Run the complete Task 5 CPU suite, compileall, `git diff --check`, and inspect the current worktree before claiming Phase 6 complete.

- [ ] **Step 7: Commit final evidence only after the audited outcome**

```bash
cd /home/xk/coding/M1
git add \
  notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md \
  notes/log/index.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: record phase6 bridge promotion and long result"
```

Expected: the documentation states the actual accepted/rejected/running/completed outcome and never equates process liveness with Phase 6 completion.
