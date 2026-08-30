# M1 + Panda Phase 6 PPO Scale and Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Phase 6 PPO reward/observation scale, enforce normalized checkpoint inference, gate the official short run behind a strict 10-update pilot, and launch the 3000-update long run only from a fresh accepted v3 promotion.

**Architecture:** Keep physical control, reward composition, and promotion tolerances frozen. Add a small source-lineage module shared by train/eval/promote, a pure pilot-decision module fed by detached runner iteration summaries, and stage-specific orchestration in the existing residual training script. The runtime alone applies `control_dt` to the PPO reward, while all checkpoint consumers use `OnPolicyRunner.get_inference_policy()` so saved empirical normalizers are never bypassed.

**Tech Stack:** Python 3.11, PyTorch, RSL-RL PPO, Isaac Lab/Isaac Sim, Gymnasium, pytest, JSON/SHA-256 manifests, CUDA GPU0.

## Global Constraints

- Work directly in `/home/xk/coding/M1` on the current `main` worktree and preserve all unrelated dirty Phase 5 changes.
- Use one agent only; do not dispatch subagents.
- Use test-driven development: establish each focused RED failure before production edits, then run the stated GREEN command.
- Preserve the M1+Panda USD, zero-clearance mount, WBC/QP/Arm-MPC implementation, force-sensor/RNE dynamics, 200 Hz WBC rate, and 50 Hz Arm-MPC rate.
- Preserve observation/action/private-effort dimensions `(103, 8, 23)` and action channel order.
- Preserve reward components, relative weights, gates, wrench normalization, EE trajectory, physical limits, safety projection, and termination thresholds.
- Resolve `control_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)`, reject non-finite/non-positive values, and multiply only the complete PPO reward by it.
- Set `empirical_normalization=true` and `init_noise_std=0.01`; retain std bounds `[0.005, 0.02]`, `desired_kl=0.01`, `kl_abort_threshold=0.015`, rollout length 256, two epochs, four mini-batches, `gamma=0.9995`, and `lam=0.995`.
- The pilot is exactly 10 updates, produces no `candidate_u*.pt`, cannot promote, and passes only the frozen gates in the approved design.
- The official short is exactly 100 updates and emits exactly `candidate_u000/u025/u050/u075/u100.pt`.
- Promotion remains nine zero-pair plus fifteen candidate workers, seeds `42/43/44`, 4000 steps per worker, with unchanged calibration and selection rules.
- Never reuse or overwrite v1/v2 artifacts; use new pilot-v3, short-v3, promotion-v3, and long-v3 directories.
- Long is exactly 3000 updates and may start only from an atomic promotion manifest with `accepted=true` and fully matching source/normalizer lineage.

---

## File Structure

- Create `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_lineage.py`: canonical SHA-256 calculation, deterministic reward/runtime bundle hash, source-lineage record creation, and fail-closed manifest validation.
- Create `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_pilot.py`: immutable pilot iteration records, exact gate evaluation, schema hash, and serializable pilot decision.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py`: resolve/validate `control_dt` and scale only the returned PPO reward.
- Modify `Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py`: enable empirical normalization and set initial std to `0.01`.
- Modify `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`: expose finite value loss in `IterationSummary` and retain fail-closed normalizer checkpoint loading.
- Modify `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py`: implement pilot/short/long stage contracts, pilot controller, source hashes, and pilot/promotion lineage checks.
- Modify `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py`: load checkpoints through normalized inference and publish worker source lineage.
- Modify `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py`: use the canonical normalized inference callable.
- Modify `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py`: validate short/worker lineage and propagate it into calibration and promotion manifests.
- Modify focused tests under `Go2Pvcnn/tests/`; do not alter Phase 5 acceptance thresholds.
- Create `notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md`: immutable command/result ledger for GPU acceptance.

### Task 1: Reward time scaling without diagnostic scaling

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py:164-215,609-721`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py`

**Interfaces:**
- Consumes: `env.cfg.sim.dt: float`, `env.cfg.decimation: int`, and `compute_residual_reward(signals).total: torch.Tensor`.
- Produces: `M1PandaArmMpcResidualRuntime.control_dt: float`; `compute_transition_reward() -> tuple[torch.Tensor, dict[str, float]]` whose reward is density times `control_dt` and whose diagnostics retain physical units.

- [ ] **Step 1: Add failing control-interval validation tests**

Add a minimal `_RuntimeCfgEnv` fixture and assertions equivalent to:

```python
class _RuntimeCfgEnv(_PhysicalEnv):
    def __init__(self, *, dt=0.0025, decimation=2):
        self.cfg = type("Cfg", (), {
            "sim": type("Sim", (), {"dt": dt})(),
            "decimation": decimation,
        })()

def test_runtime_resolves_exact_200_hz_control_dt():
    runtime = _make_physical_runtime(_RuntimeCfgEnv())
    assert runtime.control_dt == pytest.approx(0.005)

@pytest.mark.parametrize("dt,decimation", [(0.0, 2), (float("nan"), 2), (0.0025, 0)])
def test_runtime_rejects_invalid_control_dt(dt, decimation):
    with pytest.raises(ValueError, match="control interval"):
        _make_physical_runtime(_RuntimeCfgEnv(dt=dt, decimation=decimation))
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_wrapper.py -k 'control_dt or control_interval'`

Expected: FAIL because `control_dt` is not resolved or validated.

- [ ] **Step 3: Resolve and validate `control_dt` once in the runtime constructor**

Implement the constructor contract exactly:

```python
dt = float(env.cfg.sim.dt)
decimation = int(env.cfg.decimation)
control_dt = dt * decimation
if not math.isfinite(control_dt) or control_dt <= 0.0:
    raise ValueError("control interval must be finite and positive")
self.control_dt = control_dt
```

- [ ] **Step 4: Add a failing reward/diagnostic scale test**

Extend the physical runtime test so the same transition is evaluated at `control_dt=0.005`, then assert:

```python
expected_density = compute_residual_reward(expected_signals).total
reward, _ = runtime.compute_transition_reward()
assert torch.allclose(reward, expected_density * 0.005)
diagnostics = runtime.get_training_diagnostics()
assert diagnostics["wrench_error"] == pytest.approx(expected_raw_wrench_error)
assert diagnostics["samples"] == pytest.approx(1.0)
```

- [ ] **Step 5: Run the reward test and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_wrapper.py tests/test_m1_panda_arm_mpc_residual_reward.py`

Expected: FAIL only at the new `* 0.005` assertion.

- [ ] **Step 6: Scale only the returned PPO reward**

Change the reward assignment to:

```python
reward_density = compute_residual_reward(signals).total
reward = reward_density * self.control_dt
```

Do not multiply `raw_wrench_error`, normalized wrench error, counts, RMS accumulators, saturation, intervention, feasibility, or `_last_metrics`.

- [ ] **Step 7: Run focused GREEN tests**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_wrapper.py tests/test_m1_panda_arm_mpc_residual_reward.py`

Expected: all selected tests PASS.

- [ ] **Step 8: Commit only Task 1 files**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py
git commit -m "fix: scale phase6 reward by control interval"
```

### Task 2: Empirical normalization and canonical inference

**Files:**
- Modify: `Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py:8-43`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py:53-69,344-413,559-617`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py:70-160`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py:68-91`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`
- Modify: `Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py`

**Interfaces:**
- Consumes: RSL-RL `EmpiricalNormalization` and checkpoint keys `obs_norm_state_dict`, `critic_obs_norm_state_dict`.
- Produces: `IterationSummary.value_loss: float`; `runner.get_inference_policy(device) -> Callable[[torch.Tensor], torch.Tensor]`; normalized residual checkpoints that fail closed when either normalizer state is absent.

- [ ] **Step 1: Freeze the corrected PPO configuration in failing tests**

Update the config test to assert:

```python
assert cfg["empirical_normalization"] is True
assert cfg["policy"]["init_noise_std"] == pytest.approx(0.01)
assert cfg["algorithm"]["clip_min_std"] == pytest.approx(0.005)
assert cfg["algorithm"]["clip_max_std"] == pytest.approx(0.02)
assert cfg["algorithm"]["desired_kl"] == pytest.approx(0.01)
assert cfg["algorithm"]["kl_abort_threshold"] == pytest.approx(0.015)
```

Add static entrypoint assertions that eval/play contain `get_inference_policy(device=` and do not contain `policy = runner.alg.actor_critic` or `policy.act_inference(observations)`.

- [ ] **Step 2: Run config/entrypoint tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_train_static.py tests/test_m1_panda_arm_mpc_entrypoints_static.py`

Expected: FAIL on normalization, initial std, and raw inference bypass.

- [ ] **Step 3: Apply the configuration and inference changes**

Set:

```python
"empirical_normalization": True,
...
"init_noise_std": 0.01,
```

In eval/play use:

```python
runner.load(str(checkpoint), load_optimizer=False, keep_std=True)
policy = runner.get_inference_policy(device=args.device)
actions = policy(observations)
```

Change eval `_rollout` to accept a callable and invoke `policy(observations)`; retain exact zero tensors when `policy is None`.

- [ ] **Step 4: Add failing runner checkpoint and iteration-summary tests**

Create a normalized runner checkpoint, delete each normalizer key in turn, and assert `runner.load(...)` raises `KeyError` naming the missing key. Verify `get_inference_policy()` puts both normalizers in eval mode and does not update running statistics across inference calls. Extend the callback test to assert `summary.value_loss` equals the finite value returned by `alg.update()`.

- [ ] **Step 5: Run runner tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_rsl_runner_checkpoint_static.py tests/test_rsl_runner_iteration_callback.py`

Expected: FAIL because `IterationSummary` does not yet expose `value_loss`; missing normalizer state must already fail closed and is retained as a regression assertion.

- [ ] **Step 6: Publish finite value loss in `IterationSummary`**

Add:

```python
value_loss: float = 0.0
```

and populate it in the callback with:

```python
value_loss=_finite_float(mean_value_loss, label="value loss"),
```

Keep `get_inference_policy()` calling `eval_mode()` before returning the normalized callable. Do not introduce a raw-policy fallback.

- [ ] **Step 7: Run Task 2 GREEN tests**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_train_static.py tests/test_m1_panda_arm_mpc_entrypoints_static.py tests/test_rsl_runner_checkpoint_static.py tests/test_rsl_runner_iteration_callback.py`

Expected: all selected tests PASS.

- [ ] **Step 8: Commit Task 2 files**

```bash
git add Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py Go2Pvcnn/tests/test_rsl_runner_iteration_callback.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py
git commit -m "feat: normalize phase6 residual observations"
```

### Task 3: Canonical source-lineage contract

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_lineage.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_lineage.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py:20-160,290-350`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py:20-130`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py:20-190`

**Interfaces:**
- Produces: `ResidualSourcePaths(asset, config, reward, runtime)`, `sha256_file(path) -> str`, `pilot_schema_sha256() -> str`, `source_lineage(paths) -> dict[str, str]`, and `validate_source_lineage(document, paths) -> None`.
- Bundle encoding: SHA-256 of `json.dumps({"reward_sha256": reward_sha256, "runtime_sha256": runtime_sha256}, sort_keys=True, separators=(",", ":")).encode("utf-8")`, where both values are lowercase 64-character SHA-256 strings.

- [ ] **Step 1: Write pure failing lineage tests**

Test exact deterministic keys and fail-closed drift:

```python
lineage = source_lineage(paths)
assert lineage["runtime_sha256"] == sha256_file(paths.runtime)
payload = json.dumps(
    {"reward_sha256": lineage["reward_sha256"], "runtime_sha256": lineage["runtime_sha256"]},
    sort_keys=True, separators=(",", ":"),
).encode()
assert lineage["reward_runtime_bundle_sha256"] == hashlib.sha256(payload).hexdigest()
validate_source_lineage(lineage, paths)
paths.runtime.write_text("changed")
with pytest.raises(ValueError, match="runtime SHA"):
    validate_source_lineage(lineage, paths)
```

Also assert a missing bundle field is rejected and `pilot_schema_sha256()` is a stable 64-character lowercase hex string.

- [ ] **Step 2: Run lineage tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_lineage.py`

Expected: collection FAIL because the lineage module does not exist.

- [ ] **Step 3: Implement the focused lineage module**

Use frozen dataclasses and pure functions. `source_lineage()` must return paths plus `asset_sha256`, `config_sha256`, `reward_sha256`, `runtime_sha256`, `reward_runtime_bundle_sha256`, and `pilot_schema_sha256`. `validate_source_lineage()` must recompute all values, reject missing/non-string fields, and identify the mismatched label in its exception.

- [ ] **Step 4: Replace script-local hash implementations and add failing manifest assertions**

Import the canonical helpers from the new module in train/eval/promote. Extend static/promotion tests to require all five hashes in short, worker, noise calibration, promotion, and long validation payloads. Mutate `runtime_sha256` and `reward_runtime_bundle_sha256` independently and assert validation fails.

- [ ] **Step 5: Run focused tests and confirm RED at manifest boundaries**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_lineage.py tests/test_m1_panda_arm_mpc_residual_train_static.py tests/test_m1_panda_arm_mpc_residual_promote.py`

Expected: pure lineage tests PASS; manifest tests FAIL until scripts record and validate the new fields.

- [ ] **Step 6: Thread canonical lineage through train/eval/promote**

The train manifest starts with `**source_lineage(paths)`. Eval workers recompute and publish the same fields. Promotion validates the short before launching workers, validates every worker against the short/current paths, and copies the lineage into `noise_calibration.json` and `promotion_manifest.json`. Long validation invokes `validate_source_lineage()` on both short and promotion documents.

- [ ] **Step 7: Run Task 3 GREEN tests**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_lineage.py tests/test_m1_panda_arm_mpc_residual_train_static.py tests/test_m1_panda_arm_mpc_residual_promote.py`

Expected: all selected tests PASS.

- [ ] **Step 8: Commit Task 3 files**

```bash
git add Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_lineage.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_lineage.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py
git commit -m "feat: bind phase6 artifacts to runtime lineage"
```

### Task 4: Exact 10-update diagnostic pilot and short gate

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_pilot.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_pilot.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py:25-65,180-470`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`

**Interfaces:**
- Produces: `PilotIterationRecord.from_summary(summary)`, `PilotDecision(accepted, reasons, kl_abort_count, median_completed_mini_batches, median_value_loss)`, `evaluate_pilot(records) -> PilotDecision`, and `validate_pilot_manifest(path, paths) -> PilotLineage`.
- Consumes: exactly ten `IterationSummary` values with finite `value_loss`, optimizer diagnostics, completed rewards, and environment metrics.

- [ ] **Step 1: Write pure pilot-gate tests**

Build ten safe records and assert acceptance. Parameterize one mutation per frozen gate:

```python
assert evaluate_pilot(_records()).accepted is True
assert evaluate_pilot(_records(kl_aborts=4)).accepted is False
assert evaluate_pilot(_records(completed_batches=(5,) * 6 + (8,) * 4)).accepted is False
assert evaluate_pilot(_records(value_losses=(100.0,) * 10)).accepted is False
assert evaluate_pilot(_records(mpc_feasible_rate=0.989)).accepted is False
assert evaluate_pilot(_records(qp_feasible_rate=0.999)).accepted is False
assert evaluate_pilot(_records(four_contact_rate=0.999)).accepted is False
assert evaluate_pilot(_records(saturation_fraction_0=0.01)).accepted is False
assert evaluate_pilot(_records(active_action_std_min=0.0049)).accepted is False
```

Also assert nine or eleven records raise `ValueError`, NaN in any scalar/list/environment metric rejects, and reasons are deterministic.

- [ ] **Step 2: Run pilot tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_pilot.py`

Expected: collection FAIL because the pilot module does not exist.

- [ ] **Step 3: Implement the pure pilot decision module**

Use `statistics.median`, require update numbers `1..10`, require ten records, count `kl_aborted`, and evaluate these exact thresholds: hard failures `==0`; MPC `>=0.99`; QP/contact `==1.0`; each saturation `<0.01`; abort count `<=3`; median batches `>=6`; median value loss `<100`; std min/max inside `[0.005,0.02]`. Reject any non-finite optimizer, reward, or environment diagnostic.

- [ ] **Step 4: Add failing train-stage tests**

Assert parser/limits include `pilot`, pilot resolves only to 10, short requires `--pilot_manifest`, and a synthetic accepted pilot is rejected after asset/config/reward/runtime/schema drift. Use a fake runner callback to prove pilot publishes ten serialized summaries and no candidate checkpoint.

- [ ] **Step 5: Run train-stage tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_train_static.py -k 'pilot or candidate or cli'`

Expected: FAIL because `pilot` and `--pilot_manifest` are absent and the common controller always primes `candidate_u000.pt`.

- [ ] **Step 6: Implement distinct pilot orchestration**

Set `STAGE_LIMITS = {"zero": 10, "pilot": 10, "short": 100, "long": 3000}` and make pilot/short exact. Add `--pilot_manifest`. Before creating a short run directory, validate an accepted pilot with matching source and pilot-schema hashes. For pilot, do not call `ResidualTrainingSafetyController.prime()`: collect exactly ten detached records, evaluate once after `runner.learn`, and publish:

```python
{
    "status": "safe_complete",
    "accepted": False,
    "promotion_required": False,
    "pilot_accepted": decision.accepted,
    "optimizer_summaries": [asdict(record) for record in records],
    "pilot_decision": asdict(decision),
    "completed_iterations": 10,
}
```

Return exit code `0` only for an accepted pilot and a non-zero code for a completed-but-rejected pilot. A short manifest records the pilot path/hash and remains `accepted=false`, `promotion_required=true`.

- [ ] **Step 7: Run Task 4 GREEN tests**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_pilot.py tests/test_m1_panda_arm_mpc_residual_train_static.py`

Expected: all selected tests PASS.

- [ ] **Step 8: Commit Task 4 files**

```bash
git add Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_pilot.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_pilot.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py
git commit -m "feat: gate phase6 short behind diagnostic pilot"
```

### Task 5: Fail-closed v3 promotion and long lineage

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py:70-170`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py:100-335`
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py:70-170,300-470`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py`

**Interfaces:**
- Consumes: an accepted pilot lineage, a safe-complete 100-update short manifest, five normalized candidate checkpoints, 24 fresh worker manifests, and an accepted promotion manifest.
- Produces: atomic `noise_calibration.json`, `promotion_manifest.json`, `model_best.pt`, and a validated `PromotionLineage` that may seed long.

- [ ] **Step 1: Extend promotion fixtures with normalized/source lineage and write failing drift tests**

Synthetic checkpoints must contain both normalizer dictionaries. Synthetic short manifests must contain accepted pilot path/hash, all source hashes, `completed_iterations=100`, and exactly five candidates. Assert promotion rejects wrong worker `runtime_sha256`, wrong bundle hash, missing normalizer keys, wrong completed iterations, reused/non-empty worker outputs, and short manifests without accepted pilot lineage.

- [ ] **Step 2: Run promotion tests and confirm RED**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_promote.py tests/test_m1_panda_arm_mpc_residual_train_static.py -k 'promotion or long or lineage or normalizer'`

Expected: FAIL at each newly required v3 boundary.

- [ ] **Step 3: Harden short and worker validation**

Require short `completed_iterations == requested_iterations == 100`, exact candidate update set, accepted pilot SHA, and matching source lineage. Before each worker launch, reject an existing output path. In `_validate_worker`, validate source lineage plus checkpoint SHA and require candidate checkpoint normalizer keys. Preserve the exact worker schedule `9 + 15` and existing tolerance functions unchanged.

- [ ] **Step 4: Harden promotion and long validation**

Promotion copies all source/pilot hashes and publishes `accepted=true` only after all workers validate and a candidate is selected. `validate_promotion_manifest()` recomputes promotion/short/checkpoint/source hashes, verifies `model_best.pt` includes both normalizer states, verifies the promoted checkpoint belongs to the five short candidates, and rejects diagnostic pilot/short/rejected promotion inputs.

- [ ] **Step 5: Run Task 5 GREEN tests**

Run: `cd /home/xk/coding/M1/Go2Pvcnn && pytest -q tests/test_m1_panda_arm_mpc_residual_promote.py tests/test_m1_panda_arm_mpc_residual_train_static.py tests/test_m1_panda_arm_mpc_entrypoints_static.py`

Expected: all selected tests PASS; worker call counts remain nine and fifteen.

- [ ] **Step 6: Run expanded CPU regression and static verification**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q \
  tests/test_m1_panda_arm_mpc_residual_reward.py \
  tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  tests/test_m1_panda_arm_mpc_residual_guard.py \
  tests/test_m1_panda_arm_mpc_residual_promotion.py \
  tests/test_m1_panda_arm_mpc_residual_promote.py \
  tests/test_m1_panda_arm_mpc_residual_train_static.py \
  tests/test_m1_panda_arm_mpc_residual_lineage.py \
  tests/test_m1_panda_arm_mpc_residual_pilot.py \
  tests/test_rsl_runner_checkpoint_static.py \
  tests/test_rsl_runner_iteration_callback.py \
  tests/test_m1_panda_arm_mpc_entrypoints_static.py
python -m compileall -q agent go2_pvcnn scripts rsl_rl/rsl_rl
cd /home/xk/coding/M1
git diff --check
```

Expected: pytest PASS, compileall exits 0, and `git diff --check` emits no output.

- [ ] **Step 7: Commit Task 5 files**

```bash
git add Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py
git commit -m "feat: enforce phase6 v3 promotion lineage"
```

### Task 6: GPU0 acceptance, conditional long launch, and monitoring ledger

**Files:**
- Create: `notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: current Phase 5 probe command/acceptance contract, accepted pilot manifest, fresh short manifest, promotion driver, and accepted promotion manifest.
- Produces: evidence-backed Phase 6 v3 status and a running/completed long manifest; no long process exists when promotion is rejected.

- [ ] **Step 1: Record immutable artifact roots and GPU ownership**

Use these fresh absolute roots and reject them if they already exist or are non-empty:

```text
/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3
/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_scale_norm_v3
/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_scale_norm_v3
```

Record date, commit, `CUDA_VISIBLE_DEVICES=0`, exact commands, PIDs, stdout/stderr logs, and manifest paths in the execution log before each launch.

- [ ] **Step 2: Re-run the unchanged Phase 5 GPU0 regression**

Run without changing gates or controller configuration:

```bash
cd /home/xk/coding/M1
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 42 --headless
```

Expected: exit 0 and the generated Phase 5 acceptance document contains `accepted=true`, zero hard failures, MPC feasibility at least `0.99`, and exact QP/four-contact rates of `1.0`. If it fails, stop before pilot and record the evidence.

- [ ] **Step 3: Launch the exact pilot on GPU0**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 python scripts/m1_panda_arm_mpc_residual_train.py \
  --stage pilot \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3 \
  --num_envs 8 --max_iterations 10 --seed 42 \
  --device cuda:0 --headless
```

Expected: manifest `status=safe_complete`, `completed_iterations=10`, ten optimizer summaries, `pilot_accepted=true`, `accepted=false`, `promotion_required=false`, no `candidate_u*.pt`, and all approved pilot medians/counts within bounds.

- [ ] **Step 4: Launch the official fresh 100-update short only after pilot acceptance**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/m1_panda_arm_mpc_residual_train.py \
  --stage short \
  --pilot_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3/run_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_scale_norm_v3 \
  --num_envs 8 \
  --max_iterations 100 --seed 42 --device cuda:0 --headless
```

Expected: `safe_complete`, 100/100, `accepted=false`, `promotion_required=true`, and exactly five candidate checkpoints with saved actor/critic normalizer state.

- [ ] **Step 5: Execute all 24 fresh fixed-condition workers**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/m1_panda_arm_mpc_residual_promote.py \
  --short_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_scale_norm_v3/run_manifest.json \
  --device cuda:0 --headless
```

Expected: nine zero-pair and fifteen candidate JSON files, each `status=complete`, seeds exactly `42/43/44`, steps exactly `4000`, and matching checkpoint/source hashes.

- [ ] **Step 6: Enforce the conditional branch**

Parse `promotion_manifest.json` with Python/JSON rather than shell text matching. If `accepted` is not exactly `true`, record rejection reasons and verify no long-v3 process/directory was started. If it is exactly `true`, verify `model_best.pt` SHA and both normalizer state dictionaries before continuing.

- [ ] **Step 7: Launch and monitor the 3000-update long only from accepted promotion**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/m1_panda_arm_mpc_residual_train.py \
  --stage long \
  --promotion_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_scale_norm_v3/promotion_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_scale_norm_v3 \
  --num_envs 8 \
  --max_iterations 3000 --seed 42 --device cuda:0 --headless
```

Monitor process liveness, manifest status, completed iterations, stop reason, finite optimizer diagnostics, hard failures, MPC/QP/contact rates, checkpoint publication, GPU memory, and log errors. Do not claim completion while the process is merely running; on interruption, diagnose from the atomic manifest/log before any restart.

- [ ] **Step 8: Update project notes with the authoritative outcome**

Link the execution log from `notes/log/index.md`, update T400 with pilot/short/promotion/long status, and explicitly record whether Phase 6 is accepted, rejected before long, currently training, safety-stopped, or completed.

- [ ] **Step 9: Run final verification and commit documentation**

Run:

```bash
cd /home/xk/coding/M1
git diff --check
git status --short
```

Expected: no whitespace errors; status contains only understood Phase 5/6 changes and runtime artifacts remain untracked/ignored as intended.

```bash
git add notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md notes/log/index.md notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: record phase6 v3 acceptance"
```

## Completion Criteria

- Reward returned to PPO is exactly the unchanged total density times validated `control_dt=0.005`; physical diagnostics are unchanged.
- Empirical observation normalization is enabled, checkpointed, required, restored, and used by eval/play.
- Initial action std is `0.01`; all frozen PPO settings remain unchanged.
- Pilot completes exactly ten updates, records all summaries, emits no candidate checkpoints, and passes every frozen diagnostic gate.
- Official short is impossible without an accepted matching pilot and emits exactly five normalized candidates after 100 safe updates.
- All source/runtime/bundle/pilot-schema hashes validate across pilot, short, workers, promotion, and long.
- Fresh 24-worker promotion is authoritative; rejected promotion starts no long run.
- Accepted promotion alone seeds a monitored 3000-update long run on GPU0.
- CPU tests, compile checks, `git diff --check`, and unchanged Phase 5 GPU regression pass with recorded evidence.
