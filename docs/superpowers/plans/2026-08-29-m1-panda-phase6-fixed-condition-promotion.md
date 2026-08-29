# M1 + Panda Phase 6 Fixed-Condition Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a genuinely improved, safety-eligible 8D residual checkpoint by fixing wrench reward scaling, separating online safety from offline performance selection, and promoting candidates only through noise-calibrated seeds 42/43/44 evaluation.

**Architecture:** The runtime computes a bounded dimensionless wrench reward while retaining raw physical diagnostics. Training writes exact completed-update candidates and performs only fail-closed safety stopping. A pure promotion module calibrates PhysX noise and applies tolerance-aware stability-first comparison; a process driver runs every zero pair and candidate/seed worker in a fresh Isaac Sim process, atomically publishes the sole accepted checkpoint, and provides the only manifest authorized to start long training.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/Isaac Sim, RSL-RL PPO, pytest, JSON/SHA-256 manifests, subprocess isolation, GPU 0.

## Global Constraints

- Preserve the 8D normalized action order `[Fx,Fy,Fz,Mx,My,Mz,delta_height,delta_stance]` and frozen 103D observation.
- Preserve Phase 5 hard gates, WBC/QP projection, physical/slew limits, seeds `42/43/44`, and `4000` evaluation steps.
- Use `WholeBodyResidualCfg.physical_limits[:6] == (30,30,50,15,15,8)` as the only wrench scale source.
- Short training executes exactly `100` PPO updates and writes `candidate_u000/u025/u050/u075/u100.pt` by completed-update count.
- Training diagnostics may stop unsafe/non-finite runs but may not rank different trajectory phases or publish `model_best.pt`.
- Each zero-vs-zero pair and candidate/seed evaluation runs in a fresh Isaac Sim process on GPU 0.
- Noise tolerance is `max(engineering_floor, 2 * max(abs(zero-zero delta)))` over three pairs for each of three seeds.
- Long training stays blocked unless safe short and accepted promotion manifests have matching asset/config/reward/checkpoint SHA-256 values.
- Preserve unrelated dirty-worktree changes; stage and commit only files named by each task.

---

### Task 1: Dimensionless bounded wrench reward

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_residual_wbc_wrapper.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py`

**Interfaces:**
- Consumes: `WholeBodyResidualCfg.physical_limits`, `measured` and `predicted` tensors shaped `(N,6)`.
- Produces: `normalized_wrench_error(error_b, scale) -> Tensor[N]`, `M1PandaResidualWbcController.wrench_scale`, `ResidualRewardSignals.normalized_wrench_error`, and diagnostic `normalized_wrench_error` while preserving raw `wrench_error`.

- [ ] **Step 1: Write failing pure reward tests**

```python
def test_wrench_error_uses_physical_channel_scales():
    error = torch.tensor([[30.0, 0.0, 0.0, 0.0, 0.0, 8.0]])
    scale = torch.tensor([30.0, 30.0, 50.0, 15.0, 15.0, 8.0])
    assert torch.allclose(
        normalized_wrench_error(error, scale), torch.tensor([2.0**0.5])
    )


def test_wrench_tracking_penalty_is_bounded():
    signals = _signals(normalized_wrench_error=torch.full((2,), 1.0e9))
    reward = compute_residual_reward(signals)
    assert torch.allclose(reward.tracking_penalty, torch.full((2,), -0.2))
```

- [ ] **Step 2: Run RED**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py -q
```

Expected: FAIL because the normalization function and renamed signal do not exist.

- [ ] **Step 3: Implement normalization and bounded term**

```python
def normalized_wrench_error(error_b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not isinstance(error_b, torch.Tensor) or error_b.ndim != 2 or error_b.shape[1] != 6:
        raise ValueError("error_b must have shape (N,6)")
    if not isinstance(scale, torch.Tensor) or scale.shape != (6,):
        raise ValueError("scale must have shape (6,)")
    if scale.dtype != error_b.dtype or scale.device != error_b.device:
        raise ValueError("scale dtype/device must match error_b")
    if not torch.isfinite(error_b).all().item() or not torch.isfinite(scale).all().item():
        raise ValueError("wrench tensors must be finite")
    if torch.any(scale <= 0.0).item():
        raise ValueError("scale must be positive")
    return torch.linalg.vector_norm(error_b / scale, dim=1)
```

Rename the signal field and use:

```python
tracking_penalty = (
    -0.2 * torch.tanh(signals.normalized_wrench_error)
    -0.5 * signals.wheel_slip
)
```

- [ ] **Step 4: Expose scale and retain raw/normalized diagnostics**

```python
@property
def wrench_scale(self) -> torch.Tensor:
    return torch.tensor(
        self._composer.cfg.physical_limits[:6], dtype=self.dtype, device=self.device
    )
```

In the runtime, pass the normalized norm into reward signals, keep raw norm accumulation in `wrench_error`, and add normalized accumulation to `normalized_wrench_error`.

- [ ] **Step 5: Run GREEN**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py -q
```

Expected: PASS; both wrench diagnostic forms are finite.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_residual_wbc_wrapper.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py
git commit -m "fix: normalize residual wrench reward"
```

### Task 2: Pure calibration and promotion semantics

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_promotion.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promotion.py`
- Modify: `Go2Pvcnn/go2_pvcnn/training/__init__.py`

**Interfaces:**
- Consumes: `ResidualEvalMetrics`.
- Produces: `calibrate_tolerances`, `compare_with_tolerances`, `evaluate_candidate`, `select_promoted_candidate`, `PromotionDecision`, and `ENGINEERING_FLOORS`.

- [ ] **Step 1: Write failing calibration tests**

```python
def test_calibration_uses_twice_max_delta_or_floor():
    pairs = [(_metrics(roll_pitch_rms=0.0010), _metrics(roll_pitch_rms=0.00106))] * 9
    tolerances = calibrate_tolerances(pairs)
    assert tolerances["roll_pitch_rms"] == pytest.approx(0.00012)
    assert tolerances["ee_position_error"] == pytest.approx(5.0e-5)


def test_zero_equivalence_is_not_improvement():
    baseline = _metrics()
    assert compare_with_tolerances(
        baseline, baseline, _tolerances()
    ) is MetricComparison.EQUIVALENT
```

- [ ] **Step 2: Run RED**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promotion.py -q
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement exact floors and calibration**

```python
ENGINEERING_FLOORS = {
    "roll_pitch_rms": 1.0e-4,
    "base_height_rms": 2.0e-5,
    "ee_position_error": 5.0e-5,
    "ee_orientation_error": 5.0e-5,
    "wrench_error": 0.1,
    "slip": 2.0e-5,
    "intervention_ratio": 1.0 / 4000.0,
}


def calibrate_tolerances(zero_pairs):
    if len(zero_pairs) != 9:
        raise ValueError("noise calibration requires exactly nine zero pairs")
    return {
        name: max(
            floor,
            2.0 * max(abs(getattr(a, name) - getattr(b, name)) for a, b in zero_pairs),
        )
        for name, floor in ENGINEERING_FLOORS.items()
    }
```

Reject wrong counts/types and non-finite metrics/tolerances.

- [ ] **Step 4: Write failing stability-first tests**

```python
def test_ee_may_improve_when_stability_is_equivalent():
    baseline = _metrics(roll_pitch_rms=0.0010, ee_position_error=0.0100)
    candidate = _metrics(roll_pitch_rms=0.00105, ee_position_error=0.0090)
    assert compare_with_tolerances(candidate, baseline, _tolerances()) is MetricComparison.BETTER


def test_stability_regression_beats_ee_improvement():
    baseline = _metrics(roll_pitch_rms=0.0010, ee_position_error=0.0100)
    candidate = _metrics(roll_pitch_rms=0.0012, ee_position_error=0.0010)
    assert compare_with_tolerances(candidate, baseline, _tolerances()) is MetricComparison.WORSE


def test_all_three_seeds_must_pass_hard_gates():
    results = _three_seed_results()
    results[44] = (results[44][0], _metrics(hard_failure_count=1))
    assert not evaluate_candidate(results, _tolerances()).accepted
```

- [ ] **Step 5: Implement immutable decisions and tournament tie-break**

```python
@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    aggregate_baseline: ResidualEvalMetrics
    aggregate_candidate: ResidualEvalMetrics
    decisive_metric: str | None


@dataclass(frozen=True)
class PromotedCandidate:
    checkpoint: str
    completed_updates: int
    sha256: str
    decision: PromotionDecision
```

Use rank order hard failure, roll/pitch, height, EE position, EE orientation, intervention. Require all seeds, no raw wrench/slip regression beyond tolerance, at least one decisive improvement, and lower completed-update count for equivalent candidates.

Aggregate `hard_failure_count` by integer sum; average every continuous metric and each saturation channel. This preserves the `ResidualEvalMetrics` integer contract while making any seed failure visible.

- [ ] **Step 6: Run GREEN and guard regression**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promotion.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_promotion.py \
  Go2Pvcnn/go2_pvcnn/training/__init__.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promotion.py
git commit -m "feat: add fixed-condition residual promotion rules"
```

### Task 3: Safe 100-update candidate production

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py`
- Modify: `Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_guard.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py`

**Interfaces:**
- Consumes: post-update runner callback and atomic checkpoint helper.
- Produces: `ResidualTrainingSafetyController`, exact completed-update candidate paths, safe-complete non-promotional manifest, and long promotion-lineage validation.

- [ ] **Step 1: Write failing timing/no-online-best tests**

```python
def test_short_stage_is_exactly_one_hundred_updates():
    assert resolve_max_iterations("short", None) == 100


def test_candidate_names_use_completed_updates(tmp_path, fake_runner):
    controller = ResidualTrainingSafetyController(tmp_path)
    controller.prime(fake_runner)
    assert (tmp_path / "candidate_u000.pt").is_file()
    controller.on_iteration(fake_runner, _summary(iteration=24))
    assert (tmp_path / "candidate_u025.pt").is_file()


def test_online_controller_never_publishes_best(tmp_path, fake_runner):
    controller = ResidualTrainingSafetyController(tmp_path)
    controller.prime(fake_runner)
    for iteration in range(100):
        controller.on_iteration(fake_runner, _summary(iteration=iteration))
    assert not (tmp_path / "model_best.pt").exists()
```

- [ ] **Step 2: Run RED**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py -q
```

Expected: FAIL because short is not 100 and safety controller is absent.

- [ ] **Step 3: Implement safety-only candidate controller**

```python
CANDIDATE_UPDATES = (0, 25, 50, 75, 100)

def prime(self, runner) -> None:
    _atomic_save(runner, self.run_dir / "candidate_u000.pt")

def on_iteration(self, runner, summary) -> str | None:
    self._validate_finite_optimizer(summary)
    stop_reason = self.safety_guard.observe(self._metrics(summary))
    completed_updates = int(summary.iteration) + 1
    if completed_updates in CANDIDATE_UPDATES:
        _atomic_save(runner, self.run_dir / f"candidate_u{completed_updates:03d}.pt")
    return stop_reason
```

Add a pure safety guard with no rank, best, patience, or rollback state:

```python
class ResidualTrainingSafetyGuard:
    def observe(self, metrics: ResidualEvalMetrics) -> str | None:
        if not isinstance(metrics, ResidualEvalMetrics):
            raise TypeError("metrics must be ResidualEvalMetrics")
        if metrics.hard_failure_count > 0:
            return "hard_failure"
        if metrics.mpc_feasible_rate < 0.99:
            return "mpc_infeasible"
        if metrics.qp_feasible_rate < 1.0:
            return "qp_infeasible"
        if metrics.four_contact_rate < 1.0:
            return "lost_wheel_contact"
        if max(metrics.saturation_fraction) >= 0.01:
            return "residual_saturation"
        return None
```

Keep tilt/EE bounds as offline eligibility gates. Training stops immediately only on these physical/rate/saturation conditions and non-finite optimizer diagnostics.

- [ ] **Step 4: Make short completion explicitly non-promotional**

```python
manifest.update({
    "status": "safe_complete" if result.stop_reason is None else "safety_stopped",
    "accepted": False,
    "promotion_required": True,
    "completed_iterations": result.completed_iterations,
    "candidate_checkpoints": candidate_records_with_sha256,
})
```

Require all five candidates for safe completion and never write `model_best.pt`.

- [ ] **Step 5: Validate promotion lineage for long**

Add `--promotion_manifest`. Require accepted promotion, its referenced safe short manifest, matching asset/config/reward/checkpoint hashes, and an 8D checkpoint. A legacy `--short_manifest` alone fails clearly.

Record `reward_path`, `reward_sha256`, `config_path`, and `config_sha256` in short and promotion manifests; long validation recomputes all four before constructing Isaac.

- [ ] **Step 6: Run GREEN**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py \
  Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py -q
```

Expected: PASS, including pre-update `u000` and post-update timing.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py \
  Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_guard.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py
git commit -m "fix: separate residual safety and checkpoint selection"
```

### Task 4: One-seed worker and process-isolated promotion driver

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py`
- Create: `Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py`

**Interfaces:**
- Consumes: safe short manifest, five candidates, pure promotion functions, GPU device.
- Produces: nine calibration JSONs, fifteen candidate/seed JSONs, `noise_calibration.json`, atomic `promotion_manifest.json`, and `model_best.pt` only after acceptance.

- [ ] **Step 1: Write failing one-seed worker tests**

```python
def test_worker_accepts_exactly_one_seed():
    parser = build_arg_parser(include_app_launcher_args=False)
    args = parser.parse_args(["--mode", "zero-pair", "--seed", "42", "--steps", "4000"])
    assert args.seed == 42


def test_worker_has_no_plural_seeds_option():
    options = {
        action.dest
        for action in build_arg_parser(include_app_launcher_args=False)._actions
    }
    assert "seed" in options and "seeds" not in options
```

- [ ] **Step 2: Run worker RED**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py -q
```

Expected: FAIL because evaluator still accepts plural seeds.

- [ ] **Step 3: Refactor evaluator into a worker**

Use `--mode {zero-pair,candidate}`, one `--seed`, `--steps 4000`, and atomic output. In zero-pair mode run two exact-zero rollouts; in candidate mode run zero then checkpoint inference. Record checkpoint SHA in candidate mode. Return nonzero only for execution/integrity failure, not for a performance rejection.

- [ ] **Step 4: Write failing driver tests with fake worker runner**

```python
def test_driver_launches_exact_worker_count(tmp_path):
    runner = RecordingWorkerRunner()
    run_promotion(_safe_run(tmp_path), worker_runner=runner)
    assert runner.zero_pair_calls == 9
    assert runner.candidate_calls == 15


def test_driver_fails_closed_on_checkpoint_sha_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="checkpoint SHA"):
        run_promotion(_safe_run(tmp_path), worker_runner=RecordingWorkerRunner(checkpoint_sha="wrong"))


def test_equivalent_candidates_never_publish_best(tmp_path):
    manifest = run_promotion(_safe_run(tmp_path), worker_runner=EquivalentWorkerRunner())
    assert not manifest["accepted"]
    assert not (tmp_path / "model_best.pt").exists()
```

- [ ] **Step 5: Implement isolated worker runner and atomic promotion**

```python
command = [
    sys.executable,
    str(ROOT / "scripts/m1_panda_arm_mpc_residual_eval.py"),
    "--mode", mode,
    "--seed", str(seed),
    "--steps", "4000",
    "--device", args.device,
    "--headless",
    "--output_json", str(output),
]
if checkpoint is not None:
    command.extend(("--checkpoint", str(checkpoint)))
subprocess.run(
    command,
    cwd=ROOT.parent,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    check=True,
)
```

Do not use `shell=True`. Validate every child manifest, calibrate tolerances, select the candidate, atomically copy only an accepted source, and verify destination SHA.

- [ ] **Step 6: Run driver GREEN**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py -q
```

Expected: PASS without importing Isaac in fake-runner tests.

- [ ] **Step 7: Commit**

```bash
git add Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py
git commit -m "feat: promote residual checkpoints in isolated evaluations"
```

### Task 5: CPU regression, Phase 5 gate, and operator documentation

**Files:**
- Modify: `docs/superpowers/runbooks/2026-08-28-m1-panda-arm-mpc-residual-training.md`
- Create: `notes/log/2026-08-29-m1-panda-phase6-fixed-condition-implementation.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: final train/eval/promote CLI and manifests.
- Produces: exact run commands and CPU/Phase 5 evidence.

- [ ] **Step 1: Run focused Phase 6 tests**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_guard.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promotion.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_promote.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py -q
```

Expected: all PASS.

- [ ] **Step 2: Run RNE/coordination/runner regression**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc.py \
  Go2Pvcnn/tests/test_m1_panda_recursive_dynamics.py \
  Go2Pvcnn/tests/test_m1_panda_sensor_calibrated_wrench.py \
  Go2Pvcnn/tests/test_m1_panda_joint_torque_wrench.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py \
  Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py -q
```

Expected: all PASS.

- [ ] **Step 3: Compile changed Python**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  Go2Pvcnn/go2_pvcnn/tasks/mdp/m1_panda_arm_mpc_residual.py \
  Go2Pvcnn/go2_pvcnn/training/m1_panda_arm_mpc_residual_promotion.py \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_eval.py \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py
```

Expected: exit 0 without output.

- [ ] **Step 4: Re-run Phase 5 seed 42 gate**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seed 42 --headless \
  --output_json Go2Pvcnn/logs/m1_panda_arm_mpc_probe/phase5_regression_s42.json
```

Expected: accepted, zero hard failures, MPC `>=0.99`, QP/contact `1.0`, and wrench-direction gates unchanged.

- [ ] **Step 5: Update runbook and evidence**

Document exact short/promotion/long/play commands, GPU selection, process isolation, output paths, exit semantics, and the distinction between `safe_complete` and `accepted`.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/superpowers/runbooks/2026-08-28-m1-panda-arm-mpc-residual-training.md \
  notes/log/2026-08-29-m1-panda-phase6-fixed-condition-implementation.md \
  notes/log/index.md notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: record fixed-condition residual workflow"
```

### Task 6: GPU0 short training, tournament, and conditional long launch

**Files:**
- Modify: `notes/log/2026-08-29-m1-panda-phase6-fixed-condition-implementation.md`
- Modify: `notes/log/index.md`

**Interfaces:**
- Consumes: Tasks 1--5 outputs.
- Produces: safe short run, calibration/tournament artifacts, accepted promotion or evidence-backed rejection, and long PID only after acceptance.

- [ ] **Step 1: Run fresh 100-update short training**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage short --device cuda:0 --num_envs 1 --max_iterations 100 \
  --seed 42 --headless \
  --run_dir Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_fixed_condition_v1
```

Expected: `safe_complete`, `accepted=false`, five candidate checkpoints and no `model_best.pt`.

- [ ] **Step 2: Run calibration and candidate tournament**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py \
  --short_manifest Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_fixed_condition_v1/run_manifest.json \
  --device cuda:0 --headless
```

Expected: 9 calibration workers plus 15 candidate workers, each in a fresh process.

- [ ] **Step 3: Validate artifact lineage**

```bash
/home/xk/miniconda3/envs/go2/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_fixed_condition_v1")
promotion = json.loads((root / "promotion_manifest.json").read_text())
assert len(list((root / "noise_calibration").glob("seed_*_pair_*.json"))) == 9
assert len(list((root / "candidate_eval").glob("candidate_u*/seed_*.json"))) == 15
if promotion["accepted"]:
    digest = hashlib.sha256((root / "model_best.pt").read_bytes()).hexdigest()
    assert digest == promotion["best_checkpoint_sha256"]
print(promotion["accepted"], promotion.get("best_completed_updates"))
PY
```

Expected: assertions pass. Rejection is valid evidence and must not be bypassed.

- [ ] **Step 4: Launch long only when accepted**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage long --device cuda:0 --num_envs 1 --max_iterations 3000 \
  --seed 42 --headless \
  --promotion_manifest Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_fixed_condition_v1/promotion_manifest.json \
  --run_dir Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_fixed_condition_v1
```

Run this command only if promotion has `accepted=true`. Otherwise record decisive metrics and stop without weakening thresholds.

- [ ] **Step 5: Record first healthy update or rejection**

Append exact command, commit, PID, checkpoint SHA, tolerances, candidate table, decision, exit code, GPU, and first optimizer/safety diagnostics to the implementation log.

- [ ] **Step 6: Commit final evidence, excluding generated artifacts**

```bash
git add notes/log/2026-08-29-m1-panda-phase6-fixed-condition-implementation.md \
  notes/log/index.md
git commit -m "docs: record phase6 fixed-condition acceptance"
```

Do not add `Go2Pvcnn/logs/` checkpoints or generated JSON to Git.
