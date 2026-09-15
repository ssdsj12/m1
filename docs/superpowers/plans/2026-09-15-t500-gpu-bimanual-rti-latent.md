# T500 GPU Bimanual RTI + Fingertip Prior + Latent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slow online CPU teacher hot path with a fixed-shape GPU SQP-RTI teacher, use a frozen DexManipNet-derived five-fingertip prior only as a defeasible soft cost, and run the complete `[25, 43] -> z -> action` latent path on GPU without changing the public T400/T500 contracts.

**Architecture:** Keep the existing CPU `float64` implementation as the reference and final fallback. Add a private CUDA `float32` package that owns persistent workspaces, shifts the last accepted 25-node horizon, performs one Gauss-Newton RTI step, solves the structured horizon problem, evaluates fixed line-search candidates in parallel, and atomically publishes a complete 43-DOF horizon. The expert prior predicts natural palm-relative five-tip velocity distributions for only the first 0.20 seconds; true left/right O6 Jacobians map that distribution into a soft RTI cost. A separate full-action encoder compresses each accepted horizon to `z` for high-rate student control.

**Tech Stack:** Python 3.11, PyTorch 2.7/CUDA 12.8, Triton, Isaac Sim 5.1, NumPy, pytest, JSONL/SHA-256 manifests, optional isolated OSQP 1.x CUDA comparison backend.

## Global Constraints

- Preserve all existing public CPU `float64` dataclasses, shapes, joint order, state-machine behavior, safety thresholds, T400/T500 contracts, and last-safe atomic fallback.
- Production horizon is exactly `H=25`; tree padding to 32 must be neutral and must never leak into a public result.
- The teacher must reconstruct the full `[25, 43]` effective action before encoding `z`; it must not independently command all 59 articulation coordinates.
- The fingertip model receives only the 42-D natural-motion input: five palm-relative positions, five velocities, five contact bits, and seven phase values. It receives no task ID, object target, palm target, or semantic label.
- The frozen prior emits four diagonal-Gaussian components over `[20, 5, 3]` velocities at 100 Hz. Nodes after 0.20 seconds have exactly zero expert weight; conflicts with hard constraints mask or reduce the prior.
- Explicit unavailable backends fail before Isaac scene creation. `auto` may select only a benchmarked, manifest-recorded backend.
- Do not install or upgrade packages inside the existing `go2` environment while validating OSQP CUDA.
- Training, performance-sensitive Play, and formal GPU0 validation are mutually exclusive workloads.
- Every implementation task uses red-green-refactor, records focused evidence, updates `notes/index.md`, `notes/todo.md`, `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`, `notes/log/index.md`, and one dated focused log when the task changes project status.

---

## Task 1: Qualify `run_e`, train the expert ensemble, and distill the frozen prior

**Files:**

- Modify: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_artifact.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_distill.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/artifact.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_train_fingertip_expert.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_distill_fingertip_prior.py`
- Create: `notes/log/2026-09-15-t500-run-e-expert-artifact.md`
- Modify: the five note/index files listed in Global Constraints

- [ ] **Step 1: Add failing production-corpus and metadata-pin tests.**

Add tests which load `data/external/dexmanipnet/converted/run_e/aggregate_manifest.json`, recompute its aggregate hash, require 241 shards and 984,641 summed `samples`, and reject a student artifact whose external metadata SHA or teacher-ensemble SHA differs by one nibble.

```python
def test_run_e_manifest_is_the_frozen_training_corpus():
    manifest = verify_aggregate_manifest(RUN_E_MANIFEST)
    assert manifest["aggregate_sha256"] == RUN_E_SHA256
    assert len(manifest["shards"]) == 241
    assert sum(int(row["samples"]) for row in manifest["shards"]) == 984_641

def test_student_rejects_mismatched_external_metadata_sha(tmp_path):
    paths = write_valid_student_fixture(tmp_path)
    mutate_one_sha_nibble(paths.metadata)
    with pytest.raises(ValueError, match="metadata sha256"):
        load_frozen_student(paths.artifact, paths.metadata)
```

- [ ] **Step 2: Run the focused tests and observe RED.**

Run: `cd Go2Pvcnn && PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_artifact.py tests/test_m1_bimanual_expert_prior_distill.py`

Expected: failures for the absent frozen `run_e` constants and/or missing external metadata pin validation.

- [ ] **Step 3: Implement strict corpus and artifact identity validation.**

Add immutable constants for aggregate SHA `dfaa213a89d8a87b267ffd7ed9dc69d5a3f8582204e79a11d575d30140a57c7c`, expected shard/sample counts, the approved five-member teacher seeds `42..46`, and exact student gate fields. Fail closed on missing, malformed, non-finite, or mismatched identities.

- [ ] **Step 4: Run tests GREEN, then train on GPU0 without Play running.**

```bash
cd Go2Pvcnn
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_train_fingertip_expert.py \
  --dataset-manifest data/external/dexmanipnet/converted/run_e/aggregate_manifest.json \
  --output-dir data/external/dexmanipnet/models/expert_run_e \
  --epochs 200 --batch-size 128 --learning-rate 0.001 \
  --member-seeds 42,43,44,45,46 --hidden 512,512,512 --device cuda:0

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_distill_fingertip_prior.py \
  --dataset-manifest data/external/dexmanipnet/converted/run_e/aggregate_manifest.json \
  --ensemble-dir data/external/dexmanipnet/models/expert_run_e \
  --output-dir data/external/dexmanipnet/models/student_run_e \
  --samples-per-state 8 --epochs 200 --batch-size 128 \
  --learning-rate 0.001 --seed 42 --hidden 64,64 --device cuda:0
```

Expected: all five teacher members finish; distillation reports NLL delta `<=0.05`, endpoint ratio `<=1.05`, first-step and endpoint improvement `>=0.10`, and runtime P99 `<2 ms` on the pinned GPU. If a gate fails, retain the artifact as rejected evidence and do not enable expert-guided Play.

- [ ] **Step 5: Write the external metadata SHA pin and project evidence.**

Record exact dataset, source, ensemble, student artifact, metadata, CUDA, PyTorch, GPU, command, duration, and gate values. Store only small metadata in Git; keep large checkpoints under the documented external artifact root.

- [ ] **Step 6: Commit Task 1.**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/artifact.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_train_fingertip_expert.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_distill_fingertip_prior.py \
  Go2Pvcnn/tests/test_m1_bimanual_expert_prior_artifact.py \
  Go2Pvcnn/tests/test_m1_bimanual_expert_prior_distill.py notes
git commit -m "feat: qualify run-e fingertip prior"
```

## Task 2: Freeze private GPU backend contracts and CLI selection

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/config.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/contracts.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`

- [ ] **Step 1: Add RED tests for dimensions, devices, and backend parsing.**

```python
def test_gpu_trajectory_contract_is_private_float32_cuda():
    traj = GpuRtiTrajectory.zeros(batch=2, device="cuda:0")
    assert traj.action.shape == (2, 25, 43)
    assert traj.action.dtype == torch.float32
    assert traj.action.device.type == "cuda"

@pytest.mark.parametrize("flag", ["reference-cpu", "bimanual-rti-cuda", "auto"])
def test_mpc_backend_choices(flag):
    assert parse_play_args(["--mpc-backend", flag]).mpc_backend == flag
```

Also test invalid horizon/action dimensions, CPU tensor leakage into private GPU contracts, explicit CUDA request on an unavailable device, and startup rejection before wrapper construction.

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_contracts.py tests/test_m1_dual_panda_o6_entrypoints_static.py`

Expected: import/parser failures because the GPU package and flags do not exist.

- [ ] **Step 3: Implement exact private contracts.**

```python
@dataclass(frozen=True)
class GpuRtiCfg:
    horizon: int = 25
    action_dim: int = 43
    padded_horizon: int = 32
    rti_iterations: int = 1
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    dtype: torch.dtype = torch.float32
    device: str = "cuda:0"

@dataclass
class GpuRtiTrajectory:
    action: torch.Tensor       # [B, 25, 43]
    accepted: torch.Tensor     # [B]
    merit: torch.Tensor        # [B]
```

Add independent flags `--mpc-backend`, `--qp-backend`, and `--control-device`. Preserve current defaults by making both backend defaults `reference-cpu`. Emit requested/actual backend fields in diagnostics.

- [ ] **Step 4: Run GREEN and commit.**

Run the command from Step 2; expected all pass.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py notes
git commit -m "feat: define T500 GPU RTI contracts"
```

## Task 3: Add GPU state mirroring, dynamics condensation, and horizon warm shifting

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/state_adapter.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/reduced_dynamics.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/warm_start.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_state.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_dynamics.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_warm_start.py`

- [ ] **Step 1: Write parity and reset tests first.**

Cover CPU snapshot-to-GPU field order, finite/device validation, `[59+12]` KKT condensation parity against `reduced_dynamics.py`, exact 43-DOF reconstruction, per-environment reset, workspace data-pointer stability, and this shift rule:

```python
shifted[:, :-1].copy_(accepted[:, 1:])
shifted[:, -1].copy_(accepted[:, -1])
shifted_state[:, 0].copy_(measured_x0)
```

An identity/layout/phase mismatch must invalidate only the affected batch row.

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_state.py tests/test_m1_bimanual_gpu_rti_dynamics.py tests/test_m1_bimanual_gpu_rti_warm_start.py`

Expected: missing module failures.

- [ ] **Step 3: Implement persistent buffers and batched condensation.**

Allocate selection matrices, joint maps, KKT buffers, state/action horizons, validity masks, and diagnostic counters once. Use in-place value updates; disallow tensor allocation or `.cpu()` in `step()`. Compare GPU `float32` results against CPU `float64` fixtures using named tolerances, including active limit cases.

- [ ] **Step 4: Run GREEN, run leak guard, and commit.**

Run the focused tests twice with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; expected stable tensor data pointers and bounded reserved-memory delta after warm-up.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_state.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_dynamics.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_warm_start.py notes
git commit -m "feat: add GPU reduced bimanual state"
```

## Task 4: Run the frozen five-fingertip prior on GPU and map it through true O6 Jacobians

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/fingertip_prior.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_fingertip_prior.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py`

- [ ] **Step 1: Add RED tests for semantics and failure isolation.**

Require input shape `[B, 2, 42]`, output mean/precision `[B, 2, 25, 15]`, deterministic four-component selection, correct left/right reflection, nonzero weights only at teacher nodes `t={0.04,0.08,0.12,0.16,0.20}`, and exact zeros thereafter. Verify identical prior output when only task ID, object goal, or palm target changes. Verify metadata mismatch, timeout, NaN, invalid precision, excluded phase, or Jacobian failure disables only the expert cost and records an exact reason.

Use finite differences to test actual normalized left/right O6 URDF/USD fingertip Jacobians:

```python
dq = torch.randn(6, dtype=torch.float64) * 1e-6
fd = (tip_positions(q + dq) - tip_positions(q)) / 1e-6
analytic = fold_o6_fingertip_jacobians(jacobians(q)) @ (dq / 1e-6)
torch.testing.assert_close(fd.reshape(15), analytic, rtol=2e-3, atol=2e-4)
```

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_fingertip_prior.py tests/test_m1_bimanual_o6_contact.py`

Expected: missing GPU prior class and horizon mapping failures.

- [ ] **Step 3: Implement the frozen CUDA inference adapter.**

```python
@dataclass
class GpuPriorTarget:
    mean: torch.Tensor       # [B, 2, 25, 15]
    precision: torch.Tensor  # [B, 2, 25, 15]
    weight: torch.Tensor     # [B, 2, 25, 15]
    enabled: torch.Tensor    # [B, 2]
    reason_code: torch.Tensor
```

Load only the Task 1 production-approved artifact and externally pinned metadata. Keep the frozen network and buffers on `cuda:0`. Select the mixture component by the frozen baseline-compatibility score, sample only within 0.20 seconds, and map `v_tip = J_tip_O6(q) qdot`. Contact/limit/collision conflicts may lower or mask `weight` but cannot change a hard constraint.

- [ ] **Step 4: Run GREEN and commit.**

Run the focused command from Step 2 plus `tests/test_m1_bimanual_expert_prior_runtime.py`; expected all pass and no subprocess worker on the GPU path.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/fingertip_prior.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_fingertip_prior.py \
  Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py notes
git commit -m "feat: add GPU O6 fingertip prior"
```

## Task 5: Build the eager PyTorch Gauss-Newton RTI planner and parallel line search

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/lq_problem.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/line_search.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/planner.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_planner.py`

- [ ] **Step 1: Add RED tests from captured CPU fixtures.**

Freeze small feasible, active-limit, infeasible, left-side-failure, and non-finite fixtures. Test batched dynamics/cost linearization, object pose/twist, both palm targets, grasp-force allocation, arm/hand blocks, expert residual insertion, fixed-base unique-solution bypass, one RTI update, fixed alphas `(1,.5,.25,.125)`, deterministic tie-breaking, and whole-horizon atomic rejection.

```python
result = planner.step(problem, measured_x0, previous_safe)
assert result.action.shape == (batch, 25, 43)
assert result.rti_iterations == 1
assert not torch.any(result.accepted[left_failed_row])
torch.testing.assert_close(result.action[left_failed_row], previous_safe[left_failed_row])
```

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_lq.py tests/test_m1_bimanual_gpu_rti_planner.py`

Expected: missing planner/linearizer failures.

- [ ] **Step 3: Implement one eager RTI correction.**

Batch linearization across 25 nodes and both sides, form a structured Gauss-Newton LQ/QP subproblem, solve one direction initially with PyTorch dense reference operations, evaluate all fixed alphas in one batch, validate each complete candidate, and commit exactly one complete horizon or retain the shifted last-safe horizon. Fixed equal lower/upper bounds must bypass optimization after validating the nominal trajectory.

- [ ] **Step 4: Run GREEN, compare first action with CPU, and commit.**

Use named absolute/relative tolerances for KKT residuals and accepted first action; require identical acceptance/rejection decisions on frozen fixtures.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/{lq_problem.py,line_search.py,planner.py} \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_planner.py notes
git commit -m "feat: add eager GPU bimanual RTI"
```

## Task 6: Add the H=25-to-32 associative scan and validated fixed-size Triton solves

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/associative_scan.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/fixed_solve.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/fixed_solve_triton.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/trajectory_scan.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/planner.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_scan.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_triton.py`

- [ ] **Step 1: Add RED parity tests.**

Test arbitrary batch sizes used by single-env Play and multi-env training; SPD and general systems; ill-conditioned rejection; gradients disabled; fixed-address output; neutral padded factors at nodes 25–31; dense versus scan direction; and KKT residuals. Explicitly assert that perturbing padded nodes cannot change the first 25 results.

```python
scan = solve_trajectory_scan(problem.pad_to(32))[:, :25]
dense = solve_dense_reference(problem)
torch.testing.assert_close(scan, dense, rtol=5e-4, atol=5e-5)
```

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_scan.py tests/test_m1_bimanual_gpu_rti_triton.py`

Expected: missing scan and Triton kernels.

- [ ] **Step 3: Adapt only solver architecture from pinned remote commit.**

Record `d080df994a7170486e74641dc75e28c344b7a3ee` in module provenance. Retain fixed-tree/scan and fixed-solve ideas; remove the remote Go2 18-state model, gait, terrain, and loss assumptions. Keep a PyTorch fixed-solve fallback selected at startup after parity and capability checks.

- [ ] **Step 4: Run GREEN, benchmark dense versus scan, and commit.**

Require all parity tests pass before `planner.py` can select scan/Triton. Record both kernel-only and transfer-inclusive timings.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_scan.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_triton.py notes
git commit -m "feat: add structured GPU RTI solve"
```

## Task 7: Validate an isolated OSQP CUDA comparison and fallback backend

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/osqp_backend.py`
- Create: `Go2Pvcnn/scripts/check_m1_osqp_cuda_runtime.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_osqp_backend.py`
- Create: `notes/log/2026-09-15-t500-osqp-cuda-isolation.md`

- [ ] **Step 1: Add backend-selection and immutable-input RED tests.**

Test canonical `0.5 x' P x + q' x`, `l <= A x <= u`; explicit-unavailable startup failure; `auto` exclusion when unbenchmarked; repeated solve consistency; timeout/non-convergence; residual rejection; and proof that fallback cannot mutate the RTI convexified tensors or last-safe command.

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_osqp_backend.py`

Expected: missing isolated backend adapter.

- [ ] **Step 3: Create an isolated environment and probe compatibility.**

Create a dedicated environment outside the project environment, install only pinned candidate OSQP 1.x CUDA dependencies, and run `check_m1_osqp_cuda_runtime.py` against recorded convex subproblems. Record wheel/package hashes, CUDA/driver identity, solver algebra, residuals, and IPC overhead if process isolation is needed. Do not alter `/home/xk/miniconda3/envs/go2`.

- [ ] **Step 4: Implement fail-closed adapter or record it unavailable.**

If compatibility and transfer-inclusive latency gates pass, expose the backend behind `--qp-backend osqp-cuda`; otherwise keep the adapter unavailable with the exact recorded reason. In both cases, CPU remains the final reference fallback.

- [ ] **Step 5: Run GREEN and commit.**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/osqp_backend.py \
  Go2Pvcnn/scripts/check_m1_osqp_cuda_runtime.py \
  Go2Pvcnn/tests/test_m1_bimanual_osqp_backend.py notes
git commit -m "feat: isolate OSQP CUDA comparison backend"
```

## Task 8: Capture the steady-state GPU runtime and integrate it with the common Wrapper

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/cuda_graph.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/manager.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_runtime.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_runtime.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_vector_wrapper.py`

- [ ] **Step 1: Add RED lifecycle, graph, and fallback tests.**

Cover eager warm-up, stable storage addresses, first capture, replay, classified capture failure, recapture only after shape/layout reset, per-row reset, CUDA error, timeout, NaN/Inf, OSQP comparison fallback, CPU fallback, last-safe hold, consecutive-failure safe lowering, and resource release on normal exit/exception. Verify deterministic physical reset remains in the common Wrapper and Play stays open after safe `DONE`/`TERMINATED` until the user closes it or `--max-steps` expires.

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_runtime.py tests/test_m1_bimanual_runtime.py tests/test_m1_bimanual_vector_wrapper.py`

Expected: missing graph manager and GPU routing failures.

- [ ] **Step 3: Implement graph capture and backend ownership.**

The manager must resolve requested versus actual backend before scene creation, own all GPU workspaces, collect CUDA-event stage timings without per-step synchronization, publish only accepted first actions, and expose a CPU `float64` mirror only at legacy/reporting/fallback boundaries. Eager GPU fallback after graph failure is allowed only with the same validation path and a recorded reason.

- [ ] **Step 4: Run GREEN, execute a 10,000-step allocation test, and commit.**

Expected: bounded GPU memory after warm-up, no worker/process leak, no changed CPU-default behavior, and safe terminal window persistence.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/{cuda_graph.py,manager.py} \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_runtime.py \
  Go2Pvcnn/tests/test_m1_bimanual_runtime.py \
  Go2Pvcnn/tests/test_m1_bimanual_vector_wrapper.py notes
git commit -m "feat: integrate captured GPU RTI runtime"
```

## Task 9: Move full-action latent encoding and student control to GPU

**Files:**

- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_contracts.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_model.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_runtime.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_latent_model.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_latent_runtime.py`

- [ ] **Step 1: Add RED tests for full-horizon encoding and correction.**

Require the encoder input to be the complete accepted `[B,25,43]` teacher horizon. Test CUDA placement, stable dtype/device, teacher period 8, latent TTL 16, no partial-horizon encoding, per-row reset, envelope departure correction, teacher failure fallback, student NaN fallback, and separation between the fingertip-prior artifact and full-action latent artifact hashes.

```python
decision = runtime.step(snapshot)
assert decision.teacher_horizon.shape == (batch, 25, 43)
assert decision.z.device.type == "cuda"
assert decision.complete_action.shape == (batch, 43)
```

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_latent_model.py tests/test_m1_bimanual_latent_runtime.py`

Expected: CPU-only placement and/or missing complete-horizon guards.

- [ ] **Step 3: Implement GPU-resident latent/student inference.**

Keep teacher inference low-frequency and student/WBC/safety high-frequency. Do not let `z` replace the complete command contract. Maintain independent artifact metadata and acceptance diagnostics for fingertip prior, teacher horizon, encoder, and student. Correct or fall back whenever the student leaves the accepted teacher envelope.

- [ ] **Step 4: Run GREEN and commit.**

Run focused latent tests plus `tests/test_m1_bimanual_full_action_teacher.py`; expected all pass on CPU reference and GPU paths.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/{latent_contracts.py,latent_model.py,latent_runtime.py} \
  Go2Pvcnn/tests/test_m1_bimanual_latent_model.py \
  Go2Pvcnn/tests/test_m1_bimanual_latent_runtime.py notes
git commit -m "feat: run T500 latent controller on GPU"
```

## Task 10: Benchmark, run Isaac GPU0 validation, and emit formal SHA-pinned evidence

**Files:**

- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_gpu_rti_benchmark.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_verify.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_benchmark.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_verification.py`
- Create: `notes/log/2026-09-15-t500-gpu-rti-validation.md`
- Modify: the five note/index files listed in Global Constraints

- [ ] **Step 1: Add RED schema and benchmark-integrity tests.**

Require CPU/GPU runs to consume the same SHA-pinned captured corpus. Require separate single-env Play and multi-env throughput reports. Extend manifests with requested/actual backend per layer, remote commit, Triton/OSQP/CUDA/driver/PyTorch/Isaac/GPU identities, tree topology, RTI count, line-search alphas, tolerances, fallback reasons/counts, all fingertip artifact hashes and enable/disable reasons, corpus hash, and P50/P95/P99 setup/update/prior/linearize/scan/line-search/transfer/validation/total times.

- [ ] **Step 2: Run RED.**

Run: `cd Go2Pvcnn && PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_benchmark.py tests/test_m1_dual_panda_o6_verification.py`

Expected: missing benchmark fields and formal manifest validation.

- [ ] **Step 3: Implement capture and comparison tooling.**

Capture immutable seed-42 samples at 400 and 4000 steps. Benchmark CPU reference, eager RTI, scan/Triton RTI, captured RTI, and available OSQP CUDA on identical feasible/infeasible cases. Include startup and transfers in end-to-end numbers. Reject speed claims that change safety checks, mission length, rendering mode, or corpus.

- [ ] **Step 4: Run the full CPU regression suite.**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_*.py \
  tests/test_m1_dual_panda_o6_contracts.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_dual_panda_o6_verification.py
```

Expected: all existing public CPU contracts and all new GPU-independent tests pass.

- [ ] **Step 5: Run GPU0 smoke with the production prior.**

```bash
cd Go2Pvcnn
PRIOR_METADATA_SHA=$(sha256sum data/external/dexmanipnet/models/student_run_e/metadata.json | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  --mode teacher --seed 42 --max-steps 400 \
  --mpc-backend bimanual-rti-cuda --qp-backend auto --control-device cuda:0 \
  --fingertip-prior-artifact data/external/dexmanipnet/models/student_run_e \
  --fingertip-prior-metadata-sha256 "$PRIOR_METADATA_SHA" \
  --diagnostics
```

Expected: valid shared-base coordinates, supported workbench/object, correct control direction, required clearance, prior enabled only in approved phases, safe phase progression, and no immediate `DONE`/`TERMINATED` from startup or solver failure.

- [ ] **Step 6: Apply staged performance gates.**

G1 must show at least 2x end-to-end speedup from solver-independent paths. G3/G4 must show at least 10x cumulative transfer-inclusive speedup. G5 targets near-real-time Play; if missed, record the exact stage breakdown and keep the slower backend out of `auto`.

- [ ] **Step 7: Run the complete mission and formal 30-trial GPU0 matrix.**

Require every accepted formal trial to traverse `APPROACH -> PRELOAD -> GRASP -> LIFT -> HOLD -> LOWER -> RELEASE -> DONE`, emit one JSONL record per trial, then build and independently verify a SHA-pinned aggregate manifest. A failed trial remains recorded; do not overwrite or silently rerun it under the same identity.

- [ ] **Step 8: Run long-duration cleanup/recovery validation.**

Verify bounded GPU memory, fixed solver workspace identities, zero leaked subprocesses, successful fallback recovery, and release during normal exit, terminal hold-window exit, startup rejection, and exception unwinding.

- [ ] **Step 9: Update project status and commit final evidence.**

```bash
git add Go2Pvcnn/scripts/m1_dual_panda_o6_gpu_rti_benchmark.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_verify.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_benchmark.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_verification.py notes \
  docs/superpowers/2026-09-14-t500-dual-panda-o6-project-status.md
git commit -m "test: qualify T500 GPU RTI runtime"
```

## Final Review Checklist

- [ ] Confirm every public CPU `float64` contract and default CLI behavior is unchanged.
- [ ] Confirm the expert prior never sees task/object/palm goals and never overrides a hard constraint.
- [ ] Confirm true left/right O6 Jacobians, mimic folding, fingertip order, and reflection are verified by finite differences.
- [ ] Confirm only the first five teacher nodes carry expert weight and later nodes are exactly zero.
- [ ] Confirm every encoded teacher result is complete `[25,43]`, accepted atomically, and separately pinned from the fingertip model.
- [ ] Confirm fixed-base mode bypasses unnecessary optimization while retaining diagnostics.
- [ ] Confirm GPU failures have exact reasons and deterministic OSQP/CPU/last-safe fallback order.
- [ ] Confirm performance reports include transfers/startup and separate single-env latency from batched throughput.
- [ ] Confirm formal GPU0 output contains 30 immutable JSONL rows and a verified aggregate SHA-256 manifest.
- [ ] Confirm no training entrypoint, contact MPC, perception, Residual controller, or T400/T500 public-contract change was introduced.
