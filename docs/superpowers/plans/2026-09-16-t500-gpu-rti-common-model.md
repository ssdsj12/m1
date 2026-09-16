# T500 GPU RTI Common Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eager CUDA float32 predictor that optimizes a complete 43-effort horizon with private left/right 6-D grasp wrenches and atomically publishes only `[B,25,43]`.

**Architecture:** Use a fixed110-D reduced state (`6 base tangent pose + 6 base twist + 43 q + 43 qd + 6 box pose + 6 box twist`) and private55-D node control (`43 effort +12 wrench`). Derive robot/object transitions from the full71-D constrained KKT and existing rigid-box model, solve one dense Gauss-Newton direction, evaluate four complete candidates in parallel, and use Task3 last-safe storage.

**Tech Stack:** Python3.11, PyTorch2.7/CUDA float32, pytest, existing CPU float64 bimanual fixtures.

## Global Constraints

- Preserve public CPU float64 dataclasses, joint order, state machine, safety thresholds, T400/T500 contracts and last-safe fallback.
- Freeze `H=25`; public action is `[B,25,43]`. Private node control is exactly `[tau(43),w_left(6),w_right(6)]`; never publish wrench or command59 coordinates.
- Wrenches are base-frame spatial `[force_xyz,moment_xyz]` applied hand-to-box; robot reaction is `-J_left.T@w_left-J_right.T@w_right`.
- A40ms node uses zero-order-held controls, two20ms arm checks, four10ms O6 checks and one40ms box step.
- Dynamics, effort/joint/rate/acceleration, wheel, friction/moment, collision/distance and complete-horizon finiteness are hard; target tracking is soft.
- One RTI correction; alphas exactly `(1,.5,.25,.125)`; lowest index wins ties.
- Any node/side failure rejects that row's whole candidate and retains only a previously accepted complete horizon.
- Expert contribution is exactly zero; do not synthesize/train an artifact.
- Steady-state GPU methods use persistent CUDA float32 buffers: no `.cpu()`, `.item()`, `.numpy()`, host scalar extraction or new tensor storage.
- Keep GPU Play fail-closed. No scan/Triton, graphs, OSQP install, Isaac, training, perception, contact-mode MPC or Residual work.

---

## Task 1: Assemble Coupled Robot/Box Dynamics And LQ Direction

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/lq_problem.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CoupledRtiDims:
    horizon: int = 25
    state_dim: int = 110
    effort_dim: int = 43
    wrench_dim: int = 12
    control_dim: int = 55

@dataclass
class CoupledRtiInput:
    measured_state: torch.Tensor            # [B,111], Task3 warm/report only
    base_pose0: torch.Tensor                 # [B,6], position + local SO(3) tangent
    base_twist0: torch.Tensor                # [B,6]
    base_quaternion0: torch.Tensor           # [B,4], reporting reconstruction
    active_q0: torch.Tensor                 # [B,43]
    active_qd0: torch.Tensor                # [B,43]
    box_pose0: torch.Tensor                 # [B,6]
    box_twist0: torch.Tensor                # [B,6]
    active_selector: torch.Tensor           # [B,43,59]
    palm_jacobian: torch.Tensor             # [B,2,6,59]
    box_wrench_map: torch.Tensor            # [B,6,12]
    box_gravity: torch.Tensor               # [B,6]
    nominal_control: torch.Tensor           # [B,25,55]
    control_lower: torch.Tensor             # [B,25,55]
    control_upper: torch.Tensor             # [B,25,55]
    state_target: torch.Tensor              # [B,26,110]
    state_weight: torch.Tensor              # [110]
    control_weight: torch.Tensor            # [55]
    terminal_weight: torch.Tensor           # [110]
    palm_residual_offset: torch.Tensor      # [B,25,12]
    palm_state_jacobian: torch.Tensor       # [B,25,12,110]
    palm_weight: torch.Tensor               # [12]
    state_lower: torch.Tensor               # [B,26,110]
    state_upper: torch.Tensor               # [B,26,110]
    wrench_inequality_matrix: torch.Tensor  # [B,25,C,12]
    wrench_inequality_upper: torch.Tensor   # [B,25,C]
    hard_inequality_matrix: torch.Tensor    # [B,25,K,55]
    hard_inequality_upper: torch.Tensor     # [B,25,K]
    input_valid: torch.Tensor               # [B] bool

class CoupledLqWorkspace:
    def __init__(self, *, batch: int, max_wrench_constraints: int,
                 max_hard_constraints: int, device="cuda:0"): ...
    def assemble(self, problem: CoupledRtiInput, dynamics: GpuReducedDynamics) -> torch.Tensor: ...
    def rollout(self, control: torch.Tensor) -> torch.Tensor: ...
    def rollout_candidates(self, control: torch.Tensor) -> torch.Tensor: ...  # [B,4,26,110]
    def linearize(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def solve_direction(self) -> torch.Tensor: ...
    def validate(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor: ...
    def validate_candidates(self, state: torch.Tensor,
                            control: torch.Tensor) -> torch.Tensor: ...       # [B,4]
```

`assemble()` consumes a successful `GpuReducedDynamics.step()` but never mutates its borrowed buffers. All returns are persistent borrowed tensors.

- [ ] **Step 1: Write failing contract, sign, substep and validation tests.**

Use real CUDA and coupled SPD CPU fixtures. Calculate an independent float64 augmented71×56 KKT reference. Assert `qdd_wrench[B,59,12]` matches at `atol=rtol=3e-5`. In a zero-offset fixture with box twist equal to each palm twist, verify contact power cancels between `J@qd` against `-w` and the box against `+w`.

```python
assert workspace.rollout(nominal).shape == (batch, 26, 110)
assert torch.equal(workspace.state[:, 0], measured_x0)
assert workspace.arm_substep_state.shape == (batch, 25, 2, 98)
assert workspace.hand_substep_state.shape == (batch, 25, 4, 24)
```

Cover nonfinite left/right Jacobian, inherited invalid mass/rank, `input_valid=False`, effort bounds, intermediate arm/O6 limit, wrench friction/moment violation, and valid zero-wrench prior-off. Only the bad row may fail.

- [ ] **Step 2: Run RED.**

```bash
cd Go2Pvcnn
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_gpu_rti_lq.py
```

Expected: `ModuleNotFoundError: ...gpu_rti.lq_problem`.

- [ ] **Step 3: Implement fixed-shape contracts, wrench reaction and transitions.**

Validate shape/dtype/device/autograd before writes. Preallocate state/control staging, wrench RHS/solution, `A[B,25,110,110]`, `B[B,25,110,55]`, `c[B,25,110]`, rollout/substeps, Hessian/gradient/direction, masks and counters with explicit float32.

Build KKT wrench RHS top rows as exactly `[-J_left.T,-J_right.T]`, wheel rows zero, sanitize invalid rows, and solve through the already validated LU into this workspace's output. Explicitly stage side/spatial order; do not depend on a non-contiguous reshape.

Use `dt=.04`, arm checks at `.02,.04`, hand checks at `.01,.02,.03,.04`. Integrate constant acceleration with `q+=dt*qd+.5*dt**2*qdd`, `qd+=dt*qdd`; box uses `box_wrench_map` and `box_gravity`. Orientation residuals are precomputed base-frame tangent coordinates, not Euler differences.

- [ ] **Step 4: Implement dense GN assembly and one direction.**

Roll out nominal, propagate fixed affine Jacobians, and form:

```python
H = 2 * (J.transpose(-1, -2) @ W @ J + R + 1e-6 * I)
g = 2 * (J.transpose(-1, -2) @ W @ residual + R @ nominal)
direction = solve(H, -g)
```

Use persistent batched factor/solve outputs. Weights must be finite/nonnegative. Equality-bound direction is zero after validating nominal. `validate()` returns one bool per row after all nodes/substeps, bounds, wrench inequalities, residuals and finite checks.

- [ ] **Step 5: Run GREEN twice and memory guards.**

Run Step2 twice with `-s`. Require all pass on real CUDA0, stable pointers, allocated delta0, reserved delta≤2MiB after10 warmups+100 calls. A dispatch guard rejects host scalar extraction/new tensor storage.

- [ ] **Step 6: Commit.**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/lq_problem.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_lq.py
git commit -m "feat: assemble coupled GPU RTI dynamics"
```

---

## Task 2: Evaluate Four Complete-Horizon Candidates In Parallel

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/line_search.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_line_search.py`

**Interfaces:**

```python
@dataclass
class LineSearchResult:
    control: torch.Tensor       # borrowed [B,25,55]
    state: torch.Tensor         # borrowed [B,26,110]
    accepted: torch.Tensor      # borrowed [B] bool
    alpha_index: torch.Tensor   # borrowed [B] int64; -1 rejected
    merit: torch.Tensor         # borrowed [B] float32; +inf rejected

class ParallelLineSearch:
    def __init__(self, *, batch: int, max_wrench_constraints: int,
                 max_hard_constraints: int, device="cuda:0"): ...
    def step(self, workspace: CoupledLqWorkspace) -> LineSearchResult: ...
```

- [ ] **Step 1: Write RED tests for fixed candidates and atomic rejection.**

Assert storage `[B,4,25,55]`, exact alphas, lowest-index tie break, equality-bound bypass and full-horizon validation:

```python
result = search.step(workspace)
assert search.candidate_control.shape == (B, 4, 25, 55)
assert search.alphas.tolist() == [1.0, .5, .25, .125]
assert result.alpha_index[tied_row].item() == 0
assert not result.accepted[left_failed_row]
torch.testing.assert_close(result.control[left_failed_row], nominal[left_failed_row])
```

Include alpha1 invalid/alpha.5 valid, all invalid, candidate NaN, failure only at node24, left-only constraint failure and all-equal bounds. No case may accept one side or node0 alone.

- [ ] **Step 2: Run RED.**

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  -m pytest -q tests/test_m1_bimanual_gpu_rti_line_search.py
```

Expected: `ModuleNotFoundError: ...gpu_rti.line_search`.

- [ ] **Step 3: Implement persistent parallel candidates and deterministic selection.**

Broadcast four alphas into persistent buffers, clamp independent box bounds, restore equality-bound elements from nominal, roll out `B*4` candidates and validate coupled wrench inequalities without projecting them. Invalid merit is `+inf`. Accept strict improvement over valid nominal; equal-bound bypass returns validated nominal at alpha index0.

```python
self.masked_merit.copy_(self.merit)
self.masked_merit.masked_fill_(~self.candidate_valid, float("inf"))
torch.min(self.masked_merit, dim=1,
          out=(self.best_merit, self.best_index))
torch.isfinite(self.best_merit, out=self.accepted)
```

Do not extract indices/masks to the host. Rejected rows stage nominal only; the planner, not line search, owns last-safe publication.

- [ ] **Step 4: Run GREEN and allocation guards.**

Run Task2 twice and Tasks1–2 once. Require all pass on GPU0; pointer identity and allocated memory unchanged over100 post-warmup calls, reserved delta≤2MiB.

- [ ] **Step 5: Commit.**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/line_search.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_line_search.py
git commit -m "feat: add atomic GPU RTI line search"
```

---

## Task 3: Integrate One RTI Update With Last-Safe Storage

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/planner.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_planner.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py`

**Interfaces:**

```python
@dataclass
class GpuRtiPlanResult:
    action: torch.Tensor          # borrowed [B,25,43]
    accepted: torch.Tensor        # current cycle [B] bool
    safe_available: torch.Tensor  # [B] bool
    merit: torch.Tensor           # [B]
    reason_code: torch.Tensor     # [B] int64
    rti_iterations: int = 1

class EagerBimanualRtiPlanner:
    def __init__(self, *, batch: int, max_wrench_constraints: int,
                 max_hard_constraints: int, device="cuda:0"): ...
    def step(self, problem: CoupledRtiInput, dynamics: GpuReducedDynamics,
             identity: torch.Tensor, reset: torch.Tensor) -> GpuRtiPlanResult: ...
```

Private reason constants: `0 accepted`, `1 no_safe_horizon`, `2 input_invalid`, `3 dynamics_invalid`, `4 nominal_invalid`, `5 line_search_rejected`, `6 nonfinite_result`.

- [ ] **Step 1: Write RED lifecycle and parity tests.**

Use feasible, active-limit, all-infeasible, left-failure and nonfinite fixtures. Assert exactly one direction solve; result is `[B,25,43]`; no wrench appears in result. Seed a safe horizon, then prove a failed cycle returns it unchanged with `accepted=False,safe_available=True`. Reset/identity mismatch must clear availability until a new acceptance. Row0 failure cannot alter row1.

For a small unconstrained case, independently assemble the110-state/55-control CPU float64 system. Compare state at `STATE_ATOL=5e-5`, first effort at `ACTION_ATOL=5e-4,RTOL=5e-4`, and require identical accept/reject decisions. Include active public effort bounds and equality-bound bypass.

- [ ] **Step 2: Run RED.**

```bash
cd Go2Pvcnn
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_gpu_rti_planner.py \
  tests/test_m1_bimanual_gpu_rti_contracts.py
```

Expected: `ModuleNotFoundError: ...gpu_rti.planner`.

- [ ] **Step 3: Implement fixed-order orchestration and atomic publication.**

Use this order:

```python
warm.step(problem.measured_state, identity, reset=reset)
lq.assemble(problem, dynamics)
lq.rollout(lq.nominal_control)
nominal_valid = lq.validate(lq.state, lq.nominal_control)
lq.linearize()
lq.solve_direction()
candidate = line_search.step(lq)
publishable = candidate.accepted & nominal_valid & lq.valid
warm.accept(candidate.control[..., :43], reporting_state,
            publishable, identity)
```

Own a persistent `[B,26,111]` reporting horizon. Copy active and box predictions through a frozen index map; reconstruct the base quaternion from `base_quaternion0` and the predicted local SO(3) tangent using the existing shortest-path helpers. Never linearly copy tangent coordinates into quaternion slots or count carried fields as predicted residuals.

Stage before `warm.accept`. On failure return shifted last-safe only if it was valid before this solve; otherwise zero action and set `no_safe_horizon`. Fallback is never marked accepted. Wrench stays private.

- [ ] **Step 4: Run focused GREEN twice and full Task2–3 GPU regression.**

Run Step2 twice, then:

```bash
cd Go2Pvcnn
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q -s \
  tests/test_m1_bimanual_gpu_rti_contracts.py \
  tests/test_m1_bimanual_gpu_rti_state.py \
  tests/test_m1_bimanual_gpu_rti_dynamics.py \
  tests/test_m1_bimanual_gpu_rti_warm_start.py \
  tests/test_m1_bimanual_gpu_rti_lq.py \
  tests/test_m1_bimanual_gpu_rti_line_search.py \
  tests/test_m1_bimanual_gpu_rti_planner.py \
  tests/test_m1_bimanual_reduced_dynamics.py \
  tests/test_m1_bimanual_full_action_teacher.py \
  tests/test_m1_dual_panda_o6_contracts.py
```

Require no skips/warnings on GPU0. After10 warmups+100 calls, planner pointers stable, allocated delta0, reserved delta≤2MiB; dispatch guard finds no host scalar/new storage.

- [ ] **Step 5: Verify fail-closed boundary and commit.**

```bash
/home/xk/miniconda3/envs/go2/bin/python -m compileall -q \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti
git diff --check -- Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_*.py
```

Also verify explicit `bimanual-rti-cuda` still rejects before scene creation; this plan does not mark it available.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/planner.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_planner.py \
  Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py
git commit -m "feat: add eager coupled GPU RTI planner"
```

---

## Completion Boundary

This plan produces an eager prior-off correctness candidate, not GPU Play readiness. Remaining approved gates are scan/Triton parity, CUDA Graph/runtime integration, latent encoding, benchmark manifest, Isaac GPU0 smoke and formal30-trial acceptance. Prior-on remains blocked until dual-source E0/student qualification.
