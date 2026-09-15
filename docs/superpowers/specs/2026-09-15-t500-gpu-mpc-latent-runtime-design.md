# T500 GPU MPC + Latent Runtime Acceleration Design

## 1. Status

This specification records the user-approved A+C design: use a mature CUDA QP
backend to accelerate as much of the online bimanual control stack as is useful,
then combine it with the distilled latent/student path. The existing deterministic
CPU implementation remains the reference and safety fallback.

The design is approved section by section but remains unimplemented. It does not
claim that CUDA OSQP is installed, that GPU MPC is faster on this workload, or that
the full manipulation mission passes Isaac validation.

## 2. Problem And Evidence

The current `--mode teacher` Play path is much slower than real time. A current
GPU0 probe measured approximately 52 seconds of wall time for 200 physics steps
(1 second of simulated time), and 145.23 seconds for 400 physics steps (2 seconds
of simulated time). The latter reached `PRELOAD` after about 1.93 simulated
seconds with all measured QP layers feasible, so the primary symptom is compute
latency rather than universal solver fallback.

The control stack currently uses an explicitly deterministic CPU `float64`
active-set backend. At 200 Hz it solves WBC every step, both Hand MPCs every two
steps, both Arm MPCs every four steps, Object MPC every eight steps, and the
full-action teacher every eight steps. The teacher horizon contains 25 nodes.
The current fixed-base Play configuration also constrains every teacher action
component to its nominal value but still invokes a QP solve for every horizon
node.

Expert-data conversion is a separate offline workload based on NumPy, SciPy,
Trimesh, URDF parsing, and SQLite. It remains a CPU job and must not run at the
same time as performance-sensitive Play benchmarks.

## 3. Goals

- Move online tensor-heavy MPC, WBC, safety projection, latent inference, and
  student inference to GPU wherever measured end-to-end latency improves.
- Use OSQP's maintained CUDA algebra backend instead of implementing a general
  CUDA QP solver locally.
- Keep GPU matrices and solver workspaces persistent across control cycles.
- Combine horizon nodes and left/right controllers into sufficiently large GPU
  problems where small independent QPs would be launch-bound.
- Preserve all existing T400/T500 public CPU `float64` contracts and default
  behavior.
- Preserve atomic acceptance, last-safe fallback, state-machine semantics, and
  formal SHA-pinned evidence.
- Reach at least a 10x cumulative speedup before the distilled path and target
  near-real-time Play after latent/student integration.

## 4. Non-Goals

- Do not GPU-port Trimesh, SQLite, archive validation, URDF parsing, or offline
  expert-data conversion in this work.
- Do not remove the CPU reference solver or weaken residual checks.
- Do not require byte-identical CPU/GPU floating-point results.
- Do not modify the T400/T500 public snapshot, action, or safety contracts.
- Do not add perception, contact MPC, Residual control, a training entrypoint,
  or a new manipulation task.
- Do not run training and interactive Play concurrently on the single GPU0.

## 5. Selected Architecture

```text
Isaac GPU state
    -> private GPU state mirror
       -> Object MPC (persistent OSQP CUDA workspace)
       -> joint left/right Arm MPC problem
       -> joint left/right Hand MPC problem
       -> batched/block-diagonal full-action Teacher problem
       -> WBC and Safety Projection
    -> GPU residual and constraint gate
       -> accepted complete command -> Isaac
       -> rejected result -> CPU reference retry
          -> accepted CPU command -> Isaac + fallback diagnostic
          -> rejected CPU command -> last safe complete command
```

The private GPU runtime may use CUDA tensors internally. Existing public
dataclasses continue to accept and return finite CPU `float64` tensors. CUDA
tensors must not leak through those boundaries. CPU mirrors are created only for
the legacy boundary, formal reporting, explicit reference comparison, or CPU
fallback.

The Play and Probe entrypoints gain an explicit backend request:

```text
--qp-backend reference-cpu
--qp-backend osqp-cuda
--qp-backend auto
--control-device cuda:0
```

`reference-cpu` retains current behavior. `osqp-cuda` fails during startup if the
pinned CUDA backend is unavailable. `auto` may choose per-layer backends only
from a benchmarked, recorded allowlist and must report the actual choice.

## 6. Mature Solver Choice

OSQP 1.x with its CUDA algebra is the selected non-reference backend. Its problem
form directly covers the existing convex QPs:

```text
minimize 0.5 x' P x + q' x
subject to l <= A x <= u
```

The current environment's `osqp 0.6.7.post3` is not modified in place. A pinned
OSQP 1.x CUDA wheel is first tested in an isolated environment against CUDA 12.8,
PyTorch 2.7, Isaac Sim 5.1, and the RTX 5070. The selected Python wheel, CUDA
runtime dependencies, versions, and SHA-256 values become part of the runtime
manifest.

ProxSuite is not the primary choice because its documented batch acceleration is
CPU/OpenMP. cuRobo is not a direct replacement because the project has custom
object, dual-arm, dual-hand, force-allocation, WBC, and safety QPs. Neither choice
is prohibited for a later measured comparison.

## 7. Persistent Workspace And Data Flow

Each migrated layer owns a workspace with a fixed sparsity pattern. Runtime
updates change only numerical values such as `P`, `q`, `A`, `l`, and `u`. Solvers
must not be constructed inside the 200 Hz loop. The previous accepted solution
is used as a warm start only when its identity, shape, phase, and constraint
layout match the current problem.

Static tensors such as identity matrices, selection matrices, joint-order maps,
constraint indices, and block-diagonal layouts are allocated once. State,
Jacobian, dynamics, solver values, and accepted solutions remain resident on
GPU. CUDA events measure setup, update, solve, validation, transfer, and total
cycle latency without adding per-step CPU synchronization to the normal path.

## 8. Problem Aggregation

- **Teacher:** the 25 horizon nodes become one block-diagonal problem where a
  solve is required. In fixed-base mode, equal lower and upper bounds define a
  unique action; the runtime bypasses the solver, validates the bounded nominal
  trajectory, and produces the same diagnostics contract.
- **Arm MPC:** left and right problems use one joint block-diagonal solve and one
  atomic acceptance decision.
- **Hand MPC:** left and right baseline/contact problems use one joint solve.
  The optional expert prior remains a soft cost and cannot weaken hard contact
  constraints.
- **Object MPC:** retain the existing whole-horizon formulation and bind it to a
  persistent CUDA workspace.
- **WBC/Safety:** use persistent workspaces and warm starts. These small problems
  remain eligible for CPU execution under `auto` if measured transfer-inclusive
  latency is lower; explicit `osqp-cuda` still exercises the CUDA path.

Aggregation must preserve joint order, phase ownership, per-side diagnostics,
and the rule that a failed side cannot be partially committed.

## 9. Acceptance And Fallback

Every GPU candidate is independently checked before submission:

- finite shape, dtype, device, and joint ordering;
- equality residual within the existing layer tolerance;
- inequality, effort, velocity, acceleration, and joint-limit violation within
  the existing layer tolerance;
- dynamics and contact residuals within existing thresholds;
- complete left/right and whole-command atomicity;
- correct solver status, iteration count, and workspace identity.

Failure handling is deterministic:

1. A valid GPU result is committed.
2. A timeout, non-finite result, non-convergence, CUDA error, or excessive
   residual triggers the CPU reference solver on the same immutable input.
3. A valid CPU result is committed with `gpu_fallback_cpu` and a specific reason.
4. If CPU also rejects the input, the most recent safe complete command is held.
5. Existing consecutive-failure thresholds control safe lowering and terminal
   state transitions.

`--qp-backend osqp-cuda` may use the CPU only as an explicit safety fallback; it
must never silently relabel the run as CUDA-only. Startup failures occur before
Isaac scene creation when the requested backend, CUDA device, or pinned library
identity is unavailable.

## 10. Determinism

- CPU remains the strict deterministic reference.
- GPU uses fixed problem ordering, fixed sparsity, fixed iteration and tolerance
  settings, deterministic warm-start ownership, and TF32 disabled for critical
  solves.
- Repeated runs on the pinned GPU/software stack must agree within specified
  numeric tolerances and make the same acceptance and phase-transition decisions.
- Cross-GPU or cross-driver byte identity is not required.
- Formal evidence records requested and actual backends, device identity,
  versions, solver settings, and fallback history.

## 11. Dependency Isolation And Resource Rules

- Record the current environment before installing anything.
- Validate OSQP CUDA in an isolated environment first.
- Pin all accepted packages and artifact hashes before modifying M1 dependency
  files.
- If OSQP CUDA conflicts with Isaac's CUDA libraries, use an isolated solver
  process only after measuring CUDA IPC overhead; do not contaminate the Isaac
  process with incompatible libraries.
- Preallocate workspaces under a documented GPU memory budget suitable for the
  12 GB RTX 5070.
- Training and Play are mutually exclusive GPU0 workloads.
- Offline dataset conversion must remain CPU-only and must not create a CUDA
  context.
- Close and release every solver workspace during normal exit, safe termination,
  startup rejection, and exception unwinding.

## 12. Delivery Stages

### G0: Capture And Benchmark

Capture immutable real QP samples from seed 42 at 400 and 4000 steps. Measure
problem construction, setup, update, solve, validation, transfer, rendering, and
end-to-end step latency. Record the sample corpus SHA-256.

### G1: Solver-Independent Fast Paths

Implement the fixed-base teacher unique-solution path and cache invariant CPU
tensors. Require exact reference behavior. Target at least 2x end-to-end speedup
without installing OSQP CUDA.

### G2: OSQP CUDA Adapter

Add backend selection, dependency/runtime validation, persistent workspaces,
warm starts, residual gates, and CPU fallback. First validate each recorded QP
independently.

### G3: Aggregate Problems

Create the teacher block-diagonal formulation and joint left/right Arm and Hand
problems. Migrate Object MPC, WBC, and Safety where transfer-inclusive benchmarks
show a benefit. Target at least 10x cumulative speedup.

### G4: GPU-Resident Runtime

Add the private GPU snapshot and remove control-loop CPU round trips. Retain CPU
mirrors only at approved boundaries. Profile for hidden synchronization and
unbounded workspace growth.

### G5: Latent/Student Integration

Run latent and fingertip-student inference on GPU. Use the complete MPC teacher
at low frequency, while GPU WBC/Safety remains high frequency. Trigger correction
or fallback when the compact policy departs from the accepted teacher envelope.
Target near-real-time interactive Play without reducing safety checks.

Each stage is independently benchmarked, reviewable, and reversible. A later
stage cannot compensate for a failed earlier correctness gate.

## 13. Verification

1. Preserve all CPU contract and unit tests.
2. Add backend tests for malformed input, device mismatch, unavailable CUDA,
   workspace identity, warm start, timeout, non-convergence, NaN/Inf, residual
   rejection, and atomic fallback.
3. Compare every captured QP against the CPU reference, including feasible and
   infeasible cases, active limits, mimic behavior, and repeated solve behavior.
4. Benchmark setup, update, solve, transfer, validation, and total latency using
   CUDA events and wall-clock measurements.
5. Run GPU0 smoke for coordinates, workbench support, minimum clearance, control
   direction, frequency, phase progression, and safe termination.
6. Verify the full mission path:
   `APPROACH -> PRELOAD -> GRASP -> LIFT -> HOLD -> LOWER -> RELEASE -> DONE`.
7. Execute the formal 30-trial GPU0 matrix and emit per-trial JSONL plus a
   SHA-pinned aggregate manifest.
8. Run a long-duration leak test for GPU memory, solver workspaces, worker
   processes, and recovery after fallback.

The formal manifest adds fields without removing existing ones:

- requested and actual backend per layer;
- OSQP, CUDA, driver, PyTorch, Isaac, and GPU identities;
- solver tolerances, iteration limits, and warm-start mode;
- CPU fallback counts and exact reasons;
- P50/P95/P99 setup, update, solve, transfer, and end-to-end times;
- recorded QP corpus SHA-256;
- existing asset, source, artifact, report, JSONL, and metadata pins.

## 14. Performance Gates

- G1: at least 2x end-to-end improvement over the recorded baseline.
- G3/G4: at least 10x cumulative improvement, including transfers and validation.
- G5: near-real-time Play is the target; any shortfall must be explained by a
  per-layer timing breakdown.
- No performance result may exclude startup or transfer costs without labeling
  the narrower measurement.
- No speedup may be obtained by loosening safety limits, reducing formal checks,
  skipping required control phases, or running a shorter simulated duration.

## 15. External References

- OSQP algebra backends: <https://osqp.org/docs/backends/index.html>
- OSQP Python CUDA installation: <https://osqp.org/docs/get_started/python.html>
- OSQP Python backend selection: <https://osqp.org/docs/backends/python.html>
- ProxSuite reference implementation: <https://github.com/Simple-Robotics/proxsuite>
- cuRobo documentation: <https://curobo.org/>
