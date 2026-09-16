# T500 GPU MPC + Latent Runtime Acceleration Design

## 1. Status

This specification records the user-approved A+C design and its approved RTI
revision: use a fixed-shape GPU SQP-RTI backend, adapted from the architectural
patterns in Go2Pvcnn `parallelism-amp` commit
`d080df994a7170486e74641dc75e28c344b7a3ee`, to predict the complete bimanual
action horizon; then compress that horizon into the distilled latent/student
path. The existing deterministic CPU implementation remains the reference and
safety fallback. OSQP CUDA remains a comparison and fallback backend rather than
the production hot path.

The design is approved section by section, including the RTI revision.
As of 2026-09-16, private CUDA contracts/CLI and persistent state, full KKT
condensation and warm-start infrastructure are implemented and independently
reviewed (`e280b8d`, `1d12e53`; focused GPU0 tests pass). The eager RTI planner
is awaiting a common effort/grasp/object transition-model decision after
read-only Task5 investigation; no planner code was added. Production expert
qualification is still blocked. See the
[verification record](../../../notes/log/2026-09-16-t500-gpu-state-dynamics-warm-start.md).
It does not claim that Triton or CUDA Graph execution is
already wired into T500, that any GPU backend is faster on this workload, or that
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
- Use fixed-shape GPU SQP-RTI, structured horizon solves, parallel line search,
  and CUDA Graph replay as the production prediction path.
- Reuse the remote branch's solver architecture and validation patterns, not its
  Go2-specific state, gait, terrain, or loss model.
- Keep GPU matrices and solver workspaces persistent across control cycles.
- Parallelize across horizon nodes, line-search candidates, both hands/arms, and
  training environments so small independent problems do not remain launch-bound.
- Preserve all existing T400/T500 public CPU `float64` contracts and default
  behavior.
- Preserve atomic acceptance, last-safe fallback, state-machine semantics, and
  formal SHA-pinned evidence.
- Require the production-approved DexManipNet-derived fingertip-motion artifact
  before enabling expert-guided manipulation MPC.
- Keep the fingertip expert prior and the full-action latent policy as two
  separately trained, separately pinned models with different responsibilities.
- Keep OSQP CUDA as an independently testable general-QP comparison and fallback.
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
- Do not copy Go2 gait scheduling, terrain EDT, footstep losses, or the remote
  fixed 18-state model into the bimanual controller.
- Do not reinterpret mixed-precision AMP training as MPC; the production teacher
  remains an online receding-horizon optimizer.
- Do not run training and interactive Play concurrently on the single GPU0.

## 5. Selected Architecture

```text
Isaac GPU state at time t
    -> private GPU float32 state mirror
    -> shift previous accepted H=25 horizon and inject measured x0
    -> frozen expert fingertip prior
       -> current five-tip palm-relative position/velocity + contact + phase
       -> short-horizon mixture distribution of natural fingertip velocities
    -> bimanual SQP-RTI teacher
       -> object pose/twist and left/right palm targets
       -> grasp-force allocation
       -> O6 Jacobian maps expert tip distribution into a defeasible soft cost
       -> structured horizon LQ/QP direction solve
       -> parallel safety-filtered line search
       -> complete 43-D action horizon
    -> full-horizon encoder -> latent z
    -> latent/student controller -> candidate complete action
    -> GPU WBC and Safety Projection
       -> accepted complete first action -> Isaac
       -> rejected result -> OSQP CUDA comparison/fallback when applicable
          -> rejected/unavailable -> CPU reference retry
             -> rejected CPU result -> last safe complete command
    -> at time t+1 inject the new measured state and repeat
```

The private GPU runtime may use CUDA tensors internally. Existing public
dataclasses continue to accept and return finite CPU `float64` tensors. CUDA
tensors must not leak through those boundaries. CPU mirrors are created only for
the legacy boundary, formal reporting, explicit reference comparison, or CPU
fallback.

The Play and Probe entrypoints gain an explicit backend request:

```text
--mpc-backend reference-cpu | bimanual-rti-cuda | auto
--qp-backend reference-cpu | osqp-cuda | auto
--control-device cuda:0
```

The MPC and generic QP selections are independent. MPC `reference-cpu` retains
the current teacher and `bimanual-rti-cuda` selects the production rolling
teacher. QP `osqp-cuda` selects the general-QP comparison/fallback path and fails
during startup if its pinned backend is unavailable. Either `auto` value may
choose only from a benchmarked, recorded allowlist and must report the actual
choice per layer.

## 6. Solver Strategy

The production backend follows the verified structure of the remote
`joint_mpc_rti` implementation:

```text
inject measured x0
shift the previous accepted horizon
linearize dynamics, kinematics, costs, and constraints in batch
build one structured Gauss-Newton LQ/QP subproblem
solve one SQP-RTI direction using a fixed horizon tree/scan
evaluate fixed line-search candidates in parallel
atomically accept one complete horizon or retain the safe warm start
publish only the first complete action
```

The T500 production dimensions remain its own frozen contract: `H=25` and a
complete `43`-D effective action at each horizon node. The implementation may
optimize a reduced object/palm/fingertip state internally, but the accepted
teacher output is always a complete 43-D action horizon before encoding to `z`.
It must not optimize or command all 59 articulation coordinates independently.

The structured solve exploits block locality across time. The horizon may be
padded to the next power-of-two tree size, but padded factors must be identity or
zero-cost factors and cannot alter the 25-node public horizon. Fixed-size SPD and
general solves may use Triton kernels after parity validation. Stable steady-state
execution may be captured by CUDA Graph only after all storage addresses and
control-flow shapes are fixed.

OSQP 1.x CUDA remains the general convex-QP comparison/fallback. Its canonical
form covers current convex subproblems:

```text
minimize 0.5 x' P x + q' x
subject to l <= A x <= u
```

The current environment's `osqp 0.6.7.post3` is not modified in place. A pinned
OSQP 1.x CUDA wheel is first tested in an isolated environment against CUDA 12.8,
PyTorch 2.7, Isaac Sim 5.1, and the RTX 5070. Triton, OSQP, CUDA runtime versions,
and accepted package/artifact SHA-256 values become part of the runtime manifest.

ProxSuite is not the primary choice because its documented batch acceleration is
CPU/OpenMP. cuRobo is not a direct replacement because the project has custom
object, dual-arm, dual-hand, force-allocation, WBC, and safety QPs. Neither choice
is prohibited for a later measured comparison.

## 7. Persistent Workspace And Data Flow

Each migrated layer owns fixed-address workspaces. Runtime updates change tensor
values without rebuilding solver objects or allocating variable-shape buffers in
the 200 Hz loop. The previous accepted trajectory is shifted and used as a warm
start only when its identity, shape, phase, and constraint layout match the
current problem. A reset invalidates only the affected environment row.

Static tensors such as identity matrices, selection matrices, joint-order maps,
constraint indices, and block-diagonal layouts are allocated once. State,
Jacobian, dynamics, solver values, and accepted solutions remain resident on
GPU. CUDA events measure setup, update, solve, validation, transfer, and total
cycle latency without adding per-step CPU synchronization to the normal path.

## 8. Parallelism And Problem Decomposition

- **Rolling teacher:** the previous 25-node solution is shifted, measured state
  replaces `x0`, and exactly one RTI correction is computed at each teacher tick.
- **Object and palms:** object pose/twist, left/right palm targets, mounting-base
  coupling, and grasp-force allocation form the reduced task-space prediction.
- **Arms and hands:** left/right arm blocks and O6 fingertip-relative blocks are
  assembled in one structured solve. True O6 Jacobians map fingertip references
  into joint-space contributions. The expert prior remains a defeasible soft
  cost and cannot weaken contact, limit, or collision constraints.
- **Complete action:** every accepted prediction reconstructs all 43 effective
  controls for all 25 nodes before the encoder computes `z`; `z` never hides an
  incomplete or partially accepted teacher action.
- **WBC/Safety:** high-frequency WBC and safety projection retain persistent
  workspaces and warm starts. Small layers remain eligible for CPU execution
  under `auto` if transfer-inclusive latency is lower.

Training obtains the remote design's natural batch dimension from many Isaac
environments. Single-environment Play must not fake 1024 identical environments.
It instead batches the 25 horizon nodes, fixed line-search candidates, both
sides, constraint families, and—where useful—small robust model/contact
scenarios. GPU use is accepted only when end-to-end timing beats the CPU path.

In fixed-base mode, equal lower and upper action bounds define a unique action.
The runtime bypasses optimization, validates the bounded nominal trajectory, and
emits the same diagnostics contract.

Aggregation must preserve joint order, phase ownership, per-side diagnostics,
and the rule that a failed side cannot be partially committed.

### 8.1 Expert Fingertip-Motion Prior

The expert prior is learned before expert-guided MPC is enabled. The existing
offline pipeline converts DexManipNet motion with each source hand's URDF/FK into
hand-agnostic, palm-relative five-fingertip geometry. Its frozen runtime contract
is:

```text
input per hand:
  fingertip_position_palm  [5, 3]
  fingertip_velocity_palm  [5, 3]
  contact_mask             [5]
  phase_one_hot            [7]

output per hand:
  four-component diagonal Gaussian mixture
  future tip velocity      [4, 20, 5, 3] at 100 Hz
```

Thus every fingertip contributes a six-dimensional current state—three position
and three velocity coordinates—while contact and task-independent phase provide
context. The network receives no task ID, desired object trajectory, desired palm
pose, object identity, or semantic operation label. It expresses only how finger
motion tends to evolve naturally.

At each applicable RTI teacher tick, the validated frozen artifact produces
mixture logits, means, and diagonal uncertainty. The runtime deterministically
selects the component with the lowest frozen baseline-compatibility score,
resamples that component to the MPC time grid, and constructs a
covariance-weighted fingertip velocity residual. The prior covers only its frozen
0.20-second window (`20 * 0.01s`); MPC nodes beyond that time receive zero expert
weight rather than an extrapolated target. For each O6 hand:

```text
v_tip(q, qdot) = J_tip_O6(q) qdot
L_expert = w_phase * robust_norm(v_tip - mean, precision)
```

`J_tip_O6` is computed from the actual left/right O6 kinematic chain in the
normalized asset, not from a source hand and not from an identity approximation.
Left/right mirroring follows the frozen artifact metadata. Contacted fingertip
directions remain governed by hard contact constraints; the expert precision is
masked or reduced where it would conflict with contact, limits, collision,
force-closure, or object/palm tracking. Consequently MPC can perform operations
absent from the expert dataset while retaining natural finger coordination.

The prior is fail-closed. Missing or mismatched metadata SHA, unapproved metrics,
timeout, non-finite output, invalid precision, phase exclusion, or Jacobian error
disables only `L_expert` for that tick and records the exact reason. It does not
invalidate an otherwise safe MPC result and never changes hard constraints.

This fingertip model is distinct from the later full-action latent path. The
fingertip model regularizes the MPC teacher; the full-action encoder compresses
the teacher's accepted `[25, 43]` trajectory into `z`, and the student uses `z`
for high-rate control.

## 9. Acceptance And Fallback

Every RTI line-search candidate is checked on GPU before selection:

- finite shape, dtype, device, and joint ordering;
- equality residual within the existing layer tolerance;
- inequality, effort, velocity, acceleration, and joint-limit violation within
  the existing layer tolerance;
- dynamics and contact residuals within existing thresholds;
- complete left/right and whole-command atomicity;
- correct solver status, iteration count, and workspace identity.

Failure handling is deterministic:

1. A valid RTI horizon is atomically committed and becomes the next warm start.
2. If no line-search candidate is safe or improves the accepted merit, retain
   the shifted last-safe horizon and record the reject mask.
3. A timeout, non-finite result, CUDA error, or excessive residual may trigger
   OSQP CUDA on the same immutable convexified subproblem.
4. An invalid or unavailable GPU fallback triggers the CPU reference solver.
5. If CPU also rejects the input, the most recent safe complete command is held.
6. Existing consecutive-failure thresholds control safe lowering and terminal
   state transitions.

No backend may silently relabel a fallback run. Startup failures occur before
Isaac scene creation when an explicitly requested backend, CUDA device, Triton
kernel capability, or pinned library identity is unavailable. CUDA Graph capture
failure may fall back to eager GPU RTI only when the cause is classified and the
same safety checks remain active.

## 10. Determinism

- CPU remains the strict deterministic reference.
- GPU uses fixed problem ordering, fixed horizon-tree topology, one RTI update,
  fixed line-search alphas, fixed active-set refinement counts, deterministic
  warm-start ownership, and TF32 disabled for critical solves.
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
- Pin the remote reference commit. Import no remote file without a file-level
  adaptation record identifying retained algorithmic behavior and removed
  Go2-specific assumptions.
- Preallocate workspaces under a documented GPU memory budget suitable for the
  12 GB RTX 5070.
- Training and Play are mutually exclusive GPU0 workloads.
- Offline dataset conversion must remain CPU-only and must not create a CUDA
  context.
- Close and release every solver workspace during normal exit, safe termination,
  startup rejection, and exception unwinding.

## 12. Delivery Stages

### E0: Expert Artifact Readiness

Finish deterministic DexManipNet conversion, train and distill the fingertip
mixture model, pass its frozen accuracy/latency/provenance gates, and produce a
production-approved artifact plus externally pinned metadata SHA-256. MPC may be
developed and benchmarked without the soft prior, but expert-guided Play cannot
be declared ready before E0 passes.

### G0: Capture And Benchmark

Capture immutable real QP samples from seed 42 at 400 and 4000 steps. Measure
problem construction, setup, update, solve, validation, transfer, rendering, and
end-to-end step latency. Record the sample corpus SHA-256.

### G1: Solver-Independent Fast Paths

Implement the fixed-base teacher unique-solution path and cache invariant CPU
tensors. Require exact reference behavior. Target at least 2x end-to-end speedup
without installing OSQP CUDA.

### G2: GPU Tensor Model And RTI Parity

Add private float32 GPU state, reduced bimanual dynamics/kinematics blocks,
25-node warm shifting, one RTI update, and parallel line search in eager PyTorch.
Add batched GPU fingertip-prior inference and true left/right O6 fingertip
Jacobians. Compare every intermediate tensor, prior residual, and accepted first
action with frozen CPU fixtures before enabling runtime use.

### G3: Structured Scan And Fixed Solves

Adapt the fixed horizon tree, associative scan, and fixed-size Triton SPD/general
solves to T500 dimensions. Validate dense/scan direction parity, KKT residuals,
active constraints, and atomic rejection. Add OSQP CUDA as the independently
pinned comparison/fallback backend. Target at least 10x cumulative speedup.

### G4: CUDA Graph And GPU-Resident Runtime

Preallocate fixed-address workspaces, capture steady-state RTI execution, and
remove control-loop CPU round trips. Retain CPU mirrors only at approved
boundaries. Profile single-environment and multi-environment modes separately for
hidden synchronization, launch overhead, and unbounded workspace growth.

### G5: Latent/Student Integration

Run the full-action latent encoder/student on GPU while retaining the already
validated GPU fingertip-prior inference inside the teacher. Use the complete MPC
teacher at low frequency, while the latent controller and GPU WBC/Safety remain
high frequency. Trigger correction or fallback when the compact policy departs
from the accepted teacher envelope. Target near-real-time interactive Play
without reducing safety checks.

Each stage is independently benchmarked, reviewable, and reversible. A later
stage cannot compensate for a failed earlier correctness gate.

## 13. Verification

1. Preserve all CPU contract and unit tests.
2. Add backend tests for malformed input, device mismatch, unavailable CUDA,
   workspace identity, horizon reset/shift, warm start, CUDA Graph recapture,
   timeout, non-convergence, NaN/Inf, residual rejection, and atomic fallback.
3. Compare eager RTI, scan RTI, OSQP CUDA, and CPU reference on captured feasible
   and infeasible cases, active limits, mimic behavior, repeated solve behavior,
   KKT residuals, and accepted first actions.
4. Verify the expert prior's exact 42-dimensional field ordering, five-tip order, left/right
   reflection, 20-step distribution shape, uncertainty bounds, real O6 Jacobian
   directional derivatives, contact masking, soft-cost override, timeout,
   non-finite rejection, metadata SHA pin, and prior-disabled fallback.
5. Demonstrate that changing the object/palm task with identical finger state,
   contact, and phase does not change the prior distribution; then demonstrate
   that MPC can depart from the prior when a hard safety or task constraint
   requires it.
6. Benchmark setup, update, prior inference, linearization, scan solve, line
   search, transfer, validation, and total latency using
   CUDA events and wall-clock measurements.
7. Run GPU0 smoke for coordinates, workbench support, minimum clearance, control
   direction, frequency, phase progression, and safe termination.
8. Verify the full mission path:
   `APPROACH -> PRELOAD -> GRASP -> LIFT -> HOLD -> LOWER -> RELEASE -> DONE`.
9. Execute the formal 30-trial GPU0 matrix and emit per-trial JSONL plus a
   SHA-pinned aggregate manifest.
10. Run a long-duration leak test for GPU memory, solver workspaces, worker
   processes, and recovery after fallback.

The formal manifest adds fields without removing existing ones:

- requested and actual backend per layer;
- remote reference commit plus Triton, OSQP, CUDA, driver, PyTorch, Isaac, and GPU
  identities;
- horizon-tree topology, RTI count, line-search alphas, active-set refinements,
  solver tolerances, iteration limits, and warm-start mode;
- CPU fallback counts and exact reasons;
- fingertip prior artifact/metadata/dataset/teacher-ensemble hashes, enabled
  ticks, disabled ticks, and exact disable reasons;
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
- Single-environment Play and batched training results must be reported
  separately; a 1024-environment throughput number cannot establish Play latency.

## 15. External References

- Go2Pvcnn `parallelism-amp` pinned source:
  <https://github.com/lukeyang117/Go2Pvcnn/tree/d080df994a7170486e74641dc75e28c344b7a3ee>
- Remote SQP-RTI orchestration:
  <https://github.com/lukeyang117/Go2Pvcnn/blob/d080df994a7170486e74641dc75e28c344b7a3ee/Go2Pvcnn/extension/joint_mpc_rti/solver/sqp_rti.py>
- Remote associative trajectory scan:
  <https://github.com/lukeyang117/Go2Pvcnn/blob/d080df994a7170486e74641dc75e28c344b7a3ee/Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py>
- Remote CUDA Graph runtime:
  <https://github.com/lukeyang117/Go2Pvcnn/blob/d080df994a7170486e74641dc75e28c344b7a3ee/Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py>
- OSQP algebra backends: <https://osqp.org/docs/backends/index.html>
- OSQP Python CUDA installation: <https://osqp.org/docs/get_started/python.html>
- OSQP Python backend selection: <https://osqp.org/docs/backends/python.html>
- ProxSuite reference implementation: <https://github.com/Simple-Robotics/proxsuite>
- cuRobo documentation: <https://curobo.org/>
