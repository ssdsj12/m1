# Task 6 Report

Status: DONE_WITH_CONCERNS

Git Ref: unavailable

## User-Approved Round 4 Contract

The user explicitly approved a fourth round after the prior bounded stop. For each case, apply the unchanged wrench for `10` transition steps whose measurements are discarded, then collect a fresh `50`-sample evaluation window. A case passes only when the excited-channel baseline-subtracted mean has the expected sign, at least `45/50` samples have that sign (`sign_fraction >= 0.90`), `magnitude_ratio > 0.20`, and all transition/evaluation data are finite with no reset. JSON must record `transition_steps`, `sample_steps`, `sign_count`, `sign_fraction`, the measured/baseline-subtracted mean, ratio, and pass. Force/torque magnitudes, coordinate conversion, clear shim, and all other architecture remain unchanged.

## User-Approved Independent-Reset Round

After Round 4 isolated `force_z bad_orientation` as cumulative cross-axis pose drift, the user approved deterministic case isolation. Keep one environment, but before the initial settle row and before every axis: clear the external wrench, call `env.reset()`, require a valid reset result and unchanged 25-DOF/unique-body topology, run 100 zero-action settle steps, then collect a new independent 50-step baseline. Apply the unchanged case load for 10 discarded transition steps and 50 evaluation samples. Mean sign, sign fraction `>=0.90`, ratio `>0.20`, finite/no-reset gates remain unchanged. Do not alter the loads, termination policy, coordinate transform, or clear shim.

## Outcome

Implemented the deterministic one-env CPU wrench probe, the user-approved transition/fraction contract, and independent reset/re-equilibration for every axis. The final fresh CPU authority exited `0` and atomically published seven finite, passing JSONL rows. All six excited channels had the expected parent-on-child reaction sign in `50/50` samples, no case reset, and every magnitude ratio exceeded `0.20`.

## Files

- Created `Go2Pvcnn/scripts/m1_panda_wrench_probe.py`.
- Created `Go2Pvcnn/tests/test_m1_panda_wrench_probe_static.py`.
- Created `notes/log/2026-08-14-m1-panda-wrench-probe.md`.
- Updated the T400 branch/dashboard, log index, and design checkpoint.
- Created `Go2Pvcnn/tests/artifacts/m1_panda_wrench_probe.jsonl` through same-directory atomic replacement after the complete seven-row validation passed.

## Probe Contract Implemented

- Public Gym ID `Isaac-M1-Panda-Smoke-v0`; actual `gym.make`, reset, and step.
- One CPU environment and zero 16-dimensional M1 action. The initial settle row and every axis use an independent deterministic reset, 100 settle steps, 50 baseline steps, 10 discarded transition steps, and 50 evaluation samples.
- Exact unique lookup of `panda_hand`, `panda_link0`, and `BASE_LINK`; distinct IDs; 25 DOF.
- Verbatim six-case table at `20 N` and `5 N·m`.
- Desired wrench is specified in `BASE_LINK` frame and transformed each step through current base/world/hand orientations into Panda-hand local axes before `set_external_force_and_torque`.
- Clear uses `torch.zeros(0, 3, device=robot.device)`; a narrow Isaac Lab 2.1 compatibility path verifies the API already disabled `has_external_wrench` before accepting only its known empty-index shape mismatch.
- Finite/no-reset checks, baseline subtraction, expected parent-on-child reaction sign, mean-sign check, `sign_fraction >= 0.90`, and strict `magnitude_ratio > 0.2`.
- Artifact publication uses a same-directory temporary file, flush/fsync, and `os.replace` before Kit close; failures reliably exit nonzero and cannot partially overwrite the last valid artifact.

## RED / GREEN

- Initial RED: exit `1`, `5 failed` because the probe was absent.
- Initial GREEN: exit `0`, `5 passed`.
- Lifecycle repair RED/GREEN: focused exit `1` then full focused exit `0`.
- Empty-clear selector repair RED/GREEN: focused exit `1` then focused suite exit `0`; real hypothesis rejected.
- Exact empty-clear compatibility repair RED/GREEN: focused exit `1` then final focused `8 passed`, exit `0`.
- Final planned static regression: `50 passed in 0.88s`, exit `0`.
- `py_compile`: exit `0`.

## Three Real Repair Rounds

1. Apparent exit `0` with no artifact/success JSON. Root cause was post-`simulation_app.close()` code being unreachable on this Kit installation. Artifact/nonzero commit moved before close.
2. Reliable exit `1` at empty clear: Isaac Lab attempted `[0] -> [29,3]`. Empty body selector changed the target to zero rows, but the implementation still rejected `[0] -> [0,3]`.
3. Retained the mandated empty API and accepted only its exact known shape mismatch after `has_external_wrench=False`. Real dynamics then reached calibration and exited `1` on `force_y`.

At that historical checkpoint no fourth repair was attempted; the later user approval above explicitly authorized Round 4.

## Real Dynamics Evidence

- One public env created/reset/stepped on CPU; 16 actions; 25 DOF; required bodies resolved uniquely and distinctly; no finite/reset failure preceded calibration.
- `settle` and `force_x` passed in-process, but their exact rows were intentionally not persisted because the complete seven-row run failed.
- `force_y` applied `[0,20,0,0,0,0]` in `BASE_LINK` frame.
- Baseline-subtracted mean: `[10.150298118591309, -27.970840454101562, -1.3658764362335205, 26.806659698486328, 27.21747589111328, -3.362870454788208]`.
- Excited `Fy`: expected sign `-1`; mean `-27.970840454101562`; magnitude ratio `1.3985420227050782`; 47/50 expected-sign samples; sign fraction `0.9399999976158142`; `stable_sign=false`; pass false.
- Historical Round 3 stopped before `force_z` and torque cases; Round 4 superseding evidence is recorded below.

## Verification And Environment

- Generated checksum manifest: `2/2`, exit `0`.
- Hashes: `panda.usd=1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`; `m1_panda.usd=6acbd32afab08dbfb8963e0f7d990d2988cdfe8ad4fec083d0c9fa1c4585c3ff`.
- CPU verifier exit `0`: dependencies `8`, remote `0`, outside-root `0`, unresolved `0`, one articulation root, 25 DOF, 29 body names, 25 joint names, physics steps `1`, validation errors empty.
- `unshare --net true` exit `1` (`Operation not permitted`): dependency closure passed, network-denial execution unverified. No host networking was changed.
- Planned `isaaclab.sh -p` default-GPU route was not authority and was not rerun because the local RTX 5070 is `sm_120` while installed PyTorch supports only through `sm_90`; CPU was the required real authority.

## Round 4 TDD And Real Result

- User contract RED: exit `1`, `4 failed, 6 passed` for missing `TRANSITION_STEPS=10`, new `SAMPLE_STEPS=50`, 0.90/0.88 boundary behavior, mean-sign gate, and JSON schema.
- GREEN: `10 passed`, exit `0`; a diagnostic RED/GREEN then added exact termination evidence, final focused `11 passed`.
- Each unchanged load is now applied for `10 + 50` steps; only the final 50 are evaluated. `sign_count` is integer, `sign_fraction=sign_count/50`, and pass requires finite data, mean expected sign, fraction `>=0.90`, and ratio `>0.20`.
- First Round 4 authority run exit `1` at an unexpected reset during `force_z`. No temporary artifact was written.
- A single read-only diagnostic hypothesis was added and the identical authority rerun from the beginning. It again exited `1` during `force_z` with `terminated=[true]`, `truncated=[false]`, `base_contact=[false]`, `bad_orientation=[true]`, `time_out=[false]`.
- Reaching `force_z` proves `force_x` and `force_y` passed the updated gate in both runs. Exact case rows were not persisted because output remains all-or-nothing.
- The temporary and formal artifact paths are both absent; atomic replacement was not attempted.

## Fresh Round 4 Verification

- Planned static regression: `53 passed in 0.87s`, exit `0`.
- `py_compile`: exit `0`.
- Generated checksum: `2/2`, exit `0`.
- PXR behavior: exit `0`, `cleanup=pass`, `mount=pass`, root `/M1Panda/BASE_LINK`.
- CPU verifier: exit `0`, dependencies `8`, remote/outside/unresolved `0`, one articulation root, 25 DOF, 29 bodies, 25 joints, physics step `1`, validation errors empty.
- JSONL schema/7-row/all-pass validation could not run because no candidate artifact was produced.
- Network denial remains unverified; it was not retried or changed.

## Architecture Options (Historical; Round 4 Selected Option 2)

1. Pre-approve a stable-sign sample fraction threshold while keeping the magnitude gate strictly `>20%`.
2. Add a transition/ramp or discard interval, then evaluate a fresh 50-sample steady-state window.
3. Deterministically reset/re-equilibrate between axes while retaining 50 baseline and 50 loaded samples per case.

The user selected the transition-window option for Round 4 and then explicitly approved deterministic independent reset/re-equilibration after the cumulative `force_z bad_orientation` failure. The final result is recorded below. Residual policy, Teacher–Student training, IK/OSC, grasping, sensor driver, mechanical validation, and real-hardware validation remain open and were not implemented.

## User-Approved Independent-Reset Completion

- TDD added reset/baseline isolation and approved-window behavior. Final focused probe suite after the single-agent atomic-write/clear-shim review: `16 passed`; final planned six-file regression: `58 passed in 0.85s`; `py_compile` exit `0`.
- The final authority command used the compatible loco interpreter with `--device cpu --headless` and a process timeout of `360` seconds. It exited `0` in `14.06s` and printed `{"rows": 7, "pass": true}`.
- The formal artifact was atomically updated at `Go2Pvcnn/tests/artifacts/m1_panda_wrench_probe.jsonl`; schema validation exited `0` with exactly `settle`, `force_x`, `force_y`, `force_z`, `torque_x`, `torque_y`, and `torque_z`.
- Every axis recorded `sign_count=50`, `sign_fraction=1.0`, `mean_expected_sign=true`, `finite=true`, `unexpected_reset=false`, and `pass=true`.

| Case | Baseline-subtracted excited channel | Magnitude ratio | Sign samples | Result |
| --- | ---: | ---: | ---: | --- |
| `force_x` | `-34.114151 N` | `1.705708` | `50/50` | pass |
| `force_y` | `-35.731705 N` | `1.786585` | `50/50` | pass |
| `force_z` | `-18.383438 N` | `0.919172` | `50/50` | pass |
| `torque_x` | `-6.345034 N·m` | `1.269007` | `50/50` | pass |
| `torque_y` | `-53.942345 N·m` | `10.788469` | `50/50` | pass |
| `torque_z` | `-5.900999 N·m` | `1.180200` | `50/50` | pass |

- Fresh generated checksum verification passed `2/2`. Fresh PXR behavior exited `0` with cleanup/mount pass and root `/M1Panda/BASE_LINK`.
- Fresh CPU verifier exited `0`: dependencies `8`, remote/outside/unresolved `0`, one articulation root, 25 DOF, 29 bodies, 25 joints, physics step `1`, and no validation errors.
- Network-denial execution remains unverified because the safe `unshare --net` attempt was not permitted. This phase proves local dependency closure and a successful runtime with OmniHub inaccessible, but does not claim a process-enforced network-denial run.
- Remaining warnings are recorded boundaries: built-in `OmniPBR.mdl`, disabled Panda source-root disjoint-transform warning, Isaac Lab actuator deprecation warnings, and the local CUDA `sm_120` mismatch that makes CPU the authority.

Task 6 and the asset/wrench foundation child are complete with the network-denial limitation above. Git Ref: unavailable.
