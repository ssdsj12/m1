# T500 Production Training and GPU Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the approved independent OakInk geometry overlay into a qualified dual-source expert/student artifact and a benchmarked, fail-closed GPU Play path.

**Architecture:** Preserve all original archives, source checkouts, `run_e`, CPU contracts, and public `[B,25,43]` actions. Generate a new immutable geometry bundle and external-SHA manifest for all 90 compatible object references, write a fresh dual-source conversion, qualify expert/student artifacts against that new aggregate, then connect the already-reviewed eager GPU RTI planner behind explicit runtime and benchmark gates. The Play backend remains unavailable until every gate emits evidence.

**Tech Stack:** Existing `go2` Python environment, isolated CoACD geometry environment, NumPy/trimesh/XML, PyTorch CUDA0, pytest, Isaac Sim/Isaac Lab only for the final smoke and formal trial gates.

## Global Constraints

- Original OakInk/FAVOR archives, ManipTrans checkout, source tree, and `converted/run_e` are immutable evidence and are never overwritten.
- All 90 unique OakInk geometry references must pass generation, byte SHA reproducibility, URDF/FK, collision, frame, scale, and contact checks before production conversion is eligible.
- Actions remain original and diverse; 90 geometry references are not 90 identical trajectories.
- External geometry manifest SHA and new aggregate SHA are caller-supplied evidence, never self-approved by the generated artifact.
- Production expert/student artifacts must bind the new aggregate SHA, geometry manifest SHA, split identity, seeds, metrics, and canonical self-hashes; synthetic or FAVOR-only artifacts cannot be relabeled production.
- Public control remains `[B,25,43]`; private 12D wrench never enters action, warm storage, latent inputs, or execution commands.
- `bimanual-rti-cuda`, `osqp-cuda`, and `auto` remain rejected until runtime, benchmark, and Isaac gates pass.
- No contact-mode MPC, perception, Residual, task/object-ID expert input, or public T400/T500 contract changes.

---

### Task 1: Generate and Verify the 90-Reference Geometry Bundle

**Files:**

- Create: `Go2Pvcnn/scripts/m1_oakink_geometry_bundle.py`
- Create: `Go2Pvcnn/tests/test_m1_oakink_geometry_bundle.py`
- Reuse without modification: `expert_fingertip_prior/geometry_overlay.py`, existing OakInk source audit/load and mesh validation helpers.

**Interfaces:**

```python
def collect_compatible_references(sequence_root: Path) -> tuple[str, ...]: ...
def build_geometry_bundle(*, raw_archive: Path, sequence_root: Path,
                          output_root: Path, recipe: dict[str, object],
                          expected_upstream: dict[str, str],
                          jobs: int = 1) -> Path: ...
def verify_geometry_bundle(output_root: Path, expected_manifest_sha256: str,
                           expected_count: int = 90) -> dict[str, object]: ...
```

- [ ] Write RED fixtures for exact reference collection, raw member coverage, unsafe output paths, failed mesh checks, recipe drift, nonzero origin/unit-scale violations, duplicate references, second-run SHA mismatch, and a successful two-object miniature bundle.
- [ ] Run the focused RED command and record the expected missing-module/entrypoint failure.
- [ ] Implement one-object-at-a-time deterministic generation with fixed seed/thread limits, atomic staging, immutable manifest, external SHA verification, and 90/90 count enforcement. A failed object writes diagnostics only and prevents publication of the production manifest.
- [ ] Run focused tests plus existing geometry overlay/sequence/mesh regressions; run a two-pass miniature generation and compare every output SHA.
- [ ] Run the real bundle command against the 90 references with `jobs=1` first, record peak memory/timing, and do not claim qualification until `verify_geometry_bundle` reports 90/90.
- [ ] Commit only the script/tests and write `docs/superpowers/logs/2026-09-20-t500-geometry-bundle.md` with manifest SHA, reference counts, tool identities, and rejected-object diagnostics.

### Task 2: Fresh Dual-Source Conversion and Aggregate Qualification

**Files:**

- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_convert_dexmanipnet.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/storage.py`
- Create: `Go2Pvcnn/tests/test_m1_dual_source_geometry_conversion.py`
- Create: `.superpowers/sdd/production-dual-source-report.md`

**Interfaces:**

```python
def convert_dual_source(*, source_root: Path, output_root: Path,
                        geometry_manifest: Path, geometry_sha256: str,
                        expected_sources: tuple[str, ...] = ("favor", "oakinkv2")) -> Path: ...
```

- [ ] Add RED tests proving missing/incorrect geometry SHA blocks before output publication, original `run_e` is untouched, all accepted OakInk sides have geometry provenance, and FAVOR resolution remains unchanged.
- [ ] Add the geometry manifest SHA, per-reference geometry SHA, source counts, accepted/rejected reasons, and split identity to the new aggregate body before its canonical self-hash.
- [ ] Convert into a new output directory using the same resolver for audit and load; require both FAVOR and OakInkV2 accepted samples and group-exclusive splits.
- [ ] Verify all shard/audit/aggregate bytes twice and reject drift; run existing `verify_aggregate_manifest` and source regressions.
- [ ] Do not start training automatically. Produce a qualification report that either blocks with exact rejected references or records a new aggregate SHA and accepted source counts.
- [ ] Commit scoped conversion/storage/tests and report.

### Task 3: Production Expert Ensemble and Student Distillation

**Files:**

- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_train_fingertip_expert.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_distill_fingertip_prior.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/artifact.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/ensemble_artifact.py`
- Create: `Go2Pvcnn/tests/test_m1_production_artifact_gates.py`

- [ ] Write RED tests for new aggregate/geometry SHA binding, source-count minimums, synthetic provenance rejection, deterministic three-member seeds, checkpoint self-hash, student distillation identity, and pairwise artifact/metadata SHA requirements.
- [ ] Make training accept only a freshly verified dual-source aggregate; retain `--synthetic-smoke` as nonproduction and never permit it to set `production_approved=true`.
- [ ] Train/resume three expert members, select by held-out mixture metrics, distill the student, and emit canonical metadata with aggregate/geometry/split/training identities.
- [ ] Run independent deterministic repeat on the same aggregate and compare latency-excluding fingerprints; verify `weights_only` loading and runtime metadata checks.
- [ ] Commit scoped gates/tests and write the production training report. If Task 2 is blocked, stop here without creating a production artifact.

### Task 4: Connect the Reviewed GPU Planner to Runtime

**Files:**

- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/gpu_rti/config.py`
- Create: `Go2Pvcnn/tests/test_m1_gpu_rti_runtime_binding.py`

- [ ] Write RED tests for CUDA input packing, planner invocation, public 43D action extraction, reason/last-safe diagnostics, CPU parity, artifact SHA propagation, and fail-closed startup.
- [ ] Implement an explicit runtime adapter that constructs resident `CoupledRtiInput`/`GpuReducedDynamics`, calls one planner step, and maps only effort43 back to the wrapper. Preserve CPU fallback and existing termination/reset behavior.
- [ ] Keep `bimanual-rti-cuda` unavailable until benchmark evidence is supplied; add no silent fallback from an explicitly requested GPU backend.
- [ ] Integrate production student only as a soft prior after artifact/metadata SHA validation; prior failure removes only the soft term.
- [ ] Run focused runtime tests and all Gate A–C regressions. Do not open Play from this task alone.

### Task 5: Benchmark, Isaac GPU0 Smoke, and Formal 30-Trial Gate

**Files:**

- Create: `Go2Pvcnn/scripts/m1_gpu_rti_benchmark.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Create: `Go2Pvcnn/tests/test_m1_gpu_rti_benchmark_manifest.py`
- Create: `Go2Pvcnn/tests/artifacts/m1_gpu_rti_benchmark.manifest.json`
- Create: `Go2Pvcnn/tests/artifacts/m1_gpu_rti_formal_aggregate.manifest.json`

- [ ] Write RED tests for benchmark environment/device identity, p50/p95/p99 latency, memory bounds, exact H25/batch/config, SHA-pinned command/source identity, and rejection of stale/missing manifests.
- [ ] Benchmark CPU parity and GPU planner after warmup; require the configured performance threshold from the approved runtime gate before enabling explicit GPU backend selection.
- [ ] Run Isaac GPU0 smoke for coordinates, workbench, minimum distance, control direction, frequency, reset, safe DONE/TERMINATED window handling, and diagnostics JSONL.
- [ ] Run 30 formal trials with fixed seeds, one JSONL record per trial, and a SHA-pinned aggregate manifest. Require all 30 to pass; otherwise keep Play fail-closed.
- [ ] Only after all gates pass, change config selection to permit `bimanual-rti-cuda`; retain explicit rejection tests for missing or stale benchmark/formal manifests.
- [ ] Commit benchmark/verification artifacts and publish the final production status report. Do not claim success without fresh command output and manifest hashes.

## Execution Order and Stop Conditions

Tasks are sequential: Task 2 cannot start until Task 1 publishes a 90/90 geometry manifest; Task 3 cannot start until Task 2 publishes a dual-source aggregate; Task 4 can develop prior-off against the reviewed planner but production prior-on requires Task 3; Task 5 cannot open GPU Play until Tasks 3–4 and all formal evidence pass. Any failed gate leaves the previous safe backend active and records exact JSON/JSONL diagnostics.
