# T500 OakInk Sequence Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make sequence audit and rollout loading consume the same verified optional OakInk geometry resolver.

**Architecture:** Extend only the offline source loader with a keyword-only verified resolver. Preserve default and FAVOR behavior, and only replace genuinely missing safe OakInk geometry references. CLI/aggregate provenance and real corpus publication are separate reviewed steps.

**Tech Stack:** Python, pytest, existing strict HDF5 loader and geometry overlay verifier.

## Global Constraints

- Preserve original archives, source checkout and run_e unchanged.
- External manifest SHA is supplied by the caller, not trusted from a self-declaration.
- Resolver handles only declared OakInkV2 references; default/FAVOR resolution remains unchanged.
- No public T400/T500 contract changes, training entrypoint, contact MPC, perception or Residual.
- Do not bypass invalid source paths, source symlinks, interaction-side gates or any physical preprocessing geometry checks.
- This task does not implement CLI or aggregate provenance and cannot claim real geometry/corpus acceptance.

### Task 1: Shared optional geometry resolver in audit/load

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_dexmanipnet.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_geometry_sequence.py`

**Interfaces:**
Existing `verify_geometry_overlay(root: Path, manifest_path: Path, expected_sha256: str) -> VerifiedGeometryResolver` is reviewed at `bcf43d8`. Require an actual VerifiedGeometryResolver (not duck-typed callbacks) when supplied. New public signatures:
```python
def audit_sequence(path: str | Path, source: str, side: str, *, geometry_resolver: VerifiedGeometryResolver | None = None) -> SequenceAudit: ...
def load_best_successful_rollout(path: str | Path, source: str, side: str, *, geometry_resolver: VerifiedGeometryResolver | None = None) -> LoadedHandSequence: ...
```
Propagate identical object through `_load_best` and `_resolve_object_geometry`; add `source` to the latter's private arguments. Validate resolver type before loading or returning an audit, raising ValueError on wrong type. For source `oakinkv2`, validate original normalized POSIX relative URDF reference before optional fallback; reject absolute/traversal/backslash/empty-component/dot-component paths as `invalid_object_geometry`. Existing present regular geometry wins, even when overlay is supplied. A missing candidate may fall back to the declared overlay, but symlink/nonregular/non-directory parent is invalid and never overridden. Resolver failures become `_reject("invalid_object_geometry")`; no string/path guessing. FAVOR ignores a correctly typed overlay and uses its existing local geometry resolution. None preserves all historical default outputs, audit reasons and loaded array bytes. Do not change source hash calculation; resolved URDF participates in the existing hash.

- [ ] **Step 1: Add real fixture tests before implementation.** Reuse existing real sequence/HDF5 fixture construction patterns, and construct a valid pinned overlay with a minimal zero-origin/unit-scale declared URDF/mesh. Do not use mocked resolvers. Include:
```python
def test_missing_oakink_geometry_uses_same_verified_overlay(sequence_and_overlay):
    sequence, resolver, expected_urdf = sequence_and_overlay
    audit = audit_sequence(sequence, "oakinkv2", "rh", geometry_resolver=resolver)
    assert audit.accepted
    loaded = load_best_successful_rollout(sequence, "oakinkv2", "rh", geometry_resolver=resolver)
    assert loaded.object_geometry_path == expected_urdf
    assert loaded.source_sha256 == audit.input_sha256
```
Test default still rejects that missing geometry, unknown reference and byte drift reject both paths, invalid original paths and source symlink cannot be overridden, valid original geometry wins, FAVOR exact equality with/without overlay, source/interactions remain rejected, and arbitrary resolver object raises ValueError before work. Make the overlay fixture byte declarations explicit rather than sharing implementation helpers.

- [ ] **Step 2: Verify RED.**
```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_expert_prior_geometry_sequence.py -q
```
Expected: fails because optional resolver keyword is unsupported.

- [ ] **Step 3: Implement minimal source integration.** Use existing missing/invalid distinction from `_contained_regular_file` and the reviewed resolver. Preserve direct-source resolution before fallback. Keep all changes localized to parameter plumbing, source reference validation and verified fallback; no loader restructuring.

- [ ] **Step 4: Verify focused GREEN and source regressions.**
```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_expert_prior_geometry_sequence.py tests/test_m1_bimanual_expert_prior_geometry_overlay.py tests/test_m1_bimanual_expert_prior_dexmanipnet.py -q
```
Expected: all pass, no warnings. Run scoped whitespace checks.

- [ ] **Step 5: Scoped commit and report.**
```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_dexmanipnet.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_geometry_sequence.py
git commit -m "feat: share verified geometry resolver across sequence audit and load"
```
Report exact red/green command/output, commit and concerns; no real corpus acceptance claim.
