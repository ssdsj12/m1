# T500 DexManipNet Runtime External Metadata Pin

## Purpose

Close the remaining trust boundary between an externally recorded production
`metadata.json` SHA-256 and the frozen O6 fingertip-prior runtime. This change
does not add training, contact MPC, perception, Residual control, or a new
T400/T500 task contract.

## Stage And Related Todo

T500.5 / Task 11 runtime deployment gate; see the
[T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Contract

- `FrozenO6FingertipPrior.from_artifact()` now requires the caller-owned
  `expected_metadata_sha256` and forwards it to the strict artifact loader
  before any worker is created.
- Wrapper prior-on configuration is an atomic pair: artifact path plus one
  64-character lowercase metadata SHA-256. Missing peers fail before reading
  `env.unwrapped`; the same pin is used for the independent left and right
  workers. Prior-off remains both fields `None` and creates no prior worker.
- Probe and Play expose `--fingertip-prior-metadata-sha256`, reject missing or
  malformed pairs before importing AppLauncher, and bind both strictly loaded
  frozen workers to that pin before launcher/scene startup. They pass the
  prepared worker binding to the Wrapper, which does not reread the mutable
  artifact path.
- Probe report metadata and the formal aggregate manifest both record the
  external pin. A coordinated artifact rewrite therefore cannot silently move
  the runtime or formal evidence to another metadata identity.

## TDD And Verification

RED was observed before production edits: the focused runtime/entrypoint/
manifest suite reported `23 failed, 21 passed`, with failures caused by the
missing runtime keyword, CLI pair, Wrapper propagation, and manifest pin.

GREEN evidence:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_runtime.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_dual_panda_o6_verification.py
```

Result: `43 passed in 16.15s`.

All expert-prior CPU tests: `299 passed in 50.00s`. Pure Arm/Hand/WBC QP
verification: `103 passed in 3.19s`.

The final combined expert-prior, entrypoint, manifest, and pure-QP command
reported `424 passed in 51.41s`. `py_compile` for all four production files and
both Probe/Play `--help` smokes exited zero. The task-scoped `git diff --check`
exited zero.

The broad `test_m1_bimanual_*.py test_m1_dual_panda_o6_*.py` run reached
`549 passed` and one unrelated pre-existing static failure:
`test_probe_has_required_startup_smoke_cli` searches the Probe source text for
the dynamically AppLauncher-provided `--headless` flag. Neither the failing
test nor this existing launcher boundary was changed here.

Repository-wide `git diff --check` is also blocked outside this task by the
existing `.superpowers/sdd/task-6-brief.md:61` blank line at EOF; the pin-scoped
diff remains clean.

## Boundary

No production student artifact exists yet, so no prior-on Isaac/GPU execution
or metadata-pin value can be claimed. The implementation is verified at the
strict artifact/runtime, fake Wrapper lifecycle, CLI subprocess, report, and
pure-QP layers only.

## Git Refs

- Initial Pin Baseline Ref: `4df4bce` (actual parent of the delivered pin commit)
- Initial Pin Candidate Ref: `4bfc11bc60adb927b6ee033b2cffd738e8508e49`
- Startup Binding Review Baseline Ref: `4bfc11b`
- Startup Binding Submission Parent: `bc28655` (concurrent headless CLI guard fix)
- Startup Binding Review Candidate: the commit with subject
  `fix: bind pinned priors before Isaac startup`; resolve its exact SHA with
  `git log -1 --format=%H --all --grep='^fix: bind pinned priors before Isaac startup$'`.
  The final verification command/results and exact commit SHA are additionally
  recorded in `.superpowers/sdd/runtime-pin-bridge-report.md` (working ledger,
  deliberately excluded from the feature commit).
- Key Files: frozen prior runtime, bimanual Wrapper, Probe, Play, and their
  runtime/entrypoint/verification tests

## Startup Binding Review Fix

The initial path prevalidation followed by scene creation and a later Wrapper
path load left a TOCTOU gap: a substituted artifact could be rejected only
after Isaac had started. The fix chooses pre-launch runtime binding rather
than a second post-scene path check.

`PreparedO6FingertipPriors.from_artifact()` constructs both SHA-verified frozen
workers before importing AppLauncher. Each strict load deserializes its own
immutable bytes snapshot, and both metadata identities must equal the external
pin, including weight/report SHAs. A replacement between the two loads is
rejected before any scene boundary and the first worker is reaped. Once the
pair is bound, replacing or deleting the artifact cannot change either worker.

The entrypoint owns this prepared pair and closes it in `finally` on normal
completion and every launcher/scene/Wrapper exception. Probe shares the binding
across trials; Wrapper close never closes borrowed workers, while direct
Wrapper artifact construction retains its existing owned-worker lifecycle.

New path-replacement/lifecycle tests first reported `9 failed` before the
binding module and entrypoint bridge existed. Tests cover replacement after
binding, replacement between left/right loads, Wrapper no-reread and two-trial
reuse, exact real AppLauncher import ordering, pin mismatch before startup,
and cleanup after successful and exceptional entrypoint exits.

Final focused verification (original three pin test files, new startup-binding
regressions, and the refactored launcher's direct AST guard): `61 passed in
32.98s`. The `bc28655` AST guard was adjusted only to follow the parser from
`main` into `_run_with_fingertip_prior`; the exact same-parser AppLauncher
registration, parsing, and namespace-construction ordering remain required.
Before that targeted guard adjustment its RED was `1 failed`, proving the
direct refactor dependency. No launcher flag ownership was changed.

## Follow-up

After the deterministic dataset conversion, expert ensemble, and two student
distillations pass their gates, record the production student metadata bytes'
SHA-256 outside the artifact and supply that exact value to prior-on Probe and
Play. Task 12 GPU0 smoke and formal 30-trial manifests must retain the same pin.
