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
  malformed pairs before importing AppLauncher, prevalidate the artifact with
  that pin, and pass the same value to the Wrapper.
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

- Baseline Ref: `8534f6c8b026b4828df434d4d44f43964445dd80`
- Candidate Ref: uncommitted worktree change
- Key Files: frozen prior runtime, bimanual Wrapper, Probe, Play, and their
  runtime/entrypoint/verification tests

## Follow-up

After the deterministic dataset conversion, expert ensemble, and two student
distillations pass their gates, record the production student metadata bytes'
SHA-256 outside the artifact and supply that exact value to prior-on Probe and
Play. Task 12 GPU0 smoke and formal 30-trial manifests must retain the same pin.
