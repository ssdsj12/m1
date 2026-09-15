# T500 Probe Headless Baseline Test

## Purpose / Stage / Related Todo

Repair the false-positive baseline static CLI test in T500's Probe startup
boundary; closed child of [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
Production CLI, simulation, controller, and metadata-pin contracts are unchanged.

## Procedure / Input Conditions

Use the local `go2` Python environment and the existing worktree. Reproduce
`tests/test_m1_dual_panda_o6_env_static.py::test_probe_has_required_startup_smoke_cli`.
Trace Probe `_parser()` / `main()` to the installed IsaacLab
`AppLauncher.add_app_launcher_args()`, whose `--headless` action is `store_true`.
Replace only the stale literal assertion with exact top-level AST startup
wiring: parser construction, real provider import, registration on that parser,
strict parsing, and launcher construction from that namespace, in order.

## Metrics / Result / Conclusion

RED: the original focused test fails on the absent `--headless` Probe literal.
Focused file: `6 passed in 0.03s`. Actual installed AppLauncher registration
accepts `--headless` as `True` and defaults to `False`, without constructing a
SimulationApp. In-memory mutations removing registration, changing the parser,
or changing the launcher namespace are all rejected by the revised test.
Full baseline suite: `550 passed in 53.10s`; scoped `git diff --check` exits `0`.
Full-suite and scoped-diff evidence is retained in the
[delegated report](../../.superpowers/sdd/baseline-headless-fix-report.md).

## Follow-up / Boundary

No physical/Isaac run, artifact, training, or prior-on acceptance is claimed.
The existing Probe `--help` exits during early parsing before launcher flags
are registered; it is not evidence of launcher-provided help and was not changed.

## Git Refs

- Baseline Ref: `4bfc11bc60adb927b6ee033b2cffd738e8508e49`
- Candidate Ref: commit containing this log; see delegated report and `git log`
- Key Files: [static test](../../Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py)
