# Baseline Headless Fix Report

## Status / Scope

Complete: only the stale Probe CLI static test and its dedicated notes changed.
Runtime metadata-pin production files and existing pin evidence are untouched.
Existing dirty SDD/graphify/status/full-distillation files were preserved.

## Root Cause Evidence

Baseline HEAD: `4bfc11bc60adb927b6ee033b2cffd738e8508e49`.
Original reproduction from `Go2Pvcnn`:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_env_static.py::test_probe_has_required_startup_smoke_cli
```

Observed `1 failed in 0.03s`, specifically `assert '--headless' in source`.
The Probe `_parser()` owns local flags. `main()` imports IsaacLab AppLauncher,
calls `AppLauncher.add_app_launcher_args(parser)` before `parser.parse_args()`,
and constructs `AppLauncher(args)` from that final namespace. The installed
`isaaclab/app/app_launcher.py` defines `add_app_launcher_args` at line 182 and
registers `--headless` with `action="store_true"` at line 276. Searching for its
literal in the caller was therefore testing the wrong ownership boundary.

## Fix / Sensitivity

Keep local CLI and startup diagnostic assertions. Replace the stale headless
literal assertion with exact AST checks in `main.body` for parser construction,
the real provider import, registration on the same parser, strict parsing, and
launcher construction from that namespace, in increasing startup order. A
comment/string/unrelated nested launcher call does not satisfy these checks.

In-memory source mutation experiments invoke the revised test against:

- deleted registration: rejected;
- registration on `other_parser`: rejected;
- launcher construction from `early_args`: rejected.

Direct installed-provider behavior check:

```bash
/home/xk/miniconda3/envs/go2/bin/python -c 'import argparse; from isaaclab.app import AppLauncher; p=argparse.ArgumentParser(); p.add_argument("--num-envs", type=int, default=1); AppLauncher.add_app_launcher_args(p); assert p.parse_args(["--headless"]).headless is True; assert p.parse_args([]).headless is False; print("real AppLauncher headless registration: PASS (no simulation constructed)")'
```

Exit `0`, printed PASS; no AppLauncher/SimulationApp instance was constructed.
Probe `--headless --help` exits at early `parse_known_args` before registering
launcher flags, so its successful local help output is explicitly not treated
as proof of actual launcher registration or headless simulation execution.

## Regression Evidence

Focused command (same environment/PYTHONPATH):
`python -m pytest -q tests/test_m1_dual_panda_o6_env_static.py`
reported `6 passed in 0.03s`, including a fresh post-format run.

Required full baseline command:

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_*.py tests/test_m1_dual_panda_o6_*.py
```

Exit `0`: `550 passed in 53.10s` (same item count as the reported baseline).
Task-scoped `git diff --check` exits `0`; staged scope also checked before commit.

## Notes / Commit / Limits

Aligned dashboard, closed T500 child, log index, and dedicated verification log:
[headless log](../../notes/log/2026-09-15-t500-probe-headless-baseline.md).
The containing commit is the dedicated headless-fix commit; its hash is returned
to the delegating agent. No unrelated files are staged. No runtime/control/CLI
contract changed; no physical gate, production artifact, or prior-on Isaac/GPU
acceptance is claimed.
