# T500 DexManipNet Prior Probe/Play Entrypoints

## Purpose

Record Task 10 opt-in artifact configuration, worker lifecycle, and JSON/JSONL diagnostics
wiring for the frozen O6 fingertip prior.

## Stage And Related Todo

T500.5 / Task 10; see [T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Contract

- Both probe and play accept `--fingertip-prior-artifact` with default `None`; no default import,
  artifact read, worker, or behavior change occurs when it is absent.
- Wrapper construction loads the one approved artifact separately for left and right worker-owned
  adapters before reading `env.unwrapped`, reset, or physics. If construction/startup fails, every
  created adapter is closed. Wrapper close/context/finalization are idempotent owners.
- Probe trial JSON and formal JSONL expose independent `left_o6`/`right_o6` summaries: enable
  state/reason, selected component/probability, precision range, cost, enabled-only inference
  p99, tip-rate delta, and prior QP rejection/fallback counts. Disabled metric values are null.
- Help parsing remains Isaac/artifact-free; actual play retains its existing loop and explicit
  terminal exit.

## Verification

RED:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_entrypoints_static.py \
  tests/test_m1_dual_panda_o6_verification.py
```

Result: `5 failed, 12 passed` as expected before the new option, summaries, and lifecycle code.

Final focused check: `17 passed in 0.91s`.

Tasks 1–10 relevant prior/Hand/runtime command: `198 passed in 25.84s`.
Remaining non-prior bimanual command: `142 passed in 5.51s`.
Both entrypoint `--help` smokes, `py_compile`, scoped `git diff --check`, and a fresh default
`hand_mpc` import graph with no `expert_fingertip_prior` modules passed.

## Boundary

No approved production artifact, Isaac runtime, physical trial, or GPU latency measurement was
executed. The diagnostics schema is verified through pure/static and fake-environment lifecycle
tests only.
