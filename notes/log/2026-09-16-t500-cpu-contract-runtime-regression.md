# T500 CPU contract/runtime regression during GPU Task3

## Purpose / stage / related todo

Verify default CPU contract, runtime and reduced-dynamics behavior remains intact
after private GPU Task2. Related [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Command and input

Worktree root, existing go2 environment, candidate `e280b8d`, no Isaac/CUDA workload.
Initial invocation used nonexistent `test_m1_bimanual_contracts.py`: exit4, no tests
ran. Resolved actual filename with `rg --files` and reran:

```bash
PYTHONPATH="$PWD/Go2Pvcnn" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_dual_panda_o6_contracts.py Go2Pvcnn/tests/test_m1_bimanual_runtime.py Go2Pvcnn/tests/test_m1_bimanual_reduced_dynamics.py
```

## Result / conclusion / follow-up

Exit0: **29 passed in0.89s**, no skipped tests. Evidence covers these CPU tests
only, not full regression, GPU Task3, physics, latency or production prior.
Continue Task3 CUDA parity and independent review.

- Last Verified Commit: `e280b8d` (focused scope).
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.
