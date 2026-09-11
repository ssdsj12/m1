# T500 GitHub upload verification

- Parent: [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
- Stage: T500 control, contact sensing, lift motion primitives, and fixed-budget training orchestration.
- Baseline Ref: `d761df0`.
- Candidate Ref: `265fbcf`.
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.

## Included changes

- SO(3) interpolation, spatial orientation error, angular feedforward, and dual-arm MPC routing.
- Canonical raw/Box-filtered contact summaries with one Isaac Lab contact sensor per finger body.
- Lift-related motion primitives, vector-control wrapper path, and safer phase progression.
- Recoverable 10k sweep and 40k exact env-step ledger, plus teacher preparation/merge scripts.
- Focused tests and the approved 10k multi-GPU design/implementation documents.

Generated training artifacts, caches, the user-local Vulkan environment, and already-versioned
Git LFS asset payloads were not added by this upload.

## Verification

From `Go2Pvcnn` in the source workspace:

```bash
/home/hexinkun/miniconda3/envs/m1/bin/python -m pytest -q \
  tests/test_m1_bimanual_*.py tests/test_m1_dual_panda_o6_*.py
```

Result: `224 passed in 17.40s`.

Additional checks:

- focused environment/contact regression: `12 passed in 1.64s`
- changed Python modules: `compileall` exit `0`
- staged `git diff --check`: exit `0`
- staged secret scan: no matches
- largest newly staged file: below 17 KiB; no new binary/LFS payload was staged

The repository-wide pytest command was also attempted with `tests/artifacts` excluded and the
repository-local `rsl_rl` on `PYTHONPATH`. A repository test terminated collection/execution at
about 86% with process status `0` but without a pytest summary, so this run is not counted as a
full-suite pass.

## Conclusion

The uploaded T500 code surface passes its focused static and pure-control regression suite.
Real Isaac Sim cross-seed gates and the formal 30/30 lift/hold/lower/release acceptance remain
unverified and are still required before claiming physical task completion.
