# T500 DexManipNet Runtime Lifecycle Review

## Purpose

Close the Task 8 review findings without changing the frozen prior's O6/QP contract or starting
Task 9 integration.

## Stage And Related Todo

T500.5 / Task 8; see the [T500 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## Changes

- The persistent private model worker uses only the multiprocessing `spawn` context, deep-copies
  the model onto CPU, has a five-second ready handshake, and converts a model pickling/start
  failure into bounded `RuntimeError` construction failure.
- A worker timeout permanently poisons the adapter, attempts graceful sentinel shutdown, then
  terminate/join and kill/join as needed, and rejects any unreaped child. Queue endpoints close
  after reaping.
- `close()`, context-manager exit, and a `weakref.finalize` callback that retains the worker but
  never the adapter are idempotent. No `__del__` lifecycle path remains.
- `target()` takes the same lifecycle lock as `close()` before input packing and applies its hard
  deadline to lock acquisition, packing, worker request, and result processing. A waiting caller
  fails closed rather than racing a queue or exceeding the deadline.
- Public construction is only `from_artifact()`. Tests patch the strict loader inside the test
  module, not a production model-injection seam. Returned target tensors clone caller-owned data.

## Verification

Focused RED first demonstrated that an unpicklable model escaped `Process.start()` and that a
second query could wait behind the lifecycle lock past its deadline. The implementation then made
both cases green.

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_runtime.py
```

Result: `20 passed in 15.28s`. The tests cover spawn selection, start failure, a never-returning
worker deadline/poison/bypass/reaping sequence, context and GC finalization, concurrent
close/query serialization, a lock-wait deadline, forged construction rejection, generic
`BaseException`, and tensor ownership.

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_contracts.py \
  tests/test_m1_bimanual_expert_prior_download.py \
  tests/test_m1_bimanual_expert_prior_urdf_fk.py \
  tests/test_m1_bimanual_expert_prior_dexmanipnet.py \
  tests/test_m1_bimanual_expert_prior_preprocess.py \
  tests/test_m1_bimanual_expert_prior_storage.py \
  tests/test_m1_bimanual_expert_prior_model.py \
  tests/test_m1_bimanual_expert_prior_train_static.py \
  tests/test_m1_bimanual_expert_prior_distill.py \
  tests/test_m1_bimanual_expert_prior_artifact.py \
  tests/test_m1_bimanual_expert_prior_runtime.py
```

Result: `154 passed in 29.26s`.

`py_compile` passed for the adapter and its test; the scoped diff has no whitespace errors. A
fresh-process import-graph test and direct source scan reject offline, Hugging Face, HDF5, Isaac,
DexManipNet-extraction, and URDF-FK runtime imports.

## Git Refs

- Baseline Ref: `516cb4b6b745ded6157b2174f6f8b84ca0b24cc2`
- Candidate Ref: pending Task 8 lifecycle-review commit
- Key Files: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py`,
  `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_runtime.py`

## Follow-up

Task 9 may consume only the finite returned target and must retain its same-cycle baseline when
the prior or second QP is disabled. No production artifact or physical acceptance evidence was
created by this review.

## Deadline Re-review

The earlier lifecycle path performed its graceful and forced joins on the caller thread, allowing a
20 ms query deadline to spend roughly another `300 ms` in cleanup. A new RED test substitutes only
the parent response transport of a real, strict, production-approved temporary artifact; it proves
that the timeout diagnostic preserves at least `20 ms`, returns in under `60 ms`, bypasses the next
query immediately, and the detached child later signals reaping.

Timeout now atomically closes/poisons the adapter, calls `terminate()` immediately, and transfers
the detached worker to one module-level daemon reaper with a bounded queue. That one reaper owns
the bounded join/terminate/kill cleanup; target never waits for it. Queue/termination/finalizer
errors are contained in `timeout_cleanup_error` diagnostics with the original elapsed time. The
test artifact is written through `save_student_artifact()` and loaded through public
`from_artifact()`; its deterministic weights expose the exact float32 `15 + 15 + 5 + 7` packed
features in the worker response. No model-injection construction seam remains, including the former
module-level factory.

Focused runtime verification: `17 passed in 10.23s`. Tasks 1–8 regression: `151 passed in
22.53s`. The lower count replaces four direct-model seam tests with strict-artifact worker tests;
the review-specific wall-clock, eventual-reap, cleanup-containment, factory-forgery, and packing
coverage are now explicit.

## Bounded Ownership Re-review

The daemon reaper now owns a process-wide `BoundedSemaphore` slot budget and a queue with the same
capacity. A worker must acquire one slot before `spawn`; it releases that exact slot only after the
child is demonstrably dead, its queue channels have closed, and its `reaped` event is set. Thus a
timeout handoff cannot encounter a full queue: all active, queued, and currently-retrying workers
already consume the finite slot budget.

Reap failures retain ownership. The reaper repeats bounded join/terminate/kill passes after a
short event-based backoff, without setting `reaped`, closing channels, or releasing the slot until
`is_alive()` is false. Normal close and startup failure follow the same completion path. Every
post-dispatch target fault—including transport failure, mismatched/malformed worker response,
worker exception, timeout, and invalid output—now poisons/detaches asynchronously; target never
runs the multi-join close path.

Focused runtime verification: `20 passed in 13.07s`. Tasks 1–8 regression: `154 passed in
26.10s`. New evidence includes a capacity-one saturation/release test, a simulated alive-first
reap retry test, and a transport failure whose synchronous `close()` is deliberately blocked while
the query still returns. Pycompile, import boundary, direct forbidden-import scan, and scoped diff
checks passed.
