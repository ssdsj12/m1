# T500 GPU RTI contracts and eligible OakInk probe

## Purpose / stage / related todo

Continue the approved private GPU MPC plan without waiting for E0 training;
design §12 allows prior-off development. Related [T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).

## GPU Task 2 verification

Candidate `e280b8d`, baseline `d4f6119`. Private CUDA float32 contracts freeze
H25/action43/padded32/one RTI iteration and four line-search alphas. Both existing
entrypoints have independent backend/device flags and requested/actual diagnostics.
Default control remains CPU without CUDA initialization; explicit RTI/OSQP requests
reject before workers/AppLauncher because neither solver is implemented yet.
`auto` rejects without a benchmark manifest. Public CPU float64 contracts unchanged.

Implementer RED: missing module; additional static RED 3 failed/13 passed.
GREEN 46 passed, actual GPU0 allocations, no skips. Parent freshly reran from root:

```bash
PYTHONPATH="$PWD/Go2Pvcnn" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_bimanual_gpu_rti_contracts.py Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py
```

Exit0, **46 passed in12.83s**. Independent review: spec compliant, quality Approved;
Minor follow-up: assert emitted diagnostic fields rather than only source strings.
Task3 state mirror/KKT condensation/warm-start implementation dispatched; not yet
verified. GPU0 RTX5070, installed torch2.7.0+cu128/Triton3.3.0; no package changes.

## Real eligible OakInk CPU probe

Continuation of [geometry acquisition](2026-09-15-t500-oakink-geometry-acquisition.md).
Eligible staging `Go2Pvcnn/data/external/oakink_v2_geometry/eligible-probe-SyYndB`.
Object `O02@0018@00001/scan`, exact official raw tar bytes:
SHA `c7aba1523d64542b3330af992634ec99ae546aafb0c8f552b7be4c680b8414f4`.
Frozen recipe output SHA
`2474ed2ee3a51775ceb481ffa39714a635c2c5ee07a427e14f0954f4e3539eea`.
Generated mesh passed unchanged physical loader: finite/watertight/consistent winding,
four positive components, volume0.0011036067213426206m³; max bounds drift
0.0019569409366217982m. This object has not had a second independent generation.

Probe-only zero-origin/unit-scale URDF SHA
`df4b0b5de7d394a756d43693d7c86d5875272787de915e192a123315db96599d`;
external manifest SHA
`4977048f256a492cf7a96004568392f3f194fb31f44bf6769cf53094dfd2815e`.
These are not production pins.

Procedure: construct externally pinned resolver, audit and load actual sequence
`9a05b@3_bih` (142frames, bh_main) for inspire_lh/rh, verify source identity and
unchanged collision loading, then call existing `convert_loaded_sequence` using
pinned source URDF/FK on CPU. Both sides reject missing geometry without overlay,
accept with overlay, and each produce **216 in-memory windows**. Commands completed
exit0; no corpus shards written, original archives/source/run_e untouched.

## Conclusion / follow-up / Git refs

Task2 interface gate passed, not GPU solver/runtime/latency or Isaac acceptance.
Eligible sample proves real audit/load/FK/conversion integration for this object,
not full dual-source E0 qualification. No training or production student exists.
Next GPU work: batched dynamics and persistent warm-start, then eager RTI solver,
parity and performance gates. Geometry work still needs full generation, CLI and
aggregate pin wiring, new-directory dual-source conversion and qualification.

- Last Feature / Verified Commit: `e280b8d` (Task2 focused scope).
- Current Work Ref: `codex/t500-dual-panda-o6-mpc`.
- Key Files: `gpu_rti/config.py`, `gpu_rti/contracts.py`, existing Play/Probe entrypoints,
  and corresponding focused tests under `Go2Pvcnn`.
