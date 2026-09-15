# T500 run_e expert artifact qualification (BLOCKED)

## Purpose, stage and related todo

GPU RTI Task 1 / T500.5 offline corpus qualification; see [branch memory](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md).
Frozen run_e **identity passes**, but **production training qualification fails**. No teacher/student
training was started; no production artifact or runtime pin was fabricated.

## Current evidence versus historical BLOCKED state

The [older full-distillation log](2026-09-11-t500-dexmanipnet-full-distillation.md) describes an earlier
attempt with no shards. That remains historical evidence, not a description of current run_e.
Current external `Go2Pvcnn/data/external/dexmanipnet/converted/run_e/` contains a complete
241-shard conversion result with 984,641 samples, **all from FAVOR**. Complete conversion output
does not mean acceptance of both required production source corpora.

| identity | actual verified value |
| --- | --- |
| aggregate canonical-body SHA-256 | `dfaa213a89d8a87b267ffd7ed9dc69d5a3f8582204e79a11d575d30140a57c7c` |
| aggregate_manifest.json raw-file SHA-256 | `33c3fd6237cf3820e1f989f26d7d6b559a1f1d50891bf5423b8fa963311615bd` |
| audit.jsonl SHA-256 | `bf72fd4cedd45d540f524b70bb28ee53b5eb47669f06a6cef8eb9b3697e04154` |
| archive manifest SHA-256 | `054a2e000b1734ff94032a61d91f0bef3d96015b4189e2949e65d7f0d6c3cf8a` |
| source manifest SHA-256 | `dba99a64d66e09d380df3f74c7a80601cda0c78398cf88ca6bfa6b94d1deacfe` |
| shards / manifest-summed / actual NPZ samples | `241 / 984641 / 984641` |
| train / validation / test samples | `790247 / 97986 / 96408` |
| FAVOR / OakInkV2 accepted samples | `984641 / 0` |

The existing interface is `verify_aggregate_manifest(output_root)`, **not a manifest file path**.
It recomputed canonical aggregate, audit bytes/SHA and every declared shard SHA, and checked
missing/extra/unsafe shard paths. Real qualification additionally opened all 241 NPZ files with
`allow_pickle=False`, recomputed sample lengths and source counts, and compared each to its manifest
row. New helpers consume already-verified documents; they do not replace the storage verifier or
qualify a self-declared SHA. The generic student loader is not hard-bound to run_e.

## Accepted and rejected source evidence

| source | accepted | reason | sequence-side count |
| --- | --- | --- | ---: |
| favor | true | accepted | 1431 |
| favor | false | length_mismatch | 67 |
| favor | false | conversion_rejected:sequence conversion rejected: object geometry collision mesh is unusable | 496 |
| favor | false | conversion_rejected:sequence conversion rejected: object geometry collision mesh has unusable volume | 8 |
| favor | false | conversion_rejected:sequence conversion rejected: object geometry collision mesh orientation is unusable | 2 |
| oakinkv2 | false | missing_object_geometry | 1292 |
| oakinkv2 | false | interaction_mode_side_mismatch | 1224 |

No OakInkV2 source-compatible side survived geometry validation. The delegating agent explicitly
confirmed FAVOR-only is **not approved** and the existing both-source production gate remains.

## TDD and verification

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_expert_prior_artifact.py \
  tests/test_m1_bimanual_expert_prior_distill.py
```

RED: **13 failed, 71 passed in 16.03s**, missing frozen identity/source qualification helpers.
GREEN: **84 passed in 16.07s**. Tiny document/student fixtures avoid requiring external run_e in
ordinary CPU unit tests. Cases cover exact identity, nibble drift, missing facts, malformed/bool/
non-finite samples/source counts, accounting and both-source acceptance. Existing external metadata
pin and teacher provenance rejection already worked; neither was reimplemented. Teacher metadata
one-nibble drift now exercises the full strict loader and rejects before Torch deserialization.

All expert-prior CPU regression: `PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_*.py`
from `Go2Pvcnn`: **314 passed in 48.24s**. Task-scoped diff whitespace check: exit 0.

## Exact actual corpus qualification command

Run from `Go2Pvcnn`, read-only external corpus:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -c 'import json,sys; from collections import Counter; from pathlib import Path; import numpy as np; from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.storage import verify_aggregate_manifest; from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.artifact import validate_run_e_corpus_identity,validate_run_e_training_qualification,sha256_file; root=Path("data/external/dexmanipnet/converted/run_e"); d=verify_aggregate_manifest(root); validate_run_e_corpus_identity(d); actual=0; sources=Counter(); splits=Counter(); audit=Counter();
for r in d["shards"]:
 with np.load(root/r["path"],allow_pickle=False) as z:
  n=len(z["source_group"]); assert n==r["samples"]; c=Counter(str(g).split("/",1)[0] for g in z["source_group"]); assert dict(c)==r["source_counts"]; actual+=n; sources.update(c); splits[r["split"]]+=n
for line in (root/"audit.jsonl").read_text().splitlines():
 r=json.loads(line); audit[(r["source"],r["accepted"],r["reason"])]+=1
print(json.dumps({"identity":"PASS","aggregate_sha256":d["aggregate_sha256"],"manifest_file_sha256":sha256_file(root/"aggregate_manifest.json"),"audit_sha256":sha256_file(root/"audit.jsonl"),"shards":len(d["shards"]),"manifest_samples":sum(r["samples"] for r in d["shards"]),"actual_npz_samples":actual,"actual_source_samples":dict(sources),"split_samples":dict(splits),"archive_manifest_sha256":d["archive_manifest_sha256"],"source_manifest_sha256":d["source_manifest_sha256"],"audit":[{"source":s,"accepted":a,"reason":r,"count":n} for (s,a,r),n in sorted(audit.items())]},sort_keys=True),flush=True)
try: validate_run_e_training_qualification(d)
except ValueError as e: print("QUALIFICATION=REJECTED: "+str(e),flush=True); sys.exit(1)
print("QUALIFICATION=PASS")'
```

Result: **exit 1**, after the identity/sample/source values above, then:

```text
QUALIFICATION=REJECTED: run_e production training requires both favor and oakinkv2 accepted samples
```

Exact rejected corpus/audit/shards preserved in place. No source selection, accuracy, CPU latency,
geometry or production-provenance gate relaxed.

## Training, environment and artifact status

GPU0 inspected as `NVIDIA GeForce RTX 5070`, UUID `GPU-9d88c3d7-11db-66b9-47f4-f9a1bb744a8d`,
driver `580.159.03`; PyTorch `2.7.0+cu128`, CUDA build `12.8`, NumPy `1.26.0`.
Driver CUDA capability `13.0` from nvidia-smi is not the Torch build identity.
GPU processes were desktop graphics only; process check found no Play/training entrypoint.
Repeat the check before any future launch; this check does not waive a failed source gate.

The brief's existing expert/student commands were **not executed**. Planned teacher seeds remain
`42,43,44,45,46`, hidden `512,512,512`, 200 epochs; student seed `42`, hidden `64,64`,
samples-per-state `8`, 200 epochs. Approved seeds already exist in `ensemble_artifact.py`;
no duplicate roster or new training entrypoint introduced. Existing trainer/distiller unchanged.
Ensemble/student/metadata SHA, training duration/progress, NLL/RMSE/coverage, CPU latency and GPU
latency are **unavailable / not measured**. No training process/service/session exists to monitor.

## Git traceability and follow-up

- Baseline Ref: `4624abf` reviewed external metadata/startup binding.
- Candidate Ref: commit subject `feat: record frozen run-e qualification rejection`; exact SHA in task report after commit.
- Key Files: `expert_fingertip_prior/artifact.py`, `test_m1_bimanual_expert_prior_artifact.py`, this log and four notes entrypoints.
- Todo/log/index aligned to current conversion and blocked qualification; historical dirty full-distillation log untouched/unstaged.
- Next: authoritative pinned OakInkV2 geometry for rejected `ObjURDF/align_ds/...` references. Any changed corpus needs a newly approved pin; never silently reuse run_e's FAVOR-only SHA for an altered dataset. Requalify both sources before existing expert/student commands.
- Outcome: **Task 1 BLOCKED**, not complete; GPU RTI/prior-on/Isaac unverified.
