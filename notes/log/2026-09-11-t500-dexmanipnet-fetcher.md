# T500 DexManipNet 固定下载与安全解压

## Purpose

为 T500.5 指尖先验建立唯一的外部数据边界：两个 DexManipNet archive 固定到同一 Hugging Face revision，ManipTrans 固定到 commit，并在任何 extraction 安装前完成安全验证。

## Stage

T500.5 / Task 2；只涉及离线 fetch、archive manifest 和 source checkout，不改变 Hand MPC 或任何运行时依赖。

## Related Todo

[T500.5 branch page](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。

## RED

1. `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_download.py` 在 production module 缺失时按预期 collection error：`ModuleNotFoundError: ... expert_fingertip_prior.download`。
2. 在初版实现后新增 strict verify-only filesystem test 并运行 `... pytest -q tests/test_m1_bimanual_expert_prior_download.py -k verify_only`；按预期失败，因为 verify-only 重写 manifest（mtime 改变）。

## GREEN And Regression

Executed from `Go2Pvcnn`:

- `PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_download.py tests/test_m1_bimanual_expert_prior_contracts.py`
- `PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_fetch_dexmanipnet.py --help`
- `PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m py_compile go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/download.py scripts/m1_dual_panda_o6_fetch_dexmanipnet.py`
- `git diff --check`

Initial result: `33 passed in 1.78s`; help exits `0` and prints all three flags; pycompile and diff check exit `0`. Review hardening below supersedes the final count.

## Input Conditions And Metrics

- Real filesystem fixtures cover absolute and parent traversal names, char/block/FIFO entries, symbolic/hard links escaping their root, required `sequences/` validation, and staged atomic install.
- Fixture archive digest/size records are checked without downloading the 8.31 GB data set.
- `--verify-only` fixture creates a pinned local Git checkout plus existing manifest/extractions and proves exact content, mtime, and root path set remain unchanged.

## Result

`download.py` uses only the Python standard library. The CLI imports `huggingface_hub` only inside the real fetch function, uses the fixed dataset revision and both archive names, checks free space, records SHA-256/size/source URL/revision/source commit/tree, and installs validated archives via sibling temporary directories and `os.replace`. Existing destinations are not overwritten.

`--download-only` fetches archives/source and writes evidence without extraction. `--verify-only` does no network, clone, extraction, rename, or manifest write; it compares the stored manifest to recomputed archive and source facts and requires both installed `sequences/` roots.

## Review Hardening: Clean Source Checkout

Review found that matching `HEAD` and tree alone does not prove the working tree consumed by later URDF parsing is unchanged. Two focused RED cases—modified tracked `README` and untracked `untracked.txt`—both let the old existing-checkout and verify-only paths pass. A third RED showed verify-only printed the misleading `wrote pinned download manifest` message.

The fetcher now runs `git --no-optional-locks status --porcelain=v1 --untracked-files=all --ignored=matching --ignore-submodules=none` before reusing an existing checkout and before verify-only commit/tree checks. Any status output raises `ValueError`; the command avoids optional index locks. The fresh staged clone receives the same gate before installation. Verify-only now prints `verified pinned downloads and source`.

Focused RED command: `PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_download.py -k 'dirty_maniptrans or reports_verification'` produced `3 failed`. GREEN produced `3 passed, 13 deselected in 0.83s`.

Final Task 1–2 verification repeated the full focused suite, help smoke, pycompile, and diff check: `36 passed in 1.81s`; all non-pytest commands exited `0`. The dirty-tree fixture confirms manifest bytes/mtime, source index mtime, and root path set are unchanged for both tracked and untracked rejection paths.

## Conclusion And Follow-up

Task 2 local contracts pass. No production archive download, pinned remote clone, or real archive schema inspection was run; executing a real fetch remains a user-controlled external-data action. Task 3 can consume the verified pinned source checkout once data is present.

## Git Refs

- Baseline Ref: `1ee5497` (`docs: mark fingertip prior contracts complete`)
- Candidate Refs: `5442a36` (`feat: add pinned DexManipNet fetcher`) and `c78d442` (`fix: verify clean ManipTrans checkout`)
- Key Files: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/download.py`, `Go2Pvcnn/scripts/m1_dual_panda_o6_fetch_dexmanipnet.py`, and `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_download.py`
