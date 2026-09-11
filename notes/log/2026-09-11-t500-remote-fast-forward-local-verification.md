# T500 远端快进合入与本地验证

## 目的

对比 GitHub `codex/t500-dual-panda-o6-mpc` 与本地同名 worktree，将远端可在
当前 M1 环境使用的提交合入，并在本机重新验证。

## Git 对比与合入

- 本地基线：`d761df0`
- 远端叶子：`037595b`
- 分叉关系：本地无独立提交，远端领先 3 个提交
- 操作：`git merge --ff-only origin/codex/t500-dual-panda-o6-mpc`
- 合入提交：`265fbcf`、`7c6701d`、`037595b`

主要新增 SO(3) 姿态轨迹/误差修正、接触摘要、举升动作原语、向量控制入口、
10k/40k 固定步数运行器以及 teacher 数据准备链。运行代码未发现旧主机绝对路径；
README 命令已改成激活 `go2` 环境后使用仓库相对路径。

## 本地验证

环境：`/home/xk/miniconda3/envs/go2/bin/python`，工作目录 `Go2Pvcnn`。

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_*.py tests/test_m1_dual_panda_o6_*.py
```

结果：`224 passed`。

相关控制、环境和运行脚本 `compileall` 退出码为 `0`；10k、40k、teacher merge、
teacher prepare 四个命令行入口的 `--help` 均退出 `0`。

## 边界

本次只完成 Git 对比、快进合入、CPU/纯控制回归和入口检查。未运行真实 Isaac GPU0
smoke、跨 seed 物理门或正式 30/30 验收，因此不宣称完整举升任务已经通过物理验收。
Graphify 索引对当前 T500 改动覆盖不足，代码差异以 Git 对象和实际测试为准。
