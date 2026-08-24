# 2026-08-24 M1 + Panda coordinated checkpoint pruning

## Scope

按用户批准删除 long-v4 中 update 3500 之后的数字 checkpoint，同时保留 `model_3500.pt`、原 manifest 和非数字 best/final 文件。

## Safety Gates

- 清理器只接受解析后的精确目录 `coordinated_teacher_long_v4_64x5000_20260823`。
- 默认 dry-run；`--apply` 才删除。
- 拒绝 symlink、目录逃逸和执行前 SHA 改变。
- 删除前原子写 `planned` 审计；后置条件通过后才写 `completed`。
- 临时目录 RED→GREEN：`4 passed`。

## Applied Result

- 删除 15 个文件：`model_3600.pt` 至 `model_4900.pt`（每 100）以及 `model_4999.pt`。
- 审计：`checkpoint_pruning.json`，status `completed`，包含全部 15 个删除前 SHA-256。
- 原 manifest SHA-256：`8742b9eef65ce615c2221bf007f1fe27afe1f33190f0ea3e3680b9d7cda69995`，执行前后不变。
- 保留 `model_3500.pt` SHA-256：`f4911e4dfc4a2ca793dde5b3eac8ed8195f1dbb7e6f0a7d677a9b7691ea9321a`。
- 后置条件：最大数字 checkpoint 为 3500；大于 3500 的数字 checkpoint 为空。

## Recovery Boundary

这些 15 个已删除文件不在 Git 中，当前只能通过外部备份恢复。该清理不代表 long-v4 被接受；其约 update 4022 后坍塌的结论保持不变。
