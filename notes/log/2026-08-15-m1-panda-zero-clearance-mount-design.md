# 2026-08-15 M1 + Panda 零间隙安装设计

## Decision

用户批准把 Panda 与 M1 背部的构建 clearance 从 `0.01 m` 改为 `0.0 m`，使安装原点直接落在 `BASE_LINK` 顶面。

选定从 `build_m1_panda_asset.py` 的源参数重建资产，不直接手工平移生成 USD，也不在本轮增加转接板。

## Safety Boundary

- 单 articulation、fixed mount、25 DOF 和 16 维 M1 控制保持不变。
- 零 clearance 不允许明显网格穿透；若 link0 网格低于安装原点，回到转接板方案。
- 高度变化会改变重心和力臂；旧 checkpoint 只做兼容性 Play，不自动视为行为通过。
- 实机最大载荷仍受 T400.3 机械验算门约束。

## Artifact

- [书面规格](../../docs/superpowers/specs/2026-08-15-m1-panda-zero-clearance-mount-design.md)

## Self-review

- Completeness：无未决实现项。
- Consistency：零 clearance、固定关节和碰撞 mask 定义一致。
- Scope：只覆盖安装高度、资产重建和必要验证。
- Ambiguity：明确“零间隙”是安装原点对齐，不是允许网格穿透。

## State

书面规格等待用户复核；尚未修改构建脚本、USD 或 checksum。

## Git Refs

- Branch: `main`
- Current Work Ref: working tree
