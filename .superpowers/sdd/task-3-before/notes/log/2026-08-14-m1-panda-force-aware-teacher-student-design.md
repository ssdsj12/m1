# M1 + Panda 六轴力感知 Teacher–Student 设计记录

## Purpose

记录经用户逐节确认的 M1 + Panda 平衡与静止抓取设计，并验证书面规格不存在明显占位符、范围冲突或输入输出歧义。

## Stage

设计 / Teacher–Student 特权学习 / 六轴力传感器 / M1 残差平衡 / Panda 静止抓取。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## Procedure

1. 核对用户已确认的资产、传感器、训练阶段、观测、控制、安全和验收决定。
2. 写入正式设计文档。
3. 搜索 `TBD`、`TODO`、`FIXME`、`待定` 和 `待确认` 等占位内容。
4. 交叉检查物体质量输入边界、六轴信号语义、动作维度、训练阶段和验收指标。
5. 检查目标目录的 Git 状态。

## Input Conditions

- Panda 使用本地离线 USD 依赖闭包。
- 实机安装点使用物理六轴力/力矩传感器。
- Teacher 使用理想总六维反力，Student 使用带噪测量和可部署状态历史。
- M1 基础策略与 Panda IK/OSC 首期冻结。
- 首个操作任务为 M1 驻停、0–3 kg 抓取和短距离搬运。

## Key Metrics

- 设计文档：252 行。
- 占位符扫描：无匹配。
- 明确动作接口：12 个腿关节位置残差 + 4 个轮关节速度残差。
- 明确验收：20 秒随机六维扰动无跌倒率不低于 95%；姿态 P95 小于 10 度；0–3 kg 抓取搬运成功率不低于 80%；抓取阶段无跌倒率不低于 95%。
- Git：`/home/xk/coding/M1` 不是 Git 工作树，无法生成设计提交。

## Result

通过文档级自审。设计覆盖架构、组件边界、数据流、训练、异常处理、安全和验证；未修改运行时代码，未执行仿真或实机测试。

## Conclusion

书面规格可进入用户审阅门。用户批准后才能转入 writing-plans；实施前仍需传感器选型参数和机械最坏工况验算。

## Follow-up

等待用户审阅 [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)。

## Git Refs

- Baseline Ref: unavailable（目录不是 Git 工作树）
- Candidate Ref: filesystem working copy
- Key Files:
  - [设计文档](../../docs/superpowers/specs/2026-08-14-m1-panda-force-aware-teacher-student-design.md)
  - [T400 branch memory](../todo/T400-m1-panda-force-aware-teacher-student.md)
