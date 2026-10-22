# 2026-10-23 文件整理说明

本次整理将 SU 与 AD 分别归类，并按用户修改后的 README 简化中文说明。用户 README 保持原样，AI 写作要求和操作限制集中放在 `AGENTS.md`。

## 目录怎样对应

| 整理前 | 整理后 | 说明 |
|---|---|---|
| `scripts/tc_peps.jl`、`scripts/ed_checks.jl` | `model/` | 共同模型、解析态、原 SU circuit 和 ED |
| `finite-field-loop-update/reference.jl` | `model/reference.jl` | SU/AD 共用文献级数；不复制两套参考公式 |
| 旧 `inspection/` 中 SU 重放和门检查 | `simple-update/merged/` | 七月 merged L 形 SU |
| 原 `simple-update/` | `simple-update/split/` | 十月零场 path/loop 对照，两者共享 state 和 star 更新 |
| `finite-field-loop-update/` | `simple-update/loop/` | 十月一般 D 和有限场 loop 实现及边界测量 |
| `m2-su-audit/`、`m2-su-dynamics/` | `research/su-stagnation/{single-step,dynamics}/` | M2 停滞原因专题，保留逐门检查、闭式研究及全部原始数据 |
| `scripts/ad_tied_core.jl`、`ad_tied_gd.jl` | `ad/merged/` | 七月已成功的共享 merged 张量零场 AD |
| `automatic-differentiate/` | `ad/split/` | 十月 V/P/λ 的 fixed-point / unrolled AD |
| 七月 ED 与成功零场 AD 相关结果 | `model/results/`、`ad/merged/results/` | 原文件逐字节移动，没有改写历史日志 |
| `PROJECT_REVIEW_ZH.md` | 已删除 | 整理前的讨论稿，已由当前 README、PLAN 和阶段报告取代 |

各实验的 `results/`、`figures/`、`frozen_states/` 保存在对应目录。根 `RESULTS.csv` 汇总 18 条主要结果，注明算法、张量表示、初态、计算状态和来源。

`simple-update/split/final_measurements.toml`、`ad/merged/final_measurements.csv`、`model/ed_2x2.csv` 是从本地原始文件原样取出的少量测量数据，随代码保存，方便没有历史二进制文件的新 checkout 查看结果。它们与原始数据不是两次不同计算。

## 移除了什么

删除过时的限时 `CHALLENGE_SUMMARY.md`、重复的 `PROGRESS.md`、`PROJECT_AUDIT.md`、`M2_SU_FINDINGS.md`，根 M2/M3 报告按最新结果重新撰写。各算法重复的子 PLAN/DESIGN 也不再承担整体规划，统一由根 `PLAN.md` 说明。

已删除七月未完成的 `m3_hz_continuation.jl`、`m3_series_validation.jl`、对应测试及旧曲线 `m3_mz_vs_hz_invalid.svg`。PLAN、M3 和 AD 说明中的相关历史段落一并清除。当前 M3 报告保留十月弱场结果。

其他清理包括旧独立张量/对称化 AD 试验、混合 SU/AD 驱动 `groundstate_h0.jl` 及一次性检查程序。十月 AD 的未完成恢复脚本已移除，相关数据保留用于收缩问题分析。

M2 停滞研究从 `simple-update/failure-analysis/` 移至 `research/su-stagnation/`，其 38 项基础测试改为单独运行，默认测试只保留基态主线。

## 备份与历史文件

整理前完整备份位于仓库根：

```text
.superpowers/backups/anyon-reorganization-20261023-023535/
  before.tar.gz   整理前整个子项目、旧结果和 Julia Project/Manifest
  files.json      1480 个文件的 SHA-256
  moves.json      目录和文件移动清单
  removed.json    从工作目录移出的旧文件清单
```

上述备份保存第一次整理前的 1480 个文件。本轮另在 `.superpowers/backups/anyon-docs-followup-20261023-032205/` 保存了 1194 个当前文件，包括用户修改后的 README，归档后也逐文件校验过。两份备份均在本地。

原始运行记录、张量、数据和图保持原内容；当前程序引用、文档链接和运行命令随目录调整。

## 第一次整理的检查

- 整理前，保留算法的 272 项 Julia 检查全部通过。
- 整理后，377 项 Julia 检查、5 项 Python 汇总测试全部通过，63 个 Julia 源文件语法检查通过。
- 1,093 个保留的原始数据、运行时程序副本及图，与整理前逐文件 SHA-256 相同。
- 独立审阅未发现影响运行或结果解释的问题，两处遗留文档说明已修正。

仓库根 `make test` 因缺少 `pytest` 未能启动，本轮再次尝试仍是同一情况，环境未改动。第一轮日志在第一份备份的 `verification/`。

## 本轮检查

- 用户修改的 README 逐字节未变。
- 主线 Julia 测试 339/339、专题测试 38/38、Python 汇总测试 5/5 通过。
- 18 行主要结果表重新生成后完全一致；1,097 个原始数值文件、张量、运行时程序副本和图与本轮备份相同。
- 64 个 Julia 源文件及当前 Python 源文件语法检查通过；98 个本地链接和图片均可找到，空白检查通过。
- 独立审阅未发现问题。全部日志和文件校验结果保存在本轮备份的 `verification/`。
