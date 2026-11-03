# RELEASE_CLOSURE_REPORT.md — YOLO26 ROCm Day 1 · 发布收口报告

- 时间（UTC）：2026-11-03 03:20–04:10
- 审阅基线：PR #3 HEAD `372557d52ca3c2a30cd3eba5729013c7115030da`（经 GitHub API 复核，远端未更新，与基线一致；PR 状态 OPEN/MERGEABLE/无评审决定）
- 本轮产出：分支 `release-closure`（基于 `372557d`）+ 本报告等四份交付 + 最小修复提交
- 方法：先只读审计（Phase 0），再最小修改（Phase 1–5），最后真机全量复验（Phase 6，见 [VALIDATION_MATRIX.md](VALIDATION_MATRIX.md)）

---

## 1. PR #3 只读审计结论

| 审计项 | 结果 |
|---|---|
| 提交/文件 | 4 提交、77 文件（64 validation + 6 notebooks + 4 tests + 1 templates + 1 scripts + 1 generator），与声明一致 |
| 41 Code Cells vs AST | ✅ 逐格一致（pytest AST+生成器一致性双测通过；且在**无 torch 环境**同样通过——CI 可行性实证） |
| 证据完整性 | ✅ SHA256SUMS 18/18 复核；历史轮次（含 2 次有意 FAIL 实证）零删改 |
| GitHub CI | ❌ **仓库无任何 workflow**——39 项离线门禁从不自动运行（本轮新增 CI 修复） |
| Review 状态 | 无评审、无评论（待 Owner） |
| 体积负担 | 执行副本 16 个 >1MB 文件共 **41.2 MB**（单文件最大 ~4.1MB）——当前可接受；建议后续轮次把执行副本放 Release 资产或按"保留最新 2 轮 + 全部日志 JSON"策略归档（本轮不删任何东西） |
| 门禁有效性 | ✅ 累计 4 起"真实 FAIL"实证（场景覆盖、TODO 门禁×2、无填充复验） |

### P0/P1/P2 缺陷矩阵（已修复 / 仍存在 / 新发现）

| ID | 级别 | 状态 | 说明 |
|---|---|---|---|
| AST 参考损坏 / Runner 硬编码 / Kernel 回退 / --only 退出码 | P0 | **PR #3 已修复** | 无需重复工作；本轮复核通过 |
| 双语元数据交换 / 学生环节缺失 | P1 | **PR #3 已修复** | 同上 |
| N1 无 CI | P0 级（发布阻断） | **本轮修复** | `.github/workflows/ci.yml`（ubuntu+windows、无 torch）+ 守卫测试 |
| N2 B/D/E 无错误表、无等待说明、Lab E 单指标、七问不显式 | P1 | **本轮修复** | 学习地图×6 + 等待说明×2 + 错误表×3 + 四指标对比（全 md-only，41 cells 不变） |
| N3 `group{1..5}` shell 错误 | P1 | **本轮修复** | 03 报告 §2.1 改为每组各设 + 原因注明 |
| N4 README "Linux 未复验"过期声明 | P1 | **本轮修复** | 更新为 W7900D 实测结论 + 证据链接 + CI 徽章 |
| N5 执行副本体积 | P2 | 记录策略 | 不删改；建议后续归档策略 |
| N6 无课前自检工具 | P1（课堂） | **本轮修复** | `scripts/preflight_classroom.py`（11/11 真机 PASS） |
| 真实学生导出 E2E | — | **NOT_RUN** | 本机无真实样本，如实标注（见 §4） |

## 2. 本轮额外解决的实际问题（PR #3 之外）

1. **CI 缺失**（N1）：新增双平台、无 torch 的 GitHub Actions（守卫测试锁定"必跑 pytest、双平台、永不装 torch"）；本地无 torch 环境等价验证 41/41 通过。
2. **学生七问不显式**（N2）：六份"学习地图"——30 秒看懂为什么学/先懂什么/改什么参数/成功样子/失败自救/存什么证据/面试价值。
3. **训练等待焦虑**：03/05 新增"等待时看这里"（JIT 预热 ~1 分钟是正常现象 + `rocm-smi` 验证法 + A/B 独占要求）。
4. **Lab E 单指标误导**：四指标一起读（各自回答的问题 + 单独引用的陷阱 + 学生空表），方法论不预置数值。
5. **B/D/E 排障缺口**：三张常见错误表（症状→下一步），与 00/01/03 对齐。
6. **导入报错不友好**：类别越界错误现附双语提示行（有效范围 + 标签格式），带回归测试。
7. **课堂可操作性**：Preflight 11 项只读自检（真机 11/11；首轮自测抓出自身 2 缺陷并修复）；`group{1..5}` 修正；错峰表/降级路径/3 人试走验收表成文。
8. **README/讲师指南**：Linux 复验状态、CI 徽章、学生交互 vs 自动复验说明、清单入口。

## 3. 最终复验结果（详见 VALIDATION_MATRIX.md）

- **6/6 Notebook、41/41 cells、退出码 0**（第 3 次独立全链复现；全新目录）
- Lab C 12-Epoch ✅ · Lab E 双臂 12-Epoch ✅（mAP50-95 .3829 vs .3385，Δ=−0.0444，与前两轮**确定性一致**）
- Fixture E2E ✅（场景不跨集合；导入集真训练）
- gpu_smoke 17/17 ✅ · pytest 41+1skip（双环境）✅ · pip check ✅ · 跨 Lab 状态链 6/6 ✅
- 坏数据注入 5/5 拒绝 ✅ · 科学准确性 8/8 有据 ✅ · 证据哈希 18/18 ✅

## 4. 仍无真实证据的事项（如实清单）

| 事项 | 状态 | 补证路径 |
|---|---|---|
| 真实学生 X-AnyLabeling 导出 E2E | **NOT_RUN**（本机无样本） | 课堂/试走收集真实导出后跑 02→03→04 链 |
| 真实学生用户测试 | **PROVISIONAL**（工程师模拟） | 按 [CLASSROOM_READINESS_CHECKLIST.md](CLASSROOM_READINESS_CHECKLIST.md) §F 执行 3 人试走 |
| GitHub Actions 首跑 | 待 push 后触发 | 本 PR push 后 Actions 自动执行（本地无 torch 环境已等价验证） |
| 跨平台性能结论 | 永不给出 | Windows XTX vs Linux W7900 耗时差异仅作记录（条件不同，不构成结论） |

## 5. 合并建议

**建议合并 PR #3（在 CI 首跑绿后）**，随后合并本 release-closure PR（叠加于其上，仅含增量）。理由：

1. PR #3 的四个 P0 修复使 main 恢复绿色（main 当前 pytest 是红的——AST 参考损坏）；不合并则仓库带伤发布。
2. 全部修改经三轮真机复现 + 41 项自动化回归 + 4 起门禁真实 FAIL 实证；证据链 SHA256 存证。
3. 本轮 CI/教学/课堂增量与 PR #3 无耦合冲突（仅 md/脚本/测试/CI），叠加审阅成本低（PR 增量 diff）。

**课堂发布判定：CONDITIONAL GO** —— 技术与体验门禁全绿；发布前置条件为 [CLASSROOM_READINESS_CHECKLIST.md](CLASSROOM_READINESS_CHECKLIST.md) A–C 节执行完毕（核心：Preflight 11/11 + 资产预置 + Kernel 预注册 + 预热一次）。Student Learning 结论保持 **PROVISIONAL** 直至 3 人真实试走（§F 表）。

## 6. 下一步最有价值的三个动作

1. **Owner 审阅并合并 PR #3 → release-closure**（合并后 CI 在 main 常驻，任何后续改动自动过门禁；同时更新 README 状态页）。
2. **执行 3 名真实学生试走**（用 §F 验收表；这是 Student Learning 去 PROVISIONAL 的唯一路径，也是发现真实摩擦的最后机会）。
3. **课前按清单完成资产预置 + Preflight + 预热**，并在首课后收集真实 X-AnyLabeling 导出补做 E2E（当前唯一 NOT_RUN 项）。

---

*本报告全部结论由 2026-11-03 本机真实运行支撑；工件哈希与日志见本目录及 `logs/`；未删除、未改写任何历史证据。*
