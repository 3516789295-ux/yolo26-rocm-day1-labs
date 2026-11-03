# VALIDATION_MATRIX.md — Release Closure 最终复验矩阵

- 时间（UTC）：2026-11-03 03:20–04:00 · 机：AMD Radeon Pro W7900D（gfx1100, 48GB）· uv 隔离栈 torch 2.12.0+rocm7.14.1
- 代码态：release-closure 分支（基于 PR #3 HEAD `372557d`，本轮改动全部为 Markdown/文档/脚本/测试/CI，**代码单元 41 个不变**）
- 图例：PASS=真机实测通过；FAIL=失败（保留证据）；NOT_RUN=前提不满足，如实标注；BLOCKED=被外部条件阻断

## 1. 核心复验（Phase 6 全项）

| # | 项目 | 结果 | 证据 |
|---|---|---|---|
| 1 | GPU Physical Smoke（仓库脚本原样） | **PASS** 17/17 门禁 | `validation/logs/gpu_smoke.json`（本轮重跑，OVERALL: PASS） |
| 2 | pytest（ROCm 环境） | **PASS** 41 通过 / 1 跳过（Windows 专用） | 本轮输出 |
| 3 | pytest（无 torch CI 环境） | **PASS** 41 通过 / 1 跳过（0.60 s） | 本地 ci_env 实测（CI 同构） |
| 4 | pip check | **PASS** 无冲突 | 本轮输出 |
| 5 | AST / Generator 一致性 | **PASS**（pytest test_notebook_ast_reference + test_generator_consistency + test_generated_code_cells_compile） | 41/41 代码单元可编译且与生成器逐字一致 |
| 6 | **Notebook 00–05 全量** | **PASS 6/6 · 41/41 cells · 退出码 0** | `NOTEBOOK_EXECUTION_DATA_20261103_035047.json` + `executed/20261103_035047/`（全新目录 `/workspace/yolo_course_release_final`） |
| 7 | Lab B Fixture E2E（X-AnyLabeling 受控夹具） | **PASS**（02+03，退出码 0） | `executed/20261103_035426`；场景不跨集合；Lab C 在导入集上真训练（val mAP50-95 .6882，**5 图夹具口径，非基准**） |
| 8 | 真实学生导出 E2E | **NOT_RUN** | 本机经全盘检索无真实 X-AnyLabeling 导出（仅 pytest 临时夹具与明确标记的受控 Fixture）；**不以 Fixture 冒充真实用户验收**；课堂引入真实导出后须补测 |
| 9 | Lab C 12-Epoch 训练 | **PASS**（含于 #6；Notebook 41.05 s） | results.csv 12 行 + best.pt + 曲线 |
| 10 | Lab E 12-Epoch A/B | **PASS**（含于 #6；Notebook 31.6 s；计时轮独占 GPU） | 下表 |
| 11 | 工件 SHA256 与跨 Lab 状态链 | **PASS** 6/6 | 初始权重=官方 `9b09cc8b…`；cfg 记录一致；baseline≠variant；Lab D 消费 baseline；数据集 kind 正确；A/B 同验证口径 |
| 12 | Preflight 课堂自检脚本 | **PASS 11/11**（真机实测，退出码 0） | `logs/preflight_demo.json`；首轮自测抓出自身 2 个缺陷并修复（int-dtype NMS 探针、Kernel 路径显示）——自检工具有效性实证 |
| 13 | CI workflow | **已新增**；本地无 torch 环境等价验证 PASS；GitHub Actions 首跑状态见 PR | `.github/workflows/ci.yml`（ubuntu+windows 双平台；守卫测试 `test_ci_workflow.py` 确保不装 torch、必跑 pytest、双平台覆盖） |
| 14 | 证据完整性 | **PASS** SHA256SUMS 18/18 复核 | 历史证据零删改 |

## 2. 本轮真机成绩（第 3 次独立复现，与前两轮确定性一致）

| 项 | 值 | 前两轮对照 |
|---|---|---|
| Lab C baseline val | P .8395 / R .5517 / mAP50 .6237 / **mAP50-95 .3830** | .3830 / .3830 ✓ |
| Lab E baseline(512) | mAP50-95 **.3829** · 28.94 s · 1.542 GiB | .3829 / .3829 ✓ |
| Lab E imgsz416 | mAP50-95 **.3385** · 26.09 s · 1.067 GiB | .3385 / .3385 ✓ |
| Δ(A/B) | **−0.0444**（variant 更差）；耗时差 2.85 s（无有意义加速） | 同 ✓ |

> best.pt 文件哈希每轮不同（内嵌时间戳元数据）；**指标与 state_dict 数值同种子完全可复现**（口径声明见 02_MODEL_AND_DATA_AUDIT §9）。

## 3. 门禁"必须真实 FAIL"实证（累计 4 起，全部保留）

| 时间 | 事件 | 证明的门禁 |
|---|---|---|
| 20261103_022724 | Fixture scenes.csv 用错文件名 → Lab B FAIL | 导入器场景覆盖守卫（missing/extra 精确清单） |
| 20261103_024247 | 练习填充漏 lab_a_iou 键 → Lab A FAIL | 学生 TODO 门禁在无人值守运行同样生效 |
| 本轮注入演示 | 5 类坏数据（类别越界/坐标越界/缺标签/场景缺口/覆盖已存在目录）全部拒绝 | 导入器全部门禁 + 新增初学者提示行 |
| 20261103_035230 | Fixture 链漏传 `--practice-fill` → Lab C 练习格 FAIL | 门禁不因"复验"而放松；修复=补旗标重跑，非绕过 |

## 4. 科学准确性 8 项核查（Phase 4，grep 证据行号）

| 声明 | 证据（build_notebooks.py 行号） | 结果 |
|---|---|---|
| COCO8 仅冒烟 | L698/705 "不用于发布精度成绩 / COCO8 is not a benchmark" | ✓ |
| COCO128 仅教学 | lab_utils.py L255-256 `pedagogical_only:True / not a COCO benchmark` | ✓ |
| Val/Test 隔离 | L1207-1230 "仅在选定模型之后运行一次 / Test is sealed" | ✓ |
| 官方 mAP vs 固定阈值双口径 | L929 "Teaching matching only; not a COCO AP implementation" | ✓ |
| 冷/热启动分别记录与解释 | L692-696 等待说明（一次性 JIT）；报告层冷/热分列 | ✓ |
| allocated ≠ 整卡显存 | L729 warning 字段 + L1235/1247 五问第 4 条 | ✓ |
| 小样本过拟合有据有限度 | 读曲线练习要求"以你的曲线为证据"（引用数字） | ✓ |
| 单次差值 ≠ 统计显著 | L1027/1033 公平性说明 + L1240 面试题 | ✓ |

## 5. 学生体验改进落地核对（详见 STUDENT_EXPERIENCE_IMPROVEMENTS.md）

| 项 | 状态 |
|---|---|
| 六份"学习地图"（七问七答） | ✅ 生成并再生（md-only，41 代码单元不变） |
| 训练等待状态说明（03/05） | ✅ |
| Lab B/D/E 常见错误表 | ✅ |
| Lab E 四指标对比 + 学生空表 | ✅（不预置数值） |
| 学生交互 vs `--practice-fill` 区分 | ✅（README 说明块 + 练习格注释） |
| 导入错误初学者提示行 | ✅（含回归测试） |
| `group{1..5}` shell 修正 | ✅（03 报告 §2.1） |
| Preflight 脚本 | ✅ 11/11 实测 |
| README Linux 复验状态更新 + CI 徽章 | ✅ |
| INSTRUCTOR_GUIDE 清单入口 | ✅ |

## 6. 坏数据注入实录（错误信息初学者友好度）

```
[类别越界] rejected: ValueError: Invalid YOLO detection labels; no files copied:
           .../fixture_000.txt: L1: invalid class id 3.0
           Hint/提示: 每行应为 5 个数 'class x y w h'；类别 ID 必须是 0 到 2 的整数（classes.txt 共 3 行）；坐标是 0–1 的归一化小数。 / Each line ...
[坐标越界] rejected: ... L1: invalid normalized xywh / box extends outside image
[缺标签]   rejected: Missing labels: ['fixture_000.jpg']. Create an explicit empty .txt for verified negatives
[场景缺口] rejected: Scene CSV must cover every image exactly once: missing=['fixture_031_halfscale_of_001.jpg'], extra=[]
[覆盖保护] rejected: FileExistsError: Refusing to overwrite existing classroom dataset: ...
```

## 7. 结论

**Release Closure 复验：PASS（14/14 可测项；1 项 NOT_RUN 如实标注——真实学生导出 E2E）。** 三次独立全链复现指标一致；全部门禁保持且有"真实 FAIL"实证；学生体验与课堂就绪缺口已按最小修改原则补齐。
