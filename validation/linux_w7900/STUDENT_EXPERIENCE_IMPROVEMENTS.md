# STUDENT_EXPERIENCE_IMPROVEMENTS.md — 学生体验改进记录（Release Closure 轮）

- 时间（UTC）：2026-11-03 03:20–04:10
- 基线：PR #3 HEAD `372557d`（其教学评分 4.33/5，见 [06_STUDENT_USABILITY_SCORECARD.md](06_STUDENT_USABILITY_SCORECARD.md)）
- 原则：**只加最短必要提示，不复制外部文档；不改代码单元（41 个不变）；不放松学生门禁；双语经生成器再生**

## 1. 每份 Notebook 的"七个问题"现在全部有答案

新增 `lab_map()` 生成器辅助函数：每份 Notebook 头部（标题后第一格）插入"学习地图（30 秒看完）"双语表格，逐行回答任务要求的七个问题：

| 问题 | 回答位置（六份统一） |
|---|---|
| 我为什么学这个？ | 学习地图第 1 行（一句话价值锚点） |
| 我需要先懂什么？ | 第 2 行（显式前置：00→02→03→04/05 链） |
| 我应该修改哪个参数？ | 第 3 行（**精确到变量名**，如 `SMOKE_N`/`box_pred`/`MY_ANALYSIS`；Lab E 明示"A/B 训练参数不许改"） |
| 运行成功应该看到什么？ | 第 4 行 + 原有"预期输出"节 |
| 运行失败如何自救？ | 第 5 行 → 各 Notebook 常见错误表 → TROUBLESHOOTING |
| 我应该保存什么证据？ | 第 6 行（与交付清单一致） |
| 求职面试价值？ | 第 7 行（面试原话级表述） |

六份地图内容各异（见 `build_notebooks.py` 的六次 `lab_map([...])` 调用），例：Lab B 第 7 行"数据岗必问：你怎么发现训练/测试泄漏？——这节就是标准答案"。

## 2. 训练等待阶段状态解释（03 / 05 新增）

- **03**（冒烟格前）："训练开始后的正常现象"——每 Epoch 一行进度；**首次运行前 ~1 分钟可能看似卡住 = MIOpen 编译 kernel（一次性）**；用 `rocm-smi` 观察；不要中断。
- **05**（Variant 训练格前）：同上 + **"这期间不要启动其他训练/评测——A/B 的相同 GPU 条件要求独占"**（与课堂错峰表呼应）。
- 解决的痛点：上一轮 Beginner Walkthrough 发现"新手会以为卡死"（06_SCORECARD §5 步骤 6 相关）。

## 3. Lab B / D / E 常见错误表（补齐 00/01/03 之外的三个）

| Notebook | 新表覆盖（症状 → 下一步） |
|---|---|
| B | scenes.csv 覆盖缺口（按 missing/extra 清单核对）；类别越界（ID 从 0 开始）；缺标签（建空 txt 负样本，不删图）；FileExistsError（换 `_v2` 名，不删旧结果）；样本/场景不足（课上用 COCO128 路径） |
| D | "先完成 Lab C"断言；errors.csv 为空（如实报告 0 错误）；官方与固定阈值 P/R 差异（双口径）；图不显示（查 save_dir） |
| E | 初始权重 SHA256 改变（重跑 00）；baseline imgsz 断言；variant 参数改多（只许 imgsz）；比 Lab C 慢（并发污染→独占重跑） |

## 4. Lab E 四指标对比（新增"四个指标一起读"节）

- P / R / mAP50 / mAP50-95 各自回答的问题 + **单独引用的陷阱**（高 P 可能只是保守；高 R 可能靠滥报；跨数据集不可比）。
- 学生**空表填写**自己的两臂数据（从 `lab_e_ab_comparison.csv` 抄录）——报告素材直接成型。
- 读法教学："P 降 R 升 = 更敢报但更不准，必须成对解释；四项方向不一致时诚实写'不确定'"。
- **不预置本轮数值**——遵守课程自己的"不可把示例文字当作实测数据"红线。

## 5. 学生交互 vs 自动复验的显式区分

- README"运行节奏"新增说明块：练习格 TODO 断言是**设计而非故障**；`--practice-fill` 是讲师/CI 的显式无人值守通道，注入值原样出现在输出与报告中，与学生课堂互不影响。
- Notebook 练习格内注释原本已有"无人值守复验通道（学生不需要）"字样，保持不变。
- **未默认替学生填写任何 TODO**（默认值仍是 TODO 断言；填充仅在显式 flag 下发生）。

## 6. X-AnyLabeling 导入错误信息的初学者友好化

`course/annotation_import.py` 的 "Invalid YOLO detection labels" 异常现在附带双语提示行：
> Hint/提示: 每行应为 5 个数 'class x y w h'；类别 ID 必须是 0 到 2 的整数（classes.txt 共 3 行）；坐标是 0–1 的归一化小数。

- 回归测试：`test_rejects_invalid_and_missing_labels` 断言提示文本与有效范围；新增场景覆盖缺口拒绝断言（含 missing= 精确清单）。
- 坏数据注入实测（5 类全部拒绝，错误可定位到文件与行）见 [VALIDATION_MATRIX.md](VALIDATION_MATRIX.md) §6。

## 7. 课堂侧配套（详见 CLASSROOM_READINESS_CHECKLIST.md）

- `scripts/preflight_classroom.py`：11 项只读课前自检（版本/HIP/GPU/NMS/权重 SHA256/数据/Kernel/磁盘/冷热状态），FAIL 给 next 步骤，`--json` 留档；本机实测 11/11 PASS。
- `group{1..5}` 错误 shell 示例已修正（03_CLASSROOM_CAPACITY_PLAN.md §2.1）为每组各设各的正确写法并注明原因。
- INSTRUCTOR_GUIDE 顶部新增一键自检与清单入口。

## 8. 不变项（红线核对）

| 红线 | 状态 |
|---|---|
| 代码单元数量 | **41 不变**（本轮全部为 Markdown/文档/脚本/测试改动；pytest AST+生成器一致性通过） |
| 学生 TODO 门禁 | 不变（默认断言触发；无人值守仍需显式 flag） |
| ROCm/泄漏/指纹/A-B 纯度门禁 | 不变（相关测试全绿） |
| 新深度学习框架依赖 | **零**（新增 CI 仅装 pytest/nbformat/pillow/numpy/pyyaml；测试套件无 torch 依赖已实测） |
| 历史证据 | 未删改（SHA256SUMS 18/18 复核通过） |

## 9. 预期效果（对照上轮评分的薄弱项）

| 上轮扣分点（06_SCORECARD） | 本轮对策 | 预期变化 |
|---|---|---|
| 02/04/05 无错误表（P2 遗留） | §3 三张表 | 排障 4→5 |
| 训练等待无说明（易实践/反馈扣分） | §2 等待说明 | 易实践 +0.5 |
| Lab E 单指标（反馈扣分） | §4 四指标对比 | 反馈 +0.5 |
| README 无学生交互说明 | §5 | 初学者首次上手摩擦 − |
| 预计总均分 | — | 4.33 → ~4.5（待真实学生试走复核） |
