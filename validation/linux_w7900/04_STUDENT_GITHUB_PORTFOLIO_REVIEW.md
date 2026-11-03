# 04 · Student GitHub Portfolio Review — Linux W7900 Round

- 评审时间（UTC）：2026-11-03 03:05
- 问题：完成六份 Notebook 后，学生是否拥有构建**有说服力 GitHub 求职作品**的全部素材？
- 结论：**素材链完整（10/10 项）**；本轮新增学生项目 README 模板补齐最后的表达层缺口。

## 1. 最低素材清单核查（实测产物对照）

| # | 要求 | 学生实际拥有的工件（本轮真机验证存在且非空） | 判定 |
|---|---|---|---|
| 1 | 项目目标与业务价值 | Lab B 场景选择 + 模板 §1 引导 | ✓ |
| 2 | 数据集来源与标注规则 | `dataset_manifest.json`（来源/划分/种子）、classes.txt、标注画廊图、人工检查表 CSV | ✓ |
| 3 | 训练环境与模型版本 | `environment.json`（torch/HIP/GPU/ultralytics 版本 + 设备节点） | ✓ |
| 4 | 训练与评测复现命令 | README 快速开始 + `run_all_labs.py --course-home` + 本模板 §4 | ✓ |
| 5 | best.pt 的版本与哈希 | `lab_c_baseline.json`（sha256/args/时长/显存） + `lab_d_official_metrics.json.weight_sha256` | ✓ |
| 6 | P/R/mAP 真实成绩 | `lab_d_official_metrics.json`、`lab_c_baseline_val.json`、`lab_e_ab_comparison.csv` | ✓ |
| 7 | 三个有证据的错误分析 | `lab_d_errors.csv`（行级 score/max_iou）+ 失败对照图 + `lab_d_my_error_analysis.json`（四字段归因，本轮新增） | ✓ |
| 8 | A/B 对照实验及结论 | 双臂 best.pt 哈希、同口径评测 CSV、ΔmAP、假设先行工件 `lab_e_my_hypothesis.json`（本轮新增） | ✓ |
| 9 | 模型已知局限 | Notebook 内置 warning 字段 + 模板 §8 | ✓ |
| 10 | 下一步部署优化方向 | Lab E "进阶"节（ONNX/OpenVINO/多种子）+ 模板 §9 | ✓ |

**判定：10/10。** 相比上游 main，本轮新增 #7 的结构化归因工件与 #8 的假设先行工件，使"证据链"从"有数字"升级为"有方法论"。

## 2. 新增交付：学生项目 README 模板

`templates/STUDENT_PROJECT_README_template.md`：
- 全部用 `<占位符>` 而非示例数字——**结构上杜绝虚构指标**（"未完成/待补"是被明确鼓励的写法）。
- 中英双语；含 2 分钟面试陈述结构（Problem → Dataset → Method → Evidence → Impact）与"用文件回答追问"映射表。
- 与 INSTRUCTOR_GUIDE 的 GitHub 交付标准（README/QA/训练/评测/实验四报告 + 哈希）逐项对齐。

## 3. 面试陈述可支撑性（角色 C 视角模拟）

| 追问 | 学生可出示 |
|---|---|
| "你怎么保证只有一个变量？" | `lab_e_ab_comparison.csv` 同行对比 + Notebook 05 的 changed-args 断言输出 |
| "权重哪来的？改过吗？" | 00 冻结 SHA256 + 05 复核断言（同值） |
| "数据没泄漏？" | `split_manifest.csv` + QA JSON `EXACT IMAGE LEAKAGE` 检查为空 |
| "mAP50-95 什么意思？" | Lab D 思考题标准答案（IoU 0.50:0.05:0.95 平均，非置信度） |
| "为什么不用测试集调参？" | Lab E Test 封存机制的实际输出（"Test is sealed"） |

## 4. 许可与合规风险检查

| 项 | 状态 |
|---|---|
| 仓库代码/文档 | MIT（LICENSE）✓ |
| YOLO26 权重/Ultralytics 代码 | AGPL-3.0/Ultralytics License——仓库不分发权重（.gitignore `*.pt`），学生作品同样只提交哈希不提交权重 ✓ |
| COCO128 图像 | COCO 许可，教学使用；仓库不分发（.gitignore `coco128/`、`*.zip`）✓ |
| bus.jpg | 随 ultralytics 包内置（Ultralytics 条款）✓ |
| 学生自标数据 | Lab B 明确要求"先确认版权/使用权限再导入"；manual review CSV 与学生图片不入库 ✓ |
| environment.json 隐私 | 00 明确不写代理 URL/令牌；模板提醒提交前复查 ✓ |

**风险判定：低。** 唯一需课堂强调：学生个人仓库**不得**上传无权图片与大权重文件（模板 §10 已内置提醒）。

## 5. 判定

**GitHub 求职作品交付体验：GO。** 素材 10/10，模板补齐表达层；许可合规有明确护栏。
