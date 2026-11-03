# 05 · Notebook Execution Matrix — Linux W7900（最终正式轮）

- 数据来源：`validation/NOTEBOOK_EXECUTION_DATA_20261103_024403.json`（由 `scripts/run_all_labs.py` 自动生成，非人工填写）
- 执行环境：**Linux（Ubuntu 24.04 · kernel 6.8.0-79-generic）· AMD Radeon Pro W7900D（gfx1100, 48 GiB）· uv 隔离栈 torch 2.12.0+rocm7.14.1 · HIP 7.14.60850 · ultralytics 8.4.199**
- Kernel：`yolo26-rocm7141`（registered=True，argv→项目 `.venv/bin/python`）· **Kernel 探测（真实执行）：ok=True**（sys.executable / torch / hip / gpu 全部来自同一 ROCm 环境）
- 执行时间窗：2026-11-03 02:44:03 – 02:45:37（本地，热启动）
- 产物根：`/workspace/yolo_course_validation_linux_w7900_final`（全新隔离目录，运行前不存在）· 执行后副本：`validation/executed/20261103_024403/`
- 练习填充：`--practice-fill validation/linux_w7900/practice_fill_final_run.json`（验收人员填写的真实读数，注入值在输出中原文可见；学生运行时不需要）

| Notebook | Code Cells | Executed | Result | GPU Used | Duration | 关键证据 |
|---|---:|---:|---|---|---|---|
| 00_ROCm_Environment_Preflight | 6 | 6 | **PASS** | 实测（HIP 断言 + GPU matmul + 预测练习） | 4.71 s | `environment.json`；`artifacts/yolo26n.pt`（sha256 `9b09cc8b…` = 官方）；`lab00_practice.json` |
| 01_Lab_A_Predict_and_Threshold | 7 | 7 | **PASS** | 实测（device=0，3×GPU 推理） | 5.48 s | conf .10/.25/.50 → **7/5/5** 检测（单调性断言过）；3 张预测图；IoU 预测练习（实际 IoU=0.4828 vs 预测 0.55） |
| 02_Lab_B_Dataset_Annotation_QA | 7 | 7 | **PASS** | GPU 门禁探测（数据 QA 为 CPU） | 4.06 s | `coco128_classroom_3class/`（train 89/val 20/test 19，261/29/45 框，0 结构错误）；QA `pass=true`；标注画廊；人工检查表 |
| 03_Lab_C_ROCm_Training_Finetune | 7 | 7 | **PASS** | 实测（COCO8 冒烟 1 Epoch + **12-Epoch baseline 真训练**） | 40.44 s | `runs/lab_c/baseline/`（best.pt sha256 `<见 FINAL_REPORT>` + results.csv 12 行 + results.png）；训练 29.7 s；峰值 1.542 GiB；官方 val **P .839 / R .552 / mAP50 .624 / mAP50-95 .383**；读曲线练习工件 |
| 04_Lab_D_Evaluation_Error_Analysis | 7 | 7 | **PASS** | 实测（官方 val + 逐图 GPU predict） | 7.86 s | `lab_d_official_metrics.json`（P .839 / R .552 / mAP50 .624 / mAP50-95 .383）；固定阈值 **TP18/FP6/FN11**（P .750 / R .621，口径差异已在输出明示）；`lab_d_errors.csv` 17 行；**4 张真实失败对照图（如实报告不足 5）**；归因练习工件 |
| 05_Lab_E_Controlled_AB_Optimization | 7 | 7 | **PASS** | 实测（Variant 12-Epoch 真训练 + 双臂同口径 GPU 评测） | 30.65 s | `runs/lab_e/imgsz416/`（best.pt 独立 sha256）；A/B：**512→mAP50-95 .3829（29.7 s, 1.54 GiB）vs 416→.3385（25.3 s, 1.07 GiB）**，Δ=−0.0444；假设先行工件；**Test 封存**（"Test is sealed"） |

**合计：6/6 PASS，41/41 code cells 执行，0 FAIL，0 BLOCKED。退出码 0。**

## 完整性断言（runner 与 Notebook 自身执行）

- Kernel 环境探测（P0-C 修复）：真实启动 Kernel 执行 `sys.executable/torch/hip/gpu` 探针，全部来自 `.venv` ROCm 栈 ✓
- Lab B 工件检查（P0-B 修复）：从 `course_config.json` 读取本轮实际数据集（`COCO128_repartitioned_teaching_only` → `data/processed/coco128_classroom_3class/data.yaml`），并验证 config/QA 新鲜度（本轮重写）✓
- 跨 Notebook 状态链：00 冻结权重（SHA256）→ 02 登记数据集 → 03 产出 best.pt+train_settings → 04/05 消费 ✓
- A/B 纯度：variant 相对 baseline 改动集合 = `{imgsz, project, name}`（断言过）✓
- 初始权重一致性：05 校验 `yolo26n.pt` SHA256 == 00 记录值 ✓
- 阈值单调性（Lab A）、划分互斥（Lab B `pass=true`）、best.pt 存在性（Lab C）全部真实断言通过 ✓
- 学生练习门禁：TODO 断言在无人值守运行中真实触发过一次（`NOTEBOOK_EXECUTION_DATA_20261102_024247.json`，Lab A 因缺填充 FAIL）——门禁有效性有实证；正式轮以透明填充通过 ✓

## 历史 Linux 轮次（全部保留，未删改）

| 时间戳 | 内容 | 结果 |
|---|---|---|
| 20261103_022225 | 主链首轮（教学增强前，37 cells） | 6/6 PASS（冷启动，Lab C 100.05 s） |
| 20261103_022724 | B2 X-AnyLabeling 首试（fixture 场景 CSV 错误） | 02 FAIL——**导入器场景覆盖守卫真实拦截**（异常注入有效性实证） |
| 20261103_022804 | B2 X-AnyLabeling 修正 fixture（02+03） | 2/2 PASS（场景不跨集合；Lab C 在导入集上训练 mAP50-95 .688，fixture 口径） |
| 20261103_023118 | Baseline 复跑波动探针（冻结数据集） | 1/1 PASS（指标与主链 4 位小数一致；耗时 29.33 s 揭示 JIT 冷/热差异） |
| 20261103_024247 | 最终轮首试（练习填充缺 lab_a_iou 键） | Lab A FAIL——**TODO 门禁在无人值守运行真实触发** |
| **20261103_024403** | **最终正式轮（本表主体）** | **6/6 PASS，41/41 cells，退出码 0** |

## 与 Windows 历史（RX 7900 XTX）对照

Windows 执行矩阵（`validation/NOTEBOOK_EXECUTION_MATRIX.md`，20261102_170901）证明 Windows 路径；本轮证明 Linux W7900 路径。同种子同数据下指标几乎一致（mAP50-95 .380 vs .383），交叉印证真实性；耗时差异不作跨平台性能结论（见 02_MODEL_AND_DATA_AUDIT.md §10）。
