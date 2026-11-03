# 02 · Model & Data Audit — Linux W7900 Round

- 审计时间（UTC）：2026-11-03 02:30–02:55
- 审计对象：本轮 Linux W7900 真机运行产物（`/workspace/yolo_course_validation_linux_w7900` 主链、`..._xanylab` B2 链、`/workspace/yolo_course_variance_probe` 波动探针）
- 方法：**独立重算**——不信任 Notebook 自身报告，从磁盘工件重新计算哈希/划分/标签/指标链
- 结论：**无数据泄漏、无权重污染、A/B 可比性成立、指标口径标注诚实**；含一项同种子复跑波动分析

## 1. 数据集泄漏独立复核（COCO128 教学重划分）

| 检查 | 方法 | 结果 |
|---|---|---|
| Train/Val/Test 精确重复 | 全部 128 张图独立 SHA256 重算并查跨集合碰撞 | **0 碰撞**（train 89 / val 20 / test 19） |
| 同名文件跨集合 | 文件名集合交运算 | **0** |
| 近重复跨集合 | dHash 启发式（Lab B QA 内置 + 人工复核清单） | QA warnings 为空；启发式不是语义证明，Notebook 已明示需人工审查 |
| 标签合法性 | 335 行标签全部重验（类别 0–2、归一化坐标、框不出界） | **0 违例** |
| 空标签负样本 | 57 张空标签图（合法负样本，ultralytics 视为背景） | 一致，且图像侧均有对应 txt 文件 |
| X-AnyLabeling 路径 | 32 图 fixture：场景不跨集合 | **0 违例**；精确字节重复对（fixture_000/030）与半尺度近重复（001/031）均落入同一 split ✓ |

## 2. 类别/标签/模型一致性

- data.yaml `names {0: person, 1: car, 2: chair}` 与标签类别 ID、Lab D 混淆矩阵类别一一对应 ✓
- Lab D 固定阈值匹配中 TP+FN=18+11=29 = val 集 GT 框总数 ✓（口径自洽）

## 3. COCO8 / COCO128 是否被当成正式基准

- COCO8 仅用于 Lab C 第一步"训练链路冒烟"（Notebook 明示 "COCO8 is not a benchmark"）✓
- COCO128 重划分的 `dataset_manifest.json` 含 `pedagogical_only: true` 与 "not a COCO benchmark" 声明 ✓
- README/Notebook/讲师指南三处一致声明小数据集成绩不得外推 ✓
- Lab E 最终 Test 报告字段内置 `warning: Tiny classroom split; not a statistically reliable industrial benchmark` ✓

## 4. 模型选择与 Test 隔离

- Lab E `RUN_FINAL_TEST=False` 默认封存 Test；本轮输出 "Test is sealed. Current metrics are Validation only." ✓
- 模型选择逻辑（若启用）仅按 validation mAP50-95，代码与文档一致 ✓
- Lab D 全部指标基于 `split='val'` ✓

## 5. mAP 定义核对

- `get_metrics_summary`：`box.map50`=mAP50、`box.map`=mAP50-95（IoU 0.50:0.05:0.95 平均）✓
- Lab D 明示固定 conf=0.25/IoU=0.50 教学匹配 **不是** COCO AP（`note` 字段写入 JSON）✓
- 本轮实测对照：官方 P .839 / R .552 / mAP50 .624 / mAP50-95 .383 vs 教学匹配 P .750 / R .621（TP18/FP6/FN11）——两口径差异符合预期，且 Notebook 解释了原因 ✓

## 6. 权重与缓存污染检查

| 项 | 结果 |
|---|---|
| 初始权重指纹 | `yolo26n.pt` SHA256 = `9b09cc8b…4fef`（= 官方 ultralytics/assets v8.4.0，= 00 冻结值，= 05 复核值）✓ |
| A/B 初始一致性 | 05 断言 `sha256(model_cache) == cfg['model_sha256']` 真实通过（variant 从同一预训练权重初始化，非 baseline best.pt）✓ |
| baseline/variant best.pt | 两个不同 SHA256（c13589ea… / 56972089…），无互相覆盖 ✓ |
| Lab D 消费权重 | `lab_d_official_metrics.json.weight_sha256` = baseline best.pt ✓ |
| B2 链隔离 | 独立 course home，不影响主链工件 ✓ |

## 7. A/B 两臂可比性

- 训练参数差异集合实测 = `{imgsz, project, name}`（Notebook 05 断言 + 独立 JSON diff 双重确认）；`project/name` 为日志位置，算法变量仅 `imgsz` ✓
- 评测协议一致：同 val split、同 `imgsz=640`、同 batch/workers/device ✓
- 结果（val，imgsz=640）：

| 臂 | 训练 imgsz | 耗时 | peak alloc | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| A baseline | 512 | 63.61 s | 1.542 GiB | .839 | .552 | .6237 | **.3830** |
| B variant | 416 | 61.18 s | 1.067 GiB | .620 | .586 | .6355 | **.3385** |

- ΔmAP50-95 = **−0.0444**（B 更差）；训练耗时差 2.4 s（**不构成有意义的加速**，且两臂均为独立进程、各含 JIT 预热，条件相当）；显存 1.07 vs 1.54 GiB（B 更省，方向符合直觉）
- **诚实性判定**：Notebook 未预置"更小更快更好"结论；结论五问要求区分算法变量/日志变量、训练耗时/推理延迟、单次结果/统计显著性 ✓

## 8. 显存与耗时统计口径

- 显存 = PyTorch `max_memory_allocated`（allocated peak），JSON 内置 warning 明示**非整卡 VRAM**；W7900D 48 GiB 下占用 1.5/1.1 GiB，余量充足 ✓
- 耗时含 JIT 预热（见 §9），Notebook 05 已提示训练耗时受缓存与并发负载影响 ✓

## 9. 同种子复跑波动分析（本机新增证据）

同一冻结数据集、同一初始权重、同一超参（seed=42, deterministic=False）复跑 12-Epoch baseline：

| | 训练耗时 | peak alloc | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Run 1（主链，冷） | 63.61 s | 1.542 GiB | .8395 | .5517 | .6237 | .3830 |
| Run 2（探针，热） | 29.33 s | 1.542 GiB | .8395 | .5517 | .6237 | .3830 |

- **指标完全一致（4 位小数）**：同种子同栈同数据在本机上可精确复现；A/B 差值（−0.0444）远大于复跑噪声（0）
- **耗时差 34.3 s 全部来自 MIOpen JIT 冷/热缓存**：首次训练需编译 kernel；Lab C 整 Notebook 冷 100.05 s → 热 40.97 s（2.4×）
- 教学含义：**首次运行的耗时不能当作稳态训练速度**；课堂排课必须按冷启动预留（详见 03_CLASSROOM_CAPACITY_PLAN.md）
- 本次复跑为透明披露的波动分析，**未**用于挑选或替换正式成绩

## 10. 与历史 Windows RX 7900 XTX 结果的参考对比

| 指标 | Windows 11 · RX 7900 XTX · ROCm 7.14.1（历史） | Linux · W7900D · ROCm 7.14.1（本轮） |
|---|---|---|
| Lab D 官方 val | P .838 / R .552 / mAP50 .624 / mAP50-95 .380 | P .839 / R .552 / mAP50 .624 / mAP50-95 .383 |
| Lab D 固定阈值 | TP18/FP6/FN11 | TP18/FP6/FN11 |
| Lab C Notebook 耗时 | 200.19 s | 100.05 s（冷）/ 40.97 s（热） |
| Lab E Notebook 耗时 | 89.56 s | 66.87 s |

- 同种子同数据下指标几乎一致（mAP50-95 差 .003 属跨平台数值噪声），**交叉印证了两轮结果的真实性**
- 耗时差异**不可**作为两块 GPU 性能优劣结论：平台（Windows/Linux）、驱动、MIOpen 缓存状态、CPU/IO 不同，且历史 Windows 运行含其自身 JIT 预热。仅作参考记录。

## 11. 猜测 vs 已确认根因（本轮报告口径自查）

- "Lab B 2.6 s 完成" → 已确认：预置 zip 本地解压 + NVMe + 单线程小图处理，非跳过（QA 报告 128 图齐全）
- "耗时波动" → 已确认：MIOpen JIT 冷/热（§9 复跑实验证据）
- "指标复现" → 已确认：同种子精确一致（§9）
- 失败案例归因（Lab D MY_ANALYSIS 示例）→ 明确标注为 suspected_cause（假设），verify_next_step 给出验证路径，未写成已确认

## 12. 判定

**数据与评测科学性审计：PASS。** 无泄漏、无污染、口径诚实、A/B 可比；新增同种子复跑证据将"耗时波动根因"从猜测升级为已确认（JIT 缓存）。
