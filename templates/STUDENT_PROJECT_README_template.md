# 🎓 我的 YOLO26 ROCm 目标检测实训作品 · My YOLO26 ROCm Detection Portfolio Project

> **使用说明**：复制本模板到你的仓库 README，把 `<...>` 占位符换成**你自己的真实数据**。
> 没跑出来的数字不要编造——"未完成/待补"比假数字更专业。
> How to use: copy this template into your repo README and replace every `<...>`
> placeholder with **your own real numbers**. Never fabricate metrics —
> "not completed / TODO" is more professional than a fake number.

---

## 1. 问题与业务价值 · Problem & Value

<用 2–3 句话描述你要解决什么问题，例如：校园快递分拣、安全帽检测、课堂出勤统计……以及为什么目标检测是合适的技术。>

<Describe in 2–3 sentences what problem you solve and why object detection is the right tool.>

## 2. 数据集 · Dataset

- 来源：<例如：讲师提供的已授权图片 / 自拍场景 / COCO128 教学子集>
- 规模：<N 张图，M 个标注框，K 个类别：...>
- 标注工具与规范：X-AnyLabeling · YOLO Detection txt · 类别顺序见 classes.txt
- 划分：train <N1> / val <N2> / test <N3>（场景分组防泄漏，`split_manifest.csv` 可查）
- **诚实边界**：这是教学规模的小数据集，成绩不能外推为工业精度。

## 3. 环境 · Environment

| 项 | 值 |
|---|---|
| GPU | <例如 AMD Radeon Pro W7900 48GB (gfx1100)> |
| 平台 | <Linux / Windows，OS 版本> |
| ROCm / PyTorch | <例如 ROCm 7.14.1 · torch 2.12.0+rocm7.14.1> |
| 模型 | Ultralytics YOLO26n，初始权重 SHA256 `<你的 00 冻结值>` |

## 4. 复现 · Reproduce

```bash
# 1. 环境按 docs/RADEON_CLOUD_LINUX_SETUP.md（或 ROCM_WINDOWS_SETUP.md）部署
# 2. 顺序执行 notebooks/00 → 05
# 3. 一键复验（可选）
python scripts/run_all_labs.py --course-home <你的输出目录>
```

## 5. 结果 · Results（全部来自我自己的运行）

| 模型 | 训练配置 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| baseline | imgsz=512 · 12 ep · SGD · batch 8 | `<填>` | `<填>` | `<填>` | `<填>` |
| variant | imgsz=416 · 其余全同 | `<填>` | `<填>` | `<填>` | `<填>` |

- 训练耗时 <A: xx s / B: xx s>；峰值显存 <A: x.xx GiB / B: x.xx GiB>（PyTorch allocated，非整卡）
- 最终 Test（模型选择冻结后仅运行一次）：<如未运行，写"未解锁，遵循课程规范">
- 证据文件：`reports/lab_d_official_metrics.json`、`reports/lab_e_ab_comparison.csv`

## 6. 错误分析 · Error Analysis（三个有证据的案例）

1. **<图片名/错误类型>**：现象 <...>；证据 <score/max_iou 数字>；假设 <...>；验证计划 <...>
2. <...>
3. <...>

## 7. 受控实验结论 · A/B Conclusion

- 单一变量：训练 imgsz（512 → 416）；其余条件（数据/Epoch/Batch/种子/评测口径）一致。
- 我的假设（实验前封存）：`<引用 reports/lab_e_my_hypothesis.json>`
- 实际结果与假设的对照：<一致/不一致 + 一句反思>
- 结论：<哪个更好/更差/不确定>；局限：单次运行、小数据集、无统计显著性；训练耗时 ≠ 推理延迟。

## 8. 已知局限 · Limitations

<例如：128 图教学集、12 Epoch、单一种子、单一场景类型、未做部署侧延迟测试……>

## 9. 下一步 · Next Steps

<例如：多种子重复、扩充独立业务数据、ONNX/OpenVINO 部署与延迟对比、更大模型……>

## 10. 许可 · License

- 代码：MIT（本仓库）
- 模型/数据资产：遵循 Ultralytics License、COCO 许可等原始条款；**未入库**大权重与未授权图片
- 我的标注数据：<你的授权说明>

---

### 面试两分钟陈述结构 · 2-Minute Interview Pitch

**Problem → Dataset → Method → Evidence → Impact**

> "我在一块 AMD W7900 上用 ROCm PyTorch 微调了 YOLO26n 解决 <问题>。
> 数据 <N> 张、<K> 类，X-AnyLabeling 标注并做了场景分组防泄漏划分。
> 方法上我做了受控 A/B：只改训练 imgsz，同种子同口径评测。
> 证据是 <你的 mAP 数字> 与三个带证据链的错误分析，全部工件可哈希追溯。
> 收获是我完整走通了 数据→训练→评测→实验 的闭环，并知道每个数字的边界。"

面试官追问时，用实际文件回答：权重 SHA256 → `reports/`；划分证据 → `split_manifest.csv`；A/B 纯度 → `lab_e_ab_comparison.csv` 的 args 对比。
