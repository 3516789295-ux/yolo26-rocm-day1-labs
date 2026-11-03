# YOLO26 ROCm Day 1 Labs：仓库分析

## 仓库概览

本仓库面向大学生提供一天入门级 AMD GPU 目标检测实训，采用中英双语 Notebook，覆盖 ROCm 环境检查、推理与阈值、数据集 QA、GPU 训练、mAP 评价与受控 A/B 实验。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `notebooks/` | 6 份中英双语课程 Notebook，由 `build_notebooks.py` 生成 |
| `course/lab_utils.py` | 数据、QA、评测和路径相关共享逻辑 |
| `scripts/` | GPU 冒烟检查、全课程运行、平台验证及 MIOpen shim 安装 |
| `tests/` | 标签、IoU、数据泄漏、A/B 纯度、ROCm 门禁和 Notebook 结构回归测试 |
| `docs/` | Windows/Linux 部署及故障排查材料 |
| `validation/` | GPU 验证报告、执行矩阵、执行后的 Notebook 与日志 |

## 六个实验

1. 00 环境预检：检查 ROCm/HIP、GPU 冒烟测试并冻结权重。
2. 01 Lab A：推理与置信度阈值，观察边界框、IoU 和预测结果。
3. 02 Lab B：YOLO 标注、数据 QA 与防泄漏划分。
4. 03 Lab C：COCO8 冒烟及 12 Epoch 微调，生成本地权重和训练曲线。
5. 04 Lab D：比较官方 mAP 与教学固定阈值匹配，分析失败案例。
6. 05 Lab E：只改变 imgsz 的受控 A/B 对照实验。

课程另提供 X-AnyLabeling 导出数据导入路径。README 特别指出，学生自带数据的完整训练链路仍需真机走通，离线检查不能称为 GPU 验证通过。

## 平台和验证范围

仓库 README 报告 Windows 11 原生环境、Python 3.12.16、ROCm 7.14.1、Radeon RX 7900 XTX（gfx1100）验证；并新增 AMD Radeon Cloud Linux 真机验证：Radeon Pro W7900D（gfx1100、48 GB）、Ubuntu 24.04、隔离安装的 ROCm PyTorch 7.14.1。公开验证材料报告 6/6 Notebook 通过，Linux 矩阵为 41/41 code cells，并包含训练与受控 A/B 实验。具体运行轮次和平台以 validation/ 中的证据文件为准。

以上为仓库公开记录，本分析没有独立运行 ROCm、Notebook 或 GPU 实验。不可据此推定其他 GPU、系统或依赖组合已验证。README 摘要与验证报告的运行轮次和计数口径可能不同，引用数字时应标明对应证据。

Windows 部署文档记录 ROCm 7.14.1 wheel 的 MIOpen/hipRTC 编译问题，并提供 scripts/install_miopen_shims.py 解决路径。相关说明见 docs/ROCM_WINDOWS_SETUP.md 和 docs/TROUBLESHOOTING.md。

## 数据与实验诚信

- 教学数据为 COCO128 的 person/car/chair 三类重划分，按图片隔离 train/val/test；不能描述为工业数据或 COCO 榜单成绩。
- Lab D 固定阈值 TP/FP/FN 匹配用于教学，不等同于 COCO mAP 算法。
- 不得混分训练集与测试集、用测试集调参、伪造训练产物或预置性能提升结论。
- 数据集与训练权重（如 *.pt）不应提交；第三方模型、数据和图片资产须遵守各自许可。

## 贡献与复现

贡献指南要求从 main 创建 feat/<topic> 或 fix/<topic> 分支。提交 PR 前运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

涉及 Notebook 或 build_notebooks.py 时，运行生成器：

```powershell
.\.venv\Scripts\python.exe build_notebooks.py
```

Notebook 由 build_notebooks.py 生成，应修改生成器而非手改 .ipynb，并保持 Markdown 中英对照。PR 应填写现象、根因和验证证据并关联 Issue；不得移除 ROCm 门禁、数据泄漏检查或 A/B 纯度断言。

## 本文档验证

内容依据公开 README.md、CONTRIBUTING.md 与 validation/ 目录整理。本文档分析未执行仓库测试、Notebook 或 ROCm GPU 训练。
