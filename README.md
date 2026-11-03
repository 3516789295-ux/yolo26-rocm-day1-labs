# YOLO26 ROCm Day 1 Labs · AMD GPU 目标检测实训(中英双语)

**[English](README.en.md) | 中文**

![status](https://img.shields.io/badge/status-validated%20on%20real%20GPU-brightgreen) ![gpu](https://img.shields.io/badge/GPU-AMD%20Radeon%20RX%207900%20XTX%20gfx1100-red) ![rocm](https://img.shields.io/badge/ROCm-7.14.1-orange) ![torch](https://img.shields.io/badge/PyTorch-2.12.0%2Brocm7.14.1-ee4c2c) ![python](https://img.shields.io/badge/Python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![lang](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87%20%7C%20English-informational) [![CI](https://github.com/AIwork4me/yolo26-rocm-day1-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/AIwork4me/yolo26-rocm-day1-labs/actions/workflows/ci.yml)

> 面向大学生的 **一天入门级 AMD GPU 目标检测实训**:从 ROCm 环境验证 → 推理与阈值 → 数据标注与防泄漏划分 → 真实 GPU 训练 → mAP 评价与错误分析 → 受控 A/B 优化实验。
> 全部 6 份 Notebook 为**中英对照**;每一份都已在一块真实的 **AMD Radeon RX 7900 XTX(gfx1100)+ Windows 11 原生 ROCm** 环境上完整执行通过(37/37 cells,含两次 12-Epoch 真实训练)——证据见 [`validation/`](validation/FINAL_REPORT.md)。

## ✅ 已验证状态(不是承诺,是实测)

| 项 | 结果 |
|---|---|
| 环境 | Windows 11 原生(无 WSL/Docker)· uv · Python 3.12.16 |
| PyTorch 三件套 | torch `2.12.0+rocm7.14.1` · torchvision `0.27.0+rocm7.14.1` · torchaudio `2.11.0+rocm7.14.1`(repo.amd.com 官方 wheel) |
| GPU | RX 7900 XTX · `gcnArchName=gfx1100` · HIP 7.14.60850 |
| Notebook | **6/6 PASS**([执行矩阵](validation/NOTEBOOK_EXECUTION_MATRIX.md)) |
| 回归测试 | 16/16 通过(`pytest`,含数据泄漏检测、A/B 纯度、ROCm 门禁防绕过) |
| A/B 实验 | 真实受控实验:imgsz 512 vs 416,忠实报告(见[最终报告 §E](validation/FINAL_REPORT.md)) |

## 🚀 快速开始

### 方式 A · Windows 原生(AMD RDNA3 显卡,如 RX 7900 系列)

```powershell
git clone https://github.com/AIwork4me/yolo26-rocm-day1-labs.git
cd yolo26-rocm-day1-labs
```

按 **[docs/ROCM_WINDOWS_SETUP.md](docs/ROCM_WINDOWS_SETUP.md)** 一步步执行(约 15 分钟):
uv 建环境 → 安装 AMD ROCm PyTorch(wheel 来自 repo.amd.com,非 PyPI)→ 装课程依赖 → 注册 Jupyter Kernel →
**运行 `scripts/install_miopen_shims.py --setx`**(Windows 训练必需的一步修复,详见下文)→ 打开 Notebook。

> **Windows 用户必读**:ROCm 7.14.1 wheel 的 MIOpen 存在已知缺陷(hipRTC 编译训练 kernel 时找不到宿主
> `<type_traits>`,导致 BatchNorm/Conv 训练全部失败,即 [ROCm/rocm-libraries#13437](https://github.com/ROCm/rocm-libraries/pull/13437)
> 修复的问题)。本仓库内置官方扩展点方案(`HIPRTC_COMPILE_OPTIONS_APPEND` + freestanding shim 头),
> 一条命令安装,锁定版本零改动。详见 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

### 方式 B · Linux / AMD Radeon Cloud

按 **[docs/RADEON_CLOUD_LINUX_SETUP.md](docs/RADEON_CLOUD_LINUX_SETUP.md)**:上传解压到 `/workspace`,
用平台的 ROCm Kernel 或自建 uv 环境打开 Notebook。

> **Linux 已真机复验(2026-11-03)**:AMD Radeon Pro W7900D(gfx1100, 48GB)· Ubuntu 24.04 · uv 隔离栈
> torch 2.12.0+rocm7.14.1——6/6 Notebook、41/41 code cells、两次 12-Epoch 训练与完整 A/B 全部通过。
> 证据:[validation/linux_w7900/](validation/linux_w7900/FINAL_REPORT.md)(基线审计、GPU 准入、数据审计、
> 课堂容量、执行矩阵、学生体验评分、缺陷修复、兼容性共 10 份报告 + SHA256 存证)。

### 运行节奏

1. 从 `notebooks/00_ROCm_Environment_Preflight.ipynb` 开始,**按 00 → 05 顺序执行**(03 依赖 00 的权重缓存与 02 的数据集,04/05 依赖 03 的 `best.pt`)。
2. 全部输出写入项目内 `yolo_course/`(可用环境变量 `YOLO_COURSE_HOME` 重定向)。
3. 想一键复验?`powershell -File scripts/validate_windows.ps1`(依赖检查 → GPU 冒烟 → 回归测试 → 6 份 Notebook 全量执行)。

> **学生交互 vs 自动复验**:六份 Notebook 各有一个"学生主动操作"练习格(预测/读曲线/错误归因/假设先行),
> 带 TODO 占位断言——**不自己填写就无法运行通过**,这是设计而非故障。讲师或 CI 的无人值守复验走
> `run_all_labs.py --practice-fill <json>` 显式通道(注入值会原样出现在输出与报告里),与学生的课堂运行互不影响。

## 📓 六份 Notebook

| # | Notebook | 内容 | 关键产物 |
|---|---|---|---|
| 00 | [环境预检](notebooks/00_ROCm_Environment_Preflight.ipynb) | ROCm/HIP 验证、GPU 冒烟、权重冻结 | `environment.json` |
| 01 | [Lab A · 推理与阈值](notebooks/01_Lab_A_Predict_and_Threshold.ipynb) | conf 0.10/0.25/0.50、BBox 张量、IoU | 3 张预测图 + JSON |
| 02 | [Lab B · 数据与 QA](notebooks/02_Lab_B_Dataset_Annotation_QA.ipynb) | YOLO 标签、防泄漏划分、标注质检 | 三类教学数据集 + QA 报告 |
| 03 | [Lab C · GPU 训练](notebooks/03_Lab_C_ROCm_Training_Finetune.ipynb) | COCO8 冒烟 + 12-Epoch 微调 | `best.pt` + 训练曲线 |
| 04 | [Lab D · 评价分析](notebooks/04_Lab_D_Evaluation_Error_Analysis.ipynb) | 官方 mAP vs 教学固定阈值匹配、失败案例 | PR 曲线 + 错误清单 |
| 05 | [Lab E · A/B 实验](notebooks/05_Lab_E_Controlled_AB_Optimization.ipynb) | 只改 imgsz 的受控对照实验 | A/B 报告 + 图表 |

## 🗂️ 仓库结构

```
notebooks/        6 份中英对照 Notebook(由 build_notebooks.py 单一事实源生成)
course/           lab_utils.py — 数据/QA/评测/路径共享库(无 torch 依赖,可离线测试)
scripts/          gpu_smoke · run_all_labs · validate_windows · MIOpen shim 安装器
tests/            16 项离线回归(标签/IoU/泄漏/A-B 纯度/ROCm 门禁/Notebook 结构)
docs/             Windows 与 Linux 部署、故障排查(中英对照)
validation/       真实 GPU 验证证据:最终报告、执行矩阵、执行后 Notebook、日志
build_notebooks.py 双语生成器(代码 cell 与验证版字节级一致)
constraints-rocm.txt / pyproject.toml   AMD wheel 版本硬约束
```


### Lab B · X-AnyLabeling 人工标注与数据导入（新版）

推荐学生使用开源 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) 桌面软件：**手动标注至少 5 张图片 → 同伴交叉复核 5 张 → 导出 YOLO Detection txt → 上传 Radeon Cloud → Notebook 自动 QA 与分组划分**。完整可操作步骤见 [Lab B 标注指南](docs/LAB_B_X_ANYLABELING.md)。

启动 Lab B 前，设置 `YOLO_ANYLABELING_EXPORT` 指向包含 `images/`、`labels/`、`classes.txt` 的导出文件夹；可选 `YOLO_ANYLABELING_SCENES` 指向包含 `image,scene` 列的 CSV。**不设置时仍运行历史 COCO128 演示路径**，不会破坏已有 Windows ROCm 训练链路。新导入结果只写入新的 `data/processed/xanylabeling_student_v1`，旧结果不覆盖。学生数据与云端 Linux GPU 路径尚需真机走通，不把离线检查称为新路径的 GPU PASS。

## 🧪 数据与诚信原则(课程核心)

- 教学数据为官方 **COCO128** 的 person/car/chair 三类重划分(train 89 / val 20 / test 19,图片级互斥);
  **不是**工业数据集,成绩**不得**冒充 COCO 榜单或工业精度。
- 自定义数据集通过 `YOLO_USER_DATA_YAML` 指向自己的 `data.yaml`。
- **严禁**:训练/测试混分、用测试集调参、伪造 `best.pt`、预置"提升"结论、把冒烟成绩当正式成绩。
- Lab D 的固定阈值 TP/FP/FN 匹配**不是** COCO mAP 算法——Notebook 内有明示。

## 🤝 参与贡献(欢迎学生 PR!)

这是一个**为学生实验优化而开放**的仓库。做完实验,欢迎把你的改进提回来:

- 📖 改进中英翻译、补充解释 → 直接 PR
- 🐛 修一个真实 bug(先 `pytest` 复现)→ Bug Fix PR
- 🧪 新增实验想法(如新 A/B 变量、更多 QA 检查)→ 先开 Issue 讨论
- 评分维度参考 [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md) 的"GitHub 证据及表达"

流程见 **[CONTRIBUTING.md](CONTRIBUTING.md)**(分支 → 测试 → PR 模板,中英对照)。

## 📄 许可证

- 本仓库原创代码与文档:**MIT**(见 [LICENSE](LICENSE))。
- 运行时下载的第三方资产遵循各自许可:Ultralytics YOLO 权重与代码(Ultralytics License/AGPL-3.0)、COCO128 图像与标注(COCO 许可)、`bus.jpg`(Ultralytics)。本仓库不分发上述资产。
- 训练产物(`best.pt` 等)不入库,由学生本地生成。

## 📎 延伸

- [验证最终报告](validation/FINAL_REPORT.md) · [执行矩阵](validation/NOTEBOOK_EXECUTION_MATRIX.md) · [GPU 准入证据](validation/GPU_PHYSICAL_SMOKE.md)
- [讲师指南](INSTRUCTOR_GUIDE.md)(双语)· [设计规范](DESIGN_SYSTEM.md)
- 官方资料:[Ultralytics AMD 集成](https://docs.ultralytics.com/integrations/amd/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/) · [ROCm](https://rocm.docs.amd.com/)
