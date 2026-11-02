# 最终报告(FINAL_REPORT)— YOLO26 ROCm Day 1 Labs · Windows 原生端到端验证

验证日期:2026-11-02 · 项目:`C:\Users\Rocm\Desktop\YOLO\YOLO26_ROCm_Day1_Labs`

---

## A. Executive Summary

- **总体判定:`DAY-1 LABS WINDOWS ROCm = GO`**(12/12 验收条件全部满足,见 §I)
- Windows 原生验证结果:**6/6 Notebook 全部 PASS(37/37 code cells 实际执行,0 FAIL / 0 BLOCKED)**,全程无 WSL/VM/Docker/CPU 降级。
- GPU:AMD Radeon RX 7900 XTX(24GB,PCI DEV_744C = Navi31),**gfx1100**(PyTorch `gcnArchName` 实测)。
- 版本:Python 3.12.16(uv)| uv 0.13.6 | ROCm/HIP 7.14.60850(wheel 栈 rocm7.14.1)| PyTorch **2.12.0+rocm7.14.1** / torchvision **0.27.0+rocm7.14.1** / torchaudio **2.11.0+rocm7.14.1**(全部来自 repo.amd.com,精确匹配任务锁定版本)| Ultralytics 8.4.198。
- 过程中攻克一个真实硬阻塞:rocm7.14.1 wheel 的 MIOpen hipRTC 缺宿主 STL(即任务参考链接 [ROCm/rocm-libraries#13437] 所述"Windows 无法训练 Ultralytics YOLO"缺陷,实测该 wheel 未含修复)→ 以 hipRTC 官方扩展点 + freestanding shim 本地等价修复,锁定版本零改动。

## B. 环境验证

- uv 隔离环境(.venv,--seed pip),`sys.executable` 指向项目 venv;未触碰系统环境(无系统 Python/无系统 ROCm,零破坏面)。
- AMD 三件套 + gfx1100 设备栈(amd-torch-device-gfx1100 等 8 组件)从 repo.amd.com 安装,`pip check` 全程无冲突,课程依赖安装经 constraints 保护未发生替换。
- 专用 Jupyter Kernel `yolo26-rocm7141`(argv 实证指向 .venv python),nbclient 强制使用该 kernel 执行。
- HIP 真实计算证据(GPU_PHYSICAL_SMOKE.md,17/17 门禁):HIP=7.14.60850;GPU 张量分配、matmul、Conv2d、BatchNorm 训练模式、loss、**backward、optimizer step(max|Δw|=7.965e-05)**、同步、NaN/Inf 筛查全部通过。`torch.version.cuda=None` 为 HIP-only wheel 正常现象;`torch.cuda` API 即 HIP 兼容层,不代表 NVIDIA CUDA。

## C. 六份 Notebook 验收矩阵

| NB | 结果 | 耗时 | 关键输出 |
|---|---|---|---|
| 00 环境预检 | PASS | 10.21s | environment.json;yolo26n.pt 冻结缓存 sha256 `9b09cc8b…`(真实下载) |
| 01 Lab A 推理阈值 | PASS | 14.19s | 真实 GPU 推理:conf 0.10/0.25/0.50 → **7/5/5** 检测(bus/person/stop sign),单调性断言过;3 图+JSON |
| 02 Lab B 数据 QA | PASS | 6.64s | COCO128→三类教学集 train **89**/val **20**/test **19**,QA 零问题,人工审核清单 |
| 03 Lab C 训练 | PASS | 200.19s | COCO8 GPU 冒烟 1Epoch + **baseline 12 Epoch 真训练**;best/last.pt、results.csv(12 行)、results.png |
| 04 Lab D 评测 | PASS | 30.51s | 官方 val + 固定阈值 TP18/FP6/FN11 + 失败案例图 + PR/混淆矩阵 |
| 05 Lab E A/B | PASS | 89.56s | Variant(imgsz=416)12 Epoch 真训练 + 同口径双臂评测 + 图表报告 |

证据:`validation/NOTEBOOK_EXECUTION_MATRIX.md`(自动生成数据 `NOTEBOOK_EXECUTION_DATA_20261102_170901.json`)、执行后副本 `validation/executed/20261102_170901/`(6 份含输出 Notebook + 逐 cell 日志)。

## D. 真实训练结果(Baseline)

- 权重:`yolo_course/runs/lab_c/baseline/weights/best.pt`
  **SHA256 `1c058d8d1cc076c35f01e20f427acc6874fab0eab073f529a20f5d8093ae6f8b`**
- 配置:12 Epoch · imgsz=512 · batch=8 · SGD · seed=42 · amp=False · workers=0(课程默认,未改动)
- 验证集(同轮 02 划分,val=20 张,评测 imgsz=640):
  - **Precision 0.8380 · Recall 0.5517 · mAP50 0.6245 · mAP50-95 0.3795**
- 训练耗时 79.68 s;峰值分配显存 1.546 GiB(PyTorch allocated 口径)
- 训练曲线:results.csv epoch1→12(mAP50-95 0.0→0.300 训练态)与 results.png 齐备
- 教学口径声明:128 张 COCO128 子集三类划分的课堂练习,非 COCO 基准成绩。

## E. A/B 优化结果(受控实验)

| 臂 | 唯一变量 | mAP50 | mAP50-95 | Precision | Recall | 训练耗时 | 峰值分配 |
|---|---|---|---|---|---|---|---|
| Baseline | imgsz=512 | 0.6245 | **0.3795** | 0.8380 | 0.5517 | 79.68 s | 1.546 GiB |
| Variant | **imgsz=416** | 0.6407 | **0.3381** | 0.7314 | 0.5517 | 85.81 s | 1.066 GiB |

- 改动集合断言 = `{imgsz, project, name}`(Notebook 内置检查过);初始权重同源同 SHA;评测同 val 集、同 imgsz=640、同配置。
- **ΔmAP50-95 = −0.0414:Variant 更差,忠实报告**(不采纳)。mAP50 略升(+0.016)与 mAP50-95 下降并存——低分辨率削弱定位精度,符合教学预期解释。
- 耗时:Variant 反而慢 6.1 s(小数据集 + 首轮 kernel/JIT 与缓存噪声主导,非 GPU 算力瓶颈);显存峰值如预期下降(−31%)。
- 局限:单种子、微数据集(val=20)、两次训练的运行时噪声;结论不可外推为"imgsz=512 一定更优",仅证明本配置下 512 更好。
- 独立 Test 门禁保留:`RUN_FINAL_TEST=False`,Test 集未被触碰(05 输出 "Test is sealed")。

## F. 修改清单(全部差异)

| 文件 | 类型 | 理由 |
|---|---|---|
| `course/lab_utils.py` | 修改(1 处,POSIX 门控) | Windows 上 `Path('/workspace')` 为盘符相对路径,防止输出目录劫持;Linux 行为逐字不变 |
| `scripts/miopen_shims/`(4 文件)+ `install_miopen_shims.py` | 新增 | MIOpen hipRTC 宿主 STL 缺失的本地修复(等价上游 #13437),HIP 守护、仅设备编译生效、可整体移除 |
| `scripts/gpu_smoke.py` / `bn_repro.py` | 新增 | GPU 准入门禁与缺陷复现器(证据生成) |
| `scripts/run_all_labs.py` / `validate_windows.ps1` | 新增 | 自动化验证入口(nbclient 顺序执行/产物校验/矩阵数据) |
| `tests/test_windows_compat.py` | 新增 | +10 回归:路径/HIP 伪装防线/A/B 纯度/Notebook 结构/分隔符/yaml roundtrip |
| `pyproject.toml` / `constraints-rocm.txt` | 新增 | 元数据 + AMD 源声明 + 三件套硬约束 |
| `docs/ROCM_WINDOWS_SETUP.md` / `RADEON_CLOUD_LINUX_SETUP.md` / `TROUBLESHOOTING.md` | 新增 | 双平台部署与排障文档 |
| `validation/*` | 新增 | 本轮全部证据(基线/日志/矩阵/报告) |
| 6 份 Notebook / build_notebooks.py / requirements-course.txt | **零改动** | 原设计已兼容;教学目标/练习/断言完整保留 |

环境侧变更(可逆):用户级环境变量 `HIPRTC_COMPILE_OPTIONS_APPEND`(撤销命令见 DEPENDENCY_MANIFEST §5);venv 内 sdk include 增加 4 个 shim 副本。
回归:**pytest 16/16 通过**(原 6 + 新 10),`pip check` 无冲突,三件套版本未漂移。

## G. Radeon Cloud(Linux)部署就绪情况

`Linux Radeon Cloud = NOT_RUN / STATIC_COMPATIBILITY_CHECKED`

- 已核查:POSIX 分支与 /workspace 约定保留、YOLO_COURSE_HOME/YOLO_USER_DATA_YAML 可配置、Linux 同构安装命令、kernel 创建、gfx 适配说明(按 GPU 换 device extra)、Windows 改动对 Linux 零影响(MIOpen shim 仅 Windows 激活)。
- 未做:真实 Linux GPU 会话执行(本轮无该环境)。**Windows PASS ≠ Linux PASS**;上线路径与命令见 `docs/RADEON_CLOUD_LINUX_SETUP.md`。

## H. Remaining Blockers

无 P0 阻塞。遗留事项(低优先级):

| 项 | 优先级 | 说明 / 下一步 |
|---|---|---|
| Linux Radeon Cloud 真机验证 | 中 | 在云环境跑 `gpu_smoke.py`+`run_all_labs.py` 取真实证据后宣布 Linux GO |
| MIOpen wheel 修复跟踪 | 低 | AMD 发布含 #13437 的正式 wheel 后,移除 shim(删除目录+注册表项),重跑全量验证 |
| uv × AMD 索引兼容 | 低 | 等 uv 修复 `nul` 保留名解析后可切换真 uv.lock(现 constraints+freeze 方案已可复现) |
| GitHub 直链慢 | 低 | 已验证长超时可下载;课堂建议预置缓存(README 官方方案) |

## I. GO/NO-GO 条件核对(12/12)

1. ✅ uv 环境创建(Python 3.12.16,隔离验证)
2. ✅ PyTorch 三件套版本完全匹配(2.12.0/0.27.0/2.11.0+rocm7.14.1)
3. ✅ ROCm HIP 后端确认(7.14.60850,真实计算)
4. ✅ gfx1100 物理 GPU 确认(RX 7900 XTX,gcnArchName 实测)
5. ✅ GPU Forward/Backward/Optimizer Step 成功(17/17 门禁)
6. ✅ 六份 Notebook 必需 Cell 实际执行成功(37/37)
7. ✅ Train/Val/Test 隔离检查通过(89/20/19,QA 零泄漏)
8. ✅ YOLO26n 完整教学训练成功(12 Epoch 真训练)
9. ✅ best.pt/训练曲线/验证指标真实生成(SHA256 已存证)
10. ✅ A/B 实验完成且可追溯(双臂 SHA256+同口径评测+差异自动计算)
11. ✅ 全部回归测试通过(pytest 16/16 + pip check)
12. ✅ 未通过修改 Notebook 绕过任何关键检测(Notebook 零改动;ROCm 门禁原样生效)

**`DAY-1 LABS WINDOWS ROCm = GO`**
