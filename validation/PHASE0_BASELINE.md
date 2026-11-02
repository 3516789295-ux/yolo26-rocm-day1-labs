# Phase 0 — 仓库审计与环境识别基线

生成时间:2026-11-02(本地) | 审计人:ZCode 自动化验证代理
项目根目录:`C:\Users\Rocm\Desktop\YOLO\YOLO26_ROCm_Day1_Labs`

## 1. 仓库结构审计

```
YOLO26_ROCm_Day1_Labs/
├── README.md                     # 课程说明(面向 Linux Radeon Cloud /workspace)
├── requirements-course.txt       # ultralytics>=8.4,<9 等课程依赖(假定 ROCm torch 已预装)
├── build_notebooks.py            # Notebook 生成器(重建会覆盖 notebooks/,人工修复需同步此处)
├── DESIGN_SYSTEM.md / INSTRUCTOR_GUIDE.md / PACKAGE_MANIFEST.json / SHA256SUMS
├── course/
│   ├── __init__.py
│   └── lab_utils.py              # 数据/QA/评测/路径工具(核心共享库)
├── notebooks/                    # 6 份 Notebook,全部存在 ✓
│   ├── 00_ROCm_Environment_Preflight.ipynb
│   ├── 01_Lab_A_Predict_and_Threshold.ipynb
│   ├── 02_Lab_B_Dataset_Annotation_QA.ipynb
│   ├── 03_Lab_C_ROCm_Training_Finetune.ipynb
│   ├── 04_Lab_D_Evaluation_Error_Analysis.ipynb
│   └── 05_Lab_E_Controlled_AB_Optimization.ipynb
└── tests/
    ├── test_lab_utils.py         # 离线 CI:标签/IoU/数据集构建/QA/泄漏检测
    └── test_prediction_matching.py  # 固定阈值 TP/FP/FN 匹配逻辑
```

- Git 仓库:**不存在**(`git status` → fatal: not a git repository)。无历史可回滚,原始状态以本文件 SHA256 快照为准。
- 无已有输出/模型工件:notebooks 均为未执行状态(`execution_count: null`),项目内无 runs/、yolo_course/、reports/。
- `PACKAGE_MANIFEST.json` 声明 `gpu_executed_in_generation_environment: false`,`offline_unit_tests: 6 passed`(生成时离线测试通过,无 GPU 执行)。

## 2. Notebook 依赖与执行顺序分析(全部 Cell 已通读)

| NB | 依赖前序 | 关键动作 | 外部下载 | 副作用 |
|---|---|---|---|---|
| 00 | 无 | 环境识别、HIP 断言、GPU 矩阵乘、权重缓存 | `yolo26n.pt`(GitHub v8.4.0) | `reports/environment.json`、`artifacts/yolo26n.pt` |
| 01 | 00 | GPU 推理 ×3 阈值、IoU 练习 | `bus.jpg`(ultralytics.com,或用包内 assets) | 3 张预测图 + JSON |
| 02 | 无(0 GPU 可跑) | COCO128 下载→3 类教学集划分→QA | `coco128.zip`(GitHub v0.0.0) | `data/processed/.../data.yaml` 等 |
| 03 | 02(需 dataset_yaml);00(需权重) | COCO8 冒烟 1 epoch + baseline 12 epoch 训练 | COCO8(ultralytics 自动下载,GitHub) | `runs/lab_c/{smoke,baseline}`、`best.pt` |
| 04 | 03(需 best.pt) | 官方 val + 固定阈值匹配 + 失败案例 | 无 | PR 曲线、混淆矩阵、errors.csv |
| 05 | 03(baseline + train_settings) | Variant 只改 imgsz=416 重训 + 同口径评测 | 无 | `runs/lab_e/imgsz416`、A/B CSV+图 |

配置传递机制:`course_config.json`(由 `save_config` 写入 `home`),跨 Notebook 传递 dataset_yaml / baseline_weights / baseline_train_settings / model_sha256。
**顺序 00→01→02→03→04→05 为硬依赖顺序**(03 依赖 00 的权重缓存与 02 的数据集;04/05 依赖 03 的 best.pt)。

## 3. 系统环境检测(证据)

| 项 | 值 | 来源 |
|---|---|---|
| OS | Windows 11 26H2, Build 26300 (10.0.26300) | Win32_OperatingSystem |
| GPU(主) | **AMD Radeon RX 7900 XTX**,PCI VEN_1002&DEV_744C(=gfx1100/Navi31,24GB) | Win32_VideoController |
| GPU 驱动 | 32.0.21030.2001(2025-09-25,Adrenalin ≥25.x) | Win32_VideoController |
| GPU(其他) | Intel UHD 770(核显)、GameViewer 虚拟显卡 | Win32_VideoController |
| 内存 | 31.8 GB | Win32_ComputerSystem |
| 磁盘 C: | 931G 总量,645G 可用 | df |
| Python(系统) | 无(仅 WindowsApps 存根)→ 必须由 uv 安装 | where python |
| uv | 0.13.6 (329bf9448 2026-10-26) | uv --version |
| 系统级 ROCm | 未安装 `C:\Program Files\AMD\ROCm`;无 amd-smi/rocm-smi → 依赖 wheel 自带运行时 + 驱动 | 目录扫描 |
| /dev/kfd, /dev/dri | 不存在(Windows 预期;Notebook 仅探测不依赖) | ls |
| 代理环境变量 | 未设置 | env |
| Shell | Git Bash(命令经 powershell.exe 取 WMI 证据) | — |

**gfx 架构判定**:RX 7900 XTX = Navi31 = **gfx1100**,与目标 `torch[device-gfx1100]` 匹配 ✓(DEV_744C 为 Navi31 XT/XTL 的 PCI 设备 ID,rdna3 flagship)。最终以 Phase 4 中 PyTorch 实际枚举的 `torch.cuda.get_device_name/gcnArchName` 为准。

## 4. 网络与外部资源探测

| 端点 | 结果 |
|---|---|
| `https://repo.amd.com/rocm/whl-multi-arch/` | **200 OK**(4.3s)✓ 安装源可用 |
| `https://pypi.org/simple/ultralytics/` | 200 OK(25s,慢但可用) |
| `api.github.com` | 200 OK(0.4s) |
| `github.com .../releases/download/...`(coco128.zip / yolo26n.pt 直链) | **首次探测超时(000)**;objects.githubusercontent.com CDN 本身可达(404 root)→ 需长超时重试,已启动后台下载验证 |
| ultralytics.com/images/bus.jpg | 308(可达,重定向) |
| YOLO26n 权重直址(经 API 确认) | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt` |
| COCO128 直址 | `https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip` |

若 GitHub release 下载最终不可用:README 明确支持课前缓存方案(coco128.zip → `yolo_course/data/raw/coco128.zip`,yolo26n.pt → `yolo_course/artifacts/yolo26n.pt`);已另存网络缓存副本供回填。bus.jpg 可用 ultralytics 包内 `assets/bus.jpg` 替代(Notebook 已内置该优先分支)。

## 5. 关键外部参考(任务指定)核实

任务指定的 https://github.com/ROCm/rocm-libraries/pull/13437 :
- 标题:**MIOpen: make HIPRTC kernel includes self-contained without host STL**,2026-10-19 合入。
- 根因:Windows runtime-only ROCm 栈上,MIOpen hipRTC 编译 BatchNorm 训练 kernel 报 `fatal error: 'type_traits' file not found`,PyTorch 侧表现为 **`miopenStatusUnknownError`,直接导致 Windows 上 Ultralytics YOLO 无法训练**(关联 MIOpen#3956)。
- 修复方式:MIOpen kernel 头文件自包含(不再依赖宿主 STL),验证环境为 Windows ROCm 7.14.0 wheel 栈。
- **对本任务的意义**:目标 wheel `rocm7.14.1` 发布(≥2026-10-20 进入 TheRock)晚于修复合入,应已包含;若 Lab C 训练出现 `miopenStatusUnknownError`/hipRTC 头文件错误,即为该已知问题,须如实 BLOCKED 并记录,不得绕过。

## 6. Windows 兼容性预审(详修在 Phase 5)

已识别要点:
1. `lab_utils.root()`:`YOLO_COURSE_HOME` 优先 → `/workspace` 存在则用 → 否则 `Path.cwd()/yolo_course`。Windows 落第三分支 ✓(需控制 Notebook 启动 cwd)。
2. Notebook SETUP cell 候选 `[Path.cwd(), Path.cwd().parent, Path('/workspace/...')]`:Windows 上前两个命中;`Path('/workspace')` 在 Windows 解析为当前盘根 `\workspace`,不存在即安全跳过 ✓。
3. 训练全部 `workers=0` → 规避 Windows spawn 多进程问题 ✓。
4. `amp=False` 课程默认 → 规避 ROCm AMP 兼容问题 ✓。
5. `require_rocm()` 报错文案提及 `/dev/kfd`、`/dev/dri`(仅文案,非逻辑分支)。
6. `download_coco128` 用 urllib(继承系统代理,无代理时直连)——受 GitHub 可达性影响,见 §4。
7. Ultralytics `coco8.yaml`(Lab C 冒烟)会触发包内自动下载 GitHub 资产,同样受 §4 影响。
8. `build_notebooks.py` 会整体重建 notebooks/ —— 任何对 Notebook 的人工修复必须同步修改生成器,否则回归。

## 7. 原始文件 SHA256 基线快照

完整清单副本:`~/yolo26_baseline_sha256.txt`(与 PACKAGE_MANIFEST.json 中 15 个文件逐一比对一致,原始包未被改动)。摘要(lab_utils 与 notebooks):

```
1ca622045b5e218db97d70817ee550b05bd432dfbe3f3d14333c41251472aef7  course/lab_utils.py
14aff59d4800ac853b813e07add9a8c2ba3753c60e000f73278873a5713d1130  notebooks/00_ROCm_Environment_Preflight.ipynb
f0a91265fa3a5a882ea057fd99b7885f05860e23393952ecb7034551ff6cd84  notebooks/01_Lab_A_Predict_and_Threshold.ipynb
a4517d4856092029108f8c9d3be5f80e6f8577e7a498511a2ca042a5d89d0b82  notebooks/02_Lab_B_Dataset_Annotation_QA.ipynb
d8c9674dc65fb987342413dc1bba7c44a57b098be6cafbf2da19ea2f6382eb3b  notebooks/03_Lab_C_ROCm_Training_Finetune.ipynb
e3146cf56217eb441c1332458800f726e81c83fce90f4f3dfe9144e6d218abee  notebooks/04_Lab_D_Evaluation_Error_Analysis.ipynb
19ce5e689f6a6a9984fb72df7637e28d851ad402de443a7b67adf64e06a8201a  notebooks/05_Lab_E_Controlled_AB_Optimization.ipynb
```

Git 状态:非 Git 仓库,无提交历史;P0 约束"保留原始快照与差异"以本文件 + `~/yolo26_baseline_sha256.txt` + validation/ 目录内记录履行。

## 8. Phase 0 结论

- 六份 Notebook 全部存在,依赖链清晰,可按序执行。
- 目标 GPU = RX 7900 XTX(gfx1100)与任务要求匹配,准予进入 Phase 1。
- 无系统 Python / 系统级 ROCm → 完全隔离的 uv 方案无破坏风险(P0-5 满足)。
- 风险:GitHub release 下载链路不稳定(已缓解:长超时后台下载 + README 官方缓存回填路径)。
