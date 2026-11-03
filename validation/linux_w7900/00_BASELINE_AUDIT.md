# 00 · Baseline Audit — Linux W7900 Full Validation

- 审计时间（UTC）：2026-11-03 02:05–02:30
- 审计人：独立验收代理（本任务执行者）
- 审计对象：`AIwork4me/yolo26-rocm-day1-labs` main 分支最新提交
- 结论速览：仓库结构完整、平台标注诚实（Windows 证据未被冒充 Linux 证据）、忽略规则完善；**main 上存在 1 项 P0 测试失败（AST 参考）**，已在独立分支修复（见 07_DEFECTS_AND_FIXES.md）。

## 1. Git 基线

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/AIwork4me/yolo26-rocm-day1-labs（public） |
| main HEAD（上游） | `205e7006f55589cfc9115dd95d26e60158bc84db`（2026-11-03T00:41:27Z，"Revise audience description in README"） |
| 本地快照 Tree SHA | `ba86a3536fa6669d8677dac0eacd8291e3f43f8d` |
| 本地验证分支 | `linux-w7900-validation-fixes`（自快照 `fbc6908` 起修复提交 `8fc7cb8`） |
| 工作区状态 | clean（修复提交后无未跟踪/未提交文件；`yolo26n.pt`、`.venv/`、`yolo_course*/` 均被 .gitignore 覆盖） |
| 获取方式 | 本机出口代理阻断 github.com git 协议（CONNECT 503），经 `api.github.com` tarball API 获取字节一致的 main 快照后 `git init` 重建；上游 SHA 与 API 返回一致 |

> 说明：因 git 协议被代理阻断，本轮以 tarball 快照 + 本地分支方式工作；未执行任何 push/PR（遵守规则 12）。

## 2. 操作系统与硬件

| 项 | 值 |
|---|---|
| OS | Ubuntu 24.04.4 LTS（Noble Numbat） |
| Kernel | 6.8.0-79-generic x86_64 |
| CPU | AMD EPYC 9334 32-Core Processor（128 逻辑核） |
| 系统内存 | 503 GiB（可用 432 GiB） |
| GPU（实测） | **AMD Radeon Pro W7900D**（`torch.cuda.get_device_name(0)`、rocm-smi Card Model 0x744b / SKU D7070910） |
| GPU 架构（实测） | **gfx1100**（`rocminfo`：`amdgcn-amd-amdhsa--gfx1100`；torch `gcnArchName=gfx1100`） |
| GPU 显存（实测） | 51,522,830,336 B ≈ **48 GiB**（W7900 规格） |
| GPU 设备节点 | `/dev/kfd`、`/dev/dri/renderD129` 均存在；容器内可见 1 块 GPU（lspci 显示宿主机 8 块同型号卡，本会话仅分配 1 块） |
| 内核驱动 | amdgpu（ROCk module 6.16.13，PCI_ID 1002:744B） |
| 验收时 GPU 占用 | 空闲（GFX_ACTIVITY 0%，VRAM 已用 28 MB） |
| 磁盘 | `/workspace` 98 G（可用 87 G） |

## 3. 软件栈

### 3.1 平台预装 ROCm 栈（审计对象，最终未采用为课程执行栈）

| 组件 | 版本 |
|---|---|
| torch | 2.9.1+gitff65f5b（HIP 7.2.53211-e1a6bc5663） |
| torchvision | 0.24.1+d801a34 — **实测 `torchvision.ops.nms` GPU 内核缺失**（`NotImplementedError: Could not run 'torchvision::nms' with arguments from the 'CUDA' backend`） |
| 结论 | 与 `docs/RADEON_CLOUD_LINUX_SETUP.md` §2A 记录的已知限制**完全一致**（该文档明确此镜像组合不可走方式 A）。平台环境未被修改。 |

### 3.2 课程独立执行栈（uv 隔离，方式 B，本轮实际采用）

| 组件 | 版本 | 来源 |
|---|---|---|
| Python | 3.12（uv 管理，`.venv`） | uv 0.10.9 |
| torch | **2.12.0+rocm7.14.1**（HIP 7.14.60850） | repo.amd.com/rocm/whl-multi-arch（`torch[device-gfx1100]`） |
| torchvision | **0.27.0+rocm7.14.1**（GPU NMS 实测可用） | 同上 |
| torchaudio | 2.11.0+rocm7.14.1 | 同上 |
| ultralytics | 8.4.199 | PyPI（constraints-rocm.txt 保护三件套不被替换） |
| numpy / matplotlib / pandas / pillow / pyyaml | 2.2.6 / 3.10.x / 2.3.x / 11.x / 6.0.x | PyPI（清华镜像） |
| jupyter 内核 | `yolo26-rocm7141`（registered，argv→项目 `.venv/bin/python`） | ipykernel |
| `pip check` | 无冲突（rc=0） | — |

- 安装命令与 `docs/RADEON_CLOUD_LINUX_SETUP.md` §2B 逐字一致；未安装任何 PyPI 默认 CPU/CUDA PyTorch（constraints 生效，`pip check` 通过）。
- 平台 `/opt/venv`、`/opt/rocm` 未做任何改动（规则 3 遵守）。

## 4. 网络与代理

- 代理：HTTP/HTTPS 出口代理已配置（`notebook-egress-proxy.<internal>.svc:3128`，URL 无凭据）。已按任务要求不记录任何密钥。
- **允许**：`api.github.com`、`codeload.github.com`、`repo.amd.com`、`pypi.tuna.tsinghua.edu.cn`、`hf-mirror.com`、`ultralytics.com/assets`（字体）。
- **阻断**：`github.com`（git 协议与 release 资产）、`release-assets.githubusercontent.com`、`objects.githubusercontent.com`、`raw.githubusercontent.com`、`ultralytics.com/images/`。

### 受限网络的资产预置（全程 SHA256 存证）

由于 release 资产域名被阻断，正式运行所需的官方资产经 hf-mirror 镜像获取，并与仓库 `validation/DEPENDENCY_MANIFEST.md` 登记的官方 SHA256 逐一比对：

| 资产 | 大小 | SHA256（实测 = 官方登记值） | 官方来源 |
|---|---|---|---|
| yolo26n.pt | 5,544,453 B | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` ✓ | ultralytics/assets v8.4.0 |
| coco128.zip | 6,983,030 B | `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e` ✓ | ultralytics/assets v0.0.0 |
| coco8（16 文件） | 官方 Ultralytics/COCO8 数据集（HF）逐文件获取 | —（Lab C 冒烟数据，非基准） | ultralytics 官方 HF 数据集 |

> 判定：字节级等于官方发布，预置不构成"非官方数据"；网络限制属于平台基础设施约束，非仓库缺陷，已在最终报告披露。

## 5. 仓库完整性检查

| 检查项 | 结果 |
|---|---|
| 六份 Notebook 存在 | ✓（00/01/02/03/04/05，共 37 个 code cells：5/7/7/6/6/6） |
| 生成器单一事实源 | ✓ `build_notebooks.py`（772 行，生成 6 份 Notebook） |
| 共享库 | ✓ `course/lab_utils.py`（442 行）、`course/annotation_import.py`（159 行，X-AnyLabeling 导入器） |
| 脚本 | ✓ `gpu_smoke.py`、`run_all_labs.py`、`install_miopen_shims.py`、`bn_repro.py`、`audit_code_style.py`、`validate_windows.ps1`（Windows） |
| 单元测试 | ✓ 6 个测试文件；main 基线收集 21 项：**19 通过 / 1 失败（AST 参考，P0-A）/ 1 跳过（Windows 专用）** |
| 讲师指南 | ✓ `INSTRUCTOR_GUIDE.md` + `.en.md` |
| 数据导入程序 | ✓（Lab B 双路径：COCO128 fallback + X-AnyLabeling 导入器 + 异常注入测试） |
| MIOpen shim | ✓（`scripts/miopen_shims/`，仅 win32 生效路径） |

### 5.1 Linux / Windows 代码路径隔离（实测确认）

- `course/lab_utils.py::root()`：`YOLO_COURSE_HOME` > POSIX `/workspace` 检测 > cwd 本地；Windows 不会被 `C:\workspace` 劫持（测试 `test_root_workspace_only_on_posix` 覆盖）。
- `scripts/run_all_labs.py`：venv 解释器路径 `Scripts/python.exe`（nt）vs `bin/python`（POSIX）；MIOpen shim 环境变量仅 `sys.platform == 'win32'` 注入。
- `scripts/gpu_smoke.py`：驱动信息收集 POSIX 走 `rocm-smi`/sysfs，Windows 走 PowerShell；shim 注入同样仅 win32。
- Linux 路径全链路无 PowerShell 依赖。✓

### 5.2 历史证据平台标注（无 Windows 证据冒充 Linux）

- `README.md` 徽章与"已验证状态"表明确标注 **Windows 11 原生 / RX 7900 XTX**；
- Linux 路径明确标注"静态兼容核查，尚未在真实云 GPU 复验"；
- `validation/NOTEBOOK_EXECUTION_MATRIX.md` 数据源为 Windows 执行 JSON，环境列标注 Windows 11 + Kernel yolo26-rocm7141。
- `docs/RADEON_CLOUD_LINUX_SETUP.md` §6 明示 "Windows PASS ≠ Linux PASS"。
- **判定：诚实。** 本轮 Linux 证据全部新生成于 `validation/linux_w7900/`，不引用历史 Windows 文件。

### 5.3 .gitignore 覆盖

模型权重（`*.pt`/`*.onnx`）、数据集（`coco128/`、`coco8/`、`*.zip`）、课程输出（`yolo_course/`、`yolo_course_*/`、`runs/`）、Jupyter/OS 工件均已覆盖。学生图片与个人信息按 README/Lab B 指南不入库（manual review CSV 生成在 course home，不入 git）。✓

## 6. main 基线缺陷（移交 Phase 1 修复）

| ID | 级别 | 摘要 |
|---|---|---|
| P0-A | P0 | `validation/notebook_code_asts.json` Lab B 参考损坏：cell 1 为陈旧摘要（X-AnyLabeling 变更后未再生）、cell 5 为 54 字符截断摘要（手补丁）；`test_notebook_code_matches_ast_reference` 在 main 上失败 |
| P0-B | P0 | `scripts/run_all_labs.py` Lab B 工件检查硬编码 COCO128 路径，X-AnyLabeling / 用户 YAML 路径会假失败，旧残留会假通过 |
| P0-C | P0 | 运行器 Kernel 回退 `python3` 时不验证解释器归属（本机平台栈 torchvision 缺 GPU NMS，恰是该风险的现实例证） |
| P1-A | P1 | `build_notebooks.py` 五个 `make()` 调用点英文时长/学习路径参数交换（00/01/03/04/05 的中文"学习路径"显示英文时长） |

## 7. 基线判定

- 仓库可审计、可复现：**PASS**
- main 分支绿色：**FAIL（P0-A 测试失败）**——已在本轮 `linux-w7900-validation-fixes` 分支修复并回归（37 通过/1 跳过）
- 环境（Linux W7900 + uv 隔离 ROCm 7.14.1 栈）：**PASS**（准入证据见 01_GPU_PHYSICAL_VALIDATION.md）
