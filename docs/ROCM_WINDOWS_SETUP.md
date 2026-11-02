# Windows 原生 ROCm 环境搭建(YOLO26 Day 1 Labs)

本指南在 **Windows 11 + AMD Radeon RX 7900 XTX(gfx1100)+ uv** 上,从零搭建可运行全部 6 份
Notebook 的隔离环境。全程不需要 WSL / Docker / 虚拟机。

## 0. 前置条件

| 项 | 要求 | 本验证机 |
|---|---|---|
| GPU | RDNA3, gfx1100(RX 7900 XTX/XT) | RX 7900 XTX ✓ |
| 驱动 | AMD Adrenalin ≥ 25.x(2025-09 之后) | 32.0.21030.2001 (2025-09-25) ✓ |
| OS | Windows 11 22H2+ | 11 26H2 build 26300 ✓ |
| Python | 3.12 x64(由 uv 提供,无需系统 Python) | 3.12.16 ✓ |
| uv | ≥ 0.13 | 0.13.6 ✓ |
| 磁盘 | ≥ 15 GB(ROCm wheel 约 4 GB + 数据/权重) | 645 GB 可用 ✓ |

不需要预先安装系统级 ROCm SDK:`torch[device-gfx1100]` extra 会从 AMD 源安装
`rocm-sdk-device-gfx1100` 等运行时组件,wheel 自带 HIP 用户态运行时,内核态接口由 Adrenalin 驱动提供。

## 1. 创建隔离环境(项目根目录执行)

```powershell
uv python install 3.12
uv venv --python 3.12 --seed .venv     # --seed 提供 pip,支持 python -m pip 安装方式
.\.venv\Scripts\python.exe --version   # Python 3.12.16
```

> 在配置好 AMD 源之前,**不要**运行 `uv sync` / `uv pip install torch`,
> 否则解析器会从 PyPI 拉取 CPU/CUDA 版 PyTorch 覆盖 AMD 适配版。

## 2. 安装 AMD ROCm PyTorch(版本不可改动)

```powershell
.\.venv\Scripts\python.exe -m pip install `
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" `
  "torchvision[device-gfx1100]==0.27.0+rocm7.14.1" `
  "torchaudio==2.11.0+rocm7.14.1"
```

`device-gfx1100` extra 会额外安装:
`amd-torch-device-gfx1100` / `amd-torch-device-gfx110x` / `amd-torchvision-device-gfx1100`
`rocm-sdk-device-gfx1100` / `rocm-sdk-libraries` / `rocm-sdk-core` / `rocm-bootstrap` / `rocm`。

安装后校验:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip show torch torchvision torchaudio
.\.venv\Scripts\python.exe scripts\gpu_smoke.py    # 物理 GPU 准入门禁
```

### 已知问题参考(重要,均经本轮实测)

- **Windows 上训练崩溃 `miopenStatusUnknownError` / hipRTC `'type_traits' file not found`**:
  根因是 MIOpen hipRTC kernel 头依赖宿主 STL,上游修复在
  [ROCm/rocm-libraries#13437](https://github.com/ROCm/rocm-libraries/pull/13437)(2026-10-19 合入)。
  **实测 `rocm7.14.1` wheel 仍未包含该修复**(AMD 源无更新版本),表现为 BatchNorm/Conv
  训练 kernel 全部编译失败。本仓库的解法(版本零改动):运行
  `.\.venv\Scripts\python.exe scripts\install_miopen_shims.py --setx`
  注入 freestanding STL shim(详见 `validation/GPU_PHYSICAL_SMOKE.md` §4 与
  `validation/CHANGELOG_VALIDATION.md` §0)。AMD 发布含修复的正式 wheel 后可移除。
- **首次运行慢**:MIOpen 第一次编译/调优 kernel(batchnorm、conv 等)可能耗时数分钟,
  结果会写入用户级缓存(之后复用),不是故障。
- `uv lock` 在 Windows 上对该 AMD 索引存在兼容缺陷(索引含 `nul` 保留名条目,uv 报
  `failed to query metadata of symlink 'nul'`),因此本仓库用 `constraints-rocm.txt` +
  pip freeze 清单锁定环境,而不是 uv.lock(详见 validation/DEPENDENCY_MANIFEST.md)。

## 3. 安装课程依赖(保护 ROCm 三件套)

```powershell
uv pip install --python .\.venv\Scripts\python.exe --constraints constraints-rocm.txt `
  -r requirements-course.txt nbclient jupyterlab pytest
.\.venv\Scripts\python.exe -m pip check    # 三件套版本必须保持 2.12.0/0.27.0/2.11.0+rocm7.14.1
```

## 4. 注册 Jupyter Kernel

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user `
  --name yolo26-rocm7141 --display-name "YOLO26 - AMD ROCm 7.14.1"
```

Kernel 的 `argv` 必须指向本项目 `.venv\Scripts\python.exe`
(位于 `%APPDATA%\jupyter\kernels\yolo26-rocm7141\kernel.json`)。

## 5. 运行

```powershell
# 一键全流程验证(依赖检查 → GPU 冒烟 → 回归测试 → 6 份 Notebook 顺序执行)
powershell -ExecutionPolicy Bypass -File scripts\validate_windows.ps1

# 或分步
.\.venv\Scripts\python.exe scripts\gpu_smoke.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_all_labs.py
```

课程输出默认写入 `<项目根>\yolo_course\`(可设 `YOLO_COURSE_HOME` 覆盖)。
Notebook 的 SETUP cell 会自动从 cwd / 上级目录定位 `course/` 包;从项目根或 notebooks/ 启动均可。

## 6. 离线/弱网预案(GitHub 下载不稳定时)

提前把官方文件放入课程目录(README 官方缓存路径,SHA256 见 validation/DEPENDENCY_MANIFEST.md):

| 文件 | 放置路径 |
|---|---|
| `coco128.zip` | `yolo_course\data\raw\coco128.zip` |
| `yolo26n.pt` | `yolo_course\artifacts\yolo26n.pt` |
| `coco8.zip`(Lab C 冒烟) | Ultralytics 数据目录(或保持联网) |

`bus.jpg` 无需下载:ultralytics wheel 自带 `assets/bus.jpg`,Notebook 优先使用它。

## 7. 环境隔离说明

- 全部组件位于项目内 `.venv`,不触碰系统 Python、注册表或其它项目(P0 约束)。
- 卸载:删除 `.venv/`、`yolo_course/`、`%APPDATA%\jupyter\kernels\yolo26-rocm7141\` 即可。
- AMD 驱动保持原样,无需改动。
