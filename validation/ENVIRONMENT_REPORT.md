# 环境验证报告(ENVIRONMENT_REPORT)

验证时间:2026-11-02 · 平台:Windows 11 原生(无 WSL/VM/Docker)· 项目:`C:\Users\Rocm\Desktop\YOLO\YOLO26_ROCm_Day1_Labs`

## 1. 安装结果

| 组件 | 结果 | 证据 |
|---|---|---|
| uv Python 3.12.16 | ✅ `uv python install 3.12` | `sys.executable` 指向项目 `.venv\Scripts\python.exe` |
| uv venv(--seed) | ✅ pip 26.2.2 seed | `python -m pip --version` |
| torch `2.12.0+rocm7.14.1`(device-gfx1100 extra) | ✅ 精确匹配 | `pip show torch` |
| torchvision `0.27.0+rocm7.14.1` | ✅ 精确匹配 | `pip show torchvision` |
| torchaudio `2.11.0+rocm7.14.1` | ✅ 精确匹配 | `pip show torchaudio` |
| gfx1100 设备栈 | ✅ amd-torch-device-gfx1100 / amd-torch-device-gfx110x / amd-torchvision-device-gfx1100 / rocm-sdk-device-gfx1100 / rocm-sdk-libraries / rocm-sdk-core / rocm-bootstrap / rocm(全部 7.14.1 同源) | phase2_pip_torch.log |
| 课程依赖(ultralytics 8.4.198 等) | ✅ 经 `uv pip install --constraints constraints-rocm.txt` 安装,三件套未被替换 | phase3_uv_course_deps.log + 复核 pip show |
| `pip check` | ✅ No broken requirements | 每阶段复核 |
| Jupyter Kernel `yolo26-rocm7141` | ✅ argv → 项目 .venv python | `%APPDATA%\jupyter\kernels\...\kernel.json` |
| 环境锁定方式 | constraints-rocm.txt + pip freeze 全量清单(见 DEPENDENCY_MANIFEST.md;uv.lock 因 uv 与 AMD 索引在 Windows 的 `nul` 保留名缺陷不可用,任务预案允许) | uv probe 日志 |

未触碰:系统 Python(不存在)、注册表系统项、其它项目环境、AMD 驱动。
用户级新增环境变量(可逆):`HIPRTC_COMPILE_OPTIONS_APPEND=-I<项目>\scripts\miopen_shims`(MIOpen JIT 修复,详见 GPU_PHYSICAL_SMOKE.md §4)。

## 2. HIP 真实计算证据

- `torch.version.hip = 7.14.60850`(非 None → 真 ROCm 构建)。
- `torch.cuda.*` 在 ROCm wheel 上即 HIP API 兼容层;`torch.version.cuda = None` 属 HIP-only Windows wheel 正常现象,不能作为"CUDA 构建"的判据(反向判据才是:hip=None 即非 ROCm)。
- GPU 枚举:`AMD Radeon RX 7900 XTX`,`gcnArchName=gfx1100`(PCI VEN_1002&DEV_744C = Navi31)。
- 真实计算:matmul 224ms / Conv2d 571ms / BatchNorm 训练模式 / loss=0.184 / backward / optimizer step(max|Δw|=7.97e-05)/ 无 NaN-Inf —— 全部在 cuda:0(=HIP 设备)完成,详见 `GPU_PHYSICAL_SMOKE.md` 与 `logs/gpu_smoke.json`。

## 3. GPU Forward/Backward 结果

| 项 | 结果 |
|---|---|
| Forward(Conv+BN+MSE loss,GPU) | ✅(首轮含 MIOpen JIT 编译) |
| Backward(GPU 梯度) | ✅ weight.grad on cuda |
| Optimizer Step | ✅ 权重更新非零 |
| BatchNorm 训练路径(此前被 #13437 缺陷阻断) | ✅ 经 shim 修复后通过(bn_repro 前后对比日志) |

## 4. 已知限制与修复记录

| 问题 | 状态 |
|---|---|
| MIOpen hipRTC 宿主 STL 缺失 → Windows 无法训练(7.14.1 wheel 未含上游 #13437 修复) | 已本地等价修复(HIPRTC_COMPILE_OPTIONS_APPEND + freestanding shims),版本零改动 |
| uv lock 与 AMD 索引不兼容(Windows `nul` 保留名) | 采用 constraints + freeze 清单(任务预案) |
| GitHub release 直链首连超时 | 长超时重试成功;yolo26n.pt/coco128.zip 本轮真实下载,SHA256 已记录 |
| 首次 MIOpen kernel 编译/调优耗时数分钟 | 一次性成本,用户缓存(~/.miopen)复用;非故障 |
