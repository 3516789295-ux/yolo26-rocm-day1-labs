# 01 · GPU Physical Validation — Linux W7900 ROCm Admission

- 执行时间（UTC）：2026-11-03 02:14–02:17
- 执行环境：`.venv`（uv 隔离，Python 3.12.13）· torch 2.12.0+rocm7.14.1 · torchvision 0.27.0+rocm7.14.1 · torchaudio 2.11.0+rocm7.14.1 · HIP 7.14.60850
- 物理设备：**AMD Radeon Pro W7900D · gfx1100 · 47.98 GiB · 48 CUs · amdgpu 驱动 6.16.13（ROCk）**
- 结论：**PASS（17/17 门禁）+ 补充验收 PASS（GPU NMS / batched NMS / 冷热启动）**

## 1. 仓库脚本 `scripts/gpu_smoke.py`（未修改，原样执行）

```
[PASS] torch_version — 2.12.0+rocm7.14.1
[PASS] torchvision_version — 0.27.0+rocm7.14.1
[PASS] torchaudio_version — 2.11.0+rocm7.14.1
[PASS] hip_backend — 7.14.60850
[PASS] cuda_api_note — torch.version.cuda=None (HIP-only wheel 正常；torch.cuda 命名空间即 HIP API)
[PASS] gpu_available
[PASS] device_name — AMD Radeon Pro W7900D
[PASS] gfx_arch_gfx1100 — gfx1100
[PASS] gpu_tensor_alloc — cuda:0
[PASS] gpu_matmul — 337.4 ms（首调用含 MIOpen JIT 编译）
[PASS] gpu_conv2d_forward — 6120.2 ms（首调用含 MIOpen JIT 编译，见 §3 冷/热对比）
[PASS] gpu_loss_finite — loss=0.184226
[PASS] gpu_backward_grad_exists — 1012.7 ms
[PASS] optimizer_step_changed_weights — max|Δw|=7.965e-05
[PASS] no_nan_inf
[PASS] batchnorm_train_mode — num_batches_tracked=1（MIOpen hipRTC BN 训练路径，Linux 无需 Windows shim）
[PASS] real_compute_pipeline
OVERALL: PASS（failed_gates=[]）
```

- 证据文件：`validation/logs/gpu_smoke.json`（本机新生成）
- Windows 专有假设检查：`_driver_info()` POSIX 分支走 `rocm-smi --showdriverversion` + `/sys/class/kfd`，无 PowerShell 依赖；MIOpen shim 注入仅 `win32` 生效。**无 Windows 专有假设残留。**
- 版本门禁行为：POSIX 非严格模式下平台栈差异会被记录（`platform_stack` 字段）——本轮为精确匹配锁定栈，无需该宽限。

## 2. 任务要求的 GPU 验收清单核对

| 要求项 | 证据 |
|---|---|
| GPU 张量分配 | `gpu_tensor_alloc` PASS（cuda:0） |
| GPU 矩阵乘法 | `gpu_matmul` PASS |
| Conv2d Forward | `gpu_conv2d_forward` PASS |
| BatchNorm 训练模式 | `batchnorm_train_mode` PASS（running stats 更新） |
| Loss 计算 | `gpu_loss_finite` PASS（有限值） |
| Backward | `gpu_backward_grad_exists` PASS |
| Optimizer Step | `optimizer_step_changed_weights` PASS（权重实际变化） |
| GPU Synchronization | 显式 `torch.cuda.synchronize()` 多处调用并通过 |
| NaN/Inf 检查 | `no_nan_inf` PASS |
| `torchvision.ops.nms` GPU 算子 | **补充验收 PASS**（见 §3；平台预装栈该项失败，恰证明该检查必要性） |
| PyTorch 为 AMD HIP 构建 | `hip_backend` PASS（torch.version.hip=7.14.60850；torch.version.cuda=None 正常） |
| 物理 W7900 真实性/架构/显存 | device_name=W7900D · gfx1100 · 47.98 GiB（rocm-smi：0x744b / D7070910 / 51,522,830,336 B） |

## 3. 补充验收：GPU NMS 与冷/热启动（任务 Phase 2 要求，gpu_smoke.py 未覆盖 NMS）

脚本：一次性验收程序（结果存 `logs/gpu_supplemental_nms_coldwarm.json`）

| 检查 | 结果 | 明细 |
|---|---|---|
| `torchvision.ops.nms`（GPU） | **PASS** | 3 框 + IoU 0.5 → 保留 `[0, 2]`（正确抑制重叠框） |
| `torchvision.ops.batched_nms`（GPU，Ultralytics 检测路径） | **PASS** | 保留 `[0, 2]` |
| matmul 256×256 冷启动 | 44.9 ms | 含首次 kernel 编译 |
| matmul 256×256 热稳态 | **0.019 ms** | JIT 开销 ≈ 45 ms/进程 |
| Conv2d(3→16, 8×3×64×64) 冷启动 | 5.6 ms | — |
| Conv2d 热稳态 | **0.027 ms** | gpu_smoke 中的 6.1 s 为进程级首次 MIOpen 编译，非稳态算力 |
| VRAM 空闲 | 47.98 GiB 全量可用 | 验收时无其他负载 |

> 教学含义（写入 03_CLASSROOM_CAPACITY_PLAN.md）：首个训练进程需 ~10–60 s MIOpen JIT 预热；同进程后续 epoch 为稳态速度。课堂冷/热启动差异显著，需预留预热时间或课前预热。

## 4. 平台预装栈对照（为何必须 uv 隔离栈）

| 栈 | torch | torchvision | GPU NMS | 判定 |
|---|---|---|---|---|
| 平台预装（/opt/venv） | 2.9.1+gitff65f5b（HIP 7.2.53211） | 0.24.1+d801a34 | **NotImplementedError（CPU-only 编译）** | 不可用于检测类 Lab（与 docs/RADEON_CLOUD_LINUX_SETUP.md §2A 实测限制一致） |
| 课程 uv 隔离栈（本轮采用） | 2.12.0+rocm7.14.1 | 0.27.0+rocm7.14.1 | **PASS** | 与仓库 constraints-rocm.txt 精确一致 |

## 5. 判定

**LINUX W7900 ROCm 物理准入：PASS。** 真实 GPU、HIP 后端、完整训练算子链（Forward/Backward/Optimizer/BN/NMS）全部实测通过；准入后才启动 Phase 3 Notebook 全量执行。
