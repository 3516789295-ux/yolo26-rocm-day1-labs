# GPU 物理准入冒烟报告(GPU_PHYSICAL_SMOKE)

- 时间:2026-11-02(本地)· 复核轮次:第 2 轮(第 1 轮发现 MIOpen 缺陷,修复后复测全过)
- 证据:`validation/logs/gpu_smoke.json` · 复现入口:`.venv\Scripts\python.exe scripts\gpu_smoke.py`
- 准入结论:**PASS —— 允许进入训练验证阶段**

## 1. 版本与后端

| 检查 | 结果 | 值 |
|---|---|---|
| torch 精确版本 | PASS | `2.12.0+rocm7.14.1` |
| torchvision 精确版本 | PASS | `0.27.0+rocm7.14.1` |
| torchaudio 精确版本 | PASS | `2.11.0+rocm7.14.1` |
| HIP 后端 | PASS | `torch.version.hip = 7.14.60850` |
| torch.version.cuda | 说明 | `None`(HIP-only Windows wheel 的正常现象;`torch.cuda` 命名空间即 HIP API 兼容层,**不代表 NVIDIA CUDA**) |
| GPU 可见 | PASS | `torch.cuda.is_available() = True` |

## 2. 设备

| 项 | 值 |
|---|---|
| 设备名 | AMD Radeon RX 7900 XTX |
| GCN 架构 | **gfx1100**(`gcnArchName`,与目标 wheel `device-gfx1100` 匹配) |
| capability | 11.0(major.minor = 11.0,RDNA3) |
| 总显存 | 23.99 GiB |
| 计算单元 | 96 |
| 驱动 | 32.0.21030.2001(2025-09-25,Win32_VideoController) |

## 3. 真实计算管线(Forward + Backward + Optimizer Step)

| 步骤 | 结果 | 实测 |
|---|---|---|
| GPU 张量分配 | PASS | `device='cuda:0'`(HIP 报告为 cuda 设备类型,预期行为) |
| GPU 矩阵乘法 64×128 @ 128×128 | PASS | 224.1 ms(首轮含 kernel JIT) |
| Conv2d(3→16, 8×3×64×64)前向 | PASS | 570.8 ms |
| BatchNorm2d 训练模式前向 | PASS | `num_batches_tracked=1`,running stats 更新 |
| Loss(MSE,GPU) | PASS | `loss=0.184226`,有限值 |
| `loss.backward()` | PASS | 11104 ms(首轮含 MIOpen 反向 kernel 编译);`weight.grad` 在 GPU |
| `optimizer.step()`(SGD lr=0.1 momentum=0.9) | PASS | `max|Δw| = 7.965e-05`(权重真实变化) |
| `torch.cuda.synchronize()` | PASS | 全程显式同步 |
| NaN/Inf 筛查(y/out/loss/grad/weight) | PASS | 全部有限 |
| 峰值分配显存 | 0.0845 GiB(PyTorch allocated 口径,非整卡占用) |

## 4. 第 1 轮失败与根因(保留证据)

首轮 `real_compute_pipeline` **FAIL**:`RuntimeError: miopenStatusUnknownError`

```
MIOpen(HIP): Error [Compile] hiprtcCompileProgram(...) MIOpenBatchNormFwdTrainSpatial.cpp:
HIPRTC_ERROR_COMPILATION (6)
In file included from ...\include\vector_types.hpp:8:
...\include\miopen_type_traits.hpp:151:10: fatal error: 'type_traits' file not found
MIOpen Error: ... hipoc_program.cpp:299: Code object build failed.
```

- **根因**:rocm-sdk-libraries 7.14.1 内 MIOpen 在 HIP≥7.0 时无条件 `#include <type_traits>`
  (宿主 STL),而 hipRTC 设备编译在隔离选项下无任何宿主 C++ 标准库 → BatchNorm/Conv 训练
  kernel 全部编译失败 → Windows 上无法训练(同 [ROCm/MIOpen#3956](https://github.com/ROCm/MIOpen/issues/3956),
  修复 = [ROCm/rocm-libraries#13437](https://github.com/ROCm/rocm-libraries/pull/13437),2026-10-19 合入,
  但 AMD 源上的 7.14.1 wheel(索引确认无更新版)未含该修复)。
- **本地方案(版本零改动)**:hipRTC 官方扩展点
  `HIPRTC_COMPILE_OPTIONS_APPEND=-I<项目>/scripts/miopen_shims` 注入 4 个完全自包含的
  freestanding 头(`<type_traits>` `<utility>` `<initializer_list>` `<cstdint>`,
  仅在 `__HIP__/__HIPRTC__` 编译单元生效),等价复刻上游"kernel 头自包含"修复。
  安装入口:`scripts/install_miopen_shims.py`(部署 + setx 用户级持久化)。
- **验证**:同一复现器修复前 FAIL(`'type_traits' file not found`)→ 修复后
  `BatchNorm2d forward OK in 0.5s / backward OK`(validation/logs/bn_repro_*.log)。
- 排查中间证据:hipRTC 搜索路径实验(sdk include 根不受搜,`-nostdinc` 隔离,见 shim1/2 日志);
  错误逐步推进 type_traits→stddef.h→shim 内部排序→全过,每步日志均在 validation/logs/。

## 5. 准入判定

- GPU 设备(RX 7900 XTX / gfx1100)✓
- HIP 后端(7.14.60850)✓
- 实际 Forward+Backward+Optimizer Step 于 AMD GPU 完成 ✓
- **门禁通过,准予执行 Notebook 训练验证(Phase 6)。**
