# 依赖清单与复现方式(DEPENDENCY_MANIFEST)

生成:2026-11-02 · 环境:Windows 11 原生 · Python 3.12.16(uv 管理,项目内 `.venv`)

## 1. 关键版本(本轮实测安装,精确锁定)

| 包 | 版本 | 来源 |
|---|---|---|
| torch | **2.12.0+rocm7.14.1** | https://repo.amd.com/rocm/whl-multi-arch/ |
| torchvision | **0.27.0+rocm7.14.1** | 同上 |
| torchaudio | **2.11.0+rocm7.14.1** | 同上 |
| amd-torch-device-gfx1100 / amd-torch-device-gfx110x | 2.12.0+rocm7.14.1 | 同上(device extra) |
| amd-torchvision-device-gfx1100 | 0.27.0+rocm7.14.1 | 同上 |
| rocm / rocm-sdk-core / rocm-sdk-libraries / rocm-sdk-device-gfx1100 | 7.14.1 | 同上 |
| rocm-bootstrap | 0.1.0 | 同上 |
| ultralytics | 8.4.198 | PyPI |
| ultralytics-platform / ultralytics-thop | 0.1.102 / 2.2.4 | PyPI |
| numpy / pillow / matplotlib / pandas / pyyaml | 2.4.4 / 12.2.0 / 3.11.2 / 3.1.0 / 6.0.3 | PyPI |
| opencv-python | 5.0.0.93 | PyPI |
| ipykernel / jupyterlab / nbclient / nbformat / pytest | 7.4.0 / 4.6.5 / 0.11.0 / 5.11.1 / 9.2.0 | PyPI |

完整 126 项:`validation/logs/pip_freeze_full.txt`(及 `pip_list_freeze.txt`)。

## 2. 复现步骤(Windows;Linux 同构,见 docs/RADEON_CLOUD_LINUX_SETUP.md)

```powershell
uv python install 3.12
uv venv --python 3.12 --seed .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" "torchvision[device-gfx1100]==0.27.0+rocm7.14.1" `
  "torchaudio==2.11.0+rocm7.14.1"
uv pip install --python .\.venv\Scripts\python.exe --constraints constraints-rocm.txt `
  -r requirements-course.txt nbclient jupyterlab pytest
.\.venv\Scripts\python.exe -m ipykernel install --user --name yolo26-rocm7141 `
  --display-name "YOLO26 - AMD ROCm 7.14.1"
.\.venv\Scripts\python.exe scripts\install_miopen_shims.py --setx   # MIOpen JIT 修复(见下)
```

## 3. 锁定策略说明(为何不是 uv.lock)

- 本仓库**未使用 uv.lock**:`uv lock` 在 Windows 上解析 AMD 索引时报
  `failed to query metadata of symlink 'nul': 函数不正确`(AMD 索引页面含 Windows
  保留设备名条目,uv 0.13.6 会将其当符号链接查询;pip 不受影响)。已实测并留档。
- 替代锁定(任务预案允许):
  - `constraints-rocm.txt`:三件套 `==...+rocm7.14.1` 硬约束,任何后续
    `uv pip install --constraints` 都无法将其漂移到 PyPI 版本;
  - `pyproject.toml`:声明 `rocm-gfx1100` extra(同样精确版本)+ `[tool.uv.sources]`
    指向 AMD 索引(供支持的平台使用);
  - `validation/logs/pip_freeze_full.txt`:本轮全量 freeze,逐项可对账。
- 重装校验:`pip check` 通过 + `pip show torch/torchvision/torchaudio` 三版本精确匹配。

## 4. 数据与模型工件 SHA256(本轮真实下载)

| 文件 | SHA256 | 来源 |
|---|---|---|
| yolo26n.pt | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | github.com/ultralytics/assets v8.4.0(200, 5,544,453 B) |
| coco128.zip | `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e` | github.com/ultralytics/assets v0.0.0(200, 6,983,030 B) |
| 训练产物 best.pt / A/B 权重 | 见 FINAL_REPORT(D 节) | 本轮 GPU 训练生成 |

## 5. MIOpen JIT 修复工件

| 文件 | 说明 |
|---|---|
| `scripts/miopen_shims/type_traits` | freestanding `<type_traits>`(HIP 守护,仅设备编译生效) |
| `scripts/miopen_shims/utility` / `initializer_list` / `cstdint` | 同上,对应上游 #13437 涉及的宿主头 |
| `scripts/install_miopen_shims.py` | 部署器:复制 shim 到 venv sdk include + setx 用户级 `HIPRTC_COMPILE_OPTIONS_APPEND` |
| 用户级环境变量 | `HIPRTC_COMPILE_OPTIONS_APPEND=-I<项目>\scripts\miopen_shims`(撤销:reg delete HKCU\Environment /V HIPRTC_COMPILE_OPTIONS_APPEND /F) |
