# AMD Radeon Cloud(Linux)部署指南 — YOLO26 Day 1 Labs

本课程包原生面向 Linux Radeon Cloud Jupyter 环境;Windows 原生路径的加入没有改变 Linux 行为。

## 1. 上传与解压

```bash
cd /workspace
unzip YOLO26_ROCm_Day1_Labs_v1.0.zip
cd YOLO26_ROCm_Day1_Labs
```

`notebooks/` 与 `course/` 必须保持在同一项目目录。

## 2. 环境(两种形态)

### A. 平台已预装 ROCm PyTorch(推荐,零安装)

直接选择平台的 ROCm Kernel 打开 Notebook。**禁止** `pip install torch` / `pip install -U torch`。

### B. 自建 uv 隔离环境(与 Windows 同构)

```bash
uv python install 3.12
uv venv --python 3.12 --seed .venv
.venv/bin/python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" \
  "torchvision[device-gfx1100]==0.27.0+rocm7.14.1" \
  "torchaudio==2.11.0+rocm7.14.1"
uv pip install --python .venv/bin/python --constraints constraints-rocm.txt \
  -r requirements-course.txt nbclient jupyterlab pytest
.venv/bin/python -m ipykernel install --user \
  --name yolo26-rocm7141 --display-name "YOLO26 - AMD ROCm 7.14.1"
```

gfx 架构按实际 GPU 更新 extra(如 MI300 系为 `device-gfx942`,RX 7000 为 `device-gfx1100`,
RX 9060/8060 系为 gfx1151/gfx1201 等,以 `rocminfo | grep gfx` 为准)。

## 3. 路径与环境变量(Linux 行为)

| 项 | Linux 默认 | 说明 |
|---|---|---|
| 课程输出 | `/workspace/yolo_course/` | `course/lab_utils.root()` 检测到 POSIX `/workspace` 时启用 |
| 自定义 | `YOLO_COURSE_HOME=/workspace/my_course` | 优先级最高,两种 OS 一致 |
| 用户数据集 | `YOLO_USER_DATA_YAML=/abs/path/data.yaml` | Lab B 可选,绝对路径 |
| GPU 设备节点 | `/dev/kfd`、`/dev/dri` | Notebook 00 仅探测记录;不可见时先查平台资源分配 |

Windows 侧的 `root()` 改动(仅在 POSIX 分支生效)不影响上述 Linux 行为;
Linux 上 `/workspace` 检测逻辑与原版完全一致。

## 4. 执行顺序与验证

```bash
.venv/bin/python scripts/gpu_smoke.py        # GPU 准入(Fwd/Bwd/Optimizer)
.venv/bin/python -m pytest -q                # 离线回归
.venv/bin/python scripts/run_all_labs.py     # 00→05 顺序执行 + 产物校验 + 执行矩阵数据
```

## 5. gfx1100 适配说明

- 训练统一 `workers=0`(Linux 亦默认,保证 A/B 与跨平台行为一致);
  课堂如需提速可在确认无数据竞争后自行调整,并记录。
- `amp=False` 为课程默认;如平台确认 AMP 稳定,重训前统一变更并记录(影响 A/B 可比性)。
- 本包在 Windows 上验证的 wheel 栈(`2.12.0+rocm7.14.1` + MIOpen hipRTC 自包含头修复,
  ROCm/rocm-libraries#13437)同样适用于 Linux;Linux 平台预装栈版本不同时,以平台栈为准,
  在 Notebook 00 的 `environment.json` 中如实记录。

## 6. 与 Windows 验证的关系

Windows PASS ≠ Linux PASS。Windows 端的 NOTEBOOK_EXECUTION_MATRIX 仅证明 Windows 原生路径;
Linux Radeon Cloud 需在其环境真实跑一遍 `run_all_labs.py` 才能宣布 Linux PASS
(静态兼容性已核查,见 validation/COMPATIBILITY_MATRIX.md)。
