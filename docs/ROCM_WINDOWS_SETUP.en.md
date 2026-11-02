# Native Windows ROCm Environment Setup (YOLO26 Day 1 Labs)

English | [中文](ROCM_WINDOWS_SETUP.md)

This guide builds an isolated environment that runs all six notebooks on **Windows 11 + AMD Radeon RX 7900 XTX (gfx1100) + uv**, from scratch. No WSL / Docker / virtual machines needed.

## 0. Prerequisites

| Item | Requirement | Validated machine |
|---|---|---|
| GPU | RDNA3, gfx1100 (RX 7900 XTX/XT) | RX 7900 XTX ✓ |
| Driver | AMD Adrenalin ≥ 25.x (after 2025-09) | 32.0.21030.2001 (2025-09-25) ✓ |
| OS | Windows 11 22H2+ | 11 26H2 build 26300 ✓ |
| Python | 3.12 x64 (provided by uv; no system Python needed) | 3.12.16 ✓ |
| uv | ≥ 0.13 | 0.13.6 ✓ |
| Disk | ≥ 15 GB (ROCm wheels ~4 GB + data/weights) | 645 GB free ✓ |

No system-level ROCm SDK is needed: the `torch[device-gfx1100]` extra installs `rocm-sdk-device-gfx1100` and friends from AMD's index; the wheels carry the HIP user-space runtime, and the Adrenalin driver provides the kernel-mode interface.

## 1. Create the isolated environment (project root)

```powershell
uv python install 3.12
uv venv --python 3.12 --seed .venv     # --seed provides pip for python -m pip installs
.\.venv\Scripts\python.exe --version   # Python 3.12.16
```

> Before the AMD index is configured, do **not** run `uv sync` / `uv pip install torch` —
> the resolver would pull CPU/CUDA PyTorch from PyPI and clobber the AMD build.

## 2. Install AMD ROCm PyTorch (versions are mandatory)

```powershell
.\.venv\Scripts\python.exe -m pip install `
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ `
  "torch[device-gfx1100]==2.12.0+rocm7.14.1" `
  "torchvision[device-gfx1100]==0.27.0+rocm7.14.1" `
  "torchaudio==2.11.0+rocm7.14.1"
```

The `device-gfx1100` extra additionally installs:
`amd-torch-device-gfx1100` / `amd-torch-device-gfx110x` / `amd-torchvision-device-gfx1100`
`rocm-sdk-device-gfx1100` / `rocm-sdk-libraries` / `rocm-sdk-core` / `rocm-bootstrap` / `rocm`.

Verify:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip show torch torchvision torchaudio
.\.venv\Scripts\python.exe scripts\gpu_smoke.py    # physical GPU admission gate
```

### Known issues (all measured on real hardware)

- **Training crashes with `miopenStatusUnknownError` / hipRTC `'type_traits' file not found`**:
  MIOpen's hipRTC kernel headers depend on the host STL; fixed upstream in
  [ROCm/rocm-libraries#13437](https://github.com/ROCm/rocm-libraries/pull/13437) (merged 2026-10-19).
  **The `rocm7.14.1` wheels do NOT contain the fix yet** (no newer build on the AMD index) — every
  BatchNorm/Conv training kernel fails to compile. The repo's solution (zero version changes): run
  `.\.venv\Scripts\python.exe scripts\install_miopen_shims.py --setx`
  to inject freestanding STL shims (see `validation/GPU_PHYSICAL_SMOKE.md` §4 and
  `validation/CHANGELOG_VALIDATION.md` §0). Remove them once AMD ships a fixed wheel.
- **Slow first run**: MIOpen compiles/tunes kernels (batchnorm, conv, …) on first use, which can take
  minutes. Results are cached in the user profile afterwards — it is not a hang.
- `uv lock` is incompatible with this AMD index on Windows (the index contains a `nul` reserved-name
  entry; uv fails with `failed to query metadata of symlink 'nul'`). This repo therefore locks the
  environment with `constraints-rocm.txt` + a pip freeze manifest instead of uv.lock
  (see `validation/DEPENDENCY_MANIFEST.md`).

## 3. Install course dependencies (protecting the ROCm trio)

```powershell
uv pip install --python .\.venv\Scripts\python.exe --constraints constraints-rocm.txt `
  -r requirements-course.txt nbclient jupyterlab pytest
.\.venv\Scripts\python.exe -m pip check    # trio must stay at 2.12.0/0.27.0/2.11.0+rocm7.14.1
```

## 4. Register the Jupyter kernel

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user `
  --name yolo26-rocm7141 --display-name "YOLO26 - AMD ROCm 7.14.1"
```

The kernel `argv` must point to this project's `.venv\Scripts\python.exe`
(stored at `%APPDATA%\jupyter\kernels\yolo26-rocm7141\kernel.json`).

## 5. Run

```powershell
# One-command full validation (pip check -> GPU smoke -> pytest -> all six notebooks)
powershell -ExecutionPolicy Bypass -File scripts\validate_windows.ps1

# Or step by step
.\.venv\Scripts\python.exe scripts\gpu_smoke.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_all_labs.py
```

Course outputs default to `<project root>\yolo_course\` (override with `YOLO_COURSE_HOME`).
The notebook SETUP cell locates the `course/` package from the cwd or its parent; start Jupyter
from the project root or from `notebooks/`.

## 6. Offline / weak-network contingency (when GitHub downloads are flaky)

Pre-place the official files (README's official cache paths; SHA256 in validation/DEPENDENCY_MANIFEST.md):

| File | Destination |
|---|---|
| `coco128.zip` | `yolo_course\data\raw\coco128.zip` |
| `yolo26n.pt` | `yolo_course\artifacts\yolo26n.pt` |
| `coco8.zip` (Lab C smoke) | Ultralytics datasets dir (or stay online) |

`bus.jpg` needs no download: the ultralytics wheel ships `assets/bus.jpg`, which the notebook prefers.

## 7. Environment isolation notes

- Everything lives in the project's `.venv`; system Python, registry, and other projects are untouched.
- Uninstall: delete `.venv/`, `yolo_course/`, and `%APPDATA%\jupyter\kernels\yolo26-rocm7141\`.
- The AMD driver stays as-is.
- One reversible user-level env var is added by the shim installer (`HIPRTC_COMPILE_OPTIONS_APPEND`).
