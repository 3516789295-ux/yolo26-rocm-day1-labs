# AMD Radeon Cloud (Linux) Deployment — YOLO26 Day 1 Labs

English | [中文](RADEON_CLOUD_LINUX_SETUP.md)

This package was originally built for Linux Radeon Cloud Jupyter environments; the new native-Windows path changed nothing on the Linux side.

## 1. Upload and unpack

```bash
cd /workspace
unzip YOLO26_ROCm_Day1_Labs_v1.0.zip
cd YOLO26_ROCm_Day1_Labs
```

`notebooks/` and `course/` must stay in the same project directory.

## 2. Environment (two options)

### A. Platform-provided ROCm PyTorch (recommended, zero install)

Open the notebooks with the platform's ROCm kernel. **Never** run `pip install torch` / `pip install -U torch`.

### B. Self-built uv environment (same shape as Windows)

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

Adjust the device extra to your actual GPU (e.g. `device-gfx942` for MI300, `device-gfx1100` for RX 7000,
gfx1151/gfx1201 for RX 9060/8060 — confirm with `rocminfo | grep gfx`).

## 3. Paths and environment variables (Linux behavior)

| Item | Linux default | Notes |
|---|---|---|
| Course outputs | `/workspace/yolo_course/` | Enabled by `course/lab_utils.root()` when POSIX `/workspace` exists |
| Custom location | `YOLO_COURSE_HOME=/workspace/my_course` | Highest priority, identical on both OSes |
| User dataset | `YOLO_USER_DATA_YAML=/abs/path/data.yaml` | Optional for Lab B; absolute path |
| GPU device nodes | `/dev/kfd`, `/dev/dri` | Notebook 00 only probes them; if invisible, check platform resource allocation |

The Windows-side `root()` change (POSIX gate) does not affect the behavior above; the Linux
`/workspace` detection logic is character-for-character identical to v1.0.

## 4. Execution order and validation

```bash
.venv/bin/python scripts/gpu_smoke.py        # GPU admission (fwd/bwd/optimizer)
.venv/bin/python -m pytest -q                # offline regressions
.venv/bin/python scripts/run_all_labs.py     # run 00→05 in order + artifact checks + matrix data
```

## 5. gfx1100 adaptation notes

- Training pins `workers=0` on every platform (keeps A/B and cross-OS behavior identical);
  you may raise it after confirming no data races — and record the change.
- `amp=False` is the course default; if your platform confirms stable AMP, change it uniformly
  before retraining and record it (it affects A/B comparability).
- The wheel stack validated on Windows (`2.12.0+rocm7.14.1`, plus the self-contained MIOpen
  hipRTC headers of ROCm/rocm-libraries#13437) applies to Linux as well; when the platform ships
  its own preinstalled stack, record its versions honestly in Notebook 00's `environment.json`.

## 6. Relationship to the Windows validation

Windows PASS ≠ Linux PASS. The Windows execution matrix proves only the native-Windows path;
Linux Radeon Cloud must run `run_all_labs.py` in its own environment before Linux PASS can be
claimed (static compatibility is checked — see validation/COMPATIBILITY_MATRIX.md).
