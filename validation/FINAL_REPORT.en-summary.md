# Final Validation Report — English Summary

Full Chinese report: [FINAL_REPORT.md](FINAL_REPORT.md) · Date: 2026-11-02

## Verdict: `DAY-1 LABS WINDOWS ROCm = GO` (12/12 acceptance criteria met)

## Environment

- Native Windows 11 (build 26300), no WSL/VM/Docker; isolated uv venv, Python 3.12.16.
- PyTorch trio exactly as pinned: torch `2.12.0+rocm7.14.1`, torchvision `0.27.0+rocm7.14.1`,
  torchaudio `2.11.0+rocm7.14.1`, from `repo.amd.com/rocm/whl-multi-arch/` with the
  `device-gfx1100` extra; HIP 7.14.60850; GPU: AMD Radeon RX 7900 XTX, `gcnArchName=gfx1100`.
- Dedicated Jupyter kernel `yolo26-rocm7141` (argv verified to point at the project venv).

## Key engineering finding

The ROCm 7.14.1 Windows wheels ship an MIOpen whose hipRTC training kernels still require the host
STL (`miopen_type_traits.hpp:151 'type_traits' file not found` → `miopenStatusUnknownError`),
i.e. the exact defect fixed upstream by [ROCm/rocm-libraries#13437] — no fixed wheel existed on the
index at validation time. Local fix with **zero version changes**: the officially supported hipRTC
extension point `HIPRTC_COMPILE_OPTIONS_APPEND` injects four freestanding, HIP-guarded headers
(`<type_traits> <utility> <initializer_list> <cstdint>`), mirroring the upstream self-contained
approach. Before/after evidence: `logs/bn_repro_*.log` (fail → `BatchNorm2d forward OK / backward OK`).

## GPU admission gate — 17/17 PASS

Versions/HIP/gfx1100, tensor allocation, matmul, Conv2d forward, BatchNorm training mode, finite
loss, backward with GPU gradients, optimizer step changing weights (max|Δw|=7.97e-05), sync,
NaN/Inf screening. Details: [GPU_PHYSICAL_SMOKE.md](GPU_PHYSICAL_SMOKE.md).

## Notebook execution — 6/6 PASS (37/37 code cells)

| NB | Result | Duration | Key outputs |
|---|---|---|---|
| 00 preflight | PASS | 10.2 s | environment.json; pinned yolo26n.pt (SHA 9b09cc8b…) |
| 01 Lab A | PASS | 14.2 s | conf 0.10/0.25/0.50 → 7/5/5 detections, monotonicity asserted |
| 02 Lab B | PASS | 6.6 s | train 89 / val 20 / test 19, zero QA problems |
| 03 Lab C | PASS | 200.2 s | COCO8 smoke + 12-epoch training; best.pt + curves |
| 04 Lab D | PASS | 30.5 s | official val + fixed-threshold TP18/FP6/FN11 + failure cases |
| 05 Lab E | PASS | 89.6 s | variant training + same-protocol A/B evaluation |

A second, independent full run (bilingual edition, isolated course home) also passed 6/6 with
bitwise-identical A/B metrics — see `NOTEBOOK_EXECUTION_DATA_20261102_210903.json`.

## Baseline results (real training)

- `best.pt` SHA256 `1c058d8d1cc076c35f01e20f427acc6874fab0eab073f529a20f5d8093ae6f8b`
- 12 epochs · imgsz=512 · batch=8 · SGD · seed=42 · amp=False · workers=0; 79.68 s; peak allocated 1.546 GiB
- Validation (val=20, eval imgsz=640): **P 0.8380 · R 0.5517 · mAP50 0.6245 · mAP50-95 0.3795**

## A/B experiment (controlled, faithfully reported)

| Arm | Only variable | mAP50-95 | Precision | Time | Peak alloc |
|---|---|---|---|---|---|
| Baseline | imgsz=512 | **0.3795** | 0.838 | 79.7 s | 1.546 GiB |
| Variant | imgsz=416 | 0.3381 | 0.731 | 85.8 s | 1.066 GiB |

ΔmAP50-95 = −0.0414 → **the variant is worse and is reported as such** (not adopted). Memory fell
31% as expected; the time difference is noise on this tiny dataset. Limitations: single seed,
tiny val split (20 images), runtime noise — not generalizable. The test split stayed sealed.

## Regression & integrity

- `pytest`: 16/16 (original 6 + 10 new: path handling, fake-CPU-cannot-impersonate-HIP gate,
  A/B purity, notebook structure, Windows separators, YAML roundtrip).
- One-command entrypoint `scripts/validate_windows.ps1` prints `ALL VALIDATION STAGES PASSED`.
- Notebooks were never modified to bypass checks; the bilingual regeneration kept code cells
  byte-identical (verified by hash) with markdown only becoming bilingual.

## Linux Radeon Cloud

`NOT_RUN / STATIC_COMPATIBILITY_CHECKED` — POSIX branches, `/workspace` conventions and env vars
verified statically; a real cloud-GPU run is still required before claiming Linux PASS.
