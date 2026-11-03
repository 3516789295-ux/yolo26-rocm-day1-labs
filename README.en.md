# YOLO26 ROCm Day 1 Labs · Hands-on Object Detection on AMD GPUs (Bilingual)

**English | [中文](README.md)**

![status](https://img.shields.io/badge/status-validated%20on%20real%20GPU-brightgreen) ![gpu](https://img.shields.io/badge/GPU-AMD%20Radeon%20RX%207900%20XTX%20gfx1100-red) ![rocm](https://img.shields.io/badge/ROCm-7.14.1-orange) ![torch](https://img.shields.io/badge/PyTorch-2.12.0%2Brocm7.14.1-ee4c2c) ![python](https://img.shields.io/badge/Python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![lang](https://img.shields.io/badge/lang-English%20%7C%20%E4%B8%AD%E6%96%87-informational)

> A **one-day introductory lab on AMD GPU object detection** for college students: ROCm environment verification → inference & thresholds → dataset annotation & leak-proof splits → real GPU training → mAP evaluation & error analysis → a controlled A/B optimization experiment.
> All six notebooks are **bilingual (Chinese + English)**, and every one of them has been executed end-to-end on a real **AMD Radeon RX 7900 XTX (gfx1100) with native Windows 11 ROCm** (37/37 cells, including two genuine 12-epoch trainings). Evidence: [`validation/`](validation/FINAL_REPORT.en-summary.md) → see [FINAL_REPORT.md](validation/FINAL_REPORT.md) (Chinese) for full details.

## ✅ Validated Status (measured, not promised)

| Item | Result |
|---|---|
| Environment | Native Windows 11 (no WSL/Docker) · uv · Python 3.12.16 |
| PyTorch trio | torch `2.12.0+rocm7.14.1` · torchvision `0.27.0+rocm7.14.1` · torchaudio `2.11.0+rocm7.14.1` (official repo.amd.com wheels) |
| GPU | RX 7900 XTX · `gcnArchName=gfx1100` · HIP 7.14.60850 |
| Notebooks | **6/6 PASS** ([execution matrix](validation/NOTEBOOK_EXECUTION_MATRIX.md)) |
| Regression tests | 16/16 passing (`pytest` — data-leak detection, A/B purity, ROCm gate anti-bypass) |
| A/B experiment | Real controlled experiment: imgsz 512 vs 416, faithfully reported (see [FINAL_REPORT.md §E](validation/FINAL_REPORT.md)) |

## 🚀 Quick Start

### Option A · Native Windows (AMD RDNA3 GPU, e.g. RX 7900 series)

```powershell
git clone https://github.com/AIwork4me/yolo26-rocm-day1-labs.git
cd yolo26-rocm-day1-labs
```

Follow **[docs/ROCM_WINDOWS_SETUP.en.md](docs/ROCM_WINDOWS_SETUP.en.md)** step by step (~15 minutes):
create a uv environment → install AMD ROCm PyTorch (wheels from repo.amd.com, not PyPI) → install course deps → register the Jupyter kernel →
**run `scripts/install_miopen_shims.py --setx`** (a required Windows training fix, see below) → open the notebooks.

> **Windows users must read this**: the MIOpen inside the ROCm 7.14.1 wheels has a known defect — hipRTC cannot find
> the host `<type_traits>` while compiling training kernels, which breaks every BatchNorm/Conv training run. This is
> exactly what [ROCm/rocm-libraries#13437](https://github.com/ROCm/rocm-libraries/pull/13437) fixed upstream. This repo
> ships an official-extension-point workaround (`HIPRTC_COMPILE_OPTIONS_APPEND` + freestanding shim headers); one
> command installs it with zero changes to the pinned wheel versions. Details: [docs/TROUBLESHOOTING.en.md](docs/TROUBLESHOOTING.en.md).

### Option B · Linux / AMD Radeon Cloud

Follow **[docs/RADEON_CLOUD_LINUX_SETUP.en.md](docs/RADEON_CLOUD_LINUX_SETUP.en.md)**: upload and unzip into `/workspace`,
then use the platform's ROCm kernel or build the same uv environment. (The Linux path is statically compatibility-checked; it has not been re-validated on a real cloud GPU yet.)

### Running order

1. Start at `notebooks/00_ROCm_Environment_Preflight.ipynb` and **run 00 → 05 in order** (03 needs the weight cache from 00 and the dataset from 02; 04/05 need `best.pt` from 03).
2. All outputs go into `yolo_course/` inside the project (redirect with the `YOLO_COURSE_HOME` environment variable).
3. Want a one-command re-validation? `powershell -File scripts/validate_windows.ps1` (dependency check → GPU smoke → regression tests → all six notebooks).

## 📓 The Six Notebooks

| # | Notebook | Content | Key artifacts |
|---|---|---|---|
| 00 | [Environment preflight](notebooks/00_ROCm_Environment_Preflight.ipynb) | ROCm/HIP verification, GPU smoke, weight pinning | `environment.json` |
| 01 | [Lab A · Predict & threshold](notebooks/01_Lab_A_Predict_and_Threshold.ipynb) | conf 0.10/0.25/0.50, BBox tensors, IoU | 3 prediction images + JSON |
| 02 | [Lab B · Dataset & QA](notebooks/02_Lab_B_Dataset_Annotation_QA.ipynb) | YOLO labels, leak-proof splits, annotation QA | 3-class teaching dataset + QA report |
| 03 | [Lab C · GPU training](notebooks/03_Lab_C_ROCm_Training_Finetune.ipynb) | COCO8 smoke + 12-epoch fine-tune | `best.pt` + training curves |
| 04 | [Lab D · Evaluation](notebooks/04_Lab_D_Evaluation_Error_Analysis.ipynb) | official mAP vs teaching fixed-threshold matching, failure cases | PR curves + error list |
| 05 | [Lab E · A/B experiment](notebooks/05_Lab_E_Controlled_AB_Optimization.ipynb) | controlled experiment varying only imgsz | A/B report + chart |

## 🗂️ Repository Layout

```
notebooks/        six bilingual notebooks (generated from build_notebooks.py, the single source of truth)
course/           lab_utils.py — shared data/QA/eval/path helpers (torch-free, offline-testable)
scripts/          gpu_smoke · run_all_labs · validate_windows · MIOpen shim installer
tests/            16 offline regressions (labels/IoU/leaks/A-B purity/ROCm gate/notebook structure)
docs/             Windows & Linux setup + troubleshooting (Chinese/English)
validation/       real-GPU validation evidence: final report, execution matrix, executed notebooks, logs
build_notebooks.py bilingual generator (code cells byte-identical to the validated run)
constraints-rocm.txt / pyproject.toml   hard pins for the AMD wheel versions
```

## 🧪 Data & Integrity Principles (core of the course)

- The teaching dataset is a person/car/chair re-partition of the official **COCO128** (train 89 / val 20 / test 19, image-disjoint); it is **not** an industrial dataset and its scores must never be presented as COCO-benchmark or production accuracy.
- Bring your own dataset by pointing `YOLO_USER_DATA_YAML` at your `data.yaml`.
- **Strictly forbidden**: train/test contamination, tuning on the test set, faking `best.pt`, pre-baked "improvement" conclusions, dressing smoke-run numbers up as formal results.
- The fixed-threshold TP/FP/FN matching in Lab D is **not** the COCO mAP algorithm — the notebook says so explicitly.

## 🤝 Contributing (student PRs welcome!)

This repo exists to be improved by student experiments. After finishing the labs, send your improvements back:

- 📖 Better translations or explanations → PR directly
- 🐛 Fix a real bug (reproduce it with `pytest` first) → bug-fix PR
- 🧪 New experiment ideas (new A/B variables, more QA checks) → open an issue first
- Grading hints: see the "GitHub evidence & expression" dimension in [INSTRUCTOR_GUIDE.en.md](INSTRUCTOR_GUIDE.en.md)

Workflow: see **[CONTRIBUTING.md](CONTRIBUTING.md)** (branch → test → PR template, bilingual).

## 📄 License

- Original code and docs in this repo: **MIT** ([LICENSE](LICENSE)).
- Third-party assets downloaded at runtime follow their own licenses: Ultralytics YOLO weights & code (Ultralytics License/AGPL-3.0), COCO128 images & annotations (COCO license), `bus.jpg` (Ultralytics). This repo does not redistribute them.
- Training artifacts (`best.pt` etc.) stay local; they are not committed.

## 📎 Further Reading

- [Final validation report](validation/FINAL_REPORT.md) (Chinese) · [Execution matrix](validation/NOTEBOOK_EXECUTION_MATRIX.md) · [GPU admission evidence](validation/GPU_PHYSICAL_SMOKE.md)
- [Instructor guide](INSTRUCTOR_GUIDE.md) (bilingual) · [Design system](DESIGN_SYSTEM.md)
- Official docs: [Ultralytics on AMD](https://docs.ultralytics.com/integrations/amd/) · [YOLO26](https://docs.ultralytics.com/models/yolo26/) · [ROCm](https://rocm.docs.amd.com/)


### Lab B · Manual labeling with X-AnyLabeling (updated)

Students use the open-source [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) desktop app to **annotate at least 5 images manually → peer-review 5 → export YOLO detection .txt files → upload to Radeon Cloud → perform notebook dataset QA and grouped splits**. See the [Lab B guide (Chinese, with command examples)](docs/LAB_B_X_ANYLABELING.md).

Set `YOLO_ANYLABELING_EXPORT` to a folder containing `images/`, `labels/`, and `classes.txt`; optionally set `YOLO_ANYLABELING_SCENES` to an `image,scene` CSV. Without these variables the previously validated COCO128 teaching route remains unchanged. The importer creates a new `data/processed/xanylabeling_student_v1` without overwriting existing results. This new student-export route still requires a physical cloud GPU rehearsal.
