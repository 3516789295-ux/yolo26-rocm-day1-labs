# Troubleshooting — YOLO26 ROCm Day 1 Labs (Native Windows / Linux Radeon Cloud)

English | [中文](TROUBLESHOOTING.md)

## Environment

| Symptom | Root cause | Fix |
|---|---|---|
| `torch.version.hip is None` | The kernel runs CPU/CUDA PyTorch | Switch to the `yolo26-rocm7141` kernel; never `pip install torch` in this env |
| `pip check` reports torch conflicts | Some dependency tried to replace the trio | Reinstall with `--constraints constraints-rocm.txt`; if already broken, redo setup §2 |
| `torch.cuda.is_available()` False | Old driver / GPU busy elsewhere / not a ROCm build | Update Adrenalin; check GPU usage; confirm `torch.version.hip != None` |
| GPU model ≠ gfx1100 | Hardware mismatch | Reinstall with the matching `device-gfxXXXX` extra; never fake a pass |
| `uv lock` fails with `symlink 'nul'` | uv × AMD index incompatibility on Windows | Use constraints-rocm.txt + the pip freeze manifest (this repo's approach) |

## Training (Labs C / E)

| Symptom | Root cause | Fix |
|---|---|---|
| **First training / conv / BN stalls for minutes** | MIOpen's first hipRTC compile + tuning sweep (writes a user cache) | Wait; the second run reuses the cache. Growing CPU time means it is working, not hung |
| `miopenStatusUnknownError` + hipRTC `'type_traits' file not found` | Old MIOpen headers need host STL (fixed upstream in ROCm/rocm-libraries#13437) | Run `scripts/install_miopen_shims.py --setx`; do not downgrade wheels |
| NaN during training | AMP numerical issues on some stacks | The course default is `amp=False`; if you changed it, restore and record |
| DataLoader hangs / spawn errors on Windows | Multiprocessing workers | The course pins `workers=0`; never change only one A/B arm |
| OOM | Batch too large | You may lower `batch` for classroom runs, but change **both A/B arms identically** and record it |
| Slow training | First-time kernel compile, cold cache, shared GPU | Check that the MIOpen cache was created; confirm the GPU is idle; the imgsz=416 arm is naturally faster |

## Data (Lab B)

| Symptom | Root cause | Fix |
|---|---|---|
| COCO128 download times out | Slow/blocked GitHub direct link | Pre-place `coco128.zip` at `yolo_course/data/raw/` (official offline path) |
| `EXACT IMAGE LEAKAGE` | Copies across splits | That is the QA doing its job; remove the duplicated images |
| Label `box extends outside image` | Annotation out of bounds | Fix with Lab B's backup-then-correct flow; never bypass QA |
| Splits differ from a classmate's | Different seed or source data | The course pins `seed=42`; declare any custom setup |

## Evaluation (Labs D / E)

| Symptom | Root cause | Fix |
|---|---|---|
| Fixed-threshold P/R differs a lot from official mAP | Different definitions (teaching match ≠ COCO AP) | Expected — write it into your analysis; "fixing" either side is wrong |
| `best.pt` missing | Lab C unfinished or stale config path | Run in order; never hand-craft a fake weight |
| A/B differs by more than imgsz | Params copied wrongly | The `changed == {'imgsz','project','name'}` assertion catches it; revisit baseline_train_settings |
| Test score checked repeatedly | Violates test-set sealing | `RUN_FINAL_TEST` may only be flipped once, after the plan is frozen |

## Jupyter / paths

| Symptom | Root cause | Fix |
|---|---|---|
| `FileNotFoundError: 请先将 ZIP 完整解压…` | cwd is not the project root / notebooks/ | Start from the project root or notebooks/; or set `YOLO_COURSE_HOME` |
| Outputs land in an unexpected place | `YOLO_COURSE_HOME` is set | `print(home)` to confirm; use a dedicated home for isolated tests |
| Windows says `/dev/kfd` does not exist | Those are Linux device nodes | The notebook only probes them; `False` on Windows is expected |
| Plots don't show (headless nbclient) | matplotlib backend | The runner sets `MPLBACKEND=module://matplotlib_inline.backend_inline` |
