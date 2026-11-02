# Instructor Guide | AMD Radeon · YOLO26 Day 1

English | [中文](INSTRUCTOR_GUIDE.md)

> This guide focuses on classroom discipline, competency grading, and "real evidence". It covers the six `.ipynb` files: 00 + Labs A–E. The labs run natively on Windows ROCm and on Linux Radeon Cloud.

## 1. Pre-class rehearsal (recommended: run on the same image/platform the students will use)

- [ ] An AMD GPU is allocated; `torch.version.hip` is non-null, `torch.cuda.is_available()` is True; note `torch.cuda.get_device_name(0)`.
- [ ] The Jupyter kernel's pip matches the running Python; save `reports/environment.json`.
- [ ] Confirm `yolo26n.pt` sits in `<course home>/artifacts/` and record its SHA256.
- [ ] If the network is restricted, prepare `coco128.zip` / `bus.jpg` in advance under official terms, or configure the HTTPS proxy.
- [ ] Run Lab B to generate the teaching dataset; manually review duplicate images, adjacent video frames, or similar scenes in Val/Test.
- [ ] **Full dress rehearsal**: complete Lab C smoke + fine-tune and Lab D official validation at least once with the students' kernel; record model versions and runtime.
- [ ] Prepare one genuine, verified set of `best.pt`, `results.csv`, `confusion_matrix.png`, `PR_curve.png` (if available) as a backup — label it "instructor-prepared artifact"; never present it as student output.
- [ ] Back up only the industrial data you are licensed to publish; parcel images must come with parcel labels — never describe COCO128 3-class scores as parcel-model scores.
- [ ] Plan off-peak GPU training for the afternoon: let groups harvest real artifacts from short runs first, and schedule long jobs after class.

## 2. Suggested pace and grading

| Lab | In class | What the instructor checks | Acceptance |
|---|---|---|---|
| 00 environment | 10 min | Distinguishes ROCm HIP from CPU/CUDA; knows `device=0` | environment.json |
| A predict & thresholds | 20 min | Understands thresholds don't change weights; IoU geometry | 3 images + JSON |
| B data QA | 45 min | Image↔label mapping, normalized coords, split isolation | data YAML + QA + manual review |
| C training | 60 min | Pinned pretrained weights, logs/model SHA256, real GPU | best.pt + baseline record |
| D evaluation | 40 min | mAP ≠ single-threshold TP/FP/FN; evidence-backed errors | official metrics + real failure images |
| E A/B optimization | 40 min (results may be finished after class) | Controlled variable, same-protocol validation, honest limits | two trainings + CSV |

Suggested grading (100 points): data & annotation 20, training & reproducibility 20, metrics 20, single-variable experiment 20, GitHub evidence & communication 20. For students who can only submit partial training due to GPU shortage, mark it honestly as incomplete — never reward fabricated accuracy.

## 3. Common classroom misconceptions and standard answers

**Q: Why does ROCm PyTorch also use `torch.cuda`?**
A: HIP builds reuse the CUDA-style Python API. Check both `torch.version.hip` and the GPU name; the device argument is `device=0`.

**Q: Why can't COCO128 report a trustworthy mAP?**
A: Its official train/val images overlap. This package re-groups them for a small classroom drill only — no scale, no scene independence, no statistical stability. It is not a COCO benchmark or industrial-accuracy proof.

**Q: What is the 50–95 in `mAP50-95`?**
A: IoU thresholds from 0.50 to 0.95 in 0.05 steps — not confidence.

**Q: Why do fixed-threshold Precision/Recall differ from Ultralytics' output?**
A: mAP/official P and R use the full detection ranking and internal matching rules; Lab D's matching fixes `conf=0.25` with a simplified one-to-one greedy scheme, for learning purposes. Different definitions, both valid for their own question.

**Q: If Lab E runs faster than the baseline, did the algorithm win?**
A: Not necessarily. The model must meet business accuracy constraints; a full comparison adds inference latency/throughput, repeated runs, data distribution, and concurrent load. Lab E's training time says nothing about serving latency.

**Q: After peeking at Test and seeing poor metrics, may I tweak the model and keep refreshing Test?**
A: No — the same test set can no longer serve as an independent final evaluation. Return to train/validate, prepare genuinely new test data, and keep the original results and decision history.

## 4. GitHub submission standard

Each group submits: a README (business problem + reproduction path), all shareable notebooks, `environment.json` (no sensitive proxy values), data sources and licenses, `DATASET_QA.md`, `TRAINING_REPORT.md`, `EVALUATION.md`, `EXPERIMENT_REPORT.md`, model/data versions and hashes, and — if distributable — a few images and error examples. **Never upload industrial images/weights you may not publish, or private cloud credentials.**

Interview-expression bar for students: "describe the baseline → evidence → problem → controlled experiment → gains and costs" — not just "I can run yolo train".

## 5. Safety constraints

- Every label edit backs up the old file and records a reason; user data and scripts are never auto-pushed to external repos.
- Cloud GPUs are expensive: verify quota and queueing before long runs.
- Never auto-upgrade PyTorch; any `torch` change must be checked against the ROCm compatibility matrix.
- Without a proxy, use pre-downloaded, licensed samples — never fabricate results.
- Third-party models and data follow their own licenses, especially when moving toward commercial use.
