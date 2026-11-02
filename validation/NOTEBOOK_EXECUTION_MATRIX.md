# Notebook 执行矩阵(NOTEBOOK_EXECUTION_MATRIX)

- 数据来源:本轮真实执行 `validation/NOTEBOOK_EXECUTION_DATA_20261102_170901.json`(由 scripts/run_all_labs.py 自动生成,非人工填写)
- 执行环境:Windows 11 原生 · Kernel `yolo26-rocm7141`(registered,argv→项目 .venv python)· torch 2.12.0+rocm7.14.1 · HIP 7.14.60850 · RX 7900 XTX(gfx1100)
- 执行时间窗:2026-11-02 17:09:01 – 17:14:53(本地)
- 产物根:`yolo_course/`(YOLO_COURSE_HOME 默认)· 执行后副本:`validation/executed/20261102_170901/`

| Notebook | Code Cells | Executed | Result | GPU Used | Duration | Evidence |
|---|---:|---:|---|---|---|---|
| 00_ROCm_Environment_Preflight | 5 | 5 | **PASS** | 实测(HIP 断言+GPU matmul) | 10.21 s | `executed/20261102_170901/00_*.executed.ipynb` · `yolo_course/reports/environment.json` · `artifacts/yolo26n.pt`(sha256 9b09cc8b… 与官方下载一致) |
| 01_Lab_A_Predict_and_Threshold | 7 | 7 | **PASS** | 实测(device=0, 3×GPU 推理) | 14.19 s | `reports/lab_a_predictions.json`(conf 0.10/0.25/0.50 → 7/5/5 检测,单调性断言过)+ 3 张预测图 |
| 02_Lab_B_Dataset_Annotation_QA | 7 | 7 | **PASS** | N/A(CPU 数据 QA;GPU 门禁 cell 仅探测) | 6.64 s | `data/processed/coco128_classroom_3class/`(train 89/val 20/test 19,0 结构错误,`dataset_manifest.json`、`split_manifest.csv`) |
| 03_Lab_C_ROCm_Training_Finetune | 6 | 6 | **PASS** | 实测(COCO8 冒烟 1 Epoch + baseline 12 Epoch 真训练) | 200.19 s | `runs/lab_c/smoke/`、`runs/lab_c/baseline/`(best.pt+last.pt+results.csv 12 行+results.png)、`reports/lab_c_baseline.json`、`lab_c_baseline_val.json` |
| 04_Lab_D_Evaluation_Error_Analysis | 6 | 6 | **PASS** | 实测(官方 val + 逐图 GPU predict) | 30.51 s | `reports/lab_d_official_metrics.json`(P .838/R .552/mAP50 .624/mAP50-95 .380)、`lab_d_fixed_threshold.json`(TP18/FP6/FN11)、`lab_d_errors.csv`、失败对照图、PR/混淆矩阵图 |
| 05_Lab_E_Controlled_AB_Optimization | 6 | 6 | **PASS** | 实测(Variant 12 Epoch 真训练 + 双臂同口径 GPU 评测) | 89.56 s | `runs/lab_e/imgsz416/`(best.pt sha256 13da48a3…)、`reports/lab_e_variant.json`、`lab_e_ab_comparison.csv/json`、`assets/lab_e_ab_mAP.png` |

**合计:6/6 PASS,37/37 code cells 执行,0 FAIL,0 BLOCKED。**

## 关键完整性断言(由 runner 与 Notebook 自身执行)

- 六份 Notebook 顺序执行,跨 Notebook 状态(course_config.json)传递正确:00 冻结权重 → 02 登记数据集 → 03 生成 best.pt+train_settings → 04/05 消费。
- A/B 纯度:`variant_args` 相对 baseline 的改动集合 = `{imgsz, project, name}`(Notebook 05 内置断言通过)。
- 初始权重一致性:05 校验 `yolo26n.pt` SHA256 == 00 记录值(`9b09cc8b…`)通过。
- 阈值单调性(Lab A)、数据划分互斥(Lab B `pass=True`、QA 零 problem)、best.pt 存在性(Lab C 断言)全部由真实执行验证。
- 训练曲线:`results.csv` 12 行(epoch 1-12)+ `results.png` 已生成。
- 固定阈值匹配与官方 mAP 的定义差异已在 Notebook 04 输出中明示(`"note": "Teaching matching only; not a COCO AP implementation."`)。

## 执行后 Notebook 副本

`validation/executed/20261102_170901/` 内含 6 份 `*.executed.ipynb`(含全部 cell 输出、执行计数、耗时元数据)与逐 Notebook `*.log.json`(逐 cell 计时/输出尾端/错误栈)。
