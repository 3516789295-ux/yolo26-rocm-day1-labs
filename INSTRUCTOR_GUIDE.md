# 讲师使用指南｜AMD Radeon Cloud · YOLO26 Day 1

> 本指南侧重教学运行秩序、能力评价与“真实证据”。对应 00 + Lab A–E 共六份 `.ipynb`。

## 一、课前预演（建议由讲师用与学生相同的 Radeon Cloud 镜像执行）

> **一键课前自检**：`python scripts/preflight_classroom.py --course-home <课堂输出目录>`（GPU/NMS/权重/数据/Kernel/磁盘逐项 PASS/FAIL）。完整发布清单、错峰表与降级预案见 [validation/linux_w7900/CLASSROOM_READINESS_CHECKLIST.md](validation/linux_w7900/CLASSROOM_READINESS_CHECKLIST.md)。

- [ ] AMD GPU 已分配，`torch.version.hip` 非空，`torch.cuda.is_available()` 为 True；确认 `torch.cuda.get_device_name(0)`。
- [ ] Jupyter Kernel 的 pip 与运行 Python 对应同一环境；保存 `reports/environment.json`。
- [ ] 核实 `yolo26n.pt` 在 `/workspace/yolo_course/artifacts/`，保留 SHA256。
- [ ] 离线代理受限时，提前按官方许可准备 `coco128.zip`、`bus.jpg`，或设好云端 HTTPS 代理。
- [ ] 运行 Lab B 生成教学数据；对 Val/Test 里的重复图片、相邻视频帧或相似场景手动核验。
- [ ] **正式演示**：至少用与学生相同的 Kernel 完成一次 Lab C 冒烟/微调，以及 Lab D 官方验证；记录模型版本与运行时。
- [ ] 预备一套真实、已核验的 `best.pt`、`results.csv`、`confusion_matrix.png`、`PR_curve.png`（如有），备用时标为“讲师预制工件”，不得冒充学生运行。
- [ ] 提前备份有权使用/发布的工业目标检测数据；若用包裹图像，必须有相应标签，不能把 COCO128 三分类成绩描述为包裹模型成绩。
- [ ] 第一天下午准备错峰 GPU 训练：小组练习尽量通过短训练先收获真实工件，长任务安排课后完成。

## 二、建议实验节奏与评分

| 实验 | 课堂完成 | 讲师重点看什么 | 验收 |
|---|---|---|---|
| 00 环境检查 | 10 min | 能区分 ROCm HIP 与 CPU/CUDA，知道 `device=0` | environment.json |
| A 预测与阈值 | 20 min | 明白阈值改变不会修改权重；IoU 几何 | 3 张图+JSON |
| B 数据 QA | 45 min | 图像与标签对应、归一化坐标、训练测试隔离 | 数据 YAML + QA + 人工校对 |
| C 训练微调 | 60 min | 固定预训练权重、训练日志/模型 SHA256、真实 GPU | best.pt + baseline record |
| D 评价分析 | 40 min | mAP 不等于单阈值 TP/FP/FN；错误有证据 | 官方指标 + 真实错误图 |
| E A/B 优化 | 40 min（结果可课后补齐） | 受控变量、同口径验证、诚实说明局限 | 两次训练 + CSV |

建议按以下维度评分（总分 100）：数据与标注 20、训练与复现 20、评价指标 20、单变量实验 20、GitHub 证据及表达 20。对 GPU 资源不足而只能提交部分训练的同学，明确标记未完成，不鼓励虚构精度。

## 三、课堂常见误区与标准回答

**Q：为何 Radeon Cloud 的 ROCm PyTorch 也用 `torch.cuda`？**  
A：HIP 版 PyTorch 复用 CUDA-style Python API；应同时检查 `torch.version.hip` 和 GPU 名称，设备参数 `device=0`。

**Q：COCO128 为什么不能用来报告可信 mAP？**  
A：官方 train/val 图片重叠；本包重新分组划分仅能做小规模课程演练，缺乏规模、场景独立性和统计稳定性，不能对外当 COCO Benchmark 或工业精度证明。

**Q：mAP50-95 的 50–95 是什么？**  
A：IoU 阈值从 0.50 到 0.95、每 0.05 一档；不是 confidence。

**Q：为什么固定阈值匹配得到的 Precision/Recall 与 Ultralytics 验证输出不同？**  
A：mAP/官方 P、R 使用完整检测排序和内部匹配统计规则，Lab D 的匹配是为了理解错误、固定 `conf=0.25` 且采取简化一对一贪心算法，口径不同。

**Q：Lab E 比 baseline 更快就是算法赢了吗？**  
A：不一定。模型需要达到业务准确率约束；完整比较还需要评估推理端延迟/吞吐、重复实验、数据分布与并发负载。Lab E 的训练耗时不代表线上推理时延。

**Q：打开 Test 后指标不好，是否可以再修改模型然后继续刷 Test？**  
A：不能继续把同一个 Test 当独立最终评估。回到新的训练/验证阶段，另行准备真正独立的测试数据，保留原结果与决策历史。

## 四、GitHub 交付标准

每组提交：README（业务问题和复现路径）、全部可分享 Notebook、`environment.json`（不要提交敏感代理值）、数据来源和许可、`DATASET_QA.md`、`TRAINING_REPORT.md`、`EVALUATION.md`、`EXPERIMENT_REPORT.md`、模型与数据版本/哈希、若有可分发的少量图像与错误示例。**不应上传无权发布的工业图片/权重或私有云凭据。**

给学生的面试表达要求：“描述基线 → 证据 → 问题 → 受控实验 → 收益与代价”，而不是只说“我会跑 yolo train”。

## 五、实验安全约束

- 所有标签编辑备份旧文件并记录理由；用户输入的数据与脚本不自动提交到外部仓库。
- 云端 GPU 昂贵：讲师与学生执行较长训练前核实资源额度与排队方案。
- 不自动升级 PyTorch；任何 `torch` 版本调整需与 ROCm 兼容矩阵核对。
- 如果云端代理不可用，应使用事先下载并有权限使用的样例，不能虚构实验结果。
- 课程含外部第三方模型与数据，请按各自许可证执行，尤其是转向商业应用时。


## Lab B 新增实操（50–60 分钟）

使用 [X-AnyLabeling 标注与导入操作指南](docs/LAB_B_X_ANYLABELING.md) 指导各小组：①统一类别及边界标注规范；②每人手动标注至少 5 张授权图片；③同伴复核至少 5 张；④导出 YOLO Detection 与 classes.txt；⑤上传 images/、labels/，在 Lab B 设置 YOLO_ANYLABELING_EXPORT；⑥自动检查文件、类别、坐标、数据划分，人工复核 near-duplicate 和场景独立性。收集 3 个标注错误案例与修复说明。

若导出图片不够 9 张、没有 3 个独立图片/场景组，或者存在标签错误，不要伪造完整 Train/Val/Test；使用保留的 COCO128 教学路径继续 Lab C。**原有 Windows RX 7900 XTX 的 6/6 PASS 是历史验证，并不是新增 X-AnyLabeling 导出分支的 GPU 验收。**
