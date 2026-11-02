# Lab B · X-AnyLabeling 手工标注 → YOLO 数据集质量检查

适用：银川科技学院 YOLO26 Day 1 / Lab B，建议 50–60 分钟。学生在本地使用免费开源 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)，Jupyter Notebook 在 Windows ROCm 或 Linux Radeon Cloud 上做离线质量检查。标注程序**无需**安装进 ROCm/PyTorch 虚拟环境。

参考官方说明：[X-AnyLabeling 中文用户手册](https://xanylabeling.com/zh-Hans/docs/x-anylabeling/user_guide)；[Ultralytics YOLO 检测数据集格式](https://docs.ultralytics.com/datasets/detect/)。

## 0. 课程成果

每位学生手工标注至少 **5 张**授权图片，再与同伴交叉复核至少 **5 张**。每组提供至少 **9 张图片且有 3 个独立场景/分组**，才能尝试 Train/Val/Test 划分；图片太少、相似场景太集中时，使用默认 COCO128 教学数据执行剩余实验，不得用它声称可靠泛化指标。

## 1. 标注前先制定规范

- 一致的任务：**目标检测（Rectangle / Bounding Box）**，不要误导出 segmentation/pose/OBB 标签。
- 一致的类别文件：例如只检测快递箱时，classes.txt **只有一行** parcel；多类时每类一行，行号从 0 开始，不能在标注中途改变顺序。
- 一致的边界：同一目标尽可能包住可见区域；遮挡、过小、模糊、截断目标需要小组共同约定并记录。
- 标注要完整：不能只标几个容易看到的箱子，而将其他目标留下当作背景。
- 每张图片都需对应标签 .txt；已人工确认没有目标的图片使用**空 txt**，不要静默混入未标注图。

## 2. X-AnyLabeling 实操

1. 从官方 Releases 安装桌面应用，启动后打开有权使用的原始图片目录。
2. 创建并审阅 classes.txt，确定所有学生使用相同类别名及顺序。
3. 先手工绘制矩形框并保存原生项目文件；AI 自动标注只能作为后续辅助，必须有人审阅。
4. 与同学交换图片复核：漏标、错标、边界过松/过紧、重复框、类别 ID 混淆。
5. 在菜单中选择 **Export Annotations → Export YOLO Annotations → Detection**，按提示选择 classes.txt；输出按图片文件名对应的 .txt。
6. 确保图片文件放在 images/，YOLO txt 放在独立的 labels/，不要将 txt 混入图片文件夹。

注意：软件界面菜单在不同版本和语言下可能略有差异，以官方用户手册为准。

## 3. 上传 Radeon Cloud 的数据目录

在云端上传/解压到 **持久化空间**，例如：

~~~text
/workspace/yolo_course/data/raw/anylabeling/
├── images/
│   ├── parcel_001.jpg
│   ├── parcel_002.jpg
│   └── ...
├── labels/
│   ├── parcel_001.txt
│   ├── parcel_002.txt
│   └── ...
├── classes.txt
└── scenes.csv                 # 可选，强烈建议视频/同一场景使用
~~~

标注行格式：class_id x_center y_center width height；后四个值为相对于图片尺寸的归一化值，坐标通常在 [0,1]，宽高必须大于 0。

可选 scenes.csv：

~~~csv
image,scene
parcel_001.jpg,warehouse_A_day1
parcel_002.jpg,warehouse_A_day1
parcel_003.jpg,warehouse_B_day2
~~~

若启用，文件必须覆盖每一张图片且每张只出现一次。同一个 scene 下的图片只能进入一个集合；程序也会把字节相同或启发式近似的图片分到同一集合。**这不是完美的场景泄漏识别器，人工检查仍然必要**。

## 4. 在 Lab B Notebook 中启用真实学生标注

在 notebook 中对应的“选择数据来源”代码 Cell 前（或在 Jupyter 运行前设置 Kernel 环境变量）执行：

~~~python
import os
os.environ["YOLO_ANYLABELING_EXPORT"] = "/workspace/yolo_course/data/raw/anylabeling"
os.environ["YOLO_ANYLABELING_SCENES"] = "/workspace/yolo_course/data/raw/anylabeling/scenes.csv"  # 无则设为 ""
os.environ["YOLO_ANYLABELING_OUTPUT_NAME"] = "xanylabeling_student_v1"
~~~

若在 Windows 本地运行，以上路径替换成 Windows 的本地实际目录，如：

~~~python
os.environ["YOLO_ANYLABELING_EXPORT"] = r"C:\Users\Rocm\Desktop\YOLO\student_export"
os.environ["YOLO_ANYLABELING_SCENES"] = ""
~~~

如果不设置 YOLO_ANYLABELING_EXPORT，Lab B 按已验证的老流程运行 COCO128 重分组教学数据；如果设置 YOLO_USER_DATA_YAML，则优先使用完整自备 data.yaml。

执行后产生：

~~~text
<YOLO_COURSE_HOME>/data/processed/xanylabeling_student_v1/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
├── split_manifest.csv
└── dataset_manifest.json

<YOLO_COURSE_HOME>/reports/
├── lab_b_annotation_import.json
├── lab_b_dataset_qa.json
└── lab_b_manual_review.csv
~~~

Lab C 会从共享配置读取数据集路径。**如果修改原始标注**，将 YOLO_ANYLABELING_OUTPUT_NAME 修改为 xanylabeling_student_v2；不应通过覆盖 v1 掩盖历史标注变更。

## 5. 人工复核与验收

- 每人至少 5 张手工标注与 5 张同伴复核的记录；小组应提交审阅清单。
- 自动检查无错误：文件配对、类别 IDs、归一化坐标、数据集三路目录、跨集合精确重复哈希。
- 对 dHash 潜在近重复警告进行人工复核；按来源/视频/场景分组的数据要能说明独立性。
- 截图或可视化显示边界框，并解释至少 3 个实际标注问题的原因与纠正方式。
- 提交 data.yaml、split_manifest.csv、dataset_manifest.json、lab_b_dataset_qa.json 和 DATASET_QA.md（写明版权、标注策略、样本规模、局限）。
- 不在公共 GitHub 上传无权分享的图片或模型；不把小样本课程演练结果宣传为真实工业基准。

## 6. 常见阻塞

| 现象 | 原因/解决 |
|---|---|
| Missing labels | 必须补齐对应 txt；真正无目标的图片也要人工确认后显式留空 |
| Invalid YOLO detection labels | 类别编号错、坐标超界、误导出分割/旋转框；回到标注工具修正并重新导出 |
| Fewer than three independent groups | 同场景过多或重复图过多；新增独立拍摄场景，不要破坏分组来凑数 |
| Destination already exists | 保护已有实验；修改 YOLO_ANYLABELING_OUTPUT_NAME 后创建新修订 |
| QA near-duplicate warnings | 仔细核对同视频/同视角图片，并将同一场景放在同一 split |
| Lab C still uses COCO128 | 确认 Lab B 导入成功及共享 course_config 中 dataset_yaml 的路径；重启/重跑配置加载步骤 |

*注意：本流程尚需在学生实际 X-AnyLabeling 导出文件和 Radeon Cloud Linux GPU 上做一次教学走通；离线单元测试只能证明数据导入与检查逻辑。*
