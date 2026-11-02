"""Generate the Day 1 teaching notebooks with repeatable metadata and code.

Bilingual edition: every teaching markdown cell is emitted as Chinese first,
then an English translation after a `---` divider. Code cells are identical to
the validated v1.0 run (byte-for-byte); only markdown carries translations.
The generator stays the single source of truth so the two languages cannot
drift apart.
"""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

HERE=Path(__file__).resolve().parent
OUT=HERE/'notebooks';OUT.mkdir(exist_ok=True)
LANG_DIVIDER='\n\n---\n\n'
STYLE = ('<div style="border-left:5px solid #67CBEE;padding:10px 16px;background:#F7F9FB;color:#19222D;margin:12px 0">'
         '<b>AMD Radeon · YOLO26 · Day 01</b><br><span style="color:#536273">算法 → 数据 → 训练 → 评测 → 证据。每一个结论都应由实际运行结果支撑。</span><br>'
         '<span style="color:#536273">Algorithm → Data → Training → Evaluation → Evidence. Every conclusion must be backed by real execution results.</span></div>')
SETUP = r'''from pathlib import Path
import os, sys, json
candidates = [Path.cwd(), Path.cwd().parent, Path('/workspace/YOLO26_ROCm_Day1_Labs')]
package = next((p for p in candidates if (p/'course/lab_utils.py').exists()), None)
if package is None:
    raise FileNotFoundError('请先将 ZIP 完整解压，并从项目目录或 notebooks/ 目录启动 Jupyter。')
sys.path.insert(0, str(package))
from course.lab_utils import (root, ensure_dirs, load_config, save_config, write_json,
    write_csv, system_info, require_rocm, sha256)
home = ensure_dirs()
cfg = load_config()
print('Project:', package, '\nPersistent output:', home)
'''


def m(zh: str, en: str = ''):
    """Markdown cell: Chinese content, then the English translation below a divider."""
    text = dedent(zh).strip()
    if en:
        text += LANG_DIVIDER + dedent(en).strip()
    return nbf.v4.new_markdown_cell(text)


def c(text):return nbf.v4.new_code_cell(dedent(text).strip())


def make(fname, title_zh, title_en, cells, duration_zh, duration_en, tags_zh, tags_en):
    header = m(
        f'# {title_zh}\n\n**建议时长：{duration_zh}**　｜　学习路径：{tags_zh}\n\n'
        f'> 请按顺序运行；运行结果必须来自自己的 GPU 环境，不可把示例文字当作实测数据。',
        f'# {title_en}\n\n**Suggested time: {duration_en}** | Learning path: {tags_en}\n\n'
        f'> Run the cells in order; every result must come from your own GPU environment — '
        f'never present the sample text as measured data.')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(STYLE),header,*cells],
        metadata={'kernelspec':{'display_name':'Python 3 (ROCm)','language':'python','name':'python3'},
                  'language_info':{'name':'python'},
                  'course':{'version':'1.1-bilingual','lab':fname,'design':'Premium Engineering Keynote'}})
    nbf.validate(nb)
    path=OUT/fname;nbf.write(nb,path)
    print(path)


make('00_ROCm_Environment_Preflight.ipynb','00 / ROCm 环境预检与课程工作目录','00 / ROCm Environment Preflight & Course Workspace',[
 m('''## 学习目标
- 确认当前 Jupyter Kernel 的 **PyTorch 是 ROCm/HIP 版本**，不是 CPU 或 NVIDIA CUDA 版。
- 检查 GPU 设备、Ultralytics 版本和 `/workspace` 持久化路径。
- 保存不含代理凭据的环境快照供实验复现。

**重要：** 不要在已安装 AMD 定制 PyTorch 的环境直接执行 `pip install torch` 或 `pip install -U torch`。如未安装 Ultralytics，请按照 [AMD 官方集成文档](https://docs.ultralytics.com/integrations/amd/) 在匹配的 ROCm 环境中安装。''',
   '''## Learning Objectives
- Verify that the current Jupyter kernel runs a **ROCm/HIP build of PyTorch** — not CPU, not NVIDIA CUDA.
- Inspect the GPU device, the Ultralytics version and the `/workspace` persistence path.
- Save a proxy-credential-free environment snapshot for reproducibility.

**Important:** never run `pip install torch` or `pip install -U torch` inside an environment that already carries the AMD-customized PyTorch. If Ultralytics is missing, follow the [official AMD integration guide](https://docs.ultralytics.com/integrations/amd/) and install it into the matching ROCm environment.'''),
 c(SETUP),
 c('''from pprint import pprint
info=system_info()
info['gpu_device_nodes']={p:Path(p).exists() for p in ('/dev/kfd','/dev/dri')}
info['proxy_configured']={n:bool(os.getenv(n)) for n in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy')}
write_json(home/'reports/environment.json',info)
pprint(info)'''),
 m('''### 理解环境信息
- `torch.version.hip != None` 表示 ROCm PyTorch 构建。
- 在 AMD GPU 上，`torch.cuda.is_available()` 也可以返回 `True`，这是 HIP 对 PyTorch CUDA API 的兼容设计。
- 训练传 `device=0`；**不要**传 `device="rocm"`。
- 云端代理只需要继承平台设置；本课不会输出代理 URL、访问令牌或秘密。''',
   '''### Reading the Environment Info
- `torch.version.hip != None` marks a ROCm build of PyTorch.
- On AMD GPUs `torch.cuda.is_available()` may also return `True` — that is HIP's compatibility layer over the PyTorch CUDA API surface.
- Pass `device=0` for training; **do not** pass `device="rocm"`.
- Cloud proxies only need the platform defaults; this course never prints proxy URLs, tokens or secrets.'''),
 c('''STRICT_ROCM = True  # 教学 GPU 实验应保持 True；不要为了通过检查而改成 False
if STRICT_ROCM:
    device = require_rocm()
else:
    device = 'cpu'
    print('CPU-only diagnostic mode; GPU experiment conclusions are NOT valid.')
print('Device argument for Ultralytics:',device)'''),
 c('''# 可选：实际在 GPU 上计算一个小矩阵，验证张量确实被送入 GPU
import torch
if STRICT_ROCM:
    x=torch.randn((128,128),device='cuda:0')
    y=x@x
    torch.cuda.synchronize()
    assert y.device.type=='cuda'  # ROCm 也报告 cuda device type
    print('HIP kernel smoke: PASS, output shape:',tuple(y.shape))'''),
 m('''### 冻结预训练权重（所有 A/C/E 使用相同模型）
为了避免课程中途的自动下载或不同版本导致 A/B 实验起点不一致，首次联网时缓存 `yolo26n.pt`，后续统一从 `/workspace/yolo_course/artifacts/yolo26n.pt` 读取。缓存成功后记录 SHA256。''',
   '''### Pinning the Pretrained Weights (labs A/C/E share one model)
To keep every A/B experiment starting from an identical point — and to avoid mid-class downloads — cache `yolo26n.pt` on first network use, then always load it from `/workspace/yolo_course/artifacts/yolo26n.pt`. Its SHA256 is recorded once the cache is pinned.'''),
 c('''import shutil
from ultralytics import YOLO
model_cache=home/'artifacts/yolo26n.pt'
if not model_cache.is_file():
    fresh=YOLO('yolo26n.pt')
    source=Path(fresh.ckpt_path).resolve()
    assert source.is_file(),f'Model download not found at {source}'
    if source!=model_cache.resolve():shutil.copy2(source,model_cache)
assert model_cache.is_file()
print('Pinned model:',model_cache,'sha256:',sha256(model_cache))
save_config({'model':'yolo26n.pt','model_sha256':sha256(model_cache)})'''),
 m('''### 课后问题（写进环境报告）
1. 为什么 AMD GPU 使用 `torch.cuda`？
2. `torch.version.hip` 返回 `None` 意味着什么？
3. 为什么不能在已经配好的 ROCm 环境随意覆盖 PyTorch？

**验收文件**：`reports/environment.json`。''',
   '''### After-Class Questions (put answers in your environment report)
1. Why do AMD GPUs use the `torch.cuda` API?
2. What does it mean when `torch.version.hip` returns `None`?
3. Why must you never overwrite PyTorch in an already-configured ROCm environment?

**Acceptance artifact**: `reports/environment.json`.''')
], '10–15 分钟','环境 → GPU 后端 → 持久化工件','10–15 minutes','environment → GPU backend → persistent artifacts')

make('01_Lab_A_Predict_and_Threshold.ipynb','01 / Lab A · 推理、BBox 与置信度阈值','01 / Lab A · Inference, BBoxes & Confidence Thresholds',[
 m('''## 实验目标
给同一张图片设置 `conf=0.10/0.25/0.50`，测量检测数量并保存图片。理解 **Score ≠ 真实正确概率**，置信度阈值只影响显示/保留哪些预测，不会修改模型权重。

**依赖**：先完成 00；预训练模型和示例图片可从官方地址下载（如云端不能访问外网，请提前上传缓存）。''',
   '''## Goal
Run the same image at `conf=0.10/0.25/0.50`, count detections and save the images. Understand that **a score ≠ the true probability of correctness** — the confidence threshold only filters which predictions are kept/displayed; it never changes model weights.

**Prerequisite**: finish 00 first. The pretrained model and the sample image come from official sources (pre-upload them to the cache if your cloud has no external network).'''),
 c(SETUP),
 c('''device=require_rocm()
from ultralytics import YOLO
import ultralytics
from PIL import Image
from IPython.display import display
model_path = home/'artifacts/yolo26n.pt'
model = YOLO(str(model_path) if model_path.is_file() else cfg['model'])
print('Loaded:',model.ckpt_path if hasattr(model,'ckpt_path') else cfg['model'])'''),
 c('''# 优先使用 Ultralytics 包内样例，否则下载官方公交车样例
example=Path(ultralytics.__file__).resolve().parent/'assets/bus.jpg'
if not example.exists():
    example=home/'assets/bus.jpg'
    if not example.exists():
        import urllib.request
        url='https://ultralytics.com/images/bus.jpg'
        try: urllib.request.urlretrieve(url,example)
        except Exception as exc: raise RuntimeError(f'无法下载示例图。请自行上传一张图片至 {example}。原因: {exc}')
assert example.is_file()
display(Image.open(example).resize((480,640)))'''),
 c('''from collections import Counter
thresholds=[0.10,0.25,0.50]
output_dir=home/'assets/lab_a_thresholds';output_dir.mkdir(parents=True,exist_ok=True)
summary=[]
for threshold in thresholds:
    result=model.predict(source=str(example),device=device,conf=threshold,imgsz=640,
                         save=False,verbose=False)[0]
    boxes=result.boxes
    counts=Counter(int(k) for k in boxes.cls.cpu().tolist()) if boxes is not None else Counter()
    image_out=output_dir/f'conf_{threshold:.2f}.jpg'
    Image.fromarray(result.plot()[...,::-1]).save(image_out)
    summary.append({'conf':threshold,'detections':sum(counts.values()),
                    'classes':{model.names[k]:v for k,v in counts.items()},
                    'prediction_image':str(image_out)})
    print(threshold,'predictions:',sum(counts.values()),'->',image_out)
write_json(home/'reports/lab_a_predictions.json',summary)'''),
 c('''# 让学生实际看到 xyxy / conf / cls 三组张量
example_result=model.predict(str(example),device=device,conf=0.25,verbose=False)[0]
if example_result.boxes is not None and len(example_result.boxes)>0:
    print('xyxy:',example_result.boxes.xyxy[:3].cpu().numpy())
    print('confidence:',example_result.boxes.conf[:3].cpu().numpy())
    print('class ID:',example_result.boxes.cls[:3].cpu().numpy())
else: print('此阈值下没有检测目标；属于有效的实验结果。')'''),
 c('''# 练习：亲手计算两个框的交并比（几何示例，不是模型测出的 mAP）
from course.lab_utils import iou_xyxy
box_true=(160,120,480,360)
box_pred=(200,140,500,380)
overlap=iou_xyxy(box_true,box_pred)
print('IoU=',round(overlap,4),'学生可修改 box_pred 并观察变化')
assert 0<=overlap<=1'''),
 c('''for item in summary:
    print(f"conf={item['conf']:.2f}  boxes={item['detections']}")
    display(Image.open(item['prediction_image']).resize((640,480)))
assert all(summary[i]['detections'] >= summary[i+1]['detections'] for i in range(len(summary)-1)), '同模型、同图像下阈值变大，检出数不应增加'
print('Threshold monotonicity: PASS')'''),
 m('''### 思考题 / 面试必答
1. 为什么阈值升高，检测框数通常不会增加？
2. 将 `conf` 从 0.25 改为 0.50，模型参数发生改变了吗？
3. Confidence 0.95 是否保证有 95% 的概率检测正确？为什么？
4. 如果漏检严重，怎样评估降低阈值带来的 FP/Recall 权衡？

**交付**：3 张预测图片 + `reports/lab_a_predictions.json` + 简短结论。''',
   '''### Think / Interview Questions
1. Why does the detection count usually not grow as the threshold rises?
2. Does changing `conf` from 0.25 to 0.50 change any model parameter?
3. Does a confidence of 0.95 mean a 95% chance of being correct? Why not?
4. If recalls are badly missed, how would you evaluate the FP/Recall trade-off of lowering the threshold?

**Deliverables**: 3 prediction images + `reports/lab_a_predictions.json` + a short conclusion.''')
], '20 分钟','图像 → 检测结果 → 参数敏感性','20 minutes','image → detections → parameter sensitivity')

make('02_Lab_B_Dataset_Annotation_QA.ipynb','02 / Lab B · 数据集、标注、分离与 QA','02 / Lab B · Dataset, Annotation, Splits & QA',[
 m('''## 实验目标
- 检查一行 YOLO 标签 `class x_center y_center width height`，五个值中坐标为归一化值。
- 从 COCO128 **原始 128 张图**构造教学用、图片不重叠的 Train/Val/Test；默认只保留 `person/car/chair` 三类，并**不是**快递包裹数据集。
- 检查标签范围、重复图片、边界框与数据集配置。

> **关键科学边界**：官方 COCO128 训练/验证复用同样图片，仅用于冒烟测试。本 Notebook 对其重新分组和划分，尽量防止直接重叠；由于只有 128 张，无法据此宣称工业精度或严格 COCO 基准；语义相似场景泄漏还需人工审查。''',
   '''## Goal
- Parse one YOLO label line `class x_center y_center width height`; the four coordinates are normalized.
- Build a teaching-only, image-disjoint Train/Val/Test from the **original 128 COCO128 images**, keeping only `person/car/chair` — this is **not** a parcel dataset.
- Check label ranges, duplicates, bounding boxes and the dataset config.

> **Key scientific boundary**: official COCO128 reuses the same images for train and val and is for smoke tests only. This notebook re-groups and re-splits them to prevent direct overlap; with just 128 images the result can never be presented as industrial accuracy or a strict COCO benchmark, and semantically similar scenes still need human review.'''),
 m('''### 新的 Lab B：先手工标注，再自动检查
1. 在 Windows/macOS/Linux 本地安装开源 **X-AnyLabeling**（它是独立桌面软件，不要装进 ROCm PyTorch 环境）。
2. 讲师提供已获授权的图片；每名学生先手工绘制至少 5 张图片的矩形框，与同伴交换至少 5 张进行复核。先定义统一的类别顺序：classes.txt 每行一个类别，从 0 开始。
3. 在 X-AnyLabeling 选择 **Export Annotations → Export YOLO Annotations → Detection**，使用 classes.txt，得到按图片名称对应的 YOLO 检测 txt；图片保存在 images/，导出标签放在独立的 labels/ 文件夹。
4. 上传 images/、labels/、classes.txt 至 Radeon Cloud，默认示例目录：/workspace/yolo_course/data/raw/anylabeling/。**先通过人工审核并确认版权/使用权限，再进行导入**。
5. 在下面设置 YOLO_ANYLABELING_EXPORT（可设置环境变量或直接修改代码变量）。只读源文件，另建数据集并生成 train/val/test、split_manifest.csv、data.yaml、QA 报告。已有输出不会被覆盖。
6. 同一视频、同一场景、近重复图片不能跨集合。推荐提供 scenes.csv 两列 image,scene；没有此文件时只执行图像相似性启发式分组，仍需人工检查。

**教学注意**：至少需要 9 张图和 3 个独立场景/图片组；不足时继续用 COCO128 安全演示，不能把无效小数据集伪装成可泛化训练集。导入成功后 Lab C 自动读取新 data.yaml。未提供导出则保留原先 COCO128 默认路径。''',
   '''### Updated Lab B: annotate by hand, then validate
1. Install the free, open-source **X-AnyLabeling** desktop app outside your ROCm Python environment.
2. Annotate at least five licensed images per student with rectangle boxes; cross-review at least five images with a teammate. Define the shared class order using classes.txt (one nonempty name per line, zero-based IDs).
3. Use **Export Annotations → Export YOLO Annotations → Detection** and classes.txt. Keep images/ and exported labels/ separate, with matching image/label stems.
4. Upload images/, labels/, classes.txt to Radeon Cloud (example /workspace/yolo_course/data/raw/anylabeling/) after manual review and permission checks.
5. Set YOLO_ANYLABELING_EXPORT to the export folder. The importer never edits sources; it creates a new train/val/test dataset, manifest, data.yaml and QA report. Existing output is never overwritten.
6. Group images from the same video/scene before splitting. Provide optional scenes.csv with image,scene columns; without it the perceptual duplicate heuristic still requires human review.

**Teaching boundary**: at least nine images and three independent groups are required. Otherwise use the default COCO128 demo. A tiny dataset does not establish industrial generalization. Lab C reuses the successful import via shared config.'''),
 c(SETUP),
 c('''from course.lab_utils import (download_coco128,prepare_coco128_demo,inspect_dataset,
    plot_labels,read_labels,label_errors,list_images)
from course.annotation_import import import_anylabeling_yolo
from IPython.display import display
from matplotlib import pyplot as plt

# Option 1: a pre-split and independently reviewed user dataset
USER_DATA_YAML = os.environ.get('YOLO_USER_DATA_YAML', '').strip()

# Option 2: X-AnyLabeling export: images/, labels/, classes.txt
# Example: /workspace/yolo_course/data/raw/anylabeling
YOLO_ANYLABELING_EXPORT = os.environ.get('YOLO_ANYLABELING_EXPORT', '').strip()
YOLO_ANYLABELING_SCENES = os.environ.get('YOLO_ANYLABELING_SCENES', '').strip()
YOLO_ANYLABELING_OUTPUT_NAME = os.environ.get('YOLO_ANYLABELING_OUTPUT_NAME',
                                                'xanylabeling_student_v1').strip()
if USER_DATA_YAML:
    yaml_path=Path(USER_DATA_YAML).expanduser().resolve()
    if not yaml_path.is_file(): raise FileNotFoundError(yaml_path)
    save_config({'dataset_yaml':str(yaml_path),'dataset_kind':'user_provided'})
    print('Using user-owned dataset:',yaml_path)
elif YOLO_ANYLABELING_EXPORT:
    export=Path(YOLO_ANYLABELING_EXPORT).expanduser().resolve()
    dest=home/'data/processed'/YOLO_ANYLABELING_OUTPUT_NAME
    assert YOLO_ANYLABELING_OUTPUT_NAME and Path(YOLO_ANYLABELING_OUTPUT_NAME).name == YOLO_ANYLABELING_OUTPUT_NAME
    if dest.exists():
        if not (dest/'data.yaml').is_file():
            raise FileExistsError(f'Destination already exists; choose a new output name: {dest}')
        print('Using previously imported dataset; change YOLO_ANYLABELING_OUTPUT_NAME for a new revision:',dest)
    else:
        imported=import_anylabeling_yolo(
            export/'images', export/'labels', export/'classes.txt', dest,
            scene_csv=Path(YOLO_ANYLABELING_SCENES) if YOLO_ANYLABELING_SCENES else None,
            seed=42)
        write_json(home/'reports/lab_b_annotation_import.json',imported)
        print('X-AnyLabeling export imported:',imported)
    yaml_path=dest/'data.yaml'
    save_config({'dataset_yaml':str(yaml_path),
                 'dataset_kind':'X-AnyLabeling_student_reviewed_teaching'})
else:
    # Validated fallback: COCO128 with regrouping, 3 selected COCO categories.
    source=download_coco128(home)
    dest=home/'data/processed/coco128_classroom_3class'
    if (dest/'data.yaml').is_file():
        print('Reusing existing dataset; do not overwrite student corrections automatically.')
    else:
        report=prepare_coco128_demo(source,dest,seed=42)
        print(report)
    yaml_path=dest/'data.yaml'
    save_config({'dataset_yaml':str(yaml_path),'dataset_kind':'COCO128_repartitioned_teaching_only'})
print('Training dataset:',yaml_path)'''),
 c('''qa=inspect_dataset(yaml_path)
print('Class names:',qa['names'])
print('Split stats:',qa['splits'])
print('Errors:',qa['problems'][:30])
print('Near-duplicate review warnings:',qa['warnings'][:10])
write_json(home/'reports/lab_b_dataset_qa.json',{k:v for k,v in qa.items() if k!='sample_paths'})
assert qa['pass'],'发现数据结构错误，请先修复再训练'
print('Dataset structural QA: PASS')'''),
 c('''# 可视化训练标签；此步骤由学生人工复核遮挡、错标、漏标等问题
names=list(qa['names'].values()) if isinstance(qa['names'],dict) else list(qa['names'])
train_imgs=qa['sample_paths']['train']
fig=plot_labels(train_imgs,names,6)
fig.savefig(home/'assets/lab_b_label_gallery.png',dpi=150,bbox_inches='tight')
plt.show()'''),
 c('''# 代码练习：验证无效标签能够被 QA 捕获，不会污染实际数据集
valid='0 0.5 0.5 0.2 0.3\\n'
invalid='0 1.2 0.5 0.2 0.3\\n'
assert not label_errors(valid,len(names))
assert label_errors(invalid,len(names))
print('Label format & invalid-bound tests: PASS')'''),
 c('''# 创建人工审阅清单。不要直接上传敏感/受限图像到公开 GitHub。
review=[{'image':str(img),'label':str(Path(qa['root'])/'labels/train'/f'{img.stem}.txt'),
         'approved':'', 'issue_type':'', 'corrected':'', 'note':''} for img in train_imgs[:20]]
review_path=home/'reports/lab_b_manual_review.csv'
if review_path.exists():
    print('复用已填写的人工检查表，避免重复运行时覆盖:',review_path)
else:
    write_csv(review_path,review,
              ['image','label','approved','issue_type','corrected','note'])
    print('人工检查图片后填写:',review_path)'''),
 m('''### 标注修改实操：可审计，默认不写入
选择一张标注图片和其中一行，将新的标签行填写在下面；只有显式设置 `APPLY_CORRECTION=True` 时才会写入，且先备份旧标签。**请先核对人工真值，不可因为预测结果不合预期就任意修改真值标签。**''',
   '''### Hands-on Annotation Correction: auditable, off by default
Pick an annotated image and one label line, then type the corrected line below. Nothing is written unless you explicitly set `APPLY_CORRECTION=True`, and the old label is always backed up first. **Verify against human ground truth first — never bend the truth labels just because predictions disappoint you.**'''),
 c('''from course.lab_utils import label_path_for_image
import datetime
TARGET_IMAGE = train_imgs[0]
LABEL_LINE_INDEX = 0  # 一行一个目标，从 0 开始
NEW_LABEL_LINE = '0 0.500000 0.500000 0.200000 0.200000'  # 教学占位，不应盲目写入
CORRECTION_REASON = 'TODO: 请写清楚人工审核依据'
APPLY_CORRECTION = False  # 先通过可视化核对，再显式设为 True
label_file=label_path_for_image(TARGET_IMAGE)
original_lines=label_file.read_text(encoding='utf-8').splitlines()
print('编辑目标:',label_file)
print('原始标签:',original_lines[:3])
if APPLY_CORRECTION:
    assert CORRECTION_REASON and not CORRECTION_REASON.startswith('TODO:')
    assert 0 <= LABEL_LINE_INDEX < len(original_lines), '该图可能是合法的空标签，需要先选择有框的图片'
    assert not label_errors(NEW_LABEL_LINE,len(names))
    old_line=original_lines[LABEL_LINE_INDEX]
    backup=home/'artifacts/label_backups'/f'{label_file.stem}_{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.txt'
    backup.parent.mkdir(parents=True,exist_ok=True)
    backup.write_text(label_file.read_text(encoding='utf-8'),encoding='utf-8')
    original_lines[LABEL_LINE_INDEX]=NEW_LABEL_LINE
    label_file.write_text('\\n'.join(original_lines)+'\\n',encoding='utf-8')
    write_json(home/'reports/lab_b_last_correction.json',
       {'image':str(TARGET_IMAGE),'label':str(label_file),'before':old_line,'after':NEW_LABEL_LINE,
        'reason':CORRECTION_REASON,'backup':str(backup)})
    assert inspect_dataset(yaml_path)['pass'], '修改后数据集 QA 失败，请恢复备份'
    print('修正已写入，备份：',backup)
else:
    print('安全预览：未修改任何文件。')'''),
 m('''### 思考题 / 面试必答
1. 为什么 Train/Val/Test 必须隔离？同一视频相邻帧能否分到不同集合？
2. 标签缺失与空标签有何区别？哪些空标签是合理的负样本？
3. 为什么 COCO128 不能直接用于可信的模型排行榜成绩？
4. 哪些错误属于标注规范问题而非模型算法问题？

**交付**：`data/processed/.../data.yaml`、`split_manifest.csv`、`dataset_manifest.json`、QA JSON、人工检查表、标注图。''',
   '''### Think / Interview Questions
1. Why must Train/Val/Test stay isolated? May adjacent frames of one video land in different splits?
2. Missing label file vs empty label — what is the difference? Which empty labels are legitimate negatives?
3. Why can't COCO128 directly produce trustworthy leaderboard scores?
4. Which errors are annotation-spec problems rather than model problems?

**Deliverables**: `data/processed/.../data.yaml`, `split_manifest.csv`, `dataset_manifest.json`, the QA JSON, the manual review sheet, annotated images.''')
], '50–60 分钟','手工标注 → 导出 → 防泄漏划分 → QA','50–60 minutes','manual labeling → export → grouped splits → QA')

make('03_Lab_C_ROCm_Training_Finetune.ipynb','03 / Lab C · ROCm GPU 训练、微调与 Loss','03 / Lab C · ROCm GPU Training, Fine-tuning & Loss',[
 m('''## 实验目标
1. 在真实 AMD ROCm GPU 上运行 YOLO26n 预训练模型微调。
2. 先用 COCO8（4 train + 4 val）做**训练链路冒烟**，再训练 Lab B 数据。
3. 保存运行条件、`best.pt`、`last.pt`、训练曲线与 SHA256。
4. 解释 Loss、Epoch、Batch、学习率、过拟合和泛化；不得把 Smoke 成绩说成工业精度。

**注意**：本实验需要 GPU 和时间。推荐课前缓存模型/数据。`amp=False` 仅作 ROCm AMP 异常排查，不应在没有问题时随意更改对照配置。''',
   '''## Goal
1. Fine-tune the YOLO26n pretrained model on a real AMD ROCm GPU.
2. **Smoke-test the training pipeline** on COCO8 (4 train + 4 val) first, then train on the Lab B dataset.
3. Persist run conditions, `best.pt`, `last.pt`, training curves and SHA256 hashes.
4. Explain loss, epochs, batches, learning rate, overfitting and generalization; never dress smoke-run numbers up as industrial accuracy.

**Note**: this lab needs a GPU and time; pre-caching model/data before class is recommended. `amp=False` exists for ROCm AMP troubleshooting only — do not flip comparison configs without a diagnosed problem.'''),
 c(SETUP),
 c('''from course.lab_utils import get_metrics_summary
import torch, time
from ultralytics import YOLO
device=require_rocm()
assert cfg['dataset_yaml'] and Path(cfg['dataset_yaml']).is_file(), '先完成 02 Lab B，或设置 YOLO_USER_DATA_YAML'
model_initial=str(home/'artifacts/yolo26n.pt') if (home/'artifacts/yolo26n.pt').is_file() else cfg['model']
print('Training:',model_initial,'dataset:',cfg['dataset_yaml'])'''),
 c('''# 非正式数据集，验证 GPU 实际能跑 Forward/Backward，不用于发布精度成绩。
RUN_SMOKE=True
if RUN_SMOKE:
    smoke_model=YOLO(model_initial)
    smoke_model.train(data='coco8.yaml',epochs=1,imgsz=320,batch=4,device=device,
                      workers=0,amp=False,project=str(home/'runs/lab_c'),name='smoke',
                      exist_ok=True,plots=False,seed=42)
    print('GPU smoke completed; COCO8 is not a benchmark.')'''),
 c('''# 正式教学训练。为了让学生在课上完成闭环，默认 12 Epoch；高质量结果需课后延长训练。
RUN_BASELINE=True
BASELINE_EPOCHS=12
train_kwargs=dict(data=cfg['dataset_yaml'],epochs=BASELINE_EPOCHS,
                  imgsz=512,batch=8,device=device,workers=0,patience=30,
                  optimizer='SGD',seed=42,deterministic=False,
                  amp=False,plots=True,project=str(home/'runs/lab_c'),
                  name='baseline',exist_ok=True)
if RUN_BASELINE:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    baseline=YOLO(model_initial)
    train_output=baseline.train(**train_kwargs)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    peak_gib=torch.cuda.max_memory_allocated()/2**30
    run_dir=Path(baseline.trainer.save_dir).resolve()
    weights=run_dir/'weights/best.pt'
    assert weights.exists(),f'best.pt missing: {weights}'
    record={'run_dir':str(run_dir),'weights':str(weights),'sha256':sha256(weights),
            'elapsed_seconds':round(elapsed,2),'peak_allocated_GiB':round(peak_gib,3),
            'args':train_kwargs,'system':system_info(),
            'warning':'Peak memory is PyTorch allocated memory, not total device VRAM.'}
    write_json(home/'reports/lab_c_baseline.json',record)
    save_config({'baseline_run':str(run_dir),'baseline_weights':str(weights),
                 'baseline_train_settings':train_kwargs})
    print(record)
else:
    print('未执行训练。后续实验需要一个经过验证的 best.pt，请不要伪造训练结果。')'''),
 c('''# 阅读训练工件：不同版本可能输出不同的列名，请以实际 CSV 为准。
from PIL import Image
import pandas as pd
cfg=load_config()
run_dir=Path(cfg['baseline_run'])
csv_path=run_dir/'results.csv'
if csv_path.exists():
    history=pd.read_csv(csv_path)
    display(history.tail(5))
else:
    print('Missing results.csv:',csv_path)
if (run_dir/'results.png').exists():
    display(Image.open(run_dir/'results.png').resize((1120,530)))'''),
 c('''# 不混淆验证集与最终测试集：首次使用 split='val'
cfg=load_config()
best=YOLO(cfg['baseline_weights'])
metrics=best.val(data=cfg['dataset_yaml'],split='val',device=device,
                 imgsz=640,batch=8,workers=0,plots=True,verbose=False)
summary=get_metrics_summary(metrics)
summary['split']='val';summary['model_sha256']=sha256(Path(cfg['baseline_weights']))
write_json(home/'reports/lab_c_baseline_val.json',summary)
print('Validation only (NOT test):',summary)'''),
 m('''### 为什么基线固定训练用 `imgsz=512`，评测用 `imgsz=640`？
训练输入尺寸属于训练方案，下一步 Lab E 会只改变训练尺寸；为了公平比较两组模型，Lab E 会用同一个 `imgsz=640` 做评测。这是一种课堂控制变量方案，不是所有工业评测任务必须如此。

### 思考题 / 面试必答
1. `best.pt` 与 `last.pt` 有什么区别？
2. 为什么训练 Loss 下降不能证明模型在新场景一定更好？
3. `torch.version.hip` 与 `torch.cuda.is_available()` 分别证明什么？
4. 如果 AMP 检查/训练有 NaN，你会怎样排查并记录配置变动？

**交付**：`runs/lab_c/baseline/`、`reports/lab_c_baseline.json`、`lab_c_baseline_val.json`。''',
   '''### Why does the baseline train at `imgsz=512` but evaluate at `imgsz=640`?
Training input size is part of the training recipe — Lab E will vary exactly this knob. To compare both arms fairly, Lab E evaluates every model with the same `imgsz=640`. This is a classroom controlled-variable design, not a universal industrial rule.

### Think / Interview Questions
1. What is the difference between `best.pt` and `last.pt`?
2. Why doesn't a falling training loss prove the model is better on new scenes?
3. What do `torch.version.hip` and `torch.cuda.is_available()` each prove?
4. If the AMP check or training produces NaN, how do you troubleshoot and record the config change?

**Deliverables**: `runs/lab_c/baseline/`, `reports/lab_c_baseline.json`, `lab_c_baseline_val.json`.''')
], '50–70 分钟','预训练权重 → 前后向训练 → 模型工件','50–70 minutes','pretrained weights → forward/backward training → model artifacts')

make('04_Lab_D_Evaluation_Error_Analysis.ipynb','04 / Lab D · mAP、混淆与失败案例','04 / Lab D · mAP, Confusion & Failure Cases',[
 m('''## 实验目标
- 用 Ultralytics 官方验证器计算 Precision、Recall、mAP50、mAP50-95。
- 独立完成一次**固定 `conf=0.25`、IoU=0.50 的教学级 TP/FP/FN 匹配**（并非 COCO mAP 实现）。
- 查看 PR/混淆矩阵并生成至少五份真实失败案例。
- 不用 Test 集选择模型或调阈值。

`metrics.box.map` 才是官方 mAP50-95；我们的 TP/FP/FN 固定阈值分析只用于学习错误原因。''',
   '''## Goal
- Compute Precision, Recall, mAP50 and mAP50-95 with the official Ultralytics validator.
- Independently run a **fixed-threshold `conf=0.25`, IoU=0.50 teaching-level TP/FP/FN matching** (this is NOT the COCO mAP algorithm).
- Inspect PR/confusion matrices and produce at least five genuine failure cases.
- Never use the test split to pick models or tune thresholds.

`metrics.box.map` is the official mAP50-95; our fixed-threshold TP/FP/FN analysis exists only to learn about error causes.'''),
 c(SETUP),
 c('''from course.lab_utils import (get_metrics_summary, inspect_dataset, match_predictions,
                                  xywhn_to_xyxy, read_labels, label_path_for_image)
from ultralytics import YOLO
from PIL import Image,ImageDraw
from IPython.display import display
import torch
device=require_rocm()
assert cfg['baseline_weights'] and Path(cfg['baseline_weights']).is_file(), '先完成 Lab C，或经过核验设置 best.pt'
assert Path(cfg['dataset_yaml']).is_file()
model=YOLO(cfg['baseline_weights'])
print('Weight SHA256:',sha256(Path(cfg['baseline_weights'])))'''),
 c('''# 正式官方 mAP 计算：不要把 conf=0.25 当作计算 mAP 的唯一阈值
val_metrics=model.val(data=cfg['dataset_yaml'],split='val',device=device,
                      imgsz=640,batch=8,workers=0,plots=True,verbose=False)
score=get_metrics_summary(val_metrics)
score.update({'split':'val','dataset_yaml':cfg['dataset_yaml'],
              'weight_sha256':sha256(Path(cfg['baseline_weights']))})
write_json(home/'reports/lab_d_official_metrics.json',score)
print('Official validation metrics:',score)
print('mAP: IoU=0.50..0.95 average, NOT confidence 50%-95%')'''),
 c('''# 展示 Ultralytics 官方可视化（如果已生成）
from PIL import Image
out=Path(val_metrics.save_dir)
for title in ('BoxPR_curve.png','PR_curve.png','confusion_matrix.png','confusion_matrix_normalized.png'):
    p=out/title
    if p.exists():
        print(title)
        display(Image.open(p).resize((950,600)))'''),
 c('''# 手工逐图片匹配：一个 GT 最多配一个同类、IoU>=0.50 的预测，其他为 FP/FN
qa=inspect_dataset(Path(cfg['dataset_yaml']))
val_images=qa['sample_paths']['val']
rows=[]; failures=[];totals={'TP':0,'FP':0,'FN':0}
for img in val_images:
    pred=model.predict(source=str(img),device=device,imgsz=640,
                       conf=0.25,iou=0.70,verbose=False)[0]
    gt=label_path_for_image(img)
    match=match_predictions(pred,gt,iou_threshold=0.50)
    for k in totals:totals[k]+=match[k]
    for issue in match['errors']:
        rows.append({'image':img.name,'image_path':str(img),'error':issue['error'],
                     'class_id':issue['class_id'],'score':issue['score'],'max_iou_gt':issue['max_iou_gt']})
    if match['errors']:
        failures.append((img,pred,gt,match))
precision=totals['TP']/max(1,totals['TP']+totals['FP'])
recall=totals['TP']/max(1,totals['TP']+totals['FN'])
print('Fixed threshold teaching-level counts:',totals, 'P:',round(precision,3),'R:',round(recall,3))
print('Important: these are NOT the official integrated PR or mAP values.')
write_csv(home/'reports/lab_d_errors.csv',rows,
          ['image','image_path','error','class_id','score','max_iou_gt'])
write_json(home/'reports/lab_d_fixed_threshold.json',{'conf':0.25,'matching_iou':0.50,
 'nms_iou':0.70,'counts':totals,'precision_fixed':precision,'recall_fixed':recall,
 'note':'Teaching matching only; not a COCO AP implementation.'})'''),
 c('''# 保存真实失败图（左真值，右预测）；如果几乎无错误，忠实报告而不是捏造 5 个。
from PIL import Image,ImageDraw
error_dir=home/'assets/lab_d_errors';error_dir.mkdir(parents=True,exist_ok=True)
names=list(qa['names'].values()) if isinstance(qa['names'],dict) else list(qa['names'])
for idx,(img,pred,lab,match) in enumerate(failures[:5],1):
    gt=Image.open(img).convert('RGB')
    W,H=gt.size; d=ImageDraw.Draw(gt)
    for box in read_labels(lab):
        coords=xywhn_to_xyxy(box,W,H)
        d.rectangle(coords,outline=(40,210,120),width=max(2,W//160))
        d.text((coords[0],max(0,coords[1]-15)),f'GT {names[box[0]]}',fill=(0,170,85))
    predicted=Image.fromarray(pred.plot()[...,::-1]).convert('RGB')
    board=Image.new('RGB',(W*2,H),(245,248,250));board.paste(gt,(0,0));board.paste(predicted,(W,0))
    p=error_dir/f'{idx:02d}_{img.stem}.jpg';board.save(p,quality=90)
    print(p,'errors:',len(match['errors']))
    display(board.resize((960,round(960*H/(2*W)))))
if len(failures)<5:
    print(f'只有 {len(failures)} 张有失败案例。不能编造；可拓展验证集或评估更多困难样本。')'''),
 m('''### 学生工作：手动归因至少 5 个实际错误（如果有）
打开 `reports/lab_d_errors.csv` 和并排对照图，添加 **现象、证据、可能根因、下次实验验证方法**。把推测与已确认根因区分开。

### 思考题 / 面试必答
1. 为什么官方 mAP50 和单一 conf=0.25 下的 Precision 不是同一个东西？
2. IoU=0.5 的定位标准会漏掉什么信息？
3. `mAP50-95` 的 50–95 指置信度吗？
4. 为什么不能用测试集调阈值后再报告该测试集成绩？

**交付**：`lab_d_official_metrics.json`、`lab_d_fixed_threshold.json`、`lab_d_errors.csv`、至少现有真实失败图和分析报告。''',
   '''### Student Work: manually attribute at least 5 real errors (if any)
Open `reports/lab_d_errors.csv` and the side-by-side boards, then add **symptom, evidence, suspected root cause, and how the next experiment would verify it**. Keep hypotheses clearly separate from confirmed causes.

### Think / Interview Questions
1. Why is the official mAP50 not the same thing as Precision at a single conf=0.25?
2. What information does an IoU=0.5 localization criterion throw away?
3. Does the 50–95 in `mAP50-95` refer to confidence?
4. Why can't you tune thresholds on a test set and then report scores on that same set?

**Deliverables**: `lab_d_official_metrics.json`, `lab_d_fixed_threshold.json`, `lab_d_errors.csv`, the real failure images so far and an analysis report.''')
], '35–45 分钟','官方指标 → 固定阈值配对 → 错误归因','35–45 minutes','official metrics → fixed-threshold matching → error attribution')

make('05_Lab_E_Controlled_AB_Optimization.ipynb','05 / Lab E · 受控 A/B 训练优化实验','05 / Lab E · Controlled A/B Training Experiment',[
 m('''## 实验目标
基于 Lab C 的**真实 baseline**，只修改训练图像尺寸（512 → 416）；保持模型初始化权重、数据集、Epoch、Batch、随机种子等条件不变，用统一的验证输入尺寸 640 测量两组模型。不要因为速度更快就直接宣布算法优化成功。

> **公平性说明：** 降低 imgsz 会改变输入信息量和计算量。固定相同 Epoch 不保证相同训练时长 / SGD 更新质量；本实验验证一个明确的课程假设，不提供统计显著性证明。训练时间还可能受缓存及云端 GPU 并发负载影响。''',
   '''## Goal
Starting from the **real baseline produced by Lab C**, change exactly one thing — the training image size (512 → 416). Keep initial weights, dataset, epochs, batch, random seed and everything else identical, then measure both models with the same validation input size of 640. Never declare an optimization successful just because it runs faster.

> **Fairness note:** lowering imgsz changes both the information content and the compute of each step. Equal epochs do not guarantee equal training time / SGD update quality; this experiment tests one clearly stated classroom hypothesis and provides no statistical significance. Training time is also affected by caches and concurrent load on shared cloud GPUs.'''),
 c(SETUP),
 c('''from course.lab_utils import get_metrics_summary
import torch,time
from ultralytics import YOLO
device=require_rocm()
assert cfg.get('baseline_train_settings') and Path(cfg['baseline_weights']).is_file(),'请先完整运行 Lab C baseline。'
base_record=json.loads((home/'reports/lab_c_baseline.json').read_text(encoding='utf-8'))
args=cfg['baseline_train_settings']
print('Baseline train imgsz:',args['imgsz'],'epochs:',args['epochs'])
print('Baseline model sha256:',sha256(Path(cfg['baseline_weights'])))
model_cache=home/'artifacts/yolo26n.pt'
assert model_cache.is_file(), '预训练初始权重缓存不存在；重新运行 00 进行固定。'
assert sha256(model_cache)==cfg.get('model_sha256'), '初始权重 SHA256 改变，A/B 结果不可比。'
assert args['imgsz']==512,'此 Notebook 的实验方案假定 baseline 训练尺寸为 512，请先审核与修改方案' '''),
 c('''# Variant：只修改 imgsz；额外不同的仅是 run output name/project
RUN_VARIANT=True
variant_args=dict(args)
variant_args.update(imgsz=416,project=str(home/'runs/lab_e'),name='imgsz416',exist_ok=True)
changed={key:(args[key],variant_args[key]) for key in args if args[key]!=variant_args[key]}
print('Changed training arguments:',changed)
assert set(changed)=={'imgsz','project','name'}
if RUN_VARIANT:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    init_model=str(home/'artifacts/yolo26n.pt') if (home/'artifacts/yolo26n.pt').is_file() else cfg['model']
    variant=YOLO(init_model)  # 注意：不能用 baseline best.pt 作为 B 的初始权重！
    variant.train(**variant_args)
    torch.cuda.synchronize()
    run_dir=Path(variant.trainer.save_dir)
    variant_best=run_dir/'weights/best.pt'
    assert variant_best.is_file()
    variant_record={'run_dir':str(run_dir),'weights':str(variant_best),
       'sha256':sha256(variant_best),'elapsed_seconds':round(time.perf_counter()-t0,2),
       'peak_allocated_GiB':round(torch.cuda.max_memory_allocated()/2**30,3),
       'args':variant_args,'system':system_info()}
    write_json(home/'reports/lab_e_variant.json',variant_record)
    print('Variant training finished:',variant_record)
else:
    print('RUN_VARIANT=False: without real variant artifacts NO A/B conclusion can be made.')'''),
 c('''# 相同验证集、验证输入和运行条件，分别评测真实权重。
variant_record=json.loads((home/'reports/lab_e_variant.json').read_text(encoding='utf-8'))
assert Path(variant_record['weights']).exists()
reports=[]
for label,rec in [('baseline',base_record),('imgsz416',variant_record)]:
    m=YOLO(rec['weights']).val(data=cfg['dataset_yaml'],split='val',
                  imgsz=640,device=device,batch=8,workers=0,plots=False,verbose=False)
    scores=get_metrics_summary(m)
    scores.update({'variant':label,'train_imgsz':rec['args']['imgsz'],
      'training_time_seconds':rec['elapsed_seconds'],'train_peak_allocated_gib':rec['peak_allocated_GiB'],
      'model_sha256':rec['sha256'],'val_imgsz':640,'split':'val'})
    reports.append(scores)
    print(label,scores)
write_csv(home/'reports/lab_e_ab_comparison.csv',reports,
          ['variant','train_imgsz','training_time_seconds','train_peak_allocated_gib',
           'precision','recall','mAP50','mAP50_95','model_sha256','val_imgsz','split'])
write_json(home/'reports/lab_e_ab_comparison.json',reports)'''),
 c('''# 图表使用实测数值，不预置任何提升率
from matplotlib import pyplot as plt
labels=[r['variant'] for r in reports]
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(labels,[r['mAP50_95'] for r in reports])
ax.set_ylabel('Validation mAP50-95');ax.set_ylim(0,1)
ax.set_title('Same validation images and evaluation imgsz=640')
for i,r in enumerate(reports):ax.text(i,r['mAP50_95']+.01,f"{r['mAP50_95']:.3f}",ha='center')
fig.tight_layout()
fig.savefig(home/'assets/lab_e_ab_mAP.png',dpi=150)
plt.show()
print('ΔmAP50-95 B−A:',round(reports[1]['mAP50_95']-reports[0]['mAP50_95'],4))'''),
 m('''### 最终 Test：仅在选定模型之后运行一次
只使用 Validation 指标决定选择哪个候选模型。不要在查看 Test 分数后反过来改动超参数，再把同一 Test 分数当作独立最终成绩。''',
   '''### The final Test: run exactly once, after model selection
Use validation metrics alone to pick the winning candidate. Never peek at the test score, go back and tweak hyperparameters, and then present that same test score as an independent final result.'''),
 c('''RUN_FINAL_TEST = False  # 小组完成模型选择并冻结方案后，明确改成 True 执行一次
if RUN_FINAL_TEST:
    target=home/'reports/lab_e_final_test.json'
    assert not target.exists(), '独立 Test 已有成绩。避免重复查看后调参。'
    selection=max(reports,key=lambda row:row['mAP50_95'])  # 仅由 Val 指标选择
    test_weight=base_record['weights'] if selection['variant']=='baseline' else variant_record['weights']
    final_test=YOLO(test_weight).val(data=cfg['dataset_yaml'],split='test',imgsz=640,
                                    device=device,batch=8,workers=0,plots=False,verbose=False)
    result=get_metrics_summary(final_test)
    result.update({'split':'test','selected_by':'validation mAP50-95',
                   'model_variant':selection['variant'],'model_sha256':sha256(Path(test_weight)),
                   'warning':'Tiny classroom split; not a statistically reliable industrial benchmark.'})
    write_json(target,result)
    print('Frozen final test result:',result)
else:
    print('Test is sealed. Current metrics are Validation only.')'''),
 m('''### 结论必须回答五个问题
1. 只改了哪一个**算法实验变量**？`project/name` 是日志位置变化，不是算法变量。
2. 在**完全一致的验证集和评测输入尺寸**下，mAP50-95 是增加还是减少？
3. 哪组训练耗时更短？同一云端 GPU 是否有其他用户负载干扰？
4. 显存数字是 PyTorch allocated peak，不是整卡显存占用。
5. 这个结果能否推广到更大模型、不同硬件和工业现场？为什么不能直接断言？

**交付**：`lab_e_variant.json`、`lab_e_ab_comparison.csv`、`lab_e_ab_mAP.png`、文字分析。

**进阶**：用多个随机种子重复实验；在真正独立的业务数据上验证；在第二天再比较 ONNX/OpenVINO、GPU/ROCm 推理速度、显存与准确性。''',
   '''### Your conclusion must answer five questions
1. Which **single experimental variable** changed? `project/name` are log locations, not algorithm variables.
2. Under **identical validation data and evaluation input size**, did mAP50-95 go up or down?
3. Which arm trained faster? Was any other user loading the same cloud GPU?
4. The memory figure is PyTorch's allocated peak, not whole-GPU VRAM usage.
5. Can this result generalize to larger models, other hardware and production sites? Why can't you just assert it?

**Deliverables**: `lab_e_variant.json`, `lab_e_ab_comparison.csv`, `lab_e_ab_mAP.png`, written analysis.

**Going further**: repeat with multiple random seeds; validate on truly independent business data; on day 2 compare ONNX/OpenVINO vs GPU/ROCm inference speed, memory and accuracy.''')
], '30–40 分钟（完整训练可能更久）','控制变量 → 重新训练 → 同口径比较','30–40 minutes (full training may take longer)','controlled variable → retrain → same-protocol comparison')
