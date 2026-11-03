"""Generate the Day 1 teaching notebooks with repeatable metadata and code.

Bilingual edition: every teaching markdown cell is emitted as Chinese first,
then an English translation after a `---` divider. Code cells are semantically
identical to the validated v1.0 run — restyled to PEP 8 (spacing, one
statement per line, wrapped long lines) without renaming identifiers or
changing any literal, call or control flow. Cross-cell variables (e.g. `qa`,
`summary`, `reports`) are the implicit cell-to-cell interface and keep their
names. Semantic parity is proven by the canonical-AST check committed with the
repo (tests/test_notebook_ast_reference.py).
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
SETUP = '''from pathlib import Path
import os
import sys
import json

candidates = [Path.cwd(), Path.cwd().parent, Path('/workspace/YOLO26_ROCm_Day1_Labs')]
package = next((p for p in candidates if (p / 'course/lab_utils.py').exists()), None)
if package is None:
    raise FileNotFoundError('请先将 ZIP 完整解压，并从项目目录或 notebooks/ 目录启动 Jupyter。')
sys.path.insert(0, str(package))
from course.lab_utils import (root, ensure_dirs, load_config, save_config, write_json,
                              write_csv, system_info, require_rocm, sha256)

home = ensure_dirs()
cfg = load_config()
print('Project:', package, '\\nPersistent output:', home)
'''


def m(zh: str, en: str = ''):
    """Markdown cell: Chinese content, then the English translation below a divider."""
    text = dedent(zh).strip()
    if en:
        text += LANG_DIVIDER + dedent(en).strip()
    return nbf.v4.new_markdown_cell(text)


def c(text):return nbf.v4.new_code_cell(dedent(text).strip())


def lab_map(rows):
    """Compact bilingual 'lab map': the seven questions every notebook must
    answer up front (why / prerequisites / what to change / expected success /
    failure self-help / evidence to save / job value). Rows are (zh, en) pairs;
    the Chinese table comes first, the English translation after the divider."""
    zh = ['### 学习地图（30 秒看完这张表再开始）', '',
          '| 问题 | 答案 |', '|---|---|']
    en = ['### Lab Map (read this 30-second table first)', '',
          '| Question | Answer |', '|---|---|']
    for q_zh, a_zh, q_en, a_en in rows:
        zh.append(f'| {q_zh} | {a_zh} |')
        en.append(f'| {q_en} | {a_en} |')
    return m('\n'.join(zh), '\n'.join(en))


def make(fname, title_zh, title_en, cells, duration_zh, duration_en, tags_zh, tags_en):
    # Guard against transposing the duration/learning-path arguments (a real
    # defect shipped in v1.2): durations must look like durations, paths must not.
    import re
    _time_like = re.compile(r'分钟|minutes|min', re.IGNORECASE)
    assert _time_like.search(duration_zh) and _time_like.search(duration_en), \
        f'{fname}: duration args must contain a time unit: {duration_zh!r} / {duration_en!r}'
    assert not _time_like.search(tags_zh) and not _time_like.search(tags_en), \
        f'{fname}: learning-path args must not contain a time unit: {tags_zh!r} / {tags_en!r}'
    header = m(
        f'# {title_zh}\n\n**建议时长：{duration_zh}**　｜　学习路径：{tags_zh}\n\n'
        f'> 请按顺序运行；运行结果必须来自自己的 GPU 环境，不可把示例文字当作实测数据。',
        f'# {title_en}\n\n**Suggested time: {duration_en}** | Learning path: {tags_en}\n\n'
        f'> Run the cells in order; every result must come from your own GPU environment — '
        f'never present the sample text as measured data.')
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(STYLE),header,*cells],
        metadata={'kernelspec':{'display_name':'Python 3 (ROCm)','language':'python','name':'python3'},
                  'language_info':{'name':'python'},
                  'course':{'version':'1.2-bilingual-pep8','lab':fname,'design':'Premium Engineering Keynote'}})
    nbf.validate(nb)
    path=OUT/fname;nbf.write(nb,path)
    print(path)


make('00_ROCm_Environment_Preflight.ipynb','00 / ROCm 环境预检与课程工作目录','00 / ROCm Environment Preflight & Course Workspace',[
 lab_map([
  ('为什么学这个？','环境是全部实验的地基——后面 5 个实验都靠它，错了全白跑',
   'Why learn this?','The environment is the foundation of all five later labs; a broken one wastes everything'),
  ('需要先懂什么？','会点 Jupyter 运行按钮即可，概念现场解释',
   'Prerequisites?','Being able to run a Jupyter cell; concepts are explained on the spot'),
  ('我修改哪个参数？','练习格的 `SMOKE_N` 与 `MY_PREDICTION`（先预测再运行）',
   'What do I change?','`SMOKE_N` and `MY_PREDICTION` in the practice cell (predict first, then run)'),
  ('成功长什么样？','`torch_hip` 非空、GPU 名字打印、`HIP kernel smoke: PASS`',
   'What does success look like?','Non-empty `torch_hip`, your GPU name printed, `HIP kernel smoke: PASS`'),
  ('失败怎么自救？','本页"常见错误与下一步"表 → `docs/TROUBLESHOOTING.md`',
   'If it fails?','The common-errors table on this page → `docs/TROUBLESHOOTING.md`'),
  ('保存什么证据？','`reports/environment.json` + 权重 SHA256',
   'Evidence to save?','`reports/environment.json` + the weight SHA256'),
  ('求职价值？','"给我一台 AMD GPU 服务器，我 5 分钟验明环境"——运维/算法岗都考',
   'Job value?','"Hand me an AMD GPU server and I will verify the stack in 5 minutes"'),
 ]),
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

info = system_info()
info['gpu_device_nodes'] = {p: Path(p).exists() for p in ('/dev/kfd', '/dev/dri')}
info['proxy_configured'] = {n: bool(os.getenv(n)) for n in ('HTTP_PROXY', 'HTTPS_PROXY',
                                                            'http_proxy', 'https_proxy')}
write_json(home / 'reports/environment.json', info)
pprint(info)'''),
 m('''### 理解环境信息
- `torch.version.hip != None` 表示 ROCm PyTorch 构建。
- 在 AMD GPU 上，`torch.cuda.is_available()` 也可以返回 `True`，这是 HIP 对 PyTorch CUDA API 的兼容设计。
- 训练传 `device=0`；**不要**传 `device="rocm"`。
- 云端代理只需要继承平台设置；本课不会输出代理 URL、访问令牌或秘密。

**依赖**：这是第一份 Notebook，无前置实验；只需按部署文档建好 ROCm 环境。

### 预期输出（正常长什么样）
- `environment.json` 打印出的字典里：`torch_hip` 是一串版本号（非空）、`gpu_name` 是你的 AMD 显卡名、`gpu_device_nodes` 两项都是 `True`。
- ROCm 检查格打印 `AMD ROCm verified: <你的显卡名> HIP: <版本>`。
- HIP 冒烟格打印 `HIP kernel smoke: PASS, output shape: (128, 128)`。
- 若与上述形状不符（例如 `torch_hip` 为 `None` 或 GPU 不可见），先看下面的常见错误，不要继续后面的实验。

### 常见错误与下一步
| 症状 | 原因 | 下一步 |
|---|---|---|
| `torch.version.hip` 为 `None` | 当前 Kernel 不是 ROCm 构建（CPU/CUDA wheel） | 切换到课程注册的 Kernel；重装见部署文档 |
| `no GPU visible` / `/dev/kfd` 不存在 | GPU 未分配或节点不可见 | 检查平台 GPU 分配；Linux 看 `/dev/kfd`、`/dev/dri` |
| `torchvision GPU NMS is unavailable` | 平台预装 torchvision 缺 GPU NMS 内核 | 按部署文档方式 B 自建 uv 隔离栈 |
| `pip install torch` 后环境损坏 | 覆盖了 AMD 定制 PyTorch | 重建环境，永远不要在课程环境里装默认 torch |

更多排障见 `docs/TROUBLESHOOTING.md`。''',
   '''### Reading the Environment Info
- `torch.version.hip != None` marks a ROCm build of PyTorch.
- On AMD GPUs `torch.cuda.is_available()` may also return `True` — that is HIP's compatibility layer over the PyTorch CUDA API surface.
- Pass `device=0` for training; **do not** pass `device="rocm"`.
- Cloud proxies only need the platform defaults; this course never prints proxy URLs, tokens or secrets.

**Prerequisite**: this is the first notebook — no prior lab; only a working ROCm environment from the setup guide.

### Expected Output (what healthy looks like)
- In the printed `environment.json` dict: `torch_hip` is a non-empty version string, `gpu_name` is your AMD GPU, and both `gpu_device_nodes` entries are `True`.
- The ROCm gate cell prints `AMD ROCm verified: <your GPU> HIP: <version>`.
- The HIP smoke cell prints `HIP kernel smoke: PASS, output shape: (128, 128)`.
- If the shape differs (e.g. `torch_hip` is `None` or no GPU), read the common errors below before continuing.

### Common Errors & Next Steps
| Symptom | Cause | Next step |
|---|---|---|
| `torch.version.hip` is `None` | kernel is not a ROCm build (CPU/CUDA wheel) | switch to the registered course kernel; reinstall per the setup guide |
| `no GPU visible` / missing `/dev/kfd` | GPU not allocated or nodes hidden | check platform GPU assignment; on Linux inspect `/dev/kfd`, `/dev/dri` |
| `torchvision GPU NMS is unavailable` | platform torchvision lacks the GPU NMS kernel | build the uv-isolated stack (setup guide, option B) |
| environment broken after `pip install torch` | AMD-customized PyTorch overwritten | rebuild the env; never install default torch inside it |

Full troubleshooting: `docs/TROUBLESHOOTING.md`.'''),
 c('''STRICT_ROCM = True  # 教学 GPU 实验应保持 True；不要为了通过检查而改成 False
if STRICT_ROCM:
    device = require_rocm()
else:
    device = 'cpu'
    print('CPU-only diagnostic mode; GPU experiment conclusions are NOT valid.')
print('Device argument for Ultralytics:', device)'''),
 c('''# 可选：实际在 GPU 上计算一个小矩阵，验证张量确实被送入 GPU
import torch

if STRICT_ROCM:
    x = torch.randn((128, 128), device='cuda:0')
    y = x @ x
    torch.cuda.synchronize()
    assert y.device.type == 'cuda'  # ROCm 也报告 cuda device type
    print('HIP kernel smoke: PASS, output shape:', tuple(y.shape))'''),
 m('''### 练习：先预测，再运行（5 分钟）
计算思维的第一步是**预测→运行→对照**。下面这个格子默认能跑通，但直接跑没有意义：
1. 把 `SMOKE_N` 改成 256（或任意正整数）。
2. **先**把你的预测写进 `MY_PREDICTION`（替换两个 TODO）。
3. 运行，对照 `actual` 输出；猜错完全没关系——关键是养成先预测的习惯。
你的预测会保存到 `reports/lab00_practice.json`，作为你动手的证据。''',
   '''### Try It Yourself: predict, then run (5 min)
Computational thinking starts with **predict → run → compare**. The cell below runs as-is, but running it blindly teaches nothing:
1. Change `SMOKE_N` to 256 (or any positive integer).
2. **First** write your prediction into `MY_PREDICTION` (replace both TODOs).
3. Run it and compare with the `actual` output; being wrong is fine — the habit of predicting first is the point.
Your prediction is saved to `reports/lab00_practice.json` as evidence of hands-on work.'''),
 c('''# 学生练习：预测 → 运行 → 对照（修改 SMOKE_N 与 MY_PREDICTION 后运行）
SMOKE_N = 128
MY_PREDICTION = {'shape': 'TODO: 写下你预测的输出形状，如 (128, 128)',
                 'device': 'TODO: 你预测结果张量落在哪个设备？'}

# 无人值守复验通道（学生不需要）：设 YOLO_PRACTICE_FILL 指向 JSON 文件可注入已填写样例，
# 注入值会原样出现在本格输出与保存的报告里——不做任何隐藏。
if os.environ.get('YOLO_PRACTICE_FILL'):
    _p = Path(os.environ['YOLO_PRACTICE_FILL'])
    if _p.is_file():
        for _k, _v in json.loads(_p.read_text(encoding='utf-8')).get('lab00_practice', {}).items():
            MY_PREDICTION[_k] = _v

import torch

for key, value in MY_PREDICTION.items():
    assert not str(value).startswith('TODO'), f'请先填写 {key} 的预测再运行——直接运行无法形成预测习惯'
probe = torch.randn((SMOKE_N, SMOKE_N), device='cuda:0')
outcome = probe @ probe
torch.cuda.synchronize()
print('actual shape:', tuple(outcome.shape), '| actual device:', outcome.device)
matched = str(MY_PREDICTION['shape']).replace(' ', '') == str(tuple(outcome.shape))
print('your prediction:', MY_PREDICTION)
print('prediction matched:', matched, '（猜错不扣分；先预测再验证才是目的）')
write_json(home / 'reports/lab00_practice.json',
           {'smoke_n': SMOKE_N, 'prediction': MY_PREDICTION,
            'actual': {'shape': list(outcome.shape), 'device': str(outcome.device)},
            'matched': matched})'''),
 m('''### 冻结预训练权重（所有 A/C/E 使用相同模型）
为了避免课程中途的自动下载或不同版本导致 A/B 实验起点不一致，首次联网时缓存 `yolo26n.pt`，后续统一从 `/workspace/yolo_course/artifacts/yolo26n.pt` 读取。缓存成功后记录 SHA256。''',
   '''### Pinning the Pretrained Weights (labs A/C/E share one model)
To keep every A/B experiment starting from an identical point — and to avoid mid-class downloads — cache `yolo26n.pt` on first network use, then always load it from `/workspace/yolo_course/artifacts/yolo26n.pt`. Its SHA256 is recorded once the cache is pinned.'''),
 c('''import shutil
from ultralytics import YOLO

model_cache = home / 'artifacts/yolo26n.pt'
if not model_cache.is_file():
    fresh = YOLO('yolo26n.pt')
    source = Path(fresh.ckpt_path).resolve()
    assert source.is_file(), f'Model download not found at {source}'
    if source != model_cache.resolve():
        shutil.copy2(source, model_cache)
assert model_cache.is_file()
print('Pinned model:', model_cache, 'sha256:', sha256(model_cache))
save_config({'model': 'yolo26n.pt', 'model_sha256': sha256(model_cache)})'''),
 m('''### 课后问题（写进环境报告）
1. 为什么 AMD GPU 使用 `torch.cuda`？
2. `torch.version.hip` 返回 `None` 意味着什么？
3. 为什么不能在已经配好的 ROCm 环境随意覆盖 PyTorch？

**面试可能这样问**：入职后给你一台 AMD GPU 服务器，你怎么确认上面的深度学习环境能用？会检查哪些字段、分别排除什么问题？

**带走什么 / Takeaway**：环境是实验的地基——`torch.version.hip`、GPU 可见性、设备节点、权重指纹（SHA256），四样确认完才开始训练。

**验收文件**：`reports/environment.json`、`reports/lab00_practice.json`。''',
   '''### After-Class Questions (put answers in your environment report)
1. Why do AMD GPUs use the `torch.cuda` API?
2. What does it mean when `torch.version.hip` returns `None`?
3. Why must you never overwrite PyTorch in an already-configured ROCm environment?

**Interview-style question**: you are handed an AMD GPU server on day one — how do you verify its deep-learning stack works? Which fields do you check, and what does each rule out?

**Takeaway**: the environment is the foundation of every experiment — `torch.version.hip`, GPU visibility, device nodes, and the weight fingerprint (SHA256). Confirm all four before any training.

**Acceptance artifacts**: `reports/environment.json`, `reports/lab00_practice.json`.''')
], '10–15 分钟','10–15 minutes','环境 → GPU 后端 → 持久化工件','environment → GPU backend → persistent artifacts')

make('01_Lab_A_Predict_and_Threshold.ipynb','01 / Lab A · 推理、BBox 与置信度阈值','01 / Lab A · Inference, BBoxes & Confidence Thresholds',[
 lab_map([
  ('为什么学这个？','推理是"使用模型"的方式；置信度阈值是你每天要调的第一个旋钮',
   'Why learn this?','Inference is how a model is used; the confidence threshold is the first knob you will tune daily'),
  ('需要先懂什么？','完成 00；知道"框 + 类别 + 分数"是什么即可',
   'Prerequisites?','Lab 00; knowing what "box + class + score" means'),
  ('我修改哪个参数？','`box_pred` 坐标与 `MY_PREDICTED_IOU`（先预测！）',
   'What do I change?','`box_pred` coordinates and `MY_PREDICTED_IOU` (predict first!)'),
  ('成功长什么样？','三张图只差框数；`xyxy/conf/cls` 三段数组；单调性 PASS',
   'What does success look like?','Three images differing only in box count; three small arrays; monotonicity PASS'),
  ('失败怎么自救？','本页常见错误表（模型缺失 / NMS / 单调性反例）',
   'If it fails?','The common-errors table on this page (missing model / NMS / monotonicity)'),
  ('保存什么证据？','3 张预测图 + `lab_a_predictions.json` + 一句结论',
   'Evidence to save?','3 prediction images + `lab_a_predictions.json` + a one-line conclusion'),
  ('求职价值？','"conf 0.25 改 0.5，权重变了吗？"——面试必答的直觉题',
   'Job value?','"Does raising conf from 0.25 to 0.5 change the weights?" — a guaranteed interview question'),
 ]),
 m('''## 实验目标
给同一张图片设置 `conf=0.10/0.25/0.50`，测量检测数量并保存图片。理解 **Score ≠ 真实正确概率**，置信度阈值只影响显示/保留哪些预测，不会修改模型权重。

**依赖**：先完成 00；预训练模型和示例图片可从官方地址下载（如云端不能访问外网，请提前上传缓存）。

### 预期输出（看形状，不是标准答案数字）
- 先显示一张样例图；随后每个阈值打印一行 `0.1 predictions: N -> <保存路径>`，保存的 jpg 上有带类别名和分数的框。
- 三张图**只差画的框数**——同样的框，阈值越高保留越少。
- `xyxy / confidence / class ID` 单元格打印三小段 numpy 数组（像素坐标、0–1 浮点、整数类别号）。
- IoU 单元格对给定两框打印一个严格介于 0 和 1 之间的数。

### 常见错误与下一步
| 症状 | 原因 | 下一步 |
|---|---|---|
| `FileNotFoundError` 找不到 `yolo26n.pt` | 没先跑 00 / 权重未缓存 | 先跑 00（它把模型固定到 `artifacts/`） |
| `NotImplementedError ... torchvision::nms` | torchvision 构建缺 GPU NMS 内核 | 换 uv 隔离 ROCm 栈（部署文档 §2B） |
| 自己换图后检出数随阈值升高而增加 | 通常是低对比/重复目标的边界情况 | 重新阅读单调性断言的报错信息；如实报告你看到的现象 |''',
   '''## Goal
Run the same image at `conf=0.10/0.25/0.50`, count detections and save the images. Understand that **a score ≠ the true probability of correctness** — the confidence threshold only filters which predictions are kept/displayed; it never changes model weights.

**Prerequisite**: finish 00 first. The pretrained model and the sample image come from official sources (pre-upload them to the cache if your cloud has no external network).

### Expected Output (shape, not ground-truth numbers)
- One sample image is displayed, then for each threshold a line like `0.1 predictions: N -> <saved path>`; the saved jpg shows boxes with class names and scores.
- The three saved images differ **only in how many boxes are drawn** — same boxes, fewer of them as the threshold rises.
- The `xyxy / confidence / class ID` cell prints three small numpy arrays (pixel coordinates, 0–1 floats, integer class ids).
- The IoU cell prints a number strictly between 0 and 1 for the given boxes.

### Common Errors & Next Steps
| Symptom | Cause | Next step |
|---|---|---|
| `FileNotFoundError` on `yolo26n.pt` | Lab 00 not run / weights not cached | run 00 first (it pins the model into `artifacts/`) |
| `NotImplementedError ... torchvision::nms` | torchvision build lacks the GPU NMS kernel | switch to the uv-isolated ROCm stack (setup guide §2B) |
| counts **increase** as threshold rises on your own image | usually a very low-contrast/duplicate object case | re-check the monotonicity assertion message; report honestly what you saw |'''),
 c(SETUP),
 c('''device = require_rocm()
from ultralytics import YOLO
import ultralytics
from PIL import Image
from IPython.display import display

model_path = home / 'artifacts/yolo26n.pt'
model = YOLO(str(model_path) if model_path.is_file() else cfg['model'])
print('Loaded:', model.ckpt_path if hasattr(model, 'ckpt_path') else cfg['model'])'''),
 c('''# 优先使用 Ultralytics 包内样例，否则下载官方公交车样例
example = Path(ultralytics.__file__).resolve().parent / 'assets/bus.jpg'
if not example.exists():
    example = home / 'assets/bus.jpg'
    if not example.exists():
        import urllib.request

        url = 'https://ultralytics.com/images/bus.jpg'
        try:
            urllib.request.urlretrieve(url, example)
        except Exception as exc:
            raise RuntimeError(f'无法下载示例图。请自行上传一张图片至 {example}。原因: {exc}')
assert example.is_file()
display(Image.open(example).resize((480, 640)))'''),
 c('''from collections import Counter

thresholds = [0.10, 0.25, 0.50]
output_dir = home / 'assets/lab_a_thresholds'
output_dir.mkdir(parents=True, exist_ok=True)
summary = []
for threshold in thresholds:
    result = model.predict(source=str(example), device=device, conf=threshold, imgsz=640,
                           save=False, verbose=False)[0]
    boxes = result.boxes
    counts = Counter(int(k) for k in boxes.cls.cpu().tolist()) if boxes is not None else Counter()
    image_out = output_dir / f'conf_{threshold:.2f}.jpg'
    Image.fromarray(result.plot()[..., ::-1]).save(image_out)
    summary.append({'conf': threshold,
                    'detections': sum(counts.values()),
                    'classes': {model.names[k]: v for k, v in counts.items()},
                    'prediction_image': str(image_out)})
    print(threshold, 'predictions:', sum(counts.values()), '->', image_out)
write_json(home / 'reports/lab_a_predictions.json', summary)'''),
 c('''# 让学生实际看到 xyxy / conf / cls 三组张量
example_result = model.predict(str(example), device=device, conf=0.25, verbose=False)[0]
if example_result.boxes is not None and len(example_result.boxes) > 0:
    print('xyxy:', example_result.boxes.xyxy[:3].cpu().numpy())
    print('confidence:', example_result.boxes.conf[:3].cpu().numpy())
    print('class ID:', example_result.boxes.cls[:3].cpu().numpy())
else:
    print('此阈值下没有检测目标；属于有效的实验结果。')'''),
 c('''# 练习：亲手计算两个框的交并比（几何示例，不是模型测出的 mAP）
from course.lab_utils import iou_xyxy

box_true = (160, 120, 480, 360)
box_pred = (200, 140, 500, 380)  # 学生可修改这个框，重新预测后再运行
MY_PREDICTED_IOU = 'TODO: 先预测——这两个框的 IoU 大概是多少？写一个 0 到 1 之间的数'

# 无人值守复验通道（学生不需要）：YOLO_PRACTICE_FILL 指向 JSON 可注入已填写样例
if os.environ.get('YOLO_PRACTICE_FILL'):
    _p = Path(os.environ['YOLO_PRACTICE_FILL'])
    if _p.is_file():
        _v = json.loads(_p.read_text(encoding='utf-8')).get('lab_a_iou')
        if _v is not None:
            MY_PREDICTED_IOU = _v

assert not str(MY_PREDICTED_IOU).startswith('TODO'), '请先写下你的预测（0–1 之间的数）再运行'
overlap = iou_xyxy(box_true, box_pred)
print('实际 IoU =', round(overlap, 4), '| 你的预测 =', MY_PREDICTED_IOU)
print('修改 box_pred 的坐标再运行一次，观察 IoU 怎么变；猜错不扣分，先预测才是目的')
assert 0 <= overlap <= 1'''),
 c('''for item in summary:
    print(f"conf={item['conf']:.2f}  boxes={item['detections']}")
    display(Image.open(item['prediction_image']).resize((640, 480)))
assert all(summary[i]['detections'] >= summary[i + 1]['detections']
           for i in range(len(summary) - 1)), '同模型、同图像下阈值变大，检出数不应增加'
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
], '20 分钟','20 minutes','图像 → 检测结果 → 参数敏感性','image → detections → parameter sensitivity')

make('02_Lab_B_Dataset_Annotation_QA.ipynb','02 / Lab B · 数据集、标注、分离与 QA','02 / Lab B · Dataset, Annotation, Splits & QA',[
 lab_map([
  ('为什么学这个？','真实项目约七成时间在数据；脏数据与泄漏是"成绩造假"的头号来源',
   'Why learn this?','Real projects spend ~70% of time on data; dirty data and leakage are the top sources of fake scores'),
  ('需要先懂什么？','完成 00；课前装好 X-AnyLabeling 并标注 ≥5 张（见 `docs/LAB_B_X_ANYLABELING.md`）',
   'Prerequisites?','Lab 00; X-AnyLabeling installed and ≥5 images labeled before class (see `docs/LAB_B_X_ANYLABELING.md`)'),
  ('我修改哪个参数？','修正练习的 `NEW_LABEL_LINE`/`LABEL_LINE_INDEX`（须显式 `APPLY_CORRECTION=True` 才写入）',
   'What do I change?','`NEW_LABEL_LINE`/`LABEL_LINE_INDEX` in the correction exercise (writes only with an explicit `APPLY_CORRECTION=True`)'),
  ('成功长什么样？','QA `pass=true`、标注画廊带框、无效标签注入实验会报错',
   'What does success look like?','QA `pass=true`, a labeled gallery, and the injected-invalid-label experiment raising errors'),
  ('失败怎么自救？','本页常见错误表；导入器报错精确到文件与行',
   'If it falls?','The common-errors table on this page; importer errors name the exact file and line'),
  ('保存什么证据？','data.yaml、QA JSON、split_manifest.csv、人工检查表',
   'Evidence to save?','data.yaml, the QA JSON, split_manifest.csv, the manual review sheet'),
  ('求职价值？','数据岗必问："你怎么发现训练/测试泄漏？"——这节就是标准答案',
   'Job value?','Data-role staple: "How would you catch train/test leakage?" — this lab is the answer'),
 ]),
 m('''## 实验目标
- 检查一行 YOLO 标签 `class x_center y_center width height`，五个值中坐标为归一化值。
- 从 COCO128 **原始 128 张图**构造教学用、图片不重叠的 Train/Val/Test；默认只保留 `person/car/chair` 三类，并**不是**快递包裹数据集。
- 检查标签范围、重复图片、边界框与数据集配置。

**依赖**：先完成 00（环境与权重缓存）。本实验主要用 CPU 做数据 QA，不需要长时间占用 GPU。

> **关键科学边界**：官方 COCO128 训练/验证复用同样图片，仅用于冒烟测试。本 Notebook 对其重新分组和划分，尽量防止直接重叠；由于只有 128 张，无法据此宣称工业精度或严格 COCO 基准；语义相似场景泄漏还需人工审查。''',
   '''## Goal
- Parse one YOLO label line `class x_center y_center width height`; the four coordinates are normalized.
- Build a teaching-only, image-disjoint Train/Val/Test from the **original 128 COCO128 images**, keeping only `person/car/chair` — this is **not** a parcel dataset.
- Check label ranges, duplicates, bounding boxes and the dataset config.

> **Key scientific boundary**: official COCO128 reuses the same images for train and val and is for smoke tests only. This notebook re-groups and re-splits them to prevent direct overlap; with just 128 images the result can never be presented as industrial accuracy or a strict COCO benchmark, and semantically similar scenes still need human review.'''),
 m('''### 常见错误与下一步
| 症状 | 原因 | 下一步 |
|---|---|---|
| `Scene CSV must cover every image exactly once` | scenes.csv 漏图/多图/文件名对不上 | 按报错里的 missing/extra 清单逐个核对文件名（区分大小写） |
| `Invalid YOLO detection labels ... invalid class id` | 类别号 ≥ classes.txt 行数 | 类别 ID 从 0 开始且小于类别数；检查 X-AnyLabeling 导出时的类别顺序 |
| `Missing labels: [...]` | 有图片但没有对应 txt | 为确认无目标的图片创建**空** txt（合法负样本），不要删图 |
| `FileExistsError ... choose a new output name` | 同名数据集已导入过 | 换 `YOLO_ANYLABELING_OUTPUT_NAME`（如 `_v2`），不要删除旧结果 |
| `At least 9 images` / `Fewer than three independent groups` | 样本或独立场景太少 | 课上先用 COCO128 演示路径完成教学目标，课后补数据再导入 |''',
   '''### Common Errors & Next Steps
| Symptom | Cause | Next step |
|---|---|---|
| `Scene CSV must cover every image exactly once` | scenes.csv missing/extra/mismatched names | Fix filenames one by one using the missing/extra lists in the error (case-sensitive) |
| `Invalid YOLO detection labels ... invalid class id` | class id ≥ number of classes.txt lines | Class IDs are zero-based; check the class order in the X-AnyLabeling export |
| `Missing labels: [...]` | image without a matching txt | Create an **empty** txt for verified negatives — never delete the image |
| `FileExistsError ... choose a new output name` | dataset name already imported | Change `YOLO_ANYLABELING_OUTPUT_NAME` (e.g. `_v2`); never delete the old result |
| `At least 9 images` / `Fewer than three independent groups` | too few samples or scenes | Use the COCO128 demo path in class; collect more data and re-import after class |'''),
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
 c('''qa = inspect_dataset(yaml_path)
print('Class names:', qa['names'])
print('Split stats:', qa['splits'])
print('Errors:', qa['problems'][:30])
print('Near-duplicate review warnings:', qa['warnings'][:10])
write_json(home / 'reports/lab_b_dataset_qa.json',
           {k: v for k, v in qa.items() if k != 'sample_paths'})
assert qa['pass'], '发现数据结构错误，请先修复再训练'
print('Dataset structural QA: PASS')'''),
 c('''# 可视化训练标签；此步骤由学生人工复核遮挡、错标、漏标等问题
names = list(qa['names'].values()) if isinstance(qa['names'], dict) else list(qa['names'])
train_imgs = qa['sample_paths']['train']
fig = plot_labels(train_imgs, names, 6)
fig.savefig(home / 'assets/lab_b_label_gallery.png', dpi=150, bbox_inches='tight')
plt.show()'''),
 c('''# 代码练习：验证无效标签能够被 QA 捕获，不会污染实际数据集
valid = '0 0.5 0.5 0.2 0.3\\n'
invalid = '0 1.2 0.5 0.2 0.3\\n'
assert not label_errors(valid, len(names))
assert label_errors(invalid, len(names))
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

label_file = label_path_for_image(TARGET_IMAGE)
original_lines = label_file.read_text(encoding='utf-8').splitlines()
print('编辑目标:', label_file)
print('原始标签:', original_lines[:3])

if APPLY_CORRECTION:
    assert CORRECTION_REASON and not CORRECTION_REASON.startswith('TODO:')
    assert 0 <= LABEL_LINE_INDEX < len(original_lines), '该图可能是合法的空标签，需要先选择有框的图片'
    assert not label_errors(NEW_LABEL_LINE, len(names))
    old_line = original_lines[LABEL_LINE_INDEX]
    backup = (home / 'artifacts/label_backups' /
              f'{label_file.stem}_'
              f'{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.txt')
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(label_file.read_text(encoding='utf-8'), encoding='utf-8')
    original_lines[LABEL_LINE_INDEX] = NEW_LABEL_LINE
    label_file.write_text('\\n'.join(original_lines) + '\\n', encoding='utf-8')
    write_json(home / 'reports/lab_b_last_correction.json',
               {'image': str(TARGET_IMAGE), 'label': str(label_file),
                'before': old_line, 'after': NEW_LABEL_LINE,
                'reason': CORRECTION_REASON, 'backup': str(backup)})
    assert inspect_dataset(yaml_path)['pass'], '修改后数据集 QA 失败，请恢复备份'
    print('修正已写入，备份：', backup)
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
], '50–60 分钟','50–60 minutes','手工标注 → 导出 → 防泄漏划分 → QA','manual labeling → export → grouped splits → QA')

make('03_Lab_C_ROCm_Training_Finetune.ipynb','03 / Lab C · ROCm GPU 训练、微调与 Loss','03 / Lab C · ROCm GPU Training, Fine-tuning & Loss',[
 lab_map([
  ('为什么学这个？','训练把"会用模型"变成"会造模型"；读懂 Loss 曲线是分水岭',
   'Why learn this?','Training turns "can use a model" into "can build one"; reading loss curves is the watershed'),
  ('需要先懂什么？','完成 00（权重缓存）与 02（数据集登记）',
   'Prerequisites?','Lab 00 (weight cache) and Lab 02 (registered dataset)'),
  ('我修改哪个参数？','课堂保持默认（12 Epoch 是全组统一基线）；课后可试 `epochs`/`imgsz` 并记录理由',
   'What do I change?','Keep defaults in class (12 epochs is the shared baseline); after class try `epochs`/`imgsz` and record why'),
  ('成功长什么样？','冒烟几十秒完成；曲线 loss 整体降、mAP 整体升；`best.pt` 生成',
   'What does success look like?','Smoke finishes in tens of seconds; loss trends down, mAP up; `best.pt` appears'),
  ('失败怎么自救？','本页常见错误表；训练"看似卡住"多半是 JIT 预热（见下方等待说明）',
   'If it fails?','The common-errors table; a "stuck" training is usually JIT warm-up (see the wait note below)'),
  ('保存什么证据？','best.pt + SHA256、results.csv/png、训练配置 JSON',
   'Evidence to save?','best.pt + SHA256, results.csv/png, the training-config JSON'),
  ('求职价值？','"我在 AMD ROCm 上真机微调过 YOLO" + 完整复现链',
   'Job value?','"I fine-tuned YOLO on real AMD ROCm hardware" with a full reproduction chain'),
 ]),
 m('''## 实验目标
1. 在真实 AMD ROCm GPU 上运行 YOLO26n 预训练模型微调。
2. 先用 COCO8（4 train + 4 val）做**训练链路冒烟**，再训练 Lab B 数据。
3. 保存运行条件、`best.pt`、`last.pt`、训练曲线与 SHA256。
4. 解释 Loss、Epoch、Batch、学习率、过拟合和泛化；不得把 Smoke 成绩说成工业精度。

**依赖**：先完成 00（权重缓存）与 02（数据集登记）。需要 GPU 与时间；推荐课前缓存模型/数据。

**注意**：`amp=False` 仅作 ROCm AMP 异常排查，不应在没有问题时随意更改对照配置。

### 预期输出（看形状，不是标准答案数字）
- 冒烟格几分钟内完成并打印 `GPU smoke completed`。
- 正式训练逐 epoch 打印进度；结束后 `runs/lab_c/baseline/` 下出现 `weights/best.pt`、`weights/last.pt`、`results.csv`（每 epoch 一行）、`results.png`。
- `results.png` 里 **train loss 各列整体下降**、mAP 相关列整体上升——小数据集上会抖动，方向比光滑更重要。
- 验证格打印一个含 `precision / recall / mAP50 / mAP50_95` 的字典，并明确标注 `split: val`。

### 常见错误与下一步
| 症状 | 原因 | 下一步 |
|---|---|---|
| `先完成 02 Lab B` 断言失败 | course_config 里没有数据集 | 先跑 02，或设 `YOLO_USER_DATA_YAML` |
| 训练出现 NaN | AMP/数值问题或数据有坏标签 | 保持 `amp=False` 重试；用 02 的 QA 复查标签 |
| `torchvision GPU NMS` 报错 | 平台栈缺 NMS 内核 | 部署文档 §2B 自建隔离栈 |
| 显存不足 OOM | batch 过大或 imgsz 过大 | 本课程配置在 48GB 卡上余量充足；改配置前先记录并说明理由 |''',
   '''## Goal
1. Fine-tune the YOLO26n pretrained model on a real AMD ROCm GPU.
2. **Smoke-test the training pipeline** on COCO8 (4 train + 4 val) first, then train on the Lab B dataset.
3. Persist run conditions, `best.pt`, `last.pt`, training curves and SHA256 hashes.
4. Explain loss, epochs, batches, learning rate, overfitting and generalization; never dress smoke-run numbers up as industrial accuracy.

**Prerequisite**: finish 00 (weight cache) and 02 (dataset registration). This lab needs a GPU and time; pre-caching model/data before class is recommended.

**Note**: `amp=False` exists for ROCm AMP troubleshooting only — do not flip comparison configs without a diagnosed problem.

### Expected Output (shape, not ground-truth numbers)
- The smoke cell finishes within minutes and prints `GPU smoke completed`.
- Full training prints per-epoch progress; afterwards `runs/lab_c/baseline/` contains `weights/best.pt`, `weights/last.pt`, `results.csv` (one row per epoch) and `results.png`.
- In `results.png` the **train-loss columns trend down** and the mAP columns trend up — on a tiny dataset they wobble; direction matters more than smoothness.
- The validation cell prints a dict with `precision / recall / mAP50 / mAP50_95`, clearly tagged `split: val`.

### Common Errors & Next Steps
| Symptom | Cause | Next step |
|---|---|---|
| assertion `先完成 02 Lab B` fires | no dataset in course_config | run 02 first, or set `YOLO_USER_DATA_YAML` |
| NaN during training | AMP/numeric issue or bad labels | keep `amp=False` and retry; re-run 02 QA on labels |
| `torchvision GPU NMS` error | platform stack lacks the NMS kernel | build the isolated stack (setup guide §2B) |
| CUDA OOM | batch/imgsz too large | the course config fits comfortably in 48 GB; record and justify any change |'''),
 c(SETUP),
 c('''from course.lab_utils import get_metrics_summary
import torch
import time
from ultralytics import YOLO

device = require_rocm()
assert cfg['dataset_yaml'] and Path(cfg['dataset_yaml']).is_file(), \\
    '先完成 02 Lab B，或设置 YOLO_USER_DATA_YAML'
model_initial = (str(home / 'artifacts/yolo26n.pt')
                 if (home / 'artifacts/yolo26n.pt').is_file() else cfg['model'])
print('Training:', model_initial, 'dataset:', cfg['dataset_yaml'])'''),
 m('''### 训练开始后的正常现象（等待时看这里）
- 每个 Epoch 打印一行进度（各项 loss 与耗时）；W7900 级别显卡上 12 Epoch 约 **30–100 秒**。
- **首次运行前 ~1 分钟可能"看似卡住"**——那是 MIOpen 在编译 GPU kernel（一次性编译，之后全机共享缓存，第二次起不再出现）。想确认它在干活，可在另一个终端运行 `rocm-smi` 观察 GPU 利用率。
- 请不要中断训练；等待期间正好阅读下面的思考题。''',
   '''### What to expect while training runs (read while you wait)
- One progress line per epoch (losses and time); on a W7900-class GPU, 12 epochs take roughly **30–100 seconds**.
- **The first run may look frozen for ~1 minute** — that is MIOpen compiling GPU kernels (a one-time cost shared machine-wide afterwards). Run `rocm-smi` in another terminal to watch the GPU working.
- Do not interrupt; the wait is the perfect time to read the questions below.'''),
 c('''# 非正式数据集，验证 GPU 实际能跑 Forward/Backward，不用于发布精度成绩。
RUN_SMOKE = True
if RUN_SMOKE:
    smoke_model = YOLO(model_initial)
    smoke_model.train(data='coco8.yaml', epochs=1, imgsz=320, batch=4, device=device,
                      workers=0, amp=False, project=str(home / 'runs/lab_c'), name='smoke',
                      exist_ok=True, plots=False, seed=42)
    print('GPU smoke completed; COCO8 is not a benchmark.')'''),
 c('''# 正式教学训练。为了让学生在课上完成闭环，默认 12 Epoch；高质量结果需课后延长训练。
RUN_BASELINE = True
BASELINE_EPOCHS = 12
train_kwargs = dict(data=cfg['dataset_yaml'], epochs=BASELINE_EPOCHS,
                    imgsz=512, batch=8, device=device, workers=0, patience=30,
                    optimizer='SGD', seed=42, deterministic=False,
                    amp=False, plots=True, project=str(home / 'runs/lab_c'),
                    name='baseline', exist_ok=True)
if RUN_BASELINE:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    baseline = YOLO(model_initial)
    train_output = baseline.train(**train_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    run_dir = Path(baseline.trainer.save_dir).resolve()
    weights = run_dir / 'weights/best.pt'
    assert weights.exists(), f'best.pt missing: {weights}'
    record = {'run_dir': str(run_dir), 'weights': str(weights), 'sha256': sha256(weights),
              'elapsed_seconds': round(elapsed, 2), 'peak_allocated_GiB': round(peak_gib, 3),
              'args': train_kwargs, 'system': system_info(),
              'warning': 'Peak memory is PyTorch allocated memory, not total device VRAM.'}
    write_json(home / 'reports/lab_c_baseline.json', record)
    save_config({'baseline_run': str(run_dir), 'baseline_weights': str(weights),
                 'baseline_train_settings': train_kwargs})
    print(record)
else:
    print('未执行训练。后续实验需要一个经过验证的 best.pt，请不要伪造训练结果。')'''),
 c('''# 阅读训练工件：不同版本可能输出不同的列名，请以实际 CSV 为准。
from PIL import Image
import pandas as pd

cfg = load_config()
run_dir = Path(cfg['baseline_run'])
csv_path = run_dir / 'results.csv'
if csv_path.exists():
    history = pd.read_csv(csv_path)
    display(history.tail(5))
else:
    print('Missing results.csv:', csv_path)
if (run_dir / 'results.png').exists():
    display(Image.open(run_dir / 'results.png').resize((1120, 530)))'''),
 c('''# 不混淆验证集与最终测试集：首次使用 split='val'
cfg = load_config()
best = YOLO(cfg['baseline_weights'])
metrics = best.val(data=cfg['dataset_yaml'], split='val', device=device,
                   imgsz=640, batch=8, workers=0, plots=True, verbose=False)
summary = get_metrics_summary(metrics)
summary['split'] = 'val'
summary['model_sha256'] = sha256(Path(cfg['baseline_weights']))
write_json(home / 'reports/lab_c_baseline_val.json', summary)
print('Validation only (NOT test):', summary)'''),
 m('''### 练习：读懂你自己的训练曲线（先判断，再验证）
训练不是"跑完就结束"。打开上面 `results.csv` 的最后几行和 `results.png`，**自己**回答三个问题：
1. train 的 box_loss 从第 1 到第 12 个 epoch 大致怎么变？
2. 验证集 mAP50 怎么变？
3. 以你的数字为证据：过拟合了吗？为什么？

把答案写进下面的 `MY_READING` 再运行保存。你的判断会在 Lab D 与官方验证指标对照——现在写错完全没关系，先形成"用证据下结论"的习惯。''',
   '''### Try It Yourself: read your own training curve (judge first, verify later)
Training is not "run and done". Open the tail of `results.csv` above and `results.png`, and answer three questions **yourself**:
1. How did train box_loss move from epoch 1 to 12?
2. How did validation mAP50 move?
3. With your numbers as evidence: is the model overfitting? Why?

Write your answers into `MY_READING` below, then run the cell to save them. Lab D will confront your reading with the official metrics — being wrong now is fine; build the habit of concluding from evidence first.'''),
 c('''# 学生练习：读曲线（填写 MY_READING 后运行；引用你自己 results.csv 里的数字）
import csv

MY_READING = {
    'train_box_loss_epoch1_to_12': 'TODO: box_loss 从第 1 到第 12 个 epoch 大致如何变化？引用两个数字',
    'val_mAP50_trend': 'TODO: 验证集 mAP50 大致如何变化？引用两个数字',
    'overfitting_call': 'TODO: 以你的曲线为证据，过拟合了吗？为什么？',
}
# 无人值守复验通道（学生不需要）：YOLO_PRACTICE_FILL 指向 JSON 可注入已填写样例
if os.environ.get('YOLO_PRACTICE_FILL'):
    _p = Path(os.environ['YOLO_PRACTICE_FILL'])
    if _p.is_file():
        for _k, _v in json.loads(_p.read_text(encoding='utf-8')).get('lab_c_my_reading', {}).items():
            MY_READING[_k] = _v
for key, value in MY_READING.items():
    assert not str(value).startswith('TODO'), f'请先填写 {key}——用自己的 results.csv 数字回答后再运行'
with (Path(cfg['baseline_run']) / 'results.csv').open(encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('你的 epoch 数:', len(rows), '| 首行/末行 epoch:', rows[0].get('epoch'), '/', rows[-1].get('epoch'))
write_json(home / 'reports/lab_c_my_reading.json', MY_READING)
print('曲线阅读已保存到 reports/lab_c_my_reading.json；Lab D 会用官方指标对照你的判断。')'''),
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
], '50–70 分钟','50–70 minutes','预训练权重 → 前后向训练 → 模型工件','pretrained weights → forward/backward training → model artifacts')

make('04_Lab_D_Evaluation_Error_Analysis.ipynb','04 / Lab D · mAP、混淆与失败案例','04 / Lab D · mAP, Confusion & Failure Cases',[
 lab_map([
  ('为什么学这个？','没有评价的训练是盲飞；错误归因决定下一步优化方向',
   'Why learn this?','Training without evaluation is flying blind; error attribution decides the next move'),
  ('需要先懂什么？','完成 03（要它的 best.pt）；Lab A 的 IoU 概念',
   'Prerequisites?','Lab 03 (its best.pt); the IoU idea from Lab A'),
  ('我修改哪个参数？','`MY_ERROR_INDEX` 与 `MY_ANALYSIS` 四字段（选一个错误深挖）',
   'What do I change?','`MY_ERROR_INDEX` and the four fields of `MY_ANALYSIS` (dig into one error)'),
  ('成功长什么样？','官方 mAP 打印、PR/混淆矩阵图、左右对照失败图',
   'What does success look like?','Official mAP printed, PR/confusion plots, side-by-side failure boards'),
  ('失败怎么自救？','本页常见错误表；失败案例不足 5 张时如实报告',
   'If it fails?','The common-errors table; report honestly when fewer than 5 failure cases exist'),
  ('保存什么证据？','官方指标 JSON、固定阈值 JSON、errors.csv、归因 JSON',
   'Evidence to save?','Official-metrics JSON, fixed-threshold JSON, errors.csv, your attribution JSON'),
  ('求职价值？','"讲一个你定位过的 badcase"——算法岗必问',
   'Job value?','"Walk me through a bad case you diagnosed" — asked in every algorithm interview'),
 ]),
 m('''## 实验目标
- 用 Ultralytics 官方验证器计算 Precision、Recall、mAP50、mAP50-95。
- 独立完成一次**固定 `conf=0.25`、IoU=0.50 的教学级 TP/FP/FN 匹配**（并非 COCO mAP 实现）。
- 查看 PR/混淆矩阵并生成至少五份真实失败案例。
- 不用 Test 集选择模型或调阈值。

**依赖**：先完整运行 03（需要它的 `best.pt` 与数据集登记）。本实验的预测都在 GPU 上完成，但单张推理很快。

`metrics.box.map` 才是官方 mAP50-95；我们的 TP/FP/FN 固定阈值分析只用于学习错误原因。''',
   '''## Goal
- Compute Precision, Recall, mAP50 and mAP50-95 with the official Ultralytics validator.
- Independently run a **fixed-threshold `conf=0.25`, IoU=0.50 teaching-level TP/FP/FN matching** (this is NOT the COCO mAP algorithm).
- Inspect PR/confusion matrices and produce at least five genuine failure cases.
- Never use the test split to pick models or tune thresholds.

**Prerequisite**: fully run 03 first (its `best.pt` and dataset registration are required). Predictions run on the GPU but are fast per image.

`metrics.box.map` is the official mAP50-95; our fixed-threshold TP/FP/FN analysis exists only to learn about error causes.'''),
 m('''### 常见错误与下一步
| 症状 | 原因 | 下一步 |
|---|---|---|
| `先完成 Lab C` 断言失败 | course_config 里没有 baseline_weights | 先完整运行 03（训练 + 保存 best.pt） |
| `lab_d_errors.csv` 是空的 | 该阈值下确实没有错误（或阈值过高） | 如实报告"0 个错误"；可选做困难样本分析，不要编造 |
| 官方 P/R 与固定阈值 P/R 差很多 | 两套口径本就不同 | 重读上面的双口径说明；报告时分开写、分开引用 |
| PR/混淆矩阵图不显示 | plots 输出目录因版本而异 | 查看 `val_metrics.save_dir` 的打印值并打开该目录 |''',
   '''### Common Errors & Next Steps
| Symptom | Cause | Next step |
|---|---|---|
| `先完成 Lab C` assertion fires | no baseline_weights in course_config | fully run Lab 03 (training + saved best.pt) first |
| `lab_d_errors.csv` is empty | genuinely no errors at this threshold (or threshold too high) | report "0 errors" honestly; optionally analyze hard samples — never fabricate |
| Official P/R very different from fixed-threshold P/R | the two calibers are different by design | re-read the dual-caliber note above; cite them separately |
| PR/confusion plots missing | plots directory varies by version | open the printed `val_metrics.save_dir` path |'''),
 c(SETUP),
 c('''from course.lab_utils import (get_metrics_summary, inspect_dataset, match_predictions,
                              xywhn_to_xyxy, read_labels, label_path_for_image)
from ultralytics import YOLO
from PIL import Image, ImageDraw
from IPython.display import display
import torch

device = require_rocm()
assert cfg['baseline_weights'] and Path(cfg['baseline_weights']).is_file(), \\
    '先完成 Lab C，或经过核验设置 best.pt'
assert Path(cfg['dataset_yaml']).is_file()
model = YOLO(cfg['baseline_weights'])
print('Weight SHA256:', sha256(Path(cfg['baseline_weights'])))'''),
 c('''# 正式官方 mAP 计算：不要把 conf=0.25 当作计算 mAP 的唯一阈值
val_metrics = model.val(data=cfg['dataset_yaml'], split='val', device=device,
                        imgsz=640, batch=8, workers=0, plots=True, verbose=False)
score = get_metrics_summary(val_metrics)
score.update({'split': 'val', 'dataset_yaml': cfg['dataset_yaml'],
              'weight_sha256': sha256(Path(cfg['baseline_weights']))})
write_json(home / 'reports/lab_d_official_metrics.json', score)
print('Official validation metrics:', score)
print('mAP: IoU=0.50..0.95 average, NOT confidence 50%-95%')'''),
 c('''# 展示 Ultralytics 官方可视化（如果已生成）
from PIL import Image

out = Path(val_metrics.save_dir)
for title in ('BoxPR_curve.png', 'PR_curve.png', 'confusion_matrix.png',
              'confusion_matrix_normalized.png'):
    p = out / title
    if p.exists():
        print(title)
        display(Image.open(p).resize((950, 600)))'''),
 c('''# 手工逐图片匹配：一个 GT 最多配一个同类、IoU>=0.50 的预测，其他为 FP/FN
qa = inspect_dataset(Path(cfg['dataset_yaml']))
val_images = qa['sample_paths']['val']
rows = []
failures = []
totals = {'TP': 0, 'FP': 0, 'FN': 0}
for img in val_images:
    pred = model.predict(source=str(img), device=device, imgsz=640,
                         conf=0.25, iou=0.70, verbose=False)[0]
    gt = label_path_for_image(img)
    match = match_predictions(pred, gt, iou_threshold=0.50)
    for k in totals:
        totals[k] += match[k]
    for issue in match['errors']:
        rows.append({'image': img.name, 'image_path': str(img), 'error': issue['error'],
                     'class_id': issue['class_id'], 'score': issue['score'],
                     'max_iou_gt': issue['max_iou_gt']})
    if match['errors']:
        failures.append((img, pred, gt, match))
precision = totals['TP'] / max(1, totals['TP'] + totals['FP'])
recall = totals['TP'] / max(1, totals['TP'] + totals['FN'])
print('Fixed threshold teaching-level counts:', totals,
      'P:', round(precision, 3), 'R:', round(recall, 3))
print('Important: these are NOT the official integrated PR or mAP values.')
write_csv(home / 'reports/lab_d_errors.csv', rows,
          ['image', 'image_path', 'error', 'class_id', 'score', 'max_iou_gt'])
write_json(home / 'reports/lab_d_fixed_threshold.json',
           {'conf': 0.25, 'matching_iou': 0.50,
            'nms_iou': 0.70, 'counts': totals,
            'precision_fixed': precision, 'recall_fixed': recall,
            'note': 'Teaching matching only; not a COCO AP implementation.'})'''),
 c('''# 保存真实失败图（左真值，右预测）；如果几乎无错误，忠实报告而不是捏造 5 个。
from PIL import Image, ImageDraw

error_dir = home / 'assets/lab_d_errors'
error_dir.mkdir(parents=True, exist_ok=True)
names = list(qa['names'].values()) if isinstance(qa['names'], dict) else list(qa['names'])
for idx, (img, pred, lab, match) in enumerate(failures[:5], 1):
    gt = Image.open(img).convert('RGB')
    W, H = gt.size
    d = ImageDraw.Draw(gt)
    for box in read_labels(lab):
        coords = xywhn_to_xyxy(box, W, H)
        d.rectangle(coords, outline=(40, 210, 120), width=max(2, W // 160))
        d.text((coords[0], max(0, coords[1] - 15)), f'GT {names[box[0]]}', fill=(0, 170, 85))
    predicted = Image.fromarray(pred.plot()[..., ::-1]).convert('RGB')
    board = Image.new('RGB', (W * 2, H), (245, 248, 250))
    board.paste(gt, (0, 0))
    board.paste(predicted, (W, 0))
    p = error_dir / f'{idx:02d}_{img.stem}.jpg'
    board.save(p, quality=90)
    print(p, 'errors:', len(match['errors']))
    display(board.resize((960, round(960 * H / (2 * W)))))
if len(failures) < 5:
    print(f'只有 {len(failures)} 张有失败案例。不能编造；可拓展验证集或评估更多困难样本。')'''),
 m('''### 学生工作：手动归因至少 5 个实际错误（如果有）
打开 `reports/lab_d_errors.csv` 和并排对照图，添加 **现象、证据、可能根因、下次实验验证方法**。把推测与已确认根因区分开。

下面提供一个可保存的归因练习格：从 `lab_d_errors.csv` 里挑**一个**你最好奇的错误，把四项分析写进 `MY_ANALYSIS` 再运行（写错不扣分，空着或 TODO 才是不合格）。''',
   '''### Student Work: manually attribute at least 5 real errors (if any)
Open `reports/lab_d_errors.csv` and the side-by-side boards, then add **symptom, evidence, suspected root cause, and how the next experiment would verify it**. Keep hypotheses clearly separate from confirmed causes.

The practice cell below makes this concrete: pick **one** error you are most curious about from `lab_d_errors.csv`, fill the four fields of `MY_ANALYSIS`, then run it (wrong hypotheses cost nothing; empty or TODO answers do).'''),
 c('''# 学生练习：独立归因一个真实错误（填写 MY_ANALYSIS 后运行）
import csv

with (home / 'reports/lab_d_errors.csv').open(encoding='utf-8-sig') as f:
    error_rows = list(csv.DictReader(f))
MY_ERROR_INDEX = 0  # 改成你想分析的那一行（0 到 len(error_rows)-1）
MY_ANALYSIS = {
    'symptom': 'TODO: 现象——这张图上到底发生了什么？',
    'evidence': 'TODO: 证据——引用该行的 error / score / max_iou_gt 数字与图上所见',
    'suspected_cause': 'TODO: 你的假设（遮挡/小目标/类别混淆/标注问题/阈值…）',
    'verify_next_step': 'TODO: 下一次实验如何验证这个假设？',
}
# 无人值守复验通道（学生不需要）：YOLO_PRACTICE_FILL 指向 JSON 可注入已填写样例
if os.environ.get('YOLO_PRACTICE_FILL'):
    _p = Path(os.environ['YOLO_PRACTICE_FILL'])
    if _p.is_file():
        for _k, _v in json.loads(_p.read_text(encoding='utf-8')).get('lab_d_error_analysis', {}).items():
            MY_ANALYSIS[_k] = _v
assert error_rows, '没有错误行——恭喜，但请如实报告并挑一张困难样本做分析'
assert 0 <= MY_ERROR_INDEX < len(error_rows), f'索引需在 0 到 {len(error_rows) - 1} 之间'
for key, value in MY_ANALYSIS.items():
    assert not str(value).startswith('TODO'), f'请先填写 {key}——归因练习不允许留空'
chosen = error_rows[MY_ERROR_INDEX]
print('你选择的错误:', chosen['image'], chosen['error'], '| score:', chosen['score'],
      '| max_iou_gt:', chosen['max_iou_gt'])
write_json(home / 'reports/lab_d_my_error_analysis.json', {'chosen': chosen, 'analysis': MY_ANALYSIS})
print('归因已保存到 reports/lab_d_my_error_analysis.json。')'''),
 m('''### 思考题 / 面试必答
1. 为什么官方 mAP50 和单一 conf=0.25 下的 Precision 不是同一个东西？
2. IoU=0.5 的定位标准会漏掉什么信息？
3. `mAP50-95` 的 50–95 指置信度吗？
4. 为什么不能用测试集调阈值后再报告该测试集成绩？

**交付**：`lab_d_official_metrics.json`、`lab_d_fixed_threshold.json`、`lab_d_errors.csv`、`lab_d_my_error_analysis.json`、至少现有真实失败图和分析报告。''',
   '''### Think / Interview Questions
1. Why is the official mAP50 not the same thing as Precision at a single conf=0.25?
2. What information does an IoU=0.5 localization criterion throw away?
3. Does the 50–95 in `mAP50-95` refer to confidence?
4. Why can't you tune thresholds on a test set and then report scores on that same set?

**Deliverables**: `lab_d_official_metrics.json`, `lab_d_fixed_threshold.json`, `lab_d_errors.csv`, `lab_d_my_error_analysis.json`, the real failure images so far and an analysis report.''')
], '35–45 分钟','35–45 minutes','官方指标 → 固定阈值配对 → 错误归因','official metrics → fixed-threshold matching → error attribution')

make('05_Lab_E_Controlled_AB_Optimization.ipynb','05 / Lab E · 受控 A/B 训练优化实验','05 / Lab E · Controlled A/B Training Experiment',[
 lab_map([
  ('为什么学这个？','受控实验是"调参"与"科学"的分界线；面试官用它识破纸面调参',
   'Why learn this?','Controlled experiments separate "tweaking" from "science"; interviewers use them to spot paper-tuning'),
  ('需要先懂什么？','完成 03；理解"一次只改一个变量"',
   'Prerequisites?','Lab 03; the idea of changing exactly one variable at a time'),
  ('我修改哪个参数？','只填 `MY_HYPOTHESIS`（结果前封存）；**A/B 训练参数不许改**——改了就不是受控实验',
   'What do I change?','Only `MY_HYPOTHESIS` (sealed before results); **never touch the A/B training args** — that breaks the control'),
  ('成功长什么样？','双臂柱状图 + ΔmAP 打印；你的假设与真实结果并列',
   'What does success look like?','Two-arm bar chart + ΔmAP; your hypothesis next to the real result'),
  ('失败怎么自救？','本页常见错误表；初始权重 SHA256 不一致 → 重跑 00',
   'If it fails?','The common-errors table; initial-weight SHA256 mismatch → rerun Lab 00'),
  ('保存什么证据？','hypothesis JSON、A/B CSV、双 best.pt 哈希',
   'Evidence to save?','The hypothesis JSON, the A/B CSV, both best.pt hashes'),
  ('求职价值？','"你怎么保证只有一个变量？"——标准答案就在这节',
   'Job value?','"How do you guarantee a single variable?" — the standard answer is this lab'),
 ]),
 m('''## 实验目标
基于 Lab C 的**真实 baseline**，只修改训练图像尺寸（512 → 416）；保持模型初始化权重、数据集、Epoch、Batch、随机种子等条件不变，用统一的验证输入尺寸 640 测量两组模型。不要因为速度更快就直接宣布算法优化成功。

**依赖**：先完整运行 03（baseline 工件与 `baseline_train_settings` 都来自它）。两次 12-Epoch 训练共需几分钟 GPU 时间。

> **公平性说明：** 降低 imgsz 会改变输入信息量和计算量。固定相同 Epoch 不保证相同训练时长 / SGD 更新质量；本实验验证一个明确的课程假设，不提供统计显著性证明。训练时间还可能受缓存及云端 GPU 并发负载影响。''',
   '''## Goal
Starting from the **real baseline produced by Lab C**, change exactly one thing — the training image size (512 → 416). Keep initial weights, dataset, epochs, batch, random seed and everything else identical, then measure both models with the same validation input size of 640. Never declare an optimization successful just because it runs faster.

**Prerequisite**: fully run 03 first (the baseline artifacts and `baseline_train_settings` come from it). The two 12-epoch trainings need a few minutes of GPU time.

> **Fairness note:** lowering imgsz changes both the information content and the compute of each step. Equal epochs do not guarantee equal training time / SGD update quality; this experiment tests one clearly stated classroom hypothesis and provides no statistical significance. Training time is also affected by caches and concurrent load on shared cloud GPUs.'''),
 m('''### 常见错误与下一步
| 症状 | 原因 | 下一步 |
|---|---|---|
| `初始权重 SHA256 改变` 断言 | `yolo26n.pt` 被替换过 | 重跑 00 重新冻结——A/B 两臂必须同一初始权重 |
| `baseline 训练尺寸为 512` 断言 | 03 的实验方案被改动 | 恢复 03 默认方案，或明确修改本实验设计并说明 |
| `set(changed) == {...}` 断言失败 | variant 参数改多了 | 只允许 imgsz 差异；project/name 只是日志位置 |
| 这一臂比 Lab C 慢 | GPU 上有并发负载 | A/B 计时轮必须独占 GPU；重跑并在报告记录干扰 |''',
   '''### Common Errors & Next Steps
| Symptom | Cause | Next step |
|---|---|---|
| initial-weight SHA256 assertion | `yolo26n.pt` was replaced | rerun Lab 00 to re-pin — both arms must share one initial weight |
| `baseline 训练尺寸为 512` assertion | Lab 03's recipe was altered | restore the Lab 03 defaults, or consciously redesign this experiment and say so |
| `set(changed) == {...}` assertion | too many variant args changed | only imgsz may differ; project/name are log locations |
| this arm slower than Lab C | concurrent load on the GPU | the timing round must own the GPU; rerun and record the interference |'''),
 c(SETUP),
 c('''from course.lab_utils import get_metrics_summary
import torch
import time
from ultralytics import YOLO

device = require_rocm()
assert cfg.get('baseline_train_settings') and Path(cfg['baseline_weights']).is_file(), \\
    '请先完整运行 Lab C baseline。'
base_record = json.loads((home / 'reports/lab_c_baseline.json').read_text(encoding='utf-8'))
args = cfg['baseline_train_settings']
print('Baseline train imgsz:', args['imgsz'], 'epochs:', args['epochs'])
print('Baseline model sha256:', sha256(Path(cfg['baseline_weights'])))
model_cache = home / 'artifacts/yolo26n.pt'
assert model_cache.is_file(), '预训练初始权重缓存不存在；重新运行 00 进行固定。'
assert sha256(model_cache) == cfg.get('model_sha256'), '初始权重 SHA256 改变，A/B 结果不可比。'
assert args['imgsz'] == 512, '此 Notebook 的实验方案假定 baseline 训练尺寸为 512，请先审核与修改方案' '''),
 m('''### 练习：先写下你的假设（在看到任何结果之前）
受控实验的第一步不是跑代码，而是**预测**。在运行 Variant 训练之前，回答：
1. 你预测哪组的 mAP50-95 更高？（baseline 512 / imgsz416 / 不确定）
2. 你预测哪组训练更快？依据是什么？（提示：输入变小≠必然更快，想想数据加载、JIT 预热等因素）
3. 一句话说明你的理由。

把预测写进 `MY_HYPOTHESIS` 再运行——它会**在结果产生前**封存到 `reports/lab_e_my_hypothesis.json`。实验结束后回来对照：预测错了恰恰是学习收获最大的时刻。''',
   '''### Try It Yourself: write your hypothesis first (before seeing any result)
The first step of a controlled experiment is not running code — it is **predicting**. Before the variant training starts, answer:
1. Which arm do you predict will score higher on mAP50-95? (baseline 512 / imgsz416 / not sure)
2. Which arm do you predict trains faster, and why? (hint: smaller inputs ≠ automatically faster — think about data loading, JIT warm-up, …)
3. One sentence for your reasoning.

Fill `MY_HYPOTHESIS` and run the cell — it is **sealed before any result exists** into `reports/lab_e_my_hypothesis.json`. You will come back to compare after the experiment; a wrong prediction is where the learning happens.'''),
 c('''# 学生练习：假设先行（在运行 Variant 训练之前填写并运行）
MY_HYPOTHESIS = {
    'mAP50_95_winner': 'TODO: baseline(512) / imgsz416 / 不确定——你预测哪组 mAP50-95 更高？',
    'training_time_winner': 'TODO: 你预测哪组训练更快？为什么（不只是分辨率一个因素）',
    'reasoning': 'TODO: 一句话说明你的依据',
}
# 无人值守复验通道（学生不需要）：YOLO_PRACTICE_FILL 指向 JSON 可注入已填写样例
if os.environ.get('YOLO_PRACTICE_FILL'):
    _p = Path(os.environ['YOLO_PRACTICE_FILL'])
    if _p.is_file():
        for _k, _v in json.loads(_p.read_text(encoding='utf-8')).get('lab_e_hypothesis', {}).items():
            MY_HYPOTHESIS[_k] = _v
for key, value in MY_HYPOTHESIS.items():
    assert not str(value).startswith('TODO'), f'请先填写 {key}——假设必须在看到结果之前写下'
write_json(home / 'reports/lab_e_my_hypothesis.json', MY_HYPOTHESIS)
print('假设已封存到 reports/lab_e_my_hypothesis.json（先于任何 Variant 结果）。开始 Variant 训练。')'''),
 m('''### Variant 训练等待说明
- 与 Lab C 相同：12 Epoch、约 30–60 秒；若这是本机今天第一次训练，还会多出 ~1 分钟 MIOpen JIT 预热（只发生一次）。
- **这期间不要启动其他训练或评测**——A/B 的"相同 GPU 条件"要求本臂独占显卡，否则计时不可比。''',
   '''### Variant training wait note
- Same as Lab C: 12 epochs, ~30–60 s; if this is the machine's first training today, add a one-time ~1-minute MIOpen JIT warm-up.
- **Do not start other training or evaluation during this arm** — the "identical GPU conditions" requirement means this arm must own the GPU, or timings are not comparable.'''),
 c('''# Variant：只修改 imgsz；额外不同的仅是 run output name/project
RUN_VARIANT = True
variant_args = dict(args)
variant_args.update(imgsz=416, project=str(home / 'runs/lab_e'), name='imgsz416', exist_ok=True)
changed = {key: (args[key], variant_args[key]) for key in args if args[key] != variant_args[key]}
print('Changed training arguments:', changed)
assert set(changed) == {'imgsz', 'project', 'name'}
if RUN_VARIANT:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    init_model = (str(home / 'artifacts/yolo26n.pt')
                  if (home / 'artifacts/yolo26n.pt').is_file() else cfg['model'])
    variant = YOLO(init_model)  # 注意：不能用 baseline best.pt 作为 B 的初始权重！
    variant.train(**variant_args)
    torch.cuda.synchronize()
    run_dir = Path(variant.trainer.save_dir)
    variant_best = run_dir / 'weights/best.pt'
    assert variant_best.is_file()
    variant_record = {'run_dir': str(run_dir), 'weights': str(variant_best),
                      'sha256': sha256(variant_best),
                      'elapsed_seconds': round(time.perf_counter() - t0, 2),
                      'peak_allocated_GiB': round(torch.cuda.max_memory_allocated() / 2**30, 3),
                      'args': variant_args, 'system': system_info()}
    write_json(home / 'reports/lab_e_variant.json', variant_record)
    print('Variant training finished:', variant_record)
else:
    print('RUN_VARIANT=False: without real variant artifacts NO A/B conclusion can be made.')'''),
 c('''# 相同验证集、验证输入和运行条件，分别评测真实权重。
variant_record = json.loads((home / 'reports/lab_e_variant.json').read_text(encoding='utf-8'))
assert Path(variant_record['weights']).exists()
reports = []
for label, rec in [('baseline', base_record), ('imgsz416', variant_record)]:
    m = YOLO(rec['weights']).val(data=cfg['dataset_yaml'], split='val',
                                 imgsz=640, device=device, batch=8, workers=0,
                                 plots=False, verbose=False)
    scores = get_metrics_summary(m)
    scores.update({'variant': label, 'train_imgsz': rec['args']['imgsz'],
                   'training_time_seconds': rec['elapsed_seconds'],
                   'train_peak_allocated_gib': rec['peak_allocated_GiB'],
                   'model_sha256': rec['sha256'], 'val_imgsz': 640, 'split': 'val'})
    reports.append(scores)
    print(label, scores)
write_csv(home / 'reports/lab_e_ab_comparison.csv', reports,
          ['variant', 'train_imgsz', 'training_time_seconds', 'train_peak_allocated_gib',
           'precision', 'recall', 'mAP50', 'mAP50_95', 'model_sha256', 'val_imgsz', 'split'])
write_json(home / 'reports/lab_e_ab_comparison.json', reports)'''),
 c('''# 图表使用实测数值，不预置任何提升率
from matplotlib import pyplot as plt

labels = [r['variant'] for r in reports]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, [r['mAP50_95'] for r in reports])
ax.set_ylabel('Validation mAP50-95')
ax.set_ylim(0, 1)
ax.set_title('Same validation images and evaluation imgsz=640')
for i, r in enumerate(reports):
    ax.text(i, r['mAP50_95'] + .01, f"{r['mAP50_95']:.3f}", ha='center')
fig.tight_layout()
fig.savefig(home / 'assets/lab_e_ab_mAP.png', dpi=150)
plt.show()
print('ΔmAP50-95 B−A:', round(reports[1]['mAP50_95'] - reports[0]['mAP50_95'], 4))'''),
 m('''### 四个指标一起读（不要只看一个数）
| 指标 | 回答的问题 | 单独引用的陷阱 |
|---|---|---|
| Precision | 报出来的框有多少是对的？ | 偏高可能只是"保守"（少报） |
| Recall | 真实目标找回了多少？ | 偏高可能靠"滥报"换来 |
| mAP50 | IoU≥0.5 的宽松综合质量 | 掩盖定位精度问题 |
| mAP50-95 | 严格定位下的综合质量 | 数值天然偏低；跨数据集不可比 |

**把你的两臂结果抄进这张表**（数据在 `reports/lab_e_ab_comparison.csv`，这是你的报告素材）：

| 臂 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| baseline (imgsz=512) |  |  |  |  |
| variant (imgsz=416) |  |  |  |  |

读法示例：如果一臂 **P 降 R 升**，说明模型变得更"敢报"但更不准——两个数必须成对解释；如果四项指标方向不一致，结论应写"不确定"而不是挑好看的数。''',
   '''### Read all four metrics together (never just one number)
| Metric | Question it answers | Trap when quoted alone |
|---|---|---|
| Precision | Of the boxes we reported, how many are right? | High P may just mean "conservative" (under-reporting) |
| Recall | How many real targets were found? | High R may come from over-reporting |
| mAP50 | Lenient overall quality at IoU≥0.5 | Hides localization quality |
| mAP50-95 | Strict overall quality | Naturally lower; not comparable across datasets |

**Copy your two arms into this table** (from `reports/lab_e_ab_comparison.csv` — this is your report material):

| Arm | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| baseline (imgsz=512) |  |  |  |  |
| variant (imgsz=416) |  |  |  |  |

How to read it: if one arm shows **P down, R up**, the model became bolder but less precise — the pair must be interpreted together; when the four metrics disagree, the honest conclusion is "inconclusive", not cherry-picking the flattering one.'''),
 m('''### 对照你的假设（Check your hypothesis）
打开 `reports/lab_e_my_hypothesis.json`，和上面的真实结果逐项对照：
- mAP 预测对了吗？如果错了，错在哪个推理环节？
- 训练时间预测对了吗？两臂耗时差了多少？这个差值在噪声范围内吗（参考你记录的冷/热启动差异）？

把对照结论写进你的实验报告——**预测错误 + 解释为什么错**，比"我猜对了"更能体现实验素养。''',
   '''### Check your hypothesis
Open `reports/lab_e_my_hypothesis.json` and compare it item by item against the real results above:
- Was your mAP prediction right? If not, which step of your reasoning failed?
- Was your training-time prediction right? How large is the gap, and is it within noise (compare with the cold/warm-start differences you observed)?

Write the comparison into your experiment report — **a wrong prediction plus an explanation of why** demonstrates more experimental maturity than "I guessed right".'''),
 m('''### 最终 Test：仅在选定模型之后运行一次
只使用 Validation 指标决定选择哪个候选模型。不要在查看 Test 分数后反过来改动超参数，再把同一 Test 分数当作独立最终成绩。''',
   '''### The final Test: run exactly once, after model selection
Use validation metrics alone to pick the winning candidate. Never peek at the test score, go back and tweak hyperparameters, and then present that same test score as an independent final result.'''),
 c('''RUN_FINAL_TEST = False  # 小组完成模型选择并冻结方案后，明确改成 True 执行一次
if RUN_FINAL_TEST:
    target = home / 'reports/lab_e_final_test.json'
    assert not target.exists(), '独立 Test 已有成绩。避免重复查看后调参。'
    selection = max(reports, key=lambda row: row['mAP50_95'])  # 仅由 Val 指标选择
    test_weight = (base_record['weights'] if selection['variant'] == 'baseline'
                   else variant_record['weights'])
    final_test = YOLO(test_weight).val(data=cfg['dataset_yaml'], split='test', imgsz=640,
                                       device=device, batch=8, workers=0, plots=False,
                                       verbose=False)
    result = get_metrics_summary(final_test)
    result.update({'split': 'test', 'selected_by': 'validation mAP50-95',
                   'model_variant': selection['variant'],
                   'model_sha256': sha256(Path(test_weight)),
                   'warning': 'Tiny classroom split; not a statistically reliable '
                              'industrial benchmark.'})
    write_json(target, result)
    print('Frozen final test result:', result)
else:
    print('Test is sealed. Current metrics are Validation only.')'''),
 m('''### 结论必须回答五个问题
1. 只改了哪一个**算法实验变量**？`project/name` 是日志位置变化，不是算法变量。
2. 在**完全一致的验证集和评测输入尺寸**下，mAP50-95 是增加还是减少？
3. 哪组训练耗时更短？同一云端 GPU 是否有其他用户负载干扰？
4. 显存数字是 PyTorch allocated peak，不是整卡显存占用。
5. 这个结果能否推广到更大模型、不同硬件和工业现场？为什么不能直接断言？

**交付**：`lab_e_variant.json`、`lab_e_ab_comparison.csv`、`lab_e_ab_mAP.png`、`lab_e_my_hypothesis.json`、文字分析。

**面试可能这样问**：你说做过一次"受控实验"——你怎么保证只有一个变量？为什么训练耗时不能当推理延迟？单次 A/B 能证明统计显著吗？如果面试官让你再提升 2 个点，你的下一步实验设计是什么？

**进阶**：用多个随机种子重复实验；在真正独立的业务数据上验证；在第二天再比较 ONNX/OpenVINO、GPU/ROCm 推理速度、显存与准确性。''',
   '''### Your conclusion must answer five questions
1. Which **single experimental variable** changed? `project/name` are log locations, not algorithm variables.
2. Under **identical validation data and evaluation input size**, did mAP50-95 go up or down?
3. Which arm trained faster? Was any other user loading the same cloud GPU?
4. The memory figure is PyTorch's allocated peak, not whole-GPU VRAM usage.
5. Can this result generalize to larger models, other hardware and production sites? Why can't you just assert it?

**Deliverables**: `lab_e_variant.json`, `lab_e_ab_comparison.csv`, `lab_e_ab_mAP.png`, `lab_e_my_hypothesis.json`, written analysis.

**Interview-style questions**: you claim you ran a "controlled experiment" — how do you guarantee exactly one variable changed? Why can't training time stand in for inference latency? Can a single A/B run prove statistical significance? If the interviewer asks for 2 more mAP points, what is your next experiment design?

**Going further**: repeat with multiple random seeds; validate on truly independent business data; on day 2 compare ONNX/OpenVINO vs GPU/ROCm inference speed, memory and accuracy.''')
], '30–40 分钟（完整训练可能更久）','30–40 minutes (full training may take longer)','控制变量 → 重新训练 → 同口径比较','controlled variable → retrain → same-protocol comparison')
