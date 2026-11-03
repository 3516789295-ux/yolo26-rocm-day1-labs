"""Classroom dataset, evaluation, provenance and output helpers.

No torch import here: dataset QA and unit tests run on CPU/offline.
Only approved, extracted COCO128 or user-owned datasets are processed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COCO128_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip'
SOURCE_NAMES = {0: 'person', 2: 'car', 56: 'chair'}
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def root() -> Path:
    """Persistent course location. AMD Radeon Cloud commonly mounts /workspace."""
    env = os.getenv('YOLO_COURSE_HOME')
    if env:
        return Path(env).expanduser().resolve()
    # /workspace is a Linux Radeon Cloud convention; on Windows Path('/workspace')
    # resolves drive-relative (<drive>:\workspace) and must not hijack the default,
    # so the project-local directory is used instead.
    if os.name == 'posix' and Path('/workspace').exists():
        return Path('/workspace/yolo_course').resolve()
    return (Path.cwd() / 'yolo_course').resolve()


def ensure_dirs(home: Path | None = None) -> Path:
    home = Path(home or root())
    for part in ('data/raw', 'data/processed', 'runs', 'reports', 'assets', 'artifacts'):
        (home / part).mkdir(parents=True, exist_ok=True)
    return home


def config_path(home: Path | None = None) -> Path:
    return Path(home or root()) / 'course_config.json'


def load_config(home: Path | None = None) -> dict:
    home = ensure_dirs(home)
    p = config_path(home)
    defaults = {'home': str(home), 'model': 'yolo26n.pt',
                'dataset_yaml': '', 'dataset_kind': '', 'baseline_weights': '',
                'baseline_run': '', 'model_licence_reviewed': False}
    if p.exists():
        defaults.update(json.loads(p.read_text(encoding='utf-8')))
    return defaults


def save_config(changes: dict, home: Path | None = None) -> dict:
    home = ensure_dirs(home)
    cfg = load_config(home)
    cfg.update(changes)
    tmp = config_path(home).with_suffix('.json.tmp')
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(config_path(home))
    return cfg


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict | list):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def list_images(folder: Path):
    return sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in EXTS)


def download_coco128(home: Path | None = None, allow_download: bool = True) -> Path:
    """Source remains unmodified. Download only from official Ultralytics release.

    Honours HTTPS_PROXY / HTTP_PROXY environment. Validates zip entries to avoid zip slip.
    """
    home = ensure_dirs(home)
    target = home / 'data/raw/coco128'
    if (target / 'images/train2017').is_dir() and (target / 'labels/train2017').is_dir():
        return target
    if not allow_download:
        raise FileNotFoundError(f'Missing COCO128 at {target}; upload dataset to that path or enable download.')
    archive = home / 'data/raw/coco128.zip'
    if not archive.exists():
        try:
            request = urllib.request.Request(COCO128_URL, headers={'User-Agent': 'Radeon-YOLO-Day1-Education/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open('wb') as f:
                shutil.copyfileobj(response, f)
        except Exception as e:
            archive.unlink(missing_ok=True)
            raise RuntimeError(f'Cannot download COCO128 from official URL: {e}. Upload coco128.zip to {archive}.') from e
    extract_root = home / 'data/raw'
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            dst = (extract_root / item.filename).resolve()
            if not dst.is_relative_to(extract_root.resolve()):
                raise ValueError(f'Unsafe zip path: {item.filename}')
        z.extractall(extract_root)
    if not (target / 'images/train2017').exists():
        raise FileNotFoundError(f'Archive layout unexpected; expected {target / "images/train2017"}')
    return target


def label_errors(text: str, nclasses: int) -> list[str]:
    errs = []
    for line_num, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            errs.append(f'L{line_num}: expected 5 fields, got {len(parts)}'); continue
        try:
            floats = list(map(float, parts))
        except ValueError:
            errs.append(f'L{line_num}: nonnumeric field'); continue
        cls, x, y, w, h = floats
        if not np.isfinite(floats).all():
            errs.append(f'L{line_num}: NaN/inf'); continue
        if not cls.is_integer() or not (0 <= int(cls) < nclasses):
            errs.append(f'L{line_num}: invalid class id {cls}')
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errs.append(f'L{line_num}: invalid normalized xywh')
        if x - w / 2 < -1e-5 or y - h / 2 < -1e-5 or x + w / 2 > 1+1e-5 or y+h/2 > 1+1e-5:
            errs.append(f'L{line_num}: box extends outside image')
    return errs


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not Path(label_path).exists():
        return []
    rows = []
    for ln in Path(label_path).read_text(encoding='utf-8').splitlines():
        if not ln.strip(): continue
        vals = list(map(float, ln.split()))
        if len(vals) != 5: raise ValueError(f'Invalid label in {label_path}')
        rows.append((int(vals[0]), *vals[1:]))
    return rows


def dhash(path: Path, width: int = 8) -> int:
    with Image.open(path) as im:
        gray = np.asarray(im.convert('L').resize((width+1, width), Image.Resampling.LANCZOS))
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    out = 0
    for bit in bits: out = (out << 1) | int(bit)
    return out


def _groups_of_related_images(images: list[Path], near_threshold: int = 3):
    """Group exact and dHash-near images. Heuristic, NOT a semantic scene leakage proof."""
    n = len(images); parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def join(i,j):
        a,b = find(i),find(j)
        if a != b: parent[b]=a
    signatures = [dhash(p) for p in images]
    digests = [sha256(p) for p in images]
    exact = {}
    for i, h in enumerate(digests):
        if h in exact: join(i, exact[h])
        else: exact[h] = i
    # Educational small-dataset check; avoid quadratic explosion on industrial datasets.
    if n <= 1500:
        for i in range(n):
            for j in range(i+1,n):
                if (signatures[i] ^ signatures[j]).bit_count() <= near_threshold:
                    join(i,j)
    result = defaultdict(list)
    for i in range(n): result[find(i)].append(i)
    return list(result.values()), signatures, digests


def _choose_splits(groups: list[list[int]], n: int, seed: int = 42) -> dict[int,str]:
    """Group-disjoint approximate 70/15/15 split by image count."""
    rng=random.Random(seed)
    randomized = groups[:]; rng.shuffle(randomized)
    randomized.sort(key=len, reverse=True)
    target={'train':.70*n, 'val':.15*n, 'test':.15*n}
    count=Counter()
    assign={}
    for group in randomized:
        choices=['train','val','test']
        # Allocate proportionally to remaining quota, not always to largest split.
        name=max(choices, key=lambda s: (target[s]-count[s])/target[s])
        for idx in group:assign[idx]=name
        count[name]+=len(group)
    if min(count[s] for s in ('train','val','test')) < 1:
        raise ValueError(f'Groups too large to create nonempty disjoint splits: {dict(count)}')
    return assign


def prepare_coco128_demo(source: Path, dest: Path, seed: int=42) -> dict:
    """Create an honestly named *teaching* dataset with independent image splits.

    Three classes: person, car, chair (mapped to 0,1,2).
    Do not present tiny split metrics as industrial generalization results.
    """
    source=Path(source);dest=Path(dest)
    images=list_images(source/'images/train2017')
    if len(images)<40: raise ValueError(f'Expected COCO128 images, got {len(images)}')
    groups, phashes, digests = _groups_of_related_images(images)
    mapping=_choose_splits(groups,len(images),seed)
    if dest.exists(): shutil.rmtree(dest)
    names=['person','car','chair']; convert={0:0,2:1,56:2}
    counts=Counter(); cls_count=Counter(); manifest=[]
    for i,img in enumerate(images):
        split=mapping[i]; label=source/'labels/train2017'/f'{img.stem}.txt'
        raw=read_labels(label)
        selected=[(convert[row[0]],*row[1:]) for row in raw if row[0] in convert]
        dst_img=dest/'images'/split/img.name
        dst_lab=dest/'labels'/split/f'{img.stem}.txt'
        dst_img.parent.mkdir(parents=True,exist_ok=True);dst_lab.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(img,dst_img)
        dst_lab.write_text(''.join(f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for cls,x,y,w,h in selected),encoding='utf-8')
        errs=label_errors(dst_lab.read_text(encoding='utf-8'),len(names))
        if errs: raise ValueError(f'{dst_lab}: {errs}')
        counts[split]+=1;cls_count.update(f'{split}:{v[0]}' for v in selected)
        manifest.append({'file':img.name,'split':split,'sha256':digests[i], 'dhash':f'{phashes[i]:016x}', 'targets':len(selected)})
    yaml_path=dest/'data.yaml'
    import yaml
    yaml_path.write_text(yaml.safe_dump({'path':str(dest.resolve()), 'train':'images/train','val':'images/val',
                                        'test':'images/test','names':dict(enumerate(names))},allow_unicode=True,sort_keys=False),encoding='utf-8')
    write_csv(dest/'split_manifest.csv',manifest,['file','split','sha256','dhash','targets'])
    report={'source':'COCO128 (COCO train2017 first 128 images)', 'pedagogical_only':True,
            'note':'Image-disjoint internal partition; still tiny and biased, not a COCO benchmark, near-duplicate heuristic is not a guarantee.',
            'splits':dict(counts),'class_counts':dict(cls_count),'groups':len(groups),
            'seed':seed,'names':names,'data_yaml':str(yaml_path), 'source_url':COCO128_URL}
    write_json(dest/'dataset_manifest.json',report)
    return report


def label_path_for_image(img: Path) -> Path:
    """YOLO convention: images/.../name.ext -> labels/.../name.txt."""
    img = Path(img)
    parts = list(img.parts)
    indices = [i for i, part in enumerate(parts) if part == 'images']
    if not indices:
        raise ValueError(f'Not under an images/ directory: {img}')
    parts[indices[-1]] = 'labels'
    return Path(*parts).with_suffix('.txt')


def inspect_dataset(data_yaml: Path) -> dict:
    import yaml
    yaml_path=Path(data_yaml).resolve()
    spec=yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    dsroot=Path(spec.get('path',yaml_path.parent))
    if not dsroot.is_absolute(): dsroot=(yaml_path.parent/dsroot).resolve()
    names=spec['names']
    nclasses=len(names)
    problem=[]; counts={}; labels_by_class=Counter(); hashes_by_split=defaultdict(set)
    stem_names=defaultdict(set); paths={}; signatures_by_split=defaultdict(list)
    for split in ('train','val','test'):
        rel=spec.get(split)
        if not rel:
            problem.append(f'Missing {split} directory in yaml'); continue
        img_dir=dsroot/rel
        if not img_dir.exists(): problem.append(f'Missing dir: {img_dir}'); continue
        split_imgs=list_images(img_dir)
        counts[split]={'images':len(split_imgs),'boxes':0,'empty_or_missing_labels':0}
        paths[split]=split_imgs
        for img in split_imgs:
            lab=label_path_for_image(img)
            if not lab.exists() or not lab.read_text(encoding='utf-8').strip():
                counts[split]['empty_or_missing_labels']+=1
            if lab.exists():
                issues=label_errors(lab.read_text(encoding='utf-8'),nclasses)
                if issues:problem.extend(f'{lab}: {x}' for x in issues)
                else:
                    rows=read_labels(lab)
                    counts[split]['boxes']+=len(rows)
                    labels_by_class.update(f'{split}:{r[0]}' for r in rows)
            h=sha256(img)
            hashes_by_split[split].add(h)
            if sum(len(v) for v in paths.values()) <= 600:
                signatures_by_split[split].append((img.name,dhash(img)))
            stem_names[split].add(img.name)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        collisions = hashes_by_split[a] & hashes_by_split[b]
        if collisions:problem.append(f'EXACT IMAGE LEAKAGE {a}/{b}: {len(collisions)} duplicate image hashes')
        common_names=stem_names[a] & stem_names[b]
        if common_names:problem.append(f'Same filenames across splits {a}/{b}: {len(common_names)}')
    warnings = []
    total_imgs = sum(len(v) for v in paths.values())
    if total_imgs <= 600:
        # Only flag possible perceptual duplicates for human review; false positives are possible.
        for a,b in [('train','val'),('train','test'),('val','test')]:
            suspects=[]
            for name_a, h_a in signatures_by_split[a]:
                for name_b, h_b in signatures_by_split[b]:
                    if (h_a ^ h_b).bit_count() <= 3:
                        suspects.append((name_a,name_b))
                        if len(suspects)>=5:break
                if len(suspects)>=5:break
            if suspects:
                warnings.append(f'Potential cross-split dHash near duplicates {a}/{b}: {suspects}; manually inspect scenes')
    else:
        warnings.append('Skipped quadratic perceptual near-duplicate scan for >600 images; use scene/video grouping.')
    for s in ('train','val','test'):
        if not counts.get(s,{}).get('images',0):problem.append(f'No images in {s}')
    return {'yaml':str(yaml_path),'root':str(dsroot),'names':names,'splits':counts,
            'class_counts':dict(labels_by_class),'problems':problem,'warnings':warnings,
            'pass':not problem, 'sample_paths':paths}


def plot_labels(images: list[Path], names: list[str], max_items:int=6):
    import matplotlib.pyplot as plt
    images=images[:max_items]
    fig,ax=plt.subplots(2,3,figsize=(12,7)); axes=ax.ravel()
    for a in axes:a.axis('off')
    for a,img in zip(axes,images):
        with Image.open(img) as raw:
            vis=raw.convert('RGB').copy()
        split=img.parent.name
        lab=label_path_for_image(img)
        draw=ImageDraw.Draw(vis)
        W,H=vis.size
        for cls,x,y,w,h in read_labels(lab):
            xa=(x-w/2)*W;xb=(x+w/2)*W;ya=(y-h/2)*H;yb=(y+h/2)*H
            draw.rectangle([xa,ya,xb,yb],outline=(22,159,225),width=max(2,round(min(W,H)/200)))
            draw.text((xa,max(0,ya-14)), names[cls],fill=(255,40,60),stroke_width=1,stroke_fill=(255,255,255))
        a.imshow(vis);a.set_title(f'{img.stem[:18]} · {split}',fontsize=9)
    fig.tight_layout();return fig


def xywhn_to_xyxy(box, width, height):
    _,x,y,w,h=box
    return [(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]


def iou_xyxy(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]);area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(area_a+area_b-inter) if area_a+area_b-inter>0 else 0.0


def match_predictions(result, label_path:Path, iou_threshold:float=0.50):
    """One-to-one teaching-level matching at fixed conf threshold, NOT COCO mAP.

    Returns TP/FP/FN with per-prediction error records; class mismatch always FP + FN.
    """
    hh,ww=result.orig_shape
    gts=[(cls,xywhn_to_xyxy((cls,x,y,w,h),ww,hh)) for cls,x,y,w,h in read_labels(label_path)]
    pred=[]
    if result.boxes is not None:
        coords=result.boxes.xyxy.cpu().numpy().tolist()
        clss=result.boxes.cls.cpu().numpy().tolist()
        confs=result.boxes.conf.cpu().numpy().tolist()
        pred=sorted([(int(k),bbox,float(cf)) for bbox,k,cf in zip(coords,clss,confs)],key=lambda t:-t[2])
    used=set(); tp=0; fp=0; rows=[]
    for cls,box,confidence in pred:
        candidates=[(iou_xyxy(box,gtbox),i) for i,(gtcls,gtbox) in enumerate(gts) if gtcls==cls and i not in used]
        best=max(candidates,default=(0.,-1))
        if best[0]>=iou_threshold:
            used.add(best[1]);tp+=1
        else:
            fp+=1
            nearest=max((iou_xyxy(box,b) for _,b in gts),default=0.)
            etype='FP_possible_class_or_duplicate' if nearest>=iou_threshold else ('FP_localization_or_background' if nearest>=0.1 else 'FP_background')
            rows.append({'error':etype,'class_id':cls,'score':round(confidence,4),'max_iou_gt':round(nearest,3)})
    fn=len(gts)-len(used)
    for i,(cls,box) in enumerate(gts):
        if i not in used:rows.append({'error':'FN','class_id':cls,'score':'','max_iou_gt':''})
    return {'TP':tp,'FP':fp,'FN':fn,'errors':rows}


def get_metrics_summary(metrics):
    box=metrics.box
    return {'precision':float(box.mp),'recall':float(box.mr),'mAP50':float(box.map50),
            'mAP50_95':float(box.map)}


def system_info():
    d={'python':sys.version.split()[0], 'platform':platform.platform(),'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        import torch
        d.update({'torch':torch.__version__,'torch_hip':torch.version.hip,
                  'gpu_visible':bool(torch.cuda.is_available()),
                  'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except ImportError:d['torch']='missing'
    try:
        import ultralytics
        d['ultralytics']=ultralytics.__version__
    except ImportError:d['ultralytics']='missing'
    return d


def require_rocm():
    import torch
    hip=torch.version.hip
    assert hip is not None, ('This notebook requires an AMD ROCm PyTorch build (torch.version.hip is None). '
                              'Install the matching ROCm wheel in a separate environment first.')
    assert torch.cuda.is_available(), 'ROCm PyTorch installed but no GPU visible; check /dev/kfd, /dev/dri and GPU allocation.'
    # A platform-preinstalled stack can pair a working ROCm torch with a torchvision
    # built without a GPU NMS kernel. Every detection lab then dies deep inside
    # Ultralytics (Lab A cell 5) with an opaque NotImplementedError. Probe the
    # operator here so the failure is early and tells the reader what to do.
    try:
        import torchvision
        torchvision.ops.nms(torch.zeros((2, 4), device='cuda'),
                            torch.zeros(2, device='cuda'), 0.5)
    except Exception as exc:
        raise AssertionError(
            'torchvision GPU NMS is unavailable: the installed torchvision was built without a '
            'ROCm/CUDA nms kernel, so every detection lab would fail. Install the pinned stack '
            '(docs/RADEON_CLOUD_LINUX_SETUP.md section 2-B) instead of the platform stack. '
            f'Original error: {type(exc).__name__}: {exc}'
        ) from exc
    print('AMD ROCm verified:',torch.cuda.get_device_name(0),'HIP:',hip)
    return 0
