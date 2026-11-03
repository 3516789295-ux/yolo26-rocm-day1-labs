"""Import X-AnyLabeling YOLO *detection* exports into a safe classroom dataset.

The original images/labels are never modified. CPU-only; no PyTorch needed.
"""
from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from course.lab_utils import (
    _groups_of_related_images, _choose_splits, inspect_dataset,
    label_errors, list_images, sha256, write_csv, write_json,
)


def _classes(classes_file: Path) -> list[str]:
    if not classes_file.is_file():
        raise FileNotFoundError(f"Missing X-AnyLabeling classes.txt: {classes_file}")
    names = [x.strip() for x in classes_file.read_text(encoding="utf-8-sig").splitlines()]
    if not names or any(not x for x in names) or len(set(names)) != len(names):
        raise ValueError("classes.txt must contain unique, nonempty class names in class-ID order")
    return names


def _scene_groups(path: Path | None, images: list[Path]) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scene grouping CSV missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"image", "scene"} <= set(reader.fieldnames):
            raise ValueError("scene CSV requires image,scene columns")
        mapping = {}
        for row in reader:
            name, scene = row["image"].strip(), row["scene"].strip()
            if not name or not scene or name in mapping:
                raise ValueError("Scene CSV has a blank or duplicate image/scene entry")
            mapping[name] = scene
    expected = {p.name for p in images}
    if set(mapping) != expected:
        raise ValueError(f"Scene CSV must cover every image exactly once: missing={sorted(expected-set(mapping))}, extra={sorted(set(mapping)-expected)}")
    return mapping


def import_anylabeling_yolo(images_dir: Path, labels_dir: Path, classes_txt: Path,
                             dest: Path, *, scene_csv: Path | None = None,
                             seed: int = 42) -> dict:
    """Validate X-AnyLabeling YOLO detection exports, split, and copy safely.

    Refuses to overwrite a destination. Manual scene IDs (optional) and
    perceptual-image groups stay in the same split.
    """
    images_dir = Path(images_dir).expanduser().resolve()
    labels_dir = Path(labels_dir).expanduser().resolve()
    classes_txt = Path(classes_txt).expanduser().resolve()
    dest = Path(dest).expanduser().resolve()
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected images and labels directories: {images_dir} / {labels_dir}")
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing classroom dataset: {dest}")
    if images_dir in dest.parents or labels_dir in dest.parents:
        raise ValueError("Output must not be within the original annotation export")
    names = _classes(classes_txt)
    images = list_images(images_dir)
    if len(images) < 9:
        raise ValueError("At least 9 images required for a three-way teaching split")
    if len({x.name for x in images}) != len(images) or len({x.stem for x in images}) != len(images):
        raise ValueError("Image names/stems must be unique (even across extensions)")
    labels = {p.stem: p for p in labels_dir.glob("*.txt") if p.name != classes_txt.name}
    extras = set(labels) - {p.stem for p in images}
    if extras:
        raise ValueError(f"Orphan YOLO label files: {sorted(extras)}")
    missing = [p.name for p in images if p.stem not in labels]
    if missing:
        raise ValueError(f"Missing labels: {missing}. Create an explicit empty .txt for verified negatives")
    issues = []
    box_count = 0
    for im in images:
        label = labels[im.stem]
        source_text = label.read_text(encoding="utf-8-sig")
        bad = label_errors(source_text, len(names))
        if bad:
            issues.extend(f"{label}: {error}" for error in bad)
        box_count += len(source_text.splitlines()) if source_text.strip() else 0
    if issues:
        raise ValueError(
            "Invalid YOLO detection labels; no files copied:\n" + "\n".join(issues[:30]) +
            f"\nHint/提示: 每行应为 5 个数 'class x y w h'；类别 ID 必须是 0 到 {len(names) - 1} 的整数"
            f"（classes.txt 共 {len(names)} 行）；坐标是 0–1 的归一化小数。"
            f" / Each line must be 'class x y w h'; class IDs are integers in [0, {len(names) - 1}] "
            f"({len(names)} lines in classes.txt); coordinates are normalized 0-1.")
    scene_by_name = _scene_groups(scene_csv, images)
    groups, _perceptual, digests = _groups_of_related_images(images)
    parent = list(range(len(images)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    for group in groups:
        for other in group[1:]:
            union(group[0], other)
    first_by_scene = {}
    for i, im in enumerate(images):
        scene = scene_by_name.get(im.name)
        if scene:
            if scene in first_by_scene:
                union(first_by_scene[scene], i)
            else:
                first_by_scene[scene] = i
    grouped = defaultdict(list)
    for i in range(len(images)):
        grouped[find(i)].append(i)
    final_groups = list(grouped.values())
    if len(final_groups) < 3:
        raise ValueError("Fewer than three independent groups; collect more independent scenes")
    mapping = _choose_splits(final_groups, len(images), seed)
    split_counts = Counter(mapping.values())
    if any(split_counts[x] == 0 for x in ("train", "val", "test")):
        raise ValueError("Empty split after scene grouping; collect more independent scenes")
    manifest = []
    try:
        for i, image in enumerate(images):
            split = mapping[i]
            new_im = dest / "images" / split / image.name
            new_lab = dest / "labels" / split / f"{image.stem}.txt"
            new_im.parent.mkdir(parents=True, exist_ok=True)
            new_lab.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, new_im)
            shutil.copy2(labels[image.stem], new_lab)
            manifest.append({"image": image.name, "split": split, "sha256": digests[i],
                             "scene": scene_by_name.get(image.name, ""),
                             "label_sha256": sha256(labels[image.stem])})
        import yaml
        data_yaml = dest / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({"path": str(dest), "train": "images/train",
                                             "val": "images/val", "test": "images/test",
                                             "names": dict(enumerate(names))},
                                            sort_keys=False, allow_unicode=True), encoding="utf-8")
        write_csv(dest / "split_manifest.csv", manifest,
                  ["image", "split", "sha256", "scene", "label_sha256"])
        report = {"dataset_kind": "X-AnyLabeling YOLO detection export",
                  "source_images": str(images_dir), "source_labels": str(labels_dir),
                  "source_classes": str(classes_txt), "scene_csv": str(scene_csv or ""),
                  "images": len(images), "boxes": box_count, "classes": names,
                  "split_counts": dict(split_counts), "independent_groups": len(final_groups),
                  "seed": seed, "data_yaml": str(data_yaml),
                  "limits": "Small classroom split, not an industrial benchmark; humans must review scene similarity."}
        write_json(dest / "dataset_manifest.json", report)
        qa = inspect_dataset(data_yaml)
        if not qa["pass"]:
            raise ValueError("Prepared dataset QA failed: " + "; ".join(qa["problems"]))
        report["qa_warnings"] = qa["warnings"]
        return report
    except Exception:
        if dest.exists():
            shutil.rmtree(dest)
        raise
