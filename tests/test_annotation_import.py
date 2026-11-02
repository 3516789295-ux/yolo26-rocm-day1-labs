"""Offline regressions for the X-AnyLabeling manual export workflow."""
from pathlib import Path

import pytest

from course.annotation_import import import_anylabeling_yolo
from course.lab_utils import inspect_dataset, sha256
from tests.test_lab_utils import make_dataset


def make_export(tmp_path, n=30):
    raw = make_dataset(tmp_path, n=n)
    images = raw / "images/train2017"
    labels = raw / "labels/train2017"
    classes = tmp_path / "classes.txt"
    classes.write_text("person\ncar\nchair\n", encoding="utf-8")
    for file in labels.glob("*.txt"):
        lines = []
        for row in file.read_text().splitlines():
            parts = row.split()
            parts[0] = {"0": "0", "2": "1", "56": "2"}[parts[0]]
            lines.append(" ".join(parts))
        file.write_text("\n".join(lines) + "\n")
    return images, labels, classes


def test_import_preserves_sources(tmp_path):
    images, labels, classes = make_export(tmp_path)
    before = {p.name: sha256(p) for p in labels.glob("*.txt")}
    dest = tmp_path / "student"
    report = import_anylabeling_yolo(images, labels, classes, dest)
    assert report["images"] == 30
    qa = inspect_dataset(dest / "data.yaml")
    assert qa["pass"], qa["problems"]
    assert sum(x["images"] for x in qa["splits"].values()) == 30
    assert {p.name: sha256(p) for p in labels.glob("*.txt")} == before
    with pytest.raises(FileExistsError):
        import_anylabeling_yolo(images, labels, classes, dest)


def test_rejects_invalid_and_missing_labels(tmp_path):
    images, labels, classes = make_export(tmp_path)
    one = next(labels.glob("*.txt"))
    one.write_text("3 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="Invalid YOLO"):
        import_anylabeling_yolo(images, labels, classes, tmp_path / "student")
    assert not (tmp_path / "student").exists()
    one.unlink()
    with pytest.raises(ValueError, match="Missing labels"):
        import_anylabeling_yolo(images, labels, classes, tmp_path / "student")


def test_grouped_scenes_never_cross_splits(tmp_path):
    images, labels, classes = make_export(tmp_path)
    scene = tmp_path / "scenes.csv"
    all_images = sorted(images.glob("*.png"))
    scene.write_text("image,scene\n" + "".join(
        f"{p.name},scene{i//3}\n" for i, p in enumerate(all_images)
    ), encoding="utf-8")
    dest = tmp_path / "student"
    import_anylabeling_yolo(images, labels, classes, dest, scene_csv=scene)
    import csv
    with (dest / "split_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for row in rows:
        groups.setdefault(row["scene"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
