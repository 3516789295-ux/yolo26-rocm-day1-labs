"""Offline CI of dataset QA and evaluation logic; NO GPU dependency."""
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from course.lab_utils import (
    label_errors, iou_xyxy, read_labels, sha256, dhash, _groups_of_related_images,
    prepare_coco128_demo, inspect_dataset, load_config, save_config, xywhn_to_xyxy
)


def test_labels_and_iou():
    assert label_errors('0 0.5 0.5 0.5 0.5', 3) == []
    assert label_errors('', 3) == []  # empty negative image is valid
    assert label_errors('3 0.5 0.5 0.5 0.5',3)
    assert label_errors('0 -0.1 0.5 0.5 0.5',3)
    assert label_errors('0 0.5 0.5 0.0 0.5',3)
    assert label_errors('0 nan 0.5 0.1 0.1',3)
    assert label_errors('0 0.5',3)
    assert label_errors('0 0.9 0.5 0.5 0.5',3)  # outside image
    assert iou_xyxy((0,0,2,2),(1,1,3,3)) == pytest.approx(1/7)
    assert iou_xyxy((0,0,2,2),(4,4,6,6))==0
    assert xywhn_to_xyxy((0,.5,.5,.5,.5),640,480)==[160,120,480,360]


def test_config_round_trip(tmp_path):
    save_config({'model':'yolo26n.pt','dataset_kind':'test'},tmp_path)
    assert load_config(tmp_path)['dataset_kind']=='test'


def make_dataset(tmp_path, n=54):
    source=tmp_path/'coco128'
    image_dir=source/'images/train2017';image_dir.mkdir(parents=True)
    label_dir=source/'labels/train2017';label_dir.mkdir(parents=True)
    for i in range(n):
        # Make distinctly varied deterministic visual patterns (not identical flat backgrounds).
        im=Image.new('RGB',(64,64),(7+i*7%250,(i*19)%255,(i*37)%255))
        draw=ImageDraw.Draw(im)
        draw.polygon([(i%48,4),(30,(i*3)%40),(60,55)],fill=((i*11)%255,(i*13)%255,(i*17)%255))
        im.save(image_dir/f'sample_{i:04d}.png')
        (label_dir/f'sample_{i:04d}.txt').write_text(
            ('0 0.500000 0.500000 0.200000 0.200000\n' if i%2==0 else
             '2 0.400000 0.400000 0.300000 0.300000\n') +
            ('56 0.600000 0.650000 0.100000 0.120000\n' if i%5==0 else ''),
            encoding='utf-8')
    return source


def test_dataset_build_and_qa(tmp_path):
    source=make_dataset(tmp_path)
    dest=tmp_path/'processed'; report=prepare_coco128_demo(source,dest,seed=42)
    assert sum(report['splits'].values())==54
    qa=inspect_dataset(dest/'data.yaml')
    assert qa['pass'],qa['problems']
    assert sum(x['images'] for x in qa['splits'].values())==54
    assert sum(x['boxes'] for x in qa['splits'].values())>54
    assert set(qa['names'].values())=={'person','car','chair'}
    # deterministic (for same source)
    old=(dest/'split_manifest.csv').read_bytes()
    prepare_coco128_demo(source,dest,seed=42)
    assert old==(dest/'split_manifest.csv').read_bytes()
    # intentionally introduce leaked image byte duplicate in test folder
    original=qa['sample_paths']['train'][0]
    added=dest/'images/test/leak.png'
    added.write_bytes(original.read_bytes())
    issue=inspect_dataset(dest/'data.yaml')
    assert not issue['pass'] and 'EXACT IMAGE LEAKAGE' in ' '.join(issue['problems'])


def test_exact_hash_grouping(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.png';c=tmp_path/'c.png'
    Image.new('RGB',(16,16),(255,0,0)).save(a)
    b.write_bytes(a.read_bytes())
    Image.new('RGB',(16,16),(0,0,255)).save(c)
    groups,_, hashes=_groups_of_related_images([a,b,c],near_threshold=-1)
    assert hashes[0]==hashes[1]
    assert any({0,1}.issubset(set(g)) for g in groups)
