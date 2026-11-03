"""Offline regressions for the run_all_labs.py P0 fixes.

P0-B: Lab B artifact verification must follow the dataset this run actually
      configured (COCO128 / X-AnyLabeling / YOLO_USER_DATA_YAML) instead of a
      hardcoded COCO128 path, and must reject stale leftovers.
P0-C: the kernel probe code must be executable and self-describing; the runner
      must refuse to start when the probe cannot prove the ROCm environment.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import time
import types
from contextlib import redirect_stdout
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('run_all_labs', PROJECT / 'scripts' / 'run_all_labs.py')
runner = importlib.util.module_from_spec(_spec)
sys.modules['run_all_labs'] = runner
_spec.loader.exec_module(runner)


def _make_course_home(tmp_path, kind, yaml_abs=None, qa_pass=True, fresh=True,
                      with_import_report=True):
    home = tmp_path / 'course'
    processed = home / 'data' / 'processed' / 'ds_v1'
    (processed / 'images' / 'train').mkdir(parents=True)
    (processed / 'images' / 'val').mkdir(parents=True)
    (processed / 'images' / 'test').mkdir(parents=True)
    yaml_p = Path(yaml_abs) if yaml_abs else processed / 'data.yaml'
    yaml_p.parent.mkdir(parents=True, exist_ok=True)
    yaml_p.write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {0: a}\n',
                      encoding='utf-8')
    (home / 'reports').mkdir(parents=True)
    (home / 'reports' / 'lab_b_dataset_qa.json').write_text(
        json.dumps({'pass': qa_pass, 'problems': []}), encoding='utf-8')
    cfg = {'dataset_yaml': str(yaml_p), 'dataset_kind': kind}
    if kind == 'X-AnyLabeling_student_reviewed_teaching' and with_import_report:
        (home / 'reports' / 'lab_b_annotation_import.json').write_text('{}', encoding='utf-8')
    (home / 'course_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    if not fresh:
        old = time.time() - 3600
        os.utime(home / 'course_config.json', (old, old))
        os.utime(home / 'reports' / 'lab_b_dataset_qa.json', (old, old))
    return home


def test_required_artifacts_has_no_hardcoded_lab_b_dataset():
    entries = [rel for rel, _ in runner.REQUIRED_ARTIFACTS['02']]
    assert entries == ['reports/lab_b_dataset_qa.json'], (
        'Lab B must not pin a fixed dataset path; the dataset is config-driven '
        f'(P0-B regression): {entries}')


def test_lab_b_verification_accepts_coco128_mode(tmp_path):
    home = _make_course_home(tmp_path, 'COCO128_repartitioned_teaching_only')
    assert runner._verify_lab_b_dataset(home, time.time()) == []


def test_lab_b_verification_accepts_xanylabeling_mode(tmp_path):
    home = _make_course_home(tmp_path, 'X-AnyLabeling_student_reviewed_teaching')
    assert runner._verify_lab_b_dataset(home, time.time()) == []
    # without the import evidence report it must fail
    (home / 'reports' / 'lab_b_annotation_import.json').unlink()
    problems = runner._verify_lab_b_dataset(home, time.time())
    assert any('lab_b_annotation_import.json' in p for p in problems)


def test_lab_b_verification_accepts_user_yaml_mode(tmp_path):
    external = tmp_path / 'external' / 'mydata.yaml'
    home = _make_course_home(tmp_path, 'user_provided', yaml_abs=external)
    assert runner._verify_lab_b_dataset(home, time.time()) == []


def test_lab_b_verification_rejects_stale_config(tmp_path):
    home = _make_course_home(tmp_path, 'COCO128_repartitioned_teaching_only', fresh=False)
    problems = runner._verify_lab_b_dataset(home, time.time())
    assert any('stale course_config.json' in p for p in problems)
    assert any('stale lab_b_dataset_qa.json' in p for p in problems)
    # a fresh run (session predates the files) must pass
    assert runner._verify_lab_b_dataset(home, time.time() - 7200) == []


def test_lab_b_verification_rejects_failed_qa_and_missing_yaml(tmp_path):
    home = _make_course_home(tmp_path, 'COCO128_repartitioned_teaching_only', qa_pass=False)
    problems = runner._verify_lab_b_dataset(home, time.time())
    assert any('pass=false' in p for p in problems)
    cfg = json.loads((home / 'course_config.json').read_text(encoding='utf-8'))
    cfg['dataset_yaml'] = str(home / 'nonexistent.yaml')
    (home / 'course_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    problems = runner._verify_lab_b_dataset(home, time.time())
    assert any('missing on disk' in p for p in problems)


def test_lab_b_verification_rejects_wrong_course_home_dataset(tmp_path):
    other = tmp_path / 'other_course'
    home = _make_course_home(tmp_path, 'COCO128_repartitioned_teaching_only')
    # dataset yaml pointing into a DIFFERENT course home
    cfg = json.loads((home / 'course_config.json').read_text(encoding='utf-8'))
    cfg['dataset_yaml'] = str(other / 'data' / 'processed' / 'x' / 'data.yaml')
    (other / 'data' / 'processed' / 'x').mkdir(parents=True)
    (other / 'data' / 'processed' / 'x' / 'data.yaml').write_text('names: {0: a}\n', encoding='utf-8')
    (home / 'course_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    problems = runner._verify_lab_b_dataset(home, time.time())
    assert any('outside' in p for p in problems)


def test_kernel_probe_code_executes_and_emits_marker():
    code = runner.KERNEL_PROBE_CODE
    compile(code, 'probe', 'exec')  # must be valid Python
    fake_torch = types.ModuleType('torch')
    fake_version = types.SimpleNamespace(hip='7.14-test')
    fake_cuda = types.SimpleNamespace(is_available=lambda: True,
                                      get_device_name=lambda i: 'AMD TEST GPU')
    fake_torch.version = fake_version
    fake_torch.cuda = fake_cuda
    fake_torch.__version__ = '2.12.0+rocm7.14.1'
    ns = {'sys': sys, 'json': json, '__name__': '__main__'}
    saved, sys.modules['torch'] = sys.modules.get('torch'), fake_torch
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code, ns)
    finally:
        if saved is not None:
            sys.modules['torch'] = saved
        else:
            sys.modules.pop('torch', None)
    out = buf.getvalue()
    assert 'KERNEL_PROBE_JSON:' in out
    info = json.loads(out.split('KERNEL_PROBE_JSON:', 1)[1])
    assert info['executable'] == sys.executable
    assert info['hip'] == '7.14-test'
    assert info['gpu'] == 'AMD TEST GPU'
    assert 'torch_error' not in info


def test_verify_artifacts_routes_lab_b_to_config_check(tmp_path):
    home = _make_course_home(tmp_path, 'COCO128_repartitioned_teaching_only', fresh=False)
    # generic artifact (QA json) exists, but the config-driven check must fail it
    assert any('stale' in p for p in runner.verify_artifacts(home, '02', time.time()))
    assert runner.verify_artifacts(home, '02', time.time() - 7200) == []
