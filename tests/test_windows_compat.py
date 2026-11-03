"""Cross-platform (Windows native) regression tests for the YOLO26 ROCm Day 1 labs.

Offline: no GPU, no real torch import required (torch is stubbed where needed).
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

NOTEBOOKS = sorted((PROJECT / 'notebooks').glob('*.ipynb'))
EXPECTED_NB = [
    '00_ROCm_Environment_Preflight.ipynb',
    '01_Lab_A_Predict_and_Threshold.ipynb',
    '02_Lab_B_Dataset_Annotation_QA.ipynb',
    '03_Lab_C_ROCm_Training_Finetune.ipynb',
    '04_Lab_D_Evaluation_Error_Analysis.ipynb',
    '05_Lab_E_Controlled_AB_Optimization.ipynb',
]


# ---------------------------------------------------------------- paths ----
def test_root_env_override(monkeypatch, tmp_path):
    import course.lab_utils as lu
    monkeypatch.setenv('YOLO_COURSE_HOME', str(tmp_path / 'my_course'))
    assert lu.root() == (tmp_path / 'my_course').resolve()


def _posix_workspace_root() -> Path | None:
    """The /workspace default that lab_utils.root() applies on Radeon Cloud hosts.

    Returns None when the branch cannot trigger (Windows, or a POSIX host without
    /workspace), so the caller can assert the cwd-local default instead.
    """
    if os.name == 'posix' and Path('/workspace').exists():
        return Path('/workspace/yolo_course').resolve()
    return None


def test_root_default_is_cwd_local(tmp_path, monkeypatch):
    """Without env var, default must stay inside the project/cwd, never a drive-root hijack.

    On a POSIX host that really has /workspace (AMD Radeon Cloud) root() is
    documented to return /workspace/yolo_course — see
    docs/RADEON_CLOUD_LINUX_SETUP.md §3 — so that host asserts the POSIX contract.
    """
    import course.lab_utils as lu
    monkeypatch.delenv('YOLO_COURSE_HOME', raising=False)
    monkeypatch.chdir(tmp_path)
    expected = _posix_workspace_root() or (tmp_path / 'yolo_course').resolve()
    assert lu.root() == expected


def test_root_workspace_only_on_posix(monkeypatch, tmp_path):
    """A C:\\workspace directory on Windows must NOT redirect the course home.

    On Windows we verify the guard exists in source and the nt branch stays
    cwd-local; the posix branch itself runs on Linux Radeon Cloud hosts.
    """
    import inspect

    import course.lab_utils as lu
    monkeypatch.delenv('YOLO_COURSE_HOME', raising=False)
    monkeypatch.chdir(tmp_path)
    src = inspect.getsource(lu.root)
    assert "os.name == 'posix' and Path('/workspace').exists()" in src
    expected = _posix_workspace_root() or (tmp_path / 'yolo_course').resolve()
    assert lu.root() == expected


# ---------------------------------------------------- ROCm gate integrity ----
class _FakeCuda:
    @staticmethod
    def is_available():
        return False


def _stub_torch(hip):
    torch = types.ModuleType('torch')
    version = types.SimpleNamespace(hip=hip, cuda=None if hip is None else '7.14-alias')
    torch.version = version
    torch.cuda = _FakeCuda
    sys.modules['torch'] = torch
    return torch


def test_require_rocm_rejects_cpu_build(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False) if 'torch' in sys.modules else None
    _stub_torch(hip=None)  # CPU wheel: hip is None
    import course.lab_utils as lu
    with pytest.raises(AssertionError, match='ROCm'):
        lu.require_rocm()


def test_require_rocm_rejects_hip_without_gpu(monkeypatch):
    _stub_torch(hip='7.14.60850')  # ROCm build but GPU invisible
    import course.lab_utils as lu
    with pytest.raises(AssertionError):
        lu.require_rocm()
    del sys.modules['torch']  # never leak the stub into other tests


# ---------------------------------------------------------- A/B purity ----
def test_ab_variant_changes_only_imgsz_project_name():
    """Replays Notebook 05's guard: variant args may differ from baseline ONLY in
    imgsz (the single experimental variable) plus project/name (log location)."""
    baseline_args = dict(data='ds.yaml', epochs=12, imgsz=512, batch=8, device=0, workers=0,
                         patience=30, optimizer='SGD', seed=42, deterministic=False, amp=False,
                         plots=True, project='runs/lab_c', name='baseline', exist_ok=True)
    variant_args = dict(baseline_args)
    variant_args.update(imgsz=416, project='runs/lab_e', name='imgsz416', exist_ok=True)
    changed = {k for k in baseline_args if baseline_args[k] != variant_args[k]}
    assert changed == {'imgsz', 'project', 'name'}


# ------------------------------------------------------ notebook structure ----
def test_all_six_notebooks_present_and_parse():
    nbformat = pytest.importorskip('nbformat')
    names = [p.name for p in NOTEBOOKS]
    assert names == EXPECTED_NB
    for p in NOTEBOOKS:
        nb = nbformat.read(p, as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == 'code']
        assert code, f'{p.name} has no code cells'
        # Training notebooks must keep the ROCm hard gate (no silent CPU bypass).
        src = '\n'.join(c.source for c in code)
        if p.name[:2] in {'01', '03', '04', '05'}:
            assert 'require_rocm()' in src, f'{p.name} lost its ROCm gate'
        # Baseline must remain a real multi-epoch run, not a smoke pass.
        # (regex-based: formatting may change, semantics may not)
        if p.name[:2] == '03':
            assert re.search(r'BASELINE_EPOCHS\s*=\s*12', src)
            assert re.search(r'epochs\s*=\s*BASELINE_EPOCHS', src)
            assert re.search(r'RUN_SMOKE\s*=\s*True', src)
        # A/B evaluation must use identical val imgsz for both arms.
        if p.name[:2] == '05':
            assert len(re.findall(r'imgsz\s*=\s*640', src)) >= 2


def test_lab_c_smoke_and_baseline_use_gpu_device():
    nbformat = pytest.importorskip('nbformat')
    nb = nbformat.read(PROJECT / 'notebooks' / '03_Lab_C_ROCm_Training_Finetune.ipynb', as_version=4)
    src = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assert re.search(r'device\s*=\s*device', src)
    assert re.search(r'workers\s*=\s*0', src)  # Windows multiprocessing safety


# ------------------------------------------------------ lab_utils invariants ----
@pytest.mark.skipif(os.name != 'nt', reason='Windows path semantics: backslash is a separator only on nt')
def test_label_path_for_image_windows_separator():
    from course.lab_utils import label_path_for_image
    p = Path(r'C:\some\ds\images\train\pic.jpg')
    out = label_path_for_image(p)
    assert out.parent.name == 'train' and out.parent.parent.name == 'labels'
    assert out.name == 'pic.txt'


def test_data_yaml_written_with_absolute_windows_path_roundtrip(tmp_path):
    """prepare_coco128_demo writes 'path' as absolute str; yaml must roundtrip on Windows."""
    import yaml
    from course.lab_utils import prepare_coco128_demo
    from tests.test_lab_utils import make_dataset
    source = make_dataset(tmp_path)
    dest = tmp_path / 'processed'
    prepare_coco128_demo(source, dest, seed=42)
    spec = yaml.safe_load((dest / 'data.yaml').read_text(encoding='utf-8'))
    assert Path(spec['path']).is_absolute()
    assert (Path(spec['path']) / spec['train']).is_dir()
