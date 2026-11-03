"""Generator consistency: the committed notebooks must be exactly what
build_notebooks.py produces (single source of truth).

Regressions caught here:
- editing a committed notebook by hand without updating the generator;
- editing the generator without regenerating the notebooks.

The AST reference test guards semantic drift of code cells against a committed
snapshot; this test guards the *generator <-> artifact* link (code AND markdown
AND notebook metadata), so the two together close the loop in both directions.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
GENERATOR = PROJECT / 'build_notebooks.py'
NOTEBOOKS_DIR = PROJECT / 'notebooks'


def _cell_sources(path: Path) -> list[tuple[str, str]]:
    nb = json.loads(path.read_text(encoding='utf-8'))
    out = []
    for cell in nb['cells']:
        src = cell['source']
        if isinstance(src, list):
            src = ''.join(src)
        out.append((cell['cell_type'], src))
    return out


def test_generator_reproduces_committed_notebooks(tmp_path):
    pytest.importorskip('nbformat')
    isolated = tmp_path / 'gen'
    isolated.mkdir()
    (isolated / 'build_notebooks.py').write_bytes(GENERATOR.read_bytes())
    proc = subprocess.run([sys.executable, str(isolated / 'build_notebooks.py')],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]

    rebuilt = sorted((isolated / 'notebooks').glob('*.ipynb'))
    committed = sorted(NOTEBOOKS_DIR.glob('*.ipynb'))
    assert [p.name for p in rebuilt] == [p.name for p in committed], \
        'generator produced a different notebook set than committed'

    for new, old in zip(rebuilt, committed):
        assert _cell_sources(new) == _cell_sources(old), (
            f'{old.name}: committed notebook differs from build_notebooks.py output. '
            'Regenerate with `python build_notebooks.py` and review the diff, or revert '
            'the hand edit and change the generator instead.')
        meta_new = json.loads(new.read_text(encoding='utf-8'))['metadata']
        meta_old = json.loads(old.read_text(encoding='utf-8'))['metadata']
        assert meta_new == meta_old, f'{old.name}: notebook metadata differs from generator output'
