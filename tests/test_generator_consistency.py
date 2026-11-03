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


def test_generated_code_cells_compile():
    """Every generated code cell must be valid Python.

    Regression: a stray leading space on the last line of a c() string in the
    generator produced IndentationError cells that only surfaced deep in
    execution. Compile every cell here so the failure is immediate and precise.
    """
    import ast
    for old in sorted(NOTEBOOKS_DIR.glob('*.ipynb')):
        for idx, (cell_type, src) in enumerate(_cell_sources(old)):
            if cell_type != 'code':
                continue
            try:
                ast.parse(src)
            except SyntaxError as exc:
                pytest.fail(f'{old.name} code cell #{idx} does not compile: {exc}')


# Per-notebook student-action markers (mission: every notebook keeps at least
# one genuine hands-on cell where the student must write/predict before running).
STUDENT_ACTION_MARKERS = {
    '00': ['MY_PREDICTION'],
    '01': ['MY_PREDICTED_IOU'],
    '02': ['APPLY_CORRECTION', 'NEW_LABEL_LINE'],
    '03': ['MY_READING'],
    '04': ['MY_ANALYSIS'],
    '05': ['MY_HYPOTHESIS'],
}

# Notebooks whose engagement is a safety-first opt-in write (Lab B's annotation
# correction: off by default, audited when enabled) rather than a mandatory
# predict-before-run gate.
NON_STRICT_NOTEBOOKS = {'02'}


def test_every_notebook_keeps_a_student_action_cell():
    for old in sorted(NOTEBOOKS_DIR.glob('*.ipynb')):
        prefix = old.name[:2]
        markers = STUDENT_ACTION_MARKERS.get(prefix)
        assert markers, f'{old.name}: no student-action marker registered for this notebook'
        sources = [src for ctype, src in _cell_sources(old) if ctype == 'code']
        hit = [m for m in markers if any(m in src for src in sources)]
        assert hit, (f'{old.name}: no student hands-on cell found (expected one of {markers}); '
                     'every notebook must keep at least one cell where the student '
                     'predicts/writes before running')
        if prefix in NON_STRICT_NOTEBOOKS:
            # Opt-in engagement: the write only happens after an explicit, audited
            # toggle (reason + backup); Run-All must stay safe by default.
            practice = [src for src in sources if 'APPLY_CORRECTION' in src]
            assert any('APPLY_CORRECTION = False' in src for src in practice), (
                f'{old.name}: the annotation-correction cell must stay opt-in '
                '(APPLY_CORRECTION = False by default)')
            assert any("startswith('TODO:')" in src for src in practice), (
                f'{old.name}: the correction cell must refuse TODO placeholder reasons '
                'when enabled')
            continue
        # Strict engagement: the practice cell asserts that the student replaced
        # the TODO placeholder, so Run-All cannot silently skip the exercise.
        practice = [src for src in sources if any(m in src for m in hit)]
        assert any("startswith('TODO')" in src for src in practice), (
            f'{old.name}: practice cell must assert that the student replaced the TODO '
            'placeholder (Run-All must not silently skip the exercise)')
