"""Bilingual header metadata consistency for the six teaching notebooks.

Regression: v1.2 shipped five notebooks whose English "Suggested time" and
"Learning path" values were transposed in the generator call sites (the Chinese
learning path displayed the English duration, and vice versa). This test parses
both language headers so the swap can never silently return.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((PROJECT / 'notebooks').glob('*.ipynb'))

ZH_HEADER = re.compile(r'\*\*建议时长：(?P<time>[^*]+)\*\*　｜　学习路径：(?P<path>[^\n]+)')
EN_HEADER = re.compile(r'\*\*Suggested time: (?P<time>[^*]+)\*\* \| Learning path: (?P<path>[^\n]+)')
TIME_LIKE = re.compile(r'分钟|minutes|min\b', re.IGNORECASE)


@pytest.mark.parametrize('nb_path', NOTEBOOKS, ids=lambda p: p.name)
def test_bilingual_header_time_and_path_not_swapped(nb_path):
    pytest.importorskip('nbformat')
    import nbformat
    nb = nbformat.read(nb_path, as_version=4)
    # The header is the markdown cell right after the style banner (cell index 1).
    header = ''.join(nb.cells[1].source)
    zh = ZH_HEADER.search(header)
    en = EN_HEADER.search(header)
    assert zh and en, f'{nb_path.name}: bilingual header line not found in cell 1'
    for lang, match in (('zh', zh), ('en', en)):
        time_val, path_val = match['time'].strip(), match['path'].strip()
        assert TIME_LIKE.search(time_val), (
            f'{nb_path.name} ({lang}): suggested-time field does not look like a duration: {time_val!r}')
        assert not TIME_LIKE.search(path_val), (
            f'{nb_path.name} ({lang}): learning-path field contains a time unit (fields swapped?): {path_val!r}')
        assert time_val != path_val
    # The two languages must agree on the learning-path structure (arrow chain).
    assert zh['path'].strip().count('→') == en['path'].strip().count('→'), (
        f'{nb_path.name}: zh/en learning paths have different step counts')
