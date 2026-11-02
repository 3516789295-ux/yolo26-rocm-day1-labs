"""Semantic-parity guard: the six notebooks' code cells must match the committed
canonical-AST reference (validation/notebook_code_asts.json).

Why: notebooks are regenerated from build_notebooks.py (single source of truth).
Style edits are fine; semantic drift (renamed cross-cell variables, changed
calls, altered literals) is not — this test makes any such change an explicit,
reviewed reference update instead of a silent break.

The canonical form merges consecutive plain `import` statements (so PEP 8
import splitting stays legal) and compares everything else strictly via
ast.dump. Regenerate the reference after an INTENTIONAL semantic change:

    python tests/test_notebook_ast_reference.py --regenerate
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((PROJECT / 'notebooks').glob('*.ipynb'))
REFERENCE = PROJECT / 'validation' / 'notebook_code_asts.json'


def canonical(source: str) -> str:
    tree = ast.parse(source)
    out, pending = [], []

    def flush():
        if pending:
            out.append(ast.Import(names=pending))
            pending.clear()

    for node in tree.body:
        if isinstance(node, ast.Import):
            pending.extend(node.names)
        else:
            flush()
            out.append(node)
    flush()
    tree.body = out
    return ast.dump(tree, include_attributes=False)


def build_reference() -> dict:
    nbformat = pytest.importorskip('nbformat')
    ref = {}
    for p in NOTEBOOKS:
        nb = nbformat.read(p, as_version=4)
        ref[p.name] = [canonical(''.join(c.source))
                       for c in nb.cells if c.cell_type == 'code']
    return ref


def test_reference_file_exists_and_current():
    if '--regenerate' in sys.argv:  # manual regeneration path
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(build_reference(), indent=1), encoding='utf-8')
        pytest.skip(f'reference regenerated: {REFERENCE}')
    assert REFERENCE.is_file(), (
        'validation/notebook_code_asts.json missing — regenerate with '
        '`python tests/test_notebook_ast_reference.py --regenerate` and commit it '
        'together with the notebook change')


def test_notebook_code_matches_ast_reference():
    nbformat = pytest.importorskip('nbformat')
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    for p in NOTEBOOKS:
        nb = nbformat.read(p, as_version=4)
        current = [canonical(''.join(c.source))
                   for c in nb.cells if c.cell_type == 'code']
        expected = reference.get(p.name)
        assert expected is not None, f'{p.name} missing from AST reference'
        assert len(current) == len(expected), (
            f'{p.name}: code cell count changed ({len(expected)} -> {len(current)}); '
            'if intentional, regenerate the reference and explain in the PR')
        for idx, (now, ref) in enumerate(zip(current, expected)):
            ok = (hashlib.sha256(now.encode('utf-8')).hexdigest() == ref['sha256']
                  if isinstance(ref, dict) else now == ref)
            assert ok, (
                f'{p.name} cell #{idx}: code semantics changed (AST mismatch). '
                'Style-only edits never trip this. If the change is intentional, '
                'regenerate the reference via '
                '`python tests/test_notebook_ast_reference.py --regenerate` and '
                'justify the semantic change in the PR')


if __name__ == '__main__':
    test_reference_file_exists_and_current()
    test_notebook_code_matches_ast_reference()
    print('AST reference check: OK')
