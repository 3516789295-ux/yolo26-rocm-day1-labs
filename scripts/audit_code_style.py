"""Style metrics for the six notebooks' code cells (tokenize-based, robust).

Counts: semicolon-joined executable statements, >100-char lines, dense `a=b`
assignments without surrounding spaces, total lines. Used to demonstrate the
PEP 8 restyle delta; harmless to run anywhere.
"""
import io
import re
import tokenize
from pathlib import Path

import nbformat

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / 'notebooks').glob('*.ipynb'))


def code_only(src: str) -> str:
    """Return src with strings and comments blanked to spaces (same length)."""
    tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            tok = tok._replace(string=' ' * len(tok.string))
        tokens.append(tok)
    return tokenize.untokenize(tokens)


def main() -> None:
    semis = long_lines = dense = total = 0
    for p in NOTEBOOKS:
        nb = nbformat.read(p, as_version=4)
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            src = ''.join(cell.source)
            try:
                stripped = code_only(src)
            except (tokenize.TokenError, IndentationError):
                stripped = src
            for line in stripped.splitlines():
                if line.strip().count(';') >= 1:
                    semis += 1
            for line in src.splitlines():
                total += 1
                if len(line) > 100:
                    long_lines += 1
            dense += len(re.findall(r'\w=\w', stripped))
    print(f'notebooks: {len(NOTEBOOKS)}')
    print(f'code lines: {total}')
    print(f'semicolon-joined statements: {semis}')
    print(f'lines > 100 chars: {long_lines}')
    print(f'dense a=b assignments: {dense}')


if __name__ == '__main__':
    main()
