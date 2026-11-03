"""Guard the CI workflow itself.

Regression for the release-closure finding "no CI": the offline suite (39
tests) only protects the gates if it actually runs automatically. This test
fails if the workflow is deleted, stops running pytest, or ever tries to
install torch on the CI runners (the course rule: never let any resolver
touch PyTorch outside the AMD ROCm wheel index).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
WORKFLOW = PROJECT / '.github' / 'workflows' / 'ci.yml'


def test_ci_workflow_exists_and_runs_offline_suite():
    yaml = pytest.importorskip('yaml')
    assert WORKFLOW.is_file(), (
        '.github/workflows/ci.yml missing — the offline regression suite '
        '(AST parity, leakage, A/B purity, runner gates) must run on every PR')
    doc = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    jobs = doc.get('jobs', {})
    assert jobs, 'ci.yml must define at least one job'
    # The offline pytest suite must be invoked somewhere in the workflow.
    blob = json.dumps(doc)
    assert 'pytest' in blob, 'ci.yml must run python -m pytest'
    # Platform coverage: the course supports Linux Radeon Cloud and Windows native.
    # Accept either a literal runs-on or a strategy matrix listing the OSes.
    platforms = []
    for job in jobs.values():
        platforms.append(str(job.get('runs-on', '')))
        platforms.extend(str(os) for os in job.get('strategy', {}).get('matrix', {}).get('os', []))
    blob_platforms = ' '.join(platforms)
    assert 'ubuntu' in blob_platforms, 'ci.yml must include a Linux runner'
    assert 'windows' in blob_platforms, 'ci.yml must include a Windows runner (course supports Windows native ROCm)'


def test_ci_never_installs_torch():
    """CI runners must stay torch-less: no step may install torch/torchvision.

    The course pins PyTorch to the AMD ROCm wheel index; a CI-side torch
    install would both slow the suite and blur the "tests run without
    torch" contract that makes this CI safe everywhere.
    """
    text = WORKFLOW.read_text(encoding='utf-8')
    for line in text.splitlines():
        if 'pip install' in line:
            for banned in ('torch', 'torchvision', 'torchaudio'):
                assert banned not in line, (
                    f'CI installs {banned} — the offline suite must stay torch-less; '
                    f'offending line: {line.strip()}')
