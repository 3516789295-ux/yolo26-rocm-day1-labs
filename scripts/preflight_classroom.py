"""Pre-class preflight check for the YOLO26 ROCm Day 1 labs (instructor tool).

Run on the classroom machine BEFORE students arrive:

    python scripts/preflight_classroom.py [--course-home DIR] [--json OUT.json]

Checks (each PASS/FAIL with a concrete next step):
  1. ROCm PyTorch trio versions match the pinned course stack
  2. HIP backend + GPU visible (name / arch / VRAM)
  3. torchvision GPU NMS operator (the #1 platform pitfall)
  4. yolo26n.pt pre-staged with the official SHA256
  5. COCO128 archive or extracted dataset pre-staged
  6. COCO8 smoke dataset pre-staged
  7. Jupyter kernel registered and pointing at this interpreter
  8. Free disk space for a classroom of outputs
  9. (optional) tiny GPU matmul to confirm compute + show cold/warm state

Exit code 0 only if every gate passes. The script is read-only: it never
installs, downloads, or overwrites anything (assets must be pre-staged by the
instructor; see validation/linux_w7900/CLASSROOM_READINESS_CHECKLIST.md).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PINNED = {'torch': '2.12.0+rocm7.14.1',
          'torchvision': '0.27.0+rocm7.14.1',
          'torchaudio': '2.11.0+rocm7.14.1'}
OFFICIAL_YOLO26N_SHA256 = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'
OFFICIAL_COCO128_SHA256 = '61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e'
KERNEL_NAME = 'yolo26-rocm7141'

results: list[dict] = []


def check(name: str, ok: bool, detail: str = '', fix: str = ''):
    results.append({'check': name, 'status': 'PASS' if ok else 'FAIL',
                    'detail': detail, 'fix': fix})
    mark = 'PASS' if ok else 'FAIL'
    print(f'[{mark}] {name}' + (f' — {detail}' if detail else ''))
    if not ok and fix:
        print(f'       next: {fix}')


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--course-home', default=str(PROJECT / 'yolo_course'),
                    help='Classroom output root (default: project yolo_course)')
    ap.add_argument('--json', default=None, help='Also write results to this JSON file')
    args = ap.parse_args()
    home = Path(args.course_home).expanduser().resolve()

    # --- 1-3: interpreter / stack / GPU (needs torch in THIS interpreter) -----
    try:
        import torch
        import torchvision
    except Exception as exc:
        check('pytorch importable', False, str(exc)[:120],
              'Run inside the course venv: .venv/bin/python scripts/preflight_classroom.py')
        return finish(args)

    for pkg, mod in (('torch', torch), ('torchvision', torchvision)):
        want = PINNED[pkg]
        ok = mod.__version__ == want
        check(f'{pkg} version', ok,
              f'{mod.__version__}' + ('' if ok else f' (pinned: {want})'),
              '' if ok else 'Rebuild the uv-isolated stack per docs/RADEON_CLOUD_LINUX_SETUP.md §2B; '
                           'never pip-install torch into a working ROCm env')
    check('HIP backend', torch.version.hip is not None, str(torch.version.hip),
          'Wrong wheel flavor (CPU/CUDA) — rebuild the course environment')
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, 'gcnArchName', '')
        check('GPU visible', 'gfx1100' in str(arch) or 'AMD' in name or 'Radeon' in name,
              f'{name} · {arch} · {props.total_memory / 2**30:.1f} GiB',
              'Check GPU allocation / /dev/kfd / /dev/dri')
    else:
        check('GPU visible', False, 'torch.cuda.is_available() is False',
              'Check GPU allocation / /dev/kfd / /dev/dri')
    try:
        import torch as _t
        boxes = _t.tensor([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]],
                          device='cuda', dtype=_t.float32)
        keep = torchvision.ops.nms(boxes, _t.tensor([.9, .8, .7], device='cuda',
                                                    dtype=_t.float32), 0.5)
        check('torchvision GPU NMS', keep.tolist() == [0, 2], f'kept={keep.tolist()}',
              'Platform torchvision lacks the GPU NMS kernel — use the uv-isolated stack '
              '(setup guide §2B); this is the documented #1 pitfall')
    except Exception as exc:
        check('torchvision GPU NMS', False, f'{type(exc).__name__}: {str(exc)[:100]}',
              'Platform torchvision lacks the GPU NMS kernel — use the uv-isolated stack (§2B)')

    # --- 4: weights ------------------------------------------------------------
    weights = [Path(PROJECT) / 'yolo26n.pt', home / 'artifacts/yolo26n.pt']
    staged = next((p for p in weights if p.is_file()), None)
    if staged:
        digest = _sha256(staged)
        check('yolo26n.pt pre-staged', digest == OFFICIAL_YOLO26N_SHA256,
              f'{staged} (sha256 {"matches official" if digest == OFFICIAL_YOLO26N_SHA256 else digest[:16] + "… MISMATCH"})',
              'Re-stage from the official ultralytics/assets v8.4.0 release (verify sha256 '
              f'{OFFICIAL_YOLO26N_SHA256[:16]}…); restricted networks can use the mirrored '
              'asset with the same hash — see CLASSROOM_READINESS_CHECKLIST.md §assets')
    else:
        check('yolo26n.pt pre-staged', False,
              f'not found at {weights[0]} or {weights[1]}',
              'Stage the file (network-restricted procedure in the checklist) or ensure '
              'students can reach github release assets')

    # --- 5: COCO128 ------------------------------------------------------------
    zip_path = home / 'data/raw/coco128.zip'
    extracted = home / 'data/raw/coco128/images/train2017'
    if extracted.is_dir():
        check('COCO128 dataset', True, f'extracted at {extracted}')
    elif zip_path.is_file():
        digest = _sha256(zip_path)
        check('COCO128 dataset', digest == OFFICIAL_COCO128_SHA256,
              f'{zip_path} (sha256 {"matches official" if digest == OFFICIAL_COCO128_SHA256 else "MISMATCH"})',
              '' if digest == OFFICIAL_COCO128_SHA256 else
              'Zip hash mismatch — re-download from ultralytics/assets v0.0.0')
    else:
        check('COCO128 dataset', False, f'not found under {home}/data/raw/',
              'Stage coco128.zip (see checklist) or confirm classroom network reaches github')

    # --- 6: COCO8 ---------------------------------------------------------------
    try:
        from ultralytics.utils import SETTINGS
        coco8 = Path(SETTINGS['datasets_dir']) / 'coco8' / 'images'
        n = sum(1 for _ in coco8.rglob('*.jpg')) if coco8.is_dir() else 0
        check('COCO8 smoke dataset', n >= 8, f'{n} images under {coco8}',
              'Stage COCO8 (16 files, see checklist) — Lab C smoke needs it')
    except Exception as exc:
        check('COCO8 smoke dataset', False, f'ultralytics not importable: {exc}',
              'Install course deps inside the course venv')

    # --- 7: kernel ---------------------------------------------------------------
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        specs = KernelSpecManager().find_kernel_specs()
        if KERNEL_NAME in specs:
            spec = KernelSpecManager().get_kernel_spec(KERNEL_NAME)
            raw_argv0 = spec.argv[0] if spec.argv else ''
            argv0 = Path(raw_argv0).resolve() if raw_argv0 else None
            same = argv0 == Path(sys.executable).resolve()
            # Show the RAW kernelspec path (a venv python is often a symlink to a
            # base interpreter; the resolved target alone would confuse readers).
            check('course kernel registered', same,
                  f'{KERNEL_NAME} -> {raw_argv0}' + ('' if same else
                  f' (this interpreter: {sys.executable})'),
                  '' if same else f'Re-register: {sys.executable} -m ipykernel install --user '
                                 f'--name {KERNEL_NAME} --display-name "YOLO26 - AMD ROCm 7.14.1"')
        else:
            check('course kernel registered', False, f'{KERNEL_NAME} not in kernel list',
                  f'Register: {sys.executable} -m ipykernel install --user --name {KERNEL_NAME} '
                  '--display-name "YOLO26 - AMD ROCm 7.14.1"')
    except Exception as exc:
        check('course kernel registered', False, f'jupyter_client unavailable: {exc}',
              'pip install jupyter_client inside the course venv')

    # --- 8: disk ------------------------------------------------------------------
    free_gb = shutil.disk_usage(home if home.exists() else PROJECT).free / 2**30
    check('free disk space', free_gb >= 10, f'{free_gb:.1f} GiB free',
          'Need >=10 GiB for a classroom of outputs (28 students × ~50 MB + caches)')

    # --- 9: compute probe (informational) -----------------------------------------
    try:
        t0 = time.perf_counter()
        x = torch.randn(256, 256, device='cuda')
        (x @ x).sum().item()
        warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(20):
            (x @ x).sum().item()
        hot = (time.perf_counter() - t0) / 20
        check('GPU compute probe', True,
              f'first matmul {warm * 1000:.0f} ms, warm {hot * 1000:.2f} ms '
              f'({"cold" if warm > 0.5 else "warm"} MIOpen state — cold adds ~1 min to the '
              f'first training run; run one warm-up training before class)')
    except Exception as exc:
        check('GPU compute probe', False, f'{type(exc).__name__}: {exc}', 'Investigate GPU state')

    return finish(args)


def finish(args) -> int:
    failed = [r for r in results if r['status'] == 'FAIL']
    print(f"\nPreflight: {len(results) - len(failed)}/{len(results)} checks passed"
          + ('' if not failed else f" — FAILED: {[r['check'] for r in failed]}"))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                                   'python': sys.version.split()[0],
                                   'platform': platform.platform(),
                                   'results': results,
                                   'overall': 'PASS' if not failed else 'FAIL'},
                                  ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'evidence: {out}')
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
