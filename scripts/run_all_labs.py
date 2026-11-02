"""Automated end-to-end runner for the six YOLO26 ROCm Day 1 notebooks.

Usage (from project root, inside the .venv created by the validation setup):
    python scripts/run_all_labs.py [--course-home DIR] [--timeout SEC]
                                   [--only 00,03] [--keep-going] [--list]

- Executes notebooks 00..05 in order with the registered ROCm kernel
  (or falls back to a kernel spec built from the current interpreter).
- Records per-cell durations, stdout/stderr, and saves executed copies
  to validation/executed/<timestamp>/.
- Verifies required artifacts exist and are non-empty.
- Writes NOTEBOOK_EXECUTION_DATA.json (consumed to build the final matrix).

Isolation: pass --course-home to redirect all lab outputs (YOLO_COURSE_HOME)
away from the classroom directory. The default official validation run uses
<project>/yolo_course (the Windows default of course.lab_utils.root()).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT / 'notebooks'
VALIDATION = PROJECT / 'validation'
KERNEL_NAME = 'yolo26-rocm7141'

ORDER = [
    '00_ROCm_Environment_Preflight.ipynb',
    '01_Lab_A_Predict_and_Threshold.ipynb',
    '02_Lab_B_Dataset_Annotation_QA.ipynb',
    '03_Lab_C_ROCm_Training_Finetune.ipynb',
    '04_Lab_D_Evaluation_Error_Analysis.ipynb',
    '05_Lab_E_Controlled_AB_Optimization.ipynb',
]

# Required artifacts per notebook: (course-home-relative path, is_glob)
REQUIRED_ARTIFACTS = {
    '00': [('reports/environment.json', False), ('artifacts/yolo26n.pt', False)],
    '01': [('reports/lab_a_predictions.json', False), ('assets/lab_a_thresholds/*.jpg', True)],
    '02': [('data/processed/coco128_classroom_3class/data.yaml', False),
           ('reports/lab_b_dataset_qa.json', False)],
    '03': [('runs/lab_c/baseline/weights/best.pt', False),
           ('runs/lab_c/baseline/results.csv', False),
           ('reports/lab_c_baseline.json', False),
           ('reports/lab_c_baseline_val.json', False)],
    '04': [('reports/lab_d_official_metrics.json', False),
           ('reports/lab_d_fixed_threshold.json', False),
           ('reports/lab_d_errors.csv', False)],
    '05': [('runs/lab_e/imgsz416/weights/best.pt', False),
           ('reports/lab_e_variant.json', False),
           ('reports/lab_e_ab_comparison.csv', False),
           ('assets/lab_e_ab_mAP.png', False)],
}

# Cells whose code references GPU training/inference (for GPU-used evidence).
GPU_CELL_MARKERS = ('require_rocm', 'torch.cuda', 'device=device', "device='cuda", 'model.predict',
                    '.train(', '.val(')


def check_venv() -> Path:
    exe = Path(sys.executable).resolve()
    venv_python = (PROJECT / '.venv' / 'Scripts' / 'python.exe')
    if os.name != 'nt':
        venv_python = PROJECT / '.venv' / 'bin' / 'python'
    if exe != venv_python.resolve():
        print(f'ERROR: run inside the project venv. Expected {venv_python}, got {exe}', file=sys.stderr)
        sys.exit(2)
    return exe


def check_kernel(venv_exe: Path) -> tuple[str, bool]:
    """Return (kernel_name, registered). Prefer the registered yolo26-rocm7141 spec."""
    import jupyter_client
    from jupyter_client.kernelspec import KernelSpecManager
    mgr = KernelSpecManager()
    specs = mgr.find_kernel_specs()
    if KERNEL_NAME in specs:
        spec = mgr.get_kernel_spec(KERNEL_NAME)
        if spec.argv and Path(spec.argv[0]).resolve() == venv_exe:
            return KERNEL_NAME, True
        print(f'WARNING: kernel {KERNEL_NAME} argv mismatch: {spec.argv}')
    return 'python3', False


def run_notebook(nb_path: Path, kernel_name: str, course_home: Path,
                 timeout: int, log_dir: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    resources = {
        'metadata': {'path': str(PROJECT), 'kernel_name': kernel_name},
    }
    client = NotebookClient(nb, resources=resources, timeout=timeout,
                            kernel_name=kernel_name,
                            allow_errors=False,  # fail fast on first cell error
                            record_timing=True)
    started = time.perf_counter()
    info = {'notebook': nb_path.name, 'kernel': kernel_name, 'started': datetime.now(timezone.utc).isoformat()}
    try:
        client.execute()
        info['result'] = 'PASS'
    except Exception as exc:  # keep full traceback evidence
        info['result'] = 'FAIL'
        info['error_type'] = type(exc).__name__
        info['error'] = str(exc)[:4000]
        info['traceback'] = traceback.format_exc()[-8000:]
        # Attach the cell-level error info when available
        for attr in ('cell_index', 'cell_id'):
            if hasattr(exc, attr):
                info[attr] = getattr(exc, attr)
    info['duration_seconds'] = round(time.perf_counter() - started, 2)

    # Per-cell execution evidence from the (possibly partially) executed notebook
    cells = []
    gpu_used = False
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        src = cell.get('source', '')
        exec_meta = cell.get('metadata', {}).get('execution', {})
        durations = exec_meta.get('shell_execute_time')
        outputs = cell.get('outputs', [])
        out_text = []
        for o in outputs:
            if o.get('output_type') == 'stream':
                out_text.append(''.join(o.get('text', [])))
            elif 'data' in o and 'text/plain' in o['data']:
                out_text.append(''.join(o['data']['text/plain']))
        executed = exec_meta.get('iopub.status.busy') is not None or bool(outputs) or cell.get('execution_count') is not None
        if any(m in src for m in GPU_CELL_MARKERS) and executed:
            gpu_used = True
        cells.append({'index': idx, 'executed': executed,
                      'execution_count': cell.get('execution_count'),
                      'duration_seconds': durations if isinstance(durations, (int, float)) else None,
                      'output_tail': (''.join(out_text)[-1500:] if out_text else '')})
    info['code_cells'] = len(cells)
    info['executed_cells'] = sum(1 for c in cells if c['executed'])
    info['gpu_referenced_and_executed'] = gpu_used
    info['cells'] = cells

    out_nb = log_dir / nb_path.name.replace('.ipynb', '.executed.ipynb')
    import nbformat as nbf
    nbf.write(nb, out_nb)
    info['executed_copy'] = str(out_nb.relative_to(PROJECT))
    (log_dir / nb_path.name.replace('.ipynb', '.log.json')).write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return info


def verify_artifacts(course_home: Path, prefix: str) -> list[str]:
    problems = []
    for rel, is_glob in REQUIRED_ARTIFACTS.get(prefix, []):
        if is_glob:
            hits = sorted(course_home.glob(rel))
            if not hits:
                problems.append(f'missing glob {rel}')
            elif all(h.stat().st_size == 0 for h in hits):
                problems.append(f'all matches empty: {rel}')
        else:
            p = course_home / rel
            if not p.is_file():
                problems.append(f'missing {rel}')
            elif p.stat().st_size == 0:
                problems.append(f'empty {rel}')
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--course-home', default=None,
                    help='Override YOLO_COURSE_HOME (isolated output dir). Default: project yolo_course.')
    ap.add_argument('--timeout', type=int, default=7200, help='Per-cell timeout seconds (default 7200).')
    ap.add_argument('--only', default=None, help='Comma list of notebook prefixes, e.g. 00,03')
    ap.add_argument('--keep-going', action='store_true', help='Continue after failures.')
    ap.add_argument('--list', action='store_true', help='List discovered notebooks and exit.')
    args = ap.parse_args()

    venv_exe = check_venv()
    course_home = Path(args.course_home).resolve() if args.course_home else (PROJECT / 'yolo_course').resolve()
    course_home.mkdir(parents=True, exist_ok=True)

    # Discover and order-check
    found = sorted(p.name for p in NOTEBOOKS_DIR.glob('*.ipynb'))
    missing = [n for n in ORDER if n not in found]
    extra = [n for n in found if n not in ORDER]
    if missing:
        print(f'FATAL: expected notebooks missing: {missing}', file=sys.stderr)
        return 2
    if extra:
        print(f'NOTE: unexpected extra notebooks present: {extra}')
    selected = ORDER if not args.only else [n for n in ORDER if n[:2] in set(args.only.split(','))]

    if args.list:
        print(json.dumps({'order': ORDER, 'found': found, 'selected': selected}, indent=2))
        return 0

    kernel_name, registered = check_kernel(venv_exe)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = VALIDATION / 'executed' / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        'YOLO_COURSE_HOME': str(course_home),
        'MPLBACKEND': 'module://matplotlib_inline.backend_inline',
    }
    # MIOpen hipRTC freestanding shims (Windows only; see install_miopen_shims.py)
    shim_dir = PROJECT / 'scripts' / 'miopen_shims'
    if sys.platform == 'win32' and shim_dir.is_dir() and 'HIPRTC_COMPILE_OPTIONS_APPEND' not in os.environ:
        env['HIPRTC_COMPILE_OPTIONS_APPEND'] = f'-I{shim_dir}'
    os.environ.update(env)  # nbclient inherits our env into the kernel

    print(f'== YOLO26 ROCm Day1 runner ==\nvenv: {venv_exe}\nkernel: {kernel_name} (registered={registered})'
          f'\ncourse home: {course_home}\noutput: {out_dir}\n')
    results = []
    overall_fail = False
    for name in selected:
        prefix = name[:2]
        print(f'--- [{datetime.now().strftime("%H:%M:%S")}] running {name} ...', flush=True)
        info = run_notebook(NOTEBOOKS_DIR / name, kernel_name, course_home, args.timeout, out_dir)
        problems = verify_artifacts(course_home, prefix)
        if info['result'] == 'PASS' and problems:
            info['result'] = 'FAIL'
            info['artifact_problems'] = problems
        elif problems and args.keep_going:
            info['artifact_problems'] = problems
        if info['result'] != 'PASS':
            overall_fail = True
        print(f'--- {name}: {info["result"]} in {info["duration_seconds"]}s '
              f'({info["executed_cells"]}/{info["code_cells"]} cells)'
              + (f' artifacts: {info.get("artifact_problems")}' if info.get('artifact_problems') else ''), flush=True)
        results.append(info)
        if overall_fail and not args.keep_going:
            print('Stopping (use --keep-going to continue after failures).')
            break

    summary = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'kernel': kernel_name, 'kernel_registered': registered,
        'course_home': str(course_home), 'per_cell_timeout': args.timeout,
        'venv_python': str(venv_exe),
        'results': [{k: v for k, v in r.items() if k != 'cells'} for r in results],
    }
    data_path = VALIDATION / f'NOTEBOOK_EXECUTION_DATA_{stamp}.json'
    data_path.write_text(json.dumps({'summary': summary, 'notebooks': results},
                                    ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    passed = sum(1 for r in results if r['result'] == 'PASS')
    print(f'\n== {passed}/{len(results)} notebooks PASS ==\ndata: {data_path}')
    return 0 if (passed == len(ORDER) and not overall_fail) else 1


if __name__ == '__main__':
    raise SystemExit(main())
