"""Install the freestanding STL shims that unblock MIOpen hipRTC kernel
compilation on Windows runtime-only ROCm stacks (rocm-sdk-libraries 7.14.1).

Background: MIOpen shipped in rocm-sdk-libraries 7.14.1 unconditionally includes
host <type_traits>/<utility> in its hipRTC kernel headers, but hipRTC device
compiles run isolated (-nostdinc) with no host C++ standard library, so every
BatchNorm/Conv training kernel fails with HIPRTC_ERROR_COMPILATION ->
miopenStatusUnknownError. Upstream fix: ROCm/rocm-libraries#13437 (make kernel
headers self-contained); no rebuilt 7.14.1 wheel exists yet. This installer
applies the same idea locally WITHOUT touching any wheel:

1. Copies the shim headers into <venv>/Lib/site-packages/_rocm_sdk_core/include/
   (inert freestanding fallbacks, HIP-guarded).
2. Emits / optionally persists (setx --user) the hipRTC extension point
   HIPRTC_COMPILE_OPTIONS_APPEND=-I<project>/scripts/miopen_shims
   which is the mechanism that actually reaches hipRTC's option list.

Usage:
    python scripts/install_miopen_shims.py           # deploy copies + print env
    python scripts/install_miopen_shims.py --setx    # also persist user env var
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SHIM_SRC = PROJECT / 'scripts' / 'miopen_shims'
SHIM_FILES = ('type_traits', 'utility', 'initializer_list', 'cstdint')
ENV_NAME = 'HIPRTC_COMPILE_OPTIONS_APPEND'


def sdk_include_root() -> Path:
    venv = PROJECT / '.venv'
    candidates = [venv / 'Lib' / 'site-packages' / '_rocm_sdk_core' / 'include']
    for c in candidates:
        if (c / 'hip' / 'hip_runtime.h').is_file():
            return c
    raise SystemExit(f'rocm-sdk-core include root not found under {venv}; '
                     'install the ROCm wheels first (docs/ROCM_WINDOWS_SETUP.md)')


def main() -> int:
    if sys.platform != 'win32':
        print('This shim path is only needed on Windows runtime-only ROCm stacks; nothing to do.')
        return 0
    root = sdk_include_root()
    for name in SHIM_FILES:
        src = SHIM_SRC / name
        if not src.is_file():
            raise SystemExit(f'missing shim source: {src}')
        dst = root / name
        dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        print(f'deployed {dst}')
    value = f'-I{SHIM_SRC}'
    print(f'\nRequired environment for MIOpen JIT on this machine:\n  {ENV_NAME}={value}')
    if '--setx' in sys.argv:
        r = subprocess.run(['setx', ENV_NAME, value], capture_output=True, text=True)
        print(((r.stdout or '') + (r.stderr or '')).strip() or f'setx rc={r.returncode}')
        print('(user-scope; new processes only. Remove with: reg delete HKCU\\Environment /V '
              f'{ENV_NAME} /F)')
    elif ENV_NAME not in os.environ:
        print('NOTE: env var not set in the current process. Run with --setx (new shells) or '
              'export it before launching Jupyter/training.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
