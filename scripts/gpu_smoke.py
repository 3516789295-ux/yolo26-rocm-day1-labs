"""Physical AMD GPU admission smoke test for the YOLO26 ROCm Day 1 labs.

Proves, on real hardware:
  1. ROCm PyTorch trio exact versions + HIP backend
  2. GPU enumeration (name / arch / capability)
  3. GPU tensor allocation
  4. GPU matmul
  5. GPU Conv2d forward
  6. Loss computation
  7. loss.backward()
  8. Optimizer step (params actually change)
  9. torch.cuda.synchronize()
 10. NaN/Inf screening of results and gradients
 11. BatchNorm2d forward (MIOpen hipRTC regression probe, ROCm/rocm-libraries#13437)

Writes JSON evidence to validation/logs/gpu_smoke.json and prints a report.
Exit code 0 only if every gate passes.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio
import torchvision

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / 'validation' / 'logs' / 'gpu_smoke.json'

# MIOpen hipRTC on the Windows runtime-only stack needs the freestanding STL
# shims (see scripts/install_miopen_shims.py / ROCm/rocm-libraries#13437).
# Set the extension point for this process if not already configured.
_SHIM_DIR = PROJECT / 'scripts' / 'miopen_shims'
if sys.platform == 'win32' and _SHIM_DIR.is_dir():
    os.environ.setdefault('HIPRTC_COMPILE_OPTIONS_APPEND', f'-I{_SHIM_DIR}')

report: dict = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version.split()[0], 'platform': platform.platform()}

failures: list[str] = []


def gate(name: str, ok: bool, detail=None):
    report.setdefault('gates', {})[name] = {'pass': bool(ok), 'detail': detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        failures.append(name)


# --- 1. versions & backend -------------------------------------------------
# The course pins one wheel stack (constraints-rocm.txt) and the Windows run is
# validated against it exactly. On Linux Radeon Cloud the platform may ship a
# different preinstalled stack; docs/RADEON_CLOUD_LINUX_SETUP.md §5 states the
# platform stack wins there and the real versions must be recorded in
# environment.json. So: exact match on Windows, recorded-but-allowed on POSIX.
EXPECTED_STACK = {
    'torch': '2.12.0+rocm7.14.1',
    'torchvision': '0.27.0+rocm7.14.1',
    'torchaudio': '2.11.0+rocm7.14.1',
}
# YOLO_STRICT_STACK=1 forces the pinned stack everywhere (used by the Windows run).
STRICT_STACK = sys.platform == 'win32' or os.getenv('YOLO_STRICT_STACK') == '1'


def version_gate(pkg: str, actual: str) -> None:
    """Pinned-stack gate.

    Strict (Windows / YOLO_STRICT_STACK=1): the exact pinned build is required.
    Non-strict (Linux platform stack): a different build passes, but the real
    version is written into the report so the evidence chain stays auditable.
    """
    expected = EXPECTED_STACK[pkg]
    if actual == expected:
        gate(f'{pkg}_version', True, actual)
    elif STRICT_STACK:
        gate(f'{pkg}_version', False, f'{actual} (expected {expected})')
    else:
        report.setdefault('platform_stack', {})[pkg] = {'actual': actual, 'pinned': expected}
        gate(f'{pkg}_version', True, f'{actual} (platform stack; pinned is {expected})')


version_gate('torch', torch.__version__)
version_gate('torchvision', torchvision.__version__)
version_gate('torchaudio', torchaudio.__version__)
gate('hip_backend', torch.version.hip is not None, str(torch.version.hip))
# torch.version.cuda is None on HIP-only ROCm wheels — informational; the HIP
# gate above is the authoritative backend proof.
gate('cuda_api_note', True, f'torch.version.cuda={torch.version.cuda} '
                            '(None is normal on HIP-only ROCm wheels; '
                            'torch.cuda namespace is the HIP API surface)')
gate('gpu_available', torch.cuda.is_available())

# torch.cuda on a ROCm build is the HIP API surface — not NVIDIA CUDA.
report['explanation'] = ("AMD ROCm PyTorch exposes HIP through the torch.cuda API namespace; "
                         "torch.version.hip != None proves the ROCm build, torch.version.cuda "
                         "reports the HIP runtime version alias on ROCm wheels.")

# --- 2. device enumeration ---------------------------------------------------
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, 'gcnArchName', None) or getattr(props, 'arch', None)
    gate('device_name', '7900' in torch.cuda.get_device_name(0) or 'AMD' in torch.cuda.get_device_name(0),
         torch.cuda.get_device_name(0))
    gate('gfx_arch_gfx1100', arch is not None and 'gfx1100' in str(arch), str(arch))
    report['device'] = {'name': torch.cuda.get_device_name(0), 'gcnArchName': str(arch),
                        'capability': f'{props.major}.{props.minor}',
                        'total_memory_GiB': round(props.total_memory / 2**30, 2),
                        'multi_processor_count': props.multi_processor_count}
else:
    gate('device_name', False, 'no GPU visible')

# --- 3-9. real compute: forward + backward + optimizer ----------------------
try:
    torch.manual_seed(42)
    t_alloc0 = time.perf_counter()
    x = torch.randn(64, 128, device=dev)                     # 3. allocation
    w_ref = torch.randn(128, 128, device=dev)
    y = x @ w_ref                                             # 4. matmul
    torch.cuda.synchronize()
    alloc_matmul_s = time.perf_counter() - t_alloc0
    gate('gpu_tensor_alloc', x.device.type == 'cuda' and x.is_cuda, str(x.device))
    gate('gpu_matmul', y.shape == (64, 128) and y.device.type == 'cuda', f'{alloc_matmul_s*1000:.1f} ms')

    # 5. Conv2d forward  6. loss  7. backward  8. optimizer step
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(dev)
    bn = torch.nn.BatchNorm2d(16).to(dev)                     # MIOpen hipRTC probe (BN training path)
    target_w = torch.randn(16, 3, 3, 3, device=dev)
    opt = torch.optim.SGD(list(conv.parameters()) + list(bn.parameters()), lr=0.1, momentum=0.9)
    imgs = torch.randn(8, 3, 64, 64, device=dev)

    t_fw0 = time.perf_counter()
    out = conv(imgs)
    out = bn(out)
    loss = torch.nn.functional.mse_loss(out, torch.tanh(out).detach())   # 6. loss on GPU
    torch.cuda.synchronize()
    fw_s = time.perf_counter() - t_fw0
    gate('gpu_conv2d_forward', out.shape == (8, 16, 64, 64), f'{fw_s*1000:.1f} ms')
    gate('gpu_loss_finite', torch.isfinite(loss).item(), f'loss={loss.item():.6f}')

    p_before = conv.weight.detach().clone()
    t_bw0 = time.perf_counter()
    loss.backward()                                           # 7. backward on GPU
    torch.cuda.synchronize()
    bw_s = time.perf_counter() - t_bw0
    g = conv.weight.grad
    gate('gpu_backward_grad_exists', g is not None and g.device.type == 'cuda', f'{bw_s*1000:.1f} ms')

    opt.step()                                                # 8. optimizer step
    torch.cuda.synchronize()                                  # 9. explicit sync
    delta = (conv.weight.detach() - p_before).abs().max().item()
    gate('optimizer_step_changed_weights', delta > 0, f'max|Δw|={delta:.3e}')

    # 10. NaN/Inf screening
    finite = all(torch.isfinite(t).all().item() for t in (y, out, loss, g, conv.weight.detach()))
    gate('no_nan_inf', finite)

    # BatchNorm running stats updated in train mode (the exact path broken before PR 13437)
    gate('batchnorm_train_mode', bn.running_mean.abs().sum().item() >= 0 and bn.num_batches_tracked.item() >= 1,
         f'num_batches_tracked={int(bn.num_batches_tracked.item())}')

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    report['compute'] = {'alloc_matmul_ms': round(alloc_matmul_s * 1000, 2),
                         'conv_bn_loss_forward_ms': round(fw_s * 1000, 2),
                         'backward_ms': round(bw_s * 1000, 2),
                         'peak_allocated_GiB': round(peak_gib, 4)}
    print(f"\nCompute timings: {report['compute']}")
except Exception as exc:  # pragma: no cover - evidence capture
    import traceback
    report['compute_error'] = traceback.format_exc()
    gate('real_compute_pipeline', False, f'{type(exc).__name__}: {exc}')
else:
    gate('real_compute_pipeline', True)

# --- driver info (best effort, platform-aware) --------------------------------
def _driver_info() -> str:
    """Collect AMD driver information for the current platform.

    Windows: WMI via PowerShell. Linux: rocm-smi / sysfs, no PowerShell needed.
    """
    if sys.platform == 'win32':
        out_txt = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '(Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like "*AMD*"} | '
             'Select-Object Name,DriverVersion | ConvertTo-Json -Compress)'],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return out_txt
    # POSIX / AMD Radeon Cloud
    try:
        smi = subprocess.run(['rocm-smi', '--showdriverversion'],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        smi = ''
    try:
        kfd = Path('/sys/class/kfd/kfd/topology/nodes')
        if kfd.is_dir():
            smi += ('\n' if smi else '') + \
                f'kfd_nodes: {sorted(p.name for p in kfd.iterdir() if p.name.isdigit())}'
    except Exception:
        pass
    return smi or 'unavailable: rocm-smi not found and /sys/class/kfd unreadable'


try:
    report['driver'] = _driver_info()
except Exception as exc:
    report['driver'] = f'unavailable: {exc}'

report['overall'] = 'PASS' if not failures else 'FAIL'
report['failed_gates'] = failures
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"\nOVERALL: {report['overall']}" + (f' failed={failures}' if failures else ''))
print(f'evidence: {OUT}')
sys.exit(0 if not failures else 1)
