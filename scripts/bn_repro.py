"""Minimal BatchNorm2d repro for the MIOpen hipRTC include issue (PR 13437 pattern).

Run with:
  MIOPEN_DEBUG_COMGR_LOG_OPTIONS=1 (to dump compile options)
Expected failure (unpatched): hipRTC 'type_traits' file not found -> miopenStatusUnknownError
"""
import sys
import time

import torch
import torch.nn as nn

assert torch.version.hip, 'need ROCm build'
dev = 'cuda'
print('device:', torch.cuda.get_device_name(0), flush=True)
m = nn.BatchNorm2d(100).to(dev)
x = torch.randn(20, 100, 35, 45, device=dev)
t0 = time.perf_counter()
y = m(x)
torch.cuda.synchronize()
print('BatchNorm2d forward OK in %.1fs' % (time.perf_counter() - t0), 'shape:', tuple(y.shape))
y.mean().backward()
torch.cuda.synchronize()
print('BatchNorm2d backward OK')
sys.exit(0)
