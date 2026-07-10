#!/usr/bin/env python3
"""Measure moe_w2_mm weight-read bandwidth on GB10 by memory type.

Launches the real 2-bit GEMM kernel in a timing loop with the SAME weight
bytes living in: (a) cudaMalloc device memory, (b) pageable anon host memory
(ATS), (c) mmap'd file page cache (ATS, 4KB pages), (d) mmap'd file after
MADV_HUGEPAGE (if the kernel collapses read-only file THP).

Effective GB/s = bytes-of-planes-touched x iters / time.
Run: PYTHONPATH=spark CUBIN=kernels/cubins-sm120/moe_w2_mm_k4096.cubin python3 ...
"""
import ctypes
import mmap as mmap_mod
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from culaunch import Cuda  # noqa: E402

from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (  # noqa: E402
    mxfp4_to_codes, pack_fragment_major, pack_scales)

CUBIN = os.environ.get("CUBIN", "kernels/cubins-sm120/moe_w2_mm_k4096.cubin")
N, K, E, M, NWARP = 4096, 4096, 64, 4, 8   # 64 experts ~ 537 MB of planes
ITERS = 30

torch.manual_seed(7)
cu = Cuda()
fn = cu.load_kernel(CUBIN, "moe_w2_mm")

# build one set of plane bytes on GPU, replicate E times host-side
w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
plane1 = pack_fragment_major(mxfp4_to_codes(w)).cpu().numpy()          # N*K/4
sexp = torch.randint(120, 132, (N, K // 32), dtype=torch.uint8)
sc1 = pack_scales(sexp.cuda()).cpu().numpy()                           # N*K/32
planes = np.tile(plane1, (E, 1))
scales = np.tile(sc1, (E, 1))
plane_bytes = planes.nbytes + scales.nbytes

# activations (shared, on device)
a = torch.randn(M, K) * 0.5
ab = a.view(M, K // 128, 128)
a_s = (ab.abs().amax(-1).clamp_min(1e-10) / 448.0)
a8 = (ab / a_s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(M, K)
d_a = cu.to_device(a8.view(torch.uint8).numpy())
d_as = cu.to_device(a_s.float().numpy().view(np.uint8))
d_c = cu.alloc(M * N * 2 * E)

libc = ctypes.CDLL("libc.so.6", use_errno=True)


def bench(name, b_ptr, s_ptr):
    descs = np.zeros((E, 6), dtype=np.uint64)
    pb = planes.shape[1]
    sb = scales.shape[1]
    for e in range(E):
        descs[e] = [d_a.value, d_as.value, b_ptr + e * pb, s_ptr + e * sb,
                    d_c.value + e * M * N * 2, np.uint64(M)]
    d_desc = cu.to_device(descs.view(np.uint8))
    args = [d_desc, ctypes.c_uint32(K), ctypes.c_uint32(K // 64),
            ctypes.c_uint32(N * 2), ctypes.c_uint32(K // 128)]
    # warmup
    for _ in range(3):
        cu.launch(fn, (N // 16, E, 1), (NWARP * 32, 1, 1), args)
    cu.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        cu.launch(fn, (N // 16, E, 1), (NWARP * 32, 1, 1), args)
    cu.synchronize()
    dt = time.time() - t0
    gbps = plane_bytes * ITERS / dt / 1e9
    print(f"{name:34s} {dt/ITERS*1000:7.2f} ms/iter   {gbps:7.1f} GB/s")


# (a) device
d_b = cu.to_device(planes.reshape(-1))
d_s = cu.to_device(scales.reshape(-1))
bench("cudaMalloc device", d_b.value, d_s.value)

# (b) pageable anon host
bench("pageable anon host (ATS)", planes.ctypes.data, scales.ctypes.data)

# (c) mmap'd file page cache
tmp = tempfile.NamedTemporaryFile(dir=os.path.expanduser("~/models"), delete=False)
tmp.write(planes.tobytes()); tmp.write(scales.tobytes()); tmp.flush()
f = open(tmp.name, "rb")
mm = mmap_mod.mmap(f.fileno(), 0, prot=mmap_mod.PROT_READ)
base = int(np.frombuffer(mm, dtype=np.uint8).ctypes.data)
# fault everything in
mm.madvise(mmap_mod.MADV_WILLNEED)
_ = sum(mm[i] for i in range(0, len(mm), 4096 * 256))
bench("mmap file page cache (ATS 4K)", base, base + planes.nbytes)

# (d) + MADV_HUGEPAGE (read-only file THP if kernel supports)
try:
    mm.madvise(14)  # MADV_HUGEPAGE
    time.sleep(8)   # give khugepaged a moment
    bench("mmap file + MADV_HUGEPAGE", base, base + planes.nbytes)
except Exception as e:
    print("MADV_HUGEPAGE:", e)

fh = [l for l in open("/proc/meminfo") if "FileHugePages" in l or "AnonHugePages" in l]
print("".join(fh).strip())
os.unlink(tmp.name)
