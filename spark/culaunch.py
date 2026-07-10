"""Minimal CUDA driver-API launch harness for the kernel check scripts.

Reimplementation of the `culaunch.Cuda` helper the upstream check scripts
import from the (non-public) cubit-dev workspace — just enough surface for
gen/moe_w2_check.py and gen/moe_w4_check.py: load a cubin, move numpy blobs
to/from the device, memset, launch, synchronize.

Written for the DGX Spark (GB10, sm_121) port; nothing here is Spark-specific.
"""
import ctypes

import numpy as np

_cuda = ctypes.CDLL("libcuda.so.1")


def _ck(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        _cuda.cuGetErrorName(rc, ctypes.byref(name))
        raise RuntimeError(f"{what} -> {name.value.decode() if name.value else rc}")


class Cuda:
    def __init__(self, device=0):
        _ck(_cuda.cuInit(0), "cuInit")
        self.dev = ctypes.c_int()
        _ck(_cuda.cuDeviceGet(ctypes.byref(self.dev), device), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        # primary context, NOT a private one: scripts that also use torch.cuda
        # (e.g. the pack helpers) must share a context or handles go invalid
        _ck(_cuda.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.dev),
            "cuDevicePrimaryCtxRetain")
        _ck(_cuda.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
        self._mods = []

    def load_kernel(self, cubin_path, kernel_name):
        mod = ctypes.c_void_p()
        _ck(_cuda.cuModuleLoad(ctypes.byref(mod), cubin_path.encode()), f"cuModuleLoad({cubin_path})")
        self._mods.append(mod)
        fn = ctypes.c_void_p()
        _ck(_cuda.cuModuleGetFunction(ctypes.byref(fn), mod, kernel_name.encode()),
            f"cuModuleGetFunction({kernel_name})")
        return fn

    def alloc(self, nbytes):
        ptr = ctypes.c_uint64()
        _ck(_cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cuMemAlloc")
        return ptr

    def to_device(self, arr):
        arr = np.ascontiguousarray(arr)
        ptr = self.alloc(arr.nbytes)
        _ck(_cuda.cuMemcpyHtoD_v2(ptr, arr.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_size_t(arr.nbytes)), "cuMemcpyHtoD")
        return ptr

    def from_device(self, ptr, nbytes, dtype=np.uint8):
        out = np.empty(nbytes // np.dtype(dtype).itemsize, dtype=dtype)
        _ck(_cuda.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), ptr,
                                  ctypes.c_size_t(nbytes)), "cuMemcpyDtoH")
        return out

    def memset32(self, ptr, value, count_words):
        _ck(_cuda.cuMemsetD32_v2(ptr, ctypes.c_uint32(value),
                                 ctypes.c_size_t(count_words)), "cuMemsetD32")

    def launch(self, fn, grid, block, args, shared=0):
        # Each arg is a ctypes scalar (c_uint64 device ptr / c_uint32 imm);
        # kernelParams is an array of void* pointing at each arg's storage.
        holders = []
        for a in args:
            if not isinstance(a, (ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int32,
                                  ctypes.c_float, ctypes.c_int64)):
                raise TypeError(f"unsupported arg type: {type(a)}")
            holders.append(a)
        params = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        _ck(_cuda.cuLaunchKernel(fn,
                                 grid[0], grid[1], grid[2],
                                 block[0], block[1], block[2],
                                 shared, None, params, None), "cuLaunchKernel")

    def synchronize(self):
        _ck(_cuda.cuCtxSynchronize(), "cuCtxSynchronize")
