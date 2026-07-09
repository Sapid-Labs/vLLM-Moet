// Test: can the GB10 (sm_121) load the shipped sm_120 cubins?
#include <cuda.h>
#include <stdio.h>

#define CK(call) do { CUresult r = (call); if (r != CUDA_SUCCESS) { \
    const char *name = NULL, *desc = NULL; \
    cuGetErrorName(r, &name); cuGetErrorString(r, &desc); \
    printf("  FAIL %s -> %s: %s\n", #call, name ? name : "?", desc ? desc : "?"); \
    return 1; } } while (0)

int try_load(CUdevice dev, const char *path, const char *kname) {
    printf("%s:\n", path);
    CUmodule mod;
    CUresult r = cuModuleLoad(&mod, path);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL, *desc = NULL;
        cuGetErrorName(r, &name); cuGetErrorString(r, &desc);
        printf("  cuModuleLoad FAIL -> %s: %s\n", name ? name : "?", desc ? desc : "?");
        return 1;
    }
    CUfunction fn;
    r = cuModuleGetFunction(&fn, mod, kname);
    if (r != CUDA_SUCCESS) {
        const char *name = NULL;
        cuGetErrorName(r, &name);
        printf("  loaded, but cuModuleGetFunction(%s) FAIL -> %s\n", kname, name ? name : "?");
        // list nothing; still counts as module load success
        cuModuleUnload(mod);
        return 2;
    }
    int nregs = -1, smem = -1;
    cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
    cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
    printf("  OK: module + kernel '%s' loaded (regs=%d, static smem=%d)\n", kname, nregs, smem);
    cuModuleUnload(mod);
    return 0;
}

int main(void) {
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    char name[128]; CK(cuDeviceGetName(name, sizeof name, dev));
    int major, minor;
    CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    printf("device: %s sm_%d%d\n", name, major, minor);
    CUcontext ctx; CK(cuCtxCreate(&ctx, NULL, 0, dev));

    const char *base = "/home/joemuller/Dev/vLLM-Moet/kernels/cubins-sm120/";
    char p[512];
    int bad = 0;
    #define T(f, k) do { snprintf(p, sizeof p, "%s%s", base, f); bad += try_load(dev, p, k) ? 1 : 0; } while (0)
    T("moe_w2_mm_k4096.cubin", "moe_w2_mm");
    T("moe_w2_mm_k2048.cubin", "moe_w2_mm");
    T("moe_w2_mm_mc4afrag_k4096.cubin", "moe_w2_mm");
    T("moe_w4_mm_k4096.cubin", "moe_w4_mm");
    T("mla_prefill_state.cubin", "mla_prefill_state2");
    printf(bad ? "RESULT: %d cubin(s) failed to load\n" : "RESULT: all cubins load on this device\n", bad);
    return bad;
}
