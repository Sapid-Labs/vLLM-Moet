# vLLM-Moet on DGX Spark (GB10, sm_121) — port notes

Port of vLLM-Moet (SM120 2-bit MoE + FP4 delta) to a 2-node NVIDIA DGX Spark
cluster: aarch64, GB10 (**sm_121**), 121 GiB unified LPDDR5x per node (~273 GB/s),
nodes linked by 200G ConnectX-7 RoCE (~97 Gb/s measured).

## Feasibility findings (2026-07-09)

1. **The shipped SM120 cubins load and run correctly on sm_121, unmodified.**
   The driver accepts them (CUDA family compatibility — same SASS ISA family 12x;
   the cubins carry `.nv.compat` / CAPMERC sections from the `--mercury-stub`
   assembly). Verified with a driver-API load test (`cubin_load_test.c`), then
   op-validated with the upstream check scripts:
   - `moe_w2_mm` K ∈ {512, 1024, 2048, 4096, 6144}: PASS, worst rel 2.2–3.2e-3, deterministic
   - `moe_w4_mm` K ∈ {512, 1024, 2048, 4096, 6144}: PASS, worst rel 2.7–3.6e-3, deterministic

   (The upstream check scripts import a `culaunch` helper from the non-public
   cubit-dev workspace; `spark/culaunch.py` is a minimal reimplementation —
   run with `PYTHONPATH=$REPO/spark`.)

2. **The patch applies cleanly to a plain PyPI `pip install vllm==0.24.0`**
   (aarch64 wheels exist; torch 2.11/cu13/py3.12) — no Docker, no source build.
   All the patch's capability gates use CUDA *family* semantics
   (`is_device_capability_family(120)` → true on 12.1), so nothing rejects GB10.

3. **DeepGEMM nv-dev `a6b593d2` builds on aarch64** (wheel:
   `deep_gemm-2.5.0+a6b593d-cp312-cp312-linux_aarch64.whl`, ~7 min build).
   flashinfer-python 0.6.14 installs from PyPI; the `flashinfer-jit-cache`
   cu130 wheel has no aarch64 build — kernels JIT on first use instead.

4. **Unified memory simplifies, not complicates.** Full-resident 2-bit planes are
   plain device allocations = the same LPDDR pool. The BASE-cache tier
   (pinned-host base + GPU pool) makes no sense on Spark — it would double-allocate
   the same physical RAM. Likewise keep `VLLM_MOE_W2_DELTA_GB=0` (or small): the
   delta tier keeps a pinned host store that also double-dips.

## Node setup

```bash
# on the build node
python3.12 -m venv ~/venvs/vllm-moet
~/venvs/vllm-moet/bin/pip install vllm==0.24.0
cd ~/venvs/vllm-moet/lib/python3.12/site-packages
git apply /path/to/vLLM-Moet/patch/vllm-moet-v0.24.0.patch
~/venvs/vllm-moet/bin/pip uninstall -y flashinfer-cubin
~/venvs/vllm-moet/bin/pip install flashinfer-python==0.6.14
# DeepGEMM nv-dev (see docs/v024-port.md)
git clone https://github.com/deepseek-ai/DeepGEMM && cd DeepGEMM
git checkout a6b593d2826719dcf4892609af7b84ee23aaf32a
git submodule update --init --depth 1 third-party/cutlass third-party/fmt
~/venvs/vllm-moet/bin/python setup.py bdist_wheel
~/venvs/vllm-moet/bin/pip install --no-deps dist/deep_gemm-*.whl

# peer nodes are identical (same user/paths/OS) — just rsync:
rsync -a ~/venvs/vllm-moet peer:~/venvs/
rsync -a ~/Dev/vLLM-Moet peer:~/Dev/
```

Runtime env (all nodes): `VLLM_MOE_W2=1 VLLM_MOE_W2_CUBIT_DIR=~/Dev/vLLM-Moet/kernels/cubins-sm120`.

## Targets

- **DeepSeek-V4-Flash, 1 Spark** — 2-bit base 72.7 GiB + FP8 dense fits a single
  121 GiB node with room to spare (the README's "96 GB card" config, minus the
  squeeze). Bring-up model.
- **GLM-5.2 (753B), 2 Sparks, PP2** — from `zai-org/GLM-5.2-FP8` (dense already
  FP8, expert fp8→2bit requant golden-tested upstream). Per rank ≈ 95 GiB experts
  + ~12 GiB FP8 dense + overhead ≈ 111 GiB of ~114 usable. PP over the RoCE fabric
  (Ray), K=6144/K=2048 cubins, MTP-under-PP per the patch. The 121 GiB unified
  nodes hold what the upstream 2×96 GB config could only cache at 50% coverage.

## Status log

- 2026-07-09: cubins validated on GB10 (all 10 decode-path kernels PASS);
  venv + patch + DeepGEMM + flashinfer stack complete on spark-05cc; peer sync
  + model downloads in progress.
