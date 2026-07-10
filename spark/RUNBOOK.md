# Running vLLM-Moet on DGX Spark — dual-node runbook

Run DeepSeek-V4-Flash (159B) on **one** NVIDIA DGX Spark, and GLM-5.2 (753B)
across **two**, using vLLM-Moet's 2-bit expert kernels. Everything here is
measured on real hardware (GB10, sm_121, 121 GiB unified LPDDR5x per node,
CUDA 13, driver 580.159.03).

**Status (2026-07-10):**
- DeepSeek-V4-Flash, 1 Spark: **working** — coherent greedy output,
  ~21 tok/s single-stream (eager + MTP k=2; ≈ the 273 GB/s bandwidth ceiling)
- GLM-5.2, 2 Sparks (PP2): **working** — correct greedy reasoning/arithmetic,
  ~4 tok/s single-stream warm (eager, no MTP, no CUDA graphs — both are
  untapped; upstream saw 4.7x from graphs on GLM). First requests after a
  cold start run slower while the kernel faults 190 GiB of planes in from
  NVMe.

Background reading: `spark/README.md` (port notes) and the "unified-memory
load war" section of the How To Spark lab notes — six GB10-specific memory
behaviors this stack works around. You don't need to understand them to run
this; the scripts encode all of it.

---

## 0. What you need

- 1-2 DGX Sparks with a working driver (`nvidia-smi` shows GB10)
- For dual-node: the 200G ConnectX link up between the Sparks with static
  IPs (this guide assumes `.1` = head, `.2` = peer on `192.168.100.0/24`)
  and passwordless SSH both ways
- Disk per node: ~230 GiB for DS4-Flash (+planes), ~900 GiB for GLM-5.2
- No root needed (one caveat: your user must be able to run `systemd-run
  --user`, true by default)

## 1. Install (each node, or install once and rsync)

```bash
# vLLM 0.24.0 has aarch64 wheels on PyPI — no Docker, no source build
python3.12 -m venv ~/venvs/vllm-moet
~/venvs/vllm-moet/bin/pip install vllm==0.24.0 "ray[default]==2.56.0"

git clone -b spark-gb10 https://github.com/Sapid-Labs/vLLM-Moet ~/Dev/vLLM-Moet
cd ~/venvs/vllm-moet/lib/python3.12/site-packages
git apply ~/Dev/vLLM-Moet/patch/vllm-moet-v0.24.0.patch          # upstream patch
git apply ~/Dev/vLLM-Moet/spark/spark-unified-memory.patch       # this port

# dep pins (see docs/v024-port.md for why)
~/venvs/vllm-moet/bin/pip uninstall -y flashinfer-cubin
~/venvs/vllm-moet/bin/pip install flashinfer-python==0.6.14
# DeepGEMM nv-dev builds clean on aarch64 (~7 min)
git clone https://github.com/deepseek-ai/DeepGEMM ~/Dev/DeepGEMM
cd ~/Dev/DeepGEMM && git checkout a6b593d2826719dcf4892609af7b84ee23aaf32a
git submodule update --init --depth 1 third-party/cutlass third-party/fmt
~/venvs/vllm-moet/bin/python setup.py bdist_wheel
~/venvs/vllm-moet/bin/pip install --no-deps dist/deep_gemm-*.whl
```

Sanity-check the kernels on your silicon (all should PASS at rel ~2-3e-3):

```bash
cd ~/Dev/vLLM-Moet
PYTHONPATH=spark CUBIN=kernels/cubins-sm120/moe_w2_mm_k4096.cubin K=4096 \
  ~/venvs/vllm-moet/bin/python kernels/gen/moe_w2_check.py
```

Second node: same paths matter. `rsync -a ~/venvs/vllm-moet peer:~/venvs/`
and `rsync -a ~/Dev/vLLM-Moet peer:~/Dev/` is sufficient (identical OS/user).

## 2. Model prep: download + prepack

The one non-obvious step. **Do not let the server convert weights at load
time** — on unified memory that conversion OOMs the box (see README). The
`prepack_planes.py` tool converts the checkpoint's experts to the kernels'
2-bit format once, on disk; the server then just reads them.

```bash
hf download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/models/hf/DeepSeek-V4-Flash
# and/or
hf download zai-org/GLM-5.2-FP8 --local-dir ~/models/hf/GLM-5.2-FP8

# prepack (auto-detects checkpoint flavor; contained; restartable — rerun
# anytime, finished layers are skipped). DS4 ~15 min, GLM ~2 h.
systemd-run --user --scope -p MemoryMax=40G \
  ~/venvs/vllm-moet/bin/python ~/Dev/vLLM-Moet/spark/prepack_planes.py \
  --model ~/models/hf/DeepSeek-V4-Flash
```

Output lands in `<model>/moe_w2_planes/` (73 GiB for DS4, ~192 GiB for GLM).
For dual-node GLM, both nodes need the checkpoint AND the planes — prepack
once and `rsync` both to the peer (the 200G link makes this quick).

## 3. Single Spark: DeepSeek-V4-Flash

```bash
~/Dev/vLLM-Moet/spark/serve-ds4-flash-1node.sh --eager    # first run: use --eager
~/Dev/vLLM-Moet/spark/smoke.sh                            # coherence check
```

First start JIT-compiles flashinfer kernels (minutes); later starts take
~3 min (plane read + dense stream). Expect ~105/121 GiB used and ~21 tok/s
single-stream. An OpenAI-compatible API serves on `:8000`.

## 4. Two Sparks: GLM-5.2 (753B, pipeline-parallel)

```bash
# on the head node (.1):
~/Dev/vLLM-Moet/spark/start-ray-cluster.sh    # raylets on both nodes, memory-capped
~/Dev/vLLM-Moet/spark/serve-glm52-pp2.sh --eager
```

What the scripts encode (don't skip these if you roll your own):

| knob | why |
|---|---|
| `VLLM_MOE_W2_PREPACKED_DIR` | planes from disk, no load-time conversion |
| `VLLM_MOE_W2_DELTA_GB=0` | the FP4 delta tier double-allocates on unified memory |
| `VLLM_MOE_W2_FADVISE_GLOB` | `cudaMalloc` can't reclaim page cache on GB10 |
| `MALLOC_MMAP_THRESHOLD_=65536` | glibc arenas otherwise retain ~10 MB loader buffers |
| `--kv-cache-memory-bytes 2G` + util 0.85 | vLLM's KV budget math counts system-wide usage on GB10; pin KV explicitly |
| `VLLM_PP_LAYER_PARTITION=38,40` | the head node also hosts the driver/desktop — give the peer the heavier half |
| `systemd-run --scope -p MemoryMax=…` | contained failure: per-node commitments must sum **below** 121 GiB or a global OOM takes the desktop down |
| `NCCL_IB_DISABLE=1` + `*_SOCKET_IFNAME` | TCP over the 200G link; PP moves one hidden-vector per token, RoCE tuning not worth it |
| raylets from **login shells** | Ray actors inherit the raylet's PATH — flashinfer JIT needs `ninja`/`nvcc` |

## 5. Troubleshooting

- **"Free memory on device … is less than desired GPU memory utilization"**
  at startup: page cache is squatting. `python3 spark/purge-cache.py
  ~/models/hf` on both nodes (the GLM script does this automatically).
- **Engine killed at ~60% of load**: you're converting at load time (no
  prepacked planes found — check `VLLM_MOE_W2_PREPACKED_DIR`), or your
  per-node memory commitments sum above physical.
- **`ray status` shows 1 node**: peer raylet died or wrong IP; rerun
  `start-ray-cluster.sh` (it force-stops both sides first).
- **Stuck placement groups after a failed launch**: `ray stop --force` on
  both nodes, `pkill -9 -f EngineCore`, restart the cluster.
- **nvidia-smi shows no memory numbers**: normal on GB10; watch `free -g`.

## 6. Known limitations (today)

- Eager mode only so far; CUDA-graph mode untested on GB10 (expect modest
  gains — decode is bandwidth-bound and MTP already hides launch latency).
- GLM PP2 runs without MTP (the drafter's experts don't fit rank 1's
  budget); an uneven-partition + MTP config is future work.
- 8K context configs; KV is tiny (MLA), so longer windows are mostly a
  matter of raising `--kv-cache-memory-bytes` and `--max-model-len`.
- 2-bit quality: upstream's QUANT_PROBE numbers (MTP acceptance ≥ FP4
  baseline); we've verified coherence + arithmetic on-device, not full evals.
