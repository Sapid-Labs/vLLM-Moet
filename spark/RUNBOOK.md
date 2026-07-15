# Running vLLM-Moet on DGX Spark — dual-node runbook

Run DeepSeek-V4-Flash (159B) on **one** NVIDIA DGX Spark, and GLM-5.2 (753B)
across **two**, using vLLM-Moet's 2-bit expert kernels. Everything here is
measured on real hardware (GB10, sm_121, 121 GiB unified LPDDR5x per node,
CUDA 13, driver 580.159.03).

**Status (2026-07-14):**
- DeepSeek-V4-Flash, 1 Spark: **working** — coherent greedy output,
  ~21 tok/s single-stream (eager + MTP k=2; ≈ the 273 GB/s bandwidth ceiling)
- GLM-5.2, 2 Sparks (TP2): **working — ~20 tok/s single-stream (greedy,
  deterministic)**, up from 15.0. The new lever (session 9-10): **NVFP4 the
  attention + shared-expert linears** (weight-only FP4 via Marlin; GB10 DOES have
  FP4 tensor cores [corrected session 12] but Marlin W4A16 is faster for M=1 decode
  than Cutlass W4A4 → 4-bit weight read, bf16 compute). Decode is bandwidth-bound
  and — once experts were 2-bit — *attention* was ~60% of the per-token byte
  read while still at FP8, so cutting it to 4-bit bought ~1.33×. Progression:
  FP8-attn 15.0 → NVFP4 big-3 (`o/q_b/kv_b`) ~18 → full attention+shared ~20.
  See `spark/NVFP4-DENSE.md`. **Quality of the NVFP4 build not yet re-evaluated**
  (rel-L1 ~0.09 on the quantized weights; eval pending). Real-world *sampled*
  throughput varies ~13-20 (MTP acceptance depends on how well the draft
  matches the sampled token; greedy shows the clean ~20).
  The prior shipped config (**15.0 tok/s, quality battery clean, GSM8K 91%**)
  stacks three levers (sessions 1-8, `spark/handoffs/02-*.md`): FULL cudagraphs
  over RoCE (needs NCCL 2.30.7), expert-pruned planes (256→208/layer — shrinks
  planes 97→79 GB/rank so they fit page cache; decode goes fault-free), and MTP
  speculative decode k=1. Routing stays at native top-k=8. NVFP4-attn stacks on
  top of all three.
- GLM-5.2 PP2 fallback: ~5.5 tok/s single-stream, ~17 tok/s aggregate at
  8 streams. Simpler, no NCCL version pin; keep it in your pocket if TP2
  misbehaves.
- Run `spark/warm-planes.sh` after startup either way to avoid slow first
  requests (plane faults from NVMe).

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
- No root needed to serve — but for full speed, one root one-liner: DGX OS
  ships an 8 MB `memlock` limit that breaks NCCL-over-RDMA (`ibv_reg_mr:
  Cannot allocate memory` -> "unhandled system error"). Fix:
  `sudo tee /etc/security/limits.d/99-rdma-memlock.conf <<< $'youruser soft memlock unlimited\nyouruser hard memlock unlimited'`
  (applies to NEW login sessions; the cluster script self-SSHes for this
  reason). Without it the scripts still work over TCP, ~15% slower.

## 1. Install (each node, or install once and rsync)

```bash
# vLLM 0.24.0 has aarch64 wheels on PyPI — no Docker, no source build
python3.12 -m venv ~/venvs/vllm-moet
~/venvs/vllm-moet/bin/pip install vllm==0.24.0 "ray[default]==2.56.0"

git clone -b spark-gb10 https://github.com/Sapid-Labs/vLLM-Moet ~/Dev/vLLM-Moet
cd ~/venvs/vllm-moet/lib/python3.12/site-packages
git apply ~/Dev/vLLM-Moet/patch/vllm-moet-v0.24.0.patch          # upstream patch
git apply ~/Dev/vLLM-Moet/spark/spark-unified-memory.patch       # this port
git apply ~/Dev/vLLM-Moet/spark/sync-pp-spec-sched-fix.patch     # stock-0.24 bug:
#   sync-sched + PP + spec races draft delivery against in-flight steps ->
#   negative num_scheduled_tokens -> worker assert. Repro/regression test:
#   spark/test_sync_pp_spec.py. Optional per-step scheduler tracing for
#   debugging: spark/sched-trace-instrumentation.patch (VLLM_SCHED_TRACE=<path>).

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

Output lands in `<model>/moe_w2_planes/` (73 GiB for DS4, ~192 GiB for GLM
full-pool). For dual-node GLM you want the **TP2-sharded, expert-pruned**
planes instead — that's the 15 tok/s config. Either produce them
(`prepack_planes.py` TP2 variant, then `spark/routing/prune_planes.py`
with `spark/routing/keep208.json` — row-select, no requant, ~15 min), or
**skip prepack entirely** and download the exact artifact we serve:

```bash
# ~79 GB per rank; each node downloads only ITS shard
# on .1:
hf download sapidlabs/GLM-5.2-2bit-MoE-planes-pruned208-tp2 --include 'rank0/*' \
  --local-dir /tmp/planes && mv /tmp/planes/rank0 ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208
# on .2: same with 'rank1/*'
```

Both nodes still need the base checkpoint (config/tokenizer/non-expert
weights). Pruning note: 48 coldest-by-traffic experts dropped per MoE layer
(selection-masked, gates renormalize — REAP-style); this is what makes the
planes page-cache-resident (79 < ~88 GB cache), which is the whole speed
story. Quality: correctness battery clean, GSM8K 91% (300-ex, measured on
this exact config).

## 3. Single Spark: DeepSeek-V4-Flash

```bash
~/Dev/vLLM-Moet/spark/serve-ds4-flash-1node.sh --eager    # first run: use --eager
~/Dev/vLLM-Moet/spark/smoke.sh                            # coherence check
```

First start JIT-compiles flashinfer kernels (minutes); later starts take
~3 min (plane read + dense stream). Expect ~105/121 GiB used and ~21 tok/s
single-stream. An OpenAI-compatible API serves on `:8000`.

## 4. Two Sparks: GLM-5.2 (753B, tensor-parallel — the 15 tok/s config)

One extra prerequisite: **NCCL ≥ 2.30.7** staged at `~/nccl-2.30.7/libnccl.so.2`
on both nodes (extract from the `nvidia-nccl-cu13` wheel). Torch's bundled
2.28.9 deadlocks TP2 FULL cudagraph replay on GB10+CX7 (`spark/handoffs/01-*.md`);
`start-ray-cluster.sh` delivers the pin via `VLLM_NCCL_SO_PATH` and also
resolves each node's RoCEv2 GID index at start time (they drift across
reboots — check its `== RoCEv2 GID indexes ==` banner first if NCCL dies at init).

```bash
# on the head node (.1):
~/Dev/vLLM-Moet/spark/start-ray-cluster.sh    # raylets on both nodes, memory-capped
VLLM_MOE_W2_PREPACKED_DIR=$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208 \
  MTP_K=1 ~/Dev/vLLM-Moet/spark/serve-glm52-tp2-mtp.sh
~/Dev/vLLM-Moet/spark/warm-planes.sh ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208 --peer
```

Boot log must show `POOL-PRUNED 256->208` (once per MoE layer per rank) —
if not, the pruned-planes dir wasn't picked up. Expect ~15 tok/s sustained
single-stream after a few hundred tokens of settling (the resident-expert
tier warms by inference, not by `warm-planes.sh`). During decode,
`power.draw` >~30 W means working; ~17-20 W means memory-stalled. Speed
sanity: measure with `usage.completion_tokens` differentials (512 vs 1024),
never by counting stream chunks — MTP bundles tokens per chunk. Or just run
`spark/demo.py`, which does it right.

### PP2 fallback (simpler, slower)

```bash
~/Dev/vLLM-Moet/spark/start-ray-cluster.sh
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
| NCCL over RoCE (`NCCL_IB_HCA` + per-node `NCCL_IB_GID_INDEX`) | decode profiling showed most single-stream wall in comm/wait; RoCE beats TCP ~15%. Needs the memlock fix above |
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

## 6. Known limitations & hard-won verdicts (2026-07-13)

- **MTP profitability is residency-dependent — don't trust a spec-decode
  benchmark taken under page-cache thrash.** On full-pool planes (97 GB/rank
  > cache) MTP *loses* (~-17% at TP2 k=1; 3.65 vs 4.9-5.5 under PP2 k=2):
  verify's extra expert reads amplify NVMe faulting. On pruned resident
  planes the same MTP k=1 is **+31%** (11.4 → 15.0, ~68% acceptance). The
  PP2 fallback still runs without MTP for this reason.
- Expert selection for the prune is frequency-based (traffic-cold, measured
  over 5.3k tok / 12 domains at k=8), not REAP-saliency; ~5.9% of routed
  slot traffic renormalizes onto kept experts (worst layer 12%). REAP
  calibration is the open upside track, not a blocker.
- Reduced top-k (`num_experts_per_tok` override) scales speed linearly in
  bytes but **collapses quality below k≈6** (k=4: broken arithmetic,
  hallucinated facts). The pruned-pool route made it unnecessary — routing
  ships at native k=8.
- Measurement footguns (each cost us a session): rates are settle- and
  prompt-domain-dependent (measure post-settle, ×2); a node up 5+ days
  degrades bandwidth (reboot before trusting absolutes); Firefox on the
  head node steals ~40% of LPDDR5X bandwidth; count tokens via
  `usage.completion_tokens`, never SSE frames.
- Cold long-prompt TTFT is minutes on full-pool planes (~95 GiB/rank of
  plane faults, ~150 s per 512-tok chunk). Pruned planes largely fix this;
  still run `spark/warm-planes.sh` after any cache purge, and don't mistake
  a cold prefill for a hang.
- 8K context configs; KV is tiny (MLA), so longer windows are mostly a
  matter of raising `--kv-cache-memory-bytes` and `--max-model-len`.
- Quality on the shipped pruned+MTP config, measured on-device:
  correctness battery clean, GSM8K 91% strict / 92% flexible (300 ex,
  `max_gen_toks=6000` — thinking models score garbage under default gen
  caps). GPQA-Diamond / MMLU-Pro / IFEval runs in progress; scores land in
  the How To Spark evals page and the HF model card
  (`sapidlabs/GLM-5.2-2bit-MoE-planes-pruned208-tp2`).
