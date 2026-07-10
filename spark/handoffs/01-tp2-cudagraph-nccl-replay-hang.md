# 01 — TP2: FULL-cudagraph cross-node NCCL allreduce replay hangs

## Target

GLM-5.2 single-stream decode ≥ 10 tok/s on the two Sparks. TP2 with FULL
CUDA graphs is the only identified path (projected 14–20 tok/s). Everything
else about TP2 is already built and validated; this ONE bug blocks it.

## State (end of 2026-07-10)

- **Serving**: stable PP2 + graphs on :8000 (4.9–5.5 tok/s single,
  13–15 aggregate at conc 8). `spark/serve-glm52-pp2.sh`.
- **TP2 planes**: on disk on both nodes at
  `~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2` (rank 0 on .1, rank 1 on .2,
  `tp_rank` stamped in meta, byte-validated). Keep — repack not needed again.
- **TP2 serve**: `spark/serve-glm52-tp2.sh`. `--eager` works and is
  CORRECT (battery-grade coherent output). Default (FULL_AND_PIECEWISE)
  hangs on the first decode request.
- **Venv drift**: three fixes live in BOTH nodes' venvs but are not yet in
  the repo patch files (fold into `spark/*.patch` when TP2 lands):
  1. `fp8.py` — `_moe_w2_active` guard + flag in the fp8 moe_w2 branch
     (mirrors modelopt; upstream's TP was NVFP4-only).
  2. `moe_w2_cubit.py` — prepacked-loader geometry validation vs layer
     config (`local_num_experts`, `2*intermediate_size_per_partition`) +
     `tp_rank` meta check.
  3. `scheduler.py`/`async_scheduler.py` — sync-PP spec fix + dormant
     VLLM_SCHED_TRACE instrumentation (already in repo as patches, applied
     to venv).

## Attempts log

1. **TP2 boot #1 (graphs)** — crashed at init:
   `Fp8MoEMethod ... should not be called` from
   `maybe_init_modular_kernel` under TP. Root cause: fp8 branch never set
   `_moe_w2_active` and lacked the no-op guard (modelopt branch has both).
   FIXED in venv fp8.py.
2. **TP2 boot #2 (graphs)** — decoded at ~0.1–0.8 tok/s, then engine died
   (`RPC call to sample_tokens timed out`, 300 s). Root cause: BOTH ranks
   loaded the full-width PP planes — the raylets carried
   `VLLM_MOE_W2_PREPACKED_DIR=.../moe_w2_planes` baked at cluster start,
   and ray workers apply raylet env OVER driver env (setdefault). Each rank
   double-computed the MoE and thrashed 190 GiB of wrong planes. FIXED:
   `start-ray-cluster.sh` now carries node truths only; loader geometry
   guard added. (Cluster was restarted with the clean env — a future
   `start-ray-cluster.sh` run keeps it clean.)
3. **TP2 boot #3 (graphs, correct planes)** — still ~0.8 tok/s then same
   RPC-timeout death on the first real request. Partially confounded by
   cold planes (see facts), but FULL-graph replay hang confirmed
   independently in attempt 5.
4. **TP2 boot #4 (eager + NCCL_DEBUG)** — WORKS. All channels
   `via NET/IB` (RoCE verbs). First cold request 3.7 s/token (NVMe
   faults); after `warm-planes`: **2.49 tok/s** steady (differential
   method), correct output. Proves: transport fine, planes fine, kernels
   fine, correctness fine. 401 ms/token = ~370 ms launch/rendezvous
   overhead (each rank drives all 78 layers + 156 sync points per token).
5. **TP2 boot #5 (FULL graphs, warm, correct planes)** — first request
   hangs a worker inside `sample_tokens`; RoCE counters at ZERO during the
   hang; GPUs idle; engine watchdog kills at 300 s. Boot-time capture
   itself succeeds — only REPLAY hangs.
6. **TP2 boot #6 (PIECEWISE only)** — survives (no hang!) but **1.84
   tok/s** — worse than eager; small captured segments add more sync than
   they save. Not a path.

## Established facts (with evidence)

- Fabric allreduce: 12 KiB (hidden=6144 bf16) = **20 µs/op** over RoCE
  (torch.distributed bench, self-SSH memlock). 156 ops/token ≈ 3 ms.
- Expert reads under TP2: ~4.5 GB/rank/token in parallel ≈ 26 ms warm
  (170 GB/s ATS page-cache rate), 3.7 s COLD (NVMe) — warm-planes after
  every boot before judging anything.
- TP2 eager per-token budget: 401 ms total − 26 reads − 3 comm ≈ ~370 ms
  CPU-side per-layer launch + rendezvous overhead. Only FULL graph capture
  removes this class of cost on this stack (PP2 gained 4.1→4.9 from
  graphs; TP2 has ~2× the per-rank launch work).
- The hang is specifically: **replay of a graph-captured PYNCCL allreduce
  across nodes**. Eager collectives fine; capture fine; piecewise replay
  fine (but slow). PP2 never hits it (pipeline send/recv sits BETWEEN
  per-rank graphs). Upstream's TP4 was single-host (NVLink/PCIe P2P — the
  vLLM allreduce backend dispatch even picks non-NCCL paths there;
  cross-node only PYNCCL is eligible).

## Next steps (ranked)

1. **See the hang**: relaunch FULL graphs with
   `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,NET,GRAPH` (forwarded to
   workers automatically now), send one request, py-spy/GDB the stuck
   worker if possible (`ptrace_scope=1` — run py-spy via `sudo` with the
   user present, or start workers under a tracer). Look for the proxy
   thread servicing captured ops.
2. **Keep collectives OUT of the captured graphs**: add
   `vllm::all_reduce` (check the actual custom-op name in this build:
   `grep -r "all_reduce" vllm/compilation/` for splitting-op registry) to
   `splitting_ops` in `--compilation-config` so FULL decode graphs break
   at each allreduce — launch overhead of the GEMM segments still
   amortizes, collectives run eagerly. Likely the cheapest real win:
   even at 156×(20 µs + eager-op overhead ~100 µs) ≈ 19 ms/token comm-side,
   projected total ≈ 26+19+graph ~20 ≈ 65 ms → **~15 tok/s**.
3. **NCCL version**: venv NCCL 2.28.9 (torch bundle). Try
   `VLLM_NCCL_SO_PATH` pointing at a newer libnccl (2.29+/2.30) — capture
   replay fixes land regularly; GB10+CX7 is a new platform combo.
4. **Env experiments on the replay path**: `NCCL_LAUNCH_MODE=GROUP`,
   `NCCL_GRAPH_MIXING_SUPPORT=1`, `NCCL_IB_SPLIT_DATA_ON_QPS=0` — cheap
   flag sweeps on the FULL-graph boot before deeper surgery.
5. If all fails: TP2 eager + CPU-overhead reduction (multi-step decode /
   async scheduling tune) — unlikely to reach 10; treat as floor-raiser.

## Resume mechanics

- Boot cadence: every serve boot purges page cache → **always
  `bash spark/warm-planes.sh ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2
  --peer` after health** (75 s) before measuring. Each debug cycle ≈ 8 min.
- Measure with the usage-token differential (SSE frames under-count):
  two non-stream requests (16 tok, 208 tok), rate = Δtokens/Δwall.
- Correctness gate: `spark/mtp_correctness_battery.py --model glm-5.2`.
- Ray cluster: `spark/start-ray-cluster.sh` (self-SSH for memlock; raylets
  carry fabric env only). Check with
  `tr '\0' '\n' < /proc/$(pgrep -f 'raylet ' | head -1)/environ | grep MOE`
  → must be EMPTY.
- Footguns: `pkill -f <pattern>` kills your own shell if the pattern is in
  the command line; sudo needs a real TTY (`!` prefix); py-spy blocked by
  ptrace_scope=1.
- User cadence preference: status update at every phase transition; ~3
  debug cycles max, then present options and let the user decide.
