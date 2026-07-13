# 01 — TP2: FULL-cudagraph cross-node NCCL allreduce replay hangs

## Target

GLM-5.2 single-stream decode ≥ 10 tok/s on the two Sparks. TP2 with FULL
CUDA graphs is the only identified path (projected 14–20 tok/s). Everything
else about TP2 is already built and validated; this ONE bug blocks it.

## top-k reduction attempt (2026-07-11, session 4) — lever verified, measurement blocked by a settling effect

- **GOAL:** cut routed active experts/token (top-k) to reduce per-token expert
  bytes — the one lever that directly attacks the bandwidth floor (pool-REAP
  keeps top-8, so it does NOT; this does).
- **The lever works and is available:** `--hf-overrides '{"num_experts_per_tok":N}'`
  sets top-k; the custom moe_w2 kernel reads `top_k = topk_ids.shape[1]`
  dynamically (NOT hard-wired to 8, `moe_w2_cubit.py:1121`); router renormalizes
  (`norm_topk_prob=True`); GLM has 1 always-on shared expert cushioning quality.
  Kernel tiles by 16-token blocks (`moe_align_block_size`), so decode work
  scales as ~top_k tiles → fewer experts SHOULD be faster (no structural k≠8
  penalty).
- **BUT the speed numbers are unreliable — proven confound.** Two k=8
  measurements on the SAME server back-to-back gave **2.63 then 4.23 tok/s**
  (1.6× swing, no reboot). So all k=6-vs-k=8 deltas this session were settling
  noise, not top-k. Do NOT trust: k8=2.64/4.38, k6=2.55/2.50 — all confounded.
- **ROOT CAUSE — measurement protocol gap (IMPORTANT, applies to ALL perf runs):**
  `warm-planes.sh` only fills the *file* page cache. The moe_w2 path has a
  learned **resident-expert tier** (`mark_seen`/`ensure_resident` in
  moe_w2_cubit) populated by actual **inference**, not file reads. First
  inference pass = cold tier (~2.6 tok/s); after ~500 tokens of diverse decode
  it settles (~4.2). A single short warmup request is NOT enough. **Corrected
  protocol: after warm-planes, run a ~300-500 token generation to populate the
  resident tier, THEN measure (differential 384 vs 32).**
- **Second, separate factor:** desktop **Firefox on `.1`** was contending for
  the shared LPDDR5X — closing it + `compact_memory` lifted a degraded k=8 from
  2.64→4.38. GPU memory-stall signature during degradation: ~17-20 W, 2405 MHz,
  96% util but **0% mem-controller util** (GPU stalled on ATS, not computing —
  and on unified memory this still reads as 96% "util", so util alone can't
  distinguish stalled-on-memory from compute-bound).
- **REPRODUCIBLE, environment-independent finding — QUALITY at k=6 degrades.**
  Across multiple boots, k=6 broke the arithmetic probe (24×17) into off-topic
  rambling, while k=8 handled it (=408) every time. So top-6 already costs
  coherence even with the shared-expert cushion. k=4 quality untested.
- **NEXT (clean protocol):** reboot k=8 → warm-planes → 500-tok settle gen →
  measure ×2 (confirm stable) → repeat for k=6, k=4 in one stable window
  (Firefox closed). Only then is the tok/s curve interpretable. Node `.1` (5-day
  uptime) may need a fresh reboot for persistently stable bandwidth.

### CLEAN CURVE (2026-07-11, settle-first protocol, Firefox closed) — lever WORKS

| top-k | decode tok/s | ms/tok | vs k8 | quality |
|-------|--------------|--------|-------|---------|
| 8 | 4.0 | 242 | — | all probes coherent |
| 6 | 5.9 | 178 | +48% | code/reason OK; 24×17 arithmetic breaks |
| 4 | ~9–16 (settling variance) | 110 | +130–300% | COLLAPSE: Australia-capital → hallucination, arithmetic broken |

- **Top-k reduction is a real, working speed lever** — monotonic: fewer active
  routed experts → fewer expert bytes/token → faster, ~read-bound-linear
  (8→6 = 0.75× experts → +48%). **k=4 crosses the 10 tok/s target.**
- **Quality is the wall, not speed.** Naive drop+renorm (norm_topk_prob already
  on): k=6 loses one prompt class, k=4 collapses (hallucinations, broken facts).
  So the target is reachable ONLY with quality recovery at low k — a short
  distillation/finetune to adapt GLM to fewer active experts (the shared expert
  alone isn't enough below k≈6). k=5/k=3 untested; k=4 measurement still had
  settling variance (9.08 then 16.5) — true steady-state likely upper half.
- **Bottom line:** the fast-decode path to 10 tok/s is top-k reduction + a
  recovery finetune, NOT MTP (net loss) and NOT pool-REAP (doesn't cut per-tok
  bytes). This is the lever to invest in.

## MTP under TP2 (2026-07-10, session 3) — CONFIRMED net loss, even at 93.5% acceptance

- Built `serve-glm52-tp2-mtp.sh` (TP2 FULL graphs + GLM's native nextn drafter;
  arch resolves to `DeepSeekMTPModel`, method `mtp`, k=1). Boots clean on the
  NCCL-2.30.7 cluster, PYNCCL allreduce, **no hang, output coherent/correct**.
- **k=1 result: 4.30 tok/s (233 ms/token) vs 5.19 no-MTP → −17% LOSS**, despite
  an excellent **93.5% acceptance rate** (200/214 draft tokens accepted).
- **Why high acceptance still loses:** the stack is weight-read-bound. A k=1
  step runs verify (2 positions) + draft (1 position) ≈ 3 position-equivalents
  of MoE expert reads for only ~1.94 tokens/step (1 base + 0.935 accepted).
  Read bytes scale with positions; token yield caps at ~2×. 451 ms/step ÷
  1.94 tok = 233 ms/tok > 193 ms baseline. Spec decode trades **compute/launch
  latency** for tokens — but here we're **memory-bandwidth** bound, and it
  *adds* memory traffic. Confirms the prior PP-era k=2 verdict generalizes to
  TP2 and to the best case (k=1, near-ceiling acceptance).
- **When MTP would flip to a win:** only if expert weights were **resident in
  GDDR** (verify becomes compute-bound → extra positions ~free) rather than
  streamed from unified memory. So MTP's viability is gated on the *resident-
  planes* / REAP-pruning levers, not on the drafter or acceptance. k>1 not worth
  testing — reads more, yields less marginal accepted token; strictly worse here.

## RESOLUTION (2026-07-10, session 2) — hang FIXED by NCCL 2.30.7; new floor is GPU-bound

- **THE HANG IS FIXED.** Swapping pynccl's NCCL runtime 2.28.9 → **2.30.7**
  (`VLLM_NCCL_SO_PATH` in raylet COMMON, staged at `~/nccl-2.30.7/libnccl.so.2`
  both nodes) makes TP2 FULL_AND_PIECEWISE graphs **stable**: the 208-tok gen
  that reliably deadlocked on 2.28.9, plus 2×512-tok, all completed. Output
  **coherent and correct** (primes/transformer prompts). torch keeps its own
  bundled 2.28.9 for the default PG (harmless); the TP/EP allreduce dispatches
  through **PYNCCL → 2.30.7** (confirmed: `pynccl.py:113 vLLM is using
  nccl==2.30.7`, `cuda_communicator Using ['PYNCCL'] ... for group 'tp:0'`).
- **BUT the target is not met via TP2.** Clean differential (384 vs 32 tok):
  **5.19 tok/s, 193 ms/token**, ~9.8 s fixed/req. That's ~2× eager (2.49) —
  graphs removed the CPU-launch overhead as predicted — but only **≈ PP2**
  (4.9–5.5), far below the 14–20 projection.
- **New bottleneck is GPU-side, not CPU/network.** During decode **GPU util =
  96 % on BOTH nodes** (not idle → not latency/rendezvous bound). cgroup memory
  ruled out (vLLM scope at 1.9 GB vs 10 G cap, `memory.events max 0`; planes sit
  in the 87 GB system page cache, no reclaim). So the 193 ms/token is real
  GPU-side MoE work — expert-weight reads over ATS/unified-memory + compute.
  The old "26 ms warm reads" estimate was optimistic; effective per-token
  GPU-bound cost is ~193 ms. **TP2 and PP2 converge (~5 tok/s) because both move
  the same total expert bytes/token; TP2 halves the per-rank read but adds 156
  allreduces, PP2 reads serially but adds none — a wash on this stack.**
- **Implication:** beating 10 tok/s single-stream needs less per-token GPU work,
  not more parallelism — i.e. **REAP expert-pruning** (fewer active experts →
  fewer bytes/token, the cluster's stated purpose), resident planes in GDDR
  instead of CPU-mapped, or **MTP/speculative** multi-token decode to amortize.
- **Repo state:** `start-ray-cluster.sh` COMMON now carries
  `NCCL_GRAPH_MIXING_SUPPORT=1` (ruled out as the fix — harmless, can drop) and
  `VLLM_NCCL_SO_PATH=~/nccl-2.30.7/libnccl.so.2` (the actual fix).
  `serve-glm52-tp2.sh` has `NCCL_GRAPH_MIXING_SUPPORT` default line (also
  droppable). Venv `fp8.py`/`moe_w2_cubit.py` fixes still need folding into
  `spark/*.patch`. NOTE for the record: **NCCL/VLLM_NCCL env only reaches Ray
  actors via the raylet (start-ray-cluster.sh), NOT the serve script** —
  driver-forwarded env does not deliver it (verified this session).

## Session 2 update (2026-07-10, later) — flag ruled out, hang localized

- **KEY POSITIVE: FULL graphs run correctly AND fast when they don't hang.**
  Warm 16-tok request: 2.33 s wall → **~15 tok/s on the decode steps** (in
  target range). So the perf payoff is real; only the hang blocks it.
- **The hang is intermittent and generation-length dependent.** 16-tok
  requests usually survive; **208-tok reliably hangs** (240 s curl timeout →
  engine `sample_tokens` RPC watchdog → EngineDeadError). It's a function of
  the number of graph replays, not the first request per se. (This revises
  attempt-5's "first request hangs" — short first requests can pass.)
- **Silent proxy-progression deadlock confirmed.** At the hang, BOTH ranks
  logged enqueues up to the *identical* AllGather opCount (0x38) then stalled
  together; GPUs idle, RoCE counters flat, **no NCCL WARN**. Collective mix on
  the one comm: 3456 AllReduce + 57 AllGather (count 77440 bf16).
- **RULED OUT: `NCCL_DEBUG=INFO`** does not mask/prevent it (early wrong guess).
- **RULED OUT: `NCCL_GRAPH_MIXING_SUPPORT=1`.** Baked into raylets, confirmed
  present on BOTH workers, 208-tok still hung identically. Not the fix.
- **INFRA GOTCHA (fixed): NCCL env in the serve script does NOT reach Ray
  workers.** Driver-env forwarding does not deliver NCCL_* to the actors — they
  take NCCL env from the **raylet**. First graph-mixing test was invalid (flag
  only in the vllm CLI/driver, absent on both workers). Any NCCL flag / NCCL
  version experiment MUST go in `start-ray-cluster.sh` `COMMON` + cluster
  restart, then verify on the worker via
  `tr '\0' '\n' < /proc/$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|head -1)/environ | grep NCCL_`.
  (`NCCL_GRAPH_MIXING_SUPPORT=1` is now in COMMON; harmless, leave or remove.)
- **Untested after this session (ranked):** (1) `splitting_ops` to pull
  `vllm::all_reduce`/`all_gather` out of the captured decode graph — now the
  TOP candidate since collectives-in-graph is the confirmed failure locus;
  op name confirmed `all_reduce` (namespace `vllm`) in
  `distributed/parallel_state.py:324`. (2) NCCL **2.30.7** swap — libnccl.so.2
  staged at `~/nccl-2.30.7/libnccl.so.2` on BOTH nodes; deliver via
  `VLLM_NCCL_SO_PATH` in raylet COMMON (NOT the serve script — see gotcha).
  (3) `NCCL_LAUNCH_MODE=GROUP`. (4) `FULL_DECODE_ONLY` (likely same mixing).

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
