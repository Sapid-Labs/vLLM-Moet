# GOAL — GLM-5.2 on 2× DGX Spark: 20 tok/s single-stream at current quality

**Target:** sustained single-stream decode **≥ 20 tok/s** on the two Sparks
(TP2), at **no quality regression** vs today's shipped config. Ends when a
config sustains ≥20 tok/s on the standard protocol AND matches the current
quality bar (battery CLEAN + GSM8K ≥ ~90% + GPQA/MMLU-Pro within noise of the
15 tok/s baseline once those land).

**Baseline (2026-07-13, session 8):** 15.0 tok/s = **~67 ms/output-token**,
via native k=8 + pruned-208 resident planes + MTP k=1 (67.6% accept, 1.31×).
Non-MTP fault-free floor is 11.4 tok/s ≈ 88 ms/verify-step.

**The jump:** 15 → 20 tok/s is 67 → **50 ms/tok**, a **1.33×** improvement,
at fixed quality. See `spark/handoffs/02-*.md` for how we got to 15.

---

## The budget: where do the 67 ms go? (this decides everything)

The one load-bearing fact from session 6b–7b: **fault-free decode is NOT
bandwidth-bound.** Effective plane-read rate is **~52 GB/s vs 273 GB/s raw
LPDDR5X** — a ~5× gap. THP (2 MiB pages) closed the *raw* read micro-benchmark
to 170 GB/s but moved the serving rate **zero**, which means the raw plane read
is no longer the binder. Something else eats the step. The kernel source says
what:

- **Decode runs a 4-token-block kernel (MC4); single-stream feeds it 1–2 real
  rows.** The 16-token prefill kernel (MC16) "amortizes plane reads 4× for
  ~1.5× throughput" (`moe_w2_cubit.py` ~L1160). So each unique expert's 2-bit
  plane is read to compute **one token's** worth of matmul — the tensor cores
  are ~idle, the read is the cost, and we get almost none of the M-amortization
  the kernel is capable of. GPU at 17–20 W confirms: stalled, not computing.
- **Per-layer host/launch prologue × 75 layers:** every MoE layer does
  `per_token_group_quant_fp8` → `moe_align_block_size` → 2× `index_select` →
  triton desc launch → GEMM → combine. At M=1–2 these fixed costs may dominate
  the actual matmul. FULL cudagraphs should capture most of it — *verify they
  actually do* (dynamic-shape ops can graph-break).
- **TP2 per-layer allreduce over RoCE × ~75 layers/token.** Prior PP profiling
  put "most single-stream wall in comm/wait." A ~0.1–0.2 ms small-message
  allreduce × ~75 layers = **8–15 ms/tok** — plausibly a quarter of the budget.

**Thesis:** the 5× bandwidth headroom is real and reachable. The path to 20
(and likely well past it) is raising *useful tokens per plane read* and cutting
*per-layer fixed overhead* — not finding more bandwidth.

**STEP 0 (do first, it's cheap and disambiguates all of the below):** profile
one decode step. `nsys` + the `VLLM_SCHED_TRACE` hook
(`spark/sched-trace-instrumentation.patch`) + the existing
`_moe_w2_forward_timed` timings. Answer three questions:
1. What fraction of 67 ms is moe_w2 GEMM vs layer prologue vs allreduce vs attn?
2. Are the cudagraphs actually swallowing the torch prologue, or is it eager?
3. Does achieved plane-bandwidth rise with M? (compare batch-1 vs the 8-stream
   aggregate per-stream rate — if throughput/stream climbs, we're M-starved.)
Run it on a **freed** server (evals must be done first — don't co-run).

---

## Ranked levers (mechanism · expected gain · cost · quality risk)

### Tier 1 — attack the M-starvation (best expected return, no training)

1. **Tree / multi-token speculation (raise verify M from 2 → 4–8).**
   Mechanism: verify several draft positions in one moe_w2 pass. Every draft
   position that routes to an **already-read expert** is free — pure
   amortization of the plane read the kernel is currently wasting. EAGLE-2/3
   tree drafting or MTP k≥2 both raise M.
   - Gain: potentially large — this is exactly the MC4→MC16 "4× read
     amortization" lever, applied via speculation instead of batching.
   - Cost: medium (vLLM spec-tree support for GlmMoeDsa MTP; may need the
     drafter to emit a tree).
   - Risk: none to quality (verify is exact); risk is *net* loss if draft
     positions route to disjoint experts (the code-domain failure mode) —
     measure expert overlap between draft positions, not just acceptance.
   - **Cheap first probe: `MTP_K=2` sweep** (one reboot) + per-domain
     acceptance + measure plane-read bytes/token. Already flagged in the
     handoff as untested. Do this right after Step 0.

2. **Grow the decode block / batch the shared+attention reads.** Even at M=1,
   the shared expert + attention + 3 dense layers are read every token
   regardless of routing. Check whether MC8 (vs MC4) helps the always-read
   portion, and whether the delta/base GPU-resident tier can serve the hot
   experts from a small pinned pool (cutting the plane read entirely for the
   top-N experts) now that planes are 79 GB and headroom exists.
   - Gain: unknown, possibly modest. Cost: low–medium. Risk: none.

### Tier 2 — cut per-layer fixed overhead (gated on Step 0)

3. **Kill/curtail the per-layer allreduce cost.** If Step 0 says comm is
   8–15 ms/tok: tune NCCL for latency (small-message algo, `NCCL_ALGO`,
   `NCCL_PROTO=Simple/LL`), try jumbo frames (the 1500-MTU path caps us — see
   root `CLAUDE.md`; needs switch access), or overlap the allreduce with the
   next layer's plane read. Gain up to the comm fraction. Risk: none.

4. **Fold the layer prologue into the graph / fuse it.** If the torch prologue
   is eager or graph-breaking, fusing quant+align+gather into one launch (or
   ensuring capture) removes fixed per-layer latency × 75. Gain = prologue
   fraction. Cost: medium (kernel work). Risk: none.

### Tier 3 — fewer bytes per token (quality-gated, slower to validate)

5. **Sub-2-bit / mixed-precision experts** (e.g. ternary on the coldest kept
   experts, 2-bit on hot). Cuts the read linearly. Cost: high (new kernel
   path + requant). Risk: real quality cost — needs the full eval battery per
   variant. Only if Tier 1–2 stall.

6. **Adaptive-k / cascade routing** (k=8 on hard tokens, k=4 on easy). Cuts
   average bytes without the flat-k=4 quality collapse. Cost: high (routing
   logic + finetune to calibrate the gate). Risk: high. Research-grade;
   parked unless Tier 1–2 miss.

7. **REAP-saliency prune deeper than 208.** Frequency prune already at 208;
   saliency might safely go lower → smaller planes. But planes are already
   resident, so this only helps if it also cuts *per-token routed* reads
   (it doesn't — top-8 still reads 8). Value is quality-neutral shrink, not
   speed. Deprioritized for the 20 tok/s goal specifically.

---

## Rejected / parked (don't re-derive — see handoff attempts log)

- **Lower native top-k** — quality collapses below k≈6; the whole pruned-pool
  approach exists to avoid this. Off the table for "same quality."
- **NVFP4 / any ≥4-bit** — more bytes than 2-bit, wrong direction, won't fit RAM.
- **RTX 5090 drafter** — 32 GB can't hold the verify (the expensive half).
- **Naive MTP on unpruned planes** — the −17% result; residency is the
  precondition (we now have it, hence Tier 1).

---

## Recommended first three moves

1. **Step 0 profile** on the freed server → get the 67 ms breakdown. Nothing
   below is worth guessing at without it.
2. **`MTP_K=2` sweep** + per-domain acceptance + expert-overlap measurement
   (cheap, one reboot, directly tests the Tier-1 thesis).
3. Based on (1): if comm-bound → Tier 2 #3; if M-starved → Tier 1 #1 (tree
   spec). Re-measure against the standard protocol + full quality battery.

## Resume mechanics / gates

- Serve, warm, measure, quality-gate: see `spark/RUNBOOK.md` §4 and the
  MEASUREMENT PROTOCOL in `spark/handoffs/02-*.md` (settle post-domain, ×2,
  usage-token differentials, power.draw sanity, battery + GSM8K).
- Do not profile/bench while evals run on the same server (wrecks both).
