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

## The budget: where do the decode bytes go? (VERIFIED 2026-07-13, session 9)

**Decode is bandwidth-bound — at the LPDDR5X ceiling.** This overturns the
session 6b–7b "not bandwidth-bound / 52 GB/s effective / 5× headroom" thesis,
which was an **accounting artifact**: it divided only the 2-bit *expert-plane*
bytes (26% of the step) by wall time and ignored the 3× larger FP8 attention +
shared-expert reads. Count *all* the bytes and the picture inverts.

**Exact per-MoE-layer decode read** (75 such layers; routed = 2-bit planes at
top-k=8, everything else at checkpoint FP8; measured from safetensors headers,
config dims `hidden=6144 moe_int=2048 heads=64 head_dim=256 q_lora=2048 v=256`):

| Component | Bytes/layer | dtype | share |
|---|---|---|---|
| **MLA attention** (q_a/q_b/kv_a/kv_b/o_proj; o_proj alone ~100 MB) | **174.6 MB** | FP8 | **60%** |
| routed experts (8 of 256) | 75.5 MB | 2-bit | 26% |
| shared expert | 37.8 MB | FP8 | 13% |
| router gate | 3.2 MB | BF16 | 1% |
| **total/layer** | **~291 MB** | | |

**Dense (attn+shared+gate) = ~216 MB = 74% of the step; routed 2-bit = 26%.**
The single biggest category is **MLA attention (60%)**, not the experts — GLM-5.2's
attention is heavy (`o_proj` 16384→6144, `q_b` 2048→16384, big head dims).

**Reconciliation (this is why we trust it):** full per-token read ≈ 75×291 MB +
3 dense layers + lm_head(~1.9 GB) ≈ **~23 GB/token**.
- Non-MTP: 23 GB × 11.4 tok/s = **262 GB/s ≈ 273 GB/s ceiling.**
- MTP 1.31×: 23/1.31 = 17.6 GB/tok → **15.5 tok/s ≈ observed 15.0.**

Both shipped rates sit right at the bandwidth wall. The 17–20 W GPU reading is
*consistent* with memory-bound stall (SMs idle on loads), so it never
discriminated bandwidth-bound vs M-starved — the byte accounting does.

**Consequence for the lever ranking:** the win must cut the *majority* byte
category. Tree/multi-token speculation attacks only the 26% expert slice and
cannot reach 20 by itself. **Cutting the dense/attention bytes is the lever.**

**STILL DO STEP 0 — it's the empirical confirm, and it's already queued.**
`spark/profile_decode.sh` Phase A (M-scaling on the live server) predicts:
if bandwidth-bound, per-stream tok/s stays **flat** as concurrency 1→8 rises
(the memory system is already saturated at M=1); if M-starved it *rises*. Flat
result = the accounting above is nailed. Phases B/C still attribute the
attn-GEMM vs comm slices. Run on a freed server (evals must finish first).

---

## Ranked levers (mechanism · expected gain · cost · quality risk)

### Tier 1 (LEAD) — NVFP4 the dense/attention linears (cuts the 74% majority)

1. **NVFP4 (4-bit micro-scaled) on MLA attention projections + shared expert,
   keep experts at 2-bit.** This is the 20 tok/s move.
   - Mechanism: FP8 dense (1.0 B/param) → NVFP4 (~0.56 B/param incl. per-16
     block scales). Per layer 216→~122 MB dense ⇒ 291→~198 MB ⇒ **~1.47×** ⇒
     **~22 tok/s with MTP** (~16.9 non-MTP). Attention is 60% of the bytes, so
     that's where the win is.
   - **DO NOT graft `nvidia/GLM-5.2-NVFP4` as a drop-in.** Verified from its
     config + shard headers 2026-07-13: that repo does the *opposite* recipe —
     NVFP4 on the **routed experts** (U8-packed 4-bit + F8_E4M3 group-16 scale +
     F32 global scale), attention + shared expert left **BF16** (listed in
     `quantization_config.ignore`; `self_attn.o_proj.weight` is BF16 [6144,16384]
     = ~201 MB for that one tensor, ~2× our FP8). Its attention tensors as-is
     would nearly double our biggest read. Nobody ships NVFP4-attention because
     we're the only ones for whom attention is the marginal byte (we already
     beat their experts with 2-bit).
   - **We self-quantize.** Produce NVFP4 weights for `self_attn.{q_a,q_b,kv_a,
     kv_b,o}_proj` (+ `shared_experts.*`) via ModelOpt or llm-compressor.
     quant_config gets the **inverse** ignore list (ignore experts — they use the
     2-bit plane path; quantize attention+shared). Our model loader routes
     experts→2-bit planes, attention→NVFP4 linear.
   - **Quantize from the BF16 originals, NOT from our FP8 checkpoint.** NVFP4
     calibration off already-FP8-rounded weights compounds rounding error;
     BF16→NVFP4 gives cleaner per-16 block scales. Convenient source: the
     `nvidia/GLM-5.2-NVFP4` repo's **BF16 attention + shared-expert tensors are
     exactly the ones we want to quantize** — useless as a drop-in (§above) but
     ideal as the fp-source for our own NVFP4 pass (or pull them from the ZAI
     BF16 base). Only the attention/shared tensors are needed; skip its experts.
   - **Recommended precision: weight-only NVFP4 (W4A16), not w4a4.** Decode is
     weight-read-bound, so 4-bit *weights* + BF16 activations captures the full
     byte win with the least quality risk and **no activation calibration**.
     vLLM has it: `model_executor/kernels/linear/nvfp4/marlin.py` (Marlin-FP4,
     W4A16). w4a4 (cutlass `sm120_blockscaled_mma`) only adds compute savings we
     don't need and needs a calibration pass.
   - Cost: medium — a quant/calibration run + per-module quant wiring in the
     GlmMoeDsa loader (mixed scheme: NVFP4 linears + existing 2-bit experts).
     No new kernels: sm120 blockscaled cutlass + Marlin-FP4 + modelopt/
     compressed-tensors loader are all present & built for sm_121 (GB10).
   - Risk: real but modest — attention NVFP4 must pass the full battery. 4-bit
     micro-scaled is far gentler than our 2-bit experts. Gate on attention.
   - **Staging (de-risk):** first cut = NVFP4 on the big-3 plain linears
     `o_proj`(~100 MB) + `q_b_proj`(~34) + `kv_b_proj`(~15) = 149 of 174 MB,
     ~1.35× → ~18–19 tok/s, avoids the rope-entangled `q_a/kv_a` and MLA-core
     cache quant. Then add `q_a/kv_a` + shared expert for the full ~1.47×.
   - Memory: NVFP4 dense is *smaller* than FP8 → RAM headroom only improves.
   - Graph: verify the NVFP4 GEMM path captures under FULL cudagraphs + MTP.

### Tier 2 — attack M-starvation ONLY if profiler shows residual headroom

2. **Tree / multi-token speculation (raise verify M 2→4–8).** Demoted from lead:
   it amortizes only the **26% expert slice**, leaving the 60% attention read
   untouched — can't reach 20 alone. Still additive *after* Tier 1 (once
   attention is NVFP4, the expert slice's relative weight rises). Cheap probe:
   `MTP_K=2` sweep + per-domain acceptance + expert-overlap. Risk: none to
   quality (verify is exact); net loss only if draft positions route disjoint.

### Tier 3 — cut per-layer fixed overhead (gated on Step 0 Phases B/C)

3. **Curtail per-layer TP2 allreduce** (NCCL small-message algo `NCCL_PROTO=LL`,
   overlap with next layer's read, jumbo frames — needs switch access, see
   root `CLAUDE.md`). Gain = the comm fraction Phase C measures.
4. **Fold the layer prologue into the graph / fuse quant+align+gather.** Gain =
   the prologue fraction Phase B measures (if eager/graph-breaking).

### Tier 4 — fewer expert bytes (quality-gated, slow to validate)

5. **Sub-2-bit / mixed-precision experts** (ternary on cold-kept, 2-bit on hot).
   Cuts the 26% slice linearly. High cost (kernel + requant), real quality risk.
   Only if Tier 1–2 stall short of 20.
6. **Adaptive-k / cascade routing.** Research-grade; parked.

---

## Rejected / parked (don't re-derive)

- **Grafting nvidia/GLM-5.2-NVFP4 dense tensors** — they're BF16, not NVFP4
  (nvidia quantized the experts, not attention). See Tier 1.
- **Lower native top-k** — quality collapses below k≈6.
- **NVFP4 on the *experts*** — wrong direction; our 2-bit planes are already
  smaller than 4-bit and resident. NVFP4's value here is *attention*, not experts.
- **RTX 5090 drafter** — 32 GB can't hold the verify.
- **Naive MTP on unpruned planes** — the −17% result; residency is the precondition.

---

## Recommended first three moves

1. **Let Step 0 profiler land** (already queued behind evals) → confirm flat
   M-scaling (bandwidth-bound) + get the attn-GEMM/comm slices. Nothing below
   should be guessed without the flat-M confirm, but the accounting is strong
   enough to start staging the NVFP4 quant in parallel.
2. **Self-quantize attention big-3 (`o/q_b/kv_b`) to weight-only NVFP4**, wire
   the mixed quant scheme into the GlmMoeDsa loader, serve, re-measure on the
   standard protocol. Expect ~18–19 tok/s.
3. **Full quality battery on the NVFP4-attention build.** If CLEAN, extend to
   `q_a/kv_a`+shared for the full ~1.47× → ~22 tok/s. If quality moves, back off
   the most-sensitive projection to FP8 and re-measure the byte/quality trade.

## Resume mechanics / gates

- Serve, warm, measure, quality-gate: `spark/RUNBOOK.md` §4 + the MEASUREMENT
  PROTOCOL in `spark/handoffs/02-*.md` (settle post-domain, ×2, usage-token
  differentials, power.draw sanity, battery + GSM8K).
- Do not profile/bench while evals run on the same server (wrecks both).
- Byte-accounting script: reads safetensors headers only (no tensor data),
  categorizes attn/shared/routed/gate per layer — rerun to re-verify if dims change.
