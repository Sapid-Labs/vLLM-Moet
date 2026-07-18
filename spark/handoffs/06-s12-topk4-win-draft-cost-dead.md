# s12 (2026-07-14) — draft-cost thesis DEAD; top-k4 WIN → 21.8 tok/s; Marlin verdict; corrections

> Previous: 05-s11-*.md · Next: 07-s13-*.md

## STATUS (2026-07-14, session 12) — draft-cost DEAD; top-k WIN → 21.8 tok/s

**Two results: (1) the 30-tok/s "cheap draft forward" plan is refuted (Approach A
dead); (2) NEW WIN — MoE top-k 8→4 gives 18.5 → 21.8 tok/s at held quality
(GSM8K-50 96%).** Serving now = big-3 NVFP4 + REAP + MTP + **top-k=4** @ 21.8
(`--hf-overrides '{"num_experts_per_tok": 4}'`). Honest single-stream ceiling
≈ 22 (30 needs attention-quant [risky] or batching [ruled out]). Draft patch
reverted, site-pkgs clean; `spark/prepack_nvfp4_linear.py` has a kept shared-scale
fix. Full write-ups below + `spark/MTP-DRAFT-COST.md` (Approach A).

- **Approach A (surgical FULL-cudagraph the MTP draft) — BUILT, confirmed engaged,
  REFUTED.** 3-edit env-gated patch (`VLLM_DRAFT_FULL_CUDAGRAPH=1`); draft's MLA
  attn IS full-cudagraph-safe (no crash), lossless. **Zero speedup:** K=1 18.5,
  K=2 18.6 (= baseline 18.5/19.0). The ~32 ms/depth is the extra **verify**
  position's per-token expert-read bandwidth, NOT the draft forward. Kills
  Approaches A/C/D; Approach E (n-gram) dead-by-inference (same verify tax).
- **`VLLM_USE_BREAKABLE_CUDAGRAPH=1` — also negative + DANGEROUS on NVFP4:** it
  globally disables inductor → NVFP4 numerics break (accept 84.6→31.6%, off-topic
  output) and step time unchanged. Never run breakable on the NVFP4 stack.
- **⇒ New direction: the 30-tok/s lever is the VERIFY side** — cut per-position
  expert-read bytes. Spec decode is capped at ~18.5 until that moves.

### VERIFY-side bandwidth track (session 12 cont.) — top-k WIN, shared-expert marginal

- **★ BANKED WIN: MoE top-k 8→4 = 18.5 → 21.8 tok/s (1.18×), GSM8K-50 96%/94%
  (holds, ≥ the 91% top-8 reference).** `num_experts_per_tok` is a model-config
  value (per-MoE-layer expert count), overridable at serve time with
  `--hf-overrides '{"num_experts_per_tok": 4}'` — NO requant (2-bit planes hold all
  256 experts; top-k just selects how many to read). Scales ~1:1 with bytes →
  confirms bandwidth-bound. Applies to BOTH verify and draft (shared config); the
  win is the verify (76 layers) — draft is 1 layer, negligible. Acceptance
  unchanged (~82%, lossless at greedy). **This is the shipped single-stream best.**
  (top-6 not swept; may be an even safer quality point — untested.)
- **Shared-expert NVFP4 (stacked on top-k4): WORKS + COHERENT but marginal
  (~1.02×, ~22.3 clean) — not worth shipping.** The real achievement is the
  **merged-column NVFP4 loader path now works**: (1) target the MODULE name
  `gate_up_proj` in `VLLM_NVFP4_TARGETS` (NOT checkpoint names `gate_proj,up_proj`
  — the shared expert is a MergedColumnParallelLinear); (2) the prepacker now packs
  gate+up against a SHARED `weight_scale_2` (`spark/prepack_nvfp4_linear.py`,
  `wg_override`) because the W4A16 loader collapses per-shard scale_2 via `.max()`
  and mismatched scales corrupt a half (was 1.07–1.69× apart → likely a real driver
  of the earlier FULL-cut degradation). Coherent on sustained gen (no `</think>`
  loops). But throughput gain is tiny — Marlin FP4 dequant overhead on the small
  shared-expert matrices (12.6 MB each) eats the ~19 MB/layer saving. Overlay:
  `$M/nvfp4_big3_shared_overlay` (built both nodes). **The shared-scale_2 fix is
  the reusable takeaway — it may let the FULL attention/dense cut be revisited at
  coherent quality (differing merged scale_2 was likely part of why it degraded).**
- **Honest single-stream ceiling: ~22 tok/s** (top-k4). 30 needs attention quant
  (decoherence risk) or batching (ruled out — single-stream goal). Byte-cut stack
  can't clear it: top-k4 (1.18×) + shared-expert (1.02×) ≈ 22, not 30.
- **Serving now:** big-3 NVFP4 + REAP + MTP + **top-k=4** @ 21.8 (the banked win).
- **GB10 gotcha (new):** unified memory sits right at the 0.85 `--gpu-memory-util`
  line after a long session (page cache); if boot fails "Free memory ... less than
  desired", drop to `--gpu-memory-utilization 0.83`. Also: `pkill -f 'vllm serve'`
  or `'serve-glm52'` SELF-MATCHES the calling shell (the pattern is in your own
  cmdline) → kill by explicit PID instead.
- Site-pkgs clean (draft patch reverted). `spark/prepack_nvfp4_linear.py` has the
  new shared-scale_2 change (keep it — correct + needed for any merged NVFP4).
  Nothing committed yet this session (docs + prepack changed, awaiting commit).

### NVFP4 backend: Marlin is CORRECT for decode (measured) — do NOT switch

- Prompted by advice that "Marlin FP4 is 2× slower, use cutlass/flashinfer" (true
  for PREFILL/batched, compute-bound). **Head-to-head microbench on real GLM shapes
  (`~/venvs/.../kernels/linear/nvfp4/`), Marlin W4A16 vs Cutlass W4A4:**
  | shape | M=1 (decode) | M=64 |
  |---|---|---|
  | o_proj | **Marlin 1.4× faster** | ~equal |
  | q_b_proj | **Marlin 3.7× faster** | cutlass 1.06× |
  | shared.gate/down | **Marlin ~2× faster** | ~equal |
- **At M=1 (our single-stream decode), Marlin WINS every shape.** Cutlass must
  dynamically fp4-quantize the activation each call (`scaled_fp4_quant` + swizzle) —
  a fixed overhead that dominates at M=1. Marlin skips it (bf16 activations). Cutlass
  only wins at large M (M=256 ≈ 9× a bf16 matmul) — i.e. **batched serving, which we
  ruled out.** The fast non-Marlin kernels are also all **W4A4** (activations→fp4 =
  quality risk); FlashInfer needs sm_10x (unavailable — GB10 is sm_121).
- **⇒ Keep Marlin for decode. Cutlass W4A4 would be slower AND lower-quality here.
  The one thing that flips this is batching (M≳64) — then revisit Cutlass W4A4.**

### ⚠ CORRECTIONS — things I wrongly assumed/remembered (record, session 12)

1. **"GB10 has no native FP4 MMA" — FALSE.** `cutlass_fp4_supported()` returns True
   on sm_121; `cutlass_scaled_fp4_mm`/`scaled_fp4_quant` run. GB10 HAS FP4 tensor
   cores. The conclusion "use Marlin for decode" is still right, but for a DIFFERENT
   reason (M=1 activation-quant overhead, NOT missing hardware). Fixed in
   `spark/NVFP4-DENSE.md`, `spark/RUNBOOK.md`, this file's session-11 GOTCHAS.
2. **"The whole game is T_draft / make the draft forward cheap → depth pays" — FALSE
   (Approach A refuted).** The per-depth cost is the extra VERIFY position's
   expert-read bandwidth, not the draft forward. See `spark/MTP-DRAFT-COST.md`.
3. **"Verify-side NVFP4 = quantize more of the attention" — MISLEADING.** Big-3
   already NVFP4s ~90% of attention bytes; the only remaining dense FP8 is the shared
   expert (+ tiny attention down-projs). "Verify-side NVFP4" ≈ shared expert, not
   attention.
4. **"Shared-expert NVFP4 needs full merged-column loader surgery" — OVERSTATED.** It
   works by (a) targeting the MODULE name `gate_up_proj` (I first wrongly used
   checkpoint names `gate_proj,up_proj` → shape-mismatch crash) and (b) the shared
   `weight_scale_2` prepack fix. The first crash was my wrong target name, not a
   fundamental gap.
5. **`VLLM_TORCH_PROFILER_DIR` — gone in vLLM v0.24.** Profiling is now
   `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'` (gates the
   /start_profile routes). Update `spark/MTP-DRAFT-COST.md` hooks.
6. **`/proc/PID/environ` does NOT show Ray-forwarded env vars** — Ray applies
   `runtime_env` env into `os.environ` AFTER exec, so the exec-time /proc snapshot is
   a false negative. Verify worker env from inside the process (a log line), not /proc.
7. **My ROI estimates ran optimistic:** predicted shared-expert NVFP4 ~1.1× (actual
   ~1.02×), predicted ceiling ~24–26 (actual measured top-k4 ceiling ~22). Treat my
   forward "~1.Nx" estimates as upper bounds until measured.
8. **Unverified claim I made:** "TP replication is only ~8% (down-projs+router)" —
   asserted from reasoning, never measured. If pursued, actually check.

