# HANDOFF — GLM-5.2 on 2× DGX Spark (branch spark-gb10)

Short "resume here" pointer. Deep docs: `spark/NVFP4-DENSE.md` (the shipped
lever, full design/repro), `spark/GOAL.md` (20 tok/s plan), `spark/RUNBOOK.md`
(serve/measure), `spark/handoffs/02-*.md` (how we got to 15),
**`spark/MTP-DRAFT-COST.md` (the hard vLLM problem → 30 tok/s: make the MTP draft
forward cheap; the drafter is PIECEWISE-cudagraphed while the verify is FULL).**

## STATUS (2026-07-17, session 17) — ⚠️ DECODE THROUGHPUT REGRESSION: dspark K=2 now 10.4 tok/s (was 21.8), draft NET-NEGATIVE (below the no-draft floor). Localized to the VERIFY per-position expert-read; hardware/draft-code/config all proven clean. Full diagnostic + next-steps: **`spark/handoffs/04-decode-throughput-regression.md`**. (Machine left with deep C-states disabled — see that doc to revert.)


## STATUS (2026-07-16, session 15) — DSpark self-distill pipeline BUILT + root-caused; blocked on speculators version drift → resume in `spark/dspark-distill/HANDOFF.md`

Goal: squeeze more dspark acceptance (68→~80% pos-0) via self-distillation for
~+2-4 tok/s over the shipped 21.8. Full offline pipeline works (extract server +
2 venv patches, 1988-sample cache on the worker, trainer runs). BLOCKED on one
diagnosed issue: the trainer at speculators HEAD (0.7.0.dev102) is a drifted
forward vs the checkpoint's era (0.5.0.dev38) → step-0 pretrained scores pos-0
0.35 not 0.83. All other hypotheses ruled out (weights/targets/aux/mask/attn).
**NEXT = pin the trainer to ~commit 21033a7 (pre-07-13 forward rewrites), pass
the step-0 gate (pos-0 ≈0.83), then fine-tune + reasoning-data slice.** Full
resume plan, commands, gotchas, gates: **`spark/dspark-distill/HANDOFF.md`**
(+ `PLAN.md`, `FINDINGS.md`). Shipped baseline to beat: dspark K=2 + REAP +
NVFP4 + top-k4 ≈ 21.8 (RUNBOOK §4b), currently serving on :8000.

## STATUS (2026-07-16, session 14) — DSpark 3-way DONE: TIES native MTP; "regression" was COLD PAGE CACHE (session-13 diagnosis revised)

RESUME POINTER. The dspark perf question is **settled — no throughput win, no
loss**. Session 13's "2.3 tok/s regression / 3-4.7s precompute" was a cold-cache
measurement artifact, NOT a per-step cost: serve scripts purge page cache at
boot and planes are mmap'd, so the first ~5×300-tok runs fault 79 GB/rank in
from disk (measured warmup 5.3→7.8→10.8→16.4→20.1 tok/s on the untouched
session-13 server). The 2-step fadvise fix below is therefore MOOT — no code
change needed; `VLLM_MOE_W2_FADVISE_GLOB` only matters at load time.

### 3-way (steady state, ≥8 warm runs, identical fast build: nvfp4_big3_overlay
### + freq-p208 planes + top-k4 + mmap planes + FULL_AND_PIECEWISE, greedy 300 tok)

| config | steady tok/s | tok/verify | acceptance |
|---|---|---|---|
| no draft | ~15.6 | 1.00 | — |
| native MTP K=1 | ~20.4 | 1.77 | 77.2% |
| dspark K=3 | ~20.0 | 2.33 | 68/42/23 %/pos |
| **dspark K=2 (best)** | **~20.9** | 2.14 | 68/45 %/pos |

- **VERDICT: dspark K=2 ≈ native MTP K=1 (tie within ±1 tok/s).** Port validated
  and competitive; not worth switching (7 GB extra draft on both nodes, ~0 gain).
  dspark K=1 skipped — dominated (1.68 tok/verify < MTP 1.77, more draft cost).
- **Session-12's 21.8 did NOT reproduce** on freq-p208 planes (MTP acceptance
  77.2% here vs 84.6% on p208_reap) — plane choice moves acceptance more than
  drafter choice. If chasing the last tok/s, re-run the 3-way on p208_reap.
### Session 14b — acceptance ablation: the drafter was NEVER degraded; new best = dspark K=2 + REAP ≈ 21.8

Walked back every target mod to "recover" dspark acceptance (68% vs session-13's
83-100% on FP8). ALL NEGATIVE — pos-0 acceptance across fast build / REAP swap /
native top-k8 / no-NVFP4 / full-unpruned-planes = 68/69/70/68/67%. Each lever
costs ≤1pt. **Session-13's 83-100% was eager + short trivia prompts; ~68% pos-0
is this speculator's TRUE acceptance on 300-tok reasoning content.** Details +
full tables: `spark/handoffs/03-dspark-acceptance-ablation.md`.

- **REAP planes DID help both drafters' throughput (+0.5-1, plane I/O quality)
  and MTP's acceptance (77.2→79.2%) but not dspark's** (hidden-state-conditioned
  MTP head tracks target quality; the external draft is indifferent).
- **BEST MEASURED CONFIG (left serving on :8000, `~/serve-dspark-best.log`):
  dspark K=2 + p208_reap + NVFP4 big-3 + top-k4 ≈ 21.8 tok/s** (MTP K=1 same
  stack ~21 — still within noise; no-draft floor 15.6; top-k8 ~18; unpruned
  planes thrash at 2.3, don't fit page cache).
- Session-12's 84.6% MTP acceptance / 21.8 also didn't reproduce exactly
  (79.2% / ~21 today, same prompt-set caveat) — treat cross-session acceptance
  numbers as content-bound, only compare within one prompt set.
- Gotcha: one boot died `CUBLAS_STATUS_INTERNAL_ERROR` during PIECEWISE capture
  + Ray ActorHandle corruption → clean `spark/start-ray-cluster.sh` fixed it.
- Results log: `~/Dev/howtospark/models/glm-5.2.md` (session-14a/b entries).
- NEXT (optional, in value order): (1) draft self-distill against the served
  target on representative content — the ONLY remaining acceptance lever
  (~+2-4 tok/s at K=2-3 if pos rates hit ~80/60); (2) harness-benchmark the
  best config for the site (informal 300-tok numbers ≠ `bench/harness.py` rows);
  (3) GSM8K-50 sanity on dspark K=2 + REAP before calling it the ship config.

## STATUS (2026-07-16, session 13) — DSpark EXTERNAL speculator ported + working; fast-build PERF blocked (diagnosed) — **DIAGNOSIS REVISED IN SESSION 14 ABOVE**

RESUME POINTER for the dspark-speculator effort. One-line: `DSparkProposer` port is
DONE and generates correctly on 2 Sparks; on the **fast build** it currently
REGRESSES throughput (context-KV precompute is 3-4s/step), root-caused to two
fixable causes. Branch `spark-gb10`; venv `~/venvs/vllm-moet` (0.24.0); port files
in `dspark-port/` (apply.sh re-installs into a venv — RUN ON BOTH NODES).

### DONE (verified)
- **Ported `dspark` speculative decoding** (external RedHatAI/GLM-5.2-speculator.dspark,
  a `speculators`-format DSparkDraftModel, verifier zai-org/GLM-5.2-FP8) into the fork:
  - `transformers_utils/configs/speculators/algos.py`: `register_speculator("dspark")`
    (emits arch `Qwen3DSparkModel`, `dspark_bonus_anchor=True`, `dflash_config.mask_token_id`,
    aux/target layers).
  - `config/speculative.py`: `DSparkModelTypes`, method dispatch, `use_dspark()`,
    aux-hidden-states + parallel_drafting for dspark.
  - `model_executor/models/{registry.py,qwen3_dspark.py}`: `Qwen3DSparkModel`
    (DFlashQwen3 backbone + Markov head; from the 0.25.2 reference).
  - `v1/spec_decode/dspark.py`: **`DSparkProposer(DFlashProposer)`** — the real work.
    Overrides `_create_draft_vllm_config` (draft KV → "auto", DSA target's fp8_ds_mla
    has no non-causal backend), `set_inputs_first_pass` (store bonus token),
    `_sample_draft_tokens` (GREEDY sequential Markov; probabilistic/gumbel = TODO).
    Has env-gated `VLLM_DSPARK_PROFILE` timing (not forwarded to Ray workers — hardcode
    `_DSPARK_PROFILE=True` to re-profile; worker logs in `/tmp/ray/session_latest/logs/`).
  - `v1/spec_decode/llm_base_proposer.py`: `model_returns_tuple()` add "dspark";
    combine-hidden-states branch add "dspark".
  - `v1/worker/gpu_model_runner.py`: import + `elif use_dspark(): DSparkProposer(...)`.
  - `v1/spec_decode/dflash.py`: `__init__` assert accepts "dspark".
- **WORKS on GLM-5.2-FP8** (2 Sparks, eager): coherent gen, **83-100% acceptance**,
  mean accept length 3.5-4.0. Serve: `serve-glm52-tp2-dspark.sh --eager`.
- **Fast build (NVFP4 overlay + p208 REAP + top-k4) + dspark FULL graphs**: loads &
  generates, moderate acceptance (mean ~3, avg 60-70% — draft trained on full FP8 so
  it diverges from the pruned/NVFP4 target). BUT throughput **~2.3 tok/s vs 21.8
  baseline** = a REGRESSION, not a win.
- **Profiled the regression** (decisive): per draft step —
  `precompute_and_store_context_kv` **3-4.7s** (99.5%), draft forward **10ms**, Markov
  **5ms**. Recompilation ruled out. So it's ONE slow method, NOT verify bandwidth,
  NOT my code. **Refutes the MTP-DRAFT-COST bandwidth worry for this case.**
- **Confirmed cause #1 (~half):** per-step `VLLM_MOE_W2_FADVISE_GLOB` over
  `$MODEL/*.safetensors` — disabling it (`VLLM_MOE_W2_FADVISE_GLOB=`) dropped precompute
  to ~1-2s. (serve script now honors an override.)

### NEXT (the 2-step fix to try for a 25 tok/s answer)
1. **Kill per-step fadvise:** find where the W2 plane machinery calls fadvise; gate it
   to load-time / target-only so it never fires on the drafter's context-KV precompute.
2. **Kill the ~1-2s residual:** nsys/torch-profile a single `precompute_and_store_context_kv`
   (runs EAGER, cudagraph=NONE). Prime suspect: the eager **NVFP4 dense hook on the
   DRAFT's** `o_proj/q_b_proj/kv_b_proj` (`VLLM_NVFP4_TARGETS` matches the draft's own
   layer names) — try excluding the draft from NVFP4, or not applying NVFP4 to the draft.
   Variance (953-1983ms) still smells I/O/contention.
   Goal: precompute → ~ms → a spec step ≈ one verify forward → with mean-accept ~2-3,
   plausibly BEATS 21.8. If it can't get under native-MTP, report that honestly.
3. Then benchmark 3-way (dspark vs native MTP vs no-draft) + publish to ~/Dev/howtospark.

### HOW TO RESUME
- Re-apply port to a fresh venv (BOTH NODES): `dspark-port/apply.sh <vllm-site-packages>`
  then rsync each edited file to `192.168.100.2:<same path>`.
- Cheap sanity: `~/venvs/vllm-moet/bin/python -c "from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig as C; print(C.from_pretrained('RedHatAI/GLM-5.2-speculator.dspark').architectures)"` → `['Qwen3DSparkModel']`.
- Serve FP8 dspark: `bash spark/serve-glm52-tp2-dspark.sh --eager` (endpoint :8000, model `glm-5.2`).
- Serve FAST build dspark (FULL graphs): `MODEL=$HOME/models/hf/GLM-5.2-FP8/nvfp4_big3_overlay
  VLLM_MOE_W2_PREPACKED_DIR=$HOME/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208
  VLLM_NVFP4_DENSE=1 VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj VLLM_ENGINE_READY_TIMEOUT_S=2400
  bash spark/serve-glm52-tp2-dspark.sh --hf-overrides '{"num_experts_per_tok":4}'`
  (add `--eager` to skip graphs / for fast boot; add `VLLM_MOE_W2_FADVISE_GLOB=` to halve precompute).
- Ray corrupts across boots — if "ActorHandle across sessions", `ray stop` both nodes + `bash spark/start-ray-cluster.sh`.
- Speculator lives on BOTH nodes: `~/models/hf/GLM-5.2-speculator.dspark` (7GB).

### GOTCHAS / KEY FACTS
- **PATCH BOTH NODES' venvs** — Ray runs workers on head + worker, each has its own
  site-packages. A head-only patch fails on the worker ("Unknown method dspark").
- venv is NOT editable-installed; the repo `vllm/` tree lacks this code. `dspark-port/` is the tracked copy.
- The 0.24 fork uses `v1/spec_decode/` **Proposer** classes; the 0.25.2 reference uses
  `v1/worker/gpu/spec_decode/` **Speculator** classes — DIFFERENT frameworks. The port
  targets the Proposer one; the `gpu/spec_decode/dspark/*` files I copied first are DEAD (ignore).
- `kill`/`pkill -f 'vllm serve'` self-matches the shell (exit 144). Use `pgrep -f '[b]in/vllm serve'`.
- Foreground `sleep` is blocked in the Claude harness (exit 144).

### CORRECTNESS GATES
- Coherent output on FP8 (got "Paris"/"2+2=4"/reasoning); SpecDecoding metrics show
  acceptance >0 (drafts being accepted). Bad = empty/garbage output or accept ~0.
- A throughput WIN means fast-build dspark tok/s > native-MTP (21.8). Currently 2.3 (fail).

### LINKS
- Port: `dspark-port/` (new/, orig/, apply.sh). Serve: `spark/serve-glm52-tp2-dspark.sh`.
- Speculator: https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark
- dspark-capable ref image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2, has dspark+GlmMoeDsa).
- Site benchmarks repo: ~/Dev/howtospark (models/glm-5.2.md). IQ1_S llama.cpp 2-Spark
  GLM-5.2 benchmark (separate track, DONE+published): howtospark commit `1ab303c`.

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

## STATUS (2026-07-14, session 11)

**REAP (saliency expert-prune) shipped + coherence-validated for GLM-5.2.** Also
uncovered two serving-quality problems in the shipped speed stack (below).

- **Server UP now:** **big-3 NVFP4 + REAP planes + MTP** (`MODEL=$M/nvfp4_big3_overlay`,
  `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj`, `moe_w2_planes_tp2_p208_reap`,
  MTP on), TP2, healthy on :8000. Verified **fully coherent + correct** on chat:
  320-tok transformer explanation clean at greedy, "all but 9 sheep" →9, GSM
  Natalia. **REAP does not break the model, on the config you actually ship.**
- **✅ RESOLVED — the "NVFP4 incoherent" scare was the FULL cut, not big-3.**
  The **full** `nvfp4_dense_overlay` cut (all 10 targets incl. dense MLP
  `gate/up/down_proj`) degrades on sustained generation — coherent for a while
  then token-repetition / "</think>" loops (greedy fast, sampled ~250 tok then
  collapses). Reproduces on frequency AND REAP planes, nodes byte-identical
  (not a desync) → it's the aggressive quant (rel-L1 ~0.09), not the prune, not
  MTP. The **big-3 cut** (`o_proj,q_b_proj,kv_b_proj`, ~18 tok/s, the config
  actually demoed) is fully coherent. **Do not ship the full cut** without a
  real quant fix + eval; big-3 is the safe NVFP4 config. FP8 (no NVFP4) is also
  clean.
- **MTP note:** the "Paris1 and1..." garbage-draft interleave appeared only on
  the FULL-cut degraded target; on big-3 MTP is clean. So MTP is fine — its
  earlier garbage was downstream of the full-cut target collapse, not an MTP bug.
- **HF model updated (session 11):** `sapidlabs/GLM-5.2-NVFP4-attn-experimental`
  now ships the **big-3 stable** cut (4 shards `nvfp4-dense-000{0..3}`; deleted
  the 6 full-cut shards; corrected card: title, `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj`,
  coherence-validated + task-battery-pending). Commit `eb2ae11`.
- **MTP drafter — MEASURED (session 11), verdict: drafter is near-optimal, both
  levers small.** Big-3+REAP build, warmed (planes resident), greedy, same prompt:
  | K | tok/s | tok/verify | acceptance |
  |---|------|-----------|-----------|
  | 1 | 18.5 | 1.85 | 84.6% (sampled 83.8%) |
  | 2 | 19.0 | 2.52 | pos0 85.3% / pos1 66.4% |
  - **Depth (K≥2) is a DEAD END here (~1.03×):** K=2 drafts 1.36× more tok/verify
    but the extra MTP-head forward — a FULL decoder layer (layer 78: eh_proj +
    enorm/hnorm + full attn + 256-expert MoE + shared_head, ~9.7B params, run
    serially/autoregressively with its own TP2 all-reduce) — eats the gain.
  - **Accuracy fine-tune (K=1) ≈ +4% only:** 84%→~92% accept caps at 1.92/1.85.
    Not worth training a 9.7B MoE layer for. (MTP already gives ~1.62× over the
    ~11.4 non-MTP floor; the ~12% gap to the 1.84× ceiling is MTP-forward overhead,
    a serving/kernel cost, not a drafter-accuracy cost.)
  - **Recommendation: do NOT build the drafter trainer** — ROI too low.
  - **TP-free draft TESTED — negative result.** `draft_tensor_parallel_size=1`
    (added `DRAFT_TP` env knob to `serve-glm52-tp2-mtp.sh`; it's a supported vLLM
    config, no surgery) boots + is coherent + same 84.6% acceptance, but runs
    **~17.6 tok/s (~5% SLOWER** than TP2-draft's 18.5). So the ~12 ms/draft
    overhead is NOT the TP all-reduce — splitting the draft across both GPUs
    (TP2) beats running it on one + syncing. The overhead is launch/orchestration
    (Python spec-decode loop, sample/accept/reject, KV bookkeeping), which neither
    TP-free nor expert-count touches (MoE reads top-k regardless of pool). **The
    cheap "cheaper-forward" levers are exhausted.** Reducing MTP overhead further
    = deep vLLM spec-decode work (cudagraph the whole draft+verify loop), low ROI.
  - **Net: MTP at K=1 (18.5 tok/s, 84% accept) is near-optimal for this setup.**
    Bigger throughput levers live elsewhere (NVFP4 breadth capped at big-3 by
    quality; the non-MTP floor). Shipped/best MTP config = plain K=1 (no DRAFT_TP).
- **Code pushed:** `Sapid-Labs/vLLM-Moet` `spark-gb10`. REAP tooling:
  `~/Dev/reap` `add-glm_moe_dsa-support` (commits `02b838a`,`7f9f567`,`fd2b7f4`).
- **NEXT:** (a) REAP-vs-frequency quality A/B on the clean FP8 stack (GSM8K-50) —
  does REAP actually buy back quality; (b) root-cause NVFP4 collapse + MTP garbage
  (both block the 20 tok/s config); then drafter → full battery.

### (prev) STATUS session 10 — NVFP4 ~20 tok/s
Full NVFP4 attn+shared reached ~20 tok/s greedy (throughput only; see the NVFP4
quality caveat above). `sapidlabs/GLM-5.2-NVFP4-attn-experimental` pushed.

## NEXT (agreed plan — order matters)

**REAP → drafter → eval**, with cheap sanity checks between (full battery only at
the end; it's slow). Rationale: the drafter is *fit to* the target's output
distribution, and REAP changes which experts are active → do all target-changing
steps first, fit the drafter last against the frozen model.

1. **REAP (do first)** — replace the frequency-based expert prune (drop 48
   coldest by traffic) with saliency-based REAP at the same 208 experts, to buy
   back quality. Quality move, NOT speed (same bytes/token).
   - **Pipeline (confirmed session 11):** REAP observer → per-layer expert
     saliency → top-208-per-layer keep-list in `spark/routing/keep208.json`
     format → `spark/routing/prune_planes.py <src> <dst> <keep.json>` (row-select,
     no requant, ~15 min) on the FULL `moe_w2_planes_tp2` (still on disk, 97 GB)
     → serve with the new p208 dir. `spark/prepack_planes.py` packs all experts;
     it is `prune_planes.py` that applies the keep-list. Run prune_planes on BOTH
     nodes' tp2 planes (same keep.json; rank-agnostic row-select).
   - **CODE DONE (session 11, reap `add-glm_moe_dsa-support` @ 02b838a):** GLM-5.2
     REAP support is complete. Observer/prune registries were already there; the
     missing piece was the **disk-streaming FP8 path** — the 357B model is >> RAM
     so calibration must stream layer-by-layer from disk, but `disk_stream` had no
     GLM converter and no FP8 dequant (it cast raw float8 bytes, ignoring
     `weight_scale_inv`). Added block-wise FP8 dequant + `glm_converter` (fuses
     per-expert gate/up/down into native batched params); 4 GLM smoke tests pass
     (incl. a synthetic real-layout FP8 checkpoint matched to bf16 ref, rel-L1
     <5%); hy3 tests still green. Env: run reap with **`~/venvs/hf/bin/python`
     `PYTHONPATH=src`** (reap not pip-installed; that venv has pytest+transformers
     w/ GlmMoeDsa).
   - **VALIDATED ON REAL MODEL (session 11):** the streaming observer runs
     end-to-end on GLM-5.2-FP8 (all 78 blocks: data→FP8 dequant→DSA forward→
     saliency). Server was torn down (both nodes, GPUs freed 0.0/2.0). Reap deps
     (`accelerate datasets scikit-learn matplotlib seaborn`) installed into the
     **CUDA `vllm-moet` venv** on BOTH nodes (the `hf` venv is CPU-only torch —
     do NOT use it for the real run). Peer synced: reap `src/scripts/tests`
     rsync'd to `spark-c84b:~/Dev/reap` + deps installed + imports verified.
     Extra reap fixes this session (commits `02b838a`, `7f9f567`, `fd2b7f4`):
     FP8 dequant+glm_converter; DP `data_shard_index/count`+`save_raw_state`;
     **DSA cross-layer top-k threading** (GLM "shared" attn layers crash the
     isolated block-replay without it). Disk-bound: each MoE block re-reads
     ~19 GB (dequant), so a full pass is multi-hour; DP halves the compute part.
   - **RUN THE OBSERVER (both nodes, DP; from `~/Dev/reap`, env
     `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src`, python `~/venvs/vllm-moet/bin/python`):**
     ```
     python -m reap.layerwise_prune \
       --model_name ~/models/hf/GLM-5.2-FP8 \
       --dataset_name theblackcat102/evol-codealpaca-v1 \
       --run_observer_only true --disk_stream true \
       --batches_per_category <N> --batch_size <B> --model_max_length 2048 \
       --seed 42 --output_file_name reap.pt \
       --data_shard_count 2 --data_shard_index <0 on .1 / 1 on .2>
     ```
     Output: `artifacts/GLM-5.2-FP8/evol-codealpaca-v1/layerwise/reap.shard<i>.raw.pt`
     (RAW state, trackers intact — needed for the merge).
   - **MERGE → KEEP-LIST → PRUNE PLANES:**
     ```
     python scripts/merge_observer_states.py \
       reap.shard0.raw.pt <copied-from-peer>reap.shard1.raw.pt --out reap.merged.pt
     python ~/Dev/vLLM-Moet/spark/routing/reap_keep_list.py \
       reap.merged.pt ~/Dev/vLLM-Moet/spark/routing/keep208_reap.json \
       --keep 208 --compare ~/Dev/vLLM-Moet/spark/routing/keep208.json
     # on BOTH nodes (rank-agnostic row-select, ~15 min):
     python ~/Dev/vLLM-Moet/spark/routing/prune_planes.py \
       ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2 \
       ~/models/hf/GLM-5.2-FP8/moe_w2_planes_tp2_p208_reap \
       ~/Dev/vLLM-Moet/spark/routing/keep208_reap.json
     ```
     Then serve with `VLLM_MOE_W2_PREPACKED_DIR=...moe_w2_planes_tp2_p208_reap`
     (else identical to the shipped serve cmd). Sanity-check: ~10-20 prompts
     coherence or GSM8K-50.
   - **RUN IN PROGRESS (session 11):** DP calibration launched both nodes with
     `batches_per_category 16 --batch_size 4 --model_max_length 1024 --seed 42`
     (→ 8 batches/node, ~65k tokens total), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
     ~2.2s/it, ~28s/MoE block → ~40-50 min/node. Writes `reap.shard0.raw.pt`
     (this node) / `reap.shard1.raw.pt` (peer).
   - **OOM GOTCHA:** `batch_size 8 --model_max_length 1024` OOMs the GB10 (~119 GB
     peak on the first MoE block: 256 experts dequant'd to bf16 ~19 GB + per-token
     MoE/attn intermediates over 8192 tokens). `bs4 seq1024` peaks ~55 GB — safe.
     If you push tokens, add batches (num_batches only affects time, not peak),
     don't raise batch_size/seq.
   - **DONE (session 11):** DP calibration completed both nodes (8 batches each,
     16 total). Merged → `spark/routing/keep208_reap.json` (top-208/layer;
     ~37 experts/layer differ from the frequency prune; merged coverage mean
     252/256 active). `prune_planes.py` ran on BOTH nodes →
     `moe_w2_planes_tp2_p208_reap` (79 GB, 75×208, present on .1 and .2).
     Observer artifacts: `~/Dev/reap/artifacts/GLM-5.2-FP8/evol-codealpaca-v1/
     layerwise/reap.{merged,shard0.raw,shard1.raw}.pt`.
   - **PEER checkout was stale** (missing `spark/routing/`) — copied
     `prune_planes.py`+`keep208_reap.json` there manually. If re-running on the
     peer, verify `~/Dev/vLLM-Moet/spark/routing/` exists first.
   - **SERVING the REAP build:** same NVFP4 serve cmd but
     `VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208_reap`. Ray must be up
     first (`bash spark/start-ray-cluster.sh`; the serve driver forwards the
     VLLM_MOE_W2_* env to workers — not baked in raylet). Sanity-check: greedy
     coherence on ~5-10 prompts, then GSM8K-50.
2. **Drafter (second, last model change)** — fine-tune the MTP head (layer 78)
   against the frozen NVFP4+REAP target to raise MTP acceptance (the real lever
   for effective throughput; sampled tok/s currently varies ~13-20 with
   acceptance). Bigger lift than REAP. Sanity-check after.
3. **Full quality battery (end)** — GPQA + GSM8K + IFEval + MMLU-Pro on the final
   build. This is the gate for any "same quality" public claim. Run it ALONE
   (never co-run — session-9 crash) and resumable (`--use_cache`), see below.

## HOW TO RESUME

**Serve the shipped NVFP4 build (2× Spark, TP2):**
```bash
cd ~/Dev/vLLM-Moet; M=~/models/hf/GLM-5.2-FP8
MODEL=$M/nvfp4_dense_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208 \
VLLM_NVFP4_DENSE=1 \
VLLM_NVFP4_TARGETS="o_proj,q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,fused_qkv_a_proj,gate_proj,up_proj,gate_up_proj,down_proj" \
MTP_K=1 VLLM_ENGINE_READY_TIMEOUT_S=2400 \
nohup bash spark/serve-glm52-tp2-mtp.sh > ~/serve-nvfp4.log 2>&1 &
```
Big-3-only cut = same but `VLLM_NVFP4_TARGETS=o_proj,q_b_proj,kv_b_proj` +
`MODEL=$M/nvfp4_big3_overlay`. Plain FP8-attn baseline = drop `VLLM_NVFP4_DENSE`
+ `MODEL=$M` + `VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208`.

**Measure decode:** use **greedy (temperature 0)** for a clean number (~20);
sampled (temp>0) varies with MTP acceptance. Settle 3-5 warmups first (Marlin
FP4 kernels warm slowly). One-liner protocol in session-10 transcript / GOAL.md.

**Rebuild an overlay** (e.g. after REAP changes nothing here, but for new quant):
`python spark/prepack_nvfp4_linear.py --targets <basenames> --out <dir>` on BOTH
nodes (deterministic; peer needs the packer copied — it's at `~/prepack_nvfp4_linear.py`).

## GOTCHAS / KEY FACTS (hard-won this session)

- **NVFP4 code lives in site-packages** (`~/venvs/vllm-moet/.../vllm/`), captured
  in `patch/nvfp4-dense.patch` + `spark/nvfp4_dense_hook.py` (Dockerfile applies
  both). A fresh vllm install needs them applied. The PEER node's site-packages
  still has leftover debug prints (`NVFP4HOOKDBG`) — harmless, but re-sync the
  cleaned files if you rebuild: hook, parameter.py, deepseek_v2.py, deepseek_mtp.py.
- **Why the overlay works:** vLLM globs ALL *.safetensors (not just the index),
  so the overlay's NVFP4 tensor AND the symlinked original's fp8 tensor both load.
  Handled by `NVFP4SKIP2` guard (skip fp8→uint8-param) + `NVFP4ORPHAN` guards
  (skip fp8 `weight_scale_inv` with no nvfp4 param) in BOTH main + MTP loaders,
  stacked + non-stacked paths. Don't remove these.
- **GPU release between serves:** `kill -9` on the APIServer ORPHANS the Ray
  EngineCore + RayWorkerProc (one holds ~79 GB), which squat the GPUs → next boot
  fails "Cannot provide a placement group requiring 2.0 GPUs". Kill
  `EngineCore|RayWorkerProc|vllm serve` by PID on BOTH nodes, then `ray status`
  should show `0.0/2.0 GPU`.
- **Ray corrupts across many failed boots** (`ActorHandle ... across Ray
  sessions`) → `bash spark/start-ray-cluster.sh` (resolves RoCE GIDs, restarts
  clean).
- **Fresh NVFP4 compile > 600s** default engine-ready timeout → set
  `VLLM_ENGINE_READY_TIMEOUT_S=2400` (cached after first boot; ~4 min warm).
- **[CORRECTED session 12 — see CORRECTIONS above] GB10 DOES have FP4 tensor cores
  (`cutlass_fp4_supported()`=True, sm_121).** We use weight-only Marlin FP4 (4-bit read, bf16
  compute). Correct for bandwidth-bound decode. Adds a little dequant latency
  (measured 1.2× vs predicted 1.29× for big-3).
- **Never co-run evals** (session-9: IFEval co-ran with GPQA 5.5h → aiohttp
  "Session is closed" crash). Run alone, `--use_cache <dir>` + `--log_samples`
  (resumable), `max_length=8192` matching the server + `max_gen_toks=5000`
  (avoids the 8192-context overflow 400s), `num_concurrent=4`.

## CORRECTNESS GATES

- NVFP4 serve: coherent greedy output (verified — it explains memory-bound
  inference correctly) + greedy decode ≥ ~1.3× the FP8-attn baseline.
- After REAP / drafter: cheap sanity (coherence / GSM8K-50) each; full battery
  within noise of the 15 tok/s baseline (GSM8K ≥ ~90%) at the end.

## LINKS

- `spark/NVFP4-DENSE.md`, `spark/GOAL.md`, `spark/RUNBOOK.md`
- `spark/prepack_nvfp4_linear.py`, `spark/nvfp4_dense_hook.py`, `patch/nvfp4-dense.patch`
- HF: `sapidlabs/GLM-5.2-NVFP4-attn-experimental`,
  `sapidlabs/GLM-5.2-2bit-MoE-planes-pruned208-tp2` (2-bit planes)
- REAP tooling: `~/Dev/reap` (branch `add-glm_moe_dsa-support`)
- Eval harness: `~/Dev/howtospark/evals/` (run_battery.sh + run_eval.py)
