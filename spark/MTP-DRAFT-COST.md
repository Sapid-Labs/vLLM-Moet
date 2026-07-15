# MTP-DRAFT-COST — the "hard vLLM problem": cheap draft forward → 30 tok/s

Deep design/handoff doc. The short pointer is in the repo-root `HANDOFF.md`;
this is the "resume here" for the single problem blocking **30 tok/s** on
GLM-5.2 / 2× DGX Spark: **the MTP draft forward is too expensive relative to the
verify.** Written 2026-07-14 (session 11).

## STATUS

Phase: **Approach A (cheapen the draft forward) FULLY REFUTED — draft cost is a
dead end. The 30-tok/s lever moved to the VERIFY side (per-position expert-read
bandwidth).** Branch `spark-gb10`. Serving today = big-3 NVFP4 + REAP planes + MTP
K=1 = **18.5 tok/s** (warmed, greedy). Baseline re-confirmed + restored session 12.

### SESSION 12 (2026-07-14) — findings, don't re-derive

- **Root cause pinned in code (better than session-11's guess):** the drafter is
  **never wrapped in a cudagraph wrapper at all** — not just "denied FULL." With
  `VLLM_USE_BREAKABLE_CUDAGRAPH` off (default), `gpu_model_runner.py:~5874` wraps
  ONLY the main model in `CUDAGraphWrapper(FULL)`; the drafter runs its
  torch-compiled **piecewise** graphs with host relaunch at every attn/MoE
  boundary. The `initialize_cudagraph_keys` clamp (`llm_base_proposer.py:405`,
  MTP path = `SpecDecodeBaseProposer`) is a second, separate cap.
- **`VLLM_USE_BREAKABLE_CUDAGRAPH=1` (the cheap, no-code form of Approach A):
  NEGATIVE — ruled out.** It boots on both nodes (env auto-carries to the peer
  via Ray driver-env copy). But: (1) it **globally disables inductor**
  ("disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none") —
  swaps fused kernels for eager-under-one-graph for BOTH main and draft. (2)
  **Step time UNCHANGED:** K=1 breakable = **13.0 tok/s**, accept **31.6%**,
  tokens/step 1.32 → step ≈ **101 ms** vs baseline 18.5 tok/s / 84.6% / 1.85 /
  **100 ms**. So removing the draft's piecewise host-gaps via single-graph
  capture bought **~0 ms of step time** (the eager-op penalty offset it, OR the
  gaps weren't the cost). (3) **NVFP4 ⊥ inductor-off:** acceptance collapsed
  84.6→31.6% and output went off-topic — the NVFP4 path depends on the inductor
  fusions (norm_quant/act_quant/nvfp4 dequant) for correct numerics. **Never run
  breakable on the NVFP4 stack.**
- **Torch profiler is too self-distorting to isolate the 33 ms here.** Enabled it
  via `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'` (NOT the
  old `VLLM_TORCH_PROFILER_DIR` env — gone in v0.24; routes gate on
  `ProfilerConfig`). Trace shows compute-stream **duty ~32%, ~2 s idle in ~146
  large (>100 µs) host gaps** — heavily host-bound — BUT profiling itself drops
  serving to ~10 tok/s, inflating exactly those gaps, so the draft-vs-verify
  sub-split isn't trustworthy from it. The `execute_context_0(0)_generation_1(2)`
  / `execute_context_1(20)_generation_0(0)` annotations are nested/overlapping;
  don't try to draft/verify-classify by them.
- **★ Approach A (surgical FULL-cudagraph draft) BUILT, CONFIRMED ENGAGED, and
  REFUTED. This is the headline result — Approach A is DEAD; do not revisit.**
  Env-gated patch `VLLM_DRAFT_FULL_CUDAGRAPH=1`, 3 edits (all reverted after,
  backups in scratchpad; diff below to reproduce):
  1. `gpu_model_runner.py:~5878` (non-breakable FULL branch) — also wrap
     `drafter.model` in `CUDAGraphWrapper(FULL)`.
  2. `gpu_model_runner.py:~2539` — skip the `spec_decode_common_attn_metadata
     .unpadded(...)` call (keep PADDED metadata) so the draft has static shapes
     for FULL capture. (The unpad is why vLLM's comment says the drafter "only
     uses piecewise cudagraphs".)
  3. `llm_base_proposer.py:405` — clamp → `CUDAGraphMode.FULL`.
  - **It did NOT crash** — the draft's MLA decode attention IS full-cudagraph-safe
    (same kernel the verify FULL-captures). Boots, coherent, LOSSLESS (accept
    85.7% = baseline). Draft graph captures lazily on 1st decode. Both patch
    halves logged as engaged on the worker (had to add a debug log to prove it —
    the flag reaches workers via Ray runtime_env, INVISIBLE in `/proc/PID/environ`
    which is an exec-time snapshot; don't check propagation that way).
  - **RESULT — zero speedup at either depth:**
    | config | K=1 | K=2 |
    |---|---|---|
    | baseline (piecewise draft) | 18.5 | 19.0 |
    | FULL-draft (confirmed on)  | 18.5 | 18.6 |
  - **Why (the reframing that kills A/C/D and probably E):** the ~32 ms marginal
    per-depth cost (Δstep K1→K2) is UNCHANGED whether the draft forward is
    piecewise or one FULL graph → **the draft forward was never the cost.** Each
    extra spec position makes the VERIFY process one more position, and the verify
    is **bandwidth-bound on per-position expert-weight reads** (MoE reads top-k
    *per token*; K+1 positions ≈ (K+1)× expert reads). THAT extra read pass is the
    ~32 ms. No draft-side change (cheaper forward, FULL graph, TP-free, fewer
    experts — Approaches A/C/D) can touch it. **Approach E (n-gram, T_draft≈0)
    likely also fails**: it still adds verify positions, so it pays the same
    per-position expert-read tax. Only its cost is on the DRAFT side (0), but the
    depth tax is on the VERIFY side — untouched.
  - **⇒ The 30-tok/s lever is NOT speculative decoding at all — it's cutting the
    verify's per-position expert-read bandwidth** (lower MoE top-k, or a
    faster/narrower expert read, or the NVFP4 verify-side track). See the PARALLEL
    TRACK section — it is now the MAIN track, not a side one.

### (session 11) original framing
Goal = **30 tok/s** (usability bar). Need ~1.62×, and it will not come
from the drafter's *accuracy* or from adding K — only from making the draft
*forward* cheap enough that depth pays.

## THE GOAL AND WHY WE'RE STUCK (the quantitative model)

A spec-decode step = run the draft head K times (serial), then one full-model
verify over the K+1 positions. Solved from measured K=1 vs K=2 tok/s:

    time_per_step = T_verify + K · T_draft
    tokens_per_step = 1 + Σ accept_i        (accept_0=0.85, accept_1=0.66, decays)

    T_verify ≈ 67 ms   (full 78-layer forward, FULL cudagraph, bandwidth-bound)
    T_draft  ≈ 33 ms   (ONE layer, PIECEWISE cudagraph — ~35× its compute share)

**c = T_draft / T_verify ≈ 0.5.** Speculative decoding only wins when
`c ≪ 1`. At c≈0.5 each extra draft step costs ~half a verify, so with acceptance
decaying by depth the marginal token rate of draft #2 (~20 tok/s) barely exceeds
the running average (18.5) → **K=2 is a wash (measured 1.03×), K=3 negative.**

**What must become true for 30 tok/s:** drive **T_draft from ~33 ms to <~10 ms**
(c → <0.15). Then deep K + decent multi-step acceptance reaches it, e.g.:
- T_draft 8 ms, K=3, accept 0.85/0.66/0.50 → tokens 3.01 / (67+24) ms = **33 tok/s**.
- Plus a cheaper verify (better NVFP4, see parallel track) → 36–38 tok/s.

The whole game is **T_draft**. Everything else (accuracy fine-tune, K, tree
verify) is downstream of it and pointless until it's cheap.

> **⚠ SESSION-12 CORRECTION — this whole model is WRONG.** The "make T_draft cheap
> and depth pays" thesis was falsified: FULL-cudagraphing the draft forward (the
> exact fix below) changed the K1→K2 marginal by ~0. The ~32 ms/depth is the extra
> VERIFY position's per-token expert-read bandwidth, not the draft forward. So
> `c = T_draft/T_verify` is NOT the right quantity — the draft forward is ~free;
> the depth tax lives on the verify side. See the session-12 STATUS block up top.
> Read the section below only as the (refuted) original hypothesis.

## WHY T_draft is 33 ms (root cause — the hard part) — ⚠ REFUTED, see correction above

The draft is **one layer** (layer 78: MLA attn + 256-expert MoE + shared_head).
Its compute share of the 67 ms verify is ~0.9 ms. It costs **33 ms**. So ~32 ms
is **fixed per-invocation overhead**, and the specific culprits:

1. **The proposer only gets PIECEWISE cudagraphs; the verify gets FULL.**
   `vllm/v1/spec_decode/extract_hidden_states.py:~245` hard-codes
   `proposer_cudagraph_mode = PIECEWISE` ("Only supports PIECEWISE cudagraphs").
   PIECEWISE splits the forward at attention/MoE boundaries (`splitting_ops`),
   runs attention eager, and leaves **host-side gaps + relaunch at every
   boundary**. For a 1-layer draft invoked once per token, that host overhead is
   the dominant cost. FULL cudagraph (one shot, no host gaps) is what the verify
   enjoys and the draft is denied.
2. **Spec-decode host orchestration per step** — the Python propose → sample →
   accept/reject → KV-bookkeeping loop, not fully overlapped with the GPU.
3. **TP2 all-reduce** in the draft's o_proj/down_proj — real but NOT dominant
   (proven: TP-free draft was *slower*, see DONE).

## DONE (measured / ruled out this session — don't re-derive)

- **Cost decomposition:** T_verify≈67 ms, T_draft≈33 ms, c≈0.5 (from K=1 18.5 &
  K=2 19.0 tok/s, warmed greedy, same prompt).
- **K≥2 is a dead end** as-is: K=2 = 19.0 tok/s (1.03×), acceptance pos0 85% /
  pos1 66%.
- **Accuracy fine-tune ≈ +4% only** — K=1 caps at 2.0 tok/verify; acceptance is
  already 84.6% greedy / 83.8% sampled (not mismatched). Not worth a 9.7B-layer
  trainer *unless* paired with cheap depth. See repo HANDOFF for detail.
- **TP-free draft REFUTED as the lever:** `draft_tensor_parallel_size=1` (via the
  `DRAFT_TP` env knob added to `serve-glm52-tp2-mtp.sh`) boots, coherent, same
  84.6% accept, but **~17.6 tok/s (~5% slower)** — TP2 parallelizes the draft
  compute better than the all-reduce costs. ⇒ the 33 ms is host/launch overhead,
  not comm. **This points straight at the PIECEWISE-vs-FULL cudagraph gap.**
- **Verify uses FULL cudagraph, draft uses PIECEWISE** (confirmed in boot log:
  "Capturing CUDA graphs (decode, FULL)" for the model; proposer forced PIECEWISE
  in code).

## NEXT (immediate, in order) — REWRITTEN session 12 after Approach A refuted

The draft-cost problem as originally framed is **closed (negative).** The 30-tok/s
target now depends entirely on the **verify's per-position expert-read cost**.
New order:

1. **Quantify the per-position verify tax directly.** Measure decode tok/s at MTP
   K=1 vs K=0 (no spec, `--no-mtp`) AND non-MTP 1-token vs the verify's 2-token
   step. Confirm each extra verify position ≈ one extra full expert-read pass
   (bandwidth). This sets the ceiling for ANY speculative scheme (incl. n-gram).
2. **Attack the verify's per-position MoE bandwidth (the ONLY real lever):**
   - **Lower MoE top-k** on the verify (8→6→4) — directly cuts per-position expert
     bytes; measure quality hit (this is a real accuracy knob, needs GSM8K-50).
   - **Verify-side NVFP4 on the experts / a faster expert read** — the parallel
     track, now the MAIN track. Every byte off the per-position read lifts both
     base decode AND makes any spec depth cheaper.
3. **Only if (2) makes a spec position cheap:** revisit depth (K) and/or n-gram.
   Until then, more spec work is pointless — the depth tax is on the verify side.
4. **Accept 18.5 as the spec-decode ceiling** for now and redirect effort to the
   verify-bandwidth track + the pending REAP-vs-frequency quality A/B.

**DEAD — do not re-attempt (all cheapen the draft forward, which is ~free):**
Approach A (FULL-cudagraph draft — TESTED, 0 gain), C (fuse draft+verify), D
(cheaper draft compute / lower draft top-k / TP-free — TESTED slower). Approach E
(n-gram) is dead-by-inference (still pays the per-position verify tax) — verify #1
before spending on it.

## APPROACHES (ranked by expected leverage)

**A. FULL cudagraph the proposer/draft forward.** *The main event.* Remove the
PIECEWISE cap in `extract_hidden_states.py` and capture the draft as a single
graph like the verify. Blocker: the draft's MLA attention + MoE are in
`splitting_ops` (run eager under piecewise); FULL capture needs those ops
graph-safe for the draft path (static shapes, no host-side control flow). This is
the "hard vLLM problem." Payoff: potentially collapses most of the ~32 ms
overhead → c ≪ 0.15.

**B. Cut spec-decode host overhead.** Ensure async scheduling actually overlaps
the draft's host code with GPU work; batch/fuse the accept/reject + KV update;
remove per-step syncs. Complements A.

**C. Fuse draft+verify into one captured region.** Instead of two separate graph
invocations per step, capture the whole (draft→verify) as one — kills the
between-invocation host gap entirely. Deeper change; do after A/B profiling.

**D. Cheaper draft *compute* (makes TP1 viable + faster under any graph):**
lower the draft's MoE top-k (8→4), lighter draft attention, fewer draft experts.
Only ~0.9 ms of the 33 ms is compute today, so this alone is small — but it
matters if A makes the draft compute-bound, and it may let `DRAFT_TP=1` win.

**E. Zero-forward drafters (sidestep the whole problem):** n-gram / suffix
decoding (`vllm/v1/spec_decode/ngram_proposer.py`, `suffix_decoding.py`,
Arctic Inference). T_draft ≈ 0 (host lookup, no GPU forward) → c ≈ 0, depth is
free. Lower/among-domain-variable acceptance, but excellent on code/repetitive
output and **stackable** with MTP (MTP for the hard positions, n-gram for the
easy ones). Cheapest thing to try for a quick win; measure acceptance per domain.

**PARALLEL TRACK — cut T_verify (67 ms), orthogonal, also helps 30:** the verify
is bandwidth-bound at big-3 NVFP4. A better NVFP4 quant that reopens the full cut
*at coherent quality* (the full cut degraded — see repo HANDOFF) would cut the
verify read ~15–20%. Every ms off T_verify lifts the whole curve.

## HOW TO RESUME

**Serve the current 18.5 baseline (2× Spark, TP2):** Ray must be up first
(`bash spark/start-ray-cluster.sh`), then:
```bash
cd ~/Dev/vLLM-Moet; M=~/models/hf/GLM-5.2-FP8
MODEL=$M/nvfp4_big3_overlay VLLM_MOE_W2_PREPACKED_DIR=$M/moe_w2_planes_tp2_p208_reap \
VLLM_NVFP4_DENSE=1 VLLM_NVFP4_TARGETS="o_proj,q_b_proj,kv_b_proj" \
MTP_K=1 VLLM_ENGINE_READY_TIMEOUT_S=2400 \
nohup bash spark/serve-glm52-tp2-mtp.sh > ~/serve.log 2>&1 &
```
Knobs: `MTP_K=<N>` (spec tokens), `DRAFT_TP=1` (TP-free draft — currently slower).

**Warm before ANY measurement** (critical — cold planes read at 3–4 tok/s):
`bash spark/warm-planes.sh $M/moe_w2_planes_tp2_p208_reap` then ~5 warmup gens.

**Measure tok/s** (greedy, 200 tok, same prompt each time):
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Explain in detail how a transformer neural network processes a sentence, step by step."}],"temperature":0,"max_tokens":200}' \
 -w "\nTIME %{time_total}s"     # tok/s = 200/time; take runs 3-4 after warmup
```

**Measure acceptance:** hit `/metrics`, read
`spec_decode_num_{draft_tokens,accepted_tokens}_total` (and `..._per_pos_total`
for per-position). Helper: `scratchpad/measure_accept.py` (this session).

## GOTCHAS / KEY FACTS

- **Ray corrupts after many boots** ("ActorHandle ... across Ray sessions") →
  `bash spark/start-ray-cluster.sh` for a clean cluster. Killing servers: kill
  `vllm serve|EngineCore|RayWorkerProc` by PID on BOTH nodes; `ray status` → 0/2 GPU.
- **Warm or you'll misread everything** — first gens are 3–4 tok/s (plane
  faults); steady state ~18.5 needs warm-planes + a few gens.
- **Spec decode is LOSSLESS at greedy** — the target verifies, so any draft change
  (cheaper, pruned, TP-free) can't hurt output quality, only speed/acceptance.
  Coherence check is still worth running as a smoke test, but a cheaper draft
  cannot change greedy output.
- **MoE reads top-k per token regardless of pool size** — pruning the draft's
  expert *pool* (256→N) does NOT cut per-forward bandwidth; only lowering top-k does.
- **Draft "head" is a full 9.7B decoder layer** (layer 78: eh_proj/enorm/hnorm +
  MLA attn + 256-expert MoE + shared_head), not a small linear.
- Big-3 is the coherent NVFP4 cut; the full cut degrades — don't switch to it for
  a faster verify without fixing its quality first.

## CORRECTNESS GATES

- Any draft-cost change: (1) still coherent greedy (smoke), (2) **acceptance
  unchanged** (draft math must be identical unless you retrained it), (3) tok/s up.
- Reaching 30: warmed greedy ≥ 30 tok/s on the standard prompt, output identical
  to the K=1 baseline (lossless), across ≥3 stable runs.

## LINKS

- Repo `HANDOFF.md` (REAP, big-3, MTP measurements), `spark/GOAL.md` (20 tok/s
  origin), `spark/RUNBOOK.md` (serve/measure), `spark/NVFP4-DENSE.md`.
- vLLM spec-decode: `~/venvs/vllm-moet/.../vllm/v1/spec_decode/`
  (`extract_hidden_states.py` = cudagraph-mode decision, `eagle.py`,
  `llm_base_proposer.py`, `ngram_proposer.py`, `suffix_decoding.py`);
  MTP model `vllm/model_executor/models/deepseek_mtp.py`.
- `DRAFT_TP` knob: `spark/serve-glm52-tp2-mtp.sh`.
