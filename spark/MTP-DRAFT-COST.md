# MTP-DRAFT-COST — the "hard vLLM problem": cheap draft forward → 30 tok/s

Deep design/handoff doc. The short pointer is in the repo-root `HANDOFF.md`;
this is the "resume here" for the single problem blocking **30 tok/s** on
GLM-5.2 / 2× DGX Spark: **the MTP draft forward is too expensive relative to the
verify.** Written 2026-07-14 (session 11).

## STATUS

Phase: **new problem, scoped + measured, not started.** Branch `spark-gb10`.
Serving today = big-3 NVFP4 + REAP planes + MTP K=1 = **18.5 tok/s** (warmed,
greedy). Goal = **30 tok/s** (usability bar). Need ~1.62×, and it will not come
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

## WHY T_draft is 33 ms (root cause — the hard part)

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

## NEXT (immediate, in order)

1. **PROFILE the draft forward first — do not guess the 33 ms split.** Attach a
   torch profiler / nsys to one warmed decode step and separate: host gaps
   (piecewise relaunch) vs eager attention vs MoE vs all-reduce vs
   sample/accept host code. Whatever dominates decides which of the approaches
   below to pursue. (Hook: `VLLM_TORCH_PROFILER_DIR=...` then hit the server; or
   the `spark/profile_decode.sh` scaffold.)
2. **Attack the biggest slice.** Expected #1 = piecewise host overhead → pursue
   **Approach A (FULL cudagraph for the proposer).**
3. Re-measure T_draft with the same K=1/K=2 tok/s protocol; target T_draft <10 ms.
4. Once T_draft is cheap: raise K (2→4), then (only then) multi-step-distill the
   drafter for deeper positions. Re-measure to 30.

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
