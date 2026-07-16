# dspark K sweep — 2026-07-16 (session 14)

Config (all rows identical except DSPARK_K):
fast build = nvfp4_big3_overlay + moe_w2_planes_tp2_p208 (frequency prune) +
VLLM_NVFP4_DENSE=1 (o_proj,q_b_proj,kv_b_proj) + top-k4 hf-override +
PLANES_MMAP=1 + FADVISE_GLOB= (empty) + FULL_AND_PIECEWISE graphs, TP2, greedy 300 tok.

Measure: scratchpad/measure.py (streamed chat completion, temp 0, 300 tok,
decode tok/s = completion_tokens / (t_end - t_first_token)).

## KEY FINDING (revises session-13 diagnosis)

The 2.3 tok/s "regression" was COLD PAGE CACHE, not a persistent per-step cost.
Serve script purges page cache at startup; PLANES_MMAP=1 planes fault in
lazily. Warmup sequence measured on the K=3 server left up from session 13:
5.30 → 7.80 → 10.81 → 16.35 → 20.08 → 19.85 tok/s (6×300-tok greedy runs).
(Matches the session-8 memory verdict: never trust spec-decode numbers under
page-cache thrash.)

## K=3 (server from session 13, fadvise off, steady after ~5 warm runs)

- steady decode: **~20.0 tok/s** (20.08, 19.85)
- /metrics: 798 drafts, 2394 draft toks, 1063 accepted → mean accepted 1.33
  → **2.33 tok/verify**; per-pos acceptance 68% / 42% / 23%
- step rate ≈ 20.0/2.33 ≈ 8.6 verify/s (≈116 ms/step)

Baseline to beat: native MTP K=1 = 21.8 tok/s (session 12, p208_reap planes).

## K=2 (fresh serve, warmed 11 runs: 1.6→5.8→10.2→14.4→17.2→18.9→18.6→20.0→20.6→22.0→20.2)

- steady decode: **~21 tok/s** (20.6, 21.97, 20.23; mean 20.9)
- /metrics: 1469 drafts, 2938 draft toks, 1670 accepted → mean accepted 1.14
  → **2.14 tok/verify**; per-pos acceptance 68% / 45%
- step rate ≈ 20.9/2.14 ≈ 9.8 verify/s (≈102 ms/step)

## K=1 — SKIPPED (dominated, not measured)

dspark K=1 accepts 1.68 tok/verify (pos0 68%) vs native MTP K=1's 1.85 (84.6%)
on the same number of verify positions, and dspark's per-step draft cost
(ctx-KV precompute + draft fwd + Markov) ≥ MTP head cost. It cannot beat
native MTP K=1; linear model from K=2/K=3 (step ≈ 74 + 14·K ms) predicts
~19 tok/s. dspark optimum = K=2.

## Native MTP K=1, SAME fast build (p208 freq planes, mmap, top-k4)

(first boot attempt died: CUBLAS_STATUS_INTERNAL_ERROR during PIECEWISE graph
capture + Ray ActorHandle-across-sessions; clean `start-ray-cluster.sh` restart
fixed it — known remedy)

- warmed 9 runs: 1.7→5.2→8.5→16.2→17.3→20.9→20.2→19.2→21.1
- steady decode: **~20.4 tok/s** (20.9, 20.2, 19.2, 21.1)
- /metrics: 1431 drafts, 1105 accepted → 77.2% acceptance → **1.77 tok/verify**
- step rate ≈ 11.5 verify/s (≈87 ms/step)
- NOTE: session-12's 21.8 (on p208_reap planes, 84.6% accept) does NOT
  reproduce on the freq-p208 planes — acceptance is 7pts lower here.

## VERDICT (so far): dspark K=2 (~20.9) ≈ native MTP K=1 (~20.4) — TIE within
run-to-run noise (±1 tok/s), on the identical fast build.

## No-draft floor, same build (freq planes)

- warmed 10 runs: 2.5→6.8→10.6→11.9→12.0→14.7→15.2→15.3→15.4→16.1
- steady decode: **~15.6 tok/s** (15.3, 15.4, 16.1)

## REAP-planes round (acceptance recovery attempt, same fast build otherwise)

### dspark K=2 on p208_reap
- warmed 13 runs; steady (last 5): 21.6, 22.0, 20.1, 23.4, 22.1 → **~21.8 tok/s**
- /metrics: 1296 drafts, 1478 accepted → mean 1.14 accepted → **2.14 tok/verify**
  per-pos **69% / 45%** — IDENTICAL to freq planes.
- **Acceptance did NOT recover.** dspark's dense draft (trained on FP8-full)
  is insensitive to which experts survive the prune; the MTP head recovered on
  REAP because it conditions on the target's hidden states. The ~+1 tok/s vs
  freq K=2 (20.9) is plane I/O / noise, not drafter behavior.
- dspark K=3 on REAP: SKIPPED — per-pos rates unchanged ⇒ K=3 REAP ≈ K=3 freq
  (~20.0), still dominated by K=2.

### native MTP K=1 on p208_reap

- warmed 10 runs: 2.6→7.8→14.0→17.7→16.5→19.5→19.8→21.0→21.2→22.1
- steady decode: **~21 tok/s** (last 3 ≈ 21.4, still trending up slightly)
- /metrics: 1670 drafts, 1323 accepted → **79.2% acceptance**, 1.79 tok/verify
- Session-12's 84.6% does NOT reproduce today either (79.2%); REAP buys MTP
  only ~2pts over freq (77.2→79.2). Acceptance is content-dependent — within-
  experiment comparisons (same prompt set) are the fair ones.

### Standings on REAP planes: dspark K=2 ~21.8 > MTP K=1 ~21 (modest dspark lead,
### ~1 tok/s, at the edge of noise). dspark acceptance unmoved by planes ⇒ the
### 68%-vs-83-100% gap lives in the OTHER target mods; top-k4 is prime suspect.

### dspark K=2, p208_reap, native top-k8 (ablation: what does top-k4 cost in acceptance?)

- warmed 10 runs; steady ~**18 tok/s** (18.4, 18.6, 18.0, 16.6) — k8 doubles
  expert bytes, as expected (k4 = 21.8 wins on throughput).
- /metrics: 1331 drafts, 1547 accepted → per-pos **70.0% / 46.2%** — vs 69/45
  at top-k4. **Top-k4 costs dspark ~1pt of acceptance — NEGLIGIBLE.**
- Ruled out so far for the 83→69% gap: plane choice (freq vs REAP), top-k.
  Remaining suspects: NVFP4 big-3 attention; the prune / 2-bit experts.

### dspark K=2, p208_reap, NO NVFP4 (plain FP8 attn), native top-k8

- /metrics: 985 drafts, 1112 accepted → per-pos **67.8% / 45.1%** — unchanged.
- **NVFP4 big-3 also costs ~0 acceptance.** Throughput ~16-17 (still warming;
  not the point — acceptance was the question).
- Ruled out now: plane choice, top-k, NVFP4. Remaining: the prune itself
  (208 vs 256) OR the session-13 "83-100%" reference was simply measured on
  easy short prompts (eager, "Paris"/"2+2") — acceptance is content-dependent.

### dspark K=2, FULL unpruned 256-expert planes, plain FP8, native k8

- /metrics: 497 drafts, 547 accepted → per-pos **66.8% / 43.3%** — unchanged.
- Throughput ~2.3 tok/s and NOT warming — 97 GB/rank unpruned planes don't fit
  page cache (the very reason the prune exists). Irrelevant to the acceptance
  question; acceptance is speed-independent.

## INVESTIGATION CLOSED — acceptance was never lost

Per-pos-0 acceptance across ALL five target configs: 68/69/70/68/67% (fast
build, REAP swap, top-k8, no-NVFP4, full-unpruned). Every target modification
individually costs ≤1pt. Session-13's "83-100%" was measured eager on short
trivia prompts; acceptance is content-dependent, and **~68% pos-0 is this
speculator's TRUE acceptance on 300-tok reasoning content.** Nothing to
recover target-side. Remaining acceptance lever: fine-tune/self-distill the
draft on the served target + representative content (est. +2-4 tok/s at K=2-3
if pos rates reach ~80/60%).

## FINAL BEST CONFIG (left serving): dspark K=2 + p208_reap + NVFP4 big-3 +
## top-k4 ≈ **21.8 tok/s** (vs MTP K=1 same stack ~21, no-draft ~15.6).
