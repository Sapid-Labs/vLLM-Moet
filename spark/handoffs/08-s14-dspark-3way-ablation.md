# s14 (2026-07-16) — dspark 3-way settled (ties MTP); acceptance ablation; best 21.8

> Previous: 07-s13-*.md · Next: 09-s15s16-*.md · Deep doc: 03-dspark-acceptance-ablation.md

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

