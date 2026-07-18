# s18 overnight (2026-07-17→18) — K=3 flips to WIN (24.28 tok/s, new best); siro1 external draft loses to epoch-3; site rows + GSM8K-50 shipped

> Previous: 10-s18-regression-resolved-sparkulator-v3-null.md

## STATUS (2026-07-18 ~01:30) — overnight queue COMPLETE, all four items

1. **★ NEW BEST CONFIG: dspark K=3 = 24.28 tok/s sparkbench mean** (epoch-3 draft,
   same §4b stack) vs 23.15 at K=2 (+5%). tok/verify 2.618 vs 2.28; even the
   weakest category (creative 21.98) beats its K=2 number (20.75). The s14
   "K=3 loses" verdict flipped — measured acceptance on real content (0.74/0.52)
   is far above s14's ablation set (0.68/0.45), so the third position now pays for
   its ~32 ms verify tax. RUNBOOK §4b updated (DSPARK_K=3). Resting serve on :8000
   is now K=3. Caveat: K=3 vs K=2 is cross-boot, but K=2 reproduced twice
   (23.13/23.15) so +1.1 is outside observed boot noise.
2. **siro1/glm-5.2-dspark-spec-v1 (external from-scratch draft, trained vs NVFP4
   verifier, speculators 0.6.0.dev0, 10 ep lr 6e-4) — LOSES on our stack:**
   23.42 mean / pos-0 0.716 at K=3 vs epoch-3's 24.28 / 0.739. Slightly better on
   code_explain (0.717 vs 0.665), worse on creative/reasoning/math. Loads cleanly
   in the port (`Qwen3DSparkModel`, block 16). **Fourth challenger to lose to
   epoch-3** — reinforces the v3 ceiling verdict. Staged at
   `~/models/hf/glm-5.2-dspark-spec-v1` (both nodes; reclaimable).
   Idea harvested from their card: **confidence-gated dynamic K** — dspark ships a
   confidence head our `DSparkProposer` ignores; stopping the draft early on
   low-confidence positions would skip the ~32 ms verify tax exactly where
   acceptance collapses (high-entropy prose). The one untried GB10-shaped
   throughput lever. Proposer-side code change, no training.
3. **howtospark site rows published** (commit `da48e5c`): first real harness rows
   for the fast build — 17.7 tok/s (p512/c1), 21.0 (p2048/c1), 9.2/stream (c4),
   prefill 190-329 tok/s. (Harness synthetic prompts → lower acceptance than
   sparkbench content; both are honest, different distributions.) Config:
   `bench/configs/glm-5-2--vllm-moet--fast-dspark-k2--2spark-tp2.json` — re-run
   at K=3 for updated rows when convenient.
4. **GSM8K-50 on the ship config: 92% strict** (≥90% gate PASS; n=50 noise band
   includes the earlier 96%). Results:
   `~/Dev/howtospark/evals/results/glm-5.2--2026-07-17/`.

## MACHINE STATE
- Resting serve: epoch-3 dspark **K=3** §4b on :8000 (`~/serve-dspark-best.log`).
- Worker disk ~12 GB free. Reclaim candidates: `~/dspark-hs-weak` (90 GB),
  `~/models/hf/glm-5.2-dspark-spec-v1` (8 GB ×2 nodes), ckpt-v3 (7.6 GB).
- Deep C-states still disabled both nodes (harmless; reverts on reboot).

## NEXT (ideas, in value order)
1. Confidence-gated dynamic K in `DSparkProposer` (see item 2 above).
2. Re-run harness at K=3 → refresh site rows (+~1 tok/s expected).
3. Long-context sweep (8/32/128K TTFT+decode) — answers the "how much context"
   question with measurements; one evening.
4. top-k 5/6 quality/speed sweep (needs eval per point — daytime project).

## LINKS
- sparkbench records: `../dspark-distill/tools/sparkbench-{epoch3-k3,siro1-k3}-s18.json`
- siro1 card: https://huggingface.co/siro1/glm-5.2-dspark-spec-v1
- Lab notebook: `~/Dev/howtospark/models/glm-5.2.md`
