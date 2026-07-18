# HANDOFF — GLM-5.2 on 2× DGX Spark (branch spark-gb10)

**One handoff = one file** in `spark/handoffs/`, numbered sequentially — read them
in order for the full journey. This root file is only a pointer: current status,
latest handoff, index. Update it (pointer + index line) in the same commit as each
new numbered handoff.

## CURRENT STATUS → `spark/handoffs/14-s19-v4-epoch1-gateb-fail.md`

One line: best = dspark K=3 ≈ 24.3 tok/s serving on :8000; s19 ran BOTH
Sparkulator v4 workstreams to verdicts in one day: **workstream 0 (dynamic K)
NULL** (handoff 13 — trimming loses on this bandwidth-bound stack) and **v4
epoch-1 GATE B FAIL** (handoff 14 — surgery+GATE A clean, trained, live pos-0
0.733 vs bar 0.768; flat everywhere). Awaiting Joe: close speculator track per
design §7, or overrule the gate for epoch 2. Keep the worker hs caches.

## JOURNEY INDEX (chronological)

| file | what happened |
|---|---|
| `spark/handoffs/01-tp2-cudagraph-nccl-replay-hang.md` | TP2 FULL-cudagraph NCCL deadlock → NCCL 2.30.7 pin |
| `spark/handoffs/02-glm52-decode-10toks-topk-recovery.md` | the road to 15 tok/s (planes, pruning, MTP) |
| `spark/handoffs/05-s11-reap-shipped-nvfp4-verdicts.md` | s11: REAP saliency prune shipped; NVFP4 big-3 safe / full cut degrades; MTP near-optimal |
| `spark/handoffs/06-s12-topk4-win-draft-cost-dead.md` | s12: draft-cost thesis refuted; top-k 8→4 → 21.8; Marlin-for-decode verdict; corrections |
| `spark/handoffs/07-s13-dspark-port.md` | s13: dspark external speculator ported onto the fork |
| `spark/handoffs/03-dspark-acceptance-ablation.md` | s14b deep-dive: acceptance ablation (drafter never degraded) |
| `spark/handoffs/08-s14-dspark-3way-ablation.md` | s14: 3-way settled — dspark K=2 ≈ MTP K=1; best 21.8; cold-cache artifact identified |
| `spark/handoffs/09-s15s16-dspark-distill.md` | s15-16: self-distill pipeline built; v1+v2 NULL (detail: `spark/dspark-distill/HANDOFF.md`) |
| `spark/handoffs/04-decode-throughput-regression.md` | s17: 10.4 tok/s regression investigation → s18 RESOLUTION (environmental) |
| `spark/handoffs/10-s18-regression-resolved-sparkulator-v3-null.md` | s18: regression resolved; sparkbench protocol; Sparkulator v3 NULL; epoch-3 at ceiling |
| `spark/handoffs/11-s18-overnight-k3-win-siro1-ab-site-rows.md` | s18 overnight: K=3 new best 24.3; siro1 A/B loses; site rows + GSM8K-50 |
| `spark/handoffs/12-sparkulator-v4-design.md` | Sparkulator v4 design (final-state conditioning + dynamic K) — not started |
| `spark/handoffs/13-s19-dynamic-k-null.md` | s19: workstream 0 (dynamic K) built + measured NULL; per-position verify cost is small, trimming loses; v4 = acceptance lift only |
| `spark/handoffs/14-s19-v4-epoch1-gateb-fail.md` | s19: v4 surgery+GATE A clean, epoch-1 trained, GATE B live probe FAIL (0.733 vs 0.768 bar) — stop condition triggered, Joe to decide |

Sub-project logs: `spark/dspark-distill/` (HANDOFF.md, PLAN.md, FINDINGS.md, tools/).
Deep reference docs: `spark/NVFP4-DENSE.md`, `spark/GOAL.md`, `spark/RUNBOOK.md`,
`spark/MTP-DRAFT-COST.md`.
