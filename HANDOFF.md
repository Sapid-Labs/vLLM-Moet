# HANDOFF — GLM-5.2 on 2× DGX Spark (branch spark-gb10)

**One handoff = one file** in `spark/handoffs/`, numbered sequentially — read them
in order for the full journey. This root file is only a pointer: current status,
latest handoff, index. Update it (pointer + index line) in the same commit as each
new numbered handoff.

## CURRENT STATUS → `spark/handoffs/10-s18-regression-resolved-sparkulator-v3-null.md`

One line: s17 "regression" was environmental (protocol banked: fresh cluster, warm
≥8, same-session A/B via `spark/dspark-distill/tools/sparkbench.py`); Sparkulator
v3 fine-tune = NULL (third round — weak band is intrinsic entropy, epoch-3 at
ceiling); ship config (dspark K=2 + REAP + NVFP4 big-3 + top-k4, ~23 tok/s)
serving on :8000; overnight queue = harness site rows + GSM8K-50 + K=3 re-sweep.

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

Sub-project logs: `spark/dspark-distill/` (HANDOFF.md, PLAN.md, FINDINGS.md, tools/).
Deep reference docs: `spark/NVFP4-DENSE.md`, `spark/GOAL.md`, `spark/RUNBOOK.md`,
`spark/MTP-DRAFT-COST.md`.
