# s15-s16 (2026-07-16/17) — dspark self-distillation: pipeline built, v1+v2 NULL

> Previous: 08-s14-*.md · Next: 04-decode-throughput-regression.md (s17) · Full sub-project log: ../dspark-distill/HANDOFF.md

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

