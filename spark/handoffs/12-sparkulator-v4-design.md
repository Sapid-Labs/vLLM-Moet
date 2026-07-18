# s18+ (2026-07-18) — Sparkulator v4 designed (final-hidden-state conditioning); NOT started

> Previous: 11-s18-overnight-k3-win-siro1-ab-site-rows.md
> Design doc (the substance): **`../SPARKULATOR-DESIGN.md`**

## STATUS

**DESIGN ONLY.** After four same-architecture drafts converged at pos-0 ≈ 0.72–0.75
(epoch-3, v1/v2/v3, siro1) while native MTP hits 0.79–0.85 on the same target, the
diagnosis is an information bottleneck: dspark conditions on mid-layer aux states,
MTP on the final hidden state. Sparkulator v4 = epoch-3's proven DFlash parallel
backbone + the pre-norm final state (aux id 78) added to its conditioning fc
(6×6144, zero-init new columns, warm-start from epoch-3). Projected ~26–28 tok/s
at K=3 if pos-0 reaches MTP's level. Full architecture, gates, costs, and the
frozen ship bar (≥25.5 sparkbench mean): `../SPARKULATOR-DESIGN.md`.

Two workstreams, ordered:
0. **Confidence-gated dynamic K** (proposer-only, no training, ~1 session,
   +0.5–1.5 tok/s expected, pays standalone) — do this first.
1. **v4 training** (~3–5 sessions: trainer surgery in the pinned v05 clone →
   GATE A step-0 equivalence → epochs with LIVE sparkbench probes per epoch →
   ship gate).

## KEY FACTS FOR A COLD START

- Baseline to beat: **epoch-3 dspark K=3 = 24.28 tok/s** sparkbench mean
  (records `../dspark-distill/tools/sparkbench-epoch3-k3-s18.json`), serving on
  :8000 per RUNBOOK §4b.
- Measurement law (s18): fresh cluster + serve, warm ≥8, same-session A/B on
  `../dspark-distill/tools/sparkbench.py` + frozen prompts. Never trainer val.
- The extraction caches ALREADY contain the needed 6th channel (layer 78):
  hs files are `[seq, 6, 6144]`. **Do not reclaim** `~/dspark-hs-weak` (90 GB) or
  the magpie cache (205 GB, both on worker) while v4 is open — they are the
  training set. Worker disk ~12 GB free; reclaim `glm-5.2-dspark-spec-v1`
  (8 GB × both nodes) + old ckpt dirs first if space is needed; KEEP
  `~/dspark-hs-fp8test` (GATE-1).
- Inference plumbing for aux id 78 is already installed in both venvs (the s15
  distill patch to `deepseek_v2.py`, currently dormant — goes live when the
  speculator config lists 78).
- Pinned trainer: `spark-c84b:~/Dev/speculators-v05` @ 21033a7 (worker venv).
  Deploy candidates via the graft recipe (clone reference dir, swap
  model.safetensors) + `SPECULATOR=` env on `serve-glm52-tp2-dspark.sh`.

## NEXT (when Joe green-lights)

1. Workstream 0: implement confidence gating in `v1/spec_decode/dspark.py`
   `_sample_draft_tokens` (check vLLM variable-length proposal support first),
   sweep τ on sparkbench at K=3.
2. v4: trainer surgery per design doc §5, then GATE A before any training.

## MACHINE STATE (unchanged from handoff 11)
Resting serve epoch-3 K=3 on :8000; worker disk ~12 GB free; deep C-states
disabled both nodes (reverts on reboot).
