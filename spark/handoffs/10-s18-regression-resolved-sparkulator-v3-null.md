# s18 (2026-07-17) — regression resolved (environmental); Sparkulator v3 ran end-to-end: NULL; epoch-3 at ceiling

> Previous: 04-decode-throughput-regression.md (s17) · Sub-project detail:
> ../dspark-distill/HANDOFF.md (s18 sections) · Overnight results land in 11-*.md

## STATUS (end of s18, ~23:00)

- **s17 decode regression RESOLVED — not reproducible.** Clean Ray restart + fresh
  §4b serve → dspark K=2 = 21.1–26.1 tok/s warm (≥ documented 21.8), acceptance
  0.74/0.50, coherent. Code/config verified byte-clean first (pre-distill
  `deepseek_v2.py` reconstructed from the pristine vllm-0.24.0 wheel + tracked
  patches; the distill patch is inert at serve time — speculator aux layers
  [8,23,39,55,70] never hit the added branch). The 10.4 was environmental
  (measured amid full-disk/Ray-thrash, never re-measured after a clean restart).
  Full diagnostic: `04-decode-throughput-regression.md` (RESOLUTION section).
- **MEASUREMENT PROTOCOL (banked, non-negotiable):** fresh cluster + fresh serve,
  warm ≥8 runs, same-session back-to-back A/B on the same prompt set. Canonical
  tool: `../dspark-distill/tools/sparkbench.py` + frozen
  `sparkbench-prompts.jsonl` (52 prompts). epoch-3 reproduces 0.743/0.748 pos-0
  across boots; per-category slices (n=4) swing ±5–8 pt — only the 52-prompt
  mean is meaningful.
- **Sparkulator v3 (weak-band on-policy fine-tune) — NULL, third and decisive
  round.** Live A/B: epoch-3 23.15 tok/s / pos-0 0.748 vs v3 22.70 / 0.732.
  Trainer-space LR-0 on an identical 520-row weak valset: 0.773 vs 0.778
  (+0.5 pt) → the weak band (creative/chat/summarize prose) has NO learnable
  signal — intrinsic content entropy, not a transfer gap. Three rounds, three
  hypotheses, three nulls ⇒ **epoch-3 is at the practical ceiling of this draft
  architecture on this target.** Phase-5 fork-native training NOT triggered (its
  pre-condition, val-up-live-flat, did not occur) — needs a fresh decision.
  Records: `../dspark-distill/tools/sparkbench-*-s18.json`.
- **Machine state:** epoch-3 dspark K=2 §4b serving on :8000
  (`~/serve-dspark-best.log`). Deep C-states still disabled both nodes (harmless,
  reverts on reboot, sudo to revert sooner). **Worker disk ~12 GB free** —
  reclaim candidates if v3 is closed: `~/dspark-hs-weak` (90 GB),
  `~/models/hf/GLM-5.2-speculator.v3` (7.6 GB × both nodes), ckpt-v3.

## NEXT (overnight queue, running)

1. `bench/harness.py` on the ship config → real howtospark site rows
   (config `glm-5-2--vllm-moet--fast-dspark-k2--2spark-tp2.json`).
2. GSM8K-50 sanity on the ship config (last unchecked gate).
3. dspark K=3 / K=1 sparkbench re-sweep (today's higher measured acceptance may
   have flipped the s14 K=3 verdict), then restore K=2 resting serve.

## LINKS

- Sparkulator v3 full plan: `~/.claude/plans/dazzling-seeking-steele.md`
- Lab notebook: `~/Dev/howtospark/models/glm-5.2.md` (s18 + s18b entries)
