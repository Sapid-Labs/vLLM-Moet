# SPARKULATOR-DESIGN — a final-hidden-state-conditioned parallel draft for GLM-5.2

Design doc for "option 1" (2026-07-18): the one evidence-backed path to a draft
that beats RedHat epoch-3 on this stack. Companion handoff:
`handoffs/12-sparkulator-v4-design.md`. Status: **DESIGN ONLY — not started.**

## 1. The evidence this design rests on (do not re-derive)

| fact | source |
|---|---|
| Four same-arch drafts converge at pos-0 ≈ 0.72–0.75 on sparkbench: epoch-3, our v1/v2/v3 fine-tunes, siro1's from-scratch 4B (10 ep, lr 6e-4, quant-matched) | s16f, s18 (`dspark-distill/tools/sparkbench-*-s18.json`) |
| Native MTP on the same target: pos-0 **0.79–0.85** | s11/s12 measurements |
| The MTP-vs-dspark architectural difference is conditioning: MTP sees the target's **final** hidden state; dspark sees mid-layer aux states [8,23,39,55,70] of context tokens | model definitions |
| Draft forward cost is NOT the tax (~10 ms); the verify per-position expert-read (~32 ms) is | `MTP-DRAFT-COST.md`, s12 |
| Weak-band prose (~0.67 pos-0) is intrinsic entropy — no draft recovers it (MTP is also weak there) | s16g probe, s18 v3 verdict |
| Current best: epoch-3 dspark K=3 = 24.28 tok/s sparkbench mean (tok/verify 2.618) | handoff 11 |

**Thesis:** the 0.74 ceiling is an information bottleneck, not capacity/data. Give a
dspark-style parallel draft the final-layer state and pos-0 should approach MTP's
0.79–0.85 at ~1/10th of MTP's draft cost, while keeping dspark's cheap K=2–3
parallel drafting. Back-of-envelope at K=3: pos-0 0.74→0.82 with proportional
deeper-position lift ⇒ tok/verify ~2.6→~3.0 ⇒ **~26–28 tok/s** (ceiling ~28–30).

## 2. Architecture ("Sparkulator v4")

Keep everything that is proven; change only the conditioning.

- **Backbone:** epoch-3's 5-layer qwen3 DFlash backbone, warm-started from epoch-3
  weights. Same block machinery: context-KV precompute, 1+N fused query block,
  non-causal attention, mask token, bonus anchor.
- **Conditioning change:** the combine-fc that maps aux hidden states → draft input
  currently takes **5×6144** (layers 8,23,39,55,70). Extend to **6×6144**, adding
  the **pre-norm final hidden state (aux id 78 = end_layer)** per context/anchor
  token. Zero-init the new fc columns → step-0 behavior is exactly epoch-3
  (a built-in GATE: step-0 LR-0 must reproduce epoch-3's 0.773 on the weak valset).
- **Why this helps despite parallel drafting:** position 0 conditions on the
  *anchor* token, whose final state exists (the verify forward that accepted it
  computed it). Positions ≥1 have no final state of their own (unverified) but
  attend to a context whose every token now carries final-state signal. Expected
  gain profile: pos-0 largest, deeper positions partial. If gains turn out
  pos-0-only, dynamic-K (workstream 0) still monetizes them.
- **Heads:** keep Markov head (rank 256) and confidence head unchanged;
  additionally TRAIN the confidence head properly (epoch-3's may be stale) so
  workstream 0 has a good signal.

## 3. Workstream 0 (do first, no training): confidence-gated dynamic K

Proposer-side change in `v1/spec_decode/dspark.py` (`_sample_draft_tokens`): after
sampling position i, read the confidence head; if conf < τ, emit fewer than K
drafts this step (vLLM supports variable per-step proposal lengths — verify this;
if not, pad with a sentinel the verifier rejects for free... check cost first).
Sweep τ on sparkbench. Expected: +0.5–1.5 tok/s on prose-heavy content at K=3 by
skipping ~32 ms verify positions where acceptance would be ~0.3. Zero risk to
quality (greedy spec decode stays lossless). This stacks with v4 and pays even if
v4 is never built.

## 4. Inference plumbing (mostly already exists)

- The fork requests aux layers from the speculator config's
  `aux_hidden_state_layer_ids` → v4 config lists `[8,23,39,55,70,78]`.
- **id==78 (end_layer) support is ALREADY INSTALLED in both venvs** — the s15
  distill patch to `deepseek_v2.py` appends the pre-norm final state when
  `end_layer ∈ aux_hidden_state_layers` (currently dormant). It becomes live the
  moment the config asks for 78. Also tracked in `dspark-distill/patches/`.
- `llm_base_proposer.py` combine-hidden-states path already handles dspark; the fc
  input width comes from the checkpoint, so 6×6144 should load without code
  changes (VERIFY: any hardcoded `len(aux_layers)` assumptions).
- Cost delta at inference: one extra 6144-vector per context token into the fc —
  negligible vs the 32 ms verify tax.

## 5. Training plan (reuses the v3 pipeline end-to-end)

1. **Data — already on disk (do NOT reclaim these while this is open):**
   magpie cache (1988 hs, 205 GB) + weak-band cache (3199 hs, 90 GB) +
   `prepared-combined-v3` (5200 rows). Every hs file is `[seq, 6, 6144]` —
   **the 6th channel IS layer 78**; extraction needs no re-run. If more data is
   needed (likely for a conditioning change): the gen→extract pipeline produces
   ~3200 samples (~1.2M tok) per ~7 h serve day; target +2 rounds of
   reasoning/chat mix (~10M tok total) before concluding anything.
2. **Trainer mods (pinned v05 clone, `~/Dev/speculators-v05`):** extend the dspark
   draft config/forward to consume 6 aux channels (fc 5→6×6144, zero-init new
   columns, warm-start rest from epoch-3); keep using channel-6 ALSO as the
   tv/ce label source (it already does). Confidence-head loss term on
   position-level accept/reject labels (derivable offline: draft-greedy ==
   target-greedy per position).
3. **Gates (protocol from s18 — live probes decide, never trainer val):**
   - GATE A: step-0 LR-0 with zero-init columns == epoch-3 baseline (0.773 weak
     valset) — proves the surgery is clean.
   - GATE B: after epoch 1, live sparkbench probe (serve §4b,
     `SPECULATOR=<graft>`); need pos-0 ≥ +2 pt over 0.748 to continue.
   - GATE C (ship): sparkbench mean tok/s > 24.28 (K sweep 2/3/4 + dynamic-K τ),
     GSM8K-50 ≥ 90%, back-to-back same-session A/B vs epoch-3 K=3.
4. **Known failure mode to watch:** trainer→proposer transfer gap (v2's lesson).
   Mitigation: GATE B is a LIVE probe after every epoch (~40 min each); if
   trainer-space moves but live doesn't by epoch 2, stop and diagnose the
   fc/plumbing mismatch before more training.

## 6. Cost & risk

- Workstream 0: ~1 session (code + τ sweep). Pays standalone.
- v4: ~1 session trainer surgery + gates, 2–6 h per training epoch, ~1 session
  eval/ship. Realistic total 3–5 sessions.
- Disk: keep both hs caches (295 GB on worker, disk already ~12 GB free) —
  reclaim `glm-5.2-dspark-spec-v1` (8 GB×2) and old ckpts instead if space is
  needed. A third extraction round needs ~90 GB more: reclaim
  `~/dspark-hs-fp8test`? NO — GATE-1 needs it. Budget carefully.
- Honest downside case: conditioning gain is pos-0-only and small (+1–2 pt) →
  v4 ties epoch-3 and only dynamic-K ships (+~1 tok/s). The three prior nulls
  were same-arch; this is a different bet, but it is still a bet.

## 7. Success criteria (frozen now, before any code)

Ship "Sparkulator" to HF sapidlabs iff: sparkbench mean tok/s **≥ 25.5** (i.e.
> epoch-3 K=3's 24.28 by more than boot noise), no category below its epoch-3
number by >2 tok/s, GSM8K-50 ≥ 90%, same-session back-to-back, clean-restart
protocol. Otherwise: report honestly, keep epoch-3, close the speculator track.
